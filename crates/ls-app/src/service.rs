//! Composition root: orchestrates the engine (extract → chunk → embed → store,
//! and query → rerank → synthesize) over the persisted collections/settings.
//!
//! Engine handles (`Embedder`, `Reranker`, `OllamaClient`) are passed in by the
//! caller rather than held here, so the Tauri layer can own their lifetime/threading
//! and this layer stays free of UI/runtime concerns.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use ls_core::TokenCounter;
use ls_embed::{Embedder, Reranker};
use ls_index::{chunk_book, ChunkParams, Store};
use ls_llm::{build_prompt, OllamaClient};
use ls_query::{search, SearchResult};

use crate::{Collection, Db, Settings};

const EMBED_BATCH: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Db(#[from] crate::DbError),
    #[error(transparent)]
    Settings(#[from] crate::SettingsError),
    #[error(transparent)]
    Store(#[from] ls_index::StoreError),
    #[error(transparent)]
    Embed(#[from] ls_embed::EmbedError),
    #[error(transparent)]
    Query(#[from] ls_query::QueryError),
    #[error(transparent)]
    Llm(#[from] ls_llm::LlmError),
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct IndexStats {
    pub books_indexed: usize,
    pub books_unchanged: usize,
    pub books_skipped: usize,
    pub books_failed: usize,
    pub chunks_written: usize,
}

/// Progress events emitted during indexing (forwarded to the UI as Tauri events).
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IndexEvent {
    /// Engine (ONNX models) loading before the run starts.
    Loading,
    Started {
        total: usize,
    },
    /// About to read/extract file `n` (shown so a slow/large file is visible
    /// rather than looking frozen).
    Working {
        n: usize,
        total: usize,
        path: String,
    },
    /// Per-batch embedding progress within the current book.
    Embedding {
        n: usize,
        total: usize,
        title: String,
        chunks_done: usize,
        chunks_total: usize,
    },
    Indexed {
        n: usize,
        total: usize,
        title: String,
        chunks: usize,
    },
    Unchanged {
        n: usize,
        total: usize,
        title: String,
    },
    Skipped {
        n: usize,
        total: usize,
        path: String,
        reason: String,
    },
    Finished {
        stats: IndexStats,
    },
}

/// Display title from a path (used for skipped/unchanged files we don't extract).
fn title_of(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled".to_string())
}

/// `(path, size, mtime)` fingerprint — changes iff the file changes.
fn file_fingerprint(path: &Path) -> String {
    match std::fs::metadata(path) {
        Ok(m) => {
            let mtime = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            format!("{}:{}", m.len(), mtime)
        }
        Err(_) => "missing".to_string(),
    }
}

/// Stateful application service over the persisted DB + settings.
pub struct Service {
    pub db: Db,
    pub settings: Settings,
    pub data_dir: PathBuf,
}

impl Service {
    /// Open (creating dirs) the app DB and settings under `data_dir`.
    pub fn new(data_dir: impl AsRef<Path>) -> Result<Self, ServiceError> {
        let data_dir = data_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;
        let db = Db::open(data_dir.join("app.db"))?;
        let settings = Settings::load(data_dir.join("settings.toml"))?;
        Ok(Self {
            db,
            settings,
            data_dir,
        })
    }

    pub fn settings_path(&self) -> PathBuf {
        self.data_dir.join("settings.toml")
    }

    /// Index `paths` into a collection, skipping files whose fingerprint is unchanged.
    /// Reuses the engine; emits progress via `on_event`. Builds the FTS index at the end.
    pub async fn index_collection(
        &self,
        collection: &Collection,
        paths: &[String],
        embedder: &mut Embedder,
        counter: &dyn TokenCounter,
        mut on_event: impl FnMut(IndexEvent),
    ) -> Result<IndexStats, ServiceError> {
        let store = Store::open_or_create(&collection.db_path, "chunks").await?;
        let params = ChunkParams::default();
        let mut stats = IndexStats::default();
        let total = paths.len();
        on_event(IndexEvent::Started { total });

        let mut wrote = false;
        for (i, path) in paths.iter().enumerate() {
            let n = i + 1;
            let p = Path::new(path);

            // Fast skip BEFORE the costly extract: the book id is derived from the
            // path and the fingerprint from file metadata, so an unchanged file is
            // recognized with just a stat. This makes re-index / resume cheap.
            let book_id = ls_extract::stable_book_id(p);
            let fp = file_fingerprint(p);
            if self
                .db
                .book_fingerprint(&collection.id, &book_id)?
                .as_deref()
                == Some(fp.as_str())
            {
                stats.books_unchanged += 1;
                on_event(IndexEvent::Unchanged {
                    n,
                    total,
                    title: title_of(p),
                });
                continue;
            }

            // Announce the file before the (potentially slow) extract so a large
            // or problematic PDF is visible instead of the bar looking frozen.
            on_event(IndexEvent::Working {
                n,
                total,
                path: path.clone(),
            });
            let doc = match ls_extract::extract(p) {
                Ok(d) => d,
                Err(e) => {
                    stats.books_failed += 1;
                    on_event(IndexEvent::Skipped {
                        n,
                        total,
                        path: path.clone(),
                        reason: e.to_string(),
                    });
                    continue;
                }
            };

            if doc.blocks.is_empty() {
                stats.books_skipped += 1;
                self.db
                    .set_book_fingerprint(&collection.id, &doc.book_id, &fp)?;
                on_event(IndexEvent::Skipped {
                    n,
                    total,
                    path: path.clone(),
                    reason: "no extractable text".into(),
                });
                continue;
            }

            let mut chunks = chunk_book(&doc, counter, &params);
            let chunks_total = chunks.len();
            let mut chunks_done = 0;
            on_event(IndexEvent::Embedding {
                n,
                total,
                title: doc.title.clone(),
                chunks_done,
                chunks_total,
            });
            for batch in chunks.chunks_mut(EMBED_BATCH) {
                let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
                let vectors = embedder.embed(&texts)?;
                for (c, v) in batch.iter_mut().zip(vectors) {
                    c.vector = Some(v);
                }
                chunks_done += batch.len();
                on_event(IndexEvent::Embedding {
                    n,
                    total,
                    title: doc.title.clone(),
                    chunks_done,
                    chunks_total,
                });
            }
            store.delete_book(&doc.book_id).await.ok();
            stats.chunks_written += store.add_chunks(&chunks).await?;
            stats.books_indexed += 1;
            wrote = true;
            self.db
                .set_book_fingerprint(&collection.id, &doc.book_id, &fp)?;
            on_event(IndexEvent::Indexed {
                n,
                total,
                title: doc.title.clone(),
                chunks: chunks.len(),
            });
        }

        if wrote {
            store.ensure_fts_index().await?;
        }
        on_event(IndexEvent::Finished {
            stats: stats.clone(),
        });
        Ok(stats)
    }

    /// Retrieve + rerank for a collection, then stream a grounded answer; returns
    /// the cited results. `on_token` receives streamed answer chunks.
    #[allow(clippy::too_many_arguments)] // engine handles are passed in by design
    pub async fn answer(
        &self,
        collection: &Collection,
        question: &str,
        embedder: &mut Embedder,
        reranker: &mut Reranker,
        llm: &OllamaClient,
        model: &str,
        on_token: impl FnMut(&str),
    ) -> Result<Vec<SearchResult>, ServiceError> {
        let store = Store::open(&collection.db_path, "chunks").await?;
        let results = search(
            &store,
            embedder,
            reranker,
            question,
            self.settings.final_top_k,
            self.settings.hybrid_top_k,
        )
        .await?;
        if results.is_empty() {
            return Ok(results);
        }
        let prompt = build_prompt(question, &results);
        llm.generate_stream(model, &prompt, on_token, |_| {})
            .await?;
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_opens_and_manages_collections() {
        let dir = tempfile::tempdir().unwrap();
        let svc = Service::new(dir.path()).unwrap();
        svc.db
            .upsert_collection(&Collection {
                id: "c1".into(),
                name: "Tech".into(),
                db_path: dir.path().join("c1").to_string_lossy().into(),
                source_paths: vec!["/books".into()],
                embed_model: "bge-m3".into(),
            })
            .unwrap();
        assert_eq!(svc.db.list_collections().unwrap().len(), 1);
    }

    #[test]
    fn fingerprint_changes_with_file() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, "hello").unwrap();
        let fp1 = file_fingerprint(&f);
        std::fs::write(&f, "hello world longer").unwrap();
        let fp2 = file_fingerprint(&f);
        assert_ne!(fp1, fp2);
    }
}
