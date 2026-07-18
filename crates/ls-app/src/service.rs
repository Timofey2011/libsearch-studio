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
    /// Pending re-chunk work this run planned (forced embeds + forced twins +
    /// legacy remaps). The armed flag clears only at forced == 0.
    #[serde(default)]
    pub forced: usize,
    /// Per-extension `(indexed, skipped-or-failed)` counts — what makes the
    /// first post-upgrade re-scope run legible ("indexed 12 md, 3 txt ·
    /// skipped 480 pdf"). Additive; absent entries mean zero.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub by_format: std::collections::BTreeMap<String, (usize, usize)>,
}

impl IndexStats {
    /// Fold another run's stats in (the GPU phase + the standard-engine sweep
    /// report as ONE run): field sums, by_format entry-wise.
    pub fn merge(&mut self, other: IndexStats) {
        self.books_indexed += other.books_indexed;
        self.books_unchanged += other.books_unchanged;
        self.books_skipped += other.books_skipped;
        self.books_failed += other.books_failed;
        self.chunks_written += other.chunks_written;
        self.forced += other.forced;
        for (ext, (i, s)) in other.by_format {
            let e = self.by_format.entry(ext).or_default();
            e.0 += i;
            e.1 += s;
        }
    }

    pub fn count_format(&mut self, path: &str, indexed: bool) {
        let ext = ls_core::ext_of(path).unwrap_or("other").to_string();
        let e = self.by_format.entry(ext).or_default();
        if indexed {
            e.0 += 1;
        } else {
            e.1 += 1;
        }
    }
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

/// A cheap content signature for dedup that survives timestamp changes: the file
/// size plus a hash of its first and last 256 KiB. Two files with identical
/// content share a signature even if moved, re-synced, or re-timestamped — so an
/// already-embedded book is recognized regardless of `mtime` or path. Reads at
/// most ~512 KiB (cheap next to embedding), not the whole file.
pub fn content_signature(path: &Path) -> String {
    use std::hash::Hasher;
    use std::io::{Read, Seek, SeekFrom};

    const SAMPLE: u64 = 256 * 1024;
    let Ok(mut f) = std::fs::File::open(path) else {
        return CONTENT_SIG_MISSING.to_string();
    };
    let Ok(len) = f.metadata().map(|m| m.len()) else {
        return CONTENT_SIG_MISSING.to_string();
    };
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_u64(len);

    // Hash exactly `max` bytes (or to a genuine EOF); a read error or short
    // read returns None. A signature derived from PARTIAL bytes must never be
    // reported as confident: two distinct same-length files that both fail
    // mid-read (dehydrated cloud placeholder, permission flip) would otherwise
    // produce IDENTICAL signatures and dedup would remap one book onto the
    // other's path.
    fn hash_n(f: &mut std::fs::File, h: &mut impl Hasher, max: u64) -> Option<u64> {
        let mut hashed = 0u64;
        let mut buf = [0u8; 64 * 1024];
        while hashed < max {
            let want = (max - hashed).min(buf.len() as u64) as usize;
            match f.read(&mut buf[..want]) {
                Ok(0) => break, // genuine EOF
                Ok(n) => {
                    h.write(&buf[..n]);
                    hashed += n as u64;
                }
                Err(_) => return None,
            }
        }
        Some(hashed)
    }

    let head_expected = len.min(SAMPLE);
    match hash_n(&mut f, &mut h, SAMPLE) {
        Some(n) if n >= head_expected => {}
        _ => return CONTENT_SIG_MISSING.to_string(),
    }
    // Hash the tail too (only meaningfully distinct for files > 2 samples).
    if len > 2 * SAMPLE {
        if f.seek(SeekFrom::End(-(SAMPLE as i64))).is_err() {
            return CONTENT_SIG_MISSING.to_string();
        }
        match hash_n(&mut f, &mut h, SAMPLE) {
            Some(n) if n >= SAMPLE => {}
            _ => return CONTENT_SIG_MISSING.to_string(),
        }
    }
    format!("{len:x}:{:016x}", h.finish())
}

/// The failure sentinel shared by [`content_signature`] and [`file_fingerprint`].
/// It is POISON, never identity: reverse lookups treat it as no-match, writers
/// refuse to persist it, and in-run dedup maps never key on it.
pub const CONTENT_SIG_MISSING: &str = "missing";

/// Stable hex hash for capabilities fingerprints.
pub fn caps_hash(parts: &[&str]) -> String {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for p in parts {
        h.write(p.as_bytes());
        h.write_u8(0);
    }
    format!("{:016x}", h.finish())
}

/// The CPU pipeline's runtime-capabilities hash (§2.8): compile-time ingest
/// surface ⊕ the probed external-converter set. Installing/removing a
/// converter changes the hash, so skips recorded under the old hash are
/// retried automatically.
pub fn cpu_caps_ver() -> String {
    const TOOLS: &[&str] = &["textutil", "antiword", "soffice", "djvutxt", "7z"];
    let probed: Vec<String> = TOOLS
        .iter()
        .filter(|t| {
            std::process::Command::new("which")
                .arg(t)
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        })
        .map(|t| t.to_string())
        .collect();
    let ingest = ls_core::INGEST_EXTS.join(",");
    let tools = probed.join(",");
    caps_hash(&["cpu-v1", &ingest, &tools])
}

/// Outcome of [`backfill_book_state`].
#[derive(Debug, Default)]
pub struct BackfillOutcome {
    pub seeded: usize,
    pub unseedable: usize,
    pub unseedable_paths: Vec<String>,
}

/// Seed the fingerprint manifest for books already present in a lance store
/// (e.g. loaded via Parquet import), so the dedup recognizes them and never
/// re-embeds. Unreadable files are NOT seeded — a row carrying the failure
/// sentinel is a dedup trap (see [`CONTENT_SIG_MISSING`]) — they are counted
/// and listed instead, and can be seeded by a later run once readable.
/// Rows are stamped `chunker_ver = 0` (legacy): their chunks came from an
/// older scheme and the re-index nudge should stay honest about that.
pub fn backfill_book_state(
    db: &crate::Db,
    collection_id: &str,
    pairs: &[(String, String)],
) -> Result<BackfillOutcome, crate::DbError> {
    let mut out = BackfillOutcome::default();
    for (book_id, source_path) in pairs {
        let p = Path::new(source_path);
        let fp = file_fingerprint(p);
        let csig = content_signature(p);
        if crate::is_sig_sentinel(&fp) || crate::is_sig_sentinel(&csig) {
            out.unseedable += 1;
            out.unseedable_paths.push(source_path.clone());
            continue;
        }
        db.set_book_state_ver(collection_id, book_id, &fp, &csig, source_path, 0)?;
        out.seeded += 1;
    }
    Ok(out)
}

/// True when a fingerprint/content-signature value is the failure sentinel (or
/// empty) and must never participate in identity matching.
pub fn is_sig_sentinel(v: &str) -> bool {
    v.is_empty() || v == CONTENT_SIG_MISSING
}

/// `(path, size, mtime)` fingerprint — changes iff the file changes.
pub fn file_fingerprint(path: &Path) -> String {
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
    #[allow(clippy::too_many_arguments)] // engine handles + callbacks by design
    pub async fn index_collection(
        &self,
        collection: &Collection,
        paths: &[String],
        embedder: &mut Embedder,
        counter: &dyn TokenCounter,
        is_cancelled: impl Fn() -> bool,
        mut on_event: impl FnMut(IndexEvent),
    ) -> Result<IndexStats, ServiceError> {
        let store = Store::open_or_create(&collection.db_path, "chunks").await?;
        let params = ChunkParams::default();
        let mut stats = IndexStats::default();
        let total = paths.len();
        on_event(IndexEvent::Started { total });

        // Books already embedded in the index, so a re-index can skip them even
        // when the fingerprint manifest is empty (index built before the manifest
        // existed, or via Parquet import). One scan up front, then O(1) lookups.
        let indexed = store.indexed_book_ids().await.unwrap_or_default();
        let paths_by_id: std::collections::HashMap<String, String> = store
            .book_paths()
            .await
            .unwrap_or_default()
            .into_iter()
            .collect();
        // Fill source_path for manifest rows that predate the column (metadata
        // only — no file reads, so safe against dehydrated cloud placeholders).
        self.db
            .backfill_source_paths(&collection.id, &paths_by_id)?;

        // Skip-record hygiene: sweep rows whose paths left the collection's
        // sources, and compute this run's CPU capabilities hash (§2.8).
        let _ = self.db.gc_skips(&collection.id, &collection.source_paths);
        let caps_ver = cpu_caps_ver();
        // Re-chunk armed? (v0.15) Planner then forces legacy-chunker books.
        let rechunk = self.db.rechunk_pending(&collection.id).unwrap_or(false);
        // Best-effort formats (§7) convert into the app-owned cache; the
        // original file keeps identity (§0.b).
        let conv_dir = self.data_dir.join("converted");

        // The shared dedup pre-filter (also used by the GPU fast-index path):
        // decides skip / remap / refresh / embed per candidate, writing nothing.
        let candidates: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
        let ctx = crate::plan::PlanCtx {
            collection_id: &collection.id,
            db: &self.db,
            indexed_ids: &indexed,
            paths_by_id: &paths_by_id,
            pipeline: "cpu",
            caps_ver: &caps_ver,
            fp_fn: &|p| file_fingerprint(p),
            csig_fn: &|p| content_signature(p),
            rechunk,
        };
        let plan = crate::plan::plan_index_run(&candidates, &ctx)?;
        stats.forced = plan.forced_count;
        // Uncollapsed (book_id, path) pairs → per-path deletion sets: a
        // re-embedded book must replace chunks under EVERY id at its path.
        let ids_by_path: std::collections::HashMap<String, Vec<String>> = store
            .book_path_pairs()
            .await
            .unwrap_or_default()
            .into_iter()
            .fold(std::collections::HashMap::new(), |mut m, (id, p)| {
                m.entry(p).or_default().push(id);
                m
            });

        // Metadata-only actions first: manifest refreshes, then moved-file
        // re-points (chunk metadata rewrite, no re-embedding).
        for r in &plan.state_refreshes {
            self.db.refresh_book_state(
                &collection.id,
                &r.book_id,
                &r.fingerprint,
                r.content_sig.as_deref(),
                &r.path,
            )?;
        }
        for m in &plan.remaps {
            store.remap_book(&m.old_id, &m.new_id, &m.path).await.ok();
            self.db.delete_book_state(&collection.id, &m.old_id)?;
            self.db.set_book_state_ver(
                &collection.id,
                &m.new_id,
                &m.fingerprint,
                m.content_sig.as_deref().unwrap_or(""),
                &m.path,
                m.chunker_ver,
            )?;
        }

        let mut n = 0usize;
        for (path, reason) in &plan.preskips {
            n += 1;
            match reason {
                crate::plan::SkipReason::Unreadable => {
                    stats.books_failed += 1;
                    on_event(IndexEvent::Skipped {
                        n,
                        total,
                        path: path.clone(),
                        reason: "unreadable (permissions, or an offline cloud placeholder?)".into(),
                    });
                }
                // A recorded skip with unchanged file + capabilities: counted,
                // never re-announced (§2.8).
                crate::plan::SkipReason::Silenced => {
                    stats.books_skipped += 1;
                }
                _ => {
                    stats.books_unchanged += 1;
                    stats.count_format(path, false);
                    on_event(IndexEvent::Unchanged {
                        n,
                        total,
                        title: title_of(Path::new(path)),
                    });
                }
            }
        }

        let mut wrote = false;
        'files: for item in &plan.to_embed {
            // Stop promptly when the user hits Stop; books written so far are kept.
            if is_cancelled() {
                break;
            }
            n += 1;
            let path = &item.path;
            let p = Path::new(path);
            let (fp, csig) = (&item.fingerprint, &item.content_sig);

            // Formats the CPU pipeline deliberately does not handle (e.g. xps)
            // are recorded once with a platform-honest reason; the GPU helper
            // picks them up where it exists (§4.1).
            if let Some(reason) = ls_core::ext_of(path).and_then(ls_extract::cpu_directed_skip) {
                stats.books_skipped += 1;
                stats.count_format(path, false);
                self.db
                    .upsert_skip(&collection.id, path, "cpu", fp, reason, &caps_ver)?;
                on_event(IndexEvent::Skipped {
                    n,
                    total,
                    path: path.clone(),
                    reason: reason.into(),
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
            let doc = match ls_extract::extract_with_cache(p, &conv_dir) {
                Ok(d) => d,
                Err(e) => {
                    stats.books_failed += 1;
                    // Recorded so the next run is silent about it — retried
                    // when the file's bytes or this pipeline's caps change.
                    self.db.upsert_skip(
                        &collection.id,
                        path,
                        "cpu",
                        fp,
                        &e.to_string(),
                        &caps_ver,
                    )?;
                    stats.count_format(path, false);
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
                // Behavior change (§2.8): "no extractable text" is a SKIP, not
                // success-shaped book_state — an extractor upgrade (new caps
                // hash) automatically re-attempts these books.
                self.db.upsert_skip(
                    &collection.id,
                    path,
                    "cpu",
                    fp,
                    "no extractable text",
                    &caps_ver,
                )?;
                stats.count_format(path, false);
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
                // Mid-book check so a large book (hundreds of chunks) stops quickly.
                if is_cancelled() {
                    break 'files;
                }
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
            // Replace, never append: drop old chunks under every id at this
            // path, and clear same-path manifest rows across id schemes
            // BEFORE the fresh write (the path-keyed delete would otherwise
            // remove the row we are about to insert).
            let mut old_ids: Vec<String> =
                ids_by_path.get(path.as_str()).cloned().unwrap_or_default();
            if !old_ids.contains(&doc.book_id) {
                old_ids.push(doc.book_id.clone());
            }
            store.delete_books(&old_ids).await?;
            stats.chunks_written += store.add_chunks(&chunks).await?;
            stats.books_indexed += 1;
            wrote = true;
            self.db
                .clear_book_state_by_path(&collection.id, path, &doc.book_id)?;
            self.db
                .set_book_state(&collection.id, &doc.book_id, fp, csig, path)?;
            stats.count_format(path, true);
            // Success erases past skip records for this path — both pipelines.
            self.db.erase_skips(&collection.id, path)?;
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
    #[test]
    fn stats_merge_sums_fields_and_by_format() {
        let mut a = IndexStats {
            books_indexed: 2,
            books_unchanged: 10,
            books_skipped: 1,
            books_failed: 0,
            chunks_written: 40,
            forced: 1,
            by_format: [("pdf".to_string(), (2usize, 10usize))]
                .into_iter()
                .collect(),
        };
        let b = IndexStats {
            books_indexed: 1,
            books_unchanged: 0,
            books_skipped: 2,
            books_failed: 1,
            chunks_written: 8,
            forced: 2,
            by_format: [
                ("doc".to_string(), (1usize, 2usize)),
                ("pdf".to_string(), (0usize, 1usize)),
            ]
            .into_iter()
            .collect(),
        };
        a.merge(b);
        assert_eq!(a.books_indexed, 3);
        assert_eq!(a.books_unchanged, 10);
        assert_eq!(a.books_skipped, 3);
        assert_eq!(a.books_failed, 1);
        assert_eq!(a.chunks_written, 48);
        assert_eq!(a.forced, 3);
        assert_eq!(a.by_format["pdf"], (2, 11));
        assert_eq!(a.by_format["doc"], (1, 2));
    }

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

    #[test]
    fn content_signature_is_path_and_time_independent() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.pdf");
        let b = dir.path().join("sub/b.pdf"); // different path/name
        std::fs::create_dir_all(b.parent().unwrap()).unwrap();
        std::fs::write(&a, b"the same bytes in both files").unwrap();
        std::fs::write(&b, b"the same bytes in both files").unwrap();
        // Identical content -> identical signature regardless of path/mtime.
        assert_eq!(content_signature(&a), content_signature(&b));
        // Different content -> different signature.
        std::fs::write(&b, b"different bytes entirely here").unwrap();
        assert_ne!(content_signature(&a), content_signature(&b));
        // Missing file is handled.
        assert_eq!(content_signature(&dir.path().join("nope.pdf")), "missing");
    }
}
