//! Tauri bridge: thin command/event layer over the engine crates.
//!
//! Holds the expensive engine handles (Embedder/Reranker) in app state behind a
//! Tokio mutex; collection metadata lives in SQLite (opened per command — cheap).
//! `ask` streams answer tokens to the webview via the `ask-token` event.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ls_app::{
    Citation, Collection, Conversation, Db, IndexEvent, IndexStats, Message, Role, Service,
};
use ls_artifacts::{ArtifactDoc, ArtifactRenderer, Markdown, Source};
use ls_embed::{BgeTokenCounter, Embedder, Reranker};
use ls_index::Store;
use ls_llm::{
    build_prompt_with_history, AnthropicClient, HistoryTurn, Llm, OllamaClient, OpenAiCompatClient,
};
use ls_query::{search_multi, SearchResult};
use tauri::{Emitter, Manager, State, WebviewWindow};
use tokio::io::AsyncBufReadExt;
use tokio::sync::Mutex;

/// Monotonic-ish unique id from the wall clock (nanos, hex).
fn new_id() -> String {
    format!(
        "{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

/// Map a fresh search result to the persisted citation shape (kept in sync so a
/// reloaded message renders identically to a live one).
fn to_citation(r: &SearchResult) -> Citation {
    Citation {
        rank: r.rank,
        citation: r.citation.clone(),
        source_path: r.source_path.clone(),
        page: r.page,
        text: r.text.clone(),
    }
}

struct Engine {
    embedder: Embedder,
    reranker: Reranker,
}

struct AppState {
    data_dir: PathBuf,
    models_dir: PathBuf,
    // Settings and the LLM client are editable at runtime (Settings UI), so both
    // sit behind plain mutexes; values are cloned out before any await.
    settings: std::sync::Mutex<ls_app::Settings>,
    llm: std::sync::Mutex<Llm>,
    engine: Mutex<Option<Engine>>,
    /// Set by `cancel_indexing` to ask an in-progress index run to stop.
    cancel: Arc<AtomicBool>,
}

impl AppState {
    fn db(&self) -> Result<Db, String> {
        Db::open(self.data_dir.join("app.db")).map_err(|e| e.to_string())
    }
    fn settings(&self) -> ls_app::Settings {
        self.settings.lock().unwrap().clone()
    }
    fn llm(&self) -> Llm {
        self.llm.lock().unwrap().clone()
    }
}

/// OpenAI-compatible base URL for a cloud provider id, if it is one.
fn openai_compat_base(provider: &str) -> Option<&'static str> {
    match provider {
        "openai" => Some("https://api.openai.com/v1"),
        // Gemini exposes an OpenAI-compatible surface.
        "gemini" => Some("https://generativelanguage.googleapis.com/v1beta/openai"),
        "fireworks" => Some("https://api.fireworks.ai/inference/v1"),
        "ollama_cloud" => Some("https://ollama.com/v1"),
        _ => None,
    }
}

/// Build the synthesis client for the configured provider.
fn build_llm(s: &ls_app::Settings) -> Llm {
    match s.llm_provider.as_str() {
        "anthropic" => Llm::Anthropic(AnthropicClient::new(&s.creds("anthropic").api_key)),
        p if openai_compat_base(p).is_some() => Llm::OpenAiCompat(OpenAiCompatClient::new(
            openai_compat_base(p).unwrap(),
            &s.creds(p).api_key,
        )),
        _ => Llm::Ollama(OllamaClient::new(&s.ollama_host)),
    }
}

/// Prefer the int8 reranker (2.3x faster on CPU, quality preserved) when present,
/// else the fp32 one. The embedder stays fp32 to match the index's vectors.
fn reranker_dir(models: &Path) -> PathBuf {
    let int8 = models.join("bge-reranker-v2-m3-int8");
    if int8.join("model.onnx").exists() {
        int8
    } else {
        models.join("bge-reranker-v2-m3")
    }
}

#[tauri::command]
async fn list_collections(state: State<'_, AppState>) -> Result<Vec<Collection>, String> {
    state.db()?.list_collections().map_err(|e| e.to_string())
}

#[tauri::command]
async fn create_collection(
    state: State<'_, AppState>,
    name: String,
    source_paths: Vec<String>,
) -> Result<Collection, String> {
    let id = format!(
        "{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let db_path = state.data_dir.join("collections").join(&id);
    let coll = Collection {
        id,
        name,
        db_path: db_path.to_string_lossy().into_owned(),
        source_paths,
        embed_model: "bge-m3".into(),
    };
    state
        .db()?
        .upsert_collection(&coll)
        .map_err(|e| e.to_string())?;
    Ok(coll)
}

/// Replace a collection's source paths (e.g. after the user adds folders).
#[tauri::command]
async fn set_collection_paths(
    state: State<'_, AppState>,
    collection_id: String,
    source_paths: Vec<String>,
) -> Result<Collection, String> {
    let db = state.db()?;
    let mut coll = db
        .list_collections()
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|c| c.id == collection_id)
        .ok_or_else(|| format!("collection {collection_id} not found"))?;
    coll.source_paths = source_paths;
    db.upsert_collection(&coll).map_err(|e| e.to_string())?;
    Ok(coll)
}

/// Delete a collection: its DB row + fingerprints, and its LanceDB directory.
#[tauri::command]
async fn delete_collection(
    state: State<'_, AppState>,
    collection_id: String,
) -> Result<(), String> {
    let db = state.db()?;
    let db_path = db
        .list_collections()
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|c| c.id == collection_id)
        .map(|c| c.db_path);
    db.delete_collection(&collection_id)
        .map_err(|e| e.to_string())?;
    if let Some(p) = db_path {
        let _ = std::fs::remove_dir_all(&p); // best-effort; index may not exist yet
    }
    Ok(())
}

/// Switch the active synthesis provider without opening full settings. Persists
/// and rebuilds the client; the chosen provider must already have its key set
/// (or be local Ollama).
#[tauri::command]
async fn set_provider(state: State<'_, AppState>, provider: String) -> Result<(), String> {
    let s = {
        let mut g = state.settings.lock().unwrap();
        g.llm_provider = provider;
        g.clone()
    };
    s.save(state.data_dir.join("settings.toml"))
        .map_err(|e| e.to_string())?;
    *state.llm.lock().unwrap() = build_llm(&s);
    Ok(())
}

/// Index (or re-index) a collection's source paths, streaming progress to the UI
/// via `index-progress` events. Incremental: unchanged files are skipped.
#[tauri::command]
async fn index_collection(
    state: State<'_, AppState>,
    window: WebviewWindow,
    collection_id: String,
) -> Result<IndexStats, String> {
    let coll = state
        .db()?
        .list_collections()
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|c| c.id == collection_id)
        .ok_or_else(|| format!("collection {collection_id} not found"))?;

    let files = ls_app::discover_books(&coll.source_paths);
    if files.is_empty() {
        return Err("no PDF files found under the collection's source paths".into());
    }

    let models_dir = state.models_dir.clone();
    let data_dir = state.data_dir.clone();
    let w = window.clone();
    // Fresh cancellation flag for this run; the loop polls it between files.
    state.cancel.store(false, Ordering::SeqCst);
    let cancel = state.cancel.clone();

    // The embedder load below takes ~20s on first index; tell the UI so the bar
    // doesn't look frozen before the first book is processed.
    let _ = window.emit("index-progress", IndexEvent::Loading);

    // Run the whole job on a blocking thread with its own runtime: the rusqlite
    // connection and tokenizer aren't Send, so they must never cross an await on
    // the main (multi-threaded) runtime. A dedicated embedder is loaded here
    // rather than borrowing the shared one, so chat stays usable during indexing.
    let stats = tauri::async_runtime::spawn_blocking(move || -> Result<IndexStats, String> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        rt.block_on(async move {
            let mut embedder =
                Embedder::load(models_dir.join("bge-m3")).map_err(|e| e.to_string())?;
            let counter =
                BgeTokenCounter::load(models_dir.join("bge-m3")).map_err(|e| e.to_string())?;
            let svc = Service::new(&data_dir).map_err(|e| e.to_string())?;
            svc.index_collection(
                &coll,
                &files,
                &mut embedder,
                &counter,
                || cancel.load(Ordering::SeqCst),
                |ev| {
                    let _ = w.emit("index-progress", ev);
                },
            )
            .await
            .map_err(|e| e.to_string())
        })
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(stats)
}

/// Ask any in-progress index run (CPU or GPU) to stop. The CPU loop checks this
/// between files; the GPU path kills the Python helper.
#[tauri::command]
async fn cancel_indexing(state: State<'_, AppState>) -> Result<(), String> {
    state.cancel.store(true, Ordering::SeqCst);
    Ok(())
}

/// One parsed per-book line from the Python helper's stderr.
enum PyProgress {
    Book { title: String, chunks: usize },
    Skip { path: String, reason: String },
    Other,
}

/// Parse a `[i/n] …` progress line. The helper's own `i/n` is per-batch, so the
/// caller numbers books globally; we only need the title/chunks (or skip).
fn parse_py_progress(line: &str) -> PyProgress {
    let l = line.trim();
    let Some(rest) = l
        .strip_prefix('[')
        .and_then(|s| s.find(']').map(|i| &s[i + 1..]))
    else {
        return PyProgress::Other;
    };
    let rest = rest.trim();
    if rest.starts_with("skip") {
        let path = rest.rsplit(' ').next().unwrap_or(rest).to_string();
        return PyProgress::Skip {
            path,
            reason: "no extractable text".into(),
        };
    }
    // "<title>: <n> chunks …"
    let (title, chunks) = match rest.rfind(": ") {
        Some(c) => {
            let n = rest[c + 2..]
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            (rest[..c].to_string(), n)
        }
        None => (rest.to_string(), 0),
    };
    PyProgress::Book { title, chunks }
}

/// Run the Python helper over one batch of files and import the resulting Parquet
/// into `store`. Streams per-book progress (numbered from `*gi`) and the raw log,
/// and reacts to cancellation by killing the helper. Returns `(chunks, indexed,
/// skipped)`, or `Ok(None)` if the user cancelled mid-batch.
#[allow(clippy::too_many_arguments)]
async fn run_py_batch(
    window: &WebviewWindow,
    cancel: &AtomicBool,
    store: &Store,
    py: &str,
    script: &str,
    parquet: &Path,
    batch: &[String],
    gi: &mut usize,
    total: usize,
) -> Result<Option<(usize, usize, usize)>, String> {
    let mut cmd = tokio::process::Command::new(py);
    cmd.arg(script).arg("--out").arg(parquet);
    for f in batch {
        cmd.arg(f);
    }
    cmd.env("PYTHONUNBUFFERED", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to start Python helper: {e}"))?;

    let stderr = child.stderr.take().ok_or("no stderr from helper")?;
    let mut reader = tokio::io::BufReader::new(stderr).lines();
    if let Some(stdout) = child.stdout.take() {
        let w = window.clone();
        tokio::spawn(async move {
            let mut l = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(line)) = l.next_line().await {
                let _ = w.emit("index-log", line);
            }
        });
    }

    let mut indexed = 0usize;
    let mut skipped = 0usize;
    let mut tail = String::new();
    let mut cancelled = false;
    loop {
        tokio::select! {
            maybe = reader.next_line() => {
                let line = match maybe { Ok(Some(l)) => l, _ => break };
                let _ = window.emit("index-log", line.clone());
                match parse_py_progress(&line) {
                    PyProgress::Book { title, chunks } => {
                        *gi += 1;
                        indexed += 1;
                        let _ = window.emit("index-progress", IndexEvent::Indexed { n: *gi, total, title, chunks });
                    }
                    PyProgress::Skip { path, reason } => {
                        *gi += 1;
                        skipped += 1;
                        let _ = window.emit("index-progress", IndexEvent::Skipped { n: *gi, total, path, reason });
                    }
                    PyProgress::Other => {}
                }
                tail.push_str(&line);
                tail.push('\n');
                if tail.len() > 4000 { let cut = tail.len() - 4000; tail.drain(..cut); }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(250)) => {
                if cancel.load(Ordering::SeqCst) {
                    let _ = child.start_kill();
                    cancelled = true;
                    break;
                }
            }
        }
    }
    let status = child.wait().await.map_err(|e| e.to_string())?;

    if cancelled {
        let _ = std::fs::remove_file(parquet);
        return Ok(None);
    }
    if !status.success() {
        let _ = std::fs::remove_file(parquet);
        return Err(format!(
            "Python helper failed (exit {}):\n{}",
            status.code().unwrap_or(-1),
            tail.trim_end()
        ));
    }
    let chunks = store
        .import_parquet(parquet)
        .await
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(parquet);
    Ok(Some((chunks, indexed, skipped)))
}

/// Fast index a collection by offloading bulk embedding to the configured
/// Python/MPS helper, then importing the resulting Parquet into the collection's
/// LanceDB. Streams the helper's per-book progress as `index-progress` events.
#[tauri::command]
async fn fast_index_collection(
    state: State<'_, AppState>,
    window: WebviewWindow,
    collection_id: String,
) -> Result<IndexStats, String> {
    let settings = state.settings();
    let py = settings.python_bin.trim().to_string();
    let script = settings.indexer_script.trim().to_string();
    if py.is_empty() || script.is_empty() {
        return Err(
            "Set the Python interpreter and indexer script in Settings → Fast index (GPU).".into(),
        );
    }
    // Fresh cancellation flag for this run; we kill the helper if it's set.
    state.cancel.store(false, Ordering::SeqCst);

    let coll = state
        .db()?
        .list_collections()
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|c| c.id == collection_id)
        .ok_or_else(|| format!("collection {collection_id} not found"))?;
    let files = ls_app::discover_books(&coll.source_paths);
    if files.is_empty() {
        return Err("no PDF files found under the collection's source paths".into());
    }

    let _ = window.emit("index-progress", IndexEvent::Loading);

    // Embed only genuinely new/changed files: skip unchanged ones, re-point moved
    // ones, and skip any already present in the index (mirrors the CPU path). The
    // DB handle (rusqlite, !Send) is confined to blocks that don't cross `.await`.
    let store = Store::open_or_create(&coll.db_path, "chunks")
        .await
        .map_err(|e| e.to_string())?;
    let indexed = store.indexed_book_ids().await.unwrap_or_default();
    let mut to_embed: Vec<String> = Vec::new();
    let mut preskipped = 0usize;
    let mut remaps: Vec<(String, String, String)> = Vec::new();
    {
        let db = state.db()?;
        // Content signatures queued this run, so duplicate files in the same batch
        // are embedded only once.
        let mut seen_content: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for f in &files {
            let p = Path::new(f);
            let book_id = ls_app::stable_book_id(p);
            let fp = ls_app::file_fingerprint(p);
            if db
                .book_fingerprint(&coll.id, &book_id)
                .ok()
                .flatten()
                .as_deref()
                == Some(fp.as_str())
            {
                preskipped += 1;
                continue;
            }
            if let Some(old) = db.book_id_for_fingerprint(&coll.id, &fp).ok().flatten() {
                if old != book_id {
                    let _ = db.delete_book_state(&coll.id, &old);
                    let _ = db.set_book_fingerprint(&coll.id, &book_id, &fp);
                    remaps.push((old, book_id, f.clone()));
                    preskipped += 1;
                    continue;
                }
            }
            if indexed.contains(&book_id) {
                let _ = db.set_book_fingerprint(&coll.id, &book_id, &fp);
                preskipped += 1;
                continue;
            }
            // Content-checksum dedup: same bytes under a new path / changed mtime,
            // or a duplicate already queued this run — skip re-embedding.
            let csig = ls_app::content_signature(p);
            if seen_content.contains_key(&csig) {
                preskipped += 1;
                continue;
            }
            if let Some(old) = db.book_id_for_content(&coll.id, &csig).ok().flatten() {
                if old != book_id {
                    let _ = db.delete_book_state(&coll.id, &old);
                    let _ = db.set_book_state(&coll.id, &book_id, &fp, &csig);
                    remaps.push((old, book_id, f.clone()));
                    preskipped += 1;
                    continue;
                }
            }
            seen_content.insert(csig, book_id);
            to_embed.push(f.clone());
        }
    }
    // Apply any path re-points now that the DB handle is dropped.
    for (old, new, path) in &remaps {
        store.remap_book(old, new, path).await.ok();
    }

    if to_embed.is_empty() {
        let stats = IndexStats {
            books_indexed: 0,
            books_unchanged: preskipped,
            books_skipped: 0,
            books_failed: 0,
            chunks_written: 0,
        };
        let _ = window.emit(
            "index-progress",
            IndexEvent::Finished {
                stats: stats.clone(),
            },
        );
        return Ok(stats);
    }
    let total = to_embed.len();

    let tmp_dir = state.data_dir.join("tmp");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| e.to_string())?;

    let _ = window.emit("index-progress", IndexEvent::Started { total });
    let _ = window.emit(
        "index-log",
        format!("Embedding {total} new/changed file(s) on the GPU helper…"),
    );

    // Checkpoint every CHECKPOINT_N books: embed a batch, import it, and commit
    // its fingerprints. A Stop/crash then loses only the current batch — the rest
    // stays in the index and the dedup resumes from there on the next run.
    const CHECKPOINT_N: usize = 40;
    let mut gi = 0usize; // global book counter for progress numbering
    let mut indexed = 0usize;
    let mut skipped = 0usize;
    let mut chunks_written = 0usize;
    let mut cancelled = false;
    for (bi, batch) in to_embed.chunks(CHECKPOINT_N).enumerate() {
        if state.cancel.load(Ordering::SeqCst) {
            cancelled = true;
            break;
        }
        let parquet = tmp_dir.join(format!("fastindex-{collection_id}-{bi}.parquet"));
        match run_py_batch(
            &window,
            &state.cancel,
            &store,
            &py,
            &script,
            &parquet,
            batch,
            &mut gi,
            total,
        )
        .await?
        {
            None => {
                cancelled = true;
                break;
            }
            Some((chunks, idx, skip)) => {
                chunks_written += chunks;
                indexed += idx;
                skipped += skip;
                // Commit this batch's fingerprints so it's not re-embedded on resume.
                let db = state.db()?;
                for f in batch {
                    let p = Path::new(f);
                    let _ = db.set_book_state(
                        &coll.id,
                        &ls_app::stable_book_id(p),
                        &ls_app::file_fingerprint(p),
                        &ls_app::content_signature(p),
                    );
                }
            }
        }
    }

    // Build the FTS index once the run settles (cheap to rebuild; skipped on a
    // cancel — it's rebuilt when a later run completes).
    if chunks_written > 0 && !cancelled {
        store.ensure_fts_index().await.map_err(|e| e.to_string())?;
    }

    let stats = IndexStats {
        books_indexed: indexed,
        books_unchanged: preskipped,
        books_skipped: skipped,
        books_failed: 0,
        chunks_written,
    };
    let _ = window.emit(
        "index-progress",
        IndexEvent::Finished {
            stats: stats.clone(),
        },
    );
    Ok(stats)
}

// Helper scripts embedded in the binary so a packaged app can self-provision its
// own Python helper without shipping the repo.
const GPU_EMBED_PY: &str = include_str!("../../scripts/gpu_embed.py");
const EXPORT_ONNX_PY: &str = include_str!("../../scripts/export_onnx.py");

/// Run a setup subprocess, streaming its stdout+stderr to the UI as `setup-log`.
async fn stream_to_log(
    window: &WebviewWindow,
    label: &str,
    mut cmd: tokio::process::Command,
) -> Result<(), String> {
    let _ = window.emit("setup-log", format!("• {label}…"));
    cmd.env("PYTHONUNBUFFERED", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("{label}: {e}"))?;
    let out = child.stdout.take().ok_or("no stdout")?;
    let err = child.stderr.take().ok_or("no stderr")?;
    let w1 = window.clone();
    let w2 = window.clone();
    let h1 = tokio::spawn(async move {
        let mut l = tokio::io::BufReader::new(out).lines();
        while let Ok(Some(line)) = l.next_line().await {
            let _ = w1.emit("setup-log", line);
        }
    });
    let h2 = tokio::spawn(async move {
        let mut l = tokio::io::BufReader::new(err).lines();
        while let Ok(Some(line)) = l.next_line().await {
            let _ = w2.emit("setup-log", line);
        }
    });
    let status = child.wait().await.map_err(|e| e.to_string())?;
    let _ = tokio::join!(h1, h2);
    if !status.success() {
        return Err(format!(
            "{label} failed (exit {})",
            status.code().unwrap_or(-1)
        ));
    }
    Ok(())
}

/// One-click self-setup: create a local venv, install the embedding deps, export
/// the ONNX models locally, and point settings at all of it. Streams progress as
/// `setup-log` events. The dmg stays small; everything is provisioned on demand.
#[tauri::command]
async fn setup_gpu_indexing(
    state: State<'_, AppState>,
    window: WebviewWindow,
) -> Result<(), String> {
    let data_dir = state.data_dir.clone();
    let venv = data_dir.join("venv");
    let venv_py = venv.join("bin").join("python");
    let scripts_dir = data_dir.join("scripts");
    let models_dir = data_dir.join("models");

    std::fs::create_dir_all(&scripts_dir).map_err(|e| e.to_string())?;
    std::fs::write(scripts_dir.join("gpu_embed.py"), GPU_EMBED_PY).map_err(|e| e.to_string())?;
    std::fs::write(scripts_dir.join("export_onnx.py"), EXPORT_ONNX_PY)
        .map_err(|e| e.to_string())?;
    let _ = window.emit("setup-log", "Wrote helper scripts.".to_string());

    // System python3 (present via Command Line Tools) is a fine venv base.
    let base = if Path::new("/usr/bin/python3").exists() {
        "/usr/bin/python3"
    } else {
        "python3"
    };

    let mut c = tokio::process::Command::new(base);
    c.arg("-m").arg("venv").arg(&venv);
    stream_to_log(&window, "Creating virtual environment", c).await?;

    let mut c = tokio::process::Command::new(&venv_py);
    c.arg("-m")
        .arg("pip")
        .arg("install")
        .arg("-U")
        .arg("pip")
        .arg("torch")
        .arg("sentence-transformers")
        .arg("transformers")
        .arg("onnx")
        .arg("pyarrow")
        .arg("pymupdf");
    stream_to_log(
        &window,
        "Installing packages (torch, sentence-transformers, …)",
        c,
    )
    .await?;

    let mut c = tokio::process::Command::new(&venv_py);
    c.arg(scripts_dir.join("export_onnx.py"))
        .arg("--reranker")
        .arg("--out-dir")
        .arg(&models_dir);
    stream_to_log(
        &window,
        "Exporting ONNX models (downloads base models once)",
        c,
    )
    .await?;

    // Point settings at the freshly provisioned venv + scripts + models.
    {
        let mut s = state.settings();
        s.python_bin = venv_py.to_string_lossy().into_owned();
        s.indexer_script = scripts_dir
            .join("gpu_embed.py")
            .to_string_lossy()
            .into_owned();
        s.models_dir = models_dir.to_string_lossy().into_owned();
        s.save(data_dir.join("settings.toml"))
            .map_err(|e| e.to_string())?;
        *state.settings.lock().unwrap() = s;
    }
    let _ = window.emit(
        "setup-log",
        "✓ Setup complete. Restart the app to load the new models.".to_string(),
    );
    Ok(())
}

#[tauri::command]
async fn list_models(state: State<'_, AppState>) -> Result<Vec<String>, String> {
    // Best-effort: some providers (e.g. Fireworks) don't expose `/models` on the
    // inference endpoint. Never fail the call — the UI falls back to the model
    // configured in Settings, and the status check reports reachability.
    Ok(state.llm().list_models().await.unwrap_or_default())
}

/// Preload a model into Ollama so the next `ask` is warm. Called when the user
/// picks a model in the UI; errors are non-fatal (best-effort).
#[tauri::command]
async fn warm_model(state: State<'_, AppState>, model: String) -> Result<(), String> {
    if model.trim().is_empty() {
        return Ok(());
    }
    let _ = state.llm().warm(&model).await;
    Ok(())
}

#[derive(serde::Serialize)]
struct LlmStatus {
    ok: bool,
    message: String,
}

/// Health-check the active synthesis provider so the UI can show readiness.
/// For Ollama: confirm it's reachable and the chosen model is pulled. For
/// Anthropic: confirm a key is configured (a real call would cost tokens).
#[tauri::command]
async fn check_llm(state: State<'_, AppState>, model: String) -> Result<LlmStatus, String> {
    let settings = state.settings();
    let provider = settings.llm_provider.clone();

    // Cloud providers: a key is required first; then a /models probe confirms it.
    if provider != "ollama" {
        if settings.creds(&provider).api_key.trim().is_empty() {
            return Ok(LlmStatus {
                ok: false,
                message: format!("No {provider} API key — add one in Settings"),
            });
        }
        if provider == "anthropic" {
            // No cheap unauthenticated probe; trust the key is present.
            return Ok(LlmStatus {
                ok: true,
                message: "Anthropic key set".into(),
            });
        }
        // `/models` is best-effort (Fireworks 500s on it); a failure here doesn't
        // mean generation won't work, since the configured model is used directly.
        return Ok(match state.llm().list_models().await {
            Ok(models) if !models.is_empty() => LlmStatus {
                ok: true,
                message: format!("{provider} reachable · {} model(s)", models.len()),
            },
            _ => LlmStatus {
                ok: true,
                message: format!("{provider}: key set — using your configured model"),
            },
        });
    }

    // Local Ollama.
    match state.llm().list_models().await {
        Ok(models) => {
            let model = model.trim();
            if model.is_empty() || models.iter().any(|m| m == model) {
                Ok(LlmStatus {
                    ok: true,
                    message: format!("Ollama up · {} model(s)", models.len()),
                })
            } else {
                Ok(LlmStatus {
                    ok: false,
                    message: format!("'{model}' not pulled — run `ollama pull {model}`"),
                })
            }
        }
        Err(e) => Ok(LlmStatus {
            ok: false,
            message: format!("Ollama unreachable — is it running? ({e})"),
        }),
    }
}

/// Current persisted settings (for the Settings UI).
#[tauri::command]
async fn get_settings(state: State<'_, AppState>) -> Result<ls_app::Settings, String> {
    Ok(state.settings())
}

/// Whether a cited source file still exists on disk. Used by the reader to warn
/// when a book has been moved/renamed since it was indexed.
#[tauri::command]
async fn source_exists(path: String) -> Result<bool, String> {
    Ok(std::path::Path::new(&path).is_file())
}

/// Persist settings, update them in memory, and rebuild the Ollama client if the
/// host changed.
#[tauri::command]
async fn save_settings(
    state: State<'_, AppState>,
    settings: ls_app::Settings,
) -> Result<(), String> {
    settings
        .save(state.data_dir.join("settings.toml"))
        .map_err(|e| e.to_string())?;
    *state.settings.lock().unwrap() = settings.clone();
    // Rebuild the client so provider/host/key changes take effect immediately.
    *state.llm.lock().unwrap() = build_llm(&settings);
    Ok(())
}

// ---- conversations ----

#[tauri::command]
async fn list_conversations(state: State<'_, AppState>) -> Result<Vec<Conversation>, String> {
    state.db()?.list_conversations().map_err(|e| e.to_string())
}

#[tauri::command]
async fn create_conversation(
    state: State<'_, AppState>,
    collection_ids: Vec<String>,
    title: String,
) -> Result<Conversation, String> {
    let title = title.trim();
    let conv = Conversation {
        id: new_id(),
        title: if title.is_empty() {
            "New conversation".into()
        } else {
            title.chars().take(80).collect()
        },
        collection_ids,
    };
    state
        .db()?
        .create_conversation(&conv)
        .map_err(|e| e.to_string())?;
    Ok(conv)
}

#[tauri::command]
async fn list_messages(
    state: State<'_, AppState>,
    conversation_id: String,
) -> Result<Vec<Message>, String> {
    state
        .db()?
        .list_messages(&conversation_id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn rename_conversation(
    state: State<'_, AppState>,
    conversation_id: String,
    title: String,
) -> Result<(), String> {
    let title = title.trim();
    if title.is_empty() {
        return Err("title cannot be empty".into());
    }
    state
        .db()?
        .rename_conversation(
            &conversation_id,
            &title.chars().take(80).collect::<String>(),
        )
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn delete_conversation(
    state: State<'_, AppState>,
    conversation_id: String,
) -> Result<(), String> {
    state
        .db()?
        .delete_conversation(&conversation_id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn ask(
    state: State<'_, AppState>,
    window: WebviewWindow,
    collection_ids: Vec<String>,
    conversation_id: String,
    question: String,
    model: String,
) -> Result<Vec<SearchResult>, String> {
    let db = state.db()?;
    let all_colls = db.list_collections().map_err(|e| e.to_string())?;
    let colls: Vec<Collection> = all_colls
        .into_iter()
        .filter(|c| collection_ids.contains(&c.id))
        .collect();
    if colls.is_empty() {
        return Err("no valid collection selected".into());
    }

    // Prior turns for follow-up context (before we add the new question).
    let history: Vec<HistoryTurn> = db
        .list_messages(&conversation_id)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|m| HistoryTurn {
            role: m.role.as_str().to_string(),
            content: m.content,
        })
        .collect();

    // Persist the user's turn immediately.
    db.add_message(&Message {
        id: new_id(),
        conversation_id: conversation_id.clone(),
        role: Role::User,
        content: question.clone(),
        citations: vec![],
        in_tokens: 0,
        out_tokens: 0,
    })
    .map_err(|e| e.to_string())?;

    // Lazily load the engine on first ask (kept resident afterwards).
    let mut guard = state.engine.lock().await;
    if guard.is_none() {
        let embedder =
            Embedder::load(state.models_dir.join("bge-m3")).map_err(|e| e.to_string())?;
        let reranker =
            Reranker::load(reranker_dir(&state.models_dir)).map_err(|e| e.to_string())?;
        *guard = Some(Engine { embedder, reranker });
    }
    let engine = guard.as_mut().unwrap();

    let settings = state.settings();
    let mut stores = Vec::with_capacity(colls.len());
    for c in &colls {
        stores.push(
            Store::open(&c.db_path, "chunks")
                .await
                .map_err(|e| e.to_string())?,
        );
    }
    let store_refs: Vec<&Store> = stores.iter().collect();
    let mut results = search_multi(
        &store_refs,
        &mut engine.embedder,
        &mut engine.reranker,
        &question,
        settings.final_top_k,
        settings.hybrid_top_k,
    )
    .await
    .map_err(|e| e.to_string())?;

    // Drop weak matches so a query with no real answer doesn't list irrelevant
    // sources; renumber so the [n] citation labels stay contiguous.
    results.retain(|r| r.score >= settings.min_relevance);
    for (i, r) in results.iter_mut().enumerate() {
        r.rank = i + 1;
    }

    if results.is_empty() {
        let msg = "I couldn't find any matching passages in the selected collection(s).";
        db.add_message(&Message {
            id: new_id(),
            conversation_id: conversation_id.clone(),
            role: Role::Assistant,
            content: msg.into(),
            citations: vec![],
            in_tokens: 0,
            out_tokens: 0,
        })
        .map_err(|e| e.to_string())?;
        let _ = window.emit("ask-token", msg.to_string());
        let _ = window.emit("ask-done", ());
        return Ok(results);
    }

    let model = if model.trim().is_empty() {
        settings.default_model()
    } else {
        model
    };
    let prompt = build_prompt_with_history(&question, &results, &history);
    let w = window.clone();
    let wr = window.clone();
    let (answer, usage) = state
        .llm()
        .generate_stream(
            &model,
            &prompt,
            |tok| {
                let _ = w.emit("ask-token", tok.to_string());
            },
            |think| {
                let _ = wr.emit("ask-reasoning", think.to_string());
            },
        )
        .await
        .map_err(|e| e.to_string())?;

    // Persist the assistant turn with its grounding citations + token usage.
    db.add_message(&Message {
        id: new_id(),
        conversation_id: conversation_id.clone(),
        role: Role::Assistant,
        content: answer,
        citations: results.iter().map(to_citation).collect(),
        in_tokens: usage.in_tokens,
        out_tokens: usage.out_tokens,
    })
    .map_err(|e| e.to_string())?;

    let _ = window.emit("ask-usage", usage);
    let _ = window.emit("ask-done", ());
    Ok(results)
}

/// Render the given answer + citations to a Markdown artifact and write it to the
/// configured artifacts directory. Returns the absolute path written.
#[tauri::command]
async fn save_artifact(
    state: State<'_, AppState>,
    collection_ids: Vec<String>,
    question: String,
    answer: String,
    model: String,
    created: String,
    sources: Vec<Source>,
) -> Result<String, String> {
    let names: Vec<String> = state
        .db()?
        .list_collections()
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|c| collection_ids.contains(&c.id))
        .map(|c| c.name)
        .collect();
    let collection = if names.is_empty() {
        "Library".to_string()
    } else {
        names.join(", ")
    };

    let settings = state.settings();
    let model = if model.trim().is_empty() {
        settings.default_model()
    } else {
        model
    };

    // Resolve the artifacts dir: absolute as-is, else under the app data dir.
    let configured = Path::new(&settings.artifacts_dir);
    let dir = if configured.is_absolute() {
        configured.to_path_buf()
    } else {
        state.data_dir.join(configured)
    };

    let doc = ArtifactDoc {
        question,
        answer,
        model,
        collection,
        created,
        sources,
    };
    let path = ls_artifacts::write_artifact(&Markdown as &dyn ArtifactRenderer, &doc, &dir)
        .map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

fn init_state() -> AppState {
    // Load embedding models from the local HF cache only (no network at runtime).
    std::env::set_var("HF_HUB_OFFLINE", "1");
    std::env::set_var("TRANSFORMERS_OFFLINE", "1");

    let data_dir = ls_app::data_dir();
    let _ = std::fs::create_dir_all(&data_dir);
    // Prefer models provisioned by the in-app setup (app-data/models); fall back
    // to LS_MODELS_DIR or the dev repo's models/.
    let models_dir = std::env::var("LS_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let app_models = data_dir.join("models");
            if app_models.join("bge-m3").join("model.onnx").exists() {
                app_models
            } else {
                PathBuf::from(format!("{}/../models", env!("CARGO_MANIFEST_DIR")))
            }
        });
    let settings = ls_app::Settings::load(data_dir.join("settings.toml")).unwrap_or_default();

    // Seed a default collection pointing at the dev index if none exists yet.
    if let Ok(db) = Db::open(data_dir.join("app.db")) {
        if db.list_collections().map(|c| c.is_empty()).unwrap_or(false) {
            let _ = db.upsert_collection(&Collection {
                id: "default".into(),
                name: "My Library".into(),
                db_path: data_dir.join("lancedb").to_string_lossy().into_owned(),
                source_paths: vec![],
                embed_model: "bge-m3".into(),
            });
        }
    }

    let llm = build_llm(&settings);
    AppState {
        data_dir,
        models_dir,
        settings: std::sync::Mutex::new(settings),
        llm: std::sync::Mutex::new(llm),
        engine: Mutex::new(None),
        cancel: Arc::new(AtomicBool::new(false)),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            app.manage(init_state());

            // Pre-load the embedder/reranker in the background so the first ask
            // doesn't pay the ONNX load cost.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let models = handle.state::<AppState>().models_dir.clone();
                let loaded = tauri::async_runtime::spawn_blocking(move || {
                    let e = Embedder::load(models.join("bge-m3")).ok()?;
                    let r = Reranker::load(reranker_dir(&models)).ok()?;
                    Some(Engine {
                        embedder: e,
                        reranker: r,
                    })
                })
                .await
                .ok()
                .flatten();
                if let Some(engine) = loaded {
                    let state = handle.state::<AppState>();
                    let mut guard = state.engine.lock().await;
                    if guard.is_none() {
                        *guard = Some(engine);
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_collections,
            create_collection,
            set_collection_paths,
            delete_collection,
            set_provider,
            index_collection,
            fast_index_collection,
            cancel_indexing,
            setup_gpu_indexing,
            list_conversations,
            create_conversation,
            list_messages,
            rename_conversation,
            delete_conversation,
            list_models,
            warm_model,
            check_llm,
            get_settings,
            save_settings,
            source_exists,
            ask,
            save_artifact
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
