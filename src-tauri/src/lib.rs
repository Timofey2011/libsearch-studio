//! Tauri bridge: thin command/event layer over the engine crates.
//!
//! Holds the expensive engine handles (Embedder/Reranker) in app state behind a
//! Tokio mutex; collection metadata lives in SQLite (opened per command — cheap).
//! `ask` streams answer tokens to the webview via the `ask-token` event.

use std::path::{Path, PathBuf};

use ls_app::{Collection, Db};
use ls_embed::{Embedder, Reranker};
use ls_index::Store;
use ls_llm::{build_prompt, OllamaClient};
use ls_query::{search, SearchResult};
use tauri::{Emitter, Manager, State, WebviewWindow};
use tokio::sync::Mutex;

struct Engine {
    embedder: Embedder,
    reranker: Reranker,
}

struct AppState {
    data_dir: PathBuf,
    models_dir: PathBuf,
    settings: ls_app::Settings,
    llm: OllamaClient,
    engine: Mutex<Option<Engine>>,
}

impl AppState {
    fn db(&self) -> Result<Db, String> {
        Db::open(self.data_dir.join("app.db")).map_err(|e| e.to_string())
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

#[tauri::command]
async fn list_models(state: State<'_, AppState>) -> Result<Vec<String>, String> {
    state.llm.list_models().await.map_err(|e| e.to_string())
}

/// Preload a model into Ollama so the next `ask` is warm. Called when the user
/// picks a model in the UI; errors are non-fatal (best-effort).
#[tauri::command]
async fn warm_model(state: State<'_, AppState>, model: String) -> Result<(), String> {
    if model.trim().is_empty() {
        return Ok(());
    }
    let _ = state.llm.warm(&model).await;
    Ok(())
}

#[tauri::command]
async fn ask(
    state: State<'_, AppState>,
    window: WebviewWindow,
    collection_id: String,
    question: String,
    model: String,
) -> Result<Vec<SearchResult>, String> {
    let coll = state
        .db()?
        .list_collections()
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|c| c.id == collection_id)
        .ok_or_else(|| format!("collection {collection_id} not found"))?;

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

    let store = Store::open(&coll.db_path, "chunks")
        .await
        .map_err(|e| e.to_string())?;
    let results = search(
        &store,
        &mut engine.embedder,
        &mut engine.reranker,
        &question,
        state.settings.final_top_k,
        state.settings.hybrid_top_k,
    )
    .await
    .map_err(|e| e.to_string())?;

    if results.is_empty() {
        let _ = window.emit("ask-done", ());
        return Ok(results);
    }

    let model = if model.trim().is_empty() {
        state.settings.ollama_model.clone()
    } else {
        model
    };
    let prompt = build_prompt(&question, &results);
    let w = window.clone();
    state
        .llm
        .generate_stream(&model, &prompt, |tok| {
            let _ = w.emit("ask-token", tok.to_string());
        })
        .await
        .map_err(|e| e.to_string())?;
    let _ = window.emit("ask-done", ());
    Ok(results)
}

fn init_state() -> AppState {
    // Load embedding models from the local HF cache only (no network at runtime).
    std::env::set_var("HF_HUB_OFFLINE", "1");
    std::env::set_var("TRANSFORMERS_OFFLINE", "1");

    let data_dir = ls_app::data_dir();
    let _ = std::fs::create_dir_all(&data_dir);
    let models_dir = std::env::var("LS_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(format!("{}/../models", env!("CARGO_MANIFEST_DIR"))));
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

    let llm = OllamaClient::new(&settings.ollama_host);
    AppState {
        data_dir,
        models_dir,
        settings,
        llm,
        engine: Mutex::new(None),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
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
            list_models,
            warm_model,
            ask
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
