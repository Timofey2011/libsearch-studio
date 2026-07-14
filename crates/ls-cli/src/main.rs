//! Dev CLI for the LibSearch Studio engine — proves the pure-Rust pipeline
//! end-to-end (extract → chunk → embed → store → hybrid → rerank → cite) before
//! the Tauri UI exists.
//!
//!   cargo run -p ls-cli -- ingest book1.pdf book2.pdf
//!   cargo run -p ls-cli -- search "how do event-driven microservices communicate"

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use ls_embed::{BgeTokenCounter, Embedder, Reranker};
use ls_index::{chunk_book, ChunkParams, Store};
use ls_llm::{build_prompt, OllamaClient};
use ls_query::search;

const EMBED_BATCH: usize = 64;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("search") => {
            let query = args.collect::<Vec<_>>().join(" ");
            if query.trim().is_empty() {
                bail!("usage: ls-cli search <query>");
            }
            run_search(&query).await
        }
        Some("ingest") => {
            let paths: Vec<String> = args.collect();
            if paths.is_empty() {
                bail!("usage: ls-cli ingest <file.pdf> [more.pdf ...]");
            }
            run_ingest(&paths).await
        }
        Some("ask") => {
            let question = args.collect::<Vec<_>>().join(" ");
            if question.trim().is_empty() {
                bail!("usage: ls-cli ask <question>");
            }
            run_ask(&question).await
        }
        Some("import") => {
            let parquet = args.next().unwrap_or_default();
            if parquet.is_empty() {
                bail!("usage: ls-cli import <file.parquet>");
            }
            run_import(&parquet).await
        }
        Some("backfill-state") => {
            let app_dir = args.next().unwrap_or_default();
            if app_dir.is_empty() {
                bail!("usage: ls-cli backfill-state <app-data-dir> [collection_id]");
            }
            let coll = args.next().unwrap_or_else(|| "default".into());
            run_backfill_state(&app_dir, &coll).await
        }
        Some("gen-exts") => {
            // Regenerate the frontend's extension map from the ls-core
            // canonical list; a freshness test keeps the copy honest.
            let out = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../frontend/src/generated/supportedExts.ts");
            std::fs::create_dir_all(out.parent().unwrap()).context("mkdir generated/")?;
            std::fs::write(&out, ls_core::gen_supported_exts_ts()).context("write ts")?;
            eprintln!("wrote {}", out.display());
            Ok(())
        }
        _ => bail!("usage: ls-cli <search|ingest|import|backfill-state|gen-exts|ask> ..."),
    }
}

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("LS_MODELS_DIR").unwrap_or_else(|_| "models".into()))
}

fn db_path() -> String {
    std::env::var("LS_DB_PATH").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.local/share/libsearch-studio/lancedb")
    })
}

async fn run_ingest(paths: &[String]) -> Result<()> {
    let models = models_dir();
    let db = db_path();
    eprintln!("index: {db}");

    let store = Store::open_or_create(&db, "chunks")
        .await
        .context("open/create index")?;
    let mut embedder = Embedder::load(models.join("bge-m3")).context("load embedder")?;
    let counter = BgeTokenCounter::load(models.join("bge-m3")).context("load tokenizer")?;
    let params = ChunkParams::default();

    let mut total = 0usize;
    for (n, path) in paths.iter().enumerate() {
        let doc = match ls_extract::extract(Path::new(path)) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[{}/{}] FAILED {path}: {e}", n + 1, paths.len());
                continue;
            }
        };
        if doc.blocks.is_empty() {
            eprintln!("[{}/{}] skip (no text) {path}", n + 1, paths.len());
            continue;
        }
        let mut chunks = chunk_book(&doc, &counter, &params);

        // Embed in batches and attach vectors.
        for batch in chunks.chunks_mut(EMBED_BATCH) {
            let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
            let vectors = embedder.embed(&texts).context("embed")?;
            for (c, v) in batch.iter_mut().zip(vectors) {
                c.vector = Some(v);
            }
        }

        store.delete_book(&doc.book_id).await.ok();
        total += store.add_chunks(&chunks).await.context("add chunks")?;
        eprintln!(
            "[{}/{}] indexed {} ({} chunks)",
            n + 1,
            paths.len(),
            doc.title,
            chunks.len()
        );
    }

    if total > 0 {
        eprintln!("building FTS index over {total} new chunks …");
        store.ensure_fts_index().await.context("fts index")?;
    }
    eprintln!(
        "done: {total} chunks; index now has {} rows",
        store.count().await?
    );
    Ok(())
}

async fn run_import(parquet: &str) -> Result<()> {
    let db = db_path();
    eprintln!("importing {parquet} -> {db}");
    let store = Store::open_or_create(&db, "chunks")
        .await
        .context("open/create index")?;
    let n = store
        .import_parquet(parquet)
        .await
        .context("import parquet")?;
    if n > 0 {
        eprintln!("building FTS index over {n} chunks …");
        store.ensure_fts_index().await.context("fts index")?;
    }
    eprintln!(
        "imported {n} chunks; index now has {} rows",
        store.count().await?
    );
    Ok(())
}

/// Backfill the fingerprint manifest (`book_state`) for a collection's index, so
/// books loaded via `import` are recognized by the dedup and not re-embedded on a
/// later re-index. For each `(book_id, source_path)` in the index it records the
/// file fingerprint + content signature (computed from the file on disk).
async fn run_backfill_state(app_dir: &str, collection_id: &str) -> Result<()> {
    let app_dir = PathBuf::from(app_dir);
    let db = ls_app::Db::open(app_dir.join("app.db")).context("open app.db")?;
    let coll = db
        .list_collections()
        .context("list collections")?
        .into_iter()
        .find(|c| c.id == collection_id)
        .with_context(|| format!("collection {collection_id} not found"))?;
    let store = Store::open(&coll.db_path, "chunks")
        .await
        .context("open index")?;
    let pairs = store.book_paths().await.context("read book paths")?;
    eprintln!(
        "backfilling {} books for collection '{}'",
        pairs.len(),
        coll.name
    );
    let out =
        ls_app::service::backfill_book_state(&db, &coll.id, &pairs).context("write book_state")?;
    eprintln!(
        "done: {} recorded ({} unreadable — NOT seeded; re-run once they are readable)",
        out.seeded, out.unseedable
    );
    for p in &out.unseedable_paths {
        eprintln!("  unseedable: {p}");
    }
    Ok(())
}

async fn run_ask(question: &str) -> Result<()> {
    let models = models_dir();
    let db = db_path();
    let host = std::env::var("LS_OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());
    let model = std::env::var("LS_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:12b-mlx".into());

    let store = Store::open(&db, "chunks").await.context("open index")?;
    let mut embedder = Embedder::load(models.join("bge-m3")).context("load embedder")?;
    let mut reranker =
        Reranker::load(models.join("bge-reranker-v2-m3")).context("load reranker")?;

    let results = search(&store, &mut embedder, &mut reranker, question, 8, 50).await?;
    if results.is_empty() {
        println!("(no matching passages)");
        return Ok(());
    }

    let prompt = build_prompt(question, &results);
    eprintln!("synthesizing with {model} …\n");
    let client = OllamaClient::new(&host);
    use std::io::Write;
    client
        .generate_stream(
            &model,
            &prompt,
            |tok| {
                print!("{tok}");
                let _ = std::io::stdout().flush();
            },
            |think| {
                // Reasoning goes to stderr so stdout stays the clean answer.
                eprint!("\x1b[2m{think}\x1b[0m");
                let _ = std::io::stderr().flush();
            },
        )
        .await
        .context("ollama generate")?;

    println!("\n\nSources:");
    for r in &results {
        println!("  [{}] {}", r.rank, r.citation);
    }
    Ok(())
}

async fn run_search(query: &str) -> Result<()> {
    let models = models_dir();
    let db = db_path();

    eprintln!("opening index at {db} …");
    let store = Store::open(&db, "chunks").await.context("open index")?;
    eprintln!("index: {} chunks", store.count().await?);

    let mut embedder = Embedder::load(models.join("bge-m3")).context("load embedder")?;
    let mut reranker =
        Reranker::load(models.join("bge-reranker-v2-m3")).context("load reranker")?;

    let results = search(&store, &mut embedder, &mut reranker, query, 10, 50).await?;
    if results.is_empty() {
        println!("(no matching passages)");
        return Ok(());
    }
    for r in &results {
        println!("[{}] {:.3}  {}", r.rank, r.score, r.citation);
        let preview: String = r
            .text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(160)
            .collect();
        println!("     {preview}");
    }
    Ok(())
}
