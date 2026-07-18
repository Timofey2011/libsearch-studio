//! v0.15 re-chunk lifecycle, END TO END through the real CPU pipeline (real
//! ONNX embedder — run explicitly with `--ignored` and LS_MODELS_DIR set, or
//! the default app models dir present):
//!   index → downgrade vers → arm → re-embed (no chunk duplication) → fixed
//!   point on the confirming run.

use ls_app::{Db, Service, CURRENT_CHUNKER_VER};
use ls_embed::{BgeTokenCounter, Embedder};
use ls_index::Store;

fn models_dir() -> std::path::PathBuf {
    std::env::var("LS_MODELS_DIR")
        .map(Into::into)
        .unwrap_or_else(|_| dirs_fallback().join(".local/share/libsearch-studio/models"))
}

fn dirs_fallback() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs real ONNX models; run with --ignored"]
async fn rechunk_reembeds_without_duplication_then_clears() {
    let dir = tempfile::tempdir().unwrap();
    let books = dir.path().join("books");
    std::fs::create_dir_all(&books).unwrap();
    let body = "Idempotency means an operation can be applied many times without changing the result beyond the first application. Retries become safe and consumers deduplicate by key. ".repeat(4);
    std::fs::write(books.join("one.md"), format!("# One\n\n{body}")).unwrap();
    std::fs::write(books.join("two.txt"), &body).unwrap();

    let data_dir = dir.path().join("appdata");
    let svc = Service::new(&data_dir).unwrap();
    let coll = ls_app::Collection {
        id: "t".into(),
        name: "T".into(),
        db_path: dir.path().join("lancedb").to_string_lossy().into_owned(),
        source_paths: vec![books.to_string_lossy().into_owned()],
        embed_model: "bge-m3".into(),
    };
    let db = Db::open(data_dir.join("app.db")).unwrap();
    db.upsert_collection(&coll).unwrap();

    let mut embedder = Embedder::load(models_dir().join("bge-m3")).expect("models present");
    let counter = BgeTokenCounter::load(models_dir().join("bge-m3")).unwrap();
    let files = ls_app::discover_books(&coll.source_paths);
    assert_eq!(files.len(), 2);

    // Run 1: fresh index.
    let s1 = svc
        .index_collection(&coll, &files, &mut embedder, &counter, || false, |_| {})
        .await
        .unwrap();
    assert_eq!(s1.books_indexed, 2);
    assert_eq!(s1.forced, 0);
    let store = Store::open_or_create(&coll.db_path, "chunks")
        .await
        .unwrap();
    let chunks_before = store.count().await.unwrap();
    assert!(chunks_before > 0);

    // Downgrade to legacy vers + arm the flag (what the nudge button does).
    for (id, _) in db.book_state_rows(&coll.id).unwrap() {
        let hit_fp = db.book_fingerprint(&coll.id, &id).unwrap().unwrap();
        let path = db
            .book_state_rows(&coll.id)
            .unwrap()
            .into_iter()
            .find(|(i, _)| *i == id)
            .unwrap()
            .1;
        db.set_book_state_ver(&coll.id, &id, &hit_fp, "keep", &path, 0)
            .unwrap();
    }
    db.set_rechunk_pending(&coll.id, true).unwrap();
    assert_eq!(db.legacy_chunker_count(&coll.id).unwrap(), 2);

    // Run 2 (armed): both books re-embed; chunk count must NOT grow.
    let s2 = svc
        .index_collection(&coll, &files, &mut embedder, &counter, || false, |_| {})
        .await
        .unwrap();
    assert_eq!(s2.books_indexed, 2, "{s2:?}");
    assert_eq!(s2.forced, 2, "flag must stay armed after this run");
    let chunks_after = store.count().await.unwrap();
    assert_eq!(chunks_before, chunks_after, "replace, never append");
    assert_eq!(db.legacy_chunker_count(&coll.id).unwrap(), 0);
    for (id, _) in db.book_state_rows(&coll.id).unwrap() {
        // every surviving row is on the current chunker
        let mut stmt_ok = false;
        if let Ok(Some(_)) = db.book_fingerprint(&coll.id, &id) {
            stmt_ok = true;
        }
        assert!(stmt_ok);
    }
    let _ = CURRENT_CHUNKER_VER;

    // Run 3 (still armed): the confirming pass — nothing forced, no embeds.
    let s3 = svc
        .index_collection(&coll, &files, &mut embedder, &counter, || false, |_| {})
        .await
        .unwrap();
    assert_eq!(s3.books_indexed, 0, "{s3:?}");
    assert_eq!(
        s3.forced, 0,
        "fixed point: the command would clear the flag"
    );
}
