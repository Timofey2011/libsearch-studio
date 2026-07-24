//! §17.1 scan surface: scan_chunk_meta projects metadata only (chapter/page
//! null conventions round-trip) and chunk_texts fetches exactly the requested
//! id set — the two-pass shape the cite-metric sampler depends on.

use ls_core::{Chunk, Format};
use ls_index::Store;

fn chunk(id: &str, book_id: &str, chapter: Option<&str>, page: Option<u32>, text: &str) -> Chunk {
    Chunk {
        id: id.into(),
        book_id: book_id.into(),
        title: format!("{book_id} title"),
        author: None,
        source_path: format!("/lib/{book_id}.epub"),
        format: Format::Epub,
        chapter: chapter.map(Into::into),
        page,
        loc_start: 0,
        loc_end: text.len(),
        text: text.into(),
        vector: Some(vec![0.01; 1024]),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scan_meta_and_fetch_texts_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open_or_create(dir.path().to_str().unwrap(), "chunks")
        .await
        .unwrap();
    store
        .add_chunks(&[
            chunk("a:0", "a", Some("Intro"), None, "alpha text"),
            chunk("a:1", "a", None, Some(7), "beta text"),
            chunk("b:0", "b", Some("Глава 1"), None, "gamma text"),
        ])
        .await
        .unwrap();

    let mut meta = store.scan_chunk_meta().await.unwrap();
    meta.sort_by(|x, y| x.id.cmp(&y.id));
    assert_eq!(meta.len(), 3);
    assert_eq!(meta[0].chapter.as_deref(), Some("Intro"));
    assert_eq!(meta[0].page, None);
    assert_eq!(meta[1].chapter, None, "empty chapter reads back as None");
    assert_eq!(meta[1].page, Some(7));
    assert_eq!(meta[2].chapter.as_deref(), Some("Глава 1"));
    assert_eq!(meta[2].format, "epub");
    assert_eq!(meta[2].source_path, "/lib/b.epub");

    // Text fetch: exactly the requested ids, nothing more.
    let texts = store
        .chunk_texts(&["a:1".to_string(), "b:0".to_string()])
        .await
        .unwrap();
    assert_eq!(texts.len(), 2);
    assert_eq!(texts.get("a:1").map(String::as_str), Some("beta text"));
    assert_eq!(texts.get("b:0").map(String::as_str), Some("gamma text"));
    assert!(!texts.contains_key("a:0"));
}
