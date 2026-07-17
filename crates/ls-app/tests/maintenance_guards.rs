//! §2.6 Maintenance acceptance: all four scans + apply against a real temp
//! Lance store, temp SQLite, and a temp file tree — then a second scan proves
//! the fixes are complete and idempotent.

use std::path::Path;

use ls_app::maintenance;
use ls_app::types::Collection;
use ls_app::Db;
use ls_core::{Chunk, Format};
use ls_index::Store;

fn chunk(id: &str, book_id: &str, path: &str, format: Format, text: &str) -> Chunk {
    Chunk {
        id: id.into(),
        book_id: book_id.into(),
        title: Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
        author: None,
        source_path: path.into(),
        format,
        chapter: None,
        page: None,
        loc_start: 0,
        loc_end: text.len(),
        text: text.into(),
        vector: Some(vec![0.01; 1024]),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn maintenance_scans_and_fixes_all_four_classes() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("books");
    std::fs::create_dir_all(&root).unwrap();
    let p = |name: &str| root.join(name).to_string_lossy().into_owned();

    // On-disk truth: alive.pdf, twin.pdf + twin.docx (variant pair), solo.md.
    for f in ["alive.pdf", "twin.pdf", "twin.docx", "solo.md"] {
        std::fs::write(root.join(f), b"content").unwrap();
    }
    // gone.pdf and ghost.pdf are indexed/manifested but never created on disk.

    let store_dir = dir.path().join("lancedb");
    let store = Store::open_or_create(store_dir.to_str().unwrap(), "chunks")
        .await
        .unwrap();
    store
        .add_chunks(&[
            // Same-path multi-id: alive.pdf under two id schemes.
            chunk(
                "c1",
                "legacy-alive",
                &p("alive.pdf"),
                Format::Pdf,
                "alpha one",
            ),
            chunk(
                "c2",
                "stable-alive",
                &p("alive.pdf"),
                Format::Pdf,
                "alpha two",
            ),
            // Store orphan: file gone.
            chunk("c3", "gone-1", &p("gone.pdf"), Format::Pdf, "beta"),
            // Duplicate variants: pdf keeps, docx loses.
            chunk(
                "c4",
                "twin-pdf-id",
                &p("twin.pdf"),
                Format::Pdf,
                "gamma pdf",
            ),
            chunk(
                "c5",
                "twin-docx-id",
                &p("twin.docx"),
                Format::Docx,
                "gamma docx",
            ),
            // Bad stamp: an .md file stamped pdf (the imported-library debt).
            chunk("c6", "solo-md", &p("solo.md"), Format::Pdf, "delta"),
        ])
        .await
        .unwrap();

    let db = Db::open(dir.path().join("app.db")).unwrap();
    let coll = Collection {
        id: "c".into(),
        name: "Test".into(),
        db_path: store_dir.to_string_lossy().into_owned(),
        source_paths: vec![
            root.to_string_lossy().into_owned(),
            // An unreachable root: rows under it are unjudgeable, never orphans.
            dir.path().join("unmounted").to_string_lossy().into_owned(),
        ],
        embed_model: "bge-m3".into(),
    };
    // Manifest: keeper preference for alive.pdf; a row for the store orphan;
    // a manifest-only orphan (ghost.pdf, no store rows, file missing); and a
    // pre-M0a empty-path row.
    db.set_book_state(&coll.id, "legacy-alive", "1:1", "sig-a", &p("alive.pdf"))
        .unwrap();
    db.set_book_state(&coll.id, "gone-1", "1:1", "sig-g", &p("gone.pdf"))
        .unwrap();
    db.set_book_state(&coll.id, "ghost-1", "1:1", "sig-h", &p("ghost.pdf"))
        .unwrap();

    let report = maintenance::scan(&store, &db, &coll).await.unwrap();

    // Orphans: gone.pdf (store) + ghost.pdf (manifest). Nothing else.
    assert_eq!(report.orphans.len(), 2, "{:?}", report.orphans);
    assert!(report
        .orphans
        .iter()
        .any(|o| o.path.ends_with("gone.pdf") && o.kind == "store"));
    assert!(report
        .orphans
        .iter()
        .any(|o| o.path.ends_with("ghost.pdf") && o.kind == "manifest"));

    // Stamps: only solo.md (pdf → md); correct stamps untouched.
    assert_eq!(report.bad_stamps.len(), 1, "{:?}", report.bad_stamps);
    assert_eq!(report.bad_stamps[0].from, "pdf");
    assert_eq!(report.bad_stamps[0].to, "md");

    // Dup variants: twin.pdf keeps, twin.docx removed.
    assert_eq!(report.dup_variants.len(), 1, "{:?}", report.dup_variants);
    assert!(report.dup_variants[0].keep_path.ends_with("twin.pdf"));
    assert_eq!(report.dup_variants[0].remove.len(), 1);
    assert!(report.dup_variants[0].remove[0].path.ends_with("twin.docx"));

    // Multi-id: alive.pdf keeps the MANIFEST's id.
    assert_eq!(report.multi_id.len(), 1, "{:?}", report.multi_id);
    assert_eq!(report.multi_id[0].keep_id, "legacy-alive");
    assert_eq!(
        report.multi_id[0].remove_ids,
        vec!["stable-alive".to_string()]
    );

    // The unmounted root is reported, not silently ignored.
    assert_eq!(report.unreachable_roots.len(), 1);

    // Apply everything.
    let out = maintenance::apply(&store, &db, &coll, true, true, true, true)
        .await
        .unwrap();
    assert!(out.orphans_removed >= 2, "{out:?}");
    assert_eq!(out.restamped, 1, "{out:?}");
    assert_eq!(out.dup_rows_removed, 1, "{out:?}");
    assert_eq!(out.multi_id_removed, 1, "{out:?}");

    // Store truth after: keepers only, stamp repaired.
    let ids = store.indexed_book_ids().await.unwrap();
    assert!(ids.contains("legacy-alive"));
    assert!(ids.contains("twin-pdf-id"));
    assert!(ids.contains("solo-md"));
    assert!(!ids.contains("stable-alive"));
    assert!(!ids.contains("gone-1"));
    assert!(!ids.contains("twin-docx-id"));
    let formats = store.book_formats().await.unwrap();
    let solo = formats.iter().find(|(b, _, _)| b == "solo-md").unwrap();
    assert_eq!(solo.2, "md");

    // Manifest truth after: keeper row survives; dead rows gone.
    assert!(db
        .book_id_for_source_path(&coll.id, &p("alive.pdf"))
        .unwrap()
        .is_some());
    let rows = db.book_state_rows(&coll.id).unwrap();
    assert!(!rows.iter().any(|(id, _)| id == "gone-1" || id == "ghost-1"));

    // Idempotence: a fresh scan is clean (unreachable root note remains).
    let again = maintenance::scan(&store, &db, &coll).await.unwrap();
    assert!(again.orphans.is_empty(), "{:?}", again.orphans);
    assert!(again.bad_stamps.is_empty(), "{:?}", again.bad_stamps);
    assert!(again.dup_variants.is_empty(), "{:?}", again.dup_variants);
    assert!(again.multi_id.is_empty(), "{:?}", again.multi_id);
}
