//! Expand a collection's configured source paths into a concrete list of book
//! files to index. A source path may be a single file or a directory (walked
//! recursively). Only formats ingest currently accepts (`ls_core::INGEST_EXTS`,
//! flipped per ROADMAP-3 milestone) are returned; the extension is derived by
//! the one canonical rule, `ls_core::ext_of` (compound extensions included).

use std::collections::BTreeSet;
use std::path::Path;

fn is_supported(path: &Path) -> bool {
    ls_core::ext_of(&path.to_string_lossy())
        .map(|e| ls_core::INGEST_EXTS.contains(&e))
        .unwrap_or(false)
}

/// Recursively collect supported files under `dir` into `out`. I/O errors on a
/// subtree are skipped rather than failing the whole walk.
fn walk(dir: &Path, out: &mut BTreeSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, out);
        } else if is_supported(&path) {
            out.insert(path.to_string_lossy().into_owned());
        }
    }
}

/// Book-format preference for same-stem variants of one book sitting in the
/// same directory (ported from the Python engine's epub>pdf>mobi dedup):
/// lower rank wins; `None` = not a book variant (text formats always kept).
fn variant_rank(path: &str) -> Option<u8> {
    match ls_core::ext_of(path) {
        Some("epub") => Some(0),
        Some("pdf") => Some(1),
        Some("fb2") | Some("fb2.zip") => Some(2),
        Some("mobi") => Some(3),
        Some("azw3") => Some(4),
        _ => None,
    }
}

/// Directory + lowercase stem, the variant-grouping key.
fn variant_key(path: &str) -> Option<(String, String)> {
    let p = Path::new(path);
    let ext = ls_core::ext_of(path)?;
    let name = p.file_name()?.to_string_lossy().to_lowercase();
    let stem = name.strip_suffix(&format!(".{ext}"))?.to_string();
    Some((p.parent()?.to_string_lossy().into_owned(), stem))
}

/// Expand `source_paths` (files and/or directories) into a deduplicated, sorted
/// list of indexable file paths. When one book exists in several ebook formats
/// side by side (same folder, same stem: `book.epub` + `book.pdf` +
/// `book.mobi`), only the preferred format is returned — otherwise every
/// variant would embed as a duplicate retrieval hit.
pub fn discover_books(source_paths: &[String]) -> Vec<String> {
    let mut found = BTreeSet::new();
    for p in source_paths {
        let path = Path::new(p);
        if path.is_dir() {
            walk(path, &mut found);
        } else if path.is_file() && is_supported(path) {
            found.insert(path.to_string_lossy().into_owned());
        }
    }
    // Same-stem variant dedup among book formats only.
    let mut best: std::collections::HashMap<(String, String), (u8, String)> =
        std::collections::HashMap::new();
    let mut passthrough: Vec<String> = Vec::new();
    for f in found {
        match (variant_rank(&f), variant_key(&f)) {
            (Some(rank), Some(key)) => match best.get(&key) {
                Some((r, _)) if *r <= rank => {}
                _ => {
                    best.insert(key, (rank, f));
                }
            },
            _ => passthrough.push(f),
        }
    }
    let mut out: Vec<String> = passthrough;
    out.extend(best.into_values().map(|(_, f)| f));
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_supported_files_recursively_and_dedups() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("a.pdf"), b"x").unwrap();
        std::fs::write(root.join("notes.txt"), b"x").unwrap(); // M1: txt is ingested
        std::fs::write(root.join("archive.tar.gz"), b"x").unwrap(); // unknown stays out
        std::fs::write(root.join("book.epub"), b"x").unwrap(); // M2: epub is ingested
        std::fs::write(root.join("book.docx"), b"x").unwrap(); // M4 format: not yet
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/b.PDF"), b"x").unwrap();
        std::fs::write(root.join("sub/l.MD"), b"x").unwrap();

        // Pass the dir AND one of its files explicitly — result must dedup.
        let paths = vec![
            root.to_string_lossy().into_owned(),
            root.join("a.pdf").to_string_lossy().into_owned(),
        ];
        let found = discover_books(&paths);
        assert_eq!(found.len(), 5, "got {found:?}");
        assert!(found.iter().any(|p| p.ends_with("a.pdf")));
        assert!(found.iter().any(|p| p.ends_with("sub/b.PDF")));
        assert!(found.iter().any(|p| p.ends_with("notes.txt")));
        assert!(found.iter().any(|p| p.ends_with("sub/l.MD")));
        assert!(found.iter().any(|p| p.ends_with("book.epub")));
        assert!(!found.iter().any(|p| p.ends_with("archive.tar.gz")));
        assert!(!found.iter().any(|p| p.ends_with("book.docx")));
    }

    #[test]
    fn same_stem_variants_prefer_epub_over_pdf_over_mobi() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // One book in three formats: only the epub must survive discovery.
        std::fs::write(root.join("guide.epub"), b"x").unwrap();
        std::fs::write(root.join("guide.pdf"), b"x").unwrap();
        std::fs::write(root.join("guide.mobi"), b"x").unwrap();
        // A lone mobi (no better sibling) stays.
        std::fs::write(root.join("only.mobi"), b"x").unwrap();
        // Same stem in a DIFFERENT directory is a different book.
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/guide.pdf"), b"x").unwrap();
        // Text formats are never variant-deduped.
        std::fs::write(root.join("guide.md"), b"x").unwrap();

        let found = discover_books(&[root.to_string_lossy().into_owned()]);
        assert!(found.iter().any(|p| p.ends_with("guide.epub")));
        assert!(!found
            .iter()
            .any(|p| p.ends_with("/guide.pdf") && !p.contains("sub")));
        assert!(!found.iter().any(|p| p.ends_with("guide.mobi")));
        assert!(found.iter().any(|p| p.ends_with("only.mobi")));
        assert!(found.iter().any(|p| p.ends_with("sub/guide.pdf")));
        assert!(found.iter().any(|p| p.ends_with("guide.md")));
        assert_eq!(found.len(), 4, "got {found:?}");
    }

    #[test]
    fn missing_path_yields_nothing() {
        assert!(discover_books(&["/no/such/dir".into()]).is_empty());
    }
}
