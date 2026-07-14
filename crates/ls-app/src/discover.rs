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

/// Expand `source_paths` (files and/or directories) into a deduplicated, sorted
/// list of indexable file paths.
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
    found.into_iter().collect()
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
        std::fs::write(root.join("book.epub"), b"x").unwrap(); // M2 format: not yet
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/b.PDF"), b"x").unwrap();
        std::fs::write(root.join("sub/l.MD"), b"x").unwrap();

        // Pass the dir AND one of its files explicitly — result must dedup.
        let paths = vec![
            root.to_string_lossy().into_owned(),
            root.join("a.pdf").to_string_lossy().into_owned(),
        ];
        let found = discover_books(&paths);
        assert_eq!(found.len(), 4, "got {found:?}");
        assert!(found.iter().any(|p| p.ends_with("a.pdf")));
        assert!(found.iter().any(|p| p.ends_with("sub/b.PDF")));
        assert!(found.iter().any(|p| p.ends_with("notes.txt")));
        assert!(found.iter().any(|p| p.ends_with("sub/l.MD")));
        assert!(!found.iter().any(|p| p.ends_with("archive.tar.gz")));
        assert!(!found.iter().any(|p| p.ends_with("book.epub")));
    }

    #[test]
    fn missing_path_yields_nothing() {
        assert!(discover_books(&["/no/such/dir".into()]).is_empty());
    }
}
