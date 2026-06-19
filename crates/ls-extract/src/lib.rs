//! Text extraction from source files into a normalized [`BookDoc`].
//!
//! v1 supports PDF via the pure-Rust `lopdf` (per-page text → `Block`s with page
//! numbers, so PDF citations carry pages). Text is dehyphenated and whitespace is
//! collapsed (ported from the Python engine's `clean_text`). epub/mobi land later.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::OnceLock;

use ls_core::{Block, BookDoc, Format};
use regex::Regex;

/// Below this many characters of extracted text, treat the book as empty
/// (scanned/image-only PDF) — the caller logs and skips.
pub const MIN_BOOK_CHARS: usize = 200;

#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("pdf: {0}")]
    Pdf(#[from] lopdf::Error),
    #[error("unsupported format for {0}")]
    Unsupported(String),
}

fn hyphen_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(\w)-\n(\w)").unwrap())
}

fn ws_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[ \t]+").unwrap())
}

fn multinl_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\n{3,}").unwrap())
}

/// Dehyphenate line-break splits, collapse runs of whitespace/newlines.
pub fn clean_text(text: &str) -> String {
    let text = hyphen_re().replace_all(text, "$1$2");
    let text = ws_re().replace_all(&text, " ");
    let text = multinl_re().replace_all(&text, "\n\n");
    text.trim().to_string()
}

fn stable_book_id(path: &Path) -> String {
    let mut h = DefaultHasher::new();
    path.to_string_lossy().hash(&mut h);
    format!("{:016x}", h.finish())
}

fn title_from_path(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "Untitled".to_string())
}

/// Extract a PDF into a `BookDoc`, one `Block` per page (1-based page numbers).
pub fn extract_pdf(path: &Path) -> Result<BookDoc, ExtractError> {
    let doc = lopdf::Document::load(path)?;
    let mut blocks = Vec::new();
    for (&page_num, _) in doc.get_pages().iter() {
        let raw = doc.extract_text(&[page_num]).unwrap_or_default();
        let cleaned = clean_text(&raw);
        if !cleaned.is_empty() {
            blocks.push(Block::new(cleaned, None, Some(page_num)));
        }
    }
    Ok(BookDoc {
        book_id: stable_book_id(path),
        title: title_from_path(path),
        author: None,
        source_path: path.to_string_lossy().to_string(),
        format: Format::Pdf,
        blocks,
    })
}

/// Dispatch on file extension. Returns a BookDoc whose `blocks` may be empty when
/// extraction yielded no usable text (scanned PDF) — the caller should skip it.
pub fn extract(path: &Path) -> Result<BookDoc, ExtractError> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .and_then(Format::from_ext)
    {
        Some(Format::Pdf) => {
            let doc = extract_pdf(path)?;
            let chars: usize = doc.blocks.iter().map(|b| b.text.chars().count()).sum();
            if chars < MIN_BOOK_CHARS {
                Ok(BookDoc {
                    blocks: Vec::new(),
                    ..doc
                })
            } else {
                Ok(doc)
            }
        }
        _ => Err(ExtractError::Unsupported(
            path.to_string_lossy().to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_text_dehyphenates_and_collapses() {
        assert_eq!(clean_text("encyclo-\npedia"), "encyclopedia");
        assert_eq!(clean_text("a   b\t c"), "a b c");
    }

    #[test]
    fn book_id_is_stable_per_path() {
        let a = stable_book_id(Path::new("/lib/x.pdf"));
        let b = stable_book_id(Path::new("/lib/x.pdf"));
        let c = stable_book_id(Path::new("/lib/y.pdf"));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn title_from_stem() {
        assert_eq!(
            title_from_path(Path::new("/lib/Kotlin in Depth.pdf")),
            "Kotlin in Depth"
        );
    }
}
