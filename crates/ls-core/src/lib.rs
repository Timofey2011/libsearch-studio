//! Core domain types shared across the LibSearch Studio engine crates.
//!
//! This crate is UI- and IO-agnostic: it defines the records that flow between
//! extraction, chunking, indexing, and querying. Nothing here depends on Tauri,
//! ONNX, LanceDB, or the network, so every higher layer can be unit-tested
//! against these types in isolation.

use serde::{Deserialize, Serialize};

/// Source file format of a book.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Pdf,
    Epub,
    Mobi,
}

impl Format {
    /// Parse from a lowercase file extension (without the dot).
    pub fn from_ext(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "pdf" => Some(Self::Pdf),
            "epub" => Some(Self::Epub),
            "mobi" => Some(Self::Mobi),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Epub => "epub",
            Self::Mobi => "mobi",
        }
    }
}

/// An ordered unit of text from a book, with structural metadata.
///
/// `page` is meaningful for PDF only; epub/mobi carry chapter + character offset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
    pub text: String,
    pub chapter: Option<String>,
    pub page: Option<u32>,
}

impl Block {
    pub fn new(text: impl Into<String>, chapter: Option<String>, page: Option<u32>) -> Self {
        Self {
            text: text.into(),
            chapter,
            page,
        }
    }
}

/// A normalized, format-agnostic book: ordered blocks plus metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BookDoc {
    pub book_id: String,
    pub title: String,
    pub author: Option<String>,
    pub source_path: String,
    pub format: Format,
    pub blocks: Vec<Block>,
}

/// One retrievable passage — becomes exactly one row in the vector store.
///
/// Character offsets (`loc_start`/`loc_end`) are counted in Unicode scalar values
/// so they stay meaningful for non-Latin scripts (the library is ~28% Russian).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub book_id: String,
    pub title: String,
    pub author: Option<String>,
    pub source_path: String,
    pub format: Format,
    pub chapter: Option<String>,
    pub page: Option<u32>,
    pub loc_start: usize,
    pub loc_end: usize,
    pub text: String,
    /// Dense embedding, filled by the embedding stage (`None` until then).
    pub vector: Option<Vec<f32>>,
}

/// Counts tokens for chunk sizing. The real implementation (in `ls-embed`) wraps
/// the embedder's tokenizer; tests use [`WhitespaceCounter`]. Defined here in the
/// leaf crate so both `ls-index` (consumer) and `ls-embed` (provider) can use it
/// without an orphan-rule violation.
pub trait TokenCounter {
    fn count(&self, text: &str) -> usize;
}

/// Cheap, offline token estimate: one token per whitespace-delimited word.
pub struct WhitespaceCounter;

impl TokenCounter for WhitespaceCounter {
    fn count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_roundtrips_through_ext() {
        assert_eq!(Format::from_ext("PDF"), Some(Format::Pdf));
        assert_eq!(Format::from_ext("epub"), Some(Format::Epub));
        assert_eq!(Format::from_ext("txt"), None);
        assert_eq!(Format::Pdf.as_str(), "pdf");
    }
}
