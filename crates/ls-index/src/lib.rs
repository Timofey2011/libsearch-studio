//! Indexing-side logic: structural chunking (and, later, the LanceDB store).
//!
//! The chunker is a faithful Rust port of the validated Python strategy
//! (`ebook-kb/src/chunk.py`): split on structure (chapter -> paragraph), never
//! cross a chapter boundary, greedily pack into ~target-token windows with
//! overlap, hard-split oversized paragraphs in O(words) (one token count per
//! paragraph, splitting by the tokens/word ratio), and merge a short trailing
//! window into the previous one.

pub mod chunk;

pub use chunk::{chunk_book, ChunkParams, TokenCounter, WhitespaceCounter};
