//! Indexing-side logic: structural chunking + the LanceDB store.
//!
//! The chunker is a faithful Rust port of the validated Python strategy
//! (`ebook-kb/src/chunk.py`). The store reads the same on-disk LanceDB schema the
//! Python engine writes, so existing indexes are directly usable.

pub mod chunk;
pub mod store;

pub use chunk::{chunk_book, ChunkParams};
pub use ls_core::{TokenCounter, WhitespaceCounter};
pub use store::{ChunkMeta, RetrievedChunk, Store, StoreError};
