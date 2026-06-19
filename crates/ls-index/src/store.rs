//! LanceDB-backed chunk store: open an index and run vector / full-text search.
//!
//! Reads the same schema the Python engine writes (`lance` 0.33 on-disk format),
//! so an index built by `ebook-kb` is directly usable here. Writing/indexing from
//! Rust is added in a later milestone.

use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch, RecordBatchIterator,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use lancedb::index::scalar::FtsIndexBuilder;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Table};
use ls_core::{Chunk, Format};

/// Embedding dimension (bge-m3).
pub const VECTOR_DIM: i32 = 1024;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error(transparent)]
    Lance(#[from] lancedb::Error),
    #[error("result stream: {0}")]
    Stream(String),
    #[error("schema: {0}")]
    Schema(String),
}

/// A row read back from the store (vector column omitted).
#[derive(Debug, Clone)]
pub struct RetrievedChunk {
    pub id: String,
    pub book_id: String,
    pub title: String,
    pub author: Option<String>,
    pub source_path: String,
    pub format: Option<Format>,
    pub chapter: Option<String>,
    pub page: Option<u32>,
    pub loc_start: i64,
    pub loc_end: i64,
    pub text: String,
}

/// Arrow schema for the chunks table — mirrors the Python engine's columns.
pub fn chunk_schema() -> SchemaRef {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("book_id", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("author", DataType::Utf8, false),
        Field::new("source_path", DataType::Utf8, false),
        Field::new("format", DataType::Utf8, false),
        Field::new("chapter", DataType::Utf8, false),
        Field::new("page", DataType::Int64, false),
        Field::new("loc_start", DataType::Int64, false),
        Field::new("loc_end", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("vector", DataType::FixedSizeList(item, VECTOR_DIM), false),
    ]))
}

const NULL_PAGE: i64 = -1;

/// Build a RecordBatch from embedded chunks (vectors must be present).
fn chunks_to_batch(chunks: &[Chunk]) -> Result<RecordBatch, StoreError> {
    let str_of =
        |f: &dyn Fn(&Chunk) -> String| StringArray::from(chunks.iter().map(f).collect::<Vec<_>>());
    let id = str_of(&|c| c.id.clone());
    let book_id = str_of(&|c| c.book_id.clone());
    let title = str_of(&|c| c.title.clone());
    let author = str_of(&|c| c.author.clone().unwrap_or_default());
    let source_path = str_of(&|c| c.source_path.clone());
    let format = str_of(&|c| c.format.as_str().to_string());
    let chapter = str_of(&|c| c.chapter.clone().unwrap_or_default());
    let text = str_of(&|c| c.text.clone());
    let page = Int64Array::from(
        chunks
            .iter()
            .map(|c| c.page.map(|p| p as i64).unwrap_or(NULL_PAGE))
            .collect::<Vec<_>>(),
    );
    let loc_start = Int64Array::from(
        chunks
            .iter()
            .map(|c| c.loc_start as i64)
            .collect::<Vec<_>>(),
    );
    let loc_end = Int64Array::from(chunks.iter().map(|c| c.loc_end as i64).collect::<Vec<_>>());

    let mut flat = Vec::with_capacity(chunks.len() * VECTOR_DIM as usize);
    for c in chunks {
        let v = c
            .vector
            .as_ref()
            .ok_or_else(|| StoreError::Schema(format!("chunk {} has no vector", c.id)))?;
        if v.len() != VECTOR_DIM as usize {
            return Err(StoreError::Schema(format!(
                "chunk {} vector dim {}",
                c.id,
                v.len()
            )));
        }
        flat.extend_from_slice(v);
    }
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let vector =
        FixedSizeListArray::new(item, VECTOR_DIM, Arc::new(Float32Array::from(flat)), None);

    RecordBatch::try_new(
        chunk_schema(),
        vec![
            Arc::new(id),
            Arc::new(book_id),
            Arc::new(title),
            Arc::new(author),
            Arc::new(source_path),
            Arc::new(format),
            Arc::new(chapter),
            Arc::new(page),
            Arc::new(loc_start),
            Arc::new(loc_end),
            Arc::new(text),
            Arc::new(vector),
        ],
    )
    .map_err(|e| StoreError::Schema(e.to_string()))
}

pub struct Store {
    #[allow(dead_code)]
    conn: Connection,
    table: Table,
}

impl Store {
    /// Open an existing table in the LanceDB directory at `db_path`.
    pub async fn open(db_path: impl AsRef<Path>, table_name: &str) -> Result<Self, StoreError> {
        let uri = db_path.as_ref().to_string_lossy().to_string();
        let conn = lancedb::connect(&uri).execute().await?;
        let table = conn.open_table(table_name).execute().await?;
        Ok(Self { conn, table })
    }

    /// Open the table, creating an empty one with the right schema if absent.
    pub async fn open_or_create(
        db_path: impl AsRef<Path>,
        table_name: &str,
    ) -> Result<Self, StoreError> {
        let uri = db_path.as_ref().to_string_lossy().to_string();
        let conn = lancedb::connect(&uri).execute().await?;
        let names = conn.table_names().execute().await?;
        let table = if names.iter().any(|n| n == table_name) {
            conn.open_table(table_name).execute().await?
        } else {
            let empty = RecordBatchIterator::new(std::iter::empty(), chunk_schema());
            conn.create_table(table_name, Box::new(empty))
                .execute()
                .await?
        };
        Ok(Self { conn, table })
    }

    /// Append embedded chunks. Returns the number of rows written.
    pub async fn add_chunks(&self, chunks: &[Chunk]) -> Result<usize, StoreError> {
        if chunks.is_empty() {
            return Ok(0);
        }
        let batch = chunks_to_batch(chunks)?;
        let schema = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.table.add(Box::new(reader)).execute().await?;
        Ok(chunks.len())
    }

    /// Build (or rebuild) the full-text index on `text`.
    pub async fn ensure_fts_index(&self) -> Result<(), StoreError> {
        self.table
            .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
            .execute()
            .await?;
        Ok(())
    }

    /// Remove all chunks for a book (idempotent re-index).
    pub async fn delete_book(&self, book_id: &str) -> Result<(), StoreError> {
        let safe = book_id.replace('\'', "''");
        self.table.delete(&format!("book_id = '{safe}'")).await?;
        Ok(())
    }

    pub async fn count(&self) -> Result<usize, StoreError> {
        Ok(self.table.count_rows(None).await?)
    }

    /// Nearest-neighbour search by embedding vector.
    pub async fn vector_search(
        &self,
        vector: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<RetrievedChunk>, StoreError> {
        let stream = self
            .table
            .vector_search(vector)?
            .limit(limit)
            .execute()
            .await?;
        collect(stream).await
    }

    /// Full-text search over the `text` column (uses the existing FTS index).
    pub async fn fts_search(
        &self,
        text: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedChunk>, StoreError> {
        let q = lancedb::index::scalar::FullTextSearchQuery::new(text.to_string());
        let stream = self
            .table
            .query()
            .full_text_search(q)
            .limit(limit)
            .execute()
            .await?;
        collect(stream).await
    }
}

async fn collect<S, E>(mut stream: S) -> Result<Vec<RetrievedChunk>, StoreError>
where
    S: futures::Stream<Item = Result<RecordBatch, E>> + Unpin,
    E: std::fmt::Display,
{
    let mut out = Vec::new();
    while let Some(item) = stream.next().await {
        let batch = item.map_err(|e| StoreError::Stream(e.to_string()))?;
        rows_from_batch(&batch, &mut out)?;
    }
    Ok(out)
}

fn str_col<'a>(b: &'a RecordBatch, name: &str) -> Result<&'a StringArray, StoreError> {
    b.column_by_name(name)
        .ok_or_else(|| StoreError::Schema(format!("missing column {name}")))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| StoreError::Schema(format!("column {name} is not Utf8")))
}

fn int_col<'a>(b: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, StoreError> {
    b.column_by_name(name)
        .ok_or_else(|| StoreError::Schema(format!("missing column {name}")))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| StoreError::Schema(format!("column {name} is not Int64")))
}

fn rows_from_batch(batch: &RecordBatch, out: &mut Vec<RetrievedChunk>) -> Result<(), StoreError> {
    let id = str_col(batch, "id")?;
    let book_id = str_col(batch, "book_id")?;
    let title = str_col(batch, "title")?;
    let author = str_col(batch, "author")?;
    let source_path = str_col(batch, "source_path")?;
    let format = str_col(batch, "format")?;
    let chapter = str_col(batch, "chapter")?;
    let text = str_col(batch, "text")?;
    let page = int_col(batch, "page")?;
    let loc_start = int_col(batch, "loc_start")?;
    let loc_end = int_col(batch, "loc_end")?;

    let opt = |s: &str| (!s.is_empty()).then(|| s.to_string());
    for i in 0..batch.num_rows() {
        let p = page.value(i);
        out.push(RetrievedChunk {
            id: id.value(i).to_string(),
            book_id: book_id.value(i).to_string(),
            title: title.value(i).to_string(),
            author: opt(author.value(i)),
            source_path: source_path.value(i).to_string(),
            format: Format::from_ext(format.value(i)),
            chapter: opt(chapter.value(i)),
            page: if p < 0 { None } else { Some(p as u32) },
            loc_start: loc_start.value(i),
            loc_end: loc_end.value(i),
            text: text.value(i).to_string(),
        });
    }
    Ok(())
}
