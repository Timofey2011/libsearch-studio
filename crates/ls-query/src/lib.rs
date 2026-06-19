//! Query pipeline: embed → hybrid (vector + full-text) → RRF → rerank → cited results.
//!
//! Mirrors the Python engine's retrieve-broad / rerank-narrow shape. Reciprocal
//! Rank Fusion (RRF) merges the vector and full-text candidate lists with no model;
//! the cross-encoder then reranks the fused set down to the final top-k.

use std::collections::HashMap;

use ls_core::Format;
use ls_embed::{Embedder, Reranker};
use ls_index::{RetrievedChunk, Store, StoreError};

/// RRF damping constant (standard default).
const RRF_K: f64 = 60.0;

#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Embed(#[from] ls_embed::EmbedError),
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub rank: usize,
    pub score: f32,
    pub text: String,
    pub citation: String,
    pub title: String,
    pub author: Option<String>,
    pub chapter: Option<String>,
    pub page: Option<u32>,
    pub source_path: String,
    pub book_id: String,
    pub id: String,
}

/// Human-readable citation. Page numbers for pdf only; else chapter or char location.
pub fn format_citation(c: &RetrievedChunk) -> String {
    let mut head = c.title.clone();
    if let Some(author) = c.author.as_deref().filter(|a| !a.is_empty()) {
        head = format!("{head} — {author}");
    }

    let mut loc: Vec<String> = Vec::new();
    if let Some(chapter) = c.chapter.as_deref().filter(|s| !s.is_empty()) {
        loc.push(format!("Ch. {chapter}"));
    }
    if let (Some(Format::Pdf), Some(page)) = (c.format, c.page) {
        loc.push(format!("p.{page}"));
    } else if c.chapter.is_none() {
        loc.push(format!("~loc {}", c.loc_start));
    }

    if loc.is_empty() {
        head
    } else {
        format!("{head} · {}", loc.join(", "))
    }
}

/// Fuse ranked candidate lists with Reciprocal Rank Fusion, keep the top `k`.
pub fn rrf_fuse(lists: &[Vec<RetrievedChunk>], k: usize) -> Vec<RetrievedChunk> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut by_id: HashMap<String, RetrievedChunk> = HashMap::new();
    for list in lists {
        for (rank, c) in list.iter().enumerate() {
            *scores.entry(c.id.clone()).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
            by_id.entry(c.id.clone()).or_insert_with(|| c.clone());
        }
    }
    let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(k);
    ranked
        .into_iter()
        .filter_map(|(id, _)| by_id.remove(&id))
        .collect()
}

/// Run the full retrieve → rerank → cite pipeline for one query.
pub async fn search(
    store: &Store,
    embedder: &mut Embedder,
    reranker: &mut Reranker,
    query: &str,
    final_k: usize,
    hybrid_k: usize,
) -> Result<Vec<SearchResult>, QueryError> {
    let qvec = embedder.embed_query(query)?;
    let vec_hits = store.vector_search(qvec, hybrid_k).await?;
    // Degrade to vector-only if the FTS index isn't built yet (e.g. a collection
    // mid-index) instead of failing the whole query.
    let fts_hits = store.fts_search(query, hybrid_k).await.unwrap_or_default();

    let candidates = rrf_fuse(&[vec_hits, fts_hits], hybrid_k);
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let texts: Vec<&str> = candidates.iter().map(|c| c.text.as_str()).collect();
    let scores = reranker.score(query, &texts)?;

    let mut ranked: Vec<(RetrievedChunk, f32)> = candidates.into_iter().zip(scores).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Ok(ranked
        .into_iter()
        .take(final_k)
        .enumerate()
        .map(|(i, (c, score))| SearchResult {
            rank: i + 1,
            score,
            citation: format_citation(&c),
            text: c.text,
            title: c.title,
            author: c.author,
            chapter: c.chapter,
            page: c.page,
            source_path: c.source_path,
            book_id: c.book_id,
            id: c.id,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, format: Format, page: Option<u32>, chapter: Option<&str>) -> RetrievedChunk {
        RetrievedChunk {
            id: id.into(),
            book_id: "b".into(),
            title: "The Book".into(),
            author: Some("Ada".into()),
            source_path: "/x".into(),
            format: Some(format),
            chapter: chapter.map(String::from),
            page,
            loc_start: 1234,
            loc_end: 1300,
            text: "passage".into(),
        }
    }

    #[test]
    fn citation_pdf_uses_page() {
        let c = chunk("b:1", Format::Pdf, Some(42), Some("Intro"));
        assert_eq!(format_citation(&c), "The Book — Ada · Ch. Intro, p.42");
    }

    #[test]
    fn citation_epub_uses_chapter_no_page() {
        let c = chunk("b:1", Format::Epub, None, Some("Deep Dive"));
        assert_eq!(format_citation(&c), "The Book — Ada · Ch. Deep Dive");
    }

    #[test]
    fn citation_falls_back_to_loc() {
        let mut c = chunk("b:1", Format::Epub, None, None);
        c.author = None;
        assert_eq!(format_citation(&c), "The Book · ~loc 1234");
    }

    #[test]
    fn rrf_prefers_items_ranked_high_in_both_lists() {
        let a = vec![
            chunk("x", Format::Pdf, Some(1), None),
            chunk("y", Format::Pdf, Some(2), None),
        ];
        let b = vec![
            chunk("y", Format::Pdf, Some(2), None),
            chunk("z", Format::Pdf, Some(3), None),
        ];
        let fused = rrf_fuse(&[a, b], 3);
        // y appears in both lists -> highest fused score -> first.
        assert_eq!(fused[0].id, "y");
        assert_eq!(fused.len(), 3);
    }
}
