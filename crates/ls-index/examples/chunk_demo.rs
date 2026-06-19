//! Tangible demo of the chunker: build a small book, chunk it, print the result.
//!
//! Run with:  cargo run -p ls-index --example chunk_demo

use ls_core::{Block, BookDoc, Format};
use ls_index::{chunk_book, ChunkParams, WhitespaceCounter};

fn main() {
    // A tiny two-chapter "book" with normal paragraphs, one long paragraph, and
    // a short trailing line — exercises packing, oversize split, and tail-merge.
    let long = vec!["idea"; 60].join(" ");
    let doc = BookDoc {
        book_id: "demo".into(),
        title: "Demo Book".into(),
        author: Some("A. Author".into()),
        source_path: "/tmp/demo.pdf".into(),
        format: Format::Pdf,
        blocks: vec![
            Block::new(
                "Introduction to the topic with a first paragraph.\n\n\
                 A second paragraph that continues the discussion in some detail.",
                Some("Chapter 1".into()),
                Some(1),
            ),
            Block::new(
                format!("{long}\n\nshort tail"),
                Some("Chapter 2".into()),
                Some(5),
            ),
            // Russian paragraph to show multilingual char-offset handling.
            Block::new(
                "Это пример абзаца на русском языке для проверки смещений.",
                Some("Глава 3".into()),
                Some(9),
            ),
        ],
    };

    // Small params so the demo produces several chunks from little text.
    let params = ChunkParams {
        target_tokens: 20,
        overlap_tokens: 5,
        min_tokens: 4,
    };
    let chunks = chunk_book(&doc, &WhitespaceCounter, &params);

    println!("Chunked \"{}\" into {} chunks:\n", doc.title, chunks.len());
    for c in &chunks {
        let tokens = c.text.split_whitespace().count();
        let page = c.page.map(|p| p.to_string()).unwrap_or_else(|| "-".into());
        let preview: String = c.text.chars().take(70).collect();
        println!(
            "[{}] {} · p.{} · loc {}..{} · {} tok\n    {}{}\n",
            c.id,
            c.chapter.as_deref().unwrap_or("(no chapter)"),
            page,
            c.loc_start,
            c.loc_end,
            tokens,
            preview.replace('\n', " "),
            if c.text.chars().count() > 70 {
                "…"
            } else {
                ""
            },
        );
    }
}
