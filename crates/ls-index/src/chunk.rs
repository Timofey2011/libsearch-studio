//! Chapter-aware structural chunking (port of the validated Python strategy).

use ls_core::{BookDoc, Chunk, TokenCounter};
use regex::Regex;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct ChunkParams {
    pub target_tokens: usize,
    pub overlap_tokens: usize,
    pub min_tokens: usize,
}

impl Default for ChunkParams {
    fn default() -> Self {
        Self {
            target_tokens: 400,
            overlap_tokens: 80,
            min_tokens: 100,
        }
    }
}

/// A paragraph-sized unit with its place in the book.
struct Unit {
    text: String,
    chapter: Option<String>,
    page: Option<u32>,
    loc_start: usize,
    loc_end: usize,
    tokens: usize,
}

fn paragraph_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\n[ \t]*\n").unwrap())
}

/// Split a block into (paragraph, char-offset-within-block) pairs.
fn split_paragraphs(text: &str) -> Vec<(String, usize)> {
    let re = paragraph_re();
    let mut out = Vec::new();
    let mut last = 0usize;
    let mut push = |seg: &str, seg_byte_start: usize| {
        let trimmed = seg.trim();
        if trimmed.is_empty() {
            return;
        }
        let leading_ws = seg.len() - seg.trim_start().len();
        let para_byte_start = seg_byte_start + leading_ws;
        let char_start = text[..para_byte_start].chars().count();
        out.push((trimmed.to_string(), char_start));
    };
    for m in re.find_iter(text) {
        push(&text[last..m.start()], last);
        last = m.end();
    }
    push(&text[last..], last);
    out
}

/// Yield (text, char_start, tokens) pieces, splitting a paragraph over `target`
/// tokens. One token count per paragraph; split by the tokens/word ratio (O(words)).
fn split_oversized(
    para: &str,
    start: usize,
    counter: &dyn TokenCounter,
    target: usize,
) -> Vec<(String, usize, usize)> {
    let n_tok = counter.count(para);
    if n_tok <= target {
        return vec![(para.to_string(), start, n_tok)];
    }
    let words: Vec<&str> = para.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    let ratio = n_tok as f64 / words.len() as f64;
    let words_per_piece = ((target as f64 / ratio) as usize).max(1);

    let mut out = Vec::new();
    let mut cursor = start;
    let mut i = 0;
    while i < words.len() {
        let end = (i + words_per_piece).min(words.len());
        let piece = words[i..end].join(" ");
        let est = (((end - i) as f64 * ratio).round() as usize).max(1);
        let piece_chars = piece.chars().count();
        out.push((piece, cursor, est));
        cursor += piece_chars + 1;
        i = end;
    }
    out
}

/// Flatten a book into ordered paragraph units with global char offsets.
fn book_units(doc: &BookDoc, counter: &dyn TokenCounter, params: &ChunkParams) -> Vec<Unit> {
    let mut units = Vec::new();
    let mut offset = 0usize;
    for block in &doc.blocks {
        for (para, para_start_in_block) in split_paragraphs(&block.text) {
            let start = offset + para_start_in_block;
            for (piece, p_start, tok) in
                split_oversized(&para, start, counter, params.target_tokens)
            {
                let char_len = piece.chars().count();
                units.push(Unit {
                    text: piece,
                    chapter: block.chapter.clone(),
                    page: block.page,
                    loc_start: p_start,
                    loc_end: p_start + char_len,
                    tokens: tok,
                });
            }
        }
        offset += block.text.chars().count() + 2; // inter-block separation
    }
    units
}

/// Greedily pack units into windows of ~target tokens with ~overlap carryover.
/// Returns windows as index lists into `units`.
fn pack(units: &[Unit], params: &ChunkParams) -> Vec<Vec<usize>> {
    let mut windows: Vec<Vec<usize>> = Vec::new();
    let n = units.len();
    let mut i = 0;
    while i < n {
        let mut cur = Vec::new();
        let mut tok = 0usize;
        let mut j = i;
        while j < n && (cur.is_empty() || tok + units[j].tokens <= params.target_tokens) {
            cur.push(j);
            tok += units[j].tokens;
            j += 1;
        }
        windows.push(cur);
        if j >= n {
            break;
        }
        // Step back over trailing units worth ~overlap_tokens for the next window.
        let mut back = 0usize;
        let mut otok = 0usize;
        let mut k = j - 1;
        while k > i && otok < params.overlap_tokens {
            otok += units[k].tokens;
            back += 1;
            k -= 1;
        }
        i = std::cmp::max(i + 1, j.saturating_sub(back));
    }
    windows
}

/// Fold a trailing window under `min_tokens` into the previous one.
fn merge_short_tail(
    mut windows: Vec<Vec<usize>>,
    units: &[Unit],
    params: &ChunkParams,
) -> Vec<Vec<usize>> {
    if windows.len() < 2 {
        return windows;
    }
    let last_tokens: usize = windows
        .last()
        .unwrap()
        .iter()
        .map(|&i| units[i].tokens)
        .sum();
    if last_tokens < params.min_tokens {
        let last = windows.pop().unwrap();
        windows.last_mut().unwrap().extend(last);
    }
    windows
}

fn window_to_chunk(doc: &BookDoc, units: &[Unit], window: &[usize], index: usize) -> Chunk {
    let text = window
        .iter()
        .map(|&i| units[i].text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let first = &units[window[0]];
    let last = &units[*window.last().unwrap()];
    Chunk {
        id: format!("{}:{}", doc.book_id, index),
        book_id: doc.book_id.clone(),
        title: doc.title.clone(),
        author: doc.author.clone(),
        source_path: doc.source_path.clone(),
        format: doc.format,
        chapter: first.chapter.clone(),
        page: first.page, // page where the passage begins
        loc_start: first.loc_start,
        loc_end: last.loc_end,
        text,
        vector: None,
    }
}

/// Chunk a book into retrieval passages, never crossing a chapter boundary.
pub fn chunk_book(doc: &BookDoc, counter: &dyn TokenCounter, params: &ChunkParams) -> Vec<Chunk> {
    let units = book_units(doc, counter, params);
    let mut chunks = Vec::new();
    let mut index = 0usize;

    // Group consecutive same-chapter units, then pack within each group so no
    // chunk straddles two chapters.
    let mut start = 0;
    while start < units.len() {
        let chapter = &units[start].chapter;
        let mut end = start + 1;
        while end < units.len() && &units[end].chapter == chapter {
            end += 1;
        }
        let group = &units[start..end];
        let windows = merge_short_tail(pack(group, params), group, params);
        for window in windows {
            chunks.push(window_to_chunk(doc, group, &window, index));
            index += 1;
        }
        start = end;
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use ls_core::{Block, Format, WhitespaceCounter};

    const P: ChunkParams = ChunkParams {
        target_tokens: 10,
        overlap_tokens: 3,
        min_tokens: 3,
    };

    fn para(n: usize, word: &str) -> String {
        vec![word; n].join(" ")
    }

    fn doc(blocks: Vec<Block>) -> BookDoc {
        BookDoc {
            book_id: "bk".into(),
            title: "T".into(),
            author: Some("A".into()),
            source_path: "/p.pdf".into(),
            format: Format::Pdf,
            blocks,
        }
    }

    fn words(text: &str) -> std::collections::HashSet<String> {
        text.split_whitespace().map(|s| s.to_string()).collect()
    }

    #[test]
    fn never_crosses_chapter_boundary() {
        let d = doc(vec![
            Block::new(para(8, "alpha"), Some("Ch1".into()), Some(1)),
            Block::new(para(8, "beta"), Some("Ch2".into()), Some(2)),
        ]);
        let chunks = chunk_book(&d, &WhitespaceCounter, &P);
        for c in &chunks {
            let w = words(&c.text);
            let only_alpha = w.iter().all(|x| x == "alpha");
            let only_beta = w.iter().all(|x| x == "beta");
            assert!(only_alpha || only_beta, "chunk mixed chapters: {}", c.text);
        }
        let chapters: std::collections::HashSet<_> =
            chunks.iter().map(|c| c.chapter.clone()).collect();
        assert_eq!(chapters.len(), 2);
    }

    #[test]
    fn window_sizes_within_bounds() {
        let body = (0..5)
            .map(|_| para(6, "w"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let d = doc(vec![Block::new(body, Some("C".into()), Some(1))]);
        let chunks = chunk_book(&d, &WhitespaceCounter, &P);
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(WhitespaceCounter.count(&c.text) <= P.target_tokens + 6);
        }
    }

    #[test]
    fn overlap_present_between_consecutive_chunks() {
        let body = (0..6)
            .map(|i| para(4, &format!("p{i}")))
            .collect::<Vec<_>>()
            .join("\n\n");
        let d = doc(vec![Block::new(body, Some("C".into()), Some(1))]);
        let chunks = chunk_book(&d, &WhitespaceCounter, &P);
        assert!(chunks.len() >= 2);
        let a = words(&chunks[0].text);
        let b = words(&chunks[1].text);
        assert!(a.intersection(&b).next().is_some(), "expected overlap");
    }

    #[test]
    fn oversized_paragraph_is_split_not_truncated() {
        let d = doc(vec![Block::new(para(35, "big"), Some("C".into()), Some(1))]);
        let chunks = chunk_book(&d, &WhitespaceCounter, &P);
        let total: usize = chunks
            .iter()
            .map(|c| c.text.split_whitespace().count())
            .sum();
        assert!(total >= 35, "words lost: {total}");
    }

    #[test]
    fn short_tail_merged() {
        let body = format!("{}\n\n{}", para(10, "w"), para(1, "tail"));
        let d = doc(vec![Block::new(body, Some("C".into()), Some(1))]);
        let chunks = chunk_book(&d, &WhitespaceCounter, &P);
        assert!(!chunks.iter().any(|c| c.text.trim() == "tail"));
        assert!(chunks.iter().any(|c| c.text.contains("tail")));
    }

    #[test]
    fn metadata_carried_and_ids_sequential() {
        let d = doc(vec![
            Block::new(para(8, "w"), Some("Ch1".into()), Some(3)),
            Block::new(para(8, "x"), Some("Ch2".into()), Some(7)),
        ]);
        let chunks = chunk_book(&d, &WhitespaceCounter, &P);
        for c in &chunks {
            assert_eq!(c.title, "T");
            assert_eq!(c.author.as_deref(), Some("A"));
            assert_eq!(c.book_id, "bk");
            assert!(c.chapter == Some("Ch1".into()) || c.chapter == Some("Ch2".into()));
            assert!(c.page == Some(3) || c.page == Some(7));
            assert!(c.loc_end >= c.loc_start);
        }
        let ids: Vec<_> = chunks.iter().map(|c| c.id.clone()).collect();
        let expected: Vec<_> = (0..chunks.len()).map(|i| format!("bk:{i}")).collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn cyrillic_offsets_are_char_based() {
        // Two short paragraphs of Russian text; offsets should be char counts.
        let body = "привет мир тест\n\nвторой абзац здесь".to_string();
        let d = doc(vec![Block::new(body, Some("Гл1".into()), None)]);
        let chunks = chunk_book(&d, &WhitespaceCounter, &P);
        assert!(!chunks.is_empty());
        // First paragraph starts at char 0; second after "привет мир тест" (15 chars) + 2.
        assert_eq!(chunks[0].loc_start, 0);
    }
}
