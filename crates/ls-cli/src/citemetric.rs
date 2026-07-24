//! §17.1 citation-integrity metric — store↔display integrity over the real
//! store. NOT a DOM-side jump measurement (the harness re-extracts with the
//! same code that produced the chunks); the report header states the known
//! inflation/deflation caveats. No models, no network.
//!
//! The matcher replicas below are VERBATIM ports; cross-pins:
//! - norm_text / norm_chapter  ⇔ frontend/src/BookReader.tsx `normText`/`normChapter`
//! - probe_of                  ⇔ frontend/src/BookReader.tsx `citeJump` probe selection
//! - md_norm / md_fragments    ⇔ frontend/src/App.tsx TreeWalker norm + renderRich/renderInline
//!
//! JS `\w` is ASCII-only — the dehyphenation classes here deliberately exclude
//! Cyrillic (the frontend never dehyphenates RU; a Unicode `\w` port would
//! silently diverge). The extra lowercase in `fold` emulates foliate's
//! case-insensitive collator (sensitivity 'base'); diacritic folding is NOT
//! emulated (rare in this library, noted in the report).

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

fn re(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).expect("static regex"))
}

/// ⇔ BookReader.tsx `normText` (soft hyphens, ASCII line-break hyphenation,
/// whitespace collapse).
pub fn norm_text(s: &str) -> String {
    static HYPH: OnceLock<Regex> = OnceLock::new();
    static WS: OnceLock<Regex> = OnceLock::new();
    let s = s.replace('\u{00AD}', "");
    let s = re(&HYPH, r"([0-9A-Za-z_])-\s+([0-9A-Za-z_])").replace_all(&s, "$1$2");
    re(&WS, r"\s+").replace_all(&s, " ").trim().to_string()
}

/// Foliate collator emulation (case-insensitivity only).
pub fn fold(s: &str) -> String {
    s.to_lowercase()
}

/// ⇔ BookReader.tsx `normChapter` (A4c label drift + §17.2c roman numerals).
pub fn norm_chapter(s: &str) -> String {
    static PREFIX: OnceLock<Regex> = OnceLock::new();
    static NUM: OnceLock<Regex> = OnceLock::new();
    static ROMAN: OnceLock<Regex> = OnceLock::new();
    let s = norm_text(s);
    let s = re(
        &PREFIX,
        r"(?i)^(chapter|глава|часть|part)\s+(\d+|[ivxlcdm]+)\.?:?\s+",
    )
    .replace(&s, "");
    let s = re(&NUM, r"^[\d.]+\s*[.:—-]?\s*").replace(&s, "");
    let s = re(&ROMAN, r"(?i)^[ivxlcdm]+\.\s+").replace(&s, "");
    s.to_lowercase()
}

/// ⇔ BookReader.tsx `findTocEntry` (§17.2c tiered matching): exact →
/// containment → best token overlap. Returns the index of the matched label.
pub fn find_label<'a, I: Iterator<Item = &'a str> + Clone>(
    labels: I,
    stored: &str,
) -> Option<usize> {
    let want = norm_chapter(stored);
    if want.is_empty() {
        return None;
    }
    let normed: Vec<String> = labels.map(norm_chapter).collect();
    if let Some(i) = normed.iter().position(|l| *l == want) {
        return Some(i);
    }
    if want.chars().count() >= 6 {
        if let Some(i) = normed.iter().position(|l| {
            l.chars().count() >= 6 && (l.contains(&want) || want.contains(l.as_str()))
        }) {
            return Some(i);
        }
    }
    let tokens = |s: &str| -> std::collections::HashSet<String> {
        s.split(' ')
            .filter(|w| w.chars().count() >= 3)
            .map(str::to_string)
            .collect()
    };
    let wt = tokens(&want);
    if wt.len() < 2 {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for (i, l) in normed.iter().enumerate() {
        let lt = tokens(l);
        if lt.is_empty() {
            continue;
        }
        let common = wt.intersection(&lt).count();
        let score = common as f64 / (wt.len() + lt.len() - common) as f64;
        if common >= 2 && score >= 0.6 && best.is_none_or(|(_, b)| score > b) {
            best = Some((i, score));
        }
    }
    best.map(|(i, _)| i)
}

/// ⇔ BookReader.tsx `citeJump` probe selection: first run of eight
/// consecutive "wordy" words (≥2 letter chars), else fall back to the first
/// eight tokens INCLUDING junk (`at = 0`) — that fallback fires exactly on
/// the junk-prefixed legacy chunks, so it must be ported, not idealized.
pub fn probe_of(cite: &str) -> String {
    let normed = norm_text(cite);
    if normed.is_empty() {
        return String::new();
    }
    let words: Vec<&str> = normed.split(' ').collect();
    let wordy = |w: &str| w.chars().filter(|c| c.is_alphabetic()).count() >= 2;
    let mut at = None;
    if words.len() >= 8 {
        for i in 0..=(words.len() - 8) {
            if words[i..i + 8].iter().all(|w| wordy(w)) {
                at = Some(i);
                break;
            }
        }
    }
    let at = at.unwrap_or(0);
    words[at..(at + 8).min(words.len())].join(" ")
}

/// ⇔ App.tsx TreeWalker `norm` (whitespace collapse + lowercase — weaker than
/// normText by design; the md path never dehyphenates).
pub fn md_norm(s: &str) -> String {
    static WS: OnceLock<Regex> = OnceLock::new();
    re(&WS, r"\s+").replace_all(s, " ").trim().to_lowercase()
}

/// ⇔ App.tsx `renderRich` block model as seen through `el.textContent`
/// (v0.16.1 block-level matching): one string per rendered block element,
/// block markers stripped, wrapped paragraph lines joined with spaces, and
/// inline `**`/`*`/`` ` `` markers removed (textContent re-joins the inline
/// fragments seamlessly — that re-join is exactly what fixed the §17.2b 23%
/// md miss rate). `[n]` citation tokens contribute no block text.
pub fn md_blocks(text: &str) -> Vec<String> {
    static HEAD: OnceLock<Regex> = OnceLock::new();
    static BULLET: OnceLock<Regex> = OnceLock::new();
    static NUMBERED: OnceLock<Regex> = OnceLock::new();
    let head = re(&HEAD, r"^\s*#{1,4}\s+(.*)$");
    let bullet = re(&BULLET, r"^\s*[-*]\s+(.*)$");
    let numbered = re(&NUMBERED, r"^\s*\d+\.\s+(.*)$");

    let mut blocks: Vec<String> = Vec::new();
    let mut para: Vec<String> = Vec::new();
    let mut in_code = false;
    let mut code: Vec<String> = Vec::new();
    for raw in text.split('\n') {
        let line = raw.trim_end();
        if in_code {
            if line.trim_start().starts_with("```") {
                blocks.push(code.join("\n"));
                code.clear();
                in_code = false;
            } else {
                code.push(raw.to_string());
            }
            continue;
        }
        if line.trim_start().starts_with("```") {
            if !para.is_empty() {
                blocks.push(para.join(" "));
                para.clear();
            }
            in_code = true;
            continue;
        }
        if let Some(c) = head.captures(line) {
            if !para.is_empty() {
                blocks.push(para.join(" "));
                para.clear();
            }
            blocks.push(c[1].to_string());
        } else if let Some(c) = bullet.captures(line) {
            if !para.is_empty() {
                blocks.push(para.join(" "));
                para.clear();
            }
            blocks.push(c[1].to_string());
        } else if let Some(c) = numbered.captures(line) {
            if !para.is_empty() {
                blocks.push(para.join(" "));
                para.clear();
            }
            blocks.push(c[1].to_string());
        } else if line.trim().is_empty() {
            if !para.is_empty() {
                blocks.push(para.join(" "));
                para.clear();
            }
        } else {
            para.push(line.to_string());
        }
    }
    if in_code {
        blocks.push(code.join("\n"));
    }
    if !para.is_empty() {
        blocks.push(para.join(" "));
    }

    // renderInline through textContent: inline markers vanish, their inner
    // text re-joins the surrounding text seamlessly; [n] tokens render as
    // links whose text isn't the source markup — treat as removed.
    static INLINE: OnceLock<Regex> = OnceLock::new();
    let inline = re(
        &INLINE,
        r"\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|\[[\d,\s]+\]",
    );
    let mut out = Vec::new();
    for b in blocks {
        let mut joined = String::new();
        let mut last = 0;
        for m in inline.find_iter(&b) {
            joined.push_str(&b[last..m.start()]);
            let cap = inline.captures(&b[m.start()..m.end()]).unwrap();
            for g in 1..=3 {
                if let Some(inner) = cap.get(g) {
                    joined.push_str(inner.as_str());
                }
            }
            last = m.end();
        }
        joined.push_str(&b[last..]);
        if !joined.trim().is_empty() {
            out.push(joined);
        }
    }
    out
}

/// JS `.slice(0, n)` operates on UTF-16 code units; for BMP text (all of this
/// library) that equals chars.
fn js_slice(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Frontend md matcher (v0.16.1): candidate needles — first 40 normalized
/// chars of the cite, plus the same with the cite's FIRST LINE dropped (chunk
/// text often starts with its section heading, rendered as a separate block)
/// — matched against per-block textContent.
pub fn md_match(chunk_text: &str, blocks: &[String]) -> bool {
    let mut candidates: Vec<String> = Vec::new();
    let full = js_slice(&md_norm(chunk_text), 40);
    if !full.is_empty() {
        candidates.push(full);
    }
    if let Some(nl) = chunk_text.find('\n') {
        let rest = js_slice(&md_norm(&chunk_text[nl + 1..]), 40);
        if !rest.is_empty() && !candidates.contains(&rest) {
            candidates.push(rest);
        }
    }
    candidates
        .iter()
        .any(|cand| blocks.iter().any(|b| md_norm(b).contains(cand)))
}

// ---------------------------------------------------------------------------
// Outcome classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// Book: chapter label resolved AND probe in the FIRST block-run carrying
    /// that label (frontend phase-1 scope). Pdf: probe on the stored page.
    Direct,
    /// Pdf only: probe on page ±1 (chunk crossing its page boundary).
    Near,
    /// Probe found, but not where the citation claims (frontend phase-2 hit).
    Located,
    /// Chapter label resolved, probe nowhere — the app still lands the user
    /// in the chapter and shows the explicit-miss overlay.
    MissInChapter,
    /// Nothing resolved, probe nowhere — overlay at text start.
    ColdMiss,
    /// Chunk carries no chapter (excluded from the direct denominator).
    ChapterlessLocated,
    ChapterlessMiss,
    /// mobi/azw3 — extractor is best-effort; not comparable, never "miss".
    Unverifiable,
    /// Family with a different render pipeline (html/office/pages/djvu).
    UnverifiedFamily,
    ExtractTimeout,
    ExtractError,
}

pub struct BookText {
    /// (normalized+folded chapter label, folded text of that label's FIRST
    /// consecutive block-run).
    pub first_runs: Vec<(String, String)>,
    /// Folded whole-book text.
    pub all: String,
    /// Folded text per page (pdf only).
    pub pages: HashMap<u32, String>,
}

impl BookText {
    pub fn from_doc(doc: &ls_core::BookDoc) -> Self {
        let mut first_runs: Vec<(String, String)> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut cur_label: Option<String> = None;
        let mut cur_text = String::new();
        let mut all = String::new();
        let mut pages: HashMap<u32, String> = HashMap::new();
        let mut flush =
            |label: &Option<String>, text: &mut String, runs: &mut Vec<(String, String)>| {
                if let Some(l) = label {
                    if !seen.contains(l) {
                        seen.insert(l.clone());
                        runs.push((l.clone(), std::mem::take(text)));
                        return;
                    }
                }
                text.clear();
            };
        for b in &doc.blocks {
            let label = b.chapter.as_deref().map(norm_chapter);
            if label != cur_label {
                flush(&cur_label, &mut cur_text, &mut first_runs);
                cur_label = label;
            }
            let folded = fold(&norm_text(&b.text));
            if cur_label.is_some() {
                cur_text.push_str(&folded);
                cur_text.push(' ');
            }
            all.push_str(&folded);
            all.push(' ');
            if let Some(p) = b.page {
                let e = pages.entry(p).or_default();
                e.push_str(&folded);
                e.push(' ');
            }
        }
        flush(&cur_label, &mut cur_text, &mut first_runs);
        BookText {
            first_runs,
            all,
            pages,
        }
    }
}

/// Book-family classification (epub/fb2): §17.1 taxonomy.
pub fn classify_book(chapter: Option<&str>, cite_text: &str, book: &BookText) -> Outcome {
    let probe = fold(&probe_of(cite_text));
    if probe.is_empty() {
        return Outcome::ColdMiss;
    }
    let anywhere = book.all.contains(&probe);
    match chapter {
        Some(ch) => {
            let hit = find_label(book.first_runs.iter().map(|(l, _)| l.as_str()), ch)
                .map(|i| &book.first_runs[i]);
            match hit {
                Some((_, run_text)) => {
                    if run_text.contains(&probe) {
                        Outcome::Direct
                    } else if anywhere {
                        Outcome::Located
                    } else {
                        Outcome::MissInChapter
                    }
                }
                None => {
                    if anywhere {
                        Outcome::Located
                    } else {
                        Outcome::ColdMiss
                    }
                }
            }
        }
        None => {
            if anywhere {
                Outcome::ChapterlessLocated
            } else {
                Outcome::ChapterlessMiss
            }
        }
    }
}

/// Pdf classification: "stored page still contains the passage per lopdf".
pub fn classify_pdf(page: Option<u32>, cite_text: &str, book: &BookText) -> Outcome {
    let probe = fold(&probe_of(cite_text));
    if probe.is_empty() {
        return Outcome::ColdMiss;
    }
    let anywhere = book.all.contains(&probe);
    match page {
        Some(p) => {
            let on = |q: u32| book.pages.get(&q).is_some_and(|t| t.contains(&probe));
            if on(p) {
                Outcome::Direct
            } else if on(p.saturating_sub(1)) || on(p + 1) {
                Outcome::Near
            } else if anywhere {
                Outcome::Located
            } else {
                Outcome::MissInChapter // page stamp exists, passage gone
            }
        }
        None => {
            if anywhere {
                Outcome::ChapterlessLocated
            } else {
                Outcome::ChapterlessMiss
            }
        }
    }
}

/// Display family for a raw stored format string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    Book,
    BookUnverifiable, // mobi/azw3
    Text,             // md/txt (raw-file render path)
    Pdf,
    Unverified, // html/office/pages/djvu — different render pipelines
    Unknown,
}

pub fn family_of(format: &str) -> Family {
    match format {
        "epub" | "fb2" | "fb2.zip" => Family::Book,
        "mobi" | "azw3" => Family::BookUnverifiable,
        "md" | "markdown" | "txt" | "text" => Family::Text,
        "pdf" => Family::Pdf,
        "html" | "htm" | "docx" | "rtf" | "odt" | "doc" | "pages" | "webarchive" | "djvu"
        | "rst" | "adoc" | "org" | "tex" | "ipynb" | "xps" => Family::Unverified,
        _ => Family::Unknown,
    }
}

/// RU/EN stratum from the book title's Cyrillic letter fraction (title is the
/// only text available in the metadata pass; documented proxy).
pub fn script_of(title: &str) -> &'static str {
    let letters: Vec<char> = title.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.is_empty() {
        return "en";
    }
    let cyr = letters
        .iter()
        .filter(|c| ('\u{0400}'..='\u{04FF}').contains(*c))
        .count();
    // Ties (mixed titles like "Чистый Python") count as ru — the body text
    // of such books is Russian.
    if cyr * 2 >= letters.len() {
        "ru"
    } else {
        "en"
    }
}

/// FNV-1a over a chunk/book id XOR a fixed seed — deterministic, dependency-
/// free, keyed on identity (survives lance compaction and version churn;
/// std's DefaultHasher is version-unstable and unfit for a baseline).
pub fn fnv1a(key: &str, seed: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325 ^ seed;
    for b in key.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_text_matches_frontend_semantics() {
        // Soft hyphen stripped; ASCII hyphenation joined; whitespace collapsed.
        assert_eq!(norm_text("cash\u{00AD}flow"), "cashflow");
        assert_eq!(norm_text("data-\n  base systems"), "database systems");
        assert_eq!(norm_text("  a\t b\nc "), "a b c");
        // JS \w is ASCII: Cyrillic hyphenation must NOT join (frontend parity).
        assert_eq!(norm_text("програм-\nмирование"), "програм- мирование");
    }

    #[test]
    fn norm_chapter_strips_prefixes() {
        assert_eq!(norm_chapter("Chapter 3: Sagas"), "sagas");
        assert_eq!(norm_chapter("Глава 12. Репликация"), "репликация");
        assert_eq!(norm_chapter("2.4 — Consensus"), "consensus");
        assert_eq!(norm_chapter("  Intro  "), "intro");
        // §17.2c roman numerals — publishers' nav docs number in roman where
        // fitz TOCs use arabic.
        assert_eq!(norm_chapter("Part V. Appendixes"), "appendixes");
        assert_eq!(
            norm_chapter("Part I. Simplify Your Projects"),
            "simplify your projects"
        );
        assert_eq!(
            norm_chapter("I. Understanding RESTful Hypermedia"),
            "understanding restful hypermedia"
        );
        // Words starting with roman letters must survive.
        assert_eq!(norm_chapter("Ideas and Index"), "ideas and index");
    }

    #[test]
    fn find_label_tiers_handle_observed_drift() {
        // Real §17.2c drift pairs (stored fitz label vs foliate nav label).
        let toc = [
            "Change History",
            "Part I. Simplify Your Projects",
            "1. Orient; Step; Learn",
            "Part V. Appendixes",
            "3. Simplify Your Projects",
        ];
        let find = |stored: &str| find_label(toc.iter().copied(), stored);
        // Tier 1 exact (after roman stripping both sides).
        assert_eq!(find("Part 5: Appendixes"), Some(3));
        // Tier 3 token overlap: fitz broke the word ("Pa rt 1") — the tokens
        // {simplify, your, projects} still pin the right entry.
        assert_eq!(find("Pa rt 1 Simplify Your Projects"), Some(1));
        // Plain titles still exact-match.
        assert_eq!(find("Orient; Step; Learn"), Some(2));
        // Garbage matches nothing.
        assert_eq!(find("Completely Unrelated Heading Words"), None);
    }

    #[test]
    fn probe_skips_junk_prefix_with_at_zero_fallback() {
        // Junk prefix followed by 8 wordy words → skips the junk.
        let cite = "() } ; the quick brown foxes jump over lazy dogs today";
        assert_eq!(probe_of(cite), "the quick brown foxes jump over lazy dogs");
        // No run of 8 wordy words anywhere → falls back to the first 8 tokens.
        let junk = "x1 y2 } { ~ q4 z9 !! ?? aa";
        assert_eq!(probe_of(junk), "x1 y2 } { ~ q4 z9 !!");
        // Shorter than 8 words → whole thing.
        assert_eq!(probe_of("only three words"), "only three words");
    }

    #[test]
    fn md_blocks_join_inline_fragments_like_text_content() {
        let blocks = md_blocks("# Head\n\npara with **bold part** and `code` end [1]\n\n```\nfence line\n```\n- item one");
        assert_eq!(
            blocks,
            vec![
                "Head",
                "para with bold part and code end ",
                "fence line",
                "item one"
            ]
        );
        // The §17.2b 23%-miss case: a needle spanning a bold boundary now
        // matches the block's re-joined textContent.
        assert!(md_match("para with bold part and code", &blocks));
        assert!(md_match("bold part", &blocks));
        assert!(!md_match("text that appears nowhere in the doc", &blocks));
        // Heading-skip fallback: chunk text starting with its section heading
        // (own block in the DOM) matches via the second candidate needle.
        assert!(md_match("Head\npara with bold part and code end", &blocks));
    }

    fn doc_of(blocks: Vec<ls_core::Block>) -> ls_core::BookDoc {
        ls_core::BookDoc {
            book_id: "b".into(),
            title: "t".into(),
            author: None,
            source_path: "/lib/t.epub".into(),
            format: ls_core::Format::Epub,
            blocks,
        }
    }

    #[test]
    fn book_classification_taxonomy() {
        use ls_core::Block;
        let doc = doc_of(vec![
            Block::new(
                "Alpha beta gamma delta epsilon zeta eta theta",
                Some("Chapter 1: Intro".into()),
                None,
            ),
            Block::new(
                "Second chapter body words entirely different content here",
                Some("Chapter 2: Sagas".into()),
                None,
            ),
        ]);
        let bt = BookText::from_doc(&doc);
        // Probe within the named chapter's first run → Direct.
        assert_eq!(
            classify_book(
                Some("Intro"),
                "Alpha beta gamma delta epsilon zeta eta theta",
                &bt
            ),
            Outcome::Direct
        );
        // Right text, wrong chapter label → Located (phase-2 hit).
        assert_eq!(
            classify_book(
                Some("Sagas"),
                "Alpha beta gamma delta epsilon zeta eta theta",
                &bt
            ),
            Outcome::Located
        );
        // Chapter resolves, probe nowhere → MissInChapter.
        assert_eq!(
            classify_book(
                Some("Intro"),
                "totally absent probe words go here now ok",
                &bt
            ),
            Outcome::MissInChapter
        );
        // No label match, probe nowhere → ColdMiss.
        assert_eq!(
            classify_book(
                Some("Nope"),
                "totally absent probe words go here now ok",
                &bt
            ),
            Outcome::ColdMiss
        );
        // Chapterless chunk rows.
        assert_eq!(
            classify_book(
                None,
                "Second chapter body words entirely different content here",
                &bt
            ),
            Outcome::ChapterlessLocated
        );
    }

    #[test]
    fn pdf_on_page_and_near() {
        use ls_core::Block;
        let doc = doc_of(vec![
            Block::new(
                "first page words for the probe matching test here",
                None,
                Some(1),
            ),
            Block::new(
                "second page other content lives here entirely now yes",
                None,
                Some(2),
            ),
        ]);
        let bt = BookText::from_doc(&doc);
        let cite = "first page words for the probe matching test";
        assert_eq!(classify_pdf(Some(1), cite, &bt), Outcome::Direct);
        assert_eq!(classify_pdf(Some(2), cite, &bt), Outcome::Near);
        assert_eq!(classify_pdf(Some(9), cite, &bt), Outcome::Located);
    }

    #[test]
    fn sampling_hash_is_stable() {
        assert_eq!(fnv1a("book-1", 42), fnv1a("book-1", 42));
        assert_ne!(fnv1a("book-1", 42), fnv1a("book-2", 42));
        assert_ne!(fnv1a("book-1", 42), fnv1a("book-1", 43));
    }

    #[test]
    fn family_and_script_strata() {
        assert_eq!(family_of("epub"), Family::Book);
        assert_eq!(family_of("mobi"), Family::BookUnverifiable);
        assert_eq!(family_of("md"), Family::Text);
        assert_eq!(family_of("docx"), Family::Unverified);
        assert_eq!(family_of("weird"), Family::Unknown);
        assert_eq!(script_of("Чистый Python"), "ru");
        assert_eq!(script_of("Effective Kotlin"), "en");
    }
}
