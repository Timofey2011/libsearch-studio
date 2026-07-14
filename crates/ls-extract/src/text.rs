//! Text-family extraction (ROADMAP-3 §3.1/§3.2): md, txt, rst, adoc, org,
//! tex, ipynb, html — all pure Rust, all emitting `Block{text, chapter,
//! page: None}`.
//!
//! Two design rules, applied at EXTRACTION time so the chunker's
//! no-cross-chapter rule is never violated downstream:
//! - **Depth cap:** only top-two-level headings (H1–H2 and equivalents)
//!   become `chapter`; deeper headings stay in-body. Otherwise every
//!   subsection of every note floods the Index tab.
//! - **Section floor:** a section shorter than [`SECTION_FLOOR_CHARS`] merges
//!   into the PREVIOUS section (keeping the earlier `chapter` label, inlining
//!   its own heading as body text). Otherwise heading-dense notes mint
//!   thousands of tiny low-quality chunks.

use std::path::Path;

use ls_core::{Block, BookDoc, Format};

use crate::{stable_book_id, ExtractError};

/// ~120 tokens at the usual 4 chars/token — sections below this merge into
/// their predecessor. Char-based on purpose: the real tokenizer lives in
/// ls-embed and extraction must stay model-free.
pub const SECTION_FLOOR_CHARS: usize = 480;

struct Section {
    heading: Option<String>,
    body: String,
}

/// Read a file and decode it: strict UTF-8 first, then charset detection
/// (chardetng — covers RU windows-1251 txt), then latin-1 as the lossless
/// last resort.
fn read_decoded(path: &Path) -> Result<String, ExtractError> {
    let bytes = std::fs::read(path).map_err(|e| ExtractError::Io(e.to_string()))?;
    if let Ok(s) = String::from_utf8(bytes.clone()) {
        return Ok(s);
    }
    let mut det = chardetng::EncodingDetector::new();
    det.feed(&bytes, true);
    let enc = det.guess(None, true);
    let (decoded, _, _) = enc.decode(&bytes);
    Ok(decoded.into_owned())
}

/// Title for notebook/plain files: the file stem.
fn title_of(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "Untitled".to_string())
}

// ---- per-format section scanners -------------------------------------------

/// Markdown: `# ` / `## ` open a new section; deeper headings stay in-body.
fn scan_md(text: &str) -> Vec<Section> {
    let mut sections = vec![Section {
        heading: None,
        body: String::new(),
    }];
    for line in text.lines() {
        let trimmed = line.trim_start();
        let hashes = trimmed.chars().take_while(|c| *c == '#').count();
        if (1..=2).contains(&hashes)
            && trimmed[hashes..].starts_with(' ')
            && !trimmed[hashes + 1..].trim().is_empty()
        {
            sections.push(Section {
                heading: Some(trimmed[hashes + 1..].trim().to_string()),
                body: String::new(),
            });
        } else {
            let cur = sections.last_mut().unwrap();
            cur.body.push_str(line);
            cur.body.push('\n');
        }
    }
    sections
}

/// reStructuredText: a non-empty line followed by an underline of `=` or `-`
/// (at least 3 chars, at least as long as the title) is a heading.
fn scan_rst(text: &str) -> Vec<Section> {
    let lines: Vec<&str> = text.lines().collect();
    let mut sections = vec![Section {
        heading: None,
        body: String::new(),
    }];
    let mut i = 0;
    while i < lines.len() {
        let is_heading = i + 1 < lines.len() && {
            let title = lines[i].trim();
            let under = lines[i + 1].trim();
            !title.is_empty()
                && under.len() >= 3
                && under.len() + 1 >= title.len()
                && (under.chars().all(|c| c == '=') || under.chars().all(|c| c == '-'))
        };
        if is_heading {
            sections.push(Section {
                heading: Some(lines[i].trim().to_string()),
                body: String::new(),
            });
            i += 2;
        } else {
            let cur = sections.last_mut().unwrap();
            cur.body.push_str(lines[i]);
            cur.body.push('\n');
            i += 1;
        }
    }
    sections
}

/// Line-prefix heading scanners (AsciiDoc `= `/`== `, Org `* `/`** `).
fn scan_prefix(text: &str, l1: &str, l2: &str) -> Vec<Section> {
    let mut sections = vec![Section {
        heading: None,
        body: String::new(),
    }];
    for line in text.lines() {
        let heading = line.strip_prefix(l2).or_else(|| line.strip_prefix(l1));
        match heading {
            Some(h) if !h.trim().is_empty() => sections.push(Section {
                heading: Some(h.trim().to_string()),
                body: String::new(),
            }),
            _ => {
                let cur = sections.last_mut().unwrap();
                cur.body.push_str(line);
                cur.body.push('\n');
            }
        }
    }
    sections
}

/// LaTeX: `%` comments stripped; `\chapter{X}` / `\section{X}` open sections.
fn scan_tex(text: &str) -> Vec<Section> {
    let mut sections = vec![Section {
        heading: None,
        body: String::new(),
    }];
    for raw in text.lines() {
        // Strip an unescaped % comment.
        let mut line = raw;
        if let Some(pos) = raw
            .char_indices()
            .find(|(i, c)| *c == '%' && (*i == 0 || raw.as_bytes()[i - 1] != b'\\'))
            .map(|(i, _)| i)
        {
            line = &raw[..pos];
        }
        let trimmed = line.trim_start();
        let heading = ["\\chapter{", "\\section{"].iter().find_map(|cmd| {
            trimmed
                .strip_prefix(cmd)
                .and_then(|rest| rest.split_once('}'))
                .map(|(h, _)| h.trim().to_string())
        });
        match heading {
            Some(h) if !h.is_empty() => sections.push(Section {
                heading: Some(h),
                body: String::new(),
            }),
            _ => {
                let cur = sections.last_mut().unwrap();
                cur.body.push_str(line);
                cur.body.push('\n');
            }
        }
    }
    sections
}

/// Jupyter notebook: markdown cells pass through the md scanner rules; code
/// cells become fenced blocks in the current section.
fn scan_ipynb(raw: &str) -> Result<Vec<Section>, ExtractError> {
    let v: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| ExtractError::Parse(format!("ipynb: {e}")))?;
    // Reassemble as one markdown-ish stream, then reuse the md scanner.
    let mut md = String::new();
    for cell in v["cells"].as_array().unwrap_or(&Vec::new()) {
        let source = match &cell["source"] {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p.as_str())
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        if source.trim().is_empty() {
            continue;
        }
        match cell["cell_type"].as_str() {
            Some("markdown") => {
                md.push_str(&source);
                md.push('\n');
            }
            Some("code") => {
                md.push_str("```\n");
                md.push_str(&source);
                md.push_str("\n```\n");
            }
            _ => {}
        }
    }
    Ok(scan_md(&md))
}

/// HTML: DOM walk collecting visible text; `<h1>/<h2>` open sections;
/// script/style/nav/head chrome is dropped.
fn scan_html(raw: &str) -> Vec<Section> {
    use scraper::{Html, Node};

    let doc = Html::parse_document(raw);
    let mut sections = vec![Section {
        heading: None,
        body: String::new(),
    }];

    const SKIP: &[&str] = &[
        "script", "style", "nav", "head", "noscript", "svg", "iframe",
    ];
    const BLOCK: &[&str] = &[
        "p",
        "div",
        "li",
        "tr",
        "br",
        "h3",
        "h4",
        "h5",
        "h6",
        "section",
        "article",
        "blockquote",
        "pre",
    ];

    fn heading_text(el: ego_tree::NodeRef<Node>) -> String {
        let mut out = String::new();
        for d in el.descendants() {
            if let Node::Text(t) = d.value() {
                out.push_str(&t.text);
            }
        }
        out.trim().to_string()
    }

    fn walk(node: ego_tree::NodeRef<Node>, sections: &mut Vec<Section>) {
        for child in node.children() {
            match child.value() {
                Node::Element(el) => {
                    let name = el.name();
                    if SKIP.contains(&name) {
                        continue;
                    }
                    if name == "h1" || name == "h2" {
                        let h = heading_text(child);
                        if !h.is_empty() {
                            sections.push(Section {
                                heading: Some(h),
                                body: String::new(),
                            });
                        }
                        continue; // heading text lives in the label, not the body
                    }
                    walk(child, sections);
                    if BLOCK.contains(&name) {
                        sections.last_mut().unwrap().body.push('\n');
                    }
                }
                Node::Text(t) => {
                    let cur = sections.last_mut().unwrap();
                    cur.body.push_str(&t.text);
                }
                _ => {}
            }
        }
    }

    walk(doc.tree.root(), &mut sections);
    sections
}

// ---- assembly ---------------------------------------------------------------

/// Merge sections below the floor into their predecessor: the earlier section
/// keeps its `chapter` label; the small heading is inlined as body text.
fn apply_section_floor(sections: Vec<Section>) -> Vec<Section> {
    let mut out: Vec<Section> = Vec::new();
    for s in sections {
        let too_small = s.body.trim().len() < SECTION_FLOOR_CHARS;
        match out.last_mut() {
            Some(prev) if too_small => {
                if let Some(h) = &s.heading {
                    prev.body.push('\n');
                    prev.body.push_str(h);
                    prev.body.push('\n');
                }
                prev.body.push_str(&s.body);
            }
            _ => out.push(s),
        }
    }
    out
}

/// Extract any text-family file (dispatched on the canonical extension) into
/// a `BookDoc`. Returns empty `blocks` when the file has too little text —
/// the caller records a skip.
pub fn extract_text_family(path: &Path) -> Result<BookDoc, ExtractError> {
    let name = path.to_string_lossy();
    let ext = ls_core::ext_of(&name).ok_or_else(|| ExtractError::Unsupported(name.to_string()))?;
    let raw = read_decoded(path)?;

    let sections = match ext {
        "md" | "markdown" => scan_md(&raw),
        "txt" | "text" => vec![Section {
            heading: None,
            body: raw.clone(),
        }],
        "rst" => scan_rst(&raw),
        "adoc" => scan_prefix(&raw, "= ", "== "),
        "org" => scan_prefix(&raw, "* ", "** "),
        "tex" => scan_tex(&raw),
        "ipynb" => scan_ipynb(&raw)?,
        "html" | "htm" => scan_html(&raw),
        other => return Err(ExtractError::Unsupported(other.to_string())),
    };
    let sections = apply_section_floor(sections);

    let mut blocks: Vec<Block> = sections
        .into_iter()
        .filter_map(|s| {
            let text = s.body.trim().to_string();
            if text.is_empty() && s.heading.is_none() {
                return None;
            }
            let body = if text.is_empty() {
                s.heading.clone().unwrap_or_default()
            } else {
                text
            };
            Some(Block::new(body, s.heading, None))
        })
        .collect();

    let total: usize = blocks.iter().map(|b| b.text.chars().count()).sum();
    if total < crate::MIN_BOOK_CHARS {
        blocks = Vec::new();
    }

    Ok(BookDoc {
        book_id: stable_book_id(path),
        title: title_of(path),
        author: None,
        source_path: name.to_string(),
        format: Format::from_ext(ext).unwrap_or(Format::Txt),
        blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("ls-extract-text-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn body(n: usize) -> String {
        "слово word ".repeat(n)
    }

    #[test]
    fn md_h1_h2_become_chapters_deeper_stay_body() {
        let text = format!(
            "intro before any heading {b}\n# One\n{b}\n### Sub\n{b}\n## Two\n{b}",
            b = body(60)
        );
        let p = write_tmp("a.md", text.as_bytes());
        let doc = extract_text_family(&p).unwrap();
        let chapters: Vec<_> = doc.blocks.iter().map(|b| b.chapter.clone()).collect();
        assert_eq!(
            chapters,
            vec![None, Some("One".into()), Some("Two".into())],
            "H3 must not open a chapter"
        );
        assert!(doc.blocks[1].text.contains("Sub"), "H3 text stays in-body");
        assert_eq!(doc.format, Format::Md);
    }

    #[test]
    fn heading_dense_md_respects_section_floor() {
        // 30 tiny sections (ByteByteGo-lesson shape): every one below the
        // floor must merge; no block may end up shorter than the floor except
        // the last resort single block.
        let mut text = String::new();
        for i in 0..30 {
            text.push_str(&format!("## Point {i}\nshort note {i}.\n"));
        }
        text.push_str(&body(80));
        let p = write_tmp("dense.md", text.as_bytes());
        let doc = extract_text_family(&p).unwrap();
        assert!(
            doc.blocks.len() <= 2,
            "tiny sections must merge, got {} blocks",
            doc.blocks.len()
        );
        for b in &doc.blocks {
            assert!(
                b.text.len() >= SECTION_FLOOR_CHARS,
                "block below the section floor: {} chars",
                b.text.len()
            );
        }
    }

    #[test]
    fn cp1251_txt_decodes() {
        let (encoded, _, _) = encoding_rs::WINDOWS_1251.encode(
            "Проверка русского текста в кодировке windows-1251. Это обычный текстовый файл, \
             который должен корректно распознаться и декодироваться без ошибок чтения. \
             Дополнительный текст для преодоления минимального порога длины книги: \
             приложение индексирует только файлы с достаточным объёмом полезного текста.",
        );
        let p = write_tmp("ru.txt", &encoded);
        let doc = extract_text_family(&p).unwrap();
        assert_eq!(doc.blocks.len(), 1);
        assert!(doc.blocks[0].text.contains("русского текста"));
        assert_eq!(doc.format, Format::Txt);
    }

    #[test]
    fn rst_adoc_org_tex_headings() {
        let rst = format!(
            "Title One\n=========\n{b}\nPart Two\n--------\n{b}",
            b = body(60)
        );
        let p = write_tmp("a.rst", rst.as_bytes());
        let doc = extract_text_family(&p).unwrap();
        assert_eq!(doc.blocks[0].chapter.as_deref(), Some("Title One"));
        assert_eq!(doc.blocks[1].chapter.as_deref(), Some("Part Two"));

        let adoc = format!("= Doc\n{b}\n== Sect\n{b}", b = body(60));
        let p = write_tmp("a.adoc", adoc.as_bytes());
        let doc = extract_text_family(&p).unwrap();
        assert_eq!(doc.blocks[0].chapter.as_deref(), Some("Doc"));

        let org = format!("* Top\n{b}\n** Nested\n{b}", b = body(60));
        let p = write_tmp("a.org", org.as_bytes());
        let doc = extract_text_family(&p).unwrap();
        assert_eq!(doc.blocks[0].chapter.as_deref(), Some("Top"));

        let tex = format!(
            "% comment only\n\\chapter{{Alpha}}\n{b}\n\\section{{Beta}}\n{b}",
            b = body(60)
        );
        let p = write_tmp("a.tex", tex.as_bytes());
        let doc = extract_text_family(&p).unwrap();
        assert_eq!(doc.blocks[0].chapter.as_deref(), Some("Alpha"));
        assert!(!doc.blocks[0].text.contains("comment only"));
    }

    #[test]
    fn ipynb_cells_and_html_chrome() {
        let nb = serde_json::json!({
            "cells": [
                {"cell_type": "markdown", "source": ["# Lesson\n", &body(60)]},
                {"cell_type": "code", "source": "print('hi')"},
                {"cell_type": "markdown", "source": &body(60)}
            ]
        });
        let p = write_tmp("a.ipynb", nb.to_string().as_bytes());
        let doc = extract_text_family(&p).unwrap();
        assert_eq!(doc.blocks[0].chapter.as_deref(), Some("Lesson"));
        assert!(doc.blocks[0].text.contains("print('hi')"));
        assert_eq!(doc.format, Format::Md);

        let html = format!(
            "<html><head><title>t</title><script>bad()</script></head><body>\
             <nav>menu junk</nav><h1>Guide</h1><p>{b}</p><h2>Part</h2><p>{b}</p></body></html>",
            b = body(60)
        );
        let p = write_tmp("a.html", html.as_bytes());
        let doc = extract_text_family(&p).unwrap();
        assert_eq!(doc.blocks[0].chapter.as_deref(), Some("Guide"));
        assert_eq!(doc.blocks[1].chapter.as_deref(), Some("Part"));
        assert!(!doc.blocks.iter().any(|b| b.text.contains("bad()")));
        assert!(!doc.blocks.iter().any(|b| b.text.contains("menu junk")));
        assert_eq!(doc.format, Format::Html);
    }

    #[test]
    fn tiny_file_yields_empty_blocks() {
        let p = write_tmp("tiny.md", b"# x\nshort");
        let doc = extract_text_family(&p).unwrap();
        assert!(doc.blocks.is_empty(), "below MIN_BOOK_CHARS must skip");
    }
}
