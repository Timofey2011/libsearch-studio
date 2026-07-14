//! Ebook extraction (ROADMAP-3 §4.1): epub via `rbook`, fb2/fb2.zip via
//! `quick-xml` (+`zip`), mobi/azw3 via the `mobi` crate — all pure Rust.
//!
//! Chapter mapping follows the M1 anti-flood rules: only top-two-level TOC
//! entries / sections become `chapter`, and the shared section floor merges
//! fragments so heading-dense books can't mint thousands of tiny chunks.

use std::collections::HashMap;
use std::path::Path;

use ls_core::{Block, BookDoc, Format};

use crate::text::{apply_section_floor, Section};
use crate::{stable_book_id, ExtractError};

/// Rendering width for html→text conversion (long enough that reflow never
/// splits sentences mid-word in practice; the chunker re-wraps anyway).
const H2T_WIDTH: usize = 200;

fn title_of(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "Untitled".to_string())
}

fn strip_fragment(href: &str) -> String {
    href.split('#').next().unwrap_or(href).to_string()
}

fn html_to_text(html: &str) -> String {
    html2text::from_read(html.as_bytes(), H2T_WIDTH).unwrap_or_default()
}

// ---- epub -------------------------------------------------------------------

fn extract_epub(path: &Path) -> Result<BookDoc, ExtractError> {
    use rbook::ebook::manifest::ManifestEntry;
    use rbook::ebook::metadata::{MetaEntry, Metadata};
    use rbook::ebook::spine::{Spine, SpineEntry};
    use rbook::ebook::toc::{Toc, TocChildren, TocEntry};
    use rbook::ebook::Ebook;
    use rbook::Epub;

    let epub = Epub::open(path).map_err(|e| ExtractError::Parse(format!("epub: {e}")))?;

    // TOC labels by target document href (fragment stripped), top two levels
    // only — deeper nav points stay unlabeled per the Index-flood rule.
    let mut toc_by_href: HashMap<String, String> = HashMap::new();
    let toc = epub.toc();
    if let Some(root) = toc.contents() {
        for entry in root.children().flatten() {
            if entry.depth() > 2 {
                continue;
            }
            let label = entry.label().trim().to_string();
            if label.is_empty() {
                continue;
            }
            if let Some(me) = entry.manifest_entry() {
                toc_by_href
                    .entry(strip_fragment(me.href().as_str()))
                    .or_insert(label);
            }
        }
    }

    let author = {
        let meta = epub.metadata();
        meta.creators()
            .next()
            .map(|c| c.value().trim().to_string())
            .filter(|s| !s.is_empty())
    };
    let title = {
        let meta = epub.metadata();
        meta.title()
            .map(|t| t.value().trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| title_of(path))
    };

    // One section per spine item; the chapter label carries forward until the
    // next TOC-labeled item (front matter before the first label stays None).
    let mut sections: Vec<Section> = Vec::new();
    let mut current: Option<String> = None;
    for item in epub.spine().entries() {
        let Some(me) = item.manifest_entry() else {
            continue;
        };
        let href = strip_fragment(me.href().as_str());
        if let Some(label) = toc_by_href.get(&href) {
            current = Some(label.clone());
        }
        let Ok(xhtml) = me.read_str() else { continue };
        let text = html_to_text(&xhtml).trim().to_string();
        if text.is_empty() {
            continue;
        }
        sections.push(Section {
            heading: current.clone(),
            body: text,
        });
    }

    Ok(BookDoc {
        book_id: stable_book_id(path),
        title,
        author,
        source_path: path.to_string_lossy().to_string(),
        format: Format::Epub,
        blocks: sections_to_blocks(sections),
    })
}

// ---- fb2 --------------------------------------------------------------------

/// Sections + author + book title from an FB2 document.
type Fb2Parts = (Vec<Section>, Option<String>, Option<String>);

/// Parse FB2 XML (possibly windows-1251 — quick-xml's `encoding` feature
/// honors the XML declaration) into sections + author/title.
fn parse_fb2(bytes: &[u8]) -> Result<Fb2Parts, ExtractError> {
    use quick_xml::events::Event;

    let mut reader = quick_xml::Reader::from_reader(bytes);
    reader.config_mut().trim_text(true);

    let mut sections: Vec<Section> = Vec::new();
    let mut section_depth = 0usize;
    let mut in_title = false;
    let mut in_title_info = false;
    let mut author_tag = false;
    let mut book_title_tag = false;
    let mut title_buf = String::new();
    let mut author_parts: Vec<String> = Vec::new();
    let mut book_title: Option<String> = None;
    let mut buf = Vec::new();

    let ensure_section = |sections: &mut Vec<Section>| {
        if sections.is_empty() {
            sections.push(Section {
                heading: None,
                body: String::new(),
            });
        }
    };

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => match e.local_name().as_ref() {
                b"section" => section_depth += 1,
                b"title" if section_depth > 0 => {
                    in_title = true;
                    title_buf.clear();
                }
                b"title-info" => in_title_info = true,
                b"author" if in_title_info => author_tag = true,
                b"book-title" if in_title_info => book_title_tag = true,
                _ => {}
            },
            Ok(Event::End(e)) => match e.local_name().as_ref() {
                b"section" => section_depth = section_depth.saturating_sub(1),
                b"title" if in_title => {
                    in_title = false;
                    let label = title_buf.split_whitespace().collect::<Vec<_>>().join(" ");
                    // Depth cap: only top-two-level section titles open chapters.
                    if !label.is_empty() && section_depth <= 2 {
                        sections.push(Section {
                            heading: Some(label),
                            body: String::new(),
                        });
                    } else if !label.is_empty() {
                        ensure_section(&mut sections);
                        let cur = sections.last_mut().unwrap();
                        cur.body.push_str(&label);
                        cur.body.push('\n');
                    }
                }
                b"title-info" => in_title_info = false,
                b"author" => author_tag = false,
                b"book-title" => book_title_tag = false,
                b"p" | b"v" | b"subtitle" => {
                    ensure_section(&mut sections);
                    sections.last_mut().unwrap().body.push('\n');
                }
                _ => {}
            },
            Ok(Event::Text(t)) => {
                let txt = t.decode().unwrap_or_default();
                let txt = txt.trim();
                if txt.is_empty() {
                    // skip
                } else if in_title {
                    title_buf.push_str(txt);
                    title_buf.push(' ');
                } else if book_title_tag {
                    book_title = Some(txt.to_string());
                } else if author_tag {
                    author_parts.push(txt.to_string());
                } else if section_depth > 0 {
                    ensure_section(&mut sections);
                    let cur = sections.last_mut().unwrap();
                    cur.body.push_str(txt);
                    cur.body.push(' ');
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(ExtractError::Parse(format!("fb2: {e}"))),
            _ => {}
        }
        buf.clear();
    }

    let author = if author_parts.is_empty() {
        None
    } else {
        Some(author_parts.join(" "))
    };
    Ok((sections, author, book_title))
}

fn extract_fb2(path: &Path) -> Result<BookDoc, ExtractError> {
    let name = path.to_string_lossy();
    let bytes = if ls_core::ext_of(&name) == Some("fb2.zip") {
        // A single-entry zip holding the .fb2.
        let f = std::fs::File::open(path).map_err(|e| ExtractError::Io(e.to_string()))?;
        let mut ar =
            zip::ZipArchive::new(f).map_err(|e| ExtractError::Parse(format!("fb2.zip: {e}")))?;
        let idx = (0..ar.len())
            .find(|&i| {
                ar.by_index(i)
                    .map(|e| e.name().to_ascii_lowercase().ends_with(".fb2"))
                    .unwrap_or(false)
            })
            .unwrap_or(0);
        let mut entry = ar
            .by_index(idx)
            .map_err(|e| ExtractError::Parse(format!("fb2.zip: {e}")))?;
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut entry, &mut out)
            .map_err(|e| ExtractError::Io(e.to_string()))?;
        out
    } else {
        std::fs::read(path).map_err(|e| ExtractError::Io(e.to_string()))?
    };

    let (sections, author, book_title) = parse_fb2(&bytes)?;
    Ok(BookDoc {
        book_id: stable_book_id(path),
        title: book_title.unwrap_or_else(|| title_of(path)),
        author,
        source_path: name.to_string(),
        format: Format::Fb2,
        blocks: sections_to_blocks(sections),
    })
}

// ---- mobi -------------------------------------------------------------------

fn extract_mobi(path: &Path) -> Result<BookDoc, ExtractError> {
    use mobi::headers::Encryption;

    let m = mobi::Mobi::from_path(path).map_err(|e| {
        ExtractError::Parse(format!(
            "mobi parse failed — handled by Fast (GPU) indexing where available ({e})"
        ))
    })?;
    if !matches!(m.metadata.encryption(), Encryption::No) {
        return Err(ExtractError::Parse("DRM-protected — cannot index".into()));
    }
    let html = m.content_as_string().map_err(|e| {
        ExtractError::Parse(format!(
            "mobi parse failed — handled by Fast (GPU) indexing where available ({e})"
        ))
    })?;
    let text = html_to_text(&html).trim().to_string();
    // Chapterless: citations fall back to the ~loc shape.
    let sections = vec![Section {
        heading: None,
        body: text,
    }];
    let author = Some(m.author().unwrap_or_default())
        .filter(|s: &String| !s.trim().is_empty())
        .map(|s| s.trim().to_string());
    let title = {
        let t = m.title();
        if t.trim().is_empty() {
            title_of(path)
        } else {
            t.trim().to_string()
        }
    };
    Ok(BookDoc {
        book_id: stable_book_id(path),
        title,
        author,
        source_path: path.to_string_lossy().to_string(),
        format: Format::Mobi,
        blocks: sections_to_blocks(sections),
    })
}

// ---- shared -----------------------------------------------------------------

fn sections_to_blocks(sections: Vec<Section>) -> Vec<Block> {
    let sections = apply_section_floor(sections);
    let mut blocks: Vec<Block> = sections
        .into_iter()
        .filter_map(|s| {
            let text = s.body.trim().to_string();
            if text.is_empty() {
                return None;
            }
            Some(Block::new(text, s.heading, None))
        })
        .collect();
    let total: usize = blocks.iter().map(|b| b.text.chars().count()).sum();
    if total < crate::MIN_BOOK_CHARS {
        blocks = Vec::new();
    }
    blocks
}

/// The CPU pipeline cannot handle these extensions; the reason is recorded to
/// skip_state (pipeline=cpu) and worded platform-honestly — the GPU helper
/// picks them up where it exists (ROADMAP-3 §4.1/§8).
pub fn cpu_directed_skip(ext: &str) -> Option<&'static str> {
    match ext {
        "xps" => Some("xps: handled by Fast (GPU) indexing where available"),
        _ => None,
    }
}

/// Dispatch for the ebook family.
pub fn extract_ebook(path: &Path) -> Result<BookDoc, ExtractError> {
    let name = path.to_string_lossy();
    match ls_core::ext_of(&name) {
        Some("epub") => extract_epub(path),
        Some("fb2") | Some("fb2.zip") => extract_fb2(path),
        Some("mobi") | Some("azw3") => extract_mobi(path),
        other => Err(ExtractError::Unsupported(format!(
            "{name} ({})",
            other.unwrap_or("no ext")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("ls-extract-ebook-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn ru_body(n: usize) -> String {
        "Проверка русского текста в книге. ".repeat(n)
    }

    fn fb2_xml() -> String {
        format!(
            r#"<?xml version="1.0" encoding="windows-1251"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
<description><title-info>
  <author><first-name>Лев</first-name><last-name>Толстой</last-name></author>
  <book-title>Тестовая книга</book-title>
</title-info></description>
<body>
  <section><title><p>Глава первая</p></title>
    <p>{b}</p>
    <section><title><p>Часть 1.1</p></title><p>{b}</p></section>
  </section>
  <section><title><p>Глава вторая</p></title><p>{b}</p></section>
</body>
</FictionBook>"#,
            b = ru_body(20)
        )
    }

    #[test]
    fn fb2_cp1251_nested_sections_author_title() {
        let xml = fb2_xml();
        let (encoded, _, _) = encoding_rs::WINDOWS_1251.encode(&xml);
        let p = write_tmp("kniga.fb2", &encoded);
        let doc = extract_ebook(&p).unwrap();
        assert_eq!(doc.format, ls_core::Format::Fb2);
        assert_eq!(doc.title, "Тестовая книга");
        assert_eq!(doc.author.as_deref(), Some("Лев Толстой"));
        let chapters: Vec<_> = doc
            .blocks
            .iter()
            .filter_map(|b| b.chapter.as_deref())
            .collect();
        assert!(chapters.contains(&"Глава первая"), "{chapters:?}");
        assert!(chapters.contains(&"Глава вторая"), "{chapters:?}");
        assert!(chapters.contains(&"Часть 1.1"), "nested level-2 kept");
        assert!(doc.blocks[0].text.contains("русского текста"));
    }

    #[test]
    fn fb2_zip_resolves_through_compound_ext() {
        let xml = fb2_xml();
        let (encoded, _, _) = encoding_rs::WINDOWS_1251.encode(&xml);
        let mut zbuf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(std::io::Cursor::new(&mut zbuf));
            zw.start_file::<_, ()>(
                "kniga.fb2",
                zip::write::FileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )
            .unwrap();
            std::io::Write::write_all(&mut zw, &encoded).unwrap();
            zw.finish().unwrap();
        }
        let p = write_tmp("kniga.fb2.zip", &zbuf);
        let doc = extract_ebook(&p).unwrap();
        assert_eq!(doc.format, ls_core::Format::Fb2);
        assert_eq!(doc.author.as_deref(), Some("Лев Толстой"));
        assert!(!doc.blocks.is_empty());
    }

    #[test]
    fn garbage_mobi_reports_gpu_fallback_reason() {
        let p = write_tmp("broken.mobi", b"this is not a mobi file at all");
        let err = extract_ebook(&p).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("handled by Fast (GPU) indexing where available"),
            "{msg}"
        );
    }

    #[test]
    fn xps_is_a_cpu_directed_skip() {
        assert!(cpu_directed_skip("xps")
            .unwrap()
            .contains("where available"));
        assert!(cpu_directed_skip("epub").is_none());
    }
}
