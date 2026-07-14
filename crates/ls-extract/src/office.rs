//! Office-format extraction (ROADMAP-3 §6.1): docx and odt via hand-rolled
//! zip + quick-xml (~150 LoC each; the crate ecosystem is writer-focused or
//! abandoned), rtf via `rtf-parser` with a codepage pre-pass. All pure Rust —
//! Linux gets full parity.

use std::io::Read;
use std::path::Path;

use ls_core::{Block, BookDoc, Format};

use crate::text::{apply_section_floor, Section};
use crate::{stable_book_id, ExtractError};

fn title_of(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "Untitled".to_string())
}

fn read_zip_entry(path: &Path, entry: &str) -> Result<Vec<u8>, ExtractError> {
    let f = std::fs::File::open(path).map_err(|e| ExtractError::Io(e.to_string()))?;
    let mut ar = zip::ZipArchive::new(f).map_err(|e| ExtractError::Parse(format!("zip: {e}")))?;
    let mut file = ar
        .by_name(entry)
        .map_err(|e| ExtractError::Parse(format!("{entry}: {e}")))?;
    let mut out = Vec::new();
    file.read_to_end(&mut out)
        .map_err(|e| ExtractError::Io(e.to_string()))?;
    Ok(out)
}

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

// ---- docx -------------------------------------------------------------------

/// `word/document.xml`: `w:p` paragraphs of `w:t` runs; a `w:pStyle` of
/// Heading1/Heading2 turns the paragraph into a `chapter` (same depth cap as
/// markdown — deeper headings stay body text).
fn extract_docx(path: &Path) -> Result<BookDoc, ExtractError> {
    use quick_xml::events::Event;

    let xml = read_zip_entry(path, "word/document.xml")?;
    let mut reader = quick_xml::Reader::from_reader(xml.as_slice());
    let mut sections = vec![Section {
        heading: None,
        body: String::new(),
    }];
    let mut para = String::new();
    let mut para_heading = false;
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) if e.local_name().as_ref() == b"p" => {
                para.clear();
                para_heading = false;
            }
            Ok(Event::Empty(e)) if e.local_name().as_ref() == b"pStyle" => {
                let style = e
                    .attributes()
                    .flatten()
                    .find(|a| a.key.local_name().as_ref() == b"val")
                    .map(|a| String::from_utf8_lossy(&a.value).into_owned())
                    .unwrap_or_default();
                // Heading1/Heading2 (and locale variants like "Heading 1").
                let s = style.replace(' ', "");
                para_heading =
                    s.eq_ignore_ascii_case("heading1") || s.eq_ignore_ascii_case("heading2");
            }
            Ok(Event::Text(t)) => {
                para.push_str(&t.decode().unwrap_or_default());
            }
            Ok(Event::End(e)) if e.local_name().as_ref() == b"p" => {
                let text = para.trim();
                if !text.is_empty() {
                    if para_heading {
                        sections.push(Section {
                            heading: Some(text.to_string()),
                            body: String::new(),
                        });
                    } else {
                        let cur = sections.last_mut().unwrap();
                        cur.body.push_str(text);
                        cur.body.push('\n');
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(ExtractError::Parse(format!("docx: {e}"))),
            _ => {}
        }
        buf.clear();
    }

    Ok(BookDoc {
        book_id: stable_book_id(path),
        title: title_of(path),
        author: None,
        source_path: path.to_string_lossy().to_string(),
        format: Format::Docx,
        blocks: sections_to_blocks(sections),
    })
}

// ---- odt --------------------------------------------------------------------

/// `content.xml`: `text:h` with outline-level 1–2 opens a chapter; `text:p`
/// paragraphs accumulate.
fn extract_odt(path: &Path) -> Result<BookDoc, ExtractError> {
    use quick_xml::events::Event;

    let xml = read_zip_entry(path, "content.xml")?;
    let mut reader = quick_xml::Reader::from_reader(xml.as_slice());
    let mut sections = vec![Section {
        heading: None,
        body: String::new(),
    }];
    let mut cur_text = String::new();
    let mut in_heading_level: Option<u8> = None;
    let mut in_para = false;
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => match e.local_name().as_ref() {
                b"h" => {
                    let lvl: u8 = e
                        .attributes()
                        .flatten()
                        .find(|a| a.key.local_name().as_ref() == b"outline-level")
                        .and_then(|a| String::from_utf8_lossy(&a.value).parse().ok())
                        .unwrap_or(1);
                    in_heading_level = Some(lvl);
                    cur_text.clear();
                }
                b"p" => {
                    in_para = true;
                    cur_text.clear();
                }
                _ => {}
            },
            Ok(Event::Text(t)) => {
                if in_heading_level.is_some() || in_para {
                    cur_text.push_str(&t.decode().unwrap_or_default());
                }
            }
            Ok(Event::End(e)) => match e.local_name().as_ref() {
                b"h" => {
                    let text = cur_text.trim().to_string();
                    let lvl = in_heading_level.take().unwrap_or(1);
                    if !text.is_empty() {
                        if lvl <= 2 {
                            sections.push(Section {
                                heading: Some(text),
                                body: String::new(),
                            });
                        } else {
                            let cur = sections.last_mut().unwrap();
                            cur.body.push_str(&text);
                            cur.body.push('\n');
                        }
                    }
                }
                b"p" if in_para => {
                    in_para = false;
                    let text = cur_text.trim();
                    if !text.is_empty() {
                        let cur = sections.last_mut().unwrap();
                        cur.body.push_str(text);
                        cur.body.push('\n');
                    }
                }
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(e) => return Err(ExtractError::Parse(format!("odt: {e}"))),
            _ => {}
        }
        buf.clear();
    }

    Ok(BookDoc {
        book_id: stable_book_id(path),
        title: title_of(path),
        author: None,
        source_path: path.to_string_lossy().to_string(),
        format: Format::Odt,
        blocks: sections_to_blocks(sections),
    })
}

// ---- rtf --------------------------------------------------------------------

/// Decode `\'xx` byte escapes using the document's declared codepage
/// (`\ansicpgN`) BEFORE structural parsing — `rtf-parser` has no codepage
/// handling, and RU documents are typically `\ansicpg1251`.
fn decode_rtf_escapes(raw: &[u8]) -> String {
    let text = String::from_utf8_lossy(raw);
    let enc = if text.contains("\\ansicpg1251") {
        encoding_rs::WINDOWS_1251
    } else {
        // 1252 covers the default and every other single-byte western page
        // we care about; unknown pages degrade to it rather than mojibake-ing
        // the whole document.
        encoding_rs::WINDOWS_1252
    };
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 3 < bytes.len() && bytes[i + 1] == b'\'' {
            // Collect a RUN of \'xx escapes so multi-byte encodings survive.
            let mut run = Vec::new();
            while i + 3 < bytes.len() && bytes[i] == b'\\' && bytes[i + 1] == b'\'' {
                let hex = std::str::from_utf8(&bytes[i + 2..i + 4]).unwrap_or("20");
                run.push(u8::from_str_radix(hex, 16).unwrap_or(b' '));
                i += 4;
            }
            let (decoded, _, _) = enc.decode(&run);
            out.push_str(&decoded);
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn extract_rtf(path: &Path) -> Result<BookDoc, ExtractError> {
    let raw = std::fs::read(path).map_err(|e| ExtractError::Io(e.to_string()))?;
    let pre = decode_rtf_escapes(&raw);
    let doc = rtf_parser::document::RtfDocument::try_from(pre.as_str())
        .map_err(|e| ExtractError::Parse(format!("rtf: {e}")))?;
    let text = doc.get_text();
    let sections = vec![Section {
        heading: None,
        body: text,
    }];
    Ok(BookDoc {
        book_id: stable_book_id(path),
        title: title_of(path),
        author: None,
        source_path: path.to_string_lossy().to_string(),
        format: Format::Rtf,
        blocks: sections_to_blocks(sections),
    })
}

/// Dispatch for the office family.
pub fn extract_office(path: &Path) -> Result<BookDoc, ExtractError> {
    match ls_core::ext_of(&path.to_string_lossy()) {
        Some("docx") => extract_docx(path),
        Some("odt") => extract_odt(path),
        Some("rtf") => extract_rtf(path),
        other => Err(ExtractError::Unsupported(format!(
            "{} ({})",
            path.display(),
            other.unwrap_or("no ext")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("ls-extract-office-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn body(n: usize) -> String {
        "слово word content ".repeat(n)
    }

    fn zip_with(entry: &str, xml: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            zw.start_file::<_, ()>(
                entry,
                zip::write::FileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )
            .unwrap();
            std::io::Write::write_all(&mut zw, xml.as_bytes()).unwrap();
            zw.finish().unwrap();
        }
        buf
    }

    #[test]
    fn docx_headings_and_ru_text() {
        let xml = format!(
            r#"<?xml version="1.0"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Введение</w:t></w:r></w:p>
<w:p><w:r><w:t>{b}</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading3"/></w:pPr><w:r><w:t>Deep sub</w:t></w:r></w:p>
<w:p><w:r><w:t>{b}</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Раздел два</w:t></w:r></w:p>
<w:p><w:r><w:t>{b}</w:t></w:r></w:p>
</w:body></w:document>"#,
            b = body(30)
        );
        let p = write_tmp("doc.docx", &zip_with("word/document.xml", &xml));
        let doc = extract_office(&p).unwrap();
        assert_eq!(doc.format, ls_core::Format::Docx);
        let chapters: Vec<_> = doc.blocks.iter().map(|b| b.chapter.clone()).collect();
        assert_eq!(
            chapters,
            vec![Some("Введение".into()), Some("Раздел два".into())],
            "H1/H2 chapter, H3 body"
        );
        assert!(doc.blocks[0].text.contains("Deep sub"));
    }

    #[test]
    fn odt_outline_levels() {
        let xml = format!(
            r#"<?xml version="1.0"?>
<office:document-content xmlns:office="urn:o" xmlns:text="urn:t">
<office:body><office:text>
<text:h text:outline-level="1">Overview</text:h>
<text:p>{b}</text:p>
<text:h text:outline-level="3">Deep</text:h>
<text:p>{b}</text:p>
</office:text></office:body></office:document-content>"#,
            b = body(30)
        );
        let p = write_tmp("doc.odt", &zip_with("content.xml", &xml));
        let doc = extract_office(&p).unwrap();
        assert_eq!(doc.format, ls_core::Format::Odt);
        assert_eq!(doc.blocks[0].chapter.as_deref(), Some("Overview"));
        assert!(doc.blocks[0].text.contains("Deep"), "level-3 stays body");
    }

    /// THE crate-gating fixture (§6.1): RU windows-1251 `\'xx` escapes with
    /// `\ansicpg1251` must decode to real Cyrillic.
    #[test]
    fn rtf_cp1251_escapes_decode() {
        // "Проверка" in cp1251 escapes + latin filler to clear MIN_BOOK_CHARS.
        let ru = "\\'cf\\'f0\\'ee\\'e2\\'e5\\'f0\\'ea\\'e0";
        let rtf = format!(
            "{{\\rtf1\\ansi\\ansicpg1251\\deff0 {{\\fonttbl{{\\f0 Times;}}}}\\f0 {ru} plain latin filler {}\\par}}",
            "word ".repeat(60)
        );
        let p = write_tmp("ru.rtf", rtf.as_bytes());
        let doc = extract_office(&p).unwrap();
        assert_eq!(doc.format, ls_core::Format::Rtf);
        assert!(
            doc.blocks[0].text.contains("Проверка"),
            "cp1251 escapes must decode, got: {}",
            &doc.blocks[0].text[..doc.blocks[0].text.len().min(120)]
        );
    }
}
