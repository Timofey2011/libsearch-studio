//! Best-effort tier (ROADMAP-3 §7): formats with no parser worth building —
//! probe for a converter, convert into the cache, extract the artifact, and
//! stamp the ORIGINAL file's identity (§0.b). No converter → an error whose
//! message is the exact, platform-honest skip reason recorded to skip_state
//! (and retried automatically when the CPU caps_ver PATH-probe set changes).

use std::path::{Path, PathBuf};

use ls_core::{Block, BookDoc, Format};

use crate::text::{apply_section_floor, Section};
use crate::{stable_book_id, ExtractError};

/// Extensions covered by [`extract_with_cache`]'s converter ladder — the
/// lockstep test accepts these as CPU coverage.
pub const CONVERTED_EXTS: &[&str] = &["doc", "pages", "webarchive", "djvu"];

fn title_of(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "Untitled".to_string())
}

fn tool_on_path(name: &str) -> bool {
    std::process::Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Cache key: size + head sample hash. Local to the conversion cache (a
/// changed file gets a fresh artifact); NOT the dedup content signature —
/// ls-extract cannot depend on ls-app.
fn cache_key(path: &Path) -> Result<String, ExtractError> {
    use std::hash::Hasher;
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| ExtractError::Io(e.to_string()))?;
    let len = f
        .metadata()
        .map_err(|e| ExtractError::Io(e.to_string()))?
        .len();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_u64(len);
    let mut buf = [0u8; 64 * 1024];
    let n = f
        .read(&mut buf)
        .map_err(|e| ExtractError::Io(e.to_string()))?;
    h.write(&buf[..n]);
    Ok(format!("{len:x}-{:016x}", h.finish()))
}

fn cache_file(cache_dir: &Path, key: &str, ext: &str) -> Result<PathBuf, ExtractError> {
    std::fs::create_dir_all(cache_dir).map_err(|e| ExtractError::Io(e.to_string()))?;
    Ok(cache_dir.join(format!("{key}.{ext}")))
}

// ---- .doc (legacy Word) -------------------------------------------------------

fn convert_doc_to_txt(path: &Path, out: &Path) -> Result<(), ExtractError> {
    const REASON: &str = "legacy .doc: install antiword or LibreOffice";
    if tool_on_path("textutil") {
        let ok = std::process::Command::new("textutil")
            .args(["-convert", "txt", "-output"])
            .arg(out)
            .arg(path)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok && out.exists() {
            return Ok(());
        }
    }
    if tool_on_path("antiword") {
        if let Ok(o) = std::process::Command::new("antiword").arg(path).output() {
            if o.status.success() && !o.stdout.is_empty() {
                std::fs::write(out, &o.stdout).map_err(|e| ExtractError::Io(e.to_string()))?;
                return Ok(());
            }
        }
    }
    if tool_on_path("soffice") {
        let dir = out.parent().unwrap_or(Path::new("."));
        let ok = std::process::Command::new("soffice")
            .args(["--headless", "--convert-to", "txt", "--outdir"])
            .arg(dir)
            .arg(path)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        // soffice names the artifact <stem>.txt — move it onto the cache key.
        let produced = dir.join(format!(
            "{}.txt",
            path.file_stem().unwrap_or_default().to_string_lossy()
        ));
        if ok && produced.exists() {
            std::fs::rename(&produced, out).map_err(|e| ExtractError::Io(e.to_string()))?;
            return Ok(());
        }
    }
    Err(ExtractError::Parse(REASON.into()))
}

fn extract_doc(path: &Path, cache_dir: &Path) -> Result<BookDoc, ExtractError> {
    let cache = cache_file(cache_dir, &cache_key(path)?, "txt")?;
    if !cache.exists() {
        convert_doc_to_txt(path, &cache)?;
    }
    let raw = std::fs::read(&cache).map_err(|e| ExtractError::Io(e.to_string()))?;
    let text = String::from_utf8_lossy(&raw).into_owned();
    Ok(doc_from_sections(
        path,
        Format::Doc,
        vec![Section {
            heading: None,
            body: text,
        }],
    ))
}

// ---- .pages — "Preview.pdf or bust" ladder ------------------------------------

fn pages_preview_bytes(path: &Path) -> Option<Vec<u8>> {
    use std::io::Read;
    if path.is_dir() {
        // Older Pages documents are bundle directories.
        let p = path.join("QuickLook").join("Preview.pdf");
        return std::fs::read(p).ok();
    }
    let f = std::fs::File::open(path).ok()?;
    let mut ar = zip::ZipArchive::new(f).ok()?;
    let idx = (0..ar.len()).find(|&i| {
        ar.by_index(i)
            .map(|e| e.name().to_ascii_lowercase().ends_with("preview.pdf"))
            .unwrap_or(false)
    })?;
    let mut entry = ar.by_index(idx).ok()?;
    let mut out = Vec::new();
    entry.read_to_end(&mut out).ok()?;
    Some(out)
}

fn extract_pages(path: &Path, cache_dir: &Path) -> Result<BookDoc, ExtractError> {
    let cache = cache_file(cache_dir, &cache_key(path)?, "pdf")?;
    if !cache.exists() {
        let bytes = pages_preview_bytes(path).ok_or_else(|| {
            ExtractError::Parse(
                ".pages without embedded preview — open in Pages and export PDF".into(),
            )
        })?;
        std::fs::write(&cache, bytes).map_err(|e| ExtractError::Io(e.to_string()))?;
    }
    // Extract the cached preview, then restamp the ORIGINAL identity (§0.b):
    // the moved-file guard, dedup, and "Open in default app" track the .pages.
    let mut doc = crate::extract_pdf(&cache)?;
    doc.book_id = stable_book_id(path);
    doc.source_path = path.to_string_lossy().to_string();
    doc.format = Format::Pages;
    doc.title = title_of(path);
    Ok(doc)
}

/// Display artifact for a .pages file (the cached preview pdf), converting on
/// demand. Used by the bridge's `resolve_display_path`.
pub fn pages_display_pdf(path: &Path, cache_dir: &Path) -> Result<PathBuf, ExtractError> {
    let cache = cache_file(cache_dir, &cache_key(path)?, "pdf")?;
    if !cache.exists() {
        let bytes = pages_preview_bytes(path).ok_or_else(|| {
            ExtractError::Parse(
                ".pages without embedded preview — open in Pages and export PDF".into(),
            )
        })?;
        std::fs::write(&cache, bytes).map_err(|e| ExtractError::Io(e.to_string()))?;
    }
    Ok(cache)
}

// ---- .webarchive ---------------------------------------------------------------

/// Cross-platform: a Safari webarchive is a plist whose
/// `WebMainResource/WebResourceData` holds the page HTML.
pub fn webarchive_html(path: &Path) -> Result<String, ExtractError> {
    let v = plist::Value::from_file(path)
        .map_err(|e| ExtractError::Parse(format!("webarchive: {e}")))?;
    let data = v
        .as_dictionary()
        .and_then(|d| d.get("WebMainResource"))
        .and_then(|m| m.as_dictionary())
        .and_then(|m| m.get("WebResourceData"))
        .and_then(|d| d.as_data())
        .ok_or_else(|| ExtractError::Parse("webarchive: no WebMainResource data".into()))?;
    Ok(String::from_utf8_lossy(data).into_owned())
}

fn extract_webarchive(path: &Path) -> Result<BookDoc, ExtractError> {
    let html = webarchive_html(path)?;
    let sections = crate::text::scan_html(&html);
    let mut doc = doc_from_sections(path, Format::Webarchive, sections);
    doc.title = title_of(path);
    Ok(doc)
}

// ---- .djvu ----------------------------------------------------------------------

fn extract_djvu(path: &Path, cache_dir: &Path) -> Result<BookDoc, ExtractError> {
    const REASON: &str = "djvu: brew install djvulibre (scanned-only djvu needs OCR — unsupported)";
    if !tool_on_path("djvutxt") {
        return Err(ExtractError::Parse(REASON.into()));
    }
    let cache = cache_file(cache_dir, &cache_key(path)?, "txt")?;
    if !cache.exists() {
        // djvulibre is GPL: subprocess only, never linked or bundled.
        let ok = std::process::Command::new("djvutxt")
            .arg(path)
            .arg(&cache)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !ok {
            return Err(ExtractError::Parse(REASON.into()));
        }
    }
    let text = std::fs::read_to_string(&cache).unwrap_or_default();
    if text.trim().len() < crate::MIN_BOOK_CHARS {
        return Err(ExtractError::Parse(REASON.into()));
    }
    Ok(doc_from_sections(
        path,
        Format::Djvu,
        vec![Section {
            heading: None,
            body: text,
        }],
    ))
}

// ---- shared ---------------------------------------------------------------------

fn doc_from_sections(path: &Path, format: Format, sections: Vec<Section>) -> BookDoc {
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
    BookDoc {
        book_id: stable_book_id(path),
        title: title_of(path),
        author: None,
        source_path: path.to_string_lossy().to_string(),
        format,
        blocks,
    }
}

/// [`crate::extract`] plus the converter ladder for the best-effort formats.
/// This is the entry point both pipelines' CPU side should use; `cache_dir`
/// is the app's `<data_dir>/converted`.
pub fn extract_with_cache(path: &Path, cache_dir: &Path) -> Result<BookDoc, ExtractError> {
    match ls_core::ext_of(&path.to_string_lossy()) {
        Some("doc") => extract_doc(path, cache_dir),
        Some("pages") => extract_pages(path, cache_dir),
        Some("webarchive") => extract_webarchive(path),
        Some("djvu") => extract_djvu(path, cache_dir),
        _ => crate::extract(path),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join("ls-extract-convert-tests")
            .join(name);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A minimal but valid one-page PDF with real text, via lopdf.
    fn tiny_pdf_bytes(text: &str) -> Vec<u8> {
        use lopdf::content::{Content, Operation};
        use lopdf::{dictionary, Document, Object, Stream};
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
        });
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! { "F1" => font_id },
        });
        let content = Content {
            operations: vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 24.into()]),
                Operation::new("Td", vec![50.into(), 700.into()]),
                Operation::new("Tj", vec![Object::string_literal(text)]),
                Operation::new("ET", vec![]),
            ],
        };
        let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page", "Parent" => pages_id, "Contents" => content_id,
        });
        let pages = dictionary! {
            "Type" => "Pages", "Kids" => vec![page_id.into()], "Count" => 1,
            "Resources" => resources_id, "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        };
        doc.objects.insert(pages_id, Object::Dictionary(pages));
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog", "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);
        let mut out = Vec::new();
        doc.save_to(&mut out).unwrap();
        out
    }

    fn pages_zip(with_preview: bool, text: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            let opts = zip::write::FileOptions::<()>::default()
                .compression_method(zip::CompressionMethod::Deflated);
            zw.start_file("Index/Document.iwa", opts).unwrap();
            std::io::Write::write_all(&mut zw, b"opaque protobuf junk").unwrap();
            if with_preview {
                zw.start_file("QuickLook/Preview.pdf", opts).unwrap();
                std::io::Write::write_all(&mut zw, &tiny_pdf_bytes(text)).unwrap();
            }
            zw.finish().unwrap();
        }
        buf
    }

    #[test]
    fn pages_with_preview_extracts_and_keeps_original_identity() {
        let dir = tmpdir("pages-yes");
        let cache = dir.join("cache");
        let text = "The quick brown fox rehearses portfolio theory ".repeat(8);
        let p = dir.join("Notes.pages");
        std::fs::write(&p, pages_zip(true, &text)).unwrap();

        let doc = extract_with_cache(&p, &cache).unwrap();
        assert_eq!(doc.format, Format::Pages);
        assert!(
            doc.source_path.ends_with("Notes.pages"),
            "identity = original"
        );
        assert_eq!(doc.book_id, stable_book_id(&p));
        assert_eq!(doc.title, "Notes");
        assert!(doc.blocks[0].text.contains("portfolio theory"));
        // The artifact landed in the cache and is reused on the second call.
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1);
        let again = extract_with_cache(&p, &cache).unwrap();
        assert_eq!(again.blocks.len(), doc.blocks.len());
    }

    #[test]
    fn pages_without_preview_reports_the_exact_reason() {
        let dir = tmpdir("pages-no");
        let p = dir.join("NoPreview.pages");
        std::fs::write(&p, pages_zip(false, "")).unwrap();
        let err = extract_with_cache(&p, &dir.join("cache")).unwrap_err();
        assert_eq!(
            err.to_string(),
            "parse: .pages without embedded preview — open in Pages and export PDF"
        );
    }

    #[test]
    fn webarchive_plist_extracts_html() {
        let dir = tmpdir("webarchive");
        let body = "Сохранённая страница о распределённых системах. ".repeat(10);
        let html = format!("<html><body><h1>Заметка</h1><p>{body}</p></body></html>");
        let mut root = plist::Dictionary::new();
        let mut main = plist::Dictionary::new();
        main.insert(
            "WebResourceData".into(),
            plist::Value::Data(html.into_bytes()),
        );
        main.insert(
            "WebResourceURL".into(),
            plist::Value::String("https://example.org".into()),
        );
        root.insert("WebMainResource".into(), plist::Value::Dictionary(main));
        let p = dir.join("page.webarchive");
        plist::Value::Dictionary(root).to_file_binary(&p).unwrap();

        let doc = extract_with_cache(&p, &dir.join("cache")).unwrap();
        assert_eq!(doc.format, Format::Webarchive);
        assert_eq!(doc.blocks[0].chapter.as_deref(), Some("Заметка"));
        assert!(doc.blocks[0].text.contains("распределённых системах"));
    }

    #[test]
    fn djvu_without_tool_or_doc_without_converter_report_reasons() {
        // djvutxt is absent on CI; on a mac with djvulibre this test would
        // exercise the tool path instead, so only assert the no-tool branch
        // when the tool is genuinely missing.
        let dir = tmpdir("reasons");
        if !tool_on_path("djvutxt") {
            let p = dir.join("scan.djvu");
            std::fs::write(&p, b"AT&TFORM fake").unwrap();
            let err = extract_with_cache(&p, &dir.join("cache")).unwrap_err();
            assert!(err.to_string().contains("brew install djvulibre"), "{err}");
        }
        if !tool_on_path("textutil") && !tool_on_path("antiword") && !tool_on_path("soffice") {
            let p = dir.join("old.doc");
            std::fs::write(&p, b"\xd0\xcf\x11\xe0 fake ole").unwrap();
            let err = extract_with_cache(&p, &dir.join("cache")).unwrap_err();
            assert!(err.to_string().contains("install antiword"), "{err}");
        }
    }
}
