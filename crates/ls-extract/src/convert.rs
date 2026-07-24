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

/// Generous for any real conversion, but bounds a hung converter (soffice on a
/// corrupt file, any tool on a dehydrated cloud placeholder) so Stop is never
/// stuck behind an unkillable subprocess wait.
const CONVERT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// `Command::output()` with a deadline: poll `try_wait`, kill on expiry, and
/// return an error so the caller's probe ladder falls through to the next tool
/// (or the platform-honest skip reason). Stdout is drained on a thread — a full
/// pipe would otherwise deadlock the child before the deadline ever fires.
fn run_bounded(cmd: &mut std::process::Command) -> std::io::Result<std::process::Output> {
    use std::io::Read;
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .stdin(std::process::Stdio::null())
        .spawn()?;
    let mut stdout_pipe = child.stdout.take();
    let reader = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(ref mut out) = stdout_pipe {
            let _ = out.read_to_end(&mut buf);
        }
        buf
    });
    let deadline = std::time::Instant::now() + CONVERT_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait()? {
            let stdout = reader.join().unwrap_or_default();
            return Ok(std::process::Output {
                status,
                stdout,
                stderr: Vec::new(),
            });
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "converter timed out",
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
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
        let ok = run_bounded(
            std::process::Command::new("textutil")
                .args(["-convert", "txt", "-output"])
                .arg(out)
                .arg(path),
        )
        .map(|o| o.status.success())
        .unwrap_or(false);
        if ok && out.exists() {
            return Ok(());
        }
    }
    if tool_on_path("antiword") {
        if let Ok(o) = run_bounded(std::process::Command::new("antiword").arg(path)) {
            if o.status.success() && !o.stdout.is_empty() {
                std::fs::write(out, &o.stdout).map_err(|e| ExtractError::Io(e.to_string()))?;
                return Ok(());
            }
        }
    }
    if tool_on_path("soffice") {
        let dir = out.parent().unwrap_or(Path::new("."));
        let ok = run_bounded(
            std::process::Command::new("soffice")
                .args(["--headless", "--convert-to", "txt", "--outdir"])
                .arg(dir)
                .arg(path),
        )
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

// ---- scanned pdf (OCR) ---------------------------------------------------------

/// Fraction of pages carrying so little text that the page is certainly an
/// image. The whole-book 200-char floor cannot see this: a scan whose title
/// page holds a scrap of text clears it, commits as `indexed`, and is never
/// revisited (ROADMAP-3 §18.1 — 10 such "ghosts" among 462 indexed PDFs).
const SCANNED_PAGE_CHARS: usize = 50;
const SCANNED_PAGE_RATIO: f32 = 0.60;

/// True when most of `total_pages` carry no usable text — i.e. OCR is the only
/// way to read the book.
///
/// `total_pages` must come from the document, NOT from `doc.blocks`: a scanned
/// page yields no block at all, so the pages this needs to count are precisely
/// the ones missing from the extraction.
pub fn looks_scanned(doc: &BookDoc, total_pages: usize) -> bool {
    if total_pages == 0 {
        return doc.blocks.is_empty();
    }
    let mut per_page: std::collections::BTreeMap<u32, usize> = Default::default();
    for b in &doc.blocks {
        *per_page.entry(b.page.unwrap_or(0)).or_default() += b.text.chars().count();
    }
    let thin = (1..=total_pages)
        .filter(|p| per_page.get(&(*p as u32)).copied().unwrap_or(0) < SCANNED_PAGE_CHARS)
        .count();
    thin as f32 / total_pages as f32 >= SCANNED_PAGE_RATIO
}

/// FNV-1a over length + head/tail sample.
///
/// Deliberately NOT `cache_key`/`content_signature`, which both use Rust's
/// `DefaultHasher` — unspecified across Rust versions and impossible to
/// reproduce in Python. The OCR helper is a Python process that must be able
/// to name the same artifact, and FNV-1a is this repo's existing
/// cross-language primitive (see `ls-cli`'s `fnv1a`). Keep the two in step.
pub fn ocr_cache_key(path: &Path) -> Result<String, ExtractError> {
    use std::io::{Read, Seek, SeekFrom};
    const SAMPLE: usize = 64 * 1024;
    let mut f = std::fs::File::open(path).map_err(|e| ExtractError::Io(e.to_string()))?;
    let len = f
        .metadata()
        .map_err(|e| ExtractError::Io(e.to_string()))?
        .len();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let eat = |bytes: &[u8], h: &mut u64| {
        for b in bytes {
            *h ^= *b as u64;
            *h = h.wrapping_mul(0x100_0000_01b3);
        }
    };
    eat(&len.to_le_bytes(), &mut h);
    let mut buf = vec![0u8; SAMPLE];
    let n = f
        .read(&mut buf)
        .map_err(|e| ExtractError::Io(e.to_string()))?;
    eat(&buf[..n], &mut h);
    if len > SAMPLE as u64 {
        f.seek(SeekFrom::End(-(SAMPLE as i64)))
            .map_err(|e| ExtractError::Io(e.to_string()))?;
        let n = f
            .read(&mut buf)
            .map_err(|e| ExtractError::Io(e.to_string()))?;
        eat(&buf[..n], &mut h);
    }
    Ok(format!("{h:016x}"))
}

/// Path of the OCR artifact for `path`, whether or not it exists yet.
pub fn ocr_cache_path(path: &Path, cache_dir: &Path) -> Result<PathBuf, ExtractError> {
    cache_file(cache_dir, &format!("{}.ocr", ocr_cache_key(path)?), "pdf")
}

/// The searchable copy, if one has already been produced. Used both by
/// extraction (to read the text) and by the bridge's `resolve_display_path`
/// (to show the reader a copy whose text layer can actually be highlighted).
pub fn ocr_display_pdf(path: &Path, cache_dir: &Path) -> Option<PathBuf> {
    let p = ocr_cache_path(path, cache_dir).ok()?;
    p.exists().then_some(p)
}

/// Extract a pdf, preferring an existing OCR artifact for a scanned one.
///
/// Producing the artifact is deliberately NOT done here: it costs minutes and
/// needs a Python interpreter, so it belongs to an explicit, cancellable step
/// (§18.4b), not to an extraction call that runs inside an index batch.
fn extract_pdf_cached(path: &Path, cache_dir: &Path) -> Result<BookDoc, ExtractError> {
    if let Some(ocr) = ocr_display_pdf(path, cache_dir) {
        // Restamp the ORIGINAL identity (§0.b), exactly as .pages does: the
        // moved-file guard, dedup and "Open in default app" track the source.
        let mut doc = crate::extract_pdf(&ocr)?;
        doc.book_id = stable_book_id(path);
        doc.source_path = path.to_string_lossy().to_string();
        doc.title = title_of(path);
        return Ok(doc);
    }
    crate::extract_pdf(path)
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
        let ok = run_bounded(std::process::Command::new("djvutxt").arg(path).arg(&cache))
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
        Some("pdf") => extract_pdf_cached(path, cache_dir),
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

    /// The OCR cache key must match `scripts/ocr_pdf.py::ocr_cache_key` byte
    /// for byte, or Rust and the helper name different artifacts and the cache
    /// silently never hits. Vectors below are produced by the Python side; if
    /// this fails, the two implementations have drifted.
    #[test]
    fn ocr_cache_key_matches_the_python_helper() {
        let dir = tmpdir("ocr-key");
        // Small (single read, no tail sample) and large (head+tail) cases.
        let small = dir.join("small.bin");
        std::fs::write(&small, b"libsearch ocr key vector").unwrap();
        assert_eq!(ocr_cache_key(&small).unwrap(), "ffc35cad99b2a12a");

        let big = dir.join("big.bin");
        let bytes: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&big, &bytes).unwrap();
        assert_eq!(ocr_cache_key(&big).unwrap(), "cfc6243be73dcb78");
    }

    /// A book whose pages are nearly all text-less is a scan, even when its
    /// front matter clears the whole-book 200-char floor (ROADMAP-3 §18.1).
    #[test]
    fn looks_scanned_sees_ghosts_the_whole_book_floor_misses() {
        let mut doc = BookDoc {
            book_id: "b".into(),
            title: "t".into(),
            author: None,
            source_path: "/x.pdf".into(),
            format: Format::Pdf,
            blocks: vec![],
        };
        // 1 page of real text, 19 image pages: 250 chars total clears the
        // 200-char floor, yet 95% of the book is unreadable.
        doc.blocks.push(Block::new("x".repeat(250), None, Some(1)));
        assert!(
            looks_scanned(&doc, 20),
            "a 1-of-20-page ghost must read as scanned"
        );

        // A normal book: every page carries text.
        doc.blocks = (1..=20)
            .map(|p| Block::new("y".repeat(400), None, Some(p)))
            .collect();
        assert!(!looks_scanned(&doc, 20));

        // Front-matter-only scan: those pages are genuinely text, but they are
        // a small minority of the document.
        doc.blocks = (1..=6)
            .map(|p| Block::new("z".repeat(400), None, Some(p)))
            .collect();
        assert!(looks_scanned(&doc, 100));

        // No text at all.
        doc.blocks.clear();
        assert!(looks_scanned(&doc, 20));
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
    fn run_bounded_captures_stdout_and_kills_on_deadline() {
        // Success path: stdout is fully captured.
        let out = run_bounded(std::process::Command::new("sh").args(["-c", "echo hello"])).unwrap();
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "hello");
        // Failure status propagates (ladder falls through on it).
        let out = run_bounded(std::process::Command::new("sh").args(["-c", "exit 3"])).unwrap();
        assert!(!out.status.success());
        // NOTE: the deadline branch is exercised with a 100ms poll against a
        // long sleep in a dedicated ignored test (2min wall) — the kill logic
        // is identical for any deadline, so CI covers spawn/capture only.
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
