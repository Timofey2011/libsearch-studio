//! Core domain types shared across the LibSearch Studio engine crates.
//!
//! This crate is UI- and IO-agnostic: it defines the records that flow between
//! extraction, chunking, indexing, and querying. Nothing here depends on Tauri,
//! ONNX, LanceDB, or the network, so every higher layer can be unit-tested
//! against these types in isolation.

use serde::{Deserialize, Serialize};

/// Source file format FAMILY of a book. Family granularity, deliberately: the
/// store `format` column drives only citation shape (`p.N` vs `Ch. X` vs
/// `~loc`) and diagnostics; readers key off the file extension. Don't
/// over-model (ROADMAP-3 §2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Pdf,
    Epub,
    Mobi,
    Md,
    Txt,
    Html,
    Docx,
    Rtf,
    Odt,
    Doc,
    Fb2,
    Pages,
    Webarchive,
    Djvu,
    Xps,
}

/// The canonical list of file extensions the app knows about, WITHOUT dots,
/// lowercase, compound extensions included (`fb2.zip`). Every other layer —
/// ingest discovery, extractor dispatch, the GPU helper's Python mirror, the
/// generated frontend map, error strings — derives from this list; a lockstep
/// test asserts they never drift (ROADMAP-3 §2.2/§2.5).
///
/// NOTE: this is the *known* universe, not what ingest currently accepts —
/// discovery filters through [`INGEST_EXTS`] (flipped per milestone).
pub const KNOWN_EXTS: &[&str] = &[
    "pdf",
    "epub",
    "mobi",
    "azw3",
    "md",
    "markdown",
    "txt",
    "text",
    "rst",
    "adoc",
    "org",
    "tex",
    "ipynb",
    "html",
    "htm",
    "docx",
    "rtf",
    "odt",
    "doc",
    "fb2",
    "fb2.zip",
    "pages",
    "webarchive",
    "djvu",
    "xps",
];

/// Extensions ingest actually accepts TODAY. M0 ships with the pdf-only status
/// quo (dark soak of the dedup guards); milestones flip families on here.
pub const INGEST_EXTS: &[&str] = &["pdf"];

/// The one extension-derivation rule (ROADMAP-3 invariant #8): lowercase the
/// filename and LONGEST-match against `KNOWN_EXTS`, so `x.fb2.zip` is
/// `fb2.zip`, never `zip`. Unknown → None. `Path::extension` must not be used
/// for format decisions anywhere.
pub fn ext_of(name: &str) -> Option<&'static str> {
    let lower = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(name)
        .to_ascii_lowercase();
    KNOWN_EXTS
        .iter()
        .filter(|e| {
            lower.len() > e.len() + 1
                && lower.ends_with(*e)
                && lower.as_bytes()[lower.len() - e.len() - 1] == b'.'
        })
        .max_by_key(|e| e.len())
        .copied()
}

impl Format {
    /// Parse from an extension as produced by [`ext_of`] (compound extensions
    /// included). Aliases map to their family: `markdown`/`ipynb` → Md,
    /// `text`/`rst`/`adoc`/`org`/`tex` → Txt, `htm` → Html, `azw3` → Mobi,
    /// `fb2.zip` → Fb2.
    pub fn from_ext(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "pdf" => Some(Self::Pdf),
            "epub" => Some(Self::Epub),
            "mobi" | "azw3" => Some(Self::Mobi),
            "md" | "markdown" | "ipynb" => Some(Self::Md),
            "txt" | "text" | "rst" | "adoc" | "org" | "tex" => Some(Self::Txt),
            "html" | "htm" => Some(Self::Html),
            "docx" => Some(Self::Docx),
            "rtf" => Some(Self::Rtf),
            "odt" => Some(Self::Odt),
            "doc" => Some(Self::Doc),
            "fb2" | "fb2.zip" => Some(Self::Fb2),
            "pages" => Some(Self::Pages),
            "webarchive" => Some(Self::Webarchive),
            "djvu" => Some(Self::Djvu),
            "xps" => Some(Self::Xps),
            _ => None,
        }
    }

    /// Parse from a file path/name via [`ext_of`].
    pub fn from_path(name: &str) -> Option<Self> {
        ext_of(name).and_then(Self::from_ext)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Epub => "epub",
            Self::Mobi => "mobi",
            Self::Md => "md",
            Self::Txt => "txt",
            Self::Html => "html",
            Self::Docx => "docx",
            Self::Rtf => "rtf",
            Self::Odt => "odt",
            Self::Doc => "doc",
            Self::Fb2 => "fb2",
            Self::Pages => "pages",
            Self::Webarchive => "webarchive",
            Self::Djvu => "djvu",
            Self::Xps => "xps",
        }
    }

    /// Every variant, for round-trip tests and generators.
    pub const ALL: &'static [Format] = &[
        Self::Pdf,
        Self::Epub,
        Self::Mobi,
        Self::Md,
        Self::Txt,
        Self::Html,
        Self::Docx,
        Self::Rtf,
        Self::Odt,
        Self::Doc,
        Self::Fb2,
        Self::Pages,
        Self::Webarchive,
        Self::Djvu,
        Self::Xps,
    ];
}

/// Human-readable ingest-extension list for error strings.
pub fn supported_exts_display() -> String {
    INGEST_EXTS.join(", ")
}

/// Render `frontend/src/generated/supportedExts.ts` — the frontend's ONLY
/// source of extension knowledge (invariant #8: no hand-mirrored lists). The
/// `gen-exts` CLI subcommand writes it; a freshness test asserts the
/// checked-in copy is byte-identical, so drift fails CI without a JS runner.
pub fn gen_supported_exts_ts() -> String {
    let known = KNOWN_EXTS
        .iter()
        .map(|e| format!("\"{e}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let ingest = INGEST_EXTS
        .iter()
        .map(|e| format!("\"{e}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let family = KNOWN_EXTS
        .iter()
        .map(|e| {
            let fam = Format::from_ext(e).map(|f| f.as_str()).unwrap_or("other");
            format!("  \"{e}\": \"{fam}\",")
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"// GENERATED from ls-core by `cargo run -p ls-cli -- gen-exts`. DO NOT EDIT:
// a Rust test asserts this file is byte-identical to the generator's output.
export const KNOWN_EXTS = [{known}] as const;
export const INGEST_EXTS = [{ingest}] as const;
/// Extension -> format family (citation shape / reader-kind routing).
export const EXT_FAMILY: Record<string, string> = {{
{family}
}};
/// The one extension-derivation rule: lowercase, LONGEST match against
/// KNOWN_EXTS ("x.fb2.zip" is "fb2.zip", never "zip"). Unknown -> null.
export function extOf(name: string): string | null {{
  const lower = (name.split(/[\\/]/).pop() ?? name).toLowerCase();
  let best: string | null = null;
  for (const e of KNOWN_EXTS) {{
    if (
      lower.length > e.length + 1 &&
      lower.endsWith(e) &&
      lower[lower.length - e.length - 1] === "."
    ) {{
      if (!best || e.length > best.length) best = e;
    }}
  }}
  return best;
}}
"#
    )
}

/// An ordered unit of text from a book, with structural metadata.
///
/// `page` is meaningful for PDF only; epub/mobi carry chapter + character offset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
    pub text: String,
    pub chapter: Option<String>,
    pub page: Option<u32>,
}

impl Block {
    pub fn new(text: impl Into<String>, chapter: Option<String>, page: Option<u32>) -> Self {
        Self {
            text: text.into(),
            chapter,
            page,
        }
    }
}

/// A normalized, format-agnostic book: ordered blocks plus metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BookDoc {
    pub book_id: String,
    pub title: String,
    pub author: Option<String>,
    pub source_path: String,
    pub format: Format,
    pub blocks: Vec<Block>,
}

/// One retrievable passage — becomes exactly one row in the vector store.
///
/// Character offsets (`loc_start`/`loc_end`) are counted in Unicode scalar values
/// so they stay meaningful for non-Latin scripts (the library is ~28% Russian).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub book_id: String,
    pub title: String,
    pub author: Option<String>,
    pub source_path: String,
    pub format: Format,
    pub chapter: Option<String>,
    pub page: Option<u32>,
    pub loc_start: usize,
    pub loc_end: usize,
    pub text: String,
    /// Dense embedding, filled by the embedding stage (`None` until then).
    pub vector: Option<Vec<f32>>,
}

/// Counts tokens for chunk sizing. The real implementation (in `ls-embed`) wraps
/// the embedder's tokenizer; tests use [`WhitespaceCounter`]. Defined here in the
/// leaf crate so both `ls-index` (consumer) and `ls-embed` (provider) can use it
/// without an orphan-rule violation.
pub trait TokenCounter {
    fn count(&self, text: &str) -> usize;
}

/// Cheap, offline token estimate: one token per whitespace-delimited word.
pub struct WhitespaceCounter;

impl TokenCounter for WhitespaceCounter {
    fn count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_roundtrips_through_ext() {
        assert_eq!(Format::from_ext("PDF"), Some(Format::Pdf));
        assert_eq!(Format::from_ext("epub"), Some(Format::Epub));
        assert_eq!(Format::from_ext("txt"), Some(Format::Txt));
        assert_eq!(Format::Pdf.as_str(), "pdf");
        // The read path silently drops `format` to None on any as_str/from_ext
        // asymmetry, so the asymmetry itself must be impossible to ship.
        for f in Format::ALL {
            assert_eq!(Format::from_ext(f.as_str()), Some(*f), "{f:?} round-trip");
        }
    }

    #[test]
    fn ext_of_longest_match_and_negatives() {
        // Compound extension beats its suffix: fb2.zip, never zip.
        assert_eq!(ext_of("Мастер и Маргарита.fb2.zip"), Some("fb2.zip"));
        assert_eq!(
            ext_of("/lib/у папки.с точкой/book.FB2.ZIP"),
            Some("fb2.zip")
        );
        assert_eq!(ext_of("notes.tar.gz"), None); // unknown stays unknown
        assert_eq!(ext_of("a.md"), Some("md"));
        assert_eq!(ext_of("a.MARKDOWN"), Some("markdown"));
        assert_eq!(ext_of("weird.pdf.txt"), Some("txt")); // last ext wins
        assert_eq!(ext_of("noext"), None);
        assert_eq!(ext_of(".md"), None); // bare dotfile is not a book
        assert_eq!(ext_of("C:\\books\\a.docx"), Some("docx"));
        // Aliases resolve to families.
        assert_eq!(Format::from_path("l.ipynb"), Some(Format::Md));
        assert_eq!(Format::from_path("b.azw3"), Some(Format::Mobi));
        assert_eq!(Format::from_path("b.fb2.zip"), Some(Format::Fb2));
        assert_eq!(Format::from_path("b.rst"), Some(Format::Txt));
    }

    #[test]
    fn ingest_exts_is_a_subset_of_known() {
        for e in INGEST_EXTS {
            assert!(KNOWN_EXTS.contains(e), "{e} must be in KNOWN_EXTS");
            assert!(Format::from_ext(e).is_some(), "{e} must map to a Format");
        }
    }

    #[test]
    fn generated_frontend_ext_map_is_fresh() {
        let checked_in = include_str!("../../../frontend/src/generated/supportedExts.ts");
        assert_eq!(
            checked_in,
            gen_supported_exts_ts(),
            "frontend/src/generated/supportedExts.ts is stale — run `cargo run -p ls-cli -- gen-exts`"
        );
    }
}
