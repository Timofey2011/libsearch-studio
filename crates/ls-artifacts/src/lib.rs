//! Render a grounded chat answer + its citations into a saveable artifact.
//!
//! v1 emits Markdown with a YAML front-matter header (title, created, model,
//! collection, sources) followed by the answer body and a Sources section.
//! The [`ArtifactRenderer`] trait keeps the door open for other formats later
//! without changing callers.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One cited source backing an answer. Field names match `ls_query::SearchResult`
/// so the Tauri bridge can forward the frontend's citations verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub rank: usize,
    pub citation: String,
    pub source_path: String,
    #[serde(default)]
    pub page: Option<u32>,
    #[serde(default)]
    pub chapter: Option<String>,
}

/// Everything needed to render an artifact from one assistant turn.
#[derive(Debug, Clone)]
pub struct ArtifactDoc {
    /// The user's question — also used as the document title and filename slug.
    pub question: String,
    /// The model's grounded answer (may contain inline `[n]` citation markers).
    pub answer: String,
    /// Synthesis model id, recorded for provenance.
    pub model: String,
    /// Collection the answer was grounded in.
    pub collection: String,
    /// Human-readable creation timestamp (caller-supplied; e.g. ISO-8601).
    pub created: String,
    pub sources: Vec<Source>,
}

/// A pluggable artifact format.
pub trait ArtifactRenderer {
    /// File extension (no dot), e.g. `"md"`.
    fn extension(&self) -> &str;
    /// Render the document to a string.
    fn render(&self, doc: &ArtifactDoc) -> String;
}

/// Markdown renderer with YAML front-matter.
pub struct Markdown;

impl ArtifactRenderer for Markdown {
    fn extension(&self) -> &str {
        "md"
    }

    fn render(&self, doc: &ArtifactDoc) -> String {
        let title = doc.question.trim();
        let mut out = String::new();

        // --- YAML front-matter ---
        out.push_str("---\n");
        out.push_str(&format!("title: {}\n", yaml_quote(title)));
        out.push_str(&format!("created: {}\n", yaml_quote(doc.created.trim())));
        out.push_str(&format!("model: {}\n", yaml_quote(doc.model.trim())));
        out.push_str(&format!(
            "collection: {}\n",
            yaml_quote(doc.collection.trim())
        ));
        if doc.sources.is_empty() {
            out.push_str("sources: []\n");
        } else {
            out.push_str("sources:\n");
            for s in &doc.sources {
                out.push_str(&format!("  - {}\n", yaml_quote(&s.citation)));
            }
        }
        out.push_str("---\n\n");

        // --- Body ---
        out.push_str(&format!("# {title}\n\n"));
        out.push_str(doc.answer.trim());
        out.push_str("\n\n");

        // --- Sources ---
        out.push_str("## Sources\n\n");
        if doc.sources.is_empty() {
            out.push_str("_No sources retrieved._\n");
        } else {
            for s in &doc.sources {
                let loc = match s.page {
                    Some(p) => format!(" (p.{p})"),
                    None => String::new(),
                };
                out.push_str(&format!(
                    "{}. {}\n   `{}`{}\n",
                    s.rank, s.citation, s.source_path, loc
                ));
            }
        }
        out
    }
}

/// Render and write an artifact under `dir`, returning the path written. The
/// filename is a slug of the document title; a numeric suffix avoids clobbering
/// an existing file.
pub fn write_artifact(
    renderer: &dyn ArtifactRenderer,
    doc: &ArtifactDoc,
    dir: &Path,
) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let content = renderer.render(doc);
    let ext = renderer.extension();
    let slug = slugify(&doc.question);

    let mut path = dir.join(format!("{slug}.{ext}"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{slug}-{n}.{ext}"));
        n += 1;
    }
    fs::write(&path, content)?;
    Ok(path)
}

/// Lowercase, hyphenate, and trim a title into a filesystem-safe slug.
pub fn slugify(title: &str) -> String {
    let mut slug = String::new();
    let mut prev_dash = false;
    for ch in title.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash && !slug.is_empty() {
            slug.push('-');
            prev_dash = true;
        }
    }
    let slug = slug.trim_matches('-');
    // Cap length; keep whole words where possible.
    let capped: String = slug.chars().take(60).collect();
    let capped = capped.trim_matches('-').to_string();
    if capped.is_empty() {
        "artifact".to_string()
    } else {
        capped
    }
}

/// Quote a scalar for single-line YAML, escaping `"` and backslashes.
fn yaml_quote(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> ArtifactDoc {
        ArtifactDoc {
            question: "What is idempotence in microservices?".into(),
            answer: "Idempotence means calling once or many times has the same effect [1].".into(),
            model: "gemma4:12b-mlx".into(),
            collection: "My Library".into(),
            created: "2026-06-20 14:30:00".into(),
            sources: vec![Source {
                rank: 1,
                citation: "Practical Microservices — Ethan Garofolo · Ch. Idempotent, p.75".into(),
                source_path: "/books/practical-microservices.pdf".into(),
                page: Some(75),
                chapter: Some("Idempotent".into()),
            }],
        }
    }

    #[test]
    fn slugify_basic() {
        assert_eq!(slugify("Hello, World!"), "hello-world");
        assert_eq!(slugify("  spaced   out  "), "spaced-out");
        assert_eq!(slugify("***"), "artifact");
        assert_eq!(slugify("C++ & Rust: A Tale"), "c-rust-a-tale");
    }

    #[test]
    fn render_has_frontmatter_body_and_sources() {
        let md = Markdown.render(&doc());
        assert!(md.starts_with("---\n"));
        assert!(md.contains("title: \"What is idempotence in microservices?\""));
        assert!(md.contains("model: \"gemma4:12b-mlx\""));
        assert!(md.contains("collection: \"My Library\""));
        assert!(md.contains("sources:\n  - \"Practical Microservices"));
        assert!(md.contains("# What is idempotence in microservices?"));
        assert!(md.contains("same effect [1]."));
        assert!(md.contains("## Sources"));
        assert!(md.contains("1. Practical Microservices"));
        assert!(md.contains("`/books/practical-microservices.pdf` (p.75)"));
    }

    #[test]
    fn render_handles_no_sources() {
        let mut d = doc();
        d.sources.clear();
        let md = Markdown.render(&d);
        assert!(md.contains("sources: []"));
        assert!(md.contains("_No sources retrieved._"));
    }

    #[test]
    fn yaml_quote_escapes() {
        assert_eq!(yaml_quote(r#"a "b" \c"#), r#""a \"b\" \\c""#);
    }

    #[test]
    fn write_artifact_creates_and_dedups() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = write_artifact(&Markdown, &doc(), dir.path()).unwrap();
        let p2 = write_artifact(&Markdown, &doc(), dir.path()).unwrap();
        assert_eq!(
            p1.file_name().unwrap(),
            "what-is-idempotence-in-microservices.md"
        );
        assert_eq!(
            p2.file_name().unwrap(),
            "what-is-idempotence-in-microservices-2.md"
        );
        let body = std::fs::read_to_string(&p1).unwrap();
        assert!(body.contains("# What is idempotence in microservices?"));
    }
}
