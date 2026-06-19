//! LLM synthesis over reranked passages. Provider-agnostic surface with an
//! Ollama implementation (local, streaming). Cloud providers can be added behind
//! the same `OllamaClient`-shaped API later.
//!
//! The grounded prompt (answer only from numbered sources, cite as `[n]`) is built
//! by [`build_prompt`], mirroring the Python engine's synthesis contract.

use futures::StreamExt;
use ls_query::SearchResult;

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error("decode: {0}")]
    Decode(String),
}

const SYSTEM: &str = "You answer strictly from the numbered sources below. Cite the sources \
you use as [n]. If the sources do not contain the answer, say so plainly. Do not invent facts \
beyond the sources.";

/// Assemble a grounded prompt from reranked passages with citation markers.
pub fn build_prompt(question: &str, results: &[SearchResult]) -> String {
    let mut sources = String::new();
    for r in results {
        sources.push_str(&format!("[{}] {}\n{}\n\n", r.rank, r.citation, r.text));
    }
    format!("{SYSTEM}\n\nSources:\n{sources}\nQuestion: {question}\nAnswer (with [n] citations):")
}

/// Parse one NDJSON line from Ollama `/api/generate` (stream=true).
/// Returns `(token_chunk, done)`; ignores blank lines (`None`).
fn parse_generate_line(line: &str) -> Result<Option<(String, bool)>, LlmError> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    let v: serde_json::Value =
        serde_json::from_str(line).map_err(|e| LlmError::Decode(e.to_string()))?;
    let token = v
        .get("response")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();
    let done = v.get("done").and_then(|d| d.as_bool()).unwrap_or(false);
    Ok(Some((token, done)))
}

/// Parse the `/api/tags` response into a list of model names.
fn parse_tags(body: &str) -> Result<Vec<String>, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| LlmError::Decode(e.to_string()))?;
    Ok(v.get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default())
}

/// Local Ollama client.
pub struct OllamaClient {
    base: String,
    http: reqwest::Client,
}

impl OllamaClient {
    pub fn new(host: &str) -> Self {
        Self {
            base: host.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
        }
    }

    /// Models available locally (`ollama list`).
    pub async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let body = self
            .http
            .get(format!("{}/api/tags", self.base))
            .send()
            .await?
            .text()
            .await?;
        parse_tags(&body)
    }

    /// Stream a completion, invoking `on_token` per chunk; returns the full text.
    pub async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, LlmError> {
        let resp = self
            .http
            .post(format!("{}/api/generate", self.base))
            .json(&serde_json::json!({ "model": model, "prompt": prompt, "stream": true }))
            .send()
            .await?
            .error_for_status()?;

        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut full = String::new();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buf.push_str(&String::from_utf8_lossy(&bytes));
            // Process complete newline-delimited JSON objects.
            while let Some(nl) = buf.find('\n') {
                let line: String = buf.drain(..=nl).collect();
                if let Some((token, done)) = parse_generate_line(&line)? {
                    if !token.is_empty() {
                        on_token(&token);
                        full.push_str(&token);
                    }
                    if done {
                        return Ok(full);
                    }
                }
            }
        }
        // Flush any trailing partial line.
        if let Some((token, _)) = parse_generate_line(&buf)? {
            if !token.is_empty() {
                on_token(&token);
                full.push_str(&token);
            }
        }
        Ok(full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(rank: usize, citation: &str, text: &str) -> SearchResult {
        SearchResult {
            rank,
            score: 1.0,
            text: text.into(),
            citation: citation.into(),
            title: "T".into(),
            author: None,
            chapter: None,
            page: None,
            source_path: "/p".into(),
            book_id: "b".into(),
            id: format!("b:{rank}"),
        }
    }

    #[test]
    fn prompt_includes_numbered_sources_and_question() {
        let p = build_prompt(
            "how do coroutines work?",
            &[result(
                1,
                "Kotlin · p.5",
                "Coroutines suspend without blocking.",
            )],
        );
        assert!(p.contains("[1] Kotlin · p.5"));
        assert!(p.contains("Coroutines suspend without blocking."));
        assert!(p.contains("how do coroutines work?"));
        assert!(p.to_lowercase().contains("cite"));
    }

    #[test]
    fn parses_generate_stream_lines() {
        assert_eq!(parse_generate_line("").unwrap(), None);
        assert_eq!(
            parse_generate_line(r#"{"response":"Hello","done":false}"#).unwrap(),
            Some(("Hello".into(), false))
        );
        assert_eq!(
            parse_generate_line(r#"{"response":"","done":true}"#).unwrap(),
            Some(("".into(), true))
        );
        assert!(parse_generate_line("not json").is_err());
    }

    #[test]
    fn parses_tags_into_model_names() {
        let body = r#"{"models":[{"name":"gemma4:12b-mlx"},{"name":"llama3.1:8b"}]}"#;
        assert_eq!(
            parse_tags(body).unwrap(),
            vec!["gemma4:12b-mlx", "llama3.1:8b"]
        );
        assert_eq!(
            parse_tags(r#"{"models":[]}"#).unwrap(),
            Vec::<String>::new()
        );
    }
}
