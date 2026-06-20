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

/// One prior chat turn, used to give the model conversational context for follow-ups.
#[derive(Debug, Clone)]
pub struct HistoryTurn {
    /// "user" or "assistant".
    pub role: String,
    pub content: String,
}

/// Keep the prompt bounded: only the most recent turns, each truncated.
const MAX_HISTORY_TURNS: usize = 6;
const MAX_TURN_CHARS: usize = 1500;

/// Assemble a grounded prompt from reranked passages with citation markers.
pub fn build_prompt(question: &str, results: &[SearchResult]) -> String {
    build_prompt_with_history(question, results, &[])
}

/// Like [`build_prompt`] but prepends recent conversation turns so the model can
/// resolve follow-ups ("what about X?"). Retrieval still targets the current
/// question; history is context only, not a source to cite.
pub fn build_prompt_with_history(
    question: &str,
    results: &[SearchResult],
    history: &[HistoryTurn],
) -> String {
    let mut sources = String::new();
    for r in results {
        sources.push_str(&format!("[{}] {}\n{}\n\n", r.rank, r.citation, r.text));
    }

    let mut convo = String::new();
    let start = history.len().saturating_sub(MAX_HISTORY_TURNS);
    for turn in &history[start..] {
        let speaker = if turn.role == "assistant" {
            "Assistant"
        } else {
            "User"
        };
        let content = truncate(&turn.content, MAX_TURN_CHARS);
        convo.push_str(&format!("{speaker}: {content}\n"));
    }
    let history_block = if convo.is_empty() {
        String::new()
    } else {
        format!("Conversation so far:\n{convo}\n")
    };

    format!(
        "{SYSTEM}\n\n{history_block}Sources:\n{sources}\nQuestion: {question}\nAnswer (with [n] citations):"
    )
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
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

/// Context window passed to Ollama. Capping this is important for latency: some
/// models register a very large default context (e.g. 262144), whose KV-cache
/// prefill is slow. Our prompts are a handful of ~400-token passages, so 8192 is
/// ample and keeps first-token fast.
const NUM_CTX: u32 = 8192;

/// Local Ollama client.
#[derive(Clone)]
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

    /// Preload a model into memory without generating, so the first real request
    /// is warm (cold-load of a multi-GB model otherwise dominates first-token latency).
    pub async fn warm(&self, model: &str) -> Result<(), LlmError> {
        self.http
            .post(format!("{}/api/generate", self.base))
            .json(&serde_json::json!({ "model": model, "prompt": "", "keep_alive": "30m" }))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
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
            .json(&serde_json::json!({
                "model": model,
                "prompt": prompt,
                "stream": true,
                "options": { "num_ctx": NUM_CTX }
            }))
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
    fn prompt_with_history_includes_recent_turns() {
        let history = vec![
            HistoryTurn {
                role: "user".into(),
                content: "what are coroutines?".into(),
            },
            HistoryTurn {
                role: "assistant".into(),
                content: "Lightweight threads [1].".into(),
            },
        ];
        let p = build_prompt_with_history(
            "how do they differ from threads?",
            &[result(1, "Kotlin · p.5", "Coroutines suspend.")],
            &history,
        );
        assert!(p.contains("Conversation so far:"));
        assert!(p.contains("User: what are coroutines?"));
        assert!(p.contains("Assistant: Lightweight threads [1]."));
        assert!(p.contains("how do they differ from threads?"));
        // No history -> no conversation block.
        assert!(!build_prompt("q", &[]).contains("Conversation so far"));
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
