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

/// Parse one SSE `data:` line from Anthropic's `/v1/messages` stream, returning
/// any text delta. Non-data lines, non-text events, and `[DONE]` yield `None`.
fn parse_anthropic_sse_line(line: &str) -> Result<Option<String>, LlmError> {
    let line = line.trim_start();
    let Some(rest) = line.strip_prefix("data:") else {
        return Ok(None);
    };
    let rest = rest.trim();
    if rest.is_empty() || rest == "[DONE]" {
        return Ok(None);
    }
    let v: serde_json::Value =
        serde_json::from_str(rest).map_err(|e| LlmError::Decode(e.to_string()))?;
    if v.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
        if let Some(text) = v
            .get("delta")
            .and_then(|d| d.get("text"))
            .and_then(|t| t.as_str())
        {
            return Ok(Some(text.to_string()));
        }
    }
    Ok(None)
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

/// Current Claude model ids offered when the Anthropic provider is selected.
pub const ANTHROPIC_MODELS: &[&str] = &[
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-fable-5",
];

const ANTHROPIC_MAX_TOKENS: u32 = 2048;

/// Anthropic Messages API client (cloud). The API key is supplied by the user
/// via settings and never originates from code.
#[derive(Clone)]
pub struct AnthropicClient {
    api_key: String,
    http: reqwest::Client,
}

impl AnthropicClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            http: reqwest::Client::new(),
        }
    }

    fn models() -> Vec<String> {
        ANTHROPIC_MODELS.iter().map(|s| s.to_string()).collect()
    }

    async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        mut on_token: impl FnMut(&str),
    ) -> Result<String, LlmError> {
        if self.api_key.trim().is_empty() {
            return Err(LlmError::Decode(
                "no Anthropic API key set (add one in Settings)".into(),
            ));
        }
        let resp = self
            .http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&serde_json::json!({
                "model": model,
                "max_tokens": ANTHROPIC_MAX_TOKENS,
                "stream": true,
                "messages": [{ "role": "user", "content": prompt }],
            }))
            .send()
            .await?
            .error_for_status()?;

        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut full = String::new();
        while let Some(chunk) = stream.next().await {
            buf.push_str(&String::from_utf8_lossy(&chunk?));
            while let Some(nl) = buf.find('\n') {
                let line: String = buf.drain(..=nl).collect();
                if let Some(text) = parse_anthropic_sse_line(&line)? {
                    on_token(&text);
                    full.push_str(&text);
                }
            }
        }
        if let Some(text) = parse_anthropic_sse_line(&buf)? {
            on_token(&text);
            full.push_str(&text);
        }
        Ok(full)
    }
}

/// Provider-agnostic chat client. The bridge holds one of these, rebuilt from
/// settings when the provider/host/key changes.
#[derive(Clone)]
pub enum Llm {
    Ollama(OllamaClient),
    Anthropic(AnthropicClient),
}

impl Llm {
    pub async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        match self {
            Llm::Ollama(c) => c.list_models().await,
            Llm::Anthropic(_) => Ok(AnthropicClient::models()),
        }
    }

    /// Preload (Ollama only); a no-op for cloud providers.
    pub async fn warm(&self, model: &str) -> Result<(), LlmError> {
        match self {
            Llm::Ollama(c) => c.warm(model).await,
            Llm::Anthropic(_) => Ok(()),
        }
    }

    pub async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        on_token: impl FnMut(&str),
    ) -> Result<String, LlmError> {
        match self {
            Llm::Ollama(c) => c.generate_stream(model, prompt, on_token).await,
            Llm::Anthropic(c) => c.generate_stream(model, prompt, on_token).await,
        }
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
    fn parses_anthropic_text_deltas() {
        assert_eq!(
            parse_anthropic_sse_line("event: content_block_delta").unwrap(),
            None
        );
        assert_eq!(parse_anthropic_sse_line("").unwrap(), None);
        assert_eq!(parse_anthropic_sse_line("data: [DONE]").unwrap(), None);
        assert_eq!(
            parse_anthropic_sse_line(
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#
            )
            .unwrap(),
            Some("Hello".into())
        );
        // message_start / ping events carry no text delta.
        assert_eq!(
            parse_anthropic_sse_line(r#"data: {"type":"message_start","message":{}}"#).unwrap(),
            None
        );
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
