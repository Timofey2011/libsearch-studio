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
    #[error("the provider stalled — no data for {0}s (connection dropped or model too slow)")]
    Timeout(u64),
}

/// How long to wait for the initial TCP/TLS connect before giving up. Deliberately
/// short: a wrong host/port or an unreachable Ollama should fail fast, not hang.
const CONNECT_TIMEOUT_SECS: u64 = 15;
/// Idle deadline between streamed chunks. Generous so a local model's cold-load
/// (multi-GB into RAM before the first token) isn't cut off, but bounded so a
/// dropped connection or a wedged provider can't hang the ask forever. Reset on
/// every chunk, so it caps the gap between tokens, not the total generation time.
const STREAM_IDLE_TIMEOUT_SECS: u64 = 120;

/// A reqwest client with a connect timeout, shared by every provider. Streaming
/// bodies must NOT get a request-level `.timeout()` (it would kill legitimate long
/// generations); the per-chunk idle deadline lives in `run_stream` instead.
fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
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

/// Reserve this many tokens of the context window for the model's output, so a
/// long prompt can't crowd out the answer.
const OUTPUT_HEADROOM_TOKENS: u32 = 2048;

/// Rough token budget for the assembled prompt: the context window minus output
/// headroom, times a safety margin (our char-based estimate is approximate).
fn prompt_token_budget() -> usize {
    ((NUM_CTX.saturating_sub(OUTPUT_HEADROOM_TOKENS)) as f32 * 0.9) as usize
}

/// Cheap, dependency-free token estimate. English ≈ chars/4; Cyrillic BPE-fragments
/// roughly twice as hard, so use a smaller divisor when the text is substantially
/// Cyrillic (this app targets mixed EN+RU libraries).
fn estimate_tokens(s: &str) -> usize {
    let total = s.chars().count();
    if total == 0 {
        return 0;
    }
    let cyrillic = s.chars().filter(|c| ('\u{0400}'..='\u{04FF}').contains(c)).count();
    let divisor = if cyrillic * 5 >= total { 2.5 } else { 4.0 };
    ((total as f32) / divisor).ceil() as usize
}

/// Like [`build_prompt`] but prepends recent conversation turns so the model can
/// resolve follow-ups ("what about X?"). Retrieval still targets the current
/// question; history is context only, not a source to cite. If the assembled
/// prompt would exceed the context budget, the OLDEST history turns are dropped
/// first — the grounding sources and the question are never trimmed.
pub fn build_prompt_with_history(
    question: &str,
    results: &[SearchResult],
    history: &[HistoryTurn],
) -> String {
    let mut sources = String::new();
    for r in results {
        sources.push_str(&format!("[{}] {}\n{}\n\n", r.rank, r.citation, r.text));
    }

    // Most-recent window, each turn truncated (unchanged default behaviour).
    let start = history.len().saturating_sub(MAX_HISTORY_TURNS);
    let mut window: Vec<(&str, String)> = history[start..]
        .iter()
        .map(|turn| {
            let speaker = if turn.role == "assistant" { "Assistant" } else { "User" };
            (speaker, truncate(&turn.content, MAX_TURN_CHARS))
        })
        .collect();

    let assemble = |turns: &[(&str, String)]| -> String {
        let mut convo = String::new();
        for (speaker, content) in turns {
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
    };

    // Drop oldest history turns until the prompt fits the budget (or none remain).
    let budget = prompt_token_budget();
    let mut prompt = assemble(&window);
    while !window.is_empty() && estimate_tokens(&prompt) > budget {
        window.remove(0);
        prompt = assemble(&window);
    }
    prompt
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}

/// Token usage for one generation (prompt vs. completion).
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct Usage {
    pub in_tokens: u32,
    pub out_tokens: u32,
}

/// One parsed streaming line: any answer content and/or reasoning ("thinking")
/// deltas, token-usage updates, plus whether the stream is done.
#[derive(Debug, Default, PartialEq)]
struct LineOut {
    content: Option<String>,
    reasoning: Option<String>,
    in_tokens: Option<u32>,
    out_tokens: Option<u32>,
    done: bool,
}

/// Splits a content stream into answer text vs. inline `<think>…</think>`
/// reasoning, tolerating tags split across chunk boundaries.
#[derive(Default)]
struct ThinkSplitter {
    in_think: bool,
    buf: String,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Longest suffix of `s` that is a (proper) prefix of `tag` — bytes we must hold
/// back in case the next chunk completes the tag.
fn trailing_partial(s: &str, tag: &str) -> usize {
    let max = tag.len().saturating_sub(1).min(s.len());
    for k in (1..=max).rev() {
        if tag.as_bytes().starts_with(&s.as_bytes()[s.len() - k..]) {
            return k;
        }
    }
    0
}

impl ThinkSplitter {
    /// Feed a content chunk; returns `(answer_text, reasoning_text)` ready to emit.
    fn feed(&mut self, s: &str) -> (String, String) {
        self.buf.push_str(s);
        let (mut content, mut reasoning) = (String::new(), String::new());
        loop {
            let (tag, out) = if self.in_think {
                (THINK_CLOSE, &mut reasoning)
            } else {
                (THINK_OPEN, &mut content)
            };
            if let Some(i) = self.buf.find(tag) {
                out.push_str(&self.buf[..i]);
                self.buf.drain(..i + tag.len());
                self.in_think = !self.in_think;
            } else {
                let keep = trailing_partial(&self.buf, tag);
                let emit_to = self.buf.len() - keep;
                out.push_str(&self.buf[..emit_to]);
                self.buf.drain(..emit_to);
                break;
            }
        }
        (content, reasoning)
    }

    /// Flush whatever remains (stream ended mid-buffer).
    fn finish(&mut self) -> (String, String) {
        let rest = std::mem::take(&mut self.buf);
        if self.in_think {
            (String::new(), rest)
        } else {
            (rest, String::new())
        }
    }
}

/// Parse one NDJSON line from Ollama `/api/generate` (stream=true).
fn parse_generate_line(line: &str) -> Result<LineOut, LlmError> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(LineOut::default());
    }
    let v: serde_json::Value =
        serde_json::from_str(line).map_err(|e| LlmError::Decode(e.to_string()))?;
    Ok(LineOut {
        content: str_field(&v, "response"),
        reasoning: str_field(&v, "thinking"),
        // Final `done` line carries the token totals.
        in_tokens: u32_field(&v, "prompt_eval_count"),
        out_tokens: u32_field(&v, "eval_count"),
        done: v.get("done").and_then(|d| d.as_bool()).unwrap_or(false),
    })
}

/// Parse one SSE `data:` line from Anthropic's `/v1/messages` stream.
fn parse_anthropic_sse_line(line: &str) -> Result<LineOut, LlmError> {
    let Some(v) = sse_json(line)? else {
        return Ok(LineOut::default());
    };
    match v.get("type").and_then(|t| t.as_str()) {
        Some("content_block_delta") => {
            let delta = v.get("delta");
            Ok(LineOut {
                content: delta.and_then(|d| str_field(d, "text")),
                reasoning: delta.and_then(|d| str_field(d, "thinking")),
                ..Default::default()
            })
        }
        // `message_start` carries input tokens; `message_delta` the (cumulative) output.
        Some("message_start") => Ok(LineOut {
            in_tokens: v
                .get("message")
                .and_then(|m| u32_field(m.get("usage")?, "input_tokens")),
            ..Default::default()
        }),
        Some("message_delta") => Ok(LineOut {
            out_tokens: v.get("usage").and_then(|u| u32_field(u, "output_tokens")),
            ..Default::default()
        }),
        _ => Ok(LineOut::default()),
    }
}

/// Turn a failed `/chat/completions` HTTP status into an actionable message.
/// A 401/403 here is most often a non-chat model (image/embedding) rather than a
/// bad key — providers like Fireworks reject a chat call to an image model with 401.
fn chat_error_message(status: u16, body: &str) -> String {
    let snippet = body.trim();
    let snippet = if snippet.is_empty() {
        String::new()
    } else {
        format!(" — {}", snippet.chars().take(200).collect::<String>())
    };
    match status {
        401 | 403 => format!(
            "That model doesn't support chat — image and embedding models return {status}. \
             Pick a chat model, or check the API key in Settings.{snippet}"
        ),
        404 => format!("Model not found ({status}). Check the model id in Settings.{snippet}"),
        400 => format!("The provider rejected the request ({status}).{snippet}"),
        429 => format!("Rate limited ({status}). Try again in a moment."),
        _ => format!("Provider error ({status}).{snippet}"),
    }
}

/// Parse one SSE `data:` line from an OpenAI-compatible `/chat/completions` stream.
/// Reasoning models expose `delta.reasoning_content` (or `reasoning`).
fn parse_openai_sse_line(line: &str) -> Result<LineOut, LlmError> {
    let line = line.trim_start();
    if line.strip_prefix("data:").map(str::trim) == Some("[DONE]") {
        return Ok(LineOut {
            done: true,
            ..Default::default()
        });
    }
    let Some(v) = sse_json(line)? else {
        return Ok(LineOut::default());
    };
    let delta = v
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"));
    // The final chunk (with stream_options.include_usage) carries `usage`.
    let usage = v.get("usage");
    Ok(LineOut {
        content: delta.and_then(|d| str_field(d, "content")),
        reasoning: delta
            .and_then(|d| str_field(d, "reasoning_content").or(str_field(d, "reasoning"))),
        in_tokens: usage.and_then(|u| u32_field(u, "prompt_tokens")),
        out_tokens: usage.and_then(|u| u32_field(u, "completion_tokens")),
        done: false,
    })
}

/// Extract a non-empty string field from a JSON object.
fn str_field(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(|t| t.as_str())
        .filter(|t| !t.is_empty())
        .map(String::from)
}

/// Extract a `u32` field from a JSON object.
fn u32_field(v: &serde_json::Value, key: &str) -> Option<u32> {
    v.get(key).and_then(|n| n.as_u64()).map(|n| n as u32)
}

/// Parse an SSE `data:` line into JSON (None for non-data / blank / `[DONE]`).
fn sse_json(line: &str) -> Result<Option<serde_json::Value>, LlmError> {
    let line = line.trim_start();
    let Some(rest) = line.strip_prefix("data:") else {
        return Ok(None);
    };
    let rest = rest.trim();
    if rest.is_empty() || rest == "[DONE]" {
        return Ok(None);
    }
    serde_json::from_str(rest)
        .map(Some)
        .map_err(|e| LlmError::Decode(e.to_string()))
}

/// Route one parsed line to the callbacks, splitting inline `<think>` out of
/// content. Returns whether the stream is done.
fn emit_line(
    lo: LineOut,
    splitter: &mut ThinkSplitter,
    full: &mut String,
    on_token: &mut impl FnMut(&str),
    on_reasoning: &mut impl FnMut(&str),
) -> bool {
    if let Some(r) = lo.reasoning {
        if !r.is_empty() {
            on_reasoning(&r);
        }
    }
    if let Some(c) = lo.content {
        let (text, reason) = splitter.feed(&c);
        if !reason.is_empty() {
            on_reasoning(&reason);
        }
        if !text.is_empty() {
            on_token(&text);
            full.push_str(&text);
        }
    }
    lo.done
}

/// Drive a streaming response: parse each line, split inline `<think>` out of
/// content, emit answer text via `on_token` and reasoning via `on_reasoning`,
/// and accumulate token usage. Returns the answer text + final usage.
async fn run_stream<P>(
    resp: reqwest::Response,
    mut parse_line: P,
    mut on_token: impl FnMut(&str),
    mut on_reasoning: impl FnMut(&str),
) -> Result<(String, Usage), LlmError>
where
    P: FnMut(&str) -> Result<LineOut, LlmError>,
{
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    let mut full = String::new();
    let mut splitter = ThinkSplitter::default();
    let mut usage = Usage::default();

    let idle = std::time::Duration::from_secs(STREAM_IDLE_TIMEOUT_SECS);
    loop {
        // Bound the wait for the NEXT chunk (not the whole stream), so a dropped
        // connection or a wedged model surfaces a clean timeout instead of hanging.
        let chunk = match tokio::time::timeout(idle, stream.next()).await {
            Ok(Some(chunk)) => chunk,
            Ok(None) => break,
            Err(_elapsed) => return Err(LlmError::Timeout(STREAM_IDLE_TIMEOUT_SECS)),
        };
        buf.push_str(&String::from_utf8_lossy(&chunk?));
        while let Some(nl) = buf.find('\n') {
            let line: String = buf.drain(..=nl).collect();
            let lo = parse_line(&line)?;
            if let Some(i) = lo.in_tokens {
                usage.in_tokens = i;
            }
            if let Some(o) = lo.out_tokens {
                usage.out_tokens = o;
            }
            if emit_line(
                lo,
                &mut splitter,
                &mut full,
                &mut on_token,
                &mut on_reasoning,
            ) {
                let (t, _r) = splitter.finish();
                if !t.is_empty() {
                    on_token(&t);
                    full.push_str(&t);
                }
                return Ok((full, usage));
            }
        }
    }
    let lo = parse_line(&buf)?;
    if let Some(i) = lo.in_tokens {
        usage.in_tokens = i;
    }
    if let Some(o) = lo.out_tokens {
        usage.out_tokens = o;
    }
    emit_line(
        lo,
        &mut splitter,
        &mut full,
        &mut on_token,
        &mut on_reasoning,
    );
    let (t, _r) = splitter.finish();
    if !t.is_empty() {
        on_token(&t);
        full.push_str(&t);
    }
    Ok((full, usage))
}

/// Parse an OpenAI-compatible `/models` response (`{ "data": [{ "id": ... }] }`).
/// Heuristic: is this model id a chat/completions model (vs. image, embedding,
/// audio, rerank, moderation…)? Used to keep non-chat models out of the model
/// picker — sending a chat request to e.g. a Fireworks image model 401/404s.
/// Kept deliberately generic so it works across OpenAI, Gemini, Fireworks, etc.
pub fn is_chat_model(id: &str) -> bool {
    let l = id.to_lowercase();
    const NON_CHAT: &[&str] = &[
        "flux",
        "stable-diffusion",
        "sd3",
        "sdxl",
        "dall-e",
        "playground-v",
        "kandinsky",
        "-image",
        "image-",
        "embed",
        "whisper",
        "tts",
        "-audio",
        "rerank",
        "clip",
        "moderation",
        "-vae",
        "upscal",
        "controlnet",
        "guard",
    ];
    !NON_CHAT.iter().any(|n| l.contains(n))
}

fn parse_openai_models(body: &str) -> Result<Vec<String>, LlmError> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| LlmError::Decode(e.to_string()))?;
    Ok(v.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default())
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
const NUM_CTX: u32 = 16384;

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
            http: http_client(),
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

    /// Stream a completion; answer chunks go to `on_token`, reasoning to
    /// `on_reasoning`. Returns the full answer text.
    pub async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        on_token: impl FnMut(&str),
        on_reasoning: impl FnMut(&str),
    ) -> Result<(String, Usage), LlmError> {
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
        run_stream(resp, parse_generate_line, on_token, on_reasoning).await
    }
}

/// Current Claude model ids offered when the Anthropic provider is selected.
pub const ANTHROPIC_MODELS: &[&str] = &[
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-fable-5",
];

// Anthropic requires an explicit max_tokens. 2048 silently truncated long grounded
// answers mid-sentence; 4096 comfortably covers them. Output is billed as generated,
// so a higher cap only permits longer answers, it doesn't cost more per se.
const ANTHROPIC_MAX_TOKENS: u32 = 4096;

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
            http: http_client(),
        }
    }

    fn models() -> Vec<String> {
        ANTHROPIC_MODELS.iter().map(|s| s.to_string()).collect()
    }

    async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        on_token: impl FnMut(&str),
        on_reasoning: impl FnMut(&str),
    ) -> Result<(String, Usage), LlmError> {
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
        run_stream(resp, parse_anthropic_sse_line, on_token, on_reasoning).await
    }
}

/// Client for any OpenAI-compatible chat API (OpenAI, Gemini's compat endpoint,
/// Fireworks, Ollama Cloud) — same wire format, different `base_url`.
#[derive(Clone)]
pub struct OpenAiCompatClient {
    base_url: String,
    api_key: String,
    http: reqwest::Client,
}

impl OpenAiCompatClient {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            http: http_client(),
        }
    }

    pub async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let body = self
            .http
            .get(format!("{}/models", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        parse_openai_models(&body)
    }

    async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        on_token: impl FnMut(&str),
        on_reasoning: impl FnMut(&str),
    ) -> Result<(String, Usage), LlmError> {
        if self.api_key.trim().is_empty() {
            return Err(LlmError::Decode(
                "no API key set (add one in Settings)".into(),
            ));
        }
        let resp = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({
                "model": model,
                "stream": true,
                "stream_options": { "include_usage": true },
                "messages": [{ "role": "user", "content": prompt }],
            }))
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Decode(chat_error_message(status.as_u16(), &body)));
        }
        run_stream(resp, parse_openai_sse_line, on_token, on_reasoning).await
    }
}

/// Provider-agnostic chat client. The bridge holds one of these, rebuilt from
/// settings when the provider/host/key changes.
#[derive(Clone)]
pub enum Llm {
    Ollama(OllamaClient),
    Anthropic(AnthropicClient),
    OpenAiCompat(OpenAiCompatClient),
}

impl Llm {
    pub async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        match self {
            Llm::Ollama(c) => c.list_models().await,
            Llm::Anthropic(_) => Ok(AnthropicClient::models()),
            Llm::OpenAiCompat(c) => c.list_models().await,
        }
    }

    /// Preload (Ollama only); a no-op for cloud providers.
    pub async fn warm(&self, model: &str) -> Result<(), LlmError> {
        match self {
            Llm::Ollama(c) => c.warm(model).await,
            Llm::Anthropic(_) | Llm::OpenAiCompat(_) => Ok(()),
        }
    }

    pub async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        on_token: impl FnMut(&str),
        on_reasoning: impl FnMut(&str),
    ) -> Result<(String, Usage), LlmError> {
        match self {
            Llm::Ollama(c) => {
                c.generate_stream(model, prompt, on_token, on_reasoning)
                    .await
            }
            Llm::Anthropic(c) => {
                c.generate_stream(model, prompt, on_token, on_reasoning)
                    .await
            }
            Llm::OpenAiCompat(c) => {
                c.generate_stream(model, prompt, on_token, on_reasoning)
                    .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimate_is_script_aware() {
        // Latin ≈ chars/4.
        assert_eq!(estimate_tokens(&"a".repeat(40)), 10);
        // Substantially Cyrillic text uses the smaller divisor (~chars/2.5).
        let ru = "экономика".repeat(5); // 45 Cyrillic chars
        assert!(estimate_tokens(&ru) > (ru.chars().count() / 4));
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn prompt_budget_leaves_output_headroom() {
        assert!(prompt_token_budget() < NUM_CTX as usize);
        assert!(prompt_token_budget() > 8000);
    }

    #[test]
    fn chat_model_filter() {
        // Chat models pass.
        for m in [
            "gpt-4o",
            "gemini-2.0-flash",
            "accounts/fireworks/models/kimi-k2p7-code",
            "deepseek-r1",
            "claude-opus-4-8",
        ] {
            assert!(is_chat_model(m), "{m} should be a chat model");
        }
        // Image / embedding / audio / rerank / moderation are filtered out.
        for m in [
            "accounts/fireworks/models/flux-1-schnell",
            "stable-diffusion-xl",
            "dall-e-3",
            "text-embedding-3-large",
            "whisper-1",
            "tts-1",
            "bge-reranker-v2-m3",
            "omni-moderation-latest",
        ] {
            assert!(!is_chat_model(m), "{m} should be filtered out");
        }
    }

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
    fn think_splitter_extracts_reasoning_across_chunks() {
        let mut sp = ThinkSplitter::default();
        let (mut content, mut reasoning) = (String::new(), String::new());
        // Feed the tags split across chunk boundaries.
        for chunk in ["Hi <thi", "nk>plan", "ning</thi", "nk> done"] {
            let (c, r) = sp.feed(chunk);
            content.push_str(&c);
            reasoning.push_str(&r);
        }
        let (c, r) = sp.finish();
        content.push_str(&c);
        reasoning.push_str(&r);
        assert_eq!(content, "Hi  done");
        assert_eq!(reasoning, "planning");
    }

    #[test]
    fn think_splitter_passes_plain_text() {
        let mut sp = ThinkSplitter::default();
        let (c, r) = sp.feed("just an answer");
        assert_eq!(c, "just an answer");
        assert_eq!(r, "");
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

    fn content(s: &str) -> LineOut {
        LineOut {
            content: Some(s.into()),
            ..Default::default()
        }
    }

    #[test]
    fn parses_generate_stream_lines() {
        assert_eq!(parse_generate_line("").unwrap(), LineOut::default());
        assert_eq!(
            parse_generate_line(r#"{"response":"Hello","done":false}"#).unwrap(),
            content("Hello")
        );
        assert_eq!(
            parse_generate_line(r#"{"response":"","done":true}"#).unwrap(),
            LineOut {
                done: true,
                ..Default::default()
            }
        );
        // Ollama thinking field is captured as reasoning.
        assert_eq!(
            parse_generate_line(r#"{"thinking":"hmm","done":false}"#).unwrap(),
            LineOut {
                reasoning: Some("hmm".into()),
                ..Default::default()
            }
        );
        assert!(parse_generate_line("not json").is_err());
    }

    #[test]
    fn parses_anthropic_text_deltas() {
        assert_eq!(
            parse_anthropic_sse_line("event: content_block_delta").unwrap(),
            LineOut::default()
        );
        assert_eq!(
            parse_anthropic_sse_line("data: [DONE]").unwrap(),
            LineOut::default()
        );
        assert_eq!(
            parse_anthropic_sse_line(
                r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#
            )
            .unwrap(),
            content("Hello")
        );
        assert_eq!(
            parse_anthropic_sse_line(r#"data: {"type":"message_start","message":{}}"#).unwrap(),
            LineOut::default()
        );
    }

    #[test]
    fn parses_openai_content_and_reasoning() {
        assert_eq!(
            parse_openai_sse_line("data: [DONE]").unwrap(),
            LineOut {
                done: true,
                ..Default::default()
            }
        );
        assert_eq!(
            parse_openai_sse_line(r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#).unwrap(),
            LineOut::default()
        );
        assert_eq!(
            parse_openai_sse_line(r#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#).unwrap(),
            content("Hi")
        );
        assert_eq!(
            parse_openai_sse_line(r#"data: {"choices":[{"delta":{"reasoning_content":"think"}}]}"#)
                .unwrap(),
            LineOut {
                reasoning: Some("think".into()),
                ..Default::default()
            }
        );
        assert_eq!(
            parse_openai_models(r#"{"data":[{"id":"gpt-4o"},{"id":"o3"}]}"#).unwrap(),
            vec!["gpt-4o", "o3"]
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
