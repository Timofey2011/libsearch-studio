//! P4a slice 2 — answer-side fixtures: hard invariants on a REAL generation at
//! temperature 0, proving memory/history can't contaminate grounded citations.
//!
//!   1. The model never emits an `[n]` that exists only in history or notes —
//!      only the current Sources block's numbers are citable.
//!   2. A notebook note contradicting a book must not flip the cited answer
//!      (sources take precedence, as the prompt header demands).
//!
//! (A third planned invariant — "retry after a note edit re-reads the note" —
//! holds by construction: `ask()` loads the note fresh from SQLite on every
//! call, retries included; no cached copy exists to go stale.)
//!
//! Needs a running local Ollama with the fixture model pulled. Run on demand:
//!
//!     cargo test -p ls-llm --features fixtures -- --nocapture
//!
//! Env overrides: LS_OLLAMA_HOST (default http://localhost:11434),
//! LS_FIXTURE_MODEL (default deepseek-r1:1.5b — small and fast).
#![cfg(feature = "fixtures")]

use ls_llm::{build_prompt_with_history, GenOpts, HistoryTurn, OllamaClient};
use ls_query::SearchResult;

fn source(rank: usize, citation: &str, text: &str) -> SearchResult {
    SearchResult {
        rank,
        score: 0.9,
        text: text.into(),
        citation: citation.into(),
        title: citation.into(),
        author: None,
        chapter: None,
        page: Some(1),
        source_path: "/fixture".into(),
        book_id: "fx".into(),
        id: format!("fx:{rank}"),
    }
}

fn turn(role: &str, content: &str) -> HistoryTurn {
    HistoryTurn {
        role: role.into(),
        content: content.into(),
    }
}

/// All `[n]` citation numbers appearing in `text`.
fn cited_numbers(text: &str) -> Vec<u32> {
    let mut out = Vec::new();
    let bytes: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == '[' {
            let mut j = i + 1;
            let mut num = String::new();
            while j < bytes.len()
                && (bytes[j].is_ascii_digit() || bytes[j] == ',' || bytes[j] == ' ')
            {
                num.push(bytes[j]);
                j += 1;
            }
            if j < bytes.len() && bytes[j] == ']' && num.chars().any(|c| c.is_ascii_digit()) {
                for part in num.split(',') {
                    if let Ok(n) = part.trim().parse() {
                        out.push(n);
                    }
                }
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

async fn generate(prompt: &str) -> String {
    let host = std::env::var("LS_OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".into());
    let model = std::env::var("LS_FIXTURE_MODEL").unwrap_or_else(|_| "deepseek-r1:1.5b".into());
    let client = OllamaClient::new(&host);
    let opts = GenOpts {
        temperature: Some(0.0),
        seed: Some(42),
    };
    let (answer, _) = client
        .generate_stream_opts(&model, prompt, opts, |_| {}, |_| {})
        .await
        .expect("Ollama reachable with the fixture model pulled (see file header)");
    answer
}

#[tokio::test(flavor = "multi_thread")]
async fn citations_stay_within_current_sources_and_notes_never_flip_facts() {
    // Invented domain ("Zorblatt protocol") so the model has no prior knowledge —
    // whatever it asserts must come from the fixture sources.
    let results = [
        source(
            1,
            "Fixture Systems Book · p.1",
            "The Zorblatt protocol requires exactly three acknowledgements before a commit \
             is considered durable. This is a strict invariant of the protocol.",
        ),
        source(
            2,
            "Fixture Networking Book · p.9",
            "Unrelated filler: the Quux framing layer uses variable-length headers.",
        ),
    ];
    // History tempts with a stale [7]; the builder strips assistant markers, and
    // the model must not resurrect them.
    let history = [
        turn("user", "tell me about the Zorblatt protocol"),
        turn(
            "assistant",
            "The Zorblatt protocol is a commit protocol [7] used in distributed systems [7, 9].",
        ),
    ];
    // The note contradicts source [1] AND tempts with a bogus [5].
    let notes = "I'm fairly sure the Zorblatt protocol needs five acknowledgements [5], \
                 not what the books say.";

    let (prompt, meta) = build_prompt_with_history(
        "How many acknowledgements does the Zorblatt protocol require before commit?",
        &results,
        &history,
        Some(notes),
    );
    assert!(meta.notes_injected, "fixture note must be injected");

    let answer = generate(&prompt).await;
    println!("--- fixture answer ---\n{answer}\n----------------------");

    // Invariant 1: every cited number exists in the CURRENT Sources block.
    // (Whether the model cites AT ALL is a quality question for the golden set —
    // tiny fixture models sometimes write literal "[n]" — so no-citations is
    // only warned, not failed: the contamination invariant is what's hard here.)
    let cited = cited_numbers(&answer);
    if cited.is_empty() {
        println!("warning: model emitted no numeric citations (quality, not contamination)");
    }
    for n in &cited {
        assert!(
            *n == 1 || *n == 2,
            "cited [{n}] which exists only in history/notes — contamination! cited={cited:?}"
        );
    }

    // Invariant 2: the note's contradiction must not flip the sourced fact.
    let low = answer.to_lowercase();
    assert!(
        low.contains("three") || low.contains(" 3 ") || low.contains("3."),
        "answer abandoned the sourced fact (three acknowledgements): {answer}"
    );
}
