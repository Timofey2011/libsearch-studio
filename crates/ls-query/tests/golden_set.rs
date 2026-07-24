//! P4b — golden-set retrieval harness: the full pipeline (embed → hybrid
//! vector+FTS+fuzzy → RRF → rerank) runs against a small committed corpus
//! (tests/fixtures/corpus/, 8 mini-books, EN+RU) with expected-book assertions
//! per query. This is the regression gate for retrieval changes: thresholds,
//! chunking, fusion, query-rewrite experiments.
//!
//! Covers: direct keyword hits, pure paraphrases, RU queries, cross-lingual
//! (EN query → RU-only content and vice versa), typo/spell-repair, follow-up
//! context fusion, multi-collection fan-out, and a no-answer noise probe
//! (tiering honesty: nothing above min_relevance).
//!
//! Loads the real bge-m3 + reranker (int8 preferred), so it's models-gated and
//! run on demand (~2-4 min on CPU):
//!
//!     cargo test -p ls-query --features models --test golden_set -- --nocapture
#![cfg(feature = "models")]

use ls_core::{Chunk, Format};
use ls_embed::{Embedder, Reranker};
use ls_index::Store;
use ls_query::{search_multi, SearchResult};

/// Mirrors the app's default confident-tier floor (ls-app settings).
const MIN_RELEVANCE: f32 = 0.15;
const FINAL_K: usize = 5;
const HYBRID_K: usize = 12;

fn models_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn embedder() -> Embedder {
    let dir = std::env::var("LS_BGE_M3_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| models_root().join("bge-m3"));
    Embedder::load(dir).expect("load bge-m3")
}

fn reranker() -> Reranker {
    if let Ok(dir) = std::env::var("LS_RERANKER_DIR") {
        return Reranker::load(dir).expect("load reranker (env)");
    }
    let int8 = models_root().join("bge-reranker-v2-m3-int8");
    let dir = if int8.join("model.onnx").exists() {
        int8
    } else {
        models_root().join("bge-reranker-v2-m3")
    };
    Reranker::load(dir).expect("load reranker")
}

/// One fixture book: (book_id, title, embedded file content).
const CORPUS: &[(&str, &str, &str)] = &[
    (
        "sagas",
        "Saga Patterns",
        include_str!("fixtures/corpus/sagas.md"),
    ),
    (
        "tcp",
        "TCP Internals",
        include_str!("fixtures/corpus/tcp.md"),
    ),
    (
        "bloom",
        "Probabilistic Structures",
        include_str!("fixtures/corpus/bloom.md"),
    ),
    (
        "options",
        "Options Pricing",
        include_str!("fixtures/corpus/options.md"),
    ),
    (
        "rust",
        "Rust Ownership",
        include_str!("fixtures/corpus/rust-ownership.md"),
    ),
    (
        "repl-ru",
        "Репликация данных",
        include_str!("fixtures/corpus/replication-ru.md"),
    ),
    (
        "port-ru",
        "Инвестиционный портфель",
        include_str!("fixtures/corpus/portfolio-ru.md"),
    ),
    (
        "k8s-ru",
        "Kubernetes по-русски",
        include_str!("fixtures/corpus/k8s-ru.md"),
    ),
];

/// Paragraph-per-chunk (chunking has its own unit tests; this harness targets
/// retrieval). A standalone `# Heading` paragraph is CONSUMED as the running
/// chapter label (mirroring the md ingest scanner) — it never becomes a chunk,
/// so adding headings to a fixture leaves every existing chunk byte-identical.
/// A book with headings is stored as Epub with no page stamps (chapters render
/// as `Ch. …`); a heading-free book keeps the original Pdf/page shape.
fn chunks_for(book_id: &str, title: &str, text: &str) -> Vec<Chunk> {
    let mut labeled: Vec<(Option<String>, &str)> = Vec::new();
    let mut current: Option<String> = None;
    for para in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if let Some(rest) = para.strip_prefix("# ").or_else(|| para.strip_prefix("## ")) {
            current = Some(rest.trim().to_string());
        } else {
            labeled.push((current.clone(), para));
        }
    }
    let chaptered = labeled.iter().any(|(c, _)| c.is_some());
    let mut out = Vec::new();
    let mut offset = 0usize;
    for (i, (chapter, para)) in labeled.into_iter().enumerate() {
        out.push(Chunk {
            id: format!("{book_id}:{i}"),
            book_id: book_id.into(),
            title: title.into(),
            author: None,
            source_path: format!("/fixtures/{book_id}.md"),
            format: if chaptered { Format::Epub } else { Format::Pdf },
            chapter,
            page: if chaptered { None } else { Some(i as u32 + 1) },
            loc_start: offset,
            loc_end: offset + para.chars().count(),
            text: para.replace('\n', " "),
            vector: None,
        });
        offset += para.chars().count() + 2;
    }
    out
}

async fn build_store(dir: &std::path::Path, emb: &mut Embedder, books: &[usize]) -> Store {
    let store = Store::open_or_create(dir.to_str().unwrap(), "chunks")
        .await
        .expect("create store");
    let mut chunks: Vec<Chunk> = Vec::new();
    for &b in books {
        let (id, title, text) = CORPUS[b];
        chunks.extend(chunks_for(id, title, text));
    }
    for batch in chunks.chunks_mut(16) {
        let texts: Vec<&str> = batch.iter().map(|c| c.text.as_str()).collect();
        let vecs = emb.embed(&texts).expect("embed corpus");
        for (c, v) in batch.iter_mut().zip(vecs) {
            c.vector = Some(v);
        }
    }
    store.add_chunks(&chunks).await.expect("add chunks");
    store.ensure_fts_index().await.expect("fts index");
    store
}

struct Case {
    name: &'static str,
    query: &'static str,
    /// Prior-turn context for follow-up fusion cases.
    context: Option<&'static str>,
    /// The book that must appear within `within_top` results (None = noise probe).
    expect_book: Option<&'static str>,
    within_top: usize,
    /// §17.2: at least ONE expected-book hit within FINAL_K must carry a
    /// chapter containing this substring (case-insensitive). Deliberately not
    /// pinned to the top chunk — intra-book tie-breaks depend on which
    /// reranker build (int8 vs f32) a developer runs.
    expect_chapter: Option<&'static str>,
}

const fn case(name: &'static str, query: &'static str, expect: &'static str, top: usize) -> Case {
    Case {
        name,
        query,
        context: None,
        expect_book: Some(expect),
        within_top: top,
        expect_chapter: None,
    }
}

const fn case_ch(
    name: &'static str,
    query: &'static str,
    expect: &'static str,
    top: usize,
    chapter: &'static str,
) -> Case {
    Case {
        name,
        query,
        context: None,
        expect_book: Some(expect),
        within_top: top,
        expect_chapter: Some(chapter),
    }
}

fn rank_of(results: &[SearchResult], book: &str) -> Option<usize> {
    results
        .iter()
        .position(|r| r.book_id == book)
        .map(|p| p + 1)
}

#[tokio::test(flavor = "multi_thread")]
async fn golden_set() {
    let mut emb = embedder();
    let mut rer = reranker();

    // Two stores exercise the multi-collection fan-out exactly as the app does
    // (EN books in one collection, RU books in the other).
    let root = std::env::temp_dir().join(format!("ls-golden-{}", std::process::id()));
    let en = build_store(&root.join("en"), &mut emb, &[0, 1, 2, 3, 4]).await;
    let ru = build_store(&root.join("ru"), &mut emb, &[5, 6, 7]).await;
    let stores = [&en, &ru];

    // Regression (caught by this harness's first run): lance's fuzzy-FTS prefix
    // anchor byte-slices the token, so Cyrillic queries panicked a worker thread
    // and silently lost the fuzzy signal (v0.5.5–v0.6.4). Must return Ok + hits.
    // Regressions this harness's first run caught in shipped code:
    // (a) lance byte-slices the fuzzy prefix anchor → Cyrillic queries panicked
    //     a worker thread (v0.5.5–v0.6.4) and lost the fuzzy signal silently;
    // (b) fst 0.4.7's Levenshtein matches nothing for non-ASCII at fuzziness ≥1,
    //     so Cyrillic fuzzy expansion is an upstream no-op regardless.
    // fts_search_fuzzy now skips non-ASCII tokens: it must return Ok (never
    // panic/error) on RU input, and still find EN typos. RU typo tolerance is
    // guaranteed end-to-end by correct_query instead (typo-ru case below).
    let fuzzy_ru = ru
        .fts_search_fuzzy("кворум репликация", 5)
        .await
        .expect("fuzzy FTS must be a clean no-op on Cyrillic (panic regression)");
    assert!(fuzzy_ru.is_empty(), "non-ASCII fuzzy is a documented no-op");
    let fuzzy_en = en
        .fts_search_fuzzy("blom fliter", 5)
        .await
        .expect("fuzzy FTS en");
    assert!(
        !fuzzy_en.is_empty(),
        "EN typo should fuzzy-match the corpus"
    );

    let cases = [
        // Direct keyword hits (EN).
        case(
            "en-keyword-saga",
            "how does saga compensation work when a step fails",
            "sagas",
            1,
        ),
        case(
            "en-keyword-tcp",
            "what is slow start in TCP congestion control",
            "tcp",
            1,
        ),
        case(
            "en-keyword-bloom",
            "how many hash functions should a bloom filter use",
            "bloom",
            1,
        ),
        case(
            "en-keyword-vol",
            "how is implied volatility used to price options",
            "options",
            1,
        ),
        case(
            "en-keyword-rust",
            "how does the borrow checker prevent dangling references",
            "rust",
            1,
        ),
        // Pure paraphrase — no shared keywords; the vector half must carry it.
        case(
            "en-paraphrase-saga",
            "undoing already-finished steps of a distributed workflow after something breaks",
            "sagas",
            3,
        ),
        // Russian queries against Russian books.
        case(
            "ru-keyword-portfolio",
            "как работает ребалансировка инвестиционного портфеля",
            "port-ru",
            1,
        ),
        case(
            "ru-keyword-quorum",
            "что такое кворум при репликации базы данных",
            "repl-ru",
            1,
        ),
        case(
            "ru-keyword-k8s",
            "как деплоймент выкатывает новую версию без простоя",
            "k8s-ru",
            1,
        ),
        // Cross-lingual: the topic exists ONLY in the other language.
        case(
            "xling-en-to-ru",
            "how does quorum based database replication decide a write succeeded",
            "repl-ru",
            3,
        ),
        case(
            "xling-ru-to-en",
            "как работает паттерн сага и компенсирующие транзакции",
            "sagas",
            3,
        ),
        // Typo path (spell-repair + fuzzy FTS).
        case(
            "typo-saga",
            "sagga compenstion when a step fails",
            "sagas",
            3,
        ),
        case("typo-bloom", "blom filter false positive rate", "bloom", 3),
        // RU typo (ф for в) — exercises the now-Cyrillic-safe fuzzy path.
        case(
            "typo-ru",
            "ребалансирофка инвестиционного портфеля",
            "port-ru",
            3,
        ),
        // §17.2 golden chapter Q/A: the citation must carry the RIGHT chapter,
        // guarding embed→store→retrieve→citation chapter plumbing end to end.
        case_ch(
            "chapter-en-compensation",
            "undoing finished saga steps with explicit compensating actions",
            "sagas",
            3,
            "compensat",
        ),
        case_ch(
            "chapter-en-orchestration",
            "choreography versus a central orchestrator coordinating a saga",
            "sagas",
            3,
            "choreography",
        ),
        case_ch(
            "chapter-ru-quorum",
            "запись подтверждена w узлами из n при кворумной репликации",
            "repl-ru",
            3,
            "кворум",
        ),
        case_ch(
            "chapter-ru-conflicts",
            "как разрешаются конфликты при мультилидерной репликации",
            "repl-ru",
            3,
            "конфликт",
        ),
        // Follow-up fusion: bare query is contentless; the context must widen it.
        Case {
            name: "fusion-followup",
            query: "why does it sometimes fail?",
            context: Some("saga pattern compensating transactions"),
            expect_book: Some("sagas"),
            within_top: 3,
            expect_chapter: None,
        },
        // Noise probe: nothing in the corpus — the confident tier must stay empty.
        Case {
            name: "noise-no-answer",
            query: "best sourdough starter hydration schedule for baking bread",
            context: None,
            expect_book: None,
            within_top: 0,
            expect_chapter: None,
        },
    ];

    let mut failures: Vec<String> = Vec::new();
    println!("\n{:<24} {:>6} {:>8}  outcome", "case", "rank", "top1");
    for c in &cases {
        let results = search_multi(
            &stores, &mut emb, &mut rer, c.query, FINAL_K, HYBRID_K, c.context, None,
        )
        .await
        .expect("search");
        let top_score = results.first().map(|r| r.score).unwrap_or(0.0);
        match c.expect_book {
            Some(book) => {
                let rank = rank_of(&results, book);
                let ok = matches!(rank, Some(r) if r <= c.within_top);
                println!(
                    "{:<24} {:>6} {:>8.3}  {}",
                    c.name,
                    rank.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
                    top_score,
                    if ok { "ok" } else { "FAIL" }
                );
                if !ok {
                    let got: Vec<&str> = results.iter().map(|r| r.book_id.as_str()).collect();
                    failures.push(format!(
                        "{}: wanted {book} within top {}, got rank {rank:?} (results: {got:?})",
                        c.name, c.within_top
                    ));
                }
                // §17.2: some expected-book hit within FINAL_K must carry the
                // expected chapter (case-insensitive substring).
                if let Some(want) = c.expect_chapter {
                    let want_lc = want.to_lowercase();
                    let hit = results.iter().any(|r| {
                        r.book_id == book
                            && r.chapter
                                .as_deref()
                                .is_some_and(|ch| ch.to_lowercase().contains(&want_lc))
                    });
                    if !hit {
                        let got: Vec<String> = results
                            .iter()
                            .filter(|r| r.book_id == book)
                            .map(|r| format!("{:?}", r.chapter))
                            .collect();
                        failures.push(format!(
                            "{}: no {book} hit carries chapter ~ \"{want}\" (chapters: {got:?})",
                            c.name
                        ));
                    }
                }
            }
            None => {
                // Tiering honesty: no result may clear the confident floor.
                let confident = results.iter().filter(|r| r.score >= MIN_RELEVANCE).count();
                let ok = confident == 0;
                println!(
                    "{:<24} {:>6} {:>8.3}  {}",
                    c.name,
                    "-",
                    top_score,
                    if ok {
                        "ok (no confident hit)"
                    } else {
                        "FAIL (confident noise)"
                    }
                );
                if !ok {
                    failures.push(format!(
                        "{}: noise query produced {confident} result(s) above min_relevance",
                        c.name
                    ));
                }
            }
        }
    }

    let _ = std::fs::remove_dir_all(&root);
    assert!(
        failures.is_empty(),
        "golden-set regressions:\n{}",
        failures.join("\n")
    );
    println!("\ngolden set: all {} cases passed", cases.len());
}
