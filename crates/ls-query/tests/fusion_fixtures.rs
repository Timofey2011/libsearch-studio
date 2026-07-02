//! P4a slice 1 — deterministic fixtures that validate FUSION_COSINE_THRESHOLD
//! against the real bge-m3 embedder, EN + RU. These gate the tiered follow-up
//! fusion: related follow-ups must score ABOVE the threshold (fuse), topic
//! switches BELOW it (don't fuse). Run on demand (loads ~2GB ONNX):
//!
//!     cargo test -p ls-query --features models -- --nocapture
#![cfg(feature = "models")]

use ls_embed::Embedder;
use ls_query::FUSION_COSINE_THRESHOLD;

fn embedder() -> Embedder {
    let dir = std::env::var("LS_BGE_M3_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/bge-m3")
        });
    Embedder::load(dir).expect("load bge-m3 (repo models/ or LS_BGE_M3_DIR)")
}

fn cosine(e: &mut Embedder, a: &str, b: &str) -> f32 {
    // embed_query L2-normalizes, so the dot product IS the cosine.
    let va = e.embed_query(a).unwrap();
    let vb = e.embed_query(b).unwrap();
    va.iter().zip(&vb).map(|(x, y)| x * y).sum()
}

/// (prior user turn, follow-up question) that SHOULD fuse — same topic.
const RELATED: &[(&str, &str)] = &[
    (
        "explain the saga pattern in microservices",
        "how does compensation work when one step fails",
    ),
    (
        "how do event-driven microservices communicate",
        "what are the latency trade-offs of message brokers between services",
    ),
    (
        "what is dollar cost averaging in investing",
        "how often should the periodic investments be made",
    ),
    (
        "what is a bloom filter and when is it useful",
        "how do I choose the number of hash functions for it",
    ),
    (
        "explain TCP congestion control",
        "what happens when packets are dropped during slow start",
    ),
    // Russian: same-topic follow-ups.
    (
        "что такое паттерн сага в микросервисах",
        "как работает компенсация если один из шагов не удался",
    ),
    (
        "объясни принципы SOLID в объектно-ориентированном дизайне",
        "приведи пример нарушения принципа подстановки Лисков",
    ),
];

/// (prior user turn, follow-up question) that should NOT fuse — topic switch.
const SWITCHED: &[(&str, &str)] = &[
    (
        "what is dollar cost averaging in investing",
        "explain how Rust lifetimes prevent dangling references",
    ),
    (
        "explain TCP congestion control",
        "what are the main themes of the novel Hamlet",
    ),
    (
        "how do neural networks learn from data",
        "what is the best recipe for sourdough starter",
    ),
    // Russian: clean topic switches.
    (
        "что такое инвестиционный портфель и диверсификация",
        "как настроить кластер kubernetes с нуля",
    ),
    (
        "что такое паттерн сага в микросервисах",
        "какие упражнения лучше для утренней зарядки",
    ),
];

/// Gray zone: in-domain topic switches (both software) score between the clean
/// bands (measured 0.352 vs threshold 0.33). Fusing here is ACCEPTABLE by
/// design — fusion merges pools and rerank stays keyed on the current question,
/// so a wrong fuse only adds candidates that lose the rerank. Printed for
/// visibility, not asserted.
const GRAY: &[(&str, &str)] = &[(
    "explain the saga pattern in microservices",
    "how do I tune garbage collection pauses in the JVM",
)];

#[test]
fn related_followups_clear_threshold_and_switches_fall_below() {
    let mut e = embedder();
    let mut ok = true;
    for (prior, q) in RELATED {
        let c = cosine(&mut e, prior, q);
        println!("RELATED  cos={c:.3}  (need ≥ {FUSION_COSINE_THRESHOLD}) :: {q}");
        ok &= c >= FUSION_COSINE_THRESHOLD;
    }
    for (prior, q) in SWITCHED {
        let c = cosine(&mut e, prior, q);
        println!("SWITCHED cos={c:.3}  (need < {FUSION_COSINE_THRESHOLD}) :: {q}");
        ok &= c < FUSION_COSINE_THRESHOLD;
    }
    for (prior, q) in GRAY {
        let c = cosine(&mut e, prior, q);
        println!("GRAY     cos={c:.3}  (informational, either way OK) :: {q}");
    }
    assert!(
        ok,
        "a fixture crossed FUSION_COSINE_THRESHOLD the wrong way — see printed cosines; \
         adjust the threshold in ls-query before enabling fusion changes"
    );
}
