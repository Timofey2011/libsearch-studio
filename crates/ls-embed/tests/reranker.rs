//! Reranker sanity check: relevant passages must outscore irrelevant ones.
//!
//!     cargo test -p ls-embed --features models reranker -- --nocapture

#![cfg(feature = "models")]

use ls_embed::Reranker;

#[test]
fn reranker_ranks_relevant_above_irrelevant() {
    let dir = std::env::var("LS_RERANKER_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/bge-reranker-v2-m3")
        });
    let mut rr = Reranker::load(&dir).expect("load reranker onnx");

    let query = "how do neural networks learn from data?";
    let passages = [
        "Neural networks learn by minimizing a loss via gradient descent and backpropagation.",
        "The medieval castle had a large stone drawbridge over the moat.",
    ];
    let scores = rr.score(query, &passages).expect("score");

    assert_eq!(scores.len(), 2);
    assert!(
        scores.iter().all(|s| (0.0..=1.0).contains(s)),
        "scores out of range: {scores:?}"
    );
    assert!(
        scores[0] > scores[1],
        "relevant ({}) should outscore irrelevant ({})",
        scores[0],
        scores[1]
    );
    eprintln!(
        "reranker OK — relevant {:.3} > irrelevant {:.3}",
        scores[0], scores[1]
    );
}
