//! Parity gate: Rust bge-m3 embeddings must match the Python engine's.
//!
//! Run with the real model (downloads ~2GB on first use):
//!     cargo test -p ls-embed --features models -- --nocapture
//!
//! The fixture is produced by `scripts/gen_parity_fixture.py` using the exact
//! `sentence-transformers` path the Python engine uses (bge-m3, normalized).

#![cfg(feature = "models")]

use ls_embed::{cosine, Embedder};

#[derive(serde::Deserialize)]
struct Fixture {
    texts: Vec<String>,
    vectors: Vec<Vec<f32>>,
}

#[test]
fn bge_m3_matches_python_oracle() {
    let raw = include_str!("fixtures/bge_m3_parity.json");
    let fx: Fixture = serde_json::from_str(raw).expect("valid fixture json");

    // Model exported by scripts/export_onnx.py into <repo>/models/bge-m3.
    let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/bge-m3");
    let mut embedder = Embedder::load(&model_dir).expect("load bge-m3 onnx");
    let texts: Vec<&str> = fx.texts.iter().map(String::as_str).collect();
    let got = embedder.embed(&texts).expect("embed");

    assert_eq!(got.len(), fx.vectors.len());
    let mut min_cos = f32::MAX;
    for ((text, g), e) in fx.texts.iter().zip(&got).zip(&fx.vectors) {
        assert_eq!(g.len(), e.len(), "dim mismatch for: {text}");
        let c = cosine(g, e);
        min_cos = min_cos.min(c);
        assert!(c >= 0.999, "cosine {c:.5} below 0.999 for: {text}");
    }
    eprintln!("bge-m3 parity OK — worst cosine vs Python: {min_cos:.5}");
}
