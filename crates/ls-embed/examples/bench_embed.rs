//! Benchmark Rust ONNX embedding throughput (warm), on the same corpus the
//! Python benchmark uses, for a fair head-to-head.
//!
//!   cargo run --release -p ls-embed --example bench_embed -- models/bge-m3 bench/corpus.json

use std::time::Instant;

use ls_embed::Embedder;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args
        .get(1)
        .expect("usage: bench_embed <model_dir> <corpus.json>");
    let corpus_path = args
        .get(2)
        .expect("usage: bench_embed <model_dir> <corpus.json>");

    let corpus: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(corpus_path).unwrap()).unwrap();
    let mut embedder = Embedder::load(model_dir).expect("load model");

    // Warmup (also pays the first-inference graph setup cost).
    let warm: Vec<&str> = corpus.iter().take(16).map(String::as_str).collect();
    embedder.embed(&warm).unwrap();

    let all: Vec<&str> = corpus.iter().map(String::as_str).collect();
    let t = Instant::now();
    for batch in all.chunks(64) {
        embedder.embed(batch).unwrap();
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "rust/cpu ({}): {} chunks in {:.2}s = {:.1} chunks/s",
        model_dir,
        corpus.len(),
        dt,
        corpus.len() as f64 / dt
    );
}
