//! Benchmark reranker throughput (warm) on the shared corpus.
//!
//!   cargo run --release -p ls-embed --example bench_rerank -- models/bge-reranker-v2-m3 bench/corpus.json
//!   cargo run --release -p ls-embed --example bench_rerank -- models/bge-reranker-v2-m3-int8 bench/corpus.json

use std::time::Instant;

use ls_embed::Reranker;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: bench_rerank <model_dir> <corpus.json>");
    let corpus_path = args
        .get(2)
        .expect("usage: bench_rerank <model_dir> <corpus.json>");

    let corpus: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(corpus_path).unwrap()).unwrap();
    let passages: Vec<&str> = corpus.iter().map(String::as_str).collect();
    let query = "background jobs and idempotence in microservices";

    let mut reranker = Reranker::load(dir).expect("load reranker");
    let _ = reranker.score(query, &passages[..4]).unwrap(); // warmup

    let n = 24.min(passages.len());
    let t = Instant::now();
    let scores = reranker.score(query, &passages[..n]).unwrap();
    let dt = t.elapsed().as_secs_f64();
    let top = scores.iter().cloned().fold(f32::MIN, f32::max);
    println!(
        "{dir}: reranked {n} passages in {dt:.2}s = {:.1}/s (top score {top:.3})",
        n as f64 / dt
    );
}
