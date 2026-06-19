# Operations Guide — LibSearch Studio

How to build, provision models, run, and troubleshoot the app and engine.

## Prerequisites

| Tool | Why | Install (macOS) |
|------|-----|-----------------|
| **Rust ≥ 1.91** | Transitive AWS SDK crates (via `lance` object-store) require it | `rustup update stable` |
| **protoc** (Protocol Buffers) | `lance` build scripts compile `.proto` via `prost-build` | `brew install protobuf` |
| **Node ≥ 20 + pnpm** | Tauri web frontend | `brew install node && corepack enable` |
| **ONNX Runtime** | Embedding/rerank inference | auto-downloaded by the `ort` crate (`download-binaries`) |
| **Python venv (dev only)** | One-time ONNX model export + parity fixture | the `ebook-kb` repo's `.venv` |
| **Ollama** (optional) | `ask` synthesis | `brew install ollama` |

Linux: install `protobuf-compiler`, `nodejs`/`pnpm`, and the Tauri system deps
(`libwebkit2gtk-4.1-dev`, `libgtk-3-dev`, `librsvg2-dev`, `patchelf`).

## Model provisioning (one-time)

The engine runs `bge-m3` (embedder) and `bge-reranker-v2-m3` (reranker) as ONNX. Export
them once with the `ebook-kb` Python venv (which has `transformers`):

```bash
# from the libsearch-studio repo root
/path/to/ebook-kb/.venv/bin/python scripts/export_onnx.py --reranker
```

This writes `models/bge-m3/` and `models/bge-reranker-v2-m3/` (each ~2.1 GB,
`model.onnx` + external weights + `tokenizer.json`). The directory is gitignored.

Regenerate the embedding **parity fixture** (used by the test gate):

```bash
/path/to/ebook-kb/.venv/bin/python scripts/gen_parity_fixture.py
```

## Build & test

```bash
cargo build                                  # whole workspace (debug)
cargo test                                   # fast, offline unit tests
cargo test -p ls-embed --features models     # parity + reranker gates (needs models/)
cargo clippy --all-targets
cargo fmt --all --check
```

## Indexing (hybrid Python/MPS → Rust)

Bulk embedding runs on Apple's GPU via Python (ONNX Runtime in Rust is CPU-only on
macOS — ~6–14× slower; see the benchmark in git history). Because the Rust query
embedder is parity-identical to Python's (cosine 1.0000), the index is fully
compatible. Workflow:

```bash
# 1. Embed on MPS and write a Parquet of chunks+vectors (reuses the ebook-kb engine).
/path/to/ebook-kb/.venv/bin/python scripts/index_to_parquet.py \
    --out bench/books.parquet  book1.pdf book2.pdf ...

# 2. Import the Parquet into LanceDB from Rust (fast; builds the FTS index).
cargo run -p ls-cli -- import bench/books.parquet
```

Throughput (256 chunks @ ~370 tokens): Python MPS **14.4/s** vs Rust ORT CPU int8 2.2/s
/ fp32 1.0/s. The indexer sets `HF_HUB_OFFLINE` to avoid a network revision-check
stall. Pure-Rust CPU ingest (`ls-cli ingest <pdf>`) still exists for small/incremental
sets but is slow for large libraries.

## Run (engine CLI, before the GUI)

The dev CLI searches an existing LanceDB index (e.g. one built by `ebook-kb`). Because the
Rust ONNX bge-m3 is equivalent to the Python one (parity gate, cosine ≈ 1.0), a Python-built
index is directly usable.

```bash
LS_MODELS_DIR=./models \
LS_DB_PATH="$HOME/.local/share/ebook-kb/lancedb" \
cargo run -p ls-cli -- search "how do event-driven microservices communicate"
```

| Env var | Default | Meaning |
|---------|---------|---------|
| `LS_MODELS_DIR` | `models` | Dir containing `bge-m3/` and `bge-reranker-v2-m3/` |
| `LS_DB_PATH` | `~/.local/share/ebook-kb/lancedb` | LanceDB index directory |

## Data locations

- **Index (LanceDB):** keep on local disk — **never** inside Dropbox/iCloud (cloud-sync
  blocks `fsync` on LanceDB's many small writes). Default `~/.local/share/ebook-kb/lancedb`.
- **Models:** `models/` in the repo (gitignored) for dev; the packaged app will provision
  them into the OS app-data dir on first run.
- **App data (later):** conversations/settings DB under the OS app-data dir (Tauri).

## Troubleshooting

- **`rustc 1.x is not supported by the following packages` (aws-*)** → `rustup update stable`.
- **`failed to run custom build command for lance-encoding`** → install `protoc`
  (`brew install protobuf`); confirm with `protoc --version`.
- **`ort` / ONNX Runtime load errors** → ensure the `download-binaries` feature fetched the
  runtime; on restricted networks pre-provision the ORT dylib and set `ORT_DYLIB_PATH`.
- **Empty / poor search results** → confirm `LS_DB_PATH` points at a populated index
  (`ls-cli` prints the chunk count on startup) and that `models/` matches the index's
  embedding model (bge-m3).
- **Build artifacts huge / disk pressure** → `target/` and `models/` are multi-GB and
  gitignored; keep the repo outside any cloud-synced folder.
