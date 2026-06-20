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

## In-app indexing (GUI)

The app can index folders directly: open **Manage…**, create a collection (name + one or more
folders via the native folder picker) or add folders to the selected one, then **Index /
Re-index**. A progress bar shows per-file events (`index-progress`); indexing is incremental
(unchanged files are skipped via the `(path, size, mtime)` fingerprint) and rebuilds the FTS
index at the end. Each new collection gets its own LanceDB under
`~/.local/share/libsearch-studio/collections/<id>/`.

In-app indexing embeds on the **CPU** (~1 chunk/s fp32), so it suits small or incremental
collections. For a large library, use the hybrid Python/MPS → Parquet → `ls-cli import` path
above against the collection's `db_path`.

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

## Synthesis providers

The answer is synthesized by either a **local Ollama** model (default) or **Anthropic**
(cloud), chosen in Settings. Retrieval (embeddings + rerank) is always local; only the final
generation call differs. For Anthropic, paste an API key in Settings — it is stored in
plaintext in `settings.toml` under the app data dir and used only for the Messages API call;
prefer Ollama if you don't want a key on disk. The grounded `[n]`-citation prompt is identical
across providers, so citations and the reader work the same way.

## Packaging (desktop bundles)

Build the GUI app and platform installers with the Tauri CLI:

```bash
cargo tauri build                 # release .app + installers for the host platform
cargo tauri build --bundles app   # just the .app (fast; skips installer steps)
```

Outputs land under `target/release/bundle/`.

**macOS — `.app` + `.dmg`.** `cargo tauri build` produces `bundle/macos/LibSearch Studio.app`
and a `.dmg`. The dmg step runs a Finder-prettifying **AppleScript**, which fails in
non-GUI / sandboxed / SSH shells (no Finder automation — you'll see *"failed to run
bundle_dmg.sh"*). The `.app` is unaffected. For a reliable, no-Finder dmg (CI, headless),
build the `.app` then wrap it:

```bash
cargo tauri build --bundles app
scripts/make_dmg.sh               # -> bundle/dmg/LibSearch Studio_<ver>_<arch>.dmg (plain UDZO)
```

Bundles are **unsigned**; on first launch macOS Gatekeeper requires right-click → Open (or
sign + notarize with an Apple Developer ID for distribution).

**Linux — `.AppImage` + `.deb`.** Run `cargo tauri build` on Linux (cross-compiling from macOS
is not supported). Install the Tauri system deps first (`libwebkit2gtk-4.1-dev`, `libgtk-3-dev`,
`librsvg2-dev`, `patchelf`); outputs land in `bundle/appimage/` and `bundle/deb/`.

Bundle targets are configured in `src-tauri/tauri.conf.json` (`bundle.targets`); Tauri ignores
targets that don't apply to the host platform.

## CI

`.github/workflows/ci.yml` runs `fmt`/`clippy`/`cargo test` (fast, offline — no `models/`
needed) on every push, plus a macOS+Linux `cargo tauri build --bundles app` smoke build. The
real-ONNX `models` feature tests and signed installers are not run in CI (models are multi-GB
and gitignored); produce installers on a release runner.

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
- **`failed to run bundle_dmg.sh`** → the Finder AppleScript can't run in a non-GUI shell; use
  `scripts/make_dmg.sh` instead (see *Packaging*). The `.app` itself built fine.
- **"app is damaged / from an unidentified developer"** → bundles are unsigned; right-click →
  Open, or sign + notarize for distribution.
- **In-app indexing finds no files** → only `.pdf` is indexed in v1, and the collection must
  have at least one source folder (add via **Manage…**).
