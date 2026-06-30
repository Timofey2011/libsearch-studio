# LibSearch Studio

A cross-platform (macOS + Linux) desktop app for **chatting with your own ebook library**. Ask a
question; it retrieves the most relevant passages from *your* books and has a language model answer
from them, with clickable citations that open the source PDF at the cited page. Pure-Rust engine +
[Tauri](https://tauri.app) UI.

It uses **RAG (Retrieval-Augmented Generation)**: local bge-m3 embeddings + a bge-reranker-v2-m3
cross-encoder over a LanceDB hybrid (vector + full-text) index, then your choice of LLM (local
Ollama or a cloud provider) writes a grounded, cited answer. The engine is reimplemented natively
in Rust, so the core ships as a single binary with no Python runtime.

> **New here?** Read the [**User Guide**](docs/USER_GUIDE.md) — why RAG, how it works, and how to
> use the app. The same explainer is built into the app under **Settings (gear) → Help**.
> For design see [ARCHITECTURE.md](ARCHITECTURE.md); for build/run/package see [OPS.md](OPS.md).

## Features

- **Grounded chat** — hybrid retrieval (vector + full-text → RRF → cross-encoder rerank) streams a
  cited answer from your chosen LLM. A *Min relevance* threshold means off-topic questions get an
  honest "no matching passages" instead of a hallucination.
- **Clickable citations** — inline `[n]` markers and a Sources list open the source PDF at the
  cited page in a split reader pane; a clear "source not found" prompt if a book was moved.
- **Collections** — index multiple folders "by area of interest." Indexing is **incremental,
  content-deduplicated** (moved/renamed/duplicate files aren't re-embedded), **resumable**, and has
  a **Stop** button with live elapsed time / throughput / ETA and a streaming log.
- **One Index button** — auto-routes to the **GPU** helper when set up (Apple MPS / CUDA, batched +
  checkpointed), else the built-in pure-Rust **CPU** engine. One-click GPU self-setup provisions a
  local helper.
- **Pluggable LLMs** — local **Ollama** (no key, fully private) or **Anthropic / OpenAI / Gemini /
  Fireworks / Ollama Cloud** (API key, stored locally). Quick-switch from the chat bar.
- **Conversations** — multi-turn history with follow-up context, rename, and a per-conversation
  token counter (in/out).
- **Markdown export** — save any answer + its citations as a `.md` file with YAML front-matter.

## Workspace layout

```
crates/
  ls-core/      # domain types (Block, BookDoc, Chunk, …) — no IO, no UI
  ls-index/     # chunking + LanceDB store (vector + FTS, remap/dedup helpers)
  ls-embed/     # bge-m3 + reranker via ONNX Runtime
  ls-extract/   # pdf text extraction
  ls-query/     # embed → hybrid → RRF → rerank → citations (single + multi-collection)
  ls-llm/       # streaming clients: Ollama, Anthropic, OpenAI-compatible; token usage
  ls-artifacts/ # answer + citations → .md
  ls-app/       # service layer: collections, settings, conversations, indexing, dedup
  ls-cli/       # dev CLI: ingest / import / backfill-state / search / ask
src-tauri/      # Tauri bridge (commands + events); GPU helper orchestration
frontend/       # Vite + React + TS UI; PDF reader via the system WebView
scripts/        # gpu_embed.py (self-contained MPS/CUDA indexer), model export
```

Dependency rule: `frontend → src-tauri → ls-app → {engine crates} → ls-core`.

## How indexing stays fast

- **Incremental** — unchanged files are skipped via a `(size, mtime)` fingerprint.
- **Content-deduplicated** — a content signature (size + head/tail hash) recognizes a file that
  moved, was re-timestamped, or is a duplicate, and **re-points** it instead of re-embedding.
- **Resumable (GPU)** — the GPU path embeds in checkpointed batches; a Stop/crash loses only the
  current batch and the dedup resumes from there.
- **Reuse existing embeddings** — an index with the same schema can be imported directly
  (`ls-cli import`) and made dedup-aware (`ls-cli backfill-state`), turning a multi-hour re-embed
  into a minutes-long import.

## Quickstart

```bash
# 1. Provision the ONNX models once (see OPS.md) into ./models — or use the app's
#    Settings → Indexing → "Set up GPU indexing (auto)" after first run.
# 2. Develop / test
cargo test                  # all crate tests (fast, offline)
cargo clippy --all-targets
cargo fmt --all

# 3. Run the app (needs Node + pnpm; see OPS.md for prerequisites)
pnpm --dir frontend install
make dev                                          # or: ./frontend/node_modules/.bin/tauri dev

# 4. Package a desktop bundle (run the CLI from the repo root)
make build                                        # .app + installers for the host OS
make dmg                                          # reliable headless macOS .dmg (see OPS.md)
```

> Note: keep this repo and the app's data (index, models) **outside** any cloud-sync folder
> (Dropbox/iCloud). The Rust `target/`, the LanceDB index, and the downloaded models are multi-GB,
> and cloud-sync blocks the many small `fsync`s LanceDB makes.
