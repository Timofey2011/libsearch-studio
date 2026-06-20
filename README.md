# LibSearch Studio

A cross-platform (macOS + Linux) desktop app for **chatting with a local LLM grounded in
your ebook library**, with clickable citations that open the source PDF at the cited page,
multiple indices "by area of interest," persistent conversations, and `.md` artifact
generation. Pure-Rust engine + [Tauri](https://tauri.app) UI.

It is the interactive front end to the validated `ebook-kb` RAG pipeline (bge-m3 embeddings
+ bge-reranker-v2-m3 + LanceDB hybrid retrieval), reimplemented natively in Rust so the app
ships as a single binary with no Python runtime.

> **Status:** functional end-to-end — chat with grounded answers, clickable citations into the
> PDF reader, in-app folder indexing with live progress, and Markdown export. See
> [ARCHITECTURE.md](ARCHITECTURE.md) for the design and [OPS.md](OPS.md) for build/run/package.

## Features

- **Grounded chat** — retrieve (vector + full-text → RRF → cross-encoder rerank) and stream a
  cited answer from a local Ollama model.
- **Clickable citations** — inline `[n]` markers and a Sources list open the source PDF at the
  cited page in a split reader pane.
- **Collections** — index multiple folders "by area of interest"; incremental re-indexing with
  live progress.
- **Markdown export** — save any answer + its citations as a `.md` file with YAML front-matter.

## Workspace layout

```
crates/
  ls-core/      # domain types (Block, BookDoc, Chunk, …) — no IO, no UI
  ls-index/     # chunking + LanceDB store (vector + FTS)
  ls-embed/     # bge-m3 + reranker via ONNX Runtime
  ls-extract/   # pdf text extraction
  ls-query/     # embed → hybrid → RRF → rerank → citations
  ls-llm/       # Ollama client (streaming)
  ls-artifacts/ # answer + citations → .md
  ls-app/       # service layer: collections, settings, conversations, indexing
  ls-cli/       # dev CLI exercising the engine
src-tauri/      # Tauri bridge (commands + events)
frontend/       # Vite + React + TS UI; PDF reader via the system WebView
```

Dependency rule: `frontend → src-tauri → ls-app → {engine crates} → ls-core`.

## Quickstart

```bash
# 1. Provision the ONNX models once (see OPS.md) into ./models
# 2. Develop / test
cargo test                  # all crate tests (fast, offline)
cargo clippy --all-targets
cargo fmt --all

# 3. Run the app (needs Node + pnpm; see OPS.md for prerequisites)
pnpm --dir frontend install
make dev                                         # or: ./frontend/node_modules/.bin/tauri dev

# 4. Package a desktop bundle (run the CLI from the repo root)
make build                                        # .app + installers for the host OS
make dmg                                          # reliable headless macOS .dmg (see OPS.md)
```

> Note: keep this repo **outside** any cloud-sync folder (Dropbox/iCloud). The Rust
> `target/` and downloaded models are multi-GB and will thrash file-provider sync.
