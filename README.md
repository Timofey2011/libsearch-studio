# LibSearch Studio

A cross-platform (macOS + Linux) desktop app for **chatting with a local LLM grounded in
your ebook library**, with clickable citations that open the source PDF at the cited page,
multiple indices "by area of interest," persistent conversations, and `.md` artifact
generation. Pure-Rust engine + [Tauri](https://tauri.app) UI.

It is the interactive front end to the validated `ebook-kb` RAG pipeline (bge-m3 embeddings
+ bge-reranker-v2-m3 + LanceDB hybrid retrieval), reimplemented natively in Rust so the app
ships as a single binary with no Python runtime.

> **Status:** early scaffold. See the implementation plan and milestones in the repo issues
> / `ARCHITECTURE.md` (added as the engine lands).

## Workspace layout

```
crates/
  ls-core/      # domain types (Block, BookDoc, Chunk, …) — no IO, no UI
  ls-index/     # chunking (done) + LanceDB store (next)
  ls-embed/     # bge-m3 + reranker via ONNX Runtime           (planned)
  ls-extract/   # pdf/epub text extraction                      (planned)
  ls-query/     # embed → hybrid → RRF → rerank → citations      (planned)
  ls-llm/       # Ollama client (streaming)                     (planned)
  ls-artifacts/ # answer + citations → .md                       (planned)
  ls-app/       # service layer: collections, settings, conversations (planned)
src-tauri/      # Tauri bridge (planned)
frontend/       # Vite + React + TS UI, PDF.js reader (planned)
```

Dependency rule: `frontend → src-tauri → ls-app → {engine crates} → ls-core`.

## Develop

```bash
cargo test          # run all crate tests (fast, offline)
cargo clippy --all-targets
cargo fmt --all
```

> Note: keep this repo **outside** any cloud-sync folder (Dropbox/iCloud). The Rust
> `target/` and downloaded models are multi-GB and will thrash file-provider sync.
