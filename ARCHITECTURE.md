# Architecture — LibSearch Studio

LibSearch Studio is a local-first desktop app for chatting with a local LLM **grounded in
your PDF library**. It answers from retrieved passages, renders clickable citations that open
the source PDF at the cited page, manages multiple indices ("collections"), and exports
answers as Markdown. The retrieval engine is pure Rust; the UI is a [Tauri](https://tauri.app)
shell over a React/TypeScript frontend.

## Layered design

A Cargo **workspace** of UI-agnostic engine crates, a thin Tauri bridge, and a web frontend.
The dependency rule is strictly downward — no crate depends on a layer above it, and no
Tauri/UI type leaks below `src-tauri`:

```
frontend  →  src-tauri  →  ls-app  →  { ls-query, ls-index, ls-llm, ls-artifacts,
                                         ls-extract, ls-embed }  →  ls-core
```

| Crate | Responsibility | Key types / entry points |
|-------|----------------|--------------------------|
| **ls-core** | Domain types + traits, no IO | `Block`, `BookDoc`, `Chunk`, `Format`, `TokenCounter` |
| **ls-embed** | ONNX inference (CPU) via `ort` + `tokenizers` | `Embedder` (bge-m3, CLS-pool + L2-norm, 1024-d), `Reranker` (bge-reranker-v2-m3, sigmoid, max_len 512), `BgeTokenCounter`, `cosine()` |
| **ls-extract** | PDF text + structure extraction | `extract(path) -> BookDoc` (pages, TOC chapters, dehyphenation; empty `blocks` ⇒ scanned/skip) |
| **ls-index** | Chunking + LanceDB store | `chunk_book`, `ChunkParams`, `Store` (vector + native FTS, `add_chunks`, `import_parquet`, `ensure_fts_index`, `delete_book`) |
| **ls-query** | Retrieve → fuse → rerank → cite | `search()`, `rrf_fuse()`, `format_citation()`, `SearchResult` |
| **ls-llm** | Ollama client (streaming) | `OllamaClient` (`list_models`, `generate_stream`, `warm`), `build_prompt` |
| **ls-artifacts** | Render answer + citations → file | `ArtifactRenderer` trait, `Markdown`, `write_artifact()`, `slugify()` |
| **ls-app** | Service/composition layer (no UI) | `Service` (`index_collection`, `answer`), `Db` (SQLite), `Settings` (TOML), `discover_books()`, `IndexEvent` |
| **ls-cli** | Dev CLI exercising the engine | `ingest`, `import`, `search`, `ask` |
| **src-tauri** (`app`) | Tauri commands + event emitters | see *Bridge* below |
| **frontend** | React/Vite/TS UI | chat, collection manager, PDF reader pane |

Engine handles (`Embedder`, `Reranker`, `OllamaClient`) are **owned by the bridge** and passed
into `ls-app` by reference, so the service layer stays free of runtime/threading concerns and
is fully unit-testable.

## Data flow

### Indexing (`extract → chunk → embed → store`)
`Service::index_collection` walks the collection's source folders (`discover_books`, PDF-only
in v1), and for each file: extracts a `BookDoc`, checks a `(path, size, mtime)` fingerprint in
SQLite to skip unchanged files, chunks chapter-aware (~400 tokens, 80 overlap), embeds in
batches of 64, and upserts rows into LanceDB. The FTS index is (re)built once at the end.
Progress is reported as `IndexEvent`s.

### Query (`embed → hybrid → RRF → rerank → cite → synthesize`)
`ls_query::search` embeds the query, runs a **vector** search and a **full-text** search in
parallel, fuses the two ranked lists with Reciprocal Rank Fusion (no model), reranks the fused
candidates with the cross-encoder, and keeps the final top-k as `SearchResult`s with formatted
citations. `ls-llm` builds a grounded prompt (passages numbered `[1..n]`) and streams the
answer from Ollama; the model is instructed to cite with `[n]` markers.

## The Tauri bridge

`src-tauri` is intentionally thin: it owns the expensive engine handles, marshals JSON, and
forwards streams as events. State lives in `AppState { data_dir, models_dir, settings, llm,
engine: Mutex<Option<Engine>> }`; the engine is pre-warmed in the background at startup so the
first query doesn't pay the ONNX load cost.

**Commands:** `list_collections`, `create_collection`, `set_collection_paths`,
`index_collection`, `list_models`, `warm_model`, `ask`, `save_artifact`.

**Events:** `ask-token` (streamed answer chunks), `ask-done`, `index-progress` (`IndexEvent`).

Indexing runs on a dedicated blocking thread with its own Tokio runtime: the SQLite connection
and tokenizer aren't `Send`, so they must never cross an await on the main multi-threaded
runtime. It loads its own embedder so chat stays usable while indexing.

## The reader

Clicking a citation opens the source PDF in a split pane. The pane is an `<iframe>` pointing at
`convertFileSrc(path) + "#page=N"` — the system WebView's built-in PDF viewer honors the
`#page` fragment to jump to the cited page. The Tauri **asset protocol** is enabled with scope
`$HOME/**` so local files render. (EPUB/MOBI rendering is out of scope for v1.)

## Models & inference

- **Embedder:** `bge-m3`, fp32 ONNX, CLS pooling + L2 normalize, 1024-dim. Kept fp32 so query
  vectors match the indexed vectors exactly.
- **Reranker:** `bge-reranker-v2-m3`, **int8-quantized** by default (2.3× faster on CPU,
  quality preserved; falls back to fp32 if the int8 dir is absent). `max_length` is pinned at
  512 — leaving it unset crashes ONNX inference on long passages.
- **Parity gate:** the Rust ONNX bge-m3 is verified equal to Python `sentence-transformers`
  bge-m3 (cosine ≈ 1.00000), which is what makes the hybrid ingest path (below) sound.

## Ingest performance & the hybrid path

ONNX Runtime is CPU-only on macOS (CoreML measured slower), so bulk embedding in Rust is slow
(~1 chunk/s fp32). For large libraries, embed on Apple's GPU via Python (~14 chunks/s), write a
Parquet of chunks+vectors, and `ls-cli import` it into LanceDB — sound because the Rust query
embedder is parity-identical. In-app `index_collection` (pure-Rust CPU) is for small or
incremental collections; see [OPS.md](OPS.md).

## Persistence & data locations

- **SQLite** (`app.db`): collections, conversations, messages, per-collection file
  fingerprints. **TOML** (`settings.toml`): models dir, artifacts dir, Ollama host/model,
  retrieval breadth (`hybrid_top_k`, `final_top_k`).
- **LanceDB** (one row per chunk, `vector(1024)` + FTS on `text`): the index. **Must live on
  local disk, never in Dropbox/iCloud** — cloud-sync blocks `fsync` on LanceDB's many small
  writes and hangs ingest.
- Default app-data dir: `~/.local/share/libsearch-studio/` (`app.db`, `settings.toml`,
  `lancedb/`, `collections/<id>/`, `artifacts/`).

## Testing

- **Unit (fast, offline):** chunk boundaries/overlap, RRF fusion, citation formatting, artifact
  rendering + slug/dedup, Ollama request/stream parsing, settings (de)serialization, file
  discovery, fingerprinting.
- **Integration / real-ONNX:** gated behind the `models` feature (`cargo test -p ls-embed
  --features models`) — the embedding **parity** gate and the reranker sanity check — so default
  CI stays fast and offline.

See [OPS.md](OPS.md) for build, model provisioning, indexing, packaging, and troubleshooting.
