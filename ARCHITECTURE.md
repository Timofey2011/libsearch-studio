# Architecture — LibSearch Studio

LibSearch Studio is a local-first desktop app for chatting with an LLM (local Ollama or a cloud
provider) **grounded in your PDF library**. It answers from retrieved passages, renders clickable
citations that open the source PDF at the cited page, manages multiple indices ("collections"),
and exports answers as Markdown. Retrieval (extract, embed, search, rerank) is pure Rust and runs
locally; only the final answer-writing call goes to the chosen LLM. The UI is a
[Tauri](https://tauri.app) shell over a React/TypeScript frontend.

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
| **ls-index** | Chunking + LanceDB store | `chunk_book`, `ChunkParams`, `Store` (vector + native FTS; `add_chunks`, `import_parquet`, `ensure_fts_index`, `delete_book`, `remap_book`, `indexed_book_ids`, `book_paths`) |
| **ls-query** | Retrieve → fuse → rerank → cite | `search()`, `search_multi()` (multi-collection fan-out), `rrf_fuse()`, `format_citation()`, `SearchResult` |
| **ls-llm** | Streaming LLM clients | `Llm` enum {Ollama, Anthropic, OpenAiCompat}; `generate_stream` → `(text, Usage)` with token in/out; inline `<think>` + native reasoning split; `build_prompt_with_history` |
| **ls-artifacts** | Render answer + citations → file | `ArtifactRenderer` trait, `Markdown`, `write_artifact()`, `slugify()` |
| **ls-app** | Service/composition layer (no UI) | `Service` (`index_collection` w/ cancel + dedup), `Db` (SQLite), `Settings` (TOML), `file_fingerprint`, `content_signature`, `discover_books()`, `IndexEvent` |
| **ls-cli** | Dev CLI exercising the engine | `ingest`, `import`, `backfill-state`, `search`, `ask` |
| **src-tauri** (`app`) | Tauri commands + event emitters | see *Bridge* below |
| **frontend** | React/Vite/TS UI | chat, collection manager, PDF reader pane |

Engine handles (`Embedder`, `Reranker`, `OllamaClient`) are **owned by the bridge** and passed
into `ls-app` by reference, so the service layer stays free of runtime/threading concerns and
is fully unit-testable.

## Data flow

### Indexing (`extract → chunk → embed → store`)
`Service::index_collection` walks the collection's source folders (`discover_books`, PDF-only
in v1) and, for each file, applies a **dedup ladder** before doing any work:
1. `(size, mtime)` fingerprint matches → unchanged, skip.
2. fingerprint matches a *different* book id → the file **moved**; `remap_book` re-points its
   chunks (new id + path) instead of re-embedding.
3. the book id is already present in the index → skip + record the fingerprint (covers an index
   built before the manifest, or via `import`).
4. **content signature** (size + head/tail hash, timestamp-independent) matches another book →
   moved/re-timestamped/duplicate; re-point and skip.
Only genuinely new/changed files are extracted, chunked (~400 tokens, 80 overlap), embedded, and
upserted into LanceDB; each embedded book records its fingerprint **and** content signature. The
loop checks a cancellation flag (the **Stop** button) between files and embed batches, keeping
whatever was already written. The FTS index is rebuilt once the run settles. Progress is reported
as `IndexEvent`s (loading / working / embedding / indexed / unchanged / skipped / finished).

### Query (`embed → hybrid → RRF → rerank → cite → synthesize`)
`ls_query::search` embeds the query, runs a **vector** search and a **full-text** search in
parallel, fuses the two ranked lists with Reciprocal Rank Fusion (no model), reranks the fused
candidates with the cross-encoder, and keeps the final top-k as `SearchResult`s with formatted
citations. `search_multi` fans the same out across several collections (per-collection budget,
merge, dedup by id, single rerank). Results below the `min_relevance` threshold are dropped, so an
off-topic question yields an honest "no matching passages" rather than a hallucination. `ls-llm`
builds a grounded prompt (recent turns + passages numbered `[1..n]`) and streams the answer from
the configured provider; the model is instructed to cite with `[n]` markers. Token usage (in/out)
is captured from the stream and recorded per assistant message.

## The Tauri bridge

`src-tauri` is intentionally thin: it owns the expensive engine handles, marshals JSON, and
forwards streams as events. State lives in `AppState { data_dir, models_dir, settings, llm,
engine: Mutex<Option<Engine>> }`; the engine is pre-warmed in the background at startup so the
first query doesn't pay the ONNX load cost.

**Commands:** collections (`list/create/delete_collection`, `set_collection_paths`), indexing
(`index_collection` CPU, `fast_index_collection` GPU, `cancel_indexing`, `setup_gpu_indexing`),
LLM (`list_models`, `warm_model`, `check_llm`, `set_provider`), settings (`get/save_settings`),
conversations (`list/create/rename/delete_conversation`, `list_messages`), and `ask`,
`source_exists`, `save_artifact`.

**Events:** `ask-token` / `ask-reasoning` / `ask-usage` / `ask-done` (synthesis stream),
`index-progress` (`IndexEvent`), `index-log` (raw GPU-helper output), `setup-log`.

The CPU index runs on a dedicated blocking thread with its own Tokio runtime: the SQLite
connection and tokenizer aren't `Send`, so they must never cross an await on the main
multi-threaded runtime. It loads its own embedder so chat stays usable while indexing.

### Indexing engines (CPU vs GPU)
The UI shows a **single Index button** that routes to whichever engine is available:
- **CPU** (`index_collection`) — pure-Rust ONNX, zero setup, cross-platform. The always-present
  fallback; ~1 chunk/s, fine for small/incremental collections.
- **GPU** (`fast_index_collection`) — drives `scripts/gpu_embed.py` (self-contained PyMuPDF +
  sentence-transformers, fp16 + batched on Apple MPS / CUDA). The same dedup ladder runs in the
  bridge so only new files are handed to the helper. It embeds in **checkpointed batches**
  (`run_py_batch`: spawn → import Parquet → commit `book_state`), so a Stop/crash loses only the
  current batch and a re-run resumes via the dedup. fp16 is a modest win on MPS; the real wins are
  dedup + checkpointing. One-click `setup_gpu_indexing` provisions a local venv + models.

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

## Ingest performance & reuse

ONNX Runtime is CPU-only on macOS (CoreML measured slower), so bulk embedding in Rust is slow
(~1 chunk/s fp32); the GPU helper (above) is the path for large libraries. Because the Rust query
embedder is **parity-identical** to Python `sentence-transformers` bge-m3, any index with the same
schema is directly reusable: `ls-cli import <parquet>` loads chunks+vectors verbatim, and
`ls-cli backfill-state <app-dir> [collection]` records each book's fingerprint + content signature
so the dedup recognizes the imported index and a later re-index only embeds genuinely new files.
This turns a multi-hour re-embed into a minutes-long import. See [OPS.md](OPS.md).

## Persistence & data locations

- **SQLite** (`app.db`): collections, conversations, messages (with token usage), and per-book
  manifest `book_state(collection_id, book_id, fingerprint, content_sig)`. **TOML**
  (`settings.toml`): models/artifacts dirs, Ollama host/model, active `llm_provider` + per-provider
  `{api_key, model}` (plaintext, local only), GPU helper paths, retrieval breadth
  (`hybrid_top_k`, `final_top_k`) and `min_relevance`.
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
