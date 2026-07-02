#!/usr/bin/env python3
"""Self-contained GPU (Apple MPS) indexer used by the app's "Fast index (GPU)".

Unlike `index_to_parquet.py`, this has NO dependency on the ebook-kb repo — it
only needs `pymupdf`, `sentence-transformers`, and `pyarrow`, so the app can
install it into a local venv and ship a fully self-contained fast-index path.

Extract (PyMuPDF) -> concatenate pages -> token-window chunk across page breaks
(bge-m3 tokenizer, line-snapped, real char-offset loc) -> embed on MPS
(sentence-transformers bge-m3, L2-normalized — parity-identical to the Rust
query embedder) -> Parquet matching ls-index::store::chunk_schema.

    python gpu_embed.py --out books.parquet  a.pdf b.pdf ...

Per-book progress is printed to stderr as `[i/total] <title>: <n> chunks` so the
app can parse it live.

NOTE: chunk boundaries + loc metadata changed here — books indexed by an older
build keep their old chunks until re-indexed, so re-index a collection to pick up
the cross-page chunking.
"""

import argparse
import bisect
import hashlib
import pathlib
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers import logging as hf_logging

# Tokenizing a whole book at once trips "sequence length is longer than the
# maximum" — expected here (we window it ourselves), so quiet it to keep the
# stderr the app parses for progress clean.
hf_logging.set_verbosity_error()

MODEL = "BAAI/bge-m3"
TARGET_TOKENS = 400
OVERLAP_TOKENS = 80
EMBED_BATCH = 128

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("book_id", pa.string()),
    ("title", pa.string()),
    ("author", pa.string()),
    ("source_path", pa.string()),
    ("format", pa.string()),
    ("chapter", pa.string()),
    ("page", pa.int64()),
    ("loc_start", pa.int64()),
    ("loc_end", pa.int64()),
    ("text", pa.string()),
    ("vector", pa.list_(pa.float32(), 1024)),
])


def extract_pages(path: pathlib.Path):
    """Return [(page_number, text)] for a PDF (1-based pages)."""
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, 1):
            pages.append((i, page.get_text("text")))
    return pages


def build_full_text(pages):
    """Concatenate page texts into one stream so chunks can cross page breaks.
    Returns (full_text, page_offsets) where page_offsets[k] = (char_offset, page_no)
    marks where each page begins in full_text (sorted by offset)."""
    parts, page_offsets, offset = [], [], 0
    for page_no, text in pages:
        page_offsets.append((offset, page_no))
        parts.append(text)
        offset += len(text)
        parts.append("\n")  # page separator
        offset += 1
    return "".join(parts), page_offsets


def page_at(page_offsets, pos):
    """1-based page number containing character position `pos`."""
    if not page_offsets:
        return 1
    offs = [o for o, _ in page_offsets]
    i = bisect.bisect_right(offs, pos) - 1
    return page_offsets[max(i, 0)][1]


# Only snap a window edge to a line break within this many characters, so normal
# short PDF lines align cleanly but a giant no-newline paragraph doesn't collapse
# successive windows onto the same line range.
_SNAP_CHARS = 200


def chunk_book(full, page_offsets, tokenizer):
    """Token-window (~400 tok, 80 overlap) over the WHOLE book, so chunks span page
    breaks instead of being cut at every page. Edges snap to line boundaries so a
    chunk doesn't start/end mid-line. Returns [(loc_start, loc_end, start_page,
    text)] with REAL character offsets (not 0)."""
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    if n == 0:
        return []
    step = TARGET_TOKENS - OVERLAP_TOKENS
    chunks, start, prev = [], 0, None
    while start < n:
        end = min(start + TARGET_TOKENS, n)
        c_start = offsets[start][0]
        c_end = offsets[end - 1][1]
        prev_nl = full.rfind("\n", 0, c_start)
        ls = prev_nl + 1 if (prev_nl != -1 and c_start - prev_nl <= _SNAP_CHARS) else c_start
        next_nl = full.find("\n", c_end)
        le = next_nl if (next_nl != -1 and next_nl - c_end <= _SNAP_CHARS) else c_end
        if le <= ls:
            ls, le = c_start, max(c_end, c_start + 1)
        text = full[ls:le].strip()
        if text and (ls, le) != prev:
            chunks.append((ls, le, page_at(page_offsets, ls), text))
            prev = (ls, le)
        if end >= n:
            break
        start += step
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=EMBED_BATCH)
    ap.add_argument("--fp32", action="store_true", help="disable fp16 (slower, exact)")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    n = len(args.paths)
    print(f"loading bge-m3 on {args.device} (first run downloads ~2GB)…",
          file=sys.stderr, flush=True)
    t_load = time.perf_counter()
    model = SentenceTransformer(MODEL, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Half precision ~2x throughput on MPS/CUDA at a sub-1% cosine cost (vectors
    # are L2-normalized; the cross-encoder reranks anyway). CPU stays fp32.
    fp16 = not args.fp32 and args.device in ("mps", "cuda")
    if fp16:
        try:
            model = model.half()
        except Exception as e:  # noqa: BLE001
            print(f"fp16 unavailable, using fp32: {e}", file=sys.stderr, flush=True)
            fp16 = False
    dev = getattr(model, "device", args.device)
    print(f"model ready on {dev} ({'fp16' if fp16 else 'fp32'}, batch {args.batch}) "
          f"in {time.perf_counter() - t_load:.1f}s — embedding {n} file(s)",
          file=sys.stderr, flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out, SCHEMA)
    total_chunks = 0
    t_run = time.perf_counter()
    try:
        for i, p in enumerate(args.paths, 1):
            path = pathlib.Path(p)
            title = path.stem
            book_id = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]
            try:
                full, page_offsets = build_full_text(extract_pages(path))
                pieces = chunk_book(full, page_offsets, tokenizer)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{n}] skip (error) {p}: {e}", file=sys.stderr)
                continue
            if not pieces:
                print(f"[{i}/{n}] skip (no text) {p}", file=sys.stderr)
                continue

            texts = [t for (_ls, _le, _pg, t) in pieces]
            t_book = time.perf_counter()
            vectors = model.encode(
                texts, batch_size=args.batch, normalize_embeddings=True,
                show_progress_bar=False,
            )
            dt = time.perf_counter() - t_book
            rows = []
            for j, ((loc_start, loc_end, page_no, text), vec) in enumerate(zip(pieces, vectors)):
                rows.append({
                    "id": f"{book_id}:{j}",
                    "book_id": book_id,
                    "title": title,
                    "author": "",
                    "source_path": str(path.resolve()),
                    "format": "pdf",
                    "chapter": "",
                    "page": int(page_no),
                    "loc_start": int(loc_start),
                    "loc_end": int(loc_end),
                    "text": text,
                    "vector": [float(x) for x in vec],
                })
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            total_chunks += len(rows)
            rate = len(rows) / dt if dt > 0 else 0.0
            elapsed = time.perf_counter() - t_run
            print(
                f"[{i}/{n}] {title}: {len(rows)} chunks in {dt:.1f}s "
                f"({rate:.0f} ch/s) · {total_chunks} total · {elapsed:.0f}s elapsed",
                file=sys.stderr,
            )
    finally:
        writer.close()
    print(f"wrote {total_chunks} chunks in {time.perf_counter() - t_run:.0f}s -> {out}")


if __name__ == "__main__":
    main()
