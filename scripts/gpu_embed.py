#!/usr/bin/env python3
"""Self-contained GPU (Apple MPS) indexer used by the app's "Fast index (GPU)".

Unlike `index_to_parquet.py`, this has NO dependency on the ebook-kb repo — it
only needs `pymupdf`, `sentence-transformers`, and `pyarrow`, so the app can
install it into a local venv and ship a fully self-contained fast-index path.

Extract (PyMuPDF) -> token-window chunk (bge-m3 tokenizer) -> embed on MPS
(sentence-transformers bge-m3, L2-normalized — parity-identical to the Rust
query embedder) -> Parquet matching ls-index::store::chunk_schema.

    python gpu_embed.py --out books.parquet  a.pdf b.pdf ...

Per-book progress is printed to stderr as `[i/total] <title>: <n> chunks` so the
app can parse it live.
"""

import argparse
import hashlib
import pathlib
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

MODEL = "BAAI/bge-m3"
TARGET_TOKENS = 400
OVERLAP_TOKENS = 80
EMBED_BATCH = 64

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


def chunk_pages(pages, tokenizer):
    """Token-window chunks (~400 tokens, 80 overlap), carrying the start page."""
    chunks = []
    for page_no, text in pages:
        text = text.strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        step = TARGET_TOKENS - OVERLAP_TOKENS
        for start in range(0, len(ids), step):
            window = ids[start : start + TARGET_TOKENS]
            if not window:
                break
            chunk_text = tokenizer.decode(window).strip()
            if chunk_text:
                chunks.append((page_no, chunk_text))
            if start + TARGET_TOKENS >= len(ids):
                break
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    model = SentenceTransformer(MODEL, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out, SCHEMA)
    total_chunks = 0
    n = len(args.paths)
    try:
        for i, p in enumerate(args.paths, 1):
            path = pathlib.Path(p)
            title = path.stem
            book_id = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]
            try:
                pieces = chunk_pages(extract_pages(path), tokenizer)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{n}] skip (error) {p}: {e}", file=sys.stderr)
                continue
            if not pieces:
                print(f"[{i}/{n}] skip (no text) {p}", file=sys.stderr)
                continue

            texts = [t for _, t in pieces]
            vectors = model.encode(
                texts, batch_size=EMBED_BATCH, normalize_embeddings=True,
                show_progress_bar=False,
            )
            rows = []
            for j, ((page_no, text), vec) in enumerate(zip(pieces, vectors)):
                rows.append({
                    "id": f"{book_id}:{j}",
                    "book_id": book_id,
                    "title": title,
                    "author": "",
                    "source_path": str(path.resolve()),
                    "format": "pdf",
                    "chapter": "",
                    "page": int(page_no),
                    "loc_start": 0,
                    "loc_end": 0,
                    "text": text,
                    "vector": [float(x) for x in vec],
                })
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            total_chunks += len(rows)
            print(f"[{i}/{n}] {title}: {len(rows)} chunks", file=sys.stderr)
    finally:
        writer.close()
    print(f"wrote {total_chunks} chunks -> {out}")


if __name__ == "__main__":
    main()
