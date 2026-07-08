#!/usr/bin/env python3
"""Fast indexer: embed books on Apple MPS (reusing the validated ebook-kb engine)
and write chunks + vectors to a Parquet file that the Rust app imports into LanceDB.

This is the "hybrid ingest" path: bulk embedding stays on the GPU via Python (the
Rust query embedder is parity-identical, cosine 1.0000, so the index is fully
compatible). Run with the ebook-kb venv.

    /path/to/ebook-kb/.venv/bin/python scripts/index_to_parquet.py \
        --out bench/books.parquet  book1.pdf book2.epub ...

Output schema matches ls-index::store::chunk_schema exactly so Rust can stream the
Parquet batches straight into the table.
"""

import argparse
import hashlib
import os
import pathlib
import sys

# Load models from the local HF cache only — no network at index time (avoids the
# Hugging Face revision-check stall; mirrors the Rust app). Set HF_HUB_OFFLINE=0 to
# allow the one-time model download on a fresh machine.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pyarrow as pa
import pyarrow.parquet as pq

# Reuse the validated Python engine (extraction, chunking, MPS embedding).
# Override with EBOOK_KB_DIR if your ebook-kb checkout lives elsewhere.
EBOOK_KB = os.environ.get(
    "EBOOK_KB_DIR",
    os.path.expanduser("~/Library/CloudStorage/Dropbox/LibSearch"),
)
sys.path.insert(0, EBOOK_KB)

from src.chunk import ChunkParams, chunk_book, default_token_counter  # noqa: E402
from src.config import load_config  # noqa: E402
from src.discover import DiscoveredBook  # noqa: E402
from src.embed import Embedder, embed_chunks  # noqa: E402
from src.extract import extract  # noqa: E402

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


def discovered(path: pathlib.Path) -> DiscoveredBook:
    abspath = str(path.resolve())
    fmt = path.suffix.lower().lstrip(".")
    book_id = hashlib.sha1(abspath.encode()).hexdigest()[:16]
    st = path.stat()
    return DiscoveredBook(
        book_id=book_id, path=abspath, format=fmt, title=path.stem,
        size=st.st_size, mtime=st.st_mtime,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    cfg = load_config()
    embedder = Embedder(cfg.models)
    counter = default_token_counter(cfg.chunking.tokenizer)
    params = ChunkParams(
        cfg.chunking.target_tokens, cfg.chunking.overlap_tokens, cfg.chunking.min_tokens
    )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out, SCHEMA)
    total = 0
    try:
        for i, p in enumerate(args.paths, 1):
            book = discovered(pathlib.Path(p))
            doc = extract(book)
            if not doc.blocks:
                print(f"[{i}/{len(args.paths)}] skip (no text) {p}", file=sys.stderr)
                continue
            chunks = chunk_book(doc, counter, params)
            embed_chunks(embedder, chunks)
            rows = [c.to_row() for c in chunks]
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            total += len(rows)
            print(f"[{i}/{len(args.paths)}] {doc.title}: {len(rows)} chunks", file=sys.stderr)
    finally:
        writer.close()
    print(f"wrote {total} chunks -> {out}")


if __name__ == "__main__":
    main()
