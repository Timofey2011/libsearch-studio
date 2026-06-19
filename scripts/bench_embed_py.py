#!/usr/bin/env python3
"""Benchmark Python sentence-transformers bge-m3 embedding throughput (warm)."""

import json
import pathlib
import sys
import time

from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    corpus = json.loads((ROOT / "bench" / "corpus.json").read_text())
    model = SentenceTransformer("BAAI/bge-m3", device=device)

    model.encode(corpus[:16], batch_size=16, normalize_embeddings=True)  # warmup
    t = time.time()
    model.encode(corpus, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    dt = time.time() - t
    print(f"python/{device}: {len(corpus)} chunks in {dt:.2f}s = {len(corpus) / dt:.1f} chunks/s")


if __name__ == "__main__":
    main()
