#!/usr/bin/env python3
"""Generate the bge-m3 embedding parity fixture from the Python engine's exact path.

Produces crates/ls-embed/tests/fixtures/bge_m3_parity.json with EN+RU sample texts
and their normalized bge-m3 embeddings. The Rust parity test re-embeds the same
texts and asserts cosine >= 0.999, proving the pure-Rust engine is equivalent.

Run with the LibSearch Python venv (has sentence-transformers):
    /path/to/LibSearch/.venv/bin/python scripts/gen_parity_fixture.py
"""

import json
import pathlib

from sentence_transformers import SentenceTransformer

TEXTS = [
    "transformer attention mechanism",
    "how do event-driven microservices communicate",
    "the observer design pattern decouples publishers from subscribers",
    "gradient descent optimizes the loss by following the negative gradient",
    "Это пример абзаца на русском языке для проверки эмбеддингов.",
    "паттерны проектирования и принципы объектно-ориентированного программирования",
    "событийно-управляемая архитектура и микросервисы",
]

OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/ls-embed/tests/fixtures/bge_m3_parity.json"


def main() -> None:
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    vectors = model.encode(TEXTS, normalize_embeddings=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"texts": TEXTS, "vectors": [v.tolist() for v in vectors]})
    )
    print(f"wrote {OUT} ({len(TEXTS)} texts, dim={len(vectors[0])})")


if __name__ == "__main__":
    main()
