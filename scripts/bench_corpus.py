#!/usr/bin/env python3
"""Generate a fixed benchmark corpus (256 realistic ~280-word passages, EN+RU)
shared by the Python and Rust embedding benchmarks for a fair head-to-head."""

import json
import pathlib
import random

SENTENCES = [
    "Event-driven microservices communicate asynchronously through a durable message broker.",
    "The observer pattern decouples publishers from subscribers and reduces direct coupling.",
    "Gradient descent minimizes the loss by following the negative gradient of the parameters.",
    "A B-tree index accelerates range queries on a sorted column in a relational database.",
    "Transformers use self-attention to weigh the relevance of every token in the sequence.",
    "Backpropagation computes gradients layer by layer using the chain rule of derivatives.",
    "Reciprocal rank fusion merges multiple ranked lists without training any model.",
    "Эта система выполняет полнотекстовый и векторный поиск по личной библиотеке книг.",
    "Паттерны проектирования описывают проверенные решения часто встречающихся задач.",
    "Событийно-управляемая архитектура позволяет слабо связать компоненты системы.",
    "Cross-encoder rerankers score a query against each passage for high precision.",
    "Chunking splits a document into overlapping windows that respect chapter boundaries.",
]


def passage(n_words: int, rng: random.Random) -> str:
    words: list[str] = []
    while len(words) < n_words:
        words += rng.choice(SENTENCES).split()
    return " ".join(words[:n_words])


def main() -> None:
    rng = random.Random(42)
    corpus = [passage(280, rng) for _ in range(256)]
    out = pathlib.Path(__file__).resolve().parent.parent / "bench" / "corpus.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(corpus))
    print(f"wrote {len(corpus)} passages to {out}")


if __name__ == "__main__":
    main()
