#!/usr/bin/env python3
"""Export bge-m3 (embedder) and bge-reranker-v2-m3 (reranker) to ONNX.

Self-contained torch.onnx export (no optimum dependency). Each model is wrapped
to emit exactly the tensor the Rust engine consumes:
  - bge-m3:          last_hidden_state  (Rust takes the [CLS] row + L2-normalizes)
  - bge-reranker-v2: logits             (Rust applies sigmoid for the score)

Outputs go to models/<name>/{model.onnx, tokenizer.json, ...} (gitignored).

Run with the LibSearch Python venv:
    /path/to/LibSearch/.venv/bin/python scripts/export_onnx.py [--reranker]
"""

import argparse
import pathlib

import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


class LastHidden(torch.nn.Module):
    """Return only last_hidden_state (drop the unused pooler output)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(model_id: str, out_dir: pathlib.Path, wrapper, output_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.save_pretrained(out_dir)  # writes tokenizer.json + configs

    enc = tok(["hello world", "second example"], return_tensors="pt", padding=True)
    dynamic = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        output_name: {0: "batch", 1: "seq"},
    }
    torch.onnx.export(
        wrapper.eval(),
        (enc["input_ids"], enc["attention_mask"]),
        str(out_dir / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=[output_name],
        dynamic_axes=dynamic,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # use the stable TorchScript exporter (no onnxscript dep)
    )
    print(f"exported {model_id} -> {out_dir / 'model.onnx'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker", action="store_true", help="also export the reranker")
    args = ap.parse_args()

    embed = AutoModel.from_pretrained("BAAI/bge-m3")
    export("BAAI/bge-m3", MODELS / "bge-m3", LastHidden(embed), "last_hidden_state")

    if args.reranker:
        rr = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
        export("BAAI/bge-reranker-v2-m3", MODELS / "bge-reranker-v2-m3", Logits(rr), "logits")


if __name__ == "__main__":
    main()
