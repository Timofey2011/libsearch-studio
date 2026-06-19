#!/usr/bin/env python3
"""Dynamically quantize an ONNX model under models/ to int8 for faster CPU inference.

Weights -> int8 (activations stay float). Typically 2-4x faster on CPU and ~4x
smaller, with small quality drift (verify with the parity / reranker tests).

    python scripts/quantize_onnx.py bge-m3              # -> models/bge-m3-int8
    python scripts/quantize_onnx.py bge-reranker-v2-m3  # -> models/bge-reranker-v2-m3-int8
"""

import pathlib
import shutil
import sys

from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = pathlib.Path(__file__).resolve().parent.parent


def quantize(name: str) -> None:
    src = ROOT / "models" / name
    dst = ROOT / "models" / f"{name}-int8"
    dst.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src / "model.onnx"), str(dst / "model.onnx"), weight_type=QuantType.QInt8)
    shutil.copy(src / "tokenizer.json", dst / "tokenizer.json")
    size_mb = (dst / "model.onnx").stat().st_size / 1e6
    print(f"quantized {name} -> {dst} (model.onnx {size_mb:.0f} MB)")


def main() -> None:
    names = sys.argv[1:] or ["bge-m3"]
    for name in names:
        quantize(name)


if __name__ == "__main__":
    main()
