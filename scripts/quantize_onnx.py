#!/usr/bin/env python3
"""Dynamically quantize the bge-m3 ONNX model to int8 for faster CPU inference.

Weights -> int8 (activations stay float). Typically 2-4x faster on CPU and ~4x
smaller, with negligible retrieval-quality loss (verified by the Rust parity test
against the int8 model).
"""

import pathlib
import shutil

from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    src = ROOT / "models" / "bge-m3"
    dst = ROOT / "models" / "bge-m3-int8"
    dst.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(src / "model.onnx"),
        str(dst / "model.onnx"),
        weight_type=QuantType.QInt8,
    )
    shutil.copy(src / "tokenizer.json", dst / "tokenizer.json")
    size_mb = (dst / "model.onnx").stat().st_size / 1e6
    print(f"quantized -> {dst} (model.onnx {size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
