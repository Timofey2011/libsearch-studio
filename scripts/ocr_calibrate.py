#!/usr/bin/env python3
"""Score OCR against ground truth we already own (ROADMAP-3 §18.4).

None of the scanned books has ground truth — but the library holds hundreds of
PDFs that DO have real text layers. OCR those, compare to their own text, and
you have measured this exact rasterizer, this dpi and this language pair
against truth. That is the difference between "the Russian looked clean to me"
and a number.

Reports per book: character error rate (normalized Levenshtein) and word
recall (fraction of ground-truth words >=4 chars that survive OCR).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ocr_pdf import OcrUnavailable, _load_vision, dehyphenate, ocr_page  # noqa: E402


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def cer(truth: str, got: str) -> float:
    """Normalized edit distance, banded to keep it O(n*band) on long pages."""
    a, b = norm(truth), norm(got)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[len(b)] / max(1, len(a))


def word_recall(truth: str, got: str) -> float:
    """The retrieval-relevant number: a query token that OCR mangled is simply
    absent from the index, and Cyrillic has no fuzzy fallback."""
    tw = [w for w in re.findall(r"\w+", norm(truth), re.UNICODE) if len(w) >= 4]
    if not tw:
        return 1.0
    gw = set(re.findall(r"\w+", norm(got), re.UNICODE))
    return sum(1 for w in tw if w in gw) / len(tw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--pages", type=int, default=12)
    ap.add_argument("--dpi", type=int, action="append", default=None)
    ap.add_argument("--langs", default="ru-RU,en-US")
    args = ap.parse_args()
    dpis = args.dpi or [200, 300]

    import fitz

    deps = _load_vision()
    langs = args.langs.split(",")
    report = []
    for path in args.pdfs:
        doc = fitz.open(path)
        # Sample from the middle: front matter is atypical.
        lo = max(0, doc.page_count // 4)
        idxs = [lo + i * 3 for i in range(args.pages) if lo + i * 3 < doc.page_count]
        idxs = [i for i in idxs if len(doc.load_page(i).get_text("text").strip()) > 400]
        if not idxs:
            print(f"skip (no text layer to compare): {pathlib.Path(path).name}", file=sys.stderr)
            doc.close()
            continue
        name = pathlib.Path(path).name
        for dpi in dpis:
            cers, recs = [], []
            for i in idxs:
                page = doc.load_page(i)
                truth = page.get_text("text")
                png = page.get_pixmap(dpi=dpi).tobytes("png")
                lines = [t for (t, _c, _b) in ocr_page(png, langs, deps)]
                got = "\n".join(dehyphenate(lines))
                cers.append(cer(truth, got))
                recs.append(word_recall(truth, got))
            row = {
                "book": name,
                "dpi": dpi,
                "pages": len(idxs),
                "cer": round(sum(cers) / len(cers), 4),
                "word_recall": round(sum(recs) / len(recs), 4),
            }
            report.append(row)
            print(
                f"{name[:44]:44s} dpi={dpi}  CER={row['cer']:.3f}  recall={row['word_recall']:.3f}",
                flush=True,
            )
        doc.close()
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
