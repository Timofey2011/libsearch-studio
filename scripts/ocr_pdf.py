#!/usr/bin/env python3
"""OCR a scanned PDF into a *searchable* copy, using Apple's Vision framework.

Why a separate script and not a branch inside gpu_embed.py (ROADMAP-3 §18.3):

- Vision costs ~63 s to warm up and then ~0.3 s/page. Paying that inside the
  embed helper would mean re-paying it on every bisect retry, and running
  Vision's GPU buffers alongside a resident fp16 bge-m3 is the exact shape of
  the v0.15.4 unified-memory crash. A short-lived pre-pass process reclaims
  everything on exit.
- The output is a cached *artifact*, not a string. A plain text return would be
  unhighlightable: the reader searches the PDF's own text layer via pdfjs, so a
  scan yields nothing to match and every citation would show the miss overlay.
  Writing an invisible text layer (render_mode=3) makes the cached copy
  highlightable, findable with Cmd-F, selectable, and re-extractable by the
  citation metric — one artifact, four problems.

The text this prints on stdout (JSON) is what the indexer should embed; the
cached PDF is what the reader should display.

macOS only. Requires pyobjc-framework-Vision; PyMuPDF does the rasterizing and
the text-layer writing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import unicodedata
from collections import Counter

SCRIPT_VERSION = 1

# Vision's own scores are optimistically biased; used for reporting only, never
# as a gate on its own (§18.4).
MIN_OBSERVATION_CONFIDENCE = 0.0

# Base-14 fonts silently degrade Cyrillic to dots — verified by round-trip, and
# these scans are mostly Russian, so a Unicode font is mandatory, not cosmetic.
UNICODE_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")


def ocr_cache_key(path) -> str:
    """FNV-1a over length + 64 KiB head/tail.

    CROSS-PIN: ls-extract/src/convert.rs `ocr_cache_key` — the Rust side names
    the same artifact and the two must agree byte for byte. Deliberately not
    the repo's other hashes: `cache_key`/`content_signature` use Rust's
    DefaultHasher, which has no Python equivalent and is not guaranteed stable
    across Rust versions.
    """
    import os

    SAMPLE = 64 * 1024
    h = 0xCBF29CE484222325
    MASK = 0xFFFFFFFFFFFFFFFF

    def eat(bs):
        nonlocal h
        for b in bs:
            h = ((h ^ b) * 0x100000001B3) & MASK

    size = os.path.getsize(path)
    eat(size.to_bytes(8, "little"))
    with open(path, "rb") as f:
        eat(f.read(SAMPLE))
        if size > SAMPLE:
            f.seek(-SAMPLE, os.SEEK_END)
            eat(f.read(SAMPLE))
    return f"{h:016x}"


class OcrUnavailable(RuntimeError):
    """Vision (or its bridge) is not usable — a directed skip, not a crash."""


def _load_vision():
    """Import lazily: a top-level import would cost the --caps probe its
    milliseconds, and an ImportError here must read as a directed skip."""
    try:
        import objc  # noqa: F401
        import Quartz
        import Vision
        from Foundation import NSData
    except Exception as e:  # noqa: BLE001
        raise OcrUnavailable(f"OCR unavailable: pyobjc/Vision not importable ({e})") from e
    return objc, Quartz, Vision, NSData


def unicode_font() -> str:
    for f in UNICODE_FONTS:
        if pathlib.Path(f).exists():
            return f
    raise OcrUnavailable(
        "OCR unavailable: no Unicode TTF found — a base-14 font drops Cyrillic silently"
    )


def ocr_page(png: bytes, langs, deps):
    """Return [(text, confidence, (x0, y0, x1, y1) normalized bottom-left)]."""
    objc, Quartz, Vision, NSData = deps
    # Every PyObjC round-trip autoreleases (NSData, CGImage, handler, each
    # observation). Without a per-page pool a 700-page book retains gigabytes
    # until the process exits — the v0.15.4 failure with a new cause.
    with objc.autorelease_pool():
        data = NSData.dataWithBytes_length_(png, len(png))
        src = Quartz.CGImageSourceCreateWithData(data, None)
        if src is None:
            return []
        cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        if cg is None:
            return []
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(True)
        req.setRecognitionLanguages_(langs)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
        ok, _err = handler.performRequests_error_([req], None)
        if not ok:
            return []
        out = []
        for obs in req.results() or []:
            cands = obs.topCandidates_(1)
            if not cands:
                continue
            text = cands[0].string()
            if not text or not text.strip():
                continue
            conf = float(cands[0].confidence())
            if conf < MIN_OBSERVATION_CONFIDENCE:
                continue
            bb = obs.boundingBox()
            out.append(
                (
                    text,
                    conf,
                    (
                        float(bb.origin.x),
                        float(bb.origin.y),
                        float(bb.origin.x + bb.size.width),
                        float(bb.origin.y + bb.size.height),
                    ),
                )
            )
        return out


def repeated_lines(pages_lines, min_pages=5, ratio=0.30):
    """Running heads, folios and scan watermarks repeat across pages and would
    otherwise land verbatim in quoted citations (they survive the probe's
    wordiness test). Drop lines seen on more than `ratio` of pages."""
    n = len(pages_lines)
    if n < min_pages:
        return set()
    seen = Counter()
    for lines in pages_lines:
        for norm in {normalize_line(t) for t in lines}:
            if norm:
                seen[norm] += 1
    floor = max(min_pages, int(n * ratio))
    return {k for k, c in seen.items() if c >= floor}


def normalize_line(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    # A bare folio differs per page; collapse digits so they group together.
    return re.sub(r"\d+", "#", s)


def is_noise_line(s: str) -> bool:
    t = s.strip()
    if not t:
        return True
    # Bare page numbers / rules.
    return bool(re.fullmatch(r"[\d\s.·—–\-|]+", t))


def dehyphenate(lines):
    """Join line-end hyphenation. FTS cannot match a hyphen-split token, and
    Russian has no fuzzy fallback (fuzzy FTS filters to ASCII), so this is a
    retrieval fix, not cosmetics. Per-line OCR output makes it unambiguous."""
    out = []
    for line in lines:
        if out and re.search(r"[^\W\d_]-$", out[-1], re.UNICODE) and re.match(
            r"[^\W\d_]", line, re.UNICODE
        ):
            out[-1] = out[-1][:-1] + line
        else:
            out.append(line)
    return out


def script_mix_ratio(text: str) -> float:
    """Fraction of tokens mixing Cyrillic and Latin — the homoglyph failure
    (с/c, о/o, р/p). It passes a human eyeball check and still makes a book
    unfindable, so it needs its own number."""
    toks = [t for t in re.split(r"\s+", text) if len(t) >= 3]
    if not toks:
        return 0.0
    mixed = sum(1 for t in toks if CYRILLIC.search(t) and LATIN.search(t))
    return mixed / len(toks)


def ocr_document(path, out_pdf, dpi, langs, max_pages, deps, progress=None):
    import fitz

    fontfile = unicode_font()
    doc = fitz.open(path)
    if doc.needs_pass or doc.is_encrypted:
        doc.close()
        raise OcrUnavailable("DRM-protected — cannot OCR")

    n_pages = doc.page_count
    if max_pages and n_pages > max_pages:
        doc.close()
        # All-or-nothing: a partial book would commit as fully indexed and
        # could never be revisited (§18.3.8).
        raise OcrUnavailable(f"{n_pages} pages over the {max_pages}-page OCR cap")

    per_page_raw = []   # [[line, ...]]
    per_page_boxes = []  # [[(text, conf, bbox)]]
    confs = []
    t0 = time.time()
    for i in range(n_pages):
        page = doc.load_page(i)
        # Clamp by page area so a large-format scan can't blow up the raster.
        rect = page.rect
        area_in2 = max(1e-6, (rect.width / 72.0) * (rect.height / 72.0))
        use_dpi = min(dpi, int((30e6 / area_in2) ** 0.5))
        png = page.get_pixmap(dpi=use_dpi).tobytes("png")
        obs = ocr_page(png, langs, deps)
        per_page_boxes.append(obs)
        per_page_raw.append([t for (t, _c, _b) in obs])
        confs.extend(c for (_t, c, _b) in obs)
        if progress and (i + 1) % 25 == 0:
            progress(i + 1, n_pages)

    drop = repeated_lines(per_page_raw)
    pages_text = []
    for lines in per_page_raw:
        kept = [t for t in lines if normalize_line(t) not in drop and not is_noise_line(t)]
        pages_text.append("\n".join(dehyphenate(kept)))

    # Write the invisible text layer into a copy, positioned from Vision's
    # boxes so the reader's highlight lands on the right part of the page.
    #
    # Measure with a Font object: fitz.get_text_length() takes only built-in
    # font names, so passing our TTF there raises — and swallowing that per
    # line silently produced a "successful" run with an empty text layer.
    measure = fitz.Font(fontfile=fontfile)
    written = 0
    failed = 0
    for i, obs in enumerate(per_page_boxes):
        page = doc.load_page(i)
        ph, pw = page.rect.height, page.rect.width
        for text, _conf, (x0, y0, x1, y1) in obs:
            if normalize_line(text) in drop or is_noise_line(text):
                continue
            # Vision: normalized, origin bottom-left. fitz: points, top-left.
            px0, px1 = x0 * pw, x1 * pw
            baseline_y = (1.0 - y0) * ph
            box_w = max(1.0, px1 - px0)
            box_h = max(1.0, (y1 - y0) * ph)
            size = max(1.0, box_h * 0.8)
            try:
                w = measure.text_length(text, fontsize=size)
                if w > 0:
                    size = max(1.0, min(size * box_w / w, 300.0))
                # fontname must be given alongside fontfile: with fontfile
                # alone PyMuPDF falls back to base-14 and Cyrillic becomes
                # dots (verified by round-trip).
                page.insert_text(
                    (px0, baseline_y),
                    text,
                    fontname="F0",
                    fontfile=fontfile,
                    fontsize=size,
                    render_mode=3,  # invisible: searchable, not painted
                )
                written += 1
            except Exception:  # noqa: BLE001
                # One unrenderable line must not cost the whole book its layer
                # — but a wholesale failure must never read as success.
                failed += 1
                continue
    if written == 0:
        raise OcrUnavailable(
            f"text layer could not be written ({failed} line(s) failed) — refusing "
            "to emit a searchable PDF with no text"
        )

    out_pdf = pathlib.Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc.subset_fonts()  # the Unicode TTF is ~23 MB unsubsetted
    except Exception:  # noqa: BLE001
        pass
    doc.save(str(out_pdf), garbage=3, deflate=True)
    doc.close()

    # Prove the artifact: re-extract from what we just wrote. A searchable PDF
    # whose text does not come back out is the failure this whole design
    # exists to prevent, so it is checked, not assumed.
    verify = fitz.open(str(out_pdf))
    layer_chars = sum(len(verify.load_page(i).get_text("text").strip()) for i in range(verify.page_count))
    verify.close()

    text = "\n".join(pages_text)
    chars = len(text)
    if layer_chars < chars * 0.5:
        raise OcrUnavailable(
            f"text layer verification failed: {layer_chars} chars readable back "
            f"from the searchable PDF vs {chars} recognized"
        )
    return {
        "ok": True,
        "pages": n_pages,
        "chars": chars,
        "layer_chars": layer_chars,
        "lines_written": written,
        "lines_failed": failed,
        "chars_per_page": round(chars / max(1, n_pages), 1),
        "seconds": round(time.time() - t0, 1),
        "dropped_repeated_lines": len(drop),
        "mean_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "low_confidence_frac": round(sum(1 for c in confs if c < 0.5) / len(confs), 3)
        if confs
        else 0.0,
        "script_mix_ratio": round(script_mix_ratio(text), 4),
        "searchable_pdf": str(out_pdf),
        "pages_text": pages_text,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="OCR a scanned PDF into a searchable copy")
    ap.add_argument("pdf")
    ap.add_argument("--out", required=True, help="path for the searchable PDF")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--langs", default="ru-RU,en-US")
    ap.add_argument("--max-pages", type=int, default=1500)
    ap.add_argument("--text-out", help="write extracted text here (default: stdout JSON only)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        deps = _load_vision()
    except OcrUnavailable as e:
        print(json.dumps({"ok": False, "reason": str(e)}))
        return 2

    def progress(done, total):
        if not args.quiet:
            # NB: must not start with "[" — the indexer parses such lines as
            # book events and would double-count them (the v0.15.3 bug).
            print(f"ocr: {done}/{total} pages", file=sys.stderr, flush=True)

    try:
        res = ocr_document(
            args.pdf, args.out, args.dpi, args.langs.split(","), args.max_pages, deps, progress
        )
    except OcrUnavailable as e:
        print(json.dumps({"ok": False, "reason": str(e)}))
        return 2
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "reason": f"{type(e).__name__}: {e}"}))
        return 1

    if args.text_out:
        pathlib.Path(args.text_out).write_text("\n".join(res["pages_text"]), encoding="utf-8")
    summary = {k: v for k, v in res.items() if k != "pages_text"}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
