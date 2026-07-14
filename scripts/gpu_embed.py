#!/usr/bin/env python3
"""Self-contained GPU (Apple MPS) indexer used by the app's "Fast index (GPU)".

Unlike `index_to_parquet.py`, this has NO dependency on the ebook-kb repo — it
only needs `pymupdf`, `sentence-transformers`, and `pyarrow`, so the app can
install it into a local venv and ship a fully self-contained fast-index path.

Extract (PyMuPDF) -> concatenate pages -> token-window chunk across page breaks
(bge-m3 tokenizer, line-snapped, real char-offset loc) -> embed on MPS
(sentence-transformers bge-m3, L2-normalized — parity-identical to the Rust
query embedder) -> Parquet matching ls-index::store::chunk_schema.

    python gpu_embed.py --out books.parquet  a.pdf b.pdf ...
    python gpu_embed.py --caps          # stdlib-only capability probe (no torch)

Per-book progress is printed to stderr as `[i/total] <title>: <n> chunks` — for
the app's live DISPLAY only. Per-file OUTCOMES travel via the machine-readable
sidecar `<out>.outcomes.json` (ROADMAP-3 §2.10): one entry per argv file, index-
keyed (paths with spaces/colons/unicode are irrelevant), written atomically
after the parquet. stderr is never parsed for state decisions.

IMPORTANT: the module top is stdlib-only. `--caps` must answer in milliseconds
and must survive a broken torch install; every heavy import is lazy in main().
"""

import argparse
import bisect
import hashlib
import json
import pathlib
import sys
import time

SCRIPT_VERSION = 4

# The extension universe this script handles / deliberately skips. The Rust
# lockstep test parses these literals out of the embedded script and asserts
# they cover ls-core INGEST_EXTS exactly — keep them as plain literals.
HANDLED_EXTS = {"pdf", "md", "markdown", "txt", "text", "rst", "adoc", "org", "tex", "ipynb", "html", "htm", "epub", "fb2", "fb2.zip", "mobi", "azw3", "xps"}
DIRECTED_SKIPS = {}  # ext -> user-facing reason; never reaches fitz.open()

# ext -> format family stamped into the parquet `format` column. Mirrors
# ls-core Format::from_ext for every handled ext (lockstep-tested).
FAMILY = {"pdf": "pdf", "md": "md", "markdown": "md", "ipynb": "md", "txt": "txt", "text": "txt", "rst": "txt", "adoc": "txt", "org": "txt", "tex": "txt", "html": "html", "htm": "html", "epub": "epub", "fb2": "fb2", "fb2.zip": "fb2", "mobi": "mobi", "azw3": "mobi", "xps": "xps"}

# Formats fitz opens natively (fb2.zip is unzipped to a temp .fb2 first).
FITZ_EXTS = {"pdf", "epub", "fb2", "fb2.zip", "mobi", "azw3", "xps"}
# Ebook formats where the document TOC (top two levels) labels chapters.
TOC_CHAPTER_EXTS = {"epub", "fb2", "fb2.zip", "mobi", "azw3", "xps"}

# Python deps probed by --caps (importlib.util.find_spec — no import execution,
# a broken package cannot crash the probe). Installing one changes the caps
# hash, which retries past dependency skips (§2.8).
CORE_DEPS = ["fitz", "torch", "sentence_transformers", "transformers", "pyarrow"]
OPTIONAL_DEPS = []  # grows with M4 (docx, striprtf, …)

TARGET_TOKENS = 400
OVERLAP_TOKENS = 80
EMBED_BATCH = 128
MODEL = "BAAI/bge-m3"


def ext_of(name: str):
    """Longest-match extension rule mirroring ls_core::ext_of ('x.fb2.zip' is
    'fb2.zip', never 'zip'). Matches against every ext this script knows."""
    lower = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
    best = None
    for e in set(HANDLED_EXTS) | set(DIRECTED_SKIPS):
        if len(lower) > len(e) + 1 and lower.endswith(e) and lower[-len(e) - 1] == ".":
            if best is None or len(e) > len(best):
                best = e
    return best


def print_caps() -> None:
    import importlib.util

    caps = {
        "script_version": SCRIPT_VERSION,
        "handled_exts": sorted(HANDLED_EXTS),
        "directed_skips": dict(sorted(DIRECTED_SKIPS.items())),
        "optional_deps_available": {
            d: importlib.util.find_spec(d) is not None
            for d in CORE_DEPS + OPTIONAL_DEPS
        },
        "device_flag_supported": True,
    }
    print(json.dumps(caps, sort_keys=True))


def write_sidecar(out_path: pathlib.Path, outcomes: list) -> None:
    """Atomic (temp+rename) sidecar next to the parquet; one entry per argv
    file, argv-index-keyed. A missing/truncated sidecar means the whole batch
    is treated as failed by the app — never fabricated success."""
    sidecar = out_path.with_name(out_path.name + ".outcomes.json")
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(json.dumps({"v": 1, "outcomes": outcomes}))
    tmp.replace(sidecar)


# ---- text-family extraction (M1) — mirrors crates/ls-extract/src/text.rs ----

SECTION_FLOOR_CHARS = 480  # sections below this merge into their predecessor


def decode_bytes(b: bytes) -> str:
    """utf-8 strict -> cp1251 (RU txt) -> latin-1, mirroring the Rust decode."""
    for enc in ("utf-8", "cp1251"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("latin-1", errors="replace")


def scan_md_sections(text):
    """[(heading|None, body)]: `# `/`## ` open sections; deeper stay in-body."""
    sections = [[None, []]]
    for line in text.splitlines():
        t = line.lstrip()
        n = len(t) - len(t.lstrip("#"))
        if 1 <= n <= 2 and t[n:n + 1] == " " and t[n + 1:].strip():
            sections.append([t[n + 1:].strip(), []])
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(b)) for h, b in sections]


def scan_prefix_sections(text, l1, l2):
    sections = [[None, []]]
    for line in text.splitlines():
        h = None
        if line.startswith(l2):
            h = line[len(l2):]
        elif line.startswith(l1):
            h = line[len(l1):]
        if h is not None and h.strip():
            sections.append([h.strip(), []])
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(b)) for h, b in sections]


def scan_rst_sections(text):
    lines = text.splitlines()
    sections = [[None, []]]
    i = 0
    while i < len(lines):
        title = lines[i].strip()
        under = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if (title and len(under) >= 3 and len(under) + 1 >= len(title)
                and (set(under) == {"="} or set(under) == {"-"})):
            sections.append([title, []])
            i += 2
        else:
            sections[-1][1].append(lines[i])
            i += 1
    return [(h, "\n".join(b)) for h, b in sections]


def scan_tex_sections(text):
    sections = [[None, []]]
    for raw in text.splitlines():
        line = raw
        for i, c in enumerate(raw):
            if c == "%" and (i == 0 or raw[i - 1] != "\\"):
                line = raw[:i]
                break
        t = line.lstrip()
        h = None
        for cmd in ("\\chapter{", "\\section{"):
            if t.startswith(cmd) and "}" in t[len(cmd):]:
                h = t[len(cmd):t.index("}", len(cmd))].strip()
                break
        if h:
            sections.append([h, []])
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(b)) for h, b in sections]


def scan_ipynb_sections(raw):
    nb = json.loads(raw)
    md = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if not src.strip():
            continue
        if cell.get("cell_type") == "markdown":
            md.append(src)
        elif cell.get("cell_type") == "code":
            md.append("```\n" + src + "\n```")
    return scan_md_sections("\n".join(md))


def scan_html_sections(raw):
    from html.parser import HTMLParser

    SKIP = {"script", "style", "nav", "head", "noscript", "svg", "iframe"}
    BLOCK = {"p", "div", "li", "tr", "br", "h3", "h4", "h5", "h6", "section",
             "article", "blockquote", "pre"}

    class T(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.sections = [[None, []]]
            self.skip_depth = 0
            self.heading = None  # collecting h1/h2 text

        def handle_starttag(self, tag, attrs):
            if tag in SKIP:
                self.skip_depth += 1
            elif tag in ("h1", "h2") and self.skip_depth == 0:
                self.heading = []

        def handle_endtag(self, tag):
            if tag in SKIP and self.skip_depth > 0:
                self.skip_depth -= 1
            elif tag in ("h1", "h2") and self.heading is not None:
                h = "".join(self.heading).strip()
                if h:
                    self.sections.append([h, []])
                self.heading = None
            elif tag in BLOCK and self.skip_depth == 0:
                self.sections[-1][1].append("\n")

        def handle_data(self, data):
            if self.skip_depth:
                return
            if self.heading is not None:
                self.heading.append(data)
            else:
                self.sections[-1][1].append(data)

    t = T()
    t.feed(raw)
    return [(h, "".join(b)) for h, b in t.sections]


def apply_section_floor(sections):
    out = []
    for h, body in sections:
        if out and len(body.strip()) < SECTION_FLOOR_CHARS:
            ph, pb = out[-1]
            merged = pb + ("\n" + h + "\n" if h else "\n") + body
            out[-1] = (ph, merged)
        else:
            out.append((h, body))
    return out


def extract_text_sections(path, ext):
    """[(heading|None, body)] for any text-family file, floor applied."""
    raw = decode_bytes(path.read_bytes())
    if ext in ("md", "markdown"):
        sections = scan_md_sections(raw)
    elif ext in ("txt", "text"):
        sections = [(None, raw)]
    elif ext == "rst":
        sections = scan_rst_sections(raw)
    elif ext == "adoc":
        sections = scan_prefix_sections(raw, "= ", "== ")
    elif ext == "org":
        sections = scan_prefix_sections(raw, "* ", "** ")
    elif ext == "tex":
        sections = scan_tex_sections(raw)
    elif ext == "ipynb":
        sections = scan_ipynb_sections(raw)
    elif ext in ("html", "htm"):
        sections = scan_html_sections(raw)
    else:
        raise ValueError(f"not a text-family ext: {ext}")
    return apply_section_floor(sections)


def build_full_text(pages):
    """Concatenate page texts into one stream so chunks can cross page breaks.
    Returns (full_text, page_offsets) where page_offsets[k] = (char_offset, page_no)
    marks where each page begins in full_text (sorted by offset)."""
    parts, page_offsets, offset = [], [], 0
    for page_no, text in pages:
        page_offsets.append((offset, page_no))
        parts.append(text)
        offset += len(text)
        parts.append("\n")  # page separator
        offset += 1
    return "".join(parts), page_offsets


def page_at(page_offsets, pos):
    """1-based page number containing character position `pos`."""
    if not page_offsets:
        return 1
    offs = [o for o, _ in page_offsets]
    i = bisect.bisect_right(offs, pos) - 1
    return page_offsets[max(i, 0)][1]


# Only snap a window edge to a line break within this many characters, so normal
# short PDF lines align cleanly but a giant no-newline paragraph doesn't collapse
# successive windows onto the same line range.
_SNAP_CHARS = 200


def chunk_book(full, page_offsets, tokenizer):
    """Token-window (~400 tok, 80 overlap) over the WHOLE book, so chunks span page
    breaks instead of being cut at every page. Edges snap to line boundaries so a
    chunk doesn't start/end mid-line. Returns [(loc_start, loc_end, start_page,
    text)] with REAL character offsets (not 0)."""
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    if n == 0:
        return []
    step = TARGET_TOKENS - OVERLAP_TOKENS
    chunks, start, prev = [], 0, None
    while start < n:
        end = min(start + TARGET_TOKENS, n)
        c_start = offsets[start][0]
        c_end = offsets[end - 1][1]
        prev_nl = full.rfind("\n", 0, c_start)
        ls = prev_nl + 1 if (prev_nl != -1 and c_start - prev_nl <= _SNAP_CHARS) else c_start
        next_nl = full.find("\n", c_end)
        le = next_nl if (next_nl != -1 and next_nl - c_end <= _SNAP_CHARS) else c_end
        if le <= ls:
            ls, le = c_start, max(c_end, c_start + 1)
        text = full[ls:le].strip()
        if text and (ls, le) != prev:
            chunks.append((ls, le, page_at(page_offsets, ls), text))
            prev = (ls, le)
        if end >= n:
            break
        start += step
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=EMBED_BATCH)
    ap.add_argument("--fp32", action="store_true", help="disable fp16 (slower, exact)")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    # Heavy imports are LAZY: --caps (handled before main) must never pay for
    # torch, and a broken venv should fail here — inside the embed path — with
    # a readable error, not at module import.
    import pyarrow as pa
    import pyarrow.parquet as pq
    import fitz  # PyMuPDF
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    # Tokenizing a whole book at once trips "sequence length is longer than the
    # maximum" — expected here (we window it ourselves), so quiet it to keep the
    # stderr the app shows for progress clean.
    hf_logging.set_verbosity_error()

    schema = pa.schema([
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

    n = len(args.paths)
    print(f"gpu_embed v{SCRIPT_VERSION}: loading bge-m3 on {args.device} "
          f"(first run downloads ~2GB)…", file=sys.stderr, flush=True)
    t_load = time.perf_counter()
    model = SentenceTransformer(MODEL, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Half precision ~2x throughput on MPS/CUDA at a sub-1% cosine cost (vectors
    # are L2-normalized; the cross-encoder reranks anyway). CPU stays fp32.
    fp16 = not args.fp32 and args.device in ("mps", "cuda")
    if fp16:
        try:
            model = model.half()
        except Exception as e:  # noqa: BLE001
            print(f"fp16 unavailable, using fp32: {e}", file=sys.stderr, flush=True)
            fp16 = False
    dev = getattr(model, "device", args.device)
    print(f"model ready on {dev} ({'fp16' if fp16 else 'fp32'}, batch {args.batch}) "
          f"in {time.perf_counter() - t_load:.1f}s — embedding {n} file(s)",
          file=sys.stderr, flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out, schema)
    outcomes = []
    total_chunks = 0
    t_run = time.perf_counter()
    try:
        for i, p in enumerate(args.paths, 1):
            idx = i - 1
            path = pathlib.Path(p)
            title = path.stem
            ext = ext_of(path.name)
            if ext in DIRECTED_SKIPS:
                reason = DIRECTED_SKIPS[ext]
                outcomes.append({"i": idx, "status": "skipped", "reason": reason})
                print(f"[{i}/{n}] skip ({reason}) {p}", file=sys.stderr)
                continue
            if ext not in HANDLED_EXTS:
                outcomes.append({"i": idx, "status": "skipped",
                                 "reason": f"unsupported extension: {path.name}"})
                print(f"[{i}/{n}] skip (unsupported) {p}", file=sys.stderr)
                continue
            book_id = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]
            try:
                if ext in FITZ_EXTS:
                    fitz_path = path
                    tmp_fb2 = None
                    if ext == "fb2.zip":
                        # fitz can't read the zip wrapper: unzip the single
                        # .fb2 entry to a temp file first.
                        import tempfile
                        import zipfile
                        with zipfile.ZipFile(path) as zf:
                            inner = next((n for n in zf.namelist()
                                          if n.lower().endswith(".fb2")), zf.namelist()[0])
                            data = zf.read(inner)
                        tmp_fb2 = tempfile.NamedTemporaryFile(suffix=".fb2", delete=False)
                        tmp_fb2.write(data)
                        tmp_fb2.close()
                        fitz_path = pathlib.Path(tmp_fb2.name)
                    try:
                        with fitz.open(fitz_path) as doc:
                            if doc.needs_pass or doc.is_encrypted:
                                outcomes.append({"i": idx, "status": "skipped",
                                                 "reason": "DRM-protected — cannot index"})
                                print(f"[{i}/{n}] skip (DRM) {p}", file=sys.stderr)
                                continue
                            pages = [(k, page.get_text("text"))
                                     for k, page in enumerate(doc, 1)]
                            # Top-two-level TOC entries label chapters by page.
                            toc_pages = []
                            if ext in TOC_CHAPTER_EXTS:
                                for lvl, toc_title, pg in (doc.get_toc() or []):
                                    if lvl <= 2 and pg > 0 and toc_title.strip():
                                        toc_pages.append((pg, toc_title.strip()))
                                toc_pages.sort(key=lambda x: x[0])
                    finally:
                        if tmp_fb2 is not None:
                            pathlib.Path(tmp_fb2.name).unlink(missing_ok=True)

                    def chapter_at(page_no):
                        best = ""
                        for pg, toc_title in toc_pages:
                            if pg <= page_no:
                                best = toc_title
                            else:
                                break
                        return best

                    full, page_offsets = build_full_text(pages)
                    # (loc_start, loc_end, page, chapter, text)
                    pieces = [(ls, le, pg, chapter_at(pg), t)
                              for (ls, le, pg, t) in chunk_book(full, page_offsets, tokenizer)]
                else:
                    # Text family: window each heading section separately so
                    # chunks never cross sections; chapter feeds the Index tab.
                    pieces = []
                    base = 0
                    for heading, bodytext in extract_text_sections(path, ext):
                        for (ls, le, _pg, t) in chunk_book(bodytext, [], tokenizer):
                            pieces.append((base + ls, base + le, -1, heading or "", t))
                        base += len(bodytext) + 1
            except Exception as e:  # noqa: BLE001
                outcomes.append({"i": idx, "status": "error", "reason": str(e)})
                print(f"[{i}/{n}] skip (error) {p}: {e}", file=sys.stderr)
                continue
            if not pieces or sum(len(t) for (_a, _b, _c, _d, t) in pieces) < 200:
                outcomes.append({"i": idx, "status": "skipped",
                                 "reason": "no extractable text"})
                print(f"[{i}/{n}] skip (no text) {p}", file=sys.stderr)
                continue

            texts = [t for (_ls, _le, _pg, _ch, t) in pieces]
            t_book = time.perf_counter()
            vectors = model.encode(
                texts, batch_size=args.batch, normalize_embeddings=True,
                show_progress_bar=False,
            )
            dt = time.perf_counter() - t_book
            fam = FAMILY.get(ext, "pdf")
            rows = []
            for j, ((loc_start, loc_end, page_no, chapter, text), vec) in enumerate(zip(pieces, vectors)):
                rows.append({
                    "id": f"{book_id}:{j}",
                    "book_id": book_id,
                    "title": title,
                    "author": "",
                    "source_path": str(path.resolve()),
                    "format": fam,
                    "chapter": chapter,
                    "page": int(page_no),
                    "loc_start": int(loc_start),
                    "loc_end": int(loc_end),
                    "text": text,
                    "vector": [float(x) for x in vec],
                })
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
            outcomes.append({"i": idx, "status": "indexed", "chunks": len(rows)})
            total_chunks += len(rows)
            rate = len(rows) / dt if dt > 0 else 0.0
            elapsed = time.perf_counter() - t_run
            print(
                f"[{i}/{n}] {title}: {len(rows)} chunks in {dt:.1f}s "
                f"({rate:.0f} ch/s) · {total_chunks} total · {elapsed:.0f}s elapsed",
                file=sys.stderr,
            )
    finally:
        writer.close()
        # Sidecar last (after the parquet is closed), atomically: a batch whose
        # sidecar is missing or incomplete is treated as failed by the app.
        write_sidecar(out, outcomes)
    print(f"wrote {total_chunks} chunks in {time.perf_counter() - t_run:.0f}s -> {out}")


if __name__ == "__main__":
    # --caps must run in the stdlib-only prologue: answer in milliseconds even
    # in a venv where torch is broken.
    if "--caps" in sys.argv[1:]:
        print_caps()
        sys.exit(0)
    main()
