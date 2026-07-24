// PdfReader — a pdfjs-dist viewer for the source preview. Replaces WKWebView's
// native PDF widget, which is a cross-origin black box: it exposes no zoom API,
// no page tracking, no text search, and swallows keyboard focus. Rendering the
// pages ourselves makes fit modes, a page counter, find, and Esc all work.
//
// Rendering is virtualized: every page gets a fixed-size placeholder div, and
// an IntersectionObserver rasterizes only pages near the viewport (canvas +
// selectable text layer), destroying them again as they scroll far away, so
// 900-page books don't exhaust memory.
import { useEffect, useMemo, useRef, useState } from "react";
import { getDocument, GlobalWorkerOptions, TextLayer } from "pdfjs-dist";
import type { PDFDocumentLoadingTask, PDFDocumentProxy } from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { probeOf } from "./lib/cite";

GlobalWorkerOptions.workerSrc = workerUrl;

type FitMode = "width" | "page" | "custom";
type PageEntry = {
  scale: number;
  canvas: HTMLCanvasElement;
  textDiv: HTMLDivElement;
  textLayer: TextLayer | null;
  textDivs: HTMLElement[];
  task: { cancel: () => void } | null;
};
type PageText = { joined: string; starts: number[] };
type Match = { page: number; item: number };

const PAGE_GAP = 12;
const GUTTER = 28; // horizontal breathing room inside the scroll pane
const MIN_SCALE = 0.25;
const MAX_SCALE = 6;

export default function PdfReader({
  url,
  page,
  citeText,
  full,
  onFail,
}: {
  url: string;
  page?: number;
  citeText?: string;
  full: boolean;
  onFail: (err: string) => void;
}) {
  const [numPages, setNumPages] = useState(0);
  const [baseSize, setBaseSize] = useState<{ w: number; h: number } | null>(null);
  const [pageSizes, setPageSizes] = useState<Map<number, { w: number; h: number }>>(new Map());
  const [mode, setMode] = useState<FitMode>("width");
  const [customScale, setCustomScale] = useState(1);
  const [boxSize, setBoxSize] = useState<{ w: number; h: number } | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageDraft, setPageDraft] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [matches, setMatches] = useState<Match[] | null>(null);
  const [activeMatch, setActiveMatch] = useState(0);
  const [finding, setFinding] = useState(false);
  const [findAt, setFindAt] = useState(0);
  const [findError, setFindError] = useState<string | null>(null);
  const [citeMiss, setCiteMiss] = useState(false);

  const docRef = useRef<PDFDocumentProxy | null>(null);
  const loadingRef = useRef<PDFDocumentLoadingTask | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pageElsRef = useRef<Map<number, HTMLDivElement>>(new Map());
  const renderedRef = useRef<Map<number, PageEntry>>(new Map());
  const visibleRef = useRef<Set<number>>(new Set());
  const observerRef = useRef<IntersectionObserver | null>(null);
  const textCacheRef = useRef<Map<number, PageText>>(new Map());
  const scaleRef = useRef(1);
  const prevScaleRef = useRef(0);
  const anchorRef = useRef<{ page: number; frac: number }>({ page: 1, frac: 0 });
  const pendingMatchScrollRef = useRef(false);
  const findRunRef = useRef(0);
  const foundForRef = useRef("");
  const activeRef = useRef<Match | null>(null);
  const citeRunRef = useRef(0);
  const findInputRef = useRef<HTMLInputElement>(null);

  // ---- document lifecycle -------------------------------------------------
  useEffect(() => {
    let dead = false;
    const task = getDocument({
      url,
      cMapUrl: "/pdfjs/cmaps/",
      cMapPacked: true,
      standardFontDataUrl: "/pdfjs/standard_fonts/",
      wasmUrl: "/pdfjs/wasm/", // JBIG2/CCITT/JPX image decoding (scanned books)
      // WKWebView's custom-scheme handler (tauri://) does NOT intercept
      // fetches made inside Web Workers, so the pdfjs worker hangs forever
      // fetching fonts/cmaps/wasm. Fetch them on the main thread instead.
      useWorkerFetch: false,
    });
    loadingRef.current = task;
    task.promise
      .then(async (doc) => {
        if (dead) return;
        docRef.current = doc;
        const p1 = await doc.getPage(1);
        if (dead) return;
        const vp = p1.getViewport({ scale: 1 });
        setBaseSize({ w: vp.width, h: vp.height });
        setNumPages(doc.numPages);
      })
      .catch((e) => {
        if (!dead) onFail(String(e?.message ?? e));
      });
    return () => {
      dead = true;
      for (const entry of renderedRef.current.values()) destroyEntry(entry);
      renderedRef.current.clear();
      visibleRef.current.clear();
      textCacheRef.current.clear();
      observerRef.current?.disconnect();
      observerRef.current = null;
      void task.destroy();
      docRef.current = null;
      loadingRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // ---- scale --------------------------------------------------------------
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setBoxSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setBoxSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, [numPages > 0]); // eslint-disable-line react-hooks/exhaustive-deps

  const scale = useMemo(() => {
    if (!baseSize || !boxSize) return 1;
    if (mode === "custom") return customScale;
    const wScale = (boxSize.w - GUTTER) / baseSize.w;
    if (mode === "width") return clamp(wScale);
    return clamp(Math.min(wScale, (boxSize.h - PAGE_GAP * 2) / baseSize.h));
  }, [baseSize, boxSize, mode, customScale]);
  scaleRef.current = scale;

  // Re-rasterize visible pages when the scale changes, keeping the reading
  // position anchored on the same page. (Scaling scrollTop proportionally is
  // not enough: the fixed inter-page gaps don't scale, and the error compounds
  // over hundreds of pages.)
  useEffect(() => {
    const prev = prevScaleRef.current;
    prevScaleRef.current = scale;
    if (!prev || prev === scale) return;
    const cont = scrollRef.current;
    const a = anchorRef.current;
    const el = pageElsRef.current.get(a.page);
    if (cont && el) cont.scrollTop = el.offsetTop + a.frac * el.offsetHeight;
    for (const [p, entry] of renderedRef.current) {
      destroyEntry(entry);
      renderedRef.current.delete(p);
    }
    for (const p of visibleRef.current) void renderPage(p);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scale]);

  // ---- virtualized rendering ----------------------------------------------
  function destroyEntry(entry: PageEntry) {
    entry.task?.cancel();
    entry.textLayer?.cancel();
    entry.canvas.remove();
    entry.textDiv.remove();
  }

  async function renderPage(p: number) {
    const doc = docRef.current;
    const host = pageElsRef.current.get(p);
    if (!doc || !host) return;
    const want = scaleRef.current;
    const existing = renderedRef.current.get(p);
    if (existing) {
      if (existing.scale === want) return;
      destroyEntry(existing);
      renderedRef.current.delete(p);
    }
    const canvas = document.createElement("canvas");
    const textDiv = document.createElement("div");
    textDiv.className = "pdf-text";
    const entry: PageEntry = { scale: want, canvas, textDiv, textLayer: null, textDivs: [], task: null };
    renderedRef.current.set(p, entry);
    // On abort, tear down our own nodes: a superseded call must not leave its
    // canvas behind in the (possibly resized) page div.
    const bail = () => {
      canvas.remove();
      textDiv.remove();
    };
    let pg;
    try {
      pg = await doc.getPage(p);
    } catch (e) {
      bail();
      if (renderedRef.current.get(p) === entry) renderedRef.current.delete(p);
      return;
    }
    if (renderedRef.current.get(p) !== entry || scaleRef.current !== want) return bail();
    const vp = pg.getViewport({ scale: want });
    const base = { w: vp.width / want, h: vp.height / want };
    setPageSizes((m) => {
      const old = m.get(p);
      if (old && Math.abs(old.w - base.w) < 1 && Math.abs(old.h - base.h) < 1) return m;
      const next = new Map(m);
      next.set(p, base);
      return next;
    });
    try {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(vp.width * dpr);
      canvas.height = Math.floor(vp.height * dpr);
      canvas.style.width = `${Math.floor(vp.width)}px`;
      canvas.style.height = `${Math.floor(vp.height)}px`;
      host.append(canvas);
      const task = pg.render({
        canvas,
        viewport: vp,
        transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : undefined,
      });
      entry.task = task;
      await task.promise;
      entry.task = null;
      if (renderedRef.current.get(p) !== entry) return bail();
    } catch (e) {
      // Cancellations are routine (scroll/zoom churn); real errors leave the
      // placeholder blank rather than killing the viewer.
      bail();
      if (renderedRef.current.get(p) === entry) renderedRef.current.delete(p);
      if (!String(e).includes("Cancel")) console.error(`pdf page ${p}:`, e);
      return;
    }
    // Selectable (and find-highlightable) text layer over the canvas — best
    // effort: if it fails, the rendered page must stay visible. Pass the
    // stream, not getTextContent(): pdfjs v6's getTextContent() does
    // `for await` over a ReadableStream, which WKWebView doesn't support;
    // TextLayer consumes the stream with a plain reader pump.
    try {
      textDiv.style.setProperty("--scale-factor", String(vp.scale));
      host.append(textDiv);
      const tl = new TextLayer({ textContentSource: pg.streamTextContent(), container: textDiv, viewport: vp });
      entry.textLayer = tl;
      await tl.render();
      entry.textLayer = null;
      entry.textDivs = tl.textDivs as HTMLElement[];
      applyHighlight();
    } catch (e) {
      textDiv.remove();
      if (!String(e).includes("Cancel")) console.error(`pdf text layer ${p}:`, e);
    }
  }

  function evictPage(p: number) {
    const entry = renderedRef.current.get(p);
    if (!entry) return;
    destroyEntry(entry);
    renderedRef.current.delete(p);
    // Also release pdf.js's per-page caches (operator list, decoded images) —
    // they survive canvas removal and dominate memory on scanned books. The
    // delay lets a just-cancelled render settle (cleanup no-ops mid-render).
    setTimeout(() => {
      if (renderedRef.current.has(p)) return; // page came back meanwhile
      docRef.current
        ?.getPage(p)
        .then((pg) => pg.cleanup())
        .catch(() => {});
    }, 250);
  }

  useEffect(() => {
    if (!numPages) return;
    const cont = scrollRef.current;
    if (!cont) return;
    const obs = new IntersectionObserver(
      (entries) => {
        for (const en of entries) {
          const p = Number((en.target as HTMLElement).dataset.page);
          if (en.isIntersecting) {
            visibleRef.current.add(p);
            void renderPage(p);
          } else {
            visibleRef.current.delete(p);
            evictPage(p);
          }
        }
      },
      { root: cont, rootMargin: "150% 0%" }
    );
    observerRef.current = obs;
    for (const el of pageElsRef.current.values()) obs.observe(el);
    return () => {
      obs.disconnect();
      observerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numPages]);

  // Open at the cited page once the placeholders exist — and navigate again
  // whenever a new citation into the same document changes the page prop.
  useEffect(() => {
    if (!numPages || !page) return;
    requestAnimationFrame(() => scrollToPage(page));
    setCiteMiss(false);
    if (citeText?.trim()) void locateCite(page, citeText);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numPages, page, citeText]);

  /// Highlight the cited passage itself, not just its page.
  ///
  /// The stored page ordinal is right ~87% of the time (§17.2c), so the cited
  /// page is tried first and neighbours only as a fallback — landing on the
  /// stamped page is the common case and must not be second-guessed. A miss
  /// everywhere is surfaced, never silent: the page scroll already happened,
  /// so the overlay says the highlight failed, not the navigation.
  async function locateCite(want: number, cite: string) {
    const probe = probeOf(cite);
    if (!probe) return;
    const run = ++citeRunRef.current;

    // `joined` concatenates text items with no separator (only `\n` at EOL),
    // so a phrase spanning two items has arbitrary whitespace between words —
    // match the probe's words with `\s*` between them rather than literally.
    // Offsets stay indexes into the raw string, which is what `starts` maps.
    const rx = (words: string[]) =>
      new RegExp(words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s*"), "i");
    const words = probe.split(" ");
    const full = rx(words);
    // A 4-word needle is the fallback, and only on the cited page: shorter
    // needles get less distinctive, and the page is already trusted.
    const short = words.length > 4 ? rx(words.slice(0, 4)) : null;

    const tryPage = async (p: number, re: RegExp): Promise<boolean> => {
      if (p < 1 || p > numPages) return false;
      let pt: PageText;
      try {
        pt = await pageText(p);
      } catch {
        return false; // an unreadable page is a miss, not a crash
      }
      if (citeRunRef.current !== run) return true; // superseded; stop quietly
      const m = re.exec(pt.joined);
      if (!m) return false;
      activeRef.current = { page: p, item: itemAt(pt.starts, m.index) };
      pendingMatchScrollRef.current = true;
      if (p !== want) scrollToPage(p);
      applyHighlight();
      return true;
    };

    for (const [p, re] of [
      [want, full],
      [want - 1, full],
      [want + 1, full],
      ...(short ? [[want, short] as const] : []),
    ] as Array<readonly [number, RegExp]>) {
      if (await tryPage(p, re)) return;
      if (citeRunRef.current !== run) return;
    }
    setCiteMiss(true);
  }

  function scrollToPage(p: number) {
    const el = pageElsRef.current.get(Math.min(Math.max(p, 1), numPages || 1));
    const cont = scrollRef.current;
    if (el && cont) cont.scrollTop = el.offsetTop - PAGE_GAP;
  }

  function onScroll() {
    const cont = scrollRef.current;
    if (!cont || !numPages) return;
    const anchor = cont.scrollTop + cont.clientHeight * 0.35;
    let cur = 1;
    for (let p = 1; p <= numPages; p++) {
      const el = pageElsRef.current.get(p);
      if (!el || el.offsetTop > anchor) break;
      cur = p;
    }
    setCurrentPage(cur);
    const el = pageElsRef.current.get(cur);
    if (el && el.offsetHeight > 0) {
      anchorRef.current = {
        page: cur,
        frac: Math.max(-0.5, (cont.scrollTop - el.offsetTop) / el.offsetHeight),
      };
    }
  }

  // ---- find ---------------------------------------------------------------
  async function pageText(p: number): Promise<PageText> {
    const hit = textCacheRef.current.get(p);
    if (hit) return hit;
    const doc = docRef.current;
    if (!doc) return { joined: "", starts: [] };
    // Manual reader pump — NOT getTextContent(), whose `for await` over a
    // ReadableStream throws in WKWebView (no async stream iteration).
    const pg = await doc.getPage(p);
    const reader = pg.streamTextContent().getReader();
    const items: Array<{ str?: string; hasEOL?: boolean }> = [];
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value?.items) items.push(...value.items);
    }
    let joined = "";
    const starts: number[] = [];
    // Index only items with a string: TextLayer skips marked-content items,
    // so the k-th string item is exactly textDivs[k].
    for (const item of items) {
      if (typeof item.str !== "string") continue;
      starts.push(joined.length);
      joined += item.str + (item.hasEOL ? "\n" : "");
    }
    const out = { joined, starts };
    textCacheRef.current.set(p, out);
    return out;
  }

  function itemAt(starts: number[], offset: number): number {
    let lo = 0,
      hi = starts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (starts[mid] <= offset) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  async function runFind() {
    const q = query.trim().toLowerCase();
    if (!q || !numPages) return;
    const run = ++findRunRef.current;
    setFinding(true);
    setMatches(null);
    setFindError(null);
    const found: Match[] = [];
    let firstErr: string | null = null;
    for (let p = 1; p <= numPages; p++) {
      setFindAt(p);
      let pt: PageText;
      try {
        pt = await pageText(p);
      } catch (e) {
        // An unreadable page must not kill the whole search.
        firstErr ??= `p.${p}: ${String((e as { message?: string })?.message ?? e)}`;
        continue;
      }
      if (findRunRef.current !== run) return;
      const hay = pt.joined.toLowerCase();
      const starts = pt.starts;
      let i = hay.indexOf(q);
      while (i !== -1) {
        found.push({ page: p, item: itemAt(starts, i) });
        i = hay.indexOf(q, i + q.length);
      }
    }
    if (findRunRef.current !== run) return;
    foundForRef.current = q;
    setFinding(false);
    setFindError(firstErr);
    setMatches(found);
    if (found.length) {
      // Find forward from the page the user is reading, not from page 1.
      const at = found.findIndex((m) => m.page >= anchorRef.current.page);
      goToMatch(found, at === -1 ? 0 : at);
    } else {
      setActiveMatch(0);
      activeRef.current = null;
      applyHighlight();
    }
  }

  function goToMatch(list: Match[], idx: number) {
    const m = list[(idx + list.length) % list.length];
    setActiveMatch((idx + list.length) % list.length);
    activeRef.current = m;
    pendingMatchScrollRef.current = true;
    scrollToPage(m.page);
    applyHighlight();
  }

  // Re-applies the active-match highlight; scrolls to it only when navigation
  // (goToMatch) is pending — a page passively re-rasterizing must never yank
  // the user's scroll position.
  function applyHighlight() {
    const cont = scrollRef.current;
    if (!cont) return;
    for (const el of cont.querySelectorAll(".pdf-hit")) el.classList.remove("pdf-hit");
    const m = activeRef.current;
    if (!m) return;
    const el = renderedRef.current.get(m.page)?.textDivs[m.item];
    if (el) {
      el.classList.add("pdf-hit");
      if (pendingMatchScrollRef.current) {
        pendingMatchScrollRef.current = false;
        el.scrollIntoView({ block: "center" });
      }
    }
  }

  function stepMatch(dir: 1 | -1) {
    if (!matches?.length) return;
    goToMatch(matches, activeMatch + dir);
  }

  function onFindKey(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      if (matches && foundForRef.current === query.trim().toLowerCase()) stepMatch(e.shiftKey ? -1 : 1);
      else void runFind();
    } else if (e.key === "Escape") {
      // Leave the find field without exiting Reader view. Focus moves to the
      // scroll pane (not body): WKWebView only delivers Escape to a focused
      // element, so this keeps a second Esc able to exit Reader view.
      e.stopPropagation();
      scrollRef.current?.focus();
    }
  }

  // Cmd/Ctrl+F focuses find while in Reader view.
  useEffect(() => {
    if (!full) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "f") {
        e.preventDefault();
        findInputRef.current?.focus();
        findInputRef.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [full]);

  // ---- ui -----------------------------------------------------------------
  function zoomBy(f: number) {
    setCustomScale(clamp(scaleRef.current * f));
    setMode("custom");
  }

  function commitPageDraft() {
    if (pageDraft !== null) {
      const n = parseInt(pageDraft, 10);
      if (!Number.isNaN(n)) scrollToPage(n);
    }
    setPageDraft(null);
  }

  const size = (p: number) => pageSizes.get(p) ?? baseSize ?? { w: 612, h: 792 };

  return (
    <div className="pdf-reader">
      {full && numPages > 0 && (
        <div className="pdf-toolbar">
          <button
            className={"ghost" + (mode === "width" ? " on" : "")}
            onClick={() => setMode("width")}
            title="Fit page width to the window"
          >
            ⇔ Width
          </button>
          <button
            className={"ghost" + (mode === "page" ? " on" : "")}
            onClick={() => setMode("page")}
            title="Fit the entire page in the window"
          >
            ⬒ Page
          </button>
          <span className="pdf-zoom">
            <button className="ghost" onClick={() => zoomBy(1 / 1.2)} title="Zoom out">
              −
            </button>
            <span className="pdf-pct">{Math.round(scale * 100)}%</span>
            <button className="ghost" onClick={() => zoomBy(1.2)} title="Zoom in">
              +
            </button>
          </span>
          <span className="pdf-pages">
            <input
              value={pageDraft ?? String(currentPage)}
              onFocus={() => setPageDraft(String(currentPage))}
              onChange={(e) => setPageDraft(e.target.value)}
              onBlur={() => setPageDraft(null)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  commitPageDraft();
                  scrollRef.current?.focus();
                } else if (e.key === "Escape") {
                  e.stopPropagation();
                  setPageDraft(null);
                  scrollRef.current?.focus();
                }
              }}
              title="Go to page (Enter)"
            />
            <span>/ {numPages}</span>
          </span>
          <span className="pdf-find">
            <input
              ref={findInputRef}
              placeholder="Find in document…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={onFindKey}
              spellCheck={false}
              autoCorrect="off"
              autoCapitalize="off"
            />
            {finding ? (
              <span className="muted">searching… {findAt}/{numPages}</span>
            ) : matches ? (
              <span className="muted">{matches.length ? `${activeMatch + 1}/${matches.length}` : "0 results"}</span>
            ) : null}
            {!finding && findError && (
              <span className="muted" title={findError}>
                ⚠ {findError.slice(0, 80)}
              </span>
            )}
            <button className="ghost" onClick={() => stepMatch(-1)} title="Previous match (Shift+Enter)">
              ↑
            </button>
            <button className="ghost" onClick={() => stepMatch(1)} title="Next match (Enter)">
              ↓
            </button>
          </span>
        </div>
      )}
      {citeMiss && citeText && (
        <div className="cite-miss">
          <div className="cite-miss-head">
            Opened at the cited page, but couldn't highlight the passage — it reads:
            <button className="ghost" onClick={() => setCiteMiss(false)} title="Dismiss">
              ✕
            </button>
          </div>
          <blockquote>{citeText}</blockquote>
        </div>
      )}
      <div className="pdf-scroll" ref={scrollRef} onScroll={onScroll} tabIndex={-1}>
        {numPages === 0 && <div className="pdf-status">Loading document…</div>}
        {Array.from({ length: numPages }, (_, i) => {
          const p = i + 1;
          const s = size(p);
          return (
            <div
              key={p}
              className="pdf-page"
              data-page={p}
              style={{ width: Math.floor(s.w * scale), height: Math.floor(s.h * scale) }}
              ref={(el) => {
                if (el) {
                  pageElsRef.current.set(p, el);
                  observerRef.current?.observe(el);
                } else {
                  pageElsRef.current.delete(p);
                }
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

function clamp(s: number) {
  return Math.min(Math.max(s, MIN_SCALE), MAX_SCALE);
}
