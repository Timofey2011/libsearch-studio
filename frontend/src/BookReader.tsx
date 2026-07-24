// BookReader — the in-app ebook reader (epub / mobi / azw3 / fb2 / fb2.zip)
// built on the vendored foliate-js (see vendor/foliate/README.md).
//
// WKWebView-safe load (ROADMAP-3 invariant #6): the document is fetched on the
// MAIN thread from the asset: protocol (custom-scheme fetches inside workers
// never resolve), wrapped in a File, and handed to foliate — zip inflation
// then runs over in-memory bytes.
//
// Cite-jump (§5.2): normalize the cited passage, resolve its chapter to a TOC
// section and search THAT section first (a whole-book search on open can
// stall multi-MB epubs); fall back to an async whole-book search with the
// reader already visible; a total miss shows the passage in a dismissible
// overlay strip — explicit, never silent.
import { useEffect, useRef, useState } from "react";

// Side-effect import registers the <foliate-view> custom element.
import "./vendor/foliate/view.js";

type TocItem = { label: string; href: string; subitems?: TocItem[] | null };

/// The slice of foliate's View element this component uses.
interface FoliateView extends HTMLElement {
  open(file: File): Promise<void>;
  goTo(target: string): Promise<unknown>;
  goToTextStart(): Promise<unknown>;
  clearSearch(): void;
  search(opts: {
    query: string;
    index?: number;
  }): AsyncGenerator<{ cfi?: string; subitems?: { cfi: string }[] } | string>;
  renderer: {
    setAttribute(k: string, v: string): void;
    removeAttribute(k: string): void;
    prev(): void;
    next(): void;
    setStyles?(css: string): void;
  };
  book: {
    toc?: TocItem[];
    resolveHref(href: string): Promise<{ index: number } | null> | { index: number } | null;
  };
}

/// Whitespace/hyphenation normalization shared by cite text and DOM text —
/// extractor output and foliate-rendered text differ exactly there.
/// CROSS-PIN: ported verbatim in crates/ls-cli/src/citemetric.rs (norm_text,
/// norm_chapter, probe_of) for the §17.1 metric — change both together.
function normText(s: string): string {
  return s
    .replace(/­/g, "") // soft hyphens
    .replace(/(\w)-\s+(\w)/g, "$1$2") // line-break hyphenation
    .replace(/\s+/g, " ")
    .trim();
}

/// TOC-label normalization (amendment A4c): stored chapter strings and
/// foliate's parsed labels drift on numbering and whitespace. Roman-numeral
/// prefixes ("Part V.", "I. Understanding …") are stripped like arabic ones —
/// fitz TOCs number in arabic where publishers' nav docs use roman (§17.2c).
function normChapter(s: string): string {
  return normText(s)
    .replace(/^(chapter|глава|часть|part)\s+(\d+|[ivxlcdm]+)\.?:?\s+/i, "")
    .replace(/^[\d.]+\s*[.:—-]?\s*/, "")
    .replace(/^[ivxlcdm]+\.\s+/i, "")
    .toLowerCase();
}

/// Tiered chapter→TOC matching (§17.2c: exact equality resolved in only ~2 of
/// 12 books — fitz labels carry broken words ("Pa rt") and different numbering
/// than foliate's nav labels). A wrong pick is safe: phase-1 search verifies
/// with the probe and falls back to the whole-book search on a miss.
function findTocEntry(toc: TocItem[], stored: string): TocItem | null {
  const want = normChapter(stored);
  if (!want) return null;
  // Tier 1: exact normalized equality.
  const exact = toc.find((t) => normChapter(t.label ?? "") === want);
  if (exact) return exact;
  // Tier 2: containment either way (guard against tiny fragments).
  if (want.length >= 6) {
    const contained = toc.find((t) => {
      const l = normChapter(t.label ?? "");
      return l.length >= 6 && (l.includes(want) || want.includes(l));
    });
    if (contained) return contained;
  }
  // Tier 3: best token overlap — robust to broken words and numbering.
  const tokens = (s: string) => new Set(s.split(" ").filter((w) => w.length >= 3));
  const wt = tokens(want);
  if (wt.size < 2) return null;
  let best: TocItem | null = null;
  let bestScore = 0;
  for (const t of toc) {
    const lt = tokens(normChapter(t.label ?? ""));
    if (!lt.size) continue;
    let common = 0;
    for (const w of wt) if (lt.has(w)) common++;
    const score = common / (wt.size + lt.size - common);
    if (common >= 2 && score >= 0.6 && score > bestScore) {
      best = t;
      bestScore = score;
    }
  }
  return best;
}

function flattenToc(items: TocItem[] | undefined, out: TocItem[] = []): TocItem[] {
  for (const it of items ?? []) {
    out.push(it);
    flattenToc(it.subitems ?? undefined, out);
  }
  return out;
}

/// Flatten with depth — the §17.2c section-range fix needs to know where a
/// matched entry's SCOPE ends: a "Part N" label spans everything up to the
/// next same-or-shallower entry, not just up to its own first chapter.
function flattenTocDepth(
  items: TocItem[] | undefined,
  depth = 0,
  out: { item: TocItem; depth: number }[] = []
): { item: TocItem; depth: number }[] {
  for (const it of items ?? []) {
    out.push({ item: it, depth });
    flattenTocDepth(it.subitems ?? undefined, depth + 1, out);
  }
  return out;
}

export default function BookReader({
  url,
  citeText,
  chapter,
  full,
  onFail,
}: {
  url: string;
  citeText?: string;
  chapter?: string | null;
  full: boolean;
  onFail: (err: string) => void;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<FoliateView | null>(null);
  const [toc, setToc] = useState<TocItem[]>([]);
  const [flow, setFlow] = useState<"paginated" | "scrolled">("paginated");
  const [fontPct, setFontPct] = useState(100);
  const [status, setStatus] = useState<"loading" | "ready">("loading");
  const [missOverlay, setMissOverlay] = useState(false);

  // ---- open the book --------------------------------------------------------
  useEffect(() => {
    let dead = false;
    const host = hostRef.current;
    if (!host) return;
    (async () => {
      try {
        // Main-thread fetch of the asset: URL (WKWebView-safe), then File.
        const res = await fetch(url);
        if (!res.ok) throw new Error(`read failed (${res.status})`);
        const blob = await res.blob();
        const name = decodeURIComponent(url.split("/").pop() ?? "book");
        const file = new File([blob], name);
        const view = document.createElement("foliate-view") as FoliateView;
        view.style.width = "100%";
        view.style.height = "100%";
        if (dead) return;
        host.append(view);
        await view.open(file);
        if (dead) {
          view.remove();
          return;
        }
        viewRef.current = view;
        setToc(flattenToc(view.book.toc));
        setStatus("ready");
        await citeJump(view, chapter ?? null, citeText);
      } catch (e) {
        if (!dead) onFail(String((e as { message?: string })?.message ?? e));
      }
    })();
    return () => {
      dead = true;
      viewRef.current?.remove();
      viewRef.current = null;
      setStatus("loading");
      setMissOverlay(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // ---- cite-jump (§5.2): chapter-scoped, then async whole-book -------------
  async function citeJump(view: FoliateView, chap: string | null, cite?: string) {
    if (!cite?.trim()) return;
    // A short probe: long enough to be distinctive, short enough for the
    // matcher to hit despite extraction/rendering drift. Legacy-extracted
    // chunks often START with code fragments/markup junk ("() }" etc.), so
    // pick the first run of eight real words, not the first eight tokens.
    const words = normText(cite).split(" ");
    const wordy = (w: string) => (w.match(/\p{L}/gu) ?? []).length >= 2;
    let at = words.findIndex(
      (_, i) => i + 8 <= words.length && words.slice(i, i + 8).every(wordy)
    );
    if (at < 0) at = 0;
    const probe = words.slice(at, at + 8).join(" ");
    if (!probe) return;

    // Resolve the chapter to a section index via tiered TOC matching.
    let sectionIndex: number | null = null;
    let chapterHref: string | null = null;
    // A chapter often spans SEVERAL spine sections while foliate's search is
    // per-section (§17.2c: half the sample resolved the chapter yet phase-1
    // missed) — so resolve a section RANGE: the entry's section up to the
    // next TOC entry that starts a different section, capped.
    let sectionEnd: number | null = null;
    if (chap) {
      const tocd = flattenTocDepth(view.book.toc);
      const hit = findTocEntry(tocd.map((t) => t.item), chap);
      if (hit) {
        chapterHref = hit.href;
        try {
          const resolved = await view.book.resolveHref(hit.href);
          if (resolved && typeof resolved.index === "number") {
            sectionIndex = resolved.index;
            sectionEnd = sectionIndex + 40; // cap the scoped sweep
            const at = tocd.findIndex((t) => t.item === hit);
            const hitDepth = at >= 0 ? tocd[at].depth : 0;
            for (let i = at + 1; i < tocd.length; i++) {
              // The entry's scope ends at the next SAME-OR-SHALLOWER entry
              // (a Part label spans all its chapters). resolveHref may return
              // a plain value OR a promise.
              if (tocd[i].depth > hitDepth) continue;
              const r = await Promise.resolve(view.book.resolveHref(tocd[i].item.href)).catch(
                () => null
              );
              if (r && typeof r.index === "number" && r.index > sectionIndex) {
                sectionEnd = Math.min(sectionEnd, r.index);
                break;
              }
            }
          }
        } catch {
          /* fall through to whole-book */
        }
      }
    }

    const firstHit = async (index?: number): Promise<string | null> => {
      try {
        for await (const r of view.search({ query: probe, ...(index != null ? { index } : {}) })) {
          if (typeof r === "string") break; // 'done'
          if (r.subitems?.length) return r.subitems[0].cfi;
          if (r.cfi) return r.cfi;
          if (viewRef.current !== view) return null; // unmounted
        }
      } catch {
        /* search failure = miss */
      }
      return null;
    };

    // (1) chapter-scoped search across the chapter's section range.
    if (sectionIndex != null) {
      for (let i = sectionIndex; i < (sectionEnd ?? sectionIndex + 1); i++) {
        const cfi = await firstHit(i);
        if (viewRef.current !== view) return;
        if (cfi) {
          await view.goTo(cfi).catch(() => {});
          setTimeout(() => viewRef.current === view && view.clearSearch(), 2500);
          return;
        }
        view.clearSearch();
      }
    }
    // (2) land somewhere sensible immediately, then search the whole book
    // asynchronously — the UI must never block on a multi-MB epub.
    if (chapterHref) await view.goTo(chapterHref).catch(() => {});
    else await view.goToTextStart().catch(() => {});
    const cfi = await firstHit();
    if (viewRef.current !== view) return;
    if (cfi) {
      await view.goTo(cfi).catch(() => {});
      setTimeout(() => viewRef.current === view && view.clearSearch(), 2500);
    } else {
      view.clearSearch();
      setMissOverlay(true); // explicit, never silent
    }
  }

  // ---- toolbar actions -------------------------------------------------------
  function setFlowMode(mode: "paginated" | "scrolled") {
    setFlow(mode);
    viewRef.current?.renderer.setAttribute("flow", mode);
  }
  function bumpFont(delta: number) {
    const pct = Math.min(200, Math.max(60, fontPct + delta));
    setFontPct(pct);
    viewRef.current?.renderer.setStyles?.(`html { font-size: ${pct}% }`);
  }
  function goToToc(href: string) {
    viewRef.current?.goTo(href).catch(() => {});
  }

  return (
    <div className="book-reader">
      {full && status === "ready" && (
        <div className="pdf-toolbar">
          <select
            className="book-toc"
            onChange={(e) => e.target.value && goToToc(e.target.value)}
            value=""
            title="Table of contents"
          >
            <option value="">Contents…</option>
            {toc.map((t, i) => (
              <option key={i} value={t.href}>
                {t.label.trim()}
              </option>
            ))}
          </select>
          <button className="ghost" onClick={() => viewRef.current?.renderer.prev()} title="Previous">
            ◀
          </button>
          <button className="ghost" onClick={() => viewRef.current?.renderer.next()} title="Next">
            ▶
          </button>
          <button
            className={"ghost" + (flow === "paginated" ? " on" : "")}
            onClick={() => setFlowMode(flow === "paginated" ? "scrolled" : "paginated")}
            title="Toggle paginated / scrolled"
          >
            {flow === "paginated" ? "▤ Pages" : "☰ Scroll"}
          </button>
          <span className="pdf-zoom">
            <button className="ghost" onClick={() => bumpFont(-10)} title="Smaller text">
              A−
            </button>
            <span className="pdf-pct">{fontPct}%</span>
            <button className="ghost" onClick={() => bumpFont(10)} title="Larger text">
              A+
            </button>
          </span>
        </div>
      )}
      {missOverlay && citeText && (
        <div className="cite-miss">
          <div className="cite-miss-head">
            Couldn't locate the cited passage in this rendering — it reads:
            <button className="ghost" onClick={() => setMissOverlay(false)} title="Dismiss">
              ✕
            </button>
          </div>
          <blockquote>{citeText}</blockquote>
        </div>
      )}
      <div className="book-host" ref={hostRef}>
        {status === "loading" && <div className="pdf-status">Opening book…</div>}
      </div>
    </div>
  );
}
