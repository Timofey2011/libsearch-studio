/// Citation matching — the one home for every normalizer the readers use to
/// locate a stored citation inside a rendered document.
///
/// Three readers (epub via foliate, pdf via pdfjs, markdown/text via our own
/// renderer) each have to reconcile text that came out of the *extractor* with
/// text produced by a *renderer*. They drift in exactly the same handful of
/// ways, so they share these functions rather than each carrying a variant.
///
/// CROSS-PIN: `crates/ls-cli/src/citemetric.rs` holds verbatim Rust ports
/// (`norm_text`, `norm_chapter`, `probe_of`, `md_norm`, `find_label`) used by
/// the ROADMAP-3 §17.1 metric. Change the two together or the metric silently
/// stops measuring what ships.

/// Whitespace/hyphenation normalization shared by cite text and rendered text.
///
/// NB: `\w` is ASCII-only in JS, so Cyrillic is deliberately never
/// dehyphenated — the Rust port mirrors that quirk on purpose.
export function normText(s: string): string {
  return s
    .replace(/­/g, "") // soft hyphens
    .replace(/(\w)-\s+(\w)/g, "$1$2") // line-break hyphenation
    .replace(/\s+/g, " ")
    .trim();
}

/// TOC-label normalization. Stored chapter labels and the reader's parsed
/// labels come from different TOC parsers and drift on numbering and
/// whitespace; roman-numeral prefixes ("Part V.", "I. Understanding …") are
/// stripped like arabic ones because the extractor numbers in arabic where
/// publishers' nav documents use roman (§17.2c).
export function normChapter(s: string): string {
  return normText(s)
    .replace(/^(chapter|глава|часть|part)\s+(\d+|[ivxlcdm]+)\.?:?\s+/i, "")
    .replace(/^[\d.]+\s*[.:—-]?\s*/, "")
    .replace(/^[ivxlcdm]+\.\s+/i, "")
    .toLowerCase();
}

/// A short, distinctive needle taken from the cited text.
///
/// Legacy-extracted chunks often START with code fragments or markup junk
/// ("() }" etc.), so pick the first run of eight real words rather than the
/// first eight tokens. When no such run exists the fallback is the first eight
/// tokens including the junk — that path fires on exactly those legacy chunks,
/// so it is kept rather than idealized.
export function probeOf(cite: string): string {
  const words = normText(cite).split(" ");
  const wordy = (w: string) => (w.match(/\p{L}/gu) ?? []).length >= 2;
  let at = words.findIndex(
    (_, i) => i + 8 <= words.length && words.slice(i, i + 8).every(wordy)
  );
  if (at < 0) at = 0;
  return words.slice(at, at + 8).join(" ");
}

/// Markdown-path normalization: whitespace collapse + lowercase, and
/// deliberately weaker than `normText` — that renderer never dehyphenates, so
/// matching against it must not either.
export function mdNorm(s: string): string {
  return s.replace(/\s+/g, " ").trim().toLowerCase();
}

/// Tiered stored-chapter → TOC matching: exact, then containment, then best
/// token overlap. Exact equality alone resolved only ~2 of 12 sampled books
/// (§17.2c) because extractor labels carry broken words ("Pa rt") and
/// different numbering. A wrong pick is safe by construction — the caller
/// verifies with a probe search and falls back to a whole-document search.
export function findTocEntry<T extends { label?: string | null }>(
  toc: T[],
  stored: string
): T | null {
  const want = normChapter(stored);
  if (!want) return null;

  const exact = toc.find((t) => normChapter(t.label ?? "") === want);
  if (exact) return exact;

  if (want.length >= 6) {
    const contained = toc.find((t) => {
      const l = normChapter(t.label ?? "");
      return l.length >= 6 && (l.includes(want) || want.includes(l));
    });
    if (contained) return contained;
  }

  const tokens = (s: string) => new Set(s.split(" ").filter((w) => w.length >= 3));
  const wt = tokens(want);
  if (wt.size < 2) return null;
  let best: T | null = null;
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
