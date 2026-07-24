# Changelog

All notable changes to LibSearch Studio, newest first. Each version is a git tag (`vN`) and a GitHub release with the `.dmg` attached.

## v0.15.5 — 2026-07-24

Bump manifests 0.15.4 → 0.15.5.

- **Fixed: the re-chunk nudge could never resolve.** Books that are
  skip-listed (no extractable text, or quarantined for crashing the GPU
  helper) were counted as "legacy books" by the nudge and armed banner,
  promising a re-chunk that the planner (correctly) never performs. The
  count now includes only books a re-chunk can actually reach; permanently
  skipped legacy books get their own informational note instead (their
  existing chunks stay searchable).
- **Fixed: the "Armed — …" confirmation message stuck around forever**,
  hiding the live banner/nudge state even after the run that consumed it.
- **One last chance before quarantine:** a book that crashes the helper
  alone is retried once with a tiny encode micro-batch (8) — a fraction of
  the peak memory — which can squeeze very large books through; only if
  that also crashes is it quarantined.

## v0.15.4 — 2026-07-23

Bump manifests 0.15.3 → 0.15.4.

- **Fixed the root cause of the batch crashes: the GPU helper leaked memory
  across books.** The v0.15.3 bisect revealed the crashes were cumulative,
  not per-book — every book in a "failing" batch embedded fine alone, but a
  few very large PDFs in one helper process exhausted unified memory
  (torch's MPS cache never returns memory to the OS) and killed the helper
  at the Metal level. The helper (v8) now releases the MPS cache after
  every book, so full 40-book batches survive the heaviest cohorts. The
  v0.15.3 bisect stays as the safety net.

## v0.15.3 — 2026-07-23

Bump manifests 0.15.2 → 0.15.3.

- **Fixed: books that crash the GPU helper at the native (Metal) level no
  longer poison batches forever.** v0.15.2 contained Python-level embedding
  failures, but the worst books kill the helper process outright — voiding
  their whole batch, and again on every later run. A failed batch is now
  bisected: both halves retry, splits repeat until the culprit stands
  alone, and a book that crashes the helper even in isolation gets a real,
  visible skip (quarantine) instead of being retried forever. Quarantined
  books re-attempt automatically when the Python env/GPU capabilities
  change, or after Maintenance clears their skip.
- **Fixed: the progress counter could exceed the total (e.g. "663/562")
  with a bogus "ETA 0:00".** Failed batches were double-counted; retried
  books are no longer counted twice and the counter is clamped so it can
  never pass the total.

## v0.15.2 — 2026-07-23

Bump manifests 0.15.1 → 0.15.2.

- **Fixed: one problematic book could void its whole 40-book batch.** An
  embedding failure (e.g. the GPU running out of memory on a very large
  PDF) crashed the helper mid-batch, discarding all 40 books' work — and
  during a re-chunk this repeated every batch, silently re-embedding the
  same books over and over. The helper now contains failures per book:
  it frees GPU memory, retries the book once with a smaller micro-batch,
  and if it still fails records an honest per-book error while the other
  39 books commit normally.

## v0.15.1 — 2026-07-22

Bump manifests 0.15.0 → 0.15.1.

- **Fixed: index runs froze at a random book (e.g. "29/40", 0 ch/s) and
  never recovered.** Root cause: trimming the internal log buffer could cut
  a multi-byte character in half (Russian titles!), crashing the run's task
  silently — the UI kept showing "Indexing…" forever. The trim now cuts on
  character boundaries (regression-tested against the exact crash).
- Batch commits and the search-index build also moved to their own
  dedicated runtime with a 15-minute timeout — any future stall surfaces
  as a clean, resumable error instead of a frozen run, with step-by-step
  breadcrumbs in the Log.
- The Mac no longer idle-sleeps while an index run is active (caffeinate
  held for the duration). Closing the lid or quitting the app still stops
  the run — it resumes on the next Index click.

## v0.15.0 — 2026-07-17

Bump manifests 0.14.0 → 0.15.0. Re-chunk that actually re-chunks.

- **Fixed: "Re-chunk on next Index" was a silent no-op** — the dedup guard
  that protects imported libraries from pointless re-embedding also
  neutralized the re-chunk opt-in (runs finished in minutes having done
  nothing). It is now a persistent per-library flag: the next Index run
  genuinely re-embeds every book still on the old chunking scheme.
- **Resumable**: Stop (or a crash) loses at most the current 40-book batch;
  the next Index continues from where it stopped. The flag survives
  restarts and clears itself only when nothing is left to re-chunk.
- **Fixed: re-embedding could duplicate chunks** — the GPU engine appended
  a re-embedded book's new chunks without removing the old ones (latent,
  would have doubled every book during a re-chunk). Both engines now
  replace chunks, under every historical id a file has ever had.
- The re-chunk banner now shows persistently while armed.

## v0.14.0 — 2026-07-15

Bump manifests 0.13.0 → 0.14.0. Library hygiene.

- **Maintenance panel** (Settings → Maintenance): scan a library for debris
  and fix it in one click — entries whose files were deleted, wrong format
  labels (imported books were stamped "pdf" regardless of type), the same
  document indexed twice as pdf + docx, and files indexed under duplicate
  ids. Fixes touch only the index — never your files — and the fix always
  re-checks the library at apply time, so a stale report can't delete the
  wrong thing.
- **Office documents join duplicate-detection**: a document sitting as
  pdf + docx (or any office pair) with the same name now indexes once —
  the pdf wins (page-numbered citations, best reader); among office-only
  pairs the cleaner format wins and .pages never shadows a sibling.
- An index run and a maintenance fix can no longer run at the same time.
- Unreachable source folders (unmounted drive, offline share) are reported
  and their books are never misclassified as missing.

## v0.13.0 — 2026-07-14

Bump manifests 0.12.1 → 0.13.0. Hybrid indexing: no format dead-ends.

- **Standard-engine sweep**: on GPU-configured machines, converter-only
  formats (.doc, .pages, .webarchive, .djvu) are now indexed by the standard
  engine automatically at the end of the same Index run — previously the GPU
  helper skipped them with "handled by standard indexing" and nothing ever
  ran it. One progress stream, one combined summary.
- Moved converter-format files are re-pointed on every run (metadata repair
  no longer depends on models loading or the run completing).
- Converter tools (textutil/antiword/soffice/djvutxt) now run with a hard
  timeout, so Stop can't get stuck behind a hung conversion.
- Skip reasons that carry a remedy ("install antiword or LibreOffice",
  "brew install djvulibre") are now written to the persistent run Log.
- "Set up search models (auto)" now installs the Word/RTF/OpenDocument
  helpers (python-docx, striprtf, odfpy) so fresh GPU setups index office
  documents without a manual pip install.

## v0.12.1 — 2026-07-14

Bump manifests 0.12.0 → 0.12.1.

- Fixed ghost buttons showing through behind "Indexing…" and "■ Stop" while
  an index run is active (WKWebView kept stale pixels of the idle row after
  the layout shifted; the row now remounts on the state change and the Index
  button keeps a constant width).

## v0.12.0 — 2026-07-14

Bump manifests 0.11.0 → 0.12.0. Best-effort formats — ROADMAP-3 complete.

- **New indexable formats**: legacy Word (.doc, converted via textutil /
  antiword / LibreOffice — whichever is installed), Apple Pages (.pages,
  through the document's own embedded PDF preview), Safari saved pages
  (.webarchive, pure Rust), and DjVu (.djvu via djvulibre's djvutxt).
- **Honest skips, automatic retries**: when a converter is missing the file
  is skipped once with the exact fix ("install antiword or LibreOffice",
  "brew install djvulibre") and retried automatically after you install it —
  no re-index button hunting.
- **Display**: .pages opens in the PDF reader (its embedded preview);
  .doc and .webarchive render styled in-app with the usual sanitized
  offline conversion; everything falls back to extracted text.
- Converted artifacts live in an app-owned cache; the original file keeps
  its identity — moved-file detection, dedup, and "Open in default app"
  still point at your document.
- One book as pdf + djvu side by side no longer indexes twice (the better
  format wins).

## v0.11.0 — 2026-07-14

Bump manifests 0.10.0 → 0.11.0. Office documents.

- **New indexable formats**: Word (.docx, headings become chapters), RTF
  (Russian windows-1251 handled), and OpenDocument (.odt) — pure Rust on the
  standard engine (full Linux parity); the Fast/GPU engine uses optional
  Python packages and tells you exactly what to `pip install` if missing,
  retrying automatically once installed.
- **Styled in-app display**: .docx renders with formatting (headings, lists,
  tables) via a sanitized offline converter; saved .html pages now render
  styled instead of text-only; .rtf/.odt convert through macOS textutil with
  a provenance note. Everything falls back to the extracted-text reader when
  a converter isn't available — citation jump works in all of them.

## v0.10.0 — 2026-07-14

Bump manifests 0.9.0 → 0.10.0. Ebooks: searchable AND readable.

- **New indexable formats**: EPUB, FB2 (including .fb2.zip, windows-1251
  handled, author extracted), MOBI/AZW3 (non-DRM; DRM detected and reported),
  and XPS (Fast/GPU indexing). Chapters come from the book's own TOC and feed
  Titles/Index.
- **In-app book reader**: clicking an epub/fb2/mobi citation now opens the
  book in a real reader (foliate-js) — paginated or scrolled, TOC navigation,
  font size — and **jumps to the cited passage** (chapter-scoped search with a
  whole-book fallback; a miss shows the passage in a dismissible strip, never
  silently). Reader view makes it full-window.
- One book, several files: when the same book sits next to itself as
  epub + pdf + mobi, only the preferred format is indexed (epub first) —
  no more duplicate sources in answers.
- **↻ Re-index this book** (Titles view): re-embeds a single book with the
  current extractor — the way to give an old book proper chapters.
- "Show extracted text" on any format without a pretty renderer — every
  indexed file gets an in-app view with citation jump.

## v0.9.0 — 2026-07-10

Bump manifests 0.8.0 → 0.9.0. "Index your notes": the first release of the
ROADMAP-3 format expansion.

- **New indexable formats**: Markdown (.md), plain text (.txt — including
  Russian windows-1251), reStructuredText, AsciiDoc, Org, LaTeX, Jupyter
  notebooks (.ipynb), and saved HTML pages. Both indexing engines (CPU and
  Fast/GPU) handle all of them, with headings becoming chapters (feeding
  Titles/Index) — capped at the top two levels, with small sections merged,
  so heading-dense notes don't flood the index with fragments.
- The indexing summary now reports per-format counts ("indexed 12 md, 3 txt
  · skipped 480 pdf"), so the first re-scope run over existing folders is
  legible.
- Text preview: reads up to 8 MB (was 4), decodes non-UTF-8 files, renders
  big files progressively without freezing, and shows headings and code
  blocks properly. Files beyond 8 MB show a truncation banner with an
  open-externally button.
- Under the hood (shipped dark in this cycle, active now): a hardened
  dedup/identity layer that guarantees existing libraries never re-embed
  when new formats appear — verified against the real 815-book library
  (231 legacy Markdown books all skip; only genuinely new files index).
  Skips are recorded per-pipeline and retried automatically when the file,
  installed tools, or the GPU device change.

## v0.8.0 — 2026-07-09

Bump manifests 0.7.3 → 0.8.0.
- The PDF preview is now a real reader (custom PDF.js viewer replacing the
  system webview widget). Reader view opens fit-to-window-width with a
  toolbar: **⇔ Width** / **⬒ Page** fit, **− / +** zoom, a live
  **current page / total** counter (type a number to jump), and **find in
  document** (Cmd+F) with match highlighting, next/previous stepping, and
  find-forward from the page you're reading.
- Rendering is virtualized — only pages near the viewport are rasterized —
  so 900-page books stay light on memory; scanned PDFs (JBIG2/CCITT/JPX)
  decode too.
- Text is selectable on rendered pages. Esc now exits Reader view even
  after clicking into the document (the old native viewer swallowed it).
- If PDF.js can't open a file, the old native viewer is used automatically.

## v0.7.3 — 2026-07-09

Bump manifests 0.7.2 → 0.7.3.
- The source preview gains a **⛶ Reader view** button: the document expands
  to fill the whole window for distraction-free reading; "⤡ Exit reader
  view" (or Esc) returns to the split view. PDFs keep their page and scroll
  position across the toggle, and Markdown sources get a centered reading
  column at a larger font. Works for all preview kinds (PDF, Markdown, and
  the cited-passage panel for epub/mobi).

## v0.7.2 — 2026-07-08

Bump manifests 0.7.1 → 0.7.2.
- Typing (or deleting) in the Index/Titles search no longer freezes the app.
  Broad queries used to render every one of tens of thousands of matches on
  each keystroke; filtering now runs over precomputed search strings, input
  is decoupled from list rendering (useDeferredValue), and results are capped
  at 800 rows with a "first 800 of N — type to narrow" count so nothing is
  silently hidden.
- Index/Titles search fields no longer trigger the macOS autocorrect popup.

## v0.7.1 — 2026-07-08

Bump manifests 0.7.0 → 0.7.1.
- Rebuilding the Library map on a slow/reasoning model could take several
  minutes with no feedback and no way out — it looked like a hang. The build
  now shows live progress ("model is reasoning… ~N chars"), has a Cancel
  button, and no longer blocks the Themes tab: Titles and Index stay fully
  usable while it runs.

## v0.7.0 — 2026-07-08

Bump manifests 0.6.6 → 0.7.0.
- Themes gains two new views: **Titles** — an A–Z browser over every book in
  the selected collections (filter, Ask, Open) — and **Index** — a
  back-of-book-style index spanning the whole library, built from chapter
  headings (A–Z + А–Я letter rail, search, click → open the book at the
  chapter's first page). No LLM involved; instant.
- The library selector moved to the top of the sidebar and now scopes
  everything, including the conversation list (chats belong to the library
  they were asked in).

## v0.6.6 — 2026-07-08

Bump manifests 0.6.5 → 0.6.6.
- Reader: clicking an .epub or .md citation no longer shows a blank pane.
  Markdown sources render in-app (with best-effort scroll to the cited
  passage); ebook formats the reader can't display show the cited passage and
  an "Open in your ebook app" button. PDFs unchanged.

## v0.6.5 — 2026-07-08

Bump manifests 0.6.4 → 0.6.5.
- Fix: every Russian/Cyrillic query panicked a background worker thread since
  v0.5.5 (lance byte-slices the fuzzy-search prefix anchor) — the failure was
  swallowed silently, dropping the typo-tolerant keyword signal for RU. Fuzzy
  keyword expansion is now ASCII-only (fst 0.4.7's Levenshtein automaton can't
  match non-ASCII at distance ≥ 1 upstream anyway); Russian typo tolerance is
  carried by query spell-repair, proven end-to-end.
- New golden-set retrieval harness (cargo test -p ls-query --features models
  --test golden_set): 8-book EN+RU fixture corpus, 16 cases — direct keywords,
  paraphrase, cross-lingual both ways, typos, follow-up fusion, noise probe.
  It caught the bug above on its first run.

## v0.6.4 — 2026-07-08

Bump manifests 0.6.3 → 0.6.4.
- Answer-side fixture harness (cargo test -p ls-llm --features fixtures): real
  temperature-0 generations proving memory can't contaminate citations — no
  [n] cited that exists only in history/notes; a contradicting note never
  flips a sourced fact. New GenOpts (temperature/seed) on all providers.
- Memory tab nudges a re-read when notes are >90 days old (they shape every
  answer silently).
- scripts/index_to_parquet.py: personal absolute path replaced with an env
  override (repo-public audit: no secrets anywhere in history; this was the
  only personal-info exposure).

## v0.6.3 — 2026-07-08

Bump manifests 0.6.2 → 0.6.3.
- Themes: the Library-map header now labels its model as provenance ("built
  <date> with <model>") and, when that differs from the currently selected
  model, says so and points at Rebuild — it previously read like a live
  setting, so a cached gemma-built map looked wrong after switching providers.

## v0.6.2 — 2026-07-02

Bump manifests 0.6.1 → 0.6.2. Completes Roadmap-2's mid-term tier:
- Stop button: the send button becomes ■ while an answer generates — stopping
  keeps whatever already streamed (saved marked "[answer stopped]" with its
  sources). Works at any phase, including during retrieval.
- Re-index nudge: collections whose books were indexed with an older chunking
  scheme show a passive, dismissible note with an explicit "Re-chunk on next
  Index" opt-in (a full re-embed can take hours — never automatic).
- Safer schema migrations: ALTER errors other than "column already exists" now
  surface instead of being silently swallowed.

## v0.6.1 — 2026-07-02

Bump manifests 0.6.0 → 0.6.1. Cross-conversation memory lands ("Ledger, not
Brain", docs/ROADMAP-2.md):
- Settings → Memory: a user-authored notebook — the textarea IS the app's
  entire memory. Injected into prompts as explicitly non-citable context
  (never the Sources block), capped at ~600 tokens with a live counter,
  exportable to Markdown, with an off-switch. The app never writes it
  autonomously; "+ Notes" on any answer appends it explicitly.
- Per-answer "ⓘ context" chip: exactly what went into the prompt (notes,
  recent turns, digest lines, dropped turns), computed by the prompt builder
  itself and emitted as a new ask-context event.

## v0.6.0 — 2026-07-02

Bump manifests 0.5.9 → 0.6.0. First batch of the conversation-memory arc
(docs/ROADMAP-2.md, "Ledger, not Brain"):
- Earlier-topics digest: turns older than the 6-turn window are compressed into
  capped one-liners instead of vanishing (shed before full turns on budget).
- Stale [n] citation markers are stripped from assistant-role history so old
  answers' numbering can't collide with the current Sources block.
- Tiered follow-up fusion: mid-length and long follow-ups now widen retrieval
  when semantically continuous with the prior turn (bge-m3 cosine, threshold
  0.33 calibrated by new EN/RU fixtures — which falsified the initial 0.5
  guess); pronoun-led and short follow-ups fuse as before. Zero LLM calls.
- New models-gated fixture harness (cargo test -p ls-query --features models)
  gates future retrieval-quality changes.

## v0.5.9 — 2026-07-02

Bump manifests 0.5.8 → 0.5.9. Clean-install polish (found by walking the real
first-run flow end to end):
- Actionable "models aren't set up" error on the CPU index path (was a raw
  tokenizer/file error).
- No app restart after Set up: models load live and the next index/ask uses them.
- Clearer setup affordance ("Set up search models (auto)" — required to index &
  search, GPU is a bonus).

## v0.5.8 — 2026-07-02

Bump manifests 0.5.7 → 0.5.8. Final roadmap item (later tier):
- GPU indexer now chunks ACROSS page breaks (paragraph/line-snapped) instead of
  per-page, and writes real char-offset loc + the chunk's start page. Fixes
  mid-sentence splits at page boundaries and meaningless loc metadata. The
  embedded helper is refreshed on each fast-index run, so the fix ships on update;
  existing books keep old chunks until re-indexed.

## v0.5.7 — 2026-07-02

Bump manifests 0.5.6 → 0.5.7. Mid-term roadmap batch — grounding honesty +
data-safety:
- History-aware follow-ups: an anaphoric question ("why?") also retrieves on the
  previous turn, so it isn't searched in isolation (zero extra LLM calls).
- Honest caveat on whole-book / aggregative questions ("summarize this book"),
  which RAG answers from a handful of passages.
- Provenance badge: answers drawn only from the fuzzy fallback tier are flagged
  as lower-confidence.
- Cloud-sync guard: warn when the index/data dir sits on Dropbox/iCloud/etc.,
  which corrupts LanceDB + SQLite (never warns on source folders).
- Prompt token budget: bound the assembled prompt (script-aware, EN+RU) and drop
  oldest history first — grounding is never trimmed.

## v0.5.6 — 2026-07-02

Bump manifests 0.5.5 → 0.5.6. First roadmap batch — trust + first-run hardening:
- First-run onboarding card replacing the dead empty-state (add+index a library,
  pick a model), gated so it never shows once a library exists.
- Persist the partial answer on mid-stream error/timeout instead of losing it.
- Connect timeout + per-token idle deadline on all LLM calls (no more infinite
  hang on a stalled provider); no retry after first byte.
- Actionable "models aren't set up" error on the ask path (points to setup).
- Raise Anthropic max_tokens 2048 → 4096 (was truncating long answers).
- Reveal-data-folder button in Settings → General (backup + no-cloud-sync note).
- docs/ROADMAP.md: the vetted functional + UX roadmap this batch is drawn from.

## v0.5.5 — 2026-07-01

Bump manifests 0.5.4 → 0.5.5. Ships typo-tolerant retrieval: a misspelled
query (e.g. "investmenet for engineers") that previously returned "no matching
passages" is now spell-repaired against the retrieved passages and recovers the
same top result as the correctly-spelled query.

## v0.5.4 — 2026-07-01

Bump manifests 0.5.3 → 0.5.4. Ships the Settings key-validation flow: a
"Check key" button probes the provider's /models and populates the Model field
with a dropdown of chat models (image/embedding/audio/rerank models filtered
out), so a non-chat id like flux-1-schnell can't be selected. A manual "or
model id" field remains for chat models a provider's /models omits.

## v0.5.3 — 2026-07-01

Bump manifests 0.5.2 → 0.5.3. Ships the clearer LLM chat error: a chat call
to a non-chat model (e.g. a Fireworks image model) now surfaces
"That model doesn't support chat …" instead of a raw HTTP 401.

## v0.5.2 — 2026-07-01

Bump manifests 0.5.1 → 0.5.2. Bundles the post-0.5.1 chat/quality fixes:
- Persist the selected model per provider across relaunches.
- Retry (regenerate) an answer + Copy any message.
- Fuzzy retrieval fallback so niche/deep-dive questions aren't dropped as
  "no matching passages".
- Stream chain-of-thought live for reasoning models instead of a static status.
- Fix Fireworks "401": don't default to a non-chat (image) model from /models;
  always include + prefer the configured chat model.

## v0.5.1 — 2026-06-30

Bump manifests 0.5.0 → 0.5.1. Bundles the post-0.5.0 Themes + navigation work:
- Explore view: the library map as colored, size-weighted bubbles with drill-down
  and on-demand "five whys" deepening (deepen_theme) to 5 levels; an Ask at every
  level whose specificity scales with depth.
- VS Code-style icon rail (Chat / Themes / Settings) + collapsible sidebar.
- Theme-map coverage fix: dedupe version/MEAP variants, send all works (not the
  top-180-by-size), and instruct coverage of minority subjects — so finance,
  architecture, process, security, etc. are no longer dropped.

## v0.5.0 — 2026-06-30

Bump manifests 0.4.3 → 0.5.0. Headline: the Themes tab.

- Themes tab — a generated "Library map": the LLM organizes the indexed
  library into a theme → subtheme hierarchy (cached); each subtheme has
  "ask angle" chips (Overview / Key ideas / Compare / Open questions /
  Critique) that launch a fresh, grounded conversation scoped to that theme.
  Tolerant of weak models that emit stray control chars in JSON.

Also in 0.5.0 (since 0.4.2): one Index button (GPU auto-route, CPU fallback),
resumable GPU checkpointing, fp16, content-checksum dedup, ls-cli
backfill-state, in-app Help guide + docs refresh, fixed-height indexing log.
