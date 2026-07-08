# Changelog

All notable changes to LibSearch Studio, newest first. Each version is a git tag (`vN`) and a GitHub release with the `.dmg` attached.

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
