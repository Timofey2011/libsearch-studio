# LibSearch Studio — Improvement Roadmap

_Generated from a scout → plan → 5-round adversarial-critique → synthesis pass over the v0.5.5 codebase. Every item is grounded in a `file:line`. This is a living doc — reprioritize as things ship._

## Executive summary

LibSearch Studio (v0.5.5) is a mature local-first Rust+Tauri+React RAG desktop app: bge-m3 embeddings + reranker over a LanceDB hybrid store, multi-provider LLM chat with clickable [n] citations, Themes/Explore tabs, GPU sidecar indexing, and one-click self-setup. The core engine works; the gaps are at the edges — first-run dead-ends, silent failure modes (hangs, panics, lost partial answers), and honesty of grounding (follow-ups, aggregative queries, confidence provenance). The through-line of this plan: make the app trustworthy and survivable in the hands of a real first-time user without touching the working retrieval core. It front-loads a minimal onboarding path plus a cluster of small backend-hardening fixes (timeouts, graceful engine/model errors, persist-partial-on-error, prompt budgeting), layers honesty signals onto retrieval (history-aware follow-ups, aggregative caveats, loose-tier provenance), guards the Dropbox/iCloud corruption footgun, and makes one focused RAG-quality bet (cross-page GPU chunking). All items respect the constraints: offline at query time, no telemetry, plaintext keys never re-exposed, index/models out of sync mounts, single-user desktop.

## Top 3 to do first
1. **First-run onboarding: replace dead empty-state with add-folder → provider → index card** (`ux-onboarding-wizard`) — high impact / M effort
2. **Always persist the partial answer on mid-stream error/timeout** (`func-persist-partial`) — high impact / S effort
3. **Connect timeout + per-token idle deadline on LLM/network calls (no post-first-byte retry)** (`func-llm-timeouts`) — high impact / S effort

## Near-term

### First-run onboarding: replace dead empty-state with add-folder → provider → index card
`ux-onboarding-wizard` · **ux** · impact **high** · effort **M**

**Problem.** A fresh user hits a dead end: empty transcript invites 'Ask a question' but the composer is disabled, the picker says 'No collections', and nothing routes to Settings → Collections. The seeded default 'My Library' collection points at empty source_paths (src-tauri/src/lib.rs:1500-1506). This is the single biggest barrier: a new user literally cannot reach a first answer.

**Proposal.** When collIds is empty, replace the dead empty-state with an actionable onboarding card wired straight to the Collections tab (add source folder), provider pick (or local Ollama), and the index action. Extract as its own component to bound blast radius. Ship the minimal path THIS cycle; defer only the prereq-check/cost-warning polish (python3 present / Ollama running / disk free / GPU-setup ~10-20 min + several GB + restart note).

**Risks.** Adds UI surface; scope to a single component. Prereq/cost warnings deferred as polish.

**Grounded in.** Frontend gap 'No first-launch onboarding' + Ops gap 'First-run has no guided wizard' (App.tsx:1790-1792,1890,1913; src-tauri/src/lib.rs:1500-1506). Critic (high): promote from next-cycle — highest user value.

### Always persist the partial answer on mid-stream error/timeout
`func-persist-partial` · **both** · impact **high** · effort **S**

**Problem.** The persist path map_err's the stream result (verified src-tauri/src/lib.rs:1128-1140, persist only on Ok), so if generation errors or times out mid-stream, the tokens already emitted to the UI are lost from history — a long answer that fails at 90% vanishes from the conversation. Felt by every user with a flaky provider.

**Proposal.** Restructure ask() to accumulate streamed tokens as they arrive and persist the partial assistant turn on the error/timeout branch (marked incomplete/stopped), not only on Ok. Standalone S-effort fix, split out from the heavier cancel-token plumbing (which is deferred). Reuse the func-llm-timeouts stall path for the timeout case.

**Risks.** Saving the partial must not corrupt the persisted conversation; mark incomplete and never retry the aborted call.

**Grounded in.** Critic (high/missing): persist-partial-on-error is S-effort, high-value, and was trapped inside the M-effort cancel item. Verified persist only on Ok at src-tauri/src/lib.rs:1128-1140.

### Connect timeout + per-token idle deadline on LLM/network calls (no post-first-byte retry)
`func-llm-timeouts` · **functional** · impact **high** · effort **S**

**Problem.** Every provider builds a bare reqwest::Client::new() with no timeout, and generate_stream awaits with no deadline. A slow Ollama cold-load, stalled cloud SSE, or dropped connection hangs the whole ask indefinitely with no user escape; transient 429/5xx on probes are never retried.

**Proposal.** Give each reqwest client a .connect_timeout() (connect phase only) — NOT request-level .timeout(), which would kill legitimate long streaming generations. Add a per-token idle/stall deadline by wrapping stream.next().await in tokio::time::timeout inside run_stream (verified it owns the loop at crates/ls-llm/src/lib.rs:333 `while let Some(chunk) = stream.next().await`, so genuinely S); use a generous ~90-120s window reset on every chunk to survive Ollama cold-load, and surface a clean 'provider timed out / disconnected' error on stall. Add bounded exponential backoff (1-2 attempts) for transient 429/5xx/connection errors ONLY on the non-streaming probes (list_models/probe_provider), strictly pre-first-byte. Once any byte/token arrives, NEVER retry.

**Risks.** Too-aggressive idle timeouts could cut off legitimate local cold-loads; keep the stall window generous. Hard rule against retry after first byte (double-generate/double-bill).

**Grounded in.** Backend gap 'No timeout or retry on any LLM/network call' (crates/ls-llm/src/lib.rs:456-459,495-507,553-567,614-625). run_stream owns the byte loop at 318-335.

### Actionable 'models not set up' error on the ask path
`func-engine-availability` · **both** · impact **high** · effort **S**

**Problem.** ask() lazily calls Embedder::load / Reranker::load (verified src-tauri/src/lib.rs:1048-1053) and map_err's to a raw string, so a user who added a collection but never provisioned models (or whose models dir moved) gets a cryptic load error on their first question instead of a pointer to setup — the single most likely first-run failure once onboarding adds folders.

**Proposal.** Detect the models-not-found / load-failure case from Embedder::load/Reranker::load and map it to an actionable 'models not set up — run Settings → Indexing → Set up' message. Do NOT add mutex-poison recovery: verified engine is tokio::sync::Mutex (use tokio::sync::Mutex at lib.rs:24, .lock().await at 1047), which does NOT poison, so PoisonError/into_inner machinery would not compile; the guard.as_mut().unwrap() at 1055 is harmless because guard is populated two lines above.

**Risks.** Model-absence detection is heuristic on the load error — keep the message a pointer, not a claim of certainty.

**Grounded in.** Critic (high): round-5 poison premise factually wrong (tokio mutex, verified) — drop it; keep only the actionable models-absent message (lib.rs:1048-1053).

### Raise Anthropic 2048-token cap (quick win, Anthropic-only)
`func-anthropic-maxtokens` · **functional** · impact **high** · effort **S**

**Problem.** ANTHROPIC_MAX_TOKENS is a const 2048 (verified crates/ls-llm/src/lib.rs:519,560), so long grounded Claude answers are silently truncated mid-sentence — the single most user-felt generation defect.

**Proposal.** Raise ONLY the Anthropic default cap to a sensible value (e.g. 4096-8192) as a one-line const change. Do NOT inject a hardcoded max_tokens into the OpenAI-compat or Ollama paths: they currently send none (Ollama sends only num_ctx at 502) and rely on provider defaults, so adding a cap where none existed can 400 on models with lower limits and regress working setups.

**Risks.** A larger Anthropic cap raises cloud cost/latency slightly; keep the default conservative.

**Grounded in.** Backend gap 'Anthropic max_tokens hardcoded and low' (verified crates/ls-llm/src/lib.rs:519 ANTHROPIC_MAX_TOKENS=2048, 560 usage).

### Reveal data folder (quick win)
`func-reveal-data-folder` · **functional** · impact **medium** · effort **S**

**Problem.** Index, app.db, and settings.toml live under one data dir but there is no in-app way to even find it. Combined with the Dropbox prohibition, users are told 'don't sync it' with no way to locate it for a manual copy/backup.

**Proposal.** Add a single 'Reveal data folder' action (Tauri opener / reveal-in-file-manager) in Settings. Near-zero risk, one command. The sanctioned backup recipe stays 'quit the app, copy the data folder.' A fuller backup/restore is explicitly deferred.

**Risks.** None material.

**Grounded in.** Ops gap 'No backup, portability, or export' (no reveal/backup command among the 26 tauri commands).

## Mid-term

### History-aware retrieval for follow-ups via dual-query fusion (zero-LLM)
`func-followup-query-fuse` · **functional** · impact **high** · effort **S**

**Problem.** In multi-turn chat, retrieval uses only the raw current question (search_multi(..., &question), src-tauri/src/lib.rs:1071), so follow-ups like 'why?' embed/search on that fragment alone and pull passages nearest to 'why' rather than the prior topic. The answer reads coherent (history reaches the LLM) but grounding is silently wrong.

**Proposal.** When the current question is genuinely contentless/anaphoric (pronoun-only or a very small token threshold), run BOTH the bare current question AND a concatenation with the previous user turn as retrieval queries, then fuse the result sets — so a mis-gated short topic shift (e.g. 'what about Redis?' after a Postgres thread) still retrieves on its own terms instead of being dragged to the prior topic. Zero extra LLM round-trip, fully offline, instant. Keep the original question in the answer prompt. Reserve true LLM query-rewrite for a later cycle behind the eval harness only if fusion proves insufficient.

**Risks.** Fusion slightly broadens the candidate pool; the reranker still orders. Never fires on first turn.

**Grounded in.** RAG gap 'Retrieval is question-string keyed, follow-ups retrieve off-topic' (src-tauri/src/lib.rs:1067-1076). Critic (medium): fuse bare+concatenated queries so topic shifts aren't corrupted; narrow trigger to contentless turns.

### Honest caveat on aggregative / whole-book queries
`func-aggregative-intent` · **functional** · impact **high** · effort **S**

**Problem.** final_top_k=8 chunks nearest to 'summarize' get presented as a whole-book summary — actively misleading, and exactly what a personal-library-chat user asks constantly. The 8 confident-scoring chunks are inherently unrepresentative of a whole book whether or not they clear min_relevance, so the misleading summary ships with no caveat precisely when the user trusts it most.

**Proposal.** Detect aggregative/whole-book intent with tight keyword/pattern heuristics ('summarize','main themes','what is this book about','overview') and fire an honest caveat REGARDLESS of the loose flag (do not gate on weak retrieval — the canonical failure is a well-indexed book returning 8 confident-but-unrepresentative chunks). Use tight patterns to avoid false positives on localized 'summarize chapter 3's argument on X'. On a hit, prepend a caveat that the response is drawn from a limited set of retrieved passages, not the full text. Do NOT attempt cross-book widening: the store exposes no per-book chunk enumeration (verified store.rs); real widening needs a NEW chunks_for_book method + loc metadata — deferred.

**Risks.** Intent detection is heuristic; tight patterns keep false positives on localized questions down. True widening deferred.

**Grounded in.** RAG gap 'Aggregative queries architecturally unanswerable' (src-tauri/src/lib.rs:1072-1073). Critic (high): fire on intent regardless of loose flag; tighten patterns instead.

### Surface grounding provenance: loose-tier badge (+ collection attribution if threaded)
`ux-answer-provenance` · **both** · impact **medium** · effort **S**

**Problem.** The tiered-relevance logic silently switches between the confident tier and the loose fuzzy-floor fallback (verified src-tauri/src/lib.rs:1083-1090) with no user indication, so a well-grounded answer and a barely-grounded one render identically. The hard 'no matching passages' message (lib.rs:1102) and the loose-tier state also aren't visually distinguished.

**Proposal.** Compute a single loose: bool in Rust at the src-tauri/src/lib.rs:1083 branch (do NOT re-derive in JS — it would duplicate the threshold/loose-floor formula and drift). Render a small badge ('answered from loosely-related passages') only when loose fired, worded as provenance, never a numeric confidence (the 0.15 threshold is an uncalibrated cross-encoder logit). Visually distinguish the 'no matching passages' and loose-tier states so both low-confidence states read as calm orientation, not alarm. For collection attribution: VERIFY SearchResult/to_citation actually threads collection identity through the search_multi merge before committing; if it does, surface the distinct collection name(s) near the Sources list — if not, ship only the badge and treat collection attribution as a separate small add.

**Risks.** Raw logits uncalibrated (RU vs EN scores differ) — hence provenance wording, not a number. Collection attribution contingent on metadata being threaded.

**Grounded in.** RAG gap 'No confidence visibility' + missing 'which collection grounded the answer'. Critic (low): verify collection identity survives merge; ship badge regardless. Missing: distinguish no-match vs loose states.

### Warn when INDEX/models path is under a cloud-sync mount (not source paths)
`ux-dropbox-guard` · **ux** · impact **high** · effort **S**

**Problem.** Every doc warns the index and models must never live on Dropbox/iCloud (sync corrupts them), but the only enforcement is a code comment (crates/ls-app/src/lib.rs:23). Given the project's own recorded 'venv Dropbox hang' incident, a user whose data dir is redirected into a sync mount gets a silent hang or corrupted index.

**Proposal.** Apply a path check to the DATA DIR and any collection db_path (the index) only — do NOT warn on user-selected source_paths, since reading an ebook library from a synced folder is a normal, supported setup and warning there trains users to dismiss the warning that matters. macOS patterns: Dropbox, Library/CloudStorage, Library/Mobile Documents, iCloud~. Linux: check for a .dropbox.cache sibling or a Dropbox root from ~/.config/dropbox host.db; where Linux detection is impossible, keep the doc warning as fallback rather than implying the guard is comprehensive. On a hit, show a clear non-blocking, dismissible warning in Settings and at index time.

**Risks.** Path heuristics can false-positive; keep the warning dismissible, not a hard block. Linux detection is partial by nature.

**Grounded in.** Ops gap 'No runtime guard keeps index/models out of Dropbox/iCloud' (crates/ls-app/src/lib.rs:23-31, comment-only). Critic (low): scope to data dir + db_path only, not source paths.

### Bound assembled prompt via script-aware char heuristic (trim oldest history, never grounding)
`func-prompt-token-budget` · **functional** · impact **medium** · effort **S**

**Problem.** NUM_CTX is a fixed 16384 (verified crates/ls-llm/src/lib.rs:445) and nothing bounds the assembled prompt (system + history + up to 8 retrieved chunks) against the context window. Long chunks or long history can silently truncate grounding sources or 400 the request — undermining citation trust.

**Proposal.** Add a best-effort budget check in build_prompt_with_history using a script-aware size estimate (~char/4 for Latin, but ~char/2.5 when significant Cyrillic is present, since Cyrillic BPE fragments Russian ~2x more than /4 implies — this app targets mixed EN+RU libraries) with a generous safety margin derived from NUM_CTX and output headroom. When over budget, trim OLDEST history turns first and NEVER drop or truncate the numbered source passages the [n] citations point at; if the sources alone overflow, drop the lowest-ranked source(s) whole rather than mid-truncating. Lean explicitly on func-llm-timeouts' 400-handling as the real backstop. State explicitly this is best-effort, not exact.

**Risks.** Char estimation is approximate — the margin plus 400-handling absorb the slack. Trimming history loses conversational context; trim oldest-first, keep the current turn.

**Grounded in.** Backend gap NUM_CTX fixed (lib.rs:445); build_prompt_with_history concatenates unbounded. Critic (medium): script-aware divisor for Cyrillic corpus.

## Later

### Fix GPU chunker: cross-page packing + real loc metadata (Python-only)
`func-gpu-chunker-page-loc` · **functional** · impact **high** · effort **M**

**Problem.** The default GPU path token-windows each PAGE independently and hard-writes loc_start/loc_end=0 (scripts/gpu_embed.py:60-78,147-148), so chunks split at page breaks and mid-sentence — hurting reranker precision and answer coherence — and non-PDF/loc citations are meaningless.

**Proposal.** In scripts/gpu_embed.py, stop windowing per-page: concatenate page text and pack cross-page paragraph-aware windows so chunks no longer split at page boundaries, and write real page-range loc metadata instead of 0. Python-only, no new IPC contract, no engine swap, keeps PyMuPDF extraction. Pair with an index-version marker or explicit in-app 'chunking changed — re-index recommended' note so old page-windowed and new cross-page chunks don't silently coexist with inconsistent loc semantics; document re-index as recommended-not-optional. Sanity-check chunk-count/quality with manual spot-checks. Does NOT deliver 'Ch. X' citations for PDFs (Rust PDF path also emits chapter=None at crates/ls-extract/src/lib.rs:72).

**Risks.** Changes chunk boundaries so old and new indexes differ — document a re-index recommendation. Does NOT deliver chapter citations for PDFs.

**Grounded in.** RAG gap 'GPU indexer uses a cruder chunker' (scripts/gpu_embed.py:60-78,147-148). Critic (low): pair with index-version marker / re-index note.

## Deliberately deprioritized

- **Eval / golden-set harness** — Gates nothing this cycle — gpu-chunker and provenance de-gated to manual spot-checks, vector-ANN already dropped. It was the largest effort item front-loaded ahead of the wins it supposedly de-risked. Build the golden set opportunistically as regressions surface.
- **Stop-button cancellation of in-flight generation** — The high-value half of the old cancel item (persist-partial-on-error) is promoted to near-term standalone. The remaining abort-token/reqwest-cancellation plumbing is M-effort, single-user, and lower urgency; deferred to a later cycle.
- **True LLM follow-up query-rewrite** — Zero-LLM dual-query fusion (func-followup-query-fuse) covers most anaphoric follow-ups with no hot-path round-trip, no cloud cost, and no silent-drift failure. Revisit an LLM rewrite behind the eval harness only if fusion proves insufficient.
- **Citation-keyboard-activation a11y** — Retrofitting focusable controls into the hand-rolled regex markdown renderer across a long streamed transcript is a distinct, fiddlier change than modal-Escape/focus-trap; split out as its own small next-cycle item. (Modal a11y + status-text alternative also deferred out of this compact set as lower-leverage than the trust/first-run fixes.)
- **Vector ANN index** — Flat scan is tens of ms at ~292k chunks; revisit only past ~1M chunks.
- **Full backup/restore, EPUB/MOBI+OCR ingest, auto-updater/code-signing, generation-param controls, stale-index auto-scan, cross-book map-reduce summarization** — Reveal-data-folder is the safe portability primitive; the rest are either large scope, out-of-constraint, or footguns (on-open scan risks the Dropbox-hydrate hang). Cross-book summary needs a new chunks_for_book store method + loc metadata first. All explicitly out of this cycle.

## Critique history (adversarial review)

| Round | Verdict | High issues | Missing |
|---|---|---|---|
| 1 | needs_work | 2 | 4 |
| 2 | needs_work | 1 | 3 |
| 3 | needs_work | 2 | 2 |
| 4 | needs_work | 0 | 2 |
| 5 | needs_work | 3 | 2 |

_Round 4 reached 0 high-severity issues; round 5's re-raised items and both round-5 'missing' items were folded into the final synthesis above._

