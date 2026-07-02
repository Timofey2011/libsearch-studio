# LibSearch Studio — Roadmap 2: Conversation Memory + Outstanding Items

_Generated from a scout → 3-competing-designs → judge → plan → adversarial-critique pass over v0.5.9. The critique reached a genuine `solid` verdict (round 2, zero high-severity issues). Supersedes the completed [ROADMAP.md](ROADMAP.md)._

## Executive summary

LibSearch Studio v0.5.9 has shipped its entire previous 12-item roadmap (onboarding, follow-up dual-query fusion, provenance badges, token budgeting, cross-page chunking, clean-install polish). The app's remaining structural weakness is memory: turn 7+ of a conversation vanishes, mid-length follow-ups get no history in retrieval, and every conversation is an island. This plan makes conversation memory the centerpiece via a 'ledger, not brain' design — memory the user writes and can always see, never memory the app grows in secret — delivered in harness-gated small increments. The through-line is threefold: (1) every quality-affecting retrieval change ships behind a fixture slice (restoring the roadmap's own harness-before-tuning precedent); (2) memory is structurally non-citable — it never enters the numbered Sources block, so the grounded-citation promise cannot be contaminated; (3) zero new LLM calls on the hot ask path — the digest is deterministic string assembly, fusion costs only ONNX embeds, and the LLM rolling summary is a conditional escape hatch gated on felt need. Two non-memory leverage items ride along (stop button, re-index nudge), the broad golden-set harness follows the memory arc, and code-signing / repo-public remain parked external decisions. Everything respects local-first, offline-at-query-time, single-developer constraints.

## The memory design: “Ledger, not Brain”

Memory is a user-authored ledger plus computed context, one mechanism per gap. Data model: one new table — notebook(scope PK: 'global' or collection_id, content, updated_at) — in the existing SCHEMA idiom in crates/ls-app/src/store.rs, with get_note/set_note and a memory_enabled setting; delete_collection also removes the matching notebook row so no invisible orphans persist. In-conversation continuity needs no schema: turns older than the 6-turn window render as a deterministic 'Earlier in this conversation' digest (one-liners, 800-token cap) computed at prompt time from messages ask() already loads — retry-safe by construction. Prompt: build_prompt_with_history gains a notes param and returns (String, PromptMeta), enabling a new ask-context event (the bare-bool ask-provenance stays untouched); notes enter under a 'Reader's notes — context only, do NOT cite, sources take precedence' header, 600-token cap, truncate-from-end with a live counter; [n] markers are stripped from assistant-role history so stale citation numbers never collide with the current Sources block — citations stay structurally uncontaminated because memory never enters the numbered Sources. Retrieval: notebook text never touches retrieval; follow-up fusion uses a tiered gate (<=3 words always; 4-12 pronoun-led OR cosine >= ~0.5 vs the prior user turn; 13+ cosine-only) with the embedding passed into search_multi to avoid re-embedding, and rerank stays keyed on the current question. UI: Settings > Memory textarea (the textarea IS the entire memory), save-dialog Markdown export, backend-driven notes-active chip, per-ask 'context used for this answer' inspector, one-click save-to-notes, 90-day staleness cue. An LLM rolling summary (post-ask best-effort task, ask_in_flight AtomicBool skip, summary_ord retry clamp, editable drawer) ships only if the digest proves lossy in real use. Zero new LLM calls on the hot ask path.

**Why this design won.** Scoring across the five criteria: (1) User value — all three fix gaps (a)-(c), but Design 3's semantic memory retrieval mostly pays off at a scale (hundreds of conversations, thousands of notes) this single-user app may never reach, while Design 1 delivers the same felt improvements (no more turn-7 amnesia, follow-ups that work, standing preferences) immediately. (2) Trust/local-first — Design 1 wins by construction: the app never writes memory autonomously, so 'user-visible and editable' and 'no hidden profiling' are structural properties, not review checklists; Design 2's proposed/accept gate is a good second; Design 3 has autonomous LLM summaries injected by similarity score, plus an empty safeguards section — a real smell. (3) Solo-dev cost/risk — Design 1's first two increments need zero schema changes and zero LLM calls; Design 3 requires embedding-BLOB plumbing, threshold tuning without the (still unbuilt) eval harness, and Engine-mutex contention handling; Design 2 sits between. (4) Hot-path cost — Design 1: zero added LLM calls, one SQLite read; Design 3: extra embeds plus a background LLM call contending for the Engine right when the user fires a follow-up; Design 2: one background LLM call per ask past 6 turns (real money on metered cloud providers). (5) Increment independence — Design 1's increments are cleanly separable and each is roughly same-day shippable; the others front-load schema+UI before user value lands. Design 1 wins, but two losers contribute load-bearing grafts: from Design 3, the cosine-similarity gate for long follow-ups (fixes gap (b) for 13+-word questions, which Design 1's 12-word cap silently abandons), the per-answer transparency-chip idea, and the memory-eval fixture cases for the future harness; from Design 2, the summary_ord retry-clamp mechanics and editable-summary drawer for the conditional LLM-summary increment, plus the 'may be outdated / sources always take precedence' header phrasing, which is better prompt hygiene than Design 1's header alone.

## Top 3 to do first
1. **Strip stale [n] citation markers from assistant-role history in the prompt** (`P13`) — high impact / S effort
2. **User-editable Notebook: visible cap, save-dialog export, clean deletion lifecycle** (`P3`) — high impact / M effort
3. **Smarter follow-up fusion (tiered gate: pronoun OR cosine, validated on P4a slice 1)** (`P1`) — high impact / S effort

## Near-term

### Memory fixture harness slice (lands first, gates the fusion change)
`P4a` · **ops** · impact **high** · effort **M**

**Problem.** P1's cosine threshold is a guess and downstream memory items need regression checks, but the codebase has no deterministic generation (no temperature/seed anywhere in ls-llm) and the gate fns are private in src-tauri — naive answer-side fixtures would flake and time-travel against features that land later.

**Proposal.** Split scope per critique. Slice 1 (gates P1): retrieval-side only, fully deterministic, no LLM — EN/RU cosine-threshold pairs with fused/not-fused assertions, exercised via search_multi plus the gate extracted to ls-query or covered by src-tauri unit tests. Slice 2 (lands WITH P13 and P3): answer-side fixtures — model never emits an [n] that appears only in history/digest/notes; note contradicting a book must not flip the cited answer; retry after a note edit re-reads the note (documented intended behavior) — gated on a ~20-line temperature:0/seed override in ls-llm used only by the harness (doubles as a down-payment on deferred gen-params). Local, on-demand, no CI model dependency.

**Risks.** Answer-side checks stay limited to hard invariants even at temperature 0 — leave quality grading to P4b.

**Grounded in.** Critique P4a fix + 'missing' determinism item; docs/ROADMAP.md:154,156; OPS.md:201; grep-verified absence of temperature params in crates/ls-llm

### Smarter follow-up fusion (tiered gate: pronoun OR cosine, validated on P4a slice 1)
`P1` · **memory** · impact **high** · effort **S**

**Problem.** Retrieval only sees history via looks_anaphoric (1-3 words, or 4-6 pronoun-led); 7+ word follow-ups get zero history — gap (b).

**Proposal.** Tiered gate at src-tauri/src/lib.rs:1063-1091: <=3 words always fuse; 4-12 words fuse iff pronoun/connective-led OR cosine(embed(question), embed(prior_user)) >= threshold (start 0.5, validated on P4a EN/RU fixtures BEFORE flipping); 13+ words cosine-only. Pass the gate's precomputed question embedding into search_multi (Option param) so gated asks don't embed the question twice; honest cost ≈2 extra ONNX embeds per gated ask, and roughly doubled retrieval latency only when fusion actually fires. No LLM. Reuses shipped contamination-safe dual-query in crates/ls-query/src/lib.rs:247-295; rerank stays keyed on the current question.

**Risks.** Cosine distributions differ EN vs RU — P4a fixtures are the gate. Topic switches scoring above threshold still fuse; damage bounded by merge-not-replace and current-question rerank.

**Grounded in.** Memory design increment 1 + critique P1 embed-dedup fix; src-tauri/src/lib.rs:1063-1091; crates/ls-query/src/lib.rs:247-295,264

### Strip stale [n] citation markers from assistant-role history in the prompt
`P13` · **functional** · impact **high** · effort **S**

**Problem.** Prior assistant turns carry [n] markers from THEIR OWN retrieval whose numbering collides with the current Sources block; the persisted citations array maps clicks to current results only, so an echoed stale [3] opens an unrelated snippet — a live threat to the grounded-citation promise today, amplified by the P2 digest.

**Proposal.** In build_prompt_with_history (crates/ls-llm/src/lib.rs:92-139), regex-strip \[\d+\] from ASSISTANT-role turn content only (user turns verbatim — '[1984]', '[2]' list refs, code must survive); same function applied to P2 digest lines and any future P10 summary text. P4a slice-2 fixture: model never emits an [n] that only appears in history text. Ship strip first; substitute book titles only if answers regress.

**Risks.** Stripping slightly degrades the model's view of which source an old answer used — acceptable; title substitution is the fallback.

**Grounded in.** Critique round-1 'missing' item 2 + round-2 assistant-only fix; crates/ls-llm/src/lib.rs:92-139; persisted citations src-tauri/src/lib.rs:1351-1360

### Earlier-topics digest (kill the turn-7 hard drop)
`P2` · **memory** · impact **high** · effort **M**

**Problem.** Turns older than MAX_HISTORY_TURNS=6 are hard-dropped from the prompt with no summary — gap (a).

**Proposal.** In build_prompt_with_history, render turns older than 6 as one-liners (user 150 / assistant 250 chars, newest-first, 800-token cap) under 'Earlier in this conversation:'; extend the over-budget loop (crates/ls-llm/src/lib.rs:131-137) to shed digest lines before full turns. Digest lines pass through P13's stripper; truncation MUST reuse the char-boundary-safe truncate helper (byte slicing panics on Cyrillic). Pure string assembly from messages ask() already loads — no schema, no LLM, retry-safe by construction.

**Risks.** Truncation, not abstraction — early nuance can still degrade; P10 is the designed escape hatch, gated on felt need.

**Grounded in.** Memory design increment 2; crates/ls-llm/src/lib.rs:92-139,63-68,131-137; depends on P13

## Mid-term

### User-editable Notebook: visible cap, save-dialog export, clean deletion lifecycle
`P3` · **memory** · impact **high** · effort **M**

**Problem.** Zero cross-conversation memory — gap (c). A naive cap would silently inject only part of an unlimited textarea (and differ by language via the estimate_tokens EN/RU split), contradicting the user-visible-memory value; and notebook rows keyed to a deleted collection would persist invisibly.

**Proposal.** notebook(scope PK, content, updated_at) in the SCHEMA const; get_note/set_note Db methods + tauri commands in generate_handler!; memory_enabled setting; notes threaded into build_prompt_with_history under the non-citable 'Reader's notes' header (never enters Sources — contamination structurally impossible; 600-token cap, 300 floor). Visibility: live '~412 / 600 tokens injected' counter using the same estimate_tokens heuristic (labeled approximate), over-cap warning, truncate-from-END so the top of the note is the reliable region (stated in UI copy). delete_collection deletes the matching notebook row (no hidden orphans). Export-to-Markdown via a save dialog (~20 lines; the notebook's backup story — until it lands, quit-and-copy of the app SQLite covers it). Retry re-reads the note by design — P4a slice-2 fixture covers it. Global note first; per-collection only if uncluttered.

**Risks.** Notes can bias the model toward user beliefs over sources; header + strict SYSTEM line mitigate, the P4a adversarial fixture is the check.

**Grounded in.** Memory design increment 3 + critique P3 fixes + 'missing' collection-lifecycle item; crates/ls-app/src/store.rs:19-52,136; src-tauri/src/lib.rs:1776-1805

### PromptMeta plumbing: ask-context event, notes-active chip, per-ask context inspector (merged P6+P14)
`P6` · **memory** · impact **medium** · effort **M**

**Problem.** A frontend-inferred chip can lie (injection is decided inside ask() and the builder), and once notes + digest + trimmed history are assembled silently, a weird answer is undebuggable. The existing ask-provenance event is a bare bool emitted BEFORE prompt assembly, and the post-trim strings exist only inside the builder — neither can carry this signal as-is.

**Proposal.** One shared plumbing change (per critique): build_prompt_with_history returns (String, PromptMeta { notes_injected, notes_truncated, digest_lines, dropped_turns, token_counts }); emit a NEW ask-context event after assembly, leaving ask-provenance untouched at its existing site (src-tauri/src/lib.rs:1283). Two thin UI consumers: (1) notes-active chip in the answer footer (tooltip 'notes truncated for this answer' when trimmed) linking to Settings > Memory; (2) collapsible 'context used for this answer' drawer per assistant message showing the assembled non-Sources blocks post-trim — session-ephemeral, never persisted to the messages table. Plus a save-to-notes context action on assistant messages appending 'From <book title>: "<snippet>"' via set_note — capture stays explicit and user-initiated.

**Risks.** Effort is honest only as a merged item; keep the drawer read-only and ephemeral to avoid bloating storage.

**Grounded in.** Critique P6+P14 merge fix; src-tauri/src/lib.rs:1277-1290; frontend/src/App.tsx:294 bare-bool listener; design value 'nothing hidden to audit'; depends on P3

### Stop button — cancel in-flight generation
`P5` · **functional** · impact **medium** · effort **M**

**Problem.** No way to abort a running answer; persist-partial shipped (v0.5.6) but abort plumbing did not. Long generations on slow local models are uninterruptible.

**Proposal.** Copy the indexing cancel pattern (AtomicBool + cancel command, src-tauri/src/lib.rs:64-65,239-241) onto the ask path: abort flag checked in the generate_stream token callback + reqwest cancellation; on abort, reuse the shipped persist-partial path with an '[answer stopped]' suffix. Respects never-retry-after-first-byte.

**Risks.** Cancellation across ollama/cloud provider streams needs per-provider care in ls-llm.

**Grounded in.** docs/ROADMAP.md:155; scout fact: indexing cancel exists, ask path has zero cancellation; CHANGELOG v0.5.6

### Re-index nudge for v0.5.8 cross-page chunking (ships the ALTER-helper upgrade)
`P7` · **ux** · impact **medium** · effort **S**

**Problem.** Existing books keep old per-page loc=0 chunks until re-indexed; the user silently misses the v0.5.8 quality gain and the real loc metadata future cross-book summarization needs.

**Proposal.** Index-version marker per book (book_state; marker inferred by absence = old, no source-folder touches) + passive 'indexed with an older chunker — re-index recommended' badge per book with one dismissible banner. User-triggered only — never auto-scan on open (Dropbox-hydrate hang footgun). Since this is likely the FIRST new ALTER, it ships the small migration helper that swallows ONLY duplicate-column errors and logs everything else, retrofitted to the three existing ALTERs; P10 reuses it later (dependency direction fixed per critique).

**Risks.** Nudge fatigue on large libraries — per-book badge, dismissible banner.

**Grounded in.** CHANGELOG v0.5.8; docs/ROADMAP.md:146,159; libsearch-venv-dropbox-hang memory note; critique P3/P10 helper-ordering fix; store.rs:62-78

## Later

### Full golden-set retrieval harness (EN/RU expected-chunk assertions)
`P4b` · **ops** · impact **high** · effort **L**

**Problem.** 12 retrieval-affecting features shipped on manual checks only; P4a covers the memory arc but not general retrieval regressions, and it is the stated precondition for ever revisiting LLM query-rewrite.

**Proposal.** Extend the P4a runner with a broader EN+RU golden query set with expected-chunk/book assertions across fusion, fuzzy fallback, spell-repair, and cross-page chunking. Local, on-demand, no CI model dependency. Answer-quality grading later if ever.

**Risks.** Golden sets rot under re-indexing — key assertions to book+page ranges, not chunk ids.

**Grounded in.** docs/ROADMAP.md:154,156; OPS.md:201; critique P4 split

### Notebook staleness cue
`P9` · **memory** · impact **low** · effort **S**

**Problem.** Stale notes accrue silently and are injected into every ask; notes are static user prose, not auto-refreshed facts.

**Proposal.** Passive 'not edited in 90+ days — still accurate?' line under the textarea from notebook.updated_at. Never auto-deletes or rewrites user text. (Export already shipped with P3.)

**Risks.** None material — purely additive UI text.

**Grounded in.** Memory design increment 5, export half folded into P3 per critique; depends on P3

### LLM rolling summary (conditional — only if the digest proves lossy in real use)
`P10` · **memory** · impact **medium** · effort **M**

**Problem.** The deterministic digest (P2) is truncation, not abstraction; very long conversations may still lose early nuance.

**Proposal.** Only on felt need: conversations.summary + summary_ord via the P7 duplicate-column-only ALTER helper; post-ask-done spawned tokio task folds evicted turns (<=150 words, lazy recompute after 4+ new turns). Skip signal per critique: an ask_in_flight AtomicBool set/cleared around the whole ask body (same idiom as the indexing cancel flag) — Engine try_lock alone misses the generation phase since Llm is constructed per call (src-tauri/src/lib.rs:78); Engine try_lock stays as secondary guard. Failures dropped, never retried. Summary replaces digest at 300-token cap, passes the P13 stripper; editable drawer + Forget button; summary_enabled toggle; retry-safe via the summary_ord clamp (rebuild if MAX(ord) < summary_ord).

**Risks.** Gate on a felt problem after P2 ships, not speculation.

**Grounded in.** Memory design increment 6 + critique P10 skip-signal fix; src-tauri/src/lib.rs:64-65,78,1154-1195

### Citation keyboard-activation + modal a11y (explicitly last, timeboxed)
`P8` · **ux** · impact **low** · effort **M**

**Problem.** Citations in the streamed transcript aren't keyboard-activatable; the citation modal lacks Escape/focus-trap. Low-impact for a single-user desktop app whose sole user hasn't asked — acknowledged slips-forever profile.

**Proposal.** Two halves in order: (1) modal Escape-to-close + focus trap — trivial, independent, first; (2) focusable [n] buttons in the hand-rolled regex markdown renderer — fiddly over streamed text, timeboxed, dropped without guilt if it fights back. Sequenced dead last.

**Risks.** Renderer retrofit over streaming content is the fiddly half — the timebox is the mitigation.

**Grounded in.** docs/ROADMAP.md:157; critique split-halves fix

### Code-signing + notarization (EXTERNAL: Apple Developer account)
`P11` · **ops** · impact **medium** · effort **M**

**Problem.** Unsigned bundles force right-click-to-open; 'app is damaged' is a live troubleshooting entry. Blocks any future auto-updater.

**Proposal.** Parked decision, not scheduled work: user decides on the ~$99/yr account. Once purchased, sign + notarize in the release pipeline (M); defer the auto-updater itself — signing alone removes the Gatekeeper friction.

**Risks.** Blocked entirely on the external purchase.

**Grounded in.** OPS.md:187-188,226-227; docs/ROADMAP.md:159

### Repo-public audit (EXTERNAL: user decision)
`P12` · **ops** · impact **low** · effort **S**

**Problem.** Flipping Timofey2011/libsearch-studio public is desired but history may contain plaintext-key artifacts or personal library paths.

**Proposal.** Preparatory audit only: skim full history for keys, settings.toml samples, personal paths; if anything sensitive surfaces, decide between history rewrite and staying private. No feature code.

**Risks.** None beyond the user decision itself.

**Grounded in.** Outstanding-items list; constraint: API keys plaintext in settings.toml

## Deliberately deprioritized

- **True LLM follow-up query-rewrite** — Deferred behind the P4b golden set per the roadmap's own rule — only if the tiered fusion gate (P1) measurably falls short, since it adds an LLM call to the hot ask path.
- **Full backup/restore of collections** — reveal-data-folder (shipped) plus P3's notebook Markdown export cover the realistic single-user backup story; a full import/export pipeline is L-effort with no felt demand.
- **EPUB/MOBI + OCR ingest** — EPUB text may already index via the old path and the reader is PDF-only; a whole ingest+reader arc is a separate epic that would crowd out the memory centerpiece for one developer.
- **Stale-index auto-scan on open** — Actively dangerous with cloud-synced source folders (Dropbox-hydrate hang, per memory note); P7's passive user-triggered nudge delivers the value without the footgun.
- **Cross-book map-reduce summarization** — Needs a chunks_for_book store method and real loc metadata library-wide, which only exists after the user re-indexes post-v0.5.8 (P7's job); revisit once re-indexing has actually happened.
- **Generation-param controls (temperature etc.)** — No user demand; the only concrete need — deterministic output for fixtures — ships as P4a's minimal temperature:0/seed override, a down-payment if full controls are ever wanted.
- **Auto-updater** — Hard-blocked on P11's external signing decision, and signing alone removes the Gatekeeper friction that motivated it.
- **ANN / vector-index acceleration** — No observed latency problem at current library scale; premature infrastructure for a single-user local app.

## Critique history

| Round | Verdict | High issues | Missing |
|---|---|---|---|
| 1 | needs_work | 1 | 3 |
| 2 | solid | 0 | 2 |

