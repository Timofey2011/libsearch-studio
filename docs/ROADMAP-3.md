# ROADMAP-3 — Format Expansion: Plain & Rich Ingest + In-App Readers

Status: PLANNED · Owner: LibSearch Studio · Baseline: v0.8.0 (PdfReader shipped)
Scope: (A) plain-format ingest (.md/.txt + cheap extras), (B) rich-format ingest (docx, rtf, odt, fb2, epub, mobi, pages, …), (C) in-app display for rich formats including the missing EPUB reader.

Ground truth carried into this plan (verified): no `com.libsearch.studio` data dir exists on the dev machine — the live `app.db` lives where Studio runs, so §2.1.0 is a gated first step with both branches designed. Verified in code: `book_state` PK `(collection_id, book_id)` with `fingerprint/content_sig/chunker_ver` columns; `alter_ignore_duplicate` is ADD-COLUMN-only; `run_backfill_state` exists at `crates/ls-cli/src/main.rs:156–190` and **writes rows even when `content_signature` fails** (main.rs:178–186 — csig lands as the literal string `"missing"`); `content_signature()` returns `"missing"` on ANY open failure (service.rs:110–112); `book_id_for_content` guards only the empty string (ls-app/store.rs:379–381); `book_id_for_fingerprint` uses `LIMIT 1` **and returns only the book_id** (ls-app/store.rs:354/:360) — the guards in §2.1.2 need the whole row, so the lookups are widened in M0a; `fast_index_collection` additionally keeps an **in-run `seen_content: HashMap<csig, book_id>` dedup map** (src-tauri/src/lib.rs ~:535) that today inserts and matches the `"missing"` sentinel — a fourth sentinel site the hardening must cover; the stage-4 content-signature old≠new remap exists on BOTH pipelines (service.rs:272–287, src-tauri/src/lib.rs:538–546); `fast_index_collection` is a `#[tauri::command]` with the whole 4-stage pre-filter inlined (src-tauri/src/lib.rs:497–550) and no headless entry point; `scripts/gpu_embed.py` imports `torch`/`sentence_transformers`/`transformers`/`fitz` **at module level** (gpu_embed.py:29–33), so any new CLI mode must run before those imports; gpu_embed.py already accepts `--device` (gpu_embed.py:136, default `"mps"`) but `run_py_batch` never passes it and no setting carries it; the persisted `Citation` lives at `crates/ls-app/src/types.rs:46`, `to_citation()` is the mapper at src-tauri/lib.rs:39–47, and `SearchResult` already carries `chapter: Option<String>` (ls-query/src/lib.rs:31); `CURRENT_CHUNKER_VER = 1`.

---

## 0. Invariants (every milestone must respect these)

1. **Never re-embed the existing 302k-chunk library.** All format work is additive: new rows only. This explicitly covers **every book imported by the old Python pipeline** — the 231 .md books, the **69 epubs**, and any legacy pdfs from the same import. Enabling their extensions in `SUPPORTED` re-scopes existing collections over the folders that contain them; whether the current dedup protects them depends on whether `backfill-state` was ever run — a question M0a answers with a one-line query before any code is written (§2.1.0). A dedicated guard covering the whole legacy set is an M0a exit criterion, **re-verified as an M2 exit criterion** when `epub` flips on.
2. **Fully offline at runtime.** No network calls; all JS libraries vendored and pinned (no CDN). New Python deps installed at setup time only, with graceful directed-skip degradation if absent.
3. **File extension of the original file is the format signal at read time** (legacy books are stamped `format='pdf'` in the store). Readers key off the extension of `source_path` — which is **always the original file** (see §0.b for converted formats). We fix stamping *going forward* and offer an *optional* metadata repair — never required.
4. **CPU and GPU ingest paths move in lockstep.** Every extension in `discover.rs SUPPORTED` must have (a) a `Format` variant + an `ls-extract` arm (or a directed CPU skip), and (b) a handler branch *or a directed-skip branch* in `scripts/gpu_embed.py` — never a fall-through to `fitz.open()` on a format it can't open. Enforced by a CI test (§2.5).
5. **No silent failures, no permanent skip noise, and no cross-pipeline poisoning.** Every skipped/unsupported file surfaces an explicit reason via the `Skipped` IndexEvent **once per pipeline**. Skip records — including, going forward, **"no extractable text"** (§2.8, a deliberate behavior change from today's success-shaped `book_state` write at service.rs:310–321) — live in a **dedicated `skip_state` table keyed by `(collection_id, source_path, pipeline)`**, with the fingerprint stored as a *staleness column, never as identity* (§2.8 — size:mtime demonstrably collides between distinct files in this library, §2.1.2), never in `book_state`, so a skip can never masquerade as a success, be remapped by stage-2, hide a *different* file that happens to share a fingerprint, or hide any file from the *other* pipeline. Per-file outcomes on the GPU path travel over a **machine-parseable sidecar channel** (§2.10), never parsed out of prose stderr.
6. **The `"missing"` sentinel is poison, never identity — at all four matching sites.** `file_fingerprint`/`content_signature` failure sentinels are treated as no-match by every reverse lookup (`book_id_for_fingerprint`, `book_id_for_content`), are **never inserted into nor matched against the in-run `seen_content` dedup map** (the fourth site, src-tauri lib.rs ~:535 — carried over into `plan_index_run` with the same rule), are never persisted into `book_state` or `skip_state`, and pre-existing sentinel rows are purged by migration (§2.1.1). An unreadable file (permissions, Dropbox-dehydrated placeholder — a documented failure mode on this machine) produces no state, no remap, and its own once-per-run `Skipped` event — even when several unreadable files occur in the same run.
7. **WKWebView traps apply to any JS renderer:** no worker-originated fetches of custom schemes (`tauri://`/`asset://`); no `for await` over ReadableStream in JavaScriptCore. Main-thread `fetch` of `asset://` (incl. Range) works. Pattern for all new readers: fetch bytes on the main thread → hand `Blob`/`ArrayBuffer` to the library.
8. **dmg budget:** currently ~67 MB; total frontend additions across this roadmap ≤ +1.5 MB.
9. **One extension-derivation rule everywhere.** A single longest-match `ext_of(path)` (handles compounds like `.fb2.zip`) is the only way any layer derives a format from a filename; the frontend copy is **generated from the Rust canonical list**, not hand-mirrored (§2.2, §2.5).
10. **No synchronous many-MiB DOM builds on the WKWebView main thread.** Any reader that turns bytes into same-document DOM must either cap input small or render progressively (§3.5, §6.2). The same discipline applies backend-side: any Tauri command doing CPU-bound extraction runs under `spawn_blocking` with a frontend loading state (§5.3).
11. **Book identity is `source_path`, not `book_id` — and not fingerprint.** Two id schemes coexist by design (CPU `DefaultHasher`, GPU sha1, plus legacy Python ids in the store). Every guard, test assertion, maintenance action, and piece of persisted state introduced by this roadmap keys on `source_path` (or fingerprint **confirmed by content signature when it must prove identity across paths**, §2.1.2) — never on `book_id` equality across pipelines or across a re-index, and never on a bare size:mtime fingerprint.

### 0.b Conversion cache contract (applies to every converted format: .pages preview, .doc→txt, rtf/odt→html, .webarchive→html)

- **`source_path`, `book_id`, fingerprint, and content signature always refer to the ORIGINAL file.** The moved-file guard, "Open in default app", dedup, and re-index all track the original. A conversion artifact is never a book identity.
- Conversions are cached at `<data_dir>/converted/<content_signature>.<ext>` (content signature already exists and is format-agnostic). Cache invalidation is automatic: fingerprint change → new signature → new cache entry; stale entries GC'd opportunistically.
- Ingest of a converted format runs the converter, then feeds the cached artifact's *bytes* through the normal extractor — but stamps `source_path`/`book_id`/`format` from the original (e.g. a .doc ingests its cached .txt but stores `format: Doc`, `source_path: /…/report.doc`).
- Display goes through a new backend command **`resolve_display_path(path) -> { display_path, converted: bool, converter: string|null }`**: returns the original path for natively-displayable formats, or the cached artifact (creating it on demand) for converted ones. `openSource` calls it once; the reader-kind map then keys off the *display* artifact's kind while all identity/citation state keeps the original `source_path`.

Effort key: **S** ≤ 1 day · **M** = 2–4 days · **L** = 1–2 weeks · **XL** = 2–4 weeks.

---

## 1. Milestone map

| Milestone | Scope | Effort | Named exit criteria |
|---|---|---|---|
| **M0a** Identity guards, sentinel hardening, seed, harness | Ground-truth db query; **row-returning reverse lookups**; `book_state.source_path` column; legacy-book duplication guard (all ~300 legacy books) covering BOTH the fingerprint and content-signature stages; `"missing"`-sentinel hardening at **all four sites** + purge; fingerprint-collision csig confirmation; **headless `plan_index_run` extraction** + real-library snapshot fixture mechanism; seed migration + Maintenance UI surface; persistence audit; four-scenario acceptance harness | **L–XL** (no user-visible features; the highest-stakes engineering in the roadmap — the harness + snapshot fixture alone is multi-day) | ground-truth query recorded & branch chosen (§2.1.0); legacy-book acceptance test green — all four scenarios (§2.1); sentinel tests green incl. two-unreadable-files scenario (§2.1.1); persistence audit done (§2.1.4). **Blocks M1 CPU work.** |
| **M0b** State schema, GPU outcome channel, lockstep plumbing | `Format` enum growth; shared `ext_of` + TS codegen + lockstep CI test; `skip_state` table (**path-keyed**, §2.8) + stage-0.5; GPU sidecar + outcome-aware batch commit + `run_py_batch` contract change; gpu_embed.py restructure (**stdlib `--caps` prologue**, lazy heavy imports) + stamping fix; **`gpu_device` setting plumbed through `run_py_batch`**; import hard-error + batch containment; repair/orphan commands; copy strings | **L** | lockstep test green (§2.5); outcome-sidecar tests green (§2.10); cross-pipeline skip + no-text skip + **skip-collision** tests green (§2.8); `--caps` cold-start latency test green (§2.5). **Blocks the GPU legs of M1 and all M2+ skip machinery; M1 CPU dev can proceed in parallel on M0a alone.** |
| **M1** Plain ingest | .md .markdown .txt .text .rst .adoc .org .tex .ipynb .html/.htm; per-format Finished counts (sidecar-derived on GPU) | M | real-library index run: 0 new rows and 0 embeds for **all** legacy books (asserted via the §2.1 harness), 0 re-embeds of fingerprinted pdfs; fp-collision test green with a batch-exported-notes fixture. Ships only with M0a **and** M0b landed (release gate — see §12) |
| **M2** Ebook ingest | .epub .fb2 .fb2.zip .mobi .azw3 (non-DRM), .xps (GPU only); per-book "Re-index this book" action (source_path-keyed, §4.2) | M | **existing 69 epubs skip with 0 new rows (§2.1 test re-run); NEW epub fixtures index with TOC chapters**; RU fb2 golden set green |
| **M3** Reader epic | foliate-js book reader (epub/mobi/azw3/fb2/fb2.zip) + universal extracted-text fallback + renderRich upgrade (progressive, with scheduler fallback §3.5) + citation `chapter` plumbing (**ls-app `Citation` + to_citation + artifacts Source + frontend `Src`**, §5.4) | L | cite-jump metric green on the **two-bucket corpus defined in §5.5** (chaptered re-indexed epubs ≥80% direct; legacy chapterless ≥50% located, no stall); no stall on multi-MB books (§5.2); pre-upgrade messages-table compat test green |
| **M4** Office ingest + display | .docx .rtf .odt (pure Rust) + mammoth.js display + sanitized-HTML reader kind | M | RU cp1251 rtf fixture green; GPU missing-dep run leaves no success stamps |
| **M5** Best-effort tier | .doc (textutil), .pages (Preview.pdf ladder), .webarchive, .djvu/.chm (external-binary probe) | M | lockstep test still green (all M5 exts have GPU directed skips); pipeline-scoped skip test green (§2.8); **PATH-shim retry test green (install converter → past skip retried, §2.8/§7)** |

Order: M0a → (M0b ∥ M1-CPU dev) → M1 release → M2 → M3 → M4 → M5. M3 can start in parallel with M2 (frontend vs crates). **Honest sizing note:** M0a+M0b together are realistically **2.5–4 weeks solo** — they refactor both pipelines' pre-filters, rewrite the GPU batch contract, add two migrations plus an in-app seed with UI surface, and build a real-library snapshot fixture mechanism. The split exists so plain-format CPU ingest is never hostage to the sidecar rewrite; it does not make the total smaller.

---

## 2. M0a + M0b — Format plumbing, identity guards, state schema, GPU outcome channel, test harness

Foundation that de-risks everything after it. No user-visible format support yet, but these two milestones carry the highest-stakes items in the roadmap. Section-to-milestone mapping (numbering kept stable for cross-references):

| M0a (blocks M1) | M0b (blocks GPU legs of M1, all skip machinery) |
|---|---|
| §2.1.0 ground truth · §2.1.1 seed + sentinel hardening · §2.1.2 guards + row-returning lookups · §2.1.3 bootstrap check · §2.1.4 persistence audit · §2.1.5 harness + snapshot fixture · acceptance test | §2.2 `ext_of` · §2.3 `Format` enum · §2.4 import hard-error · §2.5 GPU stamping, `--caps`, codegen, lockstep test, `gpu_device` · §2.6 repair/orphans · §2.7 copy · §2.8 `skip_state` · §2.9 re-scope note · §2.10 sidecar |

### 2.1 Legacy-book duplication guard — **named exit criterion, blocks M1; re-run blocks M2** (M0a)

**Who this protects:** every store book indexed under a non-`stable_book_id` scheme — the 231 md books, the **69 epubs**, and any legacy pdfs from the same Python import (~300 books total).

#### 2.1.0 Step 0 — establish ground truth before writing code (blocks the rest of M0a)

`ls-cli` already ships `run_backfill_state` (crates/ls-cli/src/main.rs:156–190), which walks `store.book_paths()` and writes `set_book_state_ver(collection, legacy_book_id, fp, csig, 0)` — exactly the seeding this section needs, and it *may already have been run* on the imported collections. The dev machine hosting this repo has no Studio data dir (verified), so the query runs on the machine where Studio's `app.db` lives:

```sql
-- per collection: how many books have state, how many are stamped legacy (ver 0)?
SELECT collection_id, COUNT(*), SUM(chunker_ver = 0) FROM book_state GROUP BY collection_id;
-- how many rows carry the failure sentinel (see §2.1.1 purge):
SELECT COUNT(*) FROM book_state WHERE content_sig = 'missing' OR fingerprint = 'missing';
-- compare against the store's distinct book count per collection (ls-cli or a debug command)
```

- **Branch A — backfill was run** (book_state row count ≈ store book count, `chunker_ver=0` rows present): the "legacy books have no book_state" premise is **false**. They are protected against re-embedding already — but the live first-run failure mode becomes the **stage-2/stage-4 mass remap**: the reverse lookups will hit under the legacy id with `old_id != new_id` for ~300 *unmoved* files and trigger ~300 bulk LanceDB row rewrites in the 302k-row table. **Priority order: path-equality short-circuit (§2.1.2) first; seeding (§2.1.1) becomes a verification no-op — but the Branch-A migration still runs the `"missing"`-row purge (§2.1.1), since `run_backfill_state` seeded sentinel rows for any file it couldn't open.**
- **Branch B — backfill never ran**: legacy books re-embed on the first post-M1 run unless seeded. Priority order: seeding and short-circuit both required; ship together.

Either branch, the short-circuit and the sentinel hardening ship — record the query output in the M0a PR.

#### 2.1.1 Seed `book_state` — by porting `run_backfill_state`, with two deliberate behavior changes

Move the backfill body into `ls-app` as a shared function (`ls_app::backfill_book_state(collection)`), keep the CLI subcommand as a thin wrapper, and call it from a one-time in-app migration on first start after upgrade — for every collection, for every store `(book_id, source_path)` pair lacking a `book_state` row. Semantics:

- Written **under the legacy book_id** (protection flows through the **stage-2 fingerprint reverse lookup**, not stage-1 — the stage-1 path-derived id will never equal the legacy id, and that is fine).
- **`chunker_ver = 0` ("legacy") — deliberate and kept.** This feeds the existing re-index nudge (store.rs:415, :522): after seeding, ~300 books will surface as "indexed with an older chunking scheme", which is *true*. Release notes state this. The §4.2 per-book re-index action is the sanctioned way to clear the nudge (it stamps `CURRENT_CHUNKER_VER`); the nudge must never trigger an automatic re-embed (it doesn't today — verify and add a test).
- **`source_path` recorded** in the new column (§2.8 schema) — this is what powers §2.1.2 without store lookups.
- **Behavior change #1 (vs the CLI backfill): unreadable files are NOT seeded.** The existing `run_backfill_state` writes the row regardless of open failure (main.rs:178–186), leaving `content_sig = "missing"`. The ported function skips any file whose fingerprint or content signature comes back as the `"missing"` sentinel, counting and reporting it (the existing `missing` counter) instead — see "unseedable residue" below. This is a deliberate divergence from the CLI code, not a faithful port.
- **Behavior change #2: sentinel hardening at all FOUR matching sites + one migration purge.** The corruption scenario without it: a discovered candidate whose open fails (permissions, Dropbox-dehydrated placeholder) gets `csig = "missing"`; stage-4 matches an arbitrary previously-seeded `"missing"` row; `old_id != new_id` → `remap_book` re-points an *unrelated* legacy book's chunks to the wrong `source_path` and deletes its state. And within a single run there is a fourth site: `fast_index_collection`'s in-run `seen_content` map (src-tauri lib.rs ~:535) — the first unreadable candidate inserts key `"missing"`, and every *subsequent* unreadable file in the same run matches it and is preskipped with **no event and no state** (a silent-skip class violating invariant #5 that the persistent-lookup fixes alone do not touch). Fixes, all shipped in M0a:
  1. `book_id_for_content` AND the fingerprint reverse lookup treat the literal `"missing"` (and `''`) as no-match — never return a row for it.
  2. `set_book_state` (and the future skip_state writer, §2.8) refuse to persist any row whose fingerprint or content_sig is `"missing"` — an unreadable file produces no state.
  3. **The in-run `seen_content` map never has `"missing"` (or `''`) inserted as a key and never matches candidates against those values.** The §2.1.5 `plan_index_run` refactor *carries this map over* — the rule is implemented inside `plan_index_run` (each unreadable file falls through to its own "unreadable" skip event; it never joins content-dedup at all) and pinned by the extended scenario (d) below.
  4. The migration (both branches, but especially Branch A) **purges pre-existing rows** where `content_sig = 'missing'` or `fingerprint = 'missing'` (count recorded in the migration log; the affected books simply get re-evaluated on the next run, correctly).
  - **Dehydrated-file fixture test — TWO unreadable candidates, not one:** two distinct files whose opens fail (permissions/FIFO fixtures) in the same run, both pipelines → assert no `book_state`/`skip_state` rows written, no remap of any existing book, and **each file emits its own once-per-run `Skipped` (unreadable) event** — the second must not be silently preskipped via `seen_content`.

#### 2.1.2 Old≠new guards — **both** the fingerprint stage AND the content-signature stage, on **both** pipelines

Two verified remap sites exist per pipeline: stage-2 fingerprint (reverse-lookup hit, `old_id != new_id`) and stage-4 content-signature (`book_id_for_content` hit, `old_id != new_id` — service.rs:272–287 and src-tauri/src/lib.rs:538–546 both call `remap_book` + `delete_book_state`). Guarding only stage-2 is not enough: this library lives in Dropbox, which is known to re-stamp/dehydrate files (see the venv-hang incident) — if a legacy file's mtime churns, the seeded fingerprint goes stale, stage-2 *misses*, and stage-4 *hits* with old≠new, producing exactly the ~300-book bulk LanceDB rewrite this section exists to prevent.

**Prerequisite — widen the reverse lookups to return rows, not ids (explicit M0a work item).** `book_id_for_fingerprint` today returns only the book_id (ls-app/store.rs:354); every guard below needs the matched ROW. M0a replaces both lookups with row-returning variants — one query, no per-hit follow-ups, no recomputation against the wrong thing at any of the four sites:

```rust
// ls-app store — replaces book_id_for_fingerprint / book_id_for_content:
pub struct BookStateHit { pub book_id: String, pub source_path: String,
                          pub content_sig: String, pub fingerprint: String }
pub fn book_state_for_fingerprint(&self, coll: &str, fp: &str) -> Option<BookStateHit>;
pub fn book_state_for_content(&self, coll: &str, csig: &str) -> Option<BookStateHit>;
```

Both apply fix #1 of §2.1.1 (sentinel → `None`) internally. The old id-returning functions are deleted, not kept alongside — a single lookup shape at all call sites.

Changes, applied identically at **all four sites** (2 stages × 2 pipelines):

- **Path-equality short-circuit.** When a reverse lookup hits with `old_id != new_id`, compare the candidate path against `hit.source_path` (populated by the seed and by every future `set_book_state`); for pre-existing rows where the column is empty (`''` default), fall back to a **single-book lance lookup** of `old_id`'s `source_path` (the store exposes book paths already; add a per-id variant) and backfill the column on the spot. If the paths canonicalize equal → **plain skip, zero store writes** (id-*scheme* difference or metadata churn, not a move) — with one cheap follow-up on the stage-4 variant: **refresh the fingerprint under the existing id** (`set_book_state` with the current size:mtime, same book_id, same csig). Without the refresh, every subsequent run re-misses stage-2 and re-reads 512 KiB per churned file to re-hit stage-4; with it, the next run short-circuits at stage-2.
- **Fingerprint-collision identity confirmation (stage-2 only).** The fingerprint lookup is `LIMIT 1` over a size:mtime key. Today's population (hundreds of large PDFs) makes collisions negligible; M1 floods collections with thousands of small batch-exported text files — two distinct notes with equal byte size and same-second mtime (the 212 ByteByteGo lessons are exactly this shape) would collide: first indexes normally; second hits stage-2 with old≠new and *genuinely different paths* → `remap_book` re-points book A's rows to file B's path, deletes A's state, and the two files ping-pong remap on every subsequent run. Fix: in the stage-2 old≠new branch, **when the paths differ, compute the candidate's `content_signature` and compare it against `hit.content_sig`** (a 512 KiB read, paid only on the rare collision path; the stored value comes back with the hit — no second query). csig match → genuine move → remap as today. csig mismatch → *different file that happens to share a fingerprint* → fall through to normal indexing (no remap, no state touch for the other book). Stage-4 needs no such confirmation — a content-signature hit *is* the identity proof.
- Only when paths genuinely differ **and** identity is confirmed does `remap_book` run — the existing, correct moved-file behavior.

#### 2.1.3 Source-path membership in the bootstrap check (belt and braces)

Extend the "already in index" stage on both paths to also match the candidate's canonicalized path against the store's distinct `source_path` set (loaded once per run — `book_catalog` already reads it). A path match short-circuits exactly like an id match. Covers seed failures and any book_state row loss.

#### 2.1.4 Persistence audit (M0a checklist item)

Confirm nothing persisted keys on `book_id`, since moved legacy files will legitimately remap ids and §4.2 re-indexing mints new ids. Saved citations key on `source_path` — confirmed (and the persisted `Citation` struct being touched in M3 lives in `ls-app/src/types.rs:46`, see §5.4 — audit it here so M3's serde change lands on a known-safe struct). Verify artifacts export (carries source_path, page — confirmed safe) **and notebook/session persistence** before shipping; any book_id-keyed persistence found must be migrated to source_path or made remap-aware in M0a.

#### 2.1.5 Headless test harness — the refactor the acceptance test requires (explicit M0a work item, multi-day)

The acceptance test below is not executable against today's code: `fast_index_collection` is a `#[tauri::command]` taking `State<AppState>` + `WebviewWindow` with the entire 4-stage dedup pre-filter inlined in the command body (src-tauri/src/lib.rs:497–550) — there is no headless entry point, and GPU-path "embed calls" happen inside a spawned Python process. M0a therefore:

- **Extracts the pre-filter into a pure, testable ls-app function**, shared by both the command and `index_collection`'s equivalent stages:

  ```rust
  // ls-app — no tauri types, no windows, no python:
  pub fn plan_index_run(candidates: &[PathBuf], db: &Db,
                        indexed_ids: &HashSet<String>, indexed_paths: &HashSet<String>,
                        caps: &PipelineCaps)
      -> IndexPlan { to_embed: Vec<PathBuf>,
                     preskips: Vec<(PathBuf, SkipReason)>,
                     remaps: Vec<RemapAction>,
                     state_refreshes: Vec<StateRefresh> }
  ```

  `fast_index_collection` becomes a thin wrapper: plan → spawn batches over `plan.to_embed` → outcome-aware commit (§2.10). All §2.1.1–2.1.3 guard logic — **including the in-run `seen_content` map with its sentinel exclusion rule** — lives inside `plan_index_run` (or helpers it calls), so every scenario below is a plain Rust test against a fixture Db + fixture tree.
- **Builds the real-library snapshot fixture mechanism** (called out as its own work item because it is nontrivial): a captured `app.db` fixture + a representative slice of the 302k-row Lance store (a few hundred rows spanning legacy md/epub/pdf ids) + a script that regenerates the slice from the live machine. The acceptance test runs against this snapshot; the *live* re-run (row-count audit on the real store) is a release-gate manual step, not CI.
- **Defines the CPU-side "0 embeds" assertion mechanism:** `index_collection` embeds through the injected embedder — tests wrap it in a **counting stub** (increments per `embed` call; the legacy-guard tests assert the counter is 0) plus store row-count invariance. **GPU-side "0 embeds" assertion:** assert `plan.to_embed` is empty for every legacy `source_path` (the plan boundary is where embedding is decided; the Python process never sees a file the plan excluded) plus store row-count invariance on the live re-run.

#### Unseedable residue — moved/missing/unreadable legacy files

The seed covers only books whose stored `source_path` exists and is readable. The migration **reports the count and list of unseedable legacy books** to `index-log` and Settings → Maintenance (this now also includes any books whose pre-existing `"missing"`-sentinel rows were purged by §2.1.1). If such a file later reappears at a new path, it re-embeds under a new id and its legacy rows become stale duplicate retrieval hits — accepted residual risk, mitigated by the §2.6 orphan-row report (detect + prune). Stated in §10.

#### Acceptance test (exit criterion, M0a; re-run at M2) — four scenarios, all via §2.1.5's harness

Against the snapshot of the real library (fixture Db + real path tree, plus synthetic fixtures for (b)–(d)):

- **(a) Baseline post-upgrade run:** `index_collection` (counting embedder) and `plan_index_run` (GPU plan) with all M1 **and M2** extensions enabled → for **every legacy store book**: 0 embed calls / absent from `to_embed`; total store row count unchanged; no `source_path` appears under two distinct book_ids afterward. The test **tolerates metadata remap writes** (id/path updates for genuinely moved files are correct behavior) — it must NOT assert "no book_id changes". Unmoved legacy books must additionally produce **zero store writes** (verifies §2.1.2).
- **(b) Dropbox mtime-churn:** a legacy file with churned mtime (seeded fp stale, csig matching, path unmoved) → stage-2 misses, stage-4 path-equality hit → zero store writes, zero embeds, **fingerprint refreshed under the existing id**; a second run must short-circuit at stage-2 (no 512 KiB csig read — assert via a counting csig wrapper).
- **(c) Fingerprint collision:** two distinct files with equal size and same-second mtime, different content → both index under distinct ids; a second run leaves both untouched (no remap ping-pong, asserted across two consecutive runs).
- **(d) Dehydrated/unreadable candidates — two of them:** both opens fail → no state rows, no remap of any existing book, and **two distinct `Skipped` (unreadable) events** — one per file — proving the `seen_content` sentinel exclusion (§2.1.1 fix #3).

All assertions keyed on `source_path` per invariant #11.

### 2.2 Shared `ext_of` and the five call sites (M0b)

One rule: `ext_of(name)` lowercases the filename and longest-matches against `SUPPORTED` (so `.fb2.zip` beats `.zip`; unknown → `None`). The canonical list lives in `ls-core` next to `Format`; other layers derive from it:

| # | Call site | Change |
|---|---|---|
| 1 | `discover.rs is_supported` (crates/ls-app) | use `ls_core::ext_of` |
| 2 | `ls-extract::extract` dispatcher (lib.rs:87–109) | dispatch on `ext_of`, not `Path::extension` |
| 3 | `gpu_embed.py` suffix map | python `ext_of` mirroring the same list (`path.suffix` alone would call `.fb2.zip` "zip"); asserted by the lockstep test (§2.5) |
| 4 | `App.tsx openSource` ext→kind map (:943) | frontend `extOf(name)` util consuming the **generated** `supportedExts.ts` (§2.5) — last-dot split would route `.fb2.zip` to kind "other", silently bypassing the M3 BookReader |
| 5 | `supported_exts_display()` / error-string builder | built from the same canonical list |

**Fixture test per site** with `.fb2.zip` (and a `.tar.gz`-style negative case).

### 2.3 `Format` enum (crates/ls-core/src/lib.rs) (M0b)
- Extend `Format` (currently `Pdf | Epub | Mobi`) with one variant per *family*: `Md, Txt, Html, Docx, Rtf, Odt, Doc, Fb2, Pages, Webarchive, Djvu, Xps`. Keep `Copy`, `rename_all = "lowercase"`.
- `from_ext` maps aliases: `markdown→Md`, `text→Txt`, `rst|adoc|org|tex→Txt`, `htm→Html`, `azw3→Mobi`, `fb2.zip→Fb2` (via `ext_of`). `as_str` returns the family name.
- Rationale for family granularity: the store `format` column drives only citation shape (`p.N` vs `Ch. X` vs `~loc`) and diagnostics; readers key off extension. Don't over-model.
- **Dependency check, not assertion:** run `cargo tree -i quick-xml` / `-i zip` once and record in the PR whether they're already transitive; that settles the fb2/docx/odt dep story before M2/M4 estimates are trusted.

### 2.4 Store import hard-error + batch-failure containment (crates/ls-index/src/store.rs:542; src-tauri lib.rs) (M0b)
- `chunks_with_vectors`: replace `Format::from_ext(s).unwrap_or(Format::Pdf)` with a hard error that names **the unknown format string AND the offending file/book_id** (`"parquet row for '<source_path>' (book <id>) has unknown format '<s>' — gpu_embed.py and ls-core Format are out of sync"`). This is the exact mechanism that mislabeled the legacy books; we control both producers, so fail loudly.
- **Named contract change, not a wish:** today an `import_parquet` failure propagates through the `?` at src-tauri lib.rs:609 and **aborts the entire fast-index run**. M0b restructures `run_py_batch`'s Result contract: it returns `Ok(BatchOutcome)` (per-file outcomes per §2.10 + parquet path) or `Err(BatchError)`; the caller converts a batch error into: rows discarded, **no state committed for any file in the batch** (fingerprints and skip records both), error logged to `index-log` with the message above, run **continues with subsequent batches**. This is a deliberate change to the error flow, implemented and tested in M0b — not discovered mid-M2.
- **Read path unchanged:** `rows_from_batch` (:576) keeps tolerating unknown strings as `Option::None` — the hard error applies to *import only*, so a bad row that ever lands can never make the store unreadable.
- Tests: import test asserting the error message contains format string + book_id; a fast-index integration test with one poisoned batch asserting later batches still commit and no state was written for the poisoned batch's files; a read test with an unknown-format row asserting `format: None`.

### 2.5 gpu_embed.py restructure: stamping fix, stdlib `--caps` prologue, `gpu_device` plumbing, lockstep CI test (M0b)

- Replace hardcoded `"format": "pdf"` with the `ext_of`-driven family map mirroring `Format::from_ext`.
- gpu_embed.py declares two module-level literals: `HANDLED_EXTS = {...}` and `DIRECTED_SKIPS = {ext: reason, ...}`. A file whose ext is in `DIRECTED_SKIPS` never reaches `fitz.open()`; its outcome travels via the sidecar (§2.10). Additionally, skips discovered **only at runtime inside Python** (e.g. a guarded `import docx` failing) also surface as sidecar skip outcomes — the Rust pre-filter cannot know about them, and per §2.10 they are recorded as skip_state, never as success.
- **`--caps` mode — stdlib-only prologue, mandatory restructure.** Today the script imports `torch`/`sentence_transformers`/`transformers`/`fitz` at module level (gpu_embed.py:29–33), so a naive `python gpu_embed.py --caps` would pay a multi-second torch import per fast-index run *and* fail entirely if torch is broken — even though the probe needs nothing from it. The restructure, an explicit M0b work item:
  - Module top becomes stdlib-only (`sys`, `json`, `argparse`, `importlib.util`, `pathlib`). All heavy imports move inside `main()` (lazy), executed only on the embed path.
  - `--caps` is handled in the prologue and prints one JSON line — `{script_version, handled_exts, directed_skips, optional_deps_available, device_flag_supported: true}` — then exits. Optional-dep availability is probed with `importlib.util.find_spec` (no import execution, so a broken `python-docx` can't crash the probe either).
  - **Acceptance test:** `--caps` completes in < 300 ms cold in the fixture venv (asserted with a generous CI multiplier) and succeeds in a venv where `torch` is deliberately broken.
  - **Byte-drift note:** the restructure changes the embedded script's bytes. Managed users get the new script via the existing self-refresh (`include_str!` + rewrite at `<data_dir>/scripts/gpu_embed.py`, src-tauri lib.rs:466–469). Custom `indexer_script` users keep their own file — their script predates both `--caps` and the sidecar; the fallback for them is a **script-bytes hash as caps_ver** plus a logged version-mismatch note (§2.8), and batch-level failure handling for the missing sidecar (§2.10). The version banner line in script stderr (surfaced in index-log) names the expected script version so drift is visible, not inferred.
- **`gpu_device` setting — the missing plumbing.** The script already honors `--device` (gpu_embed.py:136, default `"mps"`), but `run_py_batch` never passes it and no setting carries it. M0b adds a `gpu_device` setting (Settings → Indexing; default `"mps"`; free-text `mps|cuda|cpu`), `run_py_batch` appends `--device <value>` to every spawn, and **the configured value is folded into the GPU caps_ver** (§2.8) — so switching device (e.g. configuring `cuda` on a Linux box) changes the caps hash and automatically retries past xps/mobi skips, which §8 promises. This is the one-line scope §8's Linux story depends on; without it "documents the setting" would document nothing.
- **Lockstep CI test (general rule, applies to every future milestone):** a Rust unit test iterates `SUPPORTED` and asserts, for each ext: (a) `Format::from_ext` maps it; (b) `ls-extract::extract` has an arm or a directed CPU skip; (c) the `include_str!`-embedded GPU script's `HANDLED_EXTS ∪ DIRECTED_SKIPS.keys()` (parsed from the two literals) contains it; **(d) round-trip: for every `Format` variant `v`, `Format::from_ext(v.as_str()) == Some(v)`** — the read path (§2.4) silently drops `format` to `None` on any as_str/from_ext asymmetry, so the asymmetry itself must be impossible to ship.
- **Frontend leg via codegen, not vitest** (the frontend has no test infra today): a Rust generator (`cargo xtask gen-exts`) emits `frontend/src/generated/supportedExts.ts` containing the canonical ext list and ext→reader-kind-family map; `extOf` and the `openSource` kind map consume it. A Rust test regenerates the file in-memory and asserts the checked-in copy is byte-identical — stale codegen fails CI with no JS runner involved.

### 2.6 Legacy stamp repair + orphan-row report (OPTIONAL, ship dark) (M0b)
- **`repair_format_stamps`:** for each book whose `source_path` extension (via `ext_of`) disagrees with the stored `format`, issue a **one-column in-place LanceDB update** using the same primitive `remap_book` already uses (store.rs:221–237): `update().only_if("book_id = '<X>'").column("format", "'md'")`. ~10 lines. No vector handling, no delete-then-add crash window, no re-embedding. FTS untouched. Purely cosmetic (citation shape) since readers ignore the column.
- **`report_orphans`:** lists store books whose `source_path` no longer exists on disk (superset that includes the §2.1 unseedable legacy residue). Shown in Settings → Maintenance alongside the seed migration's unseedable count, with an explicit per-book prune action (`delete_book`) — user-initiated only, never automatic.

### 2.7 Copy & error strings (M0b)
- src-tauri/lib.rs:237 and :482: `"no PDF files found…"` → `"no supported files (pdf, md, txt, …) found under the collection's source paths"` from `supported_exts_display()`.
- App.tsx:965 dialog title → `"Choose a folder of books & documents"`; App.tsx:2699 "other"-kind copy updated per-milestone.

### 2.8 State schema: `skip_state` table + `book_state.source_path` column — designed here, before any code (schema column M0a; table M0b)

`book_state` (crates/ls-app/src/store.rs, PK `(collection_id, book_id)`, columns `fingerprint/content_sig/chunker_ver`) cannot absorb pipeline-scoped skip records: SQLite cannot alter a PK, the repo's migration helper (`alter_ignore_duplicate`) only supports ADD COLUMN, and — decisive — the reverse lookups scan `book_state`, so a skip record living there would be found by stage-2, "remapped" as a book with zero store rows, and rewritten as a success-shaped fingerprint — silently converting *skipped* into *indexed*. Therefore:

**Schema (two idempotent migrations):**

```sql
-- 1. ADD COLUMN via alter_ignore_duplicate (supported migration shape) — ships in M0a:
ALTER TABLE book_state ADD COLUMN source_path TEXT NOT NULL DEFAULT '';
-- populated by the §2.1.1 seed, by every future set_book_state, and lazily
-- backfilled by the §2.1.2 fallback lookup. Powers path-equality short-circuit
-- and §4.2 source_path-keyed state clearing.

-- 2. New table via CREATE TABLE IF NOT EXISTS (no PK surgery needed) — ships in M0b:
CREATE TABLE IF NOT EXISTS skip_state (
    collection_id TEXT NOT NULL,
    source_path   TEXT NOT NULL,           -- canonicalized path of the original file: the IDENTITY key
    pipeline      TEXT NOT NULL,           -- 'cpu' | 'gpu'
    fingerprint   TEXT NOT NULL,           -- size:mtime at skip time: a STALENESS check, never identity; never 'missing'
    reason        TEXT NOT NULL,           -- exact user-facing skip reason
    caps_ver      TEXT NOT NULL,           -- per-pipeline RUN-TIME capabilities hash, see below
    created_at    INTEGER NOT NULL DEFAULT (strftime('%s','now')),
    PRIMARY KEY (collection_id, source_path, pipeline)
);
```

**Why path-keyed, spelled out (this is a correctness requirement, not a style choice):** the plan itself proves size:mtime fingerprints are not identity — §2.1.2 exists because M1 floods collections with thousands of small same-size/same-second files where fingerprints collide between DISTINCT files. A fingerprint-keyed skip table would let file A's skip row ("no extractable text", missing dep) silently suppress file B — a different file with a colliding fingerprint — from ever being indexed (exactly the silent-skip class invariant #5 forbids), and conversely B's later success would erase A's legitimate skip record. Keying on `(collection_id, source_path, pipeline)` makes a skip record about *one file at one path*, full stop. Fingerprint rides along purely to detect "the file at this path has changed since we skipped it".

**Semantics:**
- **Stage interaction:** each pipeline consults `skip_state` by the candidate's canonicalized path (a new "stage 0.5", before the book_state stages). A hit short-circuits **only when BOTH the stored fingerprint matches the candidate's current fingerprint AND the stored `caps_ver` matches the run's** — silent short-circuit, no event, no re-emitted skip. Fingerprint mismatch (file changed) or stale `caps_ver` (capabilities changed) or no row → proceed normally. Because skip records never enter `book_state`, stages 1–4 and the reverse lookups are untouched — no filtering-by-status logic anywhere. A *moved* file simply has no row at its new path (fresh attempt — correct); the old-path row is orphaned and GC'd opportunistically (a cheap sweep deleting rows whose path is no longer under any collection source, run at index start). No stale-path ambiguity remains.
- **`caps_ver` is computed at run start and includes runtime environment, not just compiled code.** A compile-time-only hash cannot retry external-tool skips: M5's .doc/.djvu/.chm CPU skips depend on **runtime PATH probes** (antiword, soffice, djvutxt, 7z, textutil), and a user who installs antiword after the first run must have the skip cleared — §7's "retried when capabilities or the file change" promise depends on it. Therefore:
  - **CPU caps_ver** = hash of (compile-time `SUPPORTED` + extractor-arm set) ⊕ hash of the **probed external-tool availability set** — a handful of `which` calls (textutil/antiword/soffice/djvutxt/7z) executed once at the start of each index run. Installing or removing a converter changes the hash → every skip stamped under the old hash is retried.
  - **GPU caps_ver** = hash of the one-line JSON from `gpu_embed.py --caps` (§2.5 — stdlib-only prologue, < 300 ms, no torch import) ⊕ hash of the configured **`gpu_device` setting value** (§2.5) — so `pip install python-docx` retries past missing-dep skips, and reconfiguring the device (e.g. `cuda` on Linux) retries past xps/mobi skips. One cheap subprocess spawn per fast-index run. If `--caps` fails (old custom script), fall back to a hash of the script bytes and log a version-mismatch note.
  - A pipeline upgrading its capabilities automatically retries its own past skips; the other pipeline's records are irrelevant to it by key.
- **Writers:** CPU path upserts skip_state directly on any skip — unsupported, directed skip, extract failure, DRM, **and "no extractable text"**. GPU path writes it from the **sidecar outcomes** (§2.10) — the Rust pre-filter writes it only for exts it can pre-classify; runtime-discovered Python skips arrive via the sidecar. **Never written when the file's fingerprint is the `"missing"` sentinel** (invariant #6 — an unreadable file gets an event only, no state, retried every run by construction).
- **"No extractable text" moves to skip_state — a named behavior change.** Today this outcome writes success-shaped `book_state` (service.rs:310–321). Going forward it writes `skip_state (reason = "no extractable text", pipeline-scoped, caps_ver-stamped)` and **no book_state row** — so invariant #5 is true as stated, and an extractor upgrade (new caps_ver) automatically re-attempts books that yielded nothing under the old extractor. **Migration note:** books already recorded this way live in `book_state` indistinguishably from real successes; no purge is attempted — they behave exactly as today (fingerprint-skip until the file's bytes change, at which point the new code path applies). Documented in release notes. **Fixture test:** image-only PDF → skip_state row (cpu, "no extractable text"), zero book_state rows, silenced on rerun, retried on caps_ver bump.
- **Erasure:** a successful index of a file deletes any skip_state rows for that `(collection_id, source_path)` across both pipelines (a plain path-keyed delete — no fingerprint reasoning needed); §4.2 re-index deletes them explicitly by path (step 4); a changed file is caught by the fingerprint-staleness check and its row overwritten by the retry's outcome.
- **What skips DON'T get:** a `book_state` row — with the no-text change above, this now holds for *every* skip class. (Today service.rs records book_state on remap/no-text/success — :278, :313, :357 — and extract failures at :299–300 record nothing; the changes are adding skip_state and relocating the no-text write, both explicit above.)

**Required tests:** CPU-index an .xps (skip_state row, pipeline=cpu) → `fast_index_collection` over the same folder → assert the .xps **is embedded** and the cpu skip row is erased (by path). Reverse-direction: a GPU sidecar skip (simulated missing python-docx) → assert **no book_state row exists** for the file, CPU path still picks it up. Skip silenced on rerun; retried on caps_ver bump; retried when the file's bytes change (fingerprint-staleness fixture). **Skip-collision fixture (mirrors §2.1 scenario (c)):** file A gets a legitimate skip_state row; file B — different content, colliding size:mtime fingerprint, different path — must hit stage 0.5 with **no row for its path** and index normally; B's success must **not** erase A's skip row; A stays silenced on the next run. **PATH-shim retry test:** record a .doc skip with no converter on PATH → shim `antiword` into PATH → next run's CPU caps_ver differs → the .doc is re-attempted and indexes (mirror test on GPU: fixture venv gains `python-docx` → `--caps` hash changes → docx retried; second mirror: change `gpu_device` setting → hash changes → xps skip retried). No-text fixture test (above). Stage-2 safety: a skip_state row must never be returned by the book_state reverse lookups (trivially true — different table — but the test pins the invariant). Orphan GC: a skip row whose path leaves the collection's sources is swept.

### 2.9 Re-scope note
Adding extensions immediately re-scopes every existing collection: the next index run picks up all md/txt already sitting in indexed folders (additive; existing books skip via fingerprint + §2.1 guard). Per-format Finished counts that make this legible ship in **M1**. Optional per-collection extension filter in Settings only if users ask.

### 2.10 GPU per-file outcome channel — sidecar JSON, not stderr prose (M0b)

**The problem (verified):** src-tauri/src/lib.rs:615–629 unconditionally commits success-shaped `book_state` for **every** file in a completed batch — including files gpu_embed.py skipped — and `run_py_batch` returns only aggregate counts. Worse, `parse_py_progress` (:320–347) extracts the skip "path" via `rest.rsplit(' ').next()`, which returns garbage for `skip (error) {path}: {msg}` lines and any path containing spaces. Consequence: a GPU skip (directed or runtime) gets stamped with full success book_state under `ls_extract::stable_book_id` — the **same key the CPU pre-filter checks** (lib.rs:505 / service.rs:218) — silencing the file on the CPU path forever. This is cross-pipeline poisoning in the direction §2.8 alone cannot fix; it must be fixed at the source.

**Design:**
- gpu_embed.py writes a **sidecar outcomes file** next to each parquet: `<parquet>.outcomes.json` — a JSON array **aligned to argv order** (index-keyed, so paths with spaces/colons/unicode are irrelevant):
  ```json
  { "v": 1, "outcomes": [
    {"i": 0, "status": "indexed", "chunks": 412},
    {"i": 1, "status": "skipped", "reason": "docx support: pip install python-docx"},
    {"i": 2, "status": "error",   "reason": "fitz: cannot open"} ] }
  ```
  Written atomically (temp + rename) after the parquet, before exit 0. Every argv file appears exactly once; the schema is versioned.
- **stderr becomes display-only.** `parse_py_progress` keeps driving the live progress UI (the `[i/n]` counter is reliable; the path/reason text is cosmetic) but is **never used for state decisions**. Its broken path extraction is retired from any decision role; optionally fix the display to resolve `[i/n]` → batch file list index.
- `run_py_batch` parses the sidecar and returns `BatchOutcome { outcomes: Vec<FileOutcome>, parquet }` (per the §2.4 contract), and — per §2.5 — now also passes `--device <gpu_device>` on every spawn. **Missing, unparseable, or incomplete sidecar → the whole batch is treated as failed** (no state committed, retried next run) — this is also the compatibility story for custom `indexer_script` users whose script predates the sidecar: batch-level fallback, never fabricated per-file success.
- **Outcome-aware per-batch commit** replaces lib.rs:615–629: `indexed` → `set_book_state` (full fingerprint + content_sig + `CURRENT_CHUNKER_VER` + `source_path`; refused if either value is the `"missing"` sentinel, per invariant #6) **plus a path-keyed delete of any skip_state rows for that file (both pipelines — §2.8 erasure)**; `skipped` → skip_state upsert keyed by the file's path (pipeline=gpu, reason from the sidecar, the run's GPU caps_ver, current fingerprint as the staleness column); `error` → **no state at all** (transient; retried next run). Import runs before commit, so a §2.4 import failure still voids the whole batch.
- **Per-format Finished counts (§3.4) ride on this channel:** GPU-side per-ext indexed/skipped tallies are derived from sidecar outcomes joined with the batch file lists; the counts land in M1 as a small delta on top of this M0b work.

**Tests:** sidecar round-trip (mixed indexed/skipped/error batch → correct book_state/skip_state/no-state per file); path-with-spaces and RU-path fixtures; truncated sidecar → batch failed, zero state; legacy script without sidecar → batch failed with a directed log message naming the script-version mismatch; poisoned-format parquet (§2.4) → batch voided including its sidecar outcomes; indexed outcome erases a pre-existing cpu skip row for the same path.

**Tests (M0a+M0b recap):** `ext_of` unit tests incl. `.fb2.zip` at all five sites; `from_ext` alias + **round-trip** tests; import hard-error trio + batch-continuation (§2.4); lockstep CI test incl. codegen-freshness leg (§2.5); `--caps` cold-latency + broken-torch tests (§2.5); legacy-book acceptance test, four scenarios via the §2.1.5 harness (baseline / mtime-churn / fp-collision / two-unreadable-files); sentinel-hardening tests (row-returning lookups reject `"missing"`; writers refuse it; `seen_content` excludes it; migration purge); path-equality short-circuit tests at **all four remap sites** incl. the empty-`source_path`-column fallback branch and the stage-4 fingerprint refresh; moved-legacy-file test (csig-confirmed remap fires, no re-embed); skip_state suite incl. cross-pipeline non-interference, **skip-collision fixture**, no-text relocation, PATH-shim + device-change retries, orphan GC, and stage-2 safety (§2.8); sidecar suite (§2.10); persistence audit recorded; `cargo tree` dep check recorded; ground-truth query output recorded.

---

## 3. M1 — Plain-format ingest (M)

### 3.1 Formats & extraction (CPU path, crates/ls-extract)

All pure Rust. New module per family; the dispatcher gains arms. All emit `BookDoc` with `Block{text, chapter, page: None}` — chunker/embedder/store are already format-agnostic. `MIN_BOOK_CHARS = 200` gate unchanged (its "no extractable text" outcome now lands in skip_state per §2.8).

| Ext | Parser | Chapter mapping |
|---|---|---|
| .md/.markdown | UTF-8 read (encoding-sniff fallback); `^#{1,6} ` heading scanner (~40 LoC, zero deps; matches ebook-kb behavior) | **H1–H2 only** become `chapter`; H3+ stay in-body (see §3.2) |
| .txt/.text | read + encoding sniff: `encoding_rs` + `chardetng` (RU windows-1251 txt) | `chapter: None`; single Block |
| .rst | txt + underline-heading sniff (`====`/`----`) | top-level underlined titles → `chapter` |
| .adoc | txt + `= `/`== ` prefixes | `chapter` |
| .org | txt + `*`/`**` prefixes | `chapter` |
| .tex | txt + `\chapter{}`/`\section{}` sniff, strip `%` comments | `chapter` |
| .ipynb | `serde_json`: markdown cells as-is + code cells fenced | md-cell H1–H2 → `chapter` |
| .html/.htm | `scraper`/`dom_query` + `html2text`; strip script/style/nav | `<h1>/<h2>` → `chapter` |

**3.2 Heading fragmentation & Index-tab flood — designed here, not discovered in production.** Two verified consequences of heading-per-Block: (1) `chunk_book` never crosses chapter groups and `merge_short_tail` only merges *within* a group, so a section shorter than `min_tokens` mints exactly one sub-100-token chunk — heading-dense notes (the 212 ByteByteGo lessons are precisely this shape) would mint thousands of tiny low-quality chunks; (2) `book_catalog` creates a catalog row per distinct non-empty `chapter`, so every heading of every note floods the Index tab. Mitigations, both at **extraction time** (before chunking, so the no-cross-chapter rule is never violated):
- **Section floor:** merge any heading section below a token floor (~120 tokens) into the *previous* section, keeping the earlier section's `chapter` label and inlining the small heading as body text.
- **Depth cap:** only H1–H2 populate `chapter` for the whole txt-family; deeper headings remain body text.
- **Fixture test:** a heading-dense md fixture (modeled on a ByteByteGo lesson) asserting no emitted chunk is below `min_tokens` and catalog entries ≤ H2 count. Live check against the actual 212-lesson corpus before release.

**Fingerprint-collision exposure note:** M1 is the milestone that expands the size:mtime-collision population ~10× (thousands of small batch-exported files). The §2.1.2 csig-confirmation guard and the §2.8 path-keyed skip table ship in M0a/M0b precisely for this; M1 re-runs scenario (c) of the §2.1 acceptance test — and the §2.8 skip-collision fixture — with a batch-exported-notes fixture (many same-second-mtime files) as part of its exit criteria.

### 3.3 GPU path (gpu_embed.py) — requires M0b
- `HANDLED_EXTS` gains the text family: stdlib-only branch — read bytes, decode (utf-8 strict → cp1251 → latin-1; `charset-normalizer` if importable), heading scan mirroring the Rust rules incl. section floor + H1–H2 cap, tokenizer-window chunks per section, stamp `chapter` (feeds the Index tab), `page: -1`. Format stamped via the M0b map. Outcomes via the §2.10 sidecar like everything else.
- `.html`: either a stdlib `html.parser` strip branch or `DIRECTED_SKIPS["html"] = "html: handled by standard indexing"` in the first cut — the lockstep test forces one or the other explicitly.

### 3.4 Per-format Finished counts
- Extend the `Finished` IndexEvent with a `by_format: {ext: {indexed, skipped}}` map (additive, serde-default so old frontends tolerate it). CPU side: counted in `index_collection`'s existing per-file loop. **GPU side: derived from the §2.10 sidecar outcomes joined to batch file lists** (per-file status is otherwise unknowable in Rust — this is why the sidecar is an M0b prerequisite). Frontend: summary line "indexed 12 md, 3 txt · skipped 480 pdf (up-to-date)". This is what makes the first post-upgrade re-scope run legible.

### 3.5 Display (mostly exists) — text cap raised carefully, not naively
- `ReaderKind "md"` → `"text"`; ext→kind map (via `extOf` + generated `supportedExts.ts`): `md|markdown|txt|text|rst|adoc|org|tex → "text"`. `.ipynb`/`.html` stay `"other"` until M3/M4.
- `read_source_text` (src-tauri lib.rs:893–904): read-bytes + `chardetng`/`encoding_rs` decode (RU cp1251 txt currently errors on `read_to_string`); cap raised 4 → **8 MiB** — not 16: the same main-thread-jank rationale that caps HTML at 4 MiB (§6.2) applies to a giant synchronous `renderRich` DOM build, so the raise is paired with progressive rendering:
  - **Progressive rendering above ~2 MiB**, with an explicit scheduler fallback: slices of ~512 KiB of paragraphs are appended per idle callback via `requestIdleCallback` **when available, else a `setTimeout(0)`/rAF-chained scheduler** — rIC is a recent WebKit addition and WebKitGTK on the Linux deb may lack it depending on distro; the fallback is one line and is named here so the Linux smoke test doesn't discover a frozen renderer. Files ≤ 2 MiB render exactly as today.
  - **Cite-jump under slicing:** locate the normalized `citeText` prefix in the *raw text* first (cheap string search), fast-forward rendering of slices up to that offset, then run the existing TreeWalker match + `cite-flash` within the rendered DOM. For ≤ 2 MiB files the current single-pass TreeWalker path is unchanged.
  - **Over 8 MiB:** error panel offers "Open in default app" and "Show first 8 MB" — a truncated read with a visible banner, where the backend **truncates on a UTF-8 char boundary** (floor to the nearest `char` boundary, à la `s.floor_char_boundary`, so RU text near the cut can't produce an invalid-UTF-8 read error or a mangled trailing character); cite-jump attempted only if the match offset falls inside the window.
- **renderRich upgrade (S):** `#`-headings (h1–h4), fenced code blocks (monospace, no highlighting), `[text](url)` as non-clickable styled text (offline guarantee). Tables deferred.

### 3.6 Testing (M1)
- Fixtures per format under `crates/ls-extract/tests/fixtures/`: md with nested headings + RU; **heading-dense md (§3.2 test)**; cp1251 txt (assert decode + char-based loc offsets per chunk.rs:344 convention); rst/adoc/org/tex; ipynb; html with nav junk.
- Chunker integration: chunks never cross H1–H2 sections; `chapter` populated; `book_catalog` returns heading entries; **no chunk below min_tokens**.
- Golden set: 3–5 Q/A pairs on md + RU txt fixtures; citation shape `Ch. <heading>`.
- **Exit criterion (regression):** the §2.1 acceptance harness re-run over the real-library snapshot with M1 extensions → **0 new rows and 0 embeds for all legacy books** (counting embedder + `plan_index_run` assertions), every fingerprinted pdf skips, row delta = new formats only, FTS rebuild succeeds; fp-collision scenario (c) and the §2.8 skip-collision fixture green with the batch-exported-notes fixture.
- Display: progressive-render test with a 6 MiB txt fixture (no long main-thread block; cite-jump into a late slice lands); >8 MiB fixture → truncation affordance with a multi-byte RU character straddling the 8 MiB boundary (char-boundary test); Linux deb manual smoke includes the scheduler-fallback path.
- Live: fast-index mixed pdf+md folder → per-format Finished counts (verify GPU counts match the sidecar); open an md citation → scroll+flash.

---

## 4. M2 — Ebook ingest (M)

### 4.1 Per-format

**.epub — CPU:** [`rbook`](https://crates.io/crates/rbook) (active, EPUB 2+3) → spine-ordered XHTML → `html2text` per spine item; `chapter` = TOC title for the item; one `Block` per item. **GPU:** PyMuPDF opens epub natively (synthetic pages); use `doc.get_toc()` to map page→chapter and stamp `chapter`. MuPDF synthetic pages will NOT match foliate-js rendering — citation jump uses text search (M3), pages stored only for citation display.

**.fb2 / .fb2.zip — CPU:** hand-rolled `quick-xml` (encoding feature on — declared windows-1251 handled) over `<section>/<title>/<p>`; `chapter` = section title chain; `author` from `<title-info>` (first format to fill `author`). `.fb2.zip`: `zip` crate, single entry, same parser. **GPU:** fitz opens fb2 directly; `.fb2.zip` gets a stdlib-`zipfile` unzip-to-temp shim (in `HANDLED_EXTS`). RU is a first-class test target.

**.mobi/.azw3 — CPU:** [`mobi`](https://crates.io/crates/mobi) crate → HTML → `html2text`; KF8/azw3 partial — parse failure → skip_state `("mobi parse failed — handled by Fast (GPU) indexing where available", pipeline=cpu)`, so the GPU path still picks the file up where it exists. **GPU (preferred):** fitz native. **DRM:** detect encrypted records → skip_state `("DRM-protected — cannot index")` recorded for **both** pipelines — no path can ever handle it.

**.xps:** GPU-only via fitz; CPU arm is a directed skip recorded pipeline=cpu with the platform-honest reason **`"xps: handled by Fast (GPU) indexing where available"`** — never "use Fast indexing", which is undeliverable advice on a Linux box without a configured GPU helper (see §8's explicit Linux GPU story). The §2.8 cross-pipeline test uses exactly this format. Display: "other" kind (extracted-text fallback after M3).

**.cbz:** **not added** — no text to embed, so it can never produce store rows, and Titles/Index are built from store rows, meaning no navigation path could ever reach a cbz reader. Dropped entirely (see §11).

### 4.2 Existing 69 epubs: skip is the *correct* outcome; chapter enrichment is opt-in, with identity spelled out

After M0a, the 69 already-imported epubs **fingerprint-skip on every index run** — invariant #1 working as designed — which also means their legacy chapterless chunks never gain TOC chapters through normal indexing. Two consequences, made explicit:

- The M2 exit criterion is worded accordingly (§1, §4.4): existing epubs produce **0 new rows**; TOC-chapter assertions run on **new epub fixtures** only.
- **Per-book "Re-index this book" action (ships in M2)** — and its identity mechanics are non-trivial because *three id schemes are in play* (invariant #11): the Titles/Index catalog carries the **store** book_id (legacy Python id for the 231 md + 69 epubs; gpu sha1 id for fast-indexed books), while `book_state` is keyed by the path-derived `stable_book_id` used by both pre-filters. The action therefore resolves everything from **`source_path`**, never from id equality:

  1. From the catalog entry take `(store_book_id, source_path)`.
  2. `delete_book(store_book_id)` — removes the store rows (this also makes the §2.1.3 source-path bootstrap check pass for the upcoming re-index; ordering is delete-first, correct).
  3. **Clear state under ALL ids matching that path:** delete `book_state` rows where `source_path` matches the new column (covers seeded legacy ids and gpu-committed ids), **plus** the row keyed by `stable_book_id(source_path)` explicitly (covers pre-M0a rows whose `source_path` column is still `''`).
  4. **Delete `skip_state` rows for that `source_path`, both pipelines** — a plain path-keyed delete (an explicit re-index overrides recorded skips; no fingerprint reasoning needed, and no risk of clearing a colliding file's records).
  5. Next index run re-extracts with the current extractor → TOC chapters, correct format stamp, `CURRENT_CHUNKER_VER` (clears the legacy nudge for that book).
  6. **The book returns under a NEW book_id** (DefaultHasher on CPU, sha1 on GPU — both ≠ the deleted store id). The frontend refreshes the catalog after the action; any UI state referencing the old id is invalidated then. Nothing persisted keys on book_id (§2.1.4 audit), so nothing else goes stale.

  Confirmation dialog states it will re-embed that one book. This is the *only* sanctioned route to chapter-enrich (or restamp) a legacy book — and it is also the corpus-preparation tool for the §5.5 cite-jump metric.

  **Smoke test — keyed on `source_path`, not book_id:** after the action + one index run, assert (a) exactly one book_id holds rows for that `source_path`; (b) chunk counts for every OTHER `source_path` are unchanged; (c) no `book_state`/`skip_state` rows reference the old ids or path; (d) the re-indexed book's rows carry chapters and the correct format stamp.

### 4.3 Cross-path identity note
CPU (`DefaultHasher` of path) and GPU (sha1) still mint different `book_id`s for the same file — pre-existing pdf behavior, inherited. The §2.1 source-path check + seeded book_state + path-equality short-circuit protect cross-path double-indexing at bootstrap. Unifying id schemes remains **deferred** (would orphan existing rows). README: "pick one indexing mode per collection."

### 4.4 Testing (M2)
- Fixtures: minimal epub (2 spine items + NCX); **RU fb2 in windows-1251 with nested sections**; fb2.zip of the same (exercises `ext_of` at all five sites); small non-DRM mobi; DRM-header stub (clean skip, recorded once, both pipelines).
- Assert: epub chunks carry TOC chapter; fb2 `author` populated; `Ch. <section>` citations; `~loc` fallback for chapterless mobi.
- GPU: run gpu_embed.py directly on fixtures → parquet schema matches `chunk_schema`, format column correct (not "pdf"), sidecar outcomes correct, import round-trips; a deliberately wrong format string exercises the §2.4 batch-failure path.
- **Exit criteria (live):** (a) fast-index over the real library → **all 69 existing epubs skip with 0 new rows and 0 embeds** (§2.1 acceptance harness re-run with epub enabled — the M2 gate); (b) **new epub fixtures index with TOC chapters** visible in the Index tab; (c) Ask a question grounded in a *newly indexed* epub fixture → `Ch. X` citation; (d) "Re-index this book" smoke test per §4.2 (source_path-keyed assertions).

---

## 5. M3 — Reader epic: foliate-js + universal fallback (L)

### 5.1 `BookReader.tsx` (foliate-js)
- **Vendor a pinned commit** of [foliate-js](https://github.com/johnfactotum/foliate-js) (MIT; API explicitly unstable → committed copy, required for offline anyway) + vendored zip.js (BSD-3)/fflate (MIT). Budget ≈ +0.3–0.5 MB.
- Contract mirrors PdfReader: `<BookReader key={path} url={convertFileSrc(path)} citeText chapter full onFail>`, self-contained (TOC dropdown, prev/next section, paginated↔scrolled toggle, font size); Reader view full-screen wraps it via the `full` prop like PdfReader.
- **WKWebView-safe load:** main-thread `fetch(asset://…)` (verified working) → `blob()` → `new File([blob], basename)` → `view.open(file)`. Zip inflation runs over the in-memory Blob — no worker fetches of custom schemes; `configure({ useWebWorkers: false })` escape hatch; never stream async-iteration (trap #2).
- Handles: **epub, mobi, azw3, fb2, fb2.zip** (cbz dropped — §4.1). `onFail` → downgrade to `"other"` kind (PdfReader→pdfNative pattern).

### 5.2 Scroll-to-citation for books — chapter-scoped search, then whole-book
- Normalize `citeText` like the md path, **plus**: collapse all whitespace runs, strip soft hyphens/dehyphenate — fitz/html2text-extracted chunk text and foliate-rendered DOM text differ exactly there.
- **Search scope, in order:** (1) if the citation carries a `chapter`, resolve it to the TOC entry and run foliate's search **within that section only** — a whole-book search on citation open can stall multi-MB epubs, and the chunk's chapter is known; (2) on a chapter miss, fall back to whole-book search, run asynchronously with the reader already open at the chapter (or start) so the UI never blocks. Navigate to the first CFI hit, flash highlight 2.5 s.
- No match at all → land on the chapter and show the cited passage in a dismissible overlay strip. Explicit, never silent.

### 5.3 Universal extracted-text fallback reader (the cheap epic)
- New command `extract_preview_text(path) -> { text, chapters }`: runs the ls-extract CPU extractor for any supported format, returns concatenated block text with chapter titles inlined as headings; 8 MiB cap (char-boundary truncation per §3.5), rendered through the same progressive text reader as §3.5 (incl. the scheduler fallback).
- **Jank discipline applies backend-side too:** a full CPU extraction of a large epub/mobi is seconds of CPU-bound work — the command body runs under **`tauri::async_runtime::spawn_blocking`**, and the frontend shows a loading state in the panel while it runs. Test with the largest fixture.
- The `"other"` panel gains "Show extracted text" → text reader (renderRich + TreeWalker cite-jump unchanged). Every ingested format instantly gets in-app display + citation jump; pretty rendering becomes an upgrade, not a prerequisite. This is also the Linux story for everything textutil-flavored.

### 5.4 ReaderKind + citation `chapter` plumbing — an **ls-app + src-tauri + artifacts + frontend** change, scoped precisely

- Union → `"pdf" | "text" | "book" | "html" | "other"`; `extOf` map (regenerated `supportedExts.ts`): `epub|mobi|azw3|fb2|fb2.zip → "book"`; render switch gains the branch.
- **Where the field actually lives (verified):**
  - `SearchResult` **already carries `chapter: Option<String>`** (crates/ls-query/src/lib.rs:31) — live search/ask responses already emit it from retrieval rows. **No retrieval-side work.**
  - The **persisted `Citation` struct is `crates/ls-app/src/types.rs:46`**, serialized into the messages table by ls-app's Db — it gains `chapter: Option<String>` with `#[serde(default)]`. Old persisted messages deserialize with `chapter: None` **inside ls-app** (that crate owns the round-trip, not src-tauri).
  - The `to_citation()` mapper (src-tauri/src/lib.rs:39–47) passes `chapter` through from `SearchResult` — a one-line change at the mapper, not a struct change at that location.
  - The **ls-artifacts `Source` mirror** gains the field (serde default) so exports carry it.
  - Frontend `Src` gains `chapter?: string`; `toSrc` (App.tsx:185–191) passes it through.
- **Scope note for the estimate:** M3 therefore touches the **ls-app crate boundary** (types + message persistence) in addition to src-tauri/frontend — small in lines, but it is a persisted-schema change and gets the compat test below; the §2.1.4 audit already confirmed the struct is safe to extend.
- **Compat test:** deserialize a pre-upgrade **messages-table row fixture via ls-app** (plus a pre-upgrade artifact JSON via ls-artifacts) → loads with `chapter: None`, UI renders, re-save round-trips.
- Titles' hardcoded `page: 1` and Index's nullable page: BookReader ignores `page`, uses `chapter`/`citeText`.

### 5.5 Testing (M3)

- **Cite-jump match-rate metric (exit criterion) — corpus defined explicitly, because the live library alone cannot exercise the shipped design.** All 69 live epubs are legacy-indexed: old-Python-extractor chunk text, no chapters — citations from them can only ever exercise the whole-book fallback of §5.2, never the chapter-scoped fast path that is the primary design. The metric therefore runs on a **two-bucket corpus prepared as the first step of the measurement**:
  - **Bucket 1 — chaptered (steady-state):** re-index **8 real epubs** via the §4.2 action (this is the shipped M2 feature, used as intended), then take **12 real citations** from Ask runs grounded in them. These carry `chapter` and new rbook/fitz-extracted chunk text — they exercise the chapter-scoped search. **Threshold: ≥ 80% direct jumps (CFI hit within the cited chapter).**
  - **Bucket 2 — legacy chapterless (fallback):** **8 real citations** from un-re-indexed legacy epubs (old-extractor text, no chapter) — they exercise the async whole-book search against realistically mismatched text. **Threshold: ≥ 50% located (direct hit or fallback highlight); 100% must at least land on the overlay-strip explicit-miss state — never a silent nothing.**
  - **Both buckets:** record time-to-highlight; **no citation open may block the UI** (the async fallback requirement). Below either threshold, iterate §5.2 normalization before shipping.
- Live matrix: open epub/fb2/fb2.zip/mobi from (a) citation chip, (b) Titles, (c) Index; paginated/scrolled toggle; full-screen Reader view; corrupt epub → onFail panel; a multi-MB epub → cite-jump lands without stall.
- WKWebView: verify on the *packaged* macOS build (dev server can mask asset:// behavior); Linux WebKitGTK deb manual smoke (incl. the §3.5 scheduler fallback).
- Fallback reader: `extract_preview_text` unit tests per format incl. the spawn_blocking/loading-state path on a large fixture; open .xps/.ipynb → "Show extracted text" → TreeWalker jump.

---

## 6. M4 — Office ingest + display (M)

### 6.1 Ingest (CPU, pure Rust — Linux gets full parity)
- **.docx:** hand-rolled `zip` + `quick-xml` over `word/document.xml` (~150 LoC): `w:p`→paragraph, `w:t`→text, `w:pStyle Heading1..2`→`chapter` (same depth cap as md). No pages → `page: None`, md-style citations. Skip `docx-rs` (writer) and `dotext` (abandoned); `docx-rust` if hand-rolling stalls.
- **.rtf:** [`rtf-parser`](https://github.com/d0rianb/rtf-parser) `to_text()`. **Gating fixture first:** RU `\'xx` cp1251 escapes + `\ansicpg1251` before committing to the crate; fallback `rtf-grimoire` lexer. `chapter: None`.
- **.odt:** zip+quick-xml over `content.xml` (`text:p`, `text:h`); ~100 LoC, no mature crate exists.
- **GPU:** venv-optional `python-docx` / `striprtf` / `odfpy` behind `try: import` (availability reported by `--caps` via `find_spec`, §2.5); a missing dep is a **runtime-discovered skip that only Python can see** — it surfaces as a sidecar `skipped` outcome (`"docx support: pip install python-docx"`) and lands in skip_state as pipeline=gpu (§2.10), keyed by the file's path, stamped with the run's GPU caps_ver — never a success stamp, never blocking the CPU path, and **automatically retried after `pip install`** because the `--caps` probe changes the caps_ver (§2.8).

### 6.2 Display — new `"html"` reader kind
- Sanitized-HTML sibling of `reader-md` — same-document DOM, so TreeWalker cite-jump + cite-flash work unchanged.
- **Sanitization:** vendored DOMPurify (~22 KB) tight allowlist; strip remote `src`/`href`; same-archive images → `data:` URIs or dropped.
- **HTML cap is 4 MiB, deliberately lower than the 8 MiB progressive text cap:** DOMPurify + `innerHTML` of a many-MiB document is a single synchronous main-thread operation. Documents over the cap route to the extracted-text reader (§5.3) — which handles them progressively — with a note.
- **.docx:** vendored mammoth.js (BSD-2), **lazy-loaded** (dynamic import, ~0.5–0.8 MB out of initial bundle): main-thread `asset://` fetch → ArrayBuffer → `convertToHtml` → DOMPurify → container. "Show extracted text" always present as fallback.
- **.html/.htm files:** `read_source_text` → DOMPurify → container (upgrades .html from "other").
- **.rtf/.odt:** no JS renderers. macOS: `convert_with_textutil(path, "html")` → cache per §0.b → `resolve_display_path` → DOMPurify → html container ("Converted via macOS textutil" label). Linux: text reader via `extract_preview_text` ("Showing extracted text"). `source_path` stays the original in both (§0.b).

### 6.3 Testing (M4)
- Fixtures: docx (Heading1/2 + RU + table); **cp1251 rtf (the crate-gating test)**; odt with `text:h`. Assert chapters, char-based locs, `Ch. X` citations.
- Display: mammoth sanitization test (hyperlink + remote image → stripped); cite-jump into docx-rendered paragraph; >4 MiB html fixture → routed to text reader.
- Cross-platform: Linux CI runs all CPU extract tests; macOS manual textutil-cache idempotence check.
- Regression: mixed pdf+docx+rtf folder → existing books fingerprint-skip; skip_state silences repeats within each pipeline only; a GPU missing-dep run leaves **no book_state rows** for the skipped docx files, and the same files index after installing the dep (caps_ver retry, §2.8).

---

## 7. M5 — Best-effort tier (M)

Pattern: **probe → convert/extract if possible → labeled skip + "Open externally"**. Every M5 extension enters `SUPPORTED` with a matching `DIRECTED_SKIPS` entry in gpu_embed.py — the lockstep test (§2.5) enforces it. Directed skips travel via the sidecar (§2.10) and land as pipeline=gpu skip_state rows keyed by the file's path — directed, once-per-file, never generic `skip (error)`, never success-stamped. Skip strings must not promise what the other path can't deliver on some platform:

| Ext | GPU directed-skip reason (sidecar `reason` field) |
|---|---|
| .doc | `.doc: handled by standard indexing where a converter is available` — not "use standard indexing", which is wrong advice on Linux without antiword/soffice |
| .pages | `.pages: handled by standard indexing — needs an embedded PDF preview` |
| .webarchive | `.webarchive: handled by standard indexing` — CPU path is cross-platform (plist crate) |
| .djvu / .chm | `djvu/chm: handled by standard indexing where djvutxt/7z is installed` |

Because these are recorded pipeline=gpu, they never block the CPU path; conversely a CPU skip (e.g. Linux .doc without a converter) never hides the file from a future GPU capability. **And because the CPU caps_ver includes the runtime PATH-probe set (§2.8), "retried when capabilities or the file change" is literally true:** installing antiword/soffice/djvutxt/7z changes the next run's caps_ver and clears every skip recorded under the old one — verified by the PATH-shim test below.

**.doc (legacy Word):** no credible pure-Rust parser — don't build one. macOS: `textutil -convert txt` → cache per §0.b → ingest cached txt, stamp `format: Doc`, `source_path` = the .doc. Linux: PATH-probe `antiword`, then `soffice --headless`; neither → skip_state `("legacy .doc: install antiword or LibreOffice", pipeline=cpu)`, retried when capabilities (caps_ver PATH probe) or the file change. Display: textutil→html kind on macOS; extracted-text elsewhere.

**.pages — "Preview.pdf or bust" ladder, identity per §0.b:** (1) open as zip or bundle-dir → if `QuickLook/Preview.pdf` exists, extract to `<data_dir>/converted/<signature>.pdf`; **ingest** runs the pdf extractor over the cached artifact but stamps `format: Pages` + original `source_path`/`book_id`; **display**: `resolve_display_path` returns the cached pdf → PdfReader (full parity) while the moved-file guard, "Open in default app", and dedup all track the original .pages. (2) Optional explicit user action: JXA export via Pages.app into the cache (automation permission; never automatic). (3) Neither → skip_state `(".pages without embedded preview — open in Pages and export PDF")`. IWA strings-scrape deferred. Expect ~half of Pages-5.5+ files to lack the preview. Linux: rung 1 only. Cache invalidation: fingerprint change → new signature (§0.b).

**.webarchive:** cross-platform — `plist` crate → `WebMainResource/WebResourceData` HTML → cached per §0.b → existing html ingest + DOMPurify display. textutil used only as a test oracle on macOS.

**.djvu:** PATH-probe `djvutxt` (djvulibre, **GPL — subprocess only, never link/bundle**); present → text-layer extract → ingest as Txt-family (original identity); absent or no text layer → skip_state `("djvu: brew install djvulibre (scanned-only djvu needs OCR — unsupported)")`. Display: open externally.

**.chm:** PATH-probe `7z x` → HTML tree → html pipeline; absent → labeled skip. Gate behind a `discover` dry-run count — implement only if the library actually contains .chm.

**Testing (M5):** ladder fixtures pages-with-preview / pages-without; **identity tests**: pages book's `source_path` is the .pages, fingerprint tracks it, moving the .pages triggers the moved-file guard, evicting the cache re-converts on next open; webarchive plist fixture (Linux CI); **PATH-shim retry test (the §2.8 test, exercised with the real M5 formats): record a .doc skip with no converter on PATH → shim `antiword` into PATH → next run computes a new CPU caps_ver → the .doc is re-attempted and indexes**; PATH-shim mocks for all subprocess tiers + macOS manual checklist; every skip path asserts its exact reason string **and pipeline scope** (re-run the §2.8 cross-pipeline test with a .doc: GPU directed skip must not suppress the CPU converter path, and vice versa); lockstep test re-run confirms all M5 exts covered.

---

## 8. Cross-platform matrix

| Format | Linux ingest | Linux display | macOS extra |
|---|---|---|---|
| md/txt/rst/adoc/org/tex/ipynb | full (pure Rust) | full (progressive text reader, scheduler fallback §3.5) | — |
| html/webarchive | full | full (sanitized html, ≤4 MiB; text reader above) | textutil as test oracle |
| epub/fb2(.zip) | full (rbook / quick-xml; fitz on GPU) | full (foliate-js) | — |
| mobi/azw3 | CPU partial (mobi crate); GPU only if configured (below) | full (foliate-js) | GPU full via fitz |
| xps | none by default — labeled skip on both paths unless GPU helper configured (below) | extracted-text fallback (when ingested) | GPU full via fitz |
| docx/rtf/odt | full (pure Rust) | docx: mammoth; rtf/odt: extracted-text | rtf/odt pretty display via textutil→html |
| doc | if antiword/soffice on PATH, else labeled skip | extracted-text (if converted) | textutil (built-in) |
| pages | Preview.pdf rung only | PdfReader via cached preview (`resolve_display_path`) | + optional Pages.app export |
| djvu/chm | if djvutxt/7z on PATH, else labeled skip | open externally | same (brew) |

**Linux GPU-path story (explicit, because three formats lean on it):** gpu_embed.py defaults to `device=mps` (gpu_embed.py:136) — the fast path is realistically **macOS-only out of the box**. On Linux, fast indexing works only if the user points `python_bin` at a venv with torch and a working device **and sets the `gpu_device` setting** (M0b, §2.5) — the concrete plumbing is: `gpu_device` in Settings → Indexing, passed by `run_py_batch` as `--device <value>` on every spawn, and folded into the GPU caps_ver. Consequently: `.xps` has **no default Linux ingest path** (it is a labeled, caps_ver-scoped skip on both pipelines there — retried automatically if the user later configures the GPU helper or changes `gpu_device`, since both change the caps hash); `.mobi/.azw3` fall back to the partial `mobi`-crate CPU extractor on Linux. All skip strings use the platform-honest phrasing ("handled by Fast (GPU) indexing **where available**") — never advice a Linux user can't follow.

Nothing degrades silently: every gap is a once-recorded, pipeline-scoped, path-keyed, caps_ver-refreshable skip_state reason or an explicit reader panel.

---

## 9. Dependency & bundle budget

**Rust crates (dmg impact negligible):** `encoding_rs` + `chardetng`, `scraper`/`dom_query` + `html2text`, `rbook`, `quick-xml` + `zip` (transitivity verified by the M0b `cargo tree` check, not assumed), `mobi`, `rtf-parser`, `plist`, `serde_json` (present).
**Frontend (vendored, pinned):** foliate-js +0.3–0.5 MB · mammoth.js +0.5–0.8 MB lazy-loaded · DOMPurify +22 KB → ≈ +1.0–1.3 MB on 67 MB. Within budget. No JS test runner added (§2.5 codegen keeps CI in Rust).
**Python venv (optional, setup-time):** `python-docx`, `striprtf`, `odfpy`, `charset-normalizer` — all guarded imports whose absence surfaces as sidecar skip outcomes (retried after install via the `--caps` caps_ver), never errors.
**External binaries (probe-only, never bundled):** textutil (macOS built-in), antiword/soffice, djvutxt (GPL — subprocess only), 7z.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Legacy-book re-embed/duplication on first post-M1/M2 index (231 md + 69 epub + legacy pdfs)** | §2.1.0 ground-truth query settles which failure mode is live before code is written; §2.1 guard (seed via ported `run_backfill_state` + row-returning reverse lookups + path-equality short-circuit + source-path bootstrap check); acceptance test (0 embeds, row-count invariance, source_path-keyed, remap-tolerant) executable via the §2.1.5 harness, an M0a/M1 exit criterion, re-run as the M2 gate |
| **Accidental bulk store rewrite on first post-upgrade run** — via stage-2 **or stage-4**: Dropbox mtime churn makes the seeded fp stale so stage-2 misses and the content-sig stage hits old≠new for ~300 unmoved files | §2.1.2 path-equality short-circuit applied at **all four remap sites** (both stages × both pipelines), consuming the widened row-returning lookups (no follow-up queries), with fingerprint refresh under the existing id on a stage-4 path-equal hit; Dropbox-churn scenario (b) in the acceptance test |
| **`"missing"`-sentinel identity corruption** — an unreadable/dehydrated candidate csig-matches a seeded `"missing"` row and remaps an unrelated book; or, in-run, the second unreadable file silently matches the first via `seen_content` | §2.1.1 hardening at all **four** sites: lookups treat `"missing"` as no-match; writers refuse to persist it; `seen_content` never inserts nor matches it (rule carried into `plan_index_run`); migration purges pre-existing sentinel rows; two-unreadable-files fixture (scenario d) |
| **size:mtime fingerprint collision remaps** — M1's thousands of small batch-exported files (e.g. the 212 ByteByteGo lessons) make same-size/same-second pairs realistic; LIMIT-1 lookup + old≠new would ping-pong remap two distinct files | §2.1.2 csig confirmation against the returned row's stored csig before any cross-path remap (512 KiB read only on the collision path); csig mismatch → fall through to normal indexing; scenario (c) fixture test, re-run in M1 with a batch-exported-notes fixture |
| **Fingerprint collision silently suppressing indexing via skip records** | §2.8 skip_state keyed on `(collection_id, source_path, pipeline)` — a skip is about one file at one path; fingerprint is a staleness column only; skip-collision fixture (file B with colliding fp indexes normally; B's success doesn't erase A's skip) |
| **Acceptance test silently unexecutable** (no headless fast-index entry; embeds inside spawned Python) | §2.1.5: pre-filter extracted into pure `plan_index_run` shared by the command; CPU assertions via counting embedder stub + row invariance; GPU assertions via `plan.to_embed` emptiness + row invariance; snapshot fixture mechanism budgeted as its own work item |
| **M0 underestimation gating everything** | Split into M0a/M0b (§1): M1 CPU dev proceeds on M0a alone; sidecar/skip_state/codegen (M0b) gate only the GPU legs; combined sizing stated honestly at 2.5–4 weeks |
| **GPU skips stamped as successes, silencing files on the CPU path forever** (verified live bug: lib.rs:615–629 commits success book_state for skipped files) | §2.10 sidecar outcome channel + outcome-aware batch commit: indexed→book_state, skipped→skip_state, error→nothing; stderr demoted to display-only; missing sidecar → batch-level failure, never fabricated success |
| **Skip records corrupting dedup stages** (skip rows in book_state would be found by the fingerprint lookup and remapped into successes) | §2.8: skip_state is a separate, path-keyed table — structurally invisible to book_state reverse lookups; stage-2 safety test pins it |
| **CPU-only skip permanently hides a file from the GPU path that can handle it (or vice versa)** | §2.8 pipeline-scoped skip_state + per-pipeline caps_ver; mandatory cross-pipeline tests (.xps CPU-skip → fast-index embeds it; GPU missing-dep skip → CPU still indexes) |
| **External-tool/missing-dep skips never retried after the user installs the tool** (compile-time caps_ver can't see PATH/venv changes) | §2.8: caps_ver computed at run start — CPU folds in the probed PATH-tool set, GPU folds in `gpu_embed.py --caps` (script + `find_spec` dep availability) ⊕ the `gpu_device` setting; PATH-shim test (antiword) + pip-install test (python-docx) + device-change test prove the retries |
| **`--caps` probe taxing every fast-index run or failing on a broken torch** | §2.5: stdlib-only prologue handles `--caps` before any heavy import; `find_spec` probes never execute imports; <300 ms cold-latency + broken-torch acceptance tests |
| **"No extractable text" silencing** (status quo writes success-shaped book_state) | §2.8: relocated to skip_state (caps_ver-stamped → extractor upgrades retry it), invariant #5 true as stated; pre-existing no-text book_state rows documented as unmigratable-but-harmless; image-only-PDF fixture test |
| **Seeded books flood the re-index nudge** (chunker_ver=0 semantics, store.rs:415/:522) | Deliberate and documented (§2.1.1): the nudge is honest — legacy chunking IS older; §4.2 re-index clears it per book with `CURRENT_CHUNKER_VER`; test that the nudge never auto-embeds |
| **Re-index action deletes rows under one id but leaves state under another** (three id schemes in play) | §4.2 resolves everything from `source_path`: clear book_state by path column + explicit `stable_book_id(path)` key, clear skip_state by path (both pipelines), refresh catalog for the new id; smoke test keyed on source_path |
| Moved/missing legacy files unseedable → later re-embed as duplicates with stale rows | Accepted residual risk: seed migration reports the unseedable list (index-log + Settings→Maintenance); §2.6 orphan-row report + explicit prune |
| Something persisted keys on book_id and breaks under remap/re-index | §2.1.4 persistence audit in M0a (citations use source_path — confirmed; artifacts confirmed; notebook/session audited before ship) |
| **Citation `chapter` persisted-schema change breaks old sessions** | §5.4: field added on the actual persisted struct (`ls-app/src/types.rs:46`) with `#[serde(default)]`; compat test deserializes a pre-upgrade messages-table row **via ls-app**; retrieval side needs nothing (SearchResult already carries chapter) |
| **Cite-jump metric validating the wrong corpus** (all live epubs are legacy/chapterless — would test only the fallback) | §5.5 two-bucket corpus: 8 epubs re-indexed via §4.2 first (chaptered bucket, ≥80% direct) + legacy citations (fallback bucket, ≥50% located, explicit-miss overlay for the rest); per-bucket thresholds |
| foliate-js API instability | pinned vendored commit; BookReader wrapper isolates a swap; `onFail`→"other" keeps users unblocked |
| WKWebView surprises in foliate/zip.js workers | in-memory File load (no worker fetches); `useWebWorkers:false`; verify packaged macOS build pre-release |
| epub cite-jump mediocre match rate / stalls on big books | whitespace/soft-hyphen normalization; chapter-scoped search first, async whole-book fallback (§5.2); §5.5 per-bucket thresholds + no-stall requirement; chapter+overlay fallback for misses |
| Large text files jank the WKWebView main thread — or the progressive renderer freezes on Linux | §3.5: 8 MiB cap + progressive slices via rIC **with named setTimeout/rAF fallback** (WebKitGTK may lack rIC) + char-boundary-safe truncation affordance; HTML stays at 4 MiB synchronous (§6.2) with extracted-text routing above it; `extract_preview_text` under spawn_blocking with a loading state (§5.3) |
| Heading-dense notes mint thousands of tiny chunks / flood Index tab | §3.2 extraction-time section floor + H1–H2 cap; fixture test on min chunk size; live check on the 212-lesson corpus |
| Legacy epubs never gain TOC chapters (correct skip behavior blocks enrichment) | Explicit, documented; opt-in per-book "Re-index this book" (§4.2) is the only sanctioned route (and the §5.5 corpus-prep tool) |
| rtf-parser fails RU cp1251 | gating fixture before adoption; fallbacks rtf-grimoire → textutil → striprtf |
| mobi crate KF8 gaps | GPU/fitz preferred where available; CPU failure is a directed, once-recorded, cpu-scoped skip with platform-honest wording |
| GPU-path formats undeliverable on Linux (.xps fully, .mobi/.azw3 partially) | §8 explicit Linux GPU story: `gpu_device` setting + `--device` plumbing shipped in M0b (§2.5), folded into caps_ver so configuring it retries skips; platform-honest skip strings ("where available") |
| Unknown-format parquet rows | M0b import hard-error naming format+file; failed batch voided (no state), run continues (§2.4 contract change to run_py_batch, named and tested); read path stays tolerant; §2.5(d) round-trip test prevents as_str/from_ext drift |
| Re-scope surprise (folders suddenly ingest notes) | M1 per-format Finished counts (sidecar-derived on GPU); §2.1 guard; release notes; optional collection filter |
| Frontend ext map drifts from Rust canonical list | §2.5 codegen (`supportedExts.ts` generated from ls-core) + Rust freshness test |
| Converted-artifact identity drift (cache tracked instead of original) | §0.b contract + `resolve_display_path`; M5 identity tests |
| Custom `indexer_script` users miss gpu_embed updates / lack the sidecar or `--caps` (incl. after the §2.5 restructure changes script bytes) | release notes + version banner in script stderr; sidecar absence → batch-level failure with a directed log message; `--caps` absence → script-bytes-hash fallback caps_ver + mismatch note (§2.5, §2.8) |
| 302k-chunk regression | every milestone: real-library snapshot run via the §2.1.5 harness → 0 embeds for fingerprinted books, row delta = new formats only; backup note before first post-upgrade index |
| textutil/subprocess untestable in CI | PATH-shim mocks + macOS manual checklist per release |

---

## 11. Deferred / won't do (with reasons)

- **.cbz** — produces no store rows (nothing to embed), so Titles/Index can never reach it; a reader for it would be dead code. Dropped from ingest *and* display. **.cbr** additionally needs `unrar` (licensing). Won't do.
- **.pages IWA parsing (strings-scrape rung)** — non-spec Snappy + undocumented protobufs; unordered text ruins citations. Deferred indefinitely; Preview.pdf ladder covers the useful half. **.pages GPU handling** (unzip→fitz shim) — deferred; directed skip until demand.
- **DRM (.azw DRM, KFX) and dead formats (.lit/.pdb/.snb)** — detect and label; never parse. Won't do.
- **OCR** (scanned djvu/pdf) — separate epic.
- **epub.js** — unmaintained, epub-only; foliate-js chosen. Won't do.
- **Unifying CPU/GPU book_id schemes** — would orphan existing rows; §2.1's source-path check + path-equality short-circuit + csig confirmation remove the practical risk. Deferred; "one indexing mode per collection" documented.
- **Rebuilding `book_state` with a new PK** (e.g. adding pipeline or status to the key) — unnecessary once skips live in `skip_state`; a PK rebuild is a full table copy with real migration risk for zero benefit. Won't do.
- **Fingerprint-keyed skip identity** — considered and rejected (the plan's own collision analysis, §2.1.2, proves size:mtime is not identity); skip_state is path-keyed with fingerprint as staleness (§2.8). Won't revisit.
- **Purging pre-existing "no extractable text" rows from `book_state`** — indistinguishable from real successes (no status column); they behave exactly as today and self-heal when the file's bytes change. Won't attempt; documented instead (§2.8).
- **Automatic chapter enrichment of legacy books** — unreachable under invariant #1 by design; opt-in per-book re-index (§4.2) only. Won't automate.
- **Default rewrite of legacy rows** — optional one-column repair only (§2.6); extension-at-read-time makes it cosmetic.
- **Frontend JS test harness (vitest)** — not needed; codegen + Rust freshness test (§2.5) covers the lockstep guarantee. Deferred until a genuine frontend-logic testing need appears.
- **Parsing file paths or reasons out of stderr prose** — retired permanently in favor of the sidecar (§2.10); stderr is a human/progress channel only. Won't do again.
- **Full CommonMark (tables, images, highlighting) in renderRich** — headings/fences/links cover 95%.
- **.chm unless the library scan finds any** — gated, not scheduled.
- **In-app display for xps/djvu** — extracted-text fallback / open-externally suffices at their frequency.
- **Chunked/progressive HTML sanitize-and-render** — only if real >4 MiB html documents appear (text reader already renders progressively, §3.5).

---

## 12. Release slicing

- **v0.9.0:** M0a + M0b + M1 (guards + state schema + sidecar + harness + plain ingest; "index your notes"). **M0a and M0b are both hard release gates even though M1's CPU development only needs M0a** — enabling any new extension in `SUPPORTED` re-scopes discovery for *both* pipelines, so the GPU stamping fix and sidecar must be in the same shipped binary as the first new extension. Gates: ground-truth query recorded; legacy-book acceptance test, all four scenarios (0 embeds, row invariance, source_path-keyed, churn/collision/two-unreadable fixtures); sentinel purge count recorded; sidecar + cross-pipeline + no-text + skip-collision tests; `--caps` latency test; real-library zero-re-embed audit via the harness; unseedable-residue count in release notes (which also note the chunker_ver=0 nudge and the no-text relocation).
- **v0.10.0:** M2 + M3 (ebooks searchable + in-app book reader — headline release). Gates: §2.1 test re-run with epub enabled (69 epubs skip, 0 new rows); new-fixture TOC-chapter assertion; §4.2 re-index smoke test (source_path-keyed); §5.5 two-bucket cite-jump metric (≥80% chaptered / ≥50% legacy, no stall); ls-app messages-table compat fixture (§5.4).
- **v0.11.0:** M4 (office). Gate: cp1251 rtf fixture; GPU missing-dep run leaves no success stamps and retries after install (caps_ver).
- **v0.12.0:** M5 (best-effort tier). Gate: pipeline-scoped skip cross-test with .doc + PATH-shim retry test + lockstep re-run.

Each release follows the existing commit/CI/release routine and records its regression run (row-count audit, zero re-embeds, lockstep test, orphan report delta) in the release notes.

---

## 13. Final-round amendments (binding — supersede the sections they reference)

The adversarial review ran six rounds (12 -> 8 -> 5 -> 8 -> 8 -> 5 issues); the last five
findings are folded in here rather than through another full replan. Each amends the plan
above and carries the same weight as the section it modifies.

**A1 (CRITICAL, amends invariant #6 and §2.1.1 — joins the M0a exit criteria).**
`content_signature` (crates/ls-app/src/service.rs:105-139) returns the `"missing"` sentinel
only when `File::open` fails; `hash_n` silently `break`s on a mid-read error, so a file that
opens but fails during read (offline Dropbox placeholder, permission flip, NFS hiccup) yields
a CONFIDENT signature derived from `len` + partial bytes. Two distinct same-length unreadable
files then produce identical non-sentinel csigs and match each other in `seen_content` /
`book_state_for_content`, and stage-4 fires an old!=new remap across genuinely different
files — the exact corruption §2.1.1 exists to prevent, waved through because the value is not
`"missing"`. Fix: `content_signature` returns the sentinel (or a `Result`) on ANY read error
or short read (bytes hashed < min(len, expected sample)); §2.1.2's "a content-signature hit
IS the identity proof" axiom holds only for non-degenerate signatures. New fixture: a
same-length pair of open-but-fail-mid-read files -> sentinel, no `seen_content` entry, no
stage-4 match.

**A2 (amends §2.1.1 seed migration).** The seed computes fingerprint + content signature
(up to 512 KiB read) for ~800 books that live in Dropbox; dehydrated placeholders either
block until hydration (the documented venv-hang failure shape) or mass-fail offline. The
seed therefore runs as a RESUMABLE BACKGROUND task — never blocking startup — with progress
in the Maintenance surface, detection of dehydrated/read-failed files, automatic re-attempt
of unseeded books on subsequent starts plus a manual Maintenance retry button (it is not
strictly one-time), and a release-note warning about hydration cost.

**A3 (amends §12 release slicing).** Do not couple the dedup-pre-filter rewrite with the
first SUPPORTED expansion in one binary. Ship (or at minimum dogfood on the live library)
M0a+M0b as a dark v0.8.x with `SUPPORTED` still `["pdf"]`: one real fast-index + CPU-index
run over the unchanged library validates the rewritten pre-filters (0 new rows, 0 embeds)
with zero new-format variables. Only then flip extensions in v0.9.0.

**A4 (spec gaps).** (a) §2.3: `.ipynb` is assigned to the **Md** family so the lockstep
test's leg (a) passes on day one. (b) §2.10: an all-skipped GPU batch produces an EMPTY (or
absent) parquet plus a full sidecar — gpu_embed.py writes the empty parquet explicitly, and
both import and the outcome-aware commit must tolerate it (add this batch shape to the §2.10
test list). (c) §5.2: chapter-scoped search must NORMALIZE chapter strings (trim, collapse
whitespace, strip leading numbering) when matching the stored chapter against foliate-js's
parsed TOC labels — not just normalize citeText — otherwise bucket-1 misses silently
downgrade to whole-book search and erode the >=80% threshold for the wrong reason.

**A5 (amends §1 sizing).** M1 as scoped ("M") is optimistic: it bundles 9 CPU extractors
with heading + section-floor logic, the mirrored GPU text-family branch, per-format Finished
counts on both pipelines, AND the §3.5 progressive display work (itself 1-2 days with its
test matrix). The §3.5 display work is split out as its own line item (**M1b — display**,
S-M) so the ingest release gate is not squeezed; M1 ingest remains M, and M1+M1b together
are honestly L.

---

## §14 Post-ship addendum (v0.13.0): the hybrid standard-engine sweep

Shipped after M5 exposed a routing gap: on a GPU-configured machine the single Index
button always runs the GPU helper, whose §7 directed skips point doc/pages/webarchive/djvu
at "standard indexing" — which never ran. Design (adversarially critiqued, 17 confirmed
amendments folded in):

- **Partition, don't filter twice.** `fast_index_collection` splits discovery on
  `ls_extract::CONVERTED_EXTS`: converter-only formats never enter the GPU plan or batches
  (no wasted helper spawn, no junk `gpu` skip rows, no double counting).
  `gpu_embed.py`'s DIRECTED_SKIPS remain as a safety net for custom/legacy scripts; a
  lockstep test (`partition_set_matches_gpu_directed_skips`) pins the two sets equal.
- **Metadata repair is unconditional.** The sweep files are pre-planned with the CPU
  pipeline (`plan_index_run`, `cpu_caps_ver()`) inside the same model-free planning block
  as the GPU plan; their state refreshes and moved-file remaps apply on every run —
  regardless of models, cancellation, or GPU failures (invariant: a moved .doc is
  re-pointed exactly like a moved .pdf).
- **The embed phase is lazy.** Only when the sweep plan's `to_embed` is non-empty (and
  Stop wasn't pressed — `state.cancel` checked directly, covering the GPU-empty path) does
  the run emit Loading, load the CPU embedder, and run `Service::index_collection` over
  exactly those paths. Steady state pays zero model loads.
- **One run, one Finished.** The sweep's internal Finished is suppressed; its stats merge
  (`IndexStats::merge`) into the GPU phase's for a single summary. Its Skipped reasons are
  mirrored into the persistent index-log (they carry the remedy: "install antiword…",
  "brew install djvulibre…").
- **Exit matrix.** GPU `to_embed` empty → sweep runs. Batch loop settle (incl. Stop) →
  sweep runs unless Stop. Fatal helper-spawn failure → FTS skipped, sweep STILL runs (it
  needs no Python), then the error is returned without a Finished (the rejected invoke is
  the frontend's terminal signal; sweep results are committed + logged). Shared-infra
  failures (tmp dir, SQLite, Lance FTS) → sweep skipped, same error would hit it too.
  Sweep embed failure → contained (log + counted as failed, no skip_state row) unless the
  sweep WAS the run (GPU had no files), then it surfaces as the command error.
- **Converter waits are bounded.** All four converter subprocesses run through
  `run_bounded` (120s deadline, kill + reap, stdout drained on a thread) so Stop is never
  wedged behind a hung soffice or a dehydrated-placeholder read. Known remaining seam:
  pure-Rust reads (cache_key head read, extract_pdf on a cached preview) can still block
  on dehydrated placeholders — pre-existing, deferred. Threading the cancel flag into
  ls-extract was considered and deferred (public-API churn for marginal gain once waits
  are bounded).
- **Setup installs office deps.** python-docx/striprtf/odfpy join the provisioning pip
  list, so fresh GPU setups handle docx/rtf/odt natively; existing venvs pick them up on
  the next Setup run (the caps-hash change then auto-retries old skips).
- **Frontend.** The sweep phase re-emits `started`: the chunk accumulator now survives
  phase changes and ETA uses a per-phase clock (`phaseStart`).

---

## §15 Post-ship addendum (v0.14.0): office variant dedup + the §2.6 Maintenance panel

Adversarially critiqued (14 confirmed amendments). Two features, one theme — library hygiene:

- **Ranking** (discover.rs): office formats join same-stem variant dedup as
  epub(0) pdf(1) fb2(2) mobi(3) azw3(4) docx(5) odt(6) rtf(7) doc(8) pages(9) djvu(10).
  Every pre-existing ebook decision is preserved; pdf beats office twins; pages ranks last
  among office because its extractability depends on file CONTENTS (embedded Preview.pdf) —
  a failure no tool install can retry past, so it must never shadow an extractable sibling.
  Known seam (deferred): the winner shadows losers before extraction is attempted and
  discovery never consults skip_state — a plan-time fallback that re-admits the next-ranked
  sibling when the sole winner carries an active skip is backlogged.
- **Maintenance** (ls-app/src/maintenance.rs + Store batch helpers): four scans — store
  orphans (file gone; dehydrated placeholders pass fs::metadata and are never flagged),
  manifest-only orphans (book_state rows with no store presence, incl. pre-M0a empty-path
  rows), format-stamp repair (family-level comparison on the RAW stored string; batched
  IN-list UPDATEs), duplicate variants (two-level grouping: variant_key → distinct ranked
  ON-DISK paths; keeper = best on-disk rank) and same-path multi-id rows (keeper = the
  manifest's id, else the path-derived id). Removal recipe = reindex_book minus the
  re-embed: store rows under EVERY id holding the path, then clear_book_state_by_path
  (path-keyed across id schemes — single-id deletion strands manifest rows that would
  silently shadow a restored file), then erase_skips. `apply` RE-DERIVES targets at apply
  time (client reports are never trusted) and returns per-category acted-on counts; after
  any change it rebuilds FTS (updates/deletes write fragments the FTS otherwise
  flat-scans) and compacts (Store::optimize). Roots that fail fs::metadata are reported as
  unreachable and everything under them is unjudgeable — never orphaned.
- **Exclusivity**: AppState.busy (RAII CAS guard) — index runs and maintenance fixes are
  mutually exclusive in the backend, both directions.
- **UI**: Maintenance tab (action tab — no Save footer), collection-scoped with the name
  in every confirm string; fix results show acted-on counts with an explicit drift note
  when they differ from the scanned counts; the Titles/Index catalog loader now refetches
  on invalidation (catalog/catalogFor joined its effect deps).

---

## §16 Post-ship addendum (v0.15.0): re-chunk that actually re-chunks

User-caught: the v0.6.2 "Re-chunk on next Index" opt-in (clear book_state) became a
structural no-op — the planner's stage-3-by-id store-presence guard re-skips and re-seeds
every store-present book. Tracing it exposed a second latent bug: the GPU batch commit
imported parquet without deleting a re-embedded book's old chunks (the CPU path deleted,
but only under one id scheme). Fix (adversarially critiqued, 9 confirmed amendments):

- **Persistent flag**: collections.rechunk_pending (SCHEMA+ALTER). The opt-in ARMS it;
  nothing is wiped. index_health reports it; the UI shows a persistent armed banner.
- **Planner re-chunk mode**: PlanCtx.rechunk. After stage 0.5, a candidate whose manifest
  chunker_ver < CURRENT (or with store presence but no manifest row — by id or by path)
  is forced into to_embed. Current-ver books keep their normal skips, so a cancelled run
  RESUMES (checkpoint batches commit ver-CURRENT rows). A MOVED legacy file cannot be
  force-embedded (its new path/id miss every membership test) — it remaps first, ver
  preserved, and re-chunks on the next armed run.
- **forced_count**: IndexPlan counts pending re-chunk work — forced embeds, forced twins
  deferred as in-run duplicates, and remaps of legacy-ver rows. IndexStats.forced carries
  it (sweep's inner re-plan zeroed before merge to avoid double counting).
- **Flag lifecycle**: both commands clear the flag only at the plan-time fixed point —
  a completed, uncancelled run with forced == 0. A run that forced work leaves the flag
  armed for one cheap confirming pass. The nudge's legacy_chunker_count is a DIFFERENT
  zero: residue after flag-clear means rows Index cannot fix (silenced skips, deleted
  files) — Maintenance territory.
- **Replace, never append**: per-path deletion sets from the UNCOLLAPSED store scan
  (book_path_pairs) — every id a path has ever had. CPU: delete_books before add_chunks.
  GPU: one delete_books per batch for sidecar-"indexed" files, after sidecar validation,
  before import_parquet (a failed import self-heals next run; bounded search gap for that
  batch only). Both commits also clear_book_state_by_path BEFORE the fresh write (the
  path-keyed delete would otherwise remove the new row) — kills the legacy-id manifest
  rows that would otherwise hold the legacy count above zero forever.
- plan-soak reads the flag: arming + soak previews the forced set without embedding.

---

## §17 addendum — citation-integrity metric + golden chapter Q/A (post-re-chunk)

The §5.5 metric predates the completed library re-chunk (v0.15.5: 764/765 books on the
current chunker; 97% of epub chunks and 96% of md chunks carry `chapter`; pdf carries
pages). Adversarially critiqued (2 lenses, 20 confirmed amendments); the critique's
central finding REFRAMES the live tier: a pure-Rust harness re-extracts with the same
code that produced the chunks, so it cannot measure real foliate-DOM jump success — it
measures **extractor determinism + store↔display integrity**, which is what it is
honestly named. True DOM-side measurement (headless webview / exported-DOM fixtures over
the actual foliate search) is a recorded follow-up, and thresholds stay UNFROZEN until
it exists.

Corrected ground truths (critique findings 5, 9): `clean_text` (dehyphenation) applies
to **pdf extraction only** — ebook/text chunk text is html2text/raw-scan output; and
`format_citation`'s `Ch. {chapter}` arm is format-independent (Epub only suppresses the
page arm). Known frontend divergences the report header must enumerate: foliate search
joins text nodes with NO separator (`strs.join('')` — `<br>`/block boundaries inflate a
text-side match); html2text decorations (bullets, 200-col wrap) exist on both proxy
sides but not in the DOM; foliate's collator is case/diacritic-insensitive while
normText is not; JS `\w` is ASCII-only so Cyrillic is never dehyphenated by the
frontend (a Unicode-`\w` Rust port would silently diverge on RU).

### 17.1 Live tier — `ls-cli cite-metric <app-dir> [coll] [--books B] [--per-book K]`

Store↔display integrity harness over the REAL store. New `Store::scan_chunks()` —
projection WITHOUT text/vector for pass 1 (metadata), then an id-IN-list fetch of text
for the sampled rows only (~300-500 MB avoided). Sampling: FNV-1a hash of chunk id XOR
fixed seed (no rand dep, survives lance compaction/version churn); stratified —
hash-select B books per (family × script[RU/EN by Cyrillic fraction]) stratum first,
then ≤K chunks per book; strata and per-book counts printed so bias is visible.

Matcher replicas ported VERBATIM from the frontend, with a fixture-string unit-test
table + cross-pin comments in BookReader.tsx (minimum anti-drift bar; gen-exts-style
codegen is the stretch goal):
- probe = `citeJump`'s exact selection incl. the `at=0` junk-prefix fallback;
- `normText` with ASCII-`\w` dehyphenation (JS parity) + an explicit lowercase step
  labeled "collator emulation";
- `normChapter` prefix-strip.

Outcomes per family:
- **book (epub/fb2/fb2.zip):** "direct" = normChapter label resolves against re-extracted
  labels AND probe occurs in the FIRST spine block-run carrying that label (frontend
  phase-1 is confined to one spine item; ls-extract carries labels across items —
  finding 3); "located" = probe anywhere; "miss-in-chapter" = label resolved, probe
  missing (frontend still lands the user in the chapter + overlay); "cold-miss" = neither.
  Chapterless chunks are their OWN row, excluded from the direct denominator. mobi/azw3
  → "unverifiable (extractor best-effort)" bucket, never "miss".
- **text (md/txt ONLY):** replicate renderRich+renderInline segmentation (~60-line pure
  port): strip block markers, split fragments at inline-markup boundaries, 60-char
  needle / 40-char match per fragment — raw-file matching is near-vacuous (finding 7).
  html/office/pages/djvu → "unverified-family" rows (different render pipelines).
- **pdf:** reframed as "stored page still contains the passage per lopdf" (integrity;
  the frontend does a page-ordinal scroll with no text match at open). "near ±1" =
  probe crossing a chunk's page boundary. lopdf-vs-pdfjs ordinal spot-check = follow-up.

Per-book extraction wrapped in spawn_blocking + timeout → explicit "extract-timeout"
outcome; `[n/N]` stderr progress (run_ingest style); no models, no network (guard: the
subcommand follows the maintenance/plan-soak pattern, never touches models_dir).
Report: per-stratum table + provisional (unfrozen) reference rates: book ≥80% direct &
≥95% located+; text ≥90% located; pdf ≥95% on-page — first run IS the baseline.

### 17.2 CI tier — golden chapter Q/A (extends golden_set.rs)

- Corpus chunker gains heading tracking: standalone heading paragraphs become the
  running `chapter` label and are CONSUMED (excluded from chunk text) — every existing
  chunk stays byte-identical, protecting the fragile within_top-3 rank assertions
  (verify pre/post ranks once before committing).
- `chapter` set on a fixture subset; 2 books flip to `Format::Epub` to also pin the
  no-page rendering (delete the stale "Pdf keeps citations page-shaped" comment; no page
  stamps on the Epub ones).
- `Case.expect_chapter: Option<&str>`: assert **at least one expected-book hit within
  FINAL_K** whose `chapter` contains the substring (case-insensitive) — the top-chunk
  variant is reranker-configuration-dependent (int8 vs f32 tie-breaks).
- ~6 new chapter-targeted cases (EN + RU). Existing cases untouched.
- New `crates/ls-index/tests/store_scan.rs`: dummy-vector chunks; scan_chunks projection
  + row shape + id-IN-list text fetch.

### 17.2b Baseline (2026-07-24, seed 0xC17E_0017, --books 12 --per-book 4)

- **Book/en (48):** Direct 8% · Located 60% · MissInChapter 2% · ColdMiss 17% ·
  ChapterlessMiss 4% · ExtractError 8%. The low Direct is dominated by label
  PROVENANCE drift, exactly as the critique predicted: stored chapters came
  from the GPU pipeline (fitz get_toc), the proxy re-extracts with rbook —
  different label strings fail norm_chapter equality and demote to Located.
  In the app, phase-2 whole-book search still lands ~68% (Direct+Located);
  ~23% would show the explicit-miss overlay. Which label set foliate's TOC
  actually matches is the DOM-side follow-up's first question.
- **Pdf (48+48):** Direct+Near 62% en / 56% ru; "page stamp exists, probe
  nowhere per lopdf" 29% en / 44% ru — fitz-extracted chunk text vs lopdf
  re-extraction drift (finding 6), worse for Cyrillic. The app's pdf open is
  a page-ordinal scroll, so user-visible jump health hinges on ordinal
  fidelity, not text match — unmeasured until the pdfjs spot-check.
- **Text/en (48):** Located 77% · Miss 23% — a REAL frontend gap (needle
  spanning inline markup / weaker md normalizer), the strongest candidate
  for an actual UX fix from this baseline. **FIXED in v0.16.1** (block-level
  textContent matching + heading-skip fallback needle): same sample re-ran at
  **Located 90% · Miss 10%**, meeting the reference rate.
- Extraction p50 374 ms · p95 2.9 s · max 3.4 s per book; no timeouts.

### 17.2c DOM-side subsample (2026-07-24, tools/dom-cite-harness) — THRESHOLDS FROZEN

Real store citations replayed through the ACTUAL matchers in a browser
(vendored foliate-js `view.search` for epubs; pdfjs page text for pdfs) —
the measurement §17.1's proxy cannot make:

- **epub (12 books × 4):** direct 25% · located 60% (**85% land**) ·
  miss-in-chapter 4% · cold-miss 10%. `chapterResolved` was false in ~10/12
  books: stored fitz TOC labels rarely equal foliate's parsed labels even
  after normChapter — the label drift, not search quality, caps "direct".
  Direct hits take 0.5–2 s; whole-book fallbacks 4–15 s (worst 44 s), which
  vindicates the async-fallback design. FOLLOW-UP (largest epub win, ~60%
  located→direct): reconcile chapter labels — either widen normChapter or
  re-stamp chapters from a foliate-compatible source at ingest.
- **pdf (8 books × 3):** **87.5% on-page**, 12.5% off-page — fitz page stamps
  align with pdfjs page ordinals; the §17.2b lopdf "probe nowhere" rows were
  the proxy's noise (lopdf text extraction), not real jump failures.

**v0.16.2 label reconciliation (same day):** tiered chapter→TOC matching
(exact → containment → token overlap; roman-numeral prefixes stripped) raised
label resolution 29%→85% of sampled books; depth-aware section-range search
(a Part label's scope runs to the next same-or-shallower TOC entry, cap 40
sections) converted the scoped misses: **direct 25% → 60%**, located 25%,
land rate 85% (median direct 0.6 s vs 4–15 s whole-book). Verified on the
DOM harness; probe verification + whole-book fallback make a wrong label
match harmless.

**Frozen thresholds** (DOM-side, re-run the harness to verify): epub ≥80%
land (direct+located), pdf ≥80% on-page. Proxy-side (`ls-cli cite-metric`)
stays a fast integrity/regression signal with unfrozen reference rates.

### 17.3 Explicitly out of scope (follow-ups, gated on measured rates)

- DOM-side jump measurement (headless webview over the real foliate search) — the true
  §5.5 metric; unblocking threshold freeze.
- Auto-highlight of citeText on PDF open; sharing one normalization helper across the
  three frontend paths; OCR for the scanned-PDF straggler.
