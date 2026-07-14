//! M0a acceptance tests (ROADMAP-3 §2.1): the four legacy-library guard
//! scenarios, exercised against the shared `plan_index_run` pre-filter with a
//! fixture manifest Db. "Legacy" rows simulate the ~300 books imported from the
//! old Python pipeline: their `book_id` differs from the path-derived id and
//! (initially) their `source_path` column is empty.
//!
//! The GPU-side "0 embeds" assertion is `plan.to_embed` emptiness — the plan
//! boundary is where embedding is decided; the Python helper never sees a file
//! the plan excluded. Injectable fp/csig closures simulate unreadable files and
//! count 512 KiB signature reads without touching a real filesystem.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use ls_app::{plan_index_run, Db, PlanCtx, SkipReason};

/// A fixture world: manifest Db + lance-store scan results + per-path fp/csig.
struct World {
    db: Db,
    coll: String,
    indexed_ids: HashSet<String>,
    paths_by_id: HashMap<String, String>,
    fps: HashMap<String, String>,
    csigs: HashMap<String, String>,
    caps_ver: String,
}

impl World {
    fn new() -> Self {
        Self {
            db: Db::open_in_memory().unwrap(),
            coll: "c1".into(),
            indexed_ids: HashSet::new(),
            paths_by_id: HashMap::new(),
            fps: HashMap::new(),
            csigs: HashMap::new(),
            caps_ver: "caps-v1".into(),
        }
    }

    /// A legacy imported book: chunks in the store under a NON-path-derived id,
    /// manifest row seeded by the old backfill (ver 0, empty source_path
    /// column — it predates the column).
    fn add_legacy_book(&mut self, path: &str, legacy_id: &str, fp: &str, csig: &str) {
        self.indexed_ids.insert(legacy_id.to_string());
        self.paths_by_id
            .insert(legacy_id.to_string(), path.to_string());
        self.db
            .set_book_state_ver(&self.coll, legacy_id, fp, csig, "", 0)
            .unwrap();
        self.fps.insert(path.to_string(), fp.to_string());
        self.csigs.insert(path.to_string(), csig.to_string());
    }

    /// A brand-new candidate file on disk (not in store, no manifest row).
    fn add_new_file(&mut self, path: &str, fp: &str, csig: &str) {
        self.fps.insert(path.to_string(), fp.to_string());
        self.csigs.insert(path.to_string(), csig.to_string());
    }

    fn plan(&self, candidates: &[&str]) -> ls_app::IndexPlan {
        self.plan_counting(candidates).0
    }

    /// Also returns how many candidate content-signature reads the plan needed
    /// (scenario (b) asserts the second run does none).
    fn plan_counting(&self, candidates: &[&str]) -> (ls_app::IndexPlan, usize) {
        let cands: Vec<PathBuf> = candidates.iter().map(PathBuf::from).collect();
        let csig_reads = Cell::new(0usize);
        let plan = {
            let fp_fn = |p: &std::path::Path| {
                self.fps
                    .get(&p.to_string_lossy().into_owned())
                    .cloned()
                    .unwrap_or_else(|| "missing".into())
            };
            let csig_fn = |p: &std::path::Path| {
                csig_reads.set(csig_reads.get() + 1);
                self.csigs
                    .get(&p.to_string_lossy().into_owned())
                    .cloned()
                    .unwrap_or_else(|| "missing".into())
            };
            let ctx = PlanCtx {
                collection_id: &self.coll,
                db: &self.db,
                indexed_ids: &self.indexed_ids,
                paths_by_id: &self.paths_by_id,
                pipeline: "cpu",
                caps_ver: &self.caps_ver,
                fp_fn: &fp_fn,
                csig_fn: &csig_fn,
            };
            plan_index_run(&cands, &ctx).unwrap()
        };
        (plan, csig_reads.get())
    }

    /// Apply the plan's metadata actions the way both pipelines do.
    fn apply(&mut self, plan: &ls_app::IndexPlan) {
        for r in &plan.state_refreshes {
            self.db
                .refresh_book_state(
                    &self.coll,
                    &r.book_id,
                    &r.fingerprint,
                    r.content_sig.as_deref(),
                    &r.path,
                )
                .unwrap();
        }
        for m in &plan.remaps {
            // (store.remap_book is a lance write — out of scope here)
            self.db.delete_book_state(&self.coll, &m.old_id).unwrap();
            self.db
                .set_book_state_ver(
                    &self.coll,
                    &m.new_id,
                    &m.fingerprint,
                    m.content_sig.as_deref().unwrap_or(""),
                    &m.path,
                    m.chunker_ver,
                )
                .unwrap();
        }
        // Embeds would stamp state after a successful embed:
        for it in &plan.to_embed {
            self.db
                .set_book_state(
                    &self.coll,
                    &ls_app::stable_book_id(std::path::Path::new(&it.path)),
                    &it.fingerprint,
                    &it.content_sig,
                    &it.path,
                )
                .unwrap();
            self.indexed_ids
                .insert(ls_app::stable_book_id(std::path::Path::new(&it.path)));
            self.paths_by_id.insert(
                ls_app::stable_book_id(std::path::Path::new(&it.path)),
                it.path.clone(),
            );
        }
    }
}

/// (a) Baseline post-upgrade run over unmoved legacy books: zero embeds, zero
/// remaps — the path-equality short-circuit recognizes the same file under a
/// different id scheme.
#[test]
fn legacy_books_never_reembed_or_remap() {
    let mut w = World::new();
    for i in 0..50 {
        let path = format!("/lib/book{i}.md");
        w.add_legacy_book(
            &path,
            &format!("legacy-{i}"),
            &format!("100{i}:5000"),
            &format!("aa{i}:bb"),
        );
    }
    let candidates: Vec<String> = (0..50).map(|i| format!("/lib/book{i}.md")).collect();
    let cand_refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();

    let plan = w.plan(&cand_refs);
    assert!(plan.to_embed.is_empty(), "legacy books must not re-embed");
    assert!(
        plan.remaps.is_empty(),
        "unmoved legacy books must not remap"
    );
    assert!(plan
        .preskips
        .iter()
        .all(|(_, r)| *r == SkipReason::Unchanged));
    assert_eq!(plan.preskips.len(), 50);
    // The only writes are source_path column backfills for rows that predate
    // the column — never fingerprint/csig changes.
    for r in &plan.state_refreshes {
        assert!(r.content_sig.is_none());
    }

    // Second run after applying: still nothing to do, and now zero refreshes.
    w.apply(&plan);
    let plan2 = w.plan(&cand_refs);
    assert!(plan2.to_embed.is_empty());
    assert!(plan2.remaps.is_empty());
    assert!(
        plan2.state_refreshes.is_empty(),
        "second run must be pure skips"
    );
}

/// (b) Dropbox mtime churn on a legacy file: stage-2 misses (stale fp), the
/// content-signature stage hits with path equality → fingerprint refreshed
/// under the EXISTING id, no remap, no embed; the SECOND run short-circuits at
/// stage 2 without any 512 KiB signature read.
#[test]
fn mtime_churn_refreshes_fingerprint_without_remap() {
    let mut w = World::new();
    w.add_legacy_book("/lib/churned.pdf", "legacy-x", "999:1111", "cs:xx");
    // Dropbox re-stamped the file: new mtime → new fp, same bytes → same csig.
    w.fps.insert("/lib/churned.pdf".into(), "999:2222".into());

    let (plan, csig_reads) = w.plan_counting(&["/lib/churned.pdf"]);
    assert!(plan.to_embed.is_empty());
    assert!(plan.remaps.is_empty(), "churn is not a move");
    assert_eq!(csig_reads, 1, "churn costs one signature read");
    assert_eq!(plan.state_refreshes.len(), 1);
    let r = &plan.state_refreshes[0];
    assert_eq!(r.book_id, "legacy-x", "refresh under the EXISTING id");
    assert_eq!(r.fingerprint, "999:2222");

    // Run 2: the refreshed fingerprint short-circuits at stage 2 — zero
    // signature reads this time.
    w.apply(&plan);
    let (plan2, csig_reads2) = w.plan_counting(&["/lib/churned.pdf"]);
    assert!(plan2.to_embed.is_empty());
    assert!(plan2.remaps.is_empty());
    assert_eq!(csig_reads2, 0, "second run must not re-read 512 KiB");
}

/// (c) Fingerprint collision: two distinct files sharing size+mtime must both
/// index under their own ids — no remap ping-pong across two consecutive runs.
#[test]
fn fingerprint_collision_never_remaps() {
    let mut w = World::new();
    // Batch-exported notes: same byte size, same-second mtime, different text.
    w.add_new_file("/notes/a.md", "500:7777", "cs:aaaa");
    w.add_new_file("/notes/b.md", "500:7777", "cs:bbbb");

    let plan = w.plan(&["/notes/a.md", "/notes/b.md"]);
    assert_eq!(plan.to_embed.len(), 2, "both files are genuinely new");
    assert!(plan.remaps.is_empty());

    // After indexing both, a second run: file b hits a's row by fingerprint
    // (LIMIT 1) — the csig confirmation must reject the remap and stage-1/3
    // keep both untouched.
    w.apply(&plan);
    let (plan2, _) = w.plan_counting(&["/notes/a.md", "/notes/b.md"]);
    assert!(plan2.to_embed.is_empty(), "no re-embeds on run 2");
    assert!(plan2.remaps.is_empty(), "no ping-pong remap");
    // Run 3 (the ping-pong would show here if state was corrupted):
    w.apply(&plan2);
    let (plan3, _) = w.plan_counting(&["/notes/a.md", "/notes/b.md"]);
    assert!(plan3.to_embed.is_empty());
    assert!(plan3.remaps.is_empty());
}

/// (d) TWO unreadable candidates in one run: each gets its own Unreadable skip
/// (the second must not be silently swallowed by the in-run content map), no
/// state rows are written, and no existing book is remapped.
#[test]
fn two_unreadable_files_two_events_no_state() {
    let mut w = World::new();
    w.add_legacy_book("/lib/good.pdf", "legacy-g", "10:10", "cs:good");
    // Two dehydrated placeholders: stat works (fp real) but reads fail
    // (csig sentinel) — the nastier variant of unreadable.
    w.add_new_file("/lib/ghost1.pdf", "77:77", "missing");
    w.add_new_file("/lib/ghost2.pdf", "77:77", "missing");

    let plan = w.plan(&["/lib/ghost1.pdf", "/lib/ghost2.pdf", "/lib/good.pdf"]);
    let unreadable: Vec<_> = plan
        .preskips
        .iter()
        .filter(|(_, r)| *r == SkipReason::Unreadable)
        .collect();
    assert_eq!(unreadable.len(), 2, "one event per unreadable file");
    assert!(plan.to_embed.is_empty());
    assert!(
        plan.remaps.is_empty(),
        "sentinels must never remap anything"
    );

    // And fully-unstattable files (fp sentinel) behave the same:
    w.fps.insert("/lib/ghost1.pdf".into(), "missing".into());
    let plan2 = w.plan(&["/lib/ghost1.pdf"]);
    assert_eq!(
        plan2.preskips,
        vec![("/lib/ghost1.pdf".to_string(), SkipReason::Unreadable)]
    );

    // Sentinel rows can never be persisted even if a caller tries:
    w.db.set_book_state(&w.coll, "evil", "missing", "missing", "/x")
        .unwrap();
    assert!(w
        .db
        .book_state_for_fingerprint(&w.coll, "missing")
        .unwrap()
        .is_none());
}

/// In-run duplicates (same bytes queued twice) embed once — and sentinel csigs
/// never join that dedup map (covered in (d) via distinct Unreadable skips).
#[test]
fn in_run_duplicate_embeds_once() {
    let mut w = World::new();
    w.add_new_file("/lib/copy1.pdf", "31:31", "cs:same");
    w.add_new_file("/lib/copy2.pdf", "32:32", "cs:same");
    let plan = w.plan(&["/lib/copy1.pdf", "/lib/copy2.pdf"]);
    assert_eq!(plan.to_embed.len(), 1);
    assert_eq!(
        plan.preskips,
        vec![("/lib/copy2.pdf".to_string(), SkipReason::DuplicateInRun)]
    );
}

/// A genuine move (new path, same fp, same csig) DOES remap — the guards must
/// not break the correct moved-file behavior — and preserves chunker_ver.
#[test]
fn genuine_move_still_remaps_preserving_ver() {
    let mut w = World::new();
    w.add_legacy_book("/old/book.pdf", "legacy-m", "44:44", "cs:mv");
    // The file moved: same fp + csig now live at a new path.
    w.fps.remove("/old/book.pdf");
    w.fps.insert("/new/book.pdf".into(), "44:44".into());
    w.csigs.insert("/new/book.pdf".into(), "cs:mv".into());
    // Old path is gone from disk; store scan still holds the old path.
    let plan = w.plan(&["/new/book.pdf"]);
    assert!(plan.to_embed.is_empty(), "moved book must not re-embed");
    assert_eq!(plan.remaps.len(), 1);
    let m = &plan.remaps[0];
    assert_eq!(m.old_id, "legacy-m");
    assert_eq!(m.path, "/new/book.pdf");
    assert_eq!(m.chunker_ver, 0, "legacy ver survives the move");
}

/// §2.8 skip records: silenced while file + capabilities are unchanged;
/// retried when either changes; pipeline-scoped; never identity-keyed.
#[test]
fn skip_state_silences_retries_and_scopes() {
    let mut w = World::new();
    w.add_new_file("/lib/broken.pdf", "60:60", "cs:broke");
    // First run recorded a CPU skip (e.g. extract failure).
    w.db.upsert_skip(
        &w.coll,
        "/lib/broken.pdf",
        "cpu",
        "60:60",
        "boom",
        "caps-v1",
    )
    .unwrap();

    // Same file, same caps → silenced (no embed, no re-announcement).
    let plan = w.plan(&["/lib/broken.pdf"]);
    assert_eq!(
        plan.preskips,
        vec![("/lib/broken.pdf".to_string(), SkipReason::Silenced)]
    );
    assert!(plan.to_embed.is_empty());

    // Capabilities changed (tool installed / device switched) → retried.
    w.caps_ver = "caps-v2".into();
    let plan = w.plan(&["/lib/broken.pdf"]);
    assert_eq!(plan.to_embed.len(), 1, "caps bump must retry the skip");

    // File changed (new fingerprint) under the old caps → retried too.
    w.caps_ver = "caps-v1".into();
    w.fps.insert("/lib/broken.pdf".into(), "61:61".into());
    let plan = w.plan(&["/lib/broken.pdf"]);
    assert_eq!(plan.to_embed.len(), 1, "changed file must retry the skip");

    // Pipeline scoping: a GPU-recorded skip must not hide the file from CPU.
    w.fps.insert("/lib/broken.pdf".into(), "60:60".into());
    w.db.erase_skips(&w.coll, "/lib/broken.pdf").unwrap();
    w.db.upsert_skip(
        &w.coll,
        "/lib/broken.pdf",
        "gpu",
        "60:60",
        "no dep",
        "gcaps",
    )
    .unwrap();
    let plan = w.plan(&["/lib/broken.pdf"]); // pipeline = cpu
    assert_eq!(plan.to_embed.len(), 1, "gpu skip must not silence cpu");
}

/// Skip-collision (mirrors §2.1 scenario (c)): a skip row for file A must not
/// suppress file B with a colliding fingerprint, and B's success must not
/// erase A's record.
#[test]
fn skip_rows_are_path_keyed_not_fingerprint_keyed() {
    let mut w = World::new();
    w.add_new_file("/notes/a.md", "500:7777", "cs:aa");
    w.add_new_file("/notes/b.md", "500:7777", "cs:bb"); // same size:mtime
    w.db.upsert_skip(
        &w.coll,
        "/notes/a.md",
        "cpu",
        "500:7777",
        "no text",
        "caps-v1",
    )
    .unwrap();

    let plan = w.plan(&["/notes/a.md", "/notes/b.md"]);
    // A silenced; B (no row for ITS path) indexes normally.
    assert_eq!(plan.to_embed.len(), 1);
    assert_eq!(plan.to_embed[0].path, "/notes/b.md");

    // B's success erases only B's rows; A stays silenced next run.
    w.db.erase_skips(&w.coll, "/notes/b.md").unwrap();
    let plan2 = w.plan(&["/notes/a.md"]);
    assert_eq!(
        plan2.preskips,
        vec![("/notes/a.md".to_string(), SkipReason::Silenced)]
    );
}

/// Stage-2 safety: skip records live in their own table and must never be
/// found by the book_state reverse lookups.
#[test]
fn skip_rows_never_reach_dedup_lookups() {
    let w = World::new();
    w.db.upsert_skip(&w.coll, "/lib/x.pdf", "cpu", "42:42", "r", "c")
        .unwrap();
    assert!(w
        .db
        .book_state_for_fingerprint(&w.coll, "42:42")
        .unwrap()
        .is_none());
    // Sentinel skip rows are refused outright.
    w.db.upsert_skip(&w.coll, "/lib/y.pdf", "cpu", "missing", "r", "c")
        .unwrap();
    assert!(w
        .db
        .skip_state_hit(&w.coll, "/lib/y.pdf", "cpu")
        .unwrap()
        .is_none());
}

/// Orphan GC: rows whose path left the collection's sources are swept.
#[test]
fn skip_gc_sweeps_paths_outside_sources() {
    let w = World::new();
    w.db.upsert_skip(&w.coll, "/lib/in.pdf", "cpu", "1:1", "r", "c")
        .unwrap();
    w.db.upsert_skip(&w.coll, "/elsewhere/out.pdf", "gpu", "2:2", "r", "c")
        .unwrap();
    let swept = w.db.gc_skips(&w.coll, &["/lib".to_string()]).unwrap();
    assert_eq!(swept, 1);
    assert!(w
        .db
        .skip_state_hit(&w.coll, "/lib/in.pdf", "cpu")
        .unwrap()
        .is_some());
    assert!(w
        .db
        .skip_state_hit(&w.coll, "/elsewhere/out.pdf", "gpu")
        .unwrap()
        .is_none());
}
