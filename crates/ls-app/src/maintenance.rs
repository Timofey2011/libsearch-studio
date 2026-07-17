//! §2.6 Maintenance: report and repair index/manifest debris for a collection.
//!
//! Four scans, all read-only; `apply` RE-DERIVES its targets at apply time
//! (it never trusts a client-held report, so stale reports can't delete the
//! wrong rows). Fixes touch index rows, book_state, and skip_state only —
//! NEVER source files.
//!
//! Removal recipe (shared by orphans and duplicate-variant losers — the full
//! `reindex_book` recipe minus the re-embed): for a removed path P, delete
//! store rows under EVERY id holding P, then `clear_book_state_by_path`
//! (path-keyed across all id schemes — a surviving manifest row whose
//! fingerprint later matches a restored file would silently shadow it from
//! retrieval forever), then erase P's skip records.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use ls_index::Store;
use serde::Serialize;

use crate::discover::{variant_key, variant_rank};
use crate::store::path_under_roots;
use crate::types::Collection;
use crate::Db;

#[derive(Debug, Clone, Serialize)]
pub struct OrphanItem {
    pub book_id: String,
    pub path: String,
    /// "store" (chunks exist, file gone) or "manifest" (book_state only).
    pub kind: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StampItem {
    pub book_id: String,
    pub path: String,
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DupItem {
    pub keep_path: String,
    pub remove: Vec<RemoveItem>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RemoveItem {
    pub book_id: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MultiIdItem {
    pub path: String,
    pub keep_id: String,
    pub remove_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UnreachableRoot {
    pub root: String,
    pub affected: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct MaintenanceReport {
    pub orphans: Vec<OrphanItem>,
    pub bad_stamps: Vec<StampItem>,
    pub dup_variants: Vec<DupItem>,
    pub multi_id: Vec<MultiIdItem>,
    /// Source roots that are currently unreachable (unmounted drive, offline
    /// share): rows under them are NOT classified as orphans this scan.
    pub unreachable_roots: Vec<UnreachableRoot>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FixOutcome {
    pub orphans_removed: usize,
    pub restamped: usize,
    pub dup_rows_removed: usize,
    pub multi_id_removed: usize,
}

fn exists(memo: &mut HashMap<String, bool>, path: &str) -> bool {
    *memo
        .entry(path.to_string())
        .or_insert_with(|| std::fs::metadata(path).is_ok())
}

pub async fn scan(store: &Store, db: &Db, coll: &Collection) -> Result<MaintenanceReport, String> {
    let mut report = MaintenanceReport::default();
    let mut fs_memo: HashMap<String, bool> = HashMap::new();

    // Root reachability gate: rows under an unmounted root are unjudgeable.
    // A missing file is an orphan only when the claim is trustworthy: rows
    // under an UNREACHABLE root (unmounted drive) are unjudgeable this scan.
    let unreachable: Vec<String> = coll
        .source_paths
        .iter()
        .filter(|r| std::fs::metadata(r).is_err())
        .cloned()
        .collect();

    let pairs = store.book_path_pairs().await.map_err(|e| e.to_string())?;
    let mut ids_by_path: BTreeMap<String, HashSet<String>> = BTreeMap::new();
    for (id, p) in &pairs {
        ids_by_path.entry(p.clone()).or_default().insert(id.clone());
    }

    let mut affected_by_root: BTreeMap<String, usize> = BTreeMap::new();

    // 1. STORE ORPHANS — chunks whose file is gone (dehydrated placeholders
    // pass fs::metadata and are NOT orphans).
    for (path, ids) in &ids_by_path {
        if exists(&mut fs_memo, path) {
            continue;
        }
        if path_under_roots(path, &unreachable) {
            for r in &unreachable {
                if path_under_roots(path, std::slice::from_ref(r)) {
                    *affected_by_root.entry(r.clone()).or_default() += 1;
                }
            }
            continue;
        }
        for id in ids {
            report.orphans.push(OrphanItem {
                book_id: id.clone(),
                path: path.clone(),
                kind: "store".into(),
            });
        }
    }

    // 2. MANIFEST ORPHANS — book_state rows with no store presence whose path
    // is empty (pre-M0a rows) or gone. Rows with store presence belong to
    // scan 1; rows whose file still exists are deliberately left alone (their
    // removal would trigger re-embedding — not this panel's job).
    let store_ids = store.indexed_book_ids().await.unwrap_or_default();
    for (book_id, path) in db.book_state_rows(&coll.id).map_err(|e| e.to_string())? {
        if store_ids.contains(&book_id) {
            continue;
        }
        if path.is_empty() {
            report.orphans.push(OrphanItem {
                book_id,
                path,
                kind: "manifest".into(),
            });
            continue;
        }
        if exists(&mut fs_memo, &path) {
            continue;
        }
        if path_under_roots(&path, &unreachable) {
            continue; // counted (if at all) by the store half; unjudgeable here
        }
        report.orphans.push(OrphanItem {
            book_id,
            path,
            kind: "manifest".into(),
        });
    }

    // 3. FORMAT STAMPS — family-level comparison on the RAW stored string, so
    // alias stamps ("markdown" on .markdown) are left alone and unparseable
    // stamps are flagged rather than silently skipped.
    for (book_id, path, raw) in store.book_formats().await.map_err(|e| e.to_string())? {
        let Some(expected) = ls_core::Format::from_path(&path) else {
            continue; // unknown extension: nothing to restamp to
        };
        if ls_core::Format::from_ext(&raw) != Some(expected) {
            report.bad_stamps.push(StampItem {
                book_id,
                path,
                from: raw,
                to: expected.as_str().to_string(),
            });
        }
    }

    // 4. DUPLICATE VARIANTS — two-level grouping: variant_key → distinct
    // ranked ON-DISK paths (one path can sit under several ids). Keeper is
    // the best-ranked path that still exists; missing losers are scan 1's job.
    let mut groups: BTreeMap<(String, String), Vec<(String, u8)>> = BTreeMap::new();
    for path in ids_by_path.keys() {
        let (Some(rank), Some(key)) = (variant_rank(path), variant_key(path)) else {
            continue;
        };
        if !exists(&mut fs_memo, path) || path_under_roots(path, &unreachable) {
            continue;
        }
        groups.entry(key).or_default().push((path.clone(), rank));
    }
    for (_, mut members) in groups {
        members.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        members.dedup_by(|a, b| a.0 == b.0);
        if members.len() < 2 {
            continue;
        }
        let keep_path = members[0].0.clone();
        let mut remove = Vec::new();
        for (path, _) in members.into_iter().skip(1) {
            for id in ids_by_path.get(&path).into_iter().flatten() {
                remove.push(RemoveItem {
                    book_id: id.clone(),
                    path: path.clone(),
                });
            }
        }
        report.dup_variants.push(DupItem { keep_path, remove });
    }

    // 5. SAME-PATH MULTI-ID — one on-disk file indexed under several id
    // schemes (double retrieval hits). Keeper preference: the manifest's id
    // for the path, else the path-derived id, else the smallest.
    for (path, ids) in &ids_by_path {
        if ids.len() < 2 || !exists(&mut fs_memo, path) {
            continue;
        }
        let manifest_id = db
            .book_id_for_source_path(&coll.id, path)
            .ok()
            .flatten()
            .filter(|id| ids.contains(id));
        let stable = crate::stable_book_id(Path::new(path));
        let keep_id = manifest_id
            .or_else(|| ids.contains(&stable).then_some(stable))
            .or_else(|| ids.iter().min().cloned())
            .unwrap_or_default();
        let remove_ids: Vec<String> = ids.iter().filter(|i| **i != keep_id).cloned().collect();
        report.multi_id.push(MultiIdItem {
            path: path.clone(),
            keep_id,
            remove_ids,
        });
    }

    report.unreachable_roots = affected_by_root
        .into_iter()
        .map(|(root, affected)| UnreachableRoot { root, affected })
        .collect();
    for r in &unreachable {
        if !report.unreachable_roots.iter().any(|u| &u.root == r) {
            report.unreachable_roots.push(UnreachableRoot {
                root: r.clone(),
                affected: 0,
            });
        }
    }
    Ok(report)
}

/// Remove path P everywhere: store rows under every id holding P, the
/// manifest across all id schemes (path-keyed), and its skip records.
async fn remove_path(
    store: &Store,
    db: &Db,
    coll_id: &str,
    path: &str,
    ids: &HashSet<String>,
) -> Result<usize, String> {
    let id_vec: Vec<String> = ids.iter().cloned().collect();
    store
        .delete_books(&id_vec)
        .await
        .map_err(|e| e.to_string())?;
    let _ = db.clear_book_state_by_path(coll_id, path, &crate::stable_book_id(Path::new(path)));
    let _ = db.erase_skips(coll_id, path);
    Ok(ids.len().max(1))
}

pub async fn apply(
    store: &Store,
    db: &Db,
    coll: &Collection,
    fix_orphans: bool,
    fix_stamps: bool,
    fix_dups: bool,
    fix_multi: bool,
) -> Result<FixOutcome, String> {
    // Never trust a client report: re-derive everything now.
    let report = scan(store, db, coll).await?;
    let mut out = FixOutcome::default();

    let pairs = store.book_path_pairs().await.map_err(|e| e.to_string())?;
    let mut ids_by_path: HashMap<String, HashSet<String>> = HashMap::new();
    for (id, p) in &pairs {
        ids_by_path.entry(p.clone()).or_default().insert(id.clone());
    }

    if fix_orphans {
        let mut done_paths: HashSet<String> = HashSet::new();
        for o in &report.orphans {
            match o.kind.as_str() {
                "store" => {
                    if done_paths.insert(o.path.clone()) {
                        let ids = ids_by_path.get(&o.path).cloned().unwrap_or_default();
                        out.orphans_removed +=
                            remove_path(store, db, &coll.id, &o.path, &ids).await?;
                    }
                }
                _ => {
                    if o.path.is_empty() {
                        // Row-precise: NEVER path-keyed with an empty path (its
                        // OR-predicate would sweep every pre-M0a row).
                        let _ = db.delete_book_state(&coll.id, &o.book_id);
                    } else if done_paths.insert(o.path.clone()) {
                        let _ = db.clear_book_state_by_path(
                            &coll.id,
                            &o.path,
                            &crate::stable_book_id(Path::new(&o.path)),
                        );
                        let _ = db.erase_skips(&coll.id, &o.path);
                    }
                    out.orphans_removed += 1;
                }
            }
        }
    }

    if fix_dups {
        for group in &report.dup_variants {
            let mut done_paths: HashSet<String> = HashSet::new();
            for r in &group.remove {
                if done_paths.insert(r.path.clone()) {
                    let ids = ids_by_path.get(&r.path).cloned().unwrap_or_default();
                    out.dup_rows_removed += remove_path(store, db, &coll.id, &r.path, &ids).await?;
                }
            }
        }
    }

    if fix_multi {
        for m in &report.multi_id {
            store
                .delete_books(&m.remove_ids)
                .await
                .map_err(|e| e.to_string())?;
            for id in &m.remove_ids {
                // Row-precise: the keeper's manifest row must survive.
                let _ = db.delete_book_state(&coll.id, id);
            }
            out.multi_id_removed += m.remove_ids.len();
        }
    }

    if fix_stamps {
        let mut by_target: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for s in &report.bad_stamps {
            by_target
                .entry(s.to.clone())
                .or_default()
                .push(s.book_id.clone());
        }
        for (to, ids) in by_target {
            store
                .restamp_formats(&ids, &to)
                .await
                .map_err(|e| e.to_string())?;
            out.restamped += ids.len();
        }
    }

    // Updates/deletes write new fragments outside the FTS index (flat-scanned
    // on every query until rebuilt) — rebuild once, then compact.
    if out.orphans_removed + out.restamped + out.dup_rows_removed + out.multi_id_removed > 0 {
        store.ensure_fts_index().await.map_err(|e| e.to_string())?;
        let _ = store.optimize().await;
    }
    Ok(out)
}
