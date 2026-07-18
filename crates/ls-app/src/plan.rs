//! The shared dedup pre-filter for BOTH ingest pipelines (CPU `index_collection`
//! and the GPU fast-index command). Pure: reads the manifest Db, WRITES NOTHING —
//! every store/db mutation is returned as an action for the caller to apply, so
//! all guard scenarios are plain Rust tests against a fixture Db.
//!
//! The four stages, per candidate file:
//!   1. fingerprint unchanged under the path-derived id       → skip
//!   2. fingerprint reverse lookup (path-independent)         → skip / remap
//!   3. already present in the lance store (by id OR by path) → refresh + skip
//!   4. content-signature reverse lookup + in-run dedup       → skip / remap
//!
//! Anything not skipped/remapped by a stage is embedded.
//!
//! Guards encoded here (ROADMAP-3 §2.1.1–§2.1.3):
//! - Failure sentinels ("missing"/'') are poison, never identity: an unreadable
//!   file is skipped with its own event, joins no lookup and no in-run map.
//! - Path-equality short-circuit: a reverse-lookup hit whose stored path equals
//!   the candidate is the SAME book under a different id scheme (legacy import)
//!   or metadata churn — plain skip, never a remap.
//! - Fingerprint-collision confirmation: a stage-2 hit with a DIFFERENT path
//!   must prove identity via content signature before remapping — size:mtime
//!   collides for real once thousands of small files are indexed.
//! - Stage-4 path-equality refreshes the fingerprint under the EXISTING id so
//!   the next run short-circuits at stage 2 instead of re-reading 512 KiB.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::store::Db;
use crate::{is_sig_sentinel, DbError};

/// Why a candidate was not embedded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// Fingerprint/store says this exact book is already indexed.
    Unchanged,
    /// The file could not be read (stat/open/short read) — no state written.
    Unreadable,
    /// Identical content already queued earlier in this same run.
    DuplicateInRun,
    /// A recorded skip (same file, same pipeline capabilities) — stage 0.5.
    /// Silent by contract (§2.8): counted, never re-announced per run.
    Silenced,
}

/// A moved file: re-point existing chunks instead of re-embedding.
#[derive(Debug, Clone)]
pub struct RemapAction {
    pub old_id: String,
    pub new_id: String,
    pub path: String,
    pub fingerprint: String,
    /// Present when the guard computed/knows the content signature.
    pub content_sig: Option<String>,
    /// The moved row's chunker version, preserved so a legacy book keeps its
    /// honest re-index nudge after the move.
    pub chunker_ver: i64,
}

/// Refresh manifest state under an EXISTING book id (no store rewrite).
#[derive(Debug, Clone)]
pub struct StateRefresh {
    pub book_id: String,
    pub path: String,
    pub fingerprint: String,
    /// None = keep whatever signature the row already has ('' for none).
    pub content_sig: Option<String>,
}

/// A file the caller must extract + embed; fp/csig captured at plan time so the
/// caller can stamp state after a successful embed without recomputing.
#[derive(Debug, Clone)]
pub struct EmbedItem {
    pub path: String,
    pub fingerprint: String,
    pub content_sig: String,
}

#[derive(Debug, Default)]
pub struct IndexPlan {
    pub to_embed: Vec<EmbedItem>,
    pub preskips: Vec<(String, SkipReason)>,
    pub remaps: Vec<RemapAction>,
    pub state_refreshes: Vec<StateRefresh>,
    /// Re-chunk work this plan still carries: forced embeds, forced twins
    /// deferred as in-run duplicates, and remaps of legacy-version rows (the
    /// moved book re-chunks NEXT run, after the re-point). The commands clear
    /// the armed flag only when a completed run planned ZERO of these — the
    /// fixed point where re-running would force nothing.
    pub forced_count: usize,
}

/// Planner context: everything the pure pre-filter needs from the world.
/// `fp_fn`/`csig_fn` are injectable so tests can simulate unreadable files and
/// count 512 KiB signature reads without touching a real filesystem.
pub struct PlanCtx<'a> {
    pub collection_id: &'a str,
    pub db: &'a Db,
    /// Book ids present in the lance store (one scan per run).
    pub indexed_ids: &'a HashSet<String>,
    /// `book_id -> source_path` from the same scan; powers both the §2.1.3
    /// path-membership check and the empty-`source_path` row fallback.
    pub paths_by_id: &'a HashMap<String, String>,
    /// 'cpu' | 'gpu' — skip records are pipeline-scoped (§2.8): one pipeline's
    /// inability must never hide a file from the other.
    pub pipeline: &'a str,
    /// This run's capabilities hash for this pipeline; a recorded skip is
    /// honored only while it still matches (tool installs/device changes
    /// retry past old skips).
    pub caps_ver: &'a str,
    pub fp_fn: &'a dyn Fn(&Path) -> String,
    pub csig_fn: &'a dyn Fn(&Path) -> String,
    /// Re-chunk mode (v0.15): books whose chunks predate CURRENT_CHUNKER_VER
    /// bypass the unchanged-skip guards into the embed queue.
    pub rechunk: bool,
}

impl PlanCtx<'_> {
    /// Stored path for a hit row: the column when present, else the store scan
    /// (rows that predate the column), else empty (unknown).
    fn hit_path(&self, hit: &crate::BookStateHit) -> String {
        if !hit.source_path.is_empty() {
            return hit.source_path.clone();
        }
        self.paths_by_id
            .get(&hit.book_id)
            .cloned()
            .unwrap_or_default()
    }
}

/// Two paths refer to the same file. Candidate paths come from discovery and
/// stored paths from prior runs of the same discovery, so string equality is
/// the common case; canonicalization covers `..`/symlink variants when both
/// still exist (best-effort — a missing stored path just compares unequal).
fn same_file(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    match (std::fs::canonicalize(a).ok(), std::fs::canonicalize(b).ok()) {
        (Some(ca), Some(cb)) => ca == cb,
        _ => false,
    }
}

pub fn plan_index_run(candidates: &[PathBuf], ctx: &PlanCtx) -> Result<IndexPlan, DbError> {
    let mut plan = IndexPlan::default();
    // Distinct store paths for the §2.1.3 membership check (seed-failure and
    // row-loss belt-and-braces).
    let indexed_paths: HashSet<&str> = ctx.paths_by_id.values().map(|s| s.as_str()).collect();
    // Same content queued earlier this run → embed once. NEVER holds sentinels.
    let mut seen_content: HashMap<String, String> = HashMap::new();

    for cand in candidates {
        let path_str = cand.to_string_lossy().into_owned();
        let book_id = ls_extract::stable_book_id(cand);

        let fp = (ctx.fp_fn)(cand);
        if is_sig_sentinel(&fp) {
            plan.preskips.push((path_str, SkipReason::Unreadable));
            continue;
        }

        // Stage 0.5: previously skipped by THIS pipeline, file unchanged AND
        // capabilities unchanged → silent short-circuit (no re-announcement).
        // Any mismatch (file changed, tools/deps/device changed) falls
        // through: the skip is retried and its row overwritten by the retry's
        // outcome.
        if let Some((skip_fp, skip_caps)) =
            ctx.db
                .skip_state_hit(ctx.collection_id, &path_str, ctx.pipeline)?
        {
            if skip_fp == fp && skip_caps == ctx.caps_ver {
                plan.preskips.push((path_str, SkipReason::Silenced));
                continue;
            }
        }

        // Re-chunk mode (v0.15): a book whose chunks predate the current
        // chunker is forced into the embed queue even though nothing about
        // the file changed. Books already at CURRENT_CHUNKER_VER keep their
        // normal skips — a cancelled re-chunk therefore RESUMES (each
        // checkpointed batch commits ver-CURRENT rows). A MOVED legacy file
        // never lands here: its new path/id miss every membership test below,
        // so it falls through to the normal stage-2/-4 remap (re-point first,
        // re-chunk on the next armed run — counted below so the flag survives
        // the intermediate run).
        if ctx.rechunk {
            let legacy = match ctx.db.book_chunker_ver(ctx.collection_id, &book_id)? {
                Some(v) => v < crate::CURRENT_CHUNKER_VER,
                // No manifest row but chunks in the store (possibly under a
                // legacy id at this same path) → legacy by definition.
                None => {
                    ctx.indexed_ids.contains(&book_id)
                        || ctx.paths_by_id.values().any(|p| p == &path_str)
                }
            };
            if legacy {
                let csig = (ctx.csig_fn)(cand);
                if is_sig_sentinel(&csig) {
                    // Unreadable can't progress by re-running: not counted.
                    plan.preskips.push((path_str, SkipReason::Unreadable));
                    continue;
                }
                if seen_content.contains_key(&csig) {
                    // Forced twin deferred this run — still pending work.
                    plan.forced_count += 1;
                    plan.preskips.push((path_str, SkipReason::DuplicateInRun));
                    continue;
                }
                seen_content.insert(csig.clone(), book_id.clone());
                plan.forced_count += 1;
                plan.to_embed.push(EmbedItem {
                    path: path_str,
                    fingerprint: fp,
                    content_sig: csig,
                });
                continue;
            }
        }

        // Stage 1: unchanged file under its own path-derived id.
        if ctx
            .db
            .book_fingerprint(ctx.collection_id, &book_id)?
            .as_deref()
            == Some(fp.as_str())
        {
            plan.preskips.push((path_str, SkipReason::Unchanged));
            continue;
        }

        // Stage 2: fingerprint reverse lookup (path-independent).
        if let Some(hit) = ctx.db.book_state_for_fingerprint(ctx.collection_id, &fp)? {
            if hit.book_id != book_id {
                let old_path = ctx.hit_path(&hit);
                if !old_path.is_empty() && same_file(&old_path, &path_str) {
                    // Same file, different id scheme (legacy import). Plain
                    // skip; refresh only fills the row's missing path column.
                    if hit.source_path.is_empty() {
                        plan.state_refreshes.push(StateRefresh {
                            book_id: hit.book_id.clone(),
                            path: path_str.clone(),
                            fingerprint: hit.fingerprint.clone(),
                            content_sig: None,
                        });
                    }
                    plan.preskips.push((path_str, SkipReason::Unchanged));
                    continue;
                }
                // Paths differ (or unknown): a size:mtime fingerprint is NOT
                // identity — confirm via content signature before remapping.
                let csig = (ctx.csig_fn)(cand);
                if is_sig_sentinel(&csig) {
                    plan.preskips.push((path_str, SkipReason::Unreadable));
                    continue;
                }
                if !is_sig_sentinel(&hit.content_sig) && csig == hit.content_sig {
                    // Confirmed move: re-point chunks, no re-embed. Under
                    // re-chunk, a legacy row moving is still-pending work.
                    if ctx.rechunk && hit.chunker_ver < crate::CURRENT_CHUNKER_VER {
                        plan.forced_count += 1;
                    }
                    plan.remaps.push(RemapAction {
                        old_id: hit.book_id,
                        new_id: book_id,
                        path: path_str.clone(),
                        fingerprint: fp,
                        content_sig: Some(csig),
                        chunker_ver: hit.chunker_ver,
                    });
                    plan.preskips.push((path_str, SkipReason::Unchanged));
                    continue;
                }
                // Different file sharing a fingerprint (collision), or the row
                // can't prove identity — fall through to the later stages with
                // the signature already in hand.
                if let Some(next) = stage_3_4(
                    &mut plan,
                    &mut seen_content,
                    ctx,
                    &indexed_paths,
                    cand,
                    &path_str,
                    &book_id,
                    &fp,
                    csig,
                )? {
                    plan.to_embed.push(next);
                }
                continue;
            }
            // Same id, fingerprint matches a stale row value (stage-1 compared
            // the CURRENT fp): treat as changed content — fall through below.
        }

        // Stage 3 by id: present in the lance store but missing manifest state
        // (index predates the manifest, or Parquet import).
        if ctx.indexed_ids.contains(&book_id) {
            plan.state_refreshes.push(StateRefresh {
                book_id: book_id.clone(),
                path: path_str.clone(),
                fingerprint: fp.clone(),
                content_sig: None,
            });
            plan.preskips.push((path_str, SkipReason::Unchanged));
            continue;
        }

        let csig = (ctx.csig_fn)(cand);
        if is_sig_sentinel(&csig) {
            plan.preskips.push((path_str, SkipReason::Unreadable));
            continue;
        }
        if let Some(item) = stage_3_4(
            &mut plan,
            &mut seen_content,
            ctx,
            &indexed_paths,
            cand,
            &path_str,
            &book_id,
            &fp,
            csig,
        )? {
            plan.to_embed.push(item);
        }
    }
    Ok(plan)
}

/// Stages shared by the fall-through paths: §2.1.3 path membership, the in-run
/// content map, and the stage-4 content-signature reverse lookup. Returns the
/// embed item when the candidate is genuinely new. `csig` is verified
/// non-sentinel by the caller.
#[allow(clippy::too_many_arguments)] // internal helper mirroring the stage inputs
fn stage_3_4(
    plan: &mut IndexPlan,
    seen_content: &mut HashMap<String, String>,
    ctx: &PlanCtx,
    indexed_paths: &HashSet<&str>,
    cand: &Path,
    path_str: &str,
    book_id: &str,
    fp: &str,
    csig: String,
) -> Result<Option<EmbedItem>, DbError> {
    let _ = cand;
    // §2.1.3: present in the store under this exact path (whatever its id) —
    // covers seed failures and manifest-row loss.
    if indexed_paths.contains(path_str) {
        if let Some((id, _)) = ctx.paths_by_id.iter().find(|(_, p)| p.as_str() == path_str) {
            plan.state_refreshes.push(StateRefresh {
                book_id: id.clone(),
                path: path_str.to_string(),
                fingerprint: fp.to_string(),
                content_sig: Some(csig),
            });
        }
        plan.preskips
            .push((path_str.to_string(), SkipReason::Unchanged));
        return Ok(None);
    }

    // In-run duplicate (same bytes queued twice this run).
    if seen_content.contains_key(&csig) {
        plan.preskips
            .push((path_str.to_string(), SkipReason::DuplicateInRun));
        return Ok(None);
    }

    // Stage 4: content-signature reverse lookup — a hit IS identity.
    if let Some(hit) = ctx.db.book_state_for_content(ctx.collection_id, &csig)? {
        let old_path = ctx.hit_path(&hit);
        if !old_path.is_empty() && same_file(&old_path, path_str) {
            // Metadata churn (e.g. cloud re-stamp): same file, same content,
            // stale fingerprint. Refresh the fingerprint under the EXISTING id
            // so the next run short-circuits at stage 2 without the 512 KiB
            // signature read.
            plan.state_refreshes.push(StateRefresh {
                book_id: hit.book_id,
                path: path_str.to_string(),
                fingerprint: fp.to_string(),
                content_sig: Some(csig),
            });
            plan.preskips
                .push((path_str.to_string(), SkipReason::Unchanged));
            return Ok(None);
        }
        if hit.book_id != book_id {
            // Genuine move (content proves identity).
            if ctx.rechunk && hit.chunker_ver < crate::CURRENT_CHUNKER_VER {
                plan.forced_count += 1;
            }
            plan.remaps.push(RemapAction {
                old_id: hit.book_id,
                new_id: book_id.to_string(),
                path: path_str.to_string(),
                fingerprint: fp.to_string(),
                content_sig: Some(csig),
                chunker_ver: hit.chunker_ver,
            });
            plan.preskips
                .push((path_str.to_string(), SkipReason::Unchanged));
            return Ok(None);
        }
        // Same id, same content, changed metadata under the same path-derived
        // id: refresh the fingerprint.
        plan.state_refreshes.push(StateRefresh {
            book_id: hit.book_id,
            path: path_str.to_string(),
            fingerprint: fp.to_string(),
            content_sig: Some(csig),
        });
        plan.preskips
            .push((path_str.to_string(), SkipReason::Unchanged));
        return Ok(None);
    }

    seen_content.insert(csig.clone(), book_id.to_string());
    Ok(Some(EmbedItem {
        path: path_str.to_string(),
        fingerprint: fp.to_string(),
        content_sig: csig,
    }))
}
