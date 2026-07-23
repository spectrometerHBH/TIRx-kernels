//! Cooperative rendezvous — CtaSync / WgSync / WarpSync / ClusterSync.
//!
//! Per-warp streams arrive independently; the record in
//! `state.values.cooperative` accumulates STREAM-granular arrival counts and
//! every participant passes once the count reaches the scope size. Blocked
//! streams are `Block(Polled)` and re-run each round, but a re-poll is a hash
//! lookup + integer compare (no set algebra, no allocation) — the poll loop
//! stays cheap even when many streams idle at a teardown barrier for
//! thousands of rounds. Trace: one `SyncArrive` per arriving stream (the
//! completing one carries count == thread_count) and one `Sync` per passing
//! stream — no re-poll echoes.

use super::super::cohort::CohortContext;
use super::super::diagnostics::{IResult, InterpreterError};
use super::super::outcomes::{StepStatus, WakeCondition};
use super::super::protocol::TraceEventKind;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::scheduler::{flatten_coord, unflatten_coord, CtaActivityStatus};
use super::super::threads::ThreadId;
use super::super::values::cooperative::{SyncKey, SyncRecord};
use crate::ir::Stmt;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::CtaSync, execute_sync);
    reg.register(StmtKind::WgSync, execute_sync);
    reg.register(StmtKind::WarpSync, execute_sync);
    reg.register(StmtKind::ClusterSync, execute_sync);
}

fn sync_kind(stmt: &Stmt) -> &'static str {
    match stmt {
        Stmt::ClusterSync => "cluster",
        Stmt::CtaSync => "cta",
        Stmt::WgSync { .. } => "warpgroup",
        Stmt::WarpSync => "warp",
        _ => unreachable!(),
    }
}

fn sync_bar_id(stmt: &Stmt) -> Option<u32> {
    match stmt {
        Stmt::WgSync { barrier_id } => Some(*barrier_id as u32),
        _ => None,
    }
}

/// The rendezvous domain of this execution (which barrier INSTANCE it joins).
fn sync_domain(stmt: &Stmt, first: &ThreadId) -> (usize, usize) {
    match stmt {
        Stmt::ClusterSync => (first.cluster_id, 0),
        Stmt::CtaSync => (first.cta_id, 0),
        Stmt::WgSync { .. } => (first.cta_id, first.warpgroup_id()),
        Stmt::WarpSync => (first.cta_id, first.warp_id),
        _ => unreachable!(),
    }
}

/// Is `t` inside the barrier's scope?
fn in_scope(stmt: &Stmt, first: &ThreadId, t: &ThreadId) -> bool {
    match stmt {
        Stmt::ClusterSync => t.cluster_id == first.cluster_id,
        Stmt::CtaSync => t.cta_id == first.cta_id,
        Stmt::WgSync { .. } => t.cta_id == first.cta_id && t.warpgroup_id() == first.warpgroup_id(),
        Stmt::WarpSync => t.cta_id == first.cta_id && t.warp_id == first.warp_id,
        _ => unreachable!(),
    }
}

/// Global CTA ids of the cluster containing `template` (source order).
fn cluster_cta_ids(ctx: &CohortContext, template: &ThreadId) -> Vec<usize> {
    let cluster = &ctx.kernel.cluster_shape;
    let launch = &ctx.kernel.launch_shape;
    let mut ids = Vec::with_capacity(ctx.cluster_cta_count());
    for ctaid_in_cluster in 0..ctx.cluster_cta_count() {
        let local = unflatten_coord(ctaid_in_cluster, cluster);
        let cta_coord: Vec<usize> = cluster
            .iter()
            .zip(template.cluster_coord.as_slice().iter())
            .zip(local.iter())
            .map(|((cl, cc), l)| cl * cc + l)
            .collect();
        ids.push(flatten_coord(&cta_coord, launch));
    }
    ids
}

/// (expected thread count, CTA ids in scope) for a fresh record.
fn sync_scope(ctx: &CohortContext, stmt: &Stmt, first: &ThreadId) -> (usize, Vec<usize>) {
    let warp_threads = 32usize;
    let cta_threads = ctx.kernel.num_warps as usize * warp_threads;
    match stmt {
        Stmt::ClusterSync => {
            let ctas = cluster_cta_ids(ctx, first);
            (ctas.len() * cta_threads, ctas)
        }
        Stmt::CtaSync => (cta_threads, vec![first.cta_id]),
        Stmt::WgSync { .. } => (4 * warp_threads, vec![first.cta_id]),
        Stmt::WarpSync => {
            // The barrier spans exactly the arriving cohort's warps (one warp
            // under per-warp streams); a sub-warp cohort can never fill it —
            // the hardware hang, surfaced as a deadlock.
            let mut warps: Vec<usize> = ctx.cohort.iter().map(|t| t.warp_id).collect();
            warps.sort_unstable();
            warps.dedup();
            (warps.len() * warp_threads, vec![first.cta_id])
        }
        _ => unreachable!(),
    }
}

fn check_cluster_peer_liveness(ctx: &CohortContext, rec: &SyncRecord) -> IResult<()> {
    for &cta_id in &rec.expected_ctas {
        if rec.arrived_per_cta.contains_key(&cta_id) {
            continue;
        }
        match ctx.cta_activity(cta_id) {
            CtaActivityStatus::Missing => {
                return Err(InterpreterError::new(
                    "cluster_sync_peer_missing",
                    "cluster sync peer CTA is missing",
                ))
            }
            CtaActivityStatus::Exited => {
                return Err(InterpreterError::new(
                    "cluster_sync_peer_exited",
                    "cluster sync peer CTA has exited",
                ))
            }
            _ => {}
        }
    }
    Ok(())
}

fn execute_sync<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let stmt_id = ctx.stmt_id(stmt);
    let first = ctx.cohort[0].clone();
    let key = SyncKey {
        stmt_id,
        domain: sync_domain(stmt, &first),
    };
    let stream_id = ctx.stream.stream_id;

    // Fast path: this stream already arrived this cycle (a Polled re-run).
    if let Some(rec) = ctx.state.values.cooperative.syncs.get(&key) {
        if rec.arrived_streams.contains_key(&stream_id) {
            if !rec.complete() {
                // Still waiting for the rest of the scope. O(1) per round.
                if matches!(stmt, Stmt::ClusterSync) {
                    check_cluster_peer_liveness(ctx, rec)?;
                }
                return Ok(StepStatus::block(WakeCondition::Polled));
            }
            if rec.passed_streams.contains(&stream_id) {
                return Err(InterpreterError::new(
                    "cooperative_sync_reentry",
                    "thread re-entered a completed sync",
                ));
            }
            // Complete: pass. The last passer re-arms the record.
            let n = rec.arrived_streams[&stream_id];
            let cycle = rec.cycle;
            let expected_count = rec.expected_count;
            if ctx.trace_mode() {
                ctx.emit(TraceEventKind::Sync {
                    sync_kind: sync_kind(stmt).to_string(),
                    thread_count: expected_count,
                    cycle,
                    bar_id: sync_bar_id(stmt),
                    scope: ctx.access_scope(),
                })?;
            }
            let rec = ctx.state.values.cooperative.syncs.get_mut(&key).unwrap();
            rec.passed_streams.insert(stream_id);
            rec.passed_count += n;
            if rec.passed_count == rec.expected_count {
                rec.advance_cycle();
            }
            return Ok(StepStatus::advance());
        }
    }

    // First arrival of this stream for the current cycle.
    for t in ctx.cohort.iter() {
        if !in_scope(stmt, &first, t) {
            return Err(InterpreterError::new(
                "invalid_sync_scope",
                "sync cohort is outside the sync scope",
            ));
        }
    }
    let n = ctx.cohort.len();
    let (expected_count, expected_ctas) = sync_scope(ctx, stmt, &first);
    {
        let rec = ctx
            .state
            .values
            .cooperative
            .syncs
            .entry(key)
            .or_insert_with(SyncRecord::default);
        if rec.expected_count == 0 {
            rec.expected_count = expected_count;
            rec.expected_ctas = expected_ctas;
        }
        if rec.complete() {
            // Everyone the scope expects has arrived, yet a new stream shows
            // up — the arrival set would exceed the scope.
            return Err(InterpreterError::new(
                "cooperative_sync_mismatch",
                "sync arrival set exceeds scope",
            ));
        }
        if rec.arrived_count + n > rec.expected_count {
            return Err(InterpreterError::new(
                "cooperative_sync_mismatch",
                "sync arrival set exceeds scope",
            ));
        }
        rec.arrived_streams.insert(stream_id, n);
        rec.arrived_count += n;
        *rec.arrived_per_cta.entry(first.cta_id).or_insert(0) += n;
    }
    let rec_snapshot = |ctx: &CohortContext| {
        let rec = &ctx.state.values.cooperative.syncs[&key];
        (rec.cycle, rec.arrived_count, rec.complete())
    };
    let (cycle, arrived_count, complete) = rec_snapshot(ctx);
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::SyncArrive {
            sync_kind: sync_kind(stmt).to_string(),
            thread_count: expected_count,
            count: arrived_count,
            cycle,
            bar_id: sync_bar_id(stmt),
            scope: ctx.access_scope(),
        })?;
    }
    if !complete {
        if matches!(stmt, Stmt::ClusterSync) {
            let rec = &ctx.state.values.cooperative.syncs[&key];
            check_cluster_peer_liveness(ctx, rec)?;
        }
        return Ok(StepStatus::block(WakeCondition::Polled));
    }
    // This arrival completed the barrier: pass immediately.
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::Sync {
            sync_kind: sync_kind(stmt).to_string(),
            thread_count: expected_count,
            cycle,
            bar_id: sync_bar_id(stmt),
            scope: ctx.access_scope(),
        })?;
    }
    let rec = ctx.state.values.cooperative.syncs.get_mut(&key).unwrap();
    rec.passed_streams.insert(stream_id);
    rec.passed_count += n;
    if rec.passed_count == rec.expected_count {
        rec.advance_cycle();
    }
    Ok(StepStatus::advance())
}
