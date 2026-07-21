//! Cooperative rendezvous — port of `semantics/sync.py`. CtaSync/WgSync/WarpSync/
//! ClusterSync block until the whole scope arrives — `Block(Polled)`, re-run each
//! round. The arrival/completion writes are direct; re-running is naturally
//! idempotent (the arrival is a set union, completion is gated on `==expected`).

use super::super::cohort::CohortContext;
use super::super::diagnostics::{IResult, InterpreterError};
use super::super::outcomes::{StepStatus, WakeCondition};
use super::super::protocol::TraceEventKind;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::scheduler::{flatten_coord, unflatten_coord, CtaActivityStatus};
use super::super::threads::{canonical_thread_mask, ThreadId, ThreadMask};
use crate::ir::Stmt;
use std::collections::HashSet;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::CtaSync, execute_sync);
    reg.register(StmtKind::WgSync, execute_sync);
    reg.register(StmtKind::NamedBarrier, execute_named_barrier);
    reg.register(StmtKind::WarpSync, execute_sync);
    reg.register(StmtKind::ClusterSync, execute_sync);
    reg.register(
        StmtKind::ClusterBarrierArrive,
        execute_cluster_barrier_arrive,
    );
    reg.register(StmtKind::ClusterBarrierWait, execute_cluster_barrier_wait);
    reg.register(StmtKind::SetMaxNReg, execute_setmaxnreg);
}

/// `setmaxnreg` is a pure per-warpgroup register-allocation hint — no value, protocol,
/// or rendezvous effect. The value/protocol model just advances past it (the threads of
/// the gated warpgroup execute it independently; it is not a collective barrier).
fn execute_setmaxnreg<'a, 'k>(
    _ctx: &mut CohortContext<'a, 'k>,
    _stmt: &'k Stmt,
) -> IResult<StepStatus> {
    Ok(StepStatus::advance())
}

/// Split cluster barrier key — one hardware named cluster barrier per cluster, used
/// once in the prologue (a non-blocking collective arrive + per-role waits). Keyed by
/// cluster, NOT by stmt_id, so the CTA-scope arrive and the per-role waits (different
/// stmt_ids) reference the SAME arrival set.
fn cluster_barrier_key(first: &ThreadId) -> String {
    format!("cluster_barrier:cluster{}", first.cluster_id)
}

/// `barrier.cluster.arrive` (aligned): every cluster thread records its arrival and
/// CONTINUES — non-blocking, unlike the fused `ClusterSync` rendezvous. The matching
/// per-role `ClusterBarrierWait`s block until the set is complete. Modeled as a
/// one-shot collective (the prologue issues it exactly once before the role loops).
/// The arrive emits a `ClusterBarrierArrive` trace event carrying the stmt's memory
/// `sem`: canon's `.relaxed` carries NO release ordering (PTX §9.7.14.3), so the
/// checker's ordering analysis only publishes a memory happens-before edge for
/// `.release`; the deadlock graph witnesses the waits against the arrival set
/// either way (control order). The release edge is what can prove prologue
/// inits/alloc happen-before a role's peer-visible accesses — with `.relaxed`
/// that proof must come from the mbarrier pipeline, as in canon.
fn execute_cluster_barrier_arrive<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let sem = match stmt {
        Stmt::ClusterBarrierArrive { sem } => *sem,
        _ => unreachable!(),
    };
    let first = ctx.cohort[0].clone();
    let expected: HashSet<ThreadId> = cluster_threads(ctx, &first).into_iter().collect();
    let arriving: HashSet<ThreadId> = ctx.cohort.iter().cloned().collect();
    if !arriving.is_subset(&expected) {
        return Err(InterpreterError::new(
            "invalid_cluster_barrier_scope",
            "cluster_barrier_arrive cohort is outside the cluster",
        ));
    }
    let key = cluster_barrier_key(&first);
    let arrived = ctx.state.values.cooperative.syncs.entry(key).or_default();
    if !arriving.is_disjoint(arrived) {
        return Err(InterpreterError::new(
            "cluster_barrier_reentry",
            "thread re-entered the one-shot cluster barrier arrive",
        ));
    }
    arrived.extend(arriving);
    let arrived_count = arrived.len();
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::ClusterBarrierArrive {
            thread_count: expected.len(),
            count: arrived_count,
            sem: sem.into(),
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}

/// `barrier.cluster.wait` (per role): block until ALL cluster threads have executed
/// `ClusterBarrierArrive`. A peer CTA that exits/goes missing without arriving makes
/// the wait unsatisfiable — surfaced as a peer-liveness error (as for `ClusterSync`).
fn execute_cluster_barrier_wait<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    _stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let first = ctx.cohort[0].clone();
    let expected: HashSet<ThreadId> = cluster_threads(ctx, &first).into_iter().collect();
    let key = cluster_barrier_key(&first);
    let arrived: HashSet<ThreadId> = ctx
        .state
        .values
        .cooperative
        .syncs
        .get(&key)
        .cloned()
        .unwrap_or_default();
    check_cluster_peer_liveness(ctx, &expected, &arrived)?;
    if expected.is_subset(&arrived) {
        // The wait passes — emit the ACQUIRE witness (hardware `barrier.cluster.wait`
        // is `.acquire`): the checker joins the cluster's published arrival clock.
        if ctx.trace_mode() {
            ctx.emit(TraceEventKind::ClusterBarrierWait {
                scope: ctx.access_scope(),
            })?;
        }
        Ok(StepStatus::advance())
    } else {
        Ok(StepStatus::block(WakeCondition::Polled))
    }
}

/// Named barrier across warpgroups — `bar.sync barrier_id, num_warps*32`. Threads
/// from DIFFERENT roles (warpgroups) rendezvous on the shared `barrier_id` (the
/// key omits the warpgroup), completion gated on the arrival COUNT == num_warps*32
/// (count-based, since the participating warps span roles). Mirrors execute_sync's
/// idempotent set-union + rendezvous bookkeeping.
fn execute_named_barrier<'a, 'k>(
    ctx: &mut CohortContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (barrier_id, num_warps) = match stmt {
        Stmt::NamedBarrier {
            barrier_id,
            num_warps,
        } => (*barrier_id, *num_warps),
        _ => unreachable!(),
    };
    let expected_count = (num_warps as usize) * 32;
    let key = format!(
        "named_barrier:cta{}:bar{}",
        ctx.cohort[0].cta_id, barrier_id
    );
    let arriving: HashSet<ThreadId> = ctx.cohort.iter().cloned().collect();
    let cycle = ctx
        .state
        .values
        .cooperative
        .sync_cycles
        .get(&key)
        .copied()
        .unwrap_or(0);
    let arrived: HashSet<ThreadId> = ctx
        .state
        .values
        .cooperative
        .syncs
        .get(&key)
        .cloned()
        .unwrap_or_default();
    let completed: HashSet<ThreadId> = ctx
        .state
        .values
        .cooperative
        .rendezvous
        .get(&key)
        .cloned()
        .unwrap_or_default();
    if !arriving.is_disjoint(&completed) {
        return Err(InterpreterError::new(
            "cooperative_sync_reentry",
            "thread re-entered a completed named barrier",
        ));
    }
    let merged: HashSet<ThreadId> = arrived.union(&arriving).cloned().collect();
    if merged.len() > expected_count {
        return Err(InterpreterError::new(
            "named_barrier_overflow",
            "named barrier arrivals exceed num_warps*32",
        ));
    }
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::SyncArrive {
            sync_kind: "named".to_string(),
            thread_count: expected_count,
            count: merged.len(),
            cycle,
            bar_id: Some(barrier_id),
            scope: ctx.access_scope(),
        })?;
    }
    if merged.len() != expected_count {
        // not all warps arrived yet — record this cohort's arrival, re-poll next round
        ctx.state.values.cooperative.syncs.insert(key, merged);
        return Ok(StepStatus::block(WakeCondition::Polled));
    }
    // count reached → complete: move this cohort into the rendezvous; the last
    // cohort (completed == count) clears both records so the next use starts fresh.
    let completed_next: HashSet<ThreadId> = completed.union(&arriving).cloned().collect();
    if completed_next.len() == expected_count {
        ctx.state.values.cooperative.syncs.remove(&key);
        ctx.state.values.cooperative.rendezvous.remove(&key);
        ctx.state
            .values
            .cooperative
            .sync_cycles
            .insert(key.clone(), cycle + 1);
    } else {
        ctx.state
            .values
            .cooperative
            .syncs
            .insert(key.clone(), merged.clone());
        ctx.state
            .values
            .cooperative
            .rendezvous
            .insert(key, completed_next);
    }
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::Sync {
            sync_kind: "named".to_string(),
            thread_count: expected_count,
            cycle,
            bar_id: Some(barrier_id),
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}

fn cta_threads(template: &ThreadId, warp_ids: &[usize]) -> ThreadMask {
    let mut threads = Vec::new();
    for &w in warp_ids {
        for lane in 0..32 {
            threads.push(ThreadId {
                warp_id: w,
                lane_id: lane,
                ..*template
            });
        }
    }
    canonical_thread_mask(threads)
}

fn cluster_threads(ctx: &CohortContext, template: &ThreadId) -> ThreadMask {
    let cluster = &ctx.kernel.cluster_shape;
    let launch = &ctx.kernel.launch_shape;
    let cluster_grid: Vec<usize> = launch
        .iter()
        .zip(cluster.iter())
        .map(|(l, c)| l / c)
        .collect();
    let num_warps = ctx.kernel.num_warps as usize;
    let mut threads = Vec::new();
    for ctaid_in_cluster in 0..ctx.cluster_cta_count() {
        let local = unflatten_coord(ctaid_in_cluster, cluster);
        let cta_coord: Vec<usize> = cluster
            .iter()
            .zip(template.cluster_coord.as_slice().iter())
            .zip(local.iter())
            .map(|((cl, cc), l)| cl * cc + l)
            .collect();
        let cta_id = flatten_coord(&cta_coord, launch);
        let cluster_id = flatten_coord(template.cluster_coord.as_slice(), &cluster_grid);
        let cta_coord_c = super::super::threads::Coord::from_slice(&cta_coord);
        let local_c = super::super::threads::Coord::from_slice(&local);
        for w in 0..num_warps {
            for lane in 0..32 {
                threads.push(ThreadId {
                    cta_id,
                    cta_coord: cta_coord_c,
                    cluster_id,
                    ctaid_in_cluster,
                    cluster_coord: template.cluster_coord,
                    cta_coord_in_cluster: local_c,
                    warp_id: w,
                    lane_id: lane,
                });
            }
        }
    }
    canonical_thread_mask(threads)
}

fn expected_threads(ctx: &CohortContext, stmt: &Stmt) -> ThreadMask {
    let first = &ctx.cohort[0];
    let num_warps = ctx.kernel.num_warps as usize;
    match stmt {
        Stmt::ClusterSync => cluster_threads(ctx, first),
        Stmt::CtaSync => cta_threads(first, &(0..num_warps).collect::<Vec<_>>()),
        Stmt::WgSync { .. } => {
            let base = first.warpgroup_id() * 4;
            cta_threads(first, &(base..base + 4).collect::<Vec<_>>())
        }
        Stmt::WarpSync => {
            let mut warps: Vec<usize> = ctx.cohort.iter().map(|t| t.warp_id).collect();
            warps.sort_unstable();
            warps.dedup();
            cta_threads(first, &warps)
        }
        _ => unreachable!(),
    }
}

fn sync_key(ctx: &CohortContext, stmt: &Stmt, stmt_id: usize) -> String {
    let first = &ctx.cohort[0];
    match stmt {
        Stmt::ClusterSync => format!("cluster_sync:{stmt_id}:cluster{}", first.cluster_id),
        Stmt::CtaSync => format!("cta_sync:{stmt_id}:cta{}", first.cta_id),
        Stmt::WgSync { barrier_id } => {
            format!(
                "wg_sync:{stmt_id}:cta{}:wg{}:bar{}",
                first.cta_id,
                first.warpgroup_id(),
                barrier_id
            )
        }
        Stmt::WarpSync => {
            let mut warps: Vec<usize> = ctx.cohort.iter().map(|t| t.warp_id).collect();
            warps.sort_unstable();
            warps.dedup();
            let joined = warps
                .iter()
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!("warp_sync:{stmt_id}:cta{}:warps{joined}", first.cta_id)
        }
        _ => unreachable!(),
    }
}

fn execute_sync<'a, 'k>(ctx: &mut CohortContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let stmt_id = ctx.stmt_id(stmt);
    let expected: HashSet<ThreadId> = expected_threads(ctx, stmt).into_iter().collect();
    let arriving: HashSet<ThreadId> = ctx.cohort.iter().cloned().collect();
    if !arriving.is_subset(&expected) {
        return Err(InterpreterError::new(
            "invalid_sync_scope",
            "sync cohort is outside the sync scope",
        ));
    }
    let key = sync_key(ctx, stmt, stmt_id);
    let cycle = ctx
        .state
        .values
        .cooperative
        .sync_cycles
        .get(&key)
        .copied()
        .unwrap_or(0);
    let arrived: HashSet<ThreadId> = ctx
        .state
        .values
        .cooperative
        .syncs
        .get(&key)
        .cloned()
        .unwrap_or_default();
    let completed: HashSet<ThreadId> = ctx
        .state
        .values
        .cooperative
        .rendezvous
        .get(&key)
        .cloned()
        .unwrap_or_default();

    if !arriving.is_disjoint(&completed) {
        return Err(InterpreterError::new(
            "cooperative_sync_reentry",
            "thread re-entered a completed sync",
        ));
    }
    let merged: HashSet<ThreadId> = arrived.union(&arriving).cloned().collect();
    if !merged.is_subset(&expected) {
        return Err(InterpreterError::new(
            "cooperative_sync_mismatch",
            "sync arrival set exceeds scope",
        ));
    }
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::SyncArrive {
            sync_kind: sync_kind(stmt).to_string(),
            thread_count: expected.len(),
            count: merged.len(),
            cycle,
            bar_id: sync_bar_id(stmt),
            scope: ctx.access_scope(),
        })?;
    }

    if matches!(stmt, Stmt::ClusterSync) {
        check_cluster_peer_liveness(ctx, &expected, &merged)?;
    }

    if merged != expected {
        // record this cohort's arrival (idempotent set union) and re-poll next round
        ctx.state.values.cooperative.syncs.insert(key, merged);
        return Ok(StepStatus::block(WakeCondition::Polled));
    }

    // all arrived → complete: move this cohort into the rendezvous; the last one
    // (completed == expected) clears both records so the next use starts fresh.
    let completed_next: HashSet<ThreadId> = completed.union(&arriving).cloned().collect();
    if completed_next == expected {
        ctx.state.values.cooperative.syncs.remove(&key);
        ctx.state.values.cooperative.rendezvous.remove(&key);
        ctx.state
            .values
            .cooperative
            .sync_cycles
            .insert(key.clone(), cycle + 1);
    } else {
        ctx.state
            .values
            .cooperative
            .syncs
            .insert(key.clone(), expected.clone());
        ctx.state
            .values
            .cooperative
            .rendezvous
            .insert(key, completed_next);
    }
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::Sync {
            sync_kind: sync_kind(stmt).to_string(),
            thread_count: expected.len(),
            cycle,
            bar_id: sync_bar_id(stmt),
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
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

fn check_cluster_peer_liveness(
    ctx: &CohortContext,
    expected: &HashSet<ThreadId>,
    merged: &HashSet<ThreadId>,
) -> IResult<()> {
    let arrived_ctas: HashSet<usize> = merged.iter().map(|t| t.cta_id).collect();
    let expected_ctas: HashSet<usize> = expected.iter().map(|t| t.cta_id).collect();
    for cta_id in expected_ctas {
        if arrived_ctas.contains(&cta_id) {
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
