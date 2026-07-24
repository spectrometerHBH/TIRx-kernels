//! Offline protocol checker passes over a completed trace.
//!
//! These passes intentionally consume only completed trace executions. Failed or
//! inconclusive trace runs return directly from the interpreter and do not enter
//! this module.

use super::diagnostics::Diagnostic;
use super::protocol::{
    BoxN, FenceEventKind, MbarTargetEvent, MemoryAccessKind, MemoryProxy, PoolId,
    ProtocolPassSummary, ProtocolReport, ProtocolStatus, ReduceOp, Region, RegionBoxes, SemKey,
    TensorAccessKind, TmemAsyncKind, TraceEvent, TraceEventKind,
};
use super::region::{boxes_overlap, region_covers, regions_overlap, TMEM_LANE_BYTES};
use super::values::indexing::numel;
use super::values::smem::dtype_size_bytes;
use super::values::tmem::TMEM_ROWS;
use crate::ir::{
    static_thread_filter, FenceScope, Kernel, Stmt, Tensor, TensorSlice, ThreadFilter, ThreadSet,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

type CheckResult = Result<(), ProtocolStatus>;

/// The `(op, exact)` of an atomic commutative reduce access (Read or Write), else `None`.
fn reduce_access(payload: &TraceEventKind) -> Option<(ReduceOp, bool)> {
    match payload {
        TraceEventKind::Read {
            access_kind: MemoryAccessKind::Tensor(k),
            ..
        }
        | TraceEventKind::Write {
            access_kind: MemoryAccessKind::Tensor(k),
            ..
        } => k.reduce(),
        _ => None,
    }
}

pub fn run_offline_checkers(
    kernel: &Kernel,
    mut report: ProtocolReport,
    events: &[TraceEvent],
) -> ProtocolReport {
    let mut cx = CheckerCx::new(kernel, events);
    run_pass(
        &mut cx,
        &mut report,
        "trace_schema_audit",
        trace_schema_audit,
    );
    run_pass(
        &mut cx,
        &mut report,
        "trace_region_audit",
        trace_region_audit,
    );
    run_pass(
        &mut cx,
        &mut report,
        "barrier_cycle_audit",
        barrier_cycle_audit,
    );
    run_pass(
        &mut cx,
        &mut report,
        "ordering_analysis",
        ordering_analysis_pass,
    );
    run_pass(&mut cx, &mut report, "deadlock_freedom", deadlock_freedom);
    run_pass(
        &mut cx,
        &mut report,
        "async_group_lifetime",
        async_group_lifetime_local,
    );
    run_pass(
        &mut cx,
        &mut report,
        "tcgen05_async_hazard",
        tcgen05_async_hazard,
    );
    run_pass(
        &mut cx,
        &mut report,
        "tmem_lifecycle_order",
        tmem_lifecycle_order_local,
    );
    run_pass(&mut cx, &mut report, "memory_race_check", memory_race_check);
    run_pass(
        &mut cx,
        &mut report,
        "cluster_peer_consistency",
        cluster_peer_consistency,
    );
    run_pass(
        &mut cx,
        &mut report,
        "scheduler_handoff_consistency",
        scheduler_handoff_consistency,
    );
    run_pass(&mut cx, &mut report, "trace_gap_audit", trace_gap_audit);
    run_pass(
        &mut cx,
        &mut report,
        "unknown_filter_scope_check",
        unknown_filter_scope_check,
    );
    report
}

/// A synchronization scope that depends on a runtime value.
///
/// The static thread filter proves which threads reach a branch from
/// warp/lane coordinates. When a predicate mixes in a runtime value the filter
/// is `Unknown`, and the shape rules that would bound a sync's scope
/// (wg_sync covers one warpgroup, cta_sync covers the CTA, ...) stand down —
/// the runtime rendezvous is the check. A sync reached under such a branch is
/// legal but its scope is only verified at run time; this warns so a scope bug
/// there surfaces as more than a bare deadlock.
fn unknown_filter_scope_check(cx: &mut CheckerCx<'_>) -> CheckResult {
    fn intersect(outer: &ThreadFilter, branch: &ThreadFilter) -> ThreadFilter {
        match (outer, branch) {
            (ThreadFilter::Known(a), ThreadFilter::Known(b)) => ThreadFilter::Known(a.intersect(b)),
            _ => ThreadFilter::Unknown,
        }
    }
    fn walk(cx: &mut CheckerCx<'_>, body: &[Stmt], filter: &ThreadFilter, num_warps: u32) {
        let unknown = filter.known().is_none();
        for stmt in body {
            match stmt {
                Stmt::If { cond, then_body } => {
                    let inner = intersect(filter, &static_thread_filter(cond, num_warps));
                    walk(cx, then_body, &inner, num_warps);
                }
                Stmt::ForLoop { body, .. }
                | Stmt::ForEachTask { body, .. }
                | Stmt::SchedulerImpl { body, .. }
                | Stmt::Loop { body } => walk(cx, body, filter, num_warps),
                _ if unknown => {
                    let op = match stmt {
                        Stmt::CtaSync => "cta_sync",
                        Stmt::ClusterSync => "cluster_sync",
                        Stmt::WgSync { .. } => "wg_sync",
                        Stmt::WarpSync => "warp_sync",
                        Stmt::NamedBarrier { .. } => "named_barrier",
                        _ => continue,
                    };
                    cx.warn(
                        "unknown_filter_sync",
                        format!(
                            "{op} is reached under a branch on a runtime value; its scope is \
                             verified at run time, not by the static shape rules"
                        ),
                    );
                }
                _ => {}
            }
        }
    }
    let num_warps = cx.kernel.num_warps;
    let full = ThreadFilter::Known(ThreadSet::full(num_warps));
    let body = cx.kernel.body.clone();
    walk(cx, &body, &full, num_warps);
    Ok(())
}

fn run_pass(
    cx: &mut CheckerCx<'_>,
    report: &mut ProtocolReport,
    name: &'static str,
    pass: fn(&mut CheckerCx<'_>) -> CheckResult,
) {
    let timer = super::runner::prof_now();
    let wall_timer = std::env::var("NYMPH_STATS")
        .is_ok()
        .then(std::time::Instant::now);
    let diagnostics_before = report.diagnostics.len();
    let warnings_before = report.warnings.len();
    let status = match pass(cx) {
        Ok(()) => ProtocolStatus::Passed,
        Err(status) => status,
    };
    report.diagnostics.extend(cx.take_diagnostics());
    report.warnings.extend(cx.take_warnings());
    if status == ProtocolStatus::Failed {
        report.status = ProtocolStatus::Failed;
    } else if status == ProtocolStatus::Inconclusive && report.status == ProtocolStatus::Passed {
        report.status = ProtocolStatus::Inconclusive;
    }
    report.pass_summary.push(ProtocolPassSummary {
        name: name.to_string(),
        status,
        diagnostics: report.diagnostics.len() - diagnostics_before,
        warnings: report.warnings.len() - warnings_before,
    });
    super::runner::prof_end(name, timer);
    if let Some(start) = wall_timer {
        eprintln!(
            "  CHECK:{name:<32} {:8.2} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
    }
}

struct CheckerCx<'a> {
    kernel: &'a Kernel,
    event_index: EventIndex<'a>,
    region_audit: TraceRegionAudit,
    /// Built by `ordering_analysis`, read by the passes that need
    /// happens-before. `None` until then, so a pass that runs before it says
    /// so instead of quietly reading an empty clock — under which nothing is
    /// ordered and every conflicting pair looks like a race.
    ordering: Option<OrderingAnalysis>,
    gaps: Vec<TraceGap>,
    diagnostics: Vec<Diagnostic>,
    warnings: Vec<super::protocol::ProtocolWarning>,
}

impl<'a> CheckerCx<'a> {
    fn new(kernel: &'a Kernel, events: &'a [TraceEvent]) -> Self {
        Self {
            kernel,
            event_index: EventIndex::new(events),
            region_audit: TraceRegionAudit::new(kernel),
            ordering: None,
            gaps: Vec::new(),
            diagnostics: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn ordering(&self) -> &OrderingAnalysis {
        self.ordering
            .as_ref()
            .expect("ordering_analysis must run before the passes that read the clock")
    }

    fn fail(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
        event: Option<&TraceEvent>,
    ) -> ProtocolStatus {
        self.diagnostics
            .push(DiagnosticBuilder::new(code, message).event(event).build());
        ProtocolStatus::Failed
    }

    fn inconclusive(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
        event: Option<&TraceEvent>,
    ) -> ProtocolStatus {
        self.diagnostics
            .push(DiagnosticBuilder::new(code, message).event(event).build());
        ProtocolStatus::Inconclusive
    }

    fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        std::mem::take(&mut self.diagnostics)
    }

    fn take_warnings(&mut self) -> Vec<super::protocol::ProtocolWarning> {
        std::mem::take(&mut self.warnings)
    }

    /// Record a non-fatal warning (does not affect Passed/Inconclusive). Deduped by code so a
    /// kernel with N commutative reduces emits one `nondeterministic_reduction`, not N.
    fn warn(&mut self, code: impl Into<String>, message: impl Into<String>) {
        let code = code.into();
        if self.warnings.iter().any(|w| w.code == code) {
            return;
        }
        self.warnings.push(super::protocol::ProtocolWarning {
            code,
            message: message.into(),
            details: BTreeMap::new(),
        });
    }

    fn gap(
        &mut self,
        code: impl Into<String>,
        message: impl Into<String>,
        event_idx: Option<usize>,
        details: BTreeMap<String, String>,
    ) {
        self.gaps.push(TraceGap {
            code: code.into(),
            message: message.into(),
            event_idx,
            details,
        });
    }
}

#[derive(Clone, Debug)]
struct TraceGap {
    code: String,
    message: String,
    event_idx: Option<usize>,
    details: BTreeMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemoryAccessMode {
    Read,
    Write,
}

impl MemoryAccessMode {
    fn as_str(self) -> &'static str {
        match self {
            MemoryAccessMode::Read => "read",
            MemoryAccessMode::Write => "write",
        }
    }
}

#[derive(Clone, Debug)]
struct MemoryAccessRecord {
    event_idx: usize,
    stream_id: usize,
    /// The executing lanes. When the region carries no per-lane attribution,
    /// every one of these lanes touches the whole footprint.
    mask: u32,
    mode: MemoryAccessMode,
    region: Region,
    // cached bounding spans: frontier scans reject on these in O(1)
    bounds: ((usize, usize), (usize, usize)),
    /// Access is performed by an async engine (`MemoryProxy::Async`), not by
    /// the lanes themselves — TMA loads/stores, s2cluster, tcgen05 mma/cp/ld/st.
    async_proxy: bool,
}

impl MemoryAccessRecord {
    /// Is `self` an engine access consuming bytes a LANE stored generically?
    /// That is the one pair whose ordering lives in the proxy boundary rather
    /// than in the lane clocks.
    fn crosses_into_engine(&self, prior: &Self) -> bool {
        self.async_proxy && !prior.async_proxy && prior.mode == MemoryAccessMode::Write
    }
}

impl MemoryAccessRecord {
    fn from_shared_event(event_idx: usize, event: &TraceEvent) -> Option<Self> {
        match &event.payload {
            TraceEventKind::Read {
                region,
                scope,
                proxy,
                ..
            } if is_shared_race_target(&region.owner) => Some(Self {
                event_idx,
                stream_id: scope.stream_id,
                mask: scope.active_lanes,
                mode: MemoryAccessMode::Read,
                bounds: super::region::region_bounding_spans(region),
                region: region.clone(),
                async_proxy: *proxy == MemoryProxy::Async,
            }),
            TraceEventKind::Write {
                region,
                scope,
                proxy,
                ..
            } if is_shared_race_target(&region.owner) => Some(Self {
                event_idx,
                stream_id: scope.stream_id,
                mask: scope.active_lanes,
                mode: MemoryAccessMode::Write,
                bounds: super::region::region_bounding_spans(region),
                region: region.clone(),
                async_proxy: *proxy == MemoryProxy::Async,
            }),
            _ => None,
        }
    }
}

struct EventIndex<'a> {
    events: &'a [TraceEvent],
}

impl<'a> EventIndex<'a> {
    fn new(events: &'a [TraceEvent]) -> Self {
        Self { events }
    }
}

struct DiagnosticBuilder {
    diagnostic: Diagnostic,
}

impl DiagnosticBuilder {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            diagnostic: Diagnostic::error(code, message),
        }
    }

    fn event(mut self, event: Option<&TraceEvent>) -> Self {
        if let Some(event) = event {
            self.diagnostic.stmt_id = Some(event.stmt_id.to_string());
            self.diagnostic
                .details
                .insert("event_kind".into(), event_kind_name(&event.payload).into());
            self.diagnostic
                .details
                .insert("stmt_kind".into(), event.stmt_kind.clone());
            if let Some(scope) = event_scope(&event.payload) {
                self.diagnostic.stream_id = Some(scope.stream_id);
                self.diagnostic
                    .details
                    .insert("cta_id".into(), scope.cta_id.to_string());
            }
        }
        self
    }

    fn detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.diagnostic.details.insert(key.into(), value.into());
        self
    }

    fn build(self) -> Diagnostic {
        self.diagnostic
    }
}

fn trace_schema_audit(cx: &mut CheckerCx<'_>) -> CheckResult {
    let mut seen_sync_arrive = false;
    for event in cx.event_index.events {
        if event.stmt_kind.is_empty() {
            return Err(cx.fail(
                "trace_schema_missing_anchor",
                "trace event is missing stmt_kind anchor",
                Some(event),
            ));
        }
        match &event.payload {
            TraceEventKind::SyncArrive {
                thread_count,
                count,
                ..
            } => {
                seen_sync_arrive = true;
                if *thread_count == 0 || *count == 0 || count > thread_count {
                    return Err(cx.fail(
                        "trace_schema_invalid_sync_arrive",
                        "SyncArrive count must be in 1..=thread_count",
                        Some(event),
                    ));
                }
            }
            TraceEventKind::Sync { thread_count, .. } if *thread_count == 0 => {
                return Err(cx.fail(
                    "trace_schema_invalid_sync",
                    "Sync thread_count must be positive",
                    Some(event),
                ));
            }
            TraceEventKind::WaitGroup { n, .. } if *n > 8 => {
                return Err(cx.fail(
                    "trace_schema_invalid_wait_group",
                    "cp.async bulk wait_group trace supports n <= 8",
                    Some(event),
                ));
            }
            _ => {}
        }
    }
    if cx
        .event_index
        .events
        .iter()
        .any(|event| matches!(&event.payload, TraceEventKind::Sync { .. }))
        && !seen_sync_arrive
    {
        return Err(cx.fail(
            "trace_schema_missing_sync_arrive",
            "Sync completion events require corresponding SyncArrive events",
            None,
        ));
    }
    Ok(())
}

fn trace_region_audit(cx: &mut CheckerCx<'_>) -> CheckResult {
    for event in cx.event_index.events {
        match &event.payload {
            TraceEventKind::Read { region, .. }
            | TraceEventKind::Write { region, .. }
            | TraceEventKind::TmemAlloc { region, .. }
            | TraceEventKind::TmemDealloc { region, .. } => {
                if let Err(status) = cx.region_audit.validate_region(region) {
                    return Err(status.into_diagnostic(cx, event));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn barrier_cycle_audit(cx: &mut CheckerCx<'_>) -> CheckResult {
    let mut mbars: HashMap<MbarKey, MbarCycle> = HashMap::new();
    let mut sync_counts: HashMap<SyncKey, usize> = HashMap::new();
    for event in cx.event_index.events {
        match &event.payload {
            TraceEventKind::MbarInit { target, count, .. } => {
                if *count < 0 {
                    return Err(cx.fail(
                        "barrier_cycle_invalid_count",
                        "mbarrier init count must be non-negative",
                        Some(event),
                    ));
                }
                mbars.insert(MbarKey::from_target(target), MbarCycle::new(*count));
            }
            TraceEventKind::MbarExpectTx { target, bytes, .. } => {
                let key = MbarKey::from_target(target);
                let Some(cycle) = mbars.get_mut(&key) else {
                    return Err(cx.fail(
                        "barrier_cycle_missing_init",
                        "mbarrier expect_tx has no initialized cell",
                        Some(event),
                    ));
                };
                if *bytes < 0 {
                    return Err(cx.fail(
                        "barrier_cycle_invalid_tx",
                        "mbarrier expect_tx bytes must be non-negative",
                        Some(event),
                    ));
                }
                cycle.pending_tx += *bytes;
            }
            TraceEventKind::MbarCompleteTx { target, bytes, .. } => {
                let key = MbarKey::from_target(target);
                let Some(cycle) = mbars.get_mut(&key) else {
                    return Err(cx.fail(
                        "barrier_cycle_missing_init",
                        "mbarrier complete_tx has no initialized cell",
                        Some(event),
                    ));
                };
                if *bytes < 0 || cycle.pending_tx < *bytes {
                    return Err(cx.fail(
                        "barrier_cycle_tx_underflow",
                        "mbarrier complete_tx exceeds pending bytes",
                        Some(event),
                    ));
                }
                cycle.pending_tx -= *bytes;
                let _ = cycle.complete_if_ready();
            }
            TraceEventKind::MbarArrive { target, count, .. } => {
                let key = MbarKey::from_target(target);
                let Some(cycle) = mbars.get_mut(&key) else {
                    return Err(cx.fail(
                        "barrier_cycle_missing_init",
                        "mbarrier arrive has no initialized cell",
                        Some(event),
                    ));
                };
                if *count < 0 || cycle.pending_arrivals < *count {
                    return Err(cx.fail(
                        "barrier_cycle_arrive_underflow",
                        "mbarrier arrive exceeds pending arrivals",
                        Some(event),
                    ));
                }
                cycle.pending_arrivals -= *count;
                let _ = cycle.complete_if_ready();
            }
            TraceEventKind::MbarWait { target, phase, .. } => {
                let key = MbarKey::from_target(target);
                let Some(cycle) = mbars.get(&key) else {
                    return Err(cx.fail(
                        "barrier_cycle_missing_init",
                        "mbarrier wait has no initialized cell",
                        Some(event),
                    ));
                };
                if *phase == cycle.parity {
                    return Err(cx.fail(
                        "barrier_cycle_wait_before_completion",
                        "mbarrier wait completed before the observed phase changed",
                        Some(event),
                    ));
                }
            }
            TraceEventKind::SyncArrive {
                sync_kind,
                thread_count,
                count,
                cycle,
                bar_id,
                scope,
            } => {
                sync_counts.insert(
                    SyncKey {
                        sync_kind: sync_kind.clone(),
                        thread_count: *thread_count,
                        cycle: *cycle,
                        bar_id: *bar_id,
                        resource_scope: sync_resource_scope(sync_kind, scope),
                    },
                    *count,
                );
            }
            TraceEventKind::Sync {
                sync_kind,
                thread_count,
                cycle,
                bar_id,
                scope,
            } => {
                let key = SyncKey {
                    sync_kind: sync_kind.clone(),
                    thread_count: *thread_count,
                    cycle: *cycle,
                    bar_id: *bar_id,
                    resource_scope: sync_resource_scope(sync_kind, scope),
                };
                if sync_counts.get(&key).copied().unwrap_or(0) != *thread_count {
                    return Err(cx.fail(
                        "barrier_cycle_sync_without_full_arrival",
                        "sync completion has no full SyncArrive witness",
                        Some(event),
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn ordering_analysis_pass(cx: &mut CheckerCx<'_>) -> CheckResult {
    cx.ordering = Some(OrderingAnalysis::new(cx.event_index.events));
    Ok(())
}

/// Wait-for cycle detection is disabled. On a completed trace the wait-for
/// graph is necessarily acyclic — the order in which the run's blocking ops
/// actually released is a topological sort of the graph's "releases-before"
/// edges — so `find_cycle` can never report a cycle, and a real deadlock is
/// already observed at runtime (the sampled schedule blocks with no progress,
/// the trace returns `Failed`, and the offline passes never run). The valuable,
/// non-vacuous half — validating that every blocking wait has a release witness
/// (e.g. a `wait_group` with no `commit`) — stays in `BlockingGraph::build`'s
/// node collection. Only the edge build + cycle search are gated off; they are
/// kept behind this flag for synthetic wait-for-cycle tests and possible future
/// predictive (cross-interleaving) analysis.
const DETECT_WAIT_FOR_CYCLES: bool = false;

fn deadlock_freedom(cx: &mut CheckerCx<'_>) -> CheckResult {
    let graph = match BlockingGraph::build(cx.event_index.events) {
        Ok(graph) => graph,
        Err(error) => {
            let event = cx.event_index.events.get(error.event_idx);
            let mut diagnostic = DiagnosticBuilder::new(error.code, error.message)
                .event(event)
                .build();
            diagnostic
                .details
                .insert("waiting_event".into(), error.event_idx.to_string());
            for (key, value) in error.details {
                diagnostic.details.insert(key, value);
            }
            cx.diagnostics.push(diagnostic);
            return Err(error.status);
        }
    };
    let cycle = if DETECT_WAIT_FOR_CYCLES {
        graph.find_cycle()
    } else {
        None
    };
    if let Some(cycle) = cycle {
        let witness = &graph.nodes[cycle[0]];
        let event = cx.event_index.events.get(witness.event_idx);
        let mut diagnostic = DiagnosticBuilder::new(
            "deadlock_freedom_wait_for_cycle",
            "blocking operations form a wait-for cycle with no release outside the cycle",
        )
        .event(event)
        .detail("resource", witness.resource_key.clone())
        .build();
        diagnostic
            .details
            .insert("waiting_event".into(), witness.event_idx.to_string());
        diagnostic.details.insert(
            "cycle_events".into(),
            cycle
                .iter()
                .map(|&node_idx| graph.nodes[node_idx].event_idx.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
        diagnostic.details.insert(
            "release_events".into(),
            witness
                .release_events
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
        cx.diagnostics.push(diagnostic);
        return Err(ProtocolStatus::Failed);
    }
    Ok(())
}

/// cp.async.bulk group lifetime (PTX semantics): every bulk TMA store opens
/// an async window at issue — its SMEM source is still being READ and its
/// GMEM destination still being WRITTEN until the store is committed into a
/// group (`commit_group`) and a `wait_group` retires that group. An
/// UNCOMMITTED store belongs to no group, so no wait can ever drain it: its
/// windows stay open for the rest of the trace. Same-stream conflicting
/// accesses on open windows fail immediately; cross-stream ones go through
/// the interval rule (before the issue, or after the drain).
fn async_group_lifetime_local(cx: &mut CheckerCx<'_>) -> CheckResult {
    #[derive(Default)]
    struct StreamGroups {
        pending: Vec<usize>,     // window indices not yet committed
        groups: Vec<Vec<usize>>, // committed groups, oldest first
    }
    let mut streams: HashMap<usize, StreamGroups> = HashMap::new();
    let mut windows: Vec<AsyncSourceWindow> = Vec::new();
    let mut dest_windows: Vec<AsyncSourceWindow> = Vec::new();
    // (window list, window index) pairs currently open, per stream — sources
    // (reads, conflict = write) and destinations (writes, conflict = any).
    let check_open = |cx: &mut CheckerCx<'_>,
                      streams: &HashMap<usize, StreamGroups>,
                      windows: &[AsyncSourceWindow],
                      dest_windows: &[AsyncSourceWindow],
                      region: &Region,
                      is_write: bool,
                      stream_id: usize,
                      stmt_id: usize,
                      event: &TraceEvent|
     -> CheckResult {
        let Some(state) = streams.get(&stream_id) else {
            return Ok(());
        };
        for &wi in state.pending.iter().chain(state.groups.iter().flatten()) {
            if windows[wi].close_idx.is_none()
                && is_write
                && windows[wi].stmt_id != stmt_id
                && windows[wi]
                    .regions
                    .iter()
                    .any(|source| regions_overlap(source, region))
            {
                return Err(cx.fail(
                    "async_group_source_overwrite",
                    "same-stream write overlaps an in-flight async source before wait_group",
                    Some(event),
                ));
            }
            if dest_windows[wi].close_idx.is_none()
                && dest_windows[wi].stmt_id != stmt_id
                && dest_windows[wi]
                    .regions
                    .iter()
                    .any(|dest| regions_overlap(dest, region))
            {
                return Err(cx.fail(
                    "async_group_dest_access_before_wait",
                    "same-stream access overlaps an in-flight async destination before \
                     wait_group",
                    Some(event),
                ));
            }
        }
        Ok(())
    };
    // per-STREAM pending store (two CTAs execute the SAME kernel stmt, so the
    // stmt id alone cannot key an in-flight store)
    type PendingStore = (
        usize,
        Option<Region>,
        Option<Region>,
        Option<(ReduceOp, bool)>,
    );
    let mut pending_store: HashMap<usize, PendingStore> = HashMap::new();
    fn flush_pending(
        pending: PendingStore,
        stream_id: usize,
        windows: &mut Vec<AsyncSourceWindow>,
        dest_windows: &mut Vec<AsyncSourceWindow>,
        streams: &mut HashMap<usize, StreamGroups>,
        idx: usize,
    ) {
        let (stmt_id, source, dest, reduce) = pending;
        let wi = windows.len();
        windows.push(AsyncSourceWindow {
            stream_id,
            stmt_id,
            start_idx: idx,
            close_idx: None,
            regions: source.into_iter().collect(),
            reduce: None, // source side is never a reduce dest
        });
        dest_windows.push(AsyncSourceWindow {
            stream_id,
            stmt_id,
            start_idx: idx,
            close_idx: None,
            regions: dest.into_iter().collect(),
            reduce,
        });
        streams.entry(stream_id).or_default().pending.push(wi);
    }
    for (idx, event) in cx.event_index.events.iter().enumerate() {
        let stream_id = event_stream_id(&event.payload);
        if let Some(stream_id) = stream_id {
            if pending_store
                .get(&stream_id)
                .is_some_and(|(stmt, _, _, _)| *stmt != event.stmt_id)
            {
                let pending = pending_store.remove(&stream_id).unwrap();
                flush_pending(
                    pending,
                    stream_id,
                    &mut windows,
                    &mut dest_windows,
                    &mut streams,
                    idx,
                );
            }
        }
        match &event.payload {
            TraceEventKind::CommitGroup { scope } => {
                let state = streams.entry(scope.stream_id).or_default();
                if !state.pending.is_empty() {
                    let pending = std::mem::take(&mut state.pending);
                    state.groups.push(pending);
                }
            }
            TraceEventKind::WaitGroup { n, scope } => {
                if let Some(state) = streams.get_mut(&scope.stream_id) {
                    let retain = *n as usize;
                    while state.groups.len() > retain {
                        for wi in state.groups.remove(0) {
                            windows[wi].close_idx = Some(idx);
                            dest_windows[wi].close_idx = Some(idx);
                        }
                    }
                }
            }
            TraceEventKind::Read {
                region,
                proxy: MemoryProxy::Async,
                access_kind: MemoryAccessKind::Tensor(TensorAccessKind::TmaStore),
                scope,
            } => {
                check_open(
                    cx,
                    &streams,
                    &windows,
                    &dest_windows,
                    region,
                    false,
                    scope.stream_id,
                    event.stmt_id,
                    event,
                )?;
                let entry = pending_store.entry(scope.stream_id).or_insert((
                    event.stmt_id,
                    None,
                    None,
                    None,
                ));
                entry.1 = Some(region.clone());
            }
            // An async bulk store dest — a plain TMA store OR an atomic reduce
            // (cp.reduce.async.bulk); both have the same commit/wait-group drain, so both open a
            // dest window. The reduce kind (if any) rides into the dest window for the
            // cross-stream reduce-vs-reduce skip.
            TraceEventKind::Write {
                region,
                access_kind: MemoryAccessKind::Tensor(k),
                scope,
                ..
            } if matches!(
                k,
                TensorAccessKind::TmaStore | TensorAccessKind::TmaReduce { .. }
            ) =>
            {
                let entry = pending_store.entry(scope.stream_id).or_insert((
                    event.stmt_id,
                    None,
                    None,
                    None,
                ));
                entry.2 = Some(region.clone());
                entry.3 = k.reduce();
            }
            TraceEventKind::Write { region, scope, .. } => {
                check_open(
                    cx,
                    &streams,
                    &windows,
                    &dest_windows,
                    region,
                    true,
                    scope.stream_id,
                    event.stmt_id,
                    event,
                )?;
            }
            TraceEventKind::Read { region, scope, .. } => {
                check_open(
                    cx,
                    &streams,
                    &windows,
                    &dest_windows,
                    region,
                    false,
                    scope.stream_id,
                    event.stmt_id,
                    event,
                )?;
            }
            _ => {}
        }
    }
    check_cross_stream_async_source_overwrite(cx, &windows)?;
    check_cross_stream_async_dest_access(cx, &dest_windows)?;
    Ok(())
}

/// Cross-stream interval rule for async store windows, shared by the source
/// (SMEM read; conflict = write) and destination (GMEM write; conflict = any
/// access) sides. One stream's window instances are HB-chained in trace
/// order (stores issue and groups retire oldest-first on the same stream),
/// so for a fixed access X `HB(close_i, X)` is true on a prefix of the chain
/// and `HB(X, start_i)` on a suffix: consecutive instances with an identical
/// footprint collapse into one group decided by two binary searches instead
/// of one interval test per instance.
fn check_cross_stream_async_windows(
    cx: &mut CheckerCx<'_>,
    windows: &[AsyncSourceWindow],
    include_reads: bool,
    code: &'static str,
    message: &'static str,
    window_key: &'static str,
    access_key: &'static str,
) -> CheckResult {
    struct WindowGroup<'a> {
        stream_id: usize,
        regions: &'a [Region],
        bounds: ((usize, usize), (usize, usize)),
        starts: Vec<usize>,         // ascending, HB-chained (same stream)
        closes: Vec<Option<usize>>, // non-decreasing; `None` only as a suffix
        // `Some((op, exact))` iff EVERY window merged here is an atomic reduce of that op
        // (else `None` — a mixed group is conservatively treated as non-reduce, i.e. flagged).
        reduce: Option<(ReduceOp, bool)>,
    }
    // Group by (stream, footprint) — NOT by run-length: stage-rotated source
    // buffers alternate footprints (A,B,A,B,…), and consecutive-only grouping
    // would degenerate back to one group per instance. A group's starts and
    // closes are a subsequence of the stream's HB chain, so the binary-search
    // monotonicity is unaffected by the interleaving.
    let mut groups: Vec<WindowGroup> = Vec::new();
    let mut group_index: HashMap<(usize, &[Region]), usize> = HashMap::new();
    for window in windows {
        if window.regions.is_empty() {
            continue;
        }
        match group_index.get(&(window.stream_id, window.regions.as_slice())) {
            Some(&gi) => {
                groups[gi].starts.push(window.start_idx);
                groups[gi].closes.push(window.close_idx);
                if groups[gi].reduce != window.reduce {
                    groups[gi].reduce = None; // mixed reduce/non-reduce -> conservative
                }
            }
            None => {
                let mut bounds = ((usize::MAX, 0), (usize::MAX, 0));
                for region in &window.regions {
                    let b = super::region::region_bounding_spans(region);
                    bounds = (
                        (bounds.0 .0.min(b.0 .0), bounds.0 .1.max(b.0 .1)),
                        (bounds.1 .0.min(b.1 .0), bounds.1 .1.max(b.1 .1)),
                    );
                }
                group_index.insert((window.stream_id, &window.regions), groups.len());
                groups.push(WindowGroup {
                    stream_id: window.stream_id,
                    regions: &window.regions,
                    bounds,
                    starts: vec![window.start_idx],
                    closes: vec![window.close_idx],
                    reduce: window.reduce,
                });
            }
        }
    }
    if groups.is_empty() {
        return Ok(());
    }
    let owners: HashSet<&PoolId> = groups
        .iter()
        .flat_map(|group| group.regions.iter().map(|region| &region.owner))
        .collect();
    struct IndexedAccess {
        event_idx: usize,
        stream_id: usize,
        bounds: ((usize, usize), (usize, usize)),
    }
    let mut index: HashMap<PoolId, Vec<IndexedAccess>> = HashMap::new();
    for (event_idx, event) in cx.event_index.events.iter().enumerate() {
        let (region, scope) = match &event.payload {
            TraceEventKind::Write { region, scope, .. } => (region, scope),
            TraceEventKind::Read { region, scope, .. } if include_reads => (region, scope),
            _ => continue,
        };
        if !owners.contains(&region.owner) {
            continue;
        }
        index
            .entry(region.owner.clone())
            .or_default()
            .push(IndexedAccess {
                event_idx,
                stream_id: scope.stream_id,
                bounds: super::region::region_bounding_spans(region),
            });
    }
    for group in &groups {
        let mut checked_owners = HashSet::<&PoolId>::new();
        for group_region in group.regions {
            if !checked_owners.insert(&group_region.owner) {
                continue;
            }
            let Some(accesses) = index.get(&group_region.owner) else {
                continue;
            };
            for access in accesses {
                if access.stream_id == group.stream_id {
                    continue;
                }
                if !super::region::bounding_spans_touch(group.bounds, access.bounds) {
                    continue;
                }
                let overlaps = {
                    let (TraceEventKind::Write { region, .. }
                    | TraceEventKind::Read { region, .. }) =
                        &cx.event_index.events[access.event_idx].payload
                    else {
                        unreachable!("access index only contains read/write events");
                    };
                    group
                        .regions
                        .iter()
                        .any(|window_region| regions_overlap(window_region, region))
                };
                if !overlaps {
                    continue;
                }
                // instances [0, drained) are proven drained before the access
                let drained = group.closes.partition_point(|close| {
                    close.is_some_and(|close| cx.ordering().happens_before(close, access.event_idx))
                });
                // instances [issued_after, n) are proven issued after it
                let issued_after = group.starts.partition_point(|&start| {
                    !cx.ordering().happens_before(access.event_idx, start)
                });
                if issued_after <= drained {
                    continue; // every instance is on one safe side
                }
                // Commutative-atomic-reduce skip. If the in-flight store window AND the
                // conflicting access are BOTH atomic reduces of the SAME op, the conflict is
                // RACE-FREE: hardware atomicity serializes the RMWs and commutativity makes the
                // order not matter for the value. NOT skipped (still a gap): a plain Read (would
                // see a partial sum), a plain Write (tears the atomic — e.g. a software
                // accumulate, which carries no TmaReduce marker), or a different reduce op. A
                // float reduction is order-dependent -> a non-determinism WARNING, not a race.
                let (reduce_safe, nondeterministic) = match (
                    group.reduce,
                    reduce_access(&cx.event_index.events[access.event_idx].payload),
                ) {
                    (Some((op_w, ex_w)), Some((op_a, ex_a))) if op_w == op_a => {
                        (true, !(ex_w && ex_a))
                    }
                    _ => (false, false),
                };
                if reduce_safe {
                    if nondeterministic {
                        cx.warn(
                            "nondeterministic_reduction",
                            "cross-stream atomic reduce-adds to one location are race-free but \
                             order-dependent (float reduction is not associative): the result is \
                             not bit-reproducible. Use the deterministic path for reproducibility.",
                        );
                    }
                    continue;
                }
                let violation = drained; // first instance proven on neither side
                let mut details = BTreeMap::new();
                details.insert(window_key.into(), group.starts[violation].to_string());
                if let Some(close_idx) = group.closes[violation] {
                    details.insert("drain_event".into(), close_idx.to_string());
                }
                details.insert(access_key.into(), access.event_idx.to_string());
                cx.gap(code, message, Some(access.event_idx), details);
            }
        }
    }
    Ok(())
}

fn check_cross_stream_async_source_overwrite(
    cx: &mut CheckerCx<'_>,
    windows: &[AsyncSourceWindow],
) -> CheckResult {
    check_cross_stream_async_windows(
        cx,
        windows,
        false,
        "async_group_cross_stream_unordered_source_overwrite",
        "cross-stream write overlaps an async source without proof it is before source read or after drain",
        "source_event",
        "writer_event",
    )
}

/// Destination-side twin: the store's GMEM write is in flight until the
/// group drains; any cross-stream access (read or write) overlapping it must
/// be before the issue or after the drain.
fn check_cross_stream_async_dest_access(
    cx: &mut CheckerCx<'_>,
    windows: &[AsyncSourceWindow],
) -> CheckResult {
    check_cross_stream_async_windows(
        cx,
        windows,
        true,
        "async_group_cross_stream_dest_access",
        "cross-stream access overlaps an async store destination without proof it is before the store or after its drain",
        "store_event",
        "access_event",
    )
}

fn event_stream_id(payload: &TraceEventKind) -> Option<usize> {
    match payload {
        TraceEventKind::Read { scope, .. }
        | TraceEventKind::Write { scope, .. }
        | TraceEventKind::CommitGroup { scope }
        | TraceEventKind::WaitGroup { scope, .. }
        | TraceEventKind::MbarArrive { scope, .. }
        | TraceEventKind::MbarExpectTx { scope, .. }
        | TraceEventKind::MbarCompleteTx { scope, .. }
        | TraceEventKind::MbarWait { scope, .. }
        | TraceEventKind::MbarInit { scope, .. }
        | TraceEventKind::TmemWait { scope, .. } => Some(scope.stream_id),
        _ => None,
    }
}

/// tcgen05 async-window hazards. The engine ops (mma / cp / ld / st) are
/// ASYNC: each opens a window at issue that closes only at its drain — the
/// stream's `tcgen05.commit` for mma/cp (pipe-wide), `tcgen05.wait_*` for
/// ld/st. The plain race check pins each access at its issue point, so it is
/// blind to anything that lands INSIDE a window with happens-before edges to
/// both ends (e.g. a buffer released by a thread arrive right after the
/// issue: the write is ordered after the issue and before the drain — fully
/// "ordered", still corrupt). Three rules over one trace sweep:
///
/// 1. same-stream TMEM accesses conflicting with an open window
///    (`tmem_async_overlap`) — mma/cp pairs exempt (the pipe executes them in
///    issue order);
/// 2. ANY write landing on an open mma/cp SMEM operand-read window
///    (`tcgen05_operand_overwrite_before_drain`);
/// 3. cross-stream TMEM accesses conflicting with an open window
///    (`tmem_access_before_drain`) — same pipe exemption; genuinely unordered
///    pairs are also memory_data_race, but a pair "synchronized" via a
///    non-drain edge is ONLY caught here.
///
/// Trace order respects the modeled happens-before edges, so an access a
/// kernel orders against the drain can never appear before the closing
/// commit/wait in the trace; one ordered only against the issue can, and the
/// round-robin scheduler exhibits it within the pipeline's next stage.
fn tcgen05_async_hazard(cx: &mut CheckerCx<'_>) -> CheckResult {
    struct OperandWindow {
        region: Region,
        stmt_id: usize,
    }
    type CellKey = (u32, usize, usize, u32);
    fn cell_key(target: &MbarTargetEvent) -> CellKey {
        (
            target.mbar_id,
            target.cluster_id,
            target.ctaid_in_cluster,
            target.stage,
        )
    }
    // Windows indexed owner-first: regions in different pools never overlap,
    // so cross-stream scans only walk the same-owner submap (different
    // clusters / CTAs are skipped wholesale).
    let mut open: HashMap<PoolId, HashMap<usize, Vec<TmemWindow>>> = HashMap::new();
    let mut operand_open: HashMap<PoolId, HashMap<usize, Vec<OperandWindow>>> = HashMap::new();
    // TMA-load SMEM writes are async until the load's tx completes its
    // mbarrier AND someone waits it: one window per tile-write event, bound
    // to the stmt's complete-tx cell by emission order (the i-th tile write
    // pairs with the i-th completed cell — also correct when the barrier
    // lives on a DIFFERENT CTA than the data), closed by the first MbarWait
    // on that cell. An access landing on an open window is ordered against
    // the load's ISSUE at best, never its completion.
    // per-STREAM pending load (two CTAs execute the SAME kernel stmt)
    let mut tma_pending: HashMap<usize, (usize, Vec<Region>, Vec<CellKey>)> = HashMap::new();
    let mut tma_open: HashMap<CellKey, Vec<(usize, Region)>> = HashMap::new();
    // The rule tracks loads whose mbar cell is never waited ANYWHERE in the
    // trace: the fake-release core (a consumer "synchronized" by something
    // other than the load's own barrier). Presence, not position, is what
    // this window needs — a load whose barrier nobody ever waits was never
    // delivered, whenever the waits happen to sit.
    let mut waited_cells: HashSet<CellKey> = HashSet::new();
    for event in cx.event_index.events.iter() {
        if let TraceEventKind::MbarWait { target, .. } = &event.payload {
            waited_cells.insert(cell_key(target));
        }
    }
    for event in cx.event_index.events.iter() {
        if let Some(stream_id) = event_stream_id(&event.payload) {
            if tma_pending
                .get(&stream_id)
                .is_some_and(|(stmt, _, _)| *stmt != event.stmt_id)
            {
                let (stmt, regions, cells) = tma_pending.remove(&stream_id).unwrap();
                for (region, cell) in regions.into_iter().zip(cells) {
                    if !waited_cells.contains(&cell) {
                        tma_open.entry(cell).or_default().push((stmt, region));
                    }
                }
            }
        }
        match &event.payload {
            TraceEventKind::Read {
                region,
                access_kind: MemoryAccessKind::Tmem(async_kind),
                scope,
                ..
            } => {
                let window = TmemWindow::new(event, *async_kind, TmemAccessMode::Read, region);
                let owner_open = open.entry(region.owner.clone()).or_default();
                check_tmem_window_conflicts(cx, owner_open.get(&scope.stream_id), &window, event)?;
                check_cross_stream_window_conflicts(
                    cx,
                    owner_open,
                    scope.stream_id,
                    &window,
                    event,
                )?;
                push_tmem_window_dedup(owner_open.entry(scope.stream_id).or_default(), window);
            }
            TraceEventKind::Read { region, scope, .. } => {
                check_tma_window_access(cx, &tma_open, region, event)?;
                if let TraceEventKind::Read {
                    access_kind:
                        MemoryAccessKind::Tensor(
                            TensorAccessKind::Tcgen05Mma | TensorAccessKind::Tcgen05Cp,
                        ),
                    ..
                } = &event.payload
                {
                    // a live identical window adds no constraint: dedup keeps
                    // the per-stream list bounded by distinct (stmt, region)
                    let windows = operand_open
                        .entry(region.owner.clone())
                        .or_default()
                        .entry(scope.stream_id)
                        .or_default();
                    if !windows
                        .iter()
                        .any(|w| w.stmt_id == event.stmt_id && w.region == *region)
                    {
                        windows.push(OperandWindow {
                            region: region.clone(),
                            stmt_id: event.stmt_id,
                        });
                    }
                }
            }
            TraceEventKind::Write {
                region,
                access_kind,
                scope,
                ..
            } => {
                if matches!(
                    access_kind,
                    MemoryAccessKind::Tensor(TensorAccessKind::TmaLoad)
                ) {
                    tma_pending
                        .entry(scope.stream_id)
                        .or_insert((event.stmt_id, Vec::new(), Vec::new()))
                        .1
                        .push(region.clone());
                }
                check_tma_window_access(cx, &tma_open, region, event)?;
                if let MemoryAccessKind::Tmem(async_kind) = access_kind {
                    let window = TmemWindow::new(event, *async_kind, TmemAccessMode::Write, region);
                    let owner_open = open.entry(region.owner.clone()).or_default();
                    check_tmem_window_conflicts(
                        cx,
                        owner_open.get(&scope.stream_id),
                        &window,
                        event,
                    )?;
                    check_cross_stream_window_conflicts(
                        cx,
                        owner_open,
                        scope.stream_id,
                        &window,
                        event,
                    )?;
                    push_tmem_window_dedup(owner_open.entry(scope.stream_id).or_default(), window);
                }
                if let Some(owner_windows) = operand_open.get(&region.owner) {
                    for (&stream, windows) in owner_windows.iter() {
                        for window in windows {
                            if stream == scope.stream_id && window.stmt_id == event.stmt_id {
                                continue;
                            }
                            if regions_overlap(&window.region, region) {
                                return Err(cx.fail(
                                    "tcgen05_operand_overwrite_before_drain",
                                    "write overlaps an in-flight tcgen05 operand read before the \
                                     issuing stream's tcgen05.commit drains it",
                                    Some(event),
                                ));
                            }
                        }
                    }
                }
            }
            TraceEventKind::MbarCompleteTx { target, scope, .. }
                if event.stmt_kind == "TmaLoad" =>
            {
                if let Some((stmt, _, cells)) = tma_pending.get_mut(&scope.stream_id) {
                    if *stmt == event.stmt_id {
                        cells.push(cell_key(target));
                    }
                }
            }
            TraceEventKind::MbarWait { target, .. } => {
                tma_open.remove(&cell_key(target));
            }
            TraceEventKind::TmemWait { async_kind, scope } => {
                for owner_open in open.values_mut() {
                    if let Some(windows) = owner_open.get_mut(&scope.stream_id) {
                        windows.retain(|window| window.async_kind != *async_kind);
                    }
                }
            }
            TraceEventKind::MbarArrive { scope, .. } if event.stmt_kind == "Tcgen05Commit" => {
                // tcgen05.commit observes completion of ALL prior async tcgen05
                // ops from this stream — mma and cp alike, TMEM and SMEM sides.
                for owner_open in open.values_mut() {
                    if let Some(windows) = owner_open.get_mut(&scope.stream_id) {
                        windows.retain(|window| {
                            !matches!(window.async_kind, TmemAsyncKind::Mma | TmemAsyncKind::Cp)
                        });
                    }
                }
                for owner_windows in operand_open.values_mut() {
                    owner_windows.remove(&scope.stream_id);
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Cheap bounding-span reject before the box-merge overlap test (regions'
/// boxes are sorted): disjoint overall byte spans can never overlap.
fn region_bounds_overlap(a: &Region, b: &Region) -> bool {
    match (a.boxes.rank1_bounds(), b.boxes.rank1_bounds()) {
        (Some((a_start, a_end)), Some((b_start, b_end))) => a_start < b_end && b_start < a_end,
        _ => false,
    }
}

fn check_tma_window_access(
    cx: &mut CheckerCx<'_>,
    tma_open: &HashMap<(u32, usize, usize, u32), Vec<(usize, Region)>>,
    region: &Region,
    event: &TraceEvent,
) -> CheckResult {
    for windows in tma_open.values() {
        for (stmt, window) in windows {
            if *stmt == event.stmt_id {
                continue;
            }
            if window.owner == region.owner
                && region_bounds_overlap(window, region)
                && regions_overlap(window, region)
            {
                return Err(cx.fail(
                    "tma_load_access_before_mbar_wait",
                    "access overlaps a TMA-loaded tile before any wait on the load's mbarrier",
                    Some(event),
                ));
            }
        }
    }
    Ok(())
}

fn check_cross_stream_window_conflicts(
    cx: &mut CheckerCx<'_>,
    open: &HashMap<usize, Vec<TmemWindow>>, // same-owner submap
    stream_id: usize,
    current: &TmemWindow,
    event: &TraceEvent,
) -> CheckResult {
    for (&stream, windows) in open.iter() {
        if stream == stream_id {
            continue;
        }
        for window in windows {
            if matches!(window.async_kind, TmemAsyncKind::Mma | TmemAsyncKind::Cp)
                && matches!(current.async_kind, TmemAsyncKind::Mma | TmemAsyncKind::Cp)
            {
                continue;
            }
            if !window.is_conflicting_access(current) || !window.overlaps(current) {
                continue;
            }
            return Err(cx.fail(
                "tmem_access_before_drain",
                "TMEM access conflicts with another stream's open async window before its \
                 drain (tcgen05.commit / wait)",
                Some(event),
            ));
        }
    }
    Ok(())
}

fn check_tmem_window_conflicts(
    cx: &mut CheckerCx<'_>,
    open: Option<&Vec<TmemWindow>>,
    current: &TmemWindow,
    event: &TraceEvent,
) -> CheckResult {
    let Some(open) = open else {
        return Ok(());
    };
    for window in open {
        if window.stmt_id == current.stmt_id {
            continue;
        }
        // tcgen05 pipe ops (mma / cp) issued by one stream execute in issue order,
        // so an open window of either kind cannot race a later same-stream access
        // of either kind (PTX: tcgen05 operations are performed in issue order).
        if matches!(window.async_kind, TmemAsyncKind::Mma | TmemAsyncKind::Cp)
            && matches!(current.async_kind, TmemAsyncKind::Mma | TmemAsyncKind::Cp)
        {
            continue;
        }
        if !window.is_conflicting_access(current) || !window.overlaps(current) {
            continue;
        }
        return Err(cx.fail(
            "tmem_async_overlap",
            "same-stream TMEM async windows overlap before the earlier window is closed",
            Some(event),
        ));
    }
    Ok(())
}

/// A TMEM access, with the point at which the trace records it as landed.
struct TmemAccess {
    event_idx: usize,
    region: Region,
    async_kind: TmemAsyncKind,
    /// The `tcgen05.wait` that retired it, when the trace records one.
    drain_idx: Option<usize>,
}

/// One `alloc .. dealloc` interval of a TMEM band.
struct TmemGeneration {
    alloc_idx: usize,
    dealloc_idx: Option<usize>,
    region: Region,
}

/// TMEM band lifetimes.
///
/// Two halves that need different machinery. The RESOURCE ALGEBRA — a band
/// may not be allocated over a live one, a dealloc must name a live band —
/// is about the allocator's own bookkeeping and has no ordering content. The
/// LIFETIME is pure happens-before: an access must be ordered after the
/// allocation it uses and before that generation's dealloc, and a band may
/// only be reallocated after the previous generation's dealloc. Those edges
/// have to come from real synchronization (an mbarrier, a cta/cluster sync);
/// trace order proves nothing, since the simulator ran one schedule and a
/// different one could put the access before the alloc or after the free.
///
/// The trace walk below therefore only ENUMERATES generations and accesses.
/// Which generation an access belongs to is decided by the clock, not by
/// where the walk happened to be — and because overlapping generations must
/// themselves be happens-before ordered, that binding is unambiguous.
///
/// `tcgen05.alloc/dealloc` and `tcgen05.ld/st` are all `.sync.aligned`, so
/// both ends of every obligation are warp-converged events and the
/// whole-event relation is exactly the right one here.
fn tmem_lifecycle_order_local(cx: &mut CheckerCx<'_>) -> CheckResult {
    let mut live: HashMap<PoolId, Vec<usize>> = HashMap::new();
    let mut generations: Vec<TmemGeneration> = Vec::new();
    let mut accesses: Vec<TmemAccess> = Vec::new();
    // Per stream, the `tcgen05.ld`/`st` accesses not yet drained by a
    // `tcgen05.wait` of their kind.
    let mut undrained: HashMap<usize, Vec<usize>> = HashMap::new();
    // Per stream, the `tcgen05.mma`/`cp` accesses not yet observed to
    // complete, and per mbar cell the accesses some commit handed to it. A
    // commit makes the barrier track EVERY async op the warp issued before
    // it, so a later commit covers them again — waiting any one of those
    // cells proves completion.
    let mut undrained_async: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut committed: HashMap<MbarKey, Vec<usize>> = HashMap::new();
    for (event_idx, event) in cx.event_index.events.iter().enumerate() {
        match &event.payload {
            TraceEventKind::TmemAlloc { region, .. } => {
                let open = live.entry(region.owner.clone()).or_default();
                if open
                    .iter()
                    .any(|g| regions_overlap(&generations[*g].region, region))
                {
                    return Err(cx.fail(
                        "tmem_lifecycle_allocation_overlap",
                        "TMEM allocation overlaps an active allocation",
                        Some(event),
                    ));
                }
                open.push(generations.len());
                generations.push(TmemGeneration {
                    alloc_idx: event_idx,
                    dealloc_idx: None,
                    region: region.clone(),
                });
            }
            TraceEventKind::Read {
                region,
                access_kind: MemoryAccessKind::Tmem(async_kind),
                scope,
                ..
            }
            | TraceEventKind::Write {
                region,
                access_kind: MemoryAccessKind::Tmem(async_kind),
                scope,
                ..
            } => {
                match async_kind {
                    TmemAsyncKind::Ld | TmemAsyncKind::St => undrained
                        .entry(scope.stream_id)
                        .or_default()
                        .push(accesses.len()),
                    TmemAsyncKind::Mma | TmemAsyncKind::Cp => undrained_async
                        .entry(scope.stream_id)
                        .or_default()
                        .push(accesses.len()),
                }
                accesses.push(TmemAccess {
                    event_idx,
                    region: region.clone(),
                    async_kind: *async_kind,
                    drain_idx: None,
                });
            }
            // `tcgen05.wait::ld/st` retires the EXECUTING thread's own loads
            // or stores of that kind — the point at which they are known to
            // have landed, as opposed to merely issued.
            TraceEventKind::TmemWait { async_kind, scope } => {
                if let Some(pending) = undrained.get_mut(&scope.stream_id) {
                    pending.retain(|a| {
                        if accesses[*a].async_kind == *async_kind {
                            accesses[*a].drain_idx = Some(event_idx);
                            false
                        } else {
                            true
                        }
                    });
                }
            }
            // `tcgen05.commit` makes an mbarrier track the issuing warp's
            // prior async work; the completion is OBSERVED when someone waits
            // that cell, and a parked wait is written into the trace at the
            // moment it wakes, so its position is the observation point.
            TraceEventKind::MbarArrive { target, scope, .. }
                if event.stmt_kind == "Tcgen05Commit" =>
            {
                if let Some(pending) = undrained_async.get(&scope.stream_id) {
                    committed
                        .entry(MbarKey::from_target(target))
                        .or_default()
                        .extend(pending.iter().copied());
                }
            }
            TraceEventKind::MbarWait { target, .. } => {
                if let Some(tracked) = committed.remove(&MbarKey::from_target(target)) {
                    for a in &tracked {
                        if accesses[*a].drain_idx.is_none() {
                            accesses[*a].drain_idx = Some(event_idx);
                        }
                    }
                    for pending in undrained_async.values_mut() {
                        pending.retain(|a| accesses[*a].drain_idx.is_none());
                    }
                }
            }
            TraceEventKind::TmemDealloc { region, .. } => {
                let open = live.entry(region.owner.clone()).or_default();
                let Some(pos) = open.iter().position(|g| generations[*g].region == *region) else {
                    return Err(cx.fail(
                        "tmem_lifecycle_dealloc_without_allocation",
                        "TMEM deallocation does not match an active allocation",
                        Some(event),
                    ));
                };
                generations[open.remove(pos)].dealloc_idx = Some(event_idx);
            }
            _ => {}
        }
    }

    // Reuse of a band must wait for the previous generation to be freed, and
    // the freeing warp and the re-allocating warp must actually synchronize.
    for (j, later) in generations.iter().enumerate() {
        for earlier in &generations[..j] {
            if !regions_overlap(&earlier.region, &later.region) {
                continue;
            }
            let ordered = earlier
                .dealloc_idx
                .is_some_and(|d| cx.ordering().happens_before(d, later.alloc_idx));
            if !ordered {
                let event = cx.event_index.events.get(later.alloc_idx);
                return Err(cx.fail(
                    "tmem_lifecycle_hb_missing",
                    "TMEM re-allocation has no happens-before edge from the previous \
                     generation's deallocation; the freeing warp and the re-allocating warp \
                     must synchronize, not just land in a favourable interleaving",
                    event,
                ));
            }
        }
    }

    // Bind each access to the generation it uses — the one whose band covers
    // it, whose allocation it is ordered after, and whose free it is not
    // already past — then require that it retires before that free.
    for access in &accesses {
        let (access_idx, region) = (&access.event_idx, &access.region);
        let generation = generations.iter().find(|g| {
            g.region.owner == region.owner
                && region_covers(&g.region, region)
                && cx.ordering().happens_before(g.alloc_idx, *access_idx)
                && !g
                    .dealloc_idx
                    .is_some_and(|d| cx.ordering().happens_before(d, *access_idx))
        });
        let Some(generation) = generation else {
            let event = cx.event_index.events.get(*access_idx);
            return Err(cx.fail(
                "tmem_lifecycle_use_without_allocation",
                "TMEM access is not covered by an allocation it is ordered after; the \
                 allocating warp and the accessing warp must synchronize",
                event,
            ));
        };
        if let Some(dealloc_idx) = generation.dealloc_idx {
            // Ordering the ISSUE before the free is not enough for an async
            // access: a barrier orders instruction streams, it does not drain
            // an engine. The free must be ordered after the COMPLETION
            // OBSERVATION — `tcgen05.wait::ld/st` for a load or store, the
            // wait on the mbarrier a `tcgen05.commit` handed the work to for
            // an mma or cp.
            let must_precede = access.drain_idx.unwrap_or(*access_idx);
            if !cx.ordering().happens_before(must_precede, dealloc_idx) {
                let event = cx.event_index.events.get(dealloc_idx);
                let message = if access.drain_idx.is_some() {
                    "TMEM deallocation has no happens-before edge from the drain that retires \
                     an access of that generation; a barrier orders the issue, it does not \
                     complete the access"
                } else {
                    "TMEM deallocation has no happens-before edge from an access of that \
                     generation; every accessing warp must be ordered before the free"
                };
                return Err(cx.fail("tmem_lifecycle_hb_missing", message, event));
            }
            if access.drain_idx.is_none() {
                let event = cx.event_index.events.get(*access_idx);
                return Err(cx.fail(
                    "tmem_lifecycle_use_not_drained",
                    "TMEM access is never observed to complete — a `tcgen05.wait` of its kind \
                     for a load or store, an awaited `tcgen05.commit` for an mma or cp — so it \
                     is not known to have landed before the band is freed",
                    event,
                ));
            }
        }
    }
    Ok(())
}

fn memory_race_check(cx: &mut CheckerCx<'_>) -> CheckResult {
    let mut frontiers: HashMap<PoolId, MemoryRaceFrontier> = HashMap::new();
    for (event_idx, event) in cx.event_index.events.iter().enumerate() {
        let Some(record) = MemoryAccessRecord::from_shared_event(event_idx, event) else {
            continue;
        };
        frontiers
            .entry(record.region.owner.clone())
            .or_default()
            .process(cx, record)?;
    }
    Ok(())
}

fn is_shared_race_target(owner: &PoolId) -> bool {
    matches!(owner, PoolId::Smem { .. } | PoolId::Tmem { .. })
}

/// Warp-collective statements (PTX `.sync.aligned`): every lane of the warp
/// converges on the instruction, executes it together, and its effects are
/// warp-visible once it completes — so everything any lane executed before it
/// completes before it, and everything after starts after it. The set holds
/// exactly the instructions with that property, including the TMEM
/// allocator ops (`tcgen05.alloc/dealloc` are `.sync.aligned`, PTX §9.7.17).
/// `tcgen05.wait::ld/st` drains the EXECUTING thread's own ld/st (PTX defines
/// it per-thread) and `tcgen05.commit`/`mma` are single-thread issues, so
/// none of them converge anything.
fn is_warp_collective_stmt(stmt_kind: &str) -> bool {
    matches!(
        stmt_kind,
        "LdMatrix"
            | "StMatrix"
            | "Tcgen05Ld"
            | "Tcgen05St"
            | "WarpMma"
            | "TmemAlloc"
            | "TmemDealloc"
    )
}

/// Every conflicting lane-slice pair of the two accesses carries a
/// happens-before order: `Ok(())` when the pair is race-free, else
/// `Err(witness)` naming the first unordered `(prior_lane, current_lane)`
/// slice pair (`None` when neither side is lane-attributed).
///
/// A slice is `(lane, bytes)`: the region's `lane_boxes` entry when the
/// builder attributed bytes to lanes, else every executing lane touching the
/// whole footprint. Same-lane pairs of one stream are program-ordered;
/// every other pair must come out of the clock — and a release projects only
/// its arriving lanes, so a masked write no lane published can never satisfy
/// this. Cross-stream pairs may be ordered either way; same-stream slices
/// only forward (a later event's lane tick is never below an earlier one's).
fn ordered_conflict(
    cx: &CheckerCx<'_>,
    prior: &MemoryAccessRecord,
    current: &MemoryAccessRecord,
) -> Result<(), ConflictWitness> {
    lane_slices_ordered(cx, prior, current).map_err(|lanes| ConflictWitness {
        lanes,
        unpublished: false,
    })?;
    // Ordering is necessary but not sufficient across the proxy boundary: a
    // lane's store reaches an ENGINE only through `fence.proxy.async`, so an
    // ordered pair whose bytes were never published is still a bug (and an
    // unordered one is a bug even if they were).
    if current.crosses_into_engine(prior) {
        let write_ordinal = cx.ordering().event_ordinal(prior.event_idx);
        let unpublished = |pl: u8| {
            !cx.ordering()
                .engine_sees(prior.stream_id, pl, write_ordinal, current.event_idx)
        };
        let offender = match prior.region.lane_boxes.as_deref() {
            Some(prior_lanes) => prior_lanes
                .iter()
                .find(|(pl, pb)| {
                    super::region::box_overlaps_region(pb, &current.region) && unpublished(*pl)
                })
                .map(|(pl, _)| *pl),
            None => (0..32u8)
                .filter(|l| prior.mask & (1u32 << l) != 0)
                .find(|pl| unpublished(*pl)),
        };
        if let Some(pl) = offender {
            return Err(ConflictWitness {
                lanes: Some((pl, 0)),
                unpublished: true,
            });
        }
    }
    Ok(())
}

/// Why a conflicting pair failed, and on which lane slices.
struct ConflictWitness {
    lanes: Option<(u8, u8)>,
    /// The pair is ordered, but the lane's store never crossed the proxy
    /// boundary into the engine that reads it.
    unpublished: bool,
}

/// The ordinary per-lane ordering obligation, with no proxy considerations.
fn lane_slices_ordered(
    cx: &CheckerCx<'_>,
    prior: &MemoryAccessRecord,
    current: &MemoryAccessRecord,
) -> Result<(), Option<(u8, u8)>> {
    let same_stream = prior.stream_id == current.stream_id;
    // The whole-event relation (every slice of one before every slice of the
    // other) settles every pair at the cost of one clock sweep — the common
    // case for converged same-stream pairs and release-chained cross-stream
    // pairs. Only when it fails does the pair need slice-level resolution.
    if cx
        .ordering()
        .happens_before(prior.event_idx, current.event_idx)
        || (!same_stream
            && cx
                .ordering()
                .happens_before(current.event_idx, prior.event_idx))
    {
        return Ok(());
    }
    let pair_ordered = |pl: u8, cl: u8| -> bool {
        cx.ordering()
            .ordered_lane(prior.event_idx, pl, current.event_idx, cl)
            || (!same_stream
                && cx
                    .ordering()
                    .ordered_lane(current.event_idx, cl, prior.event_idx, pl))
    };
    let mask_lanes = |mask: u32| (0..32u8).filter(move |l| mask & (1u32 << l) != 0);
    match (
        prior.region.lane_boxes.as_deref(),
        current.region.lane_boxes.as_deref(),
    ) {
        (Some(prior_lanes), Some(current_lanes)) => {
            for (pl, pb) in prior_lanes.iter() {
                for (cl, cb) in current_lanes.iter() {
                    if boxes_overlap(pb, cb) && !pair_ordered(*pl, *cl) {
                        return Err(Some((*pl, *cl)));
                    }
                }
            }
        }
        (Some(prior_lanes), None) => {
            for (pl, pb) in prior_lanes.iter() {
                if !super::region::box_overlaps_region(pb, &current.region) {
                    continue;
                }
                for cl in mask_lanes(current.mask) {
                    if !pair_ordered(*pl, cl) {
                        return Err(Some((*pl, cl)));
                    }
                }
            }
        }
        (None, Some(current_lanes)) => {
            for (cl, cb) in current_lanes.iter() {
                if !super::region::box_overlaps_region(cb, &prior.region) {
                    continue;
                }
                for pl in mask_lanes(prior.mask) {
                    if !pair_ordered(pl, *cl) {
                        return Err(Some((pl, *cl)));
                    }
                }
            }
        }
        (None, None) => {
            // Both uniform: every slice pair conflicts, so the whole-event
            // relation above was already the exact test.
            return Err(None);
        }
    }
    Ok(())
}

/// Both sides lane-attributed with no byte shared across DIFFERENT lanes.
/// Combined with the caller's footprint-coverage check, every prior lane
/// slice is then covered by the SAME lane's current slice, and per-lane
/// program order supplies an ordering that composes with any later conflict
/// (prior slice ≤ covering current slice ≤ whatever current is judged
/// against) — the condition frontier pruning needs. A side without
/// attribution is uniform (every executing lane touches the whole region)
/// and can never be same-lane-covered.
fn same_lane_coverage(prior: &MemoryAccessRecord, current: &MemoryAccessRecord) -> bool {
    let (Some(prior_lanes), Some(current_lanes)) =
        (&prior.region.lane_boxes, &current.region.lane_boxes)
    else {
        return false;
    };
    for (pl, pb) in prior_lanes.iter() {
        for (cl, cb) in current_lanes.iter() {
            if pl != cl && boxes_overlap(pb, cb) {
                return false;
            }
        }
    }
    true
}

/// Is a covered frontier entry retirable under `current`? Pruning requires an
/// ordering that COMPOSES: any later W conflicting with `prior` also
/// conflicts with `current` (coverage, checked by the caller), and
/// prior-before-current chained with the current-vs-W judgment settles
/// prior-vs-W. The whole-event relation composes by transitivity; same-lane
/// coverage composes through per-lane program order. An async-engine prior
/// needs no frontier entry at all — its hazards live in the engine-window
/// passes (`async_group_lifetime`, `tcgen05_async_hazard`), not in lane
/// clocks.
fn prune_ordered(
    cx: &CheckerCx<'_>,
    prior: &MemoryAccessRecord,
    current: &MemoryAccessRecord,
) -> bool {
    if prior.async_proxy {
        return true;
    }
    if cx
        .ordering()
        .happens_before(prior.event_idx, current.event_idx)
    {
        return true;
    }
    prior.stream_id == current.stream_id && same_lane_coverage(prior, current)
}

#[derive(Default)]
struct MemoryRaceFrontier {
    reads: Vec<MemoryAccessRecord>,
    writes: Vec<MemoryAccessRecord>,
}

impl MemoryRaceFrontier {
    fn process(&mut self, cx: &mut CheckerCx<'_>, current: MemoryAccessRecord) -> CheckResult {
        match current.mode {
            MemoryAccessMode::Read => {
                check_memory_conflicts(cx, &self.writes, &current)?;
                prune_read_frontier_before_read(&mut self.reads, &current, cx);
                self.reads.push(current);
            }
            MemoryAccessMode::Write => {
                check_memory_conflicts(cx, &self.writes, &current)?;
                check_memory_conflicts(cx, &self.reads, &current)?;
                self.prune_covered_before(&current, cx);
                self.writes.push(current);
            }
        }
        Ok(())
    }

    fn prune_covered_before(&mut self, current: &MemoryAccessRecord, cx: &CheckerCx<'_>) {
        prune_frontier_before(&mut self.reads, current, cx);
        prune_frontier_before(&mut self.writes, current, cx);
    }
}

fn prune_read_frontier_before_read(
    frontier: &mut Vec<MemoryAccessRecord>,
    current: &MemoryAccessRecord,
    cx: &CheckerCx<'_>,
) {
    frontier.retain(|prior| {
        !super::region::bounding_spans_contain(current.bounds, prior.bounds)
            || !region_covers(&current.region, &prior.region)
            || !prune_ordered(cx, prior, current)
    });
}

fn prune_frontier_before(
    frontier: &mut Vec<MemoryAccessRecord>,
    current: &MemoryAccessRecord,
    cx: &CheckerCx<'_>,
) {
    let mut kept = Vec::with_capacity(frontier.len());
    for prior in frontier.drain(..) {
        if super::region::bounding_spans_touch(prior.bounds, current.bounds)
            && regions_overlap(&prior.region, &current.region)
            && prune_ordered(cx, &prior, current)
        {
            if !(super::region::bounding_spans_contain(current.bounds, prior.bounds)
                && region_covers(&current.region, &prior.region))
            {
                kept.push(prior);
            }
        } else {
            kept.push(prior);
        }
    }
    *frontier = kept;
}

fn check_memory_conflicts(
    cx: &mut CheckerCx<'_>,
    frontier: &[MemoryAccessRecord],
    current: &MemoryAccessRecord,
) -> CheckResult {
    for prior in frontier {
        let same_stream = prior.stream_id == current.stream_id;
        if same_stream
            && (prior.async_proxy || current.async_proxy)
            && !current.crosses_into_engine(prior)
        {
            // An engine access conflicts through the engine's WINDOW — the
            // span from issue to the observed drain, which no pair of event
            // clocks describes — so the passes that model that window
            // (`async_group_lifetime`, `tcgen05_async_hazard`) own it. The
            // one exception is a lane's generic store feeding the engine:
            // that is a proxy-publication question the clock does answer.
            continue;
        }
        if !super::region::bounding_spans_touch(prior.bounds, current.bounds) {
            continue;
        }
        if !regions_overlap(&prior.region, &current.region) {
            continue;
        }
        if let Err(witness) = ordered_conflict(cx, prior, current) {
            report_memory_data_race(cx, prior, current, witness);
            return Err(ProtocolStatus::Failed);
        }
    }
    Ok(())
}

fn report_memory_data_race(
    cx: &mut CheckerCx<'_>,
    left: &MemoryAccessRecord,
    right: &MemoryAccessRecord,
    witness: ConflictWitness,
) {
    let ConflictWitness { lanes, unpublished } = witness;
    let left_event = cx.event_index.events.get(left.event_idx);
    let right_event = cx.event_index.events.get(right.event_idx);
    // Witness bytes: the named lane slices' first overlapping box pair when
    // the unordered pair is lane-attributed, else the region overlap.
    let (lanes_detail, witness) = match (lanes, &left.region.lane_boxes, &right.region.lane_boxes) {
        (Some((ll, rl)), Some(left_lanes), Some(right_lanes)) => {
            let witness = left_lanes
                .iter()
                .filter(|(l, _)| *l == ll)
                .flat_map(|(_, lb)| {
                    right_lanes
                        .iter()
                        .filter(|(r, _)| *r == rl)
                        .filter(|(_, rb)| boxes_overlap(lb, rb))
                        .map(|(_, rb)| {
                            let ranges = lb
                                .ranges
                                .iter()
                                .zip(&rb.ranges)
                                .map(|(&(ls, le), &(rs, re))| (ls.max(rs), le.min(re)))
                                .collect();
                            box_summary(&BoxN::new(ranges))
                        })
                })
                .next();
            (
                format!("{ll},{rl}"),
                witness.unwrap_or_else(|| "<unknown>".into()),
            )
        }
        (Some((ll, rl)), _, _) => (
            format!("{ll},{rl}"),
            region_overlap_witness(&left.region, &right.region)
                .map(|b| box_summary(&b))
                .unwrap_or_else(|| "<unknown>".into()),
        ),
        (None, _, _) => (
            "all".into(),
            region_overlap_witness(&left.region, &right.region)
                .map(|b| box_summary(&b))
                .unwrap_or_else(|| "<unknown>".into()),
        ),
    };
    // The proxy boundary gets its own code: the fix is a fence, not a
    // synchronization edge, and the two are not interchangeable.
    let (code, message) = if unpublished {
        (
            "proxy_fence_missing",
            "an async-proxy engine reads bytes a lane stored generically, with no \
             `fence.proxy.async` publishing them across the proxy boundary (the fence must be \
             executed by a thread that can already see the stores — its own, or others' after a \
             convergence point or barrier)",
        )
    } else if left.stream_id == right.stream_id {
        (
            "memory_data_race",
            "conflicting lane slices of one warp overlap without a happens-before order (lanes \
             advance independently on sm_70+; converge with warp_sync or a warp-collective \
             before the dependent access)",
        )
    } else {
        (
            "memory_data_race",
            "overlapping shared memory accesses conflict without a happens-before order",
        )
    };
    let mut diagnostic = DiagnosticBuilder::new(code, message)
        .event(right_event)
        .detail("lanes", lanes_detail)
        .detail("left_event_idx", left.event_idx.to_string())
        .detail("right_event_idx", right.event_idx.to_string())
        .detail("left_mode", left.mode.as_str())
        .detail("right_mode", right.mode.as_str())
        .detail("owner", owner_summary(&left.region.owner))
        .detail("overlap", witness)
        .build();
    if let Some(event) = left_event {
        diagnostic
            .details
            .insert("left_stmt_id".into(), event.stmt_id.to_string());
        diagnostic
            .details
            .insert("left_stmt_kind".into(), event.stmt_kind.clone());
    }
    if let Some(event) = right_event {
        diagnostic
            .details
            .insert("right_stmt_id".into(), event.stmt_id.to_string());
        diagnostic
            .details
            .insert("right_stmt_kind".into(), event.stmt_kind.clone());
    }
    cx.diagnostics.push(diagnostic);
}

fn region_overlap_witness(left: &Region, right: &Region) -> Option<BoxN> {
    if left.owner != right.owner {
        return None;
    }
    // Diagnostic-only path (runs once per reported race): materializing the
    // strided form is fine here.
    for l in &left.boxes.to_boxes() {
        for r in &right.boxes.to_boxes() {
            if !boxes_overlap(l, r) {
                continue;
            }
            let ranges = l
                .ranges
                .iter()
                .zip(&r.ranges)
                .map(|(&(ls, le), &(rs, re))| (ls.max(rs), le.min(re)))
                .collect();
            return Some(BoxN::new(ranges));
        }
    }
    None
}

fn owner_summary(owner: &PoolId) -> String {
    match owner {
        PoolId::Smem { cta_id } => format!("smem:cta{cta_id}"),
        PoolId::Tmem { cta_id } => format!("tmem:cta{cta_id}"),
        PoolId::Gmem { tensor_id } => format!("gmem:tensor{tensor_id}"),
        PoolId::Reg { cta_id, tensor_id } => format!("reg:cta{cta_id}:tensor{tensor_id}"),
    }
}

fn cluster_peer_consistency(cx: &mut CheckerCx<'_>) -> CheckResult {
    for event in cx.event_index.events {
        if let TraceEventKind::SyncArrive {
            sync_kind,
            count,
            thread_count,
            ..
        } = &event.payload
        {
            if sync_kind == "cluster" && count > thread_count {
                return Err(cx.fail(
                    "cluster_peer_arrival_overflow",
                    "cluster sync arrival count exceeds expected peer thread count",
                    Some(event),
                ));
            }
        }
    }
    Ok(())
}

fn scheduler_handoff_consistency(cx: &mut CheckerCx<'_>) -> CheckResult {
    let mut seen: HashMap<(u32, i64), usize> = HashMap::new();
    for (idx, event) in cx.event_index.events.iter().enumerate() {
        let TraceEventKind::SchedulerNext {
            scheduler_id,
            task_id,
            ..
        } = &event.payload
        else {
            continue;
        };
        if *task_id < 0 {
            continue;
        }
        let key = (*scheduler_id, *task_id);
        if let Some(prev_idx) = seen.insert(key, idx) {
            let mut diagnostic = DiagnosticBuilder::new(
                "scheduler_handoff_duplicate_task",
                "scheduler handed the same non-negative task id to multiple events",
            )
            .event(Some(event))
            .detail("previous_event", prev_idx.to_string())
            .build();
            diagnostic
                .details
                .insert("task_id".into(), task_id.to_string());
            cx.diagnostics.push(diagnostic);
            return Err(ProtocolStatus::Failed);
        }
    }
    Ok(())
}

fn trace_gap_audit(cx: &mut CheckerCx<'_>) -> CheckResult {
    if cx.gaps.is_empty() {
        return Ok(());
    }
    let first_gap = cx.gaps[0].clone();
    let event = first_gap
        .event_idx
        .and_then(|idx| cx.event_index.events.get(idx));
    let mut diagnostic = DiagnosticBuilder::new(first_gap.code, first_gap.message)
        .event(event)
        .build();
    diagnostic
        .details
        .insert("gap_count".into(), cx.gaps.len().to_string());
    for (key, value) in first_gap.details {
        diagnostic.details.insert(key, value);
    }
    cx.diagnostics.push(diagnostic);
    Err(ProtocolStatus::Inconclusive)
}

struct TraceRegionAudit {
    tensors: HashMap<u32, TensorInfo>,
    smem_size_bytes: usize,
    n_cta_threads: usize,
}

impl TraceRegionAudit {
    fn new(kernel: &Kernel) -> Self {
        let mut tensors = HashMap::new();
        for tensor in &kernel.args {
            record_tensor(&mut tensors, tensor);
        }
        walk_tensors(&kernel.body, &mut tensors);
        Self {
            tensors,
            smem_size_bytes: kernel.smem_size_bytes,
            n_cta_threads: 32 * kernel.num_warps as usize,
        }
    }

    fn validate_region(&self, region: &Region) -> Result<(), RegionError> {
        if region.boxes.is_empty() {
            return Err(RegionError::failed(
                "trace_region_empty",
                "trace region must contain at least one box",
            ));
        }
        let bounds = self.bounds_for_region(region)?;
        let expected_rank = bounds.len();
        if let RegionBoxes::Strided {
            start,
            len,
            stride,
            count,
        } = &region.boxes
        {
            // Uniform runs: validating the first and last run covers them all
            // (rank is 1 by construction; starts ascend; equal lengths).
            if expected_rank != 1 {
                return Err(RegionError::failed(
                    "trace_region_rank_mismatch",
                    "trace region box rank does not match its owner pool",
                ));
            }
            if *len == 0 {
                return Err(RegionError::failed(
                    "trace_region_empty_box",
                    "trace region box has an empty dimension",
                ));
            }
            if start + (count - 1) * stride + len > bounds[0] {
                return Err(RegionError::failed(
                    "trace_region_out_of_bounds",
                    "trace region box is outside its owner pool",
                ));
            }
            return Ok(());
        }
        let RegionBoxes::Boxes(boxes) = &region.boxes else {
            unreachable!("strided handled above");
        };
        for b in boxes {
            if b.ranges.len() != expected_rank {
                return Err(RegionError::failed(
                    "trace_region_rank_mismatch",
                    "trace region box rank does not match its owner pool",
                ));
            }
            for (dim, &(start, end)) in b.ranges.iter().enumerate() {
                if start >= end {
                    return Err(RegionError::failed(
                        "trace_region_empty_box",
                        "trace region box has an empty dimension",
                    ));
                }
                if end > bounds[dim] {
                    return Err(RegionError::failed(
                        "trace_region_out_of_bounds",
                        "trace region box is outside its owner pool",
                    ));
                }
            }
        }
        Ok(())
    }

    fn bounds_for_region(&self, region: &Region) -> Result<Vec<usize>, RegionError> {
        match region.owner {
            PoolId::Smem { .. } => Ok(vec![self.smem_size_bytes]),
            PoolId::Tmem { .. } => Ok(vec![TMEM_ROWS, TMEM_LANE_BYTES]),
            PoolId::Gmem { tensor_id } => {
                self.check_owner_tensor(region, tensor_id)?;
                Ok(vec![self.tensor_extent(tensor_id)?])
            }
            PoolId::Reg { tensor_id, .. } => {
                self.check_owner_tensor(region, tensor_id)?;
                let mut bounds = Vec::with_capacity(self.tensor_shape(tensor_id)?.len() + 1);
                bounds.push(self.n_cta_threads);
                bounds.extend(self.tensor_shape(tensor_id)?.iter().copied());
                Ok(bounds)
            }
        }
    }

    fn check_owner_tensor(&self, region: &Region, owner_tensor_id: u32) -> Result<(), RegionError> {
        if region.tensor_id != owner_tensor_id {
            return Err(RegionError::failed(
                "trace_region_owner_tensor_mismatch",
                "region tensor_id disagrees with its owner pool",
            ));
        }
        Ok(())
    }

    fn tensor_extent(&self, tensor_id: u32) -> Result<usize, RegionError> {
        let Some(info) = self.tensors.get(&tensor_id) else {
            return Err(RegionError::inconclusive(
                "trace_region_unknown_tensor",
                "trace region refers to a tensor absent from Kernel IR",
            ));
        };
        info.byte_extent.ok_or_else(|| {
            RegionError::inconclusive("trace_region_overflow", "tensor byte extent overflowed")
        })
    }

    fn tensor_shape(&self, tensor_id: u32) -> Result<&[usize], RegionError> {
        let Some(info) = self.tensors.get(&tensor_id) else {
            return Err(RegionError::inconclusive(
                "trace_region_unknown_tensor",
                "trace region refers to a tensor absent from Kernel IR",
            ));
        };
        Ok(&info.shape)
    }
}

#[derive(Clone)]
struct TensorInfo {
    shape: Vec<usize>,
    byte_extent: Option<usize>,
}

#[derive(Clone, Copy)]
enum RegionErrorStatus {
    Failed,
    Inconclusive,
}

struct RegionError {
    status: RegionErrorStatus,
    code: &'static str,
    message: &'static str,
}

impl RegionError {
    fn failed(code: &'static str, message: &'static str) -> Self {
        Self {
            status: RegionErrorStatus::Failed,
            code,
            message,
        }
    }

    fn inconclusive(code: &'static str, message: &'static str) -> Self {
        Self {
            status: RegionErrorStatus::Inconclusive,
            code,
            message,
        }
    }

    fn into_diagnostic(self, cx: &mut CheckerCx<'_>, event: &TraceEvent) -> ProtocolStatus {
        match self.status {
            RegionErrorStatus::Failed => cx.fail(self.code, self.message, Some(event)),
            RegionErrorStatus::Inconclusive => {
                cx.inconclusive(self.code, self.message, Some(event))
            }
        }
    }
}

fn record_tensor(tensors: &mut HashMap<u32, TensorInfo>, tensor: &Arc<Tensor>) {
    let byte_extent = numel(&tensor.shape).checked_mul(dtype_size_bytes(tensor.dtype));
    tensors.entry(tensor.id).or_insert_with(|| TensorInfo {
        shape: tensor.shape.clone(),
        byte_extent,
    });
}

fn record_slice(tensors: &mut HashMap<u32, TensorInfo>, slice: &TensorSlice) {
    record_tensor(tensors, &slice.tensor);
}

fn record_reg_operand(tensors: &mut HashMap<u32, TensorInfo>, operand: &crate::ir::RegOperand) {
    if let crate::ir::RegOperand::Slice(slice) = operand {
        record_slice(tensors, slice);
    }
}

fn walk_tensors(body: &[Stmt], tensors: &mut HashMap<u32, TensorInfo>) {
    for stmt in body {
        match stmt {
            Stmt::TensorDef { tensor }
            | Stmt::TmemAlloc { tensor, .. }
            | Stmt::TmemDealloc { tensor, .. } => record_tensor(tensors, tensor),
            Stmt::StoreScalar { dst, .. } => record_slice(tensors, dst),
            Stmt::TmaLoad { dst, src, .. } => {
                record_slice(tensors, dst);
                record_tensor(tensors, src);
            }
            Stmt::TmaStore { dst, src, .. } => {
                record_tensor(tensors, dst);
                record_slice(tensors, src);
            }
            Stmt::CpAsyncBulkS2Cluster { dst, src, .. } => {
                record_slice(tensors, dst);
                record_slice(tensors, src);
            }
            // SYNC ops on the semaphore tensor — no Read/Write regions are ever
            // emitted for it, but record it so a semaphore-only tensor still has
            // audit metadata.
            Stmt::GmemAtomicAdd { sem, .. } | Stmt::GmemWaitEq { sem, .. } => {
                record_slice(tensors, sem);
            }
            Stmt::Tcgen05Mma { dst, a, b, .. } => {
                record_slice(tensors, dst);
                record_slice(tensors, a);
                record_slice(tensors, b);
            }
            Stmt::Tcgen05Ld { dst, src, .. } => {
                record_slice(tensors, dst);
                record_tensor(tensors, src);
            }
            Stmt::Tcgen05St { dst, src, .. } => {
                record_tensor(tensors, dst);
                record_slice(tensors, src);
            }
            Stmt::LdMatrix { dst, src, .. }
            | Stmt::StMatrix { dst, src, .. }
            | Stmt::RegLoad { dst, src }
            | Stmt::RegStore { dst, src } => {
                record_slice(tensors, dst);
                record_slice(tensors, src);
            }
            Stmt::RegFill { dst, value } => {
                record_slice(tensors, dst);
                record_reg_operand(tensors, value);
            }
            Stmt::RegUnary { dst, src, .. } | Stmt::RegReduce { dst, src, .. } => {
                record_slice(tensors, dst);
                record_reg_operand(tensors, src);
            }
            Stmt::RegAdd { dst, lhs, rhs, .. }
            | Stmt::RegSub { dst, lhs, rhs, .. }
            | Stmt::RegMul { dst, lhs, rhs }
            | Stmt::RegMax { dst, lhs, rhs }
            | Stmt::RegMin { dst, lhs, rhs }
            | Stmt::RegBitwise { dst, lhs, rhs, .. } => {
                record_slice(tensors, dst);
                record_reg_operand(tensors, lhs);
                record_reg_operand(tensors, rhs);
            }
            Stmt::RegFma { dst, a, b, c } => {
                record_slice(tensors, dst);
                record_reg_operand(tensors, a);
                record_reg_operand(tensors, b);
                record_reg_operand(tensors, c);
            }
            Stmt::RegCondRescale {
                dst,
                src,
                scale,
                threshold,
                ..
            } => {
                record_slice(tensors, dst);
                record_reg_operand(tensors, src);
                record_reg_operand(tensors, scale);
                record_reg_operand(tensors, threshold);
            }
            Stmt::RegSoftmaxRescale {
                row_max,
                row_scale,
                row_max_old,
                row_max_new,
                scale_log2,
                threshold,
            } => {
                record_slice(tensors, row_max);
                record_slice(tensors, row_scale);
                record_reg_operand(tensors, row_max_old);
                record_reg_operand(tensors, row_max_new);
                record_reg_operand(tensors, scale_log2);
                record_reg_operand(tensors, threshold);
            }
            Stmt::RegCombineIntFracEx2 {
                dst,
                rounded,
                frac_ex2,
            } => {
                record_slice(tensors, dst);
                record_reg_operand(tensors, rounded);
                record_reg_operand(tensors, frac_ex2);
            }
            Stmt::RegCvt { dst, src, .. } => {
                record_slice(tensors, dst);
                record_slice(tensors, src);
            }
            _ => {}
        }
        for child in stmt.child_bodies() {
            walk_tensors(child, tensors);
        }
    }
}

#[derive(Clone, Debug)]
struct AsyncSourceWindow {
    stream_id: usize,
    stmt_id: usize,
    start_idx: usize,
    close_idx: Option<usize>,
    regions: Vec<Region>,
    /// For a DEST window: `Some((op, exact))` if the store is an atomic commutative reduce
    /// (cp.reduce.async.bulk), else `None`. (`start_idx` points at the flush-trigger event, not
    /// the store Write, so the reduce kind is captured here at build time.)
    reduce: Option<(ReduceOp, bool)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MbarKey {
    mbar_id: u32,
    cluster_id: usize,
    ctaid_in_cluster: usize,
    stage: u32,
}

impl MbarKey {
    fn from_target(target: &super::protocol::MbarTargetEvent) -> Self {
        Self {
            mbar_id: target.mbar_id,
            cluster_id: target.cluster_id,
            ctaid_in_cluster: target.ctaid_in_cluster,
            stage: target.stage,
        }
    }
}

/// One cooperative-barrier generation for happens-before purposes: the
/// statement identity plus the runtime rendezvous domain (cluster syncs
/// rendezvous per cluster, cta syncs per CTA, wg_syncs per (CTA, warpgroup),
/// warp syncs per (CTA, warp)) plus that statement's cycle counter.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SyncHbKey {
    stmt_id: usize,
    bar_id: Option<u32>,
    domain: (usize, usize),
    cycle: u64,
}

impl SyncHbKey {
    fn new(
        stmt_id: usize,
        sync_kind: &str,
        bar_id: Option<u32>,
        cycle: u64,
        scope: &super::protocol::AccessScope,
    ) -> Self {
        // A named barrier's identity is (cta_id, barrier_id, cycle), NOT the
        // statement: different NamedBarrier statements with one barrier_id are
        // ONE barrier (that is their point — warpgroup A's stmt and warpgroup
        // B's stmt rendezvous). Drop the stmt id to a sentinel so their
        // arrivals/releases share a generation key; every other sync kind
        // keeps its statement identity.
        if sync_kind == "named" {
            return Self {
                stmt_id: usize::MAX,
                bar_id,
                domain: (scope.cta_id, 0),
                cycle,
            };
        }
        let first_warp = scope.warp_id;
        let domain = match sync_kind {
            "cluster" => (scope.cluster_id, 0),
            "warpgroup" => (scope.cta_id, first_warp / 4),
            "warp" => (scope.cta_id, first_warp),
            _ => (scope.cta_id, 0),
        };
        Self {
            stmt_id,
            bar_id,
            domain,
            cycle,
        }
    }
}

struct MbarCycle {
    expected_arrivals: i64,
    pending_arrivals: i64,
    pending_tx: i64,
    parity: u8,
    pending_release_events: Vec<usize>,
}

impl MbarCycle {
    fn new(expected_arrivals: i64) -> Self {
        Self {
            expected_arrivals,
            pending_arrivals: expected_arrivals,
            pending_tx: 0,
            parity: 0,
            pending_release_events: Vec::new(),
        }
    }

    fn record_release_event(&mut self, idx: usize) {
        self.pending_release_events.push(idx);
    }

    fn drain_release_events(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.pending_release_events)
    }

    fn complete_if_ready(&mut self) -> bool {
        if self.pending_arrivals == 0 && self.pending_tx == 0 {
            self.pending_arrivals = self.expected_arrivals;
            self.parity ^= 1;
            return true;
        }
        false
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SyncKey {
    sync_kind: String,
    thread_count: usize,
    cycle: u64,
    bar_id: Option<u32>,
    resource_scope: String,
}

fn sync_resource_scope(sync_kind: &str, scope: &super::protocol::AccessScope) -> String {
    match sync_kind {
        "warp" => format!("cta:{}:warp:{}", scope.cta_id, scope.warp_id),
        "warpgroup" => format!("cta:{}:wg:{}", scope.cta_id, scope.warp_id / 4),
        "cta" => format!("cta:{}", scope.cta_id),
        "cluster" => format!("cluster:{}", scope.cluster_id),
        // Named barriers rendezvous streams from DIFFERENT statements/warp sets
        // on one (cta, barrier_id); the SyncKey already carries bar_id, so the
        // resource scope is the CTA — never the per-stream fallback (that would
        // split one barrier's arrivals into disjoint keys).
        "named" => format!("cta:{}", scope.cta_id),
        _ => format!("stream:{}", scope.stream_id),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TmemAccessMode {
    Read,
    Write,
}

#[derive(Clone, Debug)]
struct TmemWindow {
    stmt_id: usize,
    async_kind: TmemAsyncKind,
    mode: TmemAccessMode,
    region: Region,
    // cached bounding spans: window scans reject on these in O(1)
    bounds: ((usize, usize), (usize, usize)),
}

/// A live window identical in (stmt, kind, mode, region) constrains nothing
/// new — every conflict with the copy also conflicts with the original. Loop
/// kernels re-issue the same access each iteration, so dedup keeps the open
/// list bounded by the distinct accesses, not the trip count.
fn push_tmem_window_dedup(windows: &mut Vec<TmemWindow>, window: TmemWindow) {
    if !windows.iter().any(|w| {
        w.stmt_id == window.stmt_id
            && w.async_kind == window.async_kind
            && w.mode == window.mode
            && w.region == window.region
    }) {
        windows.push(window);
    }
}

impl TmemWindow {
    fn new(
        event: &TraceEvent,
        async_kind: TmemAsyncKind,
        mode: TmemAccessMode,
        region: &Region,
    ) -> Self {
        Self {
            stmt_id: event.stmt_id,
            async_kind,
            mode,
            bounds: super::region::region_bounding_spans(region),
            region: region.clone(),
        }
    }

    fn is_conflicting_access(&self, other: &TmemWindow) -> bool {
        self.mode == TmemAccessMode::Write || other.mode == TmemAccessMode::Write
    }

    fn overlaps(&self, other: &TmemWindow) -> bool {
        super::region::bounding_spans_touch(self.bounds, other.bounds)
            && regions_overlap(&self.region, &other.region)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct MbarWaitKey {
    target: MbarKey,
    phase: u8,
}

#[derive(Clone, Debug)]
struct BlockingNode {
    event_idx: usize,
    resource_key: String,
    release_events: Vec<usize>,
}

#[derive(Clone, Debug)]
struct BlockingGraph {
    nodes: Vec<BlockingNode>,
    edges: Vec<Vec<usize>>,
}

#[derive(Debug)]
struct DeadlockGraphError {
    status: ProtocolStatus,
    code: &'static str,
    message: &'static str,
    event_idx: usize,
    details: BTreeMap<String, String>,
}

impl DeadlockGraphError {
    fn failed(
        code: &'static str,
        message: &'static str,
        event_idx: usize,
        resource_key: String,
    ) -> Self {
        let mut details = BTreeMap::new();
        details.insert("resource".into(), resource_key);
        Self {
            status: ProtocolStatus::Failed,
            code,
            message,
            event_idx,
            details,
        }
    }
}

impl BlockingGraph {
    fn build(events: &[TraceEvent]) -> Result<Self, DeadlockGraphError> {
        let mut nodes = Vec::new();
        collect_mbar_blockers(events, &mut nodes)?;
        collect_sync_blockers(events, &mut nodes)?;
        collect_wait_group_blockers(events, &mut nodes)?;
        collect_tmem_wait_blockers(events, &mut nodes)?;
        // Edges feed only `find_cycle`, which is disabled (see DETECT_WAIT_FOR_CYCLES).
        let edges = if DETECT_WAIT_FOR_CYCLES {
            build_blocking_edges(events, &nodes)
        } else {
            Vec::new()
        };
        Ok(Self { nodes, edges })
    }

    /// Iterative Tarjan SCC (explicit DFS stack instead of recursion) so that
    /// deep wait-for chains at large task counts cannot overflow the call stack.
    /// Returns the first strongly connected component that is a cycle (more than
    /// one node, or a single node with a self-loop). The visit order, lowlink
    /// math, and first-cycle result match the recursive formulation exactly.
    fn find_cycle(&self) -> Option<Vec<usize>> {
        let mut state = vec![TarjanNode::default(); self.nodes.len()];
        let mut scc_stack: Vec<usize> = Vec::new();
        let mut index = 0usize;
        // Explicit DFS frames: (node, index of the next successor to examine).
        let mut work: Vec<(usize, usize)> = Vec::new();

        for start in 0..self.nodes.len() {
            if state[start].index.is_some() {
                continue;
            }
            work.push((start, 0));
            while let Some(&(node, edge_pos)) = work.last() {
                if edge_pos == 0 {
                    // First visit to `node`.
                    state[node].index = Some(index);
                    state[node].lowlink = index;
                    index += 1;
                    scc_stack.push(node);
                    state[node].on_stack = true;
                }
                // Walk successors from where we left off, descending into the
                // first unvisited one (the iterative stand-in for recursion).
                let mut descended = false;
                let mut pos = edge_pos;
                while pos < self.edges[node].len() {
                    let next = self.edges[node][pos];
                    pos += 1;
                    if state[next].index.is_none() {
                        work.last_mut().expect("work frame present").1 = pos;
                        work.push((next, 0));
                        descended = true;
                        break;
                    } else if state[next].on_stack {
                        let next_index =
                            state[next].index.expect("on-stack node must have an index");
                        state[node].lowlink = state[node].lowlink.min(next_index);
                    }
                }
                if descended {
                    continue;
                }
                // All successors processed: `node` roots an SCC iff lowlink == index.
                if state[node].lowlink == state[node].index.expect("node index is set") {
                    let mut component = Vec::new();
                    while let Some(member) = scc_stack.pop() {
                        state[member].on_stack = false;
                        component.push(member);
                        if member == node {
                            break;
                        }
                    }
                    if component.len() > 1
                        || component
                            .first()
                            .is_some_and(|&member| self.edges[member].contains(&member))
                    {
                        return Some(component);
                    }
                }
                // Pop `node`; propagate its lowlink to the parent frame — the
                // iterative equivalent of `parent.lowlink = min(.., child.lowlink)`
                // after a recursive call returns.
                work.pop();
                if let Some(&(parent, _)) = work.last() {
                    state[parent].lowlink = state[parent].lowlink.min(state[node].lowlink);
                }
            }
        }
        None
    }
}

#[derive(Clone, Debug, Default)]
struct TarjanNode {
    index: Option<usize>,
    lowlink: usize,
    on_stack: bool,
}

fn build_blocking_edges(events: &[TraceEvent], nodes: &[BlockingNode]) -> Vec<Vec<usize>> {
    let mut node_for_event = HashMap::new();
    for (node_idx, node) in nodes.iter().enumerate() {
        node_for_event.insert(node.event_idx, node_idx);
    }
    let mut edges = vec![Vec::new(); nodes.len()];
    let mut last_blocker_by_stream: HashMap<usize, usize> = HashMap::new();
    let mut blocker_before_event: HashMap<usize, usize> = HashMap::new();
    for (idx, event) in events.iter().enumerate() {
        if let Some(scope) = event_scope(&event.payload) {
            if let Some(&prior) = last_blocker_by_stream.get(&scope.stream_id) {
                blocker_before_event.insert(idx, prior);
            }
            if let Some(&node_idx) = node_for_event.get(&idx) {
                if let Some(prev) = last_blocker_by_stream.insert(scope.stream_id, node_idx) {
                    add_blocking_edge(&mut edges, node_idx, prev);
                }
            }
        }
    }

    for (node_idx, node) in nodes.iter().enumerate() {
        for &release_idx in &node.release_events {
            if let Some(&prior) = blocker_before_event.get(&release_idx) {
                add_blocking_edge(&mut edges, node_idx, prior);
            }
        }
    }
    edges
}

fn add_blocking_edge(edges: &mut [Vec<usize>], from: usize, to: usize) {
    if !edges[from].contains(&to) {
        edges[from].push(to);
    }
}

fn collect_mbar_blockers(
    events: &[TraceEvent],
    nodes: &mut Vec<BlockingNode>,
) -> Result<(), DeadlockGraphError> {
    let mut mbars: HashMap<MbarKey, MbarCycle> = HashMap::new();
    let mut last_release: HashMap<MbarWaitKey, Vec<usize>> = HashMap::new();
    let mut pending_waits: HashMap<MbarWaitKey, Vec<usize>> = HashMap::new();
    let mut release_for_wait: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut wait_resources: HashMap<usize, String> = HashMap::new();

    for (idx, event) in events.iter().enumerate() {
        match &event.payload {
            TraceEventKind::MbarInit { target, count, .. } => {
                mbars.insert(MbarKey::from_target(target), MbarCycle::new(*count));
            }
            TraceEventKind::MbarExpectTx { target, bytes, .. } => {
                if let Some(cycle) = mbars.get_mut(&MbarKey::from_target(target)) {
                    cycle.pending_tx += *bytes;
                }
            }
            TraceEventKind::MbarCompleteTx { target, bytes, .. } => {
                let key = MbarKey::from_target(target);
                if let Some(cycle) = mbars.get_mut(&key) {
                    cycle.pending_tx -= *bytes;
                    cycle.record_release_event(idx);
                    if cycle.complete_if_ready() {
                        let wait_key = MbarWaitKey {
                            target: key,
                            phase: cycle.parity ^ 1,
                        };
                        let releases = cycle.drain_release_events();
                        if let Some(waiters) = pending_waits.remove(&wait_key) {
                            for wait_idx in waiters {
                                release_for_wait
                                    .entry(wait_idx)
                                    .or_insert_with(|| releases.clone());
                            }
                        }
                        last_release.insert(wait_key, releases);
                    }
                }
            }
            TraceEventKind::MbarArrive { target, count, .. } => {
                let key = MbarKey::from_target(target);
                if let Some(cycle) = mbars.get_mut(&key) {
                    cycle.pending_arrivals -= *count;
                    cycle.record_release_event(idx);
                    if cycle.complete_if_ready() {
                        let wait_key = MbarWaitKey {
                            target: key,
                            phase: cycle.parity ^ 1,
                        };
                        let releases = cycle.drain_release_events();
                        if let Some(waiters) = pending_waits.remove(&wait_key) {
                            for wait_idx in waiters {
                                release_for_wait
                                    .entry(wait_idx)
                                    .or_insert_with(|| releases.clone());
                            }
                        }
                        last_release.insert(wait_key, releases);
                    }
                }
            }
            TraceEventKind::MbarWait { target, phase, .. } => {
                let key = MbarKey::from_target(target);
                let Some(cycle) = mbars.get(&key) else {
                    return Err(DeadlockGraphError::failed(
                        "deadlock_freedom_missing_release_witness",
                        "blocking mbarrier wait has no initialized cell",
                        idx,
                        mbar_wait_resource_key(&key, *phase),
                    ));
                };
                let wait_key = MbarWaitKey {
                    target: key,
                    phase: *phase,
                };
                let resource = mbar_wait_resource_key(&key, *phase);
                wait_resources.insert(idx, resource);
                if let Some(releases) = last_release.get(&wait_key) {
                    release_for_wait.insert(idx, releases.clone());
                } else if *phase == cycle.parity {
                    pending_waits.entry(wait_key).or_default().push(idx);
                }
            }
            _ => {}
        }
    }

    for (wait_key, waiters) in pending_waits {
        for wait_idx in waiters {
            return Err(DeadlockGraphError::failed(
                "deadlock_freedom_missing_release_witness",
                "blocking mbarrier wait has no release witness",
                wait_idx,
                mbar_wait_resource_key(&wait_key.target, wait_key.phase),
            ));
        }
    }

    for (wait_idx, release_events) in release_for_wait {
        if events
            .get(wait_idx)
            .and_then(|event| event_scope(&event.payload))
            .is_some()
        {
            nodes.push(BlockingNode {
                event_idx: wait_idx,
                resource_key: wait_resources
                    .remove(&wait_idx)
                    .unwrap_or_else(|| "mbar:unknown".into()),
                release_events,
            });
        }
    }
    Ok(())
}

fn collect_sync_blockers(
    events: &[TraceEvent],
    nodes: &mut Vec<BlockingNode>,
) -> Result<(), DeadlockGraphError> {
    let mut arrivals: HashMap<SyncKey, Vec<(usize, usize)>> = HashMap::new();
    for (idx, event) in events.iter().enumerate() {
        if let TraceEventKind::SyncArrive {
            sync_kind,
            count,
            thread_count,
            cycle,
            bar_id,
            scope,
        } = &event.payload
        {
            let key = SyncKey {
                sync_kind: sync_kind.clone(),
                thread_count: *thread_count,
                cycle: *cycle,
                bar_id: *bar_id,
                resource_scope: sync_resource_scope(sync_kind, scope),
            };
            arrivals.entry(key).or_default().push((idx, *count));
        }
    }

    for (idx, event) in events.iter().enumerate() {
        let TraceEventKind::Sync {
            sync_kind,
            thread_count,
            cycle,
            bar_id,
            scope,
        } = &event.payload
        else {
            continue;
        };
        let key = SyncKey {
            sync_kind: sync_kind.clone(),
            thread_count: *thread_count,
            cycle: *cycle,
            bar_id: *bar_id,
            resource_scope: sync_resource_scope(sync_kind, scope),
        };
        let Some(release_arrivals) = arrivals.get(&key) else {
            return Err(DeadlockGraphError::failed(
                "deadlock_freedom_missing_release_witness",
                "sync completion has no arrival release witness",
                idx,
                sync_resource_key(&key),
            ));
        };
        let max_count = release_arrivals
            .iter()
            .map(|(_, count)| *count)
            .max()
            .unwrap_or(0);
        if max_count < *thread_count {
            return Err(DeadlockGraphError::failed(
                "deadlock_freedom_missing_release_witness",
                "sync completion does not have enough arrivals to close the cycle",
                idx,
                sync_resource_key(&key),
            ));
        }
        nodes.push(BlockingNode {
            event_idx: idx,
            resource_key: sync_resource_key(&key),
            release_events: release_arrivals
                .iter()
                .map(|(event_idx, _)| *event_idx)
                .collect(),
        });
    }
    Ok(())
}

fn collect_wait_group_blockers(
    events: &[TraceEvent],
    nodes: &mut Vec<BlockingNode>,
) -> Result<(), DeadlockGraphError> {
    let mut open_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, event) in events.iter().enumerate() {
        match &event.payload {
            TraceEventKind::CommitGroup { scope } => {
                open_groups.entry(scope.stream_id).or_default().push(idx);
            }
            TraceEventKind::WaitGroup { n, scope } => {
                let groups = open_groups.entry(scope.stream_id).or_default();
                if groups.is_empty() {
                    return Err(DeadlockGraphError::failed(
                        "deadlock_freedom_missing_release_witness",
                        "wait_group has no committed async group release witness",
                        idx,
                        format!("wait_group:stream{}", scope.stream_id),
                    ));
                }
                let retain = *n as usize;
                let release_count = groups.len().saturating_sub(retain);
                if release_count > 0 {
                    let releases: Vec<_> = groups.drain(..release_count).collect();
                    nodes.push(BlockingNode {
                        event_idx: idx,
                        resource_key: format!("wait_group:stream{}", scope.stream_id),
                        release_events: releases,
                    });
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn collect_tmem_wait_blockers(
    events: &[TraceEvent],
    nodes: &mut Vec<BlockingNode>,
) -> Result<(), DeadlockGraphError> {
    let mut open_windows: HashMap<(usize, u8), Vec<usize>> = HashMap::new();
    for (idx, event) in events.iter().enumerate() {
        match &event.payload {
            TraceEventKind::Read {
                access_kind: MemoryAccessKind::Tmem(async_kind),
                scope,
                ..
            }
            | TraceEventKind::Write {
                access_kind: MemoryAccessKind::Tmem(async_kind),
                scope,
                ..
            } if *async_kind == TmemAsyncKind::Ld || *async_kind == TmemAsyncKind::St => {
                open_windows
                    .entry((scope.stream_id, tmem_async_key(*async_kind)))
                    .or_default()
                    .push(idx);
            }
            TraceEventKind::TmemWait { async_kind, scope } => {
                let key = (scope.stream_id, tmem_async_key(*async_kind));
                let releases = open_windows.remove(&key).unwrap_or_default();
                if releases.is_empty() {
                    return Err(DeadlockGraphError::failed(
                        "deadlock_freedom_missing_release_witness",
                        "TMEM wait has no matching ld/st window release witness",
                        idx,
                        format!("tmem_wait:{async_kind:?}:stream{}", scope.stream_id),
                    ));
                }
                nodes.push(BlockingNode {
                    event_idx: idx,
                    resource_key: format!("tmem_wait:{async_kind:?}:stream{}", scope.stream_id),
                    release_events: releases,
                });
            }
            _ => {}
        }
    }
    Ok(())
}

fn tmem_async_key(async_kind: TmemAsyncKind) -> u8 {
    match async_kind {
        TmemAsyncKind::Ld => 0,
        TmemAsyncKind::St => 1,
        TmemAsyncKind::Mma => 2,
        TmemAsyncKind::Cp => 3,
    }
}

fn mbar_wait_resource_key(target: &MbarKey, phase: u8) -> String {
    format!(
        "mbar:{}:cluster{}:cta{}:stage{}:phase{}",
        target.mbar_id, target.cluster_id, target.ctaid_in_cluster, target.stage, phase
    )
}

fn sync_resource_key(key: &SyncKey) -> String {
    format!(
        "sync:{}:{}:cycle{}:bar{:?}",
        key.sync_kind, key.resource_scope, key.cycle, key.bar_id
    )
}

fn box_summary(b: &BoxN) -> String {
    b.ranges
        .iter()
        .map(|(start, end)| format!("[{start},{end})"))
        .collect::<Vec<_>>()
        .join("x")
}

fn event_kind_name(payload: &TraceEventKind) -> &'static str {
    match payload {
        TraceEventKind::Read { .. } => "read",
        TraceEventKind::Write { .. } => "write",
        TraceEventKind::TmemWait { .. } => "tmem_wait",
        TraceEventKind::Fence { .. } => "fence",
        TraceEventKind::CommitGroup { .. } => "commit_group",
        TraceEventKind::WaitGroup { .. } => "wait_group",
        TraceEventKind::MbarInit { .. } => "mbar_init",
        TraceEventKind::MbarArrive { .. } => "mbar_arrive",
        TraceEventKind::MbarExpectTx { .. } => "mbar_expect_tx",
        TraceEventKind::MbarCompleteTx { .. } => "mbar_complete_tx",
        TraceEventKind::MbarWait { .. } => "mbar_wait",
        TraceEventKind::SyncArrive { .. } => "sync_arrive",
        TraceEventKind::Sync { .. } => "sync",
        TraceEventKind::SemRelease { .. } => "sem_release",
        TraceEventKind::SemAcquire { .. } => "sem_acquire",
        TraceEventKind::TmemAlloc { .. } => "tmem_alloc",
        TraceEventKind::TmemDealloc { .. } => "tmem_dealloc",
        TraceEventKind::SchedulerNext { .. } => "scheduler_next",
    }
}

fn event_scope(payload: &TraceEventKind) -> Option<&super::protocol::AccessScope> {
    match payload {
        TraceEventKind::Read { scope, .. }
        | TraceEventKind::Write { scope, .. }
        | TraceEventKind::TmemWait { scope, .. }
        | TraceEventKind::Fence { scope, .. }
        | TraceEventKind::CommitGroup { scope }
        | TraceEventKind::WaitGroup { scope, .. }
        | TraceEventKind::MbarInit { scope, .. }
        | TraceEventKind::MbarArrive { scope, .. }
        | TraceEventKind::MbarExpectTx { scope, .. }
        | TraceEventKind::MbarCompleteTx { scope, .. }
        | TraceEventKind::MbarWait { scope, .. }
        | TraceEventKind::SyncArrive { scope, .. }
        | TraceEventKind::Sync { scope, .. }
        | TraceEventKind::SemRelease { scope, .. }
        | TraceEventKind::SemAcquire { scope, .. }
        | TraceEventKind::TmemAlloc { scope, .. }
        | TraceEventKind::TmemDealloc { scope, .. }
        | TraceEventKind::SchedulerNext { scope, .. } => Some(scope),
    }
}

type Clock = Vec<(usize, usize)>;

// Happens-before is per LANE, not per warp. On Volta+ (Independent Thread
// Scheduling) the 32 lanes of a warp advance through their shared instruction
// stream independently: a store by one lane is unordered against another
// lane's later instructions until a warp-level convergence point joins them.
// A release publishes only the arriving lanes' progress; what a lane never
// published is, by construction, invisible to every consumer.
//
// All of a stream's lanes share ONE number line — the stream's event ordinal
// (1-based trace position within the stream). A clock entry means:
//
//   prefix_dim(S) ↦ v      every event of stream S with ordinal <= v, by ANY
//                          lane, happens-before me (the all-lane prefix: the
//                          stream's last convergence, or a full-warp arrive);
//   lane_dim(S, l) ↦ v     lane l's events with ordinal <= v happen-before me
//                          (the refinement a masked release carries).
//
// So a full-warp publication is ONE prefix entry, an elected arrive is ONE
// lane entry, and clocks stay O(streams) with a short lane tail — per-lane
// precision costs only where divergent publication actually happens.
//
// The lanes are not the only things that touch memory. The ASYNC PROXY
// engines (TMA / tensormap / tcgen05) read and write on their own, and a
// lane's ordinary stores are invisible to them until some thread executes
// `fence.proxy.async`. That fence is a RELEASE of the fencing thread's view
// into the engines (`EngineViews`), and every engine access ACQUIRES it, so
// cross-proxy visibility is an ordinary comparison against the same clocks —
// what no thread fenced is simply not in the engines' view.
const WARP_LANES: usize = 32;
const STREAM_DIMS: usize = WARP_LANES + 1;
const FULL_MASK: u32 = u32::MAX;

fn lane_dim(stream_id: usize, lane: usize) -> usize {
    stream_id * STREAM_DIMS + lane
}

fn prefix_dim(stream_id: usize) -> usize {
    stream_id * STREAM_DIMS + WARP_LANES
}

/// What the async-proxy engines have absorbed, by the address scope a
/// `fence.proxy.async` publishes into. The engines are CTA-level hardware —
/// a fence by ANY thread makes that thread's view visible to every engine
/// access of the same CTA — so the view is keyed by scope, not by stream.
#[derive(Default)]
struct EngineViews {
    cta: HashMap<usize, Clock>,
    cluster: HashMap<usize, Clock>,
    gpu: Clock,
    /// Bumped on every fence; lets an engine access reuse the previous
    /// snapshot when nothing has been published since.
    epoch: u64,
    cache: HashMap<usize, (u64, Arc<Clock>)>,
}

impl EngineViews {
    /// Publish a fencing thread's view at the fence's address scope.
    fn publish(&mut self, fence_scope: FenceScope, cta_id: usize, cluster_id: usize, view: &Clock) {
        let target = match fence_scope {
            FenceScope::Cta => self.cta.entry(cta_id).or_default(),
            FenceScope::Cluster => self.cluster.entry(cluster_id).or_default(),
            FenceScope::Gpu => &mut self.gpu,
        };
        join_clock(target, view);
        self.epoch += 1;
        self.cache.clear();
    }

    /// What an engine access in this CTA can observe: every scope that
    /// covers it.
    fn view_for(&mut self, cta_id: usize, cluster_id: usize) -> Arc<Clock> {
        if let Some((epoch, clock)) = self.cache.get(&cta_id) {
            if *epoch == self.epoch {
                return Arc::clone(clock);
            }
        }
        let mut clock = self.gpu.clone();
        if let Some(c) = self.cluster.get(&cluster_id) {
            join_clock(&mut clock, c);
        }
        if let Some(c) = self.cta.get(&cta_id) {
            join_clock(&mut clock, c);
        }
        let clock = Arc::new(clock);
        self.cache.insert(cta_id, (self.epoch, Arc::clone(&clock)));
        clock
    }
}

/// If this event's memory access runs on the async proxy (an engine touching
/// the bytes) rather than on the lanes themselves, the CTA whose shared
/// address space holds those bytes.
fn async_proxy_access_owner(event: &TraceEvent) -> Option<usize> {
    let region = match &event.payload {
        TraceEventKind::Read {
            region,
            proxy: MemoryProxy::Async,
            ..
        }
        | TraceEventKind::Write {
            region,
            proxy: MemoryProxy::Async,
            ..
        } => region,
        _ => return None,
    };
    match region.owner {
        PoolId::Smem { cta_id } => Some(cta_id),
        _ => None,
    }
}

/// A warp-collective EVENT: the collective instruction set by statement kind,
/// plus the TMEM allocator lifecycle events themselves — tcgen05.alloc /
/// dealloc are `.sync.aligned` (PTX §9.7.17) whatever statement emitted them.
fn is_warp_collective_event(event: &TraceEvent) -> bool {
    is_warp_collective_stmt(&event.stmt_kind)
        || matches!(
            &event.payload,
            TraceEventKind::TmemAlloc { .. } | TraceEventKind::TmemDealloc { .. }
        )
}

/// A full-warp convergence: every lane of the warp synchronizes at this
/// event, so the lanes' clocks join. Warp-collective events converge by
/// definition; a cooperative barrier arrival/passage converges when the whole
/// warp participates (a partial-mask rendezvous synchronizes only the masked
/// lanes and joins nothing for the rest).
fn is_convergence_event(event: &TraceEvent) -> bool {
    if is_warp_collective_event(event) {
        return true;
    }
    matches!(
        &event.payload,
        TraceEventKind::Sync { scope, .. } | TraceEventKind::SyncArrive { scope, .. }
            if scope.active_lanes == FULL_MASK
    )
}

/// A stream's live clock state during the trace scan.
///
/// Lane `l`'s effective clock is `shared ⊔ {lane_dim(l) ↦ this event's
/// ordinal} ⊔ extra[l]`. `shared` is the lower bound common to all 32 lanes —
/// its own-stream prefix entry advances at convergence points, and full-warp
/// acquires join into it directly — so consecutive events share one snapshot.
/// `extra[l]` holds clocks a masked acquire delivered to lane `l` alone; it
/// drains at the next convergence.
struct StreamClockState {
    shared: Arc<Clock>,
    extra: HashMap<u8, Arc<Clock>>,
    /// The NEXT event's 1-based ordinal on this stream's number line.
    next_ordinal: usize,
}

impl StreamClockState {
    fn new() -> Self {
        Self {
            shared: Arc::new(Clock::new()),
            extra: HashMap::new(),
            next_ordinal: 1,
        }
    }

    /// Raise the all-lane prefix to ordinal `upto` and fold the lane
    /// overlays: every lane now observes everything any lane did through
    /// `upto`.
    fn converge(&mut self, stream_id: usize, upto: usize) {
        let shared = Arc::make_mut(&mut self.shared);
        clock_set_max(shared, prefix_dim(stream_id), upto);
        for (_, clk) in self.extra.drain() {
            join_clock(shared, &clk);
        }
    }

    /// Join a release clock into exactly the acquiring lanes. A full-warp
    /// acquire raises the shared lower bound directly; a masked acquire
    /// reaches only the masked lanes' private overlays.
    fn acquire(&mut self, mask: u32, release: &Clock) {
        if release.is_empty() {
            return;
        }
        if mask == FULL_MASK {
            join_clock(Arc::make_mut(&mut self.shared), release);
            return;
        }
        for lane in 0..WARP_LANES as u8 {
            if mask & (1u32 << lane) != 0 {
                let entry = self
                    .extra
                    .entry(lane)
                    .or_insert_with(|| Arc::new(Clock::new()));
                join_clock(Arc::make_mut(entry), release);
            }
        }
    }

    /// The clock a release at ordinal `ordinal` with executing-lane set
    /// `mask` publishes: the join of exactly those lanes' effective clocks.
    /// Every arriving lane's whole history rides its own entry (its events
    /// all sit at ordinals <= `ordinal`); a full warp collapses to one
    /// prefix entry. Lanes outside the mask contribute nothing — their
    /// unconverged writes stay unpublished.
    fn published(&self, stream_id: usize, mask: u32, ordinal: usize) -> Clock {
        let mut clock = (*self.shared).clone();
        if mask == FULL_MASK {
            clock_set_max(&mut clock, prefix_dim(stream_id), ordinal);
            for (_, extra) in self.extra.iter() {
                join_clock(&mut clock, extra);
            }
            return clock;
        }
        for lane in 0..WARP_LANES {
            if mask & (1 << lane) == 0 {
                continue;
            }
            clock_set_max(&mut clock, lane_dim(stream_id, lane), ordinal);
            if let Some(extra) = self.extra.get(&(lane as u8)) {
                join_clock(&mut clock, extra);
            }
        }
        clock
    }
}

/// Per-event clock identity: which stream and lanes executed it at which
/// ordinal, the stream's shared-clock snapshot, and any live lane overlays.
#[derive(Clone)]
struct EventHb {
    stream_id: usize,
    /// Executing-lane bitmask; 0 = the event carries no scope and no HB
    /// identity.
    mask: u32,
    /// This event's 1-based position on its stream's number line.
    ordinal: usize,
    shared: Arc<Clock>,
    /// Lane overlays live at this event, for its executing lanes only.
    extras: Vec<(u8, Arc<Clock>)>,
    /// For an async-proxy access: what the stream's engines had absorbed
    /// when it was issued. `None` for everything the lanes do themselves.
    engine: Option<Arc<Clock>>,
}

impl EventHb {
    fn none() -> Self {
        Self {
            stream_id: 0,
            mask: 0,
            ordinal: 0,
            shared: Arc::new(Clock::new()),
            extras: Vec::new(),
            engine: None,
        }
    }
}

struct OrderingAnalysis {
    meta: Vec<EventHb>,
}

impl OrderingAnalysis {
    fn new(events: &[TraceEvent]) -> Self {
        let stream_count = events
            .iter()
            .filter_map(|event| event_scope(&event.payload).map(|scope| scope.stream_id))
            .max()
            .map(|max_stream| max_stream + 1)
            .unwrap_or(0);
        let mut meta: Vec<EventHb> = Vec::with_capacity(events.len());
        let no_scope = EventHb::none();
        let mut states: Vec<StreamClockState> =
            (0..stream_count).map(|_| StreamClockState::new()).collect();
        // What the async-proxy engines can see (see `EngineViews`).
        let mut engines = EngineViews::default();
        let mut mbars: HashMap<MbarKey, MbarCycle> = HashMap::new();
        // Release accumulation: each arriving event joins its published
        // (lane-projected) clock as it lands; completion freezes the join as
        // the phase's release for its waiters.
        let mut mbar_release_acc: HashMap<MbarKey, Clock> = HashMap::new();
        let mut last_release_clocks: HashMap<MbarKey, Clock> = HashMap::new();
        // Cooperative barrier (cta_sync / wg_sync / warp_sync / cluster_sync)
        // happens-before. Per-warp streams rendezvous in shared state and each
        // emits SyncArrive at arrival and Sync at passage. A barrier means
        // every arrival happens-before every passage: arrivals of a
        // generation join into `sync_arrivals`; the completing arrival
        // (count == thread_count) freezes that join as the generation's
        // release clock; every Sync passage acquires it.
        let mut sync_arrivals: HashMap<SyncHbKey, Clock> = HashMap::new();
        let mut sync_releases: HashMap<SyncHbKey, Clock> = HashMap::new();
        // GMEM-semaphore value-keyed release/acquire (gmem_atomic_add / gmem_wait_eq).
        // VALUE-KEYED, not cell-keyed: each `atomic_add(order=release)` publishes this
        // event's clock under `(SemKey, post_increment_value)`; each `wait_eq(value)`
        // joins the release published for THAT SPECIFIC value — i.e. the release of the
        // atomic_add that PRODUCED the observed counter, exactly as the mbar phase-keyed
        // lookup does with the counter-value in place of parity. This closes the
        // cross-stream HB(producer-of-value.add, waiter.post-wait): stream t's reduce-add +
        // drain happens-before stream t+1's acquire happens-before stream t+1's reduce-add.
        let mut sem_releases: HashMap<(SemKey, i64), Clock> = HashMap::new();
        for event in events.iter() {
            let Some(scope) = event_scope(&event.payload) else {
                meta.push(no_scope.clone());
                continue;
            };
            let stream_id = scope.stream_id;
            let mask = scope.active_lanes;
            let ordinal = states[stream_id].next_ordinal;
            states[stream_id].next_ordinal += 1;

            // Convergence joins the lanes BEFORE the event runs: everything
            // any lane did earlier is warp-visible from here on.
            if is_convergence_event(event) {
                states[stream_id].converge(stream_id, ordinal - 1);
            }

            match &event.payload {
                TraceEventKind::MbarWait { target, phase, .. } => {
                    let key = MbarKey::from_target(target);
                    let released = mbars.get(&key).is_none_or(|cycle| *phase != cycle.parity);
                    if released {
                        if let Some(release_clock) = last_release_clocks.get(&key) {
                            states[stream_id].acquire(mask, release_clock);
                        }
                    }
                }
                TraceEventKind::Sync {
                    sync_kind,
                    cycle,
                    bar_id,
                    ..
                } => {
                    let key = SyncHbKey::new(event.stmt_id, sync_kind, *bar_id, *cycle, scope);
                    if let Some(release_clock) = sync_releases.get(&key) {
                        states[stream_id].acquire(mask, release_clock);
                    }
                }
                // semaphore acquire (gmem_wait_eq): join the release published for the
                // SPECIFIC observed value (value-keyed) — like MbarWait absorbing the
                // producer's release clock. Done BEFORE the advance so the post-wait
                // clock dominates the producer's add (and its drained reduce-add).
                TraceEventKind::SemAcquire { key, value, .. } => {
                    if let Some(release_clock) = sem_releases.get(&(*key, *value)) {
                        states[stream_id].acquire(mask, release_clock);
                    }
                }
                _ => {}
            }

            // Record the event's clock identity.
            let extras: Vec<(u8, Arc<Clock>)> = if states[stream_id].extra.is_empty() {
                Vec::new()
            } else {
                let mut extras: Vec<(u8, Arc<Clock>)> = states[stream_id]
                    .extra
                    .iter()
                    .filter(|(lane, _)| mask & (1u32 << **lane) != 0)
                    .map(|(lane, clk)| (*lane, Arc::clone(clk)))
                    .collect();
                extras.sort_by_key(|(lane, _)| *lane);
                extras
            };
            // An engine access sees what the stream's async-proxy engines
            // absorbed, NOT what the issuing lane happens to have in its own
            // view: the bytes are touched by the engine, so only fenced
            // stores are visible to it.
            // Keyed by the CTA that OWNS the bytes: `fence.proxy.async`
            // publishes the fencing thread's accesses to a shared address
            // space, so an engine reading a peer CTA's SMEM must consult the
            // view published for THAT address space, not for its own CTA.
            let engine = async_proxy_access_owner(event)
                .map(|owner_cta| engines.view_for(owner_cta, scope.cluster_id));
            meta.push(EventHb {
                stream_id,
                mask,
                ordinal,
                shared: Arc::clone(&states[stream_id].shared),
                extras,
                engine,
            });

            // A warp-collective instruction's effects are warp-visible once
            // it completes: raise the prefix over the event itself.
            if is_warp_collective_event(event) {
                states[stream_id].converge(stream_id, ordinal);
            }
            // A proxy fence publishes the fencing lanes' view (including this
            // fence's own position) to the engines, at the fence's address
            // scope. A lane publishes what IT can see, so a lane fencing its
            // own prior stores suffices for those; other lanes' stores ride in
            // only if a convergence point or barrier already carried them into
            // a fencing lane's view.
            if let TraceEventKind::Fence {
                fence_kind: FenceEventKind::ProxyAsync,
                fence_scope,
                ..
            } = &event.payload
            {
                let published = states[stream_id].published(stream_id, mask, ordinal);
                engines.publish(*fence_scope, scope.cta_id, scope.cluster_id, &published);
            }

            match &event.payload {
                TraceEventKind::MbarInit { target, count, .. } => {
                    mbars.insert(MbarKey::from_target(target), MbarCycle::new(*count));
                }
                TraceEventKind::MbarExpectTx { target, bytes, .. } => {
                    if let Some(cycle) = mbars.get_mut(&MbarKey::from_target(target)) {
                        cycle.pending_tx += *bytes;
                    }
                }
                TraceEventKind::MbarCompleteTx { target, bytes, .. } => {
                    let key = MbarKey::from_target(target);
                    if let Some(cycle) = mbars.get_mut(&key) {
                        cycle.pending_tx -= *bytes;
                        let published = states[stream_id].published(stream_id, mask, ordinal);
                        let acc = mbar_release_acc.entry(key).or_default();
                        join_clock(acc, &published);
                        if cycle.complete_if_ready() {
                            let acc = mbar_release_acc.remove(&key).unwrap_or_default();
                            last_release_clocks.insert(key, acc);
                        }
                    }
                }
                TraceEventKind::MbarArrive { target, count, .. } => {
                    let key = MbarKey::from_target(target);
                    if let Some(cycle) = mbars.get_mut(&key) {
                        cycle.pending_arrivals -= *count;
                        let published = states[stream_id].published(stream_id, mask, ordinal);
                        let acc = mbar_release_acc.entry(key).or_default();
                        join_clock(acc, &published);
                        if cycle.complete_if_ready() {
                            let acc = mbar_release_acc.remove(&key).unwrap_or_default();
                            last_release_clocks.insert(key, acc);
                        }
                    }
                }
                TraceEventKind::SyncArrive {
                    sync_kind,
                    thread_count,
                    count,
                    cycle,
                    bar_id,
                    ..
                } => {
                    // Blocked streams re-poll and re-emit their arrival every
                    // round (including once more at passage, still with
                    // count == thread_count) — so the release ACCUMULATES:
                    // each echo joins only that stream's own program order,
                    // never foreign edges, and the frozen generation join is
                    // never clobbered.
                    let key = SyncHbKey::new(event.stmt_id, sync_kind, *bar_id, *cycle, scope);
                    let published = states[stream_id].published(stream_id, mask, ordinal);
                    let acc = sync_arrivals.entry(key.clone()).or_default();
                    join_clock(acc, &published);
                    if *count == *thread_count {
                        let acc = sync_arrivals.remove(&key).unwrap_or_default();
                        let release = sync_releases.entry(key).or_default();
                        join_clock(release, &acc);
                    }
                }
                // semaphore release (gmem_atomic_add, order=release): publish this
                // event's clock under (cell, POST-increment value) — the value-keyed
                // release a wait_eq on that value will acquire. Relaxed order
                // publishes nothing (no happens-before edge), mirroring hardware.
                TraceEventKind::SemRelease {
                    key,
                    new_value,
                    order,
                    ..
                } if *order == super::protocol::GmemAtomicOrderEvent::Release => {
                    sem_releases.insert(
                        (*key, *new_value),
                        states[stream_id].published(stream_id, mask, ordinal),
                    );
                }
                _ => {}
            }
        }
        Self { meta }
    }

    /// Is lane `lane` of `stream` at `ordinal` visible to lane `to_lane` at
    /// event `to`? Through the stream's all-lane prefix, through the lane's
    /// own refinement entry (snapshot or overlay), or — same stream, same
    /// lane — through program order.
    fn slice_visible(
        &self,
        stream: usize,
        lane: u8,
        ordinal: usize,
        to: usize,
        to_lane: u8,
    ) -> bool {
        let mt = &self.meta[to];
        if mt.stream_id == stream && to_lane == lane && mt.ordinal >= ordinal {
            return true;
        }
        let ld = lane_dim(stream, lane as usize);
        let pd = prefix_dim(stream);
        if clock_get(&mt.shared, pd) >= ordinal || clock_get(&mt.shared, ld) >= ordinal {
            return true;
        }
        if let Some((_, extra)) = mt.extras.iter().find(|(l, _)| *l == to_lane) {
            if clock_get(extra, pd) >= ordinal || clock_get(extra, ld) >= ordinal {
                return true;
            }
        }
        false
    }

    /// Does lane `to_lane` at `to` cover clock component `(dim, tick)`? A
    /// lane-dimension component is also covered by its stream's all-lane
    /// prefix, and `to_lane`'s own dimension sits at `to`'s ordinal.
    fn covers(&self, to: usize, to_lane: u8, dim: usize, tick: usize) -> bool {
        let mt = &self.meta[to];
        let mut cover = clock_get(&mt.shared, dim);
        if dim == lane_dim(mt.stream_id, to_lane as usize) {
            cover = cover.max(mt.ordinal);
        }
        if let Some((_, extra)) = mt.extras.iter().find(|(l, _)| *l == to_lane) {
            cover = cover.max(clock_get(extra, dim));
        }
        if cover >= tick {
            return true;
        }
        // A lane component is subsumed by its stream's all-lane prefix.
        let slot = dim % STREAM_DIMS;
        if slot < WARP_LANES {
            let pd = dim - slot + WARP_LANES;
            let mut prefix = clock_get(&mt.shared, pd);
            if let Some((_, extra)) = mt.extras.iter().find(|(l, _)| *l == to_lane) {
                prefix = prefix.max(clock_get(extra, pd));
            }
            if prefix >= tick {
                return true;
            }
        }
        false
    }

    /// Can the ENGINE that performs the async-proxy access `to` observe lane
    /// `lane` of `stream` at `ordinal`?
    ///
    /// This is the second projection of the same clock. The issuing lane's own
    /// view is deliberately NOT consulted: the bytes are touched by the
    /// engine, so a store reaches it only by riding a `fence.proxy.async`
    /// into the stream's engine clock — the fencing lane publishes what IT
    /// could see, which is its own prior stores plus anything a convergence
    /// point or barrier had already carried into its view.
    /// The event's position on its stream's number line.
    fn event_ordinal(&self, idx: usize) -> usize {
        self.meta[idx].ordinal
    }

    fn engine_sees(&self, stream: usize, lane: u8, ordinal: usize, to: usize) -> bool {
        let Some(engine) = self.meta[to].engine.as_ref() else {
            return false;
        };
        clock_get(engine, prefix_dim(stream)) >= ordinal
            || clock_get(engine, lane_dim(stream, lane as usize)) >= ordinal
    }

    /// Lane-slice happens-before: lane `from_lane`'s clock at `from` is at or
    /// below lane `to_lane`'s clock at `to`.
    fn ordered_lane(&self, from: usize, from_lane: u8, to: usize, to_lane: u8) -> bool {
        if from == to {
            return true;
        }
        if from >= self.meta.len() || to >= self.meta.len() {
            return false;
        }
        let mf = &self.meta[from];
        let mt = &self.meta[to];
        if mf.mask == 0 || mt.mask == 0 {
            return false;
        }
        if !self.slice_visible(mf.stream_id, from_lane, mf.ordinal, to, to_lane) {
            return false;
        }
        // A stream's shared clock only grows along its trace order, so the
        // forward same-stream case skips comparing it.
        if !(mf.stream_id == mt.stream_id && from < to) {
            for &(dim, tick) in mf.shared.iter() {
                if !self.covers(to, to_lane, dim, tick) {
                    return false;
                }
            }
        }
        if let Some((_, extra)) = mf.extras.iter().find(|(l, _)| *l == from_lane) {
            for &(dim, tick) in extra.iter() {
                if !self.covers(to, to_lane, dim, tick) {
                    return false;
                }
            }
        }
        true
    }

    /// Event-level happens-before: EVERY executing lane slice of `from`
    /// precedes EVERY executing lane slice of `to`.
    fn happens_before(&self, from: usize, to: usize) -> bool {
        if from == to {
            return true;
        }
        if from >= self.meta.len() || to >= self.meta.len() {
            return false;
        }
        let mf = &self.meta[from];
        let mt = &self.meta[to];
        if mf.mask == 0 || mt.mask == 0 {
            return false;
        }
        let same_stream_forward = mf.stream_id == mt.stream_id && from < to;
        // The join over `from`'s lane slices, reduced to the components not
        // already below `to`'s shared clock. Whatever remains must sit below
        // every executing `to` lane's private view. A stream's shared clock
        // only grows along its trace order, so the forward same-stream case
        // skips comparing it.
        let mut residual: Vec<(usize, usize)> = Vec::new();
        if !same_stream_forward {
            for &(dim, tick) in mf.shared.iter() {
                if !shared_covers(&mt.shared, dim, tick) {
                    residual.push((dim, tick));
                }
            }
        }
        // The own components {lane l of from's stream at from's ordinal}.
        if clock_get(&mt.shared, prefix_dim(mf.stream_id)) < mf.ordinal {
            for from_lane in 0..WARP_LANES {
                if mf.mask & (1 << from_lane) == 0 {
                    continue;
                }
                let dim = lane_dim(mf.stream_id, from_lane);
                if clock_get(&mt.shared, dim) < mf.ordinal {
                    residual.push((dim, mf.ordinal));
                }
            }
        }
        for (_, extra) in mf.extras.iter() {
            for &(dim, tick) in extra.iter() {
                if !shared_covers(&mt.shared, dim, tick) {
                    residual.push((dim, tick));
                }
            }
        }
        if residual.is_empty() {
            return true;
        }
        for to_lane in 0..WARP_LANES as u8 {
            if mt.mask & (1u32 << to_lane) == 0 {
                continue;
            }
            for &(dim, tick) in &residual {
                if !self.covers(to, to_lane, dim, tick) {
                    return false;
                }
            }
        }
        true
    }
}

/// Does a shared clock cover component `(dim, tick)`, counting a lane
/// component as covered by its stream's all-lane prefix?
fn shared_covers(shared: &Clock, dim: usize, tick: usize) -> bool {
    if clock_get(shared, dim) >= tick {
        return true;
    }
    let slot = dim % STREAM_DIMS;
    slot < WARP_LANES && clock_get(shared, dim - slot + WARP_LANES) >= tick
}

fn join_clock(dst: &mut Clock, src: &Clock) {
    for &(stream_id, tick) in src {
        clock_set_max(dst, stream_id, tick);
    }
}

fn clock_get(clock: &Clock, stream_id: usize) -> usize {
    clock
        .binary_search_by_key(&stream_id, |(stream, _)| *stream)
        .map(|idx| clock[idx].1)
        .unwrap_or(0)
}

fn clock_set_max(clock: &mut Clock, stream_id: usize, tick: usize) {
    match clock.binary_search_by_key(&stream_id, |(stream, _)| *stream) {
        Ok(idx) => clock[idx].1 = clock[idx].1.max(tick),
        Err(idx) => clock.insert(idx, (stream_id, tick)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::protocol::{BoxN, MemoryAccessKind, PoolId, Region};
    use super::*;
    use crate::ir::{DType, Layout, MemorySpace, SmemSwizzleLayout, Swizzle};

    fn scope(stream_id: usize) -> super::super::protocol::AccessScope {
        scope_in_cluster(stream_id, 0)
    }

    fn scope_in_cluster(
        stream_id: usize,
        cluster_id: usize,
    ) -> super::super::protocol::AccessScope {
        super::super::protocol::AccessScope {
            stream_id,
            cluster_id,
            cta_id: 0,
            ctaid_in_cluster: 0,
            lane_count: 32,
            warp_id: 0,
            active_lanes: 0xFFFF_FFFF,
        }
    }

    fn event(stmt_id: usize, payload: TraceEventKind) -> TraceEvent {
        TraceEvent::new(stmt_id, "TestStmt", payload)
    }

    fn empty_kernel(name: &str, body: Vec<Stmt>) -> Kernel {
        Kernel {
            name: name.into(),
            args: vec![],
            body,
            num_warps: 1,
            smem_size_bytes: 64,
            launch_shape: vec![1],
            cluster_shape: vec![1],
        }
    }

    fn smem_tensor(id: u32, byte_offset: usize) -> Arc<Tensor> {
        Arc::new(Tensor {
            id,
            space: MemorySpace::Smem,
            dtype: DType::U32,
            shape: vec![4],
            layout: Some(Layout::Swizzle(SmemSwizzleLayout {
                swizzle: Swizzle::None,
            })),
            byte_offset: Some(byte_offset),
        })
    }

    fn tensor_region(tensor: &Arc<Tensor>) -> Region {
        let start = tensor.byte_offset.unwrap();
        Region {
            tensor_id: tensor.id,
            owner: PoolId::Smem { cta_id: 0 },
            boxes: RegionBoxes::Boxes(vec![BoxN::new(vec![(start, start + 16)])]),
            lane_boxes: None,
        }
    }

    fn smem_region(cta_id: usize, start: usize, end: usize) -> Region {
        Region {
            tensor_id: 1,
            owner: PoolId::Smem { cta_id },
            boxes: RegionBoxes::Boxes(vec![BoxN::new(vec![(start, end)])]),
            lane_boxes: None,
        }
    }

    fn tmem_region(col_start: usize, n_cols: usize) -> Region {
        Region {
            tensor_id: 9,
            owner: PoolId::Tmem { cta_id: 0 },
            boxes: RegionBoxes::Boxes(vec![BoxN::new(vec![
                (0, 128),
                (col_start * 4, (col_start + n_cols) * 4),
            ])]),
            lane_boxes: None,
        }
    }

    fn pass_status(report: &ProtocolReport, name: &str) -> ProtocolStatus {
        report
            .pass_summary
            .iter()
            .find(|summary| summary.name == name)
            .map(|summary| summary.status)
            .expect("missing pass summary")
    }

    fn diagnostic_codes(report: &ProtocolReport) -> HashSet<&str> {
        report
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.code.as_str())
            .collect()
    }

    fn mbar_target() -> super::super::protocol::MbarTargetEvent {
        mbar_target_in_cluster(0)
    }

    fn mbar_target_in_cluster(cluster_id: usize) -> super::super::protocol::MbarTargetEvent {
        super::super::protocol::MbarTargetEvent {
            mbar_id: 0,
            cluster_id,
            ctaid_in_cluster: 0,
            stage: 0,
        }
    }

    #[test]
    fn schema_audit_requires_sync_arrive_before_sync() {
        let kernel = empty_kernel("schema_gap", vec![]);
        let events = vec![event(
            1,
            TraceEventKind::Sync {
                sync_kind: "cta".into(),
                thread_count: 32,
                cycle: 0,
                bar_id: None,
                scope: scope(0),
            },
        )];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "trace_schema_audit"),
            ProtocolStatus::Failed
        );
    }

    #[test]
    fn async_source_overwrite_uses_physical_pool_overlap_not_tensor_id() {
        let source = smem_tensor(1, 0);
        let alias = smem_tensor(2, 0);
        let kernel = empty_kernel(
            "physical_overlap",
            vec![
                Stmt::TensorDef {
                    tensor: source.clone(),
                },
                Stmt::TensorDef {
                    tensor: alias.clone(),
                },
            ],
        );
        let s = scope(7);
        let events = vec![
            event(1, TraceEventKind::CommitGroup { scope: s.clone() }),
            event(
                2,
                TraceEventKind::Read {
                    region: tensor_region(&source),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::TmaStore),
                    scope: s.clone(),
                },
            ),
            event(
                3,
                TraceEventKind::Write {
                    region: tensor_region(&alias),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: s,
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(report.diagnostics[0].code, "async_group_source_overwrite");
    }

    #[test]
    fn cross_stream_async_source_overwrite_records_trace_gap() {
        let source = smem_tensor(1, 0);
        let alias = smem_tensor(2, 0);
        let kernel = empty_kernel(
            "cross_stream_async_gap",
            vec![
                Stmt::TensorDef {
                    tensor: source.clone(),
                },
                Stmt::TensorDef {
                    tensor: alias.clone(),
                },
            ],
        );
        let events = vec![
            event(
                0,
                TraceEventKind::Write {
                    region: tensor_region(&source),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::TmaLoad),
                    scope: scope(0),
                },
            ),
            event(1, TraceEventKind::CommitGroup { scope: scope(0) }),
            event(
                2,
                TraceEventKind::Read {
                    region: tensor_region(&source),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::TmaStore),
                    scope: scope(0),
                },
            ),
            event(
                3,
                TraceEventKind::WaitGroup {
                    n: 0,
                    scope: scope(0),
                },
            ),
            event(
                4,
                TraceEventKind::Write {
                    region: tensor_region(&alias),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Failed
        );
        assert_eq!(
            pass_status(&report, "trace_gap_audit"),
            ProtocolStatus::Inconclusive
        );
        assert!(diagnostic_codes(&report).contains("memory_data_race"));
        assert!(diagnostic_codes(&report)
            .contains("async_group_cross_stream_unordered_source_overwrite"));
    }

    #[test]
    fn barrier_cycle_audit_rejects_wait_without_release_witness() {
        let kernel = empty_kernel("barrier_cycle", vec![]);
        let target = mbar_target();
        let events = vec![
            event(
                1,
                TraceEventKind::MbarInit {
                    target: target.clone(),
                    count: 1,
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::MbarWait {
                    target,
                    phase: 0,
                    scope: scope(0),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "barrier_cycle_audit"),
            ProtocolStatus::Failed
        );
        assert_eq!(
            pass_status(&report, "deadlock_freedom"),
            ProtocolStatus::Failed
        );
    }

    #[test]
    fn ordering_analysis_does_not_use_cross_stream_vector_order() {
        let events = vec![
            event(1, TraceEventKind::CommitGroup { scope: scope(0) }),
            event(2, TraceEventKind::CommitGroup { scope: scope(1) }),
        ];
        let ordering = OrderingAnalysis::new(&events);
        assert!(!ordering.happens_before(0, 1));
        assert!(!ordering.happens_before(1, 0));
    }

    #[test]
    fn same_stream_program_order_is_per_lane() {
        // Two plain full-mask events of one warp: program order holds LANE
        // against ITSELF; across lanes the slices stay unordered until a
        // convergence point, and the whole-event relation follows suit.
        let events = vec![
            event(1, TraceEventKind::CommitGroup { scope: scope(0) }),
            event(
                2,
                TraceEventKind::WaitGroup {
                    n: 0,
                    scope: scope(0),
                },
            ),
        ];
        let ordering = OrderingAnalysis::new(&events);
        assert!(ordering.ordered_lane(0, 5, 1, 5));
        assert!(!ordering.ordered_lane(0, 5, 1, 6));
        assert!(!ordering.happens_before(0, 1));
        assert!(!ordering.happens_before(1, 0));
    }

    #[test]
    fn ordering_analysis_keeps_mbar_identity_cluster_local() {
        let target_cluster0 = mbar_target_in_cluster(0);
        let target_cluster1 = mbar_target_in_cluster(1);
        let events = vec![
            event(
                1,
                TraceEventKind::MbarInit {
                    target: target_cluster0.clone(),
                    count: 1,
                    scope: scope_in_cluster(0, 0),
                },
            ),
            event(
                2,
                TraceEventKind::MbarArrive {
                    target: target_cluster0,
                    count: 1,
                    scope: scope_in_cluster(0, 0),
                },
            ),
            event(
                3,
                TraceEventKind::MbarWait {
                    target: target_cluster1,
                    phase: 0,
                    scope: scope_in_cluster(1, 1),
                },
            ),
        ];
        let ordering = OrderingAnalysis::new(&events);
        assert!(!ordering.happens_before(1, 2));
    }

    #[test]
    fn memory_race_check_uses_physical_smem_alias_regions() {
        // Two DISTINCT tensor ids at the same byte offset: the pass must
        // compare physical bytes, so the unordered cross-stream pair races.
        // A tensor-id-keyed comparison would see no overlap and pass.
        let writer = smem_tensor(1, 0);
        let alias = smem_tensor(2, 0);
        let kernel = empty_kernel(
            "memory_physical_alias",
            vec![
                Stmt::TensorDef {
                    tensor: writer.clone(),
                },
                Stmt::TensorDef {
                    tensor: alias.clone(),
                },
            ],
        );
        let events = vec![
            event(
                1,
                TraceEventKind::Write {
                    region: tensor_region(&writer),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Read {
                    region: tensor_region(&alias),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert!(diagnostic_codes(&report).contains("memory_data_race"));
    }

    #[test]
    fn memory_race_check_allows_read_without_prior_write() {
        let tensor = smem_tensor(1, 0);
        let kernel = empty_kernel(
            "memory_read_without_prior_write",
            vec![Stmt::TensorDef {
                tensor: tensor.clone(),
            }],
        );
        let events = vec![event(
            1,
            TraceEventKind::Read {
                region: tensor_region(&tensor),
                proxy: MemoryProxy::Generic,
                access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                scope: scope(0),
            },
        )];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Passed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Passed
        );
        assert!(!diagnostic_codes(&report).contains("memory_data_race"));
    }

    #[test]
    fn memory_race_check_rejects_unordered_write_write_overlap() {
        let kernel = empty_kernel("memory_write_write_race", vec![]);
        let events = vec![
            event(
                1,
                TraceEventKind::Write {
                    region: smem_region(0, 0, 16),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Write {
                    region: smem_region(0, 8, 24),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Failed
        );
        assert!(diagnostic_codes(&report).contains("memory_data_race"));
        assert_eq!(
            report.diagnostics[0]
                .details
                .get("overlap")
                .map(String::as_str),
            Some("[8,16)")
        );
    }

    #[test]
    fn memory_race_check_rejects_unordered_write_read_overlap() {
        let kernel = empty_kernel("memory_write_read_race", vec![]);
        let events = vec![
            event(
                1,
                TraceEventKind::Write {
                    region: smem_region(0, 0, 16),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Read {
                    region: smem_region(0, 8, 12),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Failed
        );
        assert!(diagnostic_codes(&report).contains("memory_data_race"));
    }

    #[test]
    fn memory_race_check_rejects_unordered_read_write_overlap() {
        let tensor = smem_tensor(1, 0);
        let kernel = empty_kernel(
            "memory_read_write_race",
            vec![Stmt::TensorDef {
                tensor: tensor.clone(),
            }],
        );
        let events = vec![
            event(
                2,
                TraceEventKind::Read {
                    region: tensor_region(&tensor),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(0),
                },
            ),
            event(
                3,
                TraceEventKind::Write {
                    region: tensor_region(&tensor),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Failed
        );
        assert!(diagnostic_codes(&report).contains("memory_data_race"));
    }

    #[test]
    fn memory_race_check_allows_unordered_read_read_overlap() {
        let kernel = empty_kernel("memory_read_read_ok", vec![]);
        let events = vec![
            event(
                1,
                TraceEventKind::Read {
                    region: smem_region(0, 0, 16),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Read {
                    region: smem_region(0, 8, 24),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Passed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Passed
        );
    }

    #[test]
    fn memory_race_check_allows_different_owner_overlap() {
        let kernel = empty_kernel("memory_different_owner_ok", vec![]);
        let events = vec![
            event(
                1,
                TraceEventKind::Write {
                    region: smem_region(0, 0, 16),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Read {
                    region: smem_region(1, 0, 16),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Passed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Passed
        );
    }

    #[test]
    fn memory_race_check_allows_same_owner_non_overlap() {
        let kernel = empty_kernel("memory_non_overlap_ok", vec![]);
        let events = vec![
            event(
                1,
                TraceEventKind::Write {
                    region: smem_region(0, 0, 16),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Read {
                    region: smem_region(0, 16, 32),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Passed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Passed
        );
    }

    #[test]
    fn memory_race_check_accepts_mbar_ordered_write_read() {
        let kernel = empty_kernel("memory_mbar_ordered", vec![]);
        let target = mbar_target();
        let events = vec![
            event(
                1,
                TraceEventKind::MbarInit {
                    target: target.clone(),
                    count: 1,
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Write {
                    region: smem_region(0, 0, 16),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(0),
                },
            ),
            event(
                3,
                TraceEventKind::MbarArrive {
                    target: target.clone(),
                    count: 1,
                    scope: scope(0),
                },
            ),
            event(
                4,
                TraceEventKind::MbarWait {
                    target,
                    phase: 0,
                    scope: scope(1),
                },
            ),
            event(
                5,
                TraceEventKind::Read {
                    region: smem_region(0, 0, 16),
                    proxy: MemoryProxy::Generic,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Generic),
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Passed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Passed
        );
    }

    #[test]
    fn memory_race_check_accepts_large_write_partial_read_when_ordered() {
        let kernel = empty_kernel("memory_large_write_partial_read_ordered", vec![]);
        let events = vec![
            event(
                1,
                TraceEventKind::Write {
                    region: smem_region(0, 0, 64),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::TmaLoad),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Read {
                    region: smem_region(0, 0, 16),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Tcgen05Mma),
                    scope: scope(0),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Passed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Passed
        );
    }

    #[test]
    fn memory_race_check_rejects_large_write_partial_read_when_unordered() {
        let kernel = empty_kernel("memory_large_write_partial_read_race", vec![]);
        let events = vec![
            event(
                1,
                TraceEventKind::Write {
                    region: smem_region(0, 0, 64),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::TmaLoad),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Read {
                    region: smem_region(0, 0, 16),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Tcgen05Mma),
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Failed
        );
        assert!(diagnostic_codes(&report).contains("memory_data_race"));
    }

    #[test]
    fn memory_race_check_keeps_tmem_layout_f_lane_gap_non_overlap() {
        let kernel = empty_kernel("memory_tmem_layout_f_gap", vec![]);
        let alloc = tmem_region(0, 16);
        let layout_f = Region {
            tensor_id: 9,
            owner: PoolId::Tmem { cta_id: 0 },
            boxes: RegionBoxes::Boxes(vec![
                BoxN::new(vec![(0, 16), (0, 64)]),
                BoxN::new(vec![(32, 48), (0, 64)]),
            ]),
            lane_boxes: None,
        };
        let gap = Region {
            tensor_id: 9,
            owner: PoolId::Tmem { cta_id: 0 },
            boxes: RegionBoxes::Boxes(vec![BoxN::new(vec![(16, 32), (0, 64)])]),
            lane_boxes: None,
        };
        let events = vec![
            event(
                1,
                TraceEventKind::TmemAlloc {
                    cta_ids: vec![0],
                    region: alloc.clone(),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Write {
                    region: layout_f,
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Mma),
                    scope: scope(0),
                },
            ),
            event(
                3,
                TraceEventKind::Read {
                    region: gap,
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Mma),
                    scope: scope(0),
                },
            ),
            // The mma is async: a commit hands it to a barrier and the wait
            // is where it is observed to have landed, which the band's free
            // must follow.
            event(
                4,
                TraceEventKind::MbarInit {
                    target: mbar_target(),
                    count: 1,
                    scope: scope(0),
                },
            ),
            TraceEvent::new(
                5,
                "Tcgen05Commit",
                TraceEventKind::MbarArrive {
                    target: mbar_target(),
                    count: 1,
                    scope: scope(0),
                },
            ),
            event(
                6,
                TraceEventKind::MbarWait {
                    target: mbar_target(),
                    phase: 0,
                    scope: scope(0),
                },
            ),
            event(
                7,
                TraceEventKind::TmemDealloc {
                    cta_ids: vec![0],
                    region: alloc,
                    scope: scope(0),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Passed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Passed
        );
        assert!(!diagnostic_codes(&report).contains("memory_data_race"));
    }

    /// A kernel whose SMEM pool holds the 32 4-byte lane cells below.
    fn lane_cells_kernel(name: &str) -> Kernel {
        let mut kernel = empty_kernel(name, vec![]);
        kernel.smem_size_bytes = 128;
        kernel
    }

    /// One 4-byte cell per lane, `row` = the lane whose cell it is: the
    /// footprint of `smem[row]` executed by lane `lane`.
    fn lane_cell_region(lane: u8, row: usize) -> Region {
        Region {
            tensor_id: 1,
            owner: PoolId::Smem { cta_id: 0 },
            boxes: RegionBoxes::Boxes(vec![BoxN::new(vec![(row * 4, row * 4 + 4)])]),
            lane_boxes: Some(Arc::new(vec![(
                lane,
                BoxN::new(vec![(row * 4, row * 4 + 4)]),
            )])),
        }
    }

    /// A whole-warp access whose lanes address `map(lane)`: the union footprint
    /// plus the per-lane attribution the region builders emit.
    fn warp_rows_region(map: impl Fn(u8) -> usize) -> Region {
        let lane_boxes: Vec<(u8, BoxN)> = (0u8..32)
            .map(|lane| {
                let row = map(lane);
                (lane, BoxN::new(vec![(row * 4, row * 4 + 4)]))
            })
            .collect();
        let mut boxes: Vec<BoxN> = lane_boxes.iter().map(|(_, b)| b.clone()).collect();
        boxes.sort_by_key(|b| b.ranges[0]);
        Region {
            tensor_id: 1,
            owner: PoolId::Smem { cta_id: 0 },
            boxes: RegionBoxes::Boxes(boxes),
            lane_boxes: Some(Arc::new(lane_boxes)),
        }
    }

    fn access_event(stmt_id: usize, stmt_kind: &str, region: Region, write: bool) -> TraceEvent {
        let (proxy, access_kind) = (
            MemoryProxy::Generic,
            MemoryAccessKind::Tensor(TensorAccessKind::Generic),
        );
        let payload = if write {
            TraceEventKind::Write {
                region,
                proxy,
                access_kind,
                scope: scope(0),
            }
        } else {
            TraceEventKind::Read {
                region,
                proxy,
                access_kind,
                scope: scope(0),
            }
        };
        TraceEvent::new(stmt_id, stmt_kind, payload)
    }

    fn warp_sync_events(stmt_id: usize) -> Vec<TraceEvent> {
        vec![
            TraceEvent::new(
                stmt_id,
                "WarpSync",
                TraceEventKind::SyncArrive {
                    sync_kind: "warp".into(),
                    thread_count: 32,
                    count: 32,
                    cycle: 0,
                    bar_id: None,
                    scope: scope(0),
                },
            ),
            TraceEvent::new(
                stmt_id,
                "WarpSync",
                TraceEventKind::Sync {
                    sync_kind: "warp".into(),
                    thread_count: 32,
                    cycle: 0,
                    bar_id: None,
                    scope: scope(0),
                },
            ),
        ]
    }

    #[test]
    fn cross_lane_rejects_lane_rotated_read_after_write_without_warp_sync() {
        // Each lane writes its own row, then reads lane+1's row: a cross-lane
        // dependency inside one warp with no intervening warp-level sync.
        let kernel = lane_cells_kernel("cross_lane_rotate");
        let events = vec![
            access_event(1, "RegStore", warp_rows_region(|lane| lane as usize), true),
            access_event(
                2,
                "RegLoad",
                warp_rows_region(|lane| ((lane + 1) % 32) as usize),
                false,
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert!(diagnostic_codes(&report).contains("memory_data_race"));
        let diagnostic = &report.diagnostics[0];
        assert_eq!(
            diagnostic.details.get("lanes").map(String::as_str),
            Some("0,31")
        );
        assert_eq!(
            diagnostic.details.get("left_mode").map(String::as_str),
            Some("write")
        );
        assert_eq!(
            diagnostic.details.get("right_mode").map(String::as_str),
            Some("read")
        );
    }

    #[test]
    fn cross_lane_accepts_lane_rotated_read_after_warp_sync() {
        let kernel = lane_cells_kernel("cross_lane_rotate_synced");
        let mut events = vec![access_event(
            1,
            "RegStore",
            warp_rows_region(|lane| lane as usize),
            true,
        )];
        events.extend(warp_sync_events(2));
        events.push(access_event(
            3,
            "RegLoad",
            warp_rows_region(|lane| ((lane + 1) % 32) as usize),
            false,
        ));
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(
            report.status,
            ProtocolStatus::Passed,
            "{:?}",
            report.diagnostics
        );
    }

    #[test]
    fn cross_lane_accepts_same_lane_reuse_without_sync() {
        // Lane i rewrites and rereads only its own row: program order per lane
        // is all the ordering hardware needs.
        let kernel = lane_cells_kernel("cross_lane_same_lane");
        let events = vec![
            access_event(1, "RegStore", warp_rows_region(|lane| lane as usize), true),
            access_event(2, "RegLoad", warp_rows_region(|lane| lane as usize), false),
            access_event(3, "RegStore", warp_rows_region(|lane| lane as usize), true),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(
            report.status,
            ProtocolStatus::Passed,
            "{:?}",
            report.diagnostics
        );
    }

    #[test]
    fn cross_lane_accepts_pair_ordered_by_a_warp_collective_member() {
        // The rotated read is a StMatrix/LdMatrix-class collective: every lane
        // converges on it, so the pair needs no separate warp sync.
        let kernel = lane_cells_kernel("cross_lane_collective");
        let events = vec![
            access_event(1, "RegStore", warp_rows_region(|lane| lane as usize), true),
            access_event(
                2,
                "LdMatrix",
                warp_rows_region(|lane| ((lane + 1) % 32) as usize),
                false,
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(
            report.status,
            ProtocolStatus::Passed,
            "{:?}",
            report.diagnostics
        );
    }

    #[test]
    fn cross_lane_accepts_async_engine_member() {
        // An elected lane's async bulk write (engine-performed, drained by the
        // async passes) followed by whole-warp lane reads of the tile.
        let kernel = lane_cells_kernel("cross_lane_async");
        let tile = Region {
            tensor_id: 1,
            owner: PoolId::Smem { cta_id: 0 },
            boxes: RegionBoxes::Boxes(vec![BoxN::new(vec![(0, 128)])]),
            lane_boxes: None,
        };
        let events = vec![
            TraceEvent::new(
                1,
                "TmaLoad",
                TraceEventKind::Write {
                    region: tile,
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::TmaLoad),
                    scope: scope(0),
                },
            ),
            access_event(2, "RegLoad", warp_rows_region(|lane| lane as usize), false),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(
            report.status,
            ProtocolStatus::Passed,
            "{:?}",
            report.diagnostics
        );
    }

    #[test]
    fn cross_lane_reports_single_lane_masks_by_lane() {
        // A single-thread mask is attributed to its lane, so lane 0's write
        // followed by lane 3's read of the same cell is cross-lane.
        let kernel = lane_cells_kernel("cross_lane_single_thread");
        let events = vec![
            access_event(1, "StoreScalar", lane_cell_region(0, 2), true),
            access_event(2, "ScalarDef", lane_cell_region(3, 2), false),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert!(diagnostic_codes(&report).contains("memory_data_race"));
        assert_eq!(
            report.diagnostics[0]
                .details
                .get("lanes")
                .map(String::as_str),
            Some("0,3")
        );
    }

    #[test]
    fn cross_lane_accepts_single_lane_reuse_of_its_own_cell() {
        let kernel = lane_cells_kernel("cross_lane_single_thread_reuse");
        let events = vec![
            access_event(1, "StoreScalar", lane_cell_region(0, 2), true),
            access_event(2, "ScalarDef", lane_cell_region(0, 2), false),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(
            report.status,
            ProtocolStatus::Passed,
            "{:?}",
            report.diagnostics
        );
    }

    #[test]
    fn deadlock_freedom_accepts_mbar_wait_with_release_witness() {
        let kernel = empty_kernel("deadlock_release", vec![]);
        let target = mbar_target();
        let events = vec![
            event(
                1,
                TraceEventKind::MbarInit {
                    target: target.clone(),
                    count: 1,
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::MbarArrive {
                    target: target.clone(),
                    count: 1,
                    scope: scope(1),
                },
            ),
            event(
                3,
                TraceEventKind::MbarWait {
                    target,
                    phase: 0,
                    scope: scope(0),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Passed);
        assert_eq!(
            pass_status(&report, "deadlock_freedom"),
            ProtocolStatus::Passed
        );
    }

    #[test]
    fn deadlock_freedom_accepts_closed_sync() {
        let kernel = empty_kernel("deadlock_sync_closed", vec![]);
        let events = vec![
            event(
                1,
                TraceEventKind::SyncArrive {
                    sync_kind: "cta".into(),
                    count: 32,
                    thread_count: 32,
                    cycle: 0,
                    bar_id: None,
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Sync {
                    sync_kind: "cta".into(),
                    thread_count: 32,
                    cycle: 0,
                    bar_id: None,
                    scope: scope(0),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Passed);
        assert_eq!(
            pass_status(&report, "deadlock_freedom"),
            ProtocolStatus::Passed
        );
    }

    #[test]
    fn deadlock_freedom_rejects_unclosed_sync() {
        let kernel = empty_kernel("deadlock_sync_missing_arrival", vec![]);
        let events = vec![
            event(
                1,
                TraceEventKind::SyncArrive {
                    sync_kind: "cta".into(),
                    count: 16,
                    thread_count: 32,
                    cycle: 0,
                    bar_id: None,
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Sync {
                    sync_kind: "cta".into(),
                    thread_count: 32,
                    cycle: 0,
                    bar_id: None,
                    scope: scope(0),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "deadlock_freedom"),
            ProtocolStatus::Failed
        );
        assert!(diagnostic_codes(&report).contains("deadlock_freedom_missing_release_witness"));
    }

    #[test]
    fn deadlock_freedom_rejects_wait_group_without_commit() {
        let kernel = empty_kernel("deadlock_wait_group_missing_commit", vec![]);
        let events = vec![event(
            1,
            TraceEventKind::WaitGroup {
                n: 0,
                scope: scope(0),
            },
        )];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "deadlock_freedom"),
            ProtocolStatus::Failed
        );
        assert!(diagnostic_codes(&report).contains("deadlock_freedom_missing_release_witness"));
    }

    #[test]
    #[ignore = "wait-for cycle detection is gated off (DETECT_WAIT_FOR_CYCLES); a \
                real cycle deadlocks the runtime so the offline search is vacuous \
                on completed traces. Re-enable the flag to run this synthetic case."]
    fn deadlock_freedom_rejects_cross_resource_cycle() {
        let kernel = empty_kernel("deadlock_cross_resource_cycle", vec![]);
        let target = mbar_target();
        let events = vec![
            event(
                1,
                TraceEventKind::MbarInit {
                    target: target.clone(),
                    count: 1,
                    scope: scope(2),
                },
            ),
            event(
                2,
                TraceEventKind::SyncArrive {
                    sync_kind: "cta".into(),
                    count: 1,
                    thread_count: 2,
                    cycle: 0,
                    bar_id: None,
                    scope: scope(1),
                },
            ),
            event(
                3,
                TraceEventKind::MbarWait {
                    target: target.clone(),
                    phase: 0,
                    scope: scope(0),
                },
            ),
            event(
                4,
                TraceEventKind::SyncArrive {
                    sync_kind: "cta".into(),
                    count: 2,
                    thread_count: 2,
                    cycle: 0,
                    bar_id: None,
                    scope: scope(0),
                },
            ),
            event(
                5,
                TraceEventKind::Sync {
                    sync_kind: "cta".into(),
                    thread_count: 2,
                    cycle: 0,
                    bar_id: None,
                    scope: scope(1),
                },
            ),
            event(
                6,
                TraceEventKind::MbarArrive {
                    target,
                    count: 1,
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "deadlock_freedom"),
            ProtocolStatus::Failed
        );
        assert!(diagnostic_codes(&report).contains("deadlock_freedom_wait_for_cycle"));
    }

    #[test]
    fn deadlock_freedom_accepts_mixed_releasable_blockers() {
        let kernel = empty_kernel("deadlock_mixed_releasable", vec![]);
        let target = mbar_target();
        let events = vec![
            event(
                1,
                TraceEventKind::MbarInit {
                    target: target.clone(),
                    count: 1,
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::MbarArrive {
                    target: target.clone(),
                    count: 1,
                    scope: scope(0),
                },
            ),
            event(
                3,
                TraceEventKind::MbarWait {
                    target,
                    phase: 0,
                    scope: scope(0),
                },
            ),
            event(4, TraceEventKind::CommitGroup { scope: scope(0) }),
            event(
                5,
                TraceEventKind::WaitGroup {
                    n: 0,
                    scope: scope(0),
                },
            ),
            event(
                6,
                TraceEventKind::SyncArrive {
                    sync_kind: "cta".into(),
                    count: 32,
                    thread_count: 32,
                    cycle: 0,
                    bar_id: None,
                    scope: scope(0),
                },
            ),
            event(
                7,
                TraceEventKind::Sync {
                    sync_kind: "cta".into(),
                    thread_count: 32,
                    cycle: 0,
                    bar_id: None,
                    scope: scope(0),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Passed);
        assert_eq!(
            pass_status(&report, "deadlock_freedom"),
            ProtocolStatus::Passed
        );
    }

    #[test]
    fn tmem_lifecycle_rejects_use_before_alloc() {
        let kernel = empty_kernel("tmem_lifecycle", vec![]);
        let region = tmem_region(0, 32);
        let events = vec![event(
            1,
            TraceEventKind::Read {
                region,
                proxy: MemoryProxy::Async,
                access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Ld),
                scope: scope(0),
            },
        )];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "tmem_lifecycle_order"),
            ProtocolStatus::Failed
        );
    }

    #[test]
    fn tmem_async_hazard_rejects_same_stream_overlap_before_wait() {
        let kernel = empty_kernel("tmem_async_overlap", vec![]);
        let region = tmem_region(0, 32);
        let events = vec![
            event(
                1,
                TraceEventKind::TmemAlloc {
                    cta_ids: vec![0],
                    region: region.clone(),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Write {
                    region: region.clone(),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::St),
                    scope: scope(0),
                },
            ),
            event(
                3,
                TraceEventKind::Write {
                    region: region.clone(),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::St),
                    scope: scope(0),
                },
            ),
            event(
                4,
                TraceEventKind::TmemDealloc {
                    cta_ids: vec![0],
                    region,
                    scope: scope(0),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "tcgen05_async_hazard"),
            ProtocolStatus::Failed
        );
    }

    #[test]
    fn tmem_async_wait_closes_same_stream_window() {
        let kernel = empty_kernel("tmem_async_wait", vec![]);
        let region = tmem_region(0, 32);
        let events = vec![
            event(
                1,
                TraceEventKind::TmemAlloc {
                    cta_ids: vec![0],
                    region: region.clone(),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Write {
                    region: region.clone(),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::St),
                    scope: scope(0),
                },
            ),
            event(
                3,
                TraceEventKind::TmemWait {
                    async_kind: TmemAsyncKind::St,
                    scope: scope(0),
                },
            ),
            event(
                4,
                TraceEventKind::Write {
                    region: region.clone(),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::St),
                    scope: scope(0),
                },
            ),
            event(
                5,
                TraceEventKind::TmemWait {
                    async_kind: TmemAsyncKind::St,
                    scope: scope(0),
                },
            ),
            event(
                6,
                TraceEventKind::TmemDealloc {
                    cta_ids: vec![0],
                    region,
                    scope: scope(0),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        assert_eq!(report.status, ProtocolStatus::Passed);
        assert_eq!(
            pass_status(&report, "deadlock_freedom"),
            ProtocolStatus::Passed
        );
        assert_eq!(
            pass_status(&report, "tcgen05_async_hazard"),
            ProtocolStatus::Passed
        );
    }

    #[test]
    fn cross_stream_tmem_overlap_is_memory_race() {
        let kernel = empty_kernel("cross_stream_tmem_gap", vec![]);
        let region = tmem_region(0, 32);
        let events = vec![
            event(
                1,
                TraceEventKind::TmemAlloc {
                    cta_ids: vec![0],
                    region: region.clone(),
                    scope: scope(0),
                },
            ),
            event(
                2,
                TraceEventKind::Write {
                    region: region.clone(),
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::St),
                    scope: scope(0),
                },
            ),
            event(
                3,
                TraceEventKind::Write {
                    region,
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::St),
                    scope: scope(1),
                },
            ),
        ];
        let report = run_offline_checkers(
            &kernel,
            ProtocolReport::new(ProtocolStatus::Passed),
            &events,
        );
        // Cross-stream conflicting TMEM writes are a data race; `memory_race_check`
        // proves it directly (no separate redundant cross-stream TMEM pass).
        assert_eq!(report.status, ProtocolStatus::Failed);
        assert_eq!(
            pass_status(&report, "memory_race_check"),
            ProtocolStatus::Failed
        );
        assert_eq!(
            pass_status(&report, "trace_gap_audit"),
            ProtocolStatus::Passed
        );
        assert!(diagnostic_codes(&report).contains("memory_data_race"));
    }
}
