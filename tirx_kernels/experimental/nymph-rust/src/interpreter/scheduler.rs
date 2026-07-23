//! Per-warp stream scheduling + frame bookkeeping.
//!
//! One stream per (cta, warp), each running the WHOLE kernel body top to
//! bottom — the warp is the hardware's own lockstep unit, everything above it
//! is concurrency. There are no epochs and no implicit barriers: cross-warp
//! and cross-CTA ordering comes only from explicit sync ops (mbarrier waits
//! park and wake event-driven; collective syncs rendezvous in shared state).
//! Each stream owns a frame stack; loop-var writes are eager. Bodies are
//! borrowed from the kernel (`'k`), zero-copy. `flatten/unflatten_coord` are
//! dimension-0-fastest.

use super::threads::{canonical_thread_mask, filter_thread_mask, Coord, ThreadId, ThreadMask};
use super::values::scalars::ScalarValues;
use crate::ir::{Kernel, Stmt};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CtaActivityStatus {
    Missing,
    NotStarted,
    Active,
    Exited,
}

pub struct Frame<'k> {
    pub body: &'k [Stmt],
    pub pc: usize,
    pub active_mask: ThreadMask,
}

pub struct LoopFrame<'k> {
    pub var_id: u32,
    pub current: i64,
    pub stop: i64,
    pub step: i64,
    pub body: &'k [Stmt],
    pub pc: usize,
    pub active_mask: ThreadMask,
}

pub struct DynamicLoopFrame<'k> {
    pub body: &'k [Stmt],
    pub pc: usize,
    pub active_mask: ThreadMask,
}

pub enum FrameKind<'k> {
    Plain(Frame<'k>),
    Loop(LoopFrame<'k>),
    DynamicLoop(DynamicLoopFrame<'k>),
}

pub struct ExecutionStream<'k> {
    pub stream_id: usize,
    pub cta_id: usize,
    pub cta_coord: Vec<usize>,
    pub cluster_id: usize,
    pub ctaid_in_cluster: usize,
    pub cluster_coord: Vec<usize>,
    pub cta_coord_in_cluster: Vec<usize>,
    pub initial_active_mask: ThreadMask,
    pub frames: Vec<FrameKind<'k>>,
    pub stream_step: usize,
    pub completed: bool,
}

/// dim 0 fastest (column-major linearization).
pub fn flatten_coord(coord: &[usize], shape: &[usize]) -> usize {
    let mut linear = 0usize;
    let mut stride = 1usize;
    for (c, s) in coord.iter().zip(shape.iter()) {
        linear += c * stride;
        stride *= s;
    }
    linear
}

pub fn unflatten_coord(linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coord = Vec::with_capacity(shape.len());
    let mut remaining = linear;
    for &s in shape {
        coord.push(remaining % s);
        remaining /= s;
    }
    coord
}

fn shape_product(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Build per-CTA thread masks (grid expansion).
pub fn expand_threads_by_cta(kernel: &Kernel) -> Vec<ThreadMask> {
    let launch = &kernel.launch_shape;
    let cluster = &kernel.cluster_shape;
    let cluster_grid: Vec<usize> = launch
        .iter()
        .zip(cluster.iter())
        .map(|(l, c)| l / c)
        .collect();
    let total = shape_product(launch);
    let num_warps = kernel.num_warps as usize;
    let mut masks = Vec::with_capacity(total);
    for linear_cta in 0..total {
        let cta_coord = unflatten_coord(linear_cta, launch);
        let cta_id = flatten_coord(&cta_coord, launch);
        let cluster_coord: Vec<usize> = cta_coord
            .iter()
            .zip(cluster.iter())
            .map(|(c, cl)| c / cl)
            .collect();
        let cta_coord_in_cluster: Vec<usize> = cta_coord
            .iter()
            .zip(cluster.iter())
            .map(|(c, cl)| c % cl)
            .collect();
        let cluster_id = flatten_coord(&cluster_coord, &cluster_grid);
        let ctaid_in_cluster = flatten_coord(&cta_coord_in_cluster, cluster);
        let mut threads = Vec::with_capacity(num_warps * 32);
        for warp_id in 0..num_warps {
            for lane_id in 0..32 {
                threads.push(ThreadId {
                    cta_id,
                    cta_coord: Coord::from_slice(&cta_coord),
                    cluster_id,
                    ctaid_in_cluster,
                    cluster_coord: Coord::from_slice(&cluster_coord),
                    cta_coord_in_cluster: Coord::from_slice(&cta_coord_in_cluster),
                    warp_id,
                    lane_id,
                });
            }
        }
        masks.push(canonical_thread_mask(threads));
    }
    masks
}

pub fn kernel_scope_matches(
    thread: &ThreadId,
    warp: Option<u32>,
    lane: Option<u32>,
    elected: bool,
) -> bool {
    if let Some(w) = warp {
        if thread.warp_id != w as usize {
            return false;
        }
    }
    if let Some(l) = lane {
        return thread.lane_id == l as usize && (warp.is_some() || thread.warp_id == 0);
    }
    if elected {
        return thread.lane_id == 0 && (warp.is_some() || thread.warp_id == 0);
    }
    true
}

pub fn role_matches(
    thread: &ThreadId,
    warp: Option<u32>,
    warpgroup: Option<u32>,
    elected: bool,
) -> bool {
    if let Some(w) = warp {
        if thread.warp_id != w as usize {
            return false;
        }
    }
    if let Some(wg) = warpgroup {
        if thread.warpgroup_id() != wg as usize {
            return false;
        }
    }
    if !elected {
        return true;
    }
    if warp.is_some() {
        return thread.lane_id == 0;
    }
    if warpgroup.is_some() {
        return thread.warp_id % 4 == 0 && thread.lane_id == 0;
    }
    thread.warp_id == 0 && thread.lane_id == 0
}

/// One CTA's streams: one per warp, all materialized eagerly at launch.
pub struct CtaSchedule {
    pub stream_ids: Vec<usize>, // indices into SchedulerState.streams
}

pub struct SchedulerState<'k> {
    pub cta_thread_masks: Vec<ThreadMask>,
    pub streams: Vec<ExecutionStream<'k>>,
    pub schedules: Vec<CtaSchedule>,
}

impl<'k> SchedulerState<'k> {
    /// Per-warp streams: every (cta, warp) runs the WHOLE kernel body from
    /// the start, exactly like hardware. There are no phases and no implicit
    /// barriers — all cross-warp ordering comes from explicit sync ops.
    pub fn from_kernel(kernel: &'k Kernel) -> SchedulerState<'k> {
        let cta_thread_masks = expand_threads_by_cta(kernel);
        let num_warps = kernel.num_warps as usize;
        let mut streams = Vec::with_capacity(cta_thread_masks.len() * num_warps);
        let mut schedules = Vec::with_capacity(cta_thread_masks.len());
        for mask in &cta_thread_masks {
            let mut stream_ids = Vec::with_capacity(num_warps);
            for warp in 0..num_warps {
                let warp_mask = filter_thread_mask(mask, |t| t.warp_id == warp);
                if warp_mask.is_empty() {
                    continue;
                }
                let sid = streams.len();
                streams.push(make_stream(sid, kernel.body.as_slice(), warp_mask));
                stream_ids.push(sid);
            }
            schedules.push(CtaSchedule { stream_ids });
        }
        SchedulerState {
            cta_thread_masks,
            streams,
            schedules,
        }
    }

    /// With eager materialization a CTA is Active until its last warp stream
    /// completes, then Exited. (`NotStarted` no longer occurs: every stream
    /// exists from round 0.)
    pub fn cta_activity_status(&self, cta_id: usize) -> CtaActivityStatus {
        if cta_id >= self.schedules.len() {
            return CtaActivityStatus::Missing;
        }
        let any_live = self.schedules[cta_id]
            .stream_ids
            .iter()
            .any(|&sid| !self.streams[sid].completed);
        if any_live {
            CtaActivityStatus::Active
        } else {
            CtaActivityStatus::Exited
        }
    }

    pub fn all_completed(&self) -> bool {
        self.streams.iter().all(|s| s.completed)
    }
}

fn make_stream<'k>(
    stream_id: usize,
    body: &'k [Stmt],
    active_mask: ThreadMask,
) -> ExecutionStream<'k> {
    let first = active_mask[0].clone();
    ExecutionStream {
        stream_id,
        cta_id: first.cta_id,
        cta_coord: first.cta_coord.to_vec(),
        cluster_id: first.cluster_id,
        ctaid_in_cluster: first.ctaid_in_cluster,
        cluster_coord: first.cluster_coord.to_vec(),
        cta_coord_in_cluster: first.cta_coord_in_cluster.to_vec(),
        initial_active_mask: active_mask.clone(),
        frames: vec![FrameKind::Plain(Frame {
            body,
            pc: 0,
            active_mask,
        })],
        stream_step: 0,
        completed: false,
    }
}

pub fn push_frame<'k>(stream: &mut ExecutionStream<'k>, body: &'k [Stmt], active_mask: ThreadMask) {
    stream.frames.push(FrameKind::Plain(Frame {
        body,
        pc: 0,
        active_mask,
    }));
}

#[allow(clippy::too_many_arguments)]
pub fn push_loop<'k>(
    stream: &mut ExecutionStream<'k>,
    var_id: u32,
    start: i64,
    stop: i64,
    step: i64,
    body: &'k [Stmt],
    active_mask: ThreadMask,
) {
    stream.frames.push(FrameKind::Loop(LoopFrame {
        var_id,
        current: start,
        stop,
        step,
        body,
        pc: 0,
        active_mask,
    }));
}

pub fn push_dynamic_loop<'k>(
    stream: &mut ExecutionStream<'k>,
    body: &'k [Stmt],
    active_mask: ThreadMask,
) {
    stream.frames.push(FrameKind::DynamicLoop(DynamicLoopFrame {
        body,
        pc: 0,
        active_mask,
    }));
}

pub fn break_dynamic_loop(stream: &mut ExecutionStream<'_>) -> bool {
    let Some(loop_idx) = stream
        .frames
        .iter()
        .rposition(|frame| matches!(frame, FrameKind::DynamicLoop(_)))
    else {
        return false;
    };
    stream.frames.truncate(loop_idx);
    true
}

/// Advance a specific frame's pc (the frame the current stmt lives in — stable
/// under pushes since pushes only append). `advance_continue`/`advance` use this.
pub fn advance_frame_at(stream: &mut ExecutionStream<'_>, frame_index: usize) {
    match stream.frames.get_mut(frame_index) {
        Some(FrameKind::Plain(f)) => f.pc += 1,
        Some(FrameKind::Loop(f)) => f.pc += 1,
        Some(FrameKind::DynamicLoop(f)) => f.pc += 1,
        None => {}
    }
}

/// Drain finished frames, then return `(stmt, active_mask, frame_index)` — the
/// frame_index is the index of the frame the stmt lives in (for advancing it).
/// Loop vars are eagerly written at the start of each iteration (pc == 0).
pub fn current_stmt<'k>(
    stream: &mut ExecutionStream<'k>,
    scalars: &mut ScalarValues,
) -> Option<(&'k Stmt, ThreadMask, usize)> {
    loop {
        let idx = match stream.frames.len().checked_sub(1) {
            None => {
                stream.completed = true;
                return None;
            }
            Some(i) => i,
        };
        match &mut stream.frames[idx] {
            FrameKind::Plain(f) => {
                if f.pc < f.body.len() {
                    return Some((&f.body[f.pc], f.active_mask.clone(), idx));
                }
                stream.frames.pop();
            }
            FrameKind::Loop(f) => {
                if f.current >= f.stop {
                    stream.frames.pop();
                    continue;
                }
                if f.pc < f.body.len() {
                    if f.pc == 0 {
                        // eager loop-var write at the start of the iteration
                        let (var_id, current, mask) = (f.var_id, f.current, f.active_mask.clone());
                        scalars.write_mask(&mask, var_id, current);
                    }
                    return Some((&f.body[f.pc], f.active_mask.clone(), idx));
                }
                // iteration body finished
                f.current += f.step;
                if f.current < f.stop {
                    f.pc = 0;
                } else {
                    stream.frames.pop();
                }
            }
            FrameKind::DynamicLoop(f) => {
                if f.body.is_empty() {
                    stream.frames.pop();
                    continue;
                }
                if f.pc < f.body.len() {
                    return Some((&f.body[f.pc], f.active_mask.clone(), idx));
                }
                f.pc = 0;
            }
        }
    }
}
