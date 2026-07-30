//! tcgen05 MMA + TMEM<->REG datapath + commit — port of `semantics/tcgen05.py`.
//! The GEMM compute core: D = A @ Bᵀ (f32 accumulate from f16/bf16), placed into
//! the TMEM accumulator; ld/st move the accumulator to/from registers.

use super::super::diagnostics::{IResult, InterpreterError};
use super::super::mbar_ops::{
    arrive_mbarrier_cell, initialized_mbar_cell, multicast_target_ctas, peer_ctaid_in_cluster,
    retarget_mbar, uniform_mbar_target, MbarTarget,
};
use super::super::outcomes::{StepStatus, WakeCondition};
use super::super::protocol::{
    MemoryAccessKind, MemoryProxy, PoolId, Region, TensorAccessKind, TmemAsyncKind, TraceEventKind,
};
use super::super::region;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::scalar_eval;
use super::super::scheduler::CtaActivityStatus;
use super::super::slice_indexing::ResolvedSlice;
use super::super::values::arrays::ValueArray1;
use super::super::values::dtypes::{decode_e2m1, decode_e4m3};
use super::super::values::mbars::{MbarCell, MbarCellKey};
use super::super::values::tcgen05_datapath::{
    datapath_has_cell_aliases_cached, datapath_index_arrays_cached, datapath_index_summary_cached,
};
use super::super::values::tmem::{TMEM_COLS, TMEM_ROWS};
use super::super::warp_context::WarpContext;
use crate::ir::{
    resolve_tcgen05_cp, resolve_tcgen05_mma, BlockScaleSpec, DType, MmaAOperand, MmaElemFormat,
    ResolvedCp, ResolvedMma, ResolvedScaleFootprint, ScaleFormat, SmemTile, Stmt, TmemAddr,
    TmemLayoutKind,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::Tcgen05Commit, execute_commit);
    reg.register(StmtKind::Tcgen05Ld, execute_ld);
    reg.register(StmtKind::Tcgen05WaitLd, execute_wait);
    reg.register(StmtKind::Tcgen05St, execute_st);
    reg.register(StmtKind::Tcgen05WaitSt, execute_wait);
    reg.register(StmtKind::Tcgen05Mma, execute_mma);
    reg.register(StmtKind::Tcgen05Cp, execute_cp);
}

fn execute_wait<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (async_kind, label) = match stmt {
        Stmt::Tcgen05WaitLd => (TmemAsyncKind::Ld, "wait_ld"),
        Stmt::Tcgen05WaitSt => (TmemAsyncKind::St, "wait_st"),
        _ => unreachable!(),
    };
    // tcgen05.wait::ld/st are warp-collective, exactly like the ld/st they
    // drain (resolve_datapath's full-warp rule).
    ctx.check_full_warp(
        format!("tcgen05_{label}_mask"),
        format!("tcgen05_{label} must be issued by one or more full warps"),
    )?;
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::TmemWait {
            async_kind,
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}

fn execute_commit<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    // Single-thread issue, exactly like the MMA/cp it drains (PTX; canon emits
    // it under `elect_sync`).
    ctx.check_single_thread_issue("tcgen05_commit_mask", "tcgen05_commit")?;
    let (mbar, stage, cta_group, multicast) = match stmt {
        Stmt::Tcgen05Commit {
            mbar,
            stage,
            cta_group,
            multicast_cta_mask,
        } => (mbar, stage, *cta_group, *multicast_cta_mask),
        _ => unreachable!(),
    };
    let base = uniform_mbar_target(ctx, mbar, stage.as_ref())?;
    let targets: Vec<MbarTarget> = match multicast {
        None => vec![base],
        Some(mask) => multicast_target_ctas(ctx, mask, "tcgen05", "Tcgen05Commit")?
            .into_iter()
            .map(|c| retarget_mbar(base, c))
            .collect(),
    };
    // peer-active gate for cta_group=2 (polled — re-checked each round)
    if cta_group == 2 {
        let peer = peer_ctaid_in_cluster(
            ctx,
            ctx.stream.ctaid_in_cluster,
            "tcgen05_peer",
            "tcgen05 peer out of range",
        )?;
        match ctx.cta_activity(ctx.global_cta_id(peer)) {
            CtaActivityStatus::Missing => {
                return Err(InterpreterError::new(
                    "tcgen05_peer_missing",
                    "tcgen05 peer CTA is missing",
                ))
            }
            CtaActivityStatus::Exited => {
                return Err(InterpreterError::new(
                    "tcgen05_peer_exited",
                    "tcgen05 peer CTA has exited",
                ))
            }
            CtaActivityStatus::Active => {}
            CtaActivityStatus::NotStarted => {
                return Ok(StepStatus::block(WakeCondition::Polled));
            }
        }
    }
    let mut cells: HashMap<MbarCellKey, MbarCell> = HashMap::new();
    for target in &targets {
        let key = target.key();
        let cell = match cells.get(&key) {
            Some(c) => *c,
            None => initialized_mbar_cell(ctx, key)?,
        };
        if ctx.trace_mode() {
            ctx.emit(TraceEventKind::MbarArrive {
                target: (*target).into(),
                count: 1,
                scope: ctx.access_scope(),
            })?;
        }
        cells.insert(key, arrive_mbarrier_cell(cell, 1)?);
    }
    let mut keys = Vec::with_capacity(cells.len());
    for (key, cell) in cells {
        ctx.state.values.mbars.cells.insert(key, cell);
        keys.push(key);
    }
    Ok(StepStatus::advance_wake(keys))
}

// ---- datapath (ld/st) ----

#[derive(Clone, Copy)]
struct DatapathBounds {
    reg_size: usize,
    col_start: usize,
    col_end: usize,
}

/// `.32x32b` atoms address a whole 32-lane subpartition, so their taddr lane
/// corner must be 32-aligned (B200-verified: lane=16 faults). The 16-lane
/// `.16x*b` atoms cover HALF a subpartition; a second issue with row=16 covers
/// lanes 16..31 of each warp's partition (the TIRx M=128 two-slab fragment).
fn check_row_alignment(row: i64, shape: &str, label: &str) -> IResult<()> {
    let align = if shape.starts_with("16x") { 16 } else { 32 };
    if row % align != 0 {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_misaligned"),
            format!("tcgen05 row must be {align}-aligned for {shape}"),
        ));
    }
    Ok(())
}

/// The atom's lane span must fit INSIDE the issuing warp's own 32-lane
/// subpartition: `.32x32b` spans all 32 lanes (row must be 0); the 16-lane
/// `.16x*b` atoms may start at row 0 or 16. A larger row lands in the NEXT
/// subpartition's lanes — hardware UB that the sim would otherwise compute
/// silently (the atom lane indices are subpartition-relative).
fn check_subpartition_span(row: i64, shape: &str, label: &str) -> IResult<()> {
    let span: i64 = if shape.starts_with("16x") { 16 } else { 32 };
    if row < 0 || row + span > 32 {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_out_of_range"),
            format!("tcgen05_{label} row escapes the warp's 32-lane TMEM subpartition"),
        ));
    }
    Ok(())
}

fn resolve_datapath(
    ctx: &WarpContext,
    addr: &TmemAddr,
    shape: &str,
    num: u32,
    label: &str,
) -> IResult<(Array2<usize>, Array2<usize>, usize)> {
    ctx.check_full_warp(
        format!("tcgen05_{label}_mask"),
        format!("tcgen05_{label} must be issued by one or more full warps"),
    )?;
    let (row, col) = eval_tmem_addr_i64(ctx, addr, "tcgen05 address")?;
    check_row_alignment(row, shape, label)?;
    check_subpartition_span(row, shape, label)?;
    let (lane_idx, col_idx) = datapath_index_arrays_cached(shape, num as usize)?;
    let reg_size = lane_idx.ncols();
    let a = ctx.lanes.len();
    let mut lanes = Array2::<usize>::zeros((a, reg_size));
    let mut cols = Array2::<usize>::zeros((a, reg_size));
    for (ai, t) in ctx.lanes.iter().enumerate() {
        let subpart = (32 * (t.warp_id % 4)) as i64;
        for r in 0..reg_size {
            let lane = row + subpart + lane_idx[[t.lane_id, r]] as i64;
            let col = col + col_idx[[t.lane_id, r]] as i64;
            if lane < 0 || lane >= TMEM_ROWS as i64 || col < 0 || col >= TMEM_COLS as i64 {
                return Err(InterpreterError::new(
                    format!("tcgen05_{label}_out_of_range"),
                    format!("tcgen05_{label} addresses a TMEM cell outside the scratchpad"),
                ));
            }
            lanes[[ai, r]] = lane as usize;
            cols[[ai, r]] = col as usize;
        }
    }
    Ok((lanes, cols, reg_size))
}

fn resolve_datapath_bounds(
    ctx: &WarpContext,
    addr: &TmemAddr,
    shape: &str,
    num: u32,
    label: &str,
) -> IResult<DatapathBounds> {
    ctx.check_full_warp(
        format!("tcgen05_{label}_mask"),
        format!("tcgen05_{label} must be issued by one or more full warps"),
    )?;
    let (row, col) = eval_tmem_addr_i64(ctx, addr, "tcgen05 address")?;
    check_row_alignment(row, shape, label)?;
    check_subpartition_span(row, shape, label)?;
    let summary = datapath_index_summary_cached(shape, num as usize)?;
    let abs_col_start = col + summary.col_min as i64;
    let abs_col_end = col + summary.col_max as i64;
    if abs_col_start < 0 || abs_col_end >= TMEM_COLS as i64 {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_out_of_range"),
            format!("tcgen05_{label} addresses a TMEM cell outside the scratchpad"),
        ));
    }
    let mut checked_warps = Vec::new();
    for t in &ctx.lanes {
        if checked_warps.contains(&t.warp_id) {
            continue;
        }
        checked_warps.push(t.warp_id);
        let subpart = (32 * (t.warp_id % 4)) as i64;
        let lane_start = row + subpart + summary.lane_min as i64;
        let lane_end = row + subpart + summary.lane_max as i64;
        if lane_start < 0 || lane_end >= TMEM_ROWS as i64 {
            return Err(InterpreterError::new(
                format!("tcgen05_{label}_out_of_range"),
                format!("tcgen05_{label} addresses a TMEM cell outside the scratchpad"),
            ));
        }
    }
    if checked_warps.is_empty() {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_mask"),
            format!("tcgen05_{label} must be issued by one or more full warps"),
        ));
    }
    Ok(DatapathBounds {
        reg_size: summary.reg_size,
        col_start: abs_col_start as usize,
        col_end: abs_col_end as usize,
    })
}

fn trace_ldst_tmem_region(
    ctx: &WarpContext,
    addr: &TmemAddr,
    shape: &str,
    num: u32,
    label: &str,
) -> IResult<(Region, usize)> {
    ctx.check_full_warp(
        format!("tcgen05_{label}_mask"),
        format!("tcgen05_{label} must be issued by one or more full warps"),
    )?;
    let (row, col) = eval_tmem_addr_i64(ctx, addr, "tcgen05 address")?;
    check_row_alignment(row, shape, label)?;
    check_subpartition_span(row, shape, label)?;
    let summary = datapath_index_summary_cached(shape, num as usize)?;
    let col_start = col + summary.col_min as i64;
    let col_end = col + summary.col_max as i64 + 1;
    if col_start < 0 || col_end > TMEM_COLS as i64 {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_out_of_range"),
            format!("tcgen05_{label} addresses a TMEM cell outside the scratchpad"),
        ));
    }

    let mut subparts = Vec::new();
    let mut rects = Vec::new();
    for thread in &ctx.lanes {
        let subpart = thread.warp_id % 4;
        if subparts.contains(&subpart) {
            continue;
        }
        subparts.push(subpart);
        let lane_start = row + (32 * subpart) as i64 + summary.lane_min as i64;
        let lane_end = row + (32 * subpart) as i64 + summary.lane_max as i64 + 1;
        if lane_start < 0 || lane_end > TMEM_ROWS as i64 {
            return Err(InterpreterError::new(
                format!("tcgen05_{label}_out_of_range"),
                format!("tcgen05_{label} addresses a TMEM cell outside the scratchpad"),
            ));
        }
        rects.push((
            lane_start as usize,
            (lane_end - lane_start) as usize,
            col_start as usize,
            (col_end - col_start) as usize,
        ));
    }
    if rects.is_empty() {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_mask"),
            format!("tcgen05_{label} must be issued by one or more full warps"),
        ));
    }
    // TMEM regions carry no tensor identity (0): the (lane, lane_byte) box IS
    // the address now that TMEM is not a tensor.
    Ok((
        region::tmem_region_from_rects(0, ctx.stream.cta_id, rects)?,
        summary.reg_size,
    ))
}

fn execute_ld<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (dst, src, shape, num) = match stmt {
        Stmt::Tcgen05Ld {
            dst,
            src,
            shape,
            num,
        } => (dst, src, shape_str(shape), *num),
        _ => unreachable!(),
    };
    let dtype = dst.tensor.dtype;
    if ctx.trace_mode() && !ctx.state.trace.records_events() {
        let t_resolve = super::super::runner::prof_now();
        let bounds = resolve_datapath_bounds(ctx, src, shape, num, "ld")?;
        super::super::runner::prof_end("TcLd:resolve", t_resolve);
        let t_dst = super::super::runner::prof_now();
        let dst_r = ctx.eval_slice(dst)?;
        check_reg_fragment(&dst_r, dtype, bounds.reg_size, "ld")?;
        if is_packed_tmem_dtype(dtype) {
            let _ = packed_half_register_slice(&dst_r, bounds.reg_size, "ld")?;
        }
        super::super::runner::prof_end("TcLd:dst_check", t_dst);
        let t_alloc = super::super::runner::prof_now();
        if !tmem_col_range_allocated(ctx, ctx.stream.cta_id, bounds.col_start, bounds.col_end) {
            let (_, cols, _) = resolve_datapath(ctx, src, shape, num, "ld")?;
            check_tmem_cells_allocated(ctx, ctx.stream.cta_id, cols.iter().copied(), "tcgen05_ld")?;
        }
        super::super::runner::prof_end("TcLd:alloc_check", t_alloc);
        return Ok(StepStatus::advance());
    }
    if ctx.trace_mode() {
        let t_resolve = super::super::runner::prof_now();
        let (region, reg_size) = trace_ldst_tmem_region(ctx, src, shape, num, "ld")?;
        super::super::runner::prof_end("TcLd:resolve", t_resolve);
        let t_dst = super::super::runner::prof_now();
        let dst_r = ctx.eval_slice(dst)?;
        check_reg_fragment(&dst_r, dtype, reg_size, "ld")?;
        let packed_dst = if is_packed_tmem_dtype(dtype) {
            Some(packed_half_register_slice(&dst_r, reg_size, "ld")?)
        } else {
            None
        };
        super::super::runner::prof_end("TcLd:dst_check", t_dst);
        let t_alloc = super::super::runner::prof_now();
        check_tmem_region_allocated(ctx, &region, "tcgen05_ld")?;
        super::super::runner::prof_end("TcLd:alloc_check", t_alloc);
        let scope = ctx.access_scope();
        ctx.emit(TraceEventKind::Read {
            region,
            proxy: MemoryProxy::Async,
            access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Ld),
            scope: scope.clone(),
        })?;
        ctx.emit_tensor_write(packed_dst.as_ref().unwrap_or(&dst_r))?;
        return Ok(StepStatus::advance());
    }
    let t_resolve = super::super::runner::prof_now();
    let (lanes, cols, reg_size) = resolve_datapath(ctx, src, shape, num, "ld")?;
    super::super::runner::prof_end("TcLd:resolve", t_resolve);
    let t_dst = super::super::runner::prof_now();
    let dst_r = ctx.eval_slice(dst)?;
    check_reg_fragment(&dst_r, dtype, reg_size, "ld")?;
    let packed_dst = if is_packed_tmem_dtype(dtype) {
        Some(packed_half_register_slice(&dst_r, reg_size, "ld")?)
    } else {
        None
    };
    super::super::runner::prof_end("TcLd:dst_check", t_dst);
    let flat_lanes: Vec<usize> = lanes.iter().copied().collect();
    let flat_cols: Vec<usize> = cols.iter().copied().collect();
    let a = ctx.lanes.len();
    if is_packed_tmem_dtype(dtype) {
        let values = {
            let scratch = ctx.state.values.tmem.scratchpad_for(ctx.stream.cta_id)?;
            let timer = super::super::runner::prof_now();
            let values = scratch.read_packed_half_cells(dtype, &flat_lanes, &flat_cols)?;
            super::super::runner::prof_end("TcLd:tmem_read_cells", timer);
            values
        };
        let mut flat = Vec::with_capacity(a * reg_size * 2);
        for (lo, hi) in values {
            flat.push(lo);
            flat.push(hi);
        }
        let native =
            ValueArray1::from_f32_compute(Array1::from(flat), dtype).reshape2((a, reg_size * 2))?;
        let timer = super::super::runner::prof_now();
        ctx.registers_write(packed_dst.as_ref().unwrap(), &native)?;
        super::super::runner::prof_end("TcLd:reg_write", timer);
    } else {
        let values = {
            let scratch = ctx.state.values.tmem.scratchpad_for(ctx.stream.cta_id)?;
            let timer = super::super::runner::prof_now();
            let values = scratch.read_cells(dtype, &flat_lanes, &flat_cols)?;
            super::super::runner::prof_end("TcLd:tmem_read_cells", timer);
            values
        };
        let native = values.reshape2((a, reg_size))?;
        let timer = super::super::runner::prof_now();
        ctx.registers_write(&dst_r, &native)?;
        super::super::runner::prof_end("TcLd:reg_write", timer);
    }
    Ok(StepStatus::advance())
}

fn execute_st<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (dst, src, shape, num) = match stmt {
        Stmt::Tcgen05St {
            dst,
            src,
            shape,
            num,
        } => (dst, src, shape_str(shape), *num),
        _ => unreachable!(),
    };
    let dtype = src.tensor.dtype;
    if ctx.trace_mode() && !ctx.state.trace.records_events() {
        let t_resolve = super::super::runner::prof_now();
        let bounds = resolve_datapath_bounds(ctx, dst, shape, num, "st")?;
        super::super::runner::prof_end("TcSt:resolve", t_resolve);
        let t_src = super::super::runner::prof_now();
        let src_r = ctx.eval_slice(src)?;
        check_reg_fragment(&src_r, dtype, bounds.reg_size, "st")?;
        if is_packed_tmem_dtype(dtype) {
            let _ = packed_half_register_slice(&src_r, bounds.reg_size, "st")?;
        }
        super::super::runner::prof_end("TcSt:src_check", t_src);
        let t_overlap = super::super::runner::prof_now();
        check_tmem_st_no_overlap(ctx, shape, num)?;
        super::super::runner::prof_end("TcSt:overlap", t_overlap);
        let t_alloc = super::super::runner::prof_now();
        if !tmem_col_range_allocated(ctx, ctx.stream.cta_id, bounds.col_start, bounds.col_end) {
            let (_, cols, _) = resolve_datapath(ctx, dst, shape, num, "st")?;
            check_tmem_cells_allocated(ctx, ctx.stream.cta_id, cols.iter().copied(), "tcgen05_st")?;
        }
        super::super::runner::prof_end("TcSt:alloc_check", t_alloc);
        return Ok(StepStatus::advance());
    }
    if ctx.trace_mode() {
        let t_resolve = super::super::runner::prof_now();
        let (region, reg_size) = trace_ldst_tmem_region(ctx, dst, shape, num, "st")?;
        super::super::runner::prof_end("TcSt:resolve", t_resolve);
        let t_src = super::super::runner::prof_now();
        let src_r = ctx.eval_slice(src)?;
        check_reg_fragment(&src_r, dtype, reg_size, "st")?;
        let packed_src = if is_packed_tmem_dtype(dtype) {
            Some(packed_half_register_slice(&src_r, reg_size, "st")?)
        } else {
            None
        };
        super::super::runner::prof_end("TcSt:src_check", t_src);
        let t_overlap = super::super::runner::prof_now();
        check_tmem_st_no_overlap(ctx, shape, num)?;
        super::super::runner::prof_end("TcSt:overlap", t_overlap);
        let t_alloc = super::super::runner::prof_now();
        check_tmem_region_allocated(ctx, &region, "tcgen05_st")?;
        super::super::runner::prof_end("TcSt:alloc_check", t_alloc);
        let scope = ctx.access_scope();
        ctx.emit_tensor_read(packed_src.as_ref().unwrap_or(&src_r))?;
        ctx.emit(TraceEventKind::Write {
            region,
            proxy: MemoryProxy::Async,
            access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::St),
            scope,
        })?;
        return Ok(StepStatus::advance());
    }
    let t_resolve = super::super::runner::prof_now();
    let (lanes, cols, reg_size) = resolve_datapath(ctx, dst, shape, num, "st")?;
    super::super::runner::prof_end("TcSt:resolve", t_resolve);
    let t_src = super::super::runner::prof_now();
    let src_r = ctx.eval_slice(src)?;
    check_reg_fragment(&src_r, dtype, reg_size, "st")?;
    let packed_src = if is_packed_tmem_dtype(dtype) {
        Some(packed_half_register_slice(&src_r, reg_size, "st")?)
    } else {
        None
    };
    super::super::runner::prof_end("TcSt:src_check", t_src);
    let t_overlap = super::super::runner::prof_now();
    check_tmem_st_no_overlap(ctx, shape, num)?;
    super::super::runner::prof_end("TcSt:overlap", t_overlap);
    let t_flat = super::super::runner::prof_now();
    let flat_lanes: Vec<usize> = lanes.iter().copied().collect();
    let flat_cols: Vec<usize> = cols.iter().copied().collect();
    super::super::runner::prof_end("TcSt:flatten", t_flat);
    if is_packed_tmem_dtype(dtype) {
        let timer = super::super::runner::prof_now();
        let packed_values =
            read_packed_half_register_pairs(ctx, packed_src.as_ref().unwrap(), reg_size)?;
        super::super::runner::prof_end("TcSt:reg_read", timer);
        let cta_id = ctx.stream.cta_id;
        let sp = ctx
            .state
            .values
            .tmem
            .by_cta
            .get_mut(&cta_id)
            .ok_or_else(|| {
                InterpreterError::new(
                    "missing_tmem_scratchpad",
                    "tcgen05.st writes a missing TMEM scratchpad",
                )
            })?;
        let timer = super::super::runner::prof_now();
        sp.write_packed_half_cells(dtype, &flat_lanes, &flat_cols, &packed_values)?;
        super::super::runner::prof_end("TcSt:tmem_write_cells", timer);
    } else {
        let timer = super::super::runner::prof_now();
        let flat_values = ctx.registers_read(&src_r)?.flatten_to_1d();
        super::super::runner::prof_end("TcSt:reg_read", timer);
        let cta_id = ctx.stream.cta_id;
        let sp = ctx
            .state
            .values
            .tmem
            .by_cta
            .get_mut(&cta_id)
            .ok_or_else(|| {
                InterpreterError::new(
                    "missing_tmem_scratchpad",
                    "tcgen05.st writes a missing TMEM scratchpad",
                )
            })?;
        let timer = super::super::runner::prof_now();
        sp.write_cells(dtype, &flat_lanes, &flat_cols, &flat_values)?;
        super::super::runner::prof_end("TcSt:tmem_write_cells", timer);
    }
    Ok(StepStatus::advance())
}

fn read_packed_half_register_pairs(
    ctx: &WarpContext,
    src_r: &ResolvedSlice,
    reg_size: usize,
) -> IResult<Vec<(f32, f32)>> {
    let values = ctx.registers_read(src_r)?.to_f32_compute();
    debug_assert_eq!(values.ncols(), reg_size * 2);
    let mut out = Vec::with_capacity(values.nrows() * reg_size);
    for ai in 0..values.nrows() {
        for r in 0..reg_size {
            out.push((values[[ai, 2 * r]], values[[ai, 2 * r + 1]]));
        }
    }
    Ok(out)
}

fn is_packed_tmem_dtype(dtype: crate::ir::DType) -> bool {
    matches!(dtype, crate::ir::DType::F16 | crate::ir::DType::Bf16)
}

fn packed_half_register_slice(
    reg_slice: &ResolvedSlice,
    reg_size: usize,
    label: &str,
) -> IResult<ResolvedSlice> {
    if reg_slice.tensor.shape.len() == 1
        && reg_slice.offsets.ncols() == 1
        && reg_slice.shape.as_slice() == [reg_size]
        && reg_slice
            .offsets
            .column(0)
            .iter()
            .all(|&o| o >= 0 && (o as usize) + reg_size * 2 <= reg_slice.tensor.shape[0])
    {
        return Ok(ResolvedSlice {
            tensor: reg_slice.tensor.clone(),
            offsets: reg_slice.offsets.clone(),
            shape: vec![reg_size * 2],
        });
    }
    Err(InterpreterError::new(
        format!("tcgen05_{label}_shape"),
        format!(
            "tcgen05_{label} packed half register slice requires {reg_size} b32 registers backed by {} half values",
            reg_size * 2
        ),
    ))
}

fn check_reg_fragment(
    reg_slice: &ResolvedSlice,
    tmem_dtype: DType,
    reg_size: usize,
    label: &str,
) -> IResult<()> {
    if reg_slice.tensor.dtype != tmem_dtype {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_dtype"),
            format!("tcgen05_{label} REG and TMEM operands must share a dtype"),
        ));
    }
    if reg_slice.shape.as_slice() != [reg_size].as_slice() {
        return Err(InterpreterError::new(
            format!("tcgen05_{label}_shape"),
            format!("tcgen05_{label} register slice must have shape ({reg_size},)"),
        ));
    }
    Ok(())
}

fn check_tmem_st_no_overlap(ctx: &WarpContext, shape: &str, num: u32) -> IResult<()> {
    if datapath_has_cell_aliases_cached(shape, num as usize)? {
        return Err(InterpreterError::new(
            "overlapping_tmem_write",
            "tcgen05.st overlaps a TMEM cell",
        ));
    }
    let mut subpart_owner = [None; 4];
    for thread in &ctx.lanes {
        let subpart = thread.warp_id % 4;
        match subpart_owner[subpart] {
            Some(owner) if owner == thread.warp_id => {}
            Some(_) => {
                return Err(InterpreterError::new(
                    "overlapping_tmem_write",
                    "tcgen05.st overlaps a TMEM cell",
                ));
            }
            None => subpart_owner[subpart] = Some(thread.warp_id),
        }
    }
    Ok(())
}

fn check_tmem_region_allocated(ctx: &WarpContext, region: &Region, label: &str) -> IResult<()> {
    let PoolId::Tmem { cta_id } = region.owner else {
        return Err(InterpreterError::new(
            "trace_region_owner",
            "TMEM region has a non-TMEM owner",
        ));
    };
    let crate::interpreter::protocol::RegionBoxes::Boxes(region_boxes) = &region.boxes else {
        return Err(InterpreterError::new(
            "trace_region_rank",
            "TMEM region must use (lane, lane_byte) boxes",
        ));
    };
    for b in region_boxes {
        if b.ranges.len() != 2 || b.ranges[1].1 > region::TMEM_LANE_BYTES {
            return Err(InterpreterError::new(
                "trace_region_rank",
                "TMEM region must use (lane, lane_byte) boxes",
            ));
        }
        let col_start = b.ranges[1].0 / 4;
        let col_end = b.ranges[1].1.div_ceil(4);
        let covered = ctx.state.tmem_allocations.keys().any(|key| {
            key.cta_id == cta_id
                && key.col_start <= col_start
                && col_end <= key.col_start + key.n_cols
        });
        if !covered {
            return Err(InterpreterError::new(
                "missing_tmem_allocation",
                format!("{label} accesses a TMEM range without an active allocation"),
            ));
        }
    }
    Ok(())
}

fn tmem_col_range_allocated(
    ctx: &WarpContext,
    cta_id: usize,
    col_start: usize,
    col_end: usize,
) -> bool {
    ctx.state.tmem_allocations.keys().any(|key| {
        key.cta_id == cta_id && key.col_start <= col_start && col_end < key.col_start + key.n_cols
    })
}

fn check_tmem_cells_allocated<I>(
    ctx: &WarpContext,
    cta_id: usize,
    cols: I,
    label: &str,
) -> IResult<()>
where
    I: IntoIterator<Item = usize> + Clone,
{
    let mut range = None::<(usize, usize)>;
    for col in cols.clone() {
        range = Some(match range {
            Some((lo, hi)) => (lo.min(col), hi.max(col)),
            None => (col, col),
        });
    }
    if let Some((col_start, col_end)) = range {
        let covered = ctx.state.tmem_allocations.keys().any(|key| {
            key.cta_id == cta_id
                && key.col_start <= col_start
                && col_end < key.col_start + key.n_cols
        });
        if covered {
            return Ok(());
        }
    } else {
        return Ok(());
    }

    let mut cols: Vec<usize> = cols.into_iter().collect();
    cols.sort_unstable();
    cols.dedup();
    for col in cols {
        let covered = ctx.state.tmem_allocations.keys().any(|key| {
            key.cta_id == cta_id && key.col_start <= col && col < key.col_start + key.n_cols
        });
        if !covered {
            return Err(InterpreterError::new(
                "missing_tmem_allocation",
                format!("{label} accesses a TMEM range without an active allocation"),
            ));
        }
    }
    Ok(())
}

fn shape_str(shape: &crate::ir::LdStShape) -> &'static str {
    shape.as_str()
}

// ---- MMA ----

fn layout_error(code: &'static str, error: crate::ir::Tcgen05LayoutError) -> InterpreterError {
    InterpreterError::new(code, error.message)
}

fn eval_tmem_addr_i64(ctx: &WarpContext, addr: &TmemAddr, label: &str) -> IResult<(i64, i64)> {
    let row = ctx.eval_scalar_uniform(&addr.row, label, "divergent_operands")?;
    let relative_col = ctx.eval_scalar_uniform(&addr.col, label, "divergent_operands")?;
    let col = i64::from(addr.tensor.start_col)
        .checked_add(relative_col)
        .ok_or_else(|| InterpreterError::new("tcgen05_address", "TMEM column overflows"))?;
    Ok((row, col))
}

fn eval_tmem_addr(ctx: &WarpContext, addr: &TmemAddr, label: &str) -> IResult<(usize, usize)> {
    let (row, col) = eval_tmem_addr_i64(ctx, addr, label)?;
    if row < 0 || col < 0 {
        return Err(InterpreterError::new(
            "tcgen05_address",
            format!("{label} is negative"),
        ));
    }
    Ok((row as usize, col as usize))
}

fn eval_nonnegative(
    ctx: &WarpContext,
    value: &crate::ir::ScalarValue,
    label: &str,
) -> IResult<usize> {
    let value = ctx.eval_scalar_uniform(value, label, "divergent_operands")?;
    if value < 0 {
        return Err(InterpreterError::new(
            "tcgen05_tile_oob",
            format!("{label} must be non-negative"),
        ));
    }
    Ok(value as usize)
}

fn eval_smem_tile_against_shape(
    ctx: &WarpContext,
    tile: &SmemTile,
    tensor_shape: &[usize],
    label: &str,
) -> IResult<(Vec<usize>, Vec<usize>)> {
    let mut offsets = Vec::with_capacity(tile.prefix_indices.len() + 2);
    for index in &tile.prefix_indices {
        offsets.push(eval_nonnegative(ctx, index, label)?);
    }
    offsets.push(eval_nonnegative(ctx, &tile.row_offset, label)?);
    offsets.push(eval_nonnegative(ctx, &tile.col_offset, label)?);

    let mut shape = vec![1; tile.prefix_indices.len()];
    shape.push(tile.rows as usize);
    shape.push(tile.cols as usize);
    if offsets.len() != tensor_shape.len()
        || offsets.iter().zip(&shape).zip(tensor_shape).any(
            |((&offset, &extent), &tensor_extent)| {
                offset
                    .checked_add(extent)
                    .map_or(true, |end| end > tensor_extent)
            },
        )
    {
        return Err(InterpreterError::new(
            "tcgen05_tile_oob",
            format!("{label} exceeds its SMEM tensor"),
        ));
    }
    Ok((offsets, shape))
}

fn eval_smem_tile(
    ctx: &WarpContext,
    tile: &SmemTile,
    label: &str,
) -> IResult<(Vec<usize>, Vec<usize>)> {
    eval_smem_tile_against_shape(ctx, tile, &tile.tensor.shape, label)
}

fn eval_cp_smem_tile(
    ctx: &WarpContext,
    tile: &SmemTile,
    label: &str,
) -> IResult<(Vec<usize>, Vec<usize>)> {
    eval_smem_tile(ctx, tile, label)
}

fn smem_tile_region(
    ctx: &WarpContext,
    tile: &SmemTile,
    cta_id: usize,
    label: &str,
) -> IResult<Region> {
    let (offsets, shape) = eval_smem_tile(ctx, tile, label)?;
    region::tensor_region_from_uniform(&tile.tensor, cta_id, &offsets, &shape)
}

fn tmem_cells_region(
    ctx: &WarpContext,
    addr: &TmemAddr,
    cta_id: usize,
    label: &str,
    cells: impl IntoIterator<Item = (u32, u32)>,
) -> IResult<Region> {
    let (row0, col0) = eval_tmem_addr(ctx, addr, label)?;
    let mut absolute_cells = Vec::new();
    for (lane, col) in cells {
        let lane = row0.checked_add(lane as usize).ok_or_else(|| {
            InterpreterError::new("tcgen05_out_of_range", format!("{label} lane overflows"))
        })?;
        let col = col0.checked_add(col as usize).ok_or_else(|| {
            InterpreterError::new("tcgen05_out_of_range", format!("{label} column overflows"))
        })?;
        absolute_cells.push((lane, col));
    }
    region::tmem_region_from_cells(0, cta_id, absolute_cells)
}

fn mma_d_region(
    ctx: &WarpContext,
    dst: &TmemAddr,
    resolved: &ResolvedMma,
    cta_id: usize,
) -> IResult<Region> {
    let mut cells = Vec::new();
    for row in 0..resolved.d.logical.rows {
        for col in 0..resolved.d.logical.cols {
            let cell = resolved.d_cell(row, col).ok_or_else(|| {
                InterpreterError::new("tcgen05_mma_layout", "D coordinate is out of range")
            })?;
            cells.push((cell.lane, cell.col));
        }
    }
    tmem_cells_region(ctx, dst, cta_id, "tcgen05_mma D", cells)
}

fn mma_a_region(
    ctx: &WarpContext,
    addr: &TmemAddr,
    resolved: &ResolvedMma,
    cta_id: usize,
) -> IResult<Region> {
    let footprint = resolved
        .a_tmem
        .ok_or_else(|| InterpreterError::new("tcgen05_mma_layout", "TMEM A has no footprint"))?;
    let banks = match footprint.layout {
        TmemLayoutKind::Replica2 | TmemLayoutKind::BankBatched => 2,
        _ => 1,
    };
    let mut cells = Vec::new();
    for bank in 0..banks {
        for row in 0..footprint.logical.rows {
            for k in 0..footprint.logical.cols {
                let element = resolved.a_tmem_element(bank, row, k).ok_or_else(|| {
                    InterpreterError::new("tcgen05_mma_layout", "TMEM A coordinate is out of range")
                })?;
                cells.push((element.lane, element.col));
            }
        }
    }
    tmem_cells_region(ctx, addr, cta_id, "tcgen05_mma A", cells)
}

fn mma_scale_region(
    ctx: &WarpContext,
    addr: &TmemAddr,
    resolved: &ResolvedScaleFootprint,
    cta_id: usize,
    label: &str,
) -> IResult<Region> {
    let mut cells = Vec::new();
    for row in 0..resolved.footprint.logical.rows {
        for unique_scale in 0..resolved.sf_k {
            let logical_scale = unique_scale
                .checked_mul(match resolved.footprint.layout {
                    TmemLayoutKind::ScaleFactor { sf_reuse, .. } => sf_reuse,
                    _ => {
                        return Err(InterpreterError::new(
                            "tcgen05_mma_layout",
                            "scale operand has a non-scale layout",
                        ))
                    }
                })
                .ok_or_else(|| {
                    InterpreterError::new(
                        "tcgen05_mma_layout",
                        "scale logical coordinate overflows",
                    )
                })?;
            for byte in resolved.physical_bytes(row, logical_scale).ok_or_else(|| {
                InterpreterError::new("tcgen05_mma_layout", "scale coordinate is out of range")
            })? {
                cells.push((byte.lane, byte.col));
            }
        }
    }
    tmem_cells_region(ctx, addr, cta_id, label, cells)
}

fn tcgen_cta_ids(ctx: &WarpContext, cta_group: u8, label: &str) -> IResult<Vec<usize>> {
    if cta_group == 1 {
        return Ok(vec![ctx.stream.cta_id]);
    }
    let local = ctx.lanes[0].ctaid_in_cluster;
    let peer_local = peer_ctaid_in_cluster(
        ctx,
        local,
        &format!("{label}_peer"),
        &format!("{label} peer out of range"),
    )?;
    let peer = ctx.global_cta_id(peer_local);
    // A cta_group::2 instruction always addresses the same rank-ordered CTA
    // pair.  Execution is controlled solely by the IR: if both CTAs reach the
    // statement, both issue it; if the kernel wants one leader, it must carry
    // an explicit ctaid_in_cluster branch.  Never make the odd CTA an implicit
    // no-op here.
    Ok(if local & 1 == 0 {
        vec![ctx.stream.cta_id, peer]
    } else {
        vec![peer, ctx.stream.cta_id]
    })
}

fn eval_accum(ctx: &WarpContext, accum: &crate::ir::ScalarValue) -> IResult<bool> {
    let value = if let Some(value) =
        scalar_eval::eval_scalar_known_uniform(accum, &ctx.lanes, &ctx.state.values.scalars)?
    {
        value
    } else {
        scalar_eval::eval_scalar_at(accum, &ctx.lanes[0], &ctx.state.values.scalars)?
    };
    Ok(value != 0)
}

fn read_smem_matrix(
    ctx: &WarpContext,
    tile: &SmemTile,
    format: MmaElemFormat,
    transpose: bool,
    cta_id: usize,
) -> IResult<Array2<f32>> {
    if format == MmaElemFormat::F4E2M1 && transpose {
        return Err(InterpreterError::new(
            "tcgen05_mma_unsupported",
            "transposed packed-f4 SMEM operands are unsupported",
        ));
    }
    let (offsets, shape) = eval_smem_tile(ctx, tile, "tcgen05_mma operand")?;
    let mut values = Vec::new();
    let pool = ctx.state.values.smem.pool_for(cta_id)?;
    if format == MmaElemFormat::F4E2M1 {
        pool.append_f32_block_fp4(&tile.tensor, &offsets, &shape, &mut values)?;
    } else {
        pool.append_f32_block(&tile.tensor, &offsets, &shape, &mut values)?;
    }
    let rows = tile.rows as usize;
    let cols = if format == MmaElemFormat::F4E2M1 {
        tile.cols as usize * 2
    } else {
        tile.cols as usize
    };
    let matrix = Array2::from_shape_vec((rows, cols), values).map_err(|_| {
        InterpreterError::new(
            "tcgen05_mma_shape",
            "SMEM operand values do not match its explicit tile",
        )
    })?;
    Ok(if transpose {
        matrix.t().to_owned()
    } else {
        matrix
    })
}

fn read_tmem_matrix_bank(
    ctx: &WarpContext,
    addr: &TmemAddr,
    resolved: &ResolvedMma,
    cta_id: usize,
    bank: u32,
) -> IResult<Array2<f32>> {
    resolved.a_tmem.as_ref().ok_or_else(|| {
        InterpreterError::new("tcgen05_mma_operand", "TMEM A has no resolved footprint")
    })?;
    let (row0, col0) = eval_tmem_addr(ctx, addr, "tcgen05_mma A")?;
    let rows = resolved.per_cta.m as usize;
    let cols = resolved.k as usize;
    let scratch = ctx.state.values.tmem.scratchpad_for(cta_id)?;
    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for k in 0..cols {
            let element = resolved
                .a_tmem_element(bank, row as u32, k as u32)
                .ok_or_else(|| {
                    InterpreterError::new(
                        "tcgen05_mma_operand",
                        "TMEM A element is outside its resolved layout",
                    )
                })?;
            let lane = row0 + element.lane as usize;
            let col = col0 + element.col as usize;
            let value = match resolved.format {
                MmaElemFormat::F16 | MmaElemFormat::BF16 => {
                    let dtype = if resolved.format == MmaElemFormat::F16 {
                        DType::F16
                    } else {
                        DType::Bf16
                    };
                    let [(lo, hi)] = scratch
                        .read_packed_half_cells(dtype, &[lane], &[col])?
                        .try_into()
                        .map_err(|_| {
                            InterpreterError::new(
                                "tcgen05_mma_operand",
                                "packed half read returned the wrong element count",
                            )
                        })?;
                    match element.bit_offset {
                        0 => lo,
                        16 => hi,
                        _ => {
                            return Err(InterpreterError::new(
                                "tcgen05_mma_layout",
                                "half TMEM A has a non-half-aligned bit offset",
                            ))
                        }
                    }
                }
                MmaElemFormat::F8E4M3 => {
                    if element.bit_offset % 8 != 0 {
                        return Err(InterpreterError::new(
                            "tcgen05_mma_layout",
                            "F8 TMEM A has a non-byte-aligned bit offset",
                        ));
                    }
                    decode_e4m3(scratch.read_cell_byte(lane, col, element.bit_offset / 8)?)
                }
                MmaElemFormat::F4E2M1 => {
                    if element.bit_offset % 4 != 0 {
                        return Err(InterpreterError::new(
                            "tcgen05_mma_layout",
                            "F4 TMEM A has a non-nibble-aligned bit offset",
                        ));
                    }
                    let byte = scratch.read_cell_byte(lane, col, element.bit_offset / 8)?;
                    let nibble = if element.bit_offset % 8 == 0 {
                        byte >> 4
                    } else {
                        byte & 0x0f
                    };
                    decode_e2m1(nibble)
                }
            };
            out.push(value);
        }
    }
    Array2::from_shape_vec((rows, cols), out).map_err(|_| {
        InterpreterError::new(
            "tcgen05_mma_shape",
            "TMEM A values do not match its resolved footprint",
        )
    })
}

fn read_local_a(
    ctx: &WarpContext,
    operand: &MmaAOperand,
    format: MmaElemFormat,
    transpose: bool,
    resolved: &ResolvedMma,
    cta_id: usize,
) -> IResult<Vec<Array2<f32>>> {
    let banks = match operand {
        MmaAOperand::Smem(tile) => vec![read_smem_matrix(ctx, tile, format, transpose, cta_id)?],
        MmaAOperand::Tmem { addr, .. } => {
            let count = if resolved
                .a_tmem
                .is_some_and(|footprint| footprint.layout == TmemLayoutKind::BankBatched)
            {
                2
            } else {
                1
            };
            (0..count)
                .map(|bank| read_tmem_matrix_bank(ctx, addr, resolved, cta_id, bank))
                .collect::<IResult<Vec<_>>>()?
        }
    };
    for bank in &banks {
        if bank.dim() != (resolved.per_cta.m as usize, resolved.k as usize) {
            return Err(InterpreterError::new(
                "tcgen05_mma_shape",
                "A operand does not match the resolver's per-CTA MxK",
            ));
        }
    }
    Ok(banks)
}

fn read_global_b(
    ctx: &WarpContext,
    tile: &SmemTile,
    format: MmaElemFormat,
    transpose: bool,
    resolved: &ResolvedMma,
    cta_ids: &[usize],
) -> IResult<Array2<f32>> {
    let matrices: Vec<Array2<f32>> = cta_ids
        .iter()
        .map(|&cta| read_smem_matrix(ctx, tile, format, transpose, cta))
        .collect::<IResult<_>>()?;
    let matrix = if matrices.len() == 1 {
        matrices.into_iter().next().unwrap()
    } else {
        let views: Vec<_> = matrices.iter().map(|matrix| matrix.view()).collect();
        ndarray::concatenate(ndarray::Axis(0), &views).map_err(|_| {
            InterpreterError::new("tcgen05_mma_shape", "B operand CTA concatenation failed")
        })?
    };
    if matrix.dim() != (resolved.n as usize, resolved.k as usize) {
        return Err(InterpreterError::new(
            "tcgen05_mma_shape",
            "B operand does not match the resolver's NxK",
        ));
    }
    Ok(matrix)
}

fn decode_e8m0_fnu(byte: u8) -> f32 {
    if byte == u8::MAX {
        f32::NAN
    } else {
        ((i32::from(byte) - 127) as f32).exp2()
    }
}

fn read_scale_values(
    ctx: &WarpContext,
    addr: &TmemAddr,
    resolved: &ResolvedScaleFootprint,
    scale_format: ScaleFormat,
    cta_id: usize,
) -> IResult<Vec<f32>> {
    let (row0, col0) = eval_tmem_addr(ctx, addr, "tcgen05_mma scale")?;
    let rows = resolved.footprint.logical.rows as usize;
    let sf_k = resolved.sf_k as usize;
    let sf_reuse = match resolved.footprint.layout {
        TmemLayoutKind::ScaleFactor { sf_reuse, .. } => sf_reuse,
        _ => {
            return Err(InterpreterError::new(
                "tcgen05_mma_layout",
                "scale operand has a non-scale layout",
            ))
        }
    };
    let scratch = ctx.state.values.tmem.scratchpad_for(cta_id)?;
    let mut out = Vec::with_capacity(rows * sf_k);
    for row in 0..rows {
        for unique_scale in 0..sf_k {
            let logical_scale = (unique_scale as u32).checked_mul(sf_reuse).ok_or_else(|| {
                InterpreterError::new("tcgen05_mma_layout", "scale logical coordinate overflows")
            })?;
            let byte_coord = resolved
                .physical_byte(row as u32, logical_scale)
                .ok_or_else(|| {
                    InterpreterError::new("tcgen05_mma_scale", "scale coordinate is out of bounds")
                })?;
            let byte = scratch.read_cell_byte(
                row0 + byte_coord.lane as usize,
                col0 + byte_coord.col as usize,
                byte_coord.subbyte,
            )?;
            out.push(match scale_format {
                ScaleFormat::E8M0FNU => decode_e8m0_fnu(byte),
                ScaleFormat::E4M3FN => decode_e4m3(byte),
            });
        }
    }
    Ok(out)
}

fn apply_resolved_scales(
    matrix: &mut Array2<f32>,
    scales: &[f32],
    resolved_scale: &ResolvedScaleFootprint,
    resolved_mma: &ResolvedMma,
) -> IResult<()> {
    let k = resolved_mma.k as usize;
    let mma_k = resolved_mma.mma_k as usize;
    let sf_per_mma = match resolved_scale.footprint.layout {
        TmemLayoutKind::ScaleFactor { sf_per_mma, .. } => sf_per_mma as usize,
        _ => unreachable!(),
    };
    let sf_reuse = match resolved_scale.footprint.layout {
        TmemLayoutKind::ScaleFactor { sf_reuse, .. } => sf_reuse as usize,
        _ => unreachable!(),
    };
    let scale_width = mma_k / sf_per_mma;
    if scale_width == 0 {
        return Err(InterpreterError::new(
            "tcgen05_mma_scale",
            "scale width resolved to zero",
        ));
    }
    let sf_k = resolved_scale.sf_k as usize;
    let expected = (resolved_scale.footprint.logical.rows as usize)
        .checked_mul(sf_k)
        .ok_or_else(|| InterpreterError::new("tcgen05_mma_scale", "scale size overflows"))?;
    if matrix.nrows() != resolved_scale.operand_rows as usize
        || scales.len() != expected
        || matrix.ncols() != k
    {
        return Err(InterpreterError::new(
            "tcgen05_mma_scale",
            "scale values do not match the resolved operand shape",
        ));
    }
    for row in 0..matrix.nrows() {
        for col in 0..k {
            let mma_iter = col / mma_k;
            let group = mma_iter / sf_reuse;
            let within_mma = (col % mma_k) / scale_width;
            let scale_index = group
                .checked_mul(sf_per_mma)
                .and_then(|value| value.checked_add(within_mma))
                .ok_or_else(|| {
                    InterpreterError::new("tcgen05_mma_scale", "scale index overflows")
                })?;
            let scale = scales.get(row * sf_k + scale_index).ok_or_else(|| {
                InterpreterError::new("tcgen05_mma_scale", "resolved scale index is out of bounds")
            })?;
            matrix[[row, col]] *= scale;
        }
    }
    Ok(())
}

fn local_product(a_banks: &[Array2<f32>], b: &Array2<f32>) -> IResult<Array2<f32>> {
    if a_banks.len() == 1 {
        return Ok(matmul_f32(&a_banks[0], b));
    }
    if a_banks.len() != 2 || b.nrows() % 2 != 0 {
        return Err(InterpreterError::new(
            "tcgen05_mma_shape",
            "bank-batched A requires two banks and an even N",
        ));
    }
    let half = b.nrows() / 2;
    let mut out = Array2::<f32>::zeros((a_banks[0].nrows(), b.nrows()));
    for (bank, a) in a_banks.iter().enumerate() {
        let b_half = b
            .slice(ndarray::s![bank * half..(bank + 1) * half, ..])
            .to_owned();
        let product = matmul_f32(a, &b_half);
        out.slice_mut(ndarray::s![.., bank * half..(bank + 1) * half])
            .assign(&product);
    }
    Ok(out)
}

fn accumulate_local_product(
    ctx: &mut WarpContext,
    dst: &TmemAddr,
    resolved: &ResolvedMma,
    cta_id: usize,
    product: &Array2<f32>,
    accum: bool,
) -> IResult<()> {
    let (row, col) = eval_tmem_addr(ctx, dst, "tcgen05_mma dst")?;
    let scratch = ctx
        .state
        .values
        .tmem
        .by_cta
        .get_mut(&cta_id)
        .ok_or_else(|| {
            InterpreterError::new(
                "missing_tmem_scratchpad",
                "tcgen05_mma writes a missing TMEM scratchpad",
            )
        })?;
    let m = resolved.per_cta.m as usize;
    let n = resolved.n as usize;
    if product.dim() != (m, n) {
        return Err(InterpreterError::new(
            "tcgen05_mma_shape",
            "MMA product does not match the resolver's per-CTA output shape",
        ));
    }
    let mut lanes = Vec::with_capacity(m * n);
    let mut cols = Vec::with_capacity(m * n);
    let mut values = Vec::with_capacity(m * n);
    for logical_row in 0..m {
        for logical_col in 0..n {
            let cell = resolved
                .d_cell(logical_row as u32, logical_col as u32)
                .ok_or_else(|| {
                    InterpreterError::new(
                        "tcgen05_mma_layout",
                        "D coordinate is outside its resolved layout",
                    )
                })?;
            lanes.push(row + cell.lane as usize);
            cols.push(col + cell.col as usize);
            values.push(product[[logical_row, logical_col]]);
        }
    }
    scratch.accumulate_f32_cells(&lanes, &cols, &values, accum)?;
    Ok(())
}

fn trace_mma(
    ctx: &mut WarpContext,
    dst: &TmemAddr,
    a: &MmaAOperand,
    b: &SmemTile,
    block_scale: Option<&BlockScaleSpec>,
    resolved: &ResolvedMma,
    cta_ids: &[usize],
    accum: bool,
) -> IResult<()> {
    let scope = ctx.access_scope();
    match a {
        MmaAOperand::Smem(tile) => {
            for &cta in cta_ids {
                let region = smem_tile_region(ctx, tile, cta, "tcgen05_mma A")?;
                ctx.emit(TraceEventKind::Read {
                    region,
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Tcgen05Mma),
                    scope: scope.clone(),
                })?;
            }
        }
        MmaAOperand::Tmem { addr, .. } => {
            for &cta in cta_ids {
                let region = mma_a_region(ctx, addr, resolved, cta)?;
                check_tmem_region_allocated(ctx, &region, "tcgen05_mma A")?;
                ctx.emit(TraceEventKind::Read {
                    region,
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Mma),
                    scope: scope.clone(),
                })?;
            }
        }
    }
    for &cta in cta_ids {
        let region = smem_tile_region(ctx, b, cta, "tcgen05_mma B")?;
        ctx.emit(TraceEventKind::Read {
            region,
            proxy: MemoryProxy::Async,
            access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Tcgen05Mma),
            scope: scope.clone(),
        })?;
    }
    if let Some(spec) = block_scale {
        for (addr, footprint, label) in [
            (&spec.sfa, resolved.sfa.as_ref(), "tcgen05_mma SFA"),
            (&spec.sfb, resolved.sfb.as_ref(), "tcgen05_mma SFB"),
        ] {
            let footprint = footprint.ok_or_else(|| {
                InterpreterError::new("tcgen05_mma_scale", "scale operand has no footprint")
            })?;
            for &cta in cta_ids {
                let region = mma_scale_region(ctx, addr, footprint, cta, label)?;
                check_tmem_region_allocated(ctx, &region, label)?;
                ctx.emit(TraceEventKind::Read {
                    region,
                    proxy: MemoryProxy::Async,
                    access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Mma),
                    scope: scope.clone(),
                })?;
            }
        }
    }
    for &cta in cta_ids {
        let region = mma_d_region(ctx, dst, resolved, cta)?;
        check_tmem_region_allocated(ctx, &region, "tcgen05_mma D")?;
        if accum {
            ctx.emit(TraceEventKind::Read {
                region: region.clone(),
                proxy: MemoryProxy::Async,
                access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Mma),
                scope: scope.clone(),
            })?;
        }
        ctx.emit(TraceEventKind::Write {
            region,
            proxy: MemoryProxy::Async,
            access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Mma),
            scope: scope.clone(),
        })?;
    }
    Ok(())
}

fn execute_mma<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (dst, a, b, mma_m, mma_n, format, block_scale, accum, trans_a, trans_b, ws, cta_group) =
        match stmt {
            Stmt::Tcgen05Mma {
                dst,
                a,
                b,
                mma_m,
                mma_n,
                format,
                block_scale,
                accum,
                trans_a,
                trans_b,
                ws,
                cta_group,
            } => (
                dst,
                a,
                b,
                *mma_m,
                *mma_n,
                *format,
                block_scale.as_ref(),
                accum,
                *trans_a,
                *trans_b,
                *ws,
                *cta_group,
            ),
            _ => unreachable!(),
        };
    let resolved = resolve_tcgen05_mma(
        dst,
        a,
        b,
        mma_m,
        mma_n,
        format,
        block_scale,
        trans_a,
        trans_b,
        ws,
        cta_group,
    )
    .map_err(|error| layout_error("tcgen05_mma_layout", error))?;
    ctx.check_single_thread_issue("tcgen05_mma_mask", "tcgen05_mma")?;
    let cta_ids = tcgen_cta_ids(ctx, cta_group, "tcgen05_mma")?;
    let accum = eval_accum(ctx, accum)?;
    if ctx.trace_mode() {
        trace_mma(ctx, dst, a, b, block_scale, &resolved, &cta_ids, accum)?;
        return Ok(StepStatus::advance());
    }

    let b_base = read_global_b(ctx, b, format, trans_b, &resolved, &cta_ids)?;
    for &cta_id in &cta_ids {
        let mut a_banks = read_local_a(ctx, a, format, trans_a, &resolved, cta_id)?;
        let mut b_values = b_base.clone();
        if let Some(spec) = block_scale {
            let sfa = resolved.sfa.as_ref().ok_or_else(|| {
                InterpreterError::new("tcgen05_mma_scale", "SFA has no resolved footprint")
            })?;
            let sfb = resolved.sfb.as_ref().ok_or_else(|| {
                InterpreterError::new("tcgen05_mma_scale", "SFB has no resolved footprint")
            })?;
            let a_scales = read_scale_values(ctx, &spec.sfa, sfa, spec.scale_format, cta_id)?;
            let b_scales = read_scale_values(ctx, &spec.sfb, sfb, spec.scale_format, cta_id)?;
            for bank in &mut a_banks {
                apply_resolved_scales(bank, &a_scales, sfa, &resolved)?;
            }
            apply_resolved_scales(&mut b_values, &b_scales, sfb, &resolved)?;
        }
        let product = local_product(&a_banks, &b_values)?;
        accumulate_local_product(ctx, dst, &resolved, cta_id, &product, accum)?;
    }
    Ok(StepStatus::advance())
}

fn matmul_f32(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.nrows();
    let a = a.as_standard_layout();
    let b = b.as_standard_layout();
    let mut c = vec![0.0f32; m * n];
    super::super::blas::sgemm_nt(
        m,
        n,
        k,
        a.as_slice().unwrap(),
        b.as_slice().unwrap(),
        &mut c,
        0.0,
    );
    Array2::from_shape_vec((m, n), c).unwrap()
}

fn cp_target_region(
    ctx: &WarpContext,
    dst: &TmemAddr,
    resolved: &ResolvedCp,
    cta_id: usize,
) -> IResult<Region> {
    let (row0, col0) = eval_tmem_addr(ctx, dst, "tcgen05_cp dst")?;
    let mut rects = Vec::new();
    for row in 0..resolved.source.rows {
        for col in 0..resolved.source_row_cells {
            let cells = resolved.target_cells(row, col).ok_or_else(|| {
                InterpreterError::new("tcgen05_cp_layout", "CP source coordinate is out of range")
            })?;
            for cell in cells {
                rects.push((row0 + cell.lane as usize, 1, col0 + cell.col as usize, 1));
            }
        }
    }
    region::tmem_region_from_rects(0, cta_id, rects)
}

fn execute_cp<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (dst, src, shape, multicast, cta_group) = match stmt {
        Stmt::Tcgen05Cp {
            dst,
            src,
            shape,
            multicast,
            cta_group,
        } => (dst, src, *shape, *multicast, *cta_group),
        _ => unreachable!(),
    };
    let resolved = resolve_tcgen05_cp(dst, src, shape, multicast, cta_group)
        .map_err(|error| layout_error("tcgen05_cp_layout", error))?;
    ctx.check_single_thread_issue("tcgen05_cp_mask", "tcgen05_cp")?;
    let cta_ids = tcgen_cta_ids(ctx, cta_group, "tcgen05_cp")?;
    let (offsets, tile_shape) = eval_cp_smem_tile(ctx, src, "tcgen05_cp src")?;
    let (row0, col0) = eval_tmem_addr(ctx, dst, "tcgen05_cp dst")?;
    let row_bytes = resolved.source_row_cells as usize * 4;
    let scope = ctx.access_scope();
    for &cta_id in &cta_ids {
        let dst_region = cp_target_region(ctx, dst, &resolved, cta_id)?;
        check_tmem_region_allocated(ctx, &dst_region, "tcgen05_cp")?;
        if ctx.trace_mode() {
            let src_region =
                region::tensor_region_from_uniform(&src.tensor, cta_id, &offsets, &tile_shape)?;
            ctx.emit(TraceEventKind::Read {
                region: src_region,
                proxy: MemoryProxy::Async,
                access_kind: MemoryAccessKind::Tensor(TensorAccessKind::Tcgen05Cp),
                scope: scope.clone(),
            })?;
            ctx.emit(TraceEventKind::Write {
                region: dst_region,
                proxy: MemoryProxy::Async,
                access_kind: MemoryAccessKind::Tmem(TmemAsyncKind::Cp),
                scope: scope.clone(),
            })?;
            continue;
        }
        let smem = ctx.state.values.smem.pool_for(cta_id)?;
        let raw = smem.read_block_bytes(&src.tensor, &offsets, &tile_shape)?;
        if raw.len() != resolved.source.rows as usize * row_bytes {
            return Err(InterpreterError::new(
                "tcgen05_cp_shape",
                "CP source byte count does not match the resolved shape",
            ));
        }
        let scratch = ctx
            .state
            .values
            .tmem
            .by_cta
            .get_mut(&cta_id)
            .ok_or_else(|| {
                InterpreterError::new(
                    "missing_tmem_scratchpad",
                    "tcgen05_cp writes a missing TMEM scratchpad",
                )
            })?;
        for row in 0..resolved.source.rows {
            for col in 0..resolved.source_row_cells {
                let begin = row as usize * row_bytes + col as usize * 4;
                let bits = u32::from_le_bytes(raw[begin..begin + 4].try_into().unwrap());
                for cell in resolved.target_cells(row, col).ok_or_else(|| {
                    InterpreterError::new(
                        "tcgen05_cp_layout",
                        "CP source coordinate is out of range",
                    )
                })? {
                    scratch.write_raw_cell(
                        row0 + cell.lane as usize,
                        col0 + cell.col as usize,
                        bits,
                    )?;
                }
            }
        }
    }
    Ok(StepStatus::advance())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_fnu_decodes_finite_range_and_nan_slot() {
        assert_eq!(decode_e8m0_fnu(0), 2.0f32.powi(-127));
        assert_eq!(decode_e8m0_fnu(127), 1.0);
        assert_eq!(decode_e8m0_fnu(254), 2.0f32.powi(127));
        assert!(decode_e8m0_fnu(255).is_nan());
    }
}
