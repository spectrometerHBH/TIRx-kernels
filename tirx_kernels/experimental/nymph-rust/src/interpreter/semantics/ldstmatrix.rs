//! PTX ldmatrix/stmatrix m8n8.b16 semantics.
//!
//! The SMEM operand is a per-thread row-address slice containing eight b16
//! elements. For each warp, `.x1/.x2/.x4` use address lanes 0..7, 0..15, or
//! 0..31 as the row starts for one, two, or four matrices respectively.

use super::super::diagnostics::{IResult, InterpreterError};
use super::super::outcomes::StepStatus;
use super::super::protocol::{MemoryAccessKind, MemoryProxy, TensorAccessKind, TraceEventKind};
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::slice_indexing::{shared_flat_indices, ResolvedSlice};
use super::super::values::arrays::ValueArray2;
use super::super::values::ldstmatrix::{
    check_num, element_coord, pack_b16x2, row_address_lane, unpack_b16x2,
};
use super::super::warp_context::WarpContext;
use crate::ir::{DType, MatrixDType, MatrixShape, Stmt};
use ndarray::Array2;
use std::collections::HashSet;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::LdMatrix, execute_ldmatrix);
    reg.register(StmtKind::StMatrix, execute_stmatrix);
}

fn execute_ldmatrix<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (dst, src, shape, num, trans, dtype) = match stmt {
        Stmt::LdMatrix {
            dst,
            src,
            shape,
            num,
            trans,
            dtype,
        } => (dst, src, *shape, *num as usize, *trans, *dtype),
        _ => unreachable!(),
    };
    check_matrix_shape(shape, dtype, num, "ldmatrix")?;
    ctx.check_full_warp(
        "ldmatrix_mask",
        "ldmatrix must be issued by one or more full warps",
    )?;
    let dst_r = ctx.eval_slice(dst)?;
    let src_r = ctx.eval_slice(src)?;
    check_reg_fragment(&dst_r, num, "ldmatrix dst")?;
    let src_idx = check_row_slice(&src_r, "ldmatrix src")?;

    if ctx.trace_mode() {
        emit_matrix_tensor_read(ctx, &src_r, num, TensorAccessKind::LdMatrix)?;
        ctx.emit_tensor_write_with_kind(&dst_r, MemoryProxy::Generic, TensorAccessKind::LdMatrix)?;
        return Ok(StepStatus::advance());
    }

    let row_lanes = lane_row_map(ctx)?;
    let mut source_flats = Vec::with_capacity(ctx.lanes.len() * num * 2);
    for (ai, thread) in ctx.lanes.iter().enumerate() {
        let lane_rows = &row_lanes[warp_index(ctx, thread.warp_id)?];
        let lane = thread.lane_id;
        for matrix_id in 0..num {
            for half in 0..2 {
                let (row, col) = element_coord(lane, half, trans);
                let addr_ai = lane_rows[row_address_lane(matrix_id, row)];
                source_flats.push(src_idx[[addr_ai, col]]);
            }
        }
        debug_assert_eq!(source_flats.len(), (ai + 1) * num * 2);
    }
    let bits = ctx
        .state
        .values
        .smem
        .pool_for(ctx.stream.cta_id)?
        .read_u16_bits_indices(&src_r.tensor, &source_flats)?;

    let mut words = Array2::<u32>::zeros((ctx.lanes.len(), num));
    for ai in 0..ctx.lanes.len() {
        for matrix_id in 0..num {
            let base = (ai * num + matrix_id) * 2;
            words[[ai, matrix_id]] = pack_b16x2(bits[base], bits[base + 1]);
        }
    }
    write_reg_words(ctx, &dst_r, words)?;
    Ok(StepStatus::advance())
}

fn execute_stmatrix<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (dst, src, shape, num, trans, dtype) = match stmt {
        Stmt::StMatrix {
            dst,
            src,
            shape,
            num,
            trans,
            dtype,
        } => (dst, src, *shape, *num as usize, *trans, *dtype),
        _ => unreachable!(),
    };
    check_matrix_shape(shape, dtype, num, "stmatrix")?;
    ctx.check_full_warp(
        "stmatrix_mask",
        "stmatrix must be issued by one or more full warps",
    )?;
    let dst_r = ctx.eval_slice(dst)?;
    let src_r = ctx.eval_slice(src)?;
    let dst_idx = check_row_slice(&dst_r, "stmatrix dst")?;
    check_reg_fragment(&src_r, num, "stmatrix src")?;
    let row_lanes = lane_row_map(ctx)?;
    let dst_flats = matrix_flat_indices(ctx, &row_lanes, &dst_idx, num, trans)?;
    check_unique(
        &dst_flats,
        "overlapping_tensor_write",
        "stmatrix SMEM write overlaps",
    )?;

    if ctx.trace_mode() {
        ctx.emit_tensor_read_with_kind(&src_r, MemoryProxy::Generic, TensorAccessKind::StMatrix)?;
        emit_matrix_tensor_write(ctx, &dst_r, num, TensorAccessKind::StMatrix)?;
        let pool = ctx
            .state
            .values
            .smem
            .pool_for_mut(ctx.stream.cta_id, ctx.kernel.smem_size_bytes);
        pool.invalidate_indices(&dst_r.tensor, &dst_flats)?;
        return Ok(StepStatus::advance());
    }

    let words = read_reg_words(ctx, &src_r, num)?;
    let mut out_bits = Vec::with_capacity(dst_flats.len());
    for ai in 0..ctx.lanes.len() {
        for matrix_id in 0..num {
            let halves = unpack_b16x2(words[[ai, matrix_id]]);
            for half in halves {
                out_bits.push(half);
            }
        }
    }
    let pool = ctx
        .state
        .values
        .smem
        .pool_for_mut(ctx.stream.cta_id, ctx.kernel.smem_size_bytes);
    pool.write_u16_bits_indices(&dst_r.tensor, &dst_flats, &out_bits)?;
    Ok(StepStatus::advance())
}

fn check_matrix_shape(
    shape: MatrixShape,
    dtype: MatrixDType,
    num: usize,
    label: &str,
) -> IResult<()> {
    if shape != MatrixShape::M8N8 || dtype != MatrixDType::B16 {
        return Err(InterpreterError::new(
            format!("{label}_shape"),
            format!("{label} supports only m8n8.x{{1,2,4}}.b16"),
        ));
    }
    check_num(num, label)
}

fn shape_numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn check_row_slice(resolved: &ResolvedSlice, label: &str) -> IResult<Array2<usize>> {
    if shape_numel(&resolved.shape) != 8 {
        return Err(InterpreterError::new(
            "ldstmatrix_shape",
            format!("{label} must contain exactly one row of eight b16 elements"),
        ));
    }
    shared_flat_indices(resolved, &resolved.tensor.shape).map(|(idx, _)| idx)
}

fn check_reg_fragment(resolved: &ResolvedSlice, num: usize, label: &str) -> IResult<()> {
    let b16 = matches!(resolved.tensor.dtype, DType::F16 | DType::Bf16);
    let want = if b16 { 2 * num } else { num };
    if shape_numel(&resolved.shape) != want {
        return Err(InterpreterError::new(
            "ldstmatrix_shape",
            format!("{label} must contain exactly {num} b32 registers (2*num b16 elements)"),
        ));
    }
    if !b16 && !matches!(resolved.tensor.dtype, DType::U32 | DType::I32) {
        return Err(InterpreterError::new(
            "ldstmatrix_dtype",
            format!("{label} must have dtype u32/i32 or a b16 fragment"),
        ));
    }
    Ok(())
}

fn warp_index(ctx: &WarpContext, warp_id: usize) -> IResult<usize> {
    // Every lane of the mask belongs to the stream's one warp, so the only
    // valid index is 0.
    if ctx.lanes[0].warp_id == warp_id {
        Ok(0)
    } else {
        Err(InterpreterError::new(
            "ldstmatrix_mask",
            "missing warp in the lane mask",
        ))
    }
}

fn lane_row_map(ctx: &WarpContext) -> IResult<Vec<[usize; 32]>> {
    let ids = [ctx.lanes[0].warp_id];
    let mut rows = vec![[usize::MAX; 32]; ids.len()];
    for (ai, thread) in ctx.lanes.iter().enumerate() {
        let wi = ids
            .iter()
            .position(|&id| id == thread.warp_id)
            .expect("warp id collected from the lane mask");
        rows[wi][thread.lane_id] = ai;
    }
    for lane_rows in &rows {
        if lane_rows.iter().any(|&x| x == usize::MAX) {
            return Err(InterpreterError::new(
                "ldstmatrix_mask",
                "ldmatrix/stmatrix must be issued by full warps",
            ));
        }
    }
    Ok(rows)
}

fn matrix_flat_indices(
    ctx: &WarpContext,
    row_lanes: &[[usize; 32]],
    row_idx: &Array2<usize>,
    num: usize,
    trans: bool,
) -> IResult<Vec<usize>> {
    let mut flats = Vec::with_capacity(ctx.lanes.len() * num * 2);
    for thread in &ctx.lanes {
        let lane_rows = &row_lanes[warp_index(ctx, thread.warp_id)?];
        let lane = thread.lane_id;
        for matrix_id in 0..num {
            for half in 0..2 {
                let (row, col) = element_coord(lane, half, trans);
                let addr_ai = lane_rows[row_address_lane(matrix_id, row)];
                flats.push(row_idx[[addr_ai, col]]);
            }
        }
    }
    Ok(flats)
}

fn check_unique(values: &[usize], code: &str, message: &str) -> IResult<()> {
    let mut seen = HashSet::with_capacity(values.len());
    for &value in values {
        if !seen.insert(value) {
            return Err(InterpreterError::new(code, message));
        }
    }
    Ok(())
}

fn read_reg_words(ctx: &WarpContext, resolved: &ResolvedSlice, num: usize) -> IResult<Array2<u32>> {
    let values = ctx.registers_read(resolved)?;
    match values {
        ValueArray2::U32(a) if a.dim() == (ctx.lanes.len(), num) => Ok(a),
        ValueArray2::I32(a) if a.dim() == (ctx.lanes.len(), num) => Ok(a.mapv(|x| x as u32)),
        // A b16 fragment: each consecutive pair IS one b32 word (the packed
        // register file the f32->b16x2 pair cvt produces on silicon — packing
        // is a register-file view, not an instruction).
        ValueArray2::F16(a) | ValueArray2::Bf16(a) if a.dim() == (ctx.lanes.len(), 2 * num) => {
            let dtype = resolved.tensor.dtype;
            Ok(Array2::from_shape_fn((ctx.lanes.len(), num), |(t, w)| {
                let lo = encode_b16(dtype, a[[t, 2 * w]]);
                let hi = encode_b16(dtype, a[[t, 2 * w + 1]]);
                u32::from(lo) | (u32::from(hi) << 16)
            }))
        }
        _ => Err(InterpreterError::new(
            "ldstmatrix_shape",
            "register fragment must be num u32/i32 words or 2*num b16 elements",
        )),
    }
}

fn encode_b16(dtype: DType, value: f32) -> u16 {
    match dtype {
        DType::F16 => half::f16::from_f32(value).to_bits(),
        DType::Bf16 => {
            (crate::interpreter::values::dtypes::round_bf16_scalar(value).to_bits() >> 16) as u16
        }
        _ => unreachable!("checked b16 fragment dtype"),
    }
}

fn write_reg_words(
    ctx: &mut WarpContext,
    resolved: &ResolvedSlice,
    words: Array2<u32>,
) -> IResult<()> {
    let values = match resolved.tensor.dtype {
        DType::U32 => ValueArray2::U32(words),
        DType::I32 => ValueArray2::I32(words.mapv(|x| x as i32)),
        // A b16 fragment dst: decode each word's halves into the pair slots —
        // consecutive b16 elements ARE the b32 word (the inverse of the pack in
        // read_reg_words; ldmatrix moves bits, not values).
        DType::F16 | DType::Bf16 => {
            let dtype = resolved.tensor.dtype;
            let (lanes, num) = words.dim();
            let out = Array2::from_shape_fn((lanes, 2 * num), |(t, e)| {
                decode_b16(
                    dtype,
                    ((words[[t, e / 2]] >> (16 * (e % 2))) & 0xffff) as u16,
                )
            });
            ValueArray2::from_f32_compute(out, dtype)
        }
        _ => {
            return Err(InterpreterError::new(
                "ldstmatrix_dtype",
                "register fragment must have dtype u32/i32 or f16/bf16",
            ))
        }
    };
    ctx.registers_write(resolved, &values)
}

fn decode_b16(dtype: DType, bits: u16) -> f32 {
    match dtype {
        DType::F16 => half::f16::from_bits(bits).to_f32(),
        DType::Bf16 => f32::from_bits(u32::from(bits) << 16),
        _ => unreachable!("checked b16 fragment dtype"),
    }
}

fn emit_matrix_tensor_read(
    ctx: &mut WarpContext,
    resolved: &ResolvedSlice,
    num: usize,
    access_kind: TensorAccessKind,
) -> IResult<()> {
    let offsets = selected_row_offsets(ctx, resolved, num)?;
    let region = ctx.tensor_region_with_offsets(resolved, &offsets)?;
    let scope = ctx.access_scope();
    ctx.emit(TraceEventKind::Read {
        region,
        proxy: MemoryProxy::Generic,
        access_kind: MemoryAccessKind::Tensor(access_kind),
        scope,
    })
}

fn emit_matrix_tensor_write(
    ctx: &mut WarpContext,
    resolved: &ResolvedSlice,
    num: usize,
    access_kind: TensorAccessKind,
) -> IResult<()> {
    let offsets = selected_row_offsets(ctx, resolved, num)?;
    let region = ctx.tensor_region_with_offsets(resolved, &offsets)?;
    let scope = ctx.access_scope();
    ctx.emit(TraceEventKind::Write {
        region,
        proxy: MemoryProxy::Generic,
        access_kind: MemoryAccessKind::Tensor(access_kind),
        scope,
    })
}

fn selected_row_offsets(
    ctx: &WarpContext,
    resolved: &ResolvedSlice,
    num: usize,
) -> IResult<Vec<Vec<i64>>> {
    let row_lanes = lane_row_map(ctx)?;
    let mut out = Vec::with_capacity(num * 8);
    for lane_rows in row_lanes {
        for lane_ai in lane_rows.iter().take(num * 8) {
            out.push(resolved.offsets.row(*lane_ai).iter().copied().collect());
        }
    }
    Ok(out)
}
