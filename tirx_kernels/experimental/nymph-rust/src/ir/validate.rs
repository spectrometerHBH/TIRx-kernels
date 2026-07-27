//! Validation — a faithful port of every check in `ir.py`'s `__post_init__`
//! methods and the helper `_check_*` functions. With `Arc<Tensor>` we can do the
//! cross-referencing checks (space/dtype/rank) too, since `slice.tensor.*` reads
//! the tensor's data directly.
//!
//! Python runs these per-object at construction; we run them as one pass over the
//! assembled kernel (`Kernel::validate`). Same checks, same intent — they just
//! fire when you call `validate()` rather than at each `__post_init__`.

use super::*;
use std::collections::{BTreeSet, HashMap, HashSet};

/// An IR validation error (mirrors Python's `ValueError`/`TypeError` messages).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct IrError {
    pub message: String,
}
impl std::fmt::Display for IrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}
type R = Result<(), IrError>;
fn err(msg: impl Into<String>) -> IrError {
    IrError {
        message: msg.into(),
    }
}
fn bail(msg: impl Into<String>) -> R {
    Err(err(msg))
}

// ---------------------------------------------------------------------------
// small leaf helpers (mirror the `_check_*` functions)
// ---------------------------------------------------------------------------

fn is_reg_dtype(d: DType) -> bool {
    matches!(
        d,
        DType::F16 | DType::Bf16 | DType::F32 | DType::I32 | DType::U32
    )
}
fn is_b16_dtype(d: DType) -> bool {
    matches!(d, DType::I16 | DType::U16 | DType::F16 | DType::Bf16)
}
fn is_b32_reg_dtype(d: DType) -> bool {
    matches!(d, DType::I32 | DType::U32)
}
fn is_float_reg_dtype(d: DType) -> bool {
    matches!(d, DType::F16 | DType::Bf16 | DType::F32)
}

/// The scalar dtype a GMEM tensor of `d` decodes to, if `d` is a scalar integer
/// or bool (mirrors `_SCALAR_GMEM_DTYPES`).
fn scalar_gmem_dtype(d: DType) -> Option<ScalarDType> {
    match d {
        DType::Bool => Some(ScalarDType::Bool),
        DType::I32 => Some(ScalarDType::I32),
        DType::U32 => Some(ScalarDType::U32),
        DType::I64 => Some(ScalarDType::I64),
        DType::U64 => Some(ScalarDType::U64),
        _ => None,
    }
}

/// The concrete integer value of a scalar, if it's a literal (else None — symbolic).
fn static_int(v: &ScalarValue) -> Option<i64> {
    match v {
        ScalarValue::Int(n) => Some(*n),
        _ => None,
    }
}

fn static_shape_numel(shape: &[ScalarValue]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        static_int(dim).and_then(|d| usize::try_from(d).ok().and_then(|u| acc.checked_mul(u)))
    })
}

fn check_positive(value: u32, label: &str) -> R {
    if value < 1 {
        return bail(format!("{label} must be a positive integer"));
    }
    Ok(())
}
fn check_cta_group(value: u8, label: &str) -> R {
    if value != 1 && value != 2 {
        return bail(format!("{label} must be 1 or 2"));
    }
    Ok(())
}
fn check_uint16(value: Option<u16>, _label: &str) -> R {
    // Stored as u16, so range is already guaranteed; nothing to check.
    let _ = value;
    Ok(())
}
fn check_tmem_cols(value: u32, label: &str) -> R {
    if value < 32 || value > 512 || (value & (value - 1)) != 0 {
        return bail(format!(
            "{label} must be a power-of-two integer in [32, 512]"
        ));
    }
    Ok(())
}
fn check_num_warps(value: u32) -> R {
    if value < 1 || value % 4 != 0 {
        return bail("kernel num_warps must be a positive multiple of 4");
    }
    Ok(())
}
fn check_execution_shape(shape: &[usize], label: &str) -> R {
    if shape.is_empty() || shape.len() > 3 {
        return bail(format!("kernel {label} rank must be in [1, 3]"));
    }
    if shape.iter().any(|&d| d < 1) {
        return bail(format!("kernel {label} must contain positive integers"));
    }
    Ok(())
}

fn validate_tma_gmem_shape(
    gmem_shape: &Option<Vec<usize>>,
    tensor_shape: &[usize],
    smem_tile_shape: &[usize],
    label: &str,
) -> R {
    let Some(gmem_shape) = gmem_shape else {
        return Ok(());
    };
    if gmem_shape.len() != tensor_shape.len() {
        return bail(format!(
            "{label} gmem_shape rank must match GMEM tensor rank"
        ));
    }
    if gmem_shape.contains(&0) {
        return bail(format!("{label} gmem_shape must contain positive integers"));
    }
    if tensor_numel(gmem_shape) != tensor_numel(smem_tile_shape) {
        return bail(format!(
            "{label} gmem_shape element count must match SMEM tile shape"
        ));
    }
    Ok(())
}

fn dtype_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::Bool | DType::I8 | DType::U8 | DType::F8E4M3 => 1,
        DType::I16 | DType::U16 | DType::F16 | DType::Bf16 => 2,
        DType::I32 | DType::U32 | DType::F32 => 4,
        DType::I64 | DType::U64 => 8,
    }
}

fn tensor_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn smem_extent_bytes(tensor: &Tensor) -> Result<usize, IrError> {
    tensor_numel(&tensor.shape)
        .and_then(|n| n.checked_mul(dtype_size_bytes(tensor.dtype)))
        .ok_or_else(|| err("smem tensor byte extent overflows usize"))
}

// ---------------------------------------------------------------------------
// leaf walkers: scalar exprs, layouts, slices
// ---------------------------------------------------------------------------

/// `ScalarExpr.__post_init__` arity check, recursively over an expr tree.
fn validate_scalar(v: &ScalarValue) -> R {
    if let ScalarValue::Expr(e) = v {
        if e.args.len() != e.op.arity() {
            return bail(format!(
                "scalar expr op {:?} expects {} args",
                e.op,
                e.op.arity()
            ));
        }
        for arg in &e.args {
            validate_scalar(arg)?;
        }
    }
    Ok(())
}

/// `Tensor.__post_init__`: SMEM tensors carry a byte offset; nothing else may.
/// Every nontrivial physical layout is explicit and validated independently of
/// tensor usage.
fn validate_tensor(t: &Tensor) -> R {
    match t.space {
        MemorySpace::Smem => {
            if t.byte_offset.is_none() {
                return bail("smem tensor byte_offset is required");
            }
        }
        _ => {
            if t.byte_offset.is_some() {
                return bail("byte_offset is only valid for SMEM tensors");
            }
        }
    }
    if let Some(Layout::Swizzle(layout)) = t.layout {
        if t.space != MemorySpace::Smem {
            return bail("smem swizzle layout is only valid for SMEM tensors");
        }
        if t.shape.len() < 2 {
            return bail("smem swizzle layout requires a tensor rank of at least 2");
        }
        if layout.swizzle != Swizzle::None {
            let rows = t.shape[t.shape.len() - 2];
            let cols = t.shape[t.shape.len() - 1];
            let row_bytes = cols
                .checked_mul(dtype_size_bytes(t.dtype))
                .ok_or_else(|| err("smem swizzle row byte width overflows usize"))?;
            let atom_bytes = match layout.swizzle {
                Swizzle::None => unreachable!(),
                Swizzle::B32 => 32,
                Swizzle::B64 => 64,
                Swizzle::B128 => 128,
            };
            if row_bytes < atom_bytes || row_bytes % atom_bytes != 0 {
                return bail(format!(
                    "smem swizzle {:?} requires the final row to be a multiple of \
                     {atom_bytes} bytes, got {row_bytes}",
                    layout.swizzle
                ));
            }
            if rows == 0 || rows % 8 != 0 {
                return bail(format!(
                    "smem swizzle {:?} requires the penultimate dimension to be a \
                     positive multiple of 8, got {rows}",
                    layout.swizzle
                ));
            }
        }
    }
    Ok(())
}

/// `TensorSlice.__post_init__`: rank matches the tensor, offsets/dims non-negative,
/// static bounds. Also validates the tensor and the offset/shape scalars.
fn validate_slice(s: &TensorSlice, label: &str) -> R {
    validate_tensor(&s.tensor)?;
    let rank = s.tensor.shape.len();
    if s.offsets.len() != rank || s.shape.len() != rank {
        return bail(format!("{label} slice rank must match tensor rank"));
    }
    for (i, (offset, dim)) in s.offsets.iter().zip(s.shape.iter()).enumerate() {
        validate_scalar(offset)?;
        validate_scalar(dim)?;
        let tdim = s.tensor.shape[i] as i64;
        if let Some(o) = static_int(offset) {
            if o < 0 {
                return bail(format!("{label} slice offset must be non-negative"));
            }
            if let Some(d) = static_int(dim) {
                if d < 0 {
                    return bail(format!(
                        "{label} slice shape dimension must be non-negative"
                    ));
                }
                if o + d > tdim {
                    return bail(format!("{label} slice is out of bounds"));
                }
            }
        }
    }
    Ok(())
}

/// `TmemAddr` checks: the address scalars are well-formed, and a constant
/// lane/column base is in range. (Column-band membership against the live
/// allocations is the liveness walk's job — see `check_tmem_alloc_bands`.)
fn validate_tmem_addr(op: &TmemAddr, label: &str) -> R {
    validate_scalar(&op.row)?;
    validate_scalar(&op.col)?;
    if op.tensor.start_col >= 512 {
        return bail(format!("{label} tensor start_col must be in [0, 512)"));
    }
    if let Some(r) = static_int(&op.row) {
        if !(0..128).contains(&r) {
            return bail(format!("{label} row (TMEM lane) must be in [0, 128)"));
        }
    }
    if let Some(c) = static_int(&op.col) {
        if !(0..512).contains(&c) {
            return bail(format!("{label} col (TMEM column) must be in [0, 512)"));
        }
        if i64::from(op.tensor.start_col) + c >= 512 {
            return bail(format!(
                "{label} start_col + col (TMEM column) must be in [0, 512)"
            ));
        }
    }
    Ok(())
}

/// Recursively validate every scalar embedded in an explicit physical SMEM
/// tile. Shape/space/bounds checks live in the shared tcgen05 resolver.
fn validate_smem_tile_scalars(tile: &SmemTile) -> R {
    validate_tensor(&tile.tensor)?;
    for index in &tile.prefix_indices {
        validate_scalar(index)?;
    }
    validate_scalar(&tile.row_offset)?;
    validate_scalar(&tile.col_offset)?;
    Ok(())
}

/// The TMEM column-cell span of a `Tcgen05Ld`/`Tcgen05St` atom: 32x32b moves
/// one 32-bit column per register; the 16x*b atoms move b/32 columns per
/// register step.
fn tmem_operand_lanes_cols(shape: &LdStShape, num: u32) -> Option<usize> {
    let per = match shape {
        LdStShape::B32x32 => 1usize,
        LdStShape::B16x32Bx2 | LdStShape::B16x64 => 2,
        LdStShape::B16x128 => 4,
        LdStShape::B16x256 => 8,
    };
    usize::try_from(num).ok().map(|n| n * per)
}

fn check_slice_covers(s: &TensorSlice, shape: &[usize], label: &str) -> R {
    for (slice_dim, &shape_dim) in s.shape.iter().zip(shape.iter()) {
        if let Some(d) = static_int(slice_dim) {
            if (d as usize) < shape_dim {
                return bail(format!("{label} does not cover requested shape"));
            }
        }
    }
    Ok(())
}

/// `_check_reg_alu_operands` (and the fma variant).
fn check_reg_alu(dst: &TensorSlice, operands: &[(&str, &RegOperand)], label: &str) -> R {
    validate_slice(dst, &format!("{label} dst"))?;
    if dst.tensor.space != MemorySpace::Reg {
        return bail(format!("{label} dst must be REG"));
    }
    if !is_reg_dtype(dst.tensor.dtype) {
        return bail(format!("{label} dtype must be f16, bf16, f32, i32, or u32"));
    }
    for (name, operand) in operands {
        check_reg_operand(dst, operand, &format!("{label} {name}"))?;
    }
    Ok(())
}

fn check_reg_operand(dst: &TensorSlice, operand: &RegOperand, label: &str) -> R {
    check_reg_operand_as(dst, operand, dst.tensor.dtype, label)
}

fn check_reg_operand_as(
    dst: &TensorSlice,
    operand: &RegOperand,
    expected_dtype: DType,
    label: &str,
) -> R {
    let Some(slice) = operand.as_slice() else {
        return Ok(());
    };
    validate_slice(slice, label)?;
    if slice.tensor.space != MemorySpace::Reg {
        return bail(format!("{label} must be REG"));
    }
    if slice.tensor.dtype != expected_dtype {
        return bail(format!("{label} dtype must be {expected_dtype:?}"));
    }
    if slice.shape == dst.shape {
        return Ok(());
    }
    match static_shape_numel(&slice.shape) {
        Some(1) | None => Ok(()),
        Some(_) => bail(format!(
            "{label} slice shape must match dst shape or contain one element per thread"
        )),
    }
}

fn reg_operand_slices<'a>(operand: &'a RegOperand, out: &mut Vec<&'a TensorSlice>) {
    if let RegOperand::Slice(slice) = operand {
        out.push(slice);
    }
}

fn reg_stmt_slices<'a>(stmt: &'a Stmt, out: &mut Vec<&'a TensorSlice>) -> bool {
    match stmt {
        Stmt::RegFill { dst, value } => {
            out.push(dst);
            reg_operand_slices(value, out);
        }
        Stmt::RegUnary { dst, src, .. } | Stmt::RegReduce { dst, src, .. } => {
            out.push(dst);
            reg_operand_slices(src, out);
        }
        Stmt::RegAdd { dst, lhs, rhs, .. }
        | Stmt::RegSub { dst, lhs, rhs, .. }
        | Stmt::RegMul { dst, lhs, rhs }
        | Stmt::RegMax { dst, lhs, rhs }
        | Stmt::RegMin { dst, lhs, rhs }
        | Stmt::RegBitwise { dst, lhs, rhs, .. } => {
            out.push(dst);
            reg_operand_slices(lhs, out);
            reg_operand_slices(rhs, out);
        }
        Stmt::RegFma { dst, a, b, c } => {
            out.push(dst);
            reg_operand_slices(a, out);
            reg_operand_slices(b, out);
            reg_operand_slices(c, out);
        }
        Stmt::RegCondRescale {
            dst,
            src,
            scale,
            threshold,
            ..
        } => {
            out.push(dst);
            reg_operand_slices(src, out);
            reg_operand_slices(scale, out);
            reg_operand_slices(threshold, out);
        }
        Stmt::RegSoftmaxRescale {
            row_max,
            row_scale,
            row_max_old,
            row_max_new,
            scale_log2,
            threshold,
        } => {
            out.push(row_max);
            out.push(row_scale);
            reg_operand_slices(row_max_old, out);
            reg_operand_slices(row_max_new, out);
            reg_operand_slices(scale_log2, out);
            reg_operand_slices(threshold, out);
        }
        Stmt::RegCausalMask {
            dst,
            src,
            mask_value,
            ..
        } => {
            out.push(dst);
            reg_operand_slices(src, out);
            reg_operand_slices(mask_value, out);
        }
        Stmt::RegCombineIntFracEx2 {
            dst,
            rounded,
            frac_ex2,
        } => {
            out.push(dst);
            reg_operand_slices(rounded, out);
            reg_operand_slices(frac_ex2, out);
        }
        Stmt::RegCvt { dst, src, .. }
        | Stmt::RegLoad { dst, src }
        | Stmt::RegStore { dst, src } => {
            out.push(dst);
            out.push(src);
        }
        _ => return false,
    }
    true
}

fn reg_stmt_vars(stmt: &Stmt, out: &mut Vec<Var>) -> bool {
    let mut slices = Vec::new();
    if !reg_stmt_slices(stmt, &mut slices) {
        return false;
    }
    for slice in slices {
        slice_vars(slice, out);
    }
    if let Stmt::RegCausalMask {
        query_start,
        key_start,
        ..
    } = stmt
    {
        collect_vars(query_start, out);
        collect_vars(key_start, out);
    }
    true
}

// ---------------------------------------------------------------------------
// per-statement validation (the bodies of each `__post_init__`)
// ---------------------------------------------------------------------------

fn validate_stmt(s: &Stmt) -> R {
    match s {
        Stmt::TensorDef { tensor } => validate_tensor(tensor)?,
        Stmt::TmemAlloc {
            base_col,
            n_cols,
            cta_group,
            addr_byte_offset: _,
        }
        | Stmt::TmemDealloc {
            base_col,
            n_cols,
            cta_group,
        } => {
            check_tmem_cols(*n_cols, "tmem n_cols")?;
            if *base_col >= 512 || *base_col + *n_cols > 512 {
                return bail("tmem column band [base_col, base_col + n_cols) must fit in [0, 512)");
            }
            check_cta_group(*cta_group, "tmem cta_group")?;
        }
        Stmt::TmemRelinquish { cta_group } => {
            check_cta_group(*cta_group, "tmem cta_group")?;
        }
        Stmt::ScalarDef { var, initial } => {
            if var.binding != VarBinding::Scalar {
                return bail("scalar_def var binding must be scalar");
            }
            match initial {
                ScalarInitial::Value(v) => validate_scalar(v)?,
                ScalarInitial::Tensor(t) => {
                    if !matches!(t.tensor.space, MemorySpace::Gmem | MemorySpace::Smem) {
                        return bail("scalar_def tensor initial must be GMEM or SMEM");
                    }
                    let scalar_dt = scalar_gmem_dtype(t.tensor.dtype).ok_or_else(|| {
                        err("scalar_def tensor initial dtype must be scalar integer or bool")
                    })?;
                    if var.dtype != scalar_dt {
                        return bail("scalar_def var dtype must match tensor initial scalar dtype");
                    }
                    if t.shape.iter().any(|d| static_int(d) != Some(1)) {
                        return bail("scalar_def tensor initial must be a scalar slice");
                    }
                }
            }
        }
        Stmt::ScalarStore { var, value } => {
            if var.binding != VarBinding::Scalar {
                return bail("scalar_store var binding must be scalar");
            }
            validate_scalar(value)?;
        }
        Stmt::ScalarLet { var, value } => {
            if var.binding != VarBinding::Scalar {
                return bail("scalar_let var binding must be scalar");
            }
            validate_scalar(value)?;
        }
        Stmt::ShuffleSync { var, src, src_lane } => {
            if var.binding != VarBinding::Scalar {
                return bail("shuffle_sync var binding must be scalar");
            }
            validate_scalar(src)?;
            validate_scalar(src_lane)?;
            if let Some(l) = static_int(src_lane) {
                if !(0..32).contains(&l) {
                    return bail("shuffle_sync src_lane must be in [0, 32)");
                }
            }
        }
        Stmt::StoreScalar { dst, value } => {
            validate_slice(dst, "store_scalar dst")?;
            validate_scalar(value)?;
            if dst.tensor.space != MemorySpace::Smem {
                return bail("store_scalar dst must be SMEM");
            }
            if dst.shape.iter().any(|d| static_int(d) != Some(1)) {
                return bail("store_scalar dst must be a scalar slice");
            }
        }
        Stmt::MBarDef { mbar } => {
            if mbar.stages < 1 {
                return bail("mbar stages must be a positive integer");
            }
            if let Some(c) = mbar.arrive_count {
                if c < 1 {
                    return bail("mbar arrive_count must be a positive integer or None");
                }
            }
        }

        Stmt::ForLoop {
            var,
            start,
            stop,
            step,
            unroll,
            ..
        } => {
            if var.binding != VarBinding::Loop {
                return bail("for_loop var binding must be loop");
            }
            validate_scalar(start)?;
            validate_scalar(stop)?;
            validate_scalar(step)?;
            if let Some(s) = static_int(step) {
                if s <= 0 {
                    return bail("for_loop step must be positive");
                }
            }
            // The pinned-rolled form (unroll=false) emits `T.serial(stop,
            // unroll=False)`, which expresses only a 0-based unit-stride range —
            // any other range would be silently re-timed at emission.
            if !unroll && (static_int(start) != Some(0) || static_int(step) != Some(1)) {
                return bail(
                    "for_loop unroll=false requires literal start=0 and step=1 \
                     (the pinned-rolled emission expresses no other range)",
                );
            }
        }
        Stmt::ForEachTask { scheduler, var, .. } => {
            validate_scheduler(scheduler)?;
            if var.binding != VarBinding::Task {
                return bail("for_each_task var binding must be task");
            }
        }
        Stmt::SchedulerImpl { scheduler, .. } => {
            validate_scheduler(scheduler)?;
            if scheduler.policy.is_functional() {
                return bail("scheduler_impl requires a concurrent scheduler policy");
            }
        }
        Stmt::SchedNext { scheduler, var } => {
            validate_scheduler(scheduler)?;
            if scheduler.policy.is_functional() {
                return bail("sched_next requires a concurrent scheduler policy");
            }
            if var.binding != VarBinding::Task {
                return bail("sched_next var binding must be task");
            }
        }
        Stmt::ClcTryCancel {
            scheduler,
            handle,
            mbar,
            cta_group,
            ..
        } => {
            validate_scheduler(scheduler)?;
            if scheduler.policy.is_functional() {
                return bail("clc_try_cancel requires a concurrent scheduler policy");
            }
            if handle.space != MemorySpace::Smem {
                return bail("clc_try_cancel handle must be SMEM");
            }
            // try_cancel completes-tx the signalled mbarrier like a TMA landing.
            if mbar.mbar.kind != MBarKind::Tma {
                return bail("clc_try_cancel mbar kind must be tma");
            }
            // The CLC response written into the handle is a 16B (uint4) cell.
            let handle_bytes = tensor_numel(&handle.shape)
                .and_then(|n| n.checked_mul(dtype_size_bytes(handle.dtype)));
            if handle_bytes.map_or(true, |b| b < 16) {
                return bail("clc_try_cancel handle must be at least 16 bytes");
            }
            check_cta_group(*cta_group, "clc_try_cancel cta_group")?;
        }
        Stmt::ClcQueryCancel {
            scheduler,
            var,
            handle,
        } => {
            validate_scheduler(scheduler)?;
            if scheduler.policy.is_functional() {
                return bail("clc_query_cancel requires a concurrent scheduler policy");
            }
            if handle.space != MemorySpace::Smem {
                return bail("clc_query_cancel handle must be SMEM");
            }
            if var.binding != VarBinding::Scalar {
                return bail("clc_query_cancel var binding must be scalar");
            }
        }
        Stmt::Loop { .. } => {}
        // If/BreakIf conditions may branch on any scope value — warp/lane
        // dispatch via `If` IS the execution model (per-warp streams with
        // masked lanes), so warp_id/lane_id predicates are the normal case.
        Stmt::BreakIf { cond } => validate_scalar(cond)?,
        Stmt::If { cond, .. } => validate_scalar(cond)?,
        Stmt::SetMaxNReg { nreg } => {
            if *nreg == 0 || *nreg % 8 != 0 {
                return bail("setmaxnreg nreg must be a positive multiple of 8");
            }
        }

        Stmt::MBarrierInit { count, stage, .. } => {
            check_positive(*count, "mbarrier_init count")?;
            // PTX mbarrier v0 layout: the arrival-count field is 20 bits.
            if *count > (1 << 20) - 1 {
                return bail("mbarrier_init count must be <= 2^20 - 1");
            }
            if let Some(v) = stage {
                validate_scalar(v)?;
            }
        }
        Stmt::MBarrierArrive { stage, count, .. } => {
            if let Some(v) = stage {
                validate_scalar(v)?;
            }
            validate_scalar(count)?;
        }
        Stmt::MBarrierWait { stage, phase, .. } => {
            if let Some(v) = stage {
                validate_scalar(v)?;
            }
            // phase is REQUIRED: the phase-less form waits on the CURRENT
            // parity in the sim while codegen emits a constant 0 — a latent
            // sim/codegen divergence. Every kernel passes it explicitly.
            let Some(v) = phase else {
                return bail("mbarrier_wait phase is required");
            };
            validate_scalar(v)?;
        }
        Stmt::MBarrierExpectTx { bytes, stage, .. }
        | Stmt::MBarrierArriveExpectTx { bytes, stage, .. } => {
            check_positive(*bytes, "mbarrier expect_tx bytes")?;
            if let Some(v) = stage {
                validate_scalar(v)?;
            }
        }

        Stmt::TmaLoad {
            dst,
            src,
            mbar,
            coords,
            shape,
            gmem_shape,
            mbar_stage,
            multicast_cta_mask,
            cache_hint: _,
            prefetch_tensormap: _,
            cta_group,
        } => {
            validate_slice(dst, "tma_load dst")?;
            validate_tensor(src)?;
            if mbar.mbar.kind != MBarKind::Tma {
                return bail("tma_load mbar kind must be tma");
            }
            if let Some(v) = mbar_stage {
                validate_scalar(v)?;
            }
            check_uint16(*multicast_cta_mask, "tma_load multicast_cta_mask")?;
            if dst.tensor.space != MemorySpace::Smem {
                return bail("tma_load dst must be SMEM");
            }
            if src.space != MemorySpace::Gmem {
                return bail("tma_load src must be GMEM");
            }
            if dst.tensor.dtype != src.dtype {
                return bail("tma_load dst and src dtype must match");
            }
            // The transfer size is DERIVED from the tile (`numel(shape) x
            // dtype`) in the sim's tx accounting — the same value TIRx derives
            // from the box extents — so there is no separate `bytes` to check.
            // A zero-extent tile would derive a 0-byte transfer (the old
            // runtime `bytes >= 1` guard errored on it) — fail at build.
            if tensor_numel(shape) == Some(0) {
                return bail("tma_load shape must have a nonzero element count");
            }
            for c in coords {
                validate_scalar(c)?;
            }
            if coords.len() != src.shape.len() {
                return bail("tma_load coords rank must match src tensor rank");
            }
            if shape.len() != dst.tensor.shape.len() {
                return bail("tma_load shape rank must match dst tensor rank");
            }
            validate_tma_gmem_shape(gmem_shape, &src.shape, shape, "tma_load")?;
            check_slice_covers(dst, shape, "tma_load dst slice")?;
            check_cta_group(*cta_group, "tma_load cta_group")?;
        }
        Stmt::TmaStore {
            dst,
            src,
            coords,
            shape,
            gmem_shape,
            reduce_add,
            allow_nondet_reduce,
            cache_hint: _,
            prefetch_tensormap: _,
        } => {
            validate_slice(src, "tma_store src")?;
            validate_tensor(dst)?;
            if dst.space != MemorySpace::Gmem {
                return bail("tma_store dst must be GMEM");
            }
            if *reduce_add && dst.dtype != DType::F32 {
                return bail("tma_reduce_add dst must be f32");
            }
            // Float reduce-add is order-dependent (float add is not associative): the
            // checker can only WARN `nondeterministic_reduction`, so the IR makes the
            // non-determinism opt-in instead — reject unless the author declared it.
            if *reduce_add && !dst.dtype.is_integer() && !allow_nondet_reduce {
                return bail(
                    "tma_reduce_add on a non-integer dst is order-dependent; \
                     pass allow_nondet_reduce=true to opt in",
                );
            }
            if src.tensor.space != MemorySpace::Smem {
                return bail("tma_store src must be SMEM");
            }
            if dst.dtype != src.tensor.dtype {
                return bail("tma_store dst and src dtype must match");
            }
            for c in coords {
                validate_scalar(c)?;
            }
            if coords.len() != dst.shape.len() {
                return bail("tma_store coords rank must match dst tensor rank");
            }
            if shape.len() != src.tensor.shape.len() {
                return bail("tma_store shape rank must match src tensor rank");
            }
            validate_tma_gmem_shape(gmem_shape, &dst.shape, shape, "tma_store")?;
            check_slice_covers(src, shape, "tma_store src slice")?;
        }
        Stmt::CpAsyncBulkS2Cluster {
            dst,
            src,
            mbar,
            bytes,
        } => {
            validate_slice(src, "cp_async_bulk_s2cluster src")?;
            validate_slice(dst, "cp_async_bulk_s2cluster dst")?;
            validate_scalar(bytes)?;
            if src.tensor.space != MemorySpace::Smem || dst.tensor.space != MemorySpace::Smem {
                return bail("cp_async_bulk_s2cluster src and dst must both be SMEM");
            }
            if dst.tensor.dtype != src.tensor.dtype {
                return bail("cp_async_bulk_s2cluster src and dst dtype must match");
            }
            // The copy targets a PEER CTA's SMEM and signals that peer's mbar —
            // without a remote_coord the mbar resolves to the issuing CTA
            // itself, which is not the cross-CTA exchange this op models.
            if mbar.remote_coord.is_none() {
                return bail("cp_async_bulk_s2cluster mbar must target a peer CTA (remote_coord)");
            }
        }
        Stmt::GmemAtomicAdd {
            sem, coords, value, ..
        } => {
            validate_slice(sem, "gmem_atomic_add sem")?;
            validate_scalar(value)?;
            if sem.tensor.space != MemorySpace::Gmem {
                return bail("gmem_atomic_add sem must be GMEM");
            }
            if sem.tensor.dtype != DType::I32 {
                return bail("gmem_atomic_add sem must be an i32 semaphore");
            }
            if coords.len() != sem.tensor.shape.len() {
                return bail("gmem_atomic_add coords rank must match the semaphore tensor rank");
            }
            for c in coords {
                validate_scalar(c)?;
            }
        }
        Stmt::GmemWaitEq { sem, coords, value } => {
            validate_slice(sem, "gmem_wait_eq sem")?;
            validate_scalar(value)?;
            if sem.tensor.space != MemorySpace::Gmem {
                return bail("gmem_wait_eq sem must be GMEM");
            }
            if sem.tensor.dtype != DType::I32 {
                return bail("gmem_wait_eq sem must be an i32 semaphore");
            }
            if coords.len() != sem.tensor.shape.len() {
                return bail("gmem_wait_eq coords rank must match the semaphore tensor rank");
            }
            for c in coords {
                validate_scalar(c)?;
            }
        }
        Stmt::CpAsyncBulkCommitGroup => {}
        Stmt::CpAsyncBulkWaitGroupRead { n } => {
            if *n > 8 {
                return bail("cp_async_bulk_wait_group_read n must be <= 8");
            }
        }

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
        } => {
            validate_tmem_addr(dst, "tcgen05_mma dst")?;
            validate_scalar(accum)?;
            validate_smem_tile_scalars(b)?;
            match a {
                MmaAOperand::Smem(tile) => validate_smem_tile_scalars(tile)?,
                MmaAOperand::Tmem { addr, .. } => validate_tmem_addr(addr, "tcgen05_mma a")?,
            }
            if let Some(scale) = block_scale {
                validate_tmem_addr(&scale.sfa, "tcgen05_mma sfa")?;
                validate_tmem_addr(&scale.sfb, "tcgen05_mma sfb")?;
            }
            resolve_tcgen05_mma(
                dst,
                a,
                b,
                *mma_m,
                *mma_n,
                *format,
                block_scale.as_ref(),
                *trans_a,
                *trans_b,
                *ws,
                *cta_group,
            )
            .map_err(|error| err(error.message))?;
        }
        Stmt::Tcgen05Cp {
            dst,
            src,
            shape,
            multicast,
            cta_group,
        } => {
            validate_tmem_addr(dst, "tcgen05_cp dst")?;
            validate_smem_tile_scalars(src)?;
            resolve_tcgen05_cp(dst, src, *shape, *multicast, *cta_group)
                .map_err(|error| err(error.message))?;
        }
        Stmt::Tcgen05Commit {
            mbar,
            stage,
            cta_group,
            multicast_cta_mask,
        } => {
            if let Some(v) = stage {
                validate_scalar(v)?;
            }
            if mbar.mbar.kind != MBarKind::Tcgen05 {
                return bail("tcgen05_commit mbar kind must be tcgen05");
            }
            check_cta_group(*cta_group, "tcgen05_commit cta_group")?;
            check_uint16(*multicast_cta_mask, "tcgen05_commit multicast_cta_mask")?;
        }
        Stmt::Tcgen05Ld {
            dst,
            src,
            shape,
            num,
        } => {
            validate_slice(dst, "tcgen05_ld dst")?;
            if dst.tensor.space != MemorySpace::Reg {
                return bail("tcgen05_ld dst must be REG");
            }
            validate_tmem_addr(src, "tcgen05_ld src")?;
            if !matches!(
                dst.tensor.dtype,
                DType::F16 | DType::Bf16 | DType::F32 | DType::I32 | DType::U32
            ) {
                return bail("tcgen05_ld dtype must be f16, bf16, f32, i32, or u32");
            }
            check_ld_atom(*shape, *num, "tcgen05_ld")?;
            check_slice_covers(
                dst,
                &[shape
                    .register_count(*num)
                    .expect("validated tcgen05 ld atom")],
                "tcgen05_ld dst",
            )?;
        }
        Stmt::Tcgen05WaitLd => {}
        Stmt::Tcgen05St {
            dst,
            src,
            shape,
            num,
        } => {
            validate_slice(src, "tcgen05_st src")?;
            validate_tmem_addr(dst, "tcgen05_st dst")?;
            if src.tensor.space != MemorySpace::Reg {
                return bail("tcgen05_st src must be REG");
            }
            if !matches!(
                src.tensor.dtype,
                DType::F16 | DType::Bf16 | DType::F32 | DType::I32 | DType::U32
            ) {
                return bail("tcgen05_st dtype must be f16, bf16, f32, i32, or u32");
            }
            check_ld_atom(*shape, *num, "tcgen05_st")?;
            check_slice_covers(
                src,
                &[shape
                    .register_count(*num)
                    .expect("validated tcgen05 st atom")],
                "tcgen05_st src",
            )?;
        }
        Stmt::Tcgen05WaitSt => {}
        Stmt::LdMatrix {
            dst,
            src,
            shape,
            num,
            dtype,
            ..
        } => {
            validate_slice(dst, "ldmatrix dst")?;
            validate_slice(src, "ldmatrix src")?;
            check_matrix_atom(*shape, *num, *dtype, "ldmatrix")?;
            if dst.tensor.space != MemorySpace::Reg {
                return bail("ldmatrix dst must be REG");
            }
            if src.tensor.space != MemorySpace::Smem {
                return bail("ldmatrix src must be SMEM");
            }
            let b16_dst = matches!(dst.tensor.dtype, DType::F16 | DType::Bf16);
            if !b16_dst && !is_b32_reg_dtype(dst.tensor.dtype) {
                return bail("ldmatrix dst dtype must be i32/u32 or an f16/bf16 fragment");
            }
            if !is_b16_dtype(src.tensor.dtype) {
                return bail("ldmatrix src dtype must be f16, bf16, i16, or u16");
            }
            if let Some(n) = static_shape_numel(&dst.shape) {
                let want = if b16_dst {
                    2 * *num as usize
                } else {
                    *num as usize
                };
                if n != want {
                    return bail(
                        "ldmatrix dst slice must contain num b32 registers (2*num b16 elements)",
                    );
                }
            }
            if let Some(n) = static_shape_numel(&src.shape) {
                if n != 8 {
                    return bail("ldmatrix src slice must contain one row of eight b16 elements");
                }
            }
        }
        Stmt::StMatrix {
            dst,
            src,
            shape,
            num,
            dtype,
            ..
        } => {
            validate_slice(dst, "stmatrix dst")?;
            validate_slice(src, "stmatrix src")?;
            check_matrix_atom(*shape, *num, *dtype, "stmatrix")?;
            if dst.tensor.space != MemorySpace::Smem {
                return bail("stmatrix dst must be SMEM");
            }
            if src.tensor.space != MemorySpace::Reg {
                return bail("stmatrix src must be REG");
            }
            if !is_b16_dtype(dst.tensor.dtype) {
                return bail("stmatrix dst dtype must be f16, bf16, i16, or u16");
            }
            // The source is either `num` b32 words, or a b16 fragment of
            // 2*num elements whose consecutive pairs ARE the b32 words — the
            // packed register file the f32->b16x2 pair cvt produces (there is
            // no pack instruction on silicon; a b16 pair IS a 32-bit register).
            let b16_src = is_b16_dtype(src.tensor.dtype);
            if !b16_src && !is_b32_reg_dtype(src.tensor.dtype) {
                return bail("stmatrix src dtype must be i32/u32 words or a b16 fragment");
            }
            if let Some(n) = static_shape_numel(&dst.shape) {
                if n != 8 {
                    return bail("stmatrix dst slice must contain one row of eight b16 elements");
                }
            }
            if let Some(n) = static_shape_numel(&src.shape) {
                let want = if b16_src {
                    2 * *num as usize
                } else {
                    *num as usize
                };
                if n != want {
                    return bail(
                        "stmatrix src slice must contain num b32 registers (2*num b16 elements)",
                    );
                }
            }
        }
        Stmt::WarpMma {
            d,
            a,
            b,
            c,
            m,
            n,
            k,
            ab_dtype,
        } => {
            for (sl, lbl) in [(d, "d"), (a, "a"), (b, "b"), (c, "c")] {
                validate_slice(sl, &format!("mma_sync {lbl}"))?;
                if sl.tensor.space != MemorySpace::Reg {
                    return bail(format!("mma_sync {lbl} must be REG"));
                }
            }
            // m16n8k{8,16}: A/B are u32 packed-16bit fragments, C/D are f32.
            if !(*m == 16 && *n == 8 && (*k == 8 || *k == 16)) {
                return bail("mma_sync supports only m16n8k8 / m16n8k16");
            }
            if !matches!(*ab_dtype, DType::Bf16 | DType::F16) {
                return bail("mma_sync ab_dtype must be bf16 or f16");
            }
            // A/B are u32 packed-bf16 words OR a bf16/f16 fragment (2 elems = 1 word).
            for (sl, lbl) in [(a, "A"), (b, "B")] {
                if !is_b32_reg_dtype(sl.tensor.dtype) && !is_b16_dtype(sl.tensor.dtype) {
                    return bail(format!(
                        "mma_sync {lbl} must be u32/i32 words or a bf16/f16 fragment"
                    ));
                }
            }
            if c.tensor.dtype != DType::F32 || d.tensor.dtype != DType::F32 {
                return bail("mma_sync C/D must be f32 register fragments");
            }
            let (la, lb, lcd) = (
                (*m * *k / 64) as usize,
                (*n * *k / 64) as usize,
                (*m * *n / 32) as usize,
            );
            for (sl, words, lbl) in [(a, la, "A"), (b, lb, "B"), (c, lcd, "C"), (d, lcd, "D")] {
                let want = if is_b16_dtype(sl.tensor.dtype) {
                    2 * words
                } else {
                    words
                };
                if let Some(got) = static_shape_numel(&sl.shape) {
                    if got != want {
                        return bail(format!(
                            "mma_sync {lbl} fragment must hold {want} elements per lane"
                        ));
                    }
                }
            }
        }

        Stmt::RegFill { dst, value } => check_reg_alu(dst, &[("value", value)], "reg_fill")?,
        Stmt::RegUnary { dst, src, .. } => {
            if !is_float_reg_dtype(dst.tensor.dtype) {
                return bail("reg_unary dst dtype must be f16, bf16, or f32");
            }
            check_reg_alu(dst, &[("src", src)], "reg_unary")?
        }
        Stmt::RegAdd { dst, lhs, rhs, .. } => {
            check_reg_alu(dst, &[("lhs", lhs), ("rhs", rhs)], "reg_add")?
        }
        Stmt::RegSub { dst, lhs, rhs, .. } => {
            check_reg_alu(dst, &[("lhs", lhs), ("rhs", rhs)], "reg_sub")?
        }
        Stmt::RegMul { dst, lhs, rhs } => {
            check_reg_alu(dst, &[("lhs", lhs), ("rhs", rhs)], "reg_mul")?
        }
        Stmt::RegMax { dst, lhs, rhs } => {
            check_reg_alu(dst, &[("lhs", lhs), ("rhs", rhs)], "reg_max")?
        }
        Stmt::RegMin { dst, lhs, rhs } => {
            check_reg_alu(dst, &[("lhs", lhs), ("rhs", rhs)], "reg_min")?
        }
        Stmt::RegFma { dst, a, b, c } => {
            check_reg_alu(dst, &[("a", a), ("b", b), ("c", c)], "reg_fma")?
        }
        Stmt::RegBitwise { dst, lhs, rhs, op } => {
            if !matches!(dst.tensor.dtype, DType::I32 | DType::U32) {
                return bail("reg_bitwise dst dtype must be i32 or u32");
            }
            if !matches!(op, RegBinaryOp::And | RegBinaryOp::Shl) {
                return bail("reg_bitwise op must be and or shl");
            }
            check_reg_alu(dst, &[("lhs", lhs), ("rhs", rhs)], "reg_bitwise")?
        }
        Stmt::RegReduce { dst, src, .. } => {
            validate_slice(dst, "reg_reduce dst")?;
            if dst.tensor.space != MemorySpace::Reg {
                return bail("reg_reduce dst must be REG");
            }
            if static_shape_numel(&dst.shape) != Some(1) {
                return bail("reg_reduce dst must contain exactly one element");
            }
            if let RegOperand::Slice(slice) = src {
                validate_slice(slice, "reg_reduce src")?;
                if slice.tensor.space != MemorySpace::Reg {
                    return bail("reg_reduce src must be REG");
                }
                if slice.tensor.dtype != dst.tensor.dtype {
                    return bail("reg_reduce src dtype must match dst dtype");
                }
            }
        }
        Stmt::RegCondRescale {
            dst,
            src,
            scale,
            threshold,
            scope,
        } => {
            if !is_float_reg_dtype(dst.tensor.dtype) {
                return bail("reg_cond_rescale dst dtype must be f16, bf16, or f32");
            }
            // A register reduction sees only the executing warp's rows, so its
            // group predicate spans one warp. A warpgroup group would need the
            // four warps' registers together, which one warp's execution cannot
            // provide.
            if matches!(scope, RegCondScope::Warpgroup) {
                return bail(
                    "reg_cond_rescale scope must be warp (a register reduction spans one warp)",
                );
            }
            check_reg_alu(
                dst,
                &[("src", src), ("scale", scale), ("threshold", threshold)],
                "reg_cond_rescale",
            )?;
        }
        Stmt::RegSoftmaxRescale {
            row_max,
            row_scale,
            row_max_old,
            row_max_new,
            scale_log2,
            threshold,
        } => {
            validate_slice(row_max, "reg_softmax_rescale row_max")?;
            validate_slice(row_scale, "reg_softmax_rescale row_scale")?;
            if row_max.tensor.space != MemorySpace::Reg {
                return bail("reg_softmax_rescale row_max must be REG");
            }
            if row_scale.tensor.space != MemorySpace::Reg {
                return bail("reg_softmax_rescale row_scale must be REG");
            }
            if !is_float_reg_dtype(row_max.tensor.dtype)
                || !is_float_reg_dtype(row_scale.tensor.dtype)
            {
                return bail("reg_softmax_rescale dst dtype must be f16, bf16, or f32");
            }
            if row_max.shape != row_scale.shape {
                return bail("reg_softmax_rescale row_max and row_scale shapes must match");
            }
            check_reg_operand_as(
                row_max,
                row_max_old,
                row_max.tensor.dtype,
                "reg_softmax_rescale row_max_old",
            )?;
            check_reg_operand_as(
                row_max,
                row_max_new,
                row_max.tensor.dtype,
                "reg_softmax_rescale row_max_new",
            )?;
            check_reg_operand_as(
                row_max,
                scale_log2,
                DType::F32,
                "reg_softmax_rescale scale_log2",
            )?;
            check_reg_operand_as(
                row_max,
                threshold,
                DType::F32,
                "reg_softmax_rescale threshold",
            )?;
        }
        Stmt::RegCausalMask {
            dst,
            src,
            query_start,
            key_start,
            group_size,
            mask_value,
            swap_qk: _,
        } => {
            if !is_float_reg_dtype(dst.tensor.dtype) {
                return bail("reg_causal_mask dst dtype must be f16, bf16, or f32");
            }
            if *group_size == 0 {
                return bail("reg_causal_mask group_size must be positive");
            }
            validate_scalar(query_start)?;
            validate_scalar(key_start)?;
            check_reg_alu(
                dst,
                &[("src", src), ("mask_value", mask_value)],
                "reg_causal_mask",
            )?;
        }
        Stmt::RegCombineIntFracEx2 {
            dst,
            rounded,
            frac_ex2,
        } => {
            if !is_float_reg_dtype(dst.tensor.dtype) {
                return bail("reg_combine_int_frac_ex2 dst dtype must be f16, bf16, or f32");
            }
            check_reg_alu(
                dst,
                &[("rounded", rounded), ("frac_ex2", frac_ex2)],
                "reg_combine_int_frac_ex2",
            )?;
        }
        Stmt::RegCvt { dst, src, rounding } => {
            validate_slice(dst, "reg_cvt dst")?;
            validate_slice(src, "reg_cvt src")?;
            if *rounding != Rounding::Rn {
                return bail("reg_cvt rounding=rm has no TIRx lowering");
            }
            if dst.tensor.space != MemorySpace::Reg || src.tensor.space != MemorySpace::Reg {
                return bail("reg_cvt dst and src must be REG");
            }
            if src.tensor.dtype != DType::F32 {
                return bail("reg_cvt src dtype must be f32");
            }
            if !matches!(dst.tensor.dtype, DType::F16 | DType::Bf16) {
                return bail("reg_cvt dst dtype must be f16 or bf16");
            }
            if dst.shape != src.shape {
                return bail("reg_cvt dst and src slice shapes must match");
            }
        }
        Stmt::RegLoad { dst, src } => check_reg_transfer(dst, src, "reg_load", true)?,
        Stmt::RegStore { dst, src } => check_reg_transfer(dst, src, "reg_store", false)?,

        Stmt::Fence { .. } => {}
        Stmt::CtaSync | Stmt::WarpSync | Stmt::ClusterSync => {}
        Stmt::ClusterBarrierArrive { .. } | Stmt::ClusterBarrierWait => {}
        Stmt::WgSync { barrier_id } => {
            if *barrier_id < 1 || *barrier_id > 15 {
                return bail("wg_sync barrier_id must be an integer in [1, 15]");
            }
        }
        Stmt::NamedBarrier {
            barrier_id,
            num_warps,
        } => {
            if *barrier_id < 1 || *barrier_id > 15 {
                return bail("named_barrier barrier_id must be an integer in [1, 15]");
            }
            // The thread count is `num_warps * 32` by construction — carrying the
            // WARP count in the IR is what keeps it a positive multiple of 32.
            if *num_warps < 1 {
                return bail("named_barrier num_warps must be >= 1");
            }
        }
    }
    // Recurse into nested bodies so every statement is validated.
    for body in s.child_bodies() {
        for st in body {
            validate_stmt(st)?;
        }
    }
    Ok(())
}

/// Shared by `_check_reg_transfer_*` (RegLoad/RegStore differ only in which side
/// must be REG).
fn check_reg_transfer(dst: &TensorSlice, src: &TensorSlice, label: &str, load: bool) -> R {
    validate_slice(dst, &format!("{label} dst"))?;
    validate_slice(src, &format!("{label} src"))?;
    let is_transfer_space =
        |sp| matches!(sp, MemorySpace::Reg | MemorySpace::Smem | MemorySpace::Gmem);
    let (reg_side, other_side, other_name) = if load {
        (dst, src, "src")
    } else {
        (src, dst, "dst")
    };
    if reg_side.tensor.space != MemorySpace::Reg {
        return bail(format!("{label} reg side must be REG"));
    }
    if !is_transfer_space(other_side.tensor.space) {
        return bail(format!("{label} {other_name} must be REG, SMEM, or GMEM"));
    }
    if dst.tensor.dtype != src.tensor.dtype {
        return bail(format!("{label} dst and src dtype must match"));
    }
    if !is_reg_dtype(dst.tensor.dtype) {
        return bail(format!("{label} dtype must be f16, bf16, f32, i32, or u32"));
    }
    // shapes must match ignoring static singleton dims
    let drop1 = |sh: &[ScalarValue]| -> Vec<ScalarValue> {
        sh.iter()
            .filter(|d| static_int(d) != Some(1))
            .cloned()
            .collect()
    };
    if drop1(&dst.shape) != drop1(&src.shape) {
        return bail(format!(
            "{label} slice shapes must match except static singleton dimensions"
        ));
    }
    Ok(())
}

fn check_ld_atom(shape: LdStShape, num: u32, label: &str) -> R {
    if shape.register_count(num).is_none() {
        return bail(format!(
            "{label} shape/num must be one of 32x32b.x{{1,2,4,8,16,32,64,128}}, \
             16x32bx2.x{{1,2,4,8,16,32,64,128}}, \
             16x64b.x{{1,2,4,8,16,32,64,128}}, \
             16x128b.x{{1,2,4,8,16,32,64}}, or 16x256b.x{{1,2,4,8,16,32}}"
        ));
    }
    Ok(())
}

fn check_matrix_atom(shape: MatrixShape, num: u32, dtype: MatrixDType, label: &str) -> R {
    match (shape, num, dtype) {
        (MatrixShape::M8N8, 1 | 2 | 4, MatrixDType::B16) => Ok(()),
        _ => bail(format!(
            "{label} shape/num/type must be m8n8.x{{1,2,4}}.b16"
        )),
    }
}

fn validate_task_space(space: &TaskSpace) -> R {
    if space.grid.is_empty() {
        return bail("task_space grid must be non-empty");
    }
    if space.fields.len() != space.grid.len() {
        return bail("task_space fields must match grid rank");
    }
    if space.grid.iter().any(|d| *d == 0) {
        return bail("task_space grid dims must be positive");
    }
    if space.task_count().is_none() {
        return bail("task_space grid size overflows usize");
    }
    let mut seen = HashSet::new();
    for field in &space.fields {
        if field.is_empty() {
            return bail("task_space field names must be non-empty");
        }
        if !seen.insert(field) {
            return bail("task_space field names must be unique");
        }
    }
    Ok(())
}

fn validate_scheduler(scheduler: &Scheduler) -> R {
    validate_task_space(&scheduler.space)
}

// ---------------------------------------------------------------------------
// the three kernel-level walks (Python's `Kernel.__post_init__`)
// ---------------------------------------------------------------------------

/// Collect the `Var`s used in a value (scalar / slice / mbar-ref), for the
/// "defined before use" check (`_check_kernel_value_vars_defined`).
fn collect_vars(v: &ScalarValue, out: &mut Vec<Var>) {
    match v {
        ScalarValue::Var(var) => out.push(*var),
        ScalarValue::Expr(e) => e.args.iter().for_each(|a| collect_vars(a, out)),
        _ => {}
    }
}
fn slice_vars(s: &TensorSlice, out: &mut Vec<Var>) {
    s.offsets.iter().for_each(|v| collect_vars(v, out));
    s.shape.iter().for_each(|v| collect_vars(v, out));
}

fn require_defined(vars: &[Var], defined: &HashSet<Var>, label: &str) -> R {
    for v in vars {
        if !defined.contains(v) {
            return bail(format!("{label} var must be defined before use"));
        }
    }
    Ok(())
}
fn define_var(var: Var, defined: &mut HashSet<Var>) -> R {
    if !defined.insert(var) {
        return bail("var is defined more than once");
    }
    Ok(())
}

/// Walks 1 (var-defs) + 2 (thread-shape rules), threading the defined-set and
/// the static thread filter of the enclosing branch chain.
///
/// Shape rules fire only when the filter is `Known` — a branch on a runtime
/// value makes the reachable thread set indeterminate, and the runtime
/// semantics (rendezvous accounting, issue gates) own the check there. The
/// exceptions are ops with NO runtime backstop (`SetMaxNReg` is sim metadata;
/// a multi-warp `SchedNext` silently work-steals instead of failing): those
/// require a statically-resolvable branch.
/// Per-`barrier_id` static participant warp sets of the two hardware-barrier
/// classes (wg_sync vs named_barrier). The 16 hardware named barriers are ONE
/// resource shared by both classes: a warp whose arrivals count toward a
/// wg_sync(id) AND a named_barrier(id) folds two different rendezvous into one
/// hardware barrier. The rule keys on the statically-known WARP SETS: the
/// warps reaching wg_sync(id) must be disjoint from the warps reaching
/// named_barrier(id). Disjoint reuse (one warpgroup's private wg_sync(1) next
/// to a named_barrier(1) among OTHER warpgroups) stays legal, as on hardware.
/// Unknown filters skip, like the other static shape rules.
#[derive(Default)]
struct BarrierClassUse {
    wg_sync_warps: HashMap<u32, BTreeSet<u32>>,
    named_warps: HashMap<u32, BTreeSet<u32>>,
}

#[allow(clippy::too_many_arguments)]
fn check_context(
    body: &[Stmt],
    filter: &ThreadFilter,
    single_lane: bool,
    defined: &mut HashSet<Var>,
    num_warps: u32,
    in_scheduler_impl: bool,
    scheduler_loop_depth: usize,
    barriers: &mut BarrierClassUse,
) -> R {
    for stmt in body {
        match stmt {
            Stmt::CtaSync | Stmt::ClusterSync => {
                if let Some(set) = filter.known() {
                    if !set.is_full_cta() {
                        let name = if matches!(stmt, Stmt::CtaSync) {
                            "cta_sync"
                        } else {
                            "cluster_sync"
                        };
                        return bail(format!(
                            "{name} must be reachable by every thread of the CTA"
                        ));
                    }
                }
            }
            Stmt::WgSync { barrier_id } => {
                if let Some(set) = filter.known() {
                    if set.is_exactly_one_full_warpgroup().is_none() {
                        return bail("wg_sync must cover exactly one full warpgroup");
                    }
                    let warps = barriers.wg_sync_warps.entry(*barrier_id).or_default();
                    warps.extend(set.warps_touched());
                    // The 16 hardware named barriers are a CTA-wide resource; a
                    // wg_sync is a warpgroup-scoped rendezvous, so one barrier_id
                    // belongs to a single warpgroup. Two warpgroups on the same
                    // id would collide — a cross-warpgroup rendezvous is a
                    // named_barrier.
                    let warpgroups: std::collections::BTreeSet<u32> =
                        warps.iter().map(|w| w / 4).collect();
                    if warpgroups.len() > 1 {
                        return bail(
                            "wg_sync barrier_id is used by more than one warpgroup \
                             (each of the 16 hardware barriers belongs to one warpgroup; \
                             a cross-warpgroup rendezvous is a named_barrier)",
                        );
                    }
                    if let Some(named) = barriers.named_warps.get(barrier_id) {
                        if warps.intersection(named).next().is_some() {
                            return bail(
                                "wg_sync barrier_id cannot alias a named_barrier reachable \
                                 by the same warp (the 16 hardware barriers are one resource)",
                            );
                        }
                    }
                }
            }
            Stmt::NamedBarrier { barrier_id, .. } => {
                if let Some(set) = filter.known() {
                    // A statically sub-warp participant can never fill a
                    // count-based bar.sync — the hardware hang.
                    let touched = set.warps_touched();
                    if touched.is_empty() || !touched.iter().all(|&w| set.is_full_warp(w)) {
                        return bail(
                            "named_barrier must cover whole warps (a sub-warp participant \
                             deadlocks)",
                        );
                    }
                    let warps = barriers.named_warps.entry(*barrier_id).or_default();
                    warps.extend(touched);
                    if let Some(wg) = barriers.wg_sync_warps.get(barrier_id) {
                        if warps.intersection(wg).next().is_some() {
                            return bail(
                                "named_barrier barrier_id cannot alias a wg_sync reachable \
                                 by the same warp (the 16 hardware barriers are one resource)",
                            );
                        }
                    }
                }
            }
            Stmt::WarpSync => {
                if let Some(set) = filter.known() {
                    let touched = set.warps_touched();
                    if touched.is_empty() || !touched.iter().all(|&w| set.is_full_warp(w)) {
                        return bail(
                            "warp_sync must cover whole warps (a sub-warp barrier deadlocks)",
                        );
                    }
                }
            }
            Stmt::TmemAlloc { .. } | Stmt::TmemDealloc { .. } | Stmt::TmemRelinquish { .. } => {
                if let Some(set) = filter.known() {
                    if set.is_exactly_one_full_warp().is_none() {
                        return bail(
                            "tmem alloc/dealloc/relinquish must be issued by exactly one full warp",
                        );
                    }
                }
            }
            Stmt::ShuffleSync { .. } => {
                // `__shfl_sync(0xffffffff, ...)` needs the whole warp converged:
                // inside a sub-warp (elected / partial-lane) region the full
                // mask names lanes that are not converged there — hardware UB.
                // (Warp-model form of #18's Scope::Single elected-scope ban.)
                if let Some(set) = filter.known() {
                    let touched = set.warps_touched();
                    if touched.is_empty() || !touched.iter().all(|&w| set.is_full_warp(w)) {
                        return bail(
                            "shuffle_sync must cover whole warps (a full-mask shfl in a                              sub-warp (elected) scope is hardware UB)",
                        );
                    }
                }
            }
            Stmt::SetMaxNReg { .. } => match filter.known() {
                None => {
                    return bail("setmaxnreg requires a statically-resolvable thread branch");
                }
                Some(set) => {
                    // setmaxnreg is a .sync.aligned warp collective: the branch must
                    // cover whole warps (fi's GDN emits per-warp budgets: 224/256 for
                    // the compute warpgroups, 24 for the single support warps).
                    let touched = set.warps_touched();
                    if touched.is_empty() || !touched.iter().all(|&w| set.is_full_warp(w)) {
                        return bail("setmaxnreg must cover whole warp(s)");
                    }
                }
            },
            _ => {}
        }
        match stmt {
            Stmt::ScalarDef { var, initial } => {
                let mut vars = Vec::new();
                match initial {
                    ScalarInitial::Value(v) => collect_vars(v, &mut vars),
                    ScalarInitial::Tensor(t) => slice_vars(t, &mut vars),
                }
                require_defined(&vars, defined, "scalar_def initial")?;
                define_var(*var, defined)?;
            }
            Stmt::ScalarStore { var, value } => {
                require_defined(&[*var], defined, "scalar_store")?;
                let mut vars = Vec::new();
                collect_vars(value, &mut vars);
                require_defined(&vars, defined, "scalar_store value")?;
            }
            Stmt::ScalarLet { var, value } => {
                let mut vars = Vec::new();
                collect_vars(value, &mut vars);
                require_defined(&vars, defined, "scalar_let value")?;
                define_var(*var, defined)?;
            }
            Stmt::ShuffleSync { var, src, src_lane } => {
                let mut vars = Vec::new();
                collect_vars(src, &mut vars);
                collect_vars(src_lane, &mut vars);
                require_defined(&vars, defined, "shuffle_sync src")?;
                define_var(*var, defined)?;
            }
            Stmt::StoreScalar { dst, value } => {
                let mut vars = Vec::new();
                slice_vars(dst, &mut vars);
                collect_vars(value, &mut vars);
                require_defined(&vars, defined, "store_scalar")?;
            }
            Stmt::RegFill { .. }
            | Stmt::RegUnary { .. }
            | Stmt::RegReduce { .. }
            | Stmt::RegAdd { .. }
            | Stmt::RegSub { .. }
            | Stmt::RegMul { .. }
            | Stmt::RegMax { .. }
            | Stmt::RegMin { .. }
            | Stmt::RegBitwise { .. }
            | Stmt::RegFma { .. }
            | Stmt::RegCondRescale { .. }
            | Stmt::RegSoftmaxRescale { .. }
            | Stmt::RegCausalMask { .. }
            | Stmt::RegCombineIntFracEx2 { .. }
            | Stmt::RegCvt { .. }
            | Stmt::RegLoad { .. }
            | Stmt::RegStore { .. } => {
                let mut vars = Vec::new();
                reg_stmt_vars(stmt, &mut vars);
                require_defined(&vars, defined, "reg operand")?;
            }
            Stmt::ForLoop {
                var,
                start,
                stop,
                step,
                body,
                ..
            } => {
                let mut vars = Vec::new();
                collect_vars(start, &mut vars);
                collect_vars(stop, &mut vars);
                collect_vars(step, &mut vars);
                require_defined(&vars, defined, "loop bound")?;
                define_var(*var, defined)?;
                check_context(
                    body,
                    filter,
                    single_lane,
                    defined,
                    num_warps,
                    in_scheduler_impl,
                    scheduler_loop_depth,
                    barriers,
                )?;
            }
            Stmt::ForEachTask { var, body, .. } => {
                define_var(*var, defined)?;
                check_context(
                    body,
                    filter,
                    single_lane,
                    defined,
                    num_warps,
                    in_scheduler_impl,
                    scheduler_loop_depth,
                    barriers,
                )?;
            }
            Stmt::SchedulerImpl { body, .. } => {
                check_context(
                    body,
                    filter,
                    single_lane,
                    defined,
                    num_warps,
                    true,
                    0,
                    barriers,
                )?;
            }
            Stmt::SchedNext { var, .. } => {
                if !in_scheduler_impl {
                    return bail("sched_next must be inside scheduler_impl");
                }
                // A sched_next reachable by more than one warp hands each warp
                // a DIFFERENT task (shared cursor) with no diagnostic — the
                // sim cannot catch it, so the branch must be statically known.
                match filter.known() {
                    None => {
                        return bail("sched_next requires a statically-resolvable thread branch");
                    }
                    Some(set) => {
                        if set.warps_touched().len() > 1 {
                            return bail(
                                "sched_next must be confined to a single warp (each warp \
                                 advancing the shared cursor takes a different task)",
                            );
                        }
                    }
                }
                define_var(*var, defined)?;
            }
            // ZERO-INFERENCE (user-mandated `single_issue_scope`): hardware
            // single-issue ops must sit under an EXPLICIT single-lane branch —
            // codegen never synthesizes a guard, so a bare site would issue
            // once per lane (32 duplicate TMA/MMA issues, or an mbarrier
            // double-init). The interpreter's `check_single_thread_issue` is
            // the runtime backstop for the async-issue ops; this rule rejects
            // at build. Provable means: the enclosing `If` chain statically
            // resolves to at most one lane per touched warp (the `if_elected`
            // sugar, `tid_in_wg == 0`, `thread_rank == 0`, ...).
            Stmt::MBarrierInit { .. }
            | Stmt::TmaLoad { .. }
            | Stmt::TmaStore { .. }
            | Stmt::CpAsyncBulkS2Cluster { .. }
            | Stmt::Tcgen05Mma { .. }
            | Stmt::Tcgen05Cp { .. }
            | Stmt::Tcgen05Commit { .. } => {
                require_single_lane_scope(single_lane, stmt)?;
            }
            Stmt::ClcTryCancel { stage, .. } => {
                if !in_scheduler_impl {
                    return bail("clc_try_cancel must be inside scheduler_impl");
                }
                require_single_lane_scope(single_lane, stmt)?;
                if let Some(s) = stage {
                    let mut vars = Vec::new();
                    collect_vars(s, &mut vars);
                    require_defined(&vars, defined, "clc_try_cancel stage")?;
                }
            }
            Stmt::ClcQueryCancel { var, .. } => {
                define_var(*var, defined)?;
            }
            Stmt::Loop { body } => {
                check_context(
                    body,
                    filter,
                    single_lane,
                    defined,
                    num_warps,
                    in_scheduler_impl,
                    scheduler_loop_depth + 1,
                    barriers,
                )?;
            }
            Stmt::BreakIf { cond } => {
                if scheduler_loop_depth == 0 {
                    return bail("break_if must be inside loop");
                }
                let mut vars = Vec::new();
                collect_vars(cond, &mut vars);
                require_defined(&vars, defined, "break_if condition")?;
            }
            Stmt::If { cond, then_body } => {
                let mut vars = Vec::new();
                collect_vars(cond, &mut vars);
                require_defined(&vars, defined, "if condition")?;
                let branch = static_thread_filter(cond, num_warps);
                let inner = match &branch {
                    ThreadFilter::Known(set) => narrow(filter, set),
                    ThreadFilter::Unknown => ThreadFilter::Unknown,
                };
                // The single-lane UPPER BOUND is sticky: once an enclosing
                // branch provably narrows to <=1 lane per warp, any deeper
                // branch (even a runtime one, e.g. `cta_rank == 0`) only
                // selects a subset, so a single-issue op still cannot
                // double-issue. This branch alone proving the bound (an
                // explicit if_elected, or an And-chain with a one-lane
                // operand like `(~kept) & (tid == 0)`) lifts it for its
                // subtree.
                let inner_single = single_lane || proves_single_lane_per_warp(cond, num_warps);
                check_context(
                    then_body,
                    &inner,
                    inner_single,
                    defined,
                    num_warps,
                    in_scheduler_impl,
                    scheduler_loop_depth,
                    barriers,
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// The `single_issue_scope` rule's scope test: the enclosing branch chain must
/// statically resolve to at most one lane per touched warp — the IR form of
/// "this op issues once" (the `if_elected` sugar / a one-lane predicate).
fn require_single_lane_scope(single_lane: bool, stmt: &Stmt) -> R {
    let op = match stmt {
        Stmt::MBarrierInit { .. } => "mbarrier_init",
        Stmt::TmaLoad { .. } => "tma_load",
        Stmt::TmaStore { .. } => "tma_store",
        Stmt::CpAsyncBulkS2Cluster { .. } => "cp_async_bulk_s2cluster",
        Stmt::Tcgen05Mma { .. } => "tcgen05_mma",
        Stmt::Tcgen05Cp { .. } => "tcgen05_cp",
        Stmt::Tcgen05Commit { .. } => "tcgen05_commit",
        Stmt::ClcTryCancel { .. } => "clc_try_cancel",
        _ => unreachable!("require_single_lane_scope on a non-single-issue op"),
    };
    if !single_lane {
        return bail(format!(
            "{op} is a hardware single-issue op: it must appear under an explicit \
             single-lane branch (the if_elected sugar or a provably \
             one-lane-per-warp predicate) — codegen never infers the guard \
             (single_issue_scope)"
        ));
    }
    Ok(())
}

/// Intersect the enclosing filter with a statically-known branch set. An
/// Unknown enclosing filter stays Unknown: the true set is SOME subset of the
/// branch set, which is not enough for the exact-shape rules above.
fn narrow(outer: &ThreadFilter, branch: &ThreadSet) -> ThreadFilter {
    match outer {
        ThreadFilter::Known(set) => ThreadFilter::Known(set.intersect(branch)),
        ThreadFilter::Unknown => ThreadFilter::Unknown,
    }
}

/// The kernel-level tcgen05 engine group, derived from the cluster size the
/// same way codegen's `ctx.cta_group` is — cluster-scope ops must carry
/// exactly this value.
fn kernel_cta_group(kernel: &Kernel) -> u8 {
    kernel.cluster_shape.iter().product::<usize>().max(1) as u8
}

/// Walk 3: `_check_tcgen05_cta_group_consistency`.
fn check_cta_group_consistency(kernel: &Kernel) -> R {
    let kernel_group = kernel_cta_group(kernel);
    let mut group: Option<u8> = None;
    fn walk(body: &[Stmt], group: &mut Option<u8>, kernel_group: u8) -> R {
        for s in body {
            // The CLC multicast width must be the kernel-level engine group —
            // the same fail-closed rule as the TMEM lifecycle ops in walk 4.
            // The field has no codegen emission site (TIRx's clc lowering
            // implies it), so an unchecked mismatch would validate one
            // semantics and run another.
            if let Stmt::ClcTryCancel { cta_group, .. } = s {
                if *cta_group != kernel_group {
                    return bail(format!(
                        "clc_try_cancel cta_group={cta_group} != kernel cta_group={kernel_group}"
                    ));
                }
            }
            let g = match s {
                Stmt::TmemAlloc { cta_group, .. }
                | Stmt::TmemDealloc { cta_group, .. }
                | Stmt::TmemRelinquish { cta_group }
                | Stmt::Tcgen05Mma { cta_group, .. }
                | Stmt::Tcgen05Cp { cta_group, .. }
                | Stmt::Tcgen05Commit { cta_group, .. } => Some(*cta_group),
                _ => None,
            };
            if let Some(g) = g {
                match group {
                    None => *group = Some(g),
                    Some(existing) if *existing != g => {
                        return bail("tcgen05 cta_group must be consistent across kernel");
                    }
                    _ => {}
                }
            }
            for child in s.child_bodies() {
                walk(child, group, kernel_group)?;
            }
        }
        Ok(())
    }
    walk(&kernel.body, &mut group, kernel_group)
}

/// Walk 4: TMEM allocation bands. TMEM is not a tensor: an allocation declares
/// a column band `[base_col, base_col + n_cols)` and every TMEM operand is an
/// absolute physical (lane, col) address whose extent the op implies. With
/// constant addresses this statically proves: (a) a dealloc matches a live
/// band, (b) every operand's column span lands inside a band that is live at
/// that program point. The interpreter re-checks both on the evaluated
/// addresses at run time — this walk only rejects what is statically provably
/// wrong.
///
/// The lifecycle itself is deliberately narrower than raw PTX, because the
/// codegen lowers the whole TMEM band as ONE base-0 view (`decl_buffer(...,
/// allocated_addr=0)`) fed by a single `tcgen05.alloc`; anything outside that
/// shape would validate one semantics and run another:
///   * every `TmemAlloc` has `base_col == 0`,
///   * at most one allocation is live at any program point (the next
///     `TmemAlloc` requires a matching `TmemDealloc` first),
///   * every alloc/dealloc/relinquish carries the kernel-level cta_group
///     (derived from the cluster size, mirroring the per-op checks codegen
///     runs against `ctx.cta_group` — commit 76600421), and
///   * no `TmemAlloc` follows a `TmemRelinquish` (PTX §9.7.17.7.1: the permit
///     is gone for the rest of the kernel; the interpreter enforces the same
///     rule per CTA at run time).
///
/// Placement (warp-model adaptation of #18's "top-level Role body" rule):
/// lifecycle ops are banned inside any re-executing body (ForLoop / Loop /
/// ForEachTask / SchedulerImpl) and inside an `If` whose predicate is NOT
/// statically resolvable (a runtime conditional). The one-pass band walk is
/// unsound in both: it visits the body once, so a `for_loop(stop=2)` alloc +
/// outside dealloc passed build but double-allocated on the second iteration
/// (`tmem_already_allocated`), and a runtime-conditional alloc may never
/// execute at all. A statically-known `If` (warp/lane dispatch — the
/// warp-model execution model's role branch) executes its body exactly once
/// per covered thread, so it does NOT taint.
fn check_tmem_alloc_bands(kernel: &Kernel) -> R {
    /// One TMEM operand use with its static (lane, column) extent, when known.
    struct Use<'a> {
        addr: &'a TmemAddr,
        max_lane_offset: Option<u32>,
        cols: Option<u32>,
        label: &'static str,
    }

    fn tmem_uses(s: &Stmt) -> Result<Vec<Use<'_>>, IrError> {
        let mut uses = Vec::new();
        match s {
            Stmt::Tcgen05Mma {
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
                ..
            } => {
                let resolved = resolve_tcgen05_mma(
                    dst,
                    a,
                    b,
                    *mma_m,
                    *mma_n,
                    *format,
                    block_scale.as_ref(),
                    *trans_a,
                    *trans_b,
                    *ws,
                    *cta_group,
                )
                .map_err(|error| err(error.message))?;
                uses.push(Use {
                    addr: dst,
                    max_lane_offset: Some(resolved.d.max_lane_offset),
                    cols: Some(resolved.d.physical_cols),
                    label: "tcgen05_mma dst",
                });
                if let (MmaAOperand::Tmem { addr, .. }, Some(footprint)) = (a, resolved.a_tmem) {
                    uses.push(Use {
                        addr,
                        max_lane_offset: Some(footprint.max_lane_offset),
                        cols: Some(footprint.physical_cols),
                        label: "tcgen05_mma a",
                    });
                }
                if let Some(scale) = block_scale {
                    for (addr, footprint, label) in [
                        (
                            &scale.sfa,
                            resolved.sfa.expect("resolved block-scale SFA").footprint,
                            "tcgen05_mma sfa",
                        ),
                        (
                            &scale.sfb,
                            resolved.sfb.expect("resolved block-scale SFB").footprint,
                            "tcgen05_mma sfb",
                        ),
                    ] {
                        uses.push(Use {
                            addr,
                            max_lane_offset: Some(footprint.max_lane_offset),
                            cols: Some(footprint.physical_cols),
                            label,
                        });
                    }
                }
            }
            Stmt::Tcgen05Cp {
                dst,
                src,
                shape,
                multicast,
                cta_group,
            } => {
                let resolved = resolve_tcgen05_cp(dst, src, *shape, *multicast, *cta_group)
                    .map_err(|error| err(error.message))?;
                uses.push(Use {
                    addr: dst,
                    max_lane_offset: Some(resolved.target.max_lane_offset),
                    cols: Some(resolved.target.physical_cols),
                    label: "tcgen05_cp dst",
                });
            }
            Stmt::Tcgen05Ld {
                src, shape, num, ..
            } => uses.push(Use {
                addr: src,
                max_lane_offset: None,
                cols: tmem_operand_lanes_cols(shape, *num).and_then(|cols| cols.try_into().ok()),
                label: "tcgen05_ld src",
            }),
            Stmt::Tcgen05St {
                dst, shape, num, ..
            } => uses.push(Use {
                addr: dst,
                max_lane_offset: None,
                cols: tmem_operand_lanes_cols(shape, *num).and_then(|cols| cols.try_into().ok()),
                label: "tcgen05_st dst",
            }),
            _ => {}
        }
        Ok(uses)
    }

    // Every TMEM lifecycle op must carry the kernel-level engine group
    // (derived from the cluster size, mirroring codegen's `ctx.cta_group`).
    let kernel_group = kernel_cta_group(kernel);
    let num_warps = kernel.num_warps;

    fn walk(
        stmts: &[Stmt],
        live: &mut Vec<(i64, i64)>,
        relinquished: &mut bool,
        kernel_cta_group: u8,
        num_warps: u32,
        lifecycle_banned: bool,
    ) -> R {
        for s in stmts {
            let banned_here = || {
                lifecycle_banned.then(|| {
                    "tmem lifecycle ops (alloc/dealloc/relinquish) are not allowed inside                      a loop or scheduler body (ForLoop/Loop/ForEachTask/SchedulerImpl) or a                      runtime-value conditional"
                })
            };
            match s {
                Stmt::TmemAlloc {
                    base_col,
                    n_cols,
                    cta_group,
                    addr_byte_offset: _,
                } => {
                    if let Some(msg) = banned_here() {
                        return bail(msg);
                    }
                    if *cta_group != kernel_cta_group {
                        return bail(format!(
                            "tmem_alloc cta_group={cta_group} != kernel cta_group={kernel_cta_group}"
                        ));
                    }
                    // PTX §9.7.17.7.1: the permit is gone for the rest of the
                    // kernel once relinquished.
                    if *relinquished {
                        return bail("tmem_alloc after tmem_relinquish_alloc_permit");
                    }
                    // The generated code bases the single TMEM view at column 0;
                    // a nonzero base would be silently dropped there.
                    if *base_col != 0 {
                        return bail("tmem_alloc base_col must be 0");
                    }
                    // One live band at a time — a second alloc would alias the
                    // same single view in the generated code. (With base_col==0
                    // enforced, two live bands always overlap; this check just
                    // says so directly.)
                    if !live.is_empty() {
                        return bail(
                            "tmem_alloc while another allocation is still live (dealloc it first)",
                        );
                    }
                    live.push((i64::from(*base_col), i64::from(*n_cols)));
                }
                Stmt::TmemDealloc {
                    base_col,
                    n_cols,
                    cta_group,
                } => {
                    if let Some(msg) = banned_here() {
                        return bail(msg);
                    }
                    if *cta_group != kernel_cta_group {
                        return bail(format!(
                            "tmem_dealloc cta_group={cta_group} != kernel cta_group={kernel_cta_group}"
                        ));
                    }
                    let band = (i64::from(*base_col), i64::from(*n_cols));
                    let Some(pos) = live.iter().position(|&x| x == band) else {
                        return bail("tmem_dealloc does not match a live allocation");
                    };
                    live.remove(pos);
                }
                Stmt::TmemRelinquish { cta_group } => {
                    if let Some(msg) = banned_here() {
                        return bail(msg);
                    }
                    if *cta_group != kernel_cta_group {
                        return bail(format!(
                            "tmem_relinquish cta_group={cta_group} != kernel cta_group={kernel_cta_group}"
                        ));
                    }
                    // Idempotent to give up (a second relinquish is a no-op);
                    // what is illegal is a later alloc — see the alloc arm.
                    *relinquished = true;
                }
                _ => {}
            }
            for u in tmem_uses(s)? {
                if let (Some(r), Some(max_lane_offset)) =
                    (static_int(&u.addr.row), u.max_lane_offset)
                {
                    if r < 0 || r + i64::from(max_lane_offset) >= 128 {
                        return bail(format!("{} lane span escapes the 128 TMEM lanes", u.label));
                    }
                }
                if let (Some(c), Some(cols)) = (static_int(&u.addr.col), u.cols) {
                    let c = i64::from(u.addr.tensor.start_col)
                        .checked_add(c)
                        .ok_or_else(|| err(format!("{} column address overflows i64", u.label)))?;
                    let cols = i64::from(cols);
                    if cols > 0 && !live.iter().any(|&(b, n)| b <= c && c + cols <= b + n) {
                        return bail(format!(
                            "{} column span is not inside a live tmem allocation band",
                            u.label
                        ));
                    }
                }
            }
            // The taint propagates downward (a loop body re-executes; a runtime
            // conditional may not execute at all — the one-pass walk is unsound
            // for both). A statically-known warp/lane `If` is the warp-model
            // dispatch branch: it executes its body exactly once per covered
            // thread, so it does not taint.
            let child_ban = lifecycle_banned
                || matches!(
                    s,
                    Stmt::ForLoop { .. }
                        | Stmt::Loop { .. }
                        | Stmt::ForEachTask { .. }
                        | Stmt::SchedulerImpl { .. }
                )
                || matches!(s, Stmt::If { cond, .. }
                    if !matches!(static_thread_filter(cond, num_warps), ThreadFilter::Known(_)));
            for body in s.child_bodies() {
                walk(
                    body,
                    live,
                    relinquished,
                    kernel_cta_group,
                    num_warps,
                    child_ban,
                )?;
            }
        }
        Ok(())
    }

    let mut live = Vec::new();
    let mut relinquished = false;
    walk(
        &kernel.body,
        &mut live,
        &mut relinquished,
        kernel_group,
        num_warps,
        false,
    )
}

/// A `ScalarLet` var is single-assignment: reject any `ScalarStore` targeting it
/// (the immutable SSA binding is what lets ptxas keep the value on the uniform
/// datapath; a store would reintroduce the mutable-local form that breaks it).
fn check_let_single_assignment(body: &[Stmt]) -> R {
    fn walk(stmts: &[Stmt], let_vars: &mut HashSet<Var>, stores: &mut Vec<Var>) {
        for stmt in stmts {
            match stmt {
                Stmt::ScalarLet { var, .. } => {
                    let_vars.insert(*var);
                }
                Stmt::ScalarStore { var, .. } => stores.push(*var),
                _ => {}
            }
            for child in stmt.child_bodies() {
                walk(child, let_vars, stores);
            }
        }
    }
    let mut let_vars = HashSet::new();
    let mut stores = Vec::new();
    walk(body, &mut let_vars, &mut stores);
    if stores.iter().any(|v| let_vars.contains(v)) {
        return bail("scalar_store cannot write a let-bound var (single assignment)");
    }
    Ok(())
}

fn check_smem_layout(kernel: &Kernel) -> R {
    // Tensor ranges are explicit aliases and may overlap freely. Metadata
    // cells (mbarriers and the tcgen05 allocation-address cell) may also reuse
    // those bytes, but only when a structured lexical live-range analysis can
    // prove that the lifetimes are disjoint. Unknown cross-branch ordering is
    // deliberately not rejected.
    fn tensor_codegen_alignment(tensor: &Tensor) -> usize {
        if matches!(tensor.layout, Some(Layout::Swizzle(_))) {
            return 1024;
        }
        1
    }

    fn check_tensor(tensor: &Tensor, smem_size_bytes: usize) -> R {
        if tensor.space != MemorySpace::Smem {
            return Ok(());
        }
        let offset = tensor
            .byte_offset
            .ok_or_else(|| err("smem tensor byte_offset is required"))?;
        let extent = smem_extent_bytes(tensor)?;
        let end = offset
            .checked_add(extent)
            .ok_or_else(|| err("smem tensor byte range overflows usize"))?;
        if end > smem_size_bytes {
            return bail("smem tensor byte range exceeds kernel smem_size_bytes");
        }
        let alignment = tensor_codegen_alignment(tensor);
        if offset % alignment != 0 {
            return bail(format!(
                "smem tensor byte_offset {offset} must be aligned to its codegen \
                 alignment {alignment}"
            ));
        }
        Ok(())
    }

    fn walk_stmt(
        stmt: &Stmt,
        smem_size_bytes: usize,
        seen_tensors: &mut HashSet<u32>,
        seen_mbars: &mut HashSet<u32>,
        tmem_addr_offset: &mut Option<usize>,
    ) -> R {
        match stmt {
            Stmt::TensorDef { tensor } => {
                if seen_tensors.insert(tensor.id) {
                    check_tensor(tensor, smem_size_bytes)?;
                }
            }
            Stmt::MBarDef { mbar } => {
                if seen_mbars.insert(mbar.id) {
                    if mbar.byte_offset % 8 != 0 {
                        return bail(format!(
                            "mbar {} byte_offset {} must be 8-byte aligned",
                            mbar.id, mbar.byte_offset
                        ));
                    }
                    let extent = (mbar.stages as usize)
                        .checked_mul(8)
                        .ok_or_else(|| err("mbar byte extent overflows usize"))?;
                    let end = mbar
                        .byte_offset
                        .checked_add(extent)
                        .ok_or_else(|| err("mbar byte range overflows usize"))?;
                    if end > smem_size_bytes {
                        return bail("mbar byte range exceeds kernel smem_size_bytes");
                    }
                }
            }
            Stmt::TmemAlloc {
                addr_byte_offset, ..
            } => {
                if *addr_byte_offset % 4 != 0 {
                    return bail(format!(
                        "tmem_addr byte_offset {addr_byte_offset} must be 4-byte aligned"
                    ));
                }
                let end = addr_byte_offset
                    .checked_add(4)
                    .ok_or_else(|| err("tmem_addr byte range overflows usize"))?;
                if end > smem_size_bytes {
                    return bail("tmem_addr byte range exceeds kernel smem_size_bytes");
                }
                match tmem_addr_offset {
                    Some(existing) if *existing != *addr_byte_offset => {
                        return bail("all TmemAlloc statements must use the same addr_byte_offset");
                    }
                    None => *tmem_addr_offset = Some(*addr_byte_offset),
                    _ => {}
                }
            }
            Stmt::ScalarDef {
                initial: ScalarInitial::Tensor(slice),
                ..
            } => {
                if seen_tensors.insert(slice.tensor.id) {
                    check_tensor(&slice.tensor, smem_size_bytes)?;
                }
            }
            Stmt::TmaLoad { dst, src, .. } => {
                if seen_tensors.insert(dst.tensor.id) {
                    check_tensor(&dst.tensor, smem_size_bytes)?;
                }
                if seen_tensors.insert(src.id) {
                    check_tensor(src, smem_size_bytes)?;
                }
            }
            Stmt::TmaStore { dst, src, .. } => {
                if seen_tensors.insert(dst.id) {
                    check_tensor(dst, smem_size_bytes)?;
                }
                if seen_tensors.insert(src.tensor.id) {
                    check_tensor(&src.tensor, smem_size_bytes)?;
                }
            }
            Stmt::Tcgen05Mma { a, b, .. } => {
                if let MmaAOperand::Smem(tile) = a {
                    if seen_tensors.insert(tile.tensor.id) {
                        check_tensor(&tile.tensor, smem_size_bytes)?;
                    }
                }
                if seen_tensors.insert(b.tensor.id) {
                    check_tensor(&b.tensor, smem_size_bytes)?;
                }
            }
            Stmt::Tcgen05Cp { src, .. } => {
                if seen_tensors.insert(src.tensor.id) {
                    check_tensor(&src.tensor, smem_size_bytes)?;
                }
            }
            Stmt::Tcgen05Ld { dst, .. } => {
                if seen_tensors.insert(dst.tensor.id) {
                    check_tensor(&dst.tensor, smem_size_bytes)?;
                }
            }
            Stmt::Tcgen05St { src, .. } => {
                if seen_tensors.insert(src.tensor.id) {
                    check_tensor(&src.tensor, smem_size_bytes)?;
                }
            }
            Stmt::ClcTryCancel { handle, .. } | Stmt::ClcQueryCancel { handle, .. } => {
                if seen_tensors.insert(handle.id) {
                    check_tensor(handle, smem_size_bytes)?;
                }
            }
            Stmt::LdMatrix { dst, src, .. } | Stmt::StMatrix { dst, src, .. } => {
                for tensor in [&dst.tensor, &src.tensor] {
                    if seen_tensors.insert(tensor.id) {
                        check_tensor(tensor, smem_size_bytes)?;
                    }
                }
            }
            Stmt::WarpMma { d, a, b, c, .. } => {
                for sl in [d, a, b, c] {
                    if seen_tensors.insert(sl.tensor.id) {
                        check_tensor(&sl.tensor, smem_size_bytes)?;
                    }
                }
            }
            Stmt::RegFill { .. }
            | Stmt::RegUnary { .. }
            | Stmt::RegReduce { .. }
            | Stmt::RegAdd { .. }
            | Stmt::RegSub { .. }
            | Stmt::RegMul { .. }
            | Stmt::RegMax { .. }
            | Stmt::RegMin { .. }
            | Stmt::RegBitwise { .. }
            | Stmt::RegFma { .. }
            | Stmt::RegCondRescale { .. }
            | Stmt::RegSoftmaxRescale { .. }
            | Stmt::RegCausalMask { .. }
            | Stmt::RegCombineIntFracEx2 { .. }
            | Stmt::RegCvt { .. }
            | Stmt::RegLoad { .. }
            | Stmt::RegStore { .. } => {
                let mut slices = Vec::new();
                reg_stmt_slices(stmt, &mut slices);
                for slice in slices {
                    if seen_tensors.insert(slice.tensor.id) {
                        check_tensor(&slice.tensor, smem_size_bytes)?;
                    }
                }
            }
            Stmt::StoreScalar { dst, .. } => {
                if seen_tensors.insert(dst.tensor.id) {
                    check_tensor(&dst.tensor, smem_size_bytes)?;
                }
            }
            _ => {}
        }
        for child in stmt.child_bodies() {
            for nested in child {
                walk_stmt(
                    nested,
                    smem_size_bytes,
                    seen_tensors,
                    seen_mbars,
                    tmem_addr_offset,
                )?;
            }
        }
        Ok(())
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum ObjectId {
        Tensor(u32),
        MBar(u32),
        TmemAddr,
    }

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct ByteObject {
        id: ObjectId,
        start: usize,
        end: usize,
        label: String,
    }

    #[derive(Clone, Default)]
    struct UseSummary {
        tensors: Vec<ByteObject>,
        mbars: Vec<ByteObject>,
    }

    #[derive(Clone)]
    struct TimedObject {
        object: ByteObject,
        first: usize,
        last: usize,
    }

    fn byte_ranges_overlap(lhs: &ByteObject, rhs: &ByteObject) -> bool {
        lhs.start < rhs.end && rhs.start < lhs.end
    }

    fn time_ranges_overlap(lhs: &TimedObject, rhs: &TimedObject) -> bool {
        lhs.first <= rhs.last && rhs.first <= lhs.last
    }

    fn push_unique(objects: &mut Vec<ByteObject>, object: ByteObject) {
        if !objects.iter().any(|existing| existing == &object) {
            objects.push(object);
        }
    }

    fn tensor_object(tensor: &Tensor) -> Result<Option<ByteObject>, IrError> {
        if tensor.space != MemorySpace::Smem {
            return Ok(None);
        }
        let start = tensor
            .byte_offset
            .ok_or_else(|| err("smem tensor byte_offset is required"))?;
        let extent = smem_extent_bytes(tensor)?;
        let end = start
            .checked_add(extent)
            .ok_or_else(|| err("smem tensor byte range overflows usize"))?;
        Ok(Some(ByteObject {
            id: ObjectId::Tensor(tensor.id),
            start,
            end,
            label: format!("smem tensor {}", tensor.id),
        }))
    }

    fn push_tensor(summary: &mut UseSummary, tensor: &Tensor) -> Result<(), IrError> {
        if let Some(object) = tensor_object(tensor)? {
            push_unique(&mut summary.tensors, object);
        }
        Ok(())
    }

    fn mbar_object(mbar: &MBarRef, stage: Option<&ScalarValue>) -> Result<ByteObject, IrError> {
        let (start, stages) = match stage.and_then(static_int) {
            Some(stage) if stage >= 0 && stage < i64::from(mbar.mbar.stages) => {
                let stage =
                    usize::try_from(stage).map_err(|_| err("mbar stage does not fit in usize"))?;
                let delta = stage
                    .checked_mul(8)
                    .ok_or_else(|| err("mbar stage byte offset overflows usize"))?;
                (
                    mbar.mbar
                        .byte_offset
                        .checked_add(delta)
                        .ok_or_else(|| err("mbar stage byte offset overflows usize"))?,
                    1usize,
                )
            }
            Some(_) => (
                mbar.mbar.byte_offset,
                usize::try_from(mbar.mbar.stages)
                    .map_err(|_| err("mbar stages do not fit in usize"))?,
            ),
            None if stage.is_some() => (
                mbar.mbar.byte_offset,
                usize::try_from(mbar.mbar.stages)
                    .map_err(|_| err("mbar stages do not fit in usize"))?,
            ),
            // A phase-less/stage-less instruction names slot zero, not every
            // slot of a staged barrier.
            None => (mbar.mbar.byte_offset, 1usize),
        };
        let bytes = stages
            .checked_mul(8)
            .ok_or_else(|| err("mbar byte extent overflows usize"))?;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| err("mbar byte range overflows usize"))?;
        Ok(ByteObject {
            id: ObjectId::MBar(mbar.mbar.id),
            start,
            end,
            label: format!("mbar {}", mbar.mbar.id),
        })
    }

    fn direct_uses(stmt: &Stmt) -> Result<UseSummary, IrError> {
        let mut summary = UseSummary::default();
        match stmt {
            Stmt::ScalarDef {
                initial: ScalarInitial::Tensor(slice),
                ..
            } => push_tensor(&mut summary, &slice.tensor)?,
            Stmt::StoreScalar { dst, .. } => push_tensor(&mut summary, &dst.tensor)?,
            Stmt::TmaLoad { dst, .. } => push_tensor(&mut summary, &dst.tensor)?,
            Stmt::TmaStore { src, .. } => push_tensor(&mut summary, &src.tensor)?,
            Stmt::CpAsyncBulkS2Cluster { dst, src, .. } => {
                push_tensor(&mut summary, &dst.tensor)?;
                push_tensor(&mut summary, &src.tensor)?;
            }
            Stmt::Tcgen05Mma { a, b, .. } => {
                if let MmaAOperand::Smem(tile) = a {
                    push_tensor(&mut summary, &tile.tensor)?;
                }
                push_tensor(&mut summary, &b.tensor)?;
            }
            Stmt::Tcgen05Cp { src, .. } => push_tensor(&mut summary, &src.tensor)?,
            Stmt::ClcTryCancel { handle, .. } | Stmt::ClcQueryCancel { handle, .. } => {
                push_tensor(&mut summary, handle)?
            }
            Stmt::LdMatrix { dst, src, .. } | Stmt::StMatrix { dst, src, .. } => {
                push_tensor(&mut summary, &dst.tensor)?;
                push_tensor(&mut summary, &src.tensor)?;
            }
            Stmt::RegFill { .. }
            | Stmt::RegUnary { .. }
            | Stmt::RegReduce { .. }
            | Stmt::RegAdd { .. }
            | Stmt::RegSub { .. }
            | Stmt::RegMul { .. }
            | Stmt::RegMax { .. }
            | Stmt::RegMin { .. }
            | Stmt::RegBitwise { .. }
            | Stmt::RegFma { .. }
            | Stmt::RegCondRescale { .. }
            | Stmt::RegSoftmaxRescale { .. }
            | Stmt::RegCausalMask { .. }
            | Stmt::RegCombineIntFracEx2 { .. }
            | Stmt::RegCvt { .. }
            | Stmt::RegLoad { .. }
            | Stmt::RegStore { .. } => {
                let mut slices = Vec::new();
                reg_stmt_slices(stmt, &mut slices);
                for slice in slices {
                    push_tensor(&mut summary, &slice.tensor)?;
                }
            }
            _ => {}
        }

        let mbar_use = match stmt {
            Stmt::MBarrierInit { mbar, stage, .. }
            | Stmt::MBarrierArrive { mbar, stage, .. }
            | Stmt::MBarrierWait { mbar, stage, .. }
            | Stmt::MBarrierExpectTx { mbar, stage, .. }
            | Stmt::MBarrierArriveExpectTx { mbar, stage, .. }
            | Stmt::Tcgen05Commit { mbar, stage, .. }
            | Stmt::ClcTryCancel { mbar, stage, .. } => Some((mbar, stage.as_ref())),
            Stmt::TmaLoad {
                mbar, mbar_stage, ..
            } => Some((mbar, mbar_stage.as_ref())),
            Stmt::CpAsyncBulkS2Cluster { mbar, .. } => Some((mbar, None)),
            _ => None,
        };
        if let Some((mbar, stage)) = mbar_use {
            push_unique(&mut summary.mbars, mbar_object(mbar, stage)?);
        }
        Ok(summary)
    }

    fn merge_summary(dst: &mut UseSummary, src: UseSummary) {
        for object in src.tensors {
            push_unique(&mut dst.tensors, object);
        }
        for object in src.mbars {
            push_unique(&mut dst.mbars, object);
        }
    }

    fn recursive_uses(stmt: &Stmt) -> Result<UseSummary, IrError> {
        let mut summary = direct_uses(stmt)?;
        for body in stmt.child_bodies() {
            for child in body {
                merge_summary(&mut summary, recursive_uses(child)?);
            }
        }
        Ok(summary)
    }

    fn conflict(lhs: &ByteObject, rhs: &ByteObject) -> R {
        bail(format!(
            "structured SMEM live ranges overlap: {} [{}, {}) and {} [{}, {})",
            lhs.label, lhs.start, lhs.end, rhs.label, rhs.start, rhs.end
        ))
    }

    fn check_structured_liveness(
        stmts: &[Stmt],
        inherited_tensors: &[ByteObject],
        inherited_metadata: &[ByteObject],
    ) -> R {
        let summaries = stmts
            .iter()
            .map(recursive_uses)
            .collect::<Result<Vec<_>, _>>()?;

        // A statement that simultaneously touches a tensor and a barrier is
        // an intrinsic overlap proof even when the barrier was initialized in
        // another structured branch (TMA and CLC are the common cases).
        for stmt in stmts {
            let direct = direct_uses(stmt)?;
            for tensor in &direct.tensors {
                for mbar in &direct.mbars {
                    if byte_ranges_overlap(tensor, mbar) {
                        return conflict(tensor, mbar);
                    }
                }
            }
        }

        let mut tensor_intervals: HashMap<u32, TimedObject> = HashMap::new();
        for (index, summary) in summaries.iter().enumerate() {
            for tensor in &summary.tensors {
                let ObjectId::Tensor(id) = tensor.id else {
                    unreachable!("tensor summary contains only tensors");
                };
                tensor_intervals
                    .entry(id)
                    .and_modify(|interval| interval.last = index)
                    .or_insert_with(|| TimedObject {
                        object: tensor.clone(),
                        first: index,
                        last: index,
                    });
            }
        }
        let tensors = tensor_intervals.into_values().collect::<Vec<_>>();

        let mut metadata = Vec::new();
        for (index, stmt) in stmts.iter().enumerate() {
            if let Stmt::MBarrierInit { mbar, stage, .. } = stmt {
                let object = mbar_object(mbar, stage.as_ref())?;
                let last = summaries
                    .iter()
                    .enumerate()
                    .skip(index)
                    .filter(|(_, summary)| {
                        summary.mbars.iter().any(|other| {
                            other.id == object.id && byte_ranges_overlap(other, &object)
                        })
                    })
                    .map(|(position, _)| position)
                    .last()
                    .unwrap_or(index);
                metadata.push(TimedObject {
                    object,
                    first: index,
                    last,
                });
            }
            if let Stmt::TmemAlloc {
                base_col,
                n_cols,
                cta_group,
                addr_byte_offset,
            } = stmt
            {
                let last =
                    stmts
                        .iter()
                        .enumerate()
                        .skip(index + 1)
                        .find_map(|(position, candidate)| match candidate {
                            Stmt::TmemDealloc {
                                base_col: dealloc_base,
                                n_cols: dealloc_cols,
                                cta_group: dealloc_group,
                            } if dealloc_base == base_col
                                && dealloc_cols == n_cols
                                && dealloc_group == cta_group =>
                            {
                                Some(position)
                            }
                            _ => None,
                        });
                if let Some(last) = last {
                    metadata.push(TimedObject {
                        object: ByteObject {
                            id: ObjectId::TmemAddr,
                            start: *addr_byte_offset,
                            end: addr_byte_offset + 4,
                            label: "tmem allocation-address cell".to_string(),
                        },
                        first: index,
                        last,
                    });
                }
            }
        }

        for meta in &metadata {
            for tensor in &tensors {
                if time_ranges_overlap(meta, tensor)
                    && byte_ranges_overlap(&meta.object, &tensor.object)
                {
                    return conflict(&meta.object, &tensor.object);
                }
            }
            for tensor in inherited_tensors {
                if byte_ranges_overlap(&meta.object, tensor) {
                    return conflict(&meta.object, tensor);
                }
            }
            for other in inherited_metadata {
                if meta.object.id != other.id && byte_ranges_overlap(&meta.object, other) {
                    return conflict(&meta.object, other);
                }
            }
        }
        for tensor in &tensors {
            for meta in inherited_metadata {
                if byte_ranges_overlap(&tensor.object, meta) {
                    return conflict(&tensor.object, meta);
                }
            }
        }
        for (index, lhs) in metadata.iter().enumerate() {
            for rhs in &metadata[index + 1..] {
                if lhs.object.id != rhs.object.id
                    && time_ranges_overlap(lhs, rhs)
                    && byte_ranges_overlap(&lhs.object, &rhs.object)
                {
                    return conflict(&lhs.object, &rhs.object);
                }
            }
        }

        for (index, stmt) in stmts.iter().enumerate() {
            let mut child_tensors = inherited_tensors.to_vec();
            child_tensors.extend(
                tensors
                    .iter()
                    // A range used only inside this child has its precise
                    // ordering analyzed recursively.  Inherit only ranges
                    // proven live across the whole child by uses on both
                    // sides; treating the child as one coarse event would
                    // otherwise reject legal sequential reuse inside it.
                    .filter(|interval| interval.first < index && index < interval.last)
                    .map(|interval| interval.object.clone()),
            );
            let mut child_metadata = inherited_metadata.to_vec();
            child_metadata.extend(
                metadata
                    .iter()
                    .filter(|interval| interval.first <= index && index <= interval.last)
                    .map(|interval| interval.object.clone()),
            );
            for body in stmt.child_bodies() {
                check_structured_liveness(body, &child_tensors, &child_metadata)?;
            }
        }
        Ok(())
    }

    let mut seen_tensors = HashSet::new();
    for tensor in &kernel.args {
        if seen_tensors.insert(tensor.id) {
            check_tensor(tensor, kernel.smem_size_bytes)?;
        }
    }
    let mut seen_mbars = HashSet::new();
    let mut tmem_addr_offset = None;
    for stmt in &kernel.body {
        walk_stmt(
            stmt,
            kernel.smem_size_bytes,
            &mut seen_tensors,
            &mut seen_mbars,
            &mut tmem_addr_offset,
        )?;
    }
    check_structured_liveness(&kernel.body, &[], &[])
}

impl Kernel {
    /// Validate the whole kernel — the faithful port of every `ir.py` check.
    pub fn validate(&self) -> R {
        check_num_warps(self.num_warps)?;
        check_execution_shape(&self.launch_shape, "launch_shape")?;
        check_execution_shape(&self.cluster_shape, "cluster_shape")?;
        if self.launch_shape.len() != self.cluster_shape.len() {
            return bail("kernel launch_shape and cluster_shape must have the same rank");
        }
        for (l, c) in self.launch_shape.iter().zip(self.cluster_shape.iter()) {
            if l % c != 0 {
                return bail("kernel launch_shape must be divisible by cluster_shape in every dim");
            }
        }
        for t in &self.args {
            validate_tensor(t)?;
        }
        for s in &self.body {
            validate_stmt(s)?;
        }
        check_smem_layout(self)?;
        let mut defined = HashSet::new();
        check_context(
            &self.body,
            &ThreadFilter::Known(ThreadSet::full(self.num_warps)),
            false,
            &mut defined,
            self.num_warps,
            false,
            0,
            &mut BarrierClassUse::default(),
        )?;
        check_cta_group_consistency(self)?;
        check_tmem_alloc_bands(self)?;
        check_let_single_assignment(&self.body)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// audit-rule tests (each fail-closed rule gets a negative and a positive)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn var(id: u32, binding: VarBinding) -> Var {
        Var {
            id: VarId(id),
            binding,
            dtype: ScalarDType::I32,
        }
    }

    fn tensor(id: u32, space: MemorySpace, dtype: DType, shape: Vec<usize>) -> Arc<Tensor> {
        Arc::new(Tensor {
            id,
            space,
            dtype,
            shape,
            layout: None,
            byte_offset: (space == MemorySpace::Smem).then_some(0),
        })
    }

    fn full_slice(t: &Arc<Tensor>) -> TensorSlice {
        TensorSlice {
            tensor: t.clone(),
            offsets: t.shape.iter().map(|_| ScalarValue::Int(0)).collect(),
            shape: t
                .shape
                .iter()
                .map(|&d| ScalarValue::Int(d as i64))
                .collect(),
        }
    }

    fn tmem_addr(start_col: u32, row: i64, col: i64) -> TmemAddr {
        TmemTensor { start_col }.at(ScalarValue::Int(row), ScalarValue::Int(col))
    }

    fn smem_tile(t: &Arc<Tensor>, rows: u32, cols: u32) -> SmemTile {
        SmemTile {
            tensor: t.clone(),
            prefix_indices: vec![],
            row_offset: ScalarValue::Int(0),
            col_offset: ScalarValue::Int(0),
            rows,
            cols,
        }
    }

    fn mbar(id: u32, kind: MBarKind) -> Arc<MBar> {
        Arc::new(MBar {
            id,
            kind,
            stages: 1,
            byte_offset: (1 << 19) + id as usize * 8,
            arrive_count: None,
        })
    }

    fn mbar_ref(m: &Arc<MBar>) -> MBarRef {
        MBarRef {
            mbar: m.clone(),
            remote_coord: None,
        }
    }

    fn kernel(body: Vec<Stmt>) -> Kernel {
        Kernel {
            name: "t".into(),
            args: vec![],
            body,
            num_warps: 4,
            smem_size_bytes: 1 << 20,
            launch_shape: vec![1],
            cluster_shape: vec![1],
        }
    }

    fn warp_if(w: i64, body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond: ScalarValue::expr(
                ScalarOp::Eq,
                vec![
                    ScalarValue::Scope(ScopeValueKind::WarpId),
                    ScalarValue::Int(w),
                ],
            ),
            then_body: body,
        }
    }

    fn elected_if(body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond: ScalarValue::Scope(ScopeValueKind::Elected),
            then_body: body,
        }
    }

    /// A plain dense cg1 m128 MMA over SMEM operands, accum literal.
    fn dense_mma(k: u32, accum: ScalarValue) -> Stmt {
        let a = tensor(1, MemorySpace::Smem, DType::F16, vec![128, k as usize]);
        let b = tensor(2, MemorySpace::Smem, DType::F16, vec![256, k as usize]);
        Stmt::Tcgen05Mma {
            dst: tmem_addr(0, 0, 0),
            a: MmaAOperand::Smem(smem_tile(&a, 128, k)),
            b: smem_tile(&b, 256, k),
            mma_m: 128,
            mma_n: 256,
            format: MmaElemFormat::F16,
            block_scale: None,
            accum,
            trans_a: false,
            trans_b: false,
            ws: false,
            cta_group: 1,
        }
    }

    /// TMEM lifecycle wrapper: alloc the band first so operands are covered.
    fn tmem_body(body: Vec<Stmt>) -> Vec<Stmt> {
        vec![
            warp_if(
                0,
                vec![Stmt::TmemAlloc {
                    base_col: 0,
                    n_cols: 512,
                    cta_group: 1,
                    addr_byte_offset: 900_000,
                }],
            ),
            warp_if(0, body),
        ]
    }

    #[test]
    fn dense_mma_full_k_accepts_multiples_of_16() {
        assert!(kernel(tmem_body(vec![elected_if(vec![dense_mma(
            16,
            ScalarValue::Int(0)
        )])]))
        .validate()
        .is_ok());
        // full-K: 3 atomic k=16 MMAs accumulated in order, issued as one IR op.
        assert!(kernel(tmem_body(vec![elected_if(vec![dense_mma(
            48,
            ScalarValue::Int(1)
        )])]))
        .validate()
        .is_ok());
    }

    #[test]
    fn dense_mma_rejects_non_multiple_k() {
        let e = kernel(tmem_body(vec![dense_mma(17, ScalarValue::Int(0))]))
            .validate()
            .unwrap_err();
        assert!(e.message.contains("multiple of mma_k=16"), "{}", e.message);
        // k=32 dense IS full-K (two k=16 atoms accumulated in order); the f8
        // block-scaled instruction at k=32 is gated by the sfa/sfb rules, not
        // the dense k rule.
        assert!(kernel(tmem_body(vec![elected_if(vec![dense_mma(
            32,
            ScalarValue::Int(1)
        )])]))
        .validate()
        .is_ok());
    }

    #[test]
    fn mma_accum_accepts_runtime_scalar_predicate() {
        // A runtime accum cell (the merged k-loop form) is IR-legal.
        let acc = var(1, VarBinding::Scalar);
        let body = tmem_body(vec![
            Stmt::ScalarDef {
                var: acc,
                initial: ScalarInitial::Value(ScalarValue::Int(0)),
            },
            elected_if(vec![dense_mma(48, ScalarValue::Var(acc))]),
        ]);
        assert!(kernel(body).validate().is_ok());
    }

    fn fp4_mma_new(trans_b: bool) -> Stmt {
        let a = tensor(20, MemorySpace::Smem, DType::U8, vec![128, 32]);
        let b_shape = if trans_b {
            vec![32, 128]
        } else {
            vec![128, 32]
        };
        let b = tensor(21, MemorySpace::Smem, DType::U8, b_shape.clone());
        Stmt::Tcgen05Mma {
            dst: tmem_addr(0, 0, 0),
            a: MmaAOperand::Smem(smem_tile(&a, 128, 32)),
            b: smem_tile(&b, b_shape[0] as u32, b_shape[1] as u32),
            mma_m: 128,
            mma_n: 128,
            format: MmaElemFormat::F4E2M1,
            block_scale: Some(BlockScaleSpec {
                sfa: tmem_addr(256, 0, 0),
                sfb: tmem_addr(264, 0, 0),
                sfa_k_offset: 0,
                sfb_k_offset: 0,
                scale_format: ScaleFormat::E4M3FN,
                sf_per_mma: 4,
                sf_reuse: 1,
            }),
            accum: ScalarValue::Int(1),
            trans_a: false,
            trans_b,
            ws: false,
            cta_group: 1,
        }
    }

    #[test]
    fn fp4_validation_uses_physical_resolver_rules() {
        assert!(
            kernel(tmem_body(vec![elected_if(vec![fp4_mma_new(false)])]))
                .validate()
                .is_ok()
        );
        let e = kernel(tmem_body(vec![elected_if(vec![fp4_mma_new(true)])]))
            .validate()
            .unwrap_err();
        assert!(
            e.message.contains("do not support transpose"),
            "{}",
            e.message
        );
    }

    fn tmem_a_mma(form: TmemAForm, start_col: u32) -> Stmt {
        let b = tensor(22, MemorySpace::Smem, DType::F16, vec![64, 32]);
        Stmt::Tcgen05Mma {
            dst: tmem_addr(0, 0, 0),
            a: MmaAOperand::Tmem {
                addr: tmem_addr(start_col, 0, 0),
                form,
            },
            b: smem_tile(&b, 64, 32),
            mma_m: 64,
            mma_n: 64,
            format: MmaElemFormat::F16,
            block_scale: None,
            accum: ScalarValue::Int(0),
            trans_a: false,
            trans_b: false,
            ws: true,
            cta_group: 1,
        }
    }

    #[test]
    fn tmem_a_form_and_absolute_band_are_validated() {
        assert!(kernel(tmem_body(vec![elected_if(vec![tmem_a_mma(
            TmemAForm::BankBatched,
            400
        )])]))
        .validate()
        .is_ok());

        let e = kernel(tmem_body(vec![elected_if(vec![tmem_a_mma(
            TmemAForm::Flat,
            400,
        )])]))
        .validate()
        .unwrap_err();
        assert!(e.message.contains("requires BankBatched"), "{}", e.message);

        let e = kernel(tmem_body(vec![elected_if(vec![tmem_a_mma(
            TmemAForm::BankBatched,
            500,
        )])]))
        .validate()
        .unwrap_err();
        assert!(e.message.contains("column footprint"), "{}", e.message);
    }

    #[test]
    fn tmem_alloc_bands_lifecycle_rules() {
        let alloc = |base_col: u32, n_cols: u32| {
            warp_if(
                0,
                vec![Stmt::TmemAlloc {
                    base_col,
                    n_cols,
                    cta_group: 1,
                    addr_byte_offset: 900_000,
                }],
            )
        };
        let dealloc = |base_col: u32, n_cols: u32| {
            warp_if(
                0,
                vec![Stmt::TmemDealloc {
                    base_col,
                    n_cols,
                    cta_group: 1,
                }],
            )
        };
        // alloc -> dealloc -> alloc: the single live band regenerates legally.
        assert!(kernel(vec![alloc(0, 64), dealloc(0, 64), alloc(0, 64)])
            .validate()
            .is_ok());
        // a nonzero base is outside the single base-0 view the codegen lowers.
        let e = kernel(vec![alloc(32, 64)]).validate().unwrap_err();
        assert!(e.message.contains("base_col must be 0"), "{}", e.message);
        // two live bands at once.
        let e = kernel(vec![alloc(0, 64), alloc(0, 64)])
            .validate()
            .unwrap_err();
        assert!(e.message.contains("still live"), "{}", e.message);
        // dealloc without a matching live band.
        let e = kernel(vec![dealloc(0, 64)]).validate().unwrap_err();
        assert!(
            e.message.contains("does not match a live allocation"),
            "{}",
            e.message
        );
        // alloc after relinquish (PTX §9.7.17.7.1).
        let relinquish = warp_if(0, vec![Stmt::TmemRelinquish { cta_group: 1 }]);
        let e = kernel(vec![alloc(0, 64), dealloc(0, 64), relinquish, alloc(0, 64)])
            .validate()
            .unwrap_err();
        assert!(e.message.contains("after tmem_relinquish"), "{}", e.message);
        // alloc must be issued by exactly one full warp (the CTA-wide top level
        // is not).
        let e = kernel(vec![Stmt::TmemAlloc {
            base_col: 0,
            n_cols: 64,
            cta_group: 1,
            addr_byte_offset: 900_000,
        }])
        .validate()
        .unwrap_err();
        assert!(e.message.contains("exactly one full warp"), "{}", e.message);
        // lifecycle ops are banned inside a loop body (the one-pass band walk
        // cannot see the re-execution).
        let e = kernel(vec![Stmt::ForLoop {
            var: var(1, VarBinding::Loop),
            start: ScalarValue::Int(0),
            stop: ScalarValue::Int(2),
            step: ScalarValue::Int(1),
            body: vec![warp_if(
                0,
                vec![Stmt::TmemAlloc {
                    base_col: 0,
                    n_cols: 64,
                    cta_group: 1,
                    addr_byte_offset: 900_000,
                }],
            )],
            unroll: true,
        }])
        .validate()
        .unwrap_err();
        assert!(e.message.contains("not allowed inside"), "{}", e.message);
        // ... and inside a runtime-value conditional (a statically-unknown
        // thread filter).
        let v = var(2, VarBinding::Scalar);
        let e = kernel(vec![
            Stmt::ScalarDef {
                var: v,
                initial: ScalarInitial::Value(ScalarValue::Int(1)),
            },
            Stmt::If {
                cond: ScalarValue::expr(
                    ScalarOp::Eq,
                    vec![ScalarValue::Var(v), ScalarValue::Int(1)],
                ),
                then_body: vec![warp_if(
                    0,
                    vec![Stmt::TmemAlloc {
                        base_col: 0,
                        n_cols: 64,
                        cta_group: 1,
                        addr_byte_offset: 900_000,
                    }],
                )],
            },
        ])
        .validate()
        .unwrap_err();
        assert!(e.message.contains("not allowed inside"), "{}", e.message);
    }

    #[test]
    fn tmem_operand_static_spans_must_land_in_a_live_band() {
        let frag = tensor(1, MemorySpace::Reg, DType::F32, vec![32]);
        let alloc = warp_if(
            0,
            vec![Stmt::TmemAlloc {
                base_col: 0,
                n_cols: 64,
                cta_group: 1,
                addr_byte_offset: 900_000,
            }],
        );
        // start_col is part of the absolute address: 48 + a 32-column LD
        // footprint escapes the live [0, 64) band.
        let ld = Stmt::Tcgen05Ld {
            dst: full_slice(&frag),
            src: tmem_addr(48, 0, 0),
            shape: LdStShape::B32x32,
            num: 32,
        };
        let e = kernel(vec![alloc.clone(), warp_if(0, vec![ld])])
            .validate()
            .unwrap_err();
        assert!(
            e.message.contains("not inside a live tmem allocation band"),
            "{}",
            e.message
        );
        // a constant address outside [0, 512) is rejected outright.
        let ld = Stmt::Tcgen05Ld {
            dst: full_slice(&frag),
            src: tmem_addr(0, 0, 512),
            shape: LdStShape::B32x32,
            num: 32,
        };
        let e = kernel(vec![alloc, warp_if(0, vec![ld])])
            .validate()
            .unwrap_err();
        assert!(e.message.contains("col (TMEM column)"), "{}", e.message);
    }

    #[test]
    fn tcgen05_cp_dtype_and_layout_rules() {
        let plain = tensor(0, MemorySpace::Smem, DType::U8, vec![256, 64]);
        let src = Arc::new(Tensor {
            id: 1,
            space: MemorySpace::Smem,
            dtype: DType::U8,
            shape: vec![256, 64],
            layout: Some(Layout::Swizzle(SmemSwizzleLayout {
                swizzle: Swizzle::B32,
            })),
            byte_offset: Some(0),
        });
        let cp = |multicast| Stmt::Tcgen05Cp {
            dst: tmem_addr(300, 0, 0),
            src: smem_tile(&src, 256, 64),
            shape: Tcgen05CpShape::B128x256,
            multicast,
            cta_group: 1,
        };
        // The shape names one physical atom; a whole-number multi-atom tile
        // is legal and its full target span is checked.
        assert!(kernel(tmem_body(vec![elected_if(vec![cp(
            Tcgen05CpMulticast::None
        )])]))
        .validate()
        .is_ok());
        let plain_cp = Stmt::Tcgen05Cp {
            dst: tmem_addr(300, 0, 0),
            src: smem_tile(&plain, 256, 64),
            shape: Tcgen05CpShape::B128x256,
            multicast: Tcgen05CpMulticast::None,
            cta_group: 1,
        };
        let e = kernel(tmem_body(vec![elected_if(vec![plain_cp])]))
            .validate()
            .unwrap_err();
        assert!(e.message.contains("explicit B32"), "{}", e.message);
        let e = kernel(tmem_body(vec![elected_if(vec![cp(
            Tcgen05CpMulticast::Warp4,
        )])]))
        .validate()
        .unwrap_err();
        assert!(e.message.contains("illegal"), "{}", e.message);

        let partial = tensor(2, MemorySpace::Smem, DType::U8, vec![192, 32]);
        let bad = Stmt::Tcgen05Cp {
            dst: tmem_addr(300, 0, 0),
            src: smem_tile(&partial, 192, 32),
            shape: Tcgen05CpShape::B128x256,
            multicast: Tcgen05CpMulticast::None,
            cta_group: 1,
        };
        let e = kernel(tmem_body(vec![elected_if(vec![bad])]))
            .validate()
            .unwrap_err();
        assert!(e.message.contains("multiple of 128"), "{}", e.message);
    }

    #[test]
    fn tcgen05_ld_st_tmem_operand_rules() {
        let frag = tensor(1, MemorySpace::Reg, DType::F32, vec![32]);
        let ld = Stmt::Tcgen05Ld {
            dst: full_slice(&frag),
            src: tmem_addr(0, 0, 0),
            shape: LdStShape::B32x32,
            num: 32,
        };
        assert!(kernel(tmem_body(vec![ld])).validate().is_ok());
        // a TMEM operand lane base outside [0, 128) fails closed.
        let bad = Stmt::Tcgen05Ld {
            dst: full_slice(&frag),
            src: tmem_addr(0, 128, 0),
            shape: LdStShape::B32x32,
            num: 32,
        };
        let e = kernel(tmem_body(vec![bad])).validate().unwrap_err();
        assert!(e.message.contains("row (TMEM lane)"), "{}", e.message);
        // ld/st column span must land in a live band (num=32 x 1 col cells).
        let oob = Stmt::Tcgen05St {
            dst: tmem_addr(0, 0, 500),
            src: full_slice(&frag),
            shape: LdStShape::B32x32,
            num: 32,
        };
        let e = kernel(tmem_body(vec![oob])).validate().unwrap_err();
        assert!(
            e.message.contains("not inside a live tmem allocation band"),
            "{}",
            e.message
        );
    }

    fn tma_load(m: &Arc<MBar>, shape: Vec<usize>, multicast: Option<u16>, cg: u8) -> Stmt {
        let src = tensor(10, MemorySpace::Gmem, DType::F32, vec![64, 64]);
        let dst = tensor(11, MemorySpace::Smem, DType::F32, vec![64, 64]);
        Stmt::TmaLoad {
            dst: full_slice(&dst),
            src: src.clone(),
            mbar: mbar_ref(m),
            coords: vec![ScalarValue::Int(0), ScalarValue::Int(0)],
            shape,
            gmem_shape: None,
            mbar_stage: None,
            multicast_cta_mask: multicast,
            cache_hint: None,
            prefetch_tensormap: false,
            cta_group: cg,
        }
    }

    #[test]
    fn tma_load_derived_bytes_rules() {
        let m = mbar(1, MBarKind::Tma);
        // a plain unicast load: legal (bytes derive from the tile).
        assert!(kernel(vec![
            Stmt::MBarDef { mbar: m.clone() },
            elected_if(vec![tma_load(&m, vec![64, 64], None, 1)]),
        ])
        .validate()
        .is_ok());
        // a zero-extent tile would derive a 0-byte transfer.
        let e = kernel(vec![
            Stmt::MBarDef { mbar: m.clone() },
            elected_if(vec![tma_load(&m, vec![0, 64], None, 1)]),
        ])
        .validate()
        .unwrap_err();
        assert!(e.message.contains("nonzero element count"), "{}", e.message);
        // cg2 multicast may target one explicit peer-referenced leader mbar;
        // every multicast destination contributes its own complete-tx.
        let peer = Arc::new(MBar {
            id: 2,
            kind: MBarKind::Tma,
            stages: 1,
            byte_offset: 800_016,
            arrive_count: None,
        });
        let mut k = kernel(vec![Stmt::MBarDef { mbar: peer.clone() }]);
        k.launch_shape = vec![2];
        k.cluster_shape = vec![2];
        let mut load = tma_load(&peer, vec![64, 64], Some(0b11), 2);
        if let Stmt::TmaLoad { mbar, .. } = &mut load {
            mbar.remote_coord = Some(ScalarValue::Int(0));
        }
        k.body.push(elected_if(vec![load]));
        assert!(k.validate().is_ok());
    }

    #[test]
    fn mbarrier_init_count_cap_and_wait_phase_required() {
        let m = mbar(1, MBarKind::Tma);
        let init = |count: u32| Stmt::MBarrierInit {
            mbar: mbar_ref(&m),
            count,
            stage: None,
        };
        assert!(kernel(vec![
            Stmt::MBarDef { mbar: m.clone() },
            elected_if(vec![init(1 << 20 - 1)]),
        ])
        .validate()
        .is_ok());
        let e = kernel(vec![
            Stmt::MBarDef { mbar: m.clone() },
            elected_if(vec![init(1 << 20)]),
        ])
        .validate()
        .unwrap_err();
        assert!(e.message.contains("2^20"), "{}", e.message);
        // phase is required (the phase-less form diverged sim vs codegen).
        let e = kernel(vec![
            Stmt::MBarDef { mbar: m.clone() },
            elected_if(vec![init(1)]),
            Stmt::MBarrierWait {
                mbar: mbar_ref(&m),
                stage: None,
                phase: None,
            },
        ])
        .validate()
        .unwrap_err();
        assert!(e.message.contains("phase is required"), "{}", e.message);
    }

    #[test]
    fn scalar_let_single_assignment() {
        let v = var(1, VarBinding::Scalar);
        let let_stmt = Stmt::ScalarLet {
            var: v,
            value: ScalarValue::Int(3),
        };
        assert!(kernel(vec![let_stmt.clone()]).validate().is_ok());
        // a let-bound var rejects any scalar_store (SSA immutability).
        let e = kernel(vec![
            let_stmt,
            Stmt::ScalarStore {
                var: v,
                value: ScalarValue::Int(4),
            },
        ])
        .validate()
        .unwrap_err();
        assert!(e.message.contains("single assignment"), "{}", e.message);
    }

    /// `single_issue_scope`: hardware single-issue ops must sit under an
    /// explicit single-lane branch — codegen never infers the guard, so a
    /// bare site would issue once per lane (double-init, duplicate TMA/MMA).
    #[test]
    fn single_issue_scope_rule() {
        let bar = mbar(7, MBarKind::Thread);
        let init = || Stmt::MBarrierInit {
            mbar: mbar_ref(&bar),
            count: 1,
            stage: None,
        };
        let def = || Stmt::MBarDef { mbar: bar.clone() };

        // bare at function scope -> rejected
        let e = kernel(vec![def(), init()]).validate().unwrap_err();
        assert!(e.message.contains("single_issue_scope"), "{}", e.message);
        // under a full-warp branch (32 lanes) -> rejected
        let e = kernel(vec![def(), warp_if(0, vec![init()])])
            .validate()
            .unwrap_err();
        assert!(e.message.contains("single_issue_scope"), "{}", e.message);
        // under warp + if_elected (one lane) -> accepted
        kernel(vec![def(), warp_if(0, vec![elected_if(vec![init()])])])
            .validate()
            .unwrap();
        // under a provably one-lane predicate (tid_in_wg == 0) -> accepted
        let tid0 = Stmt::If {
            cond: ScalarValue::expr(
                ScalarOp::Eq,
                vec![
                    ScalarValue::Scope(ScopeValueKind::TidInWg),
                    ScalarValue::Int(0),
                ],
            ),
            then_body: vec![init()],
        };
        kernel(vec![def(), tid0]).validate().unwrap();
    }

    #[test]
    fn shuffle_sync_lane_range_and_elected_scope() {
        let v = var(1, VarBinding::Scalar);
        let shuf = |lane: ScalarValue| Stmt::ShuffleSync {
            var: v,
            src: ScalarValue::Scope(ScopeValueKind::TidInWg),
            src_lane: lane,
        };
        assert!(kernel(vec![shuf(ScalarValue::Int(0))]).validate().is_ok());
        // a statically out-of-range source lane fails closed.
        let e = kernel(vec![shuf(ScalarValue::Int(32))])
            .validate()
            .unwrap_err();
        assert!(
            e.message.contains("src_lane must be in [0, 32)"),
            "{}",
            e.message
        );
        // inside an elected (single-lane) region the full-mask shfl is UB.
        let e = kernel(vec![elected_if(vec![shuf(ScalarValue::Int(0))])])
            .validate()
            .unwrap_err();
        assert!(e.message.contains("elected"), "{}", e.message);
    }

    #[test]
    fn for_loop_unroll_false_requires_zero_start_unit_step() {
        let lp = |start: i64, step: i64, unroll: bool| Stmt::ForLoop {
            var: var(1, VarBinding::Loop),
            start: ScalarValue::Int(start),
            stop: ScalarValue::Int(4),
            step: ScalarValue::Int(step),
            body: vec![],
            unroll,
        };
        assert!(kernel(vec![lp(0, 1, false)]).validate().is_ok());
        assert!(kernel(vec![lp(1, 2, true)]).validate().is_ok());
        let e = kernel(vec![lp(1, 1, false)]).validate().unwrap_err();
        assert!(e.message.contains("unroll=false"), "{}", e.message);
        let e = kernel(vec![lp(0, 2, false)]).validate().unwrap_err();
        assert!(e.message.contains("unroll=false"), "{}", e.message);
    }

    #[test]
    fn clc_try_cancel_rules() {
        let space = Arc::new(TaskSpace {
            id: 0,
            grid: vec![4],
            fields: vec!["t".into()],
        });
        let sched = Arc::new(Scheduler {
            id: 1,
            space,
            policy: SchedulerPolicy::Custom,
            scope: SchedulerScope::Cluster,
        });
        let handle = tensor(5, MemorySpace::Smem, DType::U32, vec![4]);
        let small_handle = tensor(6, MemorySpace::Smem, DType::U8, vec![8]);
        let m = mbar(1, MBarKind::Tma);
        let try_cancel =
            |sched: &Arc<Scheduler>, handle: &Arc<Tensor>, m: &Arc<MBar>| Stmt::ClcTryCancel {
                scheduler: sched.clone(),
                handle: handle.clone(),
                mbar: mbar_ref(m),
                stage: None,
                cta_group: 1,
            };
        let in_impl = |body: Vec<Stmt>| {
            vec![Stmt::SchedulerImpl {
                scheduler: sched.clone(),
                body,
            }]
        };
        // inside scheduler_impl with a TMA mbar and a 16B handle: legal.
        assert!(kernel(in_impl(vec![elected_if(vec![try_cancel(
            &sched, &handle, &m
        )])]))
        .validate()
        .is_ok());
        // outside scheduler_impl.
        let e = kernel(vec![try_cancel(&sched, &handle, &m)])
            .validate()
            .unwrap_err();
        assert!(e.message.contains("inside scheduler_impl"), "{}", e.message);
        // the handle must hold the 16B (uint4) response.
        let e = kernel(in_impl(vec![try_cancel(&sched, &small_handle, &m)]))
            .validate()
            .unwrap_err();
        assert!(e.message.contains("16 bytes"), "{}", e.message);
        // the signalled barrier completes-tx like a TMA landing: TMA kind only.
        let thread_mbar = mbar(2, MBarKind::Thread);
        let e = kernel(in_impl(vec![try_cancel(&sched, &handle, &thread_mbar)]))
            .validate()
            .unwrap_err();
        assert!(e.message.contains("mbar kind must be tma"), "{}", e.message);
        // a functional (grid_stride) scheduler cannot host a CLC impl.
        let gs = Arc::new(Scheduler {
            id: 2,
            space: Arc::new(TaskSpace {
                id: 1,
                grid: vec![4],
                fields: vec!["t".into()],
            }),
            policy: SchedulerPolicy::GridStride,
            scope: SchedulerScope::Cluster,
        });
        let e = kernel(vec![Stmt::SchedulerImpl {
            scheduler: gs.clone(),
            body: vec![],
        }])
        .validate()
        .unwrap_err();
        assert!(
            e.message.contains("concurrent scheduler policy"),
            "{}",
            e.message
        );
    }
}
