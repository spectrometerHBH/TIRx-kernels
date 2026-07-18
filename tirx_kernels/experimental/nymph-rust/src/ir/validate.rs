//! Validation — a faithful port of every check in `ir.py`'s `__post_init__`
//! methods and the helper `_check_*` functions. With `Arc<Tensor>` we can do the
//! cross-referencing checks (space/dtype/rank) too, since `slice.tensor.*` reads
//! the tensor's data directly.
//!
//! Python runs these per-object at construction; we run them as one pass over the
//! assembled kernel (`Kernel::validate`). Same checks, same intent — they just
//! fire when you call `validate()` rather than at each `__post_init__`.

use super::*;
use std::collections::{HashMap, HashSet};

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

/// Execution scope tracked by the context walk.
#[derive(Clone, Copy, PartialEq)]
enum Scope {
    Cta,
    Warpgroup,
    Warp,
    Single,
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
fn check_lane(value: Option<u32>, label: &str) -> R {
    if let Some(v) = value {
        if v >= 32 {
            return bail(format!("{label} must be in [0, 32)"));
        }
    }
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

/// `_check_tcgen05_mma_shape`. `block_scaled_f8` selects the block-scaled f8
/// instruction family (sfa/sfb present, non-fp4) — only it may step N by 16.
fn check_mma_shape(m: u32, n: u32, k: u32, cta_group: u8, block_scaled_f8: bool) -> R {
    check_positive(m, "tcgen05_mma m")?;
    check_positive(n, "tcgen05_mma n")?;
    check_positive(k, "tcgen05_mma k")?;
    // The k rule itself is per operand KIND and lives at the Tcgen05Mma arm
    // (which sees the dtypes): dense f16/bf16 is any positive multiple of the
    // k=16 atom (an ordered run of atomic MMAs — canon issues one full-K
    // gemm_async per k-tile and TVM lowers it to the atoms); the block-scaled
    // f8 instruction is k=32 (128/256 its folded k-tile forms); fp4 (mxf4) is
    // k in {64, 128, 256}.
    match cta_group {
        1 => {
            let granularity = if m == 64 { 8 } else { 16 };
            if (m != 64 && m != 128) || n > 256 || n % granularity != 0 {
                return bail("tcgen05_mma matrix shape is invalid for cta_group=1");
            }
            Ok(())
        }
        2 => {
            // The block-scaled f8 instruction (k=32) steps N by 16 (DeepGEMM's
            // swap_ab grid uses N = block_m in 16-element steps, e.g. 240);
            // the dense f16/bf16 and nvfp4 (k=64) shapes keep the 32-step rule.
            let granularity = if block_scaled_f8 && k == 32 { 16 } else { 32 };
            if (m != 128 && m != 256) || n > 256 || n % granularity != 0 {
                return bail("tcgen05_mma matrix shape is invalid for cta_group=2");
            }
            Ok(())
        }
        _ => check_cta_group(cta_group, "tcgen05_mma cta_group"),
    }
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

/// `TmemLayout.__post_init__` (the only layout with run-time validation).
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

/// `_check_slice_covers_shape`: a static slice dim may not be smaller than the
/// requested shape dim.
/// All-static slice dims, or None if any dim is a runtime scalar.
fn static_slice_shape(s: &TensorSlice) -> Option<Vec<usize>> {
    s.shape
        .iter()
        .map(|d| static_int(d).map(|v| v as usize))
        .collect()
}

/// `TmemOperand` checks: the address scalars are well-formed, and a constant
/// lane/column base is in range. (Column-band membership against the live
/// allocations is the liveness walk's job — see `check_tmem_alloc_bands`.)
fn validate_tmem_operand(op: &TmemOperand, label: &str) -> R {
    validate_scalar(&op.row)?;
    validate_scalar(&op.col)?;
    if let Some(r) = static_int(&op.row) {
        if !(0..128).contains(&r) {
            return bail(format!("{label} row (TMEM lane) must be in [0, 128)"));
        }
    }
    if let Some(c) = static_int(&op.col) {
        if !(0..512).contains(&c) {
            return bail(format!("{label} col (TMEM column) must be in [0, 512)"));
        }
    }
    Ok(())
}

/// The cell dtype an MMA A/B operand carries, regardless of which memory it
/// lives in (SMEM slice dtype, or the TMEM cell interpretation).
fn mma_operand_dtype(op: &MmaOperand) -> DType {
    match op {
        MmaOperand::Slice(s) => s.tensor.dtype,
        MmaOperand::Tmem(t) => t.dtype,
    }
}

/// The TMEM operand of a `Tcgen05Ld`/`Tcgen05St` (src / dst respectively).
fn tmem_operand_lanes_cols(shape: &LdStShape, num: u32) -> Option<usize> {
    // Column-cell span of the atom: 32x32b moves one 32-bit column per
    // register; the 16x*b atoms move b/32 columns per register step.
    let per = match shape {
        LdStShape::B32x32 => 1usize,
        LdStShape::B16x32Bx2 | LdStShape::B16x64 => 2,
        LdStShape::B16x128 => 4,
        LdStShape::B16x256 => 8,
    };
    usize::try_from(num).ok().map(|n| n * per)
}

/// `check_slice_covers` against the slice's TRAILING dims: a staged operand is
/// a (1, ..., rows, k) box of a stage-major tensor; the leading dims must be
/// unit and the trailing dims must cover the requested tile.
fn check_slice_covers_trailing(s: &TensorSlice, shape: &[usize], label: &str) -> R {
    if s.shape.len() < shape.len() {
        return bail(format!("{label} does not cover requested shape"));
    }
    let lead = s.shape.len() - shape.len();
    for dim in &s.shape[..lead] {
        if let Some(d) = static_int(dim) {
            if d != 1 {
                return bail(format!("{label} staged operand leading dims must be 1"));
            }
        }
    }
    for (slice_dim, &shape_dim) in s.shape[lead..].iter().zip(shape.iter()) {
        if let Some(d) = static_int(slice_dim) {
            if (d as usize) < shape_dim {
                return bail(format!("{label} does not cover requested shape"));
            }
        }
    }
    Ok(())
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

        Stmt::KernelInit { lane, .. } | Stmt::KernelFinalize { lane, .. } => {
            check_lane(*lane, "kernel_init/finalize lane")?;
        }
        Stmt::Role {
            warp,
            warpgroup,
            elected,
            maxnreg,
            ..
        } => {
            if warp.is_some() && warpgroup.is_some() {
                return bail("role cannot set both warp and warpgroup");
            }
            if let Some(m) = maxnreg {
                if warpgroup.is_none() || *elected {
                    return bail("role maxnreg requires a non-elected warpgroup role");
                }
                if *m < 1 || m % 8 != 0 {
                    return bail("role maxnreg must be a positive multiple of 8");
                }
            }
        }
        Stmt::ForLoop {
            var,
            start,
            stop,
            step,
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
        Stmt::BreakIf { cond } => {
            validate_scalar(cond)?;
            if uses_role_scope(cond) {
                return bail(
                    "break_if condition cannot branch on role scope values; use role scope",
                );
            }
        }
        Stmt::If { cond, .. } => {
            validate_scalar(cond)?;
            if uses_role_scope(cond) {
                return bail("if condition cannot branch on role scope values; use role scope");
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
            bytes,
            coords,
            shape,
            gmem_shape,
            mbar_stage,
            multicast_cta_mask,
            cache_hint,
            prefetch_tensormap: _,
            cta_group,
        } => {
            validate_slice(dst, "tma_load dst")?;
            validate_tensor(src)?;
            if mbar.mbar.kind != MBarKind::Tma {
                return bail("tma_load mbar kind must be tma");
            }
            validate_scalar(bytes)?;
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
            // cta_group=2 + multicast + a shared (peer-referenced) mbar: the
            // sim's tx accounting completes once per UNIQUE barrier cell, but
            // hardware completes once per multicast DESTINATION — no single
            // expect_tx count satisfies both on a shared barrier (the nvfp4
            // SFB note), so the combination is unmodelable.
            if *cta_group == 2 && multicast_cta_mask.is_some() && mbar.remote_coord.is_some() {
                return bail(
                    "tma_load cta_group=2 multicast with a peer-referenced (shared) \
                     mbar is not modeled",
                );
            }
            // The L2 eviction hint is passed through to TIRx verbatim; only the
            // policies the kernels use (and the emitted forms accept) are IR-legal.
            if let Some(hint) = cache_hint {
                if hint != "evict_normal" && hint != "evict_first" {
                    return bail("tma_load cache_hint must be \"evict_normal\" or \"evict_first\"");
                }
            }
        }
        Stmt::CpAsyncBulkS2Cluster { dst, src, mbar, .. } => {
            validate_slice(src, "cp_async_bulk_s2cluster src")?;
            validate_slice(dst, "cp_async_bulk_s2cluster dst")?;
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
        Stmt::GmemAtomicAdd { sem, coords, .. } => {
            validate_slice(sem, "gmem_atomic_add sem")?;
            if sem.tensor.space != MemorySpace::Gmem {
                return bail("gmem_atomic_add sem must be GMEM");
            }
            if sem.tensor.dtype != DType::I32 {
                return bail("gmem_atomic_add sem must be an i32 semaphore");
            }
            if coords.len() != sem.tensor.shape.len() {
                return bail("gmem_atomic_add coords rank must match the semaphore tensor rank");
            }
        }
        Stmt::GmemWaitEq { sem, coords, .. } => {
            validate_slice(sem, "gmem_wait_eq sem")?;
            if sem.tensor.space != MemorySpace::Gmem {
                return bail("gmem_wait_eq sem must be GMEM");
            }
            if sem.tensor.dtype != DType::I32 {
                return bail("gmem_wait_eq sem must be an i32 semaphore");
            }
            if coords.len() != sem.tensor.shape.len() {
                return bail("gmem_wait_eq coords rank must match the semaphore tensor rank");
            }
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
            m,
            n,
            k,
            trans_a,
            trans_b,
            cta_group,
            sfa,
            sfb,
            sf_byte,
            sf_e4m3,
            sf_block,
            a_fp4,
            b_fp4,
            lane_align,
            ..
        } => {
            validate_tmem_operand(dst, "tcgen05_mma dst")?;
            // The full-datapath accumulator layouts are lane-anchored: the dst
            // base lane must be 0 (checked statically when the address is a
            // constant; the interpreter re-checks the evaluated value).
            if let Some(r) = static_int(&dst.row) {
                if r != 0 {
                    return bail("tcgen05_mma dst row (lane) must be 0");
                }
            }
            let validate_ab = |op: &MmaOperand, lbl: &str| -> R {
                match op {
                    MmaOperand::Slice(s) => {
                        validate_slice(s, &format!("tcgen05_mma {lbl}"))?;
                        if s.tensor.space != MemorySpace::Smem {
                            return bail(format!(
                                "tcgen05_mma {lbl} slice operand must be SMEM \
                                 (a TMEM operand is a TmemOperand)"
                            ));
                        }
                    }
                    MmaOperand::Tmem(t) => {
                        validate_tmem_operand(t, &format!("tcgen05_mma {lbl}"))?;
                    }
                }
                Ok(())
            };
            validate_ab(a, "a")?;
            validate_ab(b, "b")?;
            check_cta_group(*cta_group, "tcgen05_mma cta_group")?;
            // k is coupled to the operand kind (the PTX tcgen05.mma instruction
            // shapes): dense f16/bf16 is any positive multiple of the k=16 atom
            // — one IR MMA with k = 16q means the q atomic k=16 MMAs accumulated
            // in order (canon issues ONE full-K gemm_async per k-tile; TVM
            // lowers it to the atom sequence on hardware) — the block-scaled f8
            // instruction is k=32 (k=128/256 stay valid as the explicit folded
            // k-tile abstraction, but ONLY with scale vectors); fp4 (mxf4) is
            // k in {64, 128, 256}. Anything else was silently computed with a k
            // the silicon does not have.
            check_mma_shape(
                *m,
                *n,
                *k,
                *cta_group,
                (sfa.is_some() || sfb.is_some()) && !*a_fp4,
            )?;
            let a_dtype = mma_operand_dtype(a);
            let b_dtype = mma_operand_dtype(b);
            if *a_fp4 || *b_fp4 {
                if !matches!(*k, 64 | 128 | 256) {
                    return bail("tcgen05_mma fp4 (mxf4) k must be 64, 128, or 256");
                }
            } else if sfa.is_some() || sfb.is_some() {
                if !matches!(*k, 32 | 128 | 256) {
                    return bail("tcgen05_mma block-scaled f8 k must be 32, 128, or 256");
                }
            } else if *k % 16 != 0 && a_dtype != DType::F8E4M3 && b_dtype != DType::F8E4M3 {
                return bail(
                    "tcgen05_mma dense f16/bf16 k must be a positive multiple of 16 \
                     (an ordered run of k/16 atomic MMAs — the full-K gemm_async)",
                );
            }
            if *lane_align != 0 && *lane_align != 16 {
                return bail("tcgen05_mma lane_align must be 0 or 16");
            }
            // lane_align shifts the accumulator lane field and exists only for
            // the cta_group=1 m=64 (Layout F) accumulator; the interpreter's
            // mma_blocks/inplace_geometry reject every other layout at run
            // time — reject it here instead of after the IR was trusted.
            if *lane_align != 0 && !(*cta_group == 1 && *m == 64) {
                return bail("tcgen05_mma lane_align != 0 requires cta_group=1 and m=64");
            }
            // A TMEM operand is the value model's accumulator-readback
            // abstraction (the GDN state read straight out of TMEM). It is
            // exact only for the dtypes the TMEM readback path models —
            // f16/bf16 (packed halves) and f32; an f8e4m3 or packed-fp4 (u8)
            // TMEM operand has no modeled readback semantics.
            for (op, lbl) in [(a, "a"), (b, "b")] {
                if let MmaOperand::Tmem(t) = op {
                    if !matches!(t.dtype, DType::F16 | DType::Bf16 | DType::F32) {
                        return bail(format!(
                            "tcgen05_mma {lbl} TMEM operand dtype must be f16, bf16, or f32"
                        ));
                    }
                }
            }
            // A TMEM operand may be the f32 accumulator read directly as the MMA
            // operand (e.g. the GDN state S, kept in TMEM); the value model reads
            // every operand as logical f32, so an f32 TMEM operand is exact.
            let tmem_f32 =
                |op: &MmaOperand| matches!(op, MmaOperand::Tmem(t) if t.dtype == DType::F32);
            if *a_fp4 || *b_fp4 {
                // NVFP4: operands are e2m1 fp4 packed 2-per-u8; both must be fp4.
                if !*a_fp4 || !*b_fp4 {
                    return bail("tcgen05_mma a_fp4 and b_fp4 must be set together");
                }
                if a_dtype != DType::U8 || b_dtype != DType::U8 {
                    return bail("tcgen05_mma fp4 operands must be u8 (2 packed e2m1 per byte)");
                }
                // PTX Table 54: the mxf4* shapes have no transposed form and
                // exist only as (cta_group=1, M=128) or (cta_group=2, M=256);
                // the value model's fp4 path is the in-place SMEM datapath, so
                // a TMEM fp4 operand is out too (the TMEM-operand dtype rule
                // above already rejects u8 TMEM operands with that message).
                if *trans_a || *trans_b {
                    return bail("tcgen05_mma fp4 (mxf4) does not support trans_a/trans_b");
                }
                if !((*cta_group == 1 && *m == 128) || (*cta_group == 2 && *m == 256)) {
                    return bail(
                        "tcgen05_mma fp4 requires (cta_group=1, m=128) or (cta_group=2, m=256)",
                    );
                }
            } else {
                // Operands are f16/bf16/f8e4m3 one-per-slot, OR an f32 TMEM operand
                // (e.g. the GDN state S read directly out of TMEM).
                let ok_dtype = |op: &MmaOperand| {
                    matches!(
                        mma_operand_dtype(op),
                        DType::F16 | DType::Bf16 | DType::F8E4M3
                    ) || tmem_f32(op)
                };
                if !ok_dtype(a) || !ok_dtype(b) {
                    return bail(
                        "tcgen05_mma operand dtype must be f16, bf16, f8e4m3, or an f32 TMEM operand",
                    );
                }
                // Operand dtypes must match, EXCEPT an f16/bf16 operand paired
                // with an f32 TMEM operand (the accumulator-readback
                // abstraction; both materialize to f32 in value mode). Any
                // other mix — e.g. f8e4m3 against an f32 TMEM operand — has no
                // modeled (or hardware) semantics.
                if a_dtype != b_dtype {
                    let f32_tmem_with_b16 = |tmem_side: &MmaOperand, other: &MmaOperand| {
                        tmem_f32(tmem_side)
                            && matches!(mma_operand_dtype(other), DType::F16 | DType::Bf16)
                    };
                    if !f32_tmem_with_b16(a, b) && !f32_tmem_with_b16(b, a) {
                        return bail(
                            "tcgen05_mma a and b operand dtype must match \
                             (an f32 TMEM operand mixes only with f16/bf16)",
                        );
                    }
                }
            }
            if dst.dtype != DType::F32 {
                return bail("tcgen05_mma dst dtype must be f32");
            }
            // fp4 operands are packed 2-per-byte, so the K (contraction) extent in the
            // SMEM tile is k/2 bytes, not k elements.
            let a_rows = if *cta_group == 1 { *m } else { m / 2 };
            let b_rows = if *cta_group == 1 { *n } else { n / 2 };
            let a_kdim = if *a_fp4 {
                (*k / 2) as usize
            } else {
                *k as usize
            };
            let b_kdim = if *b_fp4 {
                (*k / 2) as usize
            } else {
                *k as usize
            };
            let a_shape = if *trans_a {
                [a_kdim, a_rows as usize]
            } else {
                [a_rows as usize, a_kdim]
            };
            let b_shape = if *trans_b {
                [b_kdim, b_rows as usize]
            } else {
                [b_rows as usize, b_kdim]
            };
            // SMEM operands keep the slice-coverage check; a TMEM operand's
            // extent is implied by m/n/k and verified against the live TMEM
            // allocation bands (walk 4).
            if let MmaOperand::Slice(s) = a {
                check_slice_covers_trailing(s, &a_shape, "tcgen05_mma a")?;
            }
            if let MmaOperand::Slice(s) = b {
                check_slice_covers_trailing(s, &b_shape, "tcgen05_mma b")?;
            }
            match (sfa, sfb) {
                (None, None) => {
                    if a_dtype == DType::F8E4M3 {
                        return bail("tcgen05_mma f8e4m3 operands require sfa/sfb scale vectors");
                    }
                }
                (Some(sfa), Some(sfb)) => {
                    // PTX: the block-scaled kinds (mxf8f6f4/mxf4) with
                    // cta_group::1 exist only at M=128 — an m=64 cg1 MMA has no
                    // scale mode, so sfa/sfb there is unemittable.
                    if *cta_group == 1 && *m == 64 {
                        return bail(
                            "tcgen05_mma m=64 cta_group=1 does not support block-scaled \
                             (sfa/sfb) modes",
                        );
                    }
                    // UE8M0 path requires f8e4m3 operands; NVFP4 (sf_e4m3) uses fp4 operands.
                    if !*sf_e4m3 && (a_dtype != DType::F8E4M3 || b_dtype != DType::F8E4M3) {
                        return bail("tcgen05_mma sfa/sfb require f8e4m3 operands");
                    }
                    if *sf_e4m3 && !*a_fp4 {
                        return bail("tcgen05_mma sf_e4m3 (NVFP4) requires fp4 operands");
                    }
                    if *sf_byte >= 4 {
                        return bail("tcgen05_mma sf_byte must be in 0..4");
                    }
                    // The NVFP4 e4m3 decode reads scale bytes 0..k/16 of each
                    // cell — sf_byte exists only in the packed-UE8M0 (fp8)
                    // layout and is silently ignored on the e4m3 path.
                    if *sf_e4m3 && *sf_byte != 0 {
                        return bail("tcgen05_mma sf_byte must be 0 for sf_e4m3 (NVFP4) scales");
                    }
                    // The two supported scale modes fix sf_block: nvfp4 block-16
                    // (sf_e4m3) or fp8 per-row (sf_block=0). Anything else would be
                    // silently mis-divided by the k/sf_block block math downstream.
                    match (*sf_e4m3, *sf_block) {
                        (true, 16) => {
                            if *k % 16 != 0 {
                                return bail(
                                    "tcgen05_mma nvfp4 k must be a multiple of sf_block=16",
                                );
                            }
                        }
                        (true, _) => return bail("tcgen05_mma sf_e4m3 requires sf_block=16"),
                        (false, 0) => {}
                        (false, _) => {
                            return bail("tcgen05_mma UE8M0 (fp8) mode requires sf_block=0")
                        }
                    }
                    for (sf, label) in [(sfa, "tcgen05_mma sfa"), (sfb, "tcgen05_mma sfb")] {
                        validate_tmem_operand(sf, label)?;
                        if let Some(r) = static_int(&sf.row) {
                            if r != 0 {
                                return bail(format!("{label} row (lane) must be 0"));
                            }
                        }
                        // UE8M0 packs 4 exponent bytes per u32 cell; NVFP4 holds e4m3 bytes.
                        let want = if *sf_e4m3 { DType::F8E4M3 } else { DType::U32 };
                        if sf.dtype != want {
                            return bail(format!(
                                "{label} dtype must be {} ({})",
                                if *sf_e4m3 { "e4m3" } else { "u32" },
                                if *sf_e4m3 {
                                    "nvfp4 scales"
                                } else {
                                    "4 packed UE8M0 bytes"
                                }
                            ));
                        }
                    }
                }
                _ => return bail("tcgen05_mma sfa and sfb must be provided together"),
            }
        }
        Stmt::Tcgen05Cp {
            dst,
            src,
            cta_group,
        } => {
            validate_tmem_operand(dst, "tcgen05_cp dst")?;
            if let Some(r) = static_int(&dst.row) {
                if r != 0 {
                    return bail("tcgen05_cp dst row (lane) must be 0");
                }
            }
            validate_slice(src, "tcgen05_cp src")?;
            check_cta_group(*cta_group, "tcgen05_cp cta_group")?;
            if src.tensor.space != MemorySpace::Smem {
                return bail("tcgen05_cp src must be SMEM");
            }
            // dst/src dtype must MATCH: the u32 path writes whole cells and the
            // e4m3 path writes raw scale bytes — a mixed pair (e.g. an e4m3 src
            // into a u32 dst) would write bytes into word cells (write_sf_byte
            // carries no dtype check).
            if dst.dtype != src.tensor.dtype {
                return bail("tcgen05_cp dst and src dtype must match");
            }
            // UE8M0 path packs scale bytes as u32 cells; NVFP4 moves e4m3 scale bytes.
            if !matches!(dst.dtype, DType::U32 | DType::F8E4M3) {
                return bail("tcgen05_cp moves u32 (UE8M0) or e4m3 (nvfp4) scale cells");
            }
            let Some(src_shape) = static_slice_shape(src) else {
                return bail("tcgen05_cp slice shapes must be static");
            };
            if src.tensor.dtype == DType::F8E4M3 {
                // NVFP4: the src's innermost dim is the per-row SF-block count;
                // rows ≥ 128 fold into TMEM column super-blocks of that width.
                let nblocks = *src_shape.last().unwrap_or(&0);
                if nblocks == 0 {
                    return bail("tcgen05_cp e4m3 src innermost SF-block dim must be non-zero");
                }
            } else {
                // UE8M0 u32: the value model enumerates the src flat into
                // (lane, col) cells, so it must be a single effective vector —
                // a 2-D tile's (row, col) reading is not what the copy models.
                if src_shape.iter().filter(|&&d| d != 1).count() > 1 {
                    return bail("tcgen05_cp u32 src must be effectively 1-D");
                }
            }
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
            validate_tmem_operand(src, "tcgen05_ld src")?;
            if dst.tensor.dtype != src.dtype {
                return bail("tcgen05_ld REG and TMEM operands must share a dtype");
            }
            if !matches!(
                src.dtype,
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
            validate_tmem_operand(dst, "tcgen05_st dst")?;
            if src.tensor.space != MemorySpace::Reg {
                return bail("tcgen05_st src must be REG");
            }
            if dst.dtype != src.tensor.dtype {
                return bail("tcgen05_st REG and TMEM operands must share a dtype");
            }
            if !matches!(
                dst.dtype,
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
            if !is_b32_reg_dtype(dst.tensor.dtype) {
                return bail("ldmatrix dst dtype must be i32 or u32");
            }
            if !is_b16_dtype(src.tensor.dtype) {
                return bail("ldmatrix src dtype must be f16, bf16, i16, or u16");
            }
            if let Some(n) = static_shape_numel(&dst.shape) {
                if n != *num as usize {
                    return bail("ldmatrix dst slice must contain num b32 registers");
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
            ..
        } => {
            if !is_float_reg_dtype(dst.tensor.dtype) {
                return bail("reg_cond_rescale dst dtype must be f16, bf16, or f32");
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
            ..
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
        Stmt::RegCvt { dst, src, .. } => {
            validate_slice(dst, "reg_cvt dst")?;
            validate_slice(src, "reg_cvt src")?;
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
        Stmt::ClusterBarrierArrive | Stmt::ClusterBarrierWait => {}
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
            if *num_warps < 1 {
                return bail("named_barrier num_warps must be >= 1");
            }
        }
        Stmt::SetMaxNReg { count, .. } => {
            // PTX setmaxnreg: register count must be a multiple of 8 in [24, 256].
            if *count < 24 || *count > 256 || *count % 8 != 0 {
                return bail("setmaxnreg count must be a multiple of 8 in [24, 256]");
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

/// `_uses_scope_value` for the role-scope kinds (warp_id / warpgroup_id).
fn uses_role_scope(v: &ScalarValue) -> bool {
    match v {
        ScalarValue::Scope(k) => matches!(k, ScopeValueKind::WarpId | ScopeValueKind::WarpgroupId),
        ScalarValue::Expr(e) => e.args.iter().any(uses_role_scope),
        _ => false,
    }
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

fn nested_scope_init(warp: Option<u32>, lane: Option<u32>, elected: bool) -> Scope {
    if elected || lane.is_some() {
        Scope::Single
    } else if warp.is_some() {
        Scope::Warp
    } else {
        Scope::Cta
    }
}
fn nested_scope_role(warp: Option<u32>, warpgroup: Option<u32>, elected: bool) -> Scope {
    if elected {
        Scope::Single
    } else if warpgroup.is_some() {
        Scope::Warpgroup
    } else if warp.is_some() {
        Scope::Warp
    } else {
        Scope::Cta
    }
}

fn check_role_geometry(
    warp: Option<u32>,
    warpgroup: Option<u32>,
    num_warps: u32,
    is_role: bool,
) -> R {
    if let Some(w) = warp {
        if w >= num_warps {
            return bail("role/kernel scope warp must be in [0, kernel num_warps)");
        }
    }
    if is_role {
        if let Some(wg) = warpgroup {
            if wg >= num_warps / 4 {
                return bail("role warpgroup must be in [0, kernel num_warps / 4)");
            }
        }
    }
    Ok(())
}

/// Walks 1 (var-defs) + 2 (scope rules), threading the defined-set, current scope,
/// per-role wg_sync barrier ownership, and the per-role named-barrier id sets.
/// `role_token` identifies the enclosing role-like body (a counter); `next_token`
/// hands out fresh ones. The hardware's 16 barriers are ONE resource shared by
/// wg_sync and named_barrier, so ONE role may not use the same barrier_id for
/// both classes (its own arrivals would count toward two different rendezvous).
/// Cross-ROLE id reuse (canon's flash_bwd: a 2-warpgroup named_barrier(1) next
/// to another warpgroup's private wg_sync(1)) is a deliberate hardware pattern
/// and stays legal.
#[allow(clippy::too_many_arguments)]
fn check_context(
    body: &[Stmt],
    scope: Scope,
    role_token: Option<u32>,
    barriers: &mut HashMap<u32, u32>,
    named_barriers: &mut HashMap<u32, HashSet<u32>>,
    defined: &mut HashSet<Var>,
    num_warps: u32,
    next_token: &mut u32,
    in_scheduler_impl: bool,
    scheduler_loop_depth: usize,
    inside_role: bool,
) -> R {
    for stmt in body {
        match stmt {
            Stmt::CtaSync if inside_role => return bail("cta_sync cannot be used inside role"),
            Stmt::CtaSync if scope != Scope::Cta => return bail("cta_sync must be in CTA scope"),
            Stmt::ClusterSync if scope != Scope::Cta => {
                return bail("cluster_sync must be in CTA scope")
            }
            // Split cluster barrier: the collective ARRIVE is a CTA-scope prologue op
            // (all threads, like cluster_sync); the WAIT is per-role, so it is ALLOWED
            // inside a role (the whole point — idle warps skip it, active roles defer it
            // past their local setup). The checker verifies a role can't touch peer
            // state before its wait (premature peer mbarrier access errors).
            Stmt::ClusterBarrierArrive if inside_role => {
                return bail("cluster_barrier_arrive cannot be used inside role")
            }
            Stmt::ClusterBarrierArrive if scope != Scope::Cta => {
                return bail("cluster_barrier_arrive must be in CTA scope")
            }
            Stmt::ShuffleSync { .. } if scope == Scope::Single => {
                // shfl.sync with the full-warp mask inside a single-lane
                // (elected) region is hardware UB — the mask names lanes that
                // are not converged there.
                return bail("shuffle_sync cannot be in single-thread (elected) scope");
            }
            Stmt::SetMaxNReg { warpgroup, .. } => {
                if *warpgroup >= num_warps / 4 {
                    return bail("setmaxnreg warpgroup must be in [0, kernel num_warps / 4)");
                }
            }
            Stmt::WgSync { barrier_id } => {
                if scope != Scope::Warpgroup {
                    return bail("wg_sync must be in warpgroup scope");
                }
                let token = role_token.ok_or_else(|| err("wg_sync must be in a role"))?;
                if named_barriers
                    .get(barrier_id)
                    .is_some_and(|tokens| tokens.contains(&token))
                {
                    return bail(
                        "wg_sync barrier_id cannot alias a named_barrier in the same role",
                    );
                }
                let owner = *barriers.entry(*barrier_id).or_insert(token);
                if owner != token {
                    return bail("wg_sync barrier_id cannot be shared across roles");
                }
            }
            Stmt::NamedBarrier { barrier_id, .. } => {
                // Cross-warpgroup named barrier: warpgroup scope + a role, but
                // intentionally SHARED across roles (the rendezvous spans them), so
                // no single-owner check (unlike wg_sync). It still may not alias a
                // wg_sync id IN THE SAME role — the 16 hardware barriers are one
                // resource.
                if scope != Scope::Warpgroup {
                    return bail("named_barrier must be in warpgroup scope");
                }
                let token = role_token.ok_or_else(|| err("named_barrier must be in a role"))?;
                if barriers.get(barrier_id) == Some(&token) {
                    return bail(
                        "named_barrier barrier_id cannot alias a wg_sync in the same role",
                    );
                }
                named_barriers.entry(*barrier_id).or_default().insert(token);
            }
            Stmt::WarpSync if scope == Scope::Single => {
                return bail("warp_sync cannot be in single-thread scope");
            }
            Stmt::TmemAlloc { .. } | Stmt::TmemDealloc { .. } | Stmt::TmemRelinquish { .. }
                if scope != Scope::Warp =>
            {
                return bail("tmem alloc/dealloc/relinquish must be in warp scope");
            }
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
                unroll: _,
            } => {
                let mut vars = Vec::new();
                collect_vars(start, &mut vars);
                collect_vars(stop, &mut vars);
                collect_vars(step, &mut vars);
                require_defined(&vars, defined, "loop bound")?;
                define_var(*var, defined)?;
                check_context(
                    body,
                    scope,
                    role_token,
                    barriers,
                    named_barriers,
                    defined,
                    num_warps,
                    next_token,
                    in_scheduler_impl,
                    scheduler_loop_depth,
                    inside_role,
                )?;
            }
            Stmt::ForEachTask { var, body, .. } => {
                define_var(*var, defined)?;
                check_context(
                    body,
                    scope,
                    role_token,
                    barriers,
                    named_barriers,
                    defined,
                    num_warps,
                    next_token,
                    in_scheduler_impl,
                    scheduler_loop_depth,
                    inside_role,
                )?;
            }
            Stmt::SchedulerImpl { body, .. } => {
                check_context(
                    body,
                    scope,
                    role_token,
                    barriers,
                    named_barriers,
                    defined,
                    num_warps,
                    next_token,
                    true,
                    0,
                    inside_role,
                )?;
            }
            Stmt::SchedNext { var, .. } => {
                if !in_scheduler_impl {
                    return bail("sched_next must be inside scheduler_impl");
                }
                define_var(*var, defined)?;
            }
            Stmt::ClcTryCancel { stage, .. } => {
                if !in_scheduler_impl {
                    return bail("clc_try_cancel must be inside scheduler_impl");
                }
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
                if !in_scheduler_impl && scope == Scope::Cta {
                    return bail("loop must be inside scheduler_impl or role scope");
                }
                check_context(
                    body,
                    scope,
                    role_token,
                    barriers,
                    named_barriers,
                    defined,
                    num_warps,
                    next_token,
                    in_scheduler_impl,
                    scheduler_loop_depth + 1,
                    inside_role,
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
            Stmt::KernelInit {
                body,
                warp,
                lane,
                elected,
            } => {
                check_role_geometry(*warp, None, num_warps, false)?;
                let token = *next_token;
                *next_token += 1;
                check_context(
                    body,
                    nested_scope_init(*warp, *lane, *elected),
                    Some(token),
                    barriers,
                    named_barriers,
                    defined,
                    num_warps,
                    next_token,
                    in_scheduler_impl,
                    scheduler_loop_depth,
                    false,
                )?;
            }
            Stmt::KernelFinalize {
                body,
                warp,
                lane,
                elected,
            } => {
                check_role_geometry(*warp, None, num_warps, false)?;
                let token = *next_token;
                *next_token += 1;
                check_context(
                    body,
                    nested_scope_init(*warp, *lane, *elected),
                    Some(token),
                    barriers,
                    named_barriers,
                    defined,
                    num_warps,
                    next_token,
                    in_scheduler_impl,
                    scheduler_loop_depth,
                    false,
                )?;
            }
            Stmt::Role {
                body,
                warp,
                warpgroup,
                elected,
                ..
            } => {
                check_role_geometry(*warp, *warpgroup, num_warps, true)?;
                let token = *next_token;
                *next_token += 1;
                check_context(
                    body,
                    nested_scope_role(*warp, *warpgroup, *elected),
                    Some(token),
                    barriers,
                    named_barriers,
                    defined,
                    num_warps,
                    next_token,
                    in_scheduler_impl,
                    scheduler_loop_depth,
                    true,
                )?;
            }
            Stmt::If { cond, then_body } => {
                let mut vars = Vec::new();
                collect_vars(cond, &mut vars);
                require_defined(&vars, defined, "if condition")?;
                check_context(
                    then_body,
                    scope,
                    role_token,
                    barriers,
                    named_barriers,
                    defined,
                    num_warps,
                    next_token,
                    in_scheduler_impl,
                    scheduler_loop_depth,
                    inside_role,
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Walk 3: `_check_tcgen05_cta_group_consistency`.
fn check_cta_group_consistency(body: &[Stmt]) -> R {
    let mut group: Option<u8> = None;
    fn walk(body: &[Stmt], group: &mut Option<u8>) -> R {
        for s in body {
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
                walk(child, group)?;
            }
        }
        Ok(())
    }
    walk(body, &mut group)
}

/// Walk 4: TMEM allocation bands. TMEM is not a tensor: an allocation declares
/// a column band `[base_col, base_col + n_cols)` and every TMEM operand is an
/// absolute physical (lane, col) address whose extent the op implies. With
/// constant addresses this statically proves: (a) concurrently live bands
/// never overlap, (b) a dealloc matches a live band, (c) every operand's
/// column span lands inside a band that is live at that program point. The
/// interpreter re-checks all of this on the evaluated addresses at run time —
/// this walk only rejects what is statically provably wrong.
fn check_tmem_alloc_bands(kernel: &Kernel) -> R {
    /// One TMEM operand use with its static (lane, column) extent, when known.
    struct Use<'a> {
        op: &'a TmemOperand,
        rows: Option<usize>,
        cols: Option<usize>,
        label: &'static str,
    }

    fn tmem_uses(s: &Stmt) -> Result<Vec<Use<'_>>, IrError> {
        let mut uses = Vec::new();
        match s {
            Stmt::Tcgen05Mma {
                dst,
                a,
                b,
                m,
                n,
                k,
                trans_a,
                trans_b,
                cta_group,
                sfa,
                sfb,
                sf_block,
                ..
            } => {
                // The accumulator is f32: one cell per (lane, n-column). Layout B
                // (cta_group=2, m=128) splits n in half per CTA — the two n-halves
                // stack into the lower/upper 64 lanes over the SAME column range,
                // so the per-CTA column span is ceil(n/2), not n (mma_blocks).
                let dst_cols = if *cta_group == 2 && *m == 128 {
                    (*n as usize).div_ceil(2)
                } else {
                    *n as usize
                };
                uses.push(Use {
                    op: dst,
                    rows: None,
                    cols: Some(dst_cols),
                    label: "tcgen05_mma dst",
                });
                let kk = *k as usize;
                let a_rows = (if *cta_group == 1 { *m } else { m / 2 }) as usize;
                let b_rows = (if *cta_group == 1 { *n } else { n / 2 }) as usize;
                for (op, rows, cols, label) in [
                    (
                        a,
                        if *trans_a { kk } else { a_rows },
                        if *trans_a { a_rows } else { kk },
                        "tcgen05_mma a",
                    ),
                    (
                        b,
                        if *trans_b { kk } else { b_rows },
                        if *trans_b { b_rows } else { kk },
                        "tcgen05_mma b",
                    ),
                ] {
                    if let MmaOperand::Tmem(t) = op {
                        let cells = match t.dtype {
                            // f16/bf16 pack two elements per 32-bit cell.
                            DType::F16 | DType::Bf16 => {
                                if cols % 2 != 0 {
                                    return Err(err(format!(
                                        "{label} packed-half TMEM operand needs an even column extent"
                                    )));
                                }
                                cols / 2
                            }
                            _ => cols,
                        };
                        uses.push(Use {
                            op: t,
                            rows: Some(rows),
                            cols: Some(cells),
                            label,
                        });
                    }
                }
                // Scale vectors: rows fold into 128-lane super-blocks of
                // `nblocks` columns each (see read_scale_blocks).
                let nblocks = if *sf_block == 0 {
                    1
                } else {
                    (*k / *sf_block) as usize
                };
                for (sf, rows, label) in [
                    (sfa, a_rows, "tcgen05_mma sfa"),
                    (sfb, *n as usize, "tcgen05_mma sfb"),
                ] {
                    if let Some(sf) = sf {
                        uses.push(Use {
                            op: sf,
                            rows: None,
                            cols: Some(rows.div_ceil(128) * nblocks),
                            label,
                        });
                    }
                }
            }
            Stmt::Tcgen05Cp { dst, src, .. } => {
                // The copy enumerates the src tile lane-major from (lane 0,
                // col): `count` elements land in ceil(count / 128) columns.
                let cols = static_shape_numel(&src.shape).map(|e| e.div_ceil(128));
                uses.push(Use {
                    op: dst,
                    rows: None,
                    cols,
                    label: "tcgen05_cp dst",
                });
            }
            Stmt::Tcgen05Ld {
                src, shape, num, ..
            } => uses.push(Use {
                op: src,
                rows: None,
                cols: tmem_operand_lanes_cols(shape, *num),
                label: "tcgen05_ld src",
            }),
            Stmt::Tcgen05St {
                dst, shape, num, ..
            } => uses.push(Use {
                op: dst,
                rows: None,
                cols: tmem_operand_lanes_cols(shape, *num),
                label: "tcgen05_st dst",
            }),
            _ => {}
        }
        Ok(uses)
    }

    fn walk(stmts: &[Stmt], live: &mut Vec<(i64, i64)>) -> R {
        for s in stmts {
            match s {
                Stmt::TmemAlloc {
                    base_col, n_cols, ..
                } => {
                    let (b, n) = (i64::from(*base_col), i64::from(*n_cols));
                    for &(lb, ln) in live.iter() {
                        if b < lb + ln && lb < b + n {
                            return bail("tmem_alloc column band overlaps a live allocation");
                        }
                    }
                    live.push((b, n));
                }
                Stmt::TmemDealloc {
                    base_col, n_cols, ..
                } => {
                    let band = (i64::from(*base_col), i64::from(*n_cols));
                    let Some(pos) = live.iter().position(|&x| x == band) else {
                        return bail("tmem_dealloc does not match a live allocation");
                    };
                    live.remove(pos);
                }
                _ => {}
            }
            for u in tmem_uses(s)? {
                if let (Some(r), Some(rows)) = (static_int(&u.op.row), u.rows) {
                    if r < 0 || r + rows as i64 > 128 {
                        return bail(format!("{} lane span escapes the 128 TMEM lanes", u.label));
                    }
                }
                if let (Some(c), Some(cols)) = (static_int(&u.op.col), u.cols) {
                    if cols > 0
                        && !live
                            .iter()
                            .any(|&(b, n)| b <= c && c + cols as i64 <= b + n)
                    {
                        return bail(format!(
                            "{} column span is not inside a live tmem allocation band",
                            u.label
                        ));
                    }
                }
            }
            for body in s.child_bodies() {
                walk(body, live)?;
            }
        }
        Ok(())
    }

    let mut live = Vec::new();
    walk(&kernel.body, &mut live)
}
/// Walk 5: the `leader_routed` IR flag is the ONLY authority on cluster
/// TMA-completion routing — codegen honors it and never guesses it from the
/// usage structure. Consistency rules for a leader-routed mbar:
///   1. it must carry a peer reference (an `MBarRef` with `remote_coord` — the
///      cross-CTA wait the leader routing replaces), and
///   2. it must be named ONLY by TMA-transaction ops (`TmaLoad`,
///      `MBarrierExpectTx`, `MBarrierArriveExpectTx`) plus the bookkeeping ops
///      that keep their local form (`MBarrierInit`, `MBarrierWait`). Routing a
///      thread arrive, a tcgen05 commit, or a CLC/S2Cluster completion to the
///      leader would corrupt the barrier's accounting.
fn check_leader_routed_mbars(kernel: &Kernel) -> R {
    let mut leader_ids: HashSet<u32> = HashSet::new();
    let mut has_peer_ref: HashSet<u32> = HashSet::new();
    fn walk(stmts: &[Stmt], leader_ids: &HashSet<u32>, has_peer: &mut HashSet<u32>) -> R {
        for s in stmts {
            let (refs, is_tx_use): (Vec<&MBarRef>, bool) = match s {
                Stmt::MBarrierInit { mbar, .. }
                | Stmt::MBarrierWait { mbar, .. }
                | Stmt::MBarrierExpectTx { mbar, .. }
                | Stmt::MBarrierArriveExpectTx { mbar, .. }
                | Stmt::TmaLoad { mbar, .. } => (vec![mbar], true),
                Stmt::MBarrierArrive { mbar, .. }
                | Stmt::Tcgen05Commit { mbar, .. }
                | Stmt::ClcTryCancel { mbar, .. }
                | Stmt::CpAsyncBulkS2Cluster { mbar, .. } => (vec![mbar], false),
                _ => (vec![], true),
            };
            for mref in refs {
                if mref.remote_coord.is_some() {
                    has_peer.insert(mref.mbar.id);
                }
                if leader_ids.contains(&mref.mbar.id) && !is_tx_use {
                    return bail(format!(
                        "leader_routed mbar {} must only be used by TmaLoad/expect_tx \
                         (plus init/wait) — an arrive/commit/CLC completion cannot be \
                         leader-routed",
                        mref.mbar.id
                    ));
                }
            }
            for child in s.child_bodies() {
                walk(child, leader_ids, has_peer)?;
            }
        }
        Ok(())
    }
    fn collect_defs(stmts: &[Stmt], leader_ids: &mut HashSet<u32>) {
        for s in stmts {
            if let Stmt::MBarDef { mbar } = s {
                if mbar.leader_routed {
                    leader_ids.insert(mbar.id);
                }
            }
            for child in s.child_bodies() {
                collect_defs(child, leader_ids);
            }
        }
    }
    collect_defs(&kernel.body, &mut leader_ids);
    walk(&kernel.body, &leader_ids, &mut has_peer_ref)?;
    for id in leader_ids {
        if !has_peer_ref.contains(&id) {
            return bail(format!(
                "leader_routed mbar {id} must carry a peer reference (an MBarRef with \
                 remote_coord — the cross-CTA wait the leader routing replaces)"
            ));
        }
    }
    Ok(())
}

fn check_smem_pool_bounds(kernel: &Kernel) -> R {
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
        Ok(())
    }

    fn walk_stmt(stmt: &Stmt, smem_size_bytes: usize, seen: &mut HashSet<u32>) -> R {
        match stmt {
            Stmt::TensorDef { tensor } => {
                if seen.insert(tensor.id) {
                    check_tensor(tensor, smem_size_bytes)?;
                }
            }
            Stmt::ScalarDef {
                initial: ScalarInitial::Tensor(slice),
                ..
            } => {
                if seen.insert(slice.tensor.id) {
                    check_tensor(&slice.tensor, smem_size_bytes)?;
                }
            }
            Stmt::TmaLoad { dst, src, .. } => {
                if seen.insert(dst.tensor.id) {
                    check_tensor(&dst.tensor, smem_size_bytes)?;
                }
                if seen.insert(src.id) {
                    check_tensor(src, smem_size_bytes)?;
                }
            }
            Stmt::TmaStore { dst, src, .. } => {
                if seen.insert(dst.id) {
                    check_tensor(dst, smem_size_bytes)?;
                }
                if seen.insert(src.tensor.id) {
                    check_tensor(&src.tensor, smem_size_bytes)?;
                }
            }
            Stmt::Tcgen05Mma { a, b, .. } => {
                // TMEM operands carry no tensor; only SMEM operand tiles are
                // pool-bound (dst/sfa/sfb are TmemOperands now).
                for op in [a, b] {
                    if let MmaOperand::Slice(s) = op {
                        if seen.insert(s.tensor.id) {
                            check_tensor(&s.tensor, smem_size_bytes)?;
                        }
                    }
                }
            }
            Stmt::Tcgen05Cp { src, .. } => {
                if seen.insert(src.tensor.id) {
                    check_tensor(&src.tensor, smem_size_bytes)?;
                }
            }
            Stmt::Tcgen05Ld { dst, .. } => {
                if seen.insert(dst.tensor.id) {
                    check_tensor(&dst.tensor, smem_size_bytes)?;
                }
            }
            Stmt::Tcgen05St { src, .. } => {
                if seen.insert(src.tensor.id) {
                    check_tensor(&src.tensor, smem_size_bytes)?;
                }
            }
            Stmt::LdMatrix { dst, src, .. } | Stmt::StMatrix { dst, src, .. } => {
                for tensor in [&dst.tensor, &src.tensor] {
                    if seen.insert(tensor.id) {
                        check_tensor(tensor, smem_size_bytes)?;
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
                    if seen.insert(slice.tensor.id) {
                        check_tensor(&slice.tensor, smem_size_bytes)?;
                    }
                }
            }
            Stmt::StoreScalar { dst, .. } => {
                if seen.insert(dst.tensor.id) {
                    check_tensor(&dst.tensor, smem_size_bytes)?;
                }
            }
            Stmt::WarpMma { d, a, b, c, .. } => {
                for sl in [d, a, b, c] {
                    if seen.insert(sl.tensor.id) {
                        check_tensor(&sl.tensor, smem_size_bytes)?;
                    }
                }
            }
            _ => {}
        }
        for child in stmt.child_bodies() {
            for nested in child {
                walk_stmt(nested, smem_size_bytes, seen)?;
            }
        }
        Ok(())
    }

    let mut seen = HashSet::new();
    for tensor in &kernel.args {
        if seen.insert(tensor.id) {
            check_tensor(tensor, kernel.smem_size_bytes)?;
        }
    }
    for stmt in &kernel.body {
        walk_stmt(stmt, kernel.smem_size_bytes, &mut seen)?;
    }
    Ok(())
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
        check_smem_pool_bounds(self)?;
        let mut defined = HashSet::new();
        let mut barriers = HashMap::new();
        let mut named_barriers: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut next_token = 0u32;
        check_context(
            &self.body,
            Scope::Cta,
            None,
            &mut barriers,
            &mut named_barriers,
            &mut defined,
            self.num_warps,
            &mut next_token,
            false,
            0,
            false,
        )?;
        check_cta_group_consistency(&self.body)?;
        check_tmem_alloc_bands(self)?;
        check_leader_routed_mbars(self)?;
        Ok(())
    }
}
