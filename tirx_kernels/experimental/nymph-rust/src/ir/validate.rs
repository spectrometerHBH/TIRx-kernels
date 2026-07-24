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

/// `_check_tcgen05_mma_shape`.
fn check_mma_shape(m: u32, n: u32, k: u32, cta_group: u8, block_scaled_f8: bool) -> R {
    check_positive(m, "tcgen05_mma m")?;
    check_positive(n, "tcgen05_mma n")?;
    check_positive(k, "tcgen05_mma k")?;
    // The k rule itself is per operand KIND and lives at the Tcgen05Mma arm
    // (which sees the dtypes and fp4 flags): dense f16/bf16 is k=16, the
    // block-scaled f8 instruction is k=32, fp4 (mxf4) is k in {64, 128, 256}.
    if !matches!(k, 16 | 32 | 64 | 128 | 256) {
        return bail(
            "tcgen05_mma k must be 16 (dense f16/bf16), 32 (block-scaled f8), or 64/128/256 (fp4)",
        );
    }
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
    if let Some(Layout::Tmem(tm)) = &t.layout {
        if tm.lane_align != 0 && tm.lane_align != 16 {
            return bail("tmem layout lane_align must be 0 or 16");
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
            tensor,
            n_cols,
            cta_group,
        }
        | Stmt::TmemDealloc {
            tensor,
            n_cols,
            cta_group,
        } => {
            if tensor.space != MemorySpace::Tmem {
                return bail("tmem_alloc/dealloc tensor must be TMEM");
            }
            check_tmem_cols(*n_cols, "tmem n_cols")?;
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
            if let Some(v) = phase {
                validate_scalar(v)?;
            }
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
        }
        Stmt::TmaStore {
            dst,
            src,
            coords,
            shape,
            gmem_shape,
            reduce_add,
            allow_nondet_reduce,
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
            ..
        } => {
            validate_slice(dst, "tcgen05_mma dst")?;
            validate_slice(a, "tcgen05_mma a")?;
            validate_slice(b, "tcgen05_mma b")?;
            check_cta_group(*cta_group, "tcgen05_mma cta_group")?;
            check_mma_shape(
                *m,
                *n,
                *k,
                *cta_group,
                (sfa.is_some() || sfb.is_some()) && !*a_fp4,
            )?;
            if dst.tensor.space != MemorySpace::Tmem {
                return bail("tcgen05_mma dst must be TMEM");
            }
            if !matches!(a.tensor.space, MemorySpace::Smem | MemorySpace::Tmem)
                || !matches!(b.tensor.space, MemorySpace::Smem | MemorySpace::Tmem)
            {
                return bail("tcgen05_mma operands must be SMEM or TMEM");
            }
            if *a_fp4 || *b_fp4 {
                // NVFP4 (mxf4): operands are e2m1 fp4 packed 2-per-u8; both must be fp4.
                if !*a_fp4 || !*b_fp4 {
                    return bail("tcgen05_mma a_fp4 and b_fp4 must be set together");
                }
                if a.tensor.dtype != DType::U8 || b.tensor.dtype != DType::U8 {
                    return bail("tcgen05_mma fp4 operands must be u8 (2 packed e2m1 per byte)");
                }
                // The fp4 value model is the in-place SMEM datapath: PTX Table 54's
                // mxf4* shapes have no transposed form and exist only as
                // (cta_group=1, M=128) or (cta_group=2, M=256).
                if a.tensor.space != MemorySpace::Smem || b.tensor.space != MemorySpace::Smem {
                    return bail("tcgen05_mma fp4 operands must be SMEM");
                }
                if *trans_a || *trans_b {
                    return bail("tcgen05_mma fp4 (mxf4) does not support trans_a/trans_b");
                }
                if !((*cta_group == 1 && *m == 128) || (*cta_group == 2 && *m == 256)) {
                    return bail(
                        "tcgen05_mma fp4 requires (cta_group=1, m=128) or (cta_group=2, m=256)",
                    );
                }
                if !matches!(*k, 64 | 128 | 256) {
                    return bail("tcgen05_mma fp4 (mxf4) k must be 64, 128, or 256");
                }
            } else {
                if !matches!(a.tensor.dtype, DType::F16 | DType::Bf16 | DType::F8E4M3)
                    || !matches!(b.tensor.dtype, DType::F16 | DType::Bf16 | DType::F8E4M3)
                {
                    return bail("tcgen05_mma operand dtype must be f16, bf16, or f8e4m3");
                }
                if a.tensor.dtype != b.tensor.dtype {
                    return bail("tcgen05_mma a and b operand dtype must match");
                }
                if (*k == 32) != (a.tensor.dtype == DType::F8E4M3) {
                    return bail(
                        "tcgen05_mma k=32 is the f8e4m3 instruction shape (k=16 for f16/bf16)",
                    );
                }
            }
            if dst.tensor.dtype != DType::F32 {
                return bail("tcgen05_mma dst dtype must be f32");
            }
            let dst_rows = if *cta_group == 1 { *m } else { 128 };
            let a_rows = if *cta_group == 1 { *m } else { m / 2 };
            let b_rows = if *cta_group == 1 { *n } else { n / 2 };
            check_slice_covers(dst, &[dst_rows as usize, *n as usize], "tcgen05_mma dst")?;
            // fp4 operands are packed 2-per-byte, so the K (contraction) extent in
            // the SMEM tile is k/2 bytes, not k elements.
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
            check_slice_covers_trailing(a, &a_shape, "tcgen05_mma a")?;
            check_slice_covers_trailing(b, &b_shape, "tcgen05_mma b")?;
            match (sfa, sfb) {
                (None, None) => {
                    if a.tensor.dtype == DType::F8E4M3 {
                        return bail("tcgen05_mma f8e4m3 operands require sfa/sfb scale vectors");
                    }
                    if *sf_e4m3 || *sf_block != 0 {
                        return bail("tcgen05_mma sf_e4m3/sf_block require sfa/sfb scale vectors");
                    }
                }
                (Some(sfa), Some(sfb)) => {
                    // PTX: the block-scaled kinds (mxf8f6f4/mxf4) with cta_group::1
                    // exist only at M=128 — an m=64 cg1 MMA has no scale mode.
                    if *cta_group == 1 && *m == 64 {
                        return bail(
                            "tcgen05_mma m=64 cta_group=1 does not support block-scaled \
                             (sfa/sfb) modes",
                        );
                    }
                    // UE8M0 path requires f8e4m3 operands; NVFP4 (sf_e4m3) uses fp4 operands.
                    if !*sf_e4m3 && a.tensor.dtype != DType::F8E4M3 {
                        return bail("tcgen05_mma sfa/sfb require f8e4m3 operands");
                    }
                    if *sf_e4m3 && !*a_fp4 {
                        return bail("tcgen05_mma sf_e4m3 (NVFP4) requires fp4 operands");
                    }
                    if *sf_byte >= 4 {
                        return bail("tcgen05_mma sf_byte must be in 0..4");
                    }
                    // The NVFP4 e4m3 decode reads scale bytes 0..k/16 of each cell —
                    // sf_byte exists only in the packed-UE8M0 (fp8) layout.
                    if *sf_e4m3 && *sf_byte != 0 {
                        return bail("tcgen05_mma sf_byte must be 0 for sf_e4m3 (NVFP4) scales");
                    }
                    // The two supported scale modes fix sf_block: nvfp4 block-16
                    // (sf_e4m3) or fp8 per-row (sf_block=0). Anything else would be
                    // silently mis-divided by the k/sf_block block math downstream.
                    match (*sf_e4m3, *sf_block) {
                        (true, 16) => {
                            // The packed-u32 scale cell holds 4 e4m3 bytes; this MMA
                            // reads bytes 0..k/16, so k/16 must fit in one cell.
                            if *k % 16 != 0 || *k / 16 > 4 {
                                return bail(
                                    "tcgen05_mma nvfp4 k must be a multiple of sf_block=16 \
                                     with at most 4 blocks per packed-u32 scale cell",
                                );
                            }
                        }
                        (true, _) => return bail("tcgen05_mma sf_e4m3 requires sf_block=16"),
                        (false, 0) => {}
                        (false, _) => {
                            return bail("tcgen05_mma UE8M0 (fp8) mode requires sf_block=0")
                        }
                    }
                    for (sf, rows, label) in [
                        (sfa, a_rows as usize, "tcgen05_mma sfa"),
                        (sfb, *n as usize, "tcgen05_mma sfb"),
                    ] {
                        validate_slice(sf, label)?;
                        if sf.tensor.space != MemorySpace::Tmem {
                            return bail(format!("{label} must be TMEM"));
                        }
                        if sf.tensor.dtype != DType::U32 {
                            return bail(format!(
                                "{label} dtype must be u32 (4 packed UE8M0/e4m3 scale bytes)"
                            ));
                        }
                        let Some(shape) = static_slice_shape(sf) else {
                            return bail(format!("{label} shape must be static"));
                        };
                        if shape.len() != 2 || shape[0] != 128 {
                            return bail(format!("{label} must be a (128, cols) TMEM slice"));
                        }
                        if shape[0] * shape[1] < rows {
                            return bail(format!("{label} does not cover the scaled rows"));
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
            validate_slice(dst, "tcgen05_cp dst")?;
            validate_slice(src, "tcgen05_cp src")?;
            check_cta_group(*cta_group, "tcgen05_cp cta_group")?;
            if dst.tensor.space != MemorySpace::Tmem {
                return bail("tcgen05_cp dst must be TMEM");
            }
            if src.tensor.space != MemorySpace::Smem {
                return bail("tcgen05_cp src must be SMEM");
            }
            if dst.tensor.dtype != DType::U32 || src.tensor.dtype != DType::U32 {
                return bail("tcgen05_cp moves packed u32 scale cells");
            }
            let (Some(dst_shape), Some(src_shape)) =
                (static_slice_shape(dst), static_slice_shape(src))
            else {
                return bail("tcgen05_cp slice shapes must be static");
            };
            if dst_shape.len() != 2 || dst_shape[0] != 128 {
                return bail("tcgen05_cp dst must be a (128, cols) TMEM slice");
            }
            let dst_numel: usize = dst_shape.iter().product();
            let src_numel: usize = src_shape.iter().product();
            if dst_numel != src_numel {
                return bail("tcgen05_cp src/dst element counts must match");
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
            row,
            col,
        } => {
            validate_slice(dst, "tcgen05_ld dst")?;
            if dst.tensor.space != MemorySpace::Reg {
                return bail("tcgen05_ld dst must be REG");
            }
            if src.space != MemorySpace::Tmem {
                return bail("tcgen05_ld src must be TMEM");
            }
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
            validate_scalar(row)?;
            validate_scalar(col)?;
        }
        Stmt::Tcgen05WaitLd => {}
        Stmt::Tcgen05St {
            dst,
            src,
            shape,
            num,
            row,
            col,
        } => {
            validate_slice(src, "tcgen05_st src")?;
            if dst.space != MemorySpace::Tmem {
                return bail("tcgen05_st dst must be TMEM");
            }
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
            validate_scalar(row)?;
            validate_scalar(col)?;
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
            Stmt::TmemAlloc { .. } | Stmt::TmemDealloc { .. } => {
                if let Some(set) = filter.known() {
                    if set.is_exactly_one_full_warp().is_none() {
                        return bail("tmem alloc/dealloc must be issued by exactly one full warp");
                    }
                }
            }
            Stmt::SetMaxNReg { .. } => match filter.known() {
                None => {
                    return bail("setmaxnreg requires a statically-resolvable thread branch");
                }
                Some(set) => {
                    if !set.is_union_of_full_warpgroups() {
                        return bail("setmaxnreg must cover whole warpgroup(s)");
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
                    defined,
                    num_warps,
                    in_scheduler_impl,
                    scheduler_loop_depth,
                    barriers,
                )?;
            }
            Stmt::SchedulerImpl { body, .. } => {
                check_context(body, filter, defined, num_warps, true, 0, barriers)?;
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
            Stmt::Loop { body } => {
                check_context(
                    body,
                    filter,
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
                let inner = match static_thread_filter(cond, num_warps) {
                    ThreadFilter::Known(set) => narrow(filter, &set),
                    ThreadFilter::Unknown => ThreadFilter::Unknown,
                };
                check_context(
                    then_body,
                    &inner,
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

/// Intersect the enclosing filter with a statically-known branch set. An
/// Unknown enclosing filter stays Unknown: the true set is SOME subset of the
/// branch set, which is not enough for the exact-shape rules above.
fn narrow(outer: &ThreadFilter, branch: &ThreadSet) -> ThreadFilter {
    match outer {
        ThreadFilter::Known(set) => ThreadFilter::Known(set.intersect(branch)),
        ThreadFilter::Unknown => ThreadFilter::Unknown,
    }
}

/// Walk 3: `_check_tcgen05_cta_group_consistency`.
fn check_cta_group_consistency(body: &[Stmt]) -> R {
    let mut group: Option<u8> = None;
    fn walk(body: &[Stmt], group: &mut Option<u8>) -> R {
        for s in body {
            let g = match s {
                Stmt::TmemAlloc { cta_group, .. }
                | Stmt::TmemDealloc { cta_group, .. }
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
            Stmt::TmemAlloc { tensor, .. } | Stmt::TmemDealloc { tensor, .. } => {
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
            Stmt::Tcgen05Mma {
                dst,
                a,
                b,
                sfa,
                sfb,
                ..
            } => {
                let mut tensors = vec![&dst.tensor, &a.tensor, &b.tensor];
                if let Some(sf) = sfa {
                    tensors.push(&sf.tensor);
                }
                if let Some(sf) = sfb {
                    tensors.push(&sf.tensor);
                }
                for tensor in tensors {
                    if seen.insert(tensor.id) {
                        check_tensor(tensor, smem_size_bytes)?;
                    }
                }
            }
            Stmt::Tcgen05Cp { dst, src, .. } => {
                for tensor in [&dst.tensor, &src.tensor] {
                    if seen.insert(tensor.id) {
                        check_tensor(tensor, smem_size_bytes)?;
                    }
                }
            }
            Stmt::Tcgen05Ld { dst, src, .. } => {
                if seen.insert(dst.tensor.id) {
                    check_tensor(&dst.tensor, smem_size_bytes)?;
                }
                if seen.insert(src.id) {
                    check_tensor(src, smem_size_bytes)?;
                }
            }
            Stmt::Tcgen05St { dst, src, .. } => {
                if seen.insert(dst.id) {
                    check_tensor(dst, smem_size_bytes)?;
                }
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
            Stmt::WarpMma { d, a, b, c, .. } => {
                for sl in [d, a, b, c] {
                    if seen.insert(sl.tensor.id) {
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
        check_context(
            &self.body,
            &ThreadFilter::Known(ThreadSet::full(self.num_warps)),
            &mut defined,
            self.num_warps,
            false,
            0,
            &mut BarrierClassUse::default(),
        )?;
        check_cta_group_consistency(&self.body)?;
        Ok(())
    }
}
