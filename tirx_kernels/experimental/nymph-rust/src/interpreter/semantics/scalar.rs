//! Scalar def/store executors — port of `semantics/scalar.py`. Write each
//! thread's scalar directly (not eager, unlike loop vars).

use super::super::diagnostics::{IResult, InterpreterError};
use super::super::outcomes::StepStatus;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::scalar_eval::eval_scalar_in_env;
use super::super::transfer::write_operand;
use super::super::values::arrays::ValueArray1;
use super::super::values::tensors::tensor_instance_key;
use super::super::warp_context::WarpContext;
use crate::ir::{MemorySpace, ScalarInitial, Stmt};
use ndarray::Array1;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::ScalarDef, execute_scalar_def);
    reg.register(StmtKind::ScalarStore, execute_scalar_store);
    reg.register(StmtKind::ScalarLet, execute_scalar_let);
    reg.register(StmtKind::StoreScalar, execute_store_scalar);
    reg.register(StmtKind::ShuffleSync, execute_shuffle_sync);
}

/// Warp shuffle/broadcast: `var` = `src` evaluated on lane `src_lane` of each warp,
/// broadcast to all lanes (a faithful `__shfl_sync`). If `src` is warp-uniform the
/// broadcast is a no-op (every lane already holds the same value); if it is NOT, the
/// broadcast changes it — and because this runs in the value model, the resulting
/// mismatch is caught. The shuffle is per warp (the 32-lane `__shfl_sync` width), so
/// the source value is taken from the same (cta, warp).
fn execute_shuffle_sync<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (var, src, src_lane) = match stmt {
        Stmt::ShuffleSync { var, src, src_lane } => (var, src, src_lane),
        _ => unreachable!(),
    };
    let var_id = var.id.0;
    let per_thread = ctx.eval_scalar_vec(src)?.to_vec();
    let lanes = ctx.eval_scalar_vec(src_lane)?.to_vec();
    let mut by_lane: std::collections::HashMap<(usize, usize, usize), i64> =
        std::collections::HashMap::new();
    for (i, t) in ctx.lanes.iter().enumerate() {
        by_lane.insert((t.cta_id, t.warp_id, t.lane_id), per_thread[i]);
    }
    let mut out = Vec::with_capacity(ctx.lanes.len());
    for (i, t) in ctx.lanes.iter().enumerate() {
        // The shfl source lane must address the issuing warp's own 32 lanes —
        // an out-of-range lane is hardware UB (the old code silently clamped a
        // negative lane to 0 and computed the wrong broadcast).
        let sl = lanes[i];
        if !(0..32).contains(&sl) {
            return Err(InterpreterError::new(
                "shuffle_sync_lane_range",
                "shuffle_sync source lane is outside [0, 32)",
            ));
        }
        let v = by_lane
            .get(&(t.cta_id, t.warp_id, sl as usize))
            .copied()
            .ok_or_else(|| {
                InterpreterError::new(
                    "shuffle_sync_lane_missing",
                    "shuffle_sync source lane is not in the executing warp's lane mask",
                )
            })?;
        out.push(v);
    }
    Ok(scalar_commit(ctx, var_id, &out))
}

fn execute_scalar_def<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (var, initial) = match stmt {
        Stmt::ScalarDef { var, initial } => (var, initial),
        _ => unreachable!(),
    };
    let var_id = var.id.0;
    let values: Vec<i64> = match initial {
        ScalarInitial::Value(v) => ctx.eval_scalar_vec(v)?.to_vec(),
        ScalarInitial::Tensor(slice) => {
            if slice.tensor.space == MemorySpace::Smem {
                let resolved = ctx.eval_slice(slice)?;
                if ctx.trace_mode() {
                    ctx.emit_tensor_read(&resolved)?;
                }
                let values = match ctx.shared_read(&resolved) {
                    Ok(values) => values.to_i64_compute(),
                    Err(e)
                        if ctx.trace_mode()
                            && matches!(e.code.as_str(), "missing_tensor_value") =>
                    {
                        return Err(ctx.trace_inconclusive(
                            "trace_control_from_skipped_payload",
                            "trace control read depends on skipped payload tensor data",
                        ));
                    }
                    Err(e) => return Err(e),
                };
                return Ok(scalar_commit(
                    ctx,
                    var_id,
                    &(0..values.nrows())
                        .map(|i| values[[i, 0]])
                        .collect::<Vec<_>>(),
                ));
            }
            let mut out = Vec::with_capacity(ctx.lanes.len());
            for t in ctx.lanes.clone().iter() {
                out.push(read_scalar_initial(ctx, slice, t)?);
            }
            out
        }
    };
    Ok(scalar_commit(ctx, var_id, &values))
}

fn execute_scalar_store<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (var, value) = match stmt {
        Stmt::ScalarStore { var, value } => (var, value),
        _ => unreachable!(),
    };
    let var_id = var.id.0;
    let values = ctx.eval_scalar_vec(value)?.to_vec();
    Ok(scalar_commit(ctx, var_id, &values))
}

/// `let` binding: value-wise identical to a scalar store (evaluate + bind). Its
/// single-assignment contract is a validate-time rule; the interpreter just binds.
fn execute_scalar_let<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (var, value) = match stmt {
        Stmt::ScalarLet { var, value } => (var, value),
        _ => unreachable!(),
    };
    let var_id = var.id.0;
    let values = ctx.eval_scalar_vec(value)?.to_vec();
    Ok(scalar_commit(ctx, var_id, &values))
}

fn execute_store_scalar<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (dst, value) = match stmt {
        Stmt::StoreScalar { dst, value } => (dst, value),
        _ => unreachable!(),
    };
    let resolved_dst = ctx.eval_slice(dst)?;
    let values = ctx.eval_scalar_vec(value)?.to_vec();
    let array = ValueArray1::from_i64_compute(Array1::from(values), dst.tensor.dtype)
        .reshape2((ctx.lanes.len(), 1))?;
    if ctx.trace_mode() {
        ctx.emit_tensor_write(&resolved_dst)?;
    }
    write_operand(ctx, &resolved_dst, &array)?;
    Ok(StepStatus::advance())
}

fn scalar_commit(ctx: &mut WarpContext, var_id: u32, values: &[i64]) -> StepStatus {
    ctx.state
        .values
        .scalars
        .write_values(&ctx.lanes, var_id, values);
    StepStatus::advance()
}

fn read_scalar_initial(
    ctx: &mut WarpContext,
    slice: &crate::ir::TensorSlice,
    thread: &super::super::threads::ThreadId,
) -> IResult<i64> {
    if slice.tensor.space != MemorySpace::Gmem {
        return Err(InterpreterError::new(
            "scalar_load",
            "scalar_def tensor initial must be GMEM",
        ));
    }
    let env = ctx.state.values.scalars.by_thread.get(thread);
    let empty = std::collections::HashMap::new();
    let env = env.unwrap_or(&empty);
    let offsets: Vec<usize> = slice
        .offsets
        .iter()
        .map(|o| eval_scalar_in_env(o, thread, env).map(|v| v as usize))
        .collect::<IResult<_>>()?;
    let shape: Vec<usize> = slice
        .shape
        .iter()
        .map(|s| eval_scalar_in_env(s, thread, env).map(|v| v as usize))
        .collect::<IResult<_>>()?;
    let key = tensor_instance_key(thread.cta_id, &slice.tensor)?;
    let Some(tv) = ctx.state.values.tensors.by_instance.get(&key) else {
        if ctx.trace_mode() {
            return Err(ctx.trace_inconclusive(
                "trace_control_from_skipped_payload",
                "trace control read depends on an unavailable tensor input",
            ));
        }
        return Err(InterpreterError::new(
            "missing_input",
            "scalar_def tensor initial is not loaded",
        ));
    };
    let block = match tv.read_block(&offsets, &shape) {
        Ok(block) => block,
        Err(e)
            if ctx.trace_mode()
                && matches!(e.code.as_str(), "missing_tensor_value" | "missing_input") =>
        {
            return Err(ctx.trace_inconclusive(
                "trace_control_from_skipped_payload",
                "trace control read depends on skipped payload tensor data",
            ));
        }
        Err(e) => return Err(e),
    };
    Ok(block.to_i64_compute()[0])
}
