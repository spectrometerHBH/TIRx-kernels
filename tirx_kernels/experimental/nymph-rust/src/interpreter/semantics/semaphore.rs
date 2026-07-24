//! GMEM semaphore executors — `gmem_atomic_add` ("signal") and `gmem_wait_eq`
//! ("wait"). These are the two value-keyed release/acquire primitives that
//! serialize a cross-CTA reduce-add into a bit-reproducible order AND give the
//! protocol checker the happens-before edge it needs to ORDER the reduce-adds.
//!
//! Both are SYNC ops — like an mbar arrive/wait, they emit a SemRelease/SemAcquire
//! trace event and do NOT emit a data Read/Write on the semaphore tensor (so the
//! race checker never flags the semaphore itself). The HB plumbing lives in
//! `checker.rs::OrderingAnalysis::new`.

use super::super::diagnostics::{IResult, InterpreterError};
use super::super::outcomes::{StepStatus, WakeCondition};
use super::super::protocol::{GmemAtomicOrderEvent, SemKey, TraceEventKind};
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::values::tensors::row_major_strides;
use super::super::warp_context::WarpContext;
use crate::ir::{GmemAtomicOrder, MemorySpace, Stmt, TensorSlice};

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::GmemAtomicAdd, execute_gmem_atomic_add);
    reg.register(StmtKind::GmemWaitEq, execute_gmem_wait_eq);
}

/// Resolve `(tensor_id, flat_coord)` for an i32 GMEM semaphore cell from the
/// slice's tensor + the per-dim cell coords. The semaphore is SHARED across CTAs,
/// so the key carries no cta_id — it is global by tensor identity.
fn resolve_sem_key(
    ctx: &WarpContext,
    sem: &TensorSlice,
    coords: &[crate::ir::ScalarValue],
    label: &str,
) -> IResult<SemKey> {
    if sem.tensor.space != MemorySpace::Gmem {
        return Err(InterpreterError::new(
            "semaphore_space",
            "GMEM semaphore must live in GMEM",
        ));
    }
    if coords.len() != sem.tensor.shape.len() {
        return Err(InterpreterError::new(
            "semaphore_coords",
            "semaphore coords rank must match the semaphore tensor rank",
        ));
    }
    let coord: Vec<usize> = coords
        .iter()
        .map(|c| {
            ctx.eval_scalar_uniform(c, label, "divergent_semaphore_operands")
                .map(|x| x as usize)
        })
        .collect::<IResult<Vec<_>>>()?;
    for (&c, &dim) in coord.iter().zip(sem.tensor.shape.iter()) {
        if c >= dim {
            return Err(InterpreterError::new(
                "semaphore_coords",
                "semaphore coord is out of bounds",
            ));
        }
    }
    let strides = row_major_strides(&sem.tensor.shape);
    let flat_coord: usize = coord.iter().zip(strides.iter()).map(|(c, s)| c * s).sum();
    Ok(SemKey {
        tensor_id: sem.tensor.id,
        flat_coord,
    })
}

/// `red.<order>.gpu.global.add.s32` — atomic-add "signal". VALUE: serialized RMW
/// `sem[coords] += value`. TRACE: a SemRelease SYNC event carrying the cell key +
/// the POST-increment value, so a later `wait_eq` on that exact value can join
/// this stream's release clock (value-keyed acquire).
///
/// A single-thread issue op in this model: `red.global` is per-thread PTX, so a
/// a multi-lane mask would RMW the counter once per thread — the kernel must
/// elect one thread (all known emitters signal from an elected lane).
fn execute_gmem_atomic_add<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    ctx.check_single_thread_issue("gmem_atomic_add_mask", "gmem_atomic_add")?;
    let (sem, coords, value, order) = match stmt {
        Stmt::GmemAtomicAdd {
            sem,
            coords,
            value,
            order,
        } => (sem, coords, value, order),
        _ => unreachable!(),
    };
    let key = resolve_sem_key(ctx, sem, coords, "gmem_atomic_add coords")?;
    let add = ctx.eval_scalar_uniform(
        value,
        "gmem_atomic_add value",
        "divergent_semaphore_operands",
    )?;
    // Serialized RMW (the interpreter applies one stream's atomic at a time).
    // Run in both modes so the trace's release event carries the same
    // post-increment counter the value run produces.
    let new_value = ctx.state.values.semaphores.atomic_add(key, add);
    if ctx.trace_mode() {
        let order_event = match order {
            GmemAtomicOrder::Release => GmemAtomicOrderEvent::Release,
            GmemAtomicOrder::Relaxed => GmemAtomicOrderEvent::Relaxed,
        };
        ctx.emit(TraceEventKind::SemRelease {
            key,
            new_value,
            order: order_event,
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}

/// `ld.global.acquire.gpu` spin until `sem[coords] == value` — "wait". VALUE:
/// BLOCK (polled re-check) until the counter equals `value`; if it never does, the
/// runner's deadlock detection fires (a wrong lock-value is caught, not hidden).
/// TRACE: a SemAcquire SYNC event — joins the release clock published by the
/// atomic_add that PRODUCED this exact `value`. The event is emitted ONCE, when the
/// wait is satisfied, so the acquired clock reflects the producing release.
///
/// Any lane mask may spin (the load is read-only and the awaited value is
/// uniform across the lanes, so every lane observes the same satisfaction) — one blocking
/// check and one SemAcquire per executing mask.
fn execute_gmem_wait_eq<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (sem, coords, value) = match stmt {
        Stmt::GmemWaitEq { sem, coords, value } => (sem, coords, value),
        _ => unreachable!(),
    };
    let key = resolve_sem_key(ctx, sem, coords, "gmem_wait_eq coords")?;
    let want =
        ctx.eval_scalar_uniform(value, "gmem_wait_eq value", "divergent_semaphore_operands")?;
    let current = ctx.state.values.semaphores.get(key);
    if current != want {
        // Not our turn yet — yield and re-run next scheduler round (the polled
        // rendezvous machinery). Another stream's atomic_add will bump the
        // counter; if the awaited value is never produced, no progress in a
        // full round -> deadlock (the faithful behavior for a wrong lock-value).
        return Ok(StepStatus::block(WakeCondition::Polled));
    }
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::SemAcquire {
            key,
            value: want,
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}
