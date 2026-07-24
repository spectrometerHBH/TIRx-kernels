//! Control-flow executors — port of `semantics/control.py`. All return
//! `advance_continue` (structural; no trace entry) and mutate the frame stack,
//! except the empty-body loop terminal write.

use super::super::diagnostics::{IResult, InterpreterError};
use super::super::outcomes::StepStatus;
use super::super::protocol::TraceEventKind;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::scheduler::{break_dynamic_loop, push_dynamic_loop, push_frame, push_loop};
use super::super::threads::canonical_thread_mask;
use super::super::warp_context::WarpContext;
use crate::ir::{SchedulerPolicy, Stmt};

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::ForLoop, execute_loop);
    reg.register(StmtKind::ForEachTask, execute_for_each_task);
    reg.register(StmtKind::SchedulerImpl, execute_scheduler_impl);
    reg.register(StmtKind::SchedNext, execute_sched_next);
    reg.register(StmtKind::Loop, execute_dynamic_loop);
    reg.register(StmtKind::BreakIf, execute_break_if);
    reg.register(StmtKind::If, execute_if);
}

fn execute_scheduler_impl<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let body = match stmt {
        Stmt::SchedulerImpl { body, .. } => body,
        _ => unreachable!(),
    };
    if !body.is_empty() {
        push_frame(ctx.stream, body.as_slice(), ctx.lanes.clone());
    }
    Ok(StepStatus::advance_continue())
}

fn execute_loop<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (var, start_v, stop_v, step_v, body) = match stmt {
        Stmt::ForLoop {
            var,
            start,
            stop,
            step,
            body,
        } => (var, start, stop, step, body),
        _ => unreachable!(),
    };
    let start = ctx.eval_scalar_uniform(start_v, "loop start", "divergent_loop_bounds")?;
    let stop = ctx.eval_scalar_uniform(stop_v, "loop stop", "divergent_loop_bounds")?;
    let step = ctx.eval_scalar_uniform(step_v, "loop step", "divergent_loop_bounds")?;
    execute_counted_loop(
        ctx,
        var.id.0,
        start,
        stop,
        step,
        body,
        "invalid_loop_step",
        "loop step must be positive",
    )
}

fn execute_for_each_task<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (scheduler, var, body) = match stmt {
        Stmt::ForEachTask {
            scheduler,
            var,
            body,
        } => (scheduler, var, body),
        _ => unreachable!(),
    };
    if scheduler.policy != SchedulerPolicy::GridStride {
        return Err(InterpreterError::new(
            "unsupported_scheduler_policy",
            "only grid_stride ForEachTask is executable",
        ));
    }
    let total_tasks = scheduler.space.task_count().ok_or_else(|| {
        InterpreterError::new("invalid_scheduler", "task space size overflows usize")
    })?;
    let cluster_count = ctx.kernel.launch_cta_count() / ctx.cluster_cta_count();
    let start = i64::try_from(ctx.stream.cluster_id).map_err(|_| {
        InterpreterError::new("invalid_scheduler", "cluster id does not fit in i64")
    })?;
    let stop = i64::try_from(total_tasks).map_err(|_| {
        InterpreterError::new("invalid_scheduler", "task count does not fit in i64")
    })?;
    let step = i64::try_from(cluster_count).map_err(|_| {
        InterpreterError::new("invalid_scheduler", "cluster count does not fit in i64")
    })?;
    execute_counted_loop(
        ctx,
        var.id.0,
        start,
        stop,
        step,
        body,
        "invalid_scheduler",
        "cluster count must be positive",
    )
}

fn execute_counted_loop<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    var_id: u32,
    start: i64,
    stop: i64,
    step: i64,
    body: &'k [Stmt],
    invalid_step_reason: &'static str,
    invalid_step_message: &'static str,
) -> IResult<StepStatus> {
    if step <= 0 {
        return Err(InterpreterError::new(
            invalid_step_reason,
            invalid_step_message,
        ));
    }
    if start >= stop {
        return Ok(StepStatus::advance_continue());
    }
    if !body.is_empty() {
        push_loop(
            ctx.stream,
            var_id,
            start,
            stop,
            step,
            body,
            ctx.lanes.clone(),
        );
        Ok(StepStatus::advance_continue())
    } else {
        // Empty body: skip straight to the terminal loop-var value.
        let last = start + ((stop - start - 1) / step) * step;
        ctx.state
            .values
            .scalars
            .write_mask(&ctx.lanes, var_id, last);
        Ok(StepStatus::advance())
    }
}

fn execute_sched_next<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let (scheduler, var) = match stmt {
        Stmt::SchedNext { scheduler, var } => (scheduler, var),
        _ => unreachable!(),
    };
    if scheduler.policy == SchedulerPolicy::GridStride {
        return Err(InterpreterError::new(
            "invalid_scheduler_policy",
            "sched_next requires a concurrent scheduler policy",
        ));
    }
    let total_tasks = scheduler.space.task_count().ok_or_else(|| {
        InterpreterError::new("invalid_scheduler", "task space size overflows usize")
    })?;
    let cluster_count = ctx.kernel.launch_cta_count() / ctx.cluster_cta_count();
    if cluster_count == 0 {
        return Err(InterpreterError::new(
            "invalid_scheduler",
            "cluster count must be positive",
        ));
    }
    let cursor = ctx
        .state
        .scheduler_next_cursors
        .entry((scheduler.id, ctx.stream.cluster_id))
        .or_insert(0);
    let offset = cursor
        .checked_mul(cluster_count)
        .ok_or_else(|| InterpreterError::new("invalid_scheduler", "task index overflows usize"))?;
    let task =
        ctx.stream.cluster_id.checked_add(offset).ok_or_else(|| {
            InterpreterError::new("invalid_scheduler", "task index overflows usize")
        })?;
    if task < total_tasks {
        *cursor += 1;
    }
    let value = if task < total_tasks {
        i64::try_from(task).map_err(|_| {
            InterpreterError::new("invalid_scheduler", "task id does not fit in i64")
        })?
    } else {
        -1
    };
    ctx.state
        .values
        .scalars
        .write_mask(&ctx.lanes, var.id.0, value);
    if ctx.trace_mode() {
        ctx.emit(TraceEventKind::SchedulerNext {
            scheduler_id: scheduler.id,
            cta_id: ctx.stream.cta_id,
            task_id: value,
            scope: ctx.access_scope(),
        })?;
    }
    Ok(StepStatus::advance())
}

fn execute_dynamic_loop<'a, 'k>(
    ctx: &mut WarpContext<'a, 'k>,
    stmt: &'k Stmt,
) -> IResult<StepStatus> {
    let body = match stmt {
        Stmt::Loop { body } => body,
        _ => unreachable!(),
    };
    if !body.is_empty() {
        push_dynamic_loop(ctx.stream, body.as_slice(), ctx.lanes.clone());
    }
    Ok(StepStatus::advance_continue())
}

fn execute_break_if<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let cond = match stmt {
        Stmt::BreakIf { cond } => cond,
        _ => unreachable!(),
    };
    let value = ctx.eval_scalar_uniform(cond, "break condition", "divergent_break")?;
    if value != 0 && !break_dynamic_loop(ctx.stream) {
        return Err(InterpreterError::new(
            "invalid_break",
            "break_if must be inside loop",
        ));
    }
    Ok(StepStatus::advance_continue())
}

fn execute_if<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (cond, then_body) = match stmt {
        Stmt::If { cond, then_body } => (cond, then_body),
        _ => unreachable!(),
    };
    let conditions = ctx.eval_scalar_vec(cond)?;
    let true_threads: Vec<_> = ctx
        .lanes
        .iter()
        .zip(conditions.iter())
        .filter(|(_, &c)| c != 0)
        .map(|(t, _)| t.clone())
        .collect();
    let true_mask = if true_threads.len() == ctx.lanes.len() {
        ctx.lanes.clone()
    } else {
        canonical_thread_mask(true_threads)
    };
    if !true_mask.is_empty() && !then_body.is_empty() {
        push_frame(ctx.stream, then_body.as_slice(), true_mask);
    }
    Ok(StepStatus::advance_continue())
}
