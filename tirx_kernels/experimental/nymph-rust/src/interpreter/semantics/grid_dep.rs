//! GridDepControl — a cross-grid launch-scheduling hint (no value, memory, or
//! trace effect inside this kernel) — the interpreter steps over it.

use super::super::diagnostics::IResult;
use super::super::outcomes::StepStatus;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::warp_context::WarpContext;
use crate::ir::Stmt;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::GridDepControl, execute_grid_dep_control);
}

fn execute_grid_dep_control<'a, 'k>(
    _ctx: &mut WarpContext<'a, 'k>,
    _stmt: &'k Stmt,
) -> IResult<StepStatus> {
    Ok(StepStatus::advance())
}
