//! TensorDef / MBarDef — discovery metadata only (no-op) — port of `metadata.py`.

use super::super::diagnostics::IResult;
use super::super::outcomes::StepStatus;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::warp_context::WarpContext;
use crate::ir::Stmt;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::TensorDef, execute_metadata);
    reg.register(StmtKind::MBarDef, execute_metadata);
    // setmaxnreg: register-pressure directive for codegen; the simulator does
    // not model register file occupancy, so it is discovery metadata here.
    reg.register(StmtKind::SetMaxNReg, execute_metadata);
}

fn execute_metadata<'a, 'k>(
    _ctx: &mut WarpContext<'a, 'k>,
    _stmt: &'k Stmt,
) -> IResult<StepStatus> {
    Ok(StepStatus::advance_continue())
}
