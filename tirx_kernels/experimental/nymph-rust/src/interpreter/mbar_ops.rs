//! mbarrier target resolution + pure phase-cell algebra — port of `mbar_ops.py`.

use super::diagnostics::{IResult, InterpreterError};
use super::threads::ThreadId;
use super::values::mbars::{MbarCell, MbarCellKey, MbarIdentity};
use super::warp_context::WarpContext;
use crate::ir::{MBarRef, ScalarValue};

#[derive(Clone, Copy, Debug)]
pub struct MbarTarget {
    pub identity: MbarIdentity,
    pub stage: usize,
}
impl MbarTarget {
    pub fn key(&self) -> MbarCellKey {
        (self.identity, self.stage)
    }
}

pub fn retarget_mbar(target: MbarTarget, ctaid_in_cluster: usize) -> MbarTarget {
    MbarTarget {
        identity: MbarIdentity {
            ctaid_in_cluster,
            ..target.identity
        },
        stage: target.stage,
    }
}

pub fn resolve_mbar_target(
    ctx: &WarpContext,
    mbar: &MBarRef,
    stage_value: Option<&ScalarValue>,
    thread: &ThreadId,
) -> IResult<MbarTarget> {
    let ctaid_in_cluster = match &mbar.remote_coord {
        None => thread.ctaid_in_cluster,
        Some(rc) => ctx.eval_scalar_at(rc, thread)? as usize,
    };
    check_mbar_remote_coord(ctx, ctaid_in_cluster)?;
    let stage = match stage_value {
        None => 0,
        Some(s) => ctx.eval_scalar_at(s, thread)? as usize,
    };
    check_mbar_stage(mbar.mbar.stages as usize, stage)?;
    Ok(MbarTarget {
        identity: MbarIdentity {
            mbar_id: mbar.mbar.id,
            cluster_id: thread.cluster_id,
            ctaid_in_cluster,
        },
        stage,
    })
}

pub fn uniform_mbar_target(
    ctx: &WarpContext,
    mbar: &MBarRef,
    stage_value: Option<&ScalarValue>,
) -> IResult<MbarTarget> {
    let target = resolve_mbar_target(ctx, mbar, stage_value, &ctx.lanes[0])?;
    if let Some(s) = stage_value {
        ctx.eval_scalar_uniform(s, "mbarrier stage", "divergent_mbarrier_operands")?;
    }
    if let Some(rc) = &mbar.remote_coord {
        ctx.eval_scalar_uniform(rc, "mbarrier remote coord", "divergent_mbarrier_operands")?;
    }
    Ok(target)
}

pub fn initialized_mbar_cell(ctx: &WarpContext, key: MbarCellKey) -> IResult<MbarCell> {
    ctx.state
        .values
        .mbars
        .cells
        .get(&key)
        .copied()
        .ok_or_else(|| {
            InterpreterError::new("uninitialized_mbarrier", "mbarrier cell is not initialized")
        })
}

// ---- pure phase-cell algebra ----

pub fn complete_mbarrier_phase_if_ready(cell: MbarCell) -> MbarCell {
    if cell.pending_arrivals != 0 || cell.pending_tx_bytes != 0 {
        return cell;
    }
    MbarCell {
        expected_arrivals: cell.expected_arrivals,
        pending_arrivals: cell.expected_arrivals,
        pending_tx_bytes: 0,
        parity: cell.parity ^ 1,
        stage: cell.stage,
    }
}

pub fn arrive_mbarrier_cell(cell: MbarCell, count: i64) -> IResult<MbarCell> {
    if count < 1 {
        return Err(InterpreterError::new(
            "mbarrier_arrive_count",
            "mbarrier arrive count must be positive",
        ));
    }
    if count > cell.pending_arrivals {
        return Err(InterpreterError::new(
            "mbarrier_arrive_overflow",
            "mbarrier arrive exceeds pending arrivals",
        ));
    }
    Ok(complete_mbarrier_phase_if_ready(MbarCell {
        pending_arrivals: cell.pending_arrivals - count,
        ..cell
    }))
}

pub fn expect_tx_cell(cell: MbarCell, byte_count: i64) -> MbarCell {
    MbarCell {
        pending_tx_bytes: cell.pending_tx_bytes - byte_count,
        ..cell
    }
}

pub fn complete_mbarrier_tx(cell: MbarCell, byte_count: i64) -> IResult<MbarCell> {
    if byte_count < 0 {
        return Err(InterpreterError::new(
            "mbarrier_complete_tx_count",
            "mbarrier complete-tx byte count must be non-negative",
        ));
    }
    // The PTX transaction balance is signed: expect_tx subtracts the expected
    // bytes and complete-tx adds the actual bytes. Either instruction may run
    // first; the phase completes only at exactly zero (and with no pending
    // arrivals).
    Ok(complete_mbarrier_phase_if_ready(MbarCell {
        pending_tx_bytes: cell.pending_tx_bytes + byte_count,
        ..cell
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell() -> MbarCell {
        MbarCell {
            expected_arrivals: 1,
            pending_arrivals: 1,
            pending_tx_bytes: 0,
            parity: 0,
            stage: 0,
        }
    }

    #[test]
    fn complete_tx_may_precede_expect_tx() {
        let completed_first = complete_mbarrier_tx(cell(), 16).unwrap();
        assert_eq!(completed_first.pending_tx_bytes, 16);
        assert_eq!(completed_first.parity, 0);

        let expected = expect_tx_cell(completed_first, 16);
        assert_eq!(expected.pending_tx_bytes, 0);
        assert_eq!(expected.parity, 0);

        let arrived = arrive_mbarrier_cell(expected, 1).unwrap();
        assert_eq!(arrived.pending_tx_bytes, 0);
        assert_eq!(arrived.parity, 1);
    }

    #[test]
    fn expect_tx_may_precede_complete_tx() {
        let expected_first = expect_tx_cell(cell(), 16);
        assert_eq!(expected_first.pending_tx_bytes, -16);

        let arrived = arrive_mbarrier_cell(expected_first, 1).unwrap();
        assert_eq!(arrived.pending_arrivals, 0);
        assert_eq!(arrived.parity, 0);

        let completed = complete_mbarrier_tx(arrived, 16).unwrap();
        assert_eq!(completed.pending_tx_bytes, 0);
        assert_eq!(completed.parity, 1);
    }
}

// ---- cluster/CTA helpers ----

pub fn peer_ctaid_in_cluster(
    ctx: &WarpContext,
    ctaid_in_cluster: usize,
    code: &str,
    msg: &str,
) -> IResult<usize> {
    let peer = ctaid_in_cluster ^ 1;
    if peer >= ctx.cluster_cta_count() {
        return Err(InterpreterError::new(code, msg));
    }
    Ok(peer)
}

pub fn multicast_target_ctas(
    ctx: &WarpContext,
    mask: u16,
    code_prefix: &str,
    label: &str,
) -> IResult<Vec<usize>> {
    if mask == 0 {
        return Err(InterpreterError::new(
            format!("empty_{code_prefix}_multicast_mask"),
            format!("{label} multicast mask must select at least one CTA"),
        ));
    }
    let addressable = ctx.cluster_cta_count().min(16);
    let valid_mask: u32 = (1u32 << addressable) - 1;
    if (mask as u32) & !valid_mask != 0 {
        return Err(InterpreterError::new(
            format!("{code_prefix}_multicast_cta_mask_oob"),
            format!("{label} multicast mask selects a CTA outside the current cluster"),
        ));
    }
    Ok((0..addressable).filter(|c| mask & (1 << c) != 0).collect())
}

pub fn check_mbar_remote_coord(ctx: &WarpContext, ctaid_in_cluster: usize) -> IResult<()> {
    if ctaid_in_cluster >= ctx.cluster_cta_count() {
        return Err(InterpreterError::new(
            "mbarrier_remote_cta_oob",
            "mbarrier remote CTA coordinate is out of cluster range",
        ));
    }
    Ok(())
}

pub fn check_mbar_stage(stages: usize, stage: usize) -> IResult<()> {
    if stages < 1 {
        return Err(InterpreterError::new(
            "invalid_mbarrier_stage",
            "mbarrier stages must be positive",
        ));
    }
    if stage >= stages {
        return Err(InterpreterError::new(
            "invalid_mbarrier_stage",
            "mbarrier stage is out of range",
        ));
    }
    Ok(())
}
