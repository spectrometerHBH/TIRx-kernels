//! mbarriers. Same shared-pointer identity model as `Tensor`: an `MBar` is held by
//! `Arc<MBar>` (shared, identity by `id`). A statement that names an mbarrier holds
//! an `MBarRef` (which can also target a peer CTA via `remote_coord`), and through
//! it `mbar_ref.mbar.kind` reads the mbarrier's data directly (for kind checks).

use super::dtype::MBarKind;
use super::scalar::ScalarValue;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// `MBar` — the mbarrier object's data, plus a stable `id` for identity.
#[derive(Debug)]
pub struct MBar {
    pub id: u32,
    pub kind: MBarKind,
    pub stages: u32,
    pub arrive_count: Option<u32>,
    /// Cluster TMA-completion routing, set by the builder: when true, BOTH CTAs'
    /// TMA loads signal the LEADER CTA's (CTA-0) copy of this barrier (the
    /// canonical cta_group=2 pattern — the legal substitute for a peer
    /// `try_wait`, and the prerequisite for multicast loads, whose per-
    /// destination tx counts must accumulate on the single leader barrier).
    /// Codegen reads this flag; it never guesses it from the usage structure.
    /// Validate requires a peer reference and TMA-load/expect_tx-only use.
    pub leader_routed: bool,
}

impl PartialEq for MBar {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for MBar {}
impl Hash for MBar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// `MBarRef` — a reference to an mbarrier (via `Arc`), optionally on a peer CTA.
/// Python lets a statement take `MBar | MBarRef`; the builder coerces a bare
/// `MBar` to an `MBarRef { remote_coord: None }`, so the IR only stores `MBarRef`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct MBarRef {
    pub mbar: Arc<MBar>,
    pub remote_coord: Option<ScalarValue>,
}
