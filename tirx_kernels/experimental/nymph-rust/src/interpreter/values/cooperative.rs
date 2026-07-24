//! Cooperative barrier rendezvous state.
//!
//! One `SyncRecord` per (statement, rendezvous domain): per-warp streams
//! arrive once each (stream-granular bookkeeping — a stream's lanes arrive
//! atomically), the completing arrival flips the record to complete, and each
//! participant passes on its next poll. Counts, not thread sets: a blocked
//! stream's re-poll is a hash lookup and an integer compare.

use std::collections::{HashMap, HashSet};

/// Rendezvous identity: the sync statement plus its runtime domain
/// (cluster id for cluster_sync; CTA id for cta_sync; (CTA, warpgroup) for
/// wg_sync; (CTA, warp) for warp_sync — packed into `domain`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SyncKey {
    pub stmt_id: usize,
    pub domain: (usize, usize),
}

#[derive(Clone, Debug, Default)]
pub struct SyncRecord {
    pub cycle: u64,
    pub expected_count: usize,
    /// Global CTA ids in scope (cluster peer-liveness checks).
    pub expected_ctas: Vec<usize>,
    /// stream_id -> threads that arrived from that stream this cycle.
    pub arrived_streams: HashMap<usize, usize>,
    pub arrived_count: usize,
    /// Global CTA id -> threads arrived from it (peer-liveness).
    pub arrived_per_cta: HashMap<usize, usize>,
    pub passed_streams: HashSet<usize>,
    pub passed_count: usize,
}

impl SyncRecord {
    pub fn complete(&self) -> bool {
        self.expected_count > 0 && self.arrived_count == self.expected_count
    }

    /// Reset for the next use of the same barrier.
    pub fn advance_cycle(&mut self) {
        self.cycle += 1;
        self.arrived_streams.clear();
        self.arrived_count = 0;
        self.arrived_per_cta.clear();
        self.passed_streams.clear();
        self.passed_count = 0;
    }
}

#[derive(Clone, Debug, Default)]
pub struct CooperativeValues {
    pub syncs: HashMap<SyncKey, SyncRecord>,
}
