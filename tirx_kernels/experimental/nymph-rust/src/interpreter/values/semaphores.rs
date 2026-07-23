//! GMEM semaphore runtime values — the value-side counter map for
//! `gmem_atomic_add` / `gmem_wait_eq`. A semaphore is an i32 GMEM cell keyed by
//! `(tensor_id, flat_coord)` and SHARED across all CTAs that target it (NOT
//! per-CTA), so one CTA's atomic-add is observed by another CTA's spin-wait.
//!
//! Only the counter lives here. The release/acquire happens-before edges are a
//! pure trace-layer concept — `OrderingAnalysis` reconstructs them from the
//! `SemRelease`/`SemAcquire` trace events; the value side never needs a clock.

use super::super::protocol::SemKey;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct SemaphoreValues {
    /// Counter per semaphore cell. Absent == 0 (auto-zero-init: 0 is the first
    /// CTA's turn, so `wait_eq(0)` passes immediately on an untouched cell).
    cells: HashMap<SemKey, i64>,
}

impl SemaphoreValues {
    /// Current counter at `key` (0 if never written — zero-init).
    pub fn get(&self, key: SemKey) -> i64 {
        self.cells.get(&key).copied().unwrap_or(0)
    }

    /// Serialized RMW: `cell += value`, returns the NEW (post-increment) value.
    /// The interpreter applies one stream's atomic at a time (cooperative
    /// scheduler), so this is indivisible by construction.
    pub fn atomic_add(&mut self, key: SemKey, value: i64) -> i64 {
        let entry = self.cells.entry(key).or_insert(0);
        *entry += value;
        *entry
    }
}
