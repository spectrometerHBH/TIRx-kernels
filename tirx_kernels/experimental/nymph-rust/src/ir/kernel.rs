//! The `Kernel` — the whole executable IR. With the `Arc<Tensor>` / `Arc<MBar>`
//! model there is no separate table: each tensor/mbar's data is owned by its `Arc`
//! and shared wherever it's referenced. The kernel just lists its argument tensors.

use super::stmt::Stmt;
use super::tensor::Tensor;
use std::sync::Arc;

/// `Kernel` — executable Nymph kernel IR.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Kernel {
    pub name: String,
    /// Kernel argument tensors (shared with the body via their `Arc`s).
    pub args: Vec<Arc<Tensor>>,
    pub body: Vec<Stmt>,
    pub num_warps: u32,
    /// CTA-local dynamic shared-memory byte pool size.
    pub smem_size_bytes: usize,
    /// Multidimensional CTA grid (dim 0 fastest).
    pub launch_shape: Vec<usize>,
    /// Same-rank tile over the grid.
    pub cluster_shape: Vec<usize>,
    /// When true, the codegen emits the SMEM data buffers (A/B/SFA/SFB/D) as a
    /// single canon-style dynamic `T.SMEMPool()` allocation that the per-tensor
    /// buffers alias into (via their `byte_offset`), instead of N independent
    /// static `T.alloc_buffer(scope="shared")` allocations. This cuts the static
    /// SMEM footprint (the buffers become `shared.dyn`) and the register pressure
    /// toward canon's. Off by default; gated to the small-shape variant so the big
    /// shapes keep the byte-identical static path.
    pub smem_pool: bool,
}

impl Kernel {
    /// Total number of CTAs in `launch_shape` (mirrors `launch_cta_count`).
    pub fn launch_cta_count(&self) -> usize {
        self.launch_shape.iter().product()
    }
}
