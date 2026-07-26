//! The `Kernel` — the whole executable IR. Each tensor/mbar's data is owned by
//! its `Arc<Tensor>` / `Arc<MBar>` and shared wherever it is referenced, so the
//! kernel itself just lists its argument tensors.

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
}

impl Kernel {
    /// Total number of CTAs in `launch_shape` (mirrors `launch_cta_count`).
    pub fn launch_cta_count(&self) -> usize {
        self.launch_shape.iter().product()
    }
}
