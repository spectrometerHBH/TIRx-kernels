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
    /// Codegen-only allocation form: when true, SMEM data buffers + mbar cells
    /// are emitted as canon's dynamic `T.SMEMPool()` (one `shared.dyn` window at
    /// the IR's own `byte_offset`s) instead of static `T.alloc_buffer`s. The
    /// value model and protocol checker are allocation-form-agnostic (they read
    /// `byte_offset` either way), so this never reaches the interpreter.
    pub smem_pool: bool,
}

impl Kernel {
    /// Total number of CTAs in `launch_shape` (mirrors `launch_cta_count`).
    pub fn launch_cta_count(&self) -> usize {
        self.launch_shape.iter().product()
    }
}
