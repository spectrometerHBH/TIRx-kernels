//! Tensors and layouts.
//!
//! IDENTITY model (matches Python's `eq=False` object identity, via Rust's shared
//! pointer): a `Tensor` is heap-allocated once and SHARED through `Arc<Tensor>`
//! (the Rust equivalent of a C++ `shared_ptr` / a Python object reference). A
//! `TensorSlice` holds an `Arc<Tensor>`, so `slice.tensor.space` reads the tensor's
//! data directly — exactly like Python — which lets construction-time validation
//! cross-reference the tensor. Identity is the `id` field (assigned by the
//! builder): equality/hash compare ids only, so two `Arc` clones of the same tensor
//! are "the same tensor", and two distinct tensors with identical fields are not.

use super::dtype::{DType, MemorySpace, Swizzle, TmemLayoutKind};
use super::scalar::ScalarValue;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// `Layout` (base + 2 subclasses) -> a Rust enum.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Layout {
    Swizzle(SmemSwizzleLayout),
    Tmem(TmemLayout),
    /// NVFP4 scale-factor layout: the GMEM/SMEM buffer is laid out so the TMA loads
    /// the e4m3 scale bytes straight into the `tcgen05.cp`-ready order (canon's
    /// `sf_smem_layout(rows, sf_k, sf_per_mma)`), replacing an explicit permute warp.
    SfSmem(SfSmemLayout),
}

/// `SmemSwizzleLayout`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SmemSwizzleLayout {
    pub swizzle: Swizzle,
}

/// `SfSmemLayout` — rows/sf_k come from the tensor shape at codegen time; only
/// `sf_per_mma` (scale blocks per MMA issue, 4 for nvfp4) is carried here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SfSmemLayout {
    pub sf_per_mma: u8,
}

/// `TmemLayout`. `lane_align` is the MMA accumulator d-tmem lane field (0 or 16),
/// NOT part of the view mapping (see the Python docstring).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TmemLayout {
    pub kind: TmemLayoutKind,
    pub col_start: usize,
    pub lane_align: u8, // 0 or 16
}

/// `Tensor` — the data, plus a stable `id` for identity. Held by `Arc<Tensor>`
/// wherever it's referenced (no copies of the shape/layout).
#[derive(Debug)]
pub struct Tensor {
    pub id: u32,
    pub space: MemorySpace,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub layout: Option<Layout>,
    pub byte_offset: Option<usize>,
}

// Identity = id only (so `Arc<Tensor>` comparisons reduce to id comparisons, and a
// tensor is a stable HashMap key via its id).
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Tensor {}
impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// `TensorSlice` — a sub-region: per-dim offsets + shape (each a scalar value,
/// possibly symbolic). Holds its tensor by `Arc`, so its data is reachable here.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TensorSlice {
    pub tensor: Arc<Tensor>,
    pub offsets: Vec<ScalarValue>,
    pub shape: Vec<ScalarValue>,
}
