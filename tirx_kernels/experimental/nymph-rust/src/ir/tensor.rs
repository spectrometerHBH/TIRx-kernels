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

use super::dtype::{DType, MemorySpace, Swizzle};
use super::scalar::ScalarValue;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// `Layout` — only SMEM swizzle remains: TMEM is no longer a tensor and has no
/// layout abstraction (see `TmemOperand`).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Layout {
    Swizzle(SmemSwizzleLayout),
}

/// `SmemSwizzleLayout`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SmemSwizzleLayout {
    pub swizzle: Swizzle,
}

/// A TMEM reference: absolute physical address + cell interpretation.
///
/// TMEM is a 128-lane x 512-column grid of 32-bit cells, allocated by column
/// band (`TmemAlloc`); it is NOT a tensor and has no layout. Every TMEM
/// instruction takes this explicit absolute physical `(lane, col)` base
/// address plus the `dtype` that says how the addressed cells are (un)packed
/// (f32/i32/u32 one value per cell; f16/bf16 two per cell, low half first;
/// f8e4m3 one raw scale byte per cell). `row` is the lane in [0, 128), `col`
/// the column in [0, 512).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TmemOperand {
    pub row: ScalarValue,
    pub col: ScalarValue,
    pub dtype: DType,
}

/// A `Tcgen05Mma` A/B operand: an SMEM tile (`TensorSlice`), or TMEM cells
/// (`TmemOperand`) — the value model's accumulator-readback abstraction (the
/// GDN state read straight out of TMEM).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MmaOperand {
    Slice(TensorSlice),
    Tmem(TmemOperand),
}

/// Register-fragment physical layout for an epilogue REG tensor. `None` (the
/// default) is the plain warpgroup thread-axis tile (`T.wg_reg_tile`): each of the
/// 128 lanes owns one contiguous row. `Stmatrix` selects the `tcgen05.{ld,st}`-atom
/// layout (`T.alloc_tcgen05_ldst_frag` / `T.alloc_cast_frag`) — the per-(lane,
/// register) decomposition that the `Tx.copy(dispatch="ldstmatrix")` reg->smem store
/// requires to lower to STSM/stmatrix instead of plain STS (canon's nvfp4 epilogue).
///
/// This is a CODEGEN-ONLY concern: the value model and protocol checker see a plain
/// REG tensor either way (the physical register decomposition is below the value
/// model). So the marker rides on the IR tensor purely to drive the emitted alloc +
/// the reg->smem dispatch; it never changes interpreter/validate behaviour.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RegFrag {
    /// `tcgen05.{ld,st}`-atom fragment. `instr_shape` is the PTX atom shape
    /// (canon's `"16x256b"`). `cast_of` links a cast (output) frag to its source
    /// (read) frag by id: `None` = the read frag (`alloc_tcgen05_ldst_frag`),
    /// `Some(src_id)` = a `alloc_cast_frag(<src>, dtype)` of that read frag.
    Stmatrix {
        instr_shape: String,
        cast_of: Option<u32>,
    },
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
    /// Physical register-fragment layout for REG-space epilogue tensors (see
    /// `RegFrag`). `None` for every non-REG tensor and for the default thread-axis
    /// reg tile.
    pub reg_frag: Option<RegFrag>,
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
