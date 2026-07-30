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

/// A tensor's explicit physical layout. TMEM is no longer a regular tensor and
/// has no tensor layout abstraction (see `TmemTensor`).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Layout {
    Swizzle(SmemSwizzleLayout),
}

/// `SmemSwizzleLayout`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SmemSwizzleLayout {
    pub swizzle: Swizzle,
}

/// A logical view into TMEM. `start_col` is an absolute physical TMEM column;
/// the view owns no storage (`TmemAlloc` remains the physical allocation
/// statement).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TmemTensor {
    pub start_col: u32,
}

impl TmemTensor {
    pub fn at(self, row: ScalarValue, col: ScalarValue) -> TmemAddr {
        TmemAddr {
            tensor: self,
            row,
            col,
        }
    }
}

/// A TMEM address. Its effective physical address is
/// `(row, tensor.start_col + col)`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TmemAddr {
    pub tensor: TmemTensor,
    pub row: ScalarValue,
    pub col: ScalarValue,
}

/// An explicit two-dimensional SMEM tile used by physical tcgen05 operations.
/// Leading dimensions are fixed by `prefix_indices`; the final two dimensions
/// start at (`row_offset`, `col_offset`) and have the literal `rows x cols`
/// extent.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SmemTile {
    pub tensor: Arc<Tensor>,
    pub prefix_indices: Vec<ScalarValue>,
    pub row_offset: ScalarValue,
    pub col_offset: ScalarValue,
    pub rows: u32,
    pub cols: u32,
}

/// Element format selected by a tcgen05 MMA instruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MmaElemFormat {
    F16,
    BF16,
    F8E4M3,
    F4E2M1,
}

impl MmaElemFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            MmaElemFormat::F16 => "f16",
            MmaElemFormat::BF16 => "bf16",
            MmaElemFormat::F8E4M3 => "f8_e4m3",
            MmaElemFormat::F4E2M1 => "f4_e2m1",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "f16" => Some(MmaElemFormat::F16),
            "bf16" => Some(MmaElemFormat::BF16),
            "f8_e4m3" => Some(MmaElemFormat::F8E4M3),
            "f4_e2m1" => Some(MmaElemFormat::F4E2M1),
            _ => None,
        }
    }
}

/// Scale-factor format selected by a block-scaled tcgen05 MMA instruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScaleFormat {
    E8M0FNU,
    E4M3FN,
}

impl ScaleFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            ScaleFormat::E8M0FNU => "e8m0_fnu",
            ScaleFormat::E4M3FN => "e4m3_fn",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "e8m0_fnu" => Some(ScaleFormat::E8M0FNU),
            "e4m3_fn" => Some(ScaleFormat::E4M3FN),
            _ => None,
        }
    }
}

/// Physical form of a TMEM-resident MMA A operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TmemAForm {
    Flat,
    BankBatched,
}

impl TmemAForm {
    pub fn as_str(self) -> &'static str {
        match self {
            TmemAForm::Flat => "flat",
            TmemAForm::BankBatched => "bank_batched",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "flat" => Some(TmemAForm::Flat),
            "bank_batched" => Some(TmemAForm::BankBatched),
            _ => None,
        }
    }
}

/// The A operand of a tcgen05 MMA. B is always an explicit `SmemTile`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MmaAOperand {
    Smem(SmemTile),
    Tmem { addr: TmemAddr, form: TmemAForm },
}

/// Extra physical operands and format parameters for block-scaled MMA.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BlockScaleSpec {
    pub sfa: TmemAddr,
    pub sfb: TmemAddr,
    pub sfa_k_offset: u32,
    pub sfb_k_offset: u32,
    pub scale_format: ScaleFormat,
    pub sf_per_mma: u32,
    pub sf_reuse: u32,
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
