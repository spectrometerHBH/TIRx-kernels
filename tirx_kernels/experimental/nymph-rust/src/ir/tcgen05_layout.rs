//! Shared physical-layout resolution for tcgen05 MMA and copy instructions.
//!
//! The IR carries physical operands, but their complete TMEM footprints are a
//! function of the instruction shape, CTA group, operand form, and scale-factor
//! metadata.  This module is the single source of truth for that derivation.
//! Validation, code generation, and the interpreter must consume these results
//! instead of independently guessing a layout from operand order or tensor ids.

use super::{
    BlockScaleSpec, DType, Layout, MemorySpace, MmaAOperand, MmaElemFormat, ScalarValue,
    ScaleFormat, SmemTile, Swizzle, Tcgen05CpMulticast, Tcgen05CpShape, TmemAForm, TmemAddr,
};

/// A tcgen05 layout-resolution failure.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Tcgen05LayoutError {
    pub message: String,
}

impl std::fmt::Display for Tcgen05LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Tcgen05LayoutError {}

type LayoutResult<T> = Result<T, Tcgen05LayoutError>;

fn fail<T>(message: impl Into<String>) -> LayoutResult<T> {
    Err(Tcgen05LayoutError {
        message: message.into(),
    })
}

fn literal_i64(value: &ScalarValue) -> Option<i64> {
    match value {
        ScalarValue::Int(value) => Some(*value),
        _ => None,
    }
}

fn checked_mul(lhs: u32, rhs: u32, label: &str) -> LayoutResult<u32> {
    lhs.checked_mul(rhs).ok_or_else(|| Tcgen05LayoutError {
        message: format!("{label} overflows u32"),
    })
}

fn div_ceil(value: u32, divisor: u32, label: &str) -> LayoutResult<u32> {
    if divisor == 0 {
        return fail(format!("{label} divisor must be positive"));
    }
    value
        .checked_add(divisor - 1)
        .map(|sum| sum / divisor)
        .ok_or_else(|| Tcgen05LayoutError {
            message: format!("{label} overflows u32"),
        })
}

fn dtype_bits(dtype: DType) -> u32 {
    match dtype {
        DType::Bool | DType::I8 | DType::U8 | DType::F8E4M3 => 8,
        DType::I16 | DType::U16 | DType::F16 | DType::Bf16 => 16,
        DType::I32 | DType::U32 | DType::F32 => 32,
        DType::I64 | DType::U64 => 64,
    }
}

fn format_bits(format: MmaElemFormat) -> u32 {
    match format {
        MmaElemFormat::F16 | MmaElemFormat::BF16 => 16,
        MmaElemFormat::F8E4M3 => 8,
        MmaElemFormat::F4E2M1 => 4,
    }
}

fn format_storage_dtype(format: MmaElemFormat) -> DType {
    match format {
        MmaElemFormat::F16 => DType::F16,
        MmaElemFormat::BF16 => DType::Bf16,
        MmaElemFormat::F8E4M3 => DType::F8E4M3,
        // Nymph represents two packed e2m1 values with one U8 storage element.
        MmaElemFormat::F4E2M1 => DType::U8,
    }
}

fn mma_k_for(format: MmaElemFormat) -> u32 {
    match format {
        MmaElemFormat::F16 | MmaElemFormat::BF16 => 16,
        MmaElemFormat::F8E4M3 => 32,
        MmaElemFormat::F4E2M1 => 64,
    }
}

/// A logical row x column matrix shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LogicalMatrixShape {
    pub rows: u32,
    pub cols: u32,
}

/// The logical GEMM tile owned by one CTA.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PerCtaMmaShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// PTX tcgen05 accumulator datapath classification.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TmemDatapath {
    /// cta_group::2, instruction M=256; 128 identity rows per CTA.
    A,
    /// cta_group::2, instruction M=128; 64 rows with two folded lane banks.
    B,
    /// cta_group::1, instruction M=128; 128 identity rows.
    D,
    /// cta_group::1, instruction M=64, weight stationary; two folded lane banks.
    E,
    /// cta_group::1, instruction M=64, non-ws; 16 rows in each warp slab.
    F,
}

/// A compact, exact classification of a TMEM view.
///
/// This intentionally does not recreate TVM's TileLayout AST.  Together with
/// `logical`, `physical_cols`, and `max_lane_offset`, each variant contains all
/// metadata needed to reconstruct the canonical view.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TmemLayoutKind {
    /// row -> TLane, column -> TCol.
    Identity,
    /// Logical columns split in half; the second half is at TLane + 64.
    Fold2,
    /// Layout F: logical row `16*q+r` maps to `32*q+r`.
    Scatter4,
    /// Flat logical A in Layout B, replicated at TLane + 64.
    Replica2,
    /// Explicit `(2, rows, cols)` bank fold with bank stride 64 lanes.
    BankBatched,
    /// Canonical `sf_tmem_layout` (32-row atom, four-way lane replica).
    ScaleFactor { sf_per_mma: u32, sf_reuse: u32 },
    /// A physical tcgen05.cp destination; `ResolvedCp` carries the atom size.
    Copy(CpLaneLayout),
}

/// A logical TMEM view and its enclosing physical cell footprint.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TmemFootprint {
    pub logical: LogicalMatrixShape,
    /// Enclosing 32-bit TMEM column span from the address's base column,
    /// including any holes in a sparse canonical layout.
    pub physical_cols: u32,
    /// Largest lane offset touched relative to the address's row.
    pub max_lane_offset: u32,
    pub layout: TmemLayoutKind,
}

/// One resolved SFA/SFB view.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResolvedScaleFootprint {
    pub footprint: TmemFootprint,
    /// Exact operand rows consumed by the MMA. The statement-local scale view
    /// is padded to a 32-row atom.
    pub operand_rows: u32,
    /// Unique scale groups along K: ceil((K / mma_k) / sf_reuse).
    pub groups: u32,
    /// Unique scale elements per row passed as `SF_K` to `sf_tmem_layout`.
    pub sf_k: u32,
    /// Logical last dimension, including the stride-zero reuse dimension.
    pub logical_last: u32,
    /// Canonical `sf_tmem_layout` pack factor along its unique-K iterator.
    pub pack_factor: u32,
    /// Exact 8-bit TCol span after shifting the complete canonical layout by
    /// this statement's K offset. This includes row-stride holes and any
    /// byte-to-cell carry introduced by that offset.
    pub tcol_element_span: u32,
    /// Physical 32-bit-cell delta selected by this instruction's K offset.
    pub cell_delta: u32,
    /// Byte lane (sf_id) within that 32-bit cell.
    pub subbyte: u8,
}

/// Complete physical resolution of one tcgen05 MMA statement.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ResolvedMma {
    /// Logical cluster-wide M. This equals the physical instruction M.
    pub m: u32,
    /// Logical N covered across the CTA group.
    pub n: u32,
    /// Logical contraction extent.
    pub k: u32,
    pub mma_k: u32,
    pub format: MmaElemFormat,
    pub per_cta: PerCtaMmaShape,
    pub datapath: TmemDatapath,
    pub d: TmemFootprint,
    pub a: LogicalMatrixShape,
    pub b: LogicalMatrixShape,
    pub a_tmem: Option<TmemFootprint>,
    pub sfa: Option<ResolvedScaleFootprint>,
    pub sfb: Option<ResolvedScaleFootprint>,
}

/// Exact row-to-lane mapping of one physical tcgen05.cp instruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpLaneLayout {
    /// row r -> lane r.
    Identity,
    /// Four rows -> lanes 0, 32, 64, 96.
    Quadrant4,
    /// row r -> lane r, replicated at +64.
    Warp2_02_13,
    /// rows 0..31 -> lanes 0..31, rows 32..63 -> lanes 64..95,
    /// replicated at +32.
    Warp2_01_23,
    /// rows 0..31 -> lanes 0..31, replicated at +32/+64/+96.
    Warp4,
}

/// Complete physical resolution of one tcgen05.cp statement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ResolvedCp {
    /// Source logical element shape.
    pub source: LogicalMatrixShape,
    /// Number of 32-bit cells in one complete source row.
    pub source_row_cells: u32,
    /// Number of rows consumed by one physical instruction atom.
    pub atom_rows: u32,
    /// Destination footprint; its logical shape remains source-element based,
    /// while `physical_cols` is the enclosing 32-bit-cell column span.
    pub target: TmemFootprint,
    pub lane_layout: CpLaneLayout,
    /// Exact statement-local SMEM view needed by the hardware descriptor.
    pub source_layout: CpSmemLayout,
    pub cta_group: u8,
}

/// Physical SMEM layout required by one tcgen05.cp source descriptor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpSmemLayout {
    /// Plain row-major source with an exact 16-byte physical row.
    Plain16B,
    /// Explicit 32-byte MMA-AB swizzle, required by the 256-bit CP shapes.
    Swizzle32B,
}

/// One physical TMEM cell relative to an instruction operand's `TmemAddr`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TmemCellCoord {
    pub lane: u32,
    pub col: u32,
}

/// One packed element relative to a TMEM operand's address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TmemElementCoord {
    pub lane: u32,
    pub col: u32,
    pub bit_offset: u8,
}

/// One 8-bit scale value relative to a scale operand's address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TmemByteCoord {
    pub lane: u32,
    pub col: u32,
    pub subbyte: u8,
}

impl ResolvedScaleFootprint {
    fn physical_byte_for_replica(
        &self,
        row: u32,
        logical_sf_index: u32,
        replica: u32,
    ) -> Option<TmemByteCoord> {
        if row >= self.footprint.logical.rows
            || logical_sf_index >= self.logical_last
            || replica >= 4
        {
            return None;
        }
        let TmemLayoutKind::ScaleFactor {
            sf_per_mma,
            sf_reuse,
        } = self.footprint.layout
        else {
            return None;
        };
        let row_block = row / 32;
        let unique_sf = logical_sf_index / sf_reuse;
        let group = unique_sf / sf_per_mma;
        let inner = unique_sf % sf_per_mma;
        let tcol_element = 4 * row_block
            + (group / self.pack_factor) * (self.footprint.logical.rows / 32) * 4
            + (group % self.pack_factor) * sf_per_mma
            + inner
            + self.cell_delta * 4
            + u32::from(self.subbyte);
        Some(TmemByteCoord {
            lane: row % 32 + replica * 32,
            col: tcol_element / 4,
            subbyte: (tcol_element % 4) as u8,
        })
    }

    /// Map one logical scale to the primary copy in canonical
    /// `sf_tmem_layout`. The statement's explicit K offset is included, so
    /// byte-to-cell carry is reflected in `col` and `subbyte`.
    pub fn physical_byte(&self, row: u32, logical_sf_index: u32) -> Option<TmemByteCoord> {
        self.physical_byte_for_replica(row, logical_sf_index, 0)
    }

    /// Map one logical scale to all four lane replicas in canonical
    /// `sf_tmem_layout`.
    pub fn physical_bytes(&self, row: u32, logical_sf_index: u32) -> Option<[TmemByteCoord; 4]> {
        Some([
            self.physical_byte_for_replica(row, logical_sf_index, 0)?,
            self.physical_byte_for_replica(row, logical_sf_index, 1)?,
            self.physical_byte_for_replica(row, logical_sf_index, 2)?,
            self.physical_byte_for_replica(row, logical_sf_index, 3)?,
        ])
    }
}

impl ResolvedMma {
    /// Map one logical accumulator element to its datapath cell relative to
    /// `dst`.
    pub fn d_cell(&self, row: u32, col: u32) -> Option<TmemCellCoord> {
        if row >= self.d.logical.rows || col >= self.d.logical.cols {
            return None;
        }
        let coord = match self.d.layout {
            TmemLayoutKind::Identity => TmemCellCoord { lane: row, col },
            TmemLayoutKind::Fold2 => {
                let half = self.d.logical.cols / 2;
                TmemCellCoord {
                    lane: row + (col / half) * 64,
                    col: col % half,
                }
            }
            TmemLayoutKind::Scatter4 => TmemCellCoord {
                lane: (row / 16) * 32 + row % 16,
                col,
            },
            _ => return None,
        };
        Some(coord)
    }

    /// Map one logical TMEM-A element to one explicitly selected lane bank.
    ///
    /// Identity layouts accept only bank 0. Replica2 and BankBatched layouts
    /// accept banks 0 and 1; Replica2 holds the same logical value in both,
    /// while BankBatched assigns a distinct A matrix to each bank.
    pub fn a_tmem_element(&self, bank: u32, row: u32, k: u32) -> Option<TmemElementCoord> {
        let footprint = self.a_tmem?;
        if row >= footprint.logical.rows || k >= footprint.logical.cols {
            return None;
        }
        let lane = match footprint.layout {
            TmemLayoutKind::Identity if bank == 0 => row,
            TmemLayoutKind::Replica2 | TmemLayoutKind::BankBatched if bank < 2 => row + bank * 64,
            _ => return None,
        };
        let bit = k * format_bits(self.format);
        Some(TmemElementCoord {
            lane,
            col: bit / 32,
            bit_offset: (bit % 32) as u8,
        })
    }
}

impl ResolvedCp {
    /// Map one source 32-bit cell to every destination cell written by the
    /// selected tcgen05.cp multicast form.
    ///
    /// `row` is a source element-row and `col` is a 32-bit cell index within
    /// that complete source row. Returned coordinates are relative to `dst`;
    /// callers add the evaluated `TmemAddr` row/column base.
    pub fn target_cells(&self, row: u32, col: u32) -> Option<Vec<TmemCellCoord>> {
        if row >= self.source.rows || col >= self.source_row_cells {
            return None;
        }
        let cells = match self.lane_layout {
            CpLaneLayout::Identity => {
                let atom = row / self.atom_rows;
                let lane = row % self.atom_rows;
                vec![TmemCellCoord {
                    lane,
                    col: atom * self.source_row_cells + col,
                }]
            }
            CpLaneLayout::Quadrant4 => {
                // Repeated 4-row atoms are source-order
                // (atom_index, quadrant): logical row 4*i+q lands at
                // lane i+32*q, matching canonical Layout F.
                let atom = row / 4;
                let quadrant = row % 4;
                vec![TmemCellCoord {
                    lane: quadrant * 32 + atom,
                    col,
                }]
            }
            CpLaneLayout::Warp2_02_13 => {
                let atom = row / self.atom_rows;
                let inner = row % self.atom_rows;
                let col = atom * self.source_row_cells + col;
                vec![
                    TmemCellCoord { lane: inner, col },
                    TmemCellCoord {
                        lane: inner + 64,
                        col,
                    },
                ]
            }
            CpLaneLayout::Warp2_01_23 => {
                let atom = row / self.atom_rows;
                let inner = row % self.atom_rows;
                let lane = (inner / 32) * 64 + inner % 32;
                let col = atom * self.source_row_cells + col;
                vec![
                    TmemCellCoord { lane, col },
                    TmemCellCoord {
                        lane: lane + 32,
                        col,
                    },
                ]
            }
            CpLaneLayout::Warp4 => {
                let atom = row / self.atom_rows;
                let inner = row % self.atom_rows;
                let col = atom * self.source_row_cells + col;
                (0..4)
                    .map(|replica| TmemCellCoord {
                        lane: inner + replica * 32,
                        col,
                    })
                    .collect()
            }
        };
        Some(cells)
    }
}

fn check_cta_group(cta_group: u8, label: &str) -> LayoutResult<()> {
    if matches!(cta_group, 1 | 2) {
        Ok(())
    } else {
        fail(format!("{label} must be 1 or 2"))
    }
}

fn validate_smem_tile_against_shape(
    tile: &SmemTile,
    tensor_shape: &[usize],
    label: &str,
) -> LayoutResult<()> {
    if tile.tensor.space != MemorySpace::Smem {
        return fail(format!("{label} tensor must be SMEM"));
    }
    let rank = tensor_shape.len();
    if rank < 2 {
        return fail(format!("{label} tensor rank must be at least 2"));
    }
    if tile.prefix_indices.len() != rank - 2 {
        return fail(format!(
            "{label} prefix_indices length must equal tensor rank - 2"
        ));
    }
    if tile.rows == 0 || tile.cols == 0 {
        return fail(format!("{label} rows and cols must be positive"));
    }
    for (index, &extent) in tile
        .prefix_indices
        .iter()
        .zip(tensor_shape[..rank - 2].iter())
    {
        if let Some(index) = literal_i64(index) {
            if index < 0 || index >= extent as i64 {
                return fail(format!("{label} prefix index is out of bounds"));
            }
        }
    }
    for (offset, extent, tensor_extent, axis) in [
        (&tile.row_offset, tile.rows, tensor_shape[rank - 2], "row"),
        (&tile.col_offset, tile.cols, tensor_shape[rank - 1], "col"),
    ] {
        if let Some(offset) = literal_i64(offset) {
            if offset < 0 {
                return fail(format!("{label} {axis}_offset must be non-negative"));
            }
            let end = (offset as u64) + u64::from(extent);
            if end > tensor_extent as u64 {
                return fail(format!("{label} {axis} range is out of bounds"));
            }
        }
    }
    Ok(())
}

fn validate_smem_tile(tile: &SmemTile, label: &str) -> LayoutResult<()> {
    validate_smem_tile_against_shape(tile, &tile.tensor.shape, label)
}

fn validate_cp_smem_tile(tile: &SmemTile, label: &str) -> LayoutResult<()> {
    validate_smem_tile(tile, label)
}

/// Effective source element width seen by tcgen05.cp.
pub(crate) fn tcgen05_cp_source_bits(tile: &SmemTile) -> u32 {
    dtype_bits(tile.tensor.dtype)
}

fn validate_format_tile(tile: &SmemTile, format: MmaElemFormat, label: &str) -> LayoutResult<()> {
    validate_smem_tile(tile, label)?;
    let want = format_storage_dtype(format);
    if tile.tensor.dtype != want {
        return fail(format!(
            "{label} storage dtype {:?} does not match MMA format {} (expected {:?})",
            tile.tensor.dtype,
            format.as_str(),
            want
        ));
    }
    Ok(())
}

fn logical_k_from_storage(storage_k: u32, format: MmaElemFormat) -> LayoutResult<u32> {
    if format == MmaElemFormat::F4E2M1 {
        checked_mul(storage_k, 2, "packed-f4 logical K")
    } else {
        Ok(storage_k)
    }
}

fn resolve_datapath(mma_m: u32, cta_group: u8, ws: bool) -> LayoutResult<TmemDatapath> {
    check_cta_group(cta_group, "tcgen05_mma cta_group")?;
    if ws && !(cta_group == 1 && mma_m == 64) {
        return fail("tcgen05_mma ws is only valid for cta_group=1, mma_m=64 (Layout E)");
    }
    match (cta_group, mma_m, ws) {
        (1, 128, false) => Ok(TmemDatapath::D),
        (1, 64, false) => Ok(TmemDatapath::F),
        (1, 64, true) => Ok(TmemDatapath::E),
        (2, 256, false) => Ok(TmemDatapath::A),
        (2, 128, false) => Ok(TmemDatapath::B),
        _ => fail(format!(
            "tcgen05_mma has no supported datapath for cta_group={cta_group}, \
             mma_m={mma_m}, ws={ws}"
        )),
    }
}

fn validate_mma_n(
    mma_n: u32,
    n: u32,
    mma_m: u32,
    cta_group: u8,
    block_scaled: bool,
) -> LayoutResult<()> {
    // PTX tcgen05 instruction-descriptor N rules are kind- and M-dependent.
    // Nymph currently exposes dense f16/bf16/f8 and block-scaled f8/f4:
    //   dense cg1: M64 -> N step 8, M128 -> N step 16
    //   dense cg2: M128/M256 -> N step 32
    //   block scale cg1/cg2: N step 8/16 respectively
    // Keep this physical check here, beside the datapath resolver, so neither
    // codegen nor TVM has to repair an invalid instruction shape.
    let granularity = if block_scaled {
        if cta_group == 1 {
            8
        } else {
            16
        }
    } else {
        match (cta_group, mma_m) {
            (1, 64) => 8,
            (1, 128) => 16,
            (2, 128 | 256) => 32,
            _ => {
                return fail(format!(
                    "tcgen05_mma has no dense N rule for cta_group={cta_group}, mma_m={mma_m}"
                ));
            }
        }
    };
    if mma_n < granularity || mma_n > 256 || mma_n % granularity != 0 {
        return fail(format!(
            "tcgen05_mma mma_n must be in [{granularity}, 256] and a multiple of \
             {granularity} for cta_group={cta_group}"
        ));
    }
    if n % mma_n != 0 {
        return fail(format!(
            "tcgen05_mma logical N={n} must be divisible by mma_n={mma_n}"
        ));
    }
    Ok(())
}

fn ensure_addr_fits(addr: &TmemAddr, footprint: &TmemFootprint, label: &str) -> LayoutResult<()> {
    if addr.tensor.start_col >= 512 {
        return fail(format!("{label} tensor start_col must be in [0, 512)"));
    }
    if footprint.physical_cols == 0 || footprint.physical_cols > 512 {
        return fail(format!(
            "{label} physical column span {} cannot fit TMEM columns [0, 512)",
            footprint.physical_cols
        ));
    }
    if footprint.max_lane_offset >= 128 {
        return fail(format!(
            "{label} lane footprint cannot fit TMEM lanes [0, 128)"
        ));
    }
    if let Some(row) = literal_i64(&addr.row) {
        if row < 0 || row + i64::from(footprint.max_lane_offset) >= 128 {
            return fail(format!(
                "{label} row plus its {:?} lane footprint exceeds TMEM lanes [0, 128)",
                footprint.layout
            ));
        }
    }
    if let Some(col) = literal_i64(&addr.col) {
        if col < 0 {
            return fail(format!("{label} col must be non-negative"));
        }
        let begin = u64::from(addr.tensor.start_col) + col as u64;
        let end = begin + u64::from(footprint.physical_cols);
        if begin >= 512 || end > 512 {
            return fail(format!(
                "{label} column footprint [{begin}, {end}) exceeds TMEM columns [0, 512)"
            ));
        }
    }
    Ok(())
}

fn resolve_d_footprint(
    m_per_cta: u32,
    n: u32,
    datapath: TmemDatapath,
) -> LayoutResult<TmemFootprint> {
    let (physical_cols, max_lane_offset, layout) = match datapath {
        TmemDatapath::A | TmemDatapath::D => (n, 127, TmemLayoutKind::Identity),
        TmemDatapath::B | TmemDatapath::E => {
            if n % 2 != 0 {
                return fail(format!(
                    "tcgen05_mma {:?} accumulator layout requires even logical N, got {n}",
                    datapath
                ));
            }
            (n / 2, 127, TmemLayoutKind::Fold2)
        }
        TmemDatapath::F => (n, 111, TmemLayoutKind::Scatter4),
    };
    Ok(TmemFootprint {
        logical: LogicalMatrixShape {
            rows: m_per_cta,
            cols: n,
        },
        physical_cols,
        max_lane_offset,
        layout,
    })
}

fn resolve_a_tmem_footprint(
    m_per_cta: u32,
    k: u32,
    format: MmaElemFormat,
    datapath: TmemDatapath,
    form: TmemAForm,
) -> LayoutResult<TmemFootprint> {
    let layout = match (datapath, form) {
        (TmemDatapath::A | TmemDatapath::D | TmemDatapath::F, TmemAForm::Flat) => {
            TmemLayoutKind::Identity
        }
        (TmemDatapath::B, TmemAForm::Flat) => TmemLayoutKind::Replica2,
        (TmemDatapath::B | TmemDatapath::E, TmemAForm::BankBatched) => TmemLayoutKind::BankBatched,
        (TmemDatapath::E, TmemAForm::Flat) => {
            return fail(
                "tcgen05_mma Layout E TMEM A requires BankBatched; Flat hides the second lane bank",
            )
        }
        (_, TmemAForm::BankBatched) => {
            return fail("tcgen05_mma BankBatched TMEM A is only valid for Layout B or Layout E")
        }
    };
    let bits = checked_mul(k, format_bits(format), "tcgen05_mma TMEM A bit span")?;
    if bits % 32 != 0 {
        return fail("tcgen05_mma TMEM A K extent does not fill whole 32-bit columns");
    }
    let max_lane_offset = match (datapath, layout) {
        (TmemDatapath::F, TmemLayoutKind::Identity) => 63,
        (TmemDatapath::A | TmemDatapath::D, TmemLayoutKind::Identity) => 127,
        (_, TmemLayoutKind::Replica2 | TmemLayoutKind::BankBatched) => 127,
        _ => {
            return fail("internal tcgen05 TMEM A layout/datapath mismatch");
        }
    };
    Ok(TmemFootprint {
        logical: LogicalMatrixShape {
            rows: m_per_cta,
            cols: k,
        },
        physical_cols: bits / 32,
        max_lane_offset,
        layout,
    })
}

fn check_block_scale_pair(
    format: MmaElemFormat,
    scale: &BlockScaleSpec,
    cta_group: u8,
    mma_m: u32,
) -> LayoutResult<()> {
    let expected_sf_per_mma = match (format, scale.scale_format) {
        (MmaElemFormat::F8E4M3, ScaleFormat::E8M0FNU) => 1,
        (MmaElemFormat::F4E2M1, ScaleFormat::E4M3FN) => 4,
        _ => {
            return fail(format!(
                "tcgen05_mma unsupported block-scale pair: data={}, scale={}",
                format.as_str(),
                scale.scale_format.as_str()
            ))
        }
    };
    if scale.sf_per_mma != expected_sf_per_mma {
        return fail(format!(
            "tcgen05_mma data/scale pair requires sf_per_mma={expected_sf_per_mma}, got {}",
            scale.sf_per_mma
        ));
    }
    if scale.sf_reuse == 0 {
        return fail("tcgen05_mma sf_reuse must be positive");
    }
    if cta_group == 1 && mma_m != 128 {
        return fail("tcgen05_mma block-scaled cta_group=1 requires mma_m=128");
    }
    if format == MmaElemFormat::F4E2M1
        && !((cta_group == 1 && mma_m == 128) || (cta_group == 2 && matches!(mma_m, 128 | 256)))
    {
        return fail(
            "tcgen05_mma F4E2M1 requires (cta_group=1, mma_m=128) or \
             (cta_group=2, mma_m=128 or 256)",
        );
    }
    Ok(())
}

fn resolve_scale_footprint(
    rows: u32,
    k_iters: u32,
    spec: &BlockScaleSpec,
    k_offset: u32,
    label: &str,
) -> LayoutResult<ResolvedScaleFootprint> {
    if rows == 0 {
        return fail(format!("{label} rows must be positive"));
    }
    let padded_rows = checked_mul(div_ceil(rows, 32, label)?, 32, label)?;
    let groups = div_ceil(k_iters, spec.sf_reuse, label)?;
    let sf_k = checked_mul(groups, spec.sf_per_mma, &format!("{label} SF_K"))?;
    if sf_k % spec.sf_per_mma != 0 {
        return fail(format!("{label} SF_K must be divisible by sf_per_mma"));
    }
    if spec.sf_per_mma > 4 || 4 % spec.sf_per_mma != 0 {
        return fail(format!(
            "{label} sf_per_mma must be a positive divisor of four"
        ));
    }
    let logical_last = checked_mul(sf_k, spec.sf_reuse, &format!("{label} logical last dim"))?;
    let row_blocks = padded_rows / 32;
    let mut pack_factor = 4 / spec.sf_per_mma;
    while pack_factor > 1 && groups % pack_factor != 0 {
        pack_factor /= 2;
    }
    let last_group = groups - 1;
    let tcol_element_max = checked_mul(row_blocks - 1, 4, label)?
        .checked_add(checked_mul(
            last_group / pack_factor,
            checked_mul(row_blocks, 4, label)?,
            label,
        )?)
        .and_then(|value| value.checked_add((last_group % pack_factor) * spec.sf_per_mma))
        .and_then(|value| value.checked_add(spec.sf_per_mma - 1))
        .ok_or_else(|| Tcgen05LayoutError {
            message: format!("{label} canonical TCol span overflows u32"),
        })?;
    let tcol_element_span = tcol_element_max
        .checked_add(k_offset)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| Tcgen05LayoutError {
            message: format!("{label} shifted canonical TCol span overflows u32"),
        })?;
    let physical_cols = div_ceil(
        tcol_element_span,
        4,
        &format!("{label} physical column span"),
    )?;
    Ok(ResolvedScaleFootprint {
        footprint: TmemFootprint {
            logical: LogicalMatrixShape {
                rows: padded_rows,
                cols: logical_last,
            },
            physical_cols,
            max_lane_offset: 127,
            layout: TmemLayoutKind::ScaleFactor {
                sf_per_mma: spec.sf_per_mma,
                sf_reuse: spec.sf_reuse,
            },
        },
        operand_rows: rows,
        groups,
        sf_k,
        logical_last,
        pack_factor,
        tcol_element_span,
        cell_delta: k_offset / 4,
        subbyte: (k_offset % 4) as u8,
    })
}

/// Resolve and validate the complete logical and physical geometry of one
/// `Tcgen05Mma`.
#[allow(clippy::too_many_arguments)]
pub fn resolve_tcgen05_mma(
    dst: &TmemAddr,
    a: &MmaAOperand,
    b: &SmemTile,
    mma_m: u32,
    mma_n: u32,
    format: MmaElemFormat,
    block_scale: Option<&BlockScaleSpec>,
    trans_a: bool,
    trans_b: bool,
    ws: bool,
    cta_group: u8,
) -> LayoutResult<ResolvedMma> {
    let datapath = resolve_datapath(mma_m, cta_group, ws)?;
    let m_per_cta = mma_m / u32::from(cta_group);
    let mma_k = mma_k_for(format);
    if format == MmaElemFormat::F4E2M1 && (trans_a || trans_b) {
        return fail("tcgen05_mma F4E2M1 SMEM operands do not support transpose");
    }

    validate_format_tile(b, format, "tcgen05_mma b")?;
    let (n_per_cta, b_storage_k) = if trans_b {
        (b.cols, b.rows)
    } else {
        (b.rows, b.cols)
    };
    let k = logical_k_from_storage(b_storage_k, format)?;
    if k == 0 || k % mma_k != 0 {
        return fail(format!(
            "tcgen05_mma logical K={k} must be a positive multiple of mma_k={mma_k}"
        ));
    }

    let n = checked_mul(n_per_cta, u32::from(cta_group), "tcgen05_mma logical N")?;
    validate_mma_n(mma_n, n, mma_m, cta_group, block_scale.is_some())?;

    let (a_shape, a_tmem) = match a {
        MmaAOperand::Smem(tile) => {
            validate_format_tile(tile, format, "tcgen05_mma a")?;
            let (a_m_per_cta, a_storage_k) = if trans_a {
                (tile.cols, tile.rows)
            } else {
                (tile.rows, tile.cols)
            };
            let a_k = logical_k_from_storage(a_storage_k, format)?;
            if a_m_per_cta
                .checked_mul(u32::from(cta_group))
                .filter(|&value| value == mma_m)
                .is_none()
            {
                return fail(format!(
                    "tcgen05_mma A per-CTA M={a_m_per_cta} * cta_group={cta_group} \
                     must equal mma_m={mma_m}"
                ));
            }
            if a_k != k {
                return fail(format!(
                    "tcgen05_mma A logical K={a_k} does not match B logical K={k}"
                ));
            }
            (
                LogicalMatrixShape {
                    rows: a_m_per_cta,
                    cols: a_k,
                },
                None,
            )
        }
        MmaAOperand::Tmem { addr, form } => {
            if trans_a {
                return fail("tcgen05_mma trans_a must be false when A is in TMEM");
            }
            let footprint = resolve_a_tmem_footprint(m_per_cta, k, format, datapath, *form)?;
            ensure_addr_fits(addr, &footprint, "tcgen05_mma a")?;
            (
                LogicalMatrixShape {
                    rows: m_per_cta,
                    cols: k,
                },
                Some(footprint),
            )
        }
    };

    let d = resolve_d_footprint(m_per_cta, n, datapath)?;
    ensure_addr_fits(dst, &d, "tcgen05_mma dst")?;

    let (sfa, sfb) = match block_scale {
        None => {
            if format == MmaElemFormat::F4E2M1 {
                return fail(format!(
                    "tcgen05_mma format {} requires block_scale",
                    format.as_str()
                ));
            }
            (None, None)
        }
        Some(spec) => {
            if matches!(format, MmaElemFormat::F16 | MmaElemFormat::BF16) {
                return fail(format!(
                    "tcgen05_mma dense format {} does not accept block_scale",
                    format.as_str()
                ));
            }
            check_block_scale_pair(format, spec, cta_group, mma_m)?;
            let k_iters = k / mma_k;
            let sfa = resolve_scale_footprint(m_per_cta, k_iters, spec, spec.sfa_k_offset, "sfa")?;
            let sfb = resolve_scale_footprint(n, k_iters, spec, spec.sfb_k_offset, "sfb")?;
            if sfa.subbyte != sfb.subbyte {
                return fail(format!(
                    "tcgen05_mma sfa/sfb offsets must select the same sf_id byte; got {} and {}",
                    sfa.subbyte, sfb.subbyte
                ));
            }
            ensure_addr_fits(&spec.sfa, &sfa.footprint, "tcgen05_mma sfa")?;
            ensure_addr_fits(&spec.sfb, &sfb.footprint, "tcgen05_mma sfb")?;
            (Some(sfa), Some(sfb))
        }
    };

    Ok(ResolvedMma {
        m: mma_m,
        n,
        k,
        mma_k,
        format,
        per_cta: PerCtaMmaShape {
            m: m_per_cta,
            n: n_per_cta,
            k,
        },
        datapath,
        d,
        a: a_shape,
        b: LogicalMatrixShape {
            rows: n_per_cta,
            cols: k,
        },
        a_tmem,
        sfa,
        sfb,
    })
}

fn cp_shape_dims(shape: Tcgen05CpShape) -> (u32, u32) {
    match shape {
        Tcgen05CpShape::B128x256 => (128, 256),
        Tcgen05CpShape::B4x256 => (4, 256),
        Tcgen05CpShape::B128x128 => (128, 128),
        Tcgen05CpShape::B64x128 => (64, 128),
        Tcgen05CpShape::B32x128 => (32, 128),
    }
}

fn cp_lane_layout(
    shape: Tcgen05CpShape,
    multicast: Tcgen05CpMulticast,
) -> LayoutResult<(CpLaneLayout, u32)> {
    match (shape, multicast) {
        (Tcgen05CpShape::B128x256 | Tcgen05CpShape::B128x128, Tcgen05CpMulticast::None) => {
            Ok((CpLaneLayout::Identity, 127))
        }
        (Tcgen05CpShape::B4x256, Tcgen05CpMulticast::None) => Ok((CpLaneLayout::Quadrant4, 96)),
        (Tcgen05CpShape::B64x128, Tcgen05CpMulticast::Warp2_02_13) => {
            Ok((CpLaneLayout::Warp2_02_13, 127))
        }
        (Tcgen05CpShape::B64x128, Tcgen05CpMulticast::Warp2_01_23) => {
            Ok((CpLaneLayout::Warp2_01_23, 127))
        }
        (Tcgen05CpShape::B32x128, Tcgen05CpMulticast::Warp4) => Ok((CpLaneLayout::Warp4, 127)),
        _ => fail(format!(
            "illegal tcgen05.cp shape/multicast combination: {}/{}",
            shape.as_str(),
            multicast.as_str()
        )),
    }
}

/// Resolve and validate one physical `Tcgen05Cp`.
pub fn resolve_tcgen05_cp(
    dst: &TmemAddr,
    src: &SmemTile,
    shape: Tcgen05CpShape,
    multicast: Tcgen05CpMulticast,
    cta_group: u8,
) -> LayoutResult<ResolvedCp> {
    check_cta_group(cta_group, "tcgen05_cp cta_group")?;
    validate_cp_smem_tile(src, "tcgen05_cp src")?;
    let (atom_rows, atom_bits) = cp_shape_dims(shape);
    if src.rows % atom_rows != 0 {
        return fail(format!(
            "tcgen05_cp {} source rows must be a multiple of {atom_rows}, got {}",
            shape.as_str(),
            src.rows
        ));
    }
    let bits = tcgen05_cp_source_bits(src);
    if bits > 32 || 32 % bits != 0 {
        return fail(format!(
            "tcgen05_cp source dtype must pack exactly into 32-bit TMEM cells, got {bits} bits"
        ));
    }
    if atom_bits % bits != 0 {
        return fail(format!(
            "tcgen05_cp {} atom width {atom_bits} bits is not divisible by its {}-bit source dtype",
            shape.as_str(),
            bits
        ));
    }
    let atom_cols = atom_bits / bits;
    if src.cols % atom_cols != 0 {
        return fail(format!(
            "tcgen05_cp {} source cols must be a multiple of the {atom_cols}-element atom width, got {}",
            shape.as_str(),
            src.cols
        ));
    }
    let source_row_bits = checked_mul(src.cols, bits, "tcgen05_cp source row bit width")?;
    if source_row_bits % 32 != 0 {
        return fail("tcgen05_cp complete source rows must occupy whole 32-bit TMEM cells");
    }
    let physical_row_bits = src
        .tensor
        .shape
        .last()
        .copied()
        .ok_or_else(|| Tcgen05LayoutError {
            message: "tcgen05_cp source tensor must have at least one dimension".to_string(),
        })
        .and_then(|extent| {
            u32::try_from(extent)
                .map_err(|_| Tcgen05LayoutError {
                    message: "tcgen05_cp source row extent does not fit u32".to_string(),
                })
                .and_then(|extent| {
                    checked_mul(extent, bits, "tcgen05_cp physical source row bit width")
                })
        })?;
    let source_layout = if atom_bits == 256 {
        match src.tensor.layout.as_ref() {
            Some(Layout::Swizzle(layout)) if layout.swizzle == Swizzle::B32 => {
                CpSmemLayout::Swizzle32B
            }
            _ => {
                return fail(
                    "tcgen05_cp 256-bit source requires an explicit B32 \
                     mma_operandAB SMEM owner; plain row-major bytes cannot express \
                     the hardware descriptor walk",
                );
            }
        }
    } else {
        if src.tensor.layout.is_some() || physical_row_bits != 128 {
            return fail(format!(
                "tcgen05_cp 128-bit source requires a plain row-major SMEM owner \
                 with an exact 16-byte physical row, got layout={:?}, row_bits={physical_row_bits}",
                src.tensor.layout
            ));
        }
        CpSmemLayout::Plain16B
    };
    if source_layout == CpSmemLayout::Swizzle32B {
        let atom_cols = 256 / bits;
        let row = literal_i64(&src.row_offset).ok_or_else(|| Tcgen05LayoutError {
            message: "tcgen05_cp B32 source row_offset must be a static atom-aligned integer"
                .to_string(),
        })?;
        if row % 8 != 0 {
            return fail(format!(
                "tcgen05_cp B32 source row_offset must be 8-row atom aligned, got {row}"
            ));
        }
        let view_rows = src.rows.div_ceil(8) * 8;
        let row_extent = src.tensor.shape[src.tensor.shape.len() - 2] as u64;
        let view_end = (row as u64) + u64::from(view_rows);
        if view_end > row_extent {
            return fail(format!(
                "tcgen05_cp B32 source needs an enclosing {view_rows}-row view, \
                 but row_offset {row} reaches past tensor extent {row_extent}"
            ));
        }
        let col = literal_i64(&src.col_offset).ok_or_else(|| Tcgen05LayoutError {
            message: "tcgen05_cp B32 source col_offset must be a static atom-aligned integer"
                .to_string(),
        })?;
        if col % i64::from(atom_cols) != 0 {
            return fail(format!(
                "tcgen05_cp B32 source col_offset must be {atom_cols}-element atom aligned, got {col}"
            ));
        }
    }
    let source_row_cells = source_row_bits / 32;
    let row_atoms = src.rows / atom_rows;
    let (lane_layout, max_lane_offset) = cp_lane_layout(shape, multicast)?;
    let (physical_cols, max_lane_offset) = if lane_layout == CpLaneLayout::Quadrant4 {
        if src.rows > 128 {
            return fail("tcgen05_cp 4x256b source rows must be at most 128");
        }
        (
            source_row_cells,
            96 + src
                .rows
                .checked_div(4)
                .expect("tcgen05_cp atom rows are nonzero")
                - 1,
        )
    } else {
        (
            checked_mul(
                row_atoms,
                source_row_cells,
                "tcgen05_cp destination column span",
            )?,
            max_lane_offset,
        )
    };
    let target = TmemFootprint {
        logical: LogicalMatrixShape {
            rows: src.rows,
            cols: src.cols,
        },
        physical_cols,
        max_lane_offset,
        layout: TmemLayoutKind::Copy(lane_layout),
    };
    ensure_addr_fits(dst, &target, "tcgen05_cp dst")?;
    if let Some(col) = literal_i64(&dst.col) {
        let absolute_col = i64::from(dst.tensor.start_col)
            .checked_add(col)
            .ok_or_else(|| Tcgen05LayoutError {
                message: "tcgen05_cp destination column overflows i64".to_string(),
            })?;
        // PTX tcgen05.cp taddr operands are 128-bit aligned. TMEM columns
        // address 32-bit cells, so the physical base column is a multiple of
        // four even for the 256-bit shapes.
        if absolute_col % 4 != 0 {
            return fail(format!(
                "tcgen05_cp destination column must be 4-cell (128-bit) aligned, got {absolute_col}"
            ));
        }
    }
    Ok(ResolvedCp {
        source: LogicalMatrixShape {
            rows: src.rows,
            cols: src.cols,
        },
        source_row_cells,
        atom_rows,
        target,
        lane_layout,
        source_layout,
        cta_group,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Layout, SmemSwizzleLayout, Swizzle, Tensor, TmemTensor};
    use std::sync::Arc;

    fn smem(id: u32, dtype: DType, shape: Vec<usize>) -> Arc<Tensor> {
        Arc::new(Tensor {
            id,
            space: MemorySpace::Smem,
            dtype,
            shape,
            layout: Some(Layout::Swizzle(SmemSwizzleLayout {
                swizzle: Swizzle::B128,
            })),
            byte_offset: Some(0),
        })
    }

    fn cp_smem(id: u32, dtype: DType, shape: Vec<usize>, layout: Option<Swizzle>) -> Arc<Tensor> {
        Arc::new(Tensor {
            id,
            space: MemorySpace::Smem,
            dtype,
            shape,
            layout: layout.map(|swizzle| Layout::Swizzle(SmemSwizzleLayout { swizzle })),
            byte_offset: Some(0),
        })
    }

    fn tile(tensor: Arc<Tensor>, rows: u32, cols: u32) -> SmemTile {
        let rank = tensor.shape.len();
        SmemTile {
            tensor,
            prefix_indices: vec![ScalarValue::Int(0); rank - 2],
            row_offset: ScalarValue::Int(0),
            col_offset: ScalarValue::Int(0),
            rows,
            cols,
        }
    }

    fn addr(start_col: u32, row: i64, col: i64) -> TmemAddr {
        TmemTensor { start_col }.at(ScalarValue::Int(row), ScalarValue::Int(col))
    }

    #[test]
    fn dense_shapes_and_datapaths_resolve_exactly() {
        let a = MmaAOperand::Smem(tile(smem(1, DType::F16, vec![128, 64]), 128, 64));
        let b = tile(smem(2, DType::F16, vec![128, 64]), 128, 64);
        let r = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &a,
            &b,
            128,
            128,
            MmaElemFormat::F16,
            None,
            false,
            false,
            false,
            1,
        )
        .unwrap();
        assert_eq!(
            r.per_cta,
            PerCtaMmaShape {
                m: 128,
                n: 128,
                k: 64
            }
        );
        assert_eq!(r.datapath, TmemDatapath::D);
        assert_eq!(r.d.physical_cols, 128);
        assert_eq!(r.d.layout, TmemLayoutKind::Identity);

        let a = MmaAOperand::Smem(tile(smem(7, DType::F8E4M3, vec![128, 32]), 128, 32));
        let b = tile(smem(8, DType::F8E4M3, vec![16, 32]), 16, 32);
        let r = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &a,
            &b,
            128,
            16,
            MmaElemFormat::F8E4M3,
            None,
            false,
            false,
            false,
            1,
        )
        .unwrap();
        assert_eq!(r.format, MmaElemFormat::F8E4M3);
        assert_eq!(r.mma_k, 32);
        assert!(r.sfa.is_none() && r.sfb.is_none());

        let a = MmaAOperand::Smem(tile(smem(3, DType::F16, vec![64, 64]), 64, 64));
        let b = tile(smem(4, DType::F16, vec![64, 64]), 64, 64);
        let r = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &a,
            &b,
            128,
            64,
            MmaElemFormat::F16,
            None,
            false,
            false,
            false,
            2,
        )
        .unwrap();
        assert_eq!(r.datapath, TmemDatapath::B);
        assert_eq!(r.n, 128);
        assert_eq!(r.d.physical_cols, 64);
        assert_eq!(r.d.layout, TmemLayoutKind::Fold2);
        assert_eq!(r.d_cell(3, 70), Some(TmemCellCoord { lane: 67, col: 6 }));

        let a = MmaAOperand::Smem(tile(smem(5, DType::F16, vec![64, 32]), 64, 32));
        let b = tile(smem(6, DType::F16, vec![64, 32]), 64, 32);
        let r = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &a,
            &b,
            64,
            64,
            MmaElemFormat::F16,
            None,
            false,
            false,
            false,
            1,
        )
        .unwrap();
        assert_eq!(r.datapath, TmemDatapath::F);
        assert_eq!(r.d_cell(33, 2), Some(TmemCellCoord { lane: 65, col: 2 }));
    }

    #[test]
    fn dense_instruction_n_obeys_physical_kind_and_m_rules() {
        let a = MmaAOperand::Smem(tile(smem(1, DType::F16, vec![128, 16]), 128, 16));
        let b = tile(smem(2, DType::F16, vec![16, 16]), 16, 16);
        let error = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &a,
            &b,
            128,
            8,
            MmaElemFormat::F16,
            None,
            false,
            false,
            false,
            1,
        )
        .unwrap_err();
        assert!(
            error.message.contains("multiple of 16"),
            "{}",
            error.message
        );

        let a = MmaAOperand::Smem(tile(smem(3, DType::F16, vec![64, 16]), 64, 16));
        let b = tile(smem(4, DType::F16, vec![8, 16]), 8, 16);
        assert!(resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &a,
            &b,
            64,
            8,
            MmaElemFormat::F16,
            None,
            false,
            false,
            false,
            1,
        )
        .is_ok());

        let a = MmaAOperand::Smem(tile(smem(5, DType::Bf16, vec![64, 16]), 64, 16));
        let b = tile(smem(6, DType::Bf16, vec![16, 16]), 16, 16);
        let error = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &a,
            &b,
            128,
            16,
            MmaElemFormat::BF16,
            None,
            false,
            false,
            false,
            2,
        )
        .unwrap_err();
        assert!(
            error.message.contains("multiple of 32"),
            "{}",
            error.message
        );
    }

    #[test]
    fn tmem_a_forms_follow_datapath() {
        let b = tile(smem(1, DType::F16, vec![64, 32]), 64, 32);
        let flat = MmaAOperand::Tmem {
            addr: addr(128, 0, 0),
            form: TmemAForm::Flat,
        };
        let r = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &flat,
            &b,
            128,
            64,
            MmaElemFormat::F16,
            None,
            false,
            false,
            false,
            2,
        )
        .unwrap();
        assert_eq!(r.a_tmem.unwrap().layout, TmemLayoutKind::Replica2);
        assert_eq!(r.a_tmem.unwrap().physical_cols, 16);
        assert_eq!(
            r.a_tmem_element(1, 3, 1),
            Some(TmemElementCoord {
                lane: 67,
                col: 0,
                bit_offset: 16
            })
        );

        let error = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &flat,
            &b,
            64,
            64,
            MmaElemFormat::F16,
            None,
            false,
            false,
            true,
            1,
        )
        .unwrap_err();
        assert!(error.message.contains("requires BankBatched"));

        let banked = MmaAOperand::Tmem {
            addr: addr(128, 0, 0),
            form: TmemAForm::BankBatched,
        };
        let r = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &banked,
            &b,
            64,
            64,
            MmaElemFormat::F16,
            None,
            false,
            false,
            true,
            1,
        )
        .unwrap();
        assert_eq!(
            r.a_tmem_element(1, 7, 2),
            Some(TmemElementCoord {
                lane: 71,
                col: 1,
                bit_offset: 0
            })
        );
    }

    #[test]
    fn block_scale_metadata_and_offsets_are_physical() {
        let a = MmaAOperand::Smem(tile(smem(1, DType::F8E4M3, vec![128, 128]), 128, 128));
        let b = tile(smem(2, DType::F8E4M3, vec![128, 128]), 128, 128);
        let scale = BlockScaleSpec {
            sfa: addr(256, 0, 0),
            sfb: addr(320, 0, 0),
            sfa_k_offset: 5,
            sfb_k_offset: 9,
            scale_format: ScaleFormat::E8M0FNU,
            sf_per_mma: 1,
            sf_reuse: 4,
        };
        let r = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &a,
            &b,
            128,
            128,
            MmaElemFormat::F8E4M3,
            Some(&scale),
            false,
            false,
            false,
            1,
        )
        .unwrap();
        let sfa = r.sfa.unwrap();
        assert_eq!(sfa.groups, 1);
        assert_eq!(sfa.sf_k, 1);
        assert_eq!(sfa.logical_last, 4);
        assert_eq!(sfa.pack_factor, 1);
        assert_eq!(sfa.tcol_element_span, 18);
        assert_eq!(sfa.footprint.physical_cols, 5);
        assert_eq!((sfa.cell_delta, sfa.subbyte), (1, 1));
        assert_eq!(
            sfa.physical_byte(32, 3),
            Some(TmemByteCoord {
                lane: 0,
                col: 2,
                subbyte: 1
            })
        );
        assert_eq!(
            sfa.physical_byte(127, 0),
            Some(TmemByteCoord {
                lane: 31,
                col: 4,
                subbyte: 1
            })
        );
        assert_eq!(
            sfa.physical_bytes(32, 3).unwrap(),
            [
                TmemByteCoord {
                    lane: 0,
                    col: 2,
                    subbyte: 1
                },
                TmemByteCoord {
                    lane: 32,
                    col: 2,
                    subbyte: 1
                },
                TmemByteCoord {
                    lane: 64,
                    col: 2,
                    subbyte: 1
                },
                TmemByteCoord {
                    lane: 96,
                    col: 2,
                    subbyte: 1
                },
            ]
        );
        assert_eq!(r.sfb.unwrap().subbyte, 1);
    }

    #[test]
    fn scale_footprint_matches_canonical_sparse_tcol_span() {
        let a = MmaAOperand::Smem(tile(smem(1, DType::F8E4M3, vec![128, 384]), 128, 384));
        let b = tile(smem(2, DType::F8E4M3, vec![128, 384]), 128, 384);
        let scale = BlockScaleSpec {
            sfa: addr(128, 0, 0),
            sfb: addr(160, 0, 0),
            sfa_k_offset: 0,
            sfb_k_offset: 0,
            scale_format: ScaleFormat::E8M0FNU,
            sf_per_mma: 1,
            sf_reuse: 4,
        };
        let r = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &a,
            &b,
            128,
            128,
            MmaElemFormat::F8E4M3,
            Some(&scale),
            false,
            false,
            false,
            1,
        )
        .unwrap();
        let sfa = r.sfa.unwrap();
        assert_eq!(sfa.groups, 3);
        assert_eq!(sfa.pack_factor, 1);
        assert_eq!(sfa.tcol_element_span, 45);
        assert_eq!(sfa.footprint.physical_cols, 12);
        assert_eq!(
            sfa.physical_byte(127, 8),
            Some(TmemByteCoord {
                lane: 31,
                col: 11,
                subbyte: 0
            })
        );
    }

    #[test]
    fn block_scale_rejects_different_sf_ids() {
        let a = MmaAOperand::Smem(tile(smem(1, DType::U8, vec![128, 32]), 128, 32));
        let b = tile(smem(2, DType::U8, vec![128, 32]), 128, 32);
        let scale = BlockScaleSpec {
            sfa: addr(256, 0, 0),
            sfb: addr(320, 0, 0),
            sfa_k_offset: 0,
            sfb_k_offset: 1,
            scale_format: ScaleFormat::E4M3FN,
            sf_per_mma: 4,
            sf_reuse: 1,
        };
        let error = resolve_tcgen05_mma(
            &addr(0, 0, 0),
            &a,
            &b,
            128,
            128,
            MmaElemFormat::F4E2M1,
            Some(&scale),
            false,
            false,
            false,
            1,
        )
        .unwrap_err();
        assert!(error.message.contains("same sf_id byte"));
    }

    #[test]
    fn cp_legal_matrix_and_target_mappings() {
        let cases = [
            (
                Tcgen05CpShape::B128x256,
                Tcgen05CpMulticast::None,
                128,
                32,
                CpLaneLayout::Identity,
                127,
            ),
            (
                Tcgen05CpShape::B4x256,
                Tcgen05CpMulticast::None,
                4,
                32,
                CpLaneLayout::Quadrant4,
                96,
            ),
            (
                Tcgen05CpShape::B128x128,
                Tcgen05CpMulticast::None,
                128,
                16,
                CpLaneLayout::Identity,
                127,
            ),
            (
                Tcgen05CpShape::B64x128,
                Tcgen05CpMulticast::Warp2_02_13,
                64,
                16,
                CpLaneLayout::Warp2_02_13,
                127,
            ),
            (
                Tcgen05CpShape::B64x128,
                Tcgen05CpMulticast::Warp2_01_23,
                64,
                16,
                CpLaneLayout::Warp2_01_23,
                127,
            ),
            (
                Tcgen05CpShape::B32x128,
                Tcgen05CpMulticast::Warp4,
                32,
                16,
                CpLaneLayout::Warp4,
                127,
            ),
        ];
        for (index, (shape, multicast, rows, cols, lanes, max_lane)) in
            cases.into_iter().enumerate()
        {
            let owner_layout = if shape.as_str().ends_with("256b") {
                Some(Swizzle::B32)
            } else {
                None
            };
            let owner_rows = if rows == 4 { 8 } else { rows as usize };
            let src = tile(
                cp_smem(
                    index as u32,
                    DType::U8,
                    vec![owner_rows, cols],
                    owner_layout,
                ),
                rows,
                cols as u32,
            );
            let r = resolve_tcgen05_cp(&addr(0, 0, 0), &src, shape, multicast, 1).unwrap();
            assert_eq!(r.lane_layout, lanes);
            assert_eq!(
                r.source_layout,
                if owner_layout.is_some() {
                    CpSmemLayout::Swizzle32B
                } else {
                    CpSmemLayout::Plain16B
                }
            );
            assert_eq!(r.target.layout, TmemLayoutKind::Copy(lanes));
            assert_eq!(r.target.max_lane_offset, max_lane);
            assert_eq!(
                r.target.logical,
                LogicalMatrixShape {
                    rows,
                    cols: cols as u32
                }
            );
        }
    }

    #[test]
    fn cp_tiles_multiple_instruction_atoms() {
        let src = tile(
            cp_smem(1, DType::U8, vec![256, 64], Some(Swizzle::B32)),
            256,
            64,
        );
        let r = resolve_tcgen05_cp(
            &addr(0, 0, 0),
            &src,
            Tcgen05CpShape::B128x256,
            Tcgen05CpMulticast::None,
            1,
        )
        .unwrap();
        assert_eq!(r.source_row_cells, 16);
        assert_eq!(r.target.physical_cols, 32);
        assert_eq!(
            r.target_cells(129, 3).unwrap(),
            vec![TmemCellCoord { lane: 1, col: 19 }]
        );

        let src = tile(
            cp_smem(2, DType::U8, vec![64, 64], Some(Swizzle::B32)),
            64,
            64,
        );
        let r = resolve_tcgen05_cp(
            &addr(0, 0, 0),
            &src,
            Tcgen05CpShape::B4x256,
            Tcgen05CpMulticast::None,
            1,
        )
        .unwrap();
        assert_eq!(r.target.physical_cols, 16);
        assert_eq!(r.target.max_lane_offset, 111);
        assert_eq!(
            r.target_cells(49, 7).unwrap(),
            vec![TmemCellCoord { lane: 44, col: 7 }]
        );

        let src = tile(cp_smem(3, DType::U8, vec![64, 16], None), 64, 16);
        let r = resolve_tcgen05_cp(
            &addr(0, 0, 0),
            &src,
            Tcgen05CpShape::B32x128,
            Tcgen05CpMulticast::Warp4,
            1,
        )
        .unwrap();
        assert_eq!(r.target.physical_cols, 8);
        assert_eq!(
            r.target_cells(33, 2).unwrap(),
            vec![
                TmemCellCoord { lane: 1, col: 6 },
                TmemCellCoord { lane: 33, col: 6 },
                TmemCellCoord { lane: 65, col: 6 },
                TmemCellCoord { lane: 97, col: 6 },
            ]
        );
    }

    #[test]
    fn cp_rejects_illegal_multicast_and_non_atom_source_shape() {
        let src = tile(
            cp_smem(1, DType::U8, vec![128, 32], Some(Swizzle::B32)),
            128,
            32,
        );
        let error = resolve_tcgen05_cp(
            &addr(0, 0, 0),
            &src,
            Tcgen05CpShape::B128x256,
            Tcgen05CpMulticast::Warp4,
            1,
        )
        .unwrap_err();
        assert!(error.message.contains("illegal"));

        let narrow = tile(
            cp_smem(2, DType::U8, vec![128, 16], Some(Swizzle::B32)),
            128,
            16,
        );
        let error = resolve_tcgen05_cp(
            &addr(0, 0, 0),
            &narrow,
            Tcgen05CpShape::B128x256,
            Tcgen05CpMulticast::None,
            1,
        )
        .unwrap_err();
        assert!(error.message.contains("atom width"));

        let partial_rows = tile(
            cp_smem(3, DType::U8, vec![192, 32], Some(Swizzle::B32)),
            192,
            32,
        );
        let error = resolve_tcgen05_cp(
            &addr(0, 0, 0),
            &partial_rows,
            Tcgen05CpShape::B128x256,
            Tcgen05CpMulticast::None,
            1,
        )
        .unwrap_err();
        assert!(error.message.contains("multiple of 128"));

        let error = resolve_tcgen05_cp(
            &addr(0, 0, 2),
            &src,
            Tcgen05CpShape::B128x256,
            Tcgen05CpMulticast::None,
            1,
        )
        .unwrap_err();
        assert!(error.message.contains("4-cell"));

        let wide_dtype = tile(cp_smem(4, DType::U64, vec![128, 4], None), 128, 4);
        let error = resolve_tcgen05_cp(
            &addr(0, 0, 0),
            &wide_dtype,
            Tcgen05CpShape::B128x256,
            Tcgen05CpMulticast::None,
            1,
        )
        .unwrap_err();
        assert!(error.message.contains("pack exactly into 32-bit"));

        let plain_256 = tile(cp_smem(5, DType::U32, vec![128, 8], None), 128, 8);
        let error = resolve_tcgen05_cp(
            &addr(0, 0, 0),
            &plain_256,
            Tcgen05CpShape::B128x256,
            Tcgen05CpMulticast::None,
            1,
        )
        .unwrap_err();
        assert!(error.message.contains("explicit B32"));
    }
}
