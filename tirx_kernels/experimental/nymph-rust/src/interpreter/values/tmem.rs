//! CTA-local TMEM scratchpad — port of `interpreter/values/tmem.py`.
//!
//! A 128-lane × 512-column grid of physical 32-bit cells. Storage is lane-major:
//! `data[[lane, col]]` with shape `[TMEM_ROWS, TMEM_COLS]`, matching the Python
//! value model. Logical F32/I32/U32 TMEM values are bit-reinterpreted through
//! those cells. F16/BF16 logical values are packed two per physical cell, with
//! even logical columns in the low half and odd logical columns in the high half.
//!
//! TMEM is not a tensor: there is no logical→physical layout layer. Every
//! accessor takes absolute physical (lane, col) cell coordinates plus the
//! `DType` that says how the cells are (un)packed.

use super::super::diagnostics::{IResult, InterpreterError};
use super::arrays::ValueArray1;
use super::dtypes::round_bf16_scalar;
use crate::ir::DType;
use half::f16;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub const TMEM_ROWS: usize = 128; // lanes
pub const TMEM_COLS: usize = 512; // columns

/// (lane, col).
pub type TmemCellKey = (usize, usize);

#[derive(Clone, Debug)]
pub struct TmemScratchpad {
    /// lane-major physical cell bits: `data[[lane, col]]`.
    pub data: Array2<u32>,
    pub valid: Array2<bool>,
}

impl Default for TmemScratchpad {
    fn default() -> Self {
        TmemScratchpad {
            data: Array2::zeros((TMEM_ROWS, TMEM_COLS)),
            valid: Array2::from_elem((TMEM_ROWS, TMEM_COLS), false),
        }
    }
}

impl TmemScratchpad {
    /// A temporary f32 view over the physical u32 cells for BLAS accumulator writes.
    ///
    /// TMEM stores f32 accumulator values as raw IEEE bits. The view is only used
    /// while cblas writes the result and no u32 view is accessed concurrently.
    pub fn data_as_f32_mut(&mut self) -> &mut [f32] {
        let data = self.data.as_slice_mut().expect("tmem grid is contiguous");
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<f32>(), data.len()) }
    }

    fn check_cell(&self, lane: usize, col: usize) -> IResult<()> {
        if lane >= TMEM_ROWS || col >= TMEM_COLS {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM cell is out of bounds",
            ));
        }
        Ok(())
    }

    /// Store one physical 32-bit TMEM cell without interpreting its bits.
    pub fn write_raw_cell(&mut self, lane: usize, col: usize, bits: u32) -> IResult<()> {
        self.check_cell(lane, col)?;
        self.data[[lane, col]] = bits;
        self.valid[[lane, col]] = true;
        Ok(())
    }

    /// Read one of the four 8-bit TCol elements packed into a physical cell.
    pub fn read_cell_byte(&self, lane: usize, col: usize, subbyte: u8) -> IResult<u8> {
        self.check_cell(lane, col)?;
        if subbyte >= 4 {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM cell byte index is out of bounds",
            ));
        }
        if !self.valid[[lane, col]] {
            return Err(InterpreterError::new(
                "missing_tmem_value",
                "TMEM scale cell is unwritten",
            ));
        }
        Ok(((self.data[[lane, col]] >> (8 * subbyte)) & 0xFF) as u8)
    }

    fn check_supported_dtype(dtype: DType) -> IResult<()> {
        if !matches!(
            dtype,
            DType::F16 | DType::Bf16 | DType::F32 | DType::I32 | DType::U32
        ) {
            return Err(InterpreterError::new(
                "tmem_value",
                "unsupported TMEM cell dtype",
            ));
        }
        Ok(())
    }

    /// Arbitrary (lane,col) gather. Fails closed on unwritten or unsupported dtype.
    pub fn read_cells(
        &self,
        dtype: DType,
        lanes: &[usize],
        cols: &[usize],
    ) -> IResult<ValueArray1> {
        Self::check_supported_dtype(dtype)?;
        if lanes.len() != cols.len() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM cell gather index count mismatch",
            ));
        }
        let n = lanes.len();
        match dtype {
            DType::F32 => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let (l, c) = (lanes[i], cols[i]);
                    self.check_cell(l, c)?;
                    if !self.valid[[l, c]] {
                        return Err(InterpreterError::new(
                            "missing_tmem_value",
                            "TMEM cell is unwritten",
                        ));
                    }
                    out.push(f32::from_bits(self.data[[l, c]]));
                }
                Ok(ValueArray1::F32(Array1::from(out)))
            }
            DType::I32 => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let (l, c) = (lanes[i], cols[i]);
                    self.check_cell(l, c)?;
                    if !self.valid[[l, c]] {
                        return Err(InterpreterError::new(
                            "missing_tmem_value",
                            "TMEM cell is unwritten",
                        ));
                    }
                    out.push(self.data[[l, c]] as i32);
                }
                Ok(ValueArray1::I32(Array1::from(out)))
            }
            DType::U32 => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let (l, c) = (lanes[i], cols[i]);
                    self.check_cell(l, c)?;
                    if !self.valid[[l, c]] {
                        return Err(InterpreterError::new(
                            "missing_tmem_value",
                            "TMEM cell is unwritten",
                        ));
                    }
                    out.push(self.data[[l, c]]);
                }
                Ok(ValueArray1::U32(Array1::from(out)))
            }
            DType::F16 | DType::Bf16 => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let (l, c) = (lanes[i], cols[i]);
                    self.check_cell(l, c)?;
                    if !self.valid[[l, c]] {
                        return Err(InterpreterError::new(
                            "missing_tmem_value",
                            "TMEM cell is unwritten",
                        ));
                    }
                    out.push(decode_half(dtype, (self.data[[l, c]] & 0xffff) as u16));
                }
                Ok(ValueArray1::from_f32_compute(Array1::from(out), dtype))
            }
            _ => unreachable!(),
        }
    }

    /// Inverse scatter. Total (caller pre-validates bounds/uniqueness).
    pub fn write_cells(
        &mut self,
        dtype: DType,
        lanes: &[usize],
        cols: &[usize],
        values: &ValueArray1,
    ) -> IResult<()> {
        Self::check_supported_dtype(dtype)?;
        if lanes.len() != cols.len() || lanes.len() != values.len() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM cell scatter value count mismatch",
            ));
        }
        if values.dtype() != dtype {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM write dtype must match cell dtype",
            ));
        }
        let half_values = if is_packed_half_dtype(dtype) {
            Some(values.to_f32_compute())
        } else {
            None
        };
        for i in 0..lanes.len() {
            let (l, c) = (lanes[i], cols[i]);
            self.check_cell(l, c)?;
            self.data[[l, c]] = match dtype {
                DType::F32 => match values {
                    ValueArray1::F32(a) => a[i].to_bits(),
                    _ => unreachable!(),
                },
                DType::I32 => match values {
                    ValueArray1::I32(a) => a[i] as u32,
                    _ => unreachable!(),
                },
                DType::U32 => match values {
                    ValueArray1::U32(a) => a[i],
                    _ => unreachable!(),
                },
                DType::F16 | DType::Bf16 => {
                    let old = if self.valid[[l, c]] {
                        self.data[[l, c]]
                    } else {
                        0
                    };
                    (old & 0xffff_0000)
                        | u32::from(encode_half(dtype, half_values.as_ref().unwrap()[i]))
                }
                _ => unreachable!(),
            };
            self.valid[[l, c]] = true;
        }
        Ok(())
    }

    /// Rectangular block read [l0:l1, c0:c1] → row-major logical values.
    pub fn read_cell_block(
        &self,
        dtype: DType,
        l0: usize,
        l1: usize,
        c0: usize,
        c1: usize,
    ) -> IResult<ValueArray1> {
        let mut lanes = Vec::with_capacity((l1 - l0) * (c1 - c0));
        let mut cols = Vec::with_capacity((l1 - l0) * (c1 - c0));
        for l in l0..l1 {
            for c in c0..c1 {
                lanes.push(l);
                cols.push(c);
            }
        }
        self.read_cells(dtype, &lanes, &cols)
    }

    /// Rectangular block write (values lane-major). Total.
    pub fn write_cell_block(
        &mut self,
        dtype: DType,
        l0: usize,
        l1: usize,
        c0: usize,
        c1: usize,
        values: &ValueArray1,
    ) -> IResult<()> {
        let mut lanes = Vec::with_capacity((l1 - l0) * (c1 - c0));
        let mut cols = Vec::with_capacity((l1 - l0) * (c1 - c0));
        for l in l0..l1 {
            for c in c0..c1 {
                lanes.push(l);
                cols.push(c);
            }
        }
        self.write_cells(dtype, &lanes, &cols, values)
    }

    pub fn clear_columns(&mut self, col_start: usize, n_cols: usize) {
        let end = (col_start + n_cols).min(TMEM_COLS);
        self.data.slice_mut(ndarray::s![.., col_start..end]).fill(0);
        self.valid
            .slice_mut(ndarray::s![.., col_start..end])
            .fill(false);
    }

    pub fn accumulate_f32_cell_block_from(
        &mut self,
        dtype: DType,
        lane_start: usize,
        rows: usize,
        col_start: usize,
        cols: usize,
        src: &Array2<f32>,
        src_row: usize,
        src_col: usize,
        accum: bool,
    ) -> IResult<bool> {
        if dtype != DType::F32 {
            return Ok(false);
        }
        Self::check_supported_dtype(dtype)?;
        if lane_start + rows > TMEM_ROWS || col_start + cols > TMEM_COLS {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM cell is out of bounds",
            ));
        }
        if src_row + rows > src.nrows() || src_col + cols > src.ncols() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM source block is out of bounds",
            ));
        }
        if accum {
            for lane in lane_start..lane_start + rows {
                for col in col_start..col_start + cols {
                    if !self.valid[[lane, col]] {
                        return Err(InterpreterError::new(
                            "missing_tmem_value",
                            "TMEM cell is unwritten",
                        ));
                    }
                }
            }
        }
        {
            let grid = self.data_as_f32_mut();
            for i in 0..rows {
                let dst_base = (lane_start + i) * TMEM_COLS + col_start;
                for j in 0..cols {
                    let dst = &mut grid[dst_base + j];
                    let value = src[[src_row + i, src_col + j]];
                    if accum {
                        *dst += value;
                    } else {
                        *dst = value;
                    }
                }
            }
        }
        self.valid
            .slice_mut(ndarray::s![
                lane_start..lane_start + rows,
                col_start..col_start + cols
            ])
            .fill(true);
        Ok(true)
    }

    /// Gather-scatter f32 accumulator update at explicit physical TMEM cells.
    pub fn accumulate_f32_cells(
        &mut self,
        lanes: &[usize],
        cols: &[usize],
        values: &[f32],
        accum: bool,
    ) -> IResult<()> {
        if lanes.len() != cols.len() || lanes.len() != values.len() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM accumulator scatter value count mismatch",
            ));
        }
        for (&lane, &col) in lanes.iter().zip(cols) {
            self.check_cell(lane, col)?;
            if accum && !self.valid[[lane, col]] {
                return Err(InterpreterError::new(
                    "missing_tmem_value",
                    "TMEM cell is unwritten",
                ));
            }
        }
        for ((&lane, &col), &value) in lanes.iter().zip(cols).zip(values) {
            let next = if accum {
                f32::from_bits(self.data[[lane, col]]) + value
            } else {
                value
            };
            self.data[[lane, col]] = next.to_bits();
            self.valid[[lane, col]] = true;
        }
        Ok(())
    }

    pub fn write_packed_half_cells(
        &mut self,
        dtype: DType,
        lanes: &[usize],
        cols: &[usize],
        values: &[(f32, f32)],
    ) -> IResult<()> {
        Self::check_supported_dtype(dtype)?;
        if !is_packed_half_dtype(dtype) {
            return Err(InterpreterError::new(
                "tmem_value",
                "packed TMEM cell write requires f16 or bf16 cell dtype",
            ));
        }
        if lanes.len() != cols.len() || lanes.len() != values.len() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM packed cell scatter value count mismatch",
            ));
        }
        for i in 0..lanes.len() {
            let (l, c) = (lanes[i], cols[i]);
            self.check_cell(l, c)?;
            let lo = u32::from(encode_half(dtype, values[i].0));
            let hi = u32::from(encode_half(dtype, values[i].1));
            self.data[[l, c]] = lo | (hi << 16);
            self.valid[[l, c]] = true;
        }
        Ok(())
    }

    pub fn read_packed_half_cells(
        &self,
        dtype: DType,
        lanes: &[usize],
        cols: &[usize],
    ) -> IResult<Vec<(f32, f32)>> {
        Self::check_supported_dtype(dtype)?;
        if !is_packed_half_dtype(dtype) {
            return Err(InterpreterError::new(
                "tmem_value",
                "packed TMEM cell read requires f16 or bf16 cell dtype",
            ));
        }
        if lanes.len() != cols.len() {
            return Err(InterpreterError::new(
                "tmem_value",
                "TMEM packed cell gather index count mismatch",
            ));
        }
        let mut out = Vec::with_capacity(lanes.len());
        for i in 0..lanes.len() {
            let (l, c) = (lanes[i], cols[i]);
            self.check_cell(l, c)?;
            if !self.valid[[l, c]] {
                return Err(InterpreterError::new(
                    "missing_tmem_value",
                    "TMEM cell is unwritten",
                ));
            }
            let cell = self.data[[l, c]];
            out.push((
                decode_half(dtype, (cell & 0xffff) as u16),
                decode_half(dtype, (cell >> 16) as u16),
            ));
        }
        Ok(out)
    }
}

#[derive(Clone, Debug, Default)]
pub struct TmemValues {
    pub by_cta: HashMap<usize, TmemScratchpad>,
}

impl TmemValues {
    pub fn scratchpad_for(&self, cta_id: usize) -> IResult<&TmemScratchpad> {
        self.by_cta.get(&cta_id).ok_or_else(|| {
            InterpreterError::new(
                "missing_tmem_scratchpad",
                "CTA TMEM scratchpad is not allocated",
            )
        })
    }
}

fn is_packed_half_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::Bf16)
}

fn encode_half(dtype: DType, value: f32) -> u16 {
    match dtype {
        DType::F16 => f16::from_f32(value).to_bits(),
        DType::Bf16 => (round_bf16_scalar(value).to_bits() >> 16) as u16,
        _ => unreachable!("encode_half on non-half dtype"),
    }
}

fn decode_half(dtype: DType, bits: u16) -> f32 {
    match dtype {
        DType::F16 => f16::from_bits(bits).to_f32(),
        DType::Bf16 => f32::from_bits(u32::from(bits) << 16),
        _ => unreachable!("decode_half on non-half dtype"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tmem_cells_reinterpret_f32_i32_and_u32_bits() {
        let mut sp = TmemScratchpad::default();

        sp.write_cells(
            DType::U32,
            &[7],
            &[11],
            &ValueArray1::U32(Array1::from(vec![0x3f80_0000])),
        )
        .unwrap();
        assert_eq!(
            sp.read_cells(DType::F32, &[7], &[11])
                .unwrap()
                .to_f32_compute()
                .as_slice()
                .unwrap(),
            &[1.0]
        );

        sp.write_cells(
            DType::I32,
            &[8],
            &[12],
            &ValueArray1::I32(Array1::from(vec![-1])),
        )
        .unwrap();
        assert_eq!(
            sp.read_cells(DType::U32, &[8], &[12])
                .unwrap()
                .to_i64_compute()
                .as_slice()
                .unwrap(),
            &[0xffff_ffff_i64]
        );
    }

    #[test]
    fn tmem_valid_mask_clear_and_packed_half_values() {
        let mut sp = TmemScratchpad::default();

        let missing = sp.read_cells(DType::F32, &[0], &[0]).unwrap_err();
        assert_eq!(missing.code, "missing_tmem_value");

        sp.write_cells(
            DType::F32,
            &[0],
            &[0],
            &ValueArray1::F32(Array1::from(vec![2.0])),
        )
        .unwrap();
        assert!(sp.read_cells(DType::F32, &[0], &[0]).is_ok());
        sp.clear_columns(0, 1);
        assert_eq!(
            sp.read_cells(DType::F32, &[0], &[0]).unwrap_err().code,
            "missing_tmem_value"
        );

        // Two f16 values packed low-then-high into one cell.
        sp.write_packed_half_cells(DType::F16, &[0], &[0], &[(1.0, 2.0)])
            .unwrap();
        assert_eq!(sp.data[[0, 0]], 0x4000_3c00);
        assert_eq!(
            sp.read_packed_half_cells(DType::F16, &[0], &[0]).unwrap(),
            &[(1.0, 2.0)]
        );

        sp.write_packed_half_cells(DType::F16, &[1], &[3], &[(3.0, 4.0)])
            .unwrap();
        assert_eq!(
            sp.read_packed_half_cells(DType::F16, &[1], &[3]).unwrap(),
            &[(3.0, 4.0)]
        );
    }

    #[test]
    fn f32_block_accumulate_updates_rect_and_valid_mask() {
        let mut sp = TmemScratchpad::default();
        let src = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(sp
            .accumulate_f32_cell_block_from(DType::F32, 10, 2, 20, 3, &src, 0, 0, false)
            .unwrap());

        let read = sp
            .read_cells(
                DType::F32,
                &[10, 10, 10, 11, 11, 11],
                &[20, 21, 22, 20, 21, 22],
            )
            .unwrap();
        assert_eq!(
            read.to_f32_compute().as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        assert!(sp
            .accumulate_f32_cell_block_from(DType::F32, 10, 2, 20, 3, &src, 0, 0, true)
            .unwrap());
        let read = sp
            .read_cells(
                DType::F32,
                &[10, 10, 10, 11, 11, 11],
                &[20, 21, 22, 20, 21, 22],
            )
            .unwrap();
        assert_eq!(
            read.to_f32_compute().as_slice().unwrap(),
            &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
        );
    }
}
