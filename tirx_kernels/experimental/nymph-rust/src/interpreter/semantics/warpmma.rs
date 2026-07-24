//! Warp-level SM80 tensor-core MMA: `mma.sync.aligned.m{M}n{N}k{K}.row.col.f32.bf16.bf16.f32`.
//!
//! Computes `D[m,n] = Σ_k A[m,k]·B[n,k] + C[m,n]` over a full warp (32 lanes),
//! with the operands distributed across the warp's registers in the standard
//! mma fragment layout — the SAME layout nymph's `ldmatrix`/`stmatrix` model
//! (`values/ldstmatrix.rs::element_coord`), so an ldmatrix-loaded operand feeds
//! `mma.sync` by construction.
//!
//! Fragment layouts (groupID g = lane/4, threadID t = lane%4):
//!   A (M×K, bf16): `m*k/64` u32 regs/lane (packed b16×2). reg ru → tile
//!     (mt=ru%2, kt=ru/2); the two b16 halves h∈{0,1} hold
//!     A[mt*8 + g][kt*8 + 2t + h].
//!   B (N×K, bf16): `n*k/64` u32 regs/lane. reg ru → k-tile (kt=ru); halves hold
//!     B[2t + h][kt*8 + g]   (the `.col` / transposed operand).
//!   C,D (M×N, f32): `m*n/32` f32 regs/lane. reg ri holds
//!     [(ri/2)*8 + g][2t + ri%2].

use super::super::diagnostics::{IResult, InterpreterError};
use super::super::outcomes::StepStatus;
use super::super::registry::{StmtExecutorRegistry, StmtKind};
use super::super::values::arrays::ValueArray2;
use super::super::values::ldstmatrix::unpack_b16x2;
use super::super::warp_context::WarpContext;
use crate::ir::{DType, Stmt};
use half::f16;
use ndarray::Array2;
use std::collections::BTreeMap;

pub fn register(reg: &mut StmtExecutorRegistry) {
    reg.register(StmtKind::WarpMma, execute_warp_mma);
}

/// Decode one 16-bit A/B operand element to f32 per the mma's `.bf16`/`.f16` type.
#[inline]
fn decode_half(bits: u16, ab_dtype: DType) -> f32 {
    match ab_dtype {
        DType::F16 => f16::from_bits(bits).to_f32(),
        _ => f32::from_bits((bits as u32) << 16), // bf16: value is the high half
    }
}

/// Encode an f32 (already dtype-rounded) to its 16-bit A/B operand bits.
#[inline]
fn encode_half(x: f32, ab_dtype: DType) -> u16 {
    match ab_dtype {
        DType::F16 => f16::from_f32(x).to_bits(),
        _ => (x.to_bits() >> 16) as u16,
    }
}

/// A/B fragments may be either `num` u32 packed-16bit words (ldmatrix output, raw bits
/// of `ab_dtype`) or a bf16/f16 fragment of `2*num` elements whose consecutive pairs ARE
/// the words (the f32→16bit cvt path, e.g. a previous mma's accumulator reused as operand).
fn ab_words(arr: ValueArray2, ab_dtype: DType, label: &str) -> IResult<Array2<u32>> {
    let pack = |lo: u16, hi: u16| (lo as u32) | ((hi as u32) << 16);
    let pack_reg = |a: Array2<f32>| {
        Array2::from_shape_fn((a.nrows(), a.ncols() / 2), |(t, w)| {
            pack(
                encode_half(a[[t, 2 * w]], ab_dtype),
                encode_half(a[[t, 2 * w + 1]], ab_dtype),
            )
        })
    };
    match arr {
        ValueArray2::U32(a) => Ok(a),
        ValueArray2::I32(a) => Ok(a.mapv(|x| x as u32)),
        ValueArray2::Bf16(a) | ValueArray2::F16(a) => Ok(pack_reg(a)),
        _ => Err(InterpreterError::new(
            "warp_mma_dtype",
            format!("mma.sync {label} must be u32/i32 packed words or a bf16/f16 fragment"),
        )),
    }
}

fn execute_warp_mma<'a, 'k>(ctx: &mut WarpContext<'a, 'k>, stmt: &'k Stmt) -> IResult<StepStatus> {
    let (d, a, b, c, m, n, kk, ab_dtype) = match stmt {
        Stmt::WarpMma {
            d,
            a,
            b,
            c,
            m,
            n,
            k,
            ab_dtype,
        } => (d, a, b, c, *m as usize, *n as usize, *k as usize, *ab_dtype),
        _ => unreachable!(),
    };
    ctx.check_full_warp(
        "warp_mma_mask",
        "mma.sync must be issued by one or more full warps",
    )?;
    let d_r = ctx.eval_slice(d)?;
    let a_r = ctx.eval_slice(a)?;
    let b_r = ctx.eval_slice(b)?;
    let c_r = ctx.eval_slice(c)?;

    if ctx.trace_mode() {
        ctx.emit_tensor_read(&a_r)?;
        ctx.emit_tensor_read(&b_r)?;
        ctx.emit_tensor_read(&c_r)?;
        ctx.emit_tensor_write(&d_r)?;
        return Ok(StepStatus::advance());
    }

    let len_a = m * kk / 64; // u32 regs per lane (packed bf16)
    let len_b = n * kk / 64;
    let len_cd = m * n / 32; // f32 regs per lane

    let a_w = ab_words(ctx.registers_read(&a_r)?, ab_dtype, "A")?;
    let b_w = ab_words(ctx.registers_read(&b_r)?, ab_dtype, "B")?;
    let c_f = ctx.registers_read(&c_r)?.to_f32_compute();
    let nthreads = ctx.lanes.len();
    if a_w.ncols() != len_a || b_w.ncols() != len_b || c_f.ncols() != len_cd {
        return Err(InterpreterError::new(
            "warp_mma_shape",
            format!(
                "mma.sync m{m}n{n}k{kk} fragment lengths: A={} (want {len_a} u32), B={} (want {len_b} u32), C={} (want {len_cd} f32)",
                a_w.ncols(), b_w.ncols(), c_f.ncols()
            ),
        ));
    }
    if nthreads % 32 != 0 {
        return Err(InterpreterError::new(
            "warp_mma_mask",
            "mma.sync must execute on whole warps",
        ));
    }

    // group the lanes by warp: warp_id -> [lane -> index]
    let mut warps: BTreeMap<usize, [Option<usize>; 32]> = BTreeMap::new();
    for (ai, t) in ctx.lanes.iter().enumerate() {
        warps.entry(t.warp_id).or_insert([None; 32])[t.lane_id] = Some(ai);
    }

    let mut d_out = Array2::<f32>::zeros((nthreads, len_cd));
    for lanes in warps.values() {
        let mut amat = vec![0f32; m * kk];
        let mut bmat = vec![0f32; n * kk];
        let mut cmat = vec![0f32; m * n];
        for (lane, slot) in lanes.iter().enumerate() {
            let ai = slot.ok_or_else(|| {
                InterpreterError::new("warp_mma_mask", "mma.sync must be issued by full warps")
            })?;
            let g = lane / 4;
            let t = lane % 4;
            for ru in 0..len_a {
                let (mt, kt) = (ru % 2, ru / 2);
                let halves = unpack_b16x2(a_w[[ai, ru]]);
                for (h, &bits) in halves.iter().enumerate() {
                    amat[(mt * 8 + g) * kk + (kt * 8 + 2 * t + h)] = decode_half(bits, ab_dtype);
                }
            }
            for ru in 0..len_b {
                let kt = ru;
                let halves = unpack_b16x2(b_w[[ai, ru]]);
                for (h, &bits) in halves.iter().enumerate() {
                    bmat[(2 * t + h) * kk + (kt * 8 + g)] = decode_half(bits, ab_dtype);
                }
            }
            for ri in 0..len_cd {
                cmat[((ri / 2) * 8 + g) * n + (2 * t + ri % 2)] = c_f[[ai, ri]];
            }
        }
        // D = A · Bᵀ + C
        for (lane, slot) in lanes.iter().enumerate() {
            let ai = slot.unwrap();
            let g = lane / 4;
            let t = lane % 4;
            for ri in 0..len_cd {
                let mm = (ri / 2) * 8 + g;
                let nn = 2 * t + ri % 2;
                let mut acc = cmat[mm * n + nn];
                for kx in 0..kk {
                    acc += amat[mm * kk + kx] * bmat[nn * kk + kx];
                }
                d_out[[ai, ri]] = acc;
            }
        }
    }
    ctx.registers_write(&d_r, &ValueArray2::F32(d_out))?;
    Ok(StepStatus::advance())
}
