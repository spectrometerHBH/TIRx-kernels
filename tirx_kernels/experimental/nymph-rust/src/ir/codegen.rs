//! Rust codegen: lower a nymph `ir::Kernel` to a TVMScript (`tvm.script.tirx`)
//! source string. First working draft, scoped to ONE kernel — the bootstrap
//! cta_group=2 fp16 GEMM (M=256, N=128, K=64). The emitted source is a
//! `@T.prim_func def main(...)` whose forms mirror
//! `test_gemm_async.py::test_gemm_tcgen05_cta_group_2`, which compiles via
//! `tvm.compile(..., tir_pipeline="tirx")` and runs on B200.
//!
//! The nymph cohort model leaves per-thread guards implicit; TIRx needs them, so
//! this pass inserts `if lane_id == 0:` (lane 0 of the issuing warp) around the
//! issue ops that are single-thread (mbarrier init/arrive, TMA, MMA, commit). See
//! the per-op map in `emit_stmt`.
//!
//! Three further codegen passes recover the full-operand / correct-thread forms
//! TIRx requires from the nymph IR's instruction-granularity ops (see
//! `collapse_body` and `emit_body`):
//!   1. MMA-run collapse: fold the k=16-sliced `Tcgen05Mma` run into one full-K
//!      `gemm_async` (a 16-wide slice breaks the 128B swizzle atom -> illegal).
//!   2. Epilogue collapse: fold the 8-col `tcgen05.ld`/cvt/store run into one
//!      full-width `wg.copy_async` + `wg.cast` + `Tx.copy`.
//!   3. Peer-wait skip: a `try_wait` on a `map_shared_rank` (DSMEM) peer mbarrier
//!      is illegal and unnecessary (cta_group=2 TMA + cluster_sync already order
//!      the peer load); the canonical template has no peer wait either.
//! `cta_sync()` after a wait is emitted only at function scope (a CTA-wide
//! `__syncthreads` inside a single-warp/wg role would not be reached by all CTA
//! threads).

use super::dtype::{MemorySpace, ScalarOp, ScopeValueKind, Swizzle};
use super::kernel::Kernel;
use super::scalar::{ScalarExpr, ScalarValue, Var};
use super::stmt::Stmt;
use super::tensor::{Layout, Tensor, TensorSlice};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The imports the emitted source needs (prepended so the file is self-contained).
const HEADER_IMPORTS: &str = "\
import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import mma_shared_layout
from tvm.tirx.layout import S, TCol, TileLayout, TLane
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg
";

/// Per-kernel naming + lookup context built once, then read while walking the body.
struct Ctx {
    /// Tensor id -> emitted Python name.
    names: HashMap<u32, String>,
    /// mbar id -> emitted Python name of its `T.alloc_shared` buffer.
    mbar_names: HashMap<u32, String>,
    /// mbar id of the one mbar that has a peer (remote_coord) reference -> peer name.
    peer_names: HashMap<u32, String>,
    /// Loop var id -> emitted Python name.
    var_names: HashMap<u32, String>,
    /// cta_group for engine dispatch (TMA/MMA/commit), from the kernel cluster size.
    cta_group: u8,
    /// Base TMEM tensor (the largest-col allocation); the accum view maps onto it.
    tmem_base: Option<Arc<Tensor>>,
    /// `n_cols` passed to `tcgen05.alloc` — the column count the `tmem` view must
    /// match (the view must not exceed the allocation, else illegal tcgen05 access).
    tmem_alloc_cols: Option<u32>,
    /// Per-REG-tensor declared width AFTER the epilogue collapse widens the band.
    /// The fragment is declared `T.alloc_local(8)` in the nymph IR (instruction
    /// granularity); the collapsed wide read/cast/store needs it sized to the full
    /// column band. id -> width.
    reg_widths: HashMap<u32, usize>,
    /// REG-fragment `.view(...)` aliases already emitted (declared once, reused in
    /// the unrolled epilogue — TVMScript rejects redeclaring a name in one scope).
    declared_views: RefCell<HashSet<String>>,
}

impl Ctx {
    fn tensor_name(&self, id: u32) -> Result<&str, String> {
        self.names
            .get(&id)
            .map(|s| s.as_str())
            .ok_or_else(|| format!("codegen: no name for tensor id {id}"))
    }
}

/// dtype -> TVMScript dtype string.
fn dtype_str(dtype: super::dtype::DType) -> &'static str {
    use super::dtype::DType::*;
    match dtype {
        Bool => "bool",
        I8 => "int8",
        U8 => "uint8",
        I16 => "int16",
        U16 => "uint16",
        I32 => "int32",
        U32 => "uint32",
        I64 => "int64",
        U64 => "uint64",
        F8E4M3 => "float8_e4m3fn",
        F16 => "float16",
        Bf16 => "bfloat16",
        F32 => "float32",
    }
}

/// `Swizzle` -> mma_shared_layout swizzle mode (B128 -> 3).
fn swizzle_mode(sw: Swizzle) -> u8 {
    match sw {
        Swizzle::None => 0,
        Swizzle::B32 => 1,
        Swizzle::B64 => 2,
        Swizzle::B128 => 3,
    }
}

/// Indent helper.
fn pad(indent: usize) -> String {
    "    ".repeat(indent)
}

/// Entry point: lower a kernel to a TVMScript source string.
pub fn kernel_to_tirx_source(k: &Kernel) -> Result<String, String> {
    let ctx = build_ctx(k)?;
    let mut out = String::new();

    out.push_str(HEADER_IMPORTS);
    out.push('\n');

    // Argument tensors A/B/C by position.
    if k.args.len() != 3 {
        return Err(format!(
            "codegen: bootstrap expects 3 args (A,B,C), got {}",
            k.args.len()
        ));
    }
    let arg_names = ["A", "B", "C"];

    // SMEM tensor layout helper vars (mma_shared_layout(...)) — declared above the
    // prim_func so the parser sees plain Python values.
    for t in collect_tensors(k) {
        if t.space == MemorySpace::Smem {
            if let Some(Layout::Swizzle(sw)) = &t.layout {
                let name = ctx.tensor_name(t.id)?;
                out.push_str(&format!(
                    "{name}_layout = mma_shared_layout(\"{dt}\", {mode}, ({d0}, {d1}))\n",
                    name = name,
                    dt = dtype_str(t.dtype),
                    mode = swizzle_mode(sw.swizzle),
                    d0 = t.shape[0],
                    d1 = t.shape[1],
                ));
            }
        }
    }
    out.push('\n');

    // ---- prim_func header ----
    out.push_str("@T.prim_func\n");
    out.push_str("def main(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:\n");
    let ind = 1;
    for (i, t) in k.args.iter().enumerate() {
        let dims = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "{p}{name} = T.match_buffer({name}_ptr, ({dims}), \"{dt}\")\n",
            p = pad(ind),
            name = arg_names[i],
            dims = dims,
            dt = dtype_str(t.dtype),
        ));
    }
    out.push('\n');

    let num_warps = k.num_warps;
    let num_wg = num_warps / 4;
    out.push_str(&format!("{p}T.device_entry()\n", p = pad(ind)));
    out.push_str(&format!(
        "{p}warp_id = T.warp_id([{n}])\n",
        p = pad(ind),
        n = num_warps
    ));
    out.push_str(&format!(
        "{p}cbx, cby = T.cta_id_in_cluster([2, 1])\n",
        p = pad(ind)
    ));
    out.push_str(&format!("{p}cta_id = T.cta_id([2])\n", p = pad(ind)));
    out.push_str(&format!(
        "{p}wg_id = T.warpgroup_id([{n}])\n",
        p = pad(ind),
        n = num_wg
    ));
    out.push_str(&format!(
        "{p}tid_in_wg = T.thread_id_in_wg([128])\n",
        p = pad(ind)
    ));
    // Lane within the warp. Single-issue async ops (mbarrier / TMA / MMA / commit)
    // run inside a single-warp role, so the per-thread guard must be lane 0 of that
    // warp (`lane_id == 0`) — NOT `tid_in_wg == 0`, which is thread 0 of the whole
    // *warpgroup* and is never true for a warp whose lanes map to tid_in_wg 32..63
    // (e.g. warp 5 in warpgroup 1), so the issue would never fire and the MMA would
    // deadlock. Mirrors the canonical kernels' `lane_id == 0` / `elect_sync` guard.
    out.push_str(&format!("{p}lane_id = T.lane_id([32])\n", p = pad(ind)));
    out.push('\n');

    // ---- SMEM buffers ----
    for t in collect_tensors(k) {
        if t.space == MemorySpace::Smem {
            let name = ctx.tensor_name(t.id)?;
            out.push_str(&format!(
                "{p}{name} = T.alloc_buffer(({d0}, {d1}), \"{dt}\", scope=\"shared\", layout={name}_layout)\n",
                p = pad(ind),
                name = name,
                d0 = t.shape[0],
                d1 = t.shape[1],
                dt = dtype_str(t.dtype),
            ));
        }
    }

    // ---- mbar shared buffers + tmem_addr ----
    let mut have_peer = false;
    for s in &k.body {
        if let Stmt::MBarDef { mbar } = s {
            let name = ctx
                .mbar_names
                .get(&mbar.id)
                .ok_or_else(|| format!("codegen: no name for mbar {}", mbar.id))?;
            out.push_str(&format!(
                "{p}{name} = T.alloc_shared([1], \"uint64\")\n",
                p = pad(ind),
                name = name
            ));
        }
    }
    out.push_str(&format!(
        "{p}tmem_addr = T.alloc_shared([1], \"uint32\")\n",
        p = pad(ind)
    ));

    // peer mbar (remote_coord) decl — find the referenced mbar id.
    for (mbar_id, peer_name) in &ctx.peer_names {
        let base = ctx
            .mbar_names
            .get(mbar_id)
            .ok_or_else(|| format!("codegen: peer references unknown mbar {mbar_id}"))?;
        // The `T.let[...]` annotation makes `peer_ptr` a typed Var (a bare
        // `T.reinterpret(...)` returns a PrimExpr, which `decl_buffer(data=)` rejects).
        out.push_str(&format!(
            "{p}peer_ptr: T.let[T.Var(name=\"peer_ptr\", dtype=PointerType(PrimType(\"uint64\")))] = T.reinterpret(\"handle\", T.ptx.map_shared_rank({base}.ptr_to([0]), 1))\n",
            p = pad(ind),
            base = base,
        ));
        out.push_str(&format!(
            "{p}{peer} = T.decl_buffer([1], \"uint64\", data=peer_ptr, scope=\"shared\")\n",
            p = pad(ind),
            peer = peer_name,
        ));
        have_peer = true;
    }
    let _ = have_peer;

    // ---- REG fragments (epilogue) via T.alloc_local ----
    for t in collect_tensors(k) {
        if t.space == MemorySpace::Reg {
            let name = ctx.tensor_name(t.id)?;
            // Size the fragment to the collapsed epilogue band width (falls back to
            // the IR-declared shape when no collapse touched it).
            let n = ctx
                .reg_widths
                .get(&t.id)
                .copied()
                .filter(|w| *w > 0)
                .unwrap_or_else(|| t.shape.first().copied().unwrap_or(0));
            out.push_str(&format!(
                "{p}{name} = T.alloc_local({n}, dtype=\"{dt}\")\n",
                p = pad(ind),
                name = name,
                n = n,
                dt = dtype_str(t.dtype),
            ));
        }
    }
    out.push('\n');

    // ---- walk the body ----
    emit_body(&mut out, &k.body, ind, &ctx, false)?;

    Ok(out)
}

/// Collect every tensor referenced anywhere (args + defs + slices), deduped by id,
/// in a deterministic (id-sorted) order.
fn collect_tensors(k: &Kernel) -> Vec<Arc<Tensor>> {
    let mut map: HashMap<u32, Arc<Tensor>> = HashMap::new();
    for t in &k.args {
        map.entry(t.id).or_insert_with(|| t.clone());
    }
    fn walk(stmts: &[Stmt], map: &mut HashMap<u32, Arc<Tensor>>) {
        for s in stmts {
            collect_from_stmt(s, map);
            for body in s.child_bodies() {
                walk(body, map);
            }
        }
    }
    walk(&k.body, &mut map);
    let mut v: Vec<_> = map.into_values().collect();
    v.sort_by_key(|t| t.id);
    v
}

fn note_tensor(t: &Arc<Tensor>, map: &mut HashMap<u32, Arc<Tensor>>) {
    map.entry(t.id).or_insert_with(|| t.clone());
}
fn note_slice(s: &TensorSlice, map: &mut HashMap<u32, Arc<Tensor>>) {
    note_tensor(&s.tensor, map);
}

fn collect_from_stmt(s: &Stmt, map: &mut HashMap<u32, Arc<Tensor>>) {
    use Stmt::*;
    match s {
        TensorDef { tensor } | TmemAlloc { tensor, .. } | TmemDealloc { tensor, .. } => {
            note_tensor(tensor, map)
        }
        TmaLoad { dst, src, .. } => {
            note_slice(dst, map);
            note_tensor(src, map);
        }
        TmaStore { dst, src, .. } => {
            note_tensor(dst, map);
            note_slice(src, map);
        }
        Tcgen05Mma { dst, a, b, .. } => {
            note_slice(dst, map);
            note_slice(a, map);
            note_slice(b, map);
        }
        Tcgen05Ld { dst, src, .. } => {
            note_slice(dst, map);
            note_tensor(src, map);
        }
        Tcgen05St { dst, src, .. } => {
            note_tensor(dst, map);
            note_slice(src, map);
        }
        RegCvt { dst, src, .. } | RegLoad { dst, src } | RegStore { dst, src } => {
            note_slice(dst, map);
            note_slice(src, map);
        }
        _ => {}
    }
}

/// Extract a literal integer from a scalar value, if it is one.
fn as_int(sv: &ScalarValue) -> Option<i64> {
    match sv {
        ScalarValue::Int(i) => Some(*i),
        _ => None,
    }
}

/// Collapse codegen-level instruction-granularity runs into the single coalesced
/// op TIRx wants. The nymph IR emits ops at hardware-instruction granularity (k=16
/// MMA sub-slices, 8-col TMEM reads) because that's the value-model's unit; TIRx's
/// `gemm_async` / `wg.copy_async` take the *full* operand and tile internally. A
/// 16-wide (32-byte) sub-slice of a 128B-swizzle atom, or an 8-col tcgen05.ld,
/// hits `cudaErrorIllegalInstruction`. These passes recover the full-operand form.
///
/// Applied to every statement list (a body) before emission. General over K and
/// over the column width — not hardcoded to the bootstrap's K=64 / N=128.
fn collapse_body(stmts: &[Stmt]) -> Vec<Stmt> {
    let mut out: Vec<Stmt> = Vec::with_capacity(stmts.len());
    let mut i = 0;
    while i < stmts.len() {
        if let Some((collapsed, consumed)) = try_collapse_mma_run(&stmts[i..]) {
            out.push(collapsed);
            i += consumed;
            continue;
        }
        if let Some((collapsed, consumed)) = try_collapse_epilogue_run(&stmts[i..]) {
            out.extend(collapsed);
            i += consumed;
            continue;
        }
        out.push(stmts[i].clone());
        i += 1;
    }
    out
}

/// Suspect 1: a run of `Tcgen05Mma` with identical dst, same A/B tensors, the K
/// (last) dim advancing by `k` each step, and accum = (False, True, True, …).
/// Collapse to ONE `gemm_async` over the full coalesced K range with the first
/// op's accum. Returns (collapsed_stmt, num_consumed) or None.
fn try_collapse_mma_run(stmts: &[Stmt]) -> Option<(Stmt, usize)> {
    let Stmt::Tcgen05Mma {
        dst,
        a,
        b,
        m,
        n,
        k,
        accum,
        trans_a,
        trans_b,
        cta_group,
        sfa,
        sfb,
        sf_byte,
    } = &stmts[0]
    else {
        return None;
    };
    // Only collapse the simple (non block-scaled) GEMM run.
    if sfa.is_some() || sfb.is_some() {
        return None;
    }
    // K is the last operand dim. Record the first op's K offset/extent.
    let a_kdim = a.offsets.len().checked_sub(1)?;
    let b_kdim = b.offsets.len().checked_sub(1)?;
    let a_k0 = as_int(&a.offsets[a_kdim])?;
    let b_k0 = as_int(&b.offsets[b_kdim])?;
    let a_kext = as_int(&a.shape[a_kdim])?;
    let b_kext = as_int(&b.shape[b_kdim])?;

    let mut count = 1usize;
    let mut a_khi = a_k0 + a_kext;
    let mut b_khi = b_k0 + b_kext;
    let mut total_k = *k as i64;

    for s in &stmts[1..] {
        let Stmt::Tcgen05Mma {
            dst: d2,
            a: a2,
            b: b2,
            accum: accum2,
            sfa: sfa2,
            sfb: sfb2,
            ..
        } = s
        else {
            break;
        };
        if sfa2.is_some() || sfb2.is_some() {
            break;
        }
        // Same dst (accumulator), same A/B tensors, accum True (continuation).
        if d2 != dst || !Arc::ptr_eq(&a2.tensor, &a.tensor) || !Arc::ptr_eq(&b2.tensor, &b.tensor) {
            break;
        }
        if !*accum2 {
            break;
        }
        // K must advance contiguously from the previous hi.
        let (Some(a2k0), Some(b2k0)) = (as_int(&a2.offsets[a_kdim]), as_int(&b2.offsets[b_kdim]))
        else {
            break;
        };
        let (Some(a2ke), Some(b2ke)) = (as_int(&a2.shape[a_kdim]), as_int(&b2.shape[b_kdim]))
        else {
            break;
        };
        if a2k0 != a_khi || b2k0 != b_khi {
            break;
        }
        // All non-K dims of A/B must match the first op (same tile).
        if a2.offsets[..a_kdim] != a.offsets[..a_kdim]
            || a2.shape[..a_kdim] != a.shape[..a_kdim]
            || b2.offsets[..b_kdim] != b.offsets[..b_kdim]
            || b2.shape[..b_kdim] != b.shape[..b_kdim]
        {
            break;
        }
        a_khi = a2k0 + a2ke;
        b_khi = b2k0 + b2ke;
        total_k += a2ke; // K extent of this step
        count += 1;
    }

    if count < 2 {
        return None; // nothing to collapse — emit as-is
    }

    // Build the coalesced operands: K offset = first's, K extent = full span.
    let mut a_full = a.clone();
    let mut b_full = b.clone();
    a_full.shape[a_kdim] = ScalarValue::Int(a_khi - a_k0);
    b_full.shape[b_kdim] = ScalarValue::Int(b_khi - b_k0);

    let collapsed = Stmt::Tcgen05Mma {
        dst: dst.clone(),
        a: a_full,
        b: b_full,
        m: *m,
        n: *n,
        k: total_k as u32,
        accum: *accum,
        trans_a: *trans_a,
        trans_b: *trans_b,
        cta_group: *cta_group,
        sfa: None,
        sfb: None,
        sf_byte: *sf_byte,
    };
    Some((collapsed, count))
}

/// Suspect 2: a run of `(Tcgen05Ld, Tcgen05WaitLd, RegCvt, RegStore)` quadruples
/// where each ld reads consecutive `num`-wide column bands of the same TMEM accum
/// into the same fragment, casts, and stores to consecutive C columns. Collapse to
/// ONE wide quadruple over the full column band (mirrors the template's single
/// full-width `wg.copy_async` + `wait.ld` + `wg.cast` + `Tx.copy`). Returns
/// (collapsed_stmts, num_consumed) or None.
fn try_collapse_epilogue_run(stmts: &[Stmt]) -> Option<(Vec<Stmt>, usize)> {
    // Identify the quadruple shape from the first 4 stmts.
    let quad = |s: &[Stmt]| -> Option<(u32, i64, i64)> {
        // Returns (ld_num, ld_col, store_col0).
        if s.len() < 4 {
            return None;
        }
        let Stmt::Tcgen05Ld { num, col, .. } = &s[0] else {
            return None;
        };
        let Stmt::Tcgen05WaitLd = &s[1] else {
            return None;
        };
        let Stmt::RegCvt { .. } = &s[2] else {
            return None;
        };
        let Stmt::RegStore { dst, .. } = &s[3] else {
            return None;
        };
        let col_i = as_int(col)?;
        let store_col0 = as_int(dst.offsets.get(1)?)?;
        Some((*num, col_i, store_col0))
    };

    let first = quad(stmts)?;
    let step = first.0 as i64;

    // Walk consecutive quadruples; require contiguous, equal-width column advance.
    let mut count = 1usize;
    let mut next_ld_col = first.1 + step;
    let mut next_store_col = first.2 + step;
    let mut idx = 4;
    while idx + 4 <= stmts.len() {
        if let Some((num, ld_col, store_col0)) = quad(&stmts[idx..]) {
            if num as i64 == step && ld_col == next_ld_col && store_col0 == next_store_col {
                count += 1;
                next_ld_col += step;
                next_store_col += step;
                idx += 4;
                continue;
            }
        }
        break;
    }

    if count < 2 {
        return None;
    }

    let total_width = (step * count as i64) as u32;
    // Clone the first quadruple and widen it to the full band.
    let mut ld = stmts[0].clone();
    let mut cvt = stmts[2].clone();
    let mut store = stmts[3].clone();

    if let Stmt::Tcgen05Ld { dst, num, .. } = &mut ld {
        *num = total_width;
        // Widen the dst reg fragment slice to the full width.
        if let Some(sh) = dst.shape.first_mut() {
            *sh = ScalarValue::Int(total_width as i64);
        }
    }
    if let Stmt::RegCvt { dst, src, .. } = &mut cvt {
        if let Some(sh) = dst.shape.first_mut() {
            *sh = ScalarValue::Int(total_width as i64);
        }
        if let Some(sh) = src.shape.first_mut() {
            *sh = ScalarValue::Int(total_width as i64);
        }
    }
    if let Stmt::RegStore { dst, src } = &mut store {
        // Store dst: widen the column extent to the full band.
        if dst.shape.len() == 2 {
            dst.shape[1] = ScalarValue::Int(total_width as i64);
        }
        if let Some(sh) = src.shape.first_mut() {
            *sh = ScalarValue::Int(total_width as i64);
        }
    }

    Some((vec![ld, Stmt::Tcgen05WaitLd, cvt, store], count * 4))
}

/// Build the naming context: A/B/C for args by position; SMEM/TMEM/REG/mbar by role.
fn build_ctx(k: &Kernel) -> Result<Ctx, String> {
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut tensors: HashMap<u32, Arc<Tensor>> = HashMap::new();
    let arg_names = ["A", "B", "C"];
    for (i, t) in k.args.iter().enumerate() {
        if i < arg_names.len() {
            names.insert(t.id, arg_names[i].to_string());
        }
        tensors.insert(t.id, t.clone());
    }

    // SMEM/TMEM/REG names, deterministic by id order. The bootstrap uses exactly:
    // 2 SMEM (A_smem, B_smem), 2 TMEM (tmem base + accum view), 2 REG fragments.
    let mut smem_idx = 0usize;
    let mut tmem_idx = 0usize;
    let mut reg_idx = 0usize;
    let smem_names = ["A_smem", "B_smem"];
    for t in collect_tensors(k) {
        tensors.entry(t.id).or_insert_with(|| t.clone());
        if names.contains_key(&t.id) {
            continue;
        }
        let name = match t.space {
            MemorySpace::Smem => {
                let n = smem_names
                    .get(smem_idx)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("smem{smem_idx}"));
                smem_idx += 1;
                n
            }
            MemorySpace::Tmem => {
                // First TMEM tensor is the base allocation; the second is the accum
                // view at col_start 0. Name base "tmem"; views are emitted as
                // tmem[:, lo:hi] at the slice site, but the tensor still needs a name
                // (only used if referenced as a whole tensor, e.g. tmem_alloc).
                let n = if tmem_idx == 0 {
                    "tmem".to_string()
                } else {
                    format!("tmem_view{tmem_idx}")
                };
                tmem_idx += 1;
                n
            }
            MemorySpace::Reg => {
                let n = match reg_idx {
                    0 => "accum_frag".to_string(),
                    1 => "out_frag".to_string(),
                    _ => format!("reg{reg_idx}"),
                };
                reg_idx += 1;
                n
            }
            MemorySpace::Gmem => continue, // args already named
        };
        names.insert(t.id, name);
    }

    // mbar names + peer names.
    let mut mbar_names: HashMap<u32, String> = HashMap::new();
    let mut peer_names: HashMap<u32, String> = HashMap::new();
    let mut mbar_idx = 0usize;
    let mbar_default = ["smem_full", "mma_done"];
    fn walk_mbars(
        stmts: &[Stmt],
        mbar_names: &mut HashMap<u32, String>,
        peer_names: &mut HashMap<u32, String>,
        mbar_idx: &mut usize,
        mbar_default: &[&str],
    ) {
        for s in stmts {
            if let Stmt::MBarDef { mbar } = s {
                if !mbar_names.contains_key(&mbar.id) {
                    let n = mbar_default
                        .get(*mbar_idx)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("mbar{}", *mbar_idx));
                    mbar_names.insert(mbar.id, n);
                    *mbar_idx += 1;
                }
            }
            // discover peer references
            for mref in stmt_mbar_refs(s) {
                if mref.remote_coord.is_some() {
                    peer_names
                        .entry(mref.mbar.id)
                        .or_insert_with(|| format!("peer_{}", mref.mbar.id));
                }
            }
            for body in s.child_bodies() {
                walk_mbars(body, mbar_names, peer_names, mbar_idx, mbar_default);
            }
        }
    }
    walk_mbars(
        &k.body,
        &mut mbar_names,
        &mut peer_names,
        &mut mbar_idx,
        &mbar_default,
    );
    // Give peers stable readable names derived from their base mbar name.
    let peer_names: HashMap<u32, String> = peer_names
        .into_keys()
        .map(|id| {
            let base = mbar_names.get(&id).cloned().unwrap_or_default();
            (id, format!("peer_{base}"))
        })
        .collect();

    // cta_group from the cluster size (the bootstrap is cta_group=2).
    let cta_group = k.cluster_shape.iter().product::<usize>().max(1) as u8;

    // The base TMEM tensor = the TMEM allocation with the most columns (512); the
    // accum (128,N) view aliases it at col 0.
    let tmem_base = tensors
        .values()
        .filter(|t| t.space == MemorySpace::Tmem)
        .max_by_key(|t| t.shape.get(1).copied().unwrap_or(0))
        .cloned();

    // The TMEM alloc n_cols (the view must match it, not the base tensor's cols).
    let mut tmem_alloc_cols = None;
    fn find_alloc_cols(stmts: &[Stmt], out: &mut Option<u32>) {
        for s in stmts {
            if let Stmt::TmemAlloc { n_cols, .. } = s {
                out.get_or_insert(*n_cols);
            }
            for body in s.child_bodies() {
                find_alloc_cols(body, out);
            }
        }
    }
    find_alloc_cols(&k.body, &mut tmem_alloc_cols);

    // Per-REG-tensor width after the epilogue collapse. Walk the collapsed bodies
    // and record, for each REG fragment, the widest slice any op uses on it.
    let mut reg_widths: HashMap<u32, usize> = HashMap::new();
    fn note_reg_width(s: &TensorSlice, widths: &mut HashMap<u32, usize>) {
        if s.tensor.space == MemorySpace::Reg {
            let w = s.shape.first().and_then(as_int).unwrap_or(0).max(0) as usize;
            let e = widths.entry(s.tensor.id).or_insert(0);
            *e = (*e).max(w);
        }
    }
    fn walk_reg_widths(stmts: &[Stmt], widths: &mut HashMap<u32, usize>) {
        for s in collapse_body(stmts) {
            match &s {
                Stmt::Tcgen05Ld { dst, num, .. } => {
                    if dst.tensor.space == MemorySpace::Reg {
                        let e = widths.entry(dst.tensor.id).or_insert(0);
                        *e = (*e).max(*num as usize);
                    }
                }
                Stmt::RegCvt { dst, src, .. } => {
                    note_reg_width(dst, widths);
                    note_reg_width(src, widths);
                }
                Stmt::RegStore { src, .. } => note_reg_width(src, widths),
                _ => {}
            }
            for body in s.child_bodies() {
                walk_reg_widths(body, widths);
            }
        }
    }
    walk_reg_widths(&k.body, &mut reg_widths);

    Ok(Ctx {
        names,
        mbar_names,
        peer_names,
        var_names: HashMap::new(),
        cta_group,
        tmem_base,
        tmem_alloc_cols,
        reg_widths,
        declared_views: RefCell::new(HashSet::new()),
    })
}

/// Emit a `name_view = name.view(128, width, layout=...)` once; subsequent calls
/// with the same name are no-ops (the alias is reused).
fn ensure_view(out: &mut String, p: &str, name: &str, width: usize, ctx: &Ctx) {
    let view = format!("{name}_view");
    if ctx.declared_views.borrow().contains(&view) {
        return;
    }
    out.push_str(&format!(
        "{p}{view} = {name}.view(128, {width}, layout=TileLayout(S[(128, {width}) : (1 @ axis_tid_in_wg, 1)]))\n"
    ));
    ctx.declared_views.borrow_mut().insert(view);
}

/// Every MBarRef a statement names (for peer discovery).
fn stmt_mbar_refs(s: &Stmt) -> Vec<&super::mbar::MBarRef> {
    use Stmt::*;
    match s {
        MBarrierInit { mbar, .. }
        | MBarrierArrive { mbar, .. }
        | MBarrierWait { mbar, .. }
        | MBarrierExpectTx { mbar, .. }
        | MBarrierArriveExpectTx { mbar, .. }
        | Tcgen05Commit { mbar, .. } => vec![mbar],
        TmaLoad { mbar, .. } => vec![mbar],
        _ => vec![],
    }
}

// ===========================================================================
// scalar / slice sub-printers
// ===========================================================================

/// Precedence levels (lower binds looser). Used to parenthesize correctly.
fn op_prec(op: ScalarOp) -> u8 {
    use ScalarOp::*;
    match op {
        Or => 1,
        And => 2,
        Eq | Ne | Lt | Le | Gt | Ge => 3,
        Add | Sub => 4,
        Mul | FloorDiv | Mod => 5,
        Neg | Not => 6,
        _ => 4,
    }
}

fn scope_name(kind: ScopeValueKind) -> &'static str {
    use ScopeValueKind::*;
    match kind {
        TidInWg => "tid_in_wg",
        LaneId => "lane_id",
        WarpId => "warp_id",
        WarpgroupId => "wg_id",
        CtaidInCluster => "cbx",
        CtaId => "cta_id",
        NvshmemMyPe => "nvshmem_my_pe",
    }
}

fn var_name(ctx: &Ctx, v: &Var) -> String {
    ctx.var_names
        .get(&v.id.0)
        .cloned()
        .unwrap_or_else(|| format!("v{}", v.id.0))
}

/// Emit a scalar value as a Python expression, parenthesizing per precedence.
fn emit_scalar(sv: &ScalarValue, ctx: &Ctx) -> String {
    emit_scalar_prec(sv, ctx, 0)
}

fn emit_scalar_prec(sv: &ScalarValue, ctx: &Ctx, parent_prec: u8) -> String {
    match sv {
        ScalarValue::Int(i) => i.to_string(),
        ScalarValue::Var(v) => var_name(ctx, v),
        ScalarValue::Scope(k) => scope_name(*k).to_string(),
        ScalarValue::Expr(e) => emit_expr(e, ctx, parent_prec),
    }
}

fn binop_symbol(op: ScalarOp) -> Option<&'static str> {
    use ScalarOp::*;
    Some(match op {
        Add => "+",
        Sub => "-",
        Mul => "*",
        FloorDiv => "//",
        Mod => "%",
        And => "and",
        Or => "or",
        Eq => "==",
        Ne => "!=",
        Lt => "<",
        Le => "<=",
        Gt => ">",
        Ge => ">=",
        _ => return None,
    })
}

fn emit_expr(e: &ScalarExpr, ctx: &Ctx, parent_prec: u8) -> String {
    let prec = op_prec(e.op);
    let s = match e.op {
        ScalarOp::Neg => format!("-{}", emit_scalar_prec(&e.args[0], ctx, prec)),
        ScalarOp::Not => format!("not {}", emit_scalar_prec(&e.args[0], ctx, prec)),
        ScalarOp::Select => format!(
            "({} if {} else {})",
            emit_scalar_prec(&e.args[1], ctx, 0),
            emit_scalar_prec(&e.args[0], ctx, 0),
            emit_scalar_prec(&e.args[2], ctx, 0),
        ),
        ScalarOp::Min => format!(
            "T.min({}, {})",
            emit_scalar_prec(&e.args[0], ctx, 0),
            emit_scalar_prec(&e.args[1], ctx, 0)
        ),
        ScalarOp::Max => format!(
            "T.max({}, {})",
            emit_scalar_prec(&e.args[0], ctx, 0),
            emit_scalar_prec(&e.args[1], ctx, 0)
        ),
        _ => {
            if let Some(sym) = binop_symbol(e.op) {
                format!(
                    "{} {} {}",
                    emit_scalar_prec(&e.args[0], ctx, prec),
                    sym,
                    emit_scalar_prec(&e.args[1], ctx, prec + 1)
                )
            } else {
                format!("<unsupported op {:?}>", e.op)
            }
        }
    };
    // Parenthesize if this binds looser than the parent context demands.
    let needs_paren = matches!(
        e.op,
        ScalarOp::Add
            | ScalarOp::Sub
            | ScalarOp::Mul
            | ScalarOp::FloorDiv
            | ScalarOp::Mod
            | ScalarOp::And
            | ScalarOp::Or
            | ScalarOp::Eq
            | ScalarOp::Ne
            | ScalarOp::Lt
            | ScalarOp::Le
            | ScalarOp::Gt
            | ScalarOp::Ge
    ) && prec < parent_prec;
    if needs_paren {
        format!("({s})")
    } else {
        s
    }
}

/// Fold `lo + extent` for the hi bound when both are literals; otherwise emit `lo + extent`.
fn add_bound(lo: &ScalarValue, extent: &ScalarValue, ctx: &Ctx) -> String {
    match (lo, extent) {
        (ScalarValue::Int(a), ScalarValue::Int(b)) => (a + b).to_string(),
        (ScalarValue::Int(0), _) => emit_scalar(extent, ctx),
        _ => format!(
            "{} + {}",
            emit_scalar_prec(lo, ctx, 4),
            emit_scalar_prec(extent, ctx, 5)
        ),
    }
}

/// Emit `Name[lo0:hi0, lo1:hi1, ...]` from a slice's offsets+shape.
/// `name_override` lets the TMEM accum view print as `tmem` (the base buffer).
fn emit_slice(s: &TensorSlice, ctx: &Ctx) -> Result<String, String> {
    emit_slice_named(s, ctx, ctx.tensor_name(s.tensor.id)?)
}

fn emit_slice_named(s: &TensorSlice, ctx: &Ctx, name: &str) -> Result<String, String> {
    let mut dims = Vec::new();
    for (off, ext) in s.offsets.iter().zip(s.shape.iter()) {
        let lo = emit_scalar(off, ctx);
        let hi = add_bound(off, ext, ctx);
        dims.push(format!("{lo}:{hi}"));
    }
    Ok(format!("{name}[{}]", dims.join(", ")))
}

/// A TMEM tensor's *view* (accum) maps to the base `tmem` buffer; emit `tmem[:, lo:hi]`.
/// We detect a TMEM slice whose tensor is the accum view (shape[0]==128, not the base
/// 512-col buffer) and rewrite it onto `tmem`.
fn emit_tmem_dst(s: &TensorSlice, ctx: &Ctx) -> Result<String, String> {
    // The MMA dst / ld src is `accum[:, 0:N]`; map to `tmem[:, 0:N]`.
    // Row dim spans the whole 128 lanes -> ":"; col dim from offsets/shape.
    let row = if s.offsets.len() == 2 {
        let lo = emit_scalar(&s.offsets[0], ctx);
        let hi = add_bound(&s.offsets[0], &s.shape[0], ctx);
        if lo == "0" {
            // whole-lane span
            ":".to_string()
        } else {
            format!("{lo}:{hi}")
        }
    } else {
        ":".to_string()
    };
    let (clo, chi) = if s.offsets.len() == 2 {
        (
            emit_scalar(&s.offsets[1], ctx),
            add_bound(&s.offsets[1], &s.shape[1], ctx),
        )
    } else {
        ("0".to_string(), "0".to_string())
    };
    Ok(format!("tmem[{row}, {clo}:{chi}]"))
}

// ===========================================================================
// statement walk
// ===========================================================================

/// Emit a statement list, applying the instruction-granularity collapses
/// (MMA-run, epilogue-run) first. Every body walk goes through here so the
/// transforms apply uniformly at any nesting depth.
///
/// Also coalesces the fence/sync that follows a run of consecutive `MBarrierWait`s:
/// the template issues all the `try_wait`s, then ONE
/// `tcgen05.fence.after_thread_sync()` + ONE `T.cuda.cta_sync()` — not a
/// fence per wait and no cta_sync (the nymph IR leaves those implicit). (suspect 3)
fn emit_body(
    out: &mut String,
    stmts: &[Stmt],
    indent: usize,
    ctx: &Ctx,
    in_role: bool,
) -> Result<(), String> {
    let collapsed = collapse_body(stmts);
    let p = pad(indent);
    let mut i = 0;
    while i < collapsed.len() {
        if matches!(collapsed[i], Stmt::MBarrierWait { .. }) {
            // Emit every `try_wait` in the run, then one fence. The
            // `T.cuda.cta_sync()` (a CTA-wide `__syncthreads`) is only emitted at
            // function scope: inside a single-warp / single-warpgroup role branch
            // not all CTA threads reach it, so the barrier would deadlock / raise an
            // illegal instruction. Within one role the threads are already lockstep
            // and the mbarrier wait + tcgen05 fence give the async-engine ordering.
            //
            // A *peer* wait (remote_coord set, i.e. a `try_wait` on a
            // `map_shared_rank`-remapped DSMEM address) is SKIPPED: `mbarrier.try_wait`
            // is only legal on a local shared address, so the remapped form raises
            // `cudaErrorIllegalInstruction`. It is also unnecessary — the cta_group=2
            // TMA load (one cluster-coordinated copy signalling the local mbarrier)
            // plus the cluster_sync already order the peer CTA's load before the MMA,
            // exactly as the canonical template does (no peer wait there either).
            let mut j = i;
            let mut emitted_any = false;
            while j < collapsed.len() && matches!(collapsed[j], Stmt::MBarrierWait { .. }) {
                if let Stmt::MBarrierWait { mbar, phase, .. } = &collapsed[j] {
                    if mbar.remote_coord.is_none() {
                        let name = mbar_buf_name(mbar, ctx)?;
                        let phase_s = phase
                            .as_ref()
                            .map(|ph| emit_scalar(ph, ctx))
                            .unwrap_or_else(|| "0".to_string());
                        out.push_str(&format!(
                            "{p}T.ptx.mbarrier.try_wait({name}.ptr_to([0]), {phase_s})\n"
                        ));
                        emitted_any = true;
                    }
                }
                j += 1;
            }
            if emitted_any {
                out.push_str(&format!("{p}T.ptx.tcgen05.fence.after_thread_sync()\n"));
                if !in_role {
                    out.push_str(&format!("{p}T.cuda.cta_sync()\n"));
                }
            }
            i = j;
            continue;
        }
        emit_stmt(out, &collapsed[i], indent, ctx, in_role)?;
        i += 1;
    }
    Ok(())
}

fn emit_stmt(
    out: &mut String,
    stmt: &Stmt,
    indent: usize,
    ctx: &Ctx,
    in_role: bool,
) -> Result<(), String> {
    use Stmt::*;
    let p = pad(indent);
    match stmt {
        // ---- definitions handled in the header; skip in the body walk ----
        TensorDef { .. } | MBarDef { .. } => Ok(()),

        // ---- TMEM alloc / dealloc (already under warp==0 role guard) ----
        TmemAlloc { n_cols, .. } => {
            // The TMEM view buffer `tmem` is declared at function scope (after the
            // KernelInit block), not here; this only emits the alloc call.
            out.push_str(&format!(
                "{p}T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols={n_cols}, cta_group={cg})\n",
                cg = ctx.cta_group,
            ));
            Ok(())
        }
        TmemDealloc { n_cols, .. } => {
            let cta_group = ctx.cta_group;
            out.push_str(&format!(
                "{p}T.ptx.tcgen05.relinquish_alloc_permit(cta_group={cta_group})\n"
            ));
            out.push_str(&format!(
                "{p}T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols={n_cols}, cta_group={cta_group})\n"
            ));
            Ok(())
        }

        // ---- structural ----
        KernelInit { body, warp, .. } => {
            if let Some(w) = warp {
                out.push_str(&format!("{p}if warp_id == {w}:\n"));
                emit_body(out, body, indent + 1, ctx, true)?;
            } else {
                emit_body(out, body, indent, ctx, in_role)?;
            }
            // Declare the TMEM view buffer at function scope (mirrors the template's
            // line 490: `tmem` is visible to the MMA + epilogue, so it must NOT be
            // nested under the warp==0 alloc guard).
            if let Some(base) = &ctx.tmem_base {
                // The view's column count must match the `tcgen05.alloc` n_cols, not
                // the base tensor's (512) — a view wider than the allocation is an
                // illegal tcgen05 access. (suspect 4)
                let cols = ctx
                    .tmem_alloc_cols
                    .map(|c| c as usize)
                    .unwrap_or(base.shape[1]);
                out.push_str(&format!(
                    "{p}tmem = T.decl_buffer(({d0}, {d1}), \"{dt}\", scope=\"tmem\", allocated_addr=tmem_addr[0], layout=TileLayout(S[({d0}, {d1}) : (1 @ TLane, 1 @ TCol)]))\n",
                    d0 = base.shape[0],
                    d1 = cols,
                    dt = dtype_str(base.dtype),
                ));
            }
            // After kernel_init we emit the init fence / sync sequence (one time).
            out.push_str(&format!("{p}T.ptx.fence.mbarrier_init()\n"));
            out.push_str(&format!("{p}T.ptx.fence.proxy_async(\"shared::cta\")\n"));
            out.push_str(&format!("{p}T.cuda.cta_sync()\n"));
            out.push_str(&format!("{p}T.cuda.cluster_sync()\n"));
            Ok(())
        }
        KernelFinalize { body, warp, .. } => {
            if let Some(w) = warp {
                out.push_str(&format!("{p}if warp_id == {w}:\n"));
                emit_body(out, body, indent + 1, ctx, true)?;
            } else {
                emit_body(out, body, indent, ctx, in_role)?;
            }
            Ok(())
        }
        Role {
            body,
            warp,
            warpgroup,
            ..
        } => {
            let guard = if let Some(w) = warp {
                format!("warp_id == {w}")
            } else if let Some(wg) = warpgroup {
                format!("wg_id == {wg}")
            } else {
                return Err("codegen: role without warp/warpgroup".to_string());
            };
            out.push_str(&format!("{p}if {guard}:\n"));
            // A role is a single-warp / single-warpgroup branch: suppress CTA-wide
            // cta_sync inside it (not all CTA threads arrive).
            emit_body(out, body, indent + 1, ctx, true)?;
            Ok(())
        }
        If { cond, then_body } => {
            out.push_str(&format!("{p}if {}:\n", emit_scalar(cond, ctx)));
            emit_body(out, then_body, indent + 1, ctx, in_role)?;
            Ok(())
        }
        ForLoop {
            var,
            start,
            stop,
            step,
            body,
        } => {
            let name = var_name(ctx, var);
            let start_s = emit_scalar(start, ctx);
            let stop_s = emit_scalar(stop, ctx);
            let step_s = emit_scalar(step, ctx);
            let range = if step_s == "1" {
                format!("range({start_s}, {stop_s})")
            } else {
                format!("range({start_s}, {stop_s}, {step_s})")
            };
            out.push_str(&format!("{p}for {name} in {range}:\n"));
            emit_body(out, body, indent + 1, ctx, in_role)?;
            Ok(())
        }

        // ---- mbarrier ----
        MBarrierInit { mbar, count, .. } => {
            let name = mbar_buf_name(mbar, ctx)?;
            out.push_str(&format!("{p}if lane_id == 0:\n"));
            out.push_str(&format!(
                "{p}    T.ptx.mbarrier.init({name}.ptr_to([0]), {count})\n"
            ));
            Ok(())
        }
        MBarrierArriveExpectTx { mbar, bytes, .. } => {
            let name = mbar_buf_name(mbar, ctx)?;
            out.push_str(&format!("{p}if lane_id == 0:\n"));
            out.push_str(&format!(
                "{p}    T.ptx.mbarrier.arrive.expect_tx({name}.ptr_to([0]), {bytes})\n"
            ));
            Ok(())
        }
        MBarrierExpectTx { mbar, bytes, .. } => {
            let name = mbar_buf_name(mbar, ctx)?;
            out.push_str(&format!("{p}if lane_id == 0:\n"));
            out.push_str(&format!(
                "{p}    T.ptx.mbarrier.expect_tx({name}.ptr_to([0]), {bytes})\n"
            ));
            Ok(())
        }
        MBarrierArrive { mbar, count, .. } => {
            let name = mbar_buf_name(mbar, ctx)?;
            out.push_str(&format!("{p}if lane_id == 0:\n"));
            out.push_str(&format!(
                "{p}    T.ptx.mbarrier.arrive({name}.ptr_to([0]), {})\n",
                emit_scalar(count, ctx)
            ));
            Ok(())
        }
        MBarrierWait { mbar, phase, .. } => {
            let name = mbar_buf_name(mbar, ctx)?;
            let phase_s = phase
                .as_ref()
                .map(|p| emit_scalar(p, ctx))
                .unwrap_or_else(|| "0".to_string());
            out.push_str(&format!(
                "{p}T.ptx.mbarrier.try_wait({name}.ptr_to([0]), {phase_s})\n"
            ));
            out.push_str(&format!("{p}T.ptx.tcgen05.fence.after_thread_sync()\n"));
            Ok(())
        }

        // ---- TMA ----
        TmaLoad {
            dst,
            src,
            mbar,
            coords,
            shape,
            ..
        } => {
            let mbar_name = mbar_buf_name(mbar, ctx)?;
            let dst_s = emit_slice(dst, ctx)?;
            let src_s = emit_gmem_region(src, coords, shape, ctx)?;
            out.push_str(&format!("{p}if lane_id == 0:\n"));
            out.push_str(&format!(
                "{p}    Tx.copy_async({dst_s}, {src_s}, dispatch=\"tma\", mbar={mbar_name}.ptr_to([0]), cta_group={cg})\n",
                cg = ctx.cta_group,
            ));
            Ok(())
        }

        // ---- tcgen05 MMA ----
        Tcgen05Mma {
            dst, a, b, accum, ..
        } => {
            let dst_s = emit_tmem_dst(dst, ctx)?;
            let a_s = emit_slice(a, ctx)?;
            let b_s = emit_slice(b, ctx)?;
            let accum_s = if *accum { "True" } else { "False" };
            out.push_str(&format!("{p}if lane_id == 0:\n"));
            out.push_str(&format!(
                "{p}    Tx.gemm_async({dst_s}, {a_s}, {b_s}, accum={accum_s}, dispatch=\"tcgen05\", cta_group={cg})\n",
                cg = ctx.cta_group,
            ));
            Ok(())
        }
        Tcgen05Commit {
            mbar,
            multicast_cta_mask,
            ..
        } => {
            let name = mbar_buf_name(mbar, ctx)?;
            let mask = multicast_cta_mask.unwrap_or(0);
            out.push_str(&format!("{p}if lane_id == 0:\n"));
            out.push_str(&format!(
                "{p}    T.ptx.tcgen05.commit({name}.ptr_to([0]), cta_group={cg}, cta_mask={mask})\n",
                cg = ctx.cta_group,
            ));
            Ok(())
        }

        // ---- epilogue: tcgen05_ld -> Tx.wg.copy_async, wait_ld, reg_cvt -> Tx.wg.cast,
        // reg_store -> Tx.copy. Whole warpgroup participates (no per-thread guard). ----
        Tcgen05Ld {
            dst, num, row, col, ..
        } => {
            // dst is the f32 reg fragment (shape (num,)); src is the tmem accum view.
            let frag = ctx.tensor_name(dst.tensor.id)?.to_string();
            let width = *num as usize;
            let _ = emit_scalar(row, ctx); // row is the lane axis, captured by the view
            let col_s = emit_scalar(col, ctx);
            // The wg-collective view: each thread's lane row, `num` cols at `col`.
            ensure_view(out, &p, &frag, width, ctx);
            out.push_str(&format!(
                "{p}Tx.wg.copy_async({frag}_view[:, :], tmem[:, {col_s}:{col_s} + {width}])\n"
            ));
            Ok(())
        }
        Tcgen05WaitLd => {
            out.push_str(&format!("{p}T.ptx.tcgen05.wait.ld()\n"));
            Ok(())
        }
        RegCvt { dst, src, .. } => {
            let dname = ctx.tensor_name(dst.tensor.id)?.to_string();
            let sname = ctx.tensor_name(src.tensor.id)?.to_string();
            // Width from the collapsed band (reg_widths), falling back to the slice.
            let width = ctx
                .reg_widths
                .get(&dst.tensor.id)
                .copied()
                .filter(|w| *w > 0)
                .unwrap_or_else(|| dst.shape.first().and_then(as_int).unwrap_or(8).max(0) as usize);
            ensure_view(out, &p, &dname, width, ctx);
            out.push_str(&format!(
                "{p}Tx.wg.cast({dname}_view[:, :], {sname}_view[:, :])\n"
            ));
            Ok(())
        }
        RegStore { dst, src } => {
            // GMEM store: C[row, col:col+w] = out_frag[:]
            let dst_s = emit_gmem_row_store(dst, ctx)?;
            let frag = ctx.tensor_name(src.tensor.id)?;
            out.push_str(&format!("{p}Tx.copy({dst_s}, {frag}[:])\n"));
            Ok(())
        }

        // ---- fence / sync ----
        Fence { .. } => Ok(()),
        CtaSync => {
            // Suppress CTA-wide cta_sync inside a single-warp/wg role branch (not
            // all CTA threads reach it → illegal __syncthreads).
            if !in_role {
                out.push_str(&format!("{p}T.cuda.cta_sync()\n"));
            }
            Ok(())
        }
        ClusterSync => {
            out.push_str(&format!("{p}T.cuda.cluster_sync()\n"));
            Ok(())
        }
        WarpSync => Ok(()),
        WgSync { .. } => Ok(()),

        other => Err(format!("codegen: unimplemented stmt {other:?}")),
    }
}

/// mbar buffer name, picking the peer name if remote_coord is set.
fn mbar_buf_name(mref: &super::mbar::MBarRef, ctx: &Ctx) -> Result<String, String> {
    if mref.remote_coord.is_some() {
        if let Some(n) = ctx.peer_names.get(&mref.mbar.id) {
            return Ok(n.clone());
        }
    }
    ctx.mbar_names
        .get(&mref.mbar.id)
        .cloned()
        .ok_or_else(|| format!("codegen: no name for mbar {}", mref.mbar.id))
}

/// Build the GMEM TMA region from src tensor + coords + per-tile shape.
/// Emits `A[lo0:hi0, lo1:hi1]`.
fn emit_gmem_region(
    src: &Arc<Tensor>,
    coords: &[ScalarValue],
    shape: &[usize],
    ctx: &Ctx,
) -> Result<String, String> {
    let name = ctx.tensor_name(src.id)?;
    let mut dims = Vec::new();
    for (coord, ext) in coords.iter().zip(shape.iter()) {
        let lo = emit_scalar(coord, ctx);
        let ext_sv = ScalarValue::Int(*ext as i64);
        let hi = add_bound(coord, &ext_sv, ctx);
        dims.push(format!("{lo}:{hi}"));
    }
    Ok(format!("{name}[{}]", dims.join(", ")))
}

/// reg_store dst -> `C[row, col:col+w]`. The dst slice is a 1-row region:
/// offsets = (row_expr, col), shape = (1, w).
fn emit_gmem_row_store(dst: &TensorSlice, ctx: &Ctx) -> Result<String, String> {
    let name = ctx.tensor_name(dst.tensor.id)?;
    if dst.offsets.len() != 2 {
        return Err("codegen: reg_store dst must be 2D".to_string());
    }
    let row = emit_scalar(&dst.offsets[0], ctx);
    let clo = emit_scalar(&dst.offsets[1], ctx);
    let chi = add_bound(&dst.offsets[1], &dst.shape[1], ctx);
    Ok(format!("{name}[{row}, {clo}:{chi}]"))
}
