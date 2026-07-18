//! Rust codegen: lower a nymph `ir::Kernel` to a TVMScript (`tvm.script.tirx`)
//! source string. Covers the GEMM datapath — the full pipelined cta_group=2
//! fp16/bf16 GEMM and the NVFP4 block-scaled GEMM (every bench shape) — plus the
//! generic scaffolding they exercise (roles, rings, mbars, scheduler loop, TMA,
//! tcgen05, epilogue). IR nodes with no lowering (the flash-attention set:
//! `WarpMma`, `RegUnary`, `RegFill`, GMEM semaphores, `TmaStore.reduce_add`, …)
//! and IR field values the emitted forms cannot represent (per-op `cta_group`
//! overrides, MMA `trans_a/trans_b/lane_align`, non-NVFP4 scale modes) fail
//! closed with `Err` — never a silent different-semantics emission. The emitted
//! source is a `@T.prim_func def main(...)` that compiles via
//! `tvm.compile(..., tir_pipeline="tirx")` and runs on B200.
//!
//! The nymph cohort model leaves per-thread guards implicit; TIRx needs them, so
//! this pass inserts a scope-derived single-issue guard (`Scope::issue_guard`:
//! `tid_in_wg == 0` in a warpgroup role, `elect_sync()` in a warp role) around
//! the issue ops that are single-thread (mbarrier init/arrive, TMA, MMA, commit).
//! See the per-op map in `emit_stmt`.
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

use super::dtype::{DType, MemorySpace, ScalarOp, ScopeValueKind, Swizzle, VarBinding};
use super::kernel::Kernel;
use super::scalar::{ScalarExpr, ScalarInitial, ScalarValue, Var};
use super::stmt::{RegOperand, Stmt};
use super::tensor::{Layout, MmaOperand, Tensor, TensorSlice, TmemOperand};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The imports the emitted source needs (prepended so the file is self-contained).
const HEADER_IMPORTS: &str = "\
import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import mma_shared_layout
from tvm.tirx.layout import R, S, TCol, TileLayout, TLane
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg
";

/// SF SMEM/GMEM physical layout, computed HERE from the fixed nvfp4 SF formula
/// (128-row super-blocks, epc=4) instead of importing TVM's `sf_smem_layout`.
/// Emits the same `TileLayout(S[...])` nymph already emits for the SF *TMEM*
/// side, so the codegen is self-contained — no external SF-layout helper.
///
/// Mirrors `sf_smem_layout(rows, sf_k, sf_per_mma, pipe_depth)`: the SF is read
/// in 128-row super-blocks of 32 lanes; a logical `(rows, sf_k)` tile decomposes
/// as `M_super x M_SF_INNER(4) x LANE(32) x K_outer x sf_per_mma x in_lane_K`,
/// size-1 dims dropped, optional pipe-depth outer prepended.
fn sf_smem_tile_layout(
    rows: usize,
    sf_k: usize,
    sf_per_mma: usize,
    pipe_depth: Option<usize>,
) -> String {
    const EPC: usize = 4;
    const M_SUPER_ROWS: usize = 128;
    const LANE: usize = 32;
    let m_sf_inner = M_SUPER_ROWS / LANE; // 4
    let in_lane_k = EPC / sf_per_mma; // 1 for nvfp4 (sf_per_mma=4)
    let k_outer = sf_k / EPC;
    let m_super = rows / M_SUPER_ROWS;
    let lane_bytes = EPC * m_sf_inner; // 16
    let super_bytes = lane_bytes * LANE; // 512
    let k_total_bytes = super_bytes * k_outer;
    let stage_bytes = k_total_bytes * m_super;

    let raw_shape = [m_super, m_sf_inner, LANE, k_outer, sf_per_mma, in_lane_k];
    let raw_strides = [k_total_bytes, EPC, lane_bytes, super_bytes, in_lane_k, 1];
    let mut shape: Vec<usize> = Vec::new();
    let mut strides: Vec<usize> = Vec::new();
    for (s, st) in raw_shape.iter().zip(raw_strides.iter()) {
        if *s != 1 {
            shape.push(*s);
            strides.push(*st);
        }
    }
    if let Some(p) = pipe_depth {
        shape.insert(0, p);
        strides.insert(0, stage_bytes);
    }
    let sh = shape
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let st = strides
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("TileLayout(S[({sh}) : ({st})])")
}

/// The thread scope of the enclosing role branch, threaded through the body walk.
/// Determines (a) whether a CTA-wide `cta_sync` may be emitted (only at function
/// scope, where all CTA threads converge) and (b) which single thread issues a
/// single-issue async op (`mbarrier`/`TMA`/`MMA`/`commit`).
///
/// The single-issue guard is the crux of the cross-CTA mailbox handshake: a warp
/// role elects `lane_id == 0` (exactly 1 thread of the 1-warp branch), but a
/// *warpgroup* role spans 4 warps, so `lane_id == 0` is true for 4 threads (lane 0
/// of each warp) — a single-issue `mbarrier.arrive` issued under it would arrive
/// FOUR times, over-counting the barrier. A warpgroup role must elect
/// `tid_in_wg == 0` (thread 0 of the whole 128-thread warpgroup) instead.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// Function scope (not inside any role branch). CTA-wide sync is legal here.
    Function,
    /// Inside a single-warp role branch (`if warp_id == w:`).
    Warp,
    /// Inside a single-warpgroup role branch (`if wg_id == w:`).
    Warpgroup,
    /// Inside an ELECTED role branch (`if warp_id == w: if elect_sync():` ...): the
    /// WHOLE role body already runs on one thread, so single-issue ops emit with NO
    /// further per-op guard (canon's `if elect_sync(): while ...:` scheduler/loader/MMA
    /// loops). Matches canon's thread model exactly — one issuing thread runs the loop,
    /// its mbar waits + the CLC handshake, rather than 32 threads with per-op guards.
    Elected,
}

impl Scope {
    /// True at function scope: all CTA threads converge, so `cta_sync` is legal.
    fn is_function(self) -> bool {
        self == Scope::Function
    }
    /// The predicate electing the single thread that issues a single-issue async op
    /// (one `mbarrier`/`TMA`/`MMA`/`commit`). One thread per warp inside a warp role;
    /// one thread per warpgroup inside a warpgroup role. At function scope (e.g. the
    /// kernel_init warp==0 block, which is a warp-granularity guard) one lane suffices.
    ///
    /// Warp / function scope elect ONE lane PER WARP — semantically `lane_id == 0`, but
    /// emitted as `T.ptx.elect_sync()`, the hardware `elect.sync` warp instruction (one
    /// elected lane per warp, the active leader). This is exactly the canonical kernels'
    /// guard (`if T.ptx.elect_sync():` around the loader / MMA single-issue loops). It
    /// avoids the `T.lane_id([32])` -> `threadIdx.x % 32 == 0` lowering: TVM inlines that
    /// `%32` at EVERY guard site (~64 in the 1024 prologue/loops), and the predicated
    /// branch also drives up CBU divergence. `elect.sync` is a single uniform instruction
    /// with no modulo and no per-site recompute.
    ///
    /// Warpgroup scope MUST stay `tid_in_wg == 0`: a warpgroup spans 4 warps, so
    /// `elect.sync` would elect one lane PER WARP = 4 issuing threads in the group and
    /// over-issue every single-issue op (4× mbarrier arrive -> corrupted phase / deadlock
    /// once a slot is reused). `tid_in_wg == 0` is the single warpgroup-wide thread 0.
    ///
    /// This split is purely SCOPE-driven (a generic property of the enclosing role's
    /// thread granularity), not keyed on any kernel/shape — every kernel's warp-scope
    /// single-issue guard lowers to `elect.sync`, every warpgroup-scope one to thread 0.
    fn issue_guard(self) -> &'static str {
        match self {
            Scope::Warpgroup => "tid_in_wg == 0",
            Scope::Warp | Scope::Function => "T.ptx.elect_sync()",
            // Already inside the role-wide elect; a single-issue op needs no extra guard.
            // (Returned for completeness; `emit_guarded` skips the guard for Elected.)
            Scope::Elected => "True",
        }
    }
}

/// Per-kernel naming + lookup context built once, then read while walking the body.
struct Ctx {
    /// Tensor id -> emitted Python name.
    names: HashMap<u32, String>,
    /// mbar id -> emitted Python name of its `T.alloc_shared` buffer.
    mbar_names: HashMap<u32, String>,
    /// mbar id of the one mbar that has a peer (remote_coord) reference -> peer name.
    peer_names: HashMap<u32, String>,
    /// mbar id -> declared stage count (a multi-stage mbar allocs `[stages]`).
    mbar_stages: HashMap<u32, u32>,
    /// Loop var id -> emitted Python name.
    var_names: HashMap<u32, String>,
    /// Scalar var id (binding == Scalar) -> emitted Python name. Scalar vars emit as SSA
    /// `T.int32` register vars (`NAME: T.int32 = init`; reassigned `NAME = expr`; read as
    /// `NAME`) — NOT `alloc_local(1)` cells — so ptxas keeps them in uniform registers.
    scalar_names: HashMap<u32, String>,
    /// cta_group for engine dispatch (TMA/MMA/commit), from the kernel cluster size.
    cta_group: u8,
    /// Column span of the single TMEM view buffer: the largest
    /// `base_col + n_cols` over the kernel's `TmemAlloc`s. The view is one
    /// `decl_buffer((128, cols), allocated_addr=0)` — TMEM is not a tensor, so
    /// every TMEM instruction references it by absolute column slice.
    tmem_view_cols: Option<usize>,
    /// Per-REG-tensor declared width AFTER the epilogue collapse widens the band.
    /// The fragment is declared `T.alloc_local(8)` in the nymph IR (instruction
    /// granularity); the collapsed wide read/cast/store needs it sized to the full
    /// column band. id -> width.
    reg_widths: HashMap<u32, usize>,
    /// mbar ids of the TMA-load completion barriers (`smem_full`, `sf_full`, ...) in a
    /// cta_group=2 cluster kernel — every TMA barrier carrying a peer reference. In
    /// cluster mode the canonical pattern routes BOTH CTAs' TMA completions to the
    /// LEADER CTA's barrier (a `map_shared_rank(.., 0)` view used uniformly by both
    /// CTAs): each CTA's `Tx.copy_async` signals it, the leader (cbx==0) issues one
    /// `arrive.expect_tx` for the full cluster byte count, and the leader's MMA waits
    /// its own LOCAL barrier (which both CTAs fill). This replaces the illegal peer
    /// `try_wait` AND is the prerequisite for multicast TMA loads (the per-destination
    /// transaction count of a `multicast::cluster` copy must land on the single leader
    /// barrier, accounted via the `* cta_group` factor in the leader expect_tx). Empty
    /// when not cluster mode / no cluster TMA barrier.
    tma_leader_mbars: std::collections::HashSet<u32>,
    /// Number of launched clusters (`launch_cta_count / cta_group`) — the grid stride
    /// for a `ForEachTask` grid-stride scheduler loop.
    num_clusters: usize,
    /// NVFP4 e4m3 scale-factor TMEM views (SFA_tmem, SFB_tmem), keyed by the
    /// operand's absolute physical base column; declared via `decl_buffer`
    /// right after the `tmem` view in KernelInit.
    sf_views: Vec<SfView>,
    /// Usage-derived scale-factor tensor ids (see `collect_sf_ids`) — the ONLY
    /// authority on "is this tensor a scale factor"; dtype is never consulted.
    sf: SfIds,
}

/// One NVFP4 e4m3 scale-factor TMEM view (`SFA_tmem`/`SFB_tmem`). The IR's SF
/// operands are absolute physical TMEM addresses; the view re-materializes
/// canon's logical `(rows, SF_K)` `decl_buffer` at that column so the
/// block-scaled `Tx.gemm_async`/`Tx.copy_async` emission is unchanged.
#[derive(Clone)]
struct SfView {
    name: String,
    /// Absolute physical base column of the SF band.
    col: usize,
    /// Logical rows: the scaled-row count rounded up to whole 128-lane
    /// super-blocks (a 256-row SFB band folds into 2 column super-blocks).
    logical_rows: usize,
    /// Logical cols: the per-row scale-block count (nvfp4 `k/16`).
    logical_cols: usize,
}

impl Ctx {
    /// The `map_shared_rank(.., 0)` (leader CTA-0) DSMEM view name for a TMA-load
    /// barrier, e.g. `smem_full_cta0`. Used by the cluster TMA load + expect_tx.
    fn tma_leader_view_for(&self, id: u32) -> Option<String> {
        if !self.tma_leader_mbars.contains(&id) {
            return None;
        }
        let base = self.mbar_names.get(&id)?;
        Some(format!("{base}_cta0"))
    }
    fn is_tma_leader_mbar(&self, id: u32) -> bool {
        self.tma_leader_mbars.contains(&id)
    }
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

/// Element byte width of a dtype.
fn dtype_bytes(dt: super::dtype::DType) -> usize {
    use super::dtype::DType::*;
    match dt {
        Bool | I8 | U8 | F8E4M3 => 1,
        I16 | U16 | F16 | Bf16 => 2,
        I32 | U32 | F32 => 4,
        I64 | U64 => 8,
    }
}

/// Integer dtype predicate (the I32 scheduler mailbox vs the f16/bf16 data rings).
fn is_int_dtype(dt: super::dtype::DType) -> bool {
    use super::dtype::DType::*;
    matches!(dt, I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64 | Bool)
}

/// Pick the MMA-shared swizzle atom matching a tile row's byte width — mirrors the
/// canonical kernel's `_swizzle_for_row_bytes`. Used for the D writeback ring, whose
/// layout the value model leaves implicit.
fn swizzle_for_row_bytes(row_bytes: usize) -> u8 {
    // The atom row must FIT in and DIVIDE the tile row (mirrors the canonical
    // `_suggest_swizzle_for_row_bytes`: `row_bytes >= atom && row_bytes % atom == 0`).
    if row_bytes >= 128 && row_bytes % 128 == 0 {
        3
    } else if row_bytes >= 64 && row_bytes % 64 == 0 {
        2
    } else if row_bytes >= 32 && row_bytes % 32 == 0 {
        1
    } else {
        0
    }
}

/// Indent helper.
fn pad(indent: usize) -> String {
    "    ".repeat(indent)
}

/// Entry point: lower a kernel to a TVMScript source string.
/// Positional arg name: A, B, C, …, Z (the whole alphabet), then `arg{i}` past it.
/// The first 8 (A–H) cover every existing kernel; the extension is id-derived so a
/// 9th+ argument names stably instead of falling off the table.
fn arg_name(i: usize) -> String {
    if i < 26 {
        return ((b'A' + i as u8) as char).to_string();
    }
    format!("arg{i}")
}

pub fn kernel_to_tirx_source(k: &Kernel) -> Result<String, String> {
    let ctx = build_ctx(k)?;
    let mut out = String::new();

    out.push_str(HEADER_IMPORTS);
    out.push('\n');

    // Argument tensors, named by position (A, B, C, D, …). The fp16/bootstrap GEMM
    // has 3 (A, B, C-out); the nvfp4 GEMM has 5 (A, B, SFA, SFB, D-out). Names are
    // cosmetic — TVM matches args positionally.
    if k.args.is_empty() {
        return Err("codegen: kernel has no args".to_string());
    }

    // SMEM tensor layout helper vars (mma_shared_layout(...)) — declared above the
    // prim_func so the parser sees plain Python values. Every f16/bf16 SMEM buffer
    // (A/B operand rings AND D writeback rings) gets an MMA-compatible swizzled
    // layout. The D ring carries no explicit layout in the IR (the value model is
    // layout-agnostic), so we synthesize the swizzle from the row byte width — the
    // canonical kernel's `_swizzle_for_row_bytes(EPI_N * elem_bytes)`. The plain I32
    // mailbox (task_smem) takes no layout (a flat row-major buffer).
    for t in collect_tensors(k) {
        // Layout-less SMEM = the flat i32/u32 scheduler mailbox + mbar buffers. Packed
        // fp4 operands (u8) DO get a swizzle, and e4m3 SF buffers get sf_smem_layout.
        if t.space != MemorySpace::Smem || (is_int_dtype(t.dtype) && t.dtype != DType::U8) {
            continue;
        }
        let name = ctx.tensor_name(t.id)?;
        let shape_tuple = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        // SF-usage SMEM (a tcgen05.cp source ring) → canon's sf_smem_layout(rows,
        // sf_k, sf_per_mma=4, pipe_depth). The TMA lands the bytes in cp-ready
        // order. Classified by USAGE, not dtype — a plain fp8 DATA ring is also
        // e4m3 but must take the normal MMA swizzle below.
        if ctx.sf.smem.contains(&t.id) {
            if t.dtype != DType::F8E4M3 {
                return Err(format!(
                    "codegen: SF SMEM tensor {name} must be e4m3 (got {:?})",
                    t.dtype
                ));
            }
            let (rows, sf_k, pipe) = if t.shape.len() == 3 {
                (t.shape[1], t.shape[2], Some(t.shape[0]))
            } else {
                (t.shape[0], t.shape[1], None)
            };
            let layout = sf_smem_tile_layout(rows, sf_k, 4, pipe);
            out.push_str(&format!("{name}_layout = {layout}\n"));
            continue;
        }
        // The swizzle atom row (128/64/32 B for mode 3/2/1) must DIVIDE the tile row
        // width, else `mma_shared_layout` tiles a too-wide atom over a narrower row and
        // `Layout.tile_to` lowers to a `floormod`-by-zero ("Divide by zero"). The IR's
        // A/B operand rings carry a fixed `Swizzle::B128`, but a small `blk_k` (e.g.
        // blk_k=32 f16 -> 64-byte rows) cannot host the 128-byte atom. So clamp the
        // requested swizzle DOWN to the largest atom the row width supports — never
        // upgrade past the IR's intent. (The D ring carries no layout; its mode comes
        // straight from the row byte width, same helper.)
        let row_bytes = t.shape[t.shape.len() - 1] * dtype_bytes(t.dtype);
        let row_mode = swizzle_for_row_bytes(row_bytes);
        let mode = match &t.layout {
            Some(Layout::Swizzle(sw)) => swizzle_mode(sw.swizzle).min(row_mode),
            _ => row_mode,
        };
        out.push_str(&format!(
            "{name}_layout = mma_shared_layout(\"{dt}\", {mode}, ({shape_tuple}))\n",
            name = name,
            dt = dtype_str(t.dtype),
            mode = mode,
            shape_tuple = shape_tuple,
        ));
    }
    out.push('\n');

    // ---- prim_func header ----
    out.push_str("@T.prim_func\n");
    let sig = k
        .args
        .iter()
        .enumerate()
        .map(|(i, _)| format!("{}_ptr: T.handle", arg_name(i)))
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!("def main({sig}) -> None:\n"));
    let ind = 1;
    for (i, t) in k.args.iter().enumerate() {
        let dims = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        // SF-usage GMEM args (the TMA sources of an SF SMEM ring): lay them out with
        // canon's sf_smem_layout(rows, sf_k, sf_per_mma=4) so the TMA reads cp-ready
        // bytes. Usage-derived — a plain fp8 GMEM data arg keeps its natural layout.
        let layout = if ctx.sf.gmem.contains(&t.id) && t.shape.len() == 2 {
            format!(
                ", layout={}",
                sf_smem_tile_layout(t.shape[0], t.shape[1], 4, None)
            )
        } else {
            String::new()
        };
        out.push_str(&format!(
            "{p}{name} = T.match_buffer({name}_ptr, ({dims}), \"{dt}\"{layout})\n",
            p = pad(ind),
            name = arg_name(i),
            dims = dims,
            dt = dtype_str(t.dtype),
        ));
    }
    out.push('\n');

    let num_warps = k.num_warps;
    let num_wg = num_warps / 4;
    out.push_str(&format!("{p}T.device_entry()\n", p = pad(ind)));
    // INVARIANT I1a: persistent grid, 1 CTA/SM. Without this the compiler may
    // place 2 CTAs/SM, the software-pipelined steady state drifts under the warp
    // scheduler, and the mbarrier pipeline deadlocks at full GPU occupancy.
    out.push_str(&format!(
        "{p}T.attr({{\"tirx.launch_bounds_min_blocks_per_sm\": 1}})\n",
        p = pad(ind)
    ));
    out.push_str(&format!(
        "{p}warp_id = T.warp_id([{n}])\n",
        p = pad(ind),
        n = num_warps
    ));
    out.push_str(&format!(
        "{p}cbx, cby = T.cta_id_in_cluster([2, 1], preferred=[2, 1])\n",
        p = pad(ind)
    ));
    // The kernel→cta axis extent is the TOTAL launched CTA count, NOT the cluster
    // group size. The persistent grid launches `num_clusters * cta_group` CTAs; a
    // hardcoded `[2]` declares only 2 CTAs, so a multi-cluster launch (e.g. 4 CTAs)
    // gives `cta_id` (the cluster→task index source, `cta_id // cta_group`) the wrong
    // range — only cluster 0 gets a valid task, every other cluster's tile is never
    // computed (output left untouched). Use the real launch count so each cluster
    // pulls its own grid-stride task share.
    let launch_ctas = k.launch_cta_count().max(ctx.cta_group as usize);
    out.push_str(&format!(
        "{p}cta_id = T.cta_id([{n}])\n",
        p = pad(ind),
        n = launch_ctas
    ));
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

    // ---- SMEM buffers (N-D; the swizzled rings + the plain I32 mailbox) ----
    // Two emission forms, selected by `k.smem_pool`:
    //   * STATIC (default, the big-shape path): each SMEM data buffer is its own
    //     `T.alloc_buffer(scope="shared")`. TVM sizes the static SMEM footprint as
    //     the sum of the per-buffer extents (the 176 KB ncu observes at 1024/2048).
    //   * DYNAMIC POOL (the small-shape variant): canon's `T.SMEMPool()` form — ONE
    //     dynamic `alloc_buffer([0], "uint8", scope="shared.dyn")` that every data
    //     buffer aliases into via `pool.alloc(..., data=pool.ptr, byte_offset=...)`.
    //     The buffers carry the IR's own `byte_offset`, so `pool.move_base_to(off)`
    //     places each at exactly the offset the static form used (byte-for-byte the
    //     same physical layout) — but now as `shared.dyn`, cutting the STATIC SMEM
    //     footprint (and, with the dynamic pool, the register pressure) toward canon's
    //     159 KB / 40-reg shape. The mbar/tmem_addr buffers stay `T.alloc_shared`
    //     (tiny, and the protocol/peer-ref machinery references them by name).
    if k.smem_pool {
        out.push_str(&format!("{p}pool = T.SMEMPool()\n", p = pad(ind)));
    }
    for t in collect_tensors(k) {
        if t.space != MemorySpace::Smem {
            continue;
        }
        let name = ctx.tensor_name(t.id)?;
        let dims = t
            .shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let is_mailbox = is_int_dtype(t.dtype) && t.dtype != DType::U8;
        if k.smem_pool {
            // Alias into the dynamic pool at the IR's computed byte offset. `move_base_to`
            // sets the exact offset, then `pool.alloc` rounds it UP to `align` — the data
            // buffers' IR offsets are already 1024-aligned, so the offset is unchanged, but
            // the `align=1024` is REQUIRED: it sets the buffer's data-alignment attribute,
            // which the swizzled-SMEM / TMA-descriptor codegen assumes (canon's
            // `pool.alloc(..., align=1024)` / `alloc_mma(..., align=1024)`). With `align=0`
            // the swizzle atom indexing computes a misaligned address and the kernel faults
            // (cudaErrorMisalignedAddress) — even though the byte offset is identical. The
            // mailbox (flat row-major int, no layout) takes no alignment.
            let off = t
                .byte_offset
                .ok_or_else(|| format!("codegen: smem_pool tensor {name} has no byte_offset"))?;
            out.push_str(&format!(
                "{p}pool.move_base_to({off})\n",
                p = pad(ind),
                off = off,
            ));
            let (layout, align) = if is_mailbox {
                (String::new(), 0)
            } else {
                (format!(", layout={name}_layout"), 1024)
            };
            out.push_str(&format!(
                "{p}{name} = pool.alloc(({dims}), \"{dt}\", scope=\"shared.dyn\", align={align}{layout})\n",
                p = pad(ind),
                name = name,
                dims = dims,
                dt = dtype_str(t.dtype),
                align = align,
                layout = layout,
            ));
        } else if is_mailbox {
            // The scheduler mailbox: a flat row-major shared buffer (no swizzle).
            out.push_str(&format!(
                "{p}{name} = T.alloc_buffer(({dims}), \"{dt}\", scope=\"shared\")\n",
                p = pad(ind),
                name = name,
                dims = dims,
                dt = dtype_str(t.dtype),
            ));
        } else {
            out.push_str(&format!(
                "{p}{name} = T.alloc_buffer(({dims}), \"{dt}\", scope=\"shared\", layout={name}_layout)\n",
                p = pad(ind),
                name = name,
                dims = dims,
                dt = dtype_str(t.dtype),
            ));
        }
    }
    // ---- mbar shared buffers + tmem_addr ----
    // A multi-stage mbarrier allocates `[stages]` slots; each op indexes the slot it
    // uses. A single-stage mbar keeps the bootstrap's `[1]` form.
    //
    // In the dynamic-pool variant these buffers ALSO come from the pool (canon's
    // `TMABar(pool, ...)` / `pool.alloc([1], "uint32", align=4)`). They MUST NOT be a
    // separate static `T.alloc_shared(scope="shared")`: mixing a static `shared`
    // region (the mbars) with the `shared.dyn` pool pushes the dynamic base off its
    // 1024-byte boundary by the static region's size, so every swizzled pool buffer
    // ends up misaligned and the kernel faults (cudaErrorMisalignedAddress). Putting
    // them in the pool keeps the whole shared window one dynamic allocation, exactly
    // like canon. Emitted before `pool.commit()` so they're inside the pool's extent.
    let mut have_peer = false;
    if k.smem_pool {
        for s in &k.body {
            if let Stmt::MBarDef { mbar } = s {
                let name = ctx
                    .mbar_names
                    .get(&mbar.id)
                    .ok_or_else(|| format!("codegen: no name for mbar {}", mbar.id))?;
                let stages = ctx.mbar_stages.get(&mbar.id).copied().unwrap_or(1).max(1);
                // mbarriers are 8 B (uint64); align=8 keeps each slot naturally aligned.
                out.push_str(&format!(
                    "{p}{name} = pool.alloc([{stages}], \"uint64\", scope=\"shared.dyn\", align=8)\n",
                    p = pad(ind),
                    name = name,
                    stages = stages,
                ));
            }
        }
        out.push_str(&format!(
            "{p}tmem_addr = pool.alloc([1], \"uint32\", scope=\"shared.dyn\", align=4)\n",
            p = pad(ind)
        ));
        // Seal the dynamic pool: emit the `tirx.pool_max_bytes` size annotation from
        // the allocator high-water mark (the whole shared window: data buffers + mbars).
        out.push_str(&format!("{p}pool.commit()\n", p = pad(ind)));
    } else {
        for s in &k.body {
            if let Stmt::MBarDef { mbar } = s {
                let name = ctx
                    .mbar_names
                    .get(&mbar.id)
                    .ok_or_else(|| format!("codegen: no name for mbar {}", mbar.id))?;
                let stages = ctx.mbar_stages.get(&mbar.id).copied().unwrap_or(1).max(1);
                out.push_str(&format!(
                    "{p}{name} = T.alloc_shared([{stages}], \"uint64\")\n",
                    p = pad(ind),
                    name = name,
                    stages = stages,
                ));
            }
        }
        out.push_str(&format!(
            "{p}tmem_addr = T.alloc_shared([1], \"uint32\")\n",
            p = pad(ind)
        ));
    }

    // peer mbar (remote_coord) decl — find the referenced mbar id. The peer view
    // spans the full stage count so a multi-stage peer wait can index its slot.
    // Emitted in a stable (id-sorted) order — `peer_names` is a HashMap, so iterating
    // it directly made the emitted source nondeterministic across runs (the decls were
    // reordered run-to-run). Sort by mbar id to match the leader_ids block below.
    let mut peer_entries: Vec<(&u32, &String)> = ctx.peer_names.iter().collect();
    peer_entries.sort_unstable_by_key(|(id, _)| **id);
    for (mbar_id, peer_name) in peer_entries {
        let base = ctx
            .mbar_names
            .get(mbar_id)
            .ok_or_else(|| format!("codegen: peer references unknown mbar {mbar_id}"))?;
        let stages = ctx.mbar_stages.get(mbar_id).copied().unwrap_or(1).max(1);
        let ptr_var = format!("{peer_name}_ptr");
        // The `T.let` annotation binds the ptr as a Var typed by the exact
        // PointerType from `T.reinterpret` (a bare expr is not a Var, which
        // `decl_buffer(data=)` rejects).
        out.push_str(&format!(
            "{p}{ptr_var}: T.let = T.reinterpret(PointerType(PrimType(\"uint64\")), T.ptx.map_shared_rank({base}.ptr_to([0]), 1))\n",
            p = pad(ind),
            ptr_var = ptr_var,
            base = base,
        ));
        out.push_str(&format!(
            "{p}{peer} = T.decl_buffer([{stages}], \"uint64\", data={ptr_var}, scope=\"shared\")\n",
            p = pad(ind),
            peer = peer_name,
            stages = stages,
        ));
        have_peer = true;
    }
    let _ = have_peer;

    // Leader (CTA-0) DSMEM view of each TMA-load barrier, used uniformly by BOTH CTAs to
    // route every TMA completion to the leader's single barrier (the canonical
    // cta_group=2 pattern; replaces the illegal peer try_wait). `map_shared_rank(.., 0)`
    // is identity on CTA 0 and the cross-CTA remap on CTA 1, so one form serves both.
    // Emitted in a stable (id-sorted) order so the source is deterministic.
    let mut leader_ids: Vec<u32> = ctx.tma_leader_mbars.iter().copied().collect();
    leader_ids.sort_unstable();
    for id in leader_ids {
        let view = ctx
            .tma_leader_view_for(id)
            .ok_or_else(|| format!("codegen: tma leader references unknown mbar {id}"))?;
        let base = ctx
            .mbar_names
            .get(&id)
            .ok_or_else(|| format!("codegen: tma leader references unknown mbar {id}"))?;
        let stages = ctx.mbar_stages.get(&id).copied().unwrap_or(1).max(1);
        let ptr_var = format!("{view}_ptr");
        out.push_str(&format!(
            "{p}{ptr_var}: T.let = T.reinterpret(PointerType(PrimType(\"uint64\")), T.ptx.map_shared_rank({base}.ptr_to([0]), 0))\n",
            p = pad(ind),
            ptr_var = ptr_var,
            base = base,
        ));
        out.push_str(&format!(
            "{p}{view} = T.decl_buffer([{stages}], \"uint64\", data={ptr_var}, scope=\"shared\")\n",
            p = pad(ind),
            view = view,
            stages = stages,
        ));
    }

    // ---- REG fragments (epilogue) ----
    // Each is a warpgroup-collective `(128, width)` register tile (`T.wg_reg_tile`): the 128
    // threads of the warpgroup own one row each. These are emitted INLINE at their TensorDef
    // site (inside the epilogue task loop), NOT hoisted here — see the TensorDef arm in
    // emit_stmt: a function-scope tile is whole-kernel-lived and ptxas spills it to local
    // memory, while a loop-local declaration (like canon's) promotes to registers.
    out.push('\n');

    // ---- walk the body ----
    emit_body(&mut out, &k.body, ind, &ctx, Scope::Function)?;

    Ok(merge_adjacent_guards(&out))
}

/// Merge ADJACENT identical single-thread guard blocks (`if tid_in_wg == 0:` /
/// `if T.ptx.elect_sync():`) into one block. Canon writes the epilogue trio as
/// `if tid%128==0: fence; store; commit` under ONE guard; emitting each op with
/// its own guard costs a BSSY/BSYNC reconvergence pair per block in SASS (ncu
/// nvfp4 4096: BSSY +14K / BSYNC +22K vs canon). Both guard predicates are
/// invariant across adjacent blocks (tid constant; elect.sync re-elects the same
/// leader with no divergence in between), so folding is semantics-preserving.
/// Conservative: only exactly-adjacent lines (no blank/other line between).
fn merge_adjacent_guards(src: &str) -> String {
    const GUARDS: [&str; 2] = ["if tid_in_wg == 0:", "if T.ptx.elect_sync():"];
    let lines: Vec<&str> = src.lines().collect();
    let mut out: Vec<&str> = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();
        let indent_len = line.len() - trimmed.len();
        if GUARDS.contains(&trimmed) {
            out.push(line);
            i += 1;
            loop {
                // copy the guard's body: lines strictly deeper than the guard
                while i < lines.len() {
                    let l = lines[i];
                    let lt = l.trim_start();
                    if lt.is_empty() || l.len() - lt.len() <= indent_len {
                        break;
                    }
                    out.push(l);
                    i += 1;
                }
                // absorb an immediately-following identical guard at the same depth
                let dup = i < lines.len() && {
                    let l = lines[i];
                    let lt = l.trim_start();
                    lt == trimmed && l.len() - lt.len() == indent_len
                };
                if dup {
                    i += 1; // drop the duplicate guard line; keep copying its body
                } else {
                    break;
                }
            }
        } else {
            out.push(line);
            i += 1;
        }
    }
    let mut s = out.join("\n");
    if src.ends_with('\n') {
        s.push('\n');
    }
    s
}

/// Collect every tensor referenced anywhere (args + defs + slices), deduped by id,
/// in a deterministic (id-sorted) order.
/// Scale-factor tensor ids derived from USAGE, not dtype. A tensor is a scale
/// factor iff it feeds the block-scaled MMA datapath on its way INTO TMEM: an
/// endpoint of a `tcgen05.cp` (the SMEM→TMEM SF staging copy), or the GMEM
/// source of a `TmaLoad` that fills an SF SMEM ring. dtype alone proves
/// nothing — a plain fp8 DATA tensor is also e4m3, and laying it out as SF
/// bytes would silently corrupt it. (The TMEM side of the SF path is a
/// `TmemOperand` now — physical addresses, no tensor ids; see `sf_views`.)
#[derive(Default)]
struct SfIds {
    smem: HashSet<u32>,
    gmem: HashSet<u32>,
}

fn collect_sf_ids(k: &Kernel) -> SfIds {
    fn walk(stmts: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
        for s in stmts {
            f(s);
            for body in s.child_bodies() {
                walk(body, f);
            }
        }
    }
    let mut ids = SfIds::default();
    // Pass 1: tcgen05.cp sources (SMEM).
    walk(&k.body, &mut |s| {
        if let Stmt::Tcgen05Cp { src, .. } = s {
            ids.smem.insert(src.tensor.id);
        }
    });
    // Pass 2: GMEM sources of the TMA loads that fill an SF SMEM ring (one level —
    // SF bytes flow gmem -> smem -> tmem, there are no longer chains).
    walk(&k.body, &mut |s| {
        if let Stmt::TmaLoad { dst, src, .. } = s {
            if ids.smem.contains(&dst.tensor.id) {
                ids.gmem.insert(src.id);
            }
        }
    });
    ids
}

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
        TensorDef { tensor } => note_tensor(tensor, map),
        TmaLoad { dst, src, .. } => {
            note_slice(dst, map);
            note_tensor(src, map);
        }
        TmaStore { dst, src, .. } => {
            note_tensor(dst, map);
            note_slice(src, map);
        }
        Tcgen05Mma { a, b, .. } => {
            // TMEM operands (dst/sfa/sfb, TmemOperand-form a/b) carry no tensor.
            for op in [a, b] {
                if let MmaOperand::Slice(s) = op {
                    note_slice(s, map);
                }
            }
        }
        Tcgen05Cp { src, .. } => {
            note_slice(src, map);
        }
        Tcgen05Ld { dst, .. } => {
            note_slice(dst, map);
        }
        Tcgen05St { src, .. } => {
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

/// Collapse the MMA's value-model granularity into the single coalesced op TIRx
/// wants. The nymph IR emits MMA at the value-model's k=16 sub-slice unit, but
/// TIRx's `gemm_async` takes the *full* BLK_K operand and tiles internally (a
/// 16-wide / 32-byte sub-slice of a 128B-swizzle atom hits
/// `cudaErrorIllegalInstruction`); `try_collapse_mma_run` recovers the full-operand
/// form. This is a value/protocol-neutral source-form merge — same atoms, no fence
/// or accumulation change.
///
/// NOTE: the epilogue tmem->reg fence structure (one `wait_ld` per EPI_N / NOL band)
/// is expressed in the IR itself and verified by the protocol checker — it is NOT
/// recovered here. The per-8-col `tcgen05.ld` atoms are emitted 1:1 (they are legal
/// and run bit-exact); the IR, not codegen, owns the fence granularity.
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
        // NOTE: the reg->smem store run is intentionally NOT collapsed. The canonical
        // epilogue stores reg->smem in 8-element (128-bit) sub-slices so the copy
        // dispatches to STSM; a wider `Tx.wg.copy` drops to STS.128 or, when the
        // swizzle atom doesn't tile the wide slice, to the scalar fallback (which does
        // a direct thread-axis BufferStore and is rejected by LowerTIRxCleanup).
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
        ..
    } = &stmts[0]
    else {
        return None;
    };
    // Only collapse the simple (non block-scaled) GEMM run.
    if sfa.is_some() || sfb.is_some() {
        return None;
    }
    // K is the last operand dim of an SMEM slice (a TMEM operand names no
    // K-offset to advance, so it never collapses).
    let (MmaOperand::Slice(a_sl), MmaOperand::Slice(b_sl)) = (a, b) else {
        return None;
    };
    // Record the first op's K offset/extent.
    let a_kdim = a_sl.offsets.len().checked_sub(1)?;
    let b_kdim = b_sl.offsets.len().checked_sub(1)?;
    let a_k0 = as_int(&a_sl.offsets[a_kdim])?;
    let b_k0 = as_int(&b_sl.offsets[b_kdim])?;
    let a_kext = as_int(&a_sl.shape[a_kdim])?;
    let b_kext = as_int(&b_sl.shape[b_kdim])?;

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
        let (MmaOperand::Slice(a2), MmaOperand::Slice(b2)) = (a2, b2) else {
            break;
        };
        // Same dst (accumulator), same A/B tensors, accum True (continuation).
        if d2 != dst
            || !Arc::ptr_eq(&a2.tensor, &a_sl.tensor)
            || !Arc::ptr_eq(&b2.tensor, &b_sl.tensor)
        {
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
        if a2.offsets[..a_kdim] != a_sl.offsets[..a_kdim]
            || a2.shape[..a_kdim] != a_sl.shape[..a_kdim]
            || b2.offsets[..b_kdim] != b_sl.offsets[..b_kdim]
            || b2.shape[..b_kdim] != b_sl.shape[..b_kdim]
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
    let mut a_full = a_sl.clone();
    let mut b_full = b_sl.clone();
    a_full.shape[a_kdim] = ScalarValue::Int(a_khi - a_k0);
    b_full.shape[b_kdim] = ScalarValue::Int(b_khi - b_k0);

    let collapsed = Stmt::Tcgen05Mma {
        dst: dst.clone(),
        a: MmaOperand::Slice(a_full),
        b: MmaOperand::Slice(b_full),
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
        // The collapse only runs on the non-scaled GEMM (bailed above on any SF),
        // so the NVFP4 flags are always their dense defaults here; the dense
        // cta_group=2 m=128/256 accumulator never uses lane_align.
        sf_e4m3: false,
        sf_block: 0,
        a_fp4: false,
        b_fp4: false,
        lane_align: 0,
    };
    Some((collapsed, count))
}

/// Build the naming context: A/B/C for args by position; SMEM/TMEM/REG/mbar by role.
fn build_ctx(k: &Kernel) -> Result<Ctx, String> {
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut tensors: HashMap<u32, Arc<Tensor>> = HashMap::new();
    let sf = collect_sf_ids(k);
    for (i, t) in k.args.iter().enumerate() {
        names.insert(t.id, arg_name(i));
        tensors.insert(t.id, t.clone());
    }

    // SMEM/TMEM/REG names, deterministic by id order. The bootstrap uses 2 SMEM
    // (A_smem, B_smem); the full kernel adds per-consumer input rings, D writeback
    // rings, and the I32 task mailbox. SMEM tensors are classified by dtype/layout:
    //   - I32, no layout      -> `task_smem` (the scheduler mailbox)
    //   - swizzled (ab dtype) -> `ab_smem{i}` (A/B MMA operand rings)
    //   - ab dtype, no layout -> `d_smem{i}`  (D writeback rings; we synthesize a
    //                            swizzle layout from the row byte width)
    let mut tmem_idx = 0usize;
    let mut reg_idx = 0usize;
    let mut ab_idx = 0usize;
    let mut d_idx = 0usize;
    for t in collect_tensors(k) {
        tensors.entry(t.id).or_insert_with(|| t.clone());
        if names.contains_key(&t.id) {
            continue;
        }
        let name = match t.space {
            MemorySpace::Smem => {
                let is_int = matches!(
                    t.dtype,
                    super::dtype::DType::I8
                        | super::dtype::DType::U8
                        | super::dtype::DType::I16
                        | super::dtype::DType::U16
                        | super::dtype::DType::I32
                        | super::dtype::DType::U32
                        | super::dtype::DType::I64
                        | super::dtype::DType::U64
                );
                // u8 = packed-fp4 operand ring (NOT the i32/u32 mailbox); give it a
                // unique ab_smem name so two operands don't collide on "task_smem".
                if is_int && t.dtype != DType::U8 {
                    "task_smem".to_string()
                } else if t.dtype == DType::U8 || t.layout.is_some() {
                    let n = format!("ab_smem{ab_idx}");
                    ab_idx += 1;
                    n
                } else {
                    let n = format!("d_smem{d_idx}");
                    d_idx += 1;
                    n
                }
            }
            MemorySpace::Tmem => {
                // TMEM is not a tensor anymore: no IR op references a TMEM tensor
                // (all TMEM addressing is physical). A stray TMEM-space tensor can
                // only come from a leftover TensorDef; name it so the walk stays
                // total — nothing references the name.
                let n = format!("tmem_view{tmem_idx}");
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

    // mbar names + peer names + stage counts.
    let mut mbar_names: HashMap<u32, String> = HashMap::new();
    let mut peer_names: HashMap<u32, String> = HashMap::new();
    let mut mbar_stages: HashMap<u32, u32> = HashMap::new();
    let mut mbar_idx = 0usize;
    // The bootstrap names its two single-stage mbars smem_full / mma_done; the full
    // kernel has six (smem ring, tmem accumulator pipe, task mailbox). Naming them by
    // declaration order keeps the bootstrap output unchanged while staying general.
    let mbar_default = [
        "smem_full",
        "smem_empty",
        "tmem_full",
        "tmem_empty",
        "task_full",
        "task_empty",
    ];
    fn walk_mbars(
        stmts: &[Stmt],
        mbar_names: &mut HashMap<u32, String>,
        peer_names: &mut HashMap<u32, String>,
        mbar_stages: &mut HashMap<u32, u32>,
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
                    mbar_stages.insert(mbar.id, mbar.stages.max(1));
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
                walk_mbars(
                    body,
                    mbar_names,
                    peer_names,
                    mbar_stages,
                    mbar_idx,
                    mbar_default,
                );
            }
        }
    }
    walk_mbars(
        &k.body,
        &mut mbar_names,
        &mut peer_names,
        &mut mbar_stages,
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

    // The TMA-load completion barriers to leader-route in cluster (cta_group=2) mode:
    // EVERY mbar that a `TmaLoad` signals AND that carries a peer reference (the IR's
    // `smem_full` for A/B, `sf_full` for the scales, ...). Routing both CTAs' TMA to the
    // leader's copy of each barrier (and waiting only the local copy) is the legal
    // substitute for the peer wait, AND the prerequisite for multicast loads — a
    // `multicast::cluster` copy's per-destination transaction count must accumulate on
    // the single leader barrier (the `* cta_group` leader expect_tx accounts for it).
    // Only the barriers with a peer reference are routed; single-CTA TMA barriers stay
    // local. Only populated when cta_group > 1.
    let mut tma_leader_mbars: std::collections::HashSet<u32> = std::collections::HashSet::new();
    if cta_group > 1 {
        fn find_tma_mbars(stmts: &[Stmt], out: &mut std::collections::HashSet<u32>) {
            for s in stmts {
                if let Stmt::TmaLoad { mbar, .. } = s {
                    out.insert(mbar.mbar.id);
                }
                for body in s.child_bodies() {
                    find_tma_mbars(body, out);
                }
            }
        }
        let mut tma_mbars = std::collections::HashSet::new();
        find_tma_mbars(&k.body, &mut tma_mbars);
        // Only leader-route a barrier that actually has a peer reference (a true cluster
        // TMA barrier); a single-CTA TMA barrier stays local.
        for id in tma_mbars {
            if peer_names.contains_key(&id) {
                tma_leader_mbars.insert(id);
            }
        }
    }

    // The single TMEM view buffer spans every allocated column band:
    // `max(base_col + n_cols)` over the kernel's TmemAllocs.
    let mut tmem_view_cols: Option<usize> = None;
    fn find_tmem_view_cols(stmts: &[Stmt], out: &mut Option<usize>) {
        for s in stmts {
            if let Stmt::TmemAlloc {
                base_col, n_cols, ..
            } = s
            {
                let end = (*base_col + *n_cols) as usize;
                *out = Some(out.map_or(end, |e: usize| e.max(end)));
            }
            for body in s.child_bodies() {
                find_tmem_view_cols(body, out);
            }
        }
    }
    find_tmem_view_cols(&k.body, &mut tmem_view_cols);

    // NVFP4 SF TMEM views, keyed by the operands' physical base columns.
    let mut sf_views: Vec<SfView> = Vec::new();
    collect_sf_views(&k.body, &mut sf_views)?;

    // Per-REG-tensor width after the epilogue collapse. Walk the collapsed bodies and
    // record, for each REG fragment, `max(offset + width)` over every slice — the band
    // a single `.view(...)` alias must span. (A capped drain writes the 256-wide output
    // reg in two 128-col groups, so the FULL extent comes from offset+width, not the
    // per-op width alone.)
    let mut reg_widths: HashMap<u32, usize> = HashMap::new();
    fn note_reg_width(s: &TensorSlice, widths: &mut HashMap<u32, usize>) {
        if s.tensor.space == MemorySpace::Reg {
            let off = s.offsets.first().and_then(as_int).unwrap_or(0).max(0) as usize;
            let w = s.shape.first().and_then(as_int).unwrap_or(0).max(0) as usize;
            let e = widths.entry(s.tensor.id).or_insert(0);
            *e = (*e).max(off + w);
        }
    }
    fn walk_reg_widths(stmts: &[Stmt], widths: &mut HashMap<u32, usize>) {
        for s in collapse_body(stmts) {
            match &s {
                Stmt::Tcgen05Ld { dst, num, .. } => {
                    if dst.tensor.space == MemorySpace::Reg {
                        let off = dst.offsets.first().and_then(as_int).unwrap_or(0).max(0) as usize;
                        let e = widths.entry(dst.tensor.id).or_insert(0);
                        *e = (*e).max(off + *num as usize);
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

    // Scalar var names. Every `ScalarDef` introduces an SSA register var
    // (`NAME: T.int32 = init`, read as `NAME`). Var ids are globally unique, so
    // a per-id name (`s{id}`) is stable and collision-free.
    let mut scalar_names: HashMap<u32, String> = HashMap::new();
    fn walk_scalar_defs(stmts: &[Stmt], scalar_names: &mut HashMap<u32, String>) {
        for s in stmts {
            let defined = match s {
                Stmt::ScalarDef { var, .. }
                | Stmt::ShuffleSync { var, .. }
                | Stmt::ClcQueryCancel { var, .. } => Some(var),
                _ => None,
            };
            if let Some(var) = defined {
                scalar_names
                    .entry(var.id.0)
                    .or_insert_with(|| format!("s{}", var.id.0));
            }
            for body in s.child_bodies() {
                walk_scalar_defs(body, scalar_names);
            }
        }
    }
    walk_scalar_defs(&k.body, &mut scalar_names);

    Ok(Ctx {
        names,
        mbar_names,
        peer_names,
        mbar_stages,
        var_names: HashMap::new(),
        scalar_names,
        cta_group,
        tmem_view_cols,
        reg_widths,
        tma_leader_mbars,
        num_clusters: (k.launch_cta_count() / (cta_group as usize).max(1)).max(1),
        sf_views,
        sf,
    })
}

/// Register one SF view per physical base column, in first-use order
/// (`SFA_tmem`, `SFB_tmem`, then `sf_tmem{i}`). `rows` is the scaled-row count
/// of this use; the view's logical rows round it up to whole 128-lane
/// super-blocks — exactly canon's `(128 * n_chunks, SF_K)` decl_buffer.
fn note_sf_view(
    views: &mut Vec<SfView>,
    op: &TmemOperand,
    rows: usize,
    nblocks: usize,
) -> Result<(), String> {
    let Some(col) = as_int(&op.col) else {
        return Err(
            "codegen: SF TMEM operand base column must be a compile-time constant".to_string(),
        );
    };
    if col < 0 {
        return Err("codegen: SF TMEM operand base column is negative".to_string());
    }
    let col = col as usize;
    let logical_rows = rows.div_ceil(128) * 128;
    let logical_cols = nblocks;
    if let Some(v) = views.iter().find(|v| v.col == col) {
        if v.logical_rows != logical_rows || v.logical_cols != logical_cols {
            return Err(format!(
                "codegen: SF TMEM views at col {col} disagree on the logical shape"
            ));
        }
        return Ok(());
    }
    let name = match views.len() {
        0 => "SFA_tmem".to_string(),
        1 => "SFB_tmem".to_string(),
        i => format!("sf_tmem{i}"),
    };
    views.push(SfView {
        name,
        col,
        logical_rows,
        logical_cols,
    });
    Ok(())
}

fn collect_sf_views(stmts: &[Stmt], views: &mut Vec<SfView>) -> Result<(), String> {
    for s in stmts {
        match s {
            Stmt::Tcgen05Mma {
                m,
                n,
                k,
                cta_group,
                sfa: Some(sfa),
                sfb: Some(sfb),
                sf_block,
                ..
            } => {
                let nblocks = if *sf_block == 0 {
                    1
                } else {
                    (*k / *sf_block) as usize
                };
                let a_rows = (if *cta_group == 1 { *m } else { *m / 2 }) as usize;
                note_sf_view(views, sfa, a_rows, nblocks)?;
                note_sf_view(views, sfb, *n as usize, nblocks)?;
            }
            Stmt::Tcgen05Cp { dst, src, .. } if dst.dtype == DType::F8E4M3 => {
                // The src is the staged SF SMEM tile (..., rows, SF_CTA_K); its
                // trailing dims are the view's logical (rows, cols).
                let (Some(rows), Some(sf_k)) = (
                    src.shape
                        .get(src.shape.len().saturating_sub(2))
                        .and_then(as_int),
                    src.shape.last().and_then(as_int),
                ) else {
                    return Err(
                        "codegen: tcgen05_cp src shape must be static for the SF view".to_string(),
                    );
                };
                note_sf_view(views, dst, rows.max(0) as usize, sf_k.max(0) as usize)?;
            }
            _ => {}
        }
        for body in s.child_bodies() {
            collect_sf_views(body, views)?;
        }
    }
    Ok(())
}

/// The declared (full) width of a REG fragment's wg tile = its collapsed band width.
fn reg_view_width(t: &Arc<Tensor>, ctx: &Ctx) -> usize {
    ctx.reg_widths
        .get(&t.id)
        .copied()
        .filter(|w| *w > 0)
        .unwrap_or_else(|| t.shape.first().copied().unwrap_or(0))
}

/// Column-sliced expression on a REG wg tile: `name[:, off:off+width]` (or
/// `name[:, :]` when it spans the whole tile). The fragment is a `T.wg_reg_tile`,
/// i.e. already a 2D `(128, full)` warpgroup tile — no `.view()` indirection.
fn emit_reg_view_slice(
    _out: &mut String,
    _p: &str,
    t: &Arc<Tensor>,
    off: &ScalarValue,
    width: usize,
    ctx: &Ctx,
) -> Result<String, String> {
    let name = ctx.tensor_name(t.id)?.to_string();
    let full = reg_view_width(t, ctx);
    if as_int(off) == Some(0) && width == full {
        Ok(format!("{name}[:, :]"))
    } else {
        let off_s = emit_scalar(off, ctx)?;
        Ok(format!("{name}[:, {off_s}:{off_s} + {width}]"))
    }
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

/// Name of a loop var (`for v in range(...)`). Scalar vars are NOT named here —
/// they are SSA register vars named via `Ctx::scalar_names`.
fn var_name(ctx: &Ctx, v: &Var) -> String {
    ctx.var_names
        .get(&v.id.0)
        .cloned()
        .unwrap_or_else(|| format!("v{}", v.id.0))
}

/// The Python expression that *reads* a var: a scalar reads as its plain `NAME`
/// (an SSA `T.int32` register var, like canon's `sa_stage`), a loop var likewise.
fn var_ref(ctx: &Ctx, v: &Var) -> String {
    if v.binding == VarBinding::Scalar {
        if let Some(name) = ctx.scalar_names.get(&v.id.0) {
            return name.clone();
        }
    }
    var_name(ctx, v)
}

/// Emit a scalar value as a Python expression, parenthesizing per precedence.
fn emit_scalar(sv: &ScalarValue, ctx: &Ctx) -> Result<String, String> {
    emit_scalar_prec(sv, ctx, 0)
}

fn emit_scalar_prec(sv: &ScalarValue, ctx: &Ctx, parent_prec: u8) -> Result<String, String> {
    match sv {
        ScalarValue::Int(i) => Ok(i.to_string()),
        ScalarValue::Var(v) => Ok(var_ref(ctx, v)),
        ScalarValue::Scope(k) => Ok(scope_name(*k).to_string()),
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

/// Conservatively decide whether a scalar is provably `>= 0`. Used to rewrite
/// `x % 2` into `x & 1` only when the two are equal (true exactly for non-negative
/// `x`). The pipeline-phase parities (`occ % 2`, `(occ + 1) % 2`, ring/slot indices)
/// are all built from non-negative loop counters, `FloorDiv`s, and `Mod`s, so they
/// qualify; the only negative scalar in the kernel (`task_id`/`bcast_id == -1`
/// sentinel) is never an operand of a `% 2`.
fn is_nonneg(sv: &ScalarValue) -> bool {
    match sv {
        ScalarValue::Int(i) => *i >= 0,
        // Loop counters and scope ids (lane/warp/wg/cta/tid) are all >= 0.
        ScalarValue::Var(_) | ScalarValue::Scope(_) => true,
        ScalarValue::Expr(e) => match e.op {
            // FloorDiv / Mod by a positive divisor of a non-negative dividend is
            // non-negative; Mul/Add of non-negatives stay non-negative.
            ScalarOp::FloorDiv | ScalarOp::Mod => is_nonneg(&e.args[0]) && is_nonneg(&e.args[1]),
            ScalarOp::Mul | ScalarOp::Add | ScalarOp::Min | ScalarOp::Max => {
                e.args.iter().all(is_nonneg)
            }
            // Anything else (Sub, Neg, Select, comparisons, ...) is not assumed >= 0.
            _ => false,
        },
    }
}

/// If `sv` is a positive power-of-two literal `2^k` (k >= 1), return `k`. Drives the
/// strength-reduction of `% 2^k` -> `& (2^k - 1)` and `// 2^k` -> `>> k` (Lever 3).
fn as_pow2_shift(sv: &ScalarValue) -> Option<u32> {
    match sv {
        ScalarValue::Int(i) if *i > 1 && (*i as u64).is_power_of_two() => {
            Some((*i as u64).trailing_zeros())
        }
        _ => None,
    }
}

/// A positive-integer-literal divisor, returned as the value (for the truncdiv/truncmod
/// path). `None` for non-literal or non-positive divisors (those keep floordiv/floormod).
fn positive_int_divisor(sv: &ScalarValue) -> Option<i64> {
    match sv {
        ScalarValue::Int(i) if *i > 0 => Some(*i),
        _ => None,
    }
}

fn emit_expr(e: &ScalarExpr, ctx: &Ctx, parent_prec: u8) -> Result<String, String> {
    let prec = op_prec(e.op);
    // ---- Non-negative div/mod strength reduction (generic, driven by `is_nonneg`) ----
    // The nymph IR's ring/slot/phase indices and the L2-swizzle coords are all built
    // from non-negative loop counters and scope ids, so `is_nonneg(dividend)` holds.
    // TVM lowers a SIGNED `floordiv`/`floormod` (what Python `//`/`%` parse to) with a
    // sign-correction tail: `floormod(x,d)` -> `(x % d) + (d & ((x % d) >> 31))` and
    // `floordiv(x,d)` -> `(x / d) + ((x % d) >> 31)`. For a provably non-negative `x`
    // every correction term is 0, so it is pure wasted integer ALU recomputed at each
    // index (the L2 `% 5`//5` is recomputed per task per role). Two rewrites, both gated
    // ONLY on the generic `is_nonneg` + a positive-literal divisor (no kernel/shape key):
    //
    //   * divisor == 2^k  -> bit ops: `% 2^k` => `(x) & (2^k - 1)`, `// 2^k` => `(x) >> k`.
    //     No div/mod instruction at all. (Generalizes the prior `% 2 -> & 1`, which also
    //     dodged a TIRx->CUDA simplifier bug that mis-folds `floormod(floordiv(..),2)` to
    //     0; bit ops are left intact by the simplifier, so that rationale still holds.)
    //   * other positive literal (e.g. 5) -> `T.truncdiv` / `T.truncmod`: numerically
    //     identical to floordiv/floormod for non-negative `x`, but lowers to a plain
    //     `/` / `%` with NO sign-correction tail.
    //
    // `&`, `>>` bind looser than `% // * + -` in Python: parenthesize the dividend and
    // wrap the whole result for any parent binding tighter than bitand/shift.
    if (e.op == ScalarOp::Mod || e.op == ScalarOp::FloorDiv) && is_nonneg(&e.args[0]) {
        if let Some(k) = as_pow2_shift(&e.args[1]) {
            let lhs = emit_scalar_prec(&e.args[0], ctx, 0)?;
            let s = if e.op == ScalarOp::Mod {
                format!("({lhs}) & {mask}", mask = (1u64 << k) - 1)
            } else {
                format!("({lhs}) >> {k}")
            };
            return Ok(if parent_prec > 0 { format!("({s})") } else { s });
        }
        if let Some(d) = positive_int_divisor(&e.args[1]) {
            let fname = if e.op == ScalarOp::Mod {
                "T.truncmod"
            } else {
                "T.truncdiv"
            };
            // A function call is atomic — no surrounding parens needed for any parent.
            return Ok(format!(
                "{fname}({}, {d})",
                emit_scalar_prec(&e.args[0], ctx, 0)?
            ));
        }
    }
    let s = match e.op {
        ScalarOp::Neg => format!("-{}", emit_scalar_prec(&e.args[0], ctx, prec)?),
        ScalarOp::Not => format!("not {}", emit_scalar_prec(&e.args[0], ctx, prec)?),
        ScalarOp::Select => format!(
            "({} if {} else {})",
            emit_scalar_prec(&e.args[1], ctx, 0)?,
            emit_scalar_prec(&e.args[0], ctx, 0)?,
            emit_scalar_prec(&e.args[2], ctx, 0)?,
        ),
        ScalarOp::Min => format!(
            "T.min({}, {})",
            emit_scalar_prec(&e.args[0], ctx, 0)?,
            emit_scalar_prec(&e.args[1], ctx, 0)?
        ),
        ScalarOp::Max => format!(
            "T.max({}, {})",
            emit_scalar_prec(&e.args[0], ctx, 0)?,
            emit_scalar_prec(&e.args[1], ctx, 0)?
        ),
        _ => {
            let Some(sym) = binop_symbol(e.op) else {
                // An op with no TVMScript lowering (e.g. `Xor`) must not leak a
                // placeholder literal into the emitted Python source (a syntax
                // error at best) — fail closed like every other unsupported node.
                return Err(format!(
                    "codegen: scalar op {:?} has no TVMScript lowering",
                    e.op
                ));
            };
            format!(
                "{} {} {}",
                emit_scalar_prec(&e.args[0], ctx, prec)?,
                sym,
                emit_scalar_prec(&e.args[1], ctx, prec + 1)?
            )
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
        Ok(format!("({s})"))
    } else {
        Ok(s)
    }
}

/// Fold `lo + extent` for the hi bound when both are literals; otherwise emit `lo + extent`.
fn add_bound(lo: &ScalarValue, extent: &ScalarValue, ctx: &Ctx) -> Result<String, String> {
    match (lo, extent) {
        (ScalarValue::Int(a), ScalarValue::Int(b)) => Ok((a + b).to_string()),
        (ScalarValue::Int(0), _) => emit_scalar(extent, ctx),
        _ => Ok(format!(
            "{} + {}",
            emit_scalar_prec(lo, ctx, 4)?,
            emit_scalar_prec(extent, ctx, 5)?
        )),
    }
}

/// Emit a staged SMEM tile operand for a TMA copy / wg copy. A leading ring/d-tile
/// dim of EXTENT 1 is collapsed to an integer index (which drops the axis in
/// TVMScript), so the operand rank matches the GMEM/MMA tile — mirroring the
/// canonical `Asmem[stage, c]` / `Dsmem[wg_id, db]` indexing. Trailing dims stay as
/// ranges. (Full-extent leading dims, if any, would stay ranges too.)
fn emit_smem_tile(s: &TensorSlice, ctx: &Ctx) -> Result<String, String> {
    let name = ctx.tensor_name(s.tensor.id)?;
    let mut dims = Vec::new();
    for (off, ext) in s.offsets.iter().zip(s.shape.iter()) {
        if as_int(ext) == Some(1) {
            // size-1 ring index: drop the axis
            dims.push(emit_scalar(off, ctx)?);
        } else {
            let lo = emit_scalar(off, ctx)?;
            let hi = add_bound(off, ext, ctx)?;
            dims.push(format!("{lo}:{hi}"));
        }
    }
    Ok(format!("{name}[{}]", dims.join(", ")))
}

/// Emit a warpgroup-collective SMEM store DST tile: like `emit_smem_tile`, but a
/// size-1 dim whose offset is the per-thread `tid_in_wg` lane axis becomes a full
/// span `:` (the 128-row warpgroup tile) — the value model writes that row per
/// thread, but `Tx.wg.copy` takes the whole tile and the layout maps lanes to rows.
/// (Canonical: `Tx.wg.copy(Dsmem[0, db, :, c0:c1], ...)`.)
fn emit_smem_wg_store_tile(s: &TensorSlice, ctx: &Ctx) -> Result<String, String> {
    let name = ctx.tensor_name(s.tensor.id)?;
    let mut dims = Vec::new();
    for (off, ext) in s.offsets.iter().zip(s.shape.iter()) {
        let is_lane_axis = matches!(off, ScalarValue::Scope(ScopeValueKind::TidInWg));
        if is_lane_axis {
            // per-thread row -> the full warpgroup row span
            dims.push(":".to_string());
        } else if as_int(ext) == Some(1) {
            dims.push(emit_scalar(off, ctx)?); // size-1 ring index: drop the axis
        } else {
            let lo = emit_scalar(off, ctx)?;
            let hi = add_bound(off, ext, ctx)?;
            dims.push(format!("{lo}:{hi}"));
        }
    }
    Ok(format!("{name}[{}]", dims.join(", ")))
}

/// A scalar element address into the SMEM mailbox: every dim is a size-1 index, so
/// the result `task_smem[stage, field]` is a single cell (an lvalue for a store and
/// an rvalue for a scalar load). Used by `ScalarDef`(Tensor init), `StoreScalar`.
fn emit_scalar_addr(s: &TensorSlice, ctx: &Ctx) -> Result<String, String> {
    let name = ctx.tensor_name(s.tensor.id)?;
    let dims = s
        .offsets
        .iter()
        .map(|off| emit_scalar(off, ctx))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    Ok(format!("{name}[{dims}]"))
}

/// A scalar load from a 1-element tensor slice — same address form as a store.
fn emit_scalar_load(s: &TensorSlice, ctx: &Ctx) -> Result<String, String> {
    emit_scalar_addr(s, ctx)
}

/// The MMA accumulator dst: an absolute column slice of the single `tmem`
/// view — `tmem[:, col:col+n]`. The accumulator is lane-anchored at row 0, so
/// the lane axis spans all 128 lanes; the column band is the operand's
/// absolute physical base plus the MMA's n.
fn emit_tmem_dst(op: &TmemOperand, n: u32, ctx: &Ctx) -> Result<String, String> {
    let col_s = emit_scalar(&op.col, ctx)?;
    let hi = add_bound(&op.col, &ScalarValue::Int(i64::from(n)), ctx)?;
    Ok(format!("tmem[:, {col_s}:{hi}]"))
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
    scope: Scope,
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
            // A *peer* wait (remote_coord set) is SKIPPED: `mbarrier.try_wait` on a
            // `map_shared_rank`-remapped DSMEM address raises `cudaErrorIllegalInstruction`
            // on sm_100 (verified). The peer CTA's TMA completion is instead ordered by
            // routing BOTH CTAs' TMA loads to the leader CTA's single `smem_full` barrier
            // (the canonical cta_group=2 pattern): each CTA's `Tx.copy_async` signals the
            // leader's barrier via `map_shared_rank(.., 0)`, the leader issues one
            // `arrive.expect_tx` for the FULL cluster byte count, and waits its OWN local
            // barrier (which both CTAs fill). See `TmaLoad` / `MBarrierArriveExpectTx`.
            let mut j = i;
            let mut emitted_any = false;
            while j < collapsed.len() && matches!(collapsed[j], Stmt::MBarrierWait { .. }) {
                if let Stmt::MBarrierWait { mbar, phase, stage } = &collapsed[j] {
                    if mbar.remote_coord.is_none() {
                        let slot_ptr = mbar_slot_ptr(mbar, stage, ctx)?;
                        let phase_s = phase
                            .as_ref()
                            .map(|ph| emit_scalar(ph, ctx))
                            .transpose()?
                            .unwrap_or_else(|| "0".to_string());
                        out.push_str(&format!(
                            "{p}T.ptx.mbarrier.try_wait({slot_ptr}, {phase_s})\n"
                        ));
                        emitted_any = true;
                    }
                }
                j += 1;
            }
            if emitted_any && scope.is_function() {
                // Fence + cta_sync ONLY at function (prologue) scope, where the mbarrier-init
                // visibility ordering across the whole CTA needs it. In ROLE scope (the hot
                // MMA / loader / epilogue loops) the mbarrier handshake alone orders the async
                // engines (TMA→smem_full→MMA, MMA→tmem_full→epilogue) — exactly as the canonical
                // kernel does, which emits ZERO `tcgen05.fence.after_thread_sync()` after its
                // loop waits. Emitting one per wait over-fenced the hot loop: the FENCE/MEMBAR
                // serialized MMA issue and left the tensor core idle (~1.7% lower tensor-active),
                // the exact bench gap. The mbarrier acquire already makes the produced SMEM/TMEM
                // visible to the consuming async op, so the per-wait fence is redundant here.
                out.push_str(&format!("{p}T.ptx.tcgen05.fence.after_thread_sync()\n"));
                out.push_str(&format!("{p}T.cuda.cta_sync()\n"));
            }
            i = j;
            continue;
        }
        // Single-issue guard coalescing (Lever 2): a maximal run of adjacent stmts that
        // each emit as `if {guard}: <body>` under the SAME single-issue guard is emitted
        // under ONE `if {guard}:` block, with every body indented inside it. The prologue
        // mbarrier inits are ~18 such adjacent stmts; one shared guard collapses 18
        // `elect.sync` + 18 predicated branches into 1. This is a generic peephole keyed
        // ONLY on guard-string identity (a structural property of the emitted stream), not
        // on any kernel/shape/op — every run of same-guard single-issue ops coalesces.
        if let Some(guard) = single_issue_guard(&collapsed[i], scope, ctx) {
            let mut j = i;
            while j < collapsed.len()
                && single_issue_guard(&collapsed[j], scope, ctx) == Some(guard)
            {
                j += 1;
            }
            if j - i >= 2 {
                out.push_str(&format!("{p}if {guard}:\n"));
                for s in &collapsed[i..j] {
                    emit_stmt(out, s, indent + 1, ctx, scope, true)?;
                }
                i = j;
                continue;
            }
        }
        emit_stmt(out, &collapsed[i], indent, ctx, scope, false)?;
        i += 1;
    }
    Ok(())
}

/// The single-issue guard string for a stmt that emits as `if {guard}: <one body>` under
/// `scope`, or `None` if the stmt is not a coalescable single-issue op. Drives the Lever-2
/// run coalescing in `emit_body`. The leader-routed `arrive.expect_tx` (extra `cbx == 0`
/// nesting) and warpgroup-collective stores are NOT coalescable here (they return `None`),
/// so they keep their own emission path. Generic: the guard comes from `scope.issue_guard`,
/// the same string every single-issue op uses.
fn single_issue_guard(stmt: &Stmt, scope: Scope, ctx: &Ctx) -> Option<&'static str> {
    use Stmt::*;
    match stmt {
        // The leader expect_tx nests under `cbx == 0` first — not a plain single guard.
        MBarrierArriveExpectTx { mbar, .. } if ctx.is_tma_leader_mbar(mbar.mbar.id) => None,
        MBarrierInit { .. }
        | MBarrierArriveExpectTx { .. }
        | MBarrierExpectTx { .. }
        | MBarrierArrive { .. }
        | StoreScalar { .. }
        | TmaLoad { .. }
        // Tcgen05Cp MUST coalesce with the adjacent Tcgen05Mma/Commit under ONE elect
        // block: each tcgen05 op (cp SFA, cp SFB, gemm, commit) in its OWN `if elect_sync()`
        // forces a warp reconvergence between them, which on Blackwell breaks the tcgen05
        // async issue stream and STALLS the block-scaled MMA (a GPU deadlock — the nvfp4
        // cluster gemm never retires, so tmem_full is never committed). Coalescing all the
        // consecutive same-guard tcgen05 ops into one elect (canon's `if elect_sync(): cp;
        // cp; gemm`) keeps the single issuing lane converged across the whole issue burst.
        | Tcgen05Cp { .. }
        | Tcgen05Mma { .. }
        | Tcgen05Commit { .. } => Some(scope.issue_guard()),
        _ => None,
    }
}

fn emit_stmt(
    out: &mut String,
    stmt: &Stmt,
    indent: usize,
    ctx: &Ctx,
    scope: Scope,
    // When true, the caller already opened the single-issue `if {guard}:` block (Lever-2
    // coalescing) and `indent` is the inner level: the single-issue guarded arms emit ONLY
    // their inner body, skipping their own `if {guard}:`. False = standalone emission
    // (each guarded arm opens its own guard, the original behaviour).
    bare: bool,
) -> Result<(), String> {
    use Stmt::*;
    let p = pad(indent);
    // Emit a single-issue guarded op's body. Standalone (`bare == false`): open the
    // `if {guard}:` here and indent the body once. Coalesced (`bare == true`): the caller
    // already opened the shared guard at `indent - 1`, so emit the body at `indent` with no
    // guard. `body` is the inner line(s) WITHOUT leading indent or trailing newline.
    let emit_guarded = |out: &mut String, body: &str| {
        if bare || scope == Scope::Elected {
            // Coalesced under a shared guard, or the whole role is already elected — emit
            // the single-issue body directly with no per-op `if guard:`.
            out.push_str(&format!("{p}{body}\n"));
        } else {
            out.push_str(&format!("{p}if {}:\n", scope.issue_guard()));
            out.push_str(&format!("{p}    {body}\n"));
        }
    };
    match stmt {
        // ---- REG fragments: emit INLINE at the TensorDef site (canon's loop-local
        // `T.alloc_local(...)`), NOT hoisted to function scope. A function-scope register
        // tile is whole-kernel-lived and ptxas keeps it in LOCAL memory (the LDL/STL spill);
        // declared inside the consuming loop it is task-local and promotes to registers. ----
        TensorDef { tensor } if tensor.space == MemorySpace::Reg => {
            let name = ctx.tensor_name(tensor.id)?.to_string();
            let width = ctx
                .reg_widths
                .get(&tensor.id)
                .copied()
                .filter(|w| *w > 0)
                .unwrap_or_else(|| tensor.shape.first().copied().unwrap_or(0));
            match &tensor.reg_frag {
                // STMATRIX epilogue datapath (canon's nvfp4 epilogue): the reg frag is
                // a `tcgen05.{ld,st}`-atom fragment, not a plain thread-axis tile, so the
                // reg->smem store lowers to STSM/stmatrix (vs the thread-axis tile's plain
                // STS, which carries 5.4x the SMEM bank conflicts). The read frag is
                // `alloc_tcgen05_ldst_frag(instr_shape, (128, W), dtype)`; the cast (output)
                // frag is `alloc_cast_frag(<read_frag>, dtype)`, inheriting the read frag's
                // (lane, register) layout so the f32->bf16 cast is a per-thread no-movement op.
                Some(super::tensor::RegFrag::Stmatrix {
                    instr_shape,
                    cast_of,
                }) => match cast_of {
                    None => {
                        out.push_str(&format!(
                                "{p}{name} = T.alloc_tcgen05_ldst_frag(\"{instr_shape}\", (128, {width}), \"{dt}\")\n",
                                p = pad(indent),
                                dt = dtype_str(tensor.dtype),
                            ));
                    }
                    Some(src_id) => {
                        let src_name = ctx.tensor_name(*src_id)?.to_string();
                        out.push_str(&format!(
                            "{p}{name} = T.alloc_cast_frag({src_name}, \"{dt}\")\n",
                            p = pad(indent),
                            dt = dtype_str(tensor.dtype),
                        ));
                    }
                },
                None => {
                    out.push_str(&format!(
                        "{p}{name} = T.wg_reg_tile({width}, dtype=\"{dt}\")\n",
                        p = pad(indent),
                        dt = dtype_str(tensor.dtype),
                    ));
                }
            }
            Ok(())
        }
        // ---- definitions handled in the header; skip in the body walk ----
        TensorDef { .. } | MBarDef { .. } => Ok(()),

        // ---- TMEM alloc / dealloc / relinquish (already under warp==0 role guard) ----
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
                "{p}T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols={n_cols}, cta_group={cta_group})\n"
            ));
            Ok(())
        }
        TmemRelinquish { .. } => {
            // 1:1 translation — the IR makes the permit release explicit (it used
            // to ride along with the dealloc implicitly).
            let cta_group = ctx.cta_group;
            out.push_str(&format!(
                "{p}T.ptx.tcgen05.relinquish_alloc_permit(cta_group={cta_group})\n"
            ));
            Ok(())
        }

        // ---- structural ----
        KernelInit { body, warp, .. } => {
            if let Some(w) = warp {
                out.push_str(&format!("{p}if warp_id == {w}:\n"));
                emit_body(out, body, indent + 1, ctx, Scope::Warp)?;
            } else {
                emit_body(out, body, indent, ctx, scope)?;
            }
            // Declare the single TMEM view buffer at function scope (mirrors the
            // template's line 490: `tmem` is visible to the MMA + epilogue, so it
            // must NOT be nested under the warp==0 alloc guard). TMEM is not a
            // tensor: one (128, cols) f32 view over the whole allocated band,
            // addressed everywhere by absolute column slices. allocated_addr=0
            // (not tmem_addr[0], the SMEM-stored alloc result): the single
            // tcgen05.alloc always bases at TMEM column 0, so the view address is
            // a compile-time constant — exactly canon's form; pinning it to 0 cuts
            // the epilogue address math (VIADD/LOP3/LDS) roughly in half.
            if let Some(cols) = ctx.tmem_view_cols {
                out.push_str(&format!(
                    "{p}tmem = T.decl_buffer((128, {cols}), \"float32\", scope=\"tmem\", allocated_addr=0, layout=TileLayout(S[(128, {cols}) : (1 @ TLane, 1 @ TCol)]))\n",
                ));
            }
            // NVFP4 e4m3 scale-factor TMEM views (canon's tmem_pool.alloc_sf(...,
            // sf_per_mma=4)). The TileLayout follows `sf_tmem_layout(rows, SF_K,
            // sf_per_mma=4)`: S[(M, 32, 4, 4) : (4 @ TCol, 1 @ TLane, M*4 @ TCol,
            // 1 @ TCol)] + R[4:32 @ TLane], M = rows // 32. The view's logical
            // (rows, SF_K) re-materializes the folded physical band the IR operand
            // addresses directly (a 256-row SFB folds into 2 column super-blocks).
            for view in &ctx.sf_views {
                let m_super = view.logical_rows / 32;
                out.push_str(&format!(
                    "{p}{name} = T.decl_buffer(({rows}, {cols}), \"float8_e4m3fn\", scope=\"tmem\", layout=TileLayout(S[({m_super}, 32, 4, 4) : (4 @ TCol, 1 @ TLane, {m_stride} @ TCol, 1 @ TCol)] + R[4:32 @ TLane]), allocated_addr={col})\n",
                    name = view.name,
                    rows = view.logical_rows,
                    cols = view.logical_cols,
                    m_super = m_super,
                    m_stride = m_super * 4,
                    col = view.col,
                ));
            }
            // KernelInit emits ONLY the warp-0 body (tmem alloc + mbarrier inits). The entire
            // prologue sync — the `fence.mbarrier_init` epoch seal AND the cross-CTA barrier
            // (cluster_barrier_arrive for overlap, or proxy_async + cluster_sync for no-overlap)
            // — is written EXPLICITLY in the nymph IR right after kernel_init. codegen never
            // fabricates a sync here and has no overlap/no-overlap knowledge.
            Ok(())
        }
        KernelFinalize { body, warp, .. } => {
            if let Some(w) = warp {
                out.push_str(&format!("{p}if warp_id == {w}:\n"));
                emit_body(out, body, indent + 1, ctx, Scope::Warp)?;
            } else {
                emit_body(out, body, indent, ctx, scope)?;
            }
            Ok(())
        }
        Role {
            body,
            warp,
            warpgroup,
            elected,
            maxnreg,
        } => {
            // The role's thread scope drives the single-issue guard: a warp role
            // elects `lane_id == 0` (1 thread), a warpgroup role `tid_in_wg == 0`
            // (1 thread of the 4-warp group — `lane_id == 0` would be 4 threads and
            // over-arrive every single-issue mbarrier in the branch).
            let (guard, body_scope) = if let Some(w) = warp {
                (format!("warp_id == {w}"), Scope::Warp)
            } else if let Some(wg) = warpgroup {
                (format!("wg_id == {wg}"), Scope::Warpgroup)
            } else {
                return Err("codegen: role without warp/warpgroup".to_string());
            };
            out.push_str(&format!("{p}if {guard}:\n"));
            if let Some(n) = maxnreg {
                // Per-warpgroup register budget (canon's `setmaxnreg(False, 56)` for the
                // producer warpgroup, `(True, 224)` for the consumer): rebalances registers
                // so the compute-heavy consumer gets more and the producer fewer, raising
                // occupancy. Warpgroup-aligned — emitted by the whole group at the role
                // start. inc when the budget rises above the 128-reg default, else dec.
                let inc = if *n > 128 { "True" } else { "False" };
                out.push_str(&format!("{p}    T.ptx.setmaxnreg({inc}, {n})\n"));
            }
            // A role is a single-warp / single-warpgroup branch. For the OVERLAP split
            // barrier the role body begins (after its local setup) with an explicit
            // `ClusterBarrierWait` IR stmt — emitted 1:1 by its own arm, no longer
            // synthesized here. Idle warps own no role and never reach a wait.
            if *elected {
                // ELECTED role: the WHOLE body runs on one thread — `if <guard>:` then a
                // single role-wide elect, matching canon's `if elect_sync(): while ...:`
                // scheduler/loader/MMA loops. Inside, single-issue ops drop their per-op
                // guard (Scope::Elected). One issuing thread runs the loop + its mbar waits
                // (no 32-thread spin, no per-op elect), so the timing matches canon.
                //
                // EXCEPT a LEADING `ClusterBarrierWait`: `barrier.cluster.wait` is a
                // WARP-COLLECTIVE op and DEADLOCKS if only the elected lane waits — canon
                // emits it by ALL threads of the warp, BEFORE `if elect_sync()`. So peel any
                // leading cluster-barrier waits and emit them at warp scope (outside the
                // elect), then run the rest of the body single-issue. (Verified: the
                // elect-only wait hung the overlap path on 1024/2048; all-thread wait runs.)
                let n_lead = body
                    .iter()
                    .take_while(|s| matches!(s, Stmt::ClusterBarrierWait))
                    .count();
                for _ in 0..n_lead {
                    out.push_str(&format!(
                        "{p}    T.ptx.barrier.cluster.wait(acquire=True, aligned=False)\n"
                    ));
                }
                out.push_str(&format!("{p}    if {}:\n", body_scope.issue_guard()));
                emit_body(out, &body[n_lead..], indent + 2, ctx, Scope::Elected)?;
            } else {
                emit_body(out, body, indent + 1, ctx, body_scope)?;
            }
            Ok(())
        }
        If { cond, then_body } => {
            out.push_str(&format!("{p}if {}:\n", emit_scalar(cond, ctx)?));
            emit_body(out, then_body, indent + 1, ctx, scope)?;
            Ok(())
        }
        ForLoop {
            var,
            start,
            stop,
            step,
            body,
            unroll,
        } => {
            let name = var_name(ctx, var);
            let start_s = emit_scalar(start, ctx)?;
            let stop_s = emit_scalar(stop, ctx)?;
            let step_s = emit_scalar(step, ctx)?;
            // `unroll`: emit `T.unroll(N)` — a compile-time-unrolled `for` (same SASS as a
            // manual unroll, but written as a loop in the source, matching canon's
            // `for i in T.unroll(N)` for fixed-count loops). The emitted form only
            // expresses start=0, step=1 — anything else would be silently re-timed, so
            // reject it instead of emitting a loop with different trip semantics.
            if *unroll {
                if as_int(start) != Some(0) || as_int(step) != Some(1) {
                    return Err(format!(
                        "codegen: ForLoop unroll requires literal start=0, step=1 \
                         (got start={start_s}, step={step_s})"
                    ));
                }
                out.push_str(&format!("{p}for {name} in T.unroll({stop_s}):\n"));
                emit_body(out, body, indent + 1, ctx, scope)?;
                return Ok(());
            }
            // Emit `T.serial(...)`, NOT Python `range(...)`: T.serial is the rolled serial
            // loop with a single UNIFORM loop-induction var (one hardware counter per
            // warp), which is what keeps the ring index / operand-descriptor address math
            // in uniform registers (UIADD3/UMOV) — matching canon's `for k in T.serial(N)`.
            // A bare `range(...)` would be fully unrolled by the TVMScript parser, defeating
            // the roll (and re-exploding the 16384 compile). step must be 1 for T.serial.
            let range = if step_s == "1" {
                if start_s == "0" {
                    format!("T.serial({stop_s})")
                } else {
                    format!("T.serial({start_s}, {stop_s})")
                }
            } else {
                format!("T.serial({start_s}, {stop_s}, {step_s})")
            };
            out.push_str(&format!("{p}for {name} in {range}:\n"));
            emit_body(out, body, indent + 1, ctx, scope)?;
            Ok(())
        }
        // Grid-stride scheduler loop: each launched cluster (cta_id // cta_group) strides
        // by the cluster count through the task space. The trip count is runtime (the
        // start is runtime), so T.serial stays rolled — it cannot unroll. The task var
        // reads as `v{id}` in the body (the per-role `local_iter`/`work_idx` math the
        // kernel emits decodes it). Only the functional grid_stride policy is lowered;
        // the work-stealing CLC path is written as explicit clc_* ops, not ForEachTask.
        ForEachTask {
            scheduler,
            var,
            body,
        } => {
            use super::scheduler::SchedulerPolicy;
            if scheduler.policy != SchedulerPolicy::GridStride {
                return Err(format!(
                    "codegen: ForEachTask policy {:?} unsupported (only grid_stride)",
                    scheduler.policy
                ));
            }
            let name = var_name(ctx, var);
            let stop = scheduler
                .space
                .task_count()
                .ok_or("codegen: ForEachTask task space size overflow")?;
            // Grid-stride as a while loop (T.serial has no step): the task var starts at
            // this cluster's index and strides by the cluster count until the task space
            // is exhausted. The trip count is runtime, so it cannot unroll.
            out.push_str(&format!(
                "{p}{name}: T.int32 = cta_id // {cg}\n",
                cg = ctx.cta_group,
            ));
            out.push_str(&format!("{p}while {name} < {stop}:\n"));
            emit_body(out, body, indent + 1, ctx, scope)?;
            out.push_str(&format!(
                "{p}    {name} = {name} + {step}\n",
                step = ctx.num_clusters,
            ));
            Ok(())
        }
        // The persistent / scheduler `while True:` loop. Break is via `BreakIf`.
        Loop { body } => {
            out.push_str(&format!("{p}while True:\n"));
            emit_body(out, body, indent + 1, ctx, scope)?;
            Ok(())
        }
        // `scheduler_impl` is a sim/checker marker for the trusted scheduler region;
        // it carries no code of its own, so codegen emits its body transparently
        // (same indent + scope) — the CLC primitives inside translate 1:1.
        SchedulerImpl { body, .. } => {
            emit_body(out, body, indent, ctx, scope)?;
            Ok(())
        }
        BreakIf { cond } => {
            out.push_str(&format!("{p}if {}:\n", emit_scalar(cond, ctx)?));
            out.push_str(&format!("{p}    break\n"));
            Ok(())
        }

        // ---- scalar SSA register vars (`NAME: T.int32 = ...`, read as NAME) ----
        ScalarDef { var, initial } => {
            let name = ctx
                .scalar_names
                .get(&var.id.0)
                .ok_or_else(|| format!("codegen: no name for scalar var {}", var.id.0))?;
            let init = match initial {
                ScalarInitial::Value(v) => emit_scalar(v, ctx)?,
                // Mailbox load: read the 1-element SMEM slice (drops to a scalar load).
                ScalarInitial::Tensor(ts) => emit_scalar_load(ts, ctx)?,
            };
            // Emit an SSA `T.int32` register var (canon's `sa_stage: T.int32 = …`), NOT a
            // `T.alloc_local(1)` cell. The local-array form can defeat ptxas's uniform-register
            // analysis (the warp-uniform counters lower to VECTOR regs + R2UR + LDL/STL spill);
            // a plain typed var lowers to a uniform register (UIADD3) like canon.
            out.push_str(&format!("{p}{name}: T.int32 = {init}\n", name = name));
            Ok(())
        }
        ShuffleSync { var, src, src_lane } => {
            let name = ctx
                .scalar_names
                .get(&var.id.0)
                .ok_or_else(|| format!("codegen: no name for scalar var {}", var.id.0))?
                .clone();
            // Warp broadcast (lane `src_lane` -> all lanes of the warp). The result is
            // compiler-provably uniform, so the index/address chain derived from it
            // lowers to the uniform datapath instead of vector + R2UR.
            out.push_str(&format!(
                "{p}{name}: T.int32 = T.cuda.__shfl_sync(0xffffffff, {src}, {lane}, 32)\n",
                src = emit_scalar(src, ctx)?,
                lane = emit_scalar(src_lane, ctx)?,
            ));
            Ok(())
        }
        // CLC async work-steal issue: 1:1 to `clusterlaunchcontrol.try_cancel`. The
        // 16B response lands in `handle`; the second arg is the mbar it completes-tx
        // (the kernel's own `sched_arr` full barrier). Single-issue (the role's elect
        // guard), exactly like canon's `if T.ptx.elect_sync(): clc_try_cancel(...)`.
        ClcTryCancel {
            handle,
            mbar,
            stage,
            ..
        } => {
            let handle_name = ctx.tensor_name(handle.id)?;
            // Both args are `T.address_of(buf[i])` — exactly canon's
            // `clc_try_cancel(T.address_of(clc_handle[0]), T.address_of(sched_arr.full.buf[0]))`.
            // The mbar arg must be `address_of(name[slot])`, NOT `name.ptr_to([slot])`:
            // try_cancel writes the completion to that raw address, and the buffer-pointer
            // form lowers differently (wrong target → multi-tile fault).
            let mbar_name = ctx
                .mbar_names
                .get(&mbar.mbar.id)
                .cloned()
                .ok_or_else(|| format!("codegen: no name for mbar {}", mbar.mbar.id))?;
            let slot = stage
                .as_ref()
                .map(|s| emit_scalar(s, ctx))
                .transpose()?
                .unwrap_or_else(|| "0".to_string());
            emit_guarded(
                out,
                &format!(
                    "T.ptx.clc_try_cancel(T.address_of({handle_name}[0]), T.address_of({mbar_name}[{slot}]))"
                ),
            );
            Ok(())
        }
        // CLC handle decode: 1:1 to `clusterlaunchcontrol.query_cancel`, DEFINING the
        // scalar (the cancelled cluster's first ctaid.x, or 0xFFFFFFFF -> -1 as int32).
        // Unguarded — every thread of the cohort reads the same handle and gets the
        // same value (a pure uniform decode), like `ShuffleSync`.
        ClcQueryCancel { var, handle, .. } => {
            let name = ctx
                .scalar_names
                .get(&var.id.0)
                .ok_or_else(|| format!("codegen: no name for scalar var {}", var.id.0))?
                .clone();
            let handle_name = ctx.tensor_name(handle.id)?;
            out.push_str(&format!(
                "{p}{name}: T.int32 = T.ptx.clc_query_cancel(T.address_of({handle_name}[0]))\n"
            ));
            Ok(())
        }
        ScalarStore { var, value } => {
            let name = ctx
                .scalar_names
                .get(&var.id.0)
                .ok_or_else(|| format!("codegen: no name for scalar var {}", var.id.0))?;
            // Reassign the SSA register var (canon's `sa_stage = …`), no `[0]` cell index.
            out.push_str(&format!(
                "{p}{name} = {}\n",
                emit_scalar(value, ctx)?,
                name = name
            ));
            Ok(())
        }
        // Mailbox write: `task_smem[stage, field] = <scalar>`. Single-issue (lane 0 of
        // the scheduler warp) — the value is uniform, so one writer suffices and avoids
        // 32 redundant STS to the same address.
        StoreScalar { dst, value } => {
            let dst_s = emit_scalar_addr(dst, ctx)?;
            emit_guarded(out, &format!("{dst_s} = {}", emit_scalar(value, ctx)?));
            Ok(())
        }

        // ---- mbarrier (every op carries an optional `stage` -> slot index) ----
        MBarrierInit { mbar, count, stage } => {
            let slot_ptr = mbar_slot_ptr(mbar, stage, ctx)?;
            emit_guarded(out, &format!("T.ptx.mbarrier.init({slot_ptr}, {count})"));
            Ok(())
        }
        MBarrierArriveExpectTx { mbar, bytes, stage } => {
            // Cluster TMA barrier (leader-routed): issue ONE expect_tx on the leader's
            // (CTA-0) barrier for the FULL cluster byte count (both CTAs' loads land
            // here), and only on the leader CTA (cbx==0) so it is counted once. The IR's
            // `bytes` is the per-CTA byte count, so multiply by cta_group.
            if let Some(view) = ctx.tma_leader_view_for(mbar.mbar.id) {
                let slot = stage
                    .as_ref()
                    .map(|s| emit_scalar(s, ctx))
                    .transpose()?
                    .unwrap_or_else(|| "0".to_string());
                let total_bytes = *bytes as u64 * ctx.cta_group as u64;
                // Nest the CTA selector and the single-issue guard as separate `if`s
                // rather than `(cbx == 0) and (guard)`: the warp/function guard is now
                // `T.ptx.elect_sync()`, which returns a uint32 (not bool), and `tirx.And`
                // requires both operands be bool. `cbx == 0` is warp-uniform, so nesting
                // is equivalent (all lanes take the same branch, then elect one).
                out.push_str(&format!("{p}if cbx == 0:\n"));
                out.push_str(&format!("{p}    if {}:\n", scope.issue_guard()));
                out.push_str(&format!(
                    "{p}        T.ptx.mbarrier.arrive.expect_tx({view}.ptr_to([{slot}]), {total_bytes})\n"
                ));
                return Ok(());
            }
            let slot_ptr = mbar_slot_ptr(mbar, stage, ctx)?;
            emit_guarded(
                out,
                &format!("T.ptx.mbarrier.arrive.expect_tx({slot_ptr}, {bytes})"),
            );
            Ok(())
        }
        MBarrierExpectTx { mbar, bytes, stage } => {
            let slot_ptr = mbar_slot_ptr(mbar, stage, ctx)?;
            emit_guarded(
                out,
                &format!("T.ptx.mbarrier.expect_tx({slot_ptr}, {bytes})"),
            );
            Ok(())
        }
        MBarrierArrive { mbar, count, stage } => {
            // Two arrive forms (see `ptx_mbarrier_arrive`):
            //   * LOCAL (remote_coord=None): the implicit count-of-1 form
            //     `T.ptx.mbarrier.arrive(bar)`. (The 2nd positional arg is `remote`,
            //     NOT a count — so a count must never be passed positionally here.)
            //   * CROSS-CTA (remote_coord=Some(c)): the cluster form on the LOCAL
            //     barrier of CTA `c`: `T.ptx.mbarrier.arrive(bar, remote=c, pred=True)`
            //     — the canonical `tmem_pipe.empty.arrive(slot, remote=0, pred=True)`.
            //     This is NOT the map_shared_rank peer view; the cluster arrive remaps
            //     to CTA c internally, so we use the local mbar name + cta_id.
            //
            // The guard elects ONE issuing thread of the enclosing role. A warpgroup
            // role MUST elect `tid_in_wg == 0` (not `lane_id == 0`, which is 4 threads
            // across the group's 4 warps): the epilogue warpgroups arrive on the SMEM
            // task-mailbox `task_empty` and the cross-CTA `tmem_empty`, and 4× over-
            // arrival corrupts the barrier phase. It is latent on a single task (the
            // reused mailbox/tmem slot is never re-waited) but deadlocks the moment a
            // slot is reused across tasks (pair_tasks > broadcast/tmem stages).
            let body = if let Some(remote) = &mbar.remote_coord {
                // Use the LOCAL barrier name (not the peer reinterpret view).
                let local_name = ctx
                    .mbar_names
                    .get(&mbar.mbar.id)
                    .cloned()
                    .ok_or_else(|| format!("codegen: no name for mbar {}", mbar.mbar.id))?;
                let slot = stage
                    .as_ref()
                    .map(|s| emit_scalar(s, ctx))
                    .transpose()?
                    .unwrap_or_else(|| "0".to_string());
                format!(
                    "T.ptx.mbarrier.arrive({local_name}.ptr_to([{slot}]), remote={cta}, pred=True)",
                    cta = emit_scalar(remote, ctx)?,
                )
            } else {
                let slot_ptr = mbar_slot_ptr(mbar, stage, ctx)?;
                let cnt = as_int(count).unwrap_or(1);
                if cnt == 1 {
                    format!("T.ptx.mbarrier.arrive({slot_ptr})")
                } else {
                    // A local arrive with an explicit count>1 has no implicit form;
                    // none occur in this kernel. Emit the count via the cluster form on
                    // the local CTA (cta_id read from the runtime scope).
                    format!("T.ptx.mbarrier.arrive({slot_ptr}, remote=cbx, pred=True, count={cnt})")
                }
            };
            emit_guarded(out, &body);
            Ok(())
        }
        MBarrierWait { mbar, phase, stage } => {
            // A peer (remote_coord) wait is skipped — illegal on a remapped DSMEM
            // address; the peer's TMA is ordered via the leader-routed smem_full instead
            // (see the coalescing note in `emit_body`). A local wait emits try_wait + fence.
            if mbar.remote_coord.is_some() {
                return Ok(());
            }
            let slot_ptr = mbar_slot_ptr(mbar, stage, ctx)?;
            let phase_s = phase
                .as_ref()
                .map(|p| emit_scalar(p, ctx))
                .transpose()?
                .unwrap_or_else(|| "0".to_string());
            out.push_str(&format!(
                "{p}T.ptx.mbarrier.try_wait({slot_ptr}, {phase_s})\n"
            ));
            // NO `tcgen05.fence.after_thread_sync()` here: the canonical kernel emits
            // ZERO such fences after its mbar waits (it orders tcgen05 via the mbar
            // handshake itself + a proxy_async fence only in the epilogue). Emitting one
            // after every wait over-fences the hot loop (perf) and — critically — delays
            // the scheduler's `expect_tx` relative to the async CLC multicast tx, which
            // races the tx ahead and faults the peer CTA's barrier.
            Ok(())
        }

        // ---- TMA ----
        TmaLoad {
            dst,
            src,
            mbar,
            coords,
            shape,
            gmem_shape,
            mbar_stage,
            multicast_cta_mask,
            cache_hint,
            prefetch_tensormap,
            cta_group,
            ..
        } => {
            // The emitted `Tx.copy_async(..., cta_group=)` uses the KERNEL-level engine
            // group (`ctx.cta_group`, from the cluster size). A per-op override the IR
            // carries but codegen would silently drop is "validate one semantics, run
            // another" — reject the mismatch instead.
            if *cta_group != ctx.cta_group {
                return Err(format!(
                    "codegen: TmaLoad cta_group={} != kernel cta_group={}",
                    cta_group, ctx.cta_group
                ));
            }
            // The completion mbarrier indexes the ring slot it signals. In cluster mode
            // the TMA-load barrier is leader-routed: BOTH CTAs signal the LEADER's
            // (CTA-0) barrier via its `_cta0` map_shared_rank(.., 0) view (identity on
            // CTA 0, the remap on CTA 1), so the leader's MMA can wait its own local
            // barrier instead of an (illegal) peer try_wait.
            let mbar_name = ctx
                .tma_leader_view_for(mbar.mbar.id)
                .map(Ok)
                .unwrap_or_else(|| mbar_buf_name(mbar, ctx))?;
            let mbar_slot = mbar_stage
                .as_ref()
                .map(|s| emit_scalar(s, ctx))
                .transpose()?
                .unwrap_or_else(|| "0".to_string());
            // The SMEM dst is a staged tile (a leading size-1 ring dim): drop it to an
            // integer index so the operand rank matches the 2D GMEM region.
            let dst_s = emit_smem_tile(dst, ctx)?;
            let src_s = emit_gmem_region(src, coords, gmem_extents(gmem_shape, shape), ctx)?;
            // `multicast_cta_mask`: a `multicast::cluster` g2c copy — one TMA fills the
            // SMEM of EVERY CTA in the mask (canon's `cta_mask=pair_mask` for the shared
            // SFB scale band), so the cluster shares ONE load instead of each CTA reading
            // the full band (halving the L2/TMA traffic). The completion's transaction
            // count is added per multicast destination to the (leader-routed) barrier,
            // which the `* cta_group` factor in the leader expect_tx already accounts for.
            let cta_mask = match multicast_cta_mask {
                Some(mask) => format!(", cta_mask={mask}"),
                None => String::new(),
            };
            // `cache_hint`: the per-load L2 eviction policy (canon's `cache_hint` on its
            // g2c loads). When opted in (e.g. `"evict_normal"`) a tile read once per
            // k-tile does not pin an L2 line the next tile evicts anyway — bounding the
            // L2 cache-policy traffic, the lever that stops the full-cube launch fault.
            // None = no hint (the codegen-default policy). Generic: any kernel/load
            // chooses its own hint via the IR.
            let cache_hint_kw = match cache_hint {
                Some(hint) => format!(", cache_hint=\"{hint}\""),
                None => String::new(),
            };
            // `prefetch_tensormap` (IR-carried; the canonical prefetches the A/B
            // tensormaps at entry — a `warp_id_in_cta==0`-guarded `prefetch.tensormap`,
            // synthesized by the TMA dispatch from this config flag). On the
            // latency-bound small shapes this hides the first descriptor fetch behind
            // the prologue.
            let prefetch_kw = if *prefetch_tensormap {
                ", prefetch_tensormap=True"
            } else {
                ""
            };
            emit_guarded(
                out,
                &format!(
                    "Tx.copy_async({dst_s}, {src_s}, dispatch=\"tma\", mbar={mbar_name}.ptr_to([{mbar_slot}]), cta_group={cg}{cta_mask}{cache_hint_kw}{prefetch_kw})",
                    cg = ctx.cta_group,
                ),
            );
            Ok(())
        }

        // ---- tcgen05 MMA ----
        Tcgen05Mma {
            dst,
            a,
            b,
            m,
            n,
            k,
            accum,
            sfa,
            sfb,
            a_fp4,
            b_fp4,
            trans_a,
            trans_b,
            cta_group,
            sf_byte,
            sf_e4m3,
            sf_block,
            lane_align,
            ..
        } => {
            // `Tx.gemm_async` fixes the operand convention the slices are emitted in:
            // A=(M,K), B=(N,K), full-datapath accumulator, kernel-level cta_group. Any
            // IR field the emitted form cannot represent is rejected — the validator
            // and value model honor these fields, so dropping one here would run a
            // different semantics than was verified. (`m/n/k` vs the operand slice
            // shapes is already enforced by the validator, incl. the trans variants.)
            if *trans_a || *trans_b {
                return Err(
                    "codegen: Tcgen05Mma trans_a/trans_b have no gemm_async lowering".to_string(),
                );
            }
            if *lane_align != 0 {
                return Err(
                    "codegen: Tcgen05Mma lane_align != 0 (m=64 Layout F) not supported".to_string(),
                );
            }
            if *cta_group != ctx.cta_group {
                return Err(format!(
                    "codegen: Tcgen05Mma cta_group={} != kernel cta_group={}",
                    cta_group, ctx.cta_group
                ));
            }
            if *a_fp4 != *b_fp4 {
                return Err("codegen: Tcgen05Mma a_fp4/b_fp4 must match".to_string());
            }
            match (sfa.as_ref(), sfb.as_ref()) {
                // Block-scaled path: the emitted SFA/SFB slice form is the NVFP4
                // block-16 e4m3 layout (BASE_SF_K=16, byte 0..k/16 per cell) — the
                // only mode the SF slice math below encodes.
                (Some(_), Some(_)) => {
                    if !*sf_e4m3 || *sf_block != 16 || *sf_byte != 0 {
                        return Err(format!(
                            "codegen: block-scaled Tcgen05Mma supports only the NVFP4 mode \
                             (sf_e4m3=true, sf_block=16, sf_byte=0); got sf_e4m3={sf_e4m3}, \
                             sf_block={sf_block}, sf_byte={sf_byte}"
                        ));
                    }
                }
                (None, None) => {
                    if *a_fp4 {
                        return Err("codegen: fp4 Tcgen05Mma operands require sfa/sfb".to_string());
                    }
                }
                _ => {
                    return Err("codegen: Tcgen05Mma sfa/sfb must be set together".to_string());
                }
            }
            let dst_s = emit_tmem_dst(dst, *n, ctx)?;
            // The A/B operands are staged SMEM tiles: drop the leading ring index so
            // the operand is the 2D `(M, K)` / `(N, K)` MMA tile (canonical
            // `Asmem[stage, warp_id]` / `Bsmem[stage]`). A TMEM operand (the GDN
            // accumulator-readback) is an absolute (lane, col) slice of the single
            // `tmem` view.
            let emit_ab = |op: &MmaOperand, rows: u32| -> Result<String, String> {
                match op {
                    MmaOperand::Slice(s) => emit_smem_tile(s, ctx),
                    MmaOperand::Tmem(t) => {
                        let row_s = emit_scalar(&t.row, ctx)?;
                        let col_s = emit_scalar(&t.col, ctx)?;
                        let row_hi = add_bound(&t.row, &ScalarValue::Int(i64::from(rows)), ctx)?;
                        let cells = match t.dtype {
                            DType::F16 | DType::Bf16 => *k as i64 / 2,
                            _ => i64::from(*k),
                        };
                        let col_hi = add_bound(&t.col, &ScalarValue::Int(cells), ctx)?;
                        Ok(format!("tmem[{row_s}:{row_hi}, {col_s}:{col_hi}]"))
                    }
                }
            };
            let a_rows = if *cta_group == 1 { *m } else { *m / 2 };
            let b_rows = if *cta_group == 1 { *n } else { *n / 2 };
            let mut a_s = emit_ab(a, a_rows)?;
            let mut b_s = emit_ab(b, b_rows)?;
            let accum_s = if *accum { "True" } else { "False" };
            if let (Some(sfa), Some(sfb)) = (sfa.as_ref(), sfb.as_ref()) {
                // NVFP4 block-scaled: view the packed-u8 operand BUFFER as e2m1 fp4 (the
                // last dim doubles: bytes -> fp4 elems), then slice — `.view` is on the
                // buffer, not a region (canon: A_smem = A_smem_packed.view(...); A_smem[stage]).
                if *a_fp4 {
                    let view = |op: &MmaOperand| -> Result<String, String> {
                        let MmaOperand::Slice(s) = op else {
                            return Err(
                                "codegen: fp4 Tcgen05Mma operands must be SMEM slices".to_string()
                            );
                        };
                        let buf = ctx.tensor_name(s.tensor.id)?;
                        let stage =
                            emit_scalar(s.offsets.first().unwrap_or(&ScalarValue::Int(0)), ctx)?;
                        let rows = s.shape.get(1).and_then(as_int).unwrap_or(0);
                        let cols = s.shape.get(2).and_then(as_int).unwrap_or(0) * 2;
                        Ok(format!(
                            "{buf}.view(\"float4_e2m1fn\")[{stage}, 0:{rows}, 0:{cols}]"
                        ))
                    };
                    a_s = view(a)?;
                    b_s = view(b)?;
                }
                // Emit the SF operands as EXPLICIT logical slices `[0:rows, 0:cols]`
                // of their views, exactly like canon (`SFA_tmem[0:128, 0:16]`,
                // `SFB_tmem[0:256, 0:16]`) — the view at the operand's physical base
                // column re-materializes the folded logical shape.
                let sf_slice = |op: &TmemOperand| -> Result<String, String> {
                    let Some(col) = as_int(&op.col) else {
                        return Err(
                            "codegen: SF TMEM operand col must be a compile-time constant"
                                .to_string(),
                        );
                    };
                    let view = ctx
                        .sf_views
                        .iter()
                        .find(|v| v.col as i64 == col)
                        .ok_or_else(|| format!("codegen: no SF TMEM view at col {col}"))?;
                    Ok(format!(
                        "{}[0:{}, 0:{}]",
                        view.name, view.logical_rows, view.logical_cols
                    ))
                };
                let sfa_s = sf_slice(sfa)?;
                let sfb_s = sf_slice(sfb)?;
                emit_guarded(
                    out,
                    &format!(
                        "Tx.gemm_async({dst_s}, {a_s}, {b_s}, SFA={sfa_s}, SFB={sfb_s}, accum={accum_s}, dispatch=\"tcgen05\", cta_group={cg})",
                        cg = ctx.cta_group,
                    ),
                );
            } else {
                emit_guarded(
                    out,
                    &format!(
                        "Tx.gemm_async({dst_s}, {a_s}, {b_s}, accum={accum_s}, dispatch=\"tcgen05\", cta_group={cg})",
                        cg = ctx.cta_group,
                    ),
                );
            }
            Ok(())
        }
        // NVFP4 scale-factor copy SMEM -> e4m3 TMEM (canon's Tx.copy_async(SFA_tmem,
        // SFA_smem[stage], cta_group=2)). Single-issue, like the MMA/commit.
        Tcgen05Cp {
            dst,
            src,
            cta_group,
        } => {
            // Emit canon's EXACT cp form: `Tx.copy_async(SFB_tmem[0:256, 0:16],
            // SFB_smem[stage, 0:256, 0:16], cta_group=2)` — explicit logical slices
            // on BOTH ends. The dst's view comes from its physical base column;
            // the src is the staged SF SMEM `(rows, SF_CTA_K)`.
            let Some(col) = as_int(&dst.col) else {
                return Err(
                    "codegen: tcgen05_cp dst col must be a compile-time constant".to_string(),
                );
            };
            let view = ctx
                .sf_views
                .iter()
                .find(|v| v.col as i64 == col)
                .ok_or_else(|| format!("codegen: no SF TMEM view at col {col}"))?;
            let src_name = ctx.tensor_name(src.tensor.id)?;
            let stage = emit_scalar(src.offsets.first().unwrap_or(&ScalarValue::Int(0)), ctx)?;
            // src is a staged SF SMEM tile `(SMEM_DEPTH, rows, SF_CTA_K)`; its full
            // (rows, SF_CTA_K) at this stage — the trailing slice dims.
            let (Some(r), Some(c)) = (
                src.shape
                    .get(src.shape.len().saturating_sub(2))
                    .and_then(as_int),
                src.shape.last().and_then(as_int),
            ) else {
                return Err("codegen: tcgen05_cp src shape must be static".to_string());
            };
            emit_guarded(
                out,
                &format!(
                    "Tx.copy_async({name}[0:{rows}, 0:{cols}], {src_name}[{stage}, 0:{r}, 0:{c}], cta_group={cta_group})",
                    name = view.name,
                    rows = view.logical_rows,
                    cols = view.logical_cols,
                ),
            );
            Ok(())
        }
        Tcgen05Commit {
            mbar,
            multicast_cta_mask,
            stage,
            cta_group,
        } => {
            if *cta_group != ctx.cta_group {
                return Err(format!(
                    "codegen: Tcgen05Commit cta_group={} != kernel cta_group={}",
                    cta_group, ctx.cta_group
                ));
            }
            let slot_ptr = mbar_slot_ptr(mbar, stage, ctx)?;
            let mask = multicast_cta_mask.unwrap_or(0);
            emit_guarded(
                out,
                &format!(
                    "T.ptx.tcgen05.commit({slot_ptr}, cta_group={cg}, cta_mask={mask})",
                    cg = ctx.cta_group,
                ),
            );
            Ok(())
        }

        // ---- epilogue: tcgen05_ld -> Tx.wg.copy_async, wait_ld, reg_cvt -> Tx.wg.cast,
        // reg_store -> Tx.copy. Whole warpgroup participates (no per-thread guard). ----
        Tcgen05Ld { dst, src, num, .. } => {
            // dst is the f32 reg fragment (read as a wg view of `num` cols); src is the
            // tmem band at the operand's absolute physical `col`. The fragment is
            // filled from column 0 (scratch reused per drain group), so the view
            // slice starts at 0.
            let width = *num as usize;
            let _ = emit_scalar(&src.row, ctx)?; // row is the lane axis, captured by the view
            let col_s = emit_scalar(&src.col, ctx)?;
            // Read this atom into its sub-slice of the (wide) read fragment: the
            // epilogue issues a whole band's atoms into distinct slices, THEN a
            // single wait_ld (matching the canonical fence structure). A legacy
            // single-atom drain passes offset 0, so this is unchanged for it.
            let zero = ScalarValue::Int(0);
            let dst_off = dst.offsets.first().unwrap_or(&zero);
            let frag_s = emit_reg_view_slice(out, &p, &dst.tensor, dst_off, width, ctx)?;
            out.push_str(&format!(
                "{p}Tx.wg.copy_async({frag_s}, tmem[:, {col_s}:{col_s} + {width}])\n"
            ));
            Ok(())
        }
        Tcgen05WaitLd => {
            out.push_str(&format!("{p}T.ptx.tcgen05.wait.ld()\n"));
            Ok(())
        }
        RegCvt { dst, src, .. } => {
            // Cast a band of the f32 read fragment to the matching band of the wide
            // output reg. Both bands are sliced by their (offset, width) so a capped
            // (≤128-col) drain group writes the right slice of the 256-wide output reg.
            let zero = ScalarValue::Int(0);
            let dst_off = dst.offsets.first().unwrap_or(&zero);
            let src_off = src.offsets.first().unwrap_or(&zero);
            let dst_w = dst.shape.first().and_then(as_int).unwrap_or(0).max(0) as usize;
            let src_w = src.shape.first().and_then(as_int).unwrap_or(0).max(0) as usize;
            let dst_s = emit_reg_view_slice(out, &p, &dst.tensor, dst_off, dst_w, ctx)?;
            let src_s = emit_reg_view_slice(out, &p, &src.tensor, src_off, src_w, ctx)?;
            out.push_str(&format!("{p}Tx.wg.cast({dst_s}, {src_s})\n"));
            Ok(())
        }
        // SF-permute load (nvfp4 dev permute warp): read an SMEM scale-cell band into
        // a per-lane register fragment, symmetric to the SMEM branch of `RegStore`. The
        // byte permutation itself is below the value model — codegen only materializes
        // the buffer read into the fragment via the warpgroup-collective copy (the
        // matching `RegStore` writes it back). REG-source loads (an alias/copy between
        // two register tiles) are a structural no-op at the source level.
        RegLoad { dst, src } => {
            if src.tensor.space == MemorySpace::Smem {
                let src_s = emit_smem_wg_store_tile(src, ctx)?;
                let zero = ScalarValue::Int(0);
                let dst_off = dst.offsets.first().unwrap_or(&zero);
                let width = dst.shape.first().and_then(as_int).unwrap_or(0).max(0) as usize;
                let dst_s = emit_reg_view_slice(out, &p, &dst.tensor, dst_off, width, ctx)?;
                out.push_str(&format!("{p}Tx.wg.copy({dst_s}, {src_s})\n"));
            }
            Ok(())
        }
        RegStore { dst, src } => {
            // Two store shapes, distinguished by the dst memory space:
            //   * GMEM dst (bootstrap direct epilogue): `Tx.copy(C[row, c0:c1], reg[:])`
            //     — each thread writes its own C row from its private fragment.
            //   * SMEM dst (smem-staged epilogue): `Tx.wg.copy(d_smem[db, :, c0:c1],
            //     reg_view[:, b0:b1])` — the warpgroup-collective reg->smem store that
            //     feeds the subsequent TMA store. The reg src is the wide `out_wide`
            //     band, read through its wg view (sliced by column).
            if dst.tensor.space == MemorySpace::Smem {
                let dst_s = emit_smem_wg_store_tile(dst, ctx)?;
                let zero = ScalarValue::Int(0);
                let src_off = src.offsets.first().unwrap_or(&zero);
                let width = src.shape.first().and_then(as_int).unwrap_or(0).max(0) as usize;
                let src_s = emit_reg_view_slice(out, &p, &src.tensor, src_off, width, ctx)?;
                // STMATRIX epilogue: a `tcgen05.{ld,st}`-atom src frag stores reg->smem via
                // `dispatch="ldstmatrix"` (canon's `regs_to_smem`), lowering to STSM. The
                // kernel slices the store in 16-col chunks (stmatrix.x4 granularity). A plain
                // thread-axis frag (reg_frag=None) keeps the default `Tx.wg.copy` (STS).
                let dispatch = if src.tensor.reg_frag.is_some() {
                    ", dispatch=\"ldstmatrix\""
                } else {
                    ""
                };
                out.push_str(&format!("{p}Tx.wg.copy({dst_s}, {src_s}{dispatch})\n"));
            } else {
                // GMEM store (bootstrap direct epilogue): the warpgroup-collective
                // `Tx.wg.copy(C[row:row+128, c0:c1], reg[:, :])`. The reg fragment is a
                // `wg_reg_tile` (thread-axis layout), so the copy MUST be warpgroup-
                // scoped — a thread-scoped `Tx.copy` falls to the scalar fallback,
                // which does a direct thread-axis BufferLoad and is rejected by
                // LowerTIRxCleanup. The dst row offset carries a per-thread
                // `tid_in_wg` term in the value model; for the wg-collective store it
                // becomes the 128-row band (the layout maps lanes to rows).
                let dst_s = emit_gmem_row_store(dst, ctx)?;
                let zero = ScalarValue::Int(0);
                let w = src.shape.first().and_then(as_int).unwrap_or(0).max(0) as usize;
                let src_s = emit_reg_view_slice(out, &p, &src.tensor, &zero, w, ctx)?;
                out.push_str(&format!("{p}Tx.wg.copy({dst_s}, {src_s})\n"));
            }
            Ok(())
        }

        // ---- smem-staged epilogue: reg->smem store + TMA store + bulk-group pacing ----
        TmaStore {
            dst,
            src,
            coords,
            shape,
            gmem_shape,
            reduce_add,
            cache_hint,
            prefetch_tensormap,
        } => {
            // `reduce_add` (`cp.reduce.async.bulk...add`) has no `Tx.copy_async`-level
            // dispatch in TIRx (only the raw PTX intrinsic exists) — emitting a plain
            // overwrite store here would VALIDATE one semantics and RUN another. Fail
            // closed until a real reduce-add lowering lands with the flash-bwd datapath.
            if *reduce_add {
                return Err("codegen: TmaStore reduce_add has no TIRx lowering yet".to_string());
            }
            // The SMEM source tile (the staged D writeback band) and the GMEM
            // destination region. Single-issue: one thread of the enclosing role
            // (thread 0 of a warpgroup role, the elected lane of a warp role).
            let src_s = emit_smem_tile(src, ctx)?;
            let dst_s = emit_gmem_region(dst, coords, gmem_extents(gmem_shape, shape), ctx)?;
            // Both hints are IR-carried (builder defaults = the canonical epilogue
            // store's policy). `cache_hint="evict_first"`: the store band is write-once
            // output, never re-read by this kernel — dead lines must not pack L2 and
            // evict the live operand tiles / TMA tensormaps. Without it, a fully
            // occupied persistent launch (148 CTAs each streaming many output tiles
            // through L2) pressures the memory subsystem until a TMA store reads a
            // stale descriptor and the launch faults (cudaErrorLaunchFailure 719) —
            // an L2-policy hazard, invisible to the happens-before protocol model
            // (the checker stays green), occupancy-driven, and masked by any
            // serialization.
            let cache_hint_kw = match cache_hint {
                Some(hint) => format!(", cache_hint=\"{hint}\""),
                None => String::new(),
            };
            let prefetch_kw = if *prefetch_tensormap {
                ", prefetch_tensormap=True"
            } else {
                ""
            };
            emit_guarded(
                out,
                &format!(
                    "Tx.copy_async({dst_s}, {src_s}, dispatch=\"tma\"{cache_hint_kw}{prefetch_kw})"
                ),
            );
            Ok(())
        }
        CpAsyncBulkCommitGroup => {
            // Scope-aware like the async-proxy fence: `commit_group` batches the TMA
            // stores issued BY THIS THREAD, and the store itself is single-issue
            // (`if tid_in_wg == 0`). Inside a role branch, guard the commit to that
            // same issuing thread — canon's shape (`fence; store; commit` under ONE
            // `tid%128==0` guard). Unguarded, all 4 warps of the epilogue warpgroup
            // execute the uniform UTMACMDFLUSH (ncu: 4x canon's count) committing
            // empty groups; the non-issuing threads' wait_group is already vacuous
            // (no groups outstanding) and the wg_sync after the wait is the real
            // cross-thread barrier.
            if scope.is_function() {
                out.push_str(&format!("{p}T.ptx.cp_async.bulk.commit_group()\n"));
            } else {
                emit_guarded(out, "T.ptx.cp_async.bulk.commit_group()");
            }
            Ok(())
        }
        CpAsyncBulkWaitGroupRead { n } => {
            out.push_str(&format!(
                "{p}T.ptx.cp_async.bulk.wait_group({n}, read=True)\n"
            ));
            Ok(())
        }

        // ---- fence / sync ----
        // The epilogue's async-proxy fence makes the warpgroup's reg->smem writes
        // visible to the TMA proxy before the store. The init fences are synthesized
        // in KernelInit, so the only `Fence` stmts in the IR are these epilogue ones.
        // Emit it SINGLE-THREAD (`if tid_in_wg == 0`), like canon's `if (warp_id==0)&
        // (lane_id==0): fence.proxy_async`. Canon's own comment: "an all-128-thread fence
        // was the dominant stall" — the preceding warpgroup_sync already makes the reg->smem
        // writes CTA-visible, so only the single TMA-issuing thread needs the proxy fence.
        // (The kernel orders the wg_sync BEFORE this fence so the writes are visible first.)
        Fence { kind, .. } => {
            use super::dtype::FenceKind;
            match kind {
                // `fence.mbarrier_init` — the prologue init-epoch fence (all threads).
                FenceKind::MbarrierInit => {
                    out.push_str(&format!("{p}T.ptx.fence.mbarrier_init()\n"));
                }
                // Scope-aware, like cta_sync: at FUNCTION scope (the prologue, where all CTA
                // threads converge) emit the proxy fence for ALL threads; inside a single-
                // warp/warpgroup ROLE (the epilogue) emit it single-thread (`if tid_in_wg==0`)
                // — canon's "an all-128-thread fence was the dominant stall", the preceding
                // wg_sync already made the writes visible. Generic on the emit scope, no
                // overlap/no-overlap knowledge.
                FenceKind::AsyncProxy => {
                    if scope.is_function() {
                        out.push_str(&format!("{p}T.ptx.fence.proxy_async(\"shared::cta\")\n"));
                    } else {
                        out.push_str(&format!("{p}if tid_in_wg == 0:\n"));
                        out.push_str(&format!(
                            "{p}    T.ptx.fence.proxy_async(\"shared::cta\")\n"
                        ));
                    }
                }
                FenceKind::Memory | FenceKind::View => {}
            }
            Ok(())
        }
        CtaSync => {
            // Suppress CTA-wide cta_sync inside a single-warp/wg role branch (not
            // all CTA threads reach it → illegal __syncthreads).
            if scope.is_function() {
                out.push_str(&format!("{p}T.cuda.cta_sync()\n"));
            }
            Ok(())
        }
        ClusterSync => {
            out.push_str(&format!("{p}T.cuda.cluster_sync()\n"));
            Ok(())
        }
        ClusterBarrierArrive => {
            // Split cluster barrier, collective non-blocking arrival (all threads).
            out.push_str(&format!(
                "{p}T.ptx.barrier.cluster.arrive(sem=\"relaxed\", aligned=True)\n"
            ));
            Ok(())
        }
        ClusterBarrierWait => {
            // Split cluster barrier, per-role wait. `barrier.cluster.wait` is WARP-COLLECTIVE:
            // ALL threads of the role's warp(group) must execute it. If only the elected lane
            // waits, the overlap path DEADLOCKS on hardware (verified on 1024/2048) while the
            // protocol checker still passes (it models the wait as collective regardless of the
            // codegen elect context). The elected-Role arm hoists a leading ClusterBarrierWait
            // out of the elect to warp scope; reaching here under `Scope::Elected` means that
            // hoist was bypassed — fail LOUDLY instead of emitting a silently-hanging kernel.
            if matches!(scope, Scope::Elected) {
                return Err(
                    "codegen: ClusterBarrierWait under elect scope would emit a single-thread \
                     barrier.cluster.wait (hardware deadlock). It must be hoisted to warp scope \
                     (all threads of the warp), like canon's pre-elect cluster.wait."
                        .to_string(),
                );
            }
            out.push_str(&format!(
                "{p}T.ptx.barrier.cluster.wait(acquire=True, aligned=False)\n"
            ));
            Ok(())
        }
        WarpSync => Ok(()),
        WgSync { barrier_id } => {
            out.push_str(&format!("{p}T.cuda.warpgroup_sync({barrier_id})\n"));
            Ok(())
        }
        // Standalone per-warpgroup register budget (canon's per-role setmaxnreg). Gate on
        // `wg_id == <warpgroup>` so exactly that warpgroup's 4 warps issue the collective
        // `T.ptx.setmaxnreg`; inc when the budget rises above the 128-reg default, else dec.
        SetMaxNReg { warpgroup, count } => {
            let inc = if *count > 128 { "True" } else { "False" };
            out.push_str(&format!("{p}if wg_id == {warpgroup}:\n"));
            out.push_str(&format!("{p}    T.ptx.setmaxnreg({inc}, {count})\n"));
            Ok(())
        }
        // Cross-warpgroup named barrier — `bar.sync barrier_id, num_warps*32`. Unlike
        // WgSync (per-warpgroup), threads from different roles rendezvous on the shared
        // `barrier_id` with count-based completion. Canon writes this as
        // `T.ptx.bar.sync(<barrier_id>, <count>)` (e.g. `named_barrier_sync_8`).
        NamedBarrier {
            barrier_id,
            num_warps,
        } => {
            let count = num_warps * 32;
            out.push_str(&format!("{p}T.ptx.bar.sync({barrier_id}, {count})\n"));
            Ok(())
        }
        // ---- Set B: dev-framework ops belonging to the flash-attention / flash-bwd
        // datapaths. The GEMM/nvfp4 codegen has no warp-fragment reg-view, GMEM-
        // semaphore, or DSMEM cluster-copy lowering machinery, and no smoke-test kernel
        // (nvfp4 / fp16) exercises them. Fail closed with `Err` (a PyValueError through
        // the wrapper) like every other unsupported node — never panic. ----
        WarpMma { .. } => Err("codegen: WarpMma not yet supported".to_string()),
        GmemAtomicAdd { .. } => Err("codegen: GmemAtomicAdd not yet supported".to_string()),
        GmemWaitEq { .. } => Err("codegen: GmemWaitEq not yet supported".to_string()),
        CpAsyncBulkS2Cluster { .. } => {
            Err("codegen: CpAsyncBulkS2Cluster not yet supported".to_string())
        }
        // Carries the `Log2`/`Exp2`/`Rcp`/`Neg` RegUnaryOp; applied over flash
        // reg fragments (no GEMM-codegen reg-view path).
        RegUnary { .. } => Err("codegen: RegUnary not yet supported".to_string()),

        // NVFP4 epilogue alpha rescale: Tx.wg.mul(frag, frag, alpha). lhs is a reg slice,
        // rhs the alpha literal (or vice versa).
        RegMul { dst, lhs, rhs } => {
            let zero = ScalarValue::Int(0);
            let reg_op = |op: &RegOperand, out: &mut String| -> Result<String, String> {
                match op {
                    RegOperand::Slice(s) => {
                        let off = s.offsets.first().unwrap_or(&zero);
                        let w = s.shape.first().and_then(as_int).unwrap_or(0).max(0) as usize;
                        emit_reg_view_slice(out, &p, &s.tensor, off, w, ctx)
                    }
                    // wg.mul needs a typed scalar (a bare int literal has no .dtype).
                    RegOperand::Literal(l) => Ok(format!("T.float32({})", l.as_f32())),
                }
            };
            let dst_off = dst.offsets.first().unwrap_or(&zero);
            let dst_w = dst.shape.first().and_then(as_int).unwrap_or(0).max(0) as usize;
            let dst_s = emit_reg_view_slice(out, &p, &dst.tensor, dst_off, dst_w, ctx)?;
            let lhs_s = reg_op(lhs, out)?;
            let rhs_s = reg_op(rhs, out)?;
            out.push_str(&format!("{p}Tx.wg.mul({dst_s}, {lhs_s}, {rhs_s})\n"));
            Ok(())
        }
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

/// `NAME.ptr_to([slot])` for an mbar op — the slot is the op's `stage` scalar (a
/// multi-stage ring barrier), or `0` for a single-stage mbar.
fn mbar_slot_ptr(
    mref: &super::mbar::MBarRef,
    stage: &Option<ScalarValue>,
    ctx: &Ctx,
) -> Result<String, String> {
    let name = mbar_buf_name(mref, ctx)?;
    let slot = stage
        .as_ref()
        .map(|s| emit_scalar(s, ctx))
        .transpose()?
        .unwrap_or_else(|| "0".to_string());
    Ok(format!("{name}.ptr_to([{slot}])"))
}

/// Build the GMEM TMA region from src tensor + coords + the GMEM tile extents.
/// Emits `A[lo0:hi0, lo1:hi1]`. The extents must be the *GMEM* tile dims (one per
/// `coord`); the SMEM tile `shape` may carry an extra leading ring dim, so callers
/// pass `gmem_shape` (falling back to `shape` only when the ranks already match).
fn emit_gmem_region(
    src: &Arc<Tensor>,
    coords: &[ScalarValue],
    extents: &[usize],
    ctx: &Ctx,
) -> Result<String, String> {
    let name = ctx.tensor_name(src.id)?;
    if coords.len() != extents.len() {
        return Err(format!(
            "codegen: GMEM region rank mismatch — {} coords vs {} extents for tensor {}",
            coords.len(),
            extents.len(),
            src.id
        ));
    }
    let mut dims = Vec::new();
    for (coord, ext) in coords.iter().zip(extents.iter()) {
        let lo = emit_scalar(coord, ctx)?;
        let ext_sv = ScalarValue::Int(*ext as i64);
        let hi = add_bound(coord, &ext_sv, ctx)?;
        dims.push(format!("{lo}:{hi}"));
    }
    Ok(format!("{name}[{}]", dims.join(", ")))
}

/// The GMEM tile extents for a TMA op: the explicit `gmem_shape` if present, else the
/// SMEM `shape` (used when no leading ring dim makes the ranks differ).
fn gmem_extents<'a>(gmem_shape: &'a Option<Vec<usize>>, shape: &'a [usize]) -> &'a [usize] {
    gmem_shape.as_deref().unwrap_or(shape)
}

/// Strip a `+ tid_in_wg` addend from a row-offset expr (the per-thread lane term of
/// a value-model store), returning the warpgroup tile base. Returns None if the expr
/// has no such addend.
fn strip_tid_in_wg(sv: &ScalarValue) -> Option<ScalarValue> {
    if let ScalarValue::Expr(e) = sv {
        if e.op == ScalarOp::Add && e.args.len() == 2 {
            for (i, a) in e.args.iter().enumerate() {
                if matches!(a, ScalarValue::Scope(ScopeValueKind::TidInWg)) {
                    return Some(e.args[1 - i].clone());
                }
            }
        }
    }
    None
}

/// reg_store dst (GMEM) -> the warpgroup-collective row band `C[base:base+128,
/// col:col+w]`. The value model's row offset is `base + tid_in_wg` (one row per
/// thread); the wg-collective `Tx.copy` takes the whole 128-row tile, so the lane
/// term is stripped and the row becomes a 128-wide range.
fn emit_gmem_row_store(dst: &TensorSlice, ctx: &Ctx) -> Result<String, String> {
    let name = ctx.tensor_name(dst.tensor.id)?;
    if dst.offsets.len() != 2 {
        return Err("codegen: reg_store dst must be 2D".to_string());
    }
    let clo = emit_scalar(&dst.offsets[1], ctx)?;
    let chi = add_bound(&dst.offsets[1], &dst.shape[1], ctx)?;
    if let Some(base) = strip_tid_in_wg(&dst.offsets[0]) {
        let lo = emit_scalar(&base, ctx)?;
        let hi = add_bound(&base, &ScalarValue::Int(128), ctx)?;
        Ok(format!("{name}[{lo}:{hi}, {clo}:{chi}]"))
    } else {
        // No lane term (already a tile base): emit a 128-row band from the offset.
        let lo = emit_scalar(&dst.offsets[0], ctx)?;
        let hi = add_bound(&dst.offsets[0], &ScalarValue::Int(128), ctx)?;
        Ok(format!("{name}[{lo}:{hi}, {clo}:{chi}]"))
    }
}

// ===========================================================================
// tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::dtype::ScalarDType;
    use crate::ir::scalar::VarId;
    use crate::ir::tensor::Tensor;

    fn gmem_arg(id: u32) -> Arc<Tensor> {
        Arc::new(Tensor {
            id,
            space: MemorySpace::Gmem,
            dtype: DType::F32,
            shape: vec![16, 16],
            layout: None,
            byte_offset: None,
            reg_frag: None,
        })
    }

    fn kernel(body: Vec<Stmt>) -> Kernel {
        Kernel {
            name: "t".to_string(),
            args: vec![gmem_arg(0)],
            body,
            num_warps: 4,
            smem_size_bytes: 0,
            launch_shape: vec![2],
            cluster_shape: vec![2],
            smem_pool: false,
        }
    }

    /// A scalar op with no TVMScript lowering (Xor) must fail closed with a
    /// codegen `Err` — never leak an `<unsupported op …>` placeholder into the
    /// emitted Python source (that was a syntax error at exec time).
    #[test]
    fn unsupported_scalar_op_is_an_err_not_source_text() {
        let cond = ScalarValue::expr(
            ScalarOp::Xor,
            vec![ScalarValue::Int(1), ScalarValue::Int(2)],
        );
        let k = kernel(vec![Stmt::BreakIf { cond }]);
        let err = kernel_to_tirx_source(&k).unwrap_err();
        assert!(err.contains("no TVMScript lowering"), "{err}");
    }

    /// `ForLoop { unroll }` emits `T.unroll(stop)`, which only expresses
    /// start=0/step=1 — a wider range must be rejected, not silently re-timed.
    #[test]
    fn unroll_loop_requires_zero_start_unit_step() {
        let var = Var {
            id: VarId(0),
            binding: VarBinding::Loop,
            dtype: ScalarDType::I32,
        };
        let bad = Stmt::ForLoop {
            var,
            start: ScalarValue::Int(1),
            stop: ScalarValue::Int(4),
            step: ScalarValue::Int(1),
            body: vec![],
            unroll: true,
        };
        let err = kernel_to_tirx_source(&kernel(vec![bad])).unwrap_err();
        assert!(err.contains("unroll requires literal start=0"), "{err}");

        let ok = Stmt::ForLoop {
            var,
            start: ScalarValue::Int(0),
            stop: ScalarValue::Int(4),
            step: ScalarValue::Int(1),
            body: vec![],
            unroll: true,
        };
        let src = kernel_to_tirx_source(&kernel(vec![ok])).unwrap();
        assert!(src.contains("for v0 in T.unroll(4):"), "{src}");
    }

    /// Arg names: A–H for the existing kernels, the full alphabet after, then
    /// id-derived `arg{i}` — no panic past 8 args.
    #[test]
    fn arg_names_extend_past_the_alphabet() {
        assert_eq!(arg_name(0), "A");
        assert_eq!(arg_name(7), "H");
        assert_eq!(arg_name(8), "I");
        assert_eq!(arg_name(25), "Z");
        assert_eq!(arg_name(26), "arg26");
        assert_eq!(arg_name(40), "arg40");
    }
}
