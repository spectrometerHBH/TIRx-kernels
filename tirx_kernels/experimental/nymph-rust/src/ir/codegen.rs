//! Rust codegen: lower a nymph `ir::Kernel` to a TVMScript (`tvm.script.tirx`)
//! source string. Ported from the Role-model codegen (PR #18) onto the
//! warp-model IR: the `Role`/`KernelInit`/`KernelFinalize` nodes are gone —
//! thread dispatch is plain `Stmt::If` over scalar predicates.
//!
//! ZERO-INFERENCE guard rule (user-mandated): codegen NEVER synthesizes an
//! emission guard from the statically-computed thread scope. Every guard in
//! the output comes from the IR:
//! - The `if_elected` sugar (`lane_id == 0`) emits as `if T.ptx.elect_sync():`
//!   (canon's guard), narrowed to `if T.cuda.thread_rank() == 0:` when the
//!   enclosed thread set is provably exactly CTA thread 0 AND the body is
//!   loop-free (canon's prologue form — the same one thread, an emission
//!   spelling change only). A leading `ClusterBarrierWait` inside an
//!   elect-form `If` is peeled out of the elect (warp-collective; the
//!   elected-lane wait deadlocks on hardware).
//! - Hardware single-issue ops (TmaLoad/TmaStore/CpAsyncBulkS2Cluster,
//!   Tcgen05Mma/Tcgen05Cp/Tcgen05Commit, MBarrierInit, ClcTryCancel) are
//!   legal ONLY under an explicit single-lane `If` — the validator's
//!   `single_issue_scope` rule rejects anything else at build, and
//!   `emit_single_issue` hard-errors as the codegen-side defense. (Before
//!   this rule codegen wrapped them in a scope-inferred `elect_sync` /
//!   `tid_in_wg == 0` — exactly the inference the rule bans.)
//! - Per-thread ops (mbarrier arrive / expect_tx / arrive_expect_tx,
//!   store_scalar, async-proxy fences) emit PER-THREAD, matching the
//!   interpreter (one application per executing lane) — an inferred guard
//!   undercounts arrivals / tx bytes / drops lane-varying stores (the gdn
//!   gate_ready deadlock class).
//! - The thread scope of a body is still CLASSIFIED from the enclosing `If`
//!   conditions (`static_thread_filter`) — but only for legality checks
//!   (CTA-wide `cta_sync` at function scope, warp-collective forms) and for
//!   the `Elected` recognition above, never to invent a guard.
//! - Role dispatch chaining (`chain_top_level_ifs`): a run of >=2 ADJACENT
//!   TOP-LEVEL `If`s whose conditions are warp/warpgroup equalities re-nests
//!   into canon's if/else decision tree, partitioned by warpgroup exactly like
//!   #18's `chain_top_level_roles` (a warpgroup-equality `If` is its group's
//!   prefix; warp-equality `If`s chain behind it; duplicate conditions merge
//!   bodies). Only order-preserving runs chain: a group mixing a warp-level
//!   `If` BEFORE its warpgroup-level `If` stays flat — chaining would reorder
//!   independent statements (the TMA-after-wait deadlock probe). The
//!   R2UR-sensitive fp16 dispatch shape depends on the canonical form
//!   (docs/perf-methodology.md §5).
//! - `KernelInit`'s two side duties survive as structural rules: the single
//!   TMEM view buffer (`tmem`) + SF views are declared at function scope right
//!   after the top-level statement containing the first `TmemAlloc`.
//!
//! Everything else is the #18 pass unchanged: full-K `gemm_async` at the IR's
//! own granularity, runtime `accum` scalar, TmemOperand/SfView TMEM views,
//! leader-routed TMA barriers (peer `try_wait` is illegal and skipped), the
//! structured emitter (`fill_empty_blocks`), the pow2/trunc
//! strength reduction gated on the provable-nonneg analysis, and fail-closed
//! exhaustiveness: every `Stmt` variant either has a lowering arm below or an
//! explicit `Err` — no `..` catch-all, never a silent different-semantics
//! emission.

use super::dtype::{DType, MemorySpace, ScalarOp, ScopeValueKind, Swizzle, VarBinding};
use super::kernel::Kernel;
use super::scalar::{ScalarExpr, ScalarInitial, ScalarValue, Var};
use super::stmt::{
    LdStShape, MatrixDType, MatrixShape, RegLiteral, RegOperand, RegUnaryOp, Rounding, Stmt,
};
use super::tensor::{Layout, MmaOperand, Tensor, TensorSlice, TmemOperand};
use super::thread_filter::{static_thread_filter, ThreadSet};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The imports the emitted source needs (prepended so the file is self-contained).
const HEADER_IMPORTS: &str = "\
import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import ComposeLayout, R, S, TCol, TileLayout, TLane
from tvm.tirx.layout import tcgen05_atom_layout, tmem_datapath_layout
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg
";

/// Vendored `mma_shared_layout(dtype, swizzle_mode, shape)` — a faithful port of
/// TVM's `tile_primitive.tma_utils.mma_shared_layout`, emitted into the source so
/// the generated file depends only on the PUBLIC `tvm.tirx.layout` algebra
/// (`ComposeLayout`/`TileLayout`/`S`), not the TVM-private
/// `tvm.tirx.cuda.operator.tile_primitive.tma_utils` import path. Same
/// swizzle-atom math: the no-swizzle mode is the packed-16B atom; modes 1/2/3
/// tile the (8, 256/512/1024-bit) MMA atom over the tile via `tile_to`.
const MMA_SHARED_LAYOUT_HELPER: &str = r#"
def mma_shared_layout(dtype, swizzle_mode, shape):
    """MMA-compatible shared-memory layout for shape and dtype (vendored from
    tvm.tirx.cuda.operator.tile_primitive.tma_utils so the emitted source has
    no TVM-private import). Same default tiling of the TMA atom layout."""
    bits = tvm.DataType(dtype).bits
    if swizzle_mode == 0:
        # No-swizzle MMA smem is the packed-16B atom: offset e16*m +
        # M*e16*(k//e16) + k%e16 (e16=128/bits); plain tile if K % e16 != 0.
        e16 = 128 // bits
        m, k = int(shape[-2]), int(shape[-1])
        if k % e16 == 0:
            lead = [int(s) for s in shape[:-2]]
            extents = [*lead, m, k // e16, e16]
            strides = []
            stride = m * k
            for e in reversed(lead):
                strides.insert(0, stride)
                stride *= e
            strides += [e16, m * e16, 1]
            return TileLayout(S[tuple(extents) : tuple(strides)]).canonicalize()
        return TileLayout(S[tuple(shape)]).canonicalize()
    # The (8, 256/512/1024-bit) swizzle atom for modes 1/2/3, in element units.
    atom_shape = {1: [8, 256], 2: [8, 512], 3: [8, 1024]}[swizzle_mode]
    atom_shape[-1] //= bits
    atom_shape = [1] * (len(shape) - len(atom_shape)) + atom_shape
    per_element = (128 // bits).bit_length() - 1
    period = 1 << (per_element + swizzle_mode + 3)
    layout = ComposeLayout(per_element, swizzle_mode, 3, TileLayout(S[(period,)]))
    tile_to_shape = list(atom_shape)
    tile_to_shape[-2] = shape[-2]
    return layout.tile_to(tile_to_shape, atom_shape).tile_to(shape, tile_to_shape).canonicalize()
"#;

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

/// The thread scope of the enclosing `If`-condition stack, derived per body
/// from `static_thread_filter` (see `classify_scope`). It plays the role the
/// Role node's warp/warpgroup/elected fields played in #18: it decides (a)
/// whether a CTA-wide `cta_sync` may be emitted (only at function scope, where
/// all CTA threads converge), (b) whether a single-thread async issue
/// (`mbarrier`/`TMA`/`MMA`/`commit`) still needs a guard, and (c) which guard
/// that is.
///
/// The single-issue guard is the crux of the cross-CTA mailbox handshake: one
/// warp's elected lane is exactly 1 thread of a 1-warp branch, but a
/// *warpgroup* branch spans 4 warps, so `elect.sync` inside it is true for 4
/// threads (one per warp) — a single-issue `mbarrier.arrive` under it would
/// arrive FOUR times, over-counting the barrier. A warpgroup branch must elect
/// `tid_in_wg == 0` (thread 0 of the whole 128-thread warpgroup) instead.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scope {
    /// Function scope (no narrowing condition, or a full-CTA one). CTA-wide
    /// sync is legal here.
    Function,
    /// Inside a one-full-warp branch (`if warp_id == w:`).
    Warp,
    /// Inside a one-full-warpgroup branch (`if wg_id == g:`).
    Warpgroup,
    /// Inside a branch with at most one lane per warp (an elect guard, a
    /// single-thread branch, ...): single-issue ops emit with NO further
    /// per-op guard (canon's `if elect_sync(): while ...:` loops).
    Elected,
}

impl Scope {
    /// True at function scope: all CTA threads converge, so `cta_sync` is legal.
    fn is_function(self) -> bool {
        self == Scope::Function
    }
}

/// The scope plus the static thread set it was classified from (kept so nested
/// `If`s refine it instead of re-deriving from the whole condition stack).
#[derive(Clone)]
struct ScopeInfo {
    scope: Scope,
    /// The statically-known enclosing thread set; `None` when any enclosing
    /// condition is runtime/CTA-coordinate-dependent (`ThreadFilter::Unknown`).
    set: Option<ThreadSet>,
}

impl ScopeInfo {
    fn function(num_warps: u32) -> ScopeInfo {
        ScopeInfo {
            scope: Scope::Function,
            set: Some(ThreadSet::full(num_warps)),
        }
    }
    fn is_function(&self) -> bool {
        self.scope.is_function()
    }
}

/// Classify the child scope of an `If` body from the enclosing thread set.
/// `set` is the parent set intersected with this condition's filter (already
/// computed by the caller); `None` = a runtime condition — inherit the parent
/// scope (the tightest known granularity) and drop the set.
fn classify_scope(set: Option<ThreadSet>, parent: &ScopeInfo) -> ScopeInfo {
    let Some(s) = set else {
        return ScopeInfo {
            scope: parent.scope,
            set: None,
        };
    };
    let scope = if s.is_full_cta() {
        Scope::Function
    } else if s.single_thread().is_some() || s.count() == s.warps_touched().len() {
        // Exactly one thread, or one lane per touched warp (an elect guard's
        // set): single-issue ops are already single-issue.
        Scope::Elected
    } else if s.is_exactly_one_full_warp().is_some() {
        Scope::Warp
    } else if s.is_exactly_one_full_warpgroup().is_some() {
        Scope::Warpgroup
    } else {
        // A static subset with no single-issue guard form (a partial multi-
        // warp set): keep the parent's guard scope, but carry the refined set
        // so a nested `If` can still narrow to a classifiable scope.
        parent.scope
    };
    ScopeInfo {
        scope,
        set: Some(s),
    }
}

/// The child scope info of an `If` body: `parent` refined by `cond`.
fn child_scope_info(cond: &ScalarValue, parent: &ScopeInfo, num_warps: u32) -> ScopeInfo {
    let set = match (
        parent.set.as_ref(),
        static_thread_filter(cond, num_warps).known(),
    ) {
        (Some(p), Some(f)) => Some(p.intersect(f)),
        (None, Some(f)) => Some(f.clone()),
        _ => None,
    };
    classify_scope(set, parent)
}

/// The `if_elected` sugar condition: `lane_id == 0` (either operand order).
fn as_lane_zero_equality(cond: &ScalarValue) -> bool {
    let ScalarValue::Expr(e) = cond else {
        return false;
    };
    if e.op != ScalarOp::Eq || e.args.len() != 2 {
        return false;
    }
    e.args
        .iter()
        .any(|a| matches!(a, ScalarValue::Scope(ScopeValueKind::LaneId)))
        && e.args.iter().any(|a| matches!(a, ScalarValue::Int(0)))
}

/// True when any statement in the body is (or contains) a persistent loop —
/// used to keep the prologue `thread_rank() == 0` guard off hot worker loops.
fn body_has_loop(body: &[Stmt]) -> bool {
    body.iter().any(|s| {
        matches!(
            s,
            Stmt::ForLoop { .. }
                | Stmt::Loop { .. }
                | Stmt::ForEachTask { .. }
                | Stmt::SchedulerImpl { .. }
        ) || s.child_bodies().iter().any(|b| body_has_loop(b))
    })
}

/// A warp/warpgroup equality condition (`warp_id == w` / `wg_id == g`, either
/// operand order) — the chainable top-level dispatch predicate.
fn as_scope_equality(cond: &ScalarValue) -> Option<(ScopeValueKind, i64)> {
    let ScalarValue::Expr(e) = cond else {
        return None;
    };
    if e.op != ScalarOp::Eq || e.args.len() != 2 {
        return None;
    }
    for (a, b) in [(&e.args[0], &e.args[1]), (&e.args[1], &e.args[0])] {
        if let (ScalarValue::Scope(kind), ScalarValue::Int(v)) = (a, b) {
            if matches!(kind, ScopeValueKind::WarpId | ScopeValueKind::WarpgroupId) {
                return Some((*kind, *v));
            }
        }
    }
    None
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
    /// CTA warp count (the thread-filter geometry).
    num_warps: u32,
    /// Column span of the single TMEM view buffer: the largest
    /// `base_col + n_cols` over the kernel's `TmemAlloc`s. The view is one
    /// `decl_buffer((128, cols), allocated_addr=0)` — TMEM is not a tensor, so
    /// every TMEM instruction references it by absolute column slice.
    tmem_view_cols: Option<usize>,
    /// Per-REG-tensor declared width: the fragment is declared `T.alloc_local(8)`
    /// in the nymph IR (instruction granularity); the wide read/cast/store band
    /// needs it sized to the full column band. id -> width.
    reg_widths: HashMap<u32, usize>,
    /// Per-REG-tensor auxiliary view needs (see `RegAuxViews` /
    /// `collect_reg_aux_views`): warp-matrix ops, `.16x*b` tcgen05 atom
    /// fragments, and per-thread scalar transfers address the raw per-thread
    /// storage, which the default `(128, W)` thread-axis tile cannot express.
    reg_aux_views: HashMap<u32, RegAuxViews>,
    /// 16-bit TMEM cell dtypes used by tcgen05 st (packed halves, two elements
    /// per 32-bit cell) — each gets a `tmem_f16`/`tmem_bf16` decl_buffer over
    /// the same allocated band as the f32 view (dense-packed convention, the
    /// TIRx 16-bit `.32x32b` datapath).
    tmem_16_views: Vec<DType>,
    /// An M=64 `Tcgen05Mma` exists (gdn's BT=64 accumulators): the dst
    /// accumulator is the M=64 non-ws datapath F (64 rows scattered
    /// 16-per-warp), so a `tmem_f` decl_buffer with
    /// `tmem_datapath_layout("F", 64, cols)` joins the f32 view.
    needs_tmem_f: bool,
    /// SMEM tensors partially sliced by a TMA load/store (a slice extent
    /// strictly inside the tensor extent — e.g. gdn's row-half K double
    /// buffer). A swizzle atom tiles the FULL row band, so the partial slice
    /// breaks the tma_auto shared-chain stride rule; these take swizzle
    /// mode 0 (a plain 16B-atom tile the chain slices cleanly).
    tma_partial_smem: std::collections::HashSet<u32>,
    /// mbar ids of the TMA-load completion barriers (`smem_full`, `sf_full`, ...)
    /// flagged `leader_routed` by the IR. In cluster mode the canonical pattern
    /// routes BOTH CTAs' TMA completions to the LEADER CTA's barrier (a
    /// `map_shared_rank(.., 0)` view used uniformly by both CTAs): each CTA's
    /// `Tx.copy_async` signals it, the leader (cbx==0) issues one
    /// `arrive.expect_tx` for the full cluster byte count, and the leader's MMA
    /// waits its own LOCAL barrier (which both CTAs fill). This replaces the
    /// illegal peer `try_wait` AND is the prerequisite for multicast TMA loads
    /// (the per-destination transaction count of a `multicast::cluster` copy
    /// must land on the single leader barrier, accounted via the `* cta_group`
    /// factor in the leader expect_tx). Empty when no mbar is flagged.
    tma_leader_mbars: std::collections::HashSet<u32>,
    /// Number of launched clusters (`launch_cta_count / cta_group`) — the grid stride
    /// for a `ForEachTask` grid-stride scheduler loop.
    num_clusters: usize,
    /// NVFP4 e4m3 scale-factor TMEM views (SFA_tmem, SFB_tmem), keyed by the
    /// operand's absolute physical base column; declared via `decl_buffer`
    /// right after the `tmem` view.
    sf_views: Vec<SfView>,
    /// Usage-derived scale-factor tensor ids (see `collect_sf_ids`) — the ONLY
    /// authority on "is this tensor a scale factor"; dtype is never consulted.
    sf: SfIds,
    /// Var ids provably non-negative (see `collect_nonneg_vars`) — the ONLY
    /// authority `is_nonneg` consults for `ScalarValue::Var`: ForLoop induction
    /// vars with a non-negative-literal start and positive-literal step, plus
    /// scalar vars whose every definition is provably non-negative (fixpoint).
    /// A bare `Var(_) => true` would silently strength-reduce a `%`/`//` on a
    /// sentinel-negative scalar (e.g. a drained-scheduler `task_id == -1`).
    nonneg_vars: std::collections::HashSet<u32>,
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

/// Auxiliary physical views a REG tensor needs beyond the bare
/// `T.wg_reg_tile` (collected by `collect_reg_aux_views`). The default wg tile
/// is a `(128, W)` thread-axis layout — right for `Tx.wg.*` tile ops, but
/// warp-matrix ops (`ldmatrix`/`stmatrix`/`mma.sync`), `.16x*b` tcgen05 atom
/// fragments, and per-thread scalar transfers must address the RAW per-thread
/// storage: LowerTIRxCleanup rejects direct element access on thread-axis
/// layouts ("unable to verify that the coordinate matches the current
/// thread"). With `flat` set, the TensorDef declares
/// `{name}_flat = T.alloc_local((W,), dt)` (the raw per-thread storage) and
/// turns `{name}` into a `(128, W)` wg VIEW over it, so both addressing forms
/// share one register file.
#[derive(Clone, Default)]
struct RegAuxViews {
    /// Declare `{name}_flat` and make `{name}` a view of it.
    flat: bool,
    /// `.16x*b` tcgen05 ld/st atom view `{name}_atom` (the PTX instr shape).
    atom_shape: Option<&'static str>,
    /// `uint32` reinterpret `{name}_flat_u32` (stmatrix's packed b16x2 words).
    flat_u32: bool,
    /// bf16/f16 reinterpret `{name}_flat_ab` (WarpMma A/B operand elements).
    flat_ab: Option<DType>,
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

/// Hardware thread-width constants: lanes per warp (the PTX warp width) and
/// threads per warpgroup. A warpgroup is 4 warps = 128 threads on sm_90/100,
/// and the tcgen05 TMEM datapath is 128 lanes tall — silicon invariants, NOT
/// kernel parameters. The launch config only decides the warpgroup COUNT
/// (`num_warps / WG_WARPS`, validated a multiple of 4 by `validate`), never
/// these widths, so they stay named constants rather than per-kernel inputs.
const WARP_LANES: usize = 32;
const WG_WARPS: usize = 4;
const WG_THREADS: usize = WG_WARPS * WARP_LANES; // 128; == TMEM lane rows

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
    let mut out = Emitter::new();

    out.push_str(HEADER_IMPORTS);
    out.push_str(MMA_SHARED_LAYOUT_HELPER);
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
        // Rank<2 SMEM (a flat scalar scratch row, e.g. gdn's gcs/beta vectors)
        // takes no swizzle either — `mma_shared_layout` needs a (row, col) tile.
        if t.space != MemorySpace::Smem
            || (is_int_dtype(t.dtype) && t.dtype != DType::U8)
            || t.shape.len() < 2
        {
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
        // A partially-TMA-sliced buffer breaks the swizzle chain (see
        // `tma_partial_smem`) — fall to the no-swizzle 16B atom.
        let mode = if ctx.tma_partial_smem.contains(&t.id) {
            0
        } else {
            mode
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
    let num_wg = num_warps as usize / WG_WARPS;
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
    // The cluster-scope ids come from the kernel's cluster_shape (a 1-D shape
    // (n,) is the x extent, y=1). The emitted form names exactly two ids
    // (cbx/cby, the leader/peer selectors used below), so a cluster of rank >2
    // has no emission form — fail closed rather than drop a dimension.
    let (cx, cy) = match k.cluster_shape.as_slice() {
        [x] => (*x, 1usize),
        [x, y] => (*x, *y),
        other => {
            return Err(format!(
                "codegen: cluster_shape {other:?} has no cta_id_in_cluster emission \
                 (only rank-1 (x,) or rank-2 (x, y) clusters are supported)"
            ));
        }
    };
    if cx == 0 || cy == 0 {
        return Err(format!(
            "codegen: cluster_shape dims must be positive (got [{cx}, {cy}])"
        ));
    }
    if cx == 1 && cy == 1 {
        // A cluster-1 kernel (gdn): a (1, 1) `cta_id_in_cluster` bind collapses
        // to a constant and the scope resolver cannot find `clusterCtaIdx.x`
        // downstream — emit the constants directly (semantically exact for a
        // single-CTA cluster).
        out.push_str(&format!("{p}cbx, cby = 0, 0\n", p = pad(ind)));
    } else {
        out.push_str(&format!(
            "{p}cbx, cby = T.cta_id_in_cluster([{cx}, {cy}], preferred=[{cx}, {cy}])\n",
            p = pad(ind)
        ));
    }
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
        "{p}tid_in_wg = T.thread_id_in_wg([{wg_threads}])\n",
        p = pad(ind),
        wg_threads = WG_THREADS
    ));
    // Lane within the warp. Single-issue async ops (mbarrier / TMA / MMA / commit)
    // run under a single-warp branch, so the per-thread guard must be lane 0 of that
    // warp (`T.ptx.elect_sync()`) — NOT `tid_in_wg == 0`, which is thread 0 of the
    // whole *warpgroup* and is never true for a warp whose lanes map to tid_in_wg
    // 32..63 (e.g. warp 5 in warpgroup 1), so the issue would never fire and the MMA
    // would deadlock. Mirrors the canonical kernels' `elect_sync` guard.
    out.push_str(&format!(
        "{p}lane_id = T.lane_id([{warp_lanes}])\n",
        p = pad(ind),
        warp_lanes = WARP_LANES
    ));
    out.push('\n');

    // ---- SMEM buffers (N-D; the swizzled rings + the plain I32 mailbox) ----
    // Two emission forms, selected by `k.smem_pool`:
    //   * STATIC (default, the big-shape path): each SMEM data buffer is its own
    //     `T.alloc_buffer(scope="shared")`. TVM sizes the static SMEM footprint as
    //     the sum of the per-buffer extents.
    //   * DYNAMIC POOL (the small-shape variant): canon's `T.SMEMPool()` form — ONE
    //     dynamic `alloc_buffer([0], "uint8", scope="shared.dyn")` that every data
    //     buffer aliases into via `pool.alloc(..., data=pool.ptr, byte_offset=...)`.
    //     The buffers carry the IR's own `byte_offset`, so `pool.move_base_to(off)`
    //     places each at exactly the offset the static form used (byte-for-byte the
    //     same physical layout) — but now as `shared.dyn`, cutting the STATIC SMEM
    //     footprint toward canon's shape. The mbar/tmem_addr buffers stay in the
    //     pool too (mixing a static `shared` region with `shared.dyn` pushes the
    //     dynamic base off its 1024-byte boundary and the swizzled buffers fault).
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
        let is_mailbox = (is_int_dtype(t.dtype) && t.dtype != DType::U8) || t.shape.len() < 2;
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
    }

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
    // Adjacent TOP-LEVEL warp/warpgroup-equality `If`s are re-nested into canon's
    // if/else role-dispatch chain for emission (pure CUDA branch layout; the IR
    // the interpreter consumes stays flat). The single TMEM view + SF views are
    // declared at function scope right after the top-level statement carrying
    // the first `TmemAlloc` (#18 declared them right after `KernelInit`).
    let units = chain_top_level_ifs(&k.body);
    let fn_scope = ScopeInfo::function(k.num_warps);
    let mut tmem_declared = ctx.tmem_view_cols.is_none();
    for unit in &units {
        emit_top_unit(&mut out, unit, ind, &ctx, &fn_scope)?;
        if !tmem_declared && unit_contains_tmem_alloc(unit) {
            emit_tmem_view_decls(&mut out, ind, &ctx);
            tmem_declared = true;
        }
    }
    if !tmem_declared {
        // The view decl anchors to the alloc site; an alloc buried where the walk
        // cannot see it (validate requires lifecycle ops top-level) would leave
        // `tmem` undeclared for the tcgen05 arms — fail closed.
        return Err(
            "codegen: no top-level TmemAlloc found for the TMEM view declaration".to_string(),
        );
    }

    Ok(render_lines(fill_empty_blocks(out.finish())))
}

/// Declare the single TMEM view buffer (+ the NVFP4 SF views) at function scope.
/// `tmem` is visible to the MMA + epilogue, so it must NOT be nested under the
/// warp-0 alloc guard. TMEM is not a tensor: one (128, cols) f32 view over the
/// whole allocated band, addressed everywhere by absolute column slices.
/// allocated_addr=0 (not tmem_addr[0], the SMEM-stored alloc result): the single
/// tcgen05.alloc always bases at TMEM column 0, so the view address is a
/// compile-time constant — exactly canon's form; pinning it to 0 cuts the
/// epilogue address math (VIADD/LOP3/LDS) roughly in half.
fn emit_tmem_view_decls(out: &mut Emitter, ind: usize, ctx: &Ctx) {
    let Some(cols) = ctx.tmem_view_cols else {
        return;
    };
    out.push_str(&format!(
        "{p}tmem = T.decl_buffer(({wg_threads}, {cols}), \"float32\", scope=\"tmem\", allocated_addr=0, layout=TileLayout(S[({wg_threads}, {cols}) : (1 @ TLane, 1 @ TCol)]))\n",
        p = pad(ind),
        wg_threads = WG_THREADS,
    ));
    // The M=64 datapath-F accumulator view (gdn's BT=64 GEMM dsts): 64
    // logical rows scattered 16-per-warp over the 128 lanes.
    if ctx.needs_tmem_f {
        out.push_str(&format!(
            "{p}tmem_f = T.decl_buffer((64, {cols}), \"float32\", scope=\"tmem\", allocated_addr=0, layout=tmem_datapath_layout(\"F\", 64, {cols}))\n",
            p = pad(ind),
        ));
    }
    // 16-bit packed-half views over the same band (dense: two elements per
    // 32-bit cell) for the tcgen05.st fp16/bf16 datapath — the TIRx
    // `.32x32b` local2tmem path requires the TMEM buffer dtype to equal the
    // fragment dtype.
    for dt in &ctx.tmem_16_views {
        let (name, dt_s) = match dt {
            DType::F16 => ("tmem_f16", "float16"),
            DType::Bf16 => ("tmem_bf16", "bfloat16"),
            other => {
                debug_assert!(false, "unexpected 16-bit tmem view dtype {other:?}");
                continue;
            }
        };
        let elems = cols * 2;
        out.push_str(&format!(
            "{p}{name} = T.decl_buffer(({wg_threads}, {elems}), \"{dt_s}\", scope=\"tmem\", allocated_addr=0, layout=TileLayout(S[({wg_threads}, {elems}) : (1 @ TLane, 1 @ TCol)]))\n",
            p = pad(ind),
            wg_threads = WG_THREADS,
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
            p = pad(ind),
            name = view.name,
            rows = view.logical_rows,
            cols = view.logical_cols,
            m_super = m_super,
            m_stride = m_super * 4,
            col = view.col,
        ));
    }
}

/// Does this top-level unit (a plain stmt or a chained `If` tree) contain a
/// `TmemAlloc` anywhere?
fn unit_contains_tmem_alloc(unit: &TopUnit) -> bool {
    fn stmt_has_alloc(s: &Stmt) -> bool {
        matches!(s, Stmt::TmemAlloc { .. })
            || s.child_bodies()
                .iter()
                .any(|b| b.iter().any(stmt_has_alloc))
    }
    match unit {
        TopUnit::Stmt(s) => stmt_has_alloc(s),
        TopUnit::Chain(c) => {
            fn chain_has_alloc(c: &IfChain) -> bool {
                c.body.iter().any(stmt_has_alloc)
                    || c.inner.as_ref().is_some_and(|i| chain_has_alloc(i))
                    || c.else_.as_ref().is_some_and(|e| chain_has_alloc(e))
            }
            chain_has_alloc(c)
        }
    }
}

/// Every emitted block opener (`if`/`else`/`for`/`while`/`def` …) must own at
/// least one body line: an empty If/guard body otherwise renders a header with
/// no indented statement — invalid Python. Fill the gap with `pass`,
/// generically at the structured-line level (covers empty `else` arms, an
/// elected branch whose body was fully peeled as leading ClusterBarrierWaits,
/// and empty guard blocks alike) — a per-construct validator ban would have to
/// enumerate every one of these sites.
fn fill_empty_blocks(lines: Vec<Line>) -> Vec<Line> {
    let is_opener = |text: &str| {
        text.ends_with(':')
            && text.split_whitespace().next().is_some_and(|kw| {
                matches!(
                    kw,
                    "if" | "elif"
                        | "else:"
                        | "for"
                        | "while"
                        | "with"
                        | "def"
                        | "try:"
                        | "except"
                        | "finally:"
                )
            })
    };
    let mut out: Vec<Line> = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        let opener = is_opener(&line.text);
        out.push(line.clone());
        if !opener {
            continue;
        }
        // The body begins at the next non-blank line; it must sit strictly
        // deeper than the opener, else the block is empty.
        let body = lines[i + 1..].iter().find(|l| !l.text.is_empty());
        if body.is_none_or(|b| b.indent <= line.indent) {
            out.push(Line {
                indent: line.indent + 1,
                text: "pass".into(),
            });
        }
    }
    out
}

/// One node of the emission-time if/else decision tree `chain_top_level_ifs`
/// builds (the warp-model analog of #18's role `else_body` chains).
struct IfChain {
    cond: ScalarValue,
    body: Vec<Stmt>,
    /// A warp-level chain nested INSIDE this branch (emitted at the end of
    /// `body`, one level deeper): #18 nested the chained warp roles in the
    /// warpgroup role's body; the warp-model `If` has no else/body split, so
    /// the nesting is explicit here.
    inner: Option<Box<IfChain>>,
    /// The next group, emitted under `else:`.
    else_: Option<Box<IfChain>>,
}

/// A top-level emission unit after chaining: an untouched statement, or a
/// chained decision tree of warp/warpgroup-equality `If`s.
enum TopUnit {
    Stmt(Stmt),
    Chain(IfChain),
}

/// Emission-time re-nesting of ADJACENT TOP-LEVEL warp/warpgroup-equality `If`s
/// into canon's if/else role-dispatch chain (`if wg_id == 2 { … } else { if
/// wg_id == 0 { … } else { … } }`). This is the warp-model port of #18's
/// `chain_top_level_roles`: purely a CUDA branch-layout transform — the
/// interpreter always consumes the FLAT list, and the chained form is
/// guard-equivalent (the branches are mutually exclusive, so an if/else
/// decision tree runs exactly the same bodies on the same threads). The flat
/// form's per-branch overhead measurably inflates whole-function SASS, which
/// on fp16 1024 is what tips ptxas's uniform-register placement (R2UR 13.5K
/// vs canon 1.8K — docs/perf-methodology.md §5).
///
/// Grouping (unchanged from #18): a run of >=2 consecutive top-level equality
/// `If`s partitions by warpgroup (`warp w -> w / 4`, `wg g -> g`). Within a
/// group, at most one warpgroup-equality `If` (its body is the group prefix)
/// plus the warp-equality `If`s chained behind it in source order; groups
/// chain in first-occurrence order. Duplicate conditions merge their bodies
/// (same threads, same order — flat-equivalent). A run containing any other
/// statement stays flat — and so does a run whose group mixes a warp-level
/// `If` BEFORE its warpgroup-level `If`: chaining would run the warpgroup
/// body first, REORDERING independent statements (observed miscompile: a
/// warp-1 TMA issue moved after the warpgroup's mbarrier wait → GPU
/// deadlock). Only the canonical wg-prefix-then-warp-roles order chains.
fn chain_top_level_ifs(body: &[Stmt]) -> Vec<TopUnit> {
    struct IfParts<'a> {
        cond: &'a ScalarValue,
        body: &'a [Stmt],
        /// Warp id (a warp-level branch), or None for a warpgroup-level branch.
        warp: Option<i64>,
    }

    /// Chain a group's warp-level `If`s behind one another (else-nested).
    /// Duplicate warp ids merge their bodies (concatenation — same threads).
    fn chain_warp_ifs(members: &[&IfParts<'_>]) -> Option<Box<IfChain>> {
        let mut merged: Vec<(i64, Vec<Stmt>, ScalarValue)> = Vec::new();
        for m in members.iter().filter(|m| m.warp.is_some()) {
            let w = m.warp.unwrap();
            if let Some(e) = merged.iter_mut().find(|e| e.0 == w) {
                e.1.extend_from_slice(m.body);
                continue;
            }
            merged.push((w, m.body.to_vec(), m.cond.clone()));
        }
        let mut chain: Option<Box<IfChain>> = None;
        for (_, b, cond) in merged.into_iter().rev() {
            chain = Some(Box::new(IfChain {
                cond,
                body: b,
                inner: None,
                else_: chain.take(),
            }));
        }
        chain
    }

    fn chain_if_run(run: &[Stmt]) -> Option<IfChain> {
        struct Numbered<'a> {
            parts: IfParts<'a>,
            index: usize,
        }
        let mut groups: Vec<(i64, Vec<Numbered<'_>>)> = Vec::new();
        for (index, s) in run.iter().enumerate() {
            let Stmt::If { cond, then_body } = s else {
                return None;
            };
            let (kind, v) = as_scope_equality(cond)?;
            let (warp, group) = match kind {
                ScopeValueKind::WarpId => (Some(v), v.div_euclid(WG_WARPS as i64)),
                _ => (None, v),
            };
            let parts = IfParts {
                cond,
                body: then_body,
                warp,
            };
            if let Some((_, members)) = groups.iter_mut().find(|(g, _)| *g == group) {
                members.push(Numbered { parts, index });
            } else {
                groups.push((group, vec![Numbered { parts, index }]));
            }
        }
        // Soundness: the chained tree runs each group's warpgroup-level prefix
        // BEFORE its warp-level branches. That is only the source order when
        // every warpgroup-level `If` of a group precedes all its warp-level
        // `If`s — otherwise chaining REORDERS independent statements (a TMA
        // issue before a warpgroup wait was moved after it → deadlock). Decline
        // to chain in that case; the run emits flat (source order, always
        // sound).
        for (_, members) in &groups {
            let wg_last = members
                .iter()
                .filter(|m| m.parts.warp.is_none())
                .map(|m| m.index)
                .max();
            let warp_first = members
                .iter()
                .filter(|m| m.parts.warp.is_some())
                .map(|m| m.index)
                .min();
            if let (Some(wg), Some(w)) = (wg_last, warp_first) {
                if wg > w {
                    return None;
                }
            }
        }
        let mut else_: Option<Box<IfChain>> = None;
        for (g, members) in groups.into_iter().rev() {
            let members: Vec<IfParts<'_>> = members.into_iter().map(|m| m.parts).collect();
            let refs: Vec<&IfParts<'_>> = members.iter().collect();
            let (wg_ifs, warp_ifs): (Vec<&IfParts<'_>>, Vec<&IfParts<'_>>) =
                refs.into_iter().partition(|m| m.warp.is_none());
            // The warp-level chain nests INSIDE the group branch (after the
            // group prefix body), exactly like #18 nested chained warp roles in
            // the warpgroup role's body.
            let inner = chain_warp_ifs(&warp_ifs);
            let (cond, body) = if let Some(first) = wg_ifs.first() {
                // The warpgroup-level branch is the group prefix: its body (all
                // duplicate wg-`If` bodies concatenated) runs first.
                let mut body = Vec::new();
                for m in &wg_ifs {
                    body.extend_from_slice(m.body);
                }
                (first.cond.clone(), body)
            } else {
                // No warpgroup-level `If` in this group: synthesize the group
                // guard (`wg_id == g`) so the warp chain lands on its warpgroup.
                (
                    ScalarValue::expr(
                        ScalarOp::Eq,
                        vec![
                            ScalarValue::Scope(ScopeValueKind::WarpgroupId),
                            ScalarValue::Int(g),
                        ],
                    ),
                    Vec::new(),
                )
            };
            else_ = Some(Box::new(IfChain {
                cond,
                body,
                inner,
                else_: else_.take(),
            }));
        }
        else_.map(|e| *e)
    }

    let mut out: Vec<TopUnit> = Vec::with_capacity(body.len());
    let mut i = 0;
    while i < body.len() {
        let is_eq_if =
            |s: &Stmt| matches!(s, Stmt::If { cond, .. } if as_scope_equality(cond).is_some());
        if !is_eq_if(&body[i]) {
            out.push(TopUnit::Stmt(body[i].clone()));
            i += 1;
            continue;
        }
        let mut j = i;
        while j < body.len() && is_eq_if(&body[j]) {
            j += 1;
        }
        let run = &body[i..j];
        // Runs of >=2 chain; a lone equality `If` emits flat (its plain form
        // is already the canonical one-branch tree).
        if run.len() >= 2 {
            if let Some(chained) = chain_if_run(run) {
                out.push(TopUnit::Chain(chained));
                i = j;
                continue;
            }
        }
        for s in run {
            out.push(TopUnit::Stmt(s.clone()));
        }
        i = j;
    }
    out
}

/// One emitted source line: its indent in 4-space units, its text WITHOUT the
/// leading pad, and an optional single-issue-guard annotation. The annotation
/// is set at the emission site when the line opens a `if tid_in_wg == 0:` /
/// `if T.ptx.elect_sync():` single-issue guard block — it is what the guard
#[derive(Clone, PartialEq, Eq)]
struct Line {
    indent: usize,
    text: String,
}

/// The emission sink: accumulates the source as STRUCTURED lines (not one flat
/// string) so the guard merge works on the (indent, guard-annotation) line
/// structure instead of re-parsing text (fragile to blank lines and to
/// guard-looking non-guard lines). `push_str` keeps the call sites
/// string-shaped; the sink splits into lines and records each line's indent
/// (all padding goes through `pad()` = 4-space units).
struct Emitter {
    lines: Vec<Line>,
    /// A line started but not yet `\n`-terminated (a push_str may split mid-line).
    partial: String,
}

impl Emitter {
    fn new() -> Self {
        Emitter {
            lines: Vec::new(),
            partial: String::new(),
        }
    }

    fn push_str(&mut self, s: &str) {
        for chunk in s.split_inclusive('\n') {
            if let Some(text) = chunk.strip_suffix('\n') {
                self.partial.push_str(text);
                self.finish_line();
            } else {
                self.partial.push_str(chunk);
            }
        }
    }

    fn push(&mut self, c: char) {
        if c == '\n' {
            self.finish_line();
        } else {
            self.partial.push(c);
        }
    }

    fn finish_line(&mut self) {
        let text = std::mem::take(&mut self.partial);
        let indent = (text.len() - text.trim_start().len()) / 4;
        self.lines.push(Line {
            indent,
            text: text.trim_start().to_string(),
        });
    }

    fn finish(mut self) -> Vec<Line> {
        if !self.partial.is_empty() {
            self.finish_line();
        }
        self.lines
    }
}

/// Merge ADJACENT identical single-issue guard blocks (the annotated
/// `if tid_in_wg == 0:` / `if T.ptx.elect_sync():` lines) into one block.
/// Render the structured lines back to source text: re-pad each line from its
/// numeric indent, join with newlines, terminate the final line.
fn render_lines(lines: Vec<Line>) -> String {
    let mut s = String::new();
    for l in lines {
        if l.text.is_empty() {
            s.push('\n');
        } else {
            s.push_str(&pad(l.indent));
            s.push_str(&l.text);
            s.push('\n');
        }
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
/// `TmemOperand` — physical addresses, no tensor ids; see `sf_views`.)
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
        if let Stmt::Tcgen05Cp {
            dst: _,
            src,
            cta_group: _,
        } = s
        {
            ids.smem.insert(src.tensor.id);
        }
    });
    // Pass 2: GMEM sources of the TMA loads that fill an SF SMEM ring (one level —
    // SF bytes flow gmem -> smem -> tmem, there are no longer chains).
    walk(&k.body, &mut |s| {
        if let Stmt::TmaLoad {
            dst,
            src,
            mbar: _,
            coords: _,
            shape: _,
            gmem_shape: _,
            mbar_stage: _,
            multicast_cta_mask: _,
            cache_hint: _,
            prefetch_tensormap: _,
            cta_group: _,
        } = s
        {
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
        TmaLoad {
            dst,
            src,
            mbar: _,
            coords: _,
            shape: _,
            gmem_shape: _,
            mbar_stage: _,
            multicast_cta_mask: _,
            cache_hint: _,
            prefetch_tensormap: _,
            cta_group: _,
        } => {
            note_slice(dst, map);
            note_tensor(src, map);
        }
        TmaStore {
            dst,
            src,
            coords: _,
            shape: _,
            gmem_shape: _,
            reduce_add: _,
            allow_nondet_reduce: _,
            cache_hint: _,
            prefetch_tensormap: _,
        } => {
            note_tensor(dst, map);
            note_slice(src, map);
        }
        Tcgen05Mma {
            dst: _,
            a,
            b,
            m: _,
            n: _,
            k: _,
            accum: _,
            trans_a: _,
            trans_b: _,
            cta_group: _,
            sfa: _,
            sfb: _,
            sf_byte: _,
            sf_e4m3: _,
            sf_block: _,
            a_fp4: _,
            b_fp4: _,
            lane_align: _,
        } => {
            // TMEM operands (dst/sfa/sfb, TmemOperand-form a/b) carry no tensor.
            for op in [a, b] {
                if let MmaOperand::Slice(s) = op {
                    note_slice(s, map);
                }
            }
        }
        Tcgen05Cp {
            dst: _,
            src,
            cta_group: _,
        } => {
            note_slice(src, map);
        }
        Tcgen05Ld {
            dst,
            src: _,
            shape: _,
            num: _,
        } => {
            note_slice(dst, map);
        }
        Tcgen05St {
            dst: _,
            src,
            shape: _,
            num: _,
        } => {
            note_slice(src, map);
        }
        RegCvt {
            dst,
            src,
            rounding: _,
        }
        | RegLoad { dst, src }
        | RegStore { dst, src } => {
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

    // The TMA-load completion barriers to leader-route, from the IR's explicit
    // `MBar::leader_routed` flag (validate has already checked each carries a
    // peer reference and is only used by TmaLoad/expect_tx). Routing both CTAs'
    // TMA to the leader's copy of each barrier (and waiting only the local
    // copy) is the legal substitute for the peer wait, AND the prerequisite for
    // multicast loads — a `multicast::cluster` copy's per-destination
    // transaction count must accumulate on the single leader barrier (the
    // `* cta_group` leader expect_tx accounts for it).
    let mut tma_leader_mbars: std::collections::HashSet<u32> = std::collections::HashSet::new();
    fn find_leader_mbars(stmts: &[Stmt], out: &mut std::collections::HashSet<u32>) {
        for s in stmts {
            if let Stmt::MBarDef { mbar } = s {
                if mbar.leader_routed {
                    out.insert(mbar.id);
                }
            }
            for body in s.child_bodies() {
                find_leader_mbars(body, out);
            }
        }
    }
    find_leader_mbars(&k.body, &mut tma_leader_mbars);

    // The single TMEM view buffer spans every allocated column band:
    // `max(base_col + n_cols)` over the kernel's TmemAllocs.
    let mut tmem_view_cols: Option<usize> = None;
    fn find_tmem_view_cols(stmts: &[Stmt], out: &mut Option<usize>) {
        for s in stmts {
            if let Stmt::TmemAlloc {
                base_col,
                n_cols,
                cta_group: _,
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

    // Per-REG-tensor width. Walk the bodies and record, for each REG fragment,
    // `max(offset + width)` over every slice — the band a single `.view(...)`
    // alias must span. (A capped drain writes the 256-wide output reg in two
    // 128-col groups, so the FULL extent comes from offset+width, not the
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
        for s in stmts {
            match s {
                Stmt::Tcgen05Ld {
                    dst,
                    src,
                    shape,
                    num,
                } => {
                    if dst.tensor.space == MemorySpace::Reg {
                        let off = dst.offsets.first().and_then(as_int).unwrap_or(0).max(0) as usize;
                        let e = widths.entry(dst.tensor.id).or_insert(0);
                        *e = (*e).max(off + tcgen05_frag_regs(shape, *num as usize, src.dtype));
                    }
                }
                Stmt::Tcgen05St {
                    dst,
                    src,
                    shape: _,
                    num,
                } => {
                    if src.tensor.space == MemorySpace::Reg {
                        let off = src.offsets.first().and_then(as_int).unwrap_or(0).max(0) as usize;
                        // The IR slice counts b32 registers; a packed-half st
                        // reads two elements per register.
                        let e32 = if matches!(dst.dtype, DType::F16 | DType::Bf16) {
                            2
                        } else {
                            1
                        };
                        let e = widths.entry(src.tensor.id).or_insert(0);
                        *e = (*e).max(off + *num as usize * e32);
                    }
                }
                Stmt::RegCvt {
                    dst,
                    src,
                    rounding: _,
                } => {
                    note_reg_width(dst, widths);
                    note_reg_width(src, widths);
                }
                Stmt::RegStore { dst: _, src } => note_reg_width(src, widths),
                _ => {}
            }
            for body in s.child_bodies() {
                walk_reg_widths(body, widths);
            }
        }
    }
    walk_reg_widths(&k.body, &mut reg_widths);

    // Auxiliary REG-tensor views (flat storage / atom tcgen05 frags / dtype
    // reinterprets) required by the warp-matrix and per-thread scalar uses.
    let mut reg_aux_views: HashMap<u32, RegAuxViews> = HashMap::new();
    let mut tmem_16_views: Vec<DType> = Vec::new();
    collect_reg_aux_views(&k.body, &mut reg_aux_views, &mut tmem_16_views)?;

    // An M=64 tcgen05 MMA needs the datapath-F accumulator view.
    let mut needs_tmem_f = false;
    fn find_m64_mma(stmts: &[Stmt], out: &mut bool) {
        for s in stmts {
            if let Stmt::Tcgen05Mma { m, .. } = s {
                if *m == 64 {
                    *out = true;
                }
            }
            for body in s.child_bodies() {
                find_m64_mma(body, out);
            }
        }
    }
    find_m64_mma(&k.body, &mut needs_tmem_f);

    // SMEM tensors partially sliced by TMA (see `tma_partial_smem`).
    let mut tma_partial_smem: std::collections::HashSet<u32> = std::collections::HashSet::new();
    fn find_tma_partial(stmts: &[Stmt], out: &mut std::collections::HashSet<u32>) {
        fn note_slice(s: &TensorSlice, out: &mut std::collections::HashSet<u32>) {
            if s.tensor.space != MemorySpace::Smem || s.shape.len() != s.tensor.shape.len() {
                return;
            }
            // A leading stage-dim slice (extent 1) collapses to an outer index
            // and leaves the tiled (M, K) box whole — only a partial slice of a
            // TILED dim (extent > 1, strictly inside the tensor) breaks the
            // shared-chain stride rule.
            let partial = s
                .shape
                .iter()
                .zip(s.tensor.shape.iter())
                .any(|(e, full)| as_int(e).is_some_and(|e| e > 1 && (e as usize) < *full));
            if partial {
                out.insert(s.tensor.id);
            }
        }
        for s in stmts {
            match s {
                Stmt::TmaLoad { dst, .. } => note_slice(dst, out),
                Stmt::TmaStore { src, .. } => note_slice(src, out),
                _ => {}
            }
            for body in s.child_bodies() {
                find_tma_partial(body, out);
            }
        }
    }
    find_tma_partial(&k.body, &mut tma_partial_smem);

    // Scalar var names. Every `ScalarDef` introduces an SSA register var
    // (`NAME: T.int32 = init`, read as `NAME`). Var ids are globally unique, so
    // a per-id name (`s{id}`) is stable and collision-free.
    let mut scalar_names: HashMap<u32, String> = HashMap::new();
    fn walk_scalar_defs(stmts: &[Stmt], scalar_names: &mut HashMap<u32, String>) {
        for s in stmts {
            let defined = match s {
                Stmt::ScalarDef { var, initial: _ }
                | Stmt::ScalarLet { var, value: _ }
                | Stmt::ShuffleSync {
                    var,
                    src: _,
                    src_lane: _,
                }
                | Stmt::ClcQueryCancel {
                    scheduler: _,
                    var,
                    handle: _,
                } => Some(var),
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
        num_warps: k.num_warps,
        tmem_view_cols,
        reg_widths,
        reg_aux_views,
        tmem_16_views,
        needs_tmem_f,
        tma_partial_smem,
        tma_leader_mbars,
        num_clusters: (k.launch_cta_count() / (cta_group as usize).max(1)).max(1),
        sf_views,
        sf,
        nonneg_vars: collect_nonneg_vars(k),
    })
}

/// Walk the body once and record, per REG tensor, the auxiliary views its
/// uses require (see `RegAuxViews`).
fn collect_reg_aux_views(
    stmts: &[Stmt],
    views: &mut HashMap<u32, RegAuxViews>,
    tmem_16: &mut Vec<DType>,
) -> Result<(), String> {
    for s in stmts {
        match s {
            Stmt::Tcgen05Ld { dst, shape, .. } => {
                if *shape != LdStShape::B32x32 {
                    let v = views.entry(dst.tensor.id).or_default();
                    v.flat = true;
                    note_atom_shape(v, dst.tensor.id, shape)?;
                }
            }
            Stmt::Tcgen05St {
                dst, src, shape, ..
            } => {
                if *shape != LdStShape::B32x32 {
                    let v = views.entry(src.tensor.id).or_default();
                    v.flat = true;
                    note_atom_shape(v, src.tensor.id, shape)?;
                }
                if matches!(dst.dtype, DType::F16 | DType::Bf16) && !tmem_16.contains(&dst.dtype) {
                    tmem_16.push(dst.dtype);
                }
            }
            Stmt::Tcgen05Mma { a, b, .. } => {
                // 16-bit TMEM MMA operands read through the packed views.
                for op in [a, b] {
                    if let MmaOperand::Tmem(t) = op {
                        if matches!(t.dtype, DType::F16 | DType::Bf16)
                            && !tmem_16.contains(&t.dtype)
                        {
                            tmem_16.push(t.dtype);
                        }
                    }
                }
            }
            Stmt::WarpMma {
                d,
                a,
                b,
                c,
                ab_dtype,
                ..
            } => {
                for sl in [d, a, b, c] {
                    views.entry(sl.tensor.id).or_default().flat = true;
                }
                views.entry(a.tensor.id).or_default().flat_ab = Some(*ab_dtype);
                views.entry(b.tensor.id).or_default().flat_ab = Some(*ab_dtype);
            }
            Stmt::LdMatrix { dst, .. } => {
                views.entry(dst.tensor.id).or_default().flat = true;
            }
            Stmt::StMatrix { src, .. } => {
                let v = views.entry(src.tensor.id).or_default();
                v.flat = true;
                if matches!(src.tensor.dtype, DType::F16 | DType::Bf16) {
                    v.flat_u32 = true;
                }
            }
            // Every elementwise-touched tensor may need the flat view: the
            // `Tx.wg.*` elementwise dispatches require the full launch intra
            // (whole warpgroup), so an op under a narrowed warp/lane branch
            // lowers to the per-thread scalar form on the flat views instead
            // (the emit side picks per site via `wg_elem_ok`).
            Stmt::RegFill { dst, value } => {
                views.entry(dst.tensor.id).or_default().flat = true;
                if let RegOperand::Slice(s) = value {
                    views.entry(s.tensor.id).or_default().flat = true;
                }
            }
            Stmt::RegAdd { dst, lhs, rhs, .. }
            | Stmt::RegSub { dst, lhs, rhs, .. }
            | Stmt::RegMul { dst, lhs, rhs } => {
                views.entry(dst.tensor.id).or_default().flat = true;
                for op in [lhs, rhs] {
                    if let RegOperand::Slice(s) = op {
                        views.entry(s.tensor.id).or_default().flat = true;
                    }
                }
            }
            Stmt::RegFma { dst, a, b, c } => {
                views.entry(dst.tensor.id).or_default().flat = true;
                for op in [a, b, c] {
                    if let RegOperand::Slice(s) = op {
                        views.entry(s.tensor.id).or_default().flat = true;
                    }
                }
            }
            Stmt::RegUnary { dst, src, .. } => {
                views.entry(dst.tensor.id).or_default().flat = true;
                if let RegOperand::Slice(s) = src {
                    views.entry(s.tensor.id).or_default().flat = true;
                }
            }
            Stmt::RegCvt { dst, src, .. } => {
                views.entry(dst.tensor.id).or_default().flat = true;
                views.entry(src.tensor.id).or_default().flat = true;
            }
            // Per-thread point transfers lower to raw element assignments on
            // the flat view (see the RegLoad/RegStore arms).
            Stmt::RegLoad { dst, src } => {
                if matches!(src.tensor.space, MemorySpace::Smem | MemorySpace::Gmem)
                    && slice_all_size1(src)
                {
                    views.entry(dst.tensor.id).or_default().flat = true;
                }
            }
            Stmt::RegStore { dst, src } => {
                if src.tensor.space == MemorySpace::Reg
                    && dst.tensor.space != MemorySpace::Reg
                    && (slice_all_size1(dst) || gmem_row_run(dst))
                {
                    views.entry(src.tensor.id).or_default().flat = true;
                }
            }
            _ => {}
        }
        for body in s.child_bodies() {
            collect_reg_aux_views(body, views, tmem_16)?;
        }
    }
    Ok(())
}

fn note_atom_shape(v: &mut RegAuxViews, id: u32, shape: &LdStShape) -> Result<(), String> {
    match shape {
        LdStShape::B16x64 | LdStShape::B16x128 | LdStShape::B16x256 => {}
        other => {
            return Err(format!(
                "codegen: tcgen05 shape {} has no atom-fragment lowering \
                 (only 16x64b/16x128b/16x256b)",
                other.as_str()
            ))
        }
    }
    if let Some(prev) = v.atom_shape {
        if prev != shape.as_str() {
            return Err(format!(
                "codegen: REG tensor {id} is used as both a {prev} and a {} \
                 tcgen05 fragment (one atom view per tensor)",
                shape.as_str()
            ));
        }
    }
    v.atom_shape = Some(shape.as_str());
    Ok(())
}

/// Every slice dim is a static size-1 — a per-thread point transfer.
fn slice_all_size1(s: &TensorSlice) -> bool {
    !s.shape.is_empty() && s.shape.iter().all(|d| as_int(d) == Some(1))
}

/// A per-thread GMEM row run: rank >= 3, leading dims all size-1, trailing dim
/// > 1 (e.g. the gdn final-state store `state_g[seq, eh, tid, c0:c0+64]`).
/// Rank-2 `(1, w)` row stores keep the existing wg-band lowering.
fn gmem_row_run(s: &TensorSlice) -> bool {
    s.tensor.space == MemorySpace::Gmem
        && s.shape.len() >= 3
        && s.shape[..s.shape.len() - 1]
            .iter()
            .all(|d| as_int(d) == Some(1))
        && s.shape.last().and_then(as_int).is_some_and(|w| w > 1)
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
    // The logical rows round up to whole TMEM-lane (128-row) super-blocks.
    let logical_rows = rows.div_ceil(WG_THREADS) * WG_THREADS;
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
                dst: _,
                a: _,
                b: _,
                m,
                n,
                k,
                accum: _,
                trans_a: _,
                trans_b: _,
                cta_group,
                sfa: Some(sfa),
                sfb: Some(sfb),
                sf_byte: _,
                sf_e4m3: _,
                sf_block,
                a_fp4: _,
                b_fp4: _,
                lane_align: _,
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
            Stmt::Tcgen05Cp {
                dst,
                src,
                cta_group: _,
            } if dst.dtype == DType::F8E4M3 => {
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

/// The declared (full) width of a REG fragment's wg tile = its spanned band width.
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
    _out: &mut Emitter,
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

/// A rank-1 REG slice decomposed for view emission: (tensor, offset, static
/// width). Every nymph REG tensor is a per-thread 1-D vector — anything else
/// has no lowering.
fn reg_slice_parts(s: &TensorSlice) -> Result<(&Arc<Tensor>, &ScalarValue, usize), String> {
    if s.offsets.len() != 1 || s.shape.len() != 1 {
        return Err(format!(
            "codegen: REG slice of tensor {} must be rank-1 (got {} offsets, {} shape dims)",
            s.tensor.id,
            s.offsets.len(),
            s.shape.len()
        ));
    }
    let w = match as_int(&s.shape[0]) {
        Some(w) if w > 0 => w as usize,
        other => {
            return Err(format!(
                "codegen: REG slice of tensor {} needs a static positive width (got {other:?})",
                s.tensor.id
            ))
        }
    };
    Ok((&s.tensor, &s.offsets[0], w))
}

/// A `T.<dtype>(value)` scalar literal for `Tx.wg.*` operands (a bare number
/// has no dtype; the literal takes the DST tensor's dtype, exactly the
/// interpreter's `literal_array`).
fn typed_scalar(dtype: DType, l: RegLiteral) -> Result<String, String> {
    match dtype {
        DType::F16 | DType::Bf16 | DType::F32 => {
            Ok(format!("T.{}({})", dtype_str(dtype), l.as_f32()))
        }
        DType::I8
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::I32
        | DType::U32
        | DType::I64
        | DType::U64 => Ok(format!("T.{}({})", dtype_str(dtype), l.as_i64())),
        other => Err(format!(
            "codegen: no typed scalar literal for dtype {other:?}"
        )),
    }
}

/// One operand arm of a `Tx.wg.*` elementwise call: a wg-view slice, or a typed
/// scalar for a literal. Slice operands must share the dst dtype — the
/// interpreter coerces per-op, so a genuinely mixed-dtype op must say so with
/// an explicit RegCvt instead of being silently coerced here.
fn emit_wg_reg_operand(
    op: &RegOperand,
    dst_dtype: DType,
    out: &mut Emitter,
    p: &str,
    ctx: &Ctx,
) -> Result<String, String> {
    match op {
        RegOperand::Slice(s) => {
            let (t, off, w) = reg_slice_parts(s)?;
            if t.dtype != dst_dtype {
                return Err(format!(
                    "codegen: reg operand dtype {:?} != dst dtype {dst_dtype:?} — \
                     the interpreter coerces operands per-op; use an explicit RegCvt",
                    t.dtype
                ));
            }
            emit_reg_view_slice(out, p, t, off, w, ctx)
        }
        RegOperand::Literal(l) => typed_scalar(dst_dtype, *l),
    }
}

/// True when the enclosing scope provably covers a whole warpgroup: the
/// `Tx.wg.*` elementwise dispatches require the full launch intra (all 32
/// lanes of all four warps), so anything narrower (a lone warp, a
/// lane/warp-predicated branch) must take the per-thread scalar form.
fn wg_elem_ok(scope: &ScopeInfo) -> bool {
    scope
        .set
        .as_ref()
        .is_some_and(|s| s.is_full_cta() || s.is_exactly_one_full_warpgroup().is_some())
}

/// True when the slice spans the whole wg view (offset 0, width == the view's
/// full width). `Tx.wg.*` elementwise ops DROP the column offset of sliced
/// operands/dsts (B200-verified: `fill(frag[:, 1:2])` writes element 0; the
/// plain and dual-view forms differ in src-slice behavior), so anything
/// narrower than a full-extent slice must take the per-thread scalar form on
/// the flat views, which addresses elements exactly.
fn slice_is_full(t: &Arc<Tensor>, off: &ScalarValue, w: usize, ctx: &Ctx) -> bool {
    as_int(off) == Some(0) && w == reg_view_width(t, ctx)
}

/// True when every slice of a reg elementwise op (dst + slice operands) is
/// full-extent — the only case `Tx.wg.*` honors.
fn reg_op_slices_full(
    dst: &TensorSlice,
    operands: &[&RegOperand],
    ctx: &Ctx,
) -> Result<bool, String> {
    let (dt, doff, dw) = reg_slice_parts(dst)?;
    if !slice_is_full(dt, doff, dw, ctx) {
        return Ok(false);
    }
    for op in operands {
        if let RegOperand::Slice(s) = op {
            let (t, off, w) = reg_slice_parts(s)?;
            if !slice_is_full(t, off, w, ctx) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// One element's index into a flat view: `off + _i` (simplified for a base-0
/// slice or a constant read).
fn flat_elem_idx(off: &ScalarValue, i: &str, ctx: &Ctx) -> Result<String, String> {
    Ok(match (as_int(off), i) {
        (Some(0), _) => i.to_string(),
        (Some(b), "0") => b.to_string(),
        (Some(b), _) => format!("{b} + {i}"),
        (None, "0") => emit_scalar(off, ctx)?,
        (None, _) => format!("{} + {i}", emit_scalar(off, ctx)?),
    })
}

/// The per-thread scalar form of a reg elementwise op:
/// `for _i in range(w): <dst>_flat[d] = <expr>`. With `convert`, a 16-bit dst
/// follows the interpreter's f32-compute-then-round: operands upcast to f32,
/// the result cast back at the write (literals are rounded to the dst dtype
/// FIRST, exactly `literal_array`). Slice operands of width 1 read their base
/// element (the one-element-per-thread broadcast).
fn emit_scalar_elem(
    out: &mut Emitter,
    p: &str,
    ctx: &Ctx,
    dst: &TensorSlice,
    operands: &[&RegOperand],
    convert: bool,
    expr: impl Fn(&[&str]) -> String,
) -> Result<(), String> {
    let (dt, doff, w) = reg_slice_parts(dst)?;
    let dflat = flat_name(dt, ctx)?;
    let convert = convert && matches!(dt.dtype, DType::F16 | DType::Bf16);
    let mut elems: Vec<String> = Vec::with_capacity(operands.len());
    for op in operands {
        elems.push(match op {
            RegOperand::Literal(l) => {
                let t = typed_scalar(dt.dtype, *l)?;
                if convert {
                    format!("T.float32({t})")
                } else {
                    t
                }
            }
            RegOperand::Slice(s) => {
                let (t, off, sw) = reg_slice_parts(s)?;
                if t.dtype != dt.dtype {
                    return Err(format!(
                        "codegen: reg operand dtype {:?} != dst dtype {:?} — \
                         the interpreter coerces operands per-op; use an explicit RegCvt",
                        t.dtype, dt.dtype
                    ));
                }
                if sw != 1 && sw != w {
                    return Err(format!(
                        "codegen: reg operand width {sw} must be 1 or the dst width {w} \
                         (the interpreter matches the dst or broadcasts one element per thread)"
                    ));
                }
                let idx = flat_elem_idx(off, if sw == 1 { "0" } else { "_i" }, ctx)?;
                let e = format!("{}[{idx}]", flat_name(t, ctx)?);
                if convert {
                    format!("T.float32({e})")
                } else {
                    e
                }
            }
        });
    }
    let body = expr(&elems.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let body = if convert {
        format!("T.{}({body})", dtype_str(dt.dtype))
    } else {
        body
    };
    let d_idx = flat_elem_idx(doff, "_i", ctx)?;
    out.push_str(&format!("{p}for _i in range({w}):\n"));
    out.push_str(&format!("{p}    {dflat}[{d_idx}] = {body}\n"));
    Ok(())
}

/// The `{name}_flat` raw per-thread storage name for a REG tensor declared
/// with auxiliary views (see `RegAuxViews`).
fn flat_name(t: &Arc<Tensor>, ctx: &Ctx) -> Result<String, String> {
    Ok(format!("{}_flat", ctx.tensor_name(t.id)?))
}

/// `off + i` text for flat-view indices (simplified when the base is 0).
fn flat_add(off_s: &str, i: usize) -> String {
    if off_s == "0" {
        i.to_string()
    } else {
        format!("{off_s} + {i}")
    }
}

/// Python bool literal.
fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

/// Validate the ldmatrix/stmatrix SMEM operand: a rank-2 `(row, col)` slice of
/// exactly one 8-element b16 row (the per-thread row-address form the
/// interpreter's matrix model requires).
fn check_matrix_smem_row(s: &TensorSlice, label: &str) -> Result<(), String> {
    if s.offsets.len() != 2 || s.shape.len() != 2 {
        return Err(format!(
            "codegen: {label} SMEM operand must be a rank-2 (row, col) slice"
        ));
    }
    if as_int(&s.shape[0]) != Some(1) || as_int(&s.shape[1]) != Some(8) {
        return Err(format!(
            "codegen: {label} SMEM operand must be one row of eight b16 elements \
             (got shape {:?})",
            s.shape
        ));
    }
    if !matches!(s.tensor.dtype, DType::F16 | DType::Bf16) {
        return Err(format!(
            "codegen: {label} SMEM operand dtype {:?} must be a 16-bit type (b16 matrix)",
            s.tensor.dtype
        ));
    }
    Ok(())
}

/// Shared lowering for `RegAdd`/`RegSub`: `Tx.wg.{op}(dst, lhs, rhs)` over the
/// wg views when the scope is provably warpgroup-full, else the per-thread
/// scalar form (the elementwise dispatches reject a narrowed intra; the
/// scalar form IS the interpreter's per-thread semantics).
/// `rounding=rm` (the interpreter's post-op floor) has no TIRx elementwise
/// form — fail closed rather than silently skip the floor.
#[allow(clippy::too_many_arguments)]
fn emit_reg_binary(
    out: &mut Emitter,
    p: &str,
    dst: &TensorSlice,
    lhs: &RegOperand,
    rhs: &RegOperand,
    rounding: Rounding,
    op: &str,
    ctx: &Ctx,
    scope: &ScopeInfo,
) -> Result<(), String> {
    if rounding != Rounding::Rn {
        return Err(format!(
            "codegen: Reg{op} rounding=rm has no TIRx lowering (the elementwise \
             ops carry no post-op floor; rn only)"
        ));
    }
    let (t, off, w) = reg_slice_parts(dst)?;
    if !matches!(t.dtype, DType::F16 | DType::Bf16 | DType::F32) {
        return Err(format!(
            "codegen: Reg{op} dst dtype {:?} has no lowering (float dsts only)",
            t.dtype
        ));
    }
    if wg_elem_ok(scope) && reg_op_slices_full(dst, &[lhs, rhs], ctx)? {
        let dst_s = emit_reg_view_slice(out, p, t, off, w, ctx)?;
        let lhs_s = emit_wg_reg_operand(lhs, t.dtype, out, p, ctx)?;
        let rhs_s = emit_wg_reg_operand(rhs, t.dtype, out, p, ctx)?;
        out.push_str(&format!("{p}Tx.wg.{op}({dst_s}, {lhs_s}, {rhs_s})\n"));
        Ok(())
    } else {
        let sym = if op == "add" { "+" } else { "-" };
        emit_scalar_elem(out, p, ctx, dst, &[lhs, rhs], true, |e| {
            format!("{} {sym} {}", e[0], e[1])
        })
    }
}

/// Per-thread register count (in dtype ELEMENTS) of one tcgen05 ld/st atom
/// issue: `.32x32b`/`.16x64b` hold `num` b32 regs per thread, `.16x128b`
/// `2*num`, `.16x256b` `4*num`; 16-bit dtypes pack two elements per b32 reg.
/// (PTX ISA Table 49; M=64 slab only — the M=128 `.16x*b` two-issue form is
/// rejected by the lowering.)
fn tcgen05_frag_regs(shape: &LdStShape, num: usize, dtype: DType) -> usize {
    let b32 = match shape {
        LdStShape::B32x32 | LdStShape::B16x64 => num,
        LdStShape::B16x128 => 2 * num,
        LdStShape::B16x256 => 4 * num,
        // No codegen lowering (rejected at the op site); sized so the width
        // walk stays total.
        LdStShape::B16x32Bx2 => num,
    };
    let e32 = if matches!(dtype, DType::F16 | DType::Bf16) {
        2
    } else {
        1
    };
    b32 * e32
}

/// The `.16x*b` atom view's column count for a per-thread width-W fragment:
/// the M=64 warpgroup frag is `(64, K)` with `64*K/128 = W` per-thread
/// elements, so `K = 2W`; the implied `.xN` rep is `K / (col_factor *
/// elems_per_b32)` and must be integral.
fn atom_frag_cols(shape: &str, width: usize, dtype: DType) -> Result<usize, String> {
    let factor = match shape {
        "16x64b" => 2usize,
        "16x128b" => 4,
        "16x256b" => 8,
        other => {
            return Err(format!(
                "codegen: tcgen05 shape {other} has no atom-fragment view \
                 (only 16x64b/16x128b/16x256b)"
            ))
        }
    };
    let e32 = match dtype {
        DType::F32 => 1usize,
        DType::F16 | DType::Bf16 => 2,
        other => {
            return Err(format!(
                "codegen: tcgen05 {shape} fragment dtype {other:?} has no lowering \
                 (only f32/f16/bf16)"
            ))
        }
    };
    let k = 2 * width;
    if k % (factor * e32) != 0 {
        return Err(format!(
            "codegen: tcgen05 {shape} fragment of {width} per-thread {dtype:?} elements \
             has no integral .xN rep (K = {k} cols is not a multiple of {})",
            factor * e32
        ));
    }
    Ok(k)
}

/// Every MBarRef a statement names (for peer discovery).
fn stmt_mbar_refs(s: &Stmt) -> Vec<&super::mbar::MBarRef> {
    use Stmt::*;
    match s {
        MBarrierInit {
            mbar,
            count: _,
            stage: _,
        }
        | MBarrierArrive {
            mbar,
            stage: _,
            count: _,
        }
        | MBarrierWait {
            mbar,
            stage: _,
            phase: _,
        }
        | MBarrierExpectTx {
            mbar,
            bytes: _,
            stage: _,
        }
        | MBarrierArriveExpectTx {
            mbar,
            bytes: _,
            stage: _,
        }
        | Tcgen05Commit {
            mbar,
            stage: _,
            cta_group: _,
            multicast_cta_mask: _,
        } => vec![mbar],
        TmaLoad {
            dst: _,
            src: _,
            mbar,
            coords: _,
            shape: _,
            gmem_shape: _,
            mbar_stage: _,
            multicast_cta_mask: _,
            cache_hint: _,
            prefetch_tensormap: _,
            cta_group: _,
        } => vec![mbar],
        ClcTryCancel {
            scheduler: _,
            handle: _,
            mbar,
            stage: _,
            cta_group: _,
        } => vec![mbar],
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

/// Conservatively decide whether a scalar is provably `>= 0`, given the
/// provably-nonnegative var ids. Used to rewrite `x % d` / `x // d` into
/// `T.truncmod`/`T.truncdiv` for a non-pow2 positive-literal `d` — only valid
/// when trunc (toward 0) and floor (toward -inf) agree, i.e. for non-negative
/// `x`. (The pow2 bit-op rewrite needs no sign assumption: floormod/floordiv
/// by `2^k` IS `& (2^k-1)` / `>> k` in two's complement, negative dividends
/// included — see `emit_expr`.)
///
/// The pipeline-phase parities (`occ % 2`, `(occ + 1) % 2`, ring/slot indices)
/// are all built from non-negative loop counters, scope ids, `FloorDiv`s, and
/// `Mod`s, so they qualify; a sentinel-negative scalar (the scheduler's
/// `task_id`/`bcast_id == -1`) is never proven non-negative: scope ids are
/// hardware-non-negative, but a `Var` qualifies ONLY via `nonneg_vars` — a
/// ForLoop induction variable with a non-negative-literal start and
/// positive-literal step, or a scalar whose every definition is provably
/// non-negative (a mailbox/`ClcQueryCancel` load is never assumed so).
fn is_nonneg(sv: &ScalarValue, nonneg_vars: &HashSet<u32>) -> bool {
    match sv {
        ScalarValue::Int(i) => *i >= 0,
        // Scope ids (lane/warp/wg/cta/tid) are hardware non-negative. A var is
        // non-negative only when the analysis proved it (loop induction vars
        // with non-negative literal bounds; scalars with non-negative defs).
        ScalarValue::Var(v) => nonneg_vars.contains(&v.id.0),
        ScalarValue::Scope(_) => true,
        ScalarValue::Expr(e) => match e.op {
            // FloorDiv / Mod by a positive divisor of a non-negative dividend is
            // non-negative; Mul/Add of non-negatives stay non-negative.
            ScalarOp::FloorDiv | ScalarOp::Mod => {
                is_nonneg(&e.args[0], nonneg_vars) && is_nonneg(&e.args[1], nonneg_vars)
            }
            ScalarOp::Mul | ScalarOp::Add | ScalarOp::Min | ScalarOp::Max => {
                e.args.iter().all(|a| is_nonneg(a, nonneg_vars))
            }
            // Anything else (Sub, Neg, Select, comparisons, ...) is not assumed >= 0.
            _ => false,
        },
    }
}

/// Collect the provably-nonnegative var ids `is_nonneg` consults (see it).
///   * ForLoop induction vars with a non-negative-literal start and a
///     positive-literal step (a rolled `T.serial` counter runs start, start+step,
///     ... and stays non-negative);
///   * scalar vars whose EVERY definition is provably non-negative: the
///     `ScalarDef` initial (a `ScalarInitial::Tensor` mailbox load is never
///     assumed non-negative — it can carry the -1 sentinel), every
///     `ScalarStore` source, every `ScalarLet` value, and every `ShuffleSync`
///     source. Seeded
///     optimistically, then dropped to a fixpoint (a ring counter defined
///     through its own non-negative update chain converges). A
///     `ClcQueryCancel` result is never non-negative (0xFFFFFFFF -> -1 on drain).
fn collect_nonneg_vars(k: &Kernel) -> HashSet<u32> {
    let mut nonneg: HashSet<u32> = HashSet::new();
    fn seed_loop_vars(stmts: &[Stmt], nonneg: &mut HashSet<u32>) {
        for s in stmts {
            if let Stmt::ForLoop {
                var,
                start,
                stop: _,
                step,
                body: _,
                unroll: _,
            } = s
            {
                if as_int(start).is_some_and(|v| v >= 0) && as_int(step).is_some_and(|v| v >= 1) {
                    nonneg.insert(var.id.0);
                }
            }
            for body in s.child_bodies() {
                seed_loop_vars(body, nonneg);
            }
        }
    }
    seed_loop_vars(&k.body, &mut nonneg);

    // Gather every scalar-var definition.
    #[derive(Default)]
    struct ScalarDefs {
        /// Def exprs that must ALL be non-negative for the var to qualify.
        exprs: Vec<ScalarValue>,
        /// A definition that can never be proven non-negative (mailbox load /
        /// CLC query) — the var is excluded outright.
        poisoned: bool,
    }
    fn collect_scalar_defs(stmts: &[Stmt], defs: &mut HashMap<u32, ScalarDefs>) {
        for s in stmts {
            match s {
                Stmt::ScalarDef { var, initial } => {
                    let e = defs.entry(var.id.0).or_default();
                    match initial {
                        ScalarInitial::Value(v) => e.exprs.push(v.clone()),
                        ScalarInitial::Tensor(_) => e.poisoned = true,
                    }
                }
                Stmt::ScalarStore { var, value } => {
                    defs.entry(var.id.0).or_default().exprs.push(value.clone());
                }
                Stmt::ScalarLet { var, value } => {
                    defs.entry(var.id.0).or_default().exprs.push(value.clone());
                }
                Stmt::ShuffleSync {
                    var,
                    src,
                    src_lane: _,
                } => {
                    defs.entry(var.id.0).or_default().exprs.push(src.clone());
                }
                Stmt::ClcQueryCancel {
                    scheduler: _,
                    var,
                    handle: _,
                } => {
                    defs.entry(var.id.0).or_default().poisoned = true;
                }
                _ => {}
            }
            for body in s.child_bodies() {
                collect_scalar_defs(body, defs);
            }
        }
    }
    let mut defs: HashMap<u32, ScalarDefs> = HashMap::new();
    collect_scalar_defs(&k.body, &mut defs);

    // Fixpoint: optimistically assume every unpoisoned scalar var non-negative,
    // then drop any var with a definition not provably non-negative under the
    // current hypothesis until stable.
    let mut hyps: HashSet<u32> = defs
        .iter()
        .filter(|(_, d)| !d.poisoned)
        .map(|(id, _)| *id)
        .collect();
    loop {
        let mut dropped = Vec::new();
        for id in &hyps {
            let d = &defs[id];
            let mut trial: HashSet<u32> = hyps.clone();
            trial.extend(nonneg.iter().copied());
            if d.exprs.iter().any(|e| !is_nonneg(e, &trial)) {
                dropped.push(*id);
            }
        }
        if dropped.is_empty() {
            break;
        }
        for id in dropped {
            hyps.remove(&id);
        }
    }
    nonneg.extend(hyps);
    nonneg
}

/// If `sv` is a positive power-of-two literal `2^k` (k >= 1), return `k`. Drives the
/// strength-reduction of `% 2^k` -> `& (2^k - 1)` and `// 2^k` -> `>> k`.
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
    // ---- Div/mod strength reduction (generic; see `is_nonneg`) ----
    // The nymph IR's ring/slot/phase indices and the L2-swizzle coords are all
    // built from provably non-negative counters and scope ids. TVM lowers a
    // SIGNED `floordiv`/`floormod` (what Python `//`/`%` parse to) with a
    // sign-correction tail: `floormod(x,d)` -> `(x % d) + (d & ((x % d) >> 31))`
    // and `floordiv(x,d)` -> `(x / d) + ((x % d) >> 31)` — pure wasted integer
    // ALU when the dividend is non-negative (every correction term is 0), and
    // recomputed per index. Two rewrites:
    //
    //   * divisor == 2^k  -> bit ops: `% 2^k` => `(x) & (2^k - 1)`, `// 2^k` =>
    //     `(x) >> k`. NO sign gate: in two's complement, Python's floormod /
    //     floordiv by 2^k are EXACTLY `& (2^k-1)` / arithmetic `>> k` for
    //     negative dividends too (`-1 % 4 == 3 == -1 & 3`, `-1 // 4 == -1 ==
    //     -1 >> 2`), so the rewrite is an unconditional identity. (It also
    //     dodges a TIRx->CUDA simplifier bug that mis-folds
    //     `floormod(floordiv(..),2)` to 0; bit ops are left intact.)
    //   * other positive literal (e.g. 5) -> `T.truncdiv` / `T.truncmod`:
    //     numerically identical to floordiv/floormod ONLY for a non-negative
    //     `x` (trunc rounds toward 0, floor toward -inf), so this path is
    //     gated on the provable-`is_nonneg` set (`ctx.nonneg_vars`), never a
    //     blanket var assumption — a sentinel-negative scalar keeps its
    //     correct floordiv/floormod form.
    //
    // `&`, `>>` bind looser than `% // * + -` in Python: parenthesize the dividend and
    // wrap the whole result for any parent binding tighter than bitand/shift.
    if e.op == ScalarOp::Mod || e.op == ScalarOp::FloorDiv {
        if let Some(k) = as_pow2_shift(&e.args[1]) {
            let lhs = emit_scalar_prec(&e.args[0], ctx, 0)?;
            let s = if e.op == ScalarOp::Mod {
                format!("({lhs}) & {mask}", mask = (1u64 << k) - 1)
            } else {
                format!("({lhs}) >> {k}")
            };
            return Ok(if parent_prec > 0 { format!("({s})") } else { s });
        }
        // Divisor 1 needs no sign gate either: `x % 1 == 0` and `x // 1 == x`
        // under BOTH floor and trunc semantics, for every x.
        if as_int(&e.args[1]) == Some(1) || is_nonneg(&e.args[0], &ctx.nonneg_vars) {
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
/// absolute physical base plus the MMA's n. An M=64 accumulator (gdn's BT=64
/// GEMMs) is the non-ws datapath F — the `tmem_f` view's scattered-row layout.
fn emit_tmem_dst(op: &TmemOperand, n: u32, m: u32, ctx: &Ctx) -> Result<String, String> {
    let col_s = emit_scalar(&op.col, ctx)?;
    let hi = add_bound(&op.col, &ScalarValue::Int(i64::from(n)), ctx)?;
    if m == 64 {
        return Ok(format!("tmem_f[:, {col_s}:{hi}]"));
    }
    Ok(format!("tmem[:, {col_s}:{hi}]"))
}

// ===========================================================================
// statement walk
// ===========================================================================

/// Emit one top-level unit (a plain stmt or a chained decision tree).
fn emit_top_unit(
    out: &mut Emitter,
    unit: &TopUnit,
    indent: usize,
    ctx: &Ctx,
    scope: &ScopeInfo,
) -> Result<(), String> {
    match unit {
        TopUnit::Stmt(s) => emit_stmt(out, s, indent, ctx, scope),
        TopUnit::Chain(c) => emit_if_chain(out, c, indent, ctx, scope),
    }
}

/// Emit a chained if/else decision tree of warp/warpgroup-equality branches
/// (canon's role-dispatch shape). The `else` arm nests the next chain link; a
/// branch's warp-level `inner` chain emits at the end of its body.
fn emit_if_chain(
    out: &mut Emitter,
    chain: &IfChain,
    indent: usize,
    ctx: &Ctx,
    scope: &ScopeInfo,
) -> Result<(), String> {
    let p = pad(indent);
    let child = child_scope_info(&chain.cond, scope, ctx.num_warps);
    out.push_str(&format!("{p}if {}:\n", emit_scalar(&chain.cond, ctx)?));
    emit_body(out, &chain.body, indent + 1, ctx, &child)?;
    if let Some(inner) = &chain.inner {
        emit_if_chain(out, inner, indent + 1, ctx, &child)?;
    }
    if let Some(e) = &chain.else_ {
        out.push_str(&format!("{p}else:\n"));
        emit_if_chain(out, e, indent + 1, ctx, scope)?;
    }
    Ok(())
}

/// Emit a statement list. Every body walk goes through here so the run-level
/// coalescings below apply uniformly at any nesting depth.
///
/// Coalesces the fence/sync that follows a run of consecutive `MBarrierWait`s:
/// the template issues all the `try_wait`s, then ONE
/// `tcgen05.fence.after_thread_sync()` + ONE `T.cuda.cta_sync()` — and only at
/// function scope (inside a single-warp/wg branch not all CTA threads reach a
/// CTA-wide `__syncthreads`).
fn emit_body(
    out: &mut Emitter,
    stmts: &[Stmt],
    indent: usize,
    ctx: &Ctx,
    scope: &ScopeInfo,
) -> Result<(), String> {
    let p = pad(indent);
    let mut i = 0;
    while i < stmts.len() {
        if matches!(stmts[i], Stmt::MBarrierWait { .. }) {
            // Emit every `try_wait` in the run, then one fence. The
            // `T.cuda.cta_sync()` (a CTA-wide `__syncthreads`) is only emitted at
            // function scope: inside a single-warp / single-warpgroup branch not
            // all CTA threads reach it, so the barrier would deadlock / raise an
            // illegal instruction. Within one branch the threads are already
            // lockstep and the mbarrier wait gives the async-engine ordering.
            //
            // A *peer* wait (remote_coord set, i.e. a `try_wait` on a
            // `map_shared_rank`-remapped DSMEM address) is SKIPPED:
            // `mbarrier.try_wait` is only legal on a local shared address, so
            // the remapped form raises `cudaErrorIllegalInstruction` on sm_100
            // (verified). The peer CTA's TMA completion is instead ordered by
            // routing BOTH CTAs' TMA loads to the leader CTA's single barrier
            // (the canonical cta_group=2 pattern): each CTA's `Tx.copy_async`
            // signals the leader's barrier via `map_shared_rank(.., 0)`, the
            // leader issues one `arrive.expect_tx` for the FULL cluster byte
            // count, and waits its OWN local barrier (which both CTAs fill).
            // See `TmaLoad` / `MBarrierArriveExpectTx`.
            let mut j = i;
            let mut emitted_any = false;
            while j < stmts.len() && matches!(stmts[j], Stmt::MBarrierWait { .. }) {
                if let Stmt::MBarrierWait { mbar, phase, stage } = &stmts[j] {
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
                // visibility ordering across the whole CTA needs it. In branch scope (the hot
                // MMA / loader / epilogue loops) the mbarrier handshake alone orders the async
                // engines (TMA→smem_full→MMA, MMA→tmem_full→epilogue) — exactly as the canonical
                // kernel does, which emits ZERO `tcgen05.fence.after_thread_sync()` after its
                // loop waits. Emitting one per wait over-fenced the hot loop: the FENCE/MEMBAR
                // serialized MMA issue and left the tensor core idle, the exact bench gap.
                out.push_str(&format!("{p}T.ptx.tcgen05.fence.after_thread_sync()\n"));
                out.push_str(&format!("{p}T.cuda.cta_sync()\n"));
            }
            i = j;
            continue;
        }
        emit_stmt(out, &stmts[i], indent, ctx, scope)?;
        i += 1;
    }
    Ok(())
}

/// ZERO-INFERENCE guard rule (user-mandated): codegen NEVER synthesizes a
/// single-issue guard from the statically-computed thread scope. Hardware
/// single-issue ops — TmaLoad/TmaStore/CpAsyncBulkS2Cluster (the TMA issue
/// family), Tcgen05Mma/Tcgen05Cp/Tcgen05Commit, MBarrierInit (an unguarded
/// init is a double-init error in both models), ClcTryCancel (stream-level in
/// the interpreter) — are legal ONLY under an explicit single-lane `If` in
/// the IR (the `if_elected` sugar or a provably one-lane-per-warp predicate);
/// the validator enforces that at build (`single_issue_scope`), and this
/// helper is the codegen-side defense: bare under `Elected`, a hard error
/// anywhere else. Per-thread ops (mbarrier arrive/expect_tx, store_scalar)
/// emit per-thread — the interpreter applies them once per executing lane.
fn emit_single_issue(
    out: &mut Emitter,
    p: &str,
    scope: &ScopeInfo,
    op: &str,
    body: &str,
) -> Result<(), String> {
    if scope.scope == Scope::Elected {
        out.push_str(&format!("{p}{body}\n"));
        Ok(())
    } else {
        Err(format!(
            "codegen: {op} is a hardware single-issue op but its scope is not \
             single-lane — wrap it in an explicit elected `If` (the validator's \
             single_issue_scope rule rejects this at build)"
        ))
    }
}

fn emit_stmt(
    out: &mut Emitter,
    stmt: &Stmt,
    indent: usize,
    ctx: &Ctx,
    scope: &ScopeInfo,
) -> Result<(), String> {
    use Stmt::*;
    let p = pad(indent);
    match stmt {
        // ---- REG fragments: emit INLINE at the TensorDef site (canon's loop-local
        // `T.wg_reg_tile(...)`), NOT hoisted to function scope. A function-scope register
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
                            "{p}{name} = T.alloc_tcgen05_ldst_frag(\"{instr_shape}\", ({wg_threads}, {width}), \"{dt}\")\n",
                            p = pad(indent),
                            wg_threads = WG_THREADS,
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
                    // Auxiliary-view form (see `RegAuxViews`): the plain wg tile for
                    // full-extent `Tx.wg.*` ops, plus `{name}.local()` — the
                    // TIRx-sanctioned per-thread storage view (mem-only layout, raw
                    // element access legal) for the warp-matrix intrinsics, atom
                    // fragments, and the scalar elementwise forms. (The inverted
                    // `alloc_local + view(128, W, wg_layout)` form is physically
                    // inconsistent: its Apply maps (tid, j) to flat[tid + j].)
                    if let Some(aux) = ctx.reg_aux_views.get(&tensor.id).filter(|a| a.flat) {
                        out.push_str(&format!(
                            "{p}{name} = T.wg_reg_tile({width}, dtype=\"{dt}\")\n",
                            p = pad(indent),
                            dt = dtype_str(tensor.dtype),
                        ));
                        out.push_str(&format!(
                            "{p}{name}_flat = {name}.local()\n",
                            p = pad(indent)
                        ));
                        if let Some(shape) = aux.atom_shape {
                            let k_cols = atom_frag_cols(shape, width, tensor.dtype)?;
                            out.push_str(&format!(
                                "{p}{name}_atom = {name}_flat.view(64, {k_cols}, layout=tcgen05_atom_layout(\"{shape}\", (64, {k_cols}), \"{dt}\"))\n",
                                p = pad(indent),
                                dt = dtype_str(tensor.dtype),
                            ));
                        }
                        if aux.flat_u32 {
                            out.push_str(&format!(
                                "{p}{name}_flat_u32 = {name}_flat.view(\"uint32\")\n",
                                p = pad(indent),
                            ));
                        }
                        if let Some(ab) = aux.flat_ab {
                            out.push_str(&format!(
                                "{p}{name}_flat_ab = {name}_flat.view(\"{ab_dt}\")\n",
                                p = pad(indent),
                                ab_dt = dtype_str(ab),
                            ));
                        }
                        return Ok(());
                    }
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

        // ---- TMEM alloc / dealloc / relinquish (already under the prologue's
        // warp==0 guard) ----
        // The generated code carries exactly ONE TMEM view buffer (`tmem`) based at
        // column 0, one alloc writing `tmem_addr`, and one dealloc of `tmem_addr[0]`
        // — validate has already proven the kernel fits that shape (base_col==0, one
        // live band at a time, alloc only before relinquish, and the kernel-level
        // cta_group on every op). The arms below still check the fields they cannot
        // honor instead of dropping them: codegen is reachable without validate, and
        // a silently-dropped field is how "validate one semantics, run another" used
        // to happen here.
        TmemAlloc {
            base_col,
            n_cols,
            cta_group,
        } => {
            if *base_col != 0 {
                return Err(format!(
                    "codegen: TmemAlloc base_col={base_col} has no lowering (the TMEM view is base-0)"
                ));
            }
            if *cta_group != ctx.cta_group {
                return Err(format!(
                    "codegen: TmemAlloc cta_group={} != kernel cta_group={}",
                    cta_group, ctx.cta_group
                ));
            }
            out.push_str(&format!(
                "{p}T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols={n_cols}, cta_group={cta_group})\n",
            ));
            Ok(())
        }
        TmemDealloc {
            base_col,
            n_cols,
            cta_group,
        } => {
            if *base_col != 0 {
                return Err(format!(
                    "codegen: TmemDealloc base_col={base_col} has no lowering (the TMEM view is base-0)"
                ));
            }
            if *cta_group != ctx.cta_group {
                return Err(format!(
                    "codegen: TmemDealloc cta_group={} != kernel cta_group={}",
                    cta_group, ctx.cta_group
                ));
            }
            out.push_str(&format!(
                "{p}T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols={n_cols}, cta_group={cta_group})\n"
            ));
            Ok(())
        }
        TmemRelinquish { cta_group } => {
            if *cta_group != ctx.cta_group {
                return Err(format!(
                    "codegen: TmemRelinquish cta_group={} != kernel cta_group={}",
                    cta_group, ctx.cta_group
                ));
            }
            // 1:1 translation — the IR makes the permit release explicit (it used
            // to ride along with the dealloc implicitly).
            out.push_str(&format!(
                "{p}T.ptx.tcgen05.relinquish_alloc_permit(cta_group={cta_group})\n"
            ));
            Ok(())
        }

        // ---- structural ----
        If { cond, then_body } => {
            let child = child_scope_info(cond, scope, ctx.num_warps);
            // The `if_elected` sugar (`lane_id == 0`) emits canon's hardware
            // forms instead of the naive `lane_id == 0` predicate: normally
            // `T.ptx.elect_sync()` (one elected lane per warp, no modulo
            // recompute per site); when the enclosed set is exactly CTA thread
            // {(0, 0)} (an elect nested in the warp-0 prologue branch), the
            // CTA-uniform `T.cuda.thread_rank() == 0` — canon's prologue form.
            if as_lane_zero_equality(cond) {
                // The thread_rank form is canon's PROLOGUE guard (one-time init
                // code). A warp-0 elected region that CONTAINS a persistent loop
                // (a worker role that happens to live on warp 0 — the nvfp4 MMA
                // warp) is a hot loop guard, and canon writes `elect_sync()`
                // there: the thread_rank predicate (a %tid.x read + compare on
                // the vector path) measurably degrades ptxas's handling of the
                // loop (nvfp4 1024: 6.26 -> 5.22 us on the guard form alone).
                // So the narrowing applies only to loop-free (one-time) bodies.
                let thread_rank0 = !body_has_loop(then_body)
                    && child.set.as_ref().and_then(|s| s.single_thread()) == Some((0, 0));
                // `barrier.cluster.wait` is WARP-COLLECTIVE and deadlocks when
                // only the elected lane waits: peel any leading
                // ClusterBarrierWaits out of the elect to the enclosing (warp)
                // scope, exactly like #18 peeled them out of an elected Role.
                let n_lead = then_body
                    .iter()
                    .take_while(|s| matches!(s, Stmt::ClusterBarrierWait))
                    .count();
                for _ in 0..n_lead {
                    out.push_str(&format!(
                        "{p}T.ptx.barrier.cluster.wait(acquire=True, aligned=False)\n"
                    ));
                }
                let guard = if thread_rank0 {
                    "T.cuda.thread_rank() == 0"
                } else {
                    "T.ptx.elect_sync()"
                };
                out.push_str(&format!("{p}if {guard}:\n"));
                emit_body(out, &then_body[n_lead..], indent + 1, ctx, &child)?;
                return Ok(());
            }
            out.push_str(&format!("{p}if {}:\n", emit_scalar(cond, ctx)?));
            emit_body(out, then_body, indent + 1, ctx, &child)?;
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
            // Emit `T.serial(...)`, NOT Python `range(...)`: T.serial is the rolled serial
            // loop with a single UNIFORM loop-induction var (one hardware counter per
            // warp), which is what keeps the ring index / operand-descriptor address math
            // in uniform registers (UIADD3/UMOV) — matching canon's `for k in T.serial(N)`.
            // A bare `range(...)` would be fully unrolled by the TVMScript parser, defeating
            // the roll. step must be 1 for T.serial.
            let range = if step_s == "1" {
                if start_s == "0" {
                    format!("T.serial({stop_s})")
                } else {
                    format!("T.serial({start_s}, {stop_s})")
                }
            } else {
                format!("T.serial({start_s}, {stop_s}, {step_s})")
            };
            // `unroll == false`: `T.serial(N, unroll=False)` — the `disable_unroll`
            // annotation lowers to `#pragma unroll 1` in the CUDA source, pinning the
            // loop rolled at ptxas (otherwise ptxas re-unrolls a merged MMA loop and
            // re-inflates the whole-function static size that gates uniform placement).
            let range = if !unroll {
                format!("{}, unroll=False)", range.strip_suffix(')').unwrap())
            } else {
                range
            };
            out.push_str(&format!("{p}for {name} in {range}:\n"));
            emit_body(out, body, indent + 1, ctx, scope)?;
            Ok(())
        }
        // Grid-stride scheduler loop: each launched cluster (cta_id // cta_group) strides
        // by the cluster count through the task space. The trip count is runtime (the
        // start is runtime), so T.serial stays rolled — it cannot unroll. The task var
        // reads as `v{id}` in the body (the per-branch `local_iter`/`work_idx` math the
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
        // (same indent + scope) — the CLC primitives inside translate 1:1. The
        // `scheduler` metadata is sim-only (it keys the oracles there).
        SchedulerImpl { scheduler: _, body } => {
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
        // (the kernel's own `sched_arr` full barrier). Single-issue (the enclosing
        // elect guard), exactly like canon's `if T.ptx.elect_sync(): clc_try_cancel(...)`.
        ClcTryCancel {
            scheduler: _, // sim-only metadata (keys the CLC handle slot; no emission arg)
            handle,
            mbar,
            stage,
            // The multicast width is implied by TIRx's clc lowering, so the field
            // has no emission site — but it is NOT unchecked: validate requires it
            // == the kernel-level cta_group, and the same fail-closed re-check
            // runs here (codegen is reachable without validate), mirroring
            // TmaLoad's `cta_group` guard below.
            cta_group,
        } => {
            if *cta_group != ctx.cta_group {
                return Err(format!(
                    "codegen: ClcTryCancel cta_group={} != kernel cta_group={}",
                    cta_group, ctx.cta_group
                ));
            }
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
            emit_single_issue(
                out,
                &p,
                scope,
                "clc_try_cancel",
                &format!(
                    "T.ptx.clc_try_cancel(T.address_of({handle_name}[0]), T.address_of({mbar_name}[{slot}]))"
                ),
            )?;
            Ok(())
        }
        // CLC handle decode: 1:1 to `clusterlaunchcontrol.query_cancel`, DEFINING the
        // scalar (the cancelled cluster's first ctaid.x, or 0xFFFFFFFF -> -1 as int32).
        // Unguarded — every thread of the branch reads the same handle and gets the
        // same value (a pure uniform decode), like `ShuffleSync`.
        ClcQueryCancel {
            scheduler: _, // sim-only metadata (keys the handle-slot read)
            var,
            handle,
        } => {
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
        // Single-assignment `let`: `name: T.let[T.int32] = expr` — an immutable
        // SSA `T.Bind` (canon's tile-coord decode chain form), NOT the mutable
        // `local_scalar` cell a `T.int32 = …` annotation lowers to. The SSA
        // dataflow is what lets ptxas keep the value (and everything derived
        // from it) on the uniform datapath; a mutable local forces vector regs
        // + R2UR moves at every uniform-sink use (the fp16 1024 R2UR gap).
        ScalarLet { var, value } => {
            let name = ctx
                .scalar_names
                .get(&var.id.0)
                .ok_or_else(|| format!("codegen: no name for scalar var {}", var.id.0))?;
            out.push_str(&format!(
                "{p}{name}: T.let[T.int32] = {}\n",
                emit_scalar(value, ctx)?,
                name = name
            ));
            Ok(())
        }
        // Mailbox write: `task_smem[stage, field] = <scalar>`. PER-THREAD (the
        // interpreter writes once per executing lane); a uniform value makes the
        // redundant STS harmless, and a lane-varying value is exactly what the
        // IR asked for — no single-issue guard may be inferred.
        StoreScalar { dst, value } => {
            let dst_s = emit_scalar_addr(dst, ctx)?;
            out.push_str(&format!("{p}{dst_s} = {}\n", emit_scalar(value, ctx)?));
            Ok(())
        }

        // ---- mbarrier (every op carries an optional `stage` -> slot index) ----
        // init is effectively single-issue: an unguarded init is a double-init
        // error in the interpreter AND on hardware, so it must sit under an
        // explicit single-lane `If` (validator `single_issue_scope`) — codegen
        // emits bare there and fails closed everywhere else (zero-inference).
        MBarrierInit { mbar, count, stage } => {
            let slot_ptr = mbar_slot_ptr(mbar, stage, ctx)?;
            emit_single_issue(
                out,
                &p,
                scope,
                "mbarrier.init",
                &format!("T.ptx.mbarrier.init({slot_ptr}, {count})"),
            )
        }
        MBarrierArriveExpectTx { mbar, bytes, stage } => {
            // Cluster TMA barrier (leader-routed): issue ONE expect_tx on the leader's
            // (CTA-0) barrier for the FULL cluster byte count (both CTAs' loads land
            // here), and only on the leader CTA (cbx==0) so it is counted once. The IR's
            // `bytes` is the per-CTA byte count, so multiply by cta_group. The `cbx == 0`
            // selection is IR-driven (the mbar's `leader_routed` flag), not inferred;
            // the single-lane issue itself must come from an explicit elected `If`.
            if let Some(view) = ctx.tma_leader_view_for(mbar.mbar.id) {
                let slot = stage
                    .as_ref()
                    .map(|s| emit_scalar(s, ctx))
                    .transpose()?
                    .unwrap_or_else(|| "0".to_string());
                let total_bytes = *bytes as u64 * ctx.cta_group as u64;
                out.push_str(&format!("{p}if cbx == 0:\n"));
                return emit_single_issue(
                    out,
                    &format!("{p}    "),
                    scope,
                    "mbarrier.arrive.expect_tx (leader-routed)",
                    &format!(
                        "T.ptx.mbarrier.arrive.expect_tx({view}.ptr_to([{slot}]), {total_bytes})"
                    ),
                );
            }
            // Local (non-leader) expect_tx is PER-THREAD (the interpreter arrives
            // and adds tx once per executing lane; the gdn kernel elects one lane
            // explicitly). No guard may be inferred.
            let slot_ptr = mbar_slot_ptr(mbar, stage, ctx)?;
            out.push_str(&format!(
                "{p}T.ptx.mbarrier.arrive.expect_tx({slot_ptr}, {bytes})\n"
            ));
            Ok(())
        }
        // expect_tx is PER-THREAD like arrive (the interpreter applies it once
        // per executing lane) — emit per-thread, no inferred guard.
        MBarrierExpectTx { mbar, bytes, stage } => {
            let slot_ptr = mbar_slot_ptr(mbar, stage, ctx)?;
            out.push_str(&format!(
                "{p}T.ptx.mbarrier.expect_tx({slot_ptr}, {bytes})\n"
            ));
            Ok(())
        }
        MBarrierArrive { mbar, count, stage } => {
            // Two arrive forms:
            //   * LOCAL (remote_coord=None): the implicit count-of-1 form
            //     `T.ptx.mbarrier.arrive(bar)`. (The 2nd positional arg is `remote`,
            //     NOT a count — so a count must never be passed positionally here.)
            //   * CROSS-CTA (remote_coord=Some(c)): the cluster form on the LOCAL
            //     barrier of CTA `c`: `T.ptx.mbarrier.arrive(bar, remote=c, pred=True)`
            //     — the canonical `tmem_pipe.empty.arrive(slot, remote=0, pred=True)`.
            //     This is NOT the map_shared_rank peer view; the cluster arrive remaps
            //     to CTA c internally, so we use the local mbar name + cta_id.
            //
            // `mbarrier.arrive` is a PER-THREAD instruction: the interpreter
            // arrives once per executing lane, and the barrier's expected count
            // equals the number of lanes reaching the site (gdn's 128/32-lane
            // thread barriers, canon's 256 all-thread arrivals). Emitting it
            // under a single-issue guard (elect/tid_in_wg==0) UNDERCOUNTS —
            // one arrival on a count-32/128/256 barrier never completes a
            // phase (the gdn gate_ready deadlock). Sites needing a single
            // arrival are single-thread in the IR already (an elected/single-
            // thread If narrows the executing lanes); the checker rejects any
            // over-arrival, so an unguarded per-thread emission cannot exceed
            // the validated count.
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
            out.push_str(&format!("{p}{body}\n"));
            Ok(())
        }
        MBarrierWait { mbar, phase, stage } => {
            // A peer (remote_coord) wait is skipped — illegal on a remapped DSMEM
            // address; the peer's TMA is ordered via the leader-routed smem_full instead
            // (see the coalescing note in `emit_body`). A local wait emits try_wait.
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
            // `cache_hint`: the per-load L2 eviction policy (canon's `cache_hint` on
            // its g2c loads); None = no hint (the codegen-default policy).
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
            emit_single_issue(
                out,
                &p,
                scope,
                "tma_load",
                &format!(
                    "Tx.copy_async({dst_s}, {src_s}, dispatch=\"tma_auto\", mbar={mbar_name}.ptr_to([{mbar_slot}]), cta_group={cg}{cta_mask}{cache_hint_kw}{prefetch_kw})",
                    cg = ctx.cta_group,
                ),
            )?;
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
        } => {
            // `Tx.gemm_async` fixes the operand convention the slices are emitted in:
            // A=(M,K), B=(N,K), full-datapath accumulator, kernel-level cta_group. Any
            // IR field the emitted form cannot represent is rejected — the validator
            // and value model honor these fields, so dropping one here would run a
            // different semantics than was verified. (`m/n/k` vs the operand slice
            // shapes is already enforced by the validator, incl. the trans variants.)
            // transA/transB pass through to gemm_async (TIRx computes the MN-major
            // SMEM descriptor / TMEM window): the transposed IR slice is already the
            // (K, M) / (K, N) tile. A TMEM A cannot transpose (a TIRx/hardware rule).
            if *trans_a && matches!(a, MmaOperand::Tmem(_)) {
                return Err(
                    "codegen: Tcgen05Mma trans_a on a TMEM A operand has no lowering \
                     (tcgen05 requires transA=False from TMEM)"
                        .to_string(),
                );
            }
            // PTX + the TIRx tcgen05 schedule: the B operand comes from SMEM
            // ONLY (A may be TMEM or SMEM). A TMEM B (the GDN S^T/delta/NV
            // readback bands) is unrealizable — the kernel must stage those
            // tiles through SMEM instead (fail closed, never silently reroute).
            if matches!(b, MmaOperand::Tmem(_)) {
                return Err("codegen: Tcgen05Mma with a TMEM B operand has no lowering \
                     (tcgen05.mma reads B from SMEM only; stage the tile through SMEM)"
                    .to_string());
            }
            // The accumulator views are row-0 anchored (the m=64 datapath-F
            // scatter and the m=128 full datapath both base at lane 0); a
            // nonzero dst lane base would write different cells than emitted.
            if as_int(&dst.row) != Some(0) {
                return Err(
                    "codegen: Tcgen05Mma dst row must be a static 0 (the TMEM views base at lane 0)"
                        .to_string(),
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
            let dst_s = emit_tmem_dst(dst, *n, *m, ctx)?;
            // The A/B operands are staged SMEM tiles: drop the leading ring index so
            // the operand is the 2D `(M, K)` / `(N, K)` MMA tile (canonical
            // `Asmem[stage, warp_id]` / `Bsmem[stage]`). A TMEM operand (the GDN
            // accumulator-readback) is an absolute (lane, col) slice of the single
            // `tmem` view — or of the `tmem_f16`/`tmem_bf16` packed view for a
            // 16-bit band. `trans` swaps the (rows, cols) window: a non-transposed
            // operand spans `rows` lanes x `k` cells, a transposed one (the GDN
            // S^T readback) spans `k` lanes x `rows` cells.
            let emit_ab = |op: &MmaOperand, rows: u32, trans: bool| -> Result<String, String> {
                match op {
                    MmaOperand::Slice(s) => emit_smem_tile(s, ctx),
                    MmaOperand::Tmem(t) => {
                        let (row_ext, cell_dim) = if trans {
                            (i64::from(*k), i64::from(rows))
                        } else {
                            (i64::from(rows), i64::from(*k))
                        };
                        let row_s = emit_scalar(&t.row, ctx)?;
                        let col_s = emit_scalar(&t.col, ctx)?;
                        let row_hi = add_bound(&t.row, &ScalarValue::Int(row_ext), ctx)?;
                        match t.dtype {
                            DType::F32 => {
                                let col_hi = add_bound(&t.col, &ScalarValue::Int(cell_dim), ctx)?;
                                Ok(format!("tmem[{row_s}:{row_hi}, {col_s}:{col_hi}]"))
                            }
                            DType::F16 | DType::Bf16 => {
                                let view = if t.dtype == DType::F16 {
                                    "tmem_f16"
                                } else {
                                    "tmem_bf16"
                                };
                                if cell_dim % 2 != 0 {
                                    return Err(format!(
                                        "codegen: Tcgen05Mma 16-bit TMEM operand cell-span \
                                         {cell_dim} is odd (packed halves come in pairs)"
                                    ));
                                }
                                // Packed halves: the element window doubles the
                                // cell column.
                                Ok(format!(
                                    "{view}[{row_s}:{row_hi}, ({col_s}) * 2:({col_s}) * 2 + {cell_dim}]"
                                ))
                            }
                            other => Err(format!(
                                "codegen: Tcgen05Mma TMEM operand dtype {other:?} has no \
                                 lowering (f32 or packed f16/bf16 only)"
                            )),
                        }
                    }
                }
            };
            let a_rows = if *cta_group == 1 { *m } else { *m / 2 };
            let b_rows = if *cta_group == 1 { *n } else { *n / 2 };
            let mut a_s = emit_ab(a, a_rows, *trans_a)?;
            let mut b_s = emit_ab(b, b_rows, *trans_b)?;
            // Runtime accum flag: literal 0/1 keep the old True/False form (every
            // existing kernel); a real scalar expr (canon's loop-carried accum cell)
            // emits as-is — `Tx.gemm_async` takes a runtime accum predicate.
            let accum_s = match accum {
                ScalarValue::Int(0) => "False".to_string(),
                ScalarValue::Int(1) => "True".to_string(),
                other => emit_scalar(other, ctx)?,
            };
            // Transposed operands (the GDN S^T / K^T reads): emit the flags only
            // when set, keeping the untransposed emission byte-identical.
            let trans_kw = |t: bool, name: &str| {
                if t {
                    format!(", {name}=True")
                } else {
                    String::new()
                }
            };
            let trans_a_kw = trans_kw(*trans_a, "transA");
            let trans_b_kw = trans_kw(*trans_b, "transB");
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
                emit_single_issue(
                    out,
                    &p,
                    scope,
                    "tcgen05_mma",
                    &format!(
                        "Tx.gemm_async({dst_s}, {a_s}, {b_s}, SFA={sfa_s}, SFB={sfb_s}, accum={accum_s}, dispatch=\"tcgen05\", cta_group={cg}{trans_a_kw}{trans_b_kw})",
                        cg = ctx.cta_group,
                    ),
                )?;
            } else {
                emit_single_issue(
                    out,
                    &p,
                    scope,
                    "tcgen05_mma",
                    &format!(
                        "Tx.gemm_async({dst_s}, {a_s}, {b_s}, accum={accum_s}, dispatch=\"tcgen05\", cta_group={cg}{trans_a_kw}{trans_b_kw})",
                        cg = ctx.cta_group,
                    ),
                )?;
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
            emit_single_issue(
                out,
                &p,
                scope,
                "tcgen05_cp",
                &format!(
                    "Tx.copy_async({name}[0:{rows}, 0:{cols}], {src_name}[{stage}, 0:{r}, 0:{c}], cta_group={cta_group})",
                    name = view.name,
                    rows = view.logical_rows,
                    cols = view.logical_cols,
                ),
            )?;
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
            emit_single_issue(
                out,
                &p,
                scope,
                "tcgen05_commit",
                &format!(
                    "T.ptx.tcgen05.commit({slot_ptr}, cta_group={cg}, cta_mask={mask})",
                    cg = ctx.cta_group,
                ),
            )?;
            Ok(())
        }

        // ---- epilogue: tcgen05_ld -> Tx.wg.copy_async, wait_ld, reg_cvt -> Tx.wg.cast,
        // reg_store -> Tx.copy. Whole warpgroup participates (no per-thread guard). ----
        Tcgen05Ld {
            dst,
            src,
            shape,
            num,
        } => {
            // Fail closed on every field the emission cannot honor — the
            // `Tx.wg.copy_async` below reads the single base-0 (128, cols) f32
            // `tmem` view, so:
            //   * shape: only .32x32b addresses exactly the (all-128-lanes,
            //     num-col) window the wg-view emission encodes. A .16x*b atom
            //     reads col_factor*num columns from a 16-lane half-slab — it
            //     lowers through the `{name}_atom` atom-layout view instead.
            //   * dtype: the tmem view is f32 and TIRx requires the fragment
            //     dtype to equal it.
            //   * row: the lane base of the read. The view always starts at
            //     lane 0, and for .32x32b row must be 0 anyway (the atom spans
            //     each warp's whole 32-lane subpartition) — anything else reads
            //     different lanes in the interpreter than on silicon.
            if *shape != LdStShape::B32x32 {
                // `.16x*b` M=64 atom read into the dual-view fragment
                // (`{name}_atom`, declared at the TensorDef). The M=128
                // two-issue form (a second ld at row=16) has no lowering.
                if src.dtype != DType::F32 {
                    return Err(format!(
                        "codegen: Tcgen05Ld shape={} dtype {:?} has no lowering \
                         (the 16-bit TMEM read path is st-only; f32 only)",
                        shape.as_str(),
                        src.dtype
                    ));
                }
                if as_int(&src.row) != Some(0) {
                    return Err(format!(
                        "codegen: Tcgen05Ld shape={} row must be a static 0 (the M=64 \
                         atom frag; row=16 is the M=128 second issue — no lowering)",
                        shape.as_str()
                    ));
                }
                let (t, off, w) = reg_slice_parts(dst)?;
                if t.dtype != src.dtype {
                    return Err(format!(
                        "codegen: Tcgen05Ld REG dtype {:?} != TMEM dtype {:?} \
                         (the interpreter requires them equal)",
                        t.dtype, src.dtype
                    ));
                }
                let k_cols = atom_frag_cols(shape.as_str(), w, t.dtype)?;
                let implied = k_cols
                    / (match shape.as_str() {
                        "16x64b" => 2,
                        "16x128b" => 4,
                        "16x256b" => 8,
                        _ => unreachable!("note_atom_shape gates the shapes"),
                    });
                if implied != *num as usize {
                    return Err(format!(
                        "codegen: Tcgen05Ld num={num} disagrees with the fragment width \
                         ({w} per-thread elements is .x{implied} for {})",
                        shape.as_str()
                    ));
                }
                let full = reg_view_width(t, ctx);
                if as_int(off) != Some(0) || w != full {
                    return Err(format!(
                        "codegen: Tcgen05Ld shape={} dst must span the whole fragment \
                         (the atom fills all {full} per-thread registers; got off={off:?}, width {w})",
                        shape.as_str()
                    ));
                }
                let aux = ctx
                    .reg_aux_views
                    .get(&t.id)
                    .and_then(|a| a.atom_shape)
                    .ok_or_else(|| {
                        format!(
                            "codegen: Tcgen05Ld shape={} dst tensor {} was not declared \
                             as an atom fragment (internal view-collection bug)",
                            shape.as_str(),
                            t.id
                        )
                    })?;
                if aux != shape.as_str() {
                    return Err(format!(
                        "codegen: Tcgen05Ld shape={} mismatches the tensor's {} atom view",
                        shape.as_str(),
                        aux
                    ));
                }
                let col_s = emit_scalar(&src.col, ctx)?;
                let name = ctx.tensor_name(t.id)?;
                out.push_str(&format!(
                    "{p}Tx.wg.copy_async({name}_atom[:, :], tmem[0:64, {col_s}:{col_s} + {k_cols}])\n"
                ));
                return Ok(());
            }
            if src.dtype != DType::F32 {
                return Err(format!(
                    "codegen: Tcgen05Ld dtype {:?} has no lowering (the TMEM view is f32)",
                    src.dtype
                ));
            }
            if as_int(&src.row) != Some(0) {
                return Err(
                    "codegen: Tcgen05Ld row must be a static 0 (the TMEM view bases at lane 0)"
                        .to_string(),
                );
            }
            // dst is the f32 reg fragment (read as a wg view of `num` cols); src is the
            // tmem band at the operand's absolute physical `col`. The fragment is
            // filled from column 0 (scratch reused per drain group), so the view
            // slice starts at 0.
            let width = *num as usize;
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
        RegCvt {
            dst,
            src,
            // Inert on both sides today: the interpreter's cvt path coerces to the
            // dst dtype without consulting `rounding`, and `Tx.wg.cast` carries no
            // rounding argument. Listed explicitly so a future rounding-aware cvt
            // must touch both sides (and validate) at once.
            rounding: _,
        } => {
            let src_op = RegOperand::Slice(src.clone());
            if !(wg_elem_ok(scope) && reg_op_slices_full(dst, &[&src_op], ctx)?) {
                // Per-thread scalar cast (the interpreter's per-thread
                // convert): `dst_flat[i] = T.<dst dtype>(src_flat[i])`. The
                // dtype conversion IS the op here (unlike the elementwise
                // arithmetic ops, which require a shared dtype).
                let (dt, doff, w) = reg_slice_parts(dst)?;
                let (st, soff, sw) = reg_slice_parts(src)?;
                if sw != 1 && sw != w {
                    return Err(format!(
                        "codegen: RegCvt src width {sw} must be 1 or the dst width {w} \
                         (the interpreter matches the dst or broadcasts one element per thread)"
                    ));
                }
                let dflat = flat_name(dt, ctx)?;
                let sflat = flat_name(st, ctx)?;
                let d_idx = flat_elem_idx(doff, "_i", ctx)?;
                let s_idx = flat_elem_idx(soff, if sw == 1 { "0" } else { "_i" }, ctx)?;
                let body = if dt.dtype == st.dtype {
                    format!("{sflat}[{s_idx}]")
                } else {
                    format!("T.{}({sflat}[{s_idx}])", dtype_str(dt.dtype))
                };
                out.push_str(&format!("{p}for _i in range({w}):\n"));
                out.push_str(&format!("{p}    {dflat}[{d_idx}] = {body}\n"));
                return Ok(());
            }
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
        //
        // A per-thread POINT src (every dim size-1, e.g. gdn's `gcs_s[row]` scalar
        // reads) lowers to a raw element assignment on the flat view — the tile-op
        // anchor cannot express "each thread reads its own address".
        RegLoad { dst, src } => {
            // A per-thread POINT src (every dim size-1) lowers to a raw element
            // assignment on the flat view, for SMEM and GMEM alike — anything
            // else used to fall through and emit NOTHING (a silent miscompile:
            // the interpreter transfers values for every src space).
            if slice_all_size1(src)
                && matches!(src.tensor.space, MemorySpace::Smem | MemorySpace::Gmem)
            {
                let (dt, doff, dw) = reg_slice_parts(dst)?;
                if dt.dtype != src.tensor.dtype {
                    return Err(format!(
                        "codegen: RegLoad point src dtype {:?} != dst dtype {:?} — \
                         the interpreter coerces; use an explicit RegCvt",
                        src.tensor.dtype, dt.dtype
                    ));
                }
                if dw != 1 {
                    return Err(format!(
                        "codegen: RegLoad point src loads one element per thread \
                         (dst width {dw} != 1)"
                    ));
                }
                let dflat = flat_name(dt, ctx)?;
                let doff_s = emit_scalar(doff, ctx)?;
                let addr = emit_scalar_addr(src, ctx)?;
                out.push_str(&format!("{p}{dflat}[{doff_s}] = {addr}\n"));
                return Ok(());
            }
            if src.tensor.space == MemorySpace::Smem {
                let src_s = emit_smem_wg_store_tile(src, ctx)?;
                let zero = ScalarValue::Int(0);
                let dst_off = dst.offsets.first().unwrap_or(&zero);
                let width = dst.shape.first().and_then(as_int).unwrap_or(0).max(0) as usize;
                let dst_s = emit_reg_view_slice(out, &p, &dst.tensor, dst_off, width, ctx)?;
                out.push_str(&format!("{p}Tx.wg.copy({dst_s}, {src_s})\n"));
                return Ok(());
            }
            // REG src: a structural no-op (see the comment above the arm).
            if src.tensor.space == MemorySpace::Reg {
                return Ok(());
            }
            Err(format!(
                "codegen: RegLoad src space {:?} shape {:?} has no lowering (SMEM point/tile \
                 loads and GMEM point loads only)",
                src.tensor.space, src.shape
            ))
        }
        RegStore { dst, src } => {
            // REG -> REG per-thread copy (gdn's m16 A-broadcast): the
            // interpreter copies values elementwise. A `Tx.wg.copy` on the wg
            // views falls to the scalar copy fallback, whose raw thread-axis
            // BufferLoad LowerTIRxCleanup rejects — emit the flat scalar form.
            if dst.tensor.space == MemorySpace::Reg {
                if src.tensor.space != MemorySpace::Reg {
                    return Err(format!(
                        "codegen: RegStore with a REG dst needs a REG src (got {:?})",
                        src.tensor.space
                    ));
                }
                let (dt, _, _) = reg_slice_parts(dst)?;
                let (st, _, _) = reg_slice_parts(src)?;
                if st.dtype != dt.dtype {
                    return Err(format!(
                        "codegen: RegStore REG->REG src dtype {:?} != dst dtype {:?} — \
                         the interpreter coerces; use an explicit RegCvt",
                        st.dtype, dt.dtype
                    ));
                }
                let s_op = RegOperand::Slice(src.clone());
                return emit_scalar_elem(out, &p, ctx, dst, &[&s_op], false, |e| e[0].to_string());
            }
            // Per-thread point store (every dim size-1, e.g. gdn's
            // `m_s[row, col] = r1` epilogue cells): a raw element assignment on
            // the flat view.
            if slice_all_size1(dst) {
                let (st, soff, sw) = reg_slice_parts(src)?;
                if st.dtype != dst.tensor.dtype {
                    return Err(format!(
                        "codegen: RegStore point dst dtype {:?} != src dtype {:?} — \
                         the interpreter coerces; use an explicit RegCvt",
                        dst.tensor.dtype, st.dtype
                    ));
                }
                if sw != 1 {
                    return Err(format!(
                        "codegen: RegStore point dst stores one element per thread \
                         (src width {sw} != 1)"
                    ));
                }
                let sflat = flat_name(st, ctx)?;
                let soff_s = emit_scalar(soff, ctx)?;
                let addr = emit_scalar_addr(dst, ctx)?;
                out.push_str(&format!("{p}{addr} = {sflat}[{soff_s}]\n"));
                return Ok(());
            }
            // Per-thread GMEM row run (rank >= 3, leading size-1 dims, wide
            // trailing dim — gdn's final-state store
            // `state_g[seq, eh, tid, c0:c0+64] = frag32`): a raw per-thread loop.
            if gmem_row_run(dst) {
                let (st, soff, sw) = reg_slice_parts(src)?;
                if st.dtype != dst.tensor.dtype {
                    return Err(format!(
                        "codegen: RegStore GMEM row dst dtype {:?} != src dtype {:?} — \
                         the interpreter coerces; use an explicit RegCvt",
                        dst.tensor.dtype, st.dtype
                    ));
                }
                let w = as_int(dst.shape.last().unwrap()).unwrap() as usize;
                if sw != w {
                    return Err(format!(
                        "codegen: RegStore GMEM row run width {w} != src width {sw}"
                    ));
                }
                let sflat = flat_name(st, ctx)?;
                let soff_s = emit_scalar(soff, ctx)?;
                let name = ctx.tensor_name(dst.tensor.id)?;
                let n = dst.offsets.len();
                let lead = dst.offsets[..n - 1]
                    .iter()
                    .map(|o| emit_scalar(o, ctx))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ");
                let last_s = emit_scalar(&dst.offsets[n - 1], ctx)?;
                let elem = |off_s: &str| {
                    if off_s == "0" {
                        "_i".to_string()
                    } else {
                        format!("{off_s} + _i")
                    }
                };
                out.push_str(&format!("{p}for _i in range({w}):\n"));
                out.push_str(&format!(
                    "{p}    {name}[{lead}, {}] = {sflat}[{}]\n",
                    elem(&last_s),
                    elem(&soff_s)
                ));
                return Ok(());
            }
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
            allow_nondet_reduce: _,
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
            // destination region. Single-issue: one thread of the enclosing branch
            // (thread 0 of a warpgroup branch, the elected lane of a warp branch).
            let src_s = emit_smem_tile(src, ctx)?;
            let dst_s = emit_gmem_region(dst, coords, gmem_extents(gmem_shape, shape), ctx)?;
            // Both hints are IR-carried. `cache_hint="evict_first"` (canon's epilogue
            // store policy): the store band is write-once output, never re-read by
            // this kernel — dead lines must not pack L2 and evict the live operand
            // tiles / TMA tensormaps.
            let cache_hint_kw = match cache_hint {
                Some(hint) => format!(", cache_hint=\"{hint}\""),
                None => String::new(),
            };
            let prefetch_kw = if *prefetch_tensormap {
                ", prefetch_tensormap=True"
            } else {
                ""
            };
            emit_single_issue(
                out,
                &p,
                scope,
                "tma_store",
                &format!(
                    "Tx.copy_async({dst_s}, {src_s}, dispatch=\"tma_auto\"{cache_hint_kw}{prefetch_kw})"
                ),
            )?;
            Ok(())
        }
        CpAsyncBulkCommitGroup => {
            // `commit_group` batches the TMA stores issued BY THIS THREAD. Emit it
            // UNGUARDED at every scope — canon's exact shape (`if <lane0>: fence; store`
            // then an unguarded `commit_group` every thread executes). Threads with no
            // outstanding groups commit an empty group (vacuous); the wg_sync around the
            // wait_group is the real cross-thread barrier. The single-thread guarded
            // form was reverted for the fp16 1024 uniform-placement convergence: the
            // guard's ISETP/BRA pairs cost more whole-function static complexity than
            // the empty commits (see docs/perf-methodology.md §5).
            out.push_str(&format!("{p}T.ptx.cp_async.bulk.commit_group()\n"));
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
        // visible to the TMA proxy before the store. Emit it SINGLE-THREAD
        // (`if tid_in_wg == 0`) inside a warp/warpgroup branch, like canon's
        // `if (warp_id==0)&(lane_id==0): fence.proxy_async`. Canon's own comment:
        // "an all-128-thread fence was the dominant stall" — the preceding
        // warpgroup_sync already makes the reg->smem writes CTA-visible, so only
        // the single TMA-issuing thread needs the proxy fence. (The kernel orders
        // the wg_sync BEFORE this fence so the writes are visible first.)
        Fence {
            kind,
            scope: fence_scope,
        } => {
            use super::dtype::{FenceKind, FenceScope};
            match kind {
                // `fence.mbarrier_init` — the prologue init-epoch fence (all threads).
                FenceKind::MbarrierInit => {
                    out.push_str(&format!("{p}T.ptx.fence.mbarrier_init()\n"));
                }
                // Scope-aware, like cta_sync: at FUNCTION scope (the prologue, where all CTA
                // threads converge) emit the proxy fence for ALL threads; inside a single-
                // warp/warpgroup branch (the epilogue) emit it single-thread (`if tid_in_wg==0`)
                // — and bare when the branch is already single-threaded.
                //
                // The IR `scope` field selects the PTX address-space qualifier 1:1
                // (`T.ptx.fence.proxy_async(space)` accepts ""/"global"/"shared::cta"/
                // "shared::cluster"). Cta/Cluster are the two shared-memory levels — the
                // checker's proxy-fence visibility test reads exactly this distinction.
                // Gpu lowers to the UNQUALIFIED form, which orders every address space:
                // that is what the checker models for it (`FenceScope::Gpu => covers any
                // access`), so a weaker `.global` would run a narrower fence than sim
                // validates.
                FenceKind::AsyncProxy => {
                    let space = match fence_scope {
                        FenceScope::Cta => "shared::cta",
                        FenceScope::Cluster => "shared::cluster",
                        FenceScope::Gpu => "",
                    };
                    // PER-THREAD: a proxy fence orders only the EXECUTING lane's
                    // prior writes — guarding it to one lane would leave every
                    // other lane's writes unordered (zero-inference; hardware
                    // fence is a per-thread instruction, so this is exact).
                    out.push_str(&format!("{p}T.ptx.fence.proxy_async(\"{space}\")\n"));
                }
                // Memory/View fences are sim-only ordering markers — the interpreter
                // folds them into trace `Generic` events and there is no TIRx
                // lowering. Fail closed instead of silently emitting nothing.
                FenceKind::Memory | FenceKind::View => {
                    return Err(format!(
                        "codegen: Fence {kind:?} is sim-only (no TIRx lowering)"
                    ));
                }
            }
            Ok(())
        }
        CtaSync => {
            // Suppress CTA-wide cta_sync inside a single-warp/wg branch (not
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
        ClusterBarrierArrive { sem } => {
            // Split cluster barrier, collective non-blocking arrival (all threads).
            // `sem` is emitted 1:1 — canon's `.relaxed` carries no release ordering
            // (PTX §9.7.14.3); the checker only propagates memory HB for `.release`.
            out.push_str(&format!(
                "{p}T.ptx.barrier.cluster.arrive(sem=\"{}\", aligned=True)\n",
                sem.as_str()
            ));
            Ok(())
        }
        ClusterBarrierWait => {
            // Split cluster barrier, per-branch wait. `barrier.cluster.wait` is
            // WARP-COLLECTIVE: ALL threads of the branch's warp(group) must execute
            // it. If only the elected lane waits, the overlap path DEADLOCKS on
            // hardware (verified on 1024/2048) while the protocol checker still
            // passes (it models the wait as collective regardless of the codegen
            // elect context). The elect-form `If` arm hoists a leading
            // ClusterBarrierWait out of the elect to warp scope; reaching here
            // under `Scope::Elected` means that hoist was bypassed — fail LOUDLY
            // instead of emitting a silently-hanging kernel.
            if matches!(scope.scope, Scope::Elected) {
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
        // `bar.warp.sync` — full warp execution + memory-order barrier (the
        // gate warp's SMEM cumsum and the WY-inverse merges read lanes' SMEM
        // writes from the previous phase; dropping it lets lanes read stale
        // cells — the cumsum/gateway corruption. The checker validates the
        // site is warp-converged; emit it bare everywhere it appears.
        WarpSync => {
            out.push_str(&format!("{p}T.cuda.warp_sync()\n"));
            Ok(())
        }
        WgSync { barrier_id } => {
            out.push_str(&format!("{p}T.cuda.warpgroup_sync({barrier_id})\n"));
            Ok(())
        }
        // Per-warpgroup register budget (canon's `setmaxnreg(False, 56)` for the
        // producer warpgroup, `(True, 224)` for the consumer): rebalances registers
        // so the compute-heavy consumer gets more and the producer fewer, raising
        // occupancy. Warpgroup-aligned — validate requires the enclosing branch to
        // statically cover whole warpgroups, so the guard is the enclosing
        // warpgroup `If`; emit the directive bare. inc when the budget rises above
        // the 128-reg default, else dec.
        SetMaxNReg { nreg } => {
            let inc = if *nreg > 128 { "True" } else { "False" };
            out.push_str(&format!("{p}T.ptx.setmaxnreg({inc}, {nreg})\n"));
            Ok(())
        }
        // Cross-warpgroup named barrier — `bar.sync barrier_id, num_warps*32`. Unlike
        // WgSync (per-warpgroup), threads from different branches rendezvous on the shared
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
        // `mma.sync.aligned.m16n8k{8,16}.row.col.f32.{ab}.{ab}.f32` — warp-level
        // SM80 HMMA via the legacy WMMA-fragment intrinsic: one flat local
        // buffer + base offset per operand, the accumulator reused as both C
        // and D (the intrinsic's single accumulator slot), exactly the
        // interpreter's D = A·Bᵀ + C over the standard warp fragment layout
        // (the layout LdMatrix produces — A/B ride the `{name}_flat_ab`
        // bf16/f16 reinterpret of their packed u32 words).
        WarpMma {
            d,
            a,
            b,
            c,
            m,
            n,
            k,
            ab_dtype,
        } => {
            if !matches!((*m, *n, *k), (16, 8, 8) | (16, 8, 16)) {
                return Err(format!(
                    "codegen: WarpMma m{m}n{n}k{k} has no lowering (only m16n8k8/m16n8k16)"
                ));
            }
            if !matches!(ab_dtype, DType::F16 | DType::Bf16) {
                return Err(format!(
                    "codegen: WarpMma ab_dtype {ab_dtype:?} has no lowering (bf16/f16 only)"
                ));
            }
            // d == c: the legacy intrinsic reuses the one accumulator buffer as
            // both input and output — distinct C/D slices would need a second
            // fragment (no lowering; gdn always accumulates in place).
            if d != c {
                return Err(
                    "codegen: WarpMma d and c must be the same fragment (the mma.sync \
                     accumulator is read-modify-write)"
                        .to_string(),
                );
            }
            let (dt, doff, dw) = reg_slice_parts(d)?;
            let (at, aoff, aw) = reg_slice_parts(a)?;
            let (bt, boff, bw) = reg_slice_parts(b)?;
            if dt.dtype != DType::F32 {
                return Err(format!(
                    "codegen: WarpMma C/D dtype {:?} must be f32 (the PTX accumulator)",
                    dt.dtype
                ));
            }
            if !matches!(at.dtype, DType::U32 | DType::I32)
                || !matches!(bt.dtype, DType::U32 | DType::I32)
            {
                return Err(
                    "codegen: WarpMma A/B fragments must be u32/i32 packed words \
                     (ldmatrix output; the bf16/f16 element form has no lowering)"
                        .to_string(),
                );
            }
            let len_a = (*m * *k / 64) as usize;
            let len_b = (*n * *k / 64) as usize;
            let len_cd = (*m * *n / 32) as usize;
            for (off, w, want, label) in [
                (doff, dw, len_cd, "C/D"),
                (aoff, aw, len_a, "A"),
                (boff, bw, len_b, "B"),
            ] {
                if as_int(off) != Some(0) || w != want {
                    return Err(format!(
                        "codegen: WarpMma {label} fragment must span exactly {want} b32 \
                         registers from offset 0 (got off={off:?}, width {w})"
                    ));
                }
            }
            let a_name = ctx.tensor_name(at.id)?;
            let b_name = ctx.tensor_name(bt.id)?;
            let d_name = ctx.tensor_name(dt.id)?;
            let ab_s = dtype_str(*ab_dtype);
            out.push_str(&format!(
                "{p}T.ptx.mma.legacy(\"m{m}n{n}k{k}\", \"row\", \"col\", \"{ab_s}\", \"{ab_s}\", \"float32\", {a_name}_flat_ab.data, 0, {b_name}_flat_ab.data, 0, {d_name}_flat.data, 0, False, dtype=\"float32\")\n"
            ));
            Ok(())
        }
        GmemAtomicAdd { .. } => Err("codegen: GmemAtomicAdd not yet supported".to_string()),
        GmemWaitEq { .. } => Err("codegen: GmemWaitEq not yet supported".to_string()),
        CpAsyncBulkS2Cluster { .. } => {
            Err("codegen: CpAsyncBulkS2Cluster not yet supported".to_string())
        }

        // NVFP4 epilogue alpha rescale: Tx.wg.mul(frag, frag, alpha). lhs is a reg slice,
        // rhs the alpha literal (or vice versa). Under a narrowed branch (the
        // gdn predicated epilogues) the wg dispatch rejects the intra — fall
        // to the per-thread scalar form.
        RegMul { dst, lhs, rhs } => {
            if !(wg_elem_ok(scope) && reg_op_slices_full(dst, &[lhs, rhs], ctx)?) {
                let (t, _, _) = reg_slice_parts(dst)?;
                if !matches!(t.dtype, DType::F16 | DType::Bf16 | DType::F32) {
                    return Err(format!(
                        "codegen: RegMul dst dtype {:?} has no lowering (float dsts only)",
                        t.dtype
                    ));
                }
                return emit_scalar_elem(out, &p, ctx, dst, &[lhs, rhs], true, |e| {
                    format!("{} * {}", e[0], e[1])
                });
            }
            let zero = ScalarValue::Int(0);
            let reg_op = |op: &RegOperand, out: &mut Emitter| -> Result<String, String> {
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
        // ---- Fail closed, per variant (no catch-all): every Stmt variant is
        // either lowered above or rejected HERE with an explicit Err, so adding
        // a variant without a lowering is a compile error (match exhaustiveness)
        // and never a silent different-semantics emission. The flash-attention /
        // flash-bwd datapath set has no GEMM-codegen lowering yet.
        SchedNext { .. } => Err("codegen: SchedNext not yet supported".to_string()),
        // tcgen05.st — REG fragment -> TMEM, the mirror of Tcgen05Ld: same
        // validations (row/shape/dtype), same `.32x32b` wg-view path. The
        // `.16x*b` atoms would need an atom-layout src fragment — no gdn use,
        // fail closed. For fp16/bf16 dsts the TMEM band is DENSE-PACKED (two
        // elements per 32-bit cell): the IR slice counts b32 registers (num),
        // the interpreter reads 2*num elements, and the emission goes through
        // the `tmem_f16`/`tmem_bf16` packed views at twice the element window.
        Tcgen05St {
            dst,
            src,
            shape,
            num,
        } => {
            if *shape != LdStShape::B32x32 {
                return Err(format!(
                    "codegen: Tcgen05St shape={} has no lowering (only 32x32b; \
                     the 16x*b atoms need an atom-layout src fragment)",
                    shape.as_str()
                ));
            }
            if as_int(&dst.row) != Some(0) {
                return Err(
                    "codegen: Tcgen05St row must be a static 0 (the TMEM view bases at lane 0)"
                        .to_string(),
                );
            }
            let (t, off, w) = reg_slice_parts(src)?;
            if t.dtype != dst.dtype {
                return Err(format!(
                    "codegen: Tcgen05St REG dtype {:?} != TMEM dtype {:?} \
                     (the interpreter requires them equal)",
                    t.dtype, dst.dtype
                ));
            }
            if w != *num as usize {
                return Err(format!(
                    "codegen: Tcgen05St src width {w} != num {num} (the register \
                     slice must span exactly the atom's b32 registers)"
                ));
            }
            match dst.dtype {
                DType::F32 => {
                    let col_s = emit_scalar(&dst.col, ctx)?;
                    let frag_s = emit_reg_view_slice(out, &p, t, off, w, ctx)?;
                    out.push_str(&format!(
                        "{p}Tx.wg.copy_async(tmem[:, {col_s}:{col_s} + {w}], {frag_s})\n"
                    ));
                }
                DType::F16 | DType::Bf16 => {
                    let view = if dst.dtype == DType::F16 {
                        "tmem_f16"
                    } else {
                        "tmem_bf16"
                    };
                    let col_s = emit_scalar(&dst.col, ctx)?;
                    let frag_s = emit_reg_view_slice(out, &p, t, off, 2 * w, ctx)?;
                    out.push_str(&format!(
                        "{p}Tx.wg.copy_async({view}[:, ({col_s}) * 2:({col_s}) * 2 + {}], {frag_s})\n",
                        2 * w
                    ));
                }
                other => {
                    return Err(format!(
                        "codegen: Tcgen05St dtype {other:?} has no lowering \
                         (f32 or packed f16/bf16 TMEM cells only)"
                    ))
                }
            }
            Ok(())
        }
        Tcgen05WaitSt => {
            out.push_str(&format!("{p}T.ptx.tcgen05.wait.st()\n"));
            Ok(())
        }
        // `ldmatrix.sync.aligned.m8n8.xN.b16` — SMEM row-addresses -> packed
        // REG words. The IR src slice is the PER-THREAD row start (8 b16
        // elements); lanes 0..7N contribute addresses (PTX), exactly the
        // interpreter's row_address_lane model. The dst is N u32 words in the
        // flat view (ptr_to per register — the documented dst-handle form).
        LdMatrix {
            dst,
            src,
            shape,
            num,
            trans,
            dtype,
        } => {
            if *shape != MatrixShape::M8N8 || *dtype != MatrixDType::B16 {
                return Err(
                    "codegen: LdMatrix supports only m8n8.x{1,2,4}.b16 (the interpreter's \
                     matrix model)"
                        .to_string(),
                );
            }
            let (dt, doff, dw) = reg_slice_parts(dst)?;
            if !matches!(dt.dtype, DType::U32 | DType::I32) {
                return Err(format!(
                    "codegen: LdMatrix dst dtype {:?} has no lowering (u32/i32 packed \
                     words only; a b16 fragment dst fails in the interpreter too)",
                    dt.dtype
                ));
            }
            if dw != *num as usize {
                return Err(format!(
                    "codegen: LdMatrix dst width {dw} != num {num} (the dst spans num b32 \
                     registers)"
                ));
            }
            check_matrix_smem_row(src, "LdMatrix")?;
            let dflat = flat_name(dt, ctx)?;
            let doff_s = emit_scalar(doff, ctx)?;
            let handles = (0..*num as usize)
                .map(|i| format!("{dflat}.ptr_to([{}])", flat_add(&doff_s, i)))
                .collect::<Vec<_>>()
                .join(", ");
            let row_s = emit_scalar(&src.offsets[0], ctx)?;
            let col_s = emit_scalar(&src.offsets[1], ctx)?;
            let src_name = ctx.tensor_name(src.tensor.id)?;
            out.push_str(&format!(
                "{p}T.ptx.ldmatrix({}, {num}, \".b16\", {src_name}.ptr_to([{row_s}, {col_s}]), {handles})\n",
                py_bool(*trans),
            ));
            Ok(())
        }
        // `stmatrix.sync.aligned.m8n8.xN.b16` — REG words -> SMEM. Mirror of
        // LdMatrix; a bf16/f16 src fragment reads through the `_flat_u32`
        // reinterpret (consecutive b16 pairs ARE the words, matching the
        // interpreter's pack).
        StMatrix {
            dst,
            src,
            shape,
            num,
            trans,
            dtype,
        } => {
            if *shape != MatrixShape::M8N8 || *dtype != MatrixDType::B16 {
                return Err(
                    "codegen: StMatrix supports only m8n8.x{1,2,4}.b16 (the interpreter's \
                     matrix model)"
                        .to_string(),
                );
            }
            check_matrix_smem_row(dst, "StMatrix")?;
            let (st, soff, sw) = reg_slice_parts(src)?;
            let soff_s = emit_scalar(soff, ctx)?;
            let name = ctx.tensor_name(st.id)?;
            let handles: String = match st.dtype {
                DType::U32 | DType::I32 => {
                    if sw != *num as usize {
                        return Err(format!(
                            "codegen: StMatrix src width {sw} != num {num} (the src spans \
                             num b32 registers)"
                        ));
                    }
                    (0..*num as usize)
                        .map(|i| format!("{name}_flat.ptr_to([{}])", flat_add(&soff_s, i)))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
                DType::F16 | DType::Bf16 => {
                    if sw != 2 * *num as usize {
                        return Err(format!(
                            "codegen: StMatrix b16 src width {sw} != 2*num {} (the src spans \
                             num packed b16x2 registers)",
                            2 * *num as usize
                        ));
                    }
                    let Some(word_off) = as_int(soff) else {
                        return Err(
                            "codegen: StMatrix b16 src offset must be static (the u32 word \
                             index is offset/2)"
                                .to_string(),
                        );
                    };
                    if word_off % 2 != 0 {
                        return Err(format!(
                            "codegen: StMatrix b16 src offset {word_off} is odd (a packed \
                             b16x2 register starts at an even element)"
                        ));
                    }
                    (0..*num as usize)
                        .map(|i| format!("{name}_flat_u32.ptr_to([{}])", word_off / 2 + i as i64))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
                other => {
                    return Err(format!(
                        "codegen: StMatrix src dtype {other:?} has no lowering (u32/i32 \
                         words or a b16 fragment)"
                    ))
                }
            };
            let row_s = emit_scalar(&dst.offsets[0], ctx)?;
            let col_s = emit_scalar(&dst.offsets[1], ctx)?;
            let dst_name = ctx.tensor_name(dst.tensor.id)?;
            out.push_str(&format!(
                "{p}T.ptx.stmatrix({}, {num}, \".b16\", {dst_name}.ptr_to([{row_s}, {col_s}]), {handles})\n",
                py_bool(*trans),
            ));
            Ok(())
        }
        // Per-thread elementwise fill. Literal: dst-typed scalar (the
        // interpreter's literal_array). Slice: elementwise copy (the
        // interpreter broadcasts a one-element-per-thread src — validate the
        // two legal widths). `Tx.wg.*` when the scope is provably
        // warpgroup-full, else the per-thread scalar form.
        RegFill { dst, value } => {
            if wg_elem_ok(scope) && reg_op_slices_full(dst, &[value], ctx)? {
                let (t, off, w) = reg_slice_parts(dst)?;
                let dst_s = emit_reg_view_slice(out, &p, t, off, w, ctx)?;
                match value {
                    RegOperand::Literal(l) => {
                        let v = typed_scalar(t.dtype, *l)?;
                        out.push_str(&format!("{p}Tx.wg.fill({dst_s}, {v})\n"));
                    }
                    RegOperand::Slice(src) => {
                        let (st, soff, sw) = reg_slice_parts(src)?;
                        if st.dtype != t.dtype {
                            return Err(format!(
                                "codegen: RegFill src dtype {:?} != dst dtype {:?} — \
                                 the interpreter coerces per-op; use an explicit RegCvt",
                                st.dtype, t.dtype
                            ));
                        }
                        if sw != 1 && sw != w {
                            return Err(format!(
                                "codegen: RegFill src width {sw} must be 1 or the dst width {w} \
                                 (the interpreter matches the dst or broadcasts one element per thread)"
                            ));
                        }
                        let src_s = emit_reg_view_slice(out, &p, st, soff, sw, ctx)?;
                        out.push_str(&format!("{p}Tx.wg.copy({dst_s}, {src_s})\n"));
                    }
                }
                Ok(())
            } else {
                emit_scalar_elem(out, &p, ctx, dst, &[value], false, |e| e[0].to_string())
            }
        }
        RegAdd {
            dst,
            lhs,
            rhs,
            rounding,
        } => emit_reg_binary(out, &p, dst, lhs, rhs, *rounding, "add", ctx, scope),
        RegSub {
            dst,
            lhs,
            rhs,
            rounding,
        } => emit_reg_binary(out, &p, dst, lhs, rhs, *rounding, "sub", ctx, scope),
        // dst = a * b + c elementwise (the interpreter's unfused f32 mul+add;
        // the TIRx fma may fuse — a 1-ulp difference the GPU gates tolerate).
        RegFma { dst, a, b, c } => {
            let (t, off, w) = reg_slice_parts(dst)?;
            if !matches!(t.dtype, DType::F16 | DType::Bf16 | DType::F32) {
                return Err(format!(
                    "codegen: RegFma dst dtype {:?} has no lowering (float dsts only)",
                    t.dtype
                ));
            }
            if wg_elem_ok(scope) && reg_op_slices_full(dst, &[a, b, c], ctx)? {
                let dst_s = emit_reg_view_slice(out, &p, t, off, w, ctx)?;
                let a_s = emit_wg_reg_operand(a, t.dtype, out, &p, ctx)?;
                let b_s = emit_wg_reg_operand(b, t.dtype, out, &p, ctx)?;
                let c_s = emit_wg_reg_operand(c, t.dtype, out, &p, ctx)?;
                out.push_str(&format!("{p}Tx.wg.fma({dst_s}, {a_s}, {b_s}, {c_s})\n"));
                Ok(())
            } else {
                emit_scalar_elem(out, &p, ctx, dst, &[a, b, c], true, |e| {
                    format!("{} * {} + {}", e[0], e[1], e[2])
                })
            }
        }
        // Elementwise unary over an f32 fragment. exp2/rcp/neg map to the TIRx
        // unary tile ops at warpgroup-full scope (rcp lowers to a true `1.0 /
        // x` division, matching the interpreter; neg is an exact sign flip);
        // log2 has NO tile op and always takes the per-thread scalar form.
        // Anything under a narrowed branch takes the scalar form. A literal
        // src folds at codegen time: the interpreter applies the same rust f32
        // math, so the fold is bit-identical.
        RegUnary { dst, src, op } => {
            let (t, off, w) = reg_slice_parts(dst)?;
            if t.dtype != DType::F32 {
                return Err(format!(
                    "codegen: RegUnary {} dst dtype {:?} has no lowering — the \
                     interpreter computes unary ops in f32 and rounds; the TIRx \
                     16-bit elementwise path computes in 16 bits (f32 dst only)",
                    op.as_str(),
                    t.dtype
                ));
            }
            let folded = match (src, op) {
                (RegOperand::Literal(l), RegUnaryOp::Exp2) => Some(l.as_f32().exp2()),
                (RegOperand::Literal(l), RegUnaryOp::Log2) => Some(l.as_f32().log2()),
                (RegOperand::Literal(l), RegUnaryOp::Rcp) => Some(1.0 / l.as_f32()),
                (RegOperand::Literal(l), RegUnaryOp::Neg) => Some(-l.as_f32()),
                (RegOperand::Slice(_), _) => None,
            };
            if let Some(v) = folded {
                if wg_elem_ok(scope) && slice_is_full(t, off, w, ctx) {
                    let dst_s = emit_reg_view_slice(out, &p, t, off, w, ctx)?;
                    out.push_str(&format!("{p}Tx.wg.fill({dst_s}, T.float32({v}))\n"));
                    return Ok(());
                }
                return emit_scalar_elem(out, &p, ctx, dst, &[], false, |_| {
                    format!("T.float32({v})")
                });
            }
            let RegOperand::Slice(src) = src else {
                return Err("codegen: RegUnary src must be a slice or literal".to_string());
            };
            let (st, soff, sw) = reg_slice_parts(src)?;
            if st.dtype != DType::F32 {
                return Err(format!(
                    "codegen: RegUnary {} src dtype {:?} has no lowering (f32 only)",
                    op.as_str(),
                    st.dtype
                ));
            }
            if sw != 1 && sw != w {
                return Err(format!(
                    "codegen: RegUnary src width {sw} must be 1 or the dst width {w} \
                     (the interpreter matches the dst or broadcasts one element per thread)"
                ));
            }
            let src_op = RegOperand::Slice(src.clone());
            if wg_elem_ok(scope)
                && *op != RegUnaryOp::Log2
                && reg_op_slices_full(dst, &[&src_op], ctx)?
            {
                let dst_s = emit_reg_view_slice(out, &p, t, off, w, ctx)?;
                let src_s = emit_reg_view_slice(out, &p, st, soff, sw, ctx)?;
                match op {
                    RegUnaryOp::Exp2 => {
                        out.push_str(&format!("{p}Tx.wg.exp2({dst_s}, {src_s})\n"));
                    }
                    RegUnaryOp::Rcp => {
                        out.push_str(&format!("{p}Tx.wg.reciprocal({dst_s}, {src_s})\n"));
                    }
                    RegUnaryOp::Neg => {
                        out.push_str(&format!("{p}Tx.wg.mul({dst_s}, {src_s}, T.float32(-1))\n"));
                    }
                    RegUnaryOp::Log2 => unreachable!("log2 takes the scalar form"),
                }
                Ok(())
            } else {
                emit_scalar_elem(out, &p, ctx, dst, &[&src_op], false, |e| match op {
                    RegUnaryOp::Exp2 => format!("T.exp2({})", e[0]),
                    RegUnaryOp::Log2 => format!("T.log2({})", e[0]),
                    RegUnaryOp::Rcp => format!("1.0 / ({})", e[0]),
                    RegUnaryOp::Neg => format!("-({})", e[0]),
                })
            }
        }
        RegMax { .. } => Err("codegen: RegMax not yet supported".to_string()),
        RegMin { .. } => Err("codegen: RegMin not yet supported".to_string()),
        RegBitwise { .. } => Err("codegen: RegBitwise not yet supported".to_string()),
        RegReduce { .. } => Err("codegen: RegReduce not yet supported".to_string()),
        RegCondRescale { .. } => Err("codegen: RegCondRescale not yet supported".to_string()),
        RegSoftmaxRescale { .. } => Err("codegen: RegSoftmaxRescale not yet supported".to_string()),
        RegCausalMask { .. } => Err("codegen: RegCausalMask not yet supported".to_string()),
        RegCombineIntFracEx2 { .. } => {
            Err("codegen: RegCombineIntFracEx2 not yet supported".to_string())
        }
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
/// term is stripped and the row becomes a 128-wide range (WG_THREADS rows = one
/// row per warpgroup thread, a hardware constant).
fn emit_gmem_row_store(dst: &TensorSlice, ctx: &Ctx) -> Result<String, String> {
    let name = ctx.tensor_name(dst.tensor.id)?;
    if dst.offsets.len() != 2 {
        return Err("codegen: reg_store dst must be 2D".to_string());
    }
    let clo = emit_scalar(&dst.offsets[1], ctx)?;
    let chi = add_bound(&dst.offsets[1], &dst.shape[1], ctx)?;
    let wg_rows = ScalarValue::Int(WG_THREADS as i64);
    if let Some(base) = strip_tid_in_wg(&dst.offsets[0]) {
        let lo = emit_scalar(&base, ctx)?;
        let hi = add_bound(&base, &wg_rows, ctx)?;
        Ok(format!("{name}[{lo}:{hi}, {clo}:{chi}]"))
    } else {
        // No lane term (already a tile base): emit a 128-row band from the offset.
        let lo = emit_scalar(&dst.offsets[0], ctx)?;
        let hi = add_bound(&dst.offsets[0], &wg_rows, ctx)?;
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
        kernel_n(body, 4)
    }

    fn kernel_n(body: Vec<Stmt>, num_warps: u32) -> Kernel {
        Kernel {
            name: "t".to_string(),
            args: vec![gmem_arg(0)],
            body,
            num_warps,
            smem_size_bytes: 0,
            launch_shape: vec![2],
            cluster_shape: vec![2],
            smem_pool: false,
        }
    }

    /// `if warp_id == w: body` — the warp-model's warp-dispatch sugar.
    fn warp_if(w: i64, body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond: ScalarValue::expr(
                ScalarOp::Eq,
                vec![
                    ScalarValue::Scope(ScopeValueKind::WarpId),
                    ScalarValue::Int(w),
                ],
            ),
            then_body: body,
        }
    }

    /// `if wg_id == g: body`.
    fn wg_if(g: i64, body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond: ScalarValue::expr(
                ScalarOp::Eq,
                vec![
                    ScalarValue::Scope(ScopeValueKind::WarpgroupId),
                    ScalarValue::Int(g),
                ],
            ),
            then_body: body,
        }
    }

    /// `if lane_id == 0: body` — the `if_elected` sugar.
    fn elected_if(body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond: ScalarValue::expr(
                ScalarOp::Eq,
                vec![
                    ScalarValue::Scope(ScopeValueKind::LaneId),
                    ScalarValue::Int(0),
                ],
            ),
            then_body: body,
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

    /// `ForLoop { unroll: false }` pins the rolled `T.serial(N, unroll=False)`
    /// (canon's `#pragma unroll 1` form); the default `true` is the plain
    /// serial loop.
    #[test]
    fn serial_loop_unroll_flag_forms() {
        let var = Var {
            id: VarId(0),
            binding: VarBinding::Loop,
            dtype: ScalarDType::I32,
        };
        let plain = Stmt::ForLoop {
            var,
            start: ScalarValue::Int(0),
            stop: ScalarValue::Int(4),
            step: ScalarValue::Int(1),
            body: vec![],
            unroll: true,
        };
        let src = kernel_to_tirx_source(&kernel(vec![plain])).unwrap();
        assert!(src.contains("for v0 in T.serial(4):"), "{src}");

        let rolled = Stmt::ForLoop {
            var,
            start: ScalarValue::Int(0),
            stop: ScalarValue::Int(4),
            step: ScalarValue::Int(1),
            body: vec![],
            unroll: false,
        };
        let src = kernel_to_tirx_source(&kernel(vec![rolled])).unwrap();
        assert!(
            src.contains("for v0 in T.serial(4, unroll=False):"),
            "{src}"
        );
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

    /// The cluster-scope ids derive from the kernel's `cluster_shape` (a 1-D
    /// (n,) is the x extent with y=1); a rank>2 cluster has no emission form
    /// and fails closed. A cluster-1 kernel gets constants: the (1, 1) axis
    /// bind collapses and the scope resolver loses `clusterCtaIdx.x`.
    #[test]
    fn cluster_ids_derive_from_cluster_shape() {
        let mut k = kernel(vec![]);
        k.cluster_shape = vec![1];
        let src = kernel_to_tirx_source(&k).unwrap();
        assert!(src.contains("cbx, cby = 0, 0"), "{src}");
        assert!(!src.contains("cta_id_in_cluster"), "{src}");

        k.cluster_shape = vec![2, 2];
        let src = kernel_to_tirx_source(&k).unwrap();
        assert!(
            src.contains("cbx, cby = T.cta_id_in_cluster([2, 2], preferred=[2, 2])"),
            "{src}"
        );

        k.cluster_shape = vec![2, 2, 2];
        let err = kernel_to_tirx_source(&k).unwrap_err();
        assert!(err.contains("no cta_id_in_cluster emission"), "{err}");
    }

    fn scalar_var(id: u32) -> Var {
        Var {
            id: VarId(id),
            binding: VarBinding::Scalar,
            dtype: ScalarDType::I32,
        }
    }

    /// A scalar whose every definition is provably non-negative (0-init,
    /// `s = s + 1` updates) IS strength-reduced on a non-pow2 divisor; a
    /// mailbox-loaded scalar (possible -1 sentinel) is NOT — it keeps the
    /// floormod form. The pow2 bit-op rewrite is a two's-complement identity
    /// and applies to both.
    #[test]
    fn nonneg_analysis_gates_only_the_trunc_rewrite() {
        let good = scalar_var(0);
        let bad = scalar_var(1);
        let mailbox = Arc::new(Tensor {
            id: 9,
            space: MemorySpace::Smem,
            dtype: DType::I32,
            shape: vec![1],
            layout: None,
            byte_offset: Some(0),
            reg_frag: None,
        });
        let load = TensorSlice {
            tensor: mailbox.clone(),
            offsets: vec![ScalarValue::Int(0)],
            shape: vec![ScalarValue::Int(1)],
        };
        let body = vec![
            Stmt::TensorDef {
                tensor: mailbox.clone(),
            },
            Stmt::ScalarDef {
                var: good,
                initial: ScalarInitial::Value(ScalarValue::Int(0)),
            },
            Stmt::ScalarStore {
                var: good,
                value: ScalarValue::expr(
                    ScalarOp::Add,
                    vec![ScalarValue::Var(good), ScalarValue::Int(1)],
                ),
            },
            Stmt::ScalarDef {
                var: bad,
                initial: ScalarInitial::Tensor(load),
            },
            // `good % 5` -> truncmod (proven non-negative); `bad % 5` -> floormod;
            // `bad % 4` -> `& 3` (pow2 identity, no sign gate).
            Stmt::ScalarStore {
                var: good,
                value: ScalarValue::expr(
                    ScalarOp::Mod,
                    vec![ScalarValue::Var(good), ScalarValue::Int(5)],
                ),
            },
            Stmt::ScalarStore {
                var: bad,
                value: ScalarValue::expr(
                    ScalarOp::Mod,
                    vec![ScalarValue::Var(bad), ScalarValue::Int(5)],
                ),
            },
            Stmt::ScalarStore {
                var: bad,
                value: ScalarValue::expr(
                    ScalarOp::Mod,
                    vec![ScalarValue::Var(bad), ScalarValue::Int(4)],
                ),
            },
        ];
        let src = kernel_to_tirx_source(&kernel(body)).unwrap();
        assert!(src.contains("s0 = T.truncmod(s0, 5)"), "{src}");
        assert!(src.contains("s1 = s1 % 5"), "{src}");
        assert!(src.contains("s1 = (s1) & 3"), "{src}");
    }

    /// A `ScalarLet` emits the immutable SSA form `name: T.let[T.int32] = expr`
    /// (canon's tile-decode chain), NOT the mutable `name: T.int32 = …`
    /// local-scalar cell a plain annotation lowers to.
    #[test]
    fn scalar_let_emits_the_t_let_form() {
        let v = scalar_var(0);
        let k = kernel(vec![Stmt::ScalarLet {
            var: v,
            value: ScalarValue::expr(
                ScalarOp::Add,
                vec![ScalarValue::Int(1), ScalarValue::Int(2)],
            ),
        }]);
        let src = kernel_to_tirx_source(&k).unwrap();
        assert!(src.contains("s0: T.let[T.int32] = 1 + 2"), "{src}");
        // And the var-defs bookkeeping names it like any scalar def.
        assert!(!src.contains("s0 ="), "{src}");
    }

    /// Adjacent top-level warp/warpgroup-equality `If`s re-nest into the if/else
    /// dispatch tree (canon's role dispatch), partitioned by warpgroup: warp
    /// branches chain inside their warpgroup branch, groups chain in
    /// first-occurrence order. Non-adjacent Ifs stay flat.
    #[test]
    fn top_level_equality_ifs_chain_by_warpgroup() {
        // 8 warps. [warp 5, warp 4, wg 0]: warps 4/5 are group 1, wg 0 is
        // group 0. First-occurrence order: group 1, then group 0.
        let body = vec![
            warp_if(5, vec![Stmt::WarpSync]),
            warp_if(4, vec![Stmt::WarpSync]),
            wg_if(0, vec![Stmt::WarpSync]),
        ];
        let src = kernel_to_tirx_source(&kernel_n(body, 8)).unwrap();
        let expected = "\
    if wg_id == 1:
        if warp_id == 5:
            T.cuda.warp_sync()
        else:
            if warp_id == 4:
                T.cuda.warp_sync()
    else:
        if wg_id == 0:
            T.cuda.warp_sync()
";
        assert!(src.contains(expected), "{src}");

        // A warpgroup-equality `If` in the same group is the group PREFIX: its
        // body runs first, the warp chain nests inside the same branch.
        let body = vec![
            wg_if(1, vec![Stmt::WgSync { barrier_id: 3 }]),
            warp_if(5, vec![Stmt::WgSync { barrier_id: 4 }]),
        ];
        let src = kernel_to_tirx_source(&kernel_n(body, 8)).unwrap();
        let expected = "\
    if wg_id == 1:
        T.cuda.warpgroup_sync(3)
        if warp_id == 5:
            T.cuda.warpgroup_sync(4)
";
        assert!(src.contains(expected), "{src}");
        // No else arm — every warpgroup was covered by the one group.
        assert!(!src.contains("else:"), "{src}");
    }

    /// A lone equality `If` (or a run broken by a non-equality stmt) stays flat.
    #[test]
    fn broken_if_runs_stay_flat() {
        let body = vec![
            warp_if(0, vec![Stmt::WarpSync]),
            Stmt::CtaSync,
            warp_if(1, vec![Stmt::WarpSync]),
        ];
        let src = kernel_to_tirx_source(&kernel(body)).unwrap();
        assert!(src.contains("if warp_id == 0:"), "{src}");
        assert!(src.contains("if warp_id == 1:"), "{src}");
        assert!(!src.contains("else:"), "{src}");
    }

    /// A run whose group mixes a warp-level `If` BEFORE its warpgroup-level
    /// `If` must NOT chain — chaining would reorder the warpgroup body ahead
    /// of the warp body (observed: TMA issue moved after the mbarrier wait →
    /// deadlock). The run emits flat, preserving source order.
    #[test]
    fn warp_before_warpgroup_run_stays_flat() {
        let body = vec![
            warp_if(1, vec![Stmt::WarpSync]),
            wg_if(0, vec![Stmt::WgSync { barrier_id: 3 }]),
        ];
        let src = kernel_to_tirx_source(&kernel_n(body, 8)).unwrap();
        let warp_pos = src.find("if warp_id == 1:").unwrap();
        let wg_pos = src.find("if wg_id == 0:").unwrap();
        assert!(warp_pos < wg_pos, "{src}");
        assert!(!src.contains("else:"), "{src}");
    }

    /// The `if_elected` sugar emits `if T.ptx.elect_sync():` inside a warp
    /// branch, narrowed to `if T.cuda.thread_rank() == 0:` for the warp-0
    /// prologue elect (thread set exactly {(0, 0)} — canon's prologue form).
    #[test]
    fn elected_if_emits_the_hardware_forms() {
        use super::super::dtype::MBarKind;
        use super::super::mbar::{MBar, MBarRef};
        let mbar = Arc::new(MBar {
            id: 3,
            kind: MBarKind::Thread,
            stages: 1,
            arrive_count: None,
            leader_routed: false,
        });
        let init = || Stmt::MBarrierInit {
            mbar: MBarRef {
                mbar: mbar.clone(),
                remote_coord: None,
            },
            count: 1,
            stage: None,
        };
        // Prologue: warp-0 branch + elected init -> thread_rank guard.
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            warp_if(0, vec![elected_if(vec![init()])]),
        ]))
        .unwrap();
        assert!(src.contains("if T.cuda.thread_rank() == 0:"), "{src}");
        assert!(!src.contains("if T.ptx.elect_sync():"), "{src}");
        // Non-zero warp: the same sugar emits elect_sync.
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            warp_if(2, vec![elected_if(vec![init()])]),
        ]))
        .unwrap();
        assert!(src.contains("if T.ptx.elect_sync():"), "{src}");
        assert!(!src.contains("if T.cuda.thread_rank() == 0:"), "{src}");
        // A warp-0 elected region CONTAINING a persistent loop (a worker role
        // on warp 0, e.g. the nvfp4 MMA warp) is a hot-loop guard, not the
        // prologue: it emits elect_sync, not thread_rank.
        let loop_body = vec![Stmt::ForLoop {
            var: crate::ir::scalar::Var {
                id: crate::ir::scalar::VarId(90),
                binding: crate::ir::dtype::VarBinding::Loop,
                dtype: crate::ir::dtype::ScalarDType::I32,
            },
            start: crate::ir::scalar::ScalarValue::Int(0),
            stop: crate::ir::scalar::ScalarValue::Int(4),
            step: crate::ir::scalar::ScalarValue::Int(1),
            body: vec![init()],
            unroll: true,
        }];
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            warp_if(0, vec![elected_if(loop_body)]),
        ]))
        .unwrap();
        assert!(src.contains("if T.ptx.elect_sync():"), "{src}");
        assert!(!src.contains("if T.cuda.thread_rank() == 0:"), "{src}");
    }

    /// A `ClusterBarrierWait` directly inside an elect-form `If` peels out of
    /// the elect (warp-collective; the elected-lane wait deadlocks). Any other
    /// position under single-thread scope fails closed.
    #[test]
    fn cluster_barrier_wait_peels_out_of_the_elect() {
        let peeled = kernel_to_tirx_source(&kernel(vec![warp_if(
            1,
            vec![elected_if(vec![Stmt::ClusterBarrierWait, Stmt::WarpSync])],
        )]))
        .unwrap();
        let wait = peeled.find("T.ptx.barrier.cluster.wait").unwrap();
        let elect = peeled.find("if T.ptx.elect_sync():").unwrap();
        assert!(wait < elect, "{peeled}");

        let nested = kernel(vec![warp_if(
            1,
            vec![elected_if(vec![Stmt::WarpSync, Stmt::ClusterBarrierWait])],
        )]);
        let err = kernel_to_tirx_source(&nested).unwrap_err();
        assert!(err.contains("deadlock"), "{err}");
    }

    // ------------------------------------------------------------------
    // Differential tests for the dropped-field bug class: two IRs that
    // differ ONLY in a field the codegen used to `..`-drop must no longer
    // produce the same code — it either rejects the unrepresentable one
    // or emits different text.
    // ------------------------------------------------------------------

    fn tmem_alloc(base_col: u32, n_cols: u32, cta_group: u8) -> Stmt {
        Stmt::TmemAlloc {
            base_col,
            n_cols,
            cta_group,
        }
    }

    /// The canonical single-band prologue/teardown every shipped kernel uses:
    /// `if warp_id == 0: alloc` ... `if warp_id == 0: <suffix>`.
    fn tmem_kernel(prefix: Vec<Stmt>, suffix: Vec<Stmt>) -> Kernel {
        let mut body = vec![warp_if(0, vec![tmem_alloc(0, 512, 2)])];
        body.extend(prefix);
        body.push(warp_if(0, suffix));
        kernel(body)
    }

    #[test]
    fn tmem_alloc_fields_are_honored_or_rejected() {
        // Legal: base-0, kernel cta_group (the test kernel's cluster is 2) —
        // the alloc/dealloc/relinquish text carries the IR's own values.
        let k = tmem_kernel(
            vec![],
            vec![
                Stmt::TmemDealloc {
                    base_col: 0,
                    n_cols: 512,
                    cta_group: 2,
                },
                Stmt::TmemRelinquish { cta_group: 2 },
            ],
        );
        assert!(k.validate().is_ok(), "{:?}", k.validate().unwrap_err());
        let src = kernel_to_tirx_source(&k).unwrap();
        assert!(
            src.contains("T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=2)"),
            "{src}"
        );
        assert!(
            src.contains("T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=2)"),
            "{src}"
        );
        assert!(
            src.contains("T.ptx.tcgen05.relinquish_alloc_permit(cta_group=2)"),
            "{src}"
        );
        // The TMEM view decl lands at function scope right after the prologue
        // alloc branch (the KernelInit-era position), before any role body.
        let alloc_if = src.find("if warp_id == 0:").unwrap();
        let view = src.find("tmem = T.decl_buffer").unwrap();
        assert!(alloc_if < view, "{src}");

        // A nonzero base band used to silently generate the SAME base-0 code.
        let k = tmem_kernel(vec![warp_if(1, vec![tmem_alloc(64, 64, 2)])], vec![]);
        let err = kernel_to_tirx_source(&k).unwrap_err();
        assert!(err.contains("base_col"), "{err}");

        // A cta_group that disagrees with the kernel used to be silently
        // replaced by the kernel-level value.
        let k = tmem_kernel(vec![], vec![Stmt::TmemRelinquish { cta_group: 1 }]);
        let err = kernel_to_tirx_source(&k).unwrap_err();
        assert!(err.contains("cta_group"), "{err}");
    }

    fn reg_frag_tensor(id: u32) -> Arc<Tensor> {
        reg_frag_tensor_w(id, 8)
    }

    fn reg_frag_tensor_w(id: u32, w: usize) -> Arc<Tensor> {
        Arc::new(Tensor {
            id,
            space: MemorySpace::Reg,
            dtype: DType::F32,
            shape: vec![w],
            layout: None,
            byte_offset: None,
            reg_frag: None,
        })
    }

    fn tcgen05_ld(row: i64, shape: LdStShape, dtype: DType) -> Stmt {
        Stmt::Tcgen05Ld {
            dst: TensorSlice {
                tensor: reg_frag_tensor(7),
                offsets: vec![ScalarValue::Int(0)],
                shape: vec![ScalarValue::Int(8)],
            },
            src: TmemOperand {
                row: ScalarValue::Int(row),
                col: ScalarValue::Int(0),
                dtype,
            },
            shape,
            num: 8,
        }
    }

    fn epilogue_kernel(ld: Stmt) -> Kernel {
        epilogue_kernel_t(ld, reg_frag_tensor(7))
    }

    fn epilogue_kernel_t(ld: Stmt, tensor: Arc<Tensor>) -> Kernel {
        let mut body = vec![warp_if(0, vec![tmem_alloc(0, 512, 2)])];
        body.push(Stmt::TensorDef { tensor });
        body.push(wg_if(0, vec![ld]));
        body.push(warp_if(
            0,
            vec![Stmt::TmemDealloc {
                base_col: 0,
                n_cols: 512,
                cta_group: 2,
            }],
        ));
        kernel(body)
    }

    #[test]
    fn tcgen05_ld_dropped_fields_are_rejected() {
        // row=0 / 32x32b / f32: lowered (and the two row variants no longer
        // share one emission).
        let ok = kernel_to_tirx_source(&epilogue_kernel(tcgen05_ld(
            0,
            LdStShape::B32x32,
            DType::F32,
        )))
        .unwrap();
        assert!(
            ok.contains("Tx.wg.copy_async(accum_frag[:, :], tmem[:, 0:0 + 8])"),
            "{ok}"
        );

        // 16x256b M=64 f32: lowered through the atom view (num=8 -> 32 regs).
        let atom_ld = Stmt::Tcgen05Ld {
            dst: TensorSlice {
                tensor: reg_frag_tensor_w(7, 32),
                offsets: vec![ScalarValue::Int(0)],
                shape: vec![ScalarValue::Int(32)],
            },
            src: TmemOperand {
                row: ScalarValue::Int(0),
                col: ScalarValue::Int(0),
                dtype: DType::F32,
            },
            shape: LdStShape::B16x256,
            num: 8,
        };
        let ok =
            kernel_to_tirx_source(&epilogue_kernel_t(atom_ld, reg_frag_tensor_w(7, 32))).unwrap();
        assert!(
            ok.contains("Tx.wg.copy_async(accum_frag_atom[:, :], tmem[0:64, 0:0 + 64])"),
            "{ok}"
        );

        // num must agree with the fragment width (num=8 needs 32 f32 regs).
        let bad_num = Stmt::Tcgen05Ld {
            dst: TensorSlice {
                tensor: reg_frag_tensor_w(7, 8),
                offsets: vec![ScalarValue::Int(0)],
                shape: vec![ScalarValue::Int(8)],
            },
            src: TmemOperand {
                row: ScalarValue::Int(0),
                col: ScalarValue::Int(0),
                dtype: DType::F32,
            },
            shape: LdStShape::B16x256,
            num: 8,
        };
        let err = kernel_to_tirx_source(&epilogue_kernel(bad_num)).unwrap_err();
        assert!(err.contains("num"), "{err}");

        let err = kernel_to_tirx_source(&epilogue_kernel(tcgen05_ld(
            16,
            LdStShape::B32x32,
            DType::F32,
        )))
        .unwrap_err();
        assert!(err.contains("row"), "{err}");

        // row=16 is the M=128 second issue — no lowering.
        let atom_ld16 = Stmt::Tcgen05Ld {
            dst: TensorSlice {
                tensor: reg_frag_tensor_w(7, 32),
                offsets: vec![ScalarValue::Int(0)],
                shape: vec![ScalarValue::Int(32)],
            },
            src: TmemOperand {
                row: ScalarValue::Int(16),
                col: ScalarValue::Int(0),
                dtype: DType::F32,
            },
            shape: LdStShape::B16x256,
            num: 8,
        };
        let err = kernel_to_tirx_source(&epilogue_kernel_t(atom_ld16, reg_frag_tensor_w(7, 32)))
            .unwrap_err();
        assert!(err.contains("row"), "{err}");

        let err = kernel_to_tirx_source(&epilogue_kernel(tcgen05_ld(
            0,
            LdStShape::B16x32Bx2,
            DType::F32,
        )))
        .unwrap_err();
        assert!(err.contains("16x32bx2"), "{err}");

        let err = kernel_to_tirx_source(&epilogue_kernel(tcgen05_ld(
            0,
            LdStShape::B32x32,
            DType::F16,
        )))
        .unwrap_err();
        assert!(err.contains("dtype"), "{err}");

        // A 16-bit atom read has no lowering (the packed TMEM datapath is st-only).
        let err = kernel_to_tirx_source(&epilogue_kernel(tcgen05_ld(
            0,
            LdStShape::B16x256,
            DType::Bf16,
        )))
        .unwrap_err();
        assert!(err.contains("dtype"), "{err}");
    }

    fn tcgen05_st(dst: TmemOperand, shape: LdStShape, src_w: i64, num: u32) -> Stmt {
        Stmt::Tcgen05St {
            dst,
            src: TensorSlice {
                tensor: reg_frag_tensor(7),
                offsets: vec![ScalarValue::Int(0)],
                shape: vec![ScalarValue::Int(src_w)],
            },
            shape,
            num,
        }
    }

    fn tmem_op(row: i64, col: i64, dtype: DType) -> TmemOperand {
        TmemOperand {
            row: ScalarValue::Int(row),
            col: ScalarValue::Int(col),
            dtype,
        }
    }

    #[test]
    fn tcgen05_st_lowering_and_rejects() {
        // f32 32x32b: tmem[:, c:c+w] <- frag wg view.
        let ok = kernel_to_tirx_source(&epilogue_kernel(tcgen05_st(
            tmem_op(0, 8, DType::F32),
            LdStShape::B32x32,
            8,
            8,
        )))
        .unwrap();
        assert!(
            ok.contains("Tx.wg.copy_async(tmem[:, 8:8 + 8], accum_frag[:, :])"),
            "{ok}"
        );

        // dst width != num: the slice must span exactly the atom's registers.
        let err = kernel_to_tirx_source(&epilogue_kernel(tcgen05_st(
            tmem_op(0, 0, DType::F32),
            LdStShape::B32x32,
            4,
            8,
        )))
        .unwrap_err();
        assert!(err.contains("num"), "{err}");
        // non-32x32b atom: no st lowering (would need an atom-layout src frag).
        let err = kernel_to_tirx_source(&epilogue_kernel(tcgen05_st(
            tmem_op(0, 0, DType::F32),
            LdStShape::B16x256,
            8,
            8,
        )))
        .unwrap_err();
        assert!(err.contains("32x32b"), "{err}");
        // row != 0: the TMEM view bases at lane 0.
        let err = kernel_to_tirx_source(&epilogue_kernel(tcgen05_st(
            tmem_op(16, 0, DType::F32),
            LdStShape::B32x32,
            8,
            8,
        )))
        .unwrap_err();
        assert!(err.contains("row"), "{err}");
        // dtype mismatch REG vs TMEM: the interpreter requires them equal.
        let err = kernel_to_tirx_source(&epilogue_kernel(tcgen05_st(
            tmem_op(0, 0, DType::Bf16),
            LdStShape::B32x32,
            8,
            8,
        )))
        .unwrap_err();
        assert!(err.contains("dtype"), "{err}");
    }

    #[test]
    fn tcgen05_st_packed_half_and_wait() {
        let bf = Arc::new(Tensor {
            id: 8,
            space: MemorySpace::Reg,
            dtype: DType::Bf16,
            shape: vec![16],
            layout: None,
            byte_offset: None,
            reg_frag: None,
        });
        let mut body = vec![warp_if(0, vec![tmem_alloc(0, 512, 2)])];
        body.push(Stmt::TensorDef {
            tensor: reg_frag_tensor(7),
        });
        body.push(Stmt::TensorDef { tensor: bf.clone() });
        body.push(wg_if(
            0,
            vec![
                Stmt::Tcgen05St {
                    dst: tmem_op(0, 128, DType::Bf16),
                    src: TensorSlice {
                        tensor: bf,
                        offsets: vec![ScalarValue::Int(0)],
                        shape: vec![ScalarValue::Int(8)],
                    },
                    shape: LdStShape::B32x32,
                    num: 8,
                },
                Stmt::Tcgen05WaitSt,
            ],
        ));
        body.push(warp_if(
            0,
            vec![Stmt::TmemDealloc {
                base_col: 0,
                n_cols: 512,
                cta_group: 2,
            }],
        ));
        let src = kernel_to_tirx_source(&kernel(body)).unwrap();
        // The packed view is declared over the whole band (2 elems per cell) and
        // the st window doubles the cell column.
        assert!(
            src.contains("tmem_bf16 = T.decl_buffer((128, 1024), \"bfloat16\""),
            "{src}"
        );
        assert!(
            src.contains(
                "Tx.wg.copy_async(tmem_bf16[:, (128) * 2:(128) * 2 + 16], out_frag[:, :])"
            ),
            "{src}"
        );
        assert!(src.contains("T.ptx.tcgen05.wait.st()"), "{src}");
    }

    fn smem_tensor(id: u32, dtype: DType, shape: &[usize]) -> Arc<Tensor> {
        Arc::new(Tensor {
            id,
            space: MemorySpace::Smem,
            dtype,
            shape: shape.to_vec(),
            layout: None,
            byte_offset: None,
            reg_frag: None,
        })
    }

    #[test]
    fn ldmatrix_stmatrix_lowering() {
        let imm_a = reg_tensor(10, DType::U32, 2);
        let tile = reg_tensor(11, DType::Bf16, 2);
        let sm = smem_tensor(12, DType::Bf16, &[64, 64]);
        let smem_row = |off0: ScalarValue, off1: ScalarValue| TensorSlice {
            tensor: sm.clone(),
            offsets: vec![off0, off1],
            shape: vec![ScalarValue::Int(1), ScalarValue::Int(8)],
        };
        let lane_row = ScalarValue::expr(
            ScalarOp::Add,
            vec![
                ScalarValue::Int(8),
                ScalarValue::expr(
                    ScalarOp::Mod,
                    vec![
                        ScalarValue::Scope(ScopeValueKind::LaneId),
                        ScalarValue::Int(8),
                    ],
                ),
            ],
        );
        let body = vec![
            Stmt::TensorDef { tensor: sm.clone() },
            Stmt::TensorDef {
                tensor: imm_a.clone(),
            },
            Stmt::TensorDef {
                tensor: tile.clone(),
            },
            wg_if(
                0,
                vec![
                    Stmt::LdMatrix {
                        dst: reg_slice(&imm_a, 0, 1),
                        src: smem_row(lane_row.clone(), ScalarValue::Int(16)),
                        shape: MatrixShape::M8N8,
                        num: 1,
                        trans: false,
                        dtype: MatrixDType::B16,
                    },
                    Stmt::StMatrix {
                        dst: smem_row(lane_row.clone(), ScalarValue::Int(32)),
                        src: reg_slice(&tile, 0, 2),
                        shape: MatrixShape::M8N8,
                        num: 1,
                        trans: true,
                        dtype: MatrixDType::B16,
                    },
                ],
            ),
        ];
        let src = kernel_to_tirx_source(&kernel(body)).unwrap();
        assert!(
            src.contains(
                "T.ptx.ldmatrix(False, 1, \".b16\", d_smem0.ptr_to([8 + ((lane_id) & 7), 16]), accum_frag_flat.ptr_to([0]))"
            ),
            "{src}"
        );
        assert!(
            src.contains(
                "T.ptx.stmatrix(True, 1, \".b16\", d_smem0.ptr_to([8 + ((lane_id) & 7), 32]), out_frag_flat_u32.ptr_to([0]))"
            ),
            "{src}"
        );

        // dst width must equal num (b32 words).
        let err = kernel_to_tirx_source(&kernel(vec![
            Stmt::TensorDef { tensor: sm.clone() },
            Stmt::TensorDef {
                tensor: imm_a.clone(),
            },
            wg_if(
                0,
                vec![Stmt::LdMatrix {
                    dst: reg_slice(&imm_a, 0, 2),
                    src: smem_row(lane_row.clone(), ScalarValue::Int(0)),
                    shape: MatrixShape::M8N8,
                    num: 1,
                    trans: false,
                    dtype: MatrixDType::B16,
                }],
            ),
        ]))
        .unwrap_err();
        assert!(err.contains("num"), "{err}");
        // an f32 SMEM operand is not a b16 matrix.
        let sm32 = smem_tensor(13, DType::F32, &[64, 64]);
        let err = kernel_to_tirx_source(&kernel(vec![
            Stmt::TensorDef {
                tensor: sm32.clone(),
            },
            Stmt::TensorDef {
                tensor: imm_a.clone(),
            },
            wg_if(
                0,
                vec![Stmt::LdMatrix {
                    dst: reg_slice(&imm_a, 0, 1),
                    src: TensorSlice {
                        tensor: sm32,
                        offsets: vec![lane_row, ScalarValue::Int(0)],
                        shape: vec![ScalarValue::Int(1), ScalarValue::Int(8)],
                    },
                    shape: MatrixShape::M8N8,
                    num: 1,
                    trans: false,
                    dtype: MatrixDType::B16,
                }],
            ),
        ]))
        .unwrap_err();
        assert!(err.contains("dtype"), "{err}");
    }

    #[test]
    fn warp_mma_lowering_and_rejects() {
        let imm_a = reg_tensor(10, DType::U32, 2);
        let imm_b = reg_tensor(11, DType::U32, 1);
        let acc = reg_tensor(12, DType::F32, 4);
        let defs = vec![
            Stmt::TensorDef {
                tensor: imm_a.clone(),
            },
            Stmt::TensorDef {
                tensor: imm_b.clone(),
            },
            Stmt::TensorDef {
                tensor: acc.clone(),
            },
        ];
        let mma = |d: TensorSlice, c: TensorSlice| Stmt::WarpMma {
            d,
            a: reg_slice(&imm_a, 0, 2),
            b: reg_slice(&imm_b, 0, 1),
            c,
            m: 16,
            n: 8,
            k: 8,
            ab_dtype: DType::Bf16,
        };
        let body = {
            let mut b = defs.clone();
            b.push(wg_if(
                0,
                vec![mma(reg_slice(&acc, 0, 4), reg_slice(&acc, 0, 4))],
            ));
            b
        };
        let src = kernel_to_tirx_source(&kernel(body)).unwrap();
        assert!(
            src.contains(
                "T.ptx.mma.legacy(\"m16n8k8\", \"row\", \"col\", \"bfloat16\", \"bfloat16\", \"float32\", accum_frag_flat_ab.data, 0, out_frag_flat_ab.data, 0, reg2_flat.data, 0, False, dtype=\"float32\")"
            ),
            "{src}"
        );

        // d != c: the accumulator is read-modify-write.
        let err = kernel_to_tirx_source(&kernel({
            let mut b = defs.clone();
            b.push(wg_if(
                0,
                vec![mma(reg_slice(&acc, 0, 2), reg_slice(&acc, 0, 4))],
            ));
            b
        }))
        .unwrap_err();
        assert!(err.contains("same fragment"), "{err}");
        // A/B in a bf16 element dtype (not packed words): no lowering.
        let ab16 = reg_tensor(13, DType::Bf16, 4);
        let err = kernel_to_tirx_source(&kernel(vec![
            Stmt::TensorDef {
                tensor: ab16.clone(),
            },
            Stmt::TensorDef {
                tensor: acc.clone(),
            },
            wg_if(
                0,
                vec![Stmt::WarpMma {
                    d: reg_slice(&acc, 0, 4),
                    a: reg_slice(&ab16, 0, 4),
                    b: reg_slice(&ab16, 0, 2),
                    c: reg_slice(&acc, 0, 4),
                    m: 16,
                    n: 8,
                    k: 8,
                    ab_dtype: DType::Bf16,
                }],
            ),
        ]))
        .unwrap_err();
        assert!(err.contains("packed words"), "{err}");
        // Unsupported shape.
        let err = kernel_to_tirx_source(&kernel({
            let mut b = defs.clone();
            b.push(wg_if(
                0,
                vec![Stmt::WarpMma {
                    d: reg_slice(&acc, 0, 4),
                    a: reg_slice(&imm_a, 0, 2),
                    b: reg_slice(&imm_b, 0, 1),
                    c: reg_slice(&acc, 0, 4),
                    m: 8,
                    n: 8,
                    k: 8,
                    ab_dtype: DType::Bf16,
                }],
            ));
            b
        }))
        .unwrap_err();
        assert!(err.contains("m8n8k8"), "{err}");
    }

    #[test]
    fn mbarrier_arrive_emits_bare_per_thread() {
        use super::super::dtype::MBarKind;
        use super::super::mbar::{MBar, MBarRef};
        let mbar = Arc::new(MBar {
            id: 3,
            kind: MBarKind::Thread,
            stages: 1,
            arrive_count: None,
            leader_routed: false,
        });
        let mref = || MBarRef {
            mbar: mbar.clone(),
            remote_coord: None,
        };
        // wg scope: bare per-thread arrive (no `if elect_sync():` guard) — the
        // interpreter arrives once per executing lane and the barrier count is
        // sized for exactly that (an elect undercounts and deadlocks — the gdn
        // gate_ready bug).
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            wg_if(
                0,
                vec![Stmt::MBarrierArrive {
                    mbar: mref(),
                    count: ScalarValue::Int(1),
                    stage: None,
                }],
            ),
        ]))
        .unwrap();
        assert!(
            src.contains("T.ptx.mbarrier.arrive(smem_full.ptr_to([0]))"),
            "{src}"
        );
        assert!(!src.contains("if T.ptx.elect_sync():"), "{src}");
        assert!(!src.contains("if tid_in_wg == 0:"), "{src}");
    }

    #[test]
    fn reg_transfer_forms_lowering() {
        let r1 = reg_tensor(10, DType::F32, 1);
        let frag = reg_tensor(11, DType::F32, 4);
        let sm = smem_tensor(12, DType::F32, &[64, 64]);
        let g = Arc::new(Tensor {
            id: 0,
            space: MemorySpace::Gmem,
            dtype: DType::F32,
            shape: vec![4, 4, 64, 128],
            layout: None,
            byte_offset: None,
            reg_frag: None,
        });
        let tid_row = ScalarValue::Scope(ScopeValueKind::TidInWg);
        let body = vec![
            Stmt::TensorDef { tensor: sm.clone() },
            Stmt::TensorDef { tensor: r1.clone() },
            Stmt::TensorDef {
                tensor: frag.clone(),
            },
            wg_if(
                0,
                vec![
                    // per-thread point load: r1 = sm[tid, 3]
                    Stmt::RegLoad {
                        dst: reg_slice(&r1, 0, 1),
                        src: TensorSlice {
                            tensor: sm.clone(),
                            offsets: vec![tid_row.clone(), ScalarValue::Int(3)],
                            shape: vec![ScalarValue::Int(1), ScalarValue::Int(1)],
                        },
                    },
                    // per-thread point store: sm[tid, 5] = r1
                    Stmt::RegStore {
                        dst: TensorSlice {
                            tensor: sm,
                            offsets: vec![tid_row.clone(), ScalarValue::Int(5)],
                            shape: vec![ScalarValue::Int(1), ScalarValue::Int(1)],
                        },
                        src: reg_slice(&r1, 0, 1),
                    },
                    // REG->REG copy: frag[1] = frag[0]
                    Stmt::RegStore {
                        dst: reg_slice(&frag, 1, 1),
                        src: reg_slice(&frag, 0, 1),
                    },
                    // per-thread GMEM row run: g[1, 2, tid, 16:80] = frag
                    Stmt::RegStore {
                        dst: TensorSlice {
                            tensor: g,
                            offsets: vec![
                                ScalarValue::Int(1),
                                ScalarValue::Int(2),
                                tid_row,
                                ScalarValue::Int(16),
                            ],
                            shape: vec![
                                ScalarValue::Int(1),
                                ScalarValue::Int(1),
                                ScalarValue::Int(1),
                                ScalarValue::Int(4),
                            ],
                        },
                        src: reg_slice(&frag, 0, 4),
                    },
                ],
            ),
        ];
        let src = kernel_to_tirx_source(&kernel(body)).unwrap();
        assert!(
            src.contains("accum_frag_flat[0] = d_smem0[tid_in_wg, 3]"),
            "{src}"
        );
        assert!(
            src.contains("d_smem0[tid_in_wg, 5] = accum_frag_flat[0]"),
            "{src}"
        );
        // REG->REG copy: the flat scalar form (the wg.copy scalar fallback's
        // raw thread-axis BufferLoad is rejected by LowerTIRxCleanup).
        assert!(
            src.contains("out_frag_flat[1 + _i] = out_frag_flat[0]"),
            "{src}"
        );
        assert!(src.contains("for _i in range(4):"), "{src}");
        assert!(
            src.contains("A[1, 2, tid_in_wg, 16 + _i] = out_frag_flat[_i]"),
            "{src}"
        );
    }

    fn reg_tensor(id: u32, dtype: DType, width: i64) -> Arc<Tensor> {
        Arc::new(Tensor {
            id,
            space: MemorySpace::Reg,
            dtype,
            shape: vec![width as usize],
            layout: None,
            byte_offset: None,
            reg_frag: None,
        })
    }

    fn reg_slice(t: &Arc<Tensor>, off: i64, w: i64) -> TensorSlice {
        TensorSlice {
            tensor: t.clone(),
            offsets: vec![ScalarValue::Int(off)],
            shape: vec![ScalarValue::Int(w)],
        }
    }

    fn reg_kernel(body: Vec<Stmt>) -> Kernel {
        kernel(body)
    }

    #[test]
    fn reg_elementwise_family_lowering() {
        let r1 = reg_tensor(10, DType::F32, 1);
        let r2 = reg_tensor(11, DType::F32, 1);
        let rb = reg_tensor(12, DType::Bf16, 2);
        let body = vec![
            Stmt::TensorDef { tensor: r1.clone() },
            Stmt::TensorDef { tensor: r2.clone() },
            Stmt::TensorDef { tensor: rb.clone() },
            wg_if(
                0,
                vec![
                    Stmt::RegFill {
                        dst: reg_slice(&r1, 0, 1),
                        value: RegOperand::Literal(RegLiteral::Int(0)),
                    },
                    Stmt::RegAdd {
                        dst: reg_slice(&r1, 0, 1),
                        lhs: RegOperand::Slice(reg_slice(&r1, 0, 1)),
                        rhs: RegOperand::Slice(reg_slice(&r2, 0, 1)),
                        rounding: Rounding::Rn,
                    },
                    Stmt::RegSub {
                        dst: reg_slice(&r1, 0, 1),
                        lhs: RegOperand::Slice(reg_slice(&r1, 0, 1)),
                        rhs: RegOperand::Literal(RegLiteral::Int(-1)),
                        rounding: Rounding::Rn,
                    },
                    Stmt::RegFma {
                        dst: reg_slice(&r1, 0, 1),
                        a: RegOperand::Slice(reg_slice(&r1, 0, 1)),
                        b: RegOperand::Slice(reg_slice(&r2, 0, 1)),
                        c: RegOperand::Slice(reg_slice(&r1, 0, 1)),
                    },
                    Stmt::RegUnary {
                        dst: reg_slice(&r1, 0, 1),
                        src: RegOperand::Slice(reg_slice(&r2, 0, 1)),
                        op: RegUnaryOp::Exp2,
                    },
                    Stmt::RegUnary {
                        dst: reg_slice(&r1, 0, 1),
                        src: RegOperand::Slice(reg_slice(&r2, 0, 1)),
                        op: RegUnaryOp::Rcp,
                    },
                    Stmt::RegUnary {
                        dst: reg_slice(&r1, 0, 1),
                        src: RegOperand::Slice(reg_slice(&r2, 0, 1)),
                        op: RegUnaryOp::Neg,
                    },
                    Stmt::RegUnary {
                        dst: reg_slice(&r1, 0, 1),
                        src: RegOperand::Slice(reg_slice(&r2, 0, 1)),
                        op: RegUnaryOp::Log2,
                    },
                    Stmt::RegFill {
                        dst: reg_slice(&rb, 0, 2),
                        value: RegOperand::Literal(RegLiteral::Int(1)),
                    },
                    Stmt::RegAdd {
                        dst: reg_slice(&rb, 0, 2),
                        lhs: RegOperand::Slice(reg_slice(&rb, 0, 2)),
                        rhs: RegOperand::Slice(reg_slice(&rb, 0, 2)),
                        rounding: Rounding::Rn,
                    },
                ],
            ),
        ];
        let src = kernel_to_tirx_source(&reg_kernel(body)).unwrap();
        assert!(
            src.contains("Tx.wg.fill(accum_frag[:, :], T.float32(0))"),
            "{src}"
        );
        assert!(
            src.contains("Tx.wg.add(accum_frag[:, :], accum_frag[:, :], out_frag[:, :])"),
            "{src}"
        );
        assert!(
            src.contains("Tx.wg.sub(accum_frag[:, :], accum_frag[:, :], T.float32(-1))"),
            "{src}"
        );
        assert!(
            src.contains(
                "Tx.wg.fma(accum_frag[:, :], accum_frag[:, :], out_frag[:, :], accum_frag[:, :])"
            ),
            "{src}"
        );
        assert!(
            src.contains("Tx.wg.exp2(accum_frag[:, :], out_frag[:, :])"),
            "{src}"
        );
        assert!(
            src.contains("Tx.wg.reciprocal(accum_frag[:, :], out_frag[:, :])"),
            "{src}"
        );
        assert!(
            src.contains("Tx.wg.mul(accum_frag[:, :], out_frag[:, :], T.float32(-1))"),
            "{src}"
        );
        // log2: no tile op — per-thread scalar loop on the flat views.
        assert!(
            src.contains("accum_frag_flat = accum_frag.local()"),
            "{src}"
        );
        assert!(src.contains("for _i in range(1):"), "{src}");
        // (width-1 src broadcasts — the interpreter's one-element-per-thread rule)
        assert!(
            src.contains("accum_frag_flat[_i] = T.log2(out_frag_flat[0])"),
            "{src}"
        );
        // bf16 fill/add: literal takes the dst dtype.
        assert!(
            src.contains("Tx.wg.fill(reg2[:, :], T.bfloat16(1))"),
            "{src}"
        );
        assert!(
            src.contains("Tx.wg.add(reg2[:, :], reg2[:, :], reg2[:, :])"),
            "{src}"
        );
    }

    #[test]
    fn reg_elementwise_dropped_fields_are_rejected() {
        let r1 = reg_tensor(10, DType::F32, 1);
        let ri = reg_tensor(11, DType::I32, 1);
        let mk = |stmts: Vec<Stmt>| {
            let mut body = vec![
                Stmt::TensorDef { tensor: r1.clone() },
                Stmt::TensorDef { tensor: ri.clone() },
            ];
            body.push(wg_if(0, stmts));
            kernel_to_tirx_source(&reg_kernel(body))
        };
        // rounding=rm has no lowering (the elementwise ops carry no floor).
        let err = mk(vec![Stmt::RegAdd {
            dst: reg_slice(&r1, 0, 1),
            lhs: RegOperand::Literal(RegLiteral::Int(1)),
            rhs: RegOperand::Literal(RegLiteral::Int(2)),
            rounding: Rounding::Rm,
        }])
        .unwrap_err();
        assert!(err.contains("rm"), "{err}");
        // int dst for the binary family: no lowering.
        let err = mk(vec![Stmt::RegSub {
            dst: reg_slice(&ri, 0, 1),
            lhs: RegOperand::Literal(RegLiteral::Int(1)),
            rhs: RegOperand::Literal(RegLiteral::Int(2)),
            rounding: Rounding::Rn,
        }])
        .unwrap_err();
        assert!(err.contains("dtype"), "{err}");
        // 16-bit unary dst: the interpreter computes in f32 and rounds; the
        // TIRx 16-bit elementwise path would compute in 16 bits.
        let rb = reg_tensor(12, DType::Bf16, 1);
        let err = kernel_to_tirx_source(&reg_kernel(vec![
            Stmt::TensorDef { tensor: rb.clone() },
            wg_if(
                0,
                vec![Stmt::RegUnary {
                    dst: reg_slice(&rb, 0, 1),
                    src: RegOperand::Slice(reg_slice(&rb, 0, 1)),
                    op: RegUnaryOp::Exp2,
                }],
            ),
        ]))
        .unwrap_err();
        assert!(err.contains("dtype"), "{err}");
        // a mixed-dtype operand is NOT silently coerced.
        let err = mk(vec![Stmt::RegAdd {
            dst: reg_slice(&r1, 0, 1),
            lhs: RegOperand::Slice(reg_slice(&ri, 0, 1)),
            rhs: RegOperand::Literal(RegLiteral::Int(2)),
            rounding: Rounding::Rn,
        }])
        .unwrap_err();
        assert!(err.contains("RegCvt"), "{err}");
    }

    #[test]
    fn fence_scope_is_lowered_memory_view_rejected() {
        use super::super::dtype::{FenceKind, FenceScope};
        let fence = |kind, scope| Stmt::Fence { kind, scope };
        // Cta vs Cluster: both lowered, to DIFFERENT qualifiers (was: both
        // emitted "shared::cta" regardless of the IR scope).
        let cta =
            kernel_to_tirx_source(&kernel(vec![fence(FenceKind::AsyncProxy, FenceScope::Cta)]))
                .unwrap();
        let cluster = kernel_to_tirx_source(&kernel(vec![fence(
            FenceKind::AsyncProxy,
            FenceScope::Cluster,
        )]))
        .unwrap();
        assert!(
            cta.contains("T.ptx.fence.proxy_async(\"shared::cta\")"),
            "{cta}"
        );
        assert!(
            cluster.contains("T.ptx.fence.proxy_async(\"shared::cluster\")"),
            "{cluster}"
        );
        assert_ne!(cta, cluster);
        // Memory/View are sim-only ordering markers: fail closed (was: a
        // silent no-op emission).
        for kind in [FenceKind::Memory, FenceKind::View] {
            let err =
                kernel_to_tirx_source(&kernel(vec![fence(kind, FenceScope::Gpu)])).unwrap_err();
            assert!(err.contains("sim-only"), "{err}");
        }
    }

    #[test]
    fn empty_if_body_emits_pass() {
        // An empty `If` body used to render a bare `if warp_id == 0:` header
        // with no indented statement — invalid Python. The structured emitter
        // fills every empty block with `pass`.
        let k = kernel(vec![warp_if(0, vec![])]);
        let src = kernel_to_tirx_source(&k).unwrap();
        assert!(src.contains("if warp_id == 0:\n        pass"), "{src}");
        // And every block opener still owns a body line.
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !line.trim_end().ends_with(':') {
                continue;
            }
            let indent = line.len() - line.trim_start().len();
            let body = lines[i + 1..]
                .iter()
                .find(|l| !l.trim().is_empty())
                .expect("block opener must be followed by a body");
            let body_indent = body.len() - body.trim_start().len();
            assert!(body_indent > indent, "empty block at: {line}");
        }
    }

    /// A cta_sync is emitted at function scope and suppressed inside a
    /// warp/warpgroup branch (a CTA-wide `__syncthreads` not all threads reach).
    #[test]
    fn cta_sync_is_function_scope_only() {
        let src = kernel_to_tirx_source(&kernel(vec![Stmt::CtaSync])).unwrap();
        assert!(src.contains("T.cuda.cta_sync()"), "{src}");
        let src = kernel_to_tirx_source(&kernel(vec![warp_if(1, vec![Stmt::CtaSync])])).unwrap();
        assert!(!src.contains("T.cuda.cta_sync()"), "{src}");
    }
}
