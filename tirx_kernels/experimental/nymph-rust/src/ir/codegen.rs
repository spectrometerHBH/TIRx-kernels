//! Rust codegen: lower a nymph `ir::Kernel` to a TVMScript (`tvm.script.tirx`)
//! source string. Ported from the Role-model codegen (PR #18) onto the
//! warp-model IR: the `Role`/`KernelInit`/`KernelFinalize` nodes are gone —
//! thread dispatch is plain `Stmt::If` over scalar predicates.
//!
//! ZERO-INFERENCE guard rule (user-mandated): codegen NEVER synthesizes an
//! emission guard from the statically-computed thread scope. Every guard in
//! the output comes from the IR:
//! - Every `Stmt::If` prints its scalar predicate literally. In particular,
//!   `lane_id == 0` remains `lane_id == 0`; codegen never substitutes
//!   `elect_sync()` or `thread_rank() == 0`. The `if_elected` SUGAR is its own
//!   IR predicate (`ScopeValueKind::Elected`), printed as the
//!   `T.ptx.elect_sync()` intrinsic — canon's exact elected-region form.
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
//!   legality checks, never to invent or rewrite a guard.
//! - Control-flow structure is preserved exactly: sibling IR `If`s emit as
//!   sibling TVMScript `if`s, nested IR `If`s emit nested, and source order is
//!   unchanged. Codegen never merges conditions, synthesizes role-dispatch
//!   parents, or re-nests sibling branches.
//! - `KernelInit`'s two side duties survive as structural rules: the single
//!   TMEM view buffer (`tmem`) + SF views are declared at function scope right
//!   after the top-level statement containing the first `TmemAlloc`.
//!
//! Everything else is the #18 pass unchanged: full-K `gemm_async` at the IR's
//! own granularity, runtime `accum` scalar, TmemOperand/SfView TMEM views,
//! exact per-operation `MBarRef` lowering, the structured emitter
//! (`fill_empty_blocks`), the pow2/trunc
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
use super::tcgen05_layout::{
    resolve_tcgen05_cp, resolve_tcgen05_mma, tcgen05_cp_source_bits, CpLaneLayout, CpSmemLayout,
    TmemDatapath,
};
use super::tensor::{
    Layout, MmaAOperand, MmaElemFormat, ScaleFormat, SmemTile, Tensor, TensorSlice, TmemAForm,
    TmemAddr,
};
use super::thread_filter::{static_thread_filter, ThreadSet};
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The imports the emitted source needs (prepended so the file is self-contained).
const HEADER_IMPORTS: &str = "\
from tvm.backend.cuda.operator.tile_primitive.gemm_async.tcgen05 import sf_tmem_layout
from tvm.backend.cuda.operator.tile_primitive.tma_utils import SwizzleMode
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import ComposeLayout, R, S, TCol, TileLayout, TLane
from tvm.tirx.layout import tmem_datapath_layout, tmem_mma_operand_layout
";

/// Apply an explicit physical TMEM `(lane, 32-bit column)` offset to a
/// statement-local non-owning buffer view.  The base layout's TCol unit is the
/// buffer element, so callers convert physical columns to elements before
/// invoking this helper.
const TMEM_VIEW_LAYOUT_HELPER: &str = r#"
def tmem_view_layout(layout, lane_offset, col_offset):
    offset = dict(layout.offset)
    offset[TLane] = offset.get(TLane, 0) + lane_offset
    offset[TCol] = offset.get(TCol, 0) + col_offset
    return TileLayout.from_iters(layout.shard, layout.replica, offset)
"#;

/// The thread scope of the enclosing `If`-condition stack, derived per body
/// from `static_thread_filter` (see `classify_scope`). It plays the role the
/// Role node's warp/warpgroup/elected fields played in #18, but only as a
/// fail-closed legality check for hardware collectives and single-issue ops.
/// It never changes, deletes, or supplements the IR's control flow.
///
/// A single-issue operation is printed bare only when the enclosing explicit
/// IR predicates prove an elected scope. Otherwise codegen returns an error;
/// it never invents the missing predicate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scope {
    /// Function scope (no narrowing condition, or a full-CTA one).
    Function,
    /// Inside a one-full-warp branch (`if warp_id == w:`).
    Warp,
    /// Inside a one-full-warpgroup branch (`if wg_id == g:`).
    Warpgroup,
    /// Inside an explicit branch with at most one lane per warp (or one
    /// statically identified thread). Single-issue ops emit in place.
    Elected,
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

/// Per-kernel naming + lookup context built once, then read while walking the body.
struct Ctx {
    /// Tensor id -> emitted Python name.
    names: HashMap<u32, String>,
    /// mbar id -> emitted Python name of its dynamic-pool buffer view.
    mbar_names: HashMap<u32, String>,
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
    /// REG tensors that need a non-owning u32 reinterpret for packed b16
    /// physical instruction operands.
    reg_u32_views: HashSet<u32>,
    /// Number of launched clusters (`launch_cta_count / cta_group`) — the grid stride
    /// for a `ForEachTask` grid-stride scheduler loop.
    num_clusters: usize,
    /// Deterministic statement-local TMEM view suffix.  Each tcgen05 statement
    /// declares only the non-owning views it consumes; there is no kernel-wide
    /// TMEM buffer, cache, or hoisting pass.
    tmem_view_index: Cell<usize>,
    /// Var ids provably non-negative (see `collect_nonneg_vars`) — the ONLY
    /// authority `is_nonneg` consults for `ScalarValue::Var`: ForLoop induction
    /// vars with a non-negative-literal start and positive-literal step, plus
    /// scalar vars whose every definition is provably non-negative (fixpoint).
    /// A bare `Var(_) => true` would silently strength-reduce a `%`/`//` on a
    /// sentinel-negative scalar (e.g. a drained-scheduler `task_id == -1`).
    nonneg_vars: std::collections::HashSet<u32>,
}

impl Ctx {
    fn tensor_name(&self, id: u32) -> Result<&str, String> {
        self.names
            .get(&id)
            .map(|s| s.as_str())
            .ok_or_else(|| format!("codegen: no name for tensor id {id}"))
    }

    fn next_tmem_view_index(&self) -> usize {
        let index = self.tmem_view_index.get();
        self.tmem_view_index.set(index + 1);
        index
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

/// `Swizzle` -> the exact TIRx allocator enum member.
fn swizzle_mode_name(sw: Swizzle) -> &'static str {
    match sw {
        Swizzle::None => "SWIZZLE_NONE",
        Swizzle::B32 => "SWIZZLE_32B_ATOM",
        Swizzle::B64 => "SWIZZLE_64B_ATOM",
        Swizzle::B128 => "SWIZZLE_128B_ATOM",
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

/// Render a Python tuple shape. A one-dimensional shape needs its trailing
/// comma: `(4,)`, not the parenthesized scalar `(4)` that `SMEMPool.alloc`
/// rejects.
fn python_shape(shape: &[usize]) -> String {
    let dims = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if shape.len() == 1 {
        format!("({dims},)")
    } else {
        format!("({dims})")
    }
}

pub fn kernel_to_tirx_source(k: &Kernel) -> Result<String, String> {
    let ctx = build_ctx(k)?;
    let mut out = Emitter::new();

    out.push_str(HEADER_IMPORTS);
    out.push_str(TMEM_VIEW_LAYOUT_HELPER);
    out.push('\n');

    // Argument tensors, named by position (A, B, C, D, …). The fp16/bootstrap GEMM
    // has 3 (A, B, C-out); the nvfp4 GEMM has 5 (A, B, SFA, SFB, D-out). Names are
    // cosmetic — TVM matches args positionally.
    if k.args.is_empty() {
        return Err("codegen: kernel has no args".to_string());
    }

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
        let dims = python_shape(&t.shape);
        if t.layout.is_some() {
            return Err(format!(
                "codegen: GMEM tensor {} cannot carry a layout",
                t.id
            ));
        }
        out.push_str(&format!(
            "{p}{name} = T.match_buffer({name}_ptr, {dims}, \"{dt}\")\n",
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
    // Lane within the warp. Kernel IR uses this value directly in any
    // single-lane predicate it requires; codegen preserves that predicate.
    out.push_str(&format!(
        "{p}lane_id = T.lane_id([{warp_lanes}])\n",
        p = pad(ind),
        warp_lanes = WARP_LANES
    ));
    out.push('\n');

    // ---- one physical dynamic-SMEM pool ----
    // Tensor views, mbar cells, and the optional TMEM-address cell all carry
    // absolute byte offsets in the IR. Codegen never derives a metadata base
    // from tensor ids, declaration order, or an allocator cursor.
    out.push_str(&format!("{p}pool = T.SMEMPool()\n", p = pad(ind)));
    for t in collect_tensors(k) {
        if t.space != MemorySpace::Smem {
            continue;
        }
        let name = ctx.tensor_name(t.id)?;
        let dims = python_shape(&t.shape);
        // `move_base_to` is exact because validate has already proved every
        // explicitly aligned tensor offset satisfies the same alignment
        // codegen requests here. No silent allocator round-up is permitted.
        let off = t
            .byte_offset
            .ok_or_else(|| format!("codegen: smem tensor {name} has no byte_offset"))?;
        out.push_str(&format!(
            "{p}pool.move_base_to({off})\n",
            p = pad(ind),
            off = off,
        ));
        match t.layout {
            Some(Layout::Swizzle(layout)) => out.push_str(&format!(
                "{p}{name} = pool.alloc_tcgen05_mma_AB({dims}, \"{dt}\", \
                 swizzle_mode=SwizzleMode.{mode}, align=1024)\n",
                p = pad(ind),
                name = name,
                dims = dims,
                dt = dtype_str(t.dtype),
                mode = swizzle_mode_name(layout.swizzle),
            )),
            None => out.push_str(&format!(
                "{p}{name} = pool.alloc({dims}, \"{dt}\", scope=\"shared.dyn\")\n",
                p = pad(ind),
                name = name,
                dims = dims,
                dt = dtype_str(t.dtype),
            )),
        }
    }
    // ---- mbar shared buffers + tmem_addr ----
    // A multi-stage mbarrier allocates `[stages]` slots; each op indexes the slot it
    // uses. A single-stage mbar keeps the bootstrap's `[1]` form.
    for mbar in collect_mbars(k) {
        let name = ctx
            .mbar_names
            .get(&mbar.id)
            .ok_or_else(|| format!("codegen: no name for mbar {}", mbar.id))?;
        out.push_str(&format!(
            "{p}pool.move_base_to({off})\n",
            p = pad(ind),
            off = mbar.byte_offset,
        ));
        out.push_str(&format!(
            "{p}{name} = pool.alloc([{stages}], \"uint64\", scope=\"shared.dyn\", align=8)\n",
            p = pad(ind),
            name = name,
            stages = mbar.stages,
        ));
    }
    if let Some(off) = tmem_addr_byte_offset(k)? {
        out.push_str(&format!("{p}pool.move_base_to({off})\n", p = pad(ind),));
        out.push_str(&format!(
            "{p}tmem_addr = pool.alloc([1], \"uint32\", scope=\"shared.dyn\", align=4)\n",
            p = pad(ind)
        ));
    }
    // `smem_size_bytes` is the physical extent of the complete pool, including
    // explicit padding and metadata tail.
    out.push_str(&format!(
        "{p}pool.commit(size={size})\n",
        p = pad(ind),
        size = k.smem_size_bytes,
    ));

    // REG tensors are emitted inline at their TensorDef sites as per-thread
    // `T.alloc_local` arrays with exactly the IR shape and dtype.
    out.push('\n');

    // ---- walk the body ----
    // Emit top-level statements 1:1 in IR order. TMEM views are statement-local
    // declarations emitted immediately beside the physical instruction that
    // consumes them; nothing is hoisted to function scope or inserted after a
    // different IR statement.
    let fn_scope = ScopeInfo::function(k.num_warps);
    emit_body(&mut out, &k.body, ind, &ctx, &fn_scope)?;

    Ok(render_lines(fill_empty_blocks(out.finish())))
}

/// Every emitted block opener (`if`/`else`/`for`/`while`/`def` …) must own at
/// least one body line: an empty If/guard body otherwise renders a header with
/// no indented statement — invalid Python. Fill the gap with `pass`,
/// generically at the structured-line level (covers empty `else` arms and
/// empty guard blocks alike) — a per-construct validator ban would have to
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

/// One emitted source line: its indent in 4-space units and its text WITHOUT
/// the leading pad.
#[derive(Clone, PartialEq, Eq)]
struct Line {
    indent: usize,
    text: String,
}

/// The emission sink accumulates source as structured lines so empty blocks
/// can receive `pass` without reparsing or changing the IR's control-flow
/// nesting. `push_str` keeps call sites string-shaped; the sink splits lines
/// and records their indentation (all padding uses `pad()` = 4-space units).
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

/// Collect physical mbar objects in source declaration order. Their ids remain
/// identity only; physical placement comes exclusively from `byte_offset`.
fn collect_mbars(k: &Kernel) -> Vec<Arc<super::mbar::MBar>> {
    fn walk(stmts: &[Stmt], seen: &mut HashSet<u32>, out: &mut Vec<Arc<super::mbar::MBar>>) {
        for stmt in stmts {
            if let Stmt::MBarDef { mbar } = stmt {
                if seen.insert(mbar.id) {
                    out.push(mbar.clone());
                }
            }
            for body in stmt.child_bodies() {
                walk(body, seen, out);
            }
        }
    }

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    walk(&k.body, &mut seen, &mut out);
    out
}

/// Return the one physical SMEM cell used to receive the TMEM allocation
/// address. Multiple sequential `TmemAlloc` statements may reuse that cell,
/// but distinct offsets cannot be represented by the single `tmem_addr` view
/// consumed by `TmemDealloc`.
fn tmem_addr_byte_offset(k: &Kernel) -> Result<Option<usize>, String> {
    fn walk(stmts: &[Stmt], offset: &mut Option<usize>) -> Result<(), String> {
        for stmt in stmts {
            if let Stmt::TmemAlloc {
                addr_byte_offset, ..
            } = stmt
            {
                match offset {
                    Some(existing) if *existing != *addr_byte_offset => {
                        return Err(format!(
                            "codegen: TmemAlloc addr_byte_offset {addr_byte_offset} \
                             differs from the kernel tmem_addr offset {existing}"
                        ));
                    }
                    None => *offset = Some(*addr_byte_offset),
                    _ => {}
                }
            }
            for body in stmt.child_bodies() {
                walk(body, offset)?;
            }
        }
        Ok(())
    }

    let mut offset = None;
    walk(&k.body, &mut offset)?;
    Ok(offset)
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
            mma_m: _,
            mma_n: _,
            format: _,
            block_scale: _,
            accum: _,
            trans_a: _,
            trans_b: _,
            ws: _,
            cta_group: _,
        } => {
            if let MmaAOperand::Smem(tile) = a {
                note_tensor(&tile.tensor, map);
            }
            note_tensor(&b.tensor, map);
        }
        Tcgen05Cp {
            dst: _,
            src,
            shape: _,
            multicast: _,
            cta_group: _,
        } => {
            note_tensor(&src.tensor, map);
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
    let mut task_idx = 0usize;
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
                if t.layout.is_some() || t.dtype == DType::U8 {
                    let n = format!("ab_smem{ab_idx}");
                    ab_idx += 1;
                    n
                } else if is_int {
                    let n = if task_idx == 0 {
                        "task_smem".to_string()
                    } else {
                        format!("task_smem{task_idx}")
                    };
                    task_idx += 1;
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

    // Mbar names come only from their definitions. Remote addressing stays on
    // each MBarRef and is rendered at the instruction that consumes it.
    let mut mbar_names: HashMap<u32, String> = HashMap::new();
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
            for body in s.child_bodies() {
                walk_mbars(body, mbar_names, mbar_idx, mbar_default);
            }
        }
    }
    walk_mbars(&k.body, &mut mbar_names, &mut mbar_idx, &mbar_default);

    // cta_group from the cluster size (the bootstrap is cta_group=2).
    let cta_group = k.cluster_shape.iter().product::<usize>().max(1) as u8;

    // Packed b16 physical instructions address the same local REG storage
    // through a non-owning u32 reinterpret.
    let mut reg_u32_views = HashSet::new();
    collect_reg_u32_views(&k.body, &mut reg_u32_views);

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
        var_names: HashMap::new(),
        scalar_names,
        cta_group,
        num_warps: k.num_warps,
        reg_u32_views,
        num_clusters: (k.launch_cta_count() / (cta_group as usize).max(1)).max(1),
        tmem_view_index: Cell::new(0),
        nonneg_vars: collect_nonneg_vars(k),
    })
}

/// Record REG tensors whose packed b16 physical instructions require a u32
/// reinterpret. Every REG tensor still owns exactly one `T.alloc_local`.
fn collect_reg_u32_views(stmts: &[Stmt], views: &mut HashSet<u32>) {
    for s in stmts {
        match s {
            Stmt::Tcgen05Ld { dst, .. } => {
                if matches!(dst.tensor.dtype, DType::F16 | DType::Bf16) {
                    views.insert(dst.tensor.id);
                }
            }
            Stmt::Tcgen05St { src, .. } => {
                if matches!(src.tensor.dtype, DType::F16 | DType::Bf16) {
                    views.insert(src.tensor.id);
                }
            }
            Stmt::LdMatrix { dst, .. } => {
                if matches!(dst.tensor.dtype, DType::F16 | DType::Bf16) {
                    views.insert(dst.tensor.id);
                }
            }
            Stmt::StMatrix { src, .. } => {
                if matches!(src.tensor.dtype, DType::F16 | DType::Bf16) {
                    views.insert(src.tensor.id);
                }
            }
            _ => {}
        }
        for body in s.child_bodies() {
            collect_reg_u32_views(body, views);
        }
    }
}

/// Slice a rank-1 per-thread local REG array.
fn emit_reg_view_slice(
    _out: &mut Emitter,
    _p: &str,
    t: &Arc<Tensor>,
    off: &ScalarValue,
    width: usize,
    ctx: &Ctx,
) -> Result<String, String> {
    let name = ctx.tensor_name(t.id)?.to_string();
    let full = t.shape.first().copied().unwrap_or(0);
    let all = as_int(off) == Some(0) && width == full;
    if all {
        Ok(format!("{name}[:]"))
    } else {
        let off_s = emit_scalar(off, ctx)?;
        Ok(format!("{name}[{off_s}:{off_s} + {width}]"))
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

/// A `T.<dtype>(value)` scalar literal for thread-local REG operands (a bare number
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

/// One operand arm of a `Tx.thread.*` elementwise call: a local slice, or a typed
/// scalar for a literal. Slice operands must share the dst dtype — the
/// interpreter coerces per-op, so a genuinely mixed-dtype op must say so with
/// an explicit RegCvt instead of being silently coerced here.
fn emit_reg_operand(
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

/// The owning per-thread local storage name for a REG tensor.
fn reg_name(t: &Arc<Tensor>, ctx: &Ctx) -> Result<String, String> {
    Ok(ctx.tensor_name(t.id)?.to_string())
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

/// Validate the ldmatrix/stmatrix SMEM operand: one contiguous 8-element b16
/// row. Owning SMEM tensors may carry explicit leading stage/tile axes, so
/// those axes remain in the slice as extent-one point dimensions.
fn check_matrix_smem_row(s: &TensorSlice, label: &str) -> Result<(), String> {
    if s.shape.is_empty()
        || as_int(s.shape.last().expect("non-empty")) != Some(8)
        || s.shape[..s.shape.len() - 1]
            .iter()
            .any(|extent| as_int(extent) != Some(1))
    {
        return Err(format!(
            "codegen: {label} SMEM operand must end in one contiguous row of eight b16 elements \
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

fn emit_matrix_smem_ptr(s: &TensorSlice, ctx: &Ctx) -> Result<String, String> {
    let name = ctx.tensor_name(s.tensor.id)?;
    let offsets = s
        .offsets
        .iter()
        .map(|offset| emit_scalar(offset, ctx))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    Ok(format!("{name}.ptr_to([{offsets}])"))
}

/// Shared lowering for `RegAdd`/`RegSub`/`RegMul` on per-thread local arrays.
/// `rounding=rm` (the interpreter's post-op floor) has no TIRx elementwise
/// form — fail closed rather than silently skip the floor.
fn emit_reg_binary(
    out: &mut Emitter,
    p: &str,
    dst: &TensorSlice,
    lhs: &RegOperand,
    rhs: &RegOperand,
    rounding: Rounding,
    op: &str,
    ctx: &Ctx,
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
    let dst_s = emit_reg_view_slice(out, p, t, off, w, ctx)?;
    let lhs_s = emit_reg_operand(lhs, t.dtype, out, p, ctx)?;
    let rhs_s = emit_reg_operand(rhs, t.dtype, out, p, ctx)?;
    out.push_str(&format!("{p}Tx.thread.{op}({dst_s}, {lhs_s}, {rhs_s})\n"));
    Ok(())
}

/// Render the physical b32 register tuple carried by one tcgen05.ld/st
/// instruction.  The IR's REG slice is in dtype elements; a 16-bit fragment
/// therefore contributes two elements per physical register and is addressed
/// through its uint32 reinterpret view.
fn emit_tcgen05_reg_tuple(
    slice: &TensorSlice,
    shape: LdStShape,
    num: u32,
    ctx: &Ctx,
) -> Result<(String, bool), String> {
    let (tensor, offset, width) = reg_slice_parts(slice)?;
    let register_count = shape.register_count(num).ok_or_else(|| {
        format!(
            "codegen: invalid tcgen05.ld/st shape={} num={num}",
            shape.as_str()
        )
    })?;
    let is_b16 = matches!(tensor.dtype, DType::F16 | DType::Bf16);
    if !is_b16 && !matches!(tensor.dtype, DType::F32 | DType::I32 | DType::U32) {
        return Err(format!(
            "codegen: tcgen05.ld/st REG dtype {:?} has no physical b32 register form",
            tensor.dtype
        ));
    }
    let expected_width = register_count * if is_b16 { 2 } else { 1 };
    if width != expected_width {
        return Err(format!(
            "codegen: tcgen05.ld/st shape={} num={num} needs {expected_width} \
             REG elements of dtype {:?}, got {width}",
            shape.as_str(),
            tensor.dtype
        ));
    }

    let name = ctx.tensor_name(tensor.id)?;
    let (view, base) = if is_b16 {
        let Some(offset) = as_int(offset) else {
            return Err(
                "codegen: packed 16-bit tcgen05.ld/st REG offset must be static".to_string(),
            );
        };
        if offset < 0 || offset % 2 != 0 {
            return Err(format!(
                "codegen: packed 16-bit tcgen05.ld/st REG offset {offset} \
                 must be non-negative and even"
            ));
        }
        (format!("{name}_u32"), (offset / 2).to_string())
    } else {
        (name.to_string(), emit_scalar(offset, ctx)?)
    };
    let regs = (0..register_count)
        .map(|index| format!("{view}[{}]", flat_add(&base, index)))
        .collect::<Vec<_>>()
        .join(", ");
    Ok((regs, is_b16))
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
        // `Elected` never reaches scope_name: emit_scalar_prec special-cases
        // it to the `T.ptx.elect_sync()` intrinsic.
        Elected => "elected",
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
        // The `if_elected` predicate is the elect.sync intrinsic itself
        // (canon's `if T.ptx.elect_sync():` — one elected lane per warp,
        // single-issue ops inside emit bare).
        ScalarValue::Scope(ScopeValueKind::Elected) => Ok("T.ptx.elect_sync()".to_string()),
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
        ScalarOp::Xor => format!(
            "T.bitwise_xor({}, {})",
            emit_scalar_prec(&e.args[0], ctx, 0)?,
            emit_scalar_prec(&e.args[1], ctx, 0)?
        ),
        _ => {
            let Some(sym) = binop_symbol(e.op) else {
                // An op with no TVMScript lowering must not leak a
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

/// Emit a literal BufferRegion for a generic synchronous `Tx.copy`. Every IR
/// dimension remains an explicit range, including extent-one dimensions, so
/// the printer never turns a one-element region into a scalar BufferLoad.
fn emit_buffer_region(s: &TensorSlice, ctx: &Ctx) -> Result<String, String> {
    let name = ctx.tensor_name(s.tensor.id)?;
    let dims = s
        .offsets
        .iter()
        .zip(&s.shape)
        .map(|(offset, extent)| {
            let lo = emit_scalar(offset, ctx)?;
            let hi = add_bound(offset, extent, ctx)?;
            Ok(format!("{lo}:{hi}"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(format!("{name}[{}]", dims.join(", ")))
}

/// Emit the exact rank-2 SMEM tile carried by a physical tcgen05 statement.
/// Leading axes are explicit point indices; only the final row/column axes are
/// ranges.  No shape or stage axis is inferred here.
fn emit_named_smem_tile(tile: &SmemTile, name: &str, ctx: &Ctx) -> Result<String, String> {
    let mut dims = tile
        .prefix_indices
        .iter()
        .map(|index| emit_scalar(index, ctx))
        .collect::<Result<Vec<_>, _>>()?;
    let row_lo = emit_scalar(&tile.row_offset, ctx)?;
    let row_hi = add_bound(
        &tile.row_offset,
        &ScalarValue::Int(i64::from(tile.rows)),
        ctx,
    )?;
    let col_lo = emit_scalar(&tile.col_offset, ctx)?;
    let col_hi = add_bound(
        &tile.col_offset,
        &ScalarValue::Int(i64::from(tile.cols)),
        ctx,
    )?;
    dims.push(format!("{row_lo}:{row_hi}"));
    dims.push(format!("{col_lo}:{col_hi}"));
    Ok(format!("{name}[{}]", dims.join(", ")))
}

fn emit_explicit_smem_tile(tile: &SmemTile, ctx: &Ctx) -> Result<String, String> {
    emit_named_smem_tile(tile, ctx.tensor_name(tile.tensor.id)?, ctx)
}

/// Emit an MMA SMEM operand.  F4 is physically backed by U8 in Nymph, so the
/// explicit byte offsets/extents become twice as many logical e2m1 elements
/// after the buffer-level view; every other format uses the tile verbatim.
fn emit_mma_smem_tile(tile: &SmemTile, format: MmaElemFormat, ctx: &Ctx) -> Result<String, String> {
    if format != MmaElemFormat::F4E2M1 {
        return emit_explicit_smem_tile(tile, ctx);
    }
    let name = ctx.tensor_name(tile.tensor.id)?;
    let mut dims = tile
        .prefix_indices
        .iter()
        .map(|index| emit_scalar(index, ctx))
        .collect::<Result<Vec<_>, _>>()?;
    let row_lo = emit_scalar(&tile.row_offset, ctx)?;
    let row_hi = add_bound(
        &tile.row_offset,
        &ScalarValue::Int(i64::from(tile.rows)),
        ctx,
    )?;
    let col_lo = emit_scalar_prec(&tile.col_offset, ctx, 4)?;
    let col_hi = add_bound(
        &tile.col_offset,
        &ScalarValue::Int(i64::from(tile.cols)),
        ctx,
    )?;
    dims.push(format!("{row_lo}:{row_hi}"));
    dims.push(format!("({col_lo}) * 2:({col_hi}) * 2"));
    Ok(format!(
        "{name}.view(\"float4_e2m1fn\")[{}]",
        dims.join(", ")
    ))
}

fn mma_format_dtype(format: MmaElemFormat) -> &'static str {
    match format {
        MmaElemFormat::F16 => "float16",
        MmaElemFormat::BF16 => "bfloat16",
        MmaElemFormat::F8E4M3 => "float8_e4m3fn",
        MmaElemFormat::F4E2M1 => "float4_e2m1fn",
    }
}

fn mma_format_bits(format: MmaElemFormat) -> u32 {
    match format {
        MmaElemFormat::F16 | MmaElemFormat::BF16 => 16,
        MmaElemFormat::F8E4M3 => 8,
        MmaElemFormat::F4E2M1 => 4,
    }
}

fn scale_format_dtype(format: ScaleFormat) -> &'static str {
    match format {
        ScaleFormat::E8M0FNU => "float8_e8m0fnu",
        ScaleFormat::E4M3FN => "float8_e4m3fn",
    }
}

fn bool_py(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

/// Physical TMEM coordinates become offsets on the statement-local buffer
/// layout. `elements_per_cell` converts the IR's 32-bit column unit into the
/// declared buffer dtype's TCol element unit.
fn emit_tmem_layout_offsets(
    addr: &TmemAddr,
    elements_per_cell: u32,
    extra_col_elements: u32,
    ctx: &Ctx,
) -> Result<(String, String), String> {
    let lane = emit_scalar(&addr.row, ctx)?;
    let expr = if let Some(col) = as_int(&addr.col) {
        col.checked_mul(i64::from(elements_per_cell))
            .and_then(|value| value.checked_add(i64::from(extra_col_elements)))
            .ok_or_else(|| "codegen: TMEM column offset overflows i64".to_string())?
            .to_string()
    } else {
        let col = emit_scalar_prec(&addr.col, ctx, 4)?;
        let scaled = if elements_per_cell == 1 {
            col
        } else {
            format!("({col}) * {elements_per_cell}")
        };
        if extra_col_elements == 0 {
            scaled
        } else {
            format!("({scaled}) + {extra_col_elements}")
        }
    };
    Ok((lane, expr))
}

fn datapath_name(datapath: TmemDatapath) -> &'static str {
    match datapath {
        TmemDatapath::A => "A",
        TmemDatapath::B => "B",
        TmemDatapath::D => "D",
        TmemDatapath::E => "E",
        TmemDatapath::F => "F",
    }
}

fn emit_cp_tmem_layout(lane_layout: CpLaneLayout, rows: u32, cols: u32) -> Result<String, String> {
    let layout = match lane_layout {
        CpLaneLayout::Identity => {
            if rows % 128 != 0 {
                return Err(format!(
                    "codegen: identity tcgen05.cp rows {rows} are not divisible by 128"
                ));
            }
            format!(
                "TileLayout(S[({}, 128, {cols}) : ({cols} @ TCol, 1 @ TLane, 1 @ TCol)])",
                rows / 128
            )
        }
        CpLaneLayout::Quadrant4 => {
            if rows % 4 != 0 {
                return Err(format!(
                    "codegen: 4x tcgen05.cp rows {rows} are not divisible by 4"
                ));
            }
            format!(
                "TileLayout(S[({}, 4, {cols}) : (1 @ TLane, 32 @ TLane, 1 @ TCol)])",
                rows / 4
            )
        }
        CpLaneLayout::Warp2_02_13 => {
            if rows % 64 != 0 {
                return Err(format!(
                    "codegen: 64x tcgen05.cp rows {rows} are not divisible by 64"
                ));
            }
            format!(
                "TileLayout(S[({}, 64, {cols}) : ({cols} @ TCol, 1 @ TLane, 1 @ TCol)] + R[2:64 @ TLane])",
                rows / 64
            )
        }
        CpLaneLayout::Warp2_01_23 => {
            if rows % 64 != 0 {
                return Err(format!(
                    "codegen: 64x tcgen05.cp rows {rows} are not divisible by 64"
                ));
            }
            format!(
                "TileLayout(S[({}, 2, 32, {cols}) : ({cols} @ TCol, 64 @ TLane, 1 @ TLane, 1 @ TCol)] + R[2:32 @ TLane])",
                rows / 64
            )
        }
        CpLaneLayout::Warp4 => {
            if rows % 32 != 0 {
                return Err(format!(
                    "codegen: 32x tcgen05.cp rows {rows} are not divisible by 32"
                ));
            }
            format!(
                "TileLayout(S[({}, 32, {cols}) : ({cols} @ TCol, 1 @ TLane, 1 @ TCol)] + R[4:32 @ TLane])",
                rows / 32
            )
        }
    };
    Ok(layout)
}

fn emit_cp_smem_layout(layout: CpSmemLayout, shape: &[usize], bits: u32) -> Result<String, String> {
    let shape_s = python_shape(shape);
    match layout {
        CpSmemLayout::Plain16B => {
            let rows = shape[0];
            let cols = shape[1];
            Ok(format!("TileLayout(S[({rows}, {cols}) : ({cols}, 1)])"))
        }
        CpSmemLayout::Swizzle32B => {
            let elements_per_128b = 128 / bits;
            if !elements_per_128b.is_power_of_two() || shape.len() != 2 {
                return Err(
                    "codegen: invalid B32 tcgen05.cp source dtype or tensor rank".to_string(),
                );
            }
            let per_element = elements_per_128b.ilog2();
            let period = 1u32
                .checked_shl(per_element + 1 + 3)
                .ok_or_else(|| "codegen: B32 tcgen05.cp layout period overflows".to_string())?;
            let atom_shape = vec![8, (256 / bits) as usize];
            let mut tile_shape = atom_shape.clone();
            tile_shape[0] = shape[0];
            Ok(format!(
                "ComposeLayout({per_element}, 1, 3, \
                 TileLayout(S[{}])).tile_to({}, {}).tile_to({shape_s}, {}).canonicalize()",
                python_shape(&[period as usize]),
                python_shape(&tile_shape),
                python_shape(&atom_shape),
                python_shape(&tile_shape),
            ))
        }
    }
}

fn emit_cp_smem_elem_offset(
    layout: CpSmemLayout,
    tile: &SmemTile,
    bits: u32,
    ctx: &Ctx,
) -> Result<String, String> {
    let byte_offset = tile
        .tensor
        .byte_offset
        .ok_or_else(|| "codegen: Tcgen05Cp source has no absolute byte_offset".to_string())?;
    let base_bits = byte_offset
        .checked_mul(8)
        .ok_or_else(|| "codegen: Tcgen05Cp source byte_offset overflows".to_string())?;
    if base_bits % bits as usize != 0 {
        return Err(format!(
            "codegen: Tcgen05Cp source byte_offset {byte_offset} is not element aligned"
        ));
    }
    let base_elements = base_bits / bits as usize;
    let rank = tile.tensor.shape.len();
    let mut prefix_linear = "0".to_string();
    for (index, extent) in tile
        .prefix_indices
        .iter()
        .zip(tile.tensor.shape[..rank - 2].iter())
    {
        let index = emit_scalar_prec(index, ctx, 4)?;
        prefix_linear = format!("({prefix_linear}) * {extent} + ({index})");
    }
    let rows = tile.tensor.shape[rank - 2];
    let cols = tile.tensor.shape[rank - 1];
    let physical = match layout {
        CpSmemLayout::Plain16B => {
            let row = emit_scalar_prec(&tile.row_offset, ctx, 4)?;
            let col = emit_scalar_prec(&tile.col_offset, ctx, 4)?;
            format!("((({prefix_linear}) * {rows} + ({row})) * {cols} + ({col}))")
        }
        CpSmemLayout::Swizzle32B => {
            let row = as_int(&tile.row_offset)
                .ok_or_else(|| "codegen: B32 Tcgen05Cp row_offset is not static".to_string())?;
            let col = as_int(&tile.col_offset)
                .ok_or_else(|| "codegen: B32 Tcgen05Cp col_offset is not static".to_string())?;
            let atom_cols = i64::from(256 / bits);
            let col_block = col / atom_cols;
            let row_block = row / 8;
            format!(
                "(({prefix_linear}) * {} + {col_block} * {} + {row_block} * {})",
                rows * cols,
                rows * atom_cols as usize,
                8 * atom_cols as usize,
            )
        }
    };
    Ok(format!("{base_elements} + {physical}"))
}

fn cp_multicast_config(multicast: super::stmt::Tcgen05CpMulticast) -> &'static str {
    use super::stmt::Tcgen05CpMulticast::*;
    match multicast {
        None => "",
        Warp2_02_13 => "warpx2::02_13",
        Warp2_01_23 => "warpx2::01_23",
        Warp4 => "warpx4",
    }
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

// ===========================================================================
// statement walk
// ===========================================================================

/// Emit a statement list exactly in IR order. No statement is skipped,
/// coalesced, or supplemented with code that is absent from the IR.
fn emit_body(
    out: &mut Emitter,
    stmts: &[Stmt],
    indent: usize,
    ctx: &Ctx,
    scope: &ScopeInfo,
) -> Result<(), String> {
    for stmt in stmts {
        emit_stmt(out, stmt, indent, ctx, scope)?;
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
        // REG storage is exactly the tensor declared by the IR: one per-thread
        // local array with the IR shape and dtype, emitted at the TensorDef site.
        TensorDef { tensor } if tensor.space == MemorySpace::Reg => {
            let name = ctx.tensor_name(tensor.id)?.to_string();
            let shape = python_shape(&tensor.shape);
            out.push_str(&format!(
                "{p}{name} = T.alloc_local({shape}, \"{dt}\")\n",
                dt = dtype_str(tensor.dtype),
            ));
            if ctx.reg_u32_views.contains(&tensor.id) {
                out.push_str(&format!("{p}{name}_u32 = {name}.view(\"uint32\")\n"));
            }
            Ok(())
        }
        // ---- definitions handled in the header; skip in the body walk ----
        TensorDef { .. } | MBarDef { .. } => Ok(()),

        // ---- TMEM alloc / dealloc / relinquish ----
        // TMEM data views are declared statement-locally by the instructions
        // that consume them; only the explicit SMEM address cell is shared by
        // alloc/dealloc. Validation proves the lifecycle is base-0, balanced,
        // and uses the kernel CTA group. These arms recheck fields they cannot
        // honor because codegen is also callable without validation.
        TmemAlloc {
            base_col,
            n_cols,
            cta_group,
            addr_byte_offset: _,
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
        // (the kernel's own `sched_arr` full barrier). The enclosing explicit
        // single-lane IR branch provides its single-issue scope.
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
            // The MBarRef is the complete address authority. A remote ref uses
            // the intrinsic's explicit cluster target; bytes stay literal.
            let slot_ptr = local_mbar_slot_ptr(mbar, stage, ctx)?;
            let remote = mbar
                .remote_coord
                .as_ref()
                .map(|coord| emit_scalar(coord, ctx))
                .transpose()?;
            let args = remote
                .map(|coord| format!(", remote={coord}, pred=True"))
                .unwrap_or_default();
            out.push_str(&format!(
                "{p}T.ptx.mbarrier.arrive.expect_tx({slot_ptr}, {bytes}{args})\n"
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
            let slot_ptr = local_mbar_slot_ptr(mbar, stage, ctx)?;
            let count_arg = if as_int(count) == Some(1) {
                String::new()
            } else {
                format!(", count={}", emit_scalar(count, ctx)?)
            };
            let body = match &mbar.remote_coord {
                Some(remote) => format!(
                    "T.ptx.mbarrier.arrive({slot_ptr}{count_arg}, remote={cta}, pred=True)",
                    cta = emit_scalar(remote, ctx)?,
                ),
                None => format!("T.ptx.mbarrier.arrive({slot_ptr}{count_arg})"),
            };
            out.push_str(&format!("{p}{body}\n"));
            Ok(())
        }
        MBarrierWait { mbar, phase, stage } => {
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
            // The TMA completion target is exactly the MBarRef stored in IR.
            let mbar_ptr = mbar_slot_ptr(mbar, mbar_stage, ctx)?;
            // The SMEM dst is a staged tile (a leading size-1 ring dim): drop it to an
            // integer index so the operand rank matches the 2D GMEM region.
            let dst_s = emit_smem_tile(dst, ctx)?;
            let src_s = emit_gmem_region(src, coords, gmem_extents(gmem_shape, shape), ctx)?;
            // `multicast_cta_mask`: a `multicast::cluster` g2c copy — one TMA fills the
            // SMEM of EVERY CTA in the mask (canon's `cta_mask=pair_mask` for the shared
            // SFB scale band), so the cluster shares ONE load instead of each CTA reading
            // the full band (halving the L2/TMA traffic). The completion's transaction
            // count lands on exactly the mbar address carried by this operation.
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
                    "Tx.copy_async({dst_s}, {src_s}, dispatch=\"tma_auto\", mbar={mbar_ptr}, cta_group={cg}{cta_mask}{cache_hint_kw}{prefetch_kw})",
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
            mma_m,
            mma_n,
            format,
            block_scale,
            accum,
            trans_a,
            trans_b,
            ws,
            cta_group,
        } => {
            if *cta_group != ctx.cta_group {
                return Err(format!(
                    "codegen: Tcgen05Mma cta_group={} != kernel cta_group={}",
                    cta_group, ctx.cta_group
                ));
            }
            let resolved = resolve_tcgen05_mma(
                dst,
                a,
                b,
                *mma_m,
                *mma_n,
                *format,
                block_scale.as_ref(),
                *trans_a,
                *trans_b,
                *ws,
                *cta_group,
            )
            .map_err(|error| format!("codegen: {error}"))?;
            let view_index = ctx.next_tmem_view_index();

            // D is a statement-local, non-owning view at the address carried
            // by this exact IR operation. No kernel-wide TMEM cache exists.
            let d_name = format!("mma_d{view_index}");
            let (d_lane, d_col) = emit_tmem_layout_offsets(dst, 1, 0, ctx)?;
            let d_layout = format!(
                "tmem_view_layout(tmem_datapath_layout(\"{}\", {}, {}), {d_lane}, {d_col})",
                datapath_name(resolved.datapath),
                resolved.d.logical.rows,
                resolved.d.logical.cols,
            );
            out.push_str(&format!(
                "{p}{d_name} = T.decl_buffer(({}, {}), \"float32\", scope=\"tmem\", \
                 allocated_addr={}, layout={d_layout})\n",
                resolved.d.logical.rows, resolved.d.logical.cols, dst.tensor.start_col,
            ));
            let dst_s = format!("{d_name}[:, :]");

            let a_s = match a {
                MmaAOperand::Smem(tile) => emit_mma_smem_tile(tile, *format, ctx)?,
                MmaAOperand::Tmem { addr, form } => {
                    let footprint = resolved.a_tmem.as_ref().ok_or_else(|| {
                        "codegen: resolver omitted the Tcgen05Mma TMEM A footprint".to_string()
                    })?;
                    let a_name = format!("mma_a{view_index}");
                    let elements_per_cell = 32 / mma_format_bits(*format);
                    let (a_lane, a_col) =
                        emit_tmem_layout_offsets(addr, elements_per_cell, 0, ctx)?;
                    let (shape_s, region_s) = match form {
                        TmemAForm::Flat => (
                            format!("({}, {})", footprint.logical.rows, footprint.logical.cols),
                            format!("{a_name}[:, :]"),
                        ),
                        TmemAForm::BankBatched => (
                            format!(
                                "(2, {}, {})",
                                footprint.logical.rows, footprint.logical.cols
                            ),
                            format!("{a_name}[:, :, :]"),
                        ),
                    };
                    let a_layout = format!(
                        "tmem_view_layout(tmem_mma_operand_layout(\"A\", {shape_s}, \
                         \"{}\", M={}, cta_group={}, ws={}), {a_lane}, {a_col})",
                        mma_format_dtype(*format),
                        mma_m,
                        cta_group,
                        bool_py(*ws),
                    );
                    out.push_str(&format!(
                        "{p}{a_name} = T.decl_buffer({shape_s}, \"{}\", scope=\"tmem\", \
                         allocated_addr={}, layout={a_layout})\n",
                        mma_format_dtype(*format),
                        addr.tensor.start_col,
                    ));
                    region_s
                }
            };
            let b_s = emit_mma_smem_tile(b, *format, ctx)?;
            let accum_s = match accum {
                ScalarValue::Int(0) => "False".to_string(),
                ScalarValue::Int(1) => "True".to_string(),
                other => emit_scalar(other, ctx)?,
            };

            let call = if let Some(spec) = block_scale {
                let sfa = resolved.sfa.as_ref().ok_or_else(|| {
                    "codegen: resolver omitted the Tcgen05Mma SFA footprint".to_string()
                })?;
                let sfb = resolved.sfb.as_ref().ok_or_else(|| {
                    "codegen: resolver omitted the Tcgen05Mma SFB footprint".to_string()
                })?;
                let sfa_name = format!("mma_sfa{view_index}");
                let sfb_name = format!("mma_sfb{view_index}");
                let sfa_extra = sfa.cell_delta * 4 + u32::from(sfa.subbyte);
                let sfb_extra = sfb.cell_delta * 4 + u32::from(sfb.subbyte);
                let (sfa_lane, sfa_col) = emit_tmem_layout_offsets(&spec.sfa, 4, sfa_extra, ctx)?;
                let (sfb_lane, sfb_col) = emit_tmem_layout_offsets(&spec.sfb, 4, sfb_extra, ctx)?;
                let sfa_layout = format!(
                    "tmem_view_layout(sf_tmem_layout({}, SF_K={}, sf_per_mma={}, \
                     sf_reuse={}), {sfa_lane}, {sfa_col})",
                    sfa.footprint.logical.rows, sfa.sf_k, spec.sf_per_mma, spec.sf_reuse,
                );
                let sfb_layout = format!(
                    "tmem_view_layout(sf_tmem_layout({}, SF_K={}, sf_per_mma={}, \
                     sf_reuse={}), {sfb_lane}, {sfb_col})",
                    sfb.footprint.logical.rows, sfb.sf_k, spec.sf_per_mma, spec.sf_reuse,
                );
                out.push_str(&format!(
                    "{p}{sfa_name} = T.decl_buffer(({}, {}), \"{}\", scope=\"tmem\", \
                     allocated_addr={}, layout={sfa_layout})\n",
                    sfa.footprint.logical.rows,
                    sfa.logical_last,
                    scale_format_dtype(spec.scale_format),
                    spec.sfa.tensor.start_col,
                ));
                out.push_str(&format!(
                    "{p}{sfb_name} = T.decl_buffer(({}, {}), \"{}\", scope=\"tmem\", \
                     allocated_addr={}, layout={sfb_layout})\n",
                    sfb.footprint.logical.rows,
                    sfb.logical_last,
                    scale_format_dtype(spec.scale_format),
                    spec.sfb.tensor.start_col,
                ));
                format!(
                    "Tx.gemm_async({dst_s}, {a_s}, {b_s}, \
                     SFA={sfa_name}[:, :], SFB={sfb_name}[:, :], accum={accum_s}, \
                     transA={}, transB={}, dispatch=\"tcgen05\", cta_group={}, \
                     mma_m={}, mma_n={}, weight_stationary={})",
                    bool_py(*trans_a),
                    bool_py(*trans_b),
                    cta_group,
                    mma_m,
                    mma_n,
                    bool_py(*ws),
                )
            } else {
                format!(
                    "Tx.gemm_async({dst_s}, {a_s}, {b_s}, accum={accum_s}, \
                     transA={}, transB={}, dispatch=\"tcgen05\", cta_group={}, \
                     mma_m={}, mma_n={}, weight_stationary={})",
                    bool_py(*trans_a),
                    bool_py(*trans_b),
                    cta_group,
                    mma_m,
                    mma_n,
                    bool_py(*ws),
                )
            };
            emit_single_issue(out, &p, scope, "tcgen05_mma", &call)?;
            Ok(())
        }
        // One physical tcgen05.cp IR statement becomes one Tx.copy_async.
        // The explicit source tile, shape, multicast, and CTA group select the
        // statement-local physical source/destination views.
        Tcgen05Cp {
            dst,
            src,
            shape,
            multicast,
            cta_group,
        } => {
            if *cta_group != ctx.cta_group {
                return Err(format!(
                    "codegen: Tcgen05Cp cta_group={} != kernel cta_group={}",
                    cta_group, ctx.cta_group
                ));
            }
            let resolved = resolve_tcgen05_cp(dst, src, *shape, *multicast, *cta_group)
                .map_err(|error| format!("codegen: {error}"))?;
            let bits = tcgen05_cp_source_bits(src);
            if bits > 32 || 32 % bits != 0 {
                return Err(format!(
                    "codegen: Tcgen05Cp source dtype {:?} must divide one 32-bit TMEM cell",
                    src.tensor.dtype
                ));
            }

            let view_index = ctx.next_tmem_view_index();
            let dst_name = format!("cp_dst{view_index}");
            let src_name = format!("cp_src{view_index}");
            let elements_per_cell = 32 / bits;
            let (dst_lane, dst_col) = emit_tmem_layout_offsets(dst, elements_per_cell, 0, ctx)?;
            let base_layout = emit_cp_tmem_layout(
                resolved.lane_layout,
                resolved.source.rows,
                resolved.source.cols,
            )?;
            let src_owner = ctx.tensor_name(src.tensor.id)?;
            let src_view_rows = if resolved.source_layout == CpSmemLayout::Swizzle32B {
                resolved.source.rows.div_ceil(8) * 8
            } else {
                resolved.source.rows
            };
            let src_view_shape = [src_view_rows as usize, resolved.source.cols as usize];
            let src_layout = emit_cp_smem_layout(resolved.source_layout, &src_view_shape, bits)?;
            let src_elem_offset = emit_cp_smem_elem_offset(resolved.source_layout, src, bits, ctx)?;
            out.push_str(&format!(
                "{p}{src_name} = T.decl_buffer({}, \"{}\", data={src_owner}.data, \
                 elem_offset={src_elem_offset}, \
                 scope=\"shared.dyn\", layout={src_layout})\n",
                python_shape(&src_view_shape),
                dtype_str(src.tensor.dtype),
            ));
            let src_s = format!(
                "{src_name}[0:{}, 0:{}]",
                resolved.source.rows, resolved.source.cols
            );
            let dst_layout = format!("tmem_view_layout({base_layout}, {dst_lane}, {dst_col})");
            out.push_str(&format!(
                "{p}{dst_name} = T.decl_buffer(({}, {}), \"{}\", scope=\"tmem\", \
                 allocated_addr={}, layout={dst_layout})\n",
                resolved.source.rows,
                resolved.source.cols,
                dtype_str(src.tensor.dtype),
                dst.tensor.start_col,
            ));
            let call = format!(
                "Tx.copy_async({dst_name}[:, :], {src_s}, shape=\"{}\", \
                 multicast=\"{}\", cta_group={})",
                shape.as_str(),
                cp_multicast_config(*multicast),
                cta_group,
            );
            emit_single_issue(out, &p, scope, "tcgen05_cp", &call)?;
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

        // ---- epilogue physical load + per-thread register operations ----
        Tcgen05Ld {
            dst,
            src,
            shape,
            num,
        } => {
            let (regs, pack) = emit_tcgen05_reg_tuple(dst, *shape, *num, ctx)?;
            let row = emit_scalar(&src.row, ctx)?;
            let col = emit_scalar(&src.col, ctx)?;
            out.push_str(&format!(
                "{p}T.ptx.tcgen05.ld(T.uint32({}), {regs}, shape=\"{}\", num={}, \
                 row={row}, col={col}, pack={})\n",
                src.tensor.start_col,
                shape.as_str(),
                num,
                bool_py(pack),
            ));
            Ok(())
        }
        Tcgen05WaitLd => {
            out.push_str(&format!("{p}T.ptx.tcgen05.wait.ld()\n"));
            Ok(())
        }
        RegCvt { dst, src, rounding } => {
            if *rounding != Rounding::Rn {
                return Err(
                    "codegen: RegCvt rounding=rm has no Tx.thread.cast lowering".to_string()
                );
            }
            let (dt, doff, dw) = reg_slice_parts(dst)?;
            let (st, soff, sw) = reg_slice_parts(src)?;
            if sw != 1 && sw != dw {
                return Err(format!(
                    "codegen: RegCvt src width {sw} must be 1 or the dst width {dw}"
                ));
            }
            let dst_s = emit_reg_view_slice(out, &p, dt, doff, dw, ctx)?;
            let src_s = emit_reg_view_slice(out, &p, st, soff, sw, ctx)?;
            out.push_str(&format!("{p}Tx.thread.cast({dst_s}, {src_s})\n"));
            Ok(())
        }
        // Register transfers preserve the explicit IR slices one-for-one.
        RegLoad { dst, src } => {
            let (tensor, offset, width) = reg_slice_parts(dst)?;
            let dst_s = emit_reg_view_slice(out, &p, tensor, offset, width, ctx)?;
            let src_s = if src.tensor.space == MemorySpace::Reg {
                let (tensor, offset, width) = reg_slice_parts(src)?;
                emit_reg_view_slice(out, &p, tensor, offset, width, ctx)?
            } else {
                emit_buffer_region(src, ctx)?
            };
            out.push_str(&format!("{p}Tx.copy({dst_s}, {src_s})\n"));
            Ok(())
        }
        RegStore { dst, src } => {
            let (src_tensor, src_offset, src_width) = reg_slice_parts(src)?;
            let src_s = emit_reg_view_slice(out, &p, src_tensor, src_offset, src_width, ctx)?;
            let dst_s = if dst.tensor.space == MemorySpace::Reg {
                let (tensor, offset, width) = reg_slice_parts(dst)?;
                emit_reg_view_slice(out, &p, tensor, offset, width, ctx)?
            } else {
                emit_buffer_region(dst, ctx)?
            };
            out.push_str(&format!("{p}Tx.copy({dst_s}, {src_s})\n"));
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
            // The validator rejects statically narrowed CTA barriers. Codegen
            // must still print the statement at its exact IR position; it may
            // not silently delete or hoist control-dependent synchronization.
            out.push_str(&format!("{p}T.cuda.cta_sync()\n"));
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
            // elect context). Fail loudly instead of changing its IR scope or
            // emitting a silently-hanging kernel.
            if matches!(scope.scope, Scope::Elected) {
                return Err(
                    "codegen: ClusterBarrierWait under elect scope would emit a single-thread \
                     barrier.cluster.wait (hardware deadlock). Put it explicitly at warp scope \
                     (all threads of the warp) in the IR."
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
        // `mma.sync.aligned.m16n8k{8,16}.row.col.f32.{ab}.{ab}.f32` — one IR
        // WarpMma prints one non-legacy `T.ptx.mma` call. Operand register
        // handles come directly from the explicit local tensor slices;
        // LdMatrix defines the packed A/B word representation.
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
            let (dt, doff, dw) = reg_slice_parts(d)?;
            let (ct, coff, cw) = reg_slice_parts(c)?;
            let (at, aoff, aw) = reg_slice_parts(a)?;
            let (bt, boff, bw) = reg_slice_parts(b)?;
            if dt.dtype != DType::F32 || ct.dtype != DType::F32 {
                return Err(format!(
                    "codegen: WarpMma D/C dtypes {:?}/{:?} must both be f32",
                    dt.dtype, ct.dtype
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
                (doff, dw, len_cd, "D"),
                (coff, cw, len_cd, "C"),
                (aoff, aw, len_a, "A"),
                (boff, bw, len_b, "B"),
            ] {
                if w != want {
                    return Err(format!(
                        "codegen: WarpMma {label} fragment must span exactly {want} b32 \
                         registers (got off={off:?}, width {w})"
                    ));
                }
            }
            let ptrs = |tensor: &Arc<Tensor>,
                        offset: &ScalarValue,
                        count: usize|
             -> Result<String, String> {
                let flat = reg_name(tensor, ctx)?;
                let base = emit_scalar(offset, ctx)?;
                Ok(format!(
                    "[{}]",
                    (0..count)
                        .map(|index| { format!("{flat}.ptr_to([{}])", flat_add(&base, index)) })
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            };
            let d_ptrs = ptrs(dt, doff, len_cd)?;
            let c_ptrs = ptrs(ct, coff, len_cd)?;
            let a_ptrs = ptrs(at, aoff, len_a)?;
            let b_ptrs = ptrs(bt, boff, len_b)?;
            let ab_s = dtype_str(*ab_dtype);
            out.push_str(&format!(
                "{p}T.ptx.mma(\"m{m}n{n}k{k}\", \"row\", \"col\", \"float32\", \
                 \"{ab_s}\", \"{ab_s}\", \"float32\", {d_ptrs}, {a_ptrs}, \
                 {b_ptrs}, {c_ptrs})\n"
            ));
            Ok(())
        }
        GmemAtomicAdd { .. } => Err("codegen: GmemAtomicAdd not yet supported".to_string()),
        GmemWaitEq { .. } => Err("codegen: GmemWaitEq not yet supported".to_string()),
        CpAsyncBulkS2Cluster { .. } => {
            Err("codegen: CpAsyncBulkS2Cluster not yet supported".to_string())
        }

        // NVFP4 epilogue alpha rescale.
        RegMul { dst, lhs, rhs } => {
            emit_reg_binary(out, &p, dst, lhs, rhs, Rounding::Rn, "mul", ctx)
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
            let (regs, _) = emit_tcgen05_reg_tuple(src, *shape, *num, ctx)?;
            let row = emit_scalar(&dst.row, ctx)?;
            let col = emit_scalar(&dst.col, ctx)?;
            out.push_str(&format!(
                "{p}T.ptx.tcgen05.st(T.uint32({}), {regs}, shape=\"{}\", num={}, \
                 row={row}, col={col}, unpack={})\n",
                dst.tensor.start_col,
                shape.as_str(),
                num,
                bool_py(false),
            ));
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
            check_matrix_smem_row(src, "LdMatrix")?;
            let doff_s = emit_scalar(doff, ctx)?;
            let handles: String = match dt.dtype {
                DType::U32 | DType::I32 => {
                    if dw != *num as usize {
                        return Err(format!(
                            "codegen: LdMatrix dst width {dw} != num {num} (the dst spans \
                             num b32 registers)"
                        ));
                    }
                    let dflat = reg_name(dt, ctx)?;
                    (0..*num as usize)
                        .map(|i| format!("{dflat}.ptr_to([{}])", flat_add(&doff_s, i)))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
                // A b16 fragment dst: the xN words are written through the
                // `_u32` reinterpret — consecutive b16 pairs ARE the b32
                // registers (the mirror of the StMatrix b16 src form).
                DType::F16 | DType::Bf16 => {
                    if dw != 2 * *num as usize {
                        return Err(format!(
                            "codegen: LdMatrix b16 dst width {dw} != 2*num {} (the dst \
                             spans num packed b16x2 registers)",
                            2 * *num as usize
                        ));
                    }
                    let Some(word_off) = as_int(doff) else {
                        return Err(
                            "codegen: LdMatrix b16 dst offset must be static (the u32 word \
                             index is offset/2)"
                                .to_string(),
                        );
                    };
                    if word_off % 2 != 0 {
                        return Err(format!(
                            "codegen: LdMatrix b16 dst offset {word_off} is odd (a packed \
                             b16x2 register starts at an even element)"
                        ));
                    }
                    let name = ctx.tensor_name(dt.id)?;
                    (0..*num as usize)
                        .map(|i| format!("{name}_u32.ptr_to([{}])", word_off / 2 + i as i64))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
                _ => {
                    return Err(format!(
                        "codegen: LdMatrix dst dtype {:?} has no lowering (u32/i32 packed \
                         words or an f16/bf16 fragment)",
                        dt.dtype
                    ))
                }
            };
            let src_ptr = emit_matrix_smem_ptr(src, ctx)?;
            out.push_str(&format!(
                "{p}T.ptx.ldmatrix({}, {num}, \".b16\", {src_ptr}, {handles})\n",
                py_bool(*trans),
            ));
            Ok(())
        }
        // `stmatrix.sync.aligned.m8n8.xN.b16` — REG words -> SMEM. Mirror of
        // LdMatrix; a bf16/f16 src fragment reads through the `_u32`
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
                    let flat = reg_name(st, ctx)?;
                    (0..*num as usize)
                        .map(|i| format!("{flat}.ptr_to([{}])", flat_add(&soff_s, i)))
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
                        .map(|i| format!("{name}_u32.ptr_to([{}])", word_off / 2 + i as i64))
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
            let dst_ptr = emit_matrix_smem_ptr(dst, ctx)?;
            out.push_str(&format!(
                "{p}T.ptx.stmatrix({}, {num}, \".b16\", {dst_ptr}, {handles})\n",
                py_bool(*trans),
            ));
            Ok(())
        }
        // Per-thread elementwise fill/copy.
        RegFill { dst, value } => {
            let (t, off, w) = reg_slice_parts(dst)?;
            let dst_s = emit_reg_view_slice(out, &p, t, off, w, ctx)?;
            match value {
                RegOperand::Literal(literal) => {
                    let value_s = typed_scalar(t.dtype, *literal)?;
                    out.push_str(&format!("{p}Tx.thread.fill({dst_s}, {value_s})\n"));
                }
                RegOperand::Slice(src) => {
                    let (st, soff, sw) = reg_slice_parts(src)?;
                    if st.dtype != t.dtype {
                        return Err(format!(
                            "codegen: RegFill src dtype {:?} != dst dtype {:?}",
                            st.dtype, t.dtype
                        ));
                    }
                    if sw != 1 && sw != w {
                        return Err(format!(
                            "codegen: RegFill src width {sw} must be 1 or dst width {w}"
                        ));
                    }
                    let src_s = emit_reg_view_slice(out, &p, st, soff, sw, ctx)?;
                    out.push_str(&format!("{p}Tx.thread.copy({dst_s}, {src_s})\n"));
                }
            }
            Ok(())
        }
        RegAdd {
            dst,
            lhs,
            rhs,
            rounding,
        } => emit_reg_binary(out, &p, dst, lhs, rhs, *rounding, "add", ctx),
        RegSub {
            dst,
            lhs,
            rhs,
            rounding,
        } => emit_reg_binary(out, &p, dst, lhs, rhs, *rounding, "sub", ctx),
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
            let dst_s = emit_reg_view_slice(out, &p, t, off, w, ctx)?;
            let a_s = emit_reg_operand(a, t.dtype, out, &p, ctx)?;
            let b_s = emit_reg_operand(b, t.dtype, out, &p, ctx)?;
            let c_s = emit_reg_operand(c, t.dtype, out, &p, ctx)?;
            out.push_str(&format!("{p}Tx.thread.fma({dst_s}, {a_s}, {b_s}, {c_s})\n"));
            Ok(())
        }
        // Elementwise unary over an f32 local array maps directly to Tx.thread.
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
                let dst_s = emit_reg_view_slice(out, &p, t, off, w, ctx)?;
                out.push_str(&format!("{p}Tx.thread.fill({dst_s}, T.float32({v}))\n"));
                return Ok(());
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
            let dst_s = emit_reg_view_slice(out, &p, t, off, w, ctx)?;
            let src_s = emit_reg_view_slice(out, &p, st, soff, sw, ctx)?;
            match op {
                RegUnaryOp::Exp2 => {
                    out.push_str(&format!("{p}Tx.thread.exp2({dst_s}, {src_s})\n"));
                }
                RegUnaryOp::Log2 => {
                    out.push_str(&format!("{p}Tx.thread.log2({dst_s}, {src_s})\n"));
                }
                RegUnaryOp::Rcp => {
                    out.push_str(&format!("{p}Tx.thread.reciprocal({dst_s}, {src_s})\n"));
                }
                RegUnaryOp::Neg => {
                    out.push_str(&format!(
                        "{p}Tx.thread.mul({dst_s}, {src_s}, T.float32(-1))\n"
                    ));
                }
            }
            Ok(())
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

/// Local `NAME.ptr_to([slot])` for the mbar object named by a reference.
fn local_mbar_slot_ptr(
    mref: &super::mbar::MBarRef,
    stage: &Option<ScalarValue>,
    ctx: &Ctx,
) -> Result<String, String> {
    let name = ctx
        .mbar_names
        .get(&mref.mbar.id)
        .cloned()
        .ok_or_else(|| format!("codegen: no name for mbar {}", mref.mbar.id))?;
    let slot = stage
        .as_ref()
        .map(|s| emit_scalar(s, ctx))
        .transpose()?
        .unwrap_or_else(|| "0".to_string());
    Ok(format!("{name}.ptr_to([{slot}])"))
}

/// Pointer for the exact MBarRef carried by an operation. A local ref is the
/// local buffer pointer; a remote ref maps that same slot to its explicit
/// `remote_coord`. No mbar-id-based routing or cached peer identity exists.
fn mbar_slot_ptr(
    mref: &super::mbar::MBarRef,
    stage: &Option<ScalarValue>,
    ctx: &Ctx,
) -> Result<String, String> {
    let local = local_mbar_slot_ptr(mref, stage, ctx)?;
    match &mref.remote_coord {
        Some(remote) => Ok(format!(
            "T.reinterpret(\"handle\", T.ptx.map_shared_rank({local}, {coord}))",
            coord = emit_scalar(remote, ctx)?,
        )),
        None => Ok(local),
    }
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

// ===========================================================================
// tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::dtype::ScalarDType;
    use crate::ir::scalar::VarId;
    use crate::ir::stmt::{Tcgen05CpMulticast, Tcgen05CpShape};
    use crate::ir::tensor::Tensor;

    fn gmem_arg(id: u32) -> Arc<Tensor> {
        Arc::new(Tensor {
            id,
            space: MemorySpace::Gmem,
            dtype: DType::F32,
            shape: vec![16, 16],
            layout: None,
            byte_offset: None,
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
            smem_size_bytes: 1 << 20,
            launch_shape: vec![2],
            cluster_shape: vec![2],
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

    /// `if T.ptx.elect_sync(): body` — the `if_elected` sugar's own predicate.
    fn elected_if(body: Vec<Stmt>) -> Stmt {
        Stmt::If {
            cond: ScalarValue::Scope(ScopeValueKind::Elected),
            then_body: body,
        }
    }

    /// Scalar XOR is emitted directly as the TIR builtin.
    #[test]
    fn scalar_xor_is_direct_bitwise_xor() {
        let cond = ScalarValue::expr(
            ScalarOp::Xor,
            vec![ScalarValue::Int(1), ScalarValue::Int(2)],
        );
        let k = kernel(vec![Stmt::BreakIf { cond }]);
        let src = kernel_to_tirx_source(&k).unwrap();
        assert!(src.contains("if T.bitwise_xor(1, 2):"), "{src}");
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

    /// Top-level IR `If`s remain top-level siblings, in source order. Codegen
    /// must not synthesize parent guards, merge duplicate predicates, or turn
    /// sibling branches into an if/else tree.
    #[test]
    fn top_level_ifs_preserve_structure_and_order() {
        let body = vec![
            wg_if(1, vec![Stmt::SetMaxNReg { nreg: 232 }]),
            warp_if(4, vec![Stmt::WarpSync]),
            warp_if(5, vec![Stmt::WarpSync]),
            warp_if(4, vec![Stmt::WgSync { barrier_id: 7 }]),
        ];
        let src = kernel_to_tirx_source(&kernel_n(body, 8)).unwrap();
        let expected = "\
    if wg_id == 1:
        T.ptx.setmaxnreg(True, 232)
    if warp_id == 4:
        T.cuda.warp_sync()
    if warp_id == 5:
        T.cuda.warp_sync()
    if warp_id == 4:
        T.cuda.warpgroup_sync(7)
";
        assert!(src.contains(expected), "{src}");
        assert!(!src.contains("    else:"), "{src}");
        assert_eq!(src.matches("    if warp_id == 4:\n").count(), 2, "{src}");
    }

    /// Nesting is emitted only when nesting exists in the IR.
    #[test]
    fn nested_ifs_preserve_structure() {
        let body = vec![wg_if(1, vec![warp_if(5, vec![Stmt::WarpSync])])];
        let src = kernel_to_tirx_source(&kernel_n(body, 8)).unwrap();
        let expected = "\
    if wg_id == 1:
        if warp_id == 5:
            T.cuda.warp_sync()
";
        assert!(src.contains(expected), "{src}");
    }

    #[test]
    fn codegen_never_deletes_a_sync_from_its_ir_branch() {
        // This shape is rejected by validation because a CTA barrier cannot be
        // reached by one warp. The codegen-side contract is nevertheless
        // structural: if invoked directly, it prints the statement exactly
        // where the IR put it instead of silently suppressing or hoisting it.
        let body = vec![warp_if(1, vec![Stmt::CtaSync])];
        let src = kernel_to_tirx_source(&kernel_n(body, 8)).unwrap();
        let expected = "\
    if warp_id == 1:
        T.cuda.cta_sync()
";
        assert!(src.contains(expected), "{src}");
    }

    /// An elected predicate is still an ordinary IR `If`: its exact scalar
    /// expression and nesting are preserved for prologues and loops alike.
    #[test]
    fn elected_if_preserves_literal_predicate_and_nesting() {
        use super::super::dtype::MBarKind;
        use super::super::mbar::{MBar, MBarRef};
        let mbar = Arc::new(MBar {
            id: 3,
            kind: MBarKind::Thread,
            stages: 1,
            byte_offset: 800_024,
            arrive_count: None,
        });
        let init = || Stmt::MBarrierInit {
            mbar: MBarRef {
                mbar: mbar.clone(),
                remote_coord: None,
            },
            count: 1,
            stage: None,
        };
        // Prologue: warp-0 branch + elected branch remain nested literally.
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            warp_if(0, vec![elected_if(vec![init()])]),
        ]))
        .unwrap();
        assert!(
            src.contains("if warp_id == 0:\n        if T.ptx.elect_sync():"),
            "{src}"
        );
        assert!(!src.contains("if lane_id == 0:"), "{src}");
        assert!(!src.contains("if T.cuda.thread_rank() == 0:"), "{src}");

        // A non-zero warp uses the identical literal nested predicate.
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            warp_if(2, vec![elected_if(vec![init()])]),
        ]))
        .unwrap();
        assert!(
            src.contains("if warp_id == 2:\n        if T.ptx.elect_sync():"),
            "{src}"
        );
        assert!(!src.contains("if lane_id == 0:"), "{src}");
        assert!(!src.contains("if T.cuda.thread_rank() == 0:"), "{src}");

        // A HAND-WRITTEN `lane_id == 0` predicate stays a literal compare
        // (faithful translation: only the `if_elected` sugar is elect.sync).
        let lane0_if = Stmt::If {
            cond: ScalarValue::expr(
                ScalarOp::Eq,
                vec![
                    ScalarValue::Scope(ScopeValueKind::LaneId),
                    ScalarValue::Int(0),
                ],
            ),
            then_body: vec![init()],
        };
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            warp_if(0, vec![lane0_if]),
        ]))
        .unwrap();
        assert!(
            src.contains("if warp_id == 0:\n        if lane_id == 0:"),
            "{src}"
        );

        // A loop does not authorize a different spelling or structure.
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
        assert!(
            src.contains("if warp_id == 0:\n        if T.ptx.elect_sync():"),
            "{src}"
        );
        assert!(!src.contains("if lane_id == 0:"), "{src}");
        assert!(!src.contains("if T.cuda.thread_rank() == 0:"), "{src}");
    }

    /// A warp-collective `ClusterBarrierWait` under an elect-form `If` fails
    /// closed wherever it appears. Codegen must not hoist it out of the IR
    /// branch to make the program legal.
    #[test]
    fn cluster_barrier_wait_under_elect_is_not_hoisted() {
        for body in [
            vec![Stmt::ClusterBarrierWait, Stmt::WarpSync],
            vec![Stmt::WarpSync, Stmt::ClusterBarrierWait],
        ] {
            let nested = kernel(vec![warp_if(1, vec![elected_if(body)])]);
            let err = kernel_to_tirx_source(&nested).unwrap_err();
            assert!(err.contains("Put it explicitly at warp scope"), "{err}");
        }
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
            addr_byte_offset: 900_000,
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
        // Physical instruction operands declare their own non-owning views;
        // allocation alone must not invent a kernel-global TMEM buffer.
        assert!(!src.contains("tmem = T.decl_buffer"), "{src}");

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
        reg_frag_tensor_dtype_w(id, DType::F32, w)
    }

    fn reg_frag_tensor_dtype_w(id: u32, dtype: DType, w: usize) -> Arc<Tensor> {
        Arc::new(Tensor {
            id,
            space: MemorySpace::Reg,
            dtype,
            shape: vec![w],
            layout: None,
            byte_offset: None,
        })
    }

    fn tmem_addr(start_col: u32, row: i64, col: i64) -> TmemAddr {
        TmemAddr {
            tensor: super::super::tensor::TmemTensor { start_col },
            row: ScalarValue::Int(row),
            col: ScalarValue::Int(col),
        }
    }

    fn epilogue_kernel(ld: Stmt) -> Kernel {
        let tensor = match &ld {
            Stmt::Tcgen05Ld { dst, .. } => dst.tensor.clone(),
            Stmt::Tcgen05St { src, .. } => src.tensor.clone(),
            _ => reg_frag_tensor(7),
        };
        epilogue_kernel_t(ld, tensor)
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
    fn tcgen05_ld_preserves_physical_fields() {
        let f32 = reg_frag_tensor_dtype_w(7, DType::F32, 8);
        let ld = Stmt::Tcgen05Ld {
            dst: TensorSlice {
                tensor: f32.clone(),
                offsets: vec![ScalarValue::Int(2)],
                shape: vec![ScalarValue::Int(4)],
            },
            src: tmem_addr(37, 16, 9),
            shape: LdStShape::B16x64,
            num: 4,
        };
        let src = kernel_to_tirx_source(&epilogue_kernel_t(ld, f32)).unwrap();
        assert!(
            src.contains("accum_frag = T.alloc_local((8,), \"float32\")"),
            "{src}"
        );
        assert_eq!(src.matches("T.ptx.tcgen05.ld(").count(), 1, "{src}");
        assert!(src.contains("T.ptx.tcgen05.ld(T.uint32(37), "), "{src}");
        assert!(
            src.contains("shape=\"16x64b\", num=4, row=16, col=9, pack=False)"),
            "{src}"
        );
        assert!(!src.contains("Tx.wg.copy_async"), "{src}");

        // A packed b16 instruction still carries the same physical fields, and
        // addresses its b32 register tuple through the explicit u32 view.
        let bf16 = reg_frag_tensor_dtype_w(8, DType::Bf16, 12);
        let ld = Stmt::Tcgen05Ld {
            dst: TensorSlice {
                tensor: bf16.clone(),
                offsets: vec![ScalarValue::Int(2)],
                shape: vec![ScalarValue::Int(8)],
            },
            src: tmem_addr(211, 31, 17),
            shape: LdStShape::B16x128,
            num: 2,
        };
        let src = kernel_to_tirx_source(&epilogue_kernel_t(ld, bf16)).unwrap();
        assert!(
            src.contains("accum_frag = T.alloc_local((12,), \"bfloat16\")"),
            "{src}"
        );
        assert!(
            src.contains("accum_frag_u32 = accum_frag.view(\"uint32\")"),
            "{src}"
        );
        assert!(src.contains("T.ptx.tcgen05.ld(T.uint32(211), "), "{src}");
        assert!(
            src.contains("shape=\"16x128b\", num=2, row=31, col=17, pack=True)"),
            "{src}"
        );

        // The only rejected case here is a physically wrong register tuple
        // width; row, col, instruction shape, and 16-bit packing are all
        // representable and therefore emitted literally.
        let bad_num = Stmt::Tcgen05Ld {
            dst: TensorSlice {
                tensor: reg_frag_tensor_w(7, 8),
                offsets: vec![ScalarValue::Int(0)],
                shape: vec![ScalarValue::Int(8)],
            },
            src: tmem_addr(0, 0, 0),
            shape: LdStShape::B16x256,
            num: 8,
        };
        let err = kernel_to_tirx_source(&epilogue_kernel(bad_num)).unwrap_err();
        assert!(err.contains("num"), "{err}");
    }

    #[test]
    fn tcgen05_st_preserves_physical_fields() {
        let f32 = reg_frag_tensor_dtype_w(7, DType::F32, 12);
        let st = Stmt::Tcgen05St {
            dst: tmem_addr(73, 48, 19),
            src: TensorSlice {
                tensor: f32.clone(),
                offsets: vec![ScalarValue::Int(3)],
                shape: vec![ScalarValue::Int(8)],
            },
            shape: LdStShape::B16x256,
            num: 2,
        };
        let src = kernel_to_tirx_source(&epilogue_kernel_t(st, f32)).unwrap();
        assert!(
            src.contains("accum_frag = T.alloc_local((12,), \"float32\")"),
            "{src}"
        );
        assert_eq!(src.matches("T.ptx.tcgen05.st(").count(), 1, "{src}");
        assert!(src.contains("T.ptx.tcgen05.st(T.uint32(73), "), "{src}");
        assert!(
            src.contains("shape=\"16x256b\", num=2, row=48, col=19, unpack=False)"),
            "{src}"
        );
        assert!(!src.contains("Tx.wg.copy_async"), "{src}");

        let bad_width = Stmt::Tcgen05St {
            dst: tmem_addr(0, 0, 0),
            src: TensorSlice {
                tensor: reg_frag_tensor_w(7, 4),
                offsets: vec![ScalarValue::Int(0)],
                shape: vec![ScalarValue::Int(4)],
            },
            shape: LdStShape::B32x32,
            num: 8,
        };
        let err = kernel_to_tirx_source(&epilogue_kernel(bad_width)).unwrap_err();
        assert!(err.contains("num"), "{err}");
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
                    dst: tmem_addr(128, 7, 11),
                    src: TensorSlice {
                        tensor: bf,
                        offsets: vec![ScalarValue::Int(0)],
                        shape: vec![ScalarValue::Int(16)],
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
        assert!(
            src.contains("out_frag = T.alloc_local((16,), \"bfloat16\")"),
            "{src}"
        );
        assert!(
            src.contains("out_frag_u32 = out_frag.view(\"uint32\")"),
            "{src}"
        );
        assert!(src.contains("T.ptx.tcgen05.st(T.uint32(128), "), "{src}");
        assert!(
            src.contains("shape=\"32x32b\", num=8, row=7, col=11, unpack=False)"),
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
            byte_offset: Some(0),
        })
    }

    #[test]
    fn dense_tcgen05_mma_is_one_explicit_gemm_async() {
        let a = smem_tensor(10, DType::F16, &[64, 64]);
        let b = smem_tensor(11, DType::F16, &[64, 64]);
        let tile = |tensor: Arc<Tensor>| SmemTile {
            tensor,
            prefix_indices: vec![],
            row_offset: ScalarValue::Int(0),
            col_offset: ScalarValue::Int(0),
            rows: 64,
            cols: 64,
        };
        let mma = Stmt::Tcgen05Mma {
            dst: tmem_addr(32, 0, 0),
            a: MmaAOperand::Smem(tile(a.clone())),
            b: tile(b.clone()),
            mma_m: 128,
            mma_n: 64,
            format: MmaElemFormat::F16,
            block_scale: None,
            accum: ScalarValue::Int(1),
            trans_a: false,
            trans_b: false,
            ws: false,
            cta_group: 2,
        };
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::TensorDef { tensor: a },
            Stmt::TensorDef { tensor: b },
            warp_if(0, vec![tmem_alloc(0, 512, 2)]),
            warp_if(1, vec![elected_if(vec![mma])]),
        ]))
        .unwrap();

        assert_eq!(src.matches("Tx.gemm_async(").count(), 1, "{src}");
        assert!(
            src.contains(
                "dispatch=\"tcgen05\", cta_group=2, mma_m=128, mma_n=64, \
                 weight_stationary=False)"
            ),
            "{src}"
        );
        assert!(
            src.contains(
                "mma_d0 = T.decl_buffer((64, 128), \"float32\", scope=\"tmem\", \
                 allocated_addr=32"
            ),
            "{src}"
        );
        assert_eq!(src.matches("T.decl_buffer(").count(), 1, "{src}");
        assert!(!src.contains("\n    tmem = T.decl_buffer"), "{src}");
        assert!(
            src.find("mma_d0 = T.decl_buffer").unwrap() < src.find("Tx.gemm_async(").unwrap(),
            "{src}"
        );
    }

    #[test]
    fn tcgen05_cp_is_one_explicit_copy_async() {
        let sf = smem_tensor(10, DType::F8E4M3, &[32, 16]);
        let cp = Stmt::Tcgen05Cp {
            dst: tmem_addr(200, 0, 0),
            src: SmemTile {
                tensor: sf.clone(),
                prefix_indices: vec![],
                row_offset: ScalarValue::Int(0),
                col_offset: ScalarValue::Int(0),
                rows: 32,
                cols: 16,
            },
            shape: Tcgen05CpShape::B32x128,
            multicast: Tcgen05CpMulticast::Warp4,
            cta_group: 2,
        };
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::TensorDef { tensor: sf },
            warp_if(0, vec![tmem_alloc(0, 512, 2)]),
            warp_if(1, vec![elected_if(vec![cp])]),
        ]))
        .unwrap();

        assert_eq!(src.matches("Tx.copy_async(").count(), 1, "{src}");
        assert!(
            src.contains("shape=\"32x128b\", multicast=\"warpx4\", cta_group=2)"),
            "{src}"
        );
        assert!(
            src.contains(
                "cp_dst0 = T.decl_buffer((32, 16), \"float8_e4m3fn\", scope=\"tmem\", \
                 allocated_addr=200"
            ),
            "{src}"
        );
        assert!(
            src.contains(
                "cp_src0 = T.decl_buffer((32, 16), \"float8_e4m3fn\", \
                 data=d_smem0.data, elem_offset="
            ),
            "{src}"
        );
        assert!(
            src.contains("scope=\"shared.dyn\", layout=TileLayout(S[(32, 16) : (16, 1)]))"),
            "{src}"
        );
        assert!(
            src.contains("Tx.copy_async(cp_dst0[:, :], cp_src0[0:32, 0:16]"),
            "{src}"
        );
        assert_eq!(src.matches("T.decl_buffer(").count(), 2, "{src}");
        assert!(!src.contains("\n    tmem = T.decl_buffer"), "{src}");
        assert!(
            src.find("cp_dst0 = T.decl_buffer").unwrap() < src.find("Tx.copy_async(").unwrap(),
            "{src}"
        );
    }

    #[test]
    fn two_explicit_u32_swizzles_are_unique_and_not_flat_mailboxes() {
        let tensor0 = Arc::new(Tensor {
            id: 10,
            space: MemorySpace::Smem,
            dtype: DType::U32,
            shape: vec![128, 8],
            layout: Some(Layout::Swizzle(super::super::SmemSwizzleLayout {
                swizzle: Swizzle::B32,
            })),
            byte_offset: Some(0),
        });
        let tensor1 = Arc::new(Tensor {
            id: 11,
            space: MemorySpace::Smem,
            dtype: DType::U32,
            shape: vec![128, 8],
            layout: Some(Layout::Swizzle(super::super::SmemSwizzleLayout {
                swizzle: Swizzle::B32,
            })),
            byte_offset: Some(4096),
        });
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::TensorDef { tensor: tensor0 },
            Stmt::TensorDef { tensor: tensor1 },
        ]))
        .unwrap();
        assert!(
            src.contains(
                "ab_smem0 = pool.alloc_tcgen05_mma_AB((128, 8), \"uint32\", \
                 swizzle_mode=SwizzleMode.SWIZZLE_32B_ATOM, align=1024)"
            ),
            "{src}"
        );
        assert!(
            src.contains(
                "ab_smem1 = pool.alloc_tcgen05_mma_AB((128, 8), \"uint32\", \
                 swizzle_mode=SwizzleMode.SWIZZLE_32B_ATOM, align=1024)"
            ),
            "{src}"
        );
        assert!(!src.contains("mma_shared_layout"), "{src}");
        assert!(!src.contains("task_smem"), "{src}");
    }

    #[test]
    fn multiple_layoutless_integer_smem_views_have_unique_names() {
        let tensor0 = Arc::new(Tensor {
            id: 10,
            space: MemorySpace::Smem,
            dtype: DType::U32,
            shape: vec![5, 128],
            layout: None,
            byte_offset: Some(0),
        });
        let tensor1 = Arc::new(Tensor {
            id: 11,
            space: MemorySpace::Smem,
            dtype: DType::U32,
            shape: vec![5, 128],
            layout: None,
            byte_offset: Some(2560),
        });
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::TensorDef { tensor: tensor0 },
            Stmt::TensorDef { tensor: tensor1 },
        ]))
        .unwrap();
        assert!(
            src.contains("task_smem = pool.alloc((5, 128), \"uint32\", scope=\"shared.dyn\")"),
            "{src}"
        );
        assert!(
            src.contains("task_smem1 = pool.alloc((5, 128), \"uint32\", scope=\"shared.dyn\")"),
            "{src}"
        );
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
                "T.ptx.ldmatrix(False, 1, \".b16\", d_smem0.ptr_to([8 + ((lane_id) & 7), 16]), accum_frag.ptr_to([0]))"
            ),
            "{src}"
        );
        assert!(
            src.contains(
                "T.ptx.stmatrix(True, 1, \".b16\", d_smem0.ptr_to([8 + ((lane_id) & 7), 32]), out_frag_u32.ptr_to([0]))"
            ),
            "{src}"
        );

        // Explicit stage/tile axes remain part of the physical pointer. Only
        // the final eight-element row is consumed by the instruction.
        let staged = smem_tensor(14, DType::Bf16, &[3, 16, 128]);
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::TensorDef {
                tensor: staged.clone(),
            },
            Stmt::TensorDef {
                tensor: tile.clone(),
            },
            wg_if(
                0,
                vec![Stmt::StMatrix {
                    dst: TensorSlice {
                        tensor: staged,
                        offsets: vec![ScalarValue::Int(2), lane_row.clone(), ScalarValue::Int(32)],
                        shape: vec![
                            ScalarValue::Int(1),
                            ScalarValue::Int(1),
                            ScalarValue::Int(8),
                        ],
                    },
                    src: reg_slice(&tile, 0, 2),
                    shape: MatrixShape::M8N8,
                    num: 1,
                    trans: true,
                    dtype: MatrixDType::B16,
                }],
            ),
        ]))
        .unwrap();
        assert!(
            src.contains("ptr_to([2, 8 + ((lane_id) & 7), 32])"),
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
                "T.ptx.mma(\"m16n8k8\", \"row\", \"col\", \"float32\", \"bfloat16\", \
                 \"bfloat16\", \"float32\", [reg2.ptr_to([0]), reg2.ptr_to([1]), \
                 reg2.ptr_to([2]), reg2.ptr_to([3])], [accum_frag.ptr_to([0]), \
                 accum_frag.ptr_to([1])], [out_frag.ptr_to([0])], \
                 [reg2.ptr_to([0]), reg2.ptr_to([1]), reg2.ptr_to([2]), \
                 reg2.ptr_to([3])])"
            ),
            "{src}"
        );
        assert!(!src.contains("mma.legacy"), "{src}");

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
            byte_offset: 800_024,
            arrive_count: None,
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

    /// The per-thread emission INVARIANT, pinned for the whole per-thread op
    /// family: expect_tx / arrive_expect_tx / store_scalar / async-proxy fence
    /// at warpgroup scope emit BARE (one application per executing lane,
    /// matching the interpreter) — any guard codegen might synthesize would
    /// undercount arrivals/tx bytes or drop lane-varying stores.
    #[test]
    fn per_thread_ops_emit_bare_at_warpgroup_scope() {
        use super::super::dtype::{FenceKind, FenceScope, MBarKind};
        use super::super::mbar::{MBar, MBarRef};
        let mbar = Arc::new(MBar {
            id: 4,
            kind: MBarKind::Tma,
            stages: 1,
            byte_offset: 800_032,
            arrive_count: None,
        });
        let mref = || MBarRef {
            mbar: mbar.clone(),
            remote_coord: None,
        };
        let g = Arc::new(Tensor {
            id: 0,
            space: MemorySpace::Gmem,
            dtype: DType::I32,
            shape: vec![4],
            layout: None,
            byte_offset: None,
        });
        let point = |v: i64| TensorSlice {
            tensor: g.clone(),
            offsets: vec![ScalarValue::Int(v)],
            shape: vec![ScalarValue::Int(1)],
        };
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            wg_if(
                0,
                vec![
                    Stmt::MBarrierExpectTx {
                        mbar: mref(),
                        bytes: 64,
                        stage: None,
                    },
                    Stmt::MBarrierArriveExpectTx {
                        mbar: mref(),
                        bytes: 64,
                        stage: None,
                    },
                    Stmt::StoreScalar {
                        dst: point(0),
                        value: ScalarValue::Int(7),
                    },
                    Stmt::Fence {
                        kind: FenceKind::AsyncProxy,
                        scope: FenceScope::Cta,
                    },
                ],
            ),
        ]))
        .unwrap();
        for line in [
            "T.ptx.mbarrier.expect_tx(smem_full.ptr_to([0]), 64)",
            "T.ptx.mbarrier.arrive.expect_tx(smem_full.ptr_to([0]), 64)",
            "T.ptx.fence.proxy_async(\"shared::cta\")",
        ] {
            assert!(src.contains(line), "{line} missing from {src}");
        }
        // no synthesized guard anywhere around them
        assert!(!src.contains("if T.ptx.elect_sync():"), "{src}");
        assert!(!src.contains("if tid_in_wg == 0:"), "{src}");
        assert!(!src.contains("if T.cuda.thread_rank() == 0:"), "{src}");
    }

    /// Each MBarRef is its own complete address. Codegen must preserve both
    /// remote coordinates literally, without mbar-id routing or byte scaling.
    #[test]
    fn mbar_refs_lower_remote_coords_without_inference() {
        use super::super::dtype::MBarKind;
        use super::super::mbar::{MBar, MBarRef};
        let mbar = Arc::new(MBar {
            id: 4,
            kind: MBarKind::Tma,
            stages: 1,
            byte_offset: 800_032,
            arrive_count: None,
        });
        let src = kernel_to_tirx_source(&kernel(vec![
            Stmt::MBarDef { mbar: mbar.clone() },
            wg_if(
                0,
                vec![elected_if(vec![
                    Stmt::MBarrierArriveExpectTx {
                        mbar: MBarRef {
                            mbar: mbar.clone(),
                            remote_coord: Some(ScalarValue::Int(0)),
                        },
                        bytes: 64,
                        stage: None,
                    },
                    Stmt::MBarrierWait {
                        mbar: MBarRef {
                            mbar,
                            remote_coord: Some(ScalarValue::Int(1)),
                        },
                        phase: Some(ScalarValue::Int(0)),
                        stage: None,
                    },
                ])],
            ),
        ]))
        .unwrap();

        assert!(
            src.contains(
                "T.ptx.mbarrier.arrive.expect_tx(smem_full.ptr_to([0]), 64, remote=0, pred=True)"
            ),
            "{src}"
        );
        assert!(
            src.contains(
                "T.ptx.mbarrier.try_wait(T.reinterpret(\"handle\", \
                 T.ptx.map_shared_rank(smem_full.ptr_to([0]), 1)), 0)"
            ),
            "{src}"
        );
        assert!(!src.contains("if cbx == 0:"), "{src}");
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
            src.contains("accum_frag = T.alloc_local((1,), \"float32\")"),
            "{src}"
        );
        assert!(
            src.contains("out_frag = T.alloc_local((4,), \"float32\")"),
            "{src}"
        );
        assert!(
            src.contains("Tx.copy(accum_frag[:], d_smem0[tid_in_wg:tid_in_wg + 1, 3:4])"),
            "{src}"
        );
        assert!(
            src.contains("Tx.copy(d_smem0[tid_in_wg:tid_in_wg + 1, 5:6], accum_frag[:])"),
            "{src}"
        );
        assert!(
            src.contains("Tx.copy(out_frag[1:1 + 1], out_frag[0:0 + 1])"),
            "{src}"
        );
        assert!(
            src.contains("Tx.copy(A[1:2, 2:3, tid_in_wg:tid_in_wg + 1, 16:20], out_frag[:])"),
            "{src}"
        );
        assert!(!src.contains("Tx.wg."), "{src}");
    }

    fn reg_tensor(id: u32, dtype: DType, width: i64) -> Arc<Tensor> {
        Arc::new(Tensor {
            id,
            space: MemorySpace::Reg,
            dtype,
            shape: vec![width as usize],
            layout: None,
            byte_offset: None,
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
            src.contains("accum_frag = T.alloc_local((1,), \"float32\")"),
            "{src}"
        );
        assert!(
            src.contains("out_frag = T.alloc_local((1,), \"float32\")"),
            "{src}"
        );
        assert!(
            src.contains("reg2 = T.alloc_local((2,), \"bfloat16\")"),
            "{src}"
        );
        assert!(
            src.contains("Tx.thread.fill(accum_frag[:], T.float32(0))"),
            "{src}"
        );
        assert!(
            src.contains("Tx.thread.add(accum_frag[:], accum_frag[:], out_frag[:])"),
            "{src}"
        );
        assert!(
            src.contains("Tx.thread.sub(accum_frag[:], accum_frag[:], T.float32(-1))"),
            "{src}"
        );
        assert!(
            src.contains("Tx.thread.fma(accum_frag[:], accum_frag[:], out_frag[:], accum_frag[:])"),
            "{src}"
        );
        assert!(
            src.contains("Tx.thread.exp2(accum_frag[:], out_frag[:])"),
            "{src}"
        );
        assert!(
            src.contains("Tx.thread.reciprocal(accum_frag[:], out_frag[:])"),
            "{src}"
        );
        assert!(
            src.contains("Tx.thread.mul(accum_frag[:], out_frag[:], T.float32(-1))"),
            "{src}"
        );
        assert!(
            src.contains("Tx.thread.log2(accum_frag[:], out_frag[:])"),
            "{src}"
        );
        assert!(
            src.contains("Tx.thread.fill(reg2[:], T.bfloat16(1))"),
            "{src}"
        );
        assert!(
            src.contains("Tx.thread.add(reg2[:], reg2[:], reg2[:])"),
            "{src}"
        );
        assert!(!src.contains("Tx.wg."), "{src}");
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

    /// Codegen preserves a CTA sync even in an invalid nested branch; the
    /// validator, not codegen, is responsible for rejecting that program.
    #[test]
    fn cta_sync_is_never_deleted_or_hoisted() {
        let src = kernel_to_tirx_source(&kernel(vec![Stmt::CtaSync])).unwrap();
        assert!(src.contains("T.cuda.cta_sync()"), "{src}");
        let src = kernel_to_tirx_source(&kernel(vec![warp_if(1, vec![Stmt::CtaSync])])).unwrap();
        assert!(
            src.contains("if warp_id == 1:\n        T.cuda.cta_sync()"),
            "{src}"
        );
    }
}
