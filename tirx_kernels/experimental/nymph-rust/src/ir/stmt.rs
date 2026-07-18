//! Statements. Python has a base `Stmt` class with ~41 subclasses, dispatched by
//! `isinstance`. In Rust that's ONE enum with ~41 variants, dispatched by `match`
//! (which the compiler forces you to handle exhaustively — a safety win).
//!
//! Body-bearing control nodes hold `Vec<Stmt>` (a recursive enum; `Vec` heap-
//! allocates so the type has a finite size).

use super::dtype::{DType, FenceKind, FenceScope};
use super::mbar::{MBar, MBarRef};
use super::scalar::{ScalarInitial, ScalarValue, Var};
use super::scheduler::Scheduler;
use super::tensor::{MmaOperand, Tensor, TensorSlice, TmemOperand};
use std::sync::Arc;

/// RegCvt rounding mode (Python `Literal["rn"]` — only RN exists).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rounding {
    Rn,
    Rm,
}

impl Rounding {
    pub fn as_str(self) -> &'static str {
        match self {
            Rounding::Rn => "rn",
            Rounding::Rm => "rm",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "rn" => Some(Rounding::Rn),
            "rm" => Some(Rounding::Rm),
            _ => None,
        }
    }
}

/// Literal REG operand. Float literals are stored as raw f32 bits so the IR can
/// keep `Eq`/`Hash`-friendly structural identity without depending on float Eq.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegLiteral {
    Int(i64),
    F32Bits(u32),
}

impl RegLiteral {
    pub fn f32(value: f32) -> Self {
        Self::F32Bits(value.to_bits())
    }

    pub fn as_f32(self) -> f32 {
        match self {
            RegLiteral::Int(v) => v as f32,
            RegLiteral::F32Bits(bits) => f32::from_bits(bits),
        }
    }

    pub fn as_i64(self) -> i64 {
        match self {
            RegLiteral::Int(v) => v,
            RegLiteral::F32Bits(bits) => f32::from_bits(bits) as i64,
        }
    }
}

/// Operand for REG value ops: a per-thread register slice or a broadcast literal.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RegOperand {
    Slice(TensorSlice),
    Literal(RegLiteral),
}

impl RegOperand {
    pub fn as_slice(&self) -> Option<&TensorSlice> {
        match self {
            RegOperand::Slice(s) => Some(s),
            RegOperand::Literal(_) => None,
        }
    }
}

impl From<TensorSlice> for RegOperand {
    fn from(value: TensorSlice) -> Self {
        RegOperand::Slice(value)
    }
}

/// Generic unary REG op.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegUnaryOp {
    Exp2,
    Log2,
    Rcp,
    Neg,
}

impl RegUnaryOp {
    pub fn as_str(self) -> &'static str {
        match self {
            RegUnaryOp::Exp2 => "exp2",
            RegUnaryOp::Log2 => "log2",
            RegUnaryOp::Rcp => "rcp",
            RegUnaryOp::Neg => "neg",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "exp2" => Some(RegUnaryOp::Exp2),
            "log2" => Some(RegUnaryOp::Log2),
            "rcp" => Some(RegUnaryOp::Rcp),
            "neg" => Some(RegUnaryOp::Neg),
            _ => None,
        }
    }
}

/// Generic binary REG op. Existing RegAdd/Sub/... constructors map here through
/// dedicated statement variants for backward-compatible structure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegBinaryOp {
    Add,
    Sub,
    Mul,
    Max,
    Min,
    And,
    Shl,
}

impl RegBinaryOp {
    pub fn as_str(self) -> &'static str {
        match self {
            RegBinaryOp::Add => "add",
            RegBinaryOp::Sub => "sub",
            RegBinaryOp::Mul => "mul",
            RegBinaryOp::Max => "max",
            RegBinaryOp::Min => "min",
            RegBinaryOp::And => "and",
            RegBinaryOp::Shl => "shl",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "add" => Some(RegBinaryOp::Add),
            "sub" => Some(RegBinaryOp::Sub),
            "mul" => Some(RegBinaryOp::Mul),
            "max" => Some(RegBinaryOp::Max),
            "min" => Some(RegBinaryOp::Min),
            "and" => Some(RegBinaryOp::And),
            "shl" => Some(RegBinaryOp::Shl),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegReduceOp {
    Max,
    Sum,
}

impl RegReduceOp {
    pub fn as_str(self) -> &'static str {
        match self {
            RegReduceOp::Max => "max",
            RegReduceOp::Sum => "sum",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "max" => Some(RegReduceOp::Max),
            "sum" => Some(RegReduceOp::Sum),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegCondScope {
    Warp,
    Warpgroup,
}

impl RegCondScope {
    pub fn as_str(self) -> &'static str {
        match self {
            RegCondScope::Warp => "warp",
            RegCondScope::Warpgroup => "warpgroup",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "warp" => Some(RegCondScope::Warp),
            "warpgroup" => Some(RegCondScope::Warpgroup),
            _ => None,
        }
    }
}

/// tcgen05 ld/st datapath shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LdStShape {
    B32x32,
    B16x32Bx2,
    B16x64,
    B16x128,
    B16x256,
}

impl LdStShape {
    pub fn as_str(self) -> &'static str {
        match self {
            LdStShape::B32x32 => "32x32b",
            LdStShape::B16x32Bx2 => "16x32bx2",
            LdStShape::B16x64 => "16x64b",
            LdStShape::B16x128 => "16x128b",
            LdStShape::B16x256 => "16x256b",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "32x32b" => Some(LdStShape::B32x32),
            "16x32bx2" => Some(LdStShape::B16x32Bx2),
            "16x64b" => Some(LdStShape::B16x64),
            "16x128b" => Some(LdStShape::B16x128),
            "16x256b" => Some(LdStShape::B16x256),
            _ => None,
        }
    }

    pub fn register_count(self, num: u32) -> Option<usize> {
        match (self, num) {
            (LdStShape::B32x32, 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128) => Some(num as usize),
            (LdStShape::B16x32Bx2 | LdStShape::B16x64, 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128) => {
                Some(num as usize)
            }
            (LdStShape::B16x128, 1 | 2 | 4 | 8 | 16 | 32 | 64) => Some(2 * num as usize),
            (LdStShape::B16x256, 1 | 2 | 4 | 8 | 16 | 32) => Some(4 * num as usize),
            _ => None,
        }
    }
}

/// PTX ldmatrix/stmatrix matrix shape.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MatrixShape {
    M8N8,
}

impl MatrixShape {
    pub fn as_str(self) -> &'static str {
        match self {
            MatrixShape::M8N8 => "m8n8",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "m8n8" => Some(MatrixShape::M8N8),
            _ => None,
        }
    }
}

/// PTX ldmatrix/stmatrix element type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MatrixDType {
    B16,
}

impl MatrixDType {
    pub fn as_str(self) -> &'static str {
        match self {
            MatrixDType::B16 => "b16",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "b16" => Some(MatrixDType::B16),
            _ => None,
        }
    }
}

/// Memory order for a GMEM semaphore atomic-add (`red.<order>.gpu.global.add`).
/// `Release` carries the release fence that publishes prior writes (e.g. a drained
/// reduce-add) before the counter bump; `Relaxed` does not order memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GmemAtomicOrder {
    Release,
    Relaxed,
}

impl GmemAtomicOrder {
    pub fn as_str(self) -> &'static str {
        match self {
            GmemAtomicOrder::Release => "release",
            GmemAtomicOrder::Relaxed => "relaxed",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "release" => Some(GmemAtomicOrder::Release),
            "relaxed" => Some(GmemAtomicOrder::Relaxed),
            _ => None,
        }
    }
}

/// `Stmt` — one statement of the kernel body. `cta_group` fields are 1 or 2;
/// `*_mask` are 16-bit CTA masks.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Stmt {
    // ---- definitions / allocation ----
    TensorDef {
        tensor: Arc<Tensor>,
    },
    /// `tcgen05.alloc` — allocate the TMEM column band `[base_col, base_col +
    /// n_cols)`. TMEM is not a tensor: the allocation only declares the column
    /// interval; every TMEM instruction addresses cells by absolute physical
    /// (lane, col) via `TmemOperand`.
    TmemAlloc {
        base_col: u32,
        n_cols: u32,
        cta_group: u8,
    },
    /// `tcgen05.dealloc` — release the column band `[base_col, base_col +
    /// n_cols)` (must match a live allocation).
    TmemDealloc {
        base_col: u32,
        n_cols: u32,
        cta_group: u8,
    },
    /// `tcgen05.relinquish_alloc_permit` — give up the right to issue further
    /// `tcgen05.alloc`s. Explicit in the IR (codegen translates 1:1); it used
    /// to be emitted implicitly with the dealloc, which was not a faithful
    /// translation.
    TmemRelinquish {
        cta_group: u8,
    },
    ScalarDef {
        var: Var,
        initial: ScalarInitial,
    },
    ScalarStore {
        var: Var,
        value: ScalarValue,
    },
    /// Single-assignment `let` binding (canon's `name: T.let = expr` — an immutable
    /// SSA `T.Bind`, NOT a `local_scalar` cell). Every execution of the statement
    /// binds `var` once; validate REJECTS any `ScalarStore` to it. Codegen emits the
    /// `T.let` form so ptxas sees a pure SSA dataflow (mutable locals defeat its
    /// uniform-register analysis — the R2UR problem; see docs/perf-methodology.md).
    ScalarLet {
        var: Var,
        value: ScalarValue,
    },
    /// Warp shuffle/broadcast that DEFINES a scalar: `var` gets the value of `src`
    /// evaluated on lane `src_lane` of each warp, broadcast to all lanes (a faithful
    /// `__shfl_sync`). First-class so the value-sim verifies it — broadcasting a value
    /// that ISN'T warp-uniform changes it, which surfaces as a value mismatch. Used to
    /// promote warp-uniform SMEM reads (the scheduler mailbox) to a form the CUDA
    /// compiler proves uniform → the index/address chain lowers to the uniform datapath.
    ShuffleSync {
        var: Var,
        src: ScalarValue,
        src_lane: ScalarValue,
    },
    StoreScalar {
        dst: TensorSlice,
        value: ScalarValue,
    },
    MBarDef {
        mbar: Arc<MBar>,
    },

    // ---- structural / control flow (bodies recurse) ----
    KernelInit {
        body: Vec<Stmt>,
        warp: Option<u32>,
        lane: Option<u32>,
        elected: bool,
    },
    KernelFinalize {
        body: Vec<Stmt>,
        warp: Option<u32>,
        lane: Option<u32>,
        elected: bool,
    },
    Role {
        body: Vec<Stmt>,
        warp: Option<u32>,
        warpgroup: Option<u32>,
        elected: bool,
        maxnreg: Option<u32>,
    },
    ForLoop {
        var: Var,
        start: ScalarValue,
        stop: ScalarValue,
        step: ScalarValue,
        body: Vec<Stmt>,
        /// Emit as `T.unroll(N)` instead of `T.serial(N)` — a compile-time-unrolled loop
        /// (the TVMScript parser substitutes a constant loop var per iteration, so it keeps
        /// the same SASS as a manual unroll) but written as a `for` in the source, matching
        /// canon's `for i in T.unroll(N)` for fixed-count loops (mbarrier inits, etc.).
        unroll: bool,
    },
    ForEachTask {
        scheduler: Arc<Scheduler>,
        var: Var,
        body: Vec<Stmt>,
    },
    SchedulerImpl {
        scheduler: Arc<Scheduler>,
        body: Vec<Stmt>,
    },
    SchedNext {
        scheduler: Arc<Scheduler>,
        var: Var,
    },
    /// CLC (Cluster Launch Control) async work-steal issue: hardware
    /// `clusterlaunchcontrol.try_cancel` writes a 16B response into `handle` and
    /// completes-tx `mbar` (multicast to both cluster CTAs). Written out EXPLICITLY
    /// in the kernel's `policy="custom"` scheduler — codegen translates it 1:1 and
    /// never synthesizes it. In sim it (1) runs the canonical round-robin oracle —
    /// the trusted seam, §7 of the scheduler RFC — and stores the resulting work id
    /// into a per-cluster handle slot, and (2) completes-tx the signalled mbar (like
    /// a TMA landing) so the handshake the checker validates is real. The paired
    /// `ClcQueryCancel` reads the slot, so the scheduler and every worker that query
    /// the same handle observe the same id.
    ClcTryCancel {
        scheduler: Arc<Scheduler>,
        handle: Arc<Tensor>,
        mbar: MBarRef,
        stage: Option<ScalarValue>,
        cta_group: u8,
    },
    /// CLC decode of the response `handle`: DEFINES `var` by reading the per-cluster
    /// handle slot the paired `ClcTryCancel` filled — the cancelled cluster's first
    /// `ctaid.x` (= task * cta_group), or `0xFFFFFFFF` (→ -1 as int32) when drained.
    /// Codegen translates 1:1 to `T.ptx.clc_query_cancel`; `scheduler` is sim-only
    /// metadata (keys the slot read).
    ClcQueryCancel {
        scheduler: Arc<Scheduler>,
        var: Var,
        handle: Arc<Tensor>,
    },
    Loop {
        body: Vec<Stmt>,
    },
    BreakIf {
        cond: ScalarValue,
    },
    If {
        cond: ScalarValue,
        then_body: Vec<Stmt>,
    },

    // ---- mbarrier ----
    MBarrierInit {
        mbar: MBarRef,
        count: u32,
        stage: Option<ScalarValue>,
    },
    MBarrierArrive {
        mbar: MBarRef,
        stage: Option<ScalarValue>,
        count: ScalarValue,
    },
    MBarrierWait {
        mbar: MBarRef,
        stage: Option<ScalarValue>,
        phase: Option<ScalarValue>,
    },
    MBarrierExpectTx {
        mbar: MBarRef,
        bytes: u32,
        stage: Option<ScalarValue>,
    },
    MBarrierArriveExpectTx {
        mbar: MBarRef,
        bytes: u32,
        stage: Option<ScalarValue>,
    },

    // ---- TMA (bulk async GMEM<->SMEM) ----
    TmaLoad {
        dst: TensorSlice,
        src: Arc<Tensor>,
        mbar: MBarRef,
        bytes: ScalarValue,
        coords: Vec<ScalarValue>,
        shape: Vec<usize>,
        gmem_shape: Option<Vec<usize>>,
        mbar_stage: Option<ScalarValue>,
        /// `multicast::cluster` cta_mask: one TMA fills the SMEM of every masked CTA of
        /// the cluster (canon's shared-SF-band multicast), so the cluster shares one load
        /// instead of each CTA reading the band — halving the L2/TMA load traffic. `None`
        /// = a plain per-CTA (unicast) load.
        multicast_cta_mask: Option<u16>,
        /// L2 cache-eviction policy hint (canon's `cache_hint` on its g2c loads). `None`
        /// = no hint; `Some(hint)` emits `cache_hint="<hint>"` (e.g. `"evict_normal"` —
        /// a tile read once per k-tile should not pin an L2 line the next tile evicts
        /// anyway). Bounding the L2 cache-policy traffic is the lever that stops the
        /// full-cube launch fault. Codegen passes the string through.
        cache_hint: Option<String>,
        /// Prefetch the source tensormap at kernel entry (canon's config flag on its
        /// g2c loads — hides the first descriptor fetch behind the prologue). A pure
        /// HW hint with no value/protocol semantics, but IR-carried so the same IR
        /// always produces the same code shape.
        prefetch_tensormap: bool,
        cta_group: u8,
    },
    TmaStore {
        dst: Arc<Tensor>,
        src: TensorSlice,
        coords: Vec<ScalarValue>,
        shape: Vec<usize>,
        gmem_shape: Option<Vec<usize>>,
        /// True for `cp.reduce.async.bulk...add.f32` (TMA reduce-add): value-mode
        /// accumulates `dst += src` instead of overwriting. Trace/protocol treat it
        /// like a store (a GMEM-output bulk async write).
        reduce_add: bool,
        /// Explicit opt-in for a NON-INTEGER (today: f32) `reduce_add`. A float
        /// reduction is not associative, so cross-CTA reduce-adds to one location are
        /// race-free (hardware-atomic, commutative) but ORDER-DEPENDENT — the result is
        /// not bit-reproducible. The protocol checker can only WARN
        /// (`nondeterministic_reduction`), and warnings are easy to miss, so validate
        /// REJECTS a float reduce-add unless the kernel author sets this flag. With the
        /// flag set, the checker keeps its warning. An integer reduce-add (exact,
        /// associative) would not need it; validate currently restricts reduce_add to
        /// f32 dst anyway.
        allow_nondet_reduce: bool,
        /// L2 eviction policy for the store (canon's epilogue store carries
        /// `"evict_first"`: the output band is write-once, never re-read — dead lines
        /// must not pack L2 and evict live operand tiles / tensormaps). `None` = no hint.
        cache_hint: Option<String>,
        /// Prefetch the destination tensormap, matching the canonical epilogue store.
        prefetch_tensormap: bool,
    },
    /// `cp.async.bulk.shared::cluster.shared::cta` — async bulk copy from this CTA's
    /// SMEM (`src`) to a PEER CTA's SMEM (`dst`, the peer instance), signalling the
    /// peer's `mbar` (via its `remote_coord`) on completion. The peer CTA is the
    /// mbar's target. Trace/protocol model it as a local-SMEM async-proxy READ +
    /// a peer-CTA-SMEM async-proxy WRITE (attributed to the peer's SMEM pool, so the
    /// race checker matches it against the peer's read) + a `complete_tx` on the
    /// PEER's mbar (so the cross-CTA happens-before closes through the peer's wait).
    CpAsyncBulkS2Cluster {
        dst: TensorSlice,
        src: TensorSlice,
        mbar: MBarRef,
        bytes: ScalarValue,
    },
    /// `red.<order>.gpu.global.add.s32` — a GMEM semaphore atomic-add ("signal").
    /// VALUE: serialized RMW of the i32 semaphore cell `sem[coords]` (`+= value`).
    /// TRACE/protocol: a SYNC op (NOT a data access — no Read/Write on the
    /// semaphore tensor); it publishes this stream's clock as the RELEASE clock for
    /// the semaphore slot at its POST-increment value (value-keyed), so a later
    /// `wait_eq` on that exact value joins it (acquire). `order=release` carries the
    /// release fence ordering all prior writes (incl. drained reduce-adds) before
    /// the publish.
    GmemAtomicAdd {
        sem: TensorSlice,
        coords: Vec<ScalarValue>,
        value: ScalarValue,
        order: GmemAtomicOrder,
    },
    /// `ld.global.acquire.gpu` spin-loop until `sem[coords] == value` ("wait").
    /// VALUE: BLOCK this stream (polled re-check) until the i32 cell equals `value`;
    /// never reaching it -> the runner's deadlock detection fires. TRACE/protocol: a
    /// SYNC op (no Read/Write) that ACQUIRES — joins the release clock published by
    /// the `atomic_add` that PRODUCED this exact `value` (value-keyed, mirroring the
    /// mbar phase-keyed pattern with the counter-value in place of parity).
    GmemWaitEq {
        sem: TensorSlice,
        coords: Vec<ScalarValue>,
        value: ScalarValue,
    },
    CpAsyncBulkCommitGroup,
    CpAsyncBulkWaitGroupRead {
        n: u8, // always 0
    },

    // ---- tcgen05 (tensor core + TMEM) ----
    Tcgen05Mma {
        /// Accumulator destination: absolute physical TMEM address (lane, col)
        /// + f32 cell interpretation. The accumulator's (rows, n) footprint is
        /// implied by `m`/`n`/`cta_group`/`lane_align`; `row` must be 0 (the
        /// full-datapath layouts are lane-anchored).
        dst: TmemOperand,
        a: MmaOperand,
        b: MmaOperand,
        m: u32,
        n: u32,
        k: u32,
        accum: bool,
        trans_a: bool,
        trans_b: bool,
        cta_group: u8,
        /// Block-scaled MMA scale vectors for A and B, held in TMEM as packed u32
        /// cells (4 scale bytes each) or raw e4m3 bytes (nvfp4), addressed by
        /// absolute physical (lane, col).
        ///
        /// Two scale modes share this field set:
        /// * fp8 block-128 (UE8M0): one scale per operand row, constant over the
        ///   whole k-slice. `sf_e4m3=false`, `sf_block=0` (per-row); `sf_byte`
        ///   selects which of the 4 packed bytes applies, dequant `2^(byte-127)`.
        /// * nvfp4 block-16 (e4m3): one scale per 16 contiguous k-elements.
        ///   `sf_e4m3=true`, `sf_block=16`; this MMA's k spans `k/16` blocks whose
        ///   scales are bytes `0..k/16` of the cell, each decoded as e4m3.
        sfa: Option<TmemOperand>,
        sfb: Option<TmemOperand>,
        sf_byte: u8,
        /// scale decode: e4m3 (nvfp4) when true, UE8M0 biased exponent (fp8) when false.
        sf_e4m3: bool,
        /// scale block width in operand elements; 0 = one scale per row (fp8).
        sf_block: u32,
        /// operands are packed fp4 (e2m1, 2 per u8 byte); materialize by unpacking.
        a_fp4: bool,
        b_fp4: bool,
        /// d-tmem accumulator lane field (0 or 16): only the cta_group=1 m=64
        /// (Layout F) accumulator uses 16 to place its second 64-row half at
        /// lane 16+. A property of THIS MMA's accumulator write — hardware
        /// D-lane placement, not a layout.
        lane_align: u8,
    },
    /// `tcgen05.cp` — bulk SMEM -> TMEM copy of packed u32 scale-factor cells
    /// (or raw e4m3 scale bytes for nvfp4). `dst` is the absolute physical TMEM
    /// base (lane 0, col); `src` stays an SMEM tile. With `cta_group=2` one
    /// leader issue drives both CTAs' datapaths: each CTA copies from its own
    /// SMEM into its own TMEM. Retirement is observed via `tcgen05_commit`,
    /// like the MMA; in the value model the copy is applied at issue (the
    /// tcgen05 engine executes its ops in issue order, so a same-stream MMA
    /// reading the destination never observes a stale value).
    Tcgen05Cp {
        dst: TmemOperand,
        src: TensorSlice,
        cta_group: u8,
    },
    Tcgen05Commit {
        mbar: MBarRef,
        stage: Option<ScalarValue>,
        cta_group: u8,
        multicast_cta_mask: Option<u16>,
    },
    /// `tcgen05.ld` — TMEM -> REG datapath read. `src` is the absolute physical
    /// TMEM base address (lane, col) + cell dtype; `dst` the REG fragment.
    Tcgen05Ld {
        dst: TensorSlice,
        src: TmemOperand,
        shape: LdStShape,
        num: u32,
    },
    Tcgen05WaitLd,
    /// `tcgen05.st` — REG -> TMEM datapath write. `dst` is the absolute physical
    /// TMEM base address (lane, col) + cell dtype; `src` the REG fragment.
    Tcgen05St {
        dst: TmemOperand,
        src: TensorSlice,
        shape: LdStShape,
        num: u32,
    },
    Tcgen05WaitSt,

    // ---- warp matrix load/store (SMEM row addresses <-> packed REG fragments) ----
    LdMatrix {
        dst: TensorSlice,
        src: TensorSlice,
        shape: MatrixShape,
        num: u32,
        trans: bool,
        dtype: MatrixDType,
    },
    StMatrix {
        dst: TensorSlice,
        src: TensorSlice,
        shape: MatrixShape,
        num: u32,
        trans: bool,
        dtype: MatrixDType,
    },
    /// Warp-level SM80 tensor-core MMA (`mma.sync.aligned.m{M}n{N}k{K}.row.col`).
    /// D = A·Bᵀ + C, with A (M×K) / B (N×K) bf16 reg fragments and C/D (M×N)
    /// f32 reg accumulators, all in the standard mma warp fragment layout.
    WarpMma {
        d: TensorSlice,
        a: TensorSlice,
        b: TensorSlice,
        c: TensorSlice,
        m: u32,
        n: u32,
        k: u32,
        ab_dtype: DType, // A/B operand type — the PTX .bf16 / .f16 (C/D are f32)
    },

    // ---- register ALU ----
    RegFill {
        dst: TensorSlice,
        value: RegOperand,
    },
    RegUnary {
        dst: TensorSlice,
        src: RegOperand,
        op: RegUnaryOp,
    },
    RegAdd {
        dst: TensorSlice,
        lhs: RegOperand,
        rhs: RegOperand,
        rounding: Rounding,
    },
    RegSub {
        dst: TensorSlice,
        lhs: RegOperand,
        rhs: RegOperand,
        rounding: Rounding,
    },
    RegMul {
        dst: TensorSlice,
        lhs: RegOperand,
        rhs: RegOperand,
    },
    RegFma {
        dst: TensorSlice,
        a: RegOperand,
        b: RegOperand,
        c: RegOperand,
    },
    RegMax {
        dst: TensorSlice,
        lhs: RegOperand,
        rhs: RegOperand,
    },
    RegMin {
        dst: TensorSlice,
        lhs: RegOperand,
        rhs: RegOperand,
    },
    RegBitwise {
        dst: TensorSlice,
        lhs: RegOperand,
        rhs: RegOperand,
        op: RegBinaryOp,
    },
    RegReduce {
        dst: TensorSlice,
        src: RegOperand,
        op: RegReduceOp,
    },
    RegCondRescale {
        dst: TensorSlice,
        src: RegOperand,
        scale: RegOperand,
        threshold: RegOperand,
        scope: RegCondScope,
    },
    RegSoftmaxRescale {
        row_max: TensorSlice,
        row_scale: TensorSlice,
        row_max_old: RegOperand,
        row_max_new: RegOperand,
        scale_log2: RegOperand,
        threshold: RegOperand,
    },
    RegCausalMask {
        dst: TensorSlice,
        src: RegOperand,
        query_start: ScalarValue,
        key_start: ScalarValue,
        group_size: u32,
        mask_value: RegOperand,
        /// Fragment orientation. False = forward `[q-row, kv-col]` (q = query_start +
        /// row/group_size, k = key_start + col). True = backward `[kv-row, q-col]` (the
        /// fa-bwd fragment is transposed): k = key_start + row, q = query_start +
        /// col/group_size — group_size lands on the q (col) axis. Both mask when k > q.
        swap_qk: bool,
    },
    RegCombineIntFracEx2 {
        dst: TensorSlice,
        rounded: RegOperand,
        frac_ex2: RegOperand,
    },
    RegCvt {
        dst: TensorSlice,
        src: TensorSlice,
        rounding: Rounding,
    },
    RegLoad {
        dst: TensorSlice,
        src: TensorSlice,
    },
    RegStore {
        dst: TensorSlice,
        src: TensorSlice,
    },

    // ---- fence / sync ----
    Fence {
        kind: FenceKind,
        scope: FenceScope,
    },
    CtaSync,
    WgSync {
        barrier_id: u32,
    },
    /// Named barrier across `num_warps` warps that may span warpgroups —
    /// `bar.sync barrier_id, num_warps*32` (flashattn `NamedBarrierBwdSm100`).
    /// Unlike WgSync (per-warpgroup), threads from different roles rendezvous
    /// on the shared `barrier_id` (count-based completion).
    NamedBarrier {
        barrier_id: u32,
        num_warps: u32,
    },
    WarpSync,
    ClusterSync,
    /// Split cluster barrier — ARRIVE side. A non-blocking collective arrival
    /// (`barrier.cluster.arrive`, aligned) issued once at CTA scope after the
    /// prologue init. Paired with per-role `ClusterBarrierWait`s: decouples the
    /// cluster-barrier latency from each role's local setup, and idle warps skip the
    /// wait. Modeled faithfully (unlike a codegen-synthesized split of the fused
    /// `ClusterSync`) so the protocol checker verifies every role waits before any
    /// cross-CTA (peer mbarrier) access.
    ClusterBarrierArrive,
    /// Split cluster barrier — WAIT side. Blocks the calling role until all threads
    /// of the cluster have executed `ClusterBarrierArrive`. Allowed inside a role
    /// (unlike `ClusterSync`).
    ClusterBarrierWait,
    /// Standalone per-warpgroup register budget (canon's INVARIANT-I1b per-role
    /// `setmaxnreg`). Emitted as a warpgroup-gated `if wg_id == <warpgroup>:
    /// T.ptx.setmaxnreg(<inc>, <count>)` where `inc = count > 128` (rise above the
    /// 128-reg default → `True`/inc, else `False`/dec). Unlike the `maxnreg` field on
    /// `Role` (which is bound to a role branch), this is a free-standing statement so a
    /// warpgroup whose warps live in SEPARATE warp-level roles (e.g. the producer wg0
    /// = a `warp=0` MMA role + a `warp=2` TMA role) can still issue the collective
    /// `setmaxnreg` from all 4 of its warps before the role branches diverge. The
    /// `setmaxnreg.inc` side claims registers the consumer warpgroup released via its
    /// own `setmaxnreg.dec` — canon's producer-drop / consumer-raise rebalance.
    SetMaxNReg {
        warpgroup: u32,
        count: u32,
    },
}

impl Stmt {
    /// Nested statement bodies this node owns (empty for leaf statements) —
    /// mirrors Python `Stmt.child_bodies`, used by generic structural walks.
    pub fn child_bodies(&self) -> Vec<&[Stmt]> {
        match self {
            Stmt::KernelInit { body, .. }
            | Stmt::KernelFinalize { body, .. }
            | Stmt::Role { body, .. }
            | Stmt::ForLoop { body, .. }
            | Stmt::ForEachTask { body, .. }
            | Stmt::SchedulerImpl { body, .. }
            | Stmt::Loop { body } => vec![body],
            Stmt::If { then_body, .. } => vec![then_body],
            _ => vec![],
        }
    }
}
