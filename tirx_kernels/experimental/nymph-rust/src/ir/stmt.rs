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
use super::tensor::{Tensor, TensorSlice};
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
    TmemAlloc {
        tensor: Arc<Tensor>,
        n_cols: u32,
        cta_group: u8,
    },
    TmemDealloc {
        tensor: Arc<Tensor>,
        n_cols: u32,
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
        multicast_cta_mask: Option<u16>,
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
        dst: TensorSlice,
        a: TensorSlice,
        b: TensorSlice,
        m: u32,
        n: u32,
        k: u32,
        accum: bool,
        trans_a: bool,
        trans_b: bool,
        cta_group: u8,
        /// Block-scaled MMA scale vectors for A and B, held in TMEM as packed u32
        /// cells (4 scale bytes each).
        ///
        /// Two scale modes share this field set:
        /// * fp8 block-128 (UE8M0): one scale per operand row, constant over the
        ///   whole k-slice. `sf_e4m3=false`, `sf_block=0` (per-row); `sf_byte`
        ///   selects which of the 4 packed bytes applies, dequant `2^(byte-127)`.
        /// * nvfp4 block-16 (e4m3): one scale per 16 contiguous k-elements.
        ///   `sf_e4m3=true`, `sf_block=16`; this MMA's k spans `k/16` blocks whose
        ///   scales are bytes `0..k/16` of the cell, each decoded as e4m3.
        sfa: Option<TensorSlice>,
        sfb: Option<TensorSlice>,
        sf_byte: u8,
        /// scale decode: e4m3 (nvfp4) when true, UE8M0 biased exponent (fp8) when false.
        sf_e4m3: bool,
        /// scale block width in operand elements; 0 = one scale per row (fp8).
        sf_block: u32,
        /// operands are packed fp4 (e2m1, 2 per u8 byte); materialize by unpacking.
        a_fp4: bool,
        b_fp4: bool,
    },
    /// `tcgen05.cp` — bulk SMEM -> TMEM copy of packed u32 scale-factor cells.
    /// With `cta_group=2` one leader issue drives both CTAs' datapaths: each CTA
    /// copies from its own SMEM into its own TMEM. Retirement is observed via
    /// `tcgen05_commit`, like the MMA; in the value model the copy is applied at
    /// issue (the tcgen05 engine executes its ops in issue order, so a same-stream
    /// MMA reading the destination never observes a stale value).
    Tcgen05Cp {
        dst: TensorSlice,
        src: TensorSlice,
        cta_group: u8,
    },
    Tcgen05Commit {
        mbar: MBarRef,
        stage: Option<ScalarValue>,
        cta_group: u8,
        multicast_cta_mask: Option<u16>,
    },
    Tcgen05Ld {
        dst: TensorSlice,
        src: Arc<Tensor>,
        shape: LdStShape,
        num: u32,
        row: ScalarValue,
        col: ScalarValue,
    },
    Tcgen05WaitLd,
    Tcgen05St {
        dst: Arc<Tensor>,
        src: TensorSlice,
        shape: LdStShape,
        num: u32,
        row: ScalarValue,
        col: ScalarValue,
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
        ab_dtype: DType,   // A/B operand type — the PTX .bf16 / .f16 (C/D are f32)
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
