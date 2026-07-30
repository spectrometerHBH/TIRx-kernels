"""Chunked Gated Delta Net (GDN) prefill expressed in Nymph IR."""

# Per-warp execution model audit (mirrors the fp16/fa4 pattern).

from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..builder import IRBuilder
from ..nymph_rs import (
    DType,
    FenceKind,
    FenceScope,
    Kernel,
    LaunchShape,
    MBarKind,
    MemorySpace,
    SmemSwizzleLayout,
    Swizzle,
    TensorSlice,
    TmemTensor,
)

HQK = 4
HV = 8
GVA_RATIO = HV // HQK
K_DIM = 128
V_DIM = 128
BT = 64
SM_COUNT = 148
NUM_THREADS = 128
N_COLS_TMEM = 512
LD_NUM = 8  # tcgen05.ld.16x256b.x8 -> 32 regs (4*num)

# FlashInfer 12-warp specialization (gated_delta_net_chunked.py:232-239).
CG0_WARPS = (0, 1, 2, 3)  # compute group 0: T-pairwise, kk_epi, qk_epi, WY inverse
CG1_WARPS = (4, 5, 6, 7)  # compute group 1: new_v_epi, kv_update_epi, qkv_epilogue
MMA_WARP = 8  # issues all 7 tcgen05 GEMMs
TMA_WARP = 9  # TMA-load q/k/v
GATE_WARP = 10  # load gate/beta + log2 + warp prefix-sum cumsum
EPI_WARP = 11  # output store
NUM_WARPS = 12


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sl(t, offs, shape):
    return TensorSlice(tensor=t, offsets=offs, shape=shape)


@dataclass(frozen=True, slots=True)
class GdnPrefillConfig:
    num_seqs: int = 1
    seqlen: int = 128  # fixed-length: per-seq length (varlen=False). varlen: max for sizing.
    scale: float = 1.0 / (K_DIM**0.5)
    launch_shape: LaunchShape | None = None
    varlen: bool = False  # True: cu_seqlens arg, runtime per-tile chunk count, OOB-masked tails
    # head config (flashinfer parametrizes these).
    num_q_heads: int = 4
    num_v_heads: int = 8
    # io dtype: "bfloat16" or "float16" (flashinfer tests both).
    io_dtype: str = "bfloat16"


def _io_dt(config: GdnPrefillConfig) -> DType:
    return DType.F16 if config.io_dtype == "float16" else DType.BF16


def gdn_prefill_task_config(num_seqs: int, seqlen: int) -> GdnPrefillConfig:
    return GdnPrefillConfig(num_seqs=num_seqs, seqlen=seqlen)


# Bench-suite perf configs (the CONFIGS contract, bench/nymph_bench_guide.md).
_FI_HEADS = [(2, 8), (4, 16), (8, 32), (16, 64), (16, 32), (16, 48), (16, 16), (32, 32)]
_FI_SEQ_SHAPES = [
    ({"num_seqs": 1, "seqlen": 2048}, "1x2048"),
    ({"num_seqs": 1, "seqlen": 8192}, "1x8192"),
    ({"num_seqs": 1, "seqlen": 65536}, "1x65536"),
    ({"seqlens": [8192] * 8}, "8192x8"),
    ({"seqlens": [2048, 6144]}, "2048+6144"),
]
CONFIGS = [
    {"num_seqs": 1, "seqlen": 64, "label": "ns1_t64"},
    {"num_seqs": 1, "seqlen": 512, "label": "ns1_t512"},
    {"num_seqs": 1, "seqlen": 2048, "label": "ns1_t2048"},
    {"num_seqs": 20, "seqlen": 192, "label": "ns20_t192"},
    {"num_seqs": 48, "seqlen": 64, "label": "ns48_t64"},
    {"seqlens": [70, 130], "label": "v_70_130"},
] + [
    {**seq, "num_q_heads": q, "num_v_heads": v, "label": f"h{q}q{v}v_{sl}"}
    for q, v in _FI_HEADS
    for seq, sl in _FI_SEQ_SHAPES
]

# Build + protocol-check coverage lives in the tests.

# Build + protocol-check coverage.
PROTOCOL_CONFIGS = [
    {"num_seqs": ns, "seqlen": T, "label": f"ns{ns}_t{T}"}
    for ns, T in [
        (1, 64),
        (1, 128),
        (1, 192),
        (1, 256),
        (1, 512),
        (1, 1024),
        (1, 2048),
        (2, 128),
        (2, 256),
        (2, 512),
        (4, 128),
        (4, 256),
        (8, 64),
        (20, 64),
        (20, 192),
        (32, 128),
        (48, 64),  # num_work > SM_COUNT: multi-tile/CTA
    ]
]

# Cheaper fixed-length shapes for the cell-exact value sweep.
CONFIGS_SUPPORTED = [
    {"num_seqs": ns, "seqlen": T, "label": f"ns{ns}_t{T}"} for ns, T in [(1, 64), (1, 128)]
]

# Head configs = flashinfer test_prefill_delta_rule.py num_q/k/v_heads (head_dim=128).
HEAD_CONFIGS = [
    {"num_q_heads": q, "num_v_heads": v, "label": f"h{q}q{v}v"}
    for q, v in [(1, 1), (4, 1), (3, 3), (6, 2), (1, 2), (2, 4), (16, 32), (16, 64), (4, 8)]
]

# Varlen (cu_seqlens) batches.
VARLEN_CONFIGS = [
    {"seqlens": s, "label": "v_" + "_".join(map(str, s))}
    for s in [
        [64, 128],
        [128, 64, 192],
        [64, 64],
        [70, 130],
        [6, 100],
        [200],
        [13, 64, 191],
        [256, 320],
        [64, 64, 64, 64],
        [320, 7, 100],
    ]
] + [
    # multi-tile/CTA varlen (num_work = num_seqs·HV > SM_COUNT): mixed + non-BT-multiple.
    {
        "seqlens": [
            70,
            64,
            130,
            6,
            192,
            100,
            64,
            128,
            7,
            200,
            64,
            191,
            64,
            256,
            13,
            64,
            128,
            320,
            100,
            70,
        ],
        "label": "v_mt20",
    }
]


def build_gdn_prefill(config: GdnPrefillConfig = GdnPrefillConfig()) -> Kernel:
    NS, T = config.num_seqs, config.seqlen
    _validate_config(config)
    # Head model (flashinfer gated_delta_net_chunked.py:405-472).
    H_Q, H_V = config.num_q_heads, config.num_v_heads
    HK = min(H_Q, H_V)  # num_k_heads (k/v are paired into the kv head)
    NEFF = max(H_Q, H_V)  # effective / output heads (= num_o_heads)
    HR = NEFF // HK  # heads per kv group
    IS_GQA = H_Q > H_V
    iod = _io_dt(config)  # io / 16-bit-operand dtype (bf16 or f16)
    n_chunks = _ceil_div(T, BT)
    total_t = NS * T
    num_work = NS * NEFF
    launch_shape = config.launch_shape or (min(SM_COUNT, num_work),)

    sizes = dict(
        k=2 * BT * K_DIM * 2,
        q=BT * K_DIM * 2,
        v=BT * V_DIM * 2,  # k double-buffered (2 stages)
        vnewt=V_DIM * BT * 2,
        attn=BT * BT * 2,
        tmpt=V_DIM * BT * 2,
        out=BT * V_DIM * 2,
        m=BT * BT * 4,
        gcs=BT * NEFF * 4,
        beta=BT * NEFF * 4,
        state=K_DIM * V_DIM * 2,  # fp16 SMEM S_prev copy (GEMM3/4's B operand)
        dcs=BT * BT * 2,  # DC scratch for the hierarchical-inverse merges (mma.sync)
    )
    mma_smem_names = {"k", "q", "v", "vnewt", "attn", "tmpt", "state"}
    off, offs = 0, {}
    for name, nbytes in sizes.items():
        if name in mma_smem_names:
            off = _align(off, 1024)
        offs[name] = off
        off += nbytes
    # A_inv aliases NV (vnewt): flashinfer's sAinv holds A_inv then overwrites with NV.
    offs["ainv"] = offs["vnewt"]

    # mbarrier.arrive is PER-THREAD.
    CG0_T = 128  # CG0 arrives are warpgroup-wide (warps 0-3)
    CG1_T = 128  # CG1 arrives are warpgroup-wide (warps 4-7)
    GATE_T = 32  # gate warp arrives warp-wide (each lane signs off its gcs/beta lanes)
    # (barrier name -> (kind, arrival count)) for the 12-warp producer/consumer pipeline.
    KSTAGES = 2  # K is double-buffered (flashinfer smem_k_stages=2): prefetch next chunk
    bar_spec = {
        "tk": (MBarKind.TMA, 1, KSTAGES),  # K full (2 stages)
        # K empty: armed by the MMA PIPELINE (tcgen05_commit after GEMM7).
        "k_free": (MBarKind.TCGEN05, 1, KSTAGES),
        "tq": (MBarKind.TMA, 1),
        "tv": (MBarKind.TMA, 1),
        "tg": (MBarKind.TMA, 1),
        "tb": (MBarKind.TMA, 1),
        "d_kk": (MBarKind.TCGEN05, 1),
        "d_qk": (MBarKind.TCGEN05, 1),
        "d_ks": (MBarKind.TCGEN05, 1),
        "d_qs": (MBarKind.TCGEN05, 1),
        "d_nv": (MBarKind.TCGEN05, 1),
        "d_oi": (MBarKind.TCGEN05, 1),
        "d_ds": (MBarKind.TCGEN05, 1),
        "gate_ready0": (MBarKind.THREAD, GATE_T),
        "gate_ready1": (MBarKind.THREAD, GATE_T),
        "ainv_ready": (MBarKind.THREAD, CG0_T),
        "qkv_ready": (MBarKind.THREAD, CG0_T),
        "sT_ready": (MBarKind.THREAD, CG1_T),
        "delta_ready": (MBarKind.THREAD, CG1_T),
        "vnew_ready": (MBarKind.THREAD, CG1_T),
        "ktvng_ready": (MBarKind.THREAD, CG1_T),
        "o_ready": (MBarKind.THREAD, CG1_T),
        "f_kk": (MBarKind.THREAD, CG0_T),
        "f_qk": (MBarKind.THREAD, CG0_T),
        "f_ks": (MBarKind.THREAD, CG1_T),
        "f_qs": (MBarKind.THREAD, CG1_T),
        "f_nv": (MBarKind.THREAD, CG1_T),
        "f_oi": (MBarKind.THREAD, CG1_T),
        "chunk_free": (MBarKind.THREAD, CG1_T + 1),
    }
    cursor = _align(off, 8)
    bar_offsets = {}
    for name, spec in bar_spec.items():
        bar_offsets[name] = cursor
        cursor += (spec[2] if len(spec) > 2 else 1) * 8
    tmem_addr_offset = _align(cursor, 4)
    smem_size_bytes = tmem_addr_offset + 4

    k = IRBuilder(
        "nymph_gdn_prefill",
        num_warps=NUM_WARPS,
        smem_size_bytes=smem_size_bytes,
        launch_shape=launch_shape,
    )

    q_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(total_t, H_Q, K_DIM))
    k_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(total_t, HK, K_DIM))
    v_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(total_t, H_V, V_DIM))
    gate_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(total_t, NEFF))
    beta_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(total_t, NEFF))
    out_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(total_t, NEFF, V_DIM))
    state_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(NS, NEFF, K_DIM, V_DIM))
    # varlen: cu_seqlens[NS+1] gives each sequence's token range; num_chunks varies per tile.
    cu_g = (
        k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(NS + 1,)) if config.varlen else None
    )

    def task_geom(task):
        # per-tile (seq, head) geometry: token base + chunk count + the q/k/v load heads.
        work = task.field("work")
        seq = work // NEFF
        eh = work % NEFF
        shared = eh // HR
        q_head = eh if IS_GQA else shared
        k_head = shared
        v_head = shared if IS_GQA else eh
        if config.varlen:
            base = k.scalar(initial=_sl(cu_g, (seq,), (1,)))
            nxt = k.scalar(initial=_sl(cu_g, (seq + 1,), (1,)))
            slen = nxt - base  # seqlen_b (runtime)
            nch = (slen + (BT - 1)) // BT  # ceil_div(seqlen_b, BT)
            return seq, eh, q_head, k_head, v_head, base, nch, slen
        return seq, eh, q_head, k_head, v_head, seq * config.seqlen, n_chunks, config.seqlen

    def sm(name, dt, shape, *, mma_operand=False):
        layout = SmemSwizzleLayout(Swizzle.B128) if mma_operand else None
        return k.tensor(
            space=MemorySpace.SMEM, dtype=dt, shape=shape, byte_offset=offs[name], layout=layout
        )

    # K double-buffered: single [2, BT, K_DIM] ring; the stage is the LEADING dim (kstg = c%2).
    k_s = sm("k", iod, (2, BT, K_DIM), mma_operand=True)
    q_s = sm("q", iod, (BT, K_DIM), mma_operand=True)
    v_s = sm("v", iod, (BT, V_DIM), mma_operand=True)
    vnewt_s = sm("vnewt", iod, (V_DIM, BT), mma_operand=True)
    ainv_s = sm("ainv", iod, (BT, BT), mma_operand=True)
    attn_s = sm("attn", iod, (BT, BT), mma_operand=True)
    tmpt_s = sm("tmpt", iod, (V_DIM, BT), mma_operand=True)
    out_s = sm("out", iod, (BT, V_DIM))
    m_s = sm("m", DType.F32, (BT, BT))
    dcs_s = sm("dcs", iod, (BT, BT))  # hierarchical-inverse DC scratch
    gcs_s = sm("gcs", DType.F32, (BT, NEFF))
    beta_s = sm("beta", DType.F32, (BT, NEFF))
    s_s = sm("state", iod, (K_DIM, V_DIM), mma_operand=True)  # fp16 SMEM S_prev copy (GEMM3/4 B)

    # TMEM column plan matching flashinfer's separate allocations.
    s_tmem = k.tmem_tensor(0)  # (128, V_DIM) f32: 0-127
    qstate_tmem = k.tmem_tensor(192)  # (64, V_DIM) f32: 192-319
    # shared_acc = 2 stages of 64 cols (flashinfer 64-col × 2): kk→stage0, qk→stage1 (pipeline).
    acc_s0 = k.tmem_tensor(320)  # (64, BT) f32, stage 0: 320-383
    acc_s1 = k.tmem_tensor(384)  # (64, BT) f32, stage 1: 384-447
    acc_tmem = k.tmem_tensor(320)  # (64, V_DIM) f32, union view (ks/nv)

    def reg(dt, shape):
        return k.tensor(space=MemorySpace.REG, dtype=dt, shape=shape)

    r1 = reg(DType.F32, (1,))
    r2 = reg(DType.F32, (1,))
    r3 = reg(DType.F32, (1,))
    racc = reg(DType.F32, (1,))
    rb16 = reg(iod, (1,))
    rb16b = reg(iod, (1,))
    glast = reg(DType.F32, (1,))  # gcs[BT-1] (chunk's total log-decay)
    frag = reg(DType.F32, (32,))  # .16x256b.x8 readback = 32 regs
    frag32 = reg(DType.F32, (64,))  # .32x32b readback (state, thread=row, 64 cols)
    zrow = reg(DType.F32, (V_DIM,))
    # hierarchical-inverse (mma.sync) fragments.
    imm_a = reg(DType.U32, (2,))  # A = Qinv (m16k8 packed-bf16, m-broadcast)
    imm_b = reg(DType.U32, (1,))  # B = C / Pinv
    imm_acc = reg(DType.F32, (4,))  # mma accumulator
    imm_accb = reg(iod, (4,))  # accumulator -> bf16 (store / next-A)
    sttile = reg(iod, (2,))  # one m8n8 tile (2 bf16 = 1 stmatrix word) for reg->SMEM staging
    sttile2 = reg(iod, (2,))  # second tile (vnew gated/ungated emit two stmatrix tiles)
    v_frag = reg(iod, (64,))  # delta's ldmatrix v fragment (2 blk × 16 packed b16x2 words)
    sinp_reg = reg(iod, (64,))  # bf16 cvt slots for the s_s state staging (one half at a time)
    # Per-chunk gating fragments.
    t_frag = reg(DType.F32, (32,))
    t_row = reg(DType.F32, (2,))
    t_beta = reg(DType.F32, (2,))
    dexp2 = reg(DType.F32, (2,))
    kgate2 = reg(DType.F32, (2,))

    bars = {
        nm: k.mbar(
            kind=spec[0], byte_offset=bar_offsets[nm], stages=(spec[2] if len(spec) > 2 else 1)
        )
        for nm, spec in bar_spec.items()
    }

    sched = k.scheduler(k.task_space(grid=(num_work,), fields=("work",)))

    with k.if_warp(0):
        # tmem_alloc is warp-collective (full warp 0).
        k.tmem_alloc(0, N_COLS_TMEM, addr_byte_offset=tmem_addr_offset, cta_group=1)
        with k.if_elected():
            for nm, spec in bar_spec.items():
                stg = spec[2] if len(spec) > 2 else 1
                for s in range(stg):
                    k.mbarrier_init(bars[nm], count=spec[1], stage=s)
    # No implicit barrier between top-level statements.
    k.cta_sync()

    _emit(
        k,
        config,
        n_chunks,
        sched,
        task_geom,
        (q_g, k_g, v_g, gate_g, beta_g, out_g, state_g),
        (k_s, q_s, v_s, vnewt_s, ainv_s, attn_s, tmpt_s, out_s, m_s, gcs_s, beta_s, dcs_s, s_s),
        (s_tmem, qstate_tmem, acc_tmem, acc_s0, acc_s1),
        (
            r1,
            r2,
            r3,
            racc,
            rb16,
            rb16b,
            glast,
            frag,
            frag32,
            zrow,
            imm_a,
            imm_b,
            imm_acc,
            imm_accb,
            sttile,
            sttile2,
            sinp_reg,
            t_frag,
            t_row,
            t_beta,
            dexp2,
            kgate2,
            v_frag,
        ),
        bars,
    )

    # Teardown: every stream's pipeline work happens-before the dealloc.
    k.cta_sync()
    with k.if_warp(0):
        k.tmem_relinquish(1)
        k.tmem_dealloc(0, N_COLS_TMEM, 1)
    return k.build()


def _emit(k, config, n_chunks, sched, task_geom, args, sm, tm, rg, bars):
    NEFF = max(config.num_q_heads, config.num_v_heads)  # final-state store: per effective head
    iod = _io_dt(config)  # io / 16-bit-operand dtype (bf16 or f16)
    q_g, k_g, v_g, gate_g, beta_g, out_g, state_g = args
    (k_s, q_s, v_s, vnewt_s, ainv_s, attn_s, tmpt_s, out_s, m_s, gcs_s, beta_s, dcs_s, s_s) = sm
    s_tmem, qstate_tmem, acc_tmem, acc_s0, acc_s1 = tm
    (
        r1,
        r2,
        r3,
        racc,
        rb16,
        rb16b,
        glast,
        frag,
        frag32,
        zrow,
        imm_a,
        imm_b,
        imm_acc,
        imm_accb,
        sttile,
        sttile2,
        sinp_reg,
        t_frag,
        t_row,
        t_beta,
        dexp2,
        kgate2,
        v_frag,
    ) = rg
    scale = config.scale

    def ph(c):
        return c % 2  # single-buffered barrier phase (chunk loop is Python-unrolled)

    def fence_pub(bid):  # publish generic SMEM writes to the MMA's async proxy
        k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
        k.wg_sync(barrier_id=bid)

    def issue(
        dst, a, b, m, n, kk, done, accum0=False, trans_a=False, trans_b=False, a_stg=0, b_stg=0
    ):
        # cute.gemm -> tcgen05.mma.cta_group::1.kind::f16 (UTCHMMA); MMA warp issues+commits.
        for g in range(kk // 16):
            a_off, a_sh = ((g * 16, 0), (16, m)) if trans_a else ((0, g * 16), (m, 16))
            b_off, b_sh = ((g * 16, 0), (16, n)) if trans_b else ((0, g * 16), (n, 16))
            if isinstance(a, TmemTensor):
                a_op = k.mma_a_tmem(a.at(*a_off), form="flat")
            else:
                a_tile = k.smem_tile(
                    a,
                    prefix_indices=(a_stg,) if a is k_s else (),
                    row_offset=a_off[0],
                    col_offset=a_off[1],
                    rows=a_sh[0],
                    cols=a_sh[1],
                )
                a_op = k.mma_a_smem(a_tile)
            b_op = k.smem_tile(
                b,
                prefix_indices=(b_stg,) if b is k_s else (),
                row_offset=b_off[0],
                col_offset=b_off[1],
                rows=b_sh[0],
                cols=b_sh[1],
            )
            k.tcgen05_mma(
                dst.at(0, 0),
                a_op,
                b_op,
                mma_m=m,
                mma_n=n,
                format="f16" if iod == DType.F16 else "bf16",
                block_scale=None,
                accum=(accum0 or g != 0),
                trans_a=trans_a,
                trans_b=trans_b,
                ws=False,
                cta_group=1,
            )
        k.tcgen05_commit(bars[done])

    def rss(dst_s, fac_neg_beta, mask_strict, tid, lane, warp, dst_bf16=False, acc=None):
        # kk_epi / qk_epi (CG0).
        k.tcgen05_ld(
            frag, (acc if acc is not None else acc_tmem).at(0, 0), shape="16x256b", num=LD_NUM
        )
        k.tcgen05_wait_ld()
        for va in range(2):
            row = (lane // 4) + 8 * va + 16 * warp
            for vb in range(8):
                for v0p in range(2):
                    r = v0p + 2 * va + 4 * vb
                    col = v0p + 2 * (lane % 4) + 8 * vb
                    k.reg_mul(r1, _sl(t_frag, (r,), (1,)), _sl(frag, (r,), (1,)))
                    if fac_neg_beta:
                        k.reg_mul(r1, r1, _sl(t_beta, (va,), (1,)))
                    else:
                        k.reg_mul(r1, r1, float(scale))
                    if dst_bf16:
                        # W_qkv (attn_s) is a GEMM6 operand → stage the m8n8 tile via stmatrix.
                        k.reg_cvt(_sl(sttile, (v0p,), (1,)), r1)
                        if v0p == 1:
                            _stm(k, dst_s, 8 * va + 16 * warp, 8 * vb, sttile, lane, trans=False)
                    else:
                        k.reg_store(_sl(dst_s, (row, col), (1, 1)), r1)  # m_s f32 (inverse input)
        k.wg_sync(barrier_id=10)
        with k.if_(tid < BT):
            for j in range(BT):
                cond = (tid <= j) if mask_strict else (tid < j)
                with k.if_(cond):
                    if dst_bf16:
                        k.reg_fill(rb16, 0.0)
                        k.reg_store(_sl(dst_s, (tid, j), (1, 1)), rb16)
                    else:
                        k.reg_fill(r3, 0.0)
                        k.reg_store(_sl(dst_s, (tid, j), (1, 1)), r3)
        k.wg_sync(barrier_id=10)

    # ============== TMA-load warp (9).
    with k.if_warp(TMA_WARP):
        # fi's register budgets (CK:241-243): support warps squeeze to 24.
        k.set_maxnreg(24)
        with k.if_elected():
            gc = k.scalar(initial=0)  # cumulative chunk index, carried across the persistent
            with k.for_each_task(
                sched
            ) as task:  # tile loop (= CUTLASS PipelineState, never per-tile reset)
                seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
                with k.for_loop(stop=NCH) as c:  # runtime chunk loop (varlen num_chunks)
                    gtok = tok_base + c * BT
                    # K (double-buffered): stage gc%2.
                    kstg, kocc = gc % 2, gc // 2
                    k.mbarrier_wait(bars["k_free"], stage=kstg, phase=(kocc + 1) % 2)
                    k.mbarrier_arrive_expect_tx(bars["tk"], bytes=BT * K_DIM * 2, stage=kstg)
                    k.tma_load(
                        _sl(k_s, (kstg, 0, 0), (1, BT, K_DIM)),
                        k_g,
                        mbar=bars["tk"],
                        coords=(gtok, k_head, 0),
                        mbar_stage=kstg,
                        shape=(1, BT, K_DIM),
                        gmem_shape=(BT, 1, K_DIM),
                    )
                    # q, v (single-buffered): gated by chunk_free.
                    with k.if_(gc > 0):
                        k.mbarrier_wait(bars["chunk_free"], phase=(gc - 1) % 2)
                    for nm, dst, src, hd, cols in (
                        ("tq", q_s, q_g, q_head, K_DIM),
                        ("tv", v_s, v_g, v_head, V_DIM),
                    ):
                        k.mbarrier_arrive_expect_tx(bars[nm], bytes=BT * cols * 2)
                        k.tma_load(
                            _sl(dst, (0, 0), (BT, cols)),
                            src,
                            mbar=bars[nm],
                            coords=(gtok, hd, 0),
                            shape=(BT, cols),
                            gmem_shape=(BT, 1, cols),
                        )
                    k.scalar_store(gc, gc + 1)  # advance the pipeline state

    # ============== gate/beta warp (10): load gate/beta + cumsum -> gate_ready ==============.
    with k.if_warp(GATE_WARP):
        k.set_maxnreg(24)  # fi's support-warp budget (CK:241-243)
        lane = k.lane_id()
        gc = k.scalar(initial=0)  # cumulative chunk index (pipeline state across tiles)
        with k.for_each_task(sched) as task:
            seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
            with k.for_loop(stop=NCH) as c:
                gtok = tok_base + c * BT
                with k.if_(gc > 0):
                    k.mbarrier_wait(bars["chunk_free"], phase=ph(gc - 1))
                # TMA issue is single-thread.
                with k.if_elected():
                    # Full (BT, HV) tile.
                    k.mbarrier_arrive_expect_tx(bars["tg"], bytes=BT * NEFF * 4)
                    k.tma_load(
                        gcs_s,
                        gate_g,
                        mbar=bars["tg"],
                        coords=(gtok, 0),
                        shape=(BT, NEFF),
                        gmem_shape=(BT, NEFF),
                    )
                    k.mbarrier_arrive_expect_tx(bars["tb"], bytes=BT * NEFF * 4)
                    k.tma_load(
                        beta_s,
                        beta_g,
                        mbar=bars["tb"],
                        coords=(gtok, 0),
                        shape=(BT, NEFF),
                        gmem_shape=(BT, NEFF),
                    )
                k.mbarrier_wait(bars["tg"], phase=ph(gc))
                if (
                    config.varlen
                ):  # OOB tokens (global pos >= seqlen_b) -> gate=1 (log2=0, no decay)
                    for half in range(2):
                        i = lane + half * 32
                        with k.if_((c * BT + i) >= slen):
                            k.reg_fill(r1, 1.0)
                            k.reg_store(_sl(gcs_s, (i, eh), (1, 1)), r1)
                    k.warp_sync()
                # gate warp: raw gate a -> log2(a) in place.
                for half in range(2):
                    i = lane + half * 32
                    k.reg_load(r1, _sl(gcs_s, (i, eh), (1, 1)))
                    k.reg_unary(r1, r1, op="log2")
                    k.reg_store(_sl(gcs_s, (i, eh), (1, 1)), r1)
                k.warp_sync()
                # inclusive cumsum over BT=64, 32 lanes x 2 elems (Hillis-Steele).
                for step in range(6):
                    ov = 1 << step
                    for half in range(2):
                        i = lane + half * 32
                        dreg = r2 if half == 0 else racc
                        with k.if_(i >= ov):
                            k.reg_load(r1, _sl(gcs_s, (i - ov, eh), (1, 1)))
                            k.reg_load(dreg, _sl(gcs_s, (i, eh), (1, 1)))
                            k.reg_add(dreg, dreg, r1)
                    k.warp_sync()
                    for half in range(2):
                        i = lane + half * 32
                        dreg = r2 if half == 0 else racc
                        with k.if_(i >= ov):
                            k.reg_store(_sl(gcs_s, (i, eh), (1, 1)), dreg)
                    k.warp_sync()
                k.mbarrier_wait(
                    bars["tb"], phase=ph(gc)
                )  # beta must be loaded before consumers read it
                if config.varlen:  # OOB tokens -> beta=0 (no state update, no delta contribution)
                    for half in range(2):
                        i = lane + half * 32
                        with k.if_((c * BT + i) >= slen):
                            k.reg_fill(r1, 0.0)
                            k.reg_store(_sl(beta_s, (i, eh), (1, 1)), r1)
                    k.warp_sync()
                k.mbarrier_arrive(bars["gate_ready0"])
                k.mbarrier_arrive(bars["gate_ready1"])
                k.scalar_store(gc, gc + 1)  # advance pipeline state

    # ============== MMA warp (8).
    with k.if_warp(MMA_WARP):
        k.set_maxnreg(24)  # fi's support-warp budget (CK:241-243)
        with k.if_elected():
            # gc = every-chunk pipeline state; gc_pos = the GEMM3/4 pipeline.
            gc = k.scalar(initial=0)
            gc_pos = k.scalar(initial=0)
            with k.for_each_task(sched) as task:
                seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
                with k.for_loop(stop=NCH) as c:
                    kstg = gc % 2  # K double-buffer stage (leading dim of k_s)
                    k.mbarrier_wait(bars["tk"], stage=gc % 2, phase=(gc // 2) % 2)
                    issue(
                        acc_s0, k_s, k_s, BT, BT, K_DIM, "d_kk", a_stg=kstg, b_stg=kstg
                    )  # GEMM1 W_kk -> stage 0
                    k.mbarrier_wait(bars["f_kk"], phase=ph(gc))
                    k.mbarrier_wait(bars["tq"], phase=ph(gc))
                    issue(
                        acc_s1, q_s, k_s, BT, BT, K_DIM, "d_qk", b_stg=kstg
                    )  # GEMM2 W_qk -> stage 1
                    k.mbarrier_wait(bars["f_qk"], phase=ph(gc))
                    with k.if_(
                        c > 0
                    ):  # chunk 0: S_prev=0, skip GEMM3/4 (flashinfer is_first_chunk)
                        # GEMM3/4 read the fp16 s_s (SMEM) as B=Sᵀ (trans_b); GEMM4 → q_state.
                        k.mbarrier_wait(bars["sT_ready"], phase=ph(gc_pos))
                        issue(
                            acc_tmem, k_s, s_s, BT, V_DIM, K_DIM, "d_ks", trans_b=True, a_stg=kstg
                        )  # GEMM3 K@S
                        k.mbarrier_wait(bars["f_ks"], phase=ph(gc_pos))
                        issue(
                            qstate_tmem, q_s, s_s, BT, V_DIM, K_DIM, "d_qs", trans_b=True
                        )  # GEMM4 Q@S → q_state
                        k.mbarrier_wait(bars["f_qs"], phase=ph(gc_pos))
                    k.mbarrier_wait(bars["ainv_ready"], phase=ph(gc))
                    k.mbarrier_wait(bars["delta_ready"], phase=ph(gc))
                    with k.if_(
                        c.eq(0)
                    ):  # chunk 0: S=0 → delta=v; read v_s directly (vᵀ via trans_b)
                        issue(acc_tmem, ainv_s, v_s, BT, V_DIM, BT, "d_nv", trans_b=True)
                    with k.if_(c > 0):
                        issue(
                            acc_tmem, ainv_s, tmpt_s, BT, V_DIM, BT, "d_nv"
                        )  # GEMM5 VNEW (deltaᵀ from tmpt_s, SMEM)
                    k.mbarrier_wait(bars["f_nv"], phase=ph(gc))
                    k.mbarrier_wait(bars["qkv_ready"], phase=ph(gc))
                    k.mbarrier_wait(bars["vnew_ready"], phase=ph(gc))
                    issue(
                        acc_tmem, attn_s, vnewt_s, BT, V_DIM, BT, "d_oi"
                    )  # GEMM6 O_intra (NVᵀ from vnewt_s, SMEM)
                    k.mbarrier_wait(bars["f_oi"], phase=ph(gc))
                    k.mbarrier_wait(bars["ktvng_ready"], phase=ph(gc))
                    issue(
                        s_tmem,
                        k_s,
                        tmpt_s,
                        K_DIM,
                        V_DIM,
                        BT,
                        "d_ds",
                        accum0=True,
                        trans_a=True,
                        a_stg=kstg,
                    )  # GEMM7 dS
                    # GEMM7 was the last K[c] consumer → free this K stage for chunk gc+2.
                    k.tcgen05_commit(bars["k_free"], stage=gc % 2)
                    k.scalar_store(gc, gc + 1)
                    with k.if_(c > 0):
                        k.scalar_store(gc_pos, gc_pos + 1)

    # ============== compute group 0 (warps 0-3): kk_epi, qk_epi, WY inverse ==============.
    with k.if_warpgroup(0):
        # fi's register budgets (CK:241-243, setmaxregister at role top): CG0 224.
        k.set_maxnreg(224)
        tid = k.tid_in_wg()
        lane = tid % 32
        warp = tid // 32
        gc = k.scalar(initial=0)  # cumulative chunk index (pipeline state across tiles)
        with k.for_each_task(sched) as task:
            seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
            with k.for_loop(stop=NCH) as c:
                with k.if_(gc > 0):
                    k.mbarrier_wait(bars["chunk_free"], phase=ph(gc - 1))
                k.mbarrier_wait(bars["gate_ready0"], phase=ph(gc))
                # Wave-3 C1: build the per-chunk gating fragments ONCE.
                for va in range(2):
                    row = (lane // 4) + 8 * va + 16 * warp
                    k.reg_load(r2, _sl(beta_s, (row, eh), (1, 1)))
                    k.reg_mul(_sl(t_beta, (va,), (1,)), r2, -1.0)
                    k.reg_load(_sl(t_row, (va,), (1,)), _sl(gcs_s, (row, eh), (1, 1)))
                for vb in range(8):
                    for v0p in range(2):
                        col = v0p + 2 * (lane % 4) + 8 * vb
                        k.reg_load(r2, _sl(gcs_s, (col, eh), (1, 1)))
                        for va in range(2):
                            r = v0p + 2 * va + 4 * vb
                            k.reg_sub(r1, _sl(t_row, (va,), (1,)), r2)
                            k.reg_unary(r1, r1, op="exp2")
                            k.reg_store(_sl(t_frag, (r,), (1,)), r1)
                # kk_epi: W_kk -> M_kk.
                k.mbarrier_wait(bars["d_kk"], phase=ph(gc))
                rss(m_s, True, True, tid, lane, warp, acc=acc_s0)
                k.mbarrier_arrive(bars["f_kk"])
                # qk_epi: W_qk -> W_qkv (attn_s).
                k.mbarrier_wait(bars["d_qk"], phase=ph(gc))
                rss(attn_s, False, False, tid, lane, warp, dst_bf16=True, acc=acc_s1)
                k.mbarrier_arrive(bars["f_qk"])
                fence_pub(10)
                k.mbarrier_arrive(bars["qkv_ready"])
                # ===== WY inverse: A_inv = (I+M)⁻¹.
                for r_ in range(1, 8):
                    with k.if_((tid < BT) & ((tid % 8) < r_)):
                        base = (tid // 8) * 8
                        jc = tid % 8
                        k.reg_fill(racc, 0.0)
                        for mm in range(r_):
                            k.reg_load(r1, _sl(m_s, (base + r_, base + mm), (1, 1)))
                            k.reg_load(r2, _sl(m_s, (base + mm, base + jc), (1, 1)))
                            k.reg_fma(racc, r1, r2, racc)
                    k.wg_sync(barrier_id=10)
                    with k.if_((tid < BT) & ((tid % 8) < r_)):
                        base = (tid // 8) * 8
                        jc = tid % 8
                        k.reg_load(r1, _sl(m_s, (base + r_, base + jc), (1, 1)))
                        k.reg_add(r1, r1, racc)
                        k.reg_store(_sl(m_s, (base + r_, base + jc), (1, 1)), r1)
                    k.wg_sync(barrier_id=10)
                # cvt m_s -> ainv_s (bf16) with unit diagonal (the matrix for the merges).
                with k.if_(tid < BT):
                    for c in range(BT):
                        k.reg_load(r1, _sl(m_s, (tid, c), (1, 1)))
                        k.reg_cvt(rb16, r1)
                        k.reg_store(_sl(ainv_s, (tid, c), (1, 1)), rb16)
                    k.reg_fill(rb16, 1.0)
                    k.reg_store(_sl(ainv_s, (tid, tid), (1, 1)), rb16)
                k.wg_sync(barrier_id=10)
                fence_pub(10)

                def _ldA(src, R, C):  # A = src[R:R+8,C:C+8] non-trans, m16 broadcast
                    k.ldmatrix(
                        _sl(imm_a, (0,), (1,)),
                        _sl(src, (R + lane % 8, C), (1, 8)),
                        num=1,
                        trans=False,
                    )
                    k.reg_store(_sl(imm_a, (1,), (1,)), _sl(imm_a, (0,), (1,)))

                def _ldB(src, R, C):
                    # trans=True loads raw tile[k,n] coordinates into the MMA B fragment.
                    k.ldmatrix(imm_b, _sl(src, (R + lane % 8, C), (1, 8)), num=1, trans=True)

                def _store8(dst, R, C, neg):
                    # The accumulator's top 8×8.
                    for ri in range(2):
                        if neg:
                            k.reg_mul(_sl(imm_acc, (ri,), (1,)), _sl(imm_acc, (ri,), (1,)), -1.0)
                        k.reg_cvt(_sl(imm_accb, (ri,), (1,)), _sl(imm_acc, (ri,), (1,)))
                    k.stmatrix(
                        _sl(dst, (R + lane % 8, C), (1, 8)),
                        _sl(imm_accb, (0,), (2,)),
                        num=1,
                        trans=False,
                    )

                def _merge(R, Cc, b, w):  # newC = -Qinv·C·Pinv (m_s=-L → store neg=False)
                    tb = b // 8
                    with k.if_((tid // 32).eq(w)):
                        for mi in range(tb):
                            for ni in range(tb):
                                k.reg_fill(imm_acc, 0.0)
                                for ki in range(tb):
                                    _ldA(ainv_s, R + b + mi * 8, Cc + b + ki * 8)
                                    _ldB(ainv_s, R + b + ki * 8, Cc + ni * 8)
                                    k.mma_sync(
                                        imm_acc, imm_a, imm_b, imm_acc, m=16, n=8, k=8, ab_dtype=iod
                                    )
                                _store8(dcs_s, w * 16 + mi * 8, ni * 8, False)
                        k.warp_sync()
                        k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                        for mi in range(tb):
                            for ni in range(tb):
                                k.reg_fill(imm_acc, 0.0)
                                for ki in range(tb):
                                    _ldA(dcs_s, w * 16 + mi * 8, ki * 8)
                                    _ldB(ainv_s, R + ki * 8, Cc + ni * 8)
                                    k.mma_sync(
                                        imm_acc, imm_a, imm_b, imm_acc, m=16, n=8, k=8, ab_dtype=iod
                                    )
                                _store8(ainv_s, R + b + mi * 8, Cc + ni * 8, neg=False)

                for t in range(4):
                    _merge(t * 16, t * 16, 8, t)
                k.wg_sync(barrier_id=10)
                fence_pub(10)
                for t in range(2):
                    _merge(t * 32, t * 32, 16, t)
                k.wg_sync(barrier_id=10)
                fence_pub(10)
                _merge(0, 0, 32, 0)
                k.wg_sync(barrier_id=10)
                # fold beta[j] into A_inv columns (for GEMM5 VNEW = A_inv·diagβ @ vᵀ).
                with k.if_(tid < BT):
                    for j in range(BT):
                        k.reg_load(r2, _sl(beta_s, (j, eh), (1, 1)))
                        k.reg_cvt(rb16b, r2)
                        k.reg_load(rb16, _sl(ainv_s, (tid, j), (1, 1)))
                        k.reg_mul(rb16, rb16, rb16b)
                        k.reg_store(_sl(ainv_s, (tid, j), (1, 1)), rb16)
                k.wg_sync(barrier_id=10)
                fence_pub(10)
                k.mbarrier_arrive(bars["ainv_ready"])
                k.scalar_store(gc, gc + 1)  # advance pipeline state

    # ============== compute group 1 (warps 4-7): new_v, qkv, kv_update ==============.
    with k.if_warpgroup(1):
        # fi's register budgets (CK:241-243): CG1 256.
        k.set_maxnreg(256)
        tid = k.tid_in_wg()
        lane = tid % 32
        warp = tid // 32
        gc = k.scalar(initial=0)  # every-chunk pipeline state (across tiles)
        gc_pos = k.scalar(initial=0)  # GEMM3/4 pipeline state (chunk>0 only)
        with k.for_each_task(sched) as task:
            seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
            k.reg_fill(zrow, 0.0)  # per-tile state reset: S_prev = 0 (each sequence)
            k.tcgen05_st(s_tmem.at(0, 0), zrow, num=V_DIM)
            k.tcgen05_wait_st()
            with k.for_loop(stop=NCH) as c:
                with k.if_(gc > 0):
                    k.mbarrier_wait(bars["chunk_free"], phase=ph(gc - 1))
                k.mbarrier_wait(bars["gate_ready1"], phase=ph(gc))
                k.mbarrier_wait(bars["tv"], phase=ph(gc))  # CG1 reads v_s (delta)
                k.reg_load(glast, _sl(gcs_s, (BT - 1, eh), (1, 1)))
                # Wave-3 C1: per-row gate factors ONCE per chunk.
                for va in range(2):
                    row = (lane // 4) + 8 * va + 16 * warp
                    k.reg_load(r2, _sl(gcs_s, (row, eh), (1, 1)))
                    k.reg_unary(_sl(dexp2, (va,), (1,)), r2, op="exp2")
                    k.reg_sub(r1, glast, r2)
                    k.reg_unary(_sl(kgate2, (va,), (1,)), r1, op="exp2")
                # delta operand (deltaT -> tmpt_s) + o_inter (-> out_s).
                with k.if_(c > 0):
                    # s_s = fp16 SMEM copy of the UNDECAYED S_prev, the GEMM3/4 B operand.
                    k.reg_unary(r1, glast, op="exp2")  # Phi = exp2(glast) (gcs log2-units)
                    for half in range(2):
                        k.tcgen05_ld(frag32, s_tmem.at(0, half * 64), shape="32x32b", num=64)
                        k.tcgen05_wait_ld()
                        # Apply the chunk gate to the whole 64-element row at once.
                        k.reg_cvt(_sl(sinp_reg, (0,), (64,)), _sl(frag32, (0,), (64,)))
                        k.reg_store(
                            _sl(s_s, (tid, half * 64), (1, 64)), _sl(sinp_reg, (0,), (64,))
                        )  # undecayed fp16 S_prev[k=tid, half*64:+64] (warpgroup row store)
                        k.reg_mul(_sl(frag32, (0,), (64,)), _sl(frag32, (0,), (64,)), r1)
                        k.tcgen05_st(s_tmem.at(0, half * 64), frag32, num=64)  # decayed main state
                    k.tcgen05_wait_st()
                    # publish the generic s_s stores to the MMA's async proxy.
                    k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
                    k.wg_sync(barrier_id=11)
                    k.mbarrier_arrive(bars["sT_ready"])  # s_s ready for GEMM3/4
                    k.mbarrier_wait(bars["d_ks"], phase=ph(gc_pos))
                    _read128_delta(
                        k,
                        acc_tmem,
                        tmpt_s,
                        v_s,
                        gcs_s,
                        frag,
                        rb16b,
                        sttile,
                        r1,
                        lane,
                        warp,
                        eh,
                        dexp2,
                        v_frag,
                    )
                    k.wg_sync(barrier_id=11)
                    k.mbarrier_arrive(bars["f_ks"])
                    k.mbarrier_wait(bars["d_qs"], phase=ph(gc_pos))
                    _read128_ointer(
                        k,
                        qstate_tmem,
                        out_s,
                        gcs_s,
                        scale,
                        frag,
                        r1,
                        sttile,
                        lane,
                        warp,
                        eh,
                        dexp2,
                        frag32,
                    )
                    k.wg_sync(barrier_id=11)
                    k.mbarrier_arrive(bars["f_qs"])
                with k.if_(c.eq(0)):
                    # chunk 0: S=0 → o_inter=0.
                    k.reg_fill(_sl(frag32, (0,), (64,)), 0.0)
                    k.wg_sync(barrier_id=11)
                fence_pub(11)
                k.mbarrier_arrive(bars["delta_ready"])
                # new_v_epi: VNEW -> vnewt_s (ungated) + tmpt_s (kgate-scaled, for GEMM7).
                k.mbarrier_wait(bars["d_nv"], phase=ph(gc))
                _read128_vnew(
                    k,
                    acc_tmem,
                    vnewt_s,
                    tmpt_s,
                    gcs_s,
                    glast,
                    frag,
                    sttile,
                    sttile2,
                    r1,
                    lane,
                    warp,
                    eh,
                    kgate2,
                )
                k.wg_sync(barrier_id=11)
                k.mbarrier_arrive(bars["f_nv"])
                # NV (vnewt_s) is GEMM6's B and vnew_gated (tmpt_s) GEMM7's B.
                fence_pub(11)
                k.mbarrier_arrive(bars["vnew_ready"])
                # GEMM7 reads K directly (A=Kᵀ via the MMA transpose) and tmpt_s (vnew_gated, B).
                k.mbarrier_arrive(bars["ktvng_ready"])
                # qkv_epilogue: O_intra -> o = o_inter + O_intra -> out_s.
                k.mbarrier_wait(bars["d_oi"], phase=ph(gc))
                _read128_store_out(k, acc_tmem, out_s, frag, rb16, sttile, lane, warp, frag32)
                k.wg_sync(barrier_id=11)
                k.mbarrier_arrive(bars["f_oi"])
                fence_pub(11)
                k.mbarrier_arrive(bars["o_ready"])
                # kv_update_epi: wait GEMM7 (dS into S) — S now holds this chunk's new state.
                k.mbarrier_wait(bars["d_ds"], phase=ph(gc))
                k.mbarrier_arrive(bars["chunk_free"])
                k.scalar_store(gc, gc + 1)
                with k.if_(c > 0):
                    k.scalar_store(gc_pos, gc_pos + 1)
            # store the FINAL state S -> state_g ONCE, after the chunk loop, via scalar reg->GMEM.
            work = task.field("work")
            seq = work // NEFF
            eh = work % NEFF
            for half in range(2):
                k.tcgen05_ld(frag32, s_tmem.at(0, half * 64), shape="32x32b", num=64)
                k.tcgen05_wait_ld()
                k.reg_store(_sl(state_g, (seq, eh, tid, half * 64), (1, 1, 1, 64)), frag32)

    # ============== epilogue warp (11): output store (UTMASTG) ==============.
    with k.if_warp(EPI_WARP):
        k.set_maxnreg(24)  # fi's support-warp budget (CK:241-243)
        if not config.varlen:
            with k.if_elected():
                gc = k.scalar(initial=0)  # cumulative chunk index (pipeline state across tiles)
                with k.for_each_task(sched) as task:
                    seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
                    with k.for_loop(stop=NCH) as c:  # runtime chunk loop
                        k.mbarrier_wait(bars["o_ready"], phase=gc % 2)
                        gtok = tok_base + c * BT
                        k.tma_store(
                            out_g,
                            _sl(out_s, (0, 0), (BT, V_DIM)),
                            coords=(gtok, eh, 0),
                            shape=(BT, V_DIM),
                            gmem_shape=(BT, 1, V_DIM),
                        )
                        k.cp_async_bulk_commit_group()
                        k.cp_async_bulk_wait_group_read(0)
                        k.mbarrier_arrive(
                            bars["chunk_free"]
                        )  # out_s freed for next chunk's o_inter
                        k.scalar_store(gc, gc + 1)
        else:
            # varlen: the partial last chunk's OOB rows must NOT be stored.
            gc = k.scalar(initial=0)
            with k.for_each_task(sched) as task:
                seq, eh, q_head, k_head, v_head, tok_base, NCH, slen = task_geom(task)
                lane = k.lane_id()
                with k.for_loop(stop=NCH) as c:
                    k.mbarrier_wait(bars["o_ready"], phase=gc % 2)
                    gtok = tok_base + c * BT
                    for half in range(2):
                        row = lane + half * 32
                        with k.if_((c * BT + row) < slen):  # valid token only
                            for dv in range(V_DIM):
                                k.reg_load(rb16, _sl(out_s, (row, dv), (1, 1)))
                                k.reg_store(_sl(out_g, (gtok + row, eh, dv), (1, 1, 1)), rb16)
                    k.warp_sync()
                    with k.if_(lane.eq(0)):
                        k.mbarrier_arrive(bars["chunk_free"])
                    k.scalar_store(gc, gc + 1)


def _read128(k, acc, frag):
    # yields (row, col, r) for a 64×128 acc read via two tcgen05.ld.16x256b blocks.
    for blk in range(2):
        k.tcgen05_ld(frag, acc.at(0, blk * 64), shape="16x256b", num=8)
        k.tcgen05_wait_ld()
        for va in range(2):
            for vb in range(8):
                for v0p in range(2):
                    yield blk, va, vb, v0p


def _stm(k, dst, row_base, col_base, sttile, lane, trans):
    # stmatrix one m8n8 tile (sttile = 2 bf16 = 1 word) → dst.
    if trans:  # transposed store dst[col, row] (stmatrix.trans writes row=mma_col,col=mma_row)
        k.stmatrix(_sl(dst, (col_base + lane % 8, row_base), (1, 8)), sttile, num=1, trans=True)
    else:
        k.stmatrix(_sl(dst, (row_base + lane % 8, col_base), (1, 8)), sttile, num=1, trans=False)


def _read128_store_out(k, acc, dst_s, frag, rb16, sttile, lane, warp, o_frag):
    # o[row,col] = O_intra[row,col] + o_inter.
    for blk, va, vb, v0p in _read128(k, acc, frag):
        r = v0p + 2 * va + 4 * vb
        k.reg_cvt(rb16, _sl(frag, (r,), (1,)))  # O_intra → bf16
        k.reg_cvt(_sl(sttile, (v0p,), (1,)), _sl(o_frag, (32 * blk + r,), (1,)))  # o_inter → bf16
        k.reg_add(_sl(sttile, (v0p,), (1,)), _sl(sttile, (v0p,), (1,)), rb16)
        if v0p == 1:
            _stm(k, dst_s, 8 * va + 16 * warp, blk * 64 + 8 * vb, sttile, lane, trans=False)


def _read128_delta(
    k, acc, tmpt_s, v_s, gcs_s, frag, rb16b, sttile, r1, lane, warp, eh, dexp2, v_frag
):
    # delta[i,dv] = v[i,dv] - exp2(gcs[i])·KH[i,dv]; store deltaᵀ → tmpt_s[dv,i] (stmatrix.trans).
    for blk, va, vb, v0p in _read128(k, acc, frag):
        r = v0p + 2 * va + 4 * vb
        if vb == 0 and v0p == 0:
            for half in range(2):
                k.ldmatrix(
                    _sl(v_frag, (32 * blk + 16 * va + 8 * half,), (8,)),
                    _sl(
                        v_s,
                        (16 * warp + 8 * va + lane % 8, blk * 64 + 32 * half + 8 * (lane // 8)),
                        (1, 8),
                    ),
                    num=4,
                    trans=False,
                )
        k.reg_mul(r1, _sl(dexp2, (va,), (1,)), _sl(frag, (r,), (1,)))  # dexp·KH (f32)
        k.reg_cvt(rb16b, r1)  # → bf16
        k.reg_sub(
            _sl(sttile, (v0p,), (1,)),
            _sl(v_frag, (32 * blk + 16 * va + 2 * vb + v0p,), (1,)),
            rb16b,
        )  # delta = v - dexp·KH (bf16)
        if v0p == 1:
            _stm(k, tmpt_s, 8 * va + 16 * warp, blk * 64 + 8 * vb, sttile, lane, trans=True)


def _read128_ointer(k, acc, out_s, gcs_s, scale, frag, r1, sttile, lane, warp, eh, dexp2, o_frag):
    # o_inter[i,dv] = exp2(gcs[i])·scale·QS[i,dv]; dexp[i] from the dexp2 fragment.
    for blk, va, vb, v0p in _read128(k, acc, frag):
        r = v0p + 2 * va + 4 * vb
        k.reg_mul(r1, _sl(dexp2, (va,), (1,)), _sl(frag, (r,), (1,)))
        k.reg_mul(r1, r1, float(scale))
        k.reg_store(_sl(o_frag, (32 * blk + r,), (1,)), r1)  # o_inter (f32) -> fragment


def _read128_vnew(
    k, acc, vnewt_s, vng_s, gcs_s, glast, frag, sttile, sttile2, r1, lane, warp, eh, kgate2
):
    # VNEW readback → vnewt_s[dv,i]=vnew.
    for blk, va, vb, v0p in _read128(k, acc, frag):
        r = v0p + 2 * va + 4 * vb
        k.reg_cvt(_sl(sttile, (v0p,), (1,)), _sl(frag, (r,), (1,)))  # ungated
        k.reg_mul(r1, _sl(kgate2, (va,), (1,)), _sl(frag, (r,), (1,)))  # · vnew
        k.reg_cvt(_sl(sttile2, (v0p,), (1,)), r1)  # gated
        if v0p == 1:
            rb = 8 * va + 16 * warp
            cb = blk * 64 + 8 * vb
            _stm(k, vnewt_s, rb, cb, sttile, lane, trans=True)
            _stm(k, vng_s, rb, cb, sttile2, lane, trans=True)


def _validate_config(config: GdnPrefillConfig) -> None:
    if config.num_seqs < 1 or config.seqlen < 1:
        raise ValueError("gdn_prefill num_seqs/seqlen must be positive")
    hq, hv = config.num_q_heads, config.num_v_heads
    if hq < 1 or hv < 1 or max(hq, hv) % min(hq, hv) != 0:
        raise ValueError(
            "gdn_prefill needs num_q_heads,num_v_heads >= 1 with max a multiple of min "
            f"(GVA h_v≥h_q or GQA h_q>h_v); got num_q={hq}, num_v={hv}."
        )
    if not config.varlen and config.seqlen % BT != 0:
        raise ValueError(f"gdn_prefill fixed-length requires seqlen a multiple of {BT}")
    # Varlen (cu_seqlens) path.


# Bench-suite interface (see bench/nymph_bench_guide.md).

KERNEL_META = {"name": "nymph_gdn_prefill", "category": "experimental", "compute_capability": 10}

# Representative shapes, picked from the coverage CONFIGS / VARLEN_CONFIGS above.
BENCH_CONFIGS = CONFIGS

_CORRECTNESS_COSINE = 0.999


def _bench_seqlens(num_seqs, seqlen, seqlens) -> list[int]:
    """Normalize the two config forms (fixed / varlen) to per-sequence lengths."""
    if seqlens is not None:
        if num_seqs is not None or seqlen is not None:
            raise ValueError("pass either seqlens or num_seqs+seqlen, not both")
        return [int(s) for s in seqlens]
    if num_seqs is None or seqlen is None:
        raise ValueError("pass num_seqs+seqlen (fixed-length) or seqlens (varlen)")
    return [int(seqlen)] * int(num_seqs)


def _bench_inputs(
    seq_lens: list[int],
    io_dtype: str = "bfloat16",
    seed: int = 0,
    num_q_heads: int = HQK,
    num_v_heads: int = HV,
) -> dict:
    """One input set shared by BOTH impls, on GPU, in the sim tests' distributions."""
    import torch

    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}[io_dtype]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = sum(seq_lens)
    h_k = min(num_q_heads, num_v_heads)
    neff = max(num_q_heads, num_v_heads)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)

    q = (randn(total, num_q_heads, K_DIM) * 0.2).to(dt)
    k = (randn(total, h_k, K_DIM) * 0.2).to(dt)
    v = (randn(total, num_v_heads, K_DIM) * 0.2).to(dt)
    gate = torch.exp(torch.nn.functional.softplus(randn(total, neff)) * -0.3)
    beta = torch.sigmoid(randn(total, neff))
    cu = torch.tensor([0, *itertools.accumulate(seq_lens)], dtype=torch.int32, device="cuda")
    return {"q": q, "k": k, "v": v, "gate": gate, "beta": beta, "cu": cu, "total": total}


def _flashinfer_callable(data: dict, num_seqs: int, neff: int = HV):
    """The flashinfer CuTeDSL baseline as a pure-launch closure + its outputs."""
    import torch
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    io_dt = data["q"].dtype
    out = torch.full((data["total"], neff, K_DIM), float("nan"), dtype=io_dt, device="cuda")
    state = torch.full(
        (num_seqs, neff, K_DIM, K_DIM), float("nan"), dtype=torch.float32, device="cuda"
    )
    scale = GdnPrefillConfig().scale

    def run():
        chunk_gated_delta_rule(
            data["q"], data["k"], data["v"], data["gate"], data["beta"], scale,
            None, True, data["cu"], False,
            output=out, output_state=state, use_cp=False,
        )  # fmt: skip

    run()  # CuTeDSL compile (first call) + warm, outside timing
    torch.cuda.synchronize()
    return run, out, state


def _compile_nymph(
    seq_lens: list[int], io_dtype: str = "bfloat16", num_q_heads: int = HQK, num_v_heads: int = HV
):
    """kernel_to_tirx_source -> tvm.compile, mirroring nvfp4's _compile_nymph."""
    import importlib.util
    import os
    import tempfile

    import tvm

    from ..nymph_rs import kernel_to_tirx_source

    if seq_lens == [seq_lens[0]] * len(seq_lens) and seq_lens[0] % BT == 0:
        config = GdnPrefillConfig(
            num_seqs=len(seq_lens),
            seqlen=seq_lens[0],
            io_dtype=io_dtype,
            num_q_heads=num_q_heads,
            num_v_heads=num_v_heads,
        )
    else:
        config = GdnPrefillConfig(
            num_seqs=len(seq_lens),
            seqlen=max(seq_lens),
            varlen=True,
            io_dtype=io_dtype,
            num_q_heads=num_q_heads,
            num_v_heads=num_v_heads,
        )
    src = kernel_to_tirx_source(build_gdn_prefill(config))
    p = os.path.join(tempfile.mkdtemp(prefix="nymph_gdn_"), "g.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nymph_gdn_emitted", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    ex = tvm.compile(tvm.IRModule({"main": m.main}), tvm.target.Target("cuda"), tir_pipeline="tirx")
    return ex, config


def _nymph_callable(ex, config: GdnPrefillConfig, data: dict, seq_lens: list[int]):
    """Pure-launch closure over preallocated nymph outputs (packed layout)."""
    import torch

    io_dt = data["q"].dtype
    num_seqs = len(seq_lens)
    neff = max(config.num_q_heads, config.num_v_heads)  # out/state/gate/beta heads
    state = torch.full(
        (num_seqs, neff, K_DIM, K_DIM), float("nan"), dtype=torch.float32, device="cuda"
    )
    if config.varlen:
        pad = num_seqs * config.seqlen
        q = torch.zeros((pad, *data["q"].shape[1:]), dtype=io_dt, device="cuda")
        k = torch.zeros_like(q)
        v = torch.zeros((pad, *data["v"].shape[1:]), dtype=io_dt, device="cuda")
        gate = torch.zeros((pad, neff), dtype=torch.float32, device="cuda")
        beta = torch.zeros_like(gate)
        cu = data["cu"].tolist()
        for a, b in itertools.pairwise(cu):
            q[a:b], k[a:b], v[a:b] = data["q"][a:b], data["k"][a:b], data["v"][a:b]
            gate[a:b], beta[a:b] = data["gate"][a:b], data["beta"][a:b]
        out = torch.full((pad, neff, K_DIM), float("nan"), dtype=io_dt, device="cuda")

        def run():
            ex(q, k, v, gate, beta, out, state, data["cu"])

    else:
        out = torch.full((data["total"], neff, K_DIM), float("nan"), dtype=io_dt, device="cuda")

        def run():
            ex(data["q"], data["k"], data["v"], data["gate"], data["beta"], out, state)

    return run, out, state


def _cosine(a, b) -> float:
    import torch

    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def run_bench(
    num_seqs=None,
    seqlen=None,
    seqlens=None,
    num_q_heads=HQK,
    num_v_heads=HV,
    *,
    warmup=None,
    repeat=None,
    timer=None,
    **kwargs,
):
    """Bench the flashinfer CuTeDSL baseline vs nymph with the bench-suite's exact methodology."""
    import torch

    from tvm.tirx.bench import bench

    seq_lens = _bench_seqlens(num_seqs, seqlen, seqlens)
    ex, config = _compile_nymph(seq_lens, num_q_heads=num_q_heads, num_v_heads=num_v_heads)
    data = _bench_inputs(
        seq_lens, config.io_dtype, num_q_heads=num_q_heads, num_v_heads=num_v_heads
    )
    nymph_run, n_out, n_state = _nymph_callable(ex, config, data, seq_lens)
    fi_run, f_out, f_state = _flashinfer_callable(
        data, len(seq_lens), max(num_q_heads, num_v_heads)
    )

    nymph_run()
    fi_run()
    torch.cuda.synchronize()
    total = data["total"]
    cos_o = _cosine(n_out[:total], f_out)
    cos_s = _cosine(n_state, f_state.transpose(-1, -2))
    if min(cos_o, cos_s) < _CORRECTNESS_COSINE:
        raise AssertionError(
            f"nymph gdn_prefill diverges from flashinfer "
            f"(cos_out={cos_o:.4f} cos_state={cos_s:.4f}, need >= {_CORRECTNESS_COSINE})"
        )
    return bench(
        {"tir": fi_run, "tirx": nymph_run}, warmup=warmup, repeat=repeat, timer=timer, **kwargs
    )


def run_flashinfer_bench(
    num_seqs=None, seqlen=None, seqlens=None, *, warmup=None, repeat=None, timer=None, **kwargs
):
    """Baseline-only runner."""
    from tvm.tirx.bench import bench

    seq_lens = _bench_seqlens(num_seqs, seqlen, seqlens)
    data = _bench_inputs(seq_lens)
    fi_run, _, _ = _flashinfer_callable(data, len(seq_lens))
    return bench({"flashinfer": fi_run}, warmup=warmup, repeat=repeat, timer=timer, **kwargs)


def register_bench_interface() -> None:
    """Self-register into the bench-suite kernel cache (see bench/nymph_bench_guide.md)."""
    import sys

    from tirx_kernels.registry import _KERNEL_CACHE

    _KERNEL_CACHE[KERNEL_META["name"]] = sys.modules[__name__]
