"""Flash-Attention SM100 BACKWARD (dQ/dK/dV) expressed in Nymph IR."""

# PER-WARP EXECUTION MODEL.

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import gcd as _math_gcd

from ..builder import IRBuilder
from ..nymph_rs import (
    DType,
    FenceKind,
    FenceScope,
    Kernel,
    LaunchShape,
    MBarKind,
    MemorySpace,
    TensorSlice,
    TmemTensor,
)
from ..nymph_rs import (
    max as scalar_max,  # runtime ScalarExpr max(a,b) — used to clamp m_block_min >= 0
)

# ---- architecture constants (flashattn FlashAttentionBackwardSm100 for hd64 1-CTA).
TILE_M = 128  # Q tile (m_block_size)
TILE_N = 128  # KV tile (n_block_size)
SM_COUNT = 148
N_COLS_TMEM = 512  # tcgen05 TMEM columns (sm_100)


def _dq_reduce_cfg(hdim, use_2cta, is_causal, deterministic):
    """flashattn's per-config dQ_reduce_ncol / sdQaccum_stage (flash_bwd_sm100.py:245-258)."""
    if use_2cta and hdim == 192:
        ncol = 32 if is_causal else 24
        nslot = 1 if is_causal else 2
    elif use_2cta:
        ncol = 16 if deterministic else 8
        nslot = 2 if deterministic else 4
    else:
        ncol = 32
        nslot = 64 // ncol  # = 2
    return ncol, nslot


# 16-warp role IDs (FlashAttentionBackwardSm100.__init__:141-146).
REDUCE_WARPS = (0, 1, 2, 3)
COMPUTE_WARPS = (4, 5, 6, 7, 8, 9, 10, 11)
MMA_WARP = 12
LOAD_WARP = 13
RELAY_WARP = 14
EMPTY_WARP = 15
NUM_WARPS = 16

LOG2_E = 1.4426950408889634  # softmax_scale_log2 = softmax_scale · log2(e) (I3: exp2 not exp)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sl(t, offs, shape):
    return TensorSlice(tensor=t, offsets=offs, shape=shape)


def _mma_a(k: IRBuilder, op, offs, shape):
    """Build the explicitly tagged A operand used by one physical MMA."""
    if isinstance(op, TmemTensor):
        return k.mma_a_tmem(op.at(offs[0], offs[1] // 2), form="flat")
    return k.mma_a_smem(
        k.smem_tile(
            op,
            prefix_indices=(),
            row_offset=offs[0],
            col_offset=offs[1],
            rows=shape[0],
            cols=shape[1],
        )
    )


def _mma_b(k: IRBuilder, op, offs, shape):
    """Build the explicit SMEM B tile used by one physical MMA."""
    return k.smem_tile(
        op, prefix_indices=(), row_offset=offs[0], col_offset=offs[1], rows=shape[0], cols=shape[1]
    )


_VALID_TC = (128, 64, 32, 16, 8, 4, 2, 1)  # legal 32x32b .x{N} counts (tcgen05_datapath.rs:61)


def _tc_chunks(num: int) -> list[int]:
    """Decompose a tcgen05.ld column count into valid 32x32b .x{N} pieces, largest-first."""
    out, r = [], num
    for v in _VALID_TC:
        while r >= v:
            out.append(v)
            r -= v
    assert r == 0, f"cannot tile tcgen05 num={num}"
    return out


def _tc_ld(k, frag, src, num, col=0, base=0, row=0):
    """t2r read of `num` f32 TMEM cols."""
    off = 0
    for c in _tc_chunks(num):
        k.tcgen05_ld(_sl(frag, (base + off,), (c,)), src.at(row, col + off), shape="32x32b", num=c)
        off += c


@dataclass(frozen=True, slots=True)
class FlashBwdSm100Config:
    # Workload geometry.
    batch: int = 1
    seqlen_q: int = 128
    seqlen_k: int = 128
    num_head: int = 1  # num_head_q
    num_head_kv: int | None = (
        None  # GQA: #kv-heads. None => == num_head (MHA, G=1). G = num_head // num_head_kv.
    )
    head_dim: int = 64  # <128 -> 1-CTA path
    head_dim_v: int | None = None
    is_causal: bool = False
    scale: float | None = None  # default 1/sqrt(head_dim)
    launch_shape: LaunchShape | None = None
    # VARLEN (cu_seqlens packed variable-length sequences).
    varlen: bool = False
    # DETERMINISTIC (bit-reproducible dQ / GQA-dKV accumulation).
    deterministic: bool = False

    @property
    def head_dim_v_(self) -> int:
        return self.head_dim_v if self.head_dim_v is not None else self.head_dim

    @property
    def num_head_kv_(self) -> int:
        # #kv-heads. Default (None) => MHA (== num_head, G=1).
        return self.num_head_kv if self.num_head_kv is not None else self.num_head

    @property
    def gqa_group(self) -> int:
        # G = qhead_per_kvhead = num_q_heads // num_kv_heads. G==1 => MHA (byte-identical IR).
        assert self.num_head % self.num_head_kv_ == 0, "num_head must be divisible by num_head_kv"
        return self.num_head // self.num_head_kv_

    @property
    def softmax_scale(self) -> float:
        return self.scale if self.scale is not None else (self.head_dim**-0.5)


def _tile_hdim(d: int) -> int:
    return _ceil_div(d, 16) * 16  # pad to k_block_size (multiple of 16)


def _use_2cta(config: FlashBwdSm100Config) -> bool:
    # CONFIG MAP (interface.py:1332): head_dim>=128 -> 2-CTA (cluster_size=2). hd<128 -> 1-CTA.
    return config.head_dim >= 128


def _derived(config: FlashBwdSm100Config) -> dict:
    """Compile-time derived geometry (mirrors __init__ / _setup_attributes)."""
    tile_hdim = _tile_hdim(config.head_dim)
    tile_hdimv = _tile_hdim(config.head_dim_v_)
    # __init__:82 — hd<=128 (symmetric) or the (192,128) DeepSeek shape.
    assert tile_hdim <= 128 or (tile_hdim == 192 and tile_hdimv == 128), (
        f"unsupported head_dim/head_dim_v = {tile_hdim}/{tile_hdimv}"
    )
    use_2cta = _use_2cta(config)
    is_hd192 = tile_hdim == 192 and tile_hdimv == 128  # DeepSeek (2-CTA, dedicated TMEM layout)
    assert not is_hd192 or use_2cta, "hd192 must use 2-CTA (__init__:94)"
    cg = 2 if use_2cta else 1  # cta_group_size
    # MMA tilers (M, N, K).
    d = dict(
        tile_hdim=tile_hdim,
        tile_hdimv=tile_hdimv,
        cta_group=cg,
        use_2cta=use_2cta,
        is_hd192=is_hd192,
        mma_kq=(cg * TILE_N, TILE_M, tile_hdim),  # S  = K·Qᵀ
        mma_vdo=(cg * TILE_N, TILE_M, tile_hdimv),  # dP = V·dOᵀ
        mma_pdo=(cg * TILE_N, tile_hdimv, TILE_M),  # dV = Pᵀ·dO
        mma_dsq=(cg * TILE_N, tile_hdim, TILE_M),  # dK = dSᵀ·Q
        mma_dsk=(TILE_M, tile_hdim, cg * TILE_N),  # dQ = dS·K  (K cluster-wide)
    )
    if is_hd192:
        # hd192 DeepSeek dedicated TMEM layout.
        d["tmem_dV"] = 0
        d["tmem_dK"] = d["tmem_dV"] + tile_hdimv  # 128
        d["tmem_S"] = d["tmem_dK"] + tile_hdim  # 320
        d["tmem_P"] = d["tmem_S"]  # overlaps S
        d["tmem_dP"] = N_COLS_TMEM - TILE_M  # 384
        d["tmem_dS"] = d["tmem_dP"]  # overlaps dP
        d["tmem_dQ"] = N_COLS_TMEM - tile_hdim // 2  # 416 (half-hdim per CTA; ends at 512)
    elif use_2cta:
        # 2-CTA TMEM offsets — __init__:194-204 (general 2-CTA path; map §4).
        d["tmem_S"] = 0
        d["tmem_P"] = 0  # overlaps S
        d["tmem_dV"] = TILE_N  # 128
        d["tmem_dP"] = d["tmem_dV"] + tile_hdimv  # 256
        d["tmem_dQ"] = d["tmem_S"] + tile_hdim // 2  # 64 (2-CTA-specific; overlaps S)
        d["tmem_dK"] = d["tmem_dP"] + TILE_M  # 384
        d["tmem_dS"] = d["tmem_dP"]  # overlaps dP (256)
    else:
        # TMEM offsets — __init__:193-204 (general 1-CTA path; S/P, dS/dP, dQ/dP overlap).
        d["tmem_S"] = 0
        d["tmem_P"] = 0  # overlaps S
        d["tmem_dV"] = d["tmem_S"] + TILE_N  # 128
        d["tmem_dP"] = d["tmem_dV"] + tile_hdimv  # 128 + hdimv
        d["tmem_dQ"] = d["tmem_dP"]  # 1-CTA: dQ overlaps dP
        d["tmem_dK"] = d["tmem_dP"] + TILE_M  # dP + 128
        d["tmem_dS"] = d["tmem_dP"]  # overlaps dP
    # capacity: every accumulator's column extent must fit the 512-col TMEM (sm_100).
    extent = max(
        d["tmem_S"] + TILE_M,
        d["tmem_dV"] + tile_hdimv,
        d["tmem_dP"] + TILE_M,
        d["tmem_dK"] + tile_hdim,
        d["tmem_dQ"] + (tile_hdim // cg),
    )
    assert extent <= N_COLS_TMEM, f"TMEM overflow: extent {extent} > {N_COLS_TMEM}"
    return d


# Representative aligned shapes; the first entry is the milestone-1 kernel.
CONFIGS = [
    {
        "batch": 1,
        "num_head": 1,
        "seqlen_q": 128,
        "seqlen_k": 128,
        "head_dim": 64,
        "label": "b1h1s128_hd64",
    },
    {
        "batch": 1,
        "num_head": 2,
        "seqlen_q": 256,
        "seqlen_k": 256,
        "head_dim": 64,
        "label": "b1h2s256_hd64",
    },
    {
        "batch": 2,
        "num_head": 2,
        "seqlen_q": 256,
        "seqlen_k": 128,
        "head_dim": 64,
        "label": "b2h2_sq256sk128_hd64",
    },
]

# hd128 2-CTA configs (head_dim>=128 -> cluster_size=2).
CONFIGS_2CTA = [
    {
        "batch": 1,
        "num_head": 1,
        "seqlen_q": 128,
        "seqlen_k": 256,
        "head_dim": 128,
        "label": "b1h1_sq128sk256_hd128",
    }
]


def build_flash_bwd_sm100(config: FlashBwdSm100Config = FlashBwdSm100Config()) -> Kernel:
    """FA-bwd kernel (milestone-1: hd64 1-CTA non-causal dense). 16-warp specialized."""
    d = _derived(config)
    use_2cta = d["use_2cta"]
    is_hd192 = d["is_hd192"]  # DeepSeek (192,128): dedicated TMEM layout + S→dP→dK→dV→dQ order
    cg = d["cta_group"]
    B, H = config.batch, config.num_head
    Hkv = config.num_head_kv_  # #kv-heads (== H for MHA)
    G = config.gqa_group  # qhead_per_kvhead = H // Hkv (1 for MHA)
    gqa = G > 1  # GQA gate; G==1 path stays byte-identical
    # varlen-GQA must route dK/dV through the head_kv reduce-add.
    Sq, Sk = config.seqlen_q, config.seqlen_k
    Dq, Dv = config.head_dim, config.head_dim_v_
    hdim, hdimv = d["tile_hdim"], d["tile_hdimv"]
    Tq, Tk = B * Sq, B * Sk
    n_mb = _ceil_div(Sq, TILE_M)  # m-blocks (Q tiles) per task
    n_nb = _ceil_div(Sk, TILE_N)  # n-blocks (KV tiles) = CTAs per (head,batch)
    real_n_nb = n_nb  # the TRUE #kv-tiles (compile-time); n_nb may grow below for the grid
    # 2-CTA: each "task" is one CLUSTER (a pair of adjacent kv-tiles).
    if use_2cta:
        if config.varlen:
            # VARLEN 2-CTA: the grid is sized over the OVER-ALLOCATED upper bound.
            n_nb = ((n_nb + cg - 1) // cg) * cg
        elif n_nb % cg != 0:
            # DENSE 2-CTA, tile-aligned-but-not-cluster-aligned Sk.
            n_nb = ((n_nb + cg - 1) // cg) * cg
    n_cluster = n_nb // cg if use_2cta else n_nb
    # DENSE 2-CTA padding-tile mode.
    dense_pad = use_2cta and not config.varlen and (real_n_nb != n_nb)
    num_work = n_cluster * H * B  # cluster tasks (each handled by a CTA pair under 2-CTA)
    scale_log2 = config.softmax_scale * LOG2_E
    iod = DType.BF16

    # n_mb is COMPILE-TIME (Python int) -> the m-loop is Python-unrolled.
    NSTAGE = 1 if use_2cta else (2 if n_mb > 1 else 1)  # operand SMEM/barrier stages
    # dQ reduce staging — flashattn's exact per-config (dQ_reduce_ncol, sdQaccum_stage).
    RDQ_NCOL, RDQ_NSLOT = _dq_reduce_cfg(hdim, use_2cta, config.is_causal, config.deterministic)
    # dK/dV epilogue CHUNK staging.
    NUM_CWG = 2  # two compute warpgroups (each owns a wg-half)
    _dkv_bytes = 4 if gqa else 2  # f32 (GQA reduce) / bf16 (MHA store)
    DK_RNCOL = _math_gcd(128 // _dkv_bytes, hdim // 2)  # GQA:32  MHA:64 (chunk width, dK)
    DV_RNCOL = _math_gcd(128 // _dkv_bytes, hdimv // 2)  # GQA:32  MHA:64 (chunk width, dV)
    # ---- VALUE-CORRECT 2-CTA B-operand split.
    TM_H = TILE_M // cg  # q half (S/dP B N-half) = 64
    HD_H = hdim // cg  # d half  (dK B N-half)  = 64
    HDV_H = hdimv // cg  # dv half (dV B N-half)  = 64
    # ---- SMEM layout (bytes): operands + epilogue staging.
    if use_2cta:
        # 2-CTA B operands carry only this CTA's N/2 half.
        sizes = dict(
            sK=TILE_N * hdim * 2,
            sV=TILE_N * hdimv * 2,
            sQ=TM_H * hdim * 2,  # S  B: q-half rows, full d (contraction)
            sdO=TILE_M * HDV_H * 2,  # dV B: full q rows (contract), dv-half cols
            # NOTE: no sdS here.
            sLSE=NSTAGE * TILE_M * 4,
            sdPsum=NSTAGE * TILE_M * 4,
            sdQ=TILE_M
            * RDQ_NCOL
            * RDQ_NSLOT
            * 4,  # dQ reduce: RDQ_NSLOT-deep double buffer of RDQ_NCOL-col slices
            # GQA: f32 (4B) staging.
            sdK=TILE_N * DK_RNCOL * NUM_CWG * (4 if gqa else 2),
            sdV=TILE_N * DV_RNCOL * NUM_CWG * (4 if gqa else 2),
        )
        # sdOt (dP B): q-half rows, full dv (contraction).
        sizes["sdOt"] = TM_H * hdimv * 2  # dP B: q-half rows, full dv (contraction)
        sizes["sQt"] = TILE_M * HD_H * 2  # dK B: full q rows (contract), d-half cols
        # dQ datapath (the dS cross-CTA exchange + dQ MMA, map §3).
        sizes["sKt"] = cg * TILE_N * HD_H * 2  # dQ B: full kv, this CTA's d-half
        sizes["sdS_full"] = cg * TILE_N * TM_H * 2  # dQ A: full kv (exchanged), this CTA's q-half
        sizes["sdS_xchg"] = TILE_N * TM_H * 2  # dS export half (this kv-tile, peer's q-half)
    else:
        sizes = dict(
            sK=TILE_N * hdim * 2,
            sV=TILE_N * hdimv * 2,
            sQ=NSTAGE * TILE_M * hdim * 2,
            sdO=NSTAGE * TILE_M * hdimv * 2,
            sdS=TILE_N * TILE_M * 2,  # dS staged bf16 for dQ/dK B-operand path
            sLSE=NSTAGE * TILE_M * 4,
            sdPsum=NSTAGE * TILE_M * 4,
            sdQ=TILE_M
            * RDQ_NCOL
            * RDQ_NSLOT
            * 4,  # dQ reduce: RDQ_NSLOT-deep double buffer of RDQ_NCOL-col slices
            # epilogue dK/dV staging. GQA: f32 (4B).
            sdK=TILE_N * DK_RNCOL * NUM_CWG * (4 if gqa else 2),
            sdV=TILE_N * DV_RNCOL * NUM_CWG * (4 if gqa else 2),
        )
    # SMEM aliasing — faithfully model flashattn's physical buffer reuse.
    aliases = {"sdK": ("sK" if use_2cta else "sQ"), "sdV": ("sV" if use_2cta else "sdO")}
    if is_hd192:
        aliases["sdS_xchg"] = "sdQ"
        # hd192: sQt_size = sdOt_size = 0 (flash_bwd_sm100.py:766).
        aliases["sdOt"] = "sQ"
        aliases["sQt"] = "sQ"
    # An alias TARGET is allocated to hold the LARGER of its own data and the aliased buffer.
    alloc = dict(sizes)
    for alias, target in aliases.items():
        alloc[target] = max(alloc[target], sizes[alias])
    off, offs = 0, {}
    for nm, nb in alloc.items():
        if nm in aliases:
            continue  # aliased below (not bump-allocated)
        offs[nm] = off
        off += nb
    for alias, target in aliases.items():
        offs[alias] = offs[target]

    # ---- mbarrier storage plan.
    WG_T = 128
    bar_spec = {
        "tk": (MBarKind.TMA, 1),
        "tv": (MBarKind.TMA, 1),
        "tq": (MBarKind.TMA, 1, NSTAGE),
        "tdo": (MBarKind.TMA, 1, NSTAGE),
        "tlse": (MBarKind.TMA, 1, NSTAGE),
        "tdps": (MBarKind.TMA, 1, NSTAGE),
        "s_ready": (MBarKind.TCGEN05, 1),
        "dp_ready": (MBarKind.TCGEN05, 1),
        "p_ready": (MBarKind.THREAD, 2 * WG_T),
        "ds_ready": (MBarKind.THREAD, 2 * WG_T),
        "dv_done": (MBarKind.TCGEN05, 1),
        "dk_done": (MBarKind.TCGEN05, 1),
        "dq_done": (MBarKind.TCGEN05, 1),
        "dq_free": (MBarKind.THREAD, WG_T),
        "q_free": (MBarKind.THREAD, 1, NSTAGE),
        "do_free": (MBarKind.THREAD, 1, NSTAGE),
        "lse_free": (MBarKind.THREAD, 2 * WG_T, NSTAGE),
        "dps_free": (MBarKind.THREAD, 2 * WG_T, NSTAGE),
    }
    if use_2cta:
        bar_spec["tqt"] = (MBarKind.TMA, 1, NSTAGE)
        bar_spec["tdot"] = (MBarKind.TMA, 1, NSTAGE)
        bar_spec["tkt"] = (MBarKind.TMA, 1)
        bar_spec["qt_free"] = (MBarKind.THREAD, 1, NSTAGE)
        bar_spec["dot_free"] = (MBarKind.THREAD, 1, NSTAGE)
        bar_spec["dS_cluster_full"] = (MBarKind.TMA, 1)
        bar_spec["dS_cluster_leader"] = (MBarKind.THREAD, 2)
        bar_spec["dS_free"] = (MBarKind.THREAD, 1)
        if not is_hd192 and n_mb > 1:
            bar_spec["s_cons"] = (MBarKind.THREAD, 2 * WG_T)
        if is_hd192:
            bar_spec["s_free"] = (MBarKind.THREAD, 2 * WG_T)
            bar_spec["dQaccum_empty"] = (MBarKind.THREAD, WG_T)

    cursor = _align(off, 8)
    bar_offsets = {}
    for name, spec in bar_spec.items():
        bar_offsets[name] = cursor
        cursor += (spec[2] if len(spec) > 2 else 1) * 8
    tmem_addr_offset = _align(cursor, 4)
    smem_size_bytes = tmem_addr_offset + 4

    # INVARIANT-OVERRIDE I1a (F11).
    if use_2cta:
        default_launch = (max(cg, (min(SM_COUNT, num_work * cg) // cg) * cg),)
    else:
        default_launch = (min(SM_COUNT, num_work),)
    k = IRBuilder(
        "flash_bwd_sm100",
        num_warps=NUM_WARPS,
        smem_size_bytes=smem_size_bytes,
        launch_shape=config.launch_shape or default_launch,
        cluster_shape=(cg,) if use_2cta else (1,),
    )

    # ---- args (token-major [T, H, D]).
    dkv_dtype = DType.F32 if gqa else iod
    q_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(Tq, H, Dq))
    k_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(Tk, Hkv, Dq))
    v_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(Tk, Hkv, Dv))
    do_g = k.arg(space=MemorySpace.GMEM, dtype=iod, shape=(Tq, H, Dv))
    lse_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(Tq, H))
    dpsum_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(Tq, H))
    dq_g = k.arg(space=MemorySpace.GMEM, dtype=DType.F32, shape=(Tq, H, Dq))  # dQaccum (reduce-add)
    dk_g = k.arg(space=MemorySpace.GMEM, dtype=dkv_dtype, shape=(Tk, Hkv, Dq))
    dv_g = k.arg(space=MemorySpace.GMEM, dtype=dkv_dtype, shape=(Tk, Hkv, Dv))
    # VARLEN (C2): cu_seqlens for Q and K, shape (B+1,), i32.
    if config.varlen:
        cu_q_g = k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(B + 1,))
        cu_k_g = k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(B + 1,))
    else:
        cu_q_g = cu_k_g = None

    # DETERMINISTIC semaphores (appended LAST, after cu_*).
    deterministic = config.deterministic
    n_mb_sem = _ceil_div(Sq, TILE_M)
    n_nb_sem = _ceil_div(Sk, TILE_N)
    if deterministic:
        dq_sem_g = k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(B, H, n_mb_sem, cg))
        if gqa:
            dk_sem_g = k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(B, Hkv, n_nb_sem, 2))
            dv_sem_g = k.arg(space=MemorySpace.GMEM, dtype=DType.I32, shape=(B, Hkv, n_nb_sem, 2))
        else:
            dk_sem_g = dv_sem_g = None
    else:
        dq_sem_g = dk_sem_g = dv_sem_g = None

    def sm(name, dt, shape):
        return k.tensor(space=MemorySpace.SMEM, dtype=dt, shape=shape, byte_offset=offs[name])

    sK = sm("sK", iod, (TILE_N, hdim))
    sV = sm("sV", iod, (TILE_N, hdimv))
    if use_2cta:
        # VALUE-CORRECT half-sized B operands (each CTA holds its own N/2 half).
        sQ = sm("sQ", iod, (TM_H, hdim))
        sdO = sm("sdO", iod, (TILE_M, HDV_H))
    else:
        # double-buffered operands: stage row = (mb%2)*TILE_M / element offset (mb%2)*TILE_M.
        sQ = sm("sQ", iod, (NSTAGE * TILE_M, hdim))
        sdO = sm("sdO", iod, (NSTAGE * TILE_M, hdimv))
    sLSE = sm("sLSE", DType.F32, (NSTAGE * TILE_M,))
    sdPsum = sm("sdPsum", DType.F32, (NSTAGE * TILE_M,))
    sdQ = sm("sdQ", DType.F32, (TILE_M, RDQ_NCOL * RDQ_NSLOT))  # RDQ_NSLOT-deep reduce buffer
    # GQA epilogue staging is f32 (reduce-add into the shared kv-head's f32 accumulators).
    sdkv_dtype = DType.F32 if gqa else iod
    # CHUNKED staging buffers.
    sdK = sm("sdK", sdkv_dtype, (TILE_N, DK_RNCOL * NUM_CWG))
    sdV = sm("sdV", sdkv_dtype, (TILE_N, DV_RNCOL * NUM_CWG))
    if use_2cta:
        # Transposed B operands carry only this CTA's N/2 half.
        sdOt = sm("sdOt", iod, (TM_H, hdimv))  # transposed dO (dP B-operand)
        sQt = sm("sQt", iod, (TILE_M, HD_H))  # transposed Q  (dK B-operand)
        # dQ datapath (the dS exchange): full-kv operands, this CTA's q/d half.
        sKt = sm("sKt", iod, (cg * TILE_N, HD_H))  # dQ B: full kv, this CTA's d-half
        sdS_full = sm(
            "sdS_full", iod, (cg * TILE_N, TM_H)
        )  # dQ A: full kv (exchanged), this CTA's q-half
        sdS_xchg = sm(
            "sdS_xchg", iod, (TILE_N, TM_H)
        )  # dS export half (this kv-tile, peer's q-half)
        sdS = None  # 2-CTA dQ uses sdS_full; sdS is 1-CTA-only
    else:
        sdOt = sQt = sKt = sdS_full = sdS_xchg = None
        sdS = sm("sdS", iod, (TILE_N, TILE_M))  # 1-CTA dQ A-operand (dS, trans_a)

    # ---- TMEM column plan: idless views at the derived absolute columns.
    tS = k.tmem_tensor(d["tmem_S"])
    tP = k.tmem_tensor(d["tmem_P"])
    tdV = k.tmem_tensor(d["tmem_dV"])
    tdP = k.tmem_tensor(d["tmem_dP"])
    tdQ = k.tmem_tensor(d["tmem_dQ"])
    tdK = k.tmem_tensor(d["tmem_dK"])
    tdS = k.tmem_tensor(d["tmem_dS"])

    def reg(dt, shape):
        return k.tensor(space=MemorySpace.REG, dtype=dt, shape=shape)

    # ---- mbarriers (producer/consumer pipeline) ---- Per-m-block operand TMA barriers.
    WG_T = 128  # threads per warpgroup
    bar_spec = {
        "tk": (MBarKind.TMA, 1),
        "tv": (MBarKind.TMA, 1),
        "tq": (MBarKind.TMA, 1, NSTAGE),
        "tdo": (MBarKind.TMA, 1, NSTAGE),
        "tlse": (MBarKind.TMA, 1, NSTAGE),
        "tdps": (MBarKind.TMA, 1, NSTAGE),
        "s_ready": (MBarKind.TCGEN05, 1),  # mma S -> compute
        "dp_ready": (MBarKind.TCGEN05, 1),  # mma dP -> compute
        # compute P -> mma (dV) / compute dS -> mma (dK/dQ).
        "p_ready": (MBarKind.THREAD, 2 * WG_T),
        "ds_ready": (MBarKind.THREAD, 2 * WG_T),
        "dv_done": (MBarKind.TCGEN05, 1),
        "dk_done": (MBarKind.TCGEN05, 1),
        "dq_done": (MBarKind.TCGEN05, 1),  # mma dQ -> reduce
        # reduce -> mma: tdQ[mb] read done, mma may overwrite.
        "dq_free": (MBarKind.THREAD, WG_T),
        # NEW-3/F5: operand EMPTY barriers (per-stage).
        "q_free": (MBarKind.THREAD, 1, NSTAGE),  # mma (last read = dK) -> load
        "do_free": (MBarKind.THREAD, 1, NSTAGE),  # mma (last read = dV) -> load
        "lse_free": (MBarKind.THREAD, 2 * WG_T, NSTAGE),  # both compute wgs -> load
        "dps_free": (MBarKind.THREAD, 2 * WG_T, NSTAGE),  # both compute wgs -> load
    }
    if use_2cta:
        # Transposed-operand TMA barriers.
        bar_spec["tqt"] = (MBarKind.TMA, 1, NSTAGE)  # transposed Q -> dK B
        bar_spec["tdot"] = (MBarKind.TMA, 1, NSTAGE)  # transposed dO -> dP B
        bar_spec["tkt"] = (MBarKind.TMA, 1)  # transposed K -> dQ B (sKt) — once per task
        # NEW (2-CTA single-stage operand frees, multi-Q-tile).
        bar_spec["qt_free"] = (MBarKind.THREAD, 1, NSTAGE)  # mma dK (sQt read) -> load
        bar_spec["dot_free"] = (MBarKind.THREAD, 1, NSTAGE)  # mma dP (sdOt read) -> load
        # dS cross-CTA exchange + dQ-GEMM release (map §3).
        bar_spec["dS_cluster_full"] = (MBarKind.TMA, 1)
        # both CTAs' relay streams arrive (each a single elected thread): count=2.
        bar_spec["dS_cluster_leader"] = (MBarKind.THREAD, 2)
        # dS_free (multi-Q-tile WAR): sdS_full is SINGLE-buffer.
        bar_spec["dS_free"] = (MBarKind.THREAD, 1)
        if not is_hd192 and n_mb > 1:
            # General 2-CTA multi-Q-tile.
            bar_spec["s_cons"] = (MBarKind.THREAD, 2 * WG_T)
        if is_hd192:
            # hd192 ONLY: S/dP overlap TMEM cols.
            bar_spec["s_free"] = (MBarKind.THREAD, 2 * WG_T)
            # hd192 ONLY: sdS_xchg aliases sdQ (above).
            bar_spec["dQaccum_empty"] = (MBarKind.THREAD, WG_T)
    bars = {
        nm: k.mbar(
            kind=spec[0], byte_offset=bar_offsets[nm], stages=(spec[2] if len(spec) > 2 else 1)
        )
        for nm, spec in bar_spec.items()
    }

    # ---- 2-CTA cluster gating + peer mbars.
    is_leader = k.ctaid_in_cluster().eq(0) if use_2cta else None
    if use_2cta:
        # peer-full waits for the leader cluster MMA: it reads BOTH CTAs' operands.
        _peer_names = ["tk", "tv", "tq", "tdo", "tqt", "tdot", "tkt", "p_ready", "ds_ready"]
        if is_hd192:
            # hd192: the leader's dP MMA writes BOTH CTAs' tmem_dP, overwriting the PEER's S cols.
            _peer_names.append("s_free")
            if n_mb > 1:
                # hd192 multi-Q-tile.
                _peer_names.append("dq_free")
        if not is_hd192 and n_mb > 1:
            # general 2-CTA multi-Q-tile.
            _peer_names.append("s_cons")
            _peer_names.append("dq_free")
        peer_bars = {nm: k.mbar_ref(bars[nm], remote_coord=1) for nm in _peer_names}
        # Operand-free barriers (q_free/do_free/qt_free/dot_free).
        peer_free = {
            nm: k.mbar_ref(bars[nm], remote_coord=1)
            for nm in ("q_free", "do_free", "qt_free", "dot_free", "dS_free")
        }
    else:
        peer_bars = {}
        peer_free = {}

    sched = k.scheduler(k.task_space(grid=(num_work,), fields=("work",)))

    cta_in_cluster = k.ctaid_in_cluster() if use_2cta else 0

    with k.if_warp(0):
        # tmem_alloc is warp-collective (full warp 0).
        k.tmem_alloc(0, N_COLS_TMEM, addr_byte_offset=tmem_addr_offset, cta_group=cg)
        with k.if_elected():
            for nm, spec in bar_spec.items():
                stg = spec[2] if len(spec) > 2 else 1
                for s in range(stg):
                    k.mbarrier_init(bars[nm], count=spec[1], stage=s)
    # This sync IS the prologue ordering.
    if use_2cta:
        k.cluster_sync()
    else:
        k.cta_sync()

    def task_geom(task):
        # 1-CTA: `work` indexes (nb, head, batch) directly.
        work = task.field("work")
        c_nb = work % n_cluster
        hb = work // n_cluster
        head = hb % H
        batch = hb // H
        if use_2cta:
            nb = c_nb * cg + cta_in_cluster  # this CTA's own kv-tile
        else:
            nb = c_nb
        return batch, head, nb

    # ---- CAUSAL m_block_min tile-skip.
    OFF = Sk - Sq
    causal_skip = config.is_causal and not use_2cta

    @contextmanager
    def _discard():
        # swallow a guarded body whose compile-time predicate is False (emit nothing).
        scratch: list = []
        k._body_stack.append(scratch)
        try:
            yield
        finally:
            k._body_stack.pop()

    def _guard(pred):
        # pred is a Python bool (compile-time) or a ScalarValue (runtime).
        if isinstance(pred, (bool, int)):
            return nullcontext() if pred else _discard()
        return k.if_(pred)

    def m_block_min(nb, vg=None):
        # mmin = max(0, (nb*TILE_N - off)//TILE_M) when causal-skipping, else Python int 0.
        if not causal_skip:
            return 0
        off = vg.off_b if (varlen and vg is not None) else OFF
        mmin = (nb * TILE_N - off) // TILE_M
        # Preserve the compile-time int path (byte-identical IR) when mmin folded to a literal.
        return max(0, mmin) if isinstance(mmin, int) else scalar_max(0, mmin)

    def _ge(mb, mmin):  # mb >= mmin  (Python bool if mmin is int, else ScalarValue)
        return mb >= mmin

    def _gt(mb, mmin):  # mb > mmin
        return mb > mmin

    def _eq(mb, mmin):  # mb == mmin  (Eq needs the named .eq() for a ScalarValue mmin)
        return mb == mmin if isinstance(mmin, int) else mmin.eq(mb)

    # ---- VARLEN per-sequence runtime geometry.
    varlen = config.varlen

    class _VG:  # per-task runtime varlen geometry (scalars valid inside the role's for_each_task)
        __slots__ = ("base_k", "base_q", "n_mb_b", "n_nb_b", "off_b", "slen_k", "slen_q")

    def varlen_geom(batch):
        # load the per-sequence cu_seqlens window into runtime scalars + derive tile counts.
        vg = _VG()
        vg.base_q = k.scalar(initial=_sl(cu_q_g, (batch,), (1,)))
        nxt_q = k.scalar(initial=_sl(cu_q_g, (batch + 1,), (1,)))
        vg.slen_q = nxt_q - vg.base_q
        vg.base_k = k.scalar(initial=_sl(cu_k_g, (batch,), (1,)))
        nxt_k = k.scalar(initial=_sl(cu_k_g, (batch + 1,), (1,)))
        vg.slen_k = nxt_k - vg.base_k
        vg.n_mb_b = (vg.slen_q + (TILE_M - 1)) // TILE_M
        vg.n_nb_b = (vg.slen_k + (TILE_N - 1)) // TILE_N
        vg.off_b = vg.slen_k - vg.slen_q  # per-seq causal OFF (= Sk-Sq); 0 for self-attn
        return vg

    def _skip(mb, mmin, vg=None, nb=None, cluster=False):
        # guard the per-mb body. Causal.
        ctxs = []
        if causal_skip:
            ctxs.append(_guard(_ge(mb, mmin)))
        if varlen and vg is not None:
            if cluster and use_2cta:
                ctxs.append(k.if_((mb < vg.n_mb_b) & ((nb - cta_in_cluster) < vg.n_nb_b)))
            else:
                ctxs.append(k.if_((mb < vg.n_mb_b) & (nb < vg.n_nb_b)))
        elif dense_pad and nb is not None:
            if cluster:
                ctxs.append(k.if_((nb - cta_in_cluster) < real_n_nb))
            else:
                ctxs.append(k.if_(nb < real_n_nb))
        if not ctxs:
            return nullcontext()
        if len(ctxs) == 1:
            return ctxs[0]

        @contextmanager
        def _both():
            with ctxs[0]:
                with ctxs[1]:
                    yield

        return _both()

    def _eph(mb, mmin):
        # executed-iteration phase e%2 = (mb - mmin)%2. Python int when mmin is a literal.
        return (mb - mmin) % 2

    def _est(mb, mmin):
        # executed-iteration operand stage e%NSTAGE and occupancy e//NSTAGE.
        e = mb - mmin
        return e % NSTAGE, e // NSTAGE

    af = FenceKind.ASYNC_PROXY

    def gemm(
        dst, a, b, m, n, kk, done, *, trans_a=False, trans_b=False, accum=False, a_row0=0, b_row0=0
    ):
        # tcgen05.mma.cta_group::N.kind::f16 — per-instruction MMA-K=16, loop kk//16.
        if use_2cta:
            # PER-CTA operand slices carry m//cg rows of A and n//cg of B.
            a_m = m // cg  # per-CTA A row-extent (128)
            n_b = n // cg  # per-CTA B half-extent (model concatenates)
            # The cta_group::2 MMA is one cluster operation issued by the leader.
            with k.if_(is_leader):
                for g in range(kk // 16):
                    a_op = (
                        _mma_a(k, a, (a_row0 + g * 16, 0), (16, a_m))
                        if trans_a
                        else _mma_a(k, a, (a_row0, g * 16), (a_m, 16))
                    )
                    b_op = (
                        _mma_b(k, b, (b_row0 + g * 16, 0), (16, n_b))
                        if trans_b
                        else _mma_b(k, b, (b_row0, g * 16), (n_b, 16))
                    )
                    k.tcgen05_mma(
                        dst,
                        a_op,
                        b_op,
                        mma_m=m,
                        mma_n=n,
                        format="f16" if iod == DType.F16 else "bf16",
                        block_scale=None,
                        accum=(accum or g != 0),
                        trans_a=trans_a,
                        trans_b=trans_b,
                        ws=False,
                        cta_group=cg,
                    )
                if done is not None:
                    k.tcgen05_commit(bars[done], cta_group=cg, multicast_cta_mask=0b11)
            return
        for g in range(kk // 16):
            a_op = (
                _mma_a(k, a, (a_row0 + g * 16, 0), (16, m))
                if trans_a
                else _mma_a(k, a, (a_row0, g * 16), (m, 16))
            )
            b_op = (
                _mma_b(k, b, (b_row0 + g * 16, 0), (16, n))
                if trans_b
                else _mma_b(k, b, (b_row0, g * 16), (n, 16))
            )
            k.tcgen05_mma(
                dst,
                a_op,
                b_op,
                mma_m=m,
                mma_n=n,
                format="f16" if iod == DType.F16 else "bf16",
                block_scale=None,
                accum=(accum or g != 0),
                trans_a=trans_a,
                trans_b=trans_b,
                ws=False,
                cta_group=cg,
            )
        if done is not None:
            k.tcgen05_commit(bars[done])

    def gemm_acc(mb, mmin, *args, **kwargs):
        # Accumulating GEMM.
        if not causal_skip:
            gemm(*args, accum=(mb > 0), **kwargs)
            return
        # mmin may be a Python int (literal nb) or a ScalarValue (runtime nb).
        with _guard(_eq(mb, mmin)):
            gemm(*args, accum=False, **kwargs)  # first executed block -> zero-init
        with _guard(_gt(mb, mmin)):
            gemm(*args, accum=True, **kwargs)  # subsequent executed blocks -> accumulate

    # ============== load warp (13).
    with k.if_warp(LOAD_WARP), k.if_elected():
        with k.for_each_task(sched) as task:
            batch, head, nb = task_geom(task)
            # GQA: K/V are shared across the G q-heads of a group -> load from the kv-head head_kv.
            head_kv = head // G if gqa else head
            vg = varlen_geom(batch) if varlen else None  # VARLEN: per-seq runtime geometry
            mmin = m_block_min(nb, vg)  # CAUSAL: per-task m-loop start (Python 0 if not)
            ktok = (vg.base_k if varlen else batch * Sk) + nb * TILE_N
            # K, V (the n-tile).
            k.mbarrier_arrive_expect_tx(bars["tk"], bytes=TILE_N * hdim * 2)
            k.tma_load(
                _sl(sK, (0, 0), (TILE_N, hdim)),
                k_g,
                mbar=bars["tk"],
                coords=(ktok, head_kv, 0),
                shape=(TILE_N, hdim),
                gmem_shape=(TILE_N, 1, hdim),
            )
            k.mbarrier_arrive_expect_tx(bars["tv"], bytes=TILE_N * hdimv * 2)
            k.tma_load(
                _sl(sV, (0, 0), (TILE_N, hdimv)),
                v_g,
                mbar=bars["tv"],
                coords=(ktok, head_kv, 0),
                shape=(TILE_N, hdimv),
                gmem_shape=(TILE_N, 1, hdimv),
            )
            if use_2cta:
                # dQ B-operand sKt = K transposed (full cluster kv, this CTA's d-half).
                cl_ktok = (vg.base_k if varlen else batch * Sk) + (nb - cta_in_cluster) * TILE_N
                k.mbarrier_arrive_expect_tx(bars["tkt"], bytes=cg * TILE_N * HD_H * 2)
                k.tma_load(
                    _sl(sKt, (0, 0), (cg * TILE_N, HD_H)),
                    k_g,
                    mbar=bars["tkt"],
                    coords=(cl_ktok, head_kv, cta_in_cluster * HD_H),
                    shape=(cg * TILE_N, HD_H),
                    gmem_shape=(cg * TILE_N, 1, HD_H),
                )
            for mb in range(n_mb):
                with _skip(mb, mmin, vg, nb, cluster=True):  # CAUSAL: skip above-diagonal Q-tiles;
                    # VARLEN: skip mb>=n_mb_b / nb>=n_nb_b (past-sequence tiles).
                    qtok = (vg.base_q if varlen else batch * Sq) + mb * TILE_M
                    # operand stage/occupancy key off the EXECUTED iteration e=mb-mmin.
                    st, occ = _est(mb, mmin)  # operand SMEM/barrier stage, stage occupancy
                    rM = st * TILE_M  # stage row offset (2D operands) / elem offset (1D)
                    # NEW-3/F5: wait the consumer freed this stage before reloading it.
                    k.mbarrier_wait(bars["q_free"], stage=st, phase=(occ + 1) % 2)
                    if use_2cta:
                        # S B = Q, N-axis = q.
                        k.mbarrier_arrive_expect_tx(bars["tq"], bytes=TM_H * hdim * 2, stage=st)
                        k.tma_load(
                            _sl(sQ, (0, 0), (TM_H, hdim)),
                            q_g,
                            mbar=bars["tq"],
                            mbar_stage=st,
                            coords=(qtok + cta_in_cluster * TM_H, head, 0),
                            shape=(TM_H, hdim),
                            gmem_shape=(TM_H, 1, hdim),
                        )
                    else:
                        k.mbarrier_arrive_expect_tx(bars["tq"], bytes=TILE_M * hdim * 2, stage=st)
                        k.tma_load(
                            _sl(sQ, (rM, 0), (TILE_M, hdim)),
                            q_g,
                            mbar=bars["tq"],
                            mbar_stage=st,
                            coords=(qtok, head, 0),
                            shape=(TILE_M, hdim),
                            gmem_shape=(TILE_M, 1, hdim),
                        )
                    k.mbarrier_wait(bars["do_free"], stage=st, phase=(occ + 1) % 2)
                    if use_2cta:
                        # dV B = dO, N-axis = dv (head dim).
                        k.mbarrier_arrive_expect_tx(bars["tdo"], bytes=TILE_M * HDV_H * 2, stage=st)
                        k.tma_load(
                            _sl(sdO, (0, 0), (TILE_M, HDV_H)),
                            do_g,
                            mbar=bars["tdo"],
                            mbar_stage=st,
                            coords=(qtok, head, cta_in_cluster * HDV_H),
                            shape=(TILE_M, HDV_H),
                            gmem_shape=(TILE_M, 1, HDV_H),
                        )
                    else:
                        k.mbarrier_arrive_expect_tx(bars["tdo"], bytes=TILE_M * hdimv * 2, stage=st)
                        k.tma_load(
                            _sl(sdO, (rM, 0), (TILE_M, hdimv)),
                            do_g,
                            mbar=bars["tdo"],
                            mbar_stage=st,
                            coords=(qtok, head, 0),
                            shape=(TILE_M, hdimv),
                            gmem_shape=(TILE_M, 1, hdimv),
                        )
                    # F12: LSE/dPsum are 1D per-row vectors.
                    k.mbarrier_wait(bars["lse_free"], stage=st, phase=(occ + 1) % 2)
                    k.mbarrier_arrive_expect_tx(bars["tlse"], bytes=TILE_M * 4, stage=st)
                    k.tma_load(
                        _sl(sLSE, (rM,), (TILE_M,)),
                        lse_g,
                        mbar=bars["tlse"],
                        mbar_stage=st,
                        coords=(qtok, head),
                        shape=(TILE_M,),
                        gmem_shape=(TILE_M, 1),
                    )
                    k.mbarrier_wait(bars["dps_free"], stage=st, phase=(occ + 1) % 2)
                    k.mbarrier_arrive_expect_tx(bars["tdps"], bytes=TILE_M * 4, stage=st)
                    k.tma_load(
                        _sl(sdPsum, (rM,), (TILE_M,)),
                        dpsum_g,
                        mbar=bars["tdps"],
                        mbar_stage=st,
                        coords=(qtok, head),
                        shape=(TILE_M,),
                        gmem_shape=(TILE_M, 1),
                    )
                    if use_2cta:
                        # Transposed B operands sdOt (dP) / sQt (dK).
                        if is_hd192:
                            # hd192: sdOt aliases the sQ buffer (Q->dOt time-mux).
                            k.mbarrier_wait(bars["s_ready"], phase=_eph(mb, mmin))
                        else:
                            # NEW (multi-Q-tile): sdOt is single-stage.
                            k.mbarrier_wait(bars["dot_free"], stage=st, phase=(occ + 1) % 2)
                        k.mbarrier_arrive_expect_tx(bars["tdot"], bytes=TM_H * hdimv * 2, stage=st)
                        k.tma_load(
                            _sl(sdOt, (0, 0), (TM_H, hdimv)),
                            do_g,
                            mbar=bars["tdot"],
                            mbar_stage=st,
                            coords=(qtok + cta_in_cluster * TM_H, head, 0),
                            shape=(TM_H, hdimv),
                            gmem_shape=(TM_H, 1, hdimv),
                        )
                        # dK B = Q, N-axis = d (contraction = q).
                        if is_hd192:
                            # hd192: sQt aliases the same sQ buffer (dOt->Qt time-mux).
                            k.mbarrier_wait(bars["dp_ready"], phase=_eph(mb, mmin))
                        else:
                            # NEW (multi-Q-tile): sQt is single-stage.
                            k.mbarrier_wait(bars["qt_free"], stage=st, phase=(occ + 1) % 2)
                        k.mbarrier_arrive_expect_tx(bars["tqt"], bytes=TILE_M * HD_H * 2, stage=st)
                        k.tma_load(
                            _sl(sQt, (0, 0), (TILE_M, HD_H)),
                            q_g,
                            mbar=bars["tqt"],
                            mbar_stage=st,
                            coords=(qtok, head, cta_in_cluster * HD_H),
                            shape=(TILE_M, HD_H),
                            gmem_shape=(TILE_M, 1, HD_H),
                        )

    # ============== mma warp (12): the 5 tcgen05 GEMMs ============== F9.
    mma_m = cg * TILE_N

    # Single-thread MMA issuer stream.
    with k.if_warp(MMA_WARP), k.if_elected():
        with k.for_each_task(sched) as task:
            batch, head, nb = task_geom(task)
            vg = varlen_geom(batch) if varlen else None  # VARLEN: per-seq runtime geometry
            mmin = m_block_min(nb, vg)  # CAUSAL: per-task m-loop start (Python 0 if not)
            k.mbarrier_wait(bars["tk"], phase=0)
            k.mbarrier_wait(bars["tv"], phase=0)
            if use_2cta:
                # Leader reads BOTH CTAs' K/V (A-operands) — observe the peer's K/V loads.
                with k.if_(is_leader):
                    k.mbarrier_wait(peer_bars["tk"], phase=0)
                    k.mbarrier_wait(peer_bars["tv"], phase=0)
                # sKt (dQ B-operand = K transposed) is loaded ONCE per task (hoisted next to sK/sV).
                k.mbarrier_wait(bars["tkt"], phase=0)
                with k.if_(is_leader):
                    k.mbarrier_wait(peer_bars["tkt"], phase=0)

            # CAUSAL: every per-mb body is wrapped in `with _skip(mb, mmin)` so a skipped.
            def gS(mb):  # S[kv,q] = Σ_d K·Qᵀ  -> tS  (waits tq[mb], commits s_ready[mb])
                with _skip(
                    mb, mmin, vg, nb, cluster=True
                ):  # 2-CTA: cluster MMA -> cluster predicate
                    st, occ = _est(mb, mmin)
                    rM = st * TILE_M
                    if is_hd192:
                        # hd192: dQ overlaps S at TMEM cols.
                        if causal_skip:
                            with _guard(_gt(mb, mmin)):
                                k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 1, mmin))
                        elif mb > 0:
                            k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 1, mmin))
                            if n_mb > 1:
                                with k.if_(is_leader):
                                    k.mbarrier_wait(peer_bars["dq_free"], phase=_eph(mb - 1, mmin))
                    elif use_2cta and mb >= 2:
                        # General 2-CTA: dQ overlaps S.
                        k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 2, mmin))
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["dq_free"], phase=_eph(mb - 2, mmin))
                    k.mbarrier_wait(bars["tq"], stage=st, phase=occ % 2)
                    if use_2cta:
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["tq"], phase=occ % 2)
                    gemm(tS.at(0, 0), sK, sQ, mma_m, TILE_M, hdim, "s_ready", b_row0=rM)
                    if use_2cta and not is_hd192:
                        # general 2-CTA: sQ (S B-operand) is read ONLY by gS -> gS frees its stage.
                        with k.if_(is_leader):
                            k.mbarrier_arrive(bars["q_free"], stage=st)
                            k.mbarrier_arrive(peer_free["q_free"], stage=st)

            def gdP(mb):  # dP[kv,q] = Σ_d V·dOᵀ -> tdP. 2-CTA B = sdOt (transposed dO).
                with _skip(
                    mb, mmin, vg, nb, cluster=True
                ):  # 2-CTA: cluster MMA -> cluster predicate
                    st, occ = _est(mb, mmin)
                    rM = st * TILE_M
                    if is_hd192:
                        # hd192: dP overwrites S's overlapping cols.
                        k.mbarrier_wait(bars["s_free"], phase=_eph(mb, mmin))
                        # the leader's cta_group::2 dP MMA writes the PEER CTA's tmem_dP too.
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["s_free"], phase=_eph(mb, mmin))
                    elif not use_2cta:
                        # 1-CTA only: dP and dQ OVERLAP TMEM.
                        if causal_skip:
                            with _guard(_gt(mb, mmin)):
                                k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 1, mmin))
                        elif mb > 0:
                            k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 1, mmin))
                    k.mbarrier_wait(bars["tdo"], stage=st, phase=occ % 2)
                    if use_2cta:
                        k.mbarrier_wait(bars["tdot"], phase=occ % 2)
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["tdo"], phase=occ % 2)
                            k.mbarrier_wait(peer_bars["tdot"], phase=occ % 2)
                        # dP B-operand from the dedicated transposed buffer sdOt (rM=0, single).
                        gemm(tdP.at(0, 0), sV, sdOt, mma_m, TILE_M, hdimv, "dp_ready")
                        # 2-CTA: sdOt (dP B-operand) is read ONLY by gdP -> the leader frees BOTH.
                        with k.if_(is_leader):
                            k.mbarrier_arrive(bars["dot_free"], stage=st)
                            k.mbarrier_arrive(peer_free["dot_free"], stage=st)
                    else:
                        gemm(tdP.at(0, 0), sV, sdO, TILE_N, TILE_M, hdimv, "dp_ready", b_row0=rM)

            def gdV(mb):  # dV[kv,dv] = Σ_q Pᵀ·dO -> tdV (accum after 1st executed; waits p_ready)
                with _skip(
                    mb, mmin, vg, nb, cluster=True
                ):  # 2-CTA: cluster MMA -> cluster predicate
                    st, occ = _est(mb, mmin)
                    rM = st * TILE_M
                    k.mbarrier_wait(bars["p_ready"], phase=_eph(mb, mmin))
                    if use_2cta:
                        # leader reads the PEER's P (TMEM-A from both CTAs) -> wait peer p_ready.
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["p_ready"], phase=_eph(mb, mmin))
                    gemm_acc(
                        mb,
                        mmin,
                        tdV.at(0, 0),
                        tP,
                        sdO,
                        mma_m,
                        hdimv,
                        TILE_M,
                        "dv_done",
                        trans_b=True,
                        b_row0=rM,
                    )
                    # NEW-3/F5: gdV is the LAST reader of sdO[st].
                    if use_2cta:
                        # leader read BOTH CTAs' sdO -> leader frees both (local + peer multicast).
                        with k.if_(is_leader):
                            k.mbarrier_arrive(bars["do_free"], stage=st)
                            k.mbarrier_arrive(peer_free["do_free"], stage=st)
                    else:
                        k.mbarrier_arrive(bars["do_free"], stage=st)

            def gdK(mb):  # dK[kv,d] = Σ_q dSᵀ·Q -> tdK (accum after 1st executed; waits ds_ready)
                # dK uses THIS CTA's dS rows (no cross-CTA dep) — buildable in stage 1.
                with _skip(
                    mb, mmin, vg, nb, cluster=True
                ):  # 2-CTA: cluster MMA -> cluster predicate
                    st, occ = _est(mb, mmin)
                    rM = st * TILE_M
                    k.mbarrier_wait(bars["ds_ready"], phase=_eph(mb, mmin))
                    if use_2cta:
                        k.mbarrier_wait(bars["tqt"], phase=occ % 2)
                        with k.if_(is_leader):
                            k.mbarrier_wait(peer_bars["tqt"], phase=occ % 2)
                            # leader reads the PEER's dS.
                            k.mbarrier_wait(peer_bars["ds_ready"], phase=_eph(mb, mmin))
                        gemm_acc(
                            mb,
                            mmin,
                            tdK.at(0, 0),
                            tdS,
                            sQt,
                            mma_m,
                            hdim,
                            TILE_M,
                            "dk_done",
                            trans_b=True,
                        )
                        if is_hd192:
                            # hd192: sQt ALIASES sQ (time-mux Q->dOt->Qt).
                            with k.if_(is_leader):
                                k.mbarrier_arrive(bars["q_free"], stage=st)
                                k.mbarrier_arrive(peer_free["q_free"], stage=st)
                        else:
                            # general 2-CTA: sQt.
                            with k.if_(is_leader):
                                k.mbarrier_arrive(bars["qt_free"], stage=st)
                                k.mbarrier_arrive(peer_free["qt_free"], stage=st)
                    else:
                        gemm_acc(
                            mb,
                            mmin,
                            tdK.at(0, 0),
                            tdS,
                            sQ,
                            TILE_N,
                            hdim,
                            TILE_M,
                            "dk_done",
                            trans_b=True,
                            b_row0=rM,
                        )
                        # NEW-3/F5 (1-CTA).
                        k.mbarrier_arrive(bars["q_free"], stage=st)

            def gdQ(
                mb,
            ):  # dQ[q,d] = Σ_kv dS·K -> tdQ (consumes ds_ready[mb] via gdK; commits dq_done[mb])
                with _skip(
                    mb, mmin, vg, nb, cluster=True
                ):  # 2-CTA: cluster MMA -> cluster predicate
                    if use_2cta:
                        # 2-CTA dQ (map §3).
                        if not is_hd192 and n_mb > 1 and mb > 0:
                            # tdQ accumulator WAR.
                            k.mbarrier_wait(bars["dq_free"], phase=_eph(mb - 1, mmin))
                            with k.if_(is_leader):
                                k.mbarrier_wait(peer_bars["dq_free"], phase=_eph(mb - 1, mmin))
                        if not is_hd192 and n_mb > 1 and mb + 1 < n_mb:
                            # dQ overwrites S TMEM, so wait until the next S tile is consumed.
                            with k.if_((mb + 1) < vg.n_mb_b) if varlen else nullcontext():
                                k.mbarrier_wait(bars["s_cons"], phase=_eph(mb + 1, mmin))
                                with k.if_(is_leader):
                                    k.mbarrier_wait(peer_bars["s_cons"], phase=_eph(mb + 1, mmin))
                        with k.if_(is_leader):
                            # gate on the exchanged dS.
                            k.mbarrier_wait(bars["dS_cluster_leader"], phase=mb % 2)
                        gemm(
                            tdQ.at(0, 0),
                            sdS_full,
                            sKt,
                            TILE_M,
                            hdim,
                            cg * TILE_N,
                            "dq_done",
                            trans_a=True,
                            trans_b=True,
                        )
                        if n_mb > 1:
                            # sdS_full is single-buffer.
                            with k.if_(is_leader):
                                k.mbarrier_arrive(bars["dS_free"])
                                k.mbarrier_arrive(peer_free["dS_free"])
                        return
                    gemm(
                        tdQ.at(0, 0),
                        sdS,
                        sK,
                        TILE_M,
                        hdim,
                        TILE_N,
                        "dq_done",
                        trans_a=True,
                        trans_b=True,
                    )

            if is_hd192:
                # hd192 DeepSeek: the dedicated TMEM layout overlaps S/dP/dS/dQ.
                for mb in range(n_mb):
                    gS(mb)
                    gdP(mb)
                    gdK(mb)
                    gdV(mb)
                    gdQ(mb)
            elif use_2cta:
                # The cluster-wide GEMMs.
                gS(0)
                gdP(0)
                gdV(0)
                for mb in range(1, n_mb):
                    gS(mb)
                    gdK(mb - 1)
                    gdP(mb)
                    gdQ(mb - 1)
                    gdV(mb)
                gdK(n_mb - 1)
                gdQ(n_mb - 1)
            else:
                # PROLOGUE (block 0): S, dP, dV.
                gS(0)
                gdP(0)
                gdV(0)
                # MAIN LOOP (block mb = 1..n_mb-1): S[mb], dK[mb-1], dQ[mb-1], dP[mb], dV[mb].
                for mb in range(1, n_mb):
                    gS(mb)
                    gdK(mb - 1)
                    gdQ(mb - 1)
                    gdP(mb)
                    gdV(mb)
                # TAIL (last block): dK[n_mb-1], dQ[n_mb-1].
                gdK(n_mb - 1)
                gdQ(n_mb - 1)
            # mma warp is pure GEMM-issue (F4).

    # ============== compute warps (4-11).
    NCOL = 64

    def compute_softmax_ds(col_base, tid, task, per_mb_tail=None):
        # NCOL-wide P=exp2(S·scale−LSE) then dS=P∘(dP−dPsum) for q-cols.
        fragS = reg(DType.F32, (NCOL,))
        fragP = reg(iod, (NCOL,))
        fragdP = reg(DType.F32, (NCOL,))
        fragdS = reg(iod, (NCOL,))
        rlse = reg(DType.F32, (1,))
        rnlse = reg(DType.F32, (1,))
        rdps = reg(DType.F32, (1,))
        rt = reg(DType.F32, (1,))
        rP = reg(DType.F32, (NCOL,))
        cscale = reg(DType.F32, (1,))
        # F7: packed-pair (f32x2) temps for the dS residual (sub_packed_f32x2 + mul_packed_f32x2).
        rdps2 = reg(DType.F32, (2,))
        rt2 = reg(DType.F32, (2,))
        cbc = col_base // 2
        rninf = reg(DType.F32, (1,))  # VARLEN: -inf fill for OOB-row/col masking (→ P=0)
        # CAUSAL: this CTA's kv-tile index (for the per-tile key_start base).
        nb = None
        if config.is_causal or varlen or dense_pad:
            _batch, _, nb = task_geom(task)
        vg = varlen_geom(_batch) if varlen else None  # VARLEN: per-seq runtime geometry
        mmin = m_block_min(nb, vg) if causal_skip else 0  # CAUSAL skip: per-task m-loop start
        for mb in range(n_mb):
            with _skip(mb, mmin, vg, nb, cluster=True):  # CAUSAL: skip above-diagonal Q-tiles;
                # VARLEN: skip mb>=n_mb_b / nb>=n_nb_b.
                st, occ = _est(mb, mmin)
                ph = _eph(mb, mmin)
                rM = st * TILE_M
                k.reg_fill(cscale, scale_log2)
                # P[kv,q] = exp2(S[kv,q]·scale_log2 − LSE[q]) (LSE per q-col).
                k.mbarrier_wait(bars["s_ready"], phase=ph)
                k.tcgen05_ld(fragS, tS.at(0, col_base), shape="32x32b", num=NCOL)
                k.tcgen05_wait_ld()
                if is_hd192:
                    # hd192: S fully read into rmem (wait_ld drained the t2r).
                    k.fence(kind=af, scope=FenceScope.CTA)
                    k.mbarrier_arrive(bars["s_free"])
                elif use_2cta and n_mb > 1:
                    # general 2-CTA multi-Q-tile: S fully read into rmem (wait_ld drained the t2r).
                    k.fence(kind=af, scope=FenceScope.CTA)
                    k.mbarrier_arrive(bars["s_cons"])
                if dense_pad:
                    # DENSE 2-CTA PADDING-TILE mask (THE critical correctness item for batch>1).
                    k.reg_fill(rninf, float("-inf"))
                    with k.if_(nb >= real_n_nb):
                        for c in range(NCOL):
                            k.reg_fill(_sl(fragS, (c,), (1,)), rninf)
                if varlen:
                    # VARLEN OOB-tail validity mask.
                    k.reg_fill(rninf, float("-inf"))
                    with k.if_((nb * TILE_N + tid) >= vg.slen_k):
                        for c in range(NCOL):
                            k.reg_fill(_sl(fragS, (c,), (1,)), rninf)
                    for c in range(NCOL):
                        with k.if_((mb * TILE_M + col_base + c) >= vg.slen_q):
                            k.reg_fill(_sl(fragS, (c,), (1,)), rninf)
                if config.is_causal:
                    # CAUSAL mask on the transposed bwd fragment fragS = [kv-row=tid, q-col=j], j.
                    off = vg.off_b if varlen else (Sk - Sq)
                    k.reg_causal_mask(
                        fragS,
                        fragS,
                        query_start=mb * TILE_M + col_base,
                        key_start=nb * TILE_N - off,
                        group_size=1,
                        swap_qk=True,
                    )
                k.mbarrier_wait(bars["tlse"], stage=st, phase=occ % 2)
                for c in range(NCOL):
                    # P = exp2(S·scale − LSE).
                    k.reg_load(rlse, _sl(sLSE, (rM + col_base + c,), (1,)))
                    k.reg_unary(rnlse, rlse, op="neg")
                    k.reg_fma(rt, _sl(fragS, (c,), (1,)), cscale, rnlse)
                    k.reg_unary(_sl(rP, (c,), (1,)), rt, op="exp2")
                    k.reg_cvt(_sl(fragP, (c,), (1,)), _sl(rP, (c,), (1,)))
                # NEW-3/F5: per-thread arrive after THIS thread's sLSE[st] reads.
                k.mbarrier_arrive(bars["lse_free"], stage=st)
                # (a0) S is fully read into registers (fence drains the t2r).
                k.fence(kind=af, scope=FenceScope.CTA)
                k.named_barrier(barrier_id=1, num_warps=8)
                k.tcgen05_st(tP.at(0, cbc), _sl(fragP, (0,), (NCOL // 2,)), num=NCOL // 2)
                k.tcgen05_wait_st()
                k.fence(kind=af, scope=FenceScope.CTA)
                # (a) P r2t-write into tmem fenced+visible.
                k.named_barrier(barrier_id=1, num_warps=8)
                k.mbarrier_arrive(bars["p_ready"])
                # dS[kv,q] = P[kv,q]·(dP[kv,q] − dPsum[q]) (dPsum per q-col).
                k.mbarrier_wait(bars["dp_ready"], phase=ph)
                k.tcgen05_ld(fragdP, tdP.at(0, col_base), shape="32x32b", num=NCOL)
                k.tcgen05_wait_ld()
                k.fence(kind=af, scope=FenceScope.CTA)
                # (b) dP t2r-read fenced; dS overwrites dP's tmem cols (dS@dP overlap).
                k.named_barrier(barrier_id=1, num_warps=8)
                k.mbarrier_wait(bars["tdps"], stage=st, phase=occ % 2)
                # F7: dS = P·(dP − dPsum) in PACKED f32x2 pairs.
                for c in range(0, NCOL, 2):
                    k.reg_load(rdps2, _sl(sdPsum, (rM + col_base + c,), (2,)))
                    k.reg_sub(rt2, _sl(fragdP, (c,), (2,)), rdps2)  # (dP − dPsum) packed
                    k.reg_mul(rt2, _sl(rP, (c,), (2,)), rt2)  # P·(…) packed
                    k.reg_cvt(_sl(fragdS, (c,), (2,)), rt2)
                # NEW-3/F5: per-thread arrive after THIS thread's sdPsum[st] reads (count=2*WG_T).
                k.mbarrier_arrive(bars["dps_free"], stage=st)
                k.tcgen05_st(tdS.at(0, cbc), _sl(fragdS, (0,), (NCOL // 2,)), num=NCOL // 2)
                k.tcgen05_wait_st()
                # stage dS to SMEM sdS (dQ B-operand path: A=sdS trans_a) — each wg writes its cols.
                if use_2cta:
                    if is_hd192:
                        # hd192 has TWO single-buffer dS-export WARs against the PREVIOUS executed.
                        if n_mb > 1:
                            if causal_skip:
                                with _guard(_gt(mb, mmin)):
                                    k.mbarrier_wait(bars["dS_free"], phase=_eph(mb - 1, mmin))
                            elif mb > 0:
                                k.mbarrier_wait(bars["dS_free"], phase=_eph(mb - 1, mmin))
                        if causal_skip:
                            with _guard(_gt(mb, mmin)):
                                k.mbarrier_wait(bars["dQaccum_empty"], phase=_eph(mb - 1, mmin))
                        elif mb > 0:
                            k.mbarrier_wait(bars["dQaccum_empty"], phase=_eph(mb - 1, mmin))
                    elif n_mb > 1 and mb > 0:
                        # General 2-CTA multi-Q-tile.
                        k.mbarrier_wait(bars["dS_free"], phase=_eph(mb - 1, mmin))
                    # dS cross-CTA exchange producer (map §3).
                    kept = is_leader if col_base == 0 else (~is_leader)
                    full_row0 = (col_base // NCOL) * TILE_N  # this kv-tile slot in sdS_full
                    with k.if_(kept & (tid < TILE_N)):
                        k.reg_store(
                            _sl(sdS_full, (full_row0 + tid, 0), (1, NCOL)),
                            _sl(fragdS, (0,), (NCOL,)),
                        )
                    with k.if_((~kept) & (tid < TILE_N)):
                        k.reg_store(_sl(sdS_xchg, (tid, 0), (1, NCOL)), _sl(fragdS, (0,), (NCOL,)))
                    # The exporting wg.
                    k.wg_sync(barrier_id=(2 if col_base == 0 else 3))
                    k.fence(kind=af, scope=FenceScope.CTA)  # generic SMEM write -> async bulk copy
                    peer = 1 - cta_in_cluster
                    xchg_bytes = TILE_N * NCOL * 2
                    # peer's sdS_full slot for THIS CTA's kv-tile = rows.
                    dst_row0 = cta_in_cluster * TILE_N
                    with k.if_((~kept) & tid.eq(0)):
                        k.mbarrier_arrive_expect_tx(
                            k.mbar_ref(bars["dS_cluster_full"], remote_coord=peer), bytes=xchg_bytes
                        )
                        k.cp_async_bulk_s2cluster(
                            _sl(sdS_full, (dst_row0, 0), (TILE_N, NCOL)),
                            _sl(sdS_xchg, (0, 0), (TILE_N, NCOL)),
                            mbar=k.mbar_ref(bars["dS_cluster_full"], remote_coord=peer),
                            bytes=xchg_bytes,
                        )
                        if is_hd192 or n_mb > 1:
                            # Drain the s2cluster's SHARED-source.
                            k.cp_async_bulk_commit_group()
                            k.cp_async_bulk_wait_group_read(0)
                else:
                    with k.if_(tid < TILE_N):
                        k.reg_store(
                            _sl(sdS, (tid, col_base), (1, NCOL)), _sl(fragdS, (0,), (NCOL,))
                        )
                    k.wg_sync(barrier_id=(2 if col_base == 0 else 3))  # per-wg distinct barrier_id
                    k.fence(kind=af, scope=FenceScope.CTA)
                # (c) dS r2s-store to smem fenced.
                k.named_barrier(barrier_id=1, num_warps=8)
                k.mbarrier_arrive(bars["ds_ready"])
                if per_mb_tail is not None:
                    per_mb_tail(mb)

    # ---- dK/dV epilogue.
    hdim_half = hdim // 2
    hdimv_half = hdimv // 2
    # CHUNKED epilogue (flashattn epilogue_dK_or_dV_tma @3853).
    NEPI_K = max(1, hdim_half // DK_RNCOL)
    NEPI_V = max(1, hdimv_half // DV_RNCOL)

    def make_dk_dv_tail(
        col_base, tid, task, rt, fragK, fragV, rkb, cscale_s, mmin, vg=None, nb=None
    ):
        half_base = (col_base // NCOL) * hdim_half  # q-col half 0/64 -> hdim half 0/hdim_half
        half_base_v = (col_base // NCOL) * hdimv_half
        wg_idx = col_base // NCOL  # compute-wg index (0/1)
        wg_bar = 2 if col_base == 0 else 3  # per-wg distinct wg_sync barrier_id

        def _stage_chunk(sbuf, frag, j, rncol, scale_it):
            # r2s ONE RNCOL-wide chunk.
            slot = wg_idx * rncol
            with k.if_(tid < TILE_N):
                if gqa:
                    for c in range(rncol):
                        k.reg_store(
                            _sl(sbuf, (tid, slot + c), (1, 1)), _sl(frag, (j * rncol + c,), (1,))
                        )
                elif scale_it:
                    for c in range(rncol):
                        k.reg_mul(rt, _sl(frag, (j * rncol + c,), (1,)), cscale_s)
                        k.reg_cvt(rkb, rt)
                        k.reg_store(_sl(sbuf, (tid, slot + c), (1, 1)), rkb)
                else:
                    for c in range(rncol):
                        k.reg_cvt(rkb, _sl(frag, (j * rncol + c,), (1,)))
                        k.reg_store(_sl(sbuf, (tid, slot + c), (1, 1)), rkb)
            k.wg_sync(barrier_id=wg_bar)
            k.fence(kind=af, scope=FenceScope.CTA)

        def _chunk_drain(j, nchunks):
            # Drain each async store before reusing this warpgroup's staging slice.
            if j < nchunks - 1:
                with k.if_(tid.eq(0)):
                    k.cp_async_bulk_commit_group()
                    k.cp_async_bulk_wait_group_read(0)
                k.wg_sync(barrier_id=wg_bar)

        def _store_epilogue():
            # ---- this wg's hdim-column half.
            k.reg_fill(cscale_s, config.softmax_scale)
            _tc_ld(k, fragV, tdV, hdimv_half, col=half_base_v)
            k.tcgen05_wait_ld()
            _tc_ld(k, fragK, tdK, hdim_half, col=half_base)
            k.tcgen05_wait_ld()
            batch, head, nb_ = task_geom(task)
            head_kv = head // G if gqa else head  # GQA: dK/dV land in the shared kv-head rows
            if varlen:
                # VARLEN dK/dV store (C12).
                ktok = vg.base_k + nb * TILE_N
                if gqa:
                    # GQA varlen: dK/dV are f32 reduce-add accumulators in the SHARED kv-head.
                    with k.if_(nb < vg.n_nb_b):
                        for j in range(NEPI_V):
                            _stage_chunk(sdV, fragV, j, DV_RNCOL, scale_it=False)
                            with k.if_(tid.eq(0)):
                                k.tma_reduce_add(
                                    dv_g,
                                    _sl(sdV, (0, wg_idx * DV_RNCOL), (TILE_N, DV_RNCOL)),
                                    coords=(ktok, head_kv, half_base_v + j * DV_RNCOL),
                                    shape=(TILE_N, DV_RNCOL),
                                    gmem_shape=(TILE_N, 1, DV_RNCOL),
                                    allow_nondet_reduce=True,
                                )
                            _chunk_drain(j, NEPI_V)
                        for j in range(NEPI_K):
                            _stage_chunk(sdK, fragK, j, DK_RNCOL, scale_it=False)
                            with k.if_(tid.eq(0)):
                                k.tma_reduce_add(
                                    dk_g,
                                    _sl(sdK, (0, wg_idx * DK_RNCOL), (TILE_N, DK_RNCOL)),
                                    coords=(ktok, head_kv, half_base + j * DK_RNCOL),
                                    shape=(TILE_N, DK_RNCOL),
                                    gmem_shape=(TILE_N, 1, DK_RNCOL),
                                    allow_nondet_reduce=True,
                                )
                            _chunk_drain(j, NEPI_K)
                        with k.if_(tid.eq(0)):
                            k.cp_async_bulk_commit_group()
                            k.cp_async_bulk_wait_group_read(0)
                    return
                # MHA varlen: predicated SCALAR store, one chunk at a time.
                with k.if_(nb < vg.n_nb_b):  # per-CTA tile_valid: past-seq CTA writes nothing
                    for j in range(NEPI_V):
                        _stage_chunk(sdV, fragV, j, DV_RNCOL, scale_it=False)
                        slot = wg_idx * DV_RNCOL
                        with k.if_(tid < TILE_N):
                            with k.if_((nb * TILE_N + tid) < vg.slen_k):
                                for c in range(DV_RNCOL):
                                    k.reg_load(rkb, _sl(sdV, (tid, slot + c), (1, 1)))
                                    k.reg_store(
                                        _sl(
                                            dv_g,
                                            (ktok + tid, head, half_base_v + j * DV_RNCOL + c),
                                            (1, 1, 1),
                                        ),
                                        rkb,
                                    )
                    for j in range(NEPI_K):
                        _stage_chunk(sdK, fragK, j, DK_RNCOL, scale_it=True)
                        slot = wg_idx * DK_RNCOL
                        with k.if_(tid < TILE_N):
                            with k.if_((nb * TILE_N + tid) < vg.slen_k):
                                for c in range(DK_RNCOL):
                                    k.reg_load(rkb, _sl(sdK, (tid, slot + c), (1, 1)))
                                    k.reg_store(
                                        _sl(
                                            dk_g,
                                            (ktok + tid, head, half_base + j * DK_RNCOL + c),
                                            (1, 1, 1),
                                        ),
                                        rkb,
                                    )
                return
            # DENSE 2-CTA PADDING TILE.
            _pad_gate = k.if_(nb_ < real_n_nb) if dense_pad else nullcontext()
            with _pad_gate:
                # dK/dV epilogue on the compute role.
                ktok = batch * Sk + nb_ * TILE_N
                wg_slot = 0 if col_base == 0 else 1  # the trailing-2 sem slot = per-wg column half
                # DETERMINISTIC GQA.
                if deterministic and gqa:
                    with k.if_(tid.eq(0)):
                        k.gmem_wait_eq(
                            dk_sem_g, coords=(batch, head_kv, nb_, wg_slot), value=head % G
                        )
                        k.gmem_wait_eq(
                            dv_sem_g, coords=(batch, head_kv, nb_, wg_slot), value=head % G
                        )
                    k.wg_sync(barrier_id=wg_bar)
                # GQA: per-chunk ATOMIC reduce-add into the shared kv-head's f32 rows.
                for j in range(NEPI_V):
                    _stage_chunk(sdV, fragV, j, DV_RNCOL, scale_it=False)
                    with k.if_(tid.eq(0)):
                        cv = wg_idx * DV_RNCOL
                        if gqa:
                            k.tma_reduce_add(
                                dv_g,
                                _sl(sdV, (0, cv), (TILE_N, DV_RNCOL)),
                                coords=(ktok, head_kv, half_base_v + j * DV_RNCOL),
                                shape=(TILE_N, DV_RNCOL),
                                gmem_shape=(TILE_N, 1, DV_RNCOL),
                                allow_nondet_reduce=True,
                            )
                        else:
                            k.tma_store(
                                dv_g,
                                _sl(sdV, (0, cv), (TILE_N, DV_RNCOL)),
                                coords=(ktok, head, half_base_v + j * DV_RNCOL),
                                shape=(TILE_N, DV_RNCOL),
                                gmem_shape=(TILE_N, 1, DV_RNCOL),
                            )
                    _chunk_drain(j, NEPI_V)
                for j in range(NEPI_K):
                    _stage_chunk(sdK, fragK, j, DK_RNCOL, scale_it=True)
                    with k.if_(tid.eq(0)):
                        ck = wg_idx * DK_RNCOL
                        if gqa:
                            k.tma_reduce_add(
                                dk_g,
                                _sl(sdK, (0, ck), (TILE_N, DK_RNCOL)),
                                coords=(ktok, head_kv, half_base + j * DK_RNCOL),
                                shape=(TILE_N, DK_RNCOL),
                                gmem_shape=(TILE_N, 1, DK_RNCOL),
                                allow_nondet_reduce=True,
                            )
                        else:
                            k.tma_store(
                                dk_g,
                                _sl(sdK, (0, ck), (TILE_N, DK_RNCOL)),
                                coords=(ktok, head, half_base + j * DK_RNCOL),
                                shape=(TILE_N, DK_RNCOL),
                                gmem_shape=(TILE_N, 1, DK_RNCOL),
                            )
                    _chunk_drain(j, NEPI_K)
                with k.if_(tid.eq(0)):
                    k.cp_async_bulk_commit_group()
                    k.cp_async_bulk_wait_group_read(0)
                # DETERMINISTIC GQA.
                if deterministic and gqa:
                    k.wg_sync(barrier_id=wg_bar)
                    with k.if_(tid.eq(0)):
                        k.fence(kind=FenceKind.MEMORY, scope=FenceScope.GPU)
                        k.gmem_atomic_add(
                            dk_sem_g,
                            coords=(batch, head_kv, nb_, wg_slot),
                            value=1,
                            order="release",
                        )
                        k.gmem_atomic_add(
                            dv_sem_g,
                            coords=(batch, head_kv, nb_, wg_slot),
                            value=1,
                            order="release",
                        )

        def dk_dv_tail(mb):
            # dv_done/dk_done commit once per EXECUTED block.
            ph = _eph(mb, mmin)
            # dV/dK accumulate in TMEM across m-blocks.
            k.mbarrier_wait(bars["dv_done"], phase=ph)
            k.mbarrier_wait(bars["dk_done"], phase=ph)
            if varlen:
                # VARLEN: the store fires on the last EXECUTED block mb == n_mb_b-1 (runtime).
                with k.if_(vg.n_mb_b.eq(mb + 1)):
                    _store_epilogue()
            elif mb == n_mb - 1:
                _store_epilogue()

        return dk_dv_tail

    with k.if_warpgroup(1):  # compute wg1: warps 4-7, q-cols 0..63 / dK/dV hdim-cols [0, hdim/2)
        tid = k.tid_in_wg()  # 0..127 = kv-row of S/P/dP/dS [TILE_N, TILE_M]
        rt = reg(DType.F32, (1,))
        fragK = reg(DType.F32, (hdim_half,))
        fragV = reg(DType.F32, (hdimv_half,))
        rkb = reg(iod, (1,))
        cscale_s = reg(DType.F32, (1,))
        with k.for_each_task(sched) as task:
            b_, _, nb_ = task_geom(task)
            vg_ = varlen_geom(b_) if varlen else None
            mmin_ = m_block_min(nb_, vg_) if causal_skip else 0
            tail = make_dk_dv_tail(0, tid, task, rt, fragK, fragV, rkb, cscale_s, mmin_, vg_, nb_)
            compute_softmax_ds(0, tid, task, per_mb_tail=tail)

    with k.if_warpgroup(
        2
    ):  # compute wg2: warps 8-11, q-cols 64..127 / dK/dV hdim-cols [hdim/2, hdim)
        tid = k.tid_in_wg()
        rt = reg(DType.F32, (1,))
        fragK = reg(DType.F32, (hdim_half,))
        fragV = reg(DType.F32, (hdimv_half,))
        rkb = reg(iod, (1,))
        cscale_s = reg(DType.F32, (1,))
        with k.for_each_task(sched) as task:
            b_, _, nb_ = task_geom(task)
            vg_ = varlen_geom(b_) if varlen else None
            mmin_ = m_block_min(nb_, vg_) if causal_skip else 0
            tail = make_dk_dv_tail(64, tid, task, rt, fragK, fragV, rkb, cscale_s, mmin_, vg_, nb_)
            compute_softmax_ds(64, tid, task, per_mb_tail=tail)

    # ============== reduce warps (0-3).
    RDQ_STAGES = hdim // RDQ_NCOL  # = 2 for hd64 (RDQ_NCOL is module-level)
    assert hdim % RDQ_NCOL == 0
    with k.if_warpgroup(0):
        tid = k.tid_in_wg()
        fragdQ = reg(DType.F32, (hdim,))
        rzero = reg(DType.F32, (1,))
        with k.for_each_task(sched) as task:
            batch, head, nb = task_geom(task)
            vg = varlen_geom(batch) if varlen else None  # VARLEN: per-seq runtime geometry
            mmin = m_block_min(nb, vg) if causal_skip else 0  # CAUSAL skip: per-task m-loop start
            for mb in range(n_mb):
                with _skip(mb, mmin, vg, nb, cluster=True):  # CAUSAL: skip above-diagonal; VARLEN:
                    # skip past-seq. cluster=True (2-CTA).
                    ph = _eph(mb, mmin)
                    qtok = (vg.base_q if varlen else batch * Sq) + mb * TILE_M
                    k.mbarrier_wait(bars["dq_done"], phase=ph)
                    if use_2cta:
                        # 2-CTA dQ accumulator is Layout B (cta_group::2, m=128).
                        DQ_COL = hdim // 2
                        _tc_ld(k, fragdQ, tdQ, DQ_COL, col=0)
                        k.tcgen05_wait_ld()
                        k.mbarrier_arrive(bars["dq_free"])
                        if deterministic:
                            # 2-CTA cross-CLUSTER serialization.
                            c_turn = task.field("work") % n_cluster
                            with k.if_(tid.eq(0)):
                                k.gmem_wait_eq(
                                    dq_sem_g, coords=(batch, head, mb, cta_in_cluster), value=c_turn
                                )
                            k.wg_sync(barrier_id=1)
                        # Reduce this CTA's 64 q-rows into dQ in RDQ_NCOL-column slices.
                        cl_qtok = qtok + cta_in_cluster * TM_H
                        for s in range(RDQ_STAGES):
                            cbase = s * RDQ_NCOL
                            col0 = (s % RDQ_NSLOT) * RDQ_NCOL
                            if cbase < DQ_COL:  # low half: tid<TM_H, frag[cbase:]
                                with k.if_(tid < TM_H):
                                    for c in range(RDQ_NCOL):
                                        k.reg_store(
                                            _sl(sdQ, (tid, col0 + c), (1, 1)),
                                            _sl(fragdQ, (cbase + c,), (1,)),
                                        )
                            else:  # high half: tid>=TM_H, frag[cbase-DQ_COL:]
                                with k.if_(tid >= TM_H):
                                    for c in range(RDQ_NCOL):
                                        k.reg_store(
                                            _sl(sdQ, (tid - TM_H, col0 + c), (1, 1)),
                                            _sl(fragdQ, (cbase - DQ_COL + c,), (1,)),
                                        )
                            k.wg_sync(barrier_id=1)
                            k.fence(kind=af, scope=FenceScope.CTA)
                            with k.if_(tid.eq(0)):
                                k.tma_reduce_add(
                                    dq_g,
                                    _sl(sdQ, (0, col0), (TM_H, RDQ_NCOL)),
                                    coords=(cl_qtok, head, cbase),
                                    shape=(TM_H, RDQ_NCOL),
                                    gmem_shape=(TM_H, 1, RDQ_NCOL),
                                    allow_nondet_reduce=True,
                                )
                                k.cp_async_bulk_commit_group()
                                k.cp_async_bulk_wait_group_read(RDQ_NSLOT - 1)
                            k.wg_sync(barrier_id=1)
                        # Drain remaining reduce-adds before reusing the staging ring.
                        if RDQ_NSLOT > 1:
                            with k.if_(tid.eq(0)):
                                k.cp_async_bulk_wait_group_read(0)
                        if deterministic:
                            # release the next cluster.
                            k.wg_sync(barrier_id=1)
                            with k.if_(tid.eq(0)):
                                k.fence(kind=FenceKind.MEMORY, scope=FenceScope.GPU)
                                k.gmem_atomic_add(
                                    dq_sem_g,
                                    coords=(batch, head, mb, cta_in_cluster),
                                    value=1,
                                    order="release",
                                )
                        if is_hd192:
                            # hd192: sdQ fully drained.
                            k.wg_sync(barrier_id=1)
                            k.mbarrier_arrive(bars["dQaccum_empty"])
                        continue
                    # ONE full t2r of all hdim cols into the fragment (flashattn cute.copy @3595).
                    _tc_ld(k, fragdQ, tdQ, hdim, col=0)
                    k.tcgen05_wait_ld()
                    # tdQ[mb] is now read into registers -> let mma overwrite these cols.
                    k.mbarrier_arrive(bars["dq_free"])
                    # DETERMINISTIC: acquire this Q-tile's turn BEFORE draining the reduce-add's.
                    if deterministic:
                        with k.if_(tid.eq(0)):
                            k.gmem_wait_eq(
                                dq_sem_g, coords=(batch, head, mb, cta_in_cluster), value=nb
                            )
                        k.wg_sync(barrier_id=1)  # rendezvous behind thread-0's acquire
                    # staged reduce loop: each stage r2s + reduce-add of its own 32-col slice.
                    for s in range(RDQ_STAGES):
                        cbase = s * RDQ_NCOL
                        col0 = (s % RDQ_NSLOT) * RDQ_NCOL
                        if varlen:
                            # D1: stage fragdQ for valid rows, an EXPLICIT 0.0 for OOB rows.
                            k.reg_fill(rzero, 0.0)
                            with k.if_(tid < TILE_M):
                                with k.if_((mb * TILE_M + tid) < vg.slen_q):
                                    for c in range(RDQ_NCOL):
                                        k.reg_store(
                                            _sl(sdQ, (tid, col0 + c), (1, 1)),
                                            _sl(fragdQ, (cbase + c,), (1,)),
                                        )
                                with k.if_((mb * TILE_M + tid) >= vg.slen_q):
                                    for c in range(RDQ_NCOL):
                                        k.reg_store(_sl(sdQ, (tid, col0 + c), (1, 1)), rzero)
                        else:
                            with k.if_(tid < TILE_M):
                                for c in range(RDQ_NCOL):
                                    k.reg_store(
                                        _sl(sdQ, (tid, col0 + c), (1, 1)),
                                        _sl(fragdQ, (cbase + c,), (1,)),
                                    )
                        k.wg_sync(barrier_id=1)
                        k.fence(kind=af, scope=FenceScope.CTA)
                        with k.if_(tid.eq(0)):
                            k.tma_reduce_add(
                                dq_g,
                                _sl(sdQ, (0, col0), (TILE_M, RDQ_NCOL)),
                                coords=(qtok, head, cbase),
                                shape=(TILE_M, RDQ_NCOL),
                                gmem_shape=(TILE_M, 1, RDQ_NCOL),
                                allow_nondet_reduce=True,
                            )
                            k.cp_async_bulk_commit_group()
                            k.cp_async_bulk_wait_group_read(RDQ_NSLOT - 1)
                        k.wg_sync(barrier_id=1)
                    # Drain remaining reduce-adds before reusing the staging buffer.
                    if RDQ_NSLOT > 1:
                        with k.if_(tid.eq(0)):
                            k.cp_async_bulk_wait_group_read(0)
                    k.wg_sync(barrier_id=1)
                    # DETERMINISTIC: after ALL reduce-adds have FULLY drained.
                    if deterministic:
                        with k.if_(tid.eq(0)):
                            k.fence(kind=FenceKind.MEMORY, scope=FenceScope.GPU)
                            k.gmem_atomic_add(
                                dq_sem_g,
                                coords=(batch, head, mb, cta_in_cluster),
                                value=1,
                                order="release",
                            )

    # relay (14): idle in 1-CTA. 2-CTA.
    with k.if_warp(RELAY_WARP), k.if_elected():
        with k.for_each_task(sched) as task:
            if use_2cta:
                # VARLEN: the relay's dS_cluster_full wait + dS_cluster_leader arrive must stay.
                _b, _h, _nb = task_geom(task)
                _vg = varlen_geom(_b) if varlen else None
                _mmin = m_block_min(_nb, _vg) if causal_skip else 0
                for mb in range(n_mb):
                    with _skip(mb, _mmin, _vg, _nb, cluster=True):
                        # wait our half landed.
                        k.mbarrier_wait(bars["dS_cluster_full"], phase=mb % 2)
                        k.mbarrier_arrive(k.mbar_ref(bars["dS_cluster_leader"], remote_coord=0))
    with k.if_warp(EMPTY_WARP):
        with k.for_each_task(sched) as task:
            pass

    # Teardown: every stream's pipeline work happens-before the dealloc.
    if use_2cta:
        k.cluster_sync()
    else:
        k.cta_sync()
    with k.if_warp(0):
        k.tmem_relinquish(cg)
        k.tmem_dealloc(0, N_COLS_TMEM, cg)
    return k.build()
