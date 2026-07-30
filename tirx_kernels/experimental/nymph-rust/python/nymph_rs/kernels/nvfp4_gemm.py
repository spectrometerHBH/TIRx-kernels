"""NVFP4 GEMM (Blackwell sm100 block-scaled fp4) expressed in Nymph IR.

Faithful port of ``tirx_kernels/gemm/nvfp4_gemm.py``. The operands are FP4
(``e2m1``, two values packed per ``uint8`` byte) and the GEMM is block-scaled at
block size 16: one ``e4m3`` scale factor per 16 contiguous K-elements, applied
to BOTH operands, with a final global ``alpha`` rescale in the epilogue. The
port keeps, at the same granularity as the fp16/bf16, FA4, and fp8-blockwise
ports:

- the cluster datapath: ``CTA_GROUP=2`` with ``CLUSTER_M=2`` — the cluster pair
  takes two adjacent M tiles (A split by M across the pair) and shares one
  ``MMA_N = cta_n * CTA_GROUP`` N band (B split by N across the pair, its
  block scales held full-band in both CTAs) — canon's exact tiling. The
  block-scaled tcgen05 dispatch derives N from the accumulator's MMA_N columns
  and requires ``SFB_rows >= N``; ``SFB_tmem`` is ``128 * SFB_n_chunks`` rows
  (``SFB_n_chunks = MMA_N // 128 = 2``), i.e. a 256-row SF band that satisfies
  ``SFB_rows = 256 >= N = 256`` (physically folded into 128 lanes × 2 column
  super-blocks). The per-CTA accumulator is ``(128, MMA_N)``;
- the warp split: one TMA-load warp (loads A/B AND the e4m3 scales — no permute
  warp; the TMA lands the scales in the tcgen05.cp-ready ``sf_smem_layout``), one
  MMA warp (issuing from the cluster leader only, elected so the cp/cp/gemm
  burst stays warp-converged), and one epilogue warpgroup;
- the pipeline protocol, mirroring canon's two TMA barriers: ``smem_full``
  (tile_full_bar — A+B arrive-expect-tx per k-tile) and ``sf_full``
  (scale_full_bar — SFA+SFB arrive-expect-tx); the MMA waits BOTH (and the
  peer's) before the cp/gemm. ``smem_empty`` (a tcgen05_commit multicast to both
  CTAs) frees the SMEM ring stage; the single-stage ``tmem_pipe`` (``tmem_full``
  = tcgen05_commit multicast, ``tmem_empty`` = both CTAs' epilogues arrive at
  the leader, first wait passes via the +1 phase offset);
- the data path: per k-tile (CTA_K=256) the MMA issues ONE block-scaled
  instruction over the full K tile (canon's one-issue full-K ``gemm_async``; the
  IR's dense k is an ordered run of k/16 atomic MMAs). The 16 e4m3 scale blocks
  (``SF_CTA_K = CTA_K//16``) are TMA'd every k-tile, ``tcgen05.cp``'d into TMEM
  (the folded 128-lane super-block layout), and read by the issue. The
  accumulator (one TMEM stage of MMA_N cols) is rescaled by ``alpha`` and cast
  to bf16 in the epilogue;
- the per-shape tuning knobs (canon's TIRX_CONFIGS): ``cta_n`` / ``epi_tile`` /
  ``smem_depth`` / ``d_depth`` / ``l2_group_size`` / ``load_cache_hint`` /
  ``epilogue`` (overlap | no_overlap | stmatrix) / ``maxnreg_*``,
  resolved through GEMM_CONFIGS (below) exactly like canon's per-shape table.

Sub-value physical-layout details (SMEM swizzles, the e2m1 nibble packing, the
SF-cell TMEM broadcast) are modeled logically, exactly like the sibling
GEMM/attention ports. ``alpha`` is applied as the epilogue rescale; on silicon
it is a runtime ``(1,)`` buffer, here a power-of-two value-model constant (the
value test fixes the global scales).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..builder import IRBuilder, SfTmemBand, TmemBand
from ..nymph_rs import (
    DType,
    FenceKind,
    FenceScope,
    Kernel,
    LaunchShape,
    MBarKind,
    MemorySpace,
    TensorSlice,
)

CTA_M = 128  # per-CTA A tile rows (one M tile)
CTA_N = 128  # per-CTA B tile rows (this CTA's half of the shared N band) — canon's CTA_N
CTA_GROUP = 2
CLUSTER_M = 2
# MMA_N = CTA_N * CTA_GROUP = 256 — canon's exact tiling. The shared N band the pair
# computes together. The block-scaled tcgen05 dispatch derives N from the accumulator's
# MMA_N=256 column count and requires SFB_rows >= N=256; canon's SFB_tmem is therefore
# `128 * SFB_n_chunks` rows where SFB_n_chunks = SFB_N//128 = 2 → a 256-row SFB_tmem.
# The emitted gemm_async is `tmem[:, 0:256]` (N=256), B operand = MMA_N//CTA_GROUP = 128
# rows (CTA_N), B_N*cta_group = 128*2 = 256 = N, and SFB rows=256 >= 256 all hold.
CTA_K = 256  # K per pipeline tile
MMA_N = CTA_N * CTA_GROUP  # 256, the shared N band the pair computes together
SF_BLOCK = 16  # one e4m3 scale per 16 K-elements
SF_CTA_K = CTA_K // SF_BLOCK  # 16 e4m3 scale bytes per row per k-tile (canon SF_CTA_K)
BLK_K_BYTES = CTA_K // 2  # packed fp4 bytes per row per k-tile (2 e2m1 per byte)
EPI_TILE = 64
TMEM_LD_SIZE = 8
# NOTE: a consumer-RAISE setmaxnreg (canon's / the fp16 port's (72, 208)) is dead at the
# OVERLAP epilogue's narrow epi_tile-wide fragment (ncu: 93->208 regs/thread left SM
# throughput flat). But the OPPOSITE — a consumer-DROP setmaxnreg.dec on the epilogue wg1
# (maxnreg_epilogue) — IS a 1024 lever: capping wg1's register budget tightens ptxas's
# epilogue allocation and shortens the epilogue critical path on this latency-bound square.
# Paired with epi_tile=32 it lifts 1024 from ~0.985 to ~0.995 (see the (1024,…) GEMM_CONFIGS
# entry). A producer-side setmaxnreg.inc on wg0 (maxnreg_producer) was also built + measured
# and REGRESSES (0.983 vs 0.986), so only the consumer cap is used. Canon's register-rich
# epilogue (172 regs/thread) only pays off paired with its NO-OVERLAP MMA_N-wide-fragment
# path, which nymph does not have (the no-overlap path needs ACC_DEPTH=2 double-buffered
# TMEM, but MMA_N=256 + the SF vectors at cols 448/464 already fill the 512-col TMEM, so a
# 2nd accumulator slot does not fit without relocating the SF TMEM).
ACC_DEPTH = 1  # accumulator TMEM stages (MMA_N=256 fills half of 512; one stage)
D_DEPTH = 2  # D_smem store ring depth (store pacing)
SMEM_DEPTH = 5  # SMEM pipeline depth (mirrors TIRx PIPE_DEPTH)
N_COLS_TMEM = 512
TILE_GROUPS_ROW_SIZE = 16
SM_NUMBER = 148
SF_BASE_COL = 448  # canon's fixed SF base column (the accumulator spans 0..MMA_N)


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


@dataclass(frozen=True, slots=True)
class NvFp4GemmConfig:
    m: int = 1024
    n: int = 1024
    k: int = 1024
    # alpha = 1 / (A_global_sf * B_global_sf), applied as the epilogue rescale.
    # A power of two keeps the value model bit-exact; the default 1.0 matches a
    # unit global scale.
    alpha: float = 1.0
    launch_shape: LaunchShape | None = None
    # Per-CTA N tile (canon's CTA_N knob). The cluster pair's shared N band is
    # MMA_N = cta_n * CTA_GROUP. cta_n=128 → MMA_N=256 (the default, 2 SF super-blocks);
    # cta_n=64 → MMA_N=128 (the SFB_N==128 single-super-block path, more N bands → more
    # clusters for small N — canon uses CTA_N=64 at the 1024 square). None = default 128.
    cta_n: int | None = None
    # Epilogue store-tile column width (canon's EPI_TILE; per-shape 16/32/64). The TMEM
    # drain frag and the reg->smem/TMA store band are EPI_TILE wide. None = default.
    epi_tile: int | None = None
    # SMEM pipeline depth (canon's PIPE_DEPTH; 4 or 5). Fewer stages = less SMEM, which
    # the 8192/16384 squares need to stay under the 227 KB cap. None = default 5.
    smem_depth: int | None = None
    # D-store ring depth (canon's WB_PIPE_DEPTH; the number of output SMEM stages the
    # epilogue can have in flight before it must `cp_async.bulk.wait_group`). A deeper ring
    # lets the drain issue more store-tile TMAs back-to-back (shorter drain tail on the
    # latency-bound small squares); it costs `d_depth * blk_m * epi_tile * 2` SMEM, so the
    # cap-tight squares keep it shallow. None = default D_DEPTH (2).
    d_depth: int | None = None
    # Persistent-scheduler L2 tile-group size (canon's L2_GROUP_SIZE): the number of
    # cluster-tile ROWS per group-major L2 locality band. The scheduler walks a cluster's
    # consecutive tasks down an l2_group_size-row band (shared A row-band / B col-band) so
    # back-to-back tiles hit in L2. Per-shape (canon: 12 @ 1024, 4 @ 2048/4096). None =
    # default TILE_GROUPS_ROW_SIZE.
    l2_group_size: int | None = None
    # L2 eviction policy on the A/B/SFA/SFB g2c loads (canon's `_tma_g2c_args` cache_hint).
    # The big cubes (8192/16384, many persistent tasks/cluster) NEED `evict_normal` to bound
    # the L2 cache-policy traffic — without it a TMA reads a stale tensormap and the launch
    # faults (error 719). But on the SMALL shapes (1024/2048/4096) the operand bands fit L2
    # comfortably, so the hint is dead weight that costs ~1% (ncu: it perturbs the L2 sector
    # policy with no hit-rate gain). So this is a per-shape knob: `None` (no hint) for the
    # small shapes, `"evict_normal"` for the cubes. Default `evict_normal` keeps the cubes
    # stable when no shape override is present. The epilogue store keeps its own evict_first.
    load_cache_hint: str | None = "evict_normal"
    # Epilogue schedule (canon's OVERLAP_EPI). Two TMEM-drain schedules over ONE accumulator:
    #   "overlap"    — fuse {load EPI_TILE band; wait_ld; scale+cast; store} per store tile,
    #                  reusing a narrow (128, EPI_TILE) reg fragment. The TMEM drain overlaps
    #                  with THIS tile's own reg->smem stores. Default; the big cubes use it.
    #   "no_overlap" — canon's OVERLAP_EPI=False: load the WHOLE MMA_N-wide accumulator into a
    #                  wide (128, MMA_N) reg fragment in one burst, ONE wait_ld, signal
    #                  tmem_empty (freeing TMEM for the MMA's NEXT tile EARLY), then one wide
    #                  scale+cast over the full band, then store all tiles. The TMEM drain now
    #                  overlaps with the NEXT TILE'S MMA (not this tile's stores), so the MMA
    #                  pipe never stalls on the epilogue — the lever that lifts canon's SM
    #                  throughput (ncu 4096: canon 57.8% vs nymph-overlap 51.5%) and its
    #                  register pressure (172 vs 95 regs/thread). Used by the 4096 square,
    #                  where the epilogue is the wall-clock residual.
    #   "stmatrix"   — the OVERLAP schedule, but the reg-frag datapath is canon's
    #                  `alloc_tcgen05_ldst_frag` ("16x256b") read frag + `alloc_cast_frag`
    #                  output frag, and the reg->smem store is `dispatch="ldstmatrix"` in
    #                  16-col chunks (stmatrix.x4). The plain overlap path emits a thread-axis
    #                  `wg_reg_tile` stored via `Tx.wg.copy` → STS, which carries 5.4x the
    #                  SMEM bank conflicts of canon's STSM (ncu: nymph 535 vs canon 100;
    #                  `smsp__inst_executed_op_shared_stsm` 0 vs 16384). stmatrix makes the
    #                  store STSM, matching canon's epilogue exactly. Used by the latency-bound
    #                  small squares (1024/2048) where the epilogue store is the residual.
    epilogue: str = "overlap"
    # Per-warpgroup register budget (canon's INVARIANT-I1b per-role `setmaxnreg`). The ncu
    # 1024 diff shows nymph at 86 regs/thread vs canon's 40 — but SMEM caps occupancy at
    # 1 cluster/SM (Block Limit Shared Mem = 1 < Block Limit Registers = 2), so cutting regs
    # cannot raise occupancy; this is a TEST knob to measure whether forcing fewer registers
    # moves the latency-bound 1024 anyway. `maxnreg_epilogue` sets the wg1 (epilogue/consumer)
    # warpgroup budget via `setmaxnreg`; None = no setmaxnreg (the default codegen budget).
    # `maxnreg_producer` sets the wg0 (TMA+MMA producer) budget at the start of the explicit
    # producer-warpgroup IR branch, before its nested warp roles — so all 4 wg0 warps issue
    # the collective op. Pairs with `maxnreg_epilogue`: the consumer `setmaxnreg.dec`
    # releases registers the producer `setmaxnreg.inc` claims (canon's INVARIANT-I1b
    # producer-drop / consumer-raise rebalance). None = no producer setmaxnreg.
    maxnreg_epilogue: int | None = None
    maxnreg_producer: int | None = None


def _cfg_cta_n(config: NvFp4GemmConfig) -> int:
    return config.cta_n if config.cta_n is not None else CTA_N


def _cfg_epi_tile(config: NvFp4GemmConfig) -> int:
    return config.epi_tile if config.epi_tile is not None else EPI_TILE


def _cfg_smem_depth(config: NvFp4GemmConfig) -> int:
    return config.smem_depth if config.smem_depth is not None else SMEM_DEPTH


def _cfg_d_depth(config: NvFp4GemmConfig) -> int:
    return config.d_depth if config.d_depth is not None else D_DEPTH


def _cfg_l2_group_size(config: NvFp4GemmConfig) -> int:
    return config.l2_group_size if config.l2_group_size is not None else TILE_GROUPS_ROW_SIZE


def nvfp4_task_config(tasks: int) -> NvFp4GemmConfig:
    """The canonical single-cluster examination setting, mirroring the fp16/bf16
    and fp8 ones: one m tile per CTA (M = CTA_M * CLUSTER_M = 256, two adjacent M
    tiles), k = 16384 (64 k-tiles per task), on ONE cluster pair
    (``launch_shape=(2,)``), with the task count varied through N = MMA_N * tasks
    (each pair-task covers one 256-wide N band)."""
    return NvFp4GemmConfig(
        m=CTA_M * CLUSTER_M, n=MMA_N * tasks, k=16384, alpha=1.0, launch_shape=(2,)
    )


CONFIGS = [
    {"m": s, "n": s, "k": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]
]

# Per-shape tuning knobs (mirrors canon's TIRX_CONFIGS). Keyed by (M, N, K). Only the
# knobs that differ from the defaults are listed; everything else uses NvFp4GemmConfig's
# defaults (cta_n=128 → MMA_N=256, epi_tile=64, smem_depth=5). The codegen stays generic;
# this is pure per-shape data. `cta_n=64` halves MMA_N (128) to double the N-band count,
# matching canon's CTA_N=64 at the 1024 square where the coarse MMA_N=256 tiling leaves
# the GPU under-occupied (16 vs 32 clusters).
GEMM_CONFIGS = {
    # 1024: the coarse default MMA_N=256 tiling yields only 16 clusters (32 CTAs) on
    # 148 SMs — the GPU is ~78% idle and the pipeline fill/drain latency is unhidden.
    # cta_n=64 halves MMA_N to 128, doubling the N bands → 32 clusters (canon's CTA_N=64
    # parallelism at this square), lifting the wall-clock ratio from ~0.79 to ~0.95.
    # 1024: cta_n=64 (32 clusters) + 2-row L2 band + no load cache hint + SMEM pool +
    # smem_depth=4 + the maxnreg_epilogue=96 / epi_tile=32 epilogue pair (see below). The
    # stmatrix epilogue datapath (epilogue="stmatrix") was BUILT, validated, and ncu-proven
    # to fire (the epilogue reg->smem store becomes STSM — `smsp__inst_executed_op_shared_stsm`
    # 0->16384, exactly canon's; plain-STS count drops to 640, byte-identical to canon's store
    # mix), BUT it REGRESSES this shape on the speed verdict (stmatrix epi64 0.976, stmatrix
    # epi32 0.968, both below the overlap path's 0.995). The no_overlap epilogue also regresses
    # (0.975). At 1024 only ~32 clusters occupy 148 SMs (0.43 waves), so the kernel is
    # single-task pipeline fill/drain LATENCY-bound; STSM's extra reg pressure hurts that tail.
    # So stay on the proven overlap path — the win comes from the register cap + epi_tile, below.
    (1024, 1024, 1024): {
        "cta_n": 64,
        "l2_group_size": 2,
        "load_cache_hint": None,
        # Canon's dynamic SMEM pool (one T.SMEMPool() the buffers alias into) instead of
        # N static T.alloc_buffer(scope="shared"). The static SMEM footprint drops to 0
        # (the whole 176 KB window becomes shared.dyn), giving the latency-bound 1024 more
        # occupancy headroom. Measured (tir-bench, clean B200): 0.961 -> 0.980.
        # Shallower SMEM pipe (4 vs the default 5 stages). With the pool form in place the
        # 32-byte-per-tile-shallower ring is a real occupancy lever at this latency-bound
        # square (the pipe is never the bottleneck at K=1024 / 4 k-tiles). Measured on top
        # of the pool: 0.980 -> 0.984. (2048 keeps depth 5 — there depth 4 was 0.989 vs
        # 0.992, and the big cubes keep their own depths.)
        "smem_depth": 4,
        # The TWO levers that lift 1024 from ~0.985 to ~0.995 (clearing the >=0.99 target),
        # found by re-profiling fresh WITH the SMEM pool active. The fresh ncu diff (nymph
        # vs canon, pool on) put the epilogue/consumer warpgroup's register footprint and
        # the epilogue store-band width as the dominant residual: nymph's per-task critical
        # path is the {tcgen05_ld; wait_ld; scale; cast; STS; TMA-store} epilogue chain, and
        # at ~32 clusters on 148 SMs (0.43 waves) the kernel is single-task latency-bound, so
        # shortening that chain is the only lever left (config space — l2_group_size,
        # smem_depth, d_depth, load_cache_hint, stmatrix/no_overlap epilogue — was re-swept
        # with the pool on and is flat-to-worse; cache_hint does NOT close the L1-hit-rate
        # diff; the stmatrix/no_overlap epilogues REGRESS this latency-bound square).
        #   * maxnreg_epilogue=96 — canon's INVARIANT-I1b per-role setmaxnreg on the wg1
        #     (epilogue/consumer) warpgroup. The `setmaxnreg.dec` tightens ptxas's epilogue
        #     register allocation, shortening the epilogue critical path. (A producer-side
        #     setmaxnreg.inc on wg0 was BUILT and measured — it REGRESSES, 0.983 vs 0.986 —
        #     so only the consumer cap is used.) Alone: 0.983 -> 0.986.
        #   * epi_tile=32 — canon's EPI_TILE at 1024 (vs the default 64). Halving the store
        #     band gives 4 narrower store tiles that pipeline better behind the tighter
        #     epilogue. Alone it is flat (0.983), but it STACKS with the register cap: the
        #     two together reach 0.995 (4-round interleaved A/B, reps=12, clean B200: baseline
        #     0.983-0.986 vs this 0.994-0.997). The maxnreg plateau is broad — 80..128 all
        #     give 0.992-0.994 with epi32 — so 96 (canon's effective epilogue budget) is the
        #     robust pick, not a knife-edge.
        "maxnreg_epilogue": 96,
        "epi_tile": 32,
    },
    # 2048: l2_group_size=4 matches canon's TIRX_CONFIGS[2048] L2_GROUP_SIZE=4 (the earlier
    # l2_group_size=2 was the peak under the OLD data-asymmetric bench; under the fair bench the
    # canon-matching 4-row band benches ~0.991 vs 2's ~0.987). + canon's dynamic SMEM pool
    # and evict_normal load hint. Keeps smem_depth=5 (depth 4 was 0.989).
    # maxnreg_epilogue=96 — canon's INVARIANT-I1b per-role setmaxnreg on the epilogue/consumer
    # warpgroup (same lever as 1024), picked by a nymph-vs-nymph head-to-head bench (median
    # +0.33%, 16/16 reps >= default; 96 vs 128 a tie, so 96 = canon's budget).
    # epi_tile=32 — canon's EPI_TILE. Under the OLD epilogue shape (fence-before-sync +
    # commit_group executed by all 4 warps) it REGRESSED here (0.974); after aligning the
    # epilogue to canon (wg_sync -> single-thread fence+store+commit, ncu-driven) it WINS:
    # A/B on the fixed epilogue: default 0.986 / epi32 0.993 / stmatrix 0.980 (still worse —
    # its reg pressure, not the store op, is the cost). The narrower store band also halves
    # the tcgen05.ld drain width toward canon's 4x-narrower LDTM pipeline. cta_n=64 still
    # regresses (0.935).
    (2048, 2048, 2048): {
        "l2_group_size": 4,
        "load_cache_hint": "evict_normal",
        "maxnreg_epilogue": 96,
        "epi_tile": 32,
    },
    # 4096: the epilogue is the wall-clock residual (ncu nymph-vs-canon at the OVERLAP
    # baseline: nymph executed 3.7% FEWER instructions yet ran 4.1% LONGER, SM throughput
    # 51.5% vs canon 57.8%, 95 vs 172 regs/thread). Canon runs OVERLAP_EPI=False + EPI_TILE=32
    # here. Two levers, both canon's:
    #   * epilogue="no_overlap" — frees the accumulator TMEM right after the wide load so the
    #     MMA's NEXT tile overlaps the TMEM drain (not this tile's stores) → MMA pipe stops
    #     stalling on the epilogue (0.954 -> 0.978).
    #   * epi_tile=32 — canon's EPI_TILE at 4096. The 32-wide store tile uses the 64B-atom D
    #     swizzle and a better-shaped TMA store; the dominant remaining lever (0.978 -> 0.999).
    # Re-tuned 2026-07 (post-let batch, 10-round A/B, canon time flat at 29.6us):
    #   * l2_group_size=4 — the big lever: 0.985 -> 1.005 (per-round 1.005-1.006). The old
    #     2-row pick came from a shared-GPU sweep under the data-asymmetric bench (which had
    #     4 at 0.957); under the fair bench + lets the 4-row band wins decisively. l2=8 is
    #     back at 0.985.
    #   * maxnreg_epilogue=240 — canon's register-pair direction (ncu had nymph 95 vs canon
    #     172 regs/thread): raising the epilogue warpgroup cap alone was 0.985 -> 0.990,
    #     stacked on l2=4: 1.005 -> 1.009/1.010 (224 and 240 tie within noise; 240 kept as
    #     the wider plateau). maxnreg_producer=56 REGRESSES (0.988), d_depth=3 regresses
    #     (0.980) — both stay at defaults.
    (4096, 4096, 4096): {
        "l2_group_size": 4,
        "load_cache_hint": None,
        "epilogue": "no_overlap",
        "epi_tile": 32,
        "maxnreg_epilogue": 240,
    },
    # 8192: the same no-overlap epilogue config as 4096 (see that comment). Under the FAIR
    # bench (both kernels fed identical real data) the default OVERLAP path benches 0.965 vs
    # canon; the no_overlap epilogue + epi_tile=32 + l2_group_size=2 trio takes it to 0.994
    # (l2_group_size=2 is the knob that lifts 0.980->0.994; maxnreg_epilogue is HARMFUL here,
    # unlike 1024). Note the win is the epilogue PIPELINE restructure (frees accumulator TMEM
    # early so the next MMA tile overlaps the drain), NOT an instruction-count reduction — ncu
    # shows the no_overlap kernel actually executes slightly MORE ops than the overlap default.
    # The PRE-EXISTING timing-sensitive launch fault (err 719) that kept this at defaults
    # surfaces only at FULL async OVERLAP w/ ~10+ tasks/cluster and is "hidden by any
    # serialization"; epilogue="no_overlap" IS such a serialization, and 40+ stress reps run
    # clean — so this config is believed to also sidestep the fault. (cosine 0.9910, matches
    # canon.)
    (8192, 8192, 8192): {
        "l2_group_size": 2,
        "load_cache_hint": None,
        "epilogue": "no_overlap",
        "epi_tile": 32,
    },
    # 16384: left at default (the same OVERLAP launch-fault caveat applies at the nymph
    # default here). The bench builds nymph with the SAME alpha value canon applies at
    # runtime — the residual asymmetry is only canon's alpha LOAD vs nymph's build-time
    # immediate (a capability difference, not a different computation).
}


def gemm_config_for(m: int, n: int, k: int) -> dict:
    """The per-shape knob overrides for (m, n, k); empty dict = all defaults."""
    return dict(GEMM_CONFIGS.get((m, n, k), {}))


# Cheap shapes for the protocol + value sweeps: different k-tile counts and tile
# counts, all on the single cluster datapath. Larger squares build/protocol only.
NVFP4_CONFIGS_SUPPORTED = [
    {"m": 256, "n": 256, "k": 256, "label": "256x256x256"},  # 1 k-tile, 1 tile
    {"m": 256, "n": 512, "k": 512, "label": "256x512x512"},  # 2 k-tiles, 2 N tiles
    {"m": 512, "n": 256, "k": 512, "label": "512x256x512"},  # 2 M tiles (2 pairs)
    {"m": 512, "n": 512, "k": 1024, "label": "512x512x1024"},  # 4 k-tiles, 2x2 tiles
    {"m": 1024, "n": 1024, "k": 1024, "label": "1024x1024x1024"},
]


def build_nvfp4_gemm(config: NvFp4GemmConfig = NvFp4GemmConfig()) -> Kernel:
    M, N, K = config.m, config.n, config.k
    _validate_config(config)
    cta_group = CTA_GROUP
    # Per-shape tiling knobs (canon's CTA_N / EPI_TILE / PIPE_DEPTH). All other module
    # constants are fixed-hardware (CTA_M, CTA_K, SF_BLOCK, …).
    cta_n = _cfg_cta_n(config)
    l2_group_size = _cfg_l2_group_size(config)
    mma_n = cta_n * CTA_GROUP  # the pair's shared N band (256 default; 128 for cta_n=64)
    epi_tile = _cfg_epi_tile(config)
    smem_depth = _cfg_smem_depth(config)
    d_depth = _cfg_d_depth(config)  # output store-ring depth (canon WB_PIPE_DEPTH)
    load_cache_hint = config.load_cache_hint  # per-shape L2 hint on the g2c loads (see config)
    blk_m = CTA_M  # per-CTA A rows (its own M tile)
    blk_n = cta_n  # per-CTA B rows (its half of the shared N band)
    sched_rows = _ceil_div(M, CTA_M)  # M tiles
    sched_cols = _ceil_div(N, mma_n)  # N bands
    k_tiles = K // CTA_K
    total_work = sched_rows * sched_cols
    store_tiles = mma_n // epi_tile

    launch_shape = config.launch_shape or (
        max(cta_group, min(SM_NUMBER, total_work) // cta_group * cta_group),
    )
    _validate_launch_shape(launch_shape, cta_group)
    pair_tasks = total_work // cta_group

    # packed fp4 operand tiles, e4m3 scale bytes (1B each, canon SFA_in layout), bf16 out
    a_tile_bytes = blk_m * BLK_K_BYTES
    b_tile_bytes = blk_n * BLK_K_BYTES
    sfa_tile_bytes = blk_m * SF_CTA_K  # per k-tile, this CTA's M rows x 16 e4m3 bytes
    # canon's SFB_N==MMA_N path: the FULL MMA_N-wide N band's scales live in EVERY CTA
    # (multicast), not split by N like the B operand. The band is MMA_N=256 rows (canon's
    # SFB_N=CTA_N*CTA_GROUP=256), so SFB_smem is (MMA_N, SF_CTA_K) and SFB_tmem holds all
    # 256 rows (physically folded into 128 lanes × 2 column super-blocks).
    sfb_tile_bytes = mma_n * SF_CTA_K  # per k-tile, the full N band's rows x 16 bytes
    d_tile_bytes = blk_m * epi_tile * 2

    a_off = 0
    b_off = a_off + smem_depth * a_tile_bytes
    sfa_off = b_off + smem_depth * b_tile_bytes
    sfb_off = sfa_off + smem_depth * sfa_tile_bytes
    d_off = sfb_off + smem_depth * sfb_tile_bytes
    data_end = d_off + d_depth * d_tile_bytes
    metadata_cursor = (data_end + 7) // 8 * 8
    smem_full_off = metadata_cursor
    metadata_cursor += smem_depth * 8
    sf_full_off = metadata_cursor
    metadata_cursor += smem_depth * 8
    smem_empty_off = metadata_cursor
    metadata_cursor += smem_depth * 8
    tmem_full_off = metadata_cursor
    metadata_cursor += ACC_DEPTH * 8
    tmem_empty_off = metadata_cursor
    metadata_cursor += ACC_DEPTH * 8
    tmem_fin_off = metadata_cursor
    metadata_cursor += 8
    tmem_addr_off = (metadata_cursor + 3) // 4 * 4
    smem_size_bytes = tmem_addr_off + 4

    k = IRBuilder(
        "nymph_nvfp4_gemm",
        num_warps=8,  # wg0 = tma/mma warps, wg1 = epilogue
        smem_size_bytes=smem_size_bytes,
        launch_shape=launch_shape,
        cluster_shape=(cta_group,),
    )
    # Operands are packed fp4: uint8[rows, K//2] (two e2m1 per byte), exactly the
    # TIRx A_packed/B_packed storage. Scales are e4m3 (one byte per 16-K block),
    # laid out (rows, K//16) exactly like canon's SFA_in/SFB_in — the codegen
    # synthesizes sf_smem_layout from the e4m3 dtype.
    a_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U8, shape=(M, K // 2))
    b_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.U8, shape=(N, K // 2))
    sfa_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.F8E4M3, shape=(M, K // SF_BLOCK))
    sfb_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.F8E4M3, shape=(N, K // SF_BLOCK))
    d_gmem = k.arg(space=MemorySpace.GMEM, dtype=DType.BF16, shape=(M, N))

    # Stage-major SMEM rings, indexed by a runtime pipeline stage (the continuous
    # PipelineState seq % depth, never reset per task).
    a_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U8,
        shape=(smem_depth, blk_m, BLK_K_BYTES),
        byte_offset=a_off,
    )
    b_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.U8,
        shape=(smem_depth, blk_n, BLK_K_BYTES),
        byte_offset=b_off,
    )
    # SF SMEM: e4m3 (row, SF_CTA_K) per stage, the same (CTA_M, K//16) tile canon's
    # SFA_smem holds. The codegen gives any e4m3 SMEM buffer canon's sf_smem_layout,
    # so the TMA lands the bytes in the tcgen05.cp-ready order (no permute warp).
    sfa_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.F8E4M3,
        shape=(smem_depth, blk_m, SF_CTA_K),
        byte_offset=sfa_off,
    )
    sfb_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.F8E4M3,
        shape=(smem_depth, mma_n, SF_CTA_K),
        byte_offset=sfb_off,
    )
    d_smem = k.tensor(
        space=MemorySpace.SMEM,
        dtype=DType.BF16,
        shape=(d_depth, blk_m, epi_tile),
        byte_offset=d_off,
    )

    # TMEM: accumulator (one MMA_N stage) at col 0; the e4m3 scale vectors at canon's
    # fixed SF base col (448), the SFB band derived from the SFA band's folded span.
    # TMEM is not a tensor: the bands are pure-Python descriptors, addressed by
    # absolute physical (lane, col).
    accum = TmemBand(col0=0, dtype=DType.F32)
    sfa_tmem = SfTmemBand(col0=SF_BASE_COL, rows=blk_m, nblocks=SF_CTA_K)
    # The FULL N band's B scales — canon's SFB_tmem = `128 * SFB_n_chunks` LOGICAL rows
    # with SFB_n_chunks = SFB_N//128 = MMA_N//128 (a 256-row SF band at the default).
    # The block-scaled gemm dispatch requires SFB_rows >= N=MMA_N, so the EMITTED SFB_tmem
    # is logically (MMA_N, SF_CTA_K). But TMEM is physically 128 lanes: the 256 logical
    # rows fold into SFB_n_chunks=2 column super-blocks — logical row r -> lane = r % 128,
    # col = col0 + (r // 128) * SF_CTA_K + block (canon's sf_tmem_layout packs the 2nd
    # super-block into cols 16..32). Folded at col0=464 the band spans cols 464..496,
    # clear of SFA's 448..464.
    sfb_tmem = SfTmemBand(col0=sfa_tmem.col0 + sfa_tmem.n_cols, rows=mma_n, nblocks=SF_CTA_K)

    # The epilogue drains a full epi_tile-wide column band per store tile in ONE
    # tcgen05_ld (canon's `Tx.wg.copy_async(reg_ldst[:, :], tmem[:, n:n+EPI_TILE])`),
    # then a SINGLE wait_ld / mul / cast over the whole band — not epi_tile//8 narrow
    # 8-col loads each with its own wait_ld (which serialized the TMEM drain on 8x the
    # async-load fences canon pays).
    #
    # OVERLAP (default): the reg fragments are epi_tile wide and reused per store tile.
    # NO-OVERLAP (canon's OVERLAP_EPI=False): the fragments span the WHOLE MMA_N band so
    # all store tiles' atoms live between the single wait_ld and the stores — canon's
    # `reg_all = alloc_tcgen05_ldst_frag("16x256b", (128, MMA_N), "float32")`. The codegen
    # declares the wg_reg_tile width from the max (offset+width) seen across the loads/
    # casts/stores (reg_widths in codegen.rs), so a mma_n-wide alloc + epi_tile-sliced
    # loads emit canon's wide fragment exactly.
    frag_width = mma_n if config.epilogue == "no_overlap" else epi_tile
    # STMATRIX epilogue: the reg frags are canon's `tcgen05.{ld,st}`-atom fragments
    # (`alloc_tcgen05_ldst_frag("16x256b", (128, W), "float32")` for the f32 read frag,
    # `alloc_cast_frag(<read>, "bfloat16")` for the bf16 output frag), so the reg->smem
    # store lowers to STSM (vs the thread-axis tile's plain STS — 5.4x the bank conflicts).
    # The marker is a `(instr_shape, cast_of)` tuple on the IR REG tensor (codegen-only;
    # the value model sees a plain REG tensor). cast_of=None = the read frag; cast_of=
    # accum_frag.id links the output frag to it (alloc_cast_frag inherits its layout).
    stmatrix_epi = config.epilogue == "stmatrix"
    accum_frag = k.tensor(
        space=MemorySpace.REG,
        dtype=DType.F32,
        shape=(frag_width,),
        reg_frag=("16x256b", None) if stmatrix_epi else None,
    )
    out_frag = k.tensor(
        space=MemorySpace.REG,
        dtype=DType.BF16,
        shape=(frag_width,),
        reg_frag=("16x256b", accum_frag.id) if stmatrix_epi else None,
    )

    smem_full = k.mbar(kind=MBarKind.TMA, byte_offset=smem_full_off, stages=smem_depth)
    smem_full_leader = k.mbar_ref(smem_full, remote_coord=0)
    # Separate SF-load completion barrier (canon's `scale_full_bar`, distinct from
    # the A/B `tile_full_bar`). Canon waits BOTH in the MMA before the cp/gemm; the
    # scale and tile loads land on different barriers. Mirror canon exactly — one
    # barrier for A/B, one for SFA/SFB — so the MMA orders against each.
    sf_full = k.mbar(kind=MBarKind.TMA, byte_offset=sf_full_off, stages=smem_depth)
    sf_full_leader = k.mbar_ref(sf_full, remote_coord=0)
    smem_empty = k.mbar(kind=MBarKind.TCGEN05, byte_offset=smem_empty_off, stages=smem_depth)
    tmem_full = k.mbar(kind=MBarKind.TCGEN05, byte_offset=tmem_full_off, stages=ACC_DEPTH)
    # tmem_empty: one elected thread per CTA's epilogue arrives (count=CTA_GROUP),
    # AFTER a warpgroup_sync that proves every warp's tcgen05.ld retired — the
    # sync is what lets the single arrive stand for all 128 threads' TMEM reads
    # (the checker's cross-warp publication rule; canon's all-thread arrive pays
    # 2*128 arrivals for the same proof).
    tmem_empty = k.mbar(kind=MBarKind.THREAD, byte_offset=tmem_empty_off, stages=ACC_DEPTH)
    tmem_empty_leader = k.mbar_ref(tmem_empty, remote_coord=0)
    # tmem_fin: canon's exact lightweight 2-CTA teardown handshake (its `tmem_finished`).
    # Issued by the epilogue warp (warp 4): each CTA's warp-4/lane-0 arrives at the PEER
    # CTA's barrier, then waits its OWN — so BOTH CTAs' epilogues finished their TMEM
    # accumulator reads before either CTA deallocs the shared TMEM region. Only this warp
    # waits; the rest exit. This is the canon-faithful teardown (a bare finalize
    # `cluster_sync` is needlessly heavy — it forces all 8 warps to converge at one PC,
    # and warp 0 is the MMA/loader, not the epilogue, so it wouldn't even order against
    # the epilogue's TMEM reads). Mirrors the proven fp16 overlap port's tmem_fin.
    tmem_fin = k.mbar(kind=MBarKind.THREAD, byte_offset=tmem_fin_off, stages=1)

    cta_id = k.cta_id()
    cta_rank = k.ctaid_in_cluster()
    task_space = k.task_space(grid=(pair_tasks,), fields=("pair_idx",))
    task_scheduler = k.scheduler(task_space)
    task_start = cta_id // cta_group
    task_step = k.launch_cta_count // cta_group

    ab_bytes = a_tile_bytes + b_tile_bytes
    sf_bytes = sfa_tile_bytes + sfb_tile_bytes

    def work_coords(task_id, cta_rank):
        """Canon's ClusterPersistentScheduler2D group-major mapping, at PAIR
        (cluster) granularity — the fix for L2 tile-grouping.

        Canon threads a CLUSTER's own linear tile index through the group-major
        order: cluster ``c``'s k-th tile is ``pair_idx = c + k*num_clusters``,
        and that single ``pair_idx`` is decoded over the cluster-tile grid
        (``pair_rows = sched_rows // CLUSTER_M`` rows × ``sched_cols`` N bands),
        so a cluster's CONSECUTIVE tasks stay inside one ``l2_group``-row band
        (a shared A row-band / B col-band → L2 reuse). The per-CTA M tile is then
        ``pair_row * CLUSTER_M + cta_rank`` (the pair's two adjacent M tiles).

        The OLD mapping decoded the GLOBAL ``work_idx = task_id*CLUSTER_M + rank``
        (group rows over CTA M tiles), so a cluster's stride of
        ``num_clusters*CLUSTER_M`` through ``work_idx`` scattered its back-to-back
        tiles across the whole M/N grid — no inter-task L2 band reuse.

        Returns (m_idx, n_idx) — the CTA's own M tile and the pair's N band."""
        pair_rows = sched_rows // CLUSTER_M  # cluster-tile rows (canon CLUSTER_M_TILES)
        if pair_rows <= l2_group_size:
            pair_row = task_id % pair_rows
            n_idx = task_id // pair_rows
        else:
            group_span = l2_group_size * sched_cols
            group_id = task_id // group_span
            within = task_id % group_span
            pair_row = group_id * l2_group_size + within % l2_group_size
            n_idx = within // l2_group_size
        m_idx = pair_row * CLUSTER_M + cta_rank
        return m_idx, n_idx

    with k.if_warp(0):
        # tmem_alloc is warp-collective (full warp 0); mbarrier.init is
        # per-thread, so one elected thread runs the inits.
        k.tmem_alloc(0, N_COLS_TMEM, addr_byte_offset=tmem_addr_off, cta_group=cta_group)
        with k.if_elected():
            for s in range(smem_depth):
                k.mbarrier_init(smem_full, count=1, stage=s)
                k.mbarrier_init(sf_full, count=1, stage=s)
                k.mbarrier_init(smem_empty, count=1, stage=s)
            for s in range(ACC_DEPTH):
                k.mbarrier_init(tmem_full, count=1, stage=s)
                k.mbarrier_init(tmem_empty, count=cta_group, stage=s)
            # canon's tmem_finished (init_full=1): one cross-CTA arrival releases the wait.
            k.mbarrier_init(tmem_fin, count=1, stage=0)

    # Prologue cross-CTA sync (canon's `T.ptx.fence.mbarrier_init` + the fused
    # cluster_sync): seal the mbarrier-init epoch so the inits and the TMEM alloc are
    # visible to the async engines, then converge every CTA thread of the cluster
    # before any role touches a peer mbarrier or the shared TMEM. Written EXPLICITLY
    # here (codegen never fabricates it) — the same fused no-overlap form the
    # fp16/bf16 port uses. Without it the MMA/epilogue race the still-pending
    # alloc/init on the peer CTA (illegal tcgen05).
    k.fence(kind=FenceKind.MBARRIER_INIT)
    k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
    k.cluster_sync()

    # Per-role register rebalance (canon's INVARIANT-I1b). The wg1 (epilogue/consumer)
    # `setmaxnreg.dec` is on its branch below; the wg0 (TMA+MMA producer) `setmaxnreg.inc`
    # is emitted HERE (before the warp-level branches diverge) so all 4 wg0 warps issue
    # the collective op together. Gated to the small-shape variant via the config; the
    # big shapes leave both None.
    with k.if_warpgroup(0):
        if config.maxnreg_producer is not None:
            k.set_maxnreg(config.maxnreg_producer)

        # ---- TMA producer (wg0/warp2 — canon's WarpRole.TMA) ----
        # The try_wait runs on the whole warp (all 32 lanes poll the barrier
        # together); the single-issue ops (expect_tx, tma_load) ride ONE elected
        # lane. This is canon's producer shape (whole-warp loop, per-op elect).
        with k.if_warp(2):
            with k.for_each_task(task_scheduler) as task:
                # `let` SSA binding (canon's while-loop `T.let` decode chain), replacing
                # the shuffle_sync broadcast: the immutable binding keeps the whole
                # local_iter -> stage/occ chain on ptxas's uniform datapath WITHOUT the
                # SHFL instruction (ncu: the shuffle form still paid vector math + R2UR
                # on the g2cluster lines). Value-model-identical: task_id is CTA-uniform,
                # so the per-lane eval agrees with the old broadcast.
                local_iter = k.let((task.task_id - task_start) // task_step)
                m_idx, n_idx = work_coords(task.task_id, cta_rank)
                a_m = m_idx * CTA_M  # this CTA's own M tile
                b_n = n_idx * mma_n + cta_rank * cta_n  # this CTA's half of the N band
                sf_n = n_idx * mma_n  # the FULL N band's B scales (rank-independent)
                # Rolled k-loop (canon's T.serial) — a Python range unrolls in the IR, which
                # ~doubles the emitted CUDA tcgen05 ops vs canon and breaks multi-k-tile.
                with k.for_loop(stop=k_tiles) as t:
                    seq = local_iter * k_tiles + t
                    stage = seq % smem_depth
                    occ = seq // smem_depth
                    k.mbarrier_wait(smem_empty, stage=stage, phase=(occ + 1) % 2)
                    with k.if_elected():
                        # canon's split arrive: A/B tx -> smem_full (tile_full_bar),
                        # SFA/SFB tx -> sf_full (scale_full_bar). Two barriers, each
                        # with its own expect_tx. Both CTAs explicitly target
                        # CTA 0; CTA 0 owns each full-cluster arrive/expect.
                        with k.if_(cta_rank.eq(0)):
                            k.mbarrier_arrive_expect_tx(
                                smem_full_leader, bytes=cta_group * ab_bytes, stage=stage
                            )
                            k.mbarrier_arrive_expect_tx(
                                sf_full_leader, bytes=cta_group * sf_bytes, stage=stage
                            )
                        kb = t * BLK_K_BYTES  # packed-fp4 byte column
                        # canon tags every g2c load with the L2 `evict_normal` policy (its
                        # `_tma_g2c_args`): a tile read once per k-tile should not pin an L2
                        # line the next tile evicts anyway. Bounding the L2 cache-policy
                        # traffic on the LOADS is the lever that stops the full-cube
                        # (8192/16384) launch fault — the operand bands no longer saturate
                        # L2 until a TMA reads a stale tensormap and the launch traps.
                        # Pairs with the epilogue store's `evict_first` (write-once
                        # output). Applied to A/B and SFA/SFB alike.
                        k.tma_load(
                            TensorSlice(
                                tensor=a_smem, offsets=(stage, 0, 0), shape=(1, blk_m, BLK_K_BYTES)
                            ),
                            a_gmem,
                            mbar=smem_full_leader,
                            coords=(a_m, kb),
                            shape=(1, blk_m, BLK_K_BYTES),
                            gmem_shape=(blk_m, BLK_K_BYTES),
                            mbar_stage=stage,
                            cache_hint=load_cache_hint,
                            cta_group=cta_group,
                        )
                        k.tma_load(
                            TensorSlice(
                                tensor=b_smem, offsets=(stage, 0, 0), shape=(1, blk_n, BLK_K_BYTES)
                            ),
                            b_gmem,
                            mbar=smem_full_leader,
                            coords=(b_n, kb),
                            shape=(1, blk_n, BLK_K_BYTES),
                            gmem_shape=(blk_n, BLK_K_BYTES),
                            mbar_stage=stage,
                            cache_hint=load_cache_hint,
                            cta_group=cta_group,
                        )
                        # SFA: this CTA's M rows; SFB: the full N band. e4m3 (rows, SF_CTA_K)
                        # straight from the (rows, K//16) GMEM at this k-tile's column band,
                        # exactly canon's SFA_in/SFB_in slice.
                        sf_k = t * SF_CTA_K
                        k.tma_load(
                            TensorSlice(
                                tensor=sfa_smem, offsets=(stage, 0, 0), shape=(1, blk_m, SF_CTA_K)
                            ),
                            sfa_gmem,
                            mbar=sf_full_leader,
                            coords=(a_m, sf_k),
                            shape=(1, blk_m, SF_CTA_K),
                            gmem_shape=(blk_m, SF_CTA_K),
                            mbar_stage=stage,
                            cache_hint=load_cache_hint,
                            cta_group=cta_group,
                        )
                        # SFB: the FULL MMA_N-wide N band (rank-independent sf_n), so both
                        # CTAs hold the same full-band scales (canon's SFB_N==MMA_N path).
                        # The B operand is still N-split (b_n), but its scales are not.
                        #
                        # NOTE on multicast: canon halves this band's load traffic by
                        # having each CTA read only its cta_rank-indexed half and MULTICAST
                        # it to both CTAs (cta_mask=pair_mask). The codegen + builder
                        # support that (multicast_cta_mask on tma_load, generalized
                        # leader-routing of the SF barrier), but it is NOT wired here: a
                        # `multicast::cluster` copy's transaction count completes once PER
                        # multicast destination on the GPU (canon's SFB_BYTES carries the
                        # `* CTA_GROUP` factor for exactly this), whereas the value-model's
                        # transaction accounting completes once per UNIQUE barrier cell
                        # (`apply_mbar_complete_tx` coalesces duplicate targets — the
                        # behaviour the `tma_multicast_group2_mbar_targets_are_deduplicated`
                        # interpreter test locks in). On the shared SFA+SFB barrier the two
                        # models cannot agree on a single IR expect_tx byte count (GPU
                        # wants the full band, the value model wants the issued half), so
                        # multicasting SFB here would either desync the GPU pipeline or
                        # break the bit-exact sim. The full-cube launch fault is instead
                        # resolved by the L2 `evict_normal` cache hint on the loads above
                        # (the same lever canon pairs with its multicast); the cubes are
                        # compute-bound, so the doubled SFB read is not the wall-clock
                        # bottleneck.
                        k.tma_load(
                            TensorSlice(
                                tensor=sfb_smem, offsets=(stage, 0, 0), shape=(1, mma_n, SF_CTA_K)
                            ),
                            sfb_gmem,
                            mbar=sf_full_leader,
                            coords=(sf_n, sf_k),
                            shape=(1, mma_n, SF_CTA_K),
                            gmem_shape=(mma_n, SF_CTA_K),
                            mbar_stage=stage,
                            cache_hint=load_cache_hint,
                            cta_group=cta_group,
                        )

        # ---- MMA (wg0/warp0, cluster leader only — canon's WarpRole.MMA) ----
        # MMA on warp 0 (the same warp that did the tcgen05.alloc), exactly like canon.
        # No permute warp (canon has none): the TMA lands the e4m3 scales in the
        # cp-ready layout, the MMA warp copies them SMEM->TMEM and issues ONE
        # block-scaled gemm over the full CTA_K tile, exactly like canon's execute_mma.
        # The physical folded SFB SF-TMEM band is (128, SF_CTA_K * SFB_N_CHUNKS) cells:
        # the MMA_N-wide band's 256 logical rows fold into SFB_N_CHUNKS column
        # super-blocks (the SfTmemBand rule); the cp dst and the mma sfb operand address
        # the band's folded physical base col0 (the MMA read region spans the whole
        # written footprint).
        # The WHOLE MMA worker loop runs single-issue (canon's `if elect_sync(): while ...`).
        # The B200 tensor pipe needs the cp/cp/gemm/commit burst issued by one converged
        # lane — per-op elect guards that reconverge the warp between tcgen05 issues stall
        # the async stream (a GPU deadlock). tcgen05.mma/cp are single-thread-ISSUE ops,
        # so the value model computes the MMA from SMEM/TMEM regardless of cohort size.
        with k.if_warp(0), k.if_elected():
            with k.for_each_task(task_scheduler) as task:
                # NO shuffle_sync here: __shfl_sync(0xffffffff) inside this single-lane
                # elected region is UB on GPU (the mask demands all 32 lanes converge;
                # reproduced as a flaky launch hang on the fp16 producer). local_iter is
                # a `let` SSA binding instead — the UB-free uniform mechanism: immutable
                # and single-assignment, so ptxas keeps the derived stage/phase chain on
                # the uniform datapath without any SHFL.
                local_iter = k.let((task.task_id - task_start) // task_step)
                with k.if_(cta_rank.eq(0)):
                    tmem_idx = local_iter % ACC_DEPTH
                    k.mbarrier_wait(
                        tmem_empty, stage=tmem_idx, phase=(local_iter // ACC_DEPTH + 1) % 2
                    )
                    acc_op = accum.at(0, tmem_idx * mma_n)

                    def mma_ktile(t, accum_flag):
                        # one k-tile: wait the staged loads, cp the e4m3 scales SMEM->TMEM,
                        # issue ONE block-scaled gemm over CTA_K, free the smem stage.
                        seq = local_iter * k_tiles + t
                        stage = seq % smem_depth
                        occ = seq // smem_depth
                        # smem_full starts EMPTY (parity 0) and is flipped 0->1 only when the
                        # loader's TMA arrive + all four complete-tx land. mbarrier_wait blocks
                        # while parity == phase and wakes on the flip, so the consumer must wait
                        # phase=occ%2 (block at the OLD parity until the load completes). The
                        # flipped (occ+1)%2 passes vacuously on the first occupancy — the cp/gemm
                        # then read the SF/operand SMEM before the load lands (the
                        # tcgen05_operand_overwrite_before_drain race the checker flags).
                        # canon waits BOTH the scale_full_bar (sf_full) and the tile_full_bar
                        # (smem_full) before the cp/gemm. Wait sf_full FIRST (canon's order)
                        # so the SF cp's source is ready, then smem_full for the A/B operands.
                        k.mbarrier_wait(sf_full, stage=stage, phase=occ % 2)
                        k.mbarrier_wait(smem_full, stage=stage, phase=occ % 2)
                        k.tcgen05_cp(
                            sfa_tmem.at(),
                            TensorSlice(
                                tensor=sfa_smem, offsets=(stage, 0, 0), shape=(1, blk_m, SF_CTA_K)
                            ),
                            cta_group=cta_group,
                        )
                        k.tcgen05_cp(
                            sfb_tmem.at(),
                            TensorSlice(
                                tensor=sfb_smem, offsets=(stage, 0, 0), shape=(1, mma_n, SF_CTA_K)
                            ),
                            cta_group=cta_group,
                        )
                        # canon's cluster gemm: n = MMA_N (the full 256-wide N band the pair
                        # computes together), m = MMA_M; each CTA supplies its own blk_n-row
                        # B half (b_smem holds it) and the 2-CTA MMA writes the per-CTA
                        # (128, MMA_N) accumulator. ONE block-scaled issue over the full
                        # CTA_K tile (the IR's dense k = an ordered run of k/16 atomic
                        # MMAs); SFA (128, SF_CTA_K), SFB (mma_n, SF_CTA_K).
                        a_op = TensorSlice(
                            tensor=a_smem, offsets=(stage, 0, 0), shape=(1, blk_m, BLK_K_BYTES)
                        )
                        b_op = TensorSlice(
                            tensor=b_smem, offsets=(stage, 0, 0), shape=(1, blk_n, BLK_K_BYTES)
                        )
                        k.tcgen05_mma(
                            acc_op,
                            a_op,
                            b_op,
                            m=256,
                            n=mma_n,
                            k=CTA_K,
                            accum=accum_flag,
                            cta_group=cta_group,
                            sfa=sfa_tmem.at(),
                            sfb=sfb_tmem.at(),
                            sf_e4m3=True,
                            sf_block=SF_BLOCK,
                            a_fp4=True,
                            b_fp4=True,
                        )
                        k.tcgen05_commit(
                            smem_empty, stage=stage, cta_group=cta_group, multicast_cta_mask=0b11
                        )

                    # Peel the first k-tile (accum=False), roll the rest with accum=True
                    # (canon's `accum=0` then `accum=1`). The rolled loop keeps the emitted
                    # CUDA tcgen05 op count at canon's level (a Python range would unroll it).
                    mma_ktile(0, False)
                    with k.for_loop(stop=k_tiles - 1) as ti:
                        mma_ktile(ti + 1, True)
                    k.tcgen05_commit(
                        tmem_full, stage=tmem_idx, cta_group=cta_group, multicast_cta_mask=0b11
                    )

    # ---- epilogue (wg1) ----
    # stmatrix runs the OVERLAP schedule (canon's 1024/2048 are OVERLAP_EPI=True + stmatrix
    # store); only "no_overlap" takes the wide-frag MMA-overlap path.
    no_overlap = config.epilogue == "no_overlap"
    # reg->smem store-chunk width: 16 cols for the stmatrix path (stmatrix.x4 granularity —
    # canon's `regs_to_smem` chunks `EPI_TILE // 16`), else the plain `Tx.wg.copy`/STS path's
    # TMEM_LD_SIZE (=8) sub-slice (the proven fp16/bf16 port's granularity).
    store_chunk = 16 if stmatrix_epi else TMEM_LD_SIZE

    def store_band(local_iter, d_m, d_n, ot, frag_off):
        """reg->smem (STSM granularity) + S->G TMA store of one EPI_TILE band. The reg
        source slice starts at ``frag_off`` (0 for the narrow overlap frag, ot*epi_tile
        for the wide no-overlap frag), exactly canon's `store_epi_chunk`."""
        store_iter = local_iter * store_tiles + ot
        with k.if_(store_iter >= d_depth):
            k.cp_async_bulk_wait_group_read(d_depth - 1)
        # The wg_sync rides OUTSIDE the runtime store_iter branch — canon nests it under
        # the same condition, but a sync whose scope depends on a runtime value is exactly
        # what the warp-model checker warns about (`unknown_filter_sync`); the
        # unconditional barrier is warp-uniform here and value-identical.
        k.wg_sync(barrier_id=10)
        d_stage = store_iter % d_depth
        # Store reg->smem in store_chunk-wide sub-slices (stmatrix: 16-col chunks → STSM;
        # else TMEM_LD_SIZE → STS, mirroring the proven fp16/bf16 port's epilogue store).
        for ki in range(epi_tile // store_chunk):
            k.reg_store(
                TensorSlice(
                    tensor=d_smem,
                    offsets=(d_stage, k.tid_in_wg(), ki * store_chunk),
                    shape=(1, 1, store_chunk),
                ),
                TensorSlice(
                    tensor=out_frag, offsets=(frag_off + ki * store_chunk,), shape=(store_chunk,)
                ),
            )
        # canon's epilogue-store order (and the fp16/bf16 port's): wg_sync FIRST so
        # every thread's reg->smem STS is done, THEN the single-thread proxy fence
        # right before the single-thread TMA store.
        k.wg_sync(barrier_id=10)
        with k.if_(k.tid_in_wg().eq(0)):
            k.fence(kind=FenceKind.ASYNC_PROXY, scope=FenceScope.CTA)
            k.tma_store(
                d_gmem,
                TensorSlice(tensor=d_smem, offsets=(d_stage, 0, 0), shape=(1, blk_m, epi_tile)),
                coords=(d_m, d_n + ot * epi_tile),
                shape=(1, blk_m, epi_tile),
                gmem_shape=(blk_m, epi_tile),
            )
        k.cp_async_bulk_commit_group()

    with k.if_warpgroup(1):
        # Consumer epilogue register cap (canon's INVARIANT-I1b per-role setmaxnreg),
        # a per-shape knob (see GEMM_CONFIGS); None = the default codegen budget.
        if config.maxnreg_epilogue is not None:
            k.set_maxnreg(config.maxnreg_epilogue)
        with k.for_each_task(task_scheduler) as task:
            # Same `let` SSA form as the TMA producer (was shuffle_sync): no SHFL,
            # the tmem_idx/phase decode chain stays on the uniform datapath.
            local_iter = k.let((task.task_id - task_start) // task_step)
            m_idx, n_idx = work_coords(task.task_id, cta_rank)
            d_m = m_idx * CTA_M
            d_n = n_idx * mma_n
            tmem_idx = local_iter % ACC_DEPTH
            k.mbarrier_wait(tmem_full, stage=tmem_idx, phase=(local_iter // ACC_DEPTH) % 2)
            acc_col0 = tmem_idx * mma_n
            if no_overlap:
                # canon's OVERLAP_EPI=False (reg_all = (128, MMA_N)): load the WHOLE band
                # into the wide reg fragment in one burst (epi_tile-sliced atoms into
                # distinct frag columns), ONE wait_ld for all atoms, signal tmem_empty
                # EARLY (the MMA can start the next tile's accumulate while we scale/cast/
                # store), then ONE wide mul + ONE wide cast, then store every band. The
                # TMEM drain overlaps with the NEXT tile's MMA, not this tile's stores.
                for ot in range(store_tiles):
                    k.tcgen05_ld(
                        TensorSlice(tensor=accum_frag, offsets=(ot * epi_tile,), shape=(epi_tile,)),
                        accum.at(0, acc_col0 + ot * epi_tile),
                        num=epi_tile,
                    )
                k.tcgen05_wait_ld()
                # Free the accumulator TMEM for the MMA's next tile BEFORE the scale/cast/
                # store (canon arrives tmem_empty right after wait.ld) — all TMEM reads are
                # complete once wait_ld passes, so the early arrive is the canon-faithful
                # MMA/epilogue overlap. The wg_sync proves every warp's reads retired
                # before the single elected arrive publishes them (see the mbar note).
                k.wg_sync(barrier_id=10)
                with k.if_(k.tid_in_wg().eq(0)):
                    k.mbarrier_arrive(tmem_empty_leader, stage=tmem_idx)
                k.reg_mul(accum_frag, accum_frag, config.alpha)
                k.reg_cvt(out_frag, accum_frag)
                for ot in range(store_tiles):
                    store_band(local_iter, d_m, d_n, ot, frag_off=ot * epi_tile)
            else:
                # OVERLAP: fuse {load EPI_TILE; wait_ld; scale+cast; store} per store tile,
                # reusing the narrow (128, EPI_TILE) frag. The TMEM drain overlaps with this
                # tile's own stores. Drain the band in ONE wide tcgen05_ld + ONE wait_ld +
                # ONE mul + ONE cast — not epi_tile//8 narrow 8-col loads each with its own
                # wait_ld (which serialized the drain on 8x the async-load fences canon pays).
                for ot in range(store_tiles):
                    k.tcgen05_ld(accum_frag, accum.at(0, acc_col0 + ot * epi_tile), num=epi_tile)
                    k.tcgen05_wait_ld()
                    k.reg_mul(accum_frag, accum_frag, config.alpha)
                    k.reg_cvt(out_frag, accum_frag)
                    if ot == store_tiles - 1:
                        # Free the accumulator after the LAST tile's reads: the wg_sync
                        # proves every warp's tcgen05.ld retired before the elected
                        # arrive publishes them (see the mbar note).
                        k.wg_sync(barrier_id=10)
                        with k.if_(k.tid_in_wg().eq(0)):
                            k.mbarrier_arrive(tmem_empty_leader, stage=tmem_idx)
                    store_band(local_iter, d_m, d_n, ot, frag_off=0)
        k.cp_async_bulk_wait_group_read(0)
        k.wg_sync(barrier_id=10)

    # TMEM teardown via canon's tmem_finished 2-CTA handshake (NOT a bare cluster_sync).
    # Issued by the EPILOGUE warp (warp 4 = canon's WarpRole.EPILOGUE), which reaches the
    # finalize only after its whole warpgroup passed the post-loop `wg_sync(10)` — i.e.
    # all of wg1's accumulator (TMEM) reads are done. It arrives at the PEER CTA's tmem_fin,
    # then waits its OWN, so BOTH CTAs' epilogues finished reading the shared TMEM before
    # either deallocs. Only this one warp waits; the rest exit. This is canon's exact
    # `tmem_finished` teardown (also the proven fp16 overlap port's tmem_fin), and unlike a
    # bare `cluster_sync` it picks the EPILOGUE warp (warp 0 is the MMA/loader, whose dealloc
    # would not order against the epilogue's TMEM reads).
    epilogue_warp = 4  # wg1's first warp (num_warps=8: wg0=0-3, wg1=4-7); canon's EPILOGUE
    with k.if_warp(epilogue_warp):
        with k.if_elected():
            k.mbarrier_arrive(k.mbar_ref(tmem_fin, remote_coord=1 - cta_rank), stage=0)
        k.mbarrier_wait(tmem_fin, stage=0, phase=0)
        k.tmem_relinquish(cta_group)
        k.tmem_dealloc(0, N_COLS_TMEM, cta_group)

    return k.build()


def _validate_config(config: NvFp4GemmConfig) -> None:
    for name in ("m", "n", "k"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"nvfp4_gemm {name} must be a positive integer")
    if config.k % CTA_K != 0:
        raise ValueError(f"nvfp4_gemm k must be a multiple of {CTA_K}")
    cta_n = _cfg_cta_n(config)
    if cta_n not in (64, 128):
        raise ValueError("nvfp4_gemm cta_n must be 64 (MMA_N=128) or 128 (MMA_N=256)")
    mma_n = cta_n * CTA_GROUP
    if config.n % mma_n != 0:
        raise ValueError(f"nvfp4_gemm n must be a multiple of MMA_N={mma_n} (cta_n={cta_n})")
    epi_tile = _cfg_epi_tile(config)
    if mma_n % epi_tile != 0 or epi_tile % TMEM_LD_SIZE != 0:
        raise ValueError("nvfp4_gemm epi_tile must divide MMA_N and be a multiple of TMEM_LD_SIZE")
    d_depth = _cfg_d_depth(config)
    if not isinstance(d_depth, int) or isinstance(d_depth, bool) or d_depth < 2:
        raise ValueError("nvfp4_gemm d_depth must be an integer >= 2")
    sched_rows = _ceil_div(config.m, CTA_M)
    if sched_rows % CTA_GROUP != 0:
        raise ValueError("nvfp4_gemm requires an even number of M tiles per cluster pair")
    # The group-major scheduler decodes a cluster's linear task index over the cluster-tile
    # grid (pair_rows = sched_rows // CLUSTER_M rows). The bijection requires the band split
    # to be exact: either one short column-walk (pair_rows <= l2_group_size) or whole groups
    # with no tail (pair_rows % l2_group_size == 0).
    l2_group_size = _cfg_l2_group_size(config)
    if not isinstance(l2_group_size, int) or isinstance(l2_group_size, bool) or l2_group_size < 1:
        raise ValueError("nvfp4_gemm l2_group_size must be a positive integer")
    pair_rows = sched_rows // CLUSTER_M
    if pair_rows > l2_group_size and pair_rows % l2_group_size != 0:
        raise ValueError("nvfp4_gemm l2_group_size must divide the cluster-tile row count")
    if config.epilogue not in ("overlap", "no_overlap", "stmatrix"):
        raise ValueError('nvfp4_gemm epilogue must be "overlap", "no_overlap", or "stmatrix"')
    if config.epilogue == "stmatrix" and epi_tile % 16 != 0:
        # The reg->smem store chunks in 16-col slices (stmatrix.x4 granularity), and the
        # f32 read frag's "16x256b" atom needs cols divisible by 8 — epi_tile=16/32/64 all
        # satisfy both; epi_tile=8 would not (and is not a configured value anyway).
        raise ValueError("nvfp4_gemm stmatrix epilogue requires epi_tile a multiple of 16")


def _validate_launch_shape(launch_shape: LaunchShape, cta_group: int) -> None:
    if not isinstance(launch_shape, tuple) or len(launch_shape) != 1:
        raise ValueError("nvfp4_gemm requires a 1D launch_shape")
    count = launch_shape[0]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("nvfp4_gemm launch_shape[0] must be a positive integer")
    if count % cta_group != 0:
        raise ValueError("nvfp4_gemm launch_shape[0] must be divisible by cta_group")


# ---------------------------------------------------------------------------
# Bench-suite interface (see bench/nymph_bench_guide.md). Lives IN the kernel
# file, mirroring canon's single-file structure: pure data at module level;
# bench-only deps (tvm / torch / canon) imported lazily so the nymph_rs
# package stays importable without them (CPU-only value sim, wheel installs).
# ---------------------------------------------------------------------------

KERNEL_META = {"name": "nymph_nvfp4_gemm", "category": "experimental", "compute_capability": 10}
BENCH_CONFIGS = [
    {"M": s, "N": s, "K": s, "label": f"{s}x{s}x{s}"} for s in [1024, 2048, 4096, 8192, 16384]
]


def _compile_nymph(M, N, K, alpha):
    import importlib.util
    import os
    import tempfile

    import tvm

    from ..nymph_rs import kernel_to_tirx_source

    src = kernel_to_tirx_source(
        build_nvfp4_gemm(NvFp4GemmConfig(m=M, n=N, k=K, alpha=alpha, **gemm_config_for(M, N, K)))
    )
    p = os.path.join(tempfile.mkdtemp(prefix="nymph_nvfp4_"), "g.py")
    with open(p, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("nymph_nvfp4_emitted", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return tvm.compile(
        tvm.IRModule({"main": m.main}), tvm.target.Target("cuda"), tir_pipeline="tirx"
    )


def run_bench(M, N, K, *, warmup=None, repeat=None, timer=None, **kwargs):
    """Bench canon vs nymph with the bench-suite's exact methodology.

    Canon is imported read-only; both impls go through the identical
    ``bench()`` call, and each is correctness-gated before timing (a ratio of
    two kernels computing different results fails loudly).
    """
    import torch

    import tvm
    from tirx_kernels.gemm.nvfp4_gemm import prepare_data, tir_ws_kernel
    from tvm.tirx.bench import bench

    target = tvm.target.Target("cuda")
    with target:
        canon = tvm.compile(
            tvm.IRModule({"main": tir_ws_kernel(M, N, K)}), target, tir_pipeline="tirx"
        )
    A, B, Asf, Bsf, alpha, Cref = prepare_data(M, N, K)
    # Same math on both sides: canon applies the runtime-alpha rescale, nymph bakes
    # the SAME alpha as a build-time immediate (the remaining asymmetry is the alpha
    # LOAD, a known capability difference — not a different computation).
    nymph = _compile_nymph(M, N, K, float(alpha))
    at = torch.tensor([float(alpha)], device="cuda", dtype=torch.float)
    Ae, Be = Asf.view(torch.float8_e4m3fn), Bsf.view(torch.float8_e4m3fn)
    oc = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    on = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    funcs = {"tir": lambda: canon(A, B, Asf, Bsf, at, oc), "tirx": lambda: nymph(A, B, Ae, Be, on)}
    for fn in funcs.values():
        fn()
    torch.cuda.synchronize()
    for name, out in (("tir", oc), ("tirx", on)):
        cos = torch.nn.functional.cosine_similarity(out.float().flatten(), Cref.flatten(), dim=0)
        if cos < 0.98:
            raise AssertionError(f"{name} output diverges from reference (cosine={cos:.4f})")
    return bench(funcs, warmup=warmup, repeat=repeat, timer=timer, **kwargs)


def register_bench_interface() -> None:
    """Self-register into the bench-suite kernel cache (see bench/nymph_bench_guide.md)."""
    import sys

    from tirx_kernels.registry import _KERNEL_CACHE

    _KERNEL_CACHE[KERNEL_META["name"]] = sys.modules[__name__]
