"""Validation-parity tests: the Rust validator must reject the same invalid IR
the original ir.py did.

This captures the value of test_builder.py's ~180 `pytest.raises` assertions
without re-implementing Python-dataclass introspection: each test builds an
invalid kernel from the nymph_rs constructors and asserts Kernel construction
(which validates) raises with the expected message.
"""

import pytest

n = pytest.importorskip("nymph_rs")

# ---- small tensor/slice helpers -------------------------------------------


def smem(shape, dtype=n.DType.F16):
    return n.Tensor(space=n.MemorySpace.SMEM, dtype=dtype, shape=shape, byte_offset=0)


def tmem_op(row=0, col=0, dtype=n.DType.F32):
    # A TMEM reference is an absolute physical (lane, col) + cell dtype — no tensor.
    return n.TmemOperand(row, col, dtype)


def gmem(shape, dtype=n.DType.F16):
    return n.Tensor(space=n.MemorySpace.GMEM, dtype=dtype, shape=shape)


def reg(shape, dtype=n.DType.F32):
    return n.Tensor(space=n.MemorySpace.REG, dtype=dtype, shape=shape)


def make(body, *, num_warps=4, launch=(2,), cluster=(1,), args=(), tmem_cg=None):
    """Build a kernel (validates on construction). `tmem_cg` prepends a warp-scope
    512-column TMEM alloc (with that cta_group) so TMEM operands have a live band.
    The default geometry is a single CTA (kernel cta_group=1), matching the
    cg=1 op vocabulary these tests use; validate pins every TMEM lifecycle op
    to that kernel-level group."""
    if tmem_cg is not None:
        body = (n.KernelInit(body=(n.TmemAlloc(0, 512, cta_group=tmem_cg),), warp=0), *body)
    return n.Kernel(
        name="t",
        args=args,
        body=tuple(body),
        num_warps=num_warps,
        smem_size_bytes=1 << 20,
        launch_shape=list(launch),
        cluster_shape=list(cluster),
    )


def mma_operands():
    """A valid cta_group=1 MMA's (dst, a, b) operands — m=128, n=256, k=16."""
    return tmem_op(0, 0), smem([128, 16])[:, :], smem([256, 16])[:, :]


# ---- kernel geometry -------------------------------------------------------


def test_rejects_num_warps_not_multiple_of_4():
    with pytest.raises(ValueError, match="num_warps"):
        make([], num_warps=6)


def test_rejects_launch_shape_rank_too_high():
    with pytest.raises(ValueError, match="rank must be in"):
        make([], launch=(2, 2, 2, 2), cluster=(1, 1, 1, 1))


def test_rejects_launch_not_divisible_by_cluster():
    with pytest.raises(ValueError, match="divisible by cluster_shape"):
        make([], launch=(6,), cluster=(4,))


def test_rejects_smem_tensor_outside_pool():
    tensor = n.Tensor(space=n.MemorySpace.SMEM, dtype=n.DType.U32, shape=[4], byte_offset=16)
    with pytest.raises(ValueError, match="byte range exceeds"):
        n.Kernel(
            name="bad_smem_bounds",
            body=(n.TensorDef(tensor),),
            num_warps=4,
            smem_size_bytes=20,
            launch_shape=[1],
            cluster_shape=[1],
        )


# ---- tcgen05_mma -----------------------------------------------------------


def test_rejects_mma_dst_not_tmem():
    _, a, b = mma_operands()
    dst = smem([128, 256])[:, :]
    with pytest.raises(TypeError, match="TmemOperand"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)], tmem_cg=1)


def test_rejects_mma_operand_not_smem_or_tmem():
    dst, _, b = mma_operands()
    a = gmem([128, 16])[:, :]
    with pytest.raises(ValueError, match="slice operand must be SMEM"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)], tmem_cg=1)


def test_accepts_mma_tmem_operand():
    dst, _, b = mma_operands()
    a = tmem_op(0, 0, dtype=n.DType.F16)
    make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)], tmem_cg=1)


def test_rejects_mma_operand_dtype():
    dst, _, b = mma_operands()
    a = smem([128, 16], dtype=n.DType.F32)[:, :]  # f32 SMEM operand is still rejected
    with pytest.raises(ValueError, match="operand dtype must be f16, bf16, f8e4m3"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)], tmem_cg=1)


def test_rejects_mma_dst_dtype():
    _, a, b = mma_operands()
    dst = tmem_op(0, 0, dtype=n.DType.F16)
    with pytest.raises(ValueError, match="dst dtype must be f32"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)], tmem_cg=1)


def test_rejects_mma_bad_k():
    dst, a, b = mma_operands()
    with pytest.raises(ValueError, match="positive multiple of 16"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=8)], tmem_cg=1)


# ---- tcgen05_mma hardware-coupled rules (fail-closed) -----------------------


def mma_f8_operands():
    """A valid block-scaled f8 (UE8M0) MMA operand set: cg1, m=128, n=32, k=32."""
    dst = tmem_op(0, 0)
    a = smem([128, 32], dtype=n.DType.F8E4M3)[:, :]
    b = smem([32, 32], dtype=n.DType.F8E4M3)[:, :]
    sfa = tmem_op(0, 32, dtype=n.DType.U32)
    sfb = tmem_op(0, 36, dtype=n.DType.U32)
    return dst, a, b, sfa, sfb


def mma_fp4_operands():
    """A valid NVFP4 MMA operand set: cg2, m=256, n=256, k=64 (32 packed bytes)."""
    dst = tmem_op(0, 0)
    a = smem([128, 32], dtype=n.DType.U8)[:, :]
    b = smem([128, 32], dtype=n.DType.U8)[:, :]
    sfa = tmem_op(0, 448, dtype=n.DType.F8E4M3)
    sfb = tmem_op(0, 464, dtype=n.DType.F8E4M3)
    return dst, a, b, sfa, sfb


def fp4_kwargs(**overrides):
    kw = dict(m=256, n=256, k=64, cta_group=2, sf_e4m3=True, sf_block=16, a_fp4=True, b_fp4=True)
    kw.update(overrides)
    return kw


def test_accepts_block_scaled_f8_and_nvfp4_mma():
    dst, a, b, sfa, sfb = mma_f8_operands()
    make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=32, k=32, sfa=sfa, sfb=sfb)], tmem_cg=1)
    dst, a, b, sfa, sfb = mma_fp4_operands()
    make(
        [n.Tcgen05Mma(dst=dst, a=a, b=b, sfa=sfa, sfb=sfb, **fp4_kwargs())], tmem_cg=2, cluster=(2,)
    )


def test_accepts_mma_dense_full_k():
    """Dense f16/bf16 k is any positive multiple of the k=16 atom — one IR MMA
    is an ordered run of k/16 atomic MMAs (canon's one-issue full-K gemm_async)."""
    dst = tmem_op(0, 0)
    a = smem([128, 64])[:, :]
    b = smem([256, 64])[:, :]
    make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=64)], tmem_cg=1)


def test_rejects_mma_dense_k_not_multiple_of_16():
    dst, a, b = mma_operands()
    for k in (24, 40):
        with pytest.raises(ValueError, match="positive multiple of 16"):
            make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=k)], tmem_cg=1)


def test_rejects_mma_block_scaled_f8_k():
    dst, a, b, sfa, sfb = mma_f8_operands()
    with pytest.raises(ValueError, match="block-scaled f8 k must be 32, 128, or 256"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=32, k=16, sfa=sfa, sfb=sfb)], tmem_cg=1)


def test_rejects_mma_fp4_k():
    dst, a, b, sfa, sfb = mma_fp4_operands()
    with pytest.raises(ValueError, match=r"fp4 \(mxf4\) k must be 64, 128, or 256"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, sfa=sfa, sfb=sfb, **fp4_kwargs(k=32))], tmem_cg=2)


def test_rejects_mma_m64_cg1_scale_mode():
    dst = tmem_op(0, 0)
    a = smem([64, 32], dtype=n.DType.F8E4M3)[:, :]
    b = smem([32, 32], dtype=n.DType.F8E4M3)[:, :]
    sfa = tmem_op(0, 32, dtype=n.DType.U32)
    sfb = tmem_op(0, 36, dtype=n.DType.U32)
    with pytest.raises(ValueError, match="m=64 cta_group=1 does not support block-scaled"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=64, n=32, k=32, sfa=sfa, sfb=sfb)], tmem_cg=1)


def test_rejects_mma_fp4_transposed():
    dst, a, b, sfa, sfb = mma_fp4_operands()
    with pytest.raises(ValueError, match="does not support trans_a/trans_b"):
        make(
            [n.Tcgen05Mma(dst=dst, a=a, b=b, sfa=sfa, sfb=sfb, **fp4_kwargs(trans_b=True))],
            tmem_cg=2,
        )


def test_rejects_mma_fp4_shape():
    dst = tmem_op(0, 0)
    a = smem([64, 32], dtype=n.DType.U8)[:, :]
    b = smem([32, 32], dtype=n.DType.U8)[:, :]
    sfa = tmem_op(0, 448, dtype=n.DType.F8E4M3)
    sfb = tmem_op(0, 464, dtype=n.DType.F8E4M3)
    with pytest.raises(ValueError, match="fp4 requires"):
        make(
            [
                n.Tcgen05Mma(
                    dst=dst,
                    a=a,
                    b=b,
                    m=64,
                    n=32,
                    k=64,
                    sfa=sfa,
                    sfb=sfb,
                    sf_e4m3=True,
                    sf_block=16,
                    a_fp4=True,
                    b_fp4=True,
                )
            ],
            tmem_cg=1,
        )


def test_rejects_mma_tmem_operand_dtype():
    dst, a, _b = mma_operands()
    b_e4m3_tmem = tmem_op(0, 0, dtype=n.DType.F8E4M3)
    with pytest.raises(ValueError, match="b TMEM operand dtype must be f16, bf16, or f32"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b_e4m3_tmem, m=128, n=256, k=16)], tmem_cg=1)
    dst, a, b, sfa, sfb = mma_fp4_operands()
    a_u8_tmem = tmem_op(0, 0, dtype=n.DType.U8)
    with pytest.raises(ValueError, match="a TMEM operand dtype must be f16, bf16, or f32"):
        make([n.Tcgen05Mma(dst=dst, a=a_u8_tmem, b=b, sfa=sfa, sfb=sfb, **fp4_kwargs())], tmem_cg=2)


def test_accepts_mma_tmem_b_operand_f32():
    # The accumulator-readback abstraction: an f32 TMEM B operand mixes with an
    # f16/bf16 SMEM A (the test_tmem_operand_mma / GDN configuration).
    dst = tmem_op(0, 0)
    a = smem([128, 16], dtype=n.DType.BF16)[:, :]
    b = tmem_op(0, 32)
    make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=16, k=16)], tmem_cg=1)


def test_rejects_mma_mixed_dtype_not_b16():
    dst = tmem_op(0, 0)
    a = smem([128, 32], dtype=n.DType.F8E4M3)[:, :]
    b = tmem_op(0, 32)  # f32 TMEM
    sfa = tmem_op(0, 64, dtype=n.DType.U32)
    sfb = tmem_op(0, 68, dtype=n.DType.U32)
    with pytest.raises(ValueError, match="a and b operand dtype must match"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=32, k=32, sfa=sfa, sfb=sfb)], tmem_cg=1)


def test_rejects_mma_lane_align_wrong_layout():
    dst, a, b = mma_operands()
    with pytest.raises(ValueError, match=r"lane_align != 0 requires cta_group=1 and m=64"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16, lane_align=16)], tmem_cg=1)


def test_rejects_mma_nvfp4_sf_byte():
    dst, a, b, sfa, sfb = mma_fp4_operands()
    with pytest.raises(ValueError, match="sf_byte must be 0 for sf_e4m3"):
        make(
            [n.Tcgen05Mma(dst=dst, a=a, b=b, sfa=sfa, sfb=sfb, **fp4_kwargs(sf_byte=2))], tmem_cg=2
        )


# ---- tcgen05_cp --------------------------------------------------------------


def test_accepts_tcgen05_cp_u32_and_e4m3():
    dst = tmem_op(0, 0, dtype=n.DType.U32)
    src = smem([512], dtype=n.DType.U32)[:]
    make([n.Tcgen05Cp(dst=dst, src=src)], tmem_cg=1)
    dst = tmem_op(0, 0, dtype=n.DType.F8E4M3)
    src = smem([1, 256, 16], dtype=n.DType.F8E4M3)[:, :, :]
    make([n.Tcgen05Cp(dst=dst, src=src)], tmem_cg=1)


def test_rejects_tcgen05_cp_dtype_mismatch():
    dst = tmem_op(0, 0, dtype=n.DType.U32)
    src = smem([512], dtype=n.DType.F8E4M3)[:]
    with pytest.raises(ValueError, match="tcgen05_cp dst and src dtype must match"):
        make([n.Tcgen05Cp(dst=dst, src=src)], tmem_cg=1)


def test_rejects_tcgen05_cp_u32_src_not_1d():
    dst = tmem_op(0, 0, dtype=n.DType.U32)
    src = smem([128, 4], dtype=n.DType.U32)[:, :]
    with pytest.raises(ValueError, match="u32 src must be effectively 1-D"):
        make([n.Tcgen05Cp(dst=dst, src=src)], tmem_cg=1)


def test_rejects_tcgen05_cp_dst_row_nonzero():
    # The copy is lane-anchored: the dst operand's row (base lane) must be 0. (The
    # old dst-tensor column-extent multiple check is gone — the de-tensored operand
    # carries no extent; the fold is implied by the src tile.)
    dst = tmem_op(8, 0, dtype=n.DType.F8E4M3)
    src = smem([1, 64, 6], dtype=n.DType.F8E4M3)[:, :, :]
    with pytest.raises(ValueError, match=r"tcgen05_cp dst row \(lane\) must be 0"):
        make([n.Tcgen05Cp(dst=dst, src=src)], tmem_cg=1)


# ---- shuffle_sync ------------------------------------------------------------


def test_rejects_shuffle_sync_static_lane_out_of_range():
    for bad_lane in (-1, 32):
        v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
        with pytest.raises(ValueError, match=r"src_lane must be in \[0, 32\)"):
            make([n.ShuffleSync(var=v, src=5, src_lane=bad_lane)])


def test_rejects_shuffle_sync_in_elected_scope():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    body = [n.Role(body=(n.ShuffleSync(var=v, src=5, src_lane=0),), warp=0, elected=True)]
    with pytest.raises(ValueError, match="shuffle_sync cannot be in single-thread"):
        make(body)


def test_accepts_shuffle_sync_in_warp_scope():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    make([n.Role(body=(n.ShuffleSync(var=v, src=5, src_lane=0),), warp=0)])


# ---- mbarrier ----------------------------------------------------------------


def test_rejects_mbarrier_wait_without_phase():
    mbar = n.MBar(kind=n.MBarKind.TMA)
    with pytest.raises(ValueError, match="mbarrier_wait phase is required"):
        make([n.MBarrierWait(mbar)])


def test_rejects_mbarrier_init_count_over_ptx_layout():
    mbar = n.MBar(kind=n.MBarKind.TMA)
    with pytest.raises(ValueError, match=r"count must be <= 2\^20 - 1"):
        make([n.MBarrierInit(mbar, count=1 << 20)])
    make([n.MBarrierInit(mbar, count=(1 << 20) - 1)])


# ---- clc_try_cancel ----------------------------------------------------------


def _clc_scheduler():
    return n.Scheduler(space=n.TaskSpace(grid=(4,), fields=("task",)), policy="custom")


def test_rejects_clc_try_cancel_mbar_kind_and_handle_size():
    sched = _clc_scheduler()
    bad_mbar = n.MBar(kind=n.MBarKind.THREAD)
    with pytest.raises(ValueError, match="clc_try_cancel mbar kind must be tma"):
        make([n.ClcTryCancel(sched, smem([4], dtype=n.DType.U32), bad_mbar)])
    ok_mbar = n.MBar(kind=n.MBarKind.TMA)
    with pytest.raises(ValueError, match="handle must be at least 16 bytes"):
        make([n.ClcTryCancel(sched, smem([2], dtype=n.DType.U32), ok_mbar)])
    # A well-formed one (inside the scheduler_impl the op belongs to) validates.
    make(
        [
            n.SchedulerImpl(
                scheduler=sched,
                body=(n.ClcTryCancel(sched, smem([4], dtype=n.DType.U32), ok_mbar, cta_group=1),),
            )
        ]
    )


def test_rejects_clc_try_cancel_cta_group_mismatch():
    # The cta_group field has no codegen emission site (TIRx's clc lowering
    # implies the multicast width), so a mismatch with the kernel-level engine
    # group is rejected at validate instead of running one semantics in sim and
    # another in the generated code.
    sched = _clc_scheduler()
    mbar = n.MBar(kind=n.MBarKind.TMA)
    handle = smem([4], dtype=n.DType.U32)

    def impl(cta_group):
        return n.SchedulerImpl(
            scheduler=sched, body=(n.ClcTryCancel(sched, handle, mbar, cta_group=cta_group),)
        )

    with pytest.raises(ValueError, match=r"clc_try_cancel cta_group=2 != kernel cta_group=1"):
        make([impl(2)])
    with pytest.raises(ValueError, match=r"clc_try_cancel cta_group=1 != kernel cta_group=2"):
        make([impl(1)], launch=(2,), cluster=(2,))
    # The matching value validates in both geometries.
    make([impl(1)])
    make([impl(2)], launch=(2,), cluster=(2,))


# ---- setmaxnreg --------------------------------------------------------------


def test_rejects_setmaxnreg_warpgroup_out_of_range():
    with pytest.raises(
        ValueError, match=r"setmaxnreg warpgroup must be in \[0, kernel num_warps / 4\)"
    ):
        make([n.SetMaxNReg(1, 56)], num_warps=4)
    make([n.SetMaxNReg(0, 56)], num_warps=4)


# ---- tma_load ----------------------------------------------------------------


def _tma_load(hint, **overrides):
    kw = dict(
        dst=smem([128, 64])[:, :],
        src=gmem([1024, 1024]),
        mbar=n.MBar(kind=n.MBarKind.TMA),
        coords=(0, 0),
        shape=[128, 64],
        cache_hint=hint,
    )
    kw.update(overrides)
    return n.TmaLoad(**kw)


def test_rejects_tma_load_bad_cache_hint():
    with pytest.raises(ValueError, match="cache_hint"):
        make([_tma_load("evict_last")])
    make([_tma_load("evict_normal")])
    make([_tma_load("evict_first")])


def test_tma_load_transfer_size_is_derived_not_stated():
    # The transfer size is DERIVED from the tile (numel(shape) x dtype) — the IR
    # carries no `bytes` field, so the sim's tx accounting and TIRx's box-extent
    # derivation read one fact in one place (a mismatch is unexpressible).
    with pytest.raises(TypeError):
        n.TmaLoad(
            dst=smem([128, 64])[:, :],
            src=gmem([1024, 1024]),
            mbar=n.MBar(kind=n.MBarKind.TMA),
            bytes=16384,  # the field is gone — passing it is a plain TypeError
            coords=(0, 0),
            shape=[128, 64],
        )
    # A zero-extent tile would derive a 0-byte transfer — rejected at build.
    with pytest.raises(ValueError, match="nonzero element count"):
        make([_tma_load(None, shape=[0, 64])])
    make([_tma_load(None)])


def test_rejects_tma_load_group2_multicast_shared_mbar():
    shared = n.MBarRef(n.MBar(kind=n.MBarKind.TMA), remote_coord=0)
    with pytest.raises(ValueError, match="not modeled"):
        make(
            [_tma_load(None, mbar=shared, multicast_cta_mask=0b11, cta_group=2)],
            launch=(2,),
            cluster=(2,),
        )


# ---- cp_async_bulk_s2cluster -------------------------------------------------


def test_rejects_s2cluster_without_remote_coord():
    with pytest.raises(ValueError, match="must target a peer CTA"):
        make(
            [
                n.CpAsyncBulkS2Cluster(
                    dst=smem([4], dtype=n.DType.F32)[:],
                    src=smem([4], dtype=n.DType.F32)[:],
                    mbar=n.MBar(kind=n.MBarKind.TMA),
                    bytes=16,
                )
            ]
        )


# ---- named_barrier vs wg_sync barrier ids ------------------------------------


def test_rejects_named_barrier_aliasing_wg_sync():
    for first, second in [
        (n.WgSync(barrier_id=3), n.NamedBarrier(barrier_id=3, num_warps=8)),
        (n.NamedBarrier(barrier_id=3, num_warps=8), n.WgSync(barrier_id=3)),
    ]:
        with pytest.raises(ValueError, match="cannot alias"):
            make([n.Role(body=(first, second), warpgroup=0)], num_warps=8)


def test_accepts_distinct_named_and_wg_barrier_ids():
    body = (n.NamedBarrier(barrier_id=1, num_warps=8), n.WgSync(barrier_id=2))
    make([n.Role(body=body, warpgroup=0)], num_warps=8)


def test_accepts_cross_role_named_and_wg_barrier_id_reuse():
    # Canon's flash_bwd pattern: wg0+wg1 rendezvous on named_barrier(1) while
    # wg2 keeps its private wg_sync(1) — cross-role id reuse is deliberate.
    make(
        [
            n.Role(body=(n.NamedBarrier(barrier_id=1, num_warps=8),), warpgroup=0),
            n.Role(body=(n.NamedBarrier(barrier_id=1, num_warps=8),), warpgroup=1),
            n.Role(body=(n.WgSync(barrier_id=1),), warpgroup=2),
        ],
        num_warps=12,
    )


# ---- tma ------------------------------------------------------------------


def tma_load(dst, src, mbar):
    return n.TmaLoad(dst=dst, src=src, mbar=mbar, coords=(0, 0), shape=[128, 64])


def test_rejects_tma_dst_not_smem():
    mbar = n.MBar(kind=n.MBarKind.TMA)
    with pytest.raises(ValueError, match="dst must be SMEM"):
        make([tma_load(gmem([128, 64])[:, :], gmem([1024, 1024]), mbar)])


def test_rejects_tma_src_not_gmem():
    mbar = n.MBar(kind=n.MBarKind.TMA)
    with pytest.raises(ValueError, match="src must be GMEM"):
        make([tma_load(smem([128, 64])[:, :], smem([1024, 1024]), mbar)])


def test_rejects_tma_dtype_mismatch():
    mbar = n.MBar(kind=n.MBarKind.TMA)
    with pytest.raises(ValueError, match="dtype must match"):
        make([tma_load(smem([128, 64])[:, :], gmem([1024, 1024], dtype=n.DType.F32), mbar)])


def test_rejects_tma_mbar_kind():
    mbar = n.MBar(kind=n.MBarKind.TCGEN05)
    with pytest.raises(ValueError, match="mbar kind must be tma"):
        make([tma_load(smem([128, 64])[:, :], gmem([1024, 1024]), mbar)])


# ---- reg ops ---------------------------------------------------------------


def test_rejects_reg_add_dst_not_reg():
    s = smem([16, 16])[:, :]
    with pytest.raises(ValueError, match="dst must be REG"):
        make([n.RegAdd(dst=s, lhs=reg([16, 16])[:, :], rhs=reg([16, 16])[:, :])])


def test_rejects_reg_cvt_src_not_f32():
    dst = reg([16, 16], dtype=n.DType.F16)[:, :]
    src = reg([16, 16], dtype=n.DType.F16)[:, :]
    with pytest.raises(ValueError, match="src dtype must be f32"):
        make([n.RegCvt(dst=dst, src=src)])


def test_rejects_reg_cvt_dst_dtype():
    dst = reg([16, 16], dtype=n.DType.F32)[:, :]
    src = reg([16, 16], dtype=n.DType.F32)[:, :]
    with pytest.raises(ValueError, match="dst dtype must be f16 or bf16"):
        make([n.RegCvt(dst=dst, src=src)])


def test_reg_softmax_rescale_accepts_f32_scale_threshold_with_f16_rows():
    row_max = reg([1], dtype=n.DType.F16)[:]
    row_scale = reg([1], dtype=n.DType.F16)[:]
    old = reg([1], dtype=n.DType.F16)[:]
    new = reg([1], dtype=n.DType.F16)[:]
    scale_log2 = reg([1], dtype=n.DType.F32)[:]
    threshold = reg([1], dtype=n.DType.F32)[:]

    make(
        [
            n.RegSoftmaxRescale(
                row_max=row_max,
                row_scale=row_scale,
                row_max_old=old,
                row_max_new=new,
                scale_log2=scale_log2,
                threshold=threshold,
            )
        ]
    )


def test_rejects_reg_softmax_rescale_non_f32_scale_threshold():
    row_max = reg([1], dtype=n.DType.F16)[:]
    row_scale = reg([1], dtype=n.DType.F16)[:]
    old = reg([1], dtype=n.DType.F16)[:]
    new = reg([1], dtype=n.DType.F16)[:]
    bad_scale_log2 = reg([1], dtype=n.DType.F16)[:]
    bad_threshold = reg([1], dtype=n.DType.F16)[:]

    with pytest.raises(ValueError, match="scale_log2 dtype must be F32"):
        make(
            [
                n.RegSoftmaxRescale(
                    row_max=row_max,
                    row_scale=row_scale,
                    row_max_old=old,
                    row_max_new=new,
                    scale_log2=bad_scale_log2,
                    threshold=reg([1], dtype=n.DType.F32)[:],
                )
            ]
        )
    with pytest.raises(ValueError, match="threshold dtype must be F32"):
        make(
            [
                n.RegSoftmaxRescale(
                    row_max=row_max,
                    row_scale=row_scale,
                    row_max_old=old,
                    row_max_new=new,
                    scale_log2=reg([1], dtype=n.DType.F32)[:],
                    threshold=bad_threshold,
                )
            ]
        )


# ---- mbarrier + tmem alloc -------------------------------------------------


def test_rejects_mbarrier_init_zero_count():
    mbar = n.MBar(kind=n.MBarKind.TMA)
    with pytest.raises(ValueError, match="must be a positive integer"):
        make([n.MBarrierInit(mbar, count=0)])


def test_rejects_tmem_alloc_bad_ncols():
    with pytest.raises(ValueError, match=r"power-of-two integer in \[32, 512\]"):
        make([n.TmemAlloc(0, 33)])


# ---- role / scope ----------------------------------------------------------


def test_rejects_role_both_warp_and_warpgroup():
    with pytest.raises(ValueError, match="cannot set both"):
        make([n.Role(body=(), warp=0, warpgroup=0)])


def test_rejects_role_maxnreg_without_warpgroup():
    with pytest.raises(ValueError, match="maxnreg requires"):
        make([n.Role(body=(), warp=0, maxnreg=64)])


def test_rejects_role_warp_out_of_range():
    with pytest.raises(ValueError, match=r"warp must be in \[0, kernel num_warps\)"):
        make([n.Role(body=(), warp=10)])


def test_rejects_wg_sync_bad_barrier_id():
    with pytest.raises(ValueError, match=r"barrier_id must be an integer in \[1, 15\]"):
        make([n.WgSync(barrier_id=99)])


def test_rejects_cta_sync_in_warp_scope():
    with pytest.raises(ValueError, match="cta_sync must be in CTA scope"):
        make([n.KernelInit(body=(n.CtaSync(),), warp=0)])


def test_rejects_cta_sync_inside_role():
    with pytest.raises(ValueError, match="cta_sync cannot be used inside role"):
        make([n.Role(body=(n.CtaSync(),))])


def test_rejects_tmem_alloc_outside_warp_scope():
    with pytest.raises(ValueError, match="must be in warp scope"):
        make([n.TmemAlloc(0, 64)])  # at CTA scope


# ---- tcgen05 ld/st ---------------------------------------------------------


def test_accepts_non_32x32b_tcgen05_ld_st_atom():
    tm = tmem_op(0, 0, dtype=n.DType.U32)
    frag = reg([4], dtype=n.DType.U32)
    make(
        [
            n.Role(
                body=(
                    n.Tcgen05Ld(dst=frag[:], src=tm, shape="16x128b", num=2),
                    n.Tcgen05St(dst=tm, src=frag[:], shape="16x128b", num=2),
                ),
                warpgroup=0,
            )
        ],
        launch=(1,),
        cluster=(1,),
        tmem_cg=1,
    )


def test_rejects_invalid_tcgen05_ld_st_atom_num():
    tm = tmem_op(0, 0, dtype=n.DType.U32)
    frag = reg([128], dtype=n.DType.U32)
    with pytest.raises(ValueError, match="shape/num"):
        make([n.Role(body=(n.Tcgen05Ld(dst=frag[:], src=tm, shape="16x128b", num=128),))])


# ---- ldmatrix / stmatrix ---------------------------------------------------


def test_accepts_ldstmatrix_m8n8_b16_atoms():
    sm = smem([32, 8], dtype=n.DType.U16)
    frag = reg([4], dtype=n.DType.U32)
    make(
        [
            n.Role(
                body=(
                    n.LdMatrix(dst=frag[0:4], src=sm[0, 0:8], num=4, trans=True),
                    n.StMatrix(dst=sm[0, 0:8], src=frag[0:4], num=4, trans=True),
                ),
                warp=0,
            )
        ],
        launch=(1,),
        cluster=(1,),
    )


def test_rejects_ldstmatrix_bad_spaces_shapes_and_dtype():
    sm = smem([8, 8], dtype=n.DType.U16)
    frag = reg([4], dtype=n.DType.U32)
    with pytest.raises(ValueError, match="dst must be REG"):
        make([n.Role(body=(n.LdMatrix(dst=sm[0, 0:4], src=sm[0, 0:8], num=1),), warp=0)])
    with pytest.raises(ValueError, match="src slice must contain one row"):
        make([n.Role(body=(n.LdMatrix(dst=frag[0:1], src=sm[0, 0:4], num=1),), warp=0)])
    with pytest.raises(ValueError, match="src dtype must be i32/u32 words or a b16 fragment"):
        make([n.Role(body=(n.StMatrix(dst=sm[0, 0:8], src=reg([1], dtype=n.DType.F32)[:]),))])


# ---- var definedness -------------------------------------------------------


def test_rejects_scalar_store_to_undefined_var():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    with pytest.raises(ValueError, match="defined before use"):
        make([n.ScalarStore(var=v, value=5)])


def test_rejects_var_defined_twice():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    with pytest.raises(ValueError, match="defined more than once"):
        make([n.ScalarDef(var=v, initial=0), n.ScalarDef(var=v, initial=1)])


# ---- scalar let (single-assignment SSA binding) -----------------------------


def test_scalar_let_defines_and_feeds_later_uses():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    w = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    make([n.ScalarLet(var=v, value=1 + 2), n.ScalarDef(var=w, initial=v), n.CtaSync()])


def test_rejects_scalar_store_to_let_var():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    with pytest.raises(ValueError, match="single assignment"):
        make([n.ScalarLet(var=v, value=0), n.ScalarStore(var=v, value=5)])


# ---- cross-statement walks -------------------------------------------------


def test_rejects_inconsistent_cta_group():
    # two tmem allocs (in warp scope) with different cta_group
    body = (n.TmemAlloc(0, 64, cta_group=1), n.TmemAlloc(64, 64, cta_group=2))
    with pytest.raises(ValueError, match="cta_group must be consistent"):
        make([n.Role(body=body, warp=0)])


# ---- tmem lifecycle placement (top-level only) -------------------------------


def test_rejects_tmem_lifecycle_inside_loop_or_conditional():
    # A lifecycle op inside a loop/conditional body used to pass the one-pass
    # lifecycle walk (which visits a loop body once) and then double-allocate
    # on the second iteration at runtime. Placement is rejected outright until
    # a path-sensitive analysis exists. (The Role wrapper puts the ops in warp
    # scope so the placement rule — not the earlier scope rule — is exercised.)
    i = n.Var()
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    with pytest.raises(ValueError, match="not allowed inside a loop or conditional"):
        make(
            [
                n.Role(
                    body=(
                        n.ForLoop(
                            var=i, start=0, stop=2, step=1, body=(n.TmemAlloc(0, 128, cta_group=1),)
                        ),
                    ),
                    warp=0,
                ),
                n.KernelFinalize(body=(n.TmemDealloc(0, 128, cta_group=1),), warp=0),
            ]
        )
    with pytest.raises(ValueError, match="not allowed inside a loop or conditional"):
        make(
            [
                n.KernelInit(body=(n.TmemAlloc(0, 128, cta_group=1),), warp=0),
                n.Role(
                    body=(
                        n.ScalarDef(var=v, initial=1),
                        n.If(cond=v < 2, then_body=(n.TmemDealloc(0, 128, cta_group=1),)),
                    ),
                    warp=0,
                ),
            ]
        )
    with pytest.raises(ValueError, match="not allowed inside a loop or conditional"):
        make(
            [
                n.KernelInit(body=(n.TmemAlloc(0, 128, cta_group=1),), warp=0),
                n.Role(
                    body=(
                        n.ScalarDef(var=v, initial=1),
                        n.Loop(body=(n.TmemRelinquish(cta_group=1), n.BreakIf(v < 2))),
                    ),
                    warp=0,
                ),
            ]
        )
    # Top-level placement (KernelInit / KernelFinalize / Role body) is fine.
    make(
        [
            n.KernelInit(body=(n.TmemAlloc(0, 128, cta_group=1),), warp=0),
            n.KernelFinalize(
                body=(n.TmemRelinquish(cta_group=1), n.TmemDealloc(0, 128, cta_group=1)), warp=0
            ),
        ]
    )
    make(
        [
            n.KernelInit(body=(n.TmemAlloc(0, 128, cta_group=1),), warp=0),
            n.KernelFinalize(
                body=(n.TmemRelinquish(cta_group=1), n.TmemDealloc(0, 128, cta_group=1)), warp=0
            ),
            n.Role(body=(), warp=1),
        ]
    )


def test_rejects_tmem_lifecycle_inside_nested_role():
    # A Role nested inside another Role is NOT a top-level execute-once scope:
    # its cohort is a strict subset of the outer role's (here warp1 inside
    # warp0 — never executed), so the alloc slipped past build/codegen and only
    # the protocol check reported the missing allocation. Reject at validate.
    with pytest.raises(ValueError, match="not allowed inside a .* or a nested Role body"):
        make([n.Role(body=(n.Role(body=(n.TmemAlloc(0, 128, cta_group=1),), warp=1),), warp=0)])
    # ...also via the else-branch chain.
    with pytest.raises(ValueError, match="or a nested Role body"):
        make(
            [
                n.Role(
                    body=(),
                    else_body=(n.Role(body=(n.TmemAlloc(0, 128, cta_group=1),), warp=1),),
                    warp=0,
                )
            ]
        )
    # POSITIVE: the same alloc directly in a TOP-LEVEL Role body is fine.
    make(
        [
            n.Role(body=(n.TmemAlloc(0, 128, cta_group=1),), warp=0),
            n.KernelFinalize(body=(n.TmemDealloc(0, 128, cta_group=1),), warp=0),
        ]
    )


def test_rejects_if_branching_on_role_scope():
    cond = n.ScopeValue(kind="warp_id")
    with pytest.raises(ValueError, match="cannot branch on role scope"):
        make([n.If(cond=cond, then_body=())])


def test_rejects_loop_nonpositive_step():
    i = n.Var()
    with pytest.raises(ValueError, match="step must be positive"):
        make([n.ForLoop(var=i, start=0, stop=10, step=0, body=())])


# ---- #1: scalar_def tensor initial dtype checks ----------------------------


def test_rejects_scalar_def_nonscalar_tensor_dtype():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    init = gmem([1, 1], dtype=n.DType.F16)[:, :]  # f16 is not a scalar integer/bool
    with pytest.raises(ValueError, match="dtype must be scalar integer or bool"):
        make([n.ScalarDef(var=v, initial=init)])


def test_rejects_scalar_def_var_dtype_mismatch():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.U32)
    init = gmem([1, 1], dtype=n.DType.I32)[:, :]  # decodes to i32, but var is u32
    with pytest.raises(ValueError, match="var dtype must match"):
        make([n.ScalarDef(var=v, initial=init)])


# ---- #2: mbar stages / arrive_count (eager, at MBar construction) ----------


def test_rejects_mbar_zero_stages():
    with pytest.raises(ValueError, match="stages must be a positive integer"):
        n.MBar(kind=n.MBarKind.TMA, stages=0)


def test_rejects_mbar_zero_arrive_count():
    with pytest.raises(ValueError, match="arrive_count must be a positive integer"):
        n.MBar(kind=n.MBarKind.TMA, arrive_count=0)
