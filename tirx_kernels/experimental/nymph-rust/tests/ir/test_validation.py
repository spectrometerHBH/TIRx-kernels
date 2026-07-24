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


def tmem(shape, dtype=n.DType.F32):
    return n.Tensor(space=n.MemorySpace.TMEM, dtype=dtype, shape=shape)


def gmem(shape, dtype=n.DType.F16):
    return n.Tensor(space=n.MemorySpace.GMEM, dtype=dtype, shape=shape)


def reg(shape, dtype=n.DType.F32):
    return n.Tensor(space=n.MemorySpace.REG, dtype=dtype, shape=shape)


def make(body, *, num_warps=4, launch=(2,), cluster=(2,), args=()):
    """Build a kernel (validates on construction)."""
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
    """A valid cta_group=1 MMA's (dst, a, b) slices — m=128, n=256, k=16."""
    return tmem([128, 256])[:, :], smem([128, 16])[:, :], smem([256, 16])[:, :]


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
    with pytest.raises(ValueError, match="dst must be TMEM"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)])


def test_rejects_mma_operand_not_smem_or_tmem():
    dst, _, b = mma_operands()
    a = gmem([128, 16])[:, :]
    with pytest.raises(ValueError, match="operands must be SMEM or TMEM"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)])


def test_accepts_mma_tmem_operand():
    dst, _, b = mma_operands()
    a = tmem([128, 16], dtype=n.DType.F16)[:, :]
    make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)])


def test_rejects_mma_operand_dtype():
    dst, _, b = mma_operands()
    a = smem([128, 16], dtype=n.DType.F32)[:, :]
    with pytest.raises(ValueError, match="operand dtype must be f16, bf16, or f8e4m3"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)])


def test_rejects_mma_dst_dtype():
    _, a, b = mma_operands()
    dst = tmem([128, 256], dtype=n.DType.F16)[:, :]
    with pytest.raises(ValueError, match="dst dtype must be f32"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=16)])


def test_rejects_mma_bad_k():
    dst, a, b = mma_operands()
    with pytest.raises(ValueError, match="k must be 16"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=256, k=8)])


# ---- tcgen05_mma block-scaled (f8 UE8M0 + nvfp4 e4m3) ----------------------


def mma_f8_operands():
    """A valid block-scaled f8 (UE8M0) MMA operand set: cg1, m=128, n=32, k=32."""
    dst = tmem([128, 32])[:, :]
    a = smem([128, 32], dtype=n.DType.F8E4M3)[:, :]
    b = smem([32, 32], dtype=n.DType.F8E4M3)[:, :]
    sfa = tmem([128, 1], dtype=n.DType.U32)[:, :]
    sfb = tmem([128, 1], dtype=n.DType.U32)[:, :]
    return dst, a, b, sfa, sfb


def mma_fp4_operands():
    """A valid NVFP4 MMA operand set: cg2, m=256, n=256, k=64 (32 packed bytes)."""
    dst = tmem([128, 256])[:, :]
    a = smem([128, 32], dtype=n.DType.U8)[:, :]
    b = smem([128, 32], dtype=n.DType.U8)[:, :]
    sfa = tmem([128, 1], dtype=n.DType.U32)[:, :]
    sfb = tmem([128, 2], dtype=n.DType.U32)[:, :]
    return dst, a, b, sfa, sfb


def fp4_kwargs(**overrides):
    kw = dict(m=256, n=256, k=64, cta_group=2, sf_e4m3=True, sf_block=16, a_fp4=True, b_fp4=True)
    kw.update(overrides)
    return kw


def test_accepts_block_scaled_f8_and_nvfp4_mma():
    dst, a, b, sfa, sfb = mma_f8_operands()
    make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=32, k=32, sfa=sfa, sfb=sfb)])
    dst, a, b, sfa, sfb = mma_fp4_operands()
    make([n.Tcgen05Mma(dst=dst, a=a, b=b, sfa=sfa, sfb=sfb, **fp4_kwargs())])


def test_rejects_mma_fp4_k():
    dst, a, b, sfa, sfb = mma_fp4_operands()
    with pytest.raises(ValueError, match=r"fp4 \(mxf4\) k must be 64, 128, or 256"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, sfa=sfa, sfb=sfb, **fp4_kwargs(k=32))])


def test_rejects_mma_fp4_nblocks_beyond_packed_cell():
    # k=128 spans 8 blocks of 16 — more e4m3 bytes than one packed-u32 cell holds.
    dst, a, b, sfa, sfb = mma_fp4_operands()
    a = smem([128, 64], dtype=n.DType.U8)[:, :]
    b = smem([128, 64], dtype=n.DType.U8)[:, :]
    with pytest.raises(ValueError, match="at most 4 blocks per packed-u32 scale cell"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, sfa=sfa, sfb=sfb, **fp4_kwargs(k=128))])


def test_rejects_mma_m64_cg1_scale_mode():
    dst = tmem([64, 32])[:, :]
    a = smem([64, 32], dtype=n.DType.F8E4M3)[:, :]
    b = smem([32, 32], dtype=n.DType.F8E4M3)[:, :]
    sfa = tmem([128, 1], dtype=n.DType.U32)[:, :]
    sfb = tmem([128, 1], dtype=n.DType.U32)[:, :]
    with pytest.raises(ValueError, match="m=64 cta_group=1 does not support block-scaled"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=64, n=32, k=32, sfa=sfa, sfb=sfb)])


def test_rejects_mma_fp4_transposed():
    dst, a, b, sfa, sfb = mma_fp4_operands()
    with pytest.raises(ValueError, match="does not support trans_a/trans_b"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, sfa=sfa, sfb=sfb, **fp4_kwargs(trans_b=True))])


def test_rejects_mma_fp4_shape():
    dst, _, b, sfa, sfb = mma_fp4_operands()
    dst = tmem([64, 32])[:, :]
    a = smem([64, 32], dtype=n.DType.U8)[:, :]
    b = smem([32, 32], dtype=n.DType.U8)[:, :]
    with pytest.raises(ValueError, match="fp4 requires"):
        make(
            [
                n.Tcgen05Mma(
                    dst=dst, a=a, b=b, sfa=sfa, sfb=sfb, **fp4_kwargs(m=64, n=32, cta_group=1)
                )
            ]
        )


def test_rejects_mma_sf_e4m3_without_fp4_operands():
    dst, a, b, sfa, sfb = mma_f8_operands()
    with pytest.raises(ValueError, match=r"sf_e4m3 \(NVFP4\) requires fp4 operands"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, m=128, n=32, k=32, sfa=sfa, sfb=sfb, sf_e4m3=True)])


def test_rejects_mma_sf_e4m3_nonzero_sf_byte():
    dst, a, b, sfa, sfb = mma_fp4_operands()
    with pytest.raises(ValueError, match="sf_byte must be 0 for sf_e4m3"):
        make([n.Tcgen05Mma(dst=dst, a=a, b=b, sfa=sfa, sfb=sfb, sf_byte=1, **fp4_kwargs())])


# ---- mma_sync (WarpMma) ----------------------------------------------------


def warp_mma_operands():
    """A valid m16n8k16 mma.sync fragment set: A/B u32 packed words, C/D f32."""
    d = reg([4])[:]
    a = reg([4], dtype=n.DType.U32)[:]
    b = reg([2], dtype=n.DType.U32)[:]
    c = reg([4])[:]
    return d, a, b, c


def test_accepts_warp_mma_fragments():
    d, a, b, c = warp_mma_operands()
    make([n.WarpMma(d=d, a=a, b=b, c=c, m=16, n=8, k=16, ab_dtype=n.DType.BF16)])


def test_rejects_warp_mma_shape():
    d, a, b, c = warp_mma_operands()
    with pytest.raises(ValueError, match="supports only m16n8k8 / m16n8k16"):
        make([n.WarpMma(d=d, a=a, b=b, c=c, m=16, n=16, k=16)])


def test_rejects_warp_mma_ab_dtype():
    d, a, b, c = warp_mma_operands()
    with pytest.raises(ValueError, match="ab_dtype must be bf16 or f16"):
        make([n.WarpMma(d=d, a=a, b=b, c=c, m=16, n=8, k=16, ab_dtype=n.DType.F32)])


def test_rejects_warp_mma_non_reg_operand():
    d, a, b, c = warp_mma_operands()
    a = smem([4], dtype=n.DType.U32)[:]
    with pytest.raises(ValueError, match="mma_sync a must be REG"):
        make([n.WarpMma(d=d, a=a, b=b, c=c, m=16, n=8, k=16)])


def test_rejects_warp_mma_fragment_length():
    d, a, b, c = warp_mma_operands()
    a = reg([2], dtype=n.DType.U32)[:]  # want m*k/64 = 4 words
    with pytest.raises(ValueError, match="fragment must hold 4 elements per lane"):
        make([n.WarpMma(d=d, a=a, b=b, c=c, m=16, n=8, k=16)])


# ---- tma ------------------------------------------------------------------


def tma_load(dst, src, mbar):
    return n.TmaLoad(dst=dst, src=src, mbar=mbar, bytes=16384, coords=(0, 0), shape=[128, 64])


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
        make([n.TmemAlloc(tmem([128, 256]), n_cols=33)])


# ---- thread-shape rules ----------------------------------------------------


def test_rejects_wg_sync_bad_barrier_id():
    with pytest.raises(ValueError, match=r"barrier_id must be an integer in \[1, 15\]"):
        make([n.WgSync(barrier_id=99)])


def test_rejects_cta_sync_in_partial_branch():
    # cta_sync reached by only warp 0 can never complete on hardware.
    with pytest.raises(ValueError, match="every thread of the CTA"):
        make([n.If(cond=n.ScopeValue(kind="warp_id").eq(0), then_body=(n.CtaSync(),))])
    with pytest.raises(ValueError, match="every thread of the CTA"):
        make(
            [n.If(cond=n.ScopeValue(kind="warpgroup_id").eq(0), then_body=(n.CtaSync(),))],
            num_warps=8,
        )


def test_accepts_cta_sync_in_full_cta_branch():
    # A bare full-CTA branch (old: banned "inside role") is exactly the shape
    # the per-warp model wants: reachability, not nesting, is what matters.
    make([n.If(cond=1, then_body=(n.CtaSync(),))])


def test_rejects_tmem_alloc_outside_single_warp():
    with pytest.raises(ValueError, match="exactly one full warp"):
        make([n.TmemAlloc(tmem([128, 256]), n_cols=64)])  # full-CTA branch


def test_rejects_wg_sync_not_covering_full_warpgroup():
    cond = n.ScopeValue(kind="warp_id").eq(0)
    with pytest.raises(ValueError, match="exactly one full warpgroup"):
        make([n.If(cond=cond, then_body=(n.WgSync(barrier_id=1),))], num_warps=8)


def test_accepts_wg_sync_in_full_warpgroup_branch():
    cond = n.ScopeValue(kind="warpgroup_id").eq(1)
    make([n.If(cond=cond, then_body=(n.WgSync(barrier_id=1),))], num_warps=8)


def _wg_branch(wg, bar):
    return n.If(
        cond=n.ScopeValue(kind="warpgroup_id").eq(wg), then_body=(n.WgSync(barrier_id=bar),)
    )


def test_accepts_wg_sync_reused_within_one_warpgroup():
    # One warpgroup may reuse its barrier_id across many wg_syncs.
    make([_wg_branch(0, 1), _wg_branch(0, 1), _wg_branch(0, 1)], num_warps=8)


def test_rejects_wg_sync_barrier_id_across_two_warpgroups():
    # Two warpgroups on the same physical barrier would collide.
    with pytest.raises(ValueError, match="more than one warpgroup"):
        make([_wg_branch(0, 1), _wg_branch(1, 1)], num_warps=8)


def test_rejects_reg_cond_rescale_warpgroup_scope():
    dst = reg([4], dtype=n.DType.F32)[:]
    with pytest.raises(ValueError, match="scope must be warp"):
        make([n.RegCondRescale(dst=dst, src=dst, scale=dst, threshold=dst, scope="warpgroup")])


def test_accepts_reg_cond_rescale_warp_scope():
    dst = reg([4], dtype=n.DType.F32)[:]
    make([n.RegCondRescale(dst=dst, src=dst, scale=dst, threshold=dst, scope="warp")])


def test_rejects_warp_sync_in_subwarp_branch():
    cond = n.ScopeValue(kind="lane_id").eq(0)
    with pytest.raises(ValueError, match="whole warps"):
        make([n.If(cond=cond, then_body=(n.WarpSync(),))])


def test_dynamic_branch_skips_static_sync_rules():
    # A runtime-valued predicate makes the thread set indeterminate; the
    # static shape rules stand down (runtime rendezvous owns the check).
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.U32)
    body = (
        n.ScalarDef(var=v, initial=0),
        n.If(cond=n.ScopeValue(kind="warp_id").eq(v), then_body=(n.WgSync(barrier_id=1),)),
    )
    make(body, num_warps=8)


# ---- tcgen05 ld/st ---------------------------------------------------------


def test_accepts_non_32x32b_tcgen05_ld_st_atom():
    tm = tmem([128, 32], dtype=n.DType.U32)
    frag = reg([4], dtype=n.DType.U32)
    make(
        [
            n.If(
                cond=n.ScopeValue(kind="warpgroup_id").eq(0),
                then_body=(
                    n.Tcgen05Ld(dst=frag[:], src=tm, shape="16x128b", num=2),
                    n.Tcgen05St(dst=tm, src=frag[:], shape="16x128b", num=2),
                ),
            )
        ],
        launch=(1,),
        cluster=(1,),
    )


def test_rejects_invalid_tcgen05_ld_st_atom_num():
    tm = tmem([128, 32], dtype=n.DType.U32)
    frag = reg([128], dtype=n.DType.U32)
    with pytest.raises(ValueError, match="shape/num"):
        make(
            [n.If(cond=1, then_body=(n.Tcgen05Ld(dst=frag[:], src=tm, shape="16x128b", num=128),))]
        )


# ---- ldmatrix / stmatrix ---------------------------------------------------


def test_accepts_ldstmatrix_m8n8_b16_atoms():
    sm = smem([32, 8], dtype=n.DType.U16)
    frag = reg([4], dtype=n.DType.U32)
    make(
        [
            n.If(
                cond=n.ScopeValue(kind="warp_id").eq(0),
                then_body=(
                    n.LdMatrix(dst=frag[0:4], src=sm[0, 0:8], num=4, trans=True),
                    n.StMatrix(dst=sm[0, 0:8], src=frag[0:4], num=4, trans=True),
                ),
            )
        ],
        launch=(1,),
        cluster=(1,),
    )


def test_rejects_ldstmatrix_bad_spaces_shapes_and_dtype():
    sm = smem([8, 8], dtype=n.DType.U16)
    frag = reg([4], dtype=n.DType.U32)
    with pytest.raises(ValueError, match="dst must be REG"):
        make(
            [
                n.If(
                    cond=n.ScopeValue(kind="warp_id").eq(0),
                    then_body=(n.LdMatrix(dst=sm[0, 0:4], src=sm[0, 0:8], num=1),),
                )
            ]
        )
    with pytest.raises(ValueError, match="src slice must contain one row"):
        make(
            [
                n.If(
                    cond=n.ScopeValue(kind="warp_id").eq(0),
                    then_body=(n.LdMatrix(dst=frag[0:1], src=sm[0, 0:4], num=1),),
                )
            ]
        )
    with pytest.raises(ValueError, match="src dtype must be i32/u32 words or a b16 fragment"):
        make(
            [
                n.If(
                    cond=1,
                    then_body=(n.StMatrix(dst=sm[0, 0:8], src=reg([1], dtype=n.DType.F32)[:]),),
                )
            ]
        )


# ---- var definedness -------------------------------------------------------


def test_rejects_scalar_store_to_undefined_var():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    with pytest.raises(ValueError, match="defined before use"):
        make([n.ScalarStore(var=v, value=5)])


def test_rejects_var_defined_twice():
    v = n.Var(binding=n.VarBinding.SCALAR, dtype=n.ScalarDType.I32)
    with pytest.raises(ValueError, match="defined more than once"):
        make([n.ScalarDef(var=v, initial=0), n.ScalarDef(var=v, initial=1)])


# ---- cross-statement walks -------------------------------------------------


def test_rejects_inconsistent_cta_group():
    # two tmem allocs (in warp scope) with different cta_group
    body = (
        n.TmemAlloc(tmem([128, 256]), n_cols=64, cta_group=1),
        n.TmemAlloc(tmem([128, 256]), n_cols=64, cta_group=2),
    )
    with pytest.raises(ValueError, match="cta_group must be consistent"):
        make([n.If(cond=n.ScopeValue(kind="warp_id").eq(0), then_body=body)])


def test_if_may_branch_on_warp_and_lane_scope():
    # Warp/lane dispatch via If IS the execution model: predicates over
    # warp_id/warpgroup_id/lane_id are the normal case, not an error.
    cond = n.ScopeValue(kind="warp_id").eq(0)
    make([n.If(cond=cond, then_body=())])


def test_setmaxnreg_requires_positive_multiple_of_8():
    make([n.If(cond=n.ScopeValue(kind="warpgroup_id").eq(0), then_body=(n.SetMaxNReg(nreg=232),))])
    with pytest.raises(ValueError, match="positive multiple of 8"):
        make([n.SetMaxNReg(nreg=100)])
    with pytest.raises(ValueError, match="positive multiple of 8"):
        make([n.SetMaxNReg(nreg=0)])


def test_setmaxnreg_requires_whole_warpgroups():
    cond = n.ScopeValue(kind="warp_id").eq(0)
    with pytest.raises(ValueError, match="whole warpgroup"):
        make([n.If(cond=cond, then_body=(n.SetMaxNReg(nreg=232),))], num_warps=8)


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
