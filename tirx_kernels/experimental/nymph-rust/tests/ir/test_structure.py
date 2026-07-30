"""Targeted structure tests: a constructor must store its fields where it says,
and the operator overloading must build the right expression tree.

This is the cheap middle ground (not the full dataclass-introspection port): a
few read-back checks on the non-trivial constructors, catching realistic risks
like a swapped field or a mis-nested operator. The builder *logic* itself is
unchanged Python and already covered upstream + by the GEMM integration test.
"""

import pytest

n = pytest.importorskip("nymph_rs")


def smem(shape, dtype=n.DType.F16):
    return n.Tensor(space=n.MemorySpace.SMEM, dtype=dtype, shape=shape, byte_offset=0)


def test_mma_constructor_roundtrips_fields():
    # The instruction carries only explicit physical operands and instruction
    # parameters. Logical M/N/K are resolved from these fields, never inferred
    # by the binding constructor.
    tmem = n.TmemTensor(32)
    dst = tmem.at(4, 8)
    a_tile = n.SmemTile(smem([128, 16]), (), 0, 0, 128, 16)
    b_tile = n.SmemTile(smem([256, 16]), (), 0, 0, 256, 16)
    a = n.MmaAOperand.smem(a_tile)
    mma = n.Tcgen05Mma(
        dst=dst,
        a=a,
        b=b_tile,
        mma_m=128,
        mma_n=64,
        format="f16",
        block_scale=None,
        accum=1,
        trans_a=False,
        trans_b=False,
        ws=False,
        cta_group=1,
    )
    assert mma.mma_m == 128
    assert mma.mma_n == 64
    assert mma.format == "f16"
    assert mma.accum == 1
    assert mma.cta_group == 1
    assert mma.dst.tensor.start_col == 32
    assert mma.dst.row == 4
    assert mma.dst.col == 8
    assert mma.a.kind == "smem"
    assert mma.a.tile.tensor.dtype == n.DType.F16
    assert mma.b.tensor.dtype == n.DType.F16
    assert mma.a.tile.tensor.shape == [128, 16]
    assert mma.b.tensor.shape == [256, 16]
    assert mma.b.rows == 256
    assert mma.b.cols == 16


def test_tmem_a_and_block_scale_roundtrip():
    data = n.TmemTensor(128)
    scales = n.TmemTensor(400)
    a = n.MmaAOperand.tmem(data.at(0, 4), "bank_batched")
    scale = n.BlockScaleSpec(
        sfa=scales.at(0, 0),
        sfb=scales.at(0, 8),
        sfa_k_offset=5,
        sfb_k_offset=9,
        scale_format="e4m3_fn",
        sf_per_mma=4,
        sf_reuse=1,
    )

    assert a.kind == "tmem"
    assert a.addr.tensor.start_col == 128
    assert a.form == "bank_batched"
    assert scale.sfa.tensor.start_col == 400
    assert scale.sfb.col == 8
    assert scale.sfa_k_offset == 5
    assert scale.sfb_k_offset == 9
    assert scale.scale_format == "e4m3_fn"
    assert scale.sf_per_mma == 4
    assert scale.sf_reuse == 1


def test_operator_builds_correct_expr_tree():
    # i*16+4 must nest as ADD( MUL(i,16), 4 ) — precedence + structure.
    i = n.Var()
    e = i * 16 + 4
    assert e.op == n.ScalarOp.ADD
    assert e.args[0].op == n.ScalarOp.MUL  # the (i*16) sub-tree is the first arg
    assert e.args[0].args[1] == 16
    assert e.args[1] == 4
    # reflected operator: 3 + i -> ADD(3, i)
    r = 3 + i
    assert r.op == n.ScalarOp.ADD
    assert r.args[0] == 3


def test_loop_constructor_roundtrips_fields():
    i = n.Var()
    loop = n.ForLoop(var=i, start=2, stop=20, step=4, body=())
    assert loop.start == 2
    assert loop.stop == 20
    assert loop.step == 4
    assert loop.var.id == i.id  # same var (identity by id)


def test_kernel_body_preserves_statements_in_order():
    i = n.Var()
    j = n.Var()
    body = (
        n.ForLoop(var=i, start=0, stop=5, step=1, body=()),
        n.ForLoop(var=j, start=0, stop=10, step=1, body=()),
    )
    k = n.Kernel(name="t", args=(), body=body, num_warps=4, launch_shape=[2], cluster_shape=[2])
    assert len(k.body) == 2
    assert k.body[0].stop == 5
    assert k.body[1].stop == 10


def test_nested_body_roundtrips():
    # an If wrapping a body — confirm the nesting is preserved.
    inner = n.CtaSync()
    branch = n.If(cond=n.ScopeValue(kind="warp_id").eq(0), then_body=(inner,))
    assert len(branch.body) == 1


def test_scheduler_abstraction_roundtrips_fields():
    space = n.TaskSpace(grid=(2, 3), fields=("m_idx", "n_idx"))
    sched = n.Scheduler(space=space, policy="grid_stride")
    task = n.Var(binding=n.VarBinding.TASK)
    stmt = n.ForEachTask(scheduler=sched, var=task, body=())

    assert sched.space.id == space.id
    assert sched.policy == "grid_stride"
    assert sched.scope == "cluster"
    assert stmt.scheduler.id == sched.id
    assert stmt.var.id == task.id
    assert stmt.body == []


def test_scheduler_impl_loop_break_roundtrips():
    space = n.TaskSpace(grid=(4,), fields=("task",))
    sched = n.Scheduler(space=space, policy="custom")
    task = n.Var(binding=n.VarBinding.TASK)
    next_stmt = n.SchedNext(scheduler=sched, var=task)
    break_stmt = n.BreakIf(task < 0)
    loop = n.Loop(body=(next_stmt, break_stmt))
    impl = n.SchedulerImpl(scheduler=sched, body=(loop,))

    assert next_stmt.scheduler.id == sched.id
    assert next_stmt.var.id == task.id
    assert break_stmt.cond.op == n.ScalarOp.LT
    assert len(loop.body) == 2
    assert impl.scheduler.id == sched.id
    assert len(impl.body) == 1


def test_store_scalar_roundtrips_destination():
    dst = n.Tensor(space=n.MemorySpace.SMEM, dtype=n.DType.I32, shape=[2], byte_offset=0)
    task = n.Var(binding=n.VarBinding.TASK)
    stmt = n.StoreScalar(dst=dst[0], value=task)

    assert stmt.dst.tensor.id == dst.id
    assert stmt.dst.offsets[0] == 0


def test_layout_is_only_valid_for_smem_swizzle():
    layout = n.SmemSwizzleLayout(n.Swizzle.B128)
    tensor = n.Tensor(
        space=n.MemorySpace.SMEM, dtype=n.DType.F16, shape=[128, 64], layout=layout, byte_offset=0
    )
    assert tensor.layout.swizzle == n.Swizzle.B128

    with pytest.raises(ValueError, match="only valid for SMEM"):
        n.Tensor(space=n.MemorySpace.GMEM, dtype=n.DType.F16, shape=[128, 64], layout=layout)

    assert not hasattr(n, "ScaleFactorLayout")
    assert not hasattr(n, "sf_smem_layout")
    assert not hasattr(n, "fp8_sf_packed_u32_layout")
