"""B200 end-to-end coverage for physical tcgen05 MMA/CP Nymph IR.

Every case starts from Nymph IR, emits a real Python source file for the
TVMScript parser, compiles it through ``tir_pipeline="tirx"``, checks the
resulting CUDA source, and launches it on B200.  The cases intentionally mirror
the shapes and synchronization used by the existing Nymph interpreter tests and
TVM tcgen05 hardware tests:

* dense SMEM-A MMA for cta_group 1 and 2;
* TMEM-A Flat (Layout D) and BankBatched (Layout E);
* F8/E8M0 and packed-F4/E4M3 block-scaled MMA; and
* every legal tcgen05.cp shape/multicast pairing.

"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tvm = pytest.importorskip("tvm")
nr = pytest.importorskip("nymph_rs")


def _is_b200_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


pytestmark = pytest.mark.gpu


def _require_b200() -> None:
    if not _is_b200_available():
        pytest.skip("compiled and structurally checked; launch needs an NVIDIA B200")


def _align8(value: int) -> int:
    return (value + 7) // 8 * 8


def _smem_tile(builder, tensor, rows: int, cols: int):
    return builder.smem_tile(
        tensor, prefix_indices=(), row_offset=0, col_offset=0, rows=rows, cols=cols
    )


def _compile(kernel, tmp_path: Path, tag: str):
    """Exercise Nymph codegen, file-backed TVMScript parse, and TIRx compile."""
    source = nr.kernel_to_tirx_source(kernel)
    module_path = tmp_path / f"{tag}.py"
    module_path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"nymph_tcgen05_{tag}", module_path)
    emitted = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(emitted)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        compiled = tvm.compile(
            tvm.IRModule({"main": emitted.main}), target=target, tir_pipeline="tirx"
        )
    cuda_source = compiled.mod.imports[0].inspect_source()
    return source, cuda_source, compiled


def _f16_smem(builder, shape, byte_offset):
    return builder.tensor(
        space=nr.MemorySpace.SMEM,
        dtype=nr.DType.F16,
        shape=shape,
        layout=nr.SmemSwizzleLayout(nr.Swizzle.B32),
        byte_offset=byte_offset,
    )


def _build_dense_smem_a(cta_group: int):
    """The bootstrap cluster algorithm reduced to one 16-wide K instruction."""
    m_per_cta, n_per_cta, k_dim = 128, 16, 16
    m_dim = m_per_cta * cta_group
    n_dim = n_per_cta * cta_group
    alloc_cols = max(32, n_dim)
    a_bytes = m_per_cta * k_dim * 2
    b_bytes = n_per_cta * k_dim * 2
    metadata = _align8(a_bytes + b_bytes)
    smem_size = metadata + 2 * 8 + 4
    builder = nr.IRBuilder(
        f"tcgen05_dense_smem_cg{cta_group}",
        num_warps=8,
        smem_size_bytes=smem_size,
        launch_shape=(cta_group,),
        cluster_shape=(cta_group,),
    )
    a_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.F16, shape=(m_dim, k_dim))
    b_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.F16, shape=(n_dim, k_dim))
    out = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.F32, shape=(m_dim, n_dim))
    a_smem = _f16_smem(builder, (m_per_cta, k_dim), 0)
    b_smem = _f16_smem(builder, (n_per_cta, k_dim), a_bytes)
    accum = builder.tmem_tensor(0)
    fragment = builder.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.F32, shape=(n_dim,))
    loaded = builder.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata)
    mma_done = builder.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata + 8)
    loaded_leader = builder.mbar_ref(loaded, remote_coord=0) if cta_group == 2 else loaded
    cluster_rank = builder.ctaid_in_cluster()

    with builder.if_warp(0):
        builder.tmem_alloc(0, alloc_cols, addr_byte_offset=metadata + 16, cta_group=cta_group)
        with builder.if_elected():
            builder.mbarrier_init(loaded, count=1)
            builder.mbarrier_init(mma_done, count=1)
    builder.fence(kind=nr.FenceKind.MBARRIER_INIT)
    builder.cluster_sync()

    with builder.if_warp(4):
        with builder.if_elected():
            builder.tma_load(
                a_smem,
                a_gmem,
                mbar=loaded_leader,
                coords=(cluster_rank * m_per_cta, 0),
                shape=(m_per_cta, k_dim),
                cta_group=cta_group,
            )
            builder.tma_load(
                b_smem,
                b_gmem,
                mbar=loaded_leader,
                coords=(cluster_rank * n_per_cta, 0),
                shape=(n_per_cta, k_dim),
                cta_group=cta_group,
            )
            with builder.if_(cluster_rank.eq(0)):
                builder.mbarrier_arrive_expect_tx(
                    loaded_leader, bytes=cta_group * (a_bytes + b_bytes)
                )

    with builder.if_warp(5):
        with builder.if_(cluster_rank.eq(0)):
            builder.mbarrier_wait(loaded, phase=0)
            with builder.if_elected():
                builder.tcgen05_mma(
                    accum.at(0, 0),
                    builder.mma_a_smem(_smem_tile(builder, a_smem, m_per_cta, k_dim)),
                    _smem_tile(builder, b_smem, n_per_cta, k_dim),
                    mma_m=m_dim,
                    mma_n=n_dim,
                    format="f16",
                    block_scale=None,
                    accum=0,
                    trans_a=False,
                    trans_b=False,
                    ws=False,
                    cta_group=cta_group,
                )
                builder.tcgen05_commit(
                    mma_done,
                    cta_group=cta_group,
                    multicast_cta_mask=0b11 if cta_group == 2 else None,
                )

    with builder.if_warpgroup(0):
        builder.mbarrier_wait(mma_done, phase=0)
        builder.tcgen05_ld(fragment, accum.at(0, 0), num=n_dim)
        builder.tcgen05_wait_ld()
        builder.reg_store(out[cluster_rank * m_per_cta + builder.tid_in_wg(), 0:n_dim], fragment)

    builder.cluster_sync()
    with builder.if_warp(0):
        builder.tmem_relinquish(cta_group)
        builder.tmem_dealloc(0, alloc_cols, cta_group)
    return builder.build()


@pytest.mark.parametrize("cta_group", [1, 2], ids=["cg1", "cg2"])
def test_dense_smem_a_codegen_and_b200_values(cta_group, tmp_path):
    kernel = _build_dense_smem_a(cta_group)
    tirx_source, cuda_source, compiled = _compile(kernel, tmp_path, f"dense_cg{cta_group}")
    assert tirx_source.count("Tx.gemm_async(") == 1
    assert f"cta_group={cta_group}" in tirx_source
    assert f"mma_m={128 * cta_group}, mma_n={16 * cta_group}" in tirx_source
    assert f'mma_d0 = T.decl_buffer((128, {16 * cta_group}), "float32"' in tirx_source
    assert f"tcgen05.mma.cta_group::{cta_group}.kind::f16" in cuda_source

    _require_b200()
    m_dim, n_dim, k_dim = 128 * cta_group, 16 * cta_group, 16
    generator = torch.Generator(device="cuda").manual_seed(17 + cta_group)
    a = torch.randint(-2, 3, (m_dim, k_dim), generator=generator, device="cuda").to(torch.float16)
    b = torch.randint(-2, 3, (n_dim, k_dim), generator=generator, device="cuda").to(torch.float16)
    out = torch.empty((m_dim, n_dim), dtype=torch.float32, device="cuda")
    compiled(a, b, out)
    torch.cuda.synchronize()
    expected = a.float() @ b.float().T
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def _build_dense_feature(
    *,
    dtype,
    data_format: str,
    trans_a: bool = False,
    trans_b: bool = False,
    accumulate: bool = False,
):
    """One-CTA dense feature probe with literal physical operand shapes."""
    m_dim, n_dim = 128, 16
    k_dim = 32 if data_format == "f8_e4m3" else 16
    elem_bytes = 1 if data_format == "f8_e4m3" else 2
    a_shape = (k_dim, m_dim) if trans_a else (m_dim, k_dim)
    b_shape = (k_dim, n_dim) if trans_b else (n_dim, k_dim)
    a_bytes = int(np.prod(a_shape)) * elem_bytes
    b_bytes = int(np.prod(b_shape)) * elem_bytes
    metadata = _align8(a_bytes + b_bytes)
    builder = nr.IRBuilder(
        f"tcgen05_dense_{data_format}_ta{int(trans_a)}_tb{int(trans_b)}_acc{int(accumulate)}",
        num_warps=4,
        smem_size_bytes=metadata + 2 * 8 + 4,
        launch_shape=(1,),
        cluster_shape=(1,),
    )
    a_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=dtype, shape=a_shape)
    b_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=dtype, shape=b_shape)
    out = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.F32, shape=(m_dim, n_dim))
    swizzle = nr.SmemSwizzleLayout(nr.Swizzle.B32)
    a_smem = builder.tensor(
        space=nr.MemorySpace.SMEM, dtype=dtype, shape=a_shape, layout=swizzle, byte_offset=0
    )
    b_smem = builder.tensor(
        space=nr.MemorySpace.SMEM, dtype=dtype, shape=b_shape, layout=swizzle, byte_offset=a_bytes
    )
    accum = builder.tmem_tensor(0)
    fragment = builder.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.F32, shape=(n_dim,))
    loaded = builder.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata)
    mma_done = builder.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata + 8)

    with builder.if_warp(0):
        builder.tmem_alloc(0, 32, addr_byte_offset=metadata + 16)
        with builder.if_elected():
            builder.mbarrier_init(loaded, count=1)
            builder.mbarrier_init(mma_done, count=1)
    builder.fence(kind=nr.FenceKind.MBARRIER_INIT)
    builder.cta_sync()

    with builder.if_warpgroup(0):
        with builder.if_(builder.tid_in_wg().eq(0)):
            builder.tma_load(a_smem, a_gmem, mbar=loaded, coords=(0, 0), shape=a_shape)
            builder.tma_load(b_smem, b_gmem, mbar=loaded, coords=(0, 0), shape=b_shape)
            builder.mbarrier_arrive_expect_tx(loaded, bytes=a_bytes + b_bytes)
            builder.mbarrier_wait(loaded, phase=0)
            for accum_flag in [0, 1] if accumulate else [0]:
                builder.tcgen05_mma(
                    accum.at(0, 0),
                    builder.mma_a_smem(_smem_tile(builder, a_smem, a_shape[0], a_shape[1])),
                    _smem_tile(builder, b_smem, b_shape[0], b_shape[1]),
                    mma_m=m_dim,
                    mma_n=n_dim,
                    format=data_format,
                    block_scale=None,
                    accum=accum_flag,
                    trans_a=trans_a,
                    trans_b=trans_b,
                    ws=False,
                    cta_group=1,
                )
            builder.tcgen05_commit(mma_done)

        builder.mbarrier_wait(mma_done, phase=0)
        builder.tcgen05_ld(fragment, accum.at(0, 0), num=n_dim)
        builder.tcgen05_wait_ld()
        builder.reg_store(out[builder.tid_in_wg(), 0:n_dim], fragment)

    builder.cta_sync()
    with builder.if_warp(0):
        builder.tmem_relinquish()
        builder.tmem_dealloc(0, 32)
    return builder.build(), a_shape, b_shape


DENSE_FEATURE_CASES = [
    ("bf16", nr.DType.BF16, "bf16", False, False, False),
    ("fp8", nr.DType.F8E4M3, "f8_e4m3", False, False, False),
    ("trans_a", nr.DType.F16, "f16", True, False, False),
    ("trans_b", nr.DType.F16, "f16", False, True, False),
    ("accum", nr.DType.F16, "f16", False, False, True),
]


@pytest.mark.parametrize(
    "label,dtype,data_format,trans_a,trans_b,accumulate",
    DENSE_FEATURE_CASES,
    ids=[case[0] for case in DENSE_FEATURE_CASES],
)
def test_dense_feature_matrix_codegen_and_b200_values(
    label, dtype, data_format, trans_a, trans_b, accumulate, tmp_path
):
    kernel, a_shape, b_shape = _build_dense_feature(
        dtype=dtype,
        data_format=data_format,
        trans_a=trans_a,
        trans_b=trans_b,
        accumulate=accumulate,
    )
    tirx_source, cuda_source, compiled = _compile(kernel, tmp_path, f"dense_feature_{label}")
    issue_count = 2 if accumulate else 1
    assert tirx_source.count("Tx.gemm_async(") == issue_count
    assert f"transA={trans_a}" in tirx_source
    assert f"transB={trans_b}" in tirx_source
    assert tirx_source.count("accum=True") == int(accumulate)
    if data_format == "f8_e4m3":
        assert "SFA=" not in tirx_source and "SFB=" not in tirx_source
        assert "kind::f8f6f4" in cuda_source

    _require_b200()
    generator = torch.Generator(device="cuda").manual_seed(71 + len(label))
    a = torch.randint(-2, 3, a_shape, generator=generator, device="cuda")
    b = torch.randint(-2, 3, b_shape, generator=generator, device="cuda")
    if data_format == "f8_e4m3":
        a = a.float().to(torch.float8_e4m3fn)
        b = b.float().to(torch.float8_e4m3fn)
    elif dtype == nr.DType.BF16:
        a = a.to(torch.bfloat16)
        b = b.to(torch.bfloat16)
    else:
        a = a.to(torch.float16)
        b = b.to(torch.float16)
    out = torch.empty((128, 16), dtype=torch.float32, device="cuda")
    compiled(a, b, out)
    torch.cuda.synchronize()
    a_logical = a.float().T if trans_a else a.float()
    b_logical = b.float().T if trans_b else b.float()
    expected = a_logical @ b_logical.T
    if accumulate:
        expected *= 2
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def _build_tmem_a(form: str):
    is_batched = form == "bank_batched"
    m_dim, n_dim, k_dim = (64, 128, 16) if is_batched else (128, 16, 16)
    a_rows = 128
    b_bytes = n_dim * k_dim * 2
    metadata = _align8(b_bytes)
    smem_size = metadata + 2 * 8 + 4
    builder = nr.IRBuilder(
        f"tcgen05_tmem_a_{form}",
        num_warps=4,
        smem_size_bytes=smem_size,
        launch_shape=(1,),
        cluster_shape=(1,),
    )
    a_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.F16, shape=(a_rows, k_dim))
    b_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.F16, shape=(n_dim, k_dim))
    out = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.F32, shape=(m_dim, n_dim))
    b_smem = _f16_smem(builder, (n_dim, k_dim), 0)
    a_tmem = builder.tmem_tensor(0)
    accum = builder.tmem_tensor(16)
    a_fragment = builder.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.F16, shape=(k_dim,))
    fragment_width = n_dim // 2 if is_batched else n_dim
    fragment = builder.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.F32, shape=(fragment_width,))
    loaded = builder.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata)
    mma_done = builder.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata + 8)

    with builder.if_warp(0):
        builder.tmem_alloc(0, 128 if is_batched else 32, addr_byte_offset=metadata + 16)
        with builder.if_elected():
            builder.mbarrier_init(loaded, count=1)
            builder.mbarrier_init(mma_done, count=1)
    builder.fence(kind=nr.FenceKind.MBARRIER_INIT)
    builder.cta_sync()

    with builder.if_warpgroup(0):
        with builder.if_(builder.tid_in_wg().eq(0)):
            builder.tma_load(b_smem, b_gmem, mbar=loaded, coords=(0, 0), shape=(n_dim, k_dim))
            builder.mbarrier_arrive_expect_tx(loaded, bytes=b_bytes)
            builder.mbarrier_wait(loaded, phase=0)
        builder.reg_load(a_fragment, a_gmem[builder.tid_in_wg(), 0:k_dim])
        builder.tcgen05_st(a_tmem.at(0, 0), a_fragment, num=k_dim // 2)
        builder.tcgen05_wait_st()
        builder.wg_sync(barrier_id=1)
        with builder.if_(builder.tid_in_wg().eq(0)):
            builder.tcgen05_mma(
                accum.at(0, 0),
                builder.mma_a_tmem(a_tmem.at(0, 0), form=form),
                _smem_tile(builder, b_smem, n_dim, k_dim),
                mma_m=m_dim,
                mma_n=n_dim,
                format="f16",
                block_scale=None,
                accum=0,
                trans_a=False,
                trans_b=False,
                ws=is_batched,
                cta_group=1,
            )
            builder.tcgen05_commit(mma_done)

        builder.mbarrier_wait(mma_done, phase=0)
        builder.tcgen05_ld(fragment, accum.at(0, 0), num=fragment_width)
        builder.tcgen05_wait_ld()
        if is_batched:
            with builder.if_(builder.tid_in_wg() < 64):
                builder.reg_store(out[builder.tid_in_wg(), 0:fragment_width], fragment)
            with builder.if_(builder.tid_in_wg() >= 64):
                builder.reg_store(out[builder.tid_in_wg() - 64, fragment_width:n_dim], fragment)
        else:
            builder.reg_store(out[builder.tid_in_wg(), 0:n_dim], fragment)

    builder.cta_sync()
    with builder.if_warp(0):
        builder.tmem_relinquish()
        builder.tmem_dealloc(0, 128 if is_batched else 32)
    return builder.build()


@pytest.mark.parametrize("form", ["flat", "bank_batched"])
def test_tmem_a_codegen_and_b200_values(form, tmp_path):
    kernel = _build_tmem_a(form)
    tirx_source, cuda_source, compiled = _compile(kernel, tmp_path, f"tmem_a_{form}")
    assert tirx_source.count("Tx.gemm_async(") == 1
    assert tirx_source.count("T.ptx.tcgen05.st(") == 1
    if form == "flat":
        assert 'mma_a0 = T.decl_buffer((128, 16), "float16"' in tirx_source
        assert "ptx_tcgen05_mma_cta_1_kind_f16_TS(" in cuda_source
    else:
        assert 'mma_a0 = T.decl_buffer((2, 64, 16), "float16"' in tirx_source
        assert "ptx_tcgen05_mma_cta_1_kind_f16_TS_ws(" in cuda_source
    assert "tcgen05.st" in cuda_source

    _require_b200()
    generator = torch.Generator(device="cuda").manual_seed(31)
    a = torch.randint(-2, 3, (128, 16), generator=generator, device="cuda").to(torch.float16)
    n_dim = 128 if form == "bank_batched" else 16
    b = torch.randint(-2, 3, (n_dim, 16), generator=generator, device="cuda").to(torch.float16)
    m_dim = 64 if form == "bank_batched" else 128
    out = torch.empty((m_dim, n_dim), dtype=torch.float32, device="cuda")
    compiled(a, b, out)
    torch.cuda.synchronize()
    if form == "flat":
        expected = a.float() @ b.float().T
    else:
        expected = torch.cat(
            [a[:64].float() @ b[:64].float().T, a[64:].float() @ b[64:].float().T], dim=1
        )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def _build_block_scaled(
    data_format: str,
    *,
    sf_reuse: int = 1,
    sf_k_offset: int = 0,
    sf_col_offset: int = 0,
    accumulate: bool = False,
):
    m_dim, n_dim = 128, 32
    is_f8 = data_format == "f8_e4m3"
    storage_dtype = nr.DType.F8E4M3 if is_f8 else nr.DType.U8
    storage_k = 32 * sf_reuse if is_f8 else 32
    logical_k = storage_k if is_f8 else 64
    scale_format = "e8m0_fnu" if is_f8 else "e4m3_fn"
    sf_per_mma = 1 if is_f8 else 4
    a_bytes = m_dim * storage_k
    b_bytes = n_dim * storage_k
    scale_plane_bytes = 32 * 4 * 4
    scales_offset = a_bytes + b_bytes
    metadata = _align8(scales_offset + 2 * scale_plane_bytes)
    smem_size = metadata + 3 * 8 + 4
    builder = nr.IRBuilder(
        f"tcgen05_block_scaled_{data_format}",
        num_warps=4,
        smem_size_bytes=smem_size,
        launch_shape=(1,),
        cluster_shape=(1,),
    )
    a_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=storage_dtype, shape=(m_dim, storage_k))
    b_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=storage_dtype, shape=(n_dim, storage_k))
    scales_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.U32, shape=(2, 32, 4))
    out = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.F32, shape=(m_dim, n_dim))
    swizzle = nr.SmemSwizzleLayout(nr.Swizzle.B32)
    a_smem = builder.tensor(
        space=nr.MemorySpace.SMEM,
        dtype=storage_dtype,
        shape=(m_dim, storage_k),
        layout=swizzle,
        byte_offset=0,
    )
    b_smem = builder.tensor(
        space=nr.MemorySpace.SMEM,
        dtype=storage_dtype,
        shape=(n_dim, storage_k),
        layout=swizzle,
        byte_offset=a_bytes,
    )
    scales_smem = builder.tensor(
        space=nr.MemorySpace.SMEM, dtype=nr.DType.U32, shape=(2, 32, 4), byte_offset=scales_offset
    )
    accum = builder.tmem_tensor(0)
    sfa_tmem = builder.tmem_tensor(32)
    sfb_tmem = builder.tmem_tensor(40)
    scale_fragment = builder.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.U32, shape=(4,))
    fragment = builder.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.F32, shape=(n_dim,))
    loaded = builder.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata)
    copied = builder.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata + 8)
    mma_done = builder.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata + 16)

    with builder.if_warp(0):
        builder.tmem_alloc(0, 64, addr_byte_offset=metadata + 24)
        with builder.if_elected():
            builder.mbarrier_init(loaded, count=1)
            builder.mbarrier_init(copied, count=1)
            builder.mbarrier_init(mma_done, count=1)
    builder.fence(kind=nr.FenceKind.MBARRIER_INIT)
    builder.cta_sync()

    with builder.if_warpgroup(0):
        with builder.if_(builder.tid_in_wg().eq(0)):
            for dst, src, shape in [
                (a_smem, a_gmem, (m_dim, storage_k)),
                (b_smem, b_gmem, (n_dim, storage_k)),
            ]:
                builder.tma_load(dst, src, mbar=loaded, coords=(0, 0), shape=shape)
            builder.mbarrier_arrive_expect_tx(loaded, bytes=a_bytes + b_bytes)
            builder.mbarrier_wait(loaded, phase=0)
        with builder.if_(builder.tid_in_wg() < 32):
            builder.reg_load(scale_fragment, scales_gmem[0, builder.tid_in_wg(), 0:4])
            builder.reg_store(scales_smem[0, builder.tid_in_wg(), 0:4], scale_fragment)
            builder.reg_load(scale_fragment, scales_gmem[1, builder.tid_in_wg(), 0:4])
            builder.reg_store(scales_smem[1, builder.tid_in_wg(), 0:4], scale_fragment)
        builder.wg_sync(barrier_id=1)
        with builder.if_(builder.tid_in_wg().eq(0)):
            builder.tcgen05_cp(
                sfa_tmem.at(0, sf_col_offset),
                builder.smem_tile(
                    scales_smem, prefix_indices=(0,), row_offset=0, col_offset=0, rows=32, cols=4
                ),
                shape="32x128b",
                multicast="warp4",
                cta_group=1,
            )
            builder.tcgen05_cp(
                sfb_tmem.at(0, sf_col_offset),
                builder.smem_tile(
                    scales_smem, prefix_indices=(1,), row_offset=0, col_offset=0, rows=32, cols=4
                ),
                shape="32x128b",
                multicast="warp4",
                cta_group=1,
            )
            builder.tcgen05_commit(copied)
            builder.mbarrier_wait(copied, phase=0)
            for accum_flag in [0, 1] if accumulate else [0]:
                builder.tcgen05_mma(
                    accum.at(0, 0),
                    builder.mma_a_smem(_smem_tile(builder, a_smem, m_dim, storage_k)),
                    _smem_tile(builder, b_smem, n_dim, storage_k),
                    mma_m=m_dim,
                    mma_n=n_dim,
                    format=data_format,
                    block_scale=nr.BlockScaleSpec(
                        sfa_tmem.at(0, sf_col_offset),
                        sfb_tmem.at(0, sf_col_offset),
                        sf_k_offset,
                        sf_k_offset,
                        scale_format,
                        sf_per_mma,
                        sf_reuse,
                    ),
                    accum=accum_flag,
                    trans_a=False,
                    trans_b=False,
                    ws=False,
                    cta_group=1,
                )
            builder.tcgen05_commit(mma_done)

        builder.mbarrier_wait(mma_done, phase=0)
        builder.tcgen05_ld(fragment, accum.at(0, 0), num=n_dim)
        builder.tcgen05_wait_ld()
        builder.reg_store(out[builder.tid_in_wg(), 0:n_dim], fragment)

    builder.cta_sync()
    with builder.if_warp(0):
        builder.tmem_relinquish()
        builder.tmem_dealloc(0, 64)
    return builder.build(), logical_k


def _unpack_e2m1_even_high(packed: np.ndarray) -> np.ndarray:
    """Decode FP4 storage: logical even-K is the high nibble."""
    values = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=np.float32,
    )
    even = values[packed >> 4]
    odd = values[packed & 0xF]
    return np.stack((even, odd), axis=-1).reshape(packed.shape[0], -1)


def _pack_f4_scale_words(scale_codes: np.ndarray) -> np.ndarray:
    """Pack a 64x8 logical E4M3 SF tile into its 32x4 CP-ready words."""
    assert scale_codes.shape == (64, 8)
    words = np.zeros((32, 4), dtype=np.uint32)
    for lane in range(32):
        for k_group in range(2):
            for row_block in range(2):
                word = 0
                logical_row = row_block * 32 + lane
                for byte in range(4):
                    word |= int(scale_codes[logical_row, k_group * 4 + byte]) << (8 * byte)
                words[lane, k_group * 2 + row_block] = word
    return words


def _build_block_scaled_f4_cg2_m128():
    """Layout-B F4 GEMM: instruction M128, 64 M rows owned by each CTA."""
    cta_group = 2
    m_per_cta, n_per_cta, storage_k = 64, 32, 64
    m_dim, n_dim = m_per_cta * cta_group, n_per_cta * cta_group
    a_bytes = m_per_cta * storage_k
    b_bytes = n_per_cta * storage_k
    scales_offset = a_bytes + b_bytes
    scales_bytes = 2 * 32 * 4 * 4
    metadata = _align8(scales_offset + scales_bytes)
    smem_size = metadata + 2 * 8 + 4
    builder = nr.IRBuilder(
        "tcgen05_block_scaled_f4_cg2_m128",
        num_warps=4,
        smem_size_bytes=smem_size,
        launch_shape=(cta_group,),
        cluster_shape=(cta_group,),
    )
    a_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.U8, shape=(m_dim, storage_k))
    b_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.U8, shape=(n_dim, storage_k))
    scales_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.U32, shape=(4, 32, 4))
    out = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.F32, shape=(m_dim, n_dim))
    swizzle = nr.SmemSwizzleLayout(nr.Swizzle.B64)
    a_smem = builder.tensor(
        space=nr.MemorySpace.SMEM,
        dtype=nr.DType.U8,
        shape=(m_per_cta, storage_k),
        layout=swizzle,
        byte_offset=0,
    )
    b_smem = builder.tensor(
        space=nr.MemorySpace.SMEM,
        dtype=nr.DType.U8,
        shape=(n_per_cta, storage_k),
        layout=swizzle,
        byte_offset=a_bytes,
    )
    scales_smem = builder.tensor(
        space=nr.MemorySpace.SMEM, dtype=nr.DType.U32, shape=(2, 32, 4), byte_offset=scales_offset
    )
    accum = builder.tmem_tensor(0)
    sfa_tmem = builder.tmem_tensor(32)
    sfb_tmem = builder.tmem_tensor(36)
    scale_fragment = builder.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.U32, shape=(4,))
    out_fragment = builder.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.F32, shape=(n_dim // 2,))
    loaded = builder.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata)
    mma_done = builder.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata + 8)
    loaded_leader = builder.mbar_ref(loaded, remote_coord=0)
    cluster_rank = builder.ctaid_in_cluster()

    with builder.if_warp(0):
        builder.tmem_alloc(0, 64, addr_byte_offset=metadata + 16, cta_group=cta_group)
        with builder.if_elected():
            builder.mbarrier_init(loaded, count=1)
            builder.mbarrier_init(mma_done, count=1)
    builder.fence(kind=nr.FenceKind.MBARRIER_INIT)
    builder.cluster_sync()

    with builder.if_warp(2):
        with builder.if_elected():
            builder.tma_load(
                a_smem,
                a_gmem,
                mbar=loaded_leader,
                coords=(cluster_rank * m_per_cta, 0),
                shape=(m_per_cta, storage_k),
                cta_group=cta_group,
            )
            builder.tma_load(
                b_smem,
                b_gmem,
                mbar=loaded_leader,
                coords=(cluster_rank * n_per_cta, 0),
                shape=(n_per_cta, storage_k),
                cta_group=cta_group,
            )
            with builder.if_(cluster_rank.eq(0)):
                builder.mbarrier_arrive_expect_tx(
                    loaded_leader, bytes=cta_group * (a_bytes + b_bytes)
                )

    with builder.if_warpgroup(0):
        with builder.if_(builder.tid_in_wg() < 32):
            builder.reg_load(scale_fragment, scales_gmem[cluster_rank, builder.tid_in_wg(), 0:4])
            builder.reg_store(scales_smem[0, builder.tid_in_wg(), 0:4], scale_fragment)
            builder.reg_load(
                scale_fragment, scales_gmem[2 + cluster_rank, builder.tid_in_wg(), 0:4]
            )
            builder.reg_store(scales_smem[1, builder.tid_in_wg(), 0:4], scale_fragment)
    builder.cluster_sync()

    with builder.if_warp(0):
        with builder.if_(cluster_rank.eq(0)):
            with builder.if_elected():
                builder.mbarrier_wait(loaded, phase=0)
                builder.tcgen05_cp(
                    sfa_tmem.at(0, 0),
                    builder.smem_tile(
                        scales_smem,
                        prefix_indices=(0,),
                        row_offset=0,
                        col_offset=0,
                        rows=32,
                        cols=4,
                    ),
                    shape="32x128b",
                    multicast="warp4",
                    cta_group=cta_group,
                )
                builder.tcgen05_cp(
                    sfb_tmem.at(0, 0),
                    builder.smem_tile(
                        scales_smem,
                        prefix_indices=(1,),
                        row_offset=0,
                        col_offset=0,
                        rows=32,
                        cols=4,
                    ),
                    shape="32x128b",
                    multicast="warp4",
                    cta_group=cta_group,
                )
                builder.tcgen05_mma(
                    accum.at(0, 0),
                    builder.mma_a_smem(_smem_tile(builder, a_smem, m_per_cta, storage_k)),
                    _smem_tile(builder, b_smem, n_per_cta, storage_k),
                    mma_m=m_dim,
                    mma_n=n_dim,
                    format="f4_e2m1",
                    block_scale=nr.BlockScaleSpec(
                        sfa_tmem.at(0, 0), sfb_tmem.at(0, 0), 0, 0, "e4m3_fn", 4, 1
                    ),
                    accum=0,
                    trans_a=False,
                    trans_b=False,
                    ws=False,
                    cta_group=cta_group,
                )
                builder.tcgen05_commit(mma_done, cta_group=cta_group, multicast_cta_mask=0b11)

    with builder.if_warpgroup(0):
        builder.mbarrier_wait(mma_done, phase=0)
        builder.tcgen05_ld(out_fragment, accum.at(0, 0), num=n_dim // 2)
        builder.tcgen05_wait_ld()
        with builder.if_(builder.tid_in_wg() < 64):
            builder.reg_store(
                out[cluster_rank * m_per_cta + builder.tid_in_wg(), 0 : n_dim // 2], out_fragment
            )
        with builder.if_(builder.tid_in_wg() >= 64):
            builder.reg_store(
                out[cluster_rank * m_per_cta + builder.tid_in_wg() - 64, n_dim // 2 : n_dim],
                out_fragment,
            )

    builder.cluster_sync()
    with builder.if_warp(0):
        builder.tmem_relinquish(cta_group)
        builder.tmem_dealloc(0, 64, cta_group)
    return builder.build()


def _assert_source_nested_under(source: str, statement: str, guard: str) -> None:
    lines = source.splitlines()
    statement_index = next(i for i, line in enumerate(lines) if statement in line)
    guard_index = max(i for i, line in enumerate(lines[:statement_index]) if line.strip() == guard)
    guard_indent = len(lines[guard_index]) - len(lines[guard_index].lstrip())
    assert all(
        not line.strip() or len(line) - len(line.lstrip()) > guard_indent
        for line in lines[guard_index + 1 : statement_index + 1]
    )


def test_block_scaled_f4_cg2_m128_codegen_and_b200_values(tmp_path):
    kernel = _build_block_scaled_f4_cg2_m128()
    tirx_source, cuda_source, compiled = _compile(kernel, tmp_path, "block_scaled_f4_cg2_m128")
    assert tirx_source.count("Tx.gemm_async(") == 1
    assert tirx_source.count("Tx.copy_async(cp_dst") == 2
    assert "mma_m=128, mma_n=64" in tirx_source
    assert "cta_group=2" in tirx_source
    _assert_source_nested_under(tirx_source, "Tx.gemm_async(", "if cbx == 0:")
    assert "ptx_tcgen05_mma_block_scaled_cta_2" in cuda_source

    _require_b200()
    a_row, a_col = np.indices((128, 64))
    b_row, b_col = np.indices((64, 64))
    a_even = 1 + (a_row + 2 * a_col + a_row // 64) % 4
    a_odd = 1 + (3 * a_row + a_col + a_row // 64) % 4
    b_even = 1 + (2 * b_row + b_col + b_row // 32) % 4
    b_odd = 1 + (b_row + 3 * b_col + b_row // 32) % 4
    a_packed = ((a_even << 4) | a_odd).astype(np.uint8)
    b_packed = ((b_even << 4) | b_odd).astype(np.uint8)

    scale_codes = np.array([0x30, 0x38, 0x40], dtype=np.uint8)
    a_scale_row, a_scale_k = np.indices((128, 8))
    b_scale_row, b_scale_k = np.indices((64, 8))
    a_scale_codes = scale_codes[
        (a_scale_row + 2 * a_scale_k + a_scale_row // 64) % len(scale_codes)
    ]
    b_scale_codes = scale_codes[(2 * (b_scale_row % 32) + b_scale_k) % len(scale_codes)]
    scale_words = np.stack(
        [
            _pack_f4_scale_words(a_scale_codes[:64]),
            _pack_f4_scale_words(a_scale_codes[64:]),
            _pack_f4_scale_words(np.concatenate([b_scale_codes[:32]] * 2)),
            _pack_f4_scale_words(np.concatenate([b_scale_codes[32:]] * 2)),
        ]
    )

    scale_lut = np.zeros(256, dtype=np.float32)
    scale_lut[scale_codes] = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    a_values = _unpack_e2m1_even_high(a_packed)
    b_values = _unpack_e2m1_even_high(b_packed)
    a_dequant = a_values * np.repeat(scale_lut[a_scale_codes], 16, axis=1)
    b_dequant = b_values * np.repeat(scale_lut[b_scale_codes], 16, axis=1)
    expected = torch.from_numpy(a_dequant @ b_dequant.T).cuda()

    a = torch.from_numpy(a_packed).cuda()
    b = torch.from_numpy(b_packed).cuda()
    scales = torch.from_numpy(scale_words).cuda()
    out = torch.empty((128, 64), dtype=torch.float32, device="cuda")
    compiled(a, b, scales, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("data_format", ["f8_e4m3", "f4_e2m1"])
def test_block_scaled_codegen_and_b200_values(data_format, tmp_path):
    kernel, logical_k = _build_block_scaled(data_format)
    tirx_source, cuda_source, compiled = _compile(kernel, tmp_path, f"block_scaled_{data_format}")
    assert tirx_source.count("Tx.gemm_async(") == 1
    assert tirx_source.count('shape="32x128b"') == 2
    assert tirx_source.count("Tx.copy_async(cp_dst") == 2
    assert tirx_source.count("sf_tmem_layout(") == 2
    assert "ptx_tcgen05_mma_block_scaled_cta_1" in cuda_source

    _require_b200()
    if data_format == "f8_e4m3":
        a = torch.ones((128, 32), dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
        b = torch.ones((32, 32), dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
        sfa_word = int.from_bytes(bytes([128]) * 4, "little")
        sfb_word = int.from_bytes(bytes([129]) * 4, "little")
        expected_value = float(logical_k * 2 * 4)
        expected = None
    else:
        a_packed = np.full((128, 32), 0x21, dtype=np.uint8)
        b_packed = np.full((32, 32), 0x43, dtype=np.uint8)
        a = torch.from_numpy(a_packed).cuda()
        b = torch.from_numpy(b_packed).cuda()
        sfa_word = sfb_word = int.from_bytes(bytes([0x38]) * 4, "little")
        expected = torch.from_numpy(
            _unpack_e2m1_even_high(a_packed) @ _unpack_e2m1_even_high(b_packed).T
        ).cuda()
    scales = torch.from_numpy(
        np.stack(
            [
                np.full((32, 4), sfa_word, dtype=np.uint32),
                np.full((32, 4), sfb_word, dtype=np.uint32),
            ]
        )
    ).cuda()
    out = torch.empty((128, 32), dtype=torch.float32, device="cuda")
    compiled(a, b, scales, out)
    torch.cuda.synchronize()
    if expected is None:
        expected = torch.full_like(out, expected_value)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_block_scaled_fp8_reuse_offset_accum_codegen_and_b200_values(tmp_path):
    sf_reuse = 4
    sf_k_offset = 1
    sf_col_offset = 4
    kernel, logical_k = _build_block_scaled(
        "f8_e4m3",
        sf_reuse=sf_reuse,
        sf_k_offset=sf_k_offset,
        sf_col_offset=sf_col_offset,
        accumulate=True,
    )
    tirx_source, cuda_source, compiled = _compile(
        kernel, tmp_path, "block_scaled_f8_reuse_offset_accum"
    )
    assert tirx_source.count("Tx.gemm_async(") == 2
    assert tirx_source.count("accum=True") == 1
    assert tirx_source.count("sf_reuse=4") == 4
    assert tirx_source.count(", 0, 4))") == 2
    assert tirx_source.count(", 0, 17))") == 4
    assert "ptx_tcgen05_mma_block_scaled_cta_1" in cuda_source

    _require_b200()
    a = torch.ones((128, logical_k), dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
    b = torch.ones((32, logical_k), dtype=torch.float32).to(torch.float8_e4m3fn).cuda()
    sfa_word = int.from_bytes(bytes([127, 128, 130, 131]), "little")
    sfb_word = int.from_bytes(bytes([127, 129, 131, 132]), "little")
    scales = torch.from_numpy(
        np.stack(
            [
                np.full((32, 4), sfa_word, dtype=np.uint32),
                np.full((32, 4), sfb_word, dtype=np.uint32),
            ]
        )
    ).cuda()
    out = torch.empty((128, 32), dtype=torch.float32, device="cuda")
    compiled(a, b, scales, out)
    torch.cuda.synchronize()
    # sf_id=1 selects 2**1 for A and 2**2 for B. The second MMA
    # accumulates the same K=128 product into D.
    expected = torch.full_like(out, float(2 * logical_k * 2 * 4))
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


CP_CASES = [
    ("128x256b", "none", 128, 8),
    ("4x256b", "none", 4, 8),
    ("128x128b", "none", 128, 4),
    ("64x128b", "warp2_02_13", 64, 4),
    ("64x128b", "warp2_01_23", 64, 4),
    ("32x128b", "warp4", 32, 4),
]


def _build_cp(
    shape: str,
    multicast: str,
    rows: int,
    cols: int,
    *,
    cta_group: int = 1,
    dst_row: int = 0,
    dst_col: int = 0,
    read_cols: int = 8,
):
    # The B32 owning view is an explicit physical MMA-AB swizzle.  Its atom
    # contains eight rows even for the four-row tcgen05.cp shape; the
    # instruction still selects exactly the first four rows below.
    smem_rows = max(rows, 8) if shape.endswith("256b") else rows
    source_bytes = smem_rows * cols * 4
    metadata = _align8(source_bytes)
    smem_size = metadata + 8 + 4
    builder = nr.IRBuilder(
        f"tcgen05_cp_{shape}_{multicast}",
        num_warps=4,
        smem_size_bytes=smem_size,
        launch_shape=(cta_group,),
        cluster_shape=(cta_group,),
    )
    source_shape = (rows, cols) if cta_group == 1 else (cta_group, rows, cols)
    out_shape = (128, read_cols) if cta_group == 1 else (cta_group, 128, read_cols)
    source_gmem = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.U32, shape=source_shape)
    out = builder.arg(space=nr.MemorySpace.GMEM, dtype=nr.DType.U32, shape=out_shape)
    source_smem = builder.tensor(
        space=nr.MemorySpace.SMEM,
        dtype=nr.DType.U32,
        shape=(smem_rows, cols),
        layout=nr.SmemSwizzleLayout(nr.Swizzle.B32) if shape.endswith("256b") else None,
        byte_offset=0,
    )
    tmem = builder.tmem_tensor(0)
    source_fragment = builder.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.U32, shape=(cols,))
    zero = builder.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.U32, shape=(read_cols,))
    fragment = builder.tensor(space=nr.MemorySpace.REG, dtype=nr.DType.U32, shape=(read_cols,))
    copied = builder.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata)
    cluster_rank = builder.ctaid_in_cluster()

    with builder.if_warp(0):
        builder.tmem_alloc(0, 32, addr_byte_offset=metadata + 8, cta_group=cta_group)
        with builder.if_elected():
            builder.mbarrier_init(copied, count=1)
    builder.fence(kind=nr.FenceKind.MBARRIER_INIT)
    builder.cta_sync()

    with builder.if_warpgroup(0):
        builder.reg_fill(zero, 0)
        builder.tcgen05_st(tmem.at(0, 0), zero, num=read_cols)
        builder.tcgen05_wait_st()
        with builder.if_(builder.tid_in_wg() < rows):
            if cta_group == 1:
                source_row = source_gmem[builder.tid_in_wg(), 0:cols]
            else:
                source_row = source_gmem[cluster_rank, builder.tid_in_wg(), 0:cols]
            builder.reg_load(source_fragment, source_row)
            builder.reg_store(source_smem[builder.tid_in_wg(), 0:cols], source_fragment)
    builder.cluster_sync()

    with builder.if_warpgroup(0):
        with builder.if_(cluster_rank.eq(0)), builder.if_(builder.tid_in_wg().eq(0)):
            builder.tcgen05_cp(
                tmem.at(dst_row, dst_col),
                _smem_tile(builder, source_smem, rows, cols),
                shape=shape,
                multicast=multicast,
                cta_group=cta_group,
            )
            builder.tcgen05_commit(
                copied, cta_group=cta_group, multicast_cta_mask=0b11 if cta_group == 2 else None
            )
        builder.mbarrier_wait(copied, phase=0)
        builder.tcgen05_ld(fragment, tmem.at(0, 0), num=read_cols)
        builder.tcgen05_wait_ld()
        if cta_group == 1:
            out_row = out[builder.tid_in_wg(), 0:read_cols]
        else:
            out_row = out[cluster_rank, builder.tid_in_wg(), 0:read_cols]
        builder.reg_store(out_row, fragment)

    builder.cta_sync()
    with builder.if_warp(0):
        builder.tmem_relinquish(cta_group)
        builder.tmem_dealloc(0, 32, cta_group)
    return builder.build()


def _expected_cp(
    shape: str,
    multicast: str,
    source: np.ndarray,
    *,
    dst_row: int = 0,
    dst_col: int = 0,
    read_cols: int = 8,
) -> np.ndarray:
    rows, cols = source.shape
    expected = np.zeros((128, read_cols), dtype=np.uint32)
    for row in range(rows):
        for col in range(cols):
            if shape == "4x256b":
                targets = [(row // 4 + 32 * (row % 4), col)]
            elif multicast == "warp2_02_13":
                targets = [(row, col), (row + 64, col)]
            elif multicast == "warp2_01_23":
                lane = (row // 32) * 64 + row % 32
                targets = [(lane, col), (lane + 32, col)]
            elif multicast == "warp4":
                targets = [(row + 32 * replica, col) for replica in range(4)]
            else:
                targets = [(row, col)]
            for lane, target_col in targets:
                expected[dst_row + lane, dst_col + target_col] = source[row, col]
    return expected


@pytest.mark.parametrize(
    "shape,multicast,rows,cols",
    CP_CASES,
    ids=[f"{shape}-{multicast}" for shape, multicast, _, _ in CP_CASES],
)
@pytest.mark.parametrize("cta_group", [1, 2], ids=["cg1", "cg2"])
def test_cp_all_legal_shapes_codegen_and_b200_values(
    shape, multicast, rows, cols, cta_group, tmp_path
):
    kernel = _build_cp(shape, multicast, rows, cols, cta_group=cta_group)
    tirx_source, cuda_source, compiled = _compile(
        kernel, tmp_path, f"cp_{shape}_{multicast}_cg{cta_group}"
    )
    assert tirx_source.count(f'shape="{shape}"') == 1
    assert tirx_source.count("Tx.copy_async(cp_dst") == 1
    assert f'cp_dst0 = T.decl_buffer(({rows}, {cols}), "uint32"' in tirx_source
    if shape.endswith("256b"):
        assert (
            f'pool.alloc_tcgen05_mma_AB(({max(rows, 8)}, {cols}), "uint32", '
            "swizzle_mode=SwizzleMode.SWIZZLE_32B_ATOM, align=1024)" in tirx_source
        )
    else:
        assert f'pool.alloc(({rows}, {cols}), "uint32", scope="shared.dyn")' in tirx_source
    assert "mma_shared_layout" not in tirx_source
    assert f"tcgen05.cp.cta_group::{cta_group}.{shape}" in cuda_source
    ptx_multicast = {
        "none": "",
        "warp2_02_13": "warpx2::02_13",
        "warp2_01_23": "warpx2::01_23",
        "warp4": "warpx4",
    }[multicast]
    if ptx_multicast:
        assert ptx_multicast in cuda_source

    _require_b200()
    if cta_group == 1:
        source = np.fromfunction(
            lambda row, col: (row.astype(np.uint32) << 16) | col.astype(np.uint32),
            (rows, cols),
            dtype=int,
        ).astype(np.uint32)
        expected = _expected_cp(shape, multicast, source)
    else:
        source = np.fromfunction(
            lambda cta, row, col: (cta.astype(np.uint32) << 24)
            | (row.astype(np.uint32) << 16)
            | col.astype(np.uint32),
            (cta_group, rows, cols),
            dtype=int,
        ).astype(np.uint32)
        expected = np.stack(
            [_expected_cp(shape, multicast, source[cta]) for cta in range(cta_group)]
        )
    source_gpu = torch.from_numpy(source).cuda()
    out_shape = (128, 8) if cta_group == 1 else (cta_group, 128, 8)
    out = torch.empty(out_shape, dtype=torch.uint32, device="cuda")
    compiled(source_gpu, out)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


def test_cp_nonzero_row_col_and_multi_atom_codegen_and_b200_values(tmp_path):
    rows, cols = 4, 16
    dst_row, dst_col, read_cols = 3, 4, 32
    kernel = _build_cp(
        "4x256b", "none", rows, cols, dst_row=dst_row, dst_col=dst_col, read_cols=read_cols
    )
    tirx_source, cuda_source, compiled = _compile(kernel, tmp_path, "cp_4x256b_offset_multi")
    assert tirx_source.count("Tx.copy_async(cp_dst") == 1
    assert "tmem_view_layout(" in tirx_source
    assert ", 3, 4))" in tirx_source
    # One Nymph statement remains one Tx call; the TIRx dispatcher tiles its
    # two 256-bit column atoms into two physical PTX instructions.
    assert cuda_source.count("ptx_tcgen05_cp_cta_group_1_shape_4x256b") >= 3

    _require_b200()
    source = np.fromfunction(
        lambda row, col: (row.astype(np.uint32) << 16) | col.astype(np.uint32),
        (rows, cols),
        dtype=int,
    ).astype(np.uint32)
    out = torch.empty((128, read_cols), dtype=torch.uint32, device="cuda")
    compiled(torch.from_numpy(source).cuda(), out)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(
        out.cpu().numpy(),
        _expected_cp(
            "4x256b", "none", source, dst_row=dst_row, dst_col=dst_col, read_cols=read_cols
        ),
    )
