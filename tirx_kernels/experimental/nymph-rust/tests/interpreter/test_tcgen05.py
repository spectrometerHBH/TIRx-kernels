import numpy as np
import nymph_rs as nr
import pytest
from helpers import (
    builder,
    expect_runtime_error,
    gmem_arg,
    output,
    reg_tensor,
    run,
    smem_tensor,
    u32,
)

DP_B = 1 << 21


def _tmem(start_col=0):
    return nr.TmemTensor(start_col)


def _tile(b, tensor, rows, cols, *, row=0, col=0, prefix=()):
    return b.smem_tile(
        tensor, prefix_indices=prefix, row_offset=row, col_offset=col, rows=rows, cols=cols
    )


def _layout_eval(shape, stride, index):
    offset = 0
    for extent, step in zip(shape, stride, strict=True):
        offset += (index % extent) * step
        index //= extent
    return offset


def _decode(bit_address):
    return bit_address >> 21, (bit_address >> 5) & 0xFFFF, bit_address & 31


def _atom(shape, num):
    if shape == "32x32b" and num in [1, 2, 4, 8, 16, 32, 64, 128]:
        return {
            "val_shape": [32 * num, 32],
            "val_stride": [1, DP_B],
            "dst_thr_shape": [32],
            "dst_thr_stride": [32 * num],
            "dst_val_shape": [32 * num],
            "dst_val_stride": [1],
        }
    if shape == "16x32bx2" and num in [1, 2, 4, 8, 16, 32, 64, 128]:
        return {
            "val_shape": [64 * num, 16],
            "val_stride": [1, DP_B],
            "dst_thr_shape": [16, 2],
            "dst_thr_stride": [64 * num, 32 * num],
            "dst_val_shape": [32 * num],
            "dst_val_stride": [1],
        }
    if shape == "16x64b" and num in [1, 2, 4, 8, 16, 32, 64, 128]:
        return {
            "val_shape": [64 * num, 16],
            "val_stride": [1, DP_B],
            "dst_thr_shape": [2, 2, 8],
            "dst_thr_stride": [512 * num, 32, 64 * num],
            "dst_val_shape": [32, num],
            "dst_val_stride": [1, 64],
        }
    if shape == "16x128b" and num in [1, 2, 4, 8, 16, 32, 64]:
        return {
            "val_shape": [128 * num, 16],
            "val_stride": [1, DP_B],
            "dst_thr_shape": [4, 8],
            "dst_thr_stride": [32, 128 * num],
            "dst_val_shape": [32, 2, num],
            "dst_val_stride": [1, 1024 * num, 128],
        }
    if shape == "16x256b" and num in [1, 2, 4, 8, 16, 32]:
        return {
            "val_shape": [256 * num, 16],
            "val_stride": [1, DP_B],
            "dst_thr_shape": [4, 8],
            "dst_thr_stride": [64, 256 * num],
            "dst_val_shape": [64, 2, num],
            "dst_val_stride": [1, 2048 * num, 256],
        }
    raise ValueError((shape, num))


def register_count(shape, num):
    atom = _atom(shape, num)
    product = 1
    for extent in atom["dst_val_shape"]:
        product *= extent
    return product // 32


def datapath_index_arrays(shape, num):
    atom = _atom(shape, num)
    role = 1
    for extent in atom["dst_thr_shape"]:
        role *= extent
    regs = register_count(shape, num)
    lane_idx = np.zeros((role, regs), dtype=np.int64)
    col_idx = np.zeros((role, regs), dtype=np.int64)
    for thr in range(role):
        thr_off = _layout_eval(atom["dst_thr_shape"], atom["dst_thr_stride"], thr)
        for reg in range(regs):
            val_off = _layout_eval(atom["dst_val_shape"], atom["dst_val_stride"], reg * 32)
            lane, col, bit0 = _decode(
                _layout_eval(atom["val_shape"], atom["val_stride"], thr_off + val_off)
            )
            assert bit0 == 0
            lane_idx[thr, reg] = lane
            col_idx[thr, reg] = col
    return lane_idx, col_idx


def supported_non32_shapes():
    for shape in ["16x32bx2", "16x64b"]:
        for num in [1, 2, 4, 8, 16, 32, 64, 128]:
            yield shape, num
    for num in [1, 2, 4, 8, 16, 32, 64]:
        yield "16x128b", num
    for num in [1, 2, 4, 8, 16, 32]:
        yield "16x256b", num


def u32_sentinels(rows, cols):
    values = [((row << 16) | col) for row in range(rows) for col in range(cols)]
    return u32(values, shape=(rows, cols))


def _tcgen05_role_failure_kernel(op, *, row=0, col=0, elected=False):
    b = builder("tcgen05_datapath_failure")
    band = _tmem()
    dst = reg_tensor(b, dtype=nr.DType.F32, shape=(1,))
    # The (row, col) base rides runtime scalars so the out-of-range cases reach
    # the interpreter's fail-closed checks instead of the static TmemOperand
    # range validation at build time.
    row_v = b.scalar(initial=row, dtype=nr.ScalarDType.I32)
    col_v = b.scalar(initial=col, dtype=nr.ScalarDType.I32)
    with b.if_warp(0):
        b.tmem_alloc(0, 64, addr_byte_offset=0)

    def emit():
        if op == "ld":
            b.tcgen05_ld(dst, band.at(row_v, col_v))
        else:
            b.tcgen05_st(band.at(row_v, col_v), dst)

    if elected:
        # A single-lane mask: ld/st are warp-collective, so this is the
        # partial-warp shape the mask check rejects.
        with b.if_warp(0), b.if_elected():
            emit()
    else:
        with b.if_warp(0):
            emit()
    return b.build()


def test_tcgen05_ld_st_reject_partial_warp_and_tmem_bounds_fail_closed():
    with expect_runtime_error("tcgen05_ld_mask"):
        run(_tcgen05_role_failure_kernel("ld", elected=True))
    with expect_runtime_error("tcgen05_ld_out_of_range"):
        run(_tcgen05_role_failure_kernel("ld", row=128))
    with expect_runtime_error("tcgen05_st_mask"):
        run(_tcgen05_role_failure_kernel("st", elected=True))
    with expect_runtime_error("tcgen05_st_out_of_range"):
        run(_tcgen05_role_failure_kernel("st", col=512))


def _reg_row_cols(b, tensor, col, cols):
    return tensor[b.tid_in_wg(), col : col + cols]


def _tcgen05_datapath_kernel(shape, num, mode):
    reg_size = register_count(shape, num)
    b = builder(f"tcgen05_{mode}_datapath")
    tmem = _tmem()
    seed_g = gmem_arg(b, shape=(128, 256))
    source_g = gmem_arg(b, shape=(128, reg_size))
    out = gmem_arg(b, shape=(128, 256 if mode == "st" else reg_size))
    seed_reg = reg_tensor(b, shape=(128,))
    source_reg = reg_tensor(b, shape=(reg_size,))
    out_reg = reg_tensor(b, shape=(reg_size,))
    chunk_reg = reg_tensor(b, shape=(128,))

    with b.if_warp(0):
        b.tmem_alloc(0, 256, addr_byte_offset=0)

    with b.if_warpgroup(0):
        for col in [0, 128]:
            b.reg_load(seed_reg, _reg_row_cols(b, seed_g, col, 128))
            b.tcgen05_st(tmem.at(0, col), seed_reg, shape="32x32b", num=128)

        if mode == "st":
            b.reg_load(source_reg, _reg_row_cols(b, source_g, 0, reg_size))
            b.tcgen05_st(tmem.at(0, 0), source_reg, shape=shape, num=num)
            for col in [0, 128]:
                b.tcgen05_ld(chunk_reg, tmem.at(0, col), shape="32x32b", num=128)
                b.reg_store(_reg_row_cols(b, out, col, 128), chunk_reg)
        else:
            b.tcgen05_ld(out_reg, tmem.at(0, 0), shape=shape, num=num)
            b.reg_store(_reg_row_cols(b, out, 0, reg_size), out_reg)

    with b.if_warp(0):
        b.tmem_dealloc(0, 256)

    return b.build(), seed_g, source_g, out


def test_tcgen05_st_non32_shapes_scatter_to_modeled_physical_cells():
    for shape, num in supported_non32_shapes():
        kernel, seed_g, source_g, out = _tcgen05_datapath_kernel(shape, num, "st")
        reg_size = register_count(shape, num)
        outputs = run(
            kernel,
            {seed_g: np.zeros((128, 256), dtype=np.uint32), source_g: u32_sentinels(128, reg_size)},
        )
        dump = output(outputs, out)
        lane_idx, col_idx = datapath_index_arrays(shape, num)
        expected = np.zeros((128, 256), dtype=np.uint32)
        for tid in range(128):
            warp = tid // 32
            lane = tid % 32
            for reg in range(reg_size):
                phys_lane = 32 * (warp % 4) + lane_idx[lane, reg]
                phys_col = col_idx[lane, reg]
                expected[phys_lane, phys_col] = (tid << 16) | reg
        np.testing.assert_array_equal(dump, expected, err_msg=f"{shape}.x{num}")


def test_tcgen05_ld_non32_shapes_gather_from_modeled_physical_cells():
    for shape, num in supported_non32_shapes():
        kernel, seed_g, source_g, out = _tcgen05_datapath_kernel(shape, num, "ld")
        reg_size = register_count(shape, num)
        outputs = run(
            kernel,
            {seed_g: u32_sentinels(128, 256), source_g: np.zeros((128, reg_size), dtype=np.uint32)},
        )
        dump = output(outputs, out)
        lane_idx, col_idx = datapath_index_arrays(shape, num)
        expected = np.zeros((128, reg_size), dtype=np.uint32)
        for tid in range(128):
            warp = tid // 32
            lane = tid % 32
            for reg in range(reg_size):
                phys_lane = 32 * (warp % 4) + lane_idx[lane, reg]
                phys_col = col_idx[lane, reg]
                expected[tid, reg] = (phys_lane << 16) | phys_col
        np.testing.assert_array_equal(dump, expected, err_msg=f"{shape}.x{num}")


def _align8(value):
    return (value + 7) // 8 * 8


def _dense_smem_kernel(dtype, *, trans_a=False, trans_b=False, accumulate=False):
    m, n = 128, 16
    k = 32 if dtype == nr.DType.F8E4M3 else 16
    elem_bytes = 1 if dtype == nr.DType.F8E4M3 else 2
    a_shape = (k, m) if trans_a else (m, k)
    b_shape = (k, n) if trans_b else (n, k)
    a_bytes = int(np.prod(a_shape)) * elem_bytes
    b_bytes = int(np.prod(b_shape)) * elem_bytes
    metadata = _align8(a_bytes + b_bytes)
    b = builder("tcgen05_dense_smem", smem_size_bytes=metadata + 3 * 8 + 4)
    a_g = gmem_arg(b, dtype=dtype, shape=a_shape)
    b_g = gmem_arg(b, dtype=dtype, shape=b_shape)
    out = gmem_arg(b, dtype=nr.DType.F32, shape=(m, n))
    a_s = smem_tensor(b, dtype=dtype, shape=a_shape, byte_offset=0)
    b_s = smem_tensor(b, dtype=dtype, shape=b_shape, byte_offset=a_bytes)
    a_tile = _tile(b, a_s, *a_shape)
    b_tile = _tile(b, b_s, *b_shape)
    d = _tmem()
    frag = reg_tensor(b, dtype=nr.DType.F32, shape=(n,))
    ma = b.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata)
    mb = b.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata + 8)
    done = b.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata + 16)

    with b.if_warp(0):
        b.tmem_alloc(0, 32, addr_byte_offset=metadata + 24)
        with b.if_elected():
            b.mbarrier_init(ma, count=1)
            b.mbarrier_init(mb, count=1)
            b.mbarrier_init(done, count=1)
    b.cta_sync()

    with b.if_warpgroup(0):
        with b.if_(b.tid_in_wg().eq(0)):
            b.mbarrier_arrive_expect_tx(ma, bytes=a_bytes)
            b.tma_load(a_s, a_g, mbar=ma, coords=(0, 0), shape=a_shape)
            b.mbarrier_arrive_expect_tx(mb, bytes=b_bytes)
            b.tma_load(b_s, b_g, mbar=mb, coords=(0, 0), shape=b_shape)
            b.mbarrier_wait(ma, phase=0)
            b.mbarrier_wait(mb, phase=0)
            data_format = {nr.DType.F16: "f16", nr.DType.BF16: "bf16", nr.DType.F8E4M3: "f8_e4m3"}[
                dtype
            ]
            for accum_flag in [0, 1] if accumulate else [0]:
                b.tcgen05_mma(
                    d.at(0, 0),
                    b.mma_a_smem(a_tile),
                    b_tile,
                    mma_m=m,
                    mma_n=n,
                    format=data_format,
                    block_scale=None,
                    accum=accum_flag,
                    trans_a=trans_a,
                    trans_b=trans_b,
                    ws=False,
                    cta_group=1,
                )
            b.tcgen05_commit(done)
        b.mbarrier_wait(done, phase=0)
        b.tcgen05_ld(frag, d.at(0, 0), num=n)
        b.tcgen05_wait_ld()
        b.reg_store(out[b.tid_in_wg(), 0:n], frag)

    b.cta_sync()
    with b.if_warp(0):
        b.tmem_dealloc(0, 32)
    return b.build(), a_g, b_g, out


@pytest.mark.parametrize("dtype", [nr.DType.F16, nr.DType.BF16])
def test_tcgen05_dense_smem_a_uses_resolved_mnk_and_d_footprint(dtype):
    kernel, a_g, b_g, out = _dense_smem_kernel(dtype)
    a = np.ones((128, 16), dtype=np.float32)
    b_values = np.arange(1, 17, dtype=np.float32)[:, None] * np.ones((1, 16), np.float32)
    outputs = run(kernel, {a_g: a, b_g: b_values})
    expected = np.broadcast_to(16.0 * np.arange(1, 17, dtype=np.float32), (128, 16))
    np.testing.assert_array_equal(output(outputs, out), expected)


@pytest.mark.parametrize(
    "dtype,trans_a,trans_b,accumulate",
    [
        (nr.DType.F8E4M3, False, False, False),
        (nr.DType.F16, True, False, False),
        (nr.DType.F16, False, True, False),
        (nr.DType.F16, False, False, True),
    ],
)
def test_tcgen05_dense_fp8_transpose_and_accumulate(dtype, trans_a, trans_b, accumulate):
    kernel, a_g, b_g, out = _dense_smem_kernel(
        dtype, trans_a=trans_a, trans_b=trans_b, accumulate=accumulate
    )
    k = 32 if dtype == nr.DType.F8E4M3 else 16
    a_logical = np.fromfunction(lambda row, col: 1 + (row + col) % 3, (128, k)).astype(np.float32)
    b_logical = np.fromfunction(lambda row, col: 1 + (2 * row + col) % 3, (16, k)).astype(
        np.float32
    )
    a_physical = a_logical.T.copy() if trans_a else a_logical
    b_physical = b_logical.T.copy() if trans_b else b_logical
    dump = output(run(kernel, {a_g: a_physical, b_g: b_physical}), out)
    expected = a_logical @ b_logical.T
    if accumulate:
        expected *= 2
    np.testing.assert_array_equal(dump, expected)
    assert nr.check_protocol(kernel)["status"] == "Passed"


def _tmem_a_kernel(form):
    m, n, k = 64, 16, 16
    b_bytes = n * k * 2
    metadata = _align8(b_bytes)
    b = builder(f"tcgen05_tmem_a_{form}", smem_size_bytes=metadata + 2 * 8 + 4)
    a_g = gmem_arg(b, dtype=nr.DType.F16, shape=(128, k))
    b_g = gmem_arg(b, dtype=nr.DType.F16, shape=(n, k))
    out = gmem_arg(b, dtype=nr.DType.F32, shape=(128, n))
    b_s = smem_tensor(b, dtype=nr.DType.F16, shape=(n, k), byte_offset=0)
    b_tile = _tile(b, b_s, n, k)
    a_tmem = _tmem(0)
    d = _tmem(32)
    a_frag = reg_tensor(b, dtype=nr.DType.F16, shape=(k,))
    zero = reg_tensor(b, dtype=nr.DType.F32, shape=(n,))
    out_frag = reg_tensor(b, dtype=nr.DType.F32, shape=(n,))
    mb = b.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata)
    done = b.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata + 8)

    with b.if_warp(0):
        b.tmem_alloc(0, 64, addr_byte_offset=metadata + 16)
        with b.if_elected():
            b.mbarrier_init(mb, count=1)
            b.mbarrier_init(done, count=1)
    b.cta_sync()

    with b.if_warpgroup(0):
        b.reg_load(a_frag, a_g[b.tid_in_wg(), 0:k])
        b.tcgen05_st(a_tmem.at(0, 0), a_frag[0 : k // 2], num=k // 2)
        b.reg_fill(zero, 0.0)
        b.tcgen05_st(d.at(0, 0), zero, num=n)
        b.tcgen05_wait_st()
        b.wg_sync(barrier_id=1)
        with b.if_(b.tid_in_wg().eq(0)):
            b.mbarrier_arrive_expect_tx(mb, bytes=b_bytes)
            b.tma_load(b_s, b_g, mbar=mb, coords=(0, 0), shape=(n, k))
            b.mbarrier_wait(mb, phase=0)
            b.tcgen05_mma(
                d.at(0, 0),
                b.mma_a_tmem(a_tmem.at(0, 0), form=form),
                b_tile,
                mma_m=m,
                mma_n=n,
                format="f16",
                block_scale=None,
                accum=0,
                trans_a=False,
                trans_b=False,
                ws=form == "bank_batched",
                cta_group=1,
            )
            b.tcgen05_commit(done)
        b.mbarrier_wait(done, phase=0)
        b.tcgen05_ld(out_frag, d.at(0, 0), num=n)
        b.tcgen05_wait_ld()
        b.reg_store(out[b.tid_in_wg(), 0:n], out_frag)

    b.cta_sync()
    with b.if_warp(0):
        b.tmem_dealloc(0, 64)
    return b.build(), a_g, b_g, out


@pytest.mark.parametrize("form", ["flat", "bank_batched"])
def test_tcgen05_tmem_a_flat_and_bank_batched(form):
    kernel, a_g, b_g, out = _tmem_a_kernel(form)
    a = np.ones((128, 16), dtype=np.float32)
    a[64:, :] = 2.0
    b_values = np.ones((16, 16), dtype=np.float32)
    dump = output(run(kernel, {a_g: a, b_g: b_values}), out)
    if form == "flat":
        mapped = np.r_[0:16, 32:48, 64:80, 96:112]
        np.testing.assert_array_equal(dump[mapped], np.full((64, 16), 16.0, np.float32))
        holes = np.setdiff1d(np.arange(128), mapped)
        np.testing.assert_array_equal(dump[holes], np.zeros((64, 16), np.float32))
    else:
        np.testing.assert_array_equal(dump[:64, :8], np.full((64, 8), 16.0, np.float32))
        np.testing.assert_array_equal(dump[64:, :8], np.full((64, 8), 32.0, np.float32))
        np.testing.assert_array_equal(dump[:, 8:], np.zeros((128, 8), np.float32))


def _block_scaled_kernel(format, *, sf_reuse=1, sf_k_offset=0, sf_col_offset=0, accumulate=False):
    m, n = 128, 32
    if format == "f8_e4m3":
        storage_dtype = nr.DType.F8E4M3
        storage_k = logical_k = 32 * sf_reuse
        scale_format = "e8m0_fnu"
        scale_bytes_a = [127, 128, 130, 131]
        scale_bytes_b = [127, 129, 131, 132]
        sf_per_mma = 1
        scale_a = 2.0 ** (scale_bytes_a[sf_k_offset] - 127)
        scale_b = 2.0 ** (scale_bytes_b[sf_k_offset] - 127)
        expected = logical_k * scale_a * scale_b
    else:
        storage_dtype = nr.DType.U8
        storage_k, logical_k = 32, 64
        scale_format = "e4m3_fn"
        scale_bytes_a = scale_bytes_b = [0x38] * 4  # e4m3fn 1.0
        sf_per_mma = 4
        expected = float(logical_k)
    if accumulate:
        expected *= 2

    a_bytes = m * storage_k
    b_bytes = n * storage_k
    scale_bytes = 32 * 4 * 4
    sfa_offset = a_bytes + b_bytes
    sfb_offset = sfa_offset + scale_bytes
    metadata = _align8(sfb_offset + scale_bytes)
    b = builder(f"tcgen05_block_scaled_{format}", smem_size_bytes=metadata + 6 * 8 + 4)
    a_g = gmem_arg(b, dtype=storage_dtype, shape=(m, storage_k))
    b_g = gmem_arg(b, dtype=storage_dtype, shape=(n, storage_k))
    sfa_g = gmem_arg(b, dtype=nr.DType.U32, shape=(32, 4))
    sfb_g = gmem_arg(b, dtype=nr.DType.U32, shape=(32, 4))
    out = gmem_arg(b, dtype=nr.DType.F32, shape=(m, n))
    a_s = smem_tensor(b, dtype=storage_dtype, shape=(m, storage_k), byte_offset=0)
    b_s = smem_tensor(b, dtype=storage_dtype, shape=(n, storage_k), byte_offset=a_bytes)
    sfa_s = smem_tensor(b, dtype=nr.DType.U32, shape=(32, 4), byte_offset=sfa_offset)
    sfb_s = smem_tensor(b, dtype=nr.DType.U32, shape=(32, 4), byte_offset=sfb_offset)
    a_tile = _tile(b, a_s, m, storage_k)
    b_tile = _tile(b, b_s, n, storage_k)
    sfa_tile = _tile(b, sfa_s, 32, 4)
    sfb_tile = _tile(b, sfb_s, 32, 4)
    d = _tmem(0)
    sfa = _tmem(32)
    sfb = _tmem(40)
    frag = reg_tensor(b, dtype=nr.DType.F32, shape=(n,))
    mbars = [b.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata + 8 * i) for i in range(4)]
    cp_done = b.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata + 32)
    mma_done = b.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata + 40)

    with b.if_warp(0):
        b.tmem_alloc(0, 64, addr_byte_offset=metadata + 48)
        with b.if_elected():
            for mbar in [*mbars, cp_done, mma_done]:
                b.mbarrier_init(mbar, count=1)
    b.cta_sync()

    with b.if_warpgroup(0):
        with b.if_(b.tid_in_wg().eq(0)):
            for mbar, dst, src, shape, byte_count in [
                (mbars[0], a_s, a_g, (m, storage_k), a_bytes),
                (mbars[1], b_s, b_g, (n, storage_k), b_bytes),
                (mbars[2], sfa_s, sfa_g, (32, 4), scale_bytes),
                (mbars[3], sfb_s, sfb_g, (32, 4), scale_bytes),
            ]:
                b.mbarrier_arrive_expect_tx(mbar, bytes=byte_count)
                b.tma_load(dst, src, mbar=mbar, coords=(0, 0), shape=shape)
                b.mbarrier_wait(mbar, phase=0)
            b.tcgen05_cp(
                sfa.at(0, sf_col_offset), sfa_tile, shape="32x128b", multicast="warp4", cta_group=1
            )
            b.tcgen05_cp(
                sfb.at(0, sf_col_offset), sfb_tile, shape="32x128b", multicast="warp4", cta_group=1
            )
            b.tcgen05_commit(cp_done)
            b.mbarrier_wait(cp_done, phase=0)
            scales = nr.BlockScaleSpec(
                sfa.at(0, sf_col_offset),
                sfb.at(0, sf_col_offset),
                sf_k_offset,
                sf_k_offset,
                scale_format,
                sf_per_mma,
                sf_reuse,
            )
            for accum_flag in [0, 1] if accumulate else [0]:
                b.tcgen05_mma(
                    d.at(0, 0),
                    b.mma_a_smem(a_tile),
                    b_tile,
                    mma_m=m,
                    mma_n=n,
                    format=format,
                    block_scale=scales,
                    accum=accum_flag,
                    trans_a=False,
                    trans_b=False,
                    ws=False,
                    cta_group=1,
                )
            b.tcgen05_commit(mma_done)
        b.mbarrier_wait(mma_done, phase=0)
        b.tcgen05_ld(frag, d.at(0, 0), num=n)
        b.tcgen05_wait_ld()
        b.reg_store(out[b.tid_in_wg(), 0:n], frag)

    b.cta_sync()
    with b.if_warp(0):
        b.tmem_dealloc(0, 64)

    packed_a = int.from_bytes(bytes(scale_bytes_a), "little")
    packed_b = int.from_bytes(bytes(scale_bytes_b), "little")
    return b.build(), a_g, b_g, sfa_g, sfb_g, out, packed_a, packed_b, expected


@pytest.mark.parametrize("format", ["f8_e4m3", "f4_e2m1"])
def test_tcgen05_block_scale_f8_e8m0_and_f4_e4m3(format):
    kernel, a_g, b_g, sfa_g, sfb_g, out, packed_a, packed_b, expected = _block_scaled_kernel(format)
    if format == "f8_e4m3":
        a = np.ones((128, 32), dtype=np.float32)
        b_values = np.ones((32, 32), dtype=np.float32)
    else:
        a = np.full((128, 32), 0x22, dtype=np.uint8)
        b_values = np.full((32, 32), 0x22, dtype=np.uint8)
    scales_a = np.full((32, 4), packed_a, dtype=np.uint32)
    scales_b = np.full((32, 4), packed_b, dtype=np.uint32)
    dump = output(run(kernel, {a_g: a, b_g: b_values, sfa_g: scales_a, sfb_g: scales_b}), out)
    np.testing.assert_array_equal(dump, np.full((128, 32), expected, np.float32))


def test_tcgen05_block_scale_reuse_offset_and_accumulate():
    kernel, a_g, b_g, sfa_g, sfb_g, out, packed_a, packed_b, expected = _block_scaled_kernel(
        "f8_e4m3", sf_reuse=4, sf_k_offset=1, sf_col_offset=4, accumulate=True
    )
    a = np.ones((128, 128), dtype=np.float32)
    b_values = np.ones((32, 128), dtype=np.float32)
    scales_a = np.full((32, 4), packed_a, dtype=np.uint32)
    scales_b = np.full((32, 4), packed_b, dtype=np.uint32)
    dump = output(run(kernel, {a_g: a, b_g: b_values, sfa_g: scales_a, sfb_g: scales_b}), out)
    np.testing.assert_array_equal(dump, np.full((128, 32), expected, np.float32))
    assert nr.check_protocol(kernel)["status"] == "Passed"


def _fp4_cg2_m128_kernel(*, both_ctas_issue):
    m, n, k = 128, 32, 64
    per_cta_m, per_cta_n, packed_k = m // 2, n // 2, k // 2
    a_bytes = per_cta_m * packed_k
    b_bytes = per_cta_n * packed_k
    metadata = _align8(a_bytes + b_bytes)
    tmem_addr_offset = metadata + 16
    b = builder(
        f"tcgen05_fp4_cg2_m128_{'both' if both_ctas_issue else 'leader'}",
        smem_size_bytes=tmem_addr_offset + 4,
        launch_shape=(2,),
        cluster_shape=(2,),
    )
    a_g = gmem_arg(b, dtype=nr.DType.U8, shape=(2, per_cta_m, packed_k))
    b_g = gmem_arg(b, dtype=nr.DType.U8, shape=(2, per_cta_n, packed_k))
    # These are already packed physical scale cells.  SFA has one cell for
    # rows 0..31 and one for rows 32..63; SFB has one cell for each of the
    # cluster-wide 32 rows.  Keeping this setup explicit makes the test about
    # MMA interpretation, independent of tcgen05.cp.
    sfa_g = gmem_arg(b, dtype=nr.DType.U32, shape=(2, 32, 2))
    sfb_g = gmem_arg(b, dtype=nr.DType.U32, shape=(2, 32))
    out = gmem_arg(b, dtype=nr.DType.F32, shape=(2, 2, per_cta_m, n))
    a_s = smem_tensor(b, dtype=nr.DType.U8, shape=(per_cta_m, packed_k), byte_offset=0)
    b_s = smem_tensor(b, dtype=nr.DType.U8, shape=(per_cta_n, packed_k), byte_offset=a_bytes)
    a_tile = _tile(b, a_s, per_cta_m, packed_k)
    b_tile = _tile(b, b_s, per_cta_n, packed_k)
    d = _tmem(0)
    sfa = _tmem(64)
    sfb = _tmem(72)
    sfa_reg = reg_tensor(b, dtype=nr.DType.U32, shape=(2,))
    sfb_reg = reg_tensor(b, dtype=nr.DType.U32, shape=(1,))
    frag = reg_tensor(b, dtype=nr.DType.F32, shape=(n // 2,))
    ma = b.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata)
    mb = b.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata + 8)
    rank = b.ctaid_in_cluster()
    tid = b.tid_in_wg()

    with b.if_warp(0):
        b.tmem_alloc(0, 128, addr_byte_offset=tmem_addr_offset, cta_group=2)
        with b.if_elected():
            b.mbarrier_init(ma, count=1)
            b.mbarrier_init(mb, count=1)
    b.cluster_sync()

    with b.if_warp(0), b.if_elected():
        b.mbarrier_arrive_expect_tx(ma, bytes=a_bytes)
        b.tma_load(a_s, a_g, mbar=ma, coords=(rank, 0, 0), shape=(per_cta_m, packed_k))
        b.mbarrier_arrive_expect_tx(mb, bytes=b_bytes)
        b.tma_load(b_s, b_g, mbar=mb, coords=(rank, 0, 0), shape=(per_cta_n, packed_k))
        b.mbarrier_wait(ma, phase=0)
        b.mbarrier_wait(mb, phase=0)
    b.cluster_sync()

    with b.if_warp(0):
        b.reg_load(sfa_reg, sfa_g[rank, b.lane_id(), 0:2])
        b.reg_load(sfb_reg, sfb_g[rank, b.lane_id()])
        b.tcgen05_st(sfa.at(0, 0), sfa_reg, num=2)
        b.tcgen05_st(sfb.at(0, 0), sfb_reg)
        b.tcgen05_wait_st()
    b.cluster_sync()

    scales = nr.BlockScaleSpec(sfa.at(0, 0), sfb.at(0, 0), 0, 0, "e4m3_fn", 4, 1)
    with b.if_warp(0), b.if_elected():
        if both_ctas_issue:
            # No leader routing is implicit: each CTA issues the statement.
            # Give each issuer a disjoint D band so the odd CTA's execution is
            # directly observable after the cluster rendezvous.
            dst = d.at(0, rank * (n // 2))
            b.tcgen05_mma(
                dst,
                b.mma_a_smem(a_tile),
                b_tile,
                mma_m=m,
                mma_n=n,
                format="f4_e2m1",
                block_scale=scales,
                accum=0,
                trans_a=False,
                trans_b=False,
                ws=False,
                cta_group=2,
            )
        else:
            with b.if_(rank.eq(0)):
                b.tcgen05_mma(
                    d.at(0, 0),
                    b.mma_a_smem(a_tile),
                    b_tile,
                    mma_m=m,
                    mma_n=n,
                    format="f4_e2m1",
                    block_scale=scales,
                    accum=0,
                    trans_a=False,
                    trans_b=False,
                    ws=False,
                    cta_group=2,
                )
    b.cluster_sync()

    for issuer in range(2 if both_ctas_issue else 1):
        b.tcgen05_ld(frag, d.at(0, issuer * (n // 2)), num=n // 2)
        b.tcgen05_wait_ld()
        with b.if_(tid < per_cta_m):
            b.reg_store(out[rank, issuer, tid, 0 : n // 2], frag)
        with b.if_(tid >= per_cta_m):
            b.reg_store(out[rank, issuer, tid - per_cta_m, n // 2 : n], frag)

    b.cluster_sync()
    with b.if_warp(0):
        b.tmem_dealloc(0, 128, cta_group=2)
    return b.build(), a_g, b_g, sfa_g, sfb_g, out


def _pack_fp4_even_high(codes):
    codes = np.asarray(codes, dtype=np.uint8)
    return ((codes[..., 0::2] << 4) | codes[..., 1::2]).astype(np.uint8)


def _pack_scale_words(values):
    values = np.asarray(values, dtype=np.uint8)
    shifts = np.arange(4, dtype=np.uint32) * 8
    return np.sum(values.astype(np.uint32) << shifts, axis=-1, dtype=np.uint32)


def _fp4_cg2_inputs():
    rows_a = np.arange(64, dtype=np.uint8)[None, :, None]
    rows_b = np.arange(16, dtype=np.uint8)[None, :, None]
    cols = np.arange(64, dtype=np.uint8)[None, None, :]
    ctas = np.arange(2, dtype=np.uint8)[:, None, None]
    a_codes = 2 + (ctas + rows_a % 3 + cols // 16) % 4
    b_codes = 1 + (2 * ctas + rows_b % 4 + cols // 16) % 4

    scale_codes = np.array([0x30, 0x38, 0x40, 0x48], dtype=np.uint8)
    a_scale_bytes = scale_codes[
        (ctas + rows_a % 4 + np.arange(4, dtype=np.uint8)[None, None, :]) % 4
    ]
    out_ctas = np.arange(2, dtype=np.uint8)[:, None, None]
    n_rows = np.arange(32, dtype=np.uint8)[None, :, None]
    b_scale_bytes = scale_codes[
        (2 * out_ctas + n_rows % 4 + np.arange(4, dtype=np.uint8)[None, None, :]) % 4
    ]

    a_scale_words = _pack_scale_words(a_scale_bytes)
    sfa_cells = np.stack((a_scale_words[:, :32], a_scale_words[:, 32:]), axis=-1)
    sfb_cells = _pack_scale_words(b_scale_bytes)

    e2m1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    e4m3 = {0x30: 0.5, 0x38: 1.0, 0x40: 2.0, 0x48: 4.0}
    a_values = e2m1[a_codes]
    b_values = e2m1[b_codes].reshape(32, 64)
    a_scales = np.vectorize(e4m3.__getitem__, otypes=[np.float32])(a_scale_bytes)
    b_scales = np.vectorize(e4m3.__getitem__, otypes=[np.float32])(b_scale_bytes)
    expected = np.empty((2, 64, 32), dtype=np.float32)
    for cta in range(2):
        scaled_a = a_values[cta] * np.repeat(a_scales[cta], 16, axis=1)
        scaled_b = b_values * np.repeat(b_scales[cta], 16, axis=1)
        expected[cta] = scaled_a @ scaled_b.T
    return (
        _pack_fp4_even_high(a_codes),
        _pack_fp4_even_high(b_codes),
        sfa_cells,
        sfb_cells,
        expected,
    )


def _run_fp4_cg2_m128(*, both_ctas_issue):
    kernel, a_g, b_g, sfa_g, sfb_g, out = _fp4_cg2_m128_kernel(both_ctas_issue=both_ctas_issue)
    a, b_values, sfa, sfb, expected = _fp4_cg2_inputs()
    dump = output(run(kernel, {a_g: a, b_g: b_values, sfa_g: sfa, sfb_g: sfb}), out)
    return dump, expected


def test_tcgen05_fp4_cg2_m128_uses_both_ctas_nonuniform_operands_and_scales():
    dump, expected = _run_fp4_cg2_m128(both_ctas_issue=False)
    np.testing.assert_array_equal(dump[:, 0], expected)


def test_tcgen05_fp4_cg2_odd_cta_executes_without_implicit_leader_routing():
    dump, expected = _run_fp4_cg2_m128(both_ctas_issue=True)
    np.testing.assert_array_equal(dump[:, 0], expected)
    np.testing.assert_array_equal(dump[:, 1], expected)


CP_CASES = [
    ("128x256b", "none", 128, 8),
    ("128x256b", "none", 256, 8),
    ("4x256b", "none", 4, 8),
    ("4x256b", "none", 64, 8),
    ("128x128b", "none", 128, 4),
    ("128x128b", "none", 256, 4),
    ("64x128b", "warp2_02_13", 64, 4),
    ("64x128b", "warp2_02_13", 128, 4),
    ("64x128b", "warp2_01_23", 64, 4),
    ("64x128b", "warp2_01_23", 128, 4),
    ("32x128b", "warp4", 32, 4),
    ("32x128b", "warp4", 64, 4),
]


def _cp_kernel(shape, multicast, rows, cols):
    source_bytes = rows * cols * 4
    is_256b = shape in ("128x256b", "4x256b")
    owner_rows = max(rows, 8) if is_256b else rows
    owner_bytes = owner_rows * cols * 4
    metadata = _align8(owner_bytes)
    b = builder(f"tcgen05_cp_{shape}_{multicast}", smem_size_bytes=metadata + 2 * 8 + 4)
    source_g = gmem_arg(b, dtype=nr.DType.U32, shape=(rows, cols))
    out = gmem_arg(b, dtype=nr.DType.U32, shape=(128, 32))
    source_s = b.tensor(
        space=nr.MemorySpace.SMEM,
        dtype=nr.DType.U32,
        shape=(owner_rows, cols),
        layout=nr.SmemSwizzleLayout(nr.Swizzle.B32) if is_256b else None,
        byte_offset=0,
    )
    source_tile = _tile(b, source_s, rows, cols)
    tmem = _tmem()
    zero = reg_tensor(b, dtype=nr.DType.U32, shape=(32,))
    frag = reg_tensor(b, dtype=nr.DType.U32, shape=(32,))
    loaded = b.mbar(kind=nr.MBarKind.TMA, byte_offset=metadata)
    done = b.mbar(kind=nr.MBarKind.TCGEN05, byte_offset=metadata + 8)

    with b.if_warp(0):
        b.tmem_alloc(0, 32, addr_byte_offset=metadata + 16)
        with b.if_elected():
            b.mbarrier_init(loaded, count=1)
            b.mbarrier_init(done, count=1)
    b.cta_sync()

    with b.if_warpgroup(0):
        b.reg_fill(zero, 0)
        b.tcgen05_st(tmem.at(0, 0), zero, num=32)
        b.tcgen05_wait_st()
        b.wg_sync(barrier_id=1)
        with b.if_(b.tid_in_wg().eq(0)):
            b.mbarrier_arrive_expect_tx(loaded, bytes=source_bytes)
            b.tma_load(
                source_s[0:rows, 0:cols], source_g, mbar=loaded, coords=(0, 0), shape=(rows, cols)
            )
            b.mbarrier_wait(loaded, phase=0)
            b.tcgen05_cp(tmem.at(0, 0), source_tile, shape=shape, multicast=multicast, cta_group=1)
            b.tcgen05_commit(done)
        b.mbarrier_wait(done, phase=0)
        b.tcgen05_ld(frag, tmem.at(0, 0), num=32)
        b.tcgen05_wait_ld()
        b.reg_store(out[b.tid_in_wg(), 0:32], frag)

    b.cta_sync()
    with b.if_warp(0):
        b.tmem_dealloc(0, 32)
    return b.build(), source_g, out


def _expected_cp(shape, multicast, source):
    rows, cols = source.shape
    expected = np.zeros((128, 32), dtype=np.uint32)
    atom_rows = int(shape.split("x", maxsplit=1)[0])
    for row in range(rows):
        for col in range(cols):
            if shape == "4x256b":
                targets = [(row // 4 + 32 * (row % 4), col)]
            elif multicast == "warp2_02_13":
                inner = row % atom_rows
                target_col = (row // atom_rows) * cols + col
                targets = [(inner, target_col), (inner + 64, target_col)]
            elif multicast == "warp2_01_23":
                inner = row % atom_rows
                lane = (inner // 32) * 64 + inner % 32
                target_col = (row // atom_rows) * cols + col
                targets = [(lane, target_col), (lane + 32, target_col)]
            elif multicast == "warp4":
                inner = row % atom_rows
                target_col = (row // atom_rows) * cols + col
                targets = [(inner + 32 * replica, target_col) for replica in range(4)]
            else:
                targets = [(row % atom_rows, (row // atom_rows) * cols + col)]
            for lane, target_col in targets:
                expected[lane, target_col] = source[row, col]
    return expected


@pytest.mark.parametrize("shape,multicast,rows,cols", CP_CASES)
def test_tcgen05_cp_all_shapes_and_multicast_mappings(shape, multicast, rows, cols):
    kernel, source_g, out = _cp_kernel(shape, multicast, rows, cols)
    source = np.fromfunction(
        lambda row, col: (row.astype(np.uint32) << 16) | col.astype(np.uint32),
        (rows, cols),
        dtype=int,
    ).astype(np.uint32)
    np.testing.assert_array_equal(
        output(run(kernel, {source_g: source}), out), _expected_cp(shape, multicast, source)
    )


def _physical_scale_cp_kernel(storage):
    pipe_depth, stage = 2, 1
    dtype = nr.DType.U8 if storage == "byte" else nr.DType.U32
    source_shape = (pipe_depth, 1, 1, 32, 16) if storage == "byte" else (pipe_depth, 1, 32, 4)
    input_shape = (pipe_depth, 128)
    source_backing_bytes = pipe_depth * 128 * 4

    tmem_addr_offset = _align8(source_backing_bytes)
    b = builder(f"tcgen05_cp_physical_scale_{storage}", smem_size_bytes=tmem_addr_offset + 4)
    source_g = gmem_arg(b, dtype=nr.DType.U32, shape=input_shape)
    source_s = b.tensor(space=nr.MemorySpace.SMEM, dtype=dtype, shape=source_shape, byte_offset=0)
    source_store_s = b.tensor(
        space=nr.MemorySpace.SMEM, dtype=nr.DType.U32, shape=input_shape, byte_offset=0
    )
    out = gmem_arg(b, dtype=nr.DType.U32, shape=(128, 4))
    source_tile = _tile(
        b,
        source_s,
        32,
        16 if storage == "byte" else 4,
        prefix=(stage, 0, 0) if storage == "byte" else (stage, 0),
    )
    tmem = _tmem()
    scalar = reg_tensor(b, dtype=nr.DType.U32, shape=(1,))
    frag = reg_tensor(b, dtype=nr.DType.U32, shape=(4,))
    tid = b.tid_in_wg()

    with b.if_warp(0):
        b.tmem_alloc(0, 32, addr_byte_offset=tmem_addr_offset)

    b.reg_load(scalar, source_g[stage, tid])
    b.reg_store(source_store_s[stage, tid], scalar)
    b.cta_sync()

    with b.if_warp(0), b.if_elected():
        b.tcgen05_cp(tmem.at(0, 0), source_tile, shape="32x128b", multicast="warp4", cta_group=1)
    b.cta_sync()

    with b.if_warpgroup(0):
        b.tcgen05_ld(frag, tmem.at(0, 0), num=4)
        b.tcgen05_wait_ld()
        b.reg_store(out[tid, 0:4], frag)

    b.cta_sync()
    with b.if_warp(0):
        b.tmem_dealloc(0, 32)
    return b.build(), source_g, out


def _assert_scale_factor_cp_stage_trace(kernel):
    report = nr.check_protocol(kernel, include_events=True)
    trace_errors = [
        diagnostic
        for diagnostic in report["diagnostics"]
        if diagnostic["code"].startswith("trace_region")
    ]
    assert not trace_errors
    reads = [
        event
        for event in report["events"]
        if event["kind"] == "read" and event.get("access_kind") == "tcgen05_cp"
    ]
    assert len(reads) == 1
    assert reads[0]["region"]["owner"] == {"kind": "smem", "cta_id": 0}
    assert reads[0]["region"]["boxes"] == [{"ranges": [(512, 1024)]}]


def test_tcgen05_cp_plain_physical_byte_scale_roundtrip():
    kernel, source_g, out = _physical_scale_cp_kernel("byte")
    source = np.fromfunction(
        lambda stage, word: (
            (stage.astype(np.uint32) << 28)
            | (word.astype(np.uint32) * np.uint32(0x01010101))
            | np.uint32(0x005500AA)
        ),
        (2, 128),
        dtype=int,
    ).astype(np.uint32)
    expected = np.tile(source[1].reshape(32, 4), (4, 1))
    np.testing.assert_array_equal(output(run(kernel, {source_g: source}), out), expected)
    _assert_scale_factor_cp_stage_trace(kernel)


def test_tcgen05_cp_plain_physical_u32_scale_roundtrip():
    kernel, source_g, out = _physical_scale_cp_kernel("packed_u32")
    source = np.fromfunction(
        lambda stage, row: (
            (stage.astype(np.uint32) << 28)
            | (row.astype(np.uint32) * np.uint32(0x01010101))
            | np.uint32(0x005500AA)
        ),
        (2, 128),
        dtype=int,
    ).astype(np.uint32)
    expected = np.tile(source[1].reshape(32, 4), (4, 1))
    np.testing.assert_array_equal(output(run(kernel, {source_g: source}), out), expected)
    _assert_scale_factor_cp_stage_trace(kernel)


def test_tcgen05_new_schema_rejects_illegal_combinations_fail_closed():
    b = builder("illegal_cp", smem_size_bytes=128 * 8 * 4 + 4)
    source = b.tensor(
        space=nr.MemorySpace.SMEM,
        dtype=nr.DType.U32,
        shape=(128, 8),
        layout=nr.SmemSwizzleLayout(nr.Swizzle.B32),
        byte_offset=0,
    )
    with b.if_warp(0):
        b.tmem_alloc(0, 32, addr_byte_offset=128 * 8 * 4)
    with b.if_warp(0), b.if_elected():
        b.tcgen05_cp(
            _tmem().at(0, 0),
            _tile(b, source, 128, 8),
            shape="128x256b",
            multicast="warp4",
            cta_group=1,
        )
    with pytest.raises(ValueError, match="illegal tcgen05.cp shape/multicast"):
        b.build()

    b = builder("missing_block_scale", smem_size_bytes=128 * 32 + 32 * 32 + 4)
    a = smem_tensor(b, dtype=nr.DType.U8, shape=(128, 32), byte_offset=0)
    bb = smem_tensor(b, dtype=nr.DType.U8, shape=(32, 32), byte_offset=128 * 32)
    with b.if_warp(0):
        b.tmem_alloc(0, 32, addr_byte_offset=128 * 32 + 32 * 32)
    with b.if_warp(0), b.if_elected():
        b.tcgen05_mma(
            _tmem().at(0, 0),
            b.mma_a_smem(_tile(b, a, 128, 32)),
            _tile(b, bb, 32, 32),
            mma_m=128,
            mma_n=32,
            format="f4_e2m1",
            block_scale=None,
            accum=0,
            trans_a=False,
            trans_b=False,
            ws=False,
            cta_group=1,
        )
    with pytest.raises(ValueError, match="requires block_scale"):
        b.build()


def test_tcgen05_resolved_regions_feed_protocol_checker():
    dense, _, _, _ = _dense_smem_kernel(nr.DType.F16)
    report = nr.check_protocol(dense, include_events=True)
    assert report["status"] == "Passed", report["diagnostics"][:2]
    mma_writes = [
        event
        for event in report["events"]
        if event["kind"] == "write"
        and event.get("access_kind") == "mma"
        and event["region"]["owner"]["kind"] == "tmem"
    ]
    assert mma_writes
