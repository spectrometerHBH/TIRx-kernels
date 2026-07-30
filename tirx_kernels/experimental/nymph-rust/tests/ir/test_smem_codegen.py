"""Structural coverage for unified dynamic shared-memory emission."""

import pytest

n = pytest.importorskip("nymph_rs")


def _kernel_with_overlapping_storage():
    arg = n.Tensor(space=n.MemorySpace.GMEM, dtype=n.DType.U32, shape=[1])
    high = n.Tensor(space=n.MemorySpace.SMEM, dtype=n.DType.U32, shape=[4], byte_offset=32)
    alias_a = n.Tensor(space=n.MemorySpace.SMEM, dtype=n.DType.U32, shape=[4], byte_offset=0)
    alias_b = n.Tensor(space=n.MemorySpace.SMEM, dtype=n.DType.U32, shape=[4], byte_offset=0)
    first = n.MBar(kind=n.MBarKind.THREAD, byte_offset=0, stages=2)
    second = n.MBar(kind=n.MBarKind.TMA, byte_offset=0)
    body = (
        # Tensor ids are non-monotonic with physical offsets. Codegen may sort
        # identities, but must retain each tensor's explicit address.
        n.TensorDef(high),
        n.TensorDef(alias_a),
        n.TensorDef(alias_b),
        n.MBarDef(first),
        n.MBarDef(second),
        n.If(
            cond=n.ScopeValue(kind="warp_id").eq(0),
            then_body=(n.TmemAlloc(0, 32, addr_byte_offset=0),),
        ),
    )
    return n.Kernel(
        name="overlapping_storage",
        args=(arg,),
        body=body,
        num_warps=4,
        smem_size_bytes=64,
        launch_shape=[1],
        cluster_shape=[1],
    )


def test_codegen_uses_one_dynamic_pool_and_exact_explicit_offsets():
    src = n.kernel_to_tirx_source(_kernel_with_overlapping_storage())

    assert src.count("pool = T.SMEMPool()") == 1
    assert "T.alloc_shared" not in src
    assert "T.alloc_buffer(" not in src
    assert src.count('scope="shared.dyn"') == 6
    assert src.count("pool.move_base_to(0)") == 5
    assert src.count("pool.move_base_to(32)") == 1
    assert 'pool.alloc((4,), "uint32"' in src
    assert 'pool.alloc([2], "uint64"' in src
    assert 'pool.alloc([1], "uint32", scope="shared.dyn", align=4)' in src
    assert "pool.commit(size=64)" in src


def test_codegen_omits_tmem_address_cell_without_tmem_alloc():
    arg = n.Tensor(space=n.MemorySpace.GMEM, dtype=n.DType.U32, shape=[1])
    scratch = n.Tensor(space=n.MemorySpace.SMEM, dtype=n.DType.U32, shape=[4], byte_offset=0)
    kernel = n.Kernel(
        name="no_tmem",
        args=(arg,),
        body=(n.TensorDef(scratch),),
        num_warps=4,
        smem_size_bytes=16,
        launch_shape=[1],
        cluster_shape=[1],
    )
    src = n.kernel_to_tirx_source(kernel)

    assert "tmem_addr = pool.alloc" not in src
    assert "pool.commit(size=16)" in src
