"""gdn_prefill intermediate sim<->GPU diff — the bring-up harness applied.

Uses tests/tools/smidiff.py to inject per-chunk SMEM dumps (m_s after the
kk_epi arrive, attn_s after the qk_epi arrive, A_inv post-fold after the
ainv_ready arrive) into the gdn kernel's IR — NO kernel-body edits — then runs
the same instrumented kernel through the interpreter and the GPU and diffs
every dump point-wise.

Site discovery is structural, not name/id-based (the kernel file is unowned):
CG0's chunk body is the top-level If whose subtree contains tcgen05.ld (the
rss readbacks only happen in compute group 0); the three dump sites are its
1st/2nd/4th mbarrier.arrive in first-appearance order (f_kk / f_qk /
ainv_ready in chunk-loop order — qkv_ready is the 3rd and has no tensor to
dump).

Tolerances are CALIBRATED (see smidiff.py's note): m_s is f32 accumulated by
the tensor core in hardware order vs OpenBLAS order in sim, plus exp2 on the
approx unit (~1e-3 abs); attn/A_inv are bf16 (1-2 bf16 ulps at |x| <= ~4 →
atol 0.03). The gate catches structural breakage — the merge bug this
harness was built from produced 0.3 abs errors over 100% of off-diagonal
cells.
"""

import itertools
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tvm = pytest.importorskip("tvm")

import nymph_rs as nr  # noqa: E402
from nymph_rs.kernels import gdn_prefill as gdn  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))
import smidiff  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

BT, K_DIM, V_DIM = gdn.BT, gdn.K_DIM, gdn.V_DIM
NEFF = max(gdn.GdnPrefillConfig().num_q_heads, gdn.GdnPrefillConfig().num_v_heads)
# SMEM byte offsets, re-derived from the module's public constants (the kernel
# packs k(2 stages)/q/v/vnewt first, then attn; ainv aliases vnewt).
_OFF_VNEWT = (2 * BT * K_DIM + BT * K_DIM + BT * V_DIM) * 2
_OFF_ATTN = _OFF_VNEWT + V_DIM * BT * 2

# Calibrated tolerances (see the module docstring).
TOL = {
    "m_s": (1e-4, 0.0),  # f32; measured max_abs 1.1e-06 (ns1_t64) — 100x margin
    "attn": (3e-3, 0.0),  # bf16; measured 1.2e-04 — ~25x margin
    "ainv": (1e-4, 0.0),  # bf16; measured 9.5e-07 — 100x margin
}


def _cg0_body(kernel):
    """CG0's top-level If: the branch whose subtree contains tcgen05.ld (the
    rss readbacks only happen in compute group 0)."""
    for s in kernel.body:
        if s.kind == "if" and any(x.kind == "tcgen05_ld" for x in smidiff.walk(s.body)):
            return s.body
    raise AssertionError("CG0 branch not found (kernel structure changed?)")


def _build_instrumented(seq_lens):
    config = gdn.GdnPrefillConfig(
        num_seqs=len(seq_lens), seqlen=max(seq_lens), varlen=len(seq_lens) > 1
    )
    kernel = gdn.build_gdn_prefill(config)
    arrive_ids = []
    for s in smidiff.walk(_cg0_body(kernel)):
        if s.kind == "mbarrier_arrive" and s.mbar_id not in arrive_ids:
            arrive_ids.append(s.mbar_id)
    assert len(arrive_ids) >= 4, f"expected >=4 CG0 arrives, got {arrive_ids}"
    f_kk, f_qk, ainv_ready = arrive_ids[0], arrive_ids[1], arrive_ids[3]
    num_work = len(seq_lens) * NEFF  # (seq, eh) tasks — per-task dump slots
    specs = [
        smidiff.DumpSpec(
            "m_s",
            smidiff.smem_match(dtype=nr.DType.F32, shape=(BT, BT)),
            after=smidiff.arrive_on(f_kk),
            task_mod=num_work,
        ),
        smidiff.DumpSpec(
            "attn",
            smidiff.smem_match(dtype=nr.DType.BF16, shape=(BT, BT), byte_offset=_OFF_ATTN),
            after=smidiff.arrive_on(f_qk),
            task_mod=num_work,
        ),
        smidiff.DumpSpec(
            "ainv",
            smidiff.smem_match(dtype=nr.DType.BF16, shape=(BT, BT), byte_offset=_OFF_VNEWT),
            # ainv_s ALIASES vnewt_s: after the ainv_ready arrive the region
            # may already be overwritten by CG1's NV staging (a real race in
            # both backends). Dump in the producer-exclusive window instead —
            # right BEFORE the release arrive (post-fold, post-fence).
            before=smidiff.arrive_on(ainv_ready),
            task_mod=num_work,
        ),
    ]
    instrumented, specs = smidiff.inject_dumps(kernel, specs)
    return config, kernel, instrumented, specs


def _torch_bufs(config, orig_kernel, instrumented, data, seq_lens):
    """Positional arg mapping: the instrumented kernel's args are the original
    args in order + the dump args appended."""
    by_id = {}
    cu = data["cu"].cpu().tolist()
    names = ["q", "k", "v", "gate", "beta", "out", "state"] + (["cu"] if config.varlen else [])
    for t, name in zip(orig_kernel.args, names):
        if name in data:
            th = data[name]
            if tuple(th.shape) != tuple(t.shape):
                # varlen: the GMEM tensors carry the static padded shape; the
                # packed rows sit at their cu offsets (padding irrelevant —
                # the kernel masks OOB).
                padded = torch.zeros(tuple(t.shape), dtype=th.dtype, device="cuda")
                for a, b in itertools.pairwise(cu):
                    padded[a:b] = th[a:b]
                th = padded
            by_id[t.id] = th
        elif name == "out":
            by_id[t.id] = torch.zeros(tuple(t.shape), dtype=data["q"].dtype, device="cuda")
        elif name == "state":
            by_id[t.id] = torch.zeros(tuple(t.shape), dtype=torch.float32, device="cuda")
    bufs = {}
    for t in instrumented.args:
        if t.id in by_id:
            bufs[t.id] = by_id[t.id]
        else:  # a dump arg
            dt = torch.float32 if str(t.dtype) == str(nr.DType.F32) else torch.bfloat16
            bufs[t.id] = torch.zeros(tuple(t.shape), dtype=dt, device="cuda")
    return bufs


def _numpy_inputs(orig_kernel, data):
    """Sim inputs keyed by arg tensor. Varlen args carry the static PADDED
    shape (num_seqs*maxlen tokens; the kernel masks OOB) — mirror
    `_nymph_callable`'s zero padding with the packed rows at their cu
    offsets."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    cu = data["cu"].cpu().tolist()
    out = {}
    for t, name in zip(orig_kernel.args, ["q", "k", "v", "gate", "beta", "out", "state", "cu"]):
        if name not in data:
            continue
        th = data[name]
        np_t = tuple(t.shape)
        if th.dtype == torch.bfloat16:
            arr = th.view(torch.uint16).cpu().numpy().view(ml_dtypes.bfloat16)
        else:
            arr = th.cpu().numpy()
        if tuple(arr.shape) != np_t:
            padded = np.zeros(np_t, dtype=arr.dtype)
            for a, b in itertools.pairwise(cu):
                padded[a:b] = arr[a:b]
            arr = padded
        out[t] = arr
    return out


@pytest.mark.parametrize("seq_lens", [[64], [70, 130]], ids=["ns1_t64", "v_70_130"])
def test_gdn_intermediate_sim_gpu_diff(seq_lens):
    config, kernel, instrumented, specs = _build_instrumented(seq_lens)
    data = gdn._bench_inputs(seq_lens, config.io_dtype)
    bufs = _torch_bufs(config, kernel, instrumented, data, seq_lens)
    inputs = _numpy_inputs(kernel, data)
    sim_vals, gpu_vals = smidiff.run_both(instrumented, inputs, specs, bufs)
    report = smidiff.diff_dumps(sim_vals, gpu_vals, tolerances=TOL)
    for label, r in report.items():
        print(f"{label}: max_abs={r['max_abs']:.5g} mismatch={r['mismatch']}")
    bad = {k: v for k, v in report.items() if v["mismatch"]}
    assert not bad, f"intermediate sim<->GPU divergence: {bad}"


if __name__ == "__main__":
    test_gdn_intermediate_sim_gpu_diff([64])
