"""Wave-1 gdn_prefill bench: flashinfer baseline + first nymph numbers.

For each representative shape (``kernels/gdn_prefill.py::BENCH_CONFIGS``):

1. compile gate — ``kernel_to_tirx_source`` -> ``tvm.compile`` of the nymph
   kernel. TODAY this fails closed: the gdn IR uses flash/gdn-datapath nodes
   with no TIRx lowering yet (RegUnary, RegFill, RegAdd/RegSub/RegFma,
   Tcgen05St/Tcgen05WaitSt, LdMatrix/StMatrix, WarpMma — the "sim-only ops" of
   LIMITATIONS.md). The exact error is reported per shape.
2. full ``run_bench`` (nymph compile + cosine>=0.999 gate vs flashinfer, then
   both timed by the one bench() call) when the gate passes; otherwise the
   baseline-only ``run_flashinfer_bench`` fallback, so the flashinfer table
   exists from day one.

   CURRENT BASELINE BLOCKER (Wave-1 finding): the flashinfer CuTeDSL kernel
   itself does not compile in this environment — flashinfer 0.6.15 (repo
   517cca9c) + nvidia-cutlass-dsl 4.5.2 ICE at cute.compile
   (``tcgen05.make_tmem_copy`` over ``tmem_load<f32, 32 DP, 32 bit, x32>``:
   "failed to legalize unresolved materialization"), reproduced by
   flashinfer's own tests/gdn/test_prefill_delta_rule.py on the unmodified
   repo. ROOT CAUSE (confirmed read-only): NVIDIA/cutlass#3259 — the
   nvidia-cutlass-dsl-libs-base and -libs-cu13 4.5.2 wheels both own
   nvidia_cutlass_dsl/lib/libcute_dsl_runtime.so with different content, and
   the BASE variant won the install race here (on-disk size 37,288,696 B =
   base's RECORD, cu13's is 39,855,488 B); the base runtime lacks the CUDA-13
   MLIR bytecode tcgen05.make_tmem_copy needs. The repair is to re-extract the
   cu13 wheel last (e.g. ``pip install --force-reinstall --no-deps
   nvidia-cutlass-dsl-libs-cu13==4.5.2``) — an environment fix, deliberately
   NOT applied here: the goal's stop rule says record-and-stop, never patch
   the baseline or mutate the shared env from this wave.
3. baseline sanity — flashinfer output vs the numpy oracle
   (``tests/kernels/_gdn_oracle.py``, the FLA chunked reference the value sim
   is validated against) on the SAME inputs: cosine per shape. This exercises
   the correctness gate's baseline half while the nymph half cannot run.

    python bench/bench_gdn_wave1.py [--rounds 5] [--filter ns1] [--no-oracle]

Requires the nymph_rs extension importable (built wheel in ``_pybuild/`` or
the source-tree ``.so``), tvm on PYTHONPATH, and flashinfer installed.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_NYMPH_RUST = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
for _p in (
    os.path.join(_NYMPH_RUST, "python"),
    _REPO,
    os.path.join(_NYMPH_RUST, "tests", "kernels"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _oracle_cosines(gdn, data, seq_lens, f_out, f_state):
    """cosine(flashinfer, numpy oracle) for out and state, on the same inputs.

    The oracle takes the LOG-domain gate (it cumsums log g internally), so
    feed log(raw_gate); both kernels consume the raw gate. State is compared
    in the oracle's [K,V] layout (flashinfer's [N,H,V,K] transposed).
    """
    import _gdn_oracle as oracle
    import numpy as np
    import torch

    q = data["q"].float().cpu().numpy()
    k = data["k"].float().cpu().numpy()
    v = data["v"].float().cpu().numpy()
    g_log = np.log(data["gate"].float().cpu().numpy())
    beta = data["beta"].float().cpu().numpy()
    o_fi = f_out.float().cpu().numpy()
    s_fi = f_state.transpose(-1, -2).float().cpu().numpy()  # [N,H,K,V]
    scale = gdn.GdnPrefillConfig().scale
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)

    def cos(a, b):
        return float(
            torch.nn.functional.cosine_similarity(
                torch.from_numpy(np.asarray(a, np.float64)).flatten(),
                torch.from_numpy(np.asarray(b, np.float64)).flatten(),
                dim=0,
            )
        )

    o_ref = np.zeros_like(o_fi, dtype=np.float64)
    s_ref = np.zeros_like(s_fi, dtype=np.float64)
    for i in range(len(seq_lens)):
        a, b = cu[i], cu[i + 1]
        for hv in range(gdn.HV):
            hqk = hv // gdn.GVA_RATIO
            o, s = oracle.chunked(
                q[a:b, hqk], k[a:b, hqk], v[a:b, hv], g_log[a:b, hv], beta[a:b, hv],
                scale, None, BT=gdn.BT,
            )  # fmt: skip
            o_ref[a:b, hv], s_ref[i, hv] = o, s
    return cos(o_fi, o_ref), cos(s_fi, s_ref)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--rounds", type=int, default=5, help="bench() rounds per shape")
    ap.add_argument("--filter", default=None, help="shape-label substring filter")
    ap.add_argument("--no-oracle", action="store_true", help="skip the baseline-vs-oracle check")
    args = ap.parse_args()

    import torch
    from nymph_rs.kernels import gdn_prefill as gdn

    rows = []
    baseline_broken = False
    for cfg in gdn.BENCH_CONFIGS:
        label = cfg["label"]
        if args.filter and args.filter not in label:
            continue
        kw = {k: v for k, v in cfg.items() if k != "label"}
        seq_lens = gdn._bench_seqlens(kw.get("num_seqs"), kw.get("seqlen"), kw.get("seqlens"))
        row = {"label": label, "fi_us": None, "nymph_us": None, "note": ""}

        # 1. compile gate (the nymph side). Fast-fails today at codegen.
        gate_err = None
        try:
            gdn._compile_nymph(seq_lens)
        except Exception as e:
            gate_err = f"{type(e).__name__}: {e}"
        row["compile_gate"] = "PASS" if gate_err is None else f"FAIL ({gate_err})"

        # 2. timing: full run_bench when nymph builds, else baseline-only.
        try:
            if gate_err is None:
                res = gdn.run_bench(**kw, rounds=args.rounds)
                row["fi_us"] = res["impls"]["flashinfer"]
                row["nymph_us"] = res["impls"]["tirx"]
            else:
                res = gdn.run_flashinfer_bench(**kw, rounds=args.rounds)
                row["fi_us"] = res["impls"]["flashinfer"]
                row["note"] = "nymph N/A — see compile gate"
        except Exception as e:
            baseline_broken = True
            row["note"] = f"BASELINE ERROR: {type(e).__name__}: {e}"
            traceback.print_exc()
            rows.append(row)
            continue

        # 3. baseline vs numpy oracle (same inputs as the timed run).
        if not args.no_oracle:
            data = gdn._bench_inputs(seq_lens)
            _, f_out, f_state = gdn._flashinfer_callable(data, len(seq_lens))
            torch.cuda.synchronize()
            cos_o, cos_s = _oracle_cosines(gdn, data, seq_lens, f_out, f_state)
            row["oracle_cos"] = f"out={cos_o:.4f} state={cos_s:.4f}"
            ok = min(cos_o, cos_s) >= gdn._CORRECTNESS_COSINE
            row["note"] += "" if ok else f" ORACLE MISMATCH (<{gdn._CORRECTNESS_COSINE})"
            baseline_broken = baseline_broken or not ok
        rows.append(row)

    print("\n=== gdn_prefill Wave-1: flashinfer baseline vs nymph (us, lower is better) ===")
    print(
        f"{'shape':<12} {'flashinfer':>12} {'nymph':>12} {'fi/nymph':>9}  oracle-cos           note"
    )
    for r in rows:
        fi = f"{r['fi_us']:.1f}" if r["fi_us"] is not None else "-"
        ny = f"{r['nymph_us']:.1f}" if r["nymph_us"] is not None else "-"
        ratio = f"{r['fi_us'] / r['nymph_us']:.3f}" if r["nymph_us"] else "-"
        print(
            f"{r['label']:<12} {fi:>12} {ny:>12} {ratio:>9}  {r.get('oracle_cos', ''):<20} {r['note']}"
        )
        print(f"  compile gate: {r['compile_gate']}")
    if baseline_broken:
        sys.exit("bench_gdn_wave1: flashinfer baseline did not run cleanly — see above")


if __name__ == "__main__":
    main()
