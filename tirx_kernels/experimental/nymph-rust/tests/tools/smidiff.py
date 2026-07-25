"""SMEM intermediate dump injection + sim<->GPU diff — the kernel bring-up harness.

Standard equipment for bringing up a new kernel on the nymph IR (see
docs/kernels/gdn_prefill/DEBUG_BRINGUP.md for the discovery log this
productizes). The harness:

1. INJECTS per-chunk SMEM→GMEM dump ops into a BUILT kernel's IR — an optional
   IR pass over `Kernel.body`, no edits to the kernel's compute body (the
   kernel builder ships untouched; the instrumented variant is a new Kernel
   with one extra GMEM dump arg per dump site).
2. Runs the SAME instrumented kernel through BOTH execution backends of the
   IR: the CPU value simulator (`nr.interpret`, the semantic authority) and
   the tirx codegen + `tvm.compile` on a real GPU.
3. Diffs every dump tensor point-by-point and reports the FIRST divergence as
   `(cell, sim_value, gpu_value, abs_diff)`. Tolerance defaults to bit-exact
   and is configurable per call.

The injected dump block reuses the per-thread point-store form the shipped
kernels already use (RegLoad SMEM cell → RegStore GMEM cell under an explicit
`tid_in_wg < rows` branch), so the instrumented kernel passes the same
validator and lowering paths as the original.

3-line usage (gdn's m_s as the example):

    k2, specs = inject_dumps(build_gdn_prefill(cfg), [DumpSpec(
        "m_s", smem_match(shape=(64, 64), dtype=nr.DType.F32),
        after=arrive_on(f_kk_id), slot="enclosing_loop")])
    sim, gpu = run_both(k2, inputs, specs, torch_bufs)
    report = diff_dumps(sim, gpu)

Interpreter-vs-GPU value note: the simulator is the reference — every
comparison here is sim vs gpu directly (no numpy replica of the kernel's
math). Bit-exactness holds for exact-arithmetic recipes (see
tests/gpu/test_gpu_sim_parity.py); float kernels with bf16 tensor-core
inputs / approx transcendental units diverge at rounding level, so
per-tensor tolerances must be CALIBRATED once per kernel (run once, read the
reported max diffs, then pin tolerances ~5-10x above that with a comment —
the gate exists to catch structural breakage (wrong tile / transposed
operand / dropped chunk), not ulps).
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import nymph_rs as nr

# ---------------------------------------------------------------------------
# tensor / site matchers + discovery aids
# ---------------------------------------------------------------------------


def smem_match(*, dtype=None, shape=None, byte_offset=None):
    """Match a SMEM tensor by dtype (nr.DType), shape tuple, and/or byte_offset —
    the stable IR-level identity of a staging buffer (no kernel-internal names
    are pinned: ids/names are codegen-assigned)."""

    def match(t) -> bool:
        if str(t.space) != str(nr.MemorySpace.SMEM):
            return False
        if dtype is not None and str(t.dtype) != str(dtype):
            return False
        if shape is not None and tuple(t.shape) != tuple(shape):
            return False
        if byte_offset is not None and t.byte_offset != byte_offset:
            return False
        return True

    return match


def arrive_on(mbar_id: int):
    """Injection site: right AFTER an mbarrier.arrive on the given barrier —
    the natural 'producers done' point (the kernel's own wg_sync/fence ahead
    of the arrive makes the staging tensor readable)."""
    return lambda s: s.kind == "mbarrier_arrive" and s.mbar_id == mbar_id


def after_kind(kind: str, n: int = 1):
    """Injection site: right after the Nth statement of the given kind within
    each body (e.g. after_kind("wg_sync", 3) — structural, no ids needed)."""
    state = {"seen": 0}

    def match(s) -> bool:
        if s.kind != kind:
            return False
        state["seen"] += 1
        return state["seen"] == n

    return match


def list_arrive_ids(kernel) -> list[int]:
    """mbar ids of mbarrier.arrive statements, first-appearance order (the
    discovery aid when the kernel's name->id map is not exported)."""
    out: list[int] = []
    for s in walk(kernel.body):
        if s.kind == "mbarrier_arrive" and s.mbar_id not in out:
            out.append(s.mbar_id)
    return out


def walk(stmts):
    """Yield every statement, recursively through container bodies."""
    for s in stmts:
        yield s
        if s.kind in ("if", "for_loop", "for_each_task", "loop", "scheduler_impl"):
            yield from walk(s.body)


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------


@dataclass
class DumpSpec:
    """One injected dump: `label` (report name), `tensor` matcher, the site
    (`after` = insert right after the matched statement; `before` = right
    before it — e.g. before a release arrive when the region is still
    producer-exclusive; exactly one of the two must be set), `slot` for the
    dump's chunk index — "enclosing_loop" (innermost for_loop var at the
    site) or an explicit scalar. `max_slots` caps the chunk slots (the guard
    is emitted into the IR)."""

    label: str
    tensor: object
    after: object = None
    before: object = None
    slot: object = "enclosing_loop"
    max_slots: int = 8
    # Persistent-scheduler launches run MANY tasks (e.g. gdn's (seq, eh) work
    # items), each with its own chunk loop — dumping by chunk alone races all
    # tasks onto the same slots. `task_mod` widens the dump arg with a leading
    # task dimension indexed by the enclosing for_each_task var (guard
    # task < task_mod). Required whenever num_work > 1.
    task_mod: int | None = None
    # filled by inject_dumps:
    arg: object = field(default=None, repr=False)
    scratch: object = field(default=None, repr=False)
    rows: int = field(default=0, repr=False)
    cols: int = field(default=0, repr=False)


def _find_tensor(kernel, match):
    for t in kernel.args:
        if match(t):
            return t
    for s in walk(kernel.body):
        if s.kind == "tensor_def" and match(s.tensor):
            return s.tensor
    raise KeyError("DumpSpec tensor not found (shape/dtype/byte_offset mismatch)")


def inject_dumps(kernel, specs: list[DumpSpec]):
    """Return (instrumented_kernel, specs) with one extra GMEM dump arg per
    spec and per-thread point-store dump blocks injected after each matching
    site. 2-D SMEM tensors only (rows x cols point dump)."""
    new_args = list(kernel.args)
    resolved = {}
    for spec in specs:
        t = _find_tensor(kernel, spec.tensor)
        if len(t.shape) != 2:
            raise ValueError(f"DumpSpec {spec.label}: only 2-D tensors (got {t.shape})")
        spec.rows, spec.cols = t.shape
        shape = (spec.max_slots, spec.rows, spec.cols)
        if spec.task_mod is not None:
            shape = (spec.task_mod, *shape)
        spec.arg = nr.Tensor(space=nr.MemorySpace.GMEM, dtype=t.dtype, shape=shape)
        spec.scratch = nr.Tensor(space=nr.MemorySpace.REG, dtype=t.dtype, shape=(1,))
        resolved[spec.label] = t
        new_args.append(spec.arg)
    body = _inject_into(kernel.body, specs, resolved, loop_stack=[])
    instrumented = nr.Kernel(
        name=f"{kernel.name}_smidiff",
        args=tuple(new_args),
        body=tuple(body),
        num_warps=kernel.num_warps,
        smem_size_bytes=kernel.smem_size_bytes,
        launch_shape=list(kernel.launch_shape),
        cluster_shape=list(kernel.cluster_shape),
        smem_pool=kernel.smem_pool,
    )
    return instrumented, specs


def _dump_block(spec, t, slot_scalar, task_scalar):
    """The injected statements for one site: guard + per-thread point dump."""
    tid = nr.ScopeValue(kind="tid_in_wg")
    cond = (tid < spec.rows) & (slot_scalar < spec.max_slots)
    idx = [slot_scalar, tid]
    if task_scalar is not None:
        cond = cond & (task_scalar < spec.task_mod)
        idx = [task_scalar, *idx]
    sl = nr.TensorSlice
    body = [nr.TensorDef(spec.scratch)]
    for j in range(spec.cols):
        body.append(
            nr.RegLoad(
                sl(tensor=spec.scratch, offsets=(0,), shape=(1,)),
                sl(tensor=t, offsets=(tid, j), shape=(1, 1)),
            )
        )
        body.append(
            nr.RegStore(
                sl(tensor=spec.arg, offsets=(*idx, j), shape=(1,) * len(idx) + (1,)),
                sl(tensor=spec.scratch, offsets=(0,), shape=(1,)),
            )
        )
    return nr.If(cond=cond, then_body=tuple(body))


def _slot_task(spec, loop_stack, task_var):
    if spec.slot == "enclosing_loop":
        if not loop_stack:
            raise ValueError(
                f"DumpSpec {spec.label}: slot='enclosing_loop' but the site is "
                "not inside any for_loop"
            )
        slot_scalar = loop_stack[-1]
    else:
        slot_scalar = spec.slot
    task_scalar = None
    if spec.task_mod is not None:
        if task_var is None:
            raise ValueError(
                f"DumpSpec {spec.label}: task_mod set but the site is not inside any for_each_task"
            )
        task_scalar = task_var
    return slot_scalar, task_scalar


def _inject_into(stmts, specs, resolved, loop_stack, task_var=None):
    out = []
    for s in stmts:
        for spec in specs:
            if spec.before is not None and spec.before(s):
                out.append(
                    _dump_block(spec, resolved[spec.label], *_slot_task(spec, loop_stack, task_var))
                )
        if s.kind in ("if", "for_loop", "for_each_task", "loop", "scheduler_impl"):
            inner = [*loop_stack, s.var] if s.kind == "for_loop" else loop_stack
            inner_task = s.var if s.kind == "for_each_task" else task_var
            out.append(
                _rebuild_container(s, _inject_into(s.body, specs, resolved, inner, inner_task))
            )
        else:
            out.append(s)
        for spec in specs:
            if spec.after is None or not spec.after(s):
                continue
            out.append(
                _dump_block(spec, resolved[spec.label], *_slot_task(spec, loop_stack, task_var))
            )
    return out


def _rebuild_container(stmt, new_body):
    """Rebuild a container stmt with the (possibly extended) body; every other
    field round-trips through the public getters/constructors."""
    k = stmt.kind
    if k == "if":
        return nr.If(cond=stmt.cond, then_body=tuple(new_body))
    if k == "for_loop":
        return nr.ForLoop(
            var=stmt.var,
            start=stmt.start,
            stop=stmt.stop,
            step=stmt.step,
            body=tuple(new_body),
            unroll=stmt.unroll,
        )
    if k == "loop":
        return nr.Loop(body=tuple(new_body))
    if k == "for_each_task":
        return nr.ForEachTask(scheduler=stmt.scheduler, var=stmt.var, body=tuple(new_body))
    if k == "scheduler_impl":
        return nr.SchedulerImpl(scheduler=stmt.scheduler, body=tuple(new_body))
    raise ValueError(f"cannot rebuild container kind {k}")


# ---------------------------------------------------------------------------
# run both backends + diff
# ---------------------------------------------------------------------------


def _compile_tirx(kernel, prefix="smidiff"):
    import tvm

    src = nr.kernel_to_tirx_source(kernel)
    path = os.path.join(tempfile.mkdtemp(prefix=prefix), "k.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"nymph_{prefix}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tvm.compile(
        tvm.IRModule({"main": mod.main}), tvm.target.Target("cuda"), tir_pipeline="tirx"
    )


def run_both(kernel, inputs: dict, specs, torch_tensors: dict | None = None):
    """Run the instrumented kernel through the simulator and (optionally) the
    GPU.

    `inputs`: {arg_tensor: numpy} for the simulator (keyed by the arg tensor
    objects, as in the kernels' sim tests; dump args need no input). For the
    GPU run, `torch_tensors` maps EVERY kernel arg (inputs + outputs + dump
    args) to preallocated torch tensors. Returns (sim_vals, gpu_vals):
    {label: ndarray} for the dump args (gpu_vals empty without torch_tensors).
    """
    sim_res = nr.interpret(kernel, inputs)
    sim_vals = {spec.label: np.asarray(sim_res[spec.arg.id]) for spec in specs}
    gpu_vals = {}
    if torch_tensors is not None:
        # Keyed by tensor ID — PyTensor wrappers are recreated per .args call,
        # so object-identity dicts do not survive across accessors.
        by_id = {getattr(t, "id", t): v for t, v in torch_tensors.items()}
        fn = _compile_tirx(kernel)
        fn(*[by_id[t.id] for t in kernel.args])
        import torch

        torch.cuda.synchronize()
        for spec in specs:
            gpu_vals[spec.label] = by_id[spec.arg.id].float().cpu().numpy()
    return sim_vals, gpu_vals


def first_mismatch(sim: np.ndarray, gpu: np.ndarray, *, atol=0.0, rtol=0.0):
    """The first cell exceeding the tolerance: (index_tuple, sim, gpu, absdiff)
    or None. atol=rtol=0 is the bit-exact default (on the f64 view)."""
    s = np.asarray(sim, np.float64)
    g = np.asarray(gpu, np.float64)
    if s.shape != g.shape:
        return (("shape",), s.shape, g.shape, float("inf"))
    bad = np.abs(s - g) > (atol + rtol * np.abs(s))
    if not bad.any():
        return None
    idx = tuple(int(i) for i in np.argwhere(bad)[0])
    return (idx, float(s[idx]), float(g[idx]), float(abs(s[idx] - g[idx])))


def diff_dumps(sim_vals: dict, gpu_vals: dict, *, tolerances: dict | None = None):
    """Diff every dump tensor. Returns {label: {max_abs, mismatch}} with
    mismatch = first_mismatch(...) or None. `tolerances`:
    {label: (atol, rtol)} — defaults to bit-exact."""
    tolerances = tolerances or {}
    report = {}
    for label, sim in sim_vals.items():
        gpu = gpu_vals.get(label)
        atol, rtol = tolerances.get(label, (0.0, 0.0))
        if gpu is None:
            report[label] = {"max_abs": None, "mismatch": ("missing gpu dump",)}
            continue
        d = np.abs(np.asarray(sim, np.float64) - np.asarray(gpu, np.float64))
        report[label] = {
            "max_abs": float(d.max()) if d.size else 0.0,
            "mismatch": first_mismatch(sim, gpu, atol=atol, rtol=rtol),
        }
    return report
