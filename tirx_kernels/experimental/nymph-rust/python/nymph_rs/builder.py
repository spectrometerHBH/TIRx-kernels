"""Builder API for clean Nymph IR."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from .nymph_rs import (
    _SCALAR_GMEM_DTYPES,
    BreakIf,
    ClusterShape,
    ClusterSync,
    CpAsyncBulkCommitGroup,
    CpAsyncBulkS2Cluster,
    CpAsyncBulkWaitGroupRead,
    CtaSync,
    DType,
    Fence,
    FenceKind,
    FenceScope,
    ForEachTask,
    ForLoop,
    GmemAtomicAdd,
    GmemWaitEq,
    If,
    Kernel,
    LaunchShape,
    Layout,
    LdMatrix,
    Loop,
    MBar,
    MBarDef,
    MBarKind,
    MBarRef,
    MBarrierArrive,
    MBarrierArriveExpectTx,
    MBarrierExpectTx,
    MBarrierInit,
    MBarrierWait,
    MemorySpace,
    NamedBarrier,
    RegAdd,
    RegBitwise,
    RegCausalMask,
    RegCombineIntFracEx2,
    RegCondRescale,
    RegCvt,
    RegFill,
    RegFma,
    RegLoad,
    RegMax,
    RegMin,
    RegMul,
    RegReduce,
    RegSoftmaxRescale,
    RegStore,
    RegSub,
    RegUnary,
    ScalarDef,
    ScalarDType,
    ScalarStore,
    ScalarValue,
    SchedNext,
    Scheduler,
    SchedulerImpl,
    ScopeValue,
    SetMaxNReg,
    Shape,
    StMatrix,
    Stmt,
    StoreScalar,
    TaskSpace,
    Tcgen05Commit,
    Tcgen05Cp,
    Tcgen05Ld,
    Tcgen05Mma,
    Tcgen05St,
    Tcgen05WaitLd,
    Tcgen05WaitSt,
    Tensor,
    TensorDef,
    TensorSlice,
    TmaLoad,
    TmaStore,
    TmemAlloc,
    TmemDealloc,
    Var,
    VarBinding,
    WarpMma,
    WarpSync,
    WgSync,
)

Tcgen05LdStShape = Literal["32x32b", "16x32bx2", "16x64b", "16x128b", "16x256b"]
MatrixShape = Literal["m8n8"]
MatrixDType = Literal["b16"]
SchedulerPolicy = Literal["grid_stride", "clc", "atomic_steal", "custom"]
RegOperand = Tensor | TensorSlice | int | float
RegUnaryOp = Literal["exp2", "log2", "rcp", "neg"]
RegReduceOp = Literal["max", "sum"]
RegCondScope = Literal["warp", "warpgroup"]

# Single source of the GMEM-scalar tensor-dtype -> ScalarDType mapping lives in
# ir.py (`_SCALAR_GMEM_DTYPES`); the builder reuses it for the default scalar
# dtype of a `ScalarDef(initial=TensorSlice)`.


class TaskToken:
    def __init__(self, var: Var, space: TaskSpace, *, has_valid: bool = False):
        self.task_id = var
        self._space = space
        self._has_valid = has_valid

    @property
    def valid(self) -> ScalarValue:
        return self.task_id >= 0 if self._has_valid else 1

    def field(self, name: str) -> ScalarValue:
        fields = tuple(self._space.fields)
        if name not in fields:
            raise AttributeError(name)
        dim = fields.index(name)
        grid = tuple(self._space.grid)
        stride = 1
        for extent in grid[:dim]:
            stride *= extent
        value = self.task_id if stride == 1 else self.task_id // stride
        extent = grid[dim]
        return 0 if extent == 1 else value % extent

    def __getattr__(self, name: str) -> ScalarValue:
        return self.field(name)


class IRBuilder:
    def __init__(
        self,
        name: str,
        *,
        num_warps: int = 12,
        smem_size_bytes: int = 0,
        launch_shape: LaunchShape = (1,),
        cluster_shape: ClusterShape = (1,),
    ):
        self.name = name
        self.num_warps = num_warps
        self.smem_size_bytes = smem_size_bytes
        self.launch_shape = launch_shape
        self.cluster_shape = cluster_shape
        self._args: list[Tensor] = []
        self._body: list[Stmt] = []
        self._body_stack: list[list[Stmt]] = [self._body]

    def build(self) -> Kernel:
        return Kernel(
            name=self.name,
            args=tuple(self._args),
            body=tuple(self._body),
            num_warps=self.num_warps,
            smem_size_bytes=self.smem_size_bytes,
            launch_shape=self.launch_shape,
            cluster_shape=self.cluster_shape,
        )

    @property
    def launch_cta_count(self) -> int:
        product = 1
        for dim in self.launch_shape:
            product *= dim
        return product

    def arg(
        self,
        *,
        space: MemorySpace,
        dtype: DType,
        shape: Shape,
        layout: Layout | None = None,
        byte_offset: int | None = None,
    ) -> Tensor:
        tensor = Tensor(
            space=space, dtype=dtype, shape=shape, layout=layout, byte_offset=byte_offset
        )
        self._args.append(tensor)
        return tensor

    def tensor(
        self,
        *,
        space: MemorySpace,
        dtype: DType,
        shape: Shape,
        layout: Layout | None = None,
        byte_offset: int | None = None,
    ) -> Tensor:
        tensor = Tensor(
            space=space, dtype=dtype, shape=shape, layout=layout, byte_offset=byte_offset
        )
        self._append(TensorDef(tensor))
        return tensor

    def tmem_alloc(self, tensor: Tensor, *, n_cols: int, cta_group: Literal[1, 2] = 1) -> None:
        self._append(TmemAlloc(tensor=tensor, n_cols=n_cols, cta_group=cta_group))

    def tmem_dealloc(self, tensor: Tensor, *, n_cols: int, cta_group: Literal[1, 2] = 1) -> None:
        self._append(TmemDealloc(tensor=tensor, n_cols=n_cols, cta_group=cta_group))

    def scalar(
        self, *, initial: ScalarValue | TensorSlice = 0, dtype: ScalarDType | None = None
    ) -> Var:
        if dtype is None:
            dtype = (
                _SCALAR_GMEM_DTYPES.get(initial.tensor.dtype, ScalarDType.U32)
                if isinstance(initial, TensorSlice)
                else ScalarDType.U32
            )
        var = Var(binding=VarBinding.SCALAR, dtype=dtype)
        self._append(ScalarDef(var=var, initial=initial))
        return var

    def scalar_store(self, var: Var, value: ScalarValue) -> None:
        self._append(ScalarStore(var=var, value=value))

    def store_scalar(self, dst: Tensor | TensorSlice, value: ScalarValue) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        self._append(StoreScalar(dst=dst, value=value))

    def tid_in_wg(self) -> ScopeValue:
        return ScopeValue(kind="tid_in_wg")

    def lane_id(self) -> ScopeValue:
        return ScopeValue(kind="lane_id")

    def warp_id(self) -> ScopeValue:
        return ScopeValue(kind="warp_id")

    def warpgroup_id(self) -> ScopeValue:
        return ScopeValue(kind="warpgroup_id")

    def ctaid_in_cluster(self) -> ScopeValue:
        return ScopeValue(kind="ctaid_in_cluster")

    def cta_id(self) -> ScopeValue:
        return ScopeValue(kind="cta_id")

    def nvshmem_my_pe(self) -> ScopeValue:
        return ScopeValue(kind="nvshmem_my_pe")

    def mbar(self, *, kind: MBarKind, stages: int = 1, arrive_count: int | None = None) -> MBar:
        mbar = MBar(kind=kind, stages=stages, arrive_count=arrive_count)
        self._append(MBarDef(mbar))
        return mbar

    def mbar_ref(self, mbar: MBar, *, remote_coord: ScalarValue | None = None) -> MBarRef:
        return MBarRef(mbar=mbar, remote_coord=remote_coord)

    def mbarrier_init(
        self, mbar: MBar | MBarRef, *, count: int, stage: ScalarValue | None = None
    ) -> None:
        stmt = MBarrierInit(mbar=mbar, count=count, stage=stage)
        self._append(stmt)

    def mbarrier_arrive(
        self, mbar: MBar | MBarRef, *, stage: ScalarValue | None = None, count: ScalarValue = 1
    ) -> None:
        stmt = MBarrierArrive(mbar=mbar, stage=stage, count=count)
        self._append(stmt)

    def mbarrier_wait(
        self,
        mbar: MBar | MBarRef,
        *,
        stage: ScalarValue | None = None,
        phase: ScalarValue | None = None,
    ) -> None:
        stmt = MBarrierWait(mbar=mbar, stage=stage, phase=phase)
        self._append(stmt)

    def mbarrier_expect_tx(
        self, mbar: MBar | MBarRef, *, bytes: int, stage: ScalarValue | None = None
    ) -> None:
        stmt = MBarrierExpectTx(mbar=mbar, bytes=bytes, stage=stage)
        self._append(stmt)

    def mbarrier_arrive_expect_tx(
        self, mbar: MBar | MBarRef, *, bytes: int, stage: ScalarValue | None = None
    ) -> None:
        stmt = MBarrierArriveExpectTx(mbar=mbar, bytes=bytes, stage=stage)
        self._append(stmt)

    def tma_load(
        self,
        dst: Tensor | TensorSlice,
        src: Tensor,
        *,
        mbar: MBar | MBarRef,
        bytes: ScalarValue,
        coords: tuple[ScalarValue, ...],
        shape: Shape,
        gmem_shape: Shape | None = None,
        mbar_stage: ScalarValue | None = None,
        multicast_cta_mask: int | None = None,
        cta_group: int = 1,
    ) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        stmt = TmaLoad(
            dst=dst,
            src=src,
            mbar=mbar,
            bytes=bytes,
            coords=coords,
            shape=shape,
            gmem_shape=gmem_shape,
            mbar_stage=mbar_stage,
            multicast_cta_mask=multicast_cta_mask,
            cta_group=cta_group,
        )
        self._append(stmt)

    def tma_store(
        self,
        dst: Tensor,
        src: Tensor | TensorSlice,
        *,
        coords: tuple[ScalarValue, ...],
        shape: Shape,
        gmem_shape: Shape | None = None,
    ) -> None:
        if isinstance(src, Tensor):
            src = src[...]
        stmt = TmaStore(dst=dst, src=src, coords=coords, shape=shape, gmem_shape=gmem_shape)
        self._append(stmt)

    def tma_reduce_add(
        self,
        dst: Tensor,
        src: Tensor | TensorSlice,
        *,
        coords: tuple[ScalarValue, ...],
        shape: Shape,
        gmem_shape: Shape | None = None,
        allow_nondet_reduce: bool = False,
    ) -> None:
        """TMA reduce-add (``cp.reduce.async.bulk...add.f32``): atomically accumulate
        ``dst += src`` (SMEM->GMEM, f32). Same bulk-async path as ``tma_store`` (commit
        with ``cp_async_bulk_commit_group`` / wait with ``cp_async_bulk_wait_group_read``);
        value-mode accumulates instead of overwriting. dst must be an f32 GMEM tensor.

        A float reduction is not associative: cross-CTA reduce-adds into one location
        are race-free (hardware-atomic, commutative) but ORDER-DEPENDENT — the result is
        not bit-reproducible. The protocol checker can only warn
        (``nondeterministic_reduction``) about that, so the IR makes the non-determinism
        opt-in: validate REJECTS a float reduce-add unless ``allow_nondet_reduce=True``
        is passed (the checker's warning still fires with the flag set)."""
        if isinstance(src, Tensor):
            src = src[...]
        stmt = TmaStore(
            dst=dst,
            src=src,
            coords=coords,
            shape=shape,
            gmem_shape=gmem_shape,
            reduce_add=True,
            allow_nondet_reduce=allow_nondet_reduce,
        )
        self._append(stmt)

    def cp_async_bulk_s2cluster(
        self,
        dst: Tensor | TensorSlice,
        src: Tensor | TensorSlice,
        *,
        mbar: MBar | MBarRef,
        bytes: ScalarValue,
    ) -> None:
        """`cp.async.bulk.shared::cluster` — async bulk copy of this CTA's SMEM `src`
        into a PEER CTA's SMEM `dst` (the peer instance), signalling the peer's `mbar`
        (pass `mbar_ref(m, remote_coord=peer)`) on completion. The dS cross-CTA
        exchange in the 2-CTA backward. The peer CTA = the mbar's target."""
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(src, Tensor):
            src = src[...]
        self._append(CpAsyncBulkS2Cluster(dst=dst, src=src, mbar=mbar, bytes=bytes))

    def gmem_atomic_add(
        self,
        sem: Tensor | TensorSlice,
        *,
        coords: tuple[ScalarValue, ...],
        value: ScalarValue,
        order: Literal["release", "relaxed"] = "release",
    ) -> None:
        """GMEM semaphore atomic-add ("signal") — ``red.<order>.gpu.global.add.s32``:
        ``sem[coords] += value``. A value-keyed RELEASE: it is a SYNC op (NOT a data
        access on the semaphore tensor), publishing this stream's clock as the release
        clock for the post-increment counter value, so a later ``gmem_wait_eq`` on that
        exact value joins it (acquire). ``order='release'`` carries the release fence
        ordering prior writes (incl. a drained reduce-add) before the publish. ``sem``
        must be an i32 GMEM tensor. Single-thread issue: elect one thread."""
        if isinstance(sem, Tensor):
            sem = sem[...]
        self._append(GmemAtomicAdd(sem=sem, coords=coords, value=value, order=order))

    def gmem_wait_eq(
        self, sem: Tensor | TensorSlice, *, coords: tuple[ScalarValue, ...], value: ScalarValue
    ) -> None:
        """GMEM semaphore spin-wait ("wait") — ``ld.global.acquire.gpu`` poll until
        ``sem[coords] == value``. A value-keyed ACQUIRE: blocks this stream until the
        counter equals ``value`` (never reaching it -> deadlock), and joins the release
        clock published by the ``gmem_atomic_add`` that produced THIS exact value. A
        SYNC op (no data access on the semaphore tensor). ``sem`` must be i32 GMEM."""
        if isinstance(sem, Tensor):
            sem = sem[...]
        self._append(GmemWaitEq(sem=sem, coords=coords, value=value))

    def cp_async_bulk_commit_group(self) -> None:
        self._append(CpAsyncBulkCommitGroup())

    def cp_async_bulk_wait_group_read(self, n: int = 0) -> None:
        self._append(CpAsyncBulkWaitGroupRead(n))

    @contextmanager
    def for_loop(
        self, *, stop: ScalarValue, start: ScalarValue = 0, step: ScalarValue = 1
    ) -> Iterator[Var]:
        var = Var()
        body: list[Stmt] = []
        self._body_stack.append(body)
        try:
            yield var
        except Exception:
            self._body_stack.pop()
            raise
        else:
            self._body_stack.pop()
            self._append(ForLoop(var=var, start=start, stop=stop, step=step, body=tuple(body)))

    def task_space(self, *, grid: tuple[int, ...], fields: tuple[str, ...]) -> TaskSpace:
        return TaskSpace(grid=grid, fields=fields)

    def scheduler(
        self,
        space: TaskSpace,
        *,
        policy: SchedulerPolicy = "grid_stride",
        scope: Literal["cluster"] = "cluster",
    ) -> Scheduler:
        return Scheduler(space=space, policy=policy, scope=scope)

    @contextmanager
    def for_each_task(self, scheduler: Scheduler) -> Iterator[TaskToken]:
        var = Var(binding=VarBinding.TASK)
        token = TaskToken(var, scheduler.space)
        body: list[Stmt] = []
        self._body_stack.append(body)
        try:
            yield token
        except Exception:
            self._body_stack.pop()
            raise
        else:
            self._body_stack.pop()
            self._append(ForEachTask(scheduler=scheduler, var=var, body=tuple(body)))

    @contextmanager
    def scheduler_impl(self, scheduler: Scheduler) -> Iterator[None]:
        body: list[Stmt] = []
        self._body_stack.append(body)
        try:
            yield
        except Exception:
            self._body_stack.pop()
            raise
        else:
            self._body_stack.pop()
            self._append(SchedulerImpl(scheduler=scheduler, body=tuple(body)))

    def sched_next(self, scheduler: Scheduler) -> TaskToken:
        var = Var(binding=VarBinding.TASK)
        self._append(SchedNext(scheduler=scheduler, var=var))
        return TaskToken(var, scheduler.space, has_valid=True)

    @contextmanager
    def loop(self) -> Iterator[None]:
        body: list[Stmt] = []
        self._body_stack.append(body)
        try:
            yield
        except Exception:
            self._body_stack.pop()
            raise
        else:
            self._body_stack.pop()
            self._append(Loop(body=tuple(body)))

    def break_if(self, cond: ScalarValue) -> None:
        self._append(BreakIf(cond=cond))

    @contextmanager
    def if_(self, cond: ScalarValue) -> Iterator[None]:
        body: list[Stmt] = []
        self._body_stack.append(body)
        try:
            yield
        except Exception:
            self._body_stack.pop()
            raise
        else:
            self._body_stack.pop()
            self._append(If(cond=cond, then_body=tuple(body)))

    # -- thread-dispatch sugar -----------------------------------------------
    # Canonical `if_` shapes over thread coordinates. These are ordinary Ifs
    # (masked per-thread execution, exactly the hardware model); the canonical
    # predicate forms are what `static_thread_filter` resolves for validation.

    @contextmanager
    def if_warp(self, warp: int) -> Iterator[None]:
        """Threads of one warp: `warp_id() == warp`."""
        with self.if_(self.warp_id().eq(warp)):
            yield

    @contextmanager
    def if_warpgroup(self, warpgroup: int) -> Iterator[None]:
        """Threads of one warpgroup: `warpgroup_id() == warpgroup`."""
        with self.if_(self.warpgroup_id().eq(warpgroup)):
            yield

    @contextmanager
    def if_lane(self, lane: int) -> Iterator[None]:
        """One lane of every warp in context: `lane_id() == lane`."""
        with self.if_(self.lane_id().eq(lane)):
            yield

    @contextmanager
    def if_elected(self) -> Iterator[None]:
        """Lane 0 of EVERY warp in context (`lane_id() == 0`).

        For one thread per warpgroup use `if_(tid_in_wg().eq(0))`; for one
        thread per CTA nest this under `if_warp(w)`.
        """
        with self.if_(self.lane_id().eq(0)):
            yield

    def set_maxnreg(self, nreg: int) -> None:
        """`setmaxnreg` directive for the enclosing warpgroup(s); sim metadata.

        Validation requires the enclosing branch to statically cover whole
        warpgroups and `nreg` to be a positive multiple of 8.
        """
        self._append(SetMaxNReg(nreg=nreg))

    def tcgen05_mma(
        self,
        dst: Tensor | TensorSlice,
        a: Tensor | TensorSlice,
        b: Tensor | TensorSlice,
        *,
        m: int,
        n: int,
        k: int = 16,
        accum: bool = False,
        trans_a: bool = False,
        trans_b: bool = False,
        cta_group: Literal[1, 2] = 1,
        sfa: Tensor | TensorSlice | None = None,
        sfb: Tensor | TensorSlice | None = None,
        sf_byte: int = 0,
        sf_e4m3: bool = False,
        sf_block: int = 0,
        a_fp4: bool = False,
        b_fp4: bool = False,
    ) -> None:
        """``sfa``/``sfb`` make this a block-scaled MMA: each is a (128, cols) u32
        TMEM slice of packed scale bytes. Two scale modes share the field set:

        * fp8 (``kind::mxf8f6f4`` + UE8M0, default): one scale per operand row;
          ``sf_byte`` picks the packed byte for this MMA's k-slice, dequant
          2^(byte - 127).
        * nvfp4 (``kind::mxf4`` + e4m3, ``sf_e4m3=True, sf_block=16``): one scale
          per 16 contiguous k-elements; this MMA's k spans k/16 blocks whose
          scales are bytes 0..k/16 of the cell, each decoded as e4m3.
          ``a_fp4``/``b_fp4`` mark the operands as packed e2m1 (2 per u8 byte,
          the SMEM tile's inner extent is k/2 bytes)."""
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(a, Tensor):
            a = a[...]
        if isinstance(b, Tensor):
            b = b[...]
        if isinstance(sfa, Tensor):
            sfa = sfa[...]
        if isinstance(sfb, Tensor):
            sfb = sfb[...]
        stmt = Tcgen05Mma(
            dst=dst,
            a=a,
            b=b,
            m=m,
            n=n,
            k=k,
            accum=accum,
            trans_a=trans_a,
            trans_b=trans_b,
            cta_group=cta_group,
            sfa=sfa,
            sfb=sfb,
            sf_byte=sf_byte,
            sf_e4m3=sf_e4m3,
            sf_block=sf_block,
            a_fp4=a_fp4,
            b_fp4=b_fp4,
        )
        self._append(stmt)

    def tcgen05_cp(
        self, dst: Tensor | TensorSlice, src: Tensor | TensorSlice, *, cta_group: Literal[1, 2] = 1
    ) -> None:
        """``tcgen05.cp`` — bulk-copy packed u32 scale cells from SMEM into TMEM.
        With ``cta_group=2`` the leader's single issue drives both CTAs: each CTA
        copies from its own SMEM into its own TMEM (row r -> lane r % 128,
        col base + r // 128)."""
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(src, Tensor):
            src = src[...]
        self._append(Tcgen05Cp(dst=dst, src=src, cta_group=cta_group))

    def tcgen05_commit(
        self,
        mbar: MBar | MBarRef,
        *,
        stage: ScalarValue | None = None,
        cta_group: Literal[1, 2] = 1,
        multicast_cta_mask: int | None = None,
    ) -> None:
        stmt = Tcgen05Commit(
            mbar=mbar, stage=stage, cta_group=cta_group, multicast_cta_mask=multicast_cta_mask
        )
        self._append(stmt)

    def tcgen05_ld(
        self,
        dst: Tensor | TensorSlice,
        src: Tensor,
        *,
        shape: Tcgen05LdStShape = "32x32b",
        num: Literal[1, 2, 4, 8, 16, 32, 64, 128] = 1,
        row: ScalarValue = 0,
        col: ScalarValue = 0,
    ) -> None:
        # ``src`` is the TMEM tensor handle; the taddr corner is the (row, col) operands.
        if isinstance(dst, Tensor):
            dst = dst[...]
        stmt = Tcgen05Ld(dst=dst, src=src, shape=shape, num=num, row=row, col=col)
        self._append(stmt)

    def tcgen05_wait_ld(self) -> None:
        self._append(Tcgen05WaitLd())

    def tcgen05_st(
        self,
        dst: Tensor,
        src: Tensor | TensorSlice,
        *,
        shape: Tcgen05LdStShape = "32x32b",
        num: Literal[1, 2, 4, 8, 16, 32, 64, 128] = 1,
        row: ScalarValue = 0,
        col: ScalarValue = 0,
    ) -> None:
        # ``dst`` is the TMEM tensor handle; the taddr corner is the (row, col) operands.
        if isinstance(src, Tensor):
            src = src[...]
        stmt = Tcgen05St(dst=dst, src=src, shape=shape, num=num, row=row, col=col)
        self._append(stmt)

    def tcgen05_wait_st(self) -> None:
        self._append(Tcgen05WaitSt())

    def ldmatrix(
        self,
        dst: Tensor | TensorSlice,
        src: Tensor | TensorSlice,
        *,
        shape: MatrixShape = "m8n8",
        num: Literal[1, 2, 4] = 1,
        trans: bool = False,
        dtype: MatrixDType = "b16",
    ) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(src, Tensor):
            src = src[...]
        self._append(LdMatrix(dst=dst, src=src, shape=shape, num=num, trans=trans, dtype=dtype))

    def stmatrix(
        self,
        dst: Tensor | TensorSlice,
        src: Tensor | TensorSlice,
        *,
        shape: MatrixShape = "m8n8",
        num: Literal[1, 2, 4] = 1,
        trans: bool = False,
        dtype: MatrixDType = "b16",
    ) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(src, Tensor):
            src = src[...]
        self._append(StMatrix(dst=dst, src=src, shape=shape, num=num, trans=trans, dtype=dtype))

    def mma_sync(
        self,
        d: Tensor | TensorSlice,
        a: Tensor | TensorSlice,
        b: Tensor | TensorSlice,
        c: Tensor | TensorSlice,
        *,
        m: int = 16,
        n: int = 8,
        k: int = 16,
        ab_dtype: DType = DType.BF16,
    ) -> None:
        """Warp-level SM80 tensor-core MMA: D = A·Bᵀ + C
        (``mma.sync.aligned.m{m}n{n}k{k}.row.col.f32.{abt}.{abt}.f32``). A/B are
        packed-16bit reg fragments of ``ab_dtype`` (bf16 or f16 — the PTX operand
        type); C/D are f32 reg fragments — all in the standard mma warp fragment
        layout (the layout ldmatrix produces)."""
        d = d[...] if isinstance(d, Tensor) else d
        a = a[...] if isinstance(a, Tensor) else a
        b = b[...] if isinstance(b, Tensor) else b
        c = c[...] if isinstance(c, Tensor) else c
        self._append(WarpMma(d=d, a=a, b=b, c=c, m=m, n=n, k=k, ab_dtype=ab_dtype))

    def reg_fill(self, dst: Tensor | TensorSlice, value: RegOperand) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(value, Tensor):
            value = value[...]
        self._append(RegFill(dst=dst, value=value))

    def reg_unary(self, dst: Tensor | TensorSlice, src: RegOperand, *, op: RegUnaryOp) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(src, Tensor):
            src = src[...]
        self._append(RegUnary(dst=dst, src=src, op=op))

    def reg_cvt(
        self,
        dst: Tensor | TensorSlice,
        src: Tensor | TensorSlice,
        *,
        rounding: Literal["rn"] = "rn",
    ) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(src, Tensor):
            src = src[...]
        stmt = RegCvt(dst=dst, src=src, rounding=rounding)
        self._append(stmt)

    def reg_add(
        self,
        dst: Tensor | TensorSlice,
        lhs: RegOperand,
        rhs: RegOperand,
        *,
        rounding: Literal["rn", "rm"] = "rn",
    ) -> None:
        self._append_reg_binary(RegAdd, dst, lhs, rhs, rounding=rounding)

    def reg_sub(
        self,
        dst: Tensor | TensorSlice,
        lhs: RegOperand,
        rhs: RegOperand,
        *,
        rounding: Literal["rn", "rm"] = "rn",
    ) -> None:
        self._append_reg_binary(RegSub, dst, lhs, rhs, rounding=rounding)

    def reg_mul(self, dst: Tensor | TensorSlice, lhs: RegOperand, rhs: RegOperand) -> None:
        self._append_reg_binary(RegMul, dst, lhs, rhs)

    def reg_fma(
        self, dst: Tensor | TensorSlice, a: RegOperand, b: RegOperand, c: RegOperand
    ) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(a, Tensor):
            a = a[...]
        if isinstance(b, Tensor):
            b = b[...]
        if isinstance(c, Tensor):
            c = c[...]
        self._append(RegFma(dst=dst, a=a, b=b, c=c))

    def reg_max(self, dst: Tensor | TensorSlice, lhs: RegOperand, rhs: RegOperand) -> None:
        self._append_reg_binary(RegMax, dst, lhs, rhs)

    def reg_min(self, dst: Tensor | TensorSlice, lhs: RegOperand, rhs: RegOperand) -> None:
        self._append_reg_binary(RegMin, dst, lhs, rhs)

    def reg_bitwise(
        self,
        dst: Tensor | TensorSlice,
        lhs: RegOperand,
        rhs: RegOperand,
        *,
        op: Literal["and", "shl"],
    ) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(lhs, Tensor):
            lhs = lhs[...]
        if isinstance(rhs, Tensor):
            rhs = rhs[...]
        self._append(RegBitwise(dst=dst, lhs=lhs, rhs=rhs, op=op))

    def reg_reduce(self, dst: Tensor | TensorSlice, src: RegOperand, *, op: RegReduceOp) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(src, Tensor):
            src = src[...]
        self._append(RegReduce(dst=dst, src=src, op=op))

    def reg_cond_rescale(
        self,
        dst: Tensor | TensorSlice,
        src: RegOperand,
        scale: RegOperand,
        *,
        threshold: RegOperand = 1.0,
        scope: RegCondScope = "warp",
    ) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(src, Tensor):
            src = src[...]
        if isinstance(scale, Tensor):
            scale = scale[...]
        if isinstance(threshold, Tensor):
            threshold = threshold[...]
        self._append(
            RegCondRescale(dst=dst, src=src, scale=scale, threshold=threshold, scope=scope)
        )

    def reg_softmax_rescale(
        self,
        row_max: Tensor | TensorSlice,
        row_scale: Tensor | TensorSlice,
        row_max_old: RegOperand,
        row_max_new: RegOperand,
        scale_log2: RegOperand,
        *,
        threshold: RegOperand = 8.0,
    ) -> None:
        if isinstance(row_max, Tensor):
            row_max = row_max[...]
        if isinstance(row_scale, Tensor):
            row_scale = row_scale[...]
        if isinstance(row_max_old, Tensor):
            row_max_old = row_max_old[...]
        if isinstance(row_max_new, Tensor):
            row_max_new = row_max_new[...]
        if isinstance(scale_log2, Tensor):
            scale_log2 = scale_log2[...]
        if isinstance(threshold, Tensor):
            threshold = threshold[...]
        self._append(
            RegSoftmaxRescale(
                row_max=row_max,
                row_scale=row_scale,
                row_max_old=row_max_old,
                row_max_new=row_max_new,
                scale_log2=scale_log2,
                threshold=threshold,
            )
        )

    def reg_causal_mask(
        self,
        dst: Tensor | TensorSlice,
        src: RegOperand,
        *,
        query_start: ScalarValue,
        key_start: ScalarValue,
        group_size: int,
        mask_value: RegOperand = -float("inf"),
        swap_qk: bool = False,
    ) -> None:
        """``swap_qk``: fragment orientation. False = forward ``[q-row, kv-col]``;
        True = backward ``[kv-row, q-col]`` (the fa-bwd fragment is transposed —
        k = key_start + row, q = query_start + col/group_size). Both mask k > q."""
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(src, Tensor):
            src = src[...]
        if isinstance(mask_value, Tensor):
            mask_value = mask_value[...]
        self._append(
            RegCausalMask(
                dst=dst,
                src=src,
                query_start=query_start,
                key_start=key_start,
                group_size=group_size,
                mask_value=mask_value,
                swap_qk=swap_qk,
            )
        )

    def reg_combine_int_frac_ex2(
        self, dst: Tensor | TensorSlice, rounded: RegOperand, frac_ex2: RegOperand
    ) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(rounded, Tensor):
            rounded = rounded[...]
        if isinstance(frac_ex2, Tensor):
            frac_ex2 = frac_ex2[...]
        self._append(RegCombineIntFracEx2(dst=dst, rounded=rounded, frac_ex2=frac_ex2))

    def reg_load(self, dst: Tensor | TensorSlice, src: Tensor | TensorSlice) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(src, Tensor):
            src = src[...]
        stmt = RegLoad(dst=dst, src=src)
        self._append(stmt)

    def reg_store(self, dst: Tensor | TensorSlice, src: Tensor | TensorSlice) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(src, Tensor):
            src = src[...]
        stmt = RegStore(dst=dst, src=src)
        self._append(stmt)

    def fence(
        self, *, kind: FenceKind = FenceKind.MEMORY, scope: FenceScope = FenceScope.CTA
    ) -> None:
        stmt = Fence(kind=kind, scope=scope)
        self._append(stmt)

    def cta_sync(self) -> None:
        self._append(CtaSync())

    def wg_sync(self, *, barrier_id: int) -> None:
        self._append(WgSync(barrier_id=barrier_id))

    def named_barrier(self, *, barrier_id: int, num_warps: int) -> None:
        """Named barrier across `num_warps` warps spanning warpgroups —
        `bar.sync barrier_id, num_warps*32` (flashattn NamedBarrierBwdSm100).
        Distinct statements that call this with the same barrier_id rendezvous
        on ONE hardware barrier (count-based completion)."""
        self._append(NamedBarrier(barrier_id=barrier_id, num_warps=num_warps))

    def warp_sync(self) -> None:
        self._append(WarpSync())

    def cluster_sync(self) -> None:
        self._append(ClusterSync())

    def _append(self, stmt: Stmt) -> None:
        self._body_stack[-1].append(stmt)

    def _append_reg_binary(
        self,
        stmt_cls: type[RegAdd | RegSub | RegMul | RegMax | RegMin],
        dst: Tensor | TensorSlice,
        lhs: RegOperand,
        rhs: RegOperand,
        **kwargs,
    ) -> None:
        if isinstance(dst, Tensor):
            dst = dst[...]
        if isinstance(lhs, Tensor):
            lhs = lhs[...]
        if isinstance(rhs, Tensor):
            rhs = rhs[...]
        self._append(stmt_cls(dst=dst, lhs=lhs, rhs=rhs, **kwargs))
