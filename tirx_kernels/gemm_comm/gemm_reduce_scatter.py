# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Complete SM100 TP4 GEMM+ReduceScatter kernel and registry entry."""

from enum import Enum

import tvm
from tvm.backend.cuda.lang import RankAwareGroupMajorTileScheduler
from tvm.ir.type import PointerType, PrimType
from tvm.megakernel.dsl import TileImpl
from tvm.megakernel.transform import (
    EmissionContext,
    ExecutionPlan,
    MegakernelBackend,
    MidBodyPortAction,
    TileEmitter,
)
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.bench import CudaProfiler
from tvm.tirx.layout import TCol, TileLayout, TLane, tid_in_wg


class ProfileEventType(Enum):
    GEMM = 0
    RS = 1
    WAIT = 2
    ACCUM = 3
    PUT = 4
    SIGNAL = 5


event_type_names = ["gemm", "rs", "wait", "accum", "put", "signal"]

REDUCE_SCATTER_HOST_ENTRYPOINT = "runtime.disco.transfer_to_peers_reduce_scatter"
PARTIAL_GEMM_DEVICE_ENTRYPOINT = "test_mma_ss_tma_2sm_persistent"
REDUCE_SUM_DEVICE_ENTRYPOINT = "reduce_sum"


d_type, a_type, b_type = "float16", "float16", "float16"
nbytes = 2
WORLD_SIZE = 4
M, N, K = 16384, 12288, 49152 // WORLD_SIZE
TOTAL_K = K * WORLD_SIZE
LOCAL_M = M // WORLD_SIZE
BLK_M, BLK_N, BLK_K = 128, 256, 64
assert LOCAL_M * WORLD_SIZE == M, "M must be divisible by WORLD_SIZE"
assert LOCAL_M % BLK_M == 0, "LOCAL_M must be divisible by BLK_M"
MMA_M, MMA_N, MMA_K = 256, 256, 16
GROUP_SIZE = 8
SM_COUNT = 148
NUM_THREADS = 32 * 4 * 3
N_COLS = 512
EPI_TILE = 64
PIPE_DEPTH = 4
NUM_CONSUMER = 2
SMEM_SIZE = (
    PIPE_DEPTH * NUM_CONSUMER * BLK_K * BLK_M
    + BLK_K * BLK_N // 2 * PIPE_DEPTH
    + NUM_CONSUMER * BLK_M * EPI_TILE
) * 2 + 1024
TMEM_LD_SIZE = 128
CLUSTER_M, CLUSTER_N = 2, 1
SWIZZLE = 3
TILE_M, TILE_N = BLK_M, BLK_N
cta_group = 2
ldo, sdo = 1, 64

# special parameters for RS
GEMM_SMS = SM_COUNT
N_REPEAT = 15
RS_LOAD_PIPE_DEPTH = 6
BLK_N_RS = 128
BLK_M_RS = 128


# profiling
NUM_GROUPS = 5
PROFILER_BUFFER_SIZE = int(1e7)
PROFILER_WRITE_STRIDE = SM_COUNT * NUM_GROUPS


atomic_add_system_uint64 = """
__forceinline__ __device__ uint64_t atomic_add_system_uint64(uint64_t* addr, uint64_t value) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(addr), value);
}
"""


@T.meta_class
class Semaphore:
    def __init__(self, cnt, buffer):
        self.cnt = cnt
        self.sem = buffer
        self.state = T.alloc_buffer([1], "uint64", scope="local", align=8)

    @T.inline
    def semaphore_wait(self, *coord):
        while 1:
            T.ptx.ld_global_acquire(
                self.state[0], self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord))
            )
            if T.cuda.syncthreads_and(self.state[0] == self.cnt):
                break
            T.cuda.nano_sleep(40)

    @T.inline
    def semaphore_notify(self, tid, *coord):
        # wg is synced
        if tid % 128 == 0:
            T.cuda.func_call(
                "atomic_add_system_uint64",
                self.sem.access_ptr("rw", offset=self.sem.elem_offset_of(coord)),
                T.uint64(1),
                source_code=atomic_add_system_uint64,
            )
        T.cuda.thread_fence()


@T.meta_class
class Pipeline:
    def __init__(
        self,
        shared_buf,
        base_offset,
        pipeline_depth: int,
        pipeline_num: int,
        p_single_cta: bool = False,
        c_single_cta: bool = False,
    ):
        self.pipeline_depth = pipeline_depth
        self.pipeline_num = pipeline_num
        self.mbar_p2c = T.decl_buffer(
            (pipeline_depth, pipeline_num), "uint64", shared_buf, elem_offset=base_offset
        )
        self.mbar_c2p = T.decl_buffer(
            (pipeline_depth, pipeline_num),
            "uint64",
            shared_buf,
            elem_offset=base_offset + pipeline_depth * pipeline_num,
        )
        self.idx = T.local_scalar("int32")
        self.p2c_phase = T.local_scalar("int32")
        self.c2p_phase = T.local_scalar("int32")
        self.p_single_cta = p_single_cta
        self.c_single_cta = c_single_cta

    @T.inline
    def init(self, initializer, p2c_thread_count: int = 1, c2p_thread_count: int = 1):
        self.idx = 0
        self.p2c_phase = 0
        self.c2p_phase = 1
        if initializer:
            for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
                for i in T.serial(0, self.pipeline_depth):
                    for j in T.serial(0, self.pipeline_num):
                        if not self.c_single_cta or cbx == 0:
                            T.ptx.mbarrier.init(self.mbar_p2c.ptr_to([i, j]), p2c_thread_count)
                        if not self.p_single_cta or cbx == 0:
                            T.ptx.mbarrier.init(self.mbar_c2p.ptr_to([i, j]), c2p_thread_count)
        T.ptx.fence.proxy_async("shared::cta")

    @T.inline
    def advance(self):
        self.idx = (self.idx + 1) % self.pipeline_depth
        if self.idx == 0:
            self.p2c_phase = self.p2c_phase ^ 1
            self.c2p_phase = self.c2p_phase ^ 1

    @T.inline
    def producer_wait(self, pipeline_idx):
        for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            if not self.p_single_cta or cbx == 0:
                T.ptx.mbarrier.try_wait(
                    self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), self.c2p_phase
                )

    @T.inline
    def consumer_wait(self, pipeline_idx):
        for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            if not self.c_single_cta or cbx == 0:
                T.ptx.mbarrier.try_wait(
                    self.mbar_p2c.ptr_to([self.idx, pipeline_idx]), self.p2c_phase
                )


class TMA2MMAPipeline(Pipeline):
    @T.inline
    def consumer_release(self, pipeline_idx):
        for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            for tx in T.thread_binding(NUM_THREADS, "threadIdx.x"):
                if tx % 32 == 0:
                    if not self.c_single_cta:
                        T.ptx.tcgen05.commit(self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), 1)
                    elif cbx == 0:
                        T.ptx.tcgen05.commit(
                            self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), 2, cta_mask=3
                        )


class MMA2LDpipeline(Pipeline):
    @T.inline
    def consumer_release(self, pipeline_idx):
        for cbx in T.thread_binding(CLUSTER_M, "clusterCtaIdx.x"):
            if not self.c_single_cta or cbx == 0:
                T.ptx.mbarrier.arrive(
                    self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), remote=0, pred=True
                )


class ReducePipe(Pipeline):
    @T.inline
    def consumer_release(self, pipeline_idx: int):
        T.ptx.mbarrier.arrive(self.mbar_c2p.ptr_to([self.idx, pipeline_idx]))


half8tofloat8 = """
__forceinline__ __device__ void half8tofloat8(void* src_addr, void* dst_addr) {
    half2* source = (half2*) src_addr;
    float2* dest = (float2*) dst_addr;
    for (int i = 0; i < 4; i++) {
        dest[i] = __half22float2(source[i]);
    }
}
"""
float8tohalf8 = """
__forceinline__ __device__ void float8tohalf8(void* src_addr, void* dst_addr) {
    float2* source = (float2*) src_addr;
    half2* dest = (half2*) dst_addr;
    for (int i = 0; i < 4; i++) {
        dest[i] = __float22half2_rn(source[i]);
    }
}
"""


A_layout = T.ComposeLayout(
    3,
    3,
    3,
    T.TileLayout(
        T.S[
            (PIPE_DEPTH, NUM_CONSUMER, BLK_M, 1, 64) : (
                BLK_M * 64 * NUM_CONSUMER,
                BLK_M * 64,
                64,
                BLK_M * 64,
                1,
            )
        ]
    ),
)
B_layout = T.ComposeLayout(
    3,
    3,
    3,
    T.TileLayout(T.S[(PIPE_DEPTH, BLK_N // 2, 1, 64) : (BLK_N // 2 * 64, 64, BLK_N // 2 * 64, 1)]),
)
D_layout = T.ComposeLayout(
    3, 3, 3, T.TileLayout(T.S[(NUM_CONSUMER, BLK_M, EPI_TILE) : (BLK_M * EPI_TILE, EPI_TILE, 1)])
)


class ReduceScatterTileImpl(TileImpl):
    """Launch the existing host-side ReduceScatter transfer stage."""

    def __init__(self, tensor_specs):
        super().__init__()
        self.tensor_specs = dict(tensor_specs)

    def run(self, m_idx, n_idx, k_idx):
        raise RuntimeError("host tiles are executed by GemmCommHostExecutor")


class _PartialReadyPortBackend(MegakernelBackend):
    """Emit the private partial-ready edge at the approved epilogue port."""

    def __init__(self, sem, tid, signal_rank):
        self.sem = sem
        self.tid = tid
        self.signal_rank = signal_rank

    @T.inline
    def _emit_notify(self):
        self.sem.semaphore_notify(self.tid, self.signal_rank)

    def emit_action(self, action, context):
        del context
        if not isinstance(action, MidBodyPortAction):
            raise ValueError(f"unsupported partial-ready action {type(action).__name__}")
        if action.port != "after_store_before_pipeline_advance":
            raise ValueError(f"unsupported partial-ready port {action.port!r}")
        self._emit_notify()


@T.meta_class
class PartialReadyPortProgram:
    """Bind one validated private edge port to the partial GEMM semaphore."""

    def __init__(self, execution_plan: ExecutionPlan | None, sem: Semaphore):
        if execution_plan is None:
            self.context = None
            self.actions = (MidBodyPortAction((), "after_store_before_pipeline_advance"),)
        else:
            region = next(
                region
                for region in execution_plan.device_regions
                if region.name == "partial_gemm_device"
            )
            program = next(
                program for program in region.tile_programs if program.tile.name == "partial_gemm"
            )
            self.context = EmissionContext(execution_plan, region, "tile_action", program.tile)
            self.actions = tuple(
                action for action in program.actions if isinstance(action, MidBodyPortAction)
            )
        self.sem = sem

    @T.inline
    def emit(self, tid, signal_rank):
        backend = T.meta_var(_PartialReadyPortBackend(self.sem, tid, signal_rank))
        emitter = T.meta_var(TileEmitter(backend))
        emitter.emit_program(self.context, self.actions)


class PartialGemmTileImpl(TileImpl):
    """Execute one rank-local GEMM cluster across the persistent warp roles."""

    def __init__(self, tensor_specs=None):
        super().__init__()
        self.tensor_specs = {} if tensor_specs is None else dict(tensor_specs)
        self._resources_initialized = False

    def _attach_resources(self, **context) -> None:
        for name, value in context.items():
            setattr(self, name, value)

    def _mark_resources_initialized(self) -> None:
        self._resources_initialized = True

    @T.inline
    def host_init(self, A, B, gemm_out):
        """Create and encode TileImpl-owned TMA tensor-map descriptors."""

        A_tensor_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        B_tensor_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        C_tensor_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            A_tensor_map,
            "float16",
            2,
            A.data,
            K,
            M,
            K * 2,
            BLK_K,
            BLK_M,
            1,
            1,
            0,
            3,
            0,
            0,
        )
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            B_tensor_map,
            "float16",
            2,
            B.data,
            K,
            N,
            K * 2,
            BLK_K,
            BLK_N // 2,
            1,
            1,
            0,
            3,
            0,
            0,
        )
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            C_tensor_map,
            "float16",
            2,
            gemm_out.data,
            N,
            M,
            N * 2,
            EPI_TILE,
            BLK_M,
            1,
            1,
            0,
            3,
            0,
            0,
        )
        self._attach_resources(
            A_tensor_map=A_tensor_map, B_tensor_map=B_tensor_map, C_tensor_map=C_tensor_map
        )

    @T.inline
    def init_storage(self, buf, sem, cbx, wg_id, warp_id, lane_id, tid):
        """Create partial-GEMM smem, registers, descriptors, and pipelines."""

        tmem_addr = T.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)
        A_smem = T.decl_buffer(
            (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K),
            a_type,
            buf.data,
            elem_offset=512,
            layout=A_layout,
        )
        B_smem = T.decl_buffer(
            (PIPE_DEPTH, BLK_N // 2, BLK_K),
            b_type,
            buf.data,
            elem_offset=512 + BLK_K * BLK_M * NUM_CONSUMER * PIPE_DEPTH,
            layout=B_layout,
        )
        C_smem = T.decl_buffer(
            (NUM_CONSUMER, BLK_M, EPI_TILE),
            d_type,
            buf.data,
            elem_offset=512
            + BLK_K * BLK_M * NUM_CONSUMER * PIPE_DEPTH
            + BLK_K * BLK_N // 2 * PIPE_DEPTH,
            layout=D_layout,
        )
        reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
        reg_wg = reg.view(
            128, TMEM_LD_SIZE, layout=TileLayout(T.S[(128, TMEM_LD_SIZE) : (1 @ tid_in_wg, 1)])
        )
        reg_fp16 = T.alloc_buffer((BLK_N,), d_type, scope="local")
        descA = T.local_scalar("uint64")
        descB = T.local_scalar("uint64")
        descI = T.local_scalar("uint32")
        base_desc_A = T.local_scalar("uint64")
        base_desc_B = T.local_scalar("uint64")
        tma2mma_pipe = T.meta_var(
            TMA2MMAPipeline(buf.data, 1, PIPE_DEPTH, 1, p_single_cta=False, c_single_cta=True)
        )
        mma2ld_pipe = T.meta_var(
            MMA2LDpipeline(
                buf.data, 1 + PIPE_DEPTH * 2, 1, NUM_CONSUMER, p_single_cta=True, c_single_cta=False
            )
        )
        mma2ld_pipe.init(tid == 0, c2p_thread_count=128 * 2, p2c_thread_count=2)
        tma2mma_pipe.init(tid == 0, c2p_thread_count=NUM_CONSUMER)
        ptr: T.let[T.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = T.reinterpret(
            "handle", T.ptx.map_shared_rank(tma2mma_pipe.mbar_p2c.ptr_to([0, 0]), 0)
        )
        tma_finished = T.decl_buffer([PIPE_DEPTH], "uint64", data=ptr, scope="shared")
        self._attach_resources(
            A_smem=A_smem,
            B_smem=B_smem,
            C_smem=C_smem,
            reg=reg,
            reg_wg=reg_wg,
            reg_fp16=reg_fp16,
            descA=descA,
            descB=descB,
            descI=descI,
            base_desc_A=base_desc_A,
            base_desc_B=base_desc_B,
            tma2mma_pipe=tma2mma_pipe,
            mma2ld_pipe=mma2ld_pipe,
            tma_finished=tma_finished,
            tmem_addr=tmem_addr,
            sem=sem,
            cbx=cbx,
            wg_id=wg_id,
            warp_id=warp_id,
            lane_id=lane_id,
            tid=tid,
        )

    @T.inline
    def device_init(self):
        """Allocate tensor memory and initialize the MMA instruction descriptor."""

        if (self.wg_id == 0) & (self.warp_id == 0):
            T.ptx.tcgen05.alloc(T.address_of(self.tmem_addr), n_cols=N_COLS, cta_group=cta_group)
        T.ptx.tcgen05.encode_instr_descriptor(
            T.address_of(self.descI),
            d_dtype="float32",
            a_dtype=a_type,
            b_dtype=b_type,
            M=MMA_M,
            N=MMA_N,
            K=MMA_K,
            trans_a=False,
            trans_b=False,
            n_cta_groups=cta_group,
        )
        T.cuda.cta_sync()
        T.cuda.trap_when_assert_failed(self.tmem_addr == 0)
        tmem = T.decl_buffer(
            (128, N_COLS),
            "float32",
            scope="tmem",
            allocated_addr=0,
            layout=TileLayout(T.S[(128, N_COLS) : (1 @ TLane, 1 @ TCol)]),
        )
        self._attach_resources(tmem=tmem)
        self._mark_resources_initialized()

    @T.inline
    def release_tmem(self):
        if (self.wg_id == 0) & (self.warp_id == 0):
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
            T.ptx.tcgen05.dealloc(self.tmem_addr, n_cols=N_COLS, cta_group=cta_group)

    def bind_edge_port(self, execution_plan: ExecutionPlan | None, sem: Semaphore) -> None:
        self.partial_ready_port = PartialReadyPortProgram(execution_plan, sem)

    def build_module(self, execution_plan: ExecutionPlan):
        """Build both device regions attached to this execution plan."""

        reduce_impls = [tile.impl for tile in execution_plan.kernel.tiles if tile.name == "reduce"]
        if len(reduce_impls) != 1:
            raise ValueError("GEMM+ReduceScatter requires exactly one reduce device region")
        return build_kernel(self, reduce_impls[0], execution_plan=execution_plan)

    def set_emission_role(self, role: str) -> None:
        self.role = role

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        if not self._resources_initialized:
            raise RuntimeError("PartialGemmTileImpl resources are not initialized")

        if self.role == "load":
            for ko in range(K // BLK_K):
                if self.lane_id == 0:
                    self.tma2mma_pipe.producer_wait(0)
                    T.ptx.cp_async.bulk.tensor.g2s_cluster(
                        2,
                        self.A_smem.ptr_to([self.tma2mma_pipe.idx, 0, 0, 0]),
                        self.tma_finished.ptr_to([self.tma2mma_pipe.idx]),
                        T.address_of(self.A_tensor_map),
                        0,
                        2,
                        "",
                        ko * BLK_K,
                        (m_idx * 4 + self.cbx) * BLK_M,
                    )
                    T.ptx.cp_async.bulk.tensor.g2s_cluster(
                        2,
                        self.A_smem.ptr_to([self.tma2mma_pipe.idx, 1, 0, 0]),
                        self.tma_finished.ptr_to([self.tma2mma_pipe.idx]),
                        T.address_of(self.A_tensor_map),
                        0,
                        2,
                        "",
                        ko * BLK_K,
                        (m_idx * 4 + 2 + self.cbx) * BLK_M,
                    )
                    T.ptx.cp_async.bulk.tensor.g2s_cluster(
                        2,
                        self.B_smem.ptr_to([self.tma2mma_pipe.idx, 0, 0]),
                        self.tma_finished.ptr_to([self.tma2mma_pipe.idx]),
                        T.address_of(self.B_tensor_map),
                        0,
                        2,
                        "",
                        ko * BLK_K,
                        n_idx * BLK_N + self.cbx * BLK_N // 2,
                    )
                    if self.cbx == 0:
                        self.tma_finished_arrive()
                    self.tma2mma_pipe.advance()
        elif self.role == "mma":
            if self.lane_id == 0 and self.cbx == 0:
                self.mma2ld_pipe.producer_wait(self.warp_id)
                for ko in T.serial(0, K // BLK_K):
                    self.tma2mma_pipe.consumer_wait(0)
                    T.ptx.tcgen05.encode_matrix_descriptor(
                        T.address_of(self.base_desc_A),
                        self.A_smem.ptr_to([self.tma2mma_pipe.idx, self.warp_id, 0, 0]),
                        ldo=ldo,
                        sdo=sdo,
                        swizzle=SWIZZLE,
                    )
                    T.ptx.tcgen05.encode_matrix_descriptor(
                        T.address_of(self.base_desc_B),
                        self.B_smem.ptr_to([self.tma2mma_pipe.idx, 0, 0]),
                        ldo=ldo,
                        sdo=sdo,
                        swizzle=SWIZZLE,
                    )
                    for ki in range(BLK_K // MMA_K):
                        self.descA = self.base_desc_A + ((ki * MMA_K * 2) >> 0x4)
                        self.descB = self.base_desc_B + ((ki * MMA_K * 2) >> 0x4)
                        if ki == 0 and ko == 0:
                            T.ptx.tcgen05.mma(
                                self.tmem_addr + self.warp_id * MMA_N,
                                self.descA,
                                self.descB,
                                self.descI,
                                d_dtype="float32",
                                a_dtype=a_type,
                                b_dtype=b_type,
                                use_a_tmem=False,
                                cta_group=cta_group,
                                enable_input_d=False,
                            )
                        else:
                            T.ptx.tcgen05.mma(
                                self.tmem_addr + self.warp_id * MMA_N,
                                self.descA,
                                self.descB,
                                self.descI,
                                d_dtype="float32",
                                a_dtype=a_type,
                                b_dtype=b_type,
                                use_a_tmem=False,
                                cta_group=cta_group,
                                enable_input_d=True,
                            )
                    self.tma2mma_pipe.consumer_release(0)
                    self.tma2mma_pipe.advance()
                T.ptx.tcgen05.commit(
                    self.mma2ld_pipe.mbar_p2c.ptr_to([0, 0]), cta_group=2, cta_mask=3
                )
                self.mma2ld_pipe.advance()
        elif self.role == "epilogue":
            self.mma2ld_pipe.consumer_wait(0)
            for i in range(BLK_N // TMEM_LD_SIZE):
                col_st = T.meta_var(self.wg_id * MMA_N + i * TMEM_LD_SIZE)
                Tx.wg.copy_async(self.reg_wg[:, :], self.tmem[:, col_st : col_st + TMEM_LD_SIZE])
                Tx.cast(self.reg_fp16[i * TMEM_LD_SIZE : (i + 1) * TMEM_LD_SIZE], self.reg[:])
            self.mma2ld_pipe.consumer_release(self.wg_id)
            for i in range(BLK_N // EPI_TILE):
                Tx.copy(
                    self.C_smem[self.wg_id, self.warp_id * 32 + self.lane_id, :],
                    self.reg_fp16[i * EPI_TILE : (i + 1) * EPI_TILE],
                )
                T.cuda.warpgroup_sync(self.wg_id + 1)
                T.ptx.fence.proxy_async("shared::cta")
                if self.lane_id == 0 and self.warp_id == 0:
                    T.ptx.cp_async.bulk.tensor.s2g(
                        2,
                        self.C_smem.ptr_to([self.wg_id, 0, 0]),
                        T.address_of(self.C_tensor_map),
                        "",
                        n_idx * BLK_N + i * EPI_TILE,
                        (m_idx * 4 + self.wg_id * 2 + self.cbx) * BLK_M,
                    )
                    T.ptx.cp_async.bulk.commit_group()
                    T.ptx.cp_async.bulk.wait_group(0)
                T.cuda.warpgroup_sync(self.wg_id + 1)
            comm_m_idx = T.meta_var(m_idx * 4 + self.wg_id * 2 + self.cbx)
            signal_rank = T.meta_var(comm_m_idx // (LOCAL_M // BLK_M))
            self.partial_ready_port.emit(self.tid, signal_rank)
            self.mma2ld_pipe.advance()
        else:
            raise ValueError(f"unsupported partial GEMM role: {self.role!r}")

    @T.inline
    def tma_finished_arrive(self):
        T.ptx.mbarrier.arrive.expect_tx(
            self.tma_finished.ptr_to([self.tma2mma_pipe.idx]),
            (BLK_K * BLK_M * 2 * 2 + BLK_K * BLK_N) * 2,
        )

    @T.inline
    def finalize(self):
        for i in range(PIPE_DEPTH):
            if self.lane_id == 0:
                self.tma2mma_pipe.producer_wait(0)
                self.tma2mma_pipe.advance()


class ReduceSumTileImpl(TileImpl):
    """Reduce one local output tile across all source ranks."""

    def __init__(self, tensor_specs=None):
        super().__init__()
        self.tensor_specs = {} if tensor_specs is None else dict(tensor_specs)
        self._resources_initialized = False

    def _attach_resources(self, **context) -> None:
        for name, value in context.items():
            setattr(self, name, value)

    def _mark_resources_initialized(self) -> None:
        self._resources_initialized = True

    @T.inline
    def host_init(self, staging_buffer, out):
        """Create and encode reduction tensor-map descriptors."""

        src_tensor_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        dst_tensor_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            src_tensor_map,
            "float16",
            3,
            staging_buffer.data,
            N,
            LOCAL_M,
            WORLD_SIZE,
            N * 2,
            LOCAL_M * N * 2,
            BLK_N_RS,
            BLK_M_RS,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
        )
        T.call_packed(
            "runtime.cuTensorMapEncodeTiled",
            dst_tensor_map,
            "float16",
            2,
            out.data,
            N,
            LOCAL_M,
            N * 2,
            BLK_N_RS,
            BLK_M_RS,
            1,
            1,
            0,
            0,
            0,
            0,
        )
        self._attach_resources(src_tensor_map=src_tensor_map, dst_tensor_map=dst_tensor_map)

    @T.inline
    def init_storage(self, buf, wg_id, warp_id, lane_id, tid_in_wg):
        """Create reduction smem, register storage, and its load pipeline."""

        load_pipe = T.meta_var(
            ReducePipe(buf.data, 0, RS_LOAD_PIPE_DEPTH, 1, p_single_cta=False, c_single_cta=False)
        )
        input_smem = T.decl_buffer(
            (RS_LOAD_PIPE_DEPTH, BLK_M_RS, BLK_N_RS), d_type, buf.data, elem_offset=512
        )
        output_smem = T.decl_buffer(
            (BLK_M_RS, BLK_N_RS),
            d_type,
            buf.data,
            elem_offset=512 + RS_LOAD_PIPE_DEPTH * BLK_M_RS * BLK_N_RS,
        )
        reg_fp16 = T.alloc_buffer((8,), "float16", scope="local")
        reg_fp32_tmp = T.alloc_buffer((8,), "float32", scope="local")
        reg_fp32 = T.alloc_buffer((BLK_M_RS * BLK_N_RS // 8 // 128, 8), "float32", scope="local")
        self._attach_resources(
            load_pipe=load_pipe,
            input_smem=input_smem,
            output_smem=output_smem,
            reg_fp16=reg_fp16,
            reg_fp32_tmp=reg_fp32_tmp,
            reg_fp32=reg_fp32,
            wg_id=wg_id,
            warp_id=warp_id,
            lane_id=lane_id,
            tid_in_wg=tid_in_wg,
        )

    @T.inline
    def device_init(self, tid):
        self.load_pipe.init(tid == 0, c2p_thread_count=128)
        self._mark_resources_initialized()

    def set_emission_role(self, role: str) -> None:
        self.role = role

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        if not self._resources_initialized:
            raise RuntimeError("ReduceSumTileImpl resources are not initialized")

        if self.role == "load":
            if self.lane_id == 0:
                for i in range(WORLD_SIZE):
                    self.load_pipe.producer_wait(0)
                    T.ptx.cp_async.bulk.tensor.g2s_cta(
                        3,
                        self.input_smem.ptr_to([self.load_pipe.idx, 0, 0]),
                        self.load_pipe.mbar_p2c.ptr_to([self.load_pipe.idx, 0]),
                        T.address_of(self.src_tensor_map),
                        1,
                        "",
                        n_idx * BLK_N_RS,
                        m_idx * BLK_M_RS,
                        i,
                    )
                    T.ptx.mbarrier.arrive.expect_tx(
                        self.load_pipe.mbar_p2c.ptr_to([self.load_pipe.idx, 0]),
                        BLK_M_RS * BLK_N_RS * 2,
                    )
                    self.load_pipe.advance()
        elif self.role == "reduce":
            for rank in range(WORLD_SIZE):
                self.load_pipe.consumer_wait(0)
                for i in range(BLK_M_RS * BLK_N_RS // 8 // 128):
                    m_in_smem = T.meta_var((i * 128 + self.tid_in_wg) // (BLK_N_RS // 8))
                    n_in_smem = T.meta_var((i * 128 + self.tid_in_wg) % (BLK_N_RS // 8))
                    for j in T.vectorized(8):
                        self.reg_fp16[j] = self.input_smem[
                            self.load_pipe.idx, m_in_smem, n_in_smem * 8 + j
                        ]
                    if rank > 0:
                        T.cuda.half8tofloat8(self.reg_fp16.data, self.reg_fp32_tmp.data)
                        for j in T.vectorized(8):
                            self.reg_fp32[i, j] += self.reg_fp32_tmp[j]
                    else:
                        T.cuda.half8tofloat8(self.reg_fp16.data, self.reg_fp32.ptr_to([i, 0]))
                self.load_pipe.consumer_release(0)
                self.load_pipe.advance()
            for i in range(BLK_M_RS * BLK_N_RS // 8 // 128):
                m_in_smem = T.meta_var((i * 128 + self.tid_in_wg) // (BLK_N_RS // 8))
                n_in_smem = T.meta_var((i * 128 + self.tid_in_wg) % (BLK_N_RS // 8))
                T.cuda.float8tohalf8(self.reg_fp32.ptr_to([i, 0]), self.reg_fp16.data)
                for j in T.vectorized(8):
                    self.output_smem[m_in_smem, n_in_smem * 8 + j] = self.reg_fp16[j]
            T.cuda.warpgroup_sync(1)
            T.ptx.fence.proxy_async("shared::cta")
            if self.tid_in_wg == 0:
                T.ptx.cp_async.bulk.tensor.s2g(
                    2,
                    self.output_smem.ptr_to([0, 0]),
                    T.address_of(self.dst_tensor_map),
                    "",
                    n_idx * BLK_N_RS,
                    m_idx * BLK_M_RS,
                )
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.warpgroup_sync(1)
        else:
            raise ValueError(f"unsupported reduce role: {self.role!r}")


# fmt: off
def _build_partial_gemm(
    partial_gemm_impl: PartialGemmTileImpl, execution_plan: ExecutionPlan | None
):
    @T.prim_func
    def test_mma_ss_tma_2sm_persistent(A: T.Buffer((M, K), a_type), B: T.Buffer((N, K), b_type), gemm_out: T.Buffer((M, N), d_type),
                                    semaphore: T.Buffer((WORLD_SIZE, ), "uint64"),
                                    out: T.Buffer((LOCAL_M, N), d_type), profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64")):
        partial_gemm_impl.host_init(A, B, gemm_out)
        T.device_entry()
        cbx, cby = T.cta_id_in_cluster([CLUSTER_M, CLUSTER_N])
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([NUM_CONSUMER+1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        tid = T.thread_id([NUM_THREADS])
        rank = T.nvshmem.my_pe()
        sem = T.meta_var(Semaphore(cnt=WORLD_SIZE, buffer=semaphore))
        buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
        profiler = T.meta_var(
            CudaProfiler(
                profiler_buffer,
                write_stride=PROFILER_WRITE_STRIDE,
                num_groups=NUM_GROUPS,
            )
        )
        profiler.init(0)
        if bx < GEMM_SMS:
            profiler.start(ProfileEventType.GEMM, lane_id == 0)
            partial_gemm_impl.init_storage(buf, sem, cbx, wg_id, warp_id, lane_id, tid)
            m_clusters = T.meta_var((M + BLK_M - 1) // BLK_M // CLUSTER_M // NUM_CONSUMER)
            n_clusters = T.meta_var((N + BLK_N - 1) // BLK_N // CLUSTER_N)
            gemm_tile_scheduler = T.meta_var(RankAwareGroupMajorTileScheduler("gemm_tile_scheduler", m_clusters, n_clusters, GROUP_SIZE, WORLD_SIZE))
            gemm_tile_scheduler.init(bx//2)
            partial_gemm_impl.device_init()
            partial_gemm_impl.bind_edge_port(execution_plan, sem)
            # reset RF
            if wg_id == NUM_CONSUMER:
                T.ptx.setmaxnreg(False, 56)
                if warp_id == 3:
                    partial_gemm_impl.set_emission_role("load")
                    while gemm_tile_scheduler.valid():
                        m_idx = T.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                        n_idx = T.meta_var(gemm_tile_scheduler.n_idx)
                        partial_gemm_impl.run(m_idx, n_idx, 0)
                        gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
                    partial_gemm_impl.finalize()
                elif warp_id < NUM_CONSUMER:
                    partial_gemm_impl.set_emission_role("mma")
                    while gemm_tile_scheduler.valid():
                        m_idx = T.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                        n_idx = T.meta_var(gemm_tile_scheduler.n_idx)
                        partial_gemm_impl.run(m_idx, n_idx, 0)
                        gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
            if wg_id < NUM_CONSUMER:
                T.ptx.setmaxnreg(True, 224)
                partial_gemm_impl.set_emission_role("epilogue")
                while gemm_tile_scheduler.valid():
                    m_idx = T.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                    n_idx = T.meta_var(gemm_tile_scheduler.n_idx)
                    partial_gemm_impl.run(m_idx, n_idx, 0)
                    gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
            # dealloc TMEM
            partial_gemm_impl.release_tmem()
            profiler.end(ProfileEventType.GEMM, lane_id == 0)

    return test_mma_ss_tma_2sm_persistent


def _build_reduce_sum(reduce_sum_impl: ReduceSumTileImpl):
    @T.prim_func
    def reduce_sum(
        staging_buffer: T.Buffer((WORLD_SIZE, LOCAL_M, N), "float16"),
        out: T.Buffer((LOCAL_M, N), d_type),
    ):
        reduce_sum_impl.host_init(staging_buffer, out)
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([2])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        tid_in_wg = T.thread_id_in_wg([128])
        tid = T.thread_id([256])
        buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
        reduce_sum_impl.init_storage(buf, wg_id, warp_id, lane_id, tid_in_wg)
        iter = T.local_scalar("int32")
        iter = 0
        reduce_sum_impl.device_init(tid)
        tile_id = T.meta_var(iter * SM_COUNT + bx)
        T.tvm_storage_sync("shared")
        if warp_id == 0 and wg_id == 0:
            reduce_sum_impl.set_emission_role("load")
            while tile_id < LOCAL_M // BLK_M_RS * N // BLK_N_RS:
                m_idx = T.meta_var(tile_id // (N // BLK_N_RS))
                n_idx = T.meta_var(tile_id % (N // BLK_N_RS))
                reduce_sum_impl.run(m_idx, n_idx, 0)
                iter += 1
        elif wg_id == 1:
            reduce_sum_impl.set_emission_role("reduce")
            while tile_id < LOCAL_M // BLK_M_RS * N // BLK_N_RS:
                m_idx = T.meta_var(tile_id // (N // BLK_N_RS))
                n_idx = T.meta_var(tile_id % (N // BLK_N_RS))
                reduce_sum_impl.run(m_idx, n_idx, 0)
                iter += 1

    return reduce_sum
# fmt: on


def build_kernel(
    partial_gemm_impl: PartialGemmTileImpl | None = None,
    reduce_sum_impl: ReduceSumTileImpl | None = None,
    execution_plan: ExecutionPlan | None = None,
):
    """Build the existing IRModule with the selected concrete tile implementations."""

    partial_gemm_impl = partial_gemm_impl or PartialGemmTileImpl()
    reduce_sum_impl = reduce_sum_impl or ReduceSumTileImpl()
    return tvm.IRModule(
        {
            "test_mma_ss_tma_2sm_persistent": _build_partial_gemm(
                partial_gemm_impl, execution_plan
            ),
            "reduce_sum": _build_reduce_sum(reduce_sum_impl),
        }
    )


ReduceScatter = build_kernel()


# Runtime orchestration is kept separate from the complete device kernels above.
# Import it only after every constant and TIRx definition is available so the
# DSL modules can refer back to this first-class kernel module without a cycle.
from ._gemm_reduce_scatter_runner import (  # noqa: E402
    CONFIGS,
    KERNEL_META,
    get_kernel,
    prepare_data,
    run_bench,
    run_test,
)

__all__ = [
    "CONFIGS",
    "KERNEL_META",
    "PartialGemmTileImpl",
    "ReduceScatter",
    "ReduceScatterTileImpl",
    "ReduceSumTileImpl",
    "build_kernel",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
