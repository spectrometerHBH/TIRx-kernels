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

"""Complete SM100 TP4 AllGather+GEMM kernel and registry entry."""

from enum import Enum

import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.megakernel.dsl import TileImpl
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.tirx import tile as Tx
from tvm.tirx.bench import CudaProfiler
from tvm.tirx.layout import TCol, TileLayout, TLane, tid_in_wg


class TaskType(Enum):
    GEMM = 0
    AG = 1


class ProfileEventType(Enum):
    GEMM = 0
    AG = 1
    FETCH = 2


event_type_names = ["gemm", "ag", "fetch"]

# M, N, K = 16384, 49152, 12288
M, N, K = 8192, 8192 * 8, 8192

M_CLUSTER = 2
N_CLUSTER = 1
WG_NUMBER = 3
WARP_NUMBER = 4
NUM_CONSUMER = 2
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER
SM_NUMBER = 148

PIPELINE_DEPTH = 4

F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16

d_type, a_type, b_type = "float16", "float16", "float16"
WORLD_SIZE = 4
LOCAL_M = M // WORLD_SIZE
LOCAL_N = N // WORLD_SIZE
BLK_M, BLK_N, BLK_K = 128, 128, 64
assert LOCAL_M * WORLD_SIZE == M, "M must be divisible by WORLD_SIZE"
assert LOCAL_M % BLK_M == 0, "LOCAL_M must be divisible by BLK_M"
assert LOCAL_N * WORLD_SIZE == N, "N must be divisible by WORLD_SIZE"
assert LOCAL_N % BLK_N == 0, "LOCAL_N must be divisible by BLK_N"

MMA_M, MMA_N, MMA_K = 256, 256, 16
EPI_TILE = 64
SWIZZLE = 3
SMEM_SIZE = (
    PIPELINE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K * F16_BYTES
    + PIPELINE_DEPTH * BLK_N * BLK_K * F16_BYTES
    + NUM_CONSUMER * BLK_M * EPI_TILE * F16_BYTES
    + 1024
)
assert SMEM_SIZE <= 232448

TMEM_LD_SIZE = 64
N_COLS = 512
CTA_GROUP = 2

PIPE_CYCLE = (K // BLK_K) // PIPELINE_DEPTH
PIPE_REMAIN_NUM = (K // BLK_K) % PIPELINE_DEPTH
assert PIPELINE_DEPTH == 4

GROUP_SIZE = min(8, LOCAL_M // (BLK_M * NUM_CONSUMER * CTA_GROUP))
assert M % (NUM_CONSUMER * BLK_M * CTA_GROUP) == 0
assert N % (BLK_N * CTA_GROUP) == 0
GEMM_M_CLUSTERS = M // (NUM_CONSUMER * BLK_M * CTA_GROUP)  # gemm tile m: 512
GEMM_N_CLUSTERS = LOCAL_N // (BLK_N * CTA_GROUP)  # gemm tile n: 256
LOCAL_GEMM_M_CLUSTERS = GEMM_M_CLUSTERS // WORLD_SIZE

# dyn scheduling
CAPACITY = 2048
TASK_IDX_LEN = 2
ENABLE_WARP_BROADCAST = False
C2P_THREAD_COUNT = 12 * 2 if ENABLE_WARP_BROADCAST else NUM_THREADS * 2

# profiling
WARMUP_ITERS = 5
# WARMUP_ITERS = 0
TOTAL_ITERS = 30
# TOTAL_ITERS = 1

PROFILER_ON = False
NUM_GROUPS = 13
PROFILER_BUFFER_SIZE = int(1e7)
PROFILER_WRITE_STRIDE = SM_NUMBER * NUM_GROUPS
CUDA_EVENT_PROFILER = False
if CUDA_EVENT_PROFILER:
    PROFILER_ON = False
VALIDATE = True


pack_values = """
__forceinline__ __device__ void pack_values(int32_t rem, int32_t task_type, int32_t task_idx0, int32_t task_idx1, uint64_t* dst_addr) {
    asm volatile("st.shared::cluster.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(dst_addr), "r"(rem), "r"(task_type), "r"(task_idx0), "r"(task_idx1)
                 : "memory");
}
"""

unpack_values = """
__forceinline__ __device__ void unpack_values(uint64_t* src_addr, int32_t* rem, int32_t* task_type, int32_t* task_idx0, int32_t* task_idx1) {
    asm volatile("ld.shared::cluster.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(*rem), "=r"(*task_type), "=r"(*task_idx0), "=r"(*task_idx1)
                 : "l"(src_addr)
                 : "memory");

}
"""

semaphore_notify_remote = """
__forceinline__ __device__ uint64_t semaphore_notify_remote(int32_t signal_rank, uint64_t* addr, uint64_t signal_value) {
    auto dst_addr = reinterpret_cast<unsigned long long*>(nvshmem_ptr(addr, signal_rank));
    return atomicAdd_system(dst_addr, signal_value);
}
"""

enqueue_remote = """
__forceinline__ __device__ void enqueue_remote(int32_t* task_types, int32_t* task_idxs, int32_t* tail, int32_t mask,
                                               int32_t signal_rank, int32_t task_type, int32_t task_idx0, int32_t task_idx1) {
    int32_t* remote_task_types = (int32_t*)nvshmem_ptr(task_types, signal_rank);
    int32_t* remote_task_idxs = (int32_t*)nvshmem_ptr(task_idxs, signal_rank);
    int32_t* remote_tail = (int32_t*)nvshmem_ptr(tail, signal_rank);
    int32_t tail_r = atomicAdd(&(remote_tail[0]), 1);
    int32_t masked_pos = tail_r & mask;
    remote_task_types[masked_pos] = task_type;
    remote_task_idxs[masked_pos * 2] = task_idx0;
    remote_task_idxs[masked_pos * 2 + 1] = task_idx1;
    __threadfence();
}
"""

ld_global_acquire = """
__forceinline__ __device__ int32_t ld_global_acquire(int32_t* addr) {
  int32_t res;
  asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(res) : "l"(addr));
  return res;
}
"""

while_ld_global_acquire = """
__forceinline__ __device__ int32_t while_ld_global_acquire(int32_t* addr) {
  int32_t res;
  asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(res) : "l"(addr));
  while (res < 0) {
    __nanosleep(40);
    asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(res) : "l"(addr));
  }
  return res;
}
"""

warp_broadcast = """
__forceinline__ __device__ void warp_broadcast(int32_t* rem, int32_t* task_type, int32_t* task_idx0, int32_t* task_idx1) {{
    *rem = __shfl_sync(0xFFFFFFFF, *rem, 0);
    *task_type = __shfl_sync(0xFFFFFFFF, *task_type, 0);
    *task_idx0 = __shfl_sync(0xFFFFFFFF, *task_idx0, 0);
    *task_idx1 = __shfl_sync(0xFFFFFFFF, *task_idx1, 0);
}}
"""


@T.meta_class
class Barriers:
    def __init__(self, shared_buffer_base, shared_buffer_offs, pipe_depth, pipe_width, is_p2c):
        self.mbar: tvm.tir.Buffer = T.decl_buffer(
            (pipe_depth, pipe_width), "uint64", shared_buffer_base, elem_offset=shared_buffer_offs
        )
        self.init_phase = 0 if is_p2c else 1
        self.pipe_depth = pipe_depth
        self.pipe_width = pipe_width

    @T.inline
    def init(self, threads_num_wait, initializer):
        if initializer:
            for i in T.serial(self.pipe_depth):
                for j in T.serial(self.pipe_width):
                    T.ptx.mbarrier.init(self.mbar.ptr_to([i, j]), threads_num_wait)

    @T.inline
    def wait(self, idx_d, idx_w, phase):
        T.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx_d, idx_w]), self.init_phase ^ phase)


class BarTMA2MMA(Barriers):
    @T.inline
    def arrive(self, idx, expected_bytes):
        T.ptx.mbarrier.arrive.expect_tx(self.mbar.ptr_to([idx, 0]), expected_bytes)

    @T.inline
    def arrive_only(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([idx, 0]))


class BarMMA2LD(Barriers):
    @T.inline
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([0, idx]), cta_group=CTA_GROUP, cta_mask=3)


class BarMMA2TMA(Barriers):
    @T.inline
    def arrive(self, idx):
        T.ptx.tcgen05.commit(self.mbar.ptr_to([idx, 0]), cta_group=CTA_GROUP, cta_mask=3)


class BarLD2MMA(Barriers):
    @T.inline
    def arrive(self, idx):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([0, idx]), remote=0, pred=True)


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
            for cbx in T.thread_binding(M_CLUSTER, "clusterCtaIdx.x"):
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
        for cbx in T.thread_binding(M_CLUSTER, "clusterCtaIdx.x"):
            if not self.p_single_cta or cbx == 0:
                T.ptx.mbarrier.try_wait(
                    self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), self.c2p_phase
                )

    @T.inline
    def consumer_wait(self, pipeline_idx):
        for cbx in T.thread_binding(M_CLUSTER, "clusterCtaIdx.x"):
            if not self.c_single_cta or cbx == 0:
                T.ptx.mbarrier.try_wait(
                    self.mbar_p2c.ptr_to([self.idx, pipeline_idx]), self.p2c_phase
                )


def int_var(name: str, scope="local", dtype="int32", align=4):
    buf = T.alloc_buffer([1], dtype, scope=scope, align=align)
    return buf


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
            if self.state[0] == self.cnt:
                break
            T.cuda.nano_sleep(40)


@T.meta_class
class MPMCQueue:
    def __init__(
        self,
        capacity: int,
        task_types: T.Buffer,
        task_idxs: T.Buffer,
        head: T.Buffer,
        tail: T.Buffer,
        num_tot_tasks: int,
    ):
        if capacity & (capacity - 1):
            raise ValueError("capacity must be a power-of-two")
        self.capacity = capacity
        self.mask = capacity - 1
        self.task_types = task_types
        self.task_idxs = task_idxs
        self.head = head
        self.tail = tail
        self.head_r = int_var("head_r")
        self.tail_r = int_var("tail_r")
        self.pos = int_var("pos")
        self.masked_pos = int_var("masked_pos")
        self.num_tot_tasks = num_tot_tasks

    @T.inline
    def enqueue(self, signal_rank: int, task_type: int, *task_idx: int):
        T.cuda.func_call(
            "enqueue_remote",
            self.task_types.ptr_to([0]),
            self.task_idxs.ptr_to([0, 0]),
            self.tail.ptr_to([0]),
            self.mask,
            signal_rank,
            task_type,
            *task_idx,
            source_code=enqueue_remote,
        )


class GEMMMPMCQueue(MPMCQueue):
    @T.inline
    def dequeue(
        self,
        fetched_task_type: T.Buffer,
        fetched_task_idx0: T.Buffer,
        fetched_task_idx1: T.Buffer,
        sem: Semaphore,
        cbx,
        bx,
        rank,
    ):
        self.head_r[0] = T.cuda.atomic_add(
            self.head.access_ptr("rw", offset=self.head.elem_offset_of([T.int32(0)])), 1
        )
        if self.head_r[0] < self.num_tot_tasks:
            # TODO: modify the wait logic to make it faster
            remote_rank = (
                rank + (self.head_r[0] // (LOCAL_GEMM_M_CLUSTERS * GEMM_N_CLUSTERS))
            ) % WORLD_SIZE
            if remote_rank != rank:
                sem.semaphore_wait(remote_rank)

            self.masked_pos[0] = self.head_r[0] & self.mask
            fetched_task_type[0] = T.cuda.func_call(
                "while_ld_global_acquire",
                self.task_types.access_ptr(
                    "r", offset=self.task_types.elem_offset_of([self.masked_pos[0]])
                ),
                source_code=while_ld_global_acquire,
                return_type="int32",
            )
            self.task_types[self.masked_pos[0]] = -1
            fetched_task_idx0[0] = self.task_idxs[self.masked_pos[0], 0]
            fetched_task_idx1[0] = self.task_idxs[self.masked_pos[0], 1]
        else:
            fetched_task_type[0] = -1


# fmt: off
@T.inline
def consumer_fetch(
    sch_pipe,
    packed_value,
    rs_rem,
    fetched_task_type,
    fetched_task_idx0,
    fetched_task_idx1,
):
    sch_pipe.consumer_wait(0)
    T.cuda.func_call(
        "unpack_values",
        packed_value.ptr_to([0]),
        rs_rem.ptr_to([0]),
        fetched_task_type.ptr_to([0]),
        fetched_task_idx0.ptr_to([0]),
        fetched_task_idx1.ptr_to([0]),
        source_code=unpack_values,
    )
    T.ptx.mbarrier.arrive(sch_pipe.mbar_c2p.ptr_to([sch_pipe.idx, 0]), remote=0, pred=True)
    sch_pipe.p2c_phase = sch_pipe.p2c_phase ^ 1
# fmt: on


@T.meta_class
class SingleDynamicTileScheduler:
    def __init__(
        self, queue: MPMCQueue, packed_value: T.Buffer, sch_pipe: Pipeline, sem: Semaphore
    ):
        self.queue = queue
        self.sch_pipe = sch_pipe
        self.fetched_task_type = int_var("fetched_task_type")
        self.fetched_task_idx0 = int_var("fetched_task_idx0")
        self.fetched_task_idx1 = int_var("fetched_task_idx1")
        self.sem = sem
        self.rs_rem = int_var("rs_rem")
        self.packed_value = packed_value
        IRBuilder.current().name("packed_value", self.packed_value)

    # fmt: off
    @T.inline
    def _fetch_from_queue(
        self,
        cbx,
        bx,
        rank,
        warp_id_in_cta,
        lane_id,
    ):
        # fetch from GEMM queue
        if warp_id_in_cta == 11 and lane_id == 0:
            if cbx == 0:
                self.sch_pipe.producer_wait(0)
                self.queue.dequeue(self.fetched_task_type, self.fetched_task_idx0, self.fetched_task_idx1, self.sem, cbx, bx, rank)
                T.cuda.func_call(
                    "pack_values",
                    self.rs_rem[0],
                    self.fetched_task_type[0],
                    self.fetched_task_idx0[0],
                    self.fetched_task_idx1[0],
                    self.packed_value.ptr_to([0]),
                    source_code=pack_values,
                )
                # T.cuda.thread_fence()
                T.ptx.mbarrier.arrive(self.sch_pipe.mbar_p2c.ptr_to([self.sch_pipe.idx, 0]), remote=0, pred=True)
                T.ptx.mbarrier.arrive(self.sch_pipe.mbar_p2c.ptr_to([self.sch_pipe.idx, 0]), remote=1, pred=True)
                self.sch_pipe.c2p_phase = self.sch_pipe.c2p_phase ^ 1
        if ENABLE_WARP_BROADCAST:
            if lane_id == 0:
                consumer_fetch(self.sch_pipe, self.packed_value, self.rs_rem, self.fetched_task_type, self.fetched_task_idx0, self.fetched_task_idx1)
            T.cuda.func_call(
                "warp_broadcast",
                self.rs_rem.ptr_to([0]),
                self.fetched_task_type.ptr_to([0]),
                self.fetched_task_idx0.ptr_to([0]),
                self.fetched_task_idx1.ptr_to([0]),
                source_code=warp_broadcast,
            )
        else:
            consumer_fetch(self.sch_pipe, self.packed_value, self.rs_rem, self.fetched_task_type, self.fetched_task_idx0, self.fetched_task_idx1)

    @T.inline
    def init(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        self.rs_rem[0] = -1
        self._fetch_from_queue(cbx, bx, rank, warp_id_in_cta, lane_id)

    @T.inline
    def next_tile(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        self._fetch_from_queue(cbx, bx, rank, warp_id_in_cta, lane_id)

    def valid(self):
        return (self.fetched_task_type[0] >= 0) | (self.rs_rem[0] >= 0)
    # fmt: on


@T.meta_class
class SingleStaticTileScheduler:
    """Deterministically assign the rank-aware task order to persistent clusters."""

    def __init__(
        self, queue: MPMCQueue, packed_value: T.Buffer, sch_pipe: Pipeline, sem: Semaphore
    ):
        self.sem = sem
        self.sch_pipe = sch_pipe
        self.packed_value = packed_value
        self.linear_idx = int_var("static_linear_idx")
        self.fetched_task_type = int_var("fetched_task_type")
        self.fetched_task_idx0 = int_var("fetched_task_idx0")
        self.fetched_task_idx1 = int_var("fetched_task_idx1")
        self.rs_rem = int_var("rs_rem")

    @T.inline
    def _update_current_tile(self, rank):
        tasks_per_shard = T.meta_var(LOCAL_GEMM_M_CLUSTERS * GEMM_N_CLUSTERS)
        if self.linear_idx[0] < GEMM_M_CLUSTERS * GEMM_N_CLUSTERS:
            group_span = T.meta_var(GROUP_SIZE * GEMM_N_CLUSTERS)
            group_idx = T.meta_var(self.linear_idx[0] // group_span)
            index_in_group = T.meta_var(self.linear_idx[0] % group_span)
            local_m_idx = T.meta_var(group_idx * GROUP_SIZE + index_in_group % GROUP_SIZE)
            self.fetched_task_type[0] = TaskType.GEMM.value
            self.fetched_task_idx0[0] = (
                rank * LOCAL_GEMM_M_CLUSTERS + local_m_idx
            ) % GEMM_M_CLUSTERS
            self.fetched_task_idx1[0] = index_in_group // GROUP_SIZE
            remote_rank = T.meta_var((rank + self.linear_idx[0] // tasks_per_shard) % WORLD_SIZE)
            if remote_rank != rank:
                self.sem.semaphore_wait(remote_rank)
        else:
            self.fetched_task_type[0] = -1

    @T.inline
    def _fetch_from_schedule(self, cbx, rank, warp_id_in_cta, lane_id):
        if warp_id_in_cta == 11 and lane_id == 0:
            if cbx == 0:
                self.sch_pipe.producer_wait(0)
                self._update_current_tile(rank)
                T.cuda.func_call(
                    "pack_values",
                    self.rs_rem[0],
                    self.fetched_task_type[0],
                    self.fetched_task_idx0[0],
                    self.fetched_task_idx1[0],
                    self.packed_value.ptr_to([0]),
                    source_code=pack_values,
                )
                T.ptx.mbarrier.arrive(
                    self.sch_pipe.mbar_p2c.ptr_to([self.sch_pipe.idx, 0]), remote=0, pred=True
                )
                T.ptx.mbarrier.arrive(
                    self.sch_pipe.mbar_p2c.ptr_to([self.sch_pipe.idx, 0]), remote=1, pred=True
                )
                self.sch_pipe.c2p_phase = self.sch_pipe.c2p_phase ^ 1
        consumer_fetch(
            self.sch_pipe,
            self.packed_value,
            self.rs_rem,
            self.fetched_task_type,
            self.fetched_task_idx0,
            self.fetched_task_idx1,
        )

    @T.inline
    def init(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        self.rs_rem[0] = -1
        self.linear_idx[0] = bx // M_CLUSTER
        self._fetch_from_schedule(cbx, rank, warp_id_in_cta, lane_id)

    @T.inline
    def next_tile(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        self.linear_idx[0] += SM_NUMBER // M_CLUSTER
        self._fetch_from_schedule(cbx, rank, warp_id_in_cta, lane_id)

    def valid(self):
        return self.fetched_task_type[0] >= 0


@T.inline
def skip():
    pass


class AllGatherTileImpl(TileImpl):
    """Launch the existing host-side AllGather transfer stage."""

    execution_space = "host"
    entrypoint = "runtime.disco.transfer_to_peers_all_gather"

    def __init__(self):
        super().__init__()
        self._transfer = None
        self._args = ()

    def bind_context(self, transfer, *args) -> None:
        self._transfer = transfer
        self._args = args

    def run(self, m_idx, n_idx, k_idx):
        if self._transfer is None:
            raise RuntimeError("AllGatherTileImpl must be bound before run()")
        return self._transfer(*self._args)


class AllGatherGemmTileImpl(TileImpl):
    """Execute one GEMM cluster after its AllGather shard becomes ready."""

    execution_space = "device"
    entrypoint = "test_mma_ss_tma_2sm_persistent"

    def __init__(self):
        super().__init__()
        self._bound = False

    def bind_context(self, **context) -> None:
        """Bind buffers and pipeline state allocated by the enclosing kernel."""

        for name, value in context.items():
            setattr(self, name, value)
        self._bound = True

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        if not self._bound:
            raise RuntimeError("AllGatherGemmTileImpl must be bound before run()")

        if self.wg_id == NUM_CONSUMER:
            T.ptx.setmaxnreg(False, 56)
            if self.warp_id == 3:
                # GMEM -> SMEM (TMA)
                if T.ptx.elect_sync():
                    n_start = T.meta_var((n_idx * CTA_GROUP + self.cbx) * BLK_N)

                    @T.inline
                    def tma_load(is_remain, ks):
                        tma_copy = T.meta_var(
                            {
                                "dispatch": "tma",
                                "mbar": self.tma_finished.ptr_to([ks]),
                                "cta_group": CTA_GROUP,
                            }
                        )
                        stage_k = T.meta_var(self.stage * BLK_K)
                        self.mma2tma.wait(ks, 0, self.phase[0])
                        if (
                            self.rank * LOCAL_GEMM_M_CLUSTERS <= m_idx
                            and m_idx < (self.rank + 1) * LOCAL_GEMM_M_CLUSTERS
                        ):
                            m_start0 = T.meta_var(
                                (
                                    (m_idx % LOCAL_GEMM_M_CLUSTERS) * NUM_CONSUMER * CTA_GROUP
                                    + self.cbx
                                )
                                * BLK_M
                            )
                            m_start1 = T.meta_var(
                                (
                                    (m_idx % LOCAL_GEMM_M_CLUSTERS) * NUM_CONSUMER * CTA_GROUP
                                    + CTA_GROUP
                                    + self.cbx
                                )
                                * BLK_M
                            )
                            Tx.copy_async(
                                self.A_smem[ks, 0, :, :],
                                self.A[m_start0 : m_start0 + BLK_M, stage_k : stage_k + BLK_K],
                                **tma_copy,
                            )
                            Tx.copy_async(
                                self.A_smem[ks, 1, :, :],
                                self.A[m_start1 : m_start1 + BLK_M, stage_k : stage_k + BLK_K],
                                **tma_copy,
                            )
                        else:
                            m_start0 = T.meta_var(
                                (m_idx * NUM_CONSUMER * CTA_GROUP + self.cbx) * BLK_M
                            )
                            m_start1 = T.meta_var(
                                (m_idx * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + self.cbx) * BLK_M
                            )
                            Tx.copy_async(
                                self.A_smem[ks, 0, :, :],
                                self.ag_out[m_start0 : m_start0 + BLK_M, stage_k : stage_k + BLK_K],
                                **tma_copy,
                            )
                            Tx.copy_async(
                                self.A_smem[ks, 1, :, :],
                                self.ag_out[m_start1 : m_start1 + BLK_M, stage_k : stage_k + BLK_K],
                                **tma_copy,
                            )
                        Tx.copy_async(
                            self.B_smem[ks, :, :],
                            self.B[n_start : n_start + BLK_N, stage_k : stage_k + BLK_K],
                            **tma_copy,
                        )
                        if self.cbx == 0:
                            self.tma2mma.arrive(
                                ks,
                                NUM_CONSUMER * BLK_K * (BLK_M * NUM_CONSUMER + BLK_N) * F16_BYTES,
                            )

                    @T.inline
                    def tma_load_epilogue(ks):
                        self.mma2tma.wait(ks, 0, self.phase[0])
                        if self.cbx == 0:
                            self.tma2mma.arrive_only(ks)

                    self.partitioned_loop(tma_load, skip, tma_load_epilogue)

            elif self.warp_id < 2 and self.cbx == 0:
                if T.ptx.elect_sync():
                    self.ld2mma.wait(0, self.warp_id, self.phase_tmem[0])
                    T.ptx.tcgen05.fence.after_thread_sync()

                    @T.inline
                    def mma(is_remain, ks):
                        self.tma2mma.wait(ks, 0, self.phase[0])
                        for ki in T.unroll(BLK_K // MMA_K):
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(self.descA),
                                self.A_smem.ptr_to([ks, self.warp_id, 0, ki * MMA_K]),
                                ldo=1,
                                sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                swizzle=SWIZZLE,
                            )
                            T.ptx.tcgen05.encode_matrix_descriptor(
                                T.address_of(self.descB),
                                self.B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                ldo=1,
                                sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                swizzle=SWIZZLE,
                            )
                            if (self.stage == 0 and ki == 0) and (
                                (not is_remain) or (is_remain and PIPE_CYCLE == 0)
                            ):
                                T.ptx.tcgen05.mma(
                                    self.warp_id * MMA_N,
                                    self.descA,
                                    self.descB,
                                    self.descI,
                                    d_dtype="float32",
                                    a_dtype=a_type,
                                    b_dtype=b_type,
                                    use_a_tmem=False,
                                    cta_group=CTA_GROUP,
                                    enable_input_d=False,
                                )
                            else:
                                T.ptx.tcgen05.mma(
                                    self.warp_id * MMA_N,
                                    self.descA,
                                    self.descB,
                                    self.descI,
                                    d_dtype="float32",
                                    a_dtype=a_type,
                                    b_dtype=b_type,
                                    use_a_tmem=False,
                                    cta_group=CTA_GROUP,
                                    enable_input_d=True,
                                )
                        self.mma2tma.arrive(ks)

                    @T.inline
                    def mma_epilogue1():
                        self.mma2ld.arrive(self.warp_id)

                    @T.inline
                    def mma_epilogue2(ks):
                        self.tma2mma.wait(ks, 0, self.phase[0])
                        self.mma2tma.arrive(ks)

                    self.partitioned_loop(mma, mma_epilogue1, mma_epilogue2)
                    self.phase_tmem[0] = self.phase_tmem[0] ^ 1

        if self.wg_id < NUM_CONSUMER:
            T.ptx.setmaxnreg(True, 224)
            reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
            reg_wg = reg.view(
                128, TMEM_LD_SIZE, layout=TileLayout(T.S[(128, TMEM_LD_SIZE) : (1 @ tid_in_wg, 1)])
            )
            reg_fp16 = T.alloc_buffer((BLK_N * CTA_GROUP,), d_type, scope="local")

            self.mma2ld.wait(0, self.wg_id, self.phase_tmem[0])
            self.phase_tmem[0] = self.phase_tmem[0] ^ 1
            T.ptx.tcgen05.fence.after_thread_sync()
            for i in T.unroll(MMA_N // TMEM_LD_SIZE):
                col_st = T.meta_var(self.wg_id * MMA_N + i * TMEM_LD_SIZE)
                Tx.wg.copy_async(reg_wg[:, :], self.tmem[:, col_st : col_st + TMEM_LD_SIZE])
                Tx.cast(reg_fp16[i * TMEM_LD_SIZE : (i + 1) * TMEM_LD_SIZE], reg[:])

            self.ld2mma.arrive(self.wg_id)
            for i in T.unroll(NUM_CONSUMER * BLK_N // EPI_TILE):
                Tx.copy(
                    self.D_smem[self.wg_id, self.warp_id * 32 + self.lane_id, :],
                    reg_fp16[i * EPI_TILE : (i + 1) * EPI_TILE],
                )
                T.cuda.warpgroup_sync(self.wg_id)
                T.ptx.fence.proxy_async("shared::cta")
                if self.lane_id == 0 and self.warp_id == 0:
                    m_st = T.meta_var(
                        (m_idx * NUM_CONSUMER * CTA_GROUP + self.wg_id * CTA_GROUP + self.cbx)
                        * BLK_M
                    )
                    n_st = T.meta_var(n_idx * BLK_N * CTA_GROUP + i * EPI_TILE)
                    Tx.copy_async(
                        self.out[m_st : m_st + BLK_M, n_st : n_st + EPI_TILE],
                        self.D_smem[self.wg_id, :, :],
                        dispatch="tma",
                    )
                    T.ptx.cp_async.bulk.commit_group()
                    T.ptx.cp_async.bulk.wait_group(0)
                T.cuda.warpgroup_sync(self.wg_id)

    @T.inline
    def partitioned_loop(self, main_loop, epilogue1, epilogue2):
        for ko in T.serial(PIPE_CYCLE):
            for ks in T.unroll(PIPELINE_DEPTH):
                self.stage = ko * PIPELINE_DEPTH + ks
                main_loop(False, ks)
            self.phase[0] = self.phase[0] ^ 1
        if PIPE_REMAIN_NUM > 0:
            for ks in T.unroll(PIPE_REMAIN_NUM):
                self.stage = PIPE_CYCLE * PIPELINE_DEPTH + ks
                main_loop(True, ks)
            epilogue1()
            for ks in T.unroll(PIPE_REMAIN_NUM, PIPELINE_DEPTH):
                epilogue2(ks)
            self.phase[0] = self.phase[0] ^ 1
        else:
            epilogue1()


def build_kernel(scheduler: str = "dynamic", tile_impl: AllGatherGemmTileImpl | None = None):
    if scheduler == "dynamic":
        scheduler_class = SingleDynamicTileScheduler
    elif scheduler == "static":
        scheduler_class = SingleStaticTileScheduler
    else:
        raise ValueError(f"unsupported AllGather+GEMM scheduler: {scheduler!r}")

    A_layout = T.ComposeLayout(
        3,
        3,
        3,
        T.TileLayout(
            T.S[
                (PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K) : (
                    NUM_CONSUMER * BLK_M * BLK_K,
                    BLK_M * BLK_K,
                    BLK_K,
                    1,
                )
            ]
        ),
    )
    B_layout = T.ComposeLayout(
        3, 3, 3, T.TileLayout(T.S[(PIPELINE_DEPTH, BLK_N, BLK_K) : (BLK_N * BLK_K, BLK_K, 1)])
    )
    D_layout = T.ComposeLayout(
        3,
        3,
        3,
        T.TileLayout(T.S[(NUM_CONSUMER, BLK_M, EPI_TILE) : (BLK_M * EPI_TILE, EPI_TILE, 1)]),
    )

    tile_impl = tile_impl or AllGatherGemmTileImpl()

    # fmt: off
    @T.prim_func
    def test_mma_ss_tma_2sm_persistent(A: T.Buffer((LOCAL_M, K), a_type), B: T.Buffer((LOCAL_N, K), b_type), ag_out: T.Buffer((M, K), a_type),
                                       semaphore: T.Buffer((WORLD_SIZE,), "uint64"), out: T.Buffer((M, LOCAL_N), d_type), profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64"),
                                       gemm_task_types: T.Buffer((CAPACITY,), "int32"), gemm_task_idxs: T.Buffer((CAPACITY, 2), "int32"), gemm_head: T.Buffer((1,), "int32"), gemm_tail: T.Buffer((1,), "int32")):
        T.device_entry()
        cbx, cby = T.cta_id_in_cluster([M_CLUSTER, N_CLUSTER])
        bx = T.cta_id([SM_NUMBER])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([WARP_NUMBER])
        warp_id_in_cta = T.warp_id([WG_NUMBER * WARP_NUMBER])
        lane_id = T.lane_id([32])
        tid = T.thread_id([NUM_THREADS])
        rank = T.nvshmem.my_pe()
        # alloc shared memory
        buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
        tmem_addr = T.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)
        A_smem = T.decl_buffer((PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, buf.data, layout=A_layout,
                                elem_offset=1024 // F16_BYTES)
        B_smem = T.decl_buffer((PIPELINE_DEPTH, BLK_N, BLK_K), b_type, buf.data, layout=B_layout,
                                elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * NUM_CONSUMER * BLK_M * BLK_K)
        D_smem = T.decl_buffer((NUM_CONSUMER, BLK_M, EPI_TILE), d_type, buf.data, layout=D_layout,
                                elem_offset=1024 // F16_BYTES + PIPELINE_DEPTH * (NUM_CONSUMER * BLK_M + BLK_N) * BLK_K)

        # alloc local memory
        descA = T.local_scalar("uint64")
        descB = T.local_scalar("uint64")
        descI = T.local_scalar("uint32")
        phase = T.alloc_buffer((1,), "int32", scope="local")
        phase_tmem = T.alloc_buffer((1,), "int32", scope="local")
        stage = T.local_scalar("int32")

        # ag + gemm
        sem = T.meta_var(Semaphore(cnt=1, buffer=semaphore))
        gemm_queue = T.meta_var(GEMMMPMCQueue(CAPACITY, gemm_task_types, gemm_task_idxs, gemm_head, gemm_tail, GEMM_M_CLUSTERS * GEMM_N_CLUSTERS))
        packed_buf = T.decl_buffer((1,), "uint64", buf.data, elem_offset=64)
        packed_ptr: T.let[T.Var(name="packed_ptr", dtype=PointerType(PrimType("uint64")))] = T.reinterpret("handle", T.ptx.map_shared_rank(packed_buf.ptr_to([0]), 0)) # rank: 0
        packed_value = T.decl_buffer([1,], "uint64", data=packed_ptr, scope="shared")
        sch_pipe = T.meta_var(Pipeline(buf.data, 64 + 4, pipeline_depth=1, pipeline_num=1, p_single_cta=True, c_single_cta=False))
        tile_scheduler = T.meta_var(scheduler_class(gemm_queue, packed_value, sch_pipe, sem))
        profiler = T.meta_var(CudaProfiler(profiler_buffer, write_stride=PROFILER_WRITE_STRIDE, num_groups=NUM_GROUPS, profiler_enabled=PROFILER_ON))

        # initialize
        profiler.init(warp_id_in_cta)
        tma2mma = T.meta_var(BarTMA2MMA(buf.data, 4, PIPELINE_DEPTH, 1, is_p2c=True))
        mma2tma = T.meta_var(BarMMA2TMA(buf.data, 4 + PIPELINE_DEPTH, PIPELINE_DEPTH, 1, is_p2c=False))
        mma2ld = T.meta_var(BarMMA2LD(buf.data, 4 + 2 * PIPELINE_DEPTH, 1, NUM_CONSUMER, is_p2c=True))
        ld2mma = T.meta_var(BarLD2MMA(buf.data, 4 + 2 * PIPELINE_DEPTH + NUM_CONSUMER, 1, NUM_CONSUMER, is_p2c=False))
        tma2mma.init(1, tid == 0)
        mma2tma.init(NUM_CONSUMER, tid == 0)
        mma2ld.init(1, tid == 0)
        ld2mma.init(128 * NUM_CONSUMER, tid == 0)
        ptr: T.let[T.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = T.reinterpret("handle", T.ptx.map_shared_rank(tma2mma.mbar.ptr_to([0, 0]), 0))
        tma_finished = T.decl_buffer([PIPELINE_DEPTH], "uint64", data=ptr, scope="shared")
        phase[0] = 0
        phase_tmem[0] = 0
        sch_pipe.init(tid == 0, c2p_thread_count=C2P_THREAD_COUNT, p2c_thread_count=1)
        T.ptx.tcgen05.encode_instr_descriptor(
            T.address_of(descI),
            d_dtype="float32",
            a_dtype=a_type,
            b_dtype=b_type,
            M=MMA_M,
            N=MMA_N,
            K=MMA_K,
            trans_a=False,
            trans_b=False,
            n_cta_groups=CTA_GROUP,
        )

        # alloc TMEM
        if (wg_id == 0) & (warp_id == 0):
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=CTA_GROUP)

        T.ptx.barrier.cluster.arrive()
        T.ptx.barrier.cluster.wait()
        T.cuda.cta_sync()
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        tile_scheduler.init(cbx, bx, rank, warp_id_in_cta, lane_id)

        T.cuda.trap_when_assert_failed(tmem_addr == 0)
        tmem = T.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(T.S[(128, N_COLS) : (1@TLane, 1@TCol)]))

        tile_impl.bind_context(
            A=A,
            B=B,
            ag_out=ag_out,
            out=out,
            A_smem=A_smem,
            B_smem=B_smem,
            D_smem=D_smem,
            tmem=tmem,
            descA=descA,
            descB=descB,
            descI=descI,
            phase=phase,
            phase_tmem=phase_tmem,
            stage=stage,
            tma_finished=tma_finished,
            tma2mma=tma2mma,
            mma2tma=mma2tma,
            mma2ld=mma2ld,
            ld2mma=ld2mma,
            cbx=cbx,
            wg_id=wg_id,
            warp_id=warp_id,
            lane_id=lane_id,
            rank=rank,
        )

        while tile_scheduler.valid():
            if tile_scheduler.fetched_task_type[0] == TaskType.GEMM.value:
                profiler.start(ProfileEventType.GEMM, tid == 0)
                m_idx = T.meta_var(tile_scheduler.fetched_task_idx0[0])
                n_idx = T.meta_var(tile_scheduler.fetched_task_idx1[0])
                tile_impl.run(m_idx, n_idx, 0)
                profiler.end(ProfileEventType.GEMM, tid == 0)

            tile_scheduler.next_tile(cbx, bx, rank, warp_id_in_cta, lane_id)

        # dealloc TMEM
        if (wg_id == 0) & (warp_id == 0):
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=CTA_GROUP)

        T.ptx.barrier.cluster.arrive()
        T.ptx.barrier.cluster.wait()

    # fmt: on

    return test_mma_ss_tma_2sm_persistent


# Runtime orchestration is kept separate from the complete device kernel above.
# Import it only after every constant and TIRx definition is available so the
# DSL modules can refer back to this first-class kernel module without a cycle.
from ._allgather_gemm_runner import (  # noqa: E402
    CONFIGS,
    KERNEL_META,
    _manual_queue_state,
    _queue_state,
    get_kernel,
    prepare_data,
    run_bench,
    run_test,
)

__all__ = [
    "CONFIGS",
    "KERNEL_META",
    "AllGatherGemmTileImpl",
    "AllGatherTileImpl",
    "_manual_queue_state",
    "_queue_state",
    "build_kernel",
    "get_kernel",
    "prepare_data",
    "run_bench",
    "run_test",
]
