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

"""Direct TP4 port of the fused persistent dynamic-multimem GemmRS kernel."""

from dataclasses import dataclass
from enum import Enum

import tvm
from tvm.ir.type import PointerType, PrimType
from tvm.script import tirx as Tx
from tvm.tirx.lang.pipeline import Pipeline as DataPipeline
from tvm.tirx.layout import wg_local_layout


class TaskType(Enum):
    GEMM = 0
    RS = 1


M, N, K = (8192, 5120, 25600 // 4)
DTYPE = "float16"
SUPPORTED_WORLD_SIZES = (4,)
M_CLUSTER = 2
N_CLUSTER = 1
WG_NUMBER = 3
WARP_NUMBER = 4
NUM_CONSUMER = 2
NUM_THREADS = 32 * WARP_NUMBER * WG_NUMBER
SM_NUMBER = 148
PIPELINE_DEPTH = 4
F16_BYTES = 2
F128_BYTES = 16
d_type, a_type, b_type = ("float16", "float16", "float16")
WORLD_SIZE = 4
TOTAL_K = K * WORLD_SIZE
LOCAL_M = M // WORLD_SIZE
BLK_M, BLK_N, BLK_K = (128, 128, 64)
assert LOCAL_M * WORLD_SIZE == M, "M must be divisible by WORLD_SIZE"
assert LOCAL_M % BLK_M == 0, "LOCAL_M must be divisible by BLK_M"
MMA_M, MMA_N, MMA_K = (256, 256, 16)
EPI_TILE = 64
SWIZZLE = 3
TMEM_LD_SIZE = 64
N_COLS = 512
CTA_GROUP = 2
PIPE_CYCLE = K // BLK_K // PIPELINE_DEPTH
PIPE_REMAIN_NUM = K // BLK_K % PIPELINE_DEPTH
assert PIPELINE_DEPTH == 4
GROUP_SIZE = 8
assert M % (NUM_CONSUMER * BLK_M * CTA_GROUP) == 0
assert N % (BLK_N * CTA_GROUP) == 0
GEMM_M_CLUSTERS = M // (NUM_CONSUMER * BLK_M * CTA_GROUP)
GEMM_N_CLUSTERS = N // (BLK_N * CTA_GROUP)
TILE_M, TILE_N = (BLK_M * 2, BLK_N * 2)
RS_M_CLUSTERS = LOCAL_M // (BLK_M * CTA_GROUP)
RS_N_CLUSTERS = N // (BLK_N * CTA_GROUP)
CAPACITY = 2048
TASK_IDX_LEN = 2
C2P_THREAD_COUNT = NUM_THREADS * CTA_GROUP
FUSED_DEVICE_ENTRYPOINT = "test_mma_ss_tma_2sm_persistent"


@dataclass(frozen=True)
class GemmRSConfig:
    M: int
    N: int
    total_k: int
    world_size: int
    dtype: str
    k_local: int
    local_m: int
    pipe_cycle: int
    pipe_remainder: int
    gemm_m_clusters: int
    gemm_n_clusters: int
    rs_m_clusters: int
    rs_n_clusters: int
    gemm_task_count: int
    rs_task_count: int
    completion_count: int


def derive_config(
    M: int = M, N: int = N, K: int = TOTAL_K, world_size: int = WORLD_SIZE, dtype: str = DTYPE
) -> GemmRSConfig:
    """Validate and derive the one manually ported specialization."""

    expected = (globals()["M"], globals()["N"], TOTAL_K, WORLD_SIZE, DTYPE)
    actual = (M, N, K, world_size, dtype)
    if actual != expected:
        raise ValueError(
            "manual dynamic GemmRS currently supports only "
            f"M={expected[0]}, N={expected[1]}, K={expected[2]}, "
            f"world_size={expected[3]}, dtype={expected[4]}; got "
            f"M={M}, N={N}, K={K}, world_size={world_size}, dtype={dtype}"
        )
    config = GemmRSConfig(
        M=M,
        N=N,
        total_k=K,
        world_size=world_size,
        dtype=dtype,
        k_local=K // world_size,
        local_m=M // world_size,
        pipe_cycle=(K // world_size // BLK_K) // PIPELINE_DEPTH,
        pipe_remainder=(K // world_size // BLK_K) % PIPELINE_DEPTH,
        gemm_m_clusters=M // (NUM_CONSUMER * BLK_M * CTA_GROUP),
        gemm_n_clusters=N // (BLK_N * CTA_GROUP),
        rs_m_clusters=(M // world_size) // (BLK_M * CTA_GROUP),
        rs_n_clusters=N // (BLK_N * CTA_GROUP),
        gemm_task_count=GEMM_M_CLUSTERS * GEMM_N_CLUSTERS,
        rs_task_count=RS_M_CLUSTERS * RS_N_CLUSTERS,
        completion_count=2 * world_size,
    )
    if config.gemm_task_count > CAPACITY or config.rs_task_count > CAPACITY:
        raise AssertionError("queue capacity is too small for the tuned workload")
    return config


ld_reduce_8xfp16 = '\n__forceinline__ __device__ void ld_reduce_8_fp16(void* src_addr, void* dst_addr) {\n    int4* source = (int4*) nvshmemx_mc_ptr(NVSHMEM_TEAM_WORLD, src_addr);\n    int4* dest = (int4*) dst_addr;\n    constexpr int UNROLL = 1;\n    union {\n        uint16_t u2[8 * UNROLL];\n        uint64_t u8[2 * UNROLL];\n    };\n    for (int u = 0; u < UNROLL; u++) {\n        asm("multimem.ld_reduce.global.add.v8.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"\n            : "=h"(u2[8 * u]), "=h"(u2[8 * u + 1]), "=h"(u2[8 * u + 2]), "=h"(u2[8 * u + 3]), "=h"(u2[8 * u + 4]), "=h"(u2[8 * u + 5]), "=h"(u2[8 * u + 6]), "=h"(u2[8 * u + 7])\n            : "l"(source + u));\n    }\n    for (int u = 0; u < UNROLL; u++) {\n        asm("st.global.v2.b64 [%0], {%1, %2};" ::"l"(dest + u), "l"(u8[2 * u]),\n            "l"(u8[2 * u + 1]));\n    }\n}\n'
pack_values = '\n__forceinline__ __device__ void pack_values(int32_t rem, int32_t task_type, int32_t task_idx0, int32_t task_idx1, uint64_t* dst_addr) {\n    asm volatile("st.shared::cluster.v4.b32 [%0], {%1, %2, %3, %4};"\n                 :\n                 : "l"(dst_addr), "r"(rem), "r"(task_type), "r"(task_idx0), "r"(task_idx1)\n                 : "memory");\n}\n'
unpack_values = '\n__forceinline__ __device__ void unpack_values(uint64_t* src_addr, int32_t* rem, int32_t* task_type, int32_t* task_idx0, int32_t* task_idx1) {\n    asm volatile("ld.shared::cluster.v4.b32 {%0, %1, %2, %3}, [%4];"\n                 : "=r"(*rem), "=r"(*task_type), "=r"(*task_idx0), "=r"(*task_idx1)\n                 : "l"(src_addr)\n                 : "memory");\n\n}\n'
semaphore_notify_remote = "\n__forceinline__ __device__ uint64_t semaphore_notify_remote(int32_t signal_rank, uint64_t* addr, uint64_t signal_value) {\n    auto dst_addr = reinterpret_cast<unsigned long long*>(nvshmem_ptr(addr, signal_rank));\n    return atomicAdd_system(dst_addr, signal_value);\n}\n"
enqueue_remote = """
__forceinline__ __device__ void enqueue_remote(
        int32_t* task_types, int32_t* task_idxs, int32_t* tail, int32_t mask,
        int32_t signal_rank, int32_t task_type, int32_t task_idx0, int32_t task_idx1) {
    int32_t* remote_task_types = (int32_t*)nvshmem_ptr(task_types, signal_rank);
    int32_t* remote_task_idxs = (int32_t*)nvshmem_ptr(task_idxs, signal_rank);
    int32_t* remote_tail = (int32_t*)nvshmem_ptr(tail, signal_rank);
    int32_t tail_r = atomicAdd_system(remote_tail, 1);
    int32_t masked_pos = tail_r & mask;
    remote_task_idxs[masked_pos * 2] = task_idx0;
    remote_task_idxs[masked_pos * 2 + 1] = task_idx1;
    asm volatile(
        "st.global.release.sys.b32 [%0], %1;"
        :
        : "l"(remote_task_types + masked_pos), "r"(task_type)
        : "memory");
}
"""
ld_global_acquire = """
__forceinline__ __device__ int32_t ld_global_acquire(int32_t* addr) {
    int32_t value;
    asm volatile(
        "ld.global.acquire.sys.b32 %0, [%1];"
        : "=r"(value)
        : "l"(addr)
        : "memory");
    return value;
}
"""
while_ld_global_acquire = """
__forceinline__ __device__ int32_t while_ld_global_acquire(int32_t* addr) {
    int32_t value;
    asm volatile(
        "ld.global.acquire.sys.b32 %0, [%1];"
        : "=r"(value)
        : "l"(addr)
        : "memory");
    while (value < 0) {
        __nanosleep(40);
        asm volatile(
            "ld.global.acquire.sys.b32 %0, [%1];"
            : "=r"(value)
            : "l"(addr)
            : "memory");
    }
    return value;
}
"""


@Tx.meta_class
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
        self.mbar_p2c = Tx.decl_buffer(
            (pipeline_depth, pipeline_num), "uint64", shared_buf, elem_offset=base_offset
        )
        self.mbar_c2p = Tx.decl_buffer(
            (pipeline_depth, pipeline_num),
            "uint64",
            shared_buf,
            elem_offset=base_offset + pipeline_depth * pipeline_num,
        )
        self.idx = Tx.local_scalar("int32")
        self.p2c_phase = Tx.local_scalar("int32")
        self.c2p_phase = Tx.local_scalar("int32")
        self.p_single_cta = p_single_cta
        self.c_single_cta = c_single_cta

    @Tx.inline
    def init(self, p2c_thread_count: int = 1, c2p_thread_count: int = 1):
        tid = Tx.thread_id([NUM_THREADS])
        self.idx = 0
        self.p2c_phase = 0
        self.c2p_phase = 1
        if tid == 0:
            for cbx in Tx.thread_binding(M_CLUSTER, "clusterCtaIdx.x"):
                for i in Tx.serial(0, self.pipeline_depth):
                    for j in Tx.serial(0, self.pipeline_num):
                        if not self.c_single_cta or cbx == 0:
                            Tx.ptx.mbarrier.init(self.mbar_p2c.ptr_to([i, j]), p2c_thread_count)
                        if not self.p_single_cta or cbx == 0:
                            Tx.ptx.mbarrier.init(self.mbar_c2p.ptr_to([i, j]), c2p_thread_count)
        Tx.ptx.fence.proxy_async("shared::cta")

    @Tx.inline
    def advance(self):
        self.idx = (self.idx + 1) % self.pipeline_depth
        if self.idx == 0:
            self.p2c_phase = self.p2c_phase ^ 1
            self.c2p_phase = self.c2p_phase ^ 1

    @Tx.inline
    def producer_wait(self, pipeline_idx):
        for cbx in Tx.thread_binding(M_CLUSTER, "clusterCtaIdx.x"):
            if not self.p_single_cta or cbx == 0:
                Tx.ptx.mbarrier.try_wait(
                    self.mbar_c2p.ptr_to([self.idx, pipeline_idx]), self.c2p_phase
                )

    @Tx.inline
    def consumer_wait(self, pipeline_idx):
        for cbx in Tx.thread_binding(M_CLUSTER, "clusterCtaIdx.x"):
            if not self.c_single_cta or cbx == 0:
                Tx.ptx.mbarrier.try_wait(
                    self.mbar_p2c.ptr_to([self.idx, pipeline_idx]), self.p2c_phase
                )


def int_var(scope="local", dtype="int32", align=4):
    buf = Tx.alloc_buffer([1], dtype, scope=scope, align=align)
    return buf


@Tx.meta_class
class MPMCQueue:
    def __init__(
        self,
        capacity: int,
        task_types: Tx.Buffer,
        task_idxs: Tx.Buffer,
        head: Tx.Buffer,
        tail: Tx.Buffer,
        num_tot_tasks: int,
    ):
        if capacity & capacity - 1:
            raise ValueError("capacity must be a power-of-two")
        self.mask = capacity - 1
        self.task_types = task_types
        self.task_idxs = task_idxs
        self.head = head
        self.tail = tail
        self.head_r = int_var()
        self.masked_pos = int_var()
        self.num_tot_tasks = num_tot_tasks

    @Tx.inline
    def enqueue(self, signal_rank: int, task_type: int, *task_idx: int):
        Tx.cuda.func_call(
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
    @Tx.inline
    def dequeue(
        self,
        fetched_task_type: Tx.Buffer,
        fetched_task_idx0: Tx.Buffer,
        fetched_task_idx1: Tx.Buffer,
        rs_rem: Tx.Buffer,
        cbx,
        bx,
        rank,
    ):
        self.head_r[0] = Tx.cuda.atomic_add(
            self.head.access_ptr("rw", offset=self.head.elem_offset_of([Tx.int32(0)])), 1
        )
        if self.head_r[0] < self.num_tot_tasks:
            self.masked_pos[0] = self.head_r[0] & self.mask
            fetched_task_type[0] = Tx.cuda.func_call(
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


class RSMPMCQueue(MPMCQueue):
    @Tx.inline
    def dequeue(
        self,
        fetched_task_type: Tx.Buffer,
        fetched_task_idx0: Tx.Buffer,
        fetched_task_idx1: Tx.Buffer,
        rs_rem: Tx.Buffer,
        cbx,
        bx,
        rank,
    ):
        if rs_rem[0] >= 0:
            self.head_r[0] = rs_rem[0]
            rs_rem[0] = -1
        else:
            self.head_r[0] = Tx.cuda.atomic_add(
                self.head.access_ptr("rw", offset=self.head.elem_offset_of([Tx.int32(0)])), 1
            )
        if self.head_r[0] < self.num_tot_tasks:
            self.masked_pos[0] = self.head_r[0] & self.mask
            fetched_task_type[0] = Tx.cuda.func_call(
                "ld_global_acquire",
                self.task_types.access_ptr(
                    "r", offset=self.task_types.elem_offset_of([self.masked_pos[0]])
                ),
                source_code=ld_global_acquire,
                return_type="int32",
            )
            if fetched_task_type[0] < 0:
                rs_rem[0] = self.head_r[0]
            else:
                self.task_types[self.masked_pos[0]] = -1
                fetched_task_idx0[0] = self.task_idxs[self.masked_pos[0], 0]
                fetched_task_idx1[0] = self.task_idxs[self.masked_pos[0], 1]
        else:
            fetched_task_type[0] = -1


@Tx.inline
def consumer_fetch(
    sch_pipe, packed_value, rs_rem, fetched_task_type, fetched_task_idx0, fetched_task_idx1
):
    sch_pipe.consumer_wait(0)
    Tx.cuda.func_call(
        "unpack_values",
        packed_value.ptr_to([0]),
        rs_rem.ptr_to([0]),
        fetched_task_type.ptr_to([0]),
        fetched_task_idx0.ptr_to([0]),
        fetched_task_idx1.ptr_to([0]),
        source_code=unpack_values,
    )
    Tx.ptx.mbarrier.arrive(sch_pipe.mbar_c2p.ptr_to([sch_pipe.idx, 0]), remote=0, pred=True)
    sch_pipe.p2c_phase = sch_pipe.p2c_phase ^ 1


@Tx.meta_class
class MixedDynamicTileScheduler:
    def __init__(
        self,
        gemm_queue: GEMMMPMCQueue,
        rs_queue: RSMPMCQueue,
        packed_value: Tx.Buffer,
        sch_pipe: Pipeline,
    ):
        self.gemm_queue = gemm_queue
        self.rs_queue = rs_queue
        self.sch_pipe = sch_pipe
        self.fetched_task_type = int_var()
        self.fetched_task_idx0 = int_var()
        self.fetched_task_idx1 = int_var()
        self.rs_rem = int_var()
        self.packed_value = packed_value

    @Tx.inline
    def _fetch_from_queue(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        if (warp_id_in_cta == 11) & (lane_id == 0):
            if cbx == 0:
                self.sch_pipe.producer_wait(0)
                self.rs_queue.dequeue(
                    self.fetched_task_type,
                    self.fetched_task_idx0,
                    self.fetched_task_idx1,
                    self.rs_rem,
                    cbx,
                    bx,
                    rank,
                )
                if self.fetched_task_type[0] < 0:
                    self.gemm_queue.dequeue(
                        self.fetched_task_type,
                        self.fetched_task_idx0,
                        self.fetched_task_idx1,
                        self.rs_rem,
                        cbx,
                        bx,
                        rank,
                    )
                Tx.cuda.func_call(
                    "pack_values",
                    self.rs_rem[0],
                    self.fetched_task_type[0],
                    self.fetched_task_idx0[0],
                    self.fetched_task_idx1[0],
                    self.packed_value.ptr_to([0]),
                    source_code=pack_values,
                )
                Tx.cuda.thread_fence()
                Tx.ptx.mbarrier.arrive(
                    self.sch_pipe.mbar_p2c.ptr_to([self.sch_pipe.idx, 0]), remote=0, pred=True
                )
                Tx.ptx.mbarrier.arrive(
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

    @Tx.inline
    def init(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        self.rs_rem[0] = -1
        self._fetch_from_queue(cbx, bx, rank, warp_id_in_cta, lane_id)

    @Tx.inline
    def next_tile(self, cbx, bx, rank, warp_id_in_cta, lane_id):
        self._fetch_from_queue(cbx, bx, rank, warp_id_in_cta, lane_id)

    def valid(self):
        return tvm.tirx.any(self.fetched_task_type[0] >= 0, self.rs_rem[0] >= 0)


@Tx.meta_class
class Semaphore:
    def __init__(self, cnt, buffer):
        self.cnt = cnt
        self.sem = buffer
        self.state = Tx.alloc_buffer([1], "uint64", scope="local", align=8)

    @Tx.inline
    def semaphore_notify(self, signal_rank, tid, m_idx, n_idx, rs_queue):
        if tid % 128 == 0:
            self.state[0] = (
                Tx.cuda.func_call(
                    "semaphore_notify_remote",
                    signal_rank,
                    self.sem.access_ptr("rw", offset=self.sem.elem_offset_of((m_idx, n_idx))),
                    Tx.uint64(1),
                    source_code=semaphore_notify_remote,
                    return_type="uint64",
                )
                + 1
            )
            if self.state[0] == self.cnt:
                rs_queue.enqueue(signal_rank, TaskType.RS.value, m_idx, n_idx)
        Tx.cuda.thread_fence()


@Tx.prim_func
def test_mma_ss_tma_2sm_persistent(
    A: Tx.Buffer((M, K), a_type),
    B: Tx.Buffer((N, K), b_type),
    gemm_out: Tx.Buffer((M, N), d_type),
    semaphore: Tx.Buffer((LOCAL_M // TILE_M, N // TILE_N), "uint64"),
    out: Tx.Buffer((LOCAL_M, N), d_type),
    gemm_task_types: Tx.Buffer((CAPACITY,), "int32"),
    gemm_task_idxs: Tx.Buffer((CAPACITY, 2), "int32"),
    gemm_head: Tx.Buffer((1,), "int32"),
    gemm_tail: Tx.Buffer((1,), "int32"),
    rs_task_types: Tx.Buffer((CAPACITY,), "int32"),
    rs_task_idxs: Tx.Buffer((CAPACITY, 2), "int32"),
    rs_head: Tx.Buffer((1,), "int32"),
    rs_tail: Tx.Buffer((1,), "int32"),
):
    A_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
    B_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
    D_tensor_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
    Tx.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        A_tensor_map,
        a_type,
        2,
        A.data,
        K,
        M,
        K * F16_BYTES,
        BLK_K,
        BLK_M,
        1,
        1,
        0,
        SWIZZLE,
        0,
        0,
    )
    Tx.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        B_tensor_map,
        b_type,
        2,
        B.data,
        K,
        N,
        K * F16_BYTES,
        BLK_K,
        BLK_N,
        1,
        1,
        0,
        SWIZZLE,
        0,
        0,
    )
    Tx.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        D_tensor_map,
        d_type,
        2,
        gemm_out.data,
        N,
        M,
        N * F16_BYTES,
        EPI_TILE,
        BLK_M,
        1,
        1,
        0,
        SWIZZLE,
        0,
        0,
    )
    Tx.device_entry()
    cbx, cby = Tx.cta_id_in_cluster([M_CLUSTER, N_CLUSTER])
    bx = Tx.cta_id([SM_NUMBER])
    wg_id = Tx.warpgroup_id([WG_NUMBER])
    warp_id = Tx.warp_id_in_wg([WARP_NUMBER])
    warp_id_in_cta = Tx.warp_id([WG_NUMBER * WARP_NUMBER])
    lane_id = Tx.lane_id([32])
    tid = Tx.thread_id([NUM_THREADS])
    rank = Tx.nvshmem.my_pe()
    pool = Tx.SMEMPool()
    tmem_addr = pool.alloc([1], "uint32", align=4)
    tmem_pool = Tx.TMEMPool(pool, total_cols=N_COLS, cta_group=CTA_GROUP, tmem_addr=tmem_addr)
    smem_pipe = DataPipeline(
        pool,
        PIPELINE_DEPTH,
        full="tma",
        empty="tcgen05",
        init_empty=NUM_CONSUMER,
        empty_phase_offset=1,
    )
    tmem_pipe = DataPipeline(
        pool,
        NUM_CONSUMER,
        full="tcgen05",
        empty="mbar",
        init_empty=128 * NUM_CONSUMER,
        empty_phase_offset=1,
    )
    packed_buf = pool.alloc((1,), "uint64", align=16)
    sch_pipe_base = pool.offset // 8
    pool.move_base_to(pool.offset + 2 * 1 * 1 * 8)
    pool.move_base_to(1024)
    A_smem = pool.alloc_tcgen05_mma_AB((PIPELINE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type)
    B_smem = pool.alloc_tcgen05_mma_AB((PIPELINE_DEPTH, BLK_N, BLK_K), b_type)
    D_smem = pool.alloc_tcgen05_mma_AB((NUM_CONSUMER, BLK_M, EPI_TILE), d_type)
    pool.commit()
    reg = Tx.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
    reg_wg = reg.view(128, TMEM_LD_SIZE, layout=wg_local_layout(TMEM_LD_SIZE))
    reg_fp16 = Tx.alloc_buffer((BLK_N * CTA_GROUP,), d_type, scope="local")
    descA: Tx.uint64
    descB: Tx.uint64
    descI: Tx.uint32
    phase: Tx.int32
    phase_tmem: Tx.int32
    stage: Tx.int32
    sem = Tx.meta_var(Semaphore(cnt=2 * WORLD_SIZE, buffer=semaphore))
    offset: Tx.int32
    gemm_queue = Tx.meta_var(
        GEMMMPMCQueue(
            CAPACITY,
            gemm_task_types,
            gemm_task_idxs,
            gemm_head,
            gemm_tail,
            GEMM_M_CLUSTERS * GEMM_N_CLUSTERS,
        )
    )
    rs_queue = Tx.meta_var(
        RSMPMCQueue(
            CAPACITY, rs_task_types, rs_task_idxs, rs_head, rs_tail, RS_M_CLUSTERS * RS_N_CLUSTERS
        )
    )
    packed_ptr: Tx.let[Tx.Var(name="packed_ptr", dtype=PointerType(PrimType("uint64")))] = (
        Tx.reinterpret("handle", Tx.ptx.map_shared_rank(packed_buf.ptr_to([0]), 0))
    )
    packed_value = Tx.decl_buffer([1], "uint64", data=packed_ptr, scope="shared")
    sch_pipe = Pipeline(
        pool.ptr,
        sch_pipe_base,
        pipeline_depth=1,
        pipeline_num=1,
        p_single_cta=True,
        c_single_cta=False,
    )
    tile_scheduler = MixedDynamicTileScheduler(gemm_queue, rs_queue, packed_value, sch_pipe)
    ptr: Tx.let[Tx.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = Tx.reinterpret(
        "handle", Tx.ptx.map_shared_rank(smem_pipe.full.ptr_to([0]), 0)
    )
    tma_finished = Tx.decl_buffer([PIPELINE_DEPTH], "uint64", data=ptr, scope="shared")
    phase = 0
    phase_tmem = 0
    sch_pipe.init(c2p_thread_count=C2P_THREAD_COUNT, p2c_thread_count=1)
    Tx.ptx.tcgen05.encode_instr_descriptor(
        Tx.address_of(descI),
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
    tmem = tmem_pool.alloc((128, N_COLS), "float32")
    tmem_pool.commit()
    Tx.ptx.barrier.cluster.arrive()
    Tx.ptx.barrier.cluster.wait()
    Tx.cuda.cta_sync()
    Tx.cuda.trap_when_assert_failed(tmem_addr[0] == 0)
    Tx.ptx.fence.proxy_async("shared::cta")
    Tx.ptx.fence.mbarrier_init()
    tile_scheduler.init(cbx, bx, rank, warp_id_in_cta, lane_id)
    while tile_scheduler.valid():
        if tile_scheduler.fetched_task_type[0] == TaskType.RS.value:
            m_idx = Tx.meta_var(tile_scheduler.fetched_task_idx0[0])
            n_idx = Tx.meta_var(tile_scheduler.fetched_task_idx1[0])
            offset = tid
            while True:
                if offset < TILE_M // 2 * TILE_N // 8:
                    m_start = Tx.meta_var(offset // (TILE_N // 8))
                    n_start = Tx.meta_var(offset % (TILE_N // 8) * 8)
                    Tx.cuda.func_call(
                        "ld_reduce_8_fp16",
                        gemm_out.ptr_to(
                            [
                                rank * LOCAL_M + TILE_M * m_idx + TILE_M // 2 * cbx + m_start,
                                TILE_N * n_idx + n_start,
                            ]
                        ),
                        out.ptr_to(
                            [TILE_M * m_idx + TILE_M // 2 * cbx + m_start, TILE_N * n_idx + n_start]
                        ),
                        source_code=ld_reduce_8xfp16,
                    )
                    offset += NUM_THREADS
                else:
                    break
        elif tile_scheduler.fetched_task_type[0] == TaskType.GEMM.value:
            m_idx = Tx.meta_var(tile_scheduler.fetched_task_idx0[0])
            n_idx = Tx.meta_var(tile_scheduler.fetched_task_idx1[0])
            if (NUM_CONSUMER <= wg_id) & (wg_id < NUM_CONSUMER + 1):
                Tx.ptx.setmaxnreg(False, 56)
                if warp_id == 3:
                    if Tx.filter(lane_id, Tx.ptx.elect_sync()):
                        for ko in Tx.serial(PIPE_CYCLE):
                            for ks in Tx.unroll(PIPELINE_DEPTH):
                                stage = ko * PIPELINE_DEPTH + ks
                                smem_pipe.empty.wait(ks, phase)
                                Tx.ptx.cp_async.bulk.tensor.g2s_cluster(
                                    2,
                                    A_smem.ptr_to([ks, 0, 0, 0]),
                                    tma_finished.ptr_to([ks]),
                                    Tx.address_of(A_tensor_map),
                                    0,
                                    2,
                                    "",
                                    stage * BLK_K,
                                    (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M,
                                )
                                Tx.ptx.cp_async.bulk.tensor.g2s_cluster(
                                    2,
                                    A_smem.ptr_to([ks, 1, 0, 0]),
                                    tma_finished.ptr_to([ks]),
                                    Tx.address_of(A_tensor_map),
                                    0,
                                    2,
                                    "",
                                    stage * BLK_K,
                                    (m_idx * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + cbx) * BLK_M,
                                )
                                Tx.ptx.cp_async.bulk.tensor.g2s_cluster(
                                    2,
                                    B_smem.ptr_to([ks, 0, 0]),
                                    tma_finished.ptr_to([ks]),
                                    Tx.address_of(B_tensor_map),
                                    0,
                                    2,
                                    "",
                                    stage * BLK_K,
                                    (n_idx * CTA_GROUP + cbx) * BLK_N,
                                )
                                if cbx == 0:
                                    smem_pipe.full.arrive(
                                        ks,
                                        NUM_CONSUMER
                                        * BLK_K
                                        * (BLK_M * NUM_CONSUMER + BLK_N)
                                        * F16_BYTES,
                                    )
                            phase = phase ^ 1
                        if PIPE_REMAIN_NUM > 0:
                            for ks in Tx.unroll(PIPE_REMAIN_NUM):
                                stage = PIPE_CYCLE * PIPELINE_DEPTH + ks
                                smem_pipe.empty.wait(ks, phase)
                                Tx.ptx.cp_async.bulk.tensor.g2s_cluster(
                                    2,
                                    A_smem.ptr_to([ks, 0, 0, 0]),
                                    tma_finished.ptr_to([ks]),
                                    Tx.address_of(A_tensor_map),
                                    0,
                                    2,
                                    "",
                                    stage * BLK_K,
                                    (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M,
                                )
                                Tx.ptx.cp_async.bulk.tensor.g2s_cluster(
                                    2,
                                    A_smem.ptr_to([ks, 1, 0, 0]),
                                    tma_finished.ptr_to([ks]),
                                    Tx.address_of(A_tensor_map),
                                    0,
                                    2,
                                    "",
                                    stage * BLK_K,
                                    (m_idx * NUM_CONSUMER * CTA_GROUP + CTA_GROUP + cbx) * BLK_M,
                                )
                                Tx.ptx.cp_async.bulk.tensor.g2s_cluster(
                                    2,
                                    B_smem.ptr_to([ks, 0, 0]),
                                    tma_finished.ptr_to([ks]),
                                    Tx.address_of(B_tensor_map),
                                    0,
                                    2,
                                    "",
                                    stage * BLK_K,
                                    (n_idx * CTA_GROUP + cbx) * BLK_N,
                                )
                                if cbx == 0:
                                    smem_pipe.full.arrive(
                                        ks,
                                        NUM_CONSUMER
                                        * BLK_K
                                        * (BLK_M * NUM_CONSUMER + BLK_N)
                                        * F16_BYTES,
                                    )
                            for ks in Tx.unroll(PIPE_REMAIN_NUM, PIPELINE_DEPTH):
                                smem_pipe.empty.wait(ks, phase)
                                if cbx == 0:
                                    smem_pipe.full.arrive(ks, remote=0)
                            phase = phase ^ 1
                elif (warp_id < 2) & (cbx == 0):
                    if Tx.filter(lane_id, Tx.ptx.elect_sync()):
                        tmem_pipe.empty.wait(warp_id, phase_tmem)
                        Tx.ptx.tcgen05.fence.after_thread_sync()
                        for ko in Tx.serial(PIPE_CYCLE):
                            for ks in Tx.unroll(PIPELINE_DEPTH):
                                stage = ko * PIPELINE_DEPTH + ks
                                smem_pipe.full.wait(ks, phase)
                                for ki in Tx.unroll(BLK_K // MMA_K):
                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                        Tx.address_of(descA),
                                        A_smem.ptr_to([ks, warp_id, 0, ki * MMA_K]),
                                        ldo=1,
                                        sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                        swizzle=SWIZZLE,
                                    )
                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                        Tx.address_of(descB),
                                        B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                        ldo=1,
                                        sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                        swizzle=SWIZZLE,
                                    )
                                    if stage == 0 and ki == 0:
                                        Tx.ptx.tcgen05.mma(
                                            warp_id * MMA_N,
                                            descA,
                                            descB,
                                            descI,
                                            d_dtype="float32",
                                            a_dtype=a_type,
                                            b_dtype=b_type,
                                            use_a_tmem=False,
                                            cta_group=CTA_GROUP,
                                            enable_input_d=0,
                                        )
                                    else:
                                        Tx.ptx.tcgen05.mma(
                                            warp_id * MMA_N,
                                            descA,
                                            descB,
                                            descI,
                                            d_dtype="float32",
                                            a_dtype=a_type,
                                            b_dtype=b_type,
                                            use_a_tmem=False,
                                            cta_group=CTA_GROUP,
                                            enable_input_d=1,
                                        )
                                smem_pipe.empty.arrive(ks, cta_group=CTA_GROUP, cta_mask=3)
                            phase = phase ^ 1
                        if PIPE_REMAIN_NUM > 0:
                            for ks in Tx.unroll(PIPE_REMAIN_NUM):
                                smem_pipe.full.wait(ks, phase)
                                for ki in Tx.unroll(BLK_K // MMA_K):
                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                        Tx.address_of(descA),
                                        A_smem.ptr_to([ks, warp_id, 0, ki * MMA_K]),
                                        ldo=1,
                                        sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                        swizzle=SWIZZLE,
                                    )
                                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                                        Tx.address_of(descB),
                                        B_smem.ptr_to([ks, 0, ki * MMA_K]),
                                        ldo=1,
                                        sdo=8 * BLK_K * F16_BYTES // F128_BYTES,
                                        swizzle=SWIZZLE,
                                    )
                                    if PIPE_CYCLE == 0 and ks == 0 and (ki == 0):
                                        Tx.ptx.tcgen05.mma(
                                            warp_id * MMA_N,
                                            descA,
                                            descB,
                                            descI,
                                            d_dtype="float32",
                                            a_dtype=a_type,
                                            b_dtype=b_type,
                                            use_a_tmem=False,
                                            cta_group=CTA_GROUP,
                                            enable_input_d=0,
                                        )
                                    else:
                                        Tx.ptx.tcgen05.mma(
                                            warp_id * MMA_N,
                                            descA,
                                            descB,
                                            descI,
                                            d_dtype="float32",
                                            a_dtype=a_type,
                                            b_dtype=b_type,
                                            use_a_tmem=False,
                                            cta_group=CTA_GROUP,
                                            enable_input_d=1,
                                        )
                                smem_pipe.empty.arrive(ks, cta_group=CTA_GROUP, cta_mask=3)
                            tmem_pipe.full.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
                            for ks in Tx.unroll(PIPE_REMAIN_NUM, PIPELINE_DEPTH):
                                smem_pipe.full.wait(ks, phase)
                                smem_pipe.empty.arrive(ks, cta_group=CTA_GROUP, cta_mask=3)
                            phase = phase ^ 1
                        else:
                            tmem_pipe.full.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
                        phase_tmem = phase_tmem ^ 1
            if (0 <= wg_id) & (wg_id < NUM_CONSUMER):
                Tx.ptx.setmaxnreg(True, 224)
                Tx.cuda.trap_when_assert_failed(tmem_addr[0] == 0)
                tmem_pipe.full.wait(wg_id, phase_tmem)
                phase_tmem = phase_tmem ^ 1
                Tx.ptx.tcgen05.fence.after_thread_sync()
                for i in Tx.unroll(MMA_N // TMEM_LD_SIZE):
                    col_st = Tx.meta_var(wg_id * MMA_N + i * TMEM_LD_SIZE)
                    Tx.wg.copy_async(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                    Tx.ptx.tcgen05.wait.ld()
                    Tx.thread.cast(reg_fp16[i * TMEM_LD_SIZE : (i + 1) * TMEM_LD_SIZE], reg)
                tmem_pipe.empty.arrive(wg_id, remote=0)
                for i in Tx.unroll(NUM_CONSUMER * BLK_N // EPI_TILE):
                    for it in Tx.unroll(EPI_TILE // 8):
                        for vec in Tx.vectorized(8):
                            D_smem[wg_id, warp_id * 32 + lane_id, it * 8 + vec] = reg_fp16[
                                i * EPI_TILE + it * 8 + vec
                            ]
                    Tx.cuda.warpgroup_sync(wg_id)
                    Tx.ptx.fence.proxy_async("shared::cta")
                    if (lane_id == 0) & (warp_id == 0):
                        Tx.ptx.cp_async.bulk.tensor.s2g(
                            2,
                            D_smem.ptr_to([wg_id, 0, 0]),
                            Tx.address_of(D_tensor_map),
                            "",
                            n_idx * BLK_N * CTA_GROUP + i * EPI_TILE,
                            (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M,
                        )
                        Tx.ptx.cp_async.bulk.commit_group()
                        Tx.ptx.cp_async.bulk.wait_group(0)
                    Tx.cuda.warpgroup_sync(wg_id)
                comm_m_idx = Tx.meta_var(m_idx * 2 + wg_id)
                comm_m_idx_local = Tx.meta_var(comm_m_idx % (LOCAL_M // TILE_M))
                signal_rank = Tx.meta_var(comm_m_idx // (LOCAL_M // TILE_M))
                sem.semaphore_notify(signal_rank, tid, comm_m_idx_local, n_idx, rs_queue)
        tile_scheduler.next_tile(cbx, bx, rank, warp_id_in_cta, lane_id)
    tmem_pool.dealloc()
    Tx.ptx.barrier.cluster.arrive()
    Tx.ptx.barrier.cluster.wait()


def build_kernel(config: GemmRSConfig | None = None) -> tvm.IRModule:
    """Return the directly ported fused persistent GemmRS kernel."""

    config = config or derive_config()
    if config != derive_config():
        raise ValueError(f"unsupported GemmRS specialization: {config!r}")
    return tvm.IRModule({FUSED_DEVICE_ENTRYPOINT: test_mma_ss_tma_2sm_persistent})


__all__ = [
    "CAPACITY",
    "DTYPE",
    "FUSED_DEVICE_ENTRYPOINT",
    "GROUP_SIZE",
    "SUPPORTED_WORLD_SIZES",
    "TASK_IDX_LEN",
    "TOTAL_K",
    "GemmRSConfig",
    "M",
    "N",
    "TaskType",
    "build_kernel",
    "derive_config",
]
