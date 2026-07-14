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

from enum import Enum

from tvm.backend.cuda.lang import RankAwareGroupMajorTileScheduler
from tvm.ir.type import PointerType, PrimType
from tvm.script import ir as I
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


# fmt: off
@I.ir_module(s_tir=True)
class ReduceScatter:
    @T.prim_func
    def test_mma_ss_tma_2sm_persistent(A: T.Buffer((M, K), a_type), B: T.Buffer((N, K), b_type), gemm_out: T.Buffer((M, N), d_type),
                                    semaphore: T.Buffer((WORLD_SIZE, ), "uint64"),
                                    out: T.Buffer((LOCAL_M, N), d_type), profiler_buffer: T.Buffer((PROFILER_BUFFER_SIZE,), "uint64")):
        A_tensor_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        B_tensor_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        C_tensor_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_tensor_map, "float16", 2, A.data, K, M, K * 2, BLK_K, BLK_M, 1, 1, 0, 3, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", B_tensor_map, "float16", 2, B.data, K, N, K * 2, BLK_K, BLK_N // 2, 1, 1, 0, 3, 0, 0)
        T.call_packed("runtime.cuTensorMapEncodeTiled", C_tensor_map, "float16", 2, gemm_out.data, N, M, N * 2, EPI_TILE, BLK_M, 1, 1, 0, 3, 0, 0)
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
            tmem_addr = T.decl_scalar("uint32", buf.data, scope="shared.dyn", elem_offset=0)
            A_smem = T.decl_buffer((PIPE_DEPTH, NUM_CONSUMER,BLK_M, BLK_K), a_type, buf.data, elem_offset=512, layout=A_layout)
            B_smem = T.decl_buffer((PIPE_DEPTH, BLK_N // 2, BLK_K), b_type, buf.data, elem_offset=512 + BLK_K * BLK_M * NUM_CONSUMER * PIPE_DEPTH, layout=B_layout)
            C_smem = T.decl_buffer((NUM_CONSUMER, BLK_M, EPI_TILE), d_type, buf.data, elem_offset=512 + BLK_K * BLK_M * NUM_CONSUMER * PIPE_DEPTH + BLK_K * BLK_N // 2 * PIPE_DEPTH, layout=D_layout)
            reg = T.alloc_buffer((TMEM_LD_SIZE,), "float32", scope="local")
            reg_wg = reg.view(128, TMEM_LD_SIZE, layout=TileLayout(T.S[(128, TMEM_LD_SIZE) : (1@tid_in_wg, 1)]))
            reg_fp16 = T.alloc_buffer((BLK_N,), d_type, scope="local")
            descA = T.local_scalar("uint64")
            descB = T.local_scalar("uint64")
            descI = T.local_scalar("uint32")
            base_desc_A = T.local_scalar("uint64")
            base_desc_B = T.local_scalar("uint64")
            tma2mma_pipe = T.meta_var(TMA2MMAPipeline(buf.data, 1, PIPE_DEPTH, 1, p_single_cta=False, c_single_cta=True))
            mma2ld_pipe = T.meta_var(MMA2LDpipeline(buf.data, 1 + PIPE_DEPTH * 2, 1, NUM_CONSUMER, p_single_cta=True, c_single_cta=False))
            mma2ld_pipe.init(tid == 0, c2p_thread_count=128 * 2, p2c_thread_count=2)
            tma2mma_pipe.init(tid == 0, c2p_thread_count=NUM_CONSUMER)
            ptr: T.let[T.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = T.reinterpret("handle", T.ptx.map_shared_rank(tma2mma_pipe.mbar_p2c.ptr_to([0, 0]), 0))
            tma_finished = T.decl_buffer([PIPE_DEPTH], "uint64", data=ptr, scope="shared")
            m_clusters = T.meta_var((M + BLK_M - 1) // BLK_M // CLUSTER_M // NUM_CONSUMER)
            n_clusters = T.meta_var((N + BLK_N - 1) // BLK_N // CLUSTER_N)
            gemm_tile_scheduler = T.meta_var(RankAwareGroupMajorTileScheduler("gemm_tile_scheduler", m_clusters, n_clusters, GROUP_SIZE, WORLD_SIZE))
            gemm_tile_scheduler.init(bx//2)
            # alloc TMEM
            if (wg_id == 0) & (warp_id == 0):
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=N_COLS, cta_group=cta_group)
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
                n_cta_groups=cta_group,
            )
            T.cuda.cta_sync()
            T.cuda.trap_when_assert_failed(tmem_addr == 0)
            tmem = T.decl_buffer((128, N_COLS), "float32", scope="tmem", allocated_addr=0,
                                 layout=TileLayout(T.S[(128, N_COLS) : (1@TLane, 1@TCol)]))
            # reset RF
            if wg_id == NUM_CONSUMER:
                T.ptx.setmaxnreg(False, 56)
                if warp_id == 3:
                    while gemm_tile_scheduler.valid():
                        m_idx = T.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                        n_idx = T.meta_var(gemm_tile_scheduler.n_idx)
                        for ko in range(K // BLK_K):
                            # GMEM -> SMEM
                            if lane_id == 0:
                                tma2mma_pipe.producer_wait(0)
                                T.ptx.cp_async.bulk.tensor.g2s_cluster(2, A_smem.ptr_to([tma2mma_pipe.idx, 0, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), T.address_of(A_tensor_map), 0, 2, "", ko * BLK_K, (m_idx * 4 + cbx) * BLK_M)
                                T.ptx.cp_async.bulk.tensor.g2s_cluster(2, A_smem.ptr_to([tma2mma_pipe.idx, 1, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), T.address_of(A_tensor_map), 0, 2, "", ko * BLK_K, (m_idx * 4 + 2 + cbx) * BLK_M)
                                T.ptx.cp_async.bulk.tensor.g2s_cluster(2, B_smem.ptr_to([tma2mma_pipe.idx, 0, 0]), tma_finished.ptr_to([tma2mma_pipe.idx]), T.address_of(B_tensor_map), 0, 2, "", ko * BLK_K, n_idx * BLK_N + cbx * BLK_N // 2)
                                if cbx == 0:
                                    # notify the mma stage that tma load is finished
                                    T.ptx.mbarrier.arrive.expect_tx(tma_finished.ptr_to([tma2mma_pipe.idx]), (BLK_K * BLK_M * 2 * 2 + BLK_K * BLK_N) * 2)
                                tma2mma_pipe.advance()
                        gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
                    for i in range(PIPE_DEPTH):
                        # wait for the completion of all the mma of the last tile
                        if lane_id == 0:
                            tma2mma_pipe.producer_wait(0)
                            tma2mma_pipe.advance()
                elif warp_id < NUM_CONSUMER:
                    while gemm_tile_scheduler.valid():
                        m_idx = T.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                        n_idx = T.meta_var(gemm_tile_scheduler.n_idx)
                        # MMA
                        if lane_id == 0 and cbx == 0:
                            # wait for the last tmem result to be consumed
                            mma2ld_pipe.producer_wait(warp_id)
                            for ko in T.serial(0, K // BLK_K):
                                # wait for tma load to finish
                                tma2mma_pipe.consumer_wait(0)
                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(base_desc_A), A_smem.ptr_to([tma2mma_pipe.idx, warp_id, 0, 0]), ldo=ldo, sdo=sdo, swizzle=SWIZZLE)
                                T.ptx.tcgen05.encode_matrix_descriptor(T.address_of(base_desc_B), B_smem.ptr_to([tma2mma_pipe.idx, 0, 0]), ldo=ldo, sdo=sdo, swizzle=SWIZZLE)
                                for ki in range(BLK_K // MMA_K):
                                    descA = base_desc_A + ((ki * MMA_K * 2) >> 0x4)
                                    descB = base_desc_B + ((ki * MMA_K * 2) >> 0x4)
                                    if ki == 0 and ko == 0:
                                        T.ptx.tcgen05.mma(tmem_addr + warp_id * MMA_N, descA, descB, descI, d_dtype="float32", a_dtype=a_type, b_dtype=b_type, use_a_tmem=False, cta_group=cta_group, enable_input_d=False)
                                    else:
                                        T.ptx.tcgen05.mma(tmem_addr + warp_id * MMA_N, descA, descB, descI, d_dtype="float32", a_dtype=a_type, b_dtype=b_type, use_a_tmem=False, cta_group=cta_group, enable_input_d=True)
                                tma2mma_pipe.consumer_release(0)
                                tma2mma_pipe.advance()
                            T.ptx.tcgen05.commit(mma2ld_pipe.mbar_p2c.ptr_to([0, 0]), cta_group=2, cta_mask=3)
                            mma2ld_pipe.advance()
                        gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
            if wg_id < NUM_CONSUMER:
                T.ptx.setmaxnreg(True, 224)
                while gemm_tile_scheduler.valid():
                    m_idx = T.meta_var(gemm_tile_scheduler.m_idx) # represent cluster task id
                    n_idx = T.meta_var(gemm_tile_scheduler.n_idx)
                    # wait for the completion of all the mma of the same tile
                    mma2ld_pipe.consumer_wait(0)
                    # TMEM -> RF
                    for i in range(BLK_N // TMEM_LD_SIZE):
                        col_st = T.meta_var(wg_id * MMA_N + i * TMEM_LD_SIZE)
                        Tx.wg.copy_async(reg_wg[:, :], tmem[:, col_st : col_st + TMEM_LD_SIZE])
                        Tx.cast(reg_fp16[i * TMEM_LD_SIZE : (i + 1) * TMEM_LD_SIZE], reg[:])
                    # the tmem can be overwritten by the next tile
                    mma2ld_pipe.consumer_release(wg_id)
                    # RF -> GMEM
                    for i in range(BLK_N // EPI_TILE):
                        Tx.copy(C_smem[wg_id, warp_id * 32 + lane_id, :], reg_fp16[i * EPI_TILE : (i + 1) * EPI_TILE])
                        T.cuda.warpgroup_sync(wg_id+1)
                        T.ptx.fence.proxy_async("shared::cta")
                        if lane_id == 0 and warp_id == 0:
                            T.ptx.cp_async.bulk.tensor.s2g(2, C_smem.ptr_to([wg_id, 0, 0]), T.address_of(C_tensor_map), "", n_idx * BLK_N + i * EPI_TILE, (m_idx * 4 + wg_id * 2 + cbx) * BLK_M)
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                        T.cuda.warpgroup_sync(wg_id+1)
                    # notify RS ready
                    comm_m_idx = T.meta_var(m_idx * 4 + wg_id * 2 + cbx)
                    signal_rank = T.meta_var(comm_m_idx // (LOCAL_M // BLK_M))
                    sem.semaphore_notify(tid, signal_rank)
                    mma2ld_pipe.advance()
                    gemm_tile_scheduler.next_tile(stride=GEMM_SMS // 2)
            # dealloc TMEM
            if (wg_id == 0) & (warp_id == 0):
                T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
                T.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=cta_group)
            profiler.end(ProfileEventType.GEMM, lane_id == 0)


    @T.prim_func
    def reduce_sum(
        staging_buffer: T.Buffer((WORLD_SIZE, LOCAL_M, N), "float16"),
        out: T.Buffer((LOCAL_M, N), d_type),
    ):
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
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([2])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        tid_in_wg = T.thread_id_in_wg([128])
        tid = T.thread_id([256])
        buf = T.alloc_buffer([SMEM_SIZE], "uint8", scope="shared.dyn")
        load_pipe = T.meta_var(
            ReducePipe(
                buf.data, 0, RS_LOAD_PIPE_DEPTH, 1, p_single_cta=False, c_single_cta=False
            )
        )
        input_smem = T.decl_buffer(
            (RS_LOAD_PIPE_DEPTH, BLK_M_RS, BLK_N_RS),
            d_type,
            buf.data,
            elem_offset=512,
        )
        output_smem = T.decl_buffer(
            (BLK_M_RS, BLK_N_RS),
            d_type,
            buf.data,
            elem_offset=512 + RS_LOAD_PIPE_DEPTH * BLK_M_RS * BLK_N_RS,
        )
        reg_fp16 = T.alloc_buffer((8, ), "float16", scope="local")
        reg_fp32_tmp = T.alloc_buffer((8), "float32", scope="local")
        reg_fp32 = T.alloc_buffer((BLK_M_RS * BLK_N_RS // 8 // 128, 8), "float32", scope="local")
        iter = T.local_scalar("int32")
        iter = 0
        load_pipe.init(tid == 0, c2p_thread_count=128)
        tile_id = T.meta_var(iter * SM_COUNT + bx)
        T.tvm_storage_sync("shared")
        if warp_id == 0 and wg_id == 0:
            while tile_id < LOCAL_M // BLK_M_RS * N // BLK_N_RS:
                m_idx = T.meta_var(tile_id // (N // BLK_N_RS))
                n_idx = T.meta_var(tile_id % (N // BLK_N_RS))
                if lane_id == 0:
                    for i in range(WORLD_SIZE):
                        load_pipe.producer_wait(0)
                        T.ptx.cp_async.bulk.tensor.g2s_cta(
                            3,
                            input_smem.ptr_to([load_pipe.idx, 0, 0]),
                            load_pipe.mbar_p2c.ptr_to([load_pipe.idx, 0]),
                            T.address_of(src_tensor_map),
                            1,
                            "",
                            n_idx * BLK_N_RS,
                            m_idx * BLK_M_RS,
                            i,
                        )
                        T.ptx.mbarrier.arrive.expect_tx(
                            load_pipe.mbar_p2c.ptr_to([load_pipe.idx, 0]),
                            BLK_M_RS * BLK_N_RS * 2,
                        )
                        load_pipe.advance()
                iter += 1
        elif wg_id == 1:
            while tile_id < LOCAL_M // BLK_M_RS * N // BLK_N_RS:
                m_idx = T.meta_var(tile_id // (N // BLK_N_RS))
                n_idx = T.meta_var(tile_id % (N // BLK_N_RS))
                for rank in range(WORLD_SIZE):
                    load_pipe.consumer_wait(0)
                    for i in range(BLK_M_RS * BLK_N_RS // 8 // 128):
                        m_in_smem = T.meta_var((i * 128 + tid_in_wg) // (BLK_N_RS // 8))
                        n_in_smem = T.meta_var((i * 128 + tid_in_wg) % (BLK_N_RS // 8))
                        for j in T.vectorized(8):
                            reg_fp16[j] = input_smem[
                                load_pipe.idx, m_in_smem, n_in_smem * 8 + j
                            ]
                        if rank > 0:
                            T.cuda.half8tofloat8(reg_fp16.data, reg_fp32_tmp.data)
                            for j in T.vectorized(8):
                                reg_fp32[i, j] += reg_fp32_tmp[j]
                        else:
                            T.cuda.half8tofloat8(reg_fp16.data, reg_fp32.ptr_to([i, 0]))
                    load_pipe.consumer_release(0)
                    load_pipe.advance()
                for i in range(BLK_M_RS * BLK_N_RS // 8 // 128):
                    m_in_smem = T.meta_var((i * 128 + tid_in_wg) // (BLK_N_RS // 8))
                    n_in_smem = T.meta_var((i * 128 + tid_in_wg) % (BLK_N_RS // 8))
                    T.cuda.float8tohalf8(reg_fp32.ptr_to([i, 0]), reg_fp16.data)
                    for j in T.vectorized(8):
                        output_smem[m_in_smem, n_in_smem * 8 + j] = reg_fp16[j]
                T.cuda.warpgroup_sync(1)
                T.ptx.fence.proxy_async("shared::cta")
                if tid_in_wg == 0:
                    T.ptx.cp_async.bulk.tensor.s2g(
                        2,
                        output_smem.ptr_to([0, 0]),
                        T.address_of(dst_tensor_map),
                        "",
                        n_idx * BLK_N_RS,
                        m_idx * BLK_M_RS,
                    )
                    T.ptx.cp_async.bulk.commit_group()
                    T.ptx.cp_async.bulk.wait_group(0)
                T.cuda.warpgroup_sync(1)
                iter += 1
# fmt: on
