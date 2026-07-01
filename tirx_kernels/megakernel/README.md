# Megakernel Tile Tasks and Scheduling

This directory currently contains the minimal event-dependency megakernel
implementation used by `tirx_kernels.megakernel.moe.MegaKernelMOE`.
It is not an end-to-end MLC engine integration layer.  The public kernel entry
point is the repo-standard `megakernel_moe` module, with correctness and
benchmark coverage exposed through `run_test` and `run_bench`.

## Tile task inventory

The current MoE workload models the Qwen3-30B-A3B MoE shape:

| Parameter | Value |
| --- | --- |
| hidden size | 2048 |
| intermediate size | 768 |
| experts | 128 |
| top-k | 8 |
| gate split-k | 4 |

The megakernel decomposes the full MoE layer into these tile tasks:

| JobType | Tile class | Work computed by the tile |
| --- | --- | --- |
| `INIT_ETENSOR` | `InitETensorTile` | Initializes the event tensors/semaphores used by the scheduled tasks. |
| `MOE_GATING` | `GemmTile` | Computes the router logits, `hidden_state @ gate_weight.T`, into `gating_output`. |
| `MOE_TOPK_SOFTMAX` | `TopkSoftmaxTile` | Applies softmax over experts and writes top-k weights and expert indices. |
| `MOE_ALIGN` | `MOEAlignTile` | Counts routed tokens per expert, pads each expert segment to `MOE_M_PAD_SIZE`, writes expert block metadata, and writes `num_tokens_post_pad`. |
| `MOE_COUNT_AND_SORT` | `CountAndSortExpertTokens` | Writes `sorted_token_ids` and reorders hidden states into expert-major token order. |
| `MOE_GROUP_GEMM_GATE_UP_SILU` | `GroupGEMMSiluTile` | Runs the grouped gate/up projection and fused SiLU/mul, producing the intermediate activation. |
| `MOE_GROUP_GEMM_DOWN` | `GroupGEMMTileSM100` | Runs the grouped down projection and reduces top-k expert contributions into `topk_reduce_output`. |
| `WAIT_ETENSOR_INIT` | wrapper task | Static-scheduler-only synchronization that waits until event tensor initialization is globally visible. |
| `END` | scheduler sentinel | Terminates each scheduler work stream. |

## Tile task interface

There are two layers of interface.  A tile describes how one compute block
runs.  A wrapper task describes how that tile participates in a scheduled
event-dependency graph.

### Tile compute contract

Every compute tile inherits from `utils.base.Tile` and follows this shape:

```python
class Tile:
    need_init = True

    @classmethod
    def class_init(cls, smem_manager): ...
    @classmethod
    def class_finalize(cls): ...

    def init(self, smem_manager): ...
    def host_init(self): ...
    def run(self, m_idx, n_idx, k_idx, *args, **kwargs): ...
    def prefetch(self, m_idx, n_idx, k_idx, *args): ...
```

The required scheduler-visible index tuple is `(m_idx, n_idx, k_idx)`.
The meaning is tile-specific but consistent enough for the schedulers:

- `m_idx` selects a token/block row, persistent-SM shard, or expert-major block.
- `n_idx` selects the output-column tile for GEMM-like tasks.
- `k_idx` selects split-k work when the task has a k dimension.

Tiles also expose static metadata through class or instance attributes.  The
current scheduler and task generation code uses values such as `BLK_M`,
`BLK_N`, `M_pad_size`, `PERSISTENT_SM_NUMBER`, and `need_init` to determine
task counts, queue entries, shared-memory setup, and class-level resource
initialization.

### Scheduled task contract

The actual scheduler contract is not a standalone Python graph object yet.
It is the combination of:

1. `JobType`: the integer task type stored in the static queue or dynamic MPMC
   queue.
2. Packed task tuple: `(m_idx, n_idx, k_idx, task_type)`.
3. Event tensors: semaphore buffers created by `MegaKernelWrapper.add_etensor`.
4. `task_impl_*` methods in `MegaKernelMOE`: the binding layer that performs
   `wait`, calls `run_tile`, performs `notify`, and, for dynamic scheduling,
   registers the follow-up work through `pre_notify_and_push`.

The wrapper registers tiles with `_add_tile(tile, ProfileEventType.*)`.  This
connects a tile object to profiling metadata and class-level initialization.
`run_tile` then brackets `tile.run(...)` with profiler events and shared-memory
runtime bookkeeping.

The schedulers do not inspect tile internals.  They only understand task
tuples, event tensors, and the `wait`/`notify`/`pre_notify_and_push` calls made
by wrapper task implementations.

For a new task to be schedulable, it must provide:

- a `JobType` value;
- a tile class implementing the compute contract;
- an event tensor for each cross-task dependency that must be enforced;
- a `task_impl_*` wrapper that declares the task's waits and notifies;
- a static queue entry generator when the static scheduler must execute it;
- a dynamic push rule when the dynamic scheduler must create successor tasks.

## User event dependency in fused MoE

For the fused MoE kernel, the user-level dependency graph is:

```text
INIT_ETENSOR
  -> MOE_GATING
  -> MOE_TOPK_SOFTMAX
  -> MOE_ALIGN
  -> MOE_COUNT_AND_SORT
  -> MOE_GROUP_GEMM_GATE_UP_SILU
  -> MOE_GROUP_GEMM_DOWN
  -> END
```

The graph is encoded in `MegaKernelMOE._set_events` and the `task_impl_*`
methods, not in a separate declarative DAG file.

The event tensors are:

| Event | Initial count | Meaning |
| --- | --- | --- |
| `evt_gating` | `GATING_SPLIT_K_FACTOR * ceildiv(batch_size, GATING_BLK_M)` | All split-k gating tiles must finish before top-k/softmax reads `gating_output`. |
| `evt_topk_softmax` | `KernelConfig.SM_NUMBER` | All persistent top-k/softmax shards must finish before align reads top-k data. |
| `evt_moe_align` | `1` | The single align task must finish before count/sort reads routing layout metadata. |
| `evt_count_and_sort` | `KernelConfig.SM_NUMBER` | All count/sort shards must finish before grouped GEMM reads expert-major hidden states. |
| `evt_group_gemm_gate_up` | one count per expert-major block, or one global count in unfused mode | All gate/up/silu column tiles for a block must finish before down projection reads that block. |
| `evt_group_gemm_down` | static: allocated with the full down task count; dynamic: initialized after align computes `num_tokens_post_pad` | Dynamic scheduling uses this to push `END` after all down projection tasks finish. |

The MoE wrapper expresses dependencies as local task code.  For example:

- `MOE_TOPK_SOFTMAX` waits on `evt_gating`, runs `TopkSoftmaxTile`, then
  notifies `evt_topk_softmax`.
- `MOE_ALIGN` waits on `evt_topk_softmax`, runs `MOEAlignTile`, initializes the
  dynamic down-projection event count from `num_tokens_post_pad`, then notifies
  `evt_moe_align`.
- `MOE_COUNT_AND_SORT` waits on `evt_moe_align`, runs
  `CountAndSortExpertTokens`, then notifies `evt_count_and_sort`.
- `MOE_GROUP_GEMM_GATE_UP_SILU` waits on `evt_count_and_sort`, runs the grouped
  gate/up/silu tile, then notifies `evt_group_gemm_gate_up[m_idx]`.
- `MOE_GROUP_GEMM_DOWN` waits on `evt_group_gemm_gate_up[m_idx]`, runs the
  grouped down tile, and in dynamic mode contributes to the final END trigger.

This is the part a user changes when describing a different event-dependency
megakernel: the tile code can stay local to each task, while the wrapper task
methods define the ordering, fan-in counts, and fan-out task creation.

## Static scheduler

The static scheduler consumes a host-built queue generated by
`generate_exec_queue_moe(..., scheduler="static")`.

The host queue contains the complete task sequence:

1. all `INIT_ETENSOR` tasks;
2. all `MOE_GATING` split-k tiles;
3. `WAIT_ETENSOR_INIT` tasks, one per SM, so subsequent tasks observe initialized
   event tensors;
4. all persistent `MOE_TOPK_SOFTMAX` tasks;
5. the single `MOE_ALIGN` task;
6. all persistent `MOE_COUNT_AND_SORT` tasks;
7. all possible `MOE_GROUP_GEMM_GATE_UP_SILU` tasks for the maximum padded token
   count;
8. all possible `MOE_GROUP_GEMM_DOWN` tasks for the maximum padded token count;
9. `END` sentinels.

`StaticTileScheduler` loads a per-SM slice of this queue into persistent shared
memory, unpacks each `(m_idx, n_idx, k_idx, task_type)`, executes the matching
`task_impl_*`, and advances linearly.  The host queue order is a work ordering,
not a global barrier across SMs; cross-task ordering must still be represented
by event waits.

Correctness comes from the event tensors, not from assuming the queue order is
alone sufficient.  Each consumer waits on its producer event before reading the
producer's output.  Each producer notifies the corresponding event only after
its tile has completed and the scheduler-level synchronization scope has run.
Tasks that were inserted for the maximum padded shape check
`num_tokens_post_pad` before running tile code, so statically scheduled padded
work does not read or write invalid token blocks.

`unfused` is not a third scheduler.  It uses the static scheduler path with the
same task implementations and a different gate/up event shape so it can serve
as the unfused comparison mode.

## Dynamic scheduler

The dynamic scheduler consumes an MPMC queue.  Its host-side initial queue
starts with `INIT_ETENSOR` and gating tasks:

```text
INIT_ETENSOR*, MOE_GATING*
```

Every later task is created on device by `pre_notify_and_push`.

Dynamic semaphore notification is two-phase.  Before a tile runs,
`pre_notify_and_push` performs a pre-notify and checks whether that notification
is the one that should trigger successor work.  This lets successor tasks be
queued as soon as the last producer has been dispatched, while the later normal
`notify` still protects the actual consumer wait condition.  This avoids
deadlock while preserving the same event-dependency correctness as the static
scheduler.

For fused MoE, dynamic push rules are:

| Producer event | Triggered successor |
| --- | --- |
| `evt_gating` | `KernelConfig.SM_NUMBER` `MOE_TOPK_SOFTMAX` tasks |
| `evt_topk_softmax` | one `MOE_ALIGN` task |
| `evt_moe_align` | `KernelConfig.SM_NUMBER` `MOE_COUNT_AND_SORT` tasks |
| `evt_count_and_sort` | `num_tokens_post_pad / MOE_M_PAD_SIZE * (2 * intermediate / BLK_N)` gate/up/silu tasks |
| `evt_group_gemm_gate_up[m_idx]` | down-projection tasks for that `m_idx` block |
| `evt_group_gemm_down` | `END` tasks, one per SM |

The dynamic scheduler therefore uses runtime routing information.  `MOE_ALIGN`
computes `num_tokens_post_pad`; later push rules use that value to enqueue only
the expert-major blocks that actually exist for the current input.

## Why the generated program is correct

Both schedulers use the same task implementations and the same event tensors.
They differ only in when task tuples appear in the scheduler queue.

The generated program is correct when these invariants hold:

- The event initial count equals the number of producer task notifications that
  must happen before a consumer may run.
- Every consumer waits on the event protecting each input buffer it reads.
- Every producer notifies after it has finished writing the protected output.
- Dynamic push rules enqueue exactly the successor task set implied by the
  event trigger and runtime shape values.
- Static queue generation covers the maximum possible task set, and task
  implementations guard runtime-invalid padded blocks.
- Event tensor initialization completes before any task can rely on the
  corresponding semaphore value.  In practice this means producer notifications
  must not race with initialization, and static consumers use
  `WAIT_ETENSOR_INIT` before they rely on post-gating events.

Under these invariants, static scheduling and dynamic scheduling produce the
same logical MoE output for the same inputs.  The dynamic scheduler is allowed
to issue less work because it learns the routed token count at runtime, but it
does not change the dependency graph or the computed result.

## Adding another event-dependency megakernel

When adding a new megakernel, keep the current split of responsibilities:

1. Implement the tile's local computation in a `Tile` subclass.
2. Expose block sizes and persistent-SM/task-count metadata as class or instance
   attributes where the queue generator or wrapper needs them.
3. Add `JobType` and `ProfileEventType` entries.
4. Register the tile in the wrapper with `_add_tile`.
5. Allocate event tensors in `_set_events` with fan-in counts that match the
   producer task counts.
6. Write one `task_impl_*` per task type to bind `wait -> run_tile -> notify`.
7. Add `pre_notify_and_push` rules for dynamic scheduling.
8. Add static queue generation for the same task graph.
9. Keep correctness and benchmark entry points in the kernel module:
   `prepare_data`, `run_test`, and `run_bench`.
