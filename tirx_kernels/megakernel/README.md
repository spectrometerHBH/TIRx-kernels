# MegaKernelMOE Task Dataflow and Scheduling

This directory contains the minimal event-dependency megakernel used by
`tirx_kernels.megakernel.moe.MegaKernelMOE`.  The kernel is a standalone MoE
kernel benchmark, not an MLC engine integration layer.

The important thing to understand first is that a tile task is not just a
piece of compute.  A useful task specification must say:

1. what task instances exist;
2. what buffers each instance reads;
3. what buffers each instance writes;
4. what event makes those reads legal;
5. what event it notifies after those writes are complete;
6. what successor tasks the dynamic scheduler should enqueue.

The current implementation does not store that as a separate declarative
`TaskSpec` object.  It encodes the spec in `MegaKernelMOE.task_impl_*`,
`MegaKernelMOE._set_events`, and `generate_exec_queue_moe`.  This document
spells out that implicit contract.

## Shape vocabulary

The current benchmark models the Qwen3-30B-A3B MoE shape:

| Symbol | Meaning | Value |
| --- | --- | --- |
| `B` | input tokens / batch size | runtime config |
| `H` | hidden size | 2048 |
| `I` | intermediate size | 768 |
| `E` | number of experts | 128 |
| `K` | top-k experts per token | 8 |
| `P` | MoE expert block size / `MOE_M_PAD_SIZE` | 128 |
| `S` | gating split-k factor | 4 |
| `BN` | grouped GEMM output tile columns / `GroupGEMMTileSM100.BLK_N` | 128 |
| `T` | routed token count before padding | `B * K` |
| `TP` | routed token count after expert padding | `num_tokens_post_pad[0]` |
| `TP_MAX` | maximum possible padded token count | `get_max_num_tokens_padded(B, K, E, P)` |
| `NB` | runtime expert-major token blocks | `TP / P` |

For small batches where `B * K < E`, `TP_MAX = B * K * P`, because each active
route may occupy its own padded expert block.  Otherwise each expert gets a
block and the remaining routed tokens are rounded up by `P`.

## Buffer vocabulary

These are the logical buffers that connect the tasks:

| Buffer | Shape | DType | Producer | Consumers | Meaning |
| --- | --- | --- | --- | --- | --- |
| `hidden_state` | `[B, H]` | fp16 | kernel input | `MOE_GATING`, `MOE_COUNT_AND_SORT` | Input token hidden states. |
| `gate_weight` | `[E, H]` | fp16 | kernel input | `MOE_GATING` | Router weight. |
| `grp_gate_up_weight` | `[E, 2 * I, H]` | fp16 | kernel input | `MOE_GROUP_GEMM_GATE_UP_SILU` | Expert gate/up weights.  The benchmark passes the shuffled layout expected by this TIR path. |
| `grp_down_weight` | `[E, H, I]` | fp16 | kernel input | `MOE_GROUP_GEMM_DOWN` | Expert down-projection weights. |
| `gating_output` | `[B, E]` | fp32 | `MOE_GATING` | `MOE_TOPK_SOFTMAX` | Router logits for every token/expert pair. |
| `topk_weights` | `[B, K]` | fp32 | `MOE_TOPK_SOFTMAX` | `MOE_GROUP_GEMM_DOWN` | Routing probability for each selected route. |
| `topk_indices` | `[B, K]` | int32 | `MOE_TOPK_SOFTMAX` | `MOE_ALIGN`, `MOE_COUNT_AND_SORT` | Expert id for each selected route.  The task code uses `topk_indices.view(-1)` as flattened expert ids, one per route. |
| `num_tokens_post_pad` | `[1]` | int32 | `MOE_ALIGN` | schedulers, grouped GEMM tasks | Runtime value `TP`, the padded expert-major routed token count. |
| `cumsum_buffer` | `[E + 1]` | int32 | `MOE_ALIGN` | `MOE_COUNT_AND_SORT` | Expert prefix offsets.  `MOE_COUNT_AND_SORT` mutates it with atomics while assigning routed rows. |
| `expert_ids` | `[TP_MAX / P]` | int32 | `MOE_ALIGN` | grouped GEMM tasks | `expert_ids[block]` gives the expert for expert-major block `block`. |
| `num_valid_tokens` | `[TP_MAX / P]` | int32 | `MOE_ALIGN` | grouped GEMM tasks | Valid rows inside each padded expert block.  Used when dynamic GEMM size is enabled. |
| `sorted_token_ids` | `[TP_MAX]` | int32 | `MOE_ALIGN`, `MOE_COUNT_AND_SORT` | grouped GEMM tasks | Maps expert-major row to flattened route id `token * K + route`.  Padding rows hold sentinel `T`. |
| `reordered_hidden_state` | `[TP_MAX, H]` | fp16 | `MOE_COUNT_AND_SORT` | `MOE_GROUP_GEMM_GATE_UP_SILU` | Hidden states reordered into expert-major route order. |
| `silu_mul_output` | `[TP_MAX, I]` | fp16 | `MOE_GROUP_GEMM_GATE_UP_SILU` | `MOE_GROUP_GEMM_DOWN` | Expert intermediate activation after `silu(gate) * up`. |
| `topk_reduce_output` | `[B, H]` | fp16 | `MOE_GROUP_GEMM_DOWN` | benchmark validation | Final MoE output accumulated across the top-k experts. |
| `gate_up_output` | `[TP_MAX, 2 * I]` | fp16 | unused by current fused path | none | Allocated workspace kept in the signature; the fused gate/up/silu task writes `silu_mul_output` directly. |
| `residual`, `output` | `[B, H]` | fp16 | unused by current minimal MoE path | none | Kept in the kernel signature, but not part of the current task dataflow. |
| `etensor_workspace` | implementation-defined | int32 | event init task | schedulers | Stores event semaphore counters. |

The main dataflow is:

```text
hidden_state, gate_weight
  -> MOE_GATING
  -> gating_output
  -> MOE_TOPK_SOFTMAX
  -> topk_weights, topk_indices
  -> MOE_ALIGN
  -> num_tokens_post_pad, cumsum_buffer, expert_ids, num_valid_tokens, padded sorted_token_ids
  -> MOE_COUNT_AND_SORT
  -> sorted_token_ids, reordered_hidden_state
  -> MOE_GROUP_GEMM_GATE_UP_SILU
  -> silu_mul_output
  -> MOE_GROUP_GEMM_DOWN
  -> topk_reduce_output
```

## What a task specification needs

Use this mental model when deciding whether two tasks can be connected:

```python
TaskSpec(
    job_type=...,
    task_space=...,       # set of (m_idx, n_idx, k_idx) instances
    reads=[...],          # buffers that must be ready before run()
    writes=[...],         # buffers produced or mutated by run()
    waits=[...],          # event counters protecting reads
    notifies=[...],       # event counters decremented after writes complete
    dynamic_successors=[...]  # tasks enqueued when notify reaches trigger
)
```

Two tasks are connectable only when the producer's `writes` match the
consumer's `reads`, and the producer's `notifies` match the consumer's
`waits`.  For dynamic scheduling, the producer also needs a successor rule that
creates the consumer's task instances.

The current code maps that model as follows:

| Spec field | Current implementation |
| --- | --- |
| `job_type` | `utils.config.JobType` |
| `task_space` | static: `generate_exec_queue_moe`; dynamic: initial queue plus `pre_notify_and_push` |
| `reads` / `writes` | `task_impl_*` argument list and the tile `run(...)` call |
| `waits` | `self.tile_scheduler.wait(...)` inside `task_impl_*` |
| `notifies` | `self.tile_scheduler.notify(...)` inside `task_impl_*` |
| dynamic successors | `self.tile_scheduler.pre_notify_and_push(...)` inside `task_impl_*` |

## Task specifications

### `INIT_ETENSOR`

Task space:

```text
m_idx in [0, num_event_tensors)
n_idx = 0
k_idx = 0
```

Reads:

- no dataflow buffers.

Writes:

- initializes event counters inside `etensor_workspace`.

Connection:

- This task is scheduler infrastructure.  It must run before other tasks rely
  on the corresponding event counter value.
- Static scheduling inserts `WAIT_ETENSOR_INIT` tasks before post-gating work
  that depends on initialized event counters.

### `MOE_GATING`

Task space:

```text
m_idx in [0, ceildiv(B, 128))
n_idx = 0
k_idx in [0, S)
```

Reads:

- `hidden_state[B, H]`
- `gate_weight[E, H]`

Writes:

- `gating_output[B, E]`

Compute:

- Computes router logits:

```text
gating_output[token, expert] = dot(hidden_state[token, :], gate_weight[expert, :])
```

- The `k_idx` dimension is split-k over hidden size.  All split-k tasks for all
  token blocks jointly produce the final `gating_output`.

Synchronization:

- Waits: none.
- Notifies: `evt_gating`.
- `evt_gating` initial count is `S * ceildiv(B, 128)`.

Dynamic successor:

- When all gating tasks have been dispatched, enqueue
  `MOE_TOPK_SOFTMAX` for `m_idx in [0, KernelConfig.SM_NUMBER)`.

Connects to:

- `MOE_TOPK_SOFTMAX`, because that task reads `gating_output` and waits on
  `evt_gating`.

### `MOE_TOPK_SOFTMAX`

Task space:

```text
m_idx in [0, KernelConfig.SM_NUMBER)
n_idx = 0
k_idx = 0
```

Reads:

- `gating_output[B, E]`

Writes:

- `topk_weights[B, K]`
- `topk_indices[B, K]`

Compute:

- For each token row assigned to this persistent task shard:

```text
routing = softmax(gating_output[token, :])
topk_weights[token, :], topk_indices[token, :] = topk(routing, K)
```

- The current call uses `renormalize=False`, so top-k weights are the softmax
  probabilities from the full expert distribution, not re-normalized across
  the selected `K` experts.

Synchronization:

- Waits: `evt_gating`.
- Notifies: `evt_topk_softmax`.
- `evt_topk_softmax` initial count is `KernelConfig.SM_NUMBER`.

Dynamic successor:

- When all persistent top-k/softmax shards have been dispatched, enqueue one
  `MOE_ALIGN` task.

Connects to:

- `MOE_ALIGN`, because align reads `topk_indices` and waits on
  `evt_topk_softmax`.
- `MOE_GROUP_GEMM_DOWN` later reads `topk_weights`, protected indirectly by
  later events in the chain.

### `MOE_ALIGN`

Task space:

```text
m_idx = 0
n_idx = 0
k_idx = 0
```

Reads:

- `topk_indices.view(-1)`, length `T = B * K`

Writes:

- `num_tokens_post_pad[0] = TP`
- `cumsum_buffer[0:E + 1]`
- `expert_ids[0:TP / P]`
- `num_valid_tokens[0:TP / P]`, when that metadata is requested
- `sorted_token_ids[0:TP]`, initially filled with sentinel `T`

Compute:

1. Count how many routed token entries go to each expert.
2. Round each expert count up to a multiple of `P`.
3. Prefix-sum the padded counts into `cumsum_buffer`.
4. Write `num_tokens_post_pad[0]`.
5. For each padded expert block, write the owning `expert_ids[block]`.
6. Optionally write `num_valid_tokens[block]`.
7. Fill padded `sorted_token_ids` slots with sentinel `T`.

Synchronization:

- Waits: `evt_topk_softmax`.
- Notifies: `evt_moe_align`.
- `evt_moe_align` initial count is `1`.

Dynamic scheduler side effect:

- Initializes the dynamic `evt_group_gemm_down` counter after `TP` is known.

Dynamic successor:

- Enqueue `MOE_COUNT_AND_SORT` for
  `m_idx in [0, KernelConfig.SM_NUMBER)`.

Connects to:

- `MOE_COUNT_AND_SORT`, because it reads `topk_indices`,
  `sorted_token_ids`, `cumsum_buffer`, and waits on `evt_moe_align`.
- Grouped GEMM tasks later read `expert_ids`, `num_valid_tokens`, and
  `num_tokens_post_pad`; they are protected by `evt_count_and_sort` and the
  gate/up event.

### `MOE_COUNT_AND_SORT`

Task space:

```text
m_idx in [0, KernelConfig.SM_NUMBER)
n_idx = 0
k_idx = 0
```

Reads:

- `topk_indices.view(-1)`, length `T`
- `cumsum_buffer[E + 1]`
- `hidden_state[B, H]`

Writes:

- `sorted_token_ids[0:TP]`
- `reordered_hidden_state[0:TP, H]`
- mutates `cumsum_buffer` while assigning output rows

Compute:

- Each task shard processes route ids:

```text
route_id = m_idx + tid * KernelConfig.SM_NUMBER
token_id = route_id // K
expert_id = topk_indices.view(-1)[route_id]
row = atomic_add(cumsum_buffer[expert_id], 1)
sorted_token_ids[row] = route_id
reordered_hidden_state[row, :] = hidden_state[token_id, :]
```

- The result is expert-major route order.  Padding rows remain sentinel-filled
  from `MOE_ALIGN`.

Synchronization:

- Waits: `evt_moe_align`.
- Notifies: `evt_count_and_sort`.
- `evt_count_and_sort` initial count is `KernelConfig.SM_NUMBER`.

Dynamic successor:

- After all count/sort shards have been dispatched, enqueue exactly the runtime
  gate/up/silu task space:

```text
m_idx in [0, TP / P)
n_idx in [0, (2 * I) / BN)
k_idx = 0
```

Connects to:

- `MOE_GROUP_GEMM_GATE_UP_SILU`, because that task reads
  `reordered_hidden_state`, `sorted_token_ids`, `expert_ids`,
  `num_tokens_post_pad`, and `num_valid_tokens`, and waits on
  `evt_count_and_sort`.

### `MOE_GROUP_GEMM_GATE_UP_SILU`

Task space:

```text
m_idx in [0, TP / P)                      # expert-major token block
n_idx in [0, (2 * I) / BN)                # 128-column gate/up tile
k_idx = 0
```

Static scheduling creates this task space for `TP_MAX / P`; runtime-invalid
blocks check `m_idx < num_tokens_post_pad[0] / P` before running.

Reads:

- `reordered_hidden_state[TP, H]`
- `grp_gate_up_weight[E, 2 * I, H]`
- `expert_ids[TP / P]`
- `sorted_token_ids[TP]`
- `num_tokens_post_pad[1]`, in the wrapper's runtime-valid block check
- `num_valid_tokens[TP / P]`, when dynamic GEMM size is enabled

The shared grouped-GEMM call signature also passes `topk_weights`, but the
current fused gate/up/silu consumer does not apply routing weights in this
stage.  Routing weights are applied by `MOE_GROUP_GEMM_DOWN`.

Writes:

- `silu_mul_output[TP, I]`

Compute:

- For expert-major block `m_idx`, get `expert = expert_ids[m_idx]`.
- For each routed row in that block, multiply by the expert's gate/up weight.
- The `n_idx` tile covers `BN = 128` gate/up columns, which become `BN / 2`
  output activation columns after pairing gate and up halves.
- Apply:

```text
silu_mul_output[row, col] = silu(gate[row, col]) * up[row, col]
```

Synchronization:

- Waits: `evt_count_and_sort`.
- Notifies: `evt_group_gemm_gate_up[m_idx]` in fused static/dynamic mode.
- In `unfused` comparison mode, all gate/up/silu tasks notify one global
  `evt_group_gemm_gate_up[0]`.
- Per-block fused event count is `(2 * I) / BN`.

Dynamic successor:

- For the same `m_idx`, enqueue down-projection tasks:

```text
n_idx in [0, H / BN / down_proj_task_size)
```

Connects to:

- `MOE_GROUP_GEMM_DOWN`, because down reads `silu_mul_output` for the same
  expert-major block and waits on `evt_group_gemm_gate_up[m_idx]`.

### `MOE_GROUP_GEMM_DOWN`

Task space:

```text
m_idx in [0, TP / P)
n_idx in [0, H / BN / down_proj_task_size)
k_idx = 0
```

Static scheduling creates this task space for `TP_MAX / P`; runtime-invalid
blocks check `m_idx < num_tokens_post_pad[0] / P` before running.

Reads:

- `silu_mul_output[TP, I]`
- `grp_down_weight[E, H, I]`
- `expert_ids[TP / P]`
- `topk_weights[B, K]`
- `sorted_token_ids[TP]`
- `num_valid_tokens[TP / P]`, when dynamic GEMM size is enabled

Writes:

- `topk_reduce_output[B, H]`

Compute:

- For expert-major block `m_idx`, get `expert = expert_ids[m_idx]`.
- Run the expert down projection:

```text
partial = grp_down_weight[expert] @ silu_mul_output[row, :]
```

- Map the expert-major row back to `token_id = sorted_token_ids[row] // K`.
- Scale by `topk_weights.view(-1)[sorted_token_ids[row]]`.
- Reduce-add into `topk_reduce_output[token_id, hidden_col]`.

Synchronization:

- Waits: `evt_group_gemm_gate_up[m_idx]` in fused static/dynamic mode, or the
  global `evt_group_gemm_gate_up[0]` in `unfused` mode.
- Dynamic mode notifies `evt_group_gemm_down`.
- Static mode uses a pre-initialized `evt_group_gemm_down` only as part of the
  shared event allocation; static termination comes from the static queue
  sentinel.

Dynamic successor:

- When all runtime down-projection tasks have been dispatched, enqueue
  `END` tasks, one per SM.

Connects to:

- Kernel output validation and benchmarking, which read `topk_reduce_output`.

## Event counters

Event counters are dependency counters.  The consumer may read a producer's
output only after the corresponding counter reaches ready state.

| Event | Producer | Consumers protected | Initial count |
| --- | --- | --- | --- |
| `evt_gating` | `MOE_GATING` | `MOE_TOPK_SOFTMAX` reads `gating_output` | `S * ceildiv(B, 128)` |
| `evt_topk_softmax` | `MOE_TOPK_SOFTMAX` | `MOE_ALIGN` reads `topk_indices` | `KernelConfig.SM_NUMBER` |
| `evt_moe_align` | `MOE_ALIGN` | `MOE_COUNT_AND_SORT` reads align metadata | `1` |
| `evt_count_and_sort` | `MOE_COUNT_AND_SORT` | grouped gate/up/silu reads sorted/reordered data | `KernelConfig.SM_NUMBER` |
| `evt_group_gemm_gate_up` | `MOE_GROUP_GEMM_GATE_UP_SILU` | grouped down reads `silu_mul_output` | fused: `(2 * I) / BN` per block; unfused: all gate/up tasks globally |
| `evt_group_gemm_down` | `MOE_GROUP_GEMM_DOWN` | dynamic termination | static: full max down task count; dynamic: initialized from runtime `TP` |

The rule is simple: if task B reads a buffer written by task A, task A must
notify an event after the write, and task B must wait on that event before the
read.  Dynamic scheduling additionally uses the notify trigger to enqueue B's
task instances.

## Static scheduler

Static scheduling builds a full host queue before launch:

```text
INIT_ETENSOR*
MOE_GATING*
WAIT_ETENSOR_INIT*
MOE_TOPK_SOFTMAX*
MOE_ALIGN
MOE_COUNT_AND_SORT*
MOE_GROUP_GEMM_GATE_UP_SILU*  # max TP_MAX task space
MOE_GROUP_GEMM_DOWN*          # max TP_MAX task space
END*
```

The static queue is work ordering, not a global barrier.  Correctness still
comes from the event waits in each `task_impl_*`.  Static mode may enqueue
padded grouped GEMM tasks for `TP_MAX`; those tasks check `num_tokens_post_pad`
before running tile code.

`unfused` uses the static scheduler path with the same task implementations.
It changes the gate/up event from per-block to one global event so all
gate/up/silu work completes before any down-projection work consumes it.

## Dynamic scheduler

Dynamic scheduling starts with only:

```text
INIT_ETENSOR*
MOE_GATING*
```

Every later task is pushed by a producer's `pre_notify_and_push` rule:

| Trigger event | Runtime task space pushed |
| --- | --- |
| `evt_gating` | all `MOE_TOPK_SOFTMAX` persistent shards |
| `evt_topk_softmax` | one `MOE_ALIGN` task |
| `evt_moe_align` | all `MOE_COUNT_AND_SORT` shards |
| `evt_count_and_sort` | `MOE_GROUP_GEMM_GATE_UP_SILU` for `m_idx < TP / P`, `n_idx < (2 * I) / BN` |
| `evt_group_gemm_gate_up[m_idx]` | `MOE_GROUP_GEMM_DOWN` for that same `m_idx` |
| `evt_group_gemm_down` | `END` tasks |

This is where dynamic scheduling uses runtime data.  `MOE_ALIGN` computes
`num_tokens_post_pad[0]`; later push rules use it to avoid enqueueing grouped
GEMM tasks for nonexistent expert-major blocks.

Dynamic notify is two-phase:

1. `pre_notify_and_push` runs before the tile body.  It lets the scheduler push
   successors as soon as the last producer has been dispatched, avoiding queue
   starvation.
2. The normal `notify` runs after the tile body.  Consumers still wait for the
   normal ready state before reading producer outputs.

The push can happen early, but the read cannot happen early.

## How to add or reconnect tasks

When adding a task, write down its spec before touching scheduler code:

1. Name its logical input buffers and output buffers.
2. Define the task instance space in `(m_idx, n_idx, k_idx)`.
3. Add a `JobType`.
4. Implement a `Tile.run(m_idx, n_idx, k_idx, ...)` for local compute.
5. Add a `task_impl_*` wrapper that passes the correct buffers to the tile.
6. For every read-after-write edge, allocate an event counter in `_set_events`.
7. Initialize that event counter to the number of producer task notifications.
8. In the producer wrapper, notify after writing the output buffer.
9. In the consumer wrapper, wait before reading the producer output buffer.
10. Add the static queue entries for the consumer task space.
11. Add the dynamic `pre_notify_and_push` rule that creates the same consumer
    task space, using runtime shape values when needed.
12. Add a correctness test that compares the final output against a reference.

If steps 6-9 cannot be stated precisely, the tasks are not actually connected.
