# ScheduleBatch → ForwardBatch → 模型输入：一次前向的数据流全解

> 本文回答一个问题：调度器手里那一堆 `Req` 对象，是怎么一步步变成
> `model.forward(input_ids, positions, forward_batch)` 这三个参数的？
>
> 涉及的核心文件：
> - [schedule_batch.py](../../../python/sglang/srt/managers/schedule_batch.py) —— `Req` / `ScheduleBatch`
> - [forward_batch_info.py](../../../python/sglang/srt/model_executor/forward_batch_info.py) —— `ForwardMode` / `ForwardBatch`
> - [overlap_utils.py](../../../python/sglang/srt/managers/overlap_utils.py) —— `resolve_forward_inputs`
> - [scheduler.py](../../../python/sglang/srt/managers/scheduler.py) —— `run_batch`
> - [tp_worker.py](../../../python/sglang/srt/managers/tp_worker.py) —— `forward_batch_generation`
> - [model_runner.py](../../../python/sglang/srt/model_executor/model_runner.py) —— `forward` / `_forward_raw`
> - [eager_runner.py](../../../python/sglang/srt/model_executor/runner/eager_runner.py) —— 真正调 `model.forward` 的地方
> - [cuda_graph_buffer_registry.py](../../../python/sglang/srt/model_executor/cuda_graph_buffer_registry.py) —— FB → 静态 buffer 镜像

---

## 0. 一句话总览

SGLang 一次前向要跨越 **三个抽象层**，每层各管一件事：

| 层级 | 类型 | 归属 | 主要内容 | 所在设备 |
|---|---|---|---|---|
| L1 | `Req` | `Scheduler` | 单请求的全部状态（token、采样参数、KV 记账、完成条件） | CPU（Python 对象） |
| L2 | `ScheduleBatch`（简称 SB） | `Scheduler` | 一批请求的**调度决策结果**：谁进这一步、KV 分到哪些 slot | 大部分 CPU，少量 GPU 张量 |
| L3 | `ForwardBatch`（简称 FB） | `ModelRunner` | 一次前向的**全部张量输入**：positions、extend_*、DP 元信息 | 几乎全在 GPU |

`forward_batch_info.py` 文件开头的注释就是这个结论：

```
ScheduleBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`. Most of the data is on the CPU.
- ForwardBatch is managed by `model_runner.py::ModelRunner`. Most of the data consists of GPU tensors.
  It is constructed directly from a ScheduleBatch by `ForwardBatch.init_new`.
```

注意 **历史包袱**：老版本中间还有一层 `ModelWorkerBatch`（由
`ScheduleBatch.get_model_worker_batch()` 产出）。现在这层已经**被删掉了**，
SB 直接喂给 `ForwardBatch.init_new`。你在 `init_new` 里还能看到
"Mirror the grammars-population behavior previously done in
`ScheduleBatch.get_model_worker_batch`" 这样的注释，那是遗迹。

### 全景流水线

```
Scheduler.get_next_batch_to_run()
        │
        │  挑出这一步要跑的 Req 列表
        ▼
ScheduleBatch.init_new(reqs, pools, tree_cache, ...)      ← 只装配置，不算张量
        │
        ├── prepare_for_extend()   （prefill / chunked prefill / mixed）
        │     · 算 input_ids / seq_lens / prefix_lens / extend_lens
        │     · alloc_for_extend() 分 KV slot，写 req_to_token_pool
        │     · 建 sampling_info
        │
        └── prepare_for_decode()   （decode）
              · alloc_for_decode() 每请求分 1 个 slot
              · seq_lens += 1（out-of-place！）
        │
        ▼
Scheduler.run_batch(batch)
        │
        ├── resolve_forward_inputs(batch, future_map)     ← input_ids 在这里才真正落到 GPU
        │
        ▼
TpModelWorker.forward_batch_generation(batch)
        │
        ▼
ForwardBatch.init_new(batch, model_runner, ...)           ← SB → FB，算 positions / extend_*
        │
        ▼
ModelRunner.forward(fb) → _forward_raw(fb)
        │
        ├── decode CUDA Graph replay ────┐
        ├── prefill CUDA Graph (piecewise)│  三者都会先把 FB 拷进静态 buffer
        └── EagerRunner.execute(fb) ──────┘  （load_batch + registry）
                    │
                    ├── attn_backend.init_forward_metadata(fb)   ← 建 page_table / cu_seqlens
                    │
                    ▼
        model.forward(fb.input_ids, fb.positions, fb, **kwargs)  ★ 模型入口
                    │
                    ├── embed_tokens(input_ids) → hidden_states
                    ├── 每层: LayerNorm → qkv_proj → rotary_emb(positions) → RadixAttention(q,k,v,fb) → o_proj → MLP
                    │            └── attn_backend.forward(...) 用 fb.out_cache_loc 写 KV，用 page_table 读 KV
                    └── LogitsProcessor(input_ids, hidden_states, lm_head, fb)
                                 └── LogitsMetadata.from_forward_batch(fb)
                    │
                    ▼
        LogitsProcessorOutput → ModelRunner.sample(logits_output, fb) → next_token_ids
```

---

## 1. 第一级：`ScheduleBatch`

### 1.1 字段是按"去哪儿"分组的

`ScheduleBatch` 的定义（[schedule_batch.py:2091](../../../python/sglang/srt/managers/schedule_batch.py#L2091)）
刻意用注释分了 6 组，读代码时先认清这个分组，能省一半力气：

| 组 | 例子 | 会不会传给 FB |
|---|---|---|
| **Core：请求列表** | `reqs` | 不直传，但 FB 从它派生 `lora_ids` / `rids` / `grammars` |
| **全局配置与共享资源** | `req_to_token_pool`、`token_to_kv_pool_allocator`、`tree_cache`、`model_config`、`device` | ❌ 不传（引擎级，FB 通过 `model_runner` 拿） |
| **调度器自用状态** | `batch_is_full`、`chunked_req`、`split_index`、`hicache_consumer_index`、`req_pool_indices_cpu` | ❌ 不传 |
| **跨界 GPU 张量** | `input_ids`、`req_pool_indices`、`seq_lens`、`out_cache_loc`、`orig_seq_lens`、`mamba_*` | ✅ 传（目前是**按引用别名**，不是拷贝） |
| **跨界标量/开关** | `forward_mode`、`return_logprob`、`spec_algorithm`、`is_prefill_only`、`seq_lens_sum`、`extend_num_tokens` | ✅ 按值传 |
| **跨界主机侧元数据** | `seq_lens_cpu`、`prefix_lens`、`extend_lens`、`top_logprobs_nums`、`multimodal_inputs` | ✅ 传（CPU list / tensor） |
| **跨界复合对象** | `sampling_info`（`SamplingBatchInfo`）、`spec_info`（`SpecInput`） | ✅ 传（对象自带 GPU 张量） |

> ⚠️ FB 里那组"Borrowed from ScheduleBatch: GPU tensors"上面挂着一条 FIXME：
> 这些张量目前是**引用别名**，SB 和 FB 指的是同一块显存。这也是为什么
> [.claude/rules/schedule-batch-out-of-place-mutation.md](../../../.claude/rules/schedule-batch-out-of-place-mutation.md)
> 规定 SB 字段**禁止原地修改**——`prepare_for_decode` 里写的是
> `self.seq_lens = self.seq_lens + 1` 而不是 `self.seq_lens.add_(1)`，
> 否则 overlap 模式下上一步排队中的 FB 会被就地改坏。

### 1.2 `init_new`：只装配置，一个张量都不算

[ScheduleBatch.init_new](../../../python/sglang/srt/managers/schedule_batch.py#L2286) 做的事很朴素：

```python
batch = cls(
    reqs=reqs,
    req_to_token_pool=..., token_to_kv_pool_allocator=..., tree_cache=...,
    return_logprob=any(req.return_logprob for req in reqs),      # 批级开关 = 请求级 OR
    has_grammar=any(req.grammar for req in reqs),
    is_prefill_only=all(req.is_prefill_only for req in reqs),     # 注意这个是 all
    return_hidden_states_mode=get_batch_return_hidden_states_mode(reqs),
    spec_algorithm=spec_algorithm,
    chunked_req=chunked_req,
    ...
)
```

规律：**开关类字段取 `any`，模式类字段取 `all`**。`is_prefill_only`
用 `all` 是因为只要有一个请求需要生成，整批就得走正常采样路径。

### 1.3 `prepare_for_extend()`：prefill 侧的重头戏

[schedule_batch.py:2465](../../../python/sglang/srt/managers/schedule_batch.py#L2465)。按顺序拆开：

**Step 1 — 从 Req 提取 5 个 host list**

```python
input_ids   = [r.get_fill_ids()[len(r.prefix_indices):] for r in reqs]  # 去掉命中的前缀
seq_lens    = [r.extend_range.end for r in reqs]                        # 本步结束后的总长
prefix_lens = [len(r.prefix_indices) for r in reqs]                     # radix cache 命中长度
extend_lens = [r.extend_range.length for r in reqs]                     # 本步新算的 token 数
extend_logprob_start_lens = [...]
extend_num_tokens = sum(len(ids) for ids in input_ids)
```

- `get_fill_ids()` = `full_untruncated_fill_ids[:extend_range.end]`，
  而 `full_untruncated_fill_ids == origin_input_ids + output_ids`
  （见 [`_refresh_fill_ids`](../../../python/sglang/srt/managers/schedule_batch.py#L1337)）。
- `prefix_indices` 是 radix cache 前缀匹配的产物；**切掉它就是"增量计算"的本质**。
- `extend_range` 是 chunked prefill 的窗口：一个长 prompt 会被切成多段，
  每段走一次 `prepare_for_extend`，`prefix_lens` 逐轮变大。

**Step 2 — input_ids 不上 GPU，先留在 pinned CPU**

```python
pinned_input_ids = flatten_arrays_to_pinned_cpu(input_ids, _pin)
...
self.input_ids = None                          # ← 注意是 None！
self.prefill_input_ids_cpu = pinned_input_ids  # ← 暂存
```

这是一个容易踩坑的地方：**`prepare_for_extend` 之后 `batch.input_ids is None`**。
H2D 拷贝被推迟到 forward stream 上做（见 §1.5），目的是让 H2D 与上一步的
计算重叠，而不是卡在 schedule stream 上。

`seq_lens` 则立刻做了 H2D（`seq_lens_tensor`），同时保留一份 `seq_lens_cpu`
——因为很多 attention backend 需要 CPU 上的 seq_lens 来决定 kernel 配置
（见 `overlap_utils.py` 里的 `needs_cpu_seq_lens`）。

**Step 3 — 分配 KV，并写 `req_to_token_pool`（关键！）**

```python
out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu = alloc_for_extend(self)
```

[alloc_for_extend](../../../python/sglang/srt/mem_cache/allocation.py#L282) 干四件事：

1. `alloc_req_slots()` —— 给每个请求在 `req_to_token_pool` 里分一行，得到
   `req_pool_indices`（形状 `[b]`）。这就是后面 attention backend 查表的**行号**。
2. `alloc_token_slots()` / `alloc_paged_token_slots_extend()` —— 在
   `token_to_kv_pool` 里分 `extend_num_tokens` 个 slot，得到
   `out_cache_loc`（形状 `[extend_num_tokens]`）。**这是"KV 写到哪里"的答案。**
3. `write_cache_indices(...)` —— 把 `out_cache_loc` 按请求切片写进
   `req_to_token_pool.req_to_token[req_pool_idx, prefix_len:seq_len]`，
   同时把命中的 `prefix_indices` 也补写到 `[0:prefix_len]`。
   **这一步建立起 `(请求, 位置) → KV slot` 的映射表。**
4. 更新记账：`req.kv.kv_allocated_len = kv_committed_len = seq_len`。

三个索引张量的关系值得单独记住：

```
req_pool_indices : [b]                 → req_to_token 的行号
req_to_token     : [max_reqs, max_len] → 全量 (req, pos) → kv slot 映射（引擎级持久表）
out_cache_loc    : [num_new_tokens]    → 本步新 token 各自的 kv slot（扁平，跨请求拼接）
```

attention backend 读 KV 用 `req_to_token[req_pool_indices, :max_len]` 建 page_table；
写 KV 用 `out_cache_loc`。

**Step 4 — 逐请求循环，收集杂项**

一个 `for i, (req, seq_len, pre_len) in enumerate(...)` 循环里顺手做了：
`input_embeds` 切片、`positional_embed_overrides` → `replace_embeds/replace_positions`、
`multimodal_inputs` 收集、cached_tokens 统计（HiCache 分层命中拆分
`split_cached_prefix_by_tier`）、mamba track 条目、input logprob 的 token id 对齐。

**Step 5 — 建采样信息**

```python
self.sampling_info = SamplingBatchInfo.from_schedule_batch(self, self.model_config.vocab_size)
```

`SamplingBatchInfo` 自带 GPU 张量（temperature / top_p / top_k / penalizer buffers /
grammar bitmask），FB 只是按引用持有它。

### 1.4 `prepare_for_decode()`：轻量得多

[schedule_batch.py:3248](../../../python/sglang/srt/managers/schedule_batch.py#L3248)：

```python
self.forward_mode = ForwardMode.DECODE
self.input_embeds = None                              # 清掉 prefill 残留

if not self.spec_algorithm.is_none():
    spec_prepare_for_decode(self); return             # 投机解码自己接管

self.out_cache_loc = alloc_for_decode(self, token_per_req=1)   # 每请求 1 个 slot
self.seq_lens     = self.seq_lens + 1                 # out-of-place，见 §1.1 的规则
self.seq_lens_cpu = self.seq_lens_cpu + 1
self.orig_seq_lens = self.orig_seq_lens + 1
self.seq_lens_sum = None                              # 交给 FB.init_new 懒算
```

注意 decode **没有算 input_ids**——注释写得很清楚：

```python
# input_ids is set at end of previous run_batch (placeholder for
# overlap; next_token_ids cast for non-overlap).
```

decode 的 input_ids 就是上一步采样出来的 token，通过 `FutureMap` 中继（§1.5）。

### 1.5 `resolve_forward_inputs()`：input_ids 的最后一公里

[overlap_utils.py:87](../../../python/sglang/srt/managers/overlap_utils.py#L87)。
这个函数**在 forward stream 上执行**，是 SB → FB 之前的最后一道工序：

```python
def resolve_forward_inputs(batch, future_map):
    if batch.prefill_input_ids_cpu is not None:
        prefill_gpu = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)  # ① H2D
        if batch.mix_running_indices is not None:
            decode_gpu = future_map.output_tokens_buf[batch.mix_running_indices]        # ② mixed
            batch.input_ids = torch.cat([prefill_gpu, decode_gpu])
        else:
            batch.input_ids = prefill_gpu
        batch.prefill_input_ids_cpu = None
        batch.mix_running_indices = None
    elif batch.input_ids is None and future_map.spec_algo.is_none():
        batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]          # ③ decode
```

三条来源，对应三种 batch：

| 场景 | 来源 | 说明 |
|---|---|---|
| **prefill / extend** | `prefill_input_ids_cpu` 做 H2D | ① |
| **mixed（chunked prefill + decode 同批）** | prefill 段 H2D，decode 尾巴从 FutureMap gather，然后 `cat` | ② 顺序是 **prefill 在前、decode 在后** |
| **纯 decode** | `future_map.output_tokens_buf[req_pool_indices]` | ③ 完全在 GPU 上 gather，**零 D2H 同步** |

第 ③ 条是 overlap 调度的核心技巧：上一步的 `next_token_ids` 被
`future_map.publish()` **散射（scatter）**进一个以 `req_pool_index` 为下标的
GPU 常驻 buffer `output_tokens_buf`；下一步直接按 `req_pool_indices` **聚集
（gather）**回来。CPU 全程不需要知道采到了什么 token，因此调度线程可以在
GPU 还在算的时候就把下一批准备好。

调用点在 [`Scheduler.run_batch`](../../../python/sglang/srt/managers/scheduler.py#L3871)，
overlap 模式下它被放在 `forward_stream_ctx` 里、且**故意放在 `_forward_isolation`
之外**——因为它会"消费"掉 SB 上的暂存字段（置 None），而 isolation 的快照
恢复不能把这个消费动作撤销。

---

## 2. 第二级：`ForwardBatch.init_new`

入口是 [`TpModelWorker.forward_batch_generation`](../../../python/sglang/srt/managers/tp_worker.py#L593)：

```python
if batch is not None:
    self.set_hicache_consumer(batch.hicache_consumer_index)
    forward_batch = ForwardBatch.init_new(
        batch, self.model_runner,
        capture_hidden_mode=capture_hidden_mode,
        return_hidden_states_before_norm=False,
    )
```

### 2.0 铁律：`init_new` 不许改 SB

[.claude/rules/forward-batch-init-new-purity.md](../../../.claude/rules/forward-batch-init-new-purity.md)：
`init_new` 把传入的 SB 当**只读**。所有"这一次前向特有"的覆盖项都必须走
`init_new` 的 **keyword-only 参数**（`capture_hidden_mode`、
`return_hidden_states_before_norm`），不能塞进 SB 字段。

目前被容忍的三个例外（不要再加新的）：
1. `seq_lens_sum` 回填（`seq_lens` 家族计划被 kv-committed length 取代）；
2. `sampling_info` 的子对象写入（grammars、canary ids）；
3. `mrope_position_delta_repeated_cache` 记忆化。

### 2.1 FB 的字段也是按来源分组的

[forward_batch_info.py:393](../../../python/sglang/srt/model_executor/forward_batch_info.py#L393)
的注释把 FB 字段分成 6 组，理解这个分组基本就理解了 `init_new`：

| 组 | 含义 | 例子 |
|---|---|---|
| **Required core inputs** | 无默认值，必填 | `forward_mode`、`batch_size`、`input_ids`、`req_pool_indices`、`seq_lens`、`out_cache_loc`、`seq_lens_sum` |
| **Borrowed: GPU tensors** | 从 SB 按引用别名（带 FIXME） | `orig_seq_lens`、`mamba_*`、`input_embeds`、`encoder_*` |
| **Borrowed: 标量/开关** | 按值拷 | `return_logprob`、`is_prefill_only`、`spec_algorithm`、`is_extend_in_batch` |
| **Borrowed: host 元数据** | CPU list / tensor | `seq_lens_cpu`、`top_logprobs_nums`、`mm_inputs` |
| **Borrowed: 复合对象** | 自带 GPU 张量 | `sampling_info`、`spec_info` |
| **Derived from `reqs`** | 从请求列表现算 | `lora_ids = [r.lora_id ...]`、`rids = [r.rid ...]` |
| **Per-forward overrides** | 只能通过 kwargs 传 | `capture_hidden_mode`、`return_hidden_states_before_norm` |
| **Forward-derived** | `init_new` 里在 forward stream 上算出来，FB 自己持有 | `positions`、`extend_seq_lens/prefix_lens/start_loc`、`global_num_tokens_*`、`num_token_non_padded` |
| **Runtime-filled** | 构造时为 None，前向过程中才填 | `_attn_output`、`hidden_states`、`dp_local_*`、`attn_cp_metadata`、`mm_input_embeds` |

### 2.2 `init_new` 逐段拆解

**① 决定 `capture_hidden_mode`**

```python
if capture_hidden_mode is None:                      # 没有显式覆盖
    request_capture_hidden_mode = (
        CaptureHiddenMode.NULL if model_runner.is_draft_worker
        else max(batch.return_hidden_states_mode, get_server_return_hidden_states_mode())
    )
    capture_hidden_mode = get_required_capture_hidden_mode(
        request_capture_hidden_mode, batch.spec_info)
```

用 `max` 是刻意的：取服务端配置的**最大**模式，这样低要求的请求可以复用
同一张 CUDA Graph（`CaptureHiddenMode` 实现了 `__lt__`，是 total_ordering）。

**② extend 专属字段在 decode/idle 上置空**

```python
if batch.forward_mode.is_decode_or_idle():
    extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
else:
    extend_seq_lens = batch.extend_lens
    extend_prefix_lens = batch.prefix_lens
    extend_logprob_start_lens = batch.extend_logprob_start_lens
```

**③ grammar 注入 sampling_info**（上面提到的容忍例外之一）

```python
if batch.has_grammar:
    batch.sampling_info.grammars = [req.grammar for req in batch.reqs]
else:
    batch.sampling_info.grammars = None
```

**④ 构造 FB 主体** —— 一个大 `cls(...)`，把 §2.1 前 6 组字段一次性填好。
`batch_size=len(batch.seq_lens)`（注意用的是 seq_lens 长度而不是 `len(reqs)`，
因为 beam search 的 member row 会让两者不等）。

**⑤ 一串条件性增补**

```python
ret._maybe_init_non_generation_fields(batch)        # embedding/reward 模型：dimensions / token_type_ids / MIS delimiter
model_runner.kv_index_translator.rebind_write_loc(ret)   # 统一内存 / 虚拟 slot 翻译
# kv-canary 调试开关：rids_int / bootstrap_room_ids_int / req_all_ids_*
# extend_input_logprob_token_ids → GPU
ret.num_token_non_padded_cpu = num_tokens           # 真实（未 padding）token 数
ret.init_mlp_sync_metadata(batch, device)           # DP attention 各 rank token 数
```

`num_token_non_padded` 非常重要：CUDA Graph / PCG 会把 batch padding 到桶大小，
attention 层靠这个值把 Q/K/V **narrow 回真实长度**（见 §3.5）。

### 2.3 `positions` 是怎么算出来的（FB 最核心的派生字段）

`positions` 是 RoPE 的输入，也是唯一一个和 `input_ids` 并列传给
`model.forward` 的张量。它有 **4 条产生路径**，按优先级排列：

```python
# ⓪ IDLE：空张量，直接 return
if ret.forward_mode.is_idle():
    ret.positions = torch.empty((0,), dtype=torch.int64, device=device)
    return ret

# ① dLLM：按 block_offset 展开 block_size 个连续位置
if batch.dllm_config is not None:
    ret.positions = torch.tensor([...for block_offset...for i in range(...)])

# ② 投机解码：spec_info 自带 positions（树形结构，非连续）
elif ret.spec_info is not None and ret.spec_info.positions is not None:
    ret.positions = ret.spec_info.positions

# ③ DECODE / TARGET_VERIFY：clamp(seq_lens - 1, min=0)
if ret.forward_mode.is_decode() or ret.forward_mode.is_target_verify():
    if ret.positions is None:
        ret.positions = clamp_position(batch.seq_lens)

# ④ EXTEND：compute_position(prefix_lens, seq_lens)
else:
    ret.extend_seq_lens    = H2D(extend_seq_lens)      # [b] int32
    ret.extend_prefix_lens = H2D(extend_prefix_lens)   # [b] int32
    ret.extend_seq_lens_cpu / extend_prefix_lens_cpu = 原 list
    ret.extend_num_tokens  = batch.extend_num_tokens
    positions, ret.extend_start_loc = compute_position(
        model_runner.prefill_attention_backend_str,
        ret.extend_prefix_lens, ret.extend_seq_lens, ret.extend_num_tokens)
    if ret.positions is None:
        ret.positions = positions
```

第 ③ 条的 decode 逻辑：seq_lens 已经在 `prepare_for_decode` 里 +1 了，
所以要 -1 才是"本步这个 token 的位置"。`clamp_position` 在 CUDA/HIP 上是
一个专门的 kernel（`clamp_position_cuda`），native 版本就是
`torch.clamp(seq_lens - 1, min=0).to(torch.int64)`。

第 ④ 条 `compute_position`
（[forward_batch_info.py:1801](../../../python/sglang/srt/model_executor/forward_batch_info.py#L1801)）
产出**两个**东西：

```python
# 假设 2 个请求：prefix_lens=[100, 0], extend_lens=[3, 2]
positions       = [100, 101, 102,   0, 1]   # 每请求从 prefix_len 开始的 arange，扁平拼接
extend_start_loc = [0, 3]                    # 每请求在扁平 token 轴上的起点（exclusive cumsum）
```

`positions` 从 `prefix_len` 起算，正是"前缀已缓存、只算后缀"在位置编码上的体现。
支持 triton 的 backend 走 `compute_position_triton`（一个 kernel 搞定），
否则退化成 Python 循环 + `torch.cat`（每请求一次 `arange`，慢，但正确）。

### 2.4 `positions` 之后：mrope / LoRA / DCP

```python
if model_runner.ngram_embedding_manager.enabled:
    ret._init_ngram_embedding_info(batch, device)

if model_runner.model_config.model_is_mrope:            # Qwen-VL 系列的 3D RoPE
    ...compute_spec_mrope_positions / _compute_mrope_positions...

if model_runner.lora_manager is not None:               # draft worker 上是 None
    if not get_lora().enable_lora_overlap_loading:
        model_runner.lora_manager.fetch_new_loras(set(ret.lora_ids))
    model_runner.lora_manager.prepare_lora_batch(ret)   # 会往 FB 上挂 LoRA 元信息

if model_runner.ps.attn_dcp_size > 1 and is_hip():      # decode context parallel
    ret.dcp_kv_mask = (ret.positions % dcp_size == dcp_rank)
```

`mrope_positions` 形状是 `[3, num_tokens]`（t/h/w 三个轴），所以它在
CUDA Graph registry 里需要专门的 `slice_fn=lambda buf, n: buf[:, :n]`。

---

## 3. 第三级：从 `ForwardBatch` 到 `model.forward`

### 3.1 `ModelRunner.forward` → `_forward_raw` 的四条分支

[model_runner.py:1568](../../../python/sglang/srt/model_executor/model_runner.py#L1568)
是入口，真正的分派在
[`_forward_raw`](../../../python/sglang/srt/model_executor/model_runner.py#L1712)：

```python
def _forward_raw(self, forward_batch, ...):
    # ① decode CUDA Graph：命中就直接 replay，提前 return
    if <decode 且 graph 已捕获且 bs 落在桶里>:
        return self.graph_runner.replay(forward_batch, ...)

    # ② 准备 eager 路径（DP attention 的 padding 在这里做）
    forward_batch = self._prepare_eager_forward_batch(forward_batch)
    self._maybe_execute_deferred_mamba_cow_and_clear(forward_batch)

    # ③ 三种 eager/半 eager 走法
    if forward_batch.forward_mode.is_split_prefill():   ret = ...split...
    elif <prefill piecewise CUDA graph 可用>:           ret = ...piecewise...
    else:                                               ret = self.eager_runner.execute(forward_batch)

    # ④ DP attention 收尾
    return post_forward_mlp_sync_batch(ret)
```

要点：

- **decode graph replay 是第一优先级**，因为 decode 步数最多、CPU launch 开销占比最高。
- `_prepare_eager_forward_batch` 只在非 graph 路径上做 DP padding；graph 路径的
  padding 是通过静态 buffer 的桶天然完成的。
- mamba 的 COW / clear 被"延后"到这里执行，是为了让它跟 forward 在同一个 stream
  上排队，而不是卡住调度线程。

### 3.2 三条路径的共同前置：把 FB 拷进静态 buffer

不管是 decode graph replay、prefill piecewise graph，还是 EagerRunner，
**都要先把 FB 的动态张量镜像到一组固定地址的静态 buffer 上**。这件事由
[`CudaGraphBufferRegistry`](../../../python/sglang/srt/model_executor/cuda_graph_buffer_registry.py)
统一负责。

为什么 eager 路径也要拷？因为 `EagerRunner` 被设计成
"BaseCudaGraphRunner 的 eager 对偶"——同一套 buffer 布局、同一套
`init_forward_metadata` 调用约定，只是不 replay graph。这样 attention backend
不需要区分自己在哪条路径上。

`registry.fill_from(fb, raw_bs, padded_bs, raw_num_tokens, padded_num_tokens, pp_proxy_tensors)`
分三个阶段：

1. **重置 padding 尾部** —— 按每个 slot 声明的 `PaddingPolicy` 处理
   `[raw, padded)` 这段：

   | Policy | 行为 | 用在哪 |
   |---|---|---|
   | `ZERO` | 尾部填 0 | `input_ids`、`positions`、`out_cache_loc`、`req_pool_indices` |
   | `FILL_SENTINEL` | 尾部填哨兵值（如 1） | `seq_lens`、`seq_lens_cpu`——不能填 0，否则 attention 会算出空序列 |
   | `FILL_ONCE` | 只在捕获时填一次，之后不管 | `encoder_lens` |
   | `KEEP_PAD` | 不动，复用上次残留 | 语义上无所谓的 buffer |
   | `FOREACH_COPY` | 参与批量 `_foreach_copy_` | 大多数 D2D 拷贝 |

2. **分组批量拷贝** —— 把所有需要 D2D 的 slot 攒成一组，调一次
   `torch._foreach_copy_`，把 N 次 kernel launch 压成 1 次。
3. **`post_fill` 钩子** —— 处理不能用简单切片表达的 slot，比如
   `mrope_positions` 的 `[3, T]` 形状（`slice_fn=lambda buf, n: buf[:, :n]`）、
   结构化的 `ngram_embedding_info.*` / `pp_proxy_tensors.*` 点号路径。

然后 `registry.extract_buffer(...)` 用
`dataclasses.replace(forward_batch_template, **replace_kwargs)`
造出一个"视图 FB"——**字段全部指向静态 buffer**，标量字段（batch_size、
forward_mode）取本次真值。后续所有代码看到的都是这个视图。

> ⚠️ `SGLANG_EAGER_INPUT_NO_COPY=1` 会让 `EagerRunner.load_batch` 短路成
> `replace(forward_batch)`，跳过全部拷贝。这是给 profiling 用的，正常不要开。

### 3.3 attention 元数据：`init_forward_metadata`

拷完 buffer、进 model 之前，必须让 attention backend "规划"一次：

```python
model_runner.attn_backend.init_forward_metadata(forward_batch)
```

backend 在这一步把 FB 的索引张量翻译成自己需要的形式，典型产物：

```python
# flashattention_backend.py
metadata.page_table = self.req_to_token_pool.req_to_token[
    forward_batch.req_pool_indices, : metadata.max_seq_len_k
]
metadata.cu_seqlens_q / cu_seqlens_k = ...
```

**`page_table` 就是 §1.3 那张 `req_to_token` 表按本批请求取出来的子矩阵**——
这是"读 KV"的地址来源。而"写 KV"用的是 `fb.out_cache_loc`。

为了避免重复规划，FB 上有一组标记位：

| 字段 | 含义 |
|---|---|
| `forward_metadata_ready` | 已经规划过，别再规划 |
| `forward_metadata_planned_bs` / `_num_tokens` | 规划时用的形状，用来判断能否复用 |
| `forward_metadata_replan_equivalent` | 形状变了但语义等价，可跳过 |

判定入口是 `fb.needs_forward_metadata_init()`。旧的
`skip_attn_backend_init` 开关已废弃，由
`apply_deprecated_skip_attn_backend_init` 兼容处理。

### 3.4 模型入口：只有三个位置参数

三条 eager 路径（`_execute_decode` / `_execute_extend` / `_execute_idle`）
最后都收敛到同一行
（[eager_runner.py](../../../python/sglang/srt/model_executor/runner/eager_runner.py)）：

```python
return model_runner.model.forward(
    forward_batch.input_ids,      # [num_tokens] int32/int64
    forward_batch.positions,      # [num_tokens] int64
    forward_batch,                # 其余一切都在这里
    **kwargs,
)
```

**`input_ids` 和 `positions` 被单独拎出来，是因为它们是每个模型都必用、
且在 CUDA Graph 里必须是固定地址的两个张量。** 其他所有信息（KV 地址、
seq_lens、DP 元信息、logprob 需求）都挂在 `forward_batch` 上，由各层自己按需取。

`kwargs` 有两个来源：

- `_pp_kwargs(fb, pp_proxy_tensors)` —— pipeline parallel 的层间张量。
- `_extend_forward_kwargs(fb)` —— extend 专属：
  - `input_embeds`（用户直接传 embedding 的场景，会 `.bfloat16()`）；
  - `replace_embeds` 按 `replace_positions` scatter 进去（positional embed override）；
  - 非生成模型（embedding / reward）追加 `get_embedding=True`。

`EagerRunner.execute` 还有一个容易忽略的动作：**MIXED 被重写成 EXTEND**
（NPU 和 context-parallel 除外）。因为 mixed batch 的 decode 尾巴在
`resolve_forward_inputs` 里已经被 `cat` 到 input_ids 后面了，从模型视角看
它就是一批"extend_len=1 的请求"，没必要单独一种模式。

### 3.5 模型内部怎么用 `forward_batch`

以 [llama.py](../../../python/sglang/srt/models/llama.py) 为代表，调用链是：

```
LlamaForCausalLM.forward(input_ids, positions, forward_batch, ...)
  └─ LlamaModel.forward
       ├─ embed_tokens(input_ids) → hidden_states           ← 用 input_ids
       ├─ for layer in layers:
       │    LlamaDecoderLayer.forward(positions, hidden_states, forward_batch, residual)
       │      ├─ input_layernorm
       │      ├─ LlamaAttention.forward(positions, hidden_states, forward_batch)
       │      │    ├─ qkv_proj(hidden_states) → q, k, v
       │      │    ├─ rotary_emb(positions, q, k)            ← 用 positions
       │      │    ├─ self.attn(q, k, v, forward_batch)      ← RadixAttention
       │      │    └─ o_proj
       │      ├─ post_attention_layernorm
       │      └─ MLP
       └─ norm
  └─ LogitsProcessor(input_ids, hidden_states, lm_head, forward_batch)
```

三个消费点：

| 参数 | 消费者 | 用途 |
|---|---|---|
| `input_ids` | `embed_tokens` + `LogitsProcessor` | 查 embedding；算 input logprob 时要对齐 token id |
| `positions` | `rotary_emb` | RoPE 旋转角度 |
| `forward_batch` | 每层的 `RadixAttention` + 末尾 `LogitsProcessor` | KV 地址、page_table、采样/logprob 元信息 |

### 3.6 `RadixAttention`：KV 的读写都在这里

[radix_attention.py](../../../python/sglang/srt/layers/radix_attention.py) 的
`forward(q, k, v, forward_batch, save_kv_cache=True, ...)` 有两条出路：

- piecewise CUDA graph 下走 `unified_attention_with_output` **custom op**
  （必须是 custom op，否则 torch.compile 无法把它当成黑盒切分图）；
- 否则直接 `get_attn_backend().forward(q, k, v, self, forward_batch, ...)`。

`_unified_attention_with_output_impl` 里有一段很关键的"去 padding"操作：

```python
real = forward_batch.num_token_non_padded_cpu       # 真实 token 数
q = q[:real]; k = k[:real]; v = v[:real]            # narrow 回真实长度
saved_loc, saved_pos = forward_batch.out_cache_loc, forward_batch.positions
forward_batch.out_cache_loc = saved_loc[:real]      # 临时 narrow
forward_batch.positions = saved_pos[:real]
...
forward_batch._attn_output = output[:real_query_num_tokens]
forward_batch.out_cache_loc, forward_batch.positions = saved_loc, saved_pos  # 还原
```

**为什么要这样**：CUDA Graph / piecewise graph 会把 batch padding 到桶大小，
但 padding 出来的 token 不能真的去写 KV cache（会污染别的请求的 slot）。
`num_token_non_padded_cpu` 就是"这批到底有多少真 token"的唯一真相来源。
注意它是临时改 FB 字段再还原——这符合规则（`ForwardBatch` 字段允许原地修改，
`ScheduleBatch` 才禁止）。

backend 内部：

- **写**：`set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)`，
  把本步新算的 K/V 写到 §1.3 分配的那些 slot；
- **读**：用 `metadata.page_table`（= `req_to_token[req_pool_indices]`）
  把整条序列历史 KV 的地址喂给 kernel。

`_attn_output` 是预分配的输出 buffer，避免 custom op 返回值破坏 graph 捕获。
chunked-prefix MHA 场景还会用 `mha_return_lse` 额外返回 log-sum-exp 做二次合并。

### 3.7 收尾：`LogitsProcessor` 与 `sample`

```python
logits_output = LogitsProcessor(input_ids, hidden_states, lm_head, forward_batch)
    └─ LogitsMetadata.from_forward_batch(forward_batch)
```

[`LogitsMetadata.from_forward_batch`](../../../python/sglang/srt/layers/logits_processor.py#L281)
从 FB 里算出 logprob 相关的一堆派生量：

```python
extend_return_logprob      = 是否需要 input logprob
extend_return_top_logprob  = 是否需要 top-k
extend_token_ids_logprob   = 是否需要指定 token 的 logprob
extend_logprob_pruned_lens_cpu = extend_seq_lens_cpu - extend_logprob_start_lens_cpu
```

最后一行是核心：**只有 `[logprob_start_len, seq_len)` 这段需要算 logprob**，
前面的前缀不算，这样能省掉大量 vocab 维度的 softmax。

然后回到 tp_worker：

```python
batch_result.next_token_ids = self.model_runner.sample(logits_output, forward_batch)
```

`sample` 用 `forward_batch.sampling_info`（temperature / top_p / top_k /
penalizer / grammar bitmask）做采样，产出 `next_token_ids`。这个结果被
`future_map.publish()` 散射回 `output_tokens_buf`，成为**下一步 decode 的
`input_ids`**——回到 §1.5 的第 ③ 条，闭环完成。

---

## 4. 三级字段对照表

同一个概念在三层里的名字和形态：

| 概念 | `Req`（L1） | `ScheduleBatch`（L2） | `ForwardBatch`（L3） | 模型侧用途 |
|---|---|---|---|---|
| 待算 token | `full_untruncated_fill_ids` + `extend_range` | `prefill_input_ids_cpu` → `input_ids` | `input_ids` `[T]` | `embed_tokens` |
| 位置 | 隐含在 `extend_range` | ❌ 不存 | `positions` `[T]` | `rotary_emb` |
| 序列总长 | `extend_range.end` | `seq_lens` `[b]` + `seq_lens_cpu` | `seq_lens` / `seq_lens_cpu` / `seq_lens_sum` | attention 元数据 |
| 前缀命中长度 | `len(prefix_indices)` | `prefix_lens`（host list） | `extend_prefix_lens` `[b]` + `_cpu` | `compute_position` |
| 本步 token 数 | `extend_range.length` | `extend_lens` + `extend_num_tokens` | `extend_seq_lens` `[b]` + `extend_num_tokens` | cu_seqlens |
| token 轴起点 | — | ❌ 不存 | `extend_start_loc` `[b]` | 切片定位 |
| KV 表行号 | `req_pool_idx` | `req_pool_indices` `[b]` | `req_pool_indices` `[b]` | 建 `page_table` |
| 本步 KV 地址 | `kv.kv_allocated_len` | `out_cache_loc` `[T]` | `out_cache_loc` `[T]` | `set_kv_buffer` 写 KV |
| 历史 KV 地址 | — | `req_to_token_pool`（引擎级） | 经 `model_runner` 拿 | `page_table` 读 KV |
| 采样参数 | `sampling_params` | `sampling_info` | `sampling_info`（同一对象） | `sample()` |
| logprob 需求 | `return_logprob` / `logprob_start_len` | `return_logprob` / `extend_logprob_start_lens` | 同名 → `LogitsMetadata` | `LogitsProcessor` |
| 真实 token 数 | — | — | `num_token_non_padded(_cpu)` | 去 padding narrow |

三个"形状"要牢记：

```
[b]  = batch_size            → seq_lens、req_pool_indices、extend_* 家族
[T]  = num_tokens（扁平）     → input_ids、positions、out_cache_loc
[3,T]= mrope 三轴            → mrope_positions
```

其中 decode 时 `T == b`（每请求 1 个 token），extend 时 `T == extend_num_tokens`
（跨请求拼接，各请求长度不同）。**这就是为什么 extend 需要
`extend_start_loc` 而 decode 不需要。**

---

## 5. 常见疑问与排查要点

**Q1：为什么我在 `prepare_for_extend` 之后打印 `batch.input_ids` 是 `None`？**

设计如此。见 §1.3 Step 2 —— prefill 的 input_ids 暂存在
`batch.prefill_input_ids_cpu`（pinned CPU），H2D 推迟到
`resolve_forward_inputs`（forward stream 上）。想在调度侧看内容，
打 `prefill_input_ids_cpu`。

**Q2：decode 的 input_ids 从哪来？CPU 上能看到吗？**

看不到，而且**故意让 CPU 看不到**。上一步的 `next_token_ids` 被
`future_map.publish()` scatter 进 GPU 常驻的 `output_tokens_buf`，
下一步 `resolve_forward_inputs` 按 `req_pool_indices` gather 回来。
全程零 D2H 同步，这是 overlap 调度能成立的前提。要调试就在
forward stream 上同步后打 `future_map.output_tokens_buf`。

**Q3：`prepare_for_decode` 之后 `seq_lens_sum` 变成 `None` 了？**

是的，交给 `ForwardBatch.init_new` 懒算（`int(seq_lens_cpu.sum())`）。
这是 §2.0 列出的三个"被容忍的 init_new 改 SB"例外之一。

**Q4：为什么 `seq_lens` 的 padding 要填哨兵值而不是 0？**

`FILL_SENTINEL`。如果填 0，attention backend 会认为那些 padding 行是
"长度为 0 的序列"，某些 kernel 会因此除零或产出 NaN。填 1
（或其他小正数）让它们变成合法但无意义的短序列，配合
`num_token_non_padded` 把结果丢掉。

**Q5：改了 `ScheduleBatch` 的张量后结果错乱？**

大概率违反了
[schedule-batch-out-of-place-mutation.md](../../../.claude/rules/schedule-batch-out-of-place-mutation.md)。
SB → FB 目前是**引用别名**（FB 字段上挂着 FIXME），overlap 模式下上一步的 FB
可能还在 GPU 队列里等着执行。`self.seq_lens.add_(1)` 会把它就地改坏。
永远写 `self.seq_lens = self.seq_lens + 1`。

**Q6：我想给这次前向加一个"只这次生效"的开关，塞哪儿？**

塞 `ForwardBatch.init_new` 的 **keyword-only 参数**，一路透传到
`TpModelWorker.forward_batch_generation`。不要加 SB 字段——见
[forward-batch-init-new-purity.md](../../../.claude/rules/forward-batch-init-new-purity.md)。
现有例子：`capture_hidden_mode`、`return_hidden_states_before_norm`。

**Q7：attention 元数据被规划了两次 / 没规划？**

查 FB 上的 `forward_metadata_ready`、`forward_metadata_planned_bs`、
`forward_metadata_planned_num_tokens`、`forward_metadata_replan_equivalent`
四个标记，以及 `needs_forward_metadata_init()` 的返回值。
CUDA graph replay 和 eager 路径都会调 `init_forward_metadata`，
标记位就是防重的闸门。旧的 `skip_attn_backend_init` 已废弃，别再用。

**Q8：mixed batch（chunked prefill + decode）里 token 顺序是什么？**

**prefill 段在前，decode 段在后**（`torch.cat([prefill_gpu, decode_gpu])`）。
而且到了 `EagerRunner.execute`，MIXED 会被重写成 EXTEND——从模型视角看
decode 请求就是 `extend_len == 1` 的 extend 请求。

**Q9：CUDA Graph 下 KV 被写脏了？**

先确认有没有正确用 `num_token_non_padded_cpu` narrow（见 §3.6）。

注意 padding 本身**不会**污染真实数据：`out_cache_loc` 的 padding policy 是
`ZERO`，而 KV slot 0 和 `req_to_token` 的第 0 行都是**永不分配的保留哨兵**
（`TokenToKVPoolAllocator.free_slots = arange(1, size+1)`、
`ReqToTokenPool.free_slots = range(1, alloc_size)`）——slot 0 就是专门的
padding 垃圾桶。narrow 的真正目的是省算力、以及保证 attention 输出 /
log-sum-exp 的长度与真实 token 数一致。所以 KV 被写脏更可能是
`out_cache_loc` 与 `input_ids` 错位（长度不等 / 拼接顺序反了），
或是 `write_cache_indices` 的 `[prefix_len:seq_len]` 区间算错。

**Q10：`batch_size` 为什么用 `len(batch.seq_lens)` 而不是 `len(batch.reqs)`？**

beam search 场景下一个 `Req` 会展开成多个 member row，两者不等。
张量维度以 `seq_lens` 为准。

---

## 6. 一句话记住每一层

- **`Req`**：一个请求的全部真相，CPU Python 对象，生命周期跨多个 step。
- **`ScheduleBatch`**：这一步"跑谁、KV 放哪"的决策结果。禁止原地改。
  `prepare_for_extend` / `prepare_for_decode` 是它的两个主要变形入口。
- **`resolve_forward_inputs`**：input_ids 的最后一公里，forward stream 上跑。
- **`ForwardBatch`**：一次前向的全部张量，`positions` 和 `extend_*` 在这里才算出来。
  `init_new` 视 SB 为只读。
- **静态 buffer 镜像**：三条执行路径的共同前置，`CudaGraphBufferRegistry` 统管。
- **`model.forward(input_ids, positions, forward_batch)`**：只有两个张量显式传，
  其余全靠 FB 按需取。KV 写 `out_cache_loc`、读 `req_to_token[req_pool_indices]`。
