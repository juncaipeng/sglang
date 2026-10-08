# SGLang 流水线并行（Pipeline Parallelism, PP）方案详解

> 本文系统梳理 SGLang 中 PP（Pipeline Parallelism）的完整实现方案，从参数配置、通信初始化、模型切分、调度循环到执行流程，力求详细且浅显易懂。所有路径均基于仓库根目录 `/work/repos/sglang`。

---

## 0. 一句话理解 PP

流水线并行（PP）把一个大模型**按层（layer）纵向切成若干段**，每一段（stage）放在一组 GPU 上。
- 第 0 段（首 stage）持有 embedding + 前若干层；
- 中间各段持有中间层；
- 最后一段（末 stage）持有末尾层 + 最终 norm + lm_head（输出层）。

一次前向推理像流水线一样：数据从首 stage 流向末 stage，stage 之间通过点对点通信传递「中间隐藏态（hidden states）」。为了让每个 stage 不空等，SGLang 把一个大 batch 切成多个 **microbatch**，让它们错峰在流水线上并行流动（overlap），从而提高 GPU 利用率。

```
        stage0(GPU组0)      stage1(GPU组1)      stage2(GPU组2)
  mb0:   embed+layers  ──▶   layers       ──▶   layers+norm+head ──▶ token
  mb1:        idle           embed+layers ──▶   layers ...
  mb2:        ...                 ...
```

---

## 1. 参数与配置

### 1.1 命令行参数

定义位置：`python/sglang/srt/arg_groups/fields/parallel.py:66-79`

| 参数 | 别名 | 默认值 | 含义 |
|---|---|---|---|
| `pp_size` | `--pipeline-parallel-size` | `1` | PP 段数（stage 数量） |
| `pp_max_micro_batch_size` | — | `None` | 单个 microbatch 的最大 batch size |
| `pp_async_batch_depth` | — | `0` | 异步 batch 深度，用于在末 stage 缓冲输出，overlap GPU 计算与 CPU 后处理，避免末 stage 掉队 |

### 1.2 World size 计算

`python/sglang/srt/server_args.py:582-590`

```python
def compute_world_size(*, enable_dp_attention, dp_size, tp_size, pp_size):
    return (1 if enable_dp_attention else dp_size) * tp_size * pp_size
```

总进程数 = `dp_size × tp_size × pp_size`（启用 dp-attention 时 dp 维度按 1 计）。

### 1.3 兼容性校验（重要）

PP 与很多特性互斥，校验分布在：
- `arg_groups/validation_hook.py:62-89`
  - 非 NPU 平台：`pp_size > 1` 要求 **关闭 overlap schedule**（`disable_overlap_schedule=True`）且 **不使用投机解码**（`speculative_algorithm is None`）；
  - NPU 平台：允许 PP + EAGLE（非 multi-layer）投机解码，但 MTP 仅支持 prefill 节点；
  - `--min-free-slots-delay` 与 PP 不兼容；
  - 要求 `(tp_size × pp_size) % nnodes == 0`；
  - PD-Multiplexing 要求 `pp_size == 1`。
- `arg_groups/parallel_hook.py`
  - Context Parallelism（CP）要求 `pp_size == 1`；
  - DWDP 要求 `pp_size == 1`；
  - Elastic EP 要求 `pp_size == 1`（WORLD 不能跨 PP stage）。

**可共存**：PP 与 TP、attention DP（attn DP）可以组合使用。

### 1.4 环境变量

- `SGLANG_PP_LAYER_PARTITION`：自定义每个 stage 的层数分配（逗号分隔），要求 `元素个数 == pp_size` 且 `总和 == num_hidden_layers`。不设置时默认均分。

---

## 2. 进程与通信初始化

### 2.1 PP 通信组（group）

定义位置：`python/sglang/srt/distributed/parallel_state.py`

- 全局变量：`_PP`（PP group）、`_SELF_PP`（单例 draft PP group）— `L2164-2165`
- 访问器：`get_pp_group()`（`L2173`）、`get_self_pp_group()`（`L2168`）、别名 `get_pipeline_model_parallel_group`（`L2179`）

### 2.2 PP group 的构建

`parallel_state.py:2801-2821`（在 `initialize_model_parallel` 内）

```python
num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
group_ranks = []
for pp_group_idx in range(num_pipeline_model_parallel_groups):
    ranks = list(range(pp_group_idx, world_size, num_pipeline_model_parallel_groups))
    group_ranks.append(ranks)
_PP = init_model_parallel_group(group_ranks, ..., group_name="pp",
                                use_custom_allreduce=False)
```

**关键点**：PP 是最外层维度。同一个 PP group 内的 rank 以 `num_pipeline_model_parallel_groups` 为步长在 world 中跨越 —— 即相邻两个 PP stage 之间正好隔了一整个 TP（含 DP/CP）块。PP 不使用 custom allreduce（只做点对点通信）。

### 2.3 rank / world size 获取

`parallel_state.py:3000-3005`

```python
def get_pipeline_model_parallel_world_size(): return get_pp_group().world_size
def get_pipeline_model_parallel_rank():       return get_pp_group().rank_in_group
```

首/末 stage 判定：`GroupCoordinator.is_first_rank / is_last_rank / next_rank / prev_rank`（`L562-584`）。

运行时访问层：`runtime_context.py:369-370` 的 `ParallelContext.pp_rank`。

### 2.4 Self PP group（投机解码用）

`parallel_state.py:2823-2833`：每个 rank 单独成组 `[[r] for r in range(world_size)]`，供 one-layer draft 场景使用。另有 `patch_pipeline_parallel_group()`（`L2895`）用于临时替换 PP group。

---

## 3. 模型层的 PP 切分

### 3.1 层分配算法

`python/sglang/srt/distributed/utils.py:95-135` 的 `get_pp_indices`：
1. 若设置 `SGLANG_PP_LAYER_PARTITION`，按环境变量分配；
2. 否则默认均分：`base = num_hidden_layers // pp_size`，`remainder = num_hidden_layers % pp_size`，**余数分配给靠后的 stage**（每个多一层）；
3. 返回 `(start_layer, end_layer)`。

### 3.2 通用构造器 make_layers

`python/sglang/srt/utils/common.py:1457-1498`

```python
start_layer, end_layer = get_pp_indices(num_hidden_layers, pp_rank, pp_size)
modules = torch.nn.ModuleList(
    [PPMissingLayer(...) for _ in range(start_layer)]                       # 前置占位
    + [layer_fn(idx) for idx in range(start_layer, end_layer)]             # 本 stage 真实层
    + [PPMissingLayer(...) for _ in range(end_layer, num_hidden_layers)]   # 后置占位
)
return modules, start_layer, end_layer
```

**核心设计**：
- 层索引在所有 stage 上保持**全局一致**（`self.layers[i]` 的 `i` 始终是全局层号）；
- 非本 stage 的层用 `PPMissingLayer`（`layers/utils/common.py`）占位，不占显存也不参与计算；
- forward 时只遍历 `range(start_layer, end_layer)`，只跑本 stage 的层。

### 3.3 以 Llama 为例

`python/sglang/srt/models/llama.py`

`LlamaModel.__init__`（`L382-415`）：
- 首 stage：`embed_tokens = VocabParallelEmbedding(...)`；否则 `PPMissingLayer()`；
- `make_layers(...)` 构造本 stage 层；
- 末 stage：`norm = RMSNorm(...)`；否则 `PPMissingLayer(return_tuple=True)`。

`LlamaModel.forward`（`L418-464`）：
- **首 stage**：从 `input_ids` 计算 embedding；
- **非首 stage**：从 `pp_proxy_tensors["hidden_states"]` / `["residual"]` 取上游传来的隐藏态；
- 遍历 `range(start_layer, end_layer)` 跑本 stage 层；
- **非末 stage**：返回 `PPProxyTensors({"hidden_states", "residual"})`；
- **末 stage**：做 `norm` 后返回最终 hidden_states。

`LlamaForCausalLM.forward`（`L561-595`）：只有 `pp_group.is_last_rank` 才走 `logits_processor` / `pooler`（`lm_head` 也仅末 stage 生效）；否则直接返回 hidden_states 继续传给下游。权重加载按 `start_layer`/`end_layer` 过滤，跳过其他 stage 的层。

### 3.4 以 DeepSeek-V2 为例

`python/sglang/srt/models/deepseek_v2.py:2591-3071`

大体与 Llama 一致，额外之处：
- 非首 stage 除 `hidden_states`/`residual` 外还传递 `topk_indices`（DSA 相关）；
- 对 DSA 层的跨 stage 边界有专门断言；
- `lm_head` 仅末 stage；`world_size==1 && tie_word_embeddings` 时特殊处理。

---

## 4. Scheduler 层的 PP 调度

### 4.1 事件循环分派

`python/sglang/srt/managers/scheduler.py:5657-5684` 的 `dispatch_event_loop`：

| 模式 | 条件 | 使用的循环 |
|---|---|---|
| NULL（普通） | `pp_size > 1` | `event_loop_pp()` |
| PREFILL（PD 分离） | — | `event_loop_pp_disagg_prefill()` |
| DECODE（PD 分离） | — | `event_loop_pp_disagg_decode()` |

核心实现在 `python/sglang/srt/managers/scheduler_pp_mixin.py`（`SchedulerPPMixin`，约 1400 行）。

### 4.2 microbatch 循环状态

`scheduler_pp_mixin.py:552-562` 的 `init_pp_loop_state`：

```python
self.pp_loop_size = self.ps.pp_size + get_parallel().pp_async_batch_depth
self.mbs        = [None] * self.pp_loop_size   # 各 microbatch 待跑 batch
self.last_mbs   = [None] * self.pp_loop_size
self.running_mbs= [...]
self.mb_metadata= [None] * self.pp_loop_size
self.pp_outputs = None
self.last_rank_comm_queue = deque()            # 末 stage 输出缓冲
```

**关键公式**：`microbatch 数 = pp_size + pp_async_batch_depth`，让多个 microbatch 在流水线上 overlap。

### 4.3 主循环 event_loop_pp

`scheduler_pp_mixin.py:62-168`。外层 while 中遍历 `for mb_id in range(self.pp_loop_size)`，对每个 microbatch：
1. `ingest_requests()` 收请求；非末 rank 用**异步 send** 把 `recv_reqs` 转发给下一 stage；
2. `get_next_batch_to_run()` 规划本 mb 的 batch；
3. 有 batch 则 `_pp_recv_proxy_tensors()` 从上游**同步 recv** 隐藏态；
4. 按 `pp_async_batch_depth` 决定何时提交上一轮输出 send 并预处理输出张量；
5. `_pp_launch_batch()` 在 `forward_stream` 上跑当前 mb；
6. 处理 `next_mb_id` 的 batch 结果（`d2h_event.synchronize()` 后 `_pp_process_batch_result`）——**上一个 microbatch 的结果处理与当前 microbatch 的 GPU 计算并行**；
7. 非末 rank：等 `launch_event` 后把 `pp_hidden_states_proxy_tensors` 异步 send 给下一 stage。

**设计要点**（docstring `L64-85`）：同步 recv + 异步 send，既避免 stage 间 desync，又减少通信阻塞；`pp_async_batch_depth` 用于在末 stage 缓冲输出以 overlap GPU/CPU。

### 4.4 点对点通信原语

| 原语 | 位置 | 用途 |
|---|---|---|
| `_pp_send_pyobj_to_next_stage` / `_pp_recv_pyobj_from_prev_stage` | `L734-766` | 传 Python 对象（请求等），底层 `point_to_point_pyobj` |
| `_pp_send_dict_to_next_stage` / `_pp_recv_typed_dict` | `L801-855` | 传张量字典，带 `__msg_type__`（`"proxy"`/`"output"`）多路复用 |
| `_pp_recv_proxy_tensors` | `L857-866` | 非首 rank 收 `proxy` 类型 dict 并包成 `PPProxyTensors` |
| `_pp_launch_batch` | `L1222-1260` | 跑 batch；末 rank 把输出连同 CUDA event 压入 `last_rank_comm_queue` |

目标 rank 计算：`pp_rank * tp_size + dp_offset`，下一 stage 为 `((pp_rank+1) % pp_size) * tp_size + dp_offset`。只由每个 attn DP 组内 `attn_tp_rank==0 && attn_cp_rank==0` 的 rank 发送，其余靠 `attn_cp_tp_broadcast_pyobj` 广播。

底层张量传输用 `pp_group.send_tensor_dict` / `recv_tensor_dict`（`parallel_state.py:1698/1753`），`all_gather_group=attn_tp_group`。

### 4.5 PD 分离下的 PP

`event_loop_pp_disagg_prefill`（`L171-353`）与 `event_loop_pp_disagg_decode`（`L356-550`）在普通循环基础上，额外对 bootstrap / transfer / release / prealloc / retract 请求做**跨 stage 共识**（每个 stage 本地可能失败，需在 PP 环上取交集达成一致）。

辅助函数：`_pp_pd_get_bootstrapped_ids`、`_pp_pd_get_prefill_transferred_ids`、`_pp_pd_send_consensus_bootstrapped_ids`、`_pp_pd_get_retract_ids`、`_pp_pd_get_prealloc_ids`、`_pp_pd_get_decode_transferred_ids` 等，均用 `get_rids` + `poll_and_all_reduce_attn_cp_tp_group` 做本地投票再取交集。

### 4.6 run_batch 中的 PP 处理

`scheduler.py:4210-4413` 的 `run_batch(batch, pp_proxy_tensors=None)`：
- 把 `pp_proxy_tensors` 透传给 `model_worker.forward_batch_generation`；
- 仅 `pp_size == 1` 时才在 scheduler 端直接 D2H copy 采样结果；`pp_size > 1` 时输出要先经 PP 环传回首 rank 再 copy；
- `max_running_requests` 会按 `// pp_size` 折算。

---

## 5. ModelRunner 层的 PP 支持

`python/sglang/srt/model_executor/model_runner.py`

- **support_pp 探测**（`L479-482`）：检查 `model.forward` 签名里是否有 `pp_proxy_tensors` 参数；
- **PP kwargs 组装**（`L1574-1604`）：`_pp_kwargs = {"pp_proxy_tensors": ...} if support_pp else {}`；
- **forward 主流程**（`L1628-1687` → `_forward_raw` `L1775-1870`）：decode CUDA graph / prefill CUDA graph / eager 三条路径都会带上 proxy tensors；只有 `pp_group.is_last_rank` 才 `post_forward_mlp_sync_batch`；
- **rank 组织**（`L647-648`）：dist world = `tp_size × pp_size`，全局 rank = `tp_size × pp_rank + tp_rank`；含 dp 时 `dp_size × tp_size × pp_size`；
- CUDA graph capture 走 `get_pp_group().graph_capture(context)`。

### 5.1 TpModelWorker 的分叉

`python/sglang/srt/managers/tp_worker.py:593-703` 的 `forward_batch_generation`：
- **末 rank**：`model_runner.forward(...)` 得 `logits_output`，做采样，返回带 `next_token_ids` 的 `GenerationBatchResult`；
- **非末 rank**：`out.logits_output` 实际是 `PPProxyTensors`，包进 `GenerationBatchResult(pp_hidden_states_proxy_tensors=...)` 返回，供 scheduler 在 `event_loop_pp` 中 send 给下一 stage。

### 5.2 PPProxyTensors 数据结构

`python/sglang/srt/model_executor/forward_batch_info.py:1877-1904`：包一个 `Dict[str, torch.Tensor]`，支持 `__getitem__`（str 或 slice）、`__setitem__`。字段承载 `hidden_states`、`residual`（DeepSeek 还有 `topk_indices` 等）。

`GenerationBatchResult.pp_hidden_states_proxy_tensors` 字段定义在 `managers/utils.py:50`。

---

## 6. PP 与 TP / DP / EP 的组合关系

### 6.1 Rank 布局（PP 为最外层）

- world_size = `(dp_size if not dp_attn else 1) × tp_size × pp_size`；
- 全局 rank = `tp_size × pp_rank + tp_rank`；
- PP group 成员按 `range(pp_group_idx, world_size, num_pipeline_model_parallel_groups)` 跨越整段 TP 块 —— 相邻 stage 之间正好隔一个完整 TP（含 DP/CP）块。

### 6.2 进程启动的 rank 分配

`python/sglang/srt/entrypoints/engine.py:1845-1876` 的 `_calculate_rank_ranges`：
- `pp_size_per_node = max(pp_size // nnodes, 1)`；
- `nnodes_per_pp_rank = max(nnodes // pp_size, 1)`；
- PP stage 可跨多节点（nnodes > pp_size）或每节点多个 PP stage；
- 双层循环 `for pp_rank in pp_rank_range: for tp_rank in tp_rank_range:` 逐个 spawn scheduler 进程，`compute_local_gpu_id(...)` 计算本地 GPU。

`data_parallel_controller.py`、`ray/data_parallel_controller.py`、`ray/engine.py`、`weight_cache/daemon.py` 各自也有对应的 `for pp_rank in pp_rank_range` 进程编排。

### 6.3 互斥与共存总结

| 特性 | 与 PP 关系 |
|---|---|
| TP（张量并行） | ✅ 可组合 |
| attention DP | ✅ 可组合（PP 通信只由 attn_tp_rank==0 发送，其余广播） |
| Context Parallelism | ❌ 互斥（要求 pp_size==1） |
| DWDP | ❌ 互斥 |
| Elastic EP | ❌ 互斥 |
| PD-Multiplexing | ❌ 互斥 |
| Overlap Schedule | ❌ 非 NPU 平台与 PP 互斥 |
| 投机解码 | ❌ 非 NPU 互斥；NPU 支持 PP+EAGLE（非 multi-layer，仅 prefill 做 MTP） |

---

## 7. 关键文件清单

| 关注点 | 路径 | 关键行 |
|---|---|---|
| PP 参数定义 | `python/sglang/srt/arg_groups/fields/parallel.py` | 66-79 |
| world_size 计算 | `python/sglang/srt/server_args.py` | 582-590 |
| PP 校验 | `python/sglang/srt/arg_groups/validation_hook.py` | 62-89 |
| PP group 初始化 | `python/sglang/srt/distributed/parallel_state.py` | 2164-2179, 2801-2833, 3000-3005 |
| 层切分算法 | `python/sglang/srt/distributed/utils.py` | 95-135 |
| make_layers / PPMissingLayer | `python/sglang/srt/utils/common.py` | 1457-1498 |
| Llama PP | `python/sglang/srt/models/llama.py` | 382-464, 561-595 |
| DeepSeek-V2 PP | `python/sglang/srt/models/deepseek_v2.py` | 2591-3071 |
| Scheduler PP 事件循环 | `python/sglang/srt/managers/scheduler_pp_mixin.py` | 62-168, 552-562, 704-866, 1222-1260 |
| 事件循环分派 | `python/sglang/srt/managers/scheduler.py` | 5657-5684 |
| run_batch PP 透传 | `python/sglang/srt/managers/scheduler.py` | 4210-4413 |
| ModelRunner PP forward | `python/sglang/srt/model_executor/model_runner.py` | 1574-1604, 1628-1870 |
| TpWorker PP 分叉 | `python/sglang/srt/managers/tp_worker.py` | 593-703 |
| PPProxyTensors | `python/sglang/srt/model_executor/forward_batch_info.py` | 1877-1904 |
| 进程 rank 编排 | `python/sglang/srt/entrypoints/engine.py` | 737-767, 1845-1876 |

---

## 8. 核心机制小结

1. **参数**：`--pipeline-parallel-size`（`pp_size`）、`pp_max_micro_batch_size`、`pp_async_batch_depth`；与投机解码 / overlap / CP / EP / DWDP / PDmux 多处互斥。
2. **通信**：PP group 沿 world 以 TP 块为步长跨越；stage 间用 `point_to_point_pyobj`（Python 对象/请求）+ `send_tensor_dict`/`recv_tensor_dict`（隐藏态，带 `__msg_type__` 多路复用）做点对点，**同步 recv + 异步 send**。
3. **模型切分**：`get_pp_indices` 均分层（余数给末尾 stage），`make_layers` 用 `PPMissingLayer` 占位保持全局层号一致；embedding 仅首 stage、norm + lm_head 仅末 stage，中间用 `PPProxyTensors` 传 `hidden_states` / `residual`。
4. **调度**：`event_loop_pp` 用 `pp_loop_size = pp_size + pp_async_batch_depth` 个 microbatch 在流水线上 overlap；上一个 microbatch 的结果处理与当前 microbatch 的 GPU 计算并行；PD 分离场景额外做 bootstrap / transfer / release 的环上共识。
5. **执行**：`TpModelWorker.forward_batch_generation` 按 `is_last_rank` 分叉 —— 末 rank 采样并返回 token，非末 rank 返回 `pp_hidden_states_proxy_tensors` 供 scheduler 转发下一 stage。
