# DeepEP All-to-All Dispatch / Combine 通信架构

> 本文承接《EP 并行下 MoE 的 grouped GEMM》话题，聚焦 SGLang 的 **DeepEP**（Deep Expert Parallelism）token 分发/收拢通信层：它如何把 token 通过 all-to-all 送到「拥有目标专家」的卡上、如何反向收拢并做加权求和，以及 **normal / low-latency 两条路径** 如何分别喂给 grouped GEMM 的 **contiguous / masked** 两种布局。
>
> 阅读顺序：先看第 1~2 节建立全景与两模式直觉，再看第 3~5 节的通信语义与布局衔接，第 6~9 节是自动选择、缓冲区、通信-计算重叠等工程细节，最后第 10 节是文件行号速查表。
>
> 核心文件：`python/sglang/srt/layers/moe/token_dispatcher/deepep.py`（1018 行）。
> 注意：底层 `deep_ep.Buffer` 的 RDMA/NVLink kernel 属外部 `deep_ep` 包（deepep.py:50 import），本文从 SGLang 调用点与参数契约描述其行为，不涉及 kernel 源码。

---

## 1. 一句话定位与为什么需要 all-to-all

EP 并行下，256 个专家被切到（比如）8 张卡，每卡放 32 个本地专家。但一个 token 的 topk 路由结果是**任意的**——它的 top-6 专家可能散落在 8 张卡上。于是每张卡上算出来的 token，必须**按路由结果重新洗牌**送到目标专家所在的卡。这就是 **dispatch = all-to-all 发送**；专家算完后再洗回原位并对一个 token 的 k 个专家输出做加权和，就是 **combine = 反向 all-to-all + 加权归约**。

```
            router topk_ids (每 token 选 6 个专家, 专家散落各卡)
                              │
   ┌──────────────────────────┼──────────────────────────┐
   │ rank0 的 token            │ rank1 的 token            │  ...
   └──────────────────────────┴──────────────────────────┘
                              │  DISPATCH  (all-to-all 发送)
                              ▼
   ┌──────────────────────────┬──────────────────────────┐
   │ rank0 收到「本地 32 专家」 │ rank1 收到「本地 32 专家」 │  ...
   │ 的全部 token(按专家分桶)   │ 的全部 token             │
   └──────────────────────────┴──────────────────────────┘
                              │  Grouped GEMM (gate/up → act → down)
                              ▼
   ┌──────────────────────────┬──────────────────────────┐
   │ 各专家输出                 │ 各专家输出               │
   └──────────────────────────┴──────────────────────────┘
                              │  COMBINE  (反向 all-to-all + 加权求和)
                              ▼
                  每个 token 回到原卡原位, 6 个专家输出按 topk 权重加权合并
```

DeepEP 的价值在于：把这套 all-to-all 用 **RDMA（跨节点）+ NVLink（节点内）** 高效实现，并提供 **两种为不同负载优化的模式**。

---

## 2. 两种模式对照：NORMAL vs LOW_LATENCY

DeepEP 有两条完全独立的通信路径，由 `DeepEPMode`（`layers/moe/utils.py:171`）区分。它们的**输出布局不同**，从而直接决定了后续 grouped GEMM 走 contiguous 还是 masked 变体。

| 维度 | NORMAL（高吞吐） | LOW_LATENCY / LL（低延迟） |
|---|---|---|
| 典型场景 | prefill / extend / 大 batch | decode / 小 batch |
| 底层调用 | `Buffer.get_dispatch_layout` + `Buffer.dispatch` | `Buffer.low_latency_dispatch` / `low_latency_combine` |
| 输出布局 | **变长紧密**：所有收到的 token 按专家紧挨排列，总长=实际收到数 | **定长带掩码**：`(num_local_experts, max_m, hidden)` 三维，每专家固定槽位 |
| 携带的分组信息 | `num_recv_tokens_per_expert: List[int]` | `masked_m`（每专家真实 token 数）+ `expected_m`（均值估计） |
| 输出 NamedTuple | `DeepEPNormalDispatchOutput`（deepep.py:95） | `DeepEPLLDispatchOutput`（deepep.py:109） |
| 喂给的 grouped GEMM | contiguous 变体（`m_grouped_*_contiguous` + `m_indices`） | masked 变体（`*_masked` + `masked_m`/`expected_m`） |
| 缓冲区 | NVL + RDMA 双份 | 仅 RDMA（无 NVL） |
| 加权求和位置 | SGLang 侧 permute kernel / combine handle | **DeepEP 的 `low_latency_combine` kernel 内部** |
| CUDA graph | `deepep_mode=="normal"` 时禁用 cuda graph（server_args.py:5871） | 定长布局利于图捕获，decode 主力 |
| 有无 padding 浪费 | 无（变长） | 有（`max_m` 固定，多余槽算了丢弃），换来静态形状 |

两个实现类：
- `_DeepEPDispatcherImplNormal`（deepep.py:495）
- `_DeepEPDispatcherImplLowLatency`（deepep.py:655）

公共门面 `DeepEPDispatcher`（deepep.py:863）**同时持有两个实现**（AUTO 模式下 `enable_normal()` 与 `enable_low_latency()` 都为真，deepep.py:892-901），每次 forward 再动态选一个（见第 6 节）。

---

## 3. dispatch：all-to-all 发送的语义

### 3.1 两阶段（异步）拆分

无论哪种模式，`dispatch` 都被拆成 `dispatch_a` → hooks → `dispatch_b` 两阶段（deepep.py:919-946），配合一个 `_Stage` 状态机（deepep.py:855，`INITIAL→AFTER_DISPATCH_A→AFTER_DISPATCH_B→AFTER_COMBINE_A`）。目的是让调度器在「发起通信」与「等待通信完成」之间插入计算，实现通信-计算重叠（见第 8 节）。`dispatch_a` 把内部中间态存进 `_dispatch_intermediate_state`，`dispatch_b` 取出并等待完成。

### 3.2 NORMAL 的两次 DeepEP 调用

`_DeepEPDispatcherImplNormal._dispatch_core`（deepep.py:545-605）里是**两个** DeepEP 库调用：

1. **算路由计划** `buffer.get_dispatch_layout(topk_ids, num_experts, ...)`（deepep.py:559）
   → 返回 `num_tokens_per_rank`、`num_tokens_per_rdma_rank`、`num_tokens_per_expert`、`is_token_in_rank`。
   即「每个 token 要去哪张卡 / 哪个 RDMA rank / 哪个专家」的路由表。

2. **真正 all-to-all** `buffer.dispatch(x, topk_idx=..., topk_weights=..., num_tokens_per_rank=..., is_token_in_rank=..., num_tokens_per_expert=..., expert_alignment=128 if ENABLE_JIT_DEEPGEMM else 1, config=...)`（deepep.py:571-591）
   → 返回 `recv_x`（本卡收到的、按专家紧密排列的 token）、`recv_topk_ids`、`recv_topk_weights`、`num_recv_tokens_per_expert`、`self.handle`、`event`。

关键：`self.handle` 是成员变量，**必须保留到 combine 阶段**（deepep.py:566-569 注释说明），因为它编码了「怎么把输出洗回去」的逆向路由信息。

`dispatch_a` 里还可选做 FP8 量化（当 `ENABLE_JIT_DEEPGEMM and use_fp8`，`sglang_per_token_group_quant_fp8(hidden, 128, ...)`，deepep.py:510-518），把激活压成 FP8 减少通信量。

### 3.3 LOW_LATENCY 的单次调用 + expected_m

`_DeepEPDispatcherImplLowLatency`：
- `dispatch_a`（deepep.py:667）先算 `expected_m = (num_tokens*group_size*topk + num_experts)//num_experts`（deepep.py:675-678），即「每专家平均能收到多少 token」的估计，供 masked GEMM 调优。
- `_dispatch_core`（deepep.py:725）调 `buffer.low_latency_dispatch(hidden, topk_ids, num_max_dispatch_tokens_per_rank, num_experts, use_fp8=..., return_recv_hook=..., ...)`（deepep.py:749-767）
  → 返回 `packed_recv_hidden`、`self.packed_recv_count`（即 `masked_m`）、`self.handle`、`event`、`hook`。
- 输出布局是三维 `(num_local_experts, num_max_dispatch_tokens_per_rank, hidden)`，`masked_m[e]` 标每专家有效行数，多余槽位内容无意义（GEMM 会算但被掩掉）。

`num_max_dispatch_tokens_per_rank` 来自环境变量 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`（deepep.py:376，默认 128），断言 `<= 1024`（deepep.py:381）。

---

## 4. combine：反向 all-to-all + 加权归约

combine 同样两阶段 `combine_a` → `combine_b`（deepep.py:948-973）。`combine_a` 从 `CombineInput` 解包 `(hidden_states, topk_ids, topk_weights)`（deepep.py:960）。

**语义**：专家算完的输出要洗回每个 token 的原卡原位，并且一个 token 的 k 个专家输出要按 topk 权重加权求和。「洗回去」用的就是 dispatch 阶段留下的 `self.handle`（逆向路由）。

**加权求和发生在哪里，两模式不同**（这是最易错的点）：

| 模式 | combine 调用 | topk 加权在哪做 |
|---|---|---|
| NORMAL | `buffer.combine(x, self.handle, config=...)`（deepep.py:632） | `combine_a` 原样透传 hidden（deepep.py:607-620），**加权在 SGLang 侧的 permute kernel**：`ep_gather(hidden, topk_ids, topk_weights, output_index, gather_out)`（deep_gemm.py:913）或标准路径 `post_reorder_deepgemm(..., topk_weights, top_k, routed_scaling_factor)`（deep_gemm.py:734-748） |
| LOW_LATENCY | `buffer.low_latency_combine(x=..., topk_idx=topk_ids, topk_weights=topk_weights, handle=..., ...)`（deepep.py:830-838） | **`topk_weights` 直接传进 DeepEP kernel，加权求和在 `low_latency_combine` 内部完成** |

`combine_b`（deepep.py:622/783）等待通信事件、清理 `self.handle`。

---

## 5. dispatch 输出如何衔接 grouped GEMM（contiguous vs masked）

DeepEP 输出的两种布局，通过 `MoeRunner` 的 **pre_permute / post_permute 注册表**（`moe_runner/base.py` 的 `@register_pre_permute`/`@register_post_permute`，`moe_runner/runner.py:150-176` 按 `dispatch_output.format` 分派）接到 DeepGEMM 的两种 grouped GEMM。

| 注册键（dispatch格式, runner格式） | 函数 | 位置（deep_gemm.py） | 产出布局 |
|---|---|---|---|
| `("deepep_normal","deep_gemm")` | `pre_permute_deepep_normal_to_deep_gemm` | 800-890 | **contiguous** `(all_tokens,K)` + `m_indices`，`use_masked_gemm=False` |
| `("deep_gemm","deepep_normal")` | `post_permute_deep_gemm_to_deepep_normal` | 893-919 | `ep_gather` + topk 加权 → `DeepEPNormalCombineInput` |
| `("deepep_ll","deep_gemm")` | `pre_permute_deepep_ll_to_deep_gemm` | 756-781 | **masked** 3-D + `masked_m`/`expected_m`，`use_masked_gemm=True` |
| `("deep_gemm","deepep_ll")` | `post_permute_deep_gemm_to_deepep_ll` | 784-797 | `DeepEPLLCombineInput` |

- **NORMAL → contiguous**：`pre_permute_deepep_normal_to_deep_gemm` 里 `all_tokens = sum(num_recv_tokens_per_expert)`（deep_gemm.py:818），用 `ep_scatter(...)`（deep_gemm.py:867）把变长 token 写成紧密连续布局并生成 `m_indices`（每行→专家 id）。`DeepGemmRunnerCore.run`（deep_gemm.py:152）按 `use_masked_gemm=False` 走 `_run_contiguous_gemm`（deep_gemm.py:180，注释写着 `GroupGemm-1`/`GroupGemm-2`）。
- **LL → masked**：`pre_permute_deepep_ll_to_deep_gemm` 几乎透传，把 `masked_m`/`expected_m` 直接带下去（无需 scatter，因为 dispatch 已经是定长布局），走 `_run_masked_gemm`（deep_gemm.py:388）。

`DeepEPMoE.forward_impl`（`ep_moe/layer.py:177`）：主流的 DeepGEMM FP8/BF16 路径走 `deprecate_flag` 分支（layer.py:183）进入 `FusedMoE.forward_impl` + `MoeRunner`——真正执行上面注册表的地方。内联的 `run_moe_core`（layer.py:205）如今只剩 W4AFP8（cutlass）路径，DeepGEMM 的 contiguous/masked 内联版已用 `assert False, "...deprecated"`（layer.py:223/228）标记弃用。

---

## 6. AUTO 模式：prefill=normal，decode=low_latency

`--deepep-mode {auto,normal,low_latency}`（server_args.py:1958，默认 `auto`，help 文本明确「low_latency for decode batch and normal for prefill batch」）。

每次 forward，`DeepEPDispatcher._get_impl()`（deepep.py:975）会：
```python
is_extend_in_batch = get_is_extend_in_batch()          # deepep.py:976
resolved = self.deepep_mode.resolve(is_extend_in_batch) # deepep.py:977
```
其中 `DeepEPMode.resolve`（utils.py:181）：
```python
def resolve(self, is_extend_in_batch):
    if self != DeepEPMode.AUTO:
        return self                       # 显式指定就用指定的
    return NORMAL if is_extend_in_batch else LOW_LATENCY
```

- `is_extend_in_batch=True`（prefill/extend）→ NORMAL
- `is_extend_in_batch=False`（decode）→ LOW_LATENCY

`is_extend_in_batch` 由 `set_is_extend_in_batch`/`get_is_extend_in_batch`（dp_attention.py:274/283）在每次 forward 前写入，来源是 `batch.forward_mode.is_extend()`，并在 DP 各 rank 间同步（`scheduler_components/dp_attn.py:272`），**保证所有 rank 选同一模式**（否则 all-to-all 双方模式不一致会死锁）。

CUDA graph 一致性：`DeepEPCudaGraphRunnerAdapter.capture/replay`（`model_executor/runner_utils/deepep_adapter.py:30-42`）记录捕获时解析出的模式并回放时用 `DeepEPBuffer.set_dispatch_mode` 强制复用；decode runner 捕获时传 `is_extend_in_batch=False`，draft-extend runner 传 `True`。

---

## 7. 缓冲区管理（DeepEPBuffer）

`DeepEPBuffer`（deepep.py:161）管理进程级共享的 `deep_ep.Buffer`。要点：
- **单个共享 Buffer 服务两种模式**（不是两份），进程级状态挂在 `get_resources().buffers["deepep_ep_state"]`（deepep.py:165）。切模式只是重配/清理，不重复分配。
- `get_deepep_buffer`（deepep.py:184）按模式估算大小：
  - **NORMAL**（deepep.py:203-218）：遍历 dispatch/combine config，取 `get_nvl_buffer_size_hint` 与 `get_rdma_buffer_size_hint` 的 max → `num_nvl_bytes` + `num_rdma_bytes`（**NVL + RDMA 双份**）。
  - **LOW_LATENCY**（deepep.py:219-230）：断言 `num_experts % group.size()==0`，只算 `get_low_latency_rdma_size_hint(...)` → **仅 RDMA 字节**。
  - `num_qps_per_rank`（deepep.py:232）：NORMAL=`num_sms`，LL=`num_experts//group.size()`，AUTO 取二者 max。
- 模式切换：`set_dispatch_mode_as_low_latency`（deepep.py:301）在 NORMAL→LL 切换时会 `clean_low_latency_buffer`（deepep.py:304-305）。

配置：`--deepep-config`（server_args.py:2005）→ `DeepEPConfig`（deepep.py:318），解析 JSON 的 `normal_dispatch`/`normal_combine` 成 `deep_ep.Config`。`--deepep-dispatcher-output-dtype`（server_args.py:1966）控制 dispatch 时激活的量化类型（`get_deepep_output_dtype`，utils.py:216，优先级：server arg → 弃用的 `SGLANG_DEEPEP_BF16_DISPATCH` → NVFP4 → 量化配置 → 后端默认 → FP8 兜底）。

---

## 8. 通信-计算重叠（overlap）

DeepEP 通信开销大，SGLang 用多层重叠机制把它藏到计算背后：

1. **两阶段拆分本身**：`dispatch_a`/`dispatch_b`、`combine_a`/`combine_b` 让调度器在「发起」与「等待」之间塞计算。
2. **异步事件**：NORMAL 用 `async_finish` + `Buffer.capture()` + `event.current_stream_wait()`（deepep.py:519/530/619/624）。
3. **recv hook**：LL 的 `return_recv_hook=True` 时返回一个 `hook` 回调，稍后调用才阻塞在接收上（deepep.py:704/788），从而先发起通信、去干别的、需要结果时再 `hook()`。
4. **SBO（single-batch overlap，单批重叠）**：`CombineOverlapArgs`（`batch_overlap/single_batch_overlap.py:62`）、`DownGemmOverlapArgs`（:74），在 `compute_overlap_args`（:81）里构建，把 **LL combine 的发送与 down-GEMM 重叠**。发送用的 SM 数由 `SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS` 控制（Blackwell 默认 32、其余 3）。通过 `DeepEPDispatcher.set_overlap_args`（deepep.py:996）接入。
5. **TBO（two-batch overlap，双批重叠）**：`batch_overlap/two_batch_overlap.py:432` 按 microbatch 各自 `resolve` 模式，两个微批的通信与计算交错。

---

## 9. 完整调用时序

### NORMAL（prefill / 大 batch）
```
DeepEPMoE.forward_impl (layer.py:177 → deprecate 分支 183 → FusedMoE+MoeRunner)
 │
 ├─ dispatcher.dispatch
 │    ├─ dispatch_a (deepep.py:503)      [可选 FP8 量化激活]
 │    └─ dispatch_b → _dispatch_core:
 │         buffer.get_dispatch_layout    (deepep.py:559)  ← 算路由计划
 │         buffer.dispatch               (deepep.py:571)  ← all-to-all 发送, 存 self.handle
 │       ⇒ DeepEPNormalDispatchOutput(num_recv_tokens_per_expert)
 │
 ├─ [注册表] pre_permute_deepep_normal_to_deep_gemm (deep_gemm.py:800)
 │       ep_scatter ⇒ contiguous 布局 + m_indices
 ├─ grouped_gemm_nt_*_contiguous          (deep_gemm.py:180)  ← 一次 grouped GEMM(gate/up), 激活, 再一次(down)
 ├─ post_permute_deep_gemm_to_deepep_normal (deep_gemm.py:893)
 │       ep_gather + topk_weights 加权
 │
 └─ dispatcher.combine → _combine_core:
      buffer.combine(x, self.handle)      (deepep.py:632)  ← 反向 all-to-all(用 dispatch 的 handle)
```

### LOW_LATENCY（decode）
```
DeepEPMoE.forward_impl (layer.py:177)
 │
 ├─ dispatch_a (deepep.py:667, 算 expected_m 675)
 │    └─ _dispatch_core:
 │         buffer.low_latency_dispatch    (deepep.py:749)  ← 定长 masked 布局(E,max_m,H), masked_m
 │       ⇒ DeepEPLLDispatchOutput(masked_m, expected_m)
 │
 ├─ [注册表] pre_permute_deepep_ll_to_deep_gemm (deep_gemm.py:756)  透传 masked_m/expected_m
 ├─ grouped_gemm_nt_*_masked              (deep_gemm.py:388)  ← masked grouped GEMM (+SBO down-gemm 重叠)
 ├─ post_permute_deep_gemm_to_deepep_ll   (deep_gemm.py:784)
 │
 └─ combine_a/_combine_core:
      buffer.low_latency_combine(x, topk_idx, topk_weights, handle) (deepep.py:830)
      ← 反向 all-to-all + topk 加权求和(在 kernel 内部完成)
```

---

## 10. 文件行号速查表

| 主题 | 符号 | 位置 |
|---|---|---|
| dispatch 输出(normal) | `DeepEPNormalDispatchOutput` | deepep.py:95 |
| dispatch 输出(LL) | `DeepEPLLDispatchOutput` | deepep.py:109 |
| 模式枚举 | `DeepEPMode` / `resolve` | utils.py:171 / 181 |
| 缓冲区 | `DeepEPBuffer` / `get_deepep_buffer` | deepep.py:161 / 184 |
| DeepEP 配置 | `DeepEPConfig` | deepep.py:318 |
| normal 实现 | `_DeepEPDispatcherImplNormal` | deepep.py:495 |
| normal 核心 | `_dispatch_core`(get_dispatch_layout+dispatch) | deepep.py:545/559/571 |
| normal combine | `_combine_core`(buffer.combine) | deepep.py:632 |
| LL 实现 | `_DeepEPDispatcherImplLowLatency` | deepep.py:655 |
| LL dispatch | `low_latency_dispatch` | deepep.py:749 |
| LL combine(带加权) | `low_latency_combine` | deepep.py:830 |
| 公共门面 | `DeepEPDispatcher` | deepep.py:863 |
| 两阶段状态机 | `_Stage` | deepep.py:855 |
| 模式动态选择 | `_get_impl` | deepep.py:975 |
| overlap 接入 | `set_overlap_args` | deepep.py:996 |
| normal→contiguous | `pre_permute_deepep_normal_to_deep_gemm` | deep_gemm.py:800 |
| LL→masked | `pre_permute_deepep_ll_to_deep_gemm` | deep_gemm.py:756 |
| contiguous GEMM | `_run_contiguous_gemm` | deep_gemm.py:180 |
| masked GEMM | `_run_masked_gemm` | deep_gemm.py:388 |
| EP-MoE 层 | `DeepEPMoE.forward_impl` | ep_moe/layer.py:177 |
| a2a 后端选择 | `--moe-a2a-backend` / `get_moe_impl_class` | server_args.py:1928 / layer.py:279 |
| 模式开关 | `--deepep-mode` | server_args.py:1958 |
| SBO 重叠 | `single_batch_overlap.py` | batch_overlap/single_batch_overlap.py:62 |
| TBO 重叠 | `two_batch_overlap.py` | batch_overlap/two_batch_overlap.py:432 |
| extend 标志同步 | `set/get_is_extend_in_batch` | dp_attention.py:274/283 |

---

## 与其他文档的关系

- 上游话题《grouped GEMM》：本文的 dispatch 输出布局正是 grouped GEMM 的输入；contiguous 对应 `m_indices`、masked 对应 `masked_m`。
- `deepseek_v4_pd_disaggregation_request_lifecycle.md`：PD 分离下的跨实例 KV 传输是另一套通信（ZMQ/RDMA 单边写），与本文的 all-to-all 专家通信独立。

