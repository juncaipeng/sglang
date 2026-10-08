# SGLang R3（Return Routed Experts / 返回专家路由）架构详解

> 本文梳理 SGLang 中 **R3 特性**（`--enable-return-routed-experts`）：如何在 MoE 前向中抓取每个
> token 每一层被路由到的 top-k 专家 id，并随响应返回给客户端；重点讲清 **单轮请求 / 多轮请求（前缀缓存）/ PD 分离**
> 三种场景下的行为差异与边界。
>
> 面向读者：需要用 R3 做 RL 训练数据回放的同学、以及要改动 MoE 路由 / capturer / PD 传输的开发者。
>
> 代码基准：`main` 分支（撰写时）。文中所有行号均可点击跳转。

---

## 0. 名字由来：为什么叫 "R3"

"R3" 是这个特性的**内部代号**，就是 **R**eturn **R**outed expe**R**ts / Return Routed Experts 的缩写。
在代码与提交里都能看到：

- 代码注释：[`disable_routed_experts_capture_for_draft`](../../../python/sglang/srt/state_capturer/routed_experts.py#L159) 的 docstring 明写
  "Opt every draft MoE `TopK` out of routed-experts (**R3**) capture."
- 提交历史：`4a279d9c3 [R3] Avoid implicit CUDA sync in routed experts DP slicing (#24550)`
- 首个特性 PR：`b2420d72f [RL] DeepEP support for --enable-return-routed-experts (#16859)` —— 明确是给
  **RL（强化学习）** 用的：RL 训练需要知道推理时每个 token 命中的专家，用于重要性采样 / 回放。

所以：**问 "sglang 如何返回 r3" = 问 "sglang 如何返回专家路由（routed experts）"**。

---

## 1. 一分钟上手

### 1.1 启动开关

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-FP8 \
    --enable-return-routed-experts
```

对应 server arg：[`server_args.py:3308`](../../../python/sglang/srt/server_args.py#L3308)
`enable_return_routed_experts`（默认 `False`）。开启后会**常驻分配**一块 device buffer + 一块 pinned host
buffer（见 §3），因此不用时不要开。

### 1.2 请求级参数

| 参数 | 类型 | 含义 | 默认 |
|---|---|---|---|
| `return_routed_experts` | bool | 本次请求是否返回专家路由 | `False` |
| `routed_experts_start_len` | int | 返回的**绝对起始位置**；响应覆盖 `[start_len, seqlen-1)`；必须 ∈ `[0, prompt_tokens]`；`0` = 整条序列 | `0` |

定义见 [`io_struct.py:220-225`](../../../python/sglang/srt/managers/io_struct.py#L220)、OpenAI 协议
[`protocol.py:342`](../../../python/sglang/srt/entrypoints/openai/protocol.py#L342) 与
[`protocol.py:758`](../../../python/sglang/srt/entrypoints/openai/protocol.py#L758)。

### 1.3 返回格式

- `/generate`：`meta_info.routed_experts` —— **base64 编码的 int32 bytes**。
- OpenAI `/v1/chat/completions`、`/v1/completions`：放在 `sglext.routed_experts`。

解码回张量（形状 `[rows, num_layers, topk]`，`rows = seqlen - 1 - start_len`）：

```python
import numpy as np, pybase64
# extract_routed_experts_from_meta_info: python/sglang/srt/state_capturer/routed_experts.py:148
b64 = data["meta_info"]["routed_experts"]
arr = np.frombuffer(pybase64.b64decode(b64.encode()), dtype=np.int32)
arr = arr.reshape(rows, num_layers, topk)   # arr[t, l, k] = 第 t 个位置在第 l 层命中的第 k 个专家 id
```

参考实现：[`routed_experts.py:148`](../../../python/sglang/srt/state_capturer/routed_experts.py#L148)。

---

## 2. 一句话看懂核心设计

> **routed experts 与 KV Cache 用同一套"KV 槽号（`out_cache_loc`）"寻址，是挂在 KV 池旁边的一块平行 buffer。**

这一句话是理解全部三种场景的钥匙：

- 一个 token 的专家路由，存放位置 = 该 token 的 KV 存放槽号；
- 取出时也复用 `req_to_token[req_pool_idx]` 这张"请求 → token → 槽号"的映射表（与读 KV 完全一样）；
- **结论**：只要一个 token 的 KV 有效且槽号未被复用，它的专家路由就有效——路由与 KV 生命周期严格绑定。

正因为如此，**前缀缓存（多轮）天然正确**，而 **PD 分离会因为"路由 buffer 不随 KV 传输"而在 prompt 段出现缺口**（§7 详述）。

---

## 3. 三层数据结构

capturer 的实现分三层，定义在 [`state_capturer/base.py`](../../../python/sglang/srt/state_capturer/base.py) 与
[`state_capturer/routed_experts.py`](../../../python/sglang/srt/state_capturer/routed_experts.py)。

```
BaseTopkCapturer (base.py:100)
 ├─ device_cache : BaseDeviceCache (base.py:20)
 │     buffer 形状 (max_batch_size, num_layers, device_topk_size)  int32  在 GPU
 │     └─ 按"批内 token 位置"写：buffer[:batch, layer_id, :] = topk_ids   (base.py:41)
 │
 └─ host_cache   : BaseHostCache   (base.py:54)
       buffer 形状 (num_tokens=KV槽总数, num_layers, topk_size)   int32  pinned CPU
       └─ 按"KV 槽号 out_cache_loc"写：buffer[out_cache_loc] = slice_gpu (base.py:97/185)
```

| 层 | 位置 | 形状首维 | 索引语义 | 作用 |
|---|---|---|---|---|
| `device_cache` | GPU | `max_batch_size` | **批内位置** | 前向中每层就地写入，供 overlap 调度低成本累积 |
| `host_cache` | pinned CPU | `num_tokens`（= KV 池总槽数） | **KV 槽号** | 前向结束按 `out_cache_loc` D2H，落成与 KV 同址的持久副本 |

几个尺寸细节（[`routed_experts.py:55-103`](../../../python/sglang/srt/state_capturer/routed_experts.py#L55)）：

- `topk_size = num_experts_per_tok`；`num_layers = num_hidden_layers`。
- **device buffer 多留几列**：`device_topk_size = topk_size + num_fused_shared_experts`（融合共享专家占额外列）；
  写入 host 时用 `[:topk_size]` 截掉共享专家列，用户只拿到路由专家。
- `max_batch_size = max(chunked_prefill_size, max_running_requests) * dp_size` —— 乘 `dp_size` 覆盖 DP 拼批后的全批，
  否则 `dp_rank>0` 时切片会越界（注释见 [`routed_experts.py:68-76`](../../../python/sglang/srt/state_capturer/routed_experts.py#L68)）。

全局单例通过 RuntimeContext 的 resources 持有：
[`get_global_experts_capturer()`](../../../python/sglang/srt/state_capturer/routed_experts.py#L136) /
[`set_global_experts_capturer()`](../../../python/sglang/srt/state_capturer/routed_experts.py#L142)。

---

## 4. 端到端数据通路（系统图）

```
                       ┌──────────────── 前向：每层 MoE gate ────────────────┐
   hidden_states ───►  TopK.forward (topk.py:395)                            │
                          │                                                   │
                          ▼                                                   │
                       select_experts (topk.py:2016)                          │
                          │  → topk_ids / topk_weights / router_logits        │
                          ▼                                                   │
                       _post_process_topk_ids (topk.py:1850)                  │
                          │                                                   │
                          └─► capture_routed_experts_if_allowed (topk.py:1831)│  ← 唯一抓取点
                                    │  gated by allow_routed_experts_capture   │
                                    ▼                                          │
                          cap.capture(layer_id, topk_ids)                      │
                          device_cache.buffer[:batch, layer, :] = ids          │  (base.py:41)
                          └────────────────────────────────────────────────── ┘
                                    │  forward 结束： on_forward_end (base.py:164)
                                    ▼   按 out_cache_loc 做 D2H（overlap 下走结果拷贝流）
                          host_cache.buffer[out_cache_loc] = slice_gpu          (base.py:97/185)
                                    │
                                    │  请求 finished() 时收集一次
                                    ▼
   get_topk: req_to_token[req_pool_idx][start_len : seqlen-1] → 槽号 → 取行     (base.py:147)
   调用点 _maybe_collect_routed_experts (batch_result_processor.py:104)
                                    │  → req.routed_experts  (形状 [rows, layers, topk])
                                    ▼
   output_streamer (output_streamer.py:528) → BatchTokenIDOutput.routed_experts (:603)
                                    │
                                    ▼
   detokenizer _b64_encode_per_request (detokenizer_manager.py:411) → base64 字符串
                                    │
                                    ▼
   tokenizer_manager: meta_info["routed_experts"] = val (tokenizer_manager.py:2082)
                                    │
                                    ▼
                              客户端响应 meta_info / sglext
```

分阶段说明：

| 阶段 | 文件:行 | 干了什么 |
|---|---|---|
| ① 抓取 | [`topk.py:1866`](../../../python/sglang/srt/layers/moe/topk.py#L1866) | 每层 gate 出 `topk_ids` 后，统一在这里写 device buffer |
| ② D2H | [`base.py:164`](../../../python/sglang/srt/state_capturer/base.py#L164) `on_forward_end` | 按 `out_cache_loc` 把 device buffer 搬到 host buffer；overlap 下返回 `TopkCaptureOutput` 交给结果流异步 finalize |
| ③ 收集 | [`batch_result_processor.py:104`](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L104) | 请求 finish 时 `get_topk` 拉出 `[start_len, seqlen-1)` 行 |
| ④ 流式打包 | [`output_streamer.py:528`](../../../python/sglang/srt/managers/scheduler_components/output_streamer.py#L528) / [:603](../../../python/sglang/srt/managers/scheduler_components/output_streamer.py#L603) | 汇进 batch 输出结构 |
| ⑤ base64 | [`detokenizer_manager.py:437`](../../../python/sglang/srt/managers/detokenizer_manager.py#L437) | `_b64_encode_per_request` 编码，避开 tokenizer 热路径 |
| ⑥ 回填 meta_info | [`tokenizer_manager.py:2082`](../../../python/sglang/srt/managers/tokenizer_manager.py#L2082) | 写入 `meta_info["routed_experts"]`（`skip_tokenizer_init` 时张量在此临时编码） |

> ⚠️ **非增量返回**：R3 只在请求 `finished()` 时一次性给全量（`_maybe_collect_routed_experts` 在 finish 分支调用），
> 不做逐 token 流式增量。

---

## 5. 抓取点的三个易错细节

### 5.1 唯一抓取点（防止某个后端偷偷绕过）

不同 MoE 后端（triton / DeepGEMM / cutlass / flashinfer）走不同 kernel，但**都被强制经过**
[`capture_routed_experts_if_allowed`](../../../python/sglang/srt/layers/moe/topk.py#L1831) 这一个函数，
再由它调 `get_global_experts_capturer().capture(...)`。这样做的目的（见其 docstring）是：把 draft 侧的 opt-out
集中到一个闸门，避免被某个内联的 capturer 调用绕过。闸门条件是 `topk_config.allow_routed_experts_capture`。

### 5.2 DP + DeepEP：capture 前先 all-gather

DeepEP 的 all-to-all 之后，每个 attn-TP rank **只持有自己那一片 scatter 过的 `topk_ids`**。若直接写 device buffer，
会缺其它 rank 的 token。因此 `RoutedExpertsCapturer.capture`
（[`routed_experts.py:105`](../../../python/sglang/srt/state_capturer/routed_experts.py#L105)）在 DeepEP 下先
`attn_tp_all_gather_into_tensor` 补齐整批，再交给 `super().capture()`：

```python
def capture(self, layer_id, topk_indices):
    if get_moe_a2a_backend().is_deepep():
        local_topk = topk_indices
        topk_indices = self.gather_buffer[: local_topk.size(0) * attn_tp_size]
        attn_tp_all_gather_into_tensor(topk_indices, local_topk)   # 补全整批
    super().capture(layer_id, topk_indices)
```

对应地，取本 rank 数据的 `_get_local_slice`（[`routed_experts.py:114`](../../../python/sglang/srt/state_capturer/routed_experts.py#L114)）
要分两种排布：
- **DeepEP**：gather 后本 rank 数据落在 buffer 头部 `[0:N_local]`；
- **非 DeepEP + DP-attn**：按 `get_dp_local_slice_cpu` 给的 `[start_pos:end_pos]` 切（且在 CPU 上算偏移，避免 GPU→CPU 同步破坏 overlap，这正是 `#24550 [R3] Avoid implicit CUDA sync` 修的问题）。

### 5.3 投机解码 draft 层必须 opt-out

draft 模型也有 MoE `TopK`，但它**不能**写 target 的进程级全局 buffer（否则污染真实路由）。
[`disable_routed_experts_capture_for_draft`](../../../python/sglang/srt/state_capturer/routed_experts.py#L159) 遍历 draft
模型所有 `TopK` 模块，把 `topk_config.allow_routed_experts_capture` 置 `False`
（PR `#26980 Skip routed expert capture for draft model under spec v2`）。

---

## 6. 收集与返回的边界语义

### 6.1 为什么是 `[start_len, seqlen-1)`

`get_topk`（[`base.py:147`](../../../python/sglang/srt/state_capturer/base.py#L147)）：

```python
start_len = min(start_len, seqlen - 1)
cache_pool_idx = req_to_token_pool.req_to_token[req_pool_idx][start_len : seqlen - 1].cpu().clone()
return self.host_cache.buffer[cache_pool_idx]
```

- `seqlen = len(origin_input_ids) + len(output_ids_through_stop)`（[`batch_result_processor.py:123`](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L123)）。
- **末尾 `-1`**：最后一个位置的路由用于预测"序列之外的下一个 token"，对 RL 回放无意义，故丢弃。返回行数
  `rows = seqlen - 1 - start_len`。
- `_maybe_collect_routed_experts` 里有一条 **soft 校验**：若实际行数 ≠ `seqlen - 1 - start_len` 就打
  warning（[`batch_result_processor.py:131-147`](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L131)），用于抓静默回归。

### 6.2 `start_len` 参数校验

在 [`scheduler.py:2410-2429`](../../../python/sglang/srt/managers/scheduler.py#L2410)：
- `< 0` → abort；
- `> len(origin_input_ids)`（即超过 prompt 长度）→ abort。

即 `start_len` 只能落在 prompt 区间内，用来"跳过前面一段 prompt"，**不能**用它跳过整个 prompt 只取 decode 段
（因为上限就是 `prompt_tokens`，而返回窗口右端是 `seqlen-1`，一定包含 decode 段）。

### 6.3 base64 而非裸数组

序列越长、层数越多，路由张量越大（`rows × num_layers × topk × 4B`）。为避免 JSON 序列化开销，
detokenizer 侧统一 base64（[`detokenizer_manager.py:411`](../../../python/sglang/srt/managers/detokenizer_manager.py#L411)），
客户端自行 `np.frombuffer` 还原。多 tokenizer worker 模式下按请求下标切分
（[`multi_tokenizer_mixin.py:244`](../../../python/sglang/srt/managers/multi_tokenizer_mixin.py#L244) / [:352](../../../python/sglang/srt/managers/multi_tokenizer_mixin.py#L352)）。

---

## 7. 三种场景深度对比（本文重点）

一切差异都源于 §2 的核心设计：**路由 buffer 按 KV 槽号存，随 KV 生命周期存活，但收集发生在"请求 finish 所在的那个实例"上，读的是该实例的本地 host buffer。**

### 7.1 总览表

| 场景 | prompt 段路由 | decode 段路由 | 完整正确？ | 根因 |
|---|---|---|---|---|
| **单轮（单实例）** | ✅ extend 前向抓取 | ✅ 每步 decode 抓取 | ✅ 完整 | 同一实例、同一 host buffer，`get_topk` 一次读全 |
| **多轮（单实例 + radix cache）** | ✅ 命中前缀复用上一轮抓取值 | ✅ | ✅ 完整且一致 | host buffer 与 KV 同槽号；缓存前缀的槽被 radix 引用未被复用，路由值即当初计算值 |
| **PD 分离** | ⚠️ 抓在 **prefill 实例** buffer，**不随 KV 传输** | ✅ 抓在 **decode 实例** buffer | ⚠️ **prompt 段不可靠** | 收集在 decode 实例读**本地** buffer，其 prompt 槽位从未被写入本次路由 |

### 7.2 单轮请求（baseline）

同一个 scheduler 实例既做 prefill/extend 又做 decode：

```
实例 A（单机）
 ├─ EXTEND(prompt)  → capture 每层 → host_cache[prompt 槽]   ← prompt 路由 ✅
 ├─ DECODE step1    → capture 每层 → host_cache[gen 槽1]      ← decode 路由 ✅
 ├─ DECODE step2    → ...          → host_cache[gen 槽2]
 └─ finish → get_topk 读 [start_len, seqlen-1)（prompt+gen 全在本地 buffer）→ 返回 ✅
```

收集入口有两个，都在实例 A 本地：
- 请求在 prefill 阶段就结束（如 `max_new_tokens=1`）：[`batch_result_processor.py:240`](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L240)（`process_batch_result_prefill`）。
- 请求在 decode 阶段结束（常态）：[`batch_result_processor.py:959`](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L959)（`process_batch_result_decode`）。

### 7.3 多轮请求（前缀缓存的巧妙之处）

第二轮请求命中 radix cache 的共享前缀时，**这些前缀 token 的前向不会重算**，因此它们的路由**不会被重新 capture**。
但因为：

1. host buffer 与 KV **同槽号**寻址；
2. radix cache 命中意味着这些槽仍被引用、**未被回收复用**，槽里的 KV 就是这些 token 的 KV；
3. 上一轮（或首次计算时）写入的路由值仍留在这些槽对应的 host buffer 行里；

所以 `get_topk` 通过 `req_to_token` 拿到这些槽号，读出的正是**这些 token 当初真实的路由**——多轮天然正确、且与"不开缓存全量重算"逐位一致。

> **CI 如何守住这个不变量**：[`test_return_routed_experts.py:44-77`](../../../test/registered/rl/test_return_routed_experts.py#L44) 用两台服务器对拍：
> - baseline：`--disable-overlap-schedule --disable-cuda-graph --disable-radix-cache`（确定性 ground truth）；
> - reference：以上优化全开（含 radix cache）。
>
> 两者的 `routed_experts` 必须逐位相等。如果"缓存前缀复用旧路由值"这个假设被破坏，两边就会分叉。这条测试同时覆盖
> `--tp 4 --dp 2 --enable-dp-attention --moe-a2a-backend deepep`（attn_tp_size=2，命中 §5.2 的 all-gather 热路径）。

> 边界：若某个前缀槽在两轮之间**被驱逐并复用**给了别的 token，那么下一轮会重新计算该前缀（radix 未命中），从而刷新路由，
> 一致性仍然成立——不存在"读到别的请求路由"的窗口。

### 7.4 PD 分离（存在 prompt 段缺口）

PD 下 prefill 与 decode 在**不同实例**、各有独立的 host buffer：

```
Prefill 实例 P                             Decode 实例 D
 ├─ EXTEND(prompt)                          （PREBUILT：不重算 prompt 前向）
 │   capture 每层                            ├─ 收到 KV 传输 → 写入 D 的 KV 池
 │   → P.host_cache[prompt 槽] ✅            │   但 D.host_cache[prompt 槽] 从未被写 ❌
 │   finalize()  (prefill.py:653)           │
 ├─ 传输 KV 给 D ────────────────────────►  ├─ DECODE step → capture → D.host_cache[gen 槽] ✅
 │   （KV / aux 里都不含 routed_experts）    └─ finish → get_topk 读本地 D.host_cache
 └─ 请求转入 inflight 队列，不再返回响应体          [start_len, seqlen-1) 覆盖 prompt+gen
                                                 · gen 段 → 正确 ✅
                                                 · prompt 段 → 读到该槽遗留的旧值/零 ❌
```

关键代码事实（我查遍 `disaggregation/` 目录得到）：

- Prefill 实例只在 [`prefill.py:652-654`](../../../python/sglang/srt/disaggregation/prefill.py#L652) 把
  `routed_experts_output.finalize()`（把 prompt 路由落进 **P 自己的** host buffer），随后传 KV。
  它走的是 `process_batch_result_disagg_prefill`，**不调用** `_maybe_collect_routed_experts`，也**不把路由返回给客户端**。
- `disaggregation/` 下 `routed_experts` 仅出现两处：上面这个 `finalize`，以及 encode/session 的**参数透传**
  （[`encode_receiver.py:1991`](../../../python/sglang/srt/disaggregation/encode_receiver.py#L1991)、
  [`session_controller.py:310`](../../../python/sglang/srt/session/session_controller.py#L310)）。
  **没有任何把 routed_experts host buffer 从 P RDMA 传给 D 的通路**（KV 通道、aux 通道都不含它）。
- 请求最终在 D 上 finish 并组包，`_maybe_collect_routed_experts` 读的是 **D 的本地 buffer**
  （[`batch_result_processor.py:959`](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L959)）。

**结论**：现有 main 上，PD 分离时返回的 `routed_experts` **只有 decode 段可靠**；prompt 段对应的行读到的是
D 上那些 KV 槽的遗留内容（很可能是 0 或上一个占用该槽的请求的路由），**不是本次 prompt 的真实路由**。而
`routed_experts_start_len` 上限是 `prompt_tokens`、窗口右端固定为 `seqlen-1`，**无法**用它把 prompt 段完全裁掉。

> 使用建议：若在 **PD + RL** 下需要全序列路由，当前特性满足不了 prompt 段；只能：
> (a) 用非 PD（colocate）实例跑需要 R3 的样本；或
> (b) 仅消费 decode 段路由。
>
> 该判断基于阅读当前 main 代码、且未发现弥补的传输路径；R3 的 CI 也只覆盖 colocate（§7.3）。若后续有 PR 增加
> "prompt 路由随 KV 一并传输 / 或由 prefill 直接回传"，本节需更新。

---

## 8. 与相邻特性的关系

| 特性 | 抓什么 | capturer | 复用了什么 |
|---|---|---|---|
| **R3 / routed experts** | 每层 MoE 命中的 top-k 专家 id | `RoutedExpertsCapturer`（[routed_experts.py:19](../../../python/sglang/srt/state_capturer/routed_experts.py#L19)） | 本文所述 device/host 双 buffer + 槽号寻址 |
| **indexer topk** | DSA/V4 indexer 选中的 top-k 页/索引 | `get_global_indexer_capturer`（[indexer_topk.py](../../../python/sglang/srt/state_capturer/indexer_topk.py)） | 同一套 `BaseTopkCapturer` 基类；收集入口 `_maybe_collect_indexer_topk`（[batch_result_processor.py:149](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L149)） |
| **return hidden states** | 每 token 隐藏态 | —— | 同样按 `finished_len` 在 output_streamer 汇总 |

三者共享"请求 finish 时按 `req_to_token` 拉本地 buffer → base64 → meta_info"的返回骨架，边界坑（PD、seqlen-1）也类似。

## 9. 文件行号索引（速查）

| 关注点 | 文件:行 |
|---|---|
| 启动开关 `enable_return_routed_experts` | [server_args.py:3308](../../../python/sglang/srt/server_args.py#L3308) |
| 请求参数 `return_routed_experts` / `routed_experts_start_len` | [io_struct.py:220](../../../python/sglang/srt/managers/io_struct.py#L220) |
| OpenAI 协议字段 | [protocol.py:342](../../../python/sglang/srt/entrypoints/openai/protocol.py#L342) / [:758](../../../python/sglang/srt/entrypoints/openai/protocol.py#L758) |
| 参数校验（abort） | [scheduler.py:2410](../../../python/sglang/srt/managers/scheduler.py#L2410) |
| capturer 创建（受开关门控） | [routed_experts.py:29](../../../python/sglang/srt/state_capturer/routed_experts.py#L29) |
| device / host / TopkCaptureOutput 基类 | [base.py:20](../../../python/sglang/srt/state_capturer/base.py#L20) / [:54](../../../python/sglang/srt/state_capturer/base.py#L54) / [:79](../../../python/sglang/srt/state_capturer/base.py#L79) |
| 唯一抓取点 | [topk.py:1831](../../../python/sglang/srt/layers/moe/topk.py#L1831)（调用点 [:1866](../../../python/sglang/srt/layers/moe/topk.py#L1866)） |
| DeepEP all-gather capture | [routed_experts.py:105](../../../python/sglang/srt/state_capturer/routed_experts.py#L105) |
| DP 本地切片（CPU 上算偏移） | [routed_experts.py:114](../../../python/sglang/srt/state_capturer/routed_experts.py#L114) |
| D2H `on_forward_end` | [base.py:164](../../../python/sglang/srt/state_capturer/base.py#L164) |
| 收集 `get_topk`（[start_len, seqlen-1)） | [base.py:147](../../../python/sglang/srt/state_capturer/base.py#L147) |
| 收集入口（prefill-finish / decode-finish） | [batch_result_processor.py:240](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L240) / [:959](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L959) |
| 收集实现 + 行数校验 | [batch_result_processor.py:104](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L104) |
| PD prefill finalize（不返回、不传输） | [prefill.py:652](../../../python/sglang/srt/disaggregation/prefill.py#L652) |
| draft 层 opt-out | [routed_experts.py:159](../../../python/sglang/srt/state_capturer/routed_experts.py#L159) |
| output_streamer 汇总 | [output_streamer.py:528](../../../python/sglang/srt/managers/scheduler_components/output_streamer.py#L528) / [:603](../../../python/sglang/srt/managers/scheduler_components/output_streamer.py#L603) |
| detokenizer base64 | [detokenizer_manager.py:411](../../../python/sglang/srt/managers/detokenizer_manager.py#L411) / [:437](../../../python/sglang/srt/managers/detokenizer_manager.py#L437) |
| 回填 meta_info | [tokenizer_manager.py:2082](../../../python/sglang/srt/managers/tokenizer_manager.py#L2082) |
| 客户端解码工具 | [routed_experts.py:148](../../../python/sglang/srt/state_capturer/routed_experts.py#L148) |
| E2E 测试 | [test/registered/rl/test_return_routed_experts.py](../../../test/registered/rl/test_return_routed_experts.py) |

## 10. 演进（提交历史）

| commit | 内容 |
|---|---|
| `b2420d72f` (#16859) | `[RL]` 首次为 `--enable-return-routed-experts` 加 DeepEP 支持 |
| `08d4c2072` (#24450) | 把 capturers 迁到 `srt/state_capturer/` |
| `4a279d9c3` (#24550) | `[R3]` 避免 DP 切片里的隐式 CUDA 同步（改到 CPU 上算偏移） |
| `376635c1e` (#26123) | 修复 DP-attn 下 routed experts device buffer 溢出（`max_batch_size *= dp_size`） |
| `e4bf0043f` (#26980) | 投机 v2 下跳过 draft 模型的路由抓取 |
| `1dc48c2c3` (#31160) | 吸收 capturer 初始化、抽出共享 mooncake 门控 |

---

## 附：核心心智模型（一句话）

> **R3 把"每层命中的专家 id"当作一份与 KV Cache 同槽号、同生命周期的旁路数据；单实例（含开缓存的多轮）下按
> `req_to_token` 一次性读全即可，PD 分离下因为这份旁路数据不随 KV 传输、且收集只发生在 decode 实例本地，
> 所以 prompt 段目前是缺口。**

