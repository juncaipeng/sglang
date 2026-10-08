# SGLang PD 分离「真抢占重引导」（True-Retraction Rebootstrap）方案系统梳理

> 核心文件：`python/sglang/srt/managers/scheduler.py`（pause/continue）、`python/sglang/srt/disaggregation/decode.py`（DecodePreallocQueue）、`python/sglang/srt/disaggregation/common/conn.py`（KV manager 侧 HTTP 驱动）、`python/sglang/srt/managers/schedule_batch.py`（Req 字段与 payload 构造）
> 相关文件：`python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py`、`python/sglang/srt/managers/io_struct.py`（PauseGenerationReqInput / ContinueGenerationReqInput）、`python/sglang/srt/entrypoints/http_server.py`（/pause_generation）
> 目标读者：需要理解、调优或扩展 SGLang「PD 分离 + RL 在线更权」场景下抢占恢复机制的工程师。
>
> 本文由浅入深，覆盖：为什么普通「offload→load」恢复在 RL 场景失效、rebootstrap 的核心思想、三种 pause 模式对比、端到端流程与数据流图、每个环节的代码落点与行号、关键字段语义（边界 token 回放 / logprob 语义 / 长度计算差异）、为什么走原生 `/generate` 而非 OpenAI 接口、URL 自发现机制、并发 leader 选举、失败处理、约束与限制、行号索引。

---

## 目录

1. [总览：一句话定位](#1-总览一句话定位)
2. [背景：普通 PD decode 抢占恢复为何在 RL 下失效](#2-背景普通-pd-decode-抢占恢复为何在-rl-下失效)
3. [核心思想与三种 pause 模式对比](#3-核心思想与三种-pause-模式对比)
4. [整体流程与数据流图](#4-整体流程与数据流图)
5. [阶段 1：pause_generation(retract) —— 丢 KV + 暂存](#5-阶段-1pause_generationretract--丢-kv--暂存)
6. [阶段 2：update_weights 与暂停窗口的不变量](#6-阶段-2update_weights-与暂停窗口的不变量)
7. [阶段 3：continue_generation —— 入队 rebootstrap](#7-阶段-3continue_generation--入队-rebootstrap)
8. [阶段 4：预分配 + submit_prefill_recompute](#8-阶段-4预分配--submit_prefill_recompute)
9. [阶段 5：prefill 重算与 KV 传回](#9-阶段-5prefill-重算与-kv-传回)
10. [阶段 6：传输 commit —— 边界 token 回放](#10-阶段-6传输-commit--边界-token-回放)
11. [关键字段与数据结构](#11-关键字段与数据结构)
12. [build_rebootstrap_payload 与为什么走原生 /generate](#12-build_rebootstrap_payload-与为什么走原生-generate)
13. [URL 自发现：prefill_http_port 自注册](#13-url-自发现prefill_http_port-自注册)
14. [长度计算差异：_rebootstrap_prefill_len / _pre_alloc_fill_len](#14-长度计算差异_rebootstrap_prefill_len--_pre_alloc_fill_len)
15. [并发模型：leader 选举、执行器、per-thread session](#15-并发模型leader-选举执行器per-thread-session)
16. [失败处理](#16-失败处理)
17. [约束与限制](#17-约束与限制)
18. [关键文件 / 行号索引](#18-关键文件--行号索引)

---

## 1. 总览：一句话定位

**PD 真抢占重引导（True-Retraction Rebootstrap）= 在 PD 分离 + RL 在线更权场景下，当 decode 实例因显存压力（或收到 retract 指令）抢占一个正在解码的请求时，不把它的 KV 缓存到 host 稍后原样搬回，而是丢弃旧 KV，让「原来的那台 prefill 实例」在当前（已更新）权重下重新计算该请求的前缀 KV，再通过已经建立好的 PD KV 传输通道把新 KV 传回 decode。**

关键词拆解：

- **True-Retraction（真抢占）**：不是软暂停（in-place），是真正释放 KV 显存并把请求踢回预分配队列。
- **Rebootstrap（重引导）**：请求重新走一遍 PD 的 bootstrap → 传输流程，只不过 prefill 端做的是「重算」而不是「首次计算」。
- **on-policy（同策略）**：核心动机——RL 训练要求 KV 必须由**最新权重**产生，旧权重算出的 KV 是 off-policy 陈旧数据，不能复用。

这是一个 **PD-decode 专属**特性：只有 `disaggregation_mode == DECODE` 时才走这条路径（scheduler.py:4617）；普通单实例引擎和 PD-prefill 端都不涉及。

---

## 2. 背景：普通 PD decode 抢占恢复为何在 RL 下失效

### 2.1 普通抢占恢复（offload → load）回顾

在普通（非 RL）PD decode 场景中，当 decode 实例显存不足时：

1. `retract_decode` 把请求踢出 running batch；
2. `release_req` 调 `offload_kv_cache`，把该请求的 KV **device → host（CPU）**拷贝一份（schedule_batch.py:1914 附近）；
3. 显存恢复后 `resume_retracted_reqs` + `load_kv_cache` 把 host 上的 KV **原样搬回** device。

这套机制的隐含假设是：**KV 的数值在抢占前后不变**——搬出去什么样，搬回来还是什么样。

### 2.2 RL 在线更权打破了这个假设

RL（强化学习）训练循环中，一次典型的权重更新是：

```
pause_generation(mode="retract")   →   update_weights   →   continue_generation
```

（`PauseGenerationReqInput.mode` 的三种取值见 io_struct.py:1632；`retract` 语义见 io_struct.py:1625-1629）

问题在于：`update_weights` 改变了模型权重。如果沿用「offload→load」，恢复回来的 KV 仍然是**旧权重**算出来的。对 RL 而言这是致命的：

| | offload→load（普通） | rebootstrap（RL） |
|---|---|---|
| 恢复的 KV 来自 | 抢占前的旧权重 | 更新后的新权重 |
| 是否 on-policy | ❌ off-policy | ✅ on-policy |
| device→host 拷贝 | 需要（浪费带宽） | **跳过**（反正要丢） |
| prefill 参与 | 否 | 是（重算前缀 KV） |

所以在 RL 场景，抢占必须伴随「丢弃旧 KV + 用新权重重算」——这就是 rebootstrap 存在的根本原因。注意代码注释明确指出（scheduler.py:4602-4605）：decode 侧 retract **总是** rebootstrap，因此 `retract_all(offload_kv=False)`（scheduler.py:4613）直接跳过无用的 device→host 拷贝。

## 3. 核心思想与三种 pause 模式对比

`pause_generation` 支持三种模式（scheduler.py:4559, io_struct.py:1632），rebootstrap 只与 `retract` 模式相关：

| 模式 | 是否释放 KV | running_batch 处理 | 恢复方式 | 典型用途 |
|---|---|---|---|---|
| `abort` | 释放 | 全部 abort 返回客户端 | 不恢复 | 强制清空 |
| `in_place` | **不释放** | 原地保留（scheduler.py:4563-4571） | 用**旧 KV** 直接续跑 | 短暂暂停、无更权 |
| `retract` | **释放** | 踢回队列（decode 走 rebootstrap） | 用**新 KV** 重算续跑 | RL 在线更权 |

`in_place` 与 `retract` 的本质区别就是「KV 能不能被 flush」：`in_place` 模式下若 running_batch 非空，`flush_cache` 会失败（io_struct.py:1622-1623）；`retract` 模式下 KV 已释放，`update_weights` 的更权后 cache flush 才能安全执行（io_struct.py:1628-1629）。

**为什么 retract 后不能立刻重新入队，而要「暂存」？**（这是理解整个机制的关键设计点）

如果 retract 时立即把请求塞进 decode 的预分配队列（`DecodePreallocQueue.queue`），那么在 pause 窗口内预分配队列就是非空的，调度器不再 idle。而 `update_weights` 的更权后流程会断言「调度器处于 idle」才能刷新缓存，非空会让断言失败并**crash decode worker**（decode.py:624-629）。

因此设计上引入一个**旁路暂存列表** `held_rebootstrap_reqs`（decode.py:356）：retract 时只暂存、不入队；直到 `continue_generation` 时才 `enqueue_held_rebootstrap`（scheduler.py:4667）把它们真正入队。这样整个 pause 窗口预分配队列保持空，更权得以安全进行，且入队后重算天然发生在**新权重**下。

## 4. 整体流程与数据流图

```
┌──────────────────────── DECODE 实例 ────────────────────────┐        ┌──────── PREFILL 实例 ────────┐
│                                                              │        │                              │
│ ① pause_generation(mode="retract")   scheduler.py:4559       │        │                              │
│    ├ 收集 running/last/chunked 未完成请求 (:4578-4596)        │        │                              │
│    ├ retract_all(offload_kv=False)   丢弃 KV (:4606-4614)     │        │                              │
│    └ for req in retract_reqs (decode 分支 :4617-4622):        │        │                              │
│        ├ pd_rebootstrap_forced_output_id = output_ids.pop()  │        │                              │
│        │     (记住已发出的最后一个"边界 token")               │        │                              │
│        ├ pd_rebootstrap_in_progress = True                   │        │                              │
│        └ prealloc_queue.hold_rebootstrap(req)                │        │                              │
│              └ 暂存到 held_rebootstrap_reqs  (不入 queue)     │        │                              │
│                                                              │        │                              │
│ ② update_weights   ← RL 在此更新权重（调度器保持 idle）        │        │                              │
│                                                              │        │                              │
│ ③ continue_generation   scheduler.py:4648                    │        │                              │
│    └ enqueue_held_rebootstrap()  (:4667)                     │        │                              │
│        └ for req: add(req, is_rebootstrap=True) decode.py:638 │        │                              │
│            └ 建 KVReceiver → DecodeRequest(is_rebootstrap)    │        │                              │
│                                                              │        │                              │
│ ④ 预分配循环   decode.py:1023-1329                            │        │                              │
│    ├ input_len = _rebootstrap_prefill_len(req) (:1025)       │        │                              │
│    ├ 关掉 decode radix cache (is_rebootstrap, :1029)          │        │                              │
│    ├ send_metadata(page_indices, ...) 推 dst_kv_indices ─────┼───────▶│  已知本请求要传到哪些槽位     │
│    └ if is_rebootstrap: submit_prefill_recompute (:1315-1319) │        │                              │
│        └ POST http://{host}:{prefill_http_port}/generate ────┼───────▶│ ⑤ 按 bootstrap_room 重算前缀 │
│           payload = build_rebootstrap_payload()              │  HTTP  │    KV（新权重, max_new=1）    │
│                                                              │        │    正常 PD 传输把 KV 写回 ───┼──┐
│ ⑥ DecodeTransferQueue 轮询到 Success   decode.py:1899        │        │    dst_kv_indices           │  │
│    ├ replayed_boundary = is_rebootstrap and forced_id!=None  │        │                              │  │
│    ├ committed_output_id = forced_output_id (回放边界 token)  │◀───────┼──────────────────────────────┘
│    └ 跳过重打 logprob (:1946)                                 │        │                              │
│                                                              │        │                              │
│ ⑦ prepare_for_prebuilt → 恢复解码  decode_schedule_batch_mixin│        │                              │
│    └ pd_rebootstrap_in_progress = False (:75-76)             │        │                              │
└──────────────────────────────────────────────────────────────┘        └──────────────────────────────┘
```

一句话概括数据流：**decode 记住"抢占位置 + 边界 token" → 暂存熬过更权 → 用新权重让 prefill 重算前缀 KV → KV 传回后回放边界 token 无缝续跑**。

## 5. 阶段 1：pause_generation(retract) —— 丢 KV + 暂存

入口 `Scheduler.pause_generation`（scheduler.py:4559）。`retract` 模式下的关键动作：

**（1）设置暂停标志并处理 overlap 尾批**（scheduler.py:4561, 4573-4576）
如果开了 overlap 调度且有 last_batch，先把上一拍延后处理的结果 flush 掉（`process_batch_result`），避免残留。

**（2）收集所有未完成请求**（scheduler.py:4578-4596）

```python
retract_reqs = [r for r in self.running_batch.reqs if not r.finished()]
# last_batch 若是 extend 且非 PD-prefill，也并入
# chunked_req 若未完成且非 PD-prefill，也并入
```

**（3）丢弃 KV（关键：offload_kv=False）**（scheduler.py:4606-4614）

```python
retract_all(
    reqs=retract_reqs, ...,
    offload_kv=False,   # ← 不做 device→host 拷贝，反正要重算
)
```

`retract_all` 内部对 decode 模式本会调 `offload_kv_cache`（schedule_batch.py:1916 附近），但这里显式传 `offload_kv=False` 跳过——注释说明该拷贝"会被立即丢弃"，是纯浪费。

**（4）逐请求打标记并暂存**（scheduler.py:4616-4622，仅 decode 分支）

```python
for req in retract_reqs:
    if self.disaggregation_mode == DisaggregationMode.DECODE:
        if req.output_ids:
            req.pd_rebootstrap_forced_output_id = req.output_ids.pop()  # 摘下边界 token
        req.pd_rebootstrap_in_progress = True
        req.time_stats.set_retract_time()
        self.disagg_decode_prealloc_queue.hold_rebootstrap(req)          # 暂存，不入队
    else:
        self._add_request_to_queue(req)   # 非 decode：走普通重入队
```

注意 `output_ids.pop()`：**把已经生成并发给客户端的最后一个 token 从 output_ids 里摘出来**，单独记在 `pd_rebootstrap_forced_output_id`。这个 token 后面会在 KV 传回时"回放"（见第 10 节），而不会让 prefill 重新采样它。

**（5）刷新观测指标**（scheduler.py:4642-4646）
暂停窗口内调度器事件循环会短路，不再走 `on_idle`，所以这里手动把 `gen_throughput` 清零、强制打一条 idle 日志、flush KV events，让 dashboard 立刻反映暂停态。

`hold_rebootstrap` 本身极简（decode.py:619-631）——只是 `held_rebootstrap_reqs.append(req)`，配套的长注释解释了「为什么不能立即入队」（见第 3 节末）。

## 6. 阶段 2：update_weights 与暂停窗口的不变量

这一阶段本身不属于 rebootstrap 代码，但它是整个设计的「为什么」。在 `pause_generation(retract)` 与 `continue_generation` 之间，RL 框架会调 `update_weights` 更新模型权重。此窗口内必须维持两个不变量：

| 不变量 | 由谁保证 | 违反后果 |
|---|---|---|
| 预分配队列为空 | `hold_rebootstrap` 只暂存不入队（decode.py:619-631） | 更权后 cache flush 断言失败 → crash |
| KV 显存已释放 | `retract_all` 真正 free（scheduler.py:4606） | 更权后无法 flush cache |

这也解释了为什么 rebootstrap 是 `retract` 模式独有的：`in_place` 模式不释放 KV、running_batch 原地保留，根本没有「重算」的必要，也不允许 flush cache（io_struct.py:1622-1623）。

暂存列表的一个已知限制（decode.py:353-355）：**held 中的请求无法被 `/abort_request` 触及**。因为它们既不在 running_batch 也不在预分配队列里，abort 遍历不到。注释说明"在 RL 场景中这实际上不会发生"（更权窗口很短，且是受控流程），若要支持需在调度器额外打补丁。

## 7. 阶段 3：continue_generation —— 入队 rebootstrap

入口 `Scheduler.continue_generation`（scheduler.py:4648）：

```python
def continue_generation(self, recv_req):
    if recv_req.torch_empty_cache:          # 默认 True（io_struct.py:1641）
        torch.cuda.empty_cache()            # 归还更权期间的临时显存碎片
    if (self.disaggregation_mode == DisaggregationMode.DECODE
            and self.disagg_decode_prealloc_queue is not None):
        self.disagg_decode_prealloc_queue.enqueue_held_rebootstrap()   # ← 关键
    self._engine_paused = False
```

`enqueue_held_rebootstrap`（decode.py:633-638）把所有暂存请求以 `is_rebootstrap=True` 真正入队：

```python
def enqueue_held_rebootstrap(self):
    held = self.held_rebootstrap_reqs
    self.held_rebootstrap_reqs = []
    for req in held:
        self.add(req, is_rebootstrap=True)
```

`add(req, is_rebootstrap=True)`（decode.py:524-545）→ `_create_receiver_and_enqueue(req, is_rebootstrap=True)`（decode.py:597-617）建立 `CommonKVReceiver` 并封装成 `DecodeRequest(is_rebootstrap=True)` 压入 `self.queue`。`is_rebootstrap` 标志一路透传到预分配阶段。

至此请求重新进入标准 PD-decode 的预分配 → 传输 → prebuilt 流程，只是被 `is_rebootstrap` 标志"染色"，在几个关键分叉点走特殊路径。

## 8. 阶段 4：预分配 + submit_prefill_recompute

预分配循环（decode.py:1023 起）为每个 `DecodeRequest` 分配 KV 槽位、把目的地址推给 prefill。rebootstrap 请求在此有两处差异：

**（1）关闭 decode 侧 radix cache**（decode.py:1027-1030）

```python
use_decode_radix_cache = (
    self.scheduler.server_args.disaggregation_decode_enable_radix_cache
    and not decode_req.is_rebootstrap        # ← rebootstrap 不复用 decode radix
)
```

原因：radix 缓存里可能有旧权重算出的前缀，rebootstrap 的目的恰恰是"用新权重全部重算"，复用旧前缀会引入 off-policy 污染。

**（2）推完目的地址后立即发起重算 HTTP**（decode.py:1309-1319）

```python
decode_req.kv_receiver.send_metadata(page_indices, ..., **metadata_kwargs)  # 推 dst_kv_indices 给 prefill
if decode_req.is_rebootstrap:
    self.kv_manager.submit_prefill_recompute(
        decode_req.kv_receiver,
        decode_req.req.build_rebootstrap_payload(),
    )
```

**顺序很关键**：先 `send_metadata` 让 prefill 知道"这个 bootstrap_room 的 KV 要写到哪些槽位"，再 POST `/generate` 触发 prefill 真正计算。这样 prefill 算完就能立即通过已建立的通道把 KV 单边写回 decode 预分配好的槽位。

`submit_prefill_recompute`（conn.py:406-456）做的事：
1. **leader 选举**：只有 `attn_tp_rank==0 and attn_cp_rank==0 and pp_rank==0` 的 rank 才真正发 HTTP（conn.py:437-438），其余 rank 直接 return（见第 15 节）；
2. 解析 prefill URL（conn.py:439，见第 13 节）；
3. 把 HTTP POST 提交到共享线程池执行器异步执行（conn.py:454-455），**对调度器非阻塞**。

## 9. 阶段 5：prefill 重算与 KV 传回

decode 的 leader rank 通过 `_run_prefill_recompute`（conn.py:473-500）在后台线程发出：

```python
response = self._get_prefill_recompute_session().post(
    prefill_url.rstrip("/") + "/generate",     # 硬编码原生 /generate
    json=payload,
    timeout=self.waiting_timeout,
)
```

prefill 实例收到这个带 `bootstrap_room` / `bootstrap_host` / `bootstrap_port` 的 `/generate` 请求后，**当作一个普通的 PD-prefill disagg 请求处理**：走正常的 bootstrap 匹配（`bootstrap_room` 找到 decode 之前推来的 `dst_kv_indices`）→ 前缀 KV 计算（此时用的是**已更新的权重**）→ 采样 1 个 handoff token（payload 强制 `max_new_tokens=1`）→ 通过 RDMA/ZMQ 把新算的前缀 KV 单边写回 decode 的目的槽位。

也就是说，**prefill 端不需要任何 rebootstrap 专属代码**——它看到的就是一个普通的首次 prefill 请求。所有"特殊性"都封装在 decode 侧（记边界 token、跳过 radix、回放）和 payload 构造里。

## 10. 阶段 6：传输 commit —— 边界 token 回放

当 `DecodeTransferQueue` 轮询到该请求的 KV 传输 `Success`，进入 commit 逻辑（decode.py:1899 起）。rebootstrap 的核心难点在这里：**prefill 用新权重重新采样了一个 handoff token，但这个位置的 token 我们其实早就发给客户端了**（就是 retract 时 pop 出来的边界 token）。

如果直接用 prefill 新采样的 token，会导致：同一个位置对客户端"变卦"——先发了 token A，恢复后又变成新采样的 token B，破坏输出一致性。解决办法是 **回放（replay）边界 token**（decode.py:1909-1918）：

```python
replayed_boundary = (
    decode_req.is_rebootstrap
    and decode_req.req.pd_rebootstrap_forced_output_id is not None
)
if replayed_boundary:
    committed_output_id = decode_req.req.pd_rebootstrap_forced_output_id   # 用记住的边界 token
    decode_req.req.pd_rebootstrap_forced_output_id = None
else:
    committed_output_id = output_id[0].item()                             # 用 prefill 新采样的 token
decode_req.req.output_ids.append(committed_output_id)
```

配套地，**logprob 也不重打**（decode.py:1946）：

```python
if decode_req.req.return_logprob and not replayed_boundary:
    decode_req.req.logprob.output_token_logprobs_val.append(...)
```

因为边界 token 保留它 retract 之前的原始 logprob——我们从不用新策略去"重新给已生成的 token 打分"（注释 decode.py:1904-1906）。

**两种子情形**：

| 情形 | forced_output_id | 行为 |
|---|---|---|
| retract 时已发出过 ≥1 个 token | 非 None | 回放边界 token，跳过 logprob |
| retract 时还没发出任何 token（prefill 后立即被抢占） | None | 走普通路径，正常提交首 token 与 logprob（decode.py:1906-1908） |

commit 还会用 prefill 报告的 `cached_tokens` 播种 `already_computed`（decode.py:1919-1926），避免 decode 侧 radix 复用与 prefill 前缀命中重复计数。

最后 `prepare_for_prebuilt`（decode_schedule_batch_mixin.py:75-76）把 `pd_rebootstrap_in_progress` 复位为 False，请求回归正常解码。

## 11. 关键字段与数据结构

| 字段 / 结构 | 位置 | 生命周期 | 作用 |
|---|---|---|---|
| `DecodeRequest.is_rebootstrap` | decode.py:275 | 请求级 | 标记这是一个重引导请求；控制关 radix（:1029）、发重算 HTTP（:1315）、回放（:1910） |
| `Req.pd_rebootstrap_in_progress` | schedule_batch.py:1158 附近 | retract→prebuilt | 影响长度计算（`_rebootstrap_prefill_len` / `_pre_alloc_fill_len`）；prebuilt 后清 False |
| `Req.pd_rebootstrap_forced_output_id` | schedule_batch.py:1158 | retract→commit | 记住 pop 出的边界 token，commit 时回放；decode-local，**从不传给 prefill** |
| `DecodePreallocQueue.held_rebootstrap_reqs` | decode.py:356 | pause→continue | pause 窗口暂存列表，保证预分配队列空 |

三个字段协同覆盖请求的三个时间点：`pause` 时打标记并 pop 边界 token（`in_progress=True`, `forced_output_id=X`, 进 held），`continue` 时入队（`is_rebootstrap=True`），`commit` 时回放（用 `forced_output_id` 并清空）、`prebuilt` 时复位（`in_progress=False`）。

> 注：代码里对 `pd_rebootstrap_in_progress` 用了 `getattr(req, "...", False)`（decode.py:642/648, decode_schedule_batch_mixin.py:75），这与仓库 `no-getattr-defensive` 规则相悖，属现存代码；`Req.__init__` 其实已固定初始化该字段，直接属性访问即可。此处仅作说明，不在本文范围内修改。

## 12. build_rebootstrap_payload 与为什么走原生 /generate

payload 由 `Req.build_rebootstrap_payload`（schedule_batch.py:1733-1779）构造，**写死走原生 `/generate` 接口，不支持 OpenAI 兼容接口**（`/v1/completions`、`/v1/chat/completions`）。

payload 关键字段：

```python
{
    "input_ids": [int(x) for x in origin_input_ids] + [int(x) for x in output_ids],  # 已 tokenize
    "sampling_params": {"max_new_tokens": 1, "temperature": ..., ...},   # 强制 max_new=1
    "return_logprob": False,
    "stream": False,
    "rid": self.rid,
    "bootstrap_host": ..., "bootstrap_port": ..., "bootstrap_room": ...,   # PD 握手三元组
    "priority": ..., "extra_key": ..., "routing_key": ...,
    "disagg_prefill_dp_rank": ...,
}
```

**为什么只能是原生 `/generate` 而非 OpenAI 接口**：

| payload 字段 | 原生 `/generate` | OpenAI `/v1/*` |
|---|---|---|
| `input_ids`（token id 序列） | ✅ 直接收 token | ❌ 只收 `prompt` 文本 / `messages`，会重新 tokenize |
| `bootstrap_host/port/room` | ✅ PD 握手必需 | ❌ schema 无此字段 |
| `disagg_prefill_dp_rank` / `routing_key` / `extra_key` | ✅ | ❌ |
| `sampling_params.max_new_tokens` | ✅ 原生字段 | ❌ OpenAI 用 `max_tokens` |

核心原因：rebootstrap 传的是 **`origin_input_ids + output_ids` 的原始 token id**（schedule_batch.py:1753-1754），目的是让 prefill **逐 token 精确重算出与原序列完全对齐的前缀 KV**。走 OpenAI 接口意味着要么传文本被重新 tokenize（可能对不齐 token 边界），要么无法携带 PD bootstrap 三元组来复用已建立的传输通道。因此这条路径只能是原生 `/generate`（POST 硬编码在 conn.py:479）。

**采样参数 allow-list**：payload 只透传一小部分采样参数并强制 `max_new_tokens=1`、丢掉 stop / grammar / min_new_tokens（schedule_batch.py:1740-1742）——因为这次调用的唯一目的是"重算前缀 KV + 出一个 handoff token"，不需要真正生成。而且这个 handoff token 在有边界 token 时还会被 decode 侧回放覆盖掉（第 10 节）。

## 13. URL 自发现：prefill_http_port 自注册

rebootstrap 不依赖 router 注入的 URL，而是让 **prefill 在 bootstrap 注册时自报 HTTP 端口**，decode 从 bootstrap 信息里自行拼出目标 URL。

**（1）prefill 自注册端口**（conn.py:708-712）

prefill 启动向 bootstrap server 注册时，把自己的 HTTP API 端口放进注册信息：

```python
"prefill_http_port": get_serving().port,   # 自报 HTTP 端口
```

对应 `PrefillServerInfo.prefill_http_port` 字段（conn.py:106），注释（conn.py:102-105）解释了设计意图：decode 本来就知道 prefill 的 host（即 `bootstrap_addr` 的 host），再拿到端口即可拼出 `/generate` URL，**无需 router 注入 `pd_rebootstrap_prefill_url`**。

**（2）decode 侧解析 URL**（conn.py:388-404）

```python
def _resolve_rebootstrap_prefill_url(self, kv_receiver):
    prefill_info = self.prefill_info_table.get(kv_receiver.bootstrap_addr)
    if prefill_info is None or prefill_info.prefill_http_port is None:
        return None
    host = NetworkAddress.parse(kv_receiver.bootstrap_addr).host
    return NetworkAddress(host, prefill_info.prefill_http_port).to_url()   # http://{host}:{port}
```

即 `host`（来自 bootstrap_addr）+ `prefill_http_port`（prefill 自注册）= `http://{host}:{prefill_http_port}`，末尾再拼 `/generate`。解析失败（信息缺失）时走失败路径 abort（conn.py:440-453）。

## 14. 长度计算差异：_rebootstrap_prefill_len / _pre_alloc_fill_len

因为 retract 时 `output_ids.pop()` 摘走了边界 token，rebootstrap 请求的长度计算和普通请求不同。两个静态方法处理这个差异：

**（1）`_rebootstrap_prefill_len`（decode.py:640-644）—— 用于容量校验**

```python
if pd_rebootstrap_in_progress:
    return len(origin_input_ids) + len(output_ids)   # prompt + 已生成（不含边界 token）
return len(origin_input_ids)                          # 普通：只有 prompt
```

用于 `_check_if_req_exceed_kv_capacity`（decode.py:668）判断请求是否超出 KV 容量。

**（2）`_pre_alloc_fill_len`（decode.py:646-660）—— 用于预分配槽位数**

```python
if pd_rebootstrap_in_progress:
    return len(origin_input_ids) + len(output_ids)          # 精确 = 原 seqlen - 1，无 -1
return len(origin_input_ids) + max(len(output_ids) - 1, 0)  # 普通 decode：末 token KV 未写，-1
```

注释（decode.py:649-658）说明差异根源：

- **普通 decode**：`output_ids` 里最后一个 token 是刚采样、KV 还没写的"pending" token，所以要 `-1`；
- **rebootstrap**：边界 token 已被 pop 出去（会靠 decode 侧回放补上），此时 `output_ids` 恰好是 prompt + (已发出 token - 边界 token) = 原 seqlen - 1，prefill 会为**所有这些 token** 重算 KV，没有 pending token，所以**不减 1**。

这与 offload 式抢占的 token 数一致（offload 也保存 seqlen-1 个 token 的 KV），边界 token 的 KV 在恢复后由 decode 侧解码时补算。

预分配阶段还会调 `_rebootstrap_prefill_len`（decode.py:1025）来估算 `origin_input_len`，供内存估算与 prefix match 使用。

## 15. 并发模型：leader 选举、执行器、per-thread session

rebootstrap 涉及多 rank 广播和异步 HTTP，有三个并发要点：

**（1）leader-only 发 HTTP**（conn.py:437-438）

decode 调度器会把每个 retract 请求广播给它 attention TP/CP 组的每个 rank 和每个 PP stage——它们都会走到 `submit_prefill_recompute`。但 `/generate` 是**服务级调用**：prefill 前端会把它 fan-out 到自己所有 worker，并把重算的 KV 传回**所有** decode rank。所以只能有一个 decode rank 发起 HTTP，否则 prefill 会重复重算同一请求 N 次。

选举规则（conn.py:437）：只有 `attn_tp_rank==0 and attn_cp_rank==0 and pp_rank==0` 的 rank 发 HTTP，其余 rank 直接 return。这个 leader 与请求 receiver 的 leader 是同一个（attn-tp/attn-cp 组 leader + 首个 PP stage）。其余 rank 照常 bootstrap 并接收自己那份 KV 分片；失败时 leader-only 的 abort 与 leader-only 的输出流式对齐，其他 rank 靠 per-request waiting-timeout 安全网兜底（conn.py:425-435）。

**（2）共享线程池执行器**（conn.py:357-377）

每个 decode KV manager 惰性创建一个共享线程池（`_ensure_prefill_recompute_executor`），所有 receiver 共用。线程数取自 `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE`（默认 16），线程名前缀 `pd-rebootstrap-prefill`。HTTP POST 提交到这里异步执行，对调度器主循环非阻塞。

**（3）per-thread requests.Session**（conn.py:379-386）

`requests.Session` 不能跨线程并发安全使用，所以用 thread-local 每线程一个 session（`_get_prefill_recompute_session`）。

## 16. 失败处理

任何环节失败都统一收敛到标准 `KVPoll.Failed` 路径（conn.py:458-471 `_fail_prefill_recompute`）：

```python
def _fail_prefill_recompute(self, kv_receiver, reason):
    kv_receiver.abort()                              # 转 Failed + 通知 prefill 释放孤儿 bootstrap 条目
    self.record_failure(kv_receiver.bootstrap_room, reason)  # 覆盖成描述性原因
```

触发失败的情形：

| 失败点 | 位置 | 处理 |
|---|---|---|
| URL 解析失败（bootstrap 信息缺失） | conn.py:440-453 | 记 error + `_fail_prefill_recompute` |
| HTTP 返回 ≥400 | conn.py:483-494 | 记 error（含 body 前 512 字节）+ fail |
| HTTP 抛异常 | conn.py:495-500 | `logger.exception` + fail |

`abort()` 把 receiver 转 Failed 并通知 prefill 释放孤儿 bootstrap 条目；`record_failure` 用描述性原因覆盖默认原因，让最终 `failure_exception`（及客户端看到的 abort 消息）能说明是"rebootstrap /generate 失败"而非误报 `AbortReq`（conn.py:461-468）。失败后走调度器现有的传输失败处理，把 aborted 请求流式返回客户端。

## 17. 约束与限制

| 约束 | 位置 | 说明 |
|---|---|---|
| 仅 PD-decode 模式 | scheduler.py:4617 | 只有 `disaggregation_mode == DECODE` 才 rebootstrap；PD-prefill 与单实例不涉及 |
| 仅 retract 模式触发 | io_struct.py:1625-1629 | `abort`/`in_place` 不走此路径 |
| **不支持多模态** | schedule_batch.py:1747-1750 | payload 只带 token `input_ids`，丢弃 image/audio/video，无法复现多模态前缀 KV；需先补多模态支持才能开启 |
| held 请求不可 abort | decode.py:353-355 | 暂存窗口内 `/abort_request` 触及不到；RL 短窗口下实践中不出现 |
| 只走原生 /generate | conn.py:479 | 不支持 OpenAI 兼容接口（见第 12 节） |
| PD-prefill 端尚未真抢占 | scheduler.py:4626-4631 | prefill 侧 chunked_req 仍保留，更权会留下 stale-weight 前缀 KV（off-policy），有 TODO |
| decode radix 被绕过 | decode.py:1029 | rebootstrap 不复用 decode 侧 radix，避免旧权重前缀污染 |

**核心适用场景**：PD 分离 + RL 在线更权。普通推理服务不会触发这条路径（不调 `pause_generation(retract)` + `update_weights`）。

## 18. 关键文件 / 行号索引

> 行号基于撰写时 main 分支快照，后续可能漂移，以符号名为准。

### 调度器（scheduler.py）

| 符号 / 行 | 说明 |
|---|---|
| `pause_generation`（:4559） | 三模式暂停入口 |
| `:4573-4576` | overlap 尾批 flush |
| `:4578-4596` | 收集 running/last/chunked 未完成请求 |
| `retract_all(offload_kv=False)`（:4606-4614） | 丢 KV，跳过 device→host 拷贝 |
| `:4616-4622` | decode 分支：pop 边界 token、打标记、`hold_rebootstrap` |
| `continue_generation`（:4648） | 恢复入口 |
| `enqueue_held_rebootstrap()`（:4667） | 入队暂存请求 |

### Decode 队列（disaggregation/decode.py）

| 符号 / 行 | 说明 |
|---|---|
| `DecodeRequest.is_rebootstrap`（:275） | 请求级重引导标志 |
| `held_rebootstrap_reqs`（:356） | pause 暂存列表 |
| `add(is_rebootstrap=)`（:524-545） | 入队 |
| `_create_receiver_and_enqueue`（:597-617） | 建 receiver + DecodeRequest |
| `hold_rebootstrap`（:619-631） | 暂存不入队 |
| `enqueue_held_rebootstrap`（:633-638） | 恢复时入队 |
| `_rebootstrap_prefill_len`（:640-644） | 容量校验长度 |
| `_pre_alloc_fill_len`（:646-660） | 预分配槽位长度（无 -1） |
| `:1025` | 预分配估算 origin_input_len |
| `:1027-1030` | rebootstrap 关 decode radix |
| `:1309-1319` | send_metadata 后 submit_prefill_recompute |
| `:1899-1918` | 传输 commit：边界 token 回放 |
| `:1946` | replayed_boundary 跳过重打 logprob |

### KV manager HTTP 驱动（disaggregation/common/conn.py）

| 符号 / 行 | 说明 |
|---|---|
| `prefill_http_port`（:106） | PrefillServerInfo 自注册端口字段 |
| `:102-105` | 设计意图注释 |
| `_ensure_prefill_recompute_executor`（:357-377） | 共享线程池 |
| `_get_prefill_recompute_session`（:379-386） | per-thread session |
| `_resolve_rebootstrap_prefill_url`（:388-404） | 拼 URL |
| `submit_prefill_recompute`（:406-456） | leader 选举 + 提交异步 HTTP |
| `_fail_prefill_recompute`（:458-471） | 统一失败路径 |
| `_run_prefill_recompute`（:473-500） | 后台线程发 POST /generate |
| `:708-712` | bootstrap 注册时自报 prefill_http_port |

### Req 字段与 payload（managers/schedule_batch.py）

| 符号 / 行 | 说明 |
|---|---|
| `pd_rebootstrap_forced_output_id`（:1158） | 边界 token（decode-local） |
| `pd_rebootstrap_in_progress`（:1158 附近） | 进行中标志 |
| `build_rebootstrap_payload`（:1733-1779） | 构造 /generate payload |
| `:1916-1918` | retract_all offload_kv 语义注释 |

### 其他

| 符号 / 行 | 说明 |
|---|---|
| `prepare_for_prebuilt`（decode_schedule_batch_mixin.py:75-76） | 复位 pd_rebootstrap_in_progress |
| `PauseGenerationReqInput.mode`（io_struct.py:1632） | abort/retract/in_place |
| `ContinueGenerationReqInput.torch_empty_cache`（io_struct.py:1641） | 恢复前是否 empty_cache |
| `/pause_generation` 路由（http_server.py:1665-1671） | HTTP 入口 |

---

## 附：一句话速记

> **PD rebootstrap = RL 更权时的"真抢占重算"**：decode 丢掉旧 KV、记住边界 token、暂存熬过更权，恢复后让原 prefill 用**新权重**经原生 `/generate` 重算前缀 KV 并传回，commit 时回放边界 token 保证输出一致。只用于 PD-decode + RL，不支持多模态与 OpenAI 接口。
