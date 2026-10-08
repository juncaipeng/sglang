# 集中式部署下的调度决策核心：`get_next_batch_to_run` 架构梳理

> 适用范围：SGLang **集中式（非 PD 分离）** 部署。即单实例、单 event loop、prefill 与 decode 在同一进程内交替执行的标准形态。PD 分离下的三队列/四阶段生命周期见 `dsv4_pd_disaggregation_request_lifecycle.md`，本文不覆盖。
>
> 主文件：`python/sglang/srt/managers/scheduler.py`
> 核心函数：`get_next_batch_to_run`（scheduler.py:2597）

---

## 0. 一句话定位与 TL;DR

`get_next_batch_to_run` 是调度器每一步（step）的**决策大脑**：它本身不做任何 GPU forward，只负责在「跑一个 prefill 批（EXTEND）」和「跑一个 decode 批（DECODE）」之间**二选一**，并返回待运行的 `ScheduleBatch`。

它遵循一条铁律——**Prefill-First（能预填就预填，否则才解码）**：

```
① 把上一步 prefill 的结果并入 running_batch
② 尝试组一个新的 prefill 批 → 组得出来就跑它
③ 组不出来，才推进 running_batch 解码（按需 retract 抢占）
```

一个跨多步的分块请求锚点 `self.chunked_req` 贯穿全程，享有**绕过所有限流闸门**的特权，以保证已占显存的分块请求一定能跑完，杜绝内存泄漏。

---

## 1. 层次一：它在事件循环里的位置

调度器有两个 event loop：`event_loop_normal`（scheduler.py:1533）与 `event_loop_overlap`（scheduler.py:1563）。两者结构一致，`get_next_batch_to_run` 都是每步的第一个决策点：

```
while True:                                    # event_loop_normal (:1533)
    recv_reqs = request_receiver.recv_requests()   # 收新请求
    process_input_requests(recv_reqs)              # 入 waiting_queue
 ┌─────────────────────────────────────────────┐
 │ batch = get_next_batch_to_run()   (:1546)     │ ← 本文主角：决定这步跑什么
 └─────────────────────────────────────────────┘
    if batch:
        result = run_batch(batch)          (:1551)    # 模型 forward
        process_batch_result(batch, result)(:1552)    # 写回 token / 清 chunked_req
    else:
        on_idle()                          (:1555)    # 空闲自检
    self.last_batch = batch                (:1558) ★ 记录本步 batch，供下一步阶段①使用
```

关键的**状态传递**是最后一行 `self.last_batch = batch`：本步产出的批被存下来，下一步在阶段①里被检查、过滤并合并进 running_batch。理解 `last_batch` 是理解本函数的前提。

在 overlap 模式下（`event_loop_overlap`:1563）差异在于：本步只发射 forward、把结果压入 `result_queue`，**上一步的结果推迟一拍处理**（`pop_and_process`:1569），从而让 CPU 侧的 `get_next_batch_to_run` 与 GPU 侧 forward 通过双 CUDA stream 重叠。但 `get_next_batch_to_run` 的决策逻辑本身与 normal 模式完全相同。

---

## 2. 层次二：核心决策 —— Prefill-First 决策树

整个函数就是一棵三分支决策树：

```
                    get_next_batch_to_run()  (:2597)
                            │
              ┌─────────────┴──────────────┐
              │ ① 合并 last_batch (extend)  │   处理上一步 prefill 的结果，
              │    → running_batch          │   把已完成的请求纳入解码批
              └─────────────┬──────────────┘   (:2608-2679)
                            │
              ┌─────────────┴──────────────┐
              │ ② new_batch =               │   尝试从 waiting_queue 组
              │    get_new_batch_prefill()  │   一个新的 prefill 批
              └─────────────┬──────────────┘   (:2681-2684)
                            │
                   new_batch is not None ?
                    ┌───────┴────────┐
                  YES               NO
                    │                │
              ret = new_batch   ┌────┴───────────────────────┐
              跑 EXTEND         │ ③ running_batch 非空          │
              (:2701)          │    且 非 prefill_only ?         │
                                │    → update_running_batch()   │
                                │      跑 DECODE                 │
                                └────┬───────────────────────┘  (:2704-2711)
                                     │
                              ret = running_batch or None
```

**为什么 prefill 优先？**

| 视角 | Prefill-First 的收益 | 代价 |
|------|---------------------|------|
| 吞吐 | prefill 把 waiting 请求"拉进"running_batch，**扩大 decode 并行度**，摊薄逐-token 解码的固定开销（kernel launch、权重读取带宽） | prefill 会打断连续 decode |
| 延迟 | 新请求尽快拿到首 token（TTFT 低） | 正在解码的老请求被推迟一步，ITL 抖动 |
| 显存 | prefill 结束即释放临时缓冲 | 大 prefill 峰值显存高 |

抖动问题正是 **chunked prefill**（切片）、**prefill delayer**（跨-rank 协商推迟）、**mixed chunk**（prefill 与 decode 同批）等机制要缓解的对象。

---

## 3. 层次三：三个阶段逐段拆解

### 阶段 ① —— 合并上一步的 prefill 结果（scheduler.py:2599-2679）

这一段的使命：把**上一步**跑完的 prefill 批（`last_batch`）里已经完成 prefill 的请求，正式并入 `running_batch`，从此进入逐-token 解码；同时把"还没填完的分块请求"隔离出去。

```python
self.process_pending_chunked_abort()      # :2599 先处理挂起的 chunked abort
...
chunked_req_to_exclude = set()            # :2609 本步不能并入 running 的请求集合

# —— 剔除本步仍在分块中的请求（活指针）——
if self.chunked_req is not None:                       # :2616
    chunked_req_to_exclude.add(self.chunked_req)       # :2619
    if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):
        self.stash_chunked_request(self.chunked_req)   # :2627 已产出的 KV 缓存进 radix

# —— 只有 last_batch 是 extend(prefill) 批时才合并 ——
if self.last_batch and self.last_batch.forward_mode.is_extend():   # :2642
    if self.last_batch.chunked_req is not None:
        chunked_req_to_exclude.add(self.last_batch.chunked_req)     # :2650 上步快照(context PP 陈旧引用)
    ...
    self.last_batch.filter_batch(chunked_req_to_exclude=list(...))  # :2657 剔除未完成/已完成
    if not self.last_batch.is_empty():
        if self.running_batch.is_empty():
            self.running_batch = self.last_batch                    # :2666
        else:
            self.running_batch.merge_batch(self.last_batch)         # :2669
```

三个要点：

1. **只处理 extend 批**（:2642）。上一步若本就是 decode 批，它已经是 `running_batch`，无需再合并。
2. **两个 chunked_req 都要排除**：`self.chunked_req`（本步活指针，:2619）与 `self.last_batch.chunked_req`（上步快照，:2650）。二者常指向同一对象，但因 `last_batch` 落后一拍（尤其 context PP / overlap 下）可能不一致，故各排除一次。分块请求不能进 decode，要留到本步阶段②继续 prefill。
3. **prefill-only 批的清理**（:2676）：纯 prefill（embedding 等无 decode 阶段）批的已完成请求在此单独 filter，保证 `/v1/loads` 上报的 `num_running_reqs` 准确，即使流量停了也能清干净。

### 阶段 ② —— 尝试组一个新的 prefill 批（scheduler.py:2681-2697, 2733-2985）

入口 `get_new_batch_prefill`（:2733）是对 `_get_new_batch_prefill_raw`（:2753）的封装，外层只多做一件事：包裹 `PrefillDelayerSinglePassExecutor`（:2740）——DP attention 下跨-rank 协商是否推迟 prefill（详见 `prefill_delayer_architecture.md`）。核心组批在 raw 函数里。

**第一部分：提前返回闸门**（决定"这步不做 prefill"）

| 闸门 | 位置 | 触发条件（满足即 `return None`，转 decode） | 对 chunked_req 放行？ |
|------|------|------|:---:|
| 队列空 / 批满 | :2769 | `(batch_is_full 或 waiting_queue 空)` | ✅ 放行 |
| MinFreeSlots 延迟器 | :2776 | 空闲 slot 不足需推迟（DFlash 默认开） | ✅ 放行 |
| 无可分配名额 | :2791 | `get_num_allocatable_reqs() <= 0` 且非优先级抢占 | ✅ 放行 |
| 测试限流 | :2802 | `TEST_RETRACT` 且 running_bs 超阈值 | ❌ |

**核心特权**：除测试闸门外，所有闸门都对 `self.chunked_req is not None` 网开一面（:2771/:2778/:2793）。原因见 :2786-2790 注释——分块请求必须继续推进，否则其 KV 已占用却永不完成，造成**内存泄漏**。非 PP 场景下 chunked_req 存在时名额必然 > 0（刚释放过），PP 场景下分块可能跨 microbatch，故不能对它做严格名额检查。

**第二部分：组批**

```python
self.policy.calc_priority(self.waiting_queue, self.running_batch)   # :2800 排序 lpm/fcfs/lof/...
chunked_prefill_size = self.chunked_prefill_size                    # :2809
if self.chunked_req is not None and self.enable_dynamic_chunking:   # :2810 动态块大小
    chunked_prefill_size = self.predict_next_chunk_size(...) or chunked_prefill_size

adder = PrefillAdder(page_size, tree_cache, allocator, running_batch,
                     new_token_ratio, max_prefill_tokens, chunked_prefill_size,
                     running_bs if self.is_mixed_chunk else 0, ...)  # :2817 预算控制器

if self.chunked_req is not None:                                    # :2835 先塞未完成的块
    self.chunked_req.init_next_round_input()
    self.chunked_req = adder.add_chunked_req(self.chunked_req)
# ... adder 逐个 add_one_req，受 max_prefill_tokens/chunked_prefill_size/名额约束

can_run_list = adder.can_run_list                                   # :2928
if len(can_run_list) == 0: return None                              # :2929 一个都塞不下→转 decode

if adder.new_chunked_req is not None:                               # :2938 本批新产生了分块
    assert self.chunked_req is None
    self.chunked_req = adder.new_chunked_req                        # :2941 记为下步的锚点

new_batch = ScheduleBatch.init_new(can_run_list, ..., chunked_req=self.chunked_req)  # :2949
new_batch.contains_last_prefill_chunk = (self.chunked_req is None or len(can_run_list)!=1)  # :2960
new_batch.prepare_for_extend()                                      # :2971
```

`PrefillAdder` 是预算裁决者：`chunked_prefill_size` 限单批 prefill token 上限，超长 prompt 被切片，`self.chunked_req` 携带切片进度跨步续跑；`is_mixed_chunk` 时把当前 decode 请求数带入，允许 prefill 与 decode 混批。

### 阶段 ③ —— 无 prefill 时推进解码（scheduler.py:2704-2711, 3041-3129）

仅当阶段②返回 `None`（组不出 prefill 批）且 running_batch 非空、非 prefill_only 时执行：

```python
if not self.running_batch.is_empty() and not self.running_batch.is_prefill_only:  # :2704
    self.running_batch = self.update_running_batch(self.running_batch)
    ret = self.running_batch if not self.running_batch.is_empty() else None
```

`update_running_batch`（:3041）做三件事：

```
① filter_batch()                             # :3045 剔除已完成请求；空则直接返回
② check_decode_mem()  →  是否够下一步显存？   # :3056
      ├─ 不够(kv_full_retract_flag) ─→ retract_decode(server_args)  # :3069 抢占
      │      ├─ 释放被踢请求的 KV，重算 new_token_ratio
      │      ├─ 记录 retract 指标 + 日志 "KV cache pool is full. Retract requests."
      │      ├─ reqs_to_abort → 发 AbortReq 给 tokenizer                # :3092
      │      └─ retracted_reqs → _add_request_to_queue(is_retracted=True) # :3117 踢回 waiting
      └─ 够 ─→ new_token_ratio_tracker.decay_step()                   # :3119 乐观降低预留
③ prepare_for_decode()                        # :3128 构造 decode tensor
```

**`new_token_ratio` 自适应预留系数**是这里的灵魂：它是"每步预计生成多少 token"的动态估计，用于 `check_decode_mem` 判断显存是否够用。retract 发生时拉高（保守，多预留），平稳运行时逐步 `decay_step` 衰减（激进，提利用率），在"少 retract"与"高显存利用率"之间自适应平衡。

**retract（抢占）不是失败**：被 retract 的请求不 abort，而是踢回 `waiting_queue`（`is_retracted=True`），显存充裕时重新被阶段②的 prefill adder 拉回来续跑。retract 排序策略见 `schedule_batch.py:retract_decode`——优先踢"输入长、已生成少"的请求，且至少保留一个。

---

## 4. 层次四：`chunked_req` 贯穿全程

`self.chunked_req`（scheduler.py:1016 初始化为 None）是理解本函数的另一条主线——它是**跨多步的分块预填充进度锚点**：

```
产生: 阶段② adder.new_chunked_req 出现 → self.chunked_req = it  (:2938-2941, assert 原为 None)
  │
续跑: 阶段② 每步 adder.add_chunked_req(self.chunked_req) 继续填下一块  (:2835-2837)
  │      每次组批把它塞进 new_batch.chunked_req  (:2957)
  │      沿途绕过所有限流闸门 (:2771/:2778/:2793)
  │
清空: 最后一块填完 → process_batch_result 里 self.chunked_req = None  (:4031)
```

它与 `self.last_batch.chunked_req` 的关系（前文阶段①已用到）：

| | `self.chunked_req` | `self.last_batch.chunked_req` |
|--|--|--|
| 层级 | Scheduler 级 | Batch 级 |
| 语义 | 当前时刻的**活指针**（真理源） | 上一步建 batch 时的**快照** |
| 来源 | adder 产生 / 每步续跑 | `init_new(chunked_req=self.chunked_req)` 冻结 |
| 何时不一致 | context PP / overlap 下 last_batch 落后一拍，可能持陈旧引用 |

阶段①对两者各排除一次（:2619 / :2650），确保未填完的分块请求不会被误并入 decode 批，也确保 context PP 下的陈旧引用被丢弃。

---

## 5. 层次五：返回前的横切分支（scheduler.py:2681-2724）

阶段②③选出候选 `ret` 后，还要过几道横切处理，主要服务于 DP attention 与特殊运行时：

```python
# —— dLLM(diffusion LLM) 专用组批路径 ——
if self.dllm_config is not None:                       # :2681
    new_batch = self.get_new_batch_dllm()
else:
    new_batch = self.get_new_batch_prefill()

# —— spec + dp-attn: 强制 prefill/decode 不混批 ——
need_mlp_sync = self.require_mlp_sync                   # :2686
if need_mlp_sync and not spec_algorithm.is_none() and not speculative_skip_dp_mlp_sync:
    new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)  # :2696
    need_mlp_sync = new_batch is None

# —— prefill-first 二选一（阶段②/③）——
if new_batch is not None:
    ret = new_batch                                    # :2701 跑 EXTEND
else:
    ret = update_running_batch(...) if running 非空/非prefill_only else None  # :2708

# —— DP attention MLP 同步：跨 rank 对齐空/非空批，避免死锁 ——
ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret, need_sync=need_mlp_sync)  # :2714

# —— ngram embedding / fpm 计时 ——
ret = self._maybe_prepare_ngram_embedding(ret)         # :2719
if ret:
    set_schedule_time_batch(ret)                       # :2722
return ret
```

| 分支 | 场景 | 作用 |
|------|------|------|
| `get_new_batch_dllm` | diffusion LLM | 走 dLLM 专用组批（非本文重点） |
| `maybe_prepare_mlp_sync_batch` | DP attention | 跨 DP rank all-gather 对齐"这步是否有批"，某 rank 空则塞 **idle batch** 陪跑，防止 all-reduce 死锁 |
| spec + dp-attn 前置同步（:2687） | 投机 + DP | 保证 prefill 与 decode 批不混，避免 spec 在 mixed 批下语义错乱 |
| `_maybe_prepare_ngram_embedding` | ngram 投机 | 补 ngram 检索所需 embedding |

对**单卡 / 无 DP attention / 无投机**的最简集中式部署，这些分支基本是 no-op，主干就是纯粹的"阶段①→②→③"。

---

## 6. 层次六：Prefill / Decode 的交替时序

集中式下 prefill 与 decode 在时间上**交替**、在批次上**偶尔混合**（`is_mixed_chunk`）：

```
step:   t0       t1      t2      t3       t4      t5
mode:  EXTEND   DECODE  DECODE  EXTEND   DECODE  DECODE
        │        │               │
        │        └─ waiting 空，running_batch 稳定解码（走阶段③）
        │                        └─ 新请求到达，阶段②组出 prefill 批，
        │                           下一步阶段① merge 进 running_batch
        └─ 启动：waiting 有请求 → 阶段② 组 prefill 批
```

- 只要阶段②返回批，这步就是 EXTEND，decode 被推迟一步（prefill-first 的直接后果）。
- overlap 调度下，本步 `get_next_batch_to_run` 的 CPU 决策与上步 forward 的 GPU 计算重叠，调度几乎不占 GPU 空泡（详见 `overlap_schedule_architecture.md`）。
- 分块 prefill 会让同一大请求在连续多个 EXTEND step 里逐块推进，其间穿插 decode（若 mixed chunk 开启）。

---

## 7. 性能影响与调优要点

| 现象 | 根因 | 调优旋钮 |
|------|------|---------|
| TTFT 高 | prefill 排队 / 被 delayer 推迟 / chunk 太小 | 调大 `--chunked-prefill-size`、关 prefill delayer、检查闸门 |
| ITL 抖动大 | 大 prefill 打断连续 decode | 开 chunked prefill、开 mixed chunk、调 `--schedule-conservativeness` |
| 频繁 `Retract requests.` 日志 | decode 显存不足触发抢占 | 降 `--mem-fraction-static`、观察 `new_token_ratio` 是否长期偏高 |
| decode 批小、吞吐低 | prefill 未及时把 waiting 拉入 running | 增 `--max-running-requests`、检查是否被 MinFreeSlots/无名额闸门卡住 |
| 疑似卡死 / 显存缓慢泄漏 | chunked_req 未能推进 | 确认闸门对 `chunked_req` 的放行分支未被破坏（:2771/:2778/:2793） |
| DP 部署间歇性 hang | 某 rank 空批未对齐 | 确认 `maybe_prepare_mlp_sync_batch` 正常插 idle batch |

调度策略（`--schedule-policy`）影响阶段② `calc_priority`（:2800）的排序：`lpm`（最长前缀匹配，cache 友好，默认）、`fcfs`、`lof`、`dfs-weight`、`routing-key`。

---

## 8. 关键文件行号索引

| 符号 / 逻辑 | 位置 |
|------|------|
| `get_next_batch_to_run`（主函数） | scheduler.py:2597 |
| 阶段① 合并 last_batch | scheduler.py:2608-2679 |
| `chunked_req` 排除（活指针 / 快照） | scheduler.py:2619 / 2650 |
| prefill-only 批清理 | scheduler.py:2676 |
| 阶段② 入口 `get_new_batch_prefill` | scheduler.py:2733 |
| `_get_new_batch_prefill_raw`（组批） | scheduler.py:2753 |
| 提前返回闸门（队列空/名额/delayer） | scheduler.py:2769 / 2776 / 2791 |
| chunked_req 绕闸特权 | scheduler.py:2771 / 2778 / 2793 |
| 续跑分块 `add_chunked_req` | scheduler.py:2835-2837 |
| 记录新分块 `new_chunked_req` | scheduler.py:2938-2941 |
| `ScheduleBatch.init_new` | scheduler.py:2949 |
| prefill-first 二选一分支 | scheduler.py:2699-2711 |
| DP MLP 同步 `maybe_prepare_mlp_sync_batch` | scheduler.py:2696 / 2714 |
| 阶段③ `update_running_batch` | scheduler.py:3041 |
| decode 显存检查 `check_decode_mem` | scheduler.py:3056 |
| 抢占 `retract_decode` | scheduler.py:3069 |
| `new_token_ratio` 衰减 | scheduler.py:3119 |
| `self.chunked_req` 初始化 | scheduler.py:1016 |
| `self.chunked_req` 清空 | scheduler.py:4031 |
| event loop（normal / overlap） | scheduler.py:1533 / 1563 |
| `self.last_batch = batch` | scheduler.py:1558 / 1625 |
| 相关：overlap 调度 | `docs/pjc_1/overlap_schedule_architecture.md` |
| 相关：prefill delayer | `docs/pjc_1/prefill_delayer_architecture.md` |
| 相关：PD 分离生命周期 | `docs/pjc_1/dsv4_pd_disaggregation_request_lifecycle.md` |

---

## 9. 一句话总结

`get_next_batch_to_run` 每步做一次 **"prefill 优先、decode 兜底"** 的二选一：先把上一步 prefill 的结果并入 running_batch（阶段①），再尝试从 waiting_queue 组一个新 prefill 批（阶段②），组不出来才推进 running_batch 解码并按需 retract 抢占（阶段③）。贯穿全程的 `self.chunked_req` 是分块进度锚点，享有绕过所有限流闸门的特权，以保证已占显存的分块请求一定跑完、杜绝内存泄漏。DP attention / 投机 / dLLM 等横切分支在返回前对齐跨-rank 批状态，但不改变这条主干决策链。
