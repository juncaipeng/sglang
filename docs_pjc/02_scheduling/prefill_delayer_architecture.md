# SGLang Prefill Delayer（预填充延迟器）方案系统梳理

> 核心文件：`python/sglang/srt/managers/prefill_delayer.py`、`python/sglang/srt/managers/schedule_policy.py`、`python/sglang/srt/managers/scheduler.py`
> 相关文件：`python/sglang/srt/server_args.py`、`python/sglang/srt/environ.py`、`python/sglang/srt/observability/metrics_collector.py`、`python/sglang/srt/managers/scheduler_components/dp_attn.py`
> 测试：`test/registered/scheduler/test_prefill_delayer.py`
> 目标读者：需要理解、调优或扩展 SGLang「DP attention 场景下 prefill 调度」的工程师。
>
> 本文由浅入深，覆盖：为什么需要 prefill delayer、它解决的两类空泡、跨 DP 协商机制、三分支决策逻辑（all/none/mixed）、两类延迟触发器（slot / queue）、token 水位安全阀、单趟执行器、all-gather 同步（gloo vs NCCL）、可观测性指标、PD 分离支持、与 min-free-slots 延迟器的区别、历史演进与调优排错。

---

## 目录

1. [总览：一句话定位](#1-总览一句话定位)
2. [背景：DP attention 的 lockstep 约束与两类空泡](#2-背景dp-attention-的-lockstep-约束与两类空泡)
3. [核心思想](#3-核心思想)
4. [整体架构与数据流](#4-整体架构与数据流)
5. [开关与参数](#5-开关与参数)
6. [触发位置：在调度主循环中的挂载点](#6-触发位置在调度主循环中的挂载点)
7. [单趟执行器 PrefillDelayerSinglePassExecutor](#7-单趟执行器-prefilldelayersinglepassexecutor)
8. [核心决策：\_negotiate\_should\_allow\_prefill\_pure](#8-核心决策_negotiate_should_allow_prefill_pure)
9. [跨 DP 信息同步：\_gather\_info](#9-跨-dp-信息同步_gather_info)
10. [状态机 \_State 与超时](#10-状态机-_state-与超时)
11. [可观测性：三个指标与直方图 bug 修复](#11-可观测性三个指标与直方图-bug-修复)
12. [PD 分离模式支持](#12-pd-分离模式支持)
13. [与 MinFreeSlotsDelayer 的区别](#13-与-minfreeslotsdelayer-的区别)
14. [历史演进](#14-历史演进)
15. [调优与排错](#15-调优与排错)
16. [关键文件 / 行号索引](#16-关键文件行号索引)

---

## 1. 总览：一句话定位

**Prefill Delayer = 在 DP attention（数据并行注意力）场景下，主动"推迟"某些 DP rank 的 prefill，让多个 rank 的 prefill 尽量对齐到同一步一起做，从而消除 DP rank 之间因 prefill/decode 步调不一致而产生的 GPU 空泡（idle bubble），并把零散的小 prefill 攒成大 prefill 以提升吞吐。**

它是一个**纯调度层的软控制器**：不改动 KV cache、不改动 forward，只在"这一趟要不要把等待队列里的请求塞进 prefill 批次"这个决策点上，返回一个 `allow / deny`。deny 时该 rank 这一趟不做 prefill（转而做 decode 或 idle），等待后续时机。

默认关闭，通过 `--enable-prefill-delayer` 开启。典型收益：离线高并发长输入场景吞吐 **+20% ~ +27%**（见测试 `test_prefill_delayer.py`）。

---

## 2. 背景：DP attention 的 lockstep 约束与两类空泡

### 2.1 DP attention 为什么要 lockstep

在 DP attention 模式下（`--enable-dp-attention`），每个 DP rank 各自维护一份注意力 KV，但 **MLP / MoE 层需要跨 DP rank 做 all-gather 同步**（把各 rank 的 token 拼起来一起过 MLP，再 scatter 回去）。这带来一个硬约束：

> **所有 DP rank 必须在同一步运行"同一种" forward mode**——要么大家都 extend（prefill），要么大家都 decode。否则 all-gather 会因为形状/语义不一致而死锁或产生错误。

这个同步逻辑在 `scheduler_components/dp_attn.py:prepare_mlp_sync_batch_raw` 里：它 all-gather 各 rank 的 `is_extend_in_batch` / `global_forward_mode`，并在需要时给"本步没有真实批次"的 rank 发一个 **idle batch**（`get_idle_batch`，dp_attn.py:303）来陪跑同步。

### 2.2 空泡从哪来

设想 8 个 DP rank，某一步只有 rank 0 的等待队列里来了一条 prefill 请求，其余 7 个 rank 都在 decode：

```
不加 delayer（rank 0 立即 prefill）：

step k:
  rank 0: [   PREFILL 30k tokens   ]  ← 真实工作
  rank 1: [ idle batch (陪跑同步) ]  ← 空泡！本可继续 decode
  rank 2: [ idle batch           ]  ← 空泡
  ...
  rank 7: [ idle batch           ]  ← 空泡

→ 7/8 的 GPU 算力在这一步被浪费在 idle batch 上。
  而且 rank 0 的长 prefill 会阻塞所有 rank 的 decode 进度（TTFT/TBT 抖动）。
```

这就是 **空泡类型 A：DP 步调不一致（mixed）**。根因是 prefill 请求"零散地"到达不同 rank，而 lockstep 逼着没有 prefill 的 rank 陪跑。

```
加 delayer（rank 0 被要求"再等等"）：

step k   : rank 0 想 prefill → delayer 说 deny → rank 0 也 decode
step k+1 : rank 0 仍 deny → 继续 decode
...
step k+n : rank 3、rank 5 的队列也来了 prefill 请求
           → 现在多个 rank 都 prefillable → delayer 说 allow
           → rank 0/3/5 一起 prefill，其余 rank 陪跑的比例下降
```

### 2.3 第二类空泡：零散小 prefill

即使所有 rank 都 prefillable，如果每个 rank 只有 1~2 条小请求，频繁地做"小 prefill + merge 进 decode"会导致：
- decode 批次一直到不了最大 batch size（`max_running_requests`），GPU 利用率低；
- 每次 prefill 都要付固定开销（kernel launch、cuda graph replay 切换等）。

这是 **空泡类型 B：prefill 碎片化**。解法是"攒批"——延迟 prefill，直到等待队列积累到一定量，一次性做一个大 prefill。对应后文的 **slot_condition** 与 **queue_condition** 两个触发器。

### 2.4 两类空泡与两类触发器的对应

| 空泡类型 | 根因 | 对应决策分支 | 触发器 |
|---------|------|------------|--------|
| A：DP 步调不一致 | prefill 请求零散落在不同 rank，lockstep 逼迫陪跑 | `mixed` 分支 | 部分 rank prefillable → delay |
| B：prefill 碎片化 | 单 rank 小 prefill 频繁，decode 到不了满 batch | `all` 分支 | `slot_condition` / `queue_condition` |

---

## 3. 核心思想

Prefill delayer 的本质是一个**跨 DP rank 的分布式协商器**。每个 rank 在"要不要 prefill"的决策点上：

1. **贡献本地信息**：我这一趟是否 `prefillable`（等待队列有请求且预算够）、我的 token 使用率、我的 running batch 大小、我的 max_prefill_bs、我的等待队列长度。
2. **all-gather 全局信息**：通过 `torch.distributed.all_gather_into_tensor` 把所有 rank 的上述信息收集到一起。
3. **统一决策**：基于全局视图判断——所有 rank 都想 prefill？没人想 prefill？还是只有部分 rank 想（mixed）？据此返回统一的 `allow / deny`。

关键设计点：

- **决策必须全局一致**：因为 lockstep 要求所有 rank 同步做同种 forward，delayer 的 `allow/deny` 也必须让所有 rank 得到"逻辑上一致"的结论。这是通过 all-gather + 各 rank 用相同的纯函数计算实现的——**同样的全局输入 → 同样的输出**。
- **纯函数 + 显式状态**：核心决策 `_negotiate_should_allow_prefill_pure` 是（几乎）纯函数，`prev_state` 显式传入、`next_state` 显式返回，副作用（更新 `self._curr_state`）被隔离在薄包装 `_negotiate_should_allow_prefill` 里。这让逻辑可单测（见 `test_prefill_delayer.py` 的 4-rank gloo 分布式单测）。
- **有界延迟**：delay 不能无限持续，否则请求饿死。三重保险：
  - `mixed` 分支有 `max_delay_passes` forward-pass 上限（超时强制放行，reason=`wait_timeout`）；
  - `all` 分支的 queue trigger 有 `max_delay_ms` 墙钟上限（超时强制放行，reason=`wait_success`）；
  - **token 水位安全阀**：只要有任一 rank 的 KV 使用率低于 `token_usage_low_watermark`，说明 GPU 空着，立即放行（reason=`token_watermark`）。

---

## 4. 整体架构与数据流

```
                    Scheduler.get_new_batch_prefill()   (scheduler.py:2714)
                                 │
              ┌──────────────────┴───────────────────┐
              │ if self.prefill_delayer:              │
              │   max_pool_usage =                    │
              │     pool_stats_observer               │
              │       .get_pool_stats()               │
              │       .get_max_pool_usage()           │  ← 取所有池(full/swa/mamba)最大使用率
              │   single_pass =                       │
              │     PrefillDelayerSinglePassExecutor( │  ← 每趟 new_batch 建一个一次性执行器
              │       delayer, token_usage=usage)     │
              └──────────────────┬───────────────────┘
                                 │ 传入 PrefillAdder
                                 ▼
             PrefillAdder.add_one_req(req)   (schedule_policy.py:882)
                                 │
              ┌──────────────────┴───────────────────┐
              │ single_pass.negotiate_should_allow_   │
              │   prefill(local_prefillable=True,     │
              │     running_batch=..., max_prefill_bs │
              │     =..., max_running_requests=...,   │
              │     waiting_queue_len=...)            │
              └──────────────────┬───────────────────┘
                                 │  首次调用才真正 negotiate（幂等缓存 _result）
                                 ▼
        PrefillDelayer._negotiate_should_allow_prefill()  (prefill_delayer.py:114)
                                 │
                                 ▼
        _negotiate_should_allow_prefill_pure()  (prefill_delayer.py:136)
              │
              ├── 1. 算 local_token_watermark_force_allow
              ├── 2. _gather_info() ── all_gather ──► 全 rank 的 5 元组
              │        [prefillable, watermark_force, running_batch,
              │         max_prefill_bs, waiting_queue_len]
              ├── 3. 归约出 prefillable_status ∈ {all, none, mixed}
              └── 4. 三分支决策 → _NegotiateOutput(output_allow, reason, ...)
                                 │
                                 ▼
       返回 True → PrefillAdder 继续加入请求；
       返回 False → PrefillAdder.add_one_req 直接 return AddReqResult.OTHER
                                 │
                                 ▼
        single_pass.finalize(actual_prefill = ret is not None)  (scheduler.py:2730)
              └── _record_single_pass_result() → metrics_collector.observe_prefill_delayer_outcome()
```

**数据流要点**：
- `token_usage` 在每趟 `get_new_batch_prefill` 开头**取一次快照**（`get_max_pool_usage()`，取 full/swa/mamba 各池的最大值），整趟复用。
- delayer 的 negotiate 涉及 **all-gather 集合通信**，是有 barrier 语义的——**所有 DP rank 必须都调用**，否则死锁。因此 `PrefillDelayerSinglePassExecutor` 用"每趟至多协商一次 + finalize 时若没协商过则补一次 `local_prefillable=False`"来保证所有 rank 的调用次数严格对齐（见 §7）。

---

## 5. 开关与参数

### 5.1 ServerArgs（server_args.py:1250-1285）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--enable-prefill-delayer` | bool | `False` | 总开关。为 DP attention 减少 idle time。 |
| `--prefill-delayer-max-delay-passes` | int | `30` | `mixed` 分支最多延迟多少个 forward pass，超时强制放行（防饿死）。 |
| `--prefill-delayer-token-usage-low-watermark` | float? | `None` | KV 使用率低于此值时视为 GPU 空闲，强制放行安全阀。`None`=关闭安全阀。 |
| `--prefill-delayer-forward-passes-buckets` | list? | `None` | wait_forward_passes 直方图自定义桶。默认 `[5,20,50,100,200]`，自动补 `0` 和 `max_delay-1`。 |
| `--prefill-delayer-wait-seconds-buckets` | list? | `None` | wait_seconds 直方图自定义桶。默认 `[1,2,5,10,20,50,100,200,500]`，自动补 `0`。 |
| `--prefill-delayer-queue-min-ratio` | float? | `None` | **opt-in** 队列触发器。延迟 prefill 直到等待队列达到 `min(running_req*ratio, max_prefill_bs)`。典型 `0.1~0.5`。`None`=只用 slot 触发器（原始行为）。 |
| `--prefill-delayer-max-delay-ms` | float? | `None` | 队列触发器的墙钟上限（ms），单次延迟超此值强制放行以约束最坏 TTFT。仅在设了 queue-min-ratio 时生效，`None`=回退 5000ms。 |

### 5.2 已弃用的环境变量（向后兼容，environ.py + server_args.py:2914）

`_handle_prefill_delayer_env_compat()` 把老的环境变量映射到新 CLI 参数（并打 deprecation warning）：

| 弃用环境变量 | 等价 CLI 参数 |
|-------------|-------------|
| `SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE=1` | `--enable-prefill-delayer` |
| `SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES` | `--prefill-delayer-max-delay-passes` |
| `SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK` | `--prefill-delayer-token-usage-low-watermark` |

### 5.3 调试/内部环境变量

| 环境变量 | 作用 |
|---------|------|
| `SGLANG_PREFILL_DELAYER_DEBUG_LOG=1` | 打开 `_DEBUG_LOG`（prefill_delayer.py:15），逐趟打印 timeout / watermark 放行日志。 |
| `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1` | 让 delayer 的 all-gather 走 NCCL device group 而非 gloo CPU group（见 §9）。 |

### 5.4 强约束（构造期 assert，prefill_delayer.py:110）

```python
assert not server_args.disable_overlap_schedule, \
    "To use PrefillDelayer, disable_overlap_schedule must be False."
```

- **必须开启 overlap schedule**（即 `disable_overlap_schedule=False`，默认）。
- decode 引擎上 `--enable-prefill-delayer` 会被**忽略并打日志**（scheduler.py:1050）——decode 引擎没有 prefill 调度路径，delayer 是 no-op。
- 注意：早期版本曾强制要求 `enable_dp_attention=True`，此约束已在 PR #17456 移除（`enable_dp_attention` 现在只影响 `dp_size_dim` 与 `max_running_requests` 的归一，见 prefill_delayer.py:76-77、206-209）；但 delayer 的价值几乎完全来自 DP attention 场景。

---

## 6. 触发位置：在调度主循环中的挂载点

### 6.1 顶层入口 get_new_batch_prefill（scheduler.py:2714）

```python
def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
    prefill_delayer_single_pass = None
    if self.prefill_delayer:
        # 取所有池(full/swa/mamba)的最大使用率，作为本趟 token_usage 快照
        max_pool_usage = (
            self.pool_stats_observer.get_pool_stats().get_max_pool_usage()
        )
        prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
            self.prefill_delayer, token_usage=max_pool_usage
        )

    ret = self._get_new_batch_prefill_raw(
        prefill_delayer_single_pass=prefill_delayer_single_pass
    )

    if self.prefill_delayer:
        # 无论 raw 是否真的组出了 batch，都要 finalize（保证 all-gather 调用对齐 + 上报指标）
        prefill_delayer_single_pass.finalize(actual_prefill=ret is not None)

    return ret
```

它在 `get_next_batch_to_run`（scheduler.py:2665）里被调用——每一步调度循环都会走一遍 prefill 决策。

### 6.2 两个真正的检查点（schedule_policy.py 的 PrefillAdder）

delayer 的协商发生在 `PrefillAdder` 尝试把一条 waiting 请求加入 prefill 批次时，共两处（对应两条不同的请求加入路径）：

**路径一：普通请求 add_one_req（schedule_policy.py:885）——携带完整调度上下文**

```python
def add_one_req(self, req, has_chunked_req, truncation_align_size):
    if (self.prefill_delayer_single_pass is not None) and (
        not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
            local_prefillable=True,
            running_batch=self.running_batch.batch_size(),
            max_prefill_bs=self.max_prefill_bs,
            max_running_requests=self.max_running_requests,
            waiting_queue_len=self.waiting_queue_len,
        )
    ):
        return AddReqResult.OTHER   # 本趟拒绝加入 → 停止组 prefill 批
    ...
```

**路径二：ignore_eos 请求 add_one_req_ignore_eos（schedule_policy.py:838）——只传 prefillable**

```python
def add_one_req_ignore_eos(self, req):
    ...
    if (self.prefill_delayer_single_pass is not None) and (
        not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
            local_prefillable=True   # 不传 running_batch 等 → 走历史 kwargs.get(...,0) 回退
        )
    ):
        return AddReqResult.OTHER
```

> 差异说明：路径二不传 `running_batch/max_prefill_bs/...`，delayer 侧默认值全 0，这会让 `all` 分支的 slot/queue 触发器**天然失效**（`running_batch_max=0` 时 queue trigger 短路，slot_condition 里 `max_running_requests-0<max_prefill_bs` 一般不成立），因此 ignore_eos 路径实际只受 `mixed` 分支约束。这是测试 `NegotiateCall` 里 `running_batch=None` 分支所覆盖的"历史回退行为"。

### 6.3 决策的语义：返回 False = 本 rank 本趟不 prefill

`add_one_req` 返回 `AddReqResult.OTHER` 会让 `PrefillAdder` 停止继续从 waiting queue 取请求。若 `can_run_list` 最终为空，`get_new_batch_prefill` 返回 `None`，`get_next_batch_to_run` 就转去跑 decode（scheduler.py:2683-2692）或发 idle batch。**这就是"延迟 prefill"的落地方式——不是阻塞等待，而是这一趟直接跳过 prefill、去做 decode。**

---

## 7. 单趟执行器 PrefillDelayerSinglePassExecutor

（prefill_delayer.py:331）这个类解决一个关键的**分布式一致性问题**：

> delayer 的 negotiate 内部有 `all_gather`（barrier 语义）。而 `PrefillAdder` 在一趟里可能对**多条**请求反复调用 `add_one_req` → 若每条都触发一次 all-gather，各 rank 的等待队列长度不同（有的 rank 队列空、根本不进 for 循环），all-gather 调用次数就会**在 rank 间不对齐 → 死锁**。

解法：**每趟 new_batch 至多协商一次，结果缓存并复用**。

```python
class PrefillDelayerSinglePassExecutor:
    def __init__(self, prefill_delayer, token_usage):
        self._prefill_delayer = prefill_delayer
        self._token_usage = token_usage
        self._result: Optional[_NegotiateOutput] = None   # 幂等缓存

    @property
    def _called(self) -> bool:
        return self._result is not None

    def negotiate_should_allow_prefill(self, local_prefillable, ...):
        if not self._called:                       # 只有第一次真正 negotiate
            self._result = self._prefill_delayer._negotiate_should_allow_prefill(...)
        return self._result.output_allow            # 后续调用直接返回缓存

    def finalize(self, *, actual_prefill: bool):
        if not self._called:
            # 本 rank 这一趟从没调用过 negotiate（例如等待队列为空）
            # → 补一次 local_prefillable=False，保证 all-gather 与其他 rank 对齐
            self.negotiate_should_allow_prefill(local_prefillable=False)
        _record_single_pass_result(actual_execution=actual_prefill,
                                   output=self._result, ...)
```

**两个保证**：
1. **调用次数对齐**：每趟每 rank 恰好 negotiate 一次——要么在 `add_one_req` 里（队列非空），要么在 `finalize` 里补一次（队列空，以 `prefillable=False` 参与协商）。这保证所有 rank 的 all-gather 步调一致。
2. **指标只上报一次**：`finalize` 里调用 `_record_single_pass_result`，把本趟真实结果（是否真的产出 prefill 批 `actual_prefill`）连同 negotiate 输出一起喂给 metrics（见 §11）。

> `actual_prefill = (ret is not None)`：注意"delayer 放行"不等于"真的做了 prefill"——放行后可能因为别的预算约束（token 不够、slot 不够）仍组不出批次。指标的 `actual_execution` label 区分了这两者。

---

## 8. 核心决策：\_negotiate\_should\_allow\_prefill\_pure

（prefill_delayer.py:136）这是整个模块的心脏。输入全局信息，输出统一决策。先看归约出的三态：

```python
if   global_prefillable.min() > 0:  prefillable_status = "all"    # 所有 rank 都想 prefill
elif global_prefillable.max() == 0: prefillable_status = "none"   # 没有 rank 想 prefill
else:                               prefillable_status = "mixed"  # 部分 rank 想
```

### 8.1 三分支决策总览

```
                    ┌─────────────────────────────────────────────┐
                    │   prefillable_status                          │
                    └───────┬─────────────┬──────────────┬─────────┘
                          "all"        "none"         "mixed"
                            │             │               │
       ┌────────────────────┘             │               └──────────────────────┐
       ▼                                  ▼                                       ▼
  ┌─────────────────────┐      ┌──────────────────┐          ┌──────────────────────────────┐
  │ 水位安全阀?          │      │ allow=True        │          │ 水位安全阀?                    │
  │  yes→allow          │      │ reason=""         │          │  yes→allow reason=token_water..│
  │      token_watermark│      │ (谁 prefill 都无所谓,│         │                                │
  ├─────────────────────┤      │  为简单起见放行)   │          │ delayed_count < max-1 ?         │
  │ slot_condition       │      └──────────────────┘          │  yes→deny reason=delay          │
  │  OR queue_condition ? │                                    │      (bump delayed_count)       │
  │  yes→skip_first? ─┐  │                                    │  no →allow reason=wait_timeout  │
  │        no→deny    │  │                                    └──────────────────────────────┘
  │           delay   │  │
  │        yes→放过一次│  │  ← skip_first_delayer
  │  no→allow         ▼  │
  │      wait_success/no_wait
  └──────────────────────┘
```

### 8.2 分支 `all`：所有 rank 都想 prefill（prefill_delayer.py:194）

这是"攒批"逻辑所在。此时不存在 DP 步调不一致（大家都要 prefill），但可能存在**碎片化**——即使都 prefill，每个 rank 只有零星几条，做了也浪费。判定两个触发条件：

**① slot_condition（槽位触发器，原始逻辑）**（prefill_delayer.py:235）

```python
slot_condition = (
    max_running_requests - global_running_batch_max < global_max_prefill_bs_max
)
```

含义：如果"decode 还能容纳的空槽数"小于"一次 prefill 可能加入的请求数"，那么这次 prefill 加进去后，第一次 merge_batch 就会让 decode 批达不到最大 batch size（decode 效率受损）。所以宁可**再等等**，等 decode 腾出更多空槽，再一次性 prefill 填满。
- 非 DP attention 时，`max_running_requests` 先按 dp_size 向上取整归一（prefill_delayer.py:206-209）。

**② queue_condition（队列触发器，opt-in，PR #23189 新增）**（prefill_delayer.py:220-233）

```python
queue_condition = False
if self._queue_trigger_enabled and global_running_batch_max > 0:
    queue_min_effective = min(
        int(global_running_batch_max * self._queue_min_ratio),
        global_max_prefill_bs_max,
    )
    queue_condition = (
        queue_min_effective > 0
        and global_waiting_queue_max < queue_min_effective
    )
    # 墙钟超时保护：单次 queue-trigger 延迟超过 max_delay_ms 就强制放行
    if queue_condition and prev_state is not None:
        elapsed_ms = (time.perf_counter() - prev_state.start_time) * 1000.0
        if elapsed_ms >= self._max_delay_ms:
            queue_condition = False
```

含义：延迟 prefill，直到等待队列积累到 `queue_min = min(running*ratio, max_prefill_bs)` 条请求，才一次性做一个大 prefill。**专门针对"decode 请求一条条零散结束、把 prefill 切碎成许多小批"的负载**。用 `max_delay_ms` 墙钟上限兜底最坏 TTFT。

**判定与 skip_first**（prefill_delayer.py:240）

```python
if slot_condition or queue_condition:      # 任一触发就想延迟
    if self.skip_first_delayer:
        self.skip_first_delayer = False    # 第一次遇到延迟条件时"放过一次"
        pass                               # → 落到下面的 allow
    else:
        next_state = (prev_state or _State()).bump_delayed_count()
        return _NegotiateOutput(allow=False, reason="delay", next_state=next_state)
# 未触发 或 skip_first 放过 → 放行
exist_previous_wait = prev_state is not None
return _NegotiateOutput(allow=True,
    reason="wait_success" if exist_previous_wait else "no_wait", next_state=None)
```

> **skip_first_delayer**（PR #19836）：进程级只放过第一次会触发延迟的机会。原因见代码注释——"当 `max_decode_bs - running_bs < max_prefill_bs` 条件满足时，第一次 merge_batch 会导致 decode 达不到最大 batch"。放过第一次是为了让首个 decode 批尽快建立起来，避免冷启动被 delayer 卡住。它是 `self` 上的可变字段（**唯一破坏纯函数性的地方**），一旦置 False 不再恢复。

### 8.3 分支 `none`：没有 rank 想 prefill（prefill_delayer.py:263）

```python
return _NegotiateOutput(allow=True, reason="", next_state=None)
```

没人要 prefill，allow/deny 都无所谓，为简单起见直接 allow。此分支 reason 为空字符串 `""`。

### 8.4 分支 `mixed`：部分 rank 想 prefill（prefill_delayer.py:272）——空泡类型 A 的核心

这是 delayer 最初的设计目标：消除"部分 rank prefill、其余 rank 陪跑 idle"的空泡。

```python
if global_exists_token_watermark_force_allow:      # 安全阀优先
    return _NegotiateOutput(allow=True, reason="token_watermark", next_state=None)

prev_delayed_count = prev_state.delayed_count if prev_state else 0
if prev_delayed_count < self._max_delay_passes - 1:
    next_state = (prev_state or _State()).bump_delayed_count()
    return _NegotiateOutput(allow=False, reason="delay", next_state=next_state)  # 继续等
else:
    return _NegotiateOutput(allow=True, reason="wait_timeout", next_state=None)  # 等够了，放行
```

逻辑：只要还没等满 `max_delay_passes` 步，就 **deny**（让想 prefill 的 rank 也先去 decode，等更多 rank 攒齐 prefill 再一起做）；等满了就 **强制放行**（防饿死），reason=`wait_timeout`。

### 8.5 token 水位安全阀（贯穿 all / mixed 两分支）

```python
# 本地判定（prefill_delayer.py:147）
local_token_watermark_force_allow = (
    local_prefillable
    and (self._token_usage_low_watermark is not None)
    and (token_usage < self._token_usage_low_watermark)
)
# 全局：任一 rank 触发即触发（prefill_delayer.py:174）
global_exists_token_watermark_force_allow = global_token_watermark_force_allow.max() > 0
```

语义：**只要有任何一个 rank 的 KV 使用率低于水位线，就说明 GPU 有空闲算力被浪费，立即放行 prefill**（reason=`token_watermark`），不再攒批。这是防止 delayer 在低负载时误伤延迟的关键安全阀。`None` 时关闭。

> 测试用例 `mixed_watermark_force_allow`（token_usage=[0.5,...]）验证只要 rank0 低于 0.8 水位就放行；`mixed_watermark_disabled`（watermark=None）验证关闭时仍然 delay。

---

## 9. 跨 DP 信息同步：\_gather\_info

（prefill_delayer.py:303）每个 rank 把 5 个字段打包成一个 int64 张量，all-gather 到全局缓冲，再取 attn_tp rank 0 的信息。

```python
def _gather_info(self, local_prefillable, local_token_watermark_force_allow,
                 running_batch=0, max_prefill_bs=0, waiting_queue_len=0):
    local_info = torch.tensor(
        [
            int(local_prefillable),                      # [0]
            int(local_token_watermark_force_allow),      # [1]
            running_batch,                               # [2]
            max_prefill_bs,                              # [3]
            waiting_queue_len,                           # [4]
        ],
        device=self._gather_device, dtype=torch.int64,
    )
    torch.distributed.all_gather_into_tensor(
        self._global_info_buffer.flatten(), local_info, group=self._gather_group,
    )
    tp0_info = self._global_info_buffer[:, 0, :]   # 只取每个 DP 组的 attn_tp rank0
    return tp0_info
```

缓冲区形状 `(dp_size_dim, attn_tp_size, 5)`（prefill_delayer.py:99）。取 `[:, 0, :]` 是因为**同一 DP 组内 attn_tp 各 rank 的 prefill 决策一致**，只需 rank0 的值代表整组。

### 9.1 gloo（CPU）vs NCCL（device）两条 all-gather 路径

（prefill_delayer.py:82-94，PR #24768）

```python
use_nccl = (
    server_args.disable_overlap_schedule
    or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
)
if use_nccl:
    assert device_group is not None
    self._gather_group = device_group   # NCCL, on GPU
    self._gather_device = device        # e.g. "cuda"
else:
    self._gather_group = cpu_group      # gloo, on CPU
    self._gather_device = "cpu"
```

这刻意**镜像了 `dp_attn.py:198-206` 里 `prepare_mlp_sync_batch_raw` 的同款选择**：默认 overlap schedule 下用 gloo CPU group（避免占用 GPU compute stream、避免与 forward stream 争抢），只有在关掉 overlap 或显式开 env flag 时才走 NCCL device group。保持两处一致，delayer 的同步才不会与调度器主同步路径产生 stream/ordering 冲突。

| 模式 | group | device | 场景 |
|------|-------|--------|------|
| 默认（overlap on） | `cpu_group`（gloo） | `"cpu"` | 不占 GPU，与 mlp_sync 路径一致 |
| `disable_overlap_schedule` 或 env flag | `device_group`（NCCL） | `tp_group.device` | 关 overlap 时；需要 `device_group != None` |

### 9.2 构造入参（scheduler.py:1056）

```python
self.prefill_delayer = PrefillDelayer(
    dp_size=self.ps.dp_size,
    attn_tp_size=self.ps.attn_tp_size,
    cpu_group=self.tp_cpu_group,
    device_group=self.tp_group.device_group,
    server_args=self.server_args,
    metrics_collector=(self.metrics_collector if metrics enabled else None),
    max_delay_passes=self.server_args.prefill_delayer_max_delay_passes,
    token_usage_low_watermark=self.server_args.prefill_delayer_token_usage_low_watermark,
    device=self.tp_group.device,
)
```

---

## 10. 状态机 \_State 与超时

（prefill_delayer.py:20）`_State` 是一个 frozen dataclass，记录一次"连续延迟"的累计信息：

```python
@dataclass(frozen=True)
class _State:
    delayed_count: int = 0                                  # 已连续延迟的 forward pass 数
    start_time: float = field(default_factory=time.perf_counter)   # 这一轮延迟的起点

    def bump_delayed_count(self) -> "_State":
        return dataclasses.replace(self, delayed_count=self.delayed_count + 1)
```

> 注意：仓库有 `.claude/rules/no-dataclasses.md` 要求新代码用 `msgspec.Struct`，但 `_State` 是既有 `@dataclass`（grandfathered），本文如实描述现状。

**状态生命周期**：
- 每次返回 `delay`（deny）：`next_state = (prev or _State()).bump_delayed_count()` —— 累加计数、保留 start_time。
- 每次返回 `allow`（任何 reason）：`next_state = None` —— **清空状态**，下一轮延迟重新计数、重置 start_time。
- `self._curr_state` 在 `_negotiate_should_allow_prefill`（薄包装）里被更新（prefill_delayer.py:132）。

**两种超时的区别**：

| 超时机制 | 计量单位 | 阈值参数 | 作用分支 | 放行 reason |
|---------|---------|---------|---------|------------|
| forward-pass 超时 | `delayed_count`（步数） | `max_delay_passes` | `mixed` | `wait_timeout` |
| 墙钟超时 | `elapsed_ms`（毫秒） | `max_delay_ms` | `all` 的 queue_condition | `wait_success` |

> `mixed` 分支只用步数超时（`prev_delayed_count < max_delay_passes-1`）；queue trigger 用墙钟超时（`elapsed_ms >= max_delay_ms`）。slot_condition 本身**没有独立超时**——它依赖 decode 推进使条件自然不再成立，或依赖 token 水位安全阀兜底。

---

## 11. 可观测性：三个指标与直方图 bug 修复

### 11.1 三个 Prometheus 指标（metrics_collector.py:929-973）

| 指标名 | 类型 | 说明 | 桶 |
|-------|------|------|----|
| `sglang:prefill_delayer_wait_forward_passes` | Histogram | 被放行的 prefill 累计等待了多少 forward pass | 默认 `[5,20,50,100,200]`，只保留 `<max_delay` 的桶，并强制补 `{0, max_delay-1}` |
| `sglang:prefill_delayer_wait_seconds` | Histogram | 累计等待墙钟秒数 | 默认 `[1,2,5,10,20,50,100,200,500]`，强制补 `{0}` |
| `sglang:prefill_delayer_outcomes_total` | Counter | 各类决策结果计数 | labels: `input_estimation`(all/none/mixed) × `output_allow` × `output_reason` × `actual_execution` |

桶设计细节（metrics_collector.py:937-961）：
- forward_passes 桶过滤掉 `>= max_delay` 的值（超时前不可能观测到），并补 `0`（零延迟直接放行）和 `max_delay-1`（用于区分"刚好等到超时"）。
- 两个直方图都补 `0` 桶，因为大量放行是"无需等待"（no_wait）的零延迟。

### 11.2 上报逻辑（metrics_collector.py:1151）

```python
def observe_prefill_delayer_outcome(self, forward_passes, wait_seconds,
        input_estimation, output_allow, output_reason, actual_execution):
    if output_allow and actual_execution:
        # 只有"放行且真的做了 prefill"才记入 wait 直方图
        self._log_histogram(self.prefill_delayer_wait_forward_passes, forward_passes)
        self._log_histogram(self.prefill_delayer_wait_seconds, wait_seconds)

    self.prefill_delayer_outcomes_total.labels(
        **self.labels,
        input_estimation=input_estimation,       # all / none / mixed
        output_allow=str(output_allow).lower(),
        output_reason=output_reason,             # no_wait/wait_success/wait_timeout/token_watermark/delay/""
        actual_execution=str(actual_execution).lower(),
    ).inc(1)
```

### 11.3 直方图"永远观测到 0"的 bug 及修复（PR #25975）

**问题根因**：`_NegotiateOutput.next_state` 在**所有放行路径上都是 `None`**（因为放行即清空状态）。早期实现从 `next_state` 里读累计等待值，于是放行时读到的永远是 `None`→0，wait 直方图永远只看到 0，完全失去意义。

**修复**：在 `_NegotiateOutput` 里新增两个**显式携带**的字段（prefill_delayer.py:29-40）：

```python
class _NegotiateOutput(NamedTuple):
    next_state: Optional[_State]
    ...
    wait_forward_passes: int = 0    # 显式携带本次放行前累计的等待
    wait_seconds: float = 0.0
```

在所有放行分支上，从 `prev_state`（而非 next_state）读取累计值（prefill_delayer.py:186-191）：

```python
wait_info = dict(
    wait_forward_passes=prev_state.delayed_count if prev_state else 0,
    wait_seconds=(time.perf_counter() - prev_state.start_time) if prev_state else 0.0,
)
```

- **放行路径**（all/none/mixed 的 allow 分支）都 `**wait_info` 把真实累计值带出去；
- **延迟路径**（deny）不带 wait_info（保持默认 0），因为等待尚未结束、不该观测。

测试 `test_prefill_delayer.py` 的 `all_prefillable_with_previous_wait`（expected_wait_forward_passes=1）与 `mixed_timeout`（=2）就是这个 bug 的回归守卫。

### 11.4 各 reason 与结果的对应表

| reason | 出现分支 | allow | 触发条件 | 含义 |
|--------|---------|-------|---------|------|
| `no_wait` | all | True | 未触发 slot/queue 且无 prev_state | 直接放行，本轮无等待 |
| `wait_success` | all | True | 之前有 prev_state（延迟过），本轮放行 | 攒批完成/墙钟超时放行 |
| `""`(空) | none | True | 没有 rank 想 prefill | 无所谓，放行 |
| `token_watermark` | all / mixed | True | 有 rank KV 使用率 < 水位 | 安全阀强制放行 |
| `delay` | all / mixed | False | slot/queue 触发 或 mixed 未超时 | 延迟本趟 prefill |
| `wait_timeout` | mixed | True | `delayed_count >= max_delay_passes-1` | 步数超时，防饿死放行 |

---

## 12. PD 分离模式支持

（PR #23588）早期 delayer 断言 `disaggregation_mode == "null"`，即只在非分离模式可用。现在支持 **disaggregated-prefill** 模式：

- **decode 引擎**：`--enable-prefill-delayer` 被忽略并打日志（scheduler.py:1050-1054）——decode 引擎没有 prefill 调度路径，delayer 是纯 no-op。
- **prefill 引擎**：正常构造 delayer。PD prefill 引擎依然要给 waiting 请求组 prefill 批，因此 delayer 的攒批/对齐逻辑同样有价值。

对应的测试 `SimpleNamespace(disaggregation_mode="null", ...)` 是单测用的默认；E2E 测试通过 `--enable-dp-attention --dp 8` 覆盖 DP 场景。

> PD prefill 模式下有一个额外约束：`PrefillAdder` 在 disaggregation prefill 时还会检查 `req_to_token_pool.available_size()`（schedule_policy.py:2844-2848 对应的 scheduler 侧逻辑），因为 prealloc/transfer 队列也占内存。delayer 的决策与这些预算检查是**独立叠加**的（delayer deny 会更早短路）。

---

## 13. 与 MinFreeSlotsDelayer 的区别

仓库里还有一个**同名相近但完全独立**的延迟器 `MinFreeSlotsDelayer`（`managers/min_free_slots_delayer.py`），二者容易混淆，务必区分：

| 维度 | **PrefillDelayer** | **MinFreeSlotsDelayer** |
|------|-------------------|------------------------|
| 目标 | 跨 DP rank 对齐 prefill、消除 DP 步调空泡 + 攒批 | 单 rank 攒批：等 running slot 腾出 ≥N 个再一次性 prefill |
| 作用域 | **跨 rank**（all-gather 协商） | **per-rank 本地**（running slot 是各 rank 私有） |
| 通信 | 有 all_gather（barrier） | 无通信 |
| 挂载点 | `PrefillAdder.add_one_req`（schedule_policy.py） | `get_new_batch_prefill` 顶部（scheduler.py:2757） |
| 触发 | prefillable 状态 + slot/queue/watermark | `running_bs>0 and num_allocatable_reqs < min_free_slots` |
| 典型场景 | DP attention | DFlash（草稿 prefill 昂贵）等 |
| 开关 | `--enable-prefill-delayer` | `--min-free-slots-delay`（DFlash 默认自动开） |
| 构造独立性 | 独立对象 | 独立对象（scheduler.py:887 注释明确"Built independently of the prefill delayer"） |

两者可以**同时存在**，各管一段：MinFreeSlots 在 `get_new_batch_prefill` 一进门就可能 `return None`（本地槽位不够，跳过整趟 prefill）；PrefillDelayer 则在真正逐条加入请求时做跨 rank 协商。

---

## 14. 历史演进

按 git 提交顺序（`git log -- prefill_delayer.py`）：

| 提交 | 内容 | 关键变化 |
|------|------|---------|
| #16269 | Refactor and fix prefill delayer (scheduler enhancer) | 从旧的 `scheduler_enhancer.py` 抽出为独立 `prefill_delayer.py`，引入纯函数 + `_State` 状态机 |
| #16363 | Support offline generation scenario | 覆盖离线批量生成场景 |
| #16471 | fix not support non-fcfs schedule policy | 兼容 lpm 等非 FCFS 调度策略 |
| #16603/#16812 | add/enhance metrics | 增加 outcomes/wait 直方图等可观测性 |
| #16814 | Support token low usage watermark | 加入 token 水位安全阀 |
| #16811 | Refactor for clarity and extensibility | `_negotiate_should_allow_prefill_pure` 纯函数化 |
| #17456 | prefill delay range expanded | **移除 `enable_dp_attention` 强制断言**；global_info 从 2 元组扩到 4 元组（加 running_batch、max_prefill_bs）；buffer 支持指定 device |
| #19836 | Skip the first delayer | 引入 `skip_first_delayer`，冷启动放过第一次以尽快建立 decode 批 |
| #20134 | fix bug when enable prefill delay and DP | DP 场景 bugfix |
| #20979 | compatible with multiple types of mem pool | `token_usage` 改用 `get_max_pool_usage()`（兼容 swa/mamba 混合池） |
| #23189 | **adaptive queue-based trigger** | 新增 `queue_condition`（queue_min_ratio + max_delay_ms），global_info 扩到 **5 元组**（加 waiting_queue_len）；触发器从"仅 slot"变为"slot ∪ queue" |
| #23389 | Remove deadcode | 清理死代码 |
| #23588 | **Allow in disaggregated-prefill** | 移除 `disaggregation_mode=="null"` 断言，支持 PD prefill；decode 引擎忽略 |
| #24768 | **support NCCL all-gather** | 增加 gloo/NCCL 双路径，镜像 mlp_sync 的 group 选择 |
| #25975 | **Fix wait histograms always observing 0** | `_NegotiateOutput` 显式携带 `wait_forward_passes/wait_seconds`，从 prev_state 读累计值 |

**演进主线**：
1. 从"仅 DP attention 专用"→"通用 prefill 调度增强"（去掉硬断言）；
2. 触发器从"单一 slot 条件"→"slot + queue 双触发 + 三重超时/水位保护"；
3. 通信从"gloo-only"→"gloo/NCCL 双路径，与主同步对齐"；
4. 可观测性从"无"→"三指标 + 修复直方图 bug"。

---

## 15. 调优与排错

### 15.1 何时开启

- **典型收益场景**：DP attention（`--enable-dp-attention --dp N`）+ 高并发 + 长输入（30k tokens 级）。测试实测离线场景 +20~27% 吞吐。
- **不建议开启**：单卡/无 DP、低并发、短输入——delayer 反而可能增加 TTFT 而无吞吐收益。

### 15.2 参数调优建议

| 症状 | 调整 |
|------|------|
| TTFT 抖动过大 / 请求偶发卡顿 | 调小 `--prefill-delayer-max-delay-passes`（默认 30），或设 `--prefill-delayer-token-usage-low-watermark 0.8` 让低负载时不延迟 |
| 吞吐提升不明显、prefill 仍碎片化 | 开启队列触发器：`--prefill-delayer-queue-min-ratio 0.1~0.5` |
| 队列触发器导致个别请求 TTFT 过长 | 调小 `--prefill-delayer-max-delay-ms`（默认 5000） |
| 低负载时也在延迟 | 设/调高 `--prefill-delayer-token-usage-low-watermark` 安全阀 |

### 15.3 观测指标怎么看

- `prefill_delayer_outcomes_total{output_reason="delay"}` 增长快 → delayer 在积极攒批。
- `output_reason="wait_timeout"` 占比高 → `max_delay_passes` 可能偏小（频繁触顶超时），或负载太"稀"以致攒不齐。
- `output_reason="token_watermark"` 占比高 → 安全阀频繁触发，说明 GPU 常空闲，delayer 收益有限。
- `wait_forward_passes` / `wait_seconds` 直方图 → 观测实际等待分布（修复 #25975 后才准确）。

### 15.4 常见排错

| 现象 | 排查方向 |
|------|---------|
| 启动报 assert `disable_overlap_schedule must be False` | delayer 与 overlap 强绑定，不能同时关 overlap。 |
| decode 引擎日志"Ignoring --enable-prefill-delayer" | 正常——decode 引擎无 prefill 路径，delayer no-op。 |
| DP 场景疑似死锁在 all-gather | 检查是否所有 rank 都进入了 negotiate（SinglePassExecutor 的 finalize 补齐机制若被破坏会死锁）；确认 gloo/NCCL group 一致。 |
| wait 直方图全是 0 | 若在 #25975 之前的版本，是已知 bug；之后应正常。 |
| 开 `SGLANG_PREFILL_DELAYER_DEBUG_LOG=1` | 逐趟打印 timeout/watermark 放行决策，定位延迟原因。 |

### 15.5 分布式单测（不需要 GPU）

`test_prefill_delayer.py:TestPrefillDelayerNegotiate` 用 **4-rank gloo** 后端跑 `_negotiate_should_allow_prefill` 纯逻辑，覆盖 all/none/mixed、watermark、slot/queue 触发、两类超时、wait 累计回归等 14 个用例——是理解决策语义最好的活文档。运行：

```bash
pytest test/registered/scheduler/test_prefill_delayer.py::TestPrefillDelayerNegotiate -v
```

E2E 吞吐/精度测试需 8×H200（`register_cuda_ci(runner_config="8-gpu-h200")`），目前 CI 中标记为 `Temporarily disabled`。

---

## 16. 关键文件 / 行号索引

| 文件:行号 | 符号 | 作用 |
|-----------|------|------|
| `managers/prefill_delayer.py:20` | `_State` | frozen dataclass：delayed_count + start_time |
| `managers/prefill_delayer.py:29` | `_NegotiateOutput` | NamedTuple：决策输出，显式携带 wait_* |
| `managers/prefill_delayer.py:43` | `PrefillDelayer` | 主类 |
| `managers/prefill_delayer.py:82` | gloo/NCCL 选择 | `use_nccl` 路径判定 |
| `managers/prefill_delayer.py:99` | `_global_info_buffer` | `(dp,attn_tp,5)` all-gather 缓冲 |
| `managers/prefill_delayer.py:110` | assert | 强制 `disable_overlap_schedule=False` |
| `managers/prefill_delayer.py:114` | `_negotiate_should_allow_prefill` | 薄包装（更新 self._curr_state） |
| `managers/prefill_delayer.py:136` | `_negotiate_should_allow_prefill_pure` | **核心纯函数**，三分支决策 |
| `managers/prefill_delayer.py:194` | `all` 分支 | slot/queue 触发 + skip_first |
| `managers/prefill_delayer.py:220` | queue_condition | 队列触发器 + 墙钟超时 |
| `managers/prefill_delayer.py:235` | slot_condition | 槽位触发器 |
| `managers/prefill_delayer.py:263` | `none` 分支 | 直接放行 |
| `managers/prefill_delayer.py:272` | `mixed` 分支 | 步数超时 + 强制放行 |
| `managers/prefill_delayer.py:303` | `_gather_info` | all_gather 打包 5 元组 |
| `managers/prefill_delayer.py:331` | `PrefillDelayerSinglePassExecutor` | 单趟幂等 + finalize 补齐 |
| `managers/prefill_delayer.py:371` | `_record_single_pass_result` | 调试日志 + 上报指标 |
| `managers/scheduler.py:1047` | `self.prefill_delayer` | 构造入口 |
| `managers/scheduler.py:1050` | decode 引擎忽略 | no-op 日志 |
| `managers/scheduler.py:2714` | `get_new_batch_prefill` | 顶层：建 executor + finalize |
| `managers/scheduler.py:2942` | `self.max_prefill_bs = max(...)` | 记录历史最大 prefill 批（喂给 delayer） |
| `managers/schedule_policy.py:443` | `PrefillAdder.__init__` | 接收 `prefill_delayer_single_pass` |
| `managers/schedule_policy.py:838` | `add_one_req_ignore_eos` | 检查点二（仅 prefillable） |
| `managers/schedule_policy.py:885` | `add_one_req` | 检查点一（全上下文） |
| `managers/min_free_slots_delayer.py` | `MinFreeSlotsDelayer` | 独立的本地攒批延迟器（对照） |
| `server_args.py:1250-1285` | ServerArgs | 7 个 `prefill_delayer_*` 参数 |
| `server_args.py:2914` | `_handle_prefill_delayer_env_compat` | 弃用环境变量映射 |
| `observability/metrics_collector.py:929-973` | 三个指标定义 | Histogram×2 + Counter |
| `observability/metrics_collector.py:1151` | `observe_prefill_delayer_outcome` | 上报逻辑 |
| `scheduler_components/dp_attn.py:198-206` | gloo/NCCL 选择 | delayer 镜像的对象 |
| `test/registered/scheduler/test_prefill_delayer.py` | 单测 + E2E | 14 个 negotiate 用例 + 吞吐/精度/水位测试 |

---

## 附：一句话总结

> **Prefill Delayer 是 DP attention 场景下的跨-rank prefill 协商器**：每步用一次 all-gather 收集各 rank 的 `[prefillable, watermark, running_bs, max_prefill_bs, queue_len]`，据此判 all/none/mixed 三态——`mixed` 时延迟到多 rank 对齐（步数超时兜底）、`all` 时按 slot/queue 触发器攒批（墙钟超时兜底）、任一 rank KV 空闲时水位安全阀强制放行；用 `SinglePassExecutor` 保证每步每 rank 恰好协商一次以对齐 all-gather，用 `_NegotiateOutput` 显式携带累计等待以正确上报直方图。默认关闭，与 overlap schedule 强绑定，独立于 per-rank 的 `MinFreeSlotsDelayer`。
