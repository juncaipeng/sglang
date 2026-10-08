# SGLang Overlap Schedule（重叠调度）方案系统梳理

> 核心文件：`python/sglang/srt/managers/scheduler.py`、`python/sglang/srt/managers/overlap_utils.py`、`python/sglang/srt/managers/utils.py`
> 目标读者：需要理解、调试或扩展 SGLang 调度器与 GPU 执行重叠机制的工程师。
> 本文由浅入深，覆盖：为什么需要重叠调度、核心思想、单线程多 CUDA Stream 架构、FutureMap 与跨步中继机制、event_loop_overlap 主循环、与投机解码（spec v2）/ PD 分离 / 语法约束的交互、历史线程方案的演进，以及调优与排错。
>
> ⚠️ 重要说明：当前 SGLang 的重叠调度**不再使用后台线程**（旧的 `TpModelWorkerClient` / `tp_worker_overlap_thread.py` 已于 PR #11210、#11300 删除）。现在完全由 `Scheduler` 内部的 `schedule_stream` / `forward_stream` 双 CUDA Stream + `FutureMap` 驱动。本文以**当前实现**为主，末尾附历史演进对照。
>
> 📌 **本次更新（对齐当前 main，2026-08）**：代码相较早期实现又演进出若干机制，本文已同步。要点：
> 1. **输入物化改为模块级函数 `resolve_forward_inputs`**（`overlap_utils.py:84`），不再叫 `resolve_future`；负数占位符 sentinel 已废弃，改为 `batch.input_ids = None` + 下一步直接从 buffer gather。
> 2. **结果写回 `stash` 由 `_relay_forward_payload` + `RelayPayload` 封装**（`scheduler.py:3855`）。
> 3. **结果 D2H 拷贝挪到独立 `copy_stream`**（非 HIP），不再串行在 `forward_stream`，进一步与下一步 forward 重叠。
> 4. **新增 WAR（写后读）栅栏 `_apply_war_barrier`**（`scheduler.py:1686`）：每次发射后用细粒度 read-done event 把后续 schedule_stream 工作排在 forward 的共享 buffer 读之后。
> 5. **`publish_ready` event 链式记录**（PR #30435）：spec_v2 的 seq_lens D2H 用私有流 + pinned buffer 拉取（`needs_cpu_seq_lens` 可关，PR #30471）。
> 6. **重叠 + 投机 + 语法约束现已支持**（对能力 `supports_grammar_overlap()` 的算法）：grammar FSM 在 `verify()` 内经 `grammar_barrier` 推进，覆盖了旧文"暂不支持"的结论。
> 7. **新增 DSpark 的 `ConfidenceRelay`**（PR #30261）：置信度调度投机解码的跨步 ring buffer 中继。
> 8. `get_next_batch_to_run` 现返回 `plan`（含 `running_batch` + `batch_to_run`）。

---

## 目录

1. [总览：什么是重叠调度，为什么需要它](#1-总览)
2. [核心思想：把 CPU 调度藏到 GPU 计算背后](#2-核心思想)
3. [整体架构与数据流](#3-整体架构与数据流)
4. [开关与默认值：何时启用、何时自动禁用](#4-开关与默认值)
5. [双 Stream 模型：schedule_stream vs forward_stream](#5-双-stream-模型)
6. [event_loop_normal vs event_loop_overlap 逐行对比](#6-两个主循环对比)
7. [数据依赖的破除：FutureMap 与跨步中继](#7-futuremap-与跨步中继relay)
8. [run_batch 的重叠路径：发射、stash、publish、copy](#8-run_batch-的重叠路径)
9. [结果回收：process_batch_result 与 copy_done 同步](#9-结果回收)
10. [跨 Stream 张量生命周期：record_batch_in_overlap](#10-跨-stream-张量生命周期)
11. [与投机解码（spec v2）的交互](#11-与投机解码的交互)
12. [与语法约束 / 结构化输出的交互：延迟采样](#12-延迟采样)
13. [与 PD 分离的交互](#13-与-pd-分离的交互)
14. [历史演进：从后台线程到双 Stream](#14-历史演进)
15. [调优与排错](#15-调优与排错)
16. [关键文件 / 行号索引](#16-关键文件行号索引)

---

## 1. 总览

LLM 推理在 decode 阶段是**逐 token、串行迭代**的：每一步 forward 只产出 batch 中每条请求的 1 个 token，单步 GPU kernel 时间往往只有几百微秒到数毫秒。在如此短的步长下，**CPU 侧的调度开销**（接收请求、组 batch、分配 KV、构造 `ForwardBatch`、采样后处理、检测结束、流式输出等）极易成为瓶颈——如果 CPU 与 GPU 严格串行执行，GPU 在每一步之间都会出现“等 CPU”的空泡（bubble）。

**重叠调度（Overlap Schedule）** 的目标就是消除这些空泡：让**第 N+1 步的 CPU 调度工作**与**第 N 步的 GPU forward+sample** 并发执行，从而把 CPU 时间“藏”到 GPU 计算背后。

一句话定位：**重叠调度 = 把 GPU forward 发射到独立 CUDA Stream + 用 FutureMap 占位符打破“下一步输入依赖上一步输出”的数据依赖 + 把结果回收推迟一个迭代，使 CPU 永远不在刚发射后立即等 GPU。**

效果（典型 decode 场景）：

```
非重叠（串行）：
CPU: [调度N]            [调度N+1]            [调度N+2]
GPU:        [forward N]          [forward N+1]          [forward N+2]
            ↑ CPU 等 GPU         ↑ GPU 等 CPU（bubble）

重叠：
CPU: [调度N][调度N+1][调度N+2][调度N+3] ...
GPU:        [forward N][forward N+1][forward N+2] ...
            └──────── CPU 与 GPU 持续并行 ────────┘
```

---

## 2. 核心思想

要让“第 N+1 步调度”与“第 N 步 GPU 计算”并行，必须解决三个难题：

| 难题 | 本质 | SGLang 的解法 |
|---|---|---|
| **① 发射不能阻塞** | 普通 `forward → sample → .tolist()` 会触发 CPU 同步等待 GPU 结果 | forward 跑在**独立的 `forward_stream`** 上，发射后 CPU 立即返回继续调度；结果回收推迟一个迭代 |
| **② 数据依赖** | 第 N+1 步 decode 的输入 `input_ids` = 第 N 步采样出的 token，但第 N 步还没算完 | 步间 `batch.input_ids = None`；真正的 token 由第 N 步写进 `FutureMap` buffer，第 N+1 步在 forward 入口用 `resolve_forward_inputs` **按 `req_pool_indices` gather** 出来 |
| **③ 跨 Stream 张量生命周期** | schedule stream 产出的张量被 forward stream 在一个迭代后消费，可能被 PyTorch 缓存分配器提前回收 | 用 `batch_record_buf`（2 槽环形缓冲）把张量钉住 2 个迭代 |

三者合起来，使得：
- CPU 在迭代 N 里**只发射**迭代 N 的 GPU 工作，并处理迭代 **N-1** 的结果；
- CPU 真正“等 GPU”发生在处理 N-1 的结果时（`copy_done.synchronize()`），而那一步的 GPU 工作早已完成，所以等待几乎是零成本。

这是一种**软件流水线（software pipelining）**：把“调度”和“执行”两个阶段错开一拍。

---

## 3. 整体架构与数据流

### 进程与线程模型（当前实现）

重叠调度**不再起后台线程**。它运行在 **`Scheduler` 单线程**里，靠**两条 CUDA Stream** 实现 CPU/GPU 并行：

```
┌────────────────────── Scheduler 进程（单 Python 线程）──────────────────────┐
│  整个 event loop 运行在 schedule_stream 上下文里                             │
│  (run_event_loop: with StreamContext(schedule_stream))                      │
│                                                                             │
│  ┌──────────────┐  进入 forward_stream_ctx  ┌────────────────────────────┐ │
│  │ schedule_    │  ───────────────────────► │       forward_stream       │ │
│  │ stream       │                           │ wait_stream(schedule_strm) │ │
│  │              │                           │ resolve_forward_inputs      │ │
│  │ get_next_    │                           │  (从 FutureMap gather token) │ │
│  │ batch        │                           │ forward_batch_generation   │ │
│  │ prepare_for_ │                           │ _relay_forward_payload(stash)│ │
│  │ decode       │                           │ copy_done.record()         │ │
│  │              │                           └────────────────────────────┘ │
│  │ process_     │  copy_done.synchronize() ◄── 一个迭代后才在 CPU 端 join  │
│  │ batch_result │                                                          │
│  │ (N-1)        │                                                          │
│  └──────────────┘                                                          │
│         │                                                                  │
│         ▼  FutureMap（GPU buffer，按 req_pool_idx 索引）                    │
│   output_tokens_buf / new_seq_lens_buf / topk_p_buf / ...                  │
└────────────────────────────────────────────────────────────────────────────┘
```

### 关键对象

| 对象 | 位置 | 作用 |
|---|---|---|
| `schedule_stream` | `scheduler.py:1665`（`run_event_loop` 内建） | 整个 event loop 运行的默认流，承载 CPU 侧的输入准备 kernel |
| `forward_stream` | `model_runner.py:403`（创建），`scheduler.py:1471`（包成 ctx） | 模型 forward + 采样在此流上发射，与调度并发 |
| `copy_stream` | `scheduler.py:1474` | 结果 D2H 拷贝流（非 HIP，用于把结果回拷与下一次 forward 重叠） |
| `FutureMap` | `overlap_utils.py:232` | 跨迭代中继 buffer，存“下一步调度算不出来的值”（采样 token、spec seq_lens 等） |
| `result_queue` | `scheduler.py:1741` | `deque`，缓存 `(batch.copy(), batch_result)`，实现“推迟一拍处理” |
| `batch_record_buf` | `scheduler.py:1482` | 2 槽环形缓冲，钉住跨流张量防止被 GC |
| `GenerationBatchResult.copy_done` | `utils.py:69` | CUDA Event，标记异步 D2H 拷贝完成点 |

---

## 4. 开关与默认值

### 运行时标志（`scheduler.py:427`）

```python
self.enable_overlap = not server_args.disable_overlap_schedule and not use_mlx()
self.enable_overlap_mlx = not server_args.disable_overlap_schedule and use_mlx()
```

**重叠调度默认开启**，仅由 `disable_overlap_schedule` 控制。

### CLI / 默认值（`server_args.py`）

```python
# server_args.py:954  —— 声明式注解字段，CLI --disable-overlap-schedule 由 Arg 注解自动生成
disable_overlap_schedule: A[bool, Arg(help="Disable the overlap scheduler ..."), NS("schedule")] = False
```

> 注意只有 `--disable-overlap-schedule`，没有 `--enable-*` 开关；重叠是默认行为。

### 自动禁用 / 约束的场景

| 位置 | 条件 | 处理 |
|---|---|---|
| `server_args.py:4244`（`_handle_mps_backends`） | `device == "mps"` 且非 MLX | 自动置 `disable_overlap_schedule = True` |
| `server_args.py:8204`（`_dllm_overlap_disable` post-pass，定义于 `arg_groups/overrides.py:2627`） | Diffusion LLM (dLLM) | 后处理 pass 关闭重叠 |
| `server_args.py:5640`（`_validate_mamba_no_buffer`） | Mamba `no_buffer` 策略 | **断言**要求 `disable_overlap_schedule` 已关 |
| `server_args.py:8875` | `pp_size > 1` | **断言** `disable_overlap_schedule and speculative_algorithm is None`（PP 与重叠+投机不兼容） |
| `server_args.py:8933` | `enable_pdmux` | **断言** `disable_overlap_schedule`（同卡 PD 复用与重叠互斥） |

> 工程提示：日常 serving 不要手动加 `--disable-overlap-schedule`；只有在调试调度逻辑、复现某些时序相关 bug、或上述不兼容场景下才关闭它。

---

## 5. 双 Stream 模型

### 两条流的创建

`forward_stream` 在 `ModelRunner.__init__` 创建，并通过 `get_worker_info()` 交给 Scheduler：

```python
# model_executor/model_runner.py:403
self.forward_stream = torch.get_device_module(self.device).Stream()
...
# model_executor/model_runner.py:408  WAR 快路径用的 read-done event（下节 §10）
self.war_fastpath_read_done_event: Optional[torch.cuda.Event] = None
```

`schedule_stream` 在进入 event loop 前创建，**整个循环都运行在它的上下文里**：

```python
# scheduler.py:1665  (run_event_loop)
self.schedule_stream = self.device_module.Stream(priority=0)
if self.device == "cpu":
    self.schedule_stream.synchronize = lambda: None  # CPU 上是 no-op
with self.device_module.StreamContext(self.schedule_stream):
    dispatch_event_loop(self)
```

`init_overlap`（`scheduler.py:1428`）把 `forward_stream` 包成上下文管理器，并创建 `copy_stream`、`FutureMap` 与 `batch_record_buf`。注意 `FutureMap` 现在**总是创建**（两模式都用它中继 decode 的 `input_ids`），且构造时按后端算出 `needs_cpu_seq_lens` / `needs_confidence_relay` 两个开关：

```python
# scheduler.py:1428  init_overlap()
def init_overlap(self):
    self.device_module = torch.get_device_module(self.device)
    needs_cpu_seq_lens = decide_needs_cpu_seq_lens(self.server_args, attn_backends)   # overlap_utils.py:22
    needs_confidence_relay = decide_needs_confidence_relay(self.server_args)          # overlap_utils.py:51
    self.future_map = self.spec_algorithm.create_future_map(                          # 总是建
        self.device, self.req_to_token_pool,
        needs_cpu_seq_lens=needs_cpu_seq_lens,
        needs_confidence_relay=needs_confidence_relay,
    )
    ...
    self.forward_stream_ctx = self.device_module.stream(self.forward_stream)          # 1471
    self.copy_stream = self.device_module.Stream()                                    # 1474
    self.copy_stream_ctx = self.device_module.stream(self.copy_stream)
    if not self.enable_overlap:
        return
    self.batch_record_buf = [None] * 2                                               # 1482
    self.batch_record_ct = 0
```

### 两条流如何串起依赖

关键是 `wait_stream`：forward stream 在做事之前，先等 schedule stream 上已入队的输入准备 kernel 完成（例如构造 `req_pool_indices`、`seq_lens` 的拷贝/scatter）。

```python
# scheduler.py:3644 (run_batch 重叠路径)
self.future_map.resolve_seq_lens_cpu(batch)            # ⓪ spec 专属：懒加载上步 seq_lens
with self.forward_stream_ctx:
    self.forward_stream.wait_stream(self.schedule_stream)   # ① 跨流栅栏
    # resolve 会消费 SB 的 staging（prefill_input_ids_cpu / mix_running_indices），
    # 必须在 isolation 之外跑，好让快照抓到“已消费”后的状态。
    resolve_forward_inputs(batch, self.future_map)     # ② 物化 input_ids（decode gather / prefill H2D / mixed cat）
    with self._forward_isolation(batch, overlap=True): # ③ 事务化：SB 快照/恢复 + 张量保活
        batch_result = self.model_worker.forward_batch_generation(batch, **fwd_kwargs)  # ④ 模型+采样
```

> 变化：旧的“负数占位符 + `resolve_future` kernel”已废弃。现在步间 `batch.input_ids=None`，下一步在 forward 入口由**模块级函数 `resolve_forward_inputs`**（`overlap_utils.py:84`）从 FutureMap buffer gather 出真 token（详见 §7）。

注意 `wait_stream` 是**GPU 侧**的等待（在 GPU 流之间插入依赖），**CPU 不阻塞**——`with` 块执行完后，CPU 立刻返回主循环继续下一步调度。真正的 CPU 阻塞只发生在下一迭代处理 N-1 结果时的 `copy_done.synchronize()`。

### 分工总结

| 流 | 跑什么 | 是否阻塞 CPU |
|---|---|---|
| `schedule_stream` | event loop 全程；输入张量准备（拼 batch、scatter、seq_lens+1 等） | 否（调度是 CPU 工作，但 stream 上的 kernel 异步入队） |
| `forward_stream` | `resolve_forward_inputs` + 模型 forward + 采样 + `_relay_forward_payload`(stash) | 否（发射后立即返回） |
| `copy_stream` | 结果 D2H 回拷（非 HIP），与下一次 forward 重叠 | 否 |

---

## 6. 两个主循环对比

### `event_loop_normal`（非重叠，`scheduler.py:1704`）

```python
def event_loop_normal(self):
    while True:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused:
            continue

        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)   # ← 立即处理，CPU 在此等 GPU
        else:
            self.on_idle()

        self.last_batch = batch
```

关键点：`run_batch` 之后**立刻** `process_batch_result` 处理**同一个** batch——CPU 必须等这一步 GPU forward+sample 结束（`.tolist()` 触发隐式同步）才能调度下一步。

### `event_loop_overlap`（重叠，`scheduler.py:1739`）

```python
def event_loop_overlap(self):
    self.result_queue: Deque[...] = deque()

    def pop_and_process():
        # 处理“上一批”的结果
        tmp_batch, tmp_result = self.result_queue.popleft()
        self.process_batch_result(tmp_batch, tmp_result)

    while True:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused:
            continue

        # get_next_batch_to_run 现返回 plan（running_batch + batch_to_run）
        plan = self.get_next_batch_to_run(
            running_batch=self.running_batch, last_batch=self.last_batch)
        self.running_batch = plan.running_batch
        batch = plan.batch_to_run
        self.cur_batch_for_debug = batch
        disable_overlap_for_batch = self.is_disable_overlap_for_batch(
            batch, last_batch=self.last_batch)

        # 若本批不需与上一批重叠，先立即处理上一批
        if disable_overlap_for_batch:
            pop_and_process()

        # 发射当前批（不立即处理结果，入队）
        if batch:
            batch_result = self.run_batch(batch)
            self._apply_war_barrier()               # ← 新增：WAR 写后读栅栏
            self.result_queue.append((batch.copy(), batch_result))
        else:
            batch_result = None

        # 处理上一批（这才是 CPU 真正 join GPU 的地方，但 join 的是 N-1）
        if self.last_batch:
            if not disable_overlap_for_batch:
                pop_and_process()
        elif batch is None:
            self.on_idle()

        # 当前批的（延迟）采样：依赖上一批的 grammar，放在上一批处理之后
        if self.is_generation:
            self.launch_batch_sample_if_needed(batch_result, batch)

        self.last_batch = batch
```

### 逐项差异

| 维度 | normal | overlap |
|---|---|---|
| 结果处理时机 | 发射后**立即**处理本批 | 发射本批，处理**上一批**（推迟一拍） |
| 队列 | 无 | `result_queue: deque`，存 `(batch.copy(), result)` |
| 为什么是 `batch.copy()` | — | 入队的是快照拷贝，主循环可继续 mutate 活的 batch 进入下一步 |
| CPU 何时等 GPU | 每步都等当前 GPU | 只在处理 N-1 时等（早已完成，近乎零成本） |
| 采样 | 在 forward 内同步完成 | 普通路径在 forward 内；语法约束走 `launch_batch_sample_if_needed` 延迟采样 |

### `is_disable_overlap_for_batch`：何时“塌缩”重叠（`scheduler.py:1813`）

某些 batch 不能或不应与上一批重叠，此时把上一批的处理**提前**到发射之前（即退化为同步一拍）：

```python
def is_disable_overlap_for_batch(self, batch, last_batch) -> bool:
    if self.require_mlp_sync:
        is_extend = lambda b: b and b.is_extend_in_batch
    else:
        is_extend = lambda b: b and b.forward_mode.is_extend()

    # ① 连续两个 prefill batch：禁用重叠以改善首批的 TTFT（受环境变量控制）
    disable_overlap_for_batch = (
        envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
        and is_extend(batch) and is_extend(last_batch)
    )

    # ② spec + grammar：仅对“不支持 grammar 重叠”的算法才同步一拍。
    #    grammar_needs_sync() 为真表示需要把 FSM 推进 land 在下一批 bitmask 之前。
    need_grammar_sync = (
        batch and not batch.spec_algorithm.is_none()
        and batch.grammar_needs_sync()
        and batch.forward_mode.is_decode()
        and len(self.result_queue) > 0
    )
    return disable_overlap_for_batch or need_grammar_sync
```

- ① `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`（默认 `False`）：连续 prefill 时为了首 token 延迟（TTFT）牺牲一点吞吐。
- ② **重叠 + 投机 + 语法约束**：**已支持重叠**（对 `supports_grammar_overlap()` 为真的算法，见 §11）。这类算法在 `verify()` 内部经 `grammar_barrier`（`_advance_pending_grammar`）推进 FSM，重叠掉 target forward，无需在此塌缩；只有**不支持** grammar 重叠的（host-draft 类）算法在 `grammar_needs_sync()` 为真时才同步一拍。这一点**修正了旧文的“暂不支持”结论**。

---

## 7. FutureMap 与跨步中继（relay）

这是整个重叠调度**最核心的机制**——它打破了“第 N+1 步输入依赖第 N 步输出”的数据依赖。

> 术语更新：早期实现用**负数占位符 sentinel**（`input_ids = -req_pool_idx`）+ `resolve_future` GPU kernel 翻译负数。**当前 main 已不再用 sentinel**：forward 结束后直接把 `batch.input_ids = None`，下一步在 forward 入口由模块级函数 `resolve_forward_inputs` **按 `req_pool_indices` 直接 gather** 出真实 token。旧的负数机制见 §14 历史演进。

### 问题

decode 第 N+1 步的输入 token 就是第 N 步采样出来的 token。但在重叠模式下，当 CPU 开始准备第 N+1 步时，第 N 步的 GPU 采样**还没算完**，CPU 根本拿不到真实 token。

### 解法：input_ids 延迟到 forward 入口物化（`resolve_forward_inputs`）

调度阶段**不填** decode 的 `input_ids`（保持 `None`）。真正 forward 前，在 forward stream 上由 `resolve_forward_inputs` 物化——decode 部分从 `output_tokens_buf` 按 `req_pool_indices` gather 出上一步的采样 token；prefill 部分从 pinned CPU staging H2D；mixed batch 两段 `cat`：

```python
# overlap_utils.py:84  （模块级函数，取代旧的 FutureMap.resolve_future）
def resolve_forward_inputs(batch, future_map):
    if batch.prefill_input_ids_cpu is not None:                 # prefill / mixed
        prefill_gpu = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        if batch.mix_running_indices is not None:               # mixed：decode 段 gather 后拼接
            decode_gpu = future_map.output_tokens_buf[batch.mix_running_indices]
            batch.input_ids = torch.cat([prefill_gpu, decode_gpu])
        else:
            batch.input_ids = prefill_gpu
        batch.prefill_input_ids_cpu = None
        batch.mix_running_indices = None
    elif batch.input_ids is None and future_map.spec_algo.is_none():  # 纯 decode
        batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]

    # 重叠 + 投机：额外中继 topk_p / topk_index / bonus / hidden_states
    if batch.enable_overlap and not batch.spec_algorithm.is_none():
        future_map._resolve_spec_extras(batch)
```

因为 `resolve_forward_inputs` 跑在 `forward_stream` 上、且在 `wait_stream(schedule_stream)` 之后，GPU 会**自然保证顺序**：“上一步的写回”→“这一步的 gather 读取”。注意它在 `_forward_isolation` **之外**调用（消费 `prefill_input_ids_cpu` / `mix_running_indices` 这些 staging），使 isolation 快照捕获的是 post-consume 状态。

### 写回：`_relay_forward_payload` + `RelayPayload`

第 N 步采样完，把结果散布到 buffer，供第 N+1 步 gather。stash 现由 `_relay_forward_payload`（`scheduler.py:3855`）封装，用 `RelayPayload`（`overlap_utils.py:128`）统一承载：

```python
def _relay_forward_payload(self, future_indices, batch_result):
    if self.spec_algorithm.is_ngram():
        return                                           # ngram 走 batch.spec_info，不走 FutureMap
    if batch_result.next_draft_input is not None:        # spec：从 draft_input 抽取
        payload = RelayPayload.from_draft_input(batch_result.next_draft_input)
    elif batch_result.has_sampled_token_ids:             # 非 spec：只填 bonus_tokens
        payload = RelayPayload(bonus_tokens=batch_result.next_token_ids)
    else:
        return
    self.future_map.stash(future_indices, payload)       # overlap_utils.py:501
```

`stash`（`overlap_utils.py:501`）：非 spec 只写 `output_tokens_buf[indices] = bonus_tokens`；spec 额外写 `topk_p_buf / topk_index_buf / hidden_states_buf / draft_probs_buf`（哪些由 `spec_algo` 决定，首次非空 stash 时按张量形状懒初始化 `_lazy_init_forward_buf`）。

### FutureMap 的 buffer 设计（`overlap_utils.py:232`）

```python
class FutureMap:
    def __init__(self, device, spec_algo, req_to_token_pool,
                 needs_cpu_seq_lens=True, needs_confidence_relay=False):
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]
        self.output_tokens_buf = torch.empty((self.req_pool_size,), dtype=torch.int64, device=device)
        self.new_seq_lens_buf  = torch.empty((self.req_pool_size,), dtype=torch.int64, device=device)
        # CUDA：为 seq_lens 的 D2H 拉取准备 pinned host 镜像 + 私有流
        if _is_cuda:
            self.new_seq_lens_cpu_pinned = torch.empty((self.req_pool_size,), dtype=torch.int64, pin_memory=True)
            self.fwd_prepare_d2h_stream  = device_module.Stream()
        self.publish_ready = None            # lazy device.Event()，仅 spec_v2 用
        self.confidence_relay = ConfidenceRelay(...)   # DSpark 置信度中继
```

**关键设计点**：
- buffer 用 `req_pool_idx` 索引（每个请求池槽位一个 slot），而不是旧实现里的“环形计数器”。这消除了计数器取模碰撞的复杂逻辑——每条请求固定对应一个 slot，槽位 0 对应 KV padding 行，使 CUDA graph padding batch（`req_pool_idx == 0`）天然无害。
- **`needs_cpu_seq_lens`**（由 `decide_needs_cpu_seq_lens` 算，`overlap_utils.py:22`）：某些后端只需 GPU 上的 `seq_lens` 前进、不需要 CPU 镜像，可关掉 `.cpu()` D2H 走 GPU-only 路径（PR #30471 加了 CI 守卫）。
- **`_DEBUG_ASSERT`**（`SGLANG_IS_IN_CI`）下 buffer 用 `-1` 毒化初始化：每行必须先写后读，`gather` 后立即写回 `-1`，捕捉“未 stash 就 gather”的 bug。

### 一个 decode 步的完整数据流（非投机）

```
迭代 N（CPU on schedule_stream）           迭代 N（GPU on forward_stream / copy_stream）
──────────────────────────────────       ──────────────────────────────────
plan = get_next_batch_to_run()
prepare_for_decode():
  seq_lens = seq_lens + 1   (调度可算)
run_batch():
  resolve_seq_lens_cpu(batch)             （非 spec no-op；spec 从 buf 懒加载）
  ┌ with forward_stream_ctx:
  │   wait_stream(schedule_stream) ──────► 等输入准备 kernel
  │   resolve_forward_inputs(batch) ─────► input_ids = output_tokens_buf[req_pool_idx]
  │   ┌ _forward_isolation(overlap=True): 快照 SB + 钉张量 2 迭代
  │   │   forward_batch_generation() ────► attention + sample → next_token_ids(GPU)
  │   │   publish(idx, seq_lens+1) ──────► new_seq_lens_buf[idx] = seq_lens+1
  │   │   _relay_forward_payload(idx) ───► output_tokens_buf[idx] = next_token_ids
  │   │   copy_done = Event()
  │   └   copy_stream.wait_stream(fwd); with copy_stream: copy_to_cpu(); copy_done.record()
  └ batch.input_ids = None                （下一步在 resolve_forward_inputs 里再 gather）
_apply_war_barrier() ───────────────────► 让后续 schedule_stream 工作排在 forward 的共享读之后
result_queue.append((batch.copy(), result))
                                          （GPU 此刻仍在跑迭代 N）
pop_and_process()  ── 处理迭代 N-1 的结果:
  copy_done[N-1].synchronize() ◄─────────  join 迭代 N-1 的 D2H（早完成→不真正停顿）
  追加 token 到 req.output_ids，流式输出
last_batch = batch  → 进入迭代 N+1
```

CPU 调度迭代 N 与 GPU 计算迭代 N 并行；CPU 只在迭代 N-1（早已完成）上等待，等待成本接近零。

---

## 8. run_batch 的重叠路径

`run_batch`（`scheduler.py:3612`）是发射 GPU 工作的核心。generation 的重叠路径（`scheduler.py:3644` 起）：

```python
if self.is_generation and self.enable_overlap:
    # spec_v2 才需要：从 buf 懒加载下一步 seq_lens（accept_lens 调度时未知）
    self.future_map.resolve_seq_lens_cpu(batch)
    if self._confidence_budget_prepare is not None:      # DSpark 置信度预算
        self._confidence_budget_prepare(batch, self.future_map)

    with self.forward_stream_ctx:
        self.forward_stream.wait_stream(self.schedule_stream)
        # ★ resolve 在 isolation 之外：它消费 prefill_input_ids_cpu / mix_running_indices
        #    这些 staging，要让 isolation 快照捕获 post-consume 状态。
        resolve_forward_inputs(batch, self.future_map)   # 物化 input_ids（gather / H2D）

        with self._forward_isolation(batch, overlap=True):   # 事务化 SB + 钉张量
            future_indices = batch.req_pool_indices
            fwd_kwargs = {}
            if not batch.spec_algorithm.is_none():
                # spec：verify 后、draft_extend 前回调 publish，让调度准备与 draft_extend 重叠
                fwd_kwargs["on_publish"] = partial(self.future_map.publish, future_indices)
                if batch.spec_algorithm.supports_grammar_overlap():
                    # grammar 重叠：把推进上一批 FSM 的 barrier 交给 worker，在 verify 内推进
                    fwd_kwargs["grammar_barrier"] = self._advance_pending_grammar

            batch_result = self.model_worker.forward_batch_generation(batch, **fwd_kwargs)
            if batch.spec_algorithm.is_none():
                self.future_map.publish(future_indices, batch.seq_lens + 1)
            if batch_result.extra_keep_alive_refs:               # 额外张量保活
                self.batch_record_buf[self.batch_record_ct].extend(
                    batch_result.extra_keep_alive_refs)
            # (unified memory 池) 记 forward_done event 供惰性压缩判 src 复用
            ...
            batch_result.copy_done = self.device_module.Event()
            if batch_result.delay_sample_func is None:           # 普通路径
                self._relay_forward_payload(future_indices, batch_result)
                if _is_hip:
                    batch_result.copy_to_cpu(...)                # HIP：inline
                else:
                    # ★ 结果 D2H 挪到 copy_stream，与下一步 forward 重叠（不占 forward_stream）
                    self.copy_stream.wait_stream(self.forward_stream)
                    with self.copy_stream_ctx:
                        batch_result.copy_to_cpu(...)
            else:                                                # 延迟采样路径
                batch_result.future_indices = future_indices

    batch.input_ids = None                                       # 下一步由 resolve_forward_inputs 重新 gather
    if not batch.spec_algorithm.is_none():
        batch.spec_info = batch_result.next_draft_input
        batch.spec_info.future_indices = future_indices
```

发射阶段做的关键事（与旧文差异已标注）：

1. **`resolve_forward_inputs`**（模块函数，取代 `resolve_future`）：物化 `input_ids`——decode gather / prefill H2D；**在 isolation 之外**执行。
2. **`forward_batch_generation`**：模型 forward + 采样，产出 GPU 上的 `next_token_ids`。
3. **`publish` / `_relay_forward_payload`**：把“下一步调度算不出来的值”写进 buffer——`publish` 写下一步 `seq_lens`（spec 经 `on_publish` 提前到 verify 后），relay 写采样 token / draft 额外量。
4. **`copy_to_cpu` + `copy_done.record()`**：发起**异步 D2H**并记录 event；非 HIP 走 **`copy_stream`**，与下一步 forward 并行；HIP 走 inline。
5. **`_apply_war_barrier()`**（在 event loop 里紧跟 `run_batch`）：WAR 写后读栅栏，见 §10。

最后 `batch.input_ids = None`——不再埋负数 sentinel，下一步在 `resolve_forward_inputs` 里按 `req_pool_indices` 重新 gather。

### 异步拷贝：`GenerationBatchResult.copy_to_cpu`（`utils.py`）

```python
def copy_to_cpu(self, return_logprob, return_hidden_states=True):
    if return_logprob:
        ... # next_token_logprobs / input_token_logprobs / top_logprobs 全部 .to("cpu", non_blocking=True)
    if return_hidden_states and self.logits_output.hidden_states is not None:
        self.logits_output.hidden_states = self.logits_output.hidden_states.to("cpu", non_blocking=True)
    self.next_token_ids = self.next_token_ids.to("cpu", non_blocking=True)
    if self.accept_lens is not None:
        self.accept_lens = self.accept_lens.to("cpu", non_blocking=True)
    ...
    self.copy_done.record()   # 记录“拷贝完成”event（非 HIP 在 copy_stream 上）
```

所有拷贝都是 `non_blocking=True`（异步），最后 `copy_done.record()` 记录一个 event，作为下一迭代 CPU 端的同步点。

---

## 9. 结果回收

### 统一入口：`process_batch_result`（`scheduler.py:3900`）

```python
def process_batch_result(self, batch, result):
    if batch.forward_mode.is_decode():
        self.batch_result_processor.process_batch_result_decode(batch, result)
    elif batch.forward_mode.is_extend():
        if batch.is_dllm(): ...
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.process_batch_result_disagg_prefill(batch, result)
        else:
            self.batch_result_processor.process_batch_result_prefill(batch, result)
    elif batch.forward_mode.is_idle():
        self.batch_result_processor.process_batch_result_idle(batch, result)
    ...
```

入口在重叠 / 非重叠模式下**完全相同**。重叠特有的行为是每个具体处理器开头的 `copy_done.synchronize()`——这是**推迟一拍的 CPU/GPU join**：

```python
# scheduler_components/batch_result_processor.py:811 (decode；prefill 203、idle 799、advance_grammar_fsm 755 同理)
def process_batch_result_decode(self, batch, result):
    if result.copy_done is not None:
        result.copy_done.synchronize()    # ← 等迭代 N-1 的 D2H 完成
    ...
    next_token_ids, next_token_logprobs = self._normalize_decode_outputs(...)
```

- 非重叠模式下 `copy_done is None`（没发起异步拷贝），跳过 synchronize，张量本就已在 CPU。
- 重叠模式下 `record()` 发生在迭代 N 的发射时（copy_stream / forward_stream），`synchronize()` 发生在迭代 N+1 的 CPU 上——这一拍延迟正好把 GPU 计算藏住。

### “请求完成 / 被回撤”只在重叠下出现（`batch_result_processor.py:850`）

```python
if (self.enable_overlap or self.enable_overlap_mlx) and (
    req.finished() or req.is_retracted
):
    # 注意：这种 (finished 或 retracted) 只在重叠调度开启时发生
```

因为重叠下，第 N 步发射时还不知道第 N 步的 token，调度器是“先发射、后判定结束”，所以一条请求可能在它的最后一个 token 还在飞行时就被推进了下一步——结束判定在 N-1 的结果处理里补做。

---

## 10. 跨 Stream 张量生命周期

重叠的一个隐蔽难题：schedule stream 产出的 GPU 张量，被 forward stream 在**一个迭代之后**消费。若 Python 端引用提前释放，PyTorch 缓存分配器可能把这块显存回收复用，导致 forward 读到脏数据。

解法：`record_batch_in_overlap` 把整个 batch 的字段快照进 **2 槽环形缓冲**钉住：

```python
# scheduler.py:3552
def record_batch_in_overlap(self, batch):
    # hacky：保留引用，避免 GPU 张量被 torch GC 提前释放
    attr_snapshot = [getattr(batch, f.name, None) for f in dataclasses.fields(batch)]
    self.batch_record_ct = (self.batch_record_ct + 1) % 2
    # 用 list（非 tuple），worker 之后可通过 extra_keep_alive_refs 追加引用
    self.batch_record_buf[self.batch_record_ct] = [batch, attr_snapshot]
```

`_forward_isolation`（`scheduler.py:3568`，取代旧的 `_overlap_forward_isolation`，现同时服务 overlap 与非 overlap，用 `overlap=` 参数区分）是个上下文管理器，把一次 forward **事务化**：

```python
@contextmanager
def _forward_isolation(self, batch, *, overlap):
    # 1. 快照 SB 字段：spec V2 会在 forward 中途 rebind seq_lens/spec_info/forward_mode，需要能回滚
    snapshot_v2_full = not batch.spec_algorithm.is_none()
    sched_snapshot = ({f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
                      if snapshot_v2_full else None)
    sched_sampling_info = batch.sampling_info
    # 2. 用 forward-only 副本替换 sampling_info，避免 V2 多次 init_new 重复累加惩罚
    if sched_sampling_info is not None:
        batch.sampling_info = sched_sampling_info.copy_for_forward()
    # 3. 仅 overlap：钉住 2 个迭代的张量生命周期（必须在 sampling_info 替换之后，连副本一起钉）
    if overlap:
        self.record_batch_in_overlap(batch)
    try:
        yield
    finally:
        if snapshot_v2_full:
            for name, value in sched_snapshot.items():
                setattr(batch, name, value)   # 回滚 V2 中途的 mutate
        else:
            batch.sampling_info = sched_sampling_info
```

三个职责：
1. **快照/回滚**：spec V2 会在 forward 中途改写 `seq_lens` / `spec_info` / `forward_mode` 等，结束时恢复成调度态。
2. **sampling_info 隔离**：用 `copy_for_forward()` 的副本喂给 forward，避免重复累加惩罚项。
3. **张量保活**：`overlap=True` 时调用 `record_batch_in_overlap` 钉住 2 个迭代（非 overlap 单流不需要）。

worker 还可以通过 `GenerationBatchResult.extra_keep_alive_refs` 追加要保活的引用（`scheduler.py:3688`），它们被塞进同一个环形 slot。

### WAR 写后读栅栏 `_apply_war_barrier`（`scheduler.py:1686`）

另一个隐蔽的正确性问题：forward 读某些**共享静态 buffer**（CUDA graph 的 decode 输入区、spec 的 draft_extend 区等），而下一步 schedule_stream 的写入（result 处理、下一迭代准备）可能覆盖同一 buffer——这是**写后读（WAR）冒险**。event loop 里每次 `run_batch` 之后立即调 `_apply_war_barrier()` 把后续 schedule_stream 工作排在 forward 的“共享读完成”之后：

```python
def _apply_war_barrier(self):
    if not self._war_barrier_enabled:      # 默认 CUDA 开（is_cuda()）；也可 SGLANG_ENABLE_WAR_BARRIER 强开
        return
    runner = self.model_worker.war_fastpath_runner
    ev = runner.war_fastpath_read_done_event      # forward 在快照后 record 的细粒度 read-done event
    runner.war_fastpath_read_done_event = None
    if ev is not None and not envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get():
        self.schedule_stream.wait_event(ev)        # 快路径：只等“共享读完成”，占用率高
    else:
        self.schedule_stream.wait_stream(self.forward_stream)  # 慢路径：整个 forward 完成
```

- **快路径**：forward 在读完共享 buffer 后（decode graph replay 后 / spec draft_extend 后）record 一个 `read_done` event，schedule_stream 只等这个点，让 forward 的后半段（如采样、拷贝）仍能与调度重叠。
- **慢路径**（`SGLANG_FORCE_COARSE_WAR_BARRIER` 或无 event）：退化为等整个 forward，牺牲部分重叠换正确性。

---

## 11. 与投机解码的交互

投机解码（spec v2，EAGLE 系）下，重叠更复杂，因为**接受长度（accept_lens）在调度时未知**——它取决于 GPU verify 的结果。

> 术语更新：旧文引用的 `ScheduleBatch.is_spec_v2` 属性**已从当前 main 删除**。现在判定统一用 `not batch.spec_algorithm.is_none()`（如 `scheduler.py:3586 / 3666 / 3733` 等），是否走 spec-v2 重叠路径由 `enable_overlap` 与算法能力（`spec_algorithm.supports_spec_v2()`）共同决定。语义不变：**spec v2 = 重叠开启 + 投机算法非空**（EAGLE 系或 standalone）。

### 额外难题与解法

| 难题 | 解法 | 代码 |
|---|---|---|
| 下一步 `seq_lens` 依赖本步 accept_lens（调度算不出） | `publish` 写进 `new_seq_lens_buf`，下一步 `resolve_seq_lens_cpu` 懒加载 | `overlap_utils.py:470/412` |
| draft 输入（topk_p / topk_index / bonus_tokens / hidden_states）也要跨步中继 | `_relay_forward_payload`→`stash` 多写几个 buffer，`resolve_forward_inputs` 走 `_resolve_spec_extras` 取出 | `overlap_utils.py:501/363` |
| 输出 token 数不定（每条请求 accept 长度不同） | 结果处理时按 `accept_lens` 切分扁平的 `next_token_ids` | `batch_result_processor.py:627` |

### `on_publish`：把 publish 提前到 verify 之后

spec v2 的 `forward_batch_generation`（`eagle_worker_v2.py`）在 verify 完成后、draft_extend 之前就回调 `on_publish`，这样**调度准备能与较慢的 draft_extend 重叠**：

```python
# decode 分支（eagle_worker_v2.py）
verify_input = self.draft_worker.draft(batch)
batch.spec_info = verify_input
batch_output = self.verify(batch)
# verify 结束就 publish，把栅栏放在 verify-end
if on_publish is not None:
    on_publish(batch_output.new_seq_lens)
# 之后才做 draft_extend（可与下一步调度准备并行）
self.draft_worker._draft_extend_for_decode(batch, batch_output)
return batch_output
```

### `resolve_seq_lens_cpu`：spec 专属的懒加载（`overlap_utils.py:412`）

```python
def resolve_seq_lens_cpu(self, batch):
    fi = batch.spec_info.future_indices if batch.spec_info is not None else None
    if fi is None:
        return                          # 非 spec_v2：no-op
    if self.publish_ready is not None:
        if _is_hip: self.publish_ready.synchronize()   # AMD：Event.wait() 回归 TPOT
        else:       self.publish_ready.wait()          # 等 publish event（GPU 侧栅栏）
    batch.seq_lens = self.new_seq_lens_buf[fi]         # GPU gather（每 verify 必须前进）

    if not self.needs_cpu_seq_lens:                    # 后端可 opt-out CPU 镜像 → GPU-only
        batch.seq_lens_cpu = None; batch.seq_lens_sum = None
        return
    # CUDA：私有 fwd_prepare_d2h_stream 上 gated-on-publish 拉进 pinned buffer，不阻塞 schedule_stream
    self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
    with device_module.stream(self.fwd_prepare_d2h_stream):
        self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
    self.fwd_prepare_d2h_stream.synchronize()
    batch.seq_lens_cpu = self.new_seq_lens_cpu_pinned[batch.req_pool_indices_cpu]
    batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
```

**`publish_ready` event 链式记录（PR #30435）**：`publish`（`overlap_utils.py:470`）在 spec 路径 record 一个 `publish_ready` event 供上面的 D2H gating。当同一 event 已存在时，先 `current_stream().wait_event(publish_ready)` 再 `record()`——**链式**保证：event 触发即代表**此前所有 publish 都可见**，于是一个“非 forward-stream 上的 publish”（PD-decode 的 prebuilt seeding）不会丢掉在飞 forward 的栅栏。

> 对比：**非 spec** 的下一步 `seq_lens` 是“调度可确定”的——decode 每步 +1，`prepare_for_decode` 里直接 `self.seq_lens = self.seq_lens + 1`，无需 FutureMap 中继。这正是 FutureMap 注释里说的“schedule-deterministic 的值由 SB 自己维护，不走中继”。

### spec 结果切分（`batch_result_processor.py:627 _resolve_spec_v2_tokens`）

```python
def _resolve_spec_v2_tokens(self, result, batch):
    next_token_ids = result.next_token_ids.tolist()
    accept_lens = result.accept_lens.tolist()
    result.num_correct_drafts = sum(accept_lens) - len(batch.reqs)   # 不含 bonus
    ...
    stride = result.speculative_num_draft_tokens
    for i, req in enumerate(batch.reqs):
        predict_tokens.append(next_token_ids[i*stride : i*stride + accept_lens[i]])
```

（命名遵循仓库 spec 规则：`num_correct_drafts` 不含 bonus token，`accept_*` 含 bonus。）

### 重叠 + 投机 + 语法约束：**现已支持**（修正旧结论）

早期该组合不支持重叠，会逐批塌缩（旧文的 `TODO(lsyin)`）。**当前 main 已支持**：对 `spec_algorithm.supports_grammar_overlap()` 为真的算法，`run_batch` 把 `grammar_barrier=self._advance_pending_grammar` 传进 worker，worker 在 `verify()` 内、构造 bitmask 之前调用它，**在 GPU 跑 target verify 的同时**把上一批仍在 `result_queue` 里的 decode 结果的 grammar FSM 推进（`_advance_pending_grammar`，`scheduler.py:1851`）。这样下一次 `verify()` 的 bitmask 能看到上一批已提交的 token，无需同步塌缩。只有**不支持** grammar 重叠的算法（host-draft 类）才在 `grammar_needs_sync()` 为真时经 `is_disable_overlap_for_batch` 同步一拍（§6）。

### DSpark：置信度调度投机（`ConfidenceRelay`，PR #30261）

DSpark（confidence-scheduled speculative decoding）在 FutureMap 里挂了一个 `ConfidenceRelay`（`overlap_utils.py:153`）：把每个 draft 的置信度（`[req_pool_size, gamma]`）跨步中继给下一步的“接受预算”决策。因为置信度也在 GPU 上现算、CPU 调度时不可得，它用一个 **depth=3、lag=2 的 pinned host ring buffer**（`CONFIDENCE_RELAY_RING_DEPTH/LAG`）：`publish` 时 `scatter` 进 device buffer 并 `issue_ring_copy`（gated on `publish_ready` 的私有流异步 D2H + 记 `copy_done`）；下一步 `resolve`（滞后 2 拍、`copy_done.query()` 非阻塞）读环形槽。`needs_confidence_relay` 关闭时整条链是 no-op。

---

## 12. 延迟采样

延迟采样是一条**窄而专**的路径：只在**重叠 + 语法约束 / 结构化输出 + 非投机**时启用（`tp_worker.py:613`）。此时采样必须依赖上一批的 grammar 状态（bitmask），不能在 forward 内立即采样，于是把采样封进闭包延后执行。

worker 端（`tp_worker.py:613-626`）的门控三条同时成立才生效：`enable_overlap and not enable_spec and forward_batch.sampling_info.grammars is not None`：

```python
if (self.enable_overlap and not self.enable_spec
        and forward_batch.sampling_info.grammars is not None):
    def sample_batch_func():
        batch_result.next_token_ids = self.model_runner.sample(logits_output, forward_batch)
        return batch_result
    batch_result.delay_sample_func = sample_batch_func
    return batch_result   # 先返回，不采样
```

> 投机路径不走这里：spec worker 有自己的 `on_publish`（§11），verify 时 `is_verify=True` 直接返回、跳过采样（`tp_worker.py:609`）。

scheduler 在主循环里、处理完上一批之后再执行采样（`launch_batch_sample_if_needed`，`scheduler.py:3870`）：

```python
def launch_batch_sample_if_needed(self, batch_result, cur_batch):
    if batch_result is None or batch_result.delay_sample_func is None:
        return
    with self.forward_stream_ctx:
        self.forward_stream.wait_stream(self.schedule_stream)
        _batch_result = batch_result.delay_sample_func()          # 真正采样
        # 非 spec，只中继采样出的 bonus token
        self._relay_forward_payload(batch_result.future_indices, batch_result)   # scheduler.py:3855
        batch_result.copy_to_cpu(
            return_logprob=cur_batch.return_logprob,
            return_hidden_states=cur_batch.return_hidden_states)
    batch_result.delay_sample_func = None
    # 释放闭包持有的大显存张量：闭包捕获了 forward_batch（含 vocab_mask 的 sampling_info）
    # 与 logits_output（含 next_token_logits），不清会随 result_queue/batch_record_buf
    # 存活到下一步，结构化输出场景下稳定漏显存
    if batch_result.logits_output is not None:
        batch_result.logits_output.next_token_logits = None
```

关键变化（对齐当前 main）：
- 写回不再直接调 `future_map.stash`，改经 `_relay_forward_payload`（`scheduler.py:3855`）统一封装成 `RelayPayload`（与 §7 同一条中继路径）。非 spec 分支里它就是 `RelayPayload(bonus_tokens=next_token_ids)`。
- `return_logprob / return_hidden_states` 从传入的 `cur_batch` 读，不再读 `self.cur_batch`（遵循「传所需值而非 god object」的代码规范）。

在 `event_loop_overlap` 里，这一步紧跟在 `pop_and_process()`（处理上一批）之后调用（`scheduler.py:1805`，签名 `launch_batch_sample_if_needed(batch_result, batch)`），保证采样能看到上一批更新后的 grammar mask。

---

## 13. 与 PD 分离的交互

PD（Prefill-Decode）分离同样有重叠版本。`dispatch_event_loop`（`scheduler.py:4877`，模块级函数）按分离模式选择循环：

```python
if disaggregation_mode == DisaggregationMode.NULL:
    if scheduler.enable_pdmux:            scheduler.event_loop_pdmux()
    elif server_args.pp_size > 1:         scheduler.event_loop_pp()
    elif scheduler.enable_overlap_mlx:    scheduler.event_loop_overlap_mlx()
    elif scheduler.enable_overlap:        scheduler.event_loop_overlap()       # ← 标准重叠
    else:                                 scheduler.event_loop_normal()
elif disaggregation_mode == DisaggregationMode.PREFILL:
    if server_args.pp_size > 1:           scheduler.event_loop_pp_disagg_prefill()
    elif scheduler.enable_overlap:        scheduler.event_loop_overlap_disagg_prefill()  # prefill 重叠
    else:                                 scheduler.event_loop_normal_disagg_prefill()
elif disaggregation_mode == DisaggregationMode.DECODE:
    if server_args.pp_size > 1:           scheduler.event_loop_pp_disagg_decode()
    elif scheduler.enable_overlap:        scheduler.event_loop_overlap_disagg_decode()   # decode 重叠
    else:                                 scheduler.event_loop_normal_disagg_decode()
```

三个重叠循环结构与标准 `event_loop_overlap` 一致（发射当前 → WAR 栅栏 → 入队 `batch.copy()` → pop 处理上一批 → 延迟采样），并已同步吸收了主循环的两项新机制：`plan` 返回值（`get_next_disagg_decode_batch_to_run` 也返回 `running_batch`+`batch_to_run`）和 `_apply_war_barrier()`。以 decode 重叠为例（`disaggregation/decode.py:2181`）：

```python
def event_loop_overlap_disagg_decode(self):
    self.result_queue = deque()
    while True:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused: continue
        self.process_decode_queue()                  # PD 特有：处理 KV transfer 队列

        plan = self.get_next_disagg_decode_batch_to_run(running_batch=self.running_batch)
        self.running_batch = plan.running_batch
        batch = plan.batch_to_run
        disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch, last_batch=self.last_batch)
        if disable_overlap_for_batch and self.last_batch:
            pop_and_process()

        if batch:
            batch_result = self.run_batch(batch)
            self._apply_war_barrier()                # ← 与标准重叠同款 WAR 栅栏
            self.result_queue.append((batch.copy(), batch_result))
        else:
            batch_result = None
        if self.last_batch:
            if not disable_overlap_for_batch:
                pop_and_process()
        elif batch is None:
            self.on_idle()
        self.launch_batch_sample_if_needed(batch_result, batch)   # ← 延迟采样同款签名
        self.last_batch = batch
```

prefill 重叠（`disaggregation/prefill.py:607`）额外在循环里处理 bootstrap 队列与 inflight 队列，同样以 `launch_batch_sample_if_needed(batch_result, batch)`（`prefill.py:653`）收尾，并共用 `copy_done.synchronize()` 契约。

> 注意：`enable_pdmux`（同卡 PD 复用）与重叠**互斥**（`server_args.py:8933` 断言），它走独立的 `event_loop_pdmux()`。

---

## 14. 历史演进

> 这一节帮助理解老博客 / 老 PR / 老代码里提到的 `TpModelWorkerClient`、`forward_thread`、`future_token_ids_ct` 等概念。**这些已从当前代码删除。**

早期（PR #11210「Remove overlap thread」、#11300「Remove sampling info events and overlap thread file」之前），重叠靠**后台线程**实现：`python/sglang/srt/managers/tp_worker_overlap_thread.py` 里的 `TpModelWorkerClient` 包住同步的 `TpModelWorker`，用 `forward_thread` 后台线程 + `input_queue` / `output_queue` 传递工作；`forward_batch_generation` 立即返回占位的 `future_next_token_ids`，结果在 `resolve_last_batch_result` 里从 `output_queue` 取出。FutureMap 当时用环形计数器：

```python
# 旧实现（已删除）
class FutureMap:
    def __init__(self, max_running_requests, device):
        self.future_ct = 0
        self.future_limit = max_running_requests * 3      # 防环形碰撞
        self.token_ids_buf = torch.empty((max_running_requests * 5,), ...)
    def update_next_future(self, future_ct, bs):
        return torch.arange(-(future_ct + 1), -(future_ct + 1 + bs), -1, ...)  # 负数占位符来源
```

### 新旧对照

| 关注点 | 旧（后台线程，已删除） | 新（双 Stream） |
|---|---|---|
| 并发驱动 | `TpModelWorkerClient` 后台 `threading.Thread` | 单线程 + 两条 CUDA Stream |
| 工作传递 | `input_queue` / `output_queue`（`queue.Queue`） | `result_queue: deque` + CUDA stream 顺序 |
| `forward_batch_generation` | 立即返回占位 token | 在 `forward_stream` 上同步跑，返回 GPU 张量但 CPU 不同步 |
| Future buffer 索引 | 环形计数器 `future_ct % future_limit` | `req_pool_indices`（每槽一 slot） |
| 占位符 | `torch.arange(-(future_ct+1), ...)` | **无占位符**；步间 `batch.input_ids = None`，下一步在 forward 入口物化 |
| 解析输入 | 线程内 `future_map.resolve_future(...)` | 模块级 `resolve_forward_inputs(batch, future_map)`（`overlap_utils.py:84`） |
| 存结果 | `future_map.store_to_map(...)` | `_relay_forward_payload`→`stash(...)` / `publish(...)` |
| 输出拷贝 | 线程内拷贝后压 `output_queue` | `copy_to_cpu()` + `copy_done` event |
| 解析输出 | `resolve_last_batch_result()`（`output_queue.get`） | `process_batch_result_*()` → `copy_done.synchronize()` |
| 跨流张量保活 | `batch_lists[batch_pt % 2]` | `batch_record_buf[2]` + `extra_keep_alive_refs` |

**为什么改**：双 Stream 方案去掉了线程同步/GIL 争用、队列拷贝开销，并用 `req_pool_idx` 直接索引消除了环形计数器的碰撞窗口，使逻辑更简单、与 CUDA graph / spec v2 更易协同。

> 验证：当前仓库 `git grep TpModelWorkerClient -- '*.py'` 无任何匹配；`tp_worker_overlap_thread.py` 不存在。

---

## 15. 调优与排错

### 相关环境变量 / 开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `--disable-overlap-schedule` | 关（即重叠开启） | 全局关闭重叠调度（调试用） |
| `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` | `False` | 连续 prefill 不重叠，改善 TTFT，略损吞吐（`environ.py:441`） |
| `SGLANG_ENABLE_WAR_BARRIER` | `False` | 强制开启 WAR 写后读栅栏；CUDA 上默认已开（`_war_barrier_enabled = is_cuda() or 此开关`，`scheduler.py:1682`；env 定义 `environ.py:445`） |
| `SGLANG_IS_IN_CI` | `False` | 开启 FutureMap 的 `_DEBUG_ASSERT`：gather 前断言非负、用后写毒值，捕捉“未 stash 就 gather”的 bug（`environ.py:295`，断言逻辑 `overlap_utils.py:75`） |
| `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY` | `0` | 每步做内存不变量自检（`environ.py:399`） |

### 典型现象与定位

| 现象 | 可能原因 | 排查方向 |
|---|---|---|
| 关掉重叠后吞吐明显下降 | 说明 CPU 调度本是瓶颈，重叠在起作用 | 正常；不要长期关重叠 |
| decode 输出错乱 / 串 token | `input_ids` 未被正确物化（FutureMap gather/stash 不匹配），或跨流张量被提前释放 | 设 `SGLANG_IS_IN_CI=true` 触发 `_DEBUG_ASSERT`（gather 前断言非负、用后写毒值）；检查 `record_batch_in_overlap` 是否覆盖到相关张量 |
| spec + grammar 下行为 | 支持 grammar 重叠的算法正常重叠；host-draft 类算法（`grammar_needs_sync()` 为真）会逐批塌缩 | 预期行为（`is_disable_overlap_for_batch` / §11） |
| 结构化输出场景显存缓增 | 延迟采样闭包未释放 `logits_output.next_token_logits` | 见 `launch_batch_sample_if_needed` 末尾的释放逻辑 |
| 启动报错“overlap 与 X 不兼容” | PP / pdmux / dLLM / MPS 等场景 | 见 §4 自动禁用表 |

### 性能心智模型

- 重叠的收益 ≈ `min(CPU_schedule_time, GPU_forward_time)`，前提是两者可重叠。decode 步越短、batch 越小、CPU 后处理越重（如开 logprob、流式、复杂采样），重叠收益越大。
- 重叠**不**减少单步 GPU 时间；它只填掉 CPU 造成的 GPU 空泡。若 GPU 已是瓶颈（大 batch、长序列），重叠收益有限，但也不会变慢。
- 一拍的流水线延迟意味着请求“结束判定”晚一步生效——这是重叠下 `req.finished()` 在结果处理阶段才补判的根因（§9）。

---

## 16. 关键文件/行号索引

| 关注点 | 路径:行 |
|---|---|
| `enable_overlap` 设定 | `scheduler.py:427` |
| `init_overlap`（future_map / 流 ctx / record buf） | `scheduler.py:1428` |
| `run_event_loop`（建 `schedule_stream`） | `scheduler.py:1648`（stream 于 1665） |
| `event_loop_normal` | `scheduler.py:1704` |
| `event_loop_overlap` | `scheduler.py:1739` |
| `_apply_war_barrier`（WAR 写后读栅栏） | `scheduler.py:1686` |
| `is_disable_overlap_for_batch` | `scheduler.py:1813` |
| `_advance_pending_grammar`（grammar 重叠推进） | `scheduler.py:1851` |
| `get_next_batch_to_run`（返回 `plan`） | `scheduler.py:3001` |
| `record_batch_in_overlap` | `scheduler.py:3552` |
| `_forward_isolation`（`overlap=` 参数区分两模式） | `scheduler.py:3568` |
| `run_batch`（重叠路径起于 3644，isolation ctx 3651） | `scheduler.py:3612` |
| `_relay_forward_payload`（封装 `RelayPayload` 后 stash） | `scheduler.py:3855` |
| `launch_batch_sample_if_needed`（延迟采样） | `scheduler.py:3870` |
| `process_batch_result`（统一入口） | `scheduler.py:3900` |
| `dispatch_event_loop`（模块级） | `scheduler.py:4877` |
| `forward_stream_ctx` / `copy_stream` 创建 | `scheduler.py:1471 / 1474` |
| `batch_record_buf`（2 槽环形） | `scheduler.py:1482` |
| `decide_needs_cpu_seq_lens` | `overlap_utils.py:22` |
| `decide_needs_confidence_relay` | `overlap_utils.py:51` |
| `resolve_forward_inputs`（模块级，物化 input_ids） | `overlap_utils.py:84` |
| `RelayPayload`（中继载荷） | `overlap_utils.py:129` |
| `ConfidenceRelay`（DSpark，depth3/lag2） | `overlap_utils.py:153` |
| `FutureMap` 类 | `overlap_utils.py:232` |
| `_resolve_spec_extras`（取 draft 输入） | `overlap_utils.py:363` |
| `resolve_seq_lens_cpu`（spec 懒加载） | `overlap_utils.py:412` |
| `publish`（含 `publish_ready` 链式 record） | `overlap_utils.py:470` |
| `stash`（写中继 buffer） | `overlap_utils.py:501` |
| `GenerationBatchResult` + `copy_to_cpu` | `managers/utils.py:44 / 118` |
| `copy_done.synchronize()`（prefill/idle/decode） | `batch_result_processor.py:203 / 799 / 811` |
| `advance_grammar_fsm`（grammar 前进，自 gate） | `batch_result_processor.py:735`（sync 于 755） |
| `_resolve_spec_v2_tokens`（按 accept_lens 切分） | `batch_result_processor.py:627` |
| `forward_stream` 创建 | `model_executor/model_runner.py:403` |
| `war_fastpath_read_done_event` 初始化 | `model_executor/model_runner.py:408` |
| `create_future_map`（按算法建 FutureMap） | `speculative/spec_info.py:161` |
| `supports_grammar_overlap` / `grammar_needs_sync` | `speculative/spec_info.py:142 / …` |
| spec v2 `forward_batch_generation`（`on_publish`/`grammar_barrier` kwargs） | `speculative/eagle_worker_v2.py:1100` |
| spec `on_publish` 触发点（verify 后） | `eagle_worker_v2.py:1118 / 1178` |
| decode `seq_lens/seq_lens_cpu + 1`（out-of-place rebind） | `schedule_batch.py:3048-3050` |
| spec 判定（改用 `not spec_algorithm.is_none()`，旧 `is_spec_v2` 已删） | `scheduler.py:3586` 等 |
| disagg decode 重叠循环 | `disaggregation/decode.py:2181` |
| disagg prefill 重叠循环 | `disaggregation/prefill.py:607` |
| `disable_overlap_schedule` 字段/默认（CLI 由注解自动生成） | `server_args.py:954`（默认 False） |
| 历史删除（仅 git history） | `git show 501dfa6b4~1:python/sglang/srt/managers/tp_worker_overlap_thread.py` |

---

## 附：一句话总结

> SGLang 的重叠调度用**一条 `forward_stream` 把 GPU 计算与 CPU 调度并行**，用 **`FutureMap` + 跨步中继 buffer**（`input_ids` 延迟到 forward 入口由 `resolve_forward_inputs` 物化，步间 `batch.input_ids=None`）打破“下一步输入依赖上一步输出”的数据依赖，用 **`result_queue` 推迟一拍处理结果**让 CPU 永远只在早已完成的上一迭代上同步，用 **`_apply_war_barrier` 写后读栅栏 + `batch_record_buf` 钉住跨流张量**保证正确性。它默认开启，是 SGLang 高吞吐的关键基础设施之一。
