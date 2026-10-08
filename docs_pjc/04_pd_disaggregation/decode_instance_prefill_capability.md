# SGLang Decode 实例能否承担 Prefill 任务？——技术限制与支持方案

> 代码基线：`main` @ `33ed29a0ee`
> 关键文件：`python/sglang/srt/disaggregation/decode.py`、`python/sglang/srt/managers/scheduler.py`、`python/sglang/srt/arg_groups/*`
> 相关文档：[mix_vs_prefill_itps.md](mix_vs_prefill_itps.md)、[pd_disaggregation_architecture.md](pd_disaggregation_architecture.md)

---

## 零、一句话结论

**现状：不能。** 用 `--disaggregation-mode decode` 启动的实例，在当前代码里**永远不会执行一次 prompt 的 extend（prefill）forward**。

但要特别说明一点，这个"不能"**不是**硬件、算子或注意力后端的限制——decode 实例的模型权重、注意力后端、KV 池结构都完全具备算 prefill 的能力（甚至 `capture_prefill_graph` 在 decode 模式下也只是退化成 eager runner，而不是报错）。真正的阻塞只有**两处结构性的**：

| # | 阻塞点 | 位置 | 性质 |
|---|---|---|---|
| 1 | decode 的事件循环里**根本没有** `get_new_batch_prefill` 的调用 | [decode.py:2560](../../../python/sglang/srt/disaggregation/decode.py#L2560) | 结构性 |
| 2 | 请求进不了 `waiting_queue`，被强制塞进 `disagg_decode_prealloc_queue` | [scheduler.py:2912](../../../python/sglang/srt/managers/scheduler.py#L2912) | 结构性 |
| 3 | 没有 `bootstrap_room` 的请求直接 HTTP 400 | [scheduler.py:2573](../../../python/sglang/srt/managers/scheduler.py#L2573) | 准入校验 |

其余几十处差异（CUDA graph、radix cache、显存预留、池子形状）都是**"按 decode 形态裁剪过的配置"**，属于性能/容量问题，不是正确性壁垒。

所以问题的正确提法不是"能不能做到"，而是**"值不值得做，以及做成什么形态"**。第五、六章给出四种方案的对比和推荐路径。

---

## 一、先搞清楚：decode 实例现在到底在做什么

### 1.1 三种角色，三套完全独立的事件循环

SGLang 用 `disaggregation_mode` 这一个枚举把实例分成三种角色（[utils.py:101](../../../python/sglang/srt/disaggregation/utils.py#L101)）：

```python
class DisaggregationMode(Enum):
    NULL    = "null"      # 普通实例：prefill + decode 全干（也就是俗称的 mix 实例）
    PREFILL = "prefill"   # PD 分离的 P 实例
    DECODE  = "decode"    # PD 分离的 D 实例
```

启动时在 [scheduler.py:5294](../../../python/sglang/srt/managers/scheduler.py#L5294) 的 `dispatch_event_loop` 里一次性分派，之后**再也不会切换**：

```python
def dispatch_event_loop(scheduler: Scheduler):
    disaggregation_mode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        ... scheduler.event_loop_overlap() / event_loop_normal() ...
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        ... scheduler.event_loop_overlap_disagg_prefill() ...
    elif disaggregation_mode == DisaggregationMode.DECODE:
        ... scheduler.event_loop_overlap_disagg_decode() ...
```

这是理解整个问题的钥匙：**三套循环是三个独立函数，不是一套循环里的三个 if 分支。** decode 循环没写 prefill 的代码，所以 decode 实例做不了 prefill——就这么直白。

对照一下两套循环的骨架：

| 步骤 | NULL（mix）循环 [scheduler.py:1797](../../../python/sglang/srt/managers/scheduler.py#L1797) | DECODE 循环 [decode.py:2450](../../../python/sglang/srt/disaggregation/decode.py#L2450) |
|---|---|---|
| 1 | — | `prefetch_prefill_dp_rank_queries()` 预取 P 侧 DP rank |
| 2 | `recv_requests()` + `process_input_requests()` | 同 |
| 3 | — | **`process_decode_queue()`**（推进 prealloc→transfer→waiting 三级队列） |
| 4 | **`get_next_batch_to_run(running_batch, last_batch)`**<br>内部会调 `get_new_batch_prefill()` | **`get_next_disagg_decode_batch_to_run(running_batch)`**<br>内部只调 `get_new_prebuilt_batch()` |
| 5 | `run_batch()` → 真的跑 forward | `run_batch()`，但 PREBUILT 批被短路 |
| 6 | `process_batch_result()` | 同 |

`get_new_batch_prefill` 全仓库只有 4 个调用点：`scheduler.py:3288`（NULL 的 `get_next_batch_to_run` 内）、`prefill.py:584`、`scheduler_pp_mixin.py:265`（NULL 的 PP 循环）、`multiplex/multiplexing_mixin.py:89`（PDMux）。**没有一个在 decode 路径上。**

### 1.2 decode 实例的请求生命周期：四级队列 + 一个"假 prefill"

```
HTTP 请求（必须带 bootstrap_host/port/room）
   │
   ▼
process_input_requests → _add_request_to_queue          scheduler.py:2897
   │  (DECODE 分支：不进 waiting_queue！)
   ▼
① DecodePreallocQueue                                   decode.py:316
   · 与 P 侧握手（bootstrap server 查询 P 的 DP rank / TP 布局）
   · 按 max_new_tokens 预留 KV slot（pop_preallocated, decode.py:1080）
   · 可选：decode 侧 radix 前缀命中并 inc_lock_ref
   ▼
② DecodeTransferQueue                                   decode.py:2015
   · 轮询 KVPoll 状态，等 P 侧把 KV 通过 RDMA 写进本地 KV 池
   · Success 时 _commit_transfer_to_req (decode.py:2059) 从 metadata buffer 读出：
       → committed_output_id  ← ★P 侧已经采样好的「第一个 token」
       → cached_tokens / logprobs / hidden_states
     并执行 req.output_ids.append(committed_output_id)   decode.py:2147
   ▼
③ Scheduler.waiting_queue                               decode.py:2701
   · 这是 waiting_queue 在 decode 实例上「唯一的」入口
   ▼
④ get_new_prebuilt_batch → ForwardMode.PREBUILT          decode.py:2592
   · prepare_for_prebuilt()   伪造「好像刚 prefill 完」的 batch 元数据
   · process_prebuilt()       把第一个 token 塞进 FutureMap 中继
   · run_batch() 在 scheduler.py:3894 直接短路，不进模型
   ▼
   merge 进 running_batch → update_running_batch → ForwardMode.DECODE
```

### 1.3 `ForwardMode.PREBUILT`：名字里有 extend，实际零 forward

这是最容易误解的地方。`PREBUILT` 定义在 [forward_batch_info.py:122](../../../python/sglang/srt/model_executor/forward_batch_info.py#L122)，注释写着 "Used in disaggregated decode worker"。历史上它叫 "prebuilt extend"，代码注释里这个叫法还残留着，但在当前代码里它**不跑任何 forward**：

```python
# scheduler.py:3893
# Place holder handling for pd-disagg decode event loop
if batch.forward_mode.is_prebuilt():
    return self._run_batch_prebuilt(batch)
```

而 `_run_batch_prebuilt`（[decode.py:2548](../../../python/sglang/srt/disaggregation/decode.py#L2548)）只返回一个空结果：

```python
def _run_batch_prebuilt(self, batch):
    if batch.inner_idle_batch is not None:      # 仅 DP attention 需要
        idle_batch = batch.inner_idle_batch
        batch.inner_idle_batch = None
        return self.run_batch(idle_batch)       # 跑一个 IDLE 批做 MLP 同步
    return GenerationBatchResult()              # 空
```

唯一可能触发 GPU 计算的情况是 DP attention 下为了对齐各 DP rank 的集合通信，挂一个 `IDLE` 批（[dp_attn.py:477](../../../python/sglang/srt/managers/scheduler_components/dp_attn.py#L477)）。

`prepare_for_prebuilt`（[decode_schedule_batch_mixin.py:23](../../../python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py#L23)）的注释把设计意图说得很清楚：

```python
self.forward_mode = ForwardMode.PREBUILT
# PREBUILT never enters a model forward. Keep the legacy scalar metadata,
# but do not flatten and copy every transferred prompt to the GPU only to
# discard it before the first decode step.
...
self.input_ids = None    # 第一个 token 来自中继，不来自打平的 prompt
```

它做的事就是**伪造** `seq_lens` / `prefix_lens` / `extend_lens` / `out_cache_loc` / `sampling_info`，让这批请求"看起来"刚刚完成了 prefill，然后直接进 DECODE 步。

**关键推论：decode 实例的第一个输出 token 是 P 侧算好、通过 metadata buffer 传过来的**，不是本地算的。这一点在后面设计方案时是个硬约束（见 4.4）。

### 1.4 decode 实例上可能出现的 ForwardMode 全集

| ForwardMode | decode 实例上会出现吗 | 说明 |
|---|---|---|
| `PREBUILT` | ✅ | 零 forward 的占位批 |
| `DECODE` | ✅ | 主力 |
| `IDLE` | ✅ | DP attention 填充 / inner idle batch |
| `TARGET_VERIFY` | ✅（开投机时） | 注意 `is_extend()` 对它返回 True |
| `DRAFT_EXTEND_V2` | ✅（开投机时） | draft 模型的 extend |
| `EXTEND` | ❌ | **没有任何代码路径会设置它** |
| `MIXED` | ❌ | chunked prefill 专用 |
| `SPLIT_PREFILL` | ❌ | PDMux 专用，且 PDMux 要求 mode=null |

有意思的是：`TARGET_VERIFY` 和 `DRAFT_EXTEND_V2` 证明 **decode 实例本身完全有能力跑"多 token 一次前向"的 extend 形态计算**——投机解码的 verify 就是一次小 extend。所以说 decode 实例"算不了 prefill"是不对的，它只是"没人叫它算"。

---

## 二、结构性限制（必须改代码才能绕过）

这三条是真正的"墙"，改配置、改环境变量都没用。

### 2.1 调度器里没有 prefill 路径

[decode.py:2560](../../../python/sglang/srt/disaggregation/decode.py#L2560) 是 decode 实例整个调度决策的全部：

```python
def get_next_disagg_decode_batch_to_run(self, running_batch) -> NextBatchPlan:
    # 1) 处理上一轮的 prebuilt 批（出第一个 token + filter + merge）
    new_prebuilt_batch = self.get_new_prebuilt_batch(running_batch)
    if new_prebuilt_batch:
        assert self.chunked_req is None                       # ← 明确断言：不存在 chunked prefill
        self.batch_result_processor.process_batch_result_prebuilt(new_prebuilt_batch)
        new_prebuilt_batch.filter_batch()
        if not new_prebuilt_batch.is_empty():
            running_batch = new_prebuilt_batch if running_batch.is_empty() \
                            else (running_batch.merge_batch(new_prebuilt_batch) or running_batch)
    # 2) 跑 decode
    if running_batch.is_empty():
        ret = None
    else:
        running_batch = self.update_running_batch(running_batch)   # → prepare_for_decode()
        ret = running_batch if not running_batch.is_empty() else None
    ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
    return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)
```

和 NULL 模式的 `get_next_batch_to_run` 相比，缺失的能力有：

- 没有 `get_new_batch_prefill()` 调用 → 不会构造 EXTEND 批
- 没有 chunked prefill 的 `chunked_req` 状态机（还有 `assert self.chunked_req is None` 主动禁止）
- 没有 `last_batch` 的 extend-merge 逻辑
- 没有 `maybe_convert_decode_to_extend()`（NULL 模式在 [scheduler.py:3321](../../../python/sglang/srt/managers/scheduler.py#L3321)）
- `get_new_prebuilt_batch` 的批大小上限只按**请求条数**算（`min(req_to_token_pool.size, max_running_requests)`，[decode.py:2609](../../../python/sglang/srt/disaggregation/decode.py#L2609)），**没有 token 预算概念**——因为 KV 早就 prealloc 好了。这套准入逻辑没法直接给 prefill 用（prefill 必须按 token 预算限流）。

### 2.2 请求进不了 `waiting_queue`

[scheduler.py:2897](../../../python/sglang/srt/managers/scheduler.py#L2897)：

```python
def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
    if not self._set_or_validate_priority(req):
        return
    if self.disaggregation_mode == DisaggregationMode.NULL:
        if self._abort_on_queued_limit(req): return
        self._prefetch_kvcache(req)
        self.waiting_queue.append(req)                                   # ← prefill 的入口
    elif self.disaggregation_mode == DisaggregationMode.PREFILL:
        self._prefetch_kvcache(req)
        self.disagg_prefill_bootstrap_queue.add(req, self.model_config.num_key_value_heads)
    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)  # ← 只有这条路
    else:
        raise ValueError(f"Invalid {self.disaggregation_mode=}")
```

在 decode 实例上，`waiting_queue` 的**唯一写入点**是 `process_decode_queue` 末尾的 `self.waiting_queue.extend(transferred_reqs)`（[decode.py:2701](../../../python/sglang/srt/disaggregation/decode.py#L2701)），也就是"KV 已经从 P 侧传完的请求"。一个没有 P 侧来源的新请求，物理上到不了 `waiting_queue`。

顺带注意：`_prefetch_kvcache`（HiCache 预取）在 DECODE 分支里**也没有调用**。

### 2.3 没有 `bootstrap_room` 直接 400

[scheduler.py:2571](../../../python/sglang/srt/managers/scheduler.py#L2571)：

```python
if self.disaggregation_mode != DisaggregationMode.NULL:
    if recv_req.bootstrap_room is None and self.transfer_backend != TransferBackend.FAKE:
        error_msg = (f"Invalid request: Disaggregated request received without "
                     f"bootstrap room id. {req.rid=}")
        ... prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
```

所以哪怕内部改通了调度，普通的 `/v1/chat/completions` 请求打到 decode 实例上仍然会被拒。这也是为什么 `/health_generate` 和 warmup 必须注入假的 bootstrap 信息（[http_server.py:704](../../../python/sglang/srt/entrypoints/http_server.py#L704)、[warmup.py:97](../../../python/sglang/srt/entrypoints/warmup.py#L97)）：

```python
# http_server.py:704
bootstrap_host = FAKE_BOOTSTRAP_HOST
bootstrap_room = 0
```

### 2.4 附带的：没有 bootstrap server（只影响"decode 当 P 用"的场景）

[disagg_service.py:23](../../../python/sglang/srt/managers/disagg_service.py#L23)：

```python
if disagg_mode == DisaggregationMode.PREFILL:
    # only start bootstrap server on prefill tm
    bootstrap_server = kv_bootstrap_server_class(host=..., port=...)
    return bootstrap_server
```

同理 `disagg_prefill_bootstrap_queue` / `disagg_prefill_inflight_queue` 在 decode 实例上恒为 `None`（[scheduler.py:1355](../../../python/sglang/srt/managers/scheduler.py#L1355)），KVSender 侧的一整套也不存在。

**这一条要分清适用范围：**

- 如果目标是"decode 实例本地 prefill 完，本地接着 decode"（**自给自足**）→ 不需要 bootstrap server，这条不构成限制。
- 如果目标是"decode 实例帮别的 D 实例做 prefill 并把 KV 传出去"（**当 P 用**）→ 需要补齐 bootstrap server + KVSender + inflight queue，工作量翻倍。

---

## 三、能力被裁剪的限制（改配置可恢复，但要付代价）

这一类的共同特征是：为了 decode 场景省显存/省启动时间，代码主动关掉了 prefill 需要的东西。**它们不会让 prefill 报错，只会让它慢或者 OOM。**

### 3.1 prefill CUDA graph 被强制关闭

[cuda_graph_hook.py:580](../../../python/sglang/srt/arg_groups/cuda_graph_hook.py#L580)：

```python
elif cfg.disaggregation_mode == "decode":
    if (Phase.PREFILL, "backend") not in server_args._cuda_graph_config_locked:
        declare_resolution(server_args, "_apply_cuda_graph_disaggregation_roles",
            cuda_graph_config=with_phase(cfg.cuda_graph_config, Phase.PREFILL,
                                         backend=Backend.DISABLED))
```

注意两点：

1. **有逃生门**：判断条件是 `not in _cuda_graph_config_locked`，也就是说如果用户显式指定了 prefill phase 的 graph backend，这个 override 就不生效。
2. **关掉不等于不能跑**：`capture_prefill_graph`（[cuda_graph_setup.py:284](../../../python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py#L284)）在 backend 为 DISABLED 时返回的是 `EagerRunner`，而不是抛异常。**所以 decode 实例上的 EXTEND forward 是能跑的，只是走 eager，没有 graph 加速。**

对 prefill 而言 eager 的损失不算致命（prefill 是大 GEMM，kernel launch 开销占比低），但小 batch 短 prompt 场景会明显掉。

配套被跳过的还有：

- [overrides.py:1773](../../../python/sglang/srt/arg_groups/overrides.py#L1773) `post_capture_kv_sizing_planned`：decode 模式跳过 prefill-graph 的前置校验（`chunked_prefill_size > 0`、buffer token 是否被 captured bs 覆盖）
- [base_runner.py:425](../../../python/sglang/srt/model_executor/runner/base_runner.py#L425)、[flashinfer_autotune.py:350](../../../python/sglang/srt/model_executor/runner/flashinfer_autotune.py#L350)：warmup / autotune 的 EXTEND 形状不会被预热
- [compile_utils.py:503](../../../python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py#L503) DeepGEMM JIT 预编译把这个分工写得最直白：

  ```python
  run_decode = is_generation and disagg_mode != "prefill"
  run_extend = disagg_mode != "decode"        # ← decode 实例不预编译 extend 形状的 GEMM
  ```

  后果：decode 实例第一次跑 prefill 会触发 DeepGEMM 运行时 JIT，首次请求出现秒级卡顿。

### 3.2 Radix Cache 被强制关成 Chunk Cache

[pd_disaggregation_hook.py:101](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L101)：

```python
# 默认（未开 --disaggregation-decode-enable-radix-cache）
declare_resolution(server_args, "handle_pd_disaggregation", disable_radix_cache=True)
logger.warning("KV cache is forced as chunk cache for decode server")
```

对纯 decode 没损失（decode 不做前缀匹配），但对 prefill 是**致命的性能损失**——前缀复用是 prefill 侧最大的优化。

开 `--disaggregation-decode-enable-radix-cache` 可以打开，但它自带一串互斥约束（[pd_disaggregation_hook.py:77](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L77)）：

- 与 `--enable-hisparse` 互斥
- 与 `fake` transfer backend 互斥
- **与任何 `--speculative-algorithm` 互斥**
- 与 `dcp_size > 1` 互斥（[pd_disaggregation_hook.py:53](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L53)）
- SWA 模型还要求 `SGLANG_ENABLE_UNIFIED_RADIX_TREE`，且禁 hierarchical cache、禁 DeepSeek-V4（[kv_cache_builder.py:252](../../../python/sglang/srt/mem_cache/kv_cache_builder.py#L252)）
- 而且标注是 **EXPERIMENTAL**

另外 [schedule_policy.py:249](../../../python/sglang/srt/managers/schedule_policy.py#L249) 里有一处直接的注释：

```python
# Skip on decode (never prefills).
```

即"与 cache 无关的前缀匹配回填"在 decode 模式下被跳过。

### 3.3 显存预算按 decode 形态算

[memory_hook.py:239](../../../python/sglang/srt/arg_groups/memory_hook.py#L239)：decode 模式下，激活显存的工作集是按 `max_running_requests * speculative_num_draft_tokens`（下限 2048）算的，而不是按 `chunked_prefill_size` / `max_prefill_tokens`。

也就是说：**decode 实例的激活显存只够放几千个 token 的前向，不够放一个大 chunk 的 prefill。** 剩下的显存全被 KV 池吃掉了（这正是 D 实例的设计目标：KV 池越大 → 并发越高）。

配套差异：

| 位置 | decode 模式的行为 |
|---|---|
| [memory_hook.py:275](../../../python/sglang/srt/arg_groups/memory_hook.py#L275) | **VLM 的视觉编码器显存 headroom 不预留** → decode 实例跑多模态 prefill 会 OOM |
| [memory_hook.py:299](../../../python/sglang/srt/arg_groups/memory_hook.py#L299) | `reserve_for_graph_mb` 只留 decode graph 的量 |
| [memory_hook.py:343](../../../python/sglang/srt/arg_groups/memory_hook.py#L343) | DeepEP a2a buffer 只按 decode graph 留 |
| [overrides.py:324](../../../python/sglang/srt/arg_groups/overrides.py#L324) | `pre_capture_activation_reserve_mb` 同样的 decode/prefill 二分 |

### 3.4 各种"prefill 才需要"的校验被跳过

这些校验被跳过意味着：**如果强行在 decode 实例上跑 prefill，配置里的非法组合不会在启动时被拦下，而是在运行时炸。**

| 位置 | 被跳过的校验 |
|---|---|
| [validation_hook.py:104](../../../python/sglang/srt/arg_groups/validation_hook.py#L104) | `chunked_prefill_size % page_size == 0` 断言 |
| [moe_hook.py:304/352/463](../../../python/sglang/srt/arg_groups/moe_hook.py#L304) | MoRI / pplx / flashinfer-cutedsl 的**每 rank dispatch token 预算校验** |
| [moe_hook.py:379](../../../python/sglang/srt/arg_groups/moe_hook.py#L379) | DeepEP v2 dispatch token 预算校验 |
| [deepseek_v4_hook.py:21](../../../python/sglang/srt/arg_groups/deepseek_v4_hook.py#L21) | MegaMoE 的 prefill 预算校验（注释："decode bs is not relevant with --chunk-prefill-size"） |

MoE 的这一条尤其危险：一次大 prefill 的 token 数远超 decode，**EP dispatch buffer 可能直接不够，表现为非法内存访问或 hang**。

### 3.5 注意力后端的 prefill 优化被关

[flashinfer_mla_backend.py:243](../../../python/sglang/srt/layers/attention/flashinfer_mla_backend.py#L243)：

```python
self.enable_chunk_kv = (
    not skip_prefill
    and get_disagg().disaggregation_mode != "decode"      # ← decode 实例直接关掉
    and not get_schedule().disable_chunked_prefix_cache
    and not get_exec().kernel.flashinfer_mla_disable_ragged
)
```

也就是 **chunked prefix cache / ragged prefill 在 decode 实例上不可用**——这是 MLA 模型（DeepSeek 系）prefill 的核心优化之一。

反过来也有 decode 独占的：

- [dsa/utils.py:78](../../../python/sglang/srt/layers/attention/dsa/utils.py#L78) `should_remap_pd_dsa_seed_to_local_slots()` **要求** `disaggregation_mode == "decode"` —— DSA 的 seed 索引重映射是为"KV 从 P 传过来、slot 编号要换成本地"设计的。本地 prefill 出来的 KV 不需要重映射，混在一起会**索引错乱**。
- [kda_backend.py:475](../../../python/sglang/srt/layers/attention/linear/kda_backend.py#L475) fused-accept 快路在任何非 null 模式下返回 False（因为 PD 握手会传 `temporal` 状态）。

### 3.6 显存池 / slot 形状假设

| 位置 | decode 模式的形状 | 对 prefill 的影响 |
|---|---|---|
| [kv_cache_configurator.py:938](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L938) | `req_to_token_pool` 多分配 `disaggregation_decode_extra_slots` 个槽位（默认 `2 * max_running_requests // dp_size`，小 batch 时） | 槽位够，不是问题 |
| [pool_configurator.py:682](../../../python/sglang/srt/model_executor/pool_configurator.py#L682) | SWA 上限公式是 `per_request * num_reqs + (window + page) * extra_slots`；非 decode 才有 `chunks_in_flight * chunked_prefill_size` 项 | **SWA 模型上本地 prefill 会撑爆 SWA 池** |
| [kv_cache_configurator.py:661](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L661) | 分配器 `need_sort=True` | 有性能开销，非正确性问题 |
| [kv_cache_configurator.py:96](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L96) | DSA `index_k` elision 要求 mode==null | 只影响显存占用 |

---

## 四、语义与正确性风险（方案设计时必须处理）

这一章是最容易踩坑的部分。前面两章是"做不到"和"做得慢"，这一章是"做了会错"。

### 4.1 第一个 token 的来源冲突

PD 分离的语义约定是：**P 侧完成 prefill 并采样出第一个 token，把它写进 metadata buffer 传给 D。**

[decode.py:2147](../../../python/sglang/srt/disaggregation/decode.py#L2147)：

```python
decode_req.req.output_ids.append(committed_output_id)
```

而 `prepare_for_prebuilt` 里有这个断言（[decode_schedule_batch_mixin.py:60](../../../python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py#L60)）：

```python
seq_len = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
if len(req.output_ids) == 0:
    assert seq_len - pre_len == req.extend_range.length
```

如果引入本地 prefill，这条请求走的是**另一条路**：本地 EXTEND forward 采样出第一个 token → 走 `process_batch_result_prefill` 的正常路径。它**不能**再进 `get_new_prebuilt_batch`，否则 `req.output_ids[-1]`（`process_prebuilt` 第 120 行读的）会拿到错的东西，或者 `cached_tokens` 被重复累加（[decode_schedule_batch_mixin.py:65-73](../../../python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py#L65-L73) 里那段 `already_computed` 的 clamp 逻辑就是为了防重复计数）。

**设计要求：本地 prefill 的请求必须完全绕开 PREBUILT 路径，走 NULL 模式的 extend → decode 流程。** 两类请求在 `running_batch` 里可以共存（都是 DECODE 形态），但准入路径必须分离。

### 4.2 logprob 的归属

[batch_result_processor.py:113](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L113) 的注释：

```python
# Logprobs should be handled on the prefill engine.
```

decode 实例的 `process_batch_result_prebuilt` 不算 input logprob，它是从 metadata buffer 里读 P 侧算好的。本地 prefill 的请求必须自己算 input logprob（`extend_logprob_start_lens` / `extend_input_logprob_token_ids` 这两个字段在 `prepare_for_prebuilt` 里被显式设成 `None`）。

### 4.3 Retraction 的两套备份机制

decode 实例独有的 retraction 备份路径（[schedule_batch.py:2016](../../../python/sglang/srt/managers/schedule_batch.py#L2016)）：

```python
# 只有 decode 模式在 pause_generation/retract 时会做 retraction_backup
```

- `--disaggregation-decode-retraction-backup=host_pool` **只允许 decode 模式**（[kv_cache_hook.py:144](../../../python/sglang/srt/arg_groups/kv_cache_hook.py#L144)）
- `--disaggregation-decode-enable-offload-kvcache` **只允许 decode 模式**（[kv_cache_hook.py:173](../../../python/sglang/srt/arg_groups/kv_cache_hook.py#L173)）
- 更彻底的 "true retraction + rebootstrap"：decode 被抢占的请求可以**退回 P 侧重新走一遍 bootstrap**（`pd_rebootstrap_forced_output_id`、`hold_rebootstrap`、`enqueue_held_rebootstrap`，[scheduler.py:5034](../../../python/sglang/srt/managers/scheduler.py#L5034)，详见 [pd_true_retraction_rebootstrap_architecture.md](pd_true_retraction_rebootstrap_architecture.md)）

**本地 prefill 的请求没有"P 侧"可以退回。** 一旦它被 retract，只能走 `cpu_tensor` / `host_pool` 备份或者本地重算。这是个必须显式处理的分支。

### 4.4 与投机解码的组合

`process_prebuilt` 会调 `spec_algorithm.build_disagg_draft_input`（[decode_schedule_batch_mixin.py:145](../../../python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py#L145)），分派到 `eagle_disaggregation` / `dspark_disaggregation`（[spec_info.py:187](../../../python/sglang/srt/speculative/spec_info.py#L187)）——**这套 draft 输入是从 P 侧传来的 hidden states 构造的**。

本地 prefill 的请求要用的是 NULL 模式的 draft extend 路径。两套 draft 输入构造逻辑不同，混跑需要在 batch 层面区分，或者干脆**一期不支持"本地 prefill + 投机解码"组合**。

再叠加 3.2 提到的"decode radix cache 与任何 speculative algorithm 互斥"，这块的组合爆炸相当严重。

### 4.5 DP attention 的集合通信对齐

`PREBUILT` 在 DP attention 下被当成 `IDLE` 处理（[dp_attn.py:271/362/477/489](../../../python/sglang/srt/managers/scheduler_components/dp_attn.py#L271)），因为它不跑 forward，但其他 DP rank 可能在跑 DECODE，必须挂一个 inner idle batch 做 MLP sync。

**如果某个 DP rank 上跑本地 prefill（EXTEND），而其他 rank 在跑 DECODE，`maybe_prepare_mlp_sync_batch` 必须能处理 EXTEND/DECODE 混合的 MLP sync。** NULL 模式本来就支持这个（因为 NULL 模式下各 rank 的 forward mode 天然不一致），所以这块理论上可复用，但要实测验证。

同时注意 `enable_tp_lm_head_all_to_all` 只在 decode 模式 + DP attention 下自动开启（[overrides.py:1418](../../../python/sglang/srt/arg_groups/overrides.py#L1418)），这个优化和 EXTEND 的 lm_head 形状是否兼容需要确认。

### 4.6 PP（Pipeline Parallel）的 proxy tensor 收发

[scheduler_pp_mixin.py:435](../../../python/sglang/srt/managers/scheduler_pp_mixin.py#L435)：

```python
if not cur_batch.forward_mode.is_prebuilt():
    pp_proxy_tensors = self._pp_recv_proxy_tensors()
```

PREBUILT 批跳过 PP 的 proxy tensor 收发（因为没有 forward）。混入 EXTEND 后 PP 各 stage 的收发次数必须严格一致，否则**直接 hang**。相关排查方法参见 skill `debug-distributed-hang`。

### 4.7 明确禁止的组合（启动就会报错）

| 特性 | 与 decode 模式的关系 | 位置 |
|---|---|---|
| `--enable-pdmux` | 要求 `mode == null` | [validation_hook.py:110](../../../python/sglang/srt/arg_groups/validation_hook.py#L110) |
| `--enable-prefill-cp` | 要求 `mode != decode` | [model_hook.py:220](../../../python/sglang/srt/arg_groups/model_hook.py#L220) |
| `--enable-dsa-cache-layer-split` | **显式拒绝 decode**，只支持 PD prefill | [model_hook.py:224](../../../python/sglang/srt/arg_groups/model_hook.py#L224) |
| `--enable-linear-replayssm` | 拒绝任何 PD 模式 | [attention_hook.py:358](../../../python/sglang/srt/arg_groups/attention_hook.py#L358) |
| DWDP | 要求 `null` 或 `prefill`，**decode 被禁** | [parallel_hook.py:253](../../../python/sglang/srt/arg_groups/parallel_hook.py#L253) |
| `--hicache-host-memory-mode buffer_only` | **decode 拒绝** | [hicache_hook.py:204](../../../python/sglang/srt/arg_groups/hicache_hook.py#L204) |
| `--language-model-only` | 要求 `null` | [model_hook.py:849](../../../python/sglang/srt/arg_groups/model_hook.py#L849) |
| `--enable-prefill-delayer` | decode 上被忽略（"no prefill scheduling path"） | [scheduler.py:1292](../../../python/sglang/srt/managers/scheduler.py#L1292) |
| DLLM | **静默强制** `disaggregation_mode="null"` | [dllm_hook.py:103](../../../python/sglang/srt/arg_groups/dllm_hook.py#L103) |

其中 `--enable-dsa-cache-layer-split` 和 DWDP 这两条特别值得注意：它们是**prefill 侧的重要优化，但在 decode 模式下明确被禁**。这意味着即使打通了 decode 本地 prefill，性能上也拿不到 P 实例的全部工具箱。

---

## 五、性能层面的根本矛盾：为什么当初要禁

即使把上面所有工程问题都解决了，还有一个绕不过去的物理矛盾。这也是 PD 分离存在的意义。

### 5.1 两种负载的特性完全相反

| 维度 | Prefill | Decode |
|---|---|---|
| 瓶颈 | **compute-bound**（大 GEMM，M 维 = batch 内 token 总数） | **memory-bandwidth-bound**（读 KV，GEMM 的 M 维 = batch size） |
| 单步耗时 | 几十 ~ 几百 ms（随 prompt 长度增长） | 几 ~ 几十 ms（相对恒定） |
| 显存诉求 | 大激活工作集，KV 池可以小 | KV 池越大越好，激活很小 |
| 优化目标 | ITPS（吞吐） | TPOT / ITL（尾延迟） |
| CUDA graph 收益 | 低（kernel 大，launch 开销占比小） | **高**（kernel 小而多） |

### 5.2 混跑的直接代价：ITL 抖动

decode 实例的核心 SLO 是 **TPOT / ITL 稳定**。一次长 prompt 的 prefill 会独占 SM 几十到几百毫秒，这段时间里 running batch 里**所有**正在解码的请求全部停摆：

```
理想 decode:  |D|D|D|D|D|D|D|D|D|   ITL 均匀 ~15ms
混入 prefill: |D|D|■■■■■■■■■■|D|D|  这一步 ITL 变成 200ms+
                   ↑ 一次 8K prompt 的 prefill
```

这是 NULL（mix）模式一直存在的问题，也正是 PD 分离要解决的第一个问题。**在 D 实例上重新引入 prefill，等于把 PD 分离的主要收益退回去。**

如果非要做，必须配套限流机制：
- 只接短 prompt（比如 `< chunked_prefill_size / 4`）
- 强制 chunked prefill，把单次 extend 的 token 数压到与一次 decode step 相当
- 或者用 PDMux 的 SM 分区思路（见 6.4）

### 5.3 间接代价：prefill 的 MFU 上不去

D 实例的显存分配策略是"KV 池吃满剩余显存"（这是 D 实例并发能力的来源）。留给激活的只有 `max_running_requests * spec_num_draft_tokens`（下限 2048）这个量级（[memory_hook.py:239](../../../python/sglang/srt/arg_groups/memory_hook.py#L239)）。

prefill 的 MFU 随一次 forward 的 token 总数单调上升。**D 实例只能给出 2K 左右的 token 预算，这个 M 维远达不到 roofline。** 也就是说：D 实例上做 prefill，既伤了 decode 的延迟，自己的吞吐也很差。

这一点在 [mix_vs_prefill_itps.md](mix_vs_prefill_itps.md) 里有更完整的量化分析（那篇讨论的是反方向：mix 实例模拟 P 实例的差距），核心结论同样适用：

```
可达 ITPS ≈ (峰值FLOPs × MFU) × (prefill 计算占 GPU 时间的比例)
```

D 实例在这两个乘子上**都**吃亏。

### 5.4 还失去了 P 侧的逐层流式传输

真 P 实例的 KV 写出是**逐层**发起 RDMA 的，与下一层的计算重叠，走 NIC / copy engine，不吃 SM。D 实例上做本地 prefill 不需要写出 KV（自给自足场景），所以这一条反而**不是**劣势——但如果做的是"D 帮别的 D 做 prefill"，那就要把 P 侧那套逐层 sender 全搬过来，否则 KV 写出无法与计算重叠。

---

## 六、四种支持方案的对比

### 6.1 方案 A：直接用 NULL（mix）模式 —— 零改动

**做法**：不用 `--disaggregation-mode decode`，用默认的 `null`。这本来就是"一个实例既 prefill 又 decode"。

如果还想要跨实例的 KV 复用（模拟 PD 的效果），配合 HiCache / Mooncake 共享存储 + router 的两段调度（router 把 `max_new_tokens=1` 先发给 mix 0 做 prefill，再发给 mix 1 续 decode）。这个玩法的完整分析见 [mix_vs_prefill_itps.md](mix_vs_prefill_itps.md)。

| 项 | 评价 |
|---|---|
| 改动量 | **零** |
| 能力完整度 | 全（radix cache、chunked prefill、prefill graph、VLM、CP、DWDP 全可用） |
| 缺点 | 拿不到 D 实例的显存/graph 专门化收益；ITL 抖动 |
| 适用 | 中小规模、混合负载、不追求极致尾延迟 |

**这是 90% 场景的正确答案。** 如果你的诉求只是"一个实例既能 prefill 又能 decode"，直接用 NULL 模式，不要动 decode 模式。

### 6.2 方案 B：decode 实例开"本地 prefill 旁路" —— 中等改动（推荐）

**适用场景**（这才是真正需要动手的场景）：

1. **降级容灾**：P 实例集群整体故障 / 网络分区时，D 实例临时自给自足，保住可用性（哪怕性能差）
2. **短请求直通**：prompt 极短（比如 < 512 token）的请求，走 PD 分离的握手 + RDMA 传输开销（毫秒级 RTT）反而比本地算一遍更贵，直接本地做
3. **Warmup / health check / 调试**：不用再注入假 bootstrap 信息

**做法**：在 decode 实例上保留 PREBUILT 主路径，**额外**开一条 NULL 风格的 extend 路径，两条路径的请求在 `running_batch` 里汇合。

| 项 | 评价 |
|---|---|
| 改动量 | 中（约 5 个文件，200~400 行） |
| 能力完整度 | 部分（受第三章裁剪影响，需逐项恢复配置） |
| 风险 | 中（第四章的正确性风险需逐条处理） |
| 适用 | 容灾降级、短请求直通 |

### 6.3 方案 C：引入第四种角色 `unified` / `hybrid` —— 大改动

**做法**：新增 `--disaggregation-mode unified`，一个实例同时具备 P 和 D 的完整能力：既有 bootstrap server + KVSender，又有 prealloc/transfer queue，可以接远端 P 的 KV，也可以把 KV 发给远端 D，也可以自给自足。

注意 [utils.py:107](../../../python/sglang/srt/disaggregation/utils.py#L107) 里 `to_engine_type` 已经有 `"unified"` 这个返回值了（对应 `null` 模式给 router / metrics 看的引擎类型名），但这只是命名巧合，**不代表已有实现**。

| 项 | 评价 |
|---|---|
| 改动量 | **大**（事件循环重构、双向 KV 通道、bootstrap server、角色动态切换、router 协议扩展） |
| 能力完整度 | 全 |
| 风险 | 高（第四章每一条都要处理，PP/DP/spec 的组合矩阵爆炸） |
| 适用 | 需要弹性角色切换的大规模集群（比如按时段在 P/D 之间转换角色） |

**除非有明确的弹性调度需求，否则不建议。** 从工程性价比看，"多部署几个 mix 实例"通常比"把 D 改成 unified"便宜得多。

### 6.4 方案 D：PDMux（SM 分区）—— 已有实现，但要求 mode=null

`--enable-pdmux`（[multiplex/](../../../python/sglang/srt/multiplex/)）在**同一张卡上用 SM 分区**同时跑 prefill 和 decode，通过 `ForwardMode.SPLIT_PREFILL` 把 prefill 切成小片，避免长时间独占 SM。这正好解决 5.2 的 ITL 抖动问题。

但它 [validation_hook.py:110](../../../python/sglang/srt/arg_groups/validation_hook.py#L110) 明确要求 `disaggregation_mode == "null"`，同时还有几条硬约束：

```python
if cfg.enable_pdmux:
    assert cfg.pp_size == 1, "PD-Multiplexing is only supported with pipeline parallelism disabled (pp_size=1)."
    assert cfg.chunked_prefill_size == -1, "PD-Multiplexing is not compatible with chunked prefill."
    assert cfg.disaggregation_mode == "null", "PD-Multiplexing is not compatible with disaggregation mode."
    assert cfg.disable_overlap_schedule, ...
```

| 项 | 评价 |
|---|---|
| 改动量 | 零（如果接受 mode=null）/ 大（如果要在 decode 模式下启用） |
| 能力完整度 | 全，且 ITL 抖动被 SM 分区缓解 |
| 适用 | **想在一张卡上兼顾 prefill 吞吐和 decode 延迟 → 优先看这个，而不是改 decode 模式** |

### 6.5 决策建议

```
你的真实需求是什么？
│
├─ "我想让一个实例既 prefill 又 decode"
│     → 方案 A（用 null 模式），别动 decode 模式
│
├─ "我要兼顾 prefill 吞吐和 decode 尾延迟"
│     → 方案 D（null + --enable-pdmux）
│
├─ "P 集群挂了 D 要能顶住" / "短请求想省一次 RTT"
│     → 方案 B（decode 本地 prefill 旁路）  ← 唯一值得改代码的
│
└─ "我要 P/D 角色按负载弹性切换"
      → 方案 C，但先做成本评估，大概率不划算
```

---

## 七、方案 B 详细设计：decode 实例的本地 prefill 旁路

### 7.0 设计原则

1. **默认关闭**，用一个新开关显式打开，不影响任何现有 PD 部署。
2. **PREBUILT 主路径零改动**。本地 prefill 是"旁路"，不是"替换"。
3. **两条路径在 `running_batch` 汇合，在此之前完全隔离**。请求身上带一个明确的标记位，任何分支判断都读这个标记，不靠 `bootstrap_room is None` 之类的隐式推断。
4. **限流优先于功能**。宁可拒绝请求，也不要让一次大 prefill 打爆 ITL 或 OOM。

### 7.1 新增启动参数

在 [server_args.py](../../../python/sglang/srt/server_args.py) 的 `NS("disagg")` 命名空间下加（命名遵循现有 `disaggregation_decode_*` 前缀惯例）：

```python
# NS("disagg")
disaggregation_decode_enable_local_prefill: bool = False
# 本地 prefill 的 prompt 长度上限；超过则拒绝（或转发回 P 集群）
disaggregation_decode_local_prefill_max_prompt_tokens: int = 512
# 单次本地 extend 的 token 预算（独立于全局 chunked_prefill_size，用于压 ITL）
disaggregation_decode_local_prefill_chunk_size: int = 512
```

配套在 [pd_disaggregation_hook.py](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py) 的 `handle_pd_disaggregation` 里做校验与联动（**这是关键，不做联动就是给自己埋雷**）：

```python
if cfg.disaggregation_decode_enable_local_prefill:
    assert cfg.disaggregation_mode == "decode", \
        "--disaggregation-decode-enable-local-prefill only applies to decode mode"
    # 一期不支持的组合，显式拒绝而不是运行时炸
    assert not cfg.speculative_algorithm, \
        "local prefill does not support speculative decoding yet"   # 见 4.4
    assert cfg.pp_size == 1, \
        "local prefill does not support pipeline parallel yet"      # 见 4.6
    assert not cfg.enable_hisparse
    # SWA / DSA 模型的池子形状假设不成立，见 3.5 / 3.6
    assert not model_has_swa, "local prefill is incompatible with SWA pool sizing"
```

`pp_size == 1` 和"禁投机"这两条是一期最重要的护栏。放开它们的成本远高于收益。

### 7.2 恢复被裁剪的能力（配置层）

| 需要做的事 | 位置 | 具体改法 |
|---|---|---|
| 恢复 prefill CUDA graph | [cuda_graph_hook.py:580](../../../python/sglang/srt/arg_groups/cuda_graph_hook.py#L580) | `elif` 分支加 `and not cfg.disaggregation_decode_enable_local_prefill`。或者**一期干脆不恢复**——本地 prefill 只服务短请求，eager 可以接受，省掉一大块 graph 显存 |
| 恢复激活显存预算 | [memory_hook.py:239](../../../python/sglang/srt/arg_groups/memory_hook.py#L239) | decode 分支的工作集取 `max(decode_working_set, local_prefill_chunk_size)` |
| 恢复 page 对齐校验 | [validation_hook.py:104](../../../python/sglang/srt/arg_groups/validation_hook.py#L104) | 开关打开时不再跳过 |
| 恢复 MoE dispatch 预算校验 | [moe_hook.py:304/352/379/463](../../../python/sglang/srt/arg_groups/moe_hook.py#L304) | 用 `local_prefill_chunk_size` 代替 `chunked_prefill_size` 参与校验。**MoE 模型上这条是必须的**（见 3.4） |
| 恢复 VLM 显存 headroom | [memory_hook.py:275](../../../python/sglang/srt/arg_groups/memory_hook.py#L275) | 只在同时开了本地 prefill 且模型是 VLM 时恢复。不恢复则一期禁多模态本地 prefill |
| DeepGEMM 预编译 extend 形状 | [compile_utils.py:503](../../../python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py#L503) | `run_extend = disagg_mode != "decode" or local_prefill_enabled`，避免首请求秒级 JIT 卡顿 |
| radix cache | [pd_disaggregation_hook.py:101](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L101) | **一期不动**。让本地 prefill 跑在 chunk cache 上，损失前缀复用但避免踩 3.2 那一串互斥约束 |

### 7.3 准入路径改造

**Step 1 — 放开 bootstrap_room 校验**（[scheduler.py:2571](../../../python/sglang/srt/managers/scheduler.py#L2571)）

```python
if self.disaggregation_mode != DisaggregationMode.NULL:
    if recv_req.bootstrap_room is None and self.transfer_backend != TransferBackend.FAKE:
        if self.local_prefill_enabled:          # 新增：init 时从 server args 提取的属性
            req.is_local_prefill = True         # 显式标记，后续所有分支读这个字段
        else:
            ... prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
```

注意：判定依据用"**没带 bootstrap 信息**"，而不是"prompt 短"。router 决定一个请求走 PD 还是走本地，靠的就是发不发 bootstrap 字段——这样控制权在 router，实例侧只负责执行。

`is_local_prefill` 要在 `Req.__init__` 里初始化为 `False`（不要用 `getattr` 兜底，见仓库规则 `.claude/rules/no-getattr-defensive.md`）。

**Step 2 — 长度限流**

在同一处加：

```python
if req.is_local_prefill:
    if len(req.origin_input_ids) > self.local_prefill_max_prompt_tokens:
        prepare_abort(req, "prompt too long for local prefill on decode instance",
                      status_code=HTTPStatus.BAD_REQUEST)
        return
```

**Step 3 — 分流到 `waiting_queue`**（[scheduler.py:2912](../../../python/sglang/srt/managers/scheduler.py#L2912)）

```python
elif self.disaggregation_mode == DisaggregationMode.DECODE:
    if req.is_local_prefill:
        if self._abort_on_queued_limit(req): return
        self._prefetch_kvcache(req)                    # HiCache 预取，与 NULL 模式一致
        self.local_prefill_waiting_queue.append(req)   # ★独立队列，见下
        req.time_stats.set_wait_queue_entry_time()
    else:
        self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
        ...
```

**为什么要独立队列而不是直接用 `waiting_queue`？** 因为 `waiting_queue` 在 decode 实例上是"KV 已到位、等着进 PREBUILT 批"的队列，`get_new_prebuilt_batch` 会无条件把里面的请求当成 prebuilt 处理（[decode.py:2617](../../../python/sglang/srt/disaggregation/decode.py#L2617) 那个循环没有任何过滤）。混进去必定触发 4.1 的第一 token 冲突。

用独立队列是最小侵入的隔离方式——`get_new_prebuilt_batch` 一行都不用改。

### 7.4 事件循环改造

在 [decode.py:2560](../../../python/sglang/srt/disaggregation/decode.py#L2560) `get_next_disagg_decode_batch_to_run` 里，在 prebuilt 处理**之后**、decode 调度**之前**插入 extend 分支：

```python
def get_next_disagg_decode_batch_to_run(self, running_batch) -> NextBatchPlan:
    # ① 原有：prebuilt 批（来自远端 P）
    new_prebuilt_batch = self.get_new_prebuilt_batch(running_batch)
    if new_prebuilt_batch:
        ...  # 原逻辑不动
        running_batch = merge(...)

    # ② 新增：本地 prefill 批（优先级低于 prebuilt，且与 decode 互斥）
    if self.local_prefill_enabled:
        new_extend_batch = self.get_new_local_prefill_batch(running_batch)
        if new_extend_batch is not None:
            # 直接返回 extend 批，本轮不跑 decode（避免 MIXED 形态的额外复杂度）
            return NextBatchPlan(batch_to_run=new_extend_batch, running_batch=running_batch)

    # ③ 原有：decode
    if running_batch.is_empty(): ret = None
    else:
        running_batch = self.update_running_batch(running_batch)
        ret = running_batch if not running_batch.is_empty() else None
    ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(ret)
    return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)
```

`get_new_local_prefill_batch` 的实现要点（**不要试图复用 `get_new_batch_prefill`**——那个函数深度耦合 NULL 模式的 `chunked_req` / `PrefillAdder` / policy 状态机，硬搬进来会引入一堆隐式假设）。写一个精简版：

```python
def get_new_local_prefill_batch(self, running_batch) -> Optional[ScheduleBatch]:
    if not self.local_prefill_waiting_queue:
        return None
    # 背压门 1：running batch 的 decode 显存必须充足（decode SLO 优先）
    if not self.check_decode_mem(...):
        return None
    # 背压门 2：有 retracted 请求时一律不接新 prefill（与 process_decode_queue 的策略一致）
    if self.disagg_decode_prealloc_queue.retracted_queue:
        return None
    # 背压门 3：running batch 已满
    if running_batch.batch_size() >= self.max_running_requests:
        return None

    token_budget = self.local_prefill_chunk_size
    can_run_list = []
    for req in list(self.local_prefill_waiting_queue):
        req.init_next_round_input(self.tree_cache)
        if req.extend_input_len > token_budget:
            break                       # 一期：不做跨步 chunk，攒不下就等下一轮
        token_budget -= req.extend_input_len
        can_run_list.append(req)
    if not can_run_list:
        return None
    self.local_prefill_waiting_queue = [r for r in self.local_prefill_waiting_queue
                                        if r not in can_run_list]

    batch = ScheduleBatch.init_new(can_run_list, self.req_to_token_pool,
                                   self.token_to_kv_pool_allocator, self.tree_cache,
                                   self.model_config, self.enable_overlap,
                                   self.spec_algorithm)
    batch.prepare_for_extend()          # ★ ForwardMode.EXTEND，走真 forward
    return batch
```

**结果处理**：`run_batch` 对 EXTEND 批走的是通用路径（`scheduler.py:3894` 的 prebuilt 短路不会命中），`process_batch_result` 在 [scheduler.py:4233](../../../python/sglang/srt/managers/scheduler.py#L4233) 会按 `is_extend()` 分派——**必须确认它进的是 `process_batch_result_prefill`（NULL 路径）而不是 `process_batch_result_disagg_prefill`（PREFILL 路径）**。现有代码是按 `disaggregation_mode` 判断的，所以在 DECODE 模式下会走 NULL 分支，正好符合需要；但要加测试锁住这个行为。

extend 完成后请求带着第一个 token 自然进入 `running_batch`，后续和 prebuilt 来的请求走完全相同的 DECODE 步——**这是这个方案能成立的关键：汇合点之后没有任何区别。**

### 7.5 需要显式处理的边界

| 边界 | 处理方式 |
|---|---|
| **Retraction**（4.3） | 本地 prefill 的请求 `is_local_prefill=True`，retract 时禁止走 rebootstrap 路径（没有 P 侧可退），只允许 `cpu_tensor` / `host_pool` 备份。在 `retract_decode` 的排序里可以优先 retract 它们（本地重算比跨机重传便宜） |
| **DSA seed remap**（3.5） | `should_remap_pd_dsa_seed_to_local_slots()` 对本地 prefill 的请求必须返回 False。**一期建议直接禁 DSA 模型 + 本地 prefill 组合** |
| **投机解码**（4.4） | 一期在 7.1 的校验里直接 assert 禁止 |
| **PP**（4.6） | 一期 assert `pp_size == 1` |
| **DP attention**（4.5） | 复用 NULL 模式的 `maybe_prepare_mlp_sync_batch`。必须在 dp_size > 1 上实测，重点看各 rank forward mode 不一致时是否 hang |
| **SWA 池**（3.6） | 一期 assert 禁止 SWA 模型 |
| **grammar / structured output** | 本地 prefill 走 NULL 路径的 grammar 处理，不经过 `process_prebuilt` 里那段 `accept_token` 补偿逻辑，天然正确 |
| **logprob**（4.2） | 走 NULL 路径自己算，天然正确 |

### 7.6 观测

- 新增 metric：`sglang:decode_local_prefill_reqs_total`、`sglang:decode_local_prefill_tokens_total`、`sglang:decode_local_prefill_rejected_total`（按原因分 label）
- `RequestStage` 加一个本地 prefill 的阶段标记，让 [req_time_stats.py](../../../python/sglang/srt/observability/req_time_stats.py) 能区分两条路径的 TTFT 来源
- **必须监控**：本地 prefill 开启前后的 **P99 ITL**。这是判断这个特性是否在伤害 SLO 的唯一可靠指标
- 日志：本地 prefill 批的 token 数 + 该步耗时，用于校准 `local_prefill_chunk_size`

### 7.7 测试计划

参考 skill `write-sglang-test`，测试放 `test/registered/disaggregation/`：

1. **单测（mock）**：`_add_request_to_queue` 对带/不带 bootstrap_room 的请求分流正确；超长 prompt 被拒
2. **单测**：`get_new_local_prefill_batch` 的三道背压门各自生效
3. **端到端**：起一个 `--disaggregation-mode decode --disaggregation-decode-enable-local-prefill` 实例 + 一个 P 实例，混合发送带 bootstrap 和不带 bootstrap 的请求，校验两类请求输出都正确
4. **一致性**：同一 prompt 分别走 (a) PD 分离路径 (b) decode 本地 prefill 路径，比对输出 token 序列。**这是最重要的一条**——它同时验证了 4.1 的第一 token 语义和 4.2 的 logprob 归属
5. **回归**：不开开关时，所有现有 PD 测试必须零变化（`test/registered/disaggregation/` 全套）
6. **ITL 基线**：benchmark 对比开关开/关时的 P99 ITL

### 7.8 分期建议

| 期 | 范围 | 验收 |
|---|---|---|
| P0 | 开关 + 准入分流 + 事件循环 extend 分支；限 `pp=1`、非投机、非 SWA、非 DSA、非 VLM、chunk cache | 7.7 的第 1~5 条通过 |
| P1 | 恢复 prefill CUDA graph + DeepGEMM extend 预编译 + MoE 预算校验 | 首请求无 JIT 卡顿；MoE 模型不炸 |
| P2 | 跨步 chunked prefill（把大 prompt 切片，彻底解决 ITL 抖动）；放开 prompt 长度上限 | P99 ITL 退化 < 10% |
| P3 | 视需求放开 DP attention / 投机 / radix cache | 逐项实测 |

**如果 P0 做完发现 ITL 退化不可接受，应该果断停在这里，回到方案 A 或 D。** 这个特性的价值在容灾和短请求直通，不在通用吞吐。

---

## 八、关键代码位置索引

### 8.1 结构性阻塞

| 内容 | 位置 |
|---|---|
| 事件循环分派 | [scheduler.py:5294](../../../python/sglang/srt/managers/scheduler.py#L5294) `dispatch_event_loop` |
| decode 普通循环 | [decode.py:2450](../../../python/sglang/srt/disaggregation/decode.py#L2450) |
| decode overlap 循环 | [decode.py:2488](../../../python/sglang/srt/disaggregation/decode.py#L2488) |
| decode 调度决策（无 prefill 路径） | [decode.py:2560](../../../python/sglang/srt/disaggregation/decode.py#L2560) |
| 请求入队分流 | [scheduler.py:2897](../../../python/sglang/srt/managers/scheduler.py#L2897) `_add_request_to_queue` |
| bootstrap_room 强制校验 | [scheduler.py:2571](../../../python/sglang/srt/managers/scheduler.py#L2571) |
| bootstrap server 只在 P 侧启动 | [disagg_service.py:23](../../../python/sglang/srt/managers/disagg_service.py#L23) |
| PD 队列构造（按 mode 二分） | [scheduler.py:1408](../../../python/sglang/srt/managers/scheduler.py#L1408) |

### 8.2 PREBUILT 机制

| 内容 | 位置 |
|---|---|
| `ForwardMode.PREBUILT` 定义 | [forward_batch_info.py:122](../../../python/sglang/srt/model_executor/forward_batch_info.py#L122) |
| `is_prebuilt()` | [forward_batch_info.py:198](../../../python/sglang/srt/model_executor/forward_batch_info.py#L198) |
| forward 短路 | [scheduler.py:3893](../../../python/sglang/srt/managers/scheduler.py#L3893) |
| `_run_batch_prebuilt` | [decode.py:2548](../../../python/sglang/srt/disaggregation/decode.py#L2548) |
| `get_new_prebuilt_batch` | [decode.py:2592](../../../python/sglang/srt/disaggregation/decode.py#L2592) |
| `prepare_for_prebuilt` | [decode_schedule_batch_mixin.py:23](../../../python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py#L23) |
| `process_prebuilt` | [decode_schedule_batch_mixin.py:113](../../../python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py#L113) |
| 结果处理 | [batch_result_processor.py:99](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L99) |
| 第一个 token 从 metadata buffer 落地 | [decode.py:2059](../../../python/sglang/srt/disaggregation/decode.py#L2059) `_commit_transfer_to_req` |
| DP attention 下挂 inner idle batch | [dp_attn.py:477](../../../python/sglang/srt/managers/scheduler_components/dp_attn.py#L477) |

### 8.3 三级队列

| 内容 | 位置 |
|---|---|
| `DecodePreallocQueue` | [decode.py:316](../../../python/sglang/srt/disaggregation/decode.py#L316) |
| `pop_preallocated` | [decode.py:1080](../../../python/sglang/srt/disaggregation/decode.py#L1080) |
| `resume_retracted_reqs` | [decode.py:804](../../../python/sglang/srt/disaggregation/decode.py#L804) |
| `DecodeTransferQueue` | [decode.py:2015](../../../python/sglang/srt/disaggregation/decode.py#L2015) |
| `pop_transferred` | [decode.py:2245](../../../python/sglang/srt/disaggregation/decode.py#L2245) |
| `process_decode_queue` | [decode.py:2667](../../../python/sglang/srt/disaggregation/decode.py#L2667) |

### 8.4 按 mode 裁剪的配置

| 内容 | 位置 |
|---|---|
| prefill CUDA graph 强制关闭 | [cuda_graph_hook.py:569](../../../python/sglang/srt/arg_groups/cuda_graph_hook.py#L569) |
| radix cache 强制关闭 | [pd_disaggregation_hook.py:101](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L101) |
| 激活显存按 decode 形态算 | [memory_hook.py:239](../../../python/sglang/srt/arg_groups/memory_hook.py#L239) |
| VLM headroom 不预留 | [memory_hook.py:275](../../../python/sglang/srt/arg_groups/memory_hook.py#L275) |
| page 对齐校验跳过 | [validation_hook.py:104](../../../python/sglang/srt/arg_groups/validation_hook.py#L104) |
| MoE dispatch 预算校验跳过 | [moe_hook.py:304](../../../python/sglang/srt/arg_groups/moe_hook.py#L304) |
| DeepGEMM 不预编译 extend | [compile_utils.py:503](../../../python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py#L503) |
| MLA chunked prefix cache 关闭 | [flashinfer_mla_backend.py:243](../../../python/sglang/srt/layers/attention/flashinfer_mla_backend.py#L243) |
| DSA seed remap 仅 decode | [dsa/utils.py:78](../../../python/sglang/srt/layers/attention/dsa/utils.py#L78) |
| 前缀匹配回填跳过 | [schedule_policy.py:249](../../../python/sglang/srt/managers/schedule_policy.py#L249) |
| SWA 池上限公式 | [pool_configurator.py:682](../../../python/sglang/srt/model_executor/pool_configurator.py#L682) |
| `req_to_token_pool` extra slots | [kv_cache_configurator.py:938](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L938) |
| retraction backup 仅 decode | [kv_cache_builder.py:166](../../../python/sglang/srt/mem_cache/kv_cache_builder.py#L166) |
| PDMux 要求 mode=null | [validation_hook.py:110](../../../python/sglang/srt/arg_groups/validation_hook.py#L110) |

---

## 九、总结

1. **decode 实例现在做不了 prefill**，原因是调度器里没写这条路（[decode.py:2560](../../../python/sglang/srt/disaggregation/decode.py#L2560)）+ 请求进不了 `waiting_queue`（[scheduler.py:2912](../../../python/sglang/srt/managers/scheduler.py#L2912)）+ 没 bootstrap_room 就 400（[scheduler.py:2571](../../../python/sglang/srt/managers/scheduler.py#L2571)）。

2. **这不是硬件或算子限制。** decode 实例的 `capture_prefill_graph` 在关闭时退化成 eager 而非报错，`TARGET_VERIFY` 证明它跑得了 extend 形态的 forward。是"设计上不让"，不是"物理上不能"。

3. **真正的成本在第三、四章那几十处按 decode 形态做的裁剪**：CUDA graph、radix cache、激活显存、MoE dispatch 预算、SWA 池公式、DSA seed 重映射。这些是"改配置能恢复但要一项项验证"的长尾。

4. **最大的反对理由是性能**：一次长 prompt 的 prefill 会让 running batch 里所有解码请求停摆几百毫秒，直接摧毁 D 实例的 ITL SLO；同时 D 实例只有 ~2K token 的激活预算，prefill 自己的 MFU 也很低。**两头都不讨好。**

5. **绝大多数"想让一个实例既 prefill 又 decode"的需求，正确答案是用 `--disaggregation-mode null`（方案 A），或者 `null + --enable-pdmux`（方案 D）。** 这两个都是零改动。

6. **唯一值得改代码的场景是容灾降级和短请求直通**（方案 B）。设计要点是：默认关闭、PREBUILT 主路径零改动、用独立队列隔离两条路径、一期严格收窄组合（禁 PP/投机/SWA/DSA/VLM）、把限流放在功能之前。分期计划见 7.8。

7. **不建议做方案 C（unified 角色）**，除非有明确的 P/D 弹性切换需求。多部署几个 mix 实例通常便宜得多。





