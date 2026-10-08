# 请求处理流程与 UnifiedRadixCache 的交互时间线：集中式 / P 实例 / D 实例

> 基线 commit：`33ed29a0ee`
>
> 本文回答一个具体问题：**一个请求从进来到结束，在三种部署形态（集中式 mix、PD 的 Prefill 实例、PD 的 Decode 实例）下，
> 分别在什么时刻、以什么顺序碰到了 `self.tree_cache`（`UnifiedRadixCache`）的哪些方法？**
>
> 本文只讲**调用方视角的时间线**。`UnifiedRadixCache` 内部机制（树结构、组件、墓碑、级联淘汰、insert 状态机）
> 见 [unified_radix_cache_architecture.md](unified_radix_cache_architecture.md)，两篇是互补的。
>
> 相关篇目：
> - [centralized_scheduling_loop_architecture.md](../02_scheduling/centralized_scheduling_loop_architecture.md) — 集中式调度循环
> - [pd_disaggregation_architecture.md](../04_pd_disaggregation/pd_disaggregation_architecture.md) — PD 分离总览
> - [pd_decode_radix_cache_hicache_scheme.md](../04_pd_disaggregation/pd_decode_radix_cache_hicache_scheme.md) — D 侧 radix + HiCache 方案
> - [decode_instance_prefill_capability.md](../04_pd_disaggregation/decode_instance_prefill_capability.md) — 为什么 D 侧不做 prefill
> - [hicache_usage_and_design.md](hicache_usage_and_design.md) — HiCache 三级缓存

---

## 0. 先看结论：一张对照表

如果只想记一件事，记这个：

| | 集中式 mix | P 实例 (prefill) | D 实例 (decode) |
|---|---|---|---|
| **`match_prefix` 在哪** | 组批阶段（`init_next_round_input`） | 同 mix（组批阶段） | **准入阶段**（PreallocQueue），比 mix 早得多 |
| **`match_prefix` 为什么** | 少算一段 forward | 少算一段 forward | 少传一段 **网络**（产出 `decode_prefix_len`） |
| **`cache_unfinished_req`（插树）** | forward 结果处理时 | forward 结果处理时（`prefill.py:762`） | **prebuilt 假批次里**（唯一一处） |
| **`cache_finished_req`（插树+解锁）** | forward 结果处理时 | **RDMA 传输完成轮询时**（`prefill.py:932`） | 结果处理时（正常结束） |
| **锁的生命周期** | 准入 → 结果处理 | 准入 → **RDMA 完成** | 准入 → prebuilt 插树时换锁 |
| **是否跑 extend forward** | 是 | 是 | **否**（`ForwardMode.PREBUILT`，空结果） |
| **第一个 token 来源** | 本地 sample | 本地 sample | **P 侧 sample，经 metadata 传来** |
| **retract 是否备份 KV** | 否（重算） | 不适用 | 是（`retraction_backup`，host_pool / cpu_tensor） |
| **HiCache 事件排空点** | `get_new_batch_prefill` 开头 | 同 mix | `process_decode_queue` 开头 |

三句话版本：

1. **mix 和 P 的时间线几乎一样**，唯一的结构性差异是 **P 把「解锁」从 forward 结果处理里搬到了 RDMA 完成轮询里**。
2. **D 的时间线是另一套东西**：匹配前移到准入、插树发生在一个假批次里、没有 forward。
3. 三者共用同一套「分配前按需淘汰」机制，没有区别。

---

## 1. 前置知识：`self.tree_cache` 到底是哪个对象？

### 1.1 构造链（三种模式完全共用）

```
Scheduler.__init__
  └─ kv_cache_builder.build_kv_cache()          kv_cache_builder.py:197
       ├─ CacheInitParams(...)                  kv_cache_builder.py:290
       └─ create_tree_cache(ctx)                kv_cache_builder.py:322
            └─ registry.create_tree_cache()     registry.py:199
                 └─ default_radix_cache_factory registry.py:80   ← 选型链
                      └─ _create_unified_radix_cache
                           ├─ params.tree_components = (FULL,) / (FULL, SWA) / ...
                           ├─ UnifiedRadixCache(params)          registry.py:187
                           └─ cache.init_hicache(...)            registry.py:192
  └─ self.tree_cache = result.tree_cache        scheduler.py:582
```

选型链 [`default_radix_cache_factory`](../../../python/sglang/srt/mem_cache/registry.py#L80) 的分支顺序很重要：

| 顺序 | 条件 | 返回 |
|---|---|---|
| 1 | `disable_radix_cache` **且** retraction backup == `host_pool` | `UnifiedRadixCache`（**`disable=True`**） |
| 2 | `chunked_prefill_size is not None` 且 `disable_radix_cache` | `ChunkCache` / `PureSWAChunkCache` / `SWAChunkCache` |
| 3 | `SGLANG_EXPERIMENTAL_CPP_RADIX_TREE` | `RadixCacheCpp` |
| 4 | 纯 SWA 模型 | `PureSWARadixCache` |
| 5 | LMCache / FlexKV 后端 | 对应包装 |
| 6 | 其余（默认） | `UnifiedRadixCache` |

**第 1 条分支是最容易踩的坑**：D 实例默认不开 radix cache
（[`pd_disaggregation_hook.py:108-113`](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L108-L113) 强制
`disable_radix_cache=True`），而 [`resolve_decode_retraction_backup`](../../../python/sglang/srt/mem_cache/kv_cache_builder.py#L166-L179)
恰恰在「不开 radix」时才推断出 `host_pool` 后端。两者一撞，就走进第 1 条分支：

> **D 实例的默认配置下，`self.tree_cache` 是一个 `disable=True` 的 `UnifiedRadixCache`，不是 `ChunkCache`。**
>
> 它的行为等价于 chunk cache（`match_prefix` 返回空、`insert` 返回 0、`evict` 空转、锁引用空转，
> 见 [`unified_radix_cache.py:541`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L541)
> 的 `is_chunk_cache() → self.disable`），但**它带着一个活的 L2 host 池**，
> 专供 retraction backup 使用（`hicache_ratio = BACKUP_ONLY_HICACHE_RATIO = 0.2`，
> [`kv_cache_builder.py:119`](../../../python/sglang/srt/mem_cache/kv_cache_builder.py#L119)）。

### 1.2 D 侧两个总闸

```python
# scheduler.py:469-472
self.enable_decode_hicache = (
    disaggregation_decode_enable_radix_cache and self.enable_hierarchical_cache
)
```

- `--disaggregation-decode-enable-radix-cache`：D 侧是否建真 radix 树
- `enable_decode_hicache` = 上者 **且** `--enable-hierarchical-cache`：D 侧是否挂 L2/L3

D 侧的硬拒绝（[`pd_disaggregation_hook.py:79-94`](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L79-L94)）：
hisparse、fake transfer backend、**投机解码**；DCP 在 `:66-70` 也不兼容。

### 1.3 事件循环一次性分派，运行期不切换

[`dispatch_event_loop`](../../../python/sglang/srt/managers/scheduler.py#L5294)（`scheduler.py:5294-5321`）
按 `disaggregation_mode` 一次性选定：

| mode | 循环函数 |
|---|---|
| `NULL`（mix） | `event_loop_normal` / `event_loop_overlap` / pp / pdmux |
| `PREFILL` | `event_loop_normal_disagg_prefill` / `event_loop_overlap_disagg_prefill` / pp |
| `DECODE` | `event_loop_normal_disagg_decode` / `event_loop_overlap_disagg_decode` / pp |

---

## 2. 十个「接触点」语义速查

看时间线之前，先把这十个方法的含义记住。后面的时间线全是这十个的排列组合。

| 方法 | 一句话语义 | 定义位置 |
|---|---|---|
| `match_prefix(params)` | 查树：这段 token 有多长前缀已经缓存了？返回 device 命中长度 + host 命中长度 + 锚点节点 | [`:523`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L523) |
| `inc_lock_ref(node)` | 给命中路径上锁，**淘汰不能碰它** | [`:778`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L778) |
| `dec_lock_ref(node)` | 解锁，这段前缀重新变成可淘汰 | [`:788`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L788) |
| `cache_unfinished_req(req)` | **请求还没结束**：把已算出的 KV 插树，并把锁从旧节点**原子换到**新节点 | [`:931`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L931) |
| `cache_finished_req(req)` | **请求结束**：插树 + **释放锁** + 释放尾部零碎 slot | [`:838`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L838) |
| `evict_for_alloc(params)` | 按 allocator 缺口淘汰，够了就停 | [`:566`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L566) |
| `check_hicache_events()` | 每步排空 HiCache 异步事件（D→H 写完成、H→D 读完成、L3 事件） | [`:2787`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2787) |
| `init_load_back(params)` | 命中在 L2/L3 → 排一次 H→D 回载 | [`:2740`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2740) |
| `ready_to_load_host_cache()` | 踢一脚 controller，开始本批的 layer-wise 回载，返回 consumer index | [`:2868`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2868) |
| `retraction_backup/restore/discard(req)` | **仅 D 侧**：retract 时整段 KV 搬到 host 池，恢复时搬回 | [`:1236`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1236) / [`:1283`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1283) / [`:1330`](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1330) |

### 2.1 两条最容易混淆的语义

**`cache_unfinished_req` 不是「插树」这么简单，它是一次锁的原子交接。**

```python
# unified_radix_cache.py:997-1045（简化）
self.insert(insert_params)                    # :997  插树
new_last_node = self.match_prefix(...)        # :1001 重新匹配拿新锚点
self.req_to_token_pool.write(...)             # :1011 请求的 req_to_token 行改指向树拥有的 slot
self._dec_req_lock(req)                       # :1016 放掉旧节点的锁
self.inc_lock_ref(new_last_node, ...)         # :1028 给新节点上锁
req.last_node = new_last_node                 # :1040
```

第 `:1011` 行是理解 P 侧行为的钥匙：**插树之后，请求的 KV 页归树所有了**，
请求只是「借」着这些页。所以谁在读这些页，谁就必须持锁。

**淘汰永远是「分配前的按需淘汰」，没有后台线程。** 所有淘汰点都长成同一形状：

```python
# mem_cache/common.py:129-155  evict_from_tree_cache
if tree_cache.is_chunk_cache():  return           # disable=True 直接返回
shortfall = num_tokens - allocator.available_size()
if shortfall > 0:  tree_cache.evict_for_alloc(EvictParams(num_tokens=shortfall))
```

三种模式的淘汰触发点完全一致，都在 [`allocation.py`](../../../python/sglang/srt/mem_cache/allocation.py)：

| 位置 | 时机 |
|---|---|
| `allocation.py:155` `alloc_token_slots` | `page_size == 1` 的 extend |
| `allocation.py:187` `alloc_paged_token_slots_extend` | 分页 extend |
| `allocation.py:485` `alloc_paged_token_slots_decode` | 每个 decode step |
| `allocation.py:260` `alloc_req_slots` | Mamba state slot 不足 |
| `allocator/base.py:77` `check_decode_capacity` | 每个 decode step，**先淘汰再判断是否要 retract** |

---

## 3. 集中式（mix，`disaggregation_mode is None`）时间线

### 3.1 阶段 A：进程启动（一次性）

| 步 | 位置 | 动作 |
|---|---|---|
| A1 | `scheduler.py:582` | `self.tree_cache = result.tree_cache` |
| A2 | `scheduler.py:593` | `attach_radix_cache(self.tree_cache)` 分发句柄给各协作者 |
| A3 | `scheduler.py:1282` | `SchedulePolicy(policy, self.tree_cache, ...)`；**若 `tree_cache.disable` 则把 cache-aware 策略降级成 FCFS**（`schedule_policy.py:314`） |
| A4 | `scheduler.py:5294` | `dispatch_event_loop()` 定型 |

注意 `SchedulePolicy` 内部还额外建了**一棵独立的模拟 `RadixCache`**
（`schedule_policy.py:238`），用于批内前缀去重排序，**它不是 `self.tree_cache`**。

### 3.2 阶段 B：请求准入（每请求一次）

| 步 | 位置 | 动作 | 前置条件 |
|---|---|---|---|
| B1 | `scheduler.py:2897→2903` | `_prefetch_kvcache(req)` 然后 `waiting_queue.append(req)` | — |
| B2 | `scheduler.py:2821` | `req.init_next_round_input(self.tree_cache, cow_mamba=False)` — **第一次真 `match_prefix`**，只为找 L1/L2 锚点 | `enable_hicache_storage` |
| B3 | `scheduler.py:2831-2852` | `is_root` / `is_backuped` / `get_last_hash_value` / `get_prefix_hash_values` — 判断值不值得查 L3 | 同上 |
| B4 | `scheduler.py:2856` | `tree_cache.prefetch_from_storage(...)` 发起异步 L3 预取 | 同上 |

### 3.3 阶段 C：调度循环（**每步**，热路径）

这是整篇文档的核心。按执行顺序：

**C-1 组批前的清理与事件排空**

| 步 | 位置 | 动作 |
|---|---|---|
| C1 | `scheduler.py:3197` | `process_pending_chunked_abort()`：`release_aborted_request` + `release_kv_cache(is_insert=False)` |
| C2 | `scheduler.py:3104-3105` | `stash_chunked_request` → **`cache_unfinished_req(req, chunked=True)`** — 分块 prefill 的中间态寄存到树里 |
| C3 | `scheduler.py:3402` | **`tree_cache.check_hicache_events()`** — 排空 `writing_check`（D→H 写完成）+ `loading_check`（H→D 读完成，**这里释放准入时拿的回载锁**）+ L3 事件 |
| C4 | `scheduler.py:3404` | `_retry_missed_storage_prefetches()` → `pop_storage_prefetch_miss` + 重发预取 |

**C-2 waiting queue 排序（对整个队列做 `match_prefix`）**

| 步 | 位置 | 动作 |
|---|---|---|
| C5 | `schedule_policy.py:255` | 非 cache-aware 策略的快速匹配：对**每个排队请求**调 `match_prefix_for_req`。守卫含 `disaggregation_mode != "decode"`（`:252`） |
| C6 | `schedule_policy.py:338` | LPM/DFS_WEIGHT：真树 `match_prefix` |
| C7 | `schedule_policy.py:350/371` | 再对**模拟树**做 match + insert，做批内去重 |
| C8 | `schedule_policy.py:401` | `tree_cache.dfs_weight_order(...)`（仅 DFS_WEIGHT） |

> C5/C6 意味着**每个调度步都会对整条 waiting queue 做一遍 `match_prefix`**。队列长时这是实打实的 CPU 开销。

**C-3 逐请求准入（`PrefillAdder.add_one_req`）**

| 步 | 位置 | 动作 |
|---|---|---|
| C9 | `scheduler.py:3535` | `check_prefetch_progress(req.rid)` — 没好就 `continue` |
| C10 | `scheduler.py:3540` | `pop_prefetch_loaded_tokens(req.rid)` |
| C11 | `scheduler.py:3544` | **`req.init_next_round_input(self.tree_cache)`** → `schedule_batch.py:1419` **权威 `match_prefix`**，写 `prefix_indices` / `last_node` / `host_hit_length` / `cache_protected_len` |
| C12 | `scheduler.py:3558` | `plan_staged_splice(...)` | `buffer_only` 模式 |
| C13 | `schedule_policy.py:1028/1037` | **临时锁**：`inc_lock_ref` → 算预算 → `dec_lock_ref`。防止决策期间自己把候选前缀淘汰掉 |
| C14 | `schedule_policy.py:1289` | `tree_cache.init_load_back(...)` — host 命中就排回载 | `req.needs_host_load_back()` |
| C15 | `schedule_policy.py:1342`/`1393` | **`_req_inc_lock_ref(req)` = 持久准入锁**，一直持有到请求结束 |

**C-4 批次准备（淘汰发生在这里）**

| 步 | 位置 | 动作 |
|---|---|---|
| C16 | `scheduler.py:3624` | `ScheduleBatch.init_new(..., self.tree_cache, ...)` |
| C17 | `scheduler.py:3642` | `new_batch.hicache_consumer_index = ready_to_load_host_cache()` — 踢一脚回载 |
| C18 | `scheduler.py:3645` → `allocation.py:282` | `prepare_for_extend` → `alloc_for_extend` → **`evict_from_tree_cache`**（`allocation.py:155`/`:187`） |

**C-5 decode 批次更新**

| 步 | 位置 | 动作 |
|---|---|---|
| C19 | `scheduler.py:3738` | `check_decode_mem()` → `allocator/base.py:77` **先淘汰再判断** |
| C20 | `scheduler.py:3751` | 不够 → `retract_decode()` → `release_req` → `release_kv_cache(is_insert=False)` + `evict_from_tree_cache(remaing × RETRACT_DECODE_STEPS)`（`schedule_batch.py:2025-2028`）。**mix 不做 KV 备份，重排队后重算** |
| C21 | `scheduler.py:3809` → `allocation.py:521` | `prepare_for_decode` → `alloc_for_decode` → 再一次 `evict_from_tree_cache` |
| C22 | `schedule_batch.py:3580` | SWA：窗口滑过后 `dec_swa_lock_only(...)` 提前放掉 SWA 锁 | `supports_swa()` |

**C-6 结果处理：插树与释放**

| 步 | 位置 | 动作 |
|---|---|---|
| C23 | `batch_result_processor.py:343` | prefill 批且 `req.finished()` → **`release_kv_cache`** → `cache_finished_req`（插树 + 解锁） |
| C24 | `batch_result_processor.py:346` | prefill 批且未结束 → **`maybe_cache_unfinished_req`**（extend 末尾插树） |
| C25 | `batch_result_processor.py:1198` | decode 结束 → `release_kv_cache(is_insert=...)` |

**C-7 空闲housekeeping**

| 步 | 位置 | 动作 |
|---|---|---|
| C26 | `scheduler.py:4395` | `invariant_checker._check_tree_cache()` → `sanity_check()` | 仅 hybrid SWA/SSM 模型 |
| C27 | `scheduler.py:4401` | `publish_kv_events()` → `tree_cache.take_events()` | KV events 开启 |

### 3.4 阶段 D：管理接口（一次性）

`flush_cache` → `tree_cache.reset()`（`scheduler.py:4594`，前置 `is_fully_idle()`）；
`attach/detach_storage_backend`（`:4511`/`:4569`）；
`open/release_radix_session`（`:5231`/`:5238`）；
进程退出 `release_host_resources()`（`:1739`）。

---

## 4. P 实例（prefill）时间线

### 4.1 一句话总结

**P 的时间线 = mix 的时间线，但把「解锁」从 forward 结果处理搬到了 RDMA 传输完成轮询里。**

注意不是把「插树」搬走了 —— 插树仍然在 forward 结果处理时立即发生。
这个区别是理解 P 侧行为的关键，下面 4.5 会详细讲。

### 4.2 阶段 A：配置（一次性）

P 侧**不禁用** radix cache。对比一下两个 hook 分支：

| 位置 | 作用 |
|---|---|
| [`pd_disaggregation_hook.py:108-113`](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L108-L113) | **decode** 模式：强制 `disable_radix_cache = True` |
| [`pd_disaggregation_hook.py:131-137`](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L131-L137) | **prefill** 模式：只调 chunked prefill 相关项，**不碰 radix** |

所以 P 侧 `self.tree_cache` 就是一棵**正常的、活的 `UnifiedRadixCache`**，
构造链、HiCache 挂载方式跟 mix 一模一样（见 1.1）。

事件循环换成 `event_loop_normal_disagg_prefill` / `event_loop_overlap_disagg_prefill`。

### 4.3 阶段 B：请求准入（走 bootstrap 队列，不走 waiting_queue）

| 步 | 位置 | 动作 |
|---|---|---|
| B1 | `scheduler.py:2906-2911` | 请求进 **`disagg_prefill_bootstrap_queue`**，不是 `waiting_queue` |
| B2 | `prefill.py:404-408` | **`max_new_tokens` 被强制为 1** —— P 只算 prefill + 采样 1 个 token |
| B3 | `prefill.py:371-381` | 读 D 传来的 `decode_prefix_len`：`req.start_send_idx = decode_prefix_len` |
| B4 | — | bootstrap 完成（握手 OK）后才转入 `waiting_queue`，之后与 mix 合流 |

**B3 极其重要，而且它完全不碰 `tree_cache`：**

> `decode_prefix_len` 是 D 侧告诉 P「这段前缀我本地已经有了（L1+L2+L3 命中），你别传」。
> P 拿到它只是把发送起点 `start_send_idx` 往后挪，**既不校验、也不查自己的树**。
> 也就是说：D 侧匹配算错了 → P 少传 KV → D 侧算出垃圾，**全程不报错**。
> 这是整条 PD + radix 链路上最脆的一环，详见 §7。

### 4.4 阶段 C：组批与 forward（与 mix 完全一致）

`get_new_batch_prefill` 这一段，P 与 mix 走的是**同一份代码**，所以 §3.3 的
C1–C18 全部原样适用：

- `check_hicache_events()` / `_retry_missed_storage_prefetches()`（C3/C4）
- waiting queue 全量 `match_prefix` 排序（C5–C8）
- `init_next_round_input` 权威 `match_prefix`（C11）
- 临时锁 → 算预算 → 解临时锁（C13）
- `init_load_back` + `ready_to_load_host_cache`（C14/C17）
- **持久准入锁 `_req_inc_lock_ref`**（C15）
- `prepare_for_extend` → `evict_from_tree_cache`（C18）

P 侧没有 decode 批次，所以 C19–C22（decode 内存检查、retract）不适用。

**一个 P 独有的插入点**：

| 步 | 位置 | 动作 |
|---|---|---|
| C-P1 | `scheduler.py:3897-3900` → `prefill.py:1124-1161` | **命中前缀的提前发送**：如果这次 `match_prefix` 命中的前缀已经在 device 上，不必等 forward，可以立刻开始 RDMA 发这一段 |

这条优化的前提正是 C15 的持久锁：因为前缀被锁住了，才敢在 forward 之外的时刻去读它。

### 4.5 阶段 D：forward 结果处理 —— **插树在这里**

```python
# disaggregation/prefill.py:761-763
req.output_ids.append(next_token_id)
maybe_cache_unfinished_req(req, self.tree_cache)     # ← 插树，立即发生
self.disagg_prefill_inflight_queue.append(req)
```

三点值得注意：

1. **调的是 `cache_unfinished_req` 而不是 `cache_finished_req`。** 因为此刻请求在 P 看来
   还没结束 —— `req.finished_reason` 直到 RDMA 完成才被赋值（见 4.6）。
   所以 P 侧请求在 forward 阶段**永远不会** `finished()`，
   §3.3 的 C23（`cache_finished_req` in result processing）在 P 侧走不到。
2. **锁在这里换了一次**（`cache_unfinished_req` 的原子交接，见 §2.1）：
   旧节点解锁 → 新节点上锁，`req.last_node` 指向新节点。锁本身没断过。
3. **`req_to_token_pool` 的行被改写指向树拥有的 slot**（`unified_radix_cache.py:1011-1014`）。
   这就是下一节的因果起点。

分块 prefill 的中间块也插树：

| 位置 | 动作 |
|---|---|
| `prefill.py:1089` | 每个 chunk 结束后 `maybe_cache_unfinished_req(req, self.tree_cache)` |

### 4.6 阶段 E：RDMA 传输完成轮询 —— **解锁在这里**

```python
# disaggregation/prefill.py:929-932
elif poll == KVPoll.Success:                          # 传输完成
    if not isinstance(req.finished_reason, FINISH_ABORT):
        req.finished_reason = FINISH_LENGTH(length=0)
    release_kv_cache(req, self.tree_cache)            # ← cache_finished_req：插树 + 解锁
```

**为什么必须等到这里才解锁？** 因果链是这样的：

```
cache_unfinished_req (:762)
  └─ req_to_token_pool.write(...)         unified_radix_cache.py:1011-1014
       └─ 请求的 req_to_token 行现在指向【树拥有的】KV 页

send_kv_chunk (:1323-1325)
  └─ 读的恰恰是这些 req_to_token 行 → RDMA 直接从这些页上搬数据

⇒ 如果在 RDMA 完成前解锁：
     这些页变成 evictable
       → 另一个请求分配时把它们淘汰、复用
         → RDMA 正在搬的内容被覆写
           → D 侧收到脏数据，静默出错
```

所以「P 侧锁的生命周期 = 准入 → RDMA 完成」不是设计选择，是正确性要求。

### 4.7 其余解锁路径（异常分支）

| 位置 | 场景 | 动作 |
|---|---|---|
| `prefill.py:1003` | `KVPoll.Failed` | 同样 `release_kv_cache` 解锁 |
| `prefill.py:1039-1048` | inflight 队列里被 abort | 解锁 + 出队 |
| `scheduler.py:3143` | bootstrap 阶段被 abort | `release_aborted_request` |
| `prefill.py:1352-1385` | `optimistic_release_and_requeue` | **插树 + 完全解锁**，并把 `start_send_idx` / `disagg_decode_prefix_len` 重置为 0，请求重新排队 |

最后一条是个有意思的设计：P 侧在某些情况下（例如 D 侧迟迟不来取）会主动放弃，
把已算好的 KV 留在树里（下次命中就能省掉重算），然后让请求重走一遍流程。
**这是 P 侧唯一一处「插树后立刻完全解锁」的地方。**

### 4.8 P 侧 HiCache 说明

P 侧 HiCache 行为与 mix 无差别：write_through 在 `cache_unfinished_req` /
`cache_finished_req` 里按 `hit_count >= write_through_threshold` 触发，
write_back 在 device leaf 淘汰时触发。因为 P 侧 `cache_finished_req` 被推迟到
RDMA 完成，**write_through 的第二次机会也随之推迟**，这是一个容易被忽略的时序副作用。

---

## 5. D 实例（decode）时间线

### 5.1 先弄清「D 侧的 tree_cache 到底是什么」

三种可能，看两个开关：

| `--disaggregation-decode-enable-radix-cache` | `--enable-hierarchical-cache` | `self.tree_cache` |
|---|---|---|
| 关（**默认**） | 任意 | `UnifiedRadixCache(disable=True)` + 一个**活的 L2 host 池**（仅供 retraction backup） |
| 开 | 关 | 真 radix 树，无 L2/L3 |
| 开 | 开 | 真 radix 树 + L2/L3（`enable_decode_hicache = True`） |

第一行就是 §1.1 讲的那条坑：**默认配置下不是 `ChunkCache`，是一个 disable 的 `UnifiedRadixCache`。**

另外，是否真的启用 D 侧 radix 还有一个**请求级**开关：

```python
# disaggregation/decode.py:1199-1202
use_decode_radix_cache = (
    self.enable_decode_radix_cache and not decode_req.is_rebootstrap
)
```

**重新 bootstrap 的请求不走 radix**（它的 KV 已经被丢弃，必须让 P 全量重传）。

### 5.2 阶段 A：事件循环的每轮顺序

`event_loop_normal_disagg_decode`（`decode.py:2450-2485` 一带）每轮大致是：

```
1. recv_requests / process_input_requests
2. process_decode_queue()              decode.py:2667   ← 准入 + HiCache 事件
3. prepare_for_prebuilt()              decode.py:2658   ← 假批次
4. process_prebuilt(future_map)        decode.py:2663   ← 唯一插树点
5. update_running_batch()                               ← 内存检查 / retract
6. run_batch (真 decode forward)
7. process_batch_result
```

注意 2/3/4 的相对顺序：**准入 → 假批次插树 → 才进 running batch**。

### 5.3 阶段 B：PreallocQueue 准入 —— **`match_prefix` 提前到这里**

这是 D 与 mix/P 最大的结构性差异。

| 步 | 位置 | 动作 |
|---|---|---|
| B1 | `decode.py:2668-2669` | `if self.enable_decode_hicache:` → **`tree_cache.check_hicache_events()`**（D 侧的事件排空点，对应 mix 的 C3） |
| B2 | `decode.py:1205` → `:654-670` | **`_match_prefix_and_lock(req)`**：`match_prefix_for_req` + `inc_lock_ref` |
| B3 | `decode_hicache_mixin.py:37-38` | 算出 **`decode_prefix_len = l1_device + l2_host + l3_storage`** |
| B4 | `decode.py:1803` | `evict_from_tree_cache(...)` —— 淘汰以腾出 prealloc 空间。**失败只 warn**（`:1806-1807`） |
| B5 | `decode.py:420-438` | `_reclaim_swa_tail_capacity()`（SWA 模型） |
| B6 | `decode.py:1505` | `decode_prefix_len` 随握手包**发给 P** |

**B2/B3 的目的和 mix 完全不同：**

> mix / P 做 `match_prefix` 是为了**少算一段 forward**。
> D 做 `match_prefix` 是为了**少传一段网络** —— 命中多少，就让 P 少发多少 RDMA。
>
> 所以它必须发生在**握手之前**（准入阶段），比 mix 的组批阶段早得多。
> 这也是为什么 `schedule_policy.py:252` 那条快速匹配的守卫里明确排除了
> `disaggregation_mode == "decode"` —— D 侧不需要、也不该在组批时再匹配一遍。

**B2 带来一条必须遵守的纪律**：`inc_lock_ref` 已经发生了，所以**准入的每一条失败/拒绝路径
都必须解锁**：

| 位置 | 场景 |
|---|---|
| `decode.py:412-418` | `_release_matched_prefix_lock(req)` 定义 |
| `decode.py:1286-1287` | 分配失败 |
| `decode.py:1290-1291` | 队列满 / 其他拒绝 |
| `decode.py:1308-1309` | abort |

漏掉任何一条 = 前缀永久锁死、永远不可淘汰 = 慢性泄漏。

### 5.4 阶段 C：L2/L3 本地回载状态机

`decode_prefix_len` 里 L2/L3 那部分，P 是不管的 —— 那段 KV 在 D 自己的 host / storage 里，
需要 D 自己搬回 device：

| 组件 | 作用 |
|---|---|
| `decode_hicache_mixin.py` | D 侧本地回载状态机：`init_load_back` → 轮询 → 完成 |
| `HiCacheRestoreGatedKVReceiver` | **把 RDMA 的 `Success` 降级成 `Transferring`**，直到本地 L2/L3 回载也落地，才真正对外报 `Success` |

后者是个巧妙的设计：RDMA 传的是后缀、本地回载搬的是前缀，两者并行，
但请求必须等**两者都完成**才能开始 decode。用一个包装 receiver 把这个「与」逻辑
藏在 poll 语义里，上层调度代码不用改。

`loading_check`（H→D 回载完成确认）在 D 侧**每轮被排空两次**：

| 位置 | 路径 |
|---|---|
| `decode.py:2668-2669` | `check_hicache_events()` 内部 |
| `unified_radix_cache.py:2894` | `is_load_back_event_done(...)` 内部又调一次 |

重复调用是幂等的（排空空队列即空转），但它说明这两条路径是各自独立演化出来的。

### 5.5 阶段 D：prebuilt 假批次 —— **唯一的插树点**

D 侧不跑 extend forward，但它仍然需要把 P 传来的 KV **登记进自己的树**。
这件事被塞进了一个「假批次」里：

```python
# disaggregation/decode_schedule_batch_mixin.py:113-123  process_prebuilt（简化）
for req in self.reqs:
    ...
    maybe_cache_unfinished_req(req, self.tree_cache)   # :121  ← D 侧唯一插树点
```

配套的两件事：

| 位置 | 动作 |
|---|---|
| `decode_schedule_batch_mixin.py:29` | `prepare_for_prebuilt` 把 `forward_mode = ForwardMode.PREBUILT` |
| `decode.py:2548-2557` | `_run_batch_prebuilt` **直接返回空结果**，不进模型 |

也就是说：`ForwardMode.PREBUILT` 这个"模式"根本没有对应的 kernel，
它是一个纯调度概念，唯一作用就是给 `cache_unfinished_req` 找一个合法的挂载时机。
（`scheduler.py:3893` 处短路，见 [decode_instance_prefill_capability.md](../04_pd_disaggregation/decode_instance_prefill_capability.md)。）

**第一个 token 从哪来？** 不是本地采样，是 P 侧采样后经 metadata buffer 传过来的：

| 位置 | 动作 |
|---|---|
| `decode.py:2147` | 从 metadata 取出 P 的采样结果，append 到 `req.output_ids` |

### 5.6 阶段 E：decode 每步

| 步 | 位置 | 动作 |
|---|---|---|
| E1 | `allocator/base.py:77` `check_decode_capacity` | **先淘汰再判断**是否要 retract |
| E2 | `allocation.py:485` `alloc_paged_token_slots_decode` | 分配前 `evict_from_tree_cache` |
| E3 | `schedule_batch.py:3580` | SWA 窗口滑过 → `dec_swa_lock_only(...)` |

E2 有个细节：淘汰目标量按 `len(seq_lens) * page_size` 估，**故意高估**（每个请求都按
一整页算，而实际大多只需 1 个 token）。宁可多淘汰一点，也不要在分配那一刻失败。

### 5.7 阶段 F：retract —— D 侧独有的 KV 备份

mix 的 retract 是「丢掉重算」。D 侧不能重算（它没有 prefill 能力），
所以必须**把整段 KV 搬到 host 存起来**：

| 步 | 位置 | 动作 |
|---|---|---|
| F1 | `unified_radix_cache.py:1236` | **`retraction_backup(req)`**：整段 KV device → host 池 |
| F2 | `unified_radix_cache.py:1272-1277` | `finish_event.synchronize()` —— **同步等待，阻塞调度线程** |
| F3 | `decode.py:846-851` | `resume_retracted_reqs()` → **`retraction_restore(req)`** 搬回 device |
| F4 | `unified_radix_cache.py:1330` | `retraction_discard(req)`：请求最终被 abort 时丢弃备份 |

两个后端（`--disaggregation-decode-retraction-backup`）：

| 后端 | 存哪 | 何时被选中 |
|---|---|---|
| `host_pool` | `UnifiedRadixCache` 的 L2 host 池 | 默认（不开 radix 时由 `resolve_decode_retraction_backup` 推断） |
| `cpu_tensor` | 独立的 CPU tensor | 显式指定 |

**这条路径与 radix 树本身是正交的** —— 即使 `disable=True`，host 池依然活着，
`retraction_backup` 依然工作。这就是 §1.1 那条 registry 分支存在的全部理由。

**真 retract（rebootstrap）是另一条路**：`offload_kv=False` 时完全跳过备份，
请求重新走 bootstrap，`decode_prefix_len` 发 0，让 P 全量重传。

### 5.8 阶段 G：结束与 abort

| 位置 | 动作 |
|---|---|
| `batch_result_processor.py:1198` | 正常结束 → `release_kv_cache(is_insert=...)` → `cache_finished_req` |
| `decode.py:2681-2683` | `retracted_queue` 有内容时**提前 `return`**，跳过新请求准入 —— 恢复优先于接新活，代价是新请求可能被饿死 |

---

## 6. 三者对照总表（按 API 维度）

| `UnifiedRadixCache` API | 集中式 mix | P 实例 | D 实例 |
|---|---|---|---|
| `match_prefix` | 组批阶段 `init_next_round_input`（`scheduler.py:3544`）+ 排序阶段全队列扫（`schedule_policy.py:255/338`） | 同 mix | **准入阶段** `_match_prefix_and_lock`（`decode.py:1205`→`:654`）；组批阶段被守卫排除 |
| `inc_lock_ref` | 临时锁（`schedule_policy.py:1028`）+ 持久准入锁（`:1342/1393`） | 同 mix | 准入即锁（`decode.py:654-670`） |
| `dec_lock_ref` | 结果处理（`cache_finished_req` 内） | **RDMA 完成轮询**（`prefill.py:932`） | 准入失败路径（`decode.py:412-418`）+ prebuilt 换锁 + 结束 |
| `cache_unfinished_req` | 结果处理（`batch_result_processor.py:346`）+ chunked 寄存（`scheduler.py:3104`） | 结果处理（`prefill.py:762`）+ 每 chunk（`:1089`） | **prebuilt 假批次**（`decode_schedule_batch_mixin.py:121`），唯一 |
| `cache_finished_req` | 结果处理（`batch_result_processor.py:343`/`:1198`） | **`KVPoll.Success`**（`prefill.py:932`）/ `Failed`（`:1003`）/ abort / `optimistic_release_and_requeue`（`:1352`） | 结果处理（`batch_result_processor.py:1198`） |
| `evict_for_alloc` | `allocation.py:155/187/485/260` + `allocator/base.py:77` | 同 mix（无 decode 分支） | 同 mix + **prealloc 前**（`decode.py:1803`，warn-only）+ SWA 尾部回收（`:420-438`） |
| `check_hicache_events` | `get_new_batch_prefill` 开头（`scheduler.py:3402`） | 同 mix | `process_decode_queue` 开头（`decode.py:2668`），`loading_check` 每轮**两次** |
| `init_load_back` / `ready_to_load_host_cache` | `schedule_policy.py:1289` / `scheduler.py:3642` | 同 mix | `decode_hicache_mixin.py` 本地回载状态机 |
| `prefetch_from_storage` | 准入时（`scheduler.py:2856`） | 同 mix | 由 `decode_prefix_len` 计算驱动 |
| `retraction_backup/restore/discard` | **不用**（重算） | 不适用 | `unified_radix_cache.py:1236/1283/1330` |
| `reset` / `take_events` / `attach_storage_backend` | 管理接口，三者一致 | 同 | 同 |

---

## 7. 关键洞察与易错点

### 7.1 D 侧默认不是 `ChunkCache`

最容易误判的一条。看到 `disable_radix_cache=True` 就以为是 `ChunkCache`，
于是推断「D 侧没有 host 池」——错。`registry.py:85-89` 那条分支返回的是
`UnifiedRadixCache(disable=True)`，行为像 chunk cache（`is_chunk_cache() → self.disable`，
`unified_radix_cache.py:541`），但**带着一个 `hicache_ratio=0.2` 的活 host 池**。
读 D 侧代码时如果假设 host 池不存在，会看不懂 `retraction_backup` 为什么能工作。

### 7.2 P 侧的锁必须活过 RDMA，这是正确性而非优化

见 §4.6 的因果链。核心是 `unified_radix_cache.py:1011-1014` 把
`req_to_token` 行改指向树拥有的页，而 `send_kv_chunk`（`prefill.py:1323-1325`）
读的正是这些行。**插树 = 交出所有权，持锁 = 保留读权限。**
任何试图「提前解锁以提高缓存利用率」的改动都会引入静默数据损坏。

### 7.3 `decode_prefix_len` 全链路无校验

D 算、P 信，中间没有任何一致性检查（P 侧 `prefill.py:371-381` 拿到就直接赋给
`start_send_idx`）。后果：

- D 侧 `match_prefix` 逻辑有 bug → 多报命中 → P 少传 → **D 用垃圾 KV 继续 decode**
- 表现是**输出质量下降**，不是异常、不是崩溃，非常难定位

排查这类问题的第一步应该是把 `decode_prefix_len` 打出来，和 D 侧实际
device 命中长度对比。

### 7.4 淘汰失败的报错点和真实原因点错位

`decode.py:1806-1807` 淘汰不足时**只打 warning 就继续**，
真正的失败要等到后面 `allocation.py:507-516` 才暴露出来。
所以看到 allocation 报错时，根因日志可能在几十行 warning 之前。

### 7.5 retraction 的 `synchronize()` 会卡住调度线程

`unified_radix_cache.py:1272-1277` 的 `finish_event.synchronize()` 是同步等待。
D 侧压力大、频繁 retract 时，这里会直接体现为**调度循环的 tail latency 抖动**。
restore 侧同理。

### 7.6 write_back 把 HiCache 写入放到了淘汰的关键路径上

`_evict_components` 里：device leaf 淘汰 → `_execute_and_commit_kv_backup(write_back=True)`
→ **同步 `writing_check(write_back=True)`** → 才 demote。
host 也满时退化为整棵子树丢弃（并打 warning）。
所以 `write_back` 策略下，一次分配触发的淘汰可能同步等一次 D→H 拷贝。

### 7.7 每个调度步都对整条 waiting queue 做 `match_prefix`

`schedule_policy.py:255`（快速匹配）和 `:338`（LPM/DFS_WEIGHT）都是全队列遍历。
队列长时这是实打实的 CPU 开销，且与 batch size 无关、只与队列长度有关。
mix 和 P 都有这个成本，D 侧因为守卫排除而没有。

### 7.8 `SchedulePolicy` 里还有第二棵树

`schedule_policy.py:238` 另建了一棵**模拟 `RadixCache`**，用于批内前缀去重排序
（`:350/371` 对它做 match + insert）。它**不是** `self.tree_cache`，
调试时看到 radix 相关调用要先分清是哪一棵。

### 7.9 `tree_cache.disable` 会静默降级调度策略

`scheduler.py:1282` → `schedule_policy.py:314`：若 `tree_cache.disable`，
cache-aware 策略（lpm / dfs-weight）**自动降级为 FCFS**。
在 D 侧默认配置下这总是发生 —— 指定 `--schedule-policy lpm` 不会报错，只是不生效。

### 7.10 一处规范违反

`decode.py:465-469` 用了 `getattr` 做防御式取值，与本仓库
`.claude/rules/no-getattr-defensive.md` 的约定冲突。改这一带代码时可以顺手清掉。

---

## 8. 复习用：三条时间线并排

```
                mix                    P                       D
              ───────                ─────                   ─────
准入      ┌ prefetch_from_storage  ┌ (bootstrap queue)     ┌ check_hicache_events
          │                        │ start_send_idx ←      │ match_prefix + inc_lock_ref
          └                        └   decode_prefix_len   │ evict_for_alloc (warn-only)
                                                           └ 发出 decode_prefix_len → P

排序      ┌ match_prefix × 全队列    同 mix                  （守卫排除，不做）
          └ 模拟树 match + insert

组批      ┌ check_hicache_events   ┌ 同 mix                 （无组批阶段）
          │ match_prefix (权威)     │ + 命中前缀提前发 RDMA
          │ inc_lock_ref (持久)     │
          │ init_load_back         │
          └ evict_for_alloc        └

forward     extend forward           extend forward          ✗（PREBUILT，空结果）

插树      ┌ cache_unfinished_req   ┌ cache_unfinished_req   ┌ cache_unfinished_req
          └   @结果处理             └   @结果处理 :762        └   @prebuilt 假批次

decode      每步 evict + retract     （无）                   每步 evict + retract
                （丢弃重算）                                   （retraction_backup 到 host）

结束/解锁   cache_finished_req       cache_finished_req       cache_finished_req
            @结果处理                @RDMA Success :932        @结果处理
```
