# HiCache buffer_only 模式（buffer mode）系统梳理

> 基线 commit：`33ed29a0ee`
> 核心代码：[python/sglang/srt/mem_cache/buffer_mode/](../../../python/sglang/srt/mem_cache/buffer_mode/)
> 相关文档：[hicache_usage_and_design.md](hicache_usage_and_design.md)、
> [unified_radix_cache_architecture.md](unified_radix_cache_architecture.md)、
> [host_cache_off_tree_staging_scheme.md](host_cache_off_tree_staging_scheme.md)

---

## 0. 一句话结论

`--hicache-host-memory-mode buffer_only` 把 **主机内存（CPU RAM）从"L2 缓存层"降级成"GPU 与 L3 存储之间的一次性中转缓冲区"**。

- 常规 `cache` 模式：GPU(L1) → Host(L2，**长期驻留、可被 match 命中**) → Storage(L3)。radix 树节点上会挂 `host_value`，主机副本本身就是一层缓存。
- `buffer_only` 模式：GPU(L1) → Host(**bounce buffer，用完即还**) → Storage(L3)。**radix 树上永远不存在 host_value**，主机内存只是 DMA 中转站，寿命等于一次传输操作。

一句话记法：**cache 模式的主机内存是"仓库"，buffer 模式的主机内存是"传送带"。**

模块自带的 docstring 已经把这句话说清楚了（[buffer_mode/__init__.py:1-9](../../../python/sglang/srt/mem_cache/buffer_mode/__init__.py#L1-L9)）：

```
Host RAM is a transient staging buffer between the GPU and the L3 storage
backend, never an L2 cache tier: writes stage device KV through op-owned
host bounces into storage and free them at the storage ack; reads fetch
storage hits into op-owned bounces and publish them into the device tree at
prefill admission.
```

### 0.1 为什么要这么做

| 动机 | 说明 |
|---|---|
| 主机内存贵/紧张 | 大规模部署里 CPU RAM 常常是稀缺资源。cache 模式默认 `hicache_ratio=2.0`（主机池 = 2× 设备池）；buffer 模式默认只要 `1.2`。 |
| L3 才是真正的共享缓存 | 有 Mooncake / 3FS / 分布式文件后端时，**跨实例共享的命中率来自 L3**，本机 L2 的边际收益低，却要吃掉全部 RAM。 |
| 简化一致性 | 主机副本不进树 ⇒ 不需要 host 侧的 LRU、host lock ref、host eviction 等一整套机制（`evict_host()` 在 buffer 模式直接 `return 0`，见 [unified_radix_cache.py:1117-1127](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1117-L1127)）。 |

代价：**每次 L1 未命中都必须走 L3**，没有"本机 L2 兜底"。所以 buffer 模式**强制要求配置 L3 后端**。

---

## 1. 与 cache 模式的逐项对比

| 维度 | `cache`（默认） | `buffer_only` |
|---|---|---|
| 主机副本是否进 radix 树 | 是，节点挂 `host_value`，`match_prefix` 能命中 | **否**，树上永不出现 host value |
| 主机 slot 归属 | 树节点拥有，靠 host eviction 回收 | **operation-owned**（谁发起传输谁持有），ack 即释放 |
| `evict_host()` | 走 `tree_core.drive_host_eviction` | 直接返回 0（无可驱逐对象） |
| 写策略 | write_through / write_through_selective / **write_back** | **禁止 write_back** |
| L3 后端 | 可选 | **必须配置** |
| 默认 `hicache_ratio` | 2.0 | **1.2** |
| prefetch 容量上限 | `0.5 × 主机池` | `0.9 × 主机池` |
| 限流依据 | `prefetch_tokens_occupied` 计数 | **主机池真实占用** − 写入 staging |
| 占用额度（occupancy）授予时机 | prefetch 入队时按"请求跨度"预留 | **命中分配时**按"实际命中长度"授予 |
| L3 命中后落地位置 | 先落 host（进树，成为 L2），再按需 H2D | 停在 **op-owned bounce**，**prefill 准入时**才 H2D + insert |
| 主机池不足时的降级 | 缩短到更短前缀（部分命中） | **park 排队重试**，坚持要完整命中 |
| 写去重依据 | 树上 `is_backuped` 标记 | **`StorageExistenceCache`（存在性信念）** |
| PD decode 实例 | 支持 | **禁止** |
| Mamba / SSM 模型 | 支持 | **禁止**（只支持 FULL / FULL+SWA） |
| DSv4 sidecar 压缩池 | 支持 | **支持**（#37424 起，sidecar 骑在源池 transient slot 上一起 stage/free） |

---

## 2. 配置面：怎么开、开了之后什么被禁

### 2.1 主开关

[server_args.py:2722-2729](../../../python/sglang/srt/server_args.py#L2722-L2729)：

```python
hicache_host_memory_mode: A[
    str,
    Arg(
        help="Whether host memory is a persistent HiCache tier (cache) or a transient staging buffer between GPU and the storage backend (buffer_only). buffer_only requires --hicache-storage-backend.",
        choices=["cache", "buffer_only"],
    ),
    NS("memory"),
] = "cache"
```

只有两个取值。运行期通过 `get_memory().hicache_host_memory_mode` 读取
（`NS("memory")` 把它投影进 `memory` 命名空间 config bag，见
[runtime_context.py:1296](../../../python/sglang/srt/runtime_context.py#L1296)）。

一条最小可用命令行：

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --page-size 64 \
    --enable-hierarchical-cache \
    --hicache-host-memory-mode buffer_only \
    --hicache-storage-backend file \
    --hicache-write-policy write_through \
    --hicache-ratio 1.2
```

### 2.2 参数解析期的 3 道硬闸

全部在 [arg_groups/hicache_hook.py:167-211](../../../python/sglang/srt/arg_groups/hicache_hook.py#L167-L211)
的 `validate_hicache_host_memory_mode`（由 `handle_hicache` 调用）：

| 闸门 | 行号 | 行为 |
|---|---|---|
| 取值不在 `{cache, buffer_only}` | `:169-173` | `ValueError` |
| 主机池未定尺（`hicache_size<=0` 且 `hicache_ratio is None`） | `:178-187` | `ValueError`（兜底，上游已默认过） |
| **未配置 `--hicache-storage-backend`** | `:192-197` | `ValueError` — 主机只是中转站，数据必须有地方落 |
| **`--hicache-write-policy write_back`** | `:198-203` | `ValueError` — write_back 依赖主机长期驻留 |
| **`--disaggregation-mode decode`** | `:204-211` | `ValueError` — D 侧 prefetch/offload 路径绕过本 pipeline |

> P（prefill）实例是允许开 buffer 模式的，只有 D 实例被拒。

### 2.3 缓存构造期的 3 道硬闸

| 闸门 | 位置 | 理由 |
|---|---|---|
| 必须是 `UnifiedRadixCache` | [registry.py:216-226](../../../python/sglang/srt/mem_cache/registry.py#L216-L226) | 只在 unified 树上实现了 |
| 树组件只能是 `{FULL, SWA}` 子集（**Mamba 被拒**） | [unified_radix_cache.py:388-403](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L388-L403) | Mamba 在"准入时回载"读路径上没有 state 交接通道，也不是 layer-gated |
| `validate_buffer_only_stack`：校验 sidecar 池映射 / 拒无 host 池的 SWA / SWA host 池 ≥ 2 个窗口 | [pipeline.py:158-211](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L158-L211) | 见下 |

`validate_buffer_only_stack` 三条的具体含义：

1. **sidecar 池**（DeepSeek-V4 的压缩区）：**#37424 起已支持**。sidecar 复用其源池的
   transient slot id，因此每个 sidecar host 池必须暴露完整的源池 slot 命名空间——校验
   只要求映射完整（`source`/`sidecar` 都能在 `entry_map` 里找到）且 sidecar host 池
   不小于源池，满足即放行（[pipeline.py:170-187](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L170-L187)）。
2. **SWA + `unified_kv` 布局**：这种布局下 SWA 是"纯设备环形缓冲"，没有 host 池，
   既没法 stage 写、也没法 fetch 读 → 拒。
3. **SWA host 池必须 ≥ 2 个完整窗口**（`2 × full_window_pages × page_size`）：
   一个窗口用于承载正在写出的 staging，另一个窗口是"读优先保留额度"
   （`_aux_loads_margin` 的下界就是一个窗口）。低于 2 个窗口，**每一个带窗口的写意图
   都会被判为 oversize 而丢弃，SWA 的 L3 覆盖率会静默变成 0**。

### 2.4 联动参数一览

| 参数 | 与 buffer_only 的关系 |
|---|---|
| `--hicache-ratio` | **默认从 2.0 变成 1.2**（[hicache_hook.py:60-68](../../../python/sglang/srt/arg_groups/hicache_hook.py#L60-L68)）。理由：buffer 模式"在途中转"而非"长期保留"，只需覆盖写积压 + park 住的 prefetch |
| `--hicache-size`（GB） | 优先于 ratio；给 buffer 模式一个绝对 staging 预算的唯一手段 |
| `--hicache-storage-backend` | **必填** |
| `--hicache-write-policy` | 只能 `write_through` / `write_through_selective` |
| `--hicache-io-backend` / `--hicache-mem-layout` | 无 buffer 专属规则（通用的 layout↔IO 归一化照常） |
| `--hicache-storage-prefetch-policy` | 无专属规则；但 `wait_complete` 与 buffer 模式的"park 重试"配合最自然 |
| `--dcp-size > 1` | DCP + storage backend 本身就 `NotImplementedError`，间接排除 buffer 模式 |
| `SGLANG_ENABLE_HICACHE_BUFFER_ANCHOR_LOCK`（默认 False） | buffer 专属：见 §7 锚点锁 |
| `SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP`（默认 0.5） | 同上，锚点锁总额度占设备池比例 |

（env 定义在 [environ.py:726-729](../../../python/sglang/srt/environ.py#L726-L729)）

---

## 3. 总体结构：一个对象 + 一个分派测试

buffer 模式的**所有**状态都收在一个对象里：`BufferModePipeline`
（[pipeline.py:202](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L202)）。

```
UnifiedRadixCache
├── self.buffer_pipeline: Optional[BufferModePipeline]   # None = cache 模式
│                                                        # unified_radix_cache.py:256
├── self.storage_existence_cache: StorageExistenceCache  # 只有 buffer 模式维护
└── self.cache_controller (HiCacheController)
        └── host_write_staged_tokens_fn = lambda: buffer_pipeline.write_staged_tokens_
```

构造点只有一处：`init_hicache`
（[unified_radix_cache.py:446-467](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L446-L467)）：

```python
if self.host_memory_mode == "buffer_only":
    swa = self.components.get(ComponentType.SWA)
    validate_buffer_only_stack(...)
    self.buffer_pipeline = BufferModePipeline(
        cache=self,
        max_context_len=get_model().context_length or 0,
        swa_window_pages=(swa.full_window_pages if ... else 0),
        write_backlog_cap=2 * self.token_to_kv_pool_allocator.size_full,
    )
    self.cache_controller.host_write_staged_tokens_fn = (
        lambda: self.buffer_pipeline.write_staged_tokens_
    )
```

**关键约定：`cache.buffer_pipeline is not None` 就是全代码库唯一的模式分派测试。**
遍布 `unified_radix_cache.py` 的 20 多处 `if self.buffer_pipeline is not None:`
就是两套语义的分叉点（完整清单见 §11）。

这是一个**紧耦合的协作者**（intimate collaborator）：pipeline 通过持有的 `self._cache`
反向调用树的 `insert / match_prefix / evict_for_alloc / inc_lock_ref / dec_lock_ref /
_apply_cache_action`。所以它不是一个独立组件，而是"从 UnifiedRadixCache 里抽出来的
buffer 模式那一半"。

### 3.1 状态字段全景（`reset()`，[pipeline.py:248-278](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L248-L278)）

**读路径（load back）状态：**

| 字段 | 类型 | 含义 |
|---|---|---|
| `pending_hit_allocs` | `deque[PrefetchOperation]` | L3 已确认命中、但主机 staging 分配不到，**park 在这里等重试**（FIFO） |
| `_prefetch_prefix_ctx` | `rid → (prefix_tokens, extra_key, cache_salt)` | prefetch 入队时的设备前缀快照 + 树 key 命名空间 |
| `staged_prefetches` | `rid → _StagedPrefetch` | **已从 L3 取回、停在主机 bounce 上、等 prefill 准入消费** |
| `ongoing_buffer_load_back` | `ack_id → _OngoingBufferLoadBack` | H2D 在飞（ack_id 是合成的**负数**，见 §6.5） |

**写路径（backup）状态：**

| 字段 | 类型 | 含义 |
|---|---|---|
| `pending_write_queue` | `deque[_UnifiedBackupIntent]` | 已准入、等 D2H 名额的意图（**FIFO，head-of-line**） |
| `inflight_backup_node_ids` | `set[NodeId]` | 任何在飞阶段的 node id，用于去重重复触发 |
| `inflight_backup_hashes` | `dict[hash → refcount]` | D2H 已发起、storage ack 未到的**内容**引用计数 |
| `ongoing_write_through` | `node_id → _UnifiedBufferBackupEntry` | D2H 发起 → D2H ack 之间 |
| `ongoing_backup` | `operation_id → _UnifiedBufferBackupEntry` | storage 写发起 → storage ack 之间 |
| `write_staged_tokens_` | `int` | 当前被写路径占住的主机 token 数（喂给 controller 限流） |
| `write_backlog_tokens_` | `int` | 排队中（尚未 D2H）的 token 数 |

**锚点锁状态：** `anchor_locks: rid → _AnchorLock`、`anchor_locked_tokens_`。

三个数据结构（都是 `msgspec.Struct`，符合仓库 no-dataclass 规范）：

- `_UnifiedBackupIntent`（`:65`）：只装一个 `BufferBackupSnapshot`。**排队期间不加锁不占 slot**，
  靠快照里的 key 长度 / hash 链事后检测节点是否被 split 或 evict。
- `_UnifiedBufferBackupEntry`（`:77`）：意图 + `host_indices`（KV staging slot）+
  `aux_xfers`（SWA 窗口等辅助池 staging）+ `lock_params`（设备锁）。
- `_StagedPrefetch`（`:91`）：**只有主机 bounce，没有任何设备状态，树上也什么都没有**。
  带 `key_tokens`（前缀 + 命中跨度的完整 token 序列）、`matched_len`、`num_tokens`、
  `occupied_tokens`、`hash_values`、`operation_id`。

---

## 4. 写路径（backup pipeline）：设备 → staging → 存储

### 4.1 四个阶段

```
                     enqueue_backup_intent()
   BackupKV action ───────────────────────────► pending_write_queue
   （树上某节点变成                 4 道准入闸        （只有元数据快照，
     "该落盘"了）                                    不占 slot、不加锁）
                                                            │
                                     flush_pending_writes()  │ 每 tick，head-of-line
                                                            ▼
                                              cc.write()  = D2H 异步拷贝
                                              + 主机 staging 分配
                                              + 设备节点 inc_lock_ref
                                                            │
                                                    ongoing_write_through
                                                            │
                                     finish_backup_ack()     │ D2H 完成
                                                            ▼
                                              dec_lock_ref（设备可以驱逐了！）
                                              + cc.write_storage() 发起 L3 写
                                                            │
                                                     ongoing_backup
                                                            │
                                  finish_storage_write_ack() │ L3 写完成
                                                            ▼
                                       storage_existence_cache.add(hashes)
                                       + _free_staging_now()  ← 主机 slot 归还
```

对比 cache 模式：cache 模式在 D2H ack 时会 `tree_core.commit_backup(node_id, host_indices)`
把主机 slot **挂到树节点上**成为 L2；buffer 模式在
[pipeline.py:552-553](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L552-L553)
明确注释 **"no commit_backup"** — 节点绝不能显示为 host-resident，staging slot 只活在 entry 里。

### 4.2 入口：`_execute_and_commit_kv_backup` 的分叉

[unified_radix_cache.py:1336-1347](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1336-L1347)：

```python
if self.buffer_pipeline is not None:
    # Buffer mode bypasses the host-backup contiguity below: nothing
    # is ever host-backuped here. Contiguity comes from end-to-end
    # FIFO ordering instead (BackupKV chains are parent-before-child
    # and every pipeline stage drains in order).
    for node_id in action.node_ids:
        self.buffer_pipeline.enqueue_backup_intent(node_id)
    return 0
```

注意这里的**前缀连续性**保证换了机制：cache 模式靠"父节点必须已 host-backuped"这个
树上不变量；buffer 模式树上没有这个标记，改为靠**端到端 FIFO 顺序** +
`_backup_parent_covered` 的存在性信念检查。

### 4.3 准入 4 道闸（`enqueue_backup_intent`，[pipeline.py:309-366](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L309-L366)）

**所有丢弃都是静默的**（只记 `log_backup_dropped_tokens` 指标），节点会在下一次被命中时重新触发。

| # | 闸门 | 代码 | 为什么 |
|---|---|---|---|
| 0 | 已在飞（`node_id in inflight_backup_node_ids`） | `:315-316` | 去重 |
| 0 | 快照失败（节点已死） | `:317-321` | 无意义 |
| 1 | **信念覆盖（belief skip）** | `:325-330` | `storage_existence_cache.covers_all(KV, hashes, extra_cover=inflight_backup_hashes)` — 内容已认为在 L3、或已有另一份 D2H 在飞，就不重复写 |
| 2 | **积压上限** `write_backlog_tokens_ >= write_backlog_cap` | `:332-349` | cap = `2 × 设备 FULL 池`，是**纯泄漏兜底**。触发即视为 bug（记 `logger.error`），因为活跃积压本质被设备池跨度限制住 |
| 3 | **父节点未覆盖**（`_backup_parent_covered`） | `:357` | 在一个被丢弃的父节点之上写子节点，会在 L3 里造成**永久的最长前缀空洞**：以后任何请求都无法从这段命中 |
| 4 | **oversize**（`_backup_oversize`） | `:357-359` | 跨度大于某个池的"可写容量"，永远 stage 不下去，准入了会**永久堵死 FIFO 队头** |

`_backup_parent_covered`（`:289-302`）的判定是三选一：父是 root / 父也在飞 / 父的最后一个
page hash 在存在性信念里。

`_backup_oversize`（`:401-424`）：KV 池比总容量，**辅助池（SWA）比"总容量 − loads margin"**，
与 `_aux_budget_blocked` 的准入天花板保持一致 —— 否则会出现"准入了但永远分配不到"的死锁。

### 4.4 出队：`flush_pending_writes`（[pipeline.py:468-527](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L468-L527)）

每 tick 从 `check_hicache_events` 调用（[unified_radix_cache.py:2857-2858](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2857-L2858)）。

第一步 `_sweep_stale_backup_intents()`（`:447-466`）：**扫描整个队列**（不只队头）剔除失效意图。
失效判定在 `tree_core.validate_buffer_backup`：

- arena 查不到 → 节点已删除
- key 长度与快照不符 → 节点被 **split** 了
- FULL 设备值为 None → 已被 **evict**

不扫全队列的后果：死意图会虚增积压计数，并且霸占 FIFO 位置挡住活跃段。

第二步计算**写窗口 `live_cap`**（`:478-482`），这是"读优先"策略的核心：

```python
pool_tokens = cc.mem_pool_host.size
live_cap = max(
    int(HICACHE_WRITE_STAGING_POOL_FRACTION * pool_tokens),   # 0.2，地板
    pool_tokens - cc.prefetch_tokens_occupied - pool_tokens // 10,  # 动态
)
```

含义：**写是可延迟的，读不是**。写路径最多用"主机池 − 读占用 − 10% 缓冲"，
但保底至少有 20% 可写（否则读压力大时写路径会完全饿死，L3 覆盖率归零）。

第三步 head-of-line 循环，对队头依次判断：

| 情况 | 动作 |
|---|---|
| 父覆盖丢失 | **popleft 丢弃**（把父的丢弃级联下去，而不是造 L3 空洞），继续下一个 |
| `write_staged_tokens_ >= live_cap` | **break**（让位给读，下轮重试） |
| oversize（用真实 `comp_xfers` 重算） | **popleft 丢弃**（永久不可 stage 的队头不能堵队） |
| `_aux_budget_blocked`（SWA 池没 headroom） | **break**（在闸门让位，而不是进 `cc.write` 里分配失败） |
| `_launch_backup_intent` 返回 False（主机池满） | **break**（等 ack 释放 slot） |
| 成功 | `popleft`，继续 |

**break 与 popleft 的区别很关键**：break = 暂时不行，保留 FIFO 位置；
popleft = 永久不行，丢弃。混淆任何一个都会导致队列永久卡死或 L3 出现空洞。

### 4.5 `_aux_loads_margin`：辅助池的读优先保留额度

[pipeline.py:426-434](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L426-L434)：

```python
return max(
    self._swa_window_pages * host_pool.page_size,   # 至少一个完整窗口
    host_pool.size // 10,                          # 或 10% 突发吸收
)
```

下界必须是"一个完整 SWA 窗口"，因为 `prepare_prefetch` 会在这个池里分配整窗，
**分配失败会导致整个 L3 prefetch 被作废**（不是降级，是全丢）。
这也是 §2.3 里"SWA host 池 ≥ 2 窗口"那道闸的来源。

### 4.6 D2H ack：`finish_backup_ack`（[pipeline.py:614-645](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L614-L645)）

```python
entry = self.ongoing_write_through.pop(ack_id)
self._cache.dec_lock_ref(snapshot.node_id, entry.lock_params)   # ← 设备锁可以放了
...
operation_id = cc.write_storage(entry.host_indices, snapshot.key.token_ids,
                                snapshot.hash_values, snapshot.prefix_keys,
                                extra_pools=storage_xfers or None)
self.ongoing_backup[operation_id] = entry
```

**为什么这里能放设备锁**：L3 写读的是主机 staging 副本，不再碰设备内存，所以设备侧的
KV slot 可以随时被驱逐/复用了。这正是"中转缓冲"的价值 —— 把设备侧的持锁时间压到最短。

辅助池的 key 规则（`_aux_window_keys`，`:597-612`）：**每个辅助池写一份"尾部快照"，
用它覆盖的最后 N 个 KV page hash 作为 key**。SWA 窗口按 `page_size` 分页所以有多个 key；
Mamba state 是单 slot（host 池 page_size=1）所以只有一个 key。`hit_policy=TRAILING_PAGES`。

### 4.7 存储 ack：`finish_storage_write_ack`（[pipeline.py:647-662](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L647-L662)）

```python
entry = self.ongoing_backup.pop(operation_id, None)
if entry is None:
    return                                     # 不属于本 pipeline（reset 之后的迟到 ack）
self._cache.storage_existence_cache.add(PoolName.KV, snapshot.hash_values)
self._free_staging_now(entry.host_indices, entry.aux_xfers)
self.write_staged_tokens_ -= len(entry.host_indices)
self.inflight_backup_node_ids.discard(snapshot.node_id)
_untrack_content_refs(self.inflight_backup_hashes, snapshot.hash_values)
```

两个细节：

1. **存在性条目无条件写入**，不看 `completed_tokens`。因为后端故障时各 TP rank 的
   `completed_tokens` 可能不同，而存在性缓存要驱动准入决策，**必须 TP 一致**。
   宁可留一个 stale positive（后面靠 §5 的 hit-query 反馈治愈），也不能让 rank 分叉。
2. `_free_staging_now`（`:664-682`）是**同步释放**，直接调 `mem_pool_host.free()`，
   不走 `append_host_mem_release` 队列。原因：buffer 模式所有 ack/drop 都在 scheduler
   线程上跑，同步释放能保证**本 tick 后续闸门读到的池可用量是最新的**。

### 4.8 内容引用计数：`inflight_backup_hashes`

`_track_content_refs` / `_untrack_content_refs`（`:130-145`）维护的是
**page hash → 引用数**，不是布尔标记。原因：同一份内容可能被多个 entry 同时 staging
（一段被重新发布的 span 在新 node id 下再次触发写）。准入的"launched cover"
（`extra_cover=self.inflight_backup_hashes`）就是靠它，让重新发布的内容在原写入排空
之前不重复写。

---

## 5. 存在性信念缓存 `StorageExistenceCache`

[buffer_mode/storage_existence_cache.py](../../../python/sglang/srt/mem_cache/buffer_mode/storage_existence_cache.py)

**它解决的问题**：cache 模式靠树上的 `is_backuped` 标记知道"这段已经落 L3 了"。
buffer 模式树上什么都不留，如果没有本地信号，**每次热前缀重新 insert 都会重新 D2H + 重写 L3**。

结构极简：一个有界 LRU，`OrderedDict[(pool, page_hash) → None]`，
上限 `HICACHE_EXISTENCE_CACHE_MAX_ENTRIES = 128 * 1024`（约 131K 条，≤30MB，
在 page_size=64 时覆盖约 8M KV token）。

**语义是"信念"而非"真相"（advisory, not authoritative）：**

| 情况 | 后果 |
|---|---|
| 命中 | 跳过冗余的 D2H + L3 写 |
| **stale positive**（后端把数据淘汰了，信念还在） | 白跳过若干次写回，直到某次 prefetch 的 hit-query 反馈把条目作废；下一次 insert 重新写。**永远不是正确性问题**，最坏就是一次冷重算 |
| miss（LRU 淘汰了或从未见过） | 一次冗余写（storage key 是内容寻址的，写入幂等） |

Key 用的是 insert 时**已经算好的链式 page hash**，所以：查询不需要重新哈希；
条目能**跨越节点删除、split、重算**存活（相同 token ⇒ 相同哈希链）。

### 5.1 治愈机制：`invalidate_beyond`

[unified_radix_cache.py:2154-2165](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2154-L2165)：

```python
def _invalidate_absent_from_hit_query(self, operation) -> None:
    if self.host_memory_mode != "buffer_only":
        return
    chain = operation.all_hash_values
    if chain is None:
        return
    self.storage_existence_cache.invalidate_beyond(
        PoolName.KV, chain, keep_pages=operation.storage_hit_count // self.page_size
    )
```

每次 prefetch 的存在性查询回来时，**用真实的后端应答当 ground truth**：
把"可用切点（folded usable cut）"之后的所有信念条目丢掉。下一次 insert 就会重写这一段，
同时关掉 stale positive 和辅助池在切点上的空洞。

**注意这是一次 FULL 池的检查治愈所有池**（KV 信念作废 ⇒ 整个节点包括 SWA 都会重写）。

### 5.2 API 速查

| 方法 | 语义 |
|---|---|
| `add(pool, hashes)` | 写入信念（storage ack、prefetch 取回时） |
| `contains(pool, h)` | 查单页，**会 LRU touch** |
| `contains_all` / `covers_all(..., extra_cover)` | 全覆盖判定；`covers_all` 额外接受"已发起 D2H 的内容"作为覆盖 |
| `invalidate_beyond(pool, hashes, keep_pages)` | ground truth 治愈 |
| `clear()` | flush_cache 时清空 |

**TP 确定性**：所有变更都在 scheduler 线程的 lockstep 点上（storage-ack drain、
prefetch-hit drain、fill commit），输入都经过跨 rank 归约。docstring 明确写了
"Do not touch it from anywhere else"。

---

## 6. 读路径（load back pipeline）：存储 → staging → 设备

这是 buffer 模式**最复杂**的部分，因为它把"L3 命中落地"从"后台异步插树"改成了
**"停在主机 bounce 上，等 prefill 准入时才一次性搬进设备并插树"**。

### 6.1 五个阶段总览

```
① prefetch_from_storage()            请求到达，发起 L3 存在性查询
        │                            buffer: 不给 anchor 加 host lock
        │                                    立刻 try_lock_anchor()（设备锁）
        │                                    set_prefix_ctx() 记下前缀快照
        ▼
② _try_alloc_storage_hit()           L3 应答"命中 N token"
        │                            buffer: 4 道 IO-commit 闸（§6.3）
        │                                    分配 hit 大小的主机 bounce
        │                                    prefetch_tokens_occupied += N
        ▼
③ stage_completed_prefetch()         数据已经躺在主机 bounce 里
        │                            → staged_prefetches[rid]
        │                            **树上没有任何东西，设备上没有任何东西**
        ▼
④ scheduler 准入（get_new_batch_prefill）
        │                            plan_staged_splice() → req.host_hit_length
        │                                                   req.swa_host_hit_length
        ▼
⑤ init_load_back()                   设备分配 + layer-gated H2D + 普通 insert
        │                            → ongoing_buffer_load_back[负 ack_id]
        ▼
⑥ try_finish_load_back()             H2D ack：释放主机 bounce
```

对比 cache 模式：cache 模式在阶段 ③ 就把主机 slot **插进树**（成为 L2），
后续任何请求都能 match 到；设备侧的 H2D 是独立的 `load_back`。
buffer 模式的 staged prefetch **对 `match_prefix` 完全不可见**，
只对"那个特定的 rid"可见。

### 6.2 阶段 ①：发起（`prefetch_from_storage`）

[unified_radix_cache.py:1700-1835](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1700-L1835) 里的 buffer 分支：

| 行 | 差异 |
|---|---|
| `:1730` | **buffer 模式跳过 `prefetch_rate_limited()` 前置检查** —— 限流推迟到"已知命中"之后再判（阶段 ②），避免在还不知道有没有命中时就放弃 |
| `:1734-1739` | 额外检查 `buffer_pipeline.has_staged(req_id)` —— 已有未消费的 hold，重复发起会泄漏它的 staging slot |
| `:1743-1747` | **不给 anchor 加 host lock**（`anchor_lock_params = None`）：buffer 模式在 fetch 期间树上没有任何状态，buffer 是 operation-owned 的，anchor 不需要钉 |
| `:1821-1831` | `set_prefix_ctx(rid, matched_prefix_tokens, extra_key, cache_salt)` + 立刻 `try_lock_anchor(rid)` |
| `:1832-1835` | **cache 模式在这里就 `prefetch_tokens_occupied += len(prefetch_key)`（按请求跨度预留）；buffer 模式不加**，推迟到阶段 ② 按实际命中长度授予 |

`set_prefix_ctx` 记的三元组 `(prefix_tokens, extra_key, cache_salt)` 后面有两个用途：
阶段 ③ 拼装完整 span 的树 key；`try_lock_anchor` 重新 match 定位已失效的 anchor。

### 6.3 阶段 ②：IO commit（`_try_alloc_storage_hit`）

[unified_radix_cache.py:2272-2336](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2272-L2336)。
"L3 说命中了 N 个 token，现在决定要不要真的把它读回来"。buffer 模式在这里有 4 个专属分支：

| # | 闸门 | 行 | 行为 |
|---|---|---|---|
| 1 | `cc.prefetch_rate_limited()` | `:2284-2288` | 主机池被读占满：**park 住这个已知命中**，返回 False。op 留在 `ongoing_prefetch` 里，所以 `wait_complete` 策略仍会挡住准入 |
| 2 | `try_lock_anchor(rid) == "anchor_lost"` | `:2289-2299` | 拼接基点（splice base）已经没了 —— **这个 fetch 不值得它的 storage 读**。撤销 + 武装 paced retry（从更短的新 match 重试） |
| 3 | `staged_span_covered(rid, hit_count)` | `:2300-2307` | 活跃设备树**已经完整覆盖**这个 span：读回来也没东西可拼。跳过 storage 读，计入 `declined_device_covered` |
| 4 | 主机 bounce 分配失败 | `:2313-2325` | **cache 模式降级到更短前缀；buffer 模式直接返回 False 去 park**，坚持要完整命中 |

注意 2、3 是**在 bounce 分配和 storage 读之前**执行的 —— 取消的成本因此是一个纯 revoke，
不涉及任何已分配资源。这是"IO commit"这个命名的含义：这里才是真正决定花 I/O 的点。

`staged_span_covered`（[pipeline.py:760-779](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L760-L779)）
的做法是把 `prefix_tokens + span_key[:span_tokens]` 拼成完整 key 去 `match_prefix`，
如果设备命中长度 ≥ key 长度就说明全覆盖了。

成功后（`:2329-2335`）：截断 `hash_value` 到实际命中长度、记录 `host_indices`、
`cc.prefetch_tokens_occupied += alloc_len`（**buffer 模式的占用额度在这一刻才授予**）、
放进 `cc.prefetch_buffer` 让 controller 去做真正的 storage 读。

### 6.4 park 队列的 FIFO 公平性

`_drain_and_alloc_storage_hit`（[unified_radix_cache.py:2338-2372](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2338-L2372)）：

```python
if buffer_mode:
    parked = self.buffer_pipeline.pending_hit_allocs
    while parked:
        if not _try_alloc_storage_hit(parked[0]):
            break                       # 队头还是不行，本轮结束
        parked.popleft()
for operation in _drain_queue(cc.prefetch_hit_queue, n_storage_hit):
    ...
    if not _try_alloc_storage_hit(operation):
        self._prefetch_outcome_stats["declined_rate_limited"] += 1  # 只在首次 park 时计一次
        self.buffer_pipeline.pending_hit_allocs.append(operation)
```

**先处理 park 队列再处理新命中**，保证重试的请求不会被新请求持续插队饿死。
`declined_rate_limited` 只在首次 park 时 +1，不是每 tick 都 +1（否则指标会爆）。

### 6.5 阶段 ③：停放（`stage_completed_prefetch`）

[pipeline.py:809-865](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L809-L865)。**永远返回 True** —— "ready"
是一个稳定、可被反复访问的状态，不像 cache 模式那样一次性 commit 完就走。

两条路：

**A. `num_tokens == 0` 或没有 prefix ctx**（`:832-840`）：什么都没取到 → 释放锚点锁、
归还主机 slot、扣回 occupancy、`prefetch_loaded_tokens_by_reqid[rid] = 0`，请求走重算。

**B. 正常停放**（`:842-864`）：

```python
staged_pages = num_tokens // cache.page_size
staged_hashes = hash_value[:staged_pages]
staged_kv = host_indices[:num_tokens]
# 从 storage 取回本身就是"L3 里确实有"的证据
cache.storage_existence_cache.add(PoolName.KV, list(staged_hashes))
self.staged_prefetches[req_id] = _StagedPrefetch(...)
cache.prefetch_loaded_tokens_by_reqid[req_id] = num_tokens
```

注意 `storage_existence_cache.add` 在这里就喂了 —— **fetch 本身就是证据**，
即使这个 staged prefetch 后面被丢弃未消费，信念也是成立的。

### 6.6 阶段 ④：准入计费（`plan_staged_splice`）

调度器在 prefill 准入循环里必须**把 staged prefetch 会占用的设备 token 计入预算**，
否则批次分配会 OOM。[scheduler.py:3544-3563](../../../python/sglang/srt/managers/scheduler.py#L3544-L3563)：

```python
req.init_next_round_input(self.tree_cache)
if (self.enable_hicache_storage
        and get_memory().hicache_host_memory_mode == "buffer_only"):
    held_tokens, held_swa_tokens = self.tree_cache.plan_staged_splice(
        req.rid, len(req.prefix_indices))
    if held_tokens > 0:
        req.host_hit_length = held_tokens
        req.swa_host_hit_length = held_swa_tokens
```

> 这行 SWA 计费是 2026-08-17 一次生产崩溃的回归修复（Llama-4-Scout）：
> 没有它，SWA 窗口不计费，prefill 会 OOM。回归测试
> `test_buffer_load_back_swa_window_charged_at_admission`。

`plan_staged_splice`（[pipeline.py:867-889](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L867-L889)）
除了算数，还有一个**防泄漏职责**：如果 hold 已经不可拼接（`splice_tokens == 0`），
它会**立即 `release_staged_hold(rid)` 释放掉**，而不是只返回 0。
因为"表面报 0 但保留 hold"会泄漏 —— adder 只会消费"被 surface 出来的 host hit"，
不会再调 `init_load_back`，那个 hold 就永远没人管了。

### 6.7 可拼接性判定：`staged_splice_tokens`

[pipeline.py:148-160](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L148-L160) —— 整个读路径的核心几何：

```python
span_end = f.matched_len + f.num_tokens
if device_prefix_len < f.matched_len or device_prefix_len >= span_end:
    return 0
splice_tokens = span_end - device_prefix_len
for t in f.aux_xfers:
    if t.host_indices is not None and t.host_indices.numel() > splice_tokens:
        return 0
return splice_tokens
```

三种返回 0（hold 不可用）的情况：

1. **前缀缩短到跨度之下**（`device_prefix_len < matched_len`）：拼接基点没了。
   例如 fetch 期间锚点被驱逐。
2. **跨度已完全设备驻留**（`device_prefix_len >= span_end`）：兄弟请求已经把这段发布了。
3. **裁剪会切进 staged 的辅助尾窗**：辅助池是**整体拼接或不拼接**，不能切一半。

情况 1、2 之间的中间态是**"生长"**：`matched_len <= device_prefix_len < span_end`，
此时**只拼尾部**（trim 掉已经设备驻留的头部），而不是整个丢弃。
`trim_tokens = splice_base - f.matched_len`，且断言它必须 page 对齐。

### 6.8 阶段 ⑤：消费（`init_load_back`）

[pipeline.py:904-1083](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L904-L1083)，全 buffer 模式最需要小心的函数。

**所有权契约（docstring `:911-914`）**：

> `cc.load` 在 insert 裁决所有权**之前**就把 H2D 排进了队列，所以下面的活跃预检必须
> 证明这次 insert **只能新增节点** —— 一次去重（dedup）会释放掉在飞拷贝正在写入的 slot，
> 造成**排队式 use-after-free**。

执行顺序（每一步失败都走 `_drop()`，请求退化为重算）：

| 步 | 内容 | 行 |
|---|---|---|
| 0 | 弹出 hold；不存在则释放锚点锁后原样返回 | `:920-923` |
| 1 | **命名空间校验**：`extra_key`/`cache_salt` 与请求不符 → drop（错命名空间发布 = 重复 slot 所有权）。理论不可达，防御性 | `:938-946` |
| 2 | `staged_splice_tokens(f, len(req.prefix_indices))`，为 0 则 drop（记 warning 含 `tokens_wasted`） | `:948-960` |
| 3 | 断言 trim 量 page 对齐 | `:961-965` |
| 4 | **活跃所有权预检**：`match_prefix(key)` 必须同时满足 `len(device_indices) == splice_base` **且** `full_kv_hit_length == splice_base` | `:975-995` |
| 5 | **evict-before-alloc**：预算闸算的是"可驱逐页"，但 `cc.load` 只从空闲 slot 取 → 不够就 `evict_for_alloc(needed)` 再查一次，仍不够则 drop | `:997-1012` |
| 6 | `cc.load(host_indices=f.host_indices[trim_tokens:], node_id=-(op_id)-1, extra_pools=f.aux_xfers)` | `:1014-1023` |
| 7 | 若有 SWA 设备窗口：**立刻** `RebuildFullToSWAMapping` 注册 FULL→SWA 翻译（准入的请求在 layer-gated forward 里就要通过它读窗口） | `:1025-1041` |
| 8 | `cache.insert(...)`，`value = cat([req.prefix_indices, device_indices])`，`prev_prefix_len=splice_base` | `:1043-1054` |
| 9 | 登记 `ongoing_buffer_load_back[load_back_id]`，`host_indices` 记的是**完整 bounce**（含被 trim 的头部），ack 时整段释放 | `:1055-1064` |
| 10 | 再 `match_prefix` 一次，**fail-stop 校验**：切片必须与 `cc.load` 返回的 `device_indices` `torch.equal`，否则 `raise RuntimeError` | `:1065-1080` |
| 11 | 返回 **post-insert 的树切片**（canonical），不是 `cc.load` 的原始分配 | `:1081-1083` |

第 4 步为什么要两个条件？`len(device_indices)` 检测"请求视图过时"（req 在某次后续发布之前
match 的）；`full_kv_hit_length` 检测 insert 会去重释放的 **FULL 重叠** ——
**一个 SWA tombstone 可以让活跃的 FULL 对统一 match 不可见**，单看统一长度会漏。
回归测试 `test_buffer_only_load_back_drops_on_full_overlap_masked_by_swa_tombstone`。

第 10 步的 `raise RuntimeError` 是**故意的 fail-stop**：宁可崩，也不要静默 KV 损坏。

`load_back_id = -(f.operation_id) - 1`：**用负数作为合成 ack id**，与 cache 模式的
正数 node id ack 空间不冲突，`try_finish_load_back` 靠字典查找区分归属。

### 6.9 阶段 ⑥：H2D ack（`try_finish_load_back`）

[pipeline.py:1085-1109](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L1085-L1109)。
被 [unified_radix_cache.py:2712-2716](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2712-L2716)
在 ack 循环里**先试 buffer、命中就 continue**，否则走 cache 模式的
`ongoing_load_back.pop` 路径。

它只做三件事：释放主机 bounce、扣 `prefetch_tokens_occupied`、记指标。
**完全不碰树** —— span 在准入时（阶段 ⑤）就已经发布了。

---

## 7. 锚点锁（anchor lock）

**默认关闭**（`SGLANG_ENABLE_HICACHE_BUFFER_ANCHOR_LOCK=False`）。

### 7.1 它解决什么

staged prefetch 的拼接依赖"设备前缀还在那儿"。fetch 期间如果锚点被驱逐，
整个 fetch 白做（`splice_tokens == 0` → drop → 重算）。锚点锁把
**从 IO commit 到消费**这段时间里的设备锚点钉住，让驱逐不能浪费这次 fetch。

### 7.2 `try_lock_anchor` 的 4 种结果

[pipeline.py:686-745](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L686-L745)：

| 返回值 | 含义 | 调用方动作 |
|---|---|---|
| `"no_anchor"` | 功能关闭，或锚点是 root（没东西可钉） | 继续 |
| `"locked"` | 已钉住（或本来就钉着，幂等） | 继续 |
| `"cap_skip"` | 超出总额度 | **不加锁继续发起**（降级而非放弃） |
| `"anchor_lost"` | 重新 match 发现前缀已缩短 | **取消 storage IO**（`revoke_pending_prefetch`）+ 武装 paced retry |

关键实现细节：**它不用记下来的 node id，而是重新 match 活跃树来找锚点**（`:726-735`）。
因为 node id 会因 split 和驱逐而失效；重走一遍前缀路径是 O(prefix path)，可以接受。

EAGLE（bigram）特例（`:719-725`）：后缀持有与最后一个 matched bigram 共享的边界 token，
所以重建锚点 key 时必须把它带上，否则边界对不上。回归测试
`test_buffer_anchor_rematch_preserves_bigram_boundary`。

### 7.3 额度上限的死锁防护

[pipeline.py:234-240](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L234-L240)：

```python
self.anchor_lock_cap_tokens = max(0, min(
    int(envs.SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP.get() * full_pool.size),
    full_pool.size - max_context_len,          # ← 关键
))
```

第二项是**准入余量夹逼**：钉住的量必须给"最大允许请求"留下空间，
否则一个排队的 hold 会**永久卡死准入**（池满、没有可回收的东西、请求进不来）。
没有余量的池（`size <= max_context_len`）直接不钉（cap=0）。
回归测试 `test_buffer_only_anchor_lock_cap_clamped_by_context_headroom`。

### 7.4 释放点

`release_anchor_lock`（`:747-758`）是**幂等**的，在每一个退出路径上都会调用：
`init_load_back` 的 drop 分支与成功分支、`release_staged_hold`、
`stage_completed_prefetch` 的空命中分支、`release_aborted_request`、
`cache_finished_req` 等（清单见 §9）。带一个 `assert anchor_locked_tokens_ >= 0` 的
账目自检。

---

## 8. 主机池预算：读优先（loads have priority）

buffer 模式下主机池被两方争抢：**写 staging**（可延迟）与**读 bounce**（不可延迟）。

```
主机池 (mem_pool_host.size)
├── 写窗口 live_cap = max(0.2 × size, size − prefetch_occupied − 0.1 × size)
│      ↑ HICACHE_WRITE_STAGING_POOL_FRACTION = 0.2（地板，防写路径饿死）
└── 读容量 prefetch_capacity_limit = 0.9 × size   （cache 模式只有 0.5 × size）
```

### 8.1 限流依据的差异

[cache_controller.py:1150-1168](../../../python/sglang/srt/managers/cache_controller.py#L1150-L1168)：

```python
if self.host_memory_mode == "buffer_only":
    used = self.mem_pool_host.size - self.mem_pool_host.available_size()
    if self.host_write_staged_tokens_fn is not None:
        used -= self.host_write_staged_tokens_fn()
    return max(0, used) >= self.prefetch_capacity_limit
```

**cache 模式看的是 `prefetch_tokens_occupied` 这个计数器；buffer 模式看的是池的真实占用
减去写 staging。** 原因：buffer 模式里池里的东西要么是写 staging，要么是读 bounce，
没有"树持有的 L2 数据"这一类，所以真实占用是可信的。
`host_write_staged_tokens_fn` 由树在 `init_hicache` 时注入
（[unified_radix_cache.py:465-467](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L465-L467)）。

### 8.2 辅助池（SWA）的独立预算

KV 池靠 `live_cap`，辅助池靠 `_aux_loads_margin`（§4.5）。两者的作用相同：
给读留一块写路径碰不到的保留区。`_aux_budget_blocked`（`:565-595`）在
`flush_pending_writes` 里**提前 break**，而不是让 `cc.write` 里的分配失败 ——
否则一个分配不到的队头会挡住后面所有纯 KV 意图。

---

## 9. 资源清理矩阵（最容易出泄漏的地方）

buffer 模式每个"hold"同时占三样东西：**锚点锁（设备 lock ref）**、
**主机 bounce（KV + 辅助池 slot）**、**occupancy 额度**。
任何退出路径漏掉一样都是泄漏。

| 退出路径 | 位置 | 释放什么 |
|---|---|---|
| 空命中停放 | `stage_completed_prefetch:832-840` | 锚点锁 + `append_host_mem_release` + occupancy |
| 计划阶段判定不可拼接 | `plan_staged_splice:878-888` → `release_staged_hold` | 全部三样 |
| 消费时 `_drop()` | `init_load_back:926-933` | 锚点锁 + `_free_staging_now` + occupancy + **把 `req.host_hit_length`/`swa_host_hit_length` 归零**（保持 surface 字段诚实） |
| H2D ack | `try_finish_load_back:1096-1099` | 主机 bounce + occupancy（锚点锁已在 insert 后释放） |
| 请求 abort | `release_aborted_request` → [unified_radix_cache.py:2115-2119](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2115-L2119) | `release_staged_hold(rid)`，命中就直接 return |
| abort 但还在 fetch 中 | [unified_radix_cache.py:2140-2142](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2140-L2142) | `pop_prefix_ctx` + `release_anchor_lock` |
| 其他清理点 | `:2061-2063`、`:2220-2222` | 同上 |
| flush_cache | `:376-377` → `pipeline.reset()` | 整体重置（配 `is_fully_idle` 排空保证） |

`_free_staging_now`（`:736-751`）会跳过 `indices_from_pool is not None` 的 transfer ——
那是 sidecar，骑在别的池的 slot 上，没有自己的所有权（随源池一起 free；#37424 起
buffer 模式已支持 sidecar，写出路径见 `:694-709`）。

---

## 10. TP / PP lockstep 契约

pipeline docstring（`:20-23`）：

> 每一个变更都在 scheduler 线程的 rank 同步点上执行（insert 走查、rank-MIN 归约的 drain、
> ack drain），所以 per-rank 状态永远不会分叉。**没有运行期校验；一旦违反，表现为一次
> 无法解释的 collective hang。**

具体保障：

| 机制 | 作用 |
|---|---|
| `_drain_queue(q, n)` 的 `n` 由跨 rank MIN 归约得出 | 所有 rank 消费**相同数量**的队列元素 |
| `finish_storage_write_ack` **无条件** `add` 存在性条目 | 后端故障时 `completed_tokens` 可能各 rank 不同，不能让它影响准入决策 |
| `_invalidate_absent_from_hit_query` 用 rank-synced 的 `storage_hit_count` | 治愈动作一致 |
| `@rank_consensus` 装饰 `release_aborted_request` / `_can_terminate_prefetch` | 清理与终止决策一致 |
| controller 队列严格 FIFO 单线程 | MIN-count drain 在每个 rank 上处理相同前缀 |

### 10.1 排空（drain）保证

`is_fully_idle`（[scheduler.py:4462-4477](../../../python/sglang/srt/managers/scheduler.py#L4462-L4477)）在
flush_cache / attach / detach 这类破坏性操作前检查。buffer 模式的 staging 在树之外，
通用的 `ongoing_*` 字典检查不够，所以额外加了：

```python
if get_memory().hicache_host_memory_mode == "buffer_only":
    idle &= tc.buffer_pipeline.is_idle()
```

`is_idle()`（[pipeline.py:280-285](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py#L280-L285)）=
`pending_write_queue`、`staged_prefetches`、`ongoing_backup` 三者皆空。
（这三个都持有主机 staging 或会重新触发 IO；`ongoing_write_through` 与
`ongoing_buffer_load_back` 由通用检查覆盖。）

### 10.2 调度器三处集成点汇总

| 位置 | 函数 | 干什么 |
|---|---|---|
| [scheduler.py:2819-2865](../../../python/sglang/srt/managers/scheduler.py#L2819-L2865) | `_prefetch_kvcache` | **锚点重定位**：buffer 模式 `last_host_node` 恒为 root，改用最深设备节点；准入闸从 `is_backuped` 放宽为"锚点带 hash 值" |
| [scheduler.py:3544-3563](../../../python/sglang/srt/managers/scheduler.py#L3544-L3563) | `get_new_batch_prefill` | **准入计费** `plan_staged_splice` |
| [scheduler.py:4462-4477](../../../python/sglang/srt/managers/scheduler.py#L4462-L4477) | `is_fully_idle` | **排空检查** |

---

## 11. `unified_radix_cache.py` 里的全部分派点

按功能归类，方便改代码时对照检查（行号基于 `33ed29a0ee`）：

| 分类 | 行号 | 分派内容 |
|---|---|---|
| 字段声明 | `:256` | `self.buffer_pipeline: Optional[BufferModePipeline] = None` |
| flush | `:376-377` | `pipeline.reset()` |
| 模型闸 | `:388-403` | 只允许 FULL/SWA |
| 构造 | `:446-467` | `validate_buffer_only_stack` + `BufferModePipeline(...)` + 注入 `host_write_staged_tokens_fn` |
| host 驱逐 | `:1121-1124` | `evict_host` 直接返回 0 |
| 写入口 | `:1340-1347` | `_execute_and_commit_kv_backup` → `enqueue_backup_intent` |
| D2H ack | `:1427-1428` | `_finish_write_through_ack` → `finish_backup_ack` |
| prefetch 发起 | `:1700`、`:1730`、`:1734-1745`、`:1821-1835` | 见 §6.2 |
| 停放 | `:1927-1930` | `stage_completed_prefetch` |
| 清理 | `:2061-2063` | `pop_prefix_ctx` + `release_anchor_lock` |
| 准入计费 | `:2093-2107` | `plan_staged_splice` / `staged_prefetch_swa_tokens` |
| abort | `:2115-2119`、`:2140-2142` | `release_staged_hold` / 锚点清理 |
| 断言 | `:2232` | `assert _host_indices is None or host_memory_mode != "buffer_only"` |
| 信念治愈 | `:2154-2165` | `_invalidate_absent_from_hit_query`（非 buffer 直接 return） |
| 清理 | `:2220-2222` | 同 abort |
| IO commit | `:2270`、`:2284-2341`、`:2372`、`:2405-2407` | 见 §6.3 / §6.4 |
| H2D ack | `:2712-2716` | `try_finish_load_back` |
| 消费分派 | `:2747-2748` | `init_load_back` → pipeline |
| 每 tick flush | `:2857-2858` | `flush_pending_writes` |
| sanity check | `:3044-3052` | 用 pipeline 的 `ongoing_write_through` 构造 (id, node_id) 对 |

另外两处在组件侧：

- [swa_component.py:894-905](../../../python/sglang/srt/mem_cache/unified_cache/components/swa_component.py#L894-L905)：
  buffer 模式允许**树中部的子窗口 fetch**（窗口头部就是设备前缀自身的 ring 状态），
  而 cache 模式要求整窗嫁接。
- [registry.py:216-226](../../../python/sglang/srt/mem_cache/registry.py#L216-L226)：树实现闸。

---

## 12. 尚未接线的 `BufferPageCache`

[buffer_mode/buffer_page_cache.py](../../../python/sglang/srt/mem_cache/buffer_mode/buffer_page_cache.py)
的 docstring 第一行就写着 **"NOT WIRED YET"**。它是一个规划中的优化，目前**没有任何调用点**。

设计意图：把 buffer 模式的主机 staging 变成一个**引用计数的内容寻址页缓存**，
`(pool, page_hash) → (slots, refcount)`，这样 prefetch 可以**零拷贝地直接从本地 staging 服务**，
不用走 L3。

与 `StorageExistenceCache` 的分工：

| | `StorageExistenceCache`（已接线） | `BufferPageCache`（未接线） |
|---|---|---|
| 追踪对象 | **对 STORAGE 的信念** | **本地 HOST RAM 的实际内容** |
| 去重的是 | **写** | **读** |
| 存的东西 | 只有 key | key + 实际 host slot + refcount |

已实现的能力：`register`/`acquire`/`release`/`reclaim`（零引用 LRU 回收）、
`peek_run_len`（前导连续命中长度）、write-around 语义（`retain=False`，
读命中时 promote 成 retained）、`continuation_run` 的 SWA 折叠（联合跨度的尾窗必须
完整可服务，对齐 `batch_exists_v2` 的 `trailing_pages` 折叠）、`acquire_span`。

接线时的前置条件（docstring `:10-16`，都是 TP 确定性要求）：
① 只在 scheduler 线程的 lockstep 点变更；② 任何由 per-rank 存储结果驱动的折叠
（命中数、revoke）必须先跨 rank 归约再决定变更；③ controller 队列保持 FIFO 单线程。

> 因为它**未接线**，读这个模块时不要把它当成运行期行为的一部分。

---

## 13. 测试与压测

### 13.1 单测

主战场：[test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py](../../../test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py)，
buffer 相关用例从 `:3487` 开始。夹具：`_init_buffer_hicache`（`:3514`，强制 `file` 后端、
`prefetch_threshold=1`、跳过 Mamba 夹具）、`_produce_buffer_l3`、`_buffer_backup_and_wait`、
`_pump_hicache_until`。

| 用例 | 行 | 覆盖点 |
|---|---|---|
| `test_buffer_only_rejects_mamba` | `:3491` | FULL/SWA 闸 |
| `test_buffer_only_write_path_roundtrip` | `:3613` | 写路径全链路 + staging/锁全部释放 + 重命中跳过 |
| `test_buffer_only_read_path_roundtrip` | `:3659` | 读路径全链路 + staged 对 match 不可见 + 字节级一致 + 无 CPU 层 KV 事件 |
| `test_buffer_only_cache_salt_uses_the_request_namespace` | `:3786` | `cache_salt`/`extra_key` 命名空间路由 |
| `test_buffer_only_storage_prefetch_miss_marker_and_retry` | `:3911` | miss 标记只武装一次 + abort 清理 |
| `test_buffer_only_anchor_lock_cap_clamped_by_context_headroom` | `:3984` | 死锁不变量 `cap <= pool - context_len` |
| `test_buffer_load_back_swa_window_charged_at_admission` | `:4016` | **2026-08-17 生产崩溃回归** |
| `..._drops_on_sibling_published_span` | `:4132` | 排队式 UAF |
| `..._masked_by_swa_tombstone` | `:4185` | SWA tombstone 遮蔽 FULL |
| `..._fail_stops_on_post_check_overlap` | `:4245` | 第 10 步 fail-stop |
| `..._trims_head_published_by_sibling` | `:4301` | 生长时裁头而非整丢 |
| `test_buffer_only_plan_frees_covered_hold` | `:4373` | `plan_staged_splice` 的防泄漏职责 |
| `test_buffer_only_hit_commit_cancels_device_covered_fetch` | `:4416` | `declined_device_covered` |
| `test_buffer_only_swa_window_semantics` | `:4467` | 三种部分窗口场景（Llama-4-Scout 零复用回归） |
| `test_buffer_anchor_rematch_preserves_bigram_boundary` | `:921` | EAGLE bigram 边界 |
| `TestAnchorLockOutcomePolicy` | `:8869` | `try_lock_anchor` 四种返回 |

参数解析侧只有 `test/registered/unit/server_args/test_server_args.py:1493-1501`
的 `test_buffer_only_accepts_both_tree_cores`。
**§2.2 里三个 `ValueError` 分支目前没有单测覆盖**。

`test_hiradix_pp_sync_drain.py:59` 只是把 `cache.buffer_pipeline = None` 来走 cache 分支，
不覆盖 buffer 行为。

### 13.2 A/B 压测脚本

[benchmark/hicache/bench_buffer_mode.py](../../../benchmark/hicache/bench_buffer_mode.py)：
同模型对比 `cache`(write_through) / `cache_wb`(write_back) / `buffer` 三种模式，
两种 workload（`multiturn` / `longctx`）。

```bash
python benchmark/hicache/bench_buffer_mode.py \
    --model-path <model> --modes cache,buffer --workloads multiturn,longctx
```

buffer 分支传的 flag（`:198-208`）：`--hicache-host-memory-mode buffer_only`
`--hicache-storage-backend <backend>`，加上 `--hicache-ratio`（若 `--host-ratio>0`）
或 `--hicache-size <buffer_size_gb>`。用
`SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` 指向 per-mode 临时目录做隔离。

**它不做任何 assert**，只产出 JSON 报告（命中率 / TTFT / p90）。
`longctx` 的关键是 warm → `POST /flush_cache` → replay，replay 那一趟才真正压 L3。

> ⚠️ 脚本 bug：`:35` 和 `:37` 抓的两个指标名
> `sglang:hicache_existence_cache_skipped_pages_total`、
> `sglang:hicache_pending_write_queue_depth` **在仓库里根本不存在**，
> `scrape_metrics` 静默忽略未匹配项，所以报告里 buffer 专属计数器那一节是半空的。

---

## 14. 调试指南

### 14.1 关键日志

| 日志 | 位置 | 含义 |
|---|---|---|
| `BufferModePipeline anchor_lock_enabled=... cap_tokens=...` | `:241` | 启动时打印一次，确认锚点锁配置 |
| `HiCache write backlog cap hit (occurrence N)` **ERROR** | `:338` | **这是 bug 信号**，不是负载信号 —— 说明 stale sweep 或账目泄漏 |
| `HiCache anchor-lock cap reached (skip N)` WARNING | `:708` | 锚点锁额度打满，降级为不加锁 |
| `HiCache staged prefetch released req=... device_prefix=...` INFO | `:879` | 计划阶段发现 hold 不可用 |
| `HiCache staged prefetch dropped req=... reason=namespace` ERROR | `:939` | 理论不可达，出现就是真 bug |
| `HiCache staged prefetch dropped req=... matched=N now=M tokens_wasted=K` WARNING | `:951` | 拼接基点变化导致的浪费（`locked=` 字段可判断锚点锁是否有效） |
| `HiCache staged prefetch dropped req=... reason=overlap` WARNING | `:984` | 兄弟请求抢先发布了这段 |
| `HiCache buffer load-back ownership violation` **RuntimeError** | `:1073` | fail-stop，绝不能忽略 |
| `HiCache prefetch fill committed req=... filled=N occupied=M locked=K` INFO | `:1100` | 正常完成 |

### 14.2 可用指标

| 指标 | 说明 |
|---|---|
| `sglang:hicache_backup_dropped_tokens_total` | 写意图被丢弃的 token 数（准入 4 道闸 + flush 里的丢弃） |
| `sglang:hicache_host_used_tokens` | 主机池占用（buffer 模式下 = 写 staging + 读 bounce） |
| `prefetch_outcome_stats`（`prefetch_outcome_stats_snapshot()`） | 含 buffer 专属的 `declined_anchor_lost`、`declined_device_covered`、`declined_rate_limited` |
| `log_prefetch_aux_alloc_failed_tokens` | 辅助池（SWA 窗口）分配失败导致整个 prefetch 作废的量 —— **写突发饿死辅助池时看这个**，否则表现为泛泛的命中率下降 |

### 14.3 常见故障 → 排查方向

| 现象 | 排查 |
|---|---|
| L3 命中率≈0（SWA 模型） | 查 SWA host 池是否 ≥ 2 窗口（启动时应已报错）；查 `log_prefetch_aux_alloc_failed_tokens`；查 `_aux_loads_margin` 是否被写路径吃穿 |
| 写覆盖率低、`backup_dropped_tokens` 高 | 大概率是 `_backup_parent_covered` 级联丢弃：某个父节点被丢导致整条链丢。也可能 `live_cap` 被读压满（看 `prefetch_tokens_occupied`） |
| `write backlog cap hit` ERROR | 账目泄漏，检查 `write_backlog_tokens_` 的加减是否配平（准入 `+`、sweep `−`、launch `−`、丢弃 `−`） |
| `tokens_wasted` 大量 WARNING | fetch 期间锚点churn 严重 → 试开 `SGLANG_ENABLE_HICACHE_BUFFER_ANCHOR_LOCK=1` |
| prefill OOM | 检查 `plan_staged_splice` 的计费是否覆盖了所有池（历史 bug 就是漏了 SWA 窗口） |
| collective hang | TP lockstep 被破坏。检查是否有代码在非 scheduler 线程动 pipeline/存在性缓存，或引入了未归约的 per-rank 分支 |
| flush_cache/attach 卡住 | `is_idle()` 不满足：查 `pending_write_queue`/`staged_prefetches`/`ongoing_backup` 谁没排空 |

---

## 15. 已知粗糙边缘 / 待改进

| 项 | 说明 |
|---|---|
| `BufferPageCache` 未接线 | 完整实现了但零调用点（§12）。本地 staging 复用的机会目前完全没有利用 |
| Mamba / SSM 被禁 | `init_hicache:390-396` 有 TODO(Jialin)，需要 state 交接通道 + `req.mamba_host_hit_length` 计费 |
| PD decode 侧被禁 | D 侧 prefetch/offload 走的是另一套路径 |
| 主机池"太小"的误报警告 | `pool_host/base.py:163-171` 在 `size <= device_pool.size` 时警告 "L2 cache effectiveness is reduced" —— buffer 模式**本来就该小**，这条警告在此模式下是误导 |
| 三个参数校验分支无单测 | §2.2 的"无 storage 后端 / write_back / decode 模式" |
| 压测脚本抓不存在的指标 | §13.2 的两个指标名 |
| 无官方文档 | `docs/` 下没有 buffer 模式的用户文档，唯一的散文说明是模块 docstring |
| `_aux_window_keys` 有两份实现 | `pipeline.py:597-612` 与 `buffer_page_cache.py:234-251`（`BufferPageCacheOps.aux_window_keys`）逻辑等价、各自一份 |

---

## 16. 速查：一次请求的完整时间线

以"某个长 prompt 的前缀已在 L3、本机 L1 只命中一部分"为例（page_size=64）：

```
t0  请求到达
    init_next_round_input → 设备 match 命中 512 token（prefix_indices）
    _prefetch_kvcache: last_host_node 恒为 root → 重锚到最深设备节点
                       该节点带 hash 值 → 放行
    prefetch_from_storage:
        set_prefix_ctx(rid, prefix_tokens=512 个 token)
        try_lock_anchor(rid)  → "locked"（若开启）或 "no_anchor"
        cc.prefetch(...)  发起 L3 存在性查询
        prefetch_tokens_occupied 不动     ← 与 cache 模式的关键差异

t1  L3 应答：命中 2048 token
    _try_alloc_storage_hit:
        prefetch_rate_limited()? 否
        try_lock_anchor 二次确认 → 不是 anchor_lost
        staged_span_covered(rid, 2048)? 否
        mem_pool_host.alloc(2048) → 拿到 host bounce
        prefetch_tokens_occupied += 2048   ← 此刻才授予额度
        放进 prefetch_buffer，controller 开始真正读 L3

t2  storage → host 读完
    stage_completed_prefetch:
        storage_existence_cache.add(KV, 32 个 page hash)
        staged_prefetches[rid] = _StagedPrefetch(matched_len=512, num_tokens=2048)
        ★ 此刻树上、设备上都没有任何东西；其他请求 match 不到这 2048 token

t3  scheduler prefill 准入
    plan_staged_splice(rid, device_prefix_len=512)
        → staged_splice_tokens = (512+2048) - 512 = 2048
        → req.host_hit_length = 2048, req.swa_host_hit_length = <窗口>
    （预算按 2048 计费）

t4  init_load_back:
        命名空间校验 ✓
        splice_tokens = 2048, trim_tokens = 0
        活跃 match 预检：len==512 且 full_kv_hit_length==512 ✓
        available < 2048 → evict_for_alloc(差额)
        cc.load(host_indices, node_id=-(op_id)-1)  → H2D 排队
        （SWA）RebuildFullToSWAMapping
        cache.insert(key, cat([prefix_indices, device_indices]), prev_prefix_len=512)
        ongoing_buffer_load_back[-(op_id)-1] = ...
        再 match + torch.equal 校验 ✓
        release_anchor_lock(rid)
        返回 canonical 切片

t5  H2D ack:
    try_finish_load_back: 释放 host bounce（含 trim 掉的头）
                          prefetch_tokens_occupied -= 2048
    ★ 主机内存归零，L1 上多了 2048 token

t6  请求生成完成 → cache_finished_req → 树上新节点触发 BackupKV
    enqueue_backup_intent: 存在性信念已覆盖那 32 页 → **静默跳过，不重写 L3**
    只有新生成的那部分会走写路径
```

---

## 17. 参考文件清单

| 文件 | 行数 | 职责 |
|---|---|---|
| [buffer_mode/__init__.py](../../../python/sglang/srt/mem_cache/buffer_mode/__init__.py) | 9 | 模块 docstring（最权威的一段散文说明） |
| [buffer_mode/pipeline.py](../../../python/sglang/srt/mem_cache/buffer_mode/pipeline.py) | 1122 | `BufferModePipeline` + `validate_buffer_only_stack` + `staged_splice_tokens` |
| [buffer_mode/storage_existence_cache.py](../../../python/sglang/srt/mem_cache/buffer_mode/storage_existence_cache.py) | 89 | 存储存在性信念 LRU |
| [buffer_mode/buffer_page_cache.py](../../../python/sglang/srt/mem_cache/buffer_mode/buffer_page_cache.py) | 396 | **未接线**的本地页缓存 |
| [unified_radix_cache.py](../../../python/sglang/srt/mem_cache/unified_radix_cache.py) | — | 20+ 处模式分派（§11） |
| [arg_groups/hicache_hook.py](../../../python/sglang/srt/arg_groups/hicache_hook.py) | — | 参数校验 + ratio 默认值 |
| [managers/cache_controller.py](../../../python/sglang/srt/managers/cache_controller.py) | — | 容量/限流的模式分支 |
| [managers/scheduler.py](../../../python/sglang/srt/managers/scheduler.py) | — | 三处集成点（§10.2） |
| [benchmark/hicache/bench_buffer_mode.py](../../../benchmark/hicache/bench_buffer_mode.py) | — | A/B 压测 |

