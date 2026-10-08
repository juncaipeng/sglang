# External Linker（外部链接器）模式全景梳理

> 基线 commit：`33ed29a0ee`
> 相关 PR 系列：`c9eb475a88` [1/N] 契约 → `6a9366f036` [2/N] 设备池装配 → `5d92e60783` [3/N] 后端无关的 linker 核心 → `b21000aef1` [4/N] Mooncake 后端。基线之后该系列已继续推进并**完成接线**，包括 `908226fea2`（[Rust TreeCore] 支持 external linker，#37306）、`822e73ccdd`（[AMD] DeepSeek-V4 unified KV 直连，#38269）、`a7cf4a6fbc`（[7/N] 支持 MTP/EAGLE/DSpark draft KV，#37914）等。
> 交叉阅读：[unified_radix_cache_architecture.md](unified_radix_cache_architecture.md)、[hicache_usage_and_design.md](hicache_usage_and_design.md)、[mooncake_integration_architecture.md](mooncake_integration_architecture.md)、[host_cache_off_tree_staging_scheme.md](host_cache_off_tree_staging_scheme.md)

---

## 0. 一句话总结与当前状态

**External Linker 模式 = 让 `UnifiedRadixCache` 的 GPU 显存池直接和一个外部 KV 存储（如 Mooncake）对话，中间不经过 host（CPU 内存）这一层。**

传统 HiCache 是三层搬运：

```
GPU 显存 (L1)  <--D2H/H2D-->  Host 内存 (L2)  <--网络/磁盘-->  外部存储 (L3)
```

External Linker 是两层直连：

```
GPU 显存 (L1)  <----------  RDMA 直读/直写  ---------->  外部存储 (远端)
```

**当前状态：已接线。** 基线 `33ed29a0ee` 时这套东西还没有生产入口；其后已接到 `mem_cache/registry.py` 的缓存构建流程和 CLI 上。

- 开关是 `--enable-unified-cache-external-linker`（bool，默认 `False`）；后端由
  `--unified-cache-external-linker-backend` 选择（`choices=["mooncake", "mori"]`，默认 `mooncake`）。
  两者定义在 [arg_groups/fields/memory.py:204-214](../../../python/sglang/srt/arg_groups/fields/memory.py#L204-L214)，运行期经 `get_memory()` 读取。
- 生产实例化点在 [registry.py:199-224](../../../python/sglang/srt/mem_cache/registry.py#L199-L224)：`enable_unified_cache_external_linker`
  为真时按后端选择 `MooncakeDirectLinker` 或 `UMBPDirectLinker`（mori），再调用 `cache.init_cache_linker(...)`。
- 参数校验在 [arg_groups/hicache_hook.py:26-34](../../../python/sglang/srt/arg_groups/hicache_hook.py#L26-L34)：与
  `--enable-hierarchical-cache` **互斥**，且不使用 `--hicache-storage-backend`。

这个模式定义了 SGLang "GPU 直连远端 KV 存储"的契约。

> ⚠️ **行号说明**：本文其余章节的行号仍以基线 `33ed29a0ee` 为准。接线之后核心源码已有位移和文件重命名（例如组件文件已从 `components/{full,swa,mamba}_component.py` 改名为 `components/{full,swa,mamba}.py`），阅读具体位置时请以当前代码为准。

---

## 1. 为什么要有这个模式：和 HiCache 的动机对比

先看 HiCache（三层）在做什么。一个 prefix 命中远端存储时，数据要走：

1. 远端存储 → host 内存（`prefetch`，占用一块 host 页）
2. host 内存 → GPU（`load_back`，D2H 反向的 H2D copy）

写回同理，`GPU → host（write_through/backup）→ 远端（backup_storage）`。

这带来四个成本：

| 成本 | 说明 |
|---|---|
| **额外一次内存拷贝** | 每个 token 的 KV 都要在 host 里落一次地，PCIe 带宽被占用两遍 |
| **host 内存池** | 需要预留大块 pinned host 内存（`--hicache-ratio`），这块内存本身是纯开销 |
| **host 页分配/驱逐逻辑** | `HostPoolGroup` 的 alloc/evict/LRU 是一整套独立状态机，是 bug 高发区 |
| **延迟串联** | 远端→host 完成后才能开始 host→GPU，两段延迟相加 |

External Linker 直接砍掉中间层：Mooncake 通过 `register_buffer()` 把 **GPU 显存的裸指针**注册给传输引擎，
RDMA 直接读写显存（GPU-Direct RDMA）。于是：

- 没有 host 池、没有 host LRU、没有 D2H/H2D。
- 延迟只有一段网络。
- 代价：**远端存储必须支持 GPU 内存注册 + 按字节 range 读**（Mooncake 的 `batch_get_into_multi_buffer_ranges`）。

### 一个关键的语义差异

HiCache 的 host 层是**一个真正的缓存层（tier）**：树节点有 `host_value`、有 `backuped` 状态、有 host LRU、
会被 host 驱逐。External Linker 的远端存储**不是一个 tier，只是一个"是否已持久化"的布尔标记**：

```python
node.external_cache_stored: bool     # unified_tree_core.py:128
```

树里不再有"这个节点在 L2/L3 上占了多少页"的概念。远端要么有、要么没有。
这大幅简化了树的状态空间——这也是 `unified_cache_linker.py` 只有 570 行、而 HiCache 相关代码上千行的原因。

---

## 2. 分层架构与文件地图

一共四层，从上到下：

```
┌──────────────────────────────────────────────────────────────────────┐
│  L0  树本体：UnifiedRadixCache                                       │
│      unified_radix_cache.py                                          │
│      只有 8 个 `if self.linker is not None:` 的守卫式挂钩            │
└──────────────────────────────────────────────────────────────────────┘
                              ↓ 持有一个普通属性 self.linker
┌──────────────────────────────────────────────────────────────────────┐
│  L1  流程编排（后端无关）：UnifiedCacheLinkerWrapper                 │
│      unified_cache/unified_cache_linker.py                           │
│      负责：跨 rank 求交集、insert、锁引用计数、pending 状态机        │
└──────────────────────────────────────────────────────────────────────┘
                              ↓ 调用抽象接口 UnifiedCacheLinker (ABC，12 个方法)
┌──────────────────────────────────────────────────────────────────────┐
│  L2  设备池装配（后端无关）：DevicePoolEntry / DevicePoolGroup       │
│      hybrid_cache/linker_pool_assembler.py                           │
│      负责：把"逻辑组件传输"展开成"物理显存池 + 裸指针 + 字节偏移"    │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│  L3  后端实现：MooncakeDirectLinker                                  │
│      storage/mooncake_store/mooncake_direct_linker.py                │
│      负责：线程模型、GPU buffer 注册、layer-wise 计数器、物理 IO     │
└──────────────────────────────────────────────────────────────────────┘
```

另外有一条**横向的组件维度**（component），因为 KV 不止一种：

```
unified_cache/components/
├── tree_component.py    基类，定义 2 个 external linker 钩子（默认返回 None = 不支持）
├── full_component.py    :436 / :476   全量注意力 KV      → PoolName.KV
├── swa_component.py     :1015 / :1065 滑动窗口 KV        → PoolName.SWA
└── mamba_component.py   :652          直接 raise AssertionError（明确不支持）
```

---

## 3. 树本体只暴露 8 个挂钩

这是整个设计最值得学习的地方：**外部存储路径完全不侵入树文件**。
`unified_radix_cache.py` 里所有和 linker 相关的代码就下面这些（行号为基线值）：

| 行号 | 位置 | 挂钩 | 作用 |
|---|---|---|---|
| :351 | `init_cache_linker` | 构造 `UnifiedCacheLinkerWrapper` | 唯一开关 |
| :356 | `reset` | `linker.reset()` | 静默所有传输、释放锁 |
| :514 | `release_host_resources` | `linker.close()` | 退出清理 |
| :537 | `match_prefix` | `linker.match(...)` | **读路径第 1 步**：探测远端 |
| :1080 | `_apply_cache_action` / `ReplaceWriteThroughOnNodeSplit` | `linker.replace_pending_offload_node` | 节点分裂时改写 in-flight 写回目标 |
| :1094 | `_apply_cache_action` / `BackupKV` | `linker.offload_nodes(...)` **替代** `_execute_and_commit_kv_backup` | **写路径唯一入口** |
| :2111 | `cache_finished_req` 之类的释放路径 | `linker.release_request(rid)` | 请求提前结束时撤销未启动的 load |
| :2753 | `init_load_back` | `linker.load_back(req)` | **读路径第 2 步**：真正搬数据 + insert |
| :2789 | `check_hicache_events` | 轮询完成、跨 rank 对齐、提交 | 每个 scheduler step 调一次 |
| :2870 | `ready_to_load_host_cache` | `linker.start_layer_wise_loading()` | 触发本批 layer-wise 加载 |

注意 :1094 和 :2789 是 **`if/else` 互斥**的：一旦挂了 linker，HiCache 的写回和事件轮询完全被旁路，
两套机制**不会同时工作**。

再加上 `unified_tree_core.py` 里的 4 处：

| 行号 | 作用 |
|---|---|
| :402 | `enable_external_cache_linker = False` 默认关 |
| :915 | `_inc_hit_count_and_check`：写回判定改看 `not node.external_cache_stored`（而不是 `not node.backuped`） |
| :1166 | 节点分裂时把 `external_cache_stored` 复制到新父节点 |
| :1225 | `enable_storage or enable_external_cache_linker` → 才给节点算 `hash_value`（存储 key 的来源） |
| :2046 | 写回链向上回溯的终止条件：`not (ancestor.backuped or ancestor.external_cache_stored)` |

---

## 4. 核心数据结构

### 4.1 `PoolTransfer`：贯穿全链路的唯一载体

定义在 [hicache_storage.py:95](../../../python/sglang/srt/mem_cache/hicache_storage.py#L95)（HiCache 和 linker 共用）：

```python
@dataclass
class PoolTransfer:
    name: PoolName                              # 哪个池
    host_indices: Optional[torch.Tensor] = None # linker 模式下被复用为"物理池行号"
    device_indices: Optional[torch.Tensor] = None
    keys: Optional[List[str]] = None            # 每页一个 hash 字符串
    hit_policy: PoolHitPolicy = ALL_PAGES
    nodes_to_load: Optional[List[Any]] = None   # linker 模式不用
    indices_from_pool: Optional[PoolName] = None
```

> ⚠️ **命名陷阱**：字段叫 `host_indices`，但在 external linker 模式下它装的是**显存池的行号**
> （`DevicePoolGroup.resolve_transfers` 里 `host_indices=entry.translate_indices(device_indices)`）。
> 这是为了复用 `mooncake_store` 已有的 v2 IO 代码，不是笔误。

### 4.2 `PoolName`：逻辑池 vs 物理池

`PoolName`（[hicache_storage.py:58](../../../python/sglang/srt/mem_cache/hicache_storage.py#L58)）一共 13 个成员，
在 linker 模式下分两类：

- **逻辑池**：组件层产出的，只有 `KV`（FullComponent）、`SWA`（SWAComponent）、`MAMBA`（未支持）。
- **物理池**：`DevicePoolGroup` 展开出来的，比如 DeepSeek V4 会展开成
  `SWA / DEEPSEEK_V4_C4 / DEEPSEEK_V4_C4_INDEXER / DEEPSEEK_V4_C128 / DEEPSEEK_V4_C4_STATE / DEEPSEEK_V4_C4_INDEXER_STATE`。

映射关系记在 `DevicePoolEntry.indices_from_pool`：物理池 `DEEPSEEK_V4_C4` 的页索引来自逻辑池 `KV`，
物理池 `DEEPSEEK_V4_C4_STATE` 的页索引来自逻辑池 `SWA`。

### 4.3 `PoolHitPolicy`：两种命中语义（后面第 6 节的关键）

```python
ALL_PAGES      = "all_pages"        # [0, N) 每一页都必须存在（连续前缀语义）
TRAILING_PAGES = "trailing_pages"   # 只要求最后 N 页存在（滑动窗口 / 状态类）
```

### 4.4 两个阶段枚举

`LinkerTransferPhase`（[tree_component.py:92](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py#L92)）——
**"要构造什么样的传输描述"**，由 `build_external_linker_transfer(phase, node, keys)` 消费：

| 阶段 | node | keys | 组件要做的事 |
|---|---|---|---|
| `LOOKUP` | None | 未命中尾部的每页 hash | 只填 `keys`，不分配显存 |
| `LOAD` | None | 同上（已裁剪到命中长度） | **分配显存槽位**（不够就先 `evict`），失败返回 None |
| `OFFLOAD` | 目标树节点 | None | 从 `node.component_data[ct].value` 取显存槽 + `node.hash_value` 取 key |

`ExternalLinkerLoadPhase`（[:98](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py#L98)）——
**load 内部的三段式事务**，由 `update_external_linker_load(...)` 消费：

| 阶段 | 时机 | 组件要做的事 |
|---|---|---|
| `PREPARE` | insert 之前 | 建立跨池映射（SWA：`set_full_to_swa_mapping`）、初始化 `req.kv.swa_evicted_seqlen` |
| `COMMIT` | insert 之后 | 按 `adopted_ranges` **裁掉未被树采纳的页**，重建映射到 canonical 槽位 |
| `ABORT` | 任一组件分配失败 | **逆序**回滚：把已分配的显存槽 `free()` 掉，返回 None |

`ABORT` 用 `reversed(component_transfers)` 遍历（`unified_cache_linker.py:383-387`），
保证释放顺序与分配顺序相反——SWA 的槽位依赖 Full 的槽位建立映射，必须先撤 SWA。

### 4.5 两个 pending 记录

```python
# unified_cache_linker.py:115
class ExternalCacheHitMarker(NamedTuple):
    prefix_key: RadixKey     # 显存已命中前缀 + 远端可恢复尾部，一起，用于 insert
    tail_hashes: list[str]   # 仅尾部的逐页 hash（从 device_hit_len 开始）
    device_hit_len: int      # 显存已命中的 token 数

# :128
class _PendingOffload(NamedTuple):
    lock_node_id: NodeId              # 加锁的锚点节点（ack id）
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]    # 完成后要标记 stored 的节点集合（分裂会变多）
```

`hit_markers: dict[rid, ExternalCacheHitMarker]` 是 `match` → `load_back` 之间的**唯一状态传递通道**。

---

## 5. 读路径（远端 → 显存）逐步拆解

### 5.1 全景时序

```
Scheduler
  │
  ├─ match_prefix(key, req)                      [unified_radix_cache.py:523]
  │    ├─ tree_core.match_prefix()               → 显存命中 device_hit_len
  │    └─ linker.match(key, req, result)          [linker:163]
  │         ├─ _tail_hashes()                    → 未命中尾部的逐页 hash
  │         ├─ 每个组件 build_..._transfer(LOOKUP)
  │         ├─ cache_linker.lookup(rid, transfers)  ── RPC ──> 远端 batch_is_exist
  │         ├─ _sync_restorable_prefix()         → 跨 rank all_reduce(MIN) 求交集
  │         └─ 写 hit_markers[rid]，返回 host_hit_length = hit_tokens
  │
  ├─ (调度器根据 host_hit_length 决定是否值得 load，走 prefill 准备流程)
  │
  ├─ init_load_back(params)                      [unified_radix_cache.py:2740]
  │    └─ linker.load_back(req)                   [linker:260]
  │         ├─ 每个组件 build_..._transfer(LOAD) → 分配显存槽（可能触发 evict）
  │         │     └─ 任一失败 → _update_load(ABORT) 逆序回滚，返回空
  │         ├─ _update_load(PREPARE)             → 建立 full↔swa 映射等
  │         ├─ cache.insert(prefix_key, prefix_indices, chunked=True,
  │         │              track_adopted_ranges=True)
  │         ├─ collect_full_device_indices()     → canonical_tail（树里真正生效的槽位）
  │         ├─ _update_load(COMMIT)              → 按 adopted_ranges 裁页 + 重建映射
  │         ├─ _queue_load(rid, node, transfers) → inc_lock_ref 锁住节点 + 后端排队
  │         └─ 把新插入链上的节点全部标 external_cache_stored = True
  │
  ├─ ready_to_load_host_cache()                  [unified_radix_cache.py:2868]
  │    └─ linker.start_layer_wise_loading()      → 后端把整批丢给 load 线程，返回 counter_index
  │
  ├─ (模型 forward，每层通过 layer_done_counter.wait_until(layer) 等这一层的 KV 到位)
  │
  └─ check_hicache_events()                      [unified_radix_cache.py:2787]
       ├─ all_reduce(MIN) 对齐各 rank 的完成计数
       ├─ linker.drain_loads(n)                  → dec_lock_ref 解锁
       └─ linker.take/commit_completed_offloads()
```

### 5.2 `_tail_hashes`：为什么必须有"锚点"

```python
# unified_cache_linker.py:237
def _tail_hashes(self, key, result, device_hit_len):
    last_hash = None
    if device_hit_len > 0:
        last_hash = self.cache.get_last_hash_value(result.last_device_node)
        if last_hash is None:
            return []          # 拿不到锚点，直接放弃远端命中
    tail_len = (len(key) - device_hit_len) // page * page
    return get_hash_str(key[device_hit_len : device_hit_len + tail_len],
                        last_hash, page_size=page)
```

SGLang 的存储 key 是**链式 hash**：第 i 页的 hash = f(第 i-1 页的 hash, 第 i 页的 token)。
所以要算尾部第一页的 hash，必须先知道显存命中部分最后一页的 hash（`last_hash` 锚点）。
拿不到锚点时，如果从头开始算，得到的 key 和远端里存的 key 完全对不上——所以代码选择直接返回空、放弃命中，
注释写得很明确：*"Without the anchor the tail would hash as if it started at the sequence head, yielding keys that can never match."*

### 5.3 `load_back` 的三段式事务：为什么需要 `adopted_ranges`

这是读路径最反直觉的一段。流程是：

1. **LOAD 阶段**已经为整个"可恢复尾部"分配了显存槽（比如 4 页 = 8 token）。
2. 然后调 `cache.insert()`。但 insert 有可能发现：**树里已经有一部分页了**（并发的其它请求刚插进去，
   或者一个 evicted 节点被 unevict）。这时 insert 会**丢弃**你传进来的部分槽位，用树里已有的。
3. 于是"我分配的槽位"和"树里真正生效的槽位"就不一致了。如果照原样发起 RDMA，
   数据会被写到已经被 `FreeDeviceKV` 释放掉、可能已被别人占用的显存上——**静默数据损坏**。

`InsertParams.track_adopted_ranges=True` 就是为解决这个：insert 过程中每次真正"采纳"传入的槽位，
就 `result.record_adopted_range(component_type, start, end)`
（[unified_tree_core.py:1037](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1037) unevict 分支、
[:1092](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1092) 新建叶子分支）。

COMMIT 阶段用 `_select_adopted_pages()` 把 transfer 的 `device_indices` 和 `keys`
**按页裁剪到只剩被采纳的区间**（`unified_cache_linker.py:390-412`）。裁完为空的组件直接 `continue` 掉。

单元测试 `test_component_commit_keeps_only_adopted_pages` 精确固化了这个语义：
4 页里只有第 2、4 页被采纳（`adopted_ranges = [(2,4),(6,8)]`，token 坐标），
结果 `keys` 从 `[a,b,c,d]` 裁成 `[b,d]`，`device_indices` 从 8 个裁成 4 个。

因此后端 `load()` 拿到的 transfer 列表可能是**残缺的**——甚至可能一个 KV 页都没有、只剩 SWA：

```python
# mooncake_direct_linker.py:222-234
def load(self, rid, transfers):
    # Query establishes a boundary at which every component is restorable;
    # insert then removes pages already resident in L1. Loading is therefore
    # intentionally partial and may contain only a side pool such as SWA.
    expanded = self.pool_group.resolve_transfers(
        transfers, allow_partial=True, allow_missing_kv=True)
```

### 5.4 刚 load 回来的节点为什么不会立刻被写回

两道保险：

1. `insert(..., chunked=True)` → `_inc_hit_count_and_check` 第一行就 `if node.evicted or chunked: return False`
   （[unified_tree_core.py:909](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L909)），不会触发 `BackupKV`。
2. `load_back` 结尾显式沿新链向上打标（`unified_cache_linker.py:346-349`）：

```python
node = cache.resolve_node_handle(insert_result.last_device_node)
while node.id != req.last_node:
    node.external_cache_stored = True
    node = node.parent
```

如果没有第 2 步，下一次这条前缀被命中时 `_inc_hit_count_and_check` 会看到 `not external_cache_stored`，
把刚读回来的数据原封不动再写一遍回远端——纯浪费带宽。

---

## 6. 最精妙的一点：`lookup` 返回"集合"而不是"最大值"

### 6.1 问题

`UnifiedCacheLinker.lookup()` 的签名是：

```python
def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
    """Return every prefix length (in pages) that is fully restorable."""
```

返回的是**所有可完整恢复的前缀长度**（单位：页），不是那个最大值。为什么？

因为 SWA / 压缩状态这类池是 `TRAILING_PAGES` 语义：**只有在"某个曾经被 offload 的节点边界"上才存在窗口状态**。
举个例子，页 0..4，SWA 窗口宽 2 页：

| 前缀长度（页） | KV 是否齐（ALL_PAGES） | SWA 窗口 [len-2, len) 是否齐 | 整体可恢复？ |
|---|---|---|---|
| 1 | ✅ | 需要页 -1..0 → 只需页 0，✅ | ✅ |
| 2 | ✅ | 需要页 0,1 → ✅ | ✅ |
| 3 | ✅ | 需要页 1,2 → 页 2 的 SWA 缺 | ❌ |
| 4 | ✅ | 需要页 2,3 → ✅ | ✅ |

可恢复集合 = `{1, 2, 4}`，**不连续**。这时"局部最大值"是 4。

### 6.2 为什么不能各 rank 取自己的最大值再 reduce

假设 TP2：rank0 的可恢复集合是 `{1,2,4}`，rank1 是 `{1,3,4}`。

- 若各自先取最大值再 `MIN`：`min(4, 4) = 4`，恰好可行。
- 但若 rank0 是 `{2,4}`、rank1 是 `{3,4}`… 也还行。
- 换成 rank0 = `{2}`、rank1 = `{3}`：各自最大值 `min(2,3) = 2`，
  **但 rank1 根本没法恢复到 2 页**！于是 rank1 会读到一堆缺失的对象 → 报错或静默错数据。

结论：这必须是**集合求交**，不是数值 reduce。

### 6.3 用 0/1 mask + MIN 实现集合求交

```python
# unified_cache_linker.py:217
def _sync_restorable_prefix(self, restorable, *, num_pages, device_hit_pages):
    mask = torch.zeros(num_pages + 1, dtype=torch.int)
    for pages in restorable:
        if device_hit_pages < pages <= num_pages:
            mask[pages] = 1
    self.cache._all_reduce_attn_groups(mask, torch.distributed.ReduceOp.MIN)
    common = mask.nonzero()
    return 0 if common.numel() == 0 else int(common[-1].item())
```

技巧：**在 0/1 向量上，逐元素 MIN 就是逻辑 AND**，所以一次 `all_reduce(MIN)` 就完成了跨 rank 集合求交，
再取 `nonzero()[-1]` 得到"所有 rank 都能恢复的最长前缀"。
`mask[0]` 永远是 0，所以"无公共解"自然返回 0。

单元测试 `test_restorable_prefix_intersects_sparse_rank_results` 固化了这一点：
本地 `{2,4}` ∩ 远端 mask `{2}` = `{2}` → 返回 2（而不是 4）。

### 6.4 远端侧怎么算出这个集合

`MooncakeStore.batch_exists_v2`（`mooncake_store.py:831-897`）：

1. `restorable = list(range(1, kv_pages + 1))` —— 先假设所有长度都可以。
2. 逐个 pool 探测存在性，得到每页的 bool（一页的所有 component key 都存在才算存在）。
3. 按 policy 算该 pool 自己的可行集合：
   - `ALL_PAGES`：找第一个不存在的页作为 boundary，集合 = `range(1, boundary+1)`（连续）。
   - `TRAILING_PAGES`：窗口宽度 `trailing = len(transfer.keys)`，从大到小扫每个 `prefix_len`，
     检查 `[prefix_len - trailing, prefix_len)` 全在 → 收进集合（可以不连续）。
4. `restorable = [p for p in restorable if p in pool_set]` —— 逐 pool 求交，空了就提前返回。

注意 `DevicePoolGroup.kv_buffer = None`（`linker_pool_assembler.py:176`），
这让 `batch_exists_v2` 走"逻辑锚点"分支：`kv_pages = len(keys)`，**不单独查 KV 对象**。
KV 的存在性由 `resolve_transfers` 展开出的物理池（DSA 的 `PoolName.KV`，
DSV4 的 `DEEPSEEK_V4_C4/C128`）在第 3 步的循环里代为校验。

---

## 7. 写路径（显存 → 远端）

### 7.1 触发：`write_through_threshold = 1`

```python
# unified_cache_linker.py:151-152
cache.tree_core.enable_external_cache_linker = True
cache.write_through_threshold = 1
```

这个阈值有三层来源，注意别被 `256` 误导：

| 层 | 值 | 说明 |
|----|----|------|
| `UnifiedTreeCore.__init__`（`:403`） | `256` | 只是构造期占位，紧接着就被上层覆盖 |
| `UnifiedRadixCache.__init__`（`:470-472`） | `1` / `2` | `1 if hicache_write_policy == "write_through" else 2`；CLI 默认就是 `write_through`，所以 HiCache 常态也是 1 |
| `UnifiedCacheLinkerWrapper.__init__`（`:152`） | `1`（写死） | linker 模式**无条件**钉成 1，不再受 `--hicache-write-policy` 影响 |

写死成 1 的理由：远端存储容量近乎无限、路径上又没有中间的 host 拷贝，
**第一次插入就写出去**最划算，不需要"命中够多次才值得备份"这种针对有限 host 池的启发式。
副作用是 `write_back` / `write_through_selective` 两个策略在 linker 模式下静默失效。

判定逻辑（`unified_tree_core.py:915`）：

```python
if self.enable_external_cache_linker:
    return not node.external_cache_stored and node.hit_count >= self.write_through_threshold
```

### 7.2 写回链（chain）

`BackupKV` 动作携带的是**一条链**，不是单个节点。构造在 `unified_tree_core.py:2040-2052`：

```python
chain = [node]
ancestor = node.parent
while ancestor is not root and not (ancestor.backuped or ancestor.external_cache_stored):
    chain.append(ancestor)
    ancestor = ancestor.parent
chain.reverse()          # 祖先在前，保证"父存在才有子"的不变量
```

为什么要向上回溯？因为存储 key 是链式 hash，**父节点没落盘，子节点落了盘也没人能命中**
（`lookup` 从头逐页查，中间断一页 `ALL_PAGES` 就在那里截断）。反转顺序让祖先先写。

### 7.3 `offload_nodes` 逐节点处理

```python
# unified_cache_linker.py:461
def offload_nodes(self, node_ids):
    for node_id in node_ids:
        if not self.cache.resolve_node_handle(node_id).external_cache_stored:
            self._offload_node(node_id)
```

`_offload_node`（:467）做四件事：

1. 每个组件 `build_external_linker_transfer(OFFLOAD, node, None)`：从 `node.component_data[ct].value`
   拿显存槽、从 `node.hash_value` 拿 key。SWA 只取尾部窗口那几页并打 `TRAILING_PAGES`。
2. `inc_lock_ref(node_id)` —— **在传输完成前锁住节点，禁止被驱逐**（否则 RDMA 会读到已被复用的显存）。
3. `cache_linker.offload(transfers)` 排队；排队失败就立刻解锁并静默返回（不是错误，只是这次不写）。
4. `mark_write_through_pending(node_id)`、**乐观地**置 `node.external_cache_stored = True`，
   记入 `pending_offloads`。

> 第 4 步的"乐观"很重要：先标 True 可以避免同一节点在 in-flight 期间被重复排队。
> 如果最终失败，`commit_completed_offloads` 会把它改回 False（见 7.5）。

### 7.4 in-flight 期间节点被分裂怎么办

树是活的。一个正在写回的节点，可能被另一个请求的 insert **分裂成父+子两个节点**。
`_split_node` 检测到 `write_through_pending_id is not None` 时会发一个
`ReplaceWriteThroughOnNodeSplit` 动作（`unified_tree_core.py:1188-1196`），
`_apply_cache_action` 同时通知 HiCache 侧和 linker 侧（`unified_radix_cache.py:1074-1085`）。

linker 侧的 `replace_pending_offload_node`（`unified_cache_linker.py:492`）
把 `publish_node_ids` 里的老 id 换成新的两个 id：

```
pending_offloads[i].publish_node_ids: [7]  →  [8, 7]      # 8 是新父，7 是被截短的子
pending_offloads[i].lock_node_id:      7   →  7（不变，ack 仍认这个 id）
```

`lock_node_id` 不变是关键：解锁时仍然只对原锚点 `dec_lock_ref` 一次
（`inc_lock_ref` 会沿根路径打引用，分裂出来的新父节点天然在这条路径上）。

单元测试 `test_failed_offload_rolls_back_split_fragments` 固化：失败后父子**两个**节点的
`external_cache_stored` 都被回滚成 False，但只 `dec_lock_ref` 了 `child.id` 一次。

### 7.5 完成确认必须跨 rank 对齐

```python
# unified_radix_cache.py:2787
def check_hicache_events(self):
    if self.linker is not None:
        finish_counts = torch.tensor(
            [self.linker.num_completed_loads(), self.linker.num_completed_offloads()],
            dtype=torch.int, device="cpu")
        self._all_reduce_attn_groups(finish_counts, ReduceOp.MIN)   # ← 取各 rank 的最小完成数
        load_count, offload_count = map(int, finish_counts.tolist())
        self.linker.drain_loads(load_count)
        local_successes = self.linker.take_completed_offloads(offload_count)
        if local_successes:
            successes = torch.tensor(local_successes, dtype=torch.int, device="cpu")
            self._all_reduce_attn_groups(successes, ReduceOp.MIN)   # ← 成功也要 AND
            self.linker.commit_completed_offloads([bool(s) for s in successes.tolist()])
        return
```

两次 all_reduce，两个不同的目的：

| all_reduce | op | 语义 |
|---|---|---|
| `finish_counts` | MIN | **进度对齐**：只处理"所有 rank 都完成了"的那部分，保证各 rank 的树状态一步不差 |
| `successes` | MIN（0/1 上 = AND） | **成功对齐**：任一 rank 写失败，所有 rank 都必须把 `external_cache_stored` 回滚成 False |

第 2 条为什么必要：如果 rank0 写成功、rank1 写失败，而各自按本地结果标记，
那么下次 `lookup` 时 rank0 会说"能恢复"、rank1 会说"不能"。虽然 `_sync_restorable_prefix` 会求交集兜住不出错，
但树的 `external_cache_stored` 会永久分叉，rank1 那一份数据再也不会被补写。

`num_completed_offloads()` 还额外做了一层 clamp（`unified_cache_linker.py:509`）：

```python
return min(self.cache_linker.num_completed_offloads(), len(self.pending_offloads))
```

正常情况下后端结果队列和 `pending_offloads` 是 1:1 的（`offload()` 返回 True 才 append），
这层 clamp 是防御性的——`take_completed_offloads` 里有 `assert finish_count <= len(self.pending_offloads)`，
clamp 保证它不会因为后端多吐了结果而崩掉。

---

## 8. L2：设备池装配层

文件：[hybrid_cache/linker_pool_assembler.py](../../../python/sglang/srt/mem_cache/hybrid_cache/linker_pool_assembler.py)

这一层的职责一句话：**把"第 3 页、第 7 页"这种逻辑页号，翻译成"某块显存的裸指针 + 字节长度 + 层内偏移"。**

### 8.1 `DevicePoolEntry`：一个物理显存池的零拷贝视图

构造参数里最关键的三个：

| 参数 | 含义 |
|---|---|
| `components: Sequence[Sequence[Tensor]]` | 外层 = 组件（如 K、V 两个 buffer 组），内层 = 逐层的 tensor |
| `layer_mapping: dict[int, int]` | 全局层号 → 这个池内部的 buffer 下标。**不在表里的层直接跳过**（返回 None） |
| `rows_are_pages: bool` | buffer 的第 0 维是"页"还是"token"。决定 `_row_span` 是 1 还是 `page_size` |

`__init__` 里预计算好 `buffer_meta`：每个 buffer 的 `(data_ptr, 行字节步长, 一行/一页的字节数)`，
以及 `_component_offsets`（`packed=True` 时把多个组件的字节区间**首尾相接**，
这样一个存储对象里就顺序装着 K、V、…）。

三个对外方法：

```python
prepare_locations(indices) -> list[int]           # 页对齐校验 + 换算成行号
get_prepared_layer_range_meta(locations, layer)   # → (ptrs, sizes, offsets)，某一层的字节范围
get_hybrid_pool_buffer() -> list[Tensor]          # 给后端注册 RDMA buffer 用
```

`_rows()`（:80）做了很严的校验，任一不满足就 `ValueError`：

- `indices.numel() % page_size == 0`
- 每页内部必须是 `start, start+1, ..., start+page-1` 的**连续**且 `start % page_size == 0` 的**对齐**序列
- 行号范围不能越出 buffer 形状

这个"必须页对齐且页内连续"的约束，正是 external linker 只能按整页搬运的根本原因。

### 8.2 `DevicePoolGroup.resolve_transfers`：逻辑 → 物理的展开

```python
# linker_pool_assembler.py:178
for name, source_name in self.sources.items():      # sources: 物理池 → 逻辑池
    source = by_name.get(source_name)               # 拿逻辑 transfer（KV 或 SWA）
    resolved.append(replace(source,
        name=name,                                   # 改成物理池名
        host_indices=entry_map[name].translate_indices(source.device_indices),
        keys=list(source.keys),                      # key 复用逻辑池的页 hash
        hit_policy=ALL_PAGES if source_name == KV else source.hit_policy,
    ))
```

要点：

- **多个物理池共享同一份页 hash**（`keys`），靠 `_get_hybrid_page_component_keys` 加的池名后缀区分对象。
- 三个开关控制严格度：
  - 默认（`lookup` 用）：必须所有 source 逻辑池都在，且 KV 必须非空。
  - `allow_partial=True`（`offload` 用）：允许缺池。
  - `allow_partial=True, allow_missing_kv=True`（`load` 用）：连 KV 都可以缺（见 5.3）。
- `translate_indices` 走可选的 `index_mapper`，给未来需要"逻辑页号≠物理页号"的池留了口子（目前两个 strategy 都没用）。

### 8.3 只有两种模型形态被支持

入口 `resolve_hybrid_device_pool_group()`（:364）复用 HiCache 的 strategy 注册表 `_select_strategy`，
但基类 `StackStrategy.build_direct_linker_pool_group` **默认直接抛错**
（`hybrid_pool_assembler.py:1191-1201`："The selected hybrid pool strategy does not support the direct external linker"）。

只有两个 strategy 覆写了它：

| Strategy | 匹配条件 | 展开出的物理池 | `rank_replicated` |
|---|---|---|---|
| `_DsaStrategy`<br>(`:1502`) | `DSATokenToKVPool` 且 components == `{FULL}` | `KV`(rows_are_pages=False) + `INDEXER`(True) | **True** |
| `_DeepSeekV4Strategy`<br>(`:1231`) | `DeepSeekV4TokenToKVPool` 且 components == `{FULL, SWA}` | `SWA` + `C4` + `C4_INDEXER` + `C128` + `C4_STATE` + `C4_INDEXER_STATE` | **True** |

**普通 MHA / 普通 MLA 模型目前都不在支持列表里**——`_select_strategy` 会选到别的 strategy，
然后在基类里抛 `ValueError`。

DSV4 分支还有两条额外硬拒（`linker_pool_assembler.py:246-256`）：

```python
if kvcache._unified_kv or isinstance(kvcache.c4_kv_pool, HiSparseC4DevicePool):
    raise ValueError("The direct external linker does not support unified-KV or HiSparse.")
if kvcache.swa_page_size != page_size:
    raise ValueError("DeepSeek V4 SWA page size must match the tree page size")
```

注意两个 strategy 都是 `rank_replicated=True`（MLA 系）。这直接决定了后面的 key 命名和写入 owner。

### 8.4 `rank_replicated` 的两个后果

```python
# mooncake_direct_linker.py:110-111
rank_replicated = self.pool_group.rank_replicated
self.offload_owner = not rank_replicated or tp_rank == 0

# :30
def _storage_suffix(*, rank_replicated, tp_rank, attn_cp_rank, pp_rank):
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")     # 只有非复制时 key 里才带 tp rank
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)
```

| | `rank_replicated=True`（MLA/DSA/DSV4） | `rank_replicated=False`（MHA） |
|---|---|---|
| key 后缀 | `cp{k}_pp{k}`，**不含 tp** → 各 rank key 相同 | `tp{k}_cp{k}_pp{k}` → 各 rank key 不同 |
| 写入者 | **只有 tp_rank 0 真写**，其它 rank 直接 `offload_results.put(True)` 假 ack | 每个 rank 都写自己那份 |
| 读取者 | **所有 rank 都读同一个对象**到自己的显存（KV 复制语义，正确） | 各读各的分片 |

`is_mla_model=rank_replicated` 也被塞进 `HiCacheStorageConfig`（`:122`），影响 `mooncake_store` 内部的 key 布局选择。

---

## 9. L3：`MooncakeDirectLinker` 后端实现

文件：[storage/mooncake_store/mooncake_direct_linker.py](../../../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_direct_linker.py)（415 行）

### 9.1 构造期干的事

1. `resolve_hybrid_device_pool_group(...)` → 得到 `pool_group` / `pools` / `num_layers`。
2. 解析 TP 拓扑（优先用 `attn_tp_cache_group`，退化到 `tp_cache_group`），算出 `offload_owner` 和 key 后缀。
3. 构造或接管一个 `MooncakeStore`，然后**直接改它的三个属性**：

```python
self.storage.mem_pool_host = self.pool_group      # 冒充 host 池（实际是显存池）
self.storage.registered_pools = self.pools        # 整体替换，KV 也在里面
self.storage.mla_suffix = self.storage.mha_suffix = storage_suffix
```

> 这是复用 `mooncake_store` v2 IO 代码的手段：让它以为自己在操作 host 池，实际拿到的全是显存指针。
> 也因此 `mooncake_store.register_mem_host_pool_v2()`（会跳过 `PoolName.KV`）**被完全绕过**，
> `registered_pools` 里 KV 是齐的。

4. `register_buffers()`（:186）—— **核心的 GPU-Direct 步骤**：

```python
for pool in self.pools.values():
    for buffer in pool.get_hybrid_pool_buffer():
        storage = buffer.untyped_storage()
        result = self.storage.store.register_buffer(storage.data_ptr(), storage.nbytes())
```

用 `untyped_storage()` 而不是 `tensor.data_ptr()` + 去重 `(ptr, nbytes)`，
是因为多个层的 tensor 常常是同一块大 storage 的 view，重复注册会报错。

5. 建 `LayerWiseLoadCounter`，若有 MAMBA 池还要注册到 `req_to_token_pool`。
6. 起**两个 daemon 线程**：`mooncake-load-tp{k}` 和 `mooncake-offload-tp{k}`。

### 9.2 线程模型与队列

```
Scheduler 线程                          load 线程                    offload 线程
─────────────                          ─────────                    ────────────
load(rid, transfers)
  → pending_loads[rid] = expanded
     （只是攒着，不发起 IO）

start_layer_wise_loading()
  → counter_index = counter.update_producer()
  → device Event.record()
  → load_queue.put((idx, pending, ev)) ──→ ev.synchronize()
                                           load_layer_wise():
                                             batch_get_session_start(keys)
                                             for layer in 0..L-1:
                                               batch_get_into_multi_buffer_ranges(...)
                                               counter.complete(idx, layer) ──┐
                                             batch_get_session_end(keys)      │
                                           completed_loads.put(rids)          │
                                                                              │
模型 forward 第 layer 层                                                       │
  → counter.wait_until(layer)  ←───────── Future.result() 被唤醒 ─────────────┘

offload(transfers)
  → device Event.record()
  → offload_queue.put(...) ─────────────────────────────────→ ev.synchronize()
                                                              batch_set_v2(expanded)
                                                              offload_results.put(success)
check_hicache_events()
  ← completed_loads / offload_results 的 qsize()
```

### 9.3 `LayerWiseLoadCounter`：用 Future 做逐层同步

HiCache 的 `layer_done_counter` 是 CUDA event 数组；这里因为传输是 CPU 线程发起的网络 IO，
换成了纯 CPU 的 `concurrent.futures.Future` 数组（`:40-81`），但**保持了完全相同的接口**
（`update_producer / set_consumer / wait_until`），所以 KV pool 的逐层等待钩子可以原样复用。

```python
def update_producer(self) -> int:                 # 新一批，每层一个 Future
    self.producer_index += 1
    self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
    return self.producer_index

def wait_until(self, threshold: int) -> None:     # 模型第 threshold 层调用
    futures[threshold].result()                   # 阻塞直到该层数据到位（或抛异常）
```

失败传播设计得不错：`fail(index, error)` 把异常塞进**该批所有未完成的 Future**，
于是 forward 线程会在 `wait_until` 里拿到 `RuntimeError("Mooncake layer-wise KV load failed.")` 而不是死等。
`load_thread_func` 的 `finally` 里**无论成功失败都 `completed_loads.put(...)`**，
保证 wrapper 的 `pending_loads` 锁一定会被释放，不会泄漏引用计数。

`wait_until` 在最后一层时 `self.futures.pop(index)` 清理，避免 dict 无限增长。

### 9.4 逐层加载为什么能省延迟

一个存储对象里装的是**一个页的所有层**的 KV（key 里没有层号，见 9.5）。
`batch_get_into_multi_buffer_ranges(keys, ptrs, sizes, offsets)` 的意义是：
**同一批 key，只读其中某段字节范围**（该层对应的 offset/size），落到该层的显存指针上。

于是可以做成流水：第 0 层数据一到，模型就能开始算第 0 层，同时第 1 层还在传。
`for layer in range(self.num_layers)` + `counter.complete(idx, layer)`（`:312-337`）就是这个流水。

`batch_get_session_start/end` 是 Mooncake 的会话机制，把"同一批 key 的多次 range 读"
框在一个会话里，避免每层重复做元数据查询。`finally` 里保证 session 一定被关闭。

### 9.5 存储 key 的完整形状

```
{config_prefix}_{page_hash}_{rank_suffix}_{component}
     │              │             │            └── _k / _v / _{pool_name} / _draft_k ...
     │              │             └── mla_suffix / mha_suffix，这里被覆写成 cp{k}_pp{k}
     │              └── 链式 hash（get_hash_str）
     └── {extra_backend_tag}_{model_path 里 / 换成 -}
```

- **有模型名隔离**（`config_prefix`），不同模型不会撞 key。
- **没有层号**：一个对象覆盖所有层，层粒度靠字节 offset 表达。
- **没有 layout 握手**：同一个模型换 `page_size` / dtype / head 切分方式，
  算出来的 key 一模一样但 payload 不兼容。唯一的护栏是读回来时对字节数做校验
  （`:326-336` 要求 `list(result) == [sum(item) for item in sizes]`），
  长度恰好相同的错配布局**会静默读出错数据**。这一点和投机解码 draft 池的情况类似
  （见 [spec_decoding_cache_centralized_vs_pd.md](../05_speculative_decoding/spec_decoding_cache_centralized_vs_pd.md)）。

### 9.6 两个小优化

- `freeze_gc_once()`（`:245`）：首次有 load/offload 流量时调 `freeze_gc()`，
  把成熟的模型对象图移出循环 GC 扫描范围。因为构造 transfer 元数据会产生大量短命 list，
  否则每次 GC 都要扫一遍整个模型图。
- **device Event 同步**：`load` / `offload` 排队前 `Event.record()`，
  IO 线程 `ready_event.synchronize()`。保证 RDMA 读写显存时，
  之前 scheduler 流上对这块显存的 kernel（分配/驱逐/上一步 forward）已经完成。
  这是跨线程访问显存的必要屏障。

---

## 10. 跨 rank 一致性：全部同步点汇总

外部 linker 模式下，radix tree 的形状必须在同一 attention 组的所有 rank 上**完全一致**——
因为 tree 决定显存 slot 的分配与驱逐。某个 rank 多插了一段前缀，它的可用 token 数就和别人不同，
很快会走向不同的调度决策，直到 collective 对不上而 hang。

所以凡是"远端存储的状态"要影响 tree 形状的地方，都必须做集合归约。一共三处显式归约：

| # | 位置 | 归约对象 | 算子 | 为什么需要 |
|---|------|---------|------|-----------|
| 1 | `_sync_restorable_prefix`（`unified_cache_linker.py:217-235`） | 长度 `num_pages+1` 的 0/1 mask | `MIN` | 各 rank 的可恢复长度集合是**稀疏**的（SWA 窗口所致），必须取交集，不能取 max 再 min。见 §6 |
| 2 | `check_hicache_events` 第一次（`unified_radix_cache.py:2789`） | `finish_count`（本 rank 已完成的 offload 个数） | `MIN` | 各 rank IO 线程进度不同，只能按最慢的 rank 推进；否则 rank A 已 commit 第 3 个而 rank B 还没做完第 2 个 |
| 3 | `check_hicache_events` 第二次 | 前 `finish_count` 个 offload 的成功 flag 向量 | `MIN` | 布尔上 MIN 就是 AND。只要有一个 rank 写失败，`external_cache_stored` 必须全体置 `False`，否则下次 lookup 会有 rank 声称命中、有 rank 声称没命中 |

### 10.1 为什么必须两次 all_reduce，不能合并成一次

第一次归约决定"这一轮取几个结果"，第二次归约的**向量长度取决于第一次的结果**。
若只做一次（直接归约成功 flag 向量），各 rank 的向量长度不同，collective 本身就无法对齐。
这是天然的两阶段依赖，不是冗余。

### 10.2 `rank_replicated` 带来的读写不对称

`rank_replicated=True`（DSA 与 DSV4 两个策略都是 True）意味着**所有 tp rank 的 KV 内容
按定义相同**（MLA 的 latent KV 不按 head 切）。于是：

| 行为 | 实现 | 效果 |
|------|------|------|
| key 后缀省略 `tp{k}` | `_storage_suffix`（`mooncake_direct_linker.py:60-70`）在 `rank_replicated` 时不 append tp 段 | 所有 rank 算出**同一个 key** |
| 只有 tp_rank 0 真写 | `self.offload_owner = not rank_replicated or tp_rank == 0`（`:111`） | 远端只存一份，省 `(tp-1)/tp` 的存储与写带宽 |
| 非 owner 假 ack | `offload()` 里 `if not self.offload_owner: self.offload_results.put(True); return True` | 非 owner 也进 `pending_offloads`，让 #2/#3 的归约能对齐，且 `external_cache_stored` 在所有 rank 上同步翻成 True |
| 所有 rank 都真读 | `load()` 无 owner 判断 | 每个 rank 独立把同一个对象读进自己的显存 |

这个"假 ack"是让归约成立的关键：如果非 owner 直接返回 `False` 不入队，
`finish_count` 的 MIN 永远是 0，写通会彻底停摆。

> 反过来看，若将来支持 `rank_replicated=False` 的策略（真正按 head 切分的 MHA），
> 表里第 2、3 行都会失效，需要每个 rank 各写自己的 `tp{k}` key。
> 代码结构已经预留了这条路（`_storage_suffix` 的分支、`offload_owner` 的 `not rank_replicated or`），
> 只是目前没有策略走到。

---

## 11. 生命周期与锁：显存 slot 什么时候被钉住

远端 IO 是**异步**的：`load` / `offload` 只是把任务塞进队列，真正的 RDMA 发生在后台线程。
在此期间，被读写的那段显存**绝对不能**被 radix tree 驱逐或重新分配——否则 RDMA
会写进别的请求的 KV。钉住的手段就是 tree 自己的 `lock_ref` 引用计数。

### 11.1 两个 inc_lock_ref 点，四条释放路径

| 加锁点 | 锁哪个节点 | 正常释放 | 异常释放 |
|--------|-----------|---------|---------|
| `_queue_load`（`:358`） | `insert_result.last_device_node`，即刚插进树的尾节点 | `drain_loads` 收到该 rid 的完成通知后 `dec_lock_ref`（`:517-521`） | 排队抛异常 → 立即 dec 再 raise（`:361-363`）；`load()` 返回 False → dec 后 raise `RuntimeError`（`:364-366`） |
| `_offload_node`（`:478`） | 写通链上的那个节点 | `commit_completed_offloads` 里逐个 `dec_lock_ref`（`:536`） | 同上：抛异常 → dec 再 raise（`:481-483`）；`offload()` 返回 False → dec 后**静默 return**（`:484-486`） |

两处的 `queued == False` 处理不同，是刻意的：

- **load 失败必须炸**。树已经 `insert` 完了，尾节点已经对外可见并声称"这段前缀在显存里有效"，
  但数据永远不会到达。继续跑就是静默读脏数据，所以宁可 `RuntimeError`。
- **offload 失败可以忍**。写通只是优化，不写就是少一次远端命中，语义上无损。
  注意此时不会置 `external_cache_stored = True`（那一行在 `if not queued: return` 之后），
  所以下一次 `_inc_hit_count_and_check` 还会再挑中它重试。

### 11.2 pending 状态机

```
pending_loads : dict[rid -> (node_id, dec_lock_params)]
pending_offloads : list[_PendingOffload(lock_node_id, lock_params, publish_node_ids)]
```

- `pending_loads` 用 dict 按 rid 索引，因为 `release_request(rid)` 需要按 rid 定点取消。
- `pending_offloads` 用 **list（FIFO）**，因为后端的 `offload_results` 是一个队列，
  第 *k* 个弹出的结果就对应第 *k* 个入队的 offload。`commit_completed_offloads` 里
  `self.pending_offloads.pop(0)` 依赖这个顺序。
- `_PendingOffload` 把"加锁的节点"和"要发布状态的节点集合"**分开存**：正常情况两者相同，
  但节点在 IO 途中被 split 之后，一个 offload 对应的就是**多个**节点。
  `replace_pending_offload_node`（`:492-507`）把 `publish_node_ids` 里的旧 id
  换成 split 出的新 id 列表，`lock_node_id` 保持不变（锁是当初加在旧 id 上的，
  split 会把 lock_ref 传给新节点，但 ack 的身份标识仍是旧 id）。

### 11.3 `num_completed_offloads` 的 `min` 夹取

```python
def num_completed_offloads(self) -> int:
    return min(self.cache_linker.num_completed_offloads(), len(self.pending_offloads))
```

后端的结果队列长度和 `pending_offloads` 长度在正常路径上是 1:1 的
（`offload()` 返回 True 是二者同时增长的唯一条件）。这个 `min` 是纯防御性的，
保护下游 `take_completed_offloads` 里的 `assert finish_count <= len(self.pending_offloads)`
——让后端实现的计数 bug 退化成"少 commit 几个"（下一轮补上），而不是 assert 崩掉调度器。

### 11.4 关停顺序：先静默后端，再放锁

`reset()` / `close()` 的顺序被单测钉死了：

```python
def reset(self) -> None:
    self.cache_linker.reset()      # ① 先让后端把 in-flight 排空
    self.hit_markers.clear()
    self._release_pending_locks()  # ② 才敢放锁
```

后端的 `reset()`（`mooncake_direct_linker.py:391-405`）会 `load_queue.join()` +
`offload_queue.join()`，即**阻塞等到队列里所有任务都 `task_done`**，再清空结果队列、
reset layer counter。只有这一步返回之后，才能确定没有任何后台线程还在碰那些显存。

如果顺序反了——先 `dec_lock_ref` 再等后端——tree 就可能在 RDMA 还在进行时驱逐并
重分配那些 slot，写坏别的请求。`test_reset_quiesces_backend_before_releasing_pending_locks`
用一个记录调用顺序的 fake 后端断言 `events == ["backend", ("unlock", 7), ("unlock", 7)]`
把这个顺序固化下来。

`_release_pending_locks`（`:548-559`）在放锁时还要**回滚已发布的状态**：
清掉 `write_through_pending_id`、把 `external_cache_stored` 打回 `False`。
因为 `_offload_node` 是**乐观**地先把 `external_cache_stored = True` 写上去的
（避免同一节点被重复排队），reset 时这条写通并没有真正完成，必须撤销，
否则重启后的 lookup 会声称远端有这份数据。

`close()` 则是 `reset()` 之后再往两个队列各塞一个 `None` 哨兵、`join()` 两个线程、
打印 `stats`、关闭 storage（`:407-414`）。

### 11.5 `release_request`：请求提前结束

```python
def release_request(self, rid: str) -> None:
    self.hit_markers.pop(rid, None)
    # TODO: Roll back the published tree and component state atomically before
    # canceling; otherwise the tree may retain device slots that were never loaded.
    if self.cache_linker.cancel_queued_load(rid):
        node_id, lock_params = self.pending_loads.pop(rid)
        self.cache.dec_lock_ref(node_id, lock_params)
```

`cancel_queued_load` 只在任务**还没被 IO 线程取走**时才能成功取消（后端实现是从
`pending_loads` 集合里 discard，取走时检查是否还在集合里）。返回 True 表示取消成功，
此时才放锁。返回 False 表示已经在飞或已完成，锁交给 `drain_loads` 正常路径处理。

代码里的 TODO 指出一个真实缺陷：取消成功意味着**这段前缀的 KV 永远不会到达显存**，
但树里那些节点已经在 `load_back` 里 `insert` 进去了，还被标了
`external_cache_stored = True`。后续请求匹配到这段前缀会直接当命中用，**读到未初始化的显存**。
这是当前实现最需要注意的一处未收口的正确性问题。

---

## 12. 与 HiCache 的逐项对照

两条路径在 `UnifiedRadixCache` 里是**互斥的 if/else**，共用同一批 hook 点但实现完全不同。

| 维度 | HiCache（`enable_storage`） | External Linker（`linker is not None`） |
|------|---------------------------|--------------------------------------|
| 层级 | L1 显存 ↔ L2 host ↔ L3 storage，三层 | L1 显存 ↔ 远端，两层 |
| 中转内存 | 需要 `mem_pool_host`，占用固定的 pinned host 内存 | 不需要，GPU-Direct RDMA 直达 |
| 节点状态 | `host_value`（host slot 张量）+ `backuped` + `host_ref_counter` | 单个 `bool external_cache_stored` |
| 远端是不是"层" | L2 是真正的缓存层，有自己的 LRU、驱逐、容量 | 远端**不是层**，只是一个 bool 标记；没有本地元数据 |
| 命中查询 | 先查 tree 的 `host_value`（本地内存操作），L3 再走 prefetch 状态机 | 每次都要 RPC 到远端 `batch_exists_v2` |
| 跨 rank 一致性 | L2 由每个 rank 各自持有，形状天然一致；L3 走 prefetch 的 revoke 机制 | 必须显式 `all_reduce(MIN)` 三处（见 §10） |
| 写通阈值 | `1`/`2`，跟随 `--hicache-write-policy` | 写死 `1`，策略失效 |
| 写通粒度 | 先 D2H 进 L2（同步或异步），L3 由 backup 线程再搬 | 直接 device → 远端，一步 |
| 加载分配 | host slot 已有数据，只需分配 device slot 做 H2D | component `PREPARE` 阶段分配 device slot（含必要的 `evict`），RDMA 直接落进去 |
| 事务性 | L2 写失败可以留在 L2、下次再 backup | 三阶段 `PREPARE / COMMIT / ABORT`，任一 component 拿不到资源就整体 ABORT |
| 失败处理 | 节点保持 `backuped=False`，下轮重试 | `external_cache_stored` 回退 `False`，下轮 `_inc_hit_count_and_check` 重新挑中 |
| 逐层加载同步 | CUDA event 的 `layer_done_counter` | `LayerWiseLoadCounter`（`concurrent.futures.Future`），接口同形 |
| 事件轮询 | `check_hicache_events` 的 else 分支 | `check_hicache_events` 的 linker 分支，多两次 all_reduce |
| 适用场景 | 单机/小集群，host 内存充裕，想吃 L2 的低延迟 | 大规模共享 KV 池，多实例复用同一份前缀，host 内存不想为缓存买单 |

### 12.1 一个容易混淆的点：`last_host_node` / `host_hit_length`

linker 模式**复用了 HiCache 的 `MatchResult` 字段名**（`match_prefix` 返回的
`last_host_node` / `host_hit_length` / `swa_host_hit_length` / `mamba_host_hit_length`）。
`_ext_match` 里的 `result._replace(last_host_node=result.best_match_node, host_hit_length=hit_tokens, ...)`
填的是"远端可恢复的长度"，**跟 host 内存毫无关系**。

这么做是为了让调度器侧的 `init_load_back` / `ready_to_load_host_cache` 那套流程
一行不改就能同时服务两条路径。代价是读代码时容易误以为经过了 host 中转。
同样的命名陷阱也出现在 `PoolTransfer.host_indices` 上——linker 路径里它装的是
**device 行号**（见 §4.1）。

---

## 13. 当前限制与粗糙边缘

### 13.1 生产入口（基线之后已接线）

截至基线 `33ed29a0ee`，这套东西**还没有生产入口**（`init_cache_linker` 只有单元测试调用、`MooncakeDirectLinker` 零实例化、无 CLI/env）。基线之后已补齐：

- CLI 开关 `--enable-unified-cache-external-linker` +
  `--unified-cache-external-linker-backend`（`mooncake` / `mori`），见 §0。
- 生产实例化在 `mem_cache/registry.py:199-224`，与 `--enable-hierarchical-cache` 互斥。

也就是说 [1/N]~[4/N] 落地了契约、池装配、树侧核心、Mooncake 后端，随后的 PR 把"用什么开关打开它"补上了，并新增了 `mori`（UMBP）后端。

### 13.2 模型 / 策略覆盖面很窄

| 限制 | 位置 | 表现 |
|------|------|------|
| Mamba / 混合线性注意力不支持 | `components/mamba.py:655` | `raise AssertionError("MambaComponent does not support external linker mode, will support soon")` |
| 只支持 DSA 和 DSV4 两个策略 | `hybrid_pool_assembler.py` 基类 | 普通 MHA / 普通 MLA 走到基类实现直接抛异常。也就是说 **Llama / Qwen 这类主流模型用不了** |
| DSV4 unified-KV：基线之后已支持 | `linker_pool_assembler.py:273-342`（`is_unified_kv` 分支，#38269） | 基线时拒绝，现已实现统一 KV 布局的行号映射（含 AMD） |
| DSV4 拒绝 HiSparse | `linker_pool_assembler.py:269-270` | 稀疏索引池与 linker 的页对齐假设冲突 |
| DSV4 非 unified-KV 时要求 SWA 与主池 page_size 一致 | `linker_pool_assembler.py:276-280` | 不一致时对齐断言无法满足；unified-KV 分支不走此校验 |

`resolve_hybrid_device_pool_group` 的失败路径带上下文信息
（`test_unsupported_strategy_fails_with_context` 钉住了这一点），所以至少不会静默降级。

### 13.3 没有布局握手 → 静默数据错乱风险

key 里只有 `{模型名}_{页 hash}_{cp/pp rank}_{component}`，**不含**
`page_size` / dtype / head 切分方式 / 层数。同一个模型用不同配置启动两个实例并共享
同一个 Mooncake 池，算出的 key 完全相同，payload 却不兼容。

唯一护栏是读回时的字节数校验（`mooncake_direct_linker.py:326-336`：
`list(result) == [sum(item) for item in sizes]`）。字节数**恰好相同**的错配布局
（例如 head 切分方式改了但总量不变）会静默读出错数据。

这与投机解码 draft 池跨 PD 无握手校验是同一类问题，见
[spec_decoding_cache_centralized_vs_pd.md](../05_speculative_decoding/spec_decoding_cache_centralized_vs_pd.md)。

### 13.4 `release_request` 的非原子回滚

已在 §11.5 展开：取消排队中的 load 之后，树里已发布的节点没有回滚，
后续请求会把未初始化的显存当命中用。源码里有明确的 TODO。

### 13.5 `batch_set_v2` 没有批级原子性

`batch_set_v2` 返回的是 `dict[PoolName, list[bool]]`——**逐页逐池**的结果。
一个 page 的 KV 写成功但 SWA 写失败是可能的。上层的处理是
`success = all(all(pool_results) for pool_results in results.values())`
（`:373`），整体判失败后把 `external_cache_stored` 置 `False`。

这在语义上是安全的（不会声称命中），但**远端会留下半写的垃圾对象**。
下次重试时依赖 Mooncake 的 write-once 去重：已存在的 key 不会重写，
所以那个成功的一半会被认为"已写"，而失败的一半重写。最终会收敛，
但中间态下这个 page 永远查不出可恢复（`batch_exists_v2` 要求所有池都在）。

---

## 14. 测试与可观测性

### 14.1 两个单测文件覆盖了什么

**`test/registered/unit/mem_cache/test_unified_cache_linker.py`**（11 个 case，全部用
`UnifiedRadixCache.__new__` + `SimpleNamespace` 假造依赖，不需要 GPU）：

| 用例 | 钉住的失败模式 |
|------|--------------|
| `test_cache_linker_attachment_is_backend_independent` | 挂载副作用完整：`enable_external_cache_linker=True`、阈值变 1、`layer_done_counter` 透传 |
| `test_restorable_prefix_intersects_sparse_rank_results` | 本地 `{2,4}` ∩ 远端 `{2}` → 2。若退化成"取各 rank 最大值再 min"会得到 4，测试变红 |
| `test_async_offload_pins_node_until_completion` | offload 排队期间节点被 `inc_lock_ref` 钉住，commit 后才解锁 |
| `test_async_load_pins_node_until_completion` | load 同上 |
| `test_release_request_cancels_queued_load` | 取消成功时锁被释放，不泄漏 |
| `test_failed_offload_rolls_back_split_fragments` | 写失败时 split 出来的**所有**碎片都要回滚 `external_cache_stored` |
| `test_split_action_retargets_pending_external_offload` | in-flight 节点被 split 后，`publish_node_ids` 换成新 id 列表 |
| `test_reset_quiesces_backend_before_releasing_pending_locks` | 断言 `events == ["backend", ("unlock", 7), ("unlock", 7)]`——**顺序**不能反 |
| `test_close_quiesces_backend_before_releasing_pending_loads` | 同上，close 路径 |
| `test_check_hicache_events_commits_common_rank_results` | 双 all_reduce 的语义：只 commit 所有 rank 共同完成且共同成功的部分 |
| `test_component_commit_keeps_only_adopted_pages` | 4 页、`adopted_ranges=[(2,4),(6,8)]` → keys `[a,b,c,d]` 收缩成 `[b,d]`。删掉它就没有东西阻止 COMMIT 往已释放的 slot 里 RDMA |

**`test/registered/unit/mem_cache/test_linker_pool_assembler.py`**（7 个 case）：

| 用例 | 钉住的失败模式 |
|------|--------------|
| `test_sparse_multi_component_layer_ranges` | `DevicePoolEntry` 的稀疏层区间元数据计算 |
| `test_rejects_invalid_pages_and_empty_buffers` | 非法页数 / 空 buffer 必须拒绝，不能静默产出错误 offset |
| `test_resolve_transfers_expands_physical_pools` | 逻辑 transfer（KV/SWA）→ 物理池的展开完整性 |
| `test_partial_side_pool_requires_explicit_opt_in` | 缺池时默认返回空列表；只有 `allow_partial=True` 才允许部分 |
| `test_deepseek_v4_maps_sparse_sidecars` | DSV4 的 6 个物理池映射（注册表完整性类断言） |
| `test_dsa_uses_hybrid_assembler_strategy` | DSA 走对了策略分支 |
| `test_unsupported_strategy_fails_with_context` | 不支持的模型抛异常且带上下文，不静默降级 |

### 14.2 日志与统计

`MooncakeDirectLinker` 维护 `self.stats = {"lookup": 0, "load": 0, "offload": 0}`（`:172`），
三个计数的语义各不相同，别搞混：

| 计数 | 累加位置 | 语义 |
|------|---------|------|
| `lookup` | `:212`，**无条件** | 调了多少次 `lookup`，**不管有没有命中** |
| `load` | `:264`，`+= len(pending)` | 累计**请求数**（一次 `start_layer_wise_loading` 可能带多个 rid） |
| `offload` | `:375`，仅 `success` 时 | 累计**成功**写出的节点数 |

日志：

- `:213-219` 每次 lookup **命中就打一条** info（`rid` / 最长可恢复页数 / 候选集大小）。
  这是**逐请求**日志，高 QPS 下会很吵，生产启用时可能要降级成 debug。
- `:376-377` offload 只在**第一次**成功时打一条（带 token 数）。
- load 侧没有成功日志，只有失败的 `logger.exception("Mooncake layer-wise load batch failed")`。
- `close()` 打印全量汇总：`Mooncake direct linker stats: {'lookup': N, 'load': M, 'offload': K}`。

排查时可用的推论：

- `lookup` 涨但 `load == 0` → 远端根本没命中（lookup 计数含未命中），或
  `_sync_restorable_prefix` 的跨 rank 交集为空，或 `insert` 发现这些页 L1 已有
  （`allow_partial` 路径把 transfer 全过滤掉了）。配合 `:213` 那条命中日志有没有出现即可区分第一种。
- `offload == 0` → 先确认是不是非 owner rank。`rank_replicated` 下只有 tp0 真写，
  非 owner 的 `stats["offload"]` 永远是 0，这是**正常**的（见 §10.2）。
- 加载失败会以 `RuntimeError("Mooncake layer-wise KV load failed.")` 的形式
  从 `LayerWiseLoadCounter.wait_until` 抛到 forward 线程，而不是静默 hang。

---

## 15. 一页总结

如果只记三件事：

1. **External Linker = 把 HiCache 的三层压成两层**。显存直连远端 KV 池，
   砍掉 host 中转层。远端不是"缓存层"，节点上只有一个 `external_cache_stored` 布尔标记，
   没有本地元数据、没有本地 LRU。
2. **难点全在"树形状必须跨 rank 一致"**。因此有三处 `all_reduce(MIN)`：
   可恢复长度集合取交集、offload 完成计数取最小、offload 成功 flag 取 AND。
   `lookup` 返回**集合**而不是最大值，是 SWA 这类 `TRAILING_PAGES` 池导致可恢复长度稀疏的必然结果。
3. **加载是三阶段事务**。`PREPARE`（分配 device slot）→ `insert`（发布到树）→
   `COMMIT`（按 `adopted_ranges` 裁剪后才发起 RDMA），任一 component 失败则 `ABORT` 逆序回滚。
   `adopted_ranges` 裁剪是必须的：insert 可能丢弃调用方给的 slot，
   不裁剪就会往已释放的显存里 RDMA。

**当前状态**：契约、池装配、树侧核心、后端实现均已落地，并已**接线到 CLI**（`--enable-unified-cache-external-linker`，后端 `mooncake` / `mori`，与 `--enable-hierarchical-cache` 互斥）。策略覆盖仍以 DSA / DSV4 为主，Mamba 明确不支持（`mamba.py:655` 仍 `raise`）。

---

## 16. 交叉引用

| 想了解 | 看这篇 |
|--------|--------|
| `UnifiedRadixCache` / `UnifiedTreeCore` 的整体结构、component 模型 | [unified_radix_cache_architecture.md](unified_radix_cache_architecture.md) |
| HiCache 三层的完整设计与使用方式（本文的对照基准） | [hicache_usage_and_design.md](hicache_usage_and_design.md) |
| Mooncake 作为 L3 storage 后端的常规接入方式 | [mooncake_integration_architecture.md](mooncake_integration_architecture.md)、[mooncake_standalone_storage_integration.md](mooncake_standalone_storage_integration.md) |
| SWA 的窗口语义、为什么它的可恢复前缀是稀疏的 | [swa_radix_cache_architecture.md](swa_radix_cache_architecture.md) |
| DSV4 的多物理池布局（本文 §8 展开的那 6 个池） | [deepseek_v4_cache_management.md](../09_models/deepseek_v4_cache_management.md) |
| PD 分离下 D 侧的 radix + HiCache 方案 | [pd_decode_radix_cache_hicache_scheme.md](../04_pd_disaggregation/pd_decode_radix_cache_hicache_scheme.md) |
| host 层的 off-tree staging（linker 模式恰好绕开的那一层） | [host_cache_off_tree_staging_scheme.md](host_cache_off_tree_staging_scheme.md) |
| 同类的"无握手校验导致静默数据错乱"问题 | [spec_decoding_cache_centralized_vs_pd.md](../05_speculative_decoding/spec_decoding_cache_centralized_vs_pd.md) |

### 关键源码文件速查

| 文件 | 作用 |
|------|------|
| `mem_cache/unified_cache/unified_cache_linker.py` | `UnifiedCacheLinker` 契约 + `UnifiedCacheLinkerWrapper` 树侧编排（核心） |
| `mem_cache/hybrid_cache/linker_pool_assembler.py` | `DevicePoolEntry` / `DevicePoolGroup`，逻辑池 → 物理池展开 |
| `mem_cache/hybrid_cache/hybrid_pool_assembler.py` | 各策略的 `build_direct_linker_pool_group` |
| `mem_cache/storage/mooncake_store/mooncake_direct_linker.py` | Mooncake 后端实现（`mooncake`） |
| `mem_cache/storage/umbp/umbp_direct_linker.py` | UMBP 后端实现（`mori`） |
| `arg_groups/fields/memory.py` | `--enable-unified-cache-external-linker` / `--unified-cache-external-linker-backend` 定义 |
| `mem_cache/registry.py` | 生产实例化与 `init_cache_linker` 接线（`:199-224`） |
| `mem_cache/unified_radix_cache.py` | 8 处 linker hook |
| `mem_cache/unified_cache/unified_tree_core.py` | `external_cache_stored` / 写通链 / split 重定向 |
| `mem_cache/unified_cache/components/*.py` | 各 component 的 `build_external_linker_transfer` / `update_external_linker_load` |
| `mem_cache/hicache_storage.py` | `PoolTransfer` / `PoolName` / `PoolHitPolicy` / `PoolTransferResult` |

