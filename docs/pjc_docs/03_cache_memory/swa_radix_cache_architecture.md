# SGLang SWARadixCache（滑窗混合 KV 前缀缓存）系统梳理

> 核心文件：`python/sglang/srt/mem_cache/swa_radix_cache.py`（1349 行）
> 配套文件：`python/sglang/srt/mem_cache/swa_memory_pool.py`（分配器 + KV 池）、`python/sglang/srt/mem_cache/base_swa_memory_pool.py`（池 ABC）、`python/sglang/srt/mem_cache/base_prefix_cache.py`（基类 + 统一参数）
> 调度交互：`python/sglang/srt/managers/schedule_policy.py`、`python/sglang/srt/managers/schedule_batch.py`、`python/sglang/srt/mem_cache/common.py`
> 目标读者：需要理解、调试或扩展 SGLang 滑动窗口注意力（SWA）KV 缓存与前缀复用机制的工程师。
>
> 本文由浅入深，覆盖：为什么需要 SWARadixCache、混合 SWA 模型的内存特点、双池+映射表设计、TreeNode 的双锁与 tombstone（墓碑）机制、双 LRU 链表、滑窗感知的前缀匹配、tombstone 复活式插入、双预算驱逐、SWA 锁边界（swa_uuid）与早释放优化、与调度器的预算交互、与标准 RadixCache 的对比、实例化触发条件，以及调优开关与排错。

---

## 目录

1. [总览：SWARadixCache 解决什么问题](#1-总览)
2. [背景：混合 SWA 模型的内存非对称性](#2-背景混合-swa-模型的内存非对称性)
3. [双池 + 映射表：物理内存布局](#3-双池--映射表物理内存布局)
4. [核心数据结构：TreeNode 的双锁与 tombstone](#4-核心数据结构treenode)
5. [双 LRU 链表：full_lru_list 与 swa_lru_list](#5-双-lru-链表)
6. [整体架构与不变量](#6-整体架构与不变量)
7. [Tombstone（墓碑）机制——本模块的灵魂](#7-tombstone-机制)
8. [滑窗感知的前缀匹配 match_prefix](#8-滑窗感知的前缀匹配)
9. [插入 insert：tombstone 复活的三分支](#9-插入-inserttombstone-复活)
10. [锁机制：inc_lock_ref / dec_lock_ref / dec_swa_lock_only](#10-锁机制)
11. [双预算驱逐 evict](#11-双预算驱逐-evict)
12. [请求生命周期：cache_unfinished_req / cache_finished_req](#12-请求生命周期)
13. [与调度器的交互：双预算与主动驱逐](#13-与调度器的交互)
14. [与标准 RadixCache 的对比](#14-与标准-radixcache-的对比)
15. [实例化与触发条件](#15-实例化与触发条件)
16. [环境变量与优化开关](#16-环境变量与优化开关)
17. [关键文件 / 行号索引](#17-关键文件行号索引)

---

## 1. 总览

`SWARadixCache` 是 SGLang 为 **混合滑动窗口注意力（hybrid Sliding Window Attention）模型** 设计的前缀缓存（prefix cache / radix tree）。这类模型（Llama4、Gemma4、GPT-OSS、DeepSeek-V4、Step3.5、MiMo-V2 等）的层分为两类：

- **全注意力层（full attention）**：每个 token 都要保留整条序列的 KV。
- **滑窗层（SWA）**：每个 token 只需要最近 `sliding_window_size` 个 token 的 KV。

标准 `RadixCache` 把所有层的 KV 当成一个整体管理——一个节点要么整段缓存、要么整段驱逐。但混合 SWA 模型里，**全注意力层的 KV 远比滑窗层的 KV 珍贵**（前者随序列长度线性增长且永不过期，后者只需固定窗口）。如果用单一池、单一驱逐策略，会出现两种浪费：

- 用全注意力的尺度去配滑窗池 → 滑窗池巨大而绝大部分永远用不到；
- 用滑窗的尺度去配全注意力池 → 全注意力池频繁 OOM。

`SWARadixCache` 的核心思想：**让全注意力 KV 和滑窗 KV 在同一棵 radix 树上、但拥有各自独立的内存池、各自独立的引用计数、各自独立的 LRU 驱逐顺序**。一个树节点可以处于「全注意力 KV 还在、滑窗 KV 已被释放」的中间状态——这就是 **tombstone（墓碑）节点**。

一句话定位：**SWARadixCache = 单棵 radix 树 + 双内存池（full / swa）+ 双引用计数（full_lock_ref / swa_lock_ref）+ 双 LRU 链表 + tombstone 机制，使滑窗 KV 能在保留全注意力 KV 的前提下被独立、提前回收，同时仍保证前缀匹配的正确性。**

---

## 2. 背景：混合 SWA 模型的内存非对称性

考虑一个序列长度 `L`、滑窗大小 `W`（通常 `W ≪ L`，例如 Llama4 的 `W=8192`，但请求可达数十万 token）的请求：

| 层类型 | 单 token KV 需求 | 序列总 KV | 过期策略 |
|---|---|---|---|
| 全注意力层 | 必须永久保留 | `O(L)` | 永不过期（除非整条序列结束） |
| 滑窗层 | 只需最近 `W` 个 | `O(min(L, W))` | 滑出窗口即可释放 |

SGLang 据此把 KV 物理上拆成两个池：

- **full 池**：容量 `size_full`，给全注意力层用，尺寸大。
- **swa 池**：容量 `size_swa`，给滑窗层用，尺寸小很多。

> ⚠️ 命名陷阱（记入项目记忆）：`SWARadixCache.sliding_window_size` 来自 `CacheInitParams.sliding_window_size`，由 `model_config._get_sliding_window_size()` 读取 HF 配置。它是「**逻辑窗口大小**」，单位是 token，用于决定锁多少 SWA token、何时停止前缀匹配。它与 KV 池内部的 `swa_page_size` / 物理页布局是两个层面的概念，不要混淆。

**关键结论**：由于两个池尺寸不同，缓存系统必须能够**独立计量、独立驱逐**两类 KV。`available_size()` 在 SWA 分配器里被定义为 `min(full, swa)`（`swa_memory_pool.py:388`），但前缀缓存内部必须分别看到 `full_available_size()` 与 `swa_available_size()`，才能精确决定「驱逐多少 full、驱逐多少 swa」。这是后文双预算驱逐的根源。

---

## 3. 双池 + 映射表：物理内存布局

SWARadixCache 强制要求其 allocator 是 `SWATokenToKVPoolAllocator`（`swa_radix_cache.py:345` 的 `assert`）。理解这个分配器是理解整个缓存的前提。

### 3.1 两个子分配器 + 一张翻译表

`SWATokenToKVPoolAllocator.__init__`（`swa_memory_pool.py:305`）内部并不是一个池，而是**两个独立的子分配器**：

```
SWATokenToKVPoolAllocator
├── full_attn_allocator   (size = size_full,  给全注意力层)   ← 规范句柄（canonical handle）
├── swa_attn_allocator    (size = size_swa,   给滑窗层)
└── full_to_swa_index_mapping  : int64[size + page_size + 1]
        full token 槽位  ──映射──▶  swa token 槽位
        末元素固定为 -1，使 -1 → -1（last_loc 哨兵）
```

- `page_size == 1` → 两个 `TokenToKVPoolAllocator`（`swa_memory_pool.py:329/336`）
- `page_size > 1` → 两个 `PagedTokenToKVPoolAllocator`（NPU 上为 `NPUPagedTokenToKVPoolAllocator`）（`:348/356`）

**对外返回的索引永远是 full 池的索引**，这是「规范句柄」。需要访问 swa 池时，通过 `full_to_swa_index_mapping[full_idx]` 翻译成 swa 索引。

### 3.2 关键方法

| 方法 | 行号 | 行为 |
|---|---|---|
| `available_size()` | 388 | **`min(full, swa)`**——容量由更紧张的池决定（通常是 swa 池或 full 池，视负载而定） |
| `full_available_size()` / `swa_available_size()` | 394 / 397 | 分别返回两个子池可用量，**供 SWARadixCache 的双预算计量使用** |
| `alloc(need_size)` | 427 | （page_size==1）两池都检查、都分配，再写 `mapping[full]=swa`，只返回 full 索引 |
| `alloc_extend(...)` | 448 | （page_size>1）算新页数，两池都验证、都扩展，记录映射 |
| `alloc_extend_swa_tail(...)` | 500 | PD 分离 decode 专用（bs=1）：full 分配整条，swa 只分配末尾 `swa_tail_len` 个，窗口外的 full 槽位映射置 0 |
| `free(free_index)` | 597 | 先释放 full 池，再调 `free_swa`（受 free-group 批处理标志约束） |
| `free_swa(free_index)` | 630 | 翻译 full→swa，只释放 `swa_indices[swa_indices > 0]`，并把映射项重置为 0。**这是 tombstone 的物理基础——只释放滑窗 KV，保留全注意力 KV** |
| `backup_state` / `restore_state` | 637 / — | 返回/恢复 2 元素列表（两个子分配器各一份） |

### 3.3 与标准分配器的本质区别

```
标准 TokenToKVPoolAllocator          SWATokenToKVPoolAllocator
┌──────────────────────┐            ┌────────────┐  ┌────────────┐
│  单一连续索引空间      │            │  full 池    │  │  swa 池     │
│  available_size = 1 个 │            │ (大)        │  │ (小)        │
│  free() 全释放         │            └─────┬──────┘  └─────┬──────┘
└──────────────────────┘                  │  full_to_swa   │
                                           └─── mapping ────┘
                                     free()   → full + swa 都释放
                                     free_swa() → 仅 swa 释放（保留 full）
```

`free_swa()` 是标准分配器没有的能力，它让「释放滑窗状态但保留全注意力状态」成为可能，是 tombstone 机制能落地的硬件级支撑。

---

## 4. 核心数据结构：TreeNode

`TreeNode`（`swa_radix_cache.py:57`）是 radix 树的节点。相比标准 `RadixCache.TreeNode`，它的关键差异在于**两套引用计数**和**tombstone 标志**。

```python
class TreeNode:
    counter = 0                       # 全局自增 id 计数器
    swa_uuid_counter = 1              # swa 锁边界 uuid 计数器
    last_access_time_counter_float    # 单调递增的逻辑时钟（仅供 sanity check）

    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent
        self.key: RadixKey            # 本节点覆盖的 token 序列（页对齐）
        self.value: torch.Tensor      # 对应的 full 池索引（规范句柄）

        self.swa_tombstone = False    # ★ 是否墓碑：swa KV 已释放、full KV 仍在

        # ★ 双引用计数。不变量：full_lock_ref ≥ swa_lock_ref
        self.full_lock_ref = 0
        self.swa_lock_ref = 0

        self.last_access_time             # 仅 sanity check 用，真正 LRU 靠链表
        self.hit_count
        self.host_value                   # HiCache host 索引
        self.hash_value                   # 每页 hash（HiCache / kv-events）

        # ★ 双 LRU 链表的双向指针
        self.prev = None;     self.next = None       # full_lru_list 用
        self.swa_prev = None; self.swa_next = None    # swa_lru_list 用

        self.id = TreeNode.counter; TreeNode.counter += 1
        self.swa_uuid = None              # 标记此节点是某次 inc_lock_ref 的 swa 锁边界
```

### 4.1 双锁不变量（贯穿全模块）

> **不变量（`swa_radix_cache.py:70-72` 注释）**：
> 对任意节点，若 `swa_lock_ref > 0`，则 `full_lock_ref` 必然 `> 0`；
> 反之 `full_lock_ref > 0` 不要求 `swa_lock_ref > 0`。
> 因此恒有 **`full_lock_ref ≥ swa_lock_ref`**。

直观理解：滑窗 KV 是全注意力 KV 的「子集生命周期」。一段被滑窗锁住的 KV，其全注意力部分一定也被锁住；但全注意力可以单独被锁（例如序列很长，靠近 root 的 KV 早已滑出窗口，但全注意力层仍然需要）。

### 4.2 三种节点状态

| 状态 | `value` | `swa_tombstone` | 含义 | 在 full_lru | 在 swa_lru |
|---|---|---|---|---|---|
| **正常节点** | 非 None | False | full + swa KV 都在 | ✓ | ✓ |
| **tombstone 节点** | 非 None | True | full KV 在，swa KV 已释放 | ✓ | ✗（已移除） |
| **evicted 节点** | None | — | 已从树删除 | ✗ | ✗ |

`evicted` 属性（`:97`）= `value is None`；`backuped` 属性（`:101`）= `host_value is not None`。

### 4.3 swa_uuid：滑窗锁边界标记

`inc_lock_ref` 从某个节点向 root 上锁时，**full 锁一直锁到 root**，但 **swa 锁只锁到累计够一个 `sliding_window_size` 为止**。那个「刚好锁满窗口」的节点会被赋予一个 `swa_uuid`（由 `gen_swa_uuid()` 自增产生，`:108`），并返回给调用方。后续 `dec_lock_ref` 凭这个 uuid 知道 swa 锁该解到哪个节点为止。这就解耦了「full 锁范围」和「swa 锁范围」。

---

## 5. 双 LRU 链表

标准 RadixCache 用一个 `evictable_leaves` 集合 + 临时堆来选驱逐对象。SWARadixCache 改用**两条侵入式双向链表** `LRUList`（`swa_radix_cache.py:119`），分别维护 full 和 swa 两个维度的 LRU 顺序。

### 5.1 LRUList 设计

```python
class LRUList:
    def __init__(self, is_swa_list: bool = False):
        # 用反射切换操作的是 prev/next 还是 swa_prev/swa_next，
        # 以及检查的是 full_lock_ref 还是 swa_lock_ref
        self.prv / self.nxt / self.lock_ref = ("swa_prev","swa_next","swa_lock_ref")
                                               if is_swa_list else
                                               ("prev","next","full_lock_ref")
        self.head = TreeNode()   # MRU 哨兵
        self.tail = TreeNode()   # LRU 哨兵
        self.cache = {}          # id -> node，O(1) 判定在表内
```

链表不变量：`head` 侧最近使用（MRU），`tail` 侧最久未用（LRU）；`prev` 指向更近使用者，`next` 指向更久未用者。

为什么用**侵入式双向链表**而不是堆？因为前缀匹配 / 插入会频繁地把「一整条从叶到根的路径」整体移到 MRU（`reset_node_and_parents_mru`，`:183`），链表可以 O(路径长) 地原地移动，而堆每次都要 O(n log n) 重建（标准 RadixCache 的 `evict` 正是每次重建堆）。

### 5.2 关键方法

| 方法 | 行号 | 用途 |
|---|---|---|
| `insert_mru(node)` | 200 | 新节点插到 MRU 端；断言不可把 tombstone 插入 swa 链表 |
| `reset_node_mru(node)` | 172 | 已有节点移到 MRU |
| `reset_node_and_parents_mru(node, root)` | 183 | 把从 node 到 root 的整条路径移到 MRU，且子节点比父节点更「新」——**让靠近 root 的节点优先被驱逐** |
| `remove_node(node)` | 213 | 从链表移除（驱逐 / tombstone / split 时） |
| `get_lru_no_lock()` | 224 | 取最久未用且未锁的节点（swa 驱逐用，可以是内部节点） |
| `get_leaf_lru_no_lock()` | 230 | 取最久未用且未锁的**叶**节点（full 驱逐用，必须是叶） |
| `get_prev_no_lock(node)` | 236 | 沿链表找前一个（更新的）未锁节点，跳过被锁的 |
| `get_prev_leaf_no_lock(node)` | 254 | 同上但要求是叶（`len(children)==0`） |
| `sanity_check(tree_cache)` | 291 | 重建链表并与树对比，校验 LRU 一致性与 evictable 计量（昂贵，仅 debug / idle） |

### 5.3 为什么 full 驱逐找叶、swa 驱逐可找内部节点

这是本模块最微妙的设计之一：

- **full 驱逐**：删除节点意味着从树上摘除，radix 树要求**只能删叶子**（否则会切断子节点到 root 的路径），所以 full 驱逐用 `get_leaf_lru_no_lock`。
- **swa 驱逐**：释放滑窗 KV 不需要删节点，只要把它变成 tombstone（`free_swa` + 标记），节点仍留在树上当「全注意力路标」。所以 swa 驱逐可以挑**内部节点**（`get_lru_no_lock`），把它就地 tombstone 化。

这正是 tombstone 机制让「滑窗维度」摆脱「radix 树只能删叶」约束的精髓。

---

## 6. 整体架构与不变量

把前面的部件组装起来：

```
┌─────────────────────────────── SWARadixCache ────────────────────────────────┐
│                                                                               │
│  root_node (full_lock_ref=1, swa_lock_ref=1, 永不驱逐)                         │
│    │                                                                          │
│    ├── 节点A [正常]   value=full_idx   ──────────┐                            │
│    │     │            full_lock_ref / swa_lock_ref│                            │
│    │     ├── 节点B [tombstone]  value=full_idx    │  靠近 root 的节点          │
│    │     │     swa 已 free，仅 full 在            │  先被 swa 驱逐 → tombstone │
│    │     │     │                                  │                            │
│    │     │     └── 节点C [正常 / 叶]              ▼                            │
│    │     │                                                                    │
│    └── 节点D ...                                                              │
│                                                                               │
│  计量（4 个计数器）                          驱逐顺序（2 条链表）              │
│  ┌─────────────────────────┐               ┌──────────────────────────┐      │
│  │ full_evictable_size_     │               │ full_lru_list (含所有节点) │      │
│  │ full_protected_size_     │               │ swa_lru_list (不含tombstone)│      │
│  │ swa_evictable_size_      │               └──────────────────────────┘      │
│  │ swa_protected_size_      │                                                 │
│  └─────────────────────────┘                                                 │
│           │                                                                   │
│           ▼                                                                   │
│  token_to_kv_pool_allocator: SWATokenToKVPoolAllocator                        │
│      ├── full 池   (free / alloc)                                             │
│      └── swa 池    (free_swa / alloc)                                         │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 6.1 四个尺寸计数器

| 计数器 | 含义 | 何时增减 |
|---|---|---|
| `full_evictable_size_` | 可被 full 驱逐的 token 数（在树、未锁） | 加新节点 +，锁住 -，驱逐 - |
| `full_protected_size_` | 被 full 锁保护的 token 数 | `inc_lock_ref` 0→1 时从 evictable 转入 |
| `swa_evictable_size_` | 可被 swa 驱逐的 token 数（非 tombstone、未锁） | tombstone 化 / 锁住时 - |
| `swa_protected_size_` | 被 swa 锁保护的 token 数 | `inc_lock_ref` swa 段 0→1 时转入 |

`evictable_size()` 与 `protected_size()` 在 SWARadixCache 里**直接 `raise NotImplementedError`**（`:819` / `:829`），强制调用方使用 `full_*` / `swa_*` 分量版本——这是为了避免「把两个维度混成一个数」的语义错误。`total_size()` 返回 `Tuple[int,int]`（`:560`）。

### 6.2 核心不变量清单

1. **双锁不变量**：`full_lock_ref ≥ swa_lock_ref`（§4.1）。
2. **tombstone 不在 swa 链表**：tombstone 节点从 `swa_lru_list` 移除且 `swa_evictable_size_` 已扣减；`reset_node_mru` / `insert_mru` 对 swa 链表都断言 `not swa_tombstone`。
3. **叶节点不是 tombstone**：驱逐后用 `_iteratively_delete_tombstone_leaf`（`:1242`）保证。否则前缀匹配会卡在 tombstone 叶上。
4. **tombstone 的 swa_lock_ref 恒为 0**：tombstone 意味着 swa 已释放，不可能还被 swa 锁住。多处 assert 守护。
5. **root 永不驱逐**：`full_lock_ref = swa_lock_ref = 1`（`reset()` 中，`:378`），且不入任何 LRU 链表。

---

## 7. Tombstone 机制

这是 SWARadixCache 区别于一切其他 radix 缓存的核心。**务必先理解它，再看后面所有方法。**

### 7.1 什么是 tombstone

一个 **tombstone（墓碑）节点** 是这样的节点：
- 它的**滑窗 KV 已经被 `free_swa` 释放**（swa 池槽位还给了 swa 分配器）；
- 但它的**全注意力 KV（`value` 指向的 full 池槽位）仍然存活**；
- 它**仍留在 radix 树上**，作为全注意力前缀的「路标」，让更深的子节点仍能通过它连到 root。

### 7.2 为什么需要 tombstone

设想一条很长的对话：序列长度 `L = 100k`，滑窗 `W = 8k`。

- 全注意力层：100k token 的 KV 全要留着（后续 token 还要 attend 到它们）。
- 滑窗层：只有最后 8k 有用，前面 92k 的滑窗 KV 是死的。

如果没有 tombstone，这 92k 的节点要么整体保留（浪费 92k 的 swa 池），要么整体驱逐（连带删掉 full KV，破坏全注意力前缀，下次复用就 miss）。

tombstone 让这 92k 节点进入「滑窗 KV 释放、全注意力 KV 保留」的中间态：**swa 池立刻回收 92k，full 池继续命中前缀**。这对长上下文 / 多轮对话的缓存复用率是决定性的。

### 7.3 tombstone 的产生路径

| 路径 | 触发点 | 代码 |
|---|---|---|
| **swa 驱逐内部节点** | swa 池不够，`evict` 挑中一个有孩子的内部节点 | `evict` → `_tombstone_internal_node`（`:1277`） |
| **swa 锁早释放（叶）** | decode 越过窗口，主动释放叶的 swa 锁 | `dec_swa_lock_only`（`:760`）对叶 free_swa + 标记 |
| **插入时显式建 tombstone 段** | 插入的序列中部分滑窗 KV 已被驱逐 | `_insert_helper` case 2（`:1202`） |

### 7.4 tombstone 的消亡路径

| 路径 | 触发点 | 代码 |
|---|---|---|
| **复活（un-tombstone）** | 新请求带着未被驱逐的滑窗 KV 重新插入这段前缀 | `_insert_helper` 分支 1/2（`:1139`/`:1148`） |
| **作为父节点被迭代删除** | 其唯一的子叶被 full 驱逐后，它变成 tombstone 叶 | `_iteratively_delete_tombstone_leaf`（`:1242`） |
| **full 驱逐命中** | full 池不够，`evict` 直接删掉这个 tombstone 叶 | `evict` 第一段（`:586`） |

> 关键：**叶节点永远不能是 tombstone**（不变量 3）。一旦某个 tombstone 节点的最后一个孩子被删，它就违反了这个不变量，必须被 `_iteratively_delete_tombstone_leaf` 立即清理（连带释放其 full KV）。这保证前缀匹配不会停在一个「滑窗已死」的叶上。

---

## 8. 滑窗感知的前缀匹配

`match_prefix`（`:389`）找最长可复用前缀。它的核心区别在于：**匹配到的前缀不能「踩在」滑窗已被驱逐的洞上**。

### 8.1 滑窗匹配的正确性约束

考虑前缀路径上有 tombstone 节点（滑窗 KV 已驱逐）。如果新请求的前缀正好落在 tombstone 区域内、且距离序列末尾不足 `sliding_window_size`，那么滑窗层就缺了它实际需要的 KV——直接复用会算错。

`_match_prefix_helper`（`:866`）的保证（见其 docstring）：返回的匹配节点要么
1. **从 root 连到它的路径上没有 tombstone**（`match_len_since_tombstone = inf`），或
2. **从最后一个 tombstone 到匹配节点的 token 数 ≥ `sliding_window_size`**——即末尾这段窗口内的滑窗 KV 都是活的。

### 8.2 算法逻辑

```python
match_len_since_tombstone = inf      # 路径无 tombstone 时视为无穷大（总是可匹配）
best_value_len = 0; best_last_node = root
while 还能沿 child_key 往下匹配:
    child = node.children[child_key]
    if child.swa_tombstone:
        # 命中 tombstone 前，先看「上一段非 tombstone 累计」是否够一个窗口
        if match_len_since_tombstone >= sliding_window_size:
            best_value_len = len(value); best_last_node = node   # 记录安全点
        match_len_since_tombstone = 0                            # 重新累计
    prefix_len = child.key.match(key)
    if prefix_len < len(child.key):       # 部分匹配 → 分裂
        new_node = _split_node(...); value.append(new_node.value)
        if not new_node.swa_tombstone: match_len_since_tombstone += len(new_node.value)
        node = new_node; break
    else:                                  # 整段匹配 → 继续下沉
        value.append(child.value)
        if not child.swa_tombstone: match_len_since_tombstone += len(child.value)
        node = child; key = key[prefix_len:]
# 收尾：若末尾累计够窗口，最后一个节点也算安全匹配点
if match_len_since_tombstone >= sliding_window_size:
    best_value_len = len(value); best_last_node = node
return value, best_last_node, best_value_len
```

返回的 `best_value_len` 可能**小于**实际匹配到的总长度——多出来的部分因为滑窗不安全而被砍掉（`_match_post_processor` 里 `value = value[:best_value_len]`，`:957`）。

### 8.3 match 的副作用：LRU 更新

`_match_post_processor`（`:934`）把匹配到的节点及其所有祖先在两条 LRU 链表里都移到 MRU（`reset_node_and_parents_mru`），且**子节点比父节点更新**——这样 swa 驱逐会优先驱逐靠近 root 的旧节点。返回 `MatchResult`，其中 `last_device_node == last_host_node == best_match_node == last_node`（SWA 缓存没有独立的 host 层节点概念）。

---

## 9. 插入 insert：tombstone 复活

`insert`（`:417`）/ `_insert_helper`（`:1092`）负责把请求算出的 KV 写进树。SWA 版本最复杂的地方是处理 **`swa_evicted_seqlen`**——即「这条序列前多少个 token 的滑窗 KV 已经被主动驱逐了」。

### 9.1 两个关键入参

| 参数 | 来源 | 含义 |
|---|---|---|
| `prev_prefix_len`（= `update_kv_after_len`） | `req.cache_protected_len` | 已在树中、无需更新的前缀长度。只有超过它的部分才需要写 value |
| `swa_evicted_seqlen` | `req.swa_evicted_seqlen` | 序列中 `[?, swa_evicted_seqlen)` 区间的滑窗 KV 已被 `maybe_evict_swa` 主动释放 |

### 9.2 tombstone 复活：while 循环内的三分支（`:1129-1167`）

当插入路径命中一个 **tombstone 节点**，且新插入的 value 覆盖了它（`update_kv_after_len < total_prefix_length + prefix_len`），需要判断这段滑窗 KV 现在是否「活」了，决定能否复活：

```
布局：  |───── total_prefix_length ─────|── prefix_len ──|
                  ↑ swa_evicted_seqlen 落在哪里决定分支
```

| 分支 | 条件 | 动作 |
|---|---|---|
| **分支 1（全活，复活整节点）** | `swa_evicted_seqlen ≤ total_prefix_length` | 整段滑窗未驱逐。释放树中旧 full KV → 用新 value 覆盖 → `swa_tombstone=False` → 重新入 swa_lru、`swa_evictable_size_ +=`。节点复活。 |
| **分支 2（部分活，分裂复活）** | `total_prefix_length < swa_evicted_seqlen < +prefix_len` | 前半仍 tombstone，后半可复活。`_split_node` 在 `swa_evicted_seqlen` 处切开，对后半节点复活，前半保持 tombstone。 |
| **分支 3（全死，丢弃）** | `swa_evicted_seqlen ≥ +prefix_len` | 整段滑窗都已驱逐，无法复活。直接 `free(value[:prefix_len])`，保留树中 tombstone 不动。 |
| 非 tombstone | — | `free(value[:prefix_len])`，节点已有有效 KV，丢弃重复的新 value |

### 9.3 循环尾：为剩余 key 建新节点（`:1176-1216`）

匹配不到的剩余 `key` 要建新节点。根据 `swa_evicted_seqlen` 的位置：

```
布局： |── total_prefix_length ──|────── len(key) ──────|
       0                  total_prefix_length      total_length
```

- **case 3**：`swa_evicted_seqlen == total_length`（全部剩余都被驱逐）→ `free(value)` 直接返回，**不建节点**（叶不能是 tombstone）。注释说明这是 `_evict_swa` 的 `-page_size` 边距修复后理论上不会发生的防御性分支。
- **case 2**：`total_prefix_length < swa_evicted_seqlen < total_length` → 先建一个 **tombstone 段**（`[total_prefix_length, swa_evicted_seqlen)`），再建非 tombstone 的剩余叶。
- **case 1**：`swa_evicted_seqlen ≤ total_prefix_length`（全活）→ 直接建非 tombstone 叶。

最后若开启 `SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT`，调 `_maybe_split_leaf_for_swa_lock`（§16）把过长的叶切到「一个窗口」大小，避免 `inc_lock_ref` 过度膨胀 `swa_protected_size_`。

### 9.4 _split_node 的双链表维护（`:1053`）

分裂节点时（部分匹配 / 复活 / 切叶），新父节点 `new_node` 要继承 child 的 `swa_tombstone` / `full_lock_ref` / `swa_lock_ref` / `swa_uuid`，并把两个节点都正确插回两条 LRU 链表（只有非 tombstone 才入 swa 链表，`:1087`）。`swa_uuid` 从 child 转移到 new_node（父继承锁边界，`:1064`）。

---

## 10. 锁机制

锁（lock_ref）保护正在被运行中请求使用的 KV 不被驱逐。SWA 版本要同时管理 full 锁和 swa 锁，且两者范围不同。

### 10.1 inc_lock_ref（`:670`）

```
        root                                full 锁：[node → root) 全锁
         ↑  full + swa                      swa 锁：[node → swa_uuid 节点] 只锁够一个窗口
         │  full + swa
   swa_uuid 节点 ← swa 锁到此为止（返回 swa_uuid_for_lock）
         │  full only（已超出窗口，swa 不锁）
         │  full only
        node（请求的 last_node）
```

逻辑：从 `node` 向 root 走，每个节点：
- **full 锁**：无条件 `full_lock_ref += 1`；首次 0→1 时把 `len(value)` 从 `full_evictable_size_` 转入 `full_protected_size_`。
- **swa 锁**：只要累计锁住的 swa token 还没到 `sliding_window_size`，就 `swa_lock_ref += 1` 并累加 `swa_lock_size`；一旦累计 `≥ sliding_window_size`，给当前节点生成/取出 `swa_uuid`，记为 `swa_uuid_for_lock` 返回，之后的祖先不再加 swa 锁。

返回 `IncLockRefResult(swa_uuid_for_lock=...)`。调用方必须保存这个 uuid，传给对应的 `dec_lock_ref`。

### 10.2 dec_lock_ref（`:711`）

逆操作。`full_lock_ref` 解锁到 root（exclusive）；`swa_lock_ref` 解锁到 `swa_uuid_for_lock` 标记的节点为止（inclusive）。当走到 `node.swa_uuid == swa_uuid_for_lock` 时，把 `dec_lock_swa` 置 False，停止后续 swa 解锁。

`skip_swa=True` 时**只解 full 锁**——用于 swa 锁已经被 `dec_swa_lock_only` 提前释放过的情况（见下）。

### 10.3 dec_swa_lock_only（`:760`）—— 早释放优化

这是一个性能优化（受 `SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW` 控制）。当一条 decode 请求的位置已经**越过滑窗**，它 prefill 时锁住的那段 swa KV 就再也用不到了，可以提前从 protected 转回 evictable，让 swa 池在压力下能回收它。

```
请求 decode 位置前进 → decode_batch_idx ≥ sliding_window_size
   → maybe_evict_swa 调 dec_swa_lock_only(req.last_node, req.swa_uuid_for_lock)
   → 沿 [node, swa_uuid] 只解 swa 锁，full 锁不动
```

对**内部节点**：标准 protected → evictable（节点留在 swa_lru，后续可被 swa 驱逐）。
对**叶节点**：因为 swa_lru 不允许「带 full_lock_ref 的叶」（swa 驱逐会连带删掉这个仍被引用的叶），所以直接 `free_swa` + 标 tombstone + 移出 swa_lru。full KV 留到 full 锁释放为止。

调用方须保证每个 `(node, swa_uuid)` 对最多调一次（用 `Req.swa_prefix_lock_released` 跟踪），且之后 `dec_lock_ref` 要传 `skip_swa=True`，避免重复操作 swa 状态。

### 10.4 三者协作时序

```
prefill 结束 cache_unfinished_req:
    inc_lock_ref(new_last_node) → 得到 swa_uuid_for_lock，存入 req
decode 进行中 maybe_evict_swa（每步）:
    if decode_batch_idx ≥ W and not released:
        dec_swa_lock_only(last_node, swa_uuid)   # 提前释放 swa 锁
        req.swa_prefix_lock_released = True
请求结束 cache_finished_req:
    dec_lock_ref(last_node, swa_uuid, skip_swa=req.swa_prefix_lock_released)
    # skip_swa=True 时只解 full 锁
```

---

## 11. 双预算驱逐 evict

`evict`（`:563`）是最长的方法。它接收**两个独立预算** `num_tokens`（full）和 `swa_num_tokens`（swa），分两段执行。

### 11.1 调用约定

调用方 `evict_from_tree_cache`（`common.py:330`）针对 SWA 分配器这样算预算：

```python
full_available = allocator.full_available_size()
swa_available  = allocator.swa_available_size()
if full_available < num_tokens or swa_available < num_tokens:
    full_num_tokens = max(0, num_tokens - full_available)   # full 还差多少
    swa_num_tokens  = max(0, num_tokens - swa_available)    # swa 还差多少
    tree_cache.evict(EvictParams(num_tokens=full_num_tokens,
                                 swa_num_tokens=swa_num_tokens))
```

即「两个池各自缺多少，就各自驱逐多少」。

### 11.2 第一段：full 驱逐（`:571-607`）

```python
if full_num_tokens > 0:
    x = full_lru_list.get_leaf_lru_no_lock()        # 取未锁的 LRU 叶
    while full_num_evicted < full_num_tokens and x in list:
        free(x.value)                               # 释放 full + swa
        full_num_evicted += len(x.value)
        if not x.swa_tombstone:                     # tombstone 的 swa 早释放过了
            swa_num_evicted += len(x.value)
        x_next = get_prev_leaf_no_lock(x)
        full_lru_list.remove_node(x)
        if not x.swa_tombstone: swa_lru_list.remove_node(x)
        _delete_leaf(x)
        # 父节点若变成 tombstone 叶，迭代删除以维持「叶非 tombstone」不变量
        x, leaf_evicted = _iteratively_delete_tombstone_leaf(x)
        full_num_evicted += leaf_evicted
        if len(x.parent.children) == 0:             # 父变叶，重新取 LRU 叶
            x_next = get_leaf_lru_no_lock()
        x = x_next
```

full 驱逐**只删叶**，且驱逐一个正常叶会**连带满足 swa 预算**（full+swa 都释放）。`_iteratively_delete_tombstone_leaf` 在删完一个叶后，若其父是「无子的 tombstone」，则迭代向上删除（释放这些 tombstone 的 full KV）。

### 11.3 第二段：swa 驱逐（`:609-663`）

只有 full 段没凑够 swa 预算时才进入。`x = swa_lru_list.get_lru_no_lock()`（**可以是内部节点**）：

| 节点类型 | 动作 |
|---|---|
| **内部节点**（有孩子） | `free_swa(x.value)` 只释放滑窗 → `_tombstone_internal_node`：标 tombstone、移出 swa_lru、`swa_evictable_size_ -=`。节点留在树上当全注意力路标。 |
| **带 full 锁的叶**（早释放优化复活的特例） | `free_swa` + 移出 swa_lru + 标 tombstone（不删节点，full 锁还在） |
| **普通叶**（无 full 锁） | full+swa 全删：`free(x.value)`、两链表移除、`_delete_leaf` + `_iteratively_delete_tombstone_leaf` |

swa 驱逐的精髓：**优先把内部节点 tombstone 化**（代价小、保留全注意力前缀），实在不行才删叶。

### 11.4 两段配合的整体效果

```
请求新内存 → evict_from_tree_cache 算出 (full_need, swa_need)
   │
   ├─ full 段：删 LRU 叶，full↓ swa↓（正常叶）/ full↓（tombstone 叶）
   │           直到 full_need 满足
   │
   └─ swa 段：若 swa 仍不够，把 LRU 内部节点 tombstone 化（swa↓，full 不变）
              直到 swa_need 满足
```

返回 `EvictResult(num_tokens_evicted=full_num_evicted, swa_num_tokens_evicted=swa_num_evicted)`。

---

## 12. 请求生命周期

### 12.1 cache_unfinished_req（`:487`）—— chunked prefill 中途

每个 chunk 算完后调用，把已算 KV 写入树并重新上锁：

```python
token_ids = req.fill_ids                              # 当前已填充的 token
kv_indices = req_to_token[req_pool_idx, :len]
radix_key = RadixKey(token_ids, extra_key).page_aligned(page_size)
old_prefix_len = req.cache_protected_len
# 1. 插入（内部会 free 掉重叠/重复的 kv_indices）
result = insert(InsertParams(key, value, prev_prefix_len=old_prefix_len))
# 2. 重新匹配拿到最新前缀（可能因别的请求插入而变长）
match = match_prefix(MatchPrefixParams(key=radix_key))
new_indices, new_last_node = match.device_indices, match.last_device_node
# 3. 把新前缀写回 req_to_token_pool
req_to_token_pool.write((req_pool_idx, slice(old_prefix_len, len(new_indices))),
                        new_indices[old_prefix_len:])
req.cache_protected_len = len(new_indices)
# 4. 换锁：解旧 last_node，锁新 last_node
dec_lock_ref(req.last_node, swa_uuid_for_lock=req.swa_uuid_for_lock,
             skip_swa=req.swa_prefix_lock_released)
req.swa_prefix_lock_released = False
result = inc_lock_ref(new_last_node)
req.swa_uuid_for_lock = result.swa_uuid_for_lock      # 保存新 swa 锁边界
# 5. 更新 req.prefix_indices / req.last_node（供 PrefillAdder 复用）
```

注意第 4 步换锁时重置 `swa_prefix_lock_released=False`——新一轮锁又是完整的 swa 锁，下次越窗才能再次早释放。

### 12.2 cache_finished_req（`:438`）—— 请求结束

```python
kv_committed_len = req.pop_committed_kv_cache()
token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
radix_key = RadixKey(token_ids, extra_key).page_aligned(page_size)
page_aligned_len = len(radix_key)
old_prefix_len = req.cache_protected_len
if is_insert:
    insert(InsertParams(key, value, prev_prefix_len=old_prefix_len,
                        swa_evicted_seqlen=req.swa_evicted_seqlen))  # ★带驱逐信息
else:
    free(kv_indices[old_prefix_len:page_aligned_len])
free(kv_indices[page_aligned_len:])                   # 释放非页对齐的尾巴
dec_lock_ref(req.last_node, swa_uuid_for_lock=req.swa_uuid_for_lock,
             skip_swa=req.swa_prefix_lock_released)    # 释放锁
```

`cache_finished_req` 把 `req.swa_evicted_seqlen` 传给 insert，让插入逻辑知道哪段滑窗 KV 已经死了（§9.2/9.3 的三分支据此工作）。

### 12.3 disable 模式

当 `self.disable=True`（关闭 radix 缓存），两个方法都退化为「直接 free，不进树」。

---

## 13. 与调度器的交互

SWARadixCache 不是孤立的；调度器必须感知双池预算，否则会过度调度导致 swa 池 OOM。

### 13.1 双预算预留（schedule_policy.py）

`PrefillAdder`（`schedule_policy.py`）在 `is_hybrid_swa` 时维护**两套 remaining token 预算**：

```python
@property
def rem_total_tokens(self):       # full 维度
    if self.is_hybrid_swa:
        return (allocator.full_available_size() + tree_cache.full_evictable_size()
                - self.rem_total_token_offset)

@property
def rem_swa_tokens(self):         # ★ swa 维度
    return (allocator.swa_available_size() + tree_cache.swa_evictable_size()
            - self.rem_swa_token_offset)
```

`budget_state`（`:563`）在 `is_hybrid_swa` 时额外检查 `rem_swa_tokens <= 0`；`add_one_req`（`:859`/`:879`）调 `_swa_budget_for_req` 估算单请求 swa 需求，超过 `rem_swa_tokens` 就拒绝。

### 13.2 _swa_budget_for_req（`:543`）

```python
def _swa_budget_for_req(self, extend_input_len):
    # chunked prefill + overlap 下 swa 峰值占用：
    #   chunk N（运行中，未入树）+ 窗口（树中锁住）+ chunk N+1（新分配）
    # 前两者已从 swa_available + swa_evictable 中排除，只需覆盖 chunk N+1
    alloc = min(extend_input_len, self.rem_chunk_tokens) if rem_chunk_tokens else extend_input_len
    return max(alloc, sliding_window_size) + page_size   # 下限保留一个窗口给 decode
```

### 13.3 主动 swa 驱逐 maybe_evict_swa（schedule_batch.py:2645）

光靠缓存被动驱逐不够——运行中请求自己产生的 swa KV（还没进树）也会滑出窗口变成垃圾。`ScheduleBatch.maybe_evict_swa` 每步主动回收：

- **decode**：每 `eviction_interval` 步（≈ `sliding_window_size * SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER`，页对齐）调一次 `_evict_swa(req, seqlen-1)`。
- **chunked prefill（chunk cache）**：每个 chunk 后按 `pre_len` 驱逐。

`_evict_swa`（`:2704`）计算新的 `swa_evicted_seqlen`：

```python
req.swa_evicted_seqlen = max(req.swa_evicted_seqlen, req.cache_protected_len)
# 减一个 page_size 边距：永不触及树插入边界 page_floor(seq_len)，
# 保证至少留一页非驱逐 swa KV 给树存为非 tombstone 节点（防 swa 内存泄漏）
evict_threshold = pre_len - sliding_window_size - page_size   # 默认
new_swa_evicted_seqlen = max(req.swa_evicted_seqlen, evict_threshold)
new_swa_evicted_seqlen = page_floor(new_swa_evicted_seqlen)   # 页对齐
if new_swa_evicted_seqlen > req.swa_evicted_seqlen:
    free_slots = req_to_token[req_pool_idx, req.swa_evicted_seqlen:new_swa_evicted_seqlen]
    allocator.free_swa(free_slots)                            # 直接 free_swa
    req.swa_evicted_seqlen = new_swa_evicted_seqlen
```

这个 `-page_size` 边距与 `_insert_helper` 的 case 3 防御分支（§9.3）相呼应：边距保证插入时末段总有非 tombstone 叶可建。

### 13.4 内存泄漏自检（invariant_checker.py:147）

`_get_total_uncached_sizes` 分别核算 full / swa 的「已分配未缓存」token：

```
full_uncached = allocated_len - cache_protected_len
swa_uncached  = allocated_len - max(cache_protected_len, swa_evicted_seqlen)   # ★
```

swa 维度要扣掉 `swa_evicted_seqlen`——已主动驱逐的部分不算「占用」。这个自检在空闲期对账，防止 swa 池泄漏。

---

## 14. 与标准 RadixCache 的对比

两者同在 `mem_cache/` 下，共享 `RadixKey` / `BasePrefixCache` / 统一参数 dataclass，但内部机制差异巨大。

| 维度 | RadixCache（`radix_cache.py`） | SWARadixCache（`swa_radix_cache.py`） |
|---|---|---|
| 适用模型 | 标准全注意力 LLM | 混合 SWA 模型（Llama4/Gemma4/GPT-OSS/DSV4…） |
| 内存池 | 单池 `TokenToKVPoolAllocator` | 双池 `SWATokenToKVPoolAllocator`（full+swa） |
| 每节点引用计数 | 单 `lock_ref` | 双 `full_lock_ref` + `swa_lock_ref`（恒 full≥swa） |
| 驱逐数据结构 | `evictable_leaves` 集合 + **临时堆**（每次重建 O(n log n)） | 双侵入式双向链表 `full_lru_list` / `swa_lru_list`（O(1) 移动） |
| 驱逐预算 | 单 `num_tokens` | 双 `num_tokens` + `swa_num_tokens`，两段执行 |
| 窗口处理 | 无 | tombstone：swa 释放、full 保留 |
| 提前释放 | 无 | `dec_swa_lock_only` + `skip_swa` 参数 |
| 尺寸计量 | `evictable_size_` / `protected_size_` | 四计数器；`evictable_size()`/`protected_size()` 抛 NotImplementedError |
| `total_size()` | `int` | `Tuple[int,int]`（full, swa） |
| `inc_lock_ref` 返回 | `delta` | `swa_uuid_for_lock`（swa 锁边界） |
| 驱逐对象 | 只删叶 | full 删叶；swa 可 tombstone 内部节点 |
| `evict` 中 split | 不会 | 复活 / 切叶时频繁 `_split_node` |
| insert 额外参数 | — | `prev_prefix_len`、`swa_evicted_seqlen` |

### 标准 RadixCache.evict 对比（`radix_cache.py:561`）

```python
def evict(self, params):                          # 单预算
    leaves = list(self.evictable_leaves)
    heap = [(strategy.get_priority(n), n) for n in leaves]
    heapq.heapify(heap)                           # ★每次重建堆
    while num_evicted < params.num_tokens and heap:
        _, x = heapq.heappop(heap)
        free(x.value); _delete_leaf(x)
        if x.parent 无子且未锁: heappush(parent)
```

可见标准版用「可插拔驱逐策略 + 堆」（支持 lru/lfu/fifo/priority 等），而 SWA 版用「双链表 + 双段固定流程」（专为 full/swa 双维度优化，不支持可插拔策略）。这是为 SWA 双维度正确性和性能做的专门取舍。

---

## 15. 实例化与触发条件

### 15.1 工厂链（registry.py）

`default_radix_cache_factory`（`registry.py:76`）的选择顺序（**前面的条件优先**）：

```
1. chunked prefill + disable_radix_cache    → ChunkCache / SWAChunkCache
2. SGLANG_EXPERIMENTAL_CPP_RADIX_TREE        → RadixCacheCpp
3. SGLANG_ENABLE_UNIFIED_RADIX_TREE          → UnifiedRadixCache(+SWA component)  ← 另一条 SWA 路径
4. enable_hierarchical_cache                 → HiMambaRadixCache / HiRadixCache
5. ★ is_hybrid_swa                           → SWARadixCache(params)             ← 本模块主路径
6. is_hybrid_ssm / enable_lmcache / else     → MambaRadixCache / LMCRadixCache / RadixCache
```

> ⚠️ 重要限定：SWARadixCache 是 SWA 模型的**默认**前缀缓存，但仅当未启用 unified-tree、hierarchical-cache、cpp-tree 等 env 开关、且未 disable radix cache 时成立。`SGLANG_ENABLE_UNIFIED_RADIX_TREE` 会改走 `UnifiedRadixCache` + `ComponentType.SWA` 这条等价但不同的实现路径。

### 15.2 触发标志的来源

```
model_config.is_hybrid_swa_model(architectures)   # configs/model_config.py:1657
    架构白名单：Llama4 / DeepseekV4(+NextN) / GptOss / MiMoV2(+MTP)
              / Step3p5(+MTP) / Gemma4 / Laguna
        ↓
get_hybrid_layer_ids(...)  # :1675 推导 swa/full 层 id（如 Llama4 每第4层为 full）
        ↓
tp_worker.is_hybrid_swa / sliding_window_size
        ↓
kv_cache_builder: 构造 SWAKVPool + SWATokenToKVPoolAllocator
                  CacheInitParams(sliding_window_size=...)
                  create_tree_cache(TreeCacheBuildContext(is_hybrid_swa=True, ...))
        ↓
SWARadixCache.__init__  # assert allocator 是 SWATokenToKVPoolAllocator
        ↓
Scheduler.tree_cache = result.tree_cache  # managers/scheduler.py
```

`SWARadixCache.__init__`（`:344`）的 `assert isinstance(..., SWATokenToKVPoolAllocator)` 强制要求池和缓存配套——这两者必须由 `model_runner_kv_cache_mixin.py` 在 `is_hybrid_swa` 时一起构造。

---

## 16. 环境变量与优化开关

SWARadixCache 相关的可调开关（均在 `environ.py`）：

| 环境变量 | 默认 | 行号 | 作用 |
|---|---|---|---|
| `SGLANG_OPT_CACHE_SWA_TRANSLATION` | True | 651 | 缓存 full→swa 翻译结果（`translate_loc_from_full_to_swa` 的单条目缓存） |
| `SGLANG_OPT_SWA_RADIX_CACHE_COMPACT` | **False** | 653 | `_compact_single_child_chain`（`:970`）合并单子链节点。**有 bug**（retract 时 pool 计量漂移；window>page_size 时覆盖活跃 swa_uuid），默认关。FIXME by ispobock |
| `SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT` | False | 654 | 插入时把过长叶切到一个窗口大小（`_maybe_split_leaf_for_swa_lock`，`:1016`），避免 chunked prefill 下 `swa_protected_size_` 膨胀 ~`chunked_prefill_size/window` 倍 |
| `SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW` | False | 655 | 启用 `dec_swa_lock_only` 早释放（§10.3）：decode 越窗后提前释放 swa 锁 |
| `SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN` | False | 656 | 去掉 `_evict_swa` 的 `-page_size` 边距（§13.3）。开启后 `evict_threshold = pre_len - W`（更激进，但可能造成 tombstone 叶 / 泄漏） |
| `SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER` | 1.0 | 304 | `maybe_evict_swa` 主动驱逐间隔 = `W × 该值`（页对齐）。调大减少驱逐开销但浪费更多 swa token |

### _maybe_split_leaf_for_swa_lock 细节（`:1016`）

`inc_lock_ref` 会给叶子保护 `len(leaf.value)` 个 swa token，但实际只需最后 `W` 个。chunked prefill 下叶可能长达数千 token，导致 `swa_protected_size_` 虚高、swa 池过早耗尽、retract 抖动。该优化把叶在「最小的、页对齐的、能覆盖窗口的尾段」处切开：

```python
tail_size = ceil(sliding_window_size / page_size) * page_size
if len(leaf.value) > tail_size:
    split_at = len(leaf.value) - tail_size
    _split_node(leaf.key, leaf, split_at)   # 切出 [前半 | 窗口尾段]
```

### _compact_single_child_chain 风险（`:970`）

合并 `parent→唯一child→...` 链以减少节点数，但有两个已知问题（代码注释 FIXME）：
1. retract 时 pool 计量漂移（commit 6348cb506）；
2. `window > page_size` 时会覆盖活跃的 `swa_uuid`，破坏 swa 锁边界。

故默认关闭。合并时必须保留 `is_bigram` 标志（`:994`），否则 EAGLE/MTP 的 bigram key 会被静默降级，导致 `match()` 返回 0 触发 `_split_node` 断言。

---

## 17. 关键文件 / 行号索引

### swa_radix_cache.py（主模块，1349 行）

| 符号 | 行号 | 说明 |
|---|---|---|
| `TreeNode` | 57 | 双锁 + tombstone 节点 |
| `TreeNode.evicted` / `backuped` | 97 / 101 | `value is None` / `host_value is not None` |
| `gen_swa_uuid` | 108 | swa 锁边界 uuid 自增 |
| `LRUList` | 119 | 侵入式双向链表（full/swa 共用，反射切换字段） |
| `LRUList.reset_node_and_parents_mru` | 183 | 整条路径移 MRU（子比父新） |
| `LRUList.get_leaf_lru_no_lock` | 230 | full 驱逐用（必须叶） |
| `LRUList.get_lru_no_lock` | 224 | swa 驱逐用（可内部节点） |
| `LRUList.sanity_check` | 291 | 重建对账（昂贵，debug/idle） |
| `SWARadixCache` | 343 | 主类 |
| `__init__` | 344 | assert allocator 类型；读 sliding_window_size |
| `reset` | 373 | root 双锁=1，建两条链表 |
| `match_prefix` | 389 | 滑窗感知匹配入口 |
| `insert` | 417 | 插入入口 |
| `cache_finished_req` | 438 | 请求结束（带 swa_evicted_seqlen） |
| `cache_unfinished_req` | 487 | chunked 中途 |
| `total_size` | 560 | 返回 (full, swa) |
| `evict` | 563 | 双预算两段驱逐 |
| `inc_lock_ref` | 670 | full 锁到 root，swa 锁到窗口边界 |
| `dec_lock_ref` | 711 | 逆操作，支持 skip_swa |
| `dec_swa_lock_only` | 760 | swa 锁早释放（叶→tombstone） |
| `full/swa_evictable_size` | 823 / 826 | 分量计量 |
| `full/swa_protected_size` | 833 / 837 | 分量计量 |
| `_match_prefix_helper` | 866 | match_len_since_tombstone 逻辑 |
| `_match_post_processor` | 934 | 截断到 best_value_len + LRU 更新 |
| `_compact_single_child_chain` | 970 | 单子链合并（默认关，有 bug） |
| `_maybe_split_leaf_for_swa_lock` | 1016 | 切叶到一个窗口 |
| `_split_node` | 1053 | 分裂节点 + 双链表维护 |
| `_insert_helper` | 1092 | 三分支复活 + case1/2/3 建节点 |
| `_add_new_node` | 1220 | 建新节点（tombstone 不入 swa 链表） |
| `_iteratively_delete_tombstone_leaf` | 1242 | 维持「叶非 tombstone」不变量 |
| `_tombstone_internal_node` | 1277 | 内部节点 tombstone 化 |
| `_total_size_helper` | 1336 | DFS 统计 full/swa 总量 |

### 配套文件

| 文件 | 关键符号 / 行号 |
|---|---|
| `mem_cache/swa_memory_pool.py` | `SWATokenToKVPoolAllocator`(305)、`available_size`(388)、`alloc`(427)、`alloc_extend`(448)、`alloc_extend_swa_tail`(500)、`free`(597)、`free_swa`(630)；`SWAKVPool`(29)、`translate_loc_from_full_to_swa`(423) |
| `mem_cache/base_swa_memory_pool.py` | `BaseSWAKVPool`(9) ABC |
| `mem_cache/base_prefix_cache.py` | `BasePrefixCache`(200)；`InsertParams`(53，含 `prev_prefix_len`/`swa_evicted_seqlen`)、`EvictParams`(80，含 `swa_num_tokens`)、`IncLockRefResult`(98，含 `swa_uuid_for_lock`)、`MatchResult`(149) |
| `mem_cache/common.py` | `evict_from_tree_cache`(330)：双预算计算并调 evict |
| `mem_cache/registry.py` | `default_radix_cache_factory`(76)；`is_hybrid_swa → SWARadixCache`(129) |
| `managers/schedule_policy.py` | `is_hybrid_swa`(460)、`rem_total_tokens`(496)、`rem_swa_tokens`(515)、`_swa_budget_for_req`(543)、`budget_state`(563)、`add_chunked_req`(666)、`add_one_req`(859/879) |
| `managers/schedule_batch.py` | `Req.swa_evicted_seqlen`(721)、`swa_uuid_for_lock`(820)、`swa_prefix_lock_released`(822)、`cache_protected_len`(824)；`maybe_evict_swa`(2645)、`_evict_swa`(2704) |
| `managers/scheduler_components/invariant_checker.py` | `_get_total_uncached_sizes`(147)：full/swa 分别对账 |
| `configs/model_config.py` | `is_hybrid_swa_model`(1657)、`get_hybrid_layer_ids`(1675) |
| `environ.py` | SWA 相关开关(651-656)、`SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER`(304) |

### 相关 PR（git 历史）

- `35870d55a` Deepseek V4（引入 DSV4 的 SWA 用法）
- `4b6f77688` feat(kv-events): publish SWA radix cache events（kv-events 接入）
- `33c57b871` Fix LRU list reference cycle leak in radix_cache（LRU 链表引用环泄漏修复，`_remove_node` 清空自指针）
- `8cb957ccf` Make EAGLE bigram key an O(1) view on RadixKey（影响 `_compact_single_child_chain` 的 is_bigram 保留）

---

## 附：一图总览数据流

```
请求到达
  │
  ├─ match_prefix ──► 滑窗感知匹配，截断到 best_value_len（避开 tombstone 洞）
  │                   两链表路径移 MRU
  │
  ├─ PrefillAdder.add_one_req ──► 检查 rem_total_tokens(full) & rem_swa_tokens(swa)
  │       │                       _swa_budget_for_req 估单请求 swa 需求
  │       └─ alloc 不够 ──► evict_from_tree_cache
  │                            算 (full_need, swa_need)
  │                            └─ evict：① full 删叶  ② swa 把内部节点 tombstone 化
  │
  ├─ prefill forward（chunked）
  │       └─ cache_unfinished_req ──► insert + 重匹配 + 换锁(inc_lock_ref → swa_uuid)
  │
  ├─ decode forward（逐步）
  │       └─ maybe_evict_swa（每 interval 步）
  │              ├─ _evict_swa：free_swa 已滑出窗口的 KV，推进 swa_evicted_seqlen
  │              └─ 越窗后 dec_swa_lock_only：提前释放 swa 锁（叶→tombstone）
  │
  └─ 请求结束
          └─ cache_finished_req ──► insert(swa_evicted_seqlen) 处理 tombstone 复活/丢弃
                                    dec_lock_ref(skip_swa=已早释放)
```

