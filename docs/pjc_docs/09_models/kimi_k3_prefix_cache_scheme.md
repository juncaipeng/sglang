# SGLang 支持 Kimi-K3 的前缀缓存（Prefix Cache）方案

> 撰写基准：`/work/repos/sglang` `main` 分支（2026-08）。
>
> **一句话**：K3 是「**24 层 MLA（token 级 KV）+ 69 层 KDA（线性注意力/SSM，递归状态）**」的混合模型。
> 它的前缀缓存不能沿用普通 RadixCache，而是走 **一棵基数树 + 每个节点两类 value（FULL 逐 token KV
> + MAMBA 单个递归状态检查点）** 的 `UnifiedRadixCache(FULL, MAMBA)` 方案。核心难点全在 **MAMBA
> 那一半**：递归状态是「整段前缀的压缩摘要」，只能在 **chunk 对齐的检查点位置**被复用，且复用时必须
> **写时拷贝（CoW）**。
>
> 关联文档：
> - K3 整体支持方案见 [kimi_k3_support_scheme.md](kimi_k3_support_scheme.md)（本文是其中「双显存池 + 前缀缓存」一环的深挖）；
> - 统一缓存框架总览见 [unified_radix_cache_architecture.md](../03_cache_memory/unified_radix_cache_architecture.md)（TreeCore/Component 分层、HiCache 三级）；
> - SWA 版前缀缓存的 tombstone/双锁语义见 [swa_radix_cache_architecture.md](../03_cache_memory/swa_radix_cache_architecture.md)。

---

## 0. 一页速览

| 维度 | 内容 |
|---|---|
| 模型结构 | 93 层 = **24 MLA**（DeepSeek 式，token 级分页 KV）+ **69 KDA**（门控 delta-rule 线性注意力，递归状态） |
| 缓存实现 | `UnifiedRadixCache`，组件 `(ComponentType.FULL, ComponentType.MAMBA)`（`registry.py:175` 对 `is_hybrid_ssm` 追加 MAMBA） |
| 每节点两类 value | **FULL** = 逐 token 的 MLA KV 索引（可切分、任意前缀可复用）；**MAMBA** = **单个**递归状态检查点 slot（**不可切分**） |
| 检查点粒度 | `mamba_cache_chunk_size = max(模型 FLA/mamba chunk_size, page_size)`（`server_args.py:8812`，默认 chunk=64） |
| 读路径关键量 | `mamba_branching_seqlen`：FULL 命中超出最深 MAMBA 检查点后、最后一个 chunk 对齐位置——本次 forward 可在此新建检查点 |
| 复用方式 | MAMBA 命中的检查点是只读的 → **写时拷贝（CoW）**到请求私有 slot（`mamba_cow_src_index`），因为 forward 会原地改写状态 |
| 写入方式 | 在 chunk 边界「track」捕获活跃状态，请求结束/分块时把该 slot **donate**（转移所有权，非拷贝）给树节点 |
| 策略旋钮 | `--mamba-radix-cache-strategy {auto, no_buffer, extra_buffer, extra_buffer_lazy}`（`server_args.py:2528`）——决定每请求活跃状态 slot 数 `S` |
| 显存划分 | `--mamba-full-memory-ratio`（`server_args.py:2520`，默认 0.9）划分「KDA state 池 : MLA KV 池」 |
| 淘汰 | 两条独立 LRU（full / mamba），`available = min(full, mamba)`；MAMBA 命中只刷新命中节点本身 |
| HiCache | MAMBA 组件有完整 L1→L2(host)→L3(storage) 钩子；mamba = 叶专属数据 |
| 压缩 | `--enable-int8-mamba-checkpoint`：树里缓存的 mamba 状态用 int8 存 → 缓存容量 ~2× |

---

## 1. 核心难题：混合模型为什么不能直接用 RadixCache

普通 `RadixCache` 的前缀复用建立在一个前提上：**KV 是逐 token 的，第 i 个 token 的 KV 只依赖前 i 个 token**。
所以命中长度 = L 的前缀，就能原样复用前 L 个 token 的 KV，剩下的接着算。

K3 的两种注意力对这个前提的满足程度完全不同：

| 层类型 | 状态形态 | 能否任意前缀复用 |
|---|---|---|
| **MLA（24 层）** | 逐 token 的潜在 KV，写进分页 KV 池 | ✅ 能。和普通全注意力一样，任意前缀长度都能复用 |
| **KDA（69 层）** | **固定大小的递归状态** `state[N]`，是 token 0..N 的**压缩摘要**（类 SSM） | ❌ 不能。`state[N]` 无法「切」出 `state[L]`（L<N）；只有当年**恰好在位置 L 存过检查点**，才能从 L 续算 |

```
             token:  0   1   2   3   4   5   6   7  ...
   ┌──────────────────────────────────────────────────────
   │ MLA 层  KV:    [k0][k1][k2][k3][k4][k5][k6][k7]   ← 逐 token，任意前缀可切
   │
   │ KDA 层  state: ────────────────────────────►  state[7]
   │                 （一个不断被覆盖的递归状态，中间态不落盘）
   │                 只有 chunk 边界(如 0,64,128…)会物化出可复用的 state
   └──────────────────────────────────────────────────────
```

**结论**：混合模型的前缀缓存必须在**同一棵树**里同时表达两种复用语义——FULL 逐 token、MAMBA 只在检查点。
这正是 `UnifiedRadixCache(FULL, MAMBA)` 存在的理由。

> 说明：`main` 上另有独立类 `MambaRadixCache`（`mem_cache/mamba_radix_cache.py`），是这套语义的
> **早期/参照实现**，代码里语义与 MAMBA 组件一一对应。但 `registry.py` 的默认选择链对 `is_hybrid_ssm`
> **一律走 UnifiedRadixCache**（见 §3）。本文两处都引，因为独立版把「一个节点同时挂 `value` 和
> `mamba_value`」写得最直白。

---

## 2. 总体方案：一棵树，每节点两类 value

一棵基数树，边上是 token 序列；每个节点 `component_data` 里同时挂：

| 字段 | 组件 | 语义 | 可切分？ | 复用边界 |
|---|---|---|---|---|
| `component_data[FULL].value` | FullComponent | 该节点边上这段 token 的 **MLA KV 索引** | ✅ 节点可 `_split_node` | 命中路径全长 |
| `component_data[MAMBA].value` | MambaComponent | **一个** mamba 池 slot（该 token 位置的 **KDA 递归状态检查点**） | ❌ 「mamba cache can not be split」 | 仅有 `mamba_value` 的节点 |

关键不对称（独立版 `mamba_radix_cache.py:1197` 一行点破）：

```python
def _split_node(self, key, child, split_len):
    new_node = TreeNode()
    ...
    new_node.mamba_value = None   # mamba cache can not be split
```

- **切节点时**：FULL 的 KV 可以按 token 切成两半分给父/子；但 MAMBA 的 `value` 是「到某个 token 为止的
  整段摘要」，**切不了**——所以 split 出来的新父节点 `mamba_value = None`，只有原叶保留状态。
- 因此 **MAMBA 状态是「叶/检查点专属」数据**，不是逐 token 数据。这条不变量贯穿整个设计（HiCache
  里 `redistribute_on_node_split` 也把 mamba host_value 留在 child，`mamba_component.py:313`）。

---

## 3. 装配路径：怎么选到 `UnifiedRadixCache(FULL, MAMBA)`

`registry.py:default_radix_cache_factory`（:79）的选择链：

```
is_hybrid_ssm ?  ──► _create_unified_radix_cache(ctx)          # registry.py:114-115
                        └─ tree_components = [FULL]
                           if is_hybrid_swa: + SWA
                           if is_hybrid_ssm: + MAMBA            # registry.py:175-176
                           └─ UnifiedRadixCache(params)
```

- `is_hybrid_ssm` 是「模型含 SSM/线性注意力层」的判定（K3 的 69 层 KDA 命中）。
- 派生标志 `uses_mamba_radix_cache`（`server_args.py:2537`，`no_cli`）：由模型架构解析，标记该模型走混合-mamba 缓存路径。
- 关 radix + chunked prefill 时退化到 `ChunkCache` 系（:84-95）——**没有前缀复用**，每请求各算各的（PD 的 decode 侧就是这种，见 §10）。
- `SGLANG_ENABLE_UNIFIED_RADIX_TREE` / `use_mlx()` 会**强制**走 unified（:104）；MLX 后端用
  `MlxAuxiliaryStateComponent` 替换 MAMBA 组件（:184）。

装配完成后，`UnifiedRadixCache` 的 `is_mamba_enabled = ComponentType.MAMBA in tree_components`（`unified_radix_cache.py:209`）。

---

## 4. 检查点粒度：`mamba_cache_chunk_size`

这是整个 MAMBA 侧前缀缓存的「量子」。定义在 `server_args.py:8812`：

```python
@property
def mamba_cache_chunk_size(self) -> int:
    chunk_size = getattr(hf_config, "mamba_chunk_size", FLA_CHUNK_SIZE)  # 默认 FLA_CHUNK_SIZE=64
    page_size  = resolved_view(self).page_size
    assert max(chunk_size, page_size) % min(chunk_size, page_size) == 0
    self._mamba_cache_chunk_size = max(chunk_size, page_size)
```

为什么必须 chunk 对齐？因为 KDA 的 prefill 走 **chunk 扫描内核**（`causal_conv1d_fn` + chunk KDA，见
[kimi_k3_support_scheme.md](kimi_k3_support_scheme.md) §3.2）：递归状态**只在每个 chunk 结束时才被物化**出来。
chunk 中间的状态在 `h`（临时缓冲）里、扫描完就丢。所以：

- 能被存进 radix、被下次复用的 mamba 状态，**只能落在 chunk 边界**（64、128、192…）。
- `mamba_cache_chunk_size` = 模型 chunk_size 与 page_size 的公倍数上界，保证既对齐 FLA 扫描，又对齐分页。

> 校验：`chunked_prefill_size < mamba_cache_chunk_size` 会告警（`server_args.py:5676`）——分块比一个
> mamba chunk 还小，会让每个分块都到不了检查点边界，退化缓存效率。

---

## 5. 读路径 `match_prefix`：两套命中 + 分叉点 + CoW

以独立版 `mamba_radix_cache.py` 的 `_match_prefix_helper`（:1070）+ `_match_post_processor`（:1121）为准
（unified 版逻辑同构，见 `mamba_component.py:finalize_match_result_in_tree_core` :150）。

### 5.1 两套命中长度

沿树往下走时，同时维护两个「命中末端」：

| 命中 | 末端 | 语义 |
|---|---|---|
| FULL 命中 | `last_node`（走到底的节点） | MLA KV 复用到这里；**整条路径**都算命中前缀 |
| MAMBA 命中 | `best_last_node` = 最深的、**有 `mamba_value` 的**节点 | KDA 递归状态只能续到这个检查点 |

```
root ──A──►(n1: FULL✓)──B──►(n2: FULL✓ MAMBA✓)──C──►(n3: FULL✓)   ← 走到 n3
                                    ▲                        ▲
                                    │                        │
                      best_last_node(MAMBA 命中到 n2)   last_node(FULL 命中到 n3)
```

- FULL：`last_node = n3`，MLA KV 复用 A+B+C 全段。
- MAMBA：`best_last_node = n2`（n3 没存检查点），KDA 状态只能从 n2 的检查点续算，C 段的 KDA 要重算。

### 5.2 分叉点 `mamba_branching_seqlen`（最关键的派生量）

当 FULL 命中比 MAMBA 命中更长（`len(value) > best_value_len`），说明「C 段有 token KV 可复用，但没有
mamba 检查点」。此时算出**本次 forward 可以在哪里补一个新检查点**：

```python
# mamba_radix_cache.py:1152 / mamba_component.py:166
chunk_aligned_seqlen = (full_kv_hit_length // mamba_cache_chunk_size) * mamba_cache_chunk_size
mamba_branching_seqlen = chunk_aligned_seqlen if chunk_aligned_seqlen > mamba_boundary_len else None
```

= FULL 命中范围内、**超出当前 mamba 边界的、最后一个 chunk 对齐位置**。调度侧
（`schedule_batch.py:2675`）看到 `mamba_branching_seqlen` 落在本次 extend 批内，就会在该位置额外「track」
一份状态存回树——**让下次相同前缀能从更深处复用 KDA 状态**。这是混合缓存「让 MAMBA 命中逐渐追上 FULL
命中」的自愈机制。

### 5.3 写时拷贝（CoW）：命中的检查点是只读的

MAMBA 命中的 `mamba_value` 属于 radix 树（可能被多个请求共享）。但 forward 会**原地推进**递归状态，
所以复用前必须拷到请求私有 slot（`mamba_radix_cache.py:1164` / `mamba_component.py:182`）：

```python
if cow_mamba and last_node.mamba_value is not None:
    if req.mamba_pool_idx is None:
        dst_index = mamba_allocator.alloc(1)      # 分不到就 evict 再分
        req.mamba_pool_idx = dst_index[0]
    req.mamba_cow_src_index = last_node.mamba_value  # 记源，H2D/D2D 拷贝推迟到 forward 流
    req.mamba_needs_clear = False
```

- CoW 拷贝**推迟到 forward 流**执行（记 `mamba_cow_src_index` 即可），避免和调度线程抢 GPU。
- 分不到私有 slot 时会 `inc_lock_ref` 锁住命中节点、`evict(mamba_num=1)`、再分——锁住是为了让淘汰的
  窗口释放停在本请求边界，不误伤其他请求（`mamba_component.py:198` 注释）。

### 5.4 淘汰刷新的不对称

命中后刷新 LRU 时，两组件策略不同（`mamba_radix_cache.py:1132` / `mamba_component.py:113`）：

- **FULL**：刷新**整条命中路径**（`reset_node_and_parents_mru`）——因为整条路径都被当前缀复用。
- **MAMBA**：**只刷新命中的那个检查点节点**（`reset_node_mru(last_node)`）。若刷整条，会把一个会话的所有
  状态在 mamba LRU 里排到一起，冷会话被整窝淘汰；只碰用到的状态，老叶子能活久点。

---

## 6. 写路径：track 捕获 + donate 转移

### 6.1 track：在 chunk 边界捕获活跃状态

decode/prefill 推进时，递归状态在活跃池里不断被覆盖。要存检查点，就得在**恰好 chunk 对齐**的那一步
把状态「track」（快照）出来。调度侧 `schedule_batch.py:2633` 附近：

```python
mask = req.extend_range.length >= chunk_size
if mask:
    mamba_track_seqlen_aligned = len(prefix) + (extend_len // chunk_size) * chunk_size
    ... # 处理 1a(末位且对齐→读 last_recurrent_state) vs 2(非对齐→读 h) 的差异，必要时 _force_track_h(+1)
    req.mamba_last_track_seqlen = mamba_track_seqlen_aligned   # 实际存进树的 key 长度
```

- `mamba_track_interval`（`exec.mamba` 命名空间）= track 的步长节拍（decode 每隔若干步到达 chunk 边界才 track）。
- 若 §5.2 的 `mamba_branching_seqlen` 落在本批范围内，就把 track 点对准分叉点（:2675-2691），顺手补检查点。

### 6.2 donate：把 slot 所有权转给树（不拷贝）

请求 `cache_finished_req` / `cache_unfinished_req` 时，把 track 出来的活跃 slot **转移**给 radix 节点。
关键是「donate 而非 copy」，避免和 forward 流的数据竞争（`mamba_radix_cache.py:709` /
`mamba_component.py:518`）：

```python
# 未结束请求：把 mamba 索引 donate 给 radix，然后给请求换一个新活跃 slot
mamba_value_donated = donate_mamba_ping_pong_slot(req, new_slot)   # extra_buffer 路径
insert(InsertParams(key=..., value=page_aligned_kv_indices, mamba_value=mamba_value_donated, ...))
if mamba_exist:            # 树里已有相同检查点 → donate 的这份是重复，释放掉
    _free_mamba_value(mamba_value_donated)
```

- insert 的 key 用 `mamba_last_track_seqlen` 截断到 chunk 对齐长度（`page_aligned_len`），保证 FULL/MAMBA
  在该节点长度一致。
- `mamba_exist=True`（同检查点已被别的请求存过）→ 释放本请求这份，靠引用共享。

---

## 7. 三种策略：每请求活跃 state slot 数 `S`

`--mamba-radix-cache-strategy`（`server_args.py:2528`，默认 `auto` 按模型解析）。它决定「**每个在跑请求
在活跃池里占几个 mamba state slot**」——这就是 `--mamba-full-memory-ratio` 公式里的 `S`
（见 [kimi_k3_support_scheme.md](kimi_k3_support_scheme.md) §4）。

| 策略 | 标志 | 活跃 slot 数 `S` | 机制 | 约束 |
|---|---|---|---|---|
| `extra_buffer` | `enable_mamba_extra_buffer` | 5 | **ping-pong 追踪缓冲**：一边 track 出检查点、一边让活跃状态继续推进（overlap 下 2 个 track slot，`memory_pool.py:1170`） | page_size 可 >1 |
| `extra_buffer_lazy` | `enable_mamba_extra_buffer_lazy` | 4 | 惰性 ping-pong：第 2 个 slot 到 chunk 边界才按需分配（`schedule_batch.py:2915 mamba_lazy_prealloc_at_boundary`） | 不支持 PD 分离 / 部分投机算法（`server_args.py:5655`） |
| `no_buffer`（ReplaySSM 常见默认） | 两者皆 False | 3（关投机时更少） | 无额外缓冲；用 `replayssm_write_pos` 环形游标，donate 时回退到最后 flush 边界（`mamba_radix_cache.py:565`） | **page_size 必须 == 1**；不支持 overlap 调度、不支持 trtllm_mha（`server_args.py:5637`） |
| 关 radix | — | 1 | 无前缀复用 | — |

**ping-pong 追踪缓冲（extra_buffer）为什么需要两个 slot**：track 出的检查点要「冻结」交给 radix，但活跃
状态还得继续往后算——于是用两个 slot 交替（`mamba_next_track_idx` 在 `get_mamba_ping_pong_other_idx`
间翻转，`memory_pool.py:1384`），`donate_mamba_ping_pong_slot`（:1438）把要冻结的那半 donate 给树、
给请求补一个新 slot 接着写。

### 7.1 int8 检查点池（正交压缩）

`--enable-int8-mamba-checkpoint` 打开后，**树里缓存的 mamba 状态存 int8**（不是活跃池的 bf16），
`mamba_radix_cache.py:1035`：

- 活跃计算仍用 bf16 池；donate 时 `_commit_int8_checkpoint` 量化进独立 `mamba_ckpt_pool`。
- 效果：固定显存下**缓存前缀容量 ~2×**。CoW load 回来时再反量化。

---

## 8. 淘汰：双 LRU + 路径上限 + tombstone

### 8.1 两条独立 LRU、两套预算

`available_size = min(full_available, mamba_available)`。淘汰按各池缺口分别驱动
（`EvictParams(num_tokens=..., mamba_num=...)`）：

- `evict_full(n)`（`mamba_radix_cache.py:880`）：只删 FULL 叶。
- `evict_mamba(n)`（:845）：驱逐 mamba 检查点。内部节点的 mamba 状态可就地 tombstone（free 掉 state、
  节点还在），叶子才整删。

### 8.2 路径上限 `mamba_max_states_per_path`

一条根到叶的路径上，允许常驻的 mamba 检查点数有软上限（`mamba_component.py:247 _emit_excess_path_states_eviction`
/ :257 `_evict_excess_path_states`）。超了就从**浅处**驱逐可淘汰的检查点（保留 tail、分叉点、锁住的、
叶子）。避免长会话在一条路径上堆积几十个检查点吃满 state 池。

### 8.3 tombstone / 叶集不变量

复用 SWA 那套 tombstone 思想（见 [swa_radix_cache_architecture.md](../03_cache_memory/swa_radix_cache_architecture.md)）：
mamba 状态从设备驱逐后，若还有 host 备份，节点转入 host LRU（`mamba_component.py:344`）；纯 host 的
节点在 `evictable_host_leaves` 里参与 host 淘汰。

---

## 9. HiCache 三级存储（MAMBA 组件的钩子）

`MambaComponent` 实现了完整的 L1(GPU)↔L2(host)↔L3(storage) 传输钩子（`mamba_component.py:641` 起）：

| 相位 | 钩子 | 动作 |
|---|---|---|
| BACKUP_HOST | `build_hicache_transfers` :700 | 设备 mamba slot → host 池 |
| LOAD_BACK | :711 | host → 设备；含**逐请求 CoW**（H→D 拷进请求私有 slot :728） |
| BACKUP_STORAGE | :741 | host → L3（Mooncake 等），key = 节点末段 hash，`TRAILING_PAGES` 策略 |
| PREFETCH | :754 | L3 → host，占位 key 先探再落 |

要点：**mamba = 叶专属数据**，所以 host 备份在 split 时留在 child（:313）；load-back 前先在请求侧预分配
设备 slot（`prepare_load_back` :643），失败则回滚（`finalize_load_back` :665）。

> cookbook 备注（K3 §3.3）：DCP 配方下 host 层尚未完全 DCP 感知；L3 恒可用，L1+L2 在投机开启时需退回纯 TP。

---

## 10. PD 分离下的前缀缓存差异

K3 是混合模型，PD 传输**同时搬分页 MLA KV 和 KDA recurrent state**（见 [kimi_k3_support_scheme.md] §9）。
但前缀缓存在两侧不同：

| 侧 | 缓存形态 | 说明 |
|---|---|---|
| **Prefill 侧** | 完整 `UnifiedRadixCache(FULL, MAMBA)` | 正常前缀复用（含 mamba 检查点） |
| **Decode 侧** | **chunk cache，每请求 1 slot** | `--mamba-radix-cache-strategy` 失效；`--disaggregation-decode-extra-slots` 需显式钉住（cookbook §3.4） |

原因：decode 侧请求的前缀 KV/state 是从 prefill 侧**传过来的**（`prefix_len` 已由传输给定），decode 侧
本地不重算前缀，也就没有「在 decode 侧维护 mamba 检查点树」的收益——退化成 chunk cache（`registry.py:84`
关 radix + chunked 分支）。用户当前打开的 `disaggregation/decode.py` 正是这条路径的实现载体。

---

## 11. 与 KimiLinear / K3 的接线

K3 本体不在 `main`（见 [kimi_k3_support_scheme.md] §1），但缓存接线可从近亲 `KimiLinear` 完整看到：

| 环节 | 位置 | 作用 |
|---|---|---|
| 逐层布局表 | `configs/kimi_linear.py:139 is_kda_layer` | 告诉运行时哪些层是 KDA（走 MAMBA 组件）、哪些是 MLA（走 FULL 组件） |
| state 形状 | `configs/kimi_linear.py:154 mamba2_cache_params` → `KimiLinearStateShape` | 单个 mamba slot 的字节数（`attn_tp × heads × head_dim × short_conv_kernel`）——决定 state 池每 slot 大小、进而并发上限 |
| MLA 层 | `models/kimi_linear.py:48 KimiMLAAttention = DeepseekV2AttentionMLA` | 逐 token 潜在 KV，喂给 FULL 组件 |
| 混合请求池 | `mem_cache/memory_pool.py:1136 HybridReqToTokenPool` | 同时持有 `req_to_token`（FULL）与 `mamba_pool`/`mamba_allocator`（MAMBA），ping-pong 缓冲也在这 |
| 派生标志 | `server_args.py:2537 uses_mamba_radix_cache` | 由架构解析，标记走混合-mamba 缓存 |

K3 相对 KimiLinear 的增量（视觉塔、2.8T、MXFP4、恒开思考）**不改缓存骨架**——前缀缓存这一层是同一套。

---

## 12. 文件与行号索引

| 文件 / 位置 | 内容 |
|---|---|
| `mem_cache/registry.py:79 / :114 / :175` | 选择链：`is_hybrid_ssm` → UnifiedRadixCache + MAMBA 组件 |
| `mem_cache/unified_cache/components/mamba_component.py` | **MAMBA 组件全实现**：match/insert/CoW/淘汰/HiCache/路径上限 |
| `mem_cache/mamba_radix_cache.py` | 独立/参照版 `MambaRadixCache`：`match_prefix`:500、`cache_finished_req`:543、`cache_unfinished_req`:668、分叉点:1152、`_split_node`:1192 |
| `mem_cache/unified_radix_cache.py:209` | `is_mamba_enabled` |
| `server_args.py:8812 mamba_cache_chunk_size` | 检查点粒度 = max(chunk_size, page_size) |
| `server_args.py:2520 mamba_full_memory_ratio` / `:2528 mamba_radix_cache_strategy` | 双池比 / 三策略 |
| `server_args.py:5637 _validate_mamba_no_buffer` | no_buffer 约束（page_size=1、禁 overlap、禁 trtllm_mha） |
| `managers/schedule_batch.py:2633` | track（chunk 边界捕获）；:2675 分叉点补检查点；:2915 lazy 预分配 |
| `mem_cache/memory_pool.py:1136 HybridReqToTokenPool` | 混合请求池；:1384/:1438 ping-pong；:1170 track buffer 大小 |
| `configs/kimi_linear.py:139/:154` | 逐层布局 / state 形状（K3 参照） |

---

## 13. 待确认 / 风险点

- 本文对 K3 本体 KDA `forward` 与逐层布局的描述来自 `KimiLinear` 参照 + cookbook；`kimi-k3` 分支 merge
  进 `main` 后需按实际 `configs/kimi_k3.py` 复核层数比例（69:24）与 state 形状。
- `--mamba-radix-cache-strategy=auto` 的**逐模型解析结果**（K3 具体落到 no_buffer 还是 extra_buffer_lazy）
  未在本文用代码坐实；`server_args.py:5637-5705` 只给了各策略的约束校验。K3 cookbook 大规模配方用的是
  `extra_buffer_lazy`。
- `S` 的取值（5/4/3/1）来自 [kimi_k3_support_scheme.md] 引用的 cookbook；未在 `server_args.py` 找到把这
  几个数字硬编码的单点，实际值应以运行时 state 池预留为准。
- PD decode 侧退化为 chunk cache 的结论基于 registry 选择链 + cookbook §3.4，未逐行跟踪 `decode.py` 的
  mamba slot 预分配路径。
