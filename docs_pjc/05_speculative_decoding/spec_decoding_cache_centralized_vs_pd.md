# 投机解码层的 Cache：创建、管理与传输（集中式 vs PD 分离）

> **本文回答什么**：开了 `--speculative-algorithm` 之后，多出来的那份 Cache 是谁建的、建多大、
> 运行期怎么分配和回收、在 PD 分离下怎么从 P 搬到 D。
>
> **核心目录**：`python/sglang/srt/speculative/`、`python/sglang/srt/mem_cache/`、
> `python/sglang/srt/disaggregation/`、`python/sglang/srt/model_executor/pool_configurator.py`
>
> **目标读者**：需要调 `--mem-fraction-static`、排查投机解码 OOM、或者要给新算法/新模型
> 接投机解码 + PD 分离的人。
>
> ⚠️ **读前必看的 5 条结论**
> 1. **draft 模型有自己独立的 KV pool 对象，但共享 target 的 allocator 和 `req_to_token_pool`。**
>    一个 slot 编号在两个池里各写一份 KV。这是理解后面所有内容的地基。
> 2. **draft 的显存不是单独一笔预算**，而是把 target 的"每 token 字节数"(`cell_size`) 膨胀，
>    再用同一次 profiling 除出来。所以开投机解码的直接后果是 `max_total_num_tokens` 变小。
> 3. **被拒绝的 draft token 的 KV slot 不会被逐个释放**，它们留在 `kv_committed_len` 与
>    `kv_allocated_len` 之间的"超额区"，请求结束/被 retract 时一次性批量回收。
> 4. **PD 分离下 draft KV 是"搬过去"的，不是在 D 侧重算的。** 因为 draft 池与 target 池共享
>    slot 编号，P 侧只要把 draft 池的 buffer 指针追加到同一个注册列表，同一份页号数组就同时
>    覆盖了两个池。
> 5. **P 实例照常跑 draft 模型**（EAGLE 系没有任何 `disaggregation_mode == "prefill"` 的分支），
>    产出的 `topk_p / topk_index / hidden_states` 走 aux（metadata）通道而不是 KV 通道传给 D。

## 0. 与既有文档的分工

本目录已有若干投机解码相关文档，本文只补它们没覆盖的部分，重复的地方直接引用。

| 文档 | 覆盖 | 与本文的关系 |
|---|---|---|
| [speculative_decoding_overview.md](speculative_decoding_overview.md) | 算法家族全景、参数、三阶段主循环、verify 逻辑 | 算法语义看它；本文只讲 Cache |
| [mtp_speculative_decoding_architecture.md](mtp_speculative_decoding_architecture.md) | MTP 专题、draft 权重加载、§18 讲 P→D 三个字段的**语义** | 本文 §6 补它的**传输机制**（注册、通道、配对） |
| [dsv4_mtp_layer_forward_dataflow.md](mtp_dsv4_layer_forward_dataflow.md) | DSV4 MTP 层前向；§3 逐行讲 DSV4 draft pool 创建 | 本文 §4 讲**通用模型**的创建路径 |
| [dsv4_mtp_centralized_execution.md](mtp_dsv4_centralized_execution.md) | DSV4 集中式执行编排 | 本文 §5 讲**通用**运行期分配/释放 |
| [dsv4_pd_p2d_transfer_contents_and_method.md](../04_pd_disaggregation/dsv4_pd_p2d_transfer_contents_and_method.md) | DSV4 PD 传输内容清单与索引数学 | 本文 §6 讲**通用**的 draft 注册与门禁 |
| [pd_disaggregation_kv_transfer_architecture.md](../04_pd_disaggregation/pd_disaggregation_kv_transfer_architecture.md) | 通用 PD 传输框架、后端、握手 | 本文 §6 是它的投机解码补章 |
| **本文** | **draft cache 的创建 + 运行期管理 + 释放 + PD 传输，一条线打通** | — |

⚠️ 所有行号基于撰写时的 `main` 快照（`33ed29a0ee`），代码演进后可能漂移，引用前请重新核对。

---

## 1. 目录

1. [一分钟结论](#2-一分钟结论)
2. [前置概念：投机解码涉及哪几份 Cache](#3-前置概念投机解码涉及哪几份-cache)
3. [创建阶段：draft KV pool 从哪来、多大](#4-创建阶段draft-kv-pool-从哪来多大)
4. [集中式部署的运行期管理](#5-集中式部署的运行期管理)
5. [PD 分离部署：注册、传输、种子](#6-pd-分离部署注册传输种子)
6. [集中式 vs PD 全面对照](#7-集中式-vs-pd-全面对照)
7. [排查手册](#8-排查手册)
8. [已发现的粗糙边缘](#9-已发现的粗糙边缘)
9. [文件行号索引](#10-文件行号索引)
10. [附：最容易记混的结论](#附最容易记混的结论)

---

## 2. 一分钟结论

```
                        ┌─────────────────────────────────────────────┐
                        │        共享的一套"地址空间"                 │
                        │                                             │
   req_to_token_pool ──▶│  req_to_token[req_idx][pos] = slot_id       │
   （唯一一份）          │                                             │
                        │  token_to_kv_pool_allocator                 │
   allocator ──────────▶│  （唯一一份，唯一的分配/释放权威）          │
                        └───────────────┬─────────────────────────────┘
                                        │ 同一个 slot_id
                       ┌────────────────┴────────────────┐
                       ▼                                 ▼
              ┌──────────────────┐            ┌──────────────────┐
              │ target KV pool   │            │ draft KV pool    │
              │ L=61 层          │            │ L=1 层 (MTP)     │
              │ 由 target 前向写 │            │ 由 draft 前向写  │
              └──────────────────┘            └──────────────────┘
                       └──── 两者容量都是 max_total_num_tokens ────┘
                             （容量由同一次 profiling 决定）
```

四步法总结：

| 阶段 | 谁做 | 关键函数 | 一句话 |
|---|---|---|---|
| **建** | Scheduler 启动时 | `init_memory_pools` | target 先建，draft 复用 target 的 `MemoryPoolConfig` 再建自己的 KV 张量 |
| **算多大** | target 的 profiling | `DefaultPoolConfigurator.__init__` 里的 `cell_size` 膨胀 | 把每 token 字节数抬高，draft 不再单独 profiling |
| **分配** | 每个 decode step | `eagle_prepare_for_decode` → `alloc_for_spec_decode` | 一次预留 `2 × max(steps×topk, num_draft_tokens)`，只推进 `kv_allocated_len` |
| **回收** | 请求结束 / retract | `release_kv_cache` | `[kv_committed_len, kv_allocated_len)` 整段批量 free，不逐 token |

PD 分离额外两步：

| 阶段 | 关键函数 | 一句话 |
|---|---|---|
| **注册** | `_init_kv_manager` | draft 池的 `get_contiguous_buf_infos()` 直接**追加**到 `kv_args.kv_data_ptrs` 尾部 |
| **传** | `send_kv_chunk` | 同一份 `page_indices` 同时覆盖 target 与 draft entry，无需 draft 专属载荷 |

---

## 3. 前置概念：投机解码涉及哪几份 Cache

### 3.1 四件套

开投机解码后，一个 GPU rank 上和 KV 相关的对象一共四类：

| 对象 | 份数 | 归属 | 作用 |
|---|---|---|---|
| `req_to_token_pool` | **1 份**（共享） | 建在 target，draft 直接拿引用 | `req_to_token[req_idx][pos] → slot_id` 位置表 |
| `token_to_kv_pool_allocator` | **1 份**（共享） | 建在 target，draft 直接拿引用 | 唯一的 slot 分配/释放权威 |
| `token_to_kv_pool`（target） | 1 份 | target 的 `ModelRunner` | 存 target 的 L 层 KV 张量 |
| `token_to_kv_pool`（draft） | 1 份/draft runner | draft 的 `ModelRunner` | 存 draft 的 L' 层 KV 张量 |

Multi-Layer EAGLE 会有 `speculative_num_steps` 个 draft runner，每个一份 draft 池，但仍然共享
同一个 allocator 和 `req_to_token_pool`（`tp_worker.py:421-424`）。

### 3.2 消歧："共享池" 还是 "独立池"？

本目录里两篇 DSV4 文档的措辞看起来矛盾：一篇说"draft 与 target 共享池"，另一篇说"draft 是全新
独立的池对象"。**两句都对，说的是不同层次**：

```
共享的：  slot 编号空间（allocator）+ 位置表（req_to_token_pool）
独立的：  存放 KV 数值的张量（token_to_kv_pool 实例）
```

代码里写得很直白 —— `mem_cache/kv_cache_configurator.py:516-522`：

```python
if req_to_token_pool is None:
    req_to_token_pool = self._build_req_to_token_pool(...)
else:
    # Draft worker shares req_to_token_pool with the target worker.
    assert self.is_draft_worker
    ...
token_to_kv_pool = self._build_token_to_kv_pool(...)   # ← draft 也走这里，建自己的张量
```

而 `_build_token_to_kv_pool_allocator` 对 draft 的分支是**把传进来的 allocator 原样返回**
（`kv_cache_configurator.py:1960-1988`），只额外做一次 `register_mapping`（SWA 场景）。

这个设计的直接后果，也是本文所有结论的来源：

> **同一个 `out_cache_loc`，target 前向把它当 target 池的行号，draft 前向把它当 draft 池的行号。**
> 所以 slot 只需要分配一次、释放一次、传输一次，两份 KV 自动对齐。

`base_spec_worker.py:238-241` 的注释确认了这一点：

```python
def clear_cache_pool(self):
    """Default no-op: the allocator and kv cache pool are shared with the
    target worker and cleared by the scheduler."""
```

### 3.3 各算法的 draft cache 需求矩阵

| 算法 | 独立 draft `ModelRunner` | 独立 draft KV 张量 | 预算怎么留 | `has_draft_kv()` | `primary_draft_kv_pool` |
|---|---|---|---|---|---|
| EAGLE / EAGLE3 | 1 个 | 有，draft 层几何 | `cell_size × (1 + L'/L)` | True | draft runner 的池 |
| Multi-Layer EAGLE | `num_steps` 个 | 每步一份，共享 allocator | 同上（L' 只算一次） | True | 第 0 个 runner 的池 |
| STANDALONE | 1 个 | 有，完整小模型层数 | 同上 | True | draft runner 的池 |
| FROZEN_KV_MTP | 有，但池是 **64 token 的 dummy** | 实质无（直接读 target 池） | ⚠️ 仍按 EAGLE 膨胀（见 §9） | True | **None** |
| NGRAM | **没有 draft runner** | 无 | 不膨胀 | **False** | None |
| DFLASH / DSPARK | 1 个 | 有，自己的 head 几何 | **加法**：精确 draft 字节/token | True | `model_runner` 的池 |

判定入口 `speculative/spec_info.py:159-163`：

```python
def has_draft_kv(self) -> bool:
    """Whether the draft phase writes KV chains. NGRAM does not (its tree
    lives only in the verify mask), so per-decode KV sizing needs no
    per-topk page rounding; see get_alloc_len_per_decode."""
    return not self.is_ngram()
```

FROZEN_KV_MTP 的特殊之处 —— 它建的池只有 64 个 token，纯占位；真正读的是 target 的池
（`speculative/frozen_kv_mtp_worker_v2.py:176-196`、`:292-294`）：

```python
self.draft_pool_config = MemoryPoolConfig(
    max_total_num_tokens=64,  # Dummy value
    max_running_requests=memory_pool_config.max_running_requests)
...
ctx = draft_model.build_frozen_kv_mtp_context(
    target_model=self.target_worker.model_runner.model,
    target_token_to_kv_pool=self.target_worker.model_runner.token_to_kv_pool)
```

而 NGRAM 连 draft runner 都没有，`self.model_runner` 就是 target 的
（`speculative/ngram_worker.py:76-97`）。这两个算法在 `_draft_model_runners()` 里被直接排除
（`speculative/base_spec_worker.py:162-178`），所以 PD 分离时它们不注册任何 draft KV。

---

## 4. 创建阶段：draft KV pool 从哪来、多大

> 本节内容 **集中式和 PD 分离完全一致** —— 创建路径不看 `disaggregation_mode`。

### 4.1 创建顺序：target 先、draft 后

[scheduler.py:1009-1022](../../../python/sglang/srt/managers/scheduler.py#L1009-L1022)：

```python
def init_memory_pools(self):
    """Allocate KV cache pools for target and draft workers."""
    self.init_target_memory_pool()                       # ① target 先建
    kv_cache_builder.resolve_decode_retraction_backup(tp_worker=self.tp_worker)
    if self.draft_worker is not None:
        pool, allocator = self.tp_worker.get_memory_pool()   # ② 取出共享的两件套
        self.draft_worker.alloc_memory_pool(
            memory_pool_config=self.tp_worker.model_runner.memory_pool_config,  # ③ 复用配置
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
        )
        self.draft_worker.init_hicache_draft_plan()
```

顺序是**强制**的：draft 必须拿到 target 已经算好的 `memory_pool_config`，自己不做 profiling。

### 4.2 draft 权重先记账，再做 profiling

`scheduler.py:995-1007`：

```python
preloaded_weights_bytes = self.tp_worker.preloaded_weights_bytes
if self.draft_worker is not None:
    preloaded_weights_bytes += self.draft_worker.preloaded_weights_bytes   # ← draft 权重计入
self.tp_worker.model_runner.account_preloaded_weights(preloaded_weights_bytes)
self.tp_worker.alloc_memory_pool()
```

也就是说 draft 模型的**权重**在 KV 预算 profiling 之前就已经落地并被扣除。
`kv_cache_configurator.py:2039-2040` 的报错信息还专门提醒了这件事：
*"If using speculative decoding, draft weights are now counted."*

MTP 是个特例 —— draft 权重内嵌在 target checkpoint 里，`set_embed_and_head` 是**引用共享**而非
拷贝，所以 embedding / lm_head 不重复占显存（细节见
[mtp_speculative_decoding_architecture.md](mtp_speculative_decoding_architecture.md) §5）。

### 4.3 关键：draft 的 KV 显存靠"膨胀 cell_size"预留

profiling 出来的是一个字节预算，最后一步是简单除法
（`model_executor/pool_configurator.py:427-436`）：

```python
def calculate_pool_sizes(self, available_bytes: int, page_size: int) -> MemoryPoolConfig:
    max_total_num_tokens = available_bytes // self._cell_size
    max_total_num_tokens = max_total_num_tokens // page_size * page_size
    return MemoryPoolConfig(max_total_num_tokens=max_total_num_tokens)
```

所以给 draft 留显存的唯一手段就是**把 `self._cell_size`（每 token 字节数）抬高**。
这段逻辑在 `DefaultPoolConfigurator.__init__` 里，分成两个流派。

**流派一：EAGLE / EAGLE3 / STANDALONE / FROZEN_KV_MTP —— 按层数比例乘法膨胀**

[pool_configurator.py:183-222](../../../python/sglang/srt/model_executor/pool_configurator.py#L183-L222)：

```python
# EAGLE/STANDALONE: scale cell_size to account for draft model KV cache.
# Assumes draft and target share the same per-layer KV size (head_dim,
# num_kv_heads, dtype), which holds for EAGLE/MTP draft models that
# reuse the target architecture's attention config.
if (kvc.spec_algorithm.is_eagle() or kvc.spec_algorithm.is_standalone()
   ) and not kvc.is_draft_worker:
    eagle_draft_num_layers = kvc.spec_aux_config.eagle_draft_num_layers
    if eagle_draft_num_layers is not None and int(eagle_draft_num_layers) > 0 and int(num_layers) > 0:
        draft_num_layers = int(eagle_draft_num_layers)
        if is_deepseek_dsa(kvc.model_config.hf_config):
            ...  # DSA: KV 与 indexer 分开算，indexer 按 allocate_all_layers=True 全层计
            self._cell_size += draft_kv_size + draft_indexer_size
        else:
            self._cell_size = int(self._cell_size * (1 + draft_num_layers / int(num_layers)))
```

直觉数字：

| 场景 | L (target) | L' (draft) | 膨胀系数 | KV 容量损失 |
|---|---|---|---|---|
| DeepSeek + 1 层 MTP | 61 | 1 | 1.016 | ~1.6% |
| DSV4 + NextN | 43 | 1 | 1.023 | ~2.3% |
| Llama-70B + 2 层 EAGLE3 | 80 | 2 | 1.025 | ~2.5% |
| Qwen-7B + 2 层 EAGLE3 | 28 | 2 | 1.071 | ~7% |
| 小模型 + STANDALONE 小模型 | 32 | 16 | 1.5 | **~33%** |

注意注释里的前提假设："draft 和 target 每层 KV 尺寸相同"。对 EAGLE/MTP 成立（draft 复用 target
的 attention 配置），对 STANDALONE 不一定成立 —— 这是这套公式的已知近似。

**流派二：DFLASH / DSPARK —— 按精确字节数加法**

`pool_configurator.py:224-246`：

```python
# DFLASH/DSPARK: reserve the draft runner's *actual* per-token KV cost.
# ... an MLA-latent target paired with a full per-head K/V draft ...
if kvc.spec_algorithm.is_dflash_family() and not kvc.is_draft_worker:
    self._cell_size = scale_kv_cell_size_per_token_for_dflash(
        target_cell_size_per_token=self._cell_size,
        target_num_layers=int(num_layers),
        draft_num_layers=int(draft_num_layers) * get_parallel().attn_dcp_size,
        draft_cell_size_per_token=_dflash_draft_cell_size(kvc) or None)
```

`speculative/dflash_utils.py:140-192`：

```python
def dflash_draft_cell_size_per_token(*, draft_model_config, draft_num_layers, draft_kv_cache_dtype, tp_size):
    num_kv_heads = draft_model_config.get_num_kv_heads(tp_size)
    kv_dim_per_head = draft_model_config.head_dim + draft_model_config.v_head_dim
    return int(num_kv_heads * kv_dim_per_head * int(draft_num_layers) * dtype_size)
```

为什么 DFLASH 不能用层比例：MLA target 的一层只存压缩 latent，而 DFLASH draft 是 per-head 的
完整 K/V，两者每层字节数差一个数量级，比例法会严重低估。

**NGRAM：不膨胀**。它没有 draft 模型，`cell_size` 原样。

### 4.4 `eagle_draft_num_layers` 是怎么知道的

target 进程为了算这个膨胀系数，**在启动时会额外加载一次 draft 模型的 config**（只读 config，不载
权重）—— `model_executor/model_runner_components/spec_aux_hidden_state.py:80-107`：

```python
if ((spec_algorithm.is_eagle() or spec_algorithm.is_standalone())
    and not is_draft_worker and get_spec().speculative_draft_model_path):
    draft_model_config = ModelConfig.from_server_args(..., is_draft_model=True)
    num_nextn_predict_layers = draft_model_config.num_nextn_predict_layers
    if num_nextn_predict_layers is not None:
        config.eagle_draft_num_layers = int(num_nextn_predict_layers)
    else:
        config.eagle_draft_num_layers = int(max(draft_model_config.num_hidden_layers,
                                                draft_model_config.num_attention_layers))
```

⚠️ 已知坑：EAGLE3 的 config 里如果显式写了 `num_nextn_predict_layers: 0`，draft pool 会被建成
**0 层**，后续 `set_mla_kv_buffer` 抛 IndexError。详见
[mtp_speculative_decoding_architecture.md](mtp_speculative_decoding_architecture.md) §7.2。

### 4.5 draft worker 跳过 profiling

[kv_cache_configurator.py:296-306](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L296-L306)：

```python
def configure(self, *, pre_model_load_memory: int) -> KVCacheConfigResult:
    """Apply a resolved MemoryPoolConfig and initialize pools."""
    if not self.spec_algorithm.is_none() and self.is_draft_worker:
        assert self.memory_pool_config is not None, "Draft worker requires memory_pool_config"
        config = self.memory_pool_config          # ← 直接用 target 算好的
    else:
        config = self._resolve_memory_pool_config(pre_model_load_memory)   # ← target 才 profiling
    sizes = self._derive_pool_sizes(config=config)
    pools = self._init_pools(...)
```

这样保证 draft 池的 `max_total_num_tokens` 与 target **严格相等**，slot 编号才能通用。

### 4.6 draft 侧的四处必要偏差

复用配置不代表完全一致，`_derive_pool_sizes` 对 draft 有四处调整：

| 偏差 | 位置 | 原因 |
|---|---|---|
| **DCP 放大 `loc_space_scale`** | `kv_cache_configurator.py:335-342`、`:353-360` | draft 池是**复制**的而非 DCP 分片，但它直接用 allocator 的虚拟 loc 空间 `[0, max_total × dcp_size)`，所以要乘回 `attn_dcp_size` |
| **Unified memory 按虚拟 id 空间定尺** | `:480-513` | `draft_virtual_id_space = allocator.size_full` 向上页对齐；`UnifiedSWATokenToKVPoolAllocator` 直接抛错 |
| **DSV4 子池清零** | `:362-374` | c4 / c128 / state 三个子池只活在 target rank，draft 一律置 0 |
| **SWA 映射注册** | `:1960-1988` | draft 若满容量 SWA 用恒等映射，否则复用 target 的 `full_to_swa_index_mapping` |

`loc_space_scale` 的注释把道理说得很清楚（`kv_cache_configurator.py:331-338`）：

```python
# Note(kpham-sgl):
# 1. A replicated draft indexes the allocator's virtual locs raw, so its pools
#    span and page that space; the sharded target translates and stays per-rank.
# 2. A pool must page as its allocator does, or its last page falls short.
```

### 4.7 非 KV 的辅助 buffer 清单

这些 buffer **不在** profiling 的 KV 预算内，是额外开销，排查显存时容易漏：

| buffer | 形状 | 位置 |
|---|---|---|
| overlap 中继：`topk_p_buf` / `topk_index_buf` / `hidden_states_buf` / `draft_probs_buf` | `req_pool_size × ...`，懒创建 | `managers/overlap_utils.py:339-365` |
| topk=1 链式 buffer：`_topk1_parents_prealloc` / `_topk1_score_indices_prealloc` | `(max_bs, num_steps)` | `speculative/base_spec_worker.py:116-148` |
| draft-decode CUDA graph 静态区 | `out_cache_loc`: `max_bs × topk × num_steps` | `speculative/eagle_draft_cuda_graph_runner.py:154-238` |
| draft-extend CUDA graph 静态区 | `out_cache_loc` / `hidden_states`: `max_bs × num_draft_tokens` | `speculative/eagle_draft_extend_cuda_graph_runner.py:135-232` |
| target verify graph 静态区 | `out_cache_loc`: `max_bs × num_draft_tokens` | `model_executor/runner/decode_cuda_graph_runner.py:323-426`、`runner_utils/buffers.py:147` |
| 注意力后端索引 buffer | 随 topk 线性放大，如 `req_to_token_pool.size × topk + 1` | `speculative/frozen_kv_mtp_worker_v2.py:262-273` |

DFLASH 在剩余显存 < 1.0 GB 时会直接跳过 draft graph 捕获（`dflash_worker_v2.py:465-492`），
DSPARK 走 TP 同步的空闲显存检查（`dspark_worker_v2.py:387-394`）。

### 4.8 创建阶段公式汇总

```
budget_bytes = (free_GPU − pre_model_load_memory × (1 − mem_fraction_static) − mm_reserve) × 2^30
               ↑ draft 权重此时已经落地并被扣除

cell_size    = target_bytes_per_token
               × (1 + L'/L)                 # EAGLE / EAGLE3 / STANDALONE / FROZEN_KV_MTP
               + draft_bytes_per_token       # DFLASH / DSPARK（精确），或退化为层比例
               (+ DSA indexer 全层字节)
               (NGRAM 不变)

max_total_num_tokens = floor(budget_bytes / cell_size) // page_size × page_size
max_running_requests = f(max_total_num_tokens, context_len, mamba 上限)

draft 池：复用上面这个 MemoryPoolConfig（DCP 下 × attn_dcp_size），自建 KV 张量，
          allocator 与 req_to_token_pool 原样透传。
```

`max_running_requests` 没有 spec 专属项（`kv_cache_configurator.py:2105-2158`），它是从已经被
draft 抬高过的 token 容量推导出来的 —— **开投机解码会传递性地降低并发上限**。
预算不够时的报错来自 `MemoryPoolConfig.__post_init__`（`pool_configurator.py:81-86`）：
*"Not enough memory. Please try to increase --mem-fraction-static."*

---

## 5. 集中式部署的运行期管理

> 本节讲的是"一个 decode step 里 slot 怎么来、怎么用、怎么走"。
> PD 分离的 D 侧运行期与本节**完全相同**，差异只在首步的种子来源（§6.9）。

### 5.1 三个长度字段：理解一切的前提

`managers/schedule_batch.py:816-826` 的 `ReqKvInfo`：

```python
class ReqKvInfo:
    # The request's own KV is [cache_protected_len, kv_allocated_len).
    cache_protected_len: int = 0   # Tree cache owns [0, here)
    kv_committed_len: int = 0      # KV content committed up to here, <= kv_allocated_len
    kv_allocated_len: int = 0
```

```
 0                cache_protected_len      kv_committed_len       kv_allocated_len
 ├────────────────────┼──────────────────────────┼───────────────────────┤
 │  radix tree 拥有   │   本请求已确认的 KV      │   超额预留区（spec 专属）  │
 │  （可被其他请求命中）│   （被接受的 token）     │   放 draft 链 / 被拒草稿   │
```

**最右边那段就是投机解码的全部秘密**：非投机模式下 `kv_committed_len == kv_allocated_len`
恒成立（`mem_cache/common.py:269-272` 有 assert），投机模式下这段永远非空。

### 5.2 一次 decode step 预留多少

[mem_cache/allocation_sizing.py:17-54](../../../python/sglang/srt/mem_cache/allocation_sizing.py#L17-L54)：

```python
def get_alloc_len_per_decode() -> int:
    spec = get_spec()
    if spec.speculative_algorithm is None:
        return 1
    # Spec decoding allocates max(topk * num_steps, num_draft_tokens) per decode step.
    spec_steps = spec.speculative_num_steps or 1
    spec_topk = spec.speculative_eagle_topk or 1
    spec_tokens = max_speculative_num_draft_tokens()
    page_size = get_alloc_page_size()
    spec_algo = SpeculativeAlgorithm.from_string(spec.speculative_algorithm)
    if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        # spec v2 tree (page>1, topk>1): worst-case page-aligned footprint per
        # topk branch is ceil((page_size-1 + num_steps) / page) pages, each branch
        # duplicated -- reserve for all topk branches.
        num_new_pages_per_topk = ((page_size - 1) + spec_steps + page_size - 1) // page_size
        return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)

def get_alloc_reserve_per_decode() -> int:
    """The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap mode"""
    return 2 * get_alloc_len_per_decode()
```

三个要点：

1. **为什么取 max**：draft 阶段要写 `topk × num_steps` 个 slot（树的所有节点），verify 阶段要写
   `num_draft_tokens` 个（树的一次线性化展开）。两个布局落在同一段预留里，取 max 一次覆盖。
2. **为什么 ×2**：overlap 调度下，第 n+1 步的分配发生在第 n 步的 `kv_committed_len` 回填之前，
   双缓冲吸收这一拍滞后。
3. **`page_size > 1 且 topk > 1` 的膨胀**：每个 topk 分支要独占自己的页（§5.5），
   所以预留量是 `ceil((page_size-1+steps)/page) × page × topk`，能比 `steps × topk` 大好几倍。

举例（DSV4 MTP：`steps=3, topk=1, num_draft_tokens=4, page_size=1`）：
`get_alloc_len_per_decode() = max(3, 4) = 4`，`reserve = 8`。

### 5.3 分配路径：整批一次 allocator 调用

调用链：

```
ScheduleBatch.prepare_for_decode            schedule_batch.py:3248
  └─ spec_prepare_for_decode                spec_utils.py:1042      （非 spec 走 alloc_for_decode）
       └─ eagle_prepare_for_decode          eagle_utils.py:928
            ├─ page_aligned_decode_alloc_lens   allocation_sizing.py:57
            └─ alloc_for_spec_decode           allocation.py:665
```

`allocation_sizing.py:57-77` 算出每个请求的目标长度：

```python
for i, r in enumerate(reqs):
    cur = r.kv.kv_allocated_len
    nxt = max(cur,   # ← max 钳制：自适应投机降档时不回缩
        (r.kv.kv_committed_len + reserve + page_size - 1) // page_size * page_size)
    num_needed_tokens += nxt - cur
```

`mem_cache/allocation.py:665-712` 整批一次分配：

```python
if num_needed_tokens > 0:
    if tree_cache.token_to_kv_pool_allocator.page_size == 1:
        out_cache_loc = alloc_token_slots(tree_cache, num_needed_tokens)
    else:
        last_loc = get_last_loc(req_to_token_pool.req_to_token, req_pool_indices, cur_kv_lens)
        out_cache_loc = ALLOC_EXTEND_FUNCS[device_type](...)
    # Updating req_to_token is a write to a shared tensor: it must not overlap
    # with the previous batch's forward, which also reads req_to_token.
    assign_req_to_token_pool_func(req_pool_indices, req_to_token_pool.req_to_token,
                                  cur_kv_lens, nxt_kv_lens, out_cache_loc, len(reqs))
for req, nxt_kv_len in zip(reqs, nxt_kv_lens_list, strict=True):
    req.kv.kv_allocated_len = max(req.kv.kv_allocated_len, nxt_kv_len)
```

⚠️ **这一步只推进 `kv_allocated_len`，不设 `batch.out_cache_loc`。** slot 被写进
`req_to_token[req_idx, cur:nxt]`，各阶段的 `out_cache_loc` 视图后面再从 `req_to_token` **gather**
出来。这是很容易看错的地方。

`eagle_utils.py:957-960` 还有一道越界断言（PR #26972）：

```python
if page_size > 1 and (get_spec().speculative_eagle_topk or 1) > 1:
    max_alloc_len = int(nxt_kv_lens_cpu.max())
    row_width = batch.req_to_token_pool.req_to_token.shape[1]
    assert max_alloc_len <= row_width, (...)
```

对应的行宽预留在 `allocation_sizing.py:80-96`：

```python
extra = 4 + (max_speculative_num_draft_tokens() or 0)
if get_spec().speculative_algorithm is not None and page_size > 1:
    # ... without the headroom the row write silently lands in the neighbor row.
    extra = max(extra, get_alloc_reserve_per_decode() + page_size - 1)
```

### 5.4 draft 阶段：怎么用这段预留（写 draft 池）

`speculative/eagle_worker_common.py:212-303` 的 `prepare_for_draft`，路径 A（`page_size == 1`
或 `topk == 1`，即连续布局）：

```python
if page_size == 1 or topk == 1:
    batch.out_cache_loc = torch.empty((bs * topk * num_steps,), dtype=torch.int64, device=batch.device)
    assign_draft_cache_locs_contiguous[(bs,)](
        batch.req_pool_indices, req_to_token_pool.req_to_token,
        batch.seq_lens, batch.out_cache_loc,
        req_to_token_pool.req_to_token.shape[1], topk, num_steps)
draft_input.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
```

这个扁平 buffer 按步切片给每一轮 draft 前向用（`eagle_utils.py:64-81`）：

```python
def per_step_draft_out_cache_loc(out_cache_loc, batch_size, topk, num_steps):
    expected = batch_size * topk * num_steps
    assert out_cache_loc.shape[0] == expected, ...
    return out_cache_loc.view(batch_size, topk, num_steps).permute(2, 0, 1).reshape(num_steps, -1)
```

`speculative/eagle_worker_v2.py:573-666` 的 draft 主循环：

```python
out_cache_loc = per_step_draft_out_cache_loc(out_cache_loc, forward_batch.batch_size, self.topk, self.speculative_num_steps)
for i in range(self.speculative_num_steps):
    ...
    if i == self.speculative_num_steps - 1:
        break                             # ← 最后一步只采样，不再写 KV
    forward_batch.out_cache_loc = out_cache_loc[i]
    ... attn_backend=self.draft_attn_backend.attn_backends[i]
```

注意最后一步 `break`：第 `num_steps-1` 步的 slot 被预留了但**永远不写**，这是预留量是上界而非
精确值的原因之一。

### 5.5 `page_size > 1 且 topk > 1`：每分支页对齐 + 前缀尾巴复制

分页 allocator 下，树的每个分支必须落在自己的页里（否则同一页会被多个分支交叉写）。
`eagle_worker_common.py:254-284`：

```python
# page_size > 1 + topk > 1: per-branch page-aligned draft pages.
rows = req_to_token_pool.req_to_token[batch.req_pool_indices.long()]
last_page = seq_lens % page_size
prefix_base = seq_lens - last_page
num_new_pages = (last_page + num_steps + page_size - 1) // page_size
starts = (prefix_base.view(bs,1) + topk_ids * (num_new_pages.view(bs,1) * page_size) + last_page.view(bs,1))
pos = (starts.view(bs, topk, 1) + steps).reshape(bs, topk * num_steps)
batch.out_cache_loc = torch.gather(rows, 1, pos).reshape(-1).contiguous()
duplicate_prefix_tail_to_draft_branches(...)
```

每个分支从自己的页起，但序列的最后一页是**半满**的，所以每个分支的首页空洞必须先把公共前缀的
尾巴复制进去，否则该分支的注意力读不到前缀 —— `eagle_worker_common.py:58-102`，末尾是
`token_to_kv_pool.move_kv_cache(tgt_slots, src_slots)`。

这就是 §5.2 里 `num_new_pages_per_topk × page_size × topk` 那个膨胀预留的用途，也是为什么
`topk > 1 且 page_size > 1` 时注意力后端被限制为 `{flashinfer, fa3, triton}`
（`arg_groups/speculative_hook.py:392-404`）。

### 5.6 verify 阶段：同一段预留的另一个视图（写 target 池）

`eagle_utils.py:511-563` 的 `eagle_prepare_for_verify`：

```python
batch.input_ids = verify_input.draft_token
batch.out_cache_loc = assign_extend_cache_locs_uniform_func(
    req_pool_indices=batch.req_pool_indices,
    req_to_token=req_to_token_pool.req_to_token,
    start_offset=batch.seq_lens,
    draft_token_num=verify_input.draft_token_num, ...)
batch.forward_mode = ForwardMode.IDLE if ... else ForwardMode.TARGET_VERIFY
```

从 `seq_lens` 开始取 `bs × draft_token_num` 个 slot —— 与 §5.4 的 draft 视图**是同一段预留的不同
子区间**，写的是 target 池。

### 5.7 接受之后：把接受的 KV **搬**到连续前部（仅 topk > 1）

树形 draft 的接受路径是树上的一条链，物理上不连续；而后续步骤要求 KV 在
`[0, kv_committed_len)` 连续。所以要搬一次 —— `speculative/spec_utils.py:704-764`
`move_accept_tokens_to_target_kvcache`：

```python
size = bs * accept_index.shape[1]   # 注意：不是 bs * num_draft_tokens
tgt_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
accept_out_cache_loc = torch.zeros(size, dtype=torch.int64, device=device)
assign_extend_cache_locs[(bs,)](
    batch.req_pool_indices, batch.req_to_token_pool.req_to_token,
    batch.seq_lens, batch.seq_lens + num_correct_drafts + 1,      # ← +1 是 bonus token
    tgt_cache_loc, batch.req_to_token_pool.req_to_token.shape[1], next_power_of_2(bs))
fill_accept_out_cache_loc_func(accept_index, batch.out_cache_loc, accept_out_cache_loc, size)
token_to_kv_pool_allocator.get_kvcache().move_kv_cache(tgt_cache_loc, accept_out_cache_loc)
```

驱动点 `speculative/eagle_worker_common.py:406-425`，门禁在 `:629`：
`if finalize_tree_path and not idle and topk > 1`。

**`topk == 1` 时不搬**：链式 draft 的接受结果本来就是连续前缀，
`_compact_accept_to_front`（`:438-450`）的注释写得很清楚：
*"trailing unaccepted slots stay and are freed as overshoot"*。

### 5.8 提交：`kv_committed_len` 前进

`managers/scheduler_components/batch_result_processor.py:748-761`：

```python
if req.is_retracted or req.finished(): pass  # no worker pre-claims the bonus
...
num_accept_tokens = len(accept_tokens)
req.kv.kv_committed_len += num_accept_tokens   # ← 含 bonus token（命名规范 Rule 3）
req.spec_verify_ct += 1
```

### 5.9 释放：**没有**逐 token free 被拒草稿

这是本文最想强调的一点。全仓库找不到"遍历 draft 树把被拒节点的 slot 还给 allocator"的代码。
被拒草稿的 slot 靠两条机制回收：

**(a) 插入 radix tree 时以 `kv_committed_len` 为上界截断**

`mem_cache/common.py:215-257` 的 `release_kv_cache`：

```python
effective_kv_committed_len = req.effective_kv_committed_len()
tree_cache.cache_finished_req(req,
    is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
    kv_len_to_handle=effective_kv_committed_len)      # ← 超额区结构性地被排除在树外
...
start_p, end_p = effective_kv_committed_len, req.kv.kv_allocated_len
_release_overallocated_kv_indices(req, start_p, end_p, tree_cache)
```

**(b) 超额区整段 free**

`mem_cache/common.py:260-283`：

```python
def _release_overallocated_kv_indices(req, start_p, end_p, tree_cache) -> None:
    allocator = tree_cache.token_to_kv_pool_allocator
    page_size = allocator.page_size
    spec_algo = get_spec().speculative_algorithm
    # strip_thinking_cache intentionally reports output tokens as overallocated
    # so they fall into the free path below (#22373).
    if spec_algo is None and not get_serving().strip_thinking_cache:
        assert start_p == end_p, f"Unexpected overallocated KV cache, ..."   # ← 非 spec 才断言相等
    if page_size > 1:
        start_p = ceil_align(start_p, page_size)     # 向上页对齐，不释放与已提交内容共页的部分
    if start_p < end_p:
        indices_to_free = tree_cache.req_to_token_pool.req_to_token[req.kv.req_pool_idx][start_p:end_p]
        allocator.free_segment(indices_to_free, start_pos=start_p)
```

那句 `if spec_algo is None` 的断言正是"投机解码下 committed 与 allocated 之间必然有 gap"这个
事实的代码化表达。

另外确认两件事：
- `EagleDraftInput.filter_batch` / `merge_batch`（`speculative/eagle_info.py:208`、`:227`）
  只切张量，**不碰 allocator**。
- `ScheduleBatch.filter_batch`（`schedule_batch.py:3338-3423`）把 `out_cache_loc` 置 None 后
  委托给 `spec_info.filter_batch`，同样不释放 KV。

### 5.10 与 RadixCache 的交互：bigram key

radix cache 对投机解码的唯一感知是 **bigram 键视图**。EAGLE draft 在 slot *i* 存的是"错位一位"的
token 对的隐状态，所以命中的前缀要按 bigram 序列匹配才对齐。

`mem_cache/radix_cache.py:81`、`:156-164`、`:459-482`：

```python
# bigram view over token_ids: length = max(0, len(token_ids) - 1)
def maybe_to_bigram_view(self, is_eagle, value=None):
    if is_eagle and not self.is_bigram:
        self.is_bigram = True
...
def cache_finished_req(self, req, is_insert=True, *, kv_len_to_handle: int):
    token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
    radix_key = RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle, ...).page_aligned(self.page_size)
```

生产路径 `UnifiedRadixCache` 同构（`mem_cache/unified_radix_cache.py:838-905`）。
`is_eagle` 的注入点在 `mem_cache/kv_cache_builder.py:302`：`is_eagle=spec_algorithm.is_eagle()`。

**集中式模式下没有任何开关会因为投机解码而禁用 radix cache。** 但 PD 分离的 D 侧相反 —— 见 §6.11。

### 5.11 retract / abort：容量估算是 spec-aware 的，释放是通用的

容量估算：`schedule_batch.py:2952-2979`

```python
def _new_tokens_required_next_decode_spec_v2(...):
    reserve = get_alloc_reserve_per_decode()
    for r in requests:
        x = max(0, r.kv.kv_committed_len + reserve - r.kv.kv_allocated_len)
        cur = r.kv.kv_allocated_len; nxt = cur + x
        total += ceil_align(nxt, page_size) - ceil_align(cur, page_size)
```

它喂给 `check_decode_mem`（`:2981`）。而 `retract_decode`（`:2992-3069`）**没有任何 spec 专属分支**：
排序 → `release_req` → 最后兜底的请求标 `FINISH_ABORT` → `filter_batch`。

`release_req`（`schedule_batch.py:1997-2030`）走 `is_insert=False`，
`unified_radix_cache.py:906-910` 整段 free `kv_indices[cache_protected_len:]`，
再由 `_release_overallocated_kv_indices` 释放 `[ceil_align(committed), allocated)`。
**所有未用的 draft slot 与已提交 KV 在同一对 `free_segment` 调用里回去，没有任何代码遍历 draft 树。**

状态复位 `reset_for_retract`（`:1731-1765`）里有一条关键断言：

```python
assert not self.kv.holds_kv, "expect it is already released"
self.kv.kv_committed_len = 0
```

`kv_allocated_len` 由 `ReqKvInfo.mark_kv_released`（`:867-869`）清零。

verify 途中被 abort 的请求，在提交阶段被跳过（`batch_result_processor.py:748`），
`kv_committed_len` 不前进，因此它的**全部**预留都落在超额区，随后被整段回收。

### 5.12 集中式生命周期总表

| 阶段 | 入口 | 动的是什么 |
|---|---|---|
| 预留 | `eagle_prepare_for_decode` → `alloc_for_spec_decode` | `kv_allocated_len` → `ceil_page(kv_committed_len + 2·max(steps·topk, num_draft_tokens))`；slot 写入 `req_to_token` |
| draft 写 | `prepare_for_draft` + `draft_forward` | gather `bs·topk·num_steps` 视图，按步切片，写 **draft** 池 |
| verify 写 | `eagle_prepare_for_verify` | gather `bs·draft_token_num` 视图（从 `seq_lens` 起），写 **target** 池 |
| 搬运 | `move_accept_tokens_to_target_kvcache`（topk>1） | 接受链的 KV **拷贝**到连续前部 |
| 提交 | `batch_result_processor.py:760` | `kv_committed_len += num_accept_tokens` |
| 释放 | `release_kv_cache` | 请求结束/retract 时，`[committed, allocated)` 整段 free |

---

## 6. PD 分离部署：注册、传输、种子

### 6.1 三条主结论

1. **draft KV 是从 P 搬到 D 的，不在 D 侧重算。** D 侧对迁移过来的前缀不跑 prefill，也不跑
   `draft_extend`。
2. **P 实例照常跑 draft 模型。** EAGLE 系 worker 里根本没有 `disaggregation_mode == "prefill"`
   的分支（唯一早退是"本 PP rank 不持有 draft"）。
3. **投机的"种子"走 aux（metadata）通道，不走 KV 通道。**
   `topk_p / topk_index / hidden_states / dsa_topk_indices` 四个字段。

```
   ┌──────────────── P 实例 ────────────────┐            ┌──────────── D 实例 ────────────┐
   │ target extend (CaptureHiddenMode.FULL) │            │                                │
   │              ↓                         │            │                                │
   │ _draft_extend_for_prefill()            │            │                                │
   │   → 写满 draft KV                      │            │                                │
   │   → 产出 topk_p/topk_index/hidden      │            │                                │
   │              ↓                         │            │                                │
   │ KV 通道：send_kv_chunk(page_indices)   │ ═══RDMA══▶ │ target KV pool + draft KV pool │
   │   （同一份页号覆盖 target + draft）    │            │      （同一批 slot 编号）      │
   │              ↓                         │            │              ↓                 │
   │ aux 通道：set_buf(req) → send_aux      │ ═══RDMA══▶ │ MetadataBuffers → req.*        │
   └────────────────────────────────────────┘            │              ↓                 │
                                                         │ ForwardMode.PREBUILT           │
                                                         │ process_prebuilt()             │
                                                         │  → build_eagle_disagg_draft_   │
                                                         │      input() 拼出首个           │
                                                         │      EagleDraftInput           │
                                                         │              ↓                 │
                                                         │ 直接进 draft → verify 循环     │
                                                         └────────────────────────────────┘
```

### 6.2 为什么 draft KV 能"顺便"传过去

因为 §3.2 那个地基：**draft 池与 target 池共享 slot 编号**。于是：

- P 侧只要把 draft 池的 buffer 指针**追加**到 `kv_args.kv_data_ptrs` 尾部；
- `send_kv_chunk` 每段只需要一份 `page_indices`，它对 target entry 和 draft entry 同样有效；
- D 侧按同样顺序注册自己的 buffer，单边 RDMA WRITE 就落到正确位置。

代码里两处**一字不差**的注释（`disaggregation/prefill.py:246-247` 与
`disaggregation/decode.py:543-544`）：

```python
# We should also transfer draft model kv cache. The indices are
# always shared with a target model.
```

### 6.3 P 侧注册

[disaggregation/prefill.py:243-264](../../../python/sglang/srt/disaggregation/prefill.py#L243-L264)：

```python
draft_kv_pool = self.draft_token_to_kv_pool if transfer_draft_cache else None
num_draft_entries = 0
if draft_kv_pool is not None:
    draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = draft_kv_pool.get_contiguous_buf_infos()
    kv_data_ptrs += draft_kv_data_ptrs      # ← 追加，不是插入
    kv_data_lens += draft_kv_data_lens
    kv_item_lens += draft_kv_item_lens
    num_draft_entries = len(draft_kv_data_ptrs)

kv_args.kv_data_ptrs = kv_data_ptrs
kv_args.kv_layer_ids = build_kv_layer_ids(
    token_to_kv_pool=self.token_to_kv_pool,
    draft_token_to_kv_pool=draft_kv_pool,
    num_draft_entries=num_draft_entries,
    num_hidden_layers=self.scheduler.model_config.num_hidden_layers)
```

`transfer_draft_cache` 决定 layer-shard 场景下由谁传（`prefill.py:213-220`）：

```python
transfer_draft_cache = (not layer_shard_enabled or layer_shard_rank == layer_shard_size - 1)
```

即**只有最后一个 shard rank 注册 draft buffer**，否则每个 shard 都会重复搬同一份 draft KV。

draft 池的来源是 `scheduler.py:1378-1382` → `base_spec_worker.py:176-178` 的
`primary_draft_kv_pool`。对 NGRAM / FROZEN_KV_MTP 它返回 `None`，所以这两个算法在 PD 下不注册
任何 draft entry。

### 6.4 D 侧注册（镜像）

`disaggregation/decode.py:541-571`：

```python
num_draft_entries = 0
if self.draft_token_to_kv_pool is not None:
    draft_kv_data_ptrs, draft_kv_data_lens, draft_kv_item_lens = self.draft_token_to_kv_pool.get_contiguous_buf_infos()
    kv_data_ptrs += draft_kv_data_ptrs
    kv_data_mem_kinds += ["VRAM"] * len(draft_kv_data_ptrs)   # ← D 侧多一个 mem kind 标记
    num_draft_entries = len(draft_kv_data_ptrs)
```

pool 的传入点：`DecodePreallocQueue.__init__`（`decode.py:325`、`:345`）、
`PrefillBootstrapQueue.__init__`（`prefill.py:140`、`:155`），都来自 `scheduler.py:1440` / `:1474`。

### 6.5 layer id 保留段：追加带来的连带问题

"追加"这个动作会破坏下游一切**位置假设**：MHA 后端假定 `kv_data_ptrs` 是扁平的
`[K 全部层, V 全部层]` 两半，PP 又要按层切片。draft 自己的层号从 0 开始，会和 target 撞车。

解法是把 draft 的层号**重映射到 target 层号之上的一段保留区** ——
`disaggregation/utils.py:918-951`：

```python
def build_kv_layer_ids(*, token_to_kv_pool, draft_token_to_kv_pool, num_draft_entries, num_hidden_layers):
    """Global layer id for every entry in ``kv_args.kv_data_ptrs``.

    Draft KV buffers are appended after the target's, so they need ids of their
    own: ... The draft numbers its layers from zero, which would collide with
    the target's, so its entries are remapped into a reserved band above the
    target's layer range. Both PD peers run this against the same draft config
    and so agree on the band.
    """
    if not isinstance(token_to_kv_pool, HybridLinearKVPool):
        return []                                   # ← ⚠️ 只有 HybridLinearKVPool 走显式层号
    layer_ids = token_to_kv_pool.get_kv_layer_ids()
    if draft_token_to_kv_pool is None:
        return layer_ids
    draft_ids = _draft_entry_layer_ids(pool=draft_token_to_kv_pool, num_entries=num_draft_entries)
    band_index = {lid: i for i, lid in enumerate(dict.fromkeys(draft_ids))}
    return layer_ids + [num_hidden_layers + band_index[lid] for lid in draft_ids]
```

注意 **line 939 的早退**：非 `HybridLinearKVPool` 一律返回 `[]`，此时后端退化为"按位置配对"
（§6.7）。所以这个保留段的正确性论证只覆盖 hybrid-linear 模型。

staging buffer 路径还有第三种顺序，`utils.py:1001-1044` 的注释直说了这件事：

> "The gather writes every k_buffer and then every v_buffer, while `kv_layer_ids` follows
> `kv_data_ptrs` (`[K target, V target, K draft, V draft]`), so the two orders diverge as soon as
> a draft pool is registered."

### 6.6 state（非 KV）组件的 draft 对应物

`setup_state_kv_args`（`disaggregation/utils.py:1070-1345`）处理 `StateType` 系列
（MAMBA / SWA / SWA_RING / DSA / MINIMAX_INDEX_K / C128_STATE / BLOCK_SCALE / BLOCK_SCALE_SWA）：

| 组件 | draft 怎么处理 | 位置 |
|---|---|---|
| DSA indexer 状态 | draft 的 `get_state_buf_infos()` **拼进同一个** DSA 组件 | `utils.py:1206-1226` |
| DSV4 NextN | 注册成**独立的** SWA / SWA_RING 组件，并做严格几何校验 | `utils.py:1245-1312` |

DSV4 那几条校验值得记住（同 §6.4 一起看）：
"DSV4 draft state transfer expects SWA-only NextN layers"、
"target and draft pools must use the same unified-KV mode"、
"must share SWA ring geometry"、"must share the SWA index mapping"、
"must share paged SWA geometry"。开投机时 ring 几何还会从 8/128 抬到 16/256
（细节见 [dsv4_pd_p2d_transfer_contents_and_method.md](../04_pd_disaggregation/dsv4_pd_p2d_transfer_contents_and_method.md) §12）。

### 6.7 后端怎么消费这份变长的 entry 列表

**Mooncake**（`disaggregation/mooncake/conn.py:645-764`）：有层号就按层号配对，没有就退回半分。

```python
has_layer_ids = bool(src_layer_ids or dst_layer_ids)
if has_layer_ids:
    pairs = build_transfer_entry_pairs(src_layer_ids, dst_layer_ids,
        len(src_data_ptrs), len(dst_data_ptrs),
        allow_positional_fallback=self.pp_size == 1)
    layers_params = [(src_data_ptrs[i], dst_data_ptrs[j], item_lens[i]) for i, j in pairs]
else:
    ...get_mla_kv_ptrs_with_pp(...) / get_mha_kv_ptrs_with_pp(...)
```

`mooncake/conn.py:1080-1081` 的注释：
*"Draft buffers break the flat `[K block, V block]` layout, so pair by layer ID instead of the
half-split used by `get_mha_kv_ptrs_with_pp`."*

**非对称部署（P 不开 spec，D 开 spec）是被显式支持的**，三处独立处理：

`disaggregation/common/conn.py:908-939`：

```python
elif (num_kv_layers < dst_num_total_layers and dst_num_total_layers % num_kv_layers != 0):
    # Case: Decode has draft model KV while Prefill is deployed without speculative decoding
    # dst_kv_ptrs layout: [K_main..., V_main..., draft_K..., draft_V...]
    multiplier_ratio = dst_num_total_layers // num_kv_layers
    dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
    v_ptr_offset = num_kv_layers * multiplier_ratio
    dst_v_ptrs = dst_kv_ptrs[v_ptr_offset + start_layer : v_ptr_offset + end_layer]
```

`disaggregation/utils.py:976-998` 的 `resolve_dcp_dst_entry_indices`：
*"n_dst may exceed n_src when the decode side runs speculative decoding and the prefill side does not."*

`disaggregation/nixl/conn.py:958-1030`：

```python
# If prefill does not run speculative decoding (the usual case),
# decode with speculative decoding will have more kv items.
# Prefill having more kv items is impossible.
n_src = len(self.kv_args.kv_item_lens)
n_dst = len(peer_info.dst_kv_item_lens)
if n_dst < n_src:
    raise ValueError("NIXL PD transfer: decode registered fewer KV regions ...")
decode_only_spec_dec = n_dst > n_src
...
if decode_only_spec_dec:
    raise NotImplementedError("NIXL PD transfer does not support HiSparse combined with "
        "decode-only speculative decoding.")
if decode_only_spec_dec and dst_mem_kind != "VRAM":
    raise NotImplementedError(...)
```

反过来 **P 开 spec、D 不开 spec 是不允许的**（`n_dst < n_src` 直接报错）。

### 6.8 aux 通道：四个投机字段

`disaggregation/utils.py:297-385` 的 `MetadataBuffers`：

```python
# For PD + spec decode
self.output_topk_p = torch.zeros((size, 16), dtype=torch.float32, device=device)
self.output_topk_index = torch.zeros((size, 16), dtype=torch.int64, device=device)
self.output_hidden_states = torch.zeros((size, hidden_size), dtype=hidden_states_dtype, device=device)
if self.output_dsa_topk_indices_dim > 0:
    self.output_dsa_topk_indices = torch.full((size, self.output_dsa_topk_indices_dim), -1,
                                              dtype=torch.int32, device=device)
else:
    self.output_dsa_topk_indices = None
self.bootstrap_room = torch.zeros((size, 8), dtype=bootstrap_room_dtype, device=device)
```

⚠️ 那个 **16 是硬编码的**，约束只写在 `utils.py:545` 的注释里：
*"speculative_eagle_topk should not be greater than 16 currently"*。

**wire schema 必须两侧一致，所以形状全部从 config 推导，绝不看本地运行时状态** ——
[scheduler.py:1384-1406](../../../python/sglang/srt/managers/scheduler.py#L1384-L1406)：

```python
if self.spec_algorithm.carries_draft_hidden_states():
    # Derive the rank-uniform PD wire schema from config because only the
    # last prefill PP stage owns a draft runner.
    draft_model_config = ModelConfig.from_server_args(..., is_draft_model=True)
    disagg_hidden_size, disagg_hidden_states_dtype = get_draft_recurrent_hidden_state_spec_from_config(...)
else:
    disagg_hidden_size = 16  # minimal padding size for RDMA
    disagg_hidden_states_dtype = torch.float32
# The PD metadata wire schema must match on P and D even when only D
# enables spec decoding; a seedless prefill writes the invalid sentinel.
output_dsa_topk_indices_dim = get_dsa_seed_metadata_dim(self.model_config.hf_config)
```

`carries_draft_hidden_states()` 只对 EAGLE 系为真（`spec_info.py:165-168`）：
*"STANDALONE's vanilla draft ignores them."*

### 6.9 P 侧写入与 D 侧读取

**P 侧捕获**（`disaggregation/prefill.py:684-797`，`process_batch_result_disagg_prefill`）：

```python
assert batch.spec_info is result.next_draft_input
draft_input = result.next_draft_input
if self.spec_algorithm.is_eagle() and draft_input is not None:
    draft_hidden_states_cpu = draft_input.hidden_states.to("cpu", non_blocking=False)   # 整批一次同步
    if batch.spec_info.dsa_topk_indices is not None:
        draft_dsa_topk_indices_cpu = batch.spec_info.dsa_topk_indices.to("cpu", non_blocking=False)
...
for i, req in ...:
    if self.spec_algorithm.is_eagle() and draft_input is not None:
        req.output_topk_p = draft_input.topk_p[i]
        req.output_topk_index = draft_input.topk_index[i]
        req.hidden_states_tensor = draft_hidden_states_cpu[i].clone()
        ...
    else:
        req.hidden_states_tensor = None
        req.output_dsa_topk_indices = None
    ...
    self.send_kv_chunk(req, last_chunk=True)
```

D→H 拷贝是 `non_blocking=False`，但**整批只做一次**，不是每请求一次。

`Req` 上的载体字段（`managers/schedule_batch.py:1132-1135`）：

```python
self.hidden_states_tensor = None  # Note: use tensor instead of list to transfer hidden_states when PD + MTP
self.output_topk_p = None
self.output_topk_index = None
self.output_dsa_topk_indices = None
```

**写入时机**：只在最后一个 chunk（`prefill.py:1204-1205` 的 `if last_chunk:
self.disagg_metadata_buffers.set_buf(req)`）。`set_buf` 本体在 `utils.py:543-565`：

```python
if req.hidden_states_tensor is not None:
    topk = req.output_topk_p.size(0)          # speculative_eagle_topk should not be greater than 16
    self.output_topk_p[req.metadata_buffer_index, :topk].copy_(req.output_topk_p)
    self.output_topk_index[req.metadata_buffer_index, :topk].copy_(req.output_topk_index)
    self.output_hidden_states[req.metadata_buffer_index].copy_(req.hidden_states_tensor)
    if self.output_dsa_topk_indices is not None:
        if dsa_topk_indices is not None:
            self.output_dsa_topk_indices[req.metadata_buffer_index].copy_(dsa_topk_indices)
        else:
            self.output_dsa_topk_indices[req.metadata_buffer_index].fill_(-1)   # ← 无效哨兵
```

**顺序保证**：`utils.py:180-196` 的 `_apply_metadata_gate` 会把 `Success` 降级回 `Transferring`，
直到 `metadata_buffers.bootstrap_room[idx, 0] != 0`。没有这道闸，D 可能读到全零 hidden_states 并
静默产出垃圾 draft。

**D 侧读取**（`disaggregation/decode.py:2059-2173`，`_commit_transfer_to_req`）：

```python
if not self.spec_algorithm.is_none():
    decode_req.req.output_topk_p = output_topk_p
    decode_req.req.output_topk_index = output_topk_index
    decode_req.req.hidden_states_tensor = output_hidden_states
    if (output_dsa_topk_indices is not None and torch.all(output_dsa_topk_indices < 0).item()):
        output_dsa_topk_indices = None          # ← 哨兵解码：全负 = P 没送种子
    decode_req.req.output_dsa_topk_indices = output_dsa_topk_indices
```

`req.hidden_states_tensor` 还有一处清理：`optimistic_release_and_requeue`
（`prefill.py:1352-1365`）会把它和 `output_dsa_topk_indices` 置 None，避免被重新入队的请求带着
过期种子再发一遍。

### 6.10 P 侧确实跑 draft，D 侧首步靠种子

**P 侧**（`speculative/eagle_worker_v2.py:1163-1208`）：

```python
# A rank that does not host the draft (prefill-side PP builds it only on
# the last stage) forwards the target's proxy tensors and stops here.
if self._draft_worker is None:
    return batch_output
# Draft prefill
batch_output.next_draft_input = self.draft_worker._draft_extend_for_prefill(
    batch, batch_output.logits_output.hidden_states,
    batch_output.next_token_ids, batch_output.logits_output.mm_input_embeds)
```

唯一早退是"本 rank 不持有 draft"。`_draft_extend_for_prefill`（`:780-892`）的 docstring：
*"Run draft model extend to correctly fill the KV cache."* —— 它同时干两件事：**把 draft KV 写满**，
以及**产出种子**。

**唯一按 PD 模式分支的算法是 DSPARK**（`speculative/dspark_components/dspark_worker_v2.py`）：

```python
self._is_pd_prefill = get_disagg().disaggregation_mode == "prefill"
self._decode_graph_allowed = (... and not self._is_pd_prefill)      # :117-121
...
if self._is_pd_prefill and not self._draft_is_moe:
    self.draft_model.prune_to_ctx_kv_injection()                    # :336-337
```

DSPARK 的 P 实例只需要 draft 写 KV、不需要它采样，所以把 draft 模型裁到只剩 KV 注入路径，
并完全跳过 decode CUDA graph。

**D 侧首步**：`ForwardMode.PREBUILT` 假 prefill，`disaggregation/decode.py:2647-2663`：

```python
new_batch = ScheduleBatch.init_new(can_run_list, self.req_to_token_pool, ...)
# construct fake completed prefill
new_batch.prepare_for_prebuilt()
if self.enable_overlap:
    # A finished request can still have one redundant forward in flight.
    # Drain it before a prebuilt request seeds a potentially reused row.
    self.schedule_stream.wait_stream(self.forward_stream)
new_batch.process_prebuilt(self.future_map)
```

`disaggregation/decode_schedule_batch_mixin.py:23-111` 的注释点题：
**"PREBUILT never enters a model forward"** —— `out_cache_loc` 从 `req_to_token` 里取现成的，
`input_ids` 直接置 None。

`process_prebuilt`（`:113-158`）做种子注入：

```python
for req in self.reqs:
    last_tokens.append(req.output_ids[-1])
    maybe_cache_unfinished_req(req, self.tree_cache)
last_tokens_tensor = torch.tensor(last_tokens, dtype=torch.int64, device=self.device)
spec_info = self.spec_algorithm.build_disagg_draft_input(self, last_tokens_tensor, future_map)
if spec_info is not None:
    self.spec_info = spec_info
else:
    future_map.stash(self.req_pool_indices, RelayPayload(bonus_tokens=last_tokens_tensor))
    self.input_ids = None
```

`speculative/eagle_disaggregation.py:20-109` 把四个 CPU 张量拼回 `EagleDraftInput`：

```python
num_states = spec.speculative_eagle_topk
if spec.enable_multi_layer_eagle:
    num_states *= spec.speculative_num_steps
topk_p     = torch.stack([torch.as_tensor(req.output_topk_p[:num_states], ...) for req in batch.reqs], dim=0)
topk_index = torch.stack([torch.as_tensor(req.output_topk_index[:num_states], ...) for req in batch.reqs], dim=0)
hidden_states = torch.stack([req.hidden_states_tensor for req in batch.reqs], dim=0).to(batch.device)
...
spec_info = EagleDraftInput(topk_p=topk_p, topk_index=topk_index, hidden_states=hidden_states,
                            bonus_tokens=last_tokens_tensor,        # ← P 侧采的首 token 当树根
                            dsa_topk_indices=dsa_topk_indices)
spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
```

DSA 种子还需要一次坐标系转换（`:63-88`）：P 送的是**请求相对位置**，D 侧融合 TopK kernel 吃的是
**allocator 本地物理 slot**，所以在进 draft 循环/graph 之前 remap 一次：

```python
if should_remap_pd_dsa_seed_to_local_slots():
    # PD sends request-relative positions; fused TopK consumes
    # decode-local physical slots. Remap once before the draft loop/graph.
    req_to_token = batch.req_to_token_pool.req_to_token
    ...  # clamp → gather → local_slots，无效行掩成 -1
if torch.any(torch.all(dsa_topk_indices < 0, dim=1)).item():
    dsa_topk_indices = None
```

门禁 `layers/attention/dsa/utils.py:78-86`：

```python
def should_remap_pd_dsa_seed_to_local_slots() -> bool:
    """Whether a PD seed should enter the allocator-local fused TopK domain."""
    return ((is_cuda() or is_hip()) and envs.SGLANG_DSA_FUSE_TOPK.get()
        and get_disagg().disaggregation_mode == "decode"
        and not get_memory().enable_hisparse and not get_parallel().dcp_enabled)
```

### 6.11 参数门禁清单

最重要的一条 —— **D 侧 radix cache 与投机解码互斥**
（[arg_groups/pd_disaggregation_hook.py:77-94](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L77-L94)）：

```python
if cfg.disaggregation_mode == "decode":
    if cfg.disaggregation_decode_enable_radix_cache:
        if cfg.enable_hisparse:
            raise ValueError("--disaggregation-decode-enable-radix-cache is incompatible with --enable-hisparse")
        if cfg.disaggregation_transfer_backend == "fake":
            raise ValueError("... incompatible with --disaggregation-transfer-backend fake")
        if cfg.speculative_algorithm is not None:
            raise ValueError("--disaggregation-decode-enable-radix-cache is incompatible "
                "with speculative decoding "
                f"(--speculative-algorithm {cfg.speculative_algorithm})")
```

也就是说：**PD + 投机解码时，D 侧只能用 chunk cache**（`disable_radix_cache=True`，
`pd_disaggregation_hook.py:107-113`）。相关背景见
[pd_decode_radix_cache_hicache_scheme.md](../04_pd_disaggregation/pd_decode_radix_cache_hicache_scheme.md)。

其余门禁：

| 约束 | 位置 |
|---|---|
| PD decode + DCP 要求 backend ∈ {mooncake, nixl, fake}，且禁 decode radix cache / hierarchical cache | `pd_disaggregation_hook.py:53-75` |
| P 侧禁 `fake` backend | `pd_disaggregation_hook.py:132-134` |
| `SGLANG_DISAGG_STAGING_BUFFER` 要求 mooncake / nixl | `pd_disaggregation_hook.py:139-148` |
| P 模式禁 DECODE 相位 graph；D 模式禁 PREFILL 相位 graph | `arg_groups/cuda_graph_hook.py:569-588`；`:226` 把 PD 登记为禁捕获原因 |
| `enable_unified_memory` 与 PD / 投机解码都不兼容 | `server_args.py:876` |
| NIXL + HiSparse + decode-only spec → `NotImplementedError` | `nixl/conn.py:1006-1022` |
| Flex Attention + spec 被拒 | `arg_groups/attention_hook.py:96` |
| `topk>1 且 page_size>1` 限定 backend ∈ {flashinfer, fa3, triton} | `arg_groups/speculative_hook.py:392-404` |

### 6.12 各算法在 PD 下的支持矩阵

`spec_info.py:187-207` 是唯一的分派点：

| 算法 | `build_disagg_draft_input` → | `carries_draft_hidden_states()` | draft KV 注册 |
|---|---|---|---|
| EAGLE / EAGLE3 | `build_eagle_disagg_draft_input` | True | 有 |
| FROZEN_KV_MTP | `build_eagle_disagg_draft_input`（`is_eagle()` 覆盖它） | True | **无**（`primary_draft_kv_pool` 是 None） |
| DSPARK | `build_dspark_disagg_draft_input` | False | 有 |
| DFLASH | **None**（builder 存在但没接线，见 §9） | False | 有 |
| STANDALONE | None | False | 有 |
| NGRAM | None | False | 无（`has_draft_kv()` 为 False） |

插件算法的钩子默认 opt-out（`speculative/spec_registry.py:172-183`）。

---

## 7. 集中式 vs PD 全面对照

### 7.1 逐维度对照表

| 维度 | 集中式（mix） | PD 分离 |
|---|---|---|
| **draft KV pool 创建** | Scheduler 内 target → draft 顺序创建 | 完全一致，P/D **各自独立**创建一份 |
| **cell_size 膨胀** | 有 | 有，且 **P 与 D 必须用同一套投机参数**，否则每 token 字节数不等，注册的 entry 长度对不上 |
| **allocator** | 唯一一份，target/draft 共享 | 同样唯一一份（各实例内） |
| **`req_to_token_pool`** | 唯一一份 | 各实例一份，**req_idx 不需要对齐**（传输走的是 slot 号数组，不是 req_idx） |
| **draft 分配时机** | 每个 decode step 前预留 `2×alloc_len` | P 侧：extend 时按 extend 长度分配（draft 只写最后一个位置）；D 侧：与集中式相同 |
| **draft KV 谁写** | 本地 draft 前向 | P 侧 draft 前向写入 → RDMA 搬到 D；D 侧后续步由自己的 draft 前向写 |
| **首个 draft input 从哪来** | 上一步 verify 的 `hidden_states` 在显存里直接接着用 | 从 metadata（aux）通道读 `output_topk_p / output_topk_index / output_hidden_states`，`build_eagle_disagg_draft_input` 重建 |
| **D 侧首步 forward mode** | 无此概念 | `ForwardMode.PREBUILT`，**不进模型前向** |
| **radix cache** | 可用，EAGLE 走 bigram key view | **强制禁用**（`--disaggregation-decode-enable-radix-cache` 与 spec 互斥），D 侧只有 chunk cache |
| **超额区释放** | `release_kv_cache` 批量 free | 相同（D 侧），P 侧请求短命，随 prefill 结束整体 free |
| **retract / resume** | 支持，容量估算 spec-aware | D 侧支持；`retracted_queue` 恢复时重新按 spec 预留 |
| **CUDA graph** | draft-decode / draft-extend / target-verify 三类静态 buffer | P 模式只捕 EXTEND 相位、D 模式只捕 DECODE 相位，静态 buffer 数量因此不同 |

### 7.2 一张图看清两种部署的 draft KV 流向

```
■ 集中式（一个实例内闭环）

  step k:  [预留 2×alloc_len]
             │
             ├─ draft forward ×steps ──▶ 写 draft KV pool（超额区内）
             │                            读 draft KV pool（自回归展开）
             │
             ├─ target verify ─────────▶ 写 target KV pool（同一批 slot 的另一个视图）
             │
             ├─ accept ────────────────▶ topk>1 时 move_kv_cache 把接受段搬到连续前部
             │
             └─ commit ────────────────▶ kv_committed_len 前进；被拒段留在超额区
                                          （不 free，等请求结束批量回收）

■ PD 分离（跨实例）

  ┌──────────── P 实例 ────────────┐        ┌──────────── D 实例 ────────────┐
  │ extend forward → target KV     │        │                                 │
  │ sample → bonus_token           │        │                                 │
  │ draft forward → draft KV       │        │                                 │
  │ 产出 topk_p / topk_index /     │        │                                 │
  │      hidden_states             │        │                                 │
  │                                 │        │                                 │
  │ kv_args.kv_data_ptrs =         │        │ 同构的 kv_data_ptrs             │
  │   [target L 层..., draft L' 层] │  RDMA  │   [target L 层..., draft L' 层] │
  │        ──────────── 同一份 page_indices ────────────▶                     │
  │                                 │  WRITE │                                 │
  │ metadata buffer (aux)          │ ─────▶ │ 读 aux → build_eagle_disagg_    │
  │   bonus_token / topk_p /       │        │   draft_input                   │
  │   topk_index / hidden_states   │        │        │                        │
  └─────────────────────────────────┘        │        ▼                        │
                                             │ ForwardMode.PREBUILT（不前向） │
                                             │        │                        │
                                             │        ▼                        │
                                             │ 从 step 1 起与集中式完全相同    │
                                             └─────────────────────────────────┘
```

### 7.3 三条"看着该不同、其实相同"的地方

1. **draft pool 的容量**。P 与 D 的 `max_total_num_tokens` 通常不同（显存/并发配置不同），
   但**每 token 的字节数必须相同**。RDMA 传输按 `(layer, page)` 定位，长度对不上就是数据错乱
   而不是报错，所以两侧的 `--speculative-algorithm` / `--speculative-num-steps` /
   `--speculative-eagle-topk` / `--speculative-num-draft-tokens` 必须一致。

2. **超额区语义**。D 侧的 `[kv_committed_len, kv_allocated_len)` 与集中式完全同构。
   P 传过来的只是 `[0, kv_committed_len)` 那段（prefill 结果 + 第一个 draft 位置），
   超额区是 D 自己每步重新预留的。

3. **释放路径**。`release_kv_cache` / `_release_overallocated_kv_indices` 在两种部署下走的
   是同一份代码，没有 `disaggregation_mode` 分支。

### 7.4 一条"看着该相同、其实不同"的地方

**draft KV 的第一次写入者不同**，这是唯一实质差异：

- 集中式：请求的第 0 个 draft 位置由本实例在 extend 结束后的第一次 `draft()` 写入。
- PD：那个位置由 **P 实例**写入并 RDMA 过来，D 侧的 `PREBUILT` 步骤直接跳过前向。

这解释了为什么 D 侧不需要 hidden_states 的历史，只需要**最后一步**的 hidden_states —— 因为
EAGLE 的 draft 输入只依赖上一步的隐状态和 token，不依赖更早的。

---

## 8. 排查手册

### 8.1 开了投机解码就 OOM / 并发骤降

**现象**：加上 `--speculative-algorithm EAGLE` 后，日志里 `max_total_num_tokens` 明显变小，
`max_running_requests` 跟着掉，或者直接启动期 OOM。

**这是设计使然，不是 bug**。原因链条（详见 §4.3）：

```
draft 权重占显存  ──▶ preloaded_weights_bytes 变大 ──▶ profiling 剩余显存变小
                                                              │
cell_size 膨胀 (1 + L_draft/L_target) ──────────────────────┐  │
                                                            ▼  ▼
                                    max_total_num_tokens = 剩余显存 / cell_size
                                                            │
                                    ── 双重收窄，1 层 draft 的 61 层模型 ≈ -1.6%，
                                       但 EAGLE3（draft 层数更多）可以是两位数百分比
```

**排查顺序**：

| 步骤 | 看什么 |
|---|---|
| 1 | 日志里的 `max_total_num_tokens`，和不开 spec 时对比，算出实际收窄比例 |
| 2 | `eagle_draft_num_layers` 是否符合预期（§4.4）。若 draft config 里层数被误读成大值，膨胀会失控 |
| 3 | `get_alloc_reserve_per_decode()` = `2 × max(steps×topk, num_draft_tokens)`。这是**每个并发请求**都要留的，`max_running_requests` 的隐式上限 ≈ `max_total_num_tokens / (avg_len + reserve)` |
| 4 | 降 `--speculative-num-steps` 或 `--speculative-eagle-topk`，reserve 线性下降，比调 `--mem-fraction-static` 更有效 |

**不该做的事**：调 `--mem-fraction-static` 往上加。cell_size 膨胀是按比例的，加大 static 分数
只是让两个池同比变大，不改变 draft 占比。

### 8.2 `page_size > 1` 时 reserve 大得离谱

`get_alloc_len_per_decode()` 在 `page_size > 1 且 topk > 1 且 has_draft_kv()` 时走的是
**每分支页对齐**分支：

```python
num_new_pages_per_topk = ((page_size - 1) + spec_steps + page_size - 1) // page_size
return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)
```

例：`page_size=64, steps=3, topk=4` → `num_new_pages_per_topk = (63+3+63)//64 = 2`
→ `2 × 64 × 4 = 512` token/step/请求，再 ×2 双缓冲 = **1024**。
而 `page_size=1` 的同参数只需 `max(3×4, 4) = 12`，×2 = 24。**差 40 倍。**

**结论**：`page_size > 1` 与 `topk > 1` 组合是显存杀手。要么 `topk=1`（走
`spec_steps * spec_topk` 短路），要么 `page_size=1`。

### 8.3 PD 两侧配置不一致导致的数据错乱

**现象**：PD + spec 下 D 侧输出乱码，但没有任何异常抛出。

**根因**：RDMA 传输按 `(layer_id, page_index)` 计算偏移，`kv_data_ptrs` 的 entry 数量和每 entry
的 stride 是**两侧各自算出来的**，没有握手校验（§6.5）。任何一项不一致都会静默错位：

| 必须两侧一致 | 不一致的后果 |
|---|---|
| `--speculative-algorithm` | draft entry 数量不同 → layer id 段错位 |
| draft 模型（`--speculative-draft-model-path`）的层数 | draft entry 数量不同 |
| `--page-size` | page stride 不同 |
| KV dtype / `--kv-cache-dtype` | 每 token 字节数不同 |
| `--tp` 相关的 head 切分 | 每层 entry 长度不同 |

`--speculative-num-steps` / `--speculative-eagle-topk` / `--speculative-num-draft-tokens`
不影响 KV 布局，但影响 aux 通道的 `num_states`（§8.5），也建议一致。

**排查手段**：两侧启动日志各自打印 `max_total_num_tokens` 和层数；用同一份启动脚本的
spec 参数段是最省事的做法。

### 8.4 PD + `--disaggregation-decode-enable-radix-cache` 直接启动失败

```
ValueError: --disaggregation-decode-enable-radix-cache is incompatible with
speculative decoding (--speculative-algorithm EAGLE)
```

这是 `pd_disaggregation_hook.py:89-94` 的显式拒绝，不是 bug。PD + spec 下 D 侧只能用
chunk cache。原因是 EAGLE 的 radix key 是 bigram view（§5.10），与 D 侧靠
`decode_prefix_len` 回传给 P 做前缀跳传的协议无法对齐。

**绕法**：没有。要么关 D 侧 radix cache，要么关投机解码。

### 8.5 `topk × num_steps > 16` 时 PD 下 draft 输入被截断

`MetadataBuffers` 里 `output_topk_p` / `output_topk_index` 的宽度**硬编码为 16**
（[disaggregation/utils.py:366-371](../../../python/sglang/srt/disaggregation/utils.py#L366-L371)），
而约束只写在注释里（`utils.py:545`：`speculative_eagle_topk should not be greater than 16 currently`）。

更危险的是 `eagle_disaggregation.py:28-30`：

```python
num_states = spec.speculative_eagle_topk
if spec.enable_multi_layer_eagle:
    num_states *= spec.speculative_num_steps   # ← 可以轻易超过 16
```

D 侧随后 `req.output_topk_p[:num_states]`，`num_states > 16` 时切片越界读到相邻请求的槽位。
**代码里没有找到任何参数级校验**（`speculative_hook.py` 只校验 `topk > 1 + page_size > 1`
的 backend 限制，没有上界检查）。

**排查**：PD + `--enable-multi-layer-eagle` 时手算 `topk × num_steps`，超过 16 就换参数。

### 8.6 draft 池是 None 导致的 AttributeError

`primary_draft_kv_pool` 在这些情况下是 `None`：

| 情况 | 原因 |
|---|---|
| NGRAM | `has_draft_kv()` 返回 False（`spec_info.py:159-163`） |
| FROZEN_KV_MTP | worker 只建了 64 token 的 dummy 池（§9.1） |
| 未开投机解码 | 显然 |

PD 注册路径 `prefill.py:243-264` 对 `None` 是做了判空的，但如果给新算法接线时忘了，
表现是启动期 `AttributeError: 'NoneType' object has no attribute 'get_contiguous_buf_infos'`。

### 8.7 "被拒草稿的显存没释放" —— 不是泄漏

**现象**：观察 allocator 的 `available_size()`，发现开 spec 后可用量长期偏低。

**这是超额区（§5.9）**。`[kv_committed_len, kv_allocated_len)` 之间的 slot 被占着但不在
radix tree 里，也不可 evict，直到请求结束才 `release_kv_cache` 一次性回收。
稳态下每个运行中请求占 `2 × alloc_len` 的超额区。

**如何确认不是真泄漏**：请求全部结束后 `available_size()` 应回到初值。
`_release_overallocated_kv_indices` 里有一条 `assert`：非 spec 时
`start_p == end_p`（超额区必须为空），spec 时无此断言 —— 说明超额区非空是预期状态。

---

## 9. 已发现的粗糙边缘

以下都是**读代码得出的结论，未在运行的 server 上实测**。列出来是为了给后续维护提供线索，
不是 bug 报告。

### 9.1 FROZEN_KV_MTP 的显存被超额预留

`is_eagle()` 对 FROZEN_KV_MTP 返回 True，代码里明确挂着 FIXME：

```python
# spec_info.py:99-106
def is_eagle(self) -> bool:
    # FIXME(kpham_sgl): FROZEN_KV_MTP shouldn't be lumped in here.
    return self in (EAGLE, EAGLE3, NEXTN, FROZEN_KV_MTP)
```

于是 `pool_configurator.py:187-222` 认为它需要 draft KV，按层数比例膨胀 `cell_size`：

```python
if (kvc.spec_algorithm.is_eagle() or kvc.spec_algorithm.is_standalone()) and not kvc.is_draft_worker:
    ...
    self._cell_size = int(self._cell_size * (1 + draft_num_layers / int(num_layers)))
```

但 `frozen_kv_mtp_worker_v2.py:185-188` 实际只申请了 **64 token 的 dummy 池**：

```python
self.draft_pool_config = MemoryPoolConfig(
    max_total_num_tokens=64,  # Dummy value
    max_running_requests=memory_pool_config.max_running_requests,
)
```

**后果**：按 `draft_num_layers / num_layers` 比例预留的那笔 KV 显存被白白扣掉，
`max_total_num_tokens` 偏小。1 层 draft 对 61 层 target 约 1.6%，绝对值可观（几百 MB 量级）。

### 9.2 DFLASH 在 PD 下的 draft input builder 是死代码

`dflash_disaggregation.py:16` 定义了 `build_dflash_family_disagg_draft_input`，
但 `spec_info.py:187-207` 的分派只有两个分支：

```python
if self.is_eagle():   → build_eagle_disagg_draft_input
if self.is_dspark():  → build_dspark_disagg_draft_input
return None           # ← DFLASH 落到这里
```

全仓 grep `build_dflash_family_disagg_draft_input` 只有定义处一个命中，**零调用点**。
DFLASH 在 PD 下会拿到 `spec_info = None`，走"没有投机解码种子"的路径 —— 静默降级而非报错。

### 9.3 aux 通道的 topk 宽度硬编码 16，无参数校验

见 §8.5。三个事实叠在一起：

| 事实 | 位置 |
|---|---|
| buffer 宽度硬编码 16 | `disaggregation/utils.py:366-371` |
| 约束只在注释里 | `disaggregation/utils.py:545` |
| `num_states` 可以是 `topk × num_steps` | `eagle_disaggregation.py:28-30` |
| 无任何上界参数校验 | `arg_groups/speculative_hook.py` 只有 backend 相关校验 |

一个 `assert num_states <= 16` 就能把静默错误变成启动期报错。

### 9.4 `build_kv_layer_ids` 的保留段只覆盖 HybridLinearKVPool

函数的 docstring 把保留段（reserved band）的正确性论证写得很完整 —— draft 从 0 开始编号，
会与 target 冲突，所以重映射到 target 层号之上的一段，两端跑同一份 draft config 所以一致。
但第一行就是：

```python
if not isinstance(token_to_kv_pool, HybridLinearKVPool):
    return []
```

返回 `[]` 意味着**退化为位置配对**（positional pairing）。所以这套论证只对
hybrid-linear 模型（Mamba/GDN 混合）成立；常规 MHA/MLA 模型的 draft entry 配对完全依赖
"两侧 entry 顺序和数量一致"这个隐式约定 —— 也就是 §8.3 里那些静默错乱的根源。

### 9.5 `EAGLEWorkerV2.draft_extend()` 是空实现

```python
# eagle_worker_v2.py:777-778
def draft_extend(self):
    pass
```

真正的逻辑在紧随其后的 `_draft_extend_for_prefill()`。空的 public 方法看着像是基类接口
的占位，但对读代码的人是个陷阱 —— 以为 draft-extend 没做事。

### 9.6 `decoupled_spec_io.py` 里自我记录的潜在 bug

`VerifierCommitSegment.append_message` 里的 `raise` 运行在 drafter 的
`TokenSyncThread` 上。线程里抛异常不会传播到主线程，会**静默杀掉控制线程**，
表现为 drafter 卡住而不是报错。文件里的注释已经承认了这一点。

⚠️ 注意区分：`decoupled_spec_io.py` 是 **decoupled 投机解码**（drafter 与 verifier
分进程）的 ZMQ 控制面，与本文 §6 讲的 **PD KV 分离**完全是两回事，不要混淆。

### 9.7 行号引用本身就是个坑（已实测）

同一段 `pool_configurator.py` 里的 DSV4 投机膨胀逻辑，两篇既有文档曾给出**两套互不
相同、且都是错的**行号：

| 文档 | 曾引用的行号 | 实际 |
|---|---|---|
| `dsv4_mtp_centralized_execution.md` | `pool_configurator.py:507` / `:553-561` | 都不对 |
| `dsv4_mtp_layer_forward_dataflow.md` | `pool_configurator.py:574` / `:633-640` | 都不对 |

正确定位（快照 `33ed29a0ee`）：

- 类：[`pool_configurator.py`](../../../python/sglang/srt/model_executor/pool_configurator.py) 的
  `DSV4PoolConfigurator`（快照 :763）
- 膨胀点：`DSV4PoolConfigurator.__init__` 内 `if self.is_speculative:` 那段
  `bytes_per_full_token *= (target_layers + draft_layers) / target_layers`（快照 :825-832）

注意这里膨胀的是 `bytes_per_full_token` 而不是 `DefaultPoolConfigurator` 的
`_cell_size`（§4.2），两条路径互斥：DSV4 走前者，常规模型走后者。

上述两篇文档已修正，并统一改成**函数名/类名优先**的引用风格（链接只指到文件，行号
写成"快照 :NNN"作为次要线索）。本文 §10 的索引仍是行号表，读的时候请以符号名为准
重新定位，行号只当粗略路标。

---

## 10. 文件行号索引

按"我要改什么"组织。行号基于 `33ed29a0ee`，以函数名为准。

### 10.1 创建阶段

| 关注点 | 文件 | 位置 |
|---|---|---|
| target 池创建 + draft 权重记账 | `managers/scheduler.py` | `init_target_memory_pool` :993-1008 |
| target → draft 创建顺序 | `managers/scheduler.py` | `init_memory_pools` :1009-1022 |
| `primary_draft_kv_pool` 取出 | `managers/scheduler.py` | :1378-1382 |
| PD wire schema 推导 | `managers/scheduler.py` | :1384-1406 |
| **`cell_size` 膨胀（EAGLE/STANDALONE）** | `model_executor/pool_configurator.py` | `DefaultPoolConfigurator.__init__` :183-222 |
| `cell_size` 膨胀（DFLASH/DSPARK，加法式） | `model_executor/pool_configurator.py` | :224-246 |
| draft 复用 target 的 `MemoryPoolConfig` | `mem_cache/kv_cache_configurator.py` | `configure()` :296-306 |
| draft 共享 `req_to_token_pool` | `mem_cache/kv_cache_configurator.py` | :516-522 |
| draft 原样返回 allocator | `mem_cache/kv_cache_configurator.py` | :1960-1988 |
| `loc_space_scale` | `mem_cache/kv_cache_configurator.py` | :335-338 |
| DSV4 子池置零 | `mem_cache/kv_cache_configurator.py` | :362-374 |
| FROZEN_KV_MTP 的 dummy 池 | `speculative/frozen_kv_mtp_worker_v2.py` | `alloc_memory_pool` :176-198 |
| 模型侧门禁（SWA / DSA / SSM 拒绝） | `mem_cache/kv_cache_builder.py` | :252-284 |

### 10.2 运行期分配与释放

| 关注点 | 文件 | 位置 |
|---|---|---|
| **每步预留长度公式** | `mem_cache/allocation_sizing.py` | `get_alloc_len_per_decode` :17-45 |
| **双缓冲 2×** | `mem_cache/allocation_sizing.py` | `get_alloc_reserve_per_decode` :48-54 |
| 页对齐分配长度 | `mem_cache/allocation_sizing.py` | `page_aligned_decode_alloc_lens` :57-77 |
| `req_to_token` 额外上下文长度 | `mem_cache/allocation_sizing.py` | `get_req_to_token_extra_context_len` :80-96 |
| **整段批量释放** | `mem_cache/common.py` | `release_kv_cache` :215-257 |
| **超额区释放 + 非 spec 断言** | `mem_cache/common.py` | `_release_overallocated_kv_indices` :260-283 |
| draft KV 连续布局 | `speculative/*_utils.py` | `assign_draft_cache_locs_contiguous` |
| 每分支页对齐 + 前缀尾复制 | `speculative/*_utils.py` | `duplicate_prefix_tail_to_draft_branches` |
| 接受段搬移（topk>1） | `speculative/*_utils.py` | `move_accept_tokens_to_target_kvcache` → `move_kv_cache` |
| RadixCache bigram key | `mem_cache/radix_cache.py` | `is_bigram` / `maybe_to_bigram_view` |
| draft-extend 真实入口 | `speculative/eagle_worker_v2.py` | `_draft_extend_for_prefill` :780- |

### 10.3 PD 传输

| 关注点 | 文件 | 位置 |
|---|---|---|
| **P 侧 draft 指针追加** | `disaggregation/prefill.py` | `_init_kv_manager` :243-264 |
| layer shard 下只有末 rank 传 draft | `disaggregation/prefill.py` | :218-220 |
| D 侧镜像注册 | `disaggregation/decode.py` | :543-544 附近 |
| **layer id 保留段** | `disaggregation/utils.py` | `build_kv_layer_ids` :925-960 |
| **aux buffer 的四个投机字段** | `disaggregation/utils.py` | `MetadataBuffers.__init__` :365-374 |
| P 侧写 aux | `disaggregation/utils.py` | `set_buf` :543-560 |
| state（非 KV）组件注册 | `disaggregation/utils.py` | `setup_state_kv_args` |
| **PD draft input 分派** | `speculative/spec_info.py` | `build_disagg_draft_input` :187-207 |
| EAGLE 的 PD 种子重建 | `speculative/eagle_disaggregation.py` | `build_eagle_disagg_draft_input` :20- |
| DFLASH 死代码 | `speculative/dflash_disaggregation.py` | :16 |
| `has_draft_kv()` / `carries_draft_hidden_states()` | `speculative/spec_info.py` | :159-168 |
| **PD + spec 门禁** | `arg_groups/pd_disaggregation_hook.py` | :77-94（radix cache 互斥在 :89-94） |
| topk/page_size 的 backend 限制 | `arg_groups/speculative_hook.py` | :810-820 |
| 插件算法默认 opt-out | `speculative/spec_registry.py` | :172-183 |

---

## 附：最容易记混的结论

放在最后，方便回头速查。每条都是本文正文里论证过的。

| # | 容易记错的说法 | 正确的说法 |
|---|---|---|
| 1 | "draft 和 target 共享 KV pool" | 共享的是 **slot 编号空间**（allocator）和**位置表**（`req_to_token_pool`）；**KV 值张量是两份独立对象**。§3.2 |
| 2 | "draft 也独立做一次 profiling 算容量" | 不做。draft 直接复用 target 的 `MemoryPoolConfig`。容量由 target 的那一次 profiling 定。§4.5 |
| 3 | "开 spec 会额外多占一笔显存" | 不是"额外"。是把 target 的 `cell_size` 膨胀，用同一笔预算除出更小的 `max_total_num_tokens`。§4.3 |
| 4 | "被拒绝的 draft token 的 slot 会被 free" | 不会。留在超额区 `[kv_committed_len, kv_allocated_len)`，请求结束时整段批量回收。**没有任何代码遍历 draft 树去逐个 free**。§5.9 |
| 5 | "每步预留 `steps × topk`" | 是 `2 × max(steps × topk, num_draft_tokens)`。2× 是吸收 overlap 模式下 `kv_committed_len` 滞后的双缓冲。§5.2 |
| 6 | "`page_size` 只影响对齐，不影响用量" | `page_size > 1 且 topk > 1` 时走每分支页对齐分支，预留量可以差 40 倍。§8.2 |
| 7 | "PD 下 D 侧重算 draft KV" | 不重算。draft 池与 target 池共享 slot 编号，指针追加到同一个注册列表后，**同一份 `page_indices` 同时覆盖两个池**。§6.2 |
| 8 | "PD 下 P 实例不跑 draft 模型" | 跑。EAGLE 系代码里没有任何 `disaggregation_mode == "prefill"` 的分支。§6.10 |
| 9 | "draft 的 hidden_states 走 KV 通道传" | 走 **aux（metadata）通道**，和 `bonus_token` / logprobs 同一个 RDMA buffer。§6.8 |
| 10 | "D 侧第一步跑一次 extend 前向" | 不跑。`ForwardMode.PREBUILT` 被短路，第一个 token 和 draft 种子全部来自 P。§6.10 |
| 11 | "PD 下也能开 D 侧 radix cache" | 不能，启动期 `ValueError` 硬拒。§8.4 |
| 12 | "`decoupled_spec_io.py` 是 PD 传输的一部分" | 无关。那是 drafter↔verifier 分进程的控制面，与 PD KV 分离是两个正交概念。§9.6 |
| 13 | "`accept` 和 `correct` 是同义词" | `accept_*` **含** bonus token，`correct_*` **只算草稿**。例外：`accept_rate`（不含 bonus）与 `accept_length`（含 bonus）按论文惯例。见 `speculative-naming` skill |
| 14 | "NGRAM 也要建 draft KV pool" | 不要。`has_draft_kv()` 对 NGRAM 返回 False。§3.3 |
| 15 | "PD 两侧 spec 参数不一致会报错" | **不会报错，会静默数据错乱**。KV 布局没有握手校验。§8.3 |
