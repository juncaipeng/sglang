# DeepSeek V4 Cache 管理方案全面梳理

> 适用代码版本：`main` 分支，校准至 commit `c608f9bf75`（2026-08）。本文聚焦 DeepSeek V4（DSV4）在 SGLang 中**独有的多层压缩 KV Cache 体系**，从概念到实现逐层展开。
> 部署/启动参数请参阅同目录 [deepseek_v4_deployment_guide.md](deepseek_v4_deployment_guide.md)，本文不重复部署细节，专注 Cache 内部机制。
>
> **本次校准的主要变更**（相对上一版）：
> - 文件搬迁：`layers/attention/dsv4/metadata_kernel.py` → `kernels/ops/attention/dsv4/metadata_kernel.py`；`model_executor/model_runner_kv_cache_mixin.py` → `mem_cache/kv_cache_configurator.py`；HiSparse 逻辑从 `hisparse_memory_pool.py` 拆到 `mem_cache/allocator/hisparse.py` + `mem_cache/pool_host/hisparse.py` + `managers/hisparse_coordinator.py`。
> - 新增第 7 类布局 `DeepSeekV4UnifiedKVPool`（ATOM 融合布局，仅 HIP）。
> - **c128 压缩态改为"请求级"寻址**（不再按 SWA 页），容量作为固定项在切分 token 前先扣除。
> - Indexer 新增 FP4 变体（68 B/token）。
> - 压缩态 dtype 可由 `SGLANG_DSV4_COMPRESS_STATE_DTYPE` 选择（不再恒为 FP32）。
> - 在线 c128 + MTP 不再是绝对禁止（实验开关 `SGLANG_EXPERIMENTAL_ONLINE_C128_MTP`）。
> - 已删除的开关：`SGLANG_OPT_USE_FUSED_STORE_CACHE`、`SGLANG_OPT_CACHE_SWA_TRANSLATION`。

---

## 0. 阅读地图（由浅入深）

| 层次 | 章节 | 你将理解 |
|---|---|---|
| 概念 | §1 为什么 V4 的 Cache 不一样 | V3 vs V4 的 KV 形态差异、三层压缩动机 |
| 概念 | §2 三层 Cache 模型 | SWA / c4 / c128 与 `compress_ratios` 的关系 |
| 结构 | §3 子池与内存布局 | `DeepSeekV4TokenToKVPool` 的 6 个子池（+1 种融合布局）、584B/token 布局 |
| 结构 | §4 压缩状态环（CompressState） | 在途压缩态的 ring buffer 设计、页级 vs 请求级寻址 |
| 容量 | §5 显存容量计算 | `DSV4PoolConfigurator` 的 `bytes_per_full_token` 系数模型 + c128 固定项 |
| 流程 | §6 分配与地址翻译 | full→swa→c4/c128 的 loc 翻译链路 + Triton 元数据核 |
| 流程 | §7 写入路径 | 模型写 SWA、Compressor 写压缩池、Indexer 写 index_k |
| 流程 | §8 读取路径 | C4Indexer top-512 选择 + FlashMLA 稀疏注意力 |
| 进阶 | §9 HiSparse GPU↔Host 卸载 | c4 池的分级存储与换入换出 |
| 进阶 | §10 MTP / NextN 草稿层 | 为什么草稿层强制 ratio=0 |
| 进阶 | §11 跨实例导出：HiCache / PD 分离 | 各子池怎么暴露 buffer 信息给传输层 |
| 收口 | §12 关键约束与易错点 | page_size、命名陷阱、dtype 等硬约束 |
| 附录 | §13 端到端数据流总图 / §14 术语与文件速查 | 一页图 + 符号定位表 |

核心源码索引：
- 池实现：[deepseek_v4_memory_pool.py](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)
- 压缩状态：[deepseek_v4_compress_state.py](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py)
- 容量规划：[pool_configurator.py](../../../python/sglang/srt/model_executor/pool_configurator.py)
- 池/分配器装配：[kv_cache_configurator.py](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py)
- 元数据核：[kernels/ops/attention/dsv4/metadata_kernel.py](../../../python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py)
- 注意力后端：[deepseek_v4_backend.py](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py)
- 模型层：[models/deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py)

---

## 1. 为什么 V4 的 Cache 不一样

### 1.1 从 V3 的 MLA 说起

DeepSeek V3 用 **MLA（Multi-head Latent Attention）**：每个 token 只缓存一份低秩压缩潜变量 `c_kv`（`kv_lora_rank=512`）加上一段 RoPE 维度（`qk_rope_head_dim=64`），所有 head 共享。KV Cache 的核心特征是"**每 token 一份、全序列保留、全精度（或 FP8）**"。

V3 的 cache 形态可以概括为：

```
每 token 缓存：[ c_kv (512) | k_rope (64) ]，所有层都一样，全序列长度保留
```

### 1.2 V4 的三个根本变化

DeepSeek V4 在注意力侧引入了三项与 cache 强相关的结构变化：

| 变化 | 含义 | 对 Cache 的影响 |
|---|---|---|
| **逐层压缩比** | 每层有一个 `compress_ratio ∈ {0, 4, 128}` | 不同层缓存**不同压缩密度**的 KV，必须分池管理 |
| **稀疏选择（C4 Indexer）** | ratio=4 层用 indexer 选 top-512/1024 压缩块 | 需要额外缓存 index_k（轻量索引向量）+ 维护稀疏页表 |
| **滑窗 + 压缩并存** | 近窗用全精度 SWA，远程用压缩态 | 同一层同时存在 SWA 全精度池和压缩池两份数据 |

直观对比：

| 维度 | V3 (MLA) | V4 (压缩注意力) |
|---|---|---|
| 每层 KV 形态 | 统一 `c_kv+rope` | 按 `compress_ratio` 分三类 |
| 缓存层数 | 全部层等价 | 逐层不同（0/4/128 混合配置）|
| 注意力范围 | 全序列 dense | SWA 近窗 dense + 远程压缩 sparse |
| 稀疏选择 | 无 | C4 Indexer top-512（小模型）/ top-1024（大模型）|
| KV 精度 | FP8/BF16 | SWA FP8（584B/token）+ 压缩态 FP32（可切 BF16）|
| 子池数量 | 1 个 KV 池 | 6 类子池（另有 1 种 HIP 融合布局，见 §3）|

### 1.3 一句话总结

> **V4 Cache = "全精度滑动窗口（SWA）" + "两档压缩远程历史（c4 / c128）" + "稀疏选择索引（C4 Indexer）" + "在途压缩状态环（CompressState）" 的组合体。**

这套组合用一个统一的顶层池 `DeepSeekV4TokenToKVPool` 封装，对 Scheduler/Allocator 暴露成"一个 KV 池"，但内部按层路由到不同子池。

---

## 2. 三层 Cache 模型与 `compress_ratios`

### 2.1 三个压缩档位

V4 每一层归属于以下三档之一，由 HF config 的 `compress_ratios`（长度 = 层数）逐层指定：

| 档位 | `compress_ratio` | 别名 | 含义 | 是否有 Indexer |
|---|---|---|---|---|
| 全精度滑窗 | `0` | SWA | 不压缩，进入滑动窗口全精度池 | 否 |
| 4 倍压缩 | `4` | CSA / c4 | 每 4 个 token 压成 1 个压缩块 | **是**（top-512）|
| 128 倍压缩 | `128` | HCA / c128 | 每 128 个 token 压成 1 个压缩块 | 否 |

> 注意：`compress_ratio` 是"**多少个原始 token 压成一个压缩槽**"，所以 ratio 越大压缩越狠、保留得越少。每一层都有它自己的 SWA 近窗（ratio=0 的语义是"只有 SWA、没有压缩历史"），ratio=4/128 的层在 SWA 之外**额外**维护对应粒度的压缩历史。

### 2.2 `compress_ratios` 如何流入系统

```
HF config.compress_ratios (List[int], 逐层)
        │
        │  configs/model_config.py 提升进 ModelConfig
        ▼
ModelConfig.compress_ratios
        │
        ├─► models/deepseek_v4.py: MQALayer 用 compress_ratios[layer_id]
        │     决定本层是否建 Compressor / C4Indexer
        │
        ├─► mem_cache/deepseek_v4_memory_pool.py: 按 ratio 统计
        │     c4_layer_num / c128_layer_num，建对应子池与状态池
        │
        └─► model_executor/pool_configurator.py: DSV4PoolConfigurator
              用 ratio 分布计算 bytes_per_full_token
```

关键点：`compress_ratios` 是**逐层、PP 切片感知**的。在 PP（流水并行）下，每个 stage 只看自己负责的层区间 `[start_layer, end_layer)`，池只为本 stage 的层分配（见 [pool_configurator.py:771-773](../../../python/sglang/srt/model_executor/pool_configurator.py#L771-L773) 的 `cfg.compress_ratios[layer_info.start_layer:layer_info.end_layer]`，`pp_size>1` 时还会打一行 `DSV4 pool PP slice:` 日志便于核对）。

### 2.3 层数统计（决定子池层数）

在 [deepseek_v4_memory_pool.py:571-572](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L571-L572)：

```python
self.c4_layer_num   = sum(1 for r in stage_ratios if r == 4)
self.c128_layer_num = sum(1 for r in stage_ratios if r == 128)
```

- `swa_kv_pool` 用 `layer_num`（全部层，因为**每层都有 SWA**）。
- `c4_kv_pool` / `c4_indexer_kv_pool` 用 `c4_layer_num`。
- `c128_kv_pool` 用 `c128_layer_num`。

层映射 `layer_mapping`（[_init_compressed_layer_mapping](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L946)，构造末尾在 [654](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L654) 调用）把"绝对层 id"映射到"该层在对应压缩子池里的局部 id + 指向哪个压缩池"，用 `DeepSeekV4LayerItem(compress_ratio, compress_layer_id, compress_kv_pool)`（[:391](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L391)）表示。

---

## 3. 顶层池 `DeepSeekV4TokenToKVPool` 与子池布局

### 3.1 6 类子池的全景

`DeepSeekV4TokenToKVPool` 继承 `BaseSWAKVPool`，构造时一次性建出 6 类子池（[deepseek_v4_memory_pool.py:463-674](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L463-L674)）：

```
DeepSeekV4TokenToKVPool
│
├── swa_kv_pool          : DeepSeekV4SingleKVPool   (全部层, 全精度滑窗 KV, 584B/token)
│
├── c4_kv_pool           : DeepSeekV4SingleKVPool   (c4 层, 4x 压缩 KV)
│                          └─ HiSparse 开启时替换为 HiSparseC4DevicePool
│
├── c128_kv_pool         : DeepSeekV4SingleKVPool   (c128 层, 128x 压缩 KV)
│
├── c4_indexer_kv_pool   : DeepSeekV4IndexerPool    (c4 层, 稀疏选择用的 index_k)
│
├── compress_state_pools : List[CompressStatePool]  (c4/c128 层, 在途注意力压缩态)
│
└── indexer_compress_state_pools : List[CompressStatePool] (仅 c4 层, 在途 indexer 压缩态)

[HIP 专属替代布局]
└── unified_kv_pool      : DeepSeekV4UnifiedKVPool  (把 swa/c4/c128 三份 KV 合进一块
                                                     连续 buffer, 此时上面三个 *_kv_pool
                                                     全为 None)
```

逐池职责：

| 子池 | 类型 | 覆盖层 | 存什么 | 粒度（page_size）|
|---|---|---|---|---|
| `swa_kv_pool` | `DeepSeekV4SingleKVPool` | 全部 | 近窗全精度 KV（FP8 nope + BF16 rope）| `swa_page_size` |
| `c4_kv_pool` | `DeepSeekV4SingleKVPool` / `HiSparseC4DevicePool` | c4 层 | 4x 压缩后的 KV | `page_size // 4` |
| `c128_kv_pool` | `DeepSeekV4SingleKVPool` | c128 层 | 128x 压缩后的 KV | `page_size // 128` |
| `c4_indexer_kv_pool` | `DeepSeekV4IndexerPool` | c4 层 | indexer 的 index_k + scale | `page_size // 4` |
| `compress_state_pools[i]` | `CompressStatePool` | c4/c128 层 | 在途压缩注意力态 (kv, score) | ring buffer |
| `indexer_compress_state_pools[i]` | `CompressStatePool` | 仅 c4 层 | 在途 indexer 压缩态 | ring buffer |
| `unified_kv_pool` | `DeepSeekV4UnifiedKVPool` | 全部（融合）| swa+c4+c128 三段合一 | 见 §3.6 |

> 子池的实例化都走可被子类覆写的工厂方法：`_make_kv_pool`（[:830](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L830)）、`_make_indexer_pool`（[:859](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L859)）、`_make_attn_state_pool`（[:885](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L885)）、`_make_indexer_state_pool`（[:907](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L907)）。NPU 用 `DSV4NPUTokenToKVPool` 子类覆写它们（选择由 [kv_cache_configurator.py](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1067) 的 `pool_cls = DSV4NPUTokenToKVPool if _is_npu else DeepSeekV4TokenToKVPool` 决定）。

### 3.2 SWA 池的逐 token 字节布局（584B）

`DeepSeekV4SingleKVPool.get_bytes_per_token()`（[deepseek_v4_memory_pool.py:106](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L106)）：

```
dim_per_token = qk_nope_head_dim                              # 448  (FP8, 1 byte each)
              + qk_rope_head_dim * rope_storage_dtype.itemsize # 64 * 2 = 128 (BF16)
              + qk_nope_head_dim // quantize_block_size         # 448 // 64 = 7 (FP8 scales)
              + scale_pad                                       # 1
              = 584 bytes / token
```

构造时硬断言这一布局：

```python
assert bytes_per_token == 448 + 64 * 2 + 8, (
    "DSV4 KV layout: qk_nope_head_dim FP8 (448) + qk_rope_head_dim BF16 "
    "(64*2) + nope FP8 scales + scale_pad = 584 bytes/token")
```

字节构成可视化：

```
┌──────────────────────────────┬──────────────────┬───────────┬─────┐
│ k_nope FP8                   │ k_rope BF16      │ nope scale│ pad │
│ 448 B                        │ 128 B (64×2)     │ 7 B       │ 1 B │
└──────────────────────────────┴──────────────────┴───────────┴─────┘
        448                +           128       +     7     +  1  = 584
```

要点：
- 存储 dtype 是 `uint8`（按字节存），buffer 形状 `[num_pages, bytes_per_page_padded]`（`create_buffer`，[:115](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L115)）。
- 每页字节数 `bytes_per_page_padded` 会向上对齐到 **576 的倍数**（[deepseek_v4_memory_pool.py:119](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L119)），满足 FlashMLA kernel 对齐要求。
- `量化块大小 quantize_block_size = 64`（[:87](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L87)），即每 64 个 nope 维共享 1 个 FP8 scale → 448/64 = 7 个 scale。
- 三个 KV 池（swa/c4/c128）**复用同一个** `DeepSeekV4SingleKVPool` 实现（[:59](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L59)），只是 `size` 和 `page_size` 不同；压缩态的 KV 形态和 SWA 完全一致（都是压缩后的"一个 KV 槽"，只是代表的原始 token 数不同）。

### 3.3 Indexer 池的布局（132 B 或 FP4 的 68 B）

`DeepSeekV4IndexerPool`（[deepseek_v4_memory_pool.py:260-390](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L260-L390)，`get_bytes_per_token` 在 [:291](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L291)），只服务 c4 层，有两档精度：

| 变体 | 每 token 字节 | 组成 |
|---|---|---|
| 默认（FP8/int8） | **132** | `index_head_dim`(128) + `index_head_dim//quant_block_size * 4`（128//128×4 = 4 B FP32 scale）|
| FP4 | **68** | `index_head_dim // 2`(64) + 4 B scale |

FP4 变体由 `get_exec().kernel.enable_deepseek_v4_fp4_indexer` 决定（`C4Indexer.__init__`，[indexer.py:876-934](../../../python/sglang/srt/layers/attention/dsv4/indexer.py#L876-L934)），写入走 `set_index_k_fp4`（[deepseek_v4_memory_pool.py:1249](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1249) → 池内 `set_index_fp4` [:373](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L373)），非 FP4 走 `set_index_k_fused` → `set_index_fused`（[:359](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L359)）。读侧 view 的最后一维随之是 68 或 132（见 §8.1）。

页 buffer `index_k_with_scale_buffer` 形状 `[num_pages, page_bytes]`。Indexer 缓存的不是注意力 KV，而是**用于稀疏选择打分的轻量索引向量** index_k（见 §8）。

### 3.4 `c4_indexer_kv_pool` 的逻辑放大：`c4_logical_size = c128_size * 32`

注意 [deepseek_v4_memory_pool.py:502](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L502) 与 [:643](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L643)：

```python
self.c4_logical_size = c128_size * 32
...
indexer_size = self.c4_logical_size
self.c4_indexer_kv_pool = self._make_indexer_pool(size=indexer_size, ...)
```

indexer 池按 `c128_size * 32` 而非 `c4_size` 分配。原因：c4 池在 HiSparse 下会被 host-to-device 比例缩小（见 §9），但 indexer 的逻辑寻址空间需要覆盖未缩小的完整 c4 token 数；用 `c128_size*32`（= `full_token // 128 * 32` = `full_token // 4`）保证 indexer 寻址不受 c4 物理池缩小影响。

> ⚠️ 旧版文档写的"非 HIP 路径才放大、HIP 用 `c4_size`"已过时——现在**无条件**用 `c4_logical_size`。

### 3.5 PP 切片感知

构造函数用 `_stage_start / _stage_end` 标定本 PP stage 的层区间（[deepseek_v4_memory_pool.py:551-556](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L551-L556)），所有 per-layer 列表（`layer_mapping`、`compress_state_pools`、`indexer_compress_state_pools`）的长度都是**全局层数** `total_L`，但只有 `[_stage_start, _stage_end)` 区间内被填充，其余为 `None`。绝对层 id → SWA 池局部 id 用 `_swa_local_layer_id = layer_id - _stage_start`（[deepseek_v4_memory_pool.py:1059](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1059)）。

### 3.6 第 7 种布局：`DeepSeekV4UnifiedKVPool`（ATOM 融合，仅 HIP）

[deepseek_v4_memory_pool.py:398-462](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L398-L462) 新增了一个"统一 KV"布局：把 SWA、c4、c128 三份 KV 放进**同一块连续 buffer** 的不同区段，供 ROCm 上的融合 Triton kernel 一次寻址。

| 维度 | 说明 |
|---|---|
| 启用条件 | `is_unified_kv_triton()` = `is_hip() and SGLANG_HACK_FLASHMLA_BACKEND == "unified_kv_triton"`（[unified_kv_kernels/env_gate.py](../../../python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/env_gate.py)）|
| 生效后 | `swa_kv_pool` / `c4_kv_pool` / `c128_kv_pool` **全部为 `None`**，读写改走 `unified_kv_pool` |
| 写入 | `set_unified_key_buffer_radix_fused_norm_rope`（[:1201](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1201)）替代 `set_swa_key_buffer_radix_fused_norm_rope` |
| 传输导出 | 额外的 `get_unified_swa_ring_buf_infos`（[:730](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L730)）、`unified_region_buffers`（[:748](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L748)）|
| 与 HiSparse | **互斥**，`validate_hisparse` 直接报错（见 §9.5）|

> 排查显存/形状问题时先确认走的是哪套布局：如果 `pool.swa_kv_pool is None`，你在看的是 unified 布局，§3.1~§3.4 的分池尺寸推导不适用。

---

## 4. 压缩状态环 `CompressStatePool`（在途压缩态）

### 4.1 为什么需要"状态环"

压缩不是一次性的。要把 4 个（或 128 个）原始 token 压成一个压缩 KV 槽，压缩器需要**跨多个 forward step 累积**这些 token 的中间态（在线归约 max/sum，或缓存原始 (kv, score) 等到凑满一个块再写出）。这份"还没压完、正在累积"的中间数据，就存在 `CompressStatePool` 里。

类比：c4 层每收集到 4 个 token 才落一个压缩块到 `c4_kv_pool`；在凑满前，这 1~3 个 token 的状态暂存在 `compress_state_pools[layer]` 这个环形缓冲里。

### 4.2 ring_size：环的深度

`get_compress_state_ring_size`（[deepseek_v4_memory_pool.py:34-50](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L34-L50)）：

| 压缩比 | 普通 | 投机解码 (MTP) | 在线 c128 |
|---|---|---|---|
| 4 (c4) | 8 | 16 | — |
| 128 (c128) | 128 | 256 | **1** |

要点：
- 投机解码时 ring_size 翻倍（16/256），因为 MTP 一步会推进多个 draft token，需要更深的环避免覆盖未消费的态。
- **在线 c128**（环境变量 `SGLANG_OPT_USE_ONLINE_COMPRESS`）把 ring_size 压到 1：只保留单份 `(max, sum, kv)` 在线归约态，不再保留 128 槽原始 token 环。但每槽字节更大（`3*head_dim` vs `2*head_dim`），整体仍是巨大压缩（~3/256x）。
- 在线 c128 **默认不支持投机解码**，但现在有一个实验开关放行：`SGLANG_EXPERIMENTAL_ONLINE_C128_MTP` + EAGLE（topk=1）时允许（[pool_configurator.py:834-858](../../../python/sglang/srt/model_executor/pool_configurator.py#L834-L858)）。实现方式是给在线状态开**多份 bank**：bank 0 是已提交状态，bank 1..N 缓存每个 draft token 的前缀状态，等 target verify 后再懒提交（[deepseek_v4_compress_state.py:108-117](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L108-L117)），bank 偏移由 `online_mtp_state_slot_offset` / `get_online_c128_mtp_state_slot_offset`（[deepseek_v4_memory_pool.py:988](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L988)）给出，未被接受的 draft 状态由 `clear_unaccepted_c128_draft_states`（[:1026](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1026)）清掉。这条路仍标注"Validate correctness carefully"，生产默认关。

### 4.2.1 ring 容量反过来限制 draft token 数

新增约束 `_assert_ring_serves_draft_tokens`（[pool_configurator.py:860-878](../../../python/sglang/srt/model_executor/pool_configurator.py#L860-L878)）：一次 verify batch 会把它整条乐观尾巴都写进环，所以 **环深度就是 draft token 数的上界**：

```python
for compress_ratio, ring_size, num_layers in ((4, c4_ring_size, num_layers_ca4),
                                              (128, c128_ring_size, num_layers_ca128)):
    if num_layers == 0: continue
    if compress_ratio == 128 and SGLANG_OPT_USE_ONLINE_COMPRESS: continue
    max_draft_tokens = get_compress_state_write_pad(compress_ratio, ring_size)
    assert num_draft_tokens <= max_draft_tokens
```

`get_compress_state_write_pad`（[deepseek_v4_memory_pool.py:51-58](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L51-L58)）给出"该 ratio+ring 能服务的最大 draft 数"。开自适应投机（多档 num_steps）时，这里校验的是**最大档**（`max_speculative_num_draft_tokens`），因为环只在启动期定一次尺寸。

### 4.3 `KVAndScore`：一块 buffer 切两半

`CompressStatePool` 的底层存储是一个 `KVAndScore`（[deepseek_v4_compress_state.py:22-80](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L22-L80)）：单个张量 `[size, last_dim]`，最后一维**前半是 kv、后半是 score**：

```
kv_score: [size, last_dim]
          ┌───────────── last_dim ──────────────┐
          │  kv (前半)         │  score (后半)    │
          │  item_size         │  item_size      │
```

- `clear()`（[:57-59](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L57-L59)）：`kv.zero_()` + `score.fill_(-inf)`（score 初始为 -inf，便于 max 归约）。
- buffer 用 `torch.empty` 分配（[_alloc_kv_score_buffer](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L146)，套在 memory-saver / custom mem pool 上下文里），**内容是未初始化的**。因此构造末尾要做初始清零（[:134-144](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L134-L144)）：
  - **HIP 且 ratio==128**：`kv_score_buffer.clear()` 清**整个** buffer。原因是请求级 c128 状态按 `req_pool_idx` 寻址，冷启动时某个请求槽还没被写过就可能被读到未初始化的半成品状态。
  - 其他情况：只 `kv_score_buffer[-1].clear()`，即末槽当"哨兵槽"，承接无效 loc（-1）的写入；`set_state_by_state_loc`（[:216](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L216)）每次写完都会重新 clear 末槽。

### 4.4 `last_dim` 的两种形态

[deepseek_v4_compress_state.py:108-128](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L108-L128)：

| 模式 | last_dim | 含义 |
|---|---|---|
| 在线 c128 (`online=True`) | `3 * head_dim` | (max, sum, kv) 三份在线归约态，ring_size=1 |
| 普通 (overlap, c4) | `2 * (1+overlap) * head_dim` = `4*head_dim` | overlap 压缩需要双份 (kv,score) |
| 普通 (非 overlap, c128) | `2 * 1 * head_dim` = `2*head_dim` | 单份 (kv, score) |

`overlap = (ratio == 4)`：c4 层用 **overlap 压缩**，状态宽度翻倍。

注意 `KVAndScore` 把 `last_dim` 对半切，所以"kv 部分"宽度 = `last_dim/2`，即 overlap 时 = `2*head_dim`，非 overlap 时 = `head_dim`。

非 online 路径还有两处细节：
- **行数向上对齐**：`pad_to = ratio`，若请求了 `state_cache_page_size > 1` 则 `pad_to = lcm(ratio, state_cache_page_size)`，`_size` 向上取整到 `pad_to` 的倍数（[:119-127](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L119-L127)）。
- **3-D 视图**：`state_cache_3d`（[:179-196](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L179-L196)）把扁平 buffer 视为 `[block_num, page_size, last_dim]`，正好是融合压缩算子想要的 `2*coff*D` 布局（kv 在 `[:, :, :coff*D]`，score 在后半）。仅非 online 且 `page_size>1` 时可用。

### 4.5 两种状态寻址：SWA 页级（c4）vs 请求级（c128）

这是本次校准里**最重要的一处更正**：c4 和 c128 的状态环用**不同的寻址方式**。

**① c4：按 SWA 页寻址** —— `translate_from_swa_loc_to_state_loc`（[deepseek_v4_compress_state.py:198-204](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L198-L204)）：

```python
swa_pages = swa_loc // swa_page_size
state_loc = swa_pages * ring_size + (swa_loc % ring_size)
state_loc = where(swa_loc < 0, -1, state_loc)   # 无效 loc → 哨兵
```

即：状态环按"SWA 页"分段，每个 SWA 页对应环里 `ring_size` 个槽，页内用 `swa_loc % ring_size` 轮转。

**② c128：按请求寻址** —— `translate_from_req_position_to_state_loc`（[deepseek_v4_compress_state.py:206-211](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L206-L211)）：

```python
state_loc = req_pool_indices * ring_size + positions % ring_size
state_loc = where(positions < 0, -1, state_loc)
```

c128 一个压缩块跨 128 个 token，跨越很多 forward step，用"页"当键会在页边界丢态；改用 `req_pool_idx` 当键后，状态跟着请求槽走，生命周期与请求一致。选择哪条翻译在 `create_paged_compressor_data`（[compressor.py:251-290](../../../python/sglang/srt/layers/attention/dsv4/compressor.py#L251-L290)）里分叉：c128 走请求级（[:275](../../../python/sglang/srt/layers/attention/dsv4/compressor.py#L275)），c4 走 SWA 页级（[:280](../../../python/sglang/srt/layers/attention/dsv4/compressor.py#L280)）。

**副作用：请求槽复用必须显式清态。** PD 分离的 decode 侧复用 req 槽时要调 `clear_c128_req_state`（[deepseek_v4_memory_pool.py:1007](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1007)），否则会读到上一个请求的残留压缩态。

### 4.6 状态池的大小

来自 `DSV4PoolConfigurator`（见 §5），两档来源不同：

```
# c4：页级 → 由 token 容量推导（_compute_dsv4_sizes）
c4_state_pool_size   = swa_tokens // swa_page_size * c4_ring_size

# c128：请求级 → 由并发数推导（finalize_with_max_running_requests）
c128_state_pool_size = num_req_slots * c128_ring_size          # 离线
                     = num_req_slots                            # 在线 (ring=1)
```

> ⚠️ 旧版文档写的 `c128_state_pool_size = swa_tokens // swa_page_size * c128_ring_size` 已错误。`_compute_dsv4_sizes`（[pool_configurator.py:922-932](../../../python/sglang/srt/model_executor/pool_configurator.py#L922-L932)）现在返回 `c128_state_pool_size=0`，真值在 `finalize_with_max_running_requests`（[:994-1003](../../../python/sglang/srt/model_executor/pool_configurator.py#L994-L1003)）里等 `max_running_requests` 确定后才填。

`num_req_slots`（[`_get_num_req_slots`](../../../python/sglang/srt/model_executor/pool_configurator.py#L934)）= `max_running_requests + 1`，PD decode 模式再加 `disaggregation_decode_extra_slots`。

c4 层的 `indexer_compress_state_pools` 复用相同 size 和 ring_size，但 head_dim 用 `indexer_head_dim`（128）而非注意力的 `qk_nope+qk_rope`（`_make_indexer_state_pool`，[deepseek_v4_memory_pool.py:907](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L907)）。

两个状态池的 dtype 也分开配（`_make_attn_state_pool`，[:885-905](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L885-L905)）：

```python
CompressStatePool(
    dtype = self.c4_state_dtype if ratio == 4 else self.c128_state_dtype,
    online = (ratio == 128 and SGLANG_OPT_USE_ONLINE_COMPRESS),
    online_mtp_max_draft_tokens = (self.online_mtp_max_draft_tokens if ratio == 128 else 0),
    ...)
```

---

## 5. 显存容量计算：`DSV4PoolConfigurator`

### 5.1 系数模型的统一框架

SGLang 用统一的"系数 + 偏置"模型给所有架构算 KV 容量：

```
available_bytes = max_tokens * coeff + bias
max_tokens      = (available_bytes - bias) / coeff
```

DSV4 用 `DSV4PoolConfigurator`（[pool_configurator.py:755](../../../python/sglang/srt/model_executor/pool_configurator.py#L755)），其中 `coeff = bytes_per_full_token`、`bias = 0`（c128 状态那部分固定开销不进 bias，而是**直接从 `available_bytes` 里先扣掉**，见 §5.4）。工厂选择逻辑（[pool_configurator.py:1045-1056](../../../python/sglang/srt/model_executor/pool_configurator.py#L1045-L1056)）：

```python
def create_memory_pool_configurator(kvc: KVCacheConfigurator) -> MemoryPoolConfigurator:
    if is_deepseek_v4(kvc.model_config.hf_config) and kvc.is_hybrid_swa:
        return DSV4PoolConfigurator(kvc)          # ← V4 走这条
    if kvc.is_hybrid_swa:
        if SWAChunkCapPoolConfigurator.is_applicable(kvc):
            return SWAChunkCapPoolConfigurator(kvc)
        return HybridSWAPoolConfigurator(kvc)
    return DefaultPoolConfigurator(kvc)            # MHA/MLA/DSA/FP4
```

注意：DSV4 必然是 hybrid SWA（每层都有 SWA），所以走专用 configurator。工厂现在只接一个 `KVCacheConfigurator`（不再是旧的 `ModelRunner`）。

### 5.2 `bytes_per_full_token`：每个"完整 token"的总成本

这是整套容量计算的核心系数。它把一个逻辑 token 在**所有子池**里占用的字节加总（[pool_configurator.py:880-920](../../../python/sglang/srt/model_executor/pool_configurator.py#L880-L920)）。逐项拆解：

```
bytes_per_full_token =
    swa_ratio * kv_bytes * num_layers_total          # ① SWA 全精度池
  + (1/(4*shrink)) * kv_bytes * num_layers_ca4        # ② c4 压缩 KV
  + (1/128) * kv_bytes * num_layers_ca128             # ③ c128 压缩 KV
  + (1/4) * indexer_bytes * num_layers_ca4            # ④ c4 indexer index_k
  + swa_ratio * c4_state_ratio * c4_state_bytes * num_layers_ca4         # ⑤ c4 注意力状态环
  + c128_state_ratio * c128_state_bytes * num_layers_ca128                # ⑥ c128 状态环 ← 系数为 0
  + swa_ratio * c4_state_ratio * c4_indexer_state_bytes * num_layers_ca4 # ⑦ c4 indexer 状态环
```

| 项 | 对应子池 | 系数解读 |
|---|---|---|
| ① | swa_kv_pool | SWA 只保留近窗，`swa_ratio`（默认 0.1）缩放全序列；每层都有 |
| ② | c4_kv_pool | 每 4 token 一槽 → 1/4；HiSparse 再除以 `shrink` |
| ③ | c128_kv_pool | 每 128 token 一槽 → 1/128 |
| ④ | c4_indexer_kv_pool | indexer 也是 1/4 粒度 |
| ⑤ | compress_state_pools(c4) | 状态环大小 ∝ swa_tokens（`swa_ratio` 缩放）× ring/swa_page |
| ⑥ | compress_state_pools(c128) | **`c128_state_ratio = 0`**：c128 状态是请求级的，不随 token 容量伸缩，改为固定项单独结算（§5.4）|
| ⑦ | indexer_compress_state_pools | 同 ⑤ 但用 indexer head_dim |

各分量字节：

```python
kv_bytes     = qk_nope_head_dim + qk_rope_head_dim*2 + 8      # = 584 (与池布局一致)
indexer_bytes= indexer_head_dim + indexer_head_dim//128 * 4   # = 132
attn_head_dim= qk_nope_head_dim + qk_rope_head_dim            # = 512

# dtype size 由 SGLANG_DSV4_COMPRESS_STATE_DTYPE 决定 (默认 float32 → 4；可选 bfloat16 → 2)
c4_state_dtype_size, c128_state_dtype_size = _get_dsv4_compress_state_dtype_sizes()

c4_state_bytes        = 2 * 2 * attn_head_dim * c4_state_dtype_size    # overlap 双份 (kv,score)
c128_state_bytes      = (3 if c128_online else 2*1) * attn_head_dim * c128_state_dtype_size
c4_indexer_state_bytes= 2 * 2 * indexer_head_dim * c4_state_dtype_size
c4_state_ratio  = c4_ring_size / swa_page_size
c128_state_ratio= 0                                                    # ← 见 ⑥
```

> ⚠️ 旧版文档把 `c*_state_bytes` 里的 dtype size 写成常量 4。现在它来自 `_get_dsv4_compress_state_dtype_sizes()`（[pool_configurator.py:104](../../../python/sglang/srt/model_executor/pool_configurator.py#L104)），c4 / c128 两档**分别**取值。

### 5.3 投机解码（MTP）的显存预留

如果开了投机解码，`bytes_per_full_token` 会被放大，为 draft worker 预留显存（[pool_configurator.py:821-828](../../../python/sglang/srt/model_executor/pool_configurator.py#L821-L828)）：

```python
if self.is_speculative:
    draft_layers = 1
    target_layers = self.num_layers_total
    self.bytes_per_full_token *= (target_layers + draft_layers) / target_layers
```

思路与 dflash 的 `scale_kv_cell_size_per_token_for_dflash` 一致：在 per-token 字节上乘 `(T+D)/T`，等价于把 token 容量按比例缩小给草稿层让路。此外投机解码还会触发 §4.2.1 的 ring 容量校验。

### 5.4 从 `bytes_per_full_token` 反推各池大小

`calculate_pool_sizes`（[pool_configurator.py:1005-1033](../../../python/sglang/srt/model_executor/pool_configurator.py#L1005-L1033)）现在分**三步**，关键是 c128 状态那块固定开销要先扣：

```python
assert page_size % 128 == 0

# 第 1 步：估 c128 状态固定字节
if requested_max_running_requests_per_worker is not None:
    c128_state_fixed_bytes = _get_c128_state_fixed_bytes(requested_mrr)
else:
    # 用户没显式给并发数 → 先用未扣除的 token 容量反推一个并发估计
    full_token = int(available_bytes / bytes_per_full_token)
    c128_state_fixed_bytes = _get_c128_state_fixed_bytes_for_token_capacity(full_token)

# 第 2 步：扣掉后再切 token
available_bytes_for_tokens = max(available_bytes - c128_state_fixed_bytes, 0)
full_token = int(available_bytes_for_tokens / bytes_per_full_token)

# 第 3 步：按比例分派各池（_compute_dsv4_sizes）
full_token = full_token // page_size * page_size
swa_tokens = int(full_token * swa_ratio) // page_size * page_size
c4_max_total_num_tokens   = full_token // (4 * c4_shrink_factor)
c128_max_total_num_tokens = full_token // 128
c4_state_pool_size        = swa_tokens // swa_page_size * c4_ring_size
c128_state_pool_size      = 0        # ← 留给 finalize_with_max_running_requests
```

`_get_c128_state_fixed_bytes`（[:939-959](../../../python/sglang/srt/model_executor/pool_configurator.py#L939-L959)）就是把 §4.6 的请求级尺寸换算成字节：

```python
num_req_slots = max_running_requests (+ decode extra slots) + 1
if online:
    state_rows = (num_req_slots + c128_ring_size + 1) * (1 + online_c128_mtp_max_draft_tokens)
    state_last_dim = 3 * attn_head_dim
else:
    state_rows = ceil_div(num_req_slots*c128_ring_size + c128_ring_size + 1, 128) * 128
    state_last_dim = 2 * attn_head_dim
return state_rows * state_last_dim * c128_state_dtype_size * num_layers_ca128
```

没显式给 `--max-running-requests` 时用启发式估（[:961-972](../../../python/sglang/srt/model_executor/pool_configurator.py#L961-L972)）：`estimated = token_capacity / context_len * 512`，clamp 到 `[2048, 4096]`，再取 `min(estimated, token_capacity // 2)`。

最后等真正的 `max_running_requests` 定下来，`finalize_with_max_running_requests`（[:994-1003](../../../python/sglang/srt/model_executor/pool_configurator.py#L994-L1003)）回填 `c128_state_pool_size`。

容量关系一图流：

```
                  available_bytes
                        │
                        │ − c128_state_fixed_bytes  (请求级固定开销, ∝ max_running_requests)
                        ▼
              available_bytes_for_tokens
                        │  ÷ bytes_per_full_token
                        ▼
                  full_token  ──────────────┬─────────────┬──────────────┐
                   │                         │             │              │
       × swa_ratio │              ÷(4·shrink)│        ÷128 │   ÷swa_page  │
                   ▼                         ▼             ▼   ×c4_ring   ▼
              swa_tokens                c4_tokens    c128_tokens   c4 state pool
            (SWA 池容量)              (c4 池容量)   (c128 池容量)  (+ indexer state)

  max_running_requests ──► finalize_with_max_running_requests ──► c128_state_pool_size
```

> `swa_ratio` 即 `--swa-full-tokens-ratio`，V4 默认 **0.1**（见 §12）。它同时决定 SWA 池大小和 c4 状态环大小。

另有 `calculate_pool_sizes_from_max_tokens`（[:1035-1042](../../../python/sglang/srt/model_executor/pool_configurator.py#L1035-L1042)）：用户直接给 `--max-total-tokens` 时跳过字节推导，直接进 `_compute_dsv4_sizes`（此时不扣 c128 固定项，因为 token 数是外部指定的）。

### 5.5 硬约束：page_size 必须是 128 的倍数

`calculate_pool_sizes` / `calculate_pool_sizes_from_max_tokens` 开头都断言（[pool_configurator.py:1008-1010](../../../python/sglang/srt/model_executor/pool_configurator.py#L1008-L1010)）：

```python
assert page_size % 128 == 0, "page_size must be multiple of 128 for compressed attention"
```

因为 c128 池 page_size = `page_size // 128`，必须整除。V4 默认 `page_size=256` 满足。

---

## 6. 分配与地址翻译

### 6.1 谁来分配：Allocator 选择

> ⚠️ 旧版本文档把 allocator 选择指向 `model_executor/model_runner_kv_cache_mixin.py`，该文件已整体搬到 **`mem_cache/kv_cache_configurator.py`**，选择逻辑在 [kv_cache_configurator.py:1713-1774](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1713-L1774)。

选择是一条 if-else 链（非 NPU 分支）：

| 条件 | Allocator |
|---|---|
| `is_hybrid_swa` 且 `full_max_total_num_tokens == 0` | `PureSWATokenToKVPoolAllocator`（[#L1715](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1715)）|
| `is_hybrid_swa`（**V4 走这条**）| `SWATokenToKVPoolAllocator`（[#L1724](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1724)）|
| 非 hybrid + `enable_hisparse` | `HiSparseTokenToKVPoolAllocator`（[#L1740](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1740)）|
| 非 hybrid + `page_size==1` 且无 DCP | `TokenToKVPoolAllocator` |
| 其余 | `PagedTokenToKVPoolAllocator` |

`SWATokenToKVPoolAllocator` 管理"full（逻辑全序列）token slot"的分配，并维护 `full_to_swa_index_mapping`（逻辑 full loc → SWA 物理 loc），对外返回的句柄恒为 **full loc**。

之后有两个统一的后置装饰（[#L1770-L1779](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1770-L1779)）：

```python
if get_memory().enable_hisparse and is_dsv4_model:
    assert self.is_hybrid_swa, "DeepSeek V4 HiSparse requires SWA mode."
    token_to_kv_pool_allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator(
        token_to_kv_pool_allocator)          # 装饰器包裹基础 allocator（见 §9）

# DSV4-NPU：让 req_to_token_pool.free(req) 能顺带释放 c4/c128 页
if hasattr(req_to_token_pool, "register_dsv4_allocator"):
    req_to_token_pool.register_dsv4_allocator(token_to_kv_pool_allocator)
```

注意 V4+HiSparse 是 **hybrid 分支的 `SWATokenToKVPoolAllocator` 再被 DSV4 装饰器包一层**，而不是走上表里那条非 hybrid 的 `HiSparseTokenToKVPoolAllocator`——后者服务的是 DSA（非 V4）模型。

### 6.2 三级地址空间

V4 的 KV 地址要在三个空间之间翻译：

```
┌─────────────┐  translate_loc_from_full_to_swa   ┌─────────────┐
│  full loc   │ ────────────────────────────────► │  swa loc    │
│ (逻辑全序列) │   full_to_swa_index_mapping[idx]   │ (SWA 物理)   │
└─────────────┘                                    └─────────────┘
       │                                                  │
       │ metadata kernel:                                 │ translate_from_swa_loc_to_state_loc
       │ raw_out_loc // 4 (gated)                         ▼
       │ raw_out_loc // 128 (gated)                ┌─────────────┐
       ▼                                           │ state loc   │
┌─────────────┐                                    │ (压缩态环)   │
│ c4/c128 loc │                                    └─────────────┘
│ (压缩物理)   │
└─────────────┘
```

| 翻译 | 函数 | 公式 |
|---|---|---|
| full → swa | `translate_loc_from_full_to_swa` | `full_to_swa_index_mapping[idx]` |
| full → c4 | metadata kernel `c4_out_loc` | `raw_out_loc // 4`（仅当 `seq_len%4==0`）|
| full → c128 | metadata kernel `c128_out_loc` | `raw_out_loc // 128`（仅当 `seq_len%128==0`）|
| swa → c4 state | `translate_from_swa_loc_to_state_loc` | `swa_loc//swa_page_size*ring + swa_loc%ring` |
| req → c128 state | `translate_from_req_position_to_state_loc` | `req_pool_idx*ring + position//128%ring`（**请求级**，见 §4.5）|
| full → c4(HiSparse device) | `translate_loc_from_full_to_compressed` | `(idx+1)%4==0` 保留, `//4` |

### 6.3 Triton 元数据核：计算压缩 out_loc 与 position

> ⚠️ 文件已从 `layers/attention/dsv4/metadata_kernel.py` 搬到 **`python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py`**。

写入压缩池前，需要为每个 batch 元素算出"压缩后的写入位置 + 压缩后的 RoPE position"。这由 `_init_compressed_attn_metadata_kernel`（[metadata_kernel.py:8-88](../../../python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py#L8-L88)）一次性算好，公开入口是 `init_compression_metadata`（[#L192](../../../python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py#L192)）：

```python
is_write_token = batch_id < num_write_tokens         # CP-v2 padding 行掩掉
raw_out_loc = tl.load(raw_out_loc_ptr + batch_id, mask=is_write_token, other=0)

# c4 档
c4_should_compress = (seq_len % 4) == 0              # 只有凑满 4 的边界才落压缩块
c4_out_loc   = tl.where(c4_should_compress, raw_out_loc // 4, 0)
c4_positions = position & (~3)                       # 对齐到 4 的下界 (清低 2 位)
c4_seq_lens_raw    = seq_len // 4
c4_seq_lens_clamp1 = tl.maximum(c4_seq_lens_raw, 1)  # 供 kernel 用的下限-1 长度

# c128 档
c128_should_compress = (seq_len % 128) == 0
c128_out_loc   = tl.where(c128_should_compress, raw_out_loc // 128, 0)
c128_positions = position & (~127)                   # 对齐到 128 的下界 (清低 7 位)
c128_seq_lens_raw    = seq_len // 128
c128_seq_lens_clamp1 = tl.maximum(c128_seq_lens_raw, 1)
```

要点：
- **门控写入**：只有当 `seq_len` 是压缩比整数倍时（凑满一个压缩块），`out_loc` 才是真实位置，否则置 0（配合哨兵逻辑，不污染有效槽）。这实现了"每 4/128 个 token 才落一个压缩槽"。
- **position 对齐**：压缩块的 RoPE position 取块下界（`& ~3` / `& ~127`），保证一个压缩块内所有原始 token 用同一个对齐 position。
- **`*_seq_lens_clamp1`**（[#L44](../../../python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py#L44) / [#L55](../../../python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py#L55)）：`max(seq_len//ratio, 1)`。短序列压缩后长度可能是 0，而下游 attention kernel 不接受 0 长度；clamp 版专供 kernel 用，raw 版保留真实值用于 mask。
- **`num_write_tokens` 掩码**（[#L37](../../../python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py#L37) + 注释 [#L109-L115](../../../python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py#L109-L115)）：CP-v2 会给 attention 元数据补 padding 行，但这些行**没有 cache 写入位置**。做法是让写出 buffer（`c4_out_loc` / `c128_out_loc`）只开 `num_write_tokens = raw_out_loc.shape[0]` 行、其余"每 batch 行"的输出仍开 `bs` 行，kernel 内用 `is_write_token` 掩掉超出的行。断言 `num_write_tokens <= bs`。
- kernel 同时生成 `c128_page_indices`（c128 稀疏页表，[#L62-L87](../../../python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py#L62-L87)），把 c128 粒度的 page_table 展开成扁平页索引，越界位置填 `-1`，供注意力读取。要求 `page_size >= 128 且 page_size % 128 == 0`。

### 6.4 `set_swa_key_buffer_radix_fused_norm_rope`：写 SWA 的融合算子

模型写 SWA 时不是分步做 rmsnorm→rope→写缓存，而是一个融合 kernel 一把梭（[deepseek_v4_memory_pool.py:1180-1199](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1180-L1199)）：

```python
def set_swa_key_buffer_radix_fused_norm_rope(self, layer_id, swa_loc, kv, kv_weight,
                                             eps, freqs_cis, positions):
    fused_k_norm_rope_flashmla(                # rmsnorm + rope + 写缓存 融合
        kv=kv, kv_weight=kv_weight, eps=eps, freqs_cis=freqs_cis,
        positions=positions, out_loc=swa_loc,
        kvcache=self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)],
        page_size=self.swa_kv_pool.page_size)
```

> ⚠️ **接口已变（重要）**：该函数**不再自己做 full→swa 翻译**，参数直接就是 `swa_loc`。翻译责任上移到调用方：模型层 `MQALayer._compute_kv_to_cache`（[models/deepseek_v4.py:981-983](../../../python/sglang/srt/models/deepseek_v4.py#L981-L983)）传 `swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch)`。

翻译结果的复用（旧文档里的 `SGLANG_OPT_CACHE_SWA_TRANSLATION` 开关**已删除**）现在落在后端的 `get_swa_out_cache_loc`（[deepseek_v4_backend.py:1635-1656](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1635-L1656)）里，是三级取值：

1. **元数据初始化时算好的 `core_attn_metadata.swa_out_cache_loc`**（[#L1038-L1070](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1038-L1070)）——decode/verify 在 graph 内算，所有层共享一份，天然省掉每层重复 gather；
2. draft-extend 走提到 graph 外的 `cuda_graph_swa_out_cache_loc` 常驻 buffer（[#L1579](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1579)、`_fill_cuda_graph_swa_out_cache_loc` [#L1076](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1076)）；
3. 兜底当场翻译 `translate_loc_from_full_to_swa(out_cache_loc)`——用于跳过元数据初始化的路径、init 之后又被重新 padding 的 batch，以及 **idle**（idle 的元数据可能是陈旧的，且翻译全 0 padding 的 `out_cache_loc` 会写到哑槽）。命中缓存的条件是 `cached is not None and not is_idle() and cached.shape[0] == out_cache_loc.shape[0]`。

unified_kv（HIP 融合布局，§3.6）有对应的 `set_unified_key_buffer_radix_fused_norm_rope`（[#L1201-L1227](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1201-L1227)）：此时 `swa_kv_pool is None`，SWA K 住在共享的 bf16 unified ring 里，所以拆成 `fused_norm_rope_inplace` + `scatter_bf16_into_unified`，且 `swa_loc < 0` 的行（未提交的 verify token）由 scatter 直接跳过。

---

## 7. 写入路径：三股数据流入 Cache

一个 c4 层在一次 forward 中，会向 cache 写入**三份**数据；c128 层写两份；纯 SWA（ratio=0）层只写一份。

```
                          MQALayer forward
                                │
        ┌───────────────────────┼───────────────────────────┐
        ▼                       ▼                            ▼
 ① SWA 全精度 KV         ② 压缩注意力 KV               ③ Indexer index_k
 (每层都写)             (c4/c128 层)                  (仅 c4 层)
        │                       │                            │
 _compute_kv_to_cache   forward_core_compressor     forward_indexer_compressor
        │                       │                            │
 set_swa_key_buffer_     set_extra_key_buffer_       set_index_k_fp4 /
 radix_fused_norm_rope   fused                       set_index_k_fused
        │                       │                            │
        ▼                       ▼                            ▼
   swa_kv_pool           c4_kv_pool / c128_kv_pool    c4_indexer_kv_pool
```

### 7.1 ① 写 SWA 全精度池

由模型层 `MQALayer._compute_kv_to_cache`（[models/deepseek_v4.py:952-989](../../../python/sglang/srt/models/deepseek_v4.py#L952-L989)）触发：算出 `kv = wkv(x)`（或从融合的 `qkv_a` 切出后半段），调 `set_swa_key_buffer_radix_fused_norm_rope`，融合 rmsnorm+RoPE+写缓存（见 §6.4）。写入位置由后端给出：`swa_loc = attn_backend.get_swa_out_cache_loc(forward_batch)`。

有一条**替代精度路径** `SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE`（[#L965-L973](../../../python/sglang/srt/models/deepseek_v4.py#L965-L973)）：融合 kernel 是从 fp32 寄存器直接量化到 fp8 的，而 bf16 舍入会把值挪过 fp8 的分桶边界，跟"以 bf16 为源"的消费方对不上位。开这个开关后改走 `_compute_kv_bf16`（[#L991](../../../python/sglang/srt/models/deepseek_v4.py#L991)，先 bf16 norm+rope）再 `attn_backend.store_cache`（[deepseek_v4_backend.py:1658](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1658) → `set_swa_key_buffer_radix_fused`）。DSA prefill-CP 也走 `_compute_kv_bf16`，因为跨 rank all-gather 需要 bf16 中间量。

> **每一层都执行这一步**——SWA 是 V4 的"基础底座"，无论该层压缩比是多少，近窗 token 总是以全精度存在 SWA 池里。

### 7.2 ② 写压缩注意力 KV（c4 / c128）

由注意力后端 `CompressorBackendMixin.forward_core_compressor`（[compressor.py:163-190](../../../python/sglang/srt/layers/attention/dsv4/compressor.py#L163-L190)）触发：

```python
if forward_batch.forward_mode.is_idle():
    return                                     # idle batch 不写
new_compressed_kv = compressor(x, forward_batch, attn_backend=self)   # 压缩
core_metadata = self.forward_metadata.core_metadata
out_loc = core_metadata.c4_out_loc if compressor.ratio == 4 else core_metadata.c128_out_loc
if out_loc.shape[0] > new_compressed_kv.shape[0]:
    out_loc = out_loc[: new_compressed_kv.shape[0]]     # CP/padding 截断
token_to_kv_pool.set_extra_key_buffer_fused(layer_id=layer_id, loc=out_loc,
                                            cache_k=new_compressed_kv)
```

> ⚠️ **写入已无分支**：旧文档里 `SGLANG_OPT_USE_FUSED_STORE_CACHE` 控制的"融合写 vs 显式 `quant_to_nope_fp8_rope_bf16_pack` + `set_extra_key_buffer`"两条路已合并——环境变量删除，**只剩融合写 `set_extra_key_buffer_fused`**。

- `out_loc` 来自 §6.3 的 Triton 元数据核（`raw_out_loc // ratio`，门控在压缩边界）。
- `set_extra_key_buffer_fused` 通过 `layer_mapping` 路由到 `c4_kv_pool` 或 `c128_kv_pool`（[deepseek_v4_memory_pool.py:1229-1237](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1229-L1237) → 子池 `set_key_buffer_fused` [#L244](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L244)）。
- 压缩本身（`compress_forward` + `compress_fused_norm_rope_inplace`；HIP 走 `hip_compress_forward` + `hip_compress_fused_norm_rope[_hadamard]_inplace`）发生在 `forward_compress`（[compressor.py:68-161](../../../python/sglang/srt/layers/attention/dsv4/compressor.py#L68-L161)），用到 §4 的 `CompressStatePool` 累积在途态。`coff = 2 if is_overlap_compress(ratio) else 1`（c4 双份 / c128 单份）；在线 c128 时 buffer 视图变成 `(-1, 1, head_dim*3)`。

### 7.3 ③ 写 Indexer index_k（仅 c4 层）

由 `forward_indexer_compressor`（[compressor.py:192-219](../../../python/sglang/srt/layers/attention/dsv4/compressor.py#L192-L219)）触发，**只在 c4（overlap）层**：

```python
assert is_overlap_compress(compressor.ratio)   # 仅 ratio==4
new_compressed_kv = compressor(x, forward_batch, attn_backend=self)
out_loc = self.forward_metadata.core_metadata.c4_out_loc[: new_compressed_kv.shape[0]]
if self.enable_deepseek_v4_fp4_indexer:
    token_to_kv_pool.set_index_k_fp4(layer_id, loc=out_loc, cache_k=new_compressed_kv)
else:
    token_to_kv_pool.set_index_k_fused(layer_id, loc=out_loc, cache_k=new_compressed_kv)
```

> ⚠️ **分支含义变了**：不再是"融合 vs 量化+写"，而是 **FP4 indexer vs 默认 FP8 indexer** 两种布局（68 B/token vs 132 B/token，见 §3.3）。

index_k 写入 `c4_indexer_kv_pool`（`set_index_k_fp4` [#L1249](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1249) / `set_index_k_fused` [#L1239](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1239)，两者都 `assert compress_ratio == 4`）。它是稀疏选择打分用的轻量索引向量，不参与最终注意力的 value 计算（见 §8）。

### 7.4 写入路径相关的环境变量

| 环境变量 | 状态 | 作用 |
|---|---|---|
| `SGLANG_OPT_USE_ONLINE_COMPRESS` | 在用（默认关，[environ.py:1312](../../../python/sglang/srt/environ.py#L1312)）| c128 在线归约（`ring_size=1`，`last_dim=3*head_dim`）；配 `SGLANG_EXPERIMENTAL_ONLINE_C128_MTP` 才允许叠 MTP（见 §4.2）|
| `SGLANG_DSV4_COMPRESS_STATE_DTYPE` | 在用（默认 `float32`）| 压缩态 dtype，可切 `bfloat16` |
| `SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE` | 在用 | SWA nope 量化以 bf16 为源（见 §7.1）|
| `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` | 在用（ROCm 上被 hook 强制关，[deepseek_v4_hook.py](../../../python/sglang/srt/arg_groups/deepseek_v4_hook.py)）| 稀疏 prefill kernel |
| ~~`SGLANG_OPT_USE_FUSED_STORE_CACHE`~~ | **已删除** | 融合写已成为唯一路径 |
| ~~`SGLANG_OPT_CACHE_SWA_TRANSLATION`~~ | **已删除** | 翻译复用改由 `get_swa_out_cache_loc` 的元数据缓存承担（见 §6.4）|

---

## 8. 读取路径：C4Indexer 稀疏选择 + FlashMLA

V4 注意力不是对全序列做 dense attention，而是"**近窗全读 + 远程稀疏选读**"。读取分两步：先由 C4Indexer 选出要读哪些压缩块，再由 FlashMLA 在选中的块上算注意力。

### 8.1 C4Indexer：选 top-512/1024 压缩块

`forward_c4_indexer`（[indexer.py:637-873](../../../python/sglang/srt/layers/attention/dsv4/indexer.py#L637-L873)），**仅 c4 层**执行（idle batch 直接 return）：

```
① _forward_prepare_normal / _forward_prepare_multi_stream
       → q_indexer (fp8 单张量；fp4 时是 (q_fp4, q_sf) 二元组) + weights
② 打分：logits = fn(q, index_k cache, weights, c4_seq_lens, page_table, ...)
       paged 路径读 c4_indexer_kv_pool；符合条件时走 nonpaged 快路
③ topk_transform_512*(logits, ...) → core_metadata.c4_sparse_page_indices
④ [HiSparse] decode: swap_in_selected_pages 把选中页从 host 换入 device
              非 decode: translate_loc_to_hisparse_device 直接翻译
```

关键点：

- **index_k cache 视图**（[#L770-L777](../../../python/sglang/srt/layers/attention/dsv4/indexer.py#L770-L777)）：`get_index_k_with_scale_buffer` 返回二维张量，reshape 成 `[num_pages, 64, 1, head_dim_with_sf]`，其中
  ```python
  head_dim_with_sf = 68 if use_fp4_indexer else 132
  ```
  132 = 128 `index_head_dim` + 4 B FP32 scale；68 = `128//2` + 4（FP4 半字节 + scale），见 §3.3。
- **打分核有 6 种选择**（[#L694-L728](../../../python/sglang/srt/layers/attention/dsv4/indexer.py#L694-L728)），按优先级：

  | 条件 | kernel |
  |---|---|
  | `use_fp4_indexer` | `deep_gemm.fp8_fp4_paged_mqa_logits`（与 tilelang 互斥，会直接报错）|
  | `SGLANG_OPT_USE_TILELANG_INDEXER` | `tilelang_fp8_paged_mqa_logits` |
  | `SGLANG_OPT_USE_AITER_INDEXER` | `_aiter_fp8_paged_mqa_logits` |
  | `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` | `fp8_paged_mqa_logits_torch[_sm120]` |
  | XPU | `sgl_kernel.fp8_paged_mqa_logits_triton` |
  | 默认 | `deep_gemm.fp8_paged_mqa_logits` |

- **nonpaged 快路**（`_can_use_nonpaged_indexer` [#L479](../../../python/sglang/srt/layers/attention/dsv4/indexer.py#L479)、`_get_nonpaged_indexer_plan` [#L518](../../../python/sglang/srt/layers/attention/dsv4/indexer.py#L518)、`_forward_nonpaged_indexer` [#L608](../../../python/sglang/srt/layers/attention/dsv4/indexer.py#L608)）：开关 `SGLANG_OPT_DSV4_NONPAGED_INDEXER`，把该请求的 c4 index_k 先 gather 成连续张量再打分，绕过 paged 寻址开销。门槛很严——仅 CUDA（排除 HIP/NPU）、`ForwardMode.EXTEND`、`batch_size == 1`、非改写/非 TBO 拆分/非 graph（含 piecewise/breakable/正在 capture）、非 CP、非 HiSparse、非 fp4/tilelang/aiter/torch 打分核，且 `query_rows >= SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS`。任一不满足就回退 paged。
- **topk 有 4 条实现**（[#L811-L846](../../../python/sglang/srt/layers/attention/dsv4/indexer.py#L811-L846)）：`dsa_topk_backend.is_torch()` → `topk_transform_512_pytorch_vectorized`；`is_flashinfer()` → `topk_transform_512_flashinfer_unfused`；`SGLANG_OPT_USE_TOPK_V2` 且不需要 `raw_indices` → `topk_transform_512_v2`（融合版，不产出原始索引）；否则默认 `topk_transform_512`。结果写入 `core_metadata.c4_sparse_page_indices`。topk 的目标数（512 或 1024）由模型规模决定，后端断言 `c4_sparse_topk in (512, 1024)`（[deepseek_v4_backend.py:381](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L381)）。
- **`raw_indices` 旁路**（[#L801-L809](../../../python/sglang/srt/layers/attention/dsv4/indexer.py#L801-L809)）：页号被 topk 翻译成物理索引前的"原始 top-k 结果"。三个消费者：indexer 抓取器（`get_global_indexer_capturer`，观测用）、HiSparse decode（换入需要原始页号）、`core_metadata.c4_sparse_raw_indices`。需要它时不能用 `topk_transform_512_v2`。
- index_k 只用于"**选块打分**"，不参与最终注意力的 value 计算——这是它能用低维（128 / FP4 64）轻量表示的原因。

### 8.2 FlashMLA：在选中块上算注意力

注意力后端 `DeepseekV4AttnBackend.forward`（[deepseek_v4_backend.py:1668](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1668)）汇总三路 cache 后调一次 FlashMLA：

```python
assert k is v, "DeepseekV4 shares k and v"

# ① SWA 近窗全精度 cache
swa_k_cache = get_swa_key_buffer_radix(layer_id)
swa_k_cache = swa_k_cache[:, :swa_window_size * total_dim].view(..., swa_window_size, 1, total_dim)

# ② 压缩 extra cache + 稀疏索引（按 compress_ratio 选源）
if compress_ratio == 4:
    extra_k_cache       = get_extra_key_buffer(layer_id)          # c4_kv_pool
    extra_indices       = core_attn_metadata.c4_sparse_page_indices  # ← Indexer 选出的
    extra_topk_lengths  = core_attn_metadata.c4_sparse_topk_lengths
elif compress_ratio == 128:
    extra_k_cache       = get_extra_key_buffer(layer_id)          # c128_kv_pool
    extra_indices       = core_attn_metadata.c128_page_indices    # ← 元数据核展开的页表
    extra_topk_lengths  = core_attn_metadata.c128_topk_lengths_clamp1

o = flash_mla.flash_mla_with_kvcache(
    q=q, k_cache=swa_k_cache, head_dim_v=head_dim_v,
    is_fp8_kvcache=True,
    indices=swa_page_indices, topk_length=swa_topk_lengths,    # SWA 近窗
    attn_sink=attn_sink,
    extra_k_cache=extra_k_cache,                                # 压缩远程
    extra_indices_in_kvcache=extra_indices,
    extra_topk_length=extra_topk_lengths)[0]
```

注意力的读取来源全景：

```
                       FlashMLA (单 kernel)
   ┌──────────────────────┬─────────────────────────────┐
   │  k_cache (主)        │  extra_k_cache (附)          │
   │  = SWA 近窗全精度     │  = c4 或 c128 压缩 KV        │
   │  indices=swa_page_    │  extra_indices=             │
   │    indices            │    c4_sparse_page_indices    │
   │  topk=swa_topk_lengths│    (c4: Indexer top-512)     │
   │                       │    c128_page_indices         │
   │                       │    (c128: 元数据核展开)       │
   └──────────────────────┴─────────────────────────────┘
              │                          │
        近窗 dense 全读            远程 sparse 选读
```

| compress_ratio | 主 cache | extra cache | extra 索引来源 |
|---|---|---|---|
| 0 | SWA only | 无 | — |
| 4 | SWA 近窗 | c4_kv_pool | C4Indexer 选的 `c4_sparse_page_indices`（top-512）|
| 128 | SWA 近窗 | c128_kv_pool | 元数据核展开的 `c128_page_indices`（全 c128 块）|

要点：
- c4 层是**真稀疏**（Indexer 从所有 c4 块里选 top-512/1024）；c128 层因为压缩比已经极高、块数很少，直接读全部 c128 块（不再二次选）。
- 索引张量最后一维必须对齐 64（[deepseek_v4_backend.py:1753-1759](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1753-L1759)，`swa_page_indices` 与 `extra_indices` 各断言一次），FlashMLA tile 要求。
- `attn_sink` 必须存在（[#L1749](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1749)，V4 用 attention sink）。
- 大 prefill 另有稀疏 kernel 分支（`extend_without_speculative` 且 token 数超 `_LARGE_INDEXER_QUERY_THRESHOLD`，或显式开 `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`；SM120 不支持，[#L1761-L1767](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1761-L1767)）——读取的 cache 来源不变，详见 [dsv4_pd_disaggregation_request_lifecycle.md](../04_pd_disaggregation/dsv4_pd_disaggregation_request_lifecycle.md)。
- 后端硬约束（构造时 [#L545-L556](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L545-L556)）：`head_dim==512`、`page_size==256`、`swa_page_size=128`（`SWA_WINDOW`，[#L100](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L100)）、`is_fp8_kvcache=True`。

---

## 9. HiSparse：c4 池的 GPU↔Host 分级卸载

> 本节聚焦 HiSparse 与 V4 cache 的耦合点。HiSparse 的完整设计（哈希查找、LRU、CUDA swap kernel）见 [developer_guide/hisparse_technical_architecture.md](../../developer_guide/hisparse_technical_architecture.md)。
>
> ⚠️ **文件已拆分**：旧的单文件 `mem_cache/hisparse_memory_pool.py` 现在只剩 `HiSparseDSATokenToKVPool`（122 行）。HiSparse 主体拆成三处：
> - **`mem_cache/allocator/hisparse.py`**（588 行）—— `HiSparseTokenToKVPoolAllocator`（[#L15](../../../python/sglang/srt/mem_cache/allocator/hisparse.py#L15)）+ `DeepSeekV4HiSparseTokenToKVPoolAllocator`（[#L276](../../../python/sglang/srt/mem_cache/allocator/hisparse.py#L276)）
> - **`mem_cache/pool_host/hisparse.py`**（71 行）—— `HiSparseHostPoolMixin`（host 侧大池）
> - **`managers/hisparse_coordinator.py`**（1050 行）—— 换入换出调度、LRU、`host_to_device_ratio: int = 2`（[#L123](../../../python/sglang/srt/managers/hisparse_coordinator.py#L123)）

### 9.1 动机

c4 层保留 1/4 的压缩历史，长上下文下仍可能占满显存。HiSparse 把 c4 压缩 KV 的**大部分放在 Host（CPU）内存**，GPU 只保留一个小的"设备缓冲区（device buffer）"，按 Indexer 的 top-512 选择结果**按需换入**。

### 9.2 关键耦合点

| 组件 | HiSparse 行为 |
|---|---|
| `c4_kv_pool` | 类型从 `DeepSeekV4SingleKVPool` 换成 `HiSparseC4DevicePool`（[deepseek_v4_memory_pool.py:177](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L177)，工厂 `_make_kv_pool` [#L830](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L830)）|
| Allocator | hybrid 分支的 `SWATokenToKVPoolAllocator` 外面再包 `DeepSeekV4HiSparseTokenToKVPoolAllocator`（[allocator/hisparse.py:276](../../../python/sglang/srt/mem_cache/allocator/hisparse.py#L276)，装配处 [kv_cache_configurator.py:1770](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1770)）|
| 容量 | c4 池缩小 `c4_shrink_factor = host_to_device_ratio`（默认 2），见 §5.4 的 `full//(4*shrink)` |
| Indexer | decode 时调 `coordinator.swap_in_selected_pages(...)` 换入选中页；非 decode 直接 `translate_loc_to_hisparse_device`（[indexer.py:847-866](../../../python/sglang/srt/layers/attention/dsv4/indexer.py#L847-L866)）|

**双 allocator 结构**是 HiSparse 的核心设计：一个"逻辑"分配器发放对外可见的槽号，一个"设备"分配器管真正的 GPU 缓冲区，两者用一张映射表关联。

- `HiSparseTokenToKVPoolAllocator`（DSA 用，[#L15-L69](../../../python/sglang/srt/mem_cache/allocator/hisparse.py#L15-L69)）：`logical_attn_allocator` 容量 `size * host_to_device_ratio`，`hisparse_attn_allocator` 容量 `size`；`available_size()` 取两者的 `min`；`size` 对外报的是 `_size_full`（逻辑容量）。
- `DeepSeekV4HiSparseTokenToKVPoolAllocator`（V4 用，[#L276-L321](../../../python/sglang/srt/mem_cache/allocator/hisparse.py#L276-L321)）：不自己建逻辑分配器，而是**接管传进来的那个**（`logical_attn_allocator`），只补一个按**压缩页大小**（`hisparse_page_size = c4_kv_pool.page_size`）工作的设备分配器。构造断言 `_kvcache` 是 `DeepSeekV4TokenToKVPool` 且 `c4_kv_pool` 是 `HiSparseC4DevicePool`。注意它保留两个不同的页大小：对外 `page_size` 是逻辑 full/SWA 页，c4 侧一切分配与换入用 `hisparse_page_size`。
- PD 分离的"直写 host"路径由 `alloc_logical_only`（[#L107-L128](../../../python/sglang/srt/mem_cache/allocator/hisparse.py#L107-L128)）支撑：只分配逻辑索引、不占设备页，因为 KV 由 prefill 节点 RDMA 直写 host、跳过 GPU staging。细节见 [dsv4_pd_p2d_transfer_contents_and_method.md](../04_pd_disaggregation/dsv4_pd_p2d_transfer_contents_and_method.md)。

### 9.3 `HiSparseC4DevicePool` 的地址翻译

c4 的逻辑压缩索引要再翻译到"设备缓冲区物理索引"（[deepseek_v4_memory_pool.py:217-233](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L217-L233)）：

```python
def translate_loc_from_full_to_compressed(full_indices):
    mask = (full_indices + 1) % 4 == 0      # 只保留每 4 个的最后一个 (压缩边界)
    return full_indices[mask] // 4

def translate_loc_to_hisparse_device(compressed_indices):
    return full_to_hisparse_device_index_mapping[compressed_indices]   # → device buffer 槽
```

写 c4 KV 时（`set_key_buffer` [#L235](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L235) / `set_key_buffer_fused` [#L244](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L244)）先把 loc 翻译到 device buffer 槽再写。另外 `HiSparseC4DevicePool.get_cpu_copy` 抛 `NotImplementedError`（[#L245 附近](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L245)）——c4 数据已经在 host 上，没有"再拷一份到 CPU"的语义。

### 9.4 地址空间扩展为四级

HiSparse 开启后，c4 的地址链路多了一级：

```
full loc ──//4 (gated)──► c4 compressed loc ──mapping──► hisparse device buffer loc
                                                          (host 大池 + GPU 小缓冲)
                                                                │ swap_in_selected_pages
                                                                ▼ (decode: 按 top-512 换入)
                                                          FlashMLA 读 device buffer
```

`full_to_hisparse_device_index_mapping` 大小按 `c4_logical_size + hisparse_page_size`，末尾追一个哨兵 `-1`（[allocator/hisparse.py:310-319](../../../python/sglang/srt/mem_cache/allocator/hisparse.py#L310-L319)）。这呼应了 §3.4 为何 indexer 池按 `c4_logical_size` 而非缩小后的 `c4_size` 分配。

### 9.5 硬约束

来自 [arg_groups/hisparse_hook.py](../../../python/sglang/srt/arg_groups/hisparse_hook.py)：

| 约束 | 代码 |
|---|---|
| 模型必须是 DSA 或 V4 | `assert is_deepseek_dsa(hf_config) or is_v4_hisparse`（[#L96](../../../python/sglang/srt/arg_groups/hisparse_hook.py#L96)）|
| `--disable-radix-cache` | `assert cfg.disable_radix_cache`（[#L101-L103](../../../python/sglang/srt/arg_groups/hisparse_hook.py#L101-L103)）|
| `kv_cache_dtype ∈ {bfloat16, fp8_e4m3}` | `HISPARSE_KV_CACHE_DTYPES`（[#L18](../../../python/sglang/srt/arg_groups/hisparse_hook.py#L18)），非 V4 才校验 |
| DSA 后端白名单（非 V4）| CUDA 按 dtype：`bfloat16 → {flashmla_sparse}`、`fp8_e4m3 → {flashmla_kv, flashinfer_sparse_mla}`；ROCm：`{tilelang, aiter}` |
| **与 unified_kv 互斥** | ROCm + V4 + `is_unified_kv_triton()` 直接 `raise ValueError`（[#L109-L122](../../../python/sglang/srt/arg_groups/hisparse_hook.py#L109-L122)）|
| V4 必须 hybrid SWA | `assert self.is_hybrid_swa`（[kv_cache_configurator.py:1771](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1771)）|

`--disable-radix-cache` 是因为 HiSparse 的 device buffer 按请求维护 LRU 换入换出，与 radix 前缀树的页共享语义冲突。unified_kv 互斥是因为该布局下 `c4_kv_pool is None`，装饰器的构造断言无法通过——所以在启动期就给一条清楚的报错，而不是等池初始化时抛一个费解的 `AssertionError`（注释里写明"Remove once unified-KV HiSparse lands"）。

---

## 10. MTP / NextN 草稿层的 Cache

### 10.1 草稿层强制 `compress_ratio = 0`

V4 的投机解码用 NextN（MTP）草稿头。草稿层的所有层都强制 `compress_ratio = 0`（[kv_cache_configurator.py:1084-1093](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1084-L1093)）：

```python
# models/deepseek_v4_nextn.py:50
COMPRESS_RATIO_NEXTN_LAYER = 0

# mem_cache/kv_cache_configurator.py:1084-1093（_build_dsv4_kv_pool 内）
if self.is_draft_worker:
    compression_ratios = [COMPRESS_RATIO_NEXTN_LAYER] * self.layer_info.num_effective_layers
else:
    compression_ratios = self.model_config.compress_ratios
```

含义：**草稿 worker 只用纯 SWA 池**，没有 c4/c128 压缩池、没有 indexer、没有压缩状态环。`_init_compressed_layer_mapping` 会把所有层标成 `compress_ratio=0`，`c4_layer_num = c128_layer_num = 0`，那些压缩子池层数为 0（空壳）。

> 草稿 worker 建的是一个**全新的 `DeepSeekV4TokenToKVPool` 实例**（`layer_num=1`），不是"主池加一层"；它与 target 共享 `req_to_token` 和槽号编号，两池各写各的 KV。详见 [dsv4_mtp_layer_forward_dataflow.md](../05_speculative_decoding/mtp_dsv4_layer_forward_dataflow.md)。

### 10.2 为什么草稿层不压缩

| 原因 | 说明 |
|---|---|
| 草稿序列短 | MTP 一次只推 N 个 draft token，近窗 SWA 足够覆盖 |
| 避免回滚复杂度 | 压缩态环在 verify 失败时需要回滚，草稿层不引入这套机制 |
| 在线 c128 与 MTP 有强耦合 | 见 §4.2：默认排斥，只有显式开 `SGLANG_EXPERIMENTAL_ONLINE_C128_MTP` + EAGLE topk=1 才允许，代价是状态池按 `1 + max_draft_tokens` 倍膨胀 |
| 容量预留已在 target 侧 | §5.3 已通过 `(T+D)/T` 在 target 池为草稿预留显存 |

### 10.3 NextN 的 cache 写入

`DeepseekV4ForCausalLMNextN` 复用 `DeepseekV4DecoderLayer`（`is_nextn=True`，`compress_ratio_override=COMPRESS_RATIO_NEXTN_LAYER`，[deepseek_v4_nextn.py:110](../../../python/sglang/srt/models/deepseek_v4_nextn.py#L110)）。因此草稿层走的就是 §7.1 的 SWA 写入路径（`set_swa_key_buffer_radix_fused_norm_rope`），只是不触发压缩/indexer 分支。

---

## 11. 跨实例导出：HiCache / PD 分离

六（七）类子池的 buffer 要暴露给传输层（PD 分离 RDMA、HiCache 分层存储），`DeepSeekV4TokenToKVPool` 提供四组 `*_buf_infos` 接口。**这里的顺序是 P 侧与 D 侧的隐式契约，两端装配顺序必须一致**。

| 接口 | 位置 | 导出内容 | item_len 语义 |
|---|---|---|---|
| `get_contiguous_buf_infos` | [#L675](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L675) | 主 KV 通道，扁平列表顺序恒为 **`[c4, c4_indexer, c128]`** | `buf[0].nbytes`，即**一行 = 一页**（不同于 MLA 的 `×page_size`）|
| `get_state_buf_infos` | [#L785](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L785) | `swa_kv_pool` 全层 + c4 两类压缩态 ring（显式 `continue` 跳过 `ratio==128`）| ring 用 SWA 页号当块号，`row * ring_size` 吸收模项 |
| `get_c128_state_buf_infos` | [#L814](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L814) | c128 压缩态，**请求级**寻址（`req_pool_idx`）| 按请求槽 |
| `get_unified_swa_ring_buf_infos` | [#L730](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L730) | 仅 unified_kv 布局（§3.6）的 SWA ring | 一行 |

配套的两个生命周期钩子：

- **`wait_layer_transfer`**（[#L976](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L976)）：HiCache 逐层传输时等某一层就位。
- **`clear_c128_req_state`**（[#L1007](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1007)）：因为 c128 状态是请求级的，D 侧复用一个 req 槽前必须显式清掉上一个请求的残留状态，否则会读到别人的压缩态。这是 c128 请求级寻址付出的代价（c4 是页级，槽随页释放自然失效，无需此操作）。

> c128 状态还有一个"可跳过传输"的性质：`seq_len % 128 == 0` 时在途状态为空，`get_dsv4_c128_state_indices` 返回空列表、该请求本次不传 c128 状态。传输层把 `SWA_RING` / `C128_STATE` 归入 `_requires_exact_state_index_match`——长度不匹配直接抛错而不是截断。完整的传输内容、地址公式与索引数学见 [dsv4_pd_p2d_transfer_contents_and_method.md](../04_pd_disaggregation/dsv4_pd_p2d_transfer_contents_and_method.md)。

---

## 12. 关键约束与易错点

### 12.1 启动期硬约束

V4 的默认值现在分两处，都在**配置解析（override / resolution）流水线**里，不再是一个函数里的硬编码赋值：

1. **`arg_groups/overrides.py:1359-1410` `_deepseek_v4_overrides`** —— 声明式默认：`attention_backend="dsv4"`、`page_size=256`（NPU 128，且显式指定 prefill/decode 后端）、`swa_full_tokens_ratio=0.1`、`moe_runner_backend` 自动选择。
2. **`arg_groups/overrides.py:2173-2194` `_deepseek_v4_kv_cache_dtype`** —— 由 `apply_deepseek_v4_defaults` 通过 `run_post_process_pass` 触发：`auto → fp8_e4m3`，NPU → `bfloat16`，最后断言 `kv_cache_dtype in ["fp8_e4m3", "bfloat16"]`。
3. **`arg_groups/deepseek_v4_hook.py:103` `apply_deepseek_v4_defaults`** —— 只剩少量命令式逻辑：ROCm 上 `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)`、`max_running_requests` 兜底 256、投机算法断言。

| 约束 | 值 | 原因 |
|---|---|---|
| `attention_backend` | `dsv4` | 强制专用后端 |
| `page_size` | `256`（NPU `128`）| c128 池 = `page_size//128`，必须 128 整除 |
| `kv_cache_dtype` | `fp8_e4m3`（默认）/ `bfloat16`（NPU）| SWA 池 584 B 布局基于 FP8 |
| `max_running_requests` | 默认 256 | 状态环/缓冲容量基线；也是 c128 请求级状态池的规模 |
| `swa_full_tokens_ratio` | 默认 `0.1` | SWA 池与状态环大小的缩放系数 |
| 投机算法 | `EAGLE`（须 `speculative_eagle_topk == 1`）或 `DSPARK` | MTP 与压缩态环兼容性 |

> ⚠️ 两处旧描述已失效：①`kv_cache_dtype` 不再"仅 fp8_e4m3"，NPU 走 bfloat16；②**没有 spec-v2 断言**，允许的算法集合是 `("EAGLE", "DSPARK")`。

另有 CP 场景的专门校验 `validate_deepseek_v4_cp`（[deepseek_v4_hook.py:160](../../../python/sglang/srt/arg_groups/deepseek_v4_hook.py#L160)）：强制 interleave-only CP（`enable_dsa_prefill_context_parallel=True`、`dsa_prefill_cp_mode="round-robin-split"`）、`enable_dp_attention=True`、`moe_dense_tp_size=1`、`attn_cp_size = tp // dp`，并断言 `dp_size == 1`、`tp_size <= 8`、`moe_a2a_backend in ("none","deepep","megamoe")`。

### 12.2 `swa_page_size` 命名陷阱（重要）

代码中有**四个相关但不同**的值，文档/调试时务必区分：

| 名字 | 值 | 来源 | 含义 |
|---|---|---|---|
| `page_size` | 256 | server_args（override 设定）| 顶层逻辑页大小 |
| pool 的 `swa_page_size` / `swa_window_size` | 256 | `get_schedule().page_size`（[kv_cache_configurator.py:1080](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1080)，非 NPU 断言 `== 256`）| SWA 池页大小 = 池内 `swa_window_size` |
| pool 的 `sliding_window` | 128 | `model_config.window_size`（[#L1120](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1120)）| 真实滑窗长度 |
| 后端 `DeepseekV4AttnBackend.swa_page_size` | 128 | 后端构造写死 = `SWA_WINDOW`（[deepseek_v4_backend.py:100](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L100)）| FlashMLA 滑窗 tile |

> ⚠️ **不要把 pool 的 `swa_page_size`(256) 和 configurator/后端用的 128 混为一谈**。非 NPU 下 `assert swa_page_size == 256`；而 `DSV4PoolConfigurator` 的 `self.swa_page_size = cfg.window_size`(128) 用于 `c4_state_ratio = ring/swa_page` 与 `c4_state_pool_size = swa_tokens // swa_page_size`。两者服务不同子系统。

### 12.3 压缩边界门控

写压缩池只在 `seq_len % ratio == 0`（凑满压缩块）时生效（§6.3）。未凑满时 `out_loc=0` 配合哨兵槽，避免污染。读 c4 时用 Indexer 选 top-512/1024，读 c128 时直接读全部块。

### 12.4 `state_dtype` 可选（不再恒为 FP32）

> ⚠️ 旧结论"`state_dtype` 恒为 `torch.float32`，是架构常量"**已失效**。

现在由环境变量 `SGLANG_DSV4_COMPRESS_STATE_DTYPE`（[environ.py:1314](../../../python/sglang/srt/environ.py#L1314)，默认 `"float32"`）决定，可切 `bfloat16`；解析函数 `_get_dsv4_compress_state_dtypes`（[kv_cache_configurator.py:108](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L108)）返回**两个独立 dtype**（`c4_state_dtype` / `c128_state_dtype`），分别传给 `_make_attn_state_pool` / `_make_indexer_state_pool`。容量侧对应 `_get_dsv4_compress_state_dtype_sizes()`（pool_configurator.py），所以 §5.2 的 `c*_state_bytes` 乘的是**查出来的 dtype size**，不是硬编码 4。

选 FP32 的理由仍成立（压缩状态环的 kv/score、max/sum 累积归约需要精度余量），但它现在是可调项：切 bf16 能把 c4/c128 状态开销砍半，代价是归约精度。

### 12.5 子池与 dtype 一览

| 子池 | dtype | 单元 |
|---|---|---|
| swa/c4/c128 KV | `uint8`（FP8 nope + BF16 rope 字节流）| 584 B/token |
| indexer index_k | `uint8` | 默认 132 B/token（128 + 4 B FP32 scale）；FP4 变体 68 B/token（`128//2 + 4`）|
| compress state 环 | `float32`（可切 `bfloat16`）| 离线 `2×head_dim`（c4 再 ×2 overlap）/ 在线 c128 `3×head_dim` |
| unified_kv（HIP）| `bfloat16` | 融合布局，见 §3.6 |

### 12.6 各子池的对齐要求

- SWA/压缩 KV 页字节对齐 576 倍数（FlashMLA，[deepseek_v4_memory_pool.py:119](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L119)）。
- 稀疏索引张量最后一维对齐 64（FlashMLA tile，[deepseek_v4_backend.py:1753-1759](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1753-L1759)）。
- `page_size % swa_page_size == 0`（pool 断言，[deepseek_v4_memory_pool.py:558](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L558)）。
- `page_size % 128 == 0`（configurator 断言，[pool_configurator.py:1006](../../../python/sglang/srt/model_executor/pool_configurator.py#L1006)；metadata kernel 也断言 `page_size >= 128 且 % 128 == 0`）。
- 离线压缩态池 `_size` 按 `lcm(ratio, state_cache_page_size)` 向上取整（§4.4）。

---

## 13. 端到端数据流总图

```
                            ┌────────────────────────────────────────────┐
                            │           DeepSeekV4TokenToKVPool          │
  ┌──────────┐  out_cache   │                                            │
  │  Req     │──loc(full)──►│  ① 写 SWA                                  │
  │ schedule │              │     backend.get_swa_out_cache_loc()        │
  └──────────┘              │       → swa_loc (int32, 三级取值)          │
       │ forward            │     set_swa_key_buffer_radix_fused_norm_   │
       ▼                    │       rope(swa_loc, ...) → swa_kv_pool     │
  ┌──────────┐   metadata   │     (unified_kv: set_unified_key_buffer_…) │
  │ MQALayer │──kernel────► │                                            │
  │_forward_ │  c4/c128_    │  ② 写压缩  forward_core_compressor →       │
  │ prepare  │  out_loc     │     CompressStatePool 累积在途态 →         │
  └──────────┘  (Triton)    │     set_extra_key_buffer_fused             │
       │                    │       → c4_kv_pool / c128_kv_pool          │
       │                    │                                            │
       │                    │  ③ 写索引  forward_indexer_compressor →    │
       │                    │     set_index_k_fp4 / set_index_k_fused    │
       │                    │       → c4_indexer_kv_pool (仅 c4 层)      │
       │ 读                 │                                            │
       ▼                    │  ④ C4Indexer.forward_c4_indexer:           │
  ┌──────────┐              │     打分核(6 选 1, 默认 DeepGEMM           │
  │ C4Indexer│              │       fp8_paged_mqa_logits)                │
  └──────────┘              │     + topk_transform(4 选 1)               │
       │                    │       → c4_sparse_page_indices (top-512)   │
       ▼                    │                                            │
  ┌──────────┐              │  ⑤ FlashMLA flash_mla_with_kvcache:        │
  │ FlashMLA │◄─────────────│     SWA 近窗(dense, window=128)            │
  │ attention│  swa_k_cache │     + extra(c4 稀疏页 / c128 全块)         │
  └──────────┘  +extra_k_   │     大 prefill 走 flash_mla_sparse_fwd     │
       │        cache       │                                            │
       │                    │  [HiSparse] c4 → host 大池 + GPU 小缓冲,   │
       ▼                    │     decode 按 top-512 swap_in_selected_    │
   output                   │     pages（双分配器：逻辑 + device）       │
                            └────────────────────────────────────────────┘
```

---

## 14. 术语与文件速查

### 14.1 术语表

| 术语 | 全称 | 含义 |
|---|---|---|
| SWA | Sliding Window Attention | ratio=0 全精度近窗池（584 B/token） |
| CSA / c4 | Compressed Sparse Attention | 4x 压缩 + Indexer top-512 稀疏 |
| HCA / c128 | High Compression Attention | 128x 压缩，读全块 |
| C4Indexer | — | c4 层稀疏块选择器（打分 + topk） |
| CompressStatePool | — | 在途压缩态 ring buffer（c4 页级 / c128 请求级） |
| unified_kv | — | 第 7 种子池布局，`DeepSeekV4UnifiedKVPool`，HIP-only ATOM 融合 |
| HiSparse | Hierarchical Sparse | c4 池 GPU↔Host 分级卸载（双分配器） |
| MTP / NextN | Multi-Token Prediction | EAGLE 草稿头，强制 ratio=0，**独立池实例** |
| `compress_ratios` | — | 逐层压缩比 ∈ {0,4,128}，PP 分段切片 |
| `swa_full_tokens_ratio` | — | SWA/状态环容量缩放，默认 0.1 |
| `swa_page_size` | — | ⚠️ 池内 = `get_schedule().page_size`（256）；后端 = 128。见 §12.2 |

### 14.2 文件速查

| 关注点 | 文件 | 关键符号 |
|---|---|---|
| 顶层池 + 子池 | [deepseek_v4_memory_pool.py](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py) | `DeepSeekV4TokenToKVPool`, `DeepSeekV4SingleKVPool`, `DeepSeekV4IndexerPool`, `DeepSeekV4UnifiedKVPool`, `HiSparseC4DevicePool` |
| 压缩状态环 | [deepseek_v4_compress_state.py](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py) | `CompressStatePool`, `KVAndScore`, `get_compress_state_ring_size` |
| 容量规划 | [pool_configurator.py](../../../python/sglang/srt/model_executor/pool_configurator.py) | `DSV4PoolConfigurator`, `_get_bytes_per_full_token`, `_get_c128_state_fixed_bytes`, `finalize_with_max_running_requests` |
| 池/分配器装配 | [kv_cache_configurator.py](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py) | DSV4 池实例化(#L1067-L1141)、allocator 选择链(#L1713-L1779)、`_get_dsv4_compress_state_dtypes`(#L108)、草稿池(#L868-L957) |
| 元数据核（Triton） | [kernels/ops/attention/dsv4/metadata_kernel.py](../../../python/sglang/kernels/ops/attention/dsv4/metadata_kernel.py) | `init_compression_metadata`(#L192), `_init_compressed_attn_metadata_kernel`(#L8) |
| 注意力读 | [deepseek_v4_backend.py](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py) | `DeepseekV4AttnBackend.forward`(#L1668), `get_swa_out_cache_loc`(#L1635), `store_cache`(#L1658), `DeepseekV4MultiStepBackend` |
| 压缩写 | [dsv4/compressor.py](../../../python/sglang/srt/layers/attention/dsv4/compressor.py) | `forward_compress`(#L68), `forward_core_compressor`(#L163), `forward_indexer_compressor`(#L192) |
| 稀疏选块 | [dsv4/indexer.py](../../../python/sglang/srt/layers/attention/dsv4/indexer.py) | `C4Indexer`, `forward_c4_indexer`(#L637), `_can_use_nonpaged_indexer`(#L479) |
| 模型写 SWA | [models/deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py) | `MQALayer._compute_kv_to_cache`(#L952), `_compute_kv_bf16`(#L991) |
| 草稿层 | [models/deepseek_v4_nextn.py](../../../python/sglang/srt/models/deepseek_v4_nextn.py) | `COMPRESS_RATIO_NEXTN_LAYER`(#L50) |
| HiSparse 分配器 | [mem_cache/allocator/hisparse.py](../../../python/sglang/srt/mem_cache/allocator/hisparse.py) | `HiSparseTokenToKVPoolAllocator`(#L15), `alloc_logical_only`(#L107), `DeepSeekV4HiSparseTokenToKVPoolAllocator`(#L276) |
| HiSparse host 池 | [mem_cache/pool_host/hisparse.py](../../../python/sglang/srt/mem_cache/pool_host/hisparse.py) | `HiSparseHostPoolMixin`, `DeepSeekV4PagedHostPool` |
| HiSparse 协调器 | [managers/hisparse_coordinator.py](../../../python/sglang/srt/managers/hisparse_coordinator.py) | `HiSparseCoordinator`, `host_to_device_ratio`(#L123, 默认 2) |
| 启动默认值/约束 | [arg_groups/overrides.py](../../../python/sglang/srt/arg_groups/overrides.py) | `_deepseek_v4_overrides`(#L1359), `_deepseek_v4_kv_cache_dtype`(#L2173) |
| V4 启动钩子 | [arg_groups/deepseek_v4_hook.py](../../../python/sglang/srt/arg_groups/deepseek_v4_hook.py) | `apply_deepseek_v4_defaults`(#L103), `validate_deepseek_v4_cp`(#L160) |
| HiSparse 启动钩子 | [arg_groups/hisparse_hook.py](../../../python/sglang/srt/arg_groups/hisparse_hook.py) | `HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE`(#L13), unified_kv 互斥(#L116) |

### 14.3 关联文档

- 部署/启动参数：[deepseek_v4_deployment_guide.md](deepseek_v4_deployment_guide.md)
- 模型结构与前向：[deepseek_v4_model_architecture.md](deepseek_v4_model_architecture.md)
- 显存空间构成：[deepseek_v4_flash_pool_sizing.md](deepseek_v4_flash_pool_sizing.md)
- PD 分离传输内容：[dsv4_pd_p2d_transfer_contents_and_method.md](../04_pd_disaggregation/dsv4_pd_p2d_transfer_contents_and_method.md)
- MTP 集中式执行：[dsv4_mtp_centralized_execution.md](../05_speculative_decoding/mtp_dsv4_centralized_execution.md)
- HiSparse 完整设计：[developer_guide/hisparse_technical_architecture.md](../../developer_guide/hisparse_technical_architecture.md)
- DSA（V3.2）对比：[developer_guide/dsa_technical_architecture.md](../../developer_guide/dsa_technical_architecture.md)









