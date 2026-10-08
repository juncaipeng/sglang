# DeepSeek V4 MTP 层前向计算与数据流详解

> 本文聚焦 **MTP（Multi-Token Prediction / NextN）draft 层本身的前向计算与数据流**：
> 它到底吃什么、算什么、KV 存在哪、prefill 和 decode 两个阶段的输入输出分别是什么样。
>
> 与本目录其他 MTP 文档的分工：
> - 本文 = **MTP 层数据流**（input/output、shape、融合、KV 池归属，逐阶段拆解）
> - `dsv4_mtp_centralized_execution.md` = **集中式调度执行编排**（draft→verify→draft_extend 在调度器里怎么串）
> - `mtp_speculative_decoding_architecture.md` = MTP 投机解码总体架构
> - `speculative_decoding_overview.md` = 投机解码全景（各算法/记账/CUDA graph）
> - `deepseek_v4_model_architecture.md` = 主模型 43 层结构（mHC / MQA / 三档压缩）
> - `deepseek_v4_cache_management.md` = 主模型六子池 KV 管理
>
> 核心代码：
> - `python/sglang/srt/models/deepseek_v4_nextn.py`（MTP 模型本体）
> - `python/sglang/srt/models/deepseek_v4.py`（复用的 `MQALayer` / `DeepseekV4DecoderLayer` / mHC）
> - `python/sglang/srt/speculative/eagle_worker_v2.py`（draft-extend / `draft_forward()` 驱动）
> - `python/sglang/srt/mem_cache/kv_cache_configurator.py`（draft KV 池构建）
> - `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`（`DeepSeekV4TokenToKVPool` 定义处）
> - `python/sglang/srt/model_executor/pool_configurator.py`（显存预留）
>
> **引用风格说明**：本文所有代码引用**以「文件 : 符号名」为主**（例如
> `kv_cache_configurator.py:_build_dsv4_kv_pool()`），因为符号名比行号稳定得多。
> 只有在**没有可用外层命名符号**时（模块级常量、或塞在超长 `__init__` 里的内联代码块）
> 才会残留行号，写成「快照 :NNN-MMM」的形式。所有残留行号基于
> `main` @ `33ed29a0ee` 的快照，可能随代码演进漂移，以实际文件为准。
> Markdown 链接只指向**文件**、不带 `#L` 锚点（锚点会漂移）。

---

## 目录

1. [一句话概览](#1-一句话概览)
2. [MTP 不使用主模型的 KV cache](#2-mtp-不使用主模型的-kv-cache)
3. [MTP 的 KV cache pool：独立的单层 SWA-only 池](#3-mtp-的-kv-cache-pool独立的单层-swa-only-池)
4. [MTP attention 具体怎么算（SWA MQA，compress_ratio=0）](#4-mtp-attention-具体怎么算swa-mqacompress_ratio0)
5. [融合步骤：e_proj / h_proj 与维度差异](#5-融合步骤e_proj--h_proj-与维度差异)
6. [prefill 阶段：输入与计算](#6-prefill-阶段输入与计算)
7. [decode 阶段：输入与计算](#7-decode-阶段输入与计算)
8. [prefill vs decode 全面对照](#8-prefill-vs-decode-全面对照)
9. [文件符号索引速查](#9-文件符号索引速查)

---

## 1. 一句话概览

DSV4 的 MTP 层是一个 **EAGLE 风格的单层 draft head**，用来做投机解码。它的定位可以用三句话锁死：

- **不读主模型 43 层的 KV**——它只消费主模型输出的 hidden（具体是 `pre_hc_head`，宽度 `4*H`），再配上"下一个 token 的词嵌入"。
- **自己有一层独立的 KV**——一个**单独的、只有 1 层、纯 SWA（滑窗）**的 KV 池，和主模型的六子池是两个对象。
- **和主模型共用 slot 编号 / `req_to_token_pool` 映射**——连 allocator 对象都是同一个（见 §3.3.1），省得再维护一套位置表，KV 内容却写在自己的池里。

它每次的产出是"预测接下来若干个 token"的草稿链，交给主模型一次并行 verify。

```
         ┌─────────────────────── 主模型（target, 43 层） ───────────────────────┐
 tokens ─┤  embed → 43×DecoderLayer(mHC/MQA/压缩) → norm → lm_head → logits       │
         └──────────────────────────────────┬──────────────────────────────────┘
                                             │ 输出 pre_hc_head [*, 4*H]
                                             │ （mHC 残差流 [*,4,H] 展平，4*H=16384）
                                             ▼
         ┌────────────────── MTP draft 层（NextN, 1 层） ─────────────────────┐
 下一tok─┤ embed → 融合(e_proj+h_proj) → 1×DecoderLayer(SWA MQA, ratio=0)      │
         │            → hc_post(4通道混合) → 存 pre_hc_head [*, 4*H]          │
         │            → hc_head(4→1) → norm → 自己的 lm_head                  │
         └───────────────────────────────────────────────────────────────────┘
                     ▲ 用主模型 hidden 当"上下文表征"，无需主模型 KV
```

关键常量：DSV4 MTP 的投机参数默认 `(num_steps=3, topk=1, num_draft_tokens=4)`，`topk==1` ⇒ **链式（chain）** 草稿，不是树。默认值由
`arg_groups/speculative_hook.py:_auto_choose_speculative_params()` 的兜底分支给出。

---

## 2. MTP 不使用主模型的 KV cache

这是最容易误解的一点。MTP 层前向**完全不碰**主模型那 43 层堆起来的 KV，它和主模型之间只有一条数据通道：`spec_info.hidden_states`（DSV4 上装的是 `pre_hc_head`，宽度 `4*H`）。

原因在 EAGLE 的核心思想：draft 层不重新理解整段上下文，而是**站在主模型已经算好的表征肩膀上**。主模型 forward 完，每个位置都产出了一个融合了全部上下文信息的 `hidden_states`；draft 层拿这个 hidden 当"浓缩的历史"，再叠上"要预测的下一个 token 的词嵌入"，就能低成本地往后猜几个 token。

| 维度 | 主模型（target） | MTP draft 层 |
|------|------------------|--------------|
| 层数 | 43 层 | 1 层 |
| KV 池 | 六子池（swa/c4/c128/c4_indexer/c4_state/c128_state） | 单个纯 SWA 池 |
| 压缩比 | 逐层 ∈ {0, 4, 128} | 恒为 0（`COMPRESS_RATIO_NEXTN_LAYER=0`） |
| 上下文来源 | 自己的 43 层 KV | 主模型的 `pre_hc_head`（宽度 `4*H`） |
| 是否读对方 KV | — | **否** |

代码上 draft 层的输入 hidden 来自 `forward_batch.spec_info.hidden_states`（[deepseek_v4_nextn.py:DeepseekV4ModelNextN.forward()](../../../python/sglang/srt/models/deepseek_v4_nextn.py)），而不是任何 `token_to_kv_pool.get_kv_buffer(...)` 主模型读取路径。整个 `deepseek_v4_nextn.py` 里没有一次 `get_kv_buffer` 调用。

注意这条通道传的**不是**主模型 norm 之后的 `[*, H]`，而是 mHC 残差流展平后的 `pre_hc_head`（`[*, 4*H]`）——详见 [§7.2](#72-draft_forward-的输入)。

---

## 3. MTP 的 KV cache pool：独立的单层 SWA-only 池

开启 MTP 后**确实新增一个 KV 池**，但它不是"主模型 SWA 池 + 加一层"，而是一个**全新的 DSV4 KV 池对象**（CUDA 上是 `DeepSeekV4TokenToKVPool`，NPU 上是 `DSV4NPUTokenToKVPool`，见 §3.1），参数被裁剪成极简：只有 1 层、只有 SWA 子池、c4/c128/state 全部为 0。

### 3.1 池是怎么建出来的

在 [kv_cache_configurator.py](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py) 的 `KVCacheConfigurator` 里：

- draft worker 复用 target 的 `memory_pool_config`（`KVCacheConfigurator.configure()` 开头，`assert self.memory_pool_config is not None, "Draft worker requires memory_pool_config"`），但 `KVCacheConfigurator._derive_pool_sizes()` 把 4 个压缩相关尺寸全部清零：`c4_max_total_num_tokens = 0`、`c128_max_total_num_tokens = 0`、`c4_state_pool_size = 0`、`c128_state_pool_size = 0`。代码里的注释写得很直白：「Draft worker reuses target's full/swa sizes but does NOT own c4/c128/state pools (those live on the target rank only); zero them out regardless of what config holds.」
- `KVCacheConfigurator._build_dsv4_kv_pool()` 用 `compression_ratios = [COMPRESS_RATIO_NEXTN_LAYER] * self.layer_info.num_effective_layers`（常量 `COMPRESS_RATIO_NEXTN_LAYER = 0` 在 `deepseek_v4_nextn.py` 模块级，快照 :50）。
- 构造一个 **新实例**：`layer_num=self.layer_info.num_effective_layers`（=1）、`c4_size=0`、`c128_size=0`、`c4_state_pool_size=0`、`c128_state_pool_size=0`、`swa_size=swa_max_total_num_tokens`。
- 池类**不恒是** `DeepSeekV4TokenToKVPool`：CUDA 上是它（定义在 [mem_cache/deepseek_v4_memory_pool.py](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py) 的 `DeepSeekV4TokenToKVPool`，**不在** `memory_pool.py`），NPU 上换成 `hardware_backend/npu/dsv4/dsv4_memory_pool.py:DSV4NPUTokenToKVPool`。
- indexer 也随之为 0：`DeepSeekV4TokenToKVPool.__init__` 里 `c4_logical_size = c128_size * 32`、`indexer_size = self.c4_logical_size`，`c128_size=0` ⇒ 两者都是 0，draft 没有 c4_indexer 子池。

所以它是 SWA-only、单层的独立对象——不是往主模型池里塞一层。

#### 3.1.1 `num_effective_layers == 1` 是哪来的

这个 1 不是硬编码的，而是 `model_runner_components/layer_setup.py:_compute_model_num_layers()` 在「draft worker + 启用 MTP」时取 `model_config.num_nextn_predict_layers` 得到的（结果封进
`layer_setup.py:ModelLayerInfo`，由 `layer_setup.py:resolve_layer_indices()` 解析）。DSV4 的 `num_nextn_predict_layers = 1`，所以 draft 池只建 1 层，`start_layer` / `end_layer` 也跟着落在 `[0, 1)`。

#### 3.1.2 SWA 环要为「在途 draft token」加宽

一个自然的疑问：滑窗只有 128，可一轮 decode 要往里塞 3~4 个 draft token，会不会把还要用的历史挤掉？答案是环的**物理**大小比窗口大：

```python
# deepseek_v4_memory_pool.py:DeepSeekV4TokenToKVPool.__init__ 内（快照 :586-601）
spec_extra = (
    (get_spec().speculative_num_draft_tokens - 1)
    if get_spec().speculative_algorithm is not None
    else 0
)
...
swa_ring_size=self.sliding_window + spec_extra,   # ← 128 + (num_draft_tokens-1)
```

即 `num_draft_tokens - 1` 个在途草稿槽是额外加出来的，不占窗口内的有效历史。

### 3.2 显存怎么预留的

不是给 draft 单独切一大块，而是在 target 的 per-token 字节数上**乘一个膨胀系数**。在 [pool_configurator.py](../../../python/sglang/srt/model_executor/pool_configurator.py) 的 `DSV4PoolConfigurator` 里（膨胀逻辑塞在 `DSV4PoolConfigurator.__init__` 内，快照 :825-832）：

```python
# pool_configurator.py:DSV4PoolConfigurator.__init__ 内（快照 :825-832）
self.bytes_per_full_token = self._get_bytes_per_full_token()
if self.is_speculative:
    draft_layers = 1
    target_layers = self.num_layers_total     # = len(self.compression_ratios)，单 PP 下 = 43
    self.bytes_per_full_token *= (target_layers + draft_layers) / target_layers
    #                             = 44/43 ≈ 1.023
```

即 draft 的那 1 层 KV 通过把每 token 字节数放大 ~2.3% 预留出来，和 target 共享同一块显存预算。

> 注：`num_layers_total = len(self.compression_ratios)`，而 `compression_ratios` 在多 PP stage 下会被切成 PP-local 片段，届时膨胀系数不再是 44/43。

### 3.3 请求级 slot 怎么分配

draft **不自己维护位置表**：它共享 target 的 `req_to_token_pool` 和 slot 编号（`out_cache_loc`）。

- 调度器把 target 的 `pool, allocator = self.tp_worker.get_memory_pool()` 传给 draft（[scheduler.py:Scheduler.init_memory_pools()](../../../python/sglang/srt/managers/scheduler.py)）。
- draft worker 的 [eagle_worker_v2.py:EagleDraftWorker.alloc_memory_pool()](../../../python/sglang/srt/speculative/eagle_worker_v2.py) 接收 target 的 `req_to_token_pool` + `token_to_kv_pool_allocator`。
- prefill 时 target 已经给每个 token 分好 slot；MTP draft-extend **复用同一批 `out_cache_loc`**，只是把 KV 内容写进自己的池。

#### 3.3.1 「同一个 slot 编号」是怎么实现的：allocator 对象被原样复用

关键机制在 `kv_cache_configurator.py:KVCacheConfigurator._build_token_to_kv_pool_allocator()` 的 draft 分支——它**根本不新建 allocator**，直接把传进来的 target allocator 原样 `return`，只把 target 的 full→swa 映射表注册到 draft 的新池上：

```python
# kv_cache_configurator.py:KVCacheConfigurator._build_token_to_kv_pool_allocator() draft 分支
else:
    assert self.is_draft_worker
    if self.is_hybrid_swa:
        if self.draft_swa_full_capacity:
            ...                                   # identity 映射，见 §4.4
        else:
            swa_allocator = ...                   # target 的 SWATokenToKVPoolAllocator
            token_to_kv_pool.register_mapping(
                swa_allocator.full_to_swa_index_mapping   # ← 共用同一张地址表
            )
return token_to_kv_pool_allocator                 # ← 原样返回 target 的 allocator
```

所以「共用 slot 编号」不是两边各算一遍碰巧一致，而是**同一个 allocator 对象 + 同一张 full→swa 映射表**。这也直接解释了 [§4.1](#41-attention-的输入--输出--是否存-kv) 里 draft 池的地址翻译为什么和 target 一致。

#### 3.3.2 DCP 下 loc 空间会被放大

一个脚注级细节：开启 DCP（decode context parallel）且 draft 是 replicated 时，draft 的 loc 空间要按 `dcp_size` 放大，否则不同 DCP rank 的 draft 会撞地址：

```python
# kv_cache_configurator.py:KVCacheConfigurator.loc_space_scale
return dcp_size if (self.is_draft_worker and dcp_size > 1) else 1
```

它在 `KVCacheConfigurator._derive_pool_sizes()` 里乘到 `full_max_total_num_tokens` / `swa_max_total_num_tokens` 上，配套的 `KVCacheConfigurator.pool_page_size` 再作为 `page_size=` 传进 `_build_dsv4_kv_pool()`。非 DCP 场景该系数恒为 1，可以忽略。

一句话：**同一个 slot 编号，两个池各写各的 KV**（target 写六子池，draft 写它的单层 SWA 池）。

---

## 4. MTP attention 具体怎么算（SWA MQA，compress_ratio=0）

MTP 层复用主模型的两个类——注意这是**两个不同的类**，别混成一个：

- [deepseek_v4.py:MQALayer](../../../python/sglang/srt/models/deepseek_v4.py)（继承 `MqaAttentionBase`）——attention 本体
- [deepseek_v4.py:DeepseekV4DecoderLayer](../../../python/sglang/srt/models/deepseek_v4.py)——把 MQA + mHC(`hc_pre`/`hc_post`) + MoE 包成一层

构造时把 `compress_ratio_override = COMPRESS_RATIO_NEXTN_LAYER = 0` 一路传下去（`deepseek_v4_nextn.py:DeepseekV4ModelNextN.__init__` → `DeepseekV4DecoderLayer.__init__` → `MQALayer.__init__` 里赋给 `compress_ratio=`）。`ratio=0` 带来两个后果：

1. **不建 compressor / indexer**——`MQALayer.__init__` 里先 `self.compressor = None; self.indexer = None`，再 `if self.compress_ratio in (4, 128):` 才建，所以 MTP 没有 c4/c128/indexer 路径。
2. **纯 SWA（滑动窗口）近窗注意力**——只在最近 `window_size`（=128）个 token 上算（但见 [§4.4](#44-注意窗口是谁强制的draft_swa_full_capacity-分支)）。

### 4.1 attention 的输入 / 输出 / 是否存 KV

| 项 | 内容 |
|----|------|
| 输入 | 融合后的 `hidden_states`（见 §5），经 Q/KV 投影 |
| KV heads | **1 个**（MQA，`MqaAttentionBase.__init__` 里 `assert config.num_key_value_heads == 1`，`MQALayer.__init__` 建 `RadixAttention(..., num_kv_heads=1, ...)`），单个 512 维 KV 头被所有 Q 头共享 |
| 是否存 KV | **存**——但由融合 kernel 在 K 投影里写完，所以 attention backend 那边不重复写（`save_kv_cache` 的两条分支见 §4.2） |
| 存到哪 | draft 自己的单层 SWA 池；写入下标是 `out_cache_loc` 经 **full→swa 地址翻译**后的环内下标，不是原始 `out_cache_loc`（见 §4.2 末） |
| 只算 SWA 窗口 | **是**，`ratio=0` 只有近窗 dense，无远端压缩块 |
| 输出 | attention 结果经分组输出投影 → `[T, H]` |

### 4.2 QKV 形状与 KV 写入路径

- **Q 路径**：`wq_a`（H → `q_lora_rank=1024`）→ `q_norm` → `wq_b`（→ n_heads × head_dim），head_dim=512（`qk_nope_head_dim=448` + `qk_rope_head_dim=64`）。
- **KV 路径**：`wkv`（H → 512，**单头**）→ RMSNorm + RoPE + 写 SWA 缓存**三件事融合在一个 kernel**。调用点在 `deepseek_v4.py:MQALayer._compute_kv_to_cache()`（`token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(...)`，方法本身定义在 `deepseek_v4_memory_pool.py:DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused_norm_rope()`）；另一条等价路径是 `deepseek_v4.py:MQALayer._forward_prepare()` 里的 `fused_qk_norm_rope_swa_store(...)`。
- **`save_kv_cache` 有两条分支**（不是无条件 `False`）：

  ```python
  # deepseek_v4.py:MQALayer.forward()
  attn_k = kv if kv is not None else q
  ...
  if is_unified_kv_triton():
      o = attn_backend.forward(..., save_kv_cache=kv is not None)   # ← 由 kv 是否为 None 决定
  else:
      attn_q = q_padded if q_padded is not None else q
      save_kv_cache = False                                        # ← 非 unified 路径恒 False
      ...
      o = attn_backend.forward(..., save_kv_cache=save_kv_cache)
  ```

  而 `kv` 是否为 `None` 由 `MQALayer._forward_prepare()` 末尾决定：

  ```python
  # deepseek_v4.py:MQALayer._forward_prepare()
  if not (unified and fuse_verify):
      kv = None      # ← 只有 unified + verify 融合路径才把 kv 交回给 backend 去存
  ```

  对 draft 的 draft/decode 步，`kv` 被置 `None` ⇒ 两条分支最终都**不会重复写**（KV 已在 fused kernel 里落盘）。只有 target 的 verify 路径在 unified-kv-triton 下会走 `save_kv_cache=True`，让 backend 用它自己的因果下标去存环。
- **SWA 写入下标要翻译**：`deepseek_v4_backend.py:DeepseekV4AttnBackend.get_swa_out_cache_loc()` 会把 `forward_batch.out_cache_loc` 交给 `token_to_kv_pool.translate_loc_from_full_to_swa(...)` 再 `.to(torch.int32)`（decode/verify 时优先用 metadata 里已缓存的结果）。所以 §3.3 的「共用 slot 编号」成立，但真正落到 SWA 环里的是翻译后的下标。
- **输出投影**（分组）：attn 输出 reshape 成 `[T, 8组, *]` → `wo_a` einsum `"tgd,grd->tgr"`（bf16 路径在 `deepseek_v4.py:_apply_wo_a_bf16_matmul()`，fp8 路径走 `deep_gemm.fp8_einsum("bhr,hdr->bhd", ...)`）→ `[T, 8, o_lora_rank=1024]` → `wo_b(o.flatten(1))` → `[T, H]`。

### 4.3 一次 attention 的骨架（[deepseek_v4.py:MQALayer.forward()](../../../python/sglang/srt/models/deepseek_v4.py)）

```
_forward_prepare(融合 QKV + fused KV store，并按需把 kv 置 None)
   → attn_backend.forward(save_kv_cache = kv is not None / 恒 False)
   → 逆 RoPE(fused_rope_inplace(..., inverse=True))
   → 分组输出投影(view 成 [T,8,*] → wo_a einsum → wo_b)  →  [T, H]
```

### 4.4 注意：窗口是谁强制的（`draft_swa_full_capacity` 分支）

上面说的「SWA 环形淘汰保证只看最近 128 个 token」有一个例外分支。`kv_cache_configurator.py:KVCacheConfigurator._build_token_to_kv_pool_allocator()` 在 `self.draft_swa_full_capacity` 为真时，给 draft 池注册的是**identity 映射**：

```python
# kv_cache_configurator.py:KVCacheConfigurator._build_token_to_kv_pool_allocator() draft 分支
if self.draft_swa_full_capacity:
    # Banded depth: the SWA ring is full draft capacity, so use
    # an IDENTITY full->swa mapping — store and read locs both
    # equal out_cache_loc, and a slot is never evicted before
    # the request frees it. The window itself is enforced by the
    # FA sliding-window kernel, not by the ring.
    n = sizes.full_max_total_num_tokens + self.page_size
    identity_mapping = torch.arange(n + 1, dtype=torch.int64, device=self.device)
    identity_mapping[-1] = -1                    # ← -1 last_loc 仍映射到 -1
    token_to_kv_pool.register_mapping(identity_mapping)
```

这条分支下：写下标 == 读下标 == `out_cache_loc`（翻译退化成恒等），槽位在请求释放前**永不被淘汰**，滑窗语义改由 **FlashAttention 的 sliding-window kernel** 强制。功能结果一样（只看最近 128 个），但机制完全不同——排查「为什么 draft 池比预期大」时要认得这条分支。

---

## 5. 融合步骤：e_proj / h_proj 与维度差异

融合是 MTP 前向最独特的一步——把"主模型给的上下文表征"和"要预测的下一个 token 的词嵌入"揉到一起。代码在 [deepseek_v4_nextn.py:DeepseekV4ModelNextN.forward()](../../../python/sglang/srt/models/deepseek_v4_nextn.py)：

```python
if hidden_states.shape[0] > 0:
    n_tokens = hidden_states.shape[0]          # prefill=T_total；decode=bs
    d = self.config.hidden_size
    hc_flat = forward_batch.spec_info.hidden_states.view(n_tokens * self.hc_mult, d)  # [bs*4, H] ← 入参是 [bs, 4*H]
    h_proj_out, _ = self.h_proj(self.hnorm(hc_flat))
    h_proj_hidden_states = h_proj_out.view(n_tokens, self.hc_mult, d)   # [T, 4, H]  ← 4 通道
    e_proj_hidden_states, _ = self.e_proj(self.enorm(hidden_states))    # [T, H]     ← 单通道
    hidden_states = e_proj_hidden_states[:, None, :] + h_proj_hidden_states  # 广播到 [T, 4, H]
```

> 那句 `.view(n_tokens * self.hc_mult, d)` 就是 [§7.2](#72-draft_forward-的输入) 里「步间 hidden 宽度是 `4*H`」的直接证据：
> 若入参真是 `[bs, H]`，这个 view 根本凑不出 `bs*4` 行。

### 5.1 两条路的语义

| 路 | 输入 | 输出 shape | 携带什么 |
|----|------|-----------|----------|
| `e_proj` | draft 自己 `embed_tokens(input_ids)` 的词嵌入 | `[T, H]`（单通道） | **要预测的下一个 token 的嵌入** |
| `h_proj` | 主模型 / 上一步 draft 的 `hidden_states`（`[T, 4*H]`，view 成 `[T*4, H]`） | `[T, 4, H]`（4 通道） | **上下文表征** |

### 5.2 维度差异：核心是 mHC 的 `hc_mult=4` 通道

- `h_proj` 的输入是 `[T, 4*H]`（DSV4 的 mHC 多隐藏通道残差流展平后的结果），先 `.view(T*4, d)` 还原成每通道一行、对每个通道独立投影，再 reshape 回 `[T, 4, H]`，**保留 4 通道**。
- `e_proj` 的输入是普通词嵌入 `[T, H]`，**单通道**、没有 mHC 结构。
- 相加时用 `e_proj_hidden_states[:, None, :]` 插一个 size-1 维 → `[T, 1, H]`，靠 broadcast **把同一份"下一个 token 嵌入"复制叠到 4 个 mHC 通道的每一个上**，得到 `[T, 4, H]`。

一句话：`e_proj` 单通道 `[T,H]`、`h_proj` 四通道 `[T,4,H]`（入参 `[T,4*H]`），差一个 `hc_mult=4`；融合时前者广播到后者的 4 个通道。

---

## 6. prefill 阶段：输入与计算

prefill 阶段 MTP 走的是 **draft-extend-for-prefill**（extend/prefill 模式）：目的是**把 draft 那 1 层的 SWA KV 在整段 prompt 上一次填满**，好让 decode 第一步能基于完整历史去预测。驱动函数 [eagle_worker_v2.py:EagleDraftWorker._draft_extend_for_prefill()](../../../python/sglang/srt/speculative/eagle_worker_v2.py)。

### 6.1 输入：整段 prompt 的**全部** token（左移一位）

这里最容易误解——prefill 每个请求只**生成** 1 个 token（`next_token_ids`），但 MTP 的 **input_ids 是整段 prompt 的全部 token**，那个生成的 token 只是被拼在末尾。构造过程在 `EagleDraftWorker._draft_extend_for_prefill()` 里：

```python
if not batch.forward_mode.is_idle():                   # ← idle stub 不做左移
    # chunked-prefill 下每个请求的"尾 token"不一定就是 next_token_ids[i]
    tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)
    new_input_ids = torch.empty_like(batch.input_ids)   # 大小 = T_total，覆盖全部 prefill token
    pt = 0
    for i, extend_len in enumerate(batch.extend_lens):    # 逐请求
        input_ids = batch.input_ids[pt : pt + extend_len]
        new_input_ids[pt : pt + extend_len].copy_(
            torch.cat((input_ids[1:], tail_tokens[i].reshape(1)))  # 去头 + 补尾(生成的 token)
        )
        pt += extend_len
    assert pt == batch.input_ids.numel()   # 断言覆盖了所有 token
    batch.input_ids = new_input_ids
```

举例，某请求 prompt=`[t0,t1,t2,t3]`（extend_len=4），target 生成 `t4`：

```
原始 input_ids:      [t0, t1, t2, t3]
左移+补尾后 MTP 输入:  [t1, t2, t3, t4]   ← 长度仍是 4（不是 1）
                                    └ 这个 t4 才是"prefill 生成的那 1 个 token"
```

- 位置 0：input=`t1`，h_proj=target 在 `t0` 的 hidden → 预测 `t2`
- 位置 3：input=`t4`（生成的 token），h_proj=target 在 `t3` 的 hidden → 预测 `t5`（draft 的第一个真正预测）

这就是 EAGLE 的**错位对齐**：位置 p 上 `input_ids` 存原始序列 p+1 的 token，`hidden_states` 存主模型在 p 的表征。

### 6.2 计算：一次 forward 跑过所有 token

同在 `EagleDraftWorker._draft_extend_for_prefill()` 里：

- `batch.spec_info = EagleDraftExtendInput(hidden_states=target_hidden_states, num_tokens_per_req=1, num_tokens_for_logprob_per_req=1)`。
- `capture_hidden_mode` **不是硬编码的 `LAST`**，而是按算法挑：

  ```python
  # eagle_worker_v2.py:EagleDraftWorker._draft_extend_for_prefill()
  capture_hidden_mode = (
      CaptureHiddenMode.NULL
      if self.speculative_algorithm.is_standalone()
      else CaptureHiddenMode.LAST        # ← DSV4 MTP 走这条
  )
  forward_batch = ForwardBatch.init_new(
      batch,
      self.draft_runner,
      capture_hidden_mode=capture_hidden_mode,
      return_hidden_states_before_norm=False,   # ← 但对 DSV4 nextn 无效，见 §6.3
  )
  ```

  `ForwardBatch.init_new(...)` 没有覆写 `out_cache_loc`，所以复用的就是 target 分好的那一批。
- `n_tokens = hidden_states.shape[0] = T_total`（整个 ragged batch 展平），所以 §5 的融合、单层 decoder、SWA attention 都在**全部 T_total 个 token** 上跑，一次性填满 draft SWA KV。
- 输出 `CaptureHiddenMode.LAST` **只取每个请求最后一个位置**的 hidden，作为 decode 第一步 draft 的种子（这是"只关心一个"的地方，但那是**输出**，不是输入）。

### 6.3 一个反直觉细节：`return_hidden_states_before_norm=False` 对 DSV4 nextn 不起作用

draft-extend 显式传了 `return_hidden_states_before_norm=False`，看起来是要「只回 norm 之后的 hidden」。但 DSV4 nextn 走的是另一条更高优先级的通道：

1. `deepseek_v4_nextn.py:DeepseekV4ForCausalLMNextN.forward()` **无条件**把 `pre_hc_head` 作为 `hidden_states_before_norm=` 传给 logits processor。
2. `logits_processor.py:LogitsProcessor._get_hidden_states_to_store()` 里明确写着：

   ```python
   if hidden_states_to_store_before_norm is not None:
       # NOTE: when hidden_states_before_norm is provided, we always
       # prefer to return it.
       hidden_states_to_store = hidden_states_to_store_before_norm
   ```

所以那个 `False` 被「provided 就一定用它」的规则压过去了，种子 hidden 仍然是 `pre_hc_head`（宽度 `4*H`）。这正是 [§7.2](#72-draft_forward-的输入) 里步间 hidden 宽度结论的关键一环——不理解这一条，很容易误以为种子是 `[bs, H]`。

---

## 7. decode 阶段：输入与计算

decode 阶段 MTP 走 [eagle_worker_v2.py:EagleDraftWorker.draft_forward()](../../../python/sglang/srt/speculative/eagle_worker_v2.py)，是一个**多步循环**，每步只处理"每请求 1 个 token"，把结果串成一条 draft 链。

### 7.1 一整轮 decode 的三阶段

`eagle_worker_v2.py:EagleDraftWorker.forward_batch_generation()` 的 DECODE 分支依次跑：

```
① draft_forward          →  ② verify              →  ③ _draft_extend_for_decode
   draft 层多步循环            target 并行验证 4 候选     draft 层为下一轮热身
   产出 draft 链              拒绝/贪心采 accept          产出下一轮种子 hidden
```

用户关心的"MTP 输入/计算"主要在 ① 和 ③（都是 draft 那 1 层在跑）。

### 7.2 draft_forward 的输入

进循环前（`EagleDraftWorker.draft_forward()` 开头的 spec_info 解包）`spec_info: EagleDraftInput` 携带三样：

| 输入 | 含义 | 来源 |
|------|------|------|
| `topk_index` | 上一步/种子提出的 token（decode 每请求 1 个） | draft-extend 种子 或 上一步 draft 输出 |
| `hidden_states` | 该 token 对应的 hidden **`[bs, 4*H]`**（见下） | draft-extend LAST 输出 / 上一步 draft 的 `pre_hc_head` |
| `out_cache_loc` | 本轮预留 KV slot，按步切成 `out_cache_loc[i]` | 调度器 alloc（切分见 `eagle_utils.py:per_step_draft_out_cache_loc()`） |

#### 步间 hidden 的宽度是 `4*H`，不是 `H`

这是本文最容易记错的一点。DSV4 走 mHC，draft 步之间传的是**展平的 4 通道残差流**，宽度 `hidden_size × hc_mult = 4096 × 4 = 16384`：

```python
# configs/model_config.py:ModelConfig._derive_model_shapes()
hc_mult = getattr(self.hf_text_config, "hc_mult", 1)
self.spec_hidden_size = (
    self.hidden_size * hc_mult if hc_mult > 1 else self.hidden_size
)
```

`speculative/eagle_utils.py:get_draft_recurrent_hidden_state_spec_from_config()` 正是返回这个 `model_config.spec_hidden_size`，用来给步间 hidden buffer 定尺寸。数据侧的来源是：

```python
# deepseek_v4_nextn.py:DeepseekV4ModelNextN.forward()
pre_hc_head = hidden_states.flatten(1)     # [T, 4, H] → [T, 4*H]
```

再由 `DeepseekV4ForCausalLMNextN.forward()` 以 `hidden_states_before_norm=pre_hc_head` 交给 logits processor，而后者「provided 就一定用它」（见 [§6.3](#63-一个反直觉细节return_hidden_states_before_normfalse-对-dsv4-nextn-不起作用)）。target 侧同理——`deepseek_v4.py:DeepseekV4ForCausalLM.forward()` 也是把自己的 `pre_hc_head` 递出去。

> `EagleDraftInput.hidden_states` 的字段注释写的是 `(b, hidden_size)`，那是跨算法的通用描述；
> DSV4 的实际宽度由 `spec_hidden_size` 决定。

注意 batch 是 running batch，**每请求只有 1 个 token 位置**（`num_tokens_per_req=1`），进 MTP 模型时 `n_tokens = bs`（请求数），不再是 prefill 的 `T_total`。

### 7.3 循环内每步计算（`EagleDraftWorker.draft_forward()` 主循环）

DSV4 `topk==1` 链式：

```python
for i in range(self.speculative_num_steps):   # num_steps=3
    input_ids = topk_index.flatten()          # 上一步提出的那个 token（每请求 1 个）
    if i == self.speculative_num_steps - 1:   # skip-last：种子已给 1 个，这里只需 steps-1 个
        break
    forward_batch.input_ids     = input_ids
    forward_batch.out_cache_loc = out_cache_loc[i]   # 本步写哪块 KV slot（还要经 full→swa 翻译）
    spec_info.hidden_states     = hidden_states      # 上一步 hidden，宽度 4*H
    with forward_context(attn_backend=self.draft_attn_backend.attn_backends[i]):
        logits_output = self.draft_runner.forward(forward_batch).logits_output
    topk_index    = argmax(logits)    # topk==1 贪心（CUDA 上融进 draft_topk1_postprocess）
    positions.add_(1)                 # 位置 +1
    hidden_states = logits_output.hidden_states      # 输出 hidden 喂给下一步
```

**每次 `draft_runner.forward` 内部**（同一段 `deepseek_v4_nextn.py:DeepseekV4ModelNextN.forward()`，只是 `n_tokens=bs`）：

1. `embed_tokens(input_ids)` → 当前 token 嵌入 `[bs, H]`
2. **融合**（§5）：`e_proj(嵌入)[:,None,:] + h_proj(上一步 hidden [bs,4*H] → view [bs*4,H])` → `[bs, 4, H]`
3. **单层 SWA MQA decoder**：对这 1 个 token 从 SWA 窗口历史 KV（prompt 期填的 + 前几步 draft 写的）做 MQA，同时把新 token KV 写到 `out_cache_loc[i]` 翻译后的 SWA 环下标
4. `hc_post` 做**4 通道残差混合**（`[bs,4,H] → [bs,4,H]`，**不收拢**）→ `pre_hc_head = flatten(1)` 存下来当下一步的输入 → `hc_head` 才做 **4→1**（`[bs,4,H] → [bs,H]`）→ `shared_head.norm` → `lm_head` → logits

> §7.3 第 4 步的分工要记牢：**`hc_post` 保形，`hc_head` 才收拢**。
> `deepseek_v4.py:DeepseekV4DecoderLayer.hc_post()` 的 torch 参考实现里有
> `assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])`，空 batch 早退也返回
> `(0, self.hc_mult, x.shape[-1])`——输出始终是 4 通道。
> 而 4→1 发生在 `deepseek_v4_nextn.py:DeepseekV4ModelNextN.hc_head()`，且**在它之前**已经把
> `[bs,4,H]` 展平存成 `pre_hc_head` 了，这就是步间 hidden 为什么是 `4*H`。

### 7.4 链是怎么串起来的

第 i 步的输出 token/hidden 直接当第 i+1 步的输入：

```
种子(draft-extend 给的 1 token + hidden [bs,4*H])
        │
   step0 forward → token1, hidden1   写 KV slot[0]
        │
   step1 forward → token2, hidden2   写 KV slot[1]
        │
   step2: i==num_steps-1 → break（不 forward）
```

- 种子 1 个 + step0/step1 各 1 个 = **3 个 draft token**（=num_steps）
- verify 再加 target 的 1 个 bonus token → **4 = num_draft_tokens** ✓

这个记账正是 `EagleDraftWorker.draft_forward()` 里 `if i == self.speculative_num_steps - 1: break` 实现的效果（配套的还有 `EagleDraftWorker.draft()` 里的 `n_inner = self.speculative_num_steps - 1`）。注意代码里这两处**都没有注释**，别去源码找说明文字。

---

## 8. prefill vs decode 全面对照

| 维度 | prefill（draft-extend-for-prefill） | decode（draft_forward） |
|------|-------------------------------------|--------------------------|
| 目的 | 一次填满整段 SWA KV | 滚动产出 draft 链 |
| forward 次数 | 1 次 | num_steps-1 次（循环） |
| 每次 token 数 | T_total（全部 prompt token） | bs（每请求 1 个） |
| `n_tokens`（融合处） | T_total | bs |
| input_ids | 整段 prompt **左移一位** | 上一步提出的**单个** token |
| hidden 来源 | target 主模型的 `pre_hc_head` | 上一步 draft 的 `pre_hc_head` / 种子 |
| hidden 宽度 | `[T_total, 4*H]` | `[bs, 4*H]`（**不是 `H`**，见 §7.2） |
| KV 写入 | 一次写满整段 SWA slot | 每步写 `out_cache_loc[i]` 一个 slot |
| KV 写入下标 | `out_cache_loc` 经 full→swa 翻译 | 同左，逐步翻译 |
| capture 模式 | 非 STANDALONE ⇒ `LAST`（取每请求末位 hidden 当种子） | 每步输出 hidden 直接喂下一步 |
| 输出用途 | 种子 → decode 第一步 draft | 串成链 → verify |
| attention backend | 普通 `DeepseekV4AttnBackend` | **单个** `DeepseekV4MultiStepBackend`，内部持 num_steps 个 `DeepseekV4AttnBackend`，每步取 `attn_backends[i]` |

关于最后一行——`deepseek_v4_backend.py:DeepseekV4MultiStepBackend.__init__` 里的构造是：

```python
self.attn_backends: List[DeepseekV4AttnBackend] = []
for i in range(self.speculative_num_steps):
    self.attn_backends.append(
        DeepseekV4AttnBackend(
            model_runner, speculative_step_id=i, topk=self.topk,
            speculative_num_steps=self.speculative_num_steps,
        )
    )
```

所以**不是**「每步一个 `DeepseekV4MultiStepBackend` 实例」，而是一个 MultiStep 外壳装 num_steps 个单步 backend；§7.3 循环里取的 `self.draft_attn_backend.attn_backends[i]` 正是其中第 i 个。

**一句话总结**：
- **prefill**：input 是整段 prompt（左移一位），一次 forward 跑过全部 T_total token 填满 KV，输出只留每请求最后一个 hidden（`pre_hc_head`，宽度 `4*H`）当种子。
- **decode**：input 是"上一步提出的 1 个 token + 它的 hidden `[bs, 4*H]`"，每步 forward 只算 1 个 token 的 SWA attention 并写 1 个 slot，靠"输出喂回输入"滚成长度 num_steps 的 draft 链。

---

## 9. 文件符号索引速查

> 本节以**符号名**为索引键（符号名比行号稳定）。仅在没有外层命名符号时保留行号，
> 写成「快照 :NNN-MMM」，基于 `main` @ `33ed29a0ee`。

### MTP 模型本体 `models/deepseek_v4_nextn.py`

| 符号 | 内容 |
|------|------|
| `COMPRESS_RATIO_NEXTN_LAYER`（模块级常量，快照 :50） | `= 0`，draft 层纯 SWA |
| `DeepseekV4ModelNextN.__init__` | 构造单层 decoder（`is_nextn=True`, `compress_ratio_override=COMPRESS_RATIO_NEXTN_LAYER`, `layer_id=0`） |
| `DeepseekV4ModelNextN.forward()` | MTP 前向主体：embed → 融合 → decoder → `hc_post` → `pre_hc_head` → `hc_head` → norm |
| 同上，`self.embed_tokens(input_ids)` | 词嵌入 `[T, H]` |
| 同上，融合块 | `e_proj` + `h_proj`，广播到 4 通道 |
| 同上，`forward_batch.spec_info.hidden_states.view(...)` | 消费上游 hidden（`[T, 4*H]` → `[T*4, H]`） |
| 同上，`self.decoder(...)` + `self.decoder.hc_post(...)` | 单层 decoder 调用 + **4 通道残差混合（保形）** |
| 同上，`pre_hc_head = hidden_states.flatten(1)` | `[T,4,H]` → `[T,4*H]`，喂给下一步 draft |
| `DeepseekV4ModelNextN.hc_head()` | **4→1** 收拢，之后接 `shared_head.norm` |
| `DeepseekV4ForCausalLMNextN` | EntryClass；其 `forward()` 传 `hidden_states_before_norm=pre_hc_head` |

### 复用的主模型层 `models/deepseek_v4.py`

| 符号 | 内容 |
|------|------|
| `MqaAttentionBase` | MQA 基类；`__init__` 里 `assert self.compress_ratio in (0, 4, 128)`、`assert config.num_key_value_heads == 1`、建 `wq_a`/`q_norm`/`wq_b`/`wkv`/`wo_a`/`wo_b` |
| `MQALayer` | 继承上者；`__init__` 收 `compress_ratio_override` 并赋给 `compress_ratio=` |
| `MQALayer.__init__` 内 compressor 块 | `self.compressor = None; self.indexer = None; if self.compress_ratio in (4, 128):` —— MTP ratio=0 ⇒ 都不建 |
| `MQALayer._compute_kv_to_cache()` | 调 `token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(...)`（RMSNorm+RoPE+写 SWA 三合一） |
| `MQALayer._forward_prepare()` | 另一条融合路径 `fused_qk_norm_rope_swa_store(...)`；末尾 `if not (unified and fuse_verify): kv = None` |
| `MQALayer.forward()` | `attn_k = kv if kv is not None else q`；`save_kv_cache=kv is not None`（unified 路径）/ `save_kv_cache = False`（非 unified）；逆 RoPE；分组输出投影 |
| `_apply_wo_a_bf16_matmul()` | `torch.einsum("tgd,grd->tgr", o, wo_a)`（fp8 路径走 `deep_gemm.fp8_einsum`） |
| `DeepseekV4DecoderLayer` | 把 MQA + mHC + MoE 包成一层（**与 `MQALayer` 是两个不同的类**） |
| `DeepseekV4DecoderLayer.hc_post()` | 4 通道残差混合，输入输出都是 `[T, hc_mult, H]`，**不收拢** |
| `DeepseekV4ForCausalLM.forward()` | target 侧同样递出 `pre_hc_head` 作为 `hidden_states_before_norm` |

### draft 驱动 `speculative/eagle_worker_v2.py`

| 符号 | 内容 |
|------|------|
| `EagleDraftWorker.alloc_memory_pool()` | 接收 target 的 `req_to_token_pool` + `token_to_kv_pool_allocator` |
| `EagleDraftWorker.draft()` | 外层入口；`n_inner = self.speculative_num_steps - 1` |
| `EagleDraftWorker.draft_forward()` | decode 多步循环；`if i == self.speculative_num_steps - 1: break`（skip-last，**无注释**）；`forward_batch.out_cache_loc = out_cache_loc[i]`；`spec_info.hidden_states = hidden_states`；`attn_backends[i]`；`draft_topk1_postprocess` / `torch.argmax`；`hidden_states = logits_output.hidden_states` |
| `EagleDraftWorker._draft_extend_for_prefill()` | prefill 填 KV；`_eagle_prefill_tail_tokens(...)` + input_ids 左移补尾；`EagleDraftExtendInput(...)`；`capture_hidden_mode`（非 STANDALONE ⇒ `LAST`）；`ForwardBatch.init_new(..., return_hidden_states_before_norm=False)` |
| `EagleDraftWorker.forward_batch_generation()` | EXTEND / DECODE 分支入口（DECODE 分支：draft → verify → `_draft_extend_for_decode`） |

### 相关支撑符号

| 文件 : 符号 | 内容 |
|-------------|------|
| `speculative/eagle_info.py:EagleDraftInput` | `topk_index` / `hidden_states` / `bonus_tokens` / `num_tokens_per_req` 等字段 |
| `speculative/eagle_info.py:EagleDraftExtendInput` | draft-extend 专用 spec_info |
| `speculative/eagle_utils.py:per_step_draft_out_cache_loc()` | 把整批 `out_cache_loc` reshape 成 `(num_steps, -1)` |
| `speculative/eagle_utils.py:get_draft_recurrent_hidden_state_spec_from_config()` | 返回 `model_config.spec_hidden_size`（=`4*H`），决定步间 hidden buffer 宽度 |
| `configs/model_config.py:ModelConfig._derive_model_shapes()` | 算 `spec_hidden_size = hidden_size * hc_mult`、`hc_hidden_size` |
| `layers/logits_processor.py:LogitsProcessor._get_hidden_states_to_store()` | 「`hidden_states_before_norm` provided 就一定用它」 |
| `layers/attention/deepseek_v4_backend.py:DeepseekV4AttnBackend.get_swa_out_cache_loc()` | `translate_loc_from_full_to_swa(out_cache_loc).to(torch.int32)` |
| `layers/attention/deepseek_v4_backend.py:DeepseekV4MultiStepBackend.__init__` | 建 num_steps 个 `DeepseekV4AttnBackend` 存进 `self.attn_backends` |
| `speculative/draft_utils.py` | 按 HIP/CUDA 挑 `DeepseekV4MultiStepBackend`，返回 tag `"dsv4"` |
| `arg_groups/speculative_hook.py:_auto_choose_speculative_params()` | DSV4 落到兜底分支 `return (3, 1, 4)` |

### KV 池构建 / 显存预留

| 文件 : 符号 | 内容 |
|-------------|------|
| `mem_cache/kv_cache_configurator.py:KVCacheConfigurator.configure()` | draft 复用 target 的 `memory_pool_config` |
| `mem_cache/kv_cache_configurator.py:KVCacheConfigurator._derive_pool_sizes()` | draft 时清零 `c4_max_total_num_tokens` / `c128_max_total_num_tokens` / `c4_state_pool_size` / `c128_state_pool_size` |
| `mem_cache/kv_cache_configurator.py:KVCacheConfigurator.loc_space_scale` | DCP 下 draft loc 空间 × `dcp_size` |
| `mem_cache/kv_cache_configurator.py:KVCacheConfigurator._build_dsv4_kv_pool()` | 建新的单层 SWA-only 池；`pool_cls` = `DeepSeekV4TokenToKVPool`（CUDA）/ `DSV4NPUTokenToKVPool`（NPU） |
| `mem_cache/kv_cache_configurator.py:KVCacheConfigurator._build_token_to_kv_pool_allocator()` | draft 分支**原样返回 target allocator**，只 `register_mapping`（identity 或 target 的 `full_to_swa_index_mapping`） |
| `mem_cache/deepseek_v4_memory_pool.py:DeepSeekV4TokenToKVPool` | 池类定义处（**不在** `memory_pool.py`）；`__init__` 内 `c4_logical_size = c128_size * 32`、`indexer_size = c4_logical_size`、`swa_ring_size = sliding_window + spec_extra` |
| `mem_cache/deepseek_v4_memory_pool.py:DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_fused_norm_rope()` | 融合 kernel 入口 |
| `model_executor/pool_configurator.py:DSV4PoolConfigurator` | DSV4 显存预留；投机膨胀 `bytes_per_full_token *= (T+1)/T` 在其 `__init__` 内（快照 :825-832） |
| `model_executor/model_runner_components/layer_setup.py:_compute_model_num_layers()` | draft+MTP ⇒ `num_effective_layers = num_nextn_predict_layers = 1` |
| `managers/scheduler.py:Scheduler.init_memory_pools()` | 把 target 的 pool / allocator 传给 draft |

> 勘误备注：本文早期版本与 `dsv4_mtp_centralized_execution.md` 都把 `DSV4PoolConfigurator`
> 标在 `pool_configurator.py:574` / `:507`、投机膨胀标在 `:633-640` / `:553-561`——**两者都不对**。
> 当前快照下类在 :763、膨胀块在 :825-832。这也是本文改用符号名索引的直接原因。

---

## 附：五条最容易记混的结论

1. **MTP 不读主模型 KV**，只读主模型的 `pre_hc_head`；它有自己独立的单层 SWA-only KV 池（CUDA 上是 `DeepSeekV4TokenToKVPool`，NPU 上是 `DSV4NPUTokenToKVPool`）。
2. **prefill 的 MTP input 是全部 prompt token（左移一位）**，不是那 1 个生成的 token；生成的 token 只是补在序列末尾。
3. **decode 的 MTP 每步只处理 bs 个 token**（每请求 1 个），靠循环把输出喂回输入串成 draft 链，长度 = num_steps。
4. **步间 hidden 宽度是 `4*H`（=`spec_hidden_size`=16384），不是 `H`**——传的是 mHC 4 通道展平的 `pre_hc_head`。§5 那句 `.view(n_tokens * self.hc_mult, d)` 就是铁证。
5. **`hc_post` 保形（4→4），`hc_head` 才收拢（4→1）**；而 `pre_hc_head` 是在 `hc_head` **之前**存下来的，所以下一步拿到的一定是 4 通道版本。这条和第 4 条是同一件事的两面。

