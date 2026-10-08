# DeepSeek-V4 模型结构与计算流程详解

> 本文聚焦 **模型本体的结构与前向计算流程**：`python/sglang/srt/models/deepseek_v4.py`（3128 行）及其配置 `configs/deepseek_v4.py`、压缩器 `layers/attention/dsv4/compressor.py`、索引器 `layers/attention/dsv4/indexer.py`。
>
> 关于「KV 缓存池怎么分层」「PD 分离怎么传输」「MTP/投机怎么跑」等**运行时**话题，本目录已有专门文档，本文只做交叉引用，不重复：
> - 显存池六子池：[deepseek_v4_cache_management.md](deepseek_v4_cache_management.md)
> - 显存池空间构成：[deepseek_v4_flash_pool_sizing.md](deepseek_v4_flash_pool_sizing.md)
> - PD 分离请求全生命周期：[dsv4_pd_disaggregation_request_lifecycle.md](../04_pd_disaggregation/dsv4_pd_disaggregation_request_lifecycle.md)
> - MTP 集中式执行：[dsv4_mtp_centralized_execution.md](../05_speculative_decoding/mtp_dsv4_centralized_execution.md)
>
> 阅读顺序建议：**第 1 章看全貌 → 第 2 章记住配置 → 第 3 章吃透 mHC（V4 最独特的创新）→ 第 4 章走一遍单层 → 第 5/6/7 章深入注意力与 MoE → 第 8 章看端到端图**。

---

## 目录

1. [一句话概览：V4 和 V3 到底差在哪](#1-一句话概览v4-和-v3-到底差在哪)
2. [顶层结构与关键配置](#2-顶层结构与关键配置)
3. [mHC 多隐藏通道残差流（最核心的创新）](#3-mhc-多隐藏通道残差流最核心的创新)
4. [单层 DecoderLayer 的完整前向](#4-单层-decoderlayer-的完整前向)
5. [MQA 注意力层详解](#5-mqa-注意力层详解)
6. [三种压缩注意力：ratio 0 / 4 / 128](#6-三种压缩注意力ratio-0--4--128)
7. [MoE FFN 层](#7-moe-ffn-层)
8. [端到端数据流图](#8-端到端数据流图)
9. [附：形状速查表与易错点](#9-附形状速查表与易错点)

---

## 1. 一句话概览：V4 和 V3 到底差在哪

DeepSeek-V4 仍然是「**低秩注意力 + MoE FFN**」的老骨架，但在三个维度上做了根本性改造。先记住这张对比表，后面每一章都是在展开它：

| 维度 | DeepSeek-V3 (MLA) | DeepSeek-V4 (本文) | 影响 |
|---|---|---|---|
| **残差流** | 单条 `[T, hidden]` | **mHC：`[T, hc_mult=4, hidden]` 四通道并行残差** | 每层入口要「混合下采样」、出口要「混合上采样」，是 V4 最独特的结构 |
| **注意力** | 统一 MLA，全序列 dense | **逐层三选一压缩注意力** `compress_ratio ∈ {0,4,128}`：SWA 近窗 / CSA 稀疏 / HCA 全压缩块 | 每层显存与算力占用不同，见第 6 章 |
| **KV 头数** | MLA 隐式多头 | **MQA：`num_key_value_heads = 1`** + 分组输出投影 `o_groups=8` | KV 缓存极小，输出投影分 8 组低秩 |
| **稀疏检索** | 无 | **C4Indexer**：ratio=4 的层用 top-512 页选择做稀疏注意力 | 长上下文下算力随选中页数而非序列长度增长 |
| **FFN** | MoE（部分层 dense） | 每层都是 **MoE**（`first_k_dense_replace=0`），256 路由专家 top-6 + 1 共享专家 | 见第 7 章 |

一句话：**V4 = 把「单条残差」换成「四通道残差 + 每层通道混合」，把「统一 MLA」换成「逐层可选的三档压缩 MQA + 稀疏索引」，FFN 全 MoE。**

### 类的层次结构

```
DeepseekV4ForCausalLM            (deepseek_v4.py:2376  —— 顶层，含 lm_head / logits_processor)
└── DeepseekV4Model              (deepseek_v4.py:2039  —— embed → 43 层 → norm，管 mHC 通道展开/收拢)
    ├── embed_tokens             (VocabParallelEmbedding)
    ├── layers[0..42]            (DeepseekV4DecoderLayer  —— 每层一个)
    │   ├── self_attn : MQALayer (deepseek_v4.py:549     —— 注意力子层)
    │   │   ├── compressor       (Compressor,  ratio∈{4,128} 才建)
    │   │   ├── indexer          (C4Indexer,   仅 ratio==4 才建)
    │   │   └── attn_mqa         (RadixAttention → DeepseekV4AttnBackend)
    │   └── mlp : DeepseekV2MoE  (deepseek_v2.py，每层都是 MoE)
    └── norm                     (RMSNorm) + hc_head（把 4 通道收拢成 1）
```

`EntryClass = [DeepseekV4ForCausalLM]`（deepseek_v4.py:3088）是模型注册入口。注意 `MQALayer` 继承自 `MqaAttentionBase`（deepseek_v4.py:376），基类负责建权重与 RoPE 频率表，子类负责前向逻辑。

---

## 2. 顶层结构与关键配置

所有默认值来自 `configs/deepseek_v4.py`（这是 SGLang 侧的默认，真实值以 checkpoint 的 `config.json` 为准；`DeepSeek-V4-Flash-FP8` 即用这套）。

### 2.1 基本规模

| 字段 | 默认 | 含义 |
|---|---|---|
| `hidden_size` | 4096 | 隐藏维度（单通道） |
| `num_hidden_layers` | 43 | 解码层数 |
| `num_attention_heads` | 64 | 注意力头数 |
| `num_key_value_heads` | 1 | **KV 头数=1 → MQA**（所有 Q 头共享一份 KV） |
| `vocab_size` | 129280 | 词表 |
| `max_position_embeddings` | 65536 | 最大位置（YaRN 外推） |
| `rms_norm_eps` | 1e-6 | RMSNorm ε |

### 2.2 注意力低秩维度（MLA 风格）

| 字段 | 默认 | 含义 |
|---|---|---|
| `q_lora_rank` | 1024 | Q 的低秩瓶颈：`wq_a` 下投影到 1024 |
| `qk_nope_head_dim` | 448 | Q/K 不做 RoPE 的部分 |
| `qk_rope_head_dim` | 64 | Q/K 做 RoPE 的部分 |
| `head_dim`（派生） | **512** = 448+64 | 单头维度（代码 deepseek_v4.py:409-410 组装并 assert==config.head_dim） |
| `v_head_dim` | 512 | V 头维度（=head_dim，MLA 里 K/V 合一） |
| `kv_lora_rank` | 512 | KV 低秩（`wkv` 下投影到 head_dim=512，MQA 只 1 头） |
| `o_lora_rank` | 1024 | 输出投影低秩 |
| `o_groups` | 8 | **输出投影分 8 组低秩**（`wo_a`/`wo_b` 见 §5.4） |
| `window_size` | 128 | SWA 滑动窗口大小 |

### 2.3 压缩注意力与稀疏索引

| 字段 | 默认 | 含义 |
|---|---|---|
| `compress_ratios` | `[]`（由 checkpoint 逐层给） | **逐层压缩比**，每个元素 ∈ {0,4,128}，选定该层注意力档位（第 6 章） |
| `compress_rope_theta` | 40000 | 压缩层专用的 RoPE base（非压缩层用 `rope_theta=10000`，deepseek_v4.py:528-530） |
| `index_head_dim` | 128 | C4Indexer 打分头维度 |
| `index_n_heads` | 64 | C4Indexer 头数 |
| `index_topk` | 512 | **稀疏页 top-k=512**（ratio=4 的层选中多少页去做注意力） |

> `DeepSeek-V4-Flash-FP8` 的 43 层实测切分：**21 层 ratio=4（CSA）+ 20 层 ratio=128（HCA）+ 2 层 ratio=0（纯 SWA）**，详见 [deepseek_v4_flash_pool_sizing.md](deepseek_v4_flash_pool_sizing.md)。

### 2.4 mHC 与 MoE

| 字段 | 默认 | 含义 |
|---|---|---|
| `hc_mult` | **4** | **mHC 通道数**：残差流有 4 条并行通道（第 3 章） |
| `hc_sinkhorn_iters` | 20 | 通道混合矩阵的 Sinkhorn 归一化迭代次数 |
| `hc_eps` | 1e-6 | mHC 数值 ε |
| `n_hash_layers` | 3 | （mHC 相关的哈希层配置） |
| `n_routed_experts` | 256 | MoE 路由专家数 |
| `num_experts_per_tok` | 6 | 每 token top-6 专家 |
| `n_shared_experts` | 1 | 共享专家数 |
| `moe_intermediate_size` | 2048 | 每个专家的 FFN 中间维度 |
| `first_k_dense_replace` | 0 | **前 0 层用 dense FFN → 即每层都是 MoE** |
| `moe_layer_freq` | 1 | 每层都是 MoE |
| `topk_method` | `noaux_tc` | 路由方法（无辅助损失，group-limited） |
| `scoring_func` | `sqrtsoftplus` | 路由打分函数 |
| `routed_scaling_factor` | 1.5 | 路由权重缩放 |

---

## 3. mHC 多隐藏通道残差流（最核心的创新）

### 3.1 直觉：把「一条河」变成「四条并行河」

传统 Transformer 的残差流是**一条**向量 `[T, hidden]`，每个子层做 `x = x + sublayer(norm(x))`。

V4 把残差流拓宽成 **`hc_mult=4` 条并行通道**：`[T, 4, hidden]`。你可以想象成有 4 条并行的「信息河道」同时往前流。这样做的目的是让模型在深度方向有更大的表示带宽，同时又不把单个子层的计算量放大 4 倍——因为子层（注意力/FFN）仍然只在**一条**融合后的 `[T, hidden]` 上计算。

于是每个子层前后各夹一个「混合」操作：

```
                4 通道残差 [T,4,H]
                     │
              ┌──────┴───────┐  hc_pre：把 4 通道「混合下采样」成 1 条 [T,H]
              │   RMSNorm     │      （学习一组通道权重 pre，加权求和）
              └──────┬───────┘
                     ▼
              子层（Attention 或 MoE），只在 [T,H] 上算
                     │  sublayer_out [T,H]
              ┌──────┴───────┐  hc_post：把 1 条子层输出 + 原 4 通道残差
              │  post / comb  │      「混合上采样」回 4 通道 [T,4,H]
              └──────┬───────┘
                     ▼
                4 通道残差 [T,4,H]  ——→ 进入下一个子层
```

- **入口起点**（deepseek_v4.py:2259-2260）：embedding 出来后直接 `unsqueeze(1).repeat(1, hc_mult, 1)` → 把 `[T, H]` 复制成 `[T, 4, H]`，4 条河的起点完全相同。
- **出口终点**（deepseek_v4.py:2360-2362）：最后一层出来后，`hc_head` 把 4 通道收拢回 1 条 `[T, H]`，再过 `norm` 送进 lm_head。

### 3.2 三个混合算子的数学

三个算子都是「先对展平的 `[T, 4*H]` 做 RMSNorm，再线性投影出混合系数」。参数由 `make_hc_mixing_params`（deepseek_v4.py:245）和 `make_hc_head_params`（:262）构造。

**参数维度**（`mix_hc = (2+hc_mult)*hc_mult`，`hc_dim = hc_mult*hidden`）：

```python
mix_hc = (2 + 4) * 4 = 24        # = pre(4) + post(4) + comb(4×4=16)
hc_dim = 4 * 4096 = 16384
hc_fn  : [24, 16384]  float32    # 混合投影矩阵（attn 和 ffn 各一份：hc_attn_fn / hc_ffn_fn）
```

**① `hc_pre`（下采样，deepseek_v4.py:1383）** —— 把 4 通道 → 1 条：

```
x_flat = residual.flatten(1)                 # [T, 4H]
mixes  = Linear(hc_fn, RMSNorm(x_flat))      # [T, 24]
pre, post, comb = split_sinkhorn(mixes, ...) # pre:[T,4]  post:[T,4]  comb:[T,4,4]
y = Σ_c ( pre[:,c] · residual[:,c,:] )       # [T, H]  ← 送入子层
```

`split_sinkhorn`（`hc_split_sinkhorn`，deepseek_v4.py:1484）把 24 维拆成三份，并对 `comb` 做 `hc_sinkhorn_iters=20` 次 Sinkhorn 归一化（逼近双随机矩阵，保证通道混合是「重分配」而非「放大」）。**`post` 和 `comb` 在这里算好，但要等子层跑完在 `hc_post` 里才用**。

**② `hc_post`（上采样，deepseek_v4.py:1504）** —— 把子层输出 + 原残差 → 回 4 通道：

```
out[:,c,:] = post[:,c] · sublayer_out              # 子层输出注入第 c 通道
           + Σ_c' ( comb[:,c,c'] · residual[:,c',:] )  # 原 4 通道按 comb 重分配
        →  [T, 4, H]
```

直觉：`post` 决定「新算出来的子层结果分给每条河多少」，`comb` 决定「旧的 4 条河如何互相搅拌后继续往下流」。这就是一个**可学习的通道级门控 + 混合**。

**③ `hc_head`（收拢，deepseek_v4.py:2111 / `hc_head_torch`:273）** —— 4 通道 → 1 条，用于最终输出：

```
mixes = Linear(hc_head_fn, RMSNorm(x.flatten(-2)))   # [T, 4]
pre   = sigmoid(mixes · scale + base) + eps          # [T, 4]  sigmoid 门控
y     = Σ_c ( pre[:,c] · x[:,c,:] )                  # [T, H]
```

### 3.3 跨层融合：省掉一次「上采样再下采样」

朴素实现每层要做：`... → hc_post(上) → hc_pre(下) → ...`，相邻两层之间的「上采样再立刻下采样」是冗余的。V4 打开 `use_fused_mhc_post_pre`（deepseek_v4.py:1351/2102）后，把「**上一层的 hc_post + 本层的 hc_pre**」融成一个 kernel `mhc_fused_post_pre`（deepseek_v4.py:1564）。

后果是 `DeepseekV4DecoderLayer.forward` 的返回值不再是「完成的 4 通道残差」，而是**延迟的 hc_post 状态** `(hidden_states, residual, post, comb)`（deepseek_v4.py:1682）：

- 每层把自己 FFN 的 hc_post **推迟**，连同 `residual/post/comb` 传给下一层；
- 下一层入口用 `mhc_fused_post_pre` 一次性「完成上层 hc_post + 做本层 hc_pre」；
- **最后一层**没有下一层来接盘，由 `DeepseekV4Model.forward` 补一次 `last_layer.hc_post(...)`（deepseek_v4.py:2358-2361）收尾。

所以在 `DeepseekV4Model.forward` 的层循环里（deepseek_v4.py:2327-2340）你会看到 `prev_residual/prev_post/prev_comb` 像接力棒一样在层间传递。这是纯粹的**性能优化，不改变数学结果**——关掉 fused 时走 §3.2 的朴素 `hc_pre`/`hc_post` 分离路径（deepseek_v4.py:1655-1677）。

> **易错点**：读代码时若看到 `layer.forward` 返回 4 元组、且 `hidden_states` 形状还是 `[T,4,H]` 未收拢，不要以为 mHC 没做完——它是被**故意推迟**到下一层/模型末尾了。

### 3.4 mHC 全景图

```
          input_ids
              │ embed_tokens                         [T, H]
              │ .unsqueeze(1).repeat(1,4,1)          [T, 4, H]   ← 4 通道起点
              ▼
   ┌──────────────────── Layer 0 ────────────────────┐
   │ hc_pre(attn) → Attention → hc_post(attn)         │  子层内部只见 [T,H]
   │ hc_pre(ffn)  → MoE       → (hc_post 推迟) ───────┼──┐ 传 (residual,post,comb)
   └──────────────────────────────────────────────────┘  │
   ┌──────────────────── Layer 1 ────────────────────┐  │
   │ mhc_fused_post_pre(接上层 ffn post + 本层 attn pre)│◄─┘
   │ ... 同上 ...                             (推迟) ──┼──┐
   └──────────────────────────────────────────────────┘  │
              ⋯ 43 层接力 ⋯                                │
   ┌──────────────────── Layer 42 ───────────────────┐  │
   │ ...                                    (推迟) ────┼──┘
   └──────────────────────────────────────────────────┘
              │ Model.forward 补 last_layer.hc_post   [T, 4, H]
              │ hc_head（sigmoid 门控收拢）            [T, H]
              │ norm (RMSNorm)                         [T, H]
              ▼
        送 lm_head → logits
```

---

## 4. 单层 DecoderLayer 的完整前向

`DeepseekV4DecoderLayer.forward`（deepseek_v4.py:1545）。每层结构 = **注意力子层 + MoE 子层**，两个子层各夹一对 mHC pre/post。构造时（deepseek_v4.py:1291）建三样东西：`self_attn=MQALayer`、`mlp=DeepseekV2MoE`、两个 `RMSNorm`（`input_layernorm` / `post_attention_layernorm`）+ mHC 混合参数。

### 4.1 朴素路径（`use_fused=False`，最易懂）

按顺序读 deepseek_v4.py:1585-1677：

```
输入: hidden_states [T,4,H]（4 通道残差）

# ── 注意力子层 ──
residual = hidden_states
h, post, comb = hc_pre(hidden_states, hc_attn_*, norm=input_layernorm)   # 4→1, [T,H]
h = self_attn(x=h, positions, forward_batch)                            # MQA 注意力, [T,H]
h = hc_post(h, residual, post, comb)                                    # 1→4, [T,4,H]

# ── FFN(MoE) 子层 ──
residual = h
h, post, comb = hc_pre(h, hc_ffn_*, norm=post_attention_layernorm)      # 4→1, [T,H]
h = _run_moe_ffn_dp_sync(h, forward_batch, ...)                          # MoE, [T,H]
h = hc_post(h, residual, post, comb)                                    # 1→4, [T,4,H]
return h, None, None, None
```

注意 `hc_pre` 里传了 `norm=input_layernorm`：在 TileLang/融合路径下，**RMSNorm 被 fuse 进 mHC pre kernel**（返回的 `norm_fused=True`），此时不再单独调 `input_layernorm`；否则回退到显式 `hidden_states = self.input_layernorm(hidden_states)`（deepseek_v4.py:1603/1667）。

### 4.2 融合路径（`use_fused=True`，生产默认）

deepseek_v4.py:1561-1682。逻辑等价，但把「上一层 ffn 的 hc_post」和「本层 attn 的 hc_pre」合成 `mhc_fused_post_pre`，并把本层 ffn 的 hc_post 推迟返回：

```
# 入口：若上层传来 prev_residual，一次 kernel 完成「上层 ffn hc_post + 本层 attn hc_pre + input_layernorm」
residual, post, comb, h = mhc_fused_post_pre(hidden_states, prev_residual, prev_post, prev_comb, hc_attn_*, ...)

h = self_attn(x=h, positions, forward_batch, x_quant=None)              # 注意力

# attn hc_post + ffn hc_pre + post_attention_layernorm 融合
residual, h, post, comb, _ = try_fused_hc_post_pre(h, residual, post, comb, hc_ffn_*, ...)

h = _run_moe_ffn_dp_sync(h, forward_batch, ...)                          # MoE

return h, residual, post, comb    # ffn hc_post 推迟给下一层
```

### 4.3 `_run_moe_ffn_dp_sync`：并行通信的编排壳

`_run_moe_ffn_dp_sync`（deepseek_v4.py:1684）不改变数学，只根据并行模式把 MoE 前后的 all-gather / reduce-scatter / all-to-all 编排好，最终都调到 `self.mlp(hidden_states, forward_batch, input_ids=..., skip_shared_experts=...)`（deepseek_v4.py:1777）。几条分支：

| 场景判定 | 处理 |
|---|---|
| `_use_cp`（DSA prefill 上下文并行） | 走 CP gather/scatter |
| `attn_dp_size>1` 且无 EP a2a 后端（`_use_tp_moe_gather`） | 先在本地切片算共享专家，再 gather 到全局做路由专家，最后 scatter 回来 |
| EP（DeepEP a2a 后端） | 共享专家与路由专家分解到 all-to-all 通道分别算 |

DP-attention 下，共享专家可能先在本地 slice 上单独算 `self.mlp._forward_shared_experts(local_hidden_states)`（deepseek_v4.py:1764），再把 `skip_shared_experts=True` 传给主 `self.mlp(...)` 避免重复。这些是分布式细节，单卡时基本 no-op。

---

## 5. MQA 注意力层详解

`MQALayer`（deepseek_v4.py:549，继承 `MqaAttentionBase`:376）。这是 V4 注意力的主体。它是 **MLA 风格的低秩投影 + MQA（单 KV 头）+ 逐层压缩档位**的结合体。

### 5.1 权重清单（`MqaAttentionBase.__init__`）

| 权重 | 形状（in→out） | 作用 |
|---|---|---|
| `wq_a` / `wqkv_a` | `H → q_lora_rank(1024)`（或融合到 `1024+512`） | Q（及可选 KV）下投影到低秩瓶颈。融合与否由 `SGLANG_OPT_FUSE_WQA_WKV` 控制（deepseek_v4.py:459） |
| `q_norm` | RMSNorm(1024) | Q 低秩后归一化 |
| `wq_b` | `1024 → n_heads(64)·head_dim(512)` | Q 上投影到全部头（按 attn_tp 切分为 `n_local_heads`） |
| `wkv` | `H → head_dim(512)` | **KV 下投影，MQA 只 1 个 KV 头**（`num_key_value_heads==1` assert，:433） |
| `kv_norm` | RMSNorm(512) | KV 归一化 |
| `wo_a` | `n_heads·head_dim/o_groups → o_groups·o_lora_rank` | **分组输出低秩下投影**（8 组） |
| `wo_b` | `o_groups·o_lora_rank(8·1024) → H` | 输出低秩上投影回 hidden |
| `attn_sink` | `[n_heads]` float32 | 每头一个 attention sink（softmax 分母的额外常数项，稳定长上下文） |

RoPE 频率表 `freqs_cis` 用 YaRN 外推预计算（`precompute_freqs_cis`，deepseek_v4.py:536），压缩层用 `compress_rope_theta=40000`、非压缩层用 `rope_theta=10000`（deepseek_v4.py:528-530）。

按 `compress_ratio` 选择性地再建两个子模块（deepseek_v4.py:602-625）：
- `compress_ratio ∈ {4,128}` → 建 `Compressor`（写压缩 KV 池）；
- `compress_ratio == 4` → 额外建 `C4Indexer`（稀疏页选择）。

### 5.2 前向骨架 `MQALayer.forward`（deepseek_v4.py:1083）

```
forward(x=[T,H], positions, forward_batch):
  ① _forward_prepare(...) → 算出 q（含 RoPE、写好 SWA KV 缓存、写好压缩池），返回 (q, kv)
  ② attn_backend.forward(q, k, v, layer=attn_mqa, compress_ratio, attn_sink)  → o  ← FlashMLA
  ③ 对 o 的 rope 段做 **逆 RoPE**（fused_rope_inplace inverse，:1225）
  ④ 分组输出投影：o[T, o_groups, ...] → wo_a（einsum 分组）→ wo_b → [T,H]
  ⑤ attn_tp>1 时 all_reduce
  return o [T,H]
```

TP>1 时会把 per-rank 头 pad 到 64（deepseek_v4.py:1115）以命中 FlashMLA 的 `decode::head64` 快 kernel，`attn_sink` 同步切片并 pad（:1126-1134）。

### 5.3 `_forward_prepare`：Q/KV 投影 + RoPE + 写缓存（deepseek_v4.py:899）

这是注意力最密的部分。核心步骤（省略 NPU/CP/HIP 分支）：

```
# Q 路径
q_lora = wq_a(x)               # [T, 1024]
q_lora = q_norm(q_lora)
q      = wq_b(q_lora)          # [T, 64, 512]
q      = fused_q_norm_rope(q, freqs_cis, positions)   # 每头 RMSNorm-self + RoPE，融合 kernel(:667)

# KV 路径（MQA，单头）——关键：写缓存与 RoPE 融合，不留 bf16 中间量
kv = wkv(x)                    # [T, 512]
token_to_kv_pool.set_swa_key_buffer_radix_fused_norm_rope(   # (:690)
      layer_id, swa_loc, kv, kv_norm.weight, freqs_cis, positions)
  # = RMSNorm(kv) + RoPE + 直接写进 FlashMLA 分页 SWA 缓存

# 压缩/稀疏（仅对应档位的层）
if indexer:    indexer(x, q_lora, forward_batch, attn_backend)          # 写 c4 indexer 缓存 + 选页
if compressor: attn_backend.forward_core_compressor(x, ..., compressor) # 写 c4/c128 压缩 KV 池
```

三条「写 KV」的路径（deepseek_v4.py 内多分支，见 §6）：
1. **SWA 主缓存**：`set_swa_key_buffer_radix_fused_norm_rope`（每层都写，近窗全精度 KV）；
2. **压缩 KV 池**：`forward_core_compressor` → c4/c128 池（ratio∈{4,128}）；
3. **索引 KV 池**：`forward_indexer_compressor` → c4 indexer 池（仅 ratio==4，供打分选页）。

> 为什么要「fused norm+rope+写缓存」？因为 KV 只有 1 个头、维度 512，若先算出 bf16 中间张量再写缓存会多一次显存往返；融合 kernel 直接把归一化+旋转后的结果落到分页缓存，省带宽。DSA 上下文并行（CP）是唯一例外——它需要 bf16 KV 做跨 rank all-gather，走 `_compute_kv_bf16`（:700）。

### 5.4 分组输出投影（`o_groups=8` 的由来）

普通注意力输出投影是一个 `[n_heads·head_dim → H]` 大矩阵。V4 把它拆成**低秩 + 分组**（deepseek_v4.py:1233-1269）：

```
o          [T, n_heads·head_dim]
o.view →   [T, o_groups=8, n_heads·head_dim/8]
wo_a: einsum "tgd,grd->tgr"  → [T, 8, o_lora_rank=1024]   # 每组独立低秩下投影
o.flatten →                    [T, 8·1024]
wo_b →                         [T, H]                       # 上投影回 hidden
```

FP8 路径（`_FP8_WO_A_GEMM`）用 `deep_gemm.fp8_einsum` 做分组 FP8 GEMM（:1257）。这套「分 8 组、每组低秩 1024」的设计把输出投影的参数量和算力都压下来，同时保留足够表达力。

---

## 6. 三种压缩注意力：ratio 0 / 4 / 128

V4 每层的注意力档位由 `compress_ratios[layer_id]` 决定（deepseek_v4.py:424），值 ∈ {0,4,128}。三档共用同一个 `MQALayer.forward` 骨架和同一个 `DeepseekV4AttnBackend`，区别在于**读哪些 KV、要不要稀疏选页**。

| ratio | 名字 | 建 compressor? | 建 indexer? | KV 读取来源 | 直觉 |
|---|---|:---:|:---:|---|---|
| **0** | 纯 SWA | 否 | 否 | 仅 SWA 近窗（window=128）全精度 | 只看最近 128 token，最省 |
| **4** | CSA（压缩稀疏注意力） | 是 | 是 | SWA 近窗 dense + **top-512 稀疏页**（c4 压缩 KV） | 近处看全精度，远处只看索引器选中的 512 页 |
| **128** | HCA（压缩块注意力） | 是 | 否 | SWA 近窗 dense + **全部 c128 压缩块**（128× 压缩） | 近处全精度，远处看极致压缩的全历史 |

所有档位都写 SWA 近窗缓存（§5.3 第 1 条），差异在**远端历史**如何存与读。

### 6.1 Compressor：把 hidden 压成低精度 KV 块（`compressor.py:344`）

`Compressor` 负责把 hidden state 下投影成一份「压缩 KV 状态」，写进压缩 KV 池。关键参数：

- `wkv_gate`：`ReplicatedLinear(hidden → 2·coff·head_dim)`（compressor.py:374），`coff = 1 + (ratio==4)`——ratio=4 时 coff=2、ratio=128 时 coff=1。
- `ape`：`[ratio, coff·head_dim]` 的可学习**绝对位置嵌入表**（compressor.py:369），每个压缩槽一份。
- `norm`：RMSNorm(head_dim)。

`forward_compress`（compressor.py:64）把 ring buffer 视作 `[-1, ratio, last_dim]`，跑 `compress_forward` + `compress_fused_norm_rope_inplace`（RMSNorm+RoPE 融进压缩状态）。输出经 `forward_core_compressor`（compressor.py:159）写到 `token_to_kv_pool` 的：
- `c4_out_loc`（ratio=4）或 `c128_out_loc`（ratio=128）位置，默认量化成 fp8 打包（`quant_to_nope_fp8_rope_bf16_pack_triton`，compressor.py:188）。

**「压缩」的含义**：每 `ratio` 个 token 归约成一份压缩状态——ratio=4 即 4:1，ratio=128 即 128:1。序列越长，远端 KV 占用越小。

### 6.2 C4Indexer：稀疏选页（`indexer.py:790`）

只有 ratio=4 的层建 `C4Indexer`。它的任务是：在做注意力前，从全部历史压缩页里**挑出最相关的 top-512 页**，让 FlashMLA 只对这 512 页算稀疏注意力。

关键结构：`n_heads=64`、`head_dim=128`、`index_topk=512`、内嵌一个 `Compressor(ratio=4, rotate=True, is_in_indexer=True)` 写「索引 KV 缓存」。

`forward_c4_indexer`（indexer.py:561）核心步骤：

```
① compute_q: q_lora → wq_b → [T,64,128]，融合 rope+hadamard+fp8 量化
② weights = weights_proj(x)                                   # 每头一个权重
③ logits = fp8_paged_mqa_logits(q, c4_indexer_kv, weights,    # MQA 打分（q·k relu 加权）
             c4_seq_lens, page_table, ...)                     # block_size=64, head_dim=128
④ topk_transform_512(logits, ...) → c4_sparse_page_indices    # 选出 top-512 页(:754)
```

选中的 `c4_sparse_page_indices` 写进 backend 的 `core_metadata`，随后 FlashMLA 的 CSA 路径只读这些页。打分 kernel 有多套后端：默认 `deep_gemm.fp8_paged_mqa_logits`（indexer.py:652），另有 fp4 / tilelang / aiter / torch fallback / xpu triton 变体。

> 深入 KV 池的六子池布局、地址翻译、HiSparse 换入换出，见 [deepseek_v4_cache_management.md](deepseek_v4_cache_management.md)。

### 6.3 attention backend 如何用这三者

`MQALayer.forward` 步骤②统一调 `attn_backend.forward(q, k, v, layer=attn_mqa, compress_ratio=..., attn_sink=...)`（deepseek_v4.py:1205）。`DeepseekV4AttnBackend` 内部按 `compress_ratio` 分派：

- **近窗 dense**：所有档位都对 SWA window 内的全精度 KV 做 FlashMLA dense；
- **远端 extra**：ratio=4 读 `c4_sparse_page_indices` 指向的稀疏页；ratio=128 读全部 c128 压缩块；ratio=0 无 extra；
- `flash_mla_with_kvcache` 把「近窗 dense + 远端 extra」融合成一次注意力（softmax 里带 `attn_sink` 常数项）。

prefill 大批量还有 `_forward_prefill_sparse`（用 `flash_mla_sparse_fwd`）等专用路径。attention backend 的完整 kernel 编排属于**推理运行时**，本文不展开，可参考 [dsv4_pd_disaggregation_request_lifecycle.md](../04_pd_disaggregation/dsv4_pd_disaggregation_request_lifecycle.md) 第 6 层。

---

## 7. MoE FFN 层

V4 每层的 FFN 都是 MoE（`first_k_dense_replace=0`、`moe_layer_freq=1`），因此 `DeepseekV4DecoderLayer` 直接硬编码 `self.mlp = deepseek_v2.DeepseekV2MoE(..., is_deepseek_v4=True)`（deepseek_v4.py:1323），复用 V2 的 MoE 实现。

### 7.1 结构

`DeepseekV2MoE`（deepseek_v2.py:539）包含三部分：

| 组件 | 说明 |
|---|---|
| `gate`（MoEGate） | 路由器：`hidden → n_routed_experts(256)` logits，用 `noaux_tc` group-limited 选 top-6，`scoring_func=sqrtsoftplus`，`routed_scaling_factor=1.5`，按 `n_group=8`/`topk_group=8` 分组限制 |
| `experts`（FusedMoE 系列） | 256 个路由专家，每个是 SwiGLU MLP（`gate_up_proj` + `down_proj`，中间维 `moe_intermediate_size=2048`），每 token 激活 6 个 |
| `shared_experts` | 1 个共享专家（所有 token 都过），结构同 `DeepseekV2MLP` |

**共享专家融合**（deepseek_v4.py:2459 `determine_num_fused_shared_experts`）：默认把 1 个共享专家「融进」路由专家列表（作为一个恒被选中的专家 slot），省一次独立 GEMM；`disable_shared_experts_fusion` 或配置不满足时回退成独立共享专家。

### 7.2 计算流

```
h [T,H]  ← 来自 hc_pre(post_attention_layernorm)
   │
   ├── shared_experts(h)              # 共享专家：所有 token 都算（可能被融合/本地先算）
   │
   ├── gate(h) → top-6 专家 + 权重     # 路由
   │      → experts: 每 token 分发到 6 个专家，SwiGLU，加权求和
   │
   └── 两路相加 → [T,H]  ← 送 hc_post
```

DP/EP 下 `_run_moe_ffn_dp_sync`（§4.3）在此外层套 all-gather/all-to-all/reduce-scatter。`routed_experts_weights_of_layer`（deepseek_v4.py:2443）为 EPLB（专家负载均衡）暴露每层专家权重。

> MoE 本身不是 V4 独有，与 V3 一致；V4 的差异集中在残差流（mHC）与注意力（压缩 MQA + 稀疏索引）。

---

## 8. 端到端数据流图

把前面所有环节串起来，一次前向（单卡、prefill）的完整数据流：

```
input_ids [T]
   │ DeepseekV4ForCausalLM.forward (:2490)
   ▼
DeepseekV4Model.forward (:2250)
   │ embed_tokens(input_ids)                       [T, H]
   │ .repeat → 4 通道 mHC 残差                       [T, 4, H]
   ▼
┌─ for layer in layers[0..42] ──────────────────────────────────────┐
│                                                                    │
│  【注意力子层】                                                     │
│   hc_pre(input_layernorm) : 4→1                 [T, H]              │
│   MQALayer.forward:                                                │
│     wq_a→q_norm→wq_b→RoPE                        q [T,64,512]       │
│     wkv→kv_norm→RoPE→写 SWA 缓存 (fused)          KV(1头)            │
│     [ratio4] indexer 选 top-512 页 / compressor 写 c4               │
│     [ratio128] compressor 写 c128                                  │
│     FlashMLA(近窗 dense + 远端 extra + attn_sink) o [T,64,512]      │
│     逆 RoPE → wo_a(分8组低秩) → wo_b             [T, H]             │
│   hc_post : 1→4                                 [T, 4, H]          │
│                                                                    │
│  【MoE 子层】                                                       │
│   hc_pre(post_attention_layernorm) : 4→1        [T, H]             │
│   DeepseekV2MoE: gate→top6 路由专家 + 共享专家    [T, H]            │
│   hc_post : 1→4（融合路径下推迟到下一层）         [T, 4, H]          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
   │ (融合路径) 补最后一层 hc_post                   [T, 4, H]
   │ hc_head（sigmoid 门控收拢 4→1）                [T, H]
   │ norm (RMSNorm)                                 [T, H]
   ▼
LogitsProcessor(hidden, lm_head)                    [T, vocab=129280]
   ▼
logits → 采样 → 下一 token
```

**三个「维度变换」的关键节点**，记住它们就抓住了 V4 的骨架：
1. **通道维**：`[T,H]` ⇄ `[T,4,H]`，由 mHC 的 pre/post/head 在每个子层边界和模型两端完成（第 3 章）。
2. **头/低秩维**：`H(4096)` → `q_lora(1024)` → `64头×512` → 注意力 → `8组×1024` → `H`，MLA 风格双低秩瓶颈（第 5 章）。
3. **序列/压缩维**：远端 KV 按 `ratio` 压缩 4:1 或 128:1，ratio=4 层再用 indexer 稀疏到 top-512 页（第 6 章）。

---

## 9. 附：形状速查表与易错点

### 9.1 张量形状速查（T = token 数，H = hidden = 4096）

| 位置 | 形状 | 说明 |
|---|---|---|
| embedding 输出 | `[T, 4096]` | 单通道 |
| mHC 残差流 | `[T, 4, 4096]` | 4 通道，贯穿所有层间 |
| 子层内部（pre 之后） | `[T, 4096]` | 混合下采样成 1 条 |
| Q 低秩 | `[T, 1024]` | `wq_a` 后 |
| Q 全头 | `[T, 64, 512]` | `wq_b` 后（head_dim=448+64） |
| KV（MQA 单头） | `[T, 512]` | `wkv` 后，只 1 个 KV 头 |
| 注意力输出 o | `[T, 64, 512]` | FlashMLA 输出 |
| 输出分组 | `[T, 8, 1024]` | `wo_a` 分 8 组低秩 |
| logits | `[T, 129280]` | lm_head 后 |
| mHC 混合矩阵 `hc_*_fn` | `[24, 16384]` | 24=(2+4)·4，16384=4·4096 |

### 9.2 易错点清单

1. **hc_mult=4 是「残差通道」不是「注意力头」**。注意力头是 64（`num_attention_heads`），两者无关。
2. **head_dim=512 不在 config 显式字段里**——它由 `qk_nope(448)+qk_rope(64)` 组装（deepseek_v4.py:409），真实来自 checkpoint 的 `head_dim` 字段，代码 assert 相等（:432）。
3. **KV 只有 1 个头（MQA）**，别按 MLA 多头理解；`num_key_value_heads==1` 有 assert（:433）。
4. **融合路径下 `layer.forward` 返回 4 元组且 hidden 还是 `[T,4,H]`**——ffn 的 hc_post 被故意推迟到下一层/模型末尾，不是漏做（§3.3）。
5. **`compress_ratios` 在 SGLang 默认 config 里是空 `[]`**，真实逐层值来自 checkpoint；Flash 模型是 21×4 + 20×128 + 2×0。
6. **`compress_rope_theta=40000` 仅用于压缩层**，非压缩层用 `rope_theta=10000`（:528-530）。
7. **KV 写缓存与 RMSNorm+RoPE 是融合 kernel**（`set_swa_key_buffer_radix_fused_norm_rope`），不要去找单独的「归一化后 KV」中间张量——只有 DSA-CP 分支才留 bf16 KV。
8. **共享专家可能被「融进」路由专家**（`num_fused_shared_experts`），读权重加载/EPLB 代码时注意这个重映射（deepseek_v2.py 权重加载）。

### 9.3 关键文件与行号索引

| 关注点 | 文件:行 |
|---|---|
| 顶层入口 / lm_head | `models/deepseek_v4.py:2376` `DeepseekV4ForCausalLM` |
| 模型主体 / 层循环 / mHC 展开收拢 | `models/deepseek_v4.py:2039` `DeepseekV4Model`，forward :2250 |
| 单层前向（mHC + attn + moe） | `models/deepseek_v4.py:1545` `DeepseekV4DecoderLayer.forward` |
| mHC 三算子 | `hc_pre` :1383 / `hc_post` :1504 / `hc_head` :2111；参数 `make_hc_mixing_params` :245 |
| 注意力层 | `models/deepseek_v4.py:549` `MQALayer`，基类 `MqaAttentionBase` :376 |
| Q/KV 投影+RoPE+写缓存 | `_forward_prepare` :899；分组输出投影 :1233-1269 |
| 压缩器 | `layers/attention/dsv4/compressor.py:344` `Compressor` |
| 稀疏索引器 | `layers/attention/dsv4/indexer.py:790` `C4Indexer`，`forward_c4_indexer` :561 |
| MoE | `models/deepseek_v2.py:539` `DeepseekV2MoE`（V4 复用） |
| 配置 | `configs/deepseek_v4.py:47` `DeepSeekV4Config` |
| MTP draft 层 | `models/deepseek_v4_nextn.py`（强制 ratio=0，只用 SWA） |

### 9.4 与其他文档的分工

| 想了解 | 看哪篇 |
|---|---|
| **模型结构与前向**（本文） | 本文 |
| KV 缓存六子池 / 地址翻译 / HiSparse | deepseek_v4_cache_management.md |
| 显存池 per-token 空间构成 | deepseek_v4_flash_pool_sizing.md |
| PD 分离请求生命周期 / KV 传输 | dsv4_pd_disaggregation_request_lifecycle.md |
| MTP / 投机解码集中式执行 | dsv4_mtp_centralized_execution.md、mtp_speculative_decoding_architecture.md |
| 部署参数 | deepseek_v4_deployment_guide.md |

---

*本文档基于 SGLang main 分支源码梳理（`models/deepseek_v4.py` 3128 行），聚焦模型本体结构与计算流程。运行时/调度/传输话题见上表交叉引用。*









