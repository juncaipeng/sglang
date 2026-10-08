# DeepSeek V4 模型：层与算子系统梳理

> 目标：把 `python/sglang/srt/models/deepseek_v4.py` 里 DeepSeek V4 的**每一层、每一个算子**从上到下讲清楚。
> 阅读顺序建议：先看第 1~3 节建立"骨架"直觉，再按需精读第 4 节以后的算子细节。
> 相关文件：
> - 模型主体：`python/sglang/srt/models/deepseek_v4.py`
> - 配置：`python/sglang/srt/configs/deepseek_v4.py`
> - DSA 压缩/稀疏：`python/sglang/srt/layers/attention/dsv4/compressor.py`、`indexer.py`
> - MHC 融合核：`python/sglang/kernels/ops/layernorm/mhc.py`、`mhc_head.py`
> - MoE（复用 V2）：`python/sglang/srt/models/deepseek_v2.py`

---

## 1. 一句话概览

DeepSeek V4 相比 DeepSeek V2/V3，在"标准 MoE Transformer"骨架上叠加了 **三项结构性创新**：

| 创新 | 传统做法 | V4 的做法 | 对应代码 |
|------|----------|-----------|----------|
| **MHC（多流超连接 Hyper-Connections）** | 单条残差流 `x + block(x)` | 把残差扩成 `hc_mult=4` 条并行流，block 前后用可学习权重 + Sinkhorn 归一化混合 | `hc_pre` / `hc_post` / `hc_head` |
| **DSA（DeepSeek 稀疏注意力）** | 全序列 KV 做注意力 | 先把 KV 压缩（Compressor 4x / 128x），再用 indexer 打分 top-k 选 page（C4Indexer），主注意力只算被选中的 page | `Compressor` / `C4Indexer` / `attn_mqa` |
| **MQA + 低秩输出投影** | MHA/GQA，`o_proj` 满秩 | `num_key_value_heads=1`（MQA），输出投影拆成分组低秩 `wo_a`（升/降秩）+ `wo_b` | `MQALayer` |

其余部分（MoE 路由、共享专家、TP/DP/EP/PP 并行）基本沿用 DeepSeek V2 的实现（`DeepseekV2MoE`）。

### 关键配置（`DeepSeekV4Config`，默认值）

```
hidden_size            = 4096        num_hidden_layers      = 43
num_attention_heads    = 64          num_key_value_heads    = 1     # MQA
q_lora_rank            = 1024        kv_lora_rank           = 512
qk_nope_head_dim       = 448         qk_rope_head_dim       = 64    # head_dim = 448+64 = 512
v_head_dim             = 512         o_lora_rank            = 1024  o_groups = 8

# MoE
n_routed_experts       = 256         num_experts_per_tok    = 6     n_shared_experts = 1
n_group = 8  topk_group = 8          topk_method = noaux_tc         scoring_func = sqrtsoftplus
routed_scaling_factor  = 1.5         moe_intermediate_size  = 2048  first_k_dense_replace = 0

# DSA 稀疏注意力
index_head_dim = 128   index_n_heads = 64   index_topk = 512   window_size = 128
compress_ratios: List[int]   # 每层用哪个压缩比（0 / 4 / 128）
compress_rope_theta = 40000

# MHC 超连接
hc_mult = 4            hc_sinkhorn_iters = 20     hc_eps = 1e-6      n_hash_layers = 3
```

> 注意：`head_dim = qk_nope_head_dim + qk_rope_head_dim = 448 + 64 = 512`，注意力头维度很大；`num_key_value_heads=1` 表示 KV 只有 1 个头（MQA）。

---

## 2. 整体前向骨架（自顶向下）

```
DeepseekV4ForCausalLM.forward
  └── DeepseekV4Model.forward
        1. embed_tokens(input_ids)                      # (T, hidden)
        2. hidden_states.unsqueeze(1).repeat(1,hc_mult,1)# (T, 4, hidden)  ← 复制成 4 条残差流
        3. for layer in layers[0..42]:                  # 每层见第 3 节
              DeepseekV4DecoderLayer.forward             # 维持 (T, 4, hidden)
        4. hc_head(...)                                  # (T, 4, hidden) → (T, hidden)  收缩回单流
        5. norm(hidden_states)                           # 最终 RMSNorm
  └── logits_processor(hidden_states, lm_head)           # → logits
```

要点：
- **进入层循环前** hidden state 从 `(T, hidden)` 复制成 `(T, hc_mult=4, hidden)`（`deepseek_v4.py:3225`），整整 43 层始终保持这个三维形状。
- **PP 通信**时 3D 张量会 flatten 成 `(T, 4*hidden)` 传输，对端再 unflatten（`3229-3233` / `3322-3324`）。
- **收尾** `hc_head` 把 4 条流加权收缩回 `(T, hidden)`（`3328`），再过 `self.norm`，同时额外返回 `pre_hc_head`（收缩前 `(T, 4*hidden)`）供 MTP / 投机解码使用。

---

## 3. 一个 DecoderLayer 的算子流（核心）

`DeepseekV4DecoderLayer.forward`（`deepseek_v4.py:2374`）对每层执行 **两次 "hc_post → hc_pre" 边界**：一次围绕注意力，一次围绕 FFN。逻辑序列（非融合、易读版本）：

```
输入 hidden_states: (T, 4, hidden)         # 4 条残差流

# ========== 注意力块 ==========
residual = hidden_states
y, post, comb, norm_fused = hc_pre(hidden_states, hc_attn_*, norm=input_layernorm)
    # y:(T,hidden)  post:(T,4)  comb:(T,4,4)  ← 把 4 流按 pre 权重聚合成单流 y
if not norm_fused: y = input_layernorm(y)   # 若 kernel 未融合 norm，则单独做
attn_out = self_attn(y)                      # MQALayer，见第 4 节 → (T, hidden)
hidden_states = hc_post(attn_out, residual, post, comb)   # 注回 4 流 → (T,4,hidden)

# ========== FFN 块 ==========
residual = hidden_states
y, post, comb, norm_fused = hc_pre(hidden_states, hc_ffn_*, norm=post_attention_layernorm)
if not norm_fused: y = post_attention_layernorm(y)
ffn_out = _run_moe_ffn_dp_sync(y)            # DeepseekV2MoE，见第 6 节 → (T, hidden)
hidden_states = hc_post(ffn_out, residual, post, comb)    # → (T,4,hidden)

返回 hidden_states 供下一层
```

**跨层融合优化（`use_fused_mhc_post_pre`）**：开启后，上一层的 `hc_post` 与本层的 `hc_pre` 合并成单个 `mhc_fused_post_pre` kernel（`apply_mhc_post_pre_boundary`，`2400` / `2496`），把 `prev_residual/prev_post/prev_comb` 传给下一层。此时最后一层跑完后需在 `DeepseekV4Model.forward` 里补一次 `hc_post`（`3317-3320`）。TBO（two-batch-overlap）路径禁用跨层融合，每层自包含。

### 3.1 每层含哪些子模块（`__init__`，`2058`）

| 子模块 | 类型 | 作用 |
|--------|------|------|
| `self.self_attn` | `MQALayer` | MQA + DSA 注意力（见第 4 节） |
| `self.mlp` | `DeepseekV2MoE(is_deepseek_v4=True)` | MoE FFN（见第 6 节） |
| `self.input_layernorm` | `RMSNorm(hidden)` | 注意力前归一化 |
| `self.post_attention_layernorm` | `RMSNorm(hidden)` | FFN 前归一化 |
| `hc_attn_fn/base/scale`、`hc_ffn_fn/base/scale` | `nn.Parameter` | MHC 混合参数（见第 5 节） |

---

## 4. 注意力层 MQALayer 的算子分解

`MQALayer`（继承 `MqaAttentionBase`，`deepseek_v4.py:625` / `904`）。这是全模型算子最密集的地方。

### 4.1 权重/子模块清单（`__init__`）

| 名称 | 形状（逻辑） | 作用 |
|------|--------------|------|
| `wq_a` (或融合 `wqkv_a`) | `hidden → q_lora_rank(1024)` | Q 的低秩下投影（A） |
| `q_norm` | `RMSNorm(q_lora_rank)` | Q 低秩表示归一化 |
| `wq_b` | `q_lora_rank → n_heads*head_dim (64*512)` | Q 升维成多头（B），列并行 |
| `wkv` | `hidden → head_dim(512)` | KV 投影（MQA：单头，512 维） |
| `kv_norm` | `RMSNorm(head_dim)` | KV 归一化 |
| `wo_a` | `(n_heads*head_dim/o_groups) → o_groups*o_lora_rank` | 输出**分组低秩**升投影（A），列并行 |
| `wo_b` | `o_groups*o_lora_rank → hidden` | 输出降投影（B），行并行（可 all-reduce） |
| `attn_sink` | `Parameter(n_heads,)` fp32 | 注意力 sink（softmax 归一化偏置） |
| `attn_mqa` | `RadixAttention(n_local_heads, head_dim, scale, num_kv_heads=1)` | 主注意力核 |
| `compressor` | `Compressor` | 仅 `compress_ratio∈{4,128}` 层：KV 压缩（见 4.4） |
| `indexer` | `C4Indexer` | 仅 `compress_ratio==4` 层：稀疏 page 选择（见 4.5） |
| `rotary_emb` / `freqs_cis` | RoPE 表 | 旋转位置编码 |

**输出投影为什么拆 `wo_a`+`wo_b`？** 传统 `o_proj` 是 `(n_heads*v_head_dim) → hidden` 的满秩大矩阵。V4 把它做成**分组低秩**：先按 `o_groups=8` 分组，每组用 `wo_a` 升到 `o_lora_rank=1024`，再用 `wo_b` 统一降回 `hidden`，显著减少参数与算量。`wo_a` 有多套后端（bf16 einsum / DeepGEMM fp8 / aiter mxfp8 / NPU mxfp8），但数学等价。

### 4.2 每层的压缩类型（`compress_ratio`）

`compress_ratios[layer_id]` 决定该层属于三类之一（`__init__` 断言只允许 `{0,4,128}`）：

| compress_ratio | 名称 | 是否有 compressor | 是否有 indexer | RoPE |
|----------------|------|-------------------|----------------|------|
| **0** | 纯 SWA 层 | 否 | 否 | 主 RoPE（未缩放） |
| **4** | CSA（细粒度稀疏，overlap 压缩） | 是（rotate=False） | **是** | 压缩 YaRN RoPE |
| **128** | HCA（粗粒度压缩） | 是（rotate=False） | 否 | 压缩 YaRN RoPE |

即：只有 `compress_ratio==4` 的层才做"压缩 + 稀疏 top-k 选择"；128 层只压缩不选择；0 层就是普通滑窗注意力。

### 4.3 注意力前向算子序列（`forward` + `_forward_prepare`）

以最常见的融合 fp8 路径为主线（`1690` / `1418`）：

```
x: (T, hidden)  ← 已过 input_layernorm

# --- 1. Q 侧 ---
q_lora = wq_a(x)                       # (T, 1024)          低秩下投影
q_lora = q_norm(q_lora)                # RMSNorm
q      = wq_b(q_lora)                  # (T, 64, 512)       升维成多头
                                       # fused_q_norm_rope: 每头 RMSNorm + RoPE 融合写出
# --- 2. KV 侧（MQA，单头）---
kv = wkv(x)                            # (T, 512)
# fused_qk_norm_rope_swa_store: kv_norm(RMSNorm) + RoPE + 写入 SWA/KV cache（融合原地）

# --- 3. DSA 压缩 + 稀疏选择（仅 4/128 层）---
compressor(x, forward_batch)           # 生成压缩 KV 写入压缩池（见 4.4）
indexer(x, q_lora, forward_batch)      # 打分 top-k → c4_sparse_page_indices（见 4.5，仅 4 层）

# --- 4. 主注意力 ---
o = attn_backend.forward(q, k=kv, v=kv, layer=attn_mqa,
                         compress_ratio, attn_sink)   # (T, 64, v_head_dim)
                         # 稀疏层只在被选中的 page 上计算

# --- 5. 输出投影（inverse RoPE + 分组低秩）---
fused_rope_inplace(o[..., -qk_rope_head_dim:], inverse=True)  # 逆 RoPE
o = o.view(T, o_groups=8, -1)
o = wo_a(o)  (einsum "tgd,grd->tgr")   # (T, 8, 1024)   分组升秩
o = wo_b(o.flatten(1))                 # (T, hidden)    降回
if attn_tp: o = attn_tp_all_reduce(o)  # TP 归约
return o
```

关键融合 kernel（都在 `sglang.kernels.ops.attention.dsv4`）：
- `fused_q_norm_rope`：Q 每（token,head）做 RMSNorm-self + RoPE + 写出，一个 warp 干完。
- `fused_qk_norm_rope_swa_store`：KV 的 norm + RoPE + 写 cache 融合，避免 bf16 中间张量。
- `fused_rope_inplace(..., inverse=True)`：输出侧逆向 RoPE。
- `_apply_wo_a_bf16_matmul` / `deep_gemm.fp8_einsum` / `_wo_a_fp8_mxscale`：`wo_a` 分组批量 GEMM 的多后端。

### 4.4 Compressor（KV 压缩，`compressor.py`）

作用：把 `compress_ratio` 个相邻 token 的 KV **聚合成 1 个压缩 KV**，写入环形压缩状态池。被两处复用（`is_in_indexer` 区分）：注意力主路径（`rotate=False`）与 indexer 内部（`rotate=True`）。

子模块：
- `ape`：`Parameter(ratio, coff*head_dim)` fp32 —— 压缩位置偏置（overlap 4x 时 `coff=2`）。
- `wkv_gate`：`Linear(hidden → 2*coff*head_dim)` —— 同时产 KV 与 score/gate。
- `norm`：`RMSNorm(head_dim)` —— 压缩 K 归一化。

前向算子：
```
kv_score = linear_bf16_fp32(x, wkv_gate.weight)     # bf16 输入 × 权重 → fp32
compress_forward(kv_score, ape, plan, ratio)        # 聚合 ratio 个 token + 加 ape 偏置
compress_fused_norm_rope_inplace(kv_compressed, norm, freqs_cis)  # RMSNorm + RoPE 原地
if rotate: rotate_activation(kv_compressed)          # Hadamard 旋转（indexer 专用）
→ 写入 token_to_kv_pool（set_extra_key_buffer_fused / set_index_k_*）
```

### 4.5 C4Indexer（稀疏 top-k 选择，`indexer.py`）

作用：只在 `compress_ratio==4` 层，为每个 query token 在压缩后的 KV 上打分，**top-k 选出 `index_topk=512` 个 page**，写入 `c4_sparse_page_indices`，让主注意力只读这些 page。

子模块：
- `wq_b`：`Linear(q_lora_rank → index_n_heads*index_head_dim = 64*128)` —— indexer 专用多头 query。
- `weights_proj`：`Linear(hidden → index_n_heads=64)` —— 每头一个标量门控。
- `compressor`：内嵌 `Compressor(is_in_indexer=True, ratio=4, head_dim=128, rotate=True)` —— 生成 index-K。

前向算子：
```
q = wq_b(q_lora).view(T, 64, 128)
q_indexed = fused_q_indexer_rope_hadamard_quant(q, weights, freqs_cis, positions)
            # 融合 RoPE + Hadamard 旋转 + FP8/FP4 量化 + weight 缩放
w = weights_proj(x) * weight_scale                     # (T, 64) per-head 门控

# paged MQA logits：每个 query 对每个压缩 KV 位置算标量分
logits = fp8_paged_mqa_logits(q, index_k_cache, w, seq_lens, page_table)
       # 语义：relu(bmm(K, q)) * per_head_w，sum over 64 头，再 * kv_scale，越界 mask

# top-k 转物理 page
scores.masked(越界=-inf) → topk(min(index_topk, seq_len)) → raw indices
raw → (page_idx = raw>>page_bits, offset = raw & mask) → 物理 page
→ c4_sparse_page_indices（越界填 -1）
```

`index_topk=512` 是每个 query 最终保留的压缩 page 上限；序列长度 ≤ 512 时退化为顺序全选。

---

## 5. MHC（多流超连接）算子详解

MHC = Hyper-Connections：把残差流扩成 `hc_mult=4` 条，在每个 block 前后混合。三个方法共享混合参数（`make_hc_mixing_params`，`465`）：

```
mix_hc = (2 + hc_mult) * hc_mult = (2+4)*4 = 24
hc_*_fn    : (24, 4*hidden) fp32   # 混合投影矩阵（attn 一套、ffn 一套）
hc_*_base  : (24,)          fp32   # 偏置
hc_*_scale : (3,)           fp32   # 三段各自的 scale
```

一次 `F.linear` 产出 **24 个标量**，切成三段：前 4 个 = **pre**，中 4 个 = **post**，后 16 个 = **comb**（4×4 混合矩阵）。

### 5.1 hc_pre（block 前混合，`2156`）

```
x_flat = x.flatten(1).float()                       # (T, 4*hidden)
rsqrt  = rsqrt(x_flat.square().mean(-1) + rms_eps)  # RMSNorm
mixes  = F.linear(x_flat, hc_fn) * rsqrt            # (T, 24)
pre  = sigmoid(pre_raw  * scale[0] + base[:4]) + hc_eps          # (T, 4)
post = 2 * sigmoid(post_raw * scale[1] + base[4:8])              # (T, 4)  (_MHC_POST_MULT_VALUE=2.0)
comb = sinkhorn(comb_raw * scale[2] + base[8:], iters=20)        # (T, 4, 4)
# pre-mixing：4 流按 pre 权重加权求和成单流
layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)  # (T, hidden)
return layer_input, post, comb, norm_fused
```

`post`、`comb` 被缓存，供之后的 `hc_post` 使用。`norm_fused` 表示 input/post-attn layernorm 是否已被融进 kernel（TileLang/XPU 会融合）。

### 5.2 hc_post（block 后混合，`2316`）

```
out = post.unsqueeze(-1) * x.unsqueeze(1)                        # block 输出按 post 权重注入 4 流
    + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)    # 4 条旧流按 comb 矩阵互相混合
# out: (T, 4, hidden)
```

### 5.3 hc_head（最终收缩，`3059`）

只在最后一个 PP rank、所有层跑完后执行一次，把 `(T, 4, hidden)` 收缩回 `(T, hidden)`：

```
x = x.flatten(-2).float()                          # (T, 4*hidden)
rsqrt = rsqrt(x.square().mean(-1) + norm_eps)
mixes = F.linear(x, hc_head_fn) * rsqrt            # hc_head_fn:(4, 4*hidden) → (T, 4)
pre = sigmoid(mixes * hc_scale + hc_base) + hc_eps # (T, 4)
y = sum(pre.unsqueeze(-1) * x.view(T,4,hidden), dim=-2)   # (T, hidden)
```

### 5.4 Sinkhorn 归一化（`hc_sinkhorn_iters=20`）

只作用于 `comb`（4×4 矩阵），把任意矩阵迭代逼近成**双随机矩阵**（行/列和均趋近 1），保证 4 流混合"质量守恒"、深层数值稳定。全程 fp32：

```
comb = softmax(comb, dim=行) ; comb = comb / 列和 + eps      # 首次
for _ in range(iters-1):
    comb = comb / 行和 + eps
    comb = comb / 列和 + eps
```

`pre`/`post` 不做 sinkhorn，只做 sigmoid 门控。

### 5.5 后端分派

`hc_pre` 按平台选实现：NPU `npu_hc_pre`、XPU `mhc_pre`、FlashInfer `_flashinfer_hc_pre`、TileLang `mhc_pre`、HIP/aiter `aiter.ops.mhc.mhc_pre`、CUDA 默认 `hc_pre_torch_impl + hc_split_sinkhorn`。数学等价，只是融合程度不同。

---

## 6. MoE FFN（复用 DeepseekV2MoE，`is_deepseek_v4=True`）

`DeepseekV2MoE`（`deepseek_v2.py:545`）。V4 的路由特点：

```
# 路由打分
scores = gate(x)                                   # (T, n_routed_experts=256)
scoring_func = "sqrtsoftplus"                      # V4 专用打分函数（非 sigmoid/softmax）
topk_method  = "noaux_tc"                           # 无辅助损失的分组 top-k
# 分组 top-k：先在 n_group=8 组里选 topk_group=8 组，再组内选 num_experts_per_tok=6 个专家
correction_bias = gate.e_score_correction_bias      # 路由纠偏
routed_scaling_factor = 1.5                         # 路由输出缩放

# 专家计算
routed_out = experts(x, topk_ids, topk_weights)    # 256 选 6，每 token 6 个专家
shared_out = shared_experts(x)                     # n_shared_experts=1，始终参与
out = routed_out + shared_out
```

- **共享专家融合**：`num_fused_shared_experts` 由 loader 决定；V4 只在显式开启且 checkpoint 恰有 1 个共享专家时才融合（`shared_experts_fusion_disable_reason`，`3418`）。
- **并行**：MoE 支持 EP（专家并行）、DP attention、a2a backend（DeepEP/mori/MegaMOE），`_run_moe_ffn_dp_sync`（`2559`）负责 DP 下的 gather/scatter 同步。

---

## 7. 完整层清单（自顶向下汇总表）

| 层级 | 模块 | 主要算子 | 张量形状变化 |
|------|------|----------|--------------|
| 0 | `embed_tokens` | VocabParallelEmbedding | `input_ids → (T, hidden)` |
| 0 | 流扩展 | `unsqueeze+repeat` | `(T, hidden) → (T, 4, hidden)` |
| 1~43 | `DecoderLayer` × N | 见下 | 维持 `(T, 4, hidden)` |
| — · | `hc_pre(attn)` | RMSNorm+GEMM+sinkhorn+聚合 | `(T,4,hidden) → (T,hidden)` |
| — · | `input_layernorm` | RMSNorm | `(T,hidden)` |
| — · | `self_attn` (MQALayer) | wq_a/q_norm/wq_b + wkv/kv_norm + compressor + indexer + attn_mqa + wo_a/wo_b | `(T,hidden) → (T,hidden)` |
| — · | `hc_post(attn)` | 注回 4 流 | `(T,hidden) → (T,4,hidden)` |
| — · | `hc_pre(ffn)` + `post_attention_layernorm` | 同上 | `(T,4,hidden) → (T,hidden)` |
| — · | `mlp` (DeepseekV2MoE) | gate(sqrtsoftplus)+noaux_tc top-k + routed experts(256选6) + shared expert | `(T,hidden) → (T,hidden)` |
| — · | `hc_post(ffn)` | 注回 4 流 | `(T,hidden) → (T,4,hidden)` |
| 收尾 | `hc_head` | RMSNorm+GEMM+sigmoid 收缩 | `(T,4,hidden) → (T,hidden)` |
| 收尾 | `norm` | RMSNorm | `(T,hidden)` |
| 收尾 | `lm_head` + `logits_processor` | GEMM → vocab | `(T,hidden) → (T, vocab=129280)` |

---

## 8. 关键维度速查

| 符号 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | 4096 | 隐藏维 |
| `hc_mult` | 4 | 残差流条数（MHC） |
| `head_dim` | 512 | 注意力头维 = 448(nope)+64(rope) |
| `n_heads` | 64 | 注意力头数（Q） |
| `num_key_value_heads` | 1 | KV 头数（MQA） |
| `q_lora_rank` | 1024 | Q 低秩维 |
| `o_lora_rank` / `o_groups` | 1024 / 8 | 输出低秩维 / 分组数 |
| `index_n_heads`/`index_head_dim`/`index_topk` | 64 / 128 / 512 | indexer 头数/头维/top-k |
| `compress_ratio` | 0 / 4 / 128 | 每层压缩类型（SWA / CSA / HCA） |
| `n_routed_experts`/`num_experts_per_tok`/`n_shared_experts` | 256 / 6 / 1 | MoE 专家数/激活数/共享数 |
| `num_hidden_layers` | 43 | 层数 |

---

## 9. 阅读代码的建议入口

- 想看**整体流**：`DeepseekV4Model.forward`（`3212`）→ `DeepseekV4DecoderLayer.forward`（`2374`）。
- 想看**注意力算子**：`MQALayer.forward`（`1690`）→ `_forward_prepare`（`1418`）→ `_compute_q_b`/`_compute_kv_to_cache`（`1067`/`1081`）。
- 想看**MHC 数学**：`hc_pre`（`2156`）、`hc_post`（`2316`）、`hc_head`（`3059`）+ `kernels/ops/layernorm/mhc.py` 的 torch 参考实现。
- 想看**DSA 稀疏**：`compressor.py` 的 `forward_compress`、`indexer.py` 的 `forward_c4_indexer` / `_topk_transform_vectorized`。
- 想看**MoE**：`deepseek_v2.py` 的 `DeepseekV2MoE`（`545`）、`MoEGate`（`455`）。

> 交叉参考：本目录已有 `deepseek_v4_model_architecture.md`（更偏架构综述）、`deepseek_v4_cache_management.md`（KV/压缩池管理）、`deepseek_v4_deployment_guide.md`（部署）。本文聚焦"层与算子"的逐级拆解。
