# DeepSeek-V4 模型结构（不含 V4.1 视觉分支）

> 代码位置：[python/sglang/srt/models/deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py)
> 配置位置：[python/sglang/srt/configs/deepseek_v4.py](../../../python/sglang/srt/configs/deepseek_v4.py)
>
> 本文只梳理纯语言模型（`model_type == "deepseek_v4"`），不涉及 V4.1 的 ViT / Aligner / Engram 视觉相关分支。

---

## 一、核心配置（DeepSeekV4Config 默认值）

| 配置项 | 值 | 含义 |
|---|---|---|
| `hidden_size` | 4096 | 隐藏维度 |
| `vocab_size` | 129280 | 词表大小 |
| `num_hidden_layers` | 43 | Decoder 层数 |
| `num_attention_heads` | 64 | 注意力头数 |
| `num_key_value_heads` | 1 | KV 头数（MQA，单 KV 头） |
| `head_dim` | 512 | 单头维度 = `qk_nope_head_dim` + `qk_rope_head_dim` |
| `qk_nope_head_dim` | 448 | 不加 RoPE 的 QK 维度 |
| `qk_rope_head_dim` | 64 | 加 RoPE 的 QK 维度 |
| `v_head_dim` | 512 | V 维度 |
| `q_lora_rank` | 1024 | Q 低秩压缩维度 |
| `o_lora_rank` | 1024 | 输出投影低秩维度 |
| `o_groups` | 8 | 输出投影分组数 |
| `n_routed_experts` | 256 | 路由专家数 |
| `num_experts_per_tok` | 6 | 每 token 激活专家数 |
| `n_shared_experts` | 1 | 共享专家数 |
| `moe_intermediate_size` | 2048 | 单个专家中间维度 |
| `n_group` / `topk_group` | 8 / 8 | 专家分组（V4 实际走非分组路由） |
| `scoring_func` | `sqrtsoftplus` | 路由打分函数 |
| `routed_scaling_factor` | 1.5 | 路由缩放因子 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `first_k_dense_replace` | 0 | 前 k 层用 Dense MLP（V4 为 0，全部 MoE） |
| `compress_ratios` | List | 逐层压缩比（取值 ∈ {0,1,2,4,128}） |

---

## 二、整体结构树

```
DeepseekV4ForCausalLM                                          # 顶层：CausalLM 封装
│
├── model : DeepseekV4Model                                    # 主干
│   │
│   ├── embed_tokens : VocabParallelEmbedding                  # [129280, 4096] 词嵌入（首个 PP rank）
│   │
│   ├── layers : ModuleList(43 × DeepseekV4DecoderLayer)       # 43 层 Decoder
│   │   │
│   │   └── DeepseekV4DecoderLayer                             # ── 单层内部（下面展开）──
│   │       │
│   │       ├── input_layernorm : RMSNorm(4096)               # attn 前归一化（Pre-Norm）
│   │       │
│   │       ├── self_attn : MQALayer                          # MQA 注意力 + Q/O 双低秩
│   │       │   ├── wq_a  : ReplicatedLinear     4096 → 1024      # Q 降维 (q_lora_rank)
│   │       │   │                                                #  ※ 融合时为 wqkv_a: 4096 → 1024+512
│   │       │   ├── q_norm: RMSNorm(1024)                        # Q 低秩后归一化
│   │       │   ├── wq_b  : ColumnParallelLinear 1024 → 32768    # Q 升维 = 64 heads × 512
│   │       │   ├── wkv   : ReplicatedLinear     4096 → 512      # 单 KV 头投影 (MQA, head_dim)
│   │       │   ├── kv_norm: RMSNorm(512)                        # KV 归一化
│   │       │   ├── rotary_emb : RotaryEmbedding                 # RoPE(rope_head_dim=64)；压缩层用 YaRN
│   │       │   ├── attn_sink   : Parameter[64]                  # 每头 attention sink (fp32)
│   │       │   ├── attn_mqa    : RadixAttention                 # 注意力核(num_kv_heads=1)+KV cache
│   │       │   ├── wo_a  : ColumnParallelLinear 4096 → 8192     # 输出分组降维 (32768/8 → 8×1024)
│   │       │   ├── wo_b  : RowParallelLinear    8192 → 4096     # 输出升维回 hidden
│   │       │   ├── compressor : Compressor / DeepseekV41Compressor  # 仅压缩层(见注)
│   │       │   └── indexer    : C4Indexer / DeepseekV41Indexer      # 仅稀疏层(见注)
│   │       │
│   │       ├── post_attention_layernorm : RMSNorm(4096)      # mlp 前归一化（Pre-Norm）
│   │       │
│   │       └── mlp : DeepseekV2MoE                           # MoE 前馈
│   │           ├── gate : MoEGate                              # 路由门控
│   │           │   ├── weight                : [256, 4096]     # 256 专家打分权重
│   │           │   └── e_score_correction_bias: [256]          # 路由偏置(noaux_tc)
│   │           ├── topk : TopK                                 # top-6, 非分组 sqrtsoftplus
│   │           ├── experts : FusedMoE(256 experts)             # 路由专家(每 token 激活 6 个)
│   │           │   └── Expert_e (SwiGLU):                      #  单专家: down(silu(gate)*up)
│   │           │        gate_up_proj 4096 → 2×2048             #   inter=moe_intermediate_size
│   │           │        down_proj    2048 → 4096
│   │           └── shared_experts : DeepseekV2MLP             # 共享专家×1(始终参与, 未融合时存在)
│   │                ├── gate_up_proj : MergedColumnParallelLinear 4096 → 2×2048
│   │                ├── act_fn       : SiluAndMul
│   │                └── down_proj    : RowParallelLinear        2048 → 4096
│   │
│   └── norm : RMSNorm(4096)                                    # 主干末端归一化（末个 PP rank）
│
├── lm_head : ParallelLMHead                                   # [4096, 129280] 输出投影
│                                                               #   tie_word_embeddings=False → 独立权重
└── logits_processor : LogitsProcessor                         # 计算 logits / 采样前处理
```

> **压缩 / 稀疏层说明**（`compressor` / `indexer` 只在部分层出现，由 `compress_ratios[layer_id]` 决定）：
> - `compress_ratio == 4`  → `compressor=Compressor` + `indexer=C4Indexer`（压缩 KV + 稀疏选择）
> - `compress_ratio == 128`→ `compressor=Compressor`（仅压缩，无 indexer）
> - `compress_ratio ∈ {1,2}` 且属于 `kv_source_layer_ids` → `compressor=DeepseekV41Compressor`
> - `compress_ratio ∈ {1,2}` 且属于 `index_source_layer_ids` → `indexer=DeepseekV41Indexer`
> - `compress_ratio == 0` → 普通稠密 MQA，无 compressor / indexer

残差数据流（单层）：

```
h_in ──► input_layernorm ──► self_attn ──►(+)──► post_attention_layernorm ──► mlp ──►(+)──► h_out
  └──────────────────── 残差 ─────────────┘  └────────────────── 残差 ─────────────┘
```

> 注：V4 使用 MHC（multi-head-cross）混合/融合归一化路径，`forward` 中 `input_layernorm` / `post_attention_layernorm` 会与相邻子层的 pre-mix 融合（`hc_pre_from_prev_sublayer`、fused post+pre）。逻辑上仍等价于上面的 Pre-Norm 残差结构。

---

## 三、self_attn：MQALayer（MQA 多查询注意力 + 低秩 Q/O）

继承自 `MqaAttentionBase`。核心特点：
- **MQA**：`num_key_value_heads == 1`，64 个查询头共享同一个 KV 头（KV cache 只存 1 份）。
- **Q 低秩分解**：`wq_a`(降维 4096→1024) → `q_norm` → `wq_b`(升维 1024→32768)。
- **O 分组低秩**：输出投影拆成 `wo_a`（分组降维）→ `wo_b`（升维），共 `o_groups=8` 组。
- **attention sink**：每头一个可学习标量，作为额外"零 token"参与 softmax，稳定长序列注意力。
- **压缩层**：`compress_ratio ∈ {4,128}` 的层额外挂 `Compressor` 与 `Indexer`（稀疏/压缩 KV）。

### 3.1 维度速查

| 记号 | 值 | 来源 |
|---|---|---|
| `hidden_size (D)` | 4096 | `config.hidden_size` |
| `n_heads (H)` | 64 | `config.num_attention_heads` |
| `head_dim (Dh)` | 512 | `qk_nope_head_dim(448) + qk_rope_head_dim(64)` |
| `qk_rope_head_dim` | 64 | 每头加 RoPE 的维度 |
| `qk_nope_head_dim` | 448 | 每头不加 RoPE 的维度 |
| `q_lora_rank` | 1024 | Q 低秩维度 |
| `o_lora_rank` | 1024 | O 低秩维度 |
| `o_groups (G)` | 8 | 输出投影分组数 |
| `softmax_scale` | `Dh^-0.5` (约 0.0442) | 注意力缩放 |

### 3.2 子模块：输入 / 输出 / 权重 shape 与 dtype（T = 单 rank token 数）

| 子模块 | 类型 | 权重 shape | 权重 dtype | 输入 → 输出 (shape) | 激活 I/O dtype | 说明 |
|---|---|---|---|---|---|---|
| `wq_a` | ReplicatedLinear | [1024, 4096] | bf16 / fp8-block | [T,4096] to [T,1024] | bf16 → bf16（fp8 路径：输入动态量化 fp8-block 做 GEMM，输出回 bf16） | Q 降维。融合模式 `wqkv_a`:[1536,4096] |
| `q_norm` | RMSNorm | [1024] | bf16(内部fp32) | [T,1024] to [T,1024] | bf16 → bf16（归约 fp32） | Q 低秩归一化 |
| `wq_b` | ColumnParallelLinear | [32768, 1024] | bf16 / fp8-block | [T,1024] to [T,64,512] | bf16 → bf16（fp8 同上） | Q 升维=`H*Dh`，按 attn_tp 切列 |
| `wkv` | ReplicatedLinear | [512, 4096] | bf16 / fp8-block | [T,4096] to [T,512] | bf16 → bf16（fp8 同上） | 单 KV 头投影（MQA） |
| `kv_norm` | RMSNorm | [512] | bf16(内部fp32) | [T,512] to [T,512] | bf16 → bf16（归约 fp32） | KV 归一化 |
| `rotary_emb` | RotaryEmbedding | freqs 表(buffer) | fp32/bf16 | 旋转 q/k 的 64 维 rope 段 | bf16 → bf16（cos/sin fp32，旋转时上采 fp32 再回 bf16） | 压缩层用 YaRN |
| `attn_sink` | Parameter | [64] | **fp32** | 标量/头 | fp32（进 softmax 的额外 logit 列） | 进 softmax 的 sink logit |
| `attn_mqa` | RadixAttention | 无(管 KV cache) | KV: bf16/fp8 | (q,k,v) to [T,64,512] | q/k/v bf16 → o bf16；**KV cache 存 bf16 或 fp8**（`SGLANG_DSV4_UNIFIED_KV_FP8`） | 注意力核 + cache 读写 |
| `wo_a` | ColumnParallelLinear | [8192, 4096] | bf16 / fp8-block | [T,4096] to [T,8192] | bf16 → bf16（fp8/absorb GEMM 内部低精度） | 分组降维：in=`H*Dh/G`=4096，out=`G*o_lora`=8192 |
| `wo_b` | RowParallelLinear | [4096, 8192] | bf16 / fp8-block | [T,8192] to [T,4096] | bf16 → bf16（fp8 同上） | 升维回 D，attn_tp reduce |
| `compressor` | Compressor / DeepseekV41Compressor | 视实现 | bf16/fp8 | 生成压缩 KV | bf16 输入 → 压缩 KV **存 fp8**（投影/softmax 池化在 fp32，`finish` 回 bf16 后量化落盘） | 仅压缩层 |
| `indexer` | C4Indexer / DeepseekV41Indexer | 视实现 | bf16/fp8 | 生成 top-k KV 索引 | bf16 输入 → 索引键 **int8 或 fp4**（打分排序用，V4.1 低比恒 fp4） | 仅稀疏层 |

> **dtype 约定**：
> - **权重**：线性层默认 **bf16**；开 FP8 时按 `weight_block_size` 决定量化粒度（见下）。DeepSeek V4/V4.1 用的是 **block-wise FP8**（`weight_block_size=[128,128]`，带 `weight_scale_inv` + UE8M0 scale）。
> - **FP8 不等于一定 block-wise**：`Fp8LinearMethod`（[fp8.py:510](../../../python/sglang/srt/layers/quantization/fp8.py#L510)）按 `weight_block_size` 分派多种粒度——`[128,128]`→block-wise（权重按 128×128 块、激活按 token-group(128) 动态量化）；`None`→per-tensor 或 per-channel权重+per-token激活；`[1,32]`→MXFP8（e8m0 块 scale）。文中"block-wise FP8"是**针对本模型**的结论，不能推广到所有 fp8 线性层。
> - **激活（输入/输出）**：模块边界统一 **bf16**。**量化特殊说明**——FP8 路径并非"反量化权重后按 bf16 算"，而是把**输入激活也量化成 fp8**（block 路径下为 per-token-group(128) 动态量化）、权重 fp8，用 DeepGEMM 做 **fp8 GEMM**，输出再回 bf16；FP4（专家）同理，输入量化为对应低精度格式。
> - **激活量化时机（分平台）**：
>   - **NVIDIA CUDA（常见部署）**：norm（或被 mHC `hc_pre` 折进 pre-norm GEMM 的 norm）输出 **bf16**，每个 fp8 linear 在自己的 `w8a8_block_fp8_linear` 里**内部动态量化**（`input_scale=None`，per-token-group(128)，[fp8.py:1221](../../../python/sglang/srt/layers/quantization/fp8.py#L1221)）。**norm 不会提前把 linear 的输入变成 fp8。**
>   - **AMD ROCm gfx95 / gfx1250**：走 `_fused_rmsnorm_fp8_quant`（[deepseek_v4.py:531](../../../python/sglang/srt/models/deepseek_v4.py#L531)，aiter `fused_rms_fp8_group_quant`，group=128），把 **RMSNorm + fp8 group-quant 融合成一趟**，linear 直接吃 fp8 `(q_fp8, scale)` 元组，省一个量化 kernel。两处生效：input_layernorm→`wq_a`/`wkv`/`wqkv_a`（守卫 `_use_aiter and _is_gfx95_supported`，[:3055](../../../python/sglang/srt/models/deepseek_v4.py#L3055)）、q_norm→`wq_b`（守卫 gfx95/gfx1250，[:1682](../../../python/sglang/srt/models/deepseek_v4.py#L1682)）。
>   - 另注：mHC `hc_pre` 的 `norm_fused=True` 把 RMSNorm 折进 mix GEMM，但产出的是 **bf16** 混合流，不是 fp8——那是"norm 折进 mix"，与"norm 折进 fp8 量化"是两回事。
> - **fp32 关键路径**：RMSNorm 均方根归约、`attn_sink`、compressor 的投影/池化、logits 计算。
> - **KV cache**：默认 bf16，开 `SGLANG_DSV4_UNIFIED_KV_FP8` 后 nope 段 fp8 + rope 段 bf16（见加速文档 §3.1）。

### 3.3 计算逻辑与公式

输入 `x` 形状 [T, 4096]（bf16），`positions` 形状 [T]。

**(1) Query 低秩链路**
```
q_lora = wq_a(x)          # [T,1024]
q_lora = q_norm(q_lora)   # RMSNorm
q      = wq_b(q_lora)     # [T,32768] -> view [T,64,512]
```
RMSNorm（对最后一维，eps=1e-6）：
$$\text{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}\odot w$$

**(2) 单 KV 头链路（MQA）**
```
kv = kv_norm(wkv(x))      # [T,512] 单头，广播给 64 个 Q 头；K/V 共享该压缩表示
```

**(3) RoPE（只作用于每头后 64 维）**
`Dh=512` 切成 `[nope:448 | rope:64]`，仅对 rope 段施加旋转，nope 段(前 448 维)不变：
$$q_{rope},k_{rope}=\text{RoPE}(q_{[448:512]},k_{[448:512]},\text{positions})$$
压缩层(C4/C128)对 rope 段用压缩 YaRN 频率表。

**(4) 注意力（含 attention sink）**，KV 共享单头，对第 h 头：
$$\text{logits}_h=\frac{q_h k^\top}{\sqrt{Dh}},\quad \alpha_h=\text{softmax}([\text{logits}_h,\ \text{sink}_h])$$
sink 作为额外 logit 列（对应 value=0）；softmax 后取非 sink 部分与 v 相乘。由 `attn_mqa (RadixAttention)` 执行并读写 KV cache（因果 mask、radix 前缀复用）。输出 o 形状 [T,64,512]。

**(5) 分组低秩输出投影**
```
o   = o.reshape(T, 4096)  # 8 组 x 512 视角
a   = wo_a(o)             # [T,8192] 分组降维 -> 8 组 x 1024
out = wo_b(a)             # [T,4096] 升维回 hidden（attn_tp>1 时 all-reduce）
```

**(6) 压缩/稀疏层附加**（仅 `compress_ratio` 属于 {1,2,4,128}）
- `Compressor`：把 KV 压缩为更短序列（压缩比 4/128），减少注意力长度与 cache。
- `Indexer`(C4)：每 query 选 top-`index_topk`(=512) 个 KV block 做稀疏注意力（`index_head_dim=128`, `index_n_heads=64`）。

---

## 四、mlp：DeepseekV2MoE（MoE 前馈）

复用 `deepseek_v2.DeepseekV2MoE`，以 `is_deepseek_v4=True` 构造。43 层全部为 MoE（`first_k_dense_replace=0`，无 dense 层）。

### 4.1 子模块：输入 / 输出 / 权重 shape 与 dtype

| 子模块 | 类型 | 权重 shape | 权重 dtype | 输入 → 输出 (shape) | 激活 I/O dtype | 说明 |
|---|---|---|---|---|---|---|
| `gate.weight` | Parameter | [256, 4096] | bf16（可选 fp32） | [T,4096] to [T,256] | bf16 输入 → 分数 **fp32**（打分 GEMM 用 fp32 累加更稳） | 路由打分 GEMM |
| `gate.e_score_correction_bias` | Parameter | [256] | **fp32** | 加到分数 | fp32 | noaux_tc 均衡偏置 |
| `topk` | TopK | 无权重 | — | [T,256] to (idx[T,6], w[T,6]) | fp32 分数 → idx **int32/int64** + 权重 fp32 | 非分组 sqrtsoftplus，top-6 |
| `experts` | FusedMoE | 见下(256 专家) | bf16 / fp8 / fp4 | [T,4096]+路由 to [T,4096] | bf16 → bf16（**fp8/fp4 路径：输入动态量化为对应低精度做 grouped GEMM，输出回 bf16**） | 路由专家(SwiGLU) |
| `shared_experts` | DeepseekV2MLP | 见下 | bf16 / fp8 | [T,4096] to [T,4096] | bf16 → bf16（fp8 同上） | 共享专家 x1 |

**单个专家 / 共享专家（SwiGLU）权重**（中间维 `moe_intermediate_size=2048`）：

| 层 | 类型 | 权重 shape | 权重 dtype | 激活 I/O dtype |
|---|---|---|---|---|
| `gate_up_proj` | MergedColumnParallelLinear | [2x2048, 4096] | bf16/fp8/fp4 | bf16 → bf16（fp8/fp4 时输入量化后 GEMM） |
| `act_fn` | SiluAndMul | 无权重 | — | bf16 → bf16 |
| `down_proj` | RowParallelLinear | [4096, 2048] | bf16/fp8/fp4 | bf16 → bf16（fp8/fp4 时输入量化后 GEMM） |

- 路由专家（256 个）是模型主要权重来源，量化(fp8/fp4)通常施加于此；共享专家中间维 `2048 x n_shared_experts(1)`。

### 4.2 计算逻辑与公式

输入 `x` 形状 [T, 4096]（bf16）。

**(1) 门控打分**
```
s = gate(x) = x @ Wg^T    # [T,256]  (fp32 累加更稳)
```

**(2) Top-k 路由（V4：非分组 sqrtsoftplus）**
打分函数 `sqrtsoftplus`：
$$p_e=\sqrt{\text{softplus}(s_e)}=\sqrt{\log(1+e^{s_e})}$$
加偏置后选 top-6，并（`norm_topk_prob=True`）归一化权重：
$$\text{idx}=\text{TopK}_6(p+b_{corr}),\qquad w_e=\frac{p_e}{\sum_{j\in\text{idx}}p_j}$$

**(3) 专家计算（SwiGLU）**，每个被选专家：
```
gu   = gate_up_proj(x)    # [T,2*2048]
g, u = gu.chunk(2, -1)    # 各 [T,2048]
h    = silu(g) * u        # SiluAndMul，silu(g)=g*sigmoid(g)
e    = down_proj(h)       # [T,4096]
```
$$\text{Expert}(x)=W_{down}\big(\text{SiLU}(W_{gate}x)\odot(W_{up}x)\big)$$

**(4) 合并输出**
```
y_routed = sum_{e in idx} w_e * Expert_e(x)   # [T,4096]
y_routed *= routed_scaling_factor (=1.5)
y_shared  = shared_experts(x)                 # [T,4096] 始终参与
out       = y_routed + y_shared               # [T,4096]
```
$$\text{MoE}(x)=\lambda\!\!\sum_{e\in\text{TopK}}\! w_e\,\text{Expert}_e(x)+\text{Shared}(x),\quad \lambda=1.5$$

> 共享专家可被"融合"进 MoE kernel（`num_fused_shared_experts>0`，当作第 257 个本地专家）或走独立 `DeepseekV2MLP`，取决于 EP / A2A backend；数学结果等价。

---

## 五、归一化 / 嵌入 / 输出组件（shape、dtype、公式）

| 组件 | 权重 shape | 权重 dtype | 输入 → 输出 (shape) | 激活 I/O dtype | 计算 |
|---|---|---|---|---|---|
| `embed_tokens` (VocabParallelEmbedding) | [129280, 4096] | bf16 | [T] to [T,4096] | **int32/int64（token id）→ bf16** | 查表；TP 切词表(DP-attn 关时) |
| `input_layernorm` / `post_attention_layernorm` | [4096] | bf16(内部fp32) | [T,4096] to [T,4096] | bf16 → bf16（归约 fp32） | RMSNorm，eps=1e-6 |
| `q_norm` / `kv_norm` | [1024] / [512] | bf16 | 同维 | bf16 → bf16（归约 fp32） | RMSNorm |
| `model.norm` | [4096] | bf16 | [T,4096] to [T,4096] | bf16 → bf16（归约 fp32） | 主干末端 RMSNorm |
| `lm_head` (ParallelLMHead) | [129280, 4096] | bf16 | [T,4096] to [T,129280] | bf16 → **fp32**（GEMM `out_dtype=fp32`，[logits_processor.py:953](../../../python/sglang/srt/layers/logits_processor.py#L953)） | `logits = h @ W^T`；tie=False 独立权重 |
| `LogitsProcessor` | 无 | — | 末层隐藏态 to logits | bf16 隐藏态 → **fp32 logits** | 取末位隐藏态过 lm_head，供采样 |

RMSNorm 统一公式（eps=1e-6）：
$$y=\frac{x}{\sqrt{\text{mean}(x^2)+\epsilon}}\odot w$$

> **MHC 融合归一化**：V4 通过 `make_hc_mixing_params` / `make_hc_head_params` 在相邻子层间做混合归一化（Sinkhorn 迭代 `hc_sinkhorn_iters=20`，`hc_mult=4`），并把 pre/post RMSNorm 与该 mix 融合进同一 kernel（`hc_pre_from_prev_sublayer`、fused post+pre）。逻辑上等价于标准 Pre-Norm 残差，仅为性能优化。

---

## 六、前向总流程（含 shape）

```
input_ids [T] (int)
   |
   v embed_tokens                        -> h[T,4096] bf16
   |
   v 43 x DeepseekV4DecoderLayer         每层 Pre-Norm 残差：
   |     h = h + self_attn(input_layernorm(h))         # 注意力子层
   |     h = h + mlp(post_attention_layernorm(h))      # MoE 子层
   |
   v model.norm (RMSNorm)                -> h[T,4096]
   |
   v lm_head (ParallelLMHead)            -> logits[T,129280]
   |
   v logits_processor                    -> 采样 / logprobs
```

单层残差（Pre-Norm）：
$$h' = h + \text{Attn}(\text{RMSNorm}_{in}(h)),\qquad h'' = h' + \text{MoE}(\text{RMSNorm}_{post}(h'))$$

---

## 七、要点小结

1. **注意力是 MQA + 双低秩**：Q 走 `wq_a→q_norm→wq_b` 低秩分解；输出走 `wo_a(分组)→wo_b` 低秩投影；KV 单头（`num_key_value_heads=1`）。带 per-head `attn_sink`。
2. **部分层带压缩/稀疏 KV**：由 `compress_ratios[layer_id]` 决定（4/128 挂 `Compressor`，4 再挂 `Indexer`）。
3. **MoE 全层生效**：`first_k_dense_replace=0`，43 层全部是 `DeepseekV2MoE`；256 路由专家选 6 + 1 共享专家，非分组 `sqrtsoftplus` 路由。
4. **归一化融合**：V4 的 MHC 路径把相邻子层的 pre/post RMSNorm 与 mix 融合，等价于标准 Pre-Norm 残差。
