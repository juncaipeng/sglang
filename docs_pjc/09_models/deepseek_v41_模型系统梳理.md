# DeepSeek-V4.1 模型系统梳理（结构 / 计算 / Cache 管理）

> 本文系统梳理 **DeepSeek-V4.1**（`model_type == "deepseek_v41"`）在 SGLang 中的实现，覆盖：
> 结构、计算、权重/输入/输出 shape、以及 KV Cache 管理。
>
> **与已有文档的关系**：
> - 纯语言主干（MQA 双低秩注意力、MoE、RMSNorm、lm_head 等）已在
>   [deepseek_v4_模型结构.md](deepseek_v4_模型结构.md) / [deepseek_v4_layers_and_operators.md](deepseek_v4_layers_and_operators.md) 梳理。
> - V4 的六子池 KV Cache 基础见 [deepseek_v4_cache_management.md](deepseek_v4_cache_management.md)、
>   [deepseek_v4_flash_pool_sizing.md](deepseek_v4_flash_pool_sizing.md)。
> - **本文只聚焦 V4.1 相对 V4 的“增量”**：视觉多模态、Engram n-gram 记忆、mHC 多流超连接、
>   压缩比 1/2 的稀疏注意力、以及这些机制对应的 Cache 共享与管理。
>
> **主要代码位置**：
> - 模型主体：[python/sglang/srt/models/deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py)（V4/V4.1 共用，约 5900 行）
> - 视觉塔：[python/sglang/srt/models/deepseek_v41_vit.py](../../../python/sglang/srt/models/deepseek_v41_vit.py)
> - 配置：[python/sglang/srt/configs/deepseek_v41.py](../../../python/sglang/srt/configs/deepseek_v41.py)、[configs/deepseek_v4.py](../../../python/sglang/srt/configs/deepseek_v4.py)
> - Engram：[python/sglang/srt/layers/engram.py](../../../python/sglang/srt/layers/engram.py)
> - 稀疏注意力 dsv4 目录：[python/sglang/srt/layers/attention/dsv4/](../../../python/sglang/srt/layers/attention/dsv4/)
> - KV 池：[python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)
> - mHC 融合核：[python/sglang/kernels/ops/layernorm/mhc.py](../../../python/sglang/kernels/ops/layernorm/mhc.py)
> - 图像预处理：[python/sglang/srt/multimodal/deepseek_v41_image_processing.py](../../../python/sglang/srt/multimodal/deepseek_v41_image_processing.py)

---

## 0. 一句话概览

DeepSeek-V4.1 = **DeepSeek-V4 语言主干** ＋ **四项增量能力**：

| 增量 | 作用 | 关键代码 |
|---|---|---|
| **视觉多模态（ViT + Aligner）** | 支持图像输入，把图像编码成 LLM token 嵌入 | `deepseek_v41_vit.py`、`get_image_feature` |
| **Engram n-gram 哈希记忆** | 用最近若干 token 的 n-gram 哈希查一个可学习记忆表，门控注入残差流 | `layers/engram.py` |
| **mHC 多流超连接** | 残差流从 1 条扩成 `hc_mult=4` 条，子层前后用 Sinkhorn 双随机矩阵混合 | `hc_pre`/`hc_post`/`hc_head`、`mhc.py` |
| **压缩比 1/2 稀疏注意力** | V4 只有 {0,4,128}，V4.1 增加 {1,2}，并用 `kv_source_layer_ids` 让同比多层共享压缩存储 | `dsv4/dsv41_sparse.py`、`candidate_indexer.py` |

> ⚠️ **重要维度提示**：`DeepSeekV4Config` 数据类默认 `hidden_size=4096`，但**实际发布的 DSV4.1-Flash 权重是 `hidden_size=5120`**——代码里的融合核断言写死了这个值
> （如 [deepseek_v4.py:2953](../../../python/sglang/srt/models/deepseek_v4.py#L2953) `x.shape[1]==5120`、
> [:3486](../../../python/sglang/srt/models/deepseek_v4.py#L3486) `norm.weight.shape==(5120,)`、
> [:2729](../../../python/sglang/srt/models/deepseek_v4.py#L2729) `hc_attn_fn.shape==(24, 20480)`，20480 = 4×5120）。
> 本文正文用符号 `D = hidden_size`，出现具体数字时以实际发布配置 **D=5120** 为准，并在括号里标注。

---

## 1. 配置体系（DeepseekV41Config）

代码：[configs/deepseek_v41.py](../../../python/sglang/srt/configs/deepseek_v41.py)

V4.1 的 HF 配置是**嵌套的**（`text_config` + `vision_config`），需要拍平成运行时的“扁平 schema”。核心逻辑：

1. **`normalize_deepseek_v41_config`**（[deepseek_v41.py:23](../../../python/sglang/srt/configs/deepseek_v41.py#L23)）
   - 把 `text_config` 的字段提到顶层；
   - 把 `vision_config` 的字段按 `_VISION_FIELDS` 映射改名（`hidden_size→vision_dim`、`num_hidden_layers→vision_n_layers` 等，见下表）；
   - 把 `architectures == ["DeepseekV41ForCausalLM"]` 归一成 `["DeepseekV4ForCausalLM"]`——**因此 V4.1 复用 V4 的模型类**。
2. **`DeepseekV41Config(DeepseekV3Config)`**（[:43](../../../python/sglang/srt/configs/deepseek_v41.py#L43)）
   - 继承 V3 而非 V4 数据类，原因：V4 原生数据类**拒绝** `compress_ratio ∈ {1,2}`，V3 接受，从而放开 V4.1 的低压缩比层。
   - 默认打开 `hc_pre_from_prev_sublayer = True`、关闭 `q_head_norm = False`（与 V4 默认相反，见 §5）。

### 1.1 视觉字段映射（`_VISION_FIELDS`）

| HF `vision_config` | 运行时字段 | 默认值 |
|---|---|---|
| `hidden_size` | `vision_dim` | 1024 |
| `num_hidden_layers` | `vision_n_layers` | 0（=无视觉；>0 才建 ViT） |
| `num_attention_heads` | `vision_n_heads` | 16 |
| `intermediate_size` | `vision_inter_dim` | 2816 |
| `patch_size` | `vision_patch_size` | 14 |
| `rope_theta` | `vision_rope_theta` | 10000 |
| `downsample_ratio` | `vision_downsample_ratio` | 3 |
| `max_image_tokens` | `vision_max_n_token` | 1024 |
| `min_pixels` | `vision_min_pixels` | 295936 |

### 1.2 V4.1 新增/关键配置项（[configs/deepseek_v4.py](../../../python/sglang/srt/configs/deepseek_v4.py) 中定义 schema）

| 配置项 | 默认 | 含义 |
|---|---|---|
| `compress_ratios: List[int]` | [] | **逐层压缩比**，取值 ∈ {0,1,2,4,128}，V4.1 新增 1/2 |
| `kv_source_layer_ids: List[int]` | [] | 拥有压缩 KV 存储的“源层”（比 1/2 层共享它） |
| `index_source_layer_ids: List[int]` | [] | 拥有 indexer 索引键的源层 |
| `candidate_source_layer_id` | -1 | 两级块选择的“候选源层”（-1=关闭） |
| `candidate_topk_blocks` / `candidate_block_size` | 0 / 0 | 候选块数 / 块大小 |
| `index_head_dim / index_n_heads / index_topk` | 128 / 64 / 512 | indexer 头维 / 头数 / 选中 KV 数 |
| `compress_rope_theta` | 40000 | 压缩层 YaRN RoPE 基底 |
| `window_size` | 128 | SWA 滑窗大小 |
| `engram_layer_ids` | [] | 挂 Engram 的层 |
| `engram_num_embeddings: List[int]` | [] | 各 Engram 层哈希表大小 |
| `engram_max_ngram_size` | 1 | 最大 n-gram 长度 |
| `engram_vocab_size / engram_compressed_vocab_size` | 0 / 0 | 原始/压缩词表大小 |
| `engram_n_heads / engram_head_dim` | 0 / 0 | Engram 头数 / 头维 |
| `engram_pad_token_id` | 2 | Engram pad token |
| `n_hash_layers` | 3 | Engram 哈希层数 |
| `hc_mult` | 4 | mHC 残差流份数 |
| `hc_pre_from_prev_sublayer` | False（V4.1 覆盖为 True） | 是否把 pre-mix 从上一子层跨界融合 |
| `q_head_norm` | True（V4.1 覆盖为 False） | Q 是否做 per-head RMSNorm |
| `hc_sinkhorn_iters / hc_mult / hc_eps` | 20 / 4 / 1e-6 | mHC Sinkhorn 迭代 / 份数 / eps |
| `image_token_id` | 129264 | 图像占位 token id |

---

## 2. 视觉多模态：ViT + Aligner + 融合

代码：[deepseek_v41_vit.py](../../../python/sglang/srt/models/deepseek_v41_vit.py)（整塔仅 ~152 行）、
融合在 [deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py) 顶层模型里。

**仅当 `model_type=="deepseek_v41"` 且 `vision_n_layers>0`** 才构造视觉分支
（[deepseek_v4.py:4867](../../../python/sglang/srt/models/deepseek_v4.py#L4867)）。

### 2.1 结构树

```
DeepseekV4ForCausalLM
├── vision : ViT                                     # deepseek_v41_vit.py:112
│   ├── patch_embed : PatchEmbed                      # :38
│   │   └── proj : nn.Linear(3*14*14=588 → 1024)      # 权重[1024,588] + bias[1024]
│   ├── blocks : ModuleList(vision_n_layers × Block)  # :93 前置归一化 Transformer
│   │   └── Block
│   │       ├── norm1 : RMSNorm(1024, fp32 权重)       # force_native
│   │       ├── attn  : Attention(VisionAttention)    # 全双向注意力 + 2D RoPE
│   │       │   ├── qkv_proj : QKVParallelLinear       # [3*1024,1024]+bias, MHA(16头×64)
│   │       │   └── proj     : RowParallelLinear       # [1024,1024]+bias
│   │       ├── norm2 : RMSNorm(1024)
│   │       └── mlp   : MLP(SwiGLU)                    # :82
│   │           ├── w1 : nn.Linear(1024 → 2*2816)     # gate+up 融合, 无 bias
│   │           └── w2 : nn.Linear(2816 → 1024)       # down, 无 bias
│   └── norm : RMSNorm(1024)                          # 末端归一化
├── aligner : Aligner                                 # :138 投影到 LLM 维度
│   ├── w1 : nn.Linear(1024*9=9216 → D)  + bias       # 3×3 pixel-shuffle 后投影
│   └── w2 : nn.Linear(D → D)            + bias
├── image_start   : Parameter[D]                      # deepseek_v4.py:4880 图像起始标记向量
├── image_end     : Parameter[D]                      #                     图像结束标记
├── image_newline : Parameter[D]                      #                     每行换行标记
├── model   : DeepseekV4Model                         # 语言主干
└── lm_head : ParallelLMHead
```

符号：`Dv = vision_dim = 1024`，`Hv = vision_n_heads = 16`，`hdv = 64`，`Iv = vision_inter_dim = 2816`，
`P = patch_size = 14`，`r = downsample_ratio = 3`，`D = hidden_size(=5120)`。

### 2.2 各子模块 shape / dtype

| 子模块 | 类型 | 权重 shape | 输入 → 输出 | 说明 |
|---|---|---|---|---|
| `patch_embed.proj` | Linear | [1024, 588] +bias | `[N,3,14,14]`→flatten`[N,588]`→`[N,1024]` | N=图像 patch 数 |
| `qkv_proj` | QKVParallelLinear | [3072,1024]+bias | `[N,1024]`→q,k,v 各`[N,1024]`→`[N,16,64]` | MHA（KV 头=16，非 GQA） |
| `proj` | RowParallelLinear | [1024,1024]+bias | `[1,N,1024]`→`[1,N,1024]` | 注意力输出投影 |
| `mlp.w1` | Linear | [5632,1024] | `[N,1024]`→`[N,5632]`→chunk2→各`[N,2816]` | gate/up |
| `mlp.w2` | Linear | [1024,2816] | `silu(gate)*up`→`[N,1024]` | down |
| `norm1/2/norm` | RMSNorm | [1024] **fp32** | 同维 | force_native（融合核拒 fp32 权+bf16 激活） |
| `aligner.w1` | Linear | [D,9216]+bias | `[M,9216]`→`[M,D]` | M=下采样后 token 数 |
| `aligner.w2` | Linear | [D,D]+bias | `gelu`→`[M,D]` | 投影到 LLM 隐藏维 |
| `image_start/end/newline` | Parameter | [D] | — | 布局标记向量 |

> dtype：像素/patch 张量 **bf16**；RMSNorm 权重 **fp32**；RoPE 的 cos/sin 用 **fp32** 计算，
> `apply_rotary` 内部把 q/k 上采到 fp32 旋转后再回 bf16。

### 2.3 ViT 前向计算流（`ViT.forward(patches, n_h, n_w)`，[:123](../../../python/sglang/srt/models/deepseek_v41_vit.py#L123)）

```
patches [N=n_h*n_w, 3, 14, 14] (bf16)
  │ patch_embed:  flatten→[N,588]→Linear→ x[N,1024]
  │ 2D RoPE 表:   get_vision_cos_sin(n_h,n_w,rope_dim=32,theta) → cos/sin 各 [N,1,32]
  │              rope_dim = Dv/Hv/2 = 1024/16/2 = 32；用 hpos/wpos 两个方向拼接
  │ metadata:     cu_seqlens=[0,N] —— 整张图作为“单序列”，全 patch 互相可见（双向、无因果 mask）
  │ ×L 层 Block（前置归一化残差）:
  │      x = x + attn(norm1(x), cos, sin, metadata)   # [N,1024]
  │      x = x + mlp(norm2(x))                         # [N,1024]
  └ norm(x) → [N,1024]
```

**注意力后端**由 `VisionAttention._determine_attention_backend` 选择：SM90→`fa3`，SM100→`fa4`，
否则 `triton_attn`；NPU→`ascend_attn`；fallback `sdpa`。`use_data_parallel=True` 强制 `tp_size=1`，
即 ViT 注意力**复制式数据并行**而非按头切分。每张图独立处理，**无跨图注意力、无滑窗、无 sink**。

### 2.4 Aligner：pixel-shuffle 下采样 + 投影（`Aligner.forward`，[:146](../../../python/sglang/srt/models/deepseek_v41_vit.py#L146)）

把 ViT 的 `[N,1024]` 特征做 **3×3 无重叠窗口聚合（pixel-shuffle）**，再两层 GELU MLP 投影到 LLM 隐藏维：

```
x[N,1024] → view(n_h,n_w,1024).permute→[1024,n_h,n_w]
          → F.pad 到 r=3 的整数倍
          → F.unfold(kernel=3,stride=3) 取 3×3 窗口 → [1, 1024*9=9216, M]   (M=ceil(n_h/3)*ceil(n_w/3))
          → transpose → [M, 9216]
          → w1: [M,9216]→[M,D] → GELU → w2: [M,D]→[M,D]
```

即 **9 个相邻 patch 聚成 1 个 LLM token**，token 数从 `N` 降到 `M ≈ N/9`。

### 2.5 组装图像 token 序列 & 融合进 LLM（`get_image_feature`，[deepseek_v4.py:4989](../../../python/sglang/srt/models/deepseek_v4.py#L4989)）

对每张图，先 `features = aligner(vision(patches,h,w), h, w)`（`[M,D]`），再按 `image_token_types` 布局
拼一个 span 张量 `[num_tokens, D]`（[image_processing.py:124](../../../python/sglang/srt/multimodal/deepseek_v41_image_processing.py#L124)）：

```
布局 = [IMAGE_START] + ( IMAGE×n_llm_w + IMAGE_NEW_LINE ) × n_llm_h + [IMAGE_END]
type==0 → image_start 向量
type==1 → features 的对应行（真正的视觉特征，按阅读顺序）
type==2 → image_newline 向量
type==3 → image_end 向量
总长 = n_llm_h*(n_llm_w+1) + 2
```

**融合进 input embeds**（V4 **不用** `general_mm_embed_routine`，而是直接 `embed_mm_inputs`）：

1. `_prepare_mm_embeddings`（[:5017](../../../python/sglang/srt/models/deepseek_v4.py#L5017)）调 `embed_mm_inputs`：
   先对 `input_ids` 取文本 embedding，再用每个 mm item 的 `pad_value` 定位占位符位置，
   通过 `_scatter_mm_embedding`（cumsum 派生行号 + `index_copy_`）把视觉 span **精确写入占位符行**。
2. `forward` 里再把 `input_ids >= MM_PAD_SHIFT_VALUE` 的位置 `masked_fill` 回 `image_token_id=129264`
   （[:5105](../../../python/sglang/srt/models/deepseek_v4.py#L5105)），供 Engram 哈希/路由识别图像 token。

### 2.6 占位符插入 & 位置编码

- **`pad_input_ids`**（[:4984](../../../python/sglang/srt/models/deepseek_v4.py#L4984)）走
  `MultiModalityDataPaddingPatternMultimodalTokens.pad_input_tokens`：把每个图像 item 的
  `[start,end]` 区间就地填成 `item.pad_value`（内容 hash，供 radix 前缀匹配），后续在 embed 阶段被视觉特征覆盖。
- **positions / mrope**：DeepSeek-V4.1 **不使用 mrope**（全文件无 `mrope`/`position_ids` 引用）。
  图像 token 与文本 token 共用同一套 1D `positions`，直接透传给主干 RoPE。图像的空间结构信息
  通过 token 序列层面的 `image_newline`/`image_start`/`image_end` 标记表达，而非位置 id。
  ViT 内部有自己的 `vision_rope_theta`（2D RoPE），与 LLM 主干 positions 无关。

---

## 3. Engram：n-gram 哈希记忆

代码：[python/sglang/srt/layers/engram.py](../../../python/sglang/srt/layers/engram.py)
（文档字符串：*"gated n-gram hash memory added to the hc residual stream"*）

### 3.1 概念

对每个 token，回看它前面最多 `max_ngram_size` 个 token，把每个 n-gram（长度 2..n）**哈希**到逐层哈希表，
取出 fp8 嵌入，投影成 key/value，用 **sigmoid 门控** 把 value 加进 mHC 残差流 `x[T, hc_mult, D]`。
本质是一个**以最近 token n-gram 为键的可学习联想记忆**，只在部分层（`engram_layer_ids`）生效。

### 3.2 模块结构与权重

| 类 | 位置 | 作用 |
|---|---|---|
| `EngramLayout` | [engram.py:149](../../../python/sglang/srt/layers/engram.py#L149) | 冻结的布局元数据（primes、offsets、层列表） |
| `EngramHasher` | [:210](../../../python/sglang/srt/layers/engram.py#L210) | 产出 `[T, n_engram_layers, n_hash_cols]` int64 行号 |
| `EngramEmbedding` | [:667](../../../python/sglang/srt/layers/engram.py#L667) | 单层 fp8 哈希表（TP 行切分，可选 host 内存） |
| `Engram` | [:878](../../../python/sglang/srt/layers/engram.py#L878) | 注入 decoder 层的模块 |
| `engram_gate` | [:842](../../../python/sglang/srt/layers/engram.py#L842) | 融合/torch 门控 |

**`Engram` 权重**（[:878](../../../python/sglang/srt/layers/engram.py#L878)）：
- `embed : EngramEmbedding`：fp8 表，`weight[rows, head_dim]`（`float8_e4m3fn`）+ `scale[rows, head_dim//FP8_BLOCK_SIZE]`（`float8_e8m0fnu`），行数按 TP 切分。
- `wkv : ReplicatedLinear(n_hash_cols*head_dim → D*(hc_mult+1))`，`n_hash_cols=(max_ngram_size-1)*n_heads`。输出打包 `hc_mult` 个 key + 1 个共享 value。
- `q_weight / k_weight : Parameter[hc_mult, D]`（初始化全 1）。

### 3.3 哈希算法（`compute_engram_hash_ids`，[:187](../../../python/sglang/srt/layers/engram.py#L187)）

```
compressed = where(blocked, pad_id, token_map[tokens])   # [T,n]  token→压缩词表 id
products   = compressed.unsqueeze(1) * multipliers        # 按层广播；multipliers 为每(层,回看)一个奇数乘子
rolling = products[...,0]
for i in 1..n-1:
    rolling = xor(rolling, products[...,i])               # (i+1)-gram 的滚动 XOR 哈希
    hashes.append(rolling.unsqueeze(-1) % primes[:, i-1]) # 用该 n-gram 长度的 per-head 质数取模分桶
return cat(hashes,-1) + offsets                           # [T, n_layers, (n-1)*n_heads]，offsets 使各桶落到不相交区间
```

- `token_map`：token id → 压缩词表 id（HF normalizer 生成），要求 `vocab_size == compressed_vocab_size`。
- `blocked[T,n]`：标记回看越过序列起点的位置（填 pad）；图像 token 会强制更早的前驱为 PAD。
- `primes` 按 (层, n-gram 长度, 头) 升序取共享质数序列，起点略高于 `engram_vocab_size-1`。

哈希前向 `EngramHasher.forward`（[:275](../../../python/sglang/srt/layers/engram.py#L275)）分三种模式：
decode→`MODE_DECODE`（读写每请求 `history` 环）；target-verify→`MODE_VERIFY`（block=draft_token_num）；
extend→`MODE_EXTEND`。history commit 写回最后 `n-1` 个 token（翻转）。

### 3.4 Engram 前向 & 门控（`Engram.forward`，[:906](../../../python/sglang/srt/layers/engram.py#L906)）

```
x[T, hc_mult, D]; hash_ids[T, n_hash_cols]（本层）
emb = embed(hash_ids)                 # [T, n_hash_cols, head_dim] bf16
if x.shape[0]==0: return x            # 空批保护（MXFP8 拒绝空 M）
kv,_ = wkv(emb.flatten(-2))           # [T, D*(hc_mult+1)]
return engram_gate(x, kv, q_weight, k_weight, eps, clamp_value)
```

**`engram_gate`**（[:842](../../../python/sglang/srt/layers/engram.py#L842)）：把 `kv` 拆成 `hc_mult` 个 key `[T,hc_mult,D]` + 共享 value `[T,D]`；
逐 (token, hc 拷贝) 对 x 拷贝与 key 各做 RMSNorm（沿 D），算
`dot = (h * q_weight*k_weight * key).sum(-1) * rstd * D**-0.5`，
门控 `g = sigmoid(copysign(sqrt(|dot| clamped), dot))`，把 `g * value` 加到**每一份** hc 拷贝上。

### 3.5 注入 decoder 层

- 构造：仅当 `layer_id in engram_layout.layer_ids` 时建 `self.engram`（[deepseek_v4.py:2681](../../../python/sglang/srt/models/deepseek_v4.py#L2681)）。
- hash id 每次前向在模型体里算一次（[:4399](../../../python/sglang/srt/models/deepseek_v4.py#L4399)），层循环里逐层 `engram(hidden_states, hash_ids[:, layer_hash_index], ...)`（[:4453](../../../python/sglang/srt/models/deepseek_v4.py#L4453)）。
- **VL 特例**：图像 token 行会被恢复成 `before_engram`（[:4465](../../../python/sglang/srt/models/deepseek_v4.py#L4465)），即图像 token 不吃 Engram 记忆。

---

## 4. mHC 多流超连接（multi-head-copy Hyper-Connections）

代码：`hc_pre`/`hc_post`/`hc_head` 在 [deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py)，
Sinkhorn / 混合核在 [mhc.py](../../../python/sglang/kernels/ops/layernorm/mhc.py)。

> 传统残差：`x + block(x)`（单流）。mHC 把残差扩成 **`hc_mult=4` 条并行流**，子层前用可学习权重
> 把 4 流“预混合并塌缩成 1 流”喂给 attn/MoE，子层后再用一个 **Sinkhorn 双随机矩阵** 把结果散射回 4 流。

### 4.1 残差流形状：`[T, hc_mult, D]`

层间携带的 residual stream 是 **`[T, 4, D]`**（4 份 hidden 拷贝），非 `[T, D]`。证据：
- 入口把 embedding `[T,D]` 用 `unsqueeze(1).repeat(1,hc_mult,1)` 扩成 `[T,4,D]`（[:4692](../../../python/sglang/srt/models/deepseek_v4.py#L4692)）。
- `hc_post` 断言 `residual.shape == (T, hc_mult, D)`（[:2993](../../../python/sglang/srt/models/deepseek_v4.py#L2993)）。
- PP 传输时 3D 展平成 `[T, hc_mult*D]`，收端 `view` 还原。

### 4.2 参数

**`make_hc_mixing_params(hc_mult, D)`**（[:553](../../../python/sglang/srt/models/deepseek_v4.py#L553)）——每个 decoder 层
attn/ffn 子层各一套，核心：
- `hc_*_fn : [mix_hc, hc_dim]`，`mix_hc=(2+hc_mult)*hc_mult=24`，`hc_dim=hc_mult*D`（=4×5120=20480）——混合投影矩阵。
- `hc_*_base : [24]`，`hc_*_scale : [3]`（pre/post/comb 各一个标量温度）。

**`make_hc_head_params(hc_mult, D)`**（[:570](../../../python/sglang/srt/models/deepseek_v4.py#L570)）——lm_head 前塌缩用：
`hc_head_fn:[hc_mult, hc_dim]`、`hc_head_base:[hc_mult]`、`hc_head_scale:[1]`（仅 `hc_pre_from_prev_sublayer=False` 时用）。

### 4.3 Sinkhorn 迭代（`hc_sinkhorn_iters=20`）算什么

作用在 **`comb` 混合矩阵 `[T, hc_mult, hc_mult]=[T,4,4]`** 上，交替做行/列归一，使其收敛为**双随机矩阵**
（行、列和≈1），即“4 份拷贝之间的软置换/混合”。torch 参考 `_hc_split_sinkhorn_torch`
（[mhc.py:195](../../../python/sglang/kernels/ops/layernorm/mhc.py#L195)）：

```
# 一行 24 维 = [pre(4) | post(4) | comb(16)]
pre  = sigmoid(flat[:, :4] *scale[0] + base[:4]) + eps      # [T,4]  子层前预混合权重
post = 2*sigmoid(flat[:,4:8]*scale[1] + base[4:8])          # [T,4]  子层后门控（范围(0,2)）
comb = (flat[:,8:]*scale[2] + base[8:]).reshape(-1,4,4)     # [T,4,4]
comb = row_softmax(comb); comb = comb/comb.sum(1)+eps       # 初始行 softmax + 列归一
for _ in range(sinkhorn_iters-1):                           # 再迭代 19 次
    comb = comb/comb.sum(2)   # 行归一
    comb = comb/comb.sum(1)   # 列归一
```

**为什么**：`comb` 决定上一子层的 4 份残差如何线性重组成新的 4 份。双随机矩阵保证混合是“保质量”的凸组合，
防止残差能量放大/塌缩，是 mHC 稳定的关键。`hc_eps=1e-6` 为除法保护。

### 4.4 每层前向：hc_post → hc_pre → norm → 子层

层 forward 主体在 [:3006](../../../python/sglang/srt/models/deepseek_v4.py#L3006)，attn 与 ffn 两条子层同构。

**`hc_pre`**（[:2761](../../../python/sglang/srt/models/deepseek_v4.py#L2761)）：输入 `[T,4,D]`，输出：
- `y = hc_combine(x_flat, pre)` = **`[T,D]`**（用 `pre[T,4]` 把 4 份加权求和成 1 份），喂给 attn/MoE；
- `post[T,4]`、`comb[T,4,4]` 留给对应 `hc_post`；
- RMSNorm 对 `x.flatten(1)`（`hc_dim` 维）整体做；TileLang/flashinfer 路径可把 `input_layernorm` 权重**融进 kernel**（返回 `norm_fused=True`），否则显式做。

**`hc_pre_from_prev_sublayer` 跨界融合**（V4.1 默认 True）：上一子层不立即 hc_post，而是把
`(residual, post, comb)` 作为 `prev_*` 传给下一子层，用 `apply_mhc_post_pre_boundary` 把
“上一子层 hc_post + 本子层 hc_pre + layernorm”融合成一个 kernel 边界（专用路径
`_forward_layers_hc_pre_from_prev`，[:4388](../../../python/sglang/srt/models/deepseek_v4.py#L4388)）。
**该模式要求 `pp_group.world_size==1` 且与 TBO 互斥**。

**`hc_post`**（[:2921](../../../python/sglang/srt/models/deepseek_v4.py#L2921)）：把子层输出 `[T,D]` 用 `post/comb` 散射回 `[T,4,D]`：

```
out = post.unsqueeze(-1) * x.unsqueeze(1)                     # 新结果广播到 4 份、乘 post 门控
    + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(1)     # comb 双随机矩阵混合旧 4 份残差
```

### 4.5 `q_head_norm`

与 mHC 无关，是**注意力内部**：是否在 RoPE 前对每个 query head 单独做 RMSNorm
（`_compute_q_b`，[:1302](../../../python/sglang/srt/models/deepseek_v4.py#L1302)）。V4.1 默认 `False`（跳过 head norm，
只对 q 尾 64 维施 RoPE）；V4 默认 `True`（`fused_q_norm_rope` 每 (token,head) 先 RMSNorm 再 RoPE）。

### 4.6 塌缩回 `[T,D]` 喂 lm_head（[:4808](../../../python/sglang/srt/models/deepseek_v4.py#L4808)）

```
pre_hc_head = hidden_states.flatten(1)     # [T, hc_dim]（也传给 logits_processor 供 spec/MTP）
hidden_states = hc_combine(...) 或 hc_head(...)   # → [T,D]  用学习投影+sigmoid 门控加权求和 4 份
hidden_states = self.norm(hidden_states)   # 最终 RMSNorm
```

---

## 5. 稀疏注意力增强：压缩比 1/2 + Indexer + Candidate

V4 只有 `compress_ratios ∈ {0,4,128}`；V4.1 增加 **{1,2}**，并引入
“**源层拥有存储、后续同比层共享**”与 **两级块选择（Candidate）**。

dsv4 目录（[layers/attention/dsv4/](../../../python/sglang/srt/layers/attention/dsv4/)）文件角色：

| 文件 | 角色 |
|---|---|
| `compressor.py` | V4 核心 `Compressor`（ratio 4/128） |
| `indexer.py` | V4 `C4Indexer`（ratio-4 top-k） |
| **`dsv41_sparse.py`** | **V4.1 `DeepseekV41Compressor` + `DeepseekV41Indexer`（ratio 1/2）** |
| **`candidate_indexer.py`** | **两级块候选选择** |
| **`dense_prefill_indexer.py`** | **稠密 prefill 的 ragged top-k** |
| `metadata.py` | `PagedIndexerMetadata` 调度/分块 |
| `compressor_v2.py` / `compressor_trtllm.py` / `compress_hip.py` | 硬件变体 |

### 5.1 每层挂什么（`MQALayer.__init__`，[deepseek_v4.py:1115](../../../python/sglang/srt/models/deepseek_v4.py#L1115)）

| `compress_ratio` | compressor | indexer | 说明 |
|---|---|---|---|
| 0 | — | — | 纯 SWA 稠密层 |
| 4 | `Compressor`（每层各一） | `C4Indexer`（每层各一） | 压缩 4× + 稀疏 top-512 |
| 128 | `Compressor`（每层各一） | — | 压缩 128×（HCA） |
| 1/2 | `DeepseekV41Compressor` **仅 `kv_source_layer_ids`** | `DeepseekV41Indexer` **仅 `index_source_layer_ids`** | 非源层 compressor/indexer 为 None，**读共享存储** |

`compress_ratio` = 每个压缩槽聚合的原始 token 数。取值断言 ∈ {0,1,2,4,128}（[:793](../../../python/sglang/srt/models/deepseek_v4.py#L793)）。

### 5.2 `DeepseekV41Compressor`（ratio 1/2，[dsv41_sparse.py:98](../../../python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py#L98)）

bf16 权重，fp32 投影 + fp32 softmax 池化，`finish` 里回 bf16。
- ratio 1：单 `wkv` 投影，**无 gate**（`project`→`(kv, None)`），一 token 一 latent（不减 token 数，但投影/量化）。
- ratio 2：`wkv_gate:[D, 2*head_dim]`（sm100+ 融合）或 `wkv`+`wgate` 两个 GEMM，输出 `|kv|score|`。
  `pool_pairs`：`(kv2 * score2.softmax(dim=1)).sum(dim=1)`，对 `[n,2,D]` 的一对 token 做 **softmax 加权池化** → `[n,D]`。
- `finish(kv) = norm(kv.to(bf16))`。

**YaRN**：压缩层（ratio≠0）Q/SWA/压缩 KV 统一用 `compress_rope_theta=40000` 的 deepseek_yarn RoPE
（[:917](../../../python/sglang/srt/models/deepseek_v4.py#L917)）。

### 5.3 `DeepseekV41Indexer`（ratio 1/2 索引源层，[dsv41_sparse.py:170](../../../python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py#L170)）

用一个小型 fp4 side-attention 给压缩位置打分，选 `index_topk=512` 个位置喂主注意力。
`n_heads=64` 全 TP 复制（**每 rank 用全部头打分，top-k 无需跨 rank 归约**）。
- `wq_b:[q_lora_rank=1024 → 64*128=8192]`；`weights_proj:[D → 64]`（per-head 权重）。
- 仅源层（`owns_k`）建 `wk:[head_dim→128]` + `k_norm`。
- 索引键 `index_keys = _rope_fq4(k_norm(wk(latent)))` → `[n,128]`（fp4 假量化）。
- 打分 `scores`：`s = einsum("bhd,nd->bhn", q, k)`；`s = (relu(s)*weights).sum(dim=1)` → `[t,n]`（64 头求和）。
  下游 top-k 保留 512 个。

### 5.4 Candidate 两级块选择（[candidate_indexer.py](../../../python/sglang/srt/layers/attention/dsv4/candidate_indexer.py)）

配置默认关闭（`candidate_source_layer_id=-1`）。开启后：
- 一个指定的“**候选源层**”先做**块级**粗粒度 top-k（在压缩序列上），**发布**选中的块 id；
- 之后每个索引层（`layer_id > candidate_source_layer_id`）**消费**这些块，把自己的打分限制到候选块内——
  “**先选块、再选位置**”的两级稀疏，省去每个索引层都扫全上下文。

标志位在 `DeepseekV41Indexer.__init__`（[dsv41_sparse.py:189](../../../python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py#L189)）：
`is_candidate_source` / `uses_candidates` / `candidate_topk_blocks` / `candidate_block_size`。
块 top-k `_candidate_block_topk`（[candidate_indexer.py:107](../../../python/sglang/srt/layers/attention/dsv4/candidate_indexer.py#L107)）：
按 `block_size` 分块 `amax`，**强制保留当前（末）块**（+inf），再 `topk(topk_blocks)`。
`make_candidate_indexer` 在 sm<100（Hopper）返回 None，改走内联 mask。

### 5.5 稠密 prefill top-k（[dense_prefill_indexer.py](../../../python/sglang/srt/layers/attention/dsv4/dense_prefill_indexer.py)）

`dense_prefill_topk`：对 `fp8_fp4_mqa_logits` 做 ragged-batch top-k，按 2 GiB 分数预算分块，
可选发布/消费 `PrefillCandidateBlocks`。

---

## 6. Cache 管理（V4.1 增量）

代码：[mem_cache/deepseek_v4_memory_pool.py](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)。
顶层池 `DeepSeekV4TokenToKVPool`（[:857](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L857)）继承 `BaseSWAKVPool`。
逐层路由靠 `DeepSeekV4LayerItem`（[:708](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L708)）。

> V4 基础六子池见 [deepseek_v4_cache_management.md](deepseek_v4_cache_management.md)（注意其行号已随代码增长而偏移）。
> 本节只讲 V4.1 增量：**ratio 1/2 的共享压缩存储、fp4 索引池、ratio-2 待配对状态**。

### 6.1 每种 cache 存哪、多大

| cache 种类 | 适用 ratio | 池对象 | 关键点 |
|---|---|---|---|
| SWA 近窗 KV | 全层 | `swa_kv_pool` | FP8 nope(448)+BF16 rope(128)+scale(7)+pad(1)=**584 B/token**（uint8）；窗口 `window_size=128` |
| c4 / c128 压缩 KV | 4 / 128 | `kv_pools[4/128]` | 同 584 B 布局，页 `page_size//ratio`，**每层独占** |
| **c1 / c2 压缩 latent** | **1 / 2** | `kv_pools[1/2]` | `size=full_size//ratio`，`layer_num=len(源层)`——**仅源层有物理层** |
| c4 索引键 | 4 | `index_pools[4]` | 132 B(int8) 或 68 B(fp4) |
| **c1 / c2 索引键** | **1 / 2** | `index_pools[1/2]` | **`force_fp4=True` → 恒 68 B**，页 `DSV41_INDEX_PAGE_SIZE`(128 或 64) |
| c4/c128 attn compress-state 环 | 4/128 | `compress_state_pools` | 环形 |
| **ratio-2 待配对状态** | **2** | `compress_state_pools`（仅源层） | **fp32** 存 `(kv,score)`，等偶数 token 的奇数伙伴到达 |

**索引键字节**（`get_dsv4_indexer_bytes_per_token`，[:39](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L39)）：
fp4=`128/2 + 128/32 = 68 B`；int8=`128 + 4 = 132 B`。

### 6.2 核心机制：`kv_source_layer_ids` 共享压缩存储

这是 V4.1 最关键的 Cache 设计。`_collect_sources_by_ratio`（[:1626](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1626)）：

- **ratio 4/128**：*每一层*都拥有自己的压缩槽。
- **ratio 1/2**：**只有 `config.kv_source_layer_ids` 里的层**拥有压缩 latent；同比后续层**共享同一存储**，不再各自分配。

层→槽映射（`_init_compressed_layer_mapping`，[:1656](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1656)）：
```
if ratio in (1,2):
    compress_layer_id = sources.index(source_layer_of(idx))   # 共享 index，多个绝对层别名到同一槽
else:
    compress_layer_id = layer_counts[ratio]; layer_counts[ratio]+=1   # 每层唯一
```
`source_layer_of(layer)`（[:1648](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1648)）：ratio 4/128 返回自身；ratio 1/2 返回**最近的前驱源层**。

ratio 1/2 池按此收缩：`kv_pools[ratio]` 只有 `len(sources_by_ratio[ratio])` 个物理层
（[:1432](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1432)）。模型侧对应：非源层的 `compressor/indexer` 为 None，通过后端读共享 latent。

### 6.3 压缩 token→槽 映射（后端）

`_low_ratio_compression_metadata`（[deepseek_v4_backend.py:379](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L379)）：
```
completes_group = seq_lens % ratio == 0
out_loc = where(completes_group, raw_out_loc // ratio, -1)   # 完成一组才落盘，否则 -1（挂起）
topk_lengths_clamp1 = (seq_lens // ratio).clamp_min(1)
```
即压缩槽 = `raw_out_loc // ratio`；只有位置补满一组才写入。

### 6.4 top-k 如何喂主注意力

top-512 位置进入 `DSV4AttnMetadata` 的 per-ratio 稀疏缓冲（[deepseek_v4_backend.py:419](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L419)）：
`sparse_page_indices(ratio)`（c1/c2/c4 用 indexer top-k；c128 用全部块）、`sparse_topk_lengths`、`sparse_raw_indices`。
`mask_topk_scores` 把 `-inf`/越界位置写 -1，挡在注意力外。FlashMLA 按 ratio 各持一份调度元数据
（`c0_/c1_/c2_/c4_/c128_flashmla_metadata`）。

### 6.5 prefill vs decode

- **ratio 1/2 源层**：走 `forward_low_ratio_sources`（extend 下可能走可打断 CUDA graph 的 `bcg_*`）；
  Blackwell 有专用 decode/verify 多流路径 `low_ratio_multi_stream`。
- **ratio 4/128**：`forward_indexer_compressor` + `forward_core_compressor`。
- decode 合并表 `init_trtllm_sparse_buffers`：先 128 列 SWA、再压缩 KV，行按 64-token tile 补齐。

### 6.6 Flash 池大小（V4.1 第 8 项）

`pool_configurator.py`（[:1063](../../../python/sglang/srt/model_executor/pool_configurator.py#L1063)）在 V4 七项系数外**新增低比项**，
且**只对 kv_source 层计费**（共享存储）：
```
low_ratio_index_bytes = get_dsv4_indexer_bytes_per_token(index_head_dim, use_fp4=True)  # 68
low_ratio_bytes_per_full_token = Σ_{l ∈ kv_source_layer_ids, ratio(l)∈{1,2}} (584 + 68) / ratio(l)
```
详见 [deepseek_v4_flash_pool_sizing.md](deepseek_v4_flash_pool_sizing.md)。

### 6.7 SWA（滑窗=128）vs 压缩层

- **每层**都有 SWA 近窗（`swa_kv_pool`，全精度 584 B/token），环形寻址 `req_idx*window + pos%window`。
- **ratio 0/1 层不建 attn compress-state**（ratio 0 纯 SWA；ratio 1 一 token 一 latent 无需待配对）。
- 压缩层（2/4/128）= SWA（稠密看最近 128）＋ 压缩远程历史（c4/c1/c2 indexer top-512；c128 dense 全块）。

---

## 7. 前向总流程（含 V4.1 增量）

```
input_ids [T] (含图像占位 token / hash id)
   │
   ├─(有视觉且非 decode/verify) _prepare_mm_embeddings：
   │      ViT(patches)→Aligner→features→拼 span→scatter 到占位符行 → input_embeds[T,D]
   │      再把 input_ids>=MM_PAD_SHIFT 还原成 image_token_id=129264
   │
   v embed / 传入 input_embeds                    → [T, D]
   │ unsqueeze+repeat                             → [T, hc_mult=4, D]   （mHC 初始化）
   │
   │ 每层前一次性算 Engram hash_ids               → [T, n_layers, n_hash_cols]
   │
   v 43 × DeepseekV4DecoderLayer（mHC 残差流 [T,4,D]）：
   │      attn 子层:  hc_post→hc_pre→norm→ MQALayer(+压缩/indexer/candidate) → hc_post
   │      (若该层挂 Engram)  hidden += engram(...)     图像 token 行旁路
   │      ffn 子层:   hc_pre→norm→ DeepseekV2MoE → hc_post
   │      （hc_pre_from_prev_sublayer=True 时相邻子层的 post+pre+norm 融合成一个边界）
   │
   v flatten→hc_head/hc_combine 塌缩             → [T, D]
   v model.norm (RMSNorm)                        → [T, D]
   v lm_head                                     → logits[T, vocab=129280]
   v logits_processor                            → 采样 / logprobs（pre_hc_head 供 spec/MTP）
```

## 8. 并行支持与限制

`DeepseekV4ForCausalLM.__init__`（[:4867](../../../python/sglang/srt/models/deepseek_v4.py#L4867)）构造视觉时校验：

```python
if attn_cp_size != 1 or pp_group.world_size != 1 or not moe_a2a_backend.is_none():
    raise ValueError("V4.1 vision currently supports TP/EP/DP without CP, PP or MoE A2A")
```

- **支持**：TP / EP / DP。
- **拒绝**：CP（上下文并行，会切分含图像占位的序列、破坏 span 顺序写入）、PP（视觉融合在 first PP rank 的 embedding 阶段）、MoE A2A（token 重排与视觉占位 hash/路由不兼容）。
- 另：`hc_pre_from_prev_sublayer` 前向 assert `pp_group.world_size==1`（跨 PP 的 pre-mix 交接未接线），且与 TBO 互斥。

## 9. 要点小结

1. **复用 V4 模型类**：V4.1 配置归一化后 `architectures→DeepseekV4ForCausalLM`，只是打开视觉、Engram、ratio 1/2、mHC 跨界融合等开关。
2. **视觉**：ViT（全双向 + 2D RoPE，`use_data_parallel` 复制）→ Aligner（3×3 pixel-shuffle 9 patch 聚 1 token → 投影到 D）→ 按 `image_token_types` 拼 span → scatter 进 input embeds。无 mrope。
3. **Engram**：最近 token 的 2..n-gram 滚动 XOR 哈希 → fp8 记忆表 → sigmoid 门控加进 mHC 4 份残差流；仅部分层，图像 token 旁路。
4. **mHC**：残差流 `[T,4,D]`；子层前 `pre` 加权塌缩成 `[T,D]`，子层后 `post`+Sinkhorn 双随机 `comb` 散射回 4 份；lm_head 前再塌缩。`q_head_norm` V4.1 默认关。
5. **稀疏注意力**：新增 ratio 1/2；**`kv_source_layer_ids` 让同比多层共享压缩 latent 与 fp4 索引存储**；Candidate 两级“先选块再选位置”。
6. **Cache**：SWA 全层 584 B/token；压缩池按 `//ratio` 收缩且低比只对源层计费；ratio-2 需 fp32 待配对状态；索引键低比恒 fp4(68 B)。

> **维度再提醒**：正文 `D` 在实际 DSV4.1-Flash 权重下为 **5120**（配置数据类默认 4096，被实际发布配置覆盖，融合核断言写死 5120 / hc_dim=20480）。

