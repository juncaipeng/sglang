# DeepSeek-V4.1 推理加速方法梳理

> 代码位置：[python/sglang/srt/models/deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py)（V4/V4.1 共用）
> 及 [layers/engram.py](../../../python/sglang/srt/layers/engram.py)、[layers/attention/dsv4/](../../../python/sglang/srt/layers/attention/dsv4/)、
> [kernels/ops/layernorm/mhc.py](../../../python/sglang/kernels/ops/layernorm/mhc.py)、[kernels/ops/embeddings/engram_gate.py](../../../python/sglang/kernels/ops/embeddings/engram_gate.py)。
>
> **与已有文档的关系**：
> - V4（纯语言主干）的加速手段（MQA + Q/O 双低秩 + DSA + FP8 KV + EP/DeepEP + CUDA Graph + TBO + MTP 等）见
>   [deepseek_v4_推理加速方法.md](deepseek_v4_推理加速方法.md)。**那些手段 V4.1 全部继承**。
> - V4.1 的结构/计算/Cache 全景见 [deepseek_v41_模型系统梳理.md](deepseek_v41_模型系统梳理.md)、mHC 专题见
>   [deepseek_v41_mhc_多流超连接.md](deepseek_v41_mhc_多流超连接.md)。
> - **本文只聚焦 V4.1 相对 V4 的“增量加速”**：即 V4.1 为了让「视觉 + Engram + mHC + ratio 1/2 稀疏」这四项
>   新能力**跑得快、省显存**而引入的专门优化。
>
> 组织方式沿用 V4：**算法结构层 → 算子/kernel 融合层 → 量化/存储层 → 并行/多流层 → 调度/图层**，
> 每项说明「是什么 / 为什么快 / 代码位置 / 生效条件」。
>
> ⚠️ 维度约定：正文用 `D = hidden_size`，实际 DSV4.1-Flash 权重 **D=5120**（`hc_dim = hc_mult·D = 4×5120 = 20480`）。

---

## 〇、一句话总览

V4.1 的四项新能力各自都可能拖慢推理，SGLang 用如下增量优化把它们的开销压回可用范围：

| 新能力 | 天然开销 | V4.1 的加速对策 |
|---|---|---|
| **视觉多模态** | 每图上千 patch → LLM 序列暴涨 | pixel-shuffle **9 patch→1 token**（序列缩 9×）、ViT 复制式 DP、单图独立无跨图注意力 |
| **Engram n-gram 记忆** | 大哈希表占显存、门控是逐 token 小算子 | fp8 表 + e8m0 块 scale、**可选 host offload**、fused engram gate、TP 行切分 |
| **mHC 多流超连接** | 残差流 ×4、每子层 Sinkhorn 20 迭代 | Sinkhorn 融合核、**跨子层 post+pre+norm 融合边界**、split-K 批不变、多后端 kernel |
| **ratio 1/2 稀疏注意力** | 更多压缩层 → 更多 KV/索引存储 | **`kv_source_layer_ids` 同比多层共享压缩 latent+fp4 索引**、Candidate 两级块选择、Blackwell 低比多流 |

| 分层 | V4.1 增量加速项 | 主要收益 |
|---|---|---|
| 结构 | pixel-shuffle 视觉下采样 9→1 | LLM 视觉 token 数缩 9× |
| 结构 | ViT 复制式数据并行 | 视觉塔无 TP 通信 |
| 结构 | ratio 1/2 稀疏 + Candidate 两级块选择 | 索引扫描量再降 |
| 结构 | Engram 记忆代替部分算力 | 用查表补偿容量、算力开销小 |
| kernel | mHC Sinkhorn / big_fuse 融合 | 减 norm/mix kernel 数 |
| kernel | mHC 跨子层 post+pre+norm 融合 | 相邻子层边界省一次读写 |
| kernel | fused engram gate | 门控合成单 kernel |
| kernel | fused wo_a (V4.1 专用) | 输出投影融合 |
| 量化/存储 | **共享压缩 latent 存储**（低比源层计费） | 低比 KV 显存按源层数而非层数 |
| 量化/存储 | fp4 索引键（低比恒 68 B） | 索引存储减半 |
| 量化/存储 | fp8 Engram 表 + host offload | 大记忆表不占/少占显存 |
| 量化/存储 | MXFP8 dense GEMM（wq_b / engram wkv） | Blackwell 投影更快 |
| 并行/多流 | low_ratio_multi_stream（Blackwell） | 低比 decode/verify 多流重叠 |
| 调度/图 | breakable CUDA graph（bcg_*） | 低比源层/Engram/attn 可打断重放 |

---

## 一、算法 / 结构层增量

### 1.1 视觉 pixel-shuffle 下采样：9 patch → 1 LLM token
- **是什么**：Aligner 把 ViT 输出的 `[N,1024]` patch 特征做 **3×3 无重叠窗口聚合**
  （`F.unfold(kernel=3,stride=3)`），9 个相邻 patch 拼成一个 `[9216]` 向量，再两层 GELU MLP 投影到 `D`，
  token 数从 `N` 降到 `M ≈ N/9`。
- **为什么快**：**图像 token 是要进 LLM 主干逐层算注意力/MoE 的**——LLM 侧算力和 KV 都随 token 数线性/超线性增长。
  下采样 9× 直接把每张图占用的 LLM 序列长度砍到 1/9，是视觉侧最重要的省算力手段。
- **代码**：`Aligner.forward`（[deepseek_v41_vit.py:146](../../../python/sglang/srt/models/deepseek_v41_vit.py#L146)），
  `downsample_ratio=3`。

### 1.2 ViT 复制式数据并行 + 单图独立注意力
- **是什么**：视觉塔 `use_data_parallel=True` 时强制 `tp_size=1`，即 ViT **整塔在每个 rank 上复制**，
  按图/请求切分数据而非按注意力头切分。每张图 `cu_seqlens=[0,N]` 作为单序列，全 patch 双向可见，
  **无跨图注意力、无滑窗、无 sink**。
- **为什么快**：ViT 相对 LLM 主干很小，复制它省掉了注意力头切分带来的 all-reduce/all-gather 通信；
  单图独立让不同图可并行、且每图注意力规模有界（不随 batch 里其它图增长）。
- **代码**：`VisionAttention`（`_determine_attention_backend`：SM90→fa3 / SM100→fa4 / 否则 triton_attn），
  ViT 构造分支（[deepseek_v4.py:4867](../../../python/sglang/srt/models/deepseek_v4.py#L4867)）。

### 1.3 ratio 1/2 稀疏注意力 + Candidate 两级块选择
- **是什么**：V4 只有 `compress_ratios ∈ {0,4,128}`；V4.1 增加 **{1,2}** 更细的压缩粒度，
  并引入 **Candidate 两级块选择**：一个“候选源层”先在压缩序列上做**块级**粗粒度 top-k 并发布选中块 id，
  后续索引层只在候选块内打分（“先选块、再选位置”）。
- **为什么快**：
  - ratio 1/2 让远程历史压缩得更温和（保信息），配合 indexer top-512 仍是近线性注意力；
  - **Candidate 省去每个索引层都扫全上下文**——块级预筛后，位置级打分只在少量候选块里做，
    索引开销随索引层数摊薄。
- **代码**：`DeepseekV41Compressor` / `DeepseekV41Indexer`（[dsv41_sparse.py:98](../../../python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py#L98) / [:170](../../../python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py#L170)）；
  `_candidate_block_topk`（[candidate_indexer.py:107](../../../python/sglang/srt/layers/attention/dsv4/candidate_indexer.py#L107)，按块 `amax` + 强制保留末块）；
  `dense_prefill_topk`（[dense_prefill_indexer.py](../../../python/sglang/srt/layers/attention/dsv4/dense_prefill_indexer.py)，ragged-batch top-k，按 2 GiB 分数预算分块）。
- **生效条件**：Candidate 默认关闭（`candidate_source_layer_id=-1`）；`make_candidate_indexer` 在 sm<100（Hopper）返回 None，改走内联 mask。

### 1.4 Engram：用查表容量补偿、算力开销极小
- **是什么**：Engram 是**以最近 token n-gram 为键的可学习联想记忆**：n-gram 滚动 XOR 哈希查一张 fp8 记忆表，
  sigmoid 门控加进 mHC 残差流。它是加容量的手段，但推理成本主要是**一次哈希 + 一次查表 + 一次小投影**。
- **为什么（相对）快**：查表 O(1)、不参与注意力的 O(N) 扫描；只在部分层（`engram_layer_ids`）生效，
  且**图像 token 行旁路**（恢复成 `before_engram`），不为视觉付 Engram 代价。
- **代码**：`Engram.forward`（[engram.py:906](../../../python/sglang/srt/layers/engram.py#L906)）；
  层内注入与图像旁路（[deepseek_v4.py:4453](../../../python/sglang/srt/models/deepseek_v4.py#L4453) / [:4465](../../../python/sglang/srt/models/deepseek_v4.py#L4465)）。

---

## 二、算子 / kernel 融合层增量

### 2.1 mHC Sinkhorn 融合核 + big_fuse
- **是什么**：mHC 每子层要在 `comb` 矩阵 `[T,4,4]` 上做 20 次 Sinkhorn 行/列归一，再算 `pre/post` 门控。
  SGLang 把「24 维 logits 投影 → sigmoid 门控 → Sinkhorn 20 迭代」融合成单 kernel。
  TileLang 路径 `mhc_pre_big_fuse` 进一步把 **pre-norm GEMM + 平方和 + Sinkhorn** 合成一个大融合核。
- **为什么快**：Sinkhorn 是逐 token 的小规模迭代，若拆成 20×2 个归一 kernel 会被 launch 开销和访存吞掉；
  融合成单 kernel 后全在片上寄存器/共享内存里迭代。
- **代码**：`hc_split_sinkhorn` / `hc_split_sinkhorn_kernel`（[mhc.py:129](../../../python/sglang/kernels/ops/layernorm/mhc.py#L129)）、
  `mhc_pre_big_fuse_tilelang`（[mhc.py:402](../../../python/sglang/kernels/ops/layernorm/mhc.py#L402)）；参考实现 `_hc_split_sinkhorn_torch`（[:195](../../../python/sglang/kernels/ops/layernorm/mhc.py#L195)）。

### 2.2 mHC 跨子层 post+pre+norm 融合边界（`hc_pre_from_prev_sublayer`）
- **是什么**：V4.1 默认 `hc_pre_from_prev_sublayer=True`。普通做法是上一子层做完 `hc_post`（散射回 4 流）、
  下一子层再做 `hc_pre`（塌缩成 1 流）+ layernorm。V4.1 把这三步**跨子层融合成一个 kernel 边界**
  （`apply_mhc_post_pre_boundary`），残差不落地一次读写。
- **为什么快**：`[T,4,D]` 残差流每次落地就是 4× 的显存往返；把相邻子层的出/入混合与归一化合并，
  省掉一整轮 `hc_dim=20480` 宽张量的读写。
- **代码**：专用前向路径 `_forward_layers_hc_pre_from_prev`（[deepseek_v4.py:4388](../../../python/sglang/srt/models/deepseek_v4.py#L4388)）；
  门控 `use_fused_mhc_post_pre` / `is_cross_layer_mhc_fusion_enabled`。
- **限制**：要求 `pp_group.world_size==1`，且与 TBO 互斥。

### 2.3 mHC 多后端 kernel + split-K 批不变
- **是什么**：mHC 的 pre/post 混合按硬件走不同 kernel：TileLang（sm90/Blackwell，`mhc_post_split_h` 需 M=4、D=5120）、
  DeepGEMM tf32（大 M prefill）、FlashInfer（`mhc_pre_big_fuse`）、Triton（gfx1250）、NPU/XPU/HIP(aiter)、torch fallback。
  pre-norm GEMM 用 **split-K + 平方和** 分段规约（`mhc_pre_gemm_sqrsum_splitk`）。
- **为什么快**：各平台取各自最优 GEMM；split-K 在小 M（decode）下提高 SM 占用；**split-K 的规约顺序固定**，
  与 Sinkhorn 融合一起保证 mHC 的**批不变性**（prefill/decode 数值一致，利于投机 verify 与 KL 一致性）。
- **代码**：`mhc_pre_gemm_sqrsum_splitk_kernel`（[mhc.py:609](../../../python/sglang/kernels/ops/layernorm/mhc.py#L609)）、
  `_compute_num_split_for_mhc_pre`（[mhc.py:746](../../../python/sglang/kernels/ops/layernorm/mhc.py#L746)）；`tf32_hc_prenorm_gemm`。

### 2.4 fused engram gate
- **是什么**：Engram 门控要对每 (token, hc 拷贝) 做 RMSNorm + 点积 + sigmoid + 加权注入，是逐 token 小算子链。
  SGLang 把整条链融合成 `fused_engram_gate` 单 kernel。
- **为什么快**：避免门控合成产生大量中间张量与 kernel launch；`torch` 参考路径仅作 fallback。
- **代码**：`engram_gate`（[engram.py:842](../../../python/sglang/srt/layers/engram.py#L842)）→ `fused_engram_gate`（[kernels/ops/embeddings/engram_gate.py](../../../python/sglang/kernels/ops/embeddings/engram_gate.py)）。
- **注意**：空批（`x.shape[0]==0`）直接返回——`wkv` 背后的 MXFP8 量化拒绝空 M（[engram.py:919](../../../python/sglang/srt/layers/engram.py#L919)）。

### 2.5 fused wo_a（V4.1 专用开关）
- **是什么**：V4 已有 wo_a absorb GEMM；V4.1 提供专用融合开关 `SGLANG_DSV41_FUSED_WO_A`，把输出投影的 `wo_a`
  批量 GEMM 用 V4.1 特定 shape/量化路径融合。
- **为什么快**：把多步小算子合并成一次大 batched GEMM，配合低精度权重减少访存。
- **代码**：见 V4 文档 §2.2；env `SGLANG_DSV41_FUSED_WO_A`。

---

## 三、量化 / 存储层增量

### 3.1 共享压缩 latent 存储（`kv_source_layer_ids`）★ V4.1 最关键的省显存设计
- **是什么**：ratio 4/128 时**每层**都拥有自己的压缩 KV 槽；ratio 1/2 时**只有 `config.kv_source_layer_ids`
  里的“源层”**拥有压缩 latent，同比后续层**共享同一存储**（别名到同一物理槽），非源层的 `compressor/indexer` 为 None、
  通过后端直接读共享 latent。
- **为什么省**：低压缩比（1/2）意味着压缩后序列几乎不缩短，若每层各存一份，低比层的 KV 显存会爆。
  共享存储让 **ratio 1/2 池只有 `len(sources_by_ratio[ratio])` 个物理层**，显存按源层数而非层数计费——
  这是 V4.1 敢用 ratio 1/2 的前提。
- **代码**：`_collect_sources_by_ratio`（[memory_pool.py:1626](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1626)）、
  `_init_compressed_layer_mapping`（[:1656](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1656)：ratio 1/2 走 `sources.index(...)` 别名，其余每层唯一）、
  `source_layer_of`（[:1648](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1648)：低比返回最近前驱源层）。
- **Flash 池计费**：`pool_configurator.py`（[:1063](../../../python/sglang/srt/model_executor/pool_configurator.py#L1063)）新增低比项且**只对 kv_source 层计费**：
  `Σ_{l∈kv_source, ratio∈{1,2}} (584 + 68) / ratio(l)`。

### 3.2 fp4 索引键（低比恒 68 B）
- **是什么**：indexer 的索引键有 int8（132 B）和 fp4（68 B）两种。ratio 1/2 的 indexer 强制 `force_fp4=True`，
  索引键**恒为 fp4 假量化 68 B**（`_rope_fq4`）。
- **为什么省**：索引池是额外开销，低比层数多，索引键减半（132→68 B/token）直接削掉一半索引存储与读带宽；
  fp4 只用于**打分排序**（选 top-512），量化误差对选择结果不敏感。
- **代码**：`get_dsv4_indexer_bytes_per_token`（[memory_pool.py:39](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L39)：fp4=128/2+128/32=68；int8=128+4=132）；
  `index_pools[1/2]` 布局；`DeepseekV41Indexer._rope_fq4`（[dsv41_sparse.py](../../../python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py)）。

### 3.3 fp8 Engram 记忆表 + 可选 host offload
- **是什么**：每层 Engram 哈希表是 fp8（`float8_e4m3fn`）权重 + e8m0 块 scale（`float8_e8m0fnu`，查表时 dequant），
  按 TP **行切分**（每 rank 只存部分行，all-reduce 重组）。可选 `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`
  把表放 **host（CPU）内存**，通过 memfd + `cudaHostRegister` pin 常驻。
- **为什么省**：Engram 表可能很大（`engram_num_embeddings` 逐层），fp8 相对 bf16 体积减半；
  TP 行切分让每卡只存 `1/tp` 行；host offload 把整表移出 GPU 显存（用 PCIe/NVLink 按需读），
  在显存吃紧时用带宽换容量。
- **代码**：`EngramEmbedding`（[engram.py:667](../../../python/sglang/srt/layers/engram.py#L667)）、`_HostTable`（[:550](../../../python/sglang/srt/layers/engram.py#L550)：mapping/memfd/pin 全程不释放）、
  `_init_host_table`（[:701](../../../python/sglang/srt/layers/engram.py#L701)）；env `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`。

### 3.4 MXFP8 dense GEMM（wq_b / engram wkv，Blackwell）
- **是什么**：Blackwell 上 `wq_b`、Engram 的 `wkv` 等 dense 投影走 MXFP8 dense GEMM
  （`mxfp8_dense_backend == FLASHINFER_CUTEDSL`）。
- **为什么快**：低精度 dense GEMM 吞吐更高、访存更省；且该后端与低比多流路径协同（见 §4.1）。
- **代码**：`wq_b.quant_method.mxfp8_dense_backend`（[deepseek_v4.py:2143](../../../python/sglang/srt/models/deepseek_v4.py#L2143)）。
- **注意**：其它 MXFP8 后端可能共享可变 GEMM workspace，故 target-verify 下低比多流仅在 CUTEDSL 后端启用。

---

## 四、并行 / 多流层增量

### 4.1 low_ratio_multi_stream（Blackwell ratio 1/2 多流）
- **是什么**：Blackwell（sm100）上，ratio 1/2 层在 **decode 或 target-verify** 时把低比 Q 准备
  （`_forward_prepare_low_ratio_multi_stream`）拆到多个 CUDA stream 上并行。
- **为什么快**：ratio 1/2 的压缩/索引/投影是多条相互独立的小 GEMM，小 batch decode 下单 kernel 占用率低，
  多流重叠可填满 SM。
- **代码**：判定 `low_ratio_multi_stream`（[deepseek_v4.py:2133](../../../python/sglang/srt/models/deepseek_v4.py#L2133)：`_is_cuda and is_blackwell and compress_ratio in (1,2) and alt_streams is not None`）；
  `_forward_prepare_low_ratio_multi_stream`（[:1473](../../../python/sglang/srt/models/deepseek_v4.py#L1473)）、调用点 [:2288](../../../python/sglang/srt/models/deepseek_v4.py#L2288)。
- **注意**：与 V4 的通用 `multi_stream` 互斥——通用多流显式排除 `compress_ratio in (1,2)`（[:2124](../../../python/sglang/srt/models/deepseek_v4.py#L2124)），低比走专用路径。

### 4.2 视觉融合的并行约束（换取正确性/简单性）
- **是什么**：构造视觉分支时校验 `attn_cp_size==1 and pp_group.world_size==1 and moe_a2a_backend.is_none()`，
  否则报错。即 V4.1 视觉**支持 TP/EP/DP，拒绝 CP/PP/MoE A2A**。
- **为什么（与加速相关）**：这不是加速本身，而是**为保证视觉 span 精确 scatter 写入而对并行策略的取舍**——
  CP 会切分含图像占位的序列破坏 span 顺序、PP 视觉融合在 first rank embedding 阶段、MoE A2A 的 token 重排与视觉 hash/路由不兼容。
  语言侧仍可用全部 V4 通信加速（EP/DeepEP/reduce_scatter/DP attention）。
- **代码**：[deepseek_v4.py:4867](../../../python/sglang/srt/models/deepseek_v4.py#L4867) 附近的 `ValueError` 校验。

---

## 五、调度 / 图执行层增量

### 5.1 breakable CUDA graph（`bcg_*`：低比源层 / Engram / attn）
- **是什么**：V4.1 把几条“可能在图内被打断/需 eager 兜底”的路径包成 **breakable CUDA graph**
  （`eager_on_graph(True)` 包裹）：低比压缩源层前向、Engram hash id 计算、带输出的注意力。
- **为什么快/稳**：既保留 CUDA Graph 消 launch 开销的收益，又允许这些含动态形状/条件（如 ratio 1/2 源层
  extend、Engram history 环、投机 verify）的算子在图内以可打断方式重放，不必整段退回 eager。
- **代码**：`bcg_deepseek_v4_low_ratio_sources`（[deepseek_v4.py:730](../../../python/sglang/srt/models/deepseek_v4.py#L730)）、
  `bcg_deepseek_v4_engram_hash_ids`（[:739](../../../python/sglang/srt/models/deepseek_v4.py#L739)）、
  `bcg_deepseek_v4_attention_with_output`（[:710](../../../python/sglang/srt/models/deepseek_v4.py#L710)）；调用点 [:2044](../../../python/sglang/srt/models/deepseek_v4.py#L2044)、[:4420](../../../python/sglang/srt/models/deepseek_v4.py#L4420)。

### 5.2 投机解码下的 Engram / mHC 支持
- **是什么**：MTP/NextN target-verify 时，Engram hasher 走 `MODE_VERIFY`（`block=draft_token_num`）
  正确处理 draft token 的 n-gram history；mHC 的 split-K + Sinkhorn 融合保证 verify 与 decode 数值一致。
- **为什么重要**：投机解码是 V4 继承的核心加速；V4.1 的新算子必须**批不变**才能让 prefill/verify/decode 三路
  logprob 一致，否则接受率下降、投机收益被抵消。
- **代码**：`EngramHasher.forward` 的 `MODE_VERIFY`（[engram.py:275](../../../python/sglang/srt/layers/engram.py#L275)）；mHC 批不变见 §2.3。

---

## 六、生效开关速查（V4.1 增量部分）

| 加速项 | 开关 / 判定 |
|---|---|
| fused wo_a (V4.1) | `SGLANG_DSV41_FUSED_WO_A` |
| Engram host offload | `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` |
| 低比多流 | 自动（Blackwell + `compress_ratio∈{1,2}` + decode/verify + alt_streams） |
| mHC 跨子层融合 | `hc_pre_from_prev_sublayer`（V4.1 config 默认 True）；`use_fused_mhc_post_pre` |
| Candidate 两级块选择 | `candidate_source_layer_id`（≥0 开启）、`candidate_topk_blocks` / `candidate_block_size`；sm≥100 |
| ratio 1/2 稀疏 | `compress_ratios` 含 1/2 + `kv_source_layer_ids` / `index_source_layer_ids` |
| fp4 索引键 | 低比自动 `force_fp4=True` |
| MXFP8 dense | 权重量化选 MXFP8 + Blackwell（CUTEDSL 后端） |
| 视觉分支 | `vision_n_layers>0`（并要求 TP/EP/DP，不可 CP/PP/MoE-A2A） |
| （继承 V4 全部开关） | 见 [deepseek_v4_推理加速方法.md](deepseek_v4_推理加速方法.md) §六 |

---

## 七、按推理阶段看 V4.1 增量加速

- **视觉预处理阶段**：pixel-shuffle 9→1 下采样（缩 LLM 序列）、ViT 复制式 DP（无头切分通信）、单图独立注意力。
- **Prefill（算力密集）**：ratio 1/2 稀疏 + Candidate 两级块选择（缩索引扫描）、dense_prefill_topk（ragged top-k）、
  mHC DeepGEMM tf32 大 M 融合、bcg 低比源层。继承 V4 的 TBO / CP / TC-piecewise CUDA Graph。
- **Decode（访存密集）**：**低比多流（Blackwell）**、共享压缩 latent + fp4 索引键（省 KV/索引带宽）、
  mHC split-K + 跨子层融合（省残差往返）、fused engram gate + fp8/host 表（省 Engram 开销）。继承 V4 的 MQA+FP8 KV、CUDA Graph、MTP。
- **两阶段通用**：mHC Sinkhorn/big_fuse 融合、MXFP8 dense GEMM、Engram fp8 表、fused wo_a。

---

## 八、小结

V4.1 的加速哲学是：**为四项新能力各自“配一套专门优化”，把新增开销压回 V4 的水平**。

1. **视觉**：靠 pixel-shuffle 把进 LLM 的 token 数从根上缩 9×，再用复制式 DP 免掉视觉塔通信——
   让“多模态”几乎不给语言主干加负担。
2. **Engram**：用 fp8 表 + TP 行切分 + 可选 host offload 压住记忆表的显存，用 fused engram gate 压住门控的 kernel 开销——
   用查表容量换算力，且查表本身很便宜。
3. **mHC**：残差流 ×4 的代价靠 Sinkhorn/big_fuse 融合、跨子层 post+pre+norm 融合边界、split-K 批不变多后端 kernel
   一起消化——既省 kernel 数与残差往返，又保投机解码所需的批不变性。
4. **ratio 1/2 稀疏**：靠 `kv_source_layer_ids` 共享压缩 latent + fp4 索引键把低比层的存储按源层计费、
   Candidate 两级块选择缩索引扫描、Blackwell 低比多流填满 decode——让“更细压缩粒度”不以显存/带宽爆炸为代价。

> 一句话：**V4 决定了主干“先天省”，V4.1 的增量优化保证四项新能力“后天不拖后腿”**——
> 新能力带来的额外算力/显存/带宽，几乎都有对应的融合核、共享存储、低精度或多流手段兜住。
