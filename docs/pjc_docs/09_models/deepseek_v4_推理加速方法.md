# DeepSeek-V4 推理加速方法梳理

> 代码位置：[python/sglang/srt/models/deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py)
>
> 本文梳理 DeepSeek-V4（纯语言模型，不含 V4.1 视觉分支）在 SGLang 中的推理加速手段。
> 按 **算法结构层 → 算子/kernel 融合层 → 量化层 → 并行/通信层 → 调度/图层** 五层组织，
> 每项说明「是什么 / 为什么快 / 代码位置 / 生效开关」。

---

## 〇、一句话总览

- **算法层**：MQA + Q/O 双低秩 + 稀疏压缩注意力（DSA），从根上砍掉注意力的算力和显存。
- **kernel 层**：QK-norm+RoPE+KV store 融合、wo_a absorb GEMM、MHC 融合归一化、多流并行。
- **量化层**：KV cache FP8、权重 FP8/FP4/MXFP8（主要压在 256 个路由专家上）。
- **通信层**：EP + DeepEP、共享专家融合、reduce_scatter 替代 all_reduce、DP attention、Context Parallel。
- **调度层**：CUDA Graph、Two-Batch Overlap（TBO）、MTP/NextN 投机解码。

| 分层 | 加速项 | 主要收益 |
|---|---|---|
| 结构 | MQA 单 KV 头 | KV cache/带宽 省 ~数十倍 |
| 结构 | Q/O 双低秩分解 | 投影参数与算力下降 |
| 结构 | DSA 压缩/稀疏注意力 | 长序列 O(N²)→近线性 |
| 结构 | RadixAttention 前缀复用 | 省重复 prefill |
| kernel | fused QK-norm+RoPE+store | 减 kernel launch 与访存 |
| kernel | wo_a absorb GEMM | 融合输出投影 |
| kernel | MHC 融合归一化 | 减 norm kernel 数 |
| kernel | multi-stream overlap | 计算/计算 并行 |
| 量化 | unified_kv FP8 KV cache | KV 带宽再减半 |
| 量化 | 权重 FP8/FP4/MXFP8 | 专家权重体积/带宽下降 |
| 通信 | EP + DeepEP + 共享专家融合 | 专家分散、省一次 all_reduce |
| 通信 | reduce_scatter 替代 all_reduce | combine 通信 ~减半 |
| 调度 | CUDA Graph | 消 kernel launch 开销 |
| 调度 | Two-Batch Overlap | 计算掩盖通信 |
| 调度 | MTP/NextN | 一步多 token |

---

## 一、算法 / 结构层（从根上省算力与显存）

### 1.1 MQA 单 KV 头
- **是什么**：`config.num_key_value_heads == 1`，64 个 Query 头共享同一个 KV 头。
- **为什么快**：KV cache 只需存 1 份（而非 64 份），decode 阶段显存占用和 HBM 读带宽大幅下降——decode 是访存瓶颈，KV 越小越快。
- **代码**：`MqaAttentionBase`（`assert config.num_key_value_heads == 1`），`attn_mqa = RadixAttention(..., num_kv_heads=1)`。

### 1.2 Q / O 双低秩分解
- **是什么**：
  - Q：`wq_a (4096→1024) → q_norm → wq_b (1024→32768)`，用秩 1024 的中间表示代替直接 4096→32768。
  - O：`wo_a (分组降维 4096→8192) → wo_b (8192→4096)`，按 `o_groups=8` 分组低秩。
- **为什么快**：把一个大投影矩阵拆成两个小矩阵，参数量和 GEMM FLOPs 显著下降，同时低秩表示也减小激活体积。
- **代码**：`MqaAttentionBase.__init__` 里的 `wq_a/wq_b/wo_a/wo_b`。

### 1.3 稀疏 / 压缩注意力（DSA, DeepSeek Sparse Attention）
- **是什么**：由 `compress_ratios[layer_id]` 逐层决定注意力形态：
  - `compress_ratio == 4`：挂 `Compressor`（压缩 KV）+ `C4Indexer`（稀疏选择）。
  - `compress_ratio == 128`：仅 `Compressor`（强压缩）。
  - `compress_ratio ∈ {1,2}` 且在 `kv_source_layer_ids` / `index_source_layer_ids`：挂 `DeepseekV41Compressor` / `DeepseekV41Indexer`。
  - `compress_ratio == 0`：普通稠密 MQA。
- **为什么快**：
  - **Compressor** 把长 KV 序列压成更短序列（比例 4 或 128），注意力长度和 cache 直接缩小。
  - **Indexer** 让每个 query 只对 top-`index_topk=512` 个 KV block 做注意力（`index_head_dim=128`, `index_n_heads=64`），把长上下文注意力从 O(N²) 降到近线性。
- **代码**：`MQALayer.__init__` 里 `self.compressor` / `self.indexer` 的构造分支；`is_dsa_enable_prefill_cp`。

### 1.4 RadixAttention 前缀复用
- **是什么**：KV cache 用 radix 树组织，命中公共前缀时直接复用已算好的 KV。
- **为什么快**：多请求共享 system prompt / 少量变化的上下文时，省掉重复 prefill 计算。
- **代码**：`attn_mqa = RadixAttention(...)`。

### 1.5 attention sink
- **是什么**：每个头一个可学习标量 `attn_sink[64]`（fp32），作为一个"零 value 的额外 token"参与 softmax。
- **为什么快/稳**：给 softmax 一个"泄压"出口，稳定长序列注意力分布，避免为数值稳定性引入额外兜底逻辑。
- **代码**：`self.attn_sink = nn.Parameter(torch.empty(n_heads, dtype=torch.float32))`。

---

## 二、算子 / kernel 融合层

### 2.1 fused QK-norm + RoPE + KV store
- **是什么**：把 `q_norm`/`kv_norm`（RMSNorm）、RoPE、以及把 K/V 写入 KV cache 三个步骤融合成单个 kernel。
- **为什么快**：减少 kernel launch 次数和中间张量的显存往返（访存密集操作合并）。
- **代码**：`fused_qk_norm_rope_swa_store`（`kernels/ops/attention/fused_qk_norm_rope_store.py`），门控 `use_fused_qk_norm_rope`、`do_fused_qk_norm_rope`；env `SGLANG_OPT_USE_FUSED_QK_NORM_ROPE`、`SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY`。

### 2.2 wo_a absorb GEMM
- **是什么**：输出投影的 `wo_a` 用一个批量 GEMM「吸收」，替代独立的 inverse-RoPE + 投影。
  - CUDA：DeepGEMM `fp8_einsum`（fp8 权重直接批量算）。
  - ROCm gfx950：aiter mxscale BMM。
  - NPU arch35：批量 MXFP8 GEMM。
- **为什么快**：把多步小算子合并成一次大 batched GEMM，吞吐更高；配合 fp8 权重减少访存。
- **代码**：`use_fused_wo_a` 判定（bf16 + 特定 shape 时），`deep_gemm.fp8_einsum(...)`；`_wo_a_batched_gemm_bf16`。

### 2.3 MHC 融合归一化（V4 特有）
- **是什么**：V4 用「多头混合归一化」（Sinkhorn 迭代，`hc_sinkhorn_iters=20`, `hc_mult=4`）代替普通 RMSNorm 的一部分，并把相邻子层的 pre-norm / post-norm 与该 mix **融合进同一个 kernel**。
- **为什么快**：减少归一化相关的 kernel 数量和残差往返；`hc_pre_from_prev_sublayer` 把 pre-mix 提前到上一子层边界，cross-layer 融合进一步省一次读写。
- **代码**：`make_hc_mixing_params` / `make_hc_head_params` / `refresh_mhc_norm_weight_cache`；`use_fused_mhc_post_pre`、`is_cross_layer_mhc_fusion_enabled`；`tf32_hc_prenorm_gemm`。

### 2.4 多流并行（multi-stream overlap）
- **是什么**：在 CUDA graph 捕获 / 小 batch 时，把注意力内部的 Q、KV、Compressor、Indexer、共享专家等拆到多个 CUDA stream 上并行执行。
- **为什么快**：让互相独立的 GEMM/kernel 在不同 stream 上重叠，填满 GPU（尤其小 batch 时单 kernel 占用率低）。
- **代码**：`_forward_prepare_multi_stream` / `_forward_prepare_multi_stream_npu` / `_forward_prepare_multi_stream_hip` / `_forward_prepare_low_ratio_multi_stream`；`self.alt_streams`、`_multi_stream_bs_limit`（Blackwell 128 / 其它 64）；env `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`、`SGLANG_NPU_USE_MULTI_STREAM`、`SGLANG_ROCM_USE_MULTI_STREAM`。

### 2.5 tiny_router_gemm
- **是什么**：门控（router）GEMM 在 token 数很小时走专用小 GEMM 实现。
- **为什么快**：小 M 维下通用 GEMM 效率低，专用 kernel 更省。
- **代码**：`MoEGate.forward` 里 `hidden_states.shape[0] <= self.tiny_router_gemm_max_tokens` → `tiny_gemm_bf16`。

---

## 三、量化层

### 3.1 KV cache 量化 + unified_kv 双池
- **是什么**：`unified_kv` 把每层 KV 拆成 nope / rope 两个独立 pool（`get_unified_kv` / `get_unified_kv_rope`），并支持 **FP8 KV cache**（`is_unified_kv_fp8`）。
- **为什么快**：KV cache 是 decode 的访存大头，FP8 让 KV 体积和读带宽再减半；双池布局让 fused kernel 按 pool 行 stride 直接读，Q 与 KV 布局对齐避免额外拷贝。
- **代码**：`is_unified_kv_triton` / `is_unified_kv_fp8`（`kernels/ops/attention/dsv4/unified_kv_kernels/env_gate.py`）；forward 中 `unified_fp8_decode` / `unified_fp8_prefill` / `unified_fp8_verify` 分支；env `SGLANG_DSV4_UNIFIED_KV_FP8`。
- **注意**：fp8 two-pool + 投机 verify 需要 fused verify store（否则报错）；不支持 DSA prefill CP。

### 3.2 权重量化 FP8 / FP4 / MXFP8
- **是什么**：线性层与专家权重支持多种低精度：
  - **block-wise FP8**：带 `weight_scale_inv`，DeepGEMM UE8M0 scale（`DEEPGEMM_SCALE_UE8M0`）。
  - **FP4**：modelopt（`ModelOptFp4LinearMethod`），主要用于路由专家。
  - **MXFP8**：NPU arch35（`use_npu_arch35_mxfp8_wo_a`）。
  - **expert_pack**：专家权重打包量化格式。
- **为什么快**：256 个路由专家是模型主要权重（占绝大部分显存/带宽），低精度让专家权重体积和读带宽显著下降，GEMM 也更快。
- **代码**：`wo_a_fp8_gemm_enabled`、`_FP8_WO_A_UE8M0`；`DeepseekV2MoE` 里 `get_moe_impl_class(quant_config)`；`quant_config.is_fp4_experts`。

---

## 四、并行 / 通信层

### 4.1 专家并行 EP + DeepEP/MegaMOE + 共享专家融合
- **是什么**：
  - **EP**：256 个路由专家分散到多张卡（`moe_ep_size`），每卡只算本地专家。
  - **DeepEP / MegaMOE**：专用的 all-to-all dispatch/combine backend。
  - **共享专家融合**：把共享专家当作每个 EP rank 上「第 257 个本地专家」融进 MoE kernel（`num_fused_shared_experts>0` / `has_per_rank_fused_shared_slots`，如 EP=16 时 256→272）。
- **为什么快**：EP 让单卡只承担部分专家的计算与权重；共享专家融合省掉一次独立 MLP kernel 和它的 all_reduce。
- **代码**：`DeepseekV2MoE.__init__` 里 `num_experts_for_moe` / `top_k_for_moe` 计算；`get_moe_a2a_backend()`（`is_deepep` / `is_megamoe` / ...）。

### 4.2 reduce_scatter 替代 all_reduce
- **是什么**：MoE 合并（combine）阶段用 `reduce_scatter` / `reduce_scatterv` 完成「求和 + 分散」，替代 `all_reduce + dp_scatter`。
- **为什么快**：`all_reduce` 通信量约为 `reduce_scatter` 的 2 倍；直接 reduce_scatter 把 combine 通信量约减半，且把后续算子放在更小的每-rank 局部切片上做。
- **代码**：`_use_reduce_scatterv` / `_use_reduce_scatter` / `mlp_reduce_scatter`；`dp_reduce_scatter_tensor`、`tp_group.reduce_scatterv`；`should_use_dp_reduce_scatterv`。

### 4.3 DP attention + attn_tp
- **是什么**：注意力可走数据并行（DP attention），并按 `attn_tp` 切分注意力张量并行；`wq_b/wo_a` 按 `attn_tp_rank/size` 切列/行。
- **为什么快**：注意力与 MoE 采用不同并行策略，各自取最优（注意力 DP 减通信，MoE EP 分专家）。
- **代码**：`is_dp_attention_enabled()`；`get_attn_tp_context()`；`ColumnParallelLinear(..., tp_rank=self.attn_tp_rank, tp_size=self.attn_tp_size)`。

### 4.4 Context Parallel（prefill CP）
- **是什么**：长 prefill 序列切分到多卡并行（DSA prefill CP）。
- **为什么快**：超长上下文 prefill 的注意力/激活按序列维切分，突破单卡显存与算力上限。
- **代码**：`dsa_enable_prefill_cp`、`dsa_use_prefill_cp(forward_batch)`、`dsa_cp_reduce_scatter_hidden_states`。

---

## 五、调度 / 图执行层

### 5.1 CUDA Graph
- **是什么**：把 decode（及部分 prefill）的算子序列捕获成 CUDA Graph 重放，支持 piecewise / breakable / TC-piecewise 多种粒度。
- **为什么快**：消除逐 kernel 的 CPU launch 开销——小 batch decode 时 launch 开销占比很高。
- **代码**：`get_is_capture_mode`、`is_in_breakable_cuda_graph`、`check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)`；`cuda_graph_config` / `tc_piecewise_cuda_graph`。

### 5.2 Two-Batch Overlap（TBO，prefill 两批重叠）
- **是什么**：prefill 阶段把 batch 拆成两个 microbatch 交错执行，用一个 ubatch 的 attn+MoE 计算掩盖另一个 ubatch 的 a2a / all_gatherv / reduce_scatterv 通信。
  - EP / mori 路径：`_forward_layers_tbo` + op 分解（`op_gate` / `op_experts` / `op_combine` 等）。
  - 非 EP（DP TP-MoE）路径：重叠 DP all_gatherv（pre-MoE gather）+ reduce_scatterv（post-MoE combine）。
- **为什么快**：通信与计算重叠，隐藏 MoE all-to-all 的时延，提升 prefill 吞吐。
- **代码**：`_forward_layers_tbo`、`execute_overlapped_operations`、`OperationsStrategy`、`two_batch_overlap`；开关 `--enable-two-batch-overlap`（`_should_run_tbo` 相关门控）。

### 5.3 MTP / NextN 投机解码
- **是什么**：`is_nextn` 分支支持 Multi-Token Prediction / NextN draft-verify 投机解码，一次前向验证多个候选 token。
- **为什么快**：用 draft 模型一次产出多 token，主模型批量 verify，摊薄每 token 的访存/计算成本。
- **代码**：`is_nextn` 贯穿 `DeepseekV4DecoderLayer` / `DeepseekV2MoE`；unified_kv 的 fused verify store 专为 target-verify 服务（`forward_batch.forward_mode.is_target_verify()`）。

---

## 六、生效开关速查（env / server args）

| 加速项 | 开关 |
|---|---|
| 多流并行 | `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` / `SGLANG_NPU_USE_MULTI_STREAM` / `SGLANG_ROCM_USE_MULTI_STREAM` |
| fused QK-norm+RoPE | `SGLANG_OPT_USE_FUSED_QK_NORM_ROPE` / `SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY` |
| wqa/wkv 融合 | `SGLANG_OPT_FUSE_WQA_WKV` |
| unified_kv FP8 | `SGLANG_DSV4_UNIFIED_KV_FP8` |
| fused wo_a (V4.1) | `SGLANG_DSV41_FUSED_WO_A` |
| 共享专家 TP1 | `SGLANG_SHARED_EXPERT_TP1` |
| Two-Batch Overlap | `--enable-two-batch-overlap` |
| MoE a2a backend | `--moe-a2a-backend`（deepep / megamoe / mooncake / ...） |
| 专家/权重量化 | `--quantization`（fp8 / modelopt fp4 / expert_pack / ...） |
| 张量/专家/数据并行 | `--tp` / `--ep-size` / `--dp` / `--pp` |

---

## 七、按推理阶段看哪些加速在起作用

- **Prefill（算力密集）**：Q/O 低秩、DSA 压缩+稀疏注意力（缩短长序列）、TBO 两批重叠、Context Parallel、TC-piecewise CUDA Graph、fused QK-norm+RoPE。
- **Decode（访存密集）**：MQA 单 KV 头 + FP8 KV cache（省带宽）、CUDA Graph（省 launch）、多流并行（填满小 batch）、MTP/NextN（一步多 token）、tiny_router_gemm。
- **两阶段通用**：权重 FP8/FP4 量化、EP + 共享专家融合、reduce_scatter、MHC 融合归一化、wo_a absorb GEMM、RadixAttention 前缀复用。

---

## 八、小结

DeepSeek-V4 的加速是「**结构先天省 + kernel 后天融 + 量化压带宽 + 并行分负载 + 调度隐延迟**」的组合：

1. **结构层**（MQA + 双低秩 + DSA）决定了它天然比稠密注意力省算力/显存，尤其长上下文；
2. **kernel 融合**（QK-norm+RoPE+store、wo_a absorb、MHC 融合、多流）把访存密集的小算子合并、并行；
3. **量化**（FP8 KV + FP8/FP4 权重）把 decode 的带宽瓶颈和专家权重体积压下来；
4. **通信/并行**（EP + DeepEP + 共享专家融合 + reduce_scatter + DP/CP）让 MoE 在多卡上高效切分、少通信；
5. **调度**（CUDA Graph + TBO + MTP）消 launch 开销、用计算掩盖通信、一步多 token。




