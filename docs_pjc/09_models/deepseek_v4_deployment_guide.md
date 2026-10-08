# SGLang 部署 DeepSeek-V4 完整方案

## 1. 模型概述

### 1.1 模型变体

| 变体 | 总参数量 | 激活参数 (MoE) | HuggingFace 路径 | 用途 |
|------|---------|---------------|-----------------|------|
| DeepSeek-V4-Flash | 284B | 13B | `deepseek-ai/DeepSeek-V4-Flash` | 单节点部署，低延迟场景 |
| DeepSeek-V4-Pro | 1.6T | 49B | `deepseek-ai/DeepSeek-V4-Pro` | 高容量，多节点部署 |
| DeepSeek-V4-Flash-FP8 | 284B | 13B | `sgl-project/DeepSeek-V4-Flash-FP8` | Hopper GPU FP8 部署 |
| DeepSeek-V4-Pro-FP8 | 1.6T | 49B | `sgl-project/DeepSeek-V4-Pro-FP8` | Hopper GPU FP8 部署 |

### 1.2 V4 vs V3 架构差异

| 特性 | DeepSeek V3 | DeepSeek V4 |
|------|-------------|-------------|
| 注意力机制 | MLA (Multi-head Latent Attention) | MQA + 压缩KV + 输出LoRA |
| KV头数 | kv_lora_rank=512 (latent) | 1 KV head, FP8 nope + BF16 rope |
| KV压缩 | 无 (radix cache) | CSA (4x) + HCA (128x) 逐层配置 |
| 稀疏注意力 | DSA indexer (V3.2) | C4 Indexer (top-512) |
| 隐藏状态 | 标准 (T, D) | Multi-Head Combination (T, hc_mult=4, D) |
| MoE路由 | 学习门控 | Hash-based (前N层) + 学习门控 |
| 输出投影 | 标准 wo | LoRA: wo_a (grouped) + wo_b |
| 评分函数 | softmax/sigmoid | sqrtsoftplus |
| 注意力后端 | flashinfer_mla / dsa | dsv4 (定制) |
| 上下文长度 | 128K | 1M tokens |

### 1.3 核心架构参数

```
hidden_layers: 43
attention_heads: 64
hidden_size: 4096
qk_nope_head_dim: 448
qk_rope_head_dim: 64
kv_lora_rank: 512
q_lora_rank: 1024
o_lora_rank: 1024
o_groups: 8
hc_mult: 4 (Multi-Head Combination)
n_routed_experts: 256
num_experts_per_tok: 6
n_shared_experts: 1
n_group: 8
window_size: 128 (SWA)
index_topk: 512
vocab_size: 129,280
max_position_embeddings: 65,536
```

---

## 2. 硬件部署矩阵

### 2.1 支持的GPU平台

| GPU | 量化方式 | Flash TP | Pro TP | 多节点 | MoE Runner |
|-----|---------|----------|--------|--------|------------|
| B200 (Blackwell) | FP4 原生 | TP=4 单节点 | TP=8 单节点 | 否 | flashinfer_mxfp4 / megamoe |
| B300 (Blackwell) | FP4 原生 | TP=4 单节点 | TP=4 单节点 | 否 | flashinfer_mxfp4 / megamoe |
| GB200 (Grace Blackwell) | FP4 原生 | TP=4 单节点 | TP=8 双节点 | Pro需要 | flashinfer_mxfp4 / megamoe |
| GB300 | FP4 原生 | TP=4 单节点 | TP=4 单节点 | 否 | flashinfer_mxfp4 / megamoe |
| H200 (Hopper, FP8) | FP8 | TP=4 单节点 | TP=16 双节点 | Pro需要 | deep_gemm / deepep |
| H200 (Hopper, FP4) | FP4 (Marlin) | TP=4 单节点 | TP=8 单节点 | 否 | marlin / flashinfer_mxfp4 |
| H100 (Hopper, FP4) | FP4 (Marlin) | TP=8 单节点 | TP=16 双节点 | Pro需要 | marlin |
| H100 (Hopper, FP8) | FP8 | TP=8 单节点 | 仅Flash | 否 | deep_gemm / deepep |
| AMD MI35x | FP4 / FP8 | TP=8 单节点 | TP=8 单节点 | 否 | aiter |

### 2.2 显存需求估算

| 配置 | 模型权重 | KV Cache | 总估算 |
|------|---------|----------|--------|
| Flash FP4, TP=4 | ~70GB (FP4 experts + FP8 dense) | ~40GB (FP8 KV) | ~110GB → 4×B200(192GB) |
| Flash FP8, TP=4 | ~140GB (FP8 全精度) | ~40GB | ~180GB → 4×H200(141GB) 紧凑 |
| Pro FP4, TP=8 | ~400GB (FP4 experts + FP8 dense) | ~80GB | ~480GB → 8×B200(192GB) |
| Pro FP8, TP=16 | ~800GB (FP8 全精度) | ~160GB | ~960GB → 16×H200(141GB) |

---

## 3. 部署方案 (Recipes)

SGLang 为 DeepSeek-V4 提供五种部署方案，覆盖不同延迟/吞吐权衡：

### 3.1 Low-Latency (低延迟)

**目标**: 最小化单请求延迟 (TPOT < 4ms)

**特点**:
- EAGLE 投机解码: steps=3, topk=1, draft_tokens=4
- 无 DP-Attention (纯 TP)
- 适合交互式场景 (chatbot, IDE copilot)

**B200 Flash 启动命令**:
```bash
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --speculative-algo EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --chunked-prefill-size 4096 \
  --host 0.0.0.0 \
  --port 30000
```

**H200 Flash FP8 启动命令**:
```bash
SGLANG_DSV4_FP4_EXPERTS=0 \
sglang serve \
  --trust-remote-code \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --tp 4 \
  --speculative-algo EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 \
  --port 30000
```

**H200 Flash FP4 (Marlin) 启动命令**:
```bash
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 4 \
  --moe-runner-backend marlin \
  --speculative-algo EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 \
  --port 30000
```

### 3.2 Balanced (均衡)

**目标**: 延迟与吞吐的平衡

**特点**:
- DP-Attention + DeepEP Expert Parallelism
- 轻量 EAGLE: steps=1, topk=1, draft_tokens=2
- 适合中等并发在线服务

**B200 Flash 启动命令**:
```bash
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 4 \
  --dp 4 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --speculative-algo EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
  --host 0.0.0.0 \
  --port 30000
```

**H200 Flash FP8 启动命令**:
```bash
SGLANG_DSV4_FP4_EXPERTS=0 \
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
sglang serve \
  --trust-remote-code \
  --model-path sgl-project/DeepSeek-V4-Flash-FP8 \
  --tp 4 \
  --dp 4 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --speculative-algo EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --cuda-graph-max-bs 128 \
  --max-running-requests 128 \
  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
  --host 0.0.0.0 \
  --port 30000
```

### 3.3 Max-Throughput (最大吞吐)

**目标**: 最大化 tokens/sec 吞吐

**特点**:
- DP-Attention + DeepEP
- 禁用投机解码 (高并发下 verify 开销 > 收益)
- 适合批处理、离线推理

**B200 Flash 启动命令**:
```bash
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 4 \
  --dp 4 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
  --host 0.0.0.0 \
  --port 30000
```

### 3.4 Context-Parallel (上下文并行)

**目标**: 加速长上下文 prefill

**特点**:
- DSA prefill context-parallel (round-robin-split)
- 仅支持单机 (tp_size <= 8)
- 适合长文档处理、RAG

**B200 Flash 启动命令**:
```bash
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 4 \
  --attn-cp-size 4 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --speculative-algo EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --enable-dsa-prefill-context-parallel \
  --dsa-prefill-cp-mode round-robin-split \
  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
  --host 0.0.0.0 \
  --port 30000
```

### 3.5 PD-Disagg (Prefill-Decode 分离)

**目标**: 独立扩展 prefill 和 decode 实例

**特点**:
- Prefill 和 Decode 运行在独立 GPU 组
- 通过 Mooncake/NIXL 传输 KV Cache
- 适合大规模生产部署

**B200 Flash PD 部署** (3 进程):

Prefill 实例:
```bash
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 \
  --disaggregation-mode prefill \
  --moe-a2a-backend deepep \
  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
  --host 0.0.0.0 \
  --port 30000
```

Decode 实例:
```bash
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 2 \
  --disaggregation-mode decode \
  --moe-a2a-backend deepep \
  --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
  --host 0.0.0.0 \
  --port 30001
```

Router:
```bash
python -m sglang_router.launch_server \
  --host 0.0.0.0 \
  --port 8000
```

---

## 4. MegaMoE 加速 (Blackwell 专属)

MegaMoE 将 expert dispatch + GEMM 融合为单一 kernel，显著提升 MoE 层吞吐。

### 4.1 W4A8 模式 (FP4权重, FP8激活)

```bash
SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096 \
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 4 \
  --dp 4 \
  --enable-dp-attention \
  --moe-a2a-backend megamoe \
  --speculative-algo EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --host 0.0.0.0 \
  --port 30000
```

### 4.2 W4A4 模式 (FP4权重, FP4激活)

更高吞吐，精度损失可忽略 (~89.5 GPQA on Pro):

```bash
SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096 \
SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1 \
SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1 \
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 4 \
  --dp 4 \
  --enable-dp-attention \
  --moe-a2a-backend megamoe \
  --speculative-algo EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --host 0.0.0.0 \
  --port 30000
```

### 4.3 MegaMoE 限制

- 仅支持 Blackwell GPU (B200/B300/GB200/GB300)
- 不支持 low-latency / cp / pd-disagg 方案
- 推荐 `NUM_MAX_TOKENS_PER_RANK`: balanced=4096, max-throughput=8320

---

## 5. 高级特性

### 5.1 HiCache (分层KV缓存)

GPU → CPU → Storage 三级缓存，扩展有效上下文容量：

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 4 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-write-policy write_through \
  --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct \
  --host 0.0.0.0 \
  --port 30000
```

### 5.2 Reasoning Parser (思维链分离)

启用 `deepseek-v4` reasoning parser 将思考过程与最终答案分离：

```bash
sglang serve \
  ... \
  --reasoning-parser deepseek-v4
```

三种推理模式:
- **Non-think**: 快速直觉响应
- **Think High**: 有意识逻辑分析
- **Think Max**: 推理能力全开 (建议 ≥ 384K 上下文窗口)

### 5.3 Tool Calling (函数调用)

使用 DSML XML 格式的工具调用：

```bash
sglang serve \
  ... \
  --tool-call-parser deepseekv4
```

### 5.4 HiSparse (分层稀疏注意力)

GPU + Host 分层稀疏缓存，需要禁用 radix cache：

```bash
sglang serve \
  ... \
  --disable-radix-cache \
  --page-size 256
```

---

## 6. 多节点部署

### 6.1 适用场景

- H200 FP8 Pro: TP=16, 2 节点
- GB200 Pro: TP=8, 2 节点
- H100 FP4 Pro: TP=16, 2 节点

### 6.2 启动命令

每个节点运行相同命令，仅 `--node-rank` 不同：

```bash
sglang serve \
  --trust-remote-code \
  --model-path sgl-project/DeepSeek-V4-Pro-FP8 \
  --tp 16 \
  --nnodes 2 \
  --node-rank <0或1> \
  --dist-init-addr <node0-ip>:20000 \
  --host 0.0.0.0 \
  --port 30000
```

### 6.3 GB200 多节点额外配置

```bash
NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
sglang serve ...
```

---

## 7. 环境变量参考

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `SGLANG_DSV4_FP4_EXPERTS` | True | 启用FP4 expert权重 (H200 FP8需设为0) |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | - | DeepEP dispatch buffer大小 |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` | - | MegaMoE token buffer |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS` | False | MegaMoE FP4激活 |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND` | False | MegaMoE MXFP4格式 |
| `SGLANG_OPT_USE_ONLINE_COMPRESS` | False | 在线C128压缩 |
| `SGLANG_OPT_USE_COMPRESSOR_V2` | True | V2压缩器后端 |
| `SGLANG_OPT_FP8_WO_A_GEMM` | True | FP8输出投影GEMM |
| `SGLANG_OPT_FUSE_WQA_WKV` | True | 融合Q/KV下投影 |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` | True | 多流重叠 |
| `SGLANG_OPT_DEEPGEMM_HC_PRENORM` | True | DeepGEMM HC pre-norm |
| `SGLANG_OPT_USE_TILELANG_MHC_PRE` | True | TileLang MHC pre kernel |
| `SGLANG_OPT_USE_TILELANG_MHC_POST` | True | TileLang MHC post kernel |
| `SGLANG_ENABLE_UNIFIED_RADIX_TREE` | False | 统一Radix树 (HiCache需要) |
| `SGLANG_SHARED_EXPERT_TP1` | False | 共享expert TP=1 (H100 Pro) |
| `SGLANG_DSV4_REASONING_EFFORT` | "" | 推理努力控制 |

---

## 8. 性能基准

### 8.1 延迟基准 (Low-Latency, concurrency=1)

| 硬件 | 模型 | TPOT (ms) | TTFT (ms) | Accept Length |
|------|------|-----------|-----------|---------------|
| B200×4 | Flash FP4 | 3.40 | 102.72 | 2.73 |
| H200×4 | Flash FP4 | 3.50 | 147.26 | 2.96 |

### 8.2 吞吐基准 (Max-Throughput, concurrency=100)

| 硬件 | 模型 | Output tok/s | Total tok/s | 请求吞吐 (req/s) |
|------|------|-------------|-------------|-----------------|
| B200×4 (MegaMoE W4A4) | Flash FP4 | 4,860 | 9,740 | 9.51 |
| H200×4 | Flash FP4 | 2,575 | 5,159 | 5.04 |

### 8.3 精度基准

| 模型 | 硬件 | GSM8K | MMLU (10 subjects) |
|------|------|-------|-----|
| Pro FP4 | B300 | 0.965 | 0.879 |
| Pro FP4 | H200 | 0.975 | 0.893 |
| Flash FP4 | B200 | ≥0.93 (CI gate) | - |
| Flash FP8 | H200 | ≥0.93 (CI gate) | - |

---

## 9. 系统架构

### 9.1 请求处理流程

```
HTTP Request
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Tokenizer + encoding_dsv4 (DSML tool-call grammar)         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Scheduler (schedule_policy: lpm/fcfs)                       │
│  ├── waiting_queue → prefill batch                          │
│  └── running_batch → decode step                            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Worker (DeepseekV4ForCausalLM)                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Per-Layer Pipeline:                                   │  │
│  │  1. hc_pre (MHC Sinkhorn mixing)                      │  │
│  │  2. MQALayer (Compressed Sparse Attention)             │  │
│  │     ├── wq_a + wkv → Q/KV projection                  │  │
│  │     ├── Compressor (c4/c128) → KV compression         │  │
│  │     ├── C4Indexer → top-512 sparse selection           │  │
│  │     ├── FlashMLA decode / FlashAttn prefill            │  │
│  │     └── wo_a + wo_b (LoRA output projection)          │  │
│  │  3. hc_post (MHC recombination)                        │  │
│  │  4. hc_pre (MHC for FFN)                              │  │
│  │  5. DeepseekV2MoE (256 experts, top-6)                │  │
│  │     ├── HashTopK (first 3 layers) / Learned Gate      │  │
│  │     ├── DeepEP/MegaMoE dispatch (all-to-all)          │  │
│  │     ├── FP4/FP8 Grouped GEMM (DeepGEMM)              │  │
│  │     └── DeepEP combine (all-to-all gather)            │  │
│  │  6. hc_post (MHC for FFN)                             │  │
│  └───────────────────────────────────────────────────────┘  │
│  Final: hc_head → LM Head → LogitsProcessor                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  EAGLE Speculative Decoding (DeepseekV4ForCausalLMNextN)     │
│  ├── e_proj + h_proj (embedding/hidden projection)          │
│  ├── Single decoder layer (compress_ratio=0)                │
│  └── hc_head → draft logits                                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Detokenizer → HTTP Response                                 │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 KV Cache 架构

```
┌─────────────────────────────────────────────────────────────┐
│  DeepSeekV4SingleKVPool (per layer)                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Per-token layout (584 bytes):                          ││
│  │  ┌──────────────────┬──────────────┬──────────────────┐ ││
│  │  │ qk_nope (448B)   │ qk_rope (128B)│ scales (8B)     │ ││
│  │  │ FP8 E4M3         │ BF16          │ per-block FP8   │ ││
│  │  └──────────────────┴──────────────┴──────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  CompressStatePool (ring buffer per layer)                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  c4 layers: ring_size=8, stores KVAndScore              ││
│  │  c128 layers: ring_size=128 (or 1 for online mode)      ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  HiSparse (optional): GPU → Host offload via CUDA kernel    │
│  HiCache (optional): GPU → CPU → Storage tiered caching     │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 DP-Attention + Expert Parallelism 数据流

```
┌─────────────────────────────────────────────────────────────┐
│  DP Attention Phase (每GPU处理token子集)                      │
│  GPU0: tokens[0:N/4]  GPU1: tokens[N/4:N/2]  ...           │
└─────────────────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────────────────────┐
│  dp_gather_partial → 收集所有tokens用于MoE                    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  MoE Gate → TopK Expert Selection                            │
│  ├── HashTopK (前3层): token_id → expert_id 查表             │
│  └── Learned Gate (其余层): sigmoid + grouped_topk           │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  DeepEP Dispatch (All-to-All)                                │
│  GPU0: experts[0:64]  GPU1: experts[64:128]  ...            │
│  每GPU接收属于本地experts的tokens                              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Grouped GEMM (DeepGEMM FP8/FP4)                            │
│  gate_proj → up_proj → SiLU → down_proj                     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  DeepEP Combine (All-to-All Gather)                          │
│  结果返回原始token所在GPU                                     │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  dp_scatter → 分散回DP Attention布局                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. 关键代码文件索引

### 10.1 模型定义

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/models/deepseek_v4.py` | 主模型实现 |
| `python/sglang/srt/models/deepseek_v4_nextn.py` | EAGLE draft模型 |
| `python/sglang/srt/configs/deepseek_v4.py` | 模型配置 |
| `python/sglang/srt/arg_groups/deepseek_v4_hook.py` | 启动参数默认值/约束 |

### 10.2 注意力后端

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/layers/attention/deepseek_v4_backend.py` | DSV4 注意力后端 (CUDA) |
| `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py` | DSV4 AMD后端 |
| `python/sglang/srt/layers/attention/dsv4/compressor.py` | KV压缩器 (c4/c128) |
| `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` | V2压缩器 |
| `python/sglang/srt/layers/attention/dsv4/indexer.py` | C4 Indexer (top-512稀疏) |
| `python/sglang/srt/layers/attention/dsv4/metadata.py` | 注意力元数据管理 |

### 10.3 内存管理

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` | V4 KV Cache池 |
| `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py` | 压缩状态池 |
| `python/sglang/srt/mem_cache/hisparse_memory_pool.py` | HiSparse内存池 |

### 10.4 MoE 层

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | FusedMoE基类 |
| `python/sglang/srt/layers/moe/ep_moe/layer.py` | DeepEP MoE |
| `python/sglang/srt/layers/moe/mega_moe.py` | MegaMoE (DeepGEMM融合) |
| `python/sglang/srt/layers/moe/hash_topk.py` | Hash-based TopK |
| `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` | DeepEP dispatcher |
| `python/sglang/srt/layers/moe/deepep_waterfill.py` | DeepEP Waterfill |

### 10.5 CUDA/JIT Kernels

| 文件 | 说明 |
|------|------|
| `python/sglang/jit_kernel/dsv4/attn.py` | V4注意力kernel |
| `python/sglang/jit_kernel/dsv4/compress.py` | 压缩kernel |
| `python/sglang/jit_kernel/dsv4/elementwise.py` | 融合elementwise |
| `python/sglang/jit_kernel/dsv4/moe.py` | MoE kernel |
| `python/sglang/jit_kernel/dsv4/topk.py` | TopK kernel |
| `python/sglang/jit_kernel/csrc/deepseek_v4/` | 20+ CUDA头文件 |
| `sgl-kernel/csrc/attention/cutlass_mla_kernel.cu` | SM100 MLA kernel |

### 10.6 其他

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/layers/mhc_head.py` | MHC LM Head Triton kernel |
| `python/sglang/srt/layers/deepseek_v4_rope.py` | V4 RoPE (YaRN) |
| `python/sglang/srt/function_call/deepseekv4_detector.py` | 工具调用检测器 |
| `python/sglang/srt/entrypoints/openai/encoding_dsv4.py` | DSV4消息编码 |

---

## 11. Docker 部署

### 11.1 通用镜像

```bash
docker pull lmsysorg/sglang:latest

docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<your-hf-token>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    sglang serve <参数>
```

### 11.2 专用镜像

| 镜像标签 | 平台 | GPU |
|---------|------|-----|
| `lmsysorg/sglang:deepseek-v4-hopper` | linux/amd64 | H200 |
| `lmsysorg/sglang:deepseek-v4-blackwell` | linux/amd64 | B200 |
| `lmsysorg/sglang:deepseek-v4-b300` | linux/amd64 | B300 |
| `lmsysorg/sglang:deepseek-v4-grace-blackwell` | linux/arm64 | Grace Blackwell (ARM) |

### 11.3 PD-Disagg Docker 注意事项

H200 PD-Disagg 需要 IB 设备暴露：
```bash
docker run --privileged --ulimit memlock=-1 \
    --device /dev/infiniband:/dev/infiniband \
    --cap-add IPC_LOCK \
    ...
```

---

## 12. AMD ROCm 部署 (MI35x)

```bash
SGLANG_USE_AITER=1 \
SGLANG_USE_ROCM700A=1 \
sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --tp 8 \
  --attention-backend compressed \
  --disable-radix-cache \
  --page-size 256 \
  --chunked-prefill-size 8192 \
  --disable-shared-experts-fusion \
  --host 0.0.0.0 \
  --port 30000
```

---

## 13. 调优建议

### 13.1 DeepEP Buffer 约束

**关键约束**: `max-running-requests × MTP_draft_tokens ≤ SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`

违反此约束会导致 DeepEP dispatch buffer 溢出。调优时需同步调整:
- `--cuda-graph-max-bs`
- `--max-running-requests`
- `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`

### 13.2 MTP (投机解码) 选择

| 场景 | 推荐配置 | 原因 |
|------|---------|------|
| bs=1 交互 | steps=3, draft=4 | 最大单请求加速 |
| 中等并发 | steps=1, draft=2 | 温和MTP，减少吞吐损失 |
| 高并发批处理 | 禁用 | verify开销 > 收益 |

### 13.3 内存优化

- `--mem-fraction-static`: 默认由hook设置，OOM时降低
- Pro模型在H200 FP4: 建议 `--mem-fraction-static 0.83` (low-latency) 或 `0.88` (其他)
- H100 Pro: `--cuda-graph-max-bs 8 --max-running-requests 32` 控制内存峰值

### 13.4 V4 自动应用的默认值

`deepseek_v4_hook.py` 自动设置:
- `attention_backend = "dsv4"`
- `page_size = 256`
- `max_running_requests = 256`
- `kv_cache_dtype = "fp8_e4m3"` (唯一支持的dtype)
- `swa_full_tokens_ratio = 0.1`
- 自动启用 Spec V2 (当使用EAGLE时)
