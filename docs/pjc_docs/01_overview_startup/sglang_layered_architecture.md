# SGLang 分层架构全景梳理

> 本文从"一个请求如何从 HTTP 进来、走完整个系统、再把 token 吐出去"的视角，把 SGLang 服务端（`python/sglang/srt/`）拆成 **6 个纵向主层 + 4 个横切层**，逐层说明它的职责、关键文件、以及每层内部的核心技术点。
>
> 阅读方式：先看第 1 节的全景图建立整体印象，再按需下钻到各层。每个技术点都标注了源码文件与行号，方便定位。

---

## 1. 全景：请求的一生 + 分层地图

### 1.1 一个请求的端到端路径

```
                         ┌──────────────────────────────────────────────┐
   HTTP / gRPC 请求  ──▶ │ L0 接入层 entrypoints                         │
                         │  http_server / openai / grpc / engine         │
                         └───────────────┬──────────────────────────────┘
                                         │ (ZMQ IPC: tokenized req)
                         ┌───────────────▼──────────────────────────────┐
                         │ L1 管理调度层 managers                        │
                         │  TokenizerManager → Scheduler → Detokenizer   │
                         │  · 排队 waiting_queue                          │
                         │  · get_next_batch_to_run 决策                  │
                         │  · schedule_batch / schedule_policy            │
                         └───────────────┬──────────────────────────────┘
                                         │ (ForwardBatch)
                         ┌───────────────▼──────────────────────────────┐
                         │ L2 模型执行层 model_executor                  │
                         │  ModelRunner + ForwardBatch + CUDA Graph      │
                         └───────────────┬──────────────────────────────┘
                                         │ (调用算子)
                ┌────────────────────────▼───────────────────────────────┐
                │ L3 计算核心层 layers                                    │
                │  Attention backends / MoE / Quantization / Sampler     │
                └───────────┬─────────────────────────┬──────────────────┘
                            │ 读写 KV                  │
                ┌───────────▼──────────┐   ┌───────────▼──────────────────┐
                │ L4 内存缓存层        │   │ L5 分布式并行层               │
                │ mem_cache            │   │ distributed / parallel_state  │
                │ KV pool / radix tree │   │ TP / DP / EP / PP / CP        │
                └──────────────────────┘   └───────────────────────────────┘

  横切层（贯穿多层）：
   X1 投机解码 speculative   X2 PD 分离 disaggregation
   X3 多模态 multimodal      X4 量化 quantization
```

### 1.2 分层职责速查表

| 层 | 目录 | 一句话职责 | 进程/线程模型 |
|----|------|-----------|--------------|
| **L0 接入层** | `entrypoints/` | 协议解析、鉴权、SSE 流式返回 | HTTP server 进程 (asyncio) |
| **L1 管理调度层** | `managers/` | 分词、排队、组批、去分词 | TokenizerManager / Scheduler / Detokenizer 各独立进程，ZMQ 连接 |
| **L2 模型执行层** | `model_executor/` | 组装 ForwardBatch、跑 forward、CUDA Graph | Scheduler 进程内，双 CUDA Stream |
| **L3 计算核心层** | `layers/` | attention/MoE/量化/采样等算子 | GPU kernel |
| **L4 内存缓存层** | `mem_cache/` | KV cache 分配、前缀复用、多级卸载 | Scheduler 进程内 |
| **L5 分布式并行层** | `distributed/` | 进程组管理、集合通信 | 每 GPU 一个 TP rank 进程 |
| **X1 投机解码** | `speculative/` | draft→verify→extend | Scheduler 进程内 |
| **X2 PD 分离** | `disaggregation/` | prefill/decode 拆实例 + KV 传输 | 独立集群 |
| **X3 多模态** | `multimodal/` | 图/视频/音频编码 | 处理器进程 + ViT |
| **X4 量化** | `layers/quantization/` | 权重/激活/KV 低精度 | 贯穿 L3 |

---

## 2. L0 · 客户端接入层（entrypoints）

**目录**：`python/sglang/srt/entrypoints/`

### 2.1 职责
把外部世界的多种协议统一翻译成 SGLang 内部的请求对象，并负责流式响应、鉴权、请求解压等边界处理。它是唯一直接面对用户的层，之后所有内部交互都走 ZMQ。

### 2.2 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| HTTP Server | `http_server.py` | 主入口，FastAPI/uvicorn，`/generate`、`/v1/*`、健康检查、warmup |
| OpenAI 兼容层 | `openai/` | `/v1/chat/completions`、`/v1/completions`、`/v1/embeddings` 协议适配 |
| Anthropic 兼容层 | `anthropic/` | Claude Messages API 适配 |
| Ollama 兼容层 | `ollama/` | Ollama 协议适配 |
| gRPC | `grpc_server.py` / `grpc_bridge.py` | gRPC 服务端 + 到内部的桥接 |
| Engine（离线） | `engine.py` / `EngineBase.py` | 进程内直接调用，不走 HTTP（离线批处理/RL） |
| Warmup | `warmup.py` | 启动后跑假请求预热 CUDA Graph、编译 kernel |

### 2.3 技术点

1. **协议归一化**：不同前端（OpenAI/Anthropic/Ollama/原生）最终都转成统一的 `GenerateReqInput` / `EmbeddingReqInput`（`managers/io_struct.py`），下游只认内部结构。
2. **流式响应 (SSE)**：`http_server.py` 用 asyncio 生成器把 Detokenizer 推回的增量 token 包装成 Server-Sent Events。
3. **HTTP 解压**：`http_request_decompression.py` 处理 gzip/deflate 请求体。
4. **Engine 双形态**：既能作为 HTTP 服务（`launch_server`），也能作为 Python 库进程内嵌入（`Engine`，RL 训练场景高频使用）。
5. **Warmup 钩子**：`warmup.py` 针对特定模型（如 Kimi-K3 用 448×448 图预热 ViT patch）定制预热逻辑。
6. **鉴权与 SSL**：`ssl_utils.py`、`request_headers.py` 处理 TLS 与请求头鉴权。

---

## 3. L1 · 请求管理与调度层（managers）

**目录**：`python/sglang/srt/managers/`（SGLang 的"大脑"，也是最复杂的一层）

### 3.1 三进程架构

SGLang 把 CPU 密集（分词）、调度决策、去分词拆成 **3 个独立进程**，用 ZMQ 串起来，避免 Python GIL 争抢，让 GPU 尽量不空转：

```
   TokenizerManager  ──ZMQ──▶  Scheduler  ──ZMQ──▶  DetokenizerManager
   (分词/接收请求)              (组批/前向)            (去分词/回传)
        ▲                                                    │
        └──────────────── HTTP server 进程 ◀─────────────────┘
```

| 组件 | 文件 | 职责 |
|------|------|------|
| TokenizerManager | `tokenizer_manager.py` | 分词、请求生命周期跟踪、meta_info 组装 |
| Scheduler | `scheduler.py` | 事件循环、组批决策、显存管理、前向调度 |
| DetokenizerManager | `detokenizer_manager.py` | 增量去分词、base64 编码特殊输出 |
| DataParallelController | `data_parallel_controller.py` | DP 场景下把请求路由到多个 Scheduler 副本 |

### 3.2 Scheduler 内部技术点（核心中的核心）

Scheduler 由多个 mixin 组合而成，是整个系统的调度中枢。

1. **集中式决策 `get_next_batch_to_run`**（`scheduler.py:2597`）：Prefill-First 三分支决策树——① 合并上一步 extend 结果到 running_batch → ② 尝试 `get_new_batch_prefill` 组 prefill 批 → ③ 失败才 `update_running_batch` 跑 decode。详见 [centralized_scheduling_loop_architecture.md](../02_scheduling/centralized_scheduling_loop_architecture.md)。
2. **两条事件循环**：`event_loop_normal`（同步）与 `event_loop_overlap`（重叠调度，默认开）。
3. **重叠调度 Overlap**：单线程 + 双 CUDA Stream，处理"上一批"结果时发射"当前批"forward，用 `FutureMap` 跨步中继 token。详见 [overlap_schedule_architecture.md](../02_scheduling/overlap_schedule_architecture.md)。
4. **三大队列**：`waiting_queue`（待调度）、`running_batch`（执行中）、`retracted_queue`（因显存压力被抢占）。
5. **调度策略**（`schedule_policy.py`）：`lpm`（最长前缀匹配，缓存友好）、`fcfs`、`lof`（最长输出优先）、`dfs-weight`、`routing-key`。
6. **显存自适应**：`new_token_ratio` 动态估算每步生成 token 数；`check_decode_mem()` 预判下一步 decode 显存是否够。
7. **抢占/退避 Retraction**（`schedule_batch.py:retract_decode()`）：OOM 时按 `(-output_length, -input_length)` 排序踢请求回 `waiting_queue`（非 abort），显存恢复后 `resume_retracted_reqs()` 复原。
8. **Chunked Prefill**：长 prompt 切块，`self.chunked_req` 作为分块进度锚点，必须推进否则 KV 泄漏。
9. **Prefill 延迟器**（`prefill_delayer.py` / `min_free_slots_delayer.py`）：DP 场景下跨 rank 协商，消除空泡。详见 [prefill_delayer_architecture.md](../02_scheduling/prefill_delayer_architecture.md)。
10. **PP 支持**（`scheduler_pp_mixin.py`）：流水线并行的 micro-batch 编排。

### 3.3 请求状态机

请求在 `schedule_batch.py:Req` 中经历 `RequestStage` 多阶段：Tokenize → PrefillWaiting → PrefillForward（prefill 侧）；DecodePrepare → DecodeBootstrap → DecodeWaiting（decode 侧，PD 分离时）。

---

## 4. L2 · 模型执行层（model_executor）

**目录**：`python/sglang/srt/model_executor/`

### 4.1 职责
把 Scheduler 交来的一批请求（`ScheduleBatch`）转换成张量形式的 `ForwardBatch`，驱动模型 forward，并管理 CUDA Graph 的捕获与重放。

### 4.2 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| ModelRunner | `model_runner.py` | 加载模型、初始化 KV pool/attention backend、执行 forward（**frozen 核心文件**，改动前必读 `large-class-style` skill）|
| ForwardBatch | `forward_batch_info.py` | 一次前向所需的全部张量元数据（input_ids、positions、out_cache_loc、attn metadata 等）|
| ForwardContext | `forward_context.py` | 前向期间的进程内上下文 |
| CUDA Graph | `cuda_graph_*.py`、`runner/` | 图捕获、buffer 注册、重放 |
| Pool Configurator | `pool_configurator.py` | 计算 KV pool 容量 |

### 4.3 技术点

1. **ForwardMode**：`PREFILL` / `DECODE` / `EXTEND` / `IDLE` / `TARGET_VERIFY`（投机）等，决定走哪条计算路径。
2. **CUDA Graph 捕获与重放**：decode 阶段形状固定，捕获成图后重放消除 kernel launch 开销。`cuda_graph_buffer_registry.py` 管理静态 buffer；`breakable_cuda_graph/` 支持可打断的图。
3. **多硬件后端 runner**：`runner_backend/` 支持不同硬件（CUDA/HIP/NPU/XPU），`mindspore_runner.py`、`cpu_graph_runner.py` 是特化实现。
4. **DeepSeek MHA mixin**：`forward_batch_deepseek_mha_mixin.py` 针对 DeepSeek 系列 MLA 的特化元数据准备。
5. **图内共享输出**：`graph_shared_output.py` 让多个捕获图复用输出 buffer 省显存。
6. **Hook 管理**：`hook_manager.py` 支持模型级的启动钩子（如 DSV4/Kimi-K3 强制参数）。

---

## 5. L3 · 计算核心层（layers）

**目录**：`python/sglang/srt/layers/`（算子的家）

这一层是真正跑数值计算的地方，可再细分为 4 个子层。

### 5.1 注意力子层（`layers/attention/`）

**技术点：多 backend 可插拔**。每个 backend 实现统一接口 `base_attn_backend.py`，按模型/硬件/场景选择：

| Backend | 文件 | 适用 |
|---------|------|------|
| FlashAttention v3 | `flashattention_backend.py` | 通用 MHA/GQA 主力 |
| FlashInfer | `flashinfer_backend.py` / `flashinfer_mla_backend.py` | 树形投机 topk>1、MLA |
| FlashMLA | `flashmla_backend.py` | DeepSeek MLA |
| Triton | `triton` 系列 | 无 CUDA 依赖回退 |
| TRT-LLM | `trtllm_mha_backend.py` / `trtllm_mla_backend.py` | NVIDIA 优化 |
| 混合 | `hybrid_attn_backend.py` / `hybrid_linear_attn_backend.py` | SWA+全注意力 / 线性注意力混合模型 |
| DSV4 | `dsv4_backend/` | DeepSeek V4 三档压缩注意力 |
| NSA | `nsa_backend.py` | Native Sparse Attention |
| 硬件特化 | `ascend_backend.py` / `aiter_backend.py` / `intel_amx_backend.py` / `wave_backend.py` | NPU/ROCm/XPU |

其它技术点：`radix_attention.py`（attention 与 radix cache 的接口）、`merge_state.py`（CP/DCP 的 LSE 在线 softmax 合并）、`vision.py`（ViT 注意力）。

### 5.2 MoE 子层（`layers/moe/`）

**技术点：EP 下的 grouped GEMM + all-to-all 通信**。

1. **TopK 路由**（`topk.py`）：`select_experts` 选专家，也是 R3（Return Routed Experts）的唯一抓取点。
2. **Token Dispatcher**（`token_dispatcher/`）：DeepEP 的 all-to-all dispatch/combine，NORMAL（高吞吐 prefill）vs LOW_LATENCY（低延迟 decode）双模式。详见 [deepep_dispatch_combine_architecture.md](../07_parallelism_moe/deepep_dispatch_combine_architecture.md)。
3. **MegaMoE**（`mega_moe.py`）：Blackwell 上把 dispatch+GEMM+SwiGLU+combine 融进单个持续驻留 megakernel。详见 [megamoe_architecture.md](../07_parallelism_moe/megamoe_architecture.md)。
4. **多后端 grouped GEMM**：`fused_moe_triton/`、`cutlass_moe.py`、`flashinfer_cutedsl_moe.py`、`ep_moe/`、`moe_runner/`。
5. **Router**（`router.py`）：路由打分、noaux_tc、分组 topk。

### 5.3 量化子层（`layers/quantization/`）

**技术点：权重/激活/KV 三个维度独立低精度**（详见第 11 节横切层）。文件覆盖 FP8（`fp8.py`）、FP4/MXFP4（`fp4.py`/`mxfp4.py`）、AWQ、GPTQ、Marlin、W4AFP8、W8A8、compressed-tensors 等。

### 5.4 基础算子子层

`linear.py`（TP 切分的线性层）、`layernorm.py`（含 fused RMSNorm）、`activation.py`（SwiGLU 等）、`rotary_embedding/`（RoPE）、`vocab_parallel_embedding.py`、`logits_processor.py`、`sampler.py`（采样）、`communicator.py`（层间 all-reduce/all-gather 编排）、`dp_attention.py`（DP attention）、`cp/`+`dcp/`（上下文并行，详见 [context_parallel_architecture.md](../07_parallelism_moe/context_parallel_architecture.md)）。

---

## 6. L4 · 内存与缓存层（mem_cache）

**目录**：`python/sglang/srt/mem_cache/`（决定吞吐上限的关键层）

### 6.1 三级内存
```
   GPU (主) ──▶ Host/CPU (卸载) ──▶ Storage (HiCache 持久化)
```

### 6.2 核心抽象

| 抽象 | 文件 | 职责 |
|------|------|------|
| KV Pool | `memory_pool.py` | token→KV 映射、页分配 |
| Allocator | `allocator/` | 分配器（标准/分页/SWA/HiSparse）|
| Radix Cache | `radix_cache.py` | 基数树前缀缓存 + 驱逐 |
| SWA Radix | `swa_radix_cache.py` / `pure_swa_radix_cache.py` | 滑窗混合前缀缓存，详见 [swa_radix_cache_architecture.md](../03_cache_memory/swa_radix_cache_architecture.md) |
| Unified Cache | `unified_cache/` / `unified_radix_cache.py` | 统一 radix 树（FULL+SWA component）|
| Mamba Cache | `mamba_radix_cache.py` / `mamba_checkpoint_pool.py` | 线性注意力 recurrent state 池 |
| HiCache | `hiradix_cache.py` / `hicache_storage.py` / `pool_host/` | L1(GPU)↔L2(Host)↔L3(Storage) 多级缓存，详见 [hicache_usage_and_design.md](../03_cache_memory/hicache_usage_and_design.md) |
| Configurator | `kv_cache_configurator.py` / `pool_configurator.py` | 池容量与形状计算 |
| Registry | `registry.py` | 按模型类型选择 cache 工厂 |

### 6.3 技术点

1. **Radix Tree 前缀复用**：多请求共享公共 prompt 前缀的 KV，`match_prefix` 命中即复用，避免重算。
2. **驱逐策略**：LRU/LFU（`evict_policy.py`），`evictable_size()` 算可释放量，分配超限时触发。
3. **分页分配**：`alloc_paged_token_slots_extend()`，页对齐（如 DSV4 page_size=256）。
4. **SWA 双池双锁**：滑窗模型下 full/swa 两个子分配器 + tombstone 语义，`free_swa()` 只释放滑窗 KV 保留全注意力 KV。
5. **HiCache Layout**：`layer_first` / `page_first` / `page_first_direct` / `page_head` 多种 host 布局，配合 `kernel`/`direct` IO backend。
6. **KV 卸载**：decode OOM 时 `offload_kv_cache()` → host，`load_kv_cache()` 复原。
7. **VMM backing**：`kv_vmm_backing.py` 用 CUDA 虚拟内存管理动态扩缩池。

---

## 7. L5 · 分布式与并行层（distributed）

**目录**：`python/sglang/srt/distributed/` + `layers/parallel_state`

### 7.1 五种并行维度

| 并行 | 缩写 | 切什么 | 通信原语 | 关键文件 |
|------|------|--------|---------|---------|
| 张量并行 | TP | 权重按行/列切 | all-reduce / all-gather | `linear.py`、`parallel_state.py` |
| 数据并行 | DP | 请求批切 | 独立副本 | `data_parallel_controller.py` |
| 专家并行 | EP | MoE 专家切 | all-to-all (DeepEP) | `token_dispatcher/` |
| 流水线并行 | PP | 层切 | send/recv | `scheduler_pp_mixin.py` |
| 上下文并行 | CP/DCP | 序列/KV 切 | all-gather / LSE 合并 | `layers/cp/`、`layers/dcp/` |

### 7.2 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| ParallelState | `parallel_state.py` | 创建/管理所有进程组（_TP/_DP/_EP/_PP/_ATTN_CP/_DCP）|
| Communication Op | `communication_op.py` | all-reduce/all-gather/reduce-scatter 封装 |
| Device Communicators | `device_communicators/` | NCCL/gloo/自定义 all-reduce 后端 |
| DP Attention | `layers/dp_attention.py` | DP 场景下 attention 的 rank 内聚合 |

### 7.3 技术点

1. **进程组分层构造**：`parallel_state.py` 按 tp/dp/ep/pp/cp size 计算每个 rank 属于哪些组，跨步取 rank（如 `_ATTN_CP` 用 `range(st,en,attn_tp_size)`）。
2. **DP attention idle batch**：空 rank 插 idle batch 防 all-reduce 死锁（`maybe_prepare_mlp_sync_batch`）。
3. **EP==TP 约束**：MegaMoE 等 spanning backend 强制 EP 跨满 TP。
4. **CP zigzag 负载均衡**：因果注意力下把序列切 `2*cp_size` 块，靠前闲块 + 靠后忙块配对，唯一通信是 all-gather。
5. **DCP owner 规则**：decode 时 KV 按 `pos%dcp_size==rank` 切存储，局部 attn 后 LSE 在线 softmax 合并。
6. **自定义 all-reduce**：`device_communicators/` 含 NVLink 直连的低延迟 all-reduce（小张量优于 NCCL）。

---

## 8. X1 · 横切层：投机解码（speculative）

**目录**：`python/sglang/srt/speculative/`

**统一三阶段**：`draft → verify → draft_extend`，只有 V2 一条路径。详见 [speculative_decoding_overview.md](../05_speculative_decoding/speculative_decoding_overview.md)。

| 算法 | 文件 | 特点 |
|------|------|------|
| EAGLE/EAGLE3 | `eagle_worker_v2.py` | 主力，草稿层复用 target hidden |
| MTP (FROZEN_KV_MTP) | `frozen_kv_mtp_worker_v2.py` | MTP 层但不建 draft KV |
| DFLASH | `dflash_worker_v2.py` | target hidden 投影 KV，强制 steps=1 |
| STANDALONE | `standalone_worker_v2.py` | 独立草稿模型 |
| NGRAM | `ngram_worker.py` | 无模型 n-gram 检索 |
| DSpark | `dspark_components/` | 置信度调度 |

**技术点**：chain(topk==1) vs tree(topk>1) 由 topk 决定；`build_tree_kernel_efficient` 建树；拒绝采样 `coin*q<p`；bonus token 记账（`num_accept = num_correct + 1`）；两套 CUDA graph runner（draft-decode / draft-extend）；自适应步数（`adaptive_spec_params.py`）。

---

## 9. X2 · 横切层：PD 分离（disaggregation）

**目录**：`python/sglang/srt/disaggregation/`

把 prefill（计算密集）和 decode（显存/带宽密集）拆到不同 GPU 集群，各自独立扩缩。详见 [pd_disaggregation](docs/pjc_1/) 系列文档与 [dsv4_pd_disaggregation_request_lifecycle.md](../04_pd_disaggregation/dsv4_pd_disaggregation_request_lifecycle.md)。

| 组件 | 文件 | 职责 |
|------|------|------|
| Prefill 侧 | `prefill.py` | 三队列：Bootstrap→waiting→inflight |
| Decode 侧 | `decode.py` | 四阶段：Prealloc→Transfer→waiting→Running |
| KV 传输后端 | `mooncake/` / `nixl/` / `common/conn.py` | ZMQ/IB/Nixl/Mooncake |
| 卸载管理 | `decode_kvcache_offload_manager.py` | GPU→Host→Storage |

**技术点**：HTTP 服务发现 + ZMQ 反向推送握手；D 主动把 `dst_kv_indices` 推给 P；KVSender/KVReceiver 状态机（`all_reduce(MIN)` 保跨 rank 一致）；RDMA 单边写；PREBUILT 请求跳过 decode 侧 prefill。

---

## 10. X3 · 横切层：多模态（multimodal）

**目录**：`python/sglang/srt/multimodal/`

**技术点**：`processors/`（各模型的图/视频/音频预处理）；`encoder_preprocessing.py`（patch 切分）；ViT CUDA Graph（`vit_cuda_graph_runner.py`、`internvl_vit_cuda_graph_runner.py`、`kimi_k3_vit_cuda_graph_runner.py`）；`transport/`（多模态特征 cuda_ipc 传输）；`multimodal_cache.py`（编码结果缓存）；EVS（`evs/`，Efficient Video Sampling）。

---

## 11. X4 · 横切层：量化（quantization）

**贯穿 L3**，三个正交维度：

| 维度 | 典型格式 | 说明 |
|------|---------|------|
| 权重量化 | FP8 / FP4 / MXFP4 / AWQ(W4A16) / GPTQ / Marlin | 减少显存与带宽 |
| 激活量化 | W8A8 / W4AFP8 / per-token FP8 | 加速 GEMM |
| KV Cache 量化 | FP8 KV (`quantization/kv_cache.py`) | 减少 KV 显存，扩大 batch |

**技术点**：DeepGEMM/CUTLASS/Marlin 的量化 GEMM kernel；per-token/per-group/per-tensor 缩放粒度；MXFP4 微缩放（block=32，UE8M0 scale）；`modelopt_quant.py` 对接 NVIDIA ModelOpt。

---

## 12. 各层技术点总汇（速查）

| 层 | Top 技术点 |
|----|-----------|
| L0 接入 | 协议归一化、SSE 流式、Engine 双形态、Warmup 预热 |
| L1 调度 | 三进程 ZMQ 架构、Prefill-First 决策树、重叠调度双 Stream、抢占退避、Chunked Prefill、多调度策略、Prefill 延迟器 |
| L2 执行 | ForwardMode、CUDA Graph 捕获重放、多硬件 runner、共享输出 buffer |
| L3 计算 | 可插拔 attention backend、EP grouped GEMM + DeepEP、MegaMoE 融合、多后端 MoE、RoPE/RMSNorm 融合 |
| L4 缓存 | Radix 前缀复用、LRU/LFU 驱逐、分页分配、SWA 双池双锁、HiCache 多级、KV 卸载、VMM |
| L5 分布式 | TP/DP/EP/PP/CP 五维并行、进程组分层、DP idle batch、CP zigzag、DCP LSE 合并、自定义 all-reduce |
| X1 投机 | 三阶段 draft/verify/extend、chain vs tree、拒绝采样、bonus 记账、自适应步数 |
| X2 PD 分离 | 三队列/四阶段、ZMQ 握手、RDMA 单边写、KVSender 状态机、PREBUILT |
| X3 多模态 | 处理器管线、ViT CUDA Graph、cuda_ipc 传输、编码缓存 |
| X4 量化 | 权重/激活/KV 三维、量化 GEMM、微缩放、ModelOpt |

---

## 13. 关联文档索引

本文是**总览与地图**。各层的深度剖析见 `docs/pjc_1/` 下的专项文档：

- 调度：[centralized_scheduling_loop_architecture.md](../02_scheduling/centralized_scheduling_loop_architecture.md)、[overlap_schedule_architecture.md](../02_scheduling/overlap_schedule_architecture.md)、[prefill_delayer_architecture.md](../02_scheduling/prefill_delayer_architecture.md)
- 缓存：[swa_radix_cache_architecture.md](../03_cache_memory/swa_radix_cache_architecture.md)、[hicache_usage_and_design.md](../03_cache_memory/hicache_usage_and_design.md)
- 并行/通信：[context_parallel_architecture.md](../07_parallelism_moe/context_parallel_architecture.md)、[deepep_dispatch_combine_architecture.md](../07_parallelism_moe/deepep_dispatch_combine_architecture.md)、[megamoe_architecture.md](../07_parallelism_moe/megamoe_architecture.md)
- 投机：[speculative_decoding_overview.md](../05_speculative_decoding/speculative_decoding_overview.md)、[mtp_speculative_decoding_architecture.md](../05_speculative_decoding/mtp_speculative_decoding_architecture.md)
- 模型专项：[deepseek_v4_model_architecture.md](../09_models/deepseek_v4_model_architecture.md)、[deepseek_v4_cache_management.md](../09_models/deepseek_v4_cache_management.md)、[dsv4_pd_disaggregation_request_lifecycle.md](../04_pd_disaggregation/dsv4_pd_disaggregation_request_lifecycle.md)、[kimi_k3_support_scheme.md](../09_models/kimi_k3_support_scheme.md)
- 特性：[return_routed_experts_r3_architecture.md](../07_parallelism_moe/return_routed_experts_r3_architecture.md)
