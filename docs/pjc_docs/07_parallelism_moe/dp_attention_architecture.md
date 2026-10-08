# SGLang DP Attention (Data Parallel Attention) 技术架构

## 1. 概述

DP Attention（Data Parallel Attention）是 SGLang 中一种高级并行策略，核心思想是：**对 Attention 层采用数据并行（DP），对 MLP/MoE 层采用张量并行（TP）或专家并行（EP）**。这种设计主要服务于 MoE（Mixture of Experts）架构的模型（如 DeepSeek V2/V3），通过消除 Attention 层的跨 GPU 通信，大幅降低 Decode 阶段的延迟。

### 1.1 核心动机

传统 Tensor Parallelism（TP）在每一层都需要 AllReduce 通信：

```
传统 TP（所有层都做 AllReduce）:
  Attention → AllReduce → MLP → AllReduce → 下一层
  每层 2 次 AllReduce，延迟随 TP 增大而线性增加
```

DP Attention 的策略是：

```
DP Attention（Attention 无通信，仅 MLP 前后通信）:
  Attention（独立/DP）→ Gather → MLP/MoE（TP/EP）→ Scatter → 下一层
  Attention 零通信，MLP 层前后各一次 Gather/Scatter
```

对于 MoE 模型，MLP 层本身需要 All-to-All 通信来做 Expert 路由，因此 Gather/Scatter 的通信开销可以与 MoE dispatch/combine 合并，使得 DP Attention 的额外通信代价极低。

### 1.2 适用场景

| 场景 | 推荐 | 原因 |
|------|------|------|
| DeepSeek V2/V3 (MoE + MLA) | 强烈推荐 | MLA 无需 KV 头并行，MoE 本身需要 A2A |
| 大规模 MoE 模型 | 推荐 | EP 通信可与 DP Gather/Scatter 合并 |
| Dense 模型（Llama 等） | 不推荐 | MLP 层的 Gather/Scatter 是额外开销 |
| Decode 延迟敏感 | 推荐 | 消除 Attention 层的 AllReduce |

---

## 2. 启动参数与配置

### 2.1 核心启动参数

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --dp 8 \
    --enable-dp-attention \
    --moe-dense-tp-size 1 \
    --ep-size 8 \
    --moe-a2a-backend deepep
```

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--tp-size` / `--tp` | 总的张量并行 GPU 数 | 1 |
| `--dp-size` / `--dp` | DP Attention 的并行度 | 1 |
| `--enable-dp-attention` | 启用 DP Attention | False |
| `--moe-dense-tp-size` | Dense 层（非 MoE）的 TP 大小 | None（=tp_size） |
| `--ep-size` | Expert 并行度 | 1 |
| `--moe-a2a-backend` | MoE All-to-All 通信后端 | "none" |
| `--attn-cp-size` | Attention Context Parallelism 大小 | 1 |

### 2.2 参数约束

```
# 来自 server_args.py _handle_data_parallelism()
tp_size % dp_size == 0          # tp_size 必须是 dp_size 的整数倍
attn_tp_size = tp_size / dp_size  # 每个 DP 组内的 Attention TP 大小

# 当 dp_size == 1 时，自动关闭 dp_attention
# chunked_prefill_size 会自动除以 dp_size
# schedule_conservativeness 会乘以 0.3
```

### 2.3 典型部署配置

**单节点 8 GPU 全 DP Attention（最常用，DeepSeek V3）：**
```bash
--tp 8 --dp 8 --enable-dp-attention --moe-dense-tp-size 1 --moe-a2a-backend deepep
# attn_tp_size = 8/8 = 1  → 每 GPU 独立做 Attention
# ep_size = 8             → 每 GPU 拥有 1/8 的 Experts
```

**单节点 8 GPU 部分 DP Attention：**
```bash
--tp 8 --dp 4 --enable-dp-attention
# attn_tp_size = 8/4 = 2  → 每 2 个 GPU 做 Attention TP
# 4 个 DP 组，每组 2 GPU
```

**多节点 16 GPU：**
```bash
--tp 16 --dp 16 --enable-dp-attention --moe-dense-tp-size 1 --moe-a2a-backend deepep
# attn_tp_size = 1, ep_size = 16
```

---

## 3. 整体部署架构

### 3.1 进程拓扑

启用 `--enable-dp-attention` 后的进程结构（以 `tp=8, dp=8` 为例）：

```
                        ┌─────────────────────┐
                        │   TokenizerManager   │
                        │   (单进程)            │
                        └──────────┬──────────┘
                                   │ ZMQ PUSH
                                   ▼
                        ┌─────────────────────┐
                        │ DataParallelController│
                        │   (单进程，请求路由)    │
                        └──┬──┬──┬──┬──┬──┬──┬┘
                           │  │  │  │  │  │  │
              ZMQ分发到 dp_rank=0..7 的 leader scheduler
                           │  │  │  │  │  │  │
               ┌───────────┘  │  │  │  │  │  └───────────┐
               ▼              ▼  ▼  ▼  ▼  ▼              ▼
        ┌──────────┐   ┌──────────┐  ...  ┌──────────┐
        │Scheduler │   │Scheduler │       │Scheduler │
        │  GPU 0   │   │  GPU 1   │       │  GPU 7   │
        │dp_rank=0 │   │dp_rank=1 │       │dp_rank=7 │
        │attn_tp=0 │   │attn_tp=0 │       │attn_tp=0 │
        └────┬─────┘   └────┬─────┘       └────┬─────┘
             │               │                   │
             └───────────────┴───────────────────┘
                    共享一个 NCCL World Group
                    [GPU0, GPU1, ..., GPU7]
```

**关键要点：**

1. **每个 GPU 一个 Scheduler 进程**：不像传统 DP 那样每个 DP 组只有一个 Scheduler，DP Attention 模式下每个 GPU 都运行独立的 Scheduler
2. **单一 NCCL 通信域**：所有 GPU 在一个 NCCL World 中，子组在其中划分
3. **仅 leader 接收请求**：只有 `attn_tp_rank == 0` 的 Scheduler 从 DataParallelController 接收请求
4. **NCCL all_gather 同步**：非 leader Scheduler 通过 NCCL all_gather 获取 batch 信息

### 3.2 对比传统 DP 的进程结构

**传统 DP（dp=2, tp=4）：两个独立的 NCCL 组**

```
DataParallelController
       /            \
      ▼              ▼
DP Group 0         DP Group 1
[g0,g1,g2,g3]     [g4,g5,g6,g7]
独立 NCCL 组       独立 NCCL 组
Scheduler 0       Scheduler 1
```

**DP Attention（dp=8, tp=8）：单一 NCCL 组，每 GPU 独立调度**

```
DataParallelController
  |  |  |  |  |  |  |  |
  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
 g0 g1 g2 g3 g4 g5 g6 g7
  └──────────────────────┘
      单一 NCCL 组
   8 个独立 Scheduler
```

### 3.3 请求路由

```
用户请求 → TokenizerManager → DataParallelController → 选择 dp_rank → Scheduler(dp_rank)
```

DataParallelController 支持多种负载均衡策略：
- `round_robin`：轮询分发
- `shortest_queue`：最短队列优先
- `resources_per_random`：基于资源的随机分发
- `auto`：自动选择

---

## 4. 核心机制详解

### 4.1 Rank 布局

Rank 在 TP group 内按 `(dp, cp, tp)` 三维排列，tp 维度变化最快：

```python
# dp_attention.py:247
tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank) * attn_tp_size + attn_tp_rank
```

**示例：tp_size=8, dp_size=4, attn_cp_size=1**

```
attn_tp_size = 8 / 4 / 1 = 2

tp_rank:  0    1    2    3    4    5    6    7
dp_rank:  0    0    1    1    2    2    3    3
atp_rank: 0    1    0    1    0    1    0    1

Attention TP 子组：[g0,g1] [g2,g3] [g4,g5] [g6,g7]
DP 组:            dp=0    dp=1    dp=2    dp=3
```

**示例：tp_size=8, dp_size=8（全 DP）**

```
attn_tp_size = 8 / 8 = 1  → 每 GPU 独立做 Attention

tp_rank:  0  1  2  3  4  5  6  7
dp_rank:  0  1  2  3  4  5  6  7
atp_rank: 0  0  0  0  0  0  0  0
```

### 4.2 NCCL 通信子组

`parallel_state.py:initialize_model_parallel()` 创建以下进程组：

| 进程组 | 大小 | 用途 |
|--------|------|------|
| `_TP` | tp_size | 完整 TP 组（所有 GPU） |
| `_ATTN_TP` | attn_tp_size | Attention 层的 TP 通信 |
| `_ATTN_CP` | attn_cp_size | Context Parallelism |
| `_MOE_EP` | moe_ep_size | Expert Parallelism（A2A） |
| `_MOE_TP` | moe_tp_size | MoE 层内的 TP |
| `_MOE_DP` | moe_dp_size | MoE 数据并行 |

当 `moe_dense_tp_size=1` 时：
- `_ATTN_TP` 大小为 1（无 Attention TP 通信）
- Dense MLP 层也以 DP 方式运行（`enable_moe_dense_fully_dp() == True`）

### 4.3 ScatterMode 状态转换

`communicator.py` 中定义了三种数据分布模式：

```python
class ScatterMode(Enum):
    SCATTERED = auto()      # 每 GPU 只有自己的数据分片
    TP_ATTN_FULL = auto()   # Attention TP 组内数据完整
    FULL = auto()           # 所有 GPU 都有全部数据
```

**示例（TP=4, DP=2, 处理请求 a,b,c,d）：**

```
模型输入/输出: [ab, ab, cd, cd]   # TP_ATTN_FULL

SCATTERED:     [a,  b,  c,  d]    # 每 GPU 只有 1 个请求
TP_ATTN_FULL:  [ab, ab, cd, cd]   # DP组0 的两个 GPU 都有 ab
FULL:          [abcd, abcd, abcd, abcd]  # 所有 GPU 都有所有请求
```

### 4.4 每层数据流转

```
Layer Input (SCATTERED 或 TP_ATTN_FULL)
        │
        ▼
   prepare_attn()            # SCATTERED → TP_ATTN_FULL（如果需要）
        │
        ▼
   ┌─────────────────┐
   │   Attention 层    │      # 在 TP_ATTN_FULL 模式下运行
   │   (数据并行/独立)  │      # 每个 DP 组独立计算
   └─────────────────┘
        │
        ▼
   prepare_mlp()             # 包含以下步骤：
   │  1. Attention 输出 AllReduce（在 attn_tp_group 内）
   │  2. LayerNorm（在 local 数据上）
   │  3. dp_gather_partial()（收集所有 DP 分片）
        │
        ▼
   ┌─────────────────┐
   │   MLP / MoE 层   │      # 在 FULL 或 SCATTERED 模式下运行
   │   (TP / EP)      │      # 所有 GPU 协作
   └─────────────────┘
        │
        ▼
   postprocess_layer()       # dp_scatter() 或 dp_reduce_scatter()
   │  提取本 GPU 的数据分片
        │
        ▼
Layer Output (SCATTERED 或 TP_ATTN_FULL)
```

### 4.5 MLP 同步机制

由于每个 GPU 的 Scheduler 独立调度，在执行 MLP 层前需要同步各 GPU 的 batch 信息：

```python
# scheduler_dp_attn_mixin.py
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int
    num_tokens: int           # 本 GPU 的 token 数
    can_cuda_graph: bool      # 是否可以使用 CUDA Graph
    is_extend_in_batch: bool  # 是否包含 extend 操作
    global_num_tokens: list   # 所有 GPU 的 token 数（all_gather 后）
```

**同步流程：**

```
Scheduler 0 (token=32)  ─┐
Scheduler 1 (token=28)  ─┤
Scheduler 2 (token=30)  ─┤── all_gather ──→ global_num_tokens = [32, 28, 30, ...]
...                      ─┤                  can_cuda_graph = min(all)
Scheduler 7 (token=25)  ─┘                  is_extend_in_batch = max(all)
```

如果某个 Scheduler 当前没有 batch（空闲），会创建一个 `idle_batch`，以便在 MLP 层仍然参与 NCCL 集合通信。

### 4.6 Gather/Scatter 通信原语

**dp_gather（Attention→MLP 前的收集）：**

两种模式（由 `DpPaddingMode` 决定）：

| 模式 | 通信方式 | 适用场景 |
|------|---------|---------|
| `MAX_LEN` | `all_gather_into_tensor` | Decode（各 GPU token 数接近），支持对称内存优化 |
| `SUM_LEN` | `all_reduce`（先 memcpy 到全局 buffer 的对应位置） | Extend/Prefill（各 GPU token 数差异大） |

选择逻辑：
```python
# dp_attention.py
if is_extend_in_batch and dp_size > 1:
    return SUM_LEN  # 避免不均匀 token 分布的 padding 开销
if sum_len * 2 >= max_len * dp_size:
    return MAX_LEN  # 通信代价相当时优先 MAX_LEN
else:
    return SUM_LEN
```

**dp_scatter（MLP→下一层 Attention 前的分发）：**

```python
# 使用 Triton kernel 做高效 memcpy
def dp_scatter(local_tokens, global_tokens, forward_batch):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
    memcpy_triton(local_tokens, global_tokens, 0, local_start_pos, local_num_tokens, True)
```

或使用 `dp_reduce_scatter_tensor`（当满足 MAX_LEN 模式且 allow_reduce_scatter 时）。

---

## 5. MoE Expert Parallelism 集成

### 5.1 EP + DP Attention 协作模式

DP Attention 与 MoE EP 天然互补：

```
┌─────────────────────────────────────────────────┐
│  Attention 层：每 GPU 独立计算（Data Parallel）    │
│  GPU0: Attn(batch_a)   GPU1: Attn(batch_b) ...  │
├─────────────────────────────────────────────────┤
│  dp_gather: 收集所有 GPU 的 hidden states         │
├─────────────────────────────────────────────────┤
│  MoE Dispatch: Token → Expert 路由               │
│  All-to-All：每个 token 发送到拥有目标 expert 的 GPU│
├─────────────────────────────────────────────────┤
│  Expert 计算：每 GPU 计算本地 experts              │
│  GPU0: experts[0:31]  GPU1: experts[32:63] ...   │
├─────────────────────────────────────────────────┤
│  MoE Combine: All-to-All 将结果返回源 GPU         │
├─────────────────────────────────────────────────┤
│  dp_scatter: 分发回各 GPU 的 local batch          │
└─────────────────────────────────────────────────┘
```

### 5.2 MoE All-to-All 后端

| 后端 | 说明 | 特点 |
|------|------|------|
| `none` | 标准 TP 模式 | 无 A2A，Gather→TP MoE→Scatter |
| `deepep` | DeepEP | 支持 Normal 和 Low-Latency 模式，FP8 量化通信 |
| `mooncake` | Mooncake | 专用硬件传输 |
| `nixl` | Nixl | InfiniBand 支持 |
| `flashinfer` | FlashInfer | FlashInfer 集成的 A2A |
| `mori` | Mori | AMD GPU 支持 |
| `ascend_fuseep` | Ascend Fused EP | 华为 NPU 支持 |

### 5.3 DeepEP 通信细节

**Normal 模式（Prefill/Extend，高吞吐）：**
```
1. dispatch_a(): 可选 FP8 量化（128-block grouping）
2. dispatch_b(): get_dispatch_layout() → dispatch() [NCCL All-to-All]
3. Expert GEMM 计算
4. combine_b(): combine() [反向 All-to-All]
```

**Low-Latency 模式（Decode，低延迟）：**
```
1. low_latency_dispatch(): FP8/NVFP4 量化后直接 dispatch
2. Masked Expert GEMM（基于 expected_m）
3. low_latency_combine()
4. 支持与 GEMM 计算的异步 overlap
```

### 5.4 FP4 AllGather 优化

当条件满足时，StandardDispatcher 在 all_gather 前先量化到 FP4，减少 4x 通信量：

```python
# moe/utils.py
def should_use_flashinfer_cutlass_moe_fp4_allgather():
    return (
        get_moe_a2a_backend().is_none()
        and get_moe_runner_backend().is_flashinfer_cutlass()
        and is_dp_attention_enabled()
        and MOE_QUANTIZATION == "modelopt_fp4"
        and get_moe_expert_parallel_world_size() == get_attention_dp_size()
    )
```

通信优化：
```
原始: all_gather(BF16 hidden_states)  → 2 bytes/element
优化: fp4_quantize → all_gather(FP4)   → 0.5 bytes/element → 4x 带宽节省
```

---

## 6. DeepSeek V2/V3 模型层实现

### 6.1 Decoder Layer 结构

```python
# models/deepseek_v2.py
class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, ...):
        # 确定当前层的 scatter modes
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            is_layer_sparse=self.is_layer_sparse,        # 是否 MoE 层
            is_previous_layer_sparse=...,
            is_next_layer_sparse=...,
        )

        if self.is_layer_sparse:
            self.mlp = DeepseekV2MoE(...)   # MoE 层
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1   # Dense MLP 也 DP 运行
            self.mlp = DeepseekV2MLP(...)

        self.layer_communicator = LayerCommunicator(...)

    def forward(self, hidden_states, residual, forward_batch):
        # 1. 准备 Attention 输入
        hidden_states, residual = self.layer_communicator.prepare_attn(...)

        # 2. Attention（独立/DP）
        hidden_states = self.self_attn(hidden_states, forward_batch)

        # 3. 准备 MLP 输入（Gather）
        hidden_states, residual = self.layer_communicator.prepare_mlp(...)

        # 4. MLP/MoE
        hidden_states = self.mlp(hidden_states, forward_batch)

        # 5. 后处理（Scatter）
        hidden_states, residual = self.layer_communicator.postprocess_layer(...)

        return hidden_states, residual
```

### 6.2 LayerScatterModes 计算逻辑

```python
# communicator.py
class LayerScatterModes:
    @classmethod
    def _compute_mlp_mode(cls, context):
        if context.is_layer_sparse:
            if not get_moe_a2a_backend().is_none():
                return ScatterMode.SCATTERED    # A2A 后端自行处理通信
            else:
                return ScatterMode.FULL         # 标准方式需要 Gather 到 FULL
        else:
            if enable_moe_dense_fully_dp():
                return ScatterMode.SCATTERED    # Dense 层也 DP 运行
            else:
                return ScatterMode.FULL
```

---

## 7. 性能分析

### 7.1 通信量对比

以 8 GPU、hidden_size=7168、batch_size=256 为例：

**传统 TP（tp=8）每层通信：**
```
Attention AllReduce:  256 × 7168 × 2 bytes × 2 = 7.0 MB
MLP AllReduce:        256 × 7168 × 2 bytes × 2 = 7.0 MB
总计每层: 14.0 MB
```

**DP Attention（dp=8, tp=8）每层通信：**
```
Attention: 0 MB（无通信）
dp_gather:  256 × 7168 × 2 bytes = 3.5 MB（all_gather）
MoE A2A:   由路由决定（sparse，远小于 dense AllReduce）
dp_scatter: 256 × 7168 × 2 bytes = 3.5 MB（reduce_scatter）
总计每层: ~7 MB + sparse A2A
```

### 7.2 KV Cache 效率

DP Attention 下每个 GPU 只需存储 `1/dp_size` 的 KV Cache：

```
传统 TP=8:  每 GPU 存储全部请求的 KV Cache（按 head 分片）
DP Attn dp=8: 每 GPU 只存储 1/8 请求的完整 KV Cache

→ 每 GPU 可以服务更大的 batch 或更长的序列
```

### 7.3 Decode 延迟

DP Attention 的最大优势在 Decode 阶段：

```
传统 TP Decode: Attention(1 token) → AllReduce → MLP(1 token) → AllReduce
  → AllReduce 在小数据量上的启动开销（latency）成为瓶颈

DP Attention Decode: Attention(1 token, 独立) → Gather(dp_size tokens) → MoE → Scatter
  → Attention 零延迟
  → Gather/Scatter 在更大 batch 上分摊启动开销
```

---

## 8. 关键源文件索引

| 文件 | 路径 | 职责 |
|------|------|------|
| dp_attention.py | `python/sglang/srt/layers/dp_attention.py` | DP Attention 核心：rank 计算、gather/scatter 原语、buffer 管理 |
| communicator.py | `python/sglang/srt/layers/communicator.py` | 层级通信编排：ScatterMode 转换、LayerCommunicator |
| parallel_state.py | `python/sglang/srt/distributed/parallel_state.py` | NCCL 进程组创建（TP、ATTN_TP、MOE_EP 等） |
| server_args.py | `python/sglang/srt/server_args.py` | 启动参数解析、约束校验 |
| scheduler_dp_attn_mixin.py | `python/sglang/srt/managers/scheduler_dp_attn_mixin.py` | Scheduler DP Attention 混入：MLPSyncBatchInfo、batch 同步 |
| data_parallel_controller.py | `python/sglang/srt/managers/data_parallel_controller.py` | DataParallelController：进程拓扑创建、请求路由 |
| engine.py | `python/sglang/srt/entrypoints/engine.py` | 引擎入口：进程启动编排 |
| model_runner.py | `python/sglang/srt/model_executor/model_runner.py` | 模型执行器：分布式初始化 |
| deepseek_v2.py | `python/sglang/srt/models/deepseek_v2.py` | DeepSeek V2/V3 模型实现 |
| layer.py (fused_moe) | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | FusedMoE 层、dispatcher 工厂 |
| deepep.py | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` | DeepEP All-to-All dispatcher |
| standard.py | `python/sglang/srt/layers/moe/token_dispatcher/standard.py` | Standard dispatcher（含 FP4 AllGather 优化） |
| forward_batch_info.py | `python/sglang/srt/model_executor/forward_batch_info.py` | ForwardBatch：dp_padding_mode、global_num_tokens |

---

## 9. 架构图：完整数据流

```
                            8 GPU DP Attention 部署 (dp=8, tp=8)

      ┌──────────┐  ┌──────────┐  ┌──────────┐        ┌──────────┐
      │  GPU 0   │  │  GPU 1   │  │  GPU 2   │  ...   │  GPU 7   │
      │ batch_a  │  │ batch_b  │  │ batch_c  │        │ batch_h  │
      │ Sched 0  │  │ Sched 1  │  │ Sched 2  │        │ Sched 7  │
      └────┬─────┘  └────┬─────┘  └────┬─────┘        └────┬─────┘
           │              │              │                    │
           ▼              ▼              ▼                    ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐        ┌──────────┐
      │ LayerNorm│  │ LayerNorm│  │ LayerNorm│        │ LayerNorm│
      │  (local) │  │  (local) │  │  (local) │        │  (local) │
      └────┬─────┘  └────┬─────┘  └────┬─────┘        └────┬─────┘
           │              │              │                    │
           ▼              ▼              ▼                    ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐        ┌──────────┐
      │MLA Attn  │  │MLA Attn  │  │MLA Attn  │        │MLA Attn  │
      │ (独立)   │  │ (独立)   │  │ (独立)   │        │ (独立)   │
      │ batch_a  │  │ batch_b  │  │ batch_c  │        │ batch_h  │
      └────┬─────┘  └────┬─────┘  └────┬─────┘        └────┬─────┘
           │              │              │                    │
           ▼              ▼              ▼                    ▼
      ┌────────────────────────────────────────────────────────────┐
      │              dp_gather (all_gather / all_reduce)           │
      │    LayerNorm(local) → 收集所有 DP rank 的 hidden states     │
      │    local: [a] → global: [a, b, c, d, e, f, g, h]          │
      └───────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────────┐
      │          MoE Expert Dispatch (All-to-All / DeepEP)         │
      │     每个 token 路由到 top-K experts 所在的 GPU               │
      │     GPU0: experts[0:31], GPU1: experts[32:63], ...         │
      └───────────────────────────┬────────────────────────────────┘
                                  │
           ┌──────────┬───────────┼───────────┬──────────┐
           ▼          ▼           ▼           ▼          ▼
      ┌──────────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────┐
      │ Expert   │  │Expert│  │Expert│  │Expert│  │ Expert   │
      │ Compute  │  │Compu.│  │Compu.│  │Compu.│  │ Compute  │
      │ 0-31     │  │32-63 │  │64-95 │  │...   │  │ 224-255  │
      └────┬─────┘  └──┬───┘  └──┬───┘  └──┬───┘  └────┬─────┘
           │            │         │         │            │
           └────────────┴─────────┴─────────┴────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────────┐
      │           MoE Expert Combine (All-to-All 返回)              │
      │           Apply topk_weights, 加权求和                      │
      └───────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────────┐
      │              dp_scatter (reduce_scatter / memcpy)           │
      │    global: [a,b,c,...,h] → local: 每 GPU 取回自己的分片      │
      └───────────────────────────┬────────────────────────────────┘
           │              │              │                    │
           ▼              ▼              ▼                    ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐        ┌──────────┐
      │  下一层   │  │  下一层   │  │  下一层   │        │  下一层   │
      │  GPU 0   │  │  GPU 1   │  │  GPU 2   │        │  GPU 7   │
      │ batch_a  │  │ batch_b  │  │ batch_c  │        │ batch_h  │
      └──────────┘  └──────────┘  └──────────┘        └──────────┘
```

---

## 10. 设计原则总结

1. **Attention 零通信**：`moe_dense_tp_size=1` 时，Attention 层完全独立计算，KV Cache 本地存储，消除延迟瓶颈
2. **EP 替代 TP 用于 MoE**：每个 GPU 拥有完整的 expert 子集，通过 All-to-All 路由 token，而非 TP 方式切分每个 expert
3. **Gather-before-MoE, Scatter-after-MoE**：LayerCommunicator 管理数据重分布，MoE 前 gather，MoE 后 scatter
4. **DeepEP 内化通信**：使用 DeepEP 等后端时，token dispatch 本身就是 All-to-All，LayerCommunicator 转为 SCATTERED 模式让 dispatcher 处理通信
5. **通信前量化**：DeepEP 使用 FP8 量化、StandardDispatcher 支持 FP4 量化，在通信前压缩数据，节省 2-4x 带宽
6. **NCCL 同步保持一致性**：所有 GPU 通过 MLPSyncBatchInfo 的 all_gather 保持 batch 信息一致，空闲 GPU 创建 idle_batch 参与集合操作
