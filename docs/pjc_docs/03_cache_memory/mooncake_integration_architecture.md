# SGLang Mooncake 集成方案全面解析

## 目录

1. [概述与定位](#1-概述与定位)
2. [Mooncake 在 SGLang 中的应用场景](#2-mooncake-在-sglang-中的应用场景)
3. [核心架构：Transfer Engine](#3-核心架构transfer-engine)
4. [PD 分离 KV Cache 传输（主场景）](#4-pd-分离-kv-cache-传输主场景)
5. [HiCache 分布式存储后端](#5-hicache-分布式存储后端)
6. [MoE 专家并行通信](#6-moe-专家并行通信)
7. [弹性 EP 故障恢复](#7-弹性-ep-故障恢复)
8. [配置参考](#8-配置参考)
9. [与其他传输后端对比](#9-与其他传输后端对比)
10. [故障处理与容错](#10-故障处理与容错)
11. [性能优化细节](#11-性能优化细节)
12. [部署实践](#12-部署实践)

---

## 1. 概述与定位

### 1.1 什么是 Mooncake

Mooncake 是一个高性能 RDMA（Remote Direct Memory Access）传输引擎，专为 GPU 集群中大规模数据传输设计。它提供：

- **GPUDirect RDMA**：GPU 显存间直接跨节点传输，绕过 CPU 和系统内存
- **零拷贝传输**：消除中间拷贝开销
- **批量传输 API**：单次调用完成多块不连续内存的传输
- **P2P 发现机制**：无需中心元数据服务器的点对点连接建立

### 1.2 为什么 SGLang 选择 Mooncake

在 LLM 推理服务中，KV Cache 数据量大（数十 GB 级别），跨节点传输延迟直接影响 TTFT（Time To First Token）。Mooncake 提供：

| 需求 | Mooncake 解决方案 |
|------|------------------|
| 低延迟跨节点 KV 传输 | GPUDirect RDMA，跳过 Host 内存 |
| 高吞吐批量传输 | `batch_transfer_sync_write` 一次性提交多个传输请求 |
| 灵活 TP 配置 | 支持 M:N TP 映射下的 head-slice 传输 |
| 分布式 KV 缓存池 | `MooncakeDistributedStore` 提供集群级 KV 存储 |
| MoE 专家通信 | 专用 EP Buffer 实现低延迟 all-to-all |

### 1.3 Mooncake 在 SGLang 中的角色

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SGLang 推理框架                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  ┌────────────┐ │
│  │ PD 分离     │  │ HiCache      │  │ MoE EP    │  │ Elastic EP │ │
│  │ KV Transfer │  │ L3 Storage   │  │ All2All   │  │ 故障恢复   │ │
│  └──────┬──────┘  └──────┬───────┘  └─────┬─────┘  └─────┬──────┘ │
│         │                │                 │              │         │
│         └────────────────┼─────────────────┼──────────────┘         │
│                          │                 │                         │
│              ┌───────────┴─────────────────┴────────────┐           │
│              │     Mooncake Transfer Engine (Singleton)  │           │
│              │     - RDMA / Ascend 传输                  │           │
│              │     - 内存注册/注销                        │           │
│              │     - 批量同步写                           │           │
│              └──────────────────────────────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Mooncake 是 SGLang PD 分离场景的默认传输后端**（`disaggregation_transfer_backend = "mooncake"`）。

---

## 2. Mooncake 在 SGLang 中的应用场景

### 场景总览

| 场景 | CLI 参数 | 作用 | 核心模块 |
|------|---------|------|---------|
| PD 分离 KV 传输 | `--disaggregation-transfer-backend mooncake` | Prefill→Decode 跨节点 KV Cache RDMA 传输 | `disaggregation/mooncake/conn.py` |
| HiCache L3 存储 | `--hicache-storage-backend mooncake` | 分布式 KV Cache 池（DRAM + SSD） | `mem_cache/storage/mooncake_store/` |
| MoE 专家并行 | `--moe-a2a-backend mooncake` | 低延迟 token dispatch all-to-all | `layers/moe/token_dispatcher/mooncake.py` |
| 弹性 EP 恢复 | `--elastic-ep-backend mooncake` | 专家权重跨节点备份/恢复 | `elastic_ep/elastic_ep.py` |
| Encoder 分离 | `--encoder-transfer-backend mooncake` | 多模态 Encoder 输出传输 | `multimodal_gen/runtime/disaggregation/` |

---

## 3. 核心架构：Transfer Engine

### 3.1 引擎单例设计

文件路径：`python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`

```python
# 全局单例
_mooncake_transfer_engine: Optional[MooncakeTransferEngine] = None

def init_mooncake_transfer_engine(hostname, gpu_id, ib_device):
    """每个 GPU 进程初始化一个共享引擎实例"""
    global _mooncake_transfer_engine
    if _mooncake_transfer_engine is not None:
        return _mooncake_transfer_engine
    _mooncake_transfer_engine = MooncakeTransferEngine(hostname, gpu_id, ib_device)
    return _mooncake_transfer_engine
```

### 3.2 MooncakeTransferEngine 类

```
┌──────────────────────────────────────────────────────────────┐
│                  MooncakeTransferEngine                        │
├──────────────────────────────────────────────────────────────┤
│ 属性:                                                         │
│   engine: mooncake.engine.TransferEngine  (底层 C++ 引擎)     │
│   hostname: str            (本机 IP)                          │
│   gpu_id: int              (GPU 设备索引)                      │
│   ib_device: str           (InfiniBand 设备名)                 │
│   session_id: str          (唯一标识 "host:rpc_port")          │
├──────────────────────────────────────────────────────────────┤
│ 核心方法:                                                     │
│   initialize(hostname, device_name)    → 初始化 RDMA/Ascend   │
│   register(ptr, length)               → 注册单块 GPU 内存     │
│   batch_register(ptrs, lengths)       → 批量注册内存          │
│   transfer_sync(session_id, ...)      → 单块同步 RDMA 写     │
│   batch_transfer_sync(session_id,...) → 批量同步 RDMA 写     │
│   send_probe(peer_session_id)         → 会话存活探测          │
│   deregister(ptr) / batch_deregister  → 注销内存              │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 初始化模式

引擎支持两种传输模式：

| 模式 | 条件 | 初始化参数 |
|------|------|-----------|
| **RDMA** (默认) | NVIDIA GPU + InfiniBand | `engine.initialize(hostname, "P2PHANDSHAKE", "rdma", device_name)` |
| **Ascend** | 华为 NPU | `engine.initialize(hostname:port:npu_X, "P2PHANDSHAKE", "ascend", device_name)` |

### 3.4 初始化时机

在 `ModelRunner.init_shared_mooncake_transfer_engine()` 中，满足以下任一条件时初始化：

1. PD 分离模式 + `disaggregation_transfer_backend == "mooncake"`
2. HiCache + `hicache_storage_backend == "mooncake"` + `SGLANG_HICACHE_MOONCAKE_REUSE_TE == True`
3. Encoder 分离 + `encoder_transfer_backend == "mooncake"`
4. 弹性 EP 备份启用

### 3.5 IB 设备配置

支持三种格式指定 InfiniBand 设备：

```python
# 格式 1：所有 GPU 使用相同设备
"mlx5_0,mlx5_1"

# 格式 2：Per-GPU JSON 映射
'{"0": "mlx5_0,mlx5_1", "1": "mlx5_2,mlx5_3"}'

# 格式 3：JSON 文件路径
"/path/to/ib_config.json"
```

---

## 4. PD 分离 KV Cache 传输（主场景）

### 4.1 整体架构

```
┌─────────────────────────────────────┐     ┌─────────────────────────────────────┐
│         PREFILL 实例                 │     │          DECODE 实例                 │
│                                     │     │                                     │
│  ┌─────────────┐                    │     │                    ┌─────────────┐  │
│  │ Scheduler   │                    │     │                    │ Scheduler   │  │
│  └──────┬──────┘                    │     │                    └──────┬──────┘  │
│         │                           │     │                           │         │
│  ┌──────┴──────┐  ┌──────────────┐  │     │  ┌──────────────┐ ┌──────┴──────┐  │
│  │MooncakeKV   │  │ Transfer     │  │     │  │ Transfer     │ │MooncakeKV   │  │
│  │  Manager    │──│ Worker(s)    │──│─────│──│ Worker(s)    │─│  Manager    │  │
│  │ (Prefill)   │  │ Thread Pool  │  │RDMA │  │ Thread Pool  │ │ (Decode)    │  │
│  └─────────────┘  └──────────────┘  │     │  └──────────────┘ └─────────────┘  │
│         │                           │     │                           │         │
│  ┌──────┴──────┐                    │     │                    ┌──────┴──────┐  │
│  │ KV Cache    │                    │     │                    │ KV Cache    │  │
│  │ (GPU)       │                    │     │                    │ (GPU)       │  │
│  └─────────────┘                    │     │                    └─────────────┘  │
└─────────────────────────────────────┘     └─────────────────────────────────────┘
            │                                                 │
            └──────── ZMQ (Bootstrap/Control) ───────────────┘
```

### 4.2 类继承关系

```
CommonKVManager (公共基类)
    │
    ├── MooncakeKVManager   ← Mooncake RDMA 实现
    ├── NixlKVManager       ← Nixl RDMA 实现
    └── MoriKVManager       ← Mori 实现

CommonKVSender / CommonKVReceiver
    │
    ├── MooncakeKVSender / MooncakeKVReceiver
    └── NixlKVSender / NixlKVReceiver
```

### 4.3 传输流程详解

#### 阶段 1：连接建立（Bootstrap）

```
Decode                                    Prefill
  │                                         │
  │  1. 注册 KV Buffer 指针到引擎            │
  │     (batch_register)                    │
  │                                         │
  │  2. 发送 KVArgsRegisterInfo             │
  │──── ZMQ: dst_kv_ptrs, session_id ──────►│
  │                                         │  3. 记录 decode 端 buffer 信息
  │                                         │     decode_kv_args_table[session_id]
  │                                         │
```

**KVArgsRegisterInfo** 包含：
- `mooncake_session_id`: Decode 端引擎唯一标识
- `dst_kv_ptrs`: Decode 端各层 KV Cache 的 GPU 内存地址
- `dst_aux_ptrs`: 辅助数据（如 position encoding）内存地址
- `dst_tp_rank` / `dst_attn_tp_size`: Decode 端的 TP 配置
- `dst_kv_item_len`: 每页 KV 数据的字节大小

#### 阶段 2：预分配通知

```
Decode                                    Prefill
  │                                         │
  │  4. 预分配 KV Cache 页                   │
  │                                         │
  │  5. 发送 TransferInfo                   │
  │──── ZMQ: room, dst_indices ────────────►│
  │                                         │  6. 状态: Bootstrapping → WaitingForInput
  │                                         │
```

**TransferInfo** 包含：
- `room`: 请求唯一标识（Bootstrap Room ID）
- `dst_kv_indices`: Decode 端预分配的 KV Cache 页索引
- `dst_aux_index`: 辅助数据的目标索引
- `required_dst_info_num`: 需要接收的目标 rank 数量

#### 阶段 3：KV Cache RDMA 传输

```
Prefill                                   Decode
  │                                         │
  │  7. Prefill forward 完成                 │
  │     产生 KV Cache                        │
  │                                         │
  │  8. 构建 TransferKVChunk                 │
  │     放入 transfer_queue                  │
  │                                         │
  │  9. transfer_worker 执行:                │
  │     - group_concurrent_contiguous()      │
  │     - batch_transfer_sync_write() ──────►│  10. GPU 内存直接写入
  │                                   RDMA   │
  │ 10. send_aux() (RDMA 或 TCP)     ──────►│  11. 辅助数据写入
  │                                         │
  │ 11. sync_status_to_decode_endpoint()    │
  │──── ZMQ: room + KVPoll.Success ────────►│  12. 标记传输完成
  │                                         │
```

### 4.4 三种传输路径

根据 Prefill 和 Decode 的 TP 配置，选择不同的传输路径：

#### 路径 A：同 TP 大小 (`send_kvcache`)

**条件**：`prefill_attn_tp_size == decode_attn_tp_size`，或 MLA 后端

```python
# 1. 将连续页索引分组，减少 RDMA 调用次数
prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
    prefill_kv_indices, dst_kv_indices
)

# 2. 计算所有层的传输块
transfer_blocks = []
for layer in all_layers:
    for prefill_block, decode_block in zip(prefill_kv_blocks, dst_kv_blocks):
        src_addr = layer_ptr + prefill_block[0] * item_len
        dst_addr = dst_layer_ptr + decode_block[0] * item_len
        length = item_len * len(prefill_block)
        transfer_blocks.append((src_addr, dst_addr, length))

# 3. 一次性批量 RDMA 写入
engine.batch_transfer_sync_write(session_id, src_addrs, dst_addrs, lengths)
```

**优势**：连续页合并传输，单次 RDMA 批量调用，效率最高。

#### 路径 B：不同 TP 大小 per-token (`send_kvcache_slice`)

**条件**：`prefill_tp != decode_tp`，未启用 Staging Buffer

```python
# 计算 head 切片参数
if prefill_tp > decode_tp:
    # 多个 Prefill rank → 1 个 Decode rank
    src_head_start = 0
    num_heads_to_send = src_heads_per_rank
    dst_head_start = (unique_head_idx * src_heads_per_rank) % dst_heads_per_rank
else:
    # 1 个 Prefill rank → 多个 Decode rank
    src_head_start = (dst_tp_rank * dst_heads_per_rank) % src_heads_per_rank
    num_heads_to_send = dst_heads_per_rank
    dst_head_start = 0

# 逐 token 切片传输（每个 token 内只传输目标 head 对应的字节）
for layer in all_layers:
    for page_idx, decode_page_idx in zip(prefill_pages, decode_pages):
        for token_in_page in range(page_size):
            src_addr = layer_ptr + page_idx * item_len + token_offset + head_slice_offset
            dst_addr = ...
            # 单次传输 = num_heads_to_send * head_dim * dtype_size
```

**特点**：支持任意 M:N TP 映射，但由于逐 token 粒度，传输调用次数较多。

#### 路径 C：Staging Buffer 路径 (`send_kvcache_staged`)

**条件**：`prefill_tp != decode_tp` + `SGLANG_DISAGG_STAGING_BUFFER=1`

```
Prefill 端:                          Decode 端:
┌──────────────┐                    ┌──────────────┐
│ KV Cache     │                    │ Staging      │
│ (分散页)     │                    │ Buffer       │
└──────┬───────┘                    └──────┬───────┘
       │ gather (GPU 内操作)                │
       ▼                                   │
┌──────────────┐     RDMA bulk write       │
│ Staging      │──────────────────────────►│
│ Buffer       │                           │
└──────────────┘                    scatter │
                                    (GPU)  ▼
                                   ┌──────────────┐
                                   │ KV Cache     │
                                   │ (目标页)      │
                                   └──────────────┘
```

**原理**：
1. Prefill 端将分散 KV 数据 gather 到连续 staging buffer
2. 一次 bulk RDMA write 传输整块数据
3. Decode 端从 staging buffer scatter 到目标 KV 页

**优势**：将大量小粒度 RDMA 操作合并为单次大块传输，显著降低 RDMA 开销。

### 4.5 线程模型

```
                    Prefill Instance
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  bootstrap_thread (1)         transfer_worker threads (N)   │
│  ┌─────────────────┐         ┌─────────────────────────┐   │
│  │ ZMQ recv loop   │         │ Queue[0] → Worker[0..M] │   │
│  │ - 接收 decode   │         │ Queue[1] → Worker[0..M] │   │
│  │   注册信息      │         │ ...                     │   │
│  │ - 接收预分配    │         │ Queue[K] → Worker[0..M] │   │
│  │   通知          │         └─────────────────────────┘   │
│  └─────────────────┘                                       │
│                                                             │
│  N = SGLANG_DISAGGREGATION_QUEUE_SIZE (默认按 CPU 核数计算)  │
│  M = SGLANG_DISAGGREGATION_THREAD_POOL_SIZE / N            │
│  默认线程数: min(max(4, cpu_count * 0.5 / 8), 12)          │
└─────────────────────────────────────────────────────────────┘
```

### 4.6 辅助数据传输

KV Cache 之外的辅助数据（如 attention mask, position encoding）通过两种方式传输：

| 方式 | 条件 | 实现 |
|------|------|------|
| RDMA | 默认 | `send_aux()` → `batch_transfer_sync` |
| TCP fallback | NVLink 自定义内存池或 `SGLANG_MOONCAKE_SEND_AUX_TCP=1` | `send_aux_tcp()` → ZMQ `send_multipart` |

TCP fallback 存在的原因：NVLink 分配器分配的内存区域在某些情况下不能直接用于 RDMA。

### 4.7 数据结构

```python
@dataclass
class TransferKVChunk:
    """传输队列中的 KV 数据块"""
    room: int                              # 请求标识
    prefill_kv_indices: np.ndarray         # Prefill 端 KV 页索引
    prefill_aux_index: int                 # 辅助数据索引
    index_slice: slice                     # 对应 decode 端 dst_kv_indices 的切片
    is_last_chunk: bool                    # 是否为最后一个 chunk
    state_indices: Optional[List]          # Mamba/SWA 等额外状态索引

@dataclass
class TransferInfo:
    """Decode 端发来的传输目标信息"""
    room: int
    endpoint: str                          # Decode IP
    dst_port: int                          # Decode ZMQ 端口
    mooncake_session_id: str              # Decode 端引擎 session ID
    dst_kv_indices: np.ndarray            # Decode 预分配的 KV 页索引
    dst_aux_index: int                    # 辅助数据目标索引
    required_dst_info_num: int            # 需要的目标 rank 数
    decode_prefix_len: Optional[int]      # Decode 端已有的 prefix 长度（用于 radix cache）
```

---

## 5. HiCache 分布式存储后端

### 5.1 三级缓存架构

```
┌─────────────────────────────────────────────────────────┐
│                    SGLang 实例                            │
│                                                         │
│  L1 (GPU)  ──evict──►  L2 (Host DRAM)  ──offload──►  L3│
│  KV Cache Pool         Host Memory Pool           Mooncake│
│  (fastest, smallest)   (medium)                   Store  │
│                                                   (largest)│
└────────────────────────────────────────────────────┬────┘
                                                     │
                               ┌─────────────────────┼──────────────────────┐
                               │                     │                      │
                     ┌─────────┴────────┐  ┌─────────┴────────┐  ┌─────────┴────────┐
                     │  Store Node 1    │  │  Store Node 2    │  │  Store Node N    │
                     │  DRAM + SSD      │  │  DRAM + SSD      │  │  DRAM + SSD      │
                     └──────────────────┘  └──────────────────┘  └──────────────────┘
```

### 5.2 MooncakeStore 实现

文件路径：`python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`

```python
class MooncakeStore(HiCacheStorage, MooncakeBaseStore):
    """L3 分布式 KV 缓存存储"""

    def setup(self, ...):
        # 初始化 MooncakeDistributedStore
        # 连接到 master 或直接连接 client
        # 注册本地 KV buffer 用于 RDMA 传输

    def put(self, pool_name, pool_transfer):
        # 将 KV 数据写入分布式存储
        # 通过 RDMA 零拷贝写入远端节点

    def get(self, pool_name, pool_transfer):
        # 从分布式存储读取 KV 数据
        # RDMA 直接读取到本地 GPU/Host 内存
```

### 5.3 部署模式

| 模式 | 说明 | 环境变量 |
|------|------|---------|
| Master + Client | Master 管理元数据，Client 提供存储 | `MOONCAKE_MASTER=ip:port` |
| Standalone | 直接连接单个存储节点 | `MOONCAKE_CLIENT=ip:port` + `MOONCAKE_STANDALONE_STORAGE=True` |
| 集成模式 | 复用 PD 分离的 Transfer Engine | `SGLANG_HICACHE_MOONCAKE_REUSE_TE=True` (默认) |

### 5.4 SSD 卸载

启用 `MOONCAKE_ENABLE_SSD_OFFLOAD=True` 后，当 Store 节点 DRAM 不足时自动溢出到 SSD：

```
DRAM (热数据) → SSD (冷数据)
```

配置路径：`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/path/to/ssd`

---

## 6. MoE 专家并行通信

### 6.1 应用场景

在 Mixture-of-Experts 模型中（如 DeepSeek-V3），token 需要根据 routing 结果在不同 GPU 的专家之间分发。Mooncake 提供低延迟的 all-to-all 通信。

### 6.2 EP Buffer

文件路径：`python/sglang/srt/layers/moe/token_dispatcher/mooncake.py`

```python
class EPBuffer:
    """Mooncake EP 通信 Buffer 单例"""

    @classmethod
    def get_ep_buffer(cls, group, hidden_size, param_bytes, deepep_mode, ...):
        from mooncake.mooncake_ep_buffer import Buffer

        cls._buffer = Buffer(
            group=group,
            num_nvl_bytes=...,
            num_rdma_bytes=...,
            low_latency_mode=True,
            num_qps_per_conn=...,
        )
```

### 6.3 工作流程

```
Dispatch (分发):
  Token → Routing决策 → Mooncake EP Buffer all-to-all → 目标专家

Combine (合并):
  专家输出 → Mooncake EP Buffer all-to-all → 汇总到原始 rank
```

### 6.4 配置

```bash
# 启用 Mooncake MoE all-to-all
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --moe-a2a-backend mooncake \
    --mooncake-ib-device mlx5_0,mlx5_1
```

环境变量：
- `SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`: 每 rank 最大 dispatch token 数（默认 128）

---

## 7. 弹性 EP 故障恢复

### 7.1 原理

当 MoE 推理中某个 rank 故障时，使用 Mooncake 的弹性 EP 接口恢复：

```python
# 故障恢复
from mooncake.ep import recover_ranks, get_peer_state

# 检测故障 rank 状态
peer_state = get_peer_state(rank_id)

# 从备份恢复专家权重
recover_ranks(failed_ranks, backup_source)
```

### 7.2 Expert Backup

文件路径：`python/sglang/srt/elastic_ep/expert_backup_client.py`

通过 Mooncake Transfer Engine 将专家权重备份到远端节点，故障时直接 RDMA 恢复，无需重新加载模型。

---

## 8. 配置参考

### 8.1 CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--disaggregation-transfer-backend` | `"mooncake"` | PD 分离传输后端 |
| `--disaggregation-ib-device` | `None` (自动检测) | PD 分离使用的 IB 设备 |
| `--mooncake-ib-device` | `None` (自动检测) | MoE EP 使用的 IB 设备 |
| `--elastic-ep-backend` | `None` | 弹性 EP 后端选择 |
| `--moe-a2a-backend` | `"none"` | MoE all-to-all 后端 |
| `--hicache-storage-backend` | `"file"` | HiCache L3 存储后端 |
| `--encoder-transfer-backend` | `"zmq_to_scheduler"` | Encoder 分离传输后端 |

### 8.2 环境变量（完整列表）

#### KV Transfer 相关

| 变量 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` | str | `None` | 自定义内存池类型: `NVLINK`, `BAREX`, `INTRA_NODE_NVLINK` |
| `ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE` | bool | `False` | 华为 Ascend NPU 传输模式 |
| `ASCEND_NPU_PHY_ID` | int | `-1` | Ascend 物理设备 ID |
| `SGLANG_MOONCAKE_SEND_AUX_TCP` | bool | `False` | 强制 TCP 传输辅助数据 |
| `SGLANG_ENABLE_FAILED_SESSION_PROBE` | bool | `False` | 启用故障会话探测恢复 |
| `SGLANG_FAILED_SESSION_PROBE_INTERVAL_S` | float | `30.0` | 探测间隔（秒） |
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | int | 自动计算 | 传输线程池大小 |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | int | 自动 | 传输队列数量 |
| `SGLANG_DISAGG_STAGING_BUFFER` | bool | `False` | 启用 Staging Buffer 传输 |

#### HiCache Mooncake Store 相关

| 变量 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH` | str | `None` | JSON 配置文件路径 |
| `SGLANG_HICACHE_MOONCAKE_REUSE_TE` | bool | `True` | 复用共享 Transfer Engine |
| `MOONCAKE_MASTER` | str | `None` | Master 服务地址 |
| `MOONCAKE_CLIENT` | str | `None` | Client 服务地址 |
| `MOONCAKE_LOCAL_HOSTNAME` | str | `"localhost"` | 本地主机名 |
| `MOONCAKE_TE_META_DATA_SERVER` | str | `"P2PHANDSHAKE"` | 元数据服务器类型 |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | str | `"4gb"` | 贡献到分布式池的内存大小 |
| `MOONCAKE_PROTOCOL` | str | `"tcp"` | 传输协议: `tcp` / `rdma` |
| `MOONCAKE_DEVICE` | str | `""` | RDMA 设备名 |
| `MOONCAKE_MASTER_METRICS_PORT` | int | `9003` | Prometheus 监控端口 |
| `MOONCAKE_CHECK_SERVER` | bool | `False` | 启动时检查服务器健康 |
| `MOONCAKE_STANDALONE_STORAGE` | bool | `False` | 独立存储模式 |
| `MOONCAKE_ENABLE_SSD_OFFLOAD` | bool | `False` | 启用 SSD 卸载 |
| `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` | str | `None` | SSD 卸载路径 |

#### 测试相关

| 变量 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `SGLANG_TEST_PD_DISAGG_BACKEND` | str | `"mooncake"` | 测试默认后端 |
| `SGLANG_TEST_PD_DISAGG_DEVICES` | str | `None` | 测试 IB 设备覆盖 |

---

## 9. 与其他传输后端对比

### 9.1 特性对比

| 特性 | Mooncake | Nixl | Mori | ZMQ |
|------|----------|------|------|-----|
| **传输协议** | RDMA (GPUDirect) | RDMA (libnixl) | RDMA | TCP |
| **是否默认** | **是** | 否 | 否 | 仅 Encoder |
| **内存注册** | 显式 register/batch_register | nixl agent | Mori engine | 无 |
| **传输 API** | `batch_transfer_sync_write` | `xfer` (异步 + poll) | 自定义 | send/recv |
| **自定义内存池** | NVLink/BARex/INTRA_NODE | 无 | 无 | 无 |
| **Staging Buffer** | 支持 | 支持 | 未知 | 不适用 |
| **HiCache 存储** | 完整分布式 Store | 可选后端 | 无 | 无 |
| **MoE EP** | 原生 EP Buffer | 计划中 | 无 | 无 |
| **弹性 EP** | `mooncake.ep` 恢复 | 替代方案 | 无 | 无 |
| **Ascend NPU** | 支持 | 不支持 | 不支持 | 支持 |
| **Decode 端 Radix Cache** | 支持 | 支持 | 未知 | 不支持 |
| **Session 恢复** | Probe 机制 | 错误处理 | 未知 | 无 |

### 9.2 选择建议

| 场景 | 推荐后端 | 理由 |
|------|---------|------|
| 标准 PD 分离部署 | Mooncake | 默认后端，功能最完整 |
| 纯 InfiniBand 集群 | Mooncake 或 Nixl | 两者都支持 RDMA |
| 需要 HiCache + PD | Mooncake | 可复用同一个 Transfer Engine |
| MoE 专家并行 | Mooncake | 唯一提供 EP Buffer 的后端 |
| 华为 Ascend 环境 | Mooncake | 唯一支持 Ascend NPU 传输 |
| 无 RDMA 环境 | ZMQ (Encoder only) | TCP 回退 |

---

## 10. 故障处理与容错

### 10.1 Session 黑名单机制

```python
# 传输失败时立即标记
if ret != 0:
    with self.session_lock:
        self.session_failures[session_id] += 1
        if self.session_failures[session_id] >= 1:
            self.failed_sessions.add(session_id)  # 首次失败即黑名单

# 后续传输跳过已黑名单的 session
with self.session_lock:
    if req.mooncake_session_id in self.failed_sessions:
        # 直接标记 Failed，不尝试传输
        self.update_status(room, KVPoll.Failed)
```

### 10.2 Session 探测恢复

当 `SGLANG_ENABLE_FAILED_SESSION_PROBE=True` 时，后台线程定期探测黑名单中的 session：

```python
def _failed_session_probe_loop(self):
    while not self._failed_session_probe_shutdown.is_set():
        time.sleep(self.failed_session_probe_interval)
        with self.session_lock:
            sessions_to_probe = list(self.failed_sessions)

        for session_id in sessions_to_probe:
            ret = self.engine.send_probe(session_id)
            if ret == 0:  # 探测成功
                with self.session_lock:
                    self.failed_sessions.discard(session_id)
                    self.session_failures.pop(session_id, None)
                FAILED_SESSION_RECOVERIES.inc()  # Prometheus 计数
```

### 10.3 节点故障处理

当检测到节点级故障时：
1. 清理连接池中该节点的所有连接
2. 标记所有受影响的 room 为 Failed
3. 通知 scheduler 进行请求重试

### 10.4 Prometheus 监控指标

```
sglang:failed_session_recoveries_total  - 从黑名单恢复的 session 数
```

---

## 11. 性能优化细节

### 11.1 连续页索引合并

`group_concurrent_contiguous()` 函数将逻辑上连续的 KV 页索引合并为大块传输：

```
输入: prefill_indices = [0, 1, 2, 5, 6, 10]
      decode_indices  = [3, 4, 5, 7, 8, 12]

分组后:
  Block 1: prefill [0,1,2] → decode [3,4,5]  (1 次 RDMA, 3 页)
  Block 2: prefill [5,6]   → decode [7,8]    (1 次 RDMA, 2 页)
  Block 3: prefill [10]    → decode [12]     (1 次 RDMA, 1 页)

效果: 6 页只需 3 次 RDMA 调用（而非 6 次）
```

### 11.2 批量传输 API

```python
# 单次调用传输所有层的所有块
engine.batch_transfer_sync_write(
    session_id,
    [src_addr_1, src_addr_2, ..., src_addr_N],  # N 个源地址
    [dst_addr_1, dst_addr_2, ..., dst_addr_N],  # N 个目标地址
    [length_1, length_2, ..., length_N]          # N 个长度
)
```

Mooncake 引擎内部将这些请求合并为最少的 RDMA 操作。

### 11.3 自定义内存池

| 内存池类型 | 用途 | 优势 |
|-----------|------|------|
| **NVLINK** | NVLink 可访问内存 | 同节点 GPU 间零拷贝访问 |
| **BAREX** | BAR Extension 内存 | 跨 PCIe 直接 peer 访问 |
| **INTRA_NODE_NVLINK** | 节点内 NVLink | 当前未启用 |

当使用自定义内存池时，传输策略从 "所有层合并一次传输" 切换为 "逐层并行传输"（因为 NVLink 内存的对齐要求不同）。

### 11.4 Chunked Prefill 与流式传输

SGLang 支持 Chunked Prefill（`--chunked-prefill-size`），Mooncake 传输与之配合：

```
Prefill Forward:
  Chunk 1 完成 → 立即入队传输 (is_last_chunk=False)
  Chunk 2 完成 → 立即入队传输 (is_last_chunk=False)
  Chunk N 完成 → 入队传输 (is_last_chunk=True) + 发送 aux + 同步状态
```

这实现了 **计算与传输的 overlap**：当 Prefill 处理下一个 chunk 时，前一个 chunk 的 KV 已经在传输。

### 11.5 传输队列负载均衡

多个传输队列 + 多线程池实现并行传输：

```
Request 1 → Queue[hash(room) % K] → Worker Thread Pool
Request 2 → Queue[hash(room) % K] → Worker Thread Pool
...
```

不同请求的 KV 传输在不同队列中并行执行，互不阻塞。

---

## 12. 部署实践

### 12.1 基础 PD 分离部署

```bash
# Prefill 实例 (Node 1, 2xGPU)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --disaggregation-mode prefill \
    --tp 2 \
    --port 30000 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device mlx5_0,mlx5_1

# Decode 实例 (Node 2, 2xGPU)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --disaggregation-mode decode \
    --tp 2 \
    --port 30001 \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device mlx5_0,mlx5_1 \
    --dist-init-addr <prefill_node_ip>:30000
```

### 12.2 不同 TP 大小 PD 分离

```bash
# Prefill: TP=4 (利用更多 GPU 加速长序列 prefill)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --disaggregation-mode prefill \
    --tp 4 \
    --port 30000

# Decode: TP=2 (节省 GPU 资源，decode 阶段不需要那么多并行度)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --disaggregation-mode decode \
    --tp 2 \
    --port 30001 \
    --dist-init-addr <prefill_node_ip>:30000
```

### 12.3 带 HiCache 的部署

```bash
# 先启动 Mooncake Store Master
# (独立进程，管理分布式 KV 缓存)

# SGLang 实例
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --hicache-storage-backend mooncake \
    --hicache-storage-backend-extra-config '{"master_server_address": "10.0.0.1:50051"}'
```

或通过环境变量配置：

```bash
export MOONCAKE_MASTER=10.0.0.1:50051
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_DEVICE=mlx5_0
export MOONCAKE_GLOBAL_SEGMENT_SIZE=4gb

python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --hicache-storage-backend mooncake
```

### 12.4 MoE 专家并行部署

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --moe-a2a-backend mooncake \
    --mooncake-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3
```

### 12.5 Staging Buffer 优化（跨 TP 传输）

```bash
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=256  # Staging buffer 大小

python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --disaggregation-mode prefill \
    --tp 4 \
    --disaggregation-transfer-backend mooncake
```

### 12.6 故障恢复配置

```bash
# 启用 session 探测恢复（生产环境推荐）
export SGLANG_ENABLE_FAILED_SESSION_PROBE=1
export SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=15  # 15秒探测一次
```

### 12.7 安装 Mooncake

```bash
# 方式 1：pip 安装
pip install mooncake-transfer-engine

# 方式 2：从源码构建
# 参考: https://kvcache-ai.github.io/Mooncake/getting_started/build.html
```

要求版本：`mooncake-transfer-engine >= 0.3.4.post2`（用于 batch_transfer_sync_write）

---

## 文件索引

| 文件路径 | 核心功能 |
|---------|---------|
| `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` | Transfer Engine 单例包装 |
| `python/sglang/srt/disaggregation/mooncake/__init__.py` | 模块导出 |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | PD 分离 KV 传输完整实现 (~1750 行) |
| `python/sglang/srt/disaggregation/mooncake/utils.py` | 自定义内存池管理 |
| `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | HiCache 分布式存储 |
| `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` | MoE EP 分发器 |
| `python/sglang/srt/elastic_ep/elastic_ep.py` | 弹性 EP 恢复 |
| `python/sglang/srt/elastic_ep/expert_backup_client.py` | 专家权重备份 |
| `python/sglang/srt/server_args.py` | CLI 参数定义 |
| `python/sglang/srt/environ.py` | 环境变量定义 |
| `python/sglang/srt/model_executor/model_runner.py` | 引擎初始化入口 |
| `python/sglang/srt/disaggregation/utils.py` | TransferBackend 枚举 + 工厂 |

---

## 总结

Mooncake 在 SGLang 中扮演了 **核心传输基础设施** 的角色，是 PD 分离的默认且功能最完整的后端。其设计特点：

1. **单例共享**：一个 GPU 进程只创建一个 Transfer Engine，PD 传输、HiCache、EP 通信复用
2. **多路径传输**：根据 TP 配置自动选择最优传输路径（全页批量/Head 切片/Staging Buffer）
3. **异步流水线**：多队列多线程设计实现传输与计算的 overlap
4. **生产级容错**：Session 黑名单 + 探测恢复 + 节点故障处理
5. **灵活部署**：支持 RDMA/TCP、NVIDIA/Ascend、自定义内存池等多种配置
