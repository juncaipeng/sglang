# SGLang PD 分离 KV Cache 传输架构

## 1. 概述

SGLang 支持 **Prefill-Decode (PD) 分离**部署模式，将 LLM 推理拆分为两类独立的 GPU 实例：

- **Prefill 实例 (P)**：负责处理用户输入 prompt，计算 KV Cache
- **Decode 实例 (D)**：负责自回归解码，逐 token 生成输出

核心问题：Prefill 完成后，**KV Cache 必须从 P 实例的 GPU 显存传输到 D 实例的 GPU 显存**，这是 PD 分离架构的关键路径。

```
┌─────────────┐     KV Cache Transfer      ┌─────────────┐
│   Prefill   │  ========================>  │   Decode    │
│  Instance   │  (RDMA / ZMQ / Mooncake)    │  Instance   │
│  (GPU 0-N)  │                             │  (GPU 0-M)  │
└─────────────┘                             └─────────────┘
       │                                           │
       │  ZMQ: 元数据交换 + 完成通知                │
       └───────────────────────────────────────────┘
```

---

## 2. 传输后端总览

通过 `--disaggregation-transfer-backend` 参数选择传输后端（**默认为 mooncake**）：

| 后端 | 枚举值 | 传输机制 | 适用场景 |
|------|--------|----------|----------|
| **Mooncake** | `TransferBackend.MOONCAKE` | Mooncake Transfer Engine（RDMA，`batch_transfer_sync`） | **默认后端**。GPU-to-GPU RDMA，支持 IB/RoCE/NVLink |
| **NIXL** | `TransferBackend.NIXL` | AI Dynamo NIXL（`nixl_agent`，支持 UCX/OBJ/GDS_MT/UCCL） | 高性能 RDMA，通知式完成检测 |
| **Mori** | `TransferBackend.MORI` | Mori IOEngine（RDMA，MemoryDesc/EngineDesc） | 替代 RDMA 传输 |
| **Ascend** | `TransferBackend.ASCEND` | 扩展 Mooncake，使用 Ascend Transfer Engine | 华为昇腾 NPU |
| **Fake** | `TransferBackend.FAKE` | 空操作（立即返回成功） | Warmup / 测试 |

### 后端选择工厂

```
python/sglang/srt/disaggregation/utils.py
  → TransferBackend 枚举
  → get_kv_class(backend, class_type) 工厂函数
```

---

## 3. 类层次结构

```
BaseKVManager (ABC)         BaseKVSender (ABC)         BaseKVReceiver (ABC)        BaseKVBootstrapServer (ABC)
     │                           │                           │                            │
CommonKVManager            CommonKVSender             CommonKVReceiver            CommonKVBootstrapServer
  │    │    │    │           │    │    │    │           │    │    │    │              │    │    │    │
Mooncake Nixl Mori Ascend  Mooncake Nixl Mori Ascend  Mooncake Nixl Mori Ascend   Mooncake Nixl Mori Ascend

FakeKVManager              FakeKVSender               FakeKVReceiver
```

### 关键抽象接口（`base/conn.py`）

| 接口 | 核心方法 | 角色 |
|------|---------|------|
| `BaseKVManager` | `__init__()`, `register_to_bootstrap()` | 连接管理、传输状态维护 |
| `BaseKVSender` | `init()`, `send()`, `poll()` | Prefill 侧：发送 KV 数据 |
| `BaseKVReceiver` | `init()`, `send_metadata()`, `poll()` | Decode 侧：预分配并接收 KV |
| `BaseKVBootstrapServer` | `__init__(host, port)` | 服务发现与路由 |

### 传输状态机（`KVPoll`）

```
Failed(0) < Bootstrapping(1) < WaitingForInput(2) < Transferring(3) < Success(4)
```

状态值可比较，用于跨 TP rank 的 `all_reduce(MIN)` 同步。

---

## 4. 端到端传输流程

### 4.1 全局数据流

```
Decode 收到请求
      │
      ▼
┌─────────────────────────────────────────┐
│ Phase 1: PreallocQueue（Bootstrap 握手） │
│  - 创建 KVReceiver                      │
│  - HTTP GET /route 获取 Prefill 拓扑    │
│  - ZMQ 发送 Decode 的 RDMA 内存描述符   │
│  - 等待握手完成 → WaitingForInput       │
└─────────────────────────────────────────┘
      │
      ▼ (显存可用)
┌─────────────────────────────────────────┐
│ Phase 2: 预分配 KV 槽位                 │
│  - 分配 req_pool_idx、token slots       │
│  - ZMQ 发送目的页索引给 Prefill         │
└─────────────────────────────────────────┘
      │
      ▼                                         ┌──────────────────────────┐
┌──────────────────────┐                        │ Prefill BootstrapQueue   │
│ DecodeTransferQueue  │ ◄─── ZMQ 完成通知 ──── │  - 收到 metadata         │
│  - polling receiver  │                        │  - Status→WaitingForInput│
└──────────────────────┘                        └──────────────────────────┘
                                                         │
                                                         ▼
                                                ┌──────────────────────────┐
                                                │ Prefill Forward Pass     │
                                                │  - GPU 计算 Attention    │
                                                └──────────────────────────┘
                                                         │
                                                         ▼
                                                ┌──────────────────────────┐
                                                │ send_kv_chunk            │
                                                │  - RDMA WRITE: KV pages  │
                                                │  - 发送 metadata (aux)   │
                                                │  - ZMQ 通知 Decode 完成  │
                                                └──────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│ Phase 3: TransferQueue → Success        │
│  - poll() 返回 Success                  │
│  - 读取 metadata buffer (首 token 等)   │
│  - 移入 waiting_queue → running_batch   │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│ Phase 4: 自回归解码                     │
└─────────────────────────────────────────┘
```

### 4.2 分阶段详解

#### Phase 1: Bootstrap 握手

1. **Prefill 启动**时，每个 rank 绑定一个 ZMQ PULL socket，并通过 HTTP PUT 注册到 Bootstrap Server：
   ```
   PUT http://<bootstrap_host>:<bootstrap_port>/route
   Payload: {attn_tp_size, attn_tp_rank, attn_cp_rank, attn_dp_rank, pp_rank, rank_ip, rank_port, page_size, kv_cache_dtype}
   ```

2. **Bootstrap Server**（aiohttp）维护 Prefill 拓扑表：
   ```
   prefill_port_table[dp_rank][cp_rank][tp_rank][pp_rank] → PrefillRankInfo(rank_ip, rank_port)
   ```
   路由端点：`PUT /route`（注册）、`GET /route`（查询）、`POST /register_dp_rank`、`GET /health`

3. **Decode 创建 KVReceiver** 时：
   - HTTP GET 查询 Prefill 拓扑
   - 解析 rank 映射（支持异构 TP）
   - 通过 ZMQ 发送自身的 RDMA 内存描述符（buffer 基址、GPU id、TP 信息等）

#### Phase 2: 预分配与元数据交换

1. Decode 在自身 GPU 上**预分配 KV Cache 页**：
   - 分配 `req_pool_idx` 和 token slots
   - 支持 decode 侧 radix cache 前缀匹配

2. **发送目的地址**给 Prefill：
   ```
   ZMQ multipart → [bootstrap_room, dst_page_indices, aux_index, state_indices, ...]
   ```

3. Prefill 收到后存储 `TransferInfo`，当所有目标 rank 信息到齐，状态转为 `WaitingForInput`

#### Phase 3: Prefill 前向计算 + RDMA 传输

1. Prefill 完成 GPU forward pass 后：
   - 获取 KV indices：`req_to_token_pool.req_to_token[req.req_pool_idx, start:end]`
   - 转换为 page indices
   - 调用 `sender.send(page_indices, state_indices)`

2. 传输核心——**单边 RDMA WRITE**：
   ```python
   # Mooncake 后端
   engine.batch_transfer_sync(session_id, src_addrs, dst_addrs, lengths)

   # NIXL 后端
   agent.initialize_xfer("WRITE", src_descs, dst_descs, peer_name, notif)
   agent.transfer(handle)
   ```

3. 传输完成后通知 Decode（各后端方式不同，见第 5 节）

#### Phase 4: Decode 接收并开始解码

1. `receiver.poll()` 返回 `Success`
2. 读取 metadata buffer（首输出 token、cached_tokens 计数、logprobs 等）
3. 请求移入 `waiting_queue` → `running_batch`，开始自回归解码

### 4.3 跨 TP Rank 同步

所有 TP rank 必须就传输状态达成一致：

```python
# utils.py: poll_and_all_reduce
polls = [int(poller.poll()) for poller in pollers]
tensor = torch.tensor(polls, dtype=torch.uint8, device="cpu")
dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=gloo_group)
```

使用 MIN 规约确保任一 rank 未完成则全部等待，防止竞态条件。

---

## 5. 各传输后端实现细节

### 5.1 Mooncake（默认后端）

**核心文件：**
- `disaggregation/mooncake/conn.py`（~1900 行）
- `distributed/device_communicators/mooncake_transfer_engine.py`

**架构：**
```
Prefill                                  Decode
┌────────────────────┐                  ┌─────────────────────┐
│ MooncakeKVSender   │                  │ MooncakeKVReceiver   │
│ MooncakeKVManager  │                  │ MooncakeKVManager    │
│        │           │                  │         │            │
│  TransferEngine    │── RDMA WRITE ──▶ │  TransferEngine      │
│  (shared singleton)│                  │  (shared singleton)  │
│        │           │                  │         │            │
│  ZMQ PUSH ─────────│── completion ──▶ │── ZMQ PULL          │
└────────────────────┘                  └─────────────────────┘
```

**关键特性：**
- **Session-based**：每个 Decode 实例获得一个 `session_id`，Prefill 用此 ID 寻址
- **批量同步传输**：`engine.batch_transfer_sync(session_id, src_addrs, dst_addrs, lengths)`
- **完成通知**：RDMA 完成后通过 ZMQ 发送 `{room, status, prefill_rank}` 给 Decode
- **线程池**：多个 transfer worker（通过 `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` 配置）消费 `FastQueue`
- **自定义内存池**：支持 NVLink / BareX 分配器（`SGLANG_MOONCAKE_CUSTOM_MEM_POOL`）
- **Aux 数据回退**：当 RDMA 传输辅助数据不稳定时，可通过 `SGLANG_MOONCAKE_SEND_AUX_TCP` 回退到 TCP/ZMQ

### 5.2 NIXL

**核心文件：** `disaggregation/nixl/conn.py`（~2083 行）

**关键特性：**
- **可插拔后端**：通过 `SGLANG_DISAGGREGATION_NIXL_BACKEND` 选择（默认 UCX），支持 UCX/OBJ/GDS_MT/UCCL
- **Peer 注册**：`agent.add_remote_agent(metadata)` 交换 RDMA 元数据
- **内存注册**：`agent.register_memory(addrs, "VRAM"/"DRAM")`
- **异步传输 + RDMA 通知**：传输完成通过 RDMA notification tag 传递，**无需额外 ZMQ 通知**
  ```
  notif tag: "{room}_kv_{chunk_id}_{is_last}_{pp_rank}"
  decode: agent.get_new_notifs() 检测完成
  ```
- **优势**：零额外网络开销的完成检测

### 5.3 Mori

**核心文件：** `disaggregation/mori/conn.py`

- 基于 `mori.io.IOEngine` + `RdmaBackendConfig`
- 使用 `MemoryDesc` / `EngineDesc` 进行 peer 交换
- Session-based，同步传输，ZMQ 元数据交换
- 架构与 Mooncake 类似

### 5.4 Ascend

**核心文件：** `disaggregation/ascend/conn.py`

- 继承 `MooncakeKVManager`，覆写 `init_engine()` 使用 `AscendTransferEngine`
- 覆写 `register_buffer_to_engine()` 和 `get_mla_kv_ptrs_with_pp()` 适配 NPU 特性

### 后端对比

| 特性 | Mooncake | NIXL | Mori | Ascend |
|------|----------|------|------|--------|
| 传输方式 | 同步 RDMA | 异步 RDMA | 同步 RDMA | 同步 RDMA |
| 完成通知 | ZMQ 消息 | RDMA notification tag | ZMQ 消息 | ZMQ 消息 |
| 额外网络开销 | 有（ZMQ） | 无 | 有（ZMQ） | 有（ZMQ） |
| 自定义内存池 | NVLink/BareX | 否 | 否 | 否 |
| 可插拔子后端 | 否 | UCX/OBJ/GDS_MT/UCCL | 否 | 否 |
| Staging Buffer | 支持 | 支持 | 不支持 | 不支持 |
| 硬件适配 | 通用 GPU | 通用 GPU | 通用 GPU | 华为昇腾 NPU |

---

## 6. Staging Buffer 系统（异构 TP 传输）

当 Prefill 和 Decode 使用不同的 TP 大小时，KV heads 需要重新分布。Staging Buffer 系统处理此场景：

```
Prefill (TP=4)                          Decode (TP=2)
┌──────────┐                           ┌──────────────────┐
│ Head 0-3 │──gather──▶ Staging ──RDMA──▶ Staging Ring Buf │──scatter──▶ KV Cache
│ Head 4-7 │           Buffer           │  (Watermark FC)  │
└──────────┘                           └──────────────────┘
```

**关键文件：**
- `common/staging_handler.py` — scatter 生命周期管理
- `common/staging_buffer.py` — staging 内存分配

**流控：** Watermark 机制反压——Decode 通过 `WATERMARK` 消息通知 Prefill 已释放的 staging 空间。

**启用方式：** `SGLANG_DISAGG_STAGING_BUFFER=True`（仅 Mooncake 和 NIXL 支持）

---

## 7. Mooncake 在 SGLang 中的四种角色

Mooncake 在 SGLang 生态中扮演**四个不同角色**，不仅仅是 PD 传输后端：

| 角色 | API 层 | 用途 | 关键文件 |
|------|--------|------|----------|
| **PD Transfer Backend** | `mooncake.engine.TransferEngine` | 点对点 RDMA 传输 KV Cache | `disaggregation/mooncake/conn.py` |
| **HiCache L3 Storage** | `mooncake.store.MooncakeDistributedStore` | 分布式 KV Cache 池（L3 层） | `mem_cache/storage/mooncake_store/mooncake_store.py` |
| **MoE Expert Parallelism** | `mooncake.mooncake_ep_buffer.Buffer` | MoE 模型的 token all-to-all 分发 | `layers/moe/token_dispatcher/mooncake.py` |
| **Encoder Transfer** | `mooncake.engine.TransferEngine` | 多模态模型 embedding 传输 | `disaggregation/encode_receiver.py` |

> **注意**：PD Transfer 使用的是低层 `TransferEngine`（点对点 RDMA），而 HiCache 使用的是高层 `MooncakeDistributedStore`（分布式 KV 存储），两者是完全不同的 API 层。

---

## 8. 能否仅使用 Mooncake Cache Pool 进行 PD 传输？

### 8.1 当前 PD 传输架构（点对点模式）

当前 SGLang 的 PD 传输是**点对点 RDMA 写入**：

```
Prefill GPU ──── RDMA WRITE ────▶ Decode GPU
  (src KV)                        (dst KV slots，预分配)
```

- KV Cache 驻留在各实例自己的 GPU 显存中（由 `TokenToKVPool` 管理）
- Mooncake Transfer Engine 提供 `batch_transfer_sync()` API 进行 RDMA 写入
- **没有**共享的外部 Cache Pool 或中间存储

### 8.2 Mooncake 作为独立 Cache Pool（HiCache L3 模式）

SGLang 确实支持 Mooncake 作为**独立的 KV Cache 池**，但这是通过 **HiCache 三级缓存系统**实现的，与 PD 传输是**正交的功能**：

```
HiCache 三级存储架构：

┌──────────┐     ┌──────────┐     ┌─────────────────────┐
│  L1 GPU  │ ──▶ │  L2 CPU  │ ──▶ │ L3 Mooncake Store   │
│ (active) │     │ (pinned) │     │ (distributed pool)  │
└──────────┘     └──────────┘     └─────────────────────┘
                                           │
                                  ┌────────┴────────┐
                                  │                 │
                             RDMA (zero-copy)   SSD Offload
                                  │
                       ┌─────────────────────┐
                       │ Mooncake Master Svc  │
                       │ (管理全局 pool)       │
                       └─────────────────────┘
```

**MooncakeStore 关键操作：**
- `batch_put_from(keys, buffer_ptrs, buffer_sizes)` — 零拷贝写入分布式存储
- `batch_get_into(keys, buffer_ptrs, buffer_sizes)` — 零拷贝读取到 host 内存
- `batch_is_exist(keys)` — 检查 KV page 是否存在

**Standalone 模式**：当 `MOONCAKE_STANDALONE_STORAGE=True` 时，SGLang 作为 "Dummy Client" 连接到外部 Mooncake Store Service，实现：
- Cache 跨 SGLang 重启持久化
- 进程架构解耦
- 内存管理由 Store Service 负责

### 8.3 能否用 Mooncake Cache Pool 替代点对点 PD 传输？

**目前不支持，但架构上可行。** 具体分析：

| 维度 | 点对点 RDMA（当前） | Cache Pool 中转（假设） |
|------|---------------------|------------------------|
| 传输路径 | P-GPU → D-GPU（直连） | P-GPU → Pool → D-GPU（两跳） |
| 延迟 | 低（单次 RDMA） | 较高（两次 RDMA 或 RDMA + 内存拷贝） |
| 耦合度 | P 和 D 必须同时在线 | P 和 D 可解耦（异步） |
| 容错性 | P 掉线则传输失败 | Cache 持久化，P 掉线后 D 仍可读取 |
| 扩展性 | 1:1 对应 | N:M 灵活调度 |

**结论**：
1. **性能场景**：点对点 RDMA 是最优选择，延迟最低
2. **弹性场景**：若需 P/D 实例解耦、动态伸缩、故障恢复，Mooncake Cache Pool 是更好的架构选择
3. **当前实现**：两个系统独立运行，HiCache + Mooncake Store 可与 PD 传输并行使用（共享 TransferEngine 实例），但 Cache Pool 不作为 PD 传输的中间件

---

## 9. KV Cache 内存布局

### MHA 模型
```
kv_data_ptrs = [K_layer0, K_layer1, ..., V_layer0, V_layer1, ...]
每个 buffer: [num_pages, page_size, head_num * head_dim * dtype_size]
```

### MLA 模型
```
kv_data_ptrs = [layer0, layer1, ...]  // K 和 V 融合为单一压缩表示
```

RDMA 写入地址计算：
```
local_offset  = page_index     * item_len
remote_offset = dst_page_index * item_len
size          = item_len
```

连续页会被分组以减少 RDMA work requests（`group_concurrent_contiguous`）。

---

## 10. 配置参数

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--disaggregation-transfer-backend` | `"mooncake"` | 传输后端选择 |
| `--disaggregation-mode` | `"null"` | 设为 `prefill` 或 `decode` |
| `--disaggregation-bootstrap-port` | `8998` | Bootstrap Server HTTP 端口 |
| `--disaggregation-ib-device` | `None` | IB 设备（如 `mlx5_0,mlx5_1`） |
| `--disaggregation-decode-enable-radix-cache` | `False` | Decode 侧前缀缓存（仅 nixl/mooncake） |
| `--hicache-storage-backend` | `None` | HiCache L3 后端（可选 `mooncake`） |

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | 动态（0.5*cpu/8, 限 4-12） | Prefill 传输线程数 |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | 4 | 传输队列数 |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | 300s | Bootstrap 超时 |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | 300s | Decode 传输超时 |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | 5.0s | 心跳检测间隔 |
| `SGLANG_DISAGG_STAGING_BUFFER` | `False` | 启用异构 TP staging |
| `SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB` | 64 | Prefill staging 缓冲区大小 |
| `SGLANG_DISAGG_STAGING_POOL_SIZE_MB` | 4096 | Decode staging 环形缓冲区大小 |
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` | `None` | 自定义分配器：NVLINK/BAREX/INTRA_NODE_NVLINK |
| `SGLANG_MOONCAKE_SEND_AUX_TCP` | `False` | Aux 数据走 TCP/ZMQ |
| `SGLANG_DISAGGREGATION_NIXL_BACKEND` | `"UCX"` | NIXL 传输插件 |
| `MOONCAKE_STANDALONE_STORAGE` | `False` | HiCache Dummy Client 模式 |
| `MOONCAKE_MASTER` | `None` | Mooncake Store Master 地址 |

---

## 11. 关键文件索引

| 文件 | 说明 |
|------|------|
| `python/sglang/srt/disaggregation/base/conn.py` | 抽象基类定义 |
| `python/sglang/srt/disaggregation/common/conn.py` | 通用实现（ZMQ bootstrap、rank 映射）~1100 行 |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | Mooncake 后端 ~1900 行 |
| `python/sglang/srt/disaggregation/nixl/conn.py` | NIXL 后端 ~2083 行 |
| `python/sglang/srt/disaggregation/mori/conn.py` | Mori 后端 |
| `python/sglang/srt/disaggregation/ascend/conn.py` | Ascend NPU 后端 |
| `python/sglang/srt/disaggregation/fake/conn.py` | 空操作后端 |
| `python/sglang/srt/disaggregation/prefill.py` | Prefill 调度器集成 |
| `python/sglang/srt/disaggregation/decode.py` | Decode 调度器集成 |
| `python/sglang/srt/disaggregation/utils.py` | 后端枚举、工厂函数 |
| `python/sglang/srt/disaggregation/common/staging_handler.py` | Staging scatter 生命周期 |
| `python/sglang/srt/disaggregation/common/staging_buffer.py` | Staging 内存管理 |
| `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` | Mooncake 引擎封装 |
| `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | HiCache L3 Mooncake 存储 |
| `python/sglang/srt/server_args.py` | CLI 参数定义 |
| `python/sglang/srt/environ.py` | 环境变量定义 |
