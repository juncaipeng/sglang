# SGLang PD分离(Prefill-Decode Disaggregation)架构详解

## 1. 概述

PD分离(Prefill-Decode Disaggregation)是SGLang中的核心优化技术，通过将大语言模型推理的Prefill阶段和Decode阶段分离到不同的GPU实例上执行，实现以下优化目标：

1. **资源利用率优化**：Prefill阶段计算密集、Decode阶段内存密集，分离后可以针对不同阶段配置不同数量的GPU
2. **吞吐量提升**：Prefill实例可以并行处理多个请求，Decode实例专注于增量生成
3. **延迟优化**：避免Prefill阶段的长计算阻塞Decode阶段的小批量生成
4. **内存效率**：Decode实例可以使用更少GPU资源，专注于KV Cache管理

## 2. 整体架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HTTP Request                                    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              Bootstrap Server                                 │
│  (服务发现、Rank映射、DP Rank注册)                                             │
│  端口: --disaggregation-bootstrap-port (默认8998)                             │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│      Prefill Instance         │   │       Decode Instance          │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐   │
│  │ PrefillBootstrapQueue   │  │   │  │    PreallocQueue        │   │
│  │ (创建KVSender,握手)      │  │   │  │ (创建KVReceiver,预分配) │   │
│  └──────────┬──────────────┘  │   │  └──────────┬──────────────┘   │
│             ▼                 │   │             ▼                  │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐   │
│  │    WaitingQueue         │  │   │  │     TransferQueue       │   │
│  │ (等待调度执行Prefill)    │  │   │  │ (等待KV传输完成)        │   │
│  └──────────┬──────────────┘  │   │  └──────────┬──────────────┘   │
│             ▼                 │   │             ▼                  │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐   │
│  │   Forward (Prefill)     │  │   │  │    WaitingQueue         │   │
│  │ (执行Prefill前向计算)    │  │   │  │ (构建PrebuiltBatch)     │   │
│  └──────────┬──────────────┘  │   │  └──────────┬──────────────┘   │
│             ▼                 │   │             ▼                  │
│  ┌─────────────────────────┐  │   │  ┌─────────────────────────┐   │
│  │   InflightQueue         │  │   │  │    RunningBatch         │   │
│  │ (发送KV,等待传输完成)    │──┼──►│  │ (执行Decode生成)        │   │
│  └─────────────────────────┘  │   │  └─────────────────────────┘   │
│                               │   │                                │
│  KV Cache: 计算后立即释放      │   │  KV Cache: 持续增长,可卸载     │
└───────────────────────────────┘   └───────────────────────────────┘
        │                                       ▲
        │         KV Cache Transfer             │
        │    (Mooncake/NIXL/Mori/Ascend)        │
        └───────────────────────────────────────┘
```

### 2.2 目录结构

```
python/sglang/srt/disaggregation/
├── __init__.py
├── base/
│   └── conn.py                    # 抽象基类定义
├── common/
│   ├── conn.py                    # 通用实现(CommonKVManager等)
│   └── utils.py                   # 通用工具函数
├── nixl/
│   └── conn.py                    # NVIDIA NIXL传输后端
├── mooncake/
│   ├── conn.py                    # Mooncake传输后端(RDMA/NVLink)
│   └── utils.py
├── mori/
│   └── conn.py                    # Mori传输后端
├── ascend/
│   ├── conn.py                    # 华为Ascend NPU传输后端
│   └── transfer_engine.py
├── fake/
│   └── conn.py                    # 测试用Fake后端
├── prefill.py                     # Prefill实例逻辑
├── decode.py                      # Decode实例逻辑
├── decode_kvcache_offload_manager.py  # KV Cache卸载管理
├── decode_schedule_batch_mixin.py      # Prebuilt Batch处理
├── encode_server.py               # 多模态Encoder服务端
├── encode_receiver.py             # 多模态Encoder接收端
├── encode_grpc_server.py          # gRPC Encoder服务端
├── kv_events.py                   # KV Cache事件发布
└── utils.py                       # 共享工具(枚举、元数据缓冲区等)
```

## 3. 核心抽象与接口

### 3.1 基础类定义 (base/conn.py)

#### KVArgs - KV传输配置

```python
class KVArgs:
    engine_rank: int              # 当前引擎Rank
    kv_data_ptrs: List[int]       # KV Cache GPU内存指针
    kv_data_lens: List[int]       # 每个KV缓冲区长度
    kv_item_lens: List[int]       # 每个KV项长度
    aux_data_ptrs: List[int]      # 元数据缓冲区指针
    aux_data_lens: List[int]      # 元数据缓冲区长度
    state_data_ptrs: List[int]    # 状态数据指针(Mamba/SWA)
    state_data_lens: List[int]    # 状态数据长度
    state_item_lens: List[int]    # 状态项长度
    state_type: str               # "none", "mamba", "swa", "nsa"
    state_dim_per_tensor: List[int]  # Mamba状态维度
    ib_device: str                # InfiniBand设备
    ib_traffic_class: str         # IB流量类别
    gpu_id: int                   # GPU ID
    kv_head_num: int              # KV头数
    total_kv_head_num: int        # 总KV头数
    page_size: int                # 页大小
    pp_rank: int                  # 流水线Rank
    prefill_start_layer: int      # Prefill起始层(PP)
    system_dp_rank: int           # 系统DP Rank
```

#### KVPoll - 传输状态机

```python
class KVPoll:
    Failed = 0            # 传输失败
    Bootstrapping = 1     # 握手中
    WaitingForInput = 2   # 等待输入(KV索引)
    Transferring = 3      # 传输中
    Success = 4           # 传输成功
```

状态转换图：
```
Bootstrapping → WaitingForInput → Transferring → Success
       │              │               │
       └──────────────┴───────────────┴──→ Failed
```

#### BaseKVSender / BaseKVReceiver - 抽象接口

```python
class BaseKVSender(ABC):
    def __init__(self, mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank): ...
    def init(self, num_kv_indices, aux_index): ...
    def send(self, kv_indices, state_indices): ...
    def poll(self) -> KVPoll: ...
    def failure_exception(self): ...

class BaseKVReceiver(ABC):
    def __init__(self, mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank): ...
    def init(self, kv_indices, aux_index, state_indices): ...
    def poll(self) -> KVPoll: ...
    def failure_exception(self): ...
    def clear(self): ...
    def abort(self): ...
```

### 3.2 通用实现 (common/conn.py)

#### CommonKVManager

管理传输状态和连接的核心类：

```python
class CommonKVManager(BaseKVManager):
    def __init__(self, args, disaggregation_mode, server_args, is_mla_backend):
        # 并行信息
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_cp_size = get_attention_cp_size()
        self.attn_cp_rank = get_attention_cp_rank()
        self.attn_dp_size = get_attention_dp_size()
        self.attn_dp_rank = get_attention_dp_rank()
        self.pp_size = server_args.pp_size
        self.pp_rank = args.pp_rank

        # ZMQ Socket用于接收Bootstrap信息
        self.rank_port, self.server_socket = get_zmq_socket_on_host(
            context, zmq.PULL, host=self.local_ip
        )

        # 状态追踪
        self.request_status: Dict[int, KVPoll] = {}
        self.failure_records: Dict[int, str] = {}

        # Prefill模式: 注册到Bootstrap Server
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.register_to_bootstrap()
            self.decode_kv_args_table = {}

        # Decode模式: 维护连接池和心跳
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.connection_pool: Dict[str, Dict] = {}
            self.prefill_info_table: Dict[str, PrefillServerInfo] = {}
            self.heartbeat_failures: Dict[str, int] = {}
```

#### CommonKVBootstrapServer

HTTP服务用于服务发现和Rank映射：

```python
class CommonKVBootstrapServer:
    # API端点:
    # PUT /route - 注册Prefill Rank信息
    # GET /route?prefill_dp_rank=X&target_tp_rank=Y - 获取Prefill Rank信息
    # POST /register_dp_rank - 注册请求的DP Rank
    # POST /query_dp_ranks - 批量查询DP Ranks
    # GET /health - 健康检查

    # 数据结构:
    self.prefill_port_table: Dict[dp_group, Dict[cp_rank, Dict[tp_rank, Dict[pp_rank, PrefillRankInfo]]]]
    self.room_to_dp_rank: Dict[bootstrap_room, dp_rank]
```

## 4. Prefill实例详解

### 4.1 请求生命周期

```
HTTP Request → Tokenizer → PrefillBootstrapQueue → WaitingQueue
    → Forward (Prefill) → InflightQueue → KV Transfer → Release → Response
```

### 4.2 PrefillBootstrapQueue

```python
class PrefillBootstrapQueue:
    def __init__(self, token_to_kv_pool, ...):
        self.kv_manager = self._init_kv_manager()
        self.queue: List[Req] = []

    def add(self, req: Req, num_kv_heads: int):
        """为新请求创建KVSender"""
        # 检查请求是否超出KV容量
        if self._check_if_req_exceed_kv_capacity(req):
            return

        # 创建Sender
        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=[self.tp_rank],
            pp_rank=self.pp_rank,
        )
        self.queue.append(req)

    def pop_bootstrapped(self) -> List[Req]:
        """Poll所有Sender,返回完成Bootstrap的请求"""
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.queue],
            self.scheduler.attn_cp_cpu_group,
            self.scheduler.attn_tp_cpu_group,
        )

        for i, (req, poll) in enumerate(zip(self.queue, polls)):
            if poll == KVPoll.Bootstrapping:
                continue
            elif poll == KVPoll.Failed:
                # 处理失败
                prepare_abort(req, error_message)
                continue
            elif poll == KVPoll.WaitingForInput:
                # 分配元数据缓冲区并初始化Sender
                req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
                num_pages = kv_to_page_num(len(req.origin_input_ids), page_size)
                req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
                bootstrapped_reqs.append(req)

        return bootstrapped_reqs
```

### 4.3 SchedulerDisaggregationPrefillMixin

```python
class SchedulerDisaggregationPrefillMixin:
    def event_loop_normal_disagg_prefill(self):
        """Prefill实例的主事件循环"""
        while True:
            # 1. 接收请求
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            # 2. 从Bootstrap队列获取就绪请求
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )

            # 3. 获取下一个待执行的Batch
            batch = self.get_next_disagg_prefill_batch_to_run()

            # 4. 执行Forward
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.self_check_during_idle()

            # 5. 处理传输中的请求
            self.process_disagg_prefill_inflight_queue()

    def send_kv_chunk(self, req: Req, last_chunk: bool = False):
        """发送KV Cache块到Decode实例"""
        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        end_idx = len(req.fill_ids) if last_chunk else end_idx - end_idx % page_size

        # 获取KV索引
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, start_idx:end_idx
        ].cpu().numpy()

        req.start_send_idx = end_idx

        # 准备状态索引(Mamba/SWA模型)
        state_indices = None
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)
            # 处理Hybrid模型的状态...

        # 转换为页索引并发送
        page_indices = kv_to_page_indices(kv_indices, page_size)
        req.disagg_kv_sender.send(page_indices, state_indices)

    def process_disagg_prefill_inflight_queue(self):
        """处理传输中的请求"""
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if poll == KVPoll.Success:
                # 传输完成,释放KV Cache
                release_kv_cache(req, self.tree_cache)
                req.finished_reason = FINISH_LENGTH(length=0)
                done_reqs.append(req)
            elif poll == KVPoll.Failed:
                # 处理失败
                release_kv_cache(req, self.tree_cache)
                prepare_abort(req, error_message)

        # 流式输出完成的请求
        self.stream_output(done_reqs, ...)
```

## 5. Decode实例详解

### 5.1 请求生命周期

```
HTTP Request (with bootstrap_host/room) → PreallocQueue → TransferQueue
    → WaitingQueue → PrebuiltExtendBatch → RunningBatch → Decode Output
```

### 5.2 DecodePreallocQueue

```python
class DecodePreallocQueue:
    def __init__(self, token_to_kv_pool, ...):
        self.kv_manager = self._init_kv_manager()
        self.queue: List[Req] = []
        self.gloo_group = gloo_group

    def add(self, req: Req):
        """为新请求创建KVReceiver"""
        req.disagg_kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{req.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            prefill_dp_rank=prefill_dp_rank,
        )
        self.queue.append(req)

    def pop_preallocated(self) -> List[Req]:
        """Poll所有Receiver,返回完成KV预分配的请求"""
        for req in self.queue:
            poll = req.disagg_kv_receiver.poll()
            if poll == KVPoll.WaitingForInput:
                # 预分配KV槽位
                num_tokens = len(req.origin_input_ids)
                kv_indices = self.token_to_kv_pool_allocator.alloc(num_tokens)
                req.disagg_kv_receiver.init(kv_indices, aux_index, state_indices)
                preallocated_reqs.append(req)
            elif poll == KVPoll.Failed:
                # 处理失败
                prepare_abort(req, error_message)

        return preallocated_reqs
```

### 5.3 DecodeTransferQueue

```python
class DecodeTransferQueue:
    def pop_transferred(self) -> List[Req]:
        """Poll所有Receiver,返回完成KV传输的请求"""
        transferred_reqs = []
        undone_reqs = []

        for req in self.queue:
            poll = req.disagg_kv_receiver.poll()
            if poll == KVPoll.Success:
                # 传输完成
                transferred_reqs.append(req)
            elif poll == KVPoll.Failed:
                # 处理失败
                prepare_abort(req, error_message)
            else:
                undone_reqs.append(req)

        self.queue = undone_reqs
        return transferred_reqs
```

### 5.4 SchedulerDisaggregationDecodeMixin

```python
class SchedulerDisaggregationDecodeMixin:
    def event_loop_normal_disagg_decode(self):
        """Decode实例的主事件循环"""
        while True:
            # 1. 接收请求
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            # 2. 处理Prealloc队列
            self.process_decode_prealloc_queue()

            # 3. 处理Transfer队列
            self.process_decode_transfer_queue()

            # 4. 获取下一个待执行的Batch
            batch = self.get_next_decode_batch_to_run()

            # 5. 执行Decode Forward
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)

            # 6. 检查KV卸载进度(如果启用)
            if self.enable_decode_offload:
                self.decode_offload_manager.check_offload_progress()

    def process_decode_prealloc_queue(self):
        """处理Prealloc队列"""
        # 获取完成Bootstrap的请求
        bootstrapped = self.decode_prealloc_queue.pop_bootstrapped()

        for req in bootstrapped:
            # 预分配KV槽位
            num_tokens = len(req.origin_input_ids)
            kv_indices = self.token_to_kv_pool_allocator.alloc(num_tokens)
            # 初始化Receiver
            req.disagg_kv_receiver.init(kv_indices, aux_index, state_indices)
            # 移动到Transfer队列
            self.decode_transfer_queue.add(req)

    def process_decode_transfer_queue(self):
        """处理Transfer队列"""
        transferred = self.decode_transfer_queue.pop_transferred()

        for req in transferred:
            # 从元数据缓冲区读取第一个输出token
            self.disagg_metadata_buffers.get_buf(req)
            # 设置请求状态
            req.output_ids.append(first_output_token)
            # 移动到Waiting队列
            self.waiting_queue.append(req)
```

## 6. KV Cache传输机制

### 6.1 传输后端类型

| 后端 | 描述 | 适用场景 |
|------|------|----------|
| Mooncake | RDMA/NVLink传输 | 高吞吐量、低延迟 |
| NIXL | NVIDIA UCX后端 | NVIDIA GPU集群 |
| Mori | 自定义RDMA后端 | 特定硬件优化 |
| Ascend | 华为Ascend NPU | Ascend硬件 |
| Fake | 测试用虚拟后端 | 单元测试 |

### 6.2 Mooncake实现详解

Mooncake是默认的传输后端,支持RDMA和NVLink:

```python
class MooncakeKVSender(CommonKVSender):
    def __init__(self, mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank):
        self.engine = get_mooncake_transfer_engine()
        self.bootstrap_room = bootstrap_room
        self.transfer_queue = FastQueue(maxsize=queue_size)

    def send(self, kv_indices, state_indices):
        """发送KV Cache块"""
        # 准备传输块
        chunk = TransferKVChunk(
            room=self.bootstrap_room,
            prefill_kv_indices=kv_indices,
            index_slice=slice(0, len(kv_indices)),
            is_last_chunk=True,
        )
        self.transfer_queue.put(chunk)

    def poll(self) -> KVPoll:
        """检查传输状态"""
        return self.kv_mgr.check_status(self.bootstrap_room)

class MooncakeKVReceiver(CommonKVReceiver):
    def init(self, kv_indices, aux_index, state_indices):
        """初始化接收端,注册本地缓冲区"""
        # 注册本地KV缓冲区到传输引擎
        self.engine.register_local_buffer(kv_indices)

        # 发送Bootstrap信息到Prefill
        self._send_bootstrap_info_to_prefill()

    def poll(self) -> KVPoll:
        """检查传输状态"""
        return self.kv_mgr.check_status(self.bootstrap_room)
```

### 6.3 元数据缓冲区 (MetadataBuffers)

第一个输出token的元数据与KV Cache一起传输:

```python
class MetadataBuffers:
    def __init__(self, size, hidden_size, hidden_states_dtype):
        # 输出token ID (padding到64字节以上)
        self.output_ids = torch.zeros((size, 16), dtype=torch.int32)
        # 缓存的token数
        self.cached_tokens = torch.zeros((size, 16), dtype=torch.int32)
        # 输出token的logprob
        self.output_token_logprobs_val = torch.zeros((size, 16), dtype=torch.float32)
        self.output_token_logprobs_idx = torch.zeros((size, 16), dtype=torch.int32)
        # Top logprobs
        self.output_top_logprobs_val = torch.zeros((size, max_top_logprobs_num), dtype=torch.float32)
        self.output_top_logprobs_idx = torch.zeros((size, max_top_logprobs_num), dtype=torch.int32)
        # Speculative Decoding相关
        self.output_topk_p = torch.zeros((size, 16), dtype=torch.float32)
        self.output_topk_index = torch.zeros((size, 16), dtype=torch.int64)
        self.output_hidden_states = torch.zeros((size, hidden_size), dtype=hidden_states_dtype)
        # 请求验证
        self.bootstrap_room = torch.zeros((size, 8), dtype=torch.uint64)
```

## 7. Bootstrap Server详解

### 7.1 核心功能

Bootstrap Server是PD分离的协调中心,负责:

1. **服务注册**: Prefill实例注册自己的Rank信息
2. **服务发现**: Decode实例发现并连接Prefill实例
3. **DP Rank映射**: 将请求路由到正确的Prefill DP Rank
4. **健康检查**: 监控Prefill实例状态

### 7.2 API端点

```
PUT /route
    - 注册Prefill Rank信息
    - Payload: {
        attn_tp_size, attn_tp_rank, attn_cp_size, attn_cp_rank,
        attn_dp_size, attn_dp_rank, pp_size, pp_rank,
        system_dp_size, system_dp_rank, rank_ip, rank_port,
        page_size, kv_cache_dtype, load_balance_method
    }

GET /route?prefill_dp_rank=X&prefill_cp_rank=Y&target_tp_rank=Z&target_pp_rank=W
    - 获取指定Rank的连接信息
    - Response: { rank_ip, rank_port }

POST /register_dp_rank
    - 注册请求的DP Rank (用于负载均衡)
    - Payload: { bootstrap_room, dp_rank }

POST /query_dp_ranks
    - 批量查询DP Ranks
    - Payload: { bootstrap_rooms: [room1, room2, ...] }
    - Response: { room1: dp_rank1, room2: dp_rank2, ... }

GET /health
    - 健康检查
```

### 7.3 数据结构

```python
# Prefill Rank信息表 (嵌套字典)
prefill_port_table: {
    dp_group: {
        cp_rank: {
            tp_rank: {
                pp_rank: PrefillRankInfo(rank_ip, rank_port)
            }
        }
    }
}

# Bootstrap Room到DP Rank的映射
room_to_dp_rank: {
    bootstrap_room: {
        dp_rank: int,
        timestamp: float  # 用于过期清理
    }
}
```

## 8. KV Cache卸载管理

### 8.1 三层内存架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   GPU Memory    │ ←→  │   Host Memory   │ ←→  │    Storage      │
│   (KV Cache)    │     │    (Pinned)     │     │  (Disk/Network) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ↑                       ↑                       ↑
        │                       │                       │
        +───────────────────────+───────────────────────┘
                                │
                    HiCacheController / DecodeKVCacheOffloadManager
```

### 8.2 DecodeKVCacheOffloadManager

```python
class DecodeKVCacheOffloadManager:
    def __init__(self, req_to_token_pool, token_to_kv_pool_allocator, ...):
        # Host内存池
        self.decode_host_mem_pool = MHATokenToKVPoolHost(...)
        # Cache控制器
        self.cache_controller = HiCacheController(...)
        # 追踪状态
        self.ongoing_offload = {}      # ack_id -> (req, host_indices, ...)
        self.ongoing_backup = {}       # 存储备份追踪
        self.offloaded_state = {}      # rid -> OffloadedState

    def offload_kv_cache(self, req) -> bool:
        """卸载增量KV Cache从GPU到Host"""
        # 计算增量tokens
        all_tokens = req.origin_input_ids + req.output_ids[:-1]
        incremental_tokens = all_tokens[state.prefill_len + state.inc_len : end]

        # 异步从GPU卸载到Host
        host_indices = self.cache_controller.write(
            device_indices=incremental_indices.long(),
            node_id=ack_id,
        )

        # 卸载完成后释放GPU内存
        self.token_to_kv_pool_allocator.free(kv_indices)

        return True

    def check_offload_progress(self):
        """检查卸载进度并触发存储备份"""
        # 检查GPU->Host卸载
        self._check_offload_progress(n_write)
        # 检查Host->Storage备份
        self._check_backup_progress(n_backup)

    def _trigger_backup(self, req, host_indices, incremental_tokens, ...):
        """触发从Host到Storage的异步备份"""
        page_hashes = self._compute_prefix_hash(incremental_tokens, prior_hash)
        ack_id = self.cache_controller.write_storage(
            host_indices, incremental_tokens, hash_value=page_hashes
        )
        return page_hashes[-1]
```

### 8.3 卸载流程

```
1. 检测到内存压力 (OOM或接近OOM)
2. 选择要卸载的请求 (基于策略)
3. 异步卸载KV Cache: GPU → Host
4. 释放GPU内存
5. 可选: 异步备份到Storage: Host → Storage
6. 释放Host内存
```

## 9. 并行策略支持

### 9.1 Tensor Parallelism (TP)

PD分离支持不同TP大小的Prefill和Decode实例:

```python
# 场景1: Decode TP > Prefill TP
# 一个Decode Rank需要从多个Prefill Rank获取KV Cache
if self.kv_mgr.attn_tp_size > self.prefill_info.attn_tp_size:
    self.target_tp_ranks = [rank for rank in range(...)]
    self.required_prefill_response_num = (
        self.kv_mgr.attn_tp_size // self.prefill_info.attn_tp_size
    )

# 场景2: Prefill TP > Decode TP
# 一个Decode Rank需要从多个Prefill Rank获取部分KV Cache
elif self.kv_mgr.attn_tp_size < self.prefill_info.attn_tp_size:
    self.target_tp_ranks = [
        rank for rank in range(
            self.kv_mgr.kv_args.engine_rank * ratio,
            (self.kv_mgr.kv_args.engine_rank + 1) * ratio
        )
    ]
```

### 9.2 Context Parallelism (CP)

```python
# CP Rank过滤页面索引
def page_indices_to_cp_rank_page_indices(page_indices, total_pages, cp_rank, cp_size):
    """过滤页面索引到当前CP Rank"""
    if cp_size <= 1:
        return page_indices

    # 计算本地范围
    base = total_pages // cp_size
    rem = total_pages % cp_size
    local_start = cp_rank * base + min(cp_rank, rem)
    n_pages = base + (1 if cp_rank < rem else 0)

    # 映射回全局页ID
    start_page = first_page + local_start
    end_page = start_page + n_pages

    return page_indices[(page_indices >= start_page) & (page_indices < end_page)]
```

### 9.3 Pipeline Parallelism (PP)

PP支持通过层级切分:

```python
# PP信息存储在KVArgs中
kv_args.pp_rank = self.pp_rank
kv_args.prefill_start_layer = self.token_to_kv_pool.start_layer

# 获取当前PP阶段的KV指针
def get_mha_kv_ptrs_with_pp(self, src_kv_ptrs, dst_kv_ptrs):
    start_layer = self.kv_args.prefill_start_layer
    num_kv_layers = len(src_kv_ptrs) // 2
    end_layer = start_layer + num_kv_layers

    src_k_ptrs = src_kv_ptrs[:num_kv_layers]
    src_v_ptrs = src_kv_ptrs[num_kv_layers:]
    dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
    dst_v_ptrs = dst_kv_ptrs[dst_num_total_layers + start_layer : ...]
```

### 9.4 Data Parallelism (DP)

DP用于负载均衡:

```python
# Prefill注册DP Rank
def _register_prefill_dp_rank(self):
    url = f"http://{self.bootstrap_server_url}/register_dp_rank"
    payload = {
        "bootstrap_room": self.bootstrap_room,
        "dp_rank": self.kv_mgr.attn_dp_rank,
    }
    requests.post(url, json=payload)

# Decode查询DP Rank
def query_prefill_dp_ranks(bootstrap_addr, bootstrap_rooms):
    url = f"http://{bootstrap_addr}/query_dp_ranks"
    response = requests.post(url, json={"bootstrap_rooms": bootstrap_rooms})
    return response.json()
```

## 10. 配置参数

### 10.1 服务器参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--disaggregation-mode` | `null` | PD分离模式: `null`, `prefill`, `decode` |
| `--disaggregation-transfer-backend` | `mooncake` | 传输后端: `mooncake`, `nixl`, `mori`, `ascend` |
| `--disaggregation-bootstrap-port` | `8998` | Bootstrap Server端口 |
| `--disaggregation-ib-device` | `None` | InfiniBand设备(逗号分隔) |
| `--disaggregation-decode-enable-offload-kvcache` | `False` | 启用Decode端KV卸载 |
| `--disaggregation-decode-polling-interval` | `1` | Decode轮询间隔 |
| `--num-reserved-decode-tokens` | `512` | 保留用于卸载的token数 |

### 10.2 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | auto | 传输线程池大小 |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | `4` | 传输队列大小 |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | `300` | Bootstrap超时(秒) |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | `5.0` | 心跳间隔 |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` | `2` | 最大心跳失败次数 |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | `300` | 传输等待超时 |
| `SGLANG_DISAGGREGATION_NIXL_BACKEND` | `UCX` | NIXL后端类型 |
| `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER` | `False` | 所有CP Rank参与传输 |

## 11. 启动示例

### 11.1 单节点配置

```bash
# 启动Prefill实例 (TP=2)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --disaggregation-mode prefill \
    --tp 2 \
    --port 30000 \
    --disaggregation-bootstrap-port 8998

# 启动Decode实例 (TP=2)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --disaggregation-mode decode \
    --tp 2 \
    --port 30001 \
    --dist-init-addr localhost:8998 \
    --disaggregation-bootstrap-port 8998
```

### 11.2 多节点配置

```bash
# Prefill节点 (节点1)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --disaggregation-mode prefill \
    --tp 8 \
    --host 0.0.0.0 \
    --port 30000 \
    --disaggregation-bootstrap-port 8998

# Decode节点 (节点2)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --disaggregation-mode decode \
    --tp 4 \
    --host 0.0.0.0 \
    --port 30001 \
    --dist-init-addr prefill-node:8998 \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-decode-enable-offload-kvcache
```

### 11.3 不同TP大小配置

```bash
# Prefill: TP=4, 高吞吐量
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --disaggregation-mode prefill \
    --tp 4 \
    ...

# Decode: TP=2, 低延迟
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --disaggregation-mode decode \
    --tp 2 \
    ...
```

## 12. 混合模型支持

### 12.1 Mamba Hybrid模型

```python
# Mamba状态传输
if isinstance(token_to_kv_pool, HybridLinearKVPool):
    kv_args.state_type = "mamba"
    state_indices = [
        req_to_token_pool.req_index_to_mamba_index_mapping[req.req_pool_idx]
    ]
```

### 12.2 Sliding Window Attention (SWA)

```python
# SWA窗口传输
if isinstance(token_to_kv_pool, SWAKVPool):
    kv_args.state_type = "swa"
    # 只传输最后窗口内的KV
    window_kv_indices = req_to_token_pool.req_to_token[
        req.req_pool_idx, window_start:seq_len
    ]
    window_kv_indices_swa = translate_loc_from_full_to_swa(window_kv_indices)
    state_indices = kv_to_page_indices(window_kv_indices_swa, page_size)
```

### 12.3 Native Sparse Attention (NSA)

```python
# NSA传输
if isinstance(token_to_kv_pool, NSATokenToKVPool):
    kv_args.state_type = "nsa"
    kv_indices_full = req_to_token_pool.req_to_token[req.req_pool_idx, :seq_len]
    state_indices = kv_to_page_indices(kv_indices_full, page_size)
```

## 13. 故障处理

### 13.1 Bootstrap失败

```python
# Prefill端检测
if poll == KVPoll.Failed:
    error_message = f"Prefill bootstrap failed for request {req.rid}"
    req.disagg_kv_sender.failure_exception()  # 抛出详细异常
    prepare_abort(req, error_message, status_code=500)

# Decode端检测
if not self.kv_mgr.ensure_parallel_info(self.bootstrap_addr):
    self.kv_mgr.record_failure(self.bootstrap_room, "Could not fetch prefill info")
    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
```

### 13.2 传输失败

```python
# 重试机制
for attempt in range(max_retries):
    info = self._fetch_prefill_server_info(bootstrap_addr)
    if info is not None:
        break
    time.sleep(retry_interval)

# 超时处理
if time.time() - start_time > self.bootstrap_timeout:
    prepare_abort(req, "Bootstrap timeout")
```

### 13.3 心跳检测

```python
# Decode端心跳
self.heartbeat_interval = max(envs.SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL, 2.0)
self.max_failures = max(envs.SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE, 1)

# 失败计数
if heartbeat_failed:
    self.heartbeat_failures[bootstrap_addr] += 1
    if self.heartbeat_failures[bootstrap_addr] >= self.max_failures:
        # 标记为失败,触发重连或中止
        ...
```

## 14. 性能优化

### 14.1 Overlap调度

```python
def event_loop_overlap_disagg_prefill(self):
    """Overlap模式: Prefill下一批时传输当前批的KV"""
    while True:
        # 获取下一批
        batch = self.get_next_disagg_prefill_batch_to_run()

        # 执行Forward
        if batch:
            batch_result = self.run_batch(batch)
            self.result_queue.append((batch.copy(), batch_result))

        # 处理上一批(同时进行KV传输)
        if self.last_batch:
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        # 处理传输队列
        self.process_disagg_prefill_inflight_queue()
```

### 14.2 批量KV传输

```python
# 批量发送KV块
def send_kv_chunks(self, reqs: List[Req]):
    """批量发送多个请求的KV"""
    for req in reqs:
        if req.is_chunked <= 0:
            # 最后一块,发送全部剩余
            self.send_kv_chunk(req, last_chunk=True)
        else:
            # 分块发送
            req.is_chunked -= 1
            self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
```

### 14.3 内存池优化

```python
# 使用CUDA Memory Pool优化
if custom_mem_pool:
    with torch.cuda.use_mem_pool(custom_mem_pool):
        # 分配元数据缓冲区
        self.metadata_buffers = MetadataBuffers(...)

# 页对齐优化
end_idx = end_idx - end_idx % page_size  # 对齐到页边界
```

## 15. 调试与监控

### 15.1 日志级别

```bash
# 启用详细日志
export SGLANG_LOG_LEVEL_HTTP=DEBUG
export SGLANG_LOG_LEVEL_SERVER=DEBUG
export SGLANG_LOG_LEVEL_FLASHINFERENCE=DEBUG
```

### 15.2 关键日志点

```python
# Bootstrap注册
logger.debug(f"Register prefill bootstrap: DP{dp_group} CP{cp_rank} TP{tp_rank} PP{pp_rank}")

# 传输状态
logger.debug(f"KV transfer state: room={room}, poll={poll}")

# 卸载进度
logger.info(f"Finished backup request {req_id}, cost time: {time:.2f}s")
```

### 15.3 指标收集

```python
# 传输延迟和速度
metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
    num_tokens=len(req.origin_input_ids),
    page_size=page_size,
    bytes_per_page_all_layers=bytes_per_page,
)
self.kv_transfer_latency_ms = metrics["latency_ms"]
self.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

# Bootstrap失败计数
if self.scheduler.enable_metrics:
    self.scheduler.metrics_collector.increment_bootstrap_failed_reqs()
    self.scheduler.metrics_collector.increment_transfer_failed_reqs()
```

## 16. 最佳实践

### 16.1 容量规划

| 配置 | Prefill | Decode | 说明 |
|------|---------|--------|------|
| 小规模 | TP=2 | TP=2 | 单节点部署 |
| 中规模 | TP=4-8 | TP=2-4 | 多节点部署 |
| 大规模 | TP=8+ | TP=4+ | 大模型部署 |

### 16.2 网络配置

```bash
# 高性能网络
--disaggregation-ib-device "mlx5_0,mlx5_1"
--disaggregation-transfer-backend mooncake

# 调整队列大小
export SGLANG_DISAGGREGATION_QUEUE_SIZE=16
export SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=8
```

### 16.3 内存优化

```bash
# Decode端启用KV卸载
--disaggregation-decode-enable-offload-kvcache
--mem-fraction-static 0.7  # 为卸载预留空间

# 调整卸载stride
export SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE=128
```

## 17. 总结

SGLang的PD分离架构通过以下关键设计实现了高性能的大语言模型推理：

1. **清晰的职责分离**: Prefill专注计算,Decode专注生成
2. **灵活的并行支持**: 支持TP/CP/PP/DP多种并行策略组合
3. **高效的KV传输**: 支持多种RDMA/NVLink传输后端
4. **完善的故障处理**: Bootstrap、传输、心跳多层级容错
5. **智能的内存管理**: 三层架构(GPU→Host→Storage)实现高效KV Cache管理
6. **混合模型支持**: 支持Mamba、SWA、NSA等先进架构

该架构特别适合大规模部署场景,可以根据实际负载灵活调整Prefill和Decode实例的数量和配置,实现最优的资源利用率和吞吐量。
