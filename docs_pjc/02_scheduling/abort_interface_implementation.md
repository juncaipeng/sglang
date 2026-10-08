# SGLang Abort接口实现方案

## 1. 概述

Abort接口是LLM服务框架中的关键功能，用于取消正在执行的推理请求并释放相关资源。SGLang提供了完善的abort机制，支持：

- **单个请求abort**：通过请求ID（rid）取消特定请求
- **批量abort**：通过`abort_all`参数取消所有请求
- **自动abort**：客户端断连、超时等场景自动触发

### 整体架构流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Abort Request Flow                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HTTP Client                                                                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────┐                                                        │
│  │  HTTP Server    │  POST /abort_request                                   │
│  │  (http_server)  │                                                        │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │ TokenizerManager│  abort_request()                                       │
│  │                 │  - 检查rid是否存在                                      │
│  │                 │  - 发送AbortReq到Scheduler                             │
│  │                 │  - 记录metrics                                         │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │   Scheduler     │  abort_request()                                       │
│  │                 │  ┌─────────────────────────────────────────────────┐   │
│  │                 │  │ Method 1: Waiting Queue                        │   │
│  │                 │  │   - 直接弹出队列                                │   │
│  │                 │  │   - 释放KV Cache                               │   │
│  │                 │  │   - 发送响应到TokenizerManager                 │   │
│  │                 │  ├─────────────────────────────────────────────────┤   │
│  │                 │  │ Method 2: Grammar Queue                        │   │
│  │                 │  │   - 调用set_finish_with_abort                  │   │
│  │                 │  │   - 请求执行轻量prefill后结束                   │   │
│  │                 │  ├─────────────────────────────────────────────────┤   │
│  │                 │  │ Method 3: Running Batch                        │   │
│  │                 │  │   - 设置to_finish标记                          │   │
│  │                 │  │   - 请求执行一次decode后清理                    │   │
│  │                 │  └─────────────────────────────────────────────────┘   │
│  └─────────────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. HTTP API层

### 2.1 端点定义

**文件：** [http_server.py](python/sglang/srt/entrypoints/http_server.py)

```python
@app.post("/abort_request")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def abort_request(obj: AbortReq, request: Request):
    """Abort a request."""
    try:
        _global_state.tokenizer_manager.abort_request(
            rid=obj.rid, abort_all=obj.abort_all
        )
        return Response(status_code=200)
    except Exception as e:
        return _create_error_response(e)
```

**关键点：**
- 使用`POST`方法
- 认证级别为`ADMIN_OPTIONAL`，允许管理员级别的可选认证
- 成功返回HTTP 200状态码
- 异常时返回错误响应

### 2.2 AbortReq数据结构

**文件：** [io_struct.py](python/sglang/srt/managers/io_struct.py)

```python
@dataclass
class AbortReq(BaseReq):
    # Whether to abort all requests
    abort_all: bool = False
    # The finished reason data
    finished_reason: Optional[Dict[str, Any]] = None
    abort_message: Optional[str] = None

    def __post_init__(self):
        # FIXME: This is a hack to keep the same with the old code
        if self.rid is None:
            self.rid = ""
```

**字段说明：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rid` | str | "" | 请求ID，继承自BaseReq |
| `abort_all` | bool | False | 是否abort所有请求 |
| `finished_reason` | Dict | None | 自定义的结束原因 |
| `abort_message` | str | None | abort消息 |

### 2.3 使用示例

```bash
# Abort单个请求
curl -X POST http://localhost:30000/abort_request \
  -H "Content-Type: application/json" \
  -d '{"rid": "request-123"}'

# Abort所有请求
curl -X POST http://localhost:30000/abort_request \
  -H "Content-Type: application/json" \
  -d '{"abort_all": true}'
```

---

## 3. TokenizerManager层

### 3.1 abort_request方法

**文件：** [tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py)

```python
def abort_request(self, rid: str = "", abort_all: bool = False):
    if not abort_all and rid not in self.rid_to_state:
        return
    req = AbortReq(rid=rid, abort_all=abort_all)
    self.send_to_scheduler.send_pyobj(req)
    if self.enable_metrics:
        # TODO: also use custom_labels from the request
        self.metrics_collector.observe_one_aborted_request(
            self.metrics_collector.labels
        )
```

**处理流程：**
1. 检查请求是否存在（非abort_all时）
2. 创建AbortReq对象
3. 通过ZMQ发送到Scheduler
4. 记录metrics指标

### 3.2 客户端断连检测

SGLang会自动检测客户端断连并触发abort。针对不同场景有三种处理方式：

#### Type 1: 非流式请求在Waiting Queue

```python
# 在handle_generate_request_loop中
if (
    request is not None
    and not obj.background
    and await request.is_disconnected()
):
    # Abort the request for disconnected requests (non-streaming, waiting queue)
    self.abort_request(obj.rid)
    # Use exception to kill the whole call stack and asyncio task
    raise ValueError(
        f"Request is disconnected from the client side (type 1). Abort request {obj.rid=}"
    )
```

#### Type 3: 非流式请求在Running Batch

类似的检测机制用于running batch中的请求。

#### 流式请求的Background Abort Task

```python
def create_abort_task(self, obj: GenerateReqInput):
    # Abort the request if the client is disconnected.
    async def abort_request():
        await asyncio.sleep(2)
        if obj.is_single:
            self.abort_request(obj.rid)
        else:
            for rid in obj.rid:
                self.abort_request(rid)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(abort_request)
    return background_tasks
```

### 3.3 _handle_abort_req响应处理

**文件：** [tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py)

```python
def _handle_abort_req(self, recv_obj: AbortReq):
    if is_health_check_generate_req(recv_obj):
        return
    state = self.rid_to_state[recv_obj.rid]
    state.finished = True
    state.time_stats.set_finished_time()

    abort_message = recv_obj.abort_message or "Abort in waiting queue"
    finish_reason = {
        "type": "abort",
        "message": abort_message,
    }
    if recv_obj.finished_reason:
        finish_reason = recv_obj.finished_reason
    meta_info = {
        "id": recv_obj.rid,
        "finish_reason": finish_reason,
        "weight_version": self.server_args.weight_version,
        "e2e_latency": state.time_stats.get_e2e_latency(),
    }
    is_stream = getattr(state.obj, "stream", False)
    if getattr(state.obj, "return_logprob", False):
        self.add_logprob_to_meta_info(...)

    output_ids = state.output_ids
    meta_info["completion_tokens"] = len(output_ids)
    # ... 继续处理stream或non-stream输出
```

---

## 4. Scheduler层（核心）

### 4.1 abort_request主方法

**文件：** [scheduler.py](python/sglang/srt/managers/scheduler.py)

Scheduler的`abort_request`方法是整个abort机制的核心，实现了三种abort方法：

```python
def abort_request(self, recv_req: AbortReq):
    # ========== Method 1: Waiting Queue ==========
    # Delete requests in the waiting queue
    to_del = []
    for i, req in enumerate(self.waiting_queue):
        if recv_req.abort_all or req.rid.startswith(recv_req.rid):
            to_del.append(i)

    # Sort in reverse order to avoid index issues when deleting
    for i in reversed(to_del):
        # Abort method 1: directly pop from the queue
        # This only works for requests that have not started anything.
        req = self.waiting_queue.pop(i)
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)
        self.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)

        # Disaggregation mode cleanup
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            release_kv_cache(req, self.tree_cache)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            release_req_to_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
        if req.mamba_pool_idx is not None:
            release_kv_cache(req, self.tree_cache, is_insert=False)
        logger.debug(f"Abort queued request. {req.rid=}")

    # ========== Method 2: Grammar Queue ==========
    # Abort method 2: call `set_finish_with_abort`
    self.grammar_manager.abort_requests(recv_req)

    # ========== Disaggregation Mode Abort ==========
    # (详见第7节)

    # ========== Method 3: Running Batch ==========
    if self.cur_batch is self.running_batch or self.cur_batch is None:
        reqs = self.running_batch.reqs
    else:
        reqs = self.running_batch.reqs + self.cur_batch.reqs

    for req in reqs:
        if not req.finished() and (
            recv_req.abort_all or req.rid.startswith(recv_req.rid)
        ):
            # Abort method 3: set `to_finish`
            # The request will still run one decode forward pass.
            # Then we reuse all existing code to clean up the KV cache allocation.
            logger.debug(f"Abort running request. {req.rid=}")
            req.to_finish = FINISH_ABORT()
```

### 4.2 三种Abort方法对比

| 方法 | 适用场景 | 实现方式 | 性能影响 |
|------|----------|----------|----------|
| **Method 1** | Waiting Queue中的请求 | 直接从队列弹出 | 无，立即释放资源 |
| **Method 2** | Grammar Queue中的请求 | 设置`to_finish`，执行轻量prefill | 一次轻量forward pass |
| **Method 3** | Running Batch中的请求 | 设置`to_finish`，执行一次decode | 一次decode forward pass |

### 4.3 请求匹配规则

```python
# 匹配规则：abort_all 或 rid前缀匹配
if recv_req.abort_all or req.rid.startswith(recv_req.rid):
    # abort this request
```

这意味着：
- `abort_all=True`：abort所有请求
- `rid="req-"`：abort所有以"req-"开头的请求
- `rid="req-123"`：只abort rid为"req-123"的请求

---

## 5. 请求状态变化

### 5.1 FINISH_ABORT类

**文件：** [schedule_batch.py](python/sglang/srt/managers/schedule_batch.py)

```python
class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }
```

### 5.2 to_finish字段

```python
# If we want to abort the request in the middle of the event loop,
# set to_finish instead of directly setting finished_reason.
# Note: We should never set finished_reason in the middle, the req will get filtered and never respond
self.to_finish: Optional[BaseFinishReason] = None
```

**关键设计原则：**
- 不直接设置`finished_reason`，因为请求会被过滤导致无法响应
- 使用`to_finish`标记，让请求完成当前forward pass后再清理

### 5.3 set_finish_with_abort方法

**文件：** [schedule_batch.py](python/sglang/srt/managers/schedule_batch.py)

```python
def set_finish_with_abort(self, error_msg: str):
    if get_tensor_model_parallel_rank() == 0:
        logger.error(f"{error_msg}, {self.rid=}")
    self.multimodal_inputs = None
    self.grammar = None
    self.origin_input_ids = [0]  # set it to one token to skip the long prefill
    self.return_logprob = False
    self.logprob_start_len = -1
    self.to_finish = FINISH_ABORT(
        error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
    )
```

**优化措施：**
- 将`origin_input_ids`设为单个token，跳过长prefill
- 清理`multimodal_inputs`和`grammar`
- 禁用`return_logprob`减少计算

---

## 6. 资源清理

### 6.1 KV Cache释放

**文件：** [common.py](python/sglang/srt/mem_cache/common.py)

```python
def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    # MambaRadixCache may alloc mamba state before alloc KV cache
    if req.req_pool_idx is None:
        assert (
            tree_cache.supports_mamba()
        ), "Only MambaRadixCache allow freeing before alloc"
        if req.mamba_pool_idx is not None:
            tree_cache.req_to_token_pool.mamba_pool.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        return

    tree_cache.cache_finished_req(req, is_insert=is_insert)

    # SessionAwareCache special handling...
    if req.req_pool_idx is None:
        return

    start_p, end_p = req.pop_overallocated_kv_cache()
    # ... 释放over-allocated tokens
```

### 6.2 HiCache Storage Abort

**文件：** [hiradix_cache.py](python/sglang/srt/mem_cache/hiradix_cache.py)

```python
def release_aborted_request(self, rid: str):
    # Clean up storage hit tracking for aborted request
    self.prefetch_loaded_tokens_by_reqid.pop(rid, None)

    if rid not in self.ongoing_prefetch:
        return

    last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]
    if operation.host_indices is None:
        return

    completed_tokens, _ = self.cache_controller.terminate_prefetch(operation)
    if self.tp_world_size > 1:
        torch.distributed.barrier(group=self.tp_group)
    last_host_node.release_host()
    del self.ongoing_prefetch[rid]
    self.cache_controller.append_host_mem_release(host_indices[:completed_tokens])
    self.cache_controller.prefetch_tokens_occupied -= len(token_ids)
```

### 6.3 Grammar Manager Abort

**文件：** [grammar_manager.py](python/sglang/srt/constrained/grammar_manager.py)

```python
def abort_requests(self, recv_req: AbortReq):
    for req in self.grammar_queue:
        if recv_req.abort_all or req.rid.startswith(recv_req.rid):
            logger.debug(f"Abort grammar queue request. {req.rid=}")
            if req.grammar:
                req.grammar.cancel()
            req.set_finish_with_abort("Aborted by AbortReq.")
```

---

## 7. Disaggregation模式Abort

PD disaggregation模式下，Prefill和Decode实例分别运行，abort处理有所不同。

### 7.1 Prefill实例Abort

```python
if self.disaggregation_mode == DisaggregationMode.PREFILL:
    # Abort requests that have not yet been bootstrapped
    for req in self.disagg_prefill_bootstrap_queue.queue:
        if recv_req.abort_all or req.rid.startswith(recv_req.rid):
            logger.debug(f"Abort bootstrap queue request. {req.rid=}")
            if hasattr(req.disagg_kv_sender, "abort"):
                req.disagg_kv_sender.abort()

    # Abort in-flight requests
    for req in self.disagg_prefill_inflight_queue:
        if recv_req.abort_all or req.rid.startswith(recv_req.rid):
            logger.debug(f"Abort inflight queue request. {req.rid=}")
            if hasattr(req.disagg_kv_sender, "abort"):
                req.disagg_kv_sender.abort()
```

**涉及的队列：**
- `disagg_prefill_bootstrap_queue`：等待bootstrap的请求
- `disagg_prefill_inflight_queue`：正在传输KV cache的请求

### 7.2 Decode实例Abort

```python
elif self.disaggregation_mode == DisaggregationMode.DECODE:
    # Abort requests that have not yet finished preallocation
    for decode_req in self.disagg_decode_prealloc_queue.queue:
        if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
            logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
            decode_req.kv_receiver.abort()

    # Abort requests waiting for kvcache to release tree cache
    for decode_req in self.disagg_decode_transfer_queue.queue:
        if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
            logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
            decode_req.kv_receiver.abort()

    # Abort requests already retracted to CPU cache
    if self.disagg_decode_prealloc_queue.retracted_queue:
        remaining_retracted = []
        for decode_req in self.disagg_decode_prealloc_queue.retracted_queue:
            if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):
                assert hasattr(decode_req, "kv_cache_cpu")
                del decode_req.kv_cache_cpu  # 释放CPU上的KV cache
                self.send_to_tokenizer.send_output(
                    AbortReq(rid=decode_req.rid), decode_req
                )
            else:
                remaining_retracted.append(decode_req)
        self.disagg_decode_prealloc_queue.retracted_queue = remaining_retracted
```

**涉及的队列：**
- `disagg_decode_prealloc_queue`：预分配队列
- `disagg_decode_transfer_queue`：传输队列
- `retracted_queue`：被retract到CPU cache的请求

### 7.3 KV Sender/Receiver Abort接口

**文件：** [conn.py](python/sglang/srt/disaggregation/base/conn.py)

```python
def abort(self):
    """
    Abort the current transfer.
    """
    pass
```

---

## 8. 超时Abort机制

### 8.1 Waiting Timeout

**环境变量：** `SGLANG_REQ_WAITING_TIMEOUT`

```python
def _abort_on_waiting_timeout(self):
    if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) <= 0:
        return

    deleted_reqs = set()
    deadline = time.perf_counter() - timeout_s
    for req in self.waiting_queue:
        entry_time = req.time_stats.wait_queue_entry_time
        if 0 < entry_time < deadline:
            if self.enable_hicache_storage:
                self.tree_cache.release_aborted_request(req.rid)
            self.send_to_tokenizer.send_output(
                AbortReq(
                    finished_reason={
                        "type": "abort",
                        "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                        "message": "Request waiting timeout reached.",
                    },
                    rid=req.rid,
                ),
                req,
            )
            deleted_reqs.add(req)
    # ... 从waiting_queue中删除
```

### 8.2 Running Timeout

**环境变量：** `SGLANG_REQ_RUNNING_TIMEOUT`

```python
def _abort_on_running_timeout(self):
    timeout_s = envs.SGLANG_REQ_RUNNING_TIMEOUT.get()
    if timeout_s <= 0:
        return
    if self.running_batch.is_empty():
        return

    deadline = time.perf_counter() - timeout_s
    for req in self.running_batch.reqs:
        # Check timeout based on forward_entry_time
        # ... 设置FINISH_ABORT
```

### 8.3 Queue Limit Abort

当等待队列达到最大限制时，会触发abort：

```python
def _abort_on_queued_limit(self, recv_req: Req) -> bool:
    """Abort an incoming or existing request if the waiting queue is full."""
    if (
        self.max_queued_requests is None
        or len(self.waiting_queue) + 1 <= self.max_queued_requests
    ):
        return False

    # Priority scheduling logic for abort decision
    # ... abort低优先级请求
```

---

## 9. 客户端断连Abort

### 9.1 场景分类

| 类型 | 流式 | 请求状态 | Abort引擎 | Asyncio Task取消方式 |
|------|------|----------|-----------|---------------------|
| http | yes | validation | background task | fastapi |
| http | yes | waiting queue | background task | fastapi |
| http | yes | running | background task | fastapi |
| http | no | validation | http exception | http exception |
| http | no | waiting queue | type 1 | type 1 exception |
| http | no | running | type 3 | type 3 exception |

### 9.2 流式请求处理

流式请求使用background task检测断连：

```python
def create_abort_task(self, obj: GenerateReqInput):
    async def abort_request():
        await asyncio.sleep(2)  # 延迟检测
        if obj.is_single:
            self.abort_request(obj.rid)
        else:
            for rid in obj.rid:
                self.abort_request(rid)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(abort_request)
    return background_tasks
```

### 9.3 非流式请求处理

非流式请求在等待响应时主动检测：

```python
# Type 1: Waiting Queue
if await request.is_disconnected():
    self.abort_request(obj.rid)
    raise ValueError("Request disconnected (type 1)")

# Type 3: Running Batch
# 类似处理，但发生在不同阶段
```

---

## 10. Abort状态机总结

### 10.1 完整状态转换图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Request Abort State Machine                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐                                                               │
│  │  Tokenize │                                                              │
│  │  (入口)   │                                                              │
│  └────┬─────┘                                                               │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────┐     Abort Method 1      ┌──────────────┐                      │
│  │  Waiting │ ─────────────────────▶  │   Aborted    │                      │
│  │  Queue   │     (直接弹出)          │   (结束)     │                      │
│  └────┬─────┘                          └──────────────┘                      │
│       │                                          ▲                          │
│       │                                          │                          │
│       ▼                                          │                          │
│  ┌──────────┐     Abort Method 2      ┌──────────┴───┐                      │
│  │  Grammar │ ─────────────────────▶  │   Prefill    │                      │
│  │  Queue   │     (set_finish)        │   Forward    │                      │
│  └────┬─────┘                          └──────────────┘                      │
│       │                                                                     │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────┐     Abort Method 3      ┌──────────────┐                      │
│  │  Running │ ─────────────────────▶  │   Decode     │                      │
│  │  Batch   │     (to_finish)         │   Forward    │                      │
│  └──────────┘                          └──────┬───────┘                      │
│                                               │                             │
│                                               ▼                             │
│                                        ┌──────────────┐                      │
│                                        │  KV Cache    │                      │
│                                        │  Release     │                      │
│                                        └──────────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 不同场景Abort流程表

| 场景 | 触发条件 | 处理方法 | 资源清理 |
|------|----------|----------|----------|
| 用户主动abort | 调用`/abort_request` | Method 1/2/3 | KV Cache, HiCache |
| 客户端断连（流式） | `is_disconnected()` | background task | KV Cache |
| 客户端断连（非流式） | `is_disconnected()` | Type 1/3 exception | KV Cache |
| Waiting超时 | `SGLANG_REQ_WAITING_TIMEOUT` | Method 1 | KV Cache, HiCache |
| Running超时 | `SGLANG_REQ_RUNNING_TIMEOUT` | Method 3 | KV Cache |
| 队列满 | `max_queued_requests` | 优先级abort | KV Cache |
| Grammar错误 | 语法处理失败 | Method 2 | Grammar.cancel() |

---

## 11. 关键文件索引

| 组件 | 文件路径 | 关键函数/类 |
|------|----------|-------------|
| HTTP Endpoint | [http_server.py](python/sglang/srt/entrypoints/http_server.py) | `abort_request()` |
| AbortReq | [io_struct.py](python/sglang/srt/managers/io_struct.py) | `AbortReq` dataclass |
| TokenizerManager | [tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py) | `abort_request()`, `_handle_abort_req()` |
| Scheduler | [scheduler.py](python/sglang/srt/managers/scheduler.py) | `abort_request()` |
| FINISH_ABORT | [schedule_batch.py](python/sglang/srt/managers/schedule_batch.py) | `FINISH_ABORT`, `set_finish_with_abort()` |
| KV Cache Release | [common.py](python/sglang/srt/mem_cache/common.py) | `release_kv_cache()` |
| HiCache Abort | [hiradix_cache.py](python/sglang/srt/mem_cache/hiradix_cache.py) | `release_aborted_request()` |
| Grammar Manager | [grammar_manager.py](python/sglang/srt/constrained/grammar_manager.py) | `abort_requests()` |
| Disaggregation Prefill | [prefill.py](python/sglang/srt/disaggregation/prefill.py) | Prefill abort handling |
| Disaggregation Decode | [decode.py](python/sglang/srt/disaggregation/decode.py) | Decode abort handling |
| KV Connection | [conn.py](python/sglang/srt/disaggregation/base/conn.py) | `abort()` interface |
| Test Cases | [test_abort.py](test/registered/scheduler/test_abort.py) | Abort test suite |

---

## 12. 测试用例

SGLang提供了完整的abort测试用例，位于 [test_abort.py](test/registered/scheduler/test_abort.py)：

### 12.1 主要测试类

| 测试类 | 测试内容 |
|--------|----------|
| `TestAbort` | 基本abort功能和内存泄漏检测 |
| `TestAbortWithApiKey` | API Key认证下的abort |
| `TestAbortAll` | 批量abort功能 |
| `TestAbortAllWithRetraction` | Retraction场景下的abort |
| `TestAbortWithWaitingTimeout` | Waiting超时abort |
| `TestAbortWithRunningTimeout` | Running超时abort |

### 12.2 测试示例

```python
def test_abort_all_with_retraction(self):
    num_requests = 32
    with ThreadPoolExecutor(num_requests) as executor:
        futures = [executor.submit(self._run_decode) for _ in range(num_requests)]

        # ensure the decode has been started and retractions happen.
        time.sleep(8)

        requests.post(
            self.base_url + "/abort_request",
            json={"abort_all": True},
        )

        for future in as_completed(futures):
            result = future.result()
            finish_reason = result["meta_info"].get("finish_reason", {})
            self.assertEqual(finish_reason.get("type"), "abort")
```

---

## 13. 最佳实践

### 13.1 使用建议

1. **合理设置超时**：根据业务需求配置`SGLANG_REQ_WAITING_TIMEOUT`和`SGLANG_REQ_RUNNING_TIMEOUT`
2. **监控abort指标**：通过metrics collector监控abort率
3. **处理abort响应**：客户端应正确处理abort类型的finish_reason

### 13.2 调试建议

```bash
# 启用debug日志
export SGLANG_LOG_LEVEL_SERVER=DEBUG

# 测试retraction
export SGLANG_TEST_RETRACT=1
export SGLANG_TEST_RETRACT_INTERVAL=10
```

### 13.3 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| Abort后资源未释放 | Disaggregation模式特殊处理 | 检查retracted_queue清理 |
| 超时abort未触发 | 环境变量未设置 | 检查`SGLANG_REQ_*_TIMEOUT` |
| 流式请求abort延迟 | background task延迟 | 正常行为，延迟2秒 |
