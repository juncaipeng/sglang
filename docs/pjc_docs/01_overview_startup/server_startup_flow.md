# SGLang 服务启动流程

## 概述

SGLang 采用多进程架构，启动时依次创建 Scheduler、ModelWorker、DetokenizerManager 和 TokenizerManager，通过 ZMQ IPC 通信。完整启动流程约涉及 6 个阶段。

## 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Process                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  HTTP Server     │  │ TokenizerManager │  │TemplateManager│  │
│  │  (uvicorn/FastAPI)│  │                  │  │              │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                        │ ZMQ PUSH              ▲
         │                        ▼                       │ ZMQ PUSH
┌────────────────────────────────────────────────────────────────┐
│                   Scheduler Process (per TP rank)               │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  Scheduler   │  │ TpModelWorker│  │  KV Cache (RadixTree)│  │
│  │  (event loop)│  │              │  │                     │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
│                          │                                      │
│                    ┌─────┴──────┐                               │
│                    │ModelRunner │                               │
│                    │ - Model    │                               │
│                    │ - CUDA Graph│                              │
│                    │ - Attn Backend│                            │
│                    │ - Memory Pool│                             │
│                    └────────────┘                               │
└────────────────────────────────────────────────────────────────┘
         │ ZMQ PUSH
         ▼
┌────────────────────────────────────────┐
│        Detokenizer Process             │
│  ┌──────────────────────────────────┐  │
│  │     DetokenizerManager           │  │
│  │  (token IDs → text streaming)    │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

## 入口点

### 两种等价入口

| 方式 | 文件 | 说明 |
|------|------|------|
| `python -m sglang.launch_server` | `python/sglang/launch_server.py` | 直接调用 `prepare_server_args()` + `run_server()` |
| `sglang serve` | `python/sglang/cli/serve.py` | CLI 分发，检测模型类型后同样调用 `run_server()` |

两者最终汇聚到 `launch_server.py:run_server(server_args)`，根据模式分发：

```python
def run_server(server_args):
    if server_args.grpc_port:
        grpc_server.serve_grpc(server_args)
    elif server_args.ray_dp:
        ray.http_server.launch_server(server_args)
    else:
        http_server.launch_server(server_args)  # 默认路径
```

## 阶段一：参数解析

**文件**: `python/sglang/srt/server_args.py`

```
prepare_server_args(argv)
    → argparse 解析命令行
    → ServerArgs.from_cli_args(raw_args)
    → ServerArgs dataclass 实例
```

### ServerArgs 关键字段分组

| 分组 | 关键参数 | 说明 |
|------|----------|------|
| 模型 | `model_path`, `tokenizer_path`, `dtype` | 模型加载配置 |
| 并行 | `tp_size`, `dp_size`, `pp_size`, `ep_size` | 并行度 |
| 内存 | `mem_fraction_static`, `max_running_requests` | GPU 内存分配 |
| 调度 | `schedule_policy`, `chunked_prefill_size` | 调度策略 |
| 投机 | `speculative_algorithm`, `num_draft_tokens` | 投机解码 |
| 分离 | `disaggregation_mode` | PD 分离模式 |
| 服务 | `host`, `port`, `api_key` | HTTP 服务配置 |

### PortArgs：进程间通信地址

`PortArgs.init_new(server_args)` 分配 ZMQ IPC socket 路径：

| Socket | 方向 | 用途 |
|--------|------|------|
| `scheduler_input_ipc_name` | TokenizerManager → Scheduler | 请求输入 |
| `detokenizer_ipc_name` | Scheduler → Detokenizer | token 输出 |
| `tokenizer_ipc_name` | Detokenizer → TokenizerManager | 文本回传 |
| `rpc_ipc_name` | Engine ↔ Scheduler | 同步 RPC |
| `metrics_ipc_name` | Scheduler → Metrics | 指标发布 |

## 阶段二：子进程编排

**文件**: `python/sglang/srt/entrypoints/engine.py` — `Engine._launch_subprocesses()`

```python
def _launch_subprocesses():
    # 1. 环境配置
    configure_logger()
    _set_envs_and_config(server_args)
    server_args.check_server_args()

    # 2. IPC 地址分配
    port_args = PortArgs.init_new(server_args)

    # 3. 启动 Scheduler 进程
    _launch_scheduler_processes()

    # 4. 启动 Detokenizer 进程
    _launch_detokenizer_subprocesses()

    # 5. 主进程初始化 TokenizerManager
    init_tokenizer_manager()

    # 6. 等待所有子进程就绪
    wait_for_ready()

    # 7. 启动看门狗
    SubprocessWatchdog.start()
```

### 进程启动策略

```
dp_size == 1:
    每个 (pp_rank, tp_rank) 组合启动一个 Scheduler 进程

dp_size > 1:
    启动一个 DataParallelController 进程
    内部再启动多个 Scheduler Worker
```

## 阶段三：Scheduler 初始化

**文件**: `python/sglang/srt/managers/scheduler.py`

`run_scheduler_process()` 在子进程中执行：

```
configure_scheduler_process()     # 进程名、日志、CPU亲和性、NUMA绑定
    ↓
Scheduler.__init__()              # 核心初始化
    ↓
pipe_writer.send(get_init_info()) # 通知父进程就绪
    ↓
scheduler.run_event_loop()        # 阻塞运行事件循环
```

### Scheduler.__init__() 内部顺序

```
1.  init_model_config()          → ModelConfig
2.  init_ipc_channels()          → ZMQ sockets
3.  init_tokenizer()             → 约束解码用 tokenizer
4.  init_moe_gemm_config()       → MoE/FP8 GEMM 配置
5.  init_model_worker()          → TpModelWorker (最重步骤)
6.  kv_cache_builder.build_kv_cache() → RadixTree + 内存池
7.  init_running_status()        → waiting_queue, running_batch
8.  init_chunked_prefill()       → 分块预填充配置
9.  init_schedule_policy()       → 调度策略 (lpm/fcfs/lof)
10. init_disaggregation()        → PD 分离队列
11. init_overlap()               → overlap scheduling
12. GrammarManager, SchedulerRequestReceiver, SchedulerOutputStreamer
```

## 阶段四：模型加载与 Worker 初始化

### TpModelWorker

**文件**: `python/sglang/srt/managers/tp_worker.py`

```
TpModelWorker.__init__()
    ├── _init_model_config()      → ModelConfig.from_server_args()
    ├── _init_model_runner()      → ModelRunner (核心)
    ├── load tokenizer/processor
    ├── get_pp_group(), get_world_group()  → NCCL 通信组
    └── profile max_total_num_tokens, max_running_requests
```

### ModelRunner 初始化

**文件**: `python/sglang/srt/model_executor/model_runner.py`

```
ModelRunner.__init__()
    ├── model_specific_adjustment()     # 模型特定调整
    └── init_torch_distributed()        # NCCL 初始化
            ├── torch.cuda.set_device(gpu_id)
            ├── init_distributed_environment()   # NCCL/Gloo backend
            ├── initialize_model_parallel()      # TP/PP/EP/DP groups
            └── pre-warm NCCL (if configured)

ModelRunner.initialize()
    ├── load_model()                    # 加载模型权重
    │       ├── get_model_loader()      # 选择 loader (HF/vLLM/dummy)
    │       └── loader.load_model()     # 实际加载到 GPU
    ├── init_memory_pool()              # KV Cache 分配 (见阶段五)
    ├── init_attention_backend()        # 选择注意力后端
    └── init_device_graphs()            # CUDA Graph 捕获
```

### 注意力后端选择

根据硬件和模型自动选择：

| 后端 | 适用场景 |
|------|----------|
| FlashAttention | NVIDIA GPU 默认 |
| FlashInfer | 高性能替代 |
| Triton | 通用 GPU |
| PyTorch Native | 回退方案 |
| Ascend NPU | 华为昇腾 |
| Intel XPU | Intel GPU |

## 阶段五：KV Cache 内存分配

**文件**: `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`

### 内存 Profiling 流程

```
init_memory_pool()
    ├── _resolve_memory_pool_config(pre_model_load_memory)
    │       ├── _profile_available_bytes()
    │       │       → 测量模型加载前后 GPU 内存差
    │       │       → 可用内存 = 总内存 × mem_fraction_static - 模型占用
    │       ├── create_memory_pool_configurator()
    │       │       → 根据模型类型选择配置器 (MHA/MLA/Hybrid)
    │       ├── calculate_pool_sizes()
    │       │       → max_total_num_tokens = 可用字节 / 每token字节数
    │       └── _apply_token_constraints()
    │               → 用户上限、页对齐、PP 同步
    └── _apply_memory_pool_config()
            → 实际分配内存池
```

### 内存池类型

| 池类型 | 用途 |
|--------|------|
| `ReqToTokenPool` | 请求槽位 → token 索引映射 |
| `MHATokenToKVPool` | Multi-Head Attention KV 存储 |
| `MLATokenToKVPool` | Multi-Latent Attention (DeepSeek) KV 存储 |
| `HybridLinearKVPool` | 混合 SWA 模型 |

### 分配器类型

| 分配器 | 策略 |
|--------|------|
| `TokenToKVPoolAllocator` | 连续分配 |
| `PagedTokenToKVPoolAllocator` | 分页分配 |

### RadixTree Cache 构建

**文件**: `python/sglang/srt/mem_cache/kv_cache_builder.py`

```
build_kv_cache()
    ├── 获取 req_to_token_pool, token_to_kv_pool_allocator
    ├── CacheInitParams(eviction_policy, page_size, ...)
    └── create_tree_cache()  → RadixCache 或 DisabledCache
```

## 阶段六：HTTP 服务启动

**文件**: `python/sglang/srt/entrypoints/http_server.py`

```
_setup_and_run_http_server()
    ├── set_global_state(tokenizer_manager, template_manager, scheduler_info)
    ├── configure Prometheus middleware (if metrics enabled)
    └── uvicorn.run(app, host, port, event_loop="uvloop")
```

FastAPI `lifespan` 初始化：
- OpenAI 兼容接口 (chat, completion, embedding, rerank)
- Ollama 兼容接口
- Anthropic 兼容接口

## 完整启动时序图

```
时间 ──────────────────────────────────────────────────────────────────→

Main Process:
  [参数解析] → [PortArgs] → [启动Scheduler] → [启动Detokenizer] → [TokenizerMgr] → [wait_ready] → [HTTP Server]
                                  │                    │
Scheduler Process:                │                    │
                    [configure] → [Scheduler.__init__] │
                                       │              │
                                  [init_model_worker]  │
                                       │              │
                                  [ModelRunner]        │
                                       │              │
                                  [load_model]        │
                                       │              │
                                  [init_memory_pool]  │
                                       │              │
                                  [init_attn_backend] │
                                       │              │
                                  [CUDA graphs]       │
                                       │              │
                                  [build_kv_cache]    │
                                       │              │
                                  [send ready] ───────┼──→ Main 收到就绪信号
                                       │              │
                                  [event_loop]        │
                                                      │
Detokenizer Process:                                  │
                                       [DetokenizerManager.__init__]
                                       [event_loop]
```

## 关键配置对启动的影响

| 参数 | 影响 |
|------|------|
| `--tp N` | 启动 N 个 Scheduler 进程，每个绑定一个 GPU |
| `--dp N` | 启动 DataParallelController + N 组 Scheduler |
| `--pp N` | Pipeline 并行，N 个 stage 串联 |
| `--mem-fraction-static 0.9` | 控制 KV Cache 可用 GPU 内存比例 |
| `--max-running-requests` | 限制并发请求数，影响 ReqToTokenPool 大小 |
| `--chunked-prefill-size` | 分块预填充大小，影响调度粒度 |
| `--enable-torch-compile` | 启用 torch.compile 优化（增加启动时间） |
| `--disable-cuda-graph` | 跳过 CUDA Graph 捕获（减少启动时间） |
| `--disaggregation-mode` | 启用 PD 分离，额外初始化传输队列 |

## 错误处理与看门狗

### SubprocessWatchdog

启动后持续监控所有子进程存活状态：
- 定期检查子进程 `is_alive()`
- 任一子进程异常退出时，终止整个服务
- 通过 `wait_for_ready()` 超时检测启动失败

### 常见启动失败原因

| 错误 | 原因 | 解决 |
|------|------|------|
| CUDA OOM | 模型太大或 `mem_fraction_static` 过高 | 降低 `--mem-fraction-static` 或增加 `--tp` |
| NCCL timeout | 多 GPU 通信初始化失败 | 检查 GPU 互联、NCCL 版本 |
| Model not found | 模型路径错误或无网络 | 检查 `--model-path` |
| Port in use | 端口被占用 | 更换 `--port` |

## 关键源文件索引

| 文件 | 职责 |
|------|------|
| `python/sglang/launch_server.py` | 入口，`run_server()` |
| `python/sglang/srt/server_args.py` | `ServerArgs`, `PortArgs` |
| `python/sglang/srt/entrypoints/engine.py` | `Engine._launch_subprocesses()` |
| `python/sglang/srt/entrypoints/http_server.py` | HTTP 服务 + FastAPI |
| `python/sglang/srt/managers/scheduler.py` | `Scheduler`, 事件循环 |
| `python/sglang/srt/managers/tp_worker.py` | `TpModelWorker` |
| `python/sglang/srt/model_executor/model_runner.py` | `ModelRunner`, 模型加载 |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | 内存池分配 |
| `python/sglang/srt/mem_cache/kv_cache_builder.py` | KV Cache 构建 |
| `python/sglang/srt/managers/tokenizer_manager.py` | `TokenizerManager` |
| `python/sglang/srt/managers/detokenizer_manager.py` | `DetokenizerManager` |
