# SGLang 服务启动流程（详解 v2）

> 适用代码版本：2026-06 主干（基于当前仓库 `main` 分支梳理）。
> 本文由浅入深，覆盖从命令行入口到事件循环就绪的完整链路，并标注关键 `file:line`。

---

## 0. 一页速览

SGLang 是**多进程架构**：主进程跑 HTTP/TokenizerManager，每个 GPU rank 跑一个 Scheduler 子进程（内含 ModelRunner），另有 Detokenizer 子进程；进程间用 ZMQ IPC 通信。

```
                    ┌──────────────────────────────────────────────┐
                    │                 Main Process                 │
                    │  HTTP (uvicorn/FastAPI) + TokenizerManager    │
                    │  + TemplateManager + (P 端) BootstrapServer   │
                    └──────────────────────────────────────────────┘
   send_to_scheduler │ PUSH                  ▲ recv_from_detokenizer │ PULL
   (scheduler_input) ▼                       │ (tokenizer_ipc)
   ┌─────────────────────────────────────────┐      │
   │   Scheduler Process (每个 TP/PP rank)    │      │
   │   ┌─────────────┐   ┌─────────────────┐  │      │
   │   │  Scheduler  │   │  TpModelWorker  │  │      │
   │   │ (event loop)│   │  └ ModelRunner  │  │      │
   │   └─────────────┘   │    ├ Model      │  │      │
   │   ┌─────────────┐   │    ├ AttnBackend│  │      │
   │   │  tree_cache │   │    ├ CUDA Graph │  │      │
   │   │ (RadixCache)│   │    └ MemoryPool │  │      │
   │   └─────────────┘   └─────────────────┘  │      │
   └─────────────────────────────────────────┘      │
   send_to_detokenizer │ PUSH (detokenizer_ipc)      │
                       ▼                             │
              ┌────────────────────────┐  send_to_tokenizer │ PUSH
              │   Detokenizer Process  │────────────────────┘
              │   DetokenizerManager   │
              └────────────────────────┘
```

启动顺序（默认 HTTP 模式）：

```
launch_server.py:__main__
  → prepare_server_args()              # 解析 CLI → ServerArgs
  → run_server()                       # 按模式分发
      → http_server.launch_server()
          → Engine._launch_subprocesses()   # 编排所有子进程
              ① PortArgs.init_new()          # 分配 IPC 地址
              ② _launch_scheduler_processes() # 启动 Scheduler / DP Controller
              ③ _launch_detokenizer_subprocesses()
              ④ init_tokenizer_manager()      # 主进程内构造 TokenizerManager（含 P 端 bootstrap）
              ⑤ wait_for_ready()              # 阻塞等 Scheduler 模型加载完成
              ⑥ SubprocessWatchdog.start()
          → _setup_and_run_http_server()      # uvicorn + FastAPI lifespan + warmup
```

---

## 1. 入口与模式分发

### 1.1 两个等价入口

| 方式 | 文件 | 说明 |
|------|------|------|
| `python -m sglang.launch_server` | `python/sglang/launch_server.py` | 已标注 deprecation，仍受支持 |
| `sglang serve` | CLI 分发 | 推荐入口，最终同样落到 `run_server()` |

`launch_server.py:__main__`（`launch_server.py:53-71`）流程：`load_plugins()` → `prepare_server_args(sys.argv[1:])` → `run_server(server_args)`，`finally` 里 `kill_process_tree()` 兜底清理。

### 1.2 `run_server()` 分发阶梯

`launch_server.py:15-50`，按优先级判断：

| 优先级 | 条件 | 目标 |
|---|---|---|
| 1 | `encoder_only` 且 `grpc_mode` | `serve_grpc_encoder()`（asyncio）|
| 2 | `encoder_only` | `disaggregation.encode_server.launch_server()` |
| 3 | `grpc_mode` | `grpc_server.serve_grpc()`（legacy SMG）|
| 4 | `use_ray` | `ray.http_server.launch_server()` |
| 5 | **默认** | `http_server.launch_server()` |

> 所有 import 都是惰性的，避免为未用模式加载重依赖。

---

## 2. 参数解析：ServerArgs 与 PortArgs

`prepare_server_args()`（`server_args.py:7609-7643`）：构造 argparse → 可选 `--config` YAML 合并 → `ServerArgs.from_cli_args()`。日志在 `__post_init__` 前就配置好，保证其内部 log 有格式。

### 2.1 PortArgs：进程间通信地址

`PortArgs.init_new()`（`server_args.py:7653+`）为本地模式分配 `ipc://` 临时文件，多节点/分布式则分配 TCP 端点：

| 字段 | 方向 | 用途 |
|------|------|------|
| `scheduler_input_ipc_name` | TokenizerManager → Scheduler | 请求输入（PUSH/PULL）|
| `detokenizer_ipc_name` | Scheduler → Detokenizer | token id 输出 |
| `tokenizer_ipc_name` | Detokenizer → TokenizerManager | 解码文本回传 |
| `rpc_ipc_name` | Engine ↔ Scheduler | 同步 RPC（DEALER）|
| `metrics_ipc_name` | Scheduler → Metrics | 指标发布 |
| `tokenizer_worker_ipc_name` | 多 tokenizer 模式 worker → router | 多 HTTP worker |

---

## 3. 子进程编排：Engine._launch_subprocesses

`engine.py:735-880`，主进程内顺序执行：

```
1. configure_logger() / _set_envs_and_config()      # 全局环境
2. load_plugins() / check_server_args() / _set_gc()
3. port_args = PortArgs.init_new()                   # 若未传入
4. (可选) EngineInfoBootstrapServer                  # 远程权重加载场景
5. _launch_scheduler_processes()  ────────────────►  ② Scheduler/DP Controller
6. (node_rank>=1 守卫：非 0 节点不跑 tokenizer/detok，
    起 dummy health server 后阻塞)
7. _launch_detokenizer_subprocesses() ───────────►   ③ Detokenizer
8. init_tokenizer_manager()       ───────────────►   ④ TokenizerManager（主进程内）
9. scheduler_init_result.wait_for_ready()            # 阻塞等模型加载
10. 回填 max_req_input_len 到 tokenizer_manager
11. SubprocessWatchdog(...).start()                  # 子进程存活监控
```

### 3.1 Scheduler 启动策略（`_launch_scheduler_processes`，`engine.py:564-675`）

```
if dp_size == 1:
    pp_rank_range, tp_rank_range = _calculate_rank_ranges(nnodes, pp_size, tp_size, node_rank)
    for pp_rank in pp_rank_range:
        for tp_rank in tp_rank_range:
            gpu_id = base_gpu_id + (pp_rank%pp_size_per_node)*tp_size_per_node
                                 + (tp_rank%tp_size_per_node)*gpu_id_step
            attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(...)
            mp.Process(target=run_scheduler_process_func, args=(..., writer)).start()
else:   # dp_size > 1
    mp.Process(target=run_data_parallel_controller_process, ...).start()
```

- 每个 rank 通过 `mp.Pipe(duplex=False)` 回传就绪信息；父进程在 `wait_for_ready()` 里 `reader.recv()` 收齐。
- `dp_size>1` 时只起一个 DataParallelController，由它再 fork 出各 dp 组的 Scheduler（见 §8）。

### 3.2 Detokenizer 启动（`_launch_detokenizer_subprocesses`，`engine.py:677-732`）

- `detokenizer_worker_num <= 1`：单个 `run_detokenizer_process` 进程监听 `detokenizer_ipc_name`。
- `> 1`：每个 worker 一个私有 IPC，外加一个 `MultiDetokenizerRouter` 进程占用原始 `detokenizer_ipc_name` 并按 `crc32(rid)%N` 扇出。

> 安全提示：所有 IPC/`127.0.0.1` socket 无鉴权，依赖本机隔离。切勿跨主机暴露。

---

## 4. Scheduler 子进程初始化（核心）

`run_scheduler_process()`（`scheduler.py:3783-3848`）在子进程内执行：

```
load_plugins()
configure_scheduler_process()      # 进程名/日志/CPU亲和/NUMA绑定，返回 dp_rank
  ↓
scheduler = Scheduler(server_args, port_args, gpu_id, tp_rank,
                      moe_ep_rank, pp_rank, attn_cp_rank, moe_dp_rank, dp_rank)
  ↓
pipe_writer.send(scheduler.get_init_info())   # {status, max_total_num_tokens, max_req_input_len}
  ↓
scheduler.run_event_loop()         # 阻塞；异常时向父进程发 SIGQUIT
```

`configure_scheduler_process()`（`scheduler.py:3726-3780`）：`kill_itself_when_parent_died()` → 拼接日志前缀（`DP/PP/ATTN_CP/MOE_DP/TP/EP`）→ `setproctitle("sglang::scheduler_...")` → `faulthandler.enable()` → 可选 CPU 亲和/NUMA 绑定。

### 4.1 `Scheduler.__init__` 完整顺序

按 `scheduler.py:296-702` 实际调用次序：

```
 1. init_soft_watchdog()                 # 守护线程（首 tick 读 forward_ct/cur_batch）
 2. 解析 server_args 字段               # schedule_policy / overlap / pdmux / spec_algorithm / page_size ...
 3. compute_dp_attention_world_info()    # 推导 attn_tp/attn_dp rank/size
 4. ParallelState(...)                   # 统一保存 tp/pp/dp/attn_cp/moe 各维 rank/size + gpu_id
 5. init_model_config()                  # ModelConfig.from_server_args() (+NPU dllm 调整)
 6. SchedulerMetricsCollector.init_new()
 7. init_ipc_channels(port_args)         # 见 §4.2
 8. init_idle_sleeper()
 9. init_zbal_on_npu()                   # NPU 专用，须在任何 torch alloc 前
10. (enable_pdmux) init_pdmux()          # 创建 stream groups（见 §4.3）
11. init_tokenizer()                     # 文本: tokenizer；多模: processor+tokenizer
12. init_moe_gemm_config()               # MoE / FP8 / FP4 GEMM 配置 + require_mlp_sync
13. init_mamba_backend()
14. init_model_worker()                  # ★ 最重：TpModelWorker + (可选)draft worker（见 §5）
15. kv_cache_builder.build_kv_cache()    # ★ tree_cache + 内存池装配（见 §6）
16. (decode + offload) DecodeKVCacheOffloadManager
17. maybe_register_hicache_draft()       # spec + HiCache 共存时注册 draft KV pool
18. init_running_status()                # waiting_queue / running_batch / cur_batch / last_batch
19. init_chunked_prefill()
20. init_diffusion_llm()
21. SchedulerMetricsReporter(...)
22. init_schedule_policy()               # SchedulePolicy + NewTokenRatioTracker
23. init_watch_dog_memory_saver_input_blocker()
24. SchedulerProfilerManager(...)
25. init_disaggregation()                # ★ PD 分离队列（见 §7）
26. init_overlap()                       # overlap 用的 forward/copy stream + FutureMap
27. maybe_init_ngram_embedding()
28. init_deterministic_inference_config()
29. SchedulerWeightUpdaterManager(...)
30. init_request_dispatcher()            # 见 §4.4
31. (lora) LoRADrainer / LoRAOverlapLoader
32. GrammarManager(self)
33. SchedulerRequestReceiver / SchedulerDPAttnAdapter / PoolStatsObserver /
    InvariantChecker / KvEventsPublisher / LoadInquirer /
    OutputStreamer / BatchResultProcessor
```

> 与旧版差异：overlap 调度**不再使用独立 `TpModelWorkerClient` 线程**（`tp_worker_overlap_thread.py` 已删除）。overlap 现由 `Scheduler.init_overlap()` 内的 `forward_stream`/`copy_stream`/`FutureMap` 直接驱动；`ModelRunner.__init__` 创建 `forward_stream`。`Scheduler.__init__` 大量逻辑已拆成独立组件类（Receiver/Streamer/ResultProcessor/...），不再都内联在调度器里。

### 4.2 `init_ipc_channels`（`scheduler.py:735-750`）

仅 **rank 0**（`pp_rank==0 && attn_tp_rank==0 && attn_cp_rank==0`）创建与 tokenizer/detokenizer 的通道。`SchedulerIpcChannels.create()`（`scheduler_components/ipc_channels.py:19-73`）：

| socket | 类型 | endpoint | bind |
|---|---|---|---|
| `recv_from_tokenizer` | PULL | `scheduler_input_ipc_name` | True |
| `recv_from_rpc` | DEALER | `rpc_ipc_name` | False |
| `send_to_tokenizer` | PUSH | `tokenizer_ipc_name` | — |
| `send_to_detokenizer` | PUSH | `detokenizer_ipc_name`（`skip_tokenizer_init` 时为 `tokenizer_ipc_name`）| — |
| `send_metrics_from_scheduler` | PUSH | `metrics_ipc_name` | — |

### 4.3 `init_pdmux`（`multiplex/multiplexing_mixin.py:34-46`）

仅当 `--enable-pdmux`：`load_pdmux_config()` → `initialize_stream_groups(gpu_id, config)` → 缓存 `stream_groups` / `sm_counts` / `real_sm_group_num`。用于在同一 GPU 上按 SM 配额并行跑 prefill-only 与 decode-only 流。

### 4.4 `init_request_dispatcher`（`scheduler.py:1365+`）

构造 `TypeBasedDispatcher`，把消息类型映射到处理函数：`TokenizedGenerateReqInput→handle_generate_request`、`AbortReq→abort_request`、`FlushCacheReqInput`、各类权重更新/会话/HiCache 控制消息等。

### 4.5 事件循环分发（`dispatch_event_loop`，`scheduler.py:3695-3723`）

```
NULL（无分离）:   pdmux→event_loop_pdmux | pp>1→event_loop_pp
                | overlap_mlx→event_loop_overlap_mlx | overlap→event_loop_overlap
                | else→event_loop_normal
PREFILL:          pp>1→event_loop_pp_disagg_prefill | overlap→event_loop_overlap_disagg_prefill
                | else→event_loop_normal_disagg_prefill
DECODE:           pp>1→event_loop_pp_disagg_decode  | overlap→event_loop_overlap_disagg_decode
                | else→event_loop_normal_disagg_decode
```

`event_loop_normal`（`scheduler.py:1511+`）骨架：`recv_requests()` → `process_input_requests()` → `get_next_batch_to_run()` → `run_batch()` → `process_batch_result()`。

---

## 5. 模型加载与 Worker 初始化

### 5.1 TpModelWorker（`tp_worker.py:218-430`）

`init_model_worker()`（`scheduler.py:912-994`）= `init_tp_model_worker()` + `maybe_init_draft_worker()`，随后从 worker 拉取全局信息。

`TpModelWorker.__init__` 顺序：

```
1. 解析/保存 args（ranks, gpu_id, nccl_port, 共享 pool 引用）
2. _init_model_config()        # ModelConfig.from_server_args()（draft 用 speculative_draft_model_path）
3. _init_model_runner()        # 构造 ModelRunner（核心，见 §5.3）
4. (multi-layer eagle) _init_multi_layer_eagle_model_runners()
5. _init_dllm_algorithm()
6. tokenizer/processor（多模用 get_processor，文本用 get_tokenizer；skip 时为 None）
7. self.device = model_runner.device
8. get_pp_group() / get_world_group()
9. 计算 token/请求上限（见下）
10. 跨 TP broadcast random_seed → set_random_seed()
```

token/请求上限（`tp_worker.py:296-311`）：

| 字段 | 来源 |
|---|---|
| `max_total_num_tokens` | `model_runner.max_total_num_tokens` |
| `max_prefill_tokens` | `server_args.max_prefill_tokens`（直接取）|
| `max_running_requests` | `model_runner.max_running_requests`（断言 >0）|
| `max_req_len` | `min(context_len-1, max_token_pool_size-1)` |
| `max_req_input_len` | `max_req_len - 5` |

`get_worker_info()`（`tp_worker.py:416-430`）返回 12 元组，Scheduler 用其中的 `max_total_num_tokens / max_prefill_tokens / max_running_requests / max_queued_requests / max_req_len / max_req_input_len / random_seed / device / forward_stream / ...`。

### 5.2 模型权重下载（ModelScope / HF）

默认从 HuggingFace 拉取；设 `SGLANG_USE_MODELSCOPE=1` 时改走 ModelScope：

- 配置/tokenizer 文件：`configs/model_config.py:914-944`（`model_file_download` / snapshot）。
- 权重快照：`model_loader/loader.py:380-412` `_maybe_download_from_modelscope()`（`snapshot_download`）。

### 5.3 ModelRunner（`model_executor/model_runner.py`）

`__init__`（`model_runner.py:338-576`）关键步骤：

```
... 解析 args / Eagle3 aux-hidden 配置 / model_specific_adjustment()
pre_model_load_memory = init_torch_distributed()    # ★ NCCL + 并行组（见下）
init_shared_mooncake_transfer_engine()
self.forward_stream = device_module.Stream()        # overlap 用
set_offloader(create_offloader_from_server_args())   # CPU offload
self.initialize(pre_model_load_memory)              # ★ 见下
check_quantized_moe_compatibility()
support_pp = "pp_proxy_tensors" in model.forward 签名
```

`init_torch_distributed()`（`model_runner.py:1036-1174`）：
`set_device(gpu_id)` → 选 backend（nccl/gloo/mooncake）→ 解析 `dist_init_method` →
`init_distributed_environment(world_size=tp*pp, rank=tp_size*pp_rank+tp_rank)` →
`initialize_model_parallel(tp, attn_dp, pp, ep)` → `initialize_dp_attention()` →
（可选）NCCL 预热 all_reduce → 计算并返回 `pre_model_load_memory`（加载模型前的可用显存）。

`initialize(pre_model_load_memory)`（`model_runner.py:607-826`）**精确调用序**：

```
 1. TorchMemorySaverAdapter.create()
 2. (非 draft) set_global_expert_location_metadata() / set_global_expert_distribution_recorder()
 3. eplb_manager = EPLBManager(self) (if 需要)
 4. expert_location_updater = ExpertLocationUpdater()
 5. sampler = create_sampler()
 6. load_model()                        # ★ get_model_loader → loader.load_model → model.eval()
 7. _prepare_moe_topk()
 8. 计算 start_layer/end_layer/num_effective_layers + adjust_hybrid_swa_layers_for_pp()
 9. apply_torchao_config_to_model()
10. (tp_size>1 且支持) apply_torch_tp()  # init_device_mesh + tensor_parallel(model)
11. (lora) init_lora_manager() + MoE buffer 预分配
12. configure_kv_cache_dtype()          # 决定 kv_cache_dtype（auto/fp8_e5m2/e4m3/bf16/fp4）
13. init_memory_pool(pre_model_load_memory)   # ★ 见 §6.1
14. maybe_init_ngram_embedding()
15. init_routed_experts_capturer() / init_indexer_capturer()
16. init_aux_hidden_state_capture()     # 须在 device graph 前
17. 设备分支（CUDA）:
      init_cublas() → (hisparse) → init_attention_backend()
      → kernel_warmup() → init_device_graphs()    # CUDA Graph 捕获
    (CPU/NPU/out-of-tree 各有简化分支)
18. (forward_hooks) register_forward_hooks()
19. init_piecewise_cuda_graphs()
20. prealloc_symmetric_memory_pool()
```

- `load_model()`（`model_runner.py:1221+`）：构造 `LoadConfig` → `get_model_loader()` → `loader.load_model(model_config, DeviceConfig)`（`model.eval()` 在各 loader 内调用）→ 记录权重显存占用 → RoPE cache 预扩展 + 跨 TP monitored barrier。
- `init_cublas()`（`2217-2224`）：跑一次 16×16 fp16 矩阵乘强制初始化 cuBLAS。
- `init_attention_backend()`（`2226-2300`）：按 `server_args.get_attention_backends()` 解析 prefill/decode 后端字符串；prefill≠decode 时用 `HybridAttnBackend`；pdmux 会建 per-SM-group 的 decode backend；TBO 用 `TboAttnBackend`。
- `init_device_graphs()`（`2754-2804`）：非生成/`disable_cuda_graph`/CPU 等条件下跳过；否则按设备选 `CudaGraphRunner`/`NPUGraphRunner`/`CPUGraphRunner`，实例化即完成捕获。

#### 注意力后端选择

| 后端 | 适用场景 |
|------|----------|
| FlashAttention (fa3 等) | NVIDIA GPU 默认 |
| FlashInfer | 高性能替代，MLA/分页支持好 |
| Triton | 通用 GPU 回退 |
| Ascend / XPU / TPU | 对应硬件后端 |
| HybridAttnBackend | prefill 与 decode 用不同后端时的组合 |

---

## 6. KV Cache 内存分配与 tree_cache 装配

分两阶段，发生在两个不同对象里：

1. **`ModelRunner.init_memory_pool()`** 创建**物理池**：`req_to_token_pool` / `token_to_kv_pool` / `token_to_kv_pool_allocator`。
2. **`Scheduler` 调 `kv_cache_builder.build_kv_cache()`** 取出这些池并构造逻辑 **`tree_cache`**。

### 6.1 物理内存池（`model_runner_kv_cache_mixin.py`）

`init_memory_pool(pre_model_load_memory)`（`mixin:899-914`）→ `_resolve_memory_pool_config()` → `_apply_memory_pool_config()`：

```
_profile_available_bytes()           # rest = post_load_mem - pre_load_mem*(1-mem_fraction_static)
  ↓
configurator 把 bytes → tokens        # max_total_num_tokens = available_bytes // cell_size（页对齐）
  ↓
_apply_token_constraints()            # 用户 max_total_tokens 上限 + PP all-reduce 取 MIN
  ↓
_resolve_max_num_reqs(token_capacity) # clamp(token_capacity/context_len*512, 2048, 4096) 等
  ↓
_init_pools()                         # 构造三类池
```

**ReqToTokenPool**（`memory_pool.py:138-203`）：请求槽位 → 每 token 的 KV 槽位映射。

- `req_to_token = zeros((size+1, max_context_len), int32)`（index 0 为 padding 行，给 CUDA Graph 填充批次用）。
- `free_slots = list(range(1, size+1))`；接口 `alloc(reqs)` / `write(indices, values)` / `free(req)`。

**MHATokenToKVPool**（`memory_pool.py:796-947`）：每层一个 K、一个 V tensor。

- `k_buffer[layer] = zeros((size + page_size, head_num, head_dim), store_dtype)`；`v_buffer` 同理用 `v_head_dim`。
- 额外 `page_size` 行包含 padding slot 0。FP8 时 `store_dtype=uint8`。

**MLATokenToKVPool**（DeepSeek，`memory_pool.py:1625-1694`）：每层单个合并 latent buffer `zeros((size+page_size, 1, kv_lora_rank+qk_rope_head_dim))`，不分开 K/V。

**TokenToKVPoolAllocator**（`allocator.py`）：管理 KV 池的分配回收。

- `page_size==1`：`free_pages = arange(1, size+1)`（slot 0 保留）；`alloc(need)` 从头切片，`free()` 归还。
- `page_size>1`：`PagedTokenToKVPoolAllocator` 管理 page id，`alloc` 时展开成 token 索引 `page*page_size + arange(page_size)`；`alloc_extend`/`alloc_decode` 用 Triton kernel。
- PD 分离模式（prefill/decode）会 `need_sort=True`，free 列表延迟排序。

### 6.2 tree_cache 选择（`build_kv_cache` + `registry.py`）

`build_kv_cache()`（`kv_cache_builder.py:131-255`）返回 `KVCacheBuildResult`（`is_hybrid_swa / is_hybrid_ssm / sliding_window_size / full_tokens_per_layer / swa_tokens_per_layer / req_to_token_pool / token_to_kv_pool_allocator / disable_radix_cache / tree_cache`）。它本身**不创建物理池**，只消费 `tp_worker.get_memory_pool()` 的结果并构造 tree_cache。

`default_radix_cache_factory()`（`registry.py:76-154`）按优先级选类：

| 优先级 | 条件 | tree_cache 类 |
|---|---|---|
| 1 | chunked_prefill 开 + `disable_radix_cache`，非 SWA | `ChunkCache` |
| 1b | 同上但 hybrid SWA | `SWAChunkCache` |
| 2 | `SGLANG_EXPERIMENTAL_CPP_RADIX_TREE` | `RadixCacheCpp` |
| 3 | `SGLANG_ENABLE_UNIFIED_RADIX_TREE` | `UnifiedRadixCache` |
| 4 | hierarchical + hybrid SSM | `HiMambaRadixCache` |
| 4b | hierarchical（其余）| `HiRadixCache` |
| 5 | hybrid SWA | `SWARadixCache` |
| 6 | hybrid SSM | `MambaRadixCache` |
| 7 | `enable_lmcache` | `LMCRadixCache` |
| 8 | 兜底 | `RadixCache` |

`RadixCache.__init__`（`radix_cache.py:270-316`）：保存池引用/page_size/eviction 策略 → 选 `eviction_strategy`（lru/lfu/fifo/mru/filo/priority/slru）→ `reset()` 建 `root_node`（`lock_ref=1` 永不被驱逐）。

### 6.3 HiCache / Offload 启动期装配

- **HiRadixCache**（`hiradix_cache.py:74-189`）：建主机镜像池（`MHATokenToKVPoolHost`/`MLATokenToKVPoolHost`）→ 建 `HiCacheController`（`cache_controller.py:245-338`，含 device/host 池、`io_backend`、`storage_backend`、load/write 队列与独立 stream）→ 若 `hicache_storage_backend` 非空，启动期即 `attach_storage_backend()`（factory 创建 file/nixl/mooncake/hf3fs/eic 等后端）。
- **DecodeKVCacheOffloadManager**（`decode_kvcache_offload_manager.py:37-108`）：仅 `disaggregation_mode=="decode"` 且 `--disaggregation-decode-enable-offload-kvcache` 时构造；自带 decode host 池 + 独立 `HiCacheController`，做 GPU→Host→Storage 卸载。

---

## 7. PD 分离初始化（`init_disaggregation`，`scheduler.py:1135-1264`）

```
disaggregation_mode = DisaggregationMode(server_args.disaggregation_mode)   # NULL/PREFILL/DECODE
transfer_backend = TransferBackend(...)                                     # zmq/mooncake/nixl/...
draft_token_to_kv_pool, model_config = get_draft_kv_pool(...)               # spec 场景
```

| 角色 | 创建的对象 | 作用 |
|------|-----------|------|
| **DECODE** | `ReqToMetadataIdxAllocator` + `MetadataBuffers`（`buffer_size = req_to_token_pool.size*2`）| 元数据缓冲 |
| | `DecodeTransferQueue` | 轮询 KV cache 传输完成的请求 |
| | `DecodePreallocQueue` | 等待预分配 KV 的请求 |
| **PREFILL** | `ReqToMetadataIdxAllocator` + `MetadataBuffers`（`buffer_size = max_running_requests*2`）| 元数据缓冲 |
| | `PrefillBootstrapQueue` | 尚未完成握手的请求 |
| | `disagg_prefill_inflight_queue`（list）| 正在发送 KV 的请求 |

请求在 P 端生命周期：`bootstrap → waiting → running → inflight`；D 端：`prealloc → transfer → waiting → running`。

EPD（encoder-prefill-decode）分离：`language_only` 且 `encoder_transfer_backend=="zmq_to_scheduler"` 时额外建 `mm_receiver`。

---

## 8. 多进程管理器初始化

### 8.1 DataParallelController（`dp_size>1`，`data_parallel_controller.py`）

`run_data_parallel_controller_process()`（`621-667`）→ `DataParallelController.__init__()`（`124-189`）：

```
load_balance_method = LoadBalanceMethod.from_str(...)   # round_robin / follow_bootstrap_room / total_requests / total_tokens
context = zmq.Context(1 + dp_size)
recv_from_tokenizer = PULL(scheduler_input_ipc_name)    # node 0
if enable_dp_attention:  launch_dp_attention_schedulers()
else:                    launch_dp_schedulers()          # 每个 dp_rank 一个线程 → launch_tensor_parallel_group()
init_dispatcher()
```

- `launch_dp_schedulers()`（`237-281`）：每 dp_rank 起一个线程跑 `launch_tensor_parallel_group()`，各自共享同一组 tokenizer/detokenizer IPC；node 0 为每个 worker 建一个 PUSH socket。
- `launch_tensor_parallel_group()`（`444-559`）：嵌套 `for pp_rank, for tp_rank` 用 `mp.Process(target=run_scheduler_process_func, ...)` 启动各 rank 的 Scheduler，并通过 pipe 收齐就绪信息。
- `event_loop()`（`610-618`，仅 node 0）：非阻塞 drain `recv_from_tokenizer`，按类型分发；负载均衡选 rank（round-robin / `bootstrap_room % N` / 最少请求 / 最少 token）。

### 8.2 DetokenizerManager（`detokenizer_manager.py`）

`run_detokenizer_process()`（`426-448`）→ `__init__`（`79-94`）依次 `init_ipc_channels / init_tokenizer / init_running_status / init_request_dispatcher`：

| socket | 类型 | endpoint | bind |
|---|---|---|---|
| `recv_from_scheduler` | PULL | `detokenizer_ipc_name` | True |
| `send_to_tokenizer`（仅单 worker）| PUSH | `tokenizer_ipc_name` | False |

`event_loop()`（`145-153`）：`recv_from_scheduler.recv_pyobj()` → 增量去 token 化（按 rid 维护 `surr_offset/read_offset/sent_offset`）→ `send_to_tokenizer.send_pyobj()`。

### 8.3 TokenizerManager（主进程，`tokenizer_manager.py:236-276`）

```
init_model_config()                # ModelConfig；EAGLE 预留 token 数
init_tokenizer_and_processor()     # 文本 get_tokenizer；多模 import_processors + get_mm_processor
init_ipc_channels(port_args)       # 见下（zmq.asyncio）
init_running_status()
init_request_logging_and_dumping()
init_weight_update()
init_lora()
init_disaggregation()              # ★ start_disagg_service()
init_metric_collector_watchdog()
init_request_dispatcher()          # + init_communicators()
```

| socket | 类型 | endpoint | bind |
|---|---|---|---|
| `recv_from_detokenizer` | PULL | `tokenizer_ipc_name` | True |
| `send_to_scheduler` | PUSH | `scheduler_input_ipc_name` | True |
| （多 tokenizer）`send_to_scheduler` | PUSH | `tokenizer_worker_ipc_name` | False（外包 SenderWrapper）|

> `recv_from_rpc` 不在 TokenizerManager，而在 Scheduler 侧（DEALER）。

**`start_disagg_service()`**（`disagg_service.py:14-44`）：**仅 PREFILL 角色**创建 bootstrap server——`get_kv_class(transfer_backend, BOOTSTRAP_SERVER)` 取类（Mooncake→`MooncakeKVBootstrapServer`，Nixl/Mori/Ascend 各有对应），实例化后在守护线程里跑 aiohttp（路由 `/route`、`/register_dp_rank`、`/query_dp_ranks`、`/health`）。decode 端不起 bootstrap server。

---

## 9. HTTP 服务启动与 warmup

`_setup_and_run_http_server()`（`http_server.py:2121-2320`）：

```
set_global_state(_GlobalState(tokenizer_manager, template_manager, scheduler_info))
tokenizer_manager._subprocess_watchdog = watchdog
(metrics) add_prometheus_track_response_middleware()
uvicorn.run(app, loop="uvloop", host, port, ssl...)   # 或 http2 走 Granian
```

FastAPI `lifespan`（`http_server.py:284-400`）启动时：实例化 OpenAI/Ollama/Anthropic 各 serving handler → 可选 tracing/metrics → 在后台线程跑 `_wait_and_warmup`。

`_wait_and_warmup`（`2025-2050`）→ `_execute_server_warmup`（`1864-2022`）：轮询 `GET /model_info`（最多 120 次，每秒一次）→ 构造 warmup 请求（VLM 走 `/v1/chat/completions`，生成走 `/generate`，embedding 走 `/encode`）→ POST 成功后置 `server_status = Up`，打印 "The server is fired up and ready to roll!"。PD 分离时用 `FAKE_BOOTSTRAP_HOST` 与合成 `bootstrap_room` 做预热。

---

## 10. 完整启动时序图

```
时间 ────────────────────────────────────────────────────────────────────────►

Main Process:
 [CLI解析]→[PortArgs]→[起Scheduler]→[起Detok]→[TokenizerMgr(+P端bootstrap)]→[wait_ready]→[uvicorn+lifespan]→[warmup→Up]
                          │              │                                          ▲
Scheduler Process:        ▼              │                                          │
        [configure]→[Scheduler.__init__]                                            │
              ├ init_model_worker → TpModelWorker → ModelRunner                      │
              │      ├ init_torch_distributed (NCCL/并行组)                          │
              │      ├ load_model (权重→GPU, model.eval)                             │
              │      ├ init_memory_pool (KV池)                                       │
              │      ├ init_attention_backend                                        │
              │      └ init_device_graphs (CUDA Graph 捕获)                          │
              ├ build_kv_cache (tree_cache + 池装配)                                 │
              ├ init_disaggregation / init_overlap / dispatcher ...                  │
              └ pipe_writer.send(init_info) ──────────────────────────────────────► Main 收到就绪
                     └ run_event_loop (阻塞)

Detok Process:
        [DetokenizerManager.__init__]→[event_loop]

(dp_size>1 时 Scheduler 前面多一层 DataParallelController，负责按 dp_rank fork 各 TP 组并做请求负载均衡)
```

---

## 11. 关键配置对启动的影响

| 参数 | 影响 |
|------|------|
| `--tp N` | 启动 N 个 Scheduler 进程，每个绑定一个 GPU |
| `--dp N` | 启动 DataParallelController + N 组 Scheduler |
| `--pp N` | Pipeline 并行，rank 维度叠加 |
| `--ep N` / `--moe-dp N` | MoE 专家并行/数据并行维度 |
| `--mem-fraction-static` | 控制 KV Cache 可用显存比例（profiling 输入）|
| `--max-running-requests` | 影响 `max_num_reqs` 与 ReqToTokenPool 大小 |
| `--chunked-prefill-size` | 分块预填充粒度；与 `disable_radix_cache` 共同决定是否用 ChunkCache |
| `--enable-overlap`(默认开) | 用 forward/copy stream + FutureMap 做 CPU/GPU overlap |
| `--enable-pdmux` | 同 GPU 上 prefill/decode SM 复用，启动期建 stream groups |
| `--disable-cuda-graph` | 跳过 CUDA Graph 捕获，减少启动时间 |
| `--disaggregation-mode` | prefill/decode 角色，额外建传输队列；P 端起 bootstrap server |
| `--enable-hierarchical-cache` | 用 HiRadixCache + HiCacheController（host/storage 多级）|
| `SGLANG_USE_MODELSCOPE=1` | 权重/配置改从 ModelScope 下载 |

---

## 12. 错误处理与看门狗

- **SubprocessWatchdog**：主进程监控所有子进程 `is_alive()`，任一异常退出即终止整个服务。
- **Scheduler watchdog**：`watchdog_timeout` / `soft_watchdog_timeout` 守护线程；卡死时报错。
- **kill_itself_when_parent_died()**：每个子进程在父进程死亡时自杀，避免僵尸进程。
- 子进程异常会向父进程发 `SIGQUIT`，触发整体下线。

常见启动失败：

| 错误 | 原因 | 解决 |
|------|------|------|
| CUDA OOM | 模型太大 / `mem_fraction_static` 过高 | 降 `--mem-fraction-static` 或加 `--tp` |
| NCCL timeout | 多 GPU 通信初始化失败 | 查 GPU 互联 / NCCL 版本 / `dist_init_addr` |
| Model not found | 路径错或无网络 | 查 `--model-path`，必要时设 ModelScope |
| Port in use | 端口被占 | 换 `--port` / bootstrap port |

---

## 13. 关键源文件索引

| 文件 | 职责 |
|------|------|
| `python/sglang/launch_server.py` | 入口 `run_server()` 模式分发 |
| `python/sglang/srt/server_args.py` | `ServerArgs` / `PortArgs.init_new()` |
| `python/sglang/srt/entrypoints/engine.py` | `Engine._launch_subprocesses()` 编排 |
| `python/sglang/srt/entrypoints/http_server.py` | HTTP 服务 + FastAPI lifespan + warmup |
| `python/sglang/srt/managers/scheduler.py` | `Scheduler` / `run_scheduler_process` / 事件循环 |
| `python/sglang/srt/multiplex/multiplexing_mixin.py` | `init_pdmux` |
| `python/sglang/srt/managers/tp_worker.py` | `TpModelWorker` |
| `python/sglang/srt/model_executor/model_runner.py` | `ModelRunner` / `initialize()` |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | `init_memory_pool` |
| `python/sglang/srt/mem_cache/kv_cache_builder.py` | `build_kv_cache` |
| `python/sglang/srt/mem_cache/registry.py` | tree_cache 工厂选择 |
| `python/sglang/srt/mem_cache/memory_pool.py` / `allocator.py` | 物理池 / 分配器 |
| `python/sglang/srt/managers/cache_controller.py` | `HiCacheController` |
| `python/sglang/srt/managers/data_parallel_controller.py` | `DataParallelController` |
| `python/sglang/srt/managers/detokenizer_manager.py` | `DetokenizerManager` |
| `python/sglang/srt/managers/tokenizer_manager.py` | `TokenizerManager` |
| `python/sglang/srt/managers/disagg_service.py` | `start_disagg_service` / bootstrap server |
| `python/sglang/srt/managers/scheduler_components/ipc_channels.py` | Scheduler 侧 ZMQ 通道 |
