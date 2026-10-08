# SGLang Metrics 指标体系

SGLang 的可观测性体系以 Prometheus 为核心，通过 `/metrics` HTTP 端点暴露指标，共计 **85+** 个指标。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    /metrics endpoint                      │
│            (FastAPI / gRPC sidecar)                       │
├─────────────────────────────────────────────────────────┤
│  SchedulerMetricsCollector   │  TokenizerMetricsCollector │
│  StorageMetricsCollector     │  RadixCacheMetricsCollector│
│  ExpertDispatchCollector     │  Standalone metrics        │
└─────────────────────────────────────────────────────────┘
         ↕ ZMQ PUB                    ↕ OTLP
  ForwardPassMetrics            OpenTelemetry Traces
  (per-iteration streaming)     (distributed tracing)
```

启用方式：`--enable-metrics` 启动参数，访问 `http://<host>:<port>/metrics`。

核心代码：`python/sglang/srt/observability/metrics_collector.py`

---

## 一、请求队列与调度状态 (Gauge)

| 指标名 | 说明 |
|--------|------|
| `sglang:num_running_reqs` | 正在运行的请求数 |
| `sglang:num_queue_reqs` | 等待队列中的请求数 |
| `sglang:num_grammar_queue_reqs` | Grammar 等待队列中的请求数 |
| `sglang:num_retracted_reqs` | 当前被回撤的请求数 |
| `sglang:num_paused_reqs` | 因异步权重同步暂停的请求数 |
| `sglang:utilization` | 系统利用率 |
| `sglang:fwd_occupancy` | Forward pass GPU 占用率(%) |
| `sglang:num_unique_running_routing_keys` | Running batch 中唯一 routing key 数 |
| `sglang:max_running_requests_under_SLO` | SLO 下最大运行请求数 |

---

## 二、KV Cache 内存指标 (Gauge)

| 指标名 | 说明 |
|--------|------|
| `sglang:token_usage` | KV cache 整体使用率 |
| `sglang:full_token_usage` | Full attention 层 token 使用率 |
| `sglang:num_used_tokens` | 已使用的 token slot 数 |
| `sglang:kv_available_tokens` | KV pool 空闲 token slot 数 |
| `sglang:kv_evictable_tokens` | 可逐出(radix cache)的 token slot 数 |
| `sglang:kv_used_tokens` | 活跃使用的 token slot 数 |
| `sglang:max_total_num_tokens` | KV cache pool 最大容量 |
| `sglang:page_size` | KV cache page 大小(tokens) |
| `sglang:num_pages` | KV cache 总页数 |
| `sglang:cache_hit_rate` | Prefix cache 命中率 |

---

## 三、SWA/Mamba 内存指标 (Gauge)

| 指标名 | 说明 |
|--------|------|
| `sglang:swa_token_usage` | SWA 层 token 使用率 |
| `sglang:swa_available_tokens` | SWA pool 空闲 slot |
| `sglang:swa_evictable_tokens` | SWA pool 可逐出 slot |
| `sglang:swa_used_tokens` | SWA pool 活跃 slot |
| `sglang:mamba_usage` | Mamba 层 token 使用率 |
| `sglang:mamba_available_tokens` | Mamba SSM pool 空闲 slot |
| `sglang:mamba_evictable_tokens` | Mamba pool 可逐出 slot |
| `sglang:mamba_used_tokens` | Mamba pool 活跃 slot |

---

## 四、吞吐量与计算指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `sglang:gen_throughput` | Gauge | 生成吞吐量 (token/s) |
| `sglang:prompt_tokens_total` | Counter | 已处理 prefill tokens 总数 |
| `sglang:generation_tokens_total` | Counter | 已生成 tokens 总数 |
| `sglang:cached_tokens_total` | Counter | 缓存命中的 prompt tokens (label: device/host/storage) |
| `sglang:realtime_tokens_total` | Counter | 实时 tokens (label: prefill_compute/prefill_cache/decode) |
| `sglang:forward_execution_seconds_total` | Counter | GPU forward pass 执行总时间 |
| `sglang:estimated_flops_per_gpu_total` | Counter | 估算的每 GPU FLOPs 总数 |
| `sglang:estimated_read_bytes_per_gpu_total` | Counter | 估算的每 GPU 读取字节数 |
| `sglang:estimated_write_bytes_per_gpu_total` | Counter | 估算的每 GPU 写入字节数 |

---

## 五、延迟指标 (Histogram)

| 指标名 | 说明 |
|--------|------|
| `sglang:time_to_first_token_seconds` | 首 token 延迟 (TTFT) |
| `sglang:inter_token_latency_seconds` | Token 间延迟 (ITL) |
| `sglang:e2e_request_latency_seconds` | 端到端请求延迟 |
| `sglang:queue_time_seconds` | 排队等待时间 |
| `sglang:per_stage_req_latency_seconds` | 各阶段延迟分布 |
| `sglang:func_latency_seconds` | 函数级延迟 (通用计时器, label: name) |
| `sglang:get_loads_duration_seconds` | /v1/loads 接口响应时间 |

---

## 六、请求分布 (Histogram)

| 指标名 | 说明 |
|--------|------|
| `sglang:prompt_tokens_histogram` | Prompt 长度分布 |
| `sglang:uncached_prompt_tokens_histogram` | 未缓存 prompt 长度分布 |
| `sglang:generation_tokens_histogram` | 生成长度分布 |

---

## 七、请求计数 (Counter)

| 指标名 | 说明 |
|--------|------|
| `sglang:num_requests_total` | 已处理请求总数 |
| `sglang:num_so_requests_total` | 结构化输出请求总数 |
| `sglang:num_aborted_requests_total` | 被中止的请求总数 |
| `sglang:http_requests_total` | HTTP 请求总数 (label: endpoint, method) |
| `sglang:http_responses_total` | HTTP 响应总数 (label: endpoint, status) |
| `sglang:http_requests_active` | 当前活跃 HTTP 请求数 (Gauge) |
| `sglang:routing_keys_active` | 唯一 routing key 活跃请求数 (Gauge) |

---

## 八、Speculative Decoding 指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `sglang:spec_accept_length` | Gauge | 投机解码平均接受长度 (τ，含 bonus token) |
| `sglang:spec_accept_rate` | Gauge | 投机解码接受率 (α，不含 bonus) |
| `sglang:spec_verify_calls_total` | Counter | Verify 调用总次数 |

---

## 九、Retraction (回撤) 指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `sglang:num_retracted_reqs` | Gauge | 当前回撤请求数 |
| `sglang:num_retracted_requests_total` | Counter | 累计回撤请求数 |
| `sglang:num_retracted_input_tokens_total` | Counter | 累计回撤 input tokens 数 |
| `sglang:num_retracted_output_tokens_total` | Counter | 累计回撤 output tokens 数 |

---

## 十、PD 分离 (Disaggregation) 指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `sglang:num_prefill_bootstrap_queue_reqs` | Gauge | Prefill bootstrap 队列请求数 |
| `sglang:num_prefill_inflight_queue_reqs` | Gauge | Prefill inflight 队列请求数 |
| `sglang:num_decode_prealloc_queue_reqs` | Gauge | Decode prealloc 队列请求数 |
| `sglang:num_decode_transfer_queue_reqs` | Gauge | Decode transfer 队列请求数 |
| `sglang:pending_prealloc_token_usage` | Gauge | 预分配 token 使用量 |
| `sglang:kv_transfer_speed_gb_s` | Histogram | KV 传输速度 (GB/s) |
| `sglang:kv_transfer_latency_ms` | Histogram | KV 传输延迟 (ms) |
| `sglang:kv_transfer_bootstrap_ms` | Histogram | KV 传输 bootstrap 时间 |
| `sglang:kv_transfer_alloc_ms` | Histogram | KV 传输分配等待时间 |
| `sglang:kv_transfer_total_mb` | Histogram | KV 传输大小 (MB) |
| `sglang:num_bootstrap_failed_reqs_total` | Counter | Bootstrap 失败请求数 |
| `sglang:num_transfer_failed_reqs_total` | Counter | Transfer 失败请求数 |
| `sglang:num_prefill_retries_total` | Counter | Prefill 重试次数 |
| `sglang:failed_session_recoveries_total` | Counter | Mooncake session 恢复失败数 |

---

## 十一、HiCache / Storage 指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `sglang:hicache_host_used_tokens` | Gauge | Host KV cache 已用 tokens |
| `sglang:hicache_host_total_tokens` | Gauge | Host KV cache 总容量 |
| `sglang:evicted_tokens_total` | Counter | GPU → CPU 逐出 tokens |
| `sglang:load_back_tokens_total` | Counter | CPU → GPU 加载回 tokens |
| `sglang:eviction_duration_seconds` | Histogram | GPU → CPU 逐出耗时 |
| `sglang:load_back_duration_seconds` | Histogram | CPU → GPU 加载耗时 |
| `sglang:prefetched_tokens_total` | Counter | 预取 tokens 数 |
| `sglang:backuped_tokens_total` | Counter | 备份 tokens 数 |
| `sglang:prefetch_pgs` | Histogram | 预取页数分布 |
| `sglang:backup_pgs` | Histogram | 备份页数分布 |
| `sglang:prefetch_bandwidth` | Histogram | 预取带宽 (GB/s) |
| `sglang:backup_bandwidth` | Histogram | 备份带宽 (GB/s) |

---

## 十二、LoRA 指标 (条件启用)

| 指标名 | 说明 |
|--------|------|
| `sglang:lora_pool_slots_used` | LoRA adapter pool 已用 slot |
| `sglang:lora_pool_slots_total` | LoRA adapter pool 总 slot |
| `sglang:lora_pool_utilization` | LoRA pool 使用率 |

---

## 十三、Grammar/结构化输出指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `sglang:num_grammar_cache_hit_total` | Counter | Grammar cache 命中数 |
| `sglang:num_grammar_aborted_total` | Counter | Grammar 中止数 |
| `sglang:num_grammar_timeout_total` | Counter | Grammar 超时数 |
| `sglang:num_grammar_total` | Counter | Grammar 请求总数 |
| `sglang:grammar_compilation_time_seconds` | Histogram | Grammar 编译时间 |
| `sglang:grammar_schema_count` | Histogram | Grammar schema 数量 |
| `sglang:grammar_ebnf_size` | Histogram | Grammar EBNF 大小 |
| `sglang:grammar_tree_traversal_time_avg` | Histogram | Grammar 树遍历平均时间 |
| `sglang:grammar_tree_traversal_time_max` | Histogram | Grammar 树遍历最大时间 |

---

## 十四、CUDA Graph & 调度辅助

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `sglang:is_cuda_graph` | Gauge | 当前 batch 是否使用 CUDA Graph |
| `sglang:cuda_graph_passes_total` | Counter | CUDA Graph forward pass 总次数 |
| `sglang:new_token_ratio` | Gauge | 动态 new_token_ratio 估计值 |
| `sglang:decode_sum_seq_lens` | Gauge | Decode batch 所有序列长度之和 |

---

## 十五、启动与系统指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `sglang:engine_startup_time` | Gauge | 引擎启动耗时 |
| `sglang:engine_load_weights_time` | Gauge | 权重加载耗时 |
| `sglang:startup_available_gpu_memory_gb` | Gauge | 启动时可用 GPU 显存 (GB) |
| `sglang:context_len` | Gauge | 最大上下文长度 |
| `sglang:startup_latency_breakdown_seconds_max` | Gauge | 启动各阶段延迟分解 |
| `sglang:process_cpu_seconds_total` | Counter | 进程 CPU 消耗时间 |

---

## 十六、DP Cooperation 指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `sglang:dp_cooperation_realtime_tokens_total` | Counter | DP 协作模式下处理的 tokens |
| `sglang:dp_cooperation_forward_execution_seconds_total` | Counter | DP 协作模式 forward 执行时间 |

---

## 十七、Expert Parallelism / MoE 指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `sglang:eplb_balancedness` | Summary | MoE expert 负载均衡度 |
| `sglang:eplb_gpu_physical_count` | Histogram | 各层各 GPU 物理 expert 选中次数分布 |

---

## 十八、Prefill Delayer 指标

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `sglang:prefill_delayer_outcomes_total` | Counter | Prefill delayer 结果统计 |
| `sglang:prefill_delayer_wait_forward_passes` | Histogram | Prefill delayer 等待的 forward pass 数 |
| `sglang:prefill_delayer_wait_seconds` | Histogram | Prefill delayer 等待时间 |

---

## 十九、Streaming Session 指标 (条件启用)

| 指标名 | 说明 |
|--------|------|
| `sglang:num_streaming_sessions` | Streaming session 数 |
| `sglang:streaming_session_held_tokens` | Streaming session 持有的 KV tokens |

---

## 二十、Routing Key 分布 (GaugeHistogram)

| 指标名 | 说明 |
|--------|------|
| `sglang:routing_key_running_req_count` | Routing key 按 running 请求数分布 (Grafana heatmap) |
| `sglang:routing_key_all_req_count` | Routing key 按 running+waiting 请求数分布 |

---

## 非 Prometheus 可观测通道

| 通道 | 文件 | 说明 |
|------|------|------|
| ZMQ PUB (per-iteration) | `observability/forward_pass_metrics.py` | 每次 forward pass 推送调度快照 |
| OTLP Traces | `observability/trace.py` | OpenTelemetry 分布式链路追踪 |
| JSON Lines (per-request) | `observability/request_metrics_exporter.py` | 逐请求性能数据写入文件 |
| KV Events PUB | `managers/scheduler_components/kv_events_publisher.py` | KV cache 事件流 |

---

## 关键运维指标推荐

| 场景 | 核心指标 |
|------|----------|
| **延迟 SLA** | `time_to_first_token_seconds`, `inter_token_latency_seconds`, `e2e_request_latency_seconds` |
| **吞吐** | `gen_throughput`, `prompt_tokens_total`, `generation_tokens_total` |
| **内存压力** | `token_usage`, `kv_available_tokens`, `num_retracted_reqs` |
| **队列堆积** | `num_queue_reqs`, `num_running_reqs`, `queue_time_seconds` |
| **Cache 效率** | `cache_hit_rate`, `cached_tokens_total` |
| **PD 分离健康** | `kv_transfer_latency_ms`, `kv_transfer_speed_gb_s`, `num_transfer_failed_reqs_total` |

