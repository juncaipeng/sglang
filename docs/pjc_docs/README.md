# SGLang 技术方案文档索引（docs/pjc_1）

本目录收录对 SGLang 各子系统的深度梳理文档，按主题分为 10 类。每篇都以"由浅入深 + 代码行号引用"的方式写成，
文首一般标注**代码基线 commit**，阅读时请留意基线与当前 `main` 的差异。

| 目录 | 主题 | 篇数 |
| --- | --- | --- |
| [01_overview_startup/](01_overview_startup/) | 全景架构、启动与构建 | 4 |
| [02_scheduling/](02_scheduling/) | 调度循环、批处理与请求控制 | 5 |
| [03_cache_memory/](03_cache_memory/) | 前缀缓存、分层 KV Cache 与存储后端 | 9 |
| [04_pd_disaggregation/](04_pd_disaggregation/) | PD 分离（Prefill-Decode Disaggregation） | 13 |
| [05_speculative_decoding/](05_speculative_decoding/) | 投机解码与 MTP | 5 |
| [06_quantization/](06_quantization/) | 权重与 KV Cache 量化 | 2 |
| [07_parallelism_moe/](07_parallelism_moe/) | 并行策略与 MoE | 5 |
| [08_kernel_attention/](08_kernel_attention/) | Attention 后端、CUDA Graph 与内核 | 4 |
| [09_models/](09_models/) | 模型专项（DeepSeek-V4 / Kimi-K3） | 6 |
| [10_ops_integration/](10_ops_integration/) | 可观测性、路由与外部集成 | 3 |

---

## 01_overview_startup —— 全景架构、启动与构建

先读这一组建立整体心智模型。

- [sglang_layered_architecture.md](01_overview_startup/sglang_layered_architecture.md) —— **推荐入口**。把 `python/sglang/srt/` 拆成 6 个纵向主层 + 4 个横切层，说明一个请求从 HTTP 进来到吐出 token 的全路径。
- [server_startup_flow_v2.md](01_overview_startup/server_startup_flow_v2.md) —— 服务启动流程详解（v2，较新，优先读这篇）。
- [server_startup_flow.md](01_overview_startup/server_startup_flow.md) —— 启动流程初版，保留作对照。
- [sglang_build_install_flow.md](01_overview_startup/sglang_build_install_flow.md) —— 从源码到可运行的完整编译/安装链路（含 sgl-kernel、DeepGEMM 等子模块）。

## 02_scheduling —— 调度循环、批处理与请求控制

- [centralized_scheduling_loop_architecture.md](02_scheduling/centralized_scheduling_loop_architecture.md) —— 集中式（非 PD）部署下 `get_next_batch_to_run` 的决策核心。
- [overlap_schedule_architecture.md](02_scheduling/overlap_schedule_architecture.md) —— 重叠调度：CPU 调度与 GPU 前向的流水线化。
- [prefill_delayer_architecture.md](02_scheduling/prefill_delayer_architecture.md) —— Prefill 延迟器，用于抑制 prefill 抢占导致的 decode 抖动。
- [schedule_batch_to_model_input.md](02_scheduling/schedule_batch_to_model_input.md) —— `ScheduleBatch` → `ForwardBatch` → 模型输入的一次前向数据流全解。
- [abort_interface_implementation.md](02_scheduling/abort_interface_implementation.md) —— Abort 接口的实现方案（请求中止在各队列/各阶段的语义）。

## 03_cache_memory —— 前缀缓存、分层 KV Cache 与存储后端

- [unified_radix_cache_architecture.md](03_cache_memory/unified_radix_cache_architecture.md) —— **本组核心**。`UnifiedRadixCache` 的架构与设计思想（TreeCore / Controller 分层后）。
- [tree_cache_interaction_timeline_mix_p_d.md](03_cache_memory/tree_cache_interaction_timeline_mix_p_d.md) —— 请求处理流程与 radix 树的交互时间线，逐阶段对比集中式 / P 实例 / D 实例三形态。
- [swa_radix_cache_architecture.md](03_cache_memory/swa_radix_cache_architecture.md) —— 滑窗（SWA）混合 KV 的前缀缓存。
- [hicache_usage_and_design.md](03_cache_memory/hicache_usage_and_design.md) —— HiCache 分层 KV Cache（L1 device / L2 host / L3 storage）的使用与实现。
- [hicache_buffer_mode_scheme.md](03_cache_memory/hicache_buffer_mode_scheme.md) —— `--hicache-host-memory-mode buffer_only`：host 内存退化为 GPU↔L3 的一次性中转。
- [host_cache_off_tree_staging_scheme.md](03_cache_memory/host_cache_off_tree_staging_scheme.md) —— Host cache「去树化 + 请求级 staging」改造分析。
- [external_linker_mode_architecture.md](03_cache_memory/external_linker_mode_architecture.md) —— External Linker：显存直连远端 KV 池（GPU-Direct RDMA），砍掉 host L2。
- [mooncake_integration_architecture.md](03_cache_memory/mooncake_integration_architecture.md) —— Mooncake 集成全解（既作 KV 传输后端，也作 L3 存储）。
- [mooncake_standalone_storage_integration.md](03_cache_memory/mooncake_standalone_storage_integration.md) —— Mooncake Store 的 standalone / dummy client 接入方式。

> 相关：DSV4 专属的多层压缩 KV Cache 见 [09_models/deepseek_v4_cache_management.md](09_models/deepseek_v4_cache_management.md)；
> KV Cache 的量化存储见 [06_quantization/kv_cache_quantization_scheme.md](06_quantization/kv_cache_quantization_scheme.md)。

## 04_pd_disaggregation —— PD 分离

### 基础机制

- [pd_disaggregation_architecture.md](04_pd_disaggregation/pd_disaggregation_architecture.md) —— PD 分离总体架构（先读这篇）。
- [pd_disaggregation_kv_transfer_architecture.md](04_pd_disaggregation/pd_disaggregation_kv_transfer_architecture.md) —— KV Cache 跨实例传输架构（ZMQ / Nixl / Mooncake）。
- [pd_true_retraction_rebootstrap_architecture.md](04_pd_disaggregation/pd_true_retraction_rebootstrap_architecture.md) —— 真抢占重引导：D 侧显存压力下的 pause/continue 与重新握手。
- [decode_instance_prefill_capability.md](04_pd_disaggregation/decode_instance_prefill_capability.md) —— D 实例能否承担 prefill？三套事件循环的独立性与能力裁剪点。
- [prefill_load_counter_fix.md](04_pd_disaggregation/prefill_load_counter_fix.md) —— 一个具体缺陷修复：streaming 模式下 P 侧 `load_counter` 减 1 时机过晚。

### D 侧缓存复用（改造方案系列）

- [pd_decode_radix_cache_hicache_scheme.md](04_pd_disaggregation/pd_decode_radix_cache_hicache_scheme.md) —— 常规非 SWA 模型在 D 侧启用 RadixCache + HiCache 的完整方案（本系列最新）。
- [pd_decode_swa_hicache_support_scheme.md](04_pd_disaggregation/pd_decode_swa_hicache_support_scheme.md) —— SWA 模型 + HiCache 的 D 侧支持方案。
- [pd_decode_dsv4_radix_cache_support_scheme.md](04_pd_disaggregation/pd_decode_dsv4_radix_cache_support_scheme.md) —— DSV4 的 D 实例为何不能开 Radix Cache，及改造路径。
- [pd_decode_dsv4_hicache_reuse_scheme.md](04_pd_disaggregation/pd_decode_dsv4_hicache_reuse_scheme.md) —— DSV4 D 侧 HiCache 池化与跨轮复用。
- [pd_decode_output_kv_writeback_l3_scheme.md](04_pd_disaggregation/pd_decode_output_kv_writeback_l3_scheme.md) —— D 侧把输出 token 的 KV 写回 L3，供下一轮 P 实例复用（多轮对话场景）。

### DSV4 专项与调优

- [dsv4_pd_disaggregation_request_lifecycle.md](04_pd_disaggregation/dsv4_pd_disaggregation_request_lifecycle.md) —— DSV4 在 PD 下的请求全生命周期（P 侧 + D 侧逐阶段）。
- [dsv4_pd_p2d_transfer_contents_and_method.md](04_pd_disaggregation/dsv4_pd_p2d_transfer_contents_and_method.md) —— DSV4 的 P→D 到底传了哪些子池、怎么传。
- [mix_vs_prefill_itps.md](04_pd_disaggregation/mix_vs_prefill_itps.md) —— mix 实例模拟 P 实例的 ITPS 差距与调优手段。

## 05_speculative_decoding —— 投机解码与 MTP

- [speculative_decoding_overview.md](05_speculative_decoding/speculative_decoding_overview.md) —— **总纲**，覆盖 `python/sglang/srt/speculative/` 全体算法（EAGLE / NGram / Medusa 等）。
- [mtp_speculative_decoding_architecture.md](05_speculative_decoding/mtp_speculative_decoding_architecture.md) —— MTP（Multi-Token Prediction）方案梳理：参数、draft 模型加载、执行框架。
- [mtp_dsv4_centralized_execution.md](05_speculative_decoding/mtp_dsv4_centralized_execution.md) —— 集中式部署 DSV4 + MTP 的完整执行逻辑（启动 → 三阶段 → 调度 → CUDA graph）。
- [mtp_dsv4_layer_forward_dataflow.md](05_speculative_decoding/mtp_dsv4_layer_forward_dataflow.md) —— MTP draft 层本身的前向计算与数据流。
- [spec_decoding_cache_centralized_vs_pd.md](05_speculative_decoding/spec_decoding_cache_centralized_vs_pd.md) —— 投机解码多出来的那份 Cache：谁建、建多大、如何回收，集中式 vs PD 分离对比。

## 06_quantization —— 量化

- [sglang_quantization_overview.md](06_quantization/sglang_quantization_overview.md) —— **总纲**：四级抽象、注册表、仲裁流程、5 个挂载点、MoE 的第二套分派。
- [kv_cache_quantization_scheme.md](06_quantization/kv_cache_quantization_scheme.md) —— `--kv-cache-dtype` 全路径：FP8 / MXFP8 / FP4 三条独立实现。

## 07_parallelism_moe —— 并行策略与 MoE

- [dp_attention_architecture.md](07_parallelism_moe/dp_attention_architecture.md) —— DP Attention（attention 数据并行 + MoE 张量并行的混合切分）。
- [context_parallel_architecture.md](07_parallelism_moe/context_parallel_architecture.md) —— 两套同名不同机制的上下文并行方案对比。
- [deepep_dispatch_combine_architecture.md](07_parallelism_moe/deepep_dispatch_combine_architecture.md) —— DeepEP all-to-all dispatch/combine 通信层（normal / low-latency 两条路径）。
- [megamoe_architecture.md](07_parallelism_moe/megamoe_architecture.md) —— MegaMoE：DeepGEMM 融合式 EP-MoE 大内核。
- [return_routed_experts_r3_architecture.md](07_parallelism_moe/return_routed_experts_r3_architecture.md) —— R3（`--enable-return-routed-experts`）：抓取并返回每 token 的专家路由。

## 08_kernel_attention —— Attention 后端、CUDA Graph 与内核

- [attention_backend_architecture.md](08_kernel_attention/attention_backend_architecture.md) —— Attention Backend 全景（FlashAttention / Triton / FlashInfer / NPU 等的统一抽象与选型）。
- [cuda_graph_technology.md](08_kernel_attention/cuda_graph_technology.md) —— CUDA Graph 技术全解：捕获、replay、padding、与投机解码/DP 的交互。
- [jit_kernel_architecture.md](08_kernel_attention/jit_kernel_architecture.md) —— 历史版本：`python/sglang/jit_kernel/` 轻量 JIT 内核体系。
- [sglang_kernel_implementation_landscape.md](08_kernel_attention/sglang_kernel_implementation_landscape.md) —— **推荐入口**。按实现语言、编译形态、统一分发和平台后端，全面对比 SGLang 中的各种 kernel 实现方式。

## 09_models —— 模型专项

### DeepSeek-V4

- [deepseek_v4_model_architecture.md](09_models/deepseek_v4_model_architecture.md) —— 模型本体结构与前向计算（压缩器 / 索引器 / CSA-HCA-SWA 分层）。
- [deepseek_v4_cache_management.md](09_models/deepseek_v4_cache_management.md) —— DSV4 独有的多层压缩 KV Cache：六子池布局、地址翻译、HiSparse 换入换出。
- [deepseek_v4_flash_pool_sizing.md](09_models/deepseek_v4_flash_pool_sizing.md) —— DeepSeek-V4-Flash 显存池 per-token 空间构成与实测计算。
- [deepseek_v4_deployment_guide.md](09_models/deepseek_v4_deployment_guide.md) —— 部署参数与完整启动方案。

### Kimi-K3

- [kimi_k3_support_scheme.md](09_models/kimi_k3_support_scheme.md) —— 支持 Kimi-K3 模型的整体方案。
- [kimi_k3_prefix_cache_scheme.md](09_models/kimi_k3_prefix_cache_scheme.md) —— Kimi-K3 的前缀缓存方案。

## 10_ops_integration —— 可观测性、路由与外部集成

- [sglang_metrics_guide.md](10_ops_integration/sglang_metrics_guide.md) —— Prometheus 指标体系（85+ 指标）与排障用法。
- [sglang_router_guide.md](10_ops_integration/sglang_router_guide.md) —— Router / model gateway 使用指南与参数详解。
- [sglang_integration_design.md](10_ops_integration/sglang_integration_design.md) —— RL × SGLang 的服务创建与交互设计。

---

## 阅读路线建议

- **新人上手**：`01` 全部 → `02/centralized_scheduling_loop_architecture.md` → `03/unified_radix_cache_architecture.md`。
- **排查显存 / OOM**：`03/unified_radix_cache_architecture.md` → `09/deepseek_v4_flash_pool_sizing.md` → `05/spec_decoding_cache_centralized_vs_pd.md`（开了投机解码时显存被隐式吃掉）。
- **做 PD 分离**：`04/pd_disaggregation_architecture.md` → `04/pd_disaggregation_kv_transfer_architecture.md` → `03/tree_cache_interaction_timeline_mix_p_d.md`。
- **性能优化**：`02/overlap_schedule_architecture.md` → `08/cuda_graph_technology.md` → `07` 并行相关。

## 维护约定

- 新增文档放进对应类别子目录；不确定归类时，优先按"读者带着什么问题来"而非按代码目录归类。
- 文首标注代码基线 commit，跨文档引用一律用相对路径 markdown 链接，便于本次这类目录调整时批量校验。
- 新增后回到本文件补一行索引（含一句话说明）。
