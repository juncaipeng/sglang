# DeepSeek-V4 Flash：PD 分离下 1M 多轮使用、功能设计与优化探索

> 目标对象：`DeepSeek-V4-Flash-FP8`，`model_type=deepseek_v4`，NVIDIA GPU，Prefill/Decode（PD）分离，约 1M token 的长历史多轮会话。
>
> 本文严格区分**当前态（AS-IS）**与**目标态（TO-BE）**。结论来自当前仓库静态读码、本地模型配置和公式推导；尚未完成真实 NVIDIA、Mooncake 或 1M 端到端压测。

## 0. 执行摘要

目标不是“打开所有优化开关”，而是建立一条可解释、可恢复的长文多轮链路：首轮 1M Prefill 可完成；后续轮在稳定历史命中时，Prefill 工作和 PD bytes 主要随新增 token `Δ` 而非总长度 `N` 增长；Decode 能复用已提交的多池状态；任何不一致都能回退完整 Prefill。

当前建议如下：

1. **先以 Profile A 建立事实基线。** 不启用 HiSparse，保留模型原生 ratio-4/128 DSA，校准正确性、容量、Indexer、MoE 和约 3.85 GB 的主要 PD payload。
2. **仅在 Decode 容量不足时评估 Profile B。** 它必须是 Prefill HiSparse off、Decode HiSparse on、Mooncake、Decode radix off、`DCP=1`；它是容量实验，不是跨轮复用方案。
3. **现有 Profile A/B 都没有完整跨轮增量闭环。** 当前已有 chunk/page early send，但没有同时打通 Multi-pool Prefix、增量 Prefill、PD existing-page skip 和 Decode 原子 Attach。
4. **先建设可观测性和故障安全。** HiSparse retraction 的 CPU copy 路径存在 `NotImplementedError` 风险；在恢复完成前，应以保守 admission 和 fail-closed 避免 scheduler 退出。
5. **再由 profiler 决定计算专项。** Dense Indexer、MoE A2A、PD 非重叠尾部、H2D 或小 `Δ` 固定开销都可能主导。近似检索会改变候选集合，必须接受质量权衡验收。

不可混淆的当前事实：

- **[代码事实] DSA 不等于 HiSparse。** ratio-4/128 压缩注意力是模型结构；`--enable-hisparse` 只是 C4 GPU/Host 分层存储。
- **[代码事实]** HiSparse 要求显式关闭 radix；HiCache 又与关闭 radix 配置级互斥。
- **[代码事实]** HiSparse 的 `SWAChunkCache` 只支持同一未完成请求的 chunk 续算，没有跨请求 Prefix 命中。
- **[代码事实]** DSV4 HiSparse PD 当前只应采用 P-off/D-on；Prefill 开启 HiSparse 的源端物理映射没有进入当前传输协议。
- **[理论推导]** 特定 pool-sizing 假设下约 `7705.45 B/full-token`，主要 PD 线性 payload 为 `3850.25 B/full-token`；二者不是同一口径。

本文实施主线为：事实基线与观测 → 容量与故障安全 → 同会话多池增量 MVP → HiSparse 分层与恢复 → profiler 驱动的计算与传输优化。

# 第一部分：背景与目标

## 1. 背景、范围与术语

### 1.1 问题背景

1M 长历史多轮同时放大三类成本：首轮 Prefill 计算与临时张量、Prefill 到 Decode 的多池传输、Decode 侧历史 KV 的长期驻留。若第 2+ 轮仍重新提交完整历史而执行层无法复用，成本会再次随 `N` 增长；只解决显存、计算或网络中的一项，不能形成端到端收益。

### 1.2 四层能力与核心术语

| 层次/术语 | 含义 | 本文边界 |
|---|---|---|
| DSA | 模型原生稀疏注意力，按 ratio-4/128 生成并检索压缩历史 | 不依赖 HiSparse 开关 |
| HiSparse | C4 KV 以 Host 为主体、GPU 保留 working set 的物理分层 | 当前要求 Decode radix off |
| radix/Prefix cache | 按 token 前缀匹配并复用已计算状态 | 当前 DSV4 PD 多池尚无完整闭环 |
| HiCache | 基于 radix 的层次缓存能力 | 不能直接给 radix-off HiSparse 兜底 |
| Multi-pool Prefix | 同一稳定历史的 SWA、C4 KV/index/state、C128 KV/state 的事务性集合 | [目标设计]，不是单个 `prefix_len` |
| PD Manifest | 描述页身份、位置、状态、完整性和版本的传输契约 | [目标设计] |
| Attach | Decode 将完整且兼容的多池 Prefix 绑定到请求 | [目标设计] 采用 Prepare/Commit |
| early send | Prefill 在 page-aligned segment 就绪后提前发送 | [代码事实] 当前已有 chunk/page 粒度 |

### 1.3 范围与证据标签

本文覆盖纯语言 `DeepSeek-V4-Flash-FP8` 的 NVIDIA PD 场景，不覆盖 D=5120、带视觉/Engram/ratio-1/2 的 V4.1，不覆盖 DSpark+ratio-2 专用路径，也不把 AMD unified KV 性能结论外推到本场景。

全文统一使用：

- **[代码事实]**：当前源码或模型配置可直接证明；
- **[理论推导]**：由 shape、dtype、ratio、容量或复杂度公式推得；
- **[待实验]**：理论或代码路径允许，但收益、稳定性或参数尚待目标集群验证；
- **[目标设计]**：本文提出、当前尚未完整实现的能力。

## 2. 模型与负载基线

### 2.1 模型配置

| 项目 | 目标值 | 对 1M PD 的影响 |
|---|---:|---|
| `hidden_size` | **4096** | mHC 单流宽度 |
| `hc_mult` | 4 | 残差流 `[T,4,4096]`，PP 边界可展平为 `[T,16384]` |
| 主干层数 | **43** | 主模型 KV、计算和容量统计范围 |
| NextN 层数 | 1，独立计费 | 配置中额外第 44 个 ratio，不是普通主干层 |
| 主干 ratio | **`0×2, 4×21, 128×20`** | 决定 SWA、C4、C128 子池规模 |
| SWA window | **128** | 决定近窗 KV 和续算边界 |
| C4 top-k | 512 | 决定稀疏 attention 候选数 |
| 最大位置 | 1,048,576 | 配置上限不等于 1M 已通过质量和稳定性验收 |
| 权重量化 | FP8 block `[128,128]` | 不等于 KV cache 布局 |

配置和逐层 ratio wiring 见 [deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py)。`compress_ratios` 共 44 项，本文核心公式只统计前 43 个主干层。

### 2.2 多轮负载模型

第 `k` 轮定义：`N_k` 为本轮总输入长度，`Δ_k` 为新增 token，`H_k=N_k-Δ_k` 为理论稳定历史，`O_k` 为输出 token；`C_session` 为活跃会话数，`R_p/R_d` 为 P/D 副本数，`BW_pd` 为有效 PD 带宽，`P_stable` 为应用稳定前缀比例，`P_hit` 为经过 token、模型、布局、路由和 residency 校验后的实际命中比例。

| 场景 | 历史 `H_k` | 新增 `Δ_k` | 验证重点 |
|---|---:|---:|---|
| 首轮长文导入 | 0 | 1M | 完整 Prefill、容量与 PD |
| 长历史小增量 | 约 1M | 128～1K | 跨轮复用最大价值 |
| 长历史中增量 | 约 1M | 8K | 增量 Indexer、PD、边界状态 |
| 持续增长会话 | 128K→1M | 1K～8K/轮 | 生命周期、驱逐、准入 |

API 还必须明确：每轮是否重交完整 token、是否有稳定 `session_id`、能否固定到同一 Decode、副本迁移规则、历史是否编辑、模型版本和 prompt template 是否变化。

## 3. 目标、成功标准与非目标

### 3.1 目标与成功标准

1. 128K/512K/1M 首轮可完整结束，无 NaN、transfer failure 或多池校验错误。
2. 稳定历史命中时，Prefill 工作、PD bytes 和 TTFT 的主要增量随 `Δ_k` 而非 `N_k` 增长。
3. Prefix miss、驱逐、版本不兼容、Decode 迁移或部分失败时，自动回退完整 Prefill。
4. 容量不足在 admission 阶段排队、换路由或拒绝；retraction 后可恢复或受控失败，不能退出 scheduler。
5. P50/P95/P99 TTFT、ITL、完成率和 SLO-goodput 可由分阶段指标解释。

### 3.2 非目标

近期不承诺通用跨会话内容去重、立即替换通用 sparse-aware radix、不指定 ANN 算法、不承诺把全流程从二次复杂度降为线性，也不把 Blackwell candidate 路径外推到 Hopper。CP、TBO、Graph、MTP、算子融合和近似检索都必须由 profile 与验收决定，不能作为默认收益写入目标。

# 第二部分：当前现状

## 4. 当前请求生命周期

### 4.1 首轮完整链路

| 阶段 | 当前动作 | 当前状态 |
|---|---|---|
| 接入 | Router/Prefill 接收 token、输出上限和元数据 | 已有 |
| 分块 Prefill | chunk 执行 43 层，生成多池数据 | 已有 |
| 多池产出 | SWA、C4 KV/index、C128 KV、边界 state | 已有 |
| Early send | page-aligned segment 提前发送，最终 segment 带 state | 已有 |
| Decode 预分配 | 分配各池目标位置 | 已有；HiSparse 有后端限制 |
| 多池接收 | 接收 C4 KV/index、C128 KV 和最终 state | 已有 |
| Attach/排队 | 完整请求在 state 到达后进入 waiting/running | 已有当前全请求路径 |
| Decode | 生成 token，增长各池状态 | 已有 |
| 释放/退避 | 完成后释放，或内存压力触发 retraction | HiSparse 恢复不完整 |

普通 V4 主线性传输由 [DeepSeekV4TokenToKVPool.get_contiguous_buf_infos()](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1177-L1245) 注册：21 个 C4 KV、21 个 C4 index、20 个 C128 KV。SWA、ring、compressor/indexer state 和 request-scoped C128 state 由 [setup_state_kv_args()](../../../python/sglang/srt/disaggregation/utils.py#L1313-L1431) 处理；PP ratio 切片见 [_mla_slice_ptrs_for_pp()](../../../python/sglang/srt/disaggregation/common/conn.py#L1288-L1354)。

**[代码事实]** 当前并非完全串行，已有 chunk/page early send。第 18 章讨论的是在此基础上的更细 layer-group producer。

### 4.2 当前第 2+ 轮

Profile A 允许普通 DSA 多池工作，但 DSV4 compressed KV 的 PD Decode radix 尚未完整接入；Profile B 关闭 radix，注册的是无跨请求命中的 `SWAChunkCache`。[chunk_cache.py](../../../python/sglang/srt/mem_cache/chunk_cache.py) 的 `match_prefix()` 返回空，`insert()` 不建立可匹配节点，`cache_unfinished_req()` 仅支持同一未完成请求跨 chunk 续算。

因此两个 Profile 都可能在第 2+ 轮重复执行：

```text
完整 N_k token 进入 Prefill
  -> 历史重新经过主干和 Indexer
  -> 历史多池数据重新生成或组织
  -> 约 N_k × 3850.25 B 的主线性数据再次进入 PD
  -> Decode 重新预分配和绑定
```

**[代码事实]** 当前 A/B 均没有“Prefill skip + PD skip + Decode attach”的完整跨轮增量闭环。

### 4.3 当前内存压力路径

普通 Decode 在 `check_decode_mem()` 后可能进入 `retract_decode()`。对 DSV4 HiSparse，默认 CPU backup 最终调用未覆盖的 allocator CPU copy 接口：

```text
check_decode_mem() -> retract_decode() -> release_req(offload_kv=True)
 -> retraction_backup("cpu_tensor") -> allocator.get_cpu_copy()
 -> NotImplementedError
```

[BaseTokenToKVPoolAllocator.get_cpu_copy/load_cpu_copy](../../../python/sglang/srt/mem_cache/allocator/base.py#L223-L229) 未实现，DSV4 HiSparse allocator 未覆盖；本模型也不满足 [dsv41_dspark_needs_rebootstrap()](../../../python/sglang/srt/mem_cache/common.py#L197-L207) 的 DSpark+ratio-2 条件。该路径是上线阻塞风险。

## 5. 当前能力与部署 Profile

### 5.1 权威能力地图

| 能力 | Profile A：普通 DSA+PD | Profile B：P 普通池+D HiSparse | 含义 |
|---|---|---|---|
| ratio-4/128 DSA | 支持 | P/D 支持 | 模型能力，不由 HiSparse 决定 |
| Chunked Prefill | 支持 | 支持 | 降峰值，不改变 Indexer 渐近工作量 |
| Chunk/page early send | 支持 | 支持 | 可能仍有 final-state 尾部 |
| 普通多池 PD | 支持 | P 连续源池，D Host/Device 混合目标 | 各池和 state 必须一致 |
| 跨请求 Prefix | DSV4 PD Decode radix 未完整支持 | 不支持 | A/B 均无完整增量闭环 |
| C4 Host 驻留 | 无 | 仅 Decode | 缩小 D 侧 GPU working set |
| HiCache | 可独立研究 | 与 radix-off 互斥 | 不能给 B 直接兜底 |
| Decode retraction | 需验证 | CPU copy 未实现 | B 的阻塞风险 |
| Async decode offload | DSV4 不支持 | DSV4 不支持 | 不能替代 backup |

### 5.2 Profile A：普通 DSA + PD 基线

**定义：** P/D 均不开 HiSparse，保留模型原生 ratio-4/128 DSA。它是正确性、容量、网络和性能的首选基线，也是当前更稳妥的生产候选起点。

不默认加入 CP、TBO、MTP、Decode radix 或 DSV4 不支持的 async offload。先测 Indexer、attention、MoE、PD 和 Decode 各阶段，再做单变量实验。完整启动命令见附录 A。

### 5.3 Profile B：Decode HiSparse 容量实验

**权威定义：**

- Prefill：**HiSparse off**，保持连续 C4 源池；
- Decode：**HiSparse on**，显式 **radix off**；
- backend：必须 **Mooncake**；
- Decode：必须 **`--dcp-size 1`**；
- C4 KV 落 Host，C4 index/C128 落 Device；
- 不组合 HiCache、Decode radix、DSV4 async offload 或 Decode DCP relayout。

源池构造与映射见 [deepseek_v4_memory_pool.py](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py) 和 [allocator/hisparse.py](../../../python/sglang/srt/mem_cache/allocator/hisparse.py)；Prefill 注册见 [prefill.py](../../../python/sglang/srt/disaggregation/prefill.py)，Mooncake 特殊索引用于 Decode 目的端，见 [mooncake/conn.py](../../../python/sglang/srt/disaggregation/mooncake/conn.py)。当前协议未携带 Prefill HiSparse 的逻辑到物理映射，所以 P-on/D-off 和 P-on/D-on 均不应使用。

Profile B 只在普通 Decode 多池无法满足并发容量时评估。它不是跨轮 Prefix 方案，在 Host pool、H2D 和 retraction 故障注入通过前，不作为默认生产推荐。完整命令见附录 A，组合边界见附录 C。

### 5.4 选型

```text
普通多池能否容纳目标 1M × 并发？
  ├─ 能：Profile A，建立事实和可靠性基线
  └─ 不能：是否具备 Mooncake、大 Host pool，并接受实验性恢复？
       ├─ 是：Profile B，仅做容量与故障实验
       └─ 否：降低并发/输出上限，先开发 admission 与恢复

业务是否要求第 2+ 轮只处理新增 token？
  ├─ 是：现有 A/B 都不满足，需要第 8～18 章的目标能力
  └─ 否：按首轮容量和 SLO-goodput 选型
```

## 6. 容量、传输与计算基线

### 6.1 三种口径

1. **逻辑/预留池容量**：用于 HBM/Host 规划，受 SWA ratio/cap、HiSparse shrink、PP、MTP、page 和并发影响。
2. **PD wire payload**：实际发送数据，受 prefix skip、padding、state、重试影响。
3. **运行 working set**：实时 GPU/Host 驻留、activation、Indexer scratch 和 H2D buffer。

### 6.2 权威关键数字

普通 packed V4 KV 每个压缩 token、每层为 **584B**；NVIDIA 默认 C4 index key 为 **132B**。详细 shape/dtype 推导见附录 B。

| 数据 | B/full-token |
|---|---:|
| C4 KV：`21×584/4` | 3066.00 |
| C4 index：`21×132/4` | 693.00 |
| C128 KV：`20×584/128` | 91.25 |
| **主要 PD 线性 payload** | **3850.25** |

**[理论推导]** 1M 的主要线性 payload 约 3.85 GB（3.59 GiB）；加最终 SWA/page 和 state 后，单请求约 3.87 GB 量级。特定“43 层、584B、132B、`swa_full_tokens_ratio=0.1`、C4 ring=8、SWA page=128、state FP32、无 HiSparse/PP/MTP”假设下，pool sizing 为 **7705.45 B/full-token**。两者不可混用。

### 6.3 当前多轮成本与 Indexer 事实

首轮和当前无命中轮的主要 PD bytes 分别近似 `N_1×3850.25B` 与 `N_k×3850.25B`；理想增量轮才接近 `Δ_k×3850.25B` 加变更 state 和 partial page。完整公式见附录 B。

[dense_prefill_topk()](../../../python/sglang/srt/layers/attention/dsv4/dense_prefill_indexer.py#L22-L89) 对 `Q` 个 query 和压缩 context `C` 产生全宽 logits，单 chunk 为 **`O(Q×C)`**；增长到 N token 的累计工作近似 `O(N²/4)`。`_SCORE_BUDGET_BYTES=2<<30` 只限制 logits materialize 峰值，不减少总 score FLOPs。当前 `_select_tile()` 仍先产生全宽 logits，再发布候选块或 mask，不能描述成已完成块级粗排。

即使未来 Prefix 命中，新增 query 仍检索历史，C4 项近似：

$$
O(\Delta_kH_k/4+\Delta_k^2/4)
$$

高命中后，SWA 只看 128，C4 sparse attention 只看 top-k=512，但 C4 dense score 扫描 `H/4`，C128 attention 扫描 `H/128`。因此 PD 增量化不等于 Prefill 计算完全与 `H` 解耦。

### 6.4 时间与 goodput 下界

$$
T_{xfer,min}=\frac{B_{wire}}{BW_{pd}}
$$

3.85025GB 在 25/50/100GB/s 下的理想下界约 154/77/39ms，实际还包括注册、同步、page、拥塞、重传和非重叠尾部。

$$
G_{req}\le\min\left(\frac{R_p}{T_p},\frac{BW_{net}}{B_{wire}},\frac{R_d}{T_d}\right)
$$

该上界未扣除 queue、miss、retraction、PP bubble，以及 PD 与 EP A2A 的网络竞争。HiSparse H2D 必须单独计量。

# 第三部分：难点与总体方案

## 7. 核心难点与 Gap

### 7.1 多池一致性而非单 KV 命中

SWA tail/ring、C4 KV、C4 index、compressor/indexer state、C128 KV 和 request state 必须在同一 token 边界、generation 和布局下完整可用。只命中部分页会造成静默错误，因此 Prefix 必须是事务对象。

### 7.2 容量与复用机制冲突

普通多池语义较完整，但 1M×并发带来 HBM 压力；HiSparse 把 C4 主体放 Host，却要求 radix off；HiCache 又依赖 radix。当前没有现成路径同时获得 C4 Host 分层与 DSV4 多池跨轮 Prefix。

### 7.3 增量写入仍需全历史检索

Prefix 可消除历史 Transformer 重算，但不能自动消除新增 query 对 `H/4` 和 `H/128` 的检索。Chunked Prefill 控制峰值和公平性，不降低 `O(ΔH/4)` 总 FLOPs。若 Indexer 主导，必须另做精确结构优化或带质量权衡的近似检索。

### 7.4 Early send 不等于可消费

当前页可提前发送，但 Decode 只有在所有必要子池、有效长度和最终 state 一致时才能消费。更细 producer 不能绕过最终原子 commit。

### 7.5 利用率与恢复能力相互制约

激进 admission 提升并发，也增加 retraction 概率；HiSparse 当前 backup 不完整。恢复能力完成前，正常目标负载应尽量零 retraction，超限时 fail-closed。

### 7.6 多类资源竞争

PD、EP A2A、TP/CP collective、Host→Device、GPU compute 和 Host pinned memory 可能互相竞争。必须拆分观测后再调拓扑，不能先验认定网络或 Indexer 是唯一瓶颈。

## 8. 目标架构与总体方案

### 8.1 设计原则

**[目标设计]** 三条一致性原则：Prefix 是多池事务；Committed Prefix 不可变、当前轮尾部可变；传输完成不等于 Attach 完成。任何兼容性或完整性失败都回退已验证的 full Prefill。

### 8.2 六个逻辑组件

| 组件 | 职责 | 当前代码接缝 |
|---|---|---|
| Session/Prefix Directory | 按会话、token fingerprint、模型和布局定位稳定 Prefix | radix/chunk cache、router affinity |
| Multi-pool Prefix Store | 统一管理 SWA、C4、C128 和 state | DSV4 pool、allocator |
| Incremental Prefill Planner | 求安全命中边界、重算尾部和新增页 | scheduler、compressor/indexer state |
| PD Manifest/Coordinator | 描述已有、缺失、新增页和完成状态 | P/D transfer metadata |
| Decode Attach Manager | Prepare 容量与布局，Commit 原子绑定 | prealloc/transfer/waiting queue |
| Admission/Recovery Controller | 容量决策、snapshot、restore、rebootstrap、fallback | allocator、retraction、scheduler |

控制面负责 identity、generation、lease、容量和 commit；数据面负责各池页、state 和 completion bitmap。控制面不得在数据面不完整时发布可消费 Prefix。

### 8.3 依赖顺序

可观测性先于所有优化；Multi-pool Prefix 先于增量 Prefill、Manifest 和 Attach；完整增量 MVP 先于 HiSparse 分层和 Prefix-aware 恢复；profiler 结果先于 Indexer、MoE、Graph、MTP 或 Producer 专项。

## 9. 目标增量生命周期

**[目标设计]** 稳定会话第 `k` 轮目标链路为：

```text
(session_id, tokens, model/layout version)
 -> Prefix lookup + token fingerprint 校验
 -> 找到 committed H_k 边界并获取 lease
 -> Planner 决定复用范围、边界重算和新增 Δ_k
 -> 增量 Prefill 生成新增页与更新 state
 -> Manifest 标记已有、缺失、新增页
 -> Decode Prepare 校验容量/布局并返回目标页
 -> 沿用现有 early send，只补传缺失/新增内容
 -> final state + completeness 校验
 -> Decode Commit 原子 attach
 -> Decode 后发布新 generation
```

若 token 编辑、版本变化、容量不足、页丢失无法补传或 state 不一致：Commit 前释放本轮临时页和 lease，旧 Prefix 保持不变，请求回退 full Prefill。只有 Prefill skip、PD skip 和 Decode Attach 三者同时成立，成本才主要随 `Δ_k` 增长。

# 第四部分：功能设计

## 10. 可观测性与事实基线

### 10.1 要解决的问题

没有统一观测就无法回答“第 `k` 轮为何仍处理 1M”：可能是 token 不同、Prefix 未发布、布局不兼容、Decode 无 residency、边界 state 不安全、容量不足或主动 fallback。可观测性是阶段 0 功能，也是 admission、恢复和性能优化的共同依赖。

### 10.2 数据与指标契约

标签至少包含 request/session、prefix generation、`pool_type`、`tier`、`phase`、model/layout version：

| 维度 | 指标 |
|---|---|
| 多轮 | `N/H/Δ`、matched/skipped token、miss/fallback reason |
| Prefill | chunk/layer 时间、Indexer Q/C/scratch、C4/C128、mHC、MoE、collective |
| 计算 | shape、dtype、FLOPs/bytes、kernel/launch gap、Graph replay、GPU time/新增 token |
| MoE | expert histogram、GEMM shape、dispatch/combine/A2A、不均衡 |
| Decode | dense refresh、candidate reuse、MTP accepted tokens/step、扫描次数 |
| PD | 各池 actual/skipped/retry bytes、queue、GB/s、non-overlap tail |
| Cache | logical/device/host pages、lease、eviction、partial tail |
| HiSparse | Host pool、GPU working set、H2D pages/time/P99 |
| Attach/恢复 | prepare/commit、checksum、rollback、admission 误差、backup/restore |
| 请求 | TTFT/ITL/E2E、完成率、SLO-goodput |

每个 Prefix 事件还需记录 generation、committed length、per-pool valid rows、completion bitmap 摘要和原因码。日志应支持从请求反查页事务，但避免逐页高基数指标淹没监控系统。

### 10.3 失败与 MVP

指标写入失败不得阻塞推理；trace 可采样，但容量、fallback、retraction 和一致性错误不可采样丢失。MVP 先覆盖 Profile A 首轮/第 2 轮/OOM，再覆盖 Profile B Host/H2D/retraction；验收统一见第 21 章。

## 11. Multi-pool Prefix 对象

### 11.1 数据对象

**[目标设计]** Prefix Manifest 至少包含：

| 类别 | 字段/含义 |
|---|---|
| 身份 | `prefix_id`、session/content fingerprint、token fingerprint |
| 兼容性 | model revision、KV dtype、DSA layout、page size、TP/PP/CP、layout version |
| 范围 | full-token length、各压缩池有效长度、完整页和 partial tail |
| SWA | tail window 页、ring、续算 state |
| C4 | KV 页、index 页、compressor/indexer 边界 state |
| C128 | KV 页、request-scoped state |
| 所有权 | tier/replica、refcount 或 lease、generation |
| 完整性 | per-pool completion bitmap、checksum、commit epoch |

### 11.2 状态与一致性

```text
Building -> Committed -> Leased/Attached
         -> new generation Building
Committed -> Evicting -> Evicted/Invalid
```

`Building` 不对普通读者可见；`Committed` 内容不可变；新轮采用 copy-on-write 构建新 generation；`Evicting` 停止新 lease，已有使用者退出后再释放。model/layout/token 任一不匹配时整体 miss，禁止混用不同 generation 子池。

ratio/page 不对齐尾部不可共享写：C4 需保留 overlap 边界，C128 未满组需保留 pending request state，SWA 需保存最近 128 token 及 ring。实现可选择重算边界或 copy-on-write，但必须保证旧 generation 不被污染。

### 11.3 失败降级与 MVP

缺池、checksum 错误、lease 过期或版本不兼容均整体 miss；不能“尽量复用”部分 KV。MVP 仅支持同一会话、严格稳定前缀和优先固定路由，不做任意跨会话去重；先证明多池一致性，再扩大共享。

## 12. 增量 Prefill

### 12.1 输入、规划与输出

**[目标设计]** 输入为 Committed Prefix、本轮 token、新 generation、模型和拓扑；Planner 先求最长安全稳定边界，再决定历史复用、partial-tail 重算和 `Δ` forward 范围。输出包括新增 C4 KV/index、C128 KV、更新的 SWA tail/ring、compressor/indexer/request state，以及供第 13 章使用的页依赖描述。

命中后历史 token 不进入本轮 EXTEND tensor；`seq_lens` 仍描述完整序列，实际 forward 只放未匹配 token。调度语义接缝见 [schedule_batch.py](../../../python/sglang/srt/managers/schedule_batch.py) 与 [forward_batch_info.py](../../../python/sglang/srt/model_executor/forward_batch_info.py)。

### 12.2 边界与一致性

历史 token/template 有任何编辑时强制 miss；每个压缩 ratio 的 pending state 必须能从边界续算；partial page 采用重算或 copy-on-write。full 与 incremental 的 next-token logits、输出、多池有效长度和 checksum 必须在允许误差内一致。

### 12.3 MVP 与后续

MVP 支持 `Δ=128/1K/8K`、单会话 generation 和固定 Decode affinity；不要求消除新增 query 的全历史 Indexer 工作。若 state 不可续算或规划失败，直接 full Prefill，不发布半成品 Prefix。

## 13. PD Manifest 与增量传输

### 13.1 Manifest 契约

**[目标设计]** 每个传输事务携带 request、prefix/generation、model/layout version、pool type、layer/stage、token/page range、source/destination tier 和 page、valid rows、checksum、completion bitmap、final-state 标志和 commit epoch。

内容分三类：Decode 已有且兼容的页只传引用；缺失或新增页沿用 chunk/page early send；最终 state/commit 消息携带有效长度、state 和完整性摘要。PP 只发送本 stage 的 ratio 层；TP/DCP 目标布局无法表达时拒绝事务，不静默解释。

### 13.2 幂等与失败

同一 generation/page 重试必须幂等；接收端按 bitmap 去重并补传。超时可重试缺页，checksum 错误需重传或整体 fallback；取消时释放未提交目标页。旧 committed Prefix 在新事务成功前始终可用。

### 13.3 MVP

第一版只做同 Decode 副本的 existing-page skip、新增页补传和 final commit，不做任意远程 Prefix 拉取。实际 bytes 应接近 `Δ×3850.25B` 加 changed state/partial page；该数字是验收参考，不是硬编码协议常量。

## 14. Decode 两阶段 Attach

### 14.1 Prepare

**[目标设计]** Decode lookup Prefix 并获取 lease，校验 generation/layout，计算已有/缺失页，使用与 allocator 相同的 page rounding 预留缺失页、输出 headroom 和 working set，然后返回可复用长度、目标页和事务 ID。Prepare 不修改请求可见映射。

### 14.2 Commit

仅当所有必要池和最终 state 到达、有效长度匹配、checksum/epoch 通过时，原子更新请求的多池映射和 committed length，随后进入 waiting/running。Attach 结果要么全有，要么全无。

### 14.3 状态与异常

```text
Lookup -> Preparing -> Receiving -> ReadyToCommit -> Attached
                   \-> Aborting -> RolledBack/FullPrefill
```

Prepare 容量不足可 defer、换副本或 full Prefill；部分传输按 bitmap 补传；generation 变化重新 lookup；Commit 前取消释放临时页和 lease；Commit 后故障按第 16 章恢复。MVP 不支持跨布局迁移，也不允许部分 Prefix 命中。

## 15. HiSparse 与 Prefix 分层

### 15.1 目标

当前 HiSparse 将 C4 主体放 Host、GPU 保留 working set，但 radix-off 后没有跨请求 Prefix。**[目标设计]** 应将逻辑 Prefix ownership 与物理 GPU residency 解耦，而不是简单改 `ChunkCache.match_prefix()`。

分层节点关联 token span、parent/generation、C4 Host 逻辑页、C4 index/C128/state、SWA tail snapshot、GPU hot-page hints、lease、last access 和 Host/Device eviction 状态。GPU working set 是缓存，不是 Prefix 所有者。

### 15.2 生命周期与一致性

驱逐先撤销 GPU residency，再在无 lease 时释放 Host 页；H2D 期间禁止释放源页。Host C4、Device index 和 state 必须属于同一 generation。Attach 前可按候选/hotness 预取，但 H2D 未完成的页不可消费。

### 15.3 分阶段范围与降级

先做精确 `session_id`、token 校验和固定 Decode affinity；再支持严格相同的公共前缀；最后才研究通用 sparse-aware radix/HiCache 和 Storage tier。任何 Host 缺页、H2D 超时或 tier 不兼容都回退已有普通路径。P99 H2D 若抵消容量收益，则降低并发、提高 GPU working set 或停用分层。

## 16. Admission、Retraction 与恢复

### 16.1 容量感知 Admission

**[目标设计]** Admission 与 allocator 共用 page rounding 和 pool 口径，输入包括已 Attach Prefix、本轮新增和最大输出、SWA/C4/C128 新增与尾页、Host/GPU working set、MTP/verify 槽位、PP-local ratio、reserved token、headroom、transfer/waiting queue 和 Host pinned pool。

输出为 `accept`、`queue/defer`、`route-to-other-replica` 或 `reject`。必须记录 estimated/actual peak、误差、等待时间和“通过后仍 retraction”次数。在完整 backup 前，正常目标负载要求零意外 retraction。

### 16.2 Fail-closed 底线

**[目标设计]** 若 pool/layout 不支持 backup，retraction 应显式中止、迁移或 rebootstrap 单请求并返回诊断错误；不能让未捕获 `NotImplementedError` 终止 scheduler。该能力只是安全底线，不代表恢复完成。

### 16.3 多池快照与 Restore

快照必须事务性覆盖 SWA/ring、C4 KV/index/compressor/indexer state、C128 KV/request state、committed length、token/page mapping、model/layout/generation/checksum。不能只补 `get_cpu_copy()`；Restore 全部校验完成前不得回 running。

Rebootstrap 是故障兜底，不是正常调度。1M 全量重算/重传可能形成恢复风暴，应限流并排队；若有 committed Prefix，优先从该边界恢复。MVP 顺序为保守 admission → fail-closed → Prefix-aware rebootstrap → 完整 snapshot/restore。

## 17. 计算路径与 Indexer 优化

### 17.1 算子账本与决策原则

按 full Prefill、incremental Prefill、Decode 三种模式记录每层/每 chunk 的 token、历史宽度、shape、dtype、FLOPs/bytes、kernel 和 collective：mHC/Sinkhorn，Q/KV/Indexer 投影与动态量化，C4/C128 compressor/state，dense score/top-k/candidate/sparse attention，C128 attention，MoE router/dispatch/GEMM/combine，norm/残差/head/sampling，以及 metadata/page table/Graph/CPU gap。

每项同时报告 kernel、stage 和 E2E speedup，以 TTFT、ITL、accepted tokens/s、SLO-goodput 和每新增 token GPU time 决策。

### 17.2 高命中后的计算分解

Prefix 命中消除 `H` token 的 embedding、43 层 mHC、投影、compressor、attention、MoE、norm 和输出工作，但保留新增 query 对历史 cache 的检索。剩余近似为：

$$
O(\Delta\times\text{Transformer})+O(\Delta H/4)+O(\Delta H/128)
$$

按阶段优化：输入/metadata 复用；小 `Δ` mHC、投影和逐 token 算子合批/融合；C4 overlap 只更新尾部；C4 score 做 streaming top-k；C4 candidate/page 排序与 gather-attention 融合；C128 tiled online-softmax；MoE 跨会话合批和 grouped/persistent GEMM；只对必要位置执行 LM head。

**[待实验]** NVIDIA 默认路径不能因 FP8 block 权重就假设 norm 与激活量化已融合，应实测 `bf16 norm -> dynamic quant -> FP8 linear` 后再决定 fused norm/quant/linear。

### 17.3 精确优化层次

P0 先证明多池命中和续算边界正确。P1 复用 compressed slot mapping/page table，融合 score 与 streaming top-k，使用固定 bucket、Graph 和跨请求合批，并做 Host page 预取。P2 再研究跨层候选共享、History/K 维并行的 local-top-k merge、可证明的 block upper-bound pruning，以及 C128 分块执行。

| 并行手段 | 小 `Δ`、大 `H` 判断 |
|---|---|
| Chunked Prefill | 控内存/公平性，不减总 score FLOPs |
| Query-CP | `Δ` 太小时利用率可能不足 |
| History/K-CP | 更匹配小 Q 大 K，但需 local-top-k merge |
| TP/PP | 权重容量与流水 bubble 权衡 |
| EP | 依赖跨会话合批，并与 PD 竞争网络 |
| TBO | microbatch 太小可能退化 |
| CUDA Graph | 降固定开销，不改复杂度 |

### 17.4 近似检索与质量权衡

**[目标设计][待实验]** 当 exact 路径仍不满足 SLO，可研究 block summary、IVF/PQ、HNSW、聚类、近历史 exact+远历史 approximate、跨 decode step 候选复用和周期性 full refresh：

```text
历史压缩 token -> block summary/index -> query 粗排
 -> 候选 block -> token 精排 -> 低置信度 dense fallback
```

Block metadata 包括 token range、summary、quant scale、valid count 和 generation。此类方案会改变候选集合，必须验收 exact top-k recall、next-token logits、长文检索/多轮质量、位置分桶和 fallback 比例，不能只比较 kernel 时间。

### 17.5 MoE、Decode 与专项顺序

若 Indexer 不主导，按 profile 处理：小 `Δ` 先合批和 Graph；MoE 先看 expert histogram、有效 batch 和 A2A；通信分别统计 TP/EP/CP/PP；Decode 联合评估候选跨步复用、Graph 和 MTP。MTP 以每 accepted token 的总 GPU time 和扫描次数判断，不能只看 acceptance rate。

## 18. PD Producer 流水

当前已有 chunk/page early send。**[目标设计]** 只有 `T_PD,non-overlap/T_TTFT` 显著时，才开发更细 producer；先 layer-group，后逐层，避免小传输和事件开销。

Segment 至少包含 request/chunk、layer range、pool type、source/destination page、ready event、generation 和 final-state 标志。必须保证写完成后 RDMA 才读、PP-local ratio 正确、backpressure 不拖垮 Prefill、Decode 在 final commit 前不消费不完整 Prefix。

失败时退回现有 chunk/page early send；验收比较的是 non-overlap tail 和 E2E TTFT，而非“开始发送更早”。

# 第五部分：风险、路线图与验收

## 19. 风险与降级策略

### 19.1 上线门禁

| 风险 | 等级 | 当前降级/门禁 |
|---|---|---|
| A/B 均无完整跨轮闭环 | 高：性能 | 不承诺增量；miss 时 full Prefill |
| HiSparse retraction CPU copy 未实现 | 阻塞：稳定性 | 保守 admission；故障 fail-closed；不得退出 scheduler |
| Profile B 后端/布局受限 | 阻塞：兼容 | 固定 P-off/D-on、Mooncake、radix-off、DCP=1 |
| HiSparse 与 radix/HiCache 冲突 | 高：容量/复用 | A/B 分开，目标采用分层 Prefix |
| Dense Indexer 长历史复杂度 | 高：TTFT/ITL | profile 后推进 exact 优化；ANN 保留 dense fallback |
| 多池 generation 部分完成 | 高：正确性 | Prepare/Commit、bitmap、checksum、rollback |
| PD/EP/H2D 资源竞争 | 高：尾延迟 | 分链路观测、保守 admission、必要时拓扑隔离 |
| 近似检索质量下降 | 高：质量 | recall/logits/任务质量门禁，低置信度 dense fallback |

任何 checksum mismatch、跨请求页污染、未处理异常导致 scheduler 退出、错误 generation Attach 均为零容忍。完整风险台账见附录 D。

### 19.2 通用降级顺序

目标功能失败时依次尝试：补传同 generation 缺页 → 放弃新 generation、保留旧 Prefix → 固定副本 full Prefill → 排队或换副本 → 受控拒绝。不得在不完整状态下继续 Decode。性能优化若 SLO-goodput 不增或质量下降则单项关闭；Profile B 的 H2D P99 不可接受时退回 A 或降低并发。

## 20. 分阶段交付路线图

### 阶段 0：事实基线与可观测性

Profile A 跑通 128K/512K/1M；校准子池、PD bytes 和 sizing mode；建立首轮、当前第 2 轮和 OOM trace；拆分 Indexer、attention、MoE、PD tail 和 Decode queue。退出条件：1M 正确结束，容量/传输差异可解释，可还原请求生命周期。

### 阶段 1：容量实验与故障安全

跑通 Profile B Mooncake 混合落点；建立 Host/GPU/H2D 指标；实现保守 admission；强制 retraction 并做到恢复或受控失败；限流 rebootstrap。退出条件：非法组合 fail-fast，目标并发无意外 retraction，故障不退出 scheduler。

### 阶段 2：同会话多轮增量 MVP

交付 Multi-pool Prefix/generation、immutable committed + mutable tail、增量 Prefill、PD existing-page skip、final commit、Decode 两阶段 Attach 和 full fallback。退出条件由第 21 章统一定义。

### 阶段 3：HiSparse 分层与恢复

将 C4 Host 页纳入 Prefix ownership，解耦 GPU working set；实现多池 snapshot/restore 和 Prefix-aware rebootstrap；升级实时 admission。要求 Host 分层与跨轮 Prefix 同时正确，P99 H2D 不抵消收益。

### 阶段 4：Profiler 驱动专项

Indexer 主导时先 metadata/score-top-k，再共享候选、History/K-CP、exact pruning，最后 ANN；小 `Δ` 固定开销主导时做合批/Graph/融合；C128、MoE、网络、Decode 或 PD tail 主导时分别推进对应专项。所有功能以第 21 章 SLO-goodput 和正确性决定保留。

## 21. 验证矩阵与验收

### 21.1 权威实验矩阵

| 场景 | 固定/变量 | 核心问题 |
|---|---|---|
| 首轮正确性 | A；128K/512K/1M；单并发 | 模型、pool、PD、输出正确性 |
| Prefill | chunk 三档×合法 CP；预算可比 | TTFT、scratch、collective |
| 当前多轮 | H=128K/512K/1M；Δ=128/1K/8K；2/4/8 轮 | A/B 重算、传输、H2D |
| PD/网络 | A 可用后端；B 仅 Mooncake；并发 1/2/4/8 | 带宽、queue、EP 竞争、tail |
| 容量故障 | 并发、memory fraction、输出上限升压 | admission 误差、retraction |
| Prefix MVP | hit/miss/partial-tail/evict/版本变化/迁移 | 增量一致性和 fallback |
| 传输故障 | 丢页/重试/取消/Commit 前后失败 | 幂等、rollback、隔离 |
| 性能专项 | 每次单开一项 | kernel/stage/E2E、SLO-goodput |
| 近似算法 | H 三档、位置/主题/轮次分桶 | recall、logits、质量、fallback |

### 21.2 最低验收条件

1. 1M 首轮完整结束，无 NaN、transfer failure、checksum mismatch 或 scheduler 退出。
2. 实测各池 payload 与理论差异可由 page、state、PP/TP、skip 或 retry 解释。
3. Prefix hit 时 full/incremental 的 logits、输出、有效长度和状态校验在允许误差内一致；主要 PD bytes 随 `Δ` 增长。
4. miss、历史编辑、版本变化、驱逐和 Decode 迁移自动 fallback，不复用错误页。
5. 并发 extend 和 partial-tail copy-on-write 无跨请求 KV 污染。
6. 强制 Decode retraction；任何未处理 `NotImplementedError`、scheduler 退出或悬挂均不可上线。
7. Profile B 无 Host pool 耗尽、working buffer 异常或不可接受 H2D P99。
8. 性能项仅在正确性不变且目标 SLO-goodput 改善时保留；近似检索还必须通过质量门禁。

### 21.3 尚未动态验证

目标 NVIDIA 机器的 1M 可达性/质量、Mooncake 混合传输带宽、最大并发、Host pinned memory 和 H2D P99、Indexer/MoE/PD tail 排序均待实验。Multi-pool Prefix、增量 Prefill、PD Manifest、两阶段 Attach 和 snapshot/restore 是目标设计，不是当前能力。本次文档重构不运行 GPU、Mooncake 或 1M 测试。

# 附录

## 附录 A：完整部署命令

### A.1 Profile A Prefill

```bash
python -m sglang.launch_server \
  --model-path /work/models/DeepSeek-V4-Flash-FP8 \
  --disaggregation-mode prefill \
  --port "${PREFILL_PORT}" \
  --chunked-prefill-size "${CHUNK_TOKENS}" \
  --tp "${TP}" \
  --ep-size "${EP}" \
  --mem-fraction-static "${PREFILL_MEMORY_FRACTION}"
```

### A.2 Profile A Decode

```bash
python -m sglang.launch_server \
  --model-path /work/models/DeepSeek-V4-Flash-FP8 \
  --disaggregation-mode decode \
  --dist-init-addr "${PREFILL_HOST}:${PREFILL_PORT}" \
  --port "${DECODE_PORT}" \
  --tp "${TP}" \
  --ep-size "${EP}" \
  --mem-fraction-static "${DECODE_MEMORY_FRACTION}"
```

### A.3 Profile B Prefill

```bash
python -m sglang.launch_server \
  --model-path /work/models/DeepSeek-V4-Flash-FP8 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --port "${PREFILL_PORT}" \
  --chunked-prefill-size "${CHUNK_TOKENS}" \
  --tp "${TP}" \
  --ep-size "${EP}" \
  --mem-fraction-static "${PREFILL_MEMORY_FRACTION}"
```

### A.4 Profile B Decode

`HISPARSE_CONFIG` 示例：`{"top_k":2048,"device_buffer_size":4096,"host_to_device_ratio":2}`。

```bash
python -m sglang.launch_server \
  --model-path /work/models/DeepSeek-V4-Flash-FP8 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --dist-init-addr "${PREFILL_HOST}:${PREFILL_PORT}" \
  --enable-hisparse \
  --disable-radix-cache \
  --hisparse-config "${HISPARSE_CONFIG}" \
  --port "${DECODE_PORT}" \
  --tp "${TP}" \
  --ep-size "${EP}" \
  --mem-fraction-static "${CONSERVATIVE_DECODE_MEMORY_FRACTION}" \
  --dcp-size 1
```

## 附录 B：详细公式与推导

### B.1 行宽

普通 packed V4 KV 每个压缩 token、每层：

$$
B_{KV}=448\text{ B(NoPE FP8)}+64\times2\text{ B(RoPE BF16)}+8\text{ B(scale/pad)}=584\text{ B}
$$

NVIDIA 默认 C4 index key：

$$
B_{idx}=128\text{ B(FP8 vector)}+\frac{128}{128}\times4\text{ B(scale)}=132\text{ B}
$$

FP4 index 可为 68B，但只适用于实际启用该布局的路径，不能作为本场景默认值；见 [pool_configurator.py](../../../python/sglang/srt/model_executor/pool_configurator.py)。

### B.2 PD payload

$$
B_{wire/full}=21\times\frac{584}{4}+21\times\frac{132}{4}+20\times\frac{584}{128}=3850.25\text{ B}
$$

$$
10^6\times3850.25=3.85025\times10^9\text{ B}\approx3.85\text{ GB}\approx3.59\text{ GiB}
$$

最终 SWA window/page、C4 state、C128 request state 再增加约十几 MiB，应以 `sglang:kv_transfer_total_mb` 和 backend trace 校准。

### B.3 Pool sizing

在第 6 章限定假设下：

$$
B_{SWA-slot}=43\times584+\frac{8}{128}\times(8192+2048)\times21=38552\text{ B}
$$

$$
B_{pool/full}=0.1\times38552+3066+693+91.25=7705.45\text{ B}
$$

对应 [_get_bytes_per_swa_token() 与 _get_bytes_per_full_token()](../../../python/sglang/srt/model_executor/pool_configurator.py#L1244-L1291)。HiSparse shrink、SWA cap、PP、MTP、C128 固定 state、padding、headroom 和并发都会改变实际值。

### B.4 多轮与命中

$$
B_{first}\approx N_1\times3850.25+B_{state}+B_{padding}
$$

$$
B_{current,k}\approx N_k\times3850.25+B_{state,k}+B_{padding,k}
$$

$$
B_{incremental,k}\approx\Delta_k\times3850.25+B_{changed-state}+B_{partial-page}
$$

$$
\Delta B_k\approx P_{hit}\times H_k\times3850.25\text{ B}
$$

高命中增量时间可拆为：

$$
T_{inc}=T_{meta}+T_{new}(\Delta)+T_{C4}(\Delta,H/4)+T_{C128}(\Delta,H/128)+T_{comm}+T_{PD,tail}
$$

Graph 主要降低固定项，不降低 C4 score；History/K-CP 降低单 rank score，但增加 top-k merge。

## 附录 C：兼容矩阵

| 组合 | 当前结果 | 证据/说明 |
|---|---|---|
| DSA，不开 HiSparse | 支持 | ratio-4/128 模型路径独立工作 |
| HiSparse，未关闭 radix | 启动失败 | [validate_hisparse()](../../../python/sglang/srt/arg_groups/hisparse_hook.py#L79-L101) 只校验，不自动改参数 |
| HiSparse + radix off | 无跨请求 chunk cache | [registry.py](../../../python/sglang/srt/mem_cache/registry.py)、[chunk_cache.py](../../../python/sglang/srt/mem_cache/chunk_cache.py) |
| HiSparse + HiCache | 配置互斥 | [kv_cache_hook.py](../../../python/sglang/srt/arg_groups/kv_cache_hook.py#L393-L397) |
| DSV4 compressed KV + PD Decode radix | 尚未支持 | [kv_cache_builder.py](../../../python/sglang/srt/mem_cache/kv_cache_builder.py) |
| P off / D on + Mooncake | Profile B 正确拓扑 | 特殊目的索引见 [decode.py](../../../python/sglang/srt/disaggregation/decode.py#L1613-L1634) |
| P on / D off 或 P on / D on | 不应使用 | Prefill C4 源端物理映射未进入协议 |
| DSV4 HiSparse + 非 Mooncake | Decode 预分配失败 | [decode.py](../../../python/sglang/srt/disaggregation/decode.py#L1613-L1634) |
| Mooncake + HiSparse + Decode DCP relayout | 不支持 | [mooncake/conn.py](../../../python/sglang/srt/disaggregation/mooncake/conn.py) |
| DSV4 + Decode async offload manager | 不支持 | [decode_kvcache_offload_manager.py](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py) 仅接受 MHA/MLA pool |

`SGLANG_DSV4_UNIFIED_KV_FP8` 受 AMD HIP、`unified_kv_triton`、gfx95 等门控，NVIDIA 主线不适用；生效平台与 PD 当前会启动期 fail-fast，不进入近期路线图。Unified FP8 约 640B 相对 unified BF16 1024B 降 37.5%，不能与 packed V4 的 584B 比成“带宽减半”。

## 附录 D：完整风险台账

| ID | 严重度 | 风险/触发 | 当前规避 | 目标能力 |
|---|---|---|---|---|
| R1 | 高：多轮性能 | A Decode radix 不完整；B 无跨请求命中 | 不承诺增量 | 多池增量闭环 |
| R2 | 阻塞：稳定性 | HiSparse retraction CPU copy 未实现 | 保守 admission、故障注入 | fail-closed、snapshot/rebootstrap |
| R3 | 阻塞：后端 | B 仅 Mooncake，DCP relayout 不支持 | 固定 DCP=1 | 扩展 wire/layout 后开放 |
| R4 | 高：容量/复用 | HiSparse 与 radix/HiCache 冲突 | A/B 分开 | 分层 Multi-pool Prefix |
| R5 | 高：TTFT/ITL | Dense Indexer `O(Q×C)` | profile、chunk/CP 网格 | exact 优化，必要时 ANN |
| R6 | 中：TTFT | final state/尾部不完全重叠 | 测 non-overlap | layer-group producer |
| R7 | 高：网络 | EP、PD、H2D 竞争 | 分阶段测量、隔离 | 网络感知 admission |
| R8 | 高：一致性 | 页/state/generation 部分完成 | 目标功能默认关闭 | Manifest + 两阶段 Attach |
| R9 | 高：质量 | ANN/候选复用漏召回 | 不默认启用 | 质量门禁+dense fallback |
| R10 | 高：恢复风暴 | 多个 1M 请求同时 rebootstrap | 限流、排队 | Prefix-aware restore |
| R11 | 中：元数据 | Prefix 树锁和 page metadata 抵消收益 | 会话 MVP | 分片目录与租约优化 |
| R12 | 高：静默错误 | Host C4 与 Device index 跨 generation | 整体 miss | checksum/epoch/原子 commit |

## 附录 E：研发清单

### E.1 直接使用前

- [ ] 确认 D=4096、43 主干层、NextN 独立和 ratio 口径；
- [ ] Profile A 1M 单请求完成；
- [ ] 实测各池容量与 PD bytes；
- [ ] 明确 PD 与 EP 网络拓扑；
- [ ] Profile B 验证 Mooncake、Host pool、H2D 和 retraction；
- [ ] 所有加速项有单变量对照。

### E.2 增量 MVP 开发前

- [ ] 明确完整 prompt 或增量 API；
- [ ] 明确 session affinity 与 Decode 迁移；
- [ ] 证明 SWA/C4/C128 安全续算边界；
- [ ] 定义 model/layout/token fingerprint；
- [ ] 定义 immutable Prefix、mutable tail、lease 和 epoch；
- [ ] 定义 full Prefill fallback 与容量预算；
- [ ] 设计 full/incremental 一致性和故障测试。

### E.3 性能专项前

- [ ] Indexer、MoE、PD tail、H2D 已分别计时；
- [ ] 优化对象主导目标 SLO；
- [ ] 定义正确性、质量和 fallback 门禁；
- [ ] 以 E2E SLO-goodput 而非 kernel 加速比决策。

## 附录 F：关键源码索引

- 模型 wiring：[deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py)
- DSV4 多池与 wire buffer：[deepseek_v4_memory_pool.py](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)
- Pool sizing：[pool_configurator.py](../../../python/sglang/srt/model_executor/pool_configurator.py)
- Prefill PD 与 early send：[prefill.py](../../../python/sglang/srt/disaggregation/prefill.py)
- Decode prealloc/transfer/HiSparse：[decode.py](../../../python/sglang/srt/disaggregation/decode.py)
- State transfer：[utils.py](../../../python/sglang/srt/disaggregation/utils.py)
- PP ratio slicing：[common/conn.py](../../../python/sglang/srt/disaggregation/common/conn.py)
- Mooncake 混合目标：[mooncake/conn.py](../../../python/sglang/srt/disaggregation/mooncake/conn.py)
- Chunk continuation：[chunk_cache.py](../../../python/sglang/srt/mem_cache/chunk_cache.py)
- Cache 选择：[registry.py](../../../python/sglang/srt/mem_cache/registry.py)
- Cache/retraction 构建：[kv_cache_builder.py](../../../python/sglang/srt/mem_cache/kv_cache_builder.py)
- Retraction 公共逻辑：[common.py](../../../python/sglang/srt/mem_cache/common.py)
- Allocator CPU copy：[allocator/base.py](../../../python/sglang/srt/mem_cache/allocator/base.py)
- Decode async offload：[decode_kvcache_offload_manager.py](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py)
- Dense Prefill Indexer：[dense_prefill_indexer.py](../../../python/sglang/srt/layers/attention/dsv4/dense_prefill_indexer.py)
- Candidate Indexer：[candidate_indexer.py](../../../python/sglang/srt/layers/attention/dsv4/candidate_indexer.py)

最终决策原则：先使首轮 1M 可测、容量可解释、故障可控，再完成同会话多池增量闭环；HiSparse 分层、Indexer 算法和更细 PD 流水均建立在此基础上。
