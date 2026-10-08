# DeepSeek-V4 Flash：PD 分离下 1M 多轮使用、功能开发与架构探索

> 目标对象：`DeepSeek-V4-Flash-FP8`，`model_type=deepseek_v4`，NVIDIA GPU，Prefill/Decode（PD）分离，约 1M token 的长历史多轮会话。
>
> 本文分为三层：**当前可直接使用的部署方案**、**近期需要补齐的工程能力**、**中远期多轮增量架构探索**。结论来自当前仓库静态读码、本地模型配置与公式推导；尚未完成真实 NVIDIA、Mooncake 或 1M 端到端压测。

## 0. 执行摘要

### 0.1 目标不是“把所有优化开关打开”

目标工作负载同时要求：首轮 1M Prefill 可完成、后续轮尽量只处理新增数据、PD 网络不重复传输稳定历史、Decode 容量可控、内存压力下可恢复或受控失败。当前代码尚不能同时满足这些目标。

推荐按以下顺序推进：

1. **先用 Profile A 建立基线。** 不启用 HiSparse，保留模型原生 ratio-4/128 DSA，校准首轮 1M 的正确性、容量、Indexer、MoE 与约 3.85 GB 的主要 PD payload。
2. **容量不足时再评估 Profile B。** 当前可用拓扑是 Prefill 不开 HiSparse、Decode 开 HiSparse，并使用 Mooncake 将 C4 直接落到 Decode Host；Decode 必须关闭 radix，当前没有跨请求前缀复用，并存在 retraction 失败风险。
3. **近期先补可观测性、admission 与故障安全。** 在无法解释每个子池容量、传输和回退原因前，不应直接开发复杂优化。
4. **多轮核心功能是一个整体闭环。** 多池 Prefix、增量 Prefill、PD 增量传输和 Decode attach 必须一起设计；只优化其中一段不能让第 2+ 轮成本随新增 token 增长。
5. **性能专项由 profiler 排序。** Dense Indexer、MoE A2A、PD 非重叠尾部和 HiSparse H2D 都可能主导，不能预设 Indexer 或网络一定是第一瓶颈。

### 0.2 当前最重要的代码边界

- **[代码事实] DSA 不等于 HiSparse。** ratio-4/128 压缩注意力是模型结构；`--enable-hisparse` 只是可选的 C4 GPU/Host 分层存储。
- **[代码事实] HiSparse 不会自动关闭 radix。** 必须显式配置 `--disable-radix-cache`，否则启动校验失败。
- **[代码事实] HiSparse 当前没有跨请求命中。** `SWAChunkCache` 只保存同一未完成请求的 chunk 续算状态。
- **[代码事实] HiCache 不能直接给 HiSparse 兜底。** HiSparse 要求关闭 radix，而 HiCache 与关闭 radix 配置级互斥。
- **[代码事实] DSV4 HiSparse PD 当前要求非对称部署。** Prefill 保持普通连续 C4 源池，Decode 单侧开启 HiSparse，并由 Mooncake 把 C4 直接写入 Host；非 Mooncake 在 Decode 预分配阶段失败，Decode DCP relayout 仍不支持。
- **[代码事实] Prefill 不能同时开启 HiSparse。** Prefill HiSparse 会把 C4 改为逻辑到稀疏物理页映射，但当前发送协议只使用普通逻辑源页 ID，未携带源端物理映射；P-on/D-off 与 P-on/D-on 均不应使用。
- **[代码事实] HiSparse Decode OOM retraction 存在阻塞风险。** 普通 `cpu_tensor` backup 最终调用未实现的 CPU copy 接口。
- **[理论推导] 容量与传输是两个数字。** 特定 pool-sizing 假设下约 `7705.45 B/full-token`；主要 PD 线性 payload 为 `3850.25 B/full-token`。
- **[代码事实] 已有 chunk/page 级 early send。** 后续可探索的是更细的 layer-group/layer producer，而不是从零开始做计算传输重叠。

### 0.3 证据与成熟度标签

- **[代码事实]**：当前源码或模型配置可以直接证明。
- **[理论推导]**：由 shape、dtype、ratio、容量或复杂度公式推得。
- **[待实验]**：代码允许或理论可行，但收益、稳定性和最优参数尚待目标集群验证。
- **[目标设计]**：本文提出但当前代码尚未完整实现的能力。

## 1. 场景目标、负载模型与适用边界

### 1.1 模型配置基线

| 项目 | 目标值 | 对 1M PD 场景的影响 |
|---|---:|---|
| `hidden_size` | 4096 | mHC 单流宽度 |
| `hc_mult` | 4 | 残差流为 `[T,4,4096]`，PP 边界可展平为 `[T,16384]` |
| 主干层数 | 43 | 主模型 KV、计算和容量统计范围 |
| NextN 层数 | 1 | 配置中额外第 44 个 ratio；启用 MTP 时单独计费 |
| 主干 ratio 分布 | `0×2, 4×21, 128×20` | 决定 SWA、C4、C128 子池规模 |
| SWA window | 128 | 决定近窗 KV 和增量续算边界状态 |
| C4 index top-k | 512 | 决定稀疏注意力候选 token 数 |
| 最大位置 | 1,048,576 | 配置上限，不等于质量、稳定性和显存已通过 1M 验收 |
| 权重量化 | FP8 block `[128,128]` | 仅描述线性权重，不等于 KV cache 布局 |

配置读取与逐层 ratio wiring 见 [deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py)。`compress_ratios` 共 44 项：前 43 项属于主干，最后一项对应 `num_nextn_predict_layers=1`；本文基础公式不把 NextN 当作第 44 个普通主干层。

本文不覆盖 D=5120、带视觉/Engram/ratio-1/2 的 V4.1，DSpark+ratio-2 专用路径，以及 AMD unified KV 性能方案。

### 1.2 将“1M 多轮”参数化

第 `k` 轮定义：

- `N_k`：本轮输入总长度；
- `Δ_k`：本轮新增输入 token；
- `H_k=N_k-Δ_k`：理论上稳定、可复用的历史；
- `O_k`：本轮最大或实际输出 token；
- `C_session`：并发活跃会话数；
- `R_p/R_d`：Prefill/Decode 副本数；
- `BW_pd`：PD 链路有效带宽；
- `P_stable`：应用层稳定前缀比例；
- `P_hit`：经过模型、布局、缓存、路由和 residency 校验后的实际命中比例。

代表性负载至少应覆盖：

| 场景 | 历史 `H_k` | 新增 `Δ_k` | 主要目标 |
|---|---:|---:|---|
| 首轮长文导入 | 0 | 1M | 验证完整 Prefill、容量与 PD 传输 |
| 长历史小增量 | 约 1M | 128～1K | 验证跨轮复用的最大价值 |
| 长历史中增量 | 约 1M | 8K | 验证增量 Indexer、PD 与尾部状态更新 |
| 持续增长会话 | 128K→1M | 1K～8K/轮 | 验证生命周期、驱逐与容量准入 |

“多轮”还必须明确 API 语义：每轮是否重新提交完整 token 序列，是否提供稳定 `session_id`，能否固定路由到同一 Decode 副本，历史是否会编辑，以及模型版本或 prompt template 是否可能变化。

### 1.3 场景级成功标准

1. 首轮 128K/512K/1M 均可完整结束，无 NaN、transfer failure 或多池校验错误。
2. 对稳定历史命中的后续轮，Prefill 工作、PD bytes 和 TTFT 的主要增量应随 `Δ_k` 而非 `N_k` 增长。
3. Prefix miss、驱逐、布局不兼容或 Decode 迁移时可正确回退完整 Prefill。
4. 容量不足应在 admission 阶段排队或拒绝；进入 retraction 后必须恢复或受控失败，不能退出 scheduler。
5. P50/P95/P99 TTFT、ITL、完成率和 SLO-goodput 均可由分阶段指标解释。

## 2. 首轮、当前后续轮与目标后续轮生命周期

### 2.1 首轮完整链路

| 阶段 | 节点 | 主要数据/动作 | 当前状态 | 必要观测 |
|---|---|---|---|---|
| 1. 请求接入 | Router/Prefill | token、最大输出、会话元数据 | 已有 | 总长度、排队时间 |
| 2. 分块 Prefill | Prefill | 按 chunk 执行 43 层计算 | 已有 | 每 chunk 时延、HBM 峰值 |
| 3. 多池产出 | Prefill | SWA、C4 KV、C4 index、C128 KV、边界 state | 已有 | 各池 rows/pages/bytes |
| 4. Early send | Prefill→Decode | page-aligned segment 先发送，最终 segment 携带 state | 已有 | compute/send 时间线、非重叠尾部 |
| 5. Decode 预分配 | Decode | 为各池和 state 分配目标位置 | 已有；HiSparse 有后端限制 | alloc 时延、失败原因 |
| 6. 多池接收 | Decode | C4 KV/index、C128 KV 及最终 state | 已有 | 实传/重传 bytes、有效 GB/s |
| 7. Attach | Decode | 等待所需页和最终 state 后进入 waiting/running | 已有完整请求路径 | transfer queue、attach 等待 |
| 8. Decode | Decode | 生成 token，增长 SWA/压缩池状态 | 已有 | ITL、token usage、MTP acceptance |
| 9. 完成/退避 | Decode | 释放、缓存或 retraction | DSV4 HiSparse 恢复不完整 | 驱逐、backup、restore 结果 |

普通 V4 主线性传输由 [DeepSeekV4TokenToKVPool.get_contiguous_buf_infos()](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1177-L1245) 注册，顺序为 21 个 C4 KV、21 个 C4 index、20 个 C128 KV。SWA、ring、compressor/indexer state 和 request-scoped C128 state 通过 [setup_state_kv_args()](../../../python/sglang/srt/disaggregation/utils.py#L1313-L1431) 传输；PP 根据当前 stage ratio 层切片，见 [_mla_slice_ptrs_for_pp()](../../../python/sglang/srt/disaggregation/common/conn.py#L1288-L1354)。

**[代码事实]** 当前并非完全串行：chunked prefill 完成 page-aligned segment 后即可发送，最终 segment 再携带 state。因此本文后续所说的 producer pipeline，是在现有 chunk/page early send 之上的更细粒度探索。

### 2.2 当前第 2+ 轮为什么仍可能处理完整历史

**Profile A（不开 HiSparse）：** 模型原生 DSA 和普通多池仍工作，但 DSV4 compressed KV 的 PD Decode radix 尚未接入，不能把“允许 radix”理解为“当前已经能在 Decode 复用完整 DSV4 多池前缀”。

**Profile B（Decode 单侧开启 HiSparse）：** Prefill 使用普通连续 C4 源池；Decode 开启 HiSparse 并关闭 radix，因此 Decode 注册的是无跨请求匹配的 `SWAChunkCache`。在 [chunk_cache.py](../../../python/sglang/srt/mem_cache/chunk_cache.py) 中，`match_prefix()` 返回空、`insert()` 不保存可匹配节点；`cache_unfinished_req()` 只让同一未完成请求跨 prefill chunk 续算。即使 Prefill 侧存在本地历史命中，Decode 也没有可按 generation attach 的完整 DSV4 多池 Prefix，因此当前仍不能形成跨轮 Prefill skip、PD skip、Decode attach 的端到端闭环。

因此，当前后续轮可能重复经历：

```text
完整 N_k token 进入 Prefill
  -> 历史重新经过主干/Indexer
  -> 为历史重新生成或组织多池数据
  -> 约 N_k × 3850.25 B 的主线性数据再次进入 PD
  -> Decode 重新预分配和 attach
```

### 2.3 目标增量链路

**[目标设计]** 稳定会话的理想第 `k` 轮应是：

```text
(session_id, tokens, model/layout version)
  -> Prefix lookup 与 token fingerprint 校验
  -> 找到已提交的 H_k 稳定边界
  -> 只对 Δ_k 做增量 Prefill
  -> 生成新增多池页 + 更新后的 SWA/压缩/request state
  -> Manifest 标出 Decode 已有页、缺失页和 state 更新
  -> 现有页 skip，新增/缺失页按 chunk/page 发送
  -> Decode prepare 验证容量、版本和页完整性
  -> final state 到达后原子 commit attach
  -> 失败则回滚新增尾部并回退 full Prefill
```

该路径必须同时解决 Prefill 计算复用、PD 传输 skip 和 Decode 存储 attach。仅在 Prefill 命中但 Decode 仍需全量接收，或 Decode 有页但 Prefill 无法从压缩边界续算，都不能形成完整收益。

## 3. 当前能力地图与直接使用方案

### 3.1 能力地图

| 能力 | Profile A：普通 DSA+PD | Profile B：P 普通池 + D HiSparse | 对 1M 多轮的含义 |
|---|---|---|---|
| ratio-4/128 DSA | 支持 | P/D 均支持 | 模型固有能力，不由 HiSparse 开关决定 |
| Chunked Prefill | 支持 | 支持 | 降低单次峰值、细化调度；不改变 Indexer 渐近工作量 |
| Chunk/page early send | 支持 | 支持 | 已有计算传输重叠，但仍可能有最终非重叠尾部 |
| 普通 DSV4 多池 PD | 支持 | P 为连续源池，D 为 Host/Device 混合目标 | C4 KV/index、C128 KV 和 state 均需一致处理 |
| 跨请求 Prefix | DSV4 PD Decode radix 未完整支持 | 不支持 | 两个现有 Profile 都不能直接完成理想跨轮增量 |
| C4 Host 驻留 | 无 | 仅 Decode 支持 | 缩小 Decode C4 GPU working set，不缩小 Prefill 逻辑历史 |
| HiCache | 可独立研究，非现成 DSV4 PD 解法 | D 侧 radix-off，与 HiCache 互斥 | 不能直接给 HiSparse 补命中 |
| Decode retraction | 需目标配置验证 | CPU copy 未实现 | Profile B 上线阻塞风险 |
| Async decode offload | DSV4 不支持该 manager | DSV4 不支持该 manager | 不能作为 retraction backup 的替代 |
| CP/TBO/Graph/MTP | 条件化实验 | 条件化实验 | 一次只开一项，以 SLO-goodput 决策 |

### 3.2 Profile A：DSA + PD 基线，不开 HiSparse

**定位：** 正确性、容量、网络和性能的首选基线，也是当前更稳妥的生产候选起点。不开 HiSparse 不会关闭 ratio-4/128 DSA。

Prefill：

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

Decode：

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

该 Profile 不默认加入 CP、TBO、MTP、Decode radix 或 DSV4 不支持的 async offload。先测出 Indexer、attention、MoE、PD 和 Decode 各阶段占比，再做单变量实验。

### 3.3 Profile B：Prefill 普通池 + Decode HiSparse 容量实验

**定位：** 当普通 Decode 多池无法满足目标并发容量时，评估 C4 GPU working set 收缩。它不是跨轮复用方案，且在 retraction 验证通过前不作为默认生产推荐。

**当前正确拓扑：** Prefill 不开启 HiSparse，保持普通连续 C4 源池；Decode 单侧开启 HiSparse，由 Mooncake 将 C4 KV 写入 Host、将 C4 index/C128 写入 Device。当前协议只提供 Decode 目的端特殊页索引，没有传递 Prefill 端 HiSparse 的逻辑到物理映射，因此不能在 Prefill 增加 `--enable-hisparse`。源池构造与映射见 [deepseek_v4_memory_pool.py](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py) 和 [allocator/hisparse.py](../../../python/sglang/srt/mem_cache/allocator/hisparse.py)；Prefill 源指针注册见 [prefill.py](../../../python/sglang/srt/disaggregation/prefill.py)，Mooncake 的特殊索引用于目的端选择，见 [mooncake/conn.py](../../../python/sglang/srt/disaggregation/mooncake/conn.py)。

`HISPARSE_CONFIG` 是 Decode 侧 JSON 字符串，例如 `{"top_k":2048,"device_buffer_size":4096,"host_to_device_ratio":2}`。

Prefill：

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

Decode：

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

不要在 Prefill 增加 `--enable-hisparse`；也不要组合 `--enable-hierarchical-cache`、`--disaggregation-decode-enable-radix-cache`、`--disaggregation-decode-enable-offload-kvcache`、Decode DCP relayout 或非 Mooncake backend。必须单独压测 Host pinned memory、C4 H2D、Mooncake 混合目标页和 Decode retraction。

当前组合边界如下：

| Prefill | Decode | 当前结论 |
|---|---|---|
| HiSparse off | HiSparse off | Profile A，普通连续多池 PD |
| HiSparse off | HiSparse on | Profile B，当前预期的 Mooncake Host/Device 混合落点 |
| HiSparse on | HiSparse off | 不应使用；源端 C4 稀疏物理布局没有传输映射 |
| HiSparse on | HiSparse on | 不应使用；Decode 特殊目的索引不能修复 Prefill 源端映射缺失 |

### 3.4 选择决策树

```text
普通 DSV4 多池能否容纳目标 1M × 并发？
  ├─ 能：优先 Profile A，建立性能和可靠性基线
  └─ 不能：是否具备 Mooncake、大 Host pool，并接受实验性恢复？
       ├─ 是：Profile B 仅做容量实验和故障注入
       └─ 否：先降低并发/输出上限，或开发 Prefix/admission/恢复能力

业务是否要求第 2+ 轮只处理新增 token？
  ├─ 是：现有 A/B 都不完整，需要第 6～10 章的目标能力
  └─ 否：按首轮 TTFT、容量和 SLO-goodput 选择 A/B
```

## 4. 容量、网络与多轮成本模型

### 4.1 三类数字不能混用

1. **逻辑/预留池容量**：用于 HBM 与 Host 规划，受 SWA ratio/cap、HiSparse shrink、PP、MTP、page 和并发影响。
2. **PD wire payload**：Prefill 到 Decode 实际发送的数据；可能因 prefix skip、padding、state 和重传变化。
3. **运行 working set**：某时刻真实 GPU/Host 驻留、临时 activation、Indexer logits 与 H2D buffer。

### 4.2 基础行宽与主线性 payload

普通 packed V4 KV 每个压缩 token、每层：

$$
B_{KV}=448\text{ B(NoPE FP8)}+64\times2\text{ B(RoPE BF16)}+8\text{ B(scale/pad)}=584\text{ B}
$$

NVIDIA 默认 C4 index key：

$$
B_{idx}=128\text{ B(FP8 vector)}+\frac{128}{128}\times4\text{ B(scale)}=132\text{ B}
$$

FP4 indexer 可为 68B，但只适用于实际启用该布局的路径，不能泛化为本场景默认值。布局选择见 [pool_configurator.py](../../../python/sglang/srt/model_executor/pool_configurator.py)。

| 数据 | 公式 | B/full-token |
|---|---:|---:|
| C4 KV | `21 × 584 / 4` | 3066.00 |
| C4 index | `21 × 132 / 4` | 693.00 |
| C128 KV | `20 × 584 / 128` | 91.25 |
| **合计** | 以上求和 | **3850.25** |

因此 1M token 的主要线性 payload 为：

$$
10^6\times3850.25=3.85025\times10^9\text{ B}
\approx3.85\text{ GB}\approx3.59\text{ GiB}
$$

最终 SWA window/page、C4 state 和 C128 request state 再增加约十几 MiB，单请求约为 **3.87 GB 量级**。[理论推导] 应以 `sglang:kv_transfer_total_mb` 和 backend trace 校准。

### 4.3 Pool sizing 为什么约为 7705.45 B/full-token

在“43 层同一统计范围、584B KV、132B index、`swa_full_tokens_ratio=0.1`、C4 ring=8、SWA page=128、state FP32、不开 HiSparse、无 PP/MTP”的特定假设下：

$$
B_{SWA-slot}=43\times584+\frac{8}{128}\times(8192+2048)\times21=38552\text{ B}
$$

$$
B_{pool/full}=0.1\times38552+3066+693+91.25=7705.45\text{ B}
$$

该公式对应 [_get_bytes_per_swa_token() 与 _get_bytes_per_full_token()](../../../python/sglang/srt/model_executor/pool_configurator.py#L1244-L1291)。它包含按 full-token 预留的 SWA/state，不能当作每次 PD 都线性发送的数据。

HiSparse shrink、SWA cap、PP stage slicing、MTP draft、C128 固定 request state、page padding、allocator headroom 和并发都会改变实际容量；必须读取启动日志中的 sizing mode。

### 4.4 首轮、当前后续轮与理想增量轮

首轮：

$$
B_{first}\approx N_1\times3850.25+B_{state}+B_{padding}
$$

当前无跨轮命中时：

$$
B_{current,k}\approx N_k\times3850.25+B_{state,k}+B_{padding,k}
$$

理想增量轮：

$$
B_{incremental,k}\approx \Delta_k\times3850.25+B_{changed-state}+B_{partial-page}
$$

考虑实际命中比例后，传输节省上限为：

$$
\Delta B_k\approx P_{hit}\times H_k\times3850.25\text{ B}
$$

即使 KV 只增加 `Δ_k`，新增 query 仍可能对历史压缩上下文打分，Indexer 工作近似包含：

$$
O(\Delta_kH_k/4+\Delta_k^2/4)
$$

因此“PD bytes 已增量化”不等于“Prefill 计算已完全随 `Δ_k` 线性缩小”。

### 4.5 高 Cache 命中后的计算分解

假设第 `k` 轮有稳定历史 `H=H_k`、新增 token `Δ=Δ_k`，并且 Multi-pool Prefix 已完整命中。命中不是“把历史 KV 重新喂给模型”，而是历史 token 不再进入本轮 EXTEND tensor；`seq_lens` 仍描述完整序列，实际前向只包含未匹配的 `Δ` 个 token。该调度语义见 [schedule_batch.py](../../../python/sglang/srt/managers/schedule_batch.py) 与 [forward_batch_info.py](../../../python/sglang/srt/model_executor/forward_batch_info.py)。

**[理论推导] 被 Prefix 命中消除的工作**包括历史 `H` 个 token 的 embedding、43 层 mHC、Q/KV/Indexer 投影、compressor、attention、MoE 路由与专家、norm 和输出投影。主干计算由：

$$
O((H+\Delta)\times\text{full Transformer})
$$

缩小为：

$$
O(\Delta\times\text{full Transformer})+T_{history-query}(H,\Delta)
$$

但 Cache 命中没有消除新增 query 对历史 Cache 的检索和 attention。对本模型，剩余历史相关项主要是：

| 路径 | 历史宽度 | 命中后主要复杂度 | `H=1M, Δ=128` 的量级 |
|---|---:|---:|---:|
| SWA | 128 | `O(43×Δ×128)` | 与 1M 无关 |
| C4 Indexer | `H/4` | `O(21×Δ×H/4)` | 每个 C4 层约 3200 万 query-context 对 |
| C4 稀疏 attention | top-k 512 | `O(21×Δ×512)` | 有界，通常小于 dense score |
| C128 attention | `H/128` | `O(20×Δ×H/128)` | 每层约 100 万 query-context 对 |
| 新 token 主干 | `Δ` | `O(Δ×mHC/投影/MoE)` | 不再随 `H` 增长 |

因此高命中率下，优化目标不是继续减少已经跳过的历史 Transformer，而是把：

$$
O(\Delta\times\text{Transformer})+O(\Delta H/4)+O(\Delta H/128)
$$

逐步逼近：

$$
O(\Delta\times\text{Transformer})+O(\Delta\times C_{bounded})
$$

其中 `C_bounded` 是经过精确或可控近似候选选择后的有限历史集合。

### 4.6 高命中场景的优化优先级

#### P0：先保证命中是真的、边界可续算

- Prefix 必须同时命中 SWA tail/ring、C4 KV/index、C4 compressor/indexer state、C128 KV/request state；只命中部分 KV 不能跳过历史前向。
- C4 overlap 边界约需保留 `H % 4 + 4` 行续算状态；`H % 128 != 0` 时还需保存 C128 pending request state。
- 优先固定 session 路由并观测 `matched_tokens`、`extend_num_tokens`、各池 generation/checksum，排除“业务认为命中、执行层实际 full Prefill”。

#### P1：不改算法语义的低风险优化

1. **复用历史元数据。** 缓存长历史的 compressed slot mapping、page table、有效长度与 gather 结果，避免每轮重建 `arange(H/4)`、查询 `req_to_token` 和拼接大映射。
2. **融合 Indexer score 与 top-k。** 流式维护 top-k，避免完整写回 `[Δ,H/4]` logits；进一步融合 token top-k、candidate block 选择与 mask 生成，减少 HBM 流量和临时张量。
3. **Graph 与固定 bucket。** `Δ` 很小时 kernel launch 与 Python 调度占比上升，优先评估 Prefill CUDA Graph、稳定 token bucket 和固定 layout；Graph 不降低 FLOPs，但能降低固定开销。
4. **跨请求合批。** 将多个会话的小 `Δ` 聚合，提高 mHC、投影、MoE grouped GEMM 和 EP dispatch/combine 的有效 batch；单个 128-token 增量不宜被 TBO 再切得过小。
5. **Page/Host 预取。** Decode HiSparse 根据连续 decode step 的 top-k 稳定性保留 hot pages，并在独立 stream 预取；这优化 H2D，不减少 dense Indexer score。

#### P2：保持精确结果的结构优化

- **跨层候选共享。** 让一个 source/anchor 层做全历史 dense scan，相关 C4 层共享 candidate blocks，再在块内精确 top-k；需要先证明共享不会漏掉各层真实候选。
- **历史/K 维并行。** 当 `Δ` 很小、`H/4` 极大时，普通按 query/token 切分的 CP 利用率有限；更合适的是把历史 K 分给多个 rank，各自算 local top-k，通信只合并 `world_size×top_k` 候选，而不是聚合全宽 logits。
- **精确块上界剪枝。** 为历史 block 维护可证明的 score upper bound；只有上界可能进入全局 top-k 的 block 才精算，从而保持 exact 结果。
- **C128 分块执行。** 对 `H/128` 历史采用 tiled/streaming attention、在线 softmax 和页表复用，控制 scratch 与读放大；其优先级通常低于 C4 Indexer。

#### P3：允许质量权衡的近似检索

可研究 block summary、IVF/PQ、HNSW、聚类、近历史 exact + 远历史 approximate、跨 decode step 候选复用与周期性 full refresh、按 query 难度自适应 top-k。此类方案会改变候选集合，必须同时验收 top-k recall、next-token logits、长文检索/多轮问答质量和 fallback 比例。

#### 并行策略的场景化选择

| 手段 | 高命中、小 `Δ`、大 `H` 时的判断 |
|---|---|
| Chunked Prefill | 控制内存和公平性，不降低 `O(ΔH/4)` 总 score FLOPs |
| 普通 Query-CP | `Δ` 太小时 rank 间工作不足，收益可能差 |
| History/K-CP | 理论上最匹配小 Q、大 K，但当前需开发 local-top-k merge 路径 |
| TP | 主要解决权重容量；过大 TP 可能被小消息 collective 延迟主导 |
| PP | 单请求小增量 bubble 明显，需要并发会话填流水 |
| EP | 依赖跨请求合批改善专家利用率，并关注 PD 与 A2A 网络竞争 |
| TBO | 只有 microbatch 足够大时尝试，小 `Δ` 再切分可能退化 |
| CUDA Graph | 不改复杂度，但适合降低小 `Δ` 的 launch/调度固定开销 |

**推荐实施顺序：** 先完成 P0；随后用 profiler 验证 metadata、C4 score、C128、MoE 和 launch 的占比。若 C4 Indexer 主导，先做 P1 的元数据复用与 score/top-k 融合，再做 P2 的共享候选和 History/K-CP；只有 exact 路径仍无法满足 SLO 时才进入 P3。

### 4.7 按计算阶段拆分可优化点

高 Cache 命中后，应按 `历史检索 → 新 token 主干 → MoE/并行通信 → Decode 串行步` 分开优化，不能只用总 TTFT 判断。下表中的“收益类型”用于避免把显存优化误写成计算优化。

| 计算阶段 | 当前剩余工作 | 可优化方向 | 主要收益类型 |
|---|---|---|---|
| 输入与调度 | 构造 suffix、位置、page/slot mapping | 复用 Prefix 元数据；固定 token bucket；批量准备多个会话 | CPU、launch、HBM metadata |
| mHC | `Δ` token 在每个子层执行 pre/post、mix 与 Sinkhorn | 复用已有 fused kernel；融合相邻 norm/mix/boundary；小 `Δ` 合批 | kernel launch、HBM、中小 GEMM 利用率 |
| Q/KV/Indexer 投影 | 仅新增 `Δ` token 投影，但算子较碎 | 融合 norm/动态 FP8 quant/linear；合并兼容投影；Graph capture | launch、量化读写、Tensor Core 利用率 |
| C4 compressor | 只更新新增 token 与 overlap tail | 从 Prefix state 续算；只处理边界 tail；避免全局 token-order 重建 | FLOPs、collective、metadata |
| C4 dense score | 每层扫描 `H/4` 历史 | streaming top-k、候选共享、K 维并行、块剪枝或 ANN | 核心 FLOPs、HBM、scratch |
| C4 sparse attention | 对选中 top-k=512 做 attention | candidate/page 排序以提高连续访问；融合 gather+attention；复用 hot pages | HBM 随机读、H2D、launch |
| C128 attention | 每层读取 `H/128` 历史 | tiled online-softmax；page table 复用；历史分片并行 | HBM、scratch、尾延迟 |
| MoE | `Δ` token 路由、dispatch、专家 GEMM、combine | 跨会话合批；grouped/persistent GEMM；dispatch/GEMM/combine 重叠 | GEMM 利用率、EP A2A |
| Norm/残差/激活 | 大量逐 token memory-bound 小算子 | 在不改变数值语义下做 RMSNorm、残差、量化和激活融合 | HBM、launch |
| LM head/采样 | 每轮最后位置或 Decode token 输出 | 只计算必要位置；融合采样前处理；避免无用 logits materialize | GEMM、HBM、launch |

**[待实验] 算子融合边界：** 本模型权重是 FP8 block `[128,128]`，但在 NVIDIA 路径上不能据此假设 norm 已与激活 FP8 量化融合。目标优化应分别测 `bf16 norm 输出 → 动态量化 → FP8 linear` 的读写和 kernel 数，再决定是否开发 fused norm/quant/linear；不能照搬 AMD 专用融合结论。

#### Prefill 增量阶段

当 `Δ=128～8K` 时，优先级随 `Δ` 变化：小 `Δ` 更容易被调度、launch、小 GEMM、PP bubble 和 EP 小包主导；较大 `Δ` 更容易被 C4 score、MoE GEMM 与 collective 主导。因此需要按 `(H,Δ,batch)` 建立二维/三维 profile，而不是只测试 1M 全量 Prefill。

可将增量 Prefill 时间近似拆为：

$$
T_{inc}=T_{meta}+T_{\mathrm{new\ token}}(\Delta)+T_{\mathrm{C4\ score}}(\Delta,H/4)+T_{C128}(\Delta,H/128)+T_{comm}+T_{\mathrm{PD\ tail}}
$$

优化后必须分别证明对应分量下降。例如 Graph 主要降低 `T_meta/T_new_token` 的固定开销，不会降低 `T_C4_score`；History/K-CP 主要降低单 rank 的 `T_C4_score`，但会增加 top-k merge 通信。

#### Decode 串行阶段

Decode 每步通常只有一个或少量 query，历史越长，C4 Indexer 的全历史扫描越可能成为 ITL 主项。除 Prefill 的 exact/approximate 候选优化外，还可利用相邻 decode step 的局部性：缓存上一 token 的 candidate blocks、维护 recent+hot 集合、异步预取下一步可能访问的 Host 页，并周期性执行 full refresh。候选复用必须带置信度或 refresh 条件，低置信度时回退 dense scan。

MTP/推测解码只有在“减少串行 Decode 轮次”的收益大于 draft/verify、额外 KV、候选检索和通信成本时才保留。验收应看 accepted tokens/step、每个 accepted token 的 Indexer 扫描次数和 ITL，而不能只看 acceptance rate。

### 4.8 时间与 goodput 下界

$$
T_{xfer,min}=\frac{B_{wire}}{BW_{pd}}
$$

3.85025GB payload 在 25/50/100GB/s 下的理想传输下界约为 154/77/39ms。实际还包含注册、同步、page、拥塞、重传和非重叠尾部。

$$
G_{req}\le\min\left(\frac{R_p}{T_p},\frac{BW_{net}}{B_{wire}},\frac{R_d}{T_d}\right)
$$

该上界还未扣除 queue、prefix miss、retraction/rebootstrap、PP bubble，以及 PD 与 EP A2A 共享网络时的竞争。HiSparse 还要单独测 Host→Device 带宽；H2D 与 PD 网络不是同一项指标。

## 5. 四个核心矛盾

### 5.1 GPU 容量与跨轮复用

普通多池更容易保留完整缓存语义，但 1M×并发带来显著 HBM 压力。HiSparse 将 C4 逻辑容量与 GPU working set 分开，却要求关闭 radix；HiCache 又与关闭 radix 互斥。当前没有一条现成路径同时获得“C4 Host 分层”和“DSV4 多池跨请求 Prefix”。

### 5.2 增量写入与全历史检索

[dense_prefill_topk()](../../../python/sglang/srt/layers/attention/dsv4/dense_prefill_indexer.py#L22-L89) 对当前 `Q` 个 query 与压缩 context 宽度 `C` 产生 logits，单 chunk 为 `O(Q×C)`。完整增长式 N-token prompt 累计近似 `O(N²/4)`。

`_SCORE_BUDGET_BYTES=2<<30` 只控制一次 materialize 的 logits rows，降低 scratch 峰值，不减少总 score FLOPs。当前 `_select_tile()` 仍先调用 `fp8_fp4_mqa_logits(..., width)` 产生全宽 logits，再发布 candidate blocks 或 mask；不能把现状描述成已经完成块级粗排。

### 5.3 Early send 与可消费状态

page/chunk 可提前发送，但 Decode 能否开始使用 Prefix 还取决于所有必要子池、边界长度和最终 state 是否一致。只看到“部分页到达”不能立即 attach；未来 producer 流水必须保留最终 commit 边界。

### 5.4 高利用率与恢复能力

更激进的并发和 admission 可能提升 goodput，也更容易触发 retraction。DSV4 HiSparse 当前普通 OOM 路径为：

```text
check_decode_mem()
  -> retract_decode()
  -> release_req(offload_kv=True)
  -> retraction_backup(backend="cpu_tensor")
  -> allocator.get_cpu_copy()
  -> NotImplementedError
```

[BaseTokenToKVPoolAllocator.get_cpu_copy/load_cpu_copy](../../../python/sglang/srt/mem_cache/allocator/base.py#L223-L229) 未实现，DSV4 HiSparse allocator 没有覆盖；目标模型也不满足 [dsv41_dspark_needs_rebootstrap()](../../../python/sglang/srt/mem_cache/common.py#L197-L207) 的 DSpark+ratio-2 条件。恢复能力完成前，首要目标应是准确 admission 和受控失败，而不是依赖 retraction。

## 6. 目标架构总览

### 6.1 六个逻辑组件

**[目标设计]** 这里定义职责和契约，不锁死具体 Python 类、RPC IDL 或存储技术。

| 组件 | 职责 | 与现有代码的接缝 |
|---|---|---|
| Session/Prefix Directory | 从会话、token fingerprint、模型与布局版本定位稳定 Prefix | radix/chunk cache、router affinity |
| Multi-pool Prefix Store | 统一拥有 SWA、C4 KV/index/state、C128 KV/request state | DSV4 memory pool、allocator |
| Incremental Prefill Planner | 计算安全命中边界、需重算范围和新增页 | schedule batch、compressor/indexer state |
| PD Manifest/Transfer Coordinator | 描述已有页、缺失页、新增页、state 与完成位图 | prefill/decode transfer metadata |
| Decode Attach Manager | prepare 容量和布局，commit 后原子绑定请求 | decode prealloc/transfer/waiting queue |
| Admission/Recovery Controller | 多池容量决策、backup、restore、rebootstrap、fallback | allocator、retraction、scheduler |

### 6.2 目标端到端流程

```text
Request + session metadata
  -> Prefix lookup
  -> token/model/layout/boundary validation
  -> Incremental Prefill Planner
  -> new pages + updated state + transfer manifest
  -> page/chunk early send and existing-page skip
  -> Decode prepare
  -> final state + completeness validation
  -> atomic attach commit
  -> Decode
  -> publish new committed Prefix or rollback mutable tail
```

### 6.3 三条一致性原则

1. **Prefix 是多池事务，不是一个 `prefix_len`。** C4 KV 存在但 C4 index/state 缺失时不能算命中。
2. **Committed Prefix 不可变，当前轮尾部可变。** 新轮构建失败不得污染旧版本。
3. **传输完成不等于 attach 完成。** 所有必要页、有效长度、版本和 state 通过校验后才允许 commit。

## 7. 功能设计一：多池 Prefix 对象

### 7.1 要解决的问题

当前 ChunkCache 只维持同一未完成请求，无法把上一轮完成的 DSV4 多池历史作为下一轮可复用资产。目标是让稳定历史同时支持 Prefill skip、PD skip 和 Decode attach。

### 7.2 Prefix 契约

**[目标设计]** Prefix manifest 至少需要表达：

| 类别 | 必要信息 |
|---|---|
| 身份 | `prefix_id`、session/content fingerprint、token fingerprint |
| 兼容性 | model revision、KV dtype、DSA layout、page size、TP/PP/CP 拓扑、layout version |
| 范围 | full-token 长度、各压缩池有效长度、最后完整页和 partial tail |
| SWA | 尾部 window 页、ring 与续算所需状态 |
| C4 | KV 页、index 页、compressor/indexer 边界 state |
| C128 | KV 页、request-scoped state |
| 所有权 | GPU/Host/Decode replica、引用计数或 lease、generation |
| 完整性 | per-pool completion bitmap、checksum、commit epoch |

### 7.3 生命周期

```text
Building
  -> Committed
  -> Leased / Attached
  -> Extended as a new generation
  -> Evicting
  -> Invalid / Evicted
```

- `Building` 只对当前构建事务可见；
- `Committed` 后内容不可变；
- 新一轮基于旧 generation 构建新 generation，而不是原地覆盖；
- `Evicting` 停止发放新 lease，已有使用者退出后释放物理页；
- model/layout/token 任一不匹配时整体 miss，不能混用不同 generation 的子池。

### 7.4 MVP 边界

第一版只支持同一会话、严格稳定前缀、优先固定路由，不做任意跨会话内容去重，也不立即替换成通用 sparse-aware radix。这样可先验证多池一致性和增量收益，再扩大共享范围。

## 8. 功能设计二：增量 Prefill、PD Manifest 与 Decode Attach

### 8.1 增量 Prefill

输入：Committed Prefix、本轮新增 token、generation、当前模型和拓扑。输出：

- 新增 C4 KV 与 C4 index 页；
- 新增 C128 KV 页；
- 更新后的 SWA tail/ring；
- 更新后的 compressor、indexer 和 request state；
- 描述各池有效长度、页范围与依赖关系的 manifest。

边界要求：

- 历史 token 或 template 有任何编辑时强制 miss；
- Prefix 边界必须提供继续计算所需的全部 state；
- ratio/page 不对齐尾部采用重算或 copy-on-write，不能共享可变尾页；
- 增量与 full Prefill 的 next-token logits、输出和各池 checksum 必须在允许误差内一致。

### 8.2 PD 增量传输

Manifest 区分三类内容：

1. **Existing Prefix reference**：Decode 已有且版本匹配的页，只传引用；
2. **Missing/incremental pages**：Decode 缺失或本轮新增的页，沿用 chunk/page early send；
3. **Final state/commit**：最终有效长度、state、checksum 和 commit epoch。

传输必须幂等：同一 generation/page 的重试不能重复 attach；部分失败按 completion bitmap 补传。PP 仅发送本 stage 的 ratio 层；TP/DCP 目标布局不能表达时应拒绝 attach，而非静默解释错误。

### 8.3 Decode 两阶段 Attach

**Prepare：** 查找已有 Prefix、获取 lease、校验 generation/layout、预分配缺失页和输出 headroom，并返回可复用长度与目标页。

**Commit：** 所有必要页和最终 state 到达后，原子更新请求的多池映射与 committed length，之后才进入 Decode waiting/running。

失败规则：

- Prepare 失败：不改变旧 Prefix；
- 传输部分失败：保留完成位图并补传，或释放本轮增量；
- generation 变化：重新 lookup 或 full Prefill；
- Commit 前取消：释放本轮目标页和 lease；
- 任何子池不一致：整体回退，不允许“部分 Prefix 命中”。

### 8.4 场景级收益目标

对 `H_k≈1M、Δ_k=128/1K/8K`：

- `prefill_skipped_tokens≈P_hit×H_k`；
- 主要 PD payload 接近 `Δ_k×3850.25B` 加变更 state；
- Decode 不重复分配已 attach 的历史页；
- Prefix miss 的结果与当前完整链路一致。

## 9. 功能设计三：HiSparse 与 Prefix 分层统一

### 9.1 目标与分层原则

当前 HiSparse 把 C4 KV 主体放在 Host，并给 Decode 分配较小的 GPU working set；但关闭 radix 后没有可复用 Prefix 索引。目标不是简单给 `ChunkCache.match_prefix()` 增加返回值，而是把逻辑 Prefix ownership 与物理 GPU residency 解耦。

**[目标设计]** 分层 Prefix 节点至少关联：

- token span、parent/generation 和稳定边界；
- C4 Host 逻辑页及其有效长度；
- C4 index、C128 页和各类 state；
- SWA tail snapshot 或 copy-on-write 尾部；
- GPU C4 working-set hints，而不是把 working set 当成 Prefix 所有者；
- lease、last access、Host/Device eviction 状态。

### 9.2 分阶段范围

1. **会话级 Prefix：** 精确 `session_id` 与 token 校验，C4 主体可在 Host，固定 Decode affinity；
2. **稳定公共前缀：** 系统 prompt、文档模板等严格相同内容可以跨会话共享；
3. **通用 sparse-aware radix/HiCache：** 再研究树节点、Host/Storage tier、远程读取与通用淘汰。

每个阶段都必须覆盖 SWA、C4 KV/index/state 和 C128 KV/request state，不能只共享 C4。Prefix 驱逐应先撤销 GPU residency，再在无 lease 时释放 Host 逻辑页；异步 H2D 期间禁止释放源页。

### 9.3 主要风险与验收

- 命中后 C4 H2D 可能成为 Decode 首 token 长尾；
- SWA 尾部和 ratio-4/128 压缩边界可能不一致；
- Host C4 与 Device index 跨 generation 会造成静默错误；
- Prefix 树锁竞争、元数据和远程副本会抵消收益。

验收要求同时看到 GPU 容量下降、跨轮 Prefill/PD skip 生效，并且 H2D P99 不抵消端到端收益。

## 10. 功能设计四：Admission、Retraction 与 Rebootstrap

### 10.1 近期 MVP：容量感知 Admission

在完整 backup 成熟前，目标是正常负载零 retraction。Admission 估算应与 allocator 使用同一套 page rounding 和 pool 口径，至少输入：

- 已 attach Prefix 与本轮新增、最大输出长度；
- SWA、C4 KV、C4 index、C128 的新增和尾部页；
- C4 Host 与 GPU working set；
- MTP/verify 额外槽位；
- PP-local ratio、并发、reserved tokens 与 allocator headroom；
- Decode transfer/waiting queue 和 Host pinned pool 余量。

决策可为 `accept`、`queue/defer`、`route-to-other-replica` 或 `reject`。必须记录 estimated/actual peak、估算误差、admission wait 和“通过 admission 后仍 retraction”的次数。

### 10.2 近期安全底线：Fail-closed

若当前 pool/layout 不支持 backup，retraction 应显式中止、迁移或 rebootstrap 该请求，并返回可诊断错误；不能让未捕获的 `NotImplementedError` 终止 scheduler。该行为是安全底线，不代表恢复能力已完成。

### 10.3 中期：DSV4 多池一致快照

**[目标设计]** Retraction snapshot 必须作为一个事务覆盖：

- SWA KV 与 ring；
- C4 KV、C4 index、compressor/indexer state；
- C128 KV 与 request state；
- committed length、token-to-KV 映射、page mapping；
- model/layout/generation 与 checksum。

不能只给 allocator 补一个 `get_cpu_copy()`：若状态池或映射未一起备份，恢复后的下一 token 仍可能错误。Restore 完成全部校验前，请求不得重新进入 running batch。

### 10.4 Rebootstrap 的定位

Rebootstrap 是无法可靠备份时的故障兜底，不是正常调度机制。1M 全量 reprefill 和 PD 重传会产生高恢复时延，多个请求同时发生会形成恢复风暴；必须限流并设置恢复队列。若已有 committed Prefix，应优先从该边界恢复，而不是从 token 0 开始。

## 11. 功能设计五：计算路径与 Dense Indexer 优化

### 11.1 建立算子级计算账本

近期先按 Prefill full、Prefill incremental、Decode 三种 forward mode，记录每层/每 chunk 的 token 数、历史宽度、输入输出 shape、dtype、FLOPs/bytes、kernel 时间和 collective 时间。至少拆分：

- mHC pre/post、Sinkhorn 与边界融合；
- Q/KV/Indexer 投影及动态量化；
- C4/C128 compressor 和状态更新；
- C4 dense score、top-k、candidate/mask、sparse attention；
- C128 attention；
- MoE router、dispatch、专家 GEMM、combine；
- norm、残差、LM head、sampling；
- metadata、page-table、Graph replay 与 CPU launch gap。

对每个候选优化，同时记录 `kernel speedup`、`stage speedup` 和 `E2E speedup`，避免优化一个占比很低的算子。计算侧验收主指标应包含 TTFT、ITL、tokens/s、accepted tokens/s、SLO-goodput，以及单位新增 token 的 GPU time。

### 11.2 Dense Indexer：先证明它是瓶颈

近期先增加每层/每 chunk 的 `Q`、压缩 context `C`、logits scratch、score kernel 时间、candidate 数和占 TTFT/ITL 比例。只有 Dense Indexer 在目标负载上主导时才进入算法改造。

### 11.3 Dense Indexer 目标计算路径

**[目标设计][待实验]**

```text
历史压缩 token
  -> 构建 block summary/index
  -> query 对 block 做粗排
  -> 选择候选 block
  -> 只对候选 token 做精排
  -> exact/dense fallback
```

Block metadata 需要 token range、summary vector、quant scale、valid count 和 generation。近似候选会改变注意力集合，验收必须包括 exact top-k recall、首 token logits、长文检索/多轮问答质量、不同位置分布和 fallback 比例，而不能只比较 kernel 时间。

### 11.4 主干、MoE 与 Decode 专项

若 profiler 显示 Indexer 不是唯一主导项，则按以下顺序处理：

1. **小 `Δ` 主干：** 先合批，再评估 CUDA Graph 和 norm/quant/linear、残差/激活融合；目标是减少小 GEMM 和 memory-bound kernel 的固定成本。
2. **MoE：** 按 expert token histogram 检查负载不均衡和小专家 batch；优先跨会话连续批处理、grouped/persistent GEMM，以及 dispatch/combine 与计算重叠。任何 TBO 方案都要确认切分后每个 expert 的有效 token 数没有进一步下降。
3. **通信：** 分开统计 TP collective、EP A2A、CP gather/top-k merge 和 PP bubble；只有通信确实在关键路径上，才调整并行度或拓扑。
4. **Decode：** 联合评估 Graph、MTP 和候选跨步复用。MTP 增加一次 verify 中的 query 数，可能提高 GEMM 利用率，也可能成倍增加历史检索；应以每个 accepted token 的总 GPU 时间判断。
5. **LM head 与 sampling：** 保证只对必要位置计算 logits；若 vocabulary GEMM 或采样占比显著，再研究张量并行、融合和候选采样。

### 11.5 不提前承诺的事项

本文不指定 ANN 算法，不承诺把完整流程从二次降为线性，也不把 Blackwell candidate 路径外推到 Hopper。跨轮候选复用、Host/GPU 分层索引只有在 Prefix MVP 和质量基线建立后再研究。

## 12. 功能设计六：PD Producer 流水与多池可观测性

### 12.1 Producer 流水的启用条件

只有测得：

$$
T_{PD,non-overlap}/T_{TTFT}
$$

占比显著，才优先开发更细流水。第一版优先按 layer group 发布，而不是 43 层逐层发送，以减少事件、metadata、注册和小传输开销。

Producer segment 至少描述 request、chunk、layer range、pool type、source/destination page、ready event、generation 和 final-state 标志。必须保证：

- layer/group 写入完成后才允许 RDMA 读取；
- PP-local ratio slicing 正确；
- backpressure 不反向拖慢 Prefill；
- Decode 在最终 state commit 前不消费不完整 Prefix；
- 与现有 chunk/page early send 比较净收益，而非只看发送起点提前。

### 12.2 多池可观测性是阶段 0 功能

指标标签至少包含 request/session/prefix generation、`pool_type`、`tier` 与 `phase`：

| 维度 | 建议指标 |
|---|---|
| 多轮 | `N_k/H_k/Δ_k`、matched/skipped token、miss/fallback reason |
| Prefill | chunk/layer 时间、Indexer Q/C/scratch、C4/C128 attention、mHC、MoE、collective |
| 计算 | 各算子 input/output shape、dtype、FLOPs/bytes、kernel/launch gap、Graph replay、单位新增 token GPU time |
| MoE | expert token histogram、grouped GEMM shape、dispatch/combine/A2A、负载不均衡 |
| Decode | dense refresh/candidate reuse、MTP draft/verify、accepted tokens/step、每 accepted token 扫描次数 |
| PD | 各池 actual/skipped/retry bytes、queue、GB/s、non-overlap tail |
| Cache | logical/device/host pages、lease、eviction、partial tail |
| HiSparse | Host pool、GPU working set、H2D pages/time/P99 |
| Attach | prepare/commit latency、version/checksum failure、rollback |
| Recovery | admission estimate error、retraction、backup/restore/rebootstrap |
| 请求 | TTFT/ITL/E2E、完成率、SLO-goodput |

这些指标必须能回答：第 `k` 轮为什么仍处理了 1M token——是 token 不同、Prefix 未发布、布局不兼容、Decode 没有 residency、边界 state 不安全、容量不足，还是主动回退。

## 13. 分阶段交付路线图

### 阶段 0：事实基线与可观测性

交付：

1. Profile A 跑通 128K、512K、1M 单并发；
2. 校准每个子池容量、PD bytes 与 3850.25B 理论值；
3. 建立首轮、当前第 2 轮、OOM 三类 trace；
4. 拆分 Indexer、attention、MoE、PD non-overlap 和 Decode queue；
5. 确认启动日志中的 ratio/cap/PP/MTP sizing mode。

退出条件：1M 正确结束；实测容量和传输差异可解释；任一请求可还原端到端生命周期。

### 阶段 1：容量实验与故障安全

交付：

1. Profile B 的 Mooncake Host/Device 混合落点跑通；
2. 建立 Host、GPU working set 与 H2D 指标；
3. 实现或验证保守 admission；
4. 强制触发 retraction，做到恢复或受控失败；
5. 对 rebootstrap 限流并避免恢复风暴。

退出条件：不支持组合启动期 fail-fast；目标并发无意外 retraction；故障不导致 scheduler 退出。

### 阶段 2：同会话多轮增量 MVP

交付：

1. Multi-pool Prefix manifest 和 generation；
2. committed Prefix + mutable tail 生命周期；
3. 增量 Prefill；
4. PD existing-page skip 与 final commit；
5. Decode prepare/commit attach；
6. miss/evict/失败时 full Prefill fallback。

退出条件：`Δ_k=128/1K/8K` 均能增量处理；full/incremental 输出一致；实际 PD bytes 主要随 `Δ_k` 变化；并发会话无页污染。

### 阶段 3：HiSparse 分层 Prefix 与恢复增强

交付：

- C4 Host 逻辑页纳入 Prefix ownership；
- GPU working set 与 Prefix 生命周期解耦；
- 多池 backup/restore；
- Prefix 边界 rebootstrap；
- admission 从保守上界升级为实时多池估算。

退出条件：Host 分层与跨轮 Prefix 可同时工作；驱逐/H2D/restore 期间结果正确；容量收益未被 P99 H2D 抵消。

### 阶段 4：Profiler 驱动的性能专项

- Indexer 主导：先推进元数据/page-table 复用与 score-top-k 融合，再评估跨层共享候选、History/K-CP、精确 block pruning，最后才考虑 ANN；
- 小 `Δ` 固定开销主导：推进跨会话合批、CUDA Graph 与安全的 norm/quant/linear、残差/激活融合；
- C128 主导：推进 tiled online-softmax、页表复用和历史分片；
- MoE/网络主导：先增加 expert 有效 batch 和 grouped/persistent GEMM，再评估 TBO、EP backend、通信重叠和拓扑隔离；
- Decode 主导：联合评估候选跨步复用、MTP、Graph 和多流，以每个 accepted token 的 GPU 时间决策；
- PD 尾部主导：推进 layer-group，再评估 layer producer。

所有专项以目标 SLO-goodput 和正确性决定保留，不使用固定的先验收益百分比。

## 14. 验证矩阵与验收

### 14.1 基础实验矩阵

| 阶段 | 固定/变量 | 核心问题 |
|---|---|---|
| 首轮正确性 | A；128K/512K/1M；单并发；加速项关闭 | 模型、pool、PD 与输出是否正确 |
| Prefill | chunk 至少 3 档 × 合法 CP；GPU 总预算可比 | TTFT、scratch、collective 和拐点 |
| 当前多轮 | 历史 128K/512K/1M；新增 128/1K/8K；2/4/8 轮 | A/B 的重复计算、传输与 H2D 代价 |
| PD/网络 | A 可用后端；B 只测 Mooncake；并发 1/2/4/8 | 有效带宽、queue、EP 竞争和尾部 |
| 容量故障 | 并发、memory fraction、输出上限逐级升压 | admission 误差与 retraction 行为 |
| Prefix MVP | hit/miss/partial-tail/evict/版本变化/Decode 迁移 | 增量闭环正确性和回退 |
| 传输故障 | page 丢失/重试/取消/commit 前后失败 | 幂等、rollback 和旧 Prefix 隔离 |
| 加速项 | metadata reuse、score-top-k fusion、Graph、合批、History/K-CP、TBO、MTP 一次一项 | kernel/stage/E2E 收益与净 SLO-goodput |
| 算法近似 | H=128K/512K/1M；位置/主题/轮次分桶 | top-k recall、logits、长文质量、fallback 比例 |

### 14.2 最低验收条件

1. 1M 首轮完整结束，无 NaN、transfer failure、checksum mismatch 或 scheduler 退出。
2. 实测各池 payload 与理论值的差异能由 page、state、PP/TP layout、prefix skip 或 retry 解释。
3. Prefix hit 时 full/incremental 的 next-token logits、输出和状态校验在允许误差内一致。
4. Prefix miss、历史编辑、模型/layout 版本变化、驱逐和 Decode 迁移均自动回退，不复用错误页。
5. 并发会话、并发 extend 和 partial-tail copy-on-write 无跨请求 KV 污染。
6. 强制触发 Decode retraction；任何未处理 `NotImplementedError`、scheduler 退出或悬挂均判定不可上线。
7. Profile B 无 Host pool 耗尽、working-buffer 异常或不可接受的 H2D P99。
8. 加速项只有在正确性不变且目标 SLO-goodput 改善时保留。

### 14.3 本文尚未验证的内容

- 目标 NVIDIA 机器的 1M 可达性和输出质量；
- Mooncake 混合 Host/Device 传输的真实有效带宽；
- 实际最大并发、Host pinned memory 上限和 H2D P99；
- Indexer、MoE 与 PD 尾部的真实耗时排序；
- 多池 Prefix、增量 Prefill、Decode attach 和 backup/restore——这些是目标设计，不是当前现成功能。

## 附录 A：兼容矩阵与平台边界

| 组合 | 当前结果 | 证据/说明 |
|---|---|---|
| DSA，不开 HiSparse | 支持 | ratio-4/128 模型路径独立工作 |
| HiSparse，未关闭 radix | 启动失败 | [validate_hisparse()](../../../python/sglang/srt/arg_groups/hisparse_hook.py#L79-L101) 只校验，不自动改参数 |
| HiSparse + `--disable-radix-cache` | 使用无跨请求命中的 chunk cache | [registry.py](../../../python/sglang/srt/mem_cache/registry.py)、[chunk_cache.py](../../../python/sglang/srt/mem_cache/chunk_cache.py) |
| HiSparse + HiCache | 配置级互斥 | [kv_cache_hook.py](../../../python/sglang/srt/arg_groups/kv_cache_hook.py#L393-L397) |
| DSV4 compressed KV + PD Decode radix | 尚未支持 | [kv_cache_builder.py](../../../python/sglang/srt/mem_cache/kv_cache_builder.py) |
| DSV4 HiSparse PD：P off / D on + Mooncake | 当前预期拓扑 | P 保持连续 C4 源池；D 使用特殊 Host/Device 目的索引，见 [decode.py](../../../python/sglang/srt/disaggregation/decode.py#L1613-L1634) |
| DSV4 HiSparse PD：P on / D off 或 P on / D on | 不应使用 | P 端稀疏物理 C4 映射未进入当前传输协议，Decode 目的索引不能补偿源端映射缺失 |
| DSV4 HiSparse PD + 非 Mooncake | Decode 预分配失败 | [decode.py](../../../python/sglang/srt/disaggregation/decode.py#L1613-L1634) |
| Mooncake + HiSparse + Decode DCP relayout | 不支持 | [mooncake/conn.py](../../../python/sglang/srt/disaggregation/mooncake/conn.py) |
| DSV4 + Decode async offload manager | 不支持 | [decode_kvcache_offload_manager.py](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py) 只接受 MHA/MLA pool |

`SGLANG_DSV4_UNIFIED_KV_FP8` 真实生效受 AMD HIP、`unified_kv_triton`、gfx95 等门控；本文 NVIDIA 主线不适用。其生效平台与 PD 当前又会启动期 fail-fast，因此不进入近期路线图。Unified FP8 约 640B 相对 unified BF16 1024B 降低 37.5%，但不应与本文 packed V4 的 584B 比较成“带宽减半”。

## 附录 B：风险台账

| ID | 严重度 | 风险 | 触发阶段 | 当前规避 | 目标能力 |
|---|---|---|---|---|---|
| R1 | 高：多轮性能 | A 的 DSV4 Decode radix 未完整支持；B 明确无跨请求命中 | 第 2+ 轮 | 不承诺增量；固定负载实测 | 多池 Prefix + 增量闭环 |
| R2 | 阻塞：稳定性 | HiSparse retraction CPU copy 未实现 | Decode 内存压力 | 保守 admission、故障注入 | fail-closed、snapshot/rebootstrap |
| R3 | 阻塞：后端 | DSV4 HiSparse 只支持 Mooncake，DCP relayout 不支持 | Decode prealloc | B 固定 Mooncake、DCP=1 | 扩展 wire/layout 表达后再开放 |
| R4 | 高：容量/复用 | HiSparse 与 radix/HiCache 当前能力冲突 | 启动与跨轮 | A/B 分开评估 | 分层 Multi-pool Prefix |
| R5 | 高：潜在 TTFT | Dense Indexer 对长历史 `O(Q×C)` | Prefill | chunk/CP 网格、profiler | 分层候选 + dense fallback |
| R6 | 中：TTFT | PD 最终 state 和尾部可能无法完全重叠 | Prefill→Decode | 测 non-overlap tail | layer-group producer |
| R7 | 高：网络 | EP A2A、PD、H2D 竞争不同/共享链路 | 并发负载 | 分阶段测量、拓扑隔离 | admission 与网络感知调度 |
| R8 | 高：一致性 | 多池页、state、generation 部分完成 | 增量/故障路径 | 当前不启用目标功能 | manifest + prepare/commit/rollback |

## 附录 C：研发决策清单

### 直接使用前

- [ ] 确认模型配置、ratio 与 NextN 口径；
- [ ] Profile A 的 1M 单请求可完成；
- [ ] 实测各池容量与 PD bytes；
- [ ] 明确 PD 与 EP 网络拓扑；
- [ ] Profile B 已验证 Mooncake、Host pool、H2D 和 retraction；
- [ ] 所有加速项均有单变量对照。

### 多轮 Prefix MVP 进入开发前

- [ ] 明确每轮是完整 prompt 还是增量 API；
- [ ] 明确 session affinity 与 Decode 迁移策略；
- [ ] 证明可安全续算的 SWA/C4/C128 state 边界；
- [ ] 定义 layout/model/token fingerprint；
- [ ] 定义 immutable Prefix、mutable tail 和 commit epoch；
- [ ] 定义 full Prefill fallback 与容量预算；
- [ ] 设计 full/incremental 一致性测试。

### 性能专项进入开发前

- [ ] Indexer、MoE、PD non-overlap、H2D 已分别计时；
- [ ] 确认优化对象确实主导目标 SLO；
- [ ] 定义质量、正确性与回退标准；
- [ ] 使用端到端 goodput 决策，而非单 kernel 加速比。

## 附录 D：关键源码索引

- 模型 wiring：[deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py)
- DSV4 多池与 wire buffer：[deepseek_v4_memory_pool.py](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py)
- Pool sizing：[pool_configurator.py](../../../python/sglang/srt/model_executor/pool_configurator.py)
- Prefill PD 与 early send：[prefill.py](../../../python/sglang/srt/disaggregation/prefill.py)
- Decode prealloc、transfer 与 HiSparse backend：[decode.py](../../../python/sglang/srt/disaggregation/decode.py)
- State transfer：[utils.py](../../../python/sglang/srt/disaggregation/utils.py)
- PP ratio slicing：[common/conn.py](../../../python/sglang/srt/disaggregation/common/conn.py)
- Mooncake 混合目标：[mooncake/conn.py](../../../python/sglang/srt/disaggregation/mooncake/conn.py)
- Chunk continuation：[chunk_cache.py](../../../python/sglang/srt/mem_cache/chunk_cache.py)
- Cache 选择：[registry.py](../../../python/sglang/srt/mem_cache/registry.py)
- Cache/retraction backend 构建：[kv_cache_builder.py](../../../python/sglang/srt/mem_cache/kv_cache_builder.py)
- Retraction 公共逻辑：[common.py](../../../python/sglang/srt/mem_cache/common.py)
- Allocator CPU copy 边界：[allocator/base.py](../../../python/sglang/srt/mem_cache/allocator/base.py)
- Decode async offload：[decode_kvcache_offload_manager.py](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py)
- Dense Prefill Indexer：[dense_prefill_indexer.py](../../../python/sglang/srt/layers/attention/dsv4/dense_prefill_indexer.py)
- Candidate Indexer：[candidate_indexer.py](../../../python/sglang/srt/layers/attention/dsv4/candidate_indexer.py)

本文最终决策原则是：**先使首轮 1M 可测、容量可解释、故障可控，再完成同会话多池增量闭环；HiSparse 分层 Prefix、Indexer 算法和更细 PD 流水均建立在这个基础之上。**
