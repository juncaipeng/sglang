# PD 分离部署 DeepSeek V4：Decode 侧 HiCache 池化与跨轮复用方案

> 撰写日期：2026-07-31
> 适用基线：当前 `main`（代码行号以本次分析版本为准）
> 目标：在 Prefill/Decode 分离部署 DeepSeek V4 时，让 Decode 实例能够把已生成 token 的 KV 作为可持久化前缀缓存，并在后续请求中被同一 Decode 实例命中、恢复和继续使用。

---

## 1. 结论先行

### 1.1 需要解决的不是“把 D 的 KV 搬回 P”

正确的数据面应该是：

```text
第 N 轮请求
  P：计算 prompt，生成首 token
  P ───────── target KV ────────> D
  D：继续 decode，写入新增 token 的 target KV
  D：请求结束后把已提交前缀插入本地 Radix/HiCache

第 N+1 轮请求（包含第 N 轮历史）
  Router ── sticky/cache-aware ──> 同一个 D
  D：按完整 token 序列匹配本地 L1/L2/L3
  D ── decode_prefix_len + 目标槽位 ──> P
  P：只计算未命中的 prompt 增量
  P ── 增量 KV ──> D
  D：恢复/合并前缀，继续 decode
```

D 生成的 KV 不需要反向传给 P。D 保留前缀的物理所有权，P 只负责计算缺口并向 D 写入缺失 KV。这样才能避免长上下文在 P/D 之间重复传输。

### 1.2 当前代码的真实状态

当前主线已经具备大量基础设施，但生产组合仍未闭环：

| 能力 | 当前状态 | 关键位置 |
|---|---|---|
| D 侧 PD radix cache 开关 | 已存在 | `server_args.py:2963` |
| D 侧 HiCache prealloc/load-back mixin | 已存在 | `disaggregation/decode_hicache_mixin.py:58` |
| DSV4 六类池的 HiCache stack | 已存在 | `hybrid_cache/hybrid_pool_assembler.py:303` |
| DSV4 UnifiedRadixCache strategy | 已存在 | `hybrid_pool_assembler.py:821` |
| `decode_prefix_len` PD 协议 | 已存在 | `prefill.py:329`、`common/conn.py` |
| 普通模型 PD D 侧 HiCache 测试 | 已存在 | `test_disaggregation_decode_radix_cache.py:157` |
| DSV4 集中式 UnifiedRadixCache + HiCache 测试 | 已存在 | `test_unified_radix_cache_kl_dsv4.py:34` |
| DSV4 PD + D 侧 HiCache 集成测试 | 未见覆盖 | 需要新增 |
| DSV4 + EAGLE/MTP + D 侧 radix cache | 当前显式禁止 | `arg_groups/pd_disaggregation_hook.py:41-46` |
| D cache location 到 Router 的完整目录 | 当前没有闭环 | 需要扩展路由/控制面 |

### 1.3 推荐交付顺序

1. **Phase 0：非投机 DSV4**。复用现有 DSV4 UnifiedRadixCache/HiCache stack，打通 D 生成 KV 的提交、L2/L3 backup、load-back、PD 增量传输和同 D 路由。
2. **Phase 1：DSV4 投机兼容**。保留 target KV 的缓存语义，增加 draft KV 的提交/回滚边界，之后才解除 D 侧 radix cache 对 speculative decoding 的全局禁止。
3. **Phase 2：多 D 实例生产路由**。以 session sticky 为最小可用方案，以 KV block event/cache directory 为长期方案；缓存未命中必须有透明 fallback。

不建议第一步直接删除 `pd_disaggregation_hook.py:41-46` 的 speculative 禁止条件。该 guard 保护的是尚未验证的 target/draft KV 生命周期，不只是命令行兼容性。

---

## 2. 目标、非目标与验收标准

### 2.1 目标

- DSV4 decode 实例可以同时启用：
  - `--enable-hierarchical-cache`
  - `--disaggregation-decode-enable-radix-cache`
  - 合适的 `--hicache-size`、存储后端和 IO layout。
- D 生成的**已提交 target token**可以进入本地 RadixCache，并按 HiCache write policy 写入 Host/L3。
- 后续包含历史 token 的请求，如果被路由到相同 D：
  - L1 命中时直接复用 GPU KV；
  - L2/L3 命中时异步 load-back 到 D GPU；
  - P 收到 `decode_prefix_len` 后只计算增量；
  - D 正确接收增量 KV 并继续生成。
- DSV4 恢复后的推理状态必须完整：非 `unified_kv` 布局原子恢复 SWA、C4/C128 KV、C4 indexer 和 C4 state sidecar；`unified_kv` 布局恢复内容稳定的压缩 KV，并重新 prefill 尾部 SWA window/state。
- cache miss、缓存驱逐、恢复失败、D 重启、请求 abort 都能退化为普通 PD 流程，不返回错误 KV。

### 2.2 非目标

- 不让 P 直接读 D 的 Host/L3 cache。HiCache 仍是 D 的本地缓存。
- 不在第一阶段实现跨 D 物理迁移缓存。
- 不把尚未被 target 接受的 speculative branch 写入长期缓存。
- 不在本方案中解决 HiSparse 的 D 侧 host pool 直接 RDMA 语义。当前 DSV4 HiSparse 有独立的 host/device pool ownership，且与 decode radix cache 存在互斥约束。

### 2.3 验收标准

设原始 prompt 为 `P`，第 N 轮生成 `G`，第 N+1 轮请求输入为 `P + G + Q`：

| 场景 | 预期行为 |
|---|---|
| D L1 命中安全前缀 | P 只计算未命中的 trailing SWA window 和 `Q` |
| D L2 命中安全前缀 | D load-back 内容稳定的 pool，P 重算未持久化尾部和 `Q` |
| D L3 命中安全前缀 | D 从存储恢复持久化 recovery set，P 重算剩余窗口后继续 |
| D 无命中 | 与当前普通 PD 完全一致 |
| D cache 被驱逐 | 透明 fallback，不读旧槽位 |
| D 请求 abort | 不把不完整/未提交 token 插入可复用前缀 |
| 投机拒绝分支 | draft 分支不进入长期 cache，target committed prefix 可继续 cache |
| Router 选错 D | 只能导致 cache miss，不能读到别的 D 的槽位 |

---

## 3. 现有架构与问题定位

### 3.1 D 侧 HiCache 的入口

Scheduler 初始化时把 D 侧 HiCache 是否启用归一为：

```text
enable_decode_hicache
  = disaggregation_decode_enable_radix_cache
    AND enable_hierarchical_cache
```

代码位置为 `managers/scheduler.py:389-397`。D 预分配阶段在 `decode.py:980-988` 获取：

- `prefix_match.l1_prefix_len`：已经位于 D GPU 的前缀；
- `prefix_match.decode_prefix_len`：包括 L1、Host 和 Storage 命中的完整前缀；
- `[l1_prefix_len, decode_prefix_len)`：需要 load-back 到 D GPU 的区间。

`decode_hicache_mixin.py:58-240` 负责匹配、预取、恢复状态和重新匹配；恢复完成后在 `:300-311` 把恢复出来的槽位写回 D 的 `req_to_token_pool`。

### 3.2 P/D 已有的增量协议

P 在 bootstrap 阶段通过 `req.disagg_kv_sender.pop_decode_prefix_len()` 获取 D 传来的命中长度，位置为 `prefill.py:329`。D 侧通过 `CommonKVManager.req_to_decode_prefix_len` 保存该值，`common/conn.py` 中的 sender/receiver 负责传输 metadata。

因此，“D 缓存被 P 复用”的最小协议字段已经存在：

```text
D -> P：decode_prefix_len
D -> P：dst_kv_indices / dst_aux_index / dst_state_indices
P -> D：只写增量区域的 KV/state
```

第一阶段不需要新造一套“P 拉取 D cache”的 RPC；需要确保 DSV4 的多池索引和请求边界在这套协议上保持一致。

### 3.3 DSV4 HiCache 并不是单一 KV tensor

DSV4 目标模型的 KV 逻辑由多类池组成。`build_deepseek_v4_hicache_stack()` 位于 `hybrid_cache/hybrid_pool_assembler.py:303-527`，大致映射如下：

```text
Unified Radix FULL token/page
├── PoolName.KV：逻辑锚点
├── C4 compressed KV
├── C4 indexer K/scale
└── C128 compressed KV

Unified Radix SWA token/page
├── SWA KV（非 unified_kv 路径）
├── C4 compressor ring state
└── C4 indexer compressor ring state

请求级/层级状态
└── C128 compressor state：通常不作为普通 page sidecar 持久化，需按现有 DSV4 传输/恢复规则处理
```

`hybrid_pool_assembler.py:377-425` 创建 SWA、C4、C128、C4 indexer 的 host pool；`:451-511` 创建压缩状态和 C128 相关 pool；`:513-527` 由 `HybridCacheController` 统一协调。

这意味着不能只备份 `req_to_token_pool` 或 `c4_kv_pool`。只恢复主 KV 而没有 indexer/压缩 state，且又不重算对应尾部，会出现注意力选择页、压缩态或 SWA 边界不一致，最终表现为错误输出而不是简单 cache miss。

#### `unified_kv` 的重要例外：尾部 SWA 必须重算

`hybrid_pool_assembler.py:321-325` 明确说明：`unified_kv` 把 SWA ring 放在 per-request unified pool 中，该 ring 不是 content-stable，也不会 offload 到 Host，因此不会创建独立 SWA host pool。`:444-484` 同样只在非 `unified_kv` 时注册 C4 compressor state 和 C4 indexer state sidecar；`:499-500` 说明 C128 state 也有意不进入 HiCache，因为默认 page 边界恢复不需要消费该 state。

现有代码已经提供正确的补偿机制：

- `unified_radix_cache.py:1872` 的 `swa_reprefill_tail_tokens()` 返回需要重算的 trailing SWA window；
- `schedule_policy.py:103-112` 在 `match_prefix_for_req()` 中用该长度限制 radix match；
- `decode_hicache_mixin.py` 复用 `match_prefix_for_req()`，因此 PD D 侧不能绕过该函数自行按压缩 KV 长度声明完整命中。

目标语义应写成：

```text
可复用前缀 = 内容稳定、可恢复的压缩 KV 前缀
需要 P 重算 = trailing SWA window + 新增对话 token
```

这也意味着 `decode_prefix_len` 应是**扣除 SWA reprefill tail 后的安全前缀长度**，不是 C4/C128 的最大命中长度。现有 match 限长能力应直接复用，并新增 PD DSV4 集成测试锁定该行为。

### 3.4 DSV4 stack 已存在，但 PD 语义尚未被证明

`_DeepSeekV4Strategy.matches()` 在 `hybrid_pool_assembler.py:821-830` 识别 DSV4 `FULL + SWA` 组件组合，说明集中式 UnifiedRadixCache 已经能构造 DSV4 专用 HiCache stack。现有 DSV4 HiCache 测试验证的是集中式服务，不等价于 PD：

- 集中式请求的 producer、Radix tree、HiCache controller 都在同一个 Scheduler；
- PD 中 P 生成的 prompt KV 直接写 D 的预分配目标槽位；
- D 后续生成的 token 是否插入并 backup，要经过 Decode queue 的完成/abort/retract 生命周期；
- 下一轮命中后，P 计算边界必须与 D 的 DSV4 压缩页、SWA window 和 state index 完全一致。

### 3.5 当前最直接的硬阻塞：speculative decoding guard

`arg_groups/pd_disaggregation_hook.py:29-46` 对 D 侧 `--disaggregation-decode-enable-radix-cache` 做了三类限制：

1. 不能与 HiSparse 同时使用；
2. 不能使用 fake transfer backend；
3. `server_args.speculative_algorithm is not None` 时直接拒绝。

DSV4 的常见部署会启用 MTP/EAGLE V2，因此只删除第一、二类限制仍无法进入 D 侧 radix cache 路径。该限制必须在投机缓存语义完成后，按能力判断精确解除，而不是全局删除。

---

## 4. 端到端目标流程

### 4.1 第一轮：P 计算、D 接管、D 生成并缓存

```text
Client
  │ request(P)
  ▼
Router
  │ select P and D
  ▼
P Scheduler
  │ target prefill(P)
  │ produce first token / metadata
  │ read D-provided dst indices
  │ send target KV for P
  ▼
D PreallocQueue
  │ allocate target slots
  ▼
D TransferQueue
  │ receive P KV into six-pool destination
  ▼
D RunningBatch(PREBUILT)
  │ decode token by token
  │ append committed target KV
  ▼
D request finish / continuation point
  │ cache_finished_req or cache_unfinished_req
  │ insert Radix node
  │ asynchronous HiCache backup
  ▼
D local L1/L2/L3 cache
```

关键不变量：

- D 的 `req_to_token_pool` 是唯一的逻辑 token-to-slot owner；
- P 只能写 D 通过 bootstrap/metadata 明确给出的目标槽位；
- 当前 decode step 的 pending boundary token 不得提前作为完整 committed prefix 插入；
- C4/C128 的压缩页按各自 ratio 映射，但对外的 prefix length 仍使用 full token 坐标；
- state ring 的索引必须与 full/SWA page 的版本一致。

### 4.2 下一轮：D 命中，P 只算缺口

```text
Client request(P + G + Q)
          │
          ▼
Router ───────► D(owner of P+G)
                   │
                   ├─ match_prefix_for_req
                   │    ├─ L1 GPU hit
                   │    ├─ L2 Host hit
                   │    └─ L3 Storage hit
                   │
                   ├─ reserve destination slots for Q
                   ├─ start load_back for non-L1 hit
                   └─ send decode_prefix_len = safe_prefix_len
                                      │
                                      ▼
                                  P instance
                                      │
                         skip [0, decode_prefix_len)
                         target-prefill trailing SWA tail + Q
                                      │
                              send Q KV to D
                                      │
                                      ▼
                                  D instance
                         wait load_back + receive Q KV
                         run decode from P+G+Q
```

### 4.3 为什么不能只在 P 侧做 radix cache

P 侧缓存只能减少 P 自身的 prompt 计算，不能自动让 D 获得对应的 target KV：

- D 的 decode attention 读取的是 D GPU 上的 slot；
- D 的压缩池和 state pool 地址由 D allocator 决定；
- P 的 host/device pool 与 D 的 pool 没有共享物理地址空间；
- 即使 token hash 相同，也不能把 P 的 slot number 当作 D 的 slot number。

因此，跨轮复用的最小 owner 是 D；P 的 cache-aware 只可作为计算优化，不能替代 D 侧 cache location。

---

## 5. 重点难点与设计决策

### 5.1 难点一：DSV4 多池原子一致性

#### 问题

一次 DSV4 decode token 可能更新：

- SWA KV；
- C4 compressed KV；
- C4 indexer K/scale；
- C128 compressed KV；
- C4 compressor state；
- C4 indexer compressor state。

如果 HiCache backup 只完成其中一部分，Radix node 却已经对外可见，下一请求可能命中到“半个 token”。`unified_kv` 不要求持久化非 content-stable 的 SWA/state，但必须把它们对应的 trailing window 从可复用前缀中扣除并重算。

#### 决策

引入“逻辑 token commit”概念。对**本次布局声明为可持久化的恢复集合**，Radix Host/Storage 命中的可见性必须晚于所有必要 sidecar 的 backup 完成；对未持久化的 SWA/state，必须通过 reprefill tail 阻止其被声明为命中：

```text
forward complete
  -> target token committed
  -> six-pool device write complete
  -> sidecar backup submitted
  -> backup completion/event
  -> radix node visible / unlock
```

如果采用 `write_through`，可以在 backup event 完成后公开节点；如果采用 `write_back`，节点可以先在 D L1 可见，但在 Host/L3 backup 完成前必须标记为 local-only，不能让 Router 宣称该节点可跨重启/跨设备恢复。

推荐扩展 `HybridCacheController` 的 token/page 状态：

| 状态 | 含义 |
|---|---|
| `DEVICE_ONLY` | D GPU 有完整 sidecar，尚未完成 Host/L3 backup |
| `BACKUP_INFLIGHT` | 所有 sidecar backup 已提交，等待 event |
| `HOST_READY` | Host pool 可恢复 |
| `STORAGE_READY` | L3 storage 可恢复 |
| `INVALID` | 任一 sidecar 失败，必须从 Radix/目录撤销 |

不应让每个 sidecar 自己决定 Radix 可见性。

### 5.2 难点二：SWA window 与压缩页坐标不同

DSV4 存在多套 page size/窗口语义：

- full token 的 PD 逻辑坐标；
- C4 的 `full_position // 4`；
- C128 的 `full_position // 128`；
- SWA window/page；
- compressor ring 的 state slot。

PD 协议必须继续传 full token 坐标和 D 目标槽位，池内部再做映射。不能让 P 直接计算 C4/C128 的压缩页号。

推荐规则：

```text
PD metadata = full-token coordinate + D-owned destination
DSV4 pool   = translate full -> ratio-specific pool locally
SWA state   = translate full/SWA page -> ring slot locally
```

在 prefix 命中长度上使用 page-aligned committed length。若命中长度落在压缩 ratio/page 的中间位置，必须向下截断到可恢复边界，剩余 token 由 P 重新计算；宁可少命中，不能恢复半页压缩状态。

### 5.3 难点三：请求完成、流式中断和 retract 的所有权

普通完成路径在 `batch_result_processor.py:239-245`：

- 已完成请求调用 `release_kv_cache()`；
- 未完成但离开当前 decode batch 的请求调用 `maybe_cache_unfinished_req()`。

`mem_cache/common.py:132-153` 中 `release_kv_cache()` 最终调用 `tree_cache.cache_finished_req()`，并按 `effective_kv_committed_len` 处理 speculative tail。

D 侧要保证：

1. **正常完成**：只把完整 committed prefix 插入 D Radix；
2. **流式请求仍在生成**：可以按现有 unfinished cache 语义插入，但必须带 dirty/backup 状态；
3. **abort/error/transfer failure**：释放槽位，不插入树；
4. **decode retract**：不能把暂时 offload 的请求误认为完成。恢复时必须保留原 token 序列、cache lock 和 sidecar version；
5. **恢复失败**：撤销当前 prefix match，回退到重新 prefill，而不是继续使用部分 load-back 数据。

### 5.4 难点四：投机 decoding 的 target/draft 边界

当前 D radix cache 禁止 speculative 的原因是：长期 cache 的 owner 是 target KV，而每轮 speculative forward 同时产生 draft branch、accepted drafts 和 bonus token。必须明确：

```text
长期 cache：
  target model 已真正计算并 committed 的 token KV

短期 speculative workspace：
  draft branch、被拒绝 branch、未提交 branch
```

推荐规则：

- `correct_drafts` 不等于已经可缓存的 target KV；
- target verify 接受的序列才进入 committed token 序列；
- `bonus_token` 的 target KV 只有在对应 forward 已完成且请求状态提交后才能进入 cache；
- 被拒绝的 draft tail 必须释放/回收，不能进入 Radix node；
- `effective_kv_committed_len` 是插树和 backup 的唯一边界，不用 `req.output_ids` 长度直接推导；
- load-back 恢复 target prefix 后，draft worker 必须从恢复后的 committed boundary 重新建立它需要的 draft state，不能假设 draft sidecar 也已经在 D HiCache 中。

当前 `scheduler.py:878-887` 会把 target 的 memory pool 传给 draft worker；`eagle_worker_v2.py:181-194` 保存并分配 draft pool。实现时必须实测 DSV4 的 draft pool 是共享物理池、独立 logical view，还是独立 DSV4 pool。无论具体实现是哪一种，都必须在 prefix restore 后增加一致性检查：

```text
restored target committed length
== draft worker initialized committed length
```

若不能证明 draft state 可恢复，Phase 1 应明确只支持 non-spec DSV4。

### 5.5 难点五：Router 必须把续轮请求送回 cache owner

D 的 slot index 只在该 D 进程内有效。当前 PD Router 的普通负载均衡不能仅凭 token prefix hash 保证 D affinity。需要区分三种方案：

| 方案 | 实现量 | 命中率 | 适用阶段 |
|---|---:|---:|---|
| Session sticky | 小 | 高（同 session） | Phase 0，推荐 |
| D cache hint | 中 | 高 | Phase 1 |
| 全局 KV block directory | 大 | 最高、可扩展 | 长期 |

**Phase 0：Session sticky**

- 请求带稳定 session/conversation key；
- Router 用一致性 hash 选择 D；
- D 重启或 cache miss 时透明 fallback；
- 不宣称跨 D 迁移。

**Phase 1：D cache hint**

D 在 cache commit/evict 时发布：

```text
(model_version, tokenizer_version, tp/pp/dcp topology,
 prefix block hash, committed length, d_instance_id, cache tier,
 generation/epoch, expiration)
```

Router 用 hint 选择候选 D，再由 D 做最终精确 token match。Router 只做候选选择，不能把 hash 命中当作 KV 已存在的证明。

**长期：KV event directory**

现有 cache event/KV event 体系可作为基础，但需要把 D 侧 event 与 PD role 关联，并提供失效、版本和 owner epoch。D 退出时必须撤销所有 owner records。

### 5.6 难点六：模型版本和并行拓扑隔离

以下信息必须参与 cache namespace：

```text
model revision / weight version
backend and kv-cache dtype
page size
TP/PP/DP-attention/DCP topology
DSV4 compress_ratios
SWA window and layout
spec algorithm + draft model revision
tokenizer/chat-template revision
```

否则同一 token 序列在不同压缩配置、不同 PP stage 或不同权重版本下被误复用，会产生 silent wrong answer。

---

## 6. 详细实现方案

### Phase 0：非投机 DSV4 PD HiCache

#### 6.1 配置与能力判定

新增一个集中能力判断，替代散落的布尔组合：

```python
@dataclass(frozen=True)
class DecodeCacheCapability:
    enable_device_radix: bool
    enable_hierarchical: bool
    supports_dsv4_sidecars: bool
    supports_speculative_commit: bool
    supports_hisparse: bool
```

建议位置：`arg_groups/pd_disaggregation_hook.py` 或新的 disaggregation capability module。

第一阶段规则：

- DSV4 + non-spec + non-HiSparse：允许；
- DSV4 + HiCache + fake backend：拒绝；
- DSV4 + HiSparse：继续拒绝，除非另行实现 HiSparse host pool 与 HiCache sidecar 的统一 ownership；
- 其他不具备 sidecar adapter 的模型：保持当前行为。

不要按模型名简单放行；应按 `token_to_kv_pool` 是否有 DSV4 HiCache adapter 判断。

#### 6.2 D 侧 stack 初始化

复用现有：

- `registry.py` 的 UnifiedRadixCache 选择；
- `hybrid_pool_assembler.py:303-527` 的 DSV4 stack；
- `HybridCacheController` 的 backup/load-back；
- `DecodeHiCachePreallocMixin` 和 `DecodeHiCacheTransferMixin`。

需要补充的检查：

1. D decode role 初始化后断言 `tree_cache` 与 target DSV4 pool 的 layer mapping 一致；
2. PP/TP 场景校验各 rank 的 pool entry 顺序一致；
3. `PoolName` 顺序、layer mapping、item bytes、page size 作为 metadata fingerprint；
4. `enable_decode_hicache` 为真时，所有 DSV4 必需 sidecar 都注册到 controller；
5. 任一 sidecar 不支持 backup/load-back 时，启动阶段 fail fast，而不是运行到首个 cache hit 才失败。

#### 6.3 D 生成 token 的 cache commit

在 D 的完成/unfinished 路径和 scheduler cache 生命周期之间增加统一入口，例如：

```python
def commit_decode_prefix(
    req,
    tree_cache,
    committed_len,
    *,
    allow_device_insert=True,
    require_host_backup=False,
):
    """Insert only a fully committed DSV4 prefix and schedule sidecar backup."""
```

该入口应：

1. 使用 `req.effective_kv_committed_len()`；
2. 取得 full-token 的 req-to-token mapping；
3. 校验 DSV4 sidecar 的每个 page/state 是否完整；
4. 插入/更新 Radix node；
5. 让 `HybridCacheController` 按 write policy backup；
6. 失败时撤销节点并释放未提交槽位。

不要在 `decode.py` 中分别对 C4、SWA、state pool 做临时插入；所有 pool 必须由统一 controller 管理。

#### 6.4 PD prealloc/transfer 的增量边界

复用现有 `decode_prefix_len`，但增加如下约束：

- `decode_prefix_len` 表示 full-token 坐标的**已提交 target prefix**；
- prefix 必须 page/state 对齐；不对齐则取可恢复的最大前缀；
- P 计算 `[decode_prefix_len, prompt_len)`；
- D 预留新增 token 的目标槽位；
- DSV4 pool 内部依据 full position 把新增 token 映射到 C4/C128/SWA/state；
- P/D 两侧 metadata 必须携带 cache namespace fingerprint，发现不一致直接降级 cache miss。

#### 6.5 Phase 0 测试

单元测试：

- `decode_prefix_len` 为 0、L1 命中、L2 命中、L3 命中；
- page size 与 C4/C128 ratio 不整除；
- backup 中任一 sidecar 失败；
- load-back 中任一 sidecar 失败；
- request abort/retract/finish 的树和 allocator 状态；
- cache epoch/version 不匹配；
- restore 后 `req_to_token_pool` 与 sidecar index 对齐。

PD 集成测试：

1. P/D 启动 DSV4，D 打开 unified HiCache；
2. 第一轮生成较长 output；
3. 第二轮带完整历史，固定路由到同 D；
4. 对比关闭 D radix/HiCache 的 baseline 输出；
5. 强制 Host/L3 eviction 后重复请求；
6. 在传输中 abort；
7. D 重启后验证只发生 cache miss，不发生错误恢复。

必须同时校验：

- token output；
- logprob（如开启）；
- cache hit length；
- DSV4 C4 indexer 选页；
- SWA window 边界；
- host/storage backup bytes 和 event 顺序。

### Phase 1：解除 DSV4 + EAGLE/MTP 的组合限制

#### 6.6 明确 target/draft 的 cache ownership

先补一张运行时 ownership 表并用断言锁定：

| 对象 | 长期 HiCache owner | 生命周期 |
|---|---|---|
| target committed KV | D target tree/HiCache | 可跨轮复用 |
| target pending token | D target temporary allocation | 当前 forward 完成后才提交 |
| accepted target sequence | D target tree/HiCache | 按 committed length 提交 |
| rejected draft branch | draft temporary pool | 本轮结束释放 |
| draft recurrent/attention state | draft worker | 从 committed boundary 重建或单独持久化 |
| DSV4 target compressor state | target sidecar/controller | 与 target prefix 原子一致 |

如果 draft worker 的 state 不能从 target prefix 重建，就不能仅凭 `decode_prefix_len` 启动下一轮 speculative decode；需要增加 draft state 的恢复协议或暂时禁用投机 cache hit。

#### 6.7 speculative commit 规则

使用现有结果语义：

- `num_correct_drafts` 不包含 bonus；
- `num_accept_tokens` 包含 bonus；
- 长期缓存边界由 `effective_kv_committed_len` 决定，而不是由某个 draft tensor 长度决定。

每次 verify 后：

```text
1. 释放 rejected draft branch
2. 确认 target accepted sequence 的 KV 已写入 D target pool
3. 确认 bonus token 的 target KV 已完成
4. 更新 effective committed length
5. 只将 committed prefix 交给 Radix/HiCache controller
6. draft worker 从新的 committed boundary 继续
```

禁止把 draft-only KV 或被拒绝 tail 送给 `cache_finished_req()`。

#### 6.8 speculative 场景的测试矩阵

- topk=1 chain；
- topk>1 tree（若 DSV4 后端支持）；
- 全拒绝、部分接受、全接受；
- bonus token 在停止条件处结束；
- retract 后恢复；
- prefix L1/L2/L3 hit 后继续 EAGLE；
- draft state 无法恢复时的显式 fallback；
- overlap 与 non-overlap；
- PP/TP/DCP 组合。

只有这些测试通过，才把 `pd_disaggregation_hook.py:41-46` 从“所有 speculative 都禁止”改为“按 capability 精确禁止”。

### Phase 2：生产路由和可观测性

新增指标：

```text
pd_decode_cache_lookup_total{tier,result}
pd_decode_cache_hit_tokens
pd_decode_cache_restore_seconds{tier}
pd_decode_cache_backup_seconds{tier}
pd_decode_cache_invalidations_total{reason}
pd_decode_cache_owner_mismatch_total
pd_decode_cache_incremental_prefill_tokens
pd_decode_cache_fallback_total{reason}
```

日志至少带：

```text
request_id, session_id, d_instance_id,
cache_namespace, prefix_hash, requested_prefix_len,
matched_prefix_len, restored_prefix_len,
committed_len, cache_tier, cache_epoch
```

监控目标：

- 命中率不能只看 Router 预测，必须看 D 最终精确 match；
- restore 失败率；
- sidecar backup 不一致数；
- Router 选错 owner 的比例；
- 因 cache hit 而减少的 prefill token 数。

---

## 7. 备选方案比较

### 方案 A：只做 D 本地 HiCache + session sticky（推荐首发）

**优点**：改动最小，复用现有 DSV4 stack 和 `decode_prefix_len`，故障域清晰。
**缺点**：需要客户端/Router 提供稳定 session key；D 重启后命中率下降。
**适用**：先验证 DSV4 持久化 recovery set、SWA reprefill tail 和 PD 增量边界。

### 方案 B：P/D 共享远端 HiCache

**优点**：理论上多个 D/P 都可复用。
**缺点**：需要统一物理 layout、owner、锁、版本、sidecar 原子提交；DSV4 压缩 state 和 SWA ring 使共享存储复杂度很高。
**结论**：不建议作为第一版。

### 方案 C：P 复算 prompt，D 只缓存生成后缀

**优点**：不需要 P 访问 D 的 cache，且 D 侧只需缓存 output continuation。
**缺点**：如果下一轮请求包含长 prompt，P 仍重复计算；不能解决目标问题中的长上下文 prefill 成本。
**结论**：可以作为 cache miss fallback，不是完整方案。

### 方案 D：Router 只做 consistent hash，不做真实 cache directory

**优点**：工程简单。
**缺点**：session 不稳定时命中率有限；不能准确处理 D 驱逐和重启。
**结论**：作为 Phase 0，后续用 D cache events 增强。

---

## 8. 风险清单与回滚策略

| 风险 | 影响 | 预防/回滚 |
|---|---|---|
| 只恢复 C4/C128，遗漏 indexer/state | silent wrong answer | 启动时 sidecar completeness 校验；失败整体 cache miss |
| prefix 长度未按压缩/page 对齐 | 非法 state 或错误 attention | 向下取可恢复边界，缺口交给 P |
| D cache owner 变化 | 读错槽位 | cache namespace + owner epoch；不匹配必 miss |
| speculative rejected tail 入树 | 后续请求错误 | 统一 committed length；拒绝 tail 必释放 |
| backup 尚未完成就暴露节点 | load-back 读半成品 | controller 原子 commit/event gate |
| D reclaim 时仍有请求引用 | cross-request KV 污染 | lock/refcount；完成 load-back 前不释放 |
| HiSparse 与 radix cache 混用 | host/device ownership 冲突 | Phase 0 保持互斥，单独立项 |
| Router cache hint 过期 | 命中错误或额外延迟 | D 精确复核 token；hint 只做候选，不做事实 |
| D 重启后旧目录未清理 | 大量无效请求 | owner epoch/lease + heartbeat |
| MTP draft state 无法恢复 | speculative 结果错误 | Phase 1 先 fallback non-spec，不静默继续 |

回滚开关建议保留：

```text
--disaggregation-decode-enable-radix-cache=false
--enable-hierarchical-cache=false
```

并增加一个仅用于实验的 capability 级开关，而不是修改默认行为。任何 restore/backup 不一致都应降级为该请求 cache miss，不应关闭全局服务或返回错误模型结果。

---

## 9. 建议修改文件清单

### Phase 0

| 文件 | 修改内容 |
|---|---|
| `python/sglang/srt/arg_groups/pd_disaggregation_hook.py` | 将当前粗粒度限制改为 capability 校验；非投机 DSV4 允许 HiCache |
| `python/sglang/srt/managers/scheduler.py` | 暴露/记录 D decode HiCache capability 和 namespace |
| `python/sglang/srt/disaggregation/decode.py` | 在 prealloc、transfer、finish/retract 路径接入统一 commit/restore 边界 |
| `python/sglang/srt/disaggregation/decode_hicache_mixin.py` | 增加 DSV4 sidecar completeness、alignment、restore failure fallback |
| `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` | 完善 DSV4 pool entry fingerprint、sidecar 原子提交/恢复接口 |
| `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` | 统一 backup/load-back event 和 sidecar commit 状态 |
| `python/sglang/srt/disaggregation/common/conn.py` | 在已有 metadata 上增加可选 namespace/epoch/fingerprint |
| `python/sglang/srt/disaggregation/prefill.py` | 校验 decode prefix 边界，严格只计算增量 |
| `test/registered/disaggregation/` | 新增 DSV4 PD + HiCache 集成测试 |

### Phase 1

| 文件 | 修改内容 |
|---|---|
| `python/sglang/srt/arg_groups/pd_disaggregation_hook.py` | 仅保留不支持 capability 的 speculative 禁止 |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | prefix restore 后 draft state 初始化；verify 后 committed boundary 提交 |
| `python/sglang/srt/speculative/eagle_worker_common.py` | target/draft cache boundary 和 restore 状态传递 |
| `python/sglang/srt/mem_cache/common.py` | 统一 speculative committed length 到 DSV4 cache commit |
| `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` | accepted/bonus/rejected 分支与 cache commit 对齐 |
| `test/registered/speculative/`、`test/registered/disaggregation/` | MTP/EAGLE + DSV4 PD HiCache 矩阵测试 |

### Phase 2

| 文件/模块 | 修改内容 |
|---|---|
| `sgl-model-gateway/src/routers/http/pd_router.rs` | session sticky、D cache hint 候选选择 |
| `sgl-model-gateway/src/policies/cache_aware.rs` | D owner/cache block 维度的 cache-aware policy |
| `python/sglang/srt/disaggregation/kv_events.py` | D cache commit/evict/invalidate event |
| Router/worker discovery | owner epoch、lease、失效和版本隔离 |

---

## 10. 实施顺序与评审门槛

```text
代码能力确认
  ↓
Phase 0：non-spec DSV4 + D HiCache
  ↓ 通过 DSV4 recovery set backup/load-back + SWA 尾部重算 + PD 增量一致性
Phase 0.5：多轮 session sticky
  ↓ 通过 D cache hit / miss / restart / eviction
Phase 1：EAGLE/MTP committed boundary
  ↓ 通过拒绝采样、bonus、retract、restore 矩阵
Phase 2：D cache event + Router directory
```

每个阶段的评审门槛：

1. **正确性优先**：与关闭 cache 的逐 token 输出对拍；
2. **cache miss 等价性**：关闭或失效 cache 时走现有 PD 路径；
3. **可观测**：能区分 L1/L2/L3 hit、restore、backup、fallback；
4. **故障可恢复**：任何 sidecar/网络/存储失败都不能暴露部分 cache；
5. **性能数据**：报告减少的 P prefill tokens、D restore latency、KV transfer bytes、吞吐和首 token 延迟；
6. **并行覆盖**：至少验证实际部署的 TP/PP/attention backend/page size，不能只通过单卡 mock。

最终的验收公式是：

```text
输出正确性（cache on）
== 输出正确性（cache off）

P 计算 token 数（命中时）
< P 计算 token 数（无 D cache 时）

D cache miss 行为
== 当前已上线 PD 行为
```

---

## 11. 最终建议

当前最稳妥的工程路径是：

1. 把需求定义为“D owner 的 target KV 前缀复用”，而不是“P 读取 D cache”；
2. 先用已有 DSV4 UnifiedRadixCache/HiCache stack 打通 non-spec PD；
3. 以 full-token committed prefix 作为唯一 PD 边界，内部完成 C4/C128/SWA/state 映射；
4. 让六类 sidecar 由一个 controller 原子提交和恢复；
5. 通过 session sticky 先解决多 D 路由，再引入真实 cache directory；
6. EAGLE/MTP 只在明确 target/draft ownership、accepted/bonus/rejected 边界并完成对拍后解除 guard；
7. 任何不确定状态都降级为 cache miss，不允许以“看起来命中”的半缓存继续推理。

这样既能复用当前已经存在的 DSV4 HiCache 基础设施，也能把真正高风险的部分限制在可验证的边界内：**多池原子性、投机提交边界和 D owner 路由**。
