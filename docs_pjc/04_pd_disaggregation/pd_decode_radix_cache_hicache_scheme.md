# PD 分离：Decode 实例对「常规非 SWA 模型」支持 RadixCache + HiCache 的完整方案

> 文档日期：2026-09-02
> 代码基线：`main` @ `33ed29a0ee`
> 适用范围：`--disaggregation-mode decode` + 常规全注意力模型（dense / MoE 均可），即
> **非 SWA、非 SWA-compress、非 Mamba/SSM 混合、非 DSA（DeepSeek V4）** 的模型。
> 标注约定：结论尽量给出 `文件:行号`；凡是代码里没有直接写出、由我推导得到的结论都标注「**推断**」。

---

## 0. 先看结论

常规非 SWA 模型是 PD-Decode 侧「RadixCache + HiCache」这条链路上**最干净、限制最少的一条路**：
`kv_cache_builder.py` 里那一整块 decode-radix 门禁（SWA / SWA-compress / DSA / SSM）**一条都不会命中**，
所以不需要任何 opt-in 环境变量，两个 CLI 开关打开即可。

需要理解的四件事：

1. **两个开关是正交的，但 HiCache 依赖 RadixCache。**
   - `--disaggregation-decode-enable-radix-cache`：让 D 实例拥有真正的前缀树（否则 D 侧被强制降级成 ChunkCache）。
   - `--enable-hierarchical-cache`（+ `--hicache-ratio` / `--hicache-storage-backend`）：给这棵树挂上 L2（host DRAM）与 L3（外部存储）。
   - 二者同时打开才有 `Scheduler.enable_decode_hicache = True`（`managers/scheduler.py:455-524` 附近）。

2. **实际构建的树是 `UnifiedRadixCache`，不是 `HiRadixCache`。**
   这一点纠正了旧文档 [pd_decode_swa_hicache_support_scheme.md](pd_decode_swa_hicache_support_scheme.md) 的描述：
   当前 `main` 上 `HiRadixCache(` **没有任何实例化点**，它只作为语义参考存在。
   Decode 侧走的是 `UnifiedRadixCache`，`tree_components = (ComponentType.FULL,)`，再由
   `init_hicache()`（`mem_cache/unified_radix_cache.py:386`）挂上 L2/L3。

3. **复用的「变现方式」是省网络带宽，不是省 D 侧算力。**
   D 实例永远不跑 prefill forward（`ForwardMode.PREBUILT` 在 scheduler 里被短路），
   所以 D 侧命中前缀并不能省 FLOPs。真正的收益是：D 在 **预分配阶段**就把自己已经持有的前缀长度
   通过 PD 协议字段 `decode_prefix_len` 告诉 P，P 侧据此把 `start_send_idx` 前移，
   **RDMA 只传后缀**（`disaggregation/prefill.py:350-409`）。命中越长，跨机 KV 传输量越小。

4. **L2/L3 命中的那一段 KV，必须由 D 自己从 host/storage 搬回 device。**
   这就是 `disaggregation/decode_hicache_mixin.py` 那台**本地回载状态机**的全部职责。
   它与 P→D 的 RDMA 传输**并发**进行，用 `HiCacheRestoreGatedKVReceiver`
   把「RDMA 已完成但本地回载还没完成」的请求挡在 `KVPoll.Success` 之前。

一句话概括数据流：

```
D: 前缀匹配(L1) + 探测(L2/L3)  ──> decode_prefix_len ──> P: start_send_idx 前移，只传后缀
        │                                                        │
        ├─ 本地回载 L2/L3 ──> device（异步，layer-wise）           ├─ RDMA 传后缀 KV
        │                                                        │
        └──────────────── 两路都完成 ────────────────────────────┘
                                  │
                       req_to_token 拼成完整一行
                                  │
                    PREBUILT batch（不跑 forward）
                                  │
                  maybe_cache_unfinished_req → L1 插入 → 写回 L2/L3
```

---

## 1. 开关、默认值与门禁链

### 1.1 命令行

```bash
# Decode 实例：开启 D 侧 radix cache + 三级 HiCache
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --disaggregation-mode decode \
    --disaggregation-decode-enable-radix-cache \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-storage-backend mooncake \
    --tp 8 --port 30001 \
    --dist-init-addr prefill_host:30000
```

- `--disaggregation-decode-enable-radix-cache` 标注为 **EXPERIMENTAL**，默认关闭。
- 只开 `--enable-hierarchical-cache` 而不开 decode-radix：树被强制成 ChunkCache，HiCache 无从挂载，等于白开。

### 1.2 门禁链第 1 站：`arg_groups/pd_disaggregation_hook.py`

这是参数归一化层，decode 模式下：

| 条件 | 行为 |
|---|---|
| `disaggregation_decode_enable_radix_cache = True` | `declare_resolution(disable_radix_cache=False)`，并打日志 `"EXPERIMENTAL: Radix cache is enabled for decode server"` |
| 否则 | `declare_resolution(disable_radix_cache=True)`，日志 `"KV cache is forced as chunk cache for decode server"` |

同一个 hook 里的**硬拒绝**（与本方案相关的）：

- `--enable-hisparse` 与 PD 不兼容 → 拒绝；
- `transfer_backend == "fake"` → 拒绝（fake backend 不搬真实 KV，前缀语义无从保证）；
- `speculative_algorithm is not None` → 拒绝（**MTP / Eagle 与 D 侧 radix cache 目前互斥**，这是实践中最容易踩的一条）；
- PD-decode + DCP（decode context parallel）→ 同时拒绝 decode radix cache 和 hierarchical cache；
- DP attention → 只是 EXPERIMENTAL 警告，不拒绝。

### 1.3 门禁链第 2 站：`mem_cache/kv_cache_builder.py:252-284`

这一块是「模型结构 × decode-radix」的兼容性判定，**常规非 SWA 模型全部放行**：

| 模型形态 | 结果 |
|---|---|
| SWA（滑动窗口） | 需要（已废弃的）`SGLANG_ENABLE_UNIFIED_RADIX_TREE` 或 MLX 才放行 |
| SWA + `--enable-hierarchical-cache` | `raise ValueError` —— **SWA 至今不支持 D 侧 HiCache** |
| DSV4 / DSA | `raise ValueError` |
| `is_hybrid_swa_compress` | `raise ValueError` |
| `is_hybrid_ssm`（Mamba 混合） | `raise ValueError` |
| **常规全注意力** | **无任何限制，直接放行** |

> 注：上表中 SWA 那一行依赖的 `SGLANG_ENABLE_UNIFIED_RADIX_TREE` 已经进了
> `environ.py:1765-1794` 的 `_DEPRECATED_ENVS`（说明文字："The unified radix tree is
> the default tree cache now; unset this env."），所以这条门禁**事实上已经陈旧**——
> unified tree 本来就是默认。这属于代码清理欠账，不影响非 SWA 路径。（**推断**）

### 1.4 门禁链第 3 站：retraction backup 后端选择

`kv_cache_builder.py:134` `resolve_decode_retraction_backup()`：

选择 `host_pool` 作为 retract 备份池的前提之一是
**`not disagg.disaggregation_decode_enable_radix_cache`**（`kv_cache_builder.py:173`）。

也就是说：**一旦开了 D 侧 radix cache，retract 备份就退回 `cpu_tensor` 方案**，
不会去复用 HiCache 的 host pool。原因（**推断**）：host pool 的槽位归 radix 树管，
retract 备份如果也去抢同一批槽位，两套生命周期会打架。

`hicache_ratio` 默认 2.0（或 `BACKUP_ONLY_HICACHE_RATIO`），即 host pool 容量 ≈ 2× device KV pool。

### 1.5 门禁链第 4 站：`mem_cache/registry.py` —— 到底建了什么树

`default_radix_cache_factory` 的分支最终落到 `_create_unified_radix_cache`：

```python
# 简化示意
cache = UnifiedRadixCache(..., tree_components=(ComponentType.FULL,))   # 非 SWA/非 Mamba 只有 FULL
if enable_hierarchical_cache or retraction_backup == "host_pool":
    cache.init_hicache(server_args, params)
    tp_worker.register_hicache_layer_transfer_counter(
        cache.cache_controller.layer_done_counter
    )
```

两个关键点：

1. `tree_components` 对常规模型只有 `ComponentType.FULL`（SWA / MAMBA / C128 组件按需追加）；
2. `register_hicache_layer_transfer_counter` 把 `layer_done_counter` 注册给 TP worker ——
   这就是后面 D 侧回载状态机能做「**逐层等待**」的基础设施。

### 1.6 汇总标志位

`managers/scheduler.py:455-524`：

```python
self.enable_decode_hicache = (
    get_disagg().disaggregation_decode_enable_radix_cache
    and self.enable_hierarchical_cache
)
```

后续 `decode.py` / `decode_hicache_mixin.py` 里所有 L2/L3 相关分支都以这个布尔量为总闸。

---

## 2. `init_hicache()` 建了什么（`unified_radix_cache.py:386`）

| 动作 | 说明 |
|---|---|
| `host_memory_mode = get_memory().hicache_host_memory_mode` | `buffer_only` 模式只允许 `{FULL, SWA}` 组件，否则 `ValueError`；常规模型是 FULL，通过 |
| `attach_hybrid_pool_to_unified_cache(...)` | 真正创建 host pool + `HybridCacheController`（含 L3 storage backend） |
| `tree_core.set_hicache_enabled()` | 给树核心打上 HiCache 标记，影响 evict / backup 行为 |
| `write_through_threshold = 1 if write_policy == "write_through" else 2` | 一个节点被命中几次才触发 D→H 备份 |
| `is_write_back = (write_policy == "write_back")` | write_back 模式下 backup 不加锁 |
| `load_back_threshold = 10` | 小于该长度的 L2 命中不值得回载（**推断**：门槛过滤，避免为几个 token 起一次 H→D 传输） |
| `prefetch_stop_policy = hicache_storage_prefetch_policy` | L3 预取的中止策略 |
| `StorageAttachment(self)` + `atexit.register(self.shutdown)` | L3 后端可运行时挂载/卸载（启动、admin API、退出） |
| `BufferModePipeline`（仅 `buffer_only`） | off-tree staging 方案，见 [host_cache_off_tree_staging_scheme.md](../03_cache_memory/host_cache_off_tree_staging_scheme.md) |

L3 相关的解析参数（`prefetch_threshold` 默认 256、`prefetch_timeout_base` 1.0s、
`prefetch_timeout_per_ki_token` 0.25s、`hicache_storage_pass_prefix_keys`）都从
`--hicache-storage-backend-extra-config` 里解析（`unified_radix_cache.py:419-428`）。
**`prefetch_threshold = 256` 很重要**：短于 256 token 的后缀根本不会去查 L3
（`query_storage_hit_length` 在 `unified_radix_cache.py:1664` 直接 `return 0`）。

---

## 3. 阶段一：PreallocQueue —— 匹配、探测、预取、上报

入口：`DecodePreallocQueue.pop_preallocated()`（`disaggregation/decode.py:1080`）。
这是整套方案里信息量最大的一个函数。

### 3.1 前缀匹配 + 加锁：`_match_prefix_and_lock`（`decode.py:654`）

```python
result = match_prefix_for_req(
    self.tree_cache, req, req.origin_input_ids,
    cow_mamba=self.tree_cache.supports_mamba(),   # 常规模型 False
    include_req=True,
)
lock_result = self.tree_cache.inc_lock_ref(result.last_device_node)
req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock   # 非 SWA 为 None
return self._build_decode_prefix_match(req, result)
```

`match_prefix_for_req`（`managers/schedule_policy.py:60-200`）内部：

```python
reprefill_tail = tree_cache.swa_reprefill_tail_tokens()      # 非 SWA → 0
key_limit = max(0, len(token_ids) - reprefill_tail) if reprefill_tail else None   # → None
match_result = tree_cache.match_prefix(MatchPrefixParams(
    key=RadixKey(token_ids=token_ids, extra_key=req.extra_key,
                 limit=key_limit, cache_salt=req.cache_salt),
    cow_mamba=cow_mamba, req=req if include_req else None))
```

**非 SWA 的关键红利**：`swa_reprefill_tail_tokens()` 返回 0 ⇒ `key_limit is None` ⇒
**没有任何长度截断，可以整段前缀复用**。SWA 必须留一个窗口的尾巴重算，非 SWA 不需要。

匹配后 `match_prefix_for_req` 会填好 `req.prefix_indices` / `last_node` / `last_host_node` /
`best_match_node` / `host_hit_length`，并计算

```python
max_len = req._compute_max_prefix_len(len(token_ids))
req.num_matched_prefix_tokens = min(len(req.prefix_indices) + req.host_hit_length, max_len)
```

其中 `_compute_max_prefix_len`（`managers/schedule_batch.py`）：

```python
max_prefix_len = input_len - 1     # 至少留 1 个 token 以便算 logprob
if self.return_logprob and self.logprob_start_len >= 0:
    max_prefix_len = min(max_prefix_len, self.logprob_start_len)
return max(max_prefix_len, 0)
```

注意：这个 `input_len - 1` 的截断作用在 `num_matched_prefix_tokens`（统计口径）上，
而**真正决定传输量的 `decode_prefix_len` 是另算的**（见 3.2）。这个差异在后面
`get_new_prebuilt_batch` 重入 `init_next_round_input` 时会再次出现，需要留意。

### 3.2 `DecodePrefixMatch`：三级命中长度的载体（`decode_hicache_mixin.py:23-48`）

```python
@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor      # L1（device）命中的 KV 槽位
    l2_host_hit_length: int           # L2（host DRAM）额外命中
    l3_storage_hit_length: int        # L3（外部存储）额外命中
    last_device_node: Any
    last_host_node: Any = None
    prefetch_registered: bool = False

    @property
    def l1_prefix_len(self):      return len(self.prefix_indices)
    @property
    def decode_prefix_len(self):  return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length
    @property
    def needs_local_restore(self):return self.decode_prefix_len > self.l1_prefix_len
    @property
    def restore_token_count(self):return self.decode_prefix_len - self.l1_prefix_len
```

> 顺带一个规范问题：这里用了 `@dataclass`，而仓库规则 `.claude/rules/no-dataclasses.md`
> 要求新数据容器用 `msgspec.Struct`。属于存量欠账。

`_build_decode_prefix_match`（`decode_hicache_mixin.py:61`）的逻辑：

1. `l2_host_hit_length` 直接取匹配结果里的 host 命中；
2. L3 探测的前置条件是 `self.scheduler.enable_decode_hicache` 且
   `is_backuped(last_host_node) or is_root(last_host_node)`
   —— 即 host 侧锚点必须是「已完成备份的节点」或「根」，否则挂不上合法的 hash 链；
3. `matched_len = l1 + l2`，`suffix_tokens = req.origin_input_ids[matched_len:]`；
4. 调 `query_storage_hit_length(last_host_node, suffix_tokens, last_hash, prefix_keys)`；
5. 只有 `l3_storage_hit_length > 0` 才保留 `last_host_node`，否则置空。

`query_storage_hit_length`（`unified_radix_cache.py:1642`）值得单独看：

```python
if not enable_storage or cache_controller is None or prefetch_rate_limited():
    return 0
prefetch_key = RadixKey(new_input_tokens, extra_key, is_bigram, cache_salt).page_aligned(page_size)
if len(prefetch_key) < self.prefetch_threshold:      # 默认 256
    return 0
_, storage_hit_count = cache_controller._storage_hit_query(operation)
self._all_reduce_attn_groups(tensor(storage_hit_count), ReduceOp.MIN)   # 跨 attn group 取最小
storage_hit_count -= storage_hit_count % self.page_size                # 向下对齐到 page
return storage_hit_count
```

三个必须记住的性质：

- **同步阻塞**：它在 scheduler 主循环里同步查 L3。慢的 storage backend 会直接拖慢 D 侧调度。
- **`ReduceOp.MIN` 跨 attention group 归约**：TP 内所有 rank 必须一致，取最保守值。
- **向下 page 对齐**：L3 命中长度一定是 `page_size` 的整数倍。

### 3.3 启动 L3 预取：`_start_hicache_prefetch`（`decode_hicache_mixin.py:103`）

```python
tree_cache.prefetch_from_storage(
    req.rid, last_host_node, suffix, last_hash, prefix_keys,
    extra_key=..., cache_salt=...)
prefix_match.prefetch_registered = req.rid in tree_cache.ongoing_prefetch
```

异常处理是**优雅降级**：捕获异常 → 打 warning → `l3_storage_hit_length = 0`，退化成只用 L2。
这个设计对生产很关键：L3 后端抖动不会打死请求。

### 3.4 准入账本：为「还没回载完」的 token 预留额度

`pop_preallocated` 开头（`decode.py:1080` 起）：

```python
reserved_restore_tokens = self._hicache_pending_restore_tokens()
full_allocatable_tokens -= reserved_restore_tokens
```

`_hicache_pending_restore_tokens`（`decode_hicache_mixin.py:148`）遍历
`transfer_queue.queue`，累加所有满足「`hicache_restore_status == PENDING` 且
`hicache_restored_node is None`」的请求的 `restore_token_count`。

**语义**：这些 token 是「已经向 P 承诺过、但 device 上还没落地」的窗口。
如果不预留，L2/L3 回载和新请求预分配会抢同一批 device 槽位，导致回载失败率飙升。

循环体内每处理一个请求就把额度加上去：

```python
decode_req.prefix_match = prefix_match
if self.scheduler.enable_decode_hicache:
    self._start_hicache_prefetch(...)
reserved_restore_tokens += prefix_match.restore_token_count
req.kv.cache_protected_len = total_prefix_len
```

预算刷新走 `_allocatable_token_budgets(..., hicache_reserved_tokens=reserved_restore_tokens)`
（`decode.py:1620`）：

```python
available_size = allocator.available_size()
if disaggregation_decode_enable_radix_cache:
    available_size += self._radix_full_evictable()      # decode.py:441，树里可驱逐的部分也算可用
available_size -= max(reserved_tokens, need_space_for_single_req)
available_size -= <PREBUILT batch 预留>
available_size -= <retracted_queue 恢复所需>
available_size -= hicache_reserved_tokens              # 最后扣掉待回载窗口
```

即：**开了 radix cache 后，"可用" = 空闲 + 可驱逐**，这是 radix cache 能提高 D 侧并发的原因；
同时 HiCache 的待回载窗口被单独扣除，两者不重叠计账。

### 3.5 每请求的开关

```python
use_decode_radix_cache = (
    server_args.disaggregation_decode_enable_radix_cache
    and not decode_req.is_rebootstrap
)
```

**rebootstrap（真 retract 后重新握手）的请求不走前缀复用**。原因（**推断**）：
rebootstrap 时 `output_ids` 已非空，`_pre_alloc_fill_len` 走的是
`len(origin_input_ids) + len(output_ids)` 分支（`decode.py:740`），
前缀语义与首次 bootstrap 不同，混用容易错位。参见
[pd_true_retraction_rebootstrap_architecture.md](pd_true_retraction_rebootstrap_architecture.md)。

---

## 4. `decode_prefix_len` 协议：D 承诺，P 兑现

### 4.1 D 侧上报（`decode.py:1374` 附近 / `1505`）

```python
kv_indices = req_to_token[req_pool_idx][total_prefix_len : origin_input_len]
kv_indices = translate_kv_indices_for_transfer(kv_indices)
page_indices = kv_to_page_indices(kv_indices, page_size).astype(np.int32)

metadata_kwargs = {"decode_prefix_len": total_prefix_len}
sender.send_metadata(page_indices, metadata_buffer_index, state_indices, **metadata_kwargs)
```

注意 `kv_indices` 的切片起点是 `total_prefix_len`（= L1+L2+L3），
**不是** `prefix_len`（= L1）。也就是说 D 把 L2/L3 命中的那段也算进「我已经有了」，
承诺自己负责把它搬到 device 上。这是整套设计的核心契约。

### 4.2 P 侧兑现（`disaggregation/prefill.py:350-409` `finalize_bootstrap`）

```python
decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
req.start_send_idx = decode_prefix_len
req.disagg_decode_prefix_len = decode_prefix_len
num_kv_indices_to_send = len(req.origin_input_ids) - decode_prefix_len
req.disagg_kv_sender.init(num_pages, metadata_buffer_index)
```

P 侧**照单全收**，不做二次校验。所以如果 D 侧的回载失败而没有正确回滚，
就会出现「D 声称有、实际没有」的静默数据损坏。代码里对此的防线是
`_try_hicache_queue_load_back` 里的覆盖度检查（见 5.3）+ 失败即 abort。

> **风险点**：P 侧没有独立的一致性校验（例如 hash 校验），完全信任 D 的
> `decode_prefix_len`。这是设计上的信任边界，值得在压测时重点关注。（**推断**）

---

## 5. 阶段二：预分配的分配数学与 `req_to_token` 的「空洞」

### 5.1 `_pre_alloc`（`decode.py:1755`）

```python
# 1) 把 L1 命中的槽位写进 req_to_token 的 [0, prefix_len)
self.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, prefix_len)), prefix_indices)

# 2) 需要新分配的长度：从 total_prefix_len 开始，而不是 prefix_len
delta_len = fill_len - total_prefix_len

# 3) 不够就驱逐
if self._radix_full_available() < required_alloc_tokens:
    self.tree_cache.evict_for_alloc(EvictParams(num_tokens=required_alloc_tokens - ...))

kv_loc = alloc_for_decode_prealloc(...)      # decode.py:1936

# 4) 把新槽位写进 [total_prefix_len, total_prefix_len + len(kv_loc))
self.req_to_token_pool.write(
    (req.kv.req_pool_idx, slice(total_prefix_len, total_prefix_len + len(kv_loc))), kv_loc)

req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
req.prefix_indices = prefix_indices if prefix_len > 0 else torch.empty((0,), dtype=torch.int64)
req.set_extend_range(total_prefix_len, req.kv.kv_committed_len)
```

**这里出现了本方案最需要理解的一个中间状态：**

```
req_to_token[req_pool_idx]:
┌──────────────┬──────────────────────┬─────────────────────────┐
│ [0, L1)      │ [L1, total_prefix)   │ [total_prefix, fill_len) │
│ L1 命中槽位   │ ★ 空洞，尚未写入 ★    │ 新分配槽位（等 RDMA 填）   │
│ 已写         │                      │ 已写                     │
└──────────────┴──────────────────────┴─────────────────────────┘
```

中间那段 `[L1, total_prefix_len)` 对应 L2/L3 命中的 token，
**在 `_pre_alloc` 结束时是未写入的**，要等回载完成后由
`_commit_hicache_local_restore_to_req` 补上（见 5.4）。

`_required_alloc_tokens`（`decode.py:1743`）：

```python
if page_size == 1:
    return fill_len - prefix_len
return get_num_new_pages(seq_lens=[fill_len], prefix_lens=[prefix_len], page_size) * page_size
```

### 5.2 `alloc_for_decode_prealloc` 的非 SWA 分支（`decode.py:1936`）

```python
last_loc = prefix_indices[-1:] if prefix_len > 0 else torch.tensor([-1])
kv_loc = allocator.alloc_extend(
    prefix_lens=[total_prefix_len], prefix_lens_cpu=[total_prefix_len],
    seq_lens=[fill_len],           seq_lens_cpu=[fill_len],
    last_loc=last_loc, extend_num_tokens=delta_len, **extra_kwargs)
```

> **需要留意的隐含假设**：`last_loc` 取的是 `prefix_indices[-1]`，也就是
> **索引 `prefix_len - 1`** 处的槽位；但传给分配器的 `prefix_lens` 是 `total_prefix_len`。
> 当 `total_prefix_len > prefix_len` 时，这两者并不指向同一个位置。
> 之所以没出问题：`alloc_extend` 只在「`prefix_lens` 不是 page 对齐、需要接着上一页尾部继续写」
> 时才使用 `last_loc`；而 L1/L2/L3 的命中长度都被 page 对齐（`query_storage_hit_length`
> 显式向下取整，L1/L2 由树的 page-aligned insert 保证），`total_prefix_len % page_size == 0`
> ⇒ `last_loc` 不被使用。**这是一个正确但脆弱的隐式契约**：一旦将来出现非 page 对齐的命中，
> 这里会静默分配到错误的物理页。（**推断**）

### 5.3 SWA 尾巴相关的两处代码规范问题

- `_uses_swa_tail_prealloc()`（`decode.py:405`）用 `hasattr(..., "alloc_extend_swa_tail")` 做能力探测；
- `_swa_tail_len()`（`decode.py:456`）用 `getattr(server_args, "disaggregation_decode_enable_radix_cache", False)`。

两处都违反 `.claude/rules/no-getattr-defensive.md`（字段一定存在，应直接访问）。
对非 SWA 路径没有功能影响（`_swa_tail_len` 返回 0），但会掩盖字段改名类的真实 bug。

### 5.4 回载完成后补空洞：`_commit_hicache_local_restore_to_req`（`decode_hicache_mixin.py:301`）

```python
self.tree_cache.dec_lock_ref(prefix_match.last_device_node)      # 释放匹配时加的锁
self.tree_cache.req_to_token_pool.write(
    (decode_req.req.kv.req_pool_idx,
     slice(prefix_match.l1_prefix_len, prefix_match.decode_prefix_len)),   # ★ 正是那个空洞
    decode_req.hicache_restored_kv_indices)
decode_req.req.prefix_indices = torch.cat(
    [prefix_match.prefix_indices, decode_req.hicache_restored_kv_indices])
decode_req.req.last_node = decode_req.hicache_restored_node
```

三件事一次做完：填空洞、把 `prefix_indices` 拼成完整的 `[0, decode_prefix_len)`、
把 `last_node` 推进到回载后的更深节点（后续 `cache_unfinished_req` 从这里继续插入）。

---

## 6. 阶段三：TransferQueue —— 本地回载状态机

这是 `DecodeHiCacheTransferMixin`（`decode_hicache_mixin.py:177`）的地盘。

### 6.1 状态机定义

```python
class HiCacheRestoreResult(Enum):
    PENDING   # 回载还没完成
    READY     # 回载完成（或不需要回载），可以放行
    FAILED    # 回载失败，请求必须 abort
```

`DecodeRequest` 上的相关字段（`decode.py:286-410` 区间）：
`prefix_match` / `hicache_restored_kv_indices` / `hicache_restored_node` /
`hicache_load_consumer_index = -1` / `hicache_restore_status = PENDING`。

### 6.2 门控 poll：`HiCacheRestoreGatedKVReceiver`（`decode_hicache_mixin.py:161`）

```python
def poll(self) -> KVPoll:
    poll = self._inner.poll()
    if poll == KVPoll.Success and self._decode_req.hicache_restore_status is PENDING:
        return KVPoll.Transferring       # ★ 把 Success 降级
    return poll
```

一句话：**RDMA 完成 ≠ 请求就绪**。只有本地回载也完成了，才允许 `Success` 冒出来。
`_poll_with_metadata_gate`（`decode.py:2213`）在 `enable_decode_hicache` 时把每个
receiver 包一层这个 wrapper。

### 6.3 单请求推进：`_try_hicache_queue_load_back`（`decode_hicache_mixin.py:190`）

流程（有 L3 命中时先等预取）：

1. `check_prefetch_progress(rid)` + `pop_prefetch_loaded_tokens(rid)`
   —— 等 L3→L2 的预取落地，拿到真实到达的 token 数；
2. **重新匹配**一次（预取把数据放进了 host pool，树结构变了，必须 re-match）；
3. `init_load_back(InitLoadBackParams(best_match_node, host_hit_length, req))`
   → 返回 `(new_indices, restored_node)`，这一步做的是 H→D 分配 + 提交传输；
4. **覆盖度检查**（关键的正确性防线）：

```python
if len(rematch.device_indices) + len(new_indices) < pm.decode_prefix_len:
    dr.hicache_restore_status = HiCacheRestoreResult.FAILED
    return False
```

   即：**实际能凑出来的长度必须 ≥ 之前承诺给 P 的 `decode_prefix_len`**，
   否则宁可失败也不能放行（P 已经不传这段了）。

5. 成功则：

```python
dr.hicache_restored_kv_indices = torch.cat(
    [rematch.device_indices[pm.l1_prefix_len:], new_indices])
dr.hicache_restored_node = restored_node
self.tree_cache.inc_lock_ref(restored_node)
if len(new_indices) == 0:
    dr.hicache_restore_status = HiCacheRestoreResult.READY   # 全靠 re-match 就够了，无需等传输
    return False
```

`init_load_back`（`unified_radix_cache.py:2740`）内部条件：

```python
if (tree_core.is_full_device_evicted(best_match_node_id)
        or params.host_hit_length > 0
        or req.swa_host_hit_length > 0 or req.mamba_host_hit_length > 0):
    if self.load_back(best_match_node_id, mem_quota, req=req):
        new_indices = tree_core.collect_full_device_indices(best_match_node_id, req.last_node)
        ...
```

### 6.4 批量推进：`_process_hicache_local_restores`（`decode_hicache_mixin.py:249`）

三个阶段，每个 scheduler step 走一遍：

| 阶段 | 动作 | 目的 |
|---|---|---|
| A | 对已入队的请求 `is_load_back_event_done(dr.hicache_load_consumer_index)` | 收割已完成的 H→D 传输 |
| B | 槽位守卫：`is_load_back_event_done((counter.producer_index + 1) % counter.num_counters)` 通过后才允许新请求入队 | `layer_done_counter` 是**环形**的，producer 不能追尾 consumer |
| C | `consumer_index = ready_to_load_host_cache()`；`< 0` 表示无待办 ⇒ 本轮全部标 READY | 真正触发 `cache_controller.start_loading()` |

`is_load_back_event_done`（`unified_radix_cache.py:2876`）：

```python
if consumer_index < 0 or self.cache_controller is None:
    return True
finish_event = self.cache_controller.layer_done_counter.events[consumer_index]
...
```

`ready_to_load_host_cache`（`unified_radix_cache.py:2868`）就是
`cache_controller.start_loading()`（或 linker 的 `start_layer_wise_loading()`）。

**阶段 B 的环形槽位守卫是这套机制的并发上限**：同一时刻在飞的回载事件数受
`layer_done_counter.num_counters` 限制，超了就得等。这是 D 侧回载吞吐的一个硬约束。（**推断**）

### 6.5 事件驱动的入口：`process_decode_queue`（`decode.py:2667`）

```python
if self.enable_decode_hicache:
    self.tree_cache.check_hicache_events()
```

放在每次迭代的**最前面**。`check_hicache_events`（`unified_radix_cache.py:2787`）负责：

- `_drain_async_work()` 回收上一轮 PP 同步的 send；
- `pp_size != 1` 时用 `ReduceOp.MIN` 跨 PP rank 同步 write/load 完成计数（保证各 rank 步调一致）；
- `writing_check(...)` / `loading_check(...)` 推进树侧簿记；
- `drain_storage_control_queues()` 处理 L3 的 storage_hit / ack_prefetch / backup / release 四类队列；
- 上报 storage metrics。

### 6.6 `pop_transferred`（`decode.py:2245`）

顺序很重要：

```python
if self.enable_decode_hicache:
    self._process_hicache_local_restores(...)     # ① 先推进本地回载
polls = self._poll_with_metadata_gate()           # ② 再 poll（带门控）
for ...:
    if poll == KVPoll.Failed or restore == FAILED:
        self._clean_hicache_prefetch_resources(dr)
        <abort>
        release_kv_cache(..., is_insert=False)    # ★ is_insert=False：失败的请求不进树
    elif poll == KVPoll.Success and restore == PENDING:
        continue                                  # 还得等
```

失败路径上 `is_insert=False` 是必须的：半成品前缀绝不能污染 radix 树。

---

## 7. 阶段四：commit → PREBUILT batch → L1 插入 → 写回

### 7.1 `_commit_transfer_to_req`（`decode.py:2059`）

```python
self._commit_hicache_local_restore_to_req(decode_req)     # 先补空洞（§5.4）
<把 metadata buffer 里 P 侧采样出的第一个 token 写进 req.output_ids>
req.already_computed = req.cached_tokens                  # ★ 防止重复计数
```

最后那行：P 侧自己也有前缀命中（`cached_tokens`），D 侧再统计一次会把
cache hit rate 算重，所以用 `already_computed` 做基线扣减。

### 7.2 PREBUILT batch（`decode.py:2592` `get_new_prebuilt_batch`）

```python
if <decode radix cache enabled>:
    tree_cache = self.tree_cache if req.last_node is None else None
    req.init_next_round_input(tree_cache)
req.set_extend_range(len(req.prefix_indices), req.kv.kv_committed_len)
batch.prepare_for_prebuilt()
batch.process_prebuilt(self.future_map)
```

`tree_cache = ... if req.last_node is None else None` 这个条件的含义：
**如果 prealloc 阶段已经匹配并锁定过 `last_node`，就不要再匹配一次**（会重复加锁 + 语义错位）；
只有从未匹配过的请求才在这里补一次。

`prepare_for_prebuilt` / `process_prebuilt`（`decode_schedule_batch_mixin.py`）：

- 设 `forward_mode = ForwardMode.PREBUILT`；
- `out_cache_loc = req_to_token[pre_len : pre_len + extend_range.length]`；
- `cached_tokens` 记账：`delta = max(0, pre_len - req.already_computed)`；
- **`process_prebuilt` 里对每个 req 调 `maybe_cache_unfinished_req(req, self.tree_cache)`**
  —— 这就是 D 侧的 L1 插入点。

PREBUILT batch **不跑 model forward**（scheduler 里被短路），它只是把
「KV 已经在 device 上、第一个 token 已由 P 给出」这个事实登记进树和 batch 状态。

### 7.3 L1 插入：`cache_unfinished_req`（`unified_radix_cache.py:931`）

要点：

```python
insert_params = InsertParams(prev_prefix_len=req.kv.cache_protected_len, ...)
radix_key = RadixKey(token_ids[:effective_cache_len], ...).page_aligned(self.page_size)
result = self.insert(insert_params)
match_result = self.match_prefix(MatchPrefixParams(key=radix_key, req=req))
assert req.kv.cache_protected_len <= len(new_indices) + self.page_size - 1
self.req_to_token_pool.write(
    (req.kv.req_pool_idx, slice(req.kv.cache_protected_len, len(new_indices))),
    new_indices[req.kv.cache_protected_len:])
self._dec_req_lock(req)
lock_result = self.inc_lock_ref(new_last_node)
req.kv.cache_protected_len = len(new_indices)
req.last_node = new_last_node
```

**insert 后立刻 re-match 并把 `req_to_token` 指向树里的槽位**（去重：如果树里已有
相同前缀，请求会被重定向到树的槽位，自己那份被释放）。`cache_protected_len`
在这里从 prealloc 时设的 `total_prefix_len` 推进到实际插入后的长度。
注意 `prev_prefix_len=req.kv.cache_protected_len` —— **§3.4 里
`req.kv.cache_protected_len = total_prefix_len` 那一行就是为这里服务的**：
它告诉 insert「前面这一段已经在树里/已被保护，别重复处理」。

### 7.4 写回链：L1 → L2 → L3

`insert`（`unified_radix_cache.py:544`）是一个**可恢复的分步 walk**，每一步在 barrier 处
`_apply_cache_actions(step.actions)`。其中的 `BackupKV` action 落到
`_execute_and_commit_kv_backup`（`unified_radix_cache.py:1336`）：

```python
for node_id in action.node_ids:
    device_value, comp_xfers = tree_core.build_backup_spec(node_id)
    host_indices = self._execute_kv_backup(node_id, device_value, comp_xfers, sidecar_xfers)
    if host_indices is None: return 0            # host 不够且驱逐失败 → 放弃备份
    tree_core.commit_backup(node_id, host_indices, comp_xfers)
    if not write_back:
        lock_params = self.inc_lock_ref(node_id).to_dec_params()   # write_through 期间锁住
    self._track_write_through_node(node_id, lock_params)
```

`_execute_kv_backup`（`:1376`）在 host 空间不足时会先 `evict_host(needed)`，
仍不足就返回 `None`（**备份失败是可容忍的降级，不影响请求正确性**）。

D→H 的 ack 回来后走 `_finish_write_through_ack`（`:1426`）：

```python
tree_core.finish_write_through(publish_node_ids, ack_id)
if lock_params is not None: self.dec_lock_ref(lock_node_id, lock_params)
if self.enable_storage:
    for node_id in publish_node_ids:
        self.write_backup_storage(node_id)        # ★ H→L3
```

所以完整写回链是：

```
insert(L1) ──BackupKV──> cache_controller.write (D→H)
                              │ ack
                              ▼
              _finish_write_through_ack ──> write_backup_storage (H→L3)
```

触发时机由 `write_through_threshold` 控制（`write_through` → 1 次命中即备份；
否则 2 次）。**这意味着 D 侧产生的 decode KV（output token 的 KV）也会沿这条链
沉淀到 L2/L3**，供后续同前缀请求复用 —— 这正是
[pd_decode_output_kv_writeback_l3_scheme.md](pd_decode_output_kv_writeback_l3_scheme.md)
讨论的主题。

---

## 8. 准入 / 驱逐 / 失败 路径汇总

### 8.1 三个额度概念别混

| 名称 | 位置 | 含义 |
|---|---|---|
| `_radix_full_evictable()` | `decode.py:441` | 树里可被驱逐的 token 数，**算作可用容量** |
| `_radix_full_available()` | `decode.py:451` | 分配器空闲 + 可驱逐，`_pre_alloc` 判断是否需要驱逐用它 |
| `hicache_reserved_tokens` | `decode.py:1620` 参数 | 已承诺给 P、待回载的窗口，**从可用容量里扣掉** |

### 8.2 驱逐时机

`_pre_alloc` 里：`if self._radix_full_available() < required_alloc_tokens:`
→ `tree_cache.evict_for_alloc(EvictParams(num_tokens=...))`
（`unified_radix_cache.py:566`，注意它接受的是「缺口」而非「绝对配额」，
组件间驱逐可能级联，所以先把缺口翻译成 `available_size_targets`）。

被 `inc_lock_ref` 锁住的节点不会被驱逐 —— 这是 §3.1 匹配后立刻加锁的意义：
**从匹配到 commit 这段窗口内，命中的前缀必须不被驱逐**。
锁在 `_commit_hicache_local_restore_to_req`（`dec_lock_ref(last_device_node)`）
和 `cache_unfinished_req`（`_dec_req_lock(req)`）里成对释放。

### 8.3 失败与中止

| 场景 | 处理 |
|---|---|
| L3 查询/预取抛异常 | warning + `l3_storage_hit_length = 0`，退化到 L2（`decode_hicache_mixin.py:103`） |
| 回载覆盖度不足 | `FAILED` → abort 请求 → `release_kv_cache(is_insert=False)` |
| RDMA `KVPoll.Failed` | `_clean_hicache_prefetch_resources` + abort + `release_kv_cache(is_insert=False)` |
| host pool 不足，备份失败 | `_execute_kv_backup` 返回 `None`，跳过备份，请求正常继续 |
| 请求被 abort | `release_aborted_request` 清理 ongoing prefetch |
| retract / rebootstrap | `use_decode_radix_cache = False`，不走前缀复用 |

`release_kv_cache`（`mem_cache/common.py:122-281`）→
`cache_finished_req(req, is_insert=..., kv_len_to_handle=effective_kv_committed_len)`
→ `_release_overallocated_kv_indices` → `req_to_token_pool.free(req)`。

---

## 9. 兼容性矩阵（Decode 侧 radix cache / HiCache）

| 特性 | D-radix | D-HiCache | 依据 |
|---|---|---|---|
| 常规全注意力（dense/MoE） | ✅ | ✅ | 无门禁 |
| SWA | ⚠️ 需 opt-in 且仅 device | ❌ `ValueError` | `kv_cache_builder.py:253-268` |
| SWA-compress | ❌ | ❌ | `:271` |
| Mamba / SSM 混合 | ❌ | ❌ | `:276` |
| DSV4 / DSA | ❌ | ❌ | `:281`，另见 [pd_decode_dsv4_radix_cache_support_scheme.md](pd_decode_dsv4_radix_cache_support_scheme.md) |
| Speculative（MTP / Eagle / NGRAM 全部） | ❌ | ❌ | `pd_disaggregation_hook.py:89-94`，展开见 §9.1 |
| `transfer_backend = fake` | ❌ | ❌ | 同上 |
| `--enable-hisparse` | ❌ | ❌ | 同上 |
| DCP（decode context parallel） | ❌ | ❌ | 同上 |
| DP attention | ⚠️ EXPERIMENTAL | ⚠️ | 同上，仅警告 |
| retract 备份用 host_pool | ❌（退回 `cpu_tensor`） | — | `kv_cache_builder.py:173` |
| `hicache_host_memory_mode=buffer_only` | ✅ | ✅（FULL 组件） | `unified_radix_cache.py:389-403` |

### 9.1 为什么与投机解码（MTP / Eagle）硬互斥

这是实践中最容易踩、也最容易误解的一条，单独展开。

#### 9.1.1 先说事实：代码里没给理由

```python
# arg_groups/pd_disaggregation_hook.py:89-94
if cfg.speculative_algorithm is not None:
    raise ValueError(
        "--disaggregation-decode-enable-radix-cache is incompatible "
        "with speculative decoding "
        f"(--speculative-algorithm {cfg.speculative_algorithm})"
    )
```

一条**裸的 `raise ValueError`，没有注释**。追 blame：这条门禁跟着功能本身一起进来
（PR #19746 `[P/D disagg] - support decode side radix cache`，2026-05-01），
**commit body 是空的**。所以它的性质是「没实现 / 没验证过，先拦住」，
而**不是**「原理上不可能」。

两个旁证：

1. **一刀切，连 NGRAM 都拦。** 判定条件是 `speculative_algorithm is not None`，
   但 NGRAM 根本没有 draft KV 池（`kv_cache_builder.py:74`：
   `if draft_worker is None or spec_algorithm.is_ngram(): return None`）。
   如果这是针对 draft KV 做的精确判断，NGRAM 不该被拦。
2. **同一个 hook 里的措辞对比**：DP attention 只给 EXPERIMENTAL 警告不拒绝
   （`:96-100`），DCP 用的是 "PD decode DCP **currently** requires chunk cache"（`:73`）。
   spec 这条是硬 raise —— 作者认为「一定会错」，但没写清错在哪。

#### 9.1.2 先排掉一个常见误解：不是因为 draft KV 没法复用

- draft KV 走**同一次 RDMA**、与 target 池**共用同一份 page indices**：
  `prefill.py:243-254` 的原注释是
  "We should also transfer draft model kv cache. **The indices are always shared
  with a target model.**"（draft 池只是往 `kv_data_ptrs` 后面追加几个 buffer entry）。
  所以 `decode_prefix_len` 前移时，target 和 draft 是**同步**跳过的，语义自洽。
- HiCache **已经**有 draft sidecar：
  `mem_cache/hybrid_cache/hybrid_pool_assembler.py:1146` `build_hicache_draft_sidecars`
  （注释 "Build draft KV/DSA sidecars whose indices follow target full KV"），
  外加 `kv_cache_builder.py:89` `maybe_register_hicache_draft` / `HiCacheDraftMode`。

结论：**mix 模式下 radix + HiCache + MTP 是打通的**。
问题只出在 PD-decode 那套预分配 / 传输契约上。

#### 9.1.3 真正需要解决的三处错配（以下为**推断**，非代码明示）

**A. bigram 键的单位错配 —— 最硬的一条**

eagle / MTP 会把 radix 树切成 **bigram 视图**，这条链路是明确的：

```
kv_cache_builder.py:302        is_eagle=spec_algorithm.is_eagle()
  → unified_cache/unified_tree_core.py:399
        self.is_eagle = params.is_eagle and ComponentType.MAMBA not in components
  → radix_cache.py:99-103      def __len__:  return n - 1 if is_bigram else n
  → radix_cache.py:156-167     maybe_to_bigram_view:  value = value[: len(self)]
```

于是匹配返回的 `device_indices` 长度是 **bigram 数（= token 数 − 1）**。
而 D 侧把它**直接当 token 下标**用：

```python
# decode_hicache_mixin.py:33-38
l1_prefix_len     = len(self.prefix_indices)          # ← 实际是 bigram 计数
decode_prefix_len = l1_prefix_len + l2_host_hit_length + l3_storage_hit_length
```

这个值一路传到 P 侧，被当作 `origin_input_ids` 的切片起点：

```python
# prefill.py:373
req.start_send_idx      = decode_prefix_len
num_kv_indices_to_send  = len(req.origin_input_ids) - decode_prefix_len
```

**两套单位（bigram 计数 vs token 位置）在同一个跨进程协议字段里混用。**
ChunkCache 下 `decode_prefix_len` 恒为 0，这个错配永远不暴露；
换成真树立刻暴露，而且是**跨机错位**，比单机内部错位难查得多。

**B. `_pre_alloc_fill_len` 按「一步一 token」写的**

```python
# decode.py:753
return len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
```

没有 `speculative_num_draft_tokens` 的额度。配合 PREBUILT 里的断言：

```python
# decode_schedule_batch_mixin.py:58-63
seq_len = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
if len(req.output_ids) == 0:
    assert seq_len - pre_len == req.extend_range.length
```

radix 命中让 `pre_len > 0`，spec 又让 `seq_lens` 的推进变成「一步多 token」，
两个变量同时动 —— 这套记账没人对齐过。

**C. `process_prebuilt` 里「搬 req_to_token」与「读 req_to_token」的顺序耦合**

```python
# decode_schedule_batch_mixin.py:121, 145
for req in self.reqs:
    maybe_cache_unfinished_req(req, self.tree_cache)               # ① 可能重写 req_to_token
...
spec_info = self.spec_algorithm.build_disagg_draft_input(...)      # ② 读 req_to_token
```

- ① 在**真树**下会 insert + re-match + **重写 `req_to_token` 并释放请求自己那份槽位**
  （`unified_radix_cache.py:1011-1014`）；ChunkCache 下这基本是空操作
  （`chunk_cache.py:89`），`req_to_token` 不会被搬动。
- ② `speculative/eagle_disaggregation.py:66-86` 要读 `req_to_token`，
  把 P 发来的 request-relative DSA topk 位置映射成 decode 本地物理槽位，
  并用 `local_slots <= 0` 判无效。

再叠上 HiCache 的**回载空洞**（`[l1_prefix_len, decode_prefix_len)` 在
`_commit_hicache_local_restore_to_req` 之前是未写入的，见 §5.1），
这个映射会读到 0 → 判为 invalid → `eagle_disaggregation.py:87-88` 直接把
`dsa_topk_indices` 整个置 `None`。

> **这是静默的精度退化**：不抛异常、不掉吞吐指标，只是投机接受率变差。
> 比直接崩掉难发现得多 —— 大概也是作者宁愿在参数层一刀切拒绝的原因。（**推断**）

#### 9.1.4 如果要放开，需要做什么

| 项 | 工作 |
|---|---|
| 单位统一 | 给 `DecodePrefixMatch` 明确「bigram 计数 → token 位置」的换算，或在 D 侧对 eagle 树强制用 unigram 键 |
| 协议校验 | P 侧对 `decode_prefix_len` 加边界/一致性校验（顺带修 §10 风险 #2） |
| 预分配额度 | `_pre_alloc_fill_len` / `_required_alloc_tokens` 纳入 `speculative_num_draft_tokens` |
| 时序 | 明确 `maybe_cache_unfinished_req` 与 `build_disagg_draft_input` 对 `req_to_token` 的先后契约；HiCache 空洞必须在 ② 之前填完 |
| 测试 | 补一条 PD + decode-radix + MTP 的 e2e 精度用例（当前 `test_disaggregation_decode_radix_cache.py` 无 spec 覆盖） |


---

## 10. 已识别的风险与欠账

| # | 问题 | 位置 | 影响 |
|---|---|---|---|
| 1 | `last_loc` 取自索引 `prefix_len-1`，而 `prefix_lens` 传 `total_prefix_len` | `decode.py:1936` | 仅因命中长度必然 page 对齐才安全；隐式契约，无 assert 保护 |
| 2 | P 侧不校验 `decode_prefix_len` | `prefill.py:350-409` | D 侧回滚不彻底会静默数据损坏 |
| 3 | `query_storage_hit_length` 在调度主循环里**同步**查 L3 | `unified_radix_cache.py:1642` | 慢 backend 直接拖慢 D 侧调度 |
| 4 | `suffix_tokens = origin_input_ids[matched_len:]` 无上限 clamp | `decode_hicache_mixin.py:61` | 非 SWA 无害，超长 prompt 下查询 key 较大 |
| 5 | `hasattr` / `getattr` 防御式取值 | `decode.py:405`, `decode.py:456` | 违反 `no-getattr-defensive.md`，掩盖改名 bug |
| 6 | `DecodePrefixMatch` 用 `@dataclass` | `decode_hicache_mixin.py:23` | 违反 `no-dataclasses.md` |
| 7 | SWA 门禁依赖已废弃的 `SGLANG_ENABLE_UNIFIED_RADIX_TREE` | `kv_cache_builder.py:257` + `environ.py:1765` | 死逻辑，需清理 |
| 8 | `layer_done_counter` 环形槽位限制在飞回载数 | `decode_hicache_mixin.py:249` 阶段 B | 高并发下回载吞吐上限，无显式指标暴露 |

---

## 11. 调优与排查建议

**先确认开关真的生效**：启动日志里必须同时看到
`"EXPERIMENTAL: Radix cache is enabled for decode server"`，
且没有 `"KV cache is forced as chunk cache for decode server"`。

**命中率不涨的常见原因（按概率排序）**：

1. 后缀短于 `prefetch_threshold`（默认 256）⇒ L3 从不被查询。调
   `--hicache-storage-backend-extra-config` 里的 `prefetch_threshold`。
2. `page_size` 较大 ⇒ L3 命中被向下对齐吃掉一大截。
3. `last_host_node` 未 `is_backuped` ⇒ L3 探测被跳过。说明 D→H 备份还没跟上，
   看 `write_through_threshold`（改成 `--hicache-write-policy write_through` 让备份更激进）。
4. 开了 speculative decoding ⇒ 参数层就被拒了，根本没建树。

**回载失败率高**：看 `hicache_reserved_tokens` 是否把可用容量压得太紧
（`_allocatable_token_budgets` 里它是最后一项扣减）；`--hicache-ratio` 太小会让
`_execute_kv_backup` 频繁 `evict_host` 甚至返回 `None`，导致该备份的没备份、
下次也就命不中。

**D 侧调度变慢**：优先怀疑第 3 号风险（同步 L3 查询）。可以先把
`--hicache-storage-backend` 摘掉只留 L2 做 A/B。

---

## 12. 交叉引用

- [unified_radix_cache_architecture.md](../03_cache_memory/unified_radix_cache_architecture.md) —— `UnifiedRadixCache` 的组件模型与 insert/evict walk
- [hicache_usage_and_design.md](../03_cache_memory/hicache_usage_and_design.md) —— HiCache 三级结构与 cache controller
- [pd_decode_swa_hicache_support_scheme.md](pd_decode_swa_hicache_support_scheme.md) —— SWA 模型的对照方案（本文纠正了其中 `HiRadixCache` 的描述）
- [pd_decode_dsv4_radix_cache_support_scheme.md](pd_decode_dsv4_radix_cache_support_scheme.md) / [pd_decode_dsv4_hicache_reuse_scheme.md](pd_decode_dsv4_hicache_reuse_scheme.md) —— DSV4/DSA 的独立路径
- [pd_decode_output_kv_writeback_l3_scheme.md](pd_decode_output_kv_writeback_l3_scheme.md) —— D 侧输出 KV 写回 L3
- [decode_instance_prefill_capability.md](decode_instance_prefill_capability.md) —— 为什么 D 实例永不跑 prefill forward
- [pd_disaggregation_kv_transfer_architecture.md](pd_disaggregation_kv_transfer_architecture.md) —— `send_metadata` / RDMA 传输细节
- [pd_true_retraction_rebootstrap_architecture.md](pd_true_retraction_rebootstrap_architecture.md) —— rebootstrap 为何绕开前缀复用
- [host_cache_off_tree_staging_scheme.md](../03_cache_memory/host_cache_off_tree_staging_scheme.md) —— `buffer_only` 模式
