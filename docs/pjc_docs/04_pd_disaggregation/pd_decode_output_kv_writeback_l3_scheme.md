# PD 分离：D 实例把输出 token 的 KV 写出到 L3，P 实例下一轮复用

> 目标场景：多轮对话。第 N 轮 D 实例产出 `output_ids`，把这些 token 的 KV 写到 L3
> storage；第 N+1 轮用户把上一轮输出拼进 prompt，P 实例直接从 L3 读回，不再重算
> prefill。
>
> **先说结论：这个功能在 main 上已经有两条完整实现，不需要从零写。**
> 真正的工作量在「键空间对齐 / 可见性时序 / 背压 / 覆盖面」这四类难点上。

---

## 0. 结论速览

| 现状 | 路径 A：Decode Offload Manager | 路径 B：D 侧 HiRadixCache |
| --- | --- | --- |
| 开关 | `--disaggregation-decode-enable-offload-kvcache` | `--disaggregation-decode-enable-radix-cache` + `--enable-hierarchical-cache` |
| D 侧前缀树 | 无（chunk cache） | 有（`HiRadixCache`，L1/L2/L3 三层） |
| 写出内容 | **只写增量输出段**（prompt 段由 P 写） | 整条 prefix（prompt + output）都会再写一遍 |
| 写出粒度 | 每 `offload_stride` 个 token 流式写 | 请求结束时一次性写（write_through） |
| 省 P 的 prefill 计算 | ✅ | ✅ |
| 省 P→D 的 RDMA 传输 | ❌ | ✅（`decode_prefix_len` 增量传输） |
| 支持的 KV 池 | 仅 MHA / MLA | 不支持 hybrid SWA / SSM |
| 与投机解码共存 | ✅（有 spec v2 单测） | ❌（显式不兼容） |
| 现有 E2E 测试 | `test_disaggregation_decode_offload.py` | `TestDisaggregationDecodeRadixHiCacheFileBackend` |

**要"D 写出输出 token 的 cache，P 下一轮少算 prefill"这一件事，路径 A 就是为它设计的**
（注释 [decode_kvcache_offload_manager.py:142](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L142)
写得很直白：`Prefill side offloads page-aligned origin_input_ids, decode side
offloads the incremental part`）。路径 B 是更重的方案，顺带还能省 P→D 传输。

---

## 1. 数据流全景

```
  第 N 轮
  ┌──────────────── P 实例 ────────────────┐        ┌──────────── D 实例 ────────────┐
  │ prefill(prompt)                        │  RDMA  │ decode → output_ids            │
  │ cache_finished_req                     │ ═════> │                                │
  │   └ insert → write_backup(L1→L2)       │  KV    │ 每 stride 个 token:            │
  │       └ write_backup_storage(L2→L3) ───┼──┐     │  offload_kv_cache()            │
  │   ※ 只写到 len(origin_input_ids) 页对齐 │  │     │   ├ write()      L1→L2 (D2H)   │
  └────────────────────────────────────────┘  │     │   └ write_storage() L2→L3 ──┐  │
                                              │     └────────────────────────────┼──┘
                                        ┌─────▼─────────────────────────────────▼────┐
                                        │  L3 storage（file / mooncake / hf3fs / …）  │
                                        │  key = SHA256 链(page tokens) + 后缀        │
                                        │  ├ page[0..k]   ← P 写（prompt 段）         │
                                        │  └ page[k+1..n] ← D 写（输出段）★本需求      │
                                        └─────┬───────────────────────────────────────┘
  第 N+1 轮                                   │ prefetch
  ┌──────────────── P 实例 ──────────────────▼─┐
  │ _prefetch_kvcache → prefetch_from_storage  │  prompt = 上轮 prompt+output+新后缀
  │ L3 → L2(host node) → init_load_back → L1   │  → extend_len 只剩新后缀，prefill 计算省掉
  └────────────────────────────────────────────┘
```

关键一点：**P 和 D 写的是同一个 hash 链的不同区段**，靠 token 内容做 key，天然接得上；
接不上的地方全部集中在「key 后缀」和「blob 布局」上（见第 4 节难点 1~3）。

---

## 2. 路径 A：Decode Offload Manager（现成实现，逐行）

### 2.1 启动与装配

| 环节 | 位置 |
| --- | --- |
| 参数定义 | [server_args.py:2968](../../../python/sglang/srt/server_args.py#L2968) |
| 参数校验（必须 decode 模式 + 必须配 storage backend） | [server_args.py:7456-7464](../../../python/sglang/srt/server_args.py#L7456) |
| 构造（仅 decode 实例） | [scheduler.py:524-540](../../../python/sglang/srt/managers/scheduler.py#L524) |
| 自建 host 池 + 自建 HiCacheController | [decode_kvcache_offload_manager.py:57-105](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L57) |
| 写出步长 `offload_stride`（默认 = page_size） | [decode_kvcache_offload_manager.py:50-56](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L50)，env `SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE`（[environ.py:482](../../../python/sglang/srt/environ.py#L482)） |

注意它**自己 new 了一个 host KV 池**（`hicache_ratio` / `hicache_size` 再吃一份 pinned
内存）和一个独立 `HiCacheController`，与 tree cache 用的那套完全分开。

### 2.2 触发点

```python
# batch_result_processor.py:941-945   —— 每个 decode step，请求还没结束时的流式写出
if self.server_args.disaggregation_decode_enable_offload_kvcache and not req.finished():
    self.decode_offload_manager.offload_kv_cache(req)

# batch_result_processor.py:962-965   —— 请求结束时最后一次；没有可写的增量就直接释放
if self.server_args.disaggregation_decode_enable_offload_kvcache:
    if not self.decode_offload_manager.offload_kv_cache(req):
        self.decode_offload_manager.finalize_release_on_finish(req)
else:
    ...
    release_kv_cache(req, self.tree_cache, is_insert=is_insert)   # :979
```

**这里是互斥的**（[batch_result_processor.py:962-979](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L962)）：
开了 offload manager 就完全不走 `release_kv_cache` → 不会 insert radix、不会走
HiRadixCache 的 write_through。所以路径 A 和路径 B 不能叠加使用。

进度轮询在 decode 事件循环里：
[decode.py:2251-2252](../../../python/sglang/srt/disaggregation/decode.py#L2251)（`process_decode_queue` 内），
PP 路径另有 [scheduler_pp_mixin.py:482-483](../../../python/sglang/srt/managers/scheduler_pp_mixin.py#L482)。

### 2.3 核心逻辑 `offload_kv_cache`（:129-203）

```python
all_tokens = req.origin_input_ids + req.output_ids[:-1]              # :143  最后一个 token 的 KV 还没落盘
prefill_offloaded_len = len(req.origin_input_ids) // page_size * page_size   # :144  P 已经写过的部分
if state is None:                                                    # :148  第一次：只补算 hash 锚点，不搬数据
    prefill_hashes = self._compute_prefix_hash(req.origin_input_ids[:prefill_offloaded_len])
    state = OffloadedState(prefill_len=..., inc_len=0, last_hash=prefill_hashes[-1])
incremental_new     = len(all_tokens) - state.prefill_len - state.inc_len    # :161-162
incremental_aligned = incremental_new // offload_stride * offload_stride     # :163-165
host_indices = self.cache_controller.write(device_indices=..., node_id=ack_id)  # :185  D2H
```

- **不重复写 prompt**：只从 `prefill_offloaded_len` 之后开始搬，prompt 段仅在 CPU 上重算
  SHA256 用来接 hash 链（[:329-336](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L329)，
  `prior_hash=""` 从 root 起锚，与 P 的链完全一致）。
- D2H ack 后再发 L3：`_check_offload_progress`（:225-256）→ `_trigger_backup`（:316-327）
  → `cache_controller.write_storage(...)`，并把本段最后一个 page hash 存回
  `state.last_hash`（:250），供下一段续链。
- **GPU 槽位延迟释放**：只有 `req.finished()` 且没有 in-flight offload 时才
  `_release_finished_req`（:252-255 → :258-300），里面依次释放 prompt 段、增量段、
  spec v2 的 over-alloc 段，并 `tree_cache.protected_size_ -=`（:298）。
  历史上曾在 decode 中途就释放，导致槽位被并发 admission 复用、读到别人的 KV
  （注释 [:176-180](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L176) / [:268-272](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L268)）。
- TP 对齐用 `all_reduce(MIN)` 取各 rank 队列长度的最小值（:209-219），再
  `ack.finish_event.synchronize()`（:229）——**这两步在 decode 主循环里，是 SLO 敏感点**。
- MLA 模型只有 rank 0 真正写 L3（`backup_skip`，[cache_controller.py:464-469](../../../python/sglang/srt/managers/cache_controller.py#L464)），
  但 ack 照样入队（[:1222-1224](../../../python/sglang/srt/managers/cache_controller.py#L1222)），不会卡住。

### 2.4 已有 E2E 测试

[test/registered/disaggregation/test_disaggregation_decode_offload.py](../../../test/registered/disaggregation/test_disaggregation_decode_offload.py)：
P 用 `--enable-hierarchical-cache --hicache-storage-backend file`，D 用
`--disaggregation-decode-enable-offload-kvcache --hicache-storage-backend file`，
`page-size 16` / `stride 16`，跑两轮 MMLU（中间重启清内存缓存）比对分数。

---

## 3. 路径 B：D 侧完整 HiRadixCache（更重，但顺带省传输）

### 3.1 装配

- `--disaggregation-decode-enable-radix-cache` 让 hook 放开 D 的 radix：
  [arg_groups/pd_disaggregation_hook.py:29-58](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L29)
  （不开的话 `disable_radix_cache = True`，"KV cache is forced as chunk cache for decode server"）。
- 再叠 `--enable-hierarchical-cache` → [registry.py:117-129](../../../python/sglang/srt/mem_cache/registry.py#L117) 建 `HiRadixCache`。
- 注意 `--enable-hierarchical-cache` 与 `--disable-radix-cache` 互斥
  （[server_args.py:7450-7454](../../../python/sglang/srt/server_args.py#L7450)），所以
  **D 上想开 hicache 就必须同时开 decode radix cache**，两个开关是绑死的。

### 3.2 写路径（含输出 token）

| 步骤 | 位置 |
| --- | --- |
| prompt 段 insert（PREBUILT 之后） | [decode.py:2242-2243](../../../python/sglang/srt/disaggregation/decode.py#L2242) → [decode_schedule_batch_mixin.py:121](../../../python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py#L121) |
| 结束时 insert `prompt+output` | [batch_result_processor.py:979](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L979) → [common.py:132-153](../../../python/sglang/srt/mem_cache/common.py#L132) → [radix_cache.py:455](../../../python/sglang/srt/mem_cache/radix_cache.py#L455) `(origin_input_ids + output_ids)[:kv_len_to_handle]` |
| L1→L2 | [hiradix_cache.py:973-982](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L973) `_inc_hit_count` → [:836-866](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L836) `write_backup` |
| L2→L3 | [:984-1016](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L984) `writing_check` → [:900-910](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L900) `_finish_write_through_ack` → [:912-937](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L912) `write_backup_storage` |
| 事件驱动 | [decode.py:2248-2249](../../../python/sglang/srt/disaggregation/decode.py#L2248) `check_hicache_events` |

`write_through_threshold`（[hiradix_cache.py:200-203](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L200)）：
`write_through`=1（默认，insert 即写）/ `write_through_selective`=2 / `write_back`=只在
逐出时写（[:1183](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1183)）。

### 3.3 读路径 + 省传输

D 侧读是 [decode_hicache_mixin.py](../../../python/sglang/srt/disaggregation/decode_hicache_mixin.py) 那一整套
（`query_storage_hit_length` :84 / `prefetch_from_storage` :126 / `init_load_back` :207 /
commit :294-311），命中长度打包成 `decode_prefix_len` 发给 P，P 在
[prefill.py:321-338](../../../python/sglang/srt/disaggregation/prefill.py#L321) `finalize_bootstrap` 里
`req.start_send_idx = decode_prefix_len`，直接少传这一段 KV。
**这是路径 A 拿不到的额外收益。**

### 3.4 限制（写进代码里的硬约束）

- hybrid SWA / Mamba SSM 模型直接 `raise`：[kv_cache_builder.py:188-201](../../../python/sglang/srt/mem_cache/kv_cache_builder.py#L188)
- 与投机解码 / `--enable-hisparse` / `fake` 传输后端不兼容（参数说明
  [server_args.py:2963-2967](../../../python/sglang/srt/server_args.py#L2963)）
- DSV4 NPU 不支持 decode 侧前缀缓存：[decode.py:1052-1061](../../../python/sglang/srt/disaggregation/decode.py#L1052) / [:1182](../../../python/sglang/srt/disaggregation/decode.py#L1182)
- HiSparse 要求 `prefix_len == 0`：[decode.py:1503](../../../python/sglang/srt/disaggregation/decode.py#L1503)
- 开了 hierarchical cache 会静默关掉 optimistic prefill：[server_args.py:7734-7743](../../../python/sglang/srt/server_args.py#L7734)

---

## 4. 难点清单（按危险程度排序）

### 难点 1（最隐蔽）：key 空间必须逐字节对齐，而校验几乎没有

L3 的 key = `SHA256 链(prior_digest ‖ page 内 token id 的 LE-uint32)` + 后端后缀。
hash 本体只吃 token id 和 `is_bigram`
（[cpp_utils/native_hash.py:45-86](../../../python/sglang/srt/mem_cache/cpp_utils/native_hash.py#L45)，
[cpp_utils/hash_binding.cpp:62-75](../../../python/sglang/srt/mem_cache/cpp_utils/hash_binding.cpp#L62)）；
后缀由各后端拼（file 后端：[hicache_storage.py:366-386](../../../python/sglang/srt/mem_cache/hicache_storage.py#L366)
→ `{hash}_{model-name}[_{tp_rank}_{tp_size}][_{pp_size}_{pp_rank}][_cp{r}_{s}].bin`）。

| 差异项 | 在 hash 里？ | 在 key 后缀里？ | P/D 不一致的后果 |
| --- | --- | --- | --- |
| `--page-size` | 隐含（决定分页边界） | 否 | **全 miss**，功能静默失效 |
| `is_bigram`（= `spec_algorithm.is_eagle()`） | **是** | 否 | **全 miss**（见下） |
| `--served-model-name` | 否 | 是（file/nixl/mooncake/eic） | 全 miss |
| `tp_rank/tp_size`（非 MLA） | 否 | 是 | 全 miss |
| `--hicache-mem-layout` | 否 | **仅 eic 有** | **静默读到错乱数据** |
| kv cache dtype（同 itemsize，如 bf16↔fp16） | 否 | 否 | **静默数值错误** |
| layer 数 / head 数 / dtype 改字节数 | 否 | 否 | file 后端 `IOError`（[hicache_storage.py:461-483](../../../python/sglang/srt/mem_cache/hicache_storage.py#L461)） |
| `extra_key`（LoRA id / cache_salt） | **否** | 否 | 跨 LoRA / 跨租户串味 |

整套存储**没有任何版本或配置指纹校验**，`hicache_mem_layout` 和 dtype 不一致是静默数据
损坏，不是报错。

**`is_bigram` 是 PD 场景最容易踩的一个**：它等于 `spec_algorithm.is_eagle()`
（[kv_cache_builder.py:219](../../../python/sglang/srt/mem_cache/kv_cache_builder.py#L219)
→ [hiradix_cache.py:1405/1690](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1405)）。
而路径 A 的 `_compute_prefix_hash` 传的是裸 list，`is_bigram` 恒为 False
（[native_hash.py:54](../../../python/sglang/srt/mem_cache/cpp_utils/native_hash.py#L54) 用 `getattr(..., False)`）。
于是：

- P 不开 spec、D 开 MTP + 路径 A → key 一致，✅ 能用；
- **P 开 spec（EAGLE/MTP）→ P 侧 key 是 bigram，D 用路径 A 写的非 bigram key 永远读不到，静默 0 命中**；
- 路径 B 下 D 开 spec 本身就被禁掉了，所以 B 不会出现这个不一致。

### 难点 2：P/D 并行度不同时，MHA 模型基本没法共享

非 MLA 模型 key 里带 `tp_rank_tp_size`。生产里 P/D 常常是 P=TP4、D=TP8+DP-attn，
这种组合下 MHA 模型的 L3 条目互相看不见。只有 MLA / DSV4 池
（`is_mla_model=True`，[cache_controller.py:580-641](../../../python/sglang/srt/managers/cache_controller.py#L580)）
会把 rank 从 key 里摘掉、只 rank 0 写，才能跨异构并行共享。
异构 TP 的 MHA 只有 `page_head` layout + `tp_lcm_size` 拆头这条路
（[mooncake_store.py:555-568](../../../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L555)）。

→ **选型建议：这个功能优先在 MLA 系模型上落地。**

### 难点 3：可见性时序竞争

D 的 `write_storage` 是异步的，而响应早就流回给用户了。用户下一轮请求可能在 L3 写完之前
就打到 P → miss。现有测试是靠 `time.sleep(1)` 规避的
（[test_disaggregation_decode_radix_cache.py:263](../../../test/registered/disaggregation/test_disaggregation_decode_radix_cache.py#L263)）。
没有任何"cache ready"信号暴露给 router 或客户端。

### 难点 4：L3 里 prompt 段和输出段是两个独立的 key 组，会出空洞

prompt 段由 P 写、输出段由 D 写，两边各自 LRU 逐出。P 下一轮的命中查询是
**顺序 `batch_exists`，遇到第一个 miss 就停**
（[cache_controller.py:1023-1045](../../../python/sglang/srt/managers/cache_controller.py#L1023)）。
一旦 prompt 段被逐出而输出段还在，输出段就成了读不到的孤儿——白写一遍。

### 难点 5：D 侧 SLO 干扰与显存/内存成本

- 每个 decode step 都有 `all_reduce(MIN)` + `finish_event.synchronize()`
  （[:209-229](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L209)）落在主循环上。
- 路径 A 额外开一份 pinned host 池（`hicache_ratio`），与 tree cache 的 host 池不共享。
- D2H 拷贝与 RDMA 收 KV 抢 PCIe。
- GPU 槽位要等 D2H ack 才释放（:252-255），高负载下等价于 KV 池有效容量下降，并与
  retract 逻辑交互（[decode.py:1471](../../../python/sglang/srt/disaggregation/decode.py#L1471) 的 TODO）。
- `cache_controller.py` 里 rate limiting 至今是 TODO。

### 难点 6：覆盖面缺口

- 路径 A 只认 `MHATokenToKVPool` / `MLATokenToKVPool`，其他直接
  `raise ValueError("Unsupported KV cache type for decode offload")`
  （[:60-79](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L60)）。
- 路径 B 不支持 hybrid SWA / SSM。
- 所以 SWA 混合模型、DSV4 六子池两条路都不通 —— 这正是
  [pd_decode_swa_hicache_support_scheme.md](pd_decode_swa_hicache_support_scheme.md) 和
  [pd_decode_dsv4_hicache_reuse_scheme.md](pd_decode_dsv4_hicache_reuse_scheme.md) 两篇文档要解决的问题。

### 难点 7：收益前提 —— 读 L3 必须比重算 prefill 便宜

P 侧命中后走 L3→L2→L1，`wait_complete` 策略会让请求卡在 `waiting_queue` 直到全部取回
（[hiradix_cache.py:1492-1531](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1492)）。
MLA 每 token 约 576B，读回很划算；MHA 大模型每 token 字节数大一到两个数量级，
存储带宽不够时"读比算慢"，TTFT 反而变差。上线前必须按
`bytes_per_token / 存储带宽` vs `prefill FLOPs / 算力` 算一遍。

### 难点 8：尾部 token 必然丢一小段（可接受）

- 路径 A：`output_ids[:-1]`（最后一个 token 的 KV 还没写）+ 增量按 `offload_stride` 向下对齐。
- 路径 B：`kv_len_to_handle = effective_kv_committed_len()` 再按 page 向下对齐。

最多少命中 `page_size-1` 个 token，P 重算即可，不是正确性问题。

### 难点 9：路径 A 不传 `prefix_keys`

[:321-325](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L321) 调
`write_storage` 时没有 `prefix_keys`；若后端配了
`hicache_storage_pass_prefix_keys`（[hiradix_cache.py:733](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L733)），
D 的写和 P 的读携带的 `extra_info` 不一致。

### 难点 10：路径 B 的双写浪费

D 上 write_through 会把整条 prefix（含 P 已经写过的 prompt）再写一遍 L2+L3，而且
`write_backup` 有"父节点必须已 backup"的连续性约束
（[:840-843](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L840)），没法只写输出段。
prompt 越长浪费越大。

---

## 5. 具体方案（分阶段）

### Phase 0：先用现成的路径 A，把配置对齐（不写代码）

选型判据：

| 你的诉求 | 选 |
| --- | --- |
| 只要"P 下一轮少算 prefill" | **路径 A** |
| 还要"P→D 少传 KV" | 路径 B（或 Phase 2 给 A 补 `decode_prefix_len`） |
| 模型是 hybrid SWA / DSV4 六子池 | 两条都不通，见 Phase 3 |
| D 侧开了投机解码 | 只能路径 A（B 显式禁用） |

**P/D 配置一致性 checklist**（任一不一致 → 静默失效或静默损坏，见难点 1）：

| 项 | 必须一致 | 不一致后果 |
| --- | --- | --- |
| `--page-size` | ✅ | 全 miss |
| `--served-model-name` | ✅ | 全 miss |
| `--hicache-storage-backend` + 同一后端端点 | ✅ | 读不到 |
| `--hicache-mem-layout` | ✅ | **静默数据损坏** |
| `--kv-cache-dtype` | ✅ | **静默数值错误** |
| TP size（非 MLA 模型） | ✅ | 全 miss |
| `--speculative-algorithm` 是否 eagle 系 | P 侧必须**不开**（路径 A） | 全 miss |
| LoRA / `cache_salt` | 单一租户 | 串味 |

启动示例（MLA 模型，最推荐的落地组合）：

```bash
# P 实例
python -m sglang.launch_server --model-path <mla-model> \
  --disaggregation-mode prefill --page-size 64 \
  --enable-hierarchical-cache --hicache-write-policy write_through \
  --hicache-storage-backend file --hicache-mem-layout page_first \
  --hicache-storage-prefetch-policy wait_complete --hicache-ratio 2

# D 实例
SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE=64 \
python -m sglang.launch_server --model-path <mla-model> \
  --disaggregation-mode decode --page-size 64 \
  --disaggregation-decode-enable-offload-kvcache \
  --hicache-storage-backend file --hicache-mem-layout page_first \
  --hicache-ratio 2 --num-reserved-decode-tokens 128
```

先跑 [test_disaggregation_decode_offload.py](../../../test/registered/disaggregation/test_disaggregation_decode_offload.py)
拿基线，确认 `cached_tokens` 在第二轮确实覆盖到上一轮输出段——**这一步就能验证功能是否已满足需求**，
如果满足，后面几个 Phase 都是加固而非新功能。

### Phase 1：给路径 A 补四个缺口（真正要写的代码）

#### 1.1 key 加配置指纹（对应难点 1，优先级最高）

问题：`hicache_mem_layout` / kv dtype / `is_bigram` / `extra_key` 都不进 key，不一致时静默错。

做法（改动集中在 `mem_cache/hicache_storage.py` 的 key 后缀拼接 +
`HiCacheStorageConfig`）：

```
key = f"{hash}_{model_name}_{layout}_{kvdtype}_{bigram_flag}[_{tp_rank}_{tp_size}]..."
```

- 把 `hicache_mem_layout`、`kv_cache_dtype`、`is_bigram` 三项加入**所有**后端的后缀
  （现在只有 eic 带 layout，见 [hicache_storage.py:366-386](../../../python/sglang/srt/mem_cache/hicache_storage.py#L366)）。
- 或者更省事的折中：在后端里放一个 `_meta` 记录（模型名/层数/head 数/dtype/layout/page_size），
  启动时读一次做校验，不一致直接 `raise`，把静默损坏变成启动失败。
- `extra_key`（LoRA id / cache_salt）必须进 hash 或后缀，否则跨租户串味。

#### 1.2 让路径 A 的 `is_bigram` 跟随 P（对应难点 2）

`_compute_prefix_hash`（[decode_kvcache_offload_manager.py:329-336](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L329)）
和 `_trigger_backup` 传的 page hash 都是 `is_bigram=False`。若 P 侧开 EAGLE/MTP，
key 永远对不上。两种修法：

1. **保守**：启动校验——D 开 offload 时，若集群内 P 开了 eagle 系算法，直接报错提示不兼容。
2. **彻底**：给 offload manager 传入 P 侧的 `is_bigram`（新增 server arg 或从 bootstrap
   元数据带过来），`_compute_prefix_hash` 与 `write_storage` 都按该值算 hash。

推荐先做 1（一行校验，避免静默 0 命中），2 作为后续。

#### 1.3 可见性与背压（对应难点 3 / 5）

- **可见性**：在响应的 `meta_info` 里加一个 `kv_writeback_pending`（或暴露
  `/flush_kv_writeback` 接口），让 router / 客户端知道"上一轮 cache 还没落盘"。
  最低成本版：暴露 Prometheus gauge `decode_offload_inflight_pages`，运维可观测，
  测试也不必再 `time.sleep(1)`。
- **背压**：`cache_controller` 的 rate limiting 至今是 TODO。在
  `offload_kv_cache`（:129）入口加水位判断——`ack_write_queue` 或 `ack_backup_queue`
  超阈值就本步跳过（返回 False 但不释放槽位），并自适应放大 `offload_stride`。
- **SLO**：把 `_check_offload_progress` 里的 `all_reduce(MIN)` 从"每 step"降为
  "每 N step"（N 可配），`finish_event.synchronize()` 换成 `query()` 非阻塞轮询，
  避免 decode 主循环被 D2H 拖住（[:209-229](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L209)）。

#### 1.4 消除 L3 空洞 + 补 `prefix_keys`（对应难点 4 / 9）

- 空洞：给同一请求的 prompt 段 + 输出段打同一个"组标签"（后端支持的话用同一
  namespace / 同一 TTL），或让 D 在写输出段时顺带 `touch` 一次 prompt 段的 key，
  把两段的 LRU 时间戳拉齐。最简单可行版：D 写之前先 `batch_exists` 探一次 prompt 段，
  不存在就跳过本次写出（不写孤儿）。
- `prefix_keys`：`_trigger_backup`（[:321-325](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L321)）
  调 `write_storage` 时补 `prefix_keys`（D 侧本来就算出了整条 hash 链，取前缀即可），
  与 [hiradix_cache.py:733](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L733) 的行为对齐。

### Phase 2：如果还想省 P→D 传输

两个选择：

| 方案 | 改动量 | 说明 |
| --- | --- | --- |
| 直接换路径 B | 0（只改启动参数） | 但要放弃投机解码、承受双写浪费（难点 10） |
| 给路径 A 补 `decode_prefix_len` | 中 | D 侧现在没有 radix，没法查本地命中；需要在 offload manager 里记录"这个请求的哪些页我已经写到 L3 了"，在下一轮 bootstrap 时把命中长度报给 P（复用 [prefill.py:321-338](../../../python/sglang/srt/disaggregation/prefill.py#L321) 的 `req.start_send_idx` 通路） |

注意路径 B 的双写：真要减小浪费，得放松
`write_backup` 的"父节点必须已 backup"约束（[hiradix_cache.py:840-843](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L840)），
允许"父节点在 L3 已存在（远端 exists 为真）"也算 backuped。这是个侵入性改动，
需要单独评估。

### Phase 3：覆盖 SWA / DSV4

不在本文范围，已有两篇专门文档：

- [pd_decode_swa_hicache_support_scheme.md](pd_decode_swa_hicache_support_scheme.md)
- [pd_decode_dsv4_hicache_reuse_scheme.md](pd_decode_dsv4_hicache_reuse_scheme.md)

要点：这两类模型 KV 不是"一条线性序列 + 单一池"，路径 A 的
`isinstance(kv_cache, MHATokenToKVPool/MLATokenToKVPool)` 断言
（[:60-79](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py#L60)）
和路径 B 的 hybrid 拒绝
（[kv_cache_builder.py:188-201](../../../python/sglang/srt/mem_cache/kv_cache_builder.py#L188)）
都要先解决多子池的 key/blob 组织问题。

### Phase 4：指标与 CI

新增指标（`managers/metrics_collector.py`）：

| 指标 | 类型 | 用途 |
| --- | --- | --- |
| `decode_offload_pages_written_total` | Counter | 写出量 |
| `decode_offload_inflight_pages` | Gauge | 背压/可见性 |
| `decode_offload_skip_total{reason}` | Counter | 跳过原因（水位/无增量/不支持） |
| `decode_offload_d2h_seconds` | Histogram | SLO 归因 |
| P 侧 `hicache_storage_hit_tokens{segment=prompt\|output}` | Counter | **直接量化本需求的收益** |

最后一项最关键：没有它就说不清"输出段到底被复用了多少"。

---

## 6. 验证方法

### 6.1 复用现有 E2E

| 测试 | 验证什么 |
| --- | --- |
| [test_disaggregation_decode_offload.py](../../../test/registered/disaggregation/test_disaggregation_decode_offload.py)::`test_mmlu_double_eval` | 路径 A 端到端：重启清内存缓存后，第二轮 MMLU 分数不掉 → L3 里的输出段确实读回来了 |
| [test_disaggregation_decode_radix_cache.py](../../../test/registered/disaggregation/test_disaggregation_decode_radix_cache.py)::`test_decode_hicache_file_backend_l3_reuses_decode_output_after_flush` | 路径 B：5 轮对话，每轮 flush 掉 P/D 的内存缓存，断言 `cached_tokens >= prev_prompt_len + prev_output_len` —— **这正是本需求的验收断言** |

### 6.2 需要新增的单测

1. **hash 一致性（纯 CPU，最有价值）**：同一串 token，分别按
   "P 侧 HiRadixCache 的算法"和"D 侧 `_compute_prefix_hash` + `write_storage`"
   算 page hash，断言逐字节相等；再对 `is_bigram=True` 断言**不**相等（把难点 2 钉死）。
2. **配置指纹校验**：故意用不同 `hicache_mem_layout` / dtype 启动 P 和 D，
   断言启动即失败（Phase 1.1 落地后）。
3. **背压**：mock 一个慢后端，断言 `offload_kv_cache` 在队列超水位时返回 False
   且不释放 GPU 槽位、不泄漏（检查 `protected_size_` 归零）。

### 6.3 收益量化（上线前必做，对应难点 7）

```
读回成本 = bytes_per_token × hit_tokens / 存储带宽
重算成本 = 2 × params × hit_tokens / 算力       (prefill FLOPs 粗估)
```

MLA 约 576 B/token，几乎必赚；MHA 大模型每 token 字节数高一到两个数量级，
先量一遍存储带宽再上。同时关注 `wait_complete` 带来的排队时间
（[hiradix_cache.py:1492-1531](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1492)），
它直接进 TTFT。

---

## 7. 文件索引

| 主题 | 文件:行 |
| --- | --- |
| 路径 A 全部逻辑 | [disaggregation/decode_kvcache_offload_manager.py](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py)（361 行，全文即本功能） |
| 路径 A 构造 | [managers/scheduler.py:524-540](../../../python/sglang/srt/managers/scheduler.py#L524) |
| 路径 A/B 触发点（互斥） | [scheduler_components/batch_result_processor.py:929-984](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L929) |
| 路径 A 进度轮询 | [disaggregation/decode.py:2251-2252](../../../python/sglang/srt/disaggregation/decode.py#L2251) / [scheduler_pp_mixin.py:482-483](../../../python/sglang/srt/managers/scheduler_pp_mixin.py#L482) |
| D 侧 hicache 读路径（路径 B） | [disaggregation/decode_hicache_mixin.py](../../../python/sglang/srt/disaggregation/decode_hicache_mixin.py)（311 行） |
| `decode_prefix_len` 落到 P | [disaggregation/prefill.py:321-338](../../../python/sglang/srt/disaggregation/prefill.py#L321) |
| 通用 L1→L2→L3 写路径 | [mem_cache/hiradix_cache.py:836-1016](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L836) |
| L3 顺序命中查询（空洞根源） | [managers/cache_controller.py:1023-1045](../../../python/sglang/srt/managers/cache_controller.py#L1023) |
| MLA rank 无关 + backup_skip | [managers/cache_controller.py:464-469](../../../python/sglang/srt/managers/cache_controller.py#L464) / [:580-641](../../../python/sglang/srt/managers/cache_controller.py#L580) |
| page hash 算法 | [mem_cache/cpp_utils/native_hash.py:45-86](../../../python/sglang/srt/mem_cache/cpp_utils/native_hash.py#L45) / [hash_binding.cpp:62-75](../../../python/sglang/srt/mem_cache/cpp_utils/hash_binding.cpp#L62) |
| key 后缀拼接（file 后端） | [mem_cache/hicache_storage.py:366-386](../../../python/sglang/srt/mem_cache/hicache_storage.py#L366) |
| decode radix 开关 hook | [arg_groups/pd_disaggregation_hook.py:29-58](../../../python/sglang/srt/arg_groups/pd_disaggregation_hook.py#L29) |
| 参数定义与校验 | [server_args.py:2963-2972](../../../python/sglang/srt/server_args.py#L2963) / [:7450-7464](../../../python/sglang/srt/server_args.py#L7450) |
| hybrid 拒绝 + `is_eagle` | [mem_cache/kv_cache_builder.py:185-219](../../../python/sglang/srt/mem_cache/kv_cache_builder.py#L185) |
| stride 环境变量 | [environ.py:482](../../../python/sglang/srt/environ.py#L482) |
| 路径 A E2E | [test/registered/disaggregation/test_disaggregation_decode_offload.py](../../../test/registered/disaggregation/test_disaggregation_decode_offload.py) |
| 路径 B E2E（验收断言） | [test/registered/disaggregation/test_disaggregation_decode_radix_cache.py](../../../test/registered/disaggregation/test_disaggregation_decode_radix_cache.py) |

相关文档：
[hicache_usage_and_design.md](../03_cache_memory/hicache_usage_and_design.md)（HiCache 三层与 layout/IO backend）、
[pd_decode_swa_hicache_support_scheme.md](pd_decode_swa_hicache_support_scheme.md)、
[pd_decode_dsv4_hicache_reuse_scheme.md](pd_decode_dsv4_hicache_reuse_scheme.md)。




