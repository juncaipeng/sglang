# PD 分离 Decode 实例支持 SWA 模型 + HiCache 的技术方案

> 撰写日期：2026-07-30 · 基线：`main` @ `cce587351`
> 本文回答两个问题：（1）今天为什么 D 实例开不了 SWA + HiCache；（2）要支持需要改哪些代码、怎么改、风险在哪。
> 文中所有行号均为本次实读代码所得；凡属推断而非代码验证的结论，都在句末显式标注「（推断）」。

---

## 0. 结论速览（先看这一节）

**一句话**：D 实例的 HiCache 必须搭 `--disaggregation-decode-enable-radix-cache`
一起用，而这个开关在 `kv_cache_builder.py` 里被**显式拒绝**用于 SWA 模型；根因不是
HiCache 本身不支持 SWA（聚合场景早就支持了），而是 **PD 的「前缀复用」与 SWA 的
「滑窗 KV 只有最后一个窗口有效 + full→swa 映射表是全局共享的」这三件事在 D 侧对不上账**。

| 维度 | 现状 |
|---|---|
| 聚合（非 PD）场景 SWA + HiCache | **已支持**。走 `UnifiedRadixCache` + `{FULL, SWA}` 双组件 + 双 host 池 |
| PD Prefill 实例 + HiCache | **已支持**（只是会关掉 optimistic prefill） |
| PD Decode 实例 + HiCache（非 SWA 模型） | **已支持**，是 `HiRadixCache` + `decode_hicache_mixin.py` 那套 |
| PD Decode 实例 + HiCache（SWA 模型） | **被硬拒**（ValueError） |
| PD Decode 实例 + SWA（不开 radix/HiCache） | **已支持**，走 `SWAChunkCache` + SWA 尾窗预分配 |

**好消息**：需要的三块基础设施 main 上都已经有了 —— ① 尾窗式 SWA 预分配
`alloc_extend_swa_tail`；② 前缀匹配长度上限机制 `RadixKey.limit` +
`swa_reprefill_tail_tokens()`；③「有 SWA 组件但不建 SWA host 池」的 HiCache 栈形态
（DSV4 unified_kv 已经这么干）。所以方案 A 基本是**把三块现成积木拼起来 + 修 4 个记账 bug**，
不需要新的通信协议、不需要改 P 侧、不需要改 RDMA 传输格式。

---

## 1. 名词与前置约定

| 记号 | 含义 |
|---|---|
| `L` = `fill_len` | 本请求预分配时的 KV 长度，普通新请求 = `len(origin_input_ids)` |
| `W` | 滑窗大小 `sliding_window_size` |
| `window_start` | `page_align_floor(max(0, L - W), page_size)`，滑窗左边界（按页向下对齐） |
| `swa_tail_len` | `L - window_start`，需要 SWA KV 的尾段长度 |
| `P1` = `prefix_len` | L1（显存里已有）的复用前缀长度 |
| `Pt` = `total_prefix_len` | L1+L2+L3 合计的复用前缀长度，也就是告诉 P 侧「这段你别发」的 `decode_prefix_len` |

**SWA 混合模型的显存结构**（`SWAKVPool` + `SWATokenToKVPoolAllocator`）：

```
        full 层 KV 池 (全注意力层)          swa 层 KV 池 (滑窗层)
        ┌───────────────────────┐          ┌──────────────┐
        │ 每个 token 一个 slot  │          │ 只需最后 W 个 │
        └───────────▲───────────┘          └──────▲───────┘
                    │                             │
        full_to_swa_index_mapping[full_slot] = swa_slot     ← 全局一张表！
```

关键点：对外返回的句柄**恒为 full 索引**，SWA 索引靠
`full_to_swa_index_mapping` 这张**按 full slot 下标寻址的全局表**翻译
（[swa.py:82-101](../../../python/sglang/srt/mem_cache/allocator/swa.py#L82-L101)、
`translate_loc_from_full_to_swa` [swa.py:147](../../../python/sglang/srt/mem_cache/allocator/swa.py#L147)）。
**这张表是「一个 full slot 只能映射到一个 swa slot」**——记住这句，第 3 节的根因全从这里来。

---

## 2. 为什么现在不支持：三道 gate 的准确位置

### 2.1 决策链

```
--disaggregation-mode decode
        │
        ├─ 没开 --disaggregation-decode-enable-radix-cache
        │     └─ pd_disaggregation_hook.py:57  disable_radix_cache = True
        │           └─ server_args.py:7450     hicache ⊥ disable-radix-cache → ValueError ❌
        │
        └─ 开了 --disaggregation-decode-enable-radix-cache
              ├─ pd_disaggregation_hook.py:31/36/41  ⊥ hisparse / fake / 投机解码
              └─ kv_cache_builder.py:192-196  is_hybrid_swa → ValueError ❌   ← 真正卡住 SWA 的那一条
```

### 2.2 逐条对照代码

**Gate ①（间接）**：不开 decode radix cache 时，D 侧被强制 chunk cache。

```python
# python/sglang/srt/arg_groups/pd_disaggregation_hook.py:56-58
else:
    server_args.disable_radix_cache = True
    logger.warning("KV cache is forced as chunk cache for decode server")
```
随后撞上互斥校验（`server_args.py:7450`，`_handle_cache_compatibility`）：
`enable_hierarchical_cache and disable_radix_cache` → `ValueError`。
执行顺序对这条链是成立的：`_handle_pd_disaggregation()`（3438）→ `_handle_hicache()`（3506）
→ `_handle_cache_compatibility()`（3556）。

**Gate ②（直接）**：开了 decode radix cache，SWA 被点名拒绝。

```python
# python/sglang/srt/mem_cache/kv_cache_builder.py:185-201
# Decode radix cache is unsupported with hybrid SWA/SSM models —
# these use specialized memory pools incompatible with the
# prefix-match-and-lock allocation path.
if (server_args.disaggregation_decode_enable_radix_cache
        and server_args.disaggregation_mode == "decode"):
    if is_hybrid_swa:
        raise ValueError("--disaggregation-decode-enable-radix-cache is incompatible "
                         "with sliding window attention (SWA) models")
    if is_hybrid_ssm:
        raise ValueError("... incompatible with Mamba/SSM models")
```

而 D 侧 HiCache 的开关本身就是两个 flag 的 AND：

```python
# python/sglang/srt/managers/scheduler.py:394-397
self.enable_decode_hicache = (
    server_args.disaggregation_decode_enable_radix_cache
    and self.enable_hierarchical_cache
)
```

**Gate ③（不是障碍，容易误判）**：`HiRadixCache.__init__` 对池类型做白名单，
`SWAKVPool` 落到 else 分支报
`"HiRadixCache only supports MHA, MLA, DSA, and MSA models"`
（[hiradix_cache.py:110-111](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L110)）。
但 SWA 模型在工厂里**根本走不到** `HiRadixCache`——
`default_radix_cache_factory` 在第 107 行就把 `is_hybrid_swa` 截走去了
`UnifiedRadixCache`（[registry.py:107-129](../../../python/sglang/srt/mem_cache/registry.py#L107-L129)）。
所以「HiRadixCache 不支持 SWA」是真的，但**不是** PD 场景的阻塞点，改它是白费功夫。

顺带一提：`SWARadixCache` 在 main 上已经不再被生产代码实例化，只剩
`kv_canary/radix_cache_walker.py` 做类型判断和单测在构造；SWA 现在统一走
`UnifiedRadixCache` + `ComponentType.SWA`。`docs/pjc_1/swa_radix_cache_architecture.md`
里「is_hybrid_swa → SWARadixCache」的链路描述已过期。

---

## 3. 深层原因：把 gate 拆掉会立刻炸的 5 处

Gate ② 的注释只写了「specialized memory pools incompatible」，太笼统。下面是实际会坏的具体点。
理解它们才知道方案该怎么设计。

### 3.1 根因：`full_to_swa_index_mapping` 是全局表，而 full slot 会被 radix 树跨请求共享

正常（非 PD）的 SWA 请求：一次 extend 同时分配 full 和 swa 两侧的 slot，然后
`set_full_to_swa_mapping(full_indices, swa_indices)` 建立一一对应。

一旦 D 侧开了 radix 树，`[0, Pt)` 这段的 **full slot 是从树里借来的、可能同时被其它请求引用**。
但 SWA 侧不同：滑窗层只关心最后 `W` 个 token，两个共享同一段 full 前缀的请求，
它们的**滑窗位置不同**，需要的是**各自私有**的 swa slot。
而映射表是 `mapping[full_slot] → swa_slot` 一对一的，
**没有办法让同一个 full slot 对请求 A 映射到 swa_a、对请求 B 映射到 swa_b**。

> 这就是「specialized memory pools incompatible with the prefix-match-and-lock path」的真正含义：
> 不是池子不能分配，而是 **full slot 共享 ⇒ swa slot 不可能私有**。

推论：**只要「复用的前缀」和「滑窗」不重叠，冲突就不存在**。方案 A 就是把这一点变成硬不变式。

### 3.2 `alloc_extend_swa_tail` 的前置断言在 `Pt > 0` 时不成立

```python
# python/sglang/srt/mem_cache/allocator/swa.py:234-237
assert self.page_size > 1
assert len(seq_lens_cpu) == 1, "SWA tail allocation currently supports bs=1"
assert 0 <= swa_tail_len <= extend_num_tokens
```
它内部把 SWA 侧的 `prefix_lens` 硬编码成 0（[swa.py:261-275](../../../python/sglang/srt/mem_cache/allocator/swa.py#L261-L275)），
只给尾段新分配 SWA，并把窗口外的 full slot 映射置 0（:281-284）。
调用侧目前是这样绕过去的：

```python
# python/sglang/srt/disaggregation/decode.py:1526
uses_swa_tail = self._uses_swa_tail_prealloc() and prefix_len == 0
```
```python
# python/sglang/srt/disaggregation/decode.py:1661-1665
if uses_swa_tail:
    # Tail-only SWA allocation: only valid when prefix_len == 0.
    # When prefix_len > 0 (radix cache hit), we fall back to
    # alloc_extend which allocates SWA at full page count; the
    # SWA budget in that case may slightly under-estimate.
```
即：**一旦命中前缀，就退化成 `alloc_extend`**（full/swa 等长分配）。这条退化路径有下面三个 bug。

### 3.3 退化到 `alloc_extend` 的三个记账/正确性 bug

| # | 问题 | 位置 | 后果 |
|---|---|---|---|
| a | `swa_last_loc = translate_loc_from_full_to_swa(last_loc)`，`last_loc` 是**复用前缀最后一个 full slot**；它的 SWA 映射早已被 `free_swa` 置 0 或指向别人的 slot | [swa.py:191](../../../python/sglang/srt/mem_cache/allocator/swa.py#L191) | 分页分配器用 `last_loc` 续写「上一页的未满部分」，会写进**错误的 SWA 页**→ KV 静默错乱 |
| b | 退化分支**没有**设置 `req.kv.swa_evicted_seqlen`（尾窗分支在 [decode.py:1676](../../../python/sglang/srt/disaggregation/decode.py#L1676) 才设） | decode.py:1678-1688 | 保持 0 ⇒ 向上层谎报「本请求全部 SWA KV 都有效」，后续树侧匹配门禁失效 |
| c | SWA 预算按尾窗估、实际按 `delta_len` 全量分配 | `_swa_tail_allocatable_token_budget`（[decode.py:1362](../../../python/sglang/srt/disaggregation/decode.py#L1362)） | 代码注释自认 under-estimate ⇒ SWA 池 OOM / 断言失败 |

### 3.4 传输协议侧：SWA 走「状态通道」恒发整窗，与 `Pt` 完全解耦

这是**好事**，但也是必须对齐的约束。主 KV 通道只发增量：

```python
# decode.py:1102-1104（D 侧算 dst）
kv_indices = self.req_to_token_pool.req_to_token[req_pool_idx][total_prefix_len:origin_input_len]
```

而 SWA 走 `StateType.SWA` 状态通道，两侧都按「整个窗口」算，**不看前缀**：

```python
# decode.py:1117-1129（D 侧 dst）   /  prefill.py:1134-1146（P 侧 src），公式完全一致
window_start = page_align_floor(max(0, seq_len - window_size), page_size)
window_kv_indices_full = req_to_token[req_pool_idx, window_start:seq_len]
window_kv_indices_swa  = translate_loc_from_full_to_swa(window_kv_indices_full)
return kv_to_page_indices(window_kv_indices_swa, page_size)
```

于是**如果 `Pt > window_start`（复用前缀伸进了滑窗）**，`[window_start, Pt)` 这段：

1. 它的 full slot 属于 radix 树、被别的请求共享 → 翻译出来的 swa slot 要么是 0（page 0 是真实 slot，写进去就是踩内存），要么是**别人的 swa slot** → P 侧 RDMA 会**远端覆写其它请求的 SWA KV**；
2. 开了 HiCache 时更糟：`[P1, Pt)` 这段的 `req_to_token` 在 `pop_preallocated`
   计算 `_swa_payload` 的时刻**还没写**（要等 L2/L3 loadback 完成才由
   `_commit_hicache_local_restore_to_req` 写入，
   [decode_hicache_mixin.py:294-311](../../../python/sglang/srt/disaggregation/decode_hicache_mixin.py#L294)）
   → 用**未初始化的 req_to_token 行**去算 RDMA 目标地址。

反过来，**只要保证 `Pt ≤ window_start`，窗口就整段落在 `[Pt, L)` 里，
那段 `req_to_token` 在 `_pre_alloc` 里已经写好（[decode.py:1551-1557](../../../python/sglang/srt/disaggregation/decode.py#L1551)），
且 slot 是本请求私有的 → 协议一个字节都不用改。**

### 3.5 树侧语义：聚合场景的兜底路径在 D 侧不成立

聚合场景里，若「复用前缀的尾窗 SWA 不可用」，做法是**把那段尾窗重新 prefill 一遍**：

```python
# python/sglang/srt/managers/schedule_policy.py:103-108
# unified_kv SWA lives in a per-request ring that's not content-stable and is
# never stored in the radix tree, so a reused prefix carries stale SWA. Cap
# the match by the trailing sliding window so it gets re-prefilled, rewriting
# this request's SWA ring. No-op for other layouts.
reprefill_tail = tree_cache.swa_reprefill_tail_tokens()
key_limit = max(0, len(token_ids) - reprefill_tail) if reprefill_tail else None
```

注意它的真实机制**不是**「重算」，而是 **`RadixKey.limit` 把匹配长度截断到 `len - W`**，
让尾窗落进 extend 区间从而被本请求自己写一遍。这个机制**在 D 侧同样成立**：D 侧的
「extend 区间」= P 侧传过来的 `[Pt, L)`，尾窗落在里面就会被 P 侧的 SWA 状态通道填满。
**所以我们要的不是新机制，而是让 D 侧也走这条 `key_limit` 通路。**

---

## 4. 方案 A（推荐）：D 侧「不 offload SWA 的 HiCache 栈」+「前缀不进滑窗」不变式

### 4.1 核心不变式

```
                     Pt ≤ window_start = page_align_floor(L - W)

 token 轴:  0                     Pt          window_start        L
            ├──────────────────────┼───────────────┼──────────────┤
            │  复用前缀(树/L2/L3)  │  P 侧新传     │  P 侧新传    │
 full 层 KV │  ← 树里借           │  ← RDMA 主通道 [Pt, L)      │
 swa  层 KV │  不需要（滑窗读不到）│ 不需要 │ ← RDMA 状态通道整窗 │
                                           ↑ 全部是本请求私有 slot
```

三条推论，正好把第 3 节的 5 个问题全部消掉：

| 第 3 节问题 | 被不变式消掉的原因 |
|---|---|
| 3.1 full slot 共享 vs swa slot 私有 | 复用段**根本不需要 swa slot** |
| 3.2 `alloc_extend_swa_tail` 断言 | `swa_tail_len = L - window_start ≤ L - Pt = extend_num_tokens` ✅ 恒成立 |
| 3.3 三个 bug | 不再走 `alloc_extend` 退化分支，尾窗分支本来就正确设置 `swa_evicted_seqlen` |
| 3.4 传输协议 | 窗口 ⊂ `[Pt, L)`，dst 索引已写且私有；P 侧零改动 |
| 3.5 树侧语义 | 直接复用 `key_limit` 通路 |

### 4.2 怎么让不变式自动成立：不建 SWA host 池

`swa_reprefill_tail_tokens()` 的触发条件是「有 SWA 组件 + 开了 hicache + **没有** SWA host 池」：

```python
# python/sglang/srt/mem_cache/unified_radix_cache.py:1872-1887
swa = self.components.get(ComponentType.SWA)
unified_compress_only_hicache = (
    self.cache_controller is not None
    and swa is not None
    and not self.tree_core.has_swa_host_pool
)
return swa.sliding_window_size if unified_compress_only_hicache else 0
```

所以只要 **D 侧建栈时不建 SWA host 池**，`swa_reprefill_tail_tokens()` 自动返回 `W`，
`match_prefix_for_req` 自动把匹配截到 `L - W`，不变式**免费获得**。

这个形态**已有先例**：DSV4 的 unified_kv 栈就是「有 SWA 组件、无 SWA host 池」
（[hybrid_pool_assembler.py:323-325](../../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L323)
注释：*"unified_kv keeps the SWA ring inside the unified pool and never offloads it,
so there is no separate SWA host pool to map"*），
`SWAComponent._swa_kv_pool_host = None` 是被支持的合法状态
（[swa_component.py:74](../../../python/sglang/srt/mem_cache/unified_cache/components/swa_component.py#L74)、:118、:688）。

而且这件事**本身就是对的**：滑窗 KV 只有「最后一个窗口」有效，把它 offload 到 L2/L3
毫无复用价值（下一个请求的窗口位置不同），纯浪费 host 内存和带宽。

### 4.3 改动清单

> 记号：**[必须]** = 不改就错/崩；**[必须-记账]** = 不改会 OOM 或预算失真；**[建议]** = 质量项。

#### 改动 1 —— 放开 gate **[必须]**

`python/sglang/srt/mem_cache/kv_cache_builder.py:188-201`

```python
if (server_args.disaggregation_decode_enable_radix_cache
        and server_args.disaggregation_mode == "decode"):
    if is_hybrid_swa and not <SWA 尾窗预分配可用>:
        raise ValueError(...)          # 保留兜底：page_size == 1 / 无 alloc_extend_swa_tail
    if is_hybrid_ssm:
        raise ValueError(...)          # SSM 继续禁（mamba state 是 per-req 的，另一码事）
```

判据要和运行期一致：SWA 放开的前提是 `page_size > 1` 且分配器提供
`alloc_extend_swa_tail`——也就是 `DecodePreallocQueue._uses_swa_tail_prealloc()`
（[decode.py:368-373](../../../python/sglang/srt/disaggregation/decode.py#L368)）为真。
建议把这个判据提取成一个 `mem_cache` 层的纯函数（例如
`supports_decode_swa_tail_prealloc(allocator, page_size) -> bool`），
供 builder 和 `DecodePreallocQueue` 共用，避免两处判据漂移。
顺便把 `_uses_swa_tail_prealloc` 里的 `hasattr(...)` 探测换成 `isinstance`
判断（仓库规则 `.claude/rules/no-getattr-defensive.md`）。

#### 改动 2 —— D 侧建栈时跳过 SWA host 池 **[必须]**

`python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py:183-264`
（`build_hybrid_swa_stack`）加一个显式参数 `offload_swa: bool`：

- `offload_swa=True`（聚合场景，现状）：保持现在的两条 entry（`PoolName.KV` + `PoolName.SWA`）。
- `offload_swa=False`（PD decode）：
  - 不调用 `build_kv_host_pool(kv_pool=swa_kv_pool, ...)`；
  - `HostPoolGroup` 只放 `PoolName.KV` 一条 entry，`transfer_layer_num` 只算 full 层；
  - `_split_hicache_size` 不再切分，`hicache_size` 全给 full 池；
  - `SWAComponent._swa_kv_pool_host` 保持 `None` ⇒ `has_swa_host_pool = False`
    （由 [unified_radix_cache.py:349](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L349) 自动推出）。

参数来源：`disaggregation_mode == "decode"`。建议在 `CacheInitParams` 或建栈的
策略匹配处显式传入，不要在 assembler 里读全局 `server_args`（保持可测试性）。

#### 改动 3 —— full 组件的 load-back 只分配 full slot **[必须]**

L2/L3 恢复的是 `[P1, Pt)` 这段，**全部在 `window_start` 左侧，不需要 SWA slot**。
但 `build_hybrid_swa_stack` 里 `PoolName.KV` 这条 entry 没有指定 `device_alloc_fn`，
默认走复合分配器 `SWATokenToKVPoolAllocator.alloc`，它会**连带分配 SWA slot 并建立映射**
（[swa.py:151-164](../../../python/sglang/srt/mem_cache/allocator/swa.py#L151)）——
在 `offload_swa=False` 下这是纯浪费，且会让 SWA 池被无效 slot 占满。

改法：`offload_swa=False` 时给 KV entry 显式指定

```python
device_alloc_fn=full_attn_allocator.alloc,   # 只动 full 侧
device_free_fn=full_attn_allocator.free,
```
（对照 SWA entry 现在就是这么做的：[hybrid_pool_assembler.py:241-242](../../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L241)
`device_alloc_fn=swa_attn_allocator.alloc`。）

⚠️ 需要实测确认两点（推断）：① `PagedTokenToKVPoolAllocator.alloc(need_size)`
在 `page_size > 1` 下的语义是否满足 load-back 的调用约定
（复合分配器的 `alloc` 有 `assert self.page_size == 1`，内层分页分配器不一定）；
② 新分配的 full slot 其 `full_to_swa_index_mapping` 必须是 0
（初始为 0、`free_swa` 也会写回 0，[swa.py:359](../../../python/sglang/srt/mem_cache/allocator/swa.py#L359)，
所以大概率天然满足，但要加断言锁住）。

#### 改动 4 —— `alloc_extend_swa_tail` 支持 full 侧带前缀 **[必须]**

分配器侧（`swa.py:218-285`）实际上**已经**能工作：它对 full 侧原样转发
`prefix_lens / last_loc / extend_num_tokens`，只把 SWA 侧从 0 起算。
真正要改的是调用侧 —— 现在传的是「假前缀 0 + 全长 fill_len」：

```python
# 现状 python/sglang/srt/disaggregation/decode.py:1666-1675（uses_swa_tail 分支）
prefix_lens     = 0            # ← 与 last_loc 自相矛盾的根源
seq_lens        = fill_len
extend_num_tokens = fill_len
```

改为传真实前缀：

```python
prefix_lens       = total_prefix_len
prefix_lens_cpu   = total_prefix_len
seq_lens          = fill_len
extend_num_tokens = delta_len          # = fill_len - total_prefix_len
last_loc          = prefix_indices[-1] if prefix_len > 0 else -1
swa_tail_len      = fill_len - window_start      # 不变
```

并去掉 `decode.py:1526` 的 `and prefix_len == 0` 限制，改成断言不变式：

```python
uses_swa_tail = self._uses_swa_tail_prealloc()
swa_tail_len = self._swa_tail_len(fill_len)
if uses_swa_tail:
    assert total_prefix_len <= fill_len - swa_tail_len, (
        f"decode prefix must not reach into the sliding window: "
        f"total_prefix_len={total_prefix_len}, window_start={fill_len - swa_tail_len}"
    )
```

⚠️ `last_loc` 的语义要留意：它是 **full 侧**「前缀最后一个 slot」，用于分页续写；
SWA 侧在函数内部另用 `swa_last_loc = -1`（[swa.py:265](../../../python/sglang/srt/mem_cache/allocator/swa.py#L265)），
**不会**再去翻译 full 前缀的映射 —— 这正好躲开了 3.3(a) 的坑。改完后
`translate_loc_from_full_to_swa(last_loc)` 在这条路径上不再被调用。

顺便：`assert len(seq_lens_cpu) == 1`（bs=1）在预分配路径上天然成立（逐请求调用），无需改。

#### 改动 5 —— L3 命中长度也要受 `limit` 约束 **[必须]**

`RadixKey.limit` 只约束了树内匹配（L1 + L2 host）。L3 的查询用的是**未截断**的后缀：

```python
# python/sglang/srt/disaggregation/decode_hicache_mixin.py:76-89
matched_len = l1_prefix_len + l2_host_hit_length
suffix_tokens = req.origin_input_ids[matched_len:]        # ← 没有上限
l3_storage_hit_length = self.tree_cache.query_storage_hit_length(...)
```

必须夹一刀，否则 `Pt = L1+L2+L3` 会突破 `window_start`，不变式失效：

```python
cap = fill_len - self.scheduler.tree_cache.swa_reprefill_tail_tokens()   # 或直接用 window_start
suffix_tokens = req.origin_input_ids[matched_len : max(matched_len, cap)]
...
l3_storage_hit_length = min(l3_storage_hit_length, max(0, cap - matched_len))
```

同时 `_start_hicache_prefetch`（[decode_hicache_mixin.py:101-139](../../../python/sglang/srt/disaggregation/decode_hicache_mixin.py#L101)）
里的 `suffix` 切片也要用同一个夹紧后的长度，保证 prefetch 的 page 数与
`l3_storage_hit_length` 一致（两处不一致会让 `_try_hicache_queue_load_back`
的覆盖度检查判 FAILED，退化成整段重传）。

**建议**把 `cap` 计算收敛成 `DecodePreallocQueue` 的一个方法（例如
`_max_reusable_prefix_len(fill_len)`），三处（match 的 limit、L3 clamp、prefetch 切片）
都读它，避免公式漂移。

#### 改动 6 —— 双池预算与双池驱逐 **[必须-记账]**

现在 D 侧的准入/驱逐都只按一个池算。开了树以后要按 full / swa 两个预算算：

| 位置 | 现状 | 改法 |
|---|---|---|
| `_allocatable_token_budgets` [decode.py:1330-1333](../../../python/sglang/srt/disaggregation/decode.py#L1330) | SWA 分支里 `available_size += self.tree_cache.evictable_size()` | 换成 `full_evictable_size()`（unified 树提供，[unified_radix_cache.py:1924](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1924)）；`evictable_size()` 语义含 SWA 分量，加到 full 预算上是高估 |
| `_swa_tail_allocatable_token_budget` [decode.py:1362](../../../python/sglang/srt/disaggregation/decode.py#L1362) | 完全不考虑树 | 加上 `swa_evictable_size()`（若树确实持有 SWA 分量；不建 SWA host 池时仍可能有设备侧 SWA 挂在叶节点上） |
| `_pre_alloc` 的驱逐 [decode.py:1480-1499](../../../python/sglang/srt/disaggregation/decode.py#L1480) | `available_size()`（= min(full,swa)）+ `EvictParams(num_tokens=...)` | 分别算 full 缺口与 swa 缺口，传 `EvictParams(num_tokens=..., swa_num_tokens=...)`（聚合路径就是这么调的） |

另外 `decode.py:357-366` 那个「SWA 分配器不支持尾窗时把 `max_total_num_tokens`
压到 `swa_max_total_num_tokens`」的兜底，在尾窗可用时本来就跳过，无需改动；
但要确认它与新的树预算不会双重保守（推断：不会，因为它只在 `not _uses_swa_tail_prealloc()` 时生效）。

#### 改动 7 —— 请求结束时 SWA 的释放路径 **[必须]**

`cache_finished_req` 把 full KV 交给树（不释放），SWA 侧必须释放。
基础设施已就绪：`SWATokenToKVPoolAllocator.free_swa(free_index)`
（[swa.py:347-359](../../../python/sglang/srt/mem_cache/allocator/swa.py#L347)）
接受 **full 索引**，按页展开、翻译、释放 swa slot、并把映射写回 0。
保留 `{FULL, SWA}` 双组件正是为了让 `SWAComponent` 现成的
tombstone / 提前释放逻辑接管这件事（这也是不选「FULL-only 树」的主要理由）。

需要验证（推断）：`SWAComponent` 在 `_swa_kv_pool_host is None` 时的
insert / evict / free 分支（:118、:688 有专门判断）在 **PD decode 的 PREBUILT
批次**下也走得通 —— D 侧没有真正的 prefill forward，`cache_unfinished_req`
的调用时机与聚合场景不同，需要 e2e 验证。

#### 改动 8 —— `swa_evicted_seqlen` 的取值确认 **[建议]**

尾窗分支设 `req.kv.swa_evicted_seqlen = fill_len - swa_tail_len = window_start`
（[decode.py:1676](../../../python/sglang/srt/disaggregation/decode.py#L1676)），
语义正确：「本请求 `[0, window_start)` 的 SWA KV 视为已驱逐」。
改动 4 之后这条路径恒定生效，第 3.3(b) 的 bug 自然消失。
建议补一条断言把它和树侧的 tombstone 边界对齐。

---

## 5. 边界与风险

### 5.1 ⚠️ L3 存储键与 P 侧布局冲突（上线前必须定）

HiCache 的 L3 键 = **token id 链式哈希**，不含层数/布局指纹：

```python
# python/sglang/srt/mem_cache/utils.py:106-112 → get_native_hash(token_ids, prior_digest, page_size)
```
各后端只在 base hash 上加「model / tp_rank / K-V 分量」等命名空间后缀。

于是：**P 实例的 L3 页含「全部层（full + swa）」，D 实例（本方案）只含「full 层」**，
若两者共享同一个 L3 后端 + 同一个 `model_name`，**同一个 key 会对应两种不同的 payload 布局** → 静默数据错乱。

两条出路：

| 选项 | 做法 | 代价 |
|---|---|---|
| **B1（推荐先做）** | 给 D 侧的 storage 命名空间加后缀（如 `hicache_storage_backend_extra_config` 里加 `layout_tag="full_only"`），P/D 各自独立 L3 | 失去「D 复用 P 写的 L3」这一潜在收益 |
| **B2** | 让 P 侧 L3 也只存 full 层（P 侧 SWA 同样没有跨请求复用价值） | 改动扩散到 P 侧建栈；但语义上更自洽，长期更优 |

我倾向 v1 先 B1（隔离、零风险），把 B2 作为后续统一项。**这一节需要在动手前和 HiCache owner 确认现网是否已有 P/D 共享 L3 的部署。**

### 5.2 收益边界：`W` 与 prompt 长度的关系

不变式把可复用前缀上限压到 `L - W`：

| 场景 | `W` | prompt `L` | 可复用上限 | 评价 |
|---|---|---|---|---|
| 长上下文 RAG | 4096 | 32768 | 28672（87%） | 收益基本无损 |
| 一般对话 | 4096 | 8192 | 4096（50%） | 收益减半 |
| 短 prompt | 4096 | 2048 | **0** | **完全无收益**，命中率 0 |

所以这个特性的定位很明确：**面向 `prompt >> sliding_window` 的长上下文 PD 部署**。
短 prompt 场景下开了也白开（且多了一次匹配开销），建议在日志里 warn 一次。

### 5.3 仍然保持禁止的组合

| 组合 | 原因 | 位置 |
|---|---|---|
| decode radix cache + hisparse | `_pre_alloc` 里 `assert prefix_len == 0` | [decode.py:1502-1505](../../../python/sglang/srt/disaggregation/decode.py#L1502) |
| decode radix cache + 投机解码 | 现有显式拒绝 | pd_disaggregation_hook.py:41 |
| decode radix cache + `transfer_backend=fake` | 现有显式拒绝 | pd_disaggregation_hook.py:36 |
| decode radix cache + Mamba/SSM | state 是 per-req 的，问题性质不同 | kv_cache_builder.py:197 |
| DSV4 on NPU + 非零前缀 | 已有显式 RuntimeError | [decode.py:1180-1185](../../../python/sglang/srt/disaggregation/decode.py#L1180) |
| `page_size == 1` 的 SWA | `alloc_extend_swa_tail` 断言 `page_size > 1` | swa.py:234 |

### 5.4 DP attention 路由

D 侧 radix cache + DP attention 已被标记 EXPERIMENTAL（pd_disaggregation_hook.py:49-53），
需要前缀感知的 DP rank 路由才有命中率。本方案不改变这一点。

---

## 6. 分阶段落地建议

| 阶段 | 内容 | 可独立合并 |
|---|---|---|
| **P0** | 改动 6（双池预算/驱逐）+ 改动 8（断言）—— 纯记账修正，对现状（`prefix_len==0`）也是正向 | ✅ |
| **P1** | 改动 4（`alloc_extend_swa_tail` 带前缀）+ 单测，**先只对不开 HiCache 的 decode radix cache 放开 SWA** | ✅ |
| **P2** | 改动 2 + 3（不建 SWA host 池的栈形态）+ 改动 5（L3 clamp）+ 改动 1（放开 gate） | 需 P1 |
| **P3** | 5.1 的 L3 命名空间隔离；e2e 压测与命中率指标 | 需 P2 |

P1 先落地「SWA + decode radix cache（无 HiCache）」是个很好的中间态：
它验证了不变式和分配器改动，风险面小得多，而 HiCache 只是在其上多了 L2/L3 两级。

---

## 7. 测试计划

### 7.1 单测（无需 GPU，`test/registered/unit/mem_cache/`）

1. `alloc_extend_swa_tail` 带前缀：构造 `page_size=64, W=256, L=1024, Pt=512`，
   断言 ① full 只新分配 `(1024-512)/64` 页；② swa 只分配
   `ceil((1024 - page_align_floor(768))/64)` 页；③ `[Pt, window_start)` 的 full slot
   映射为 0；④ `[window_start, L)` 的映射一一对应且不与任何已有 slot 冲突。
2. 不变式断言：`Pt > window_start` 时必须抛异常而不是静默错分配。
3. `key_limit` 端到端（mock 树）：`swa_reprefill_tail_tokens()` 返回 `W` 时，
   `match_prefix_for_req` 的返回长度 `≤ L - W`。
4. L3 clamp：mock `query_storage_hit_length` 返回超长值，断言 `Pt ≤ window_start`。
5. 建栈形态：`offload_swa=False` 时 `HostPoolGroup` 只有一条 KV entry、
   `has_swa_host_pool is False`、`swa_reprefill_tail_tokens() == W`。
6. 预算：`full_evictable_size` / `swa_evictable_size` 分别注入不同值，
   断言两个预算函数各取各的。

参照 `.claude/skills/write-sglang-test/SKILL.md` 与 `test/README.md` 做 CI 注册。

### 7.2 e2e（需 2 卡以上）

- 模型：任一 `is_hybrid_swa` 白名单模型（`model_config.py:1657`），如 Gemma / Llama4 / GptOss 系。
- 配置：P 实例正常；D 实例 `--disaggregation-decode-enable-radix-cache
  --enable-hierarchical-cache --hicache-ratio 2`。
- 断言：① 相同 prompt 二次请求，输出与「关 HiCache」的 baseline **逐 token 一致**
  （这是唯一能抓到 SWA KV 错乱的手段）；② `decode_prefix_len` 上报值恒 `≤ L - W`；
  ③ 并发 64 路长短混合跑 10 分钟无 SWA 池断言失败、无 `Eviction insufficient` 刷屏。
- 回归：关掉新特性时行为与改动前完全一致（`SWAChunkCache` 路径不受影响）。

---

## 8. 相关文件索引

| 文件 | 关键位置 |
|---|---|
| `arg_groups/pd_disaggregation_hook.py` | 29-58 decode 模式的 flag 归一与拒绝 |
| `server_args.py` | 2963 flag 定义；7450 hicache ⊥ disable-radix；6961 `_handle_hicache` |
| `mem_cache/kv_cache_builder.py` | 185-201 SWA/SSM 拒绝；73-127 draft host 池 |
| `mem_cache/registry.py` | 79-160 树选择链；163-193 `_create_unified_radix_cache` |
| `mem_cache/allocator/swa.py` | 20 分配器；82 映射表；147 翻译；174 `alloc_extend`；218 `alloc_extend_swa_tail`；318 `free`；347 `free_swa` |
| `mem_cache/hybrid_cache/hybrid_pool_assembler.py` | 86 `_split_hicache_size`；128 `build_kv_only_stack`；183 `build_hybrid_swa_stack`；303 DSV4 栈（无 SWA host 池先例）；782 组件→host 池属性表 |
| `mem_cache/unified_radix_cache.py` | 296 `init_hicache`；349 `has_swa_host_pool`；1872 `swa_reprefill_tail_tokens`；1924 `full_evictable_size` |
| `mem_cache/unified_cache/components/swa_component.py` | 74 `_swa_kv_pool_host=None`；118/688 无 host 池分支 |
| `mem_cache/hiradix_cache.py` | 85-111 池类型白名单（SWA 落 else） |
| `managers/schedule_policy.py` | 92-114 `match_prefix_for_req` 与 `key_limit` |
| `managers/scheduler.py` | 394-397 `enable_decode_hicache`；472-501 建树；524-540 offload manager |
| `disaggregation/decode.py` | 286 PreallocQueue；357-373 SWA 尾窗判据；866 `pop_preallocated`；1117 `_swa_payload`；1306/1362 双预算；1437 `_pre_alloc`；1619 `alloc_for_decode_prealloc`；1696 TransferQueue |
| `disaggregation/decode_hicache_mixin.py` | 61 `_build_decode_prefix_match`；101 prefetch；183 load-back；294 commit |
| `disaggregation/prefill.py` | 1134-1146 `_swa_payload`（src 侧，公式与 D 侧一致） |
| `mem_cache/utils.py` | 106 `get_hash_str`（L3 键不含布局指纹 → 5.1 风险） |

---

## 9. 与既有文档的关系

- `docs/pjc_1/swa_radix_cache_architecture.md`：讲 `SWARadixCache`（**已不再被生产代码实例化**，
  能力已并入 `UnifiedRadixCache` + `ComponentType.SWA`）。读它时注意这条过期信息。
- `docs/pjc_1/hicache_usage_and_design.md`：L1↔L2 layout / IO backend / L3 键构造。
- `docs/pjc_1/dsv4_pd_disaggregation_request_lifecycle.md`：PD 请求全生命周期与状态通道传输，
  本文 3.4 节的协议部分是它的 SWA 特化。
