# PD 分离下 DSV4 的 D 实例为何不能开 Radix Cache，以及改造方案

> 适用版本：`main` @ `c608f9bf75`（2026-08-28）
> 关注对象：`--disaggregation-mode decode` + DeepSeek-V4（DSA）+ `--disaggregation-decode-enable-radix-cache`
> 关联文档（本文不重复其内容，只做交叉引用）：
> - [`dsv4_pd_disaggregation_request_lifecycle.md`](dsv4_pd_disaggregation_request_lifecycle.md) —— DSV4 在 PD 下的完整请求生命周期与六子池传输通道
> - [`deepseek_v4_cache_management.md`](../09_models/deepseek_v4_cache_management.md) —— 六子池/三套索引空间/地址翻译
> - [`unified_radix_cache_architecture.md`](../03_cache_memory/unified_radix_cache_architecture.md) —— 统一 radix 树的 component 模型
> - [`pd_decode_dsv4_hicache_reuse_scheme.md`](pd_decode_dsv4_hicache_reuse_scheme.md)、[`pd_decode_swa_hicache_support_scheme.md`](pd_decode_swa_hicache_support_scheme.md)、[`deepseek_pd_decode_output_kv_l3_scheme.md`](pd_decode_output_kv_writeback_l3_scheme.md)

---

## 0. 一句话结论

**不是"物理上做不到"，而是"当前实现显式拒绝"。**

`--disaggregation-decode-enable-radix-cache` 这条路径的全部假设是「**一个 KV 索引空间 + 一个标量 `decode_prefix_len` + 一棵 `[FULL, SWA]` component 的树**」；而 DSV4 的 KV 由**三套互不重合的索引空间**组成（full 逻辑槽 / SWA 槽 / 压缩态 ring），其中**第三套（compress state ring）完全没有被 radix 树建模，也没有被 PD 前缀协议表达**。因此 PR #27770 在为通用 SWA 混合模型打开 decode-side radix 时，把 DSV4 显式圈成了 "not supported yet"。

同时，生产上 DSV4 的 D 实例几乎必然还会踩到另外四道**独立**的互斥门（HiSparse、投机解码 MTP/EAGLE、DCP、`--enable-hierarchical-cache`），所以即使去掉 DSV4 这一条 `raise`，也不能直接跑通。

---

## 1. 先说清楚：decode 侧 radix cache 到底在省什么

普通 PD 分离下 D 实例是"纯搬运方"：P 算完整个 prompt 的 KV，全量 RDMA 写进 D 预分配好的槽位。开启 decode 侧 radix cache 后，D 变成"**会砍单的搬运方**"：

```
        ┌────────── D 实例 ──────────┐
请求到达 │ ① match_prefix_and_lock    │  在 D 本地 radix 树上匹配已有前缀
        │    → DecodePrefixMatch     │  (l1_prefix_len / decode_prefix_len)
        │ ② _pre_alloc               │  前缀行直接写已有 slot，只为 delta 分配新槽
        │ ③ send_metadata            │  把 decode_prefix_len 推给 P
        └────────────┬───────────────┘
                     ↓ ZMQ
        ┌────────── P 实例 ──────────┐
        │ start_send_idx = prefix_len│  只发后缀，前缀那段 KV 不上网
        └────────────────────────────┘
```

关键代码锚点：

| 环节 | 位置 |
|---|---|
| 前缀匹配 + 加锁 | `disaggregation/decode.py:645`（`_match_prefix_and_lock`）|
| 准入时读取三层命中长度 | `disaggregation/decode.py:1190-1258` |
| 提前释放 SWA 锁 | `disaggregation/decode.py:1224-1233`（`dec_swa_lock_only`）|
| 只传 delta 的 kv_indices | `disaggregation/decode.py:1362-1372` |
| `decode_prefix_len` 上线 | `disaggregation/decode.py:1496` |
| 预分配（前缀行复用 + delta 新分配）| `disaggregation/decode.py:1746-1884` |
| P 侧按 prefix 砍单 | `disaggregation/prefill.py:344-349` |

收益方向很清楚：**省 P 的重复 prefill 算力 + 省网络带宽**。对多轮对话 / RL rollout 这类长公共前缀负载，收益随前缀命中率线性增长。这也是为什么值得改造。

---

## 2. 拦截链：报错到底在哪、有几道

### 2.1 直接拦截点

`mem_cache/kv_cache_builder.py:230-262`，一段针对 `disaggregation_mode == "decode"` 且开了该 flag 的集中校验：

```python
if (
    get_disagg().disaggregation_decode_enable_radix_cache
    and get_disagg().disaggregation_mode == "decode"
):
    if is_hybrid_swa:
        if not (envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get() or use_mlx()):
            raise ValueError("... requires the unified radix tree (set SGLANG_ENABLE_UNIFIED_RADIX_TREE=1).")   # :235-240
        if enable_hierarchical_cache:
            raise ValueError("... supports only device-resident cache ...")                                      # :241-247
        if getattr(model_config, "is_deepseek_v4_arch", False):
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache does not support "
                "DeepSeek-V4 (DSA) compressed KV (c4/c128/indexer) yet."                                         # :248-252  ← DSV4 在此被拒
            )
        if getattr(model_config, "is_hybrid_swa_compress", False):
            raise ValueError("... does not support SWA-compress models (e.g. Gemma4 / MiMo-V2) yet.")             # :253-257
    if is_hybrid_ssm:
        raise ValueError("... is incompatible with Mamba/SSM models")                                            # :258-262
```

DSV4 命中 `:248`，因为 `is_deepseek_v4_arch` 在 `configs/model_config.py:789-827` 被设置（架构名 `DeepseekV4ForCausalLM` / `...NextN` / `...DSpark`），且**该字段是嵌在 `if self.is_hybrid_swa:` 分支内部赋值的**。

### 2.2 这条 raise 的来历（重要：它是"暂未实现"，不是"证明不可行"）

```
git log -L 226,262:python/sglang/srt/mem_cache/kv_cache_builder.py
→ 978244d671  2026-08-20
  [P/D disagg] Decode-side radix cache for SWA hybrid models (unified radix tree) (#27770)
```

这次提交之前，这里是一条**笼统的**拒绝：`"--disaggregation-decode-enable-radix-cache is incompatible with sliding window attention (SWA) models"`。#27770 把通用 SWA 混合模型放开（走统一 radix 树），同时把 **DSV4** 和 **SWA-compress 模型** 两类"KV 不只有 full+SWA 两层"的模型单独刻出来继续拒绝。

**结论：这是一次保守的能力边界切分，措辞 "yet" 是作者自己留的口子。**

### 2.3 还有四道独立的门（生产上必然同时踩到）

即使把 `:248-252` 删掉，DSV4 的 D 实例仍跑不起来：

| 门 | 位置 | 说明 |
|---|---|---|
| DCP（decode context parallel）| `arg_groups/pd_disaggregation_hook.py:63-67` | 开 flag 与 `dcp_size>1` 互斥 |
| HiSparse | `arg_groups/pd_disaggregation_hook.py:76-80` + `arg_groups/hisparse_hook.py:101-103`（`assert cfg.disable_radix_cache`）+ `decode.py:1813-1816`（`assert prefix_len == 0`）| **三处**，HiSparse 是 DSV4 大并发部署的常用配置 |
| 投机解码（MTP/EAGLE/DSPARK）| `arg_groups/pd_disaggregation_hook.py:86-91` | DSV4 生产几乎默认开 MTP |
| 分层缓存 L2/L3 | `kv_cache_builder.py:241-247` | DSV4 无 `get_cpu_copy`，见 §3.6 |

外加一条**必须显式打开**的前置：`SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`（`kv_cache_builder.py:235-240`）。

---

## 3. 底层技术原因

### 3.1 根因：一边是"单索引空间"假设，一边是"三索引空间"现实

decode 侧 radix 路径的数据结构，从头到尾只表达**一条**索引向量和**一个**标量长度：

- 树：`mem_cache/registry.py:143-176` `_create_unified_radix_cache` 只组装 `tree_components = [FULL] (+ SWA if hybrid_swa) (+ MAMBA if hybrid_ssm)`；
- 命中描述：`DecodePrefixMatch` 三个字段 `l1_prefix_len` / `decode_prefix_len` / `restore_token_count`，都是**token 数**；
- 线上协议：`decode.py:1496` 只带一个 `decode_prefix_len`；
- 预分配：`decode.py:1775-1778` 把前缀行原样写回 `req_to_token`，`:1783` 只为 `delta_len = fill_len - total_prefix_len` 拿新槽。

而 DSV4 的 KV 分布在**三套地址空间**上（详见 `deepseek_v4_cache_management.md`）：

```
            full 逻辑槽号 (out_cache_loc)            ← radix 树管的就是这一套
                    │
      ┌─────────────┼──────────────────────────────┐
      │             │                              │
   ① //ratio     ② full_to_swa_index_mapping    ③ ring 寻址
      │             │                              │
  c4_kv_pool    swa_kv_pool                 compress_state_pools
  c128_kv_pool  (滑窗全精度)                indexer_compress_state_pools
  c4_indexer                                 ├ c4/indexer: key = SWA page
  (纯函数派生)                                └ c128:       key = req_pool_idx
```

- **① 是免费的**：c4/c128/indexer 的行号是 full 槽号的**纯函数**（`// ratio`）。所以只要 full 槽被复用，这三个池的内容天然跟着复用，无需任何额外簿记。这也是为什么 CUDA 上这三个池**根本没有 free / 引用计数**（`allocator/swa.py:317-330` 只释放 full 和 SWA）。
- **② 是可变映射**：`full_to_swa_index_mapping` 是单值表，一个 full 槽在任意时刻只映射到一个 SWA 槽；SWA 槽被驱逐后表项失效。
- **③ 完全在 radix 树的视野之外**：没有任何 tree component 建模 compress state ring，没有任何 `free` 路径管它，`decode_prefix_len` 也无法表达"ring 里哪些块是有效的"。

**这就是 `raise` 里 "compressed KV (c4/c128/indexer)" 这句话的真实含义**：不是 c4/c128 的 KV 本身有问题（那部分是 ① 免费的），而是**它们的中间态 ring 没有被前缀协议覆盖**。

### 3.2 传输通道的索引粒度本来就是错位的

`mem_cache/deepseek_v4_memory_pool.py:660-840` 定义了 PD 传输契约，两条通道用的**不是同一种页号**：

| 通道 | 构造函数 | item 索引含义 | item_len |
|---|---|---|---|
| 主 KV 通道 | `get_contiguous_buf_infos()` | **full 页号**（c4/c128 各自的 page_size 被按比例缩小成 `256//4=64`、`256//128=2`，从而与 full 页号 **1:1 恒等**）| `buf[0].nbytes`（一页）|
| 状态通道 | `get_state_buf_infos()` | **SWA 页号** | `t[0].nbytes * ring_size`（**一整个 ring 块**）|
| c128 状态通道 | `get_c128_state_buf_infos()` | `req_pool_idx` | online 时一行，离线时 ×128 |

`decode_prefix_len` 是一个 full-token 长度。主 KV 通道能直接用它（1:1 恒等映射），状态通道**必须先经 `translate_loc_from_full_to_swa`（`:671-673`）翻译**，而 c128 那条连翻译都不成立（它按请求而非按位置寻址）。

现状代码里，这个翻译**只在 NPU 上实现了 prefix-aware 版本**：

```python
# disaggregation/decode.py:1447-1460
if _is_npu and isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
    from sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks import dsv4_state_payloads
    payloads.update(dsv4_state_payloads(..., prefix_len=total_prefix_len))
```

**CUDA 路径没有任何带 `prefix_len` 的 DSV4 状态 payload。** 这是最直接、最可操作的缺口。

### 3.3 具体的正确性隐患：提前释放 SWA 锁 × `other=0` 静默读 0 行

这是本文认为**最需要在改造中优先处理**的一处。

decode radix 为了让"SWA 窗口永远由新传输的 delta 覆盖"，在准入时**主动放掉前缀的 SWA 锁**：

```python
# disaggregation/decode.py:1224-1233
self.tree_cache.dec_swa_lock_only(...)
req.swa_prefix_lock_released = True
# 注释原文：Decode transfers the SWA tail fresh, so retain only the
#           full-attention prefix lock needed for reuse.
```

而 c4 的压缩态地址，是**穿过 `full_to_swa_index_mapping` 解析出来的**：

```python
# python/sglang/kernels/ops/attention/dsv4/attn.py:87-215
#   create_paged_compress_data_kernel
prefix_len = seq_len - extend_len
write_pos          = ((seq_len    - 1) // cr) * cr
load_pos           = ((prefix_len - 1) // cr) * cr      # ← 落在"复用前缀"里
write_overlap_pos  = write_pos - cr
load_overlap_pos   = load_pos  - cr
...
if compress_ratio == 128:
    state_loc = rid * ring_size + (pos % ring_size)
else:
    loc     = tl.load(req_to_token_ptr + rid*stride0 + pos*stride1, mask=mask, other=0)
    swa_loc = tl.load(full_to_swa_index_mapping_ptr + loc,          mask=mask, other=0)
    state_loc = (swa_loc // swa_page_size) * ring_size + (swa_loc % ring_size)
state_loc = state_loc // cr
```

两个致命细节：

1. `load_pos` / `load_overlap_pos` 定位的是**块边界的在途压缩态**，位置落在前缀区间内；
2. 两次 `tl.load` 都用 `other=0`。映射表项若已失效（SWA 槽被驱逐、表项为 `-1` 或陈旧），kernel **不会报错，而是静默读第 0 行**。

叠加起来：decode radix 故意释放了前缀 SWA 锁 ⇒ 前缀的 SWA 槽可被驱逐 ⇒ `load_pos` 处的 state 地址解析到错误行 ⇒ **压缩 KV 静默算错，无任何异常信号**。这类 bug 只会表现为精度缓慢劣化，极难定位，因此把它挡在 `raise` 后面是合理的工程选择。

现状里已有一半的缓解：`decode.py:1190-1258` 的 `swa_prefix_cap = fill_len - self._swa_tail_len(fill_len)` 会**把前缀复用长度截断到 SWA 窗口之外**，保证整个滑窗都由新 delta 覆盖。但 `_swa_tail_len`（`decode.py:459-480`）的 radix 分支用的是 `max(0, seq_len - 1 - max(window_size, page_size))` 再按页下取整——它只保证 SWA **KV** 落在 delta 里，**没有覆盖 `load_overlap_pos` 这个再往前一个压缩块（`- cr`）的读取点**。

> ⚠️ **本节结论仅来自静态代码阅读，未实机执行验证。** 是否真的会读到错误行、`swa_prefix_cap` 的页对齐余量是否恰好把 `load_overlap_pos` 也一起兜住，需要实测（见 §6）。

### 3.4 online c128 是真正的递推状态

离线 ring 存的是**原始 `(kv, score)` 对**（`c4.cuh:41-62` 布局 `| kv overlap | kv | score overlap | score |`），第 *t* 项不依赖第 *t-1* 项 —— 它是 staging，不是累加器，因此天然容忍"前缀不重算"。

但 `SGLANG_OPT_USE_ONLINE_COMPRESS` 打开后的 c128 是**真 online-softmax 累加器**（`c128_online.cuh:120-192`，保存 `(max, sum, kv)`），`ring_size == 1`，按 `req_pool_idx` 寻址，靠 `pos_in_chunk == 0` 复位，并需要 `clear_c128_req_state`（`deepseek_v4_memory_pool.py:1007-1024`；PD 侧调用在 `decode.py:1420-1436`）。

**递推状态无法从"复用了前缀 KV"中恢复**：跳过前缀就等于跳过了递推的前若干步。所以 online c128 与前缀复用在语义上是硬冲突，必须么关掉、么为它单独传输/重建边界态。

### 3.5 压缩 KV 在语义上是"内容决定"的 —— 这一点反而是好消息

压缩 KV 的定义（`layers/attention/dsv4/compressor.py:335-380`，kernel `c4.cuh` / `c128.cuh`）：

```
compressed_kv(block) = pool_j( wkv_gate(x_j) + ape[j - block_start] )
```

`ape` 是 `nn.Parameter(torch.empty(ratio, coff*head_dim))`，**只按块内偏移索引**，不含绝对位置。因此：

- 同样的 token 序列、同样的块边界 ⇒ **压缩 KV 逐位相同**，与该块在哪个请求、哪个绝对位置无关；
- `page_size = 256` 同时被 4 和 128 整除 ⇒ radix 的页对齐插入边界（`radix_cache.py:150-154` `RadixKey.page_aligned`）**天然就是压缩块边界**，不会出现"半个压缩块被复用"的情况。

这两点把"复用 DSV4 压缩 KV 是否合法"这个理论问题直接消掉了。**合法。**

### 3.6 DSV4 没有 `get_cpu_copy` ⇒ 与 L2/L3 及 host 回退互斥

`KVCache.get_cpu_copy` / `load_cpu_copy` 基类抛 `NotImplementedError`（`mem_cache/memory_pool.py:1776-1780`），`HiSparseC4DevicePool` 再抛一次（`deepseek_v4_memory_pool.py:245`），DSV4 池从未 override。后果：

- `--enable-hierarchical-cache`（L2 host / L3 storage）不可用 —— 正是 `kv_cache_builder.py:241-247` 那条 raise；
- retraction 的 host 备份路径不可用 —— `kv_cache_builder.py:144-157` 的 `resolve_decode_retraction_backup` 只在 `not disaggregation_decode_enable_radix_cache` 时才选 `host_pool`。

所以 DSV4 的 decode radix 第一步只能是**纯 device-resident**（L1）方案。想上 L2/L3 是另一个独立工作项，见 `pd_decode_dsv4_hicache_reuse_scheme.md`。

### 3.7 特性互锁的本质原因（不只是"参数没适配"）

| 特性 | 为什么和前缀复用冲突 |
|---|---|
| **HiSparse** | 引入**第四套**引用计数索引空间 `full_to_hisparse_device_index_mapping`（`allocator/hisparse.py:278-330, 488-547`），且 c4 KV 由 P **直写 D 的 host DRAM**、跳过 GPU staging（`decode.py:409-463`）。前缀复用意味着"这段 c4 不会被写入"，但 host 池的逻辑分配与 device 池的换入换出都假设整段是新的。硬断言 `decode.py:1813-1816` `assert prefix_len == 0` |
| **投机解码** | draft 层有独立单层 SWA-only KV 池（`kv_cache_configurator.py:868-957`），ring 深度也翻倍（c4 8→16、c128 128→256，`deepseek_v4_memory_pool.py:34-48`）。前缀复用后 draft 池的对应行是空的，需要单独定义"draft KV 是否复用" |
| **DCP** | 要求 `decode_prefix_len % virtual_page_size == 0`（`disaggregation/common/utils.py:154-158`），且 MLA KV 按 `pos % dcp_size == rank` 分片存储，前缀在各 rank 上的本地长度不同，标量 `decode_prefix_len` 无法表达 |

---

## 4. 有利条件：为什么这件事是可做的

在动手之前，先把已经成立的前提列清楚 —— 它们决定了工作量是"补齐一层簿记"而不是"重做 KV 体系"。

| # | 有利条件 | 依据 |
|---|---|---|
| 1 | **压缩 KV 位置无关、块边界与 radix 页边界天然对齐** | §3.5，`compressor.py:335-380`；`page_size=256` 可被 4 / 128 整除 |
| 2 | **c4/c128/indexer 的 KV 行是 full 槽的纯函数，复用 full 槽即自动复用它们** | `deepseek_v4_memory_pool.py:660-700`；`allocator/swa.py:317-330` 无需为它们 free |
| 3 | **主 KV 传输通道的 item 索引与 full 页号 1:1 恒等**，`decode_prefix_len` 可直接用 | `get_contiguous_buf_infos()`，`c4_page_size = 256//4` |
| 4 | **非 PD（集中式）DSV4 已经在跑 `UnifiedRadixCache(FULL+SWA)` 并已复用前缀** —— DSV4 的 hook 既不强制 `disable_radix_cache` 也不强制统一树 | `arg_groups/deepseek_v4_hook.py`、`overrides.py:1359-1410`；`registry.py:143` 无条件走统一树 |
| 5 | **离线 ring 是 scratch 而非累加器**，跳过前缀不破坏语义 | `c4.cuh:41-62` |
| 6 | **NPU 已经有两个可直接照搬的实现范本** | `C128SidecarComponent`（`registry.py:167-176`）+ `dsv4_state_payloads(..., prefix_len=...)`（`decode.py:1447-1460`、`prefill.py:1269`）|

第 4 条尤其关键：**"DSV4 的压缩 KV 能不能被前缀复用"这个问题，集中式部署已经用生产流量回答了"能"。** PD 缺的只是"跨实例把这件事表达出来"的协议与簿记。

---

## 5. 改造方案

总原则：**先把 device-resident 的最小闭环做对做稳，再逐个解特性互锁。** 不要一次性打开所有维度。

### 阶段划分总览

```
P0  能力面收敛 + 前置门禁细化        （不改语义，先让"能跑的子集"能跑）
P1  状态通道 prefix-aware 化         （NPU 已有实现下沉为公共路径）
P2  压缩态 ring 的边界安全           （消掉 §3.3 隐患，本方案的正确性核心）
P3  radix 树补 DSV4 component        （簿记完备化，照搬 C128SidecarComponent）
P4  解特性互锁：MTP → HiSparse → DCP → HiCache
```

### P0：能力面收敛与门禁细化

目标：把 `kv_cache_builder.py:248-252` 那条**粗粒度 raise 换成细粒度 raise**，只挡真正不安全的组合。

1. 在 `kv_cache_builder.py` 里，把 DSV4 分支从"一律拒绝"改为按条件拒绝，需同时满足才放行：
   - `envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get()`（已有，`:235`）
   - `not enable_hierarchical_cache`（已有，`:241`）
   - `not envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()` —— **新增**，对应 §3.4 的递推冲突
   - `not enable_hisparse`、`speculative_algorithm.is_none()`、`dcp_size == 1` —— 这三条已在 `pd_disaggregation_hook.py:63-91` 拦下，此处只需保证错误信息指向 DSV4 语境，便于定位
2. 新增一条只对 DSV4 生效的断言：`page_size % max(compress_ratios) == 0`（CUDA 上 `256 % 128 == 0` 恒成立，NPU `page_size=128` 需单独确认），把"页边界即压缩块边界"这个隐含前提**显式化**，防止将来 page_size 变更时静默失效。
3. 顺手核对一个绕过风险：`is_deepseek_v4_arch` 在 `model_config.py:789-827` 是**嵌在 `if self.is_hybrid_swa:` 内部赋值**的，若 `disable_hybrid_swa_memory` 把 `is_hybrid_swa` 关掉，整个 `if is_hybrid_swa:` 块（含 DSV4 的 raise）都会被跳过。应把 DSV4 判定移出该嵌套，或在 `is_hybrid_swa == False` 分支补一条兜底 raise。

交付物：一个只在"离线压缩 + 无 HiSparse + 无投机 + 无 DCP + 无 L2/L3"下放行的窄口子，其余组合报错信息明确指出缺哪一项。

### P1：状态通道 prefix-aware 化

目标：让 CUDA 走上和 NPU 同一套「按 `prefix_len` 裁剪状态 payload」的逻辑。

1. 把 `hardware_backend/npu/dsv4/dsv4_common_hooks.py::dsv4_state_payloads` 的**平台无关部分**下沉到公共模块（建议 `disaggregation/utils.py`，与 `setup_state_kv_args`（`:966-1096`）同处），保留 NPU 特有部分作为 hook。
2. 去掉 `decode.py:1447-1460` 的 `_is_npu` 守卫，改为 `isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)` 单条件；P 侧对应 `prefill.py:1269` 的 `dsv4_state_payloads(..., prefix_len=...)` 同步放开。
3. 三条状态通道分别定义裁剪规则（**必须 P/D 严格同序，否则位置错位**，见 `disaggregation/utils.py:966-1096` 的 `StateType` 注册顺序）：

| 通道 | 索引 | 裁剪规则 |
|---|---|---|
| `StateType.SWA`（swa_kv_pool 页行）| SWA 页号 | 起点 = `page_align_floor(max(total_prefix_len, seq_len - window_size))`，经 `translate_loc_from_full_to_swa` 翻译；这就是现有 `_swa_payload()`（`decode.py:1387-1399`）的逻辑，本身已 prefix-aware |
| `StateType.SWA`（c4/indexer ring 块，`item_len = nbytes * ring_size`）| SWA 页号 | **不能简单按 `prefix_len` 砍**，见 P2 |
| `StateType.C128_STATE` | `req_pool_idx` | 离线：整块传输 + `clear_c128_req_state`（`decode.py:1420-1436`）保持现状即可；online：P0 已禁 |

4. 单测层面先加一个**纯 CPU 的 payload 一致性测试**：给定 `(seq_len, prefix_len, page_size, window_size, compress_ratios)`，断言 P 侧发送的 item 索引集合 == D 侧期望写入的 item 索引集合。这类测试不需要 GPU，能挡住绝大多数错位。

### P2：压缩态 ring 的边界安全（正确性核心）

目标：消掉 §3.3 的隐患 —— `create_paged_compress_data_kernel` 的 `load_pos` / `load_overlap_pos` 落在复用前缀里时，其 state 地址必须**可靠有效**。

先算清楚需要保护的位置。设 `cr = 4`（c4）、`P = total_prefix_len`：

```
load_pos          = ((P - 1) // cr) * cr        # 前缀最后一个压缩块的起点
load_overlap_pos  = load_pos - cr               # 再往前一个块
需要保证有效的最靠前位置 = load_overlap_pos
```

三条候选路线，**建议按 A → B 的顺序取舍**：

**方案 A（推荐，改动最小）—— 把边界页排除在"可复用前缀"之外**

调整 `_swa_tail_len`（`decode.py:459-480`）的 radix 分支，让 `swa_prefix_cap`（`decode.py:1190-1258`）额外多留一个压缩块的余量：

```
window_start = max(0, seq_len - 1 - max(window_size, page_size) - max_compress_ratio)
swa_tail_len = seq_len - page_align_floor(window_start)
```

由于 `page_size = 256`、`max_compress_ratio = 128`，多留的余量最多再吃掉一个页；对长前缀场景（数千 token）命中率损失可忽略（≤ 一页 = 256 token）。

代价：纯算术改动，不动 kernel，不动传输协议。**收益/风险比最好，应作为首选。**

**方案 B（彻底，但要动传输）—— 把边界 ring 块纳入传输**

在 P1 的状态通道裁剪里，对 c4/indexer ring 通道**不按 `prefix_len` 砍到底**，而是从 `load_overlap_pos` 对应的 SWA 页开始传：即 P 额外多发 1~2 个 ring 块。这样 D 端 kernel 读到的边界态一定是本次请求刚写入的，与 SWA 锁是否释放无关。

代价：需要在 `get_state_buf_infos()` 的 item 语义上引入"前缀边界块"的概念，P/D 两侧都要改；且 ring 块 item_len = `nbytes * ring_size`（整块），粒度较粗，带宽开销比方案 A 的"少复用一页"更大。

**方案 C（防御性，建议无论 A/B 都一并做）—— 让越界读变成显式失败**

把 `attn.py:87-215` 里两处 `other=0` 换成 `other=-1`，并在 debug 构建（或 `SGLANG_TEST_*` 门控）下断言 `swa_loc >= 0`。理由：当前的 `other=0` 会把"映射失效"这一严重错误伪装成"读第 0 行"，是这类静默精度 bug 的温床。这条改动独立于本方案的其余部分，**即使不做 decode radix 也值得做**。

> 注意：`other` 的语义变更会影响所有调用方（含集中式路径），需确认没有代码依赖"越界读 0"的行为。

### P3：radix 树补一个 DSV4 component

到 P2 为止已经能跑对，但簿记仍不完备：压缩态 ring 依旧不在树里，未来叠加 L2/L3、retraction、eviction 时会再次出现"树不知道 ring 状态"的问题。

范本就在 NPU：`registry.py:167-176` 在 `params.req_to_token_pool` 具备 `req_to_c128_sidecar` 时挂一个 `ComponentType.C128` / `C128SidecarComponent`。

做法：
1. 为 CUDA DSV4 定义一个 component（命名建议 `ComponentType.DSV4_COMPRESS`），职责是：
   - 在 `inc_lock_ref` / `dec_lock_ref` 时把"压缩态边界块所在的 SWA 页"一并纳入锁的保护范围 —— 这样方案 A 的算术余量可以由 component 语义替代，更不容易在后续改动中被破坏；
   - 在 eviction 时暴露 ring 的可回收量（离线 ring 是 scratch，可回收量恒为 0，但 online c128 不是，为将来放开 online 预留接口）；
   - 对 `match_prefix` 施加"前缀右端必须落在有效压缩块边界"的截断，与 SWA component 的滑窗感知截断（`swa_radix_cache.py:866`）同构。
2. component 只在 `is_deepseek_v4_arch and not online_compress` 时挂载，保持默认路径零开销。

这一阶段可以**推迟**，不阻塞 P0-P2 的上线；但如果预期很快要接 L2/L3（`pd_decode_dsv4_hicache_reuse_scheme.md`），建议提前做，否则会重复劳动。

### P4：逐个解特性互锁（按业务优先级排序）

| 顺序 | 特性 | 关键工作 | 难度 |
|---|---|---|---|
| 1 | **投机解码（MTP/EAGLE）** | draft 有独立单层 SWA-only 池（`kv_cache_configurator.py:868-957`）。最简做法：**draft KV 一律不复用前缀**，即 `decode_prefix_len` 只作用于 target 池，draft 池仍从 0 填。因 draft 只有 1 层且只用 SWA，重算成本低。需核对 ring 深度翻倍（`deepseek_v4_memory_pool.py:34-48`）下的地址算术 | 中 |
| 2 | **HiSparse** | 需引入第四套索引空间的前缀语义：host 池的逻辑分配要支持"前缀段沿用已有 host 页"，device 小池的 refcount 也要跟着。同时 `decode.py:1813-1816` 的 `assert prefix_len == 0` 要替换为真实实现。这是四项中最重的 | 高 |
| 3 | **DCP** | 把线上协议从标量 `decode_prefix_len` 扩展为"全局前缀长度 + 各 rank 本地长度"，或强制 `decode_prefix_len % (page_size * dcp_size) == 0` 使各 rank 本地长度可由全局长度推出。后者改动小得多，建议先做 | 中 |
| 4 | **HiCache L2/L3** | 前置是为 DSV4 池实现 `get_cpu_copy` / `load_cpu_copy`（六子池都要），这本身是独立大工作项，见 `pd_decode_dsv4_hicache_reuse_scheme.md` | 高 |

---

## 6. 验证方案

正确性验证的难点在于：**这类 bug 不崩溃，只掉精度。** 因此必须有"逐位对拍"级别的手段，不能只看 benchmark 分数。

### 6.1 逐位对拍（最强、也最应该先做）

参照 R3（return routed experts）CI 的做法（`test/registered/.../test_return_routed_experts.py:44` 的 baseline/reference 对拍思路）：

- **baseline**：D 实例 `--disable-radix-cache`（全量传输）
- **reference**：D 实例开 `--disaggregation-decode-enable-radix-cache`
- 同一批请求、贪心解码（`temperature=0`），断言**输出 token 序列逐位相等**
- 关键是构造**必然命中前缀**的请求集：同一个长 prompt 连发两次，第二次必须复用第一次的前缀

再加一组针对边界的用例：让 `prefix_len` 落在 `page_size` 的整数倍附近（`P = 256`、`512`、`768`），以及落在 `128` 的整数倍附近（c128 块边界），专门打 §3.3 的 `load_overlap_pos`。

### 6.2 payload 索引集合对拍（纯 CPU，快）

见 P1 第 4 点。这是唯一能在没有多机 GPU 环境时跑的测试，应作为 CI 常驻。

### 6.3 KL / logprob 一致性

用仓库已有的 prefill-vs-decode logprob 一致性测试框架（见 `kl-consistency-test` skill 的方法论）：前缀复用路径与全量路径的 logprob 分布 KL 应在阈值内。注意这类测试**只能证伪不能证明**——KL 小不代表 ring 边界没问题，因为错误只在特定 `prefix_len` 对齐下触发，所以 6.1 优先级更高。

### 6.4 CI 注册位置

已有两个同族测试可直接照搬结构：

```
test/registered/disaggregation/test_disaggregation_decode_radix_cache.py       # 普通模型
test/registered/disaggregation/test_disaggregation_decode_radix_cache_swa.py   # SWA 混合模型（#27770 配套）
→ 新增 test_disaggregation_decode_radix_cache_dsv4.py
```

DSV4 需要的卡数较多，注意放入合适的 GPU 分组（参考 `test/registered/4-gpu-models`、`8-gpu-models` 的组织方式），并在 `test/README.md` 的 CI 布局里登记。

### 6.5 性能验证

decode radix 的收益是**省 P 的算力和网络**，所以指标要看 P 侧：

- P 侧 prefill token 吞吐（应下降 ≈ 前缀命中率）
- KV 传输字节数（应下降同比例）
- **D 侧 TTFT 不应变差** —— 前缀匹配、加锁、`_pre_alloc` 的前缀行写入都在 D 的准入路径上，是新增的同步开销
- 前缀命中率本身（需要新增指标；现有 `pd_disaggregation` 指标里没有 decode 侧命中率）

---

## 7. 明确标注的未验证项

本文全部结论来自**静态代码阅读**，未启动服务、未执行任何测试。以下几条尤其需要实机确认：

1. **§3.3 的核心隐患是否真会触发**：`create_paged_compress_data_kernel` 在复用前缀边界读 `load_overlap_pos` 时，`full_to_swa_index_mapping` 表项是否真的可能已失效。也可能现有的 `swa_prefix_cap` + 页对齐余量恰好已经把它兜住 —— 若如此，P2 方案 A 就退化为"把隐含保证显式化"，改动更小。
2. **NPU 的 `dsv4_state_payloads(prefix_len=...)` 是否已解决同一问题**，还是 NPU 的 `page_size=128` / 不同 ring 布局让它绕开了。这决定 P1 能否直接下沉复用。
3. **`is_deepseek_v4_arch` 在 `disable_hybrid_swa_memory` 下是否真会被绕过**（§P0 第 3 点），需要实际构造该参数组合确认。
4. **P2 方案 A 的命中率损失量级**，需要用真实多轮负载测量，而非按最坏情况估算。

---

## 8. 关键代码索引（速查）

| 主题 | 文件:行 |
|---|---|
| DSV4 拦截 raise | `mem_cache/kv_cache_builder.py:248-252` |
| 同族其余门禁 | `mem_cache/kv_cache_builder.py:230-262` |
| 拦截来历 | commit `978244d671`（PR #27770，2026-08-20）|
| PD decode 模式其他互斥门 | `arg_groups/pd_disaggregation_hook.py:63-111` |
| HiSparse 与 radix 互斥 | `arg_groups/hisparse_hook.py:101-103`、`disaggregation/decode.py:1813-1816` |
| retraction host 备份的 flag 依赖 | `mem_cache/kv_cache_builder.py:144-157` |
| 统一树 component 组装 | `mem_cache/registry.py:143-176` |
| NPU C128 sidecar component（范本）| `mem_cache/registry.py:167-176` |
| D 侧前缀匹配加锁 | `disaggregation/decode.py:645-661` |
| D 侧准入 / prefix_cap / 放 SWA 锁 | `disaggregation/decode.py:1190-1258` |
| `_swa_tail_len` radix 分支 | `disaggregation/decode.py:459-480` |
| SWA payload | `disaggregation/decode.py:1387-1399` |
| c128 state payload + clear | `disaggregation/decode.py:1420-1436` |
| **NPU-only** prefix-aware DSV4 state payload | `disaggregation/decode.py:1447-1460` |
| `decode_prefix_len` 上线 | `disaggregation/decode.py:1496` |
| `_pre_alloc` | `disaggregation/decode.py:1746-1884` |
| P 侧按 prefix 砍单 | `disaggregation/prefill.py:344-349` |
| P 侧 DSV4 state payload | `disaggregation/prefill.py:1269` |
| 传输契约（三条通道）| `mem_cache/deepseek_v4_memory_pool.py:660-840` |
| full→SWA 翻译 | `mem_cache/deepseek_v4_memory_pool.py:671-673` |
| ring 深度常量 | `mem_cache/deepseek_v4_memory_pool.py:34-48` |
| `clear_c128_req_state` | `mem_cache/deepseek_v4_memory_pool.py:1007-1024` |
| `StateType` 注册 | `disaggregation/utils.py:966-1096` |
| **压缩态地址 kernel（隐患所在）** | `python/sglang/kernels/ops/attention/dsv4/attn.py:87-215` |
| 压缩 KV 定义 / `ape` | `layers/attention/dsv4/compressor.py:335-380` |
| 离线 ring 布局 | `c4.cuh:41-62` |
| online c128 递推 | `c128_online.cuh:120-192` |
| `is_deepseek_v4_arch` 赋值 | `configs/model_config.py:789-827` |
| DSV4 参数 hook（不强制关 radix）| `arg_groups/deepseek_v4_hook.py`、`overrides.py:1359-1410` |
| SWA 分配器 free（不管压缩池）| `mem_cache/allocator/swa.py:317-330` |
| HiSparse 第四套索引 | `mem_cache/allocator/hisparse.py:278-330, 488-547` |
| `get_cpu_copy` 未实现 | `mem_cache/memory_pool.py:1776-1780`、`deepseek_v4_memory_pool.py:245` |
| radix 页对齐插入 | `mem_cache/radix_cache.py:150-154`；`mem_cache/common.py:69-71` |
| DCP 前缀页对齐要求 | `disaggregation/common/utils.py:154-158` |
| draft 独立 KV 池 | `mem_cache/kv_cache_configurator.py:868-957` |

