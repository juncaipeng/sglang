# DeepSeek-V4 PD 分离：P → D 到底传了什么、怎么传的

> 面向读者：需要排查 DSV4 PD 分离传输问题、或要给 DSV4 新增/修改某个 KV 子池的工程师。
> 全文结论均来自代码逐行核对，关键处标注 `文件:行号`。

## 0. 阅读指引：与既有文档的分工

| 文档 | 讲什么 | 与本文关系 |
|---|---|---|
| `dsv4_pd_disaggregation_request_lifecycle.md` | 请求在 P/D 两侧的**队列与状态机全生命周期**（三队列/四阶段、握手、retract、kernel 分派） | 本文不重复；需要"请求什么时候被传"看它 |
| `pd_disaggregation_kv_transfer_architecture.md` | 通用（非 DSV4）PD KV 传输框架 | 本文是它在 DSV4 上的特化 |
| `deepseek_v4_cache_management.md` | DSV4 六个子池的**结构与容量**（谁多大、怎么算） | 本文只用它的结论，聚焦"哪些字节被搬走" |
| **本文** | **传输内容清单 + 传输方法 + 索引数学** | 单点深挖 |

## 1. 一分钟结论

P 向 D 传输的东西分四类，走两个平面：

```
数据平面（RDMA 单边写，P 主动写 D 的显存/内存）
 ├─ 主 KV 通道   kv_data_ptrs      : c4 KV 页、c4 indexer K(+scale) 页、c128 KV 页
 ├─ 状态通道     state_data_ptrs   : SWA 全精度近窗 KV 页、c4 压缩态 ring 块、
 │                                   c4 indexer 压缩态 ring 块、(unified) SWA ring 行、
 │                                   c128 压缩态（请求级）、(可选) NextN draft 池
 └─ aux 通道     aux_data_ptrs     : 首个输出 token、cached_tokens/多模态计数、
                                     logprobs/top-logprobs、(投机) topk_p/topk_index/hidden_states、
                                     bootstrap_room

控制平面（ZMQ，D 主动推给 P）
 ├─ KVArgsRegisterInfo : 每会话一次，D 的各通道基址 + item_len + 拓扑
 └─ TransferInfo       : 每请求一次，D 的目的索引列表（dst_kv_indices / dst_state_indices / dst_aux_index）
    ← 反向：P 把 KVPoll 状态同步回 D
```

方法四步：**两侧注册缓冲 → D 反向推送目的地址 → P 计算源索引并 `batch_transfer_sync` 单边写 → P 回同步状态**。
所有写入地址都是同一个公式：`addr = base_ptr + index * item_len`，`index` 的语义按通道不同（页号 / ring 块号 / 请求槽号）。

## 2. 前置：DSV4 的池子长什么样（只取传输相关的最小结论）

DSV4 每层有一个压缩比 `compress_ratio ∈ {0, 4, 128}`（HF config `compress_ratios` 逐层给出）：

| ratio | 别名 | 该层需要的存储 |
|---|---|---|
| 0 | 纯 SWA | 只有 SWA 近窗全精度 KV |
| 4 | CSA | SWA 近窗 + c4 压缩 KV + c4 lightning indexer K + 两类压缩态 ring |
| 128 | HCA | SWA 近窗 + c128 压缩 KV + c128 压缩态 |

`DeepSeekV4TokenToKVPool`（`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`）里的子池：

```
swa_kv_pool                  每层一个 buffer（全 43 层）  ← 状态通道 SWA
c4_kv_pool                   ratio==4 的层                ← 主通道
c4_indexer_kv_pool           ratio==4 的层（index K+scale）← 主通道
c128_kv_pool                 ratio==128 的层              ← 主通道
compress_state_pools         attention 压缩态 ring（4/128）← ratio4 走 SWA 组件，ratio128 走 C128_STATE
indexer_compress_state_pools indexer 压缩态 ring（仅 4）   ← 状态通道 SWA
```

**关键页大小关系**（`deepseek_v4_memory_pool.py:558-574`）：

```python
assert page_size % swa_page_size == 0
self.swa_window_size = self.swa_page_size = swa_page_size   # = get_schedule().page_size
self.c4_page_size   = page_size // 4                        # 256 // 4   = 64
self.c128_page_size = page_size // 128                      # 256 // 128 = 2
self._unified_kv    = is_unified_kv_triton()
```

每个子 KV 池按 `[(size + page_size + 1) // page_size, page_bytes]` 分配 —— **一行 = 一页**。
这是后面所有 `item_len = buf[0].nbytes` 的由来，也是"页号可以跨子池恒等复用"的根因（§8.2 证明）。

还有一条 `unified_kv` 变体路径（`is_unified_kv_triton()` 为真）：此时
`swa_kv_pool / c4_kv_pool / c128_kv_pool` 全为 `None`，每层只有一个 `unified_kv_pool.kv_buffer[l]`，
前 `swa_pages` 行是 SWA ring、其后是压缩区。两种布局的传输组件构成不同，务必区分（§3.5）。

## 3. 传输内容总清单

### 3.1 主 KV 通道（`kv_data_ptrs` / `kv_data_lens` / `kv_item_lens`）

来源：`DeepSeekV4TokenToKVPool.get_contiguous_buf_infos()`（`deepseek_v4_memory_pool.py:675-728`）。
非 unified 分支把三组 buffer **按固定顺序**拼成扁平列表：

```python
buf_groups = [
    self.c4_kv_pool.kv_buffer,                              # 每层 [pages, page_bytes]
    self.c4_indexer_kv_pool.index_k_with_scale_buffer,       # 每层
    self.c128_kv_pool.kv_buffer,                            # 每层
]
for bufs in buf_groups:
    for buf in bufs:
        data_ptrs.append(buf.data_ptr())
        data_lens.append(buf.nbytes)
        item_lens.append(buf[0].nbytes)      # 一行 = 一页
```

于是扁平列表长度 `= 2 * c4_layer_num + c128_layer_num`，顺序恒为 `[c4..., c4_indexer..., c128...]`。
**这个顺序是契约**：PP 切片（`common/conn.py:_mla_slice_ptrs_for_pp`）和 HiSparse 的
`dst_kv_ptrs[c4_layer_num:]` 都依赖它，改动顺序会静默写错地址。

| 内容 | 层范围 | 一个 index 代表 | item_len |
|---|---|---|---|
| c4 压缩 KV（FP8） | ratio==4 各层 | 一个 c4 页（= 一个 full 页对应的压缩页） | `buf[0].nbytes` |
| c4 lightning indexer K + scale | ratio==4 各层 | 同上 | `buf[0].nbytes` |
| c128 压缩 KV | ratio==128 各层 | 一个 c128 页 | `buf[0].nbytes` |

注意 **`item_len` 不乘 `page_size`**（这与普通 MLA 池不同）——因为 DSV4 一行已经就是一整页。

### 3.2 状态通道组件一：`StateType.SWA`

来源：`get_state_buf_infos()`（`deepseek_v4_memory_pool.py:785-812`），一个组件内部又是扁平列表：

```python
for buf in self.swa_kv_pool.kv_buffer:            # 全 43 层
    ptrs.append(buf.data_ptr()); item_lens.append(buf[0].nbytes)      # 一页
for pool in (*self.compress_state_pools, *self.indexer_compress_state_pools):
    if pool.ratio == 128:
        continue                                   # ★ c128 attn 压缩态另走 C128_STATE
    t = pool.buffer
    ptrs.append(t.data_ptr()); item_lens.append(t[0].nbytes * pool.ring_size)   # 一整个 ring 块
```

扁平列表长度 `= swa_layer_num(=43) + 2 * c4_layer_num`，顺序 `[swa..., c4_compress_state..., c4_indexer_compress_state...]`
（与 `common/conn.py:956-1031` 的 PP 切片假设一致）。

| 子项 | 一个 index 代表 | item_len |
|---|---|---|
| SWA 近窗全精度 KV（每层） | 一个 **SWA 页号** | 一页字节数 |
| c4 attention 压缩态 | 一个 **ring 块**（`ring_size` 行） | `row_bytes * ring_size` |
| c4 indexer 压缩态 | 同上 | `row_bytes * ring_size` |

压缩态是"在途状态"：`CompressStatePool`（`deepseek_v4_compress_state.py:84-218`）是扁平
`torch.empty((size, last_dim))` ring buffer，离线 `ring_size` c4=8（投机 16），
所以传一个 ring 块 = 把该 SWA 页对应的整段在途压缩状态一次搬走。

### 3.3 状态通道组件二：`StateType.SWA_RING`（仅 unified 布局）

来源 `get_unified_swa_ring_buf_infos()`（`deepseek_v4_memory_pool.py:730-746`）：取每层 unified buffer
的 `[0, swa_pages)` 区域，**`item_len = row_bytes`（按行，不按页也不按 ring 块）**。
非 unified 布局下该组件不存在。

### 3.4 状态通道组件三：`StateType.C128_STATE`

来源 `get_c128_state_buf_infos()`（`deepseek_v4_memory_pool.py:814-828`）：只收 `ratio == 128` 的
attention 压缩态池，长度 `= c128_layer_num`。

```python
item_len = t[0].nbytes if ONLINE_C128 else t[0].nbytes * 128
```

它是**请求级**而非页级的（§8.5）：在线模式 `ring_size == 1`，一个请求一行；
离线模式一个请求占 `ring_size // 128` 块。只有 `seq_len % 128 != 0` 时才需要传
（`disaggregation/utils.py:82-98`），因为整块边界上压缩态已落盘到 c128 KV、无在途残留。

### 3.5 两种布局下状态通道的组件构成

`setup_state_kv_args()`（`disaggregation/utils.py:966-1241`）是唯一装配点，对 DSV4 依次：

| 顺序 | 组件 | 非 unified | unified |
|---|---|---|---|
| 1 | `SWA` | ✅ `get_state_buf_infos()` | ✅（内容为空/退化，见代码） |
| 2 | `SWA_RING` | ❌ | ✅ `get_unified_swa_ring_buf_infos()` |
| 3 | `C128_STATE` | ✅（有 c128 层时） | ✅（有 c128 层时） |
| 4 | draft（`SWA`/`SWA_RING`） | ✅（开 MTP 时） | ✅（开 MTP 时） |

**P/D 两侧必须产生完全相同的 `state_types` 顺序**，因为 `TransferInfo.dst_state_indices` 是
按位置对齐的 `List[List[int]]`，顺序错位 = 写到别的组件里。

### 3.6 draft / NextN 组件（开 MTP 时）

`setup_state_kv_args` 尾部：当 target 与 draft 都是 `DeepSeekV4TokenToKVPool` 时，校验
draft 是 **SWA-only**（`all(ratio == 0)`）、`_unified_kv` 模式一致、ring 几何一致、
共享同一份 `full_to_swa_index_mapping`，然后**追加一个独立组件**（`SWA_RING` 或 `SWA`）。

这与 MTP 的池设计吻合：draft 是全新的单层 SWA-only `DeepSeekV4TokenToKVPool` 实例
（`kv_cache_configurator.py:868-957`），没有 c4/c128/indexer，所以只需要多一个 SWA 类组件。
主通道那侧也会把 draft 池的 `get_contiguous_buf_infos()` 拼在 target 之后
（`prefill.py:_init_kv_manager` 191-297）。

### 3.7 aux / metadata 通道（`aux_data_ptrs`）

来源 `MetadataBuffers.get_buf_infos()`（`disaggregation/utils.py:297-566`），buffer 在 **CPU**（或 npu/cuda 变体），
每个请求占一个 `metadata_buffer_index` 槽。顺序即传输顺序：

| buffer | shape | 内容 |
|---|---|---|
| `output_ids` | `(size, 16)` int32 | **prefill 产出的第一个 token**（PD 的关键交接物） |
| `cached_tokens` | `(size, 16)` int32 | slot 0-3 = cached/device/host/storage 命中量；4-6 = image/audio/video 多模态 token 数 |
| `output_token_logprobs_val / _idx` | `(size, 16)` | 首 token logprob |
| `output_top_logprobs_val / _idx` | `(size, 128)` | top-logprobs（`max_top_logprobs_num=128`） |
| 采样 mask 三件套 | 可选 | 开启时才注册 |
| `output_topk_p` / `output_topk_index` | 投机用 | 草稿采样分布 |
| `output_hidden_states` | `(size, hidden_size)` | 投机用：MTP 的种子 hidden（DSV4+MTP 必需） |
| `output_dsa_topk_indices` | 可选 | DSA 变体 |
| `bootstrap_room` | `(size, 8)` | 校验/对齐用 |

写入时机：**只在最后一个 chunk**，`send_kv_chunk` 里 `self.disagg_metadata_buffers.set_buf(req)`
（`prefill.py:1131-1318`），传输由 `send_aux`（`mooncake/conn.py:1084-1106`）完成。

### 3.8 控制平面（ZMQ，不是 RDMA）

| 消息 | 方向 | 频次 | 载荷 |
|---|---|---|---|
| `KVArgsRegisterInfo`（`mooncake/conn.py:129-195`） | D → P | 每会话一次（`room == "None"` 特殊包） | `dst_kv_ptrs`、`dst_aux_ptrs`、`dst_state_data_ptrs`、`dst_kv_item_len`、`dst_state_item_lens`、`dst_state_dim_per_tensor`、`dst_kv_layer_ids`、`dst_state_layer_ids`、`dst_tp_rank`/`dst_attn_tp_size`、staging 基址/大小、dcp size/rank |
| `TransferInfo`（`mooncake/conn.py:79-124`） | D → P | 每请求一次 | `room`、`endpoint`/`dst_port`、`mooncake_session_id`、`dst_kv_indices`(int32)、`dst_aux_index`、`dst_state_indices: List[List[int]]`、`required_dst_info_num`、`is_dummy`、`decode_prefix_len`、`dst_device_kv_indices`(HiSparse) |
| `KVPoll` 状态 | P → D | 传输完成/失败 | `sync_status_to_decode_endpoint` |

### 3.9 明确**不传**的东西

- **主模型 43 层的 hidden states**（除 MTP 需要的最后一层种子 hidden 外）——不传。
- **模型权重**：P/D 各自加载。
- **radix cache 树结构**：D 侧 PD 预分配路径 `assert prefix_len == 0`，不复用 P 的前缀结构。
- **routed experts（R3）路由记录**：`disaggregation/` 全目录无相关传输，只透传参数 —— 因此
  PD 下 prompt 段的 routed experts 不可靠（见 MEMORY 的 R3 条目）。
- **c128 在整块边界上的压缩态**：`seq_len % 128 == 0` 时 index 列表为空，直接跳过。

## 4. 方法总览：四步

```
                 P 侧                                        D 侧
 ┌──────────────────────────────────┐          ┌──────────────────────────────────┐
 │ ① _init_kv_manager               │          │ ① _init_kv_manager               │
 │   组装 KVArgs（kv/state/aux）    │          │   组装 KVArgs（同顺序！）        │
 │   register_buffer_to_engine      │          │   register_buffer_to_engine      │
 └──────────────────────────────────┘          └──────────────────────────────────┘
              ▲                                              │
              │  ② ZMQ: KVArgsRegisterInfo（每会话一次）      │
              │◄─────────────────────────────────────────────┤
              │  ② ZMQ: TransferInfo（每请求，含目的索引）     │
              │◄─────────────────────────────────────────────┤
 ┌────────────┴─────────────────────┐                        │
 │ ③ send_kv_chunk 算源索引          │                        │
 │   → sender.send                   │   RDMA 单边写           │
 │   → add_transfer_request(分片入队) │ ═════════════════════► │ D 的显存/内存
 │   → transfer_worker               │  batch_transfer_sync    │ （D 不参与 CPU 工作）
 │     send_kvcache / maybe_send_extra│                        │
 │     / send_aux                     │                        │
 └────────────┬─────────────────────┘                        │
              │  ④ ZMQ: sync_status_to_decode_endpoint       │
              ├─────────────────────────────────────────────►│ KVPoll.Success
```

要点：**D 是完全被动的目标**，所有地址计算和写操作都在 P。D 唯一的主动动作是把
"你该往哪写"（指针 + 索引）通过 ZMQ 反向推给 P。

## 5. 第一步：注册（P/D 对称）

### 5.1 装配 `KVArgs`

`prefill.py:_init_kv_manager`（191-297）/ `decode.py:_init_kv_manager`（508-606）几乎镜像：

```python
kv_args.engine_rank / pp_rank / system_dp_rank / kv_cache_dtype_str
kv_args.prefill_start_layer, kv_args.prefill_end_layer            # PP 切片依据
kv_args.kv_data_ptrs, kv_data_lens, kv_item_lens = token_to_kv_pool.get_contiguous_buf_infos()
#   + 开 MTP 时 concat draft 池的同名三件套
kv_args.aux_data_ptrs, ... = metadata_buffers.get_buf_infos()
setup_state_kv_args(kv_args, target_kv_pool=..., draft_kv_pool=...)   # 装状态通道
# DSV4 专属：
kv_args.mla_compression_ratios = list(self.token_to_kv_pool.compression_ratios)
```

`mla_compression_ratios` 是 DSV4 特有字段（`base/conn.py:1-120` 有定义），存在的唯一理由是：
主通道/状态通道是**按压缩桶而非按层**组织的扁平列表，PP 阶段切指针时必须知道
"前 k 层里有几层是 ratio4、几层是 ratio128"才能算出前缀偏移（§10）。

`is_mla_backend`（`utils.py:746-750`）对 `DeepSeekV4TokenToKVPool` 返回 True，
所以走"每层单 buffer 扁平列表"路径（而不是 MHA 的 K/V 双 buffer 路径）。

### 5.2 向 RDMA 引擎注册内存

`_registerable_regions` / `register_buffer_to_engine`（`mooncake/conn.py:288-322`）：
把 kv + aux + **所有 state 组件**的 `(ptr, len)` 去重后一次 `engine.batch_register(ptrs, lens)`。
去重是必要的：unified 布局下主通道指针是同一 buffer 的偏移，SWA_RING 指向同一 buffer 起点。

漏注册的表现是 RDMA 写返回非零 rc → 会话被拉黑（`transfer_worker` 里 session-failure blacklisting），
所以**新增子池时必须同时改 `get_*_buf_infos` 和 `setup_state_kv_args`**，否则要么不传、要么写失败。

## 6. 第二步：反向协商

1. **HTTP 服务发现**：P 启动向 bootstrap server `PUT /route` 注册；D `GET /route` 拿拓扑并算出
   自己该配哪个 P rank（细节见 `dsv4_pd_disaggregation_request_lifecycle.md` §3）。
2. **一次性指针注册**：D 用 `room = "None"` 的特殊 ZMQ 多帧包发 `KVArgsRegisterInfo`
   （帧序见 `mooncake/conn.py:_register_kv_args` 2373-2442）。
3. **每请求目的索引**：D 预分配好六个子池的槽位后，`send_metadata`
   （`decode.py:1352-1519`）把 `page_indices` / `metadata_buffer_index` / `state_indices` 推给 P：

```python
page_size = allocator.page_size
kv_transfer_page_size = page_size                       # HiSparse 会覆盖
page_indices = kv_to_page_indices(kv_indices, kv_transfer_page_size).astype(np.int32)
# state_indices 用与 P 完全相同的 payload 闭包计算，只是用 D 自己的 req_pool_idx
if StateType.C128_STATE in state_types:
    clear_c128_req_state(req_pool_idx)                   # 复用槽前先清，避免读到前一个请求残留
kv_receiver.send_metadata(page_indices, metadata_buffer_index, state_indices, **metadata_kwargs)
```

**P/D 用同一套 payload 计算逻辑、各自代入自己的 `req_pool_idx`** —— 这是保证
"源第 i 个 index ↔ 目的第 i 个 index" 一一对应的机制。

## 7. 第三步：P 侧写入全链路

### 7.1 `send_kv_chunk`（`prefill.py:1131-1318`）

```python
# ① 决定本 chunk 的 token 边界：非 last_chunk 必须页对齐（staging 时按 staging grid 对齐）
end_idx = ...        # 页对齐
# ② 只有最后一个 chunk 才写 aux + 算 state_indices
if last_chunk:
    self.disagg_metadata_buffers.set_buf(req)
    payloads = {
        StateType.MAMBA:      _mamba_payload,
        StateType.SWA:        _swa_payload,
        StateType.SWA_RING:   _swa_ring_payload,
        StateType.C128_STATE: _c128_state_payload,
        ...
    }
    state_indices = [payloads[st]() if st in payloads else None for st in state_types]
# ③ 分段发送
for seg_start, seg_end in segments:
    kv_indices  = req_to_token[req_pool_idx, seg_start:seg_end]
    kv_indices  = translate_kv_indices_for_transfer(kv_indices)     # DCP 等变体
    page_indices = kv_to_page_indices(kv_indices, page_size)
    req.disagg_kv_sender.send(
        page_indices,
        state_indices if segment_is_last else None,
        num_kv_tokens=...,
    )
```

一个易踩的细节：多数 payload 用 `seq_len = min(req.extend_range.end, transfer_input_len)`，
但 **c128 用 `c128_seq_len = transfer_input_len`**（不夹 `extend_range`）——因为 c128 压缩态是
按整条序列位置对 128 取模判断的，用 chunk 内的局部长度会算出错误的 ring 块。

发送闸门 `should_send_kv_chunk(num_pages, last_chunk) = num_pages > 0 or last_chunk`
（`common/conn.py:1238-1239`）：即使本 chunk 没凑满一页，last_chunk 也必须发（要带 state + aux）。

### 7.2 入队与 worker

`add_transfer_request`（`mooncake/conn.py:2134-2184`）按 `sum(session port) % len(transfer_queues)`
把请求分片到多个后台队列；`transfer_worker`（1573-1925）出队 `TransferKVChunk`，对每个目的 rank：

```python
选择路径：send_kvcache_dcp / send_kvcache / staging / send_kvcache_slice
if is_last_chunk:
    maybe_send_extra(...)      # 状态通道所有组件
    send_aux(...)              # metadata
polls.append(rc)
if len(polls) == req.required_dst_info_num:      # 所有目的 rank 都写完
    更新 KVPoll 状态 + sync_status_to_decode_endpoint(每个 rank)
```

### 7.3 地址计算：唯一的公式

`_send_kvcache_generic`（`mooncake/conn.py:617-762`）：

```python
prefill_blocks, dst_blocks = group_concurrent_contiguous(prefill_data_indices, dst_data_indices)
# 把连续 index 段合并成大块，减少 RDMA descriptor 数量

def set_transfer_blocks(src_ptr, dst_ptr, item_len):
    for p_block, d_block in zip(prefill_blocks, dst_blocks):
        src_addrs.append(src_ptr + int(p_block[0]) * item_len)
        dst_addrs.append(dst_ptr + int(d_block[0]) * item_len)
        lengths.append(item_len * len(p_block))
```

然后所有层一次性 `engine.batch_transfer_sync(session_id, src_addrs, dst_addrs, lengths)`
（`_transfer_data` 608-615）；只有 `enable_custom_mem_pool` 时才拆开逐层发。

**这一个公式覆盖全部通道**，通道之间的差别只在 `(ptr, index 语义, item_len)` 三元组 —— 这正是 §8 的内容。

### 7.4 `maybe_send_extra`（`mooncake/conn.py:1222-1420`）

按 `state_types` 顺序逐组件处理：

- `indices is None` → 跳过（例如非 unified 布局下的 SWA_RING）。
- `C128_STATE` 且源/目的 index 列表都空 → 跳过（`seq_len % 128 == 0` 的正常情况）。
- `_requires_exact_state_index_match = {SWA_RING, C128_STATE}`（1218-1220）：源/目的**长度必须相等**，
  不等直接抛异常。其余类型长度不等则**截断并打 warning**。
- 最终仍然复用 `_send_kvcache_generic(..., state_type=st)`。

判定"这个组件能不能走通用 KV 路径"的白名单 `_is_generic_kvcache_state_type`（1206-1216）
= `{SWA, DSA, SWA_RING, C128_STATE, BLOCK_SCALE, BLOCK_SCALE_SWA}`，DSV4 用到的四种全在内。

### 7.5 `send_aux`（`mooncake/conn.py:1084-1106`）

```python
for ptr, item_len, dst_ptr in zip(aux_data_ptrs, aux_item_lens, dst_aux_ptrs):
    src = ptr     + item_len * prefill_aux_index
    dst = dst_ptr + item_len * req.dst_aux_index
```

一次 `batch_transfer_sync` 搬完所有 metadata buffer；RDMA 不可用时退化到 `send_aux_tcp`。

## 8. 核心：索引数学（DSV4 最容易搞错的地方）

### 8.1 总对照表

| 通道 / 组件 | index 语义 | index 从哪来 | item_len |
|---|---|---|---|
| 主通道（c4 / c4_indexer / c128） | **full 页号**（直接复用！） | `kv_to_page_indices(req_to_token[...], page_size)` | `buf[0].nbytes`（一页） |
| `SWA` 之 swa_kv_pool | **SWA 页号** | `translate_loc_from_full_to_swa` 后再 `kv_to_page_indices` | 一页 |
| `SWA` 之 c4 压缩态 | **ring 块号** = SWA 页号 | 同上（页号即块号） | `row_bytes * ring_size` |
| `SWA_RING`（unified） | **行号** | `state_slot * ring_stride + positions % ring_stride` | `row_bytes` |
| `C128_STATE` | **请求级槽号** | `get_dsv4_c128_state_indices` | 在线 `row`；离线 `row * 128` |
| aux | metadata 槽号 | `metadata_buffer_index` | 单条 metadata 字节 |

### 8.2 为什么主通道能直接复用 full 页号（恒等映射）

这是 DSV4 传输代码看起来"没做任何压缩比换算"的原因，值得单独证明。

设 full 页大小 `P = 256`。第 `t` 个 token 的 full 页号 `pf = t // P`。
c4 池的页大小是 `P4 = P // 4 = 64`，而 c4 池里存的是压缩后的 token（4:1），
所以第 `t` 个 full token 对应第 `t // 4` 个 c4 token，其 c4 页号：

```
pc4 = (t // 4) // P4 = (t // 4) // (P // 4) = t // P = pf      ✅ 恒等
```

同理 c128：`pc128 = (t // 128) // (P // 128) = t // P = pf`。

**结论**：只要子池的 `page_size` 是 `full page_size` 除以自己的压缩比，
一个 full 页号就能同时寻址主通道的所有子池，无需任何翻译表。
这也解释了为什么 `page_size` 被硬约束为 256（`arg_groups/deepseek_v4_hook.py`）——
它必须能被 128 整除，否则 `c128_page_size` 退化为 0，映射崩掉。

同时注意 `deepseek_v4_memory_pool.py:558` 的 `assert page_size % swa_page_size == 0`：
SWA 那边不满足恒等（SWA 池是"近窗环形"，物理槽位与 full 槽位无线性关系），
所以 SWA 必须走显式翻译表。

### 8.3 SWA 页号：显式翻译

```python
def translate_loc_from_full_to_swa(self, kv_indices):        # :671-673
    return self.full_to_swa_index_mapping[kv_indices]
```

`_swa_payload`（`prefill.py`）三步：
1. 把窗口起点 floor 到页边界（`seq_len - window_size` 向下取整到 `page_size` 倍数）；
2. `translate_loc_from_full_to_swa(kv_indices)` 逐槽翻译；
3. `kv_to_page_indices(...)` 转页号。

只发窗口内的页 —— 窗口外的 SWA KV 在 D 侧永远读不到，传了是浪费带宽。

### 8.4 c4 压缩态：ring 块寻址

```python
# deepseek_v4_compress_state.py: translate_from_swa_loc_to_state_loc
swa_pages = swa_loc // self.swa_page_size
state_loc = swa_pages * self.ring_size + (swa_loc % self.ring_size)
```

而组件的 `item_len = row_bytes * ring_size`。两者一乘：

```
addr = ptr + index * (row_bytes * ring_size)
```

若 `index` 直接取 SWA 页号，则 `addr = ptr + swa_pages * ring_size * row_bytes` ——
正好落在该 SWA 页对应 ring 块的起点，一次搬 `ring_size` 行。
**所以 c4 压缩态的传输 index 就是 SWA 页号本身，"页号 = ring 块号"**，
`translate_from_swa_loc_to_state_loc` 里的 `swa_loc % ring_size` 项在整块传输时被 item_len 吸收了。

### 8.5 c128 压缩态：请求级寻址 + 取模判据

```python
# disaggregation/utils.py:82-98
def get_dsv4_c128_state_indices(req_pool_idx, seq_len, ring_size):
    if seq_len == 0 or seq_len % 128 == 0:
        return []                                    # ★ 整块边界，无在途状态
    if ONLINE_C128:                                  # ring_size == 1
        return [req_pool_idx]
    return [req_pool_idx * (ring_size // 128) + ((seq_len - 1) % ring_size) // 128]
```

三点解读：

1. **判据**：c128 每 128 个 token 压成一块。`seq_len % 128 == 0` 说明最后一块刚好压完并写进
   `c128_kv_pool`（主通道已经传过了），在途状态为空 → index 列表空 → `maybe_send_extra` 跳过。
   否则最后一块只累了 `seq_len % 128` 个 token，这段"半成品"必须传给 D 让它接着累。
2. **请求级**：c128 状态按 `req_pool_idx` 而非页号索引（`translate_from_req_position_to_state_loc`
   = `req_pool_indices * ring_size + positions % ring_size`）。所以 D 侧复用 req 槽时
   必须 `clear_c128_req_state(req_pool_idx)`，否则读到上一个请求的残留半成品 → 静默错误输出。
3. **在线 vs 离线**：`SGLANG_OPT_USE_ONLINE_COMPRESS` 时 `ring_size == 1`、`last_dim = 3 * head_dim`，
   一个请求一行；离线 `ring_size = 128`（投机 256）、`last_dim = 2*(1+overlap)*head_dim`，
   一个请求占 `ring_size // 128` 块。两侧配置不一致会触发 `_requires_exact_state_index_match` 的长度断言。

### 8.6 严格等长约束的意义

`SWA_RING` 和 `C128_STATE` 被列入 `_requires_exact_state_index_match`，长度不等直接抛异常而不截断。
原因是这两者的 index 不是"页序列"而是"环形槽位"——截断会导致 D 侧拿到不完整的环状态，
表现为输出在某个 128/ring 边界后突然乱掉，比直接崩溃难查得多。其余组件（SWA）是页序列，
截断只是少传尾页，D 侧还能靠重算兜底，所以只 warning。

## 9. 分块语义与顺序保证

| 约束 | 代码 | 原因 |
|---|---|---|
| 非最后 chunk 的 `end_idx` 必须页对齐 | `prefill.py:send_kv_chunk` | 传输单位是页；半页会让下个 chunk 重复写同一页 |
| 开 staging 时还要按 staging grid 对齐 | 同上 | staging buffer 是固定网格 |
| state + aux 只在 `is_last_chunk` 发 | `transfer_worker` | 压缩态/首 token 只有全部 prefill 完才有终值 |
| `num_pages == 0` 但 last_chunk 也要发 | `should_send_kv_chunk` | 否则 state/aux 永远发不出去，D 侧挂死 |
| 多目的 rank 全部完成才置 Success | `len(polls) == required_dst_info_num` | D 侧 TP 组的每个 rank 都要收到自己那份 |

顺序保证靠"最后一个 chunk 里，主 KV 先写、再 state、再 aux，最后才同步状态"。
D 侧看到 `KVPoll.Success` 时所有字节已落地。中间态由 `KVPoll` 的数值有序性
（`Failed=0 < Bootstrapping < WaitingForInput < Transferring < Success=4`）配合
`all_reduce(MIN)` 保证跨 rank 一致 —— 任一 rank 没好，全组就不算好。

## 10. PP / TP / DP 切分

**PP** 是唯一需要特殊处理扁平列表的维度。`get_mla_kv_ptrs_with_pp` /
`_mla_slice_ptrs_for_pp`（`common/conn.py:924-1031`）文档化了三种扁平布局：

| 列表 | 长度 | 顺序 |
|---|---|---|
| `kv_data` | `2 * c4_L + c128_L` | `[c4, c4_indexer, c128]` |
| `SWA` state_data | `swa_L + 2 * c4_L` | `[swa, c4_compress_state, c4_indexer_compress_state]` |
| `C128_STATE` state_data | `c128_L` | `[c128_state]` |

因为列表**按压缩桶分段**而不是按层号顺序排列，PP 阶段 `[prefill_start_layer, prefill_end_layer)`
不能直接切一段连续区间，必须用 `mla_compression_ratios` 数一遍
"前 `start_layer` 层里有多少个 ratio4、多少个 ratio128"，得到每个分段各自的前缀偏移再分别切。
这是 `mla_compression_ratios` 字段存在的全部理由。

**TP**：`dst_tp_rank` / `dst_attn_tp_size` 在 `KVArgsRegisterInfo` 里，
P/D TP 不等时走 `send_kvcache_slice`（按 head 维切片）。DSV4 是 MLA 族、`num_kv_heads == 1`，
KV 不按 head 切，所以常规部署下走的是整块 `send_kvcache`。

**DCP**：`translate_kv_indices_for_transfer` 负责 relayout（owner 规则 `idx % dcp == rank`），
并走 `send_kvcache_dcp` 分支。

## 11. HiSparse 变体：c4 直写 D 的 host 内存

D 侧开 HiSparse 时传输目标变了（`decode.py:_init_kv_manager` 508-606）：

```python
transfer_kv_pool = hisparse_coordinator.mem_pool_host if enable_hisparse else token_to_kv_pool
mem_kind = "DRAM" if enable_hisparse else "VRAM"
# c4 层之后的（c4_indexer + c128）仍追加为 VRAM 指针：
device_kv_data_ptrs = dst_kv_ptrs[c4_layer_num:]
```

即：**P 把 c4 KV 直接 RDMA 写进 D 的 CPU pinned 内存，跳过 GPU staging**；
c4_indexer / c128 / SWA 状态仍进 D 的显存。D 侧 decode 时按 indexer 选中的页
`load_to_device_per_layer` 换入小显存 device 池。

配套：预分配走 `alloc_for_decode_prealloc_hisparse`（只 `alloc_logical_only`，不占 device 页）+
host 池 `alloc_paged_token_slots` → `req_to_host_pool`，返回 host_indices 当 RDMA dst；
`TransferInfo.dst_device_kv_indices` 携带那部分仍走显存的索引。
`HiSparseC4DevicePool.get_cpu_copy` 抛 `NotImplementedError` —— 因此 HiSparse 下 retract 的
offload 路径不可用，且要求 `--disable-radix-cache`。

## 12. MTP / NextN

- 主通道：draft 池的 `get_contiguous_buf_infos()` 拼在 target 之后（draft 是 SWA-only，
  其实这里贡献为空/退化）。
- 状态通道：额外一个 `SWA` / `SWA_RING` 组件（§3.6），带 draft 池的 SWA 页。
- aux 通道：`output_hidden_states` 变成**必需**项 —— MTP draft 层的输入是
  "target 最后一层 hidden + 下一 token 嵌入"，那个 hidden 必须从 P 传到 D。
- ring 几何：投机时 c4 `ring_size` 8→16、c128 128→256，`setup_state_kv_args` 会校验 P/D 一致。

## 13. 失败与完整性

| 场景 | 行为 |
|---|---|
| RDMA rc != 0 | `transfer_worker` 拉黑该 mooncake session，`KVPoll.Failed` 同步回 D |
| `KVPoll` 跨 rank 不一致 | `all_reduce(MIN)` 取最小，任一 Failed 全组 Failed |
| metadata gate | `utils.py:78` —— 状态是 Success 但 `room == 0` 时降级回 `Transferring`（防早读） |
| state index 长度不匹配（严格类型） | 直接抛异常 |
| state index 长度不匹配（宽松类型） | 截断 + warning |
| P 侧 OOM | `optimistic_release_and_requeue`（`prefill.py`） |
| D 侧显存不足 | `retract_decode` → `release_req` → `offload_kv_cache` 六子池 `get_cpu_copy` 到 host |

传输字节量统计口径（`common/conn.py:1241-1249`）：

```
bytes = num_kv_indices * sum(kv_item_lens) + num_state_indices * sum(state_item_lens)
        （再乘副本因子 required_dst_info_num）
```

## 14. 速查表

### 14.1 "我改了 X，还要动哪里"

| 改动 | 必须同步修改 |
|---|---|
| 新增一个 KV 子池 | `get_contiguous_buf_infos` 或新 `StateType` + `setup_state_kv_args` + P/D 两侧 payload 闭包 + `_is_generic_kvcache_state_type` |
| 改子池排列顺序 | `_mla_slice_ptrs_for_pp` 的三个布局假设 + HiSparse 的 `dst_kv_ptrs[c4_layer_num:]` |
| 改 ring_size / last_dim | `setup_state_kv_args` 的几何校验 + `get_dsv4_c128_state_indices` |
| 改 page_size | 检查 §8.2 的恒等映射前提（必须能被 128 整除） |
| 新增 metadata 字段 | `MetadataBuffers.get_buf_infos()` 顺序（P/D 必须一致） |

### 14.2 文件行号索引

| 位置 | 内容 |
|---|---|
| `disaggregation/base/conn.py:1-120` | `StateType` / `KVArgs` / `KVPoll` 定义 |
| `mem_cache/deepseek_v4_memory_pool.py:558-574` | 页大小关系与 `_unified_kv` |
| `mem_cache/deepseek_v4_memory_pool.py:671-673` | `translate_loc_from_full_to_swa` |
| `mem_cache/deepseek_v4_memory_pool.py:675-728` | `get_contiguous_buf_infos`（主通道） |
| `mem_cache/deepseek_v4_memory_pool.py:730-746` | `get_unified_swa_ring_buf_infos` |
| `mem_cache/deepseek_v4_memory_pool.py:785-812` | `get_state_buf_infos`（SWA 组件，跳过 ratio128） |
| `mem_cache/deepseek_v4_memory_pool.py:814-828` | `get_c128_state_buf_infos` |
| `mem_cache/deepseek_v4_compress_state.py:84-218` | `CompressStatePool` 与两种 loc 翻译 |
| `disaggregation/utils.py:82-98` | `get_dsv4_c128_state_indices` |
| `disaggregation/utils.py:297-566` | `MetadataBuffers`（aux 通道全部字段） |
| `disaggregation/utils.py:746-750` | `is_mla_backend`（含 DSV4） |
| `disaggregation/utils.py:943-963` | `append_state_component` |
| `disaggregation/utils.py:966-1241` | `setup_state_kv_args`（状态通道唯一装配点） |
| `disaggregation/prefill.py:191-297` | P 侧 `_init_kv_manager` |
| `disaggregation/prefill.py:1131-1318` | `send_kv_chunk`（源索引 + payload 闭包） |
| `disaggregation/decode.py:508-606` | D 侧 `_init_kv_manager`（含 HiSparse） |
| `disaggregation/decode.py:1352-1519` | 预分配 + `send_metadata`（目的索引） |
| `disaggregation/mooncake/conn.py:79-124` | `TransferInfo` |
| `disaggregation/mooncake/conn.py:129-195` | `KVArgsRegisterInfo` |
| `disaggregation/mooncake/conn.py:288-322` | `register_buffer_to_engine` |
| `disaggregation/mooncake/conn.py:608-615` | `_transfer_data` / `batch_transfer_sync` |
| `disaggregation/mooncake/conn.py:617-762` | `_send_kvcache_generic`（地址公式） |
| `disaggregation/mooncake/conn.py:1084-1106` | `send_aux` |
| `disaggregation/mooncake/conn.py:1206-1220` | 通用状态类型白名单 / 严格等长集合 |
| `disaggregation/mooncake/conn.py:1222-1420` | `maybe_send_extra` |
| `disaggregation/mooncake/conn.py:1573-1925` | `transfer_worker` |
| `disaggregation/mooncake/conn.py:2373-2442` | `_register_kv_args` ZMQ 帧序 |
| `disaggregation/common/conn.py:924-1031` | PP 扁平列表切片 |
| `disaggregation/common/conn.py:1238-1249` | `should_send_kv_chunk` / `get_transfer_metric` |
| `mem_cache/kv_cache_configurator.py:1068-1131` | `swa_page_size` 与 `num_req_slots` |

