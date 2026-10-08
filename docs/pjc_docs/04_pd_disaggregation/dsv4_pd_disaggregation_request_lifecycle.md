# DeepSeek V4 在 PD 分离部署下的请求全生命周期

> 本文梳理 **DeepSeek V4（DSV4）** 模型在 **PD 分离（Prefill-Decode Disaggregation）** 部署形态下，一个请求分别在 **Prefill 实例（P）** 与 **Decode 实例（D）** 上的完整执行链路，覆盖三个维度：
>
> 1. **调度（Scheduling）**：请求在 P/D 两侧各自流经哪些队列、如何攒批、如何触发 forward。
> 2. **Kernel 计算（Compute）**：DSV4 特有的三层压缩 KV（ratio 0/4/128）在 prefill 与 decode 两阶段各自走哪些 attention kernel。
> 3. **传输（Transfer）**：DSV4 的六个子池如何注册到传输引擎、KV 如何从 P 单边 RDMA 写入 D 预分配的显存。
>
> 阅读前建议先读同目录下三篇前置文档：
> - [`pd_disaggregation_architecture.md`](pd_disaggregation_architecture.md)（PD 分离通用架构）
> - [`pd_disaggregation_kv_transfer_architecture.md`](pd_disaggregation_kv_transfer_architecture.md)（KV 传输通用机制）
> - [`deepseek_v4_cache_management.md`](../09_models/deepseek_v4_cache_management.md)（DSV4 六子池压缩 KV 结构）
>
> 传输部分以默认后端 **Mooncake** 为例。所有行号引用均可点击跳转。

---

## 目录

1. [第一层：一分钟全景](#第一层一分钟全景)
2. [第二层：端到端时序图](#第二层端到端时序图)
3. [第三层：Prefill 实例详解](#第三层prefill-实例详解)
4. [第四层：Decode 实例详解](#第四层decode-实例详解)
5. [第五层：DSV4 特有的 KV 传输机制](#第五层dsv4-特有的-kv-传输机制)
6. [第六层：Kernel 计算 —— Prefill vs Decode](#第六层kernel-计算--prefill-vs-decode)
7. [附录：P/D 差异速查表与文件索引](#附录pd-差异速查表与文件索引)

---

## 第一层：一分钟全景

PD 分离把一次推理拆到两组 GPU 上：**Prefill 实例**只做首次全量前向（吃满算力、产出全部 KV cache + 首 token），**Decode 实例**只做逐 token 自回归（吃满显存带宽）。两者之间通过 RDMA 把 KV cache 从 P 搬到 D。

对 DSV4 而言，这个"搬 KV"比普通模型复杂得多，因为 DSV4 的 KV 不是单一张量，而是**六个子池**的异构组合（详见 [第五层](#第五层dsv4-特有的-kv-传输机制)）。

```
                            ┌─────────────────────────────────────┐
   HTTP 请求 ──────────────▶│           Prefill 实例 (P)           │
   (bootstrap_room 唯一标识) │                                     │
                            │  调度: Bootstrap→Waiting→Inflight   │
                            │  计算: 写六子池 + dense/sparse attn  │
                            │  产出: 全量 KV + 首 token            │
                            └──────────────┬──────────────────────┘
                                           │ ① 握手(HTTP bootstrap server 服务发现)
                                           │ ② decode 经 ZMQ 推送目标地址 TransferInfo
                                           │ ③ P 后台线程 RDMA 单边写六子池 + 首token
                                           ▼
                            ┌─────────────────────────────────────┐
                            │           Decode 实例 (D)            │
                            │                                     │
                            │  调度: Prealloc→Transfer→Waiting    │
                            │        →PrebuiltBatch→RunningBatch  │
                            │  计算: 逐 token 三层 attn(SWA+稀疏) │
                            │  产出: 后续所有 token(流式返回)      │
                            └─────────────────────────────────────┘
```

**三个维度的核心结论先行**：

| 维度 | Prefill 实例 | Decode 实例 |
|------|-------------|-------------|
| **调度** | 三队列：`Bootstrap`（握手）→ `Waiting`（复用普通队列，做 forward）→ `Inflight`（KV 传输中） | 三队列：`Prealloc`（握手+预分配显存）→ `Transfer`（轮询 KV 到达）→ `Waiting`（组 `PrebuiltBatch` 跳过 forward）→ `RunningBatch`（真正 decode） |
| **Kernel** | 写入六子池（SWA + c4/c128 压缩 + indexer）；attention 分**小 prefill（FlashMLA dense/paged）**与**大 prefill（`flash_mla_sparse_fwd` + dequant workspace）**两条路径；因果掩码经 `expand_prefill_casually` 展开 | 不做 prefill forward（KV 已传好）；每步 c4 层跑 `fp8_paged_mqa_logits`+`topk_transform` 选稀疏页，`flash_mla_with_kvcache` 融合 SWA 近窗 dense + extra（c4 稀疏 / c128 全块） |
| **传输** | `send_kv_chunk` 用 `req_to_token[req_pool_idx]` 把 token 位置翻译成各子池 slot；主 KV 通道（c4/c128/indexer）+ 状态通道（SWA + 压缩态 ring）分别搬 | `_pre_alloc` 预分配六子池显存 → 把 `dst_kv_indices` 经 ZMQ 告知 P；`poll` 轮询 + metadata gate 确认到齐 |

**关键设计约束**：DSV4 强制 `page_size=256`、`kv_cache_dtype=fp8_e4m3`、attention backend 为 `dsv4`；PD 分离要求 overlap 与传输后端支持 DSV4 的多状态类型（`StateType.SWA` / `StateType.SWA_RING` / `StateType.DSA`）。

---

## 第二层：端到端时序图

下图把一个请求从进入 P 到在 D 上跑完 decode 的完整时序串起来，标注了每一步涉及的**队列状态**、**KVPoll 状态机**与**关键函数**。

```
 时间 ─────────────────────────────────────────────────────────────────────▶

 ┌── Prefill 实例 (P) ───────────────────────────────────────────────────┐
 │                                                                        │
 │ [调度] handle_generate_request                                         │
 │        └─▶ disagg_prefill_bootstrap_queue.add()   scheduler.py:2291    │
 │            └─▶ create_sender() 建 MooncakeKVSender prefill.py:228      │
 │                KVPoll: Bootstrapping ◀─── update_status common:779     │
 │                                                                        │
 │ [握手] ①P 启动时 PUT /route 注册拓扑到 bootstrap server                │
 │        ②D 侧 GET /route 查 P 拓扑 → 算 target rank                     │
 │        ③D 侧 ZMQ send_metadata 推 TransferInfo(dst_kv_indices…)        │
 │        ④P bootstrap_thread 收齐 required_dst_info_num 份               │
 │           KVPoll: Bootstrapping → WaitingForInput  mooncake:1527       │
 │                                                                        │
 │ [调度] pop_bootstrapped() → waiting_queue.extend  prefill.py:309/476   │
 │        get_new_batch_prefill() 组批            prefill.py:461          │
 │                                                                        │
 │ [计算] run_batch → DSV4 forward                                        │
 │        写 SWA(set_swa_key_buffer_radix_fused_norm_rope)               │
 │        写 c4/c128(forward_core_compressor)                             │
 │        写 c4_indexer(forward_indexer_compressor)                       │
 │        attn: FlashMLA dense/paged 或 sparse-prefill                    │
 │                                                                        │
 │ [传输] process_batch_result_disagg_prefill    prefill.py:546          │
 │        └─▶ inflight_queue.append(req)         prefill.py:632          │
 │        └─▶ send_kv_chunk(last_chunk=True)     prefill.py:656/971      │
 │            主KV通道: c4/c128/indexer 页                                │
 │            状态通道: SWA + 压缩态 ring 块                              │
 │            aux: 首 token + logprob + hidden_states                    │
 │        └─▶ transfer_worker 后台线程 RDMA 单边写  mooncake:1198         │
 │            KVPoll: Transferring → Success       mooncake:1387          │
 │                                                                        │
 │ [结束] process_disagg_prefill_inflight_queue   prefill.py:719         │
 │        Success → release_kv_cache + stream_output 首token + 释放buffer │
 └────────────────────────────────────────────────────────────────────────┘
        │
        │  (①②③④ 握手 + RDMA 写贯穿于 P 的整个生命周期)
        ▼
 ┌── Decode 实例 (D) ───────────────────────────────────────────────────┐
 │                                                                        │
 │ [调度] handle_generate_request                                         │
 │        └─▶ disagg_decode_prealloc_queue.add()  decode.py:478          │
 │            └─▶ 建 MooncakeKVReceiver, KVPoll: Bootstrapping           │
 │                                                                        │
 │ [握手] receiver.init() GET /route 解析 P 拓扑                          │
 │        KVPoll: Bootstrapping → WaitingForInput  common:987            │
 │                                                                        │
 │ [调度+预分配] pop_preallocated()               decode.py:778          │
 │        └─▶ _pre_alloc() 预分配六子池显存        decode.py:1288         │
 │        └─▶ send_metadata() ZMQ 把 dst_kv_indices 推给 P decode:1075   │
 │        └─▶ transfer_queue.extend                decode.py:1977         │
 │                                                                        │
 │ [传输] pop_transferred() 轮询                   decode.py:1636         │
 │        poll + metadata gate(检查首token元数据落地) utils:78           │
 │        KVPoll: Transferring → Success                                 │
 │        └─▶ _commit_transfer_to_req 读首token     decode.py:1491        │
 │        └─▶ waiting_queue.extend                 decode.py:1985         │
 │                                                                        │
 │ [调度] get_new_prebuilt_batch()                 decode.py:1886         │
 │        prepare_for_prebuilt() forward_mode=PREBUILT(跳过prefill)      │
 │        merge 进 running_batch                                          │
 │                                                                        │
 │ [计算] run_batch → 逐步 DECODE forward                                 │
 │        c4层: fp8_paged_mqa_logits + topk_transform 选稀疏页            │
 │        flash_mla_with_kvcache: SWA近窗dense + extra(c4稀疏/c128全块)   │
 │        每步在 seq_len%ratio==0 边界写压缩态                            │
 │        (OOM 时 retract_decode → KV 存 CPU → resume)                   │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 第三层：Prefill 实例详解

Prefill 侧的官方生命周期在 [prefill.py:1-18](../../../python/sglang/srt/disaggregation/prefill.py#L1-L18) 的 docstring 里定义，代码实现完全对应三段式：**Bootstrap Queue → Waiting Queue → Inflight Queue**。

### 3.1 三个调度队列

| 阶段 | 队列 | 类名 / 类型 | 定义位置 | 作用 |
|------|------|-------------|----------|------|
| 1 | `disagg_prefill_bootstrap_queue` | `PrefillBootstrapQueue`（内部 `self.queue: List[Req]`） | [prefill.py:102](../../../python/sglang/srt/disaggregation/prefill.py#L102)，实例化于 [scheduler.py:1202](../../../python/sglang/srt/managers/scheduler.py#L1202) | 握手 + 等 decode 预分配；为每个请求建 KVSender、轮询状态 |
| 2 | `waiting_queue` | `List[Req]`（与普通模式共用的 Scheduler 属性） | 由 `pop_bootstrapped()` 灌入：[prefill.py:476](../../../python/sglang/srt/disaggregation/prefill.py#L476)（normal）/ [prefill.py:508](../../../python/sglang/srt/disaggregation/prefill.py#L508)（overlap） | 握手完成、等 `PrefillAdder` 选中做 forward |
| 3 | `disagg_prefill_inflight_queue` | `List[Req]` | 初始化 [scheduler.py:1219](../../../python/sglang/srt/managers/scheduler.py#L1219)，入队 [prefill.py:632](../../../python/sglang/srt/disaggregation/prefill.py#L632) | forward 已完成、**KV 正在传输中**；非阻塞轮询 sender |
| 4（后台） | `transfer_queues[shard_idx]` | `List[FastQueue]`，元素为 `TransferKVChunk` | [mooncake/conn.py:181](../../../python/sglang/srt/disaggregation/mooncake/conn.py#L181) | KVManager 内部实际传输流水线，由 `transfer_worker` 线程消费执行 RDMA |

前三个在调度器层（`Req` 粒度），第四个在 KVManager 内部（KV chunk 粒度），两者正交。事件循环入口：`event_loop_normal_disagg_prefill`（[prefill.py:469](../../../python/sglang/srt/disaggregation/prefill.py#L469)）与 `event_loop_overlap_disagg_prefill`（[prefill.py:500](../../../python/sglang/srt/disaggregation/prefill.py#L500)）。

**队列间搬运**：
- Bootstrap → Waiting：`waiting_queue.extend(pop_bootstrapped())`（[prefill.py:476](../../../python/sglang/srt/disaggregation/prefill.py#L476)）
- Waiting → forward：`get_new_batch_prefill()`（[prefill.py:461](../../../python/sglang/srt/disaggregation/prefill.py#L461)）
- forward 完成 → Inflight：`inflight_queue.append(req)`（[prefill.py:632](../../../python/sglang/srt/disaggregation/prefill.py#L632)）
- Inflight → 释放：`process_disagg_prefill_inflight_queue()`（[prefill.py:719](../../../python/sglang/srt/disaggregation/prefill.py#L719)）

### 3.2 握手：bootstrap server + ZMQ 反向推送

握手是一个 **HTTP 服务发现 + ZMQ 元数据推送** 的两段式协议。核心思想：**P 不主动"找" D，而是被动等待 D 通过 ZMQ 把目标显存地址推过来。**

**Bootstrap Server 的角色**：`CommonKVBootstrapServer`（[common/conn.py:1205](../../../python/sglang/srt/disaggregation/common/conn.py#L1205)，Mooncake 继承为 `MooncakeKVBootstrapServer`）是一个 aiohttp HTTP 服务，只做**服务发现注册表**，不碰 KV 数据本身。维护两张表：`prefill_port_table`（四层嵌套 `{dp:{cp:{tp:{pp: (ip,port)}}}}`，[common/conn.py:1220](../../../python/sglang/srt/disaggregation/common/conn.py#L1220)）和 `room_to_dp_rank`（[common/conn.py:1223](../../../python/sglang/srt/disaggregation/common/conn.py#L1223)）。

HTTP 路由（[common/conn.py:1250](../../../python/sglang/srt/disaggregation/common/conn.py#L1250)）：
- `PUT /route`：**P 各 rank 启动时注册**自己的 `(ip, port, 并行拓扑)`。
- `GET /route`：**D 查询** P 的拓扑或某 rank 的 `(ip, port)`。
- `GET /health`：D 端心跳探活。

握手全流程：

```
   Prefill 实例 (P)                Bootstrap Server            Decode 实例 (D)
        │                              (HTTP)                       │
        │ ① 启动: PUT /route 注册拓扑    │                            │
        ├─────────────────────────────▶│                            │
        │   (rank_ip, rank_port,        │                            │
        │    tp/cp/dp/pp size+rank)     │                            │
        │                              │  ② init(): GET /route       │
        │                              │◀────────────────────────────┤
        │                              │  查 P 拓扑,算 target rank    │
        │                              ├────────────────────────────▶│
        │                              │  返回 (rank_ip, rank_port)  │
        │                              │                            │
        │ ③ ZMQ 反向推送 (绕过 HTTP server, D→P 直连 rank_port)         │
        │◀──────────────────────────────────────────────────────────┤
        │   send_metadata: room, decode_ip, session_id,             │
        │   dst_kv_indices, dst_aux_index, dst_state_indices,       │
        │   required_dst_info_num, decode_prefix_len                │
        │                                                            │
        │ ④ bootstrap_thread 收齐 required_dst_info_num 份            │
        │   update_status(room, WaitingForInput)  mooncake:1527      │
        │                                                            │
```

**"KV 发到哪个 D 的哪块显存"如何确定？** 全在 D 通过 ZMQ 推来的 `TransferInfo`（[mooncake/conn.py:68](../../../python/sglang/srt/disaggregation/mooncake/conn.py#L68)）里：

| 字段 | 含义 |
|------|------|
| `mooncake_session_id` | Mooncake 引擎会话（对应 D 侧一块已注册 RDMA 内存） |
| `dst_kv_indices` | **D 侧已预分配好的 KV slot 索引数组** |
| `dst_aux_index` | D 侧 metadata buffer 索引 |
| `dst_state_indices` | SWA/DSA/mamba 等状态页索引 |
| `decode_prefix_len` | D 已命中前缀、P 无需重发的长度 |

目标基地址指针 `dst_kv_ptrs` 存在 `KVArgsRegisterInfo`（[mooncake/conn.py:112](../../../python/sglang/srt/disaggregation/mooncake/conn.py#L112)）里，D 侧启动时经一个 `room=="None"` 的特殊 ZMQ 包上报。最终写入地址 = `dst_kv_ptrs[layer] + dst_kv_index * item_len`。

### 3.3 计算：DSV4 forward 写六子池

握手完成的请求进 `waiting_queue`，被 `PrefillAdder` 选批后 `run_batch` 触发 DSV4 forward。DSV4 在 prefill 阶段要把 KV 写进**六个子池**（编排入口 `MQALayer._forward_prepare`，[deepseek_v4.py:746](../../../python/sglang/srt/models/deepseek_v4.py#L746)，多流版 [:568](../../../python/sglang/srt/models/deepseek_v4.py#L568)）：

1. **SWA KV**（所有层）：`set_swa_key_buffer_radix_fused_norm_rope`（[deepseek_v4_memory_pool.py:1040](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1040)），单 kernel `fused_k_norm_rope_flashmla` 做 RMSNorm+RoPE+scatter 融合写。
2. **压缩态**（ratio≠0 层）：`forward_core_compressor`（[compressor.py:166](../../../python/sglang/srt/layers/attention/dsv4/compressor.py#L166)）→ c4/c128 pool。
3. **indexer**（仅 ratio==4 层）：`forward_indexer_compressor`（[compressor.py:198](../../../python/sglang/srt/layers/attention/dsv4/compressor.py#L198)）→ c4_indexer pool。

attention 计算细节见 [第六层](#第六层kernel-计算--prefill-vs-decode)。

### 3.4 传输：`send_kv_chunk` —— token 索引到六子池内存的映射

forward 结果由 `process_batch_result_disagg_prefill`（[prefill.py:546](../../../python/sglang/srt/disaggregation/prefill.py#L546)）处理，对每个完成的请求：追加首 token（[prefill.py:630](../../../python/sglang/srt/disaggregation/prefill.py#L630)）→ `maybe_cache_unfinished_req` 插 radix 树 → 入 inflight 队列（[prefill.py:632](../../../python/sglang/srt/disaggregation/prefill.py#L632)）→ `send_kv_chunk(last_chunk=True)`（[prefill.py:656](../../../python/sglang/srt/disaggregation/prefill.py#L656)）。

`send_kv_chunk`（[prefill.py:971](../../../python/sglang/srt/disaggregation/prefill.py#L971)）是 token 索引 → KV pool 内存的核心桥梁：

```python
# prefill.py:1001 —— req_to_token 是关键映射表：请求槽位+token位置 → KV pool 物理 slot
kv_indices = self.req_to_token_pool.req_to_token[
    req.req_pool_idx, start_idx:end_idx
].cpu().numpy()
...
page_indices = kv_to_page_indices(kv_indices, page_size)   # prefill.py:1082 token→页
req.disagg_kv_sender.send(page_indices, state_indices)     # prefill.py:1085
```

**最后一块**额外构造 `state_indices`（[prefill.py:1062-1080](../../../python/sglang/srt/disaggregation/prefill.py#L1062-L1080)），按 `state_types` 顺序分派 —— 对 DSV4 这里会出现 `StateType.SWA`（`_swa_payload`，[prefill.py:1027](../../../python/sglang/srt/disaggregation/prefill.py#L1027)，经 `translate_loc_from_full_to_swa` 翻译）和可能的 `StateType.SWA_RING`（`_swa_ring_payload`，[prefill.py:1049](../../../python/sglang/srt/disaggregation/prefill.py#L1049)，unified_kv 场景）。这与 [第五层](#第五层dsv4-特有的-kv-传输机制) 的六子池双通道一一对应。

发送后由后台 `transfer_worker`（[mooncake/conn.py:1198](../../../python/sglang/srt/disaggregation/mooncake/conn.py#L1198)）取 chunk，用 `req.dst_kv_indices` 算目标地址，`engine.batch_transfer_sync` 执行 RDMA 单边写。最后一块还会 `send_aux` 发首 token/logprob/hidden_states，全部目标 rank 收齐后 `update_status(room, Success)`（[mooncake/conn.py:1387](../../../python/sglang/srt/disaggregation/mooncake/conn.py#L1387)）。

### 3.5 结束：释放与首 token 回传

`process_disagg_prefill_inflight_queue`（[prefill.py:719](../../../python/sglang/srt/disaggregation/prefill.py#L719)）轮询 inflight 队列。`KVPoll.Success`（[prefill.py:762](../../../python/sglang/srt/disaggregation/prefill.py#L762)）时：

1. `release_kv_cache`（[prefill.py:763](../../../python/sglang/srt/disaggregation/prefill.py#L763)）—— 解锁 radix 节点，KV 归还缓存（可被后续复用/驱逐）。
2. `req.finished_reason = FINISH_LENGTH(length=0)`（[prefill.py:764](../../../python/sglang/srt/disaggregation/prefill.py#L764)）—— P 只生成 1 token。
3. `sender.clear()`（[prefill.py:766](../../../python/sglang/srt/disaggregation/prefill.py#L766)）—— 从各字典弹出 room，防泄漏与复用污染。
4. `stream_output`（[prefill.py:820](../../../python/sglang/srt/disaggregation/prefill.py#L820)）—— **把首 token 流式返回 client**。
5. `maybe_release_metadata_buffer`（[prefill.py:828](../../../python/sglang/srt/disaggregation/prefill.py#L828)）—— 归还 metadata buffer 槽。

至此 P 侧请求结束，后续 decode 完全在 D 上进行。

### 3.6 `KVSender` 状态机

五个状态（`KVPoll`，[base/conn.py:77-82](../../../python/sglang/srt/disaggregation/base/conn.py#L77-L82)）：`Failed=0`、`Bootstrapping=1`、`WaitingForInput=2`、`Transferring=3`、`Success=4`（数值有序，`all_reduce(MIN)` 保证跨 rank 一致：任一 rank Failed 则全组 Failed，必须全 Success 才全组 Success）。

```
                    create_sender()                  bootstrap_thread 收齐
  [新建] ──────────────────────────▶ Bootstrapping ──────────────────▶ WaitingForInput
        update_status(Bootstrapping)      │  common:779   mooncake:1527       │
                                          │                                  │ send_kv_chunk→send()
                             超时 _check_bootstrap_timeout                     ▼
                             common:889 → Failed                        (transfer_worker
                                          │                              逐块 RDMA 写)
                                          ▼                                   │ 最后块收齐所有 dst rank
                                       Failed ◀──── 传输失败 ────────── update_status
                                                                       mooncake:1387
                                                                              ▼
                                                                          Success
```

---

## 第四层：Decode 实例详解

Decode 侧的官方生命周期在 [decode.py:1-19](../../../python/sglang/srt/disaggregation/decode.py#L1-L19) 的 docstring 里定义：**Prealloc Queue → Transfer Queue → Waiting Queue → Running Batch**。驱动入口是 `process_decode_queue`（[decode.py:1953](../../../python/sglang/srt/disaggregation/decode.py#L1953)），每次 event loop 迭代调用。

### 4.1 四个调度阶段

| 阶段 | 队列 | 类名 | 定义位置 | 作用 |
|------|------|------|----------|------|
| 1 | `disagg_decode_prealloc_queue` | `DecodePreallocQueue` | [decode.py:273](../../../python/sglang/srt/disaggregation/decode.py#L273) | 建 receiver、握手、**预分配本地六子池 KV**、把 dst 索引告知 P |
| 2 | `disagg_decode_transfer_queue` | `DecodeTransferQueue` | [decode.py:1453](../../../python/sglang/srt/disaggregation/decode.py#L1453) | 轮询 receiver 状态，KV 到齐后 commit 首 token 元数据 |
| 3 | `waiting_queue` | Scheduler 共用属性 | — | 攒批，构造 `PrebuiltExtendBatch` |
| 4 | `running_batch` | `ScheduleBatch` | — | PREBUILT 填元数据（跳过 forward）→ merge → 跑 decode |
| 旁支 | `retracted_queue` | `DecodePreallocQueue.retracted_queue` | [decode.py:321](../../../python/sglang/srt/disaggregation/decode.py#L321) | OOM 被 retract 回来的请求（KV 已存 CPU） |

阶段间搬运（`process_decode_queue`，[decode.py:1961-1985](../../../python/sglang/srt/disaggregation/decode.py#L1961-L1985)）：

```python
resumed_reqs = resume_retracted_reqs(); waiting_queue.extend(resumed_reqs)  # 先恢复被 retract 的
if len(retracted_queue) > 0: return                                          # 有 retract 就不收新请求
...
req_conns, _ = pop_preallocated(); transfer_queue.extend(req_conns)          # 阶段1→2
transferred_reqs = pop_transferred(); waiting_queue.extend(transferred_reqs) # 阶段2→3
```

注意有个 **polling_interval 节流**（[decode.py:1967-1975](../../../python/sglang/srt/disaggregation/decode.py#L1967-L1975)）：预分配/传输轮询每 `disaggregation_decode_polling_interval` 步才做一次。承载阶段 1、2 的载体是 `DecodeRequest`（[decode.py:250](../../../python/sglang/srt/disaggregation/decode.py#L250)），字段含 `req`、`kv_receiver`、`waiting_for_input`、`metadata_buffer_index`。

### 4.2 阶段 1：预分配与握手（`DecodePreallocQueue`）

**入队与握手**：`add()`（[decode.py:478](../../../python/sglang/srt/disaggregation/decode.py#L478)）先检查 KV 容量上限，创建 receiver（传入 `bootstrap_addr` + `bootstrap_room`）包成 `DecodeRequest`。握手推进由 `_update_handshake_waiters`（[decode.py:643](../../../python/sglang/srt/disaggregation/decode.py#L643)）完成：对所有 receiver 调 `poll_and_all_reduce`（跨 attn-TP 组 `all_reduce(MIN)`），返回 `WaitingForInput` 时置 `waiting_for_input=True`（[decode.py:667](../../../python/sglang/srt/disaggregation/decode.py#L667)）。

**预分配六子池显存**（核心：`pop_preallocated` → `_pre_alloc`）：`pop_preallocated`（[decode.py:778](../../../python/sglang/srt/disaggregation/decode.py#L778)）先算可分配预算（要预留 `retractable_tokens` 防死锁，[decode.py:789](../../../python/sglang/srt/disaggregation/decode.py#L789)），对 `waiting_for_input=True` 的请求调 `_pre_alloc`（[decode.py:1288](../../../python/sglang/srt/disaggregation/decode.py#L1288)）：

- `req_to_token_pool.alloc([req])` 拿 `req_pool_idx`（[decode.py:1308](../../../python/sglang/srt/disaggregation/decode.py#L1308)）—— 用特化的 `DecodeReqToTokenPool`（[decode.py:105](../../../python/sglang/srt/disaggregation/decode.py#L105)），额外开 `pre_alloc_size` 行使预分配不占 `max_running_requests` 配额。
- 按 page_size 调 `alloc`/`alloc_extend`，SWA 尾部走 `alloc_extend_swa_tail`（[decode.py:1393](../../../python/sglang/srt/disaggregation/decode.py#L1393)）。**对 DSV4，这一步分配的就是六子池的显存 slot**（顶层 `DeepSeekV4TokenToKVPoolAllocator` 内部同时管理 swa/c4/c128/indexer/压缩态）。
- 写 `kv_loc` 到 `req_to_token`（[decode.py:1426](../../../python/sglang/srt/disaggregation/decode.py#L1426)），返回 `kv_loc`。

**告知 P 目标地址**：拿到 `dst_kv_indices` 后，`pop_preallocated` 提取 delta 索引（超出 prefix 的部分），按 `state_types` 组装各类状态索引（[decode.py:1052-1068](../../../python/sglang/srt/disaggregation/decode.py#L1052-L1068)），最后：

```python
decode_req.kv_receiver.send_metadata(               # decode.py:1075
    page_indices, metadata_buffer_index,
    state_indices, decode_prefix_len=total_prefix_len)
```

以 Mooncake 为例，`send_metadata`（[mooncake/conn.py:1866](../../../python/sglang/srt/disaggregation/mooncake/conn.py#L1866)）通过 ZMQ 把 `dst_kv_indices` 等推给 P（即 3.2 节的第③步）。请求随后移入 `transfer_queue`。

### 4.3 阶段 2：轮询 KV 到达（`DecodeTransferQueue`）

主循环 `pop_transferred`（[decode.py:1636](../../../python/sglang/srt/disaggregation/decode.py#L1636)）通过 `_poll_with_metadata_gate`（[decode.py:1602](../../../python/sglang/srt/disaggregation/decode.py#L1602)）→ `poll_and_all_reduce` 对每个 receiver 调 `poll()` 并跨 TP 组归约。按结果分类（[decode.py:1656-1732](../../../python/sglang/srt/disaggregation/decode.py#L1656-L1732)）：

- `Failed`：`prepare_abort` + `release_kv_cache(is_insert=False)` 释放预分配显存。
- `Success`：`_commit_transfer_to_req`（[decode.py:1491](../../../python/sglang/srt/disaggregation/decode.py#L1491)）从 metadata buffer 读首 token + 元数据（先校验 `bootstrap_room` 防串号），`req.output_ids.append`，移入 `transferred_reqs`。
- `Bootstrapping/WaitingForInput/Transferring`：继续等待。

**关键的 metadata gate**（`_apply_metadata_gate`，[utils.py:78](../../../python/sglang/srt/disaggregation/utils.py#L78)）：即使 receiver 报 `Success`，若 metadata buffer 里 `bootstrap_room` 仍为 0（首 token 元数据尚未 RDMA 落地），就把状态**降级回 `Transferring`**。由于降级发生在 `all_reduce(MIN)` 之前，任一 rank 没就绪都会让全组回退，保证跨 rank 一致提交。

### 4.4 阶段 3→4：PrebuiltBatch 与开始 decode

`transferred_reqs` 进 `waiting_queue`。`get_new_prebuilt_batch`（[decode.py:1886](../../../python/sglang/srt/disaggregation/decode.py#L1886)）按名额取请求，`prepare_for_prebuilt()`（[decode_schedule_batch_mixin.py:25](../../../python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py#L25)）把 `forward_mode` 设为 `ForwardMode.PREBUILT`，**从 `req_to_token` 重建元数据但跳过 prefill forward**（KV 已由传输填好）。然后 merge 进 running batch（[decode.py:1860-1872](../../../python/sglang/srt/disaggregation/decode.py#L1860-L1872)），正常走 DECODE forward。

### 4.5 OOM Retract：PD 场景下 KV 存 CPU

**检测**：`update_running_batch` 每步调 `check_decode_mem()`（[schedule_batch.py:2453](../../../python/sglang/srt/managers/schedule_batch.py#L2453)），不足触发 retract。

**执行**：`retract_decode`（[schedule_batch.py:2470](../../../python/sglang/srt/managers/schedule_batch.py#L2470)）排序（优先踢输出最短、输入最长者）后 pop，至少保留一个请求。**PD decode 专属**：`release_req`（[schedule_batch.py:1590](../../../python/sglang/srt/managers/schedule_batch.py#L1590)）里对 decode 模式调 `req.offload_kv_cache`（[schedule_batch.py:1603](../../../python/sglang/srt/managers/schedule_batch.py#L1603)）—— 通过 `get_cpu_copy` 把该请求六子池的 KV 拷到 host `req.kv_cache_cpu`，再释放显存。这样 retract **保留了已生成的 KV，不必重新 prefill**。

**回队与恢复**：retract 出来的请求进 `retracted_queue`（不重新握手）。`resume_retracted_reqs`（[decode.py:593](../../../python/sglang/srt/disaggregation/decode.py#L593)）在内存足够时重新 `_pre_alloc` + `req.load_kv_cache`（从 CPU 拷回显存），再进 `waiting_queue`。**只要 `retracted_queue` 非空就不收新请求**（[decode.py:1963-1965](../../../python/sglang/srt/disaggregation/decode.py#L1963-L1965)），优先恢复老请求。

### 4.6 `KVReceiver` 状态机

```
             init() 解析 P 拓扑成功          P 收齐 metadata,RDMA 写入中,收齐所有 P rank Success
  Bootstrapping ──────────────────▶ WaitingForInput ──────────────────────────────────▶ Success
       │(_setup 失败)                     │(waiting_timeout)          │(+ metadata gate 通过)
       ▼                                 ▼                           │
     Failed ◀──────────────────────────────────────────────────────┘
```

- `Bootstrapping→WaitingForInput` 由 D 侧 `init()`（[common/conn.py:987](../../../python/sglang/srt/disaggregation/common/conn.py#L987)）推进。
- `→Success` 由 D 后台 `decode_thread` 收齐 P 的状态同步后 `update_status(Success)`（[mooncake/conn.py:1586](../../../python/sglang/srt/disaggregation/mooncake/conn.py#L1586)）。
- 跨 TP 一致性靠 `poll_and_all_reduce` 的 `ReduceOp.MIN`（[utils.py:114](../../../python/sglang/srt/disaggregation/utils.py#L114)）。

---

## 第五层：DSV4 特有的 KV 传输机制

这是 DSV4 PD 分离最独特、也最复杂的部分。普通 MLA 模型只有单一 KV 缓冲，而 DSV4 的顶层池 `DeepSeekV4TokenToKVPool`（[deepseek_v4_memory_pool.py:438](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L438)，继承 `BaseSWAKVPool`）含**六个子池**，被拆到**两条独立传输通道**上。

### 5.1 六子池 → 两通道的映射

| # | 子池 | 缓冲字段 | 传输通道 | item_len 粒度 |
|---|------|---------|----------|--------------|
| 1 | `swa_kv_pool` | `kv_buffer[L]` | **状态通道**（`StateType.SWA`） | 一页 |
| 2 | `c4_kv_pool` | `kv_buffer[L]` | **主 KV 通道** | 一个压缩页 |
| 3 | `c128_kv_pool` | `kv_buffer[L]` | **主 KV 通道** | 一个压缩页 |
| 4 | `c4_indexer_kv_pool` | `index_k_with_scale_buffer[L]` | **主 KV 通道** | 一页 |
| 5 | `compress_state_pools` | `kv_score_buffer.kv_score` | **状态通道**（`StateType.SWA`） | **一个 ring 块**（ring_size 行） |
| 6 | `indexer_compress_state_pools` | `kv_score_buffer.kv_score` | **状态通道**（`StateType.SWA`） | **一个 ring 块** |

两条通道对应顶层池的两个方法：

| 通道 | 方法 | 位置 | 承载 |
|------|------|------|------|
| **主 KV 数据** | `get_contiguous_buf_infos()` | [deepseek_v4_memory_pool.py:643](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L643) | c4 + c4_indexer + c128（顺序固定） |
| **状态** | `get_state_buf_infos()` | [deepseek_v4_memory_pool.py:716](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L716) | swa_kv_pool + 两类压缩态 ring 池 |
| **状态(unified)** | `get_unified_swa_ring_buf_infos()` | [deepseek_v4_memory_pool.py:698](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L698) | unified 缓冲的 SWA ring 区（`StateType.SWA_RING`） |

三个方法都返回三元组 `(data_ptrs, data_lens, item_lens)`：`data_ptrs`=每个缓冲的设备指针（RDMA 注册基址）；`data_lens`=总字节数；`item_lens`=**每个可寻址单元（page/ring 块）的字节数**，即 `src_addr = base_ptr + index * item_len` 的步长。

### 5.2 主 KV 通道 `get_contiguous_buf_infos`

非 unified 路径（[deepseek_v4_memory_pool.py:683-696](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L683-L696)）按固定顺序 `[c4, c4_indexer, c128]` 拼接：

```python
buf_groups = [
    self.c4_kv_pool.kv_buffer,                          # c4 压缩 KV
    self.c4_indexer_kv_pool.index_k_with_scale_buffer,  # c4 indexer 的 index_k
    self.c128_kv_pool.kv_buffer,                        # c128 压缩 KV
]
for bufs in buf_groups:
    for buf in bufs:
        data_ptrs.append(buf.data_ptr())
        data_lens.append(buf.nbytes)
        item_lens.append(buf[0].nbytes)   # 注意：一行=一页，不 ×page_size
```

**与标准 MLA 的关键差异**：标准 MLA 的 `item_len = buf[0].nbytes * page_size`，而 DSV4 用 `buf[0].nbytes`。因为 DSV4 每个池的 `kv_buffer` 形状本身就是 `[num_pages, bytes_per_page_padded]`，**每一行就是一整页**（见 [deepseek_v4_memory_pool.py:103](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L103) 的 `DeepSeekV4SingleKVPool.create_buffer`），所以 `item_len` 已经是一页字节数。

### 5.3 状态通道 `get_state_buf_infos`

三段拼成一个扁平列表，整体作为**单个 `StateType.SWA` 组件**上报（[deepseek_v4_memory_pool.py:716-741](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L716-L741)）：

```python
# ① SWA 全精度近窗 KV，item_len = 一页
for buf in self.swa_kv_pool.kv_buffer:
    item_lens.append(buf[0].nbytes)
# ② 注意力压缩态 ring + ③ indexer 压缩态 ring，item_len = 一整个 ring 块
for pools in [self.compress_state_pools, self.indexer_compress_state_pools]:
    for pool in pools:
        if pool is None: continue
        t = pool.kv_score_buffer.kv_score
        item_lens.append(t[0].nbytes * pool.ring_size)   # 一个 ring 块 = ring_size 行
```

### 5.4 CompressStatePool ring buffer 的传输处理（关键设计）

`CompressStatePool`（[deepseek_v4_compress_state.py:84](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L84)）是扁平 ring buffer，ring 寻址逻辑（`translate_from_swa_loc_to_state_loc`，[:192](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py#L192)）：

```python
swa_pages = swa_loc // self.swa_page_size
state_loc = swa_pages * self.ring_size + (swa_loc % self.ring_size)
```

传输时 `item_len = t[0].nbytes * ring_size`，**一个"item"就是一整个 ring 块**。因此传输索引用 **swa 页号**，`src_addr = ptr + swa_page * (row_bytes * ring_size)` 正好落在 `swa_page * ring_size` 这个 ring 块基址上，与上面 ring 寻址的块基址完全一致。换言之，**传输以整个 ring 块为粒度、用 swa 页号作块索引**，块内的 `swa_loc % ring_size` 偏移随块一起搬走，无需单独计算。ring_size：c4=8（spec 16）、c128=128（spec 256）、online c128=1。

### 5.5 描述符组装：`KVArgs` 与 `setup_state_kv_args`

传输引擎用 `KVArgs`（[base/conn.py:36](../../../python/sglang/srt/disaggregation/base/conn.py#L36)）描述所有待传缓冲：主通道 `kv_data_ptrs/lens/item_lens`；状态通道 `state_types`（`List[StateType]`）、`state_data_ptrs`（每个 StateType 一个子列表）等；DSV4 专用 `mla_compression_ratios`（[base/conn.py:70](../../../python/sglang/srt/disaggregation/base/conn.py#L70)）。

`StateType` 枚举（[base/conn.py:17](../../../python/sglang/srt/disaggregation/base/conn.py#L17)）：`MAMBA / SWA / DSA / MINIMAX_INDEX_K / SWA_RING`。

组装在 `setup_state_kv_args`（[utils.py:632](../../../python/sglang/srt/disaggregation/utils.py#L632)，P/D 共用）：对 `BaseSWAKVPool`（含 DSV4）调 `get_state_buf_infos()` 整体作为一个 `StateType.SWA` 组件（[utils.py:667-668](../../../python/sglang/srt/disaggregation/utils.py#L667-L668)）；unified 时再追加 `StateType.SWA_RING`。P 侧还会填 `kv_args.mla_compression_ratios = list(pool.compression_ratios)`（[prefill.py:197](../../../python/sglang/srt/disaggregation/prefill.py#L197) 附近）。

### 5.6 token 索引 → 内存地址映射（full vs 各子池）

这是整个设计最精巧的部分。**主 KV 通道用 full 页号"直接"寻址压缩池（靠 page_size 缩放做到 1:1 恒等），状态通道对每个组件各自翻译。**

**主 KV 通道 —— full 页号 = 压缩页号（1:1 恒等）**：因为各压缩池 page_size 已按压缩比缩放：`c4_page_size = page_size // 4`（[deepseek_v4_memory_pool.py:536](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L536)）、`c128_page_size = page_size // 128`（[:537](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L537)）。于是一个 full 页（256 token）压缩后正好 = 一个 c4 页（64 token）= 一个 c128 页（2 token），**full 页号 P ↔ 压缩页号 P 是恒等映射**，主通道无需翻译。

**状态通道 —— 各组件独立翻译**：P 与 D 用完全相同的 StateType 顺序保证位置对齐：
- `_swa_payload`（[prefill.py:1027](../../../python/sglang/srt/disaggregation/prefill.py#L1027)）：窗口内 full 索引 → `translate_loc_from_full_to_swa`（[deepseek_v4_memory_pool.py:639](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L639)）→ swa 页号。这组 swa 页号同时应用到 swa_kv_pool（item_len=一页）**和**压缩态 ring 池（item_len=一个 ring 块），swa 页号在 ring 池里恰好充当 ring 块索引。
- `_swa_ring_payload`（[prefill.py:1049](../../../python/sglang/srt/disaggregation/prefill.py#L1049)）：unified 场景，`ring_rows = req_pool_idx * ring_stride + (positions % ring_stride)`，逐行位置寻址。

映射机制总表：

| 通道/组件 | 发送索引来源 | 翻译方式 | item_len 粒度 |
|-----------|-------------|---------|--------------|
| 主 KV(c4/c128/indexer) | full token idx | `kv_to_page_indices` → full 页号，**恒等**映射到压缩页 | 一个压缩页 |
| 状态 SWA(swa_kv_pool) | full → swa 翻译表 | `translate_loc_from_full_to_swa` + 页化 | 一页 |
| 状态 SWA(compress_state) | 同上的 swa 页号 | swa 页号 = ring 块索引 | 一个 ring 块 |
| 状态 SWA_RING(unified) | `req_pool_idx*stride + pos%stride` | 逐行位置 | 一行 |

### 5.7 PP 指针切片

因为主 KV / 状态的扁平指针列表**按缓冲类型（压缩桶）组织而非按层**，PP（流水并行）切分需特殊处理。`get_mla_kv_ptrs_with_pp`（[common/conn.py:524](../../../python/sglang/srt/disaggregation/common/conn.py#L524)）检测到 `mla_compression_ratios` 就调 `_mla_slice_ptrs_for_pp`（[common/conn.py:553](../../../python/sglang/srt/disaggregation/common/conn.py#L553)），在每个分段（`[c4, c4_indexer, c128]` 或 `[swa, compress_state, indexer_compress_state]`）内部定位本 PP stage 的子区间。`KVArgs.mla_compression_ratios` 就是为此而设。

---

## 第六层：Kernel 计算 —— Prefill vs Decode

DSV4 的 attention 计算是 PD 两侧算力/带宽差异的根源。**P 和 D 共用同一个 attention forward 入口** `DeepseekV4AttnBackend.forward`（[deepseek_v4_backend.py:1309](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1309)），靠 `compress_ratio` 与 forward_mode 分流。

### 6.1 三层 compress_ratio 回顾

每层类型由 `config.compress_ratios[layer_id]` 决定（`MQALayer.__init__`，[deepseek_v4.py:316](../../../python/sglang/srt/models/deepseek_v4.py#L316)）：

| ratio | 语义 | compressor | indexer | 额外压缩池 |
|-------|------|-----------|---------|-----------|
| **0** | SWA 近窗全精度 | None | None | 无 |
| **4** | CSA 稀疏（C4 top-512/1024） | Compressor(ratio=4) | C4Indexer | c4_kv_pool + c4_indexer_kv_pool |
| **128** | HCA 全块 | Compressor(ratio=128) | None | c128_kv_pool |

所有层都有 SWA 写入与 `attn_mqa`（RadixAttention）；只有 ratio==4 层才有 indexer。

### 6.2 Prefill 阶段的 attention

**不是所有层都简单 dense**。Prefill 有两条 kernel 路径，由 query 规模阈值切换（[deepseek_v4_backend.py:1402-1405](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1402-L1405)）：

```python
if forward_batch.forward_mode.is_extend_without_speculative() and (
    q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD    # =11673
    or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
):
    return self._forward_prefill_sparse(...)         # 大 prefill：稀疏
# 否则 fall-through 到 flash_mla_with_kvcache        # 小 prefill：dense/paged
```

- **小 prefill（默认）**：`flash_mla.flash_mla_with_kvcache`（[deepseek_v4_backend.py:1436](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1436)），与 decode 同一 kernel，三种 ratio 层通过 `extra_k_cache`+`extra_indices` 差异化。
- **大 prefill（>11673 tokens）**：`_forward_prefill_sparse`（[deepseek_v4_backend.py:1458](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1458)），用 `flash_mla_sparse_fwd`，把 SWA 窗口 + 压缩缓存 dequant 进扁平 bf16 workspace 再按 rebased 索引消费。

因果掩码经 `expand_prefill_casually`（[deepseek_v4_backend.py:1559](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1559)）展开——每个 query token 生成自己的 `seq_lens_casual`。

**C4Indexer 在 prefill 也执行**（`forward_c4_indexer`，[indexer.py:433](../../../python/sglang/srt/layers/attention/indexer.py#L433)，P/D 都跑）：`fp8_paged_mqa_logits` 算 logits → `topk_transform_512` 选 top-k 页。prefill 特有：额外分配 `c4_sparse_raw_indices` 供稀疏 prefill 重建 workspace 索引。

### 6.3 Decode 阶段的 attention

**C4Indexer 选稀疏页**（`forward_c4_indexer`，decode 时 query 数=batch size）：
1. `compute_q` 融合 rope+hadamard+fp8 量化（[indexer.py:697](../../../python/sglang/srt/layers/attention/indexer.py#L697)）；
2. `logits = fp8_paged_mqa_logits(...)`（[indexer.py:553](../../../python/sglang/srt/layers/attention/indexer.py#L553)）—— fp8 paged MQA 点积打分；
3. `topk_transform_512`（[indexer.py:605](../../../python/sglang/srt/layers/attention/indexer.py#L605)）选 top-512/1024 页 → `c4_sparse_page_indices`。

**FlashMLA 融合 SWA + extra**（[deepseek_v4_backend.py:1336-1454](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1336-L1454)）：一次 kernel 同时消费两段 KV——SWA 近窗（dense，窗口 128）和 extra：

| ratio | extra 缓存 | extra 索引 | 稀疏? |
|-------|-----------|-----------|-------|
| 0 | 无 | 无 | 仅 SWA 近窗 dense |
| 4 | c4 pool | `c4_sparse_page_indices`（indexer 选出） | SWA dense + C4 top-k 稀疏 |
| 128 | c128 pool | `c128_page_indices`（全块） | SWA dense + C128 全块 |

decode 的压缩态写入：每步只在 `seq_len % ratio == 0` 边界 token 写（`CompressorDecodePlan`）。

### 6.4 P/D kernel 归属表

| Kernel / 函数 | Prefill | Decode | 位置 |
|---|:---:|:---:|---|
| `flash_mla_with_kvcache` | ✅（小 prefill） | ✅ | [deepseek_v4_backend.py:1436](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py#L1436) |
| `flash_mla_sparse_fwd` | ✅（大 prefill only） | ❌ | :1548 |
| `dequantize_k_cache_paged`（建 bf16 workspace） | ✅（仅稀疏 prefill） | ❌ | :1534 |
| `fp8_paged_mqa_logits`（indexer 打分） | ✅ | ✅ | [indexer.py:553](../../../python/sglang/srt/layers/attention/indexer.py#L553) |
| `topk_transform_512`（选稀疏页） | ✅ | ✅ | [indexer.py:605](../../../python/sglang/srt/layers/attention/indexer.py#L605) |
| `expand_prefill_casually`（因果展开） | ✅ | ❌ | :1559 |
| `set_swa_key_buffer_radix_fused_norm_rope`（SWA 写） | ✅ | ✅ | [memory_pool.py:1040](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py#L1040) |
| `CompressorPrefillPlan` / `CompressorDecodePlan` | ✅ prefill | ✅ decode | compress_old.py:107/192 |

**核心结论**：
1. 同一 attention forward 服务 P/D，差异在元数据构造与是否走 sparse-prefill 分支。
2. indexer 打分 P/D 都跑（仅 c4 层）；prefill 额外产 raw indices。
3. **稀疏 prefill（`flash_mla_sparse_fwd` + dequant workspace）是 decode 完全没有的路径**，只在超大 prefill（>11673 query tokens）启用。
4. compressor plan 类型不同：prefill 是 C++ ragged 因果 plan（每个边界都写），decode 是单 token 边界写。
5. SWA 写入 kernel 两阶段通用（每个新 token 都要进 SWA ring）。

---

## 附录：P/D 差异速查表与文件索引

### A. Prefill vs Decode 全维度对照

| 维度 | Prefill 实例 | Decode 实例 |
|------|-------------|-------------|
| **角色** | 全量前向，吃满算力 | 逐 token 自回归，吃满带宽 |
| **调度队列** | Bootstrap → Waiting → Inflight | Prealloc → Transfer → Waiting → RunningBatch（+ retracted 旁支） |
| **握手方向** | 被动等 D 推 `TransferInfo` | 主动 `GET /route` 查拓扑 + ZMQ 推 `dst_kv_indices` |
| **传输句柄** | `KVSender`（`send`/`poll`） | `KVReceiver`（`init`/`send_metadata`/`poll`） |
| **传输动作** | 后台 `transfer_worker` RDMA **单边写** | `_pre_alloc` 预分配显存 + `poll` 轮询到达 |
| **KVPoll 起点** | `create_sender` → Bootstrapping | `add` → Bootstrapping |
| **首 token** | 采样产出，随 aux 传给 D | 从 metadata buffer 读入 |
| **是否 forward** | 完整 prefill forward | PREBUILT 跳过 prefill，只做 decode |
| **Attention kernel** | FlashMLA dense/paged 或 sparse-prefill；因果展开 | `fp8_paged_mqa_logits`+`topk` 选页 + FlashMLA 融合 |
| **OOM 处理** | `optimistic_release_and_requeue` 重排队 | `retract_decode` → KV 存 CPU → `resume_retracted_reqs` |
| **结束** | `release_kv_cache` + `stream_output` 首 token | 生成 EOS/达长度后正常结束 |

### B. 关键文件索引

| 文件 | 关键内容 |
|------|----------|
| [disaggregation/prefill.py](../../../python/sglang/srt/disaggregation/prefill.py) | `PrefillBootstrapQueue`(:102)、`pop_bootstrapped`(:309)、事件循环(:469/:500)、`send_kv_chunk`(:971) |
| [disaggregation/decode.py](../../../python/sglang/srt/disaggregation/decode.py) | `DecodePreallocQueue`(:273)、`DecodeTransferQueue`(:1453)、`_pre_alloc`(:1288)、`process_decode_queue`(:1953) |
| [disaggregation/base/conn.py](../../../python/sglang/srt/disaggregation/base/conn.py) | `KVArgs`(:36)、`StateType`(:17)、`KVPoll`(:77) |
| [disaggregation/common/conn.py](../../../python/sglang/srt/disaggregation/common/conn.py) | `CommonKVManager`、`CommonKVBootstrapServer`(:1205)、`get_mla_kv_ptrs_with_pp`(:524) |
| [disaggregation/utils.py](../../../python/sglang/srt/disaggregation/utils.py) | `setup_state_kv_args`(:632)、`poll_and_all_reduce`、`_apply_metadata_gate`(:78) |
| [disaggregation/mooncake/conn.py](../../../python/sglang/srt/disaggregation/mooncake/conn.py) | `TransferInfo`(:68)、`transfer_worker`(:1198)、`MooncakeKVSender`(:1701) |
| [mem_cache/deepseek_v4_memory_pool.py](../../../python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py) | `get_contiguous_buf_infos`(:643)、`get_state_buf_infos`(:716) |
| [mem_cache/deepseek_v4_compress_state.py](../../../python/sglang/srt/mem_cache/deepseek_v4_compress_state.py) | `CompressStatePool`(:84)、`translate_from_swa_loc_to_state_loc`(:192) |
| [layers/attention/deepseek_v4_backend.py](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py) | `forward`(:1309)、`_forward_prefill_sparse`(:1458)、`expand_prefill_casually`(:1559) |
| [layers/attention/indexer.py](../../../python/sglang/srt/layers/attention/indexer.py) | `forward_c4_indexer`(:433)、`fp8_paged_mqa_logits`(:553)、`topk_transform`(:605) |
| [models/deepseek_v4.py](../../../python/sglang/srt/models/deepseek_v4.py) | `MQALayer.__init__`(:316)、`_forward_prepare`(:746) |

### C. 一句话总结

> DSV4 PD 分离下，一个请求先在 **P 实例**经 Bootstrap→Waiting→Inflight 三队列，做完整 forward 把 KV 写进**六个子池**并采样首 token；握手靠 HTTP bootstrap server 服务发现 + D 侧 ZMQ 反向推送目标地址；KV 经**两条通道**（主 KV 通道搬 c4/c128/indexer，状态通道搬 SWA + 压缩态 ring 块）由后台线程 RDMA 单边写到 D 预分配的显存，其中主通道靠 page_size 缩放实现 full↔压缩页恒等映射、ring 池以整块为粒度用 swa 页号寻址。随后请求在 **D 实例**经 Prealloc→Transfer→Waiting→RunningBatch，跳过 prefill forward，逐 token 做「SWA 近窗 dense + c4 稀疏/c128 全块」的 FlashMLA 融合注意力，OOM 时把六子池 KV 卸到 CPU 再恢复。





