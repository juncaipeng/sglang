# SGLang 上下文并行（Context Parallelism）方案梳理

> 本文梳理 SGLang 中**两套名字相近但机制完全不同**的上下文并行方案：
> - **Prefill CP（Attention Context Parallel，`attn_cp`）**：把一条 prefill 序列在 token 维度切给多张卡，每卡只算自己那一段 Q 的注意力，KV 靠 all-gather 补全。
> - **Decode DCP（Decode Context Parallel，`dcp`）**：把 decode 阶段一条序列的 KV cache 按 token 位置切给多张卡，每卡只存 `pos % dcp_size == rank` 的 KV，各自算局部注意力后用在线 softmax（LSE 修正）合并。
>
> 二者的参数、通信组、ForwardBatch 元数据字段都是独立的，**切勿混淆**。全文行号基于当前 `main`。

---

## 0. 一张表看清两套方案

| 维度 | Prefill CP（`attn_cp`） | Decode DCP（`dcp`） |
|---|---|---|
| 解决的阶段 | Prefill / Extend | Decode（也覆盖 MLA extend 的 prefix） |
| 切什么 | 序列 token 维（切 Q 和本地 KV 写入） | KV cache（按 token 位置切存储） |
| 切分规则 | zigzag（负载均衡块）/ interleave（轮询） | owner 规则 `pos % dcp_size == rank` |
| 通信组 | `_ATTN_CP`（`attn_cp_group`） | `_DCP`（`dcp_group`） |
| 主通信原语 | all-gather（补全 KV / 收拢 hidden） | all-gather LSE + all_reduce / reduce_scatter 输出 |
| 合并方式 | 每卡算半段 Q，拼接（无跨卡 softmax 合并） | 在线 softmax，LSE 修正合并局部输出 |
| ServerArgs | `attn_cp_size`, `enable_prefill_cp`, `cp_strategy` | `dcp_size`（别名 `--decode-context-parallel-size`） |
| ForwardBatch 元数据 | `attn_cp_metadata` | `attn_dcp_metadata`, `dcp_kv_mask` |
| 核心目录 | `layers/cp/`（v2）、`layers/utils/cp_utils.py`（v1） | `layers/dcp/` + `kernels/ops/attention/dcp_kernels.py` |
| 平台约束 | tp_size 需整除 attn_cp_size | HIP 全支持；CUDA 支持但禁投机；其他平台不支持 |

```
                    ┌─────────────────────────────────────────┐
                    │            一条请求的生命周期              │
                    └─────────────────────────────────────────┘
   Prefill 阶段                              Decode 阶段
   ┌──────────────┐                         ┌──────────────┐
   │ 序列 token 切 │  <- attn_cp (prefill)   │  KV 按位置切  │ <- dcp (decode)
   │ zigzag/轮询   │     _ATTN_CP 组          │ pos%N==rank  │    _DCP 组
   │ KV all-gather│                         │ LSE 合并输出  │
   └──────────────┘                         └──────────────┘
```

---

# 第一部分：Prefill CP（Attention Context Parallel）

## 1.1 为什么需要 zigzag：因果注意力的负载不均

Prefill 阶段做的是因果（causal）注意力：token *i* 只能看到 token `0..i`。如果把序列**连续等分**给 N 张卡（rank0 拿前 1/N，rank(N-1) 拿后 1/N），计算量严重失衡：

```
连续切分（naive），序列长 L，cp_size=4：
  rank0: tokens[0:L/4]      每个 token 平均看 ~L/8   →  最闲
  rank3: tokens[3L/4:L]     每个 token 平均看 ~7L/8  →  最忙（是 rank0 的 ~7 倍）
```

**zigzag** 的思路：把序列切成 `2*cp_size` 个块，每张卡拿 **一个靠前块 + 一个靠后块**，让"闲块"和"忙块"配对，各卡总计算量拉平。

```
zigzag 切分，cp_size=4 → 8 块：
  块号:   0    1    2    3    4    5    6    7
  归属:  cp0  cp1  cp2  cp3  cp3  cp2  cp1  cp0
         └──靠前(闲)──┘    └──靠后(忙)──┘
  rank r 拥有 块[r] 和 块[2*cp_size-1-r]
```

| rank | 靠前块 | 靠后块 | 合计负载 |
|---|---|---|---|
| cp0 | 0 | 7 | 闲+最忙 ≈ 均衡 |
| cp1 | 1 | 6 | 均衡 |
| cp2 | 2 | 5 | 均衡 |
| cp3 | 3 | 4 | 均衡 |

## 1.2 两代实现并存

SGLang 里同时有两套 Prefill CP 代码，数学完全一致（都是 zigzag），只是封装不同：

| | CP-v1（当前生产路径） | CP-v2（策略对象重构） |
|---|---|---|
| 位置 | `layers/utils/cp_utils.py` | `layers/cp/*` |
| 风格 | 一堆自由函数 | ABC + 策略类 |
| 开关 | `enable_prefill_context_parallel` + `prefill_cp_mode` | `SGLANG_ENABLE_CP_V2` + `enable_prefill_cp` + `cp_strategy` |
| zigzag | ✅ `prepare_context_parallel_metadata` (:470) | ✅ `ZigzagCPStrategy` (zigzag.py) |
| interleave（轮询） | ✅ `cp_round_robin_input_ids` (:174) | ⚠️ 桩代码，`NotImplementedError`（后续 PR） |

> CLI 层已把旧参数标记为 `[Deprecated]`，统一收敛到 `--cp-strategy {zigzag,interleave}`（server_args.py:968-974），旧 `--prefill-cp-mode` / `--enable-prefill-context-parallel` 通过 `_handle_legacy_cp_arguments`（server_args.py:5445-5485）反向填充。

## 1.3 zigzag 元数据构造（核心索引数组）

以 CP-v2 `ZigzagCPStrategy.build_metadata`（zigzag.py:114）为准，CP-v1 `prepare_context_parallel_metadata`（cp_utils.py:470）逻辑等价。

分块：每条序列长 `L` 切成 `cp_segment_num = 2*cp_size` 块（zigzag.py:132），`base = L // cp_segment_num`，前 `L % cp_segment_num` 块 `+1`（余数前置，zigzag.py:144-152）。

关键索引数组（zigzag.py 行号）：

| 数组 | 行号 | 作用 |
|---|---|---|
| `split_list` | :143 | 全序列所有块大小的扁平列表，供 `torch.split` 把整段 token 切成块 |
| `per_rank_actual_token` | :154 | 每 rank 拥有 token 数 = `Σ_seq (块[r] + 块[2*cp_size-1-r])` |
| `zigzag_index` | :165 | 本 rank 保留哪些块：`range(r, r+bs*seg, seg)`（靠前）+ `range(seg-r-1, bs*seg, seg)`（靠后） |
| `cp_reverse_index` | :175 | all-gather 后把 `[rank0前,rank0后,rank1前,...]` 逆置回原始 token 顺序的逆排列 |
| `reverse_split_len` | :188 | 匹配 all-gather 后拼接布局的 split 桶大小 |
| `kv_len_prev/next_list` | :202/:205 | 每半段 Q 因果需要看的 KV 长度（**已把 radix 前缀 `prefix_offsets` 折进去**，:134-138） |
| `cu_seqlens_*` / `max_seqlen_q_*` | :217/:263 | FlashAttention 需要的两半段 cu_seqlens 与 max 长度 |

不变式断言（zigzag.py:223-228）保证 `cp_reverse_index` 是真排列且 token 数守恒——切分/收拢是无损的。

前缀处理：radix cache 命中的前缀不参与切分，通过 `prefix_offsets` 折进 `kv_len_prev/next`，因果长度直接算对（zigzag.py:134-138）。

## 1.4 分片与收拢（序列维 = dim 0）

```
        整段 hidden [T, H]
             │  shard_hidden_states (zigzag.py:277)
             │    torch.split(split_list) → cat(zigzag_index) → pad_local_rows
             ▼
   本卡 hidden [T_local, H]  ──► model.forward（各卡独立跑）
             │  gather_hidden_states (zigzag.py:291)
             │    _all_gather_reorganized → split(reverse_split_len) → reorder(cp_reverse_index)
             ▼
        整段 hidden [T, H]（最后一层收拢，token 顺序还原）
```

- `shard_hidden_states`（zigzag.py:277）/ `shard_position_ids`（:284）：切给本 rank + `pad_local_rows` 补齐。
- `gather_hidden_states`（:291）/ `gather_kv_cache`（:302）：all-gather 后按 `reverse_split_len` 切、按 `cp_reverse_index` 重排回原序。
- `run_attention`（:319）：把本卡 Q 在 `total_q_prev_tokens` 处切成 `q_prev`/`q_next` 两半，各自带自己的 `cu_seqlens_q`、`kv_len`、`max_seqlen` 调 `attn_fn` **两次**，拼接。支持 `FLASH_ATTENTION` 与 `TRTLLM_MHA` 两种调用约定（:313-317）。
- `materialize_full_kv`（:365）/ `materialize_full_mla_kv`（:392）：all-gather 本地 K/V 补成完整布局，写进 KV pool。

## 1.5 通信原语：唯一是 all-gather

Prefill CP **只有 all-gather** 一种集合通信，没有 ring-attention / send-recv / all-to-all。

`_all_gather_reorganized`（zigzag.py:405）：本地行 pad 到 `max_len` → 在 `use_symmetric_memory`（NVLink 对称内存，:421）下分配输出 → `group.all_gather_into_tensor(gathered, x)`（:433，`group = get_parallel().attn_cp_group`）→ 按 `per_rank_logical_token` 去掉每 rank 的 padding。

> 设计取舍：每卡把**完整 K/V** all-gather 到本地，再对自己那半段 Q 重算注意力。相比 ring-attention（KV 流水传递）实现更简单，但每卡都持有全量 KV，显存换编程复杂度。

## 1.6 padding：让各卡形状一致

集合通信要求各 rank 贡献**相同 shape**，但各卡拥有 token 数不等，需补齐到公共物理长度（`layers/cp/padding.py`）：

| 函数 | 行号 | 作用 |
|---|---|---|
| `get_cp_padding_align_size` | :24 | 对齐粒度：zigzag（每卡 2 块）→ `attn_cp_size*2`，轮询 → `attn_cp_size` |
| `pad_logical_token_to_physical` | :35 | 算 `physical = ceil(max(logical)/align)*align`，把逻辑长度存进 `per_rank_logical_token`，`per_rank_actual_token`/`max_rank_len` 改写成统一物理长度 |
| `pad_local_rows` | :47 | 把本地 tensor 从逻辑长度 pad 零到 `per_rank_actual_token[0]` |

两段式：先"元数据 padding"（算统一长度）再"行 padding"（补零），all-gather 后用保留的逻辑长度剥掉 padding。

## 1.7 CP 组的创建与访问（`_ATTN_CP`）

CP 组即"注意力上下文并行组"，全局 `_ATTN_CP`（parallel_state.py:1720）。

创建逻辑（parallel_state.py:2214-2249）：
- `attn_tp_size = tp_size // attn_cp_size // attn_dp_size`（:2215）。
- 若 `attn_cp_size == tp_size`，直接令 `_ATTN_CP = _TP`（复用 TP 组，:2221-2222）。
- 否则按 `range(st, en, attn_tp_size)`（:2238）跨步取 rank——CP 伙伴是**同一 attn-TP 位置、CP 维不同**的 rank，`group_name="attn_cp"`（:2240）。
- MoE 交互：若 `attn_cp_size > moe_dp_size`，令 `_MOE_DP = _ATTN_CP`（:2295-2299），因为 CP 各卡在进 MoE 前必须先共享 token。

访问：所有 CP 代码经 `runtime_context.get_parallel()`（runtime_context.py:693）读取：
- `attn_cp_group`（:258）→ `get_attn_cp_group()`
- `attn_cp_size`（:185）→ `get_attn_context_model_parallel_world_size()`（parallel_state.py:2567）
- `attn_cp_rank`（:189）→ `get_attn_context_model_parallel_rank()`（parallel_state.py:2572）

## 1.8 attention backend 集成（FlashAttention）

`layers/attention/flashattention_backend.py`：
- `self.attn_cp_size = model_runner.ps.attn_cp_size`（:183）。
- **元数据 pad 修正**（:927-948）：非 CP-v2 的 MLA/MHA CP 且 `attn_cp_size>1` 时，把 `max_seq_len_k`/`page_table` 按 extend pad 增量加宽（因为 `prepare_mlp_sync_batch` 会把 extend token pad 到 `lcm(attn_tp_size, attn_cp_size)`），保证 FA3 因果读不越界。
- **KV 写入**（:1117-1173）：`is_cp_mode = is_context_parallel_extend() + attn_cp_metadata != None + attn_cp_size>1`。MLA CP-v2 走 `cp_strategy.materialize_full_mla_kv`（:1133），CP-v1 走 `set_mla_kv_buffer`（:1141）；Dense-MHA CP-v2 走 `materialize_full_kv`（:1159），CP-v1 走 `cp_allgather_and_save_kv_cache`（:1165）。
- **注意力计算 + 合并**（:1317-1359）：CP-v2 派发到 `cp_strategy.run_attention`（:1346），CP-v1 到 `cp_attn_forward_extend`（:1356）——两半段 Q 分别算再拼接。

`attn_cp_metadata` 由**模型侧**填充（不在 ForwardBatch 里）：deepseek_v2.py:2853、deepseek_v4.py:2500、qwen3_moe.py:986 等通过 `prepare_context_parallel_metadata(...)`；CP-v2 经 `layers/cp/utils.py:168` `strategy.build_metadata`。

## 1.9 DSA/DSV4 的 CP 变体：按"层"切 KV

针对 DeepSeek 稀疏注意力（DSA）/ MLA 模型，还有一条**按层切分 KV cache** 的 CP 优化（区别于按 token 切序列）。

`mem_cache/dsa_cache_layer_split.py`（`LayerSplitDSATokenToKVPool`，:56）：把 DSA GPU KV/indexer cache **按层**分给 CP 各 rank，每 rank 只物化自己拥有的层；非拥有者读该层时触发 owner 广播到一小块 remote scratch buffer。仅 DSA MLA 模型、PD prefill worker、开 prefill-CP 时启用（`enable_dsa_cache_layer_split`，server_args.py:976）。

- ownership：`_local_layer_idx`(:81)、`_is_layer_owned`(:89)、`_get_layer_owner_rank`(:94)。
- 广播通信：`_init_layer_broadcast_comm`(:115，用 `attn_cp_group`)、`_broadcast_tensor_from_owner`(:133)、`prefetch_kv_buffer`(:344，提前一层预取，广播与当前层计算重叠)。

`layers/communicator_dsa_cp.py`（247 行）：DSA prefill-CP 的通信层，用 `attn_cp_all_gather_into_tensor` / `attn_cp_reduce_scatter_tensor`。核心类 `DSACPLayerCommunicator`（:95），`dsa_cp_gather_hidden_states`（:71，all-gather，断言 attn_dp/attn_tp==1）、`dsa_cp_reduce_scatter_hidden_states`（:83）。

---

# 第二部分：Decode DCP（Decode Context Parallel）

## 2.1 核心机制：owner 规则 + 在线 softmax 合并

DCP 在 **decode 阶段**把一条序列的 KV cache 切给 CP 组各 rank：token 绝对位置 `pos` 归属 rank `pos % dcp_size`。每 rank 只存自己的 KV 分片、只算局部注意力，产出**局部输出 + 局部 LSE（log-sum-exp）**，再跨卡用在线 softmax 合并，结果与全量注意力**逐位等价**。

```
序列 KV (pos): 0  1  2  3  4  5  6  7  ...   (dcp_size=2)
owner:         r0 r1 r0 r1 r0 r1 r0 r1
                │            │
       ┌────────┘            └────────┐
   rank0 存 pos 0,2,4,6...     rank1 存 pos 1,3,5,7...
   算局部 attn → out0, lse0    算局部 attn → out1, lse1
                │            │
                └──── 合并 ───┘
    all-gather LSE → global_lse = logsumexp(lse0, lse1)
    scale_r = exp(lse_r - global_lse)
    out = Σ_r (out_r * scale_r)   ← all_reduce / reduce_scatter
```

两套实现合入同一目录（模块 docstring）：MHA/Triton 路径（PR #25090）、MLA/FlashInfer-MLA 路径（PR #14194）。

组拓扑（parallel_state.py:2190-2211）：DCP 组在**每个 TP 组内部**切——把每个 TP 组按 `dcp_size` 连续切块。约束 `tp_size % dcp_size == 0`（:2139），仅 CUDA/HIP（:2132）。

## 2.2 索引数学：owner 规则的实现（`layers/dcp/layout.py`）

| 函数 | 行号 | 作用 |
|---|---|---|
| `get_dcp_lens` | :23 | 每 rank 可见 KV 长度。`start=None`：`lens//dcp_size + (rank < lens%dcp_size)`；带 `start`：算首个拥有位置再向上取整 |
| `filter_dcp_local_kv_indices` | :44 | 只留 `idx % dcp_size == rank` 的索引，再 `// dcp_size` 压到本地存储索引 |
| `update_local_kv_lens_for_dcp` | :54 | `start=0` 的 in-place 版本；恒等式 `floor((len-rank-1)/N)+1 == len//N+(rank<len%N)` 已被单测逐位验证 |

`// dcp_size` 是关键：全局索引 → 本地紧凑存储索引，让每 rank 的 KV pool 只存自己那份。

## 2.3 planner：切分与规划（`layers/dcp/planner.py`）

两个构造器：

**`prepare_decode_context_parallel_metadata`（:32）** — MLA extend/prefix 路径，构建 all-gather buffer 的全局 KV 索引布局：
1. `dcp_enabled` 否则返回 None（:46）。
2. 算前缀 cumsum、填 `dcp_prefix_kv_indices`（:58-80）。
3. 起 Triton kernel `create_dcp_kv_indices`（:102）铺全局 prefix+extend 索引。
4. **实际按 rank 切**（:112-117）：`dcp_local_prefix_kv_indices = 全局[全局 % dcp_size == rank] // dcp_size`（即 owner 过滤 + 压缩）。
5. 分配 `dcp_kv_buffer`，组装 `DecodeContextParallelMetadata`。

**`plan_dcp_decode_metadata`（:136）** — 每 decode step（含 CUDA graph replay）调用：
1. clone `kv_lens` → 应用 `update_local_kv_lens_for_dcp`（owner 收缩）+ clamp（:145-147）。
2. 起 kernel `update_kv_lens_and_indices`（:170）把本 rank 拥有的项从全局 `kv_indices` 收拢到本地。
3. **原地写回**（:186-188）：`kv_indices`、`kv_lens`、`kv_indptr`——attention backend 透明地只看到本地分片。

## 2.4 LSE 合并（`layers/dcp/comm.py` + kernel）

这是 DCP 的数学核心：各卡局部注意力如何拼成全局精确结果。

**MHA 版 `cp_lse_ag_out_rs_mha`（comm.py:77，自然对数）**：
```
lses = all_gather(local_lse)                    # [N, B, H]
global_lse = logsumexp(lses, dim=0)             # comm.py:89
scale = exp(local_lse - global_lse)             # :90，nan 清零
out = all_reduce(local_out * scale)             # :93-94，跨卡求和
out = out[:, head_start:head_end, :]            # :96-100，每卡取自己 head 分片
```

**MLA 版 `cp_lse_ag_out_rs_mla`（comm.py:106，log2 空间 Triton kernel）**：
```
new_output (fp32, [H,B,D]) under use_symmetric_memory   # :123-127
lses = all_gather(local_lse)                            # :129
out = correct_attn_out(out, lses, rank, ctx, new_output)# :130，Triton 修正
out = reduce_scatter_along_dim(out, dim=0)              # :133
```

两版差异：MHA 用 `all_reduce + 显式 head 切片`（自然对数）；MLA 用 `reduce_scatter`（log2/exp2）。合并 kernel `_correct_attn_out_kernel`（dcp_kernels.py:153）在 log2 空间做在线 softmax：`lse_max=max`, `exp2`, `final_lse=log2(sum)+lse_max`，修正因子 `factor=exp2(local_lse-final_lse)`。

## 2.5 支撑 kernels（`kernels/ops/attention/dcp_kernels.py`）

| kernel | 行号 | 作用 |
|---|---|---|
| `create_triton_kv_indices_for_dcp_triton` | :33 | MHA 路径：算首个 owner 位置后按 `dcp_size` 跨步取，存 `data//dcp_size` |
| `create_dcp_kv_indices` | :79 | MLA 路径：铺 all-gather buffer 的全局 prefix+extend 索引 |
| `update_kv_lens_and_indices` | :115 | 按 rank 切/压缩，owner 映射 `offsets*dcp_world_size + rank`，存 `//dcp_world_size` |
| `_correct_attn_cp_out_kernel` | :153 | LSE 在线 softmax 合并（log2 空间） |
| `correct_attn_out` | :261 | host wrapper，规整 shape + 起 kernel |

## 2.6 KV pool 扩容（`mem_cache/kv_cache_configurator.py`）

DCP 下 KV pool 容量按 `dcp_size` 放大，让各卡分片聚合等于完整逻辑大小（:1464-1484）：
- `page_size==1 且 dcp_size==1` → 普通 `TokenToKVPoolAllocator`。
- 否则 `PagedTokenToKVPoolAllocator`，`max_total_num_tokens * dcp_size`（:1477）、`page_size * dcp_size`（:1478）。

## 2.7 ForwardBatch 字段与填充

`forward_batch_info.py`：
- `attn_dcp_metadata`（:538，`DecodeContextParallelMetadata`）、`dcp_kv_mask`（:541）。
- `dcp_kv_mask` 在 `init_new`（:892-899）填：`dcp_size>1 + out_cache_loc 存在 + HIP` → `positions % dcp_size == dcp_rank`。
- `attn_dcp_metadata` 在 `model_executor/runner/eager_runner.py`（:266-280）经 `model.prepare_context_parallel_metadata_for_dcp(...)` 填。

## 2.8 平台约束与验证（`_handle_dcp_validation`）

server_args.py:3157-3186（在 2948 调用）：
- `dcp_size < 1` → 报错；`dcp_size == 1` 直接返回。
- **HIP**：全支持。
- **CUDA**：支持，但**拒绝任何投机算法**（:3171-3179）。
- **其他平台**：不支持，报 "currently only supported on the AMD HIP platform"（:3180）。

其他门控：`dcp_size>1` 禁 prefill cuda graph（:3734）、post-capture KV sizing 要求 `dcp_size==1`（:4094）、unified memory 要求 `dcp_size==1`（:6874）。

---

# 第三部分：集成点与配置速查

## 3.1 ServerArgs 参数一览（`server_args.py`）

| 参数 | 行号 | 默认 | 说明 |
|---|---|---|---|
| `attn_cp_size` | 934 | 1 | 注意力上下文并行大小，别名 `--attention-context-parallel-size`，`resolvable=True` |
| `enable_prefill_cp` | 964 | False | 开 prefill CP，需配 `--cp-strategy` |
| `cp_strategy` | 968 | None | `zigzag` / `interleave` |
| `dcp_size` | 895/949 | 1 | decode CP，别名 `--decode-context-parallel-size`（**⚠️ 定义了两次，见已知问题**）|
| `enable_dsa_cache_layer_split` | 976 | False | DSA KV/indexer 按层切给 CP 各 rank，仅 mooncake 传输后端 |
| `enable_dsa_prefill_context_parallel` | 980 | - | 旧内部参数（no_cli），已归到 `--cp-strategy` |
| `prefill_cp_mode` | 983 | in-seq-split | 旧内部参数（no_cli），映射 zigzag |

CP 校验 `_handle_context_parallelism`（server_args.py:5487-5572）关键约束：
- `attn_cp_size>1`：断言 `tp_size % attn_cp_size == 0` 且 `tp_size % (dp_size*attn_cp_size) == 0`，禁 aiter allreduce fusion（:5533-5544）。
- `enable_prefill_cp` 必须设 `cp_strategy`（:5514）。
- prefill-CP 与 dsa-prefill-CP 互斥（:5519）。
- `CP_V2_DEFAULT_MODEL_CLASSES`（GptOss / MiMoV2Flash / Qwen3Moe / DeepseekV3）自动开 `SGLANG_ENABLE_CP_V2`（:5493）。
- 末尾调 `init_cp_strategy(self)`（:5570）绑定策略单例。

## 3.2 数据流对照总览

```
┌─────────────────────────── Prefill CP (attn_cp) ───────────────────────────┐
│ 模型首层 shard_hidden_states/position_ids  (序列切给本 rank, dim 0)          │
│   ↓ 各卡独立跑 decoder layers                                                │
│   每层 attention: run_attention → q_prev/q_next 两半各算一次 → 拼接           │
│   KV 写入: materialize_full_kv (all-gather 补全 → 写 pool)                    │
│   ↓                                                                          │
│ 模型末层 gather_hidden_states  (all-gather + cp_reverse_index 还原 token 序)  │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── Decode DCP (dcp) ───────────────────────────────┐
│ plan_dcp_decode_metadata: owner 规则切 kv_indices/kv_lens → 原地写回          │
│   ↓ 各卡只对本地 KV 分片算局部 attention → (out_r, lse_r)                     │
│ comm 合并: all_gather(lse) → global_lse → scale → all_reduce/reduce_scatter  │
│   ↓ 精确等价于全量注意力                                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

## 3.3 已知问题 / 注意点

1. **`dcp_size` 在 server_args.py 定义了两次**（:895 和 :949，内容相同），第二个 shadow 第一个——潜在冗余，值得清理。
2. **CP-v2 的 interleave 尚未实现**（`layers/cp/interleave.py` 全是 `NotImplementedError`）；轮询目前只在 CP-v1 `cp_round_robin_input_ids`（cp_utils.py:174）可用。
3. **`DecodeContextParallelMetadata` 仍是 `@dataclass`**（metadata.py:30），违反仓库 `no-dataclasses` 规则，但作为 #14194 的原样搬迁被 grandfather。
4. **CUDA 上 DCP 禁投机**（server_args.py:3171），HIP 无此限制。
5. **zigzag 要求每条序列 `≥ 2*cp_size`**（zigzag.py:102 `can_apply` / cp_utils.py:86 `can_cp_split`），不足则回退非-CP prefill。
6. Prefill CP **无 ring-attention**：每卡 all-gather 全量 KV 后本地重算，显存换实现简单度。

## 3.4 关键文件索引

| 文件 | 职责 |
|---|---|
| `layers/cp/base.py` | CP-v2 策略 ABC + 单例（`init_cp_strategy`）|
| `layers/cp/zigzag.py` | zigzag 切分数学 + 分片/收拢/run_attention |
| `layers/cp/interleave.py` | 轮询策略（桩，未实现）|
| `layers/cp/padding.py` | CP 整除对齐补齐 |
| `layers/utils/cp_utils.py` | CP-v1 生产路径（自由函数）|
| `layers/dcp/layout.py` | owner 规则索引数学 |
| `layers/dcp/planner.py` | DCP KV 切分与规划 |
| `layers/dcp/comm.py` | LSE 合并集合通信 |
| `layers/dcp/metadata.py` | `DecodeContextParallelMetadata` |
| `kernels/ops/attention/dcp_kernels.py` | DCP 索引构建 + LSE 合并 Triton kernel |
| `mem_cache/dsa_cache_layer_split.py` | DSA 按层切 KV pool |
| `layers/communicator_dsa_cp.py` | DSA prefill-CP 通信层 |
| `distributed/parallel_state.py` | `_ATTN_CP` / `_DCP` 组创建 |

