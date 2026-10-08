# 《context_parallel_architecture.md》基于最新代码的校验报告

> 校验基线：`main` 分支 HEAD `8a20062652`（2026-09-16）。
> 校验对象：[context_parallel_architecture.md](context_parallel_architecture.md)。
> 校验方式：四个并行探查任务逐条核对文档声明（函数/参数存在性、行号、语义），加上人工复核关键文件。

## 总体结论

| 维度 | 结论 |
|---|---|
| **核心机制描述** | ✅ 基本正确：zigzag 切分数学、owner 规则（`pos % dcp_size == rank`）、LSE 在线 softmax 合并、"Prefill CP 只有 all-gather"、DCP 组拓扑、通信组创建逻辑等数学/机制层面与代码一致 |
| **框架性内容** | ❌ **多处已失效**：仓库经历了 "CP V1 Deprecation" 系列重构（PR #36228、#38293 等）和 ServerArgs 参数大迁移，文档基于旧结构写成 |
| **行号引用** | ⚠️ 系统性漂移：约 80% 的行号引用已失效（偏差 +7 ~ +370 不等），原因是文件头部新增代码 |
| **内容覆盖** | ⚠️ 遗漏了 DCP A2A 通信后端、`bcg.py`、`cp_decode_attn_tp.py` 等新模块 |

**一句话结论：文档可以继续作为理解 CP/DCP 数学原理的入门材料，但所有涉及"CP-v1 与 CP-v2 并存"、server_args 参数位置、行号引用的内容都不可信，需要按本报告修订。**

---

## 一、重大内容性错误（必须修订）

### 1.1 "两代实现并存"的框架已彻底失效（影响 §1.2、§0 表格、§3.3）

文档的核心叙事是 "CP-v1（`layers/utils/cp_utils.py`，生产路径）与 CP-v2（`layers/cp/`，策略对象重构）并存"。**现在 CP-v1 已被完全删除**：

- `layers/utils/cp_utils.py` 现在只剩 **23 行**，是一个 deprecated shim（[cp_utils.py](../../../python/sglang/srt/layers/utils/cp_utils.py)），docstring 写明 "The legacy CP algorithms have been removed"，调用任何函数都会 `raise ValueError("Prefill CP on HIP/NPU/MUSA is deprecated...")`。
- 文档引用的 CP-v1 函数**全部不存在**：`prepare_context_parallel_metadata`(:470)、`cp_round_robin_input_ids`(:174)、`can_cp_split`(:86)、`cp_attn_forward_extend`、`cp_allgather_and_save_kv_cache`（仅 MUSA 后端残留 `musa_cp_attn_forward_extend`）。
- git log 证实删除系列：`85d39401c8 [CP V1 Deprecation 3.5/5]`、`b6c31b155c [CP V1 Deprecation 3/3] (#36228)`。
- 现行实现**只有策略对象一条路径**：`layers/cp/base.py` 的 `ContextParallelStrategy` ABC + `ZigzagCPStrategy` / `InterleaveCPStrategy` + `init_cp_strategy` / `get_cp_strategy` 单例。

### 1.2 interleave 不再是桩代码（影响 §1.2、§3.3 已知问题 #2）

文档说 "interleave ⚠️ 桩代码，`NotImplementedError`（后续 PR）"。**实际 interleave.py（273 行）是一套面向 DSA 后端的完整实现**：

- `build_metadata`(:71-96)：round-robin 均分 token；
- `_interleave_shard`(:108-123)：每 rank 取每隔 cp_size 的 token（即轮询切分，v1 `cp_round_robin_input_ids` 的后继）；
- `shard_per_request`(:128-160)：调 Triton kernel `dsa_cp_interleave_q_seqs_kernel`；
- `_gather_interleaved_tensor`(:172-212)：padding → all-gather → `index_select` 还原；
- `run_attention`(:222-233) 是刻意的 no-op（DSA 后端自己跑 attention）；
- 唯一的 `NotImplementedError` 是 `materialize_full_kv`(:254)。
- 支持的后端：`get_supported_attention_backend` 返回 `[CPAttentionBackendKind.DSA]`。

### 1.3 `SGLANG_ENABLE_CP_V2` 环境变量已删除（影响 §1.2 开关列）

`environ.py:1856-1858` 中该变量已 deprecated 且无替代，note 写明 "Strategy-based prefill context parallelism is now the only generic implementation"。文档中所有 "CP-v2 开关" 的说法失效。

### 1.4 ServerArgs 参数与校验逻辑大迁移（影响 §1.2 引用、§3.1 全表）

这是**涉及面最广**的结构变化：

- `ServerArgs` 字段声明已从 `server_args.py` **整体迁出**到 `python/sglang/srt/arg_groups/fields/*.py`，CP/DCP 参数在 `arg_groups/fields/parallel.py`：
  - `dcp_size` → `fields/parallel.py:59-65`（默认 1，别名 `--decode-context-parallel-size`）
  - `attn_cp_size` → `:100-107`（默认 1，别名 `--attention-context-parallel-size`）
  - `enable_prefill_cp` → `:149-152`（默认 False）
  - `cp_strategy` → `:153-159`（choices `{zigzag, interleave}`，**默认 None**，开 prefill CP 必须显式指定）
  - `enable_dsa_cache_layer_split` → `:161-164`（默认 False）
- 校验逻辑迁到 `arg_groups/parallel_hook.py`（函数名去掉下划线前缀，由 `arg_groups/pipeline.py` 管线调用）：
  - `handle_context_parallelism`（parallel_hook.py:32-121，pipeline.py:300 调用）
  - `handle_decode_context_parallelism`（parallel_hook.py:124-155，pipeline.py:182 调用）
- 文档引用的 `_handle_legacy_cp_arguments`(:5445-5485)、`_handle_context_parallelism`(:5487-5572)、`_handle_dcp_validation`(:3157-3186) 及全部 server_args.py 行号**均不存在**。

**已被删除的参数/机制**（文档 §3.1 表中仍列出）：
- `enable_dsa_prefill_context_parallel`（:980）→ 全仓库无匹配，已删除
- `prefill_cp_mode`（:983）→ 已删除，由 `cp_strategy` 替代且**无默认值**
- CLI 旧参数 `[Deprecated]` 标记 / legacy 反向填充 → 不存在
- `CP_V2_DEFAULT_MODEL_CLASSES`（GptOss/MiMoV2Flash/Qwen3Moe/DeepseekV3 自动开 CP_V2）→ 不存在。现行模型级限制改为：DeepseekV32 仅 interleave（parallel_hook.py:45-53）、MiMoV2* 仅 zigzag（:54-70）、DeepSeek-V4 仅 interleave（deepseek_v4_hook.py:204-209）、DSA 模型自动 `enable_dp_attention=True`。

**文档"已知问题 #1"（dcp_size 定义两次）已修复**：现在全仓库只有 `fields/parallel.py:59` 一处定义。

### 1.5 `dcp_kernels.py` 路径错误（影响 §0 表、§2.5、§3.4）

实际路径是 `python/sglang/kernels/ops/attention/dcp_kernels.py`（在 `sglang/kernels/` 包下），**不是**文档写的 `python/sglang/srt/kernels/ops/attention/dcp_kernels.py`。

### 1.6 `attn_cp_metadata` 的填充方描述错误（影响 §1.8）

文档说 "attn_cp_metadata 由模型侧填充：deepseek_v2.py:2853、deepseek_v4.py:2500、qwen3_moe.py:986 通过 `prepare_context_parallel_metadata(...)`"。**实际已集中化**：

- 填充点：`model_executor/runner/eager_runner.py:284-286` → `prepare_cp_forward`（layers/cp/utils.py:148-167）→ `strategy.build_metadata`（utils.py:162）。
- deepseek_v2.py:3124 是 `prepare_context_parallel_metadata_for_dcp`（**DCP 用**，非 prefill CP）；qwen3_moe.py:986 附近是 `__init__`，无 CP metadata 调用（只有 :1011-1017 的 `attn_cp_size % moe_dp_size == 0` 断言）；deepseek_v4.py 根本没有该函数，直接用 communicator 函数（:2627/:2670）。
- DCP 的 `prepare_context_parallel_metadata_for_dcp` 还被 kimi_k3.py(:3097/:3737)、kimi_linear.py(:833)、kimi_k25.py(:801) 实现——文档未提。

### 1.7 flashattention_backend.py 的 CP 集成描述大面积失效（影响 §1.8）

| 文档声明 | 实际情况 |
|---|---|
| `self.attn_cp_size = ...`(:183) | 内容对，实际 :197 |
| 元数据 pad 修正（:927-948，pad 到 `lcm(attn_tp_size, attn_cp_size)`） | **不存在**。:927-948 是 spec-decode target_verify 逻辑；整个文件只有 :197 一处 attn_cp。现行 pad 在 `layers/cp/padding.py` + `layers/attention/dsa/utils.py:197`。且 `prepare_mlp_sync_batch`（forward_batch_info.py:1373）实际只 pad 到 **attn_tp_size**（:1387），**不是 lcm** |
| `is_cp_mode`(:1117-1173) 四分支（v2/v1 × MLA/MHA） | 实际 :1289-1337，变量名 `cp_active`；只有两分支：MLA CP → `cp_strategy.materialize_full_mla_kv`(:1304)，Dense CP → `cp_strategy.materialize_full_kv`(:1324)。`set_mla_kv_buffer`(:1308) 是**非 CP** 分支；`cp_allgather_and_save_kv_cache` 已删 |
| CP-v1 派发 `cp_attn_forward_extend`(:1356) | **不存在**。`cp_strategy.run_attention` 在 :1490（MHA extend）与 :1766（MLA extend） |
| （隐含）FA backend 参与 DCP | **完全不涉及**：`attn_dcp/plan_dcp_decode_metadata` 在该文件零命中。DCP 在 `flashinfer_mla_backend.py:916` 和 `aiter_backend.py` |

另外 §1.4 中 "run_attention 把 Q 切两半各调 attn_fn 一次" **只对 FLASH_ATTENTION 路径成立**（zigzag.py:372-386）；TRTLLM_MHA 路径是**单次调用** + combined cu_seqlens + `use_zigzag_page_table=True`（:362-370），不切两半。

### 1.8 DCP 平台/投机约束已过时（影响 §2.8、§0 表、§3.3 #4）

- **平台**：现在 **HIP 和 CUDA 都允许**。实际报错文案（parallel_state.py:2541-2547，检查已移到组构建处而非参数校验）："...currently only supported on the AMD HIP platform **or CUDA platform**..."（文档引用的 "only supported on the AMD HIP platform" 是 :2486 的陈旧 docstring）。
- **"CUDA 禁投机"已不是一律禁**：现在是后端级规则——仅 `trtllm_mla` decode 后端禁 DCP+投机（attention_registry.py:82-90，建议改用 cutedsl_mla/tokenspeed_mla）；HiCache+dcp>1 仅支持 DSPARK 投机（hicache_hook.py:99）。
- `dcp_size<1` 报错、`dcp_size>1` 禁 prefill CUDA graph（现 cuda_graph_hook.py:248-251/:301-305）、post-capture KV sizing 要求 dcp_size==1（现 overrides.py:1803-1814）等内容**仍然成立**，只是位置变了。
- 新增校验文档未提：`dcp_comm_backend` 为 a2a/fi_a2a 时要求 dcp_size>1（parallel_hook.py:133-139）、fi_a2a 仅 CUDA（:140-146）、`dcp_replicate_q_proj` 限制（:147-155）。

### 1.9 layout.py 两个函数语义被混写（影响 §2.2）

文档说 `filter_dcp_local_kv_indices`(:44) "只留 `idx % dcp_size == rank` 再 `// dcp_size`"。实际是**两个不同函数**：

- `filter_dcp_local_kv_indices`（实际 :57-65）：**只做掩码筛选**，docstring 明说 "Selection only; the caller collapses via translate_dcp_read_ids"；
- `maybe_dcp_kernel_indices`（实际 :44-54）：做 strided 切片 + `// dcp_size`（`indices[dcp_rank::dcp_size] // dcp_size`）。

即文档把后者（:44）的行号和前者的名字绑在一起了。另有文档未提的 `filter_dcp_local_chunk_kv_indices`(:68-84)。

### 1.10 DSA layer split 启用条件不完整（影响 §1.9）

`enable_dsa_cache_layer_split` 的实际启用条件（model_hook.py:267-310 + layers/cp/utils.py:45-58）比文档严格得多：**必须 `--enable-prefill-cp` 且 `--cp-strategy interleave`**、仅 PD prefill worker、仅 mooncake 传输后端、不支持 PP>1、use_mla_backend + is_deepseek_dsa、非 draft worker。文档只说了 "开 prefill-CP" 和 "仅 mooncake"，漏了 interleave 要求。

### 1.11 其他内容性更正

- `_handle_context_parallelism` 中 prefill-CP 与 attn_dp>1 的一般性互斥**已不存在**，方向甚至相反：DSA 模型开 `enable_prefill_cp` 会**强制** `enable_dp_attention=True`（deepseek_v2.py:95）；仅 interleave DSA CP 有 `assert cfg.dp_size == 1`（:105-107）。
- 末尾初始化调用的就是 `init_cp_strategy(...)`（parallel_hook.py:115-121），不是 `get_cp_strategy()`。
- tp 整除约束的中文表述应更正为 "tp_size 必须能被 attn_cp_size 整除"（`cfg.tp_size % view.attn_cp_size == 0`），且第二个约束是 `tp_size % (dp_size * attn_cp_size) == 0`。
- `ParallelState` 不含 `attn_cp_group` 字段：它是 frozen dataclass（parallel_state_wrapper.py，`attn_cp_rank` :15、`attn_cp_size` :16），group 是 `ParallelContext` 的 property（runtime_context.py:435-436）。
- dcp_kernels.py:79 现在是**新 kernel `create_mla_kv_page_table_for_dcp`**，`create_dcp_kv_indices` 实际在 :130（文档行号指向了别的函数）。
- LSE 合并 kernel 名确认为 `_correct_attn_cp_out_kernel`（文档 §2.4 写 `_correct_attn_out_kernel` 是笔误，§2.5 的写法正确）。

---

## 二、行号漂移对照表（内容正确但行号失效）

> 漂移原因：各文件头部新增代码（parallel_state.py 新增约 370 行；comm.py 新增 deprecated accessor + `_ag_lse` helper；dcp_kernels.py 新增 `create_mla_kv_page_table_for_dcp`；dsa_cache_layer_split.py 新增辅助类等）。

### zigzag.py（整体 +7 ~ +36）

| 符号 | 文档 | 实际 |
|---|---|---|
| `ZigzagCPStrategy.build_metadata` | :114 | **:121** |
| `cp_segment_num = 2*cp_size` | :132 | **:139** |
| `split_list` | :143 | **:150** |
| `per_rank_actual_token` | :154 | **:161** |
| `zigzag_index` | :165 | **:172** |
| `cp_reverse_index` | :175 | **:182** |
| `reverse_split_len` | :188 | **:195** |
| `kv_len_prev/next_list` | :202/:205 | **:204/:205** |
| prefix_offsets 折入 | :134-138 | **:140-145（定义）/:209-214（折入）** |
| 不变式断言 | :223-228 | **:232-237** |
| `shard_hidden_states` / `shard_position_ids` | :277/:284 | **:301/:308** |
| `gather_hidden_states` / `gather_kv_cache` | :291/:302 | **:315/:326** |
| `run_attention` | :319 | **:343** |
| `materialize_full_kv` / `materialize_full_mla_kv` | :365/:392 | **:396/:428** |
| `_all_gather_reorganized` | :405 | **:441** |
| symmetric memory 判断 | :421 | **:457-461** |
| `all_gather_into_tensor` | :433 | **:469** |

### padding.py（偏差 1 行）

| 符号 | 文档 | 实际 |
|---|---|---|
| `get_cp_padding_align_size` | :24 | :24 ✅ |
| `pad_logical_token_to_physical` | :35 | **:34** |
| `pad_local_rows` | :47 | **:46** |

### layers/cp/utils.py

`strategy.build_metadata` 调用：文档 :168 → 实际 **:162**。

### parallel_state.py（整体 +370 ~ +420）

| 符号 | 文档 | 实际 |
|---|---|---|
| `_ATTN_CP` / `_DCP` 全局 | :1720 | **:2094 / :2095** |
| `_ATTN_CP` 组创建 | :2214-2249 | **:2637-2669**（`== tp_size → _TP` :2641-2642；range 跨步 :2658；group_name :2665） |
| `attn_tp_size` 公式 | :2215 | **移至 runtime_context.py:148**（`derive_attention_widths`） |
| DCP 组构建 | :2190-2211 | **:2599-2620** |
| `tp_size % dcp_size` 约束 | :2139 | **:2548-2552** |
| CUDA/HIP 检查 | :2132 | **:2541-2547** |
| `_MOE_DP = _ATTN_CP` | :2295-2299 | **:2714-2718** |
| `get_attn_context_model_parallel_world_size` | :2567 | **:2988** |
| `get_attn_context_model_parallel_rank` | :2572 | **:2993** |

### runtime_context.py

`get_parallel()`：文档 :693 → 实际 **:1355**；`attn_cp_group` 是 ParallelContext property（:435-436）而非 :258 字段。

### layers/dcp/（comm.py +10、kernels +47、planner ±1-8）

| 符号 | 文档 | 实际 |
|---|---|---|
| `cp_lse_ag_out_rs_mha` | :77 | **:87**（logsumexp :98、scale :99-100、all_reduce :104、head 切片 :106-110） |
| `cp_lse_ag_out_rs_mla` | :106 | **:116**（new_output :140-142、correct :145-152、reduce_scatter :153） |
| `create_triton_kv_indices_for_dcp_triton` | :33 | :33 ✅ |
| `create_dcp_kv_indices` | :79 | **:130**（:79 是新 kernel `create_mla_kv_page_table_for_dcp`） |
| `update_kv_lens_and_indices` | :115 | **:166** |
| `_correct_attn_cp_out_kernel` | :153 | **:199** |
| `correct_attn_out` | :261 | **:308** |
| `prepare_decode_context_parallel_metadata` | :32 | :33（±1 内子项：None :47-48、cumsum :64、kernel :103、rank 切 :115-118） |
| `plan_dcp_decode_metadata` | :136 | :137（clone :153-155、kernel :183-193、写回 :194-196） |
| `update_local_kv_lens_for_dcp` | :54 | **:87** |
| `get_dcp_lens` | :23 | :23 ✅ |

### 其他文件

| 符号/位置 | 文档 | 实际 |
|---|---|---|
| DCP KV pool 容量放大（kv_cache_configurator.py） | :1464-1484 | **:2090-2099**（:1464-1484 现在是 NPU SWA pool 的 head 切分） |
| ForwardBatch `attn_dcp_metadata` / `dcp_kv_mask` | :538/:541 | **:618/:621** |
| `dcp_kv_mask` 填充（init_new） | :892-899 | **:990-998**（条件确认：`attn_dcp_size>1 and out_cache_loc is not None and is_hip()`） |
| eager_runner 填 attn_dcp_metadata | :266-280 | **:299-317** |
| `LayerSplitDSATokenToKVPool` | :56 | **:194**（:57 是辅助类 LayerSplitIndexKeyCache） |
| 其成员函数（`_local_layer_idx` 等 6 个） | :81-:344 | **:219-:460**（+100~+138） |
| communicator_dsa_cp.py 总行数 | 247 行 | **239 行**（`dsa_cp_gather_hidden_states` :63、`reduce_scatter` :75、`DSACPLayerCommunicator` :87） |
| flashattention_backend.py `attn_cp_size` 赋值 | :183 | **:197** |

---

## 三、经校验仍然正确的内容

以下核心机制描述与最新代码一致，可放心保留（仅行号需更新）：

1. **zigzag 切分数学**：`2*cp_size` 块、余数前置、`range(r, r+bs*seg, seg)` + `range(seg-r-1, bs*seg, seg)` 公式、radix 前缀经 `prefix_offsets` 折入 `kv_len_prev/next`、排列/守恒断言——全部与代码一致（zigzag.py:139-237）。
2. **分片/收拢流程**：`shard → 各卡独立跑 → all-gather → cp_reverse_index 还原`，padding 两段式（元数据 padding + 行 padding）。
3. **"Prefill CP 只有 all-gather"**：全目录 grep 验证仅 `zigzag.py:469` 与 `interleave.py:206` 两处 all-gather，无 ring-attention/send-recv/all-to-all。
4. **`_all_gather_reorganized` 机制**：pad 到 max_len → symmetric memory 分配 → all_gather_into_tensor → 按 `per_rank_logical_token` 去 padding。
5. **通信组创建逻辑**：`attn_cp_size == tp_size` 复用 `_TP`；跨步取 rank；`attn_cp_size > moe_dp_size` 时 `_MOE_DP = _ATTN_CP`（"CP 各卡进 MoE 前必须先共享 token"）；DCP 组在每个 TP 组内部按 dcp_size 连续切块。
6. **DCP owner 规则** `pos % dcp_size == rank` 与 `// dcp_size` 本地索引压缩：在 layout/dcp_kernels/planner/forward_batch_info/comm 多处体现。
7. **LSE 合并数学**：MHA 版 all_gather LSE → logsumexp → scale → all_reduce + head 切片（自然对数）；MLA 版 log2 空间 Triton kernel（`correct_attn_out`）+ reduce_scatter。`_correct_attn_cp_out_kernel` 的 `lse_max/exp2/factor` 描述正确。
8. **DCP KV pool 容量按 dcp_size 放大**：`max_total_num_tokens * dcp_size`、`page_size * dcp_size`（内容对，位置改为 kv_cache_configurator.py:2090-2099）。
9. **`dcp_kv_mask` 填充条件含 HIP**：确认 `is_hip()` 是条件之一（forward_batch_info.py:990-998）。
10. **`attn_dcp_metadata` 由 eager runner 填充**：eager_runner.py:299-317，调 `model.prepare_context_parallel_metadata_for_dcp(...)`。
11. **模块 docstring 提及 PR #25090（MHA/Triton）与 #14194（MLA/FlashInfer-MLA）**：在 `__init__.py:15-17` 等 5 处确认。
12. **`DecodeContextParallelMetadata` 仍是 `@dataclass`**（metadata.py:30，字段 `dcp_kv_indptr/dcp_kv_buffer/dcp_kv_indices/dcp_local_prefix_kv_indices/dcp_extend_prefix_lens_sum`）——文档"已知问题 #3"仍成立。
13. **zigzag 要求每条序列 ≥ 2*cp_size**：`can_apply`（zigzag.py:109-119）确认 `num_tokens < cp_size*2` 或任一 extend 序列不足即回退（但 cp_utils.py:86 的引用已失效）。
14. **`enable_prefill_cp` 必须配 `cp_strategy`**（parallel_hook.py:72-75 报错）。
15. **`is_context_parallel_extend()`**：forward_batch_info.py:146-155，EXTEND/MIXED 判定。
16. **DSA CP 变体的机制描述**（按层切 KV + owner 广播 + prefetch 重叠；`dsa_cp_gather_hidden_states` 断言 `attn_dp==1 and attn_tp==1`；deepseek_v2 用 `DSACPLayerCommunicator`，deepseek_v4 只用两个自由函数）。

---

## 四、文档完全遗漏的新内容

1. **DCP A2A 通信后端家族**（文档只写了 all-gather LSE + all_reduce/reduce_scatter 两条路径）：
   - 新参数 `--dcp-comm-backend`（a2a / ag_rs / fi_a2a 三选一，parallel_hook.py:133-146）；
   - `comm.py`：`init_fi_a2a_workspace`(:393，FlashInfer MNNVL，要求 Blackwell 同一 MNNVL domain)、`dcp_a2a_lse_reduce`(:468)、`_dcp_fi_a2a_lse_reduce`(:534)；
   - `dcp_kernels.py`：`dcp_pack_a2a_send`(:445)、`dcp_lse_combine_triton`(:586)、`_lse_weighted_combine_cpu`(:638)。
2. **`create_mla_kv_page_table_for_dcp`**（dcp_kernels.py:80-126）：页表化 owner 切片，支持 v2p 虚拟页转换。
3. **fp8 KV cache 经 uint8 视图传输**（comm.py:164-169、:518-523）。
4. **`plan_dcp_decode_metadata` 的 `init_metadata_replay`/`fast_decode_kwargs` 免同步路径**（planner.py:157-174，CUDA graph replay 时用 `kv_len_arr_cpu` 避免同步）。
5. **`layers/cp/bcg.py`**（317 行，文档未提）：Breakable CUDA Graph——让 prefill CP 在 BCG 下运行。限定 `enable_prefill_cp && pp_size==1 && attn_cp_size==tp_size && cp_strategy=="zigzag" && prefill backend=="trtllm_mha"`；重放 CP-local 图主体 + eager 执行全局 gather/logits。
6. **`layers/cp/cp_decode_attn_tp.py`**（213 行，文档未提）：CP 模式下 `tp_size=1` 时，decode 阶段把注意力权重矩阵沿 CP rank 切片以复刻普通 TP 行为；白名单 DeepseekV4* / GlmMoeDsa*；纯本地切片无集合通信。
7. **comm.py 顶部 3 个 deprecated 兼容 accessor**（:46-73）与公共 helper `_ag_lse`（:76-84）。
8. **DCP 模型方法扩展**：`prepare_context_parallel_metadata_for_dcp` 还在 kimi_k3 / kimi_linear / kimi_k25 中实现。
9. **模型级 CP 策略限制**（替代已删的 CP_V2 默认模型列表）：DeepseekV32 仅 interleave、MiMoV2* 仅 zigzag、DeepSeek-V4 仅 interleave、DSA 模型自动开 dp attention。

---

## 五、修订建议（按优先级）

1. **删除整个 "CP-v1 / CP-v2 两代并存" 框架**：§1.2 表格、§0 表格相关行、§3.3 已知问题 #2、所有 `layers/utils/cp_utils.py` 引用。改为单一策略对象架构（base.py ABC + zigzag/interleave 策略 + `init_cp_strategy` 单例）。
2. **重写 §3.1 参数表**：全部指向 `arg_groups/fields/parallel.py` 与 `arg_groups/parallel_hook.py`；删除已不存在的参数（`prefill_cp_mode`、`enable_dsa_prefill_context_parallel`）和"已知问题 #1"（已修复）；补充 `--dcp-comm-backend`。
3. **重写 §1.8**：反映集中式填充（eager_runner → prepare_cp_forward）、`cp_active` 两分支、run_attention 的 TRTLLM_MHA 单次调用路径；删除 lcm 说法（实际 pad 到 attn_tp_size）。
4. **更新 §2.8 平台约束**：HIP + CUDA 均支持；禁投机改为后端级规则（仅 trtllm_mla）。
5. **更新 §1.9 DSA 启用条件**：补 `--cp-strategy interleave` 硬性要求。
6. **修正 dcp_kernels.py 路径**（`python/sglang/kernels/ops/attention/`）与 layout.py 函数职责描述。
7. **全局刷新行号**：按本报告第二节对照表；或将行号引用弱化为"函数名 + 文件"以降低维护成本。
8. **补充第四节列出的新模块**：尤其是 DCP A2A 后端（`--dcp-comm-backend`）与 bcg.py，它们已经是独立的功能面。

---

## 附：本次校验涉及的关键文件（现行位置）

| 文件 | 说明 |
|---|---|
| `python/sglang/srt/layers/cp/{base,zigzag,interleave,padding,utils,bcg,cp_decode_attn_tp}.py` | Prefill CP 策略实现（v1 已删） |
| `python/sglang/srt/layers/utils/cp_utils.py` | 23 行 deprecated shim（仅保 NPU/MUSA 可导入） |
| `python/sglang/srt/layers/dcp/{layout,planner,comm,metadata}.py` | DCP 实现 |
| `python/sglang/kernels/ops/attention/dcp_kernels.py` | DCP Triton kernels（注意在 kernels 包） |
| `python/sglang/srt/arg_groups/fields/parallel.py` | CP/DCP ServerArgs 字段（新位置） |
| `python/sglang/srt/arg_groups/parallel_hook.py` | `handle_context_parallelism` / `handle_decode_context_parallelism`（新位置） |
| `python/sglang/srt/distributed/parallel_state.py` | `_ATTN_CP`/`_DCP` 组创建 |
| `python/sglang/srt/distributed/parallel_state_wrapper.py` | ParallelState frozen dataclass |
| `python/sglang/srt/runtime_context.py` | `get_parallel()`、ParallelContext、宽度推导 |
| `python/sglang/srt/model_executor/runner/eager_runner.py` | attn_cp_metadata / attn_dcp_metadata 填充点 |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | FA backend CP 集成（无 DCP） |
| `python/sglang/srt/mem_cache/{dsa_cache_layer_split,kv_cache_configurator}.py` | DSA 按层切分 / DCP pool 扩容 |
| `python/sglang/srt/layers/communicator_dsa_cp.py` | DSA prefill-CP 通信层 |
