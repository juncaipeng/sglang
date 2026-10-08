# SGLang 集中式部署 DeepSeek V4 + MTP 执行逻辑详解

> 本文梳理 **集中式（co-located，非 PD 分离）** 部署下，SGLang 运行 DeepSeek V4（DSV4）模型并开启 MTP（Multi-Token Prediction，多 token 预测投机解码）时的完整执行逻辑。由浅入深，覆盖启动配置 → draft 模型结构 → 三阶段执行 → 调度器集成 → KV cache → CUDA graph → 完整生命周期。
>
> 姊妹文档：`docs/pjc_1/dsv4_pd_disaggregation_request_lifecycle.md`（PD 分离场景）、`docs/pjc_1/speculative_decoding_overview.md`（投机解码总览）、`docs/pjc_1/deepseek_v4_cache_management.md`（DSV4 缓存管理）。本文只讲**集中式 + MTP**这一条路径。

> **引用风格说明（重要）**
>
> 本文所有代码引用一律**先给符号名**（`文件名:类名.方法名()`），不给行号 —— 因为行号会随重构漂移，符号名不会。
> 少数确实没有外层命名符号可指的地方（模块级常量、超长 `__init__` / 超长 `if` 链里的内联块），才附一个「快照行号」，形如
> `` `pool_configurator.py:DSV4PoolConfigurator.__init__` 内（快照 :825-832）``。
> 这些残留行号基于 `main` @ `33ed29a0ee`，仅作定位辅助，**以符号名为准**。
>
> 文中路径若无特别说明，均相对 `python/sglang/srt/`。

---

## 第 0 层：一句话概览

DSV4 的 MTP 在 SGLang 里**不是独立的算法**，而是被规约为 **EAGLE 投机解码**，其中 draft（草稿）模型就是 DSV4 权重里自带的 **NextN 层**（1 个 transformer 层）。整个「草稿 → 验证 → 草稿续推」三阶段在**单进程单 event loop** 内、由 `EAGLEWorkerV2` 一次 `forward_batch_generation` 调用内部完成，与 PD 分离场景（跨实例、KV 传输、PREBUILT）完全不同。

```
集中式 DSV4 + MTP 的宏观数据流

  ┌─────────────────────────────────────────────────────────────┐
  │  Scheduler（单实例，event_loop_overlap）                        │
  │                                                                │
  │  self.model_worker  ──swap──▶  EAGLEWorkerV2                    │
  │                                  │  (内部持 target_worker)       │
  │                                  ▼                              │
  │   一次 forward_batch_generation() 内部：                         │
  │                                                                │
  │   EXTEND 迭代：  target prefill ──▶ draft prefill               │
  │   DECODE 迭代：  ① draft ──▶ ② verify ──▶ ③ draft_extend        │
  └─────────────────────────────────────────────────────────────┘
```

关键前置事实（后文逐一展开）：

| 事实 | 值 / 说明 |
|---|---|
| MTP 算法归一 | `NEXTN`/`MTP` → `EAGLE`（topk==1 → 链式而非树） |
| 默认投机参数 | `(num_steps, topk, num_draft_tokens) = (3, 1, 4)` |
| draft 模型 | DSV4 自带 NextN 层，`DeepseekV4ForCausalLMNextN`，仅 1 层 |
| draft 层压缩比 | `COMPRESS_RATIO_NEXTN_LAYER = 0`（仅 SWA 全精度，无 c4/c128 压缩池） |
| 强制后端 | `attention_backend = "dsv4"`，`page_size = 256`，`kv_cache_dtype = fp8_e4m3` |
| 执行框架 | 永远走 spec **V2** worker；overlap 与非 overlap 共用同一 V2 worker |
| 另有一条 draft 路径 | checkpoint 若自带 DSpark draft，则走 `DeepseekV4ForCausalLMDSpark`（算法 `DSPARK`）；NextN 是 else 兜底分支，详见 §2.1 |

---

## 第 1 层：启动配置——参数如何被"归一"

用户通常这样启动（示意）：

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V4 \
    --speculative-algorithm NEXTN \
    --tp 8
```

从 CLI 到内部生效，经历两个 hook 的联合改写。

### 1.1 DSV4 硬约束——已拆成「声明式 override + 命令式校验」两段

这里有一个**近期的重要重构**：DSV4 的强制默认值原先集中写在 `arg_groups/deepseek_v4_hook.py` 里（命令式赋值），现在绝大部分已迁到 **override registry**（声明式）。`apply_deepseek_v4_defaults` 的 docstring 自己说明了这次搬迁：

```python
# arg_groups/deepseek_v4_hook.py: apply_deepseek_v4_defaults()
"""Residual imperative arm of the DeepSeek V4 defaults.

The attention/page/window/MoE-runner declarations moved to the override
registry (arg_groups/overrides.py: _deepseek_v4_overrides) and the
kv-cache dtype default to the resolution pipeline
(_deepseek_v4_kv_cache_dtype, invoked below at its legacy slot). This
keeps, at the legacy slot: the ROCm env fill (env-write policy), the
max_running_requests fill ... and the validations.
"""
```

所以查这些默认值时**不要再去 hook 里找**：

> 小坑：上面这段 docstring 说 override 在 `arg_groups/overrides.py`，但实际 `_deepseek_v4_overrides()` 已经进一步拆到 `arg_groups/model_overrides/deepseek_v4.py` 了（docstring 自己没跟上）。以下表为准。

| 参数 | 值 | 当前位置 |
|---|---|---|
| `attention_backend` | `"dsv4"` | [`arg_groups/model_overrides/deepseek_v4.py:_deepseek_v4_overrides()`](../../../python/sglang/srt/arg_groups/model_overrides/deepseek_v4.py) |
| `page_size` | `256`（NPU 为 128） | 同上（`_deepseek_v4_overrides()` 内的 NPU 分支） |
| `swa_full_tokens_ratio` | `0.1`（仅当仍为默认时） | 同上 |
| MoE runner 相关声明 | — | 同上 |
| `kv_cache_dtype` | `auto` → `fp8_e4m3`（NPU → `bfloat16`），断言 ∈ `{fp8_e4m3, bfloat16}` | `arg_groups/overrides.py:_deepseek_v4_kv_cache_dtype()`（由 hook 在 legacy slot 用 `run_post_process_pass()` 调起） |
| `max_running_requests` | 默认 256 | `arg_groups/deepseek_v4_hook.py:apply_deepseek_v4_defaults()`（**仅此一项仍留在 hook**，因为投机 hook 是该字段的后续写者） |
| ROCm 下关闭 `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL` | — | 同上（env-write 策略只能命令式做） |

投机相关的关键断言仍留在 hook 里（`apply_deepseek_v4_defaults()` 末尾）：

```python
if cfg.speculative_algorithm is not None:                 # ← cfg = resolving_view(server_args)
    assert cfg.speculative_algorithm in (
        "EAGLE",
        "DSPARK",                                         # ← DSPARK 现在也被允许
    ), f"Only EAGLE and DSPARK speculative algorithms are supported for {model_arch}"
    if cfg.speculative_algorithm == "EAGLE":               # ← topk 断言嵌在 EAGLE 分支内
        assert (
            cfg.speculative_eagle_topk == 1
        ), f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"
```

即：**DSV4 允许 `EAGLE` 或 `DSPARK`；`topk == 1` 的强制只对 EAGLE 生效**。本文讲的 MTP 就是 `EAGLE` 这一支。topk==1 意味着 draft 产出是一条**链**（chain），不是树（tree），后续所有 kernel 走 chain 快路径。

> 注意 snippet 里用的是 `cfg = resolving_view(server_args)` 这个解析视图，不是 `server_args` 本身 —— 配置解析管线（`resolving_view` / `declare_resolution` / `run_post_process_pass`）让「谁是某字段的最终写者」变得有序，这也是默认值能从 hook 搬到 registry 的前提。


### 1.2 算法别名归一与自动参数（`arg_groups/speculative_hook.py`）

- **NEXTN → EAGLE 别名**：`speculative_hook.py:_resolve_speculative_algorithm_alias()` 把 `"NEXTN"`/`"EAGLE"` 都映射为 `"EAGLE"`（仅 Gemma4 assistant draft 才提升为 `FROZEN_KV_MTP`）。所以 `--speculative-algorithm NEXTN` 与 `EAGLE` 在 DSV4 下等价。
- **draft 路径自动填充**：DSV4 在 `speculative_hook.py:_handle_eagle_family()` 的自动 draft-path 架构列表里，若未显式给 `--speculative-draft-model-path`，则默认取 `model_path` 本身——因为 NextN 权重就打包在主 checkpoint 里，**无需单独下载 draft 模型**。
- **spec-v2 恒定**：`SGLANG_ENABLE_SPEC_V2` 已移除（`speculative_hook.py` 里只剩一段移除告知）；投机永远跑 V2 worker，overlap 与否由 `--disable-overlap-schedule` 决定。
- **自动投机参数**：`speculative_hook.py:_auto_choose_speculative_params()` 对 DeepSeek 系返回 `(3, 1, 4)`（DSV4 落到函数末尾的 `else` 分支同样得 `(3,1,4)`）：`num_steps=3, topk=1, num_draft_tokens=4`。
- **num_draft_tokens 校正**：topk==1 时强制 `num_draft_tokens = num_steps + 1`（`speculative_hook.py:_handle_eagle_family()` 内的校正块）。所以 `3 → 4`，链长恒等式 `num_draft_tokens == num_steps + 1` 成立。

> **不变式（贯穿全文）**：DSV4 + MTP ⇒ `topk == 1`，`num_steps == 3`，`num_draft_tokens == 4`，链式投机，dsv4 后端，page_size 256，fp8 KV。
>
> ⚠️ 其中 `num_steps == 3` 这条**只在未开启自适应投机时严格成立**。运行期 `num_steps` 可被自适应控制器改小（甚至为 0），详见 §3.5。凡是做显存/池 sizing 的地方都必须用上界 `max_speculative_num_draft_tokens()`，不能用当前值。


---

## 第 2 层：draft 模型——DSV4 的 NextN 层

MTP 的 draft 不是外挂模型，而是 checkpoint 里紧跟 target 主干之后的**一个 NextN transformer 层**。代码在 [`models/deepseek_v4_nextn.py`](../../../python/sglang/srt/models/deepseek_v4_nextn.py)（约 300 行）。

### 2.1 架构改写与加载

启动时，`configs/model_config.py:ModelConfig._config_draft_model()` 把 draft 的架构串改写。**注意这里现在是 if/else 两条路**：

```python
# configs/model_config.py: ModelConfig._config_draft_model()
if checkpoint_bundles_dspark_draft(self.hf_config) and (
    self.speculative_algorithm in (None, "DSPARK")
):
    self.hf_config.architectures[0] = "DeepseekV4ForCausalLMDSpark"   # ← DSpark draft 优先
else:
    self.hf_config.architectures[0] = "DeepseekV4ForCausalLMNextN"    # ← NextN 是兜底分支
    self.hf_config.num_nextn_predict_layers = 1
```

- checkpoint 若自带 DSpark draft **且**算法为 `None`/`DSPARK`，走 `DeepseekV4ForCausalLMDSpark`（另一条 draft 实现，不在本文范围）。
- 其余情况（包括 `--speculative-algorithm NEXTN`/`EAGLE`）走 `DeepseekV4ForCausalLMNextN` 并声明 `num_nextn_predict_layers = 1`。**本文讲的就是这一支。**

权重加载在 `models/deepseek_v4.py:DeepseekV4ForCausalLM.load_weights(is_nextn=True)` 分支：
- `assert num_nextn_layers == 1`——只支持 1 个 NextN 层。
- checkpoint 里 NextN 权重以 `mtp.` 前缀存放，经 `models/deepseek_v4.py:remap_weight_name_to_dpsk_hf_format()` 重映射到 `model.layers.{num_hidden_layers}.` 下：`emb.tok_emb→embed_tokens`、`norm.weight→shared_head.norm.weight`、`head.→shared_head.head.weight`。
- 加载 target 时会**跳过**所有 `layer_id ≥ num_hidden_layers` 或 `mtp` 前缀的权重；加载 NextN 时只保留单个 NextN 层的权重（两处都在 `load_weights()` 的权重遍历循环里）。target 与 draft 权重互不干扰。

### 2.2 两个类

| 类 | 位置 | 作用 |
|---|---|---|
| `DeepseekV4ModelNextN` | `models/deepseek_v4_nextn.py` | draft 骨干（embedding + 融合投影 + 1 个 decoder 层 + mHC head） |
| `DeepseekV4ForCausalLMNextN` | 同文件（`EntryClass`） | 继承 target `ForCausalLM`，但只建 `model` + `lm_head` + `logits_processor`，不建完整层栈（`__init__` 里直接调 `nn.Module.__init__(self)` 绕过父类构造） |

`DeepseekV4ModelNextN.__init__()` 关键组件：

| 组件 | 作用 |
|---|---|
| `embed_tokens` | 词嵌入（DP attention 下关闭 TP） |
| `enorm` / `hnorm`（RMSNorm） | 分别归一「嵌入分支」「hidden-state 分支」 |
| `e_proj` / `h_proj`（无 bias） | 把两分支投影到 hidden_size |
| `decoder`（`DeepseekV4DecoderLayer`） | 唯一的 transformer 层，`layer_id=0`，`is_nextn=True`，`compress_ratio_override=COMPRESS_RATIO_NEXTN_LAYER`（=0） |
| `hc_head_*` 参数 + `DeepseekV4ModelNextN.hc_head()` 方法 | DSV4 特有的多头组合（mHC）输出头，float32 sigmoid 门控加权和 |
| `shared_head.norm` | lm_head 前的最终 norm |

### 2.3 NextN 前向：EAGLE 条件融合的本质

`DeepseekV4ModelNextN.forward()` 的核心是 **把 target 上一层 hidden state 与「下一个输入 token 的嵌入」融合** 作为 draft 输入：

```python
# models/deepseek_v4_nextn.py: DeepseekV4ModelNextN.forward() 的融合块
hidden_states = self.embed_tokens(input_ids)               # 下一个 token 的嵌入
hc_flat = forward_batch.spec_info.hidden_states.view(...)   # target 传来的 hidden state
h_proj_out, _ = self.h_proj(self.hnorm(hc_flat))            # hidden-state 分支
e_proj_hidden_states, _ = self.e_proj(self.enorm(hidden_states))  # 嵌入分支
hidden_states = e_proj_hidden_states[:, None, :] + h_proj_hidden_states  # 融合相加
```

这就是 EAGLE/MTP 的标准 conditioning：draft 用 target 的隐状态 + 下一 token 嵌入来"续写"。DSV4 的特殊之处是这一切在 `hc_mult` 个子状态维度上进行（多头组合）。随后（均在同一个 `forward()` 内）：
- 单层 decoder 前向。
- 因只有一层，`hc_post` 融合必须就地完成（注释明确 "NextN has a single decoder layer"）。
- `hc_head()` 把 `hc_mult` 子状态收缩为一个向量，再过 `shared_head.norm`。
- 返回 `(hidden_states, pre_hc_head)`，其中 `pre_hc_head` 供下一 draft step 作 hidden-state 携带。

### 2.4 为什么 draft 层"更简单"——只用 SWA

因为模块级常量 `models/deepseek_v4_nextn.py:COMPRESS_RATIO_NEXTN_LAYER = 0`（快照 :50）经 `compress_ratio_override` 注入到唯一的 decoder 层：

- **解析位置注意**：compress_ratio 的解析已从 `MQALayer.__init__` 上移到基类 `models/deepseek_v4.py:MqaAttentionBase.__init__()`：`compress_ratio = override if override is not None else config.compress_ratios[layer_id]`，并新增了断言
  ```python
  # models/deepseek_v4.py: MqaAttentionBase.__init__()
  assert self.compress_ratio in (0, 4, 128)      # ← 只允许这三档
  ```
  `MQALayer` 现在只是把 `compress_ratio=compress_ratio_override` 透传给 `super().__init__(...)`。override=0 非 None，故 draft 层 `compress_ratio == 0`。
- ratio 0 时**不建** compressor / C4Indexer：`MqaAttentionBase.__init__()` 里 `self.compressor = None; self.indexer = None; if self.compress_ratio in (4, 128): ...`，两者保持 `None`。
- 因此 draft 只读 **SWA 近窗全精度 KV**，没有 C4Indexer 稀疏打分、没有 c128 全块读取。这是 draft 比 target 简单一大截的根本原因。


---

## 第 3 层：核心执行——三阶段 draft → verify → draft_extend

所有投机逻辑收敛在 [`speculative/eagle_worker_v2.py`](../../../python/sglang/srt/speculative/eagle_worker_v2.py) 的 `EAGLEWorkerV2.forward_batch_generation()`。它内部持有 `target_worker`（即真正的 DSV4 模型 runner）和 `draft_worker`（`EagleDraftWorker`，其 `.draft_worker` 才是 NextN 层的 `TpModelWorker`）。入口按 forward mode 二分：

```python
# speculative/eagle_worker_v2.py: EAGLEWorkerV2.forward_batch_generation()
def forward_batch_generation(
    self,
    batch: ScheduleBatch,
    on_publish: Optional[Callable] = None,
    grammar_barrier: Optional[Callable] = None,        # ← grammar overlap 用，见 §4.7
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> GenerationBatchResult:
    if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
        # ===== EXTEND / PREFILL 迭代 =====
        target_capture_mode = (                        # ← 注意：不再写 batch.capture_hidden_mode
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        )
        batch_output = self.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=target_capture_mode,   # ← 以 kwarg 传入
        )                                              # ① target prefill
        batch_output.new_seq_lens = batch.seq_lens      # spec_v2 约定
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        batch_output.next_draft_input = \
            self.draft_worker._draft_extend_for_prefill(...)   # ② draft prefill
        return batch_output
    else:
        # ===== DECODE 迭代（稳态三阶段）=====
        self.activate_step_by_batch(batch.seq_lens.shape[0])   # 自适应步数，见 §3.5
        if self.speculative_num_steps == 0:                    # ← 自适应可压到 0 步
            verify_input = self._build_trivial_verify_input(batch)
        else:
            verify_input = self.draft_worker.draft(batch)       # ① draft
        batch.spec_info = verify_input
        batch_output = self.verify(batch, grammar_barrier=grammar_barrier)   # ② verify
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        self.draft_worker._draft_extend_for_decode(batch, batch_output)      # ③ draft_extend
        return batch_output
```

> **一个已被规范禁止的旧写法**：早期代码是 `batch.capture_hidden_mode = CaptureHiddenMode.FULL`，直接写 `ScheduleBatch` 字段。现在改为把 `capture_hidden_mode` 作为 **kw-only 参数**传给 `managers/tp_worker.py:TpModelWorker.forward_batch_generation()`。原因是仓库规范 `.claude/rules/forward-batch-init-new-purity.md` 要求 `ForwardBatch.init_new()` 及其上游不得就地修改传入的 `ScheduleBatch`——一旦 overlap 下同一个 `ScheduleBatch` 被跨迭代复用，就地写字段会造成难以定位的串味。读旧文档/旧 PR 时注意这个差别。

对 DSV4（`num_steps=3 > 0`），DECODE 迭代跑完整三阶段。下面逐阶段展开。

### 3.1 EXTEND 迭代（首次 prefill）

发生在请求刚进入 running batch、需要预填充 prompt 时：

1. **target prefill**：DSV4 主模型跑一遍 prefill，`CaptureHiddenMode.FULL` 让每个位置的 hidden state 都被保留（draft 需要它作条件）。这一步走完整 DSV4 kernel——小 prefill 用 FlashMLA dense、大 prefill 用 sparse 路径、C4Indexer 打分等（详见 DSV4 缓存文档）。
2. **draft prefill** = `eagle_worker_v2.py:EagleDraftWorker._draft_extend_for_prefill()`：
   - input_ids 左移一位并补 chunked-prefill 尾 token（模块级函数 `eagle_worker_v2.py:_eagle_prefill_tail_tokens()`，见 PR #26329）；
   - `batch.spec_info = EagleDraftExtendInput(hidden_states=target_hidden_states, num_tokens_per_req=1, ...)`；
   - `capture_hidden_mode = CaptureHiddenMode.LAST`（STANDALONE 时为 `NULL`），跑 draft NextN 前向；
   - 组装下一迭代的 `EagleDraftInput`：非拒绝采样路径用 `fast_topk()`，拒绝采样路径用 `fast_sample()`（见 §3.6），携带 `bonus_tokens=next_token_ids` 与 `hidden_states`。

结果：产出首个 draft 输入，供后续 DECODE 迭代使用。**没有第一次 verify**——首 token 由 target prefill 直接给出。

### 3.2 DECODE 阶段①：draft（多步草稿生成）

`eagle_worker_v2.py:EagleDraftWorker.draft()` → `EagleDraftWorker.draft_forward()`。

注意这里有一次**逻辑外提**：`draft()` 的准备与建树都变成了 [`speculative/eagle_worker_common.py`](../../../python/sglang/srt/speculative/eagle_worker_common.py) 的**模块级函数**，`draft()` 只是编排：

- `eagle_worker_common.py:prepare_for_draft()` 返回 `(forward_batch, can_run_decode_cuda_graph)`；eager 路径下若 `num_steps > 1` 才初始化 draft attention backend 元数据（注释：1-step draft 只 sample，无需 init）。
- `EagleDraftWorker.draft_forward()` 多步循环 `for i in range(self.speculative_num_steps)`：
  - `select_top_k_tokens()` 选取链 token；
  - **skip-last-forward**：`if i == self.speculative_num_steps - 1: break`。因为 draft prefill 已给了 1 个 token，这里只需再产 `num_steps-1` 个；
  - 每步用**独立的 per-step attention backend**：`ForwardContext(attn_backend=self.draft_attn_backend.attn_backends[i])`；
  - **topk==1 快路（已换成融合 kernel）**：
    ```python
    # eagle_worker_v2.py: EagleDraftWorker.draft_forward()
    elif self.topk == 1 and not _is_hip:
        if _is_cuda:
            topk_p, topk_index = draft_topk1_postprocess(      # ← 融合 kernel：argmax + positions.add_(1)
                logits_output.next_token_logits,               #   + 写链槽位，一次做完
                forward_batch.positions,
                draft_tokens_topk1,
                i + 1,
            )
        else:
            topk_index = torch.argmax(...)                     # ← 非 CUDA 回退分支才是裸 argmax
            topk_p = torch.ones_like(...)
            forward_batch.positions.add_(1)
    ```
    DSV4 在 CUDA 上恒走 `draft_topk1_postprocess` 这条融合路径。
- **topk==1 链组装**：用预分配的运行不变常量 `_topk1_score_indices_prealloc` / `_topk1_parents_prealloc`，`draft_tokens = cat(token_list)`，跳过一般树路径的排序/gather。这两个 buffer 与链长不变式断言都在基类里：`speculative/base_spec_worker.py:BaseSpecWorker._rebuild_topk1_chain_buffers()`（断言 `num_draft_tokens == num_steps + 1`）。
- 最后 `eagle_worker_common.py:build_eagle_verify_input()` 内部调 `build_tree_kernel_efficient()` 写树 mask（链是退化的树），返回 `EagleVerifyInput`。

**draft 用的 DSV4 后端**：`layers/attention/deepseek_v4_backend.py:DeepseekV4MultiStepBackend`，它为每个 draft step 建一个 `DeepseekV4AttnBackend` 实例（在其 `__init__` 的循环里）。该后端在 `DeepseekV4AttnBackend.__init__()` 里 `assert self.topk in [0, 1], "MTP Topk > 1 not supported for DeepSeek V4"`，并设 `self.mtp_enabled = self.topk > 0`、`self.swa_page_size = 128`（注意与 pool 的 page_size 256 区分）。因 draft 层 ratio 0，`extra_k_cache` 恒为 None（ratio 4/128 才建），FlashMLA 只读 SWA 近窗。多步元数据只遍历 `range(self.speculative_num_steps - 1)`（`DeepseekV4MultiStepBackend.init_forward_metadata()`），与 skip-last 对齐。

### 3.3 DECODE 阶段②：verify（target 验证）

**结构变化（重要）**：`EAGLEWorkerV2.verify()` 现在只是一个十几行的**薄委托**，真正的实体是模块级函数 [`eagle_worker_common.py:run_eagle_verify()`](../../../python/sglang/srt/speculative/eagle_worker_common.py)：

```python
# speculative/eagle_worker_v2.py: EAGLEWorkerV2.verify()
def verify(self, batch: ScheduleBatch, grammar_barrier=None):
    return run_eagle_verify(                       # ← 实体在 eagle_worker_common.py
        batch,
        target_worker=self.target_worker,
        req_to_token_pool=self.req_to_token_pool,
        token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
        topk=self.topk,
        num_draft_tokens=self.speculative_num_draft_tokens,
        metadata_ready_pre_pad=False,
        finalize_tree_path=True,
        grammar_barrier=grammar_barrier,
    )
```

`run_eagle_verify()` 内部依次是：

- **`num_tokens_per_req` 不在这里赋值了**。它是 `speculative/eagle_info.py:EagleVerifyInput` 的字段，默认哨兵 `-1`，在 `EagleVerifyInput.__post_init__()` 里自动回填为 `draft_token_num`：
  ```python
  # speculative/eagle_info.py: EagleVerifyInput
  num_tokens_per_req: int = -1        # -1 auto-fills from draft_token_num.

  def __post_init__(self):
      super().__init__(SpecInputType.EAGLE_VERIFY)
      if self.num_tokens_per_req < 0:
          self.num_tokens_per_req = self.draft_token_num
  ```
  DSV4 下结果仍是 4（= `num_steps + 1`），一次性把 4 个候选 token 交给 target 并行验证；但**赋值点在 `__post_init__`，不在 verify**，且构造方可以显式覆盖。
- 在 plan stream 里 `eagle_prepare_for_verify()`，随后 `update_verify_buffers_to_fill_after_draft()` 补算依赖 draft 输出的 buffer。
- **target 验证前向**：`target_worker.forward_batch_generation(batch=None, forward_batch=verify_forward_batch, is_verify=True)`。这一步 target DSV4 对 4 个候选 token 做一次 forward（TARGET_VERIFY 模式，走完整 c4/c128/SWA kernel）。
- **采样接受**：`eagle_sample()` 返回 `(predict, accept_lens, accept_index)`；`new_seq_lens = batch.seq_lens + accept_lens`。链式接受用拒绝采样 `coin*q < p`。
- **DSV4 专属的压缩态回滚**（通用 EAGLE 路径没有这一步，见 §3.4）：`clear_unaccepted_c128_draft_states()`。
- **bonus token**：`fill_bonus_tokens_func()`（注意函数名带 `_func` 后缀）：
  ```python
  # speculative/eagle_worker_common.py: run_eagle_verify()
  accept_tokens = predict[accept_index]
  bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
  # stride = accept_tokens per-req width = accept_index.shape[1]
  # (spec_steps + 1); NOT num_draft_tokens, wrong for topk > 1 trees.
  fill_bonus_tokens_func(
      accept_tokens, accept_lens, bonus_tokens,
      accept_index.shape[1],        # ← stride 用这个，不是 num_draft_tokens
      bs,
  )
  ```
  bonus token 就是接受链尾部之后 target 免费给出的那一个 token。DSV4 下 `accept_index.shape[1]` 与 `num_draft_tokens` 恰好都是 4，但语义上必须用前者（topk>1 的树会不等）。
- **topk==1 恒等**：树路径压缩 `eagle_worker_common.py:_finalize_accept_tree_path()` 仅在 `topk > 1` 时才跑；注释说得很直白——"topk == 1 needs nothing here: the accepted path is already the front chain, so the whole compaction is an identity transform"。
- 返回 `GenerationBatchResult(accept_lens=..., new_seq_lens=..., next_draft_input=EagleDraftInput(bonus_tokens=...), extra_keep_alive_refs=[verify_forward_batch], ...)`。最后那个 `extra_keep_alive_refs` 是 overlap 正确性的关键，见 §3.8。

### 3.4 DSV4 专属：verify 后清理未被接受的 c128 压缩态

这是 DSV4 + MTP 与通用 EAGLE 最容易被忽略的差异。verify 阶段 target 会把**整条乐观尾巴**（4 个候选 token）的 c128 压缩态写进 state ring，但最终只有 `accept_lens` 个 token 被接受。剩下那些必须清掉，否则下一轮读到脏压缩态：

```python
# speculative/eagle_worker_common.py: run_eagle_verify()
clear_unaccepted_c128 = getattr(
    token_to_kv_pool_allocator.get_kvcache(),
    "clear_unaccepted_c128_draft_states",
    None,
)
if clear_unaccepted_c128 is not None and not batch.forward_mode.is_idle():
    clear_unaccepted_c128(
        batch.req_pool_indices,
        batch.seq_lens,
        accept_lens,          # ← 只保留前 accept_lens 个，其余作废
        num_draft_tokens,
    )
```

只有 DSV4 的 KV pool（`mem_cache/deepseek_v4_memory_pool.py:DeepSeekV4TokenToKVPool`）提供 `clear_unaccepted_c128_draft_states()`，其他模型的 pool 没这个方法所以取到 `None`、整段跳过。这也解释了 §6.4 为什么投机时要把 c128 state ring 从 128 抬到 256——ring 必须能同时容纳整条乐观尾巴。

### 3.5 自适应投机：`num_steps` 是运行期可变的

文档 §1.2 给的 `num_steps == 3` 是**配置期的上界/默认值**，运行期会被自适应控制器调整。相关符号（全在 `speculative/eagle_worker_v2.py:EAGLEWorkerV2` 上）：

| 符号 | 作用 |
|---|---|
| `EAGLEWorkerV2.activate_step_by_batch()` | 每次 DECODE 迭代开头按 batch size 选定本轮步数 |
| `EAGLEWorkerV2.build_adaptive_runtime_state()` | 构造自适应运行状态（步数 / draft token 数等） |
| `EAGLEWorkerV2.apply_runtime_state()` / `_override_worker_state()` | 把状态落到 worker 的 `speculative_num_steps` 等字段上 |
| `EAGLEWorkerV2.on_verify_complete_cpu()` | 接收 CPU 侧的接受统计，喂给控制器（调用方见 §4.6） |

两个直接后果：

1. **`num_steps` 可以被压到 0**。此时 draft 阶段整体跳过，`EAGLEWorkerV2._build_trivial_verify_input()` 造一个"只有 bonus token"的平凡 verify input，`_stub_skipped_draft_extend()` 补上被跳过的 draft_extend 出参。这是 §3 入口 snippet 里那个 `if self.speculative_num_steps == 0:` 分支的用途。
2. **所有显存/池 sizing 必须用上界**，即 `max_speculative_num_draft_tokens()`，而不是当前的 `speculative_num_draft_tokens`。§6 里 `allocation_sizing.py` 与 `pool_configurator.py` 用的都是这个上界函数——否则自适应一升档就会越界写。

### 3.6 `speculative_use_rejection_sampling`：DSV4 恰好是它的适用对象

`--speculative-use-rejection-sampling` 开启时，链式投机改用严格拒绝采样。它在 `EagleDraftWorker.__init__()` 里就断言 `topk == 1`（"Chain speculative sampling supports only topk=1"）——DSV4 恒 topk==1，**天然满足**。差异点：

- `EagleDraftWorker._draft_extend_for_prefill()` 里用 `fast_sample(probs, num_samples=1)` 取样，而非 `fast_topk(probs, self.topk, dim=-1)`；
- 返回的 `EagleDraftInput` 会带上 `draft_probs=probs`（普通路径为 `None`），供 verify 端做 `p/q` 比值判定；
- `EagleDraftWorker.alloc_memory_pool()` 会额外要求 draft 与 target 同词表。

### 3.7 DECODE 阶段③：draft_extend（草稿续推，热身下一轮）

`eagle_worker_v2.py:EagleDraftWorker._draft_extend_for_decode()`：用本轮**接受的 token**再跑一次 draft，为下一轮 draft 准备 hidden state / KV：

- 计数按规范命名（见 `.claude/skills/speculative-naming/SKILL.md`）：`num_correct_drafts = batch_result.accept_lens - 1`（不含 bonus）、`num_accept_tokens = batch_result.accept_lens`（含 bonus）；
- `num_tokens_per_req = self.speculative_num_draft_tokens = 4` 填满整个链宽度；
- 用 `select_index` gather 出接受路径那一行；
- 跑 draft-extend 前向（**CUDA 上现在会走 CUDA graph**，见 §7.1）；
- topk==1 用 `argmax`（PR #26358 后限 CUDA——ROCm argmax tie-break 会破坏 FP8 MTP 选择）。

**draft_extend 用的后端**：`speculative/draft_utils.py:DraftBackendFactory._create_dsv4_prefill_backend()` 在 CUDA 分支返回普通 `DeepseekV4AttnBackend(self.draft_model_runner, skip_prefill=False)`；元数据用 `need_compress=False`（`DeepseekV4AttnBackend.init_forward_metadata_draft_extend()`，注释 "Draft extend is SWA-only"）——draft 层 ratio 0 无压缩态可 materialize。

### 3.8 verify buffer 的跨迭代保活

`run_eagle_verify()` 返回的 `GenerationBatchResult` 带一个 `extra_keep_alive_refs=[verify_forward_batch]`，代码注释解释得很清楚：

```python
# speculative/eagle_worker_common.py: run_eagle_verify() 返回前
# verify_forward_batch transitively holds verify-time GPU tensors
# (draft_token / out_cache_loc / ...) that must outlive the imminent
# batch.input_ids rebind in prepare_for_draft_extend.
# Scheduler pins it in batch_record_buf for the 2-iter window.
```

即：verify 期的 GPU 张量（`draft_token` / `out_cache_loc` 等）必须活过紧随其后的 `prepare_for_draft_extend()` 里对 `batch.input_ids` 的 rebind。调度器侧在 `scheduler.py:Scheduler.run_batch()` 里把这些引用挂进 `batch_record_buf`，保活两个迭代窗口。**这是 overlap + spec 的正确性前提**，不是可选优化——去掉它会出现随机的显存复用踩踏。


---

## 第 4 层：调度器集成——集中式 event loop 如何驱动三阶段

投机不改变调度器的骨架。集中式下用的是普通的重叠调度 event loop，投机只是"塞进" `run_batch` 里的一个分支。

### 4.1 worker 偷梁换柱

`scheduler.py:Scheduler.init_attention_backend_and_cuda_graph()` 末尾（快照 :1067-1070）把 `model_worker` 换成 draft worker：

```python
if self.spec_algorithm.is_none():
    self.model_worker = self.tp_worker          # 非投机：直接用 target
else:
    self.model_worker = self.draft_worker       # 投机：用 EAGLEWorkerV2（内部持 target）
```

于是调度器只知道调用 `self.model_worker.forward_batch_generation(...)`——这一个调用内部就完成了 target-forward + draft + verify + draft_extend。[`scheduler.py:Scheduler.maybe_init_draft_worker()`](../../../python/sglang/srt/managers/scheduler.py) 在 `__init__` 里于 `init_tp_model_worker()` 之后创建 draft worker，并把 `target_worker=self.tp_worker` 塞进 `draft_worker_kwargs`；具体类由 `spec_algorithm.create_worker(self.server_args)` 决定（DSV4 + MTP ⇒ `EAGLEWorkerV2`）。初始化顺序：TP worker → draft worker → memory pool → attention backend → cuda graph → worker swap。

### 4.2 集中式 overlap event loop

`scheduler.py:Scheduler.event_loop_overlap()`（无投机专属 loop）：

```python
plan = self.get_next_batch_to_run(                      # ← 现在返回 NextBatchPlan
    running_batch=self.running_batch, last_batch=self.last_batch
)
self.running_batch = plan.running_batch                 # ← running_batch 由 plan 回写
batch = plan.batch_to_run
disable_overlap_for_batch = self.is_disable_overlap_for_batch(
    batch, last_batch=self.last_batch                   # ← last_batch 显式传参
)

if disable_overlap_for_batch:
    pop_and_process()                                   # 处理上一批（前移）

if batch:
    batch_result = self.run_batch(batch)                # 发射当前批
    self._apply_war_barrier()                           # ← WAR 屏障：把结果处理挡在本次
    self.result_queue.append((batch.copy(), batch_result))  #   forward 的共享读之后

if self.last_batch and not disable_overlap_for_batch:
    pop_and_process()                                   # 处理 N-1 批结果（推迟一拍）

if self.is_generation:
    self.launch_batch_sample_if_needed(batch_result, batch)  # ← 采样后置：依赖上一批
                                                             #   已处理完（如 grammar）
```

要点三条：

1. **`get_next_batch_to_run()` 已改成"纯函数式"签名**：入参 `running_batch=` / `last_batch=` 显式传，返回一个 `NextBatchPlan`（含 `batch_to_run` 与更新后的 `running_batch`），不再靠就地改 `self.running_batch`。
2. **`_apply_war_barrier()`**（Write-After-Read 屏障）在 `run_batch()` 之后立刻调用，保证结果处理不会抢在本次 forward 对共享 buffer 的读之前动手。
3. **`launch_batch_sample_if_needed()`** 被有意放在"处理上一批"之后——采样可能依赖上一批已提交的状态（最典型的就是 grammar bitmask）。

这就是"发射当前批 + 处理上一批结果"的一拍延迟机制（详见 `overlap_schedule_architecture.md`）。

### 4.3 何时强制退化为同步

`scheduler.py:Scheduler.is_disable_overlap_for_batch()` 只在两种情况关闭 overlap：

```python
disable_overlap_for_batch = (
    envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
    and batch_is_extend and last_batch_is_extend        # ① 连续 prefill
)

need_grammar_sync = (                                   # ② grammar 需要同步
    batch
    and not batch.spec_algorithm.is_none()
    and batch.grammar_needs_sync()                      # ← 关键：算法是否支持 grammar overlap
    and batch.forward_mode.is_decode()
    and len(self.result_queue) > 0                      # ← 队列为空就没必要同步
)
return disable_overlap_for_batch or need_grammar_sync
```

第二条已经不是旧文档里那句"We do not support overlap + spec + grammar yet"了——那条注释已被删除。现在的判定核心是 `schedule_batch.py:ScheduleBatch.grammar_needs_sync()`：

```python
def grammar_needs_sync(self) -> bool:
    return self.has_grammar and not self.spec_algorithm.supports_grammar_overlap()
```

而 `speculative/spec_info.py:SpeculativeAlgorithm.supports_grammar_overlap()` 对 **EAGLE / STANDALONE / dflash 家族返回 `True`**。也就是说：

- **DSV4 + MTP（算法 EAGLE）走的是"支持 grammar overlap"这一侧**，`grammar_needs_sync()` 恒为 `False`，因此**即使带 grammar 也不会退化为同步**——FSM 的推进被搬进 verify 内部（见 §4.7 的 grammar barrier）。
- 只有 NGRAM 这类"draft 来自 host 语料查表、没有 GPU draft 阶段可藏 CPU 开销"的算法才返回 `False`，需要真正同步。
- 末尾那个 `len(self.result_queue) > 0` 是个纯优化：队列为空时根本没有待推进的 FSM，同步毫无意义。

**纯 DSV4 + MTP（无 grammar）自然保持完全 overlap**。关闭时 `pop_and_process()` 在发射当前批之前执行，坍缩掉延迟。

### 4.4 batch 选择：PREFILL 优先，DECODE 兜底

`scheduler.py:Scheduler.get_next_batch_to_run()`：先 `get_new_batch_prefill(running_batch)`（它自己也返回一个带 `batch_to_run` / `running_batch` 的 plan），有新 prefill 就走 EXTEND；否则 `update_running_batch()` 走 DECODE。

spec + dp-attn 下**绝不混合 prefill 与 decode**，这条门禁现在还多了一个开关：

```python
if (
    need_mlp_sync
    and not self.spec_algorithm.is_none()
    and not get_spec().speculative_skip_dp_mlp_sync   # ← 新增：可显式跳过这条门禁
):
    new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)
    need_mlp_sync = new_batch is None
```

`scheduler.py:Scheduler.update_running_batch()` 内部照常 `ScheduleBatch.check_decode_mem()` / `ScheduleBatch.retract_decode()`——集中式 DSV4 的 OOM 回退用的是**本地 retract 路径**，而非 PD 的跨实例 offload。

### 4.5 run_batch 里的投机分支

`scheduler.py:Scheduler.run_batch()` 对投机 + overlap 是热路径：

```python
if self.enable_overlap:
    self.future_map.resolve_seq_lens_cpu(batch)         # 投机专属：懒取 new_seq_lens
    with self.forward_stream_ctx:
        self.forward_stream.wait_stream(self.schedule_stream)
        resolve_forward_inputs(batch, self.future_map)  # 用真 token 替换负数占位
        with self._forward_isolation(batch, overlap=True):
            future_indices = batch.req_pool_indices

            fwd_kwargs = {}
            if not batch.spec_algorithm.is_none():
                fwd_kwargs["on_publish"] = partial(
                    self.future_map.publish, future_indices
                )
                if batch.spec_algorithm.supports_grammar_overlap():
                    fwd_kwargs["grammar_barrier"] = (   # ← grammar barrier 注入点
                        self._advance_pending_grammar
                    )

            batch_result = self.model_worker.forward_batch_generation(
                batch, **fwd_kwargs
            )
            if batch.spec_algorithm.is_none():
                self.future_map.publish(future_indices, batch.seq_lens + 1)

            if batch_result.extra_keep_alive_refs:      # ← §3.8 的保活引用停车场
                self.batch_record_buf[self.batch_record_ct].extend(
                    batch_result.extra_keep_alive_refs
                )
            batch_result.copy_done = self.device_module.Event()
            if batch_result.delay_sample_func is None:
                self._relay_forward_payload(batch, future_indices, batch_result)
                ...                                     # copy_stream 上做结果 D2H

    batch.input_ids = None                              # 下一轮 input_ids 走 future_map 中继
    if not batch.spec_algorithm.is_none():
        batch.spec_info = batch_result.next_draft_input # draft 状态传给下一迭代
        batch.spec_info.future_dsa_topk_indices_available = (
            batch.spec_info.dsa_topk_indices is not None
        )
        batch.spec_info.future_indices = future_indices
```

关键点：

- **`on_publish` 传进 worker**，让 future-map 的发布发生在 **verify 与 draft_extend 之间**（`eagle_worker_v2.py:EAGLEWorkerV2.forward_batch_generation()` 内）——这样下一迭代的调度能与本轮 draft_extend 重叠。非投机则在 worker 返回后由调度器自己 `publish`。
- **`_forward_isolation(batch, overlap=...)`** 包住整个 forward：worker 在 forward 期间对 `ScheduleBatch` 的临时改写会在退出时被还原，所以 `batch.spec_info` / `batch.seq_lens` 这些"必须带到下一轮"的字段要在 isolation 之后显式重新赋值。
- **`extra_keep_alive_refs` 停车场**：worker 要求保活的引用被塞进 `self.batch_record_buf[self.batch_record_ct]`，与 SB 属性快照同一个 ring slot，覆盖两个迭代窗口（见 §3.8）。
- **`_relay_forward_payload()`** 把本轮输出（bonus token / spec extras）写进 future map，供下一轮 `resolve_forward_inputs()` 取用（见 §5）。
- **非 overlap + spec** 走另一条分支：同步调 `forward_batch_generation(batch, pp_proxy_tensors=...)`，然后手工把 `batch.spec_info` / `batch.seq_lens` / `batch.seq_lens_cpu` / `batch.seq_lens_sum` 从 `batch_result` 抄回来——因为 isolation 的还原把 worker 的改动都撤了。

### 4.6 结果处理与 bonus token 记账

`scheduler.py:Scheduler.process_batch_result()` 按 forward mode 分派；集中式 decode 走 `batch_result_processor.py:BatchResultProcessor.process_batch_result_decode()`，集中式 extend 走 `BatchResultProcessor.process_batch_result_prefill()`，**不是** PD 的 `Scheduler.process_batch_result_disagg_prefill()`。

投机 token 解析在 [`batch_result_processor.py:BatchResultProcessor._resolve_spec_v2_tokens()`](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py)：

```python
result.num_correct_drafts = sum(accept_lens) - len(batch.reqs)   # 去掉每请求 1 个 bonus
result.num_correct_drafts_per_req_cpu = [x - 1 for x in accept_lens]

result.num_block_accept_tokens = sum(block_accept_lens) if block_accept_lens else 0
result.num_cap_tokens = sum(cap_lens) if cap_lens else 0          # ← 两个新增聚合指标

self.model_worker.on_verify_complete_cpu(                         # 喂自适应控制器（§3.5）
    result.num_correct_drafts_per_req_cpu, batch_size=len(batch.reqs)
)
self.advance_grammar_fsm(result, batch)                           # 幂等：barrier 已推进过就 no-op

stride = result.speculative_num_draft_tokens                      # ← 用 result 上记录的值，
assert stride is not None                                         #   不用 worker 当前值（自适应）
for i, req in enumerate(batch.reqs):
    accept_tokens = next_token_ids[i * stride : i * stride + accept_lens[i]]
    ...
    req.kv.kv_committed_len += len(accept_tokens)                 # ← 注意是 req.kv 子对象
```

四个易错点：

- **两指标差一个 bonus**：`accept_lens[i]` **含** bonus token；`num_correct_drafts` 每请求剥掉一个 bonus。KV 提交长度用整条接受链（drafts + bonus）。这是投机解码最易记错的账（参见 `.claude/skills/speculative-naming/SKILL.md`）。
- **`kv_committed_len` 已搬到 `req.kv` 子对象下**，写法是 `req.kv.kv_committed_len`，不再是 `req.kv_committed_len`。
- **`stride` 取 `result.speculative_num_draft_tokens`**，不是 worker 当前的 `speculative_num_draft_tokens`——自适应投机下（§3.5）延迟一拍处理结果时，worker 的档位可能已经切换了。
- **新增两个聚合指标** `num_block_accept_tokens` / `num_cap_tokens`（来自 `result.block_accept_lens` / `result.cap_lens`），并按请求累加到 `req.spec_num_block_accept_tokens` / `req.spec_num_cap_tokens`。DSV4 + MTP 的链式路径这两个通常为 `None`。

### 4.7 grammar barrier：spec + grammar 也能 overlap

这是让 §4.3 那条"EAGLE 不需要为 grammar 关 overlap"成立的机制，链条有三环：

| 环节 | 符号 | 做什么 |
|---|---|---|
| ① 调度器提供屏障 | `scheduler.py:Scheduler._advance_pending_grammar()` | 遍历 `self.result_queue` 里尚未处理的结果，对每个调 `BatchResultProcessor.advance_grammar_fsm()`，把上一批已提交的 token 推进 FSM。幂等、队列空则 no-op |
| ② 沿参数链传下去 | `run_batch()` → `EAGLEWorkerV2.forward_batch_generation(grammar_barrier=...)` → `EAGLEWorkerV2.verify(grammar_barrier=...)` → `eagle_worker_common.py:run_eagle_verify(grammar_barrier=...)` | 一路 kw 传递，不落任何全局状态 |
| ③ 在 verify 内部触发 | `run_eagle_verify()` 里 `build_grammar_vocab_mask(..., barrier=grammar_barrier)` | 构造 bitmask **之前**调一次 barrier，于是这段 CPU 上的 FSM 推进与 GPU 上的 target verify 前向**重叠** |

代码注释把设计意图写得很明白（`Scheduler._advance_pending_grammar()` 的 docstring）：

```python
"""Grammar barrier (spec-v2 overlap): advance the FSM over any not-yet
-processed decode result still in the queue, so a following verify()'s
bitmask sees the previous batch's committed tokens. Invoked mid-worker
(before generate_token_bitmask) so the CPU advance overlaps the target
verify forward. Idempotent; no-op when the queue is empty or has no grammar.
"""
```

因为 barrier 已经在 verify 里推进过 FSM，§4.6 里 `_resolve_spec_v2_tokens()` 再调的那次 `advance_grammar_fsm()` 就是幂等 no-op，并直接复用 `result.grammar_retained_tokens[i]`（被 grammar 截断后的合法接受串）而不是重新推进。

**对 DSV4 + MTP 的实际意义**：开 grammar（JSON schema / 正则约束输出）时，投机解码**不会**掉回同步模式，overlap 收益完整保留。

---

## 第 5 层：重叠调度与 FutureMap 如何跨阶段协作

spec v2 = overlap + spec。集中式下双 CUDA stream（`schedule_stream` 跑 event loop、`forward_stream` 跑模型）与 `FutureMap` 协作，让 CPU 调度不阻塞在 GPU 采样上。

### 5.1 负数占位符打破数据依赖

[`overlap_utils.py:resolve_forward_inputs()`](../../../python/sglang/srt/managers/overlap_utils.py) 在 forward 入口把占位的 `input_ids` 换成从 `FutureMap.output_tokens_buf` gather 的真 token。因此调度器可以在 N 迭代的采样尚未完成时，就先编排 N+1 迭代的输入，打破一拍间的数据依赖。

### 5.2 中继载荷 `RelayPayload`：到底传了些什么

每一轮 forward 结束后，`Scheduler._relay_forward_payload()` 把要跨迭代传递的东西打包成一个 `overlap_utils.py:RelayPayload`，再由 `FutureMap.stash()` 按 `req_pool_indices` 散写进各自的 buffer。它的字段是理解"投机到底比普通 decode 多传了什么"的最直接清单：

| 字段 | 含义 | 非投机 | DSV4 + MTP |
|---|---|---|---|
| `bonus_tokens` | 本轮 target 免费给出的那个 token（下一轮 draft 的起点） | ✅ 唯一填的字段 | ✅ |
| `topk_p` | draft 的 top-k 概率 | — | ✅（topk==1 ⇒ 宽度 1） |
| `topk_index` | draft 的 top-k token id | — | ✅（topk==1 ⇒ 宽度 1） |
| `hidden_states` | target 最后一层 hidden state，喂给 NextN 层的融合输入 | — | ✅ |
| `draft_probs` | 拒绝采样用的 draft 分布 `q` | — | 仅开 `--speculative-use-rejection-sampling` 时（§3.6） |
| `dsa_topk_indices` | DSA / 稀疏注意力的 topk 索引复用 | — | 视后端而定，`future_dsa_topk_indices_available` 决定 |
| `accept_tokens` | 整条接受串（NGRAM 专用，因为它把 draft-extend 延后了） | — | — |
| `accept_lens` | 每请求接受长度（NGRAM 专用） | — | — |

两个 `@classmethod` 构造器分别对应两条路径：

- `RelayPayload.from_draft_input(draft_input: EagleDraftInput)` —— EAGLE 家族（含 DSV4 + MTP）走这条，把 `bonus_tokens / topk_p / topk_index / hidden_states / draft_probs / dsa_topk_indices` 一并搬过来；
- `RelayPayload.from_ngram(draft_input: NgramVerifyInput)` —— NGRAM 走这条，只填 `accept_tokens / accept_lens`，`bonus_tokens=None`。

类的 docstring 强调了一个容易误解的点：**"哪些 spec extras 真的被中继，由 `FutureMap.spec_algo` 决定，而不是由 payload 的形状决定"**。也就是说 payload 里字段非空 ≠ 一定会被 stash——`FutureMap` 会按算法能力（`need_topk` / `need_hidden_states` 等）挑着写。

> 注：`RelayPayload` 目前还是 `@dataclass`。仓库规范 `.claude/rules/no-dataclasses.md` 要求**新**数据容器用 `msgspec.Struct`，已有的 `@dataclass` 属于历史遗留、按需迁移，所以这里不算违规。

### 5.3 投机专属的 spec extras 中继

`overlap_utils.py:FutureMap` 为投机额外提供：

- `FutureMap._resolve_spec_extras()`：按 `draft_input.future_indices` 从各 buffer gather 上一轮 verify 的产物，回填到下一轮的 `EagleDraftInput` 上。`need_topk` 为真时走融合的 `overlap_utils.py:gather_spec_extras()`（一次 kernel 同时取 `topk_p` / `topk_index` / `bonus_tokens` / `hidden_states`），否则只取 `bonus_tokens`。`draft_probs` / `dsa_topk_indices` 各有独立的 `is not None` 门禁。**这是跨 stream 边界喂给下一轮 draft 阶段的唯一通道。**
- `FutureMap.resolve_seq_lens_cpu()`：投机专属。因为 `accept_lens` 在调度时未知，`new_seq_lens` 要等 verify 完成，故先在 `publish_ready` event 上等待（HIP 上用 `synchronize()` 规避 MI355 的 TPOT 回退），再从 `new_seq_lens_buf` gather，需要 CPU 镜像时在私有 D2H stream 上拷。
- `FutureMap.publish()`：写 `new_seq_lens_buf`，并且**只有投机**才 record `publish_ready` event（`if self.spec_algo.is_some()`）——这个 event 就是上一条等待的对象。
- `FutureMap.stash()`：把 `RelayPayload` 按 pool 索引散写进各 buffer；只需要传 bonus token 的场合有个轻量版 `FutureMap.stash_bonus_tokens()`。

### 5.4 发布时机 = verify 结束

`on_publish` 在 worker 内部于 verify 之后、draft_extend 之前触发（`eagle_worker_v2.py:EAGLEWorkerV2.forward_batch_generation()` 内），把 fence 定在 verify-end。于是：调度器下一迭代的准备工作与本轮 draft_extend 在 `forward_stream` 上并行。这是集中式投机能吃满 GPU 的关键。

```
时间轴（stream 视角）

schedule_stream:  [prep N] ── wait ──▶ ... [prep N+1] ──────────▶
                                              ▲ on_publish 后立即可开始
forward_stream:   [draft N][verify N]──publish──[draft_extend N] ...
                                          │
                                          └─ 下一迭代调度与 draft_extend 重叠
```

非 overlap + spec 的分支里没有 `on_publish`，worker 同步跑完三阶段再返回（见 §4.5 末尾）。

---

## 第 6 层：KV cache——draft 步如何分配显存

### 6.1 每步分配长度：三个 sizing 函数已搬家

三个函数**已从 `mem_cache/common.py` 移到新模块 [`mem_cache/allocation_sizing.py`](../../../python/sglang/srt/mem_cache/allocation_sizing.py)**，而且都改成了"从运行期 context bag 读配置"（`get_spec()` / `get_schedule()` / `get_parallel()`），不再吃 `server_args` 参数——因为自适应投机（§3.5）会在 publish 之后改动步数与 draft token 上界，这些函数每个 decode batch 都要重算。

| 函数 | 作用 | DSV4 + MTP 的值 |
|---|---|---|
| `allocation_sizing.py:get_alloc_page_size()` | `page_size × attn_dcp_size` | 256（未开 DCP） |
| `allocation_sizing.py:get_alloc_len_per_decode()` | 单个 decode step 一个请求可能分配的 KV 长度 | `max(3×1, 4) = 4` |
| `allocation_sizing.py:get_alloc_reserve_per_decode()` | 每 decode step 每请求的预留 | `2 × 4 = 8` |
| `allocation_sizing.py:get_req_to_token_extra_context_len()` | `req_to_token` 行相对模型 context 长度的额外余量 | **263**（不是 8！见下） |

`get_alloc_len_per_decode()` 对 DSV4（`topk == 1`）走第一条分支：

```python
spec_tokens = max_speculative_num_draft_tokens()     # ← 用上界，不用当前值（自适应）
page_size = get_alloc_page_size()
if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
    return max(spec_steps * spec_topk, spec_tokens)  # max(3*1, 4) = 4
else:
    ...   # spec v2 树（page>1 且 topk>1）才需要按 topk 分支各留整页
```

**`get_req_to_token_extra_context_len()` 现在多了一条 page>1 分支，这是最容易踩错的一处**：

```python
extra = 4 + (max_speculative_num_draft_tokens() or 0)          # ← 4 + 4 = 8（旧文档只写到这）
page_size = get_alloc_page_size()
if get_spec().speculative_algorithm is not None and page_size > 1:
    # kv_allocated_len is page-aligned (eagle_prepare_for_decode), so near
    # the context limit the aligned reserve can overshoot by page_size - 1;
    # without the headroom the row write silently lands in the neighbor row.
    extra = max(extra, get_alloc_reserve_per_decode() + page_size - 1)
return extra
```

DSV4 的 `page_size = 256`、投机开启，所以走进这条分支：

```
extra = max(8, get_alloc_reserve_per_decode() + page_size - 1)
      = max(8, 8 + 256 - 1)
      = max(8, 263)
      = 263          ← 真实值
```

为什么需要这么大的余量？因为 `kv_allocated_len` 是**按整页对齐**的（见 §6.2）。逼近 context 上限时，对齐后的预留最多可以比"逻辑长度"多出 `page_size - 1 = 255` 个位置。如果 `req_to_token` 那一行没有这个余量，写入就会**静默越界写到邻居请求的行里**——不报错，只是数据被悄悄污染。这是个非常隐蔽的 bug 类型，所以代码注释专门点出了 "silently lands in the neighbor row"。

这些长度是后端无关的（用于 `req_to_token` 映射与外层 allocator）。DSV4 的特殊性在于**一个 token slot 会扇出到六个子池**，扇出由 `DeepSeekV4TokenToKVPoolAllocator` 处理，与这些长度函数无关。

### 6.2 请求侧的 KV 长度账本：`req.kv` 与整页分配

理解 263 的前提是理解**长度记账已经从 `Req` 上的散字段收拢到一个子对象 `req.kv`（`schedule_batch.py:ReqKvInfo`）里**。所以现在写法一律是 `req.kv.kv_committed_len`，不是 `req.kv_committed_len`。

`ReqKvInfo` 里与本文相关的三个长度，注释一句话讲清了它们的关系——**"The request's own KV is `[cache_protected_len, kv_allocated_len)`"**：

| 字段 | 含义 |
|---|---|
| `req.kv.cache_protected_len` | 前缀树（radix cache）拥有 `[0, here)` 这一段，请求自己不能碰、也不负责释放 |
| `req.kv.kv_committed_len` | KV **内容**已经写实到这里（`<= kv_allocated_len`）。投机每轮按 `accept_lens` 往前推（§4.6） |
| `req.kv.kv_allocated_len` | KV **槽位**已经分配到这里。page_size>1 时**按整页对齐** |

`kv_committed_len` 与 `kv_allocated_len` 的差额就是"已占坑但还没写内容"的部分——投机的乐观预分配全落在这段里。

具体的整页分配逻辑在 `allocation_sizing.py:page_aligned_decode_alloc_lens()`：

```python
for i, r in enumerate(reqs):
    cur = r.kv.kv_allocated_len
    nxt = max(
        cur,
        (r.kv.kv_committed_len + reserve + page_size - 1) // page_size * page_size,
    )                                       # ← 向上取整到整页（DSV4：256 的倍数）
    num_needed_tokens += nxt - cur
```

docstring 解释了为什么必须"整页"：**"nxt rounds committed up to page so allocated == recorded (unaligned tails leak at ps>1)"**——如果只按逻辑长度分配、不补齐到页边界，`page_size > 1` 时那些没对齐的尾巴会**泄漏**（分配了但没记账，最终无人释放）。

于是 DSV4 的分配粒度就是 **256 个 token 一整页**：一个请求即使只需要再写 4 个 token（3 draft + 1 bonus），只要跨过了页边界就要整整多要一页。这既是 §6.1 里 `extra = 263` 的根源（`8 + 255`），也是 DSV4 显存账看起来"很粗"的原因。

### 6.3 draft worker 的池：只加一层 SWA

旧文档引用的 `model_executor/model_runner_kv_cache_mixin.py` **已被删除**（mixin 被仓库规范 `.claude/rules/general-code-style.md` 劝退），逻辑搬到了 [`mem_cache/kv_cache_configurator.py`](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py) 的 `KVCacheConfigurator` 建池路径里（快照 :1263-1272）。draft worker 建池时**所有层用全 0 压缩比**：

```python
if self.is_draft_worker:
    from sglang.srt.models.deepseek_v4_nextn import COMPRESS_RATIO_NEXTN_LAYER

    compression_ratios = [
        COMPRESS_RATIO_NEXTN_LAYER                      # 全 0
    ] * self.layer_info.num_effective_layers            # ← 现在从 layer_info 取层数
else:
    compression_ratios = self.model_config.compress_ratios
```

ratio 全 0 ⇒ draft 只用 SWA 路径，不建 c4/c4_indexer/c128 及其状态 ring 池。同一段代码里还有一条硬断言：非 NPU 上 `swa_page_size` 必须等于 256（`assert swa_page_size == 256, "In paged swa mode, page_size must be 256."`）。

### 6.4 draft 与 target 共享池

`eagle_worker_v2.py:EagleDraftWorker.alloc_memory_pool()` 从调度器接过 `req_to_token_pool` 与 `token_to_kv_pool_allocator`，原样转给它内部那个 `is_draft_worker=True` 的 `TpModelWorker`——所以 draft 与 target 用的是**同一个** pool 对象。draft 的 KV 写入落到共享的六子池 allocator（NextN 层只写 `swa_kv_pool`）。开了 `--speculative-use-rejection-sampling` 时这里还会额外校验 draft 与 target 同词表（§3.6）。

容量层面由 [`model_executor/pool_configurator.py`](../../../python/sglang/srt/model_executor/pool_configurator.py) 的 `DSV4PoolConfigurator` 负责。

> ⚠️ **一处必须纠正的行号**：`DSV4PoolConfigurator` 在 `pool_configurator.py` 的 **`:763`**，`(T+D)/T` 膨胀那段在 **`:825-832`**。旧文档写的 `:507` / `:553-561`，以及姊妹文档 `dsv4_mtp_layer_forward_dataflow.md` 写的 `:574` / `:633-640`，**两个都是错的**。

投机时的两项容量调整：

```python
# pool_configurator.py: DSV4PoolConfigurator.__init__ 内（快照 :825-832）
if self.is_speculative:
    # Reserve memory for the speculative draft worker by inflating
    # per-token bytes by (target+draft)/target.
    draft_layers = 1                                    # ← NextN 恒 1 层
    target_layers = self.num_layers_total
    self.bytes_per_full_token *= (target_layers + draft_layers) / target_layers
```

1. **每 token 字节数按 `(T + D) / T` 抬高**（D = draft 层数 = 1），等价于给 draft worker 预留显存。总 token 数 = `avail / (bpft × (T+D)/T)`。
2. **c4 / c128 压缩态 ring 尺寸抬高**，由 `mem_cache/deepseek_v4_memory_pool.py:get_compress_state_ring_size()` 决定：

   | compress_ratio | 非投机 ring_size | 投机 ring_size |
   |---|---|---|
   | 4 | 8 | **16** |
   | 128 | 128 | **256** |

   抬高的原因就是 §3.4 说的那件事：一次 verify 会把**整条乐观尾巴**写进 ring，ring 必须容得下。

### 6.5 反向门禁：ring 容量倒过来限制 draft token 数

`DSV4PoolConfigurator.__init__()` 里在算完 ring 尺寸后立刻做一次反向校验：

```python
if self.is_speculative:
    # Ring is sized once here, so it must serve the largest adaptive tier.
    self._assert_ring_serves_draft_tokens(max_speculative_num_draft_tokens() or 0)
```

`DSV4PoolConfigurator._assert_ring_serves_draft_tokens()` 对 c4 / c128 各算一次上限，用的是 `deepseek_v4_memory_pool.py:get_compress_state_write_pad()`：

```python
def get_compress_state_write_pad(compress_ratio: int, ring_size: int) -> int:
    """Largest draft-token count this ring can serve; mirrors `mtp_pad` in `c_plan.cuh`."""
    window_size = compress_ratio * (2 if compress_ratio == 4 else 1)
    return ring_size - window_size + 2 if ring_size > window_size else 0
```

代入 DSV4 投机档位：

```
c4 :  window = 4 × 2 = 8    ring = 16   ⇒ 上限 = 16 - 8 + 2   = 10
c128: window = 128         ring = 256  ⇒ 上限 = 256 - 128 + 2 = 130
```

DSV4 的 `num_draft_tokens = 4`，远在两个上限之内，所以断言恒过。断言失败时的报错信息还给了两条出路——"Lower the draft count, or grow the ring in `get_compress_state_ring_size()`"。

三个值得记住的细节：

- **`_assert_ring_serves_draft_tokens()` 收的参数是 `max_speculative_num_draft_tokens()`（上界）而非当前值**，因为 ring 只在启动时 size 一次，必须服务得了自适应的最高档（§3.5）。这正是 §1.2 那条"sizing 必须用上界"的具体落地。
- `num_layers == 0` 的压缩档直接跳过——没有 c4 层就不用校验 c4 ring。
- 开了 `SGLANG_OPT_USE_ONLINE_COMPRESS` 的 online c128 走另一套（每 draft 一份状态而非 ring，`ring_size` 塌成 1），因此在这里被显式跳过；而且 `get_compress_state_ring_size()` 里对"online c128 + 投机"是直接抛 `AssertionError("online c128 does not support MTP")` 的，除非另开 `SGLANG_EXPERIMENTAL_ONLINE_C128_MTP`。**也就是说：DSV4 + MTP 与 online c128 默认互斥。**

---

## 第 7 层：CUDA graph——哪些阶段被捕获

`eagle_worker_v2.py:EagleDraftWorker._capture_cuda_graphs()` 分别处理 draft-decode 与 draft-extend 两个 graph runner。

| 阶段 | CUDA 上是否捕获 graph | 依据 |
|---|---|---|
| target decode（TARGET_VERIFY） | 是（target worker 的 decode graph） | target runner |
| draft decode（多步） | **仅当 `num_steps > 1`** | `EagleDraftWorker._capture_cuda_graphs()` 里 `if self.speculative_num_steps > 1:`；`draft_utils.py:DraftBackendFactory.create_decode_backend()` 在 `num_steps <= 1` 时早退返回 `None` |
| draft extend | **是**（CUDA / MUSA / NPU / XPU 均捕获；`SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH` 可关） | `EagleDraftWorker._capture_cuda_graphs()` 的后端白名单 |
| target prefill | eager（DSV4 因 breakable 被禁而落到 eager） | `arg_groups/cuda_graph_hook.py:disable_breakable_cudagraph_if_incompatible()` |

### 7.1 draft-extend 现在**会**被捕获（旧文档结论已反转）

⚠️ **这是本文档上一版最严重的错误**：旧版写"`DeepseekV4AttnBackend` 不在 draft-extend graph 白名单里，所以 DSV4 的 draft-extend 在 CUDA 上跑 eager"。**这个结论已经反了**——`DeepseekV4AttnBackend` 已经被显式加进白名单：

```python
# speculative/eagle_worker_v2.py: EagleDraftWorker._capture_cuda_graphs()
graph_supported_backend_types = [
    TritonAttnBackend, TRTLLMMLABackend, TRTLLMHAAttnBackend,
    TokenspeedMLABackend, FlashInferAttnBackend,
]
if _is_cuda or _is_musa:
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    graph_supported_backend_types.append(DeepseekSparseAttnBackend)
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    graph_supported_backend_types.append(DeepseekV4AttnBackend)   # ← DSV4 已在白名单
if _is_cuda:
    from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend
    graph_supported_backend_types.append(FlashMLABackend)

graph_supported_backend = isinstance(
    self.draft_extend_attn_backend, tuple(graph_supported_backend_types)
)
supports_cuda_draft_extend_graph = (_is_cuda or _is_musa) and graph_supported_backend
```

最终的捕获门禁是：

```python
if (
    self.draft_extend_attn_backend
    and not envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH.get()   # ← 唯一的关闭开关
    and (
        _is_npu
        or _is_xpu
        or supports_cuda_draft_extend_graph                     # ← CUDA / MUSA 走这条
        or supports_hip_draft_extend_graph                      # ← HIP 有自己的判定
    )
):
```

所以对 DSV4 + MTP：

- **CUDA / MUSA**：`draft_extend_attn_backend` 是 `DeepseekV4AttnBackend`（`draft_utils.py:DraftBackendFactory._create_dsv4_prefill_backend()` 造的），命中白名单 ⇒ **draft-extend 被捕获成 CUDA graph**。
- **NPU / XPU**：靠 `_is_npu` / `_is_xpu` 直接短路，也捕获。
- **HIP（ROCm）**：走独立谓词 `supports_hip_draft_extend_graph`，只认 `AiterMultiStepDraftBackend` / `DeepseekV4HipRadixBackend` / `DeepseekSparseAttnBackend` 这三种。
- **想关掉**：设 `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1`（调试 draft-extend 数值问题时很有用），此时才退回 eager。

`EagleDraftWorker._draft_extend_for_decode()` 里那个 `can_cuda_graph` 守卫依然存在，但它现在是"batch size 是否落在捕获过的桶里"的运行期判断，**不再是"这个后端根本没 graph"**。

### 7.2 target prefill 为什么还是 eager（机制已换）

旧文档说"EAGLE target 的 prefill 本就路由到 eager（`model_runner.py:2680-2691`、`prefill_cuda_graph_runner.py:152-154`）"。这两处引用**都已失效**：`prefill_cuda_graph_runner.py` 这个文件**已被删除**，`model_runner.py` 里那段判定也搬走了。现在是两套彼此独立的机制：

**机制一：EAGLE + tc_piecewise 的窄条件跳过**（搬到了 `model_runner_components/cuda_graph_setup.py:capture_prefill_graph()`）：

```python
if (
    model_runner.spec_algorithm.is_eagle()
    and not model_runner.is_draft_worker
    and get_server_return_hidden_states_mode() < CaptureHiddenMode.FULL   # ← 新增的窄化条件
    and check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)     # ← 只针对 tc_piecewise
):
    logger.info(
        "Disable prefill CUDA graph for EAGLE target on tc_piecewise "
        "to avoid FP4/MoE decode-replay corruption (#28386)."
    )
    return result(eager_runner)
```

注意它比旧文档描述的窄得多：**只在 prefill 后端是 `tc_piecewise` 且服务端的 capture 上限低于 `FULL` 时才跳过**。注释解释了原因——EAGLE target 的 prefill 需要 `CaptureHiddenMode.FULL`，若固定的 capture 上限低于 FULL，那捕出来的 NULL/LAST graph 是死的，还可能扰动 FP4/TRTLLM-MoE 状态、污染 decode replay（#28386、#28870）。BCG（breakable）与 FullCG 会为 EAGLE target 捕 FULL，所以**不需要**这条跳过。

**机制二（这才是 DSV4 的真实原因）：DSV4 专属的 breakable 禁用规则**。CUDA 上 prefill 的默认后端是 `Backend.BREAKABLE`，于是 `arg_groups/cuda_graph_hook.py:apply_cuda_graph_compatibility()` 会走进 `disable_breakable_cudagraph_if_incompatible()`，而它的规则表第一条就是 DSV4：

```python
# arg_groups/cuda_graph_hook.py: disable_breakable_cudagraph_if_incompatible()
rules = [
    # DSV4 is BCG-compatible but introduces heavy memory pressure: the
    # c4 indexer scratch is pinned in the capture pool and OOMs. Disable.
    (
        "DeepSeek-V4 (heavy capture-pool memory pressure)",
        lambda: is_deepseek_v4(model_config_of(server_args).hf_config),
    ),
    ...
]
for name, predicate in rules:
    if predicate():
        logger.warning(...)
        declare_resolution(                     # ← 直接把 prefill 后端改成 DISABLED
            server_args, "_disable_breakable_cudagraph_if_incompatible",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
            ),
        )
        return
```

结论要点：

- DSV4 走 eager prefill **不是因为它是 EAGLE target**，而是因为它是 DSV4——**c4 indexer 的 scratch buffer 会被钉在 capture pool 里，直接把显存打爆**。注释明确说 DSV4 本身"is BCG-compatible"（兼容 breakable），纯粹是显存压力问题。
- 这条规则**与投机无关**：DSV4 不开 MTP 时 prefill 同样是 eager。
- prefill 后端一旦被解析成 `Backend.DISABLED`，`capture_prefill_graph()` 开头那个 `check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)` 分支就会命中，把 prefill 路由到 `EagerRunner`（它的 `can_run_graph()` 恒 `False`）。

顺带修正另一处：旧文档提到的 `dsr1` prefill graph 禁用（`disable_prefill_cuda_graph_for_deepseek_trtllm_mla()`）只对 `DeepseekV3ForCausalLM` + `trtllm_mla` 后端生效，**DSV4（arch `DeepseekV4ForCausalLM` + `dsv4` 后端）确实不受它影响**——这句原文没错，但它并不是 DSV4 走 eager 的原因，真正的原因是上面的机制二。

---

## 第 8 层：完整生命周期串讲

一个请求从进入到完成，在集中式 DSV4 + MTP 下的完整旅程：

```
请求生命周期（集中式 DSV4 + MTP）

┌──────────────────────────────────────────────────────────────────────────┐
│ 阶段 A：入队 + prefill（EXTEND 迭代，1 次）                                    │
│                                                                            │
│  get_new_batch_prefill 选中 → run_batch → model_worker(=EAGLEWorkerV2)      │
│    ① target prefill（DSV4 全 kernel，capture_hidden_mode=FULL 以 kwarg 传入） │
│       → 出首 token                                                          │
│    ② _draft_extend_for_prefill：NextN 层跑一遍，产出首个 EagleDraftInput      │
│       （携带 target hidden state + bonus token）                            │
└──────────────────────────────────────────────────────────────────────────┘
                              │  batch.spec_info = next_draft_input
                              ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 阶段 B：稳态解码（DECODE 迭代，循环 N 次直到 EOS/max_tokens）                  │
│                                                                            │
│  update_running_batch 兜底 → run_batch → model_worker.forward_batch_gen     │
│                                                                            │
│    ⓿ activate_step_by_batch：按 batch size 定本轮 num_steps（自适应，可为 0）  │
│                                                                            │
│    ① draft（draft_forward）                                                 │
│         · 循环 num_steps-1 次 NextN 前向（默认 3-1=2 次；每步独立 dsv4         │
│           backend，仅 SWA）                                                 │
│         · topk==1 走 draft_topk1_postprocess 融合 kernel → 链式 draft_tokens │
│           （默认长度 4 = num_steps + 1）                                     │
│         · build_eagle_verify_input → EagleVerifyInput                       │
│         · num_steps == 0 时整段跳过，走 _build_trivial_verify_input           │
│                                                                            │
│    ② verify（run_eagle_verify，target 验证）                                 │
│         · target DSV4 对整条乐观尾巴并行 forward（TARGET_VERIFY，             │
│           全 c4/c128/SWA）                                                  │
│         · 有 grammar 时先经 grammar_barrier 推进上一批 FSM，再建 bitmask       │
│         · eagle_sample → accept_lens（含 bonus）+ accept_index                │
│         · clear_unaccepted_c128_draft_states：作废未被接受的 c128 压缩态       │
│         · new_seq_lens = seq_lens + accept_lens                            │
│         · on_publish（fence @ verify-end，供下一迭代重叠）                    │
│         · 返回 extra_keep_alive_refs=[verify_forward_batch]（跨迭代保活）      │
│                                                                            │
│    ③ draft_extend（_draft_extend_for_decode）                              │
│         · 用接受路径再跑 NextN，热身下一轮 draft 的 hidden/KV                  │
│         · CUDA 上走 CUDA graph（§7.1）                                      │
│                                                                            │
│  process_batch_result_decode → _resolve_spec_v2_tokens                     │
│    · 提交接受 token（drafts + bonus）到 req.kv.kv_committed_len               │
│    · on_verify_complete_cpu 喂自适应控制器                                    │
│    · 更新 accept_length / accept_rate 指标                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

单次 DECODE 迭代的 GPU 前向次数（默认 `num_steps = 3`）：**draft 2 次（NextN，轻）+ verify 1 次（DSV4 target，重）+ draft_extend 1 次（NextN，轻）**。每次迭代最多前进 4 个 token（3 draft + 1 bonus），期望前进 `τ = α × 3 + 1`（α 为接受率）。

> ⚠️ 上面这组数字是"未开自适应投机"时的稳定值。开了自适应后 `num_steps` 会随 batch size 变化（§3.5）：draft 前向次数变成 `num_steps - 1`，单迭代最多前进 `num_steps + 1` 个 token；`num_steps == 0` 时 draft 与 draft_extend 都被跳过，退化成普通 decode（每轮 1 个 token）。

### 与 PD 分离场景的关键区别

| 维度 | 集中式（本文） | PD 分离 |
|---|---|---|
| 实例数 | 单实例，单 event loop | prefill / decode 双实例 |
| 三阶段位置 | 一次 `forward_batch_generation` 内全跑完 | prefill 侧只做 target prefill + draft prefill；decode 侧跑 draft/verify/draft_extend |
| forward mode | EXTEND / DECODE | 额外 PREBUILT（decode 侧跳过 prefill） |
| KV 传输 | 无（同实例共享池） | KVSender/KVReceiver、`send_kv_chunk`、六子池两通道 |
| 队列 | running_batch / waiting_queue / retracted_queue | 额外 PreallocQueue / TransferQueue / inflight_queue |
| OOM 回退 | 本地 `ScheduleBatch.retract_decode()` | 跨实例 `Req.offload_kv_cache()` / `Scheduler.resume_retracted_reqs()` |
| extend 结果处理 | `BatchResultProcessor.process_batch_result_prefill()` | `Scheduler.process_batch_result_disagg_prefill()` |

> 注：PD 分离下的投机解码有额外限制，例如 `--disaggregation-decode-enable-radix-cache` 与投机解码互斥（门禁在 `arg_groups/pd_disaggregation_hook.py`，直接 `raise ValueError`）。D 侧 radix/HiCache 的细节见 `docs/pjc_1/pd_decode_radix_cache_hicache_scheme.md`。

---

## 第 9 层：关键文件符号索引

> 下表一律给**符号名**。少数没有外层命名符号可指的条目附「快照 :行号」（基于 `main` @ `33ed29a0ee`）。路径若无特别说明均相对 `python/sglang/srt/`。

### 启动配置
| 项 | 符号 |
|---|---|
| DSV4 声明式默认值（attention_backend / page_size / swa_full_tokens_ratio / MoE） | `arg_groups/model_overrides/deepseek_v4.py:_deepseek_v4_overrides()` |
| DSV4 KV dtype 后处理 | `arg_groups/overrides.py:_deepseek_v4_kv_cache_dtype()` |
| DSV4 命令式部分（`max_running_requests` + 投机断言 + ROCm env） | `arg_groups/deepseek_v4_hook.py:apply_deepseek_v4_defaults()` |
| NEXTN→EAGLE 别名归一 | `arg_groups/speculative_hook.py:_resolve_speculative_algorithm_alias()` |
| EAGLE 家族处理（含 DSV4 自动 draft-path、topk==1 ⇒ `num_draft_tokens = num_steps + 1`） | `arg_groups/speculative_hook.py:_handle_eagle_family()` |
| 自动投机参数 (3, 1, 4) | `arg_groups/speculative_hook.py:_auto_choose_speculative_params()` |
| draft 架构改写（DSpark / NextN 二选一） | `configs/model_config.py:ModelConfig._config_draft_model()` |

### draft 模型（NextN）
| 项 | 符号 |
|---|---|
| `COMPRESS_RATIO_NEXTN_LAYER = 0` | `models/deepseek_v4_nextn.py` 模块级常量（快照 :50） |
| NextN 模型主体 | `models/deepseek_v4_nextn.py:DeepseekV4ModelNextN` |
| NextN forward（EAGLE 条件融合） | `models/deepseek_v4_nextn.py:DeepseekV4ModelNextN.forward()` |
| EntryClass | `models/deepseek_v4_nextn.py:DeepseekV4ForCausalLMNextN` |
| compress_ratio 解析 | `models/deepseek_v4.py:DeepseekV4ForCausalLM.__init__()` 内 |
| compressor / indexer 仅 ratio 4/128 建 | `models/deepseek_v4.py:MqaAttentionBase.__init__()` |
| `assert self.compress_ratio in (0, 4, 128)` | `models/deepseek_v4.py:MqaAttentionBase.__init__()` |
| NextN 权重加载 / 重映射 | `models/deepseek_v4.py:DeepseekV4ForCausalLM.load_weights()` |

### 三阶段执行
| 项 | 符号 |
|---|---|
| 入口（三阶段编排 + `num_steps == 0` 旁路） | `speculative/eagle_worker_v2.py:EAGLEWorkerV2.forward_batch_generation()` |
| draft 多步循环 | `speculative/eagle_worker_v2.py:EagleDraftWorker.draft()` / `.draft_forward()` |
| draft 前置准备 / 验证输入构造（已提到模块级） | `speculative/eagle_worker_common.py:prepare_for_draft()` / `:build_eagle_verify_input()` |
| topk==1 链缓冲重建 + 链式不变式断言 | `speculative/base_spec_worker.py:BaseSpecWorker._rebuild_topk1_chain_buffers()` |
| verify 主体（薄委托的落点） | `speculative/eagle_worker_common.py:run_eagle_verify()` |
| verify 薄委托 | `speculative/eagle_worker_v2.py:EAGLEWorkerV2.verify()` |
| topk>1 才跑的树路径压缩 | `speculative/eagle_worker_common.py:_finalize_accept_tree_path()` |
| DSV4 专属 c128 回滚 | `mem_cache/deepseek_v4_memory_pool.py:DeepSeekV4TokenToKVPool.clear_unaccepted_c128_draft_states()` |
| draft_extend（prefill / decode 两条） | `speculative/eagle_worker_v2.py:EagleDraftWorker._draft_extend_for_prefill()` / `._draft_extend_for_decode()` |
| 自适应投机 | `speculative/eagle_worker_v2.py:EAGLEWorkerV2.activate_step_by_batch()` / `.build_adaptive_runtime_state()` / `.apply_runtime_state()` / `.on_verify_complete_cpu()` |
| `num_steps == 0` 旁路 | `speculative/eagle_worker_v2.py:EAGLEWorkerV2._build_trivial_verify_input()` / `._stub_skipped_draft_extend()` |
| `num_tokens_per_req` 自动填充 | `speculative/eagle_info.py:EagleDraftInput.__post_init__()` |
| DSV4 draft-decode / draft-extend 后端工厂 | `speculative/draft_utils.py:DraftBackendFactory._create_dsv4_decode_backend()` / `._create_dsv4_prefill_backend()` |
| 多步 draft 后端 | `layers/attention/deepseek_v4_backend.py:DeepseekV4MultiStepBackend` |
| topk 断言 / `mtp_enabled` | `layers/attention/deepseek_v4_backend.py:DeepseekV4AttnBackend.__init__()` |
| draft-extend 元数据 `need_compress=False` | `layers/attention/deepseek_v4_backend.py:DeepseekV4AttnBackend.init_forward_metadata_draft_extend()` |
| `capture_hidden_mode` kw-only 参数 | `managers/tp_worker.py:TpModelWorker.forward_batch_generation()` |

### 调度器 + overlap
| 项 | 符号 |
|---|---|
| `model_worker = draft_worker` 替换 | `managers/scheduler.py:Scheduler.init_attention_backend_and_cuda_graph()` 内（快照 :1067-1070） |
| draft worker 创建 | `managers/scheduler.py:Scheduler.maybe_init_draft_worker()` |
| 集中式 overlap 主循环 | `managers/scheduler.py:Scheduler.event_loop_overlap()` |
| WAR 屏障 / 后置采样 | `managers/scheduler.py:Scheduler._apply_war_barrier()` / `.launch_batch_sample_if_needed()` |
| overlap 关闭判定 | `managers/scheduler.py:Scheduler.is_disable_overlap_for_batch()` |
| grammar 是否强制同步 | `managers/schedule_batch.py:ScheduleBatch.grammar_needs_sync()` |
| 算法是否支持 grammar overlap | `speculative/spec_info.py:SpeculativeAlgorithm.supports_grammar_overlap()` |
| grammar barrier | `managers/scheduler.py:Scheduler._advance_pending_grammar()` |
| batch 选择（返回 `NextBatchPlan`） | `managers/scheduler.py:Scheduler.get_next_batch_to_run()` / `.get_new_batch_prefill()` / `.update_running_batch()` |
| OOM 检查 / 回退 | `managers/schedule_batch.py:ScheduleBatch.check_decode_mem()` / `.retract_decode()` |
| forward 发射（overlap + spec 分支） | `managers/scheduler.py:Scheduler.run_batch()` |
| 结果分派 | `managers/scheduler.py:Scheduler.process_batch_result()` |
| bonus token 记账 | `managers/scheduler_components/batch_result_processor.py:BatchResultProcessor._resolve_spec_v2_tokens()` |
| 请求侧 KV 长度账本 | `managers/schedule_batch.py:ReqKvInfo` |

### overlap / FutureMap
| 项 | 符号 |
|---|---|
| future map 主体 | `managers/overlap_utils.py:FutureMap` |
| 负数占位符解析 | `managers/overlap_utils.py:resolve_forward_inputs()` |
| 中继载荷 | `managers/overlap_utils.py:RelayPayload`（`.from_draft_input()` / `.from_ngram()`） |
| spec extras gather | `managers/overlap_utils.py:FutureMap._resolve_spec_extras()` / `:gather_spec_extras()` |
| 懒取 `new_seq_lens` | `managers/overlap_utils.py:FutureMap.resolve_seq_lens_cpu()` |
| 发布 / 暗存 | `managers/overlap_utils.py:FutureMap.publish()` / `.stash()` / `.stash_bonus_tokens()` |

### KV cache
| 项 | 符号 |
|---|---|
| 分配长度 / 预留 / 行余量（**DSV4 余量 = 263**） | `mem_cache/allocation_sizing.py:get_alloc_len_per_decode()` / `:get_alloc_reserve_per_decode()` / `:get_req_to_token_extra_context_len()` |
| 整页分配长度计算 | `mem_cache/allocation_sizing.py:page_aligned_decode_alloc_lens()` |
| 分配页大小 | `mem_cache/allocation_sizing.py:get_alloc_page_size()` |
| draft worker 全 0 压缩比建池 | `mem_cache/kv_cache_configurator.py:KVCacheConfigurator` 内（快照 :1263-1272） |
| draft / target 共享池 | `speculative/eagle_worker_v2.py:EagleDraftWorker.alloc_memory_pool()` |
| DSV4 池容量规划 | `model_executor/pool_configurator.py:DSV4PoolConfigurator`（快照 :763） |
| `(T+D)/T` 膨胀 | `model_executor/pool_configurator.py:DSV4PoolConfigurator.__init__` 内（快照 :825-832） |
| ring 容量反向门禁 | `model_executor/pool_configurator.py:DSV4PoolConfigurator._assert_ring_serves_draft_tokens()` |
| ring 尺寸（投机 16/256） / 写入 pad | `mem_cache/deepseek_v4_memory_pool.py:get_compress_state_ring_size()` / `:get_compress_state_write_pad()` |

### CUDA graph
| 项 | 符号 |
|---|---|
| draft-decode / draft-extend graph 捕获（**含 DSV4 白名单**） | `speculative/eagle_worker_v2.py:EagleDraftWorker._capture_cuda_graphs()` |
| draft-extend graph 关闭开关 | `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH`（`environ.py`） |
| prefill graph 捕获 / EAGLE+tc_piecewise 跳过 | `model_executor/model_runner_components/cuda_graph_setup.py:capture_prefill_graph()` |
| prefill 后端兼容性解析 | `arg_groups/cuda_graph_hook.py:apply_cuda_graph_compatibility()` |
| **DSV4 禁用 breakable ⇒ prefill eager 的真正原因** | `arg_groups/cuda_graph_hook.py:disable_breakable_cudagraph_if_incompatible()` |
| dsr1 + trtllm_mla 专属禁用（与 DSV4 无关） | `arg_groups/cuda_graph_hook.py:disable_prefill_cuda_graph_for_deepseek_trtllm_mla()` |

---

## 附：一句话总结每层

1. **配置**：`NEXTN→EAGLE`，topk 强制 1，参数 (3,1,4)，dsv4 后端 + page 256 + fp8；默认值已拆成「`model_overrides/` 声明式 override + hook 里命令式校验」两段，且现在也允许 `DSPARK`。
2. **draft 模型**：checkpoint 自带的 1 层 NextN，融合 target hidden + 下一 token 嵌入，`compress_ratio=0` 只用 SWA；checkpoint 若自带 DSpark draft 则走 `DeepseekV4ForCausalLMDSpark`，NextN 是兜底分支。
3. **三阶段**：draft（NextN 多步链）→ verify（target 并行验证整条乐观尾巴，并清掉未被接受的 c128 压缩态）→ draft_extend（用接受路径热身）；`num_steps` 运行期可被自适应改小甚至为 0。
4. **调度器**：`model_worker` 被换成 EAGLEWorkerV2，普通 overlap event loop 驱动，一个 forward 调用跑完三阶段；带 grammar 也不掉回同步——FSM 推进经 grammar barrier 藏在 verify 里。
5. **重叠**：负数占位 + `RelayPayload` 中继 spec extras，发布 fence 定在 verify-end 让调度与 draft_extend 并行；verify buffer 靠 `extra_keep_alive_refs` 保活两个迭代。
6. **KV**：draft 与 target 共享六子池 allocator，draft 只写 SWA 池，容量按 `(T+D)/T` 预留、c4/c128 ring 抬到 16/256；分配按 **256 整页**对齐，因此 `req_to_token` 行余量是 **263**（不是 8）。
7. **CUDA graph**：target decode/verify 捕获；draft-decode 仅 `num_steps > 1` 捕获；**draft-extend 在 CUDA/MUSA/NPU/XPU 上也捕获**（`DeepseekV4AttnBackend` 已进白名单，`SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH` 可关）；target prefill 是 eager，原因是 DSV4 专属的 breakable 禁用规则（capture pool 显存压力），与 EAGLE 无关。






