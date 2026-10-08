# SGLang MTP（Multi-Token Prediction）方案系统梳理

> 核心文件：`python/sglang/srt/arg_groups/speculative_hook.py`、`python/sglang/srt/models/deepseek_nextn.py`、`python/sglang/srt/speculative/eagle_worker_v2.py`、`python/sglang/srt/speculative/eagle_utils.py`、`python/sglang/srt/model_loader/loader.py`
> 目标读者：需要理解、部署、调试或扩展 SGLang MTP / NextN 投机解码的工程师。
> 本文由浅入深，覆盖：MTP 是什么、它与 EAGLE 家族的关系、如何启用、模型侧的 `num_nextn_predict_layers` 信号、draft 权重加载与 embedding/lm_head 共享、NextN 层的 `enorm/hnorm/eh_proj` 融合结构、`draft→verify→draft_extend` 主循环、tree building、KV 分配、CUDA Graph 捕获、注意力后端、调度器 overlap/non-overlap 集成、Multi-Layer/Frozen-KV/DeepSeek-V4 变体、PD 分离、命名规范与指标。
>
> ⚠️ 重要说明（读前必看）：
> 1. **MTP 不是独立算法，而是 EAGLE 的一种配置**。CLI 里的 `NEXTN` 只是 `EAGLE` 的别名（`_resolve_speculative_algorithm_alias`），最终都走 EAGLE 家族的 V2 worker。
> 2. **MTP 的本质区别**：draft"模型"是 target 权重内部的一层（`num_nextn_predict_layers` 切出来的 MTP 层），`speculative_draft_model_path` 被自动设为 `model_path`，无需单独下载 draft ckpt。
> 3. **V1/V2 之分已消失**：现在一律走 V2 worker，`SGLANG_ENABLE_SPEC_V2` 已删除。真正的运行时开关是 overlap vs non-overlap（`--disable-overlap-schedule`），两者都用同一个 V2 worker，non-overlap 时 scheduler 同步驱动它。
> 4. 典型 MTP 是 **`topk==1` 的链式（chain）**投机；`topk>1` 会切换到树形（tree）拓扑，与 EAGLE3 共用同一套 tree kernel。

---

## 目录

1. [总览：MTP 是什么，为什么它属于 EAGLE 家族](#1-总览)
2. [快速上手：三种启用姿势](#2-快速上手)
3. [算法别名解析：NEXTN → EAGLE](#3-算法别名解析)
4. [模型侧信号：num_nextn_predict_layers 与架构改写](#4-模型侧信号)
5. [Draft 权重加载：切片、重映射、共享 embedding/lm_head](#5-draft-权重加载)
6. [NextN 层结构：enorm/hnorm/eh_proj 融合](#6-nextn-层结构)
7. [Draft worker 初始化流程](#7-draft-worker-初始化)
8. [核心循环：draft → verify → draft_extend](#8-核心循环)
9. [Draft forward：多步草稿与 topk 快路径](#9-draft-forward)
10. [Tree building：chain vs tree](#10-tree-building)
11. [Verify：采样、bonus token、accept 逻辑](#11-verify)
12. [Draft extend：为下一轮准备隐藏态](#12-draft-extend)
13. [KV 缓存分配](#13-kv-缓存分配)
14. [CUDA Graph 捕获](#14-cuda-graph-捕获)
15. [注意力后端：MultiStepDraftBackend](#15-注意力后端)
16. [调度器集成：overlap 与 non-overlap](#16-调度器集成)
17. [变体：Multi-Layer / Frozen-KV / DeepSeek-V4 NextN](#17-变体)
18. [PD 分离下的 MTP](#18-pd-分离)
19. [命名规范与观测指标](#19-命名规范与指标)
20. [关键文件 / 行号索引](#20-关键文件行号索引)

---

## 1. 总览

### 1.1 MTP 与投机解码的定位

投机解码（Speculative Decoding）的核心思想：用一个**便宜的 draft 阶段**一次性猜出未来 k 个 token，再用**昂贵的 target 模型**一次 forward 并行校验这 k 个 token，接受连续正确的前缀。若平均能接受 α 个，则一次 target forward 顶多步 decode，从而在**不改变输出分布**（贪心/拒绝采样保证）的前提下降低延迟。

**MTP（Multi-Token Prediction，多 token 预测）** 是投机解码的一种具体实现范式，最早由 DeepSeek-V3 引入：在训练阶段就为模型附加一个（或多个）**MTP 层**，让模型在预测第 t 个 token 的同时，也基于第 t 个 token 的隐藏态去预测第 t+1 个 token。推理时，这个 MTP 层就天然可以充当 draft，逐 token 地把上一步的隐藏态"接力"下去，产出后续候选。

### 1.2 MTP 与 EAGLE 的关系（关键）

SGLang **没有为 MTP 写一套独立的调度/worker 代码**，而是把它实现为 **EAGLE 家族的一个配置分支**。原因是二者的运行时骨架几乎完全一致：

| 维度 | EAGLE / EAGLE3 | MTP（NextN） |
|---|---|---|
| draft 用什么算下一个 token | 一个**独立小模型**（单独 ckpt） | target 权重**内部的一层**（MTP 层） |
| draft 的输入 | 上一步 target 的 hidden state + 当前 token embedding | 完全相同 |
| draft 的循环结构 | `for i in range(num_steps)` 逐步 forward | 完全相同 |
| 校验方式 | target 一次 `TARGET_VERIFY` forward + 采样接受 | 完全相同 |
| 拓扑 | topk==1 链 / topk>1 树 | 完全相同 |
| worker 类 | `EAGLEWorkerV2` | `EAGLEWorkerV2`（同一个类） |

差异只在**"draft 从哪来、权重怎么加载、embedding/lm_head 是否共享"**这几点，而这些都被收敛进了：
- `speculative_hook.py`（把 draft_model_path 设成 model_path）
- `model_config.py` 的 `_config_draft_model`（把架构名改写成 `...NextN` 变体）
- `loader.py` 的 `_filter_mtp_weights`（从 target ckpt 里切出 MTP 层权重）
- `eagle_worker_v2.py` 的 `set_embed_and_head`（让 draft 复用 target 的 embedding 和 lm_head）

一句话定位：**MTP = EAGLE 的 `topk` 链式配置 + draft 层内嵌于 target 权重（`num_nextn_predict_layers` 切片、embedding/lm_head 共享）。理解了 EAGLE V2 的 draft→verify→draft_extend 循环，就理解了 MTP 的 99%。**

### 1.3 端到端数据流（一步 decode）

```
       ┌─────────────────────────────────────────────────────────────────┐
       │  上一轮 verify 结束后，next_draft_input 里带着：                     │
       │   bonus_tokens（target 采样出的下一个真 token）                     │
       │   hidden_states（那个 token 位置的 target 隐藏态）                   │
       └───────────────────────────────┬─────────────────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │  ① draft_forward  (draft 层，num_steps 次) │
                    │  每步：                                   │
                    │   embed(cur_token) ─┐                    │
                    │                     ├─ eh_proj(enorm||hnorm) → MTP 层 forward │
                    │   prev_hidden ──────┘                    │
                    │   → lm_head → topk_p / topk_index         │
                    │   下一步 cur_token = argmax(topk)         │
                    └───────────────────┬───────────────────┘
                                        │  draft_tokens（num_steps 个候选）
                    ┌───────────────────▼───────────────────┐
                    │  ② build_tree_kernel_efficient          │
                    │  把 bonus_token 作为根，draft_tokens 挂成 │
                    │  链(topk==1) 或 树(topk>1)，产出 mask/    │
                    │  retrieve_index/positions → EagleVerifyInput │
                    └───────────────────┬───────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │  ③ verify  (target 全量层, TARGET_VERIFY) │
                    │  一次 forward 算出所有候选位置的 logits，  │
                    │  eagle_sample 沿树/链接受最长正确前缀，    │
                    │  产出 accept_index / predict(含 bonus)    │
                    └───────────────────┬───────────────────┘
                                        │  accept_length 个 token 提交
                    ┌───────────────────▼───────────────────┐
                    │  ④ _draft_extend_for_decode             │
                    │  用被接受位置的 target 隐藏态，喂给 draft  │
                    │  层再跑一次，得到下一轮 draft 的起始       │
                    │  hidden_states / topk → next_draft_input  │
                    └───────────────────────────────────────┘
```

> 注意 ①④ 都用 **draft 层**（MTP 层），②③ 用 **target 全量层**。draft 层只有 1 层（或少数几层），所以 ①④ 很便宜；一轮里 target 只 forward 一次（③），却校验了 `num_steps+1` 个位置，这正是投机解码提速的来源。

---

## 2. 快速上手

MTP 有三种等价/近似的启用姿势，都以 `python -m sglang.launch_server` 为入口。

### 2.1 姿势一：显式 NEXTN（推荐，最贴合语义）

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --speculative-algorithm NEXTN \
    --tp 8 --trust-remote-code
# 无需 --speculative-draft-model-path：hook 自动设为 model-path
# 无需手写 num-steps/topk/num-draft-tokens：auto 默认 (3, 1, 4)
```

### 2.2 姿势二：显式 EAGLE（对内嵌 MTP 层的模型完全等价）

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --speculative-algorithm EAGLE \
    --tp 8 --trust-remote-code
```
对 DeepSeek/GLM/Bailing 这类**架构里带 MTP 层**的模型，`EAGLE` 与 `NEXTN` 走完全相同的代码路径（都在 `_handle_eagle_family` 的架构白名单里被自动补 draft_model_path）。

### 2.3 姿势三：手动指定全部参数

```bash
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp 8 --trust-remote-code
```

### 2.4 关键参数速查

| 参数 | 含义 | MTP 典型值 | 备注 |
|---|---|---|---|
| `--speculative-algorithm` | 算法名 | `NEXTN` | 别名，解析为 `EAGLE`；见 §3 |
| `--speculative-draft-model-path` | draft ckpt 路径 | **省略** | hook 自动设为 `--model-path`；见 §3.2 |
| `--speculative-num-steps` | draft 前向步数 | `3` | 决定 draft 层跑几次 |
| `--speculative-eagle-topk` | 每步保留候选数 | `1` | ==1 → 链；>1 → 树 |
| `--speculative-num-draft-tokens` | 送入 verify 的候选 token 数 | `4` | topk==1 时强制 = `num_steps+1`（`speculative_hook.py:382`） |
| `--speculative-use-rejection-sampling` | 用拒绝采样代替贪心接受 | 可选 | 仅 EAGLE/EAGLE3、topk==1；见 `speculative_hook.py:341` |
| `--disable-overlap-schedule` | 关重叠调度 | 默认不加 | 关掉后 scheduler **同步**驱动同一个 V2 worker |
| `--enable-multi-layer-eagle` | 多层 EAGLE（每步独立 ModelRunner） | 可选 | 见 §17.1 |

### 2.5 auto 默认参数表（`_auto_choose_speculative_params`, `speculative_hook.py:512`）

当用户未显式给 `--speculative-num-steps` 时，按架构选默认 `(num_steps, topk, num_draft_tokens)`：

| 模型架构 | 默认 `(steps, topk, draft_tokens)` | 拓扑 |
|---|---|---|
| `LlamaForCausalLM`、`Grok1*` | `(5, 4, 8)` | 树（topk=4） |
| `DeepseekV3/V32`、`GLM4Moe*`、`Bailing*`、`MiMoV2*`、`Pixtral`、`MistralLarge3` | `(3, 1, 4)` | 链（topk=1） |
| `STANDALONE` 算法 | `(3, 1, 4)` | 链 |
| 其它（兜底） | `(3, 1, 4)` | 链 |

> 观察：DeepSeek 系 MTP 默认走 **`topk=1` 链式**，因为其 MTP 层是为逐 token 接力训练的，链式即可获得很好的接受率；Llama EAGLE 走 topk=4 树，是典型独立 draft 模型的做法。

---

## 3. 算法别名解析

入口在 `handle_speculative_decoding`（`speculative_hook.py:54`），它先调用 `_resolve_speculative_algorithm_alias`（`:14`）把 CLI 字符串规整成内部算法名，再根据算法名分派到具体的 `_handle_*`。

### 3.1 NEXTN → EAGLE / FROZEN_KV_MTP

```
_resolve_speculative_algorithm_alias  (speculative_hook.py:14)
   ├─ 读 speculative_draft_model_path 的 HF config.architectures
   ├─ 若 draft 架构 ∈ {Gemma4AssistantForCausalLM, Gemma4UnifiedAssistantForCausalLM}
   │     → is_gemma4_draft = True
   ├─ EAGLE3 + gemma4_draft            → raise（EAGLE3 不支持该 draft）
   ├─ NEXTN / EAGLE + gemma4_draft     → 返回 "FROZEN_KV_MTP"（见 §17.2）
   ├─ NEXTN / EAGLE（其它）             → 返回 "EAGLE"        ← MTP 的主路径
   └─ 其它                              → 原样返回
```

要点：
- **`NEXTN` 本身在运行时并不存在**——它在 arg 处理阶段就被翻译成了 `EAGLE`。`SpeculativeAlgorithm` 枚举（`spec_info.py:28`）里也没有 `NEXTN` 成员，只有 `EAGLE / EAGLE3 / FROZEN_KV_MTP / DFLASH / STANDALONE / NGRAM / NONE`。
- 唯一会让 `NEXTN` 变成非 `EAGLE` 的情况，是 draft 架构是 Gemma4 助手模型，此时升级为 `FROZEN_KV_MTP`（一种"draft 只读 target KV、不做 draft-extend"的特殊 MTP，见 §17.2）。

### 3.2 EAGLE 家族的参数补全（`_handle_eagle_family`, `speculative_hook.py:260`）

解析成 `EAGLE` 后，`_handle_eagle_family` 做几件对 MTP 至关重要的事：

1. **`max_running_requests` 默认降到 48**（`:270`）——投机解码单请求占用更多显存/算力，降并发。
2. **`enable_mixed_chunk` 强制关闭**（`:282`）——mixed chunked prefill 与 EAGLE 不兼容。
3. **架构白名单自动补 draft_model_path**（`:289-314`）：若 `model_arch ∈ {DeepseekV3/V32/V4, Glm4Moe*, GlmMoeDsa, Bailing*, MistralLarge3, Pixtral, HYV3}` 且用户没给 draft 路径，则
   ```python
   server_args.speculative_draft_model_path = server_args.model_path   # ← MTP 的定义性动作
   server_args.speculative_draft_model_revision = server_args.revision
   ```
   这就是"MTP 不需要单独 draft ckpt"的落地点：draft 路径 = target 路径。
4. **auto 选参**（`:325`）：见 §2.5。
5. **topk==1 时对齐 draft_tokens**（`:382`）：`num_draft_tokens` 被强制为 `num_steps + 1`（链式下 draft 产出 num_steps 个候选 + 1 个 bonus 根）。
6. **拒绝采样约束**（`:341-380`）：`--speculative-use-rejection-sampling` 仅 EAGLE/EAGLE3、要求 topk==1、与确定性推理/多层 EAGLE/自定义 accept 阈值互斥。
7. **trtllm_mha 后端**只允许 topk==1（`:331`）。

---

## 4. 模型侧信号

### 4.1 `num_nextn_predict_layers`：MTP 支持的规范信号

一个模型是否"自带 MTP 层"，由 HF config 里的字段 `num_nextn_predict_layers` 决定。它在 `ModelConfig.__init__` 里被读入（`model_config.py:942`）：

```python
self.num_nextn_predict_layers = getattr(
    self.hf_text_config, "num_nextn_predict_layers", None
)
```

- 若为 `None`：模型没有 MTP 层，不能用 NEXTN（用 EAGLE 则必须提供独立 draft ckpt）。
- 若为正整数 `k`：checkpoint 里附带了 `k` 个 MTP 层，可切片其中一层当 draft。DeepSeek-V3 通常 `k=1`。

这个字段在后续多处被用来：
- 决定 draft worker 要加载几层（`model_runner.py:742`，见 §7）；
- 决定从 target ckpt 里切哪一层 MTP 权重（`draft_model_idx`，见 §5.1）。

### 4.2 架构名改写：`_config_draft_model`（`model_config.py:523`）

当 `ModelConfig.is_draft_model == True`（即这是 draft worker 的 config）时，`_config_draft_model` 会把 HF 架构字符串**改写成对应的 NextN/MTP 变体**，从而在模型注册表里命中"只含 MTP 层"的那个类，而不是完整的 target 类。改写表（节选）：

| target 架构 | 改写后 draft 架构 | 附加动作 |
|---|---|---|
| `DeepseekV3ForCausalLM` / `V32` / `GlmMoeDsaForCausalLM` | `DeepseekV3ForCausalLMNextN` | — |
| `DeepseekV4ForCausalLM` | `DeepseekV4ForCausalLMNextN` | **强制 `num_nextn_predict_layers = 1`**（`:538`） |
| `Glm4MoeForCausalLM` | `Glm4MoeForCausalLMNextN` | — |
| `Glm4MoeLiteForCausalLM` | `Glm4MoeLiteForCausalLMNextN` | — |
| `LongcatFlashForCausalLM` | `LongcatFlashForCausalLMNextN` | `num_hidden_layers = num_nextn_predict_layers`（`:558`） |
| `MiMoForCausalLM` | `MiMoMTP` | — |
| `MIMO_V2_MODEL_ARCHS` | `MiMoV2MTP` | — |
| `Step3p5ForCausalLM` | `Step3p5MTP` | — |
| `Bailing*` | `BailingMoeForCausalLMNextN` | — |
| `Ernie4_5_MoeForCausalLM` | `Ernie4_5_MoeForCausalLMMTP` | — |
| `Qwen3NextForCausalLM` | `Qwen3NextForCausalLMMTP` | 强制 `num_nextn_predict_layers = 1`（`:586`） |
| `Qwen3MoeForCausalLM` | `Qwen3MoeForCausalLMMTP` | 强制 `num_nextn_predict_layers = 1`（`:590`） |

> 命名不统一是历史遗留：DeepSeek/GLM/Bailing 系用后缀 `NextN`，Qwen/MiMo/Step/Ernie 系用后缀 `MTP`。二者语义相同——都是"从 target 架构里析出的、只含 MTP 预测层的 draft 模型类"。这些 `...NextN` / `...MTP` 类都通过 `EntryClass` 注册到模型注册表（如 `deepseek_nextn.py` 末尾 `EntryClass = [DeepseekV3ForCausalLMNextN]`）。

---

## 5. Draft 权重加载

MTP draft 的权重**不来自单独文件**，而是和 target 权重躺在同一批 safetensors 里，需要在加载时"筛选 + 重映射"。

### 5.1 `_filter_mtp_weights`：切片 + 重映射（`loader.py:622`）

target checkpoint 里 MTP 层的权重名形如 `model.mtp.layers.{idx}.xxx`（正则 `_MTP_PATTERN = model\.mtp\.layers\.(\d+)\.`, `loader.py:358`）。当 `load_config.draft_model_idx` 非空（即当前在加载 draft），`_get_weights_iterator` 转入 `_filter_mtp_weights`：

```python
for name, tensor in weights_iterator:
    match = cls._MTP_PATTERN.match(name)
    if match is not None:
        idx = int(match.group(1))
        if idx != draft_model_idx:      # 只保留指定那一层 MTP
            continue
        new_name = name.replace(match.group(), "model.mtp.layers.0.")  # 重映射到 layer 0
    else:
        new_name = name               # 非 MTP 权重（embedding、lm_head 等）原样透传
    yield (prefix + new_name, tensor)
```

要点：
- **只保留 `draft_model_idx` 那一层**：多 MTP 层的模型（或 multi-layer EAGLE 每个 step 用不同层）借此选层；单层 MTP 就是 `idx=0`。
- **重映射到 `model.mtp.layers.0.`**：draft 模型内部只有一层，统一挂到 index 0。
- **惰性 yield**（注释在 `:626`）：配合上游 buffered iterator 的滑动窗口限制 CPU 内存——之前 eager 物化会在大 MoE ckpt + 多层 EAGLE 下触发换页卡死。
- **非 MTP 权重原样透传**：embedding、lm_head、norm 等 target 主干权重会一起被 yield 出来，但——见下节，draft 实际上**并不真正加载**这些，而是运行时从 target 借。

### 5.2 draft_model_idx 从哪来（`tp_worker.py`）

在 `TpModelWorker._init_model_runner`（`tp_worker.py:395`）里：
- 单层 MTP / 普通 EAGLE：`draft_model_idx = 0 if is_multi_layer_eagle else None`（None 表示"不筛选、按普通模型加载"——但对内嵌 MTP 的模型，config 已改写成 NextN 类，模型自身的 `load_weights(is_nextn=True)` 会做筛选）。
- multi-layer EAGLE：`_init_multi_layer_eagle_model_runners`（`:398`）为每个 step `i` 建一个 `ModelRunner`，传 `draft_model_idx=i`，从而每步加载不同的 MTP 层。

### 5.3 共享 target 的 embedding 与 lm_head（`eagle_worker_v2.py:315`）

这是 MTP 省显存、保证分布一致的关键。draft 层**不独立持有** embedding 和 lm_head，而是运行时直接引用 target 的：

```python
# EagleDraftWorker.init_lm_head  (eagle_worker_v2.py:315)
embed, head = self.target_worker.model_runner.model.get_embed_and_head()
# MTP / NextN draft：把 target 的 embed 和 head 装到 draft 上
self.draft_model_runner.model.set_embed_and_head(embed, head)
```

以 DeepSeek 为例（`deepseek_v2.py:2860` 附近）：
- `get_embed_and_head()` 返回 `(model.embed_tokens.weight, lm_head.weight)`；
- `set_embed_and_head(embed, head)` 先 `del` draft 侧原有的两者，再把 target 的张量**引用**过来（不是 copy），因此 draft 与 target 用的是同一块权重显存。

这带来两个直接后果：
1. **词表投影完全一致**：draft 用 target 的 lm_head 投影，其 logits 与 target 在同一空间，接受率天然更高。
2. **省显存**：embedding + lm_head 是大权重（尤其大词表），共享后 draft 几乎只额外占用 1 层 transformer 的权重。

---

## 6. NextN 层结构

### 6.1 组成部件（以 `DeepseekModelNextN` 为例，`deepseek_nextn.py:70`）

NextN 层是 MTP draft 的"计算核心"。它比 target 的一个 decoder layer 多了三个专属部件，用来把"当前 token"和"上一步隐藏态"融合：

| 部件 | 作用 | 代码位置 |
|---|---|---|
| `embed_tokens` | 当前 token 的词嵌入（运行时会被 `set_embed_and_head` 换成 target 的） | `:96` |
| `enorm` | 对**当前 token embedding** 做 RMSNorm | `:103` |
| `hnorm` | 对**上一步隐藏态** `spec_info.hidden_states` 做 RMSNorm | `:104` |
| `eh_proj` | 把 `[enorm(embed) ‖ hnorm(hidden)]`（`2*hidden`）投影回 `hidden` | `:115` |
| `decoder` | 一个完整的 `DeepseekV2DecoderLayer(is_nextn=True)`（含 MLA + MoE） | `:148` |
| `shared_head.norm` | 输出前的最终 RMSNorm，随后接共享的 lm_head | — |

### 6.2 前向：融合"当前 token embedding"与"上一步 target 隐藏态"（`deepseek_nextn.py:163`）

NextN 的 `forward` 核心就一段（`:198-215`）：

```python
if hidden_states.shape[0] > 0:
    eh_input = torch.cat(
        (
            self.enorm(hidden_states),                       # 当前 token 的 embedding，归一化
            self.hnorm(forward_batch.spec_info.hidden_states)  # 上一步 target/draft 的隐藏态，归一化
            # (NPU 上还可能先 matmul(rot_weight) 做旋转对齐)
        ),
        dim=-1,
    )
    hidden_states = self.eh_proj(eh_input)   # 2*hidden → hidden
# 之后送入 self.decoder（MLA + MoE），再 shared_head.norm → lm_head
```

这正是 MTP/EAGLE 的**"接力"机制**：

```
     current_token ──embed──► enorm ─┐
                                      ├─ concat ─► eh_proj ─► [MLA+MoE decoder] ─► norm ─► lm_head ─► 下一 token 的 logits
  prev_hidden_state ────────► hnorm ─┘
        ▲
        └── 来自 forward_batch.spec_info.hidden_states
            （上一轮 verify 里 target 那个 token 位置的隐藏态，
              或 draft 循环里上一步的隐藏态）
```

- 第一步 draft：`spec_info.hidden_states` 来自上一轮 verify 的 bonus token 位置的 **target 隐藏态**；`current_token` 是 bonus token 本身。
- 后续 draft 步：`spec_info.hidden_states` 来自上一 draft 步的输出隐藏态；`current_token` 是上一步 argmax 出的 draft token。

> `eh_proj` 的输入维度是 `2*hidden`（`:108`/`:116`），因为它要吃下 embedding 和 hidden 拼接后的向量。这是识别一个模型 NextN 实现的标志性结构。GLM4-MoE 用同样的 `enorm/hnorm/eh_proj`；Qwen3 系改用 `fc` + `pre_fc_norm_embedding/pre_fc_norm_hidden`（换了名字，语义一致）；DeepSeek-V4 用 `e_proj/h_proj + hc_head`（mHC，见 §17.3）。

### 6.3 `DeepseekV3ForCausalLMNextN`：draft 的顶层封装（`deepseek_nextn.py:261`）

- 继承自 `DeepseekV3ForCausalLM`，复用其 `load_weights` / MoE 逻辑，但只实例化**一个** NextN 层。
- `hf_to_sglang_mapper`（`:266`）把 ckpt 里的 `model.layers.61`（DeepSeek-R1 MXFP4 命名）重映射到 `model.decoder`，从而对齐到 draft 内部的单层。
- `lm_head` 的 prefix 设为 `model.shared_head.head`（对齐 ckpt 里 MTP 层自带的 head 命名），但运行时该 head 会被 `set_embed_and_head` 覆盖成 target 的 lm_head。
- `load_weights(is_nextn=True)`：走 NextN 专用加载分支。
- `EntryClass = [DeepseekV3ForCausalLMNextN]`：注册到模型注册表，供 `_config_draft_model` 改写后的架构名命中。

---

## 7. Draft worker 初始化

### 7.1 两个 worker 的诞生（`scheduler.py:777` → `spec_info.py:193`）

启用投机解码后，`Scheduler.maybe_init_draft_worker`（`scheduler.py:777`）会：

```python
DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)  # scheduler.py:804
```

`SpeculativeAlgorithm.create_worker`（`spec_info.py:193`）按算法分派 worker 类。对 MTP（=EAGLE）：
- 普通 EAGLE / MTP → `EAGLEWorkerV2`（`spec_info.py:226`）
- EAGLE + `--enable-multi-layer-eagle` → `MultiLayerEagleWorkerV2`
- （Gemma4 助手 draft 被别名成 FROZEN_KV_MTP → `FrozenKVMTPWorkerV2`）

`EAGLEWorkerV2` 内部同时持有 **target worker** 和 **draft worker**（`EagleDraftWorker`）。二者各自是一个 `TpModelWorker`，但 draft worker 建 config 时走 `is_draft_worker=True` 分支：

```python
# tp_worker.py:_init_model_config (357)
model_path = server_args.speculative_draft_model_path if is_draft_worker else server_args.model_path
# 对 MTP，speculative_draft_model_path 已被 hook 设成 == model_path
ModelConfig.from_server_args(..., is_draft_model=is_draft_worker)  # 触发 §4.2 的架构改写
```

### 7.2 draft 只加载 MTP 层（`model_runner.py:742`）

`ModelRunner` 初始化时决定要建多少层：

```python
_nnpl = self.model_config.num_nextn_predict_layers
model_has_mtp_layers = _nnpl is not None and _nnpl > 0
model_num_layers = (
    _nnpl                                    # ← draft：只建 MTP 层数（通常 1）
    if self.is_draft_worker and model_has_mtp_layers
    else max(num_hidden_layers, num_attention_layers)  # target：建全部层
)
```

几个坑点（注释在 `model_runner.py:734-741`）：
- **EAGLE3 draft 显式 `num_nextn_predict_layers: 0`**（如 Kimi-K2.5 Eagle3 带了完整 DeepSeek-V3 config schema）必须当作"字段缺失"处理，否则会走进 MTP 分支、把 draft KV pool 建成 0 层，首次 forward 时 `set_mla_kv_buffer` 索引 `kv_buffer[layer_id - start_layer]` 越界 IndexError。这就是 `model_has_mtp_layers` 要求 `> 0` 的原因。
- `MiMoV2MTP` / `Step3p5MTP` 强制 `model_num_layers = 1`（`:752`）。
- **PP 与 MTP 不兼容**的断言（`:767`）：MTP 模型必须 `num_effective_layers == model_num_layers`，即 draft 层不能被 pipeline 切分。

### 7.3 embedding/lm_head 共享的时机

`EAGLEWorkerV2` 构造完 target 和 draft worker 后，调用 `EagleDraftWorker.init_lm_head`（`eagle_worker_v2.py:315`），把 target 的 embedding 和 lm_head 装到 draft（见 §5.3）。到此，draft worker 具备：
- 自己的 1 层 MTP transformer（含 `enorm/hnorm/eh_proj`）；
- 借来的 target embedding + lm_head；
- 一套 draft 专用的 attention 后端（`MultiStepDraftBackend`，见 §15）和 CUDA graph runner（见 §14）。

---

## 8. 核心循环

MTP 一步 decode 的顶层入口是 `EAGLEWorkerV2.forward_batch_generation`（`eagle_worker_v2.py:1081`）。它按 forward_mode 分两大分支：**prefill**（首次，需要给出第一个 draft 的隐藏态起点）和 **decode**（稳态循环）。

### 8.1 decode 稳态：draft → verify → draft_extend

```
forward_batch_generation (decode 分支, eagle_worker_v2.py:1081)
   │
   ├─ ① draft(batch)                        # eagle_worker_v2.py:476
   │      prepare_for_draft → draft_forward(num_steps 次) → build_tree_kernel_efficient
   │      产出 EagleVerifyInput（draft_token / tree_mask / retrieve_* / positions）
   │
   ├─ ② verify(batch)                       # eagle_worker_v2.py:1447
   │      eagle_prepare_for_verify（input_ids=draft_token, mode=TARGET_VERIFY）
   │      → target.forward(is_verify=True)  一次全量前向
   │      → eagle_sample 沿树/链接受最长正确前缀
   │      → fill_bonus_tokens
   │      产出 accept_index / predict（含 bonus）/ next_draft_input(bonus_tokens, ...)
   │
   └─ ③ _draft_extend_for_decode(batch)     # eagle_worker_v2.py:816
          用被接受位置的 target 隐藏态再跑一次 draft 层，
          把 topk_p / topk_index / hidden_states 写到 next_draft_input，
          作为下一轮 ① 的起点；num_correct_drafts = accept_lens - 1
```

三步产物 `next_draft_input`（一个 `EagleDraftInput`）会被 scheduler 挂回 `batch.spec_info`，驱动下一轮。

### 8.2 prefill 分支

prefill 时还没有"上一步隐藏态"，需要：
1. target 正常 extend/prefill 一遍（算出 prompt 的隐藏态和第一个真 token）；
2. `_draft_extend_for_prefill`（`eagle_worker_v2.py:729`）用 prompt 末位隐藏态跑一次 draft 层，产出首个 `next_draft_input`；
3. 从下一步起进入 §8.1 的 decode 稳态。

---

## 9. Draft forward

`draft_forward`（`eagle_worker_v2.py:582`）是 draft 层的多步驱动器。它从 `spec_info` 拿到起点 `(topk_p, topk_index, hidden_states)`，循环 `speculative_num_steps` 次逐步生成候选。

### 9.1 主循环骨架（`eagle_worker_v2.py:616`）

```python
for i in range(self.speculative_num_steps):
    # ① 从上一步的 topk 里挑当前步要 forward 的 token（链取 top1，树展开 topk）
    input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
        i, topk_p, topk_index, hidden_states, scores, self.topk)
    score_list.append(tree_info[0]); token_list.append(tree_info[1]); parents_list.append(tree_info[2])

    if i == self.speculative_num_steps - 1:
        break   # 最后一步不必再 forward：draft-prefill 给 1 个 + 这里循环给 (num_steps-1) 个

    # ② 送入 draft 层 forward（用第 i 个 draft attn backend）
    forward_batch.input_ids = input_ids
    forward_batch.out_cache_loc = out_cache_loc[i]
    spec_info.hidden_states = hidden_states
    with forward_context(ForwardContext(attn_backend=self.draft_attn_backend.attn_backends[i])):
        logits_output = self.draft_runner.forward(forward_batch).logits_output

    # ③ 从 logits 采下一步候选
    if speculative_use_rejection_sampling:
        probs = renorm_draft_probs(...); topk_p, topk_index = fast_sample(probs, 1)
    elif self.topk == 1 and not _is_hip:           # ← MTP 主路径：贪心 argmax 快路径
        topk_index = torch.argmax(logits_output.next_token_logits, dim=-1, keepdim=True)
        topk_p = torch.ones_like(topk_index, dtype=torch.float32)
    else:
        probs = renorm_draft_probs(...); topk_p, topk_index = fast_topk(probs, self.topk)
```

### 9.2 关键设计点

| 点 | 说明 | 代码 |
|---|---|---|
| **每步一个独立 attn backend** | `attn_backends[i]`：draft 逐步产生越来越长的 KV，各步的 metadata（seq_len、页表）不同，用 `MultiStepDraftBackend` 的第 i 个实例 | `:651` |
| **最后一步跳过 forward** | draft-prefill 已产出 1 个候选，循环再产出 `num_steps-1` 个，共 `num_steps` 个，正好凑齐 verify 需要的候选（+bonus 根 = `num_steps+1`） | `:625` |
| **topk==1 argmax 快路径** | 链式时直接 argmax，比 renorm+topk 省一次采样。**门控 `not _is_hip`**：ROCm 上 argmax 的 tie-break 会破坏 MTP FP8 draft 选择（PR #26358），故 HIP 走通用 topk 路径 | `:667` |
| **rejection sampling 分支** | `--speculative-use-rejection-sampling` 时保存每步 `draft_probs`，供 verify 侧拒绝采样用 | `:608/:659` |
| **`index_share_for_mtp_iteration`** | topk==1 专用优化：draft 各步共享 mtp_topk_indices，减少 KV 写入的索引计算（`__init__:194`） | `:613` |
| **NaN/Inf/OOB 检测** | 每步 `maybe_detect_nan/inf/oob`，debug 用 | `:657` |

### 9.3 `select_top_k_tokens`（`spec_utils.py:262`）

它是"链 vs 树"分叉的实际执行者：
- **第 0 步**（`_select_top_k_tokens_first`, `spec_utils.py:205`）：从起点 topk 展开初始候选。
- **后续步**（`_select_top_k_tokens_later`, `:226`，`torch.compile` 加速）：把上一步 topk × 当前候选组合，按 score 排序，形成树的下一层；topk==1 时退化成一条链。

产出的 `tree_info = (score_list, token_list, parents_list)` 记录了每个候选的父子关系，供后续 `build_tree_kernel_efficient` 构造 mask 和 retrieve 索引。

---

## 10. Tree building

draft 循环产出的一堆候选 token（及其父子关系）需要组织成一个可供 target 一次并行校验的结构，这就是 `build_tree_kernel_efficient`（`eagle_utils.py:128`）的职责。

### 10.1 chain（链） vs tree（树）

| 拓扑 | 触发条件 | 结构 | verify 一次校验 |
|---|---|---|---|
| **链 chain** | `topk == 1`（MTP 默认） | bonus → d0 → d1 → ... → d(k-1)，单条路径 | `num_steps+1` 个位置 |
| **树 tree** | `topk > 1`（Llama EAGLE 默认） | bonus 为根，每层展开 topk 个分支 | `num_draft_tokens` 个位置 |

链式是树的退化特例（每个节点只有一个孩子），因此二者共用同一套 kernel，只是 topk==1 时树只有一条枝。

### 10.2 bonus token 作为树根（`eagle_utils.py:142`）

```python
draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()
```

**bonus token** 是上一轮 verify 里 target 采出的那个"必对的下一个真 token"。它被放在候选序列最前面当**根节点**：因为它一定会被接受（它就是 target 的真输出），draft 候选是挂在它之后的"预测的预测"。这也是为什么链式下候选总数是 `num_steps + 1`（1 个 bonus + num_steps 个 draft）。

### 10.3 三大产物

`build_tree_kernel_efficient` 用 CUDA kernel（`sgl_build_tree_kernel` / NPU 分支）一次性算出：

1. **`tree_mask`**：每个候选 token 能 attend 到哪些前驱（含 prompt + 树上祖先）。三种编码模式：
   - `FULL_MASK`：完整 bool 矩阵，`seq_lens_sum * num_verify_tokens + num_verify_tokens² * bs`（`:175`）；
   - `QLEN_ONLY`：只存候选之间的 `num_verify × num_verify` 子块（`:160`），前缀部分默认全可见；
   - `QLEN_ONLY_BITPACKING`：把上面的 bool 按位打包成 uint8/16/32（`:167`），省显存/带宽。
2. **`retrieve_index / retrieve_next_token / retrieve_next_sibling`**（`:188`）：树的遍历索引，供 verify 后沿最长接受路径回溯 token。
3. **`positions`**（`:195`）：每个候选在序列中的位置编号（= prompt 长度 + 树深度），供 RoPE 用。例如深度 `[0,1,1,2]`、prompt 长 7 → `positions=[7,8,8,9]`。

产物打包进 `EagleVerifyInput`（`eagle_info.py:18`），其 `max_tree_depth = spec_steps + 1`。

### 10.4 直接写 CUDA graph buffer

注意 draft 会先取 verify 后端的预分配 buffer（`get_verify_buffers_to_fill_after_draft`, `eagle_worker_v2.py:527`），把 `tree_mask` / `positions` **直接写进 verify 的 CUDA graph buffer**，避免额外拷贝——这样 verify 用 CUDA graph replay 时能直接读到本轮的树结构。

---

## 11. Verify

`verify`（`eagle_worker_v2.py:1447`）用 target 全量模型对候选树做一次并行前向，然后采样接受最长正确前缀。

### 11.1 verify 主流程

```
verify (eagle_worker_v2.py:1447)
   ├─ verify_input.num_tokens_per_req = num_steps + 1
   ├─ eagle_prepare_for_verify (eagle_utils.py:336)
   │      batch.input_ids = verify_input.draft_token        # 树上所有候选 token 铺平
   │      forward_mode = TARGET_VERIFY (:387)
   │      capture_hidden_mode = FULL                        # 需要全部位置隐藏态供 draft-extend
   ├─ (overlap 下) 重算 custom_mask / positions            # 依赖 draft 输出，plan 阶段用的是旧值
   ├─ (has_grammar) 把 retrieve_next_token/sibling/draft_token 拷到 CPU
   ├─ target_worker.forward_batch_generation(is_verify=True) # ← target 一次全量前向
   ├─ eagle_sample (eagle_utils.py:416)                     # 采样接受
   └─ fill_bonus_tokens → next_draft_input = EagleDraftInput(bonus_tokens=..., ...)
```

`eagle_prepare_for_verify` 把整棵树的候选 token 当作一次 "extend/prefill"（固定长度 `num_steps+1`）送进 target，`ForwardMode.TARGET_VERIFY`（`forward_batch_info.py:90`）配合 tree_mask 让每个候选只 attend 到自己的祖先，从而在一次 forward 里并行算出所有候选位置的 logits。

### 11.2 采样接受（`eagle_sample`, `eagle_utils.py:416`）

根据是否贪心 / 是否拒绝采样，走不同 kernel：

| 模式 | kernel | 说明 |
|---|---|---|
| 贪心（默认 temperature=0） | `verify_tree_greedy_func`（`:500`） | 沿树取 argmax，逐层比对候选是否 == target argmax，接受最长匹配 |
| 随机采样（tree） | `tree_speculative_sampling_target_only`（`:574`） | 按 target_probs 采样，配合 accept 阈值 |
| 拒绝采样（chain, topk==1） | `chain_speculative_sampling_triton`（`:572`） | 需要 draft_probs（`--speculative-use-rejection-sampling`）；用 coins 做经典拒绝采样，理论上分布无偏 |

三者都原地写 `predict`（每步接受的 token）、`accept_index`（被接受候选在树里的索引）、`num_correct_drafts`（接受的 draft 数）。

- **target_probs 构造**（`:526-548`）：`softmax(logits/temperature)` → `top_k_renorm` → `top_p_renorm` → reshape 成 `(bs, num_draft_tokens, vocab)`。
- **拒绝采样校验**（`:557`）：若开了 rejection sampling 但 draft_probs 缺失或词表不匹配，直接 `raise`（防插件/子类绕过 startup 校验）。
- **TP 一致性广播**（`:594`）：不同 GPU 的 softmax/topk 浮点非确定性可能导致采出不同 token，从 rank 0 广播 `predict/accept_index/num_correct_drafts` 保证一致。

### 11.3 命名关键点：`num_correct_drafts` vs `num_accept_tokens`（`eagle_utils.py:619`）

这是 speculative-naming skill 的核心约定，务必分清：

```python
# eagle_sample 内部 num_correct_drafts 始终是"仅 draft 数"（不含 bonus）
# 返回时 out-of-place +1，把 bonus token 算进去，语义变为 num_accept_tokens
return predict, num_correct_drafts + 1, accept_index
#                └─────────┬─────────┘
#                num_accept_tokens = num_correct_drafts + 1（+1 是 bonus token）
```

| 名字 | 含义 | 值 |
|---|---|---|
| `num_correct_drafts` | 被接受的 **draft** token 数（不含 bonus） | `0 ~ num_steps` |
| `bonus_token` | 那个"必对"的 target 真 token（树根 / 尾部 +1） | 恒 1 个 |
| `num_accept_tokens` | 本轮总共提交的 token 数 | `= num_correct_drafts + 1` |
| `accept_length` / `accept_rate` | 论文口径的接受长度 / 接受率 | 用于指标 |

> 记住：**`accept_*` 含 bonus，`correct_*` 只算 draft**。`num_correct_drafts` 在函数内部保持"仅 draft"语义，只在 return 那一刻 `+1` 变成 `num_accept_tokens`，以免中途语义翻转（naming doc C2）。

### 11.4 bonus token 回填与 next_draft_input（`eagle_worker_v2.py:1583`）

verify 结束后调用 `fill_bonus_tokens`，把 target 采出的下一个真 token 填成 bonus，并构造下一轮的 `EagleDraftInput`（`eagle_info.py:145`）：
- `bonus_tokens`：下一轮树根；
- 隐藏态/topk 会在紧接着的 `_draft_extend_for_decode` 里补齐（见 §12）。

---

## 12. Draft extend

`_draft_extend_for_decode`（`eagle_worker_v2.py:816`）是"承上启下"的一步：verify 已经确定了本轮接受哪些 token，但**下一轮 draft 需要一个隐藏态起点和 topk 起点**——这个起点来自"被接受的最后一个 token 位置"上再跑一次 draft 层。

### 12.1 为什么需要 draft-extend

draft 循环的每一步都需要 `spec_info.hidden_states`（上一步隐藏态）和 `spec_info.topk_index`（当前 token）。verify 之后：
- 接受了 `num_accept_tokens` 个 token，序列往前走了这么多；
- 下一轮 draft 要从"新序列末尾"开始预测，因此需要末尾那个 token 位置的 **draft 层隐藏态** 和它的 topk 分布。

`_draft_extend_for_decode` 就是把 verify 阶段 target 产出的、**被接受位置**的隐藏态，喂进 draft 层再前向一次（`DRAFT_EXTEND_V2` 模式），得到下一轮起点。

### 12.2 select_index：定位每条请求的接受末位（`eagle_worker_v2.py:830`）

```python
draft_extend_input = EagleDraftExtendInput(
    hidden_states=batch_result.logits_output.hidden_states,  # verify 的全部候选位置隐藏态
    num_correct_drafts=batch_result.accept_lens - 1,         # accept_lens 含 bonus，减 1 得 draft 数
    num_accept_tokens=batch_result.accept_lens,
    num_tokens_per_req=self.speculative_num_draft_tokens,     # 填满整棵树宽（不是 num_steps+1），保 DP MLP-sync padding 一致
    ...
)
# 每条请求接受末位在铺平张量中的下标：req_base + accept_len - 1
select_index = torch.arange(0, bs * num_draft_tokens, num_draft_tokens, device=...) + accept_lens - 1
```

`select_index` 精确地从 `bs * num_draft_tokens` 个候选位置里，挑出每条请求"最后一个被接受的 token"所在行。

### 12.3 前向 + 选行 + 采下一轮 topk（`eagle_worker_v2.py:884-931`）

```python
draft_logits_output = draft_runner.forward(forward_batch).logits_output   # draft 层前向（可 CUDA graph）
# 只取被接受末位那一行
draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[select_index]
draft_logits_output.hidden_states     = draft_logits_output.hidden_states[select_index]

# 采下一轮 draft 起点的 topk
if speculative_use_rejection_sampling:
    ret_topk_p, ret_topk_index = fast_sample(probs, 1); ret_draft_probs = probs
elif self.topk == 1 and not _is_hip:          # ← MTP 主路径：argmax 快路径（同 §9.2，#26358 门控）
    ret_topk_index = torch.argmax(draft_logits_output.next_token_logits, dim=-1, keepdim=True)
    ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
else:
    ret_topk_p, ret_topk_index = fast_topk(probs, self.topk)
```

### 12.4 写回 next_draft_input（`eagle_worker_v2.py:933`）

最后把 `(topk_p, topk_index, hidden_states)` 写到 `next_draft_input`：

```python
next_draft_input = batch_result.next_draft_input
(next_draft_input.topk_p, next_draft_input.topk_index, next_draft_input.hidden_states) = (
    ret_topk_p, ret_topk_index, ret_hidden_states)
```

这个 `next_draft_input`（`EagleDraftInput`，含 `bonus_tokens`+`topk_*`+`hidden_states`）就是下一轮 `draft()` 的完整输入。scheduler 会把它挂回 `batch.spec_info`（见 §16），闭环完成。

> 对比 **Frozen-KV MTP**（§17.2）：它**没有 draft-extend 这一步**——因为它的 draft（Gemma4 助手）只读 target KV、不维护自己的 draft KV 链，所以不需要为下一轮准备 draft 隐藏态。这是普通 MTP 与 Frozen-KV MTP 最大的运行时差异。

---

## 13. KV 缓存分配

投机解码要为 draft 候选和 verify 树预留 KV 槽位，与普通 decode 每步只分配 1 个 token 不同。MTP（topk==1 链式）的分配相对简单，topk>1 树式则要按页对齐。

### 13.1 draft 阶段的 cache locs（`base_spec_worker.py:176`）

`prepare_for_draft` 为 draft 循环分配写入位置，分两条路：

| 条件 | 分配方式 | 代码 |
|---|---|---|
| `page_size == 1` **或** `topk == 1`（MTP 主路径） | **连续分配** `bs*topk*num_steps` 个槽，`assign_draft_cache_locs_contiguous` Triton kernel 一把填 | `:199-214` |
| `page_size > 1` **且** `topk > 1`（树 + 分页） | **每分支页对齐**：各 branch 独立页，`duplicate_prefix_tail_to_draft_branches` 把前缀尾页 KV 复制到每个分支，保证整页读一致 | `:215-255` |

MTP 因为 topk==1，永远走连续分配那条简单路径——这也是 `has_draft_kv()`（`spec_info.py:121`）在链式下无需 per-topk 页对齐的原因。

### 13.2 decode 阶段的双倍预留（`eagle_utils.py:625`）

`eagle_prepare_for_decode` 为下一步 decode 预分配 KV：

```python
double_alloc = get_alloc_reserve_per_decode()
for i, r in enumerate(batch.reqs):
    cur = r.kv_allocated_len
    nxt = max(cur, r.kv_committed_len + double_alloc)   # max 钳制，防自适应降档使 nxt < cur
    num_needed_tokens += nxt - cur
    r.kv_allocated_len = nxt
```

关键点：
- **`kv_committed_len` 落后 `batch.seq_lens` 约一拍**（overlap 下 bonus 在 resolve 阶段才提交，不在这里），所以用 `2*alloc`（`double_alloc`）吸收这个滞后。
- **page>1 + topk>1 的越界防护**（`:664`）：若过量分配超出 `req_to_token` 行宽，直接 `assert` 报清晰错误（PR #26972 加宽了行），而不是等到后面某个 CUDA assert 崩溃。
- **non-blocking H2D**（`:673`）：用非阻塞拷贝避免同步 schedule stream 导致 host 停顿一整个 forward。

### 13.3 draft KV pool 的层数与压缩

- draft 只建 `num_nextn_predict_layers`（通常 1）层的 KV pool（见 §7.2），远小于 target 全量层。
- 对 DeepSeek-V4，draft 层强制 `COMPRESS_RATIO_NEXTN_LAYER = 0`（`deepseek_v4_nextn.py:47`，`model_runner_kv_cache_mixin.py:463`），即 **draft 只用全精度 SWA、不做压缩**——draft 追求速度和简单，不需要 V4 target 那套三层压缩 KV。

---

## 14. CUDA Graph 捕获

MTP 因为一步 decode 里有 draft/verify/draft-extend 三个不同形状的 GPU 阶段，需要**三套独立的 CUDA graph**。

### 14.1 三套 CUDA graph runner

| 阶段 | runner | forward_mode | num_tokens_per_bs | 位置 |
|---|---|---|---|---|
| **verify**（target 全量前向） | `CudaGraphRunner`（target 侧，`decode_cuda_graph_runner.py`） | `TARGET_VERIFY` | `get_num_tokens_per_bs_for_target_verify(num_draft_tokens)` | `:233-247` |
| **draft decode**（draft 层多步） | `EAGLEDraftCudaGraphRunner`（`eagle_draft_cuda_graph_runner.py`） | 内部 DECODE | `self.topk` | `:143` |
| **draft extend**（承上启下那一次） | `EagleDraftExtendCudaGraphRunner` | `DRAFT_EXTEND_V2` | — | — |

### 14.2 verify 的捕获（`decode_cuda_graph_runner.py:233`）

target worker 的 CUDA graph runner 检测到 spec 开启后，把捕获模式设为 `TARGET_VERIFY`（`:242`），每 bs 的 token 数 = `num_draft_tokens`（树宽/链长）。对 MTP 模型，还要为 `spec_info.hidden_states` 分配零 buffer（因为 NextN 前向要读它，见 §6.2）。

关键约束（`:238`）：draft worker 想用 `TARGET_VERIFY` 模式捕获，必须 `supports_target_verify_for_draft()` 返回 True——目前只有 DFLASH 支持，普通 MTP/EAGLE 的 draft worker 不走这条（它们的 draft 用下面的 draft runner）。

### 14.3 draft decode 的捕获（`eagle_draft_cuda_graph_runner.py:128`）

这是 draft 层多步循环的图。要点：
- **`num_tokens_per_bs = self.topk`**（`:143`）：draft 每步处理 topk 个候选（MTP topk==1 时就是 1）。
- **`out_cache_loc` 尺寸 = `max_num_token * speculative_num_steps`**（`:162`）：要容纳所有 draft 步的写入位置。
- **静态 buffer**（`:157-201`）：`topk_p`、`topk_index`（`(max_bs, topk)`）、`hidden_states`（`(max_bs, hidden_size)`，尺寸由 `get_draft_recurrent_hidden_state_spec` 决定）、rejection sampling 时的 `draft_probs`（`(max_bs, vocab)`）。
- **每步一个 attn backend**：`draft_attn_backend.attn_backends[i]`（`:148`），与 §9.2 呼应。
- 关掉了父类里对 EAGLE 不适用的路径：`compile_bs=[]`（不 torch.compile 包装）、`enable_pdmux=False`、`is_dllm=False`（`:129-133`）。

### 14.4 图与真实 forward 的衔接

draft/verify 用 CUDA graph replay 时，本轮的动态数据（tree_mask、positions、topk 起点）必须写进图的静态 buffer。§10.4 提到 draft 直接写 verify 的 buffer；overlap 下 plan 阶段先用旧值占位、`update_verify_buffers_to_fill_after_draft`（`eagle_worker_v2.py:1489`）在 draft 出结果后重算并覆盖，保证 replay 读到正确的树。

---

## 15. 注意力后端

draft 循环的每一步都会写入越来越长的 draft KV 链，每步的 attention metadata（seq_len、页表、mask）都不同。SGLang 用 **`*MultiStepDraftBackend`** 把"每步一个 attention backend 实例"打包成一个对象。

### 15.1 DraftBackendFactory（`draft_utils.py:9`）

`EAGLEWorkerV2.__init__` 用 `DraftBackendFactory` 造两个后端（`eagle_worker_v2.py:348-360`）：

```python
draft_backend_factory = DraftBackendFactory(server_args, draft_model_runner, topk, num_steps)
self.draft_attn_backend = draft_backend_factory.create_decode_backend()        # draft 多步循环用
self.draft_extend_attn_backend = draft_backend_factory.create_draft_extend_backend()  # draft-extend 用
```

- **`create_decode_backend`**（`:39`）：`num_steps <= 1` 时返回 None（单步无需多步后端，draft() 里只 sample 不 forward）；否则按 `decode_attention_backend` 选具体实现，包成 `*MultiStepDraftBackend`。
- **`create_draft_extend_backend`**（`:72`）：按 `prefill` 系后端选，供 draft-extend（固定长度 extend）用。

### 15.2 支持的后端矩阵

`create_decode_backend` 的 backend_map（`draft_utils.py:44-64`）覆盖：

| 后端 | draft decode | 备注 |
|---|---|---|
| `flashinfer` / `fa3` / `fa4` / `triton` | ✅ | 通用；page-tree（topk>1 + page>1）仅 flashinfer/fa3/triton（`speculative_hook.py:395`） |
| `flashmla` / `trtllm_mla` / `cutedsl_mla` / `tokenspeed_mla` | ✅ | MLA 系（DeepSeek） |
| `dsa` / `nsa`（别名） | ✅ | DeepSeek 稀疏注意力 |
| `dsv4` | ✅ | DeepSeek-V4 专用 |
| `aiter` / `ascend` | ✅ | AMD / NPU |
| `trtllm_mha` | ✅ | 仅 topk==1（`speculative_hook.py:331`） |

### 15.3 `TritonMultiStepDraftBackend`（示例，`triton_backend.py:1726`）

以 Triton 为例，`TritonMultiStepDraftBackend` 内部持有 `attn_backends[0..num_steps-1]`，"把多个 triton attention backend 包装成一个，服务连续多步 draft decoding"。draft_forward 第 i 步用 `attn_backends[i]`（`eagle_worker_v2.py:651`）。`needs_cpu_seq_lens = False`。各后端都有对应的 `*MultiStepDraftBackend`（flashinfer/fa3/flashmla/aiter/trtllm 等）。

---

## 16. 调度器集成

MTP 复用 EAGLE 的 V2 worker，而 V2 worker 有 **overlap（重叠）** 和 **non-overlap（同步）** 两种被 scheduler 驱动的方式（详见 `overlap_schedule_architecture.md`）。二者用**同一个** worker 类，区别只在 scheduler 如何调用它。

### 16.1 worker 的创建（`scheduler.py:777`）

```python
# maybe_init_draft_worker (scheduler.py:777)
DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)  # :804
# → 对 MTP：EAGLEWorkerV2
```

### 16.2 overlap 路径（`scheduler.py:3267`）

重叠调度下，forward 在 `forward_stream` 上异步发射，结果推迟一拍处理。MTP 的闭环靠这两行：

```python
if not batch.spec_algorithm.is_none():
    batch.spec_info = batch_result.next_draft_input   # scheduler.py:3268
    batch.spec_info.future_indices = future_indices
```

即：把本轮 `_draft_extend_for_decode` 产出的 `next_draft_input`（含下一轮 draft 起点）挂回 `batch.spec_info`，`input_ids` 置 None（下一轮从 draft_token 重建）。这样下一轮 `draft()` 就能拿到起点。

### 16.3 non-overlap 路径（`scheduler.py:3275`）

```python
elif not batch.spec_algorithm.is_none():
    # Non-overlap: 同步驱动 V2 worker（无 future_map relay）
    resolve_forward_inputs(batch, self.future_map)
    with self._forward_isolation(batch, overlap=False):
        batch_result = self.model_worker.forward_batch_generation(batch)  # 同步阻塞
    batch.spec_info = batch_result.next_draft_input
    if batch_result.new_seq_lens is not None:
        batch.seq_lens = batch_result.new_seq_lens
        ...
    batch.input_ids = None
    self.update_cache_from_scheduler(batch, batch_result)
    batch_result.copy_done = self.device_module.Event()
    batch_result.copy_to_cpu(...)   # 立即 D2H，供 result processor 读 CPU 张量
```

要点：
- **同一个 V2 worker**，只是这里**同步**调用 `forward_batch_generation`（`--disable-overlap-schedule` 触发，`speculative_hook.py:276` 会打日志）；
- forward isolation 会回滚 worker 在 forward 内对 ScheduleBatch 的修改，所以要**手动重新应用** `spec_info` / `seq_lens`；
- 立即做 D2H 拷贝（overlap 下这一步是延迟的）。

> 历史注记：早期有 spec v1（scheduler 直接管 draft/verify）和 spec v2 之分，且 v2 靠环境变量 `SGLANG_ENABLE_SPEC_V2` 开启。现在 **v1/v2 之分已消失**，一律走 V2 worker，`SGLANG_ENABLE_SPEC_V2` 已删除。真正的运行时开关就是 overlap vs non-overlap。

### 16.4 结果结算（`batch_result_processor.py:536`）

`_resolve_padded_next_token_ids_spec_v2` 把 verify 结果落到每条 Req：

```python
result.num_correct_drafts = sum(accept_lens) - len(batch.reqs)   # 全 batch draft 命中数（不含 bonus）
result.num_correct_drafts_per_req_cpu = [x - 1 for x in accept_lens]  # 每条请求 draft 命中
for i, req in enumerate(batch.reqs):
    accept_tokens = next_token_ids[i*stride : i*stride + accept_lens[i]]
    if req.grammar is not None:
        accept_tokens = self._accept_grammar_tokens(req, accept_tokens)  # 语法终止则截断
    num_accept_tokens = len(accept_tokens)
    req.kv_committed_len += num_accept_tokens   # 提交 draft+bonus
    req.spec_verify_ct += 1
    req.spec_num_correct_drafts += result.num_correct_drafts_per_req_cpu[i]
```

- `stride = speculative_num_draft_tokens`（自适应下用 result 上记录的值，防 worker 状态已切换）；
- `kv_committed_len += num_accept_tokens`：把本轮接受的 draft+bonus 一起提交（bonus 不在这里单独预占，见 `eagle_prepare_for_decode` 注释）；
- grammar：一旦文法状态机终止，截断过量草稿的后缀，不提交也不输出。

---

## 17. 变体

MTP 是 EAGLE 家族的一个配置，家族里还有几个近亲，它们共享 verify 契约但在 draft 侧有差异。

### 17.1 Multi-Layer EAGLE（`multi_layer_eagle_worker_v2.py`）

- **开关**：`--enable-multi-layer-eagle`；worker 类 `MultiLayerEagleWorkerV2`（`spec_info.py:218`）。
- **核心区别**：draft 不是"一层跑 num_steps 次"，而是 **num_steps 层，每层跑一次**。`_init_multi_layer_eagle_model_runners`（`tp_worker.py:398`）为每个 step `i` 建一个 `ModelRunner`，`draft_model_idx=i`，各加载不同的 MTP 层。
- `draft_runner_list`（`:158`）= 每步一个 ModelRunner；`draft_runner`（`:160`）= list[0]（兼容通用访问）。
- **强制约束**：`num_draft_tokens == num_steps + 1`（`:127`，即 topk==1 链式）。
- **`chain_mtp_hidden_states`**（`:165`）：`Step3p5MTP` 等模型每步把自己的输出隐藏态传给下一步（链式接力）；非链式则每步都用 target 的隐藏态。
- 适用于 checkpoint 里带了**多个 MTP 层**、想逐层加深预测的模型。

### 17.2 Frozen-KV MTP（`frozen_kv_mtp_worker_v2.py`）

- **触发**：`--speculative-algorithm NEXTN`（或 EAGLE）+ draft 架构是 `Gemma4AssistantForCausalLM` / `Gemma4UnifiedAssistantForCausalLM`，由 `_resolve_speculative_algorithm_alias`（`speculative_hook.py:42`）升级为 `FROZEN_KV_MTP`。worker 类 `FrozenKVMTPWorkerV2`（`frozen_kv_mtp_worker_v2.py:639`）。
- **核心区别**（`:80` docstring）：assistant draft **只读 target KV**，自己**不维护 draft KV 链**，因此：
  - **没有 `_draft_extend_for_decode` 这一步**（普通 MTP 最大差异，见 §12.4）；
  - draft 循环里自己管 seed 和递归，但不做 KV 扩展；
  - target pool 以只读方式在 `alloc_memory_pool` 绑定（`:117`，延迟绑定以支持 target pool 尚未建立时先造 worker，见 #29021）。
- `is_frozen_kv_mtp()` 被 `is_eagle()`（`spec_info.py:97`）**纳入** EAGLE 家族（有 FIXME 注释说等 scheduler 完整支持后移除）。
- Frozen-KV 意为"target 的 KV 冻结不动，draft 蹭着读"，适合 Gemma4 这类"助手模型 + 主模型共享 KV"的结构。

### 17.3 DeepSeek-V4 NextN（`deepseek_v4_nextn.py`）

DeepSeek-V4 的 MTP 层因 V4 的三层压缩 KV 架构而特殊：

| 特性 | 普通 NextN（V3） | DeepSeek-V4 NextN |
|---|---|---|
| 融合投影 | 单个 `eh_proj`（`2h→h`） | `e_proj` + `h_proj` + `hc_head`（mHC）（`:81`/`:75`） |
| 压缩比 | 跟随 target 各层 | 强制 `COMPRESS_RATIO_NEXTN_LAYER = 0`（`:47`），即**全精度 SWA、不压缩** |
| 层数 | `num_nextn_predict_layers` | `_config_draft_model` 强制 `= 1`（`model_config.py:538`） |
| 额外 head | 无 | `hc_head_fn` / `hc_head_base` / `hc_head_scale`（`:75-79`，multi-Head Compression 相关 float32 参数） |

> V4 draft 强制 ratio=0 的原因：draft 追求速度和实现简单，用近窗全精度 SWA 足以，无需 target 那套 C4/C128 稀疏压缩池（详见 `deepseek_v4_cache_management.md`）。

### 17.4 其它 EAGLE 家族成员（非 MTP，仅作区分）

| 算法 | worker | 与 MTP 的关系 |
|---|---|---|
| **EAGLE / EAGLE3** | `EAGLEWorkerV2` | 同一 worker；draft 是**独立 ckpt**（不内嵌于 target），是否 MTP 只看 draft 从哪来 |
| **STANDALONE** | `StandaloneWorkerV2` | 用一个完整的小模型当 draft，不共享 target embedding；`need_topk` 但 `carries_draft_hidden_states=False` |
| **NGRAM** | `NGRAMWorker` | 无 draft 模型，用 n-gram 匹配猜 token；`has_draft_kv()=False`（树只在 verify mask 里） |
| **DFLASH** | `DFlashWorkerV2` | 支持 `target_verify_for_draft`；draft 可用 TARGET_VERIFY CUDA graph |

---

## 18. PD 分离

PD 分离（Prefill-Decode 分离）下，prefill 实例算出 prompt，把 KV 和**投机所需的元数据**传给 decode 实例，decode 实例接力做 MTP。

### 18.1 需要额外传输的元数据

普通 PD 只传 KV cache。MTP（EAGLE 家族）额外需要把 draft 起点传过去。`carries_draft_hidden_states()`（`spec_info.py:127`）对 EAGLE 家族返回 True（STANDALONE 的 vanilla draft 忽略隐藏态，返回 False）。传输三样：

| 字段 | Req 上的名字 | 用途 |
|---|---|---|
| topk 概率 | `req.output_topk_p` | draft 第一步的 topk 起点 |
| topk 索引 | `req.output_topk_index` | draft 第一步的候选 token |
| 隐藏态 | `req.hidden_states_tensor` | NextN 层 `hnorm` 的输入 |

decode 实例在 `decode.py:1574` 附近，从 PD 传输的元数据里把这三样填回 Req。

### 18.2 build_eagle_disagg_draft_input（`eagle_disaggregation.py:17`）

decode 实例侧用它把传来的元数据组装成第一个 `EagleDraftInput`：

```python
def build_eagle_disagg_draft_input(batch, server_args, last_tokens_tensor, future_map):
    num_states = server_args.speculative_eagle_topk
    if server_args.enable_multi_layer_eagle:
        num_states *= server_args.speculative_num_steps
    topk_p       = torch.stack([req.output_topk_p[:num_states] for req in batch.reqs])
    topk_index   = torch.stack([req.output_topk_index[:num_states] for req in batch.reqs])
    hidden_states = torch.stack([req.hidden_states_tensor for req in batch.reqs]).to(device)
    spec_info = EagleDraftInput(
        topk_p=topk_p, topk_index=topk_index,
        hidden_states=hidden_states,
        bonus_tokens=last_tokens_tensor,   # prefill 采出的第一个真 token 当树根
    )
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
    if batch.enable_overlap:               # overlap 下还要 publish/stash 到 future_map
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(...); future_map.stash(...)
    return spec_info
```

这个 `spec_info` 相当于普通 MTP 里 prefill 阶段 `_draft_extend_for_prefill` 的产物，只不过跨实例传输过来。之后 decode 实例进入正常的 draft→verify→draft_extend 稳态。

### 18.3 分派入口

`SpeculativeAlgorithm.build_disagg_draft_input`（`spec_info.py:142`）在 EAGLE 家族时调用上面的函数：

```python
def build_disagg_draft_input(self, batch, server_args, last_tokens_tensor, future_map):
    if self.is_eagle():
        return build_eagle_disagg_draft_input(batch, server_args, last_tokens_tensor, future_map)
    return None
```

---

## 19. 命名规范与指标

MTP 复用 EAGLE 的命名规范（`speculative-naming` skill），务必分清几组易混概念。

### 19.1 计数命名

| 名字 | 含义 | 说明 |
|---|---|---|
| `num_correct_drafts` | 被接受的 **draft** token 数 | **不含** bonus；`eagle_sample` 内部语义 |
| `num_accept_tokens` | 本轮提交的总 token 数 | `= num_correct_drafts + 1`（含 bonus） |
| `bonus_token` | target 采出的"必对"真 token | 树根 / 尾部 "+1"，恒 1 个 |
| `accept_length` | 论文口径接受长度 | 用于观测指标 |
| `accept_rate` | 接受率 | 论文口径 |

命名约定的几条硬规则：
- **`accept_*` 含 bonus，`correct_*` 只算 draft**——不可混用。
- 计数用 `num_X`（如 `num_correct_drafts`），累加器用 `_ct` 后缀（如 `spec_verify_ct`），比率用 `_rate`。
- `eagle_sample` 内 `num_correct_drafts` 全程"仅 draft"，只在 return 时 out-of-place `+1` 变 `num_accept_tokens`，避免函数中途语义翻转（naming doc C2，`eagle_utils.py:619`）。

### 19.2 Req 上的累加字段（`batch_result_processor.py:575`）

```python
req.kv_committed_len += num_accept_tokens          # KV 已提交长度
req.spec_verify_ct += 1                             # 该请求经历的 verify 次数
req.spec_num_correct_drafts += num_correct_drafts   # 累计 draft 命中数
req.update_spec_correct_drafts_histogram(num_correct_drafts)  # 命中分布直方图
```

### 19.3 全局指标：`avg_spec_accept_length`（`scheduler.py:3733`）

```python
ret["avg_spec_accept_length"] = (
    self.metrics_reporter.spec_total_num_accept_tokens
    / self.metrics_reporter.spec_total_num_forward_ct
)
```

- 分子 `spec_total_num_accept_tokens`：所有 verify 累计接受 token 数（含 bonus）；
- 分母 `spec_total_num_forward_ct`：verify（target forward）次数；
- 商 = **平均每次 target forward 提交多少 token**。这是衡量 MTP 收益的核心指标：
  - 值 `≈ 1`：几乎没接受 draft，投机白做（甚至因 draft 开销变慢）；
  - 值 `= num_steps + 1`：所有 draft 全中，达到理论上限；
  - 实践中 DeepSeek-V3 MTP 常见 `1.8 ~ 2.5`（取决于任务）。

可通过 `--speculative-num-steps` 调节：步数越多潜在加速越大，但每步 draft 开销和 KV 占用也越大、且深层 draft 命中率下降，需实测权衡（`scripts/playground/bench_speculative.py`）。

也可用 `SIMULATE_ACC_LEN`（`spec_utils.py:98`）强制模拟固定接受长度，用于隔离测量 draft/verify 开销。

---

## 20. 关键文件 / 行号索引

## 20. 关键文件 / 行号索引

### 20.1 启用与参数

| 文件 | 关键符号 / 行 | 作用 |
|---|---|---|
| `arg_groups/speculative_hook.py` | `_resolve_speculative_algorithm_alias:14` | NEXTN→EAGLE / Gemma4→FROZEN_KV_MTP |
| | `handle_speculative_decoding:54` | 投机解码 arg 处理入口 |
| | `_handle_eagle_family:260` | EAGLE 家族参数补全（含 draft_model_path=model_path `:304`） |
| | `_auto_choose_speculative_params:512` | auto 默认 (steps,topk,draft_tokens) |
| | topk==1 对齐 `:382` | `num_draft_tokens = num_steps + 1` |
| `server_args.py` | `:1490-1559` | 所有 `speculative_*` CLI 参数定义 |

### 20.2 模型侧

| 文件 | 关键符号 / 行 | 作用 |
|---|---|---|
| `configs/model_config.py` | `_config_draft_model:523` | draft 架构名改写为 `...NextN`/`...MTP` |
| | `num_nextn_predict_layers:942` | MTP 支持的规范信号 |
| `models/deepseek_nextn.py` | `DeepseekModelNextN:70`，`forward:163` | NextN 层结构与 `enorm/hnorm/eh_proj` 融合 |
| | `DeepseekV3ForCausalLMNextN:261` | draft 顶层类，`EntryClass:362` |
| `models/deepseek_v4_nextn.py` | `COMPRESS_RATIO_NEXTN_LAYER=0 :47`，`e_proj/h_proj/hc_head:75-84` | V4 draft 全精度 + mHC |
| `models/deepseek_v2.py` | `get_embed_and_head/set_embed_and_head :2860` | embedding+lm_head 共享 |
| `model_loader/loader.py` | `_MTP_PATTERN:358`，`_filter_mtp_weights:622` | 切片 + 重映射 MTP 权重 |
| `model_executor/model_runner.py` | `:742-774` | draft 只建 MTP 层数 + PP 不兼容断言 |
| `managers/tp_worker.py` | `_init_model_config:357`，`_init_model_runner:375`，`_init_multi_layer_eagle_model_runners:398` | draft worker config/runner 构造 |

### 20.3 核心循环与算子

| 文件 | 关键符号 / 行 | 作用 |
|---|---|---|
| `speculative/eagle_worker_v2.py` | `EagleDraftWorker.init_lm_head:315` | 共享 target embed/head |
| | `draft:476` | draft 阶段入口 |
| | `draft_forward:582` | draft 多步循环（argmax 快路径 `:667`） |
| | `verify:1447` | target 校验入口 |
| | `_draft_extend_for_decode:816` | 承上启下，产 next_draft_input |
| | `_draft_extend_for_prefill:729` | prefill 阶段首个 draft 起点 |
| | `forward_batch_generation:1081` | 一步 decode 顶层（prefill/decode 分支） |
| `speculative/eagle_utils.py` | `build_tree_kernel_efficient:128` | 树/链构造（bonus 作根 `:142`） |
| | `eagle_prepare_for_verify:336` | 置 TARGET_VERIFY 模式 |
| | `eagle_sample:416` | 采样接受（return `num_correct_drafts+1 :622`） |
| | `eagle_prepare_for_decode:625` | decode KV 双倍预留 |
| `speculative/spec_utils.py` | `select_top_k_tokens:262`，`_select_top_k_tokens_first/later:205/226` | chain/tree token 选择 |
| | `SIMULATE_ACC_LEN:98` | 模拟固定接受长度 |
| `speculative/base_spec_worker.py` | `prepare_for_draft:176` | draft cache locs 分配（连续/页对齐） |
| `speculative/eagle_info.py` | `EagleVerifyInput:18`，`EagleDraftInput:145`，`EagleDraftExtendInput:280` | 投机 SpecInput 数据结构 |

### 20.4 CUDA Graph / 注意力后端

| 文件 | 关键符号 / 行 | 作用 |
|---|---|---|
| `model_executor/runner/decode_cuda_graph_runner.py` | `:233-247`（TARGET_VERIFY），`get_spec_info:1041` | verify 图捕获 |
| `speculative/eagle_draft_cuda_graph_runner.py` | `:128-201`（`num_tokens_per_bs=topk :143`） | draft decode 图捕获 |
| `speculative/draft_utils.py` | `DraftBackendFactory:9`，`create_decode_backend:39`，`create_draft_extend_backend:72` | 造 draft/draft-extend 后端 |
| `layers/attention/triton_backend.py` | `TritonMultiStepDraftBackend:1726` | 每步一个 attn backend（示例） |

### 20.5 调度器 / 分离 / 分派

| 文件 | 关键符号 / 行 | 作用 |
|---|---|---|
| `speculative/spec_info.py` | `SpeculativeAlgorithm:28`，`create_worker:193`，`is_eagle:94`，`carries_draft_hidden_states:127`，`build_disagg_draft_input:142` | 算法枚举与 worker 分派 |
| `managers/scheduler.py` | `maybe_init_draft_worker:777`，overlap `:3267`，non-overlap `:3275`，`avg_spec_accept_length:3733` | 调度器集成与指标 |
| `managers/scheduler_components/batch_result_processor.py` | `_resolve_padded_next_token_ids_spec_v2:536` | verify 结果结算到 Req |
| `speculative/eagle_disaggregation.py` | `build_eagle_disagg_draft_input:17` | PD 分离下组装首个 draft input |
| `disaggregation/decode.py` | `:1574` 附近 | decode 侧填回 topk_p/index/hidden |

### 20.6 变体 worker

| 文件 | 关键符号 / 行 | 作用 |
|---|---|---|
| `speculative/multi_layer_eagle_worker_v2.py` | `MultiLayerEagleDraftWorker:98`（`draft_runner_list:158`，`chain_mtp_hidden_states:165`） | 多层 EAGLE |
| `speculative/frozen_kv_mtp_worker_v2.py` | `FrozenKVMTPDraftWorker:80`，`FrozenKVMTPWorkerV2:639` | Gemma4 助手 draft，只读 target KV |

---

## 附录：一句话总结

> **MTP = EAGLE 家族的一个配置分支**：CLI 的 `NEXTN` 被别名成 `EAGLE`（`speculative_hook.py`），draft"模型"是 target checkpoint 内部由 `num_nextn_predict_layers` 切出的 1 层 MTP 层（`_filter_mtp_weights` + `_config_draft_model` 架构改写），运行时通过 `set_embed_and_head` 共享 target 的 embedding 和 lm_head。它复用完整的 EAGLE V2 `draft → verify → draft_extend` 循环（`eagle_worker_v2.py`），DeepSeek 系默认 `topk==1` 走链式。理解 EAGLE V2 的这三步，就理解了 MTP 的全部——差异只在 draft 从哪来、权重怎么加载、embedding/head 是否共享。


