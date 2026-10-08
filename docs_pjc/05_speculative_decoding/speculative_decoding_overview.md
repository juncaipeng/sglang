# SGLang 投机解码（Speculative Decoding）方案系统梳理

> 核心目录：`python/sglang/srt/speculative/`（含子包 `cpp_ngram/`、`dspark_components/`）
> Triton / CUDA kernel：`python/sglang/kernels/ops/speculative/`（**注意：已不在 `srt/speculative/triton_ops/`**）
> 关键外围：`python/sglang/srt/arg_groups/speculative_hook.py`、`python/sglang/srt/managers/scheduler.py`、`python/sglang/srt/mem_cache/allocation_sizing.py`、`python/sglang/srt/layers/attention/`
> 目标读者：需要理解、部署、调优或扩展 SGLang 投机解码的工程师。
>
> 本文由浅入深，系统覆盖 SGLang 主线当前的各投机解码方案：统一的 V2 框架、算法枚举与分发、三阶段主循环、EAGLE/EAGLE3 树形草稿、接受逻辑（贪心树 / 拒绝采样 / bonus token）、CUDA Graph 双 runner、以及 6 大算法家族（EAGLE 系、STANDALONE、NGRAM、DFLASH、**DSPARK**、FROZEN_KV_MTP）+ Multi-Layer EAGLE + 自适应 + 分离式（decoupled）部署，最后给出调度器集成、KV 预算、注意力后端支持矩阵、指标与命名规范。
>
> 📌 **引用口径（快照说明）**：本文的代码引用**以符号名（函数 / 类 / 方法）为准**，形如 `` `spec_info.py:SpeculativeAlgorithm.is_eagle()` ``——因为行号随重构漂移，符号名不会。仅当目标**没有可命名的外层符号**（模块级常量、或埋在长 `__init__` / 长函数体里的内联代码块）时才保留行号，并显式写成"（快照 :NNN）"。所有残留行号与本文全部事实陈述均基于 `main` @ `33ed29a0ee`。
>
> ⚠️ 读前必看的 5 条结论：
> 1. **只有一条运行路径（V2）**。旧的 spec V1 worker 已删除，`SGLANG_ENABLE_SPEC_V2` 环境变量已移除。overlap 与 non-overlap 都跑同一个 V2 worker，non-overlap 时由 scheduler 同步驱动。
> 2. **MTP / NEXTN 不是独立算法，是 EAGLE 的别名**。CLI 里的 `NEXTN` 在 `_resolve_speculative_algorithm_alias` 里解析成 `EAGLE`，枚举里根本没有 NEXTN 成员（只是保留字）。MTP 的本质是：draft "模型" 是 target 权重内部的 MTP 层，`speculative_draft_model_path` 自动设为 `model_path`。
> 3. **chain vs tree 完全由 `topk` 决定**。`topk==1` 走链式（chain），`topk>1` 走树形（tree），共用同一套 tree kernel。
> 4. **两个接受指标差一个 bonus token**：`accept_rate`（α，不含 bonus）vs `accept_length`（τ，含 bonus）。这是最易错的点。
> 5. **算法能力已"谓词化"，调用方不再 if-else 枚举**。`supports_mixed_chunk()` / `supports_grammar_overlap()` / `supports_ragged_verify()` / `has_draft_kv()` 这批谓词是参数钩子与调度器的唯一判据（见 §2.2）。例如"有 grammar 就强制关 overlap"这条老规则已作废，正是 `supports_grammar_overlap()` 的直接结果（见 §19.2）。

---

## 目录

1. [总览：投机解码与 SGLang 的统一框架](#1-总览)
2. [算法家族与分发机制](#2-算法家族与分发机制)
3. [快速上手：各算法启动姿势](#3-快速上手)
4. [CLI 参数全表](#4-cli-参数全表)
5. [参数钩子：别名解析与默认推导](#5-参数钩子)
6. [统一三阶段主循环：draft → verify → draft_extend](#6-统一三阶段主循环)
7. [核心数据结构：SpecInput 家族](#7-核心数据结构)
8. [EAGLE/EAGLE3 详解：多步草稿与树构建](#8-eagleeagle3-详解)
9. [Verify 与接受逻辑：贪心树 / 拒绝采样 / bonus token](#9-verify-与接受逻辑)
10. [CUDA Graph：为何两个 draft runner](#10-cuda-graph)
11. [算法详解：STANDALONE](#11-standalone)
12. [算法详解：NGRAM](#12-ngram)
13. [算法详解：DFLASH](#13-dflash)
14. [算法详解：DSPARK](#14-dspark)
15. [算法详解：FROZEN_KV_MTP](#15-frozen_kv_mtp)
16. [变体：Multi-Layer EAGLE](#16-multi-layer-eagle)
17. [自适应投机：动态 num_steps](#17-自适应投机)
18. [分离式投机：Decoupled Spec IO](#18-分离式投机)
19. [调度器集成：overlap vs non-overlap](#19-调度器集成)
20. [内存与 KV 预算](#20-内存与-kv-预算)
21. [注意力后端支持矩阵](#21-注意力后端支持矩阵)
22. [指标与可观测性](#22-指标与可观测性)
23. [命名规范](#23-命名规范)
24. [方案对比总表](#24-方案对比总表)
25. [文件索引](#25-文件索引)

---

## 1. 总览

### 1.1 投机解码解决什么问题

自回归解码每步只产出 1 个 token，且每步都要把**整个模型**的权重从 HBM 搬到 SM（decode 阶段严重 **memory-bound**，算力利用率极低）。投机解码（Speculative Decoding）的核心思想：

- 用一个**便宜**的 draft（草稿）机制一次性提出 `k` 个候选 token；
- 用**昂贵**的 target（目标）模型**一次 forward**并行验证这 `k` 个候选；
- 通过一套接受（accept）规则保留一段前缀，保证**输出分布与 target 单步采样严格等价**（贪心/拒绝采样两种保证方式）。

一次 target forward 若能接受 `τ` 个 token（`τ≥1`），则等效吞吐提升约 `τ` 倍，而 target forward 的开销几乎不变（decode 本来就 memory-bound，多验证几个 token 近乎免费）。`τ` 即 **accept length**。

```
传统自回归：            投机解码（一次 verify 接受 3 个）：
step1: forward → t1     draft:  propose [d1 d2 d3 d4]   （便宜）
step2: forward → t2     verify: target forward 一次     （一次昂贵）
step3: forward → t3             接受 d1 d2 d3 + bonus b
...每步 1 token         → 一步产出 4 个 token（3 draft + 1 bonus）
```

### 1.2 两个关键量：draft 便宜 & 接受率高

投机解码的收益 = `接受长度 τ` × `(1 − draft开销/target开销)`。因此所有方案都在两个维度上做文章：

| 维度 | 目标 | SGLang 的手段 |
|---|---|---|
| **draft 便宜** | 草稿开销远小于 target | EAGLE 单层头复用 target hidden；NGRAM 无模型；FROZEN_KV_MTP 不建 draft KV；DFLASH 从 target hidden 投影 KV |
| **接受率高** | draft 分布贴近 target | EAGLE 复用 target 最后一层 hidden；树形（topk>1）一次提多条候选路径取最优；自适应动态调 num_steps |

### 1.3 SGLang 的统一抽象

SGLang 把"如何产生草稿"抽象成可插拔的**算法（algorithm）**，把"如何调度/验证/记账"抽象成**统一的 V2 框架**。所有算法都产出/消费统一的 `SpecInput`，都跑同一套 `draft → verify → draft_extend` 三阶段循环，都通过统一的 KV 预算公式与注意力后端接口接入。

```
                    ┌─────────────────────────────────────────┐
                    │            Scheduler (单线程)              │
                    │  event_loop_overlap / event_loop_normal   │
                    └───────────────────┬───────────────────────┘
                                        │ run_batch
                                        ▼
              ┌──────────────────────────────────────────────────┐
              │        BaseSpecWorker (统一 V2 worker 接口)         │
              │   forward_batch_generation(batch, on_publish)      │
              └───────┬───────────────┬──────────────────┬─────────┘
                      │               │                  │
         create_worker() 分发    target_worker      draft_worker
                      │          (TpModelWorker,     (算法相关)
                      ▼           跑 target 模型)
   ┌──────────┬───────────┬──────────┬──────────┬──────────┬──────────────┐
   │ EAGLE/   │ STANDALONE│  NGRAM   │  DFLASH  │  DSPARK  │ FROZEN_KV_MTP │
   │ EAGLE3   │           │          │          │          │               │
   │ (+MTP别名)│(独立小模型)│(无模型)  │(块级linear)│(块+SPS/STS)│ (读target KV) │
   └──────────┴───────────┴──────────┴──────────┴──────────┴──────────────┘
        │                                              也可 +Multi-Layer / +自适应
        └── EAGLEWorkerV2 (eagle_worker_v2.py:EAGLEWorkerV2) 是 EAGLE 系家族基类
```

**关键文件**：
- [`spec_info.py:SpeculativeAlgorithm`](../../../python/sglang/srt/speculative/spec_info.py) — 算法枚举 + `is_*()` / `supports_*()` 谓词 + `create_worker()` 分发。
- [`base_spec_worker.py:BaseSpecWorker`](../../../python/sglang/srt/speculative/base_spec_worker.py) — worker 契约；同文件 `base_spec_worker.py:EagleDraftWorkerBase` 是 draft 侧契约。
- [`eagle_worker_common.py`](../../../python/sglang/srt/speculative/eagle_worker_common.py) — EAGLE 系**公共实现**：`prepare_for_draft()` / `prepare_for_draft_extend()` / `build_eagle_verify_input()` / `run_eagle_verify()`。⚠️ 这些函数**曾经**在 `base_spec_worker.py` / `eagle_worker_v2.py` 里，现已抽到此文件。
- `spec_info.py:SpecInput` + `spec_info.py:SpecInputType` — 所有算法草稿 / 验证载荷的基类与相位枚举。

---

## 2. 算法家族与分发机制

### 2.1 枚举成员（`spec_info.py:SpeculativeAlgorithm`）

```python
class SpeculativeAlgorithm(Enum):
    DFLASH = auto()          # 块级 linear 草稿，KV 从 target hidden 投影
    DSPARK = auto()          # DFLASH 的块级近亲：+SPS/STS 表、块接受估计、ragged verify
    EAGLE = auto()           # EAGLE-1；也是 NEXTN/MTP 的落点
    EAGLE3 = auto()          # EAGLE-3（多层 aux hidden + hot-token 缩表）
    FROZEN_KV_MTP = auto()   # 读 target KV 只读、不建 draft KV 的 MTP
    STANDALONE = auto()      # 独立小模型（经典 Leviathan 式）
    NGRAM = auto()           # 无模型，n-gram/后缀自动机匹配
    NONE = auto()            # 关闭
```

注意：**没有 NEXTN / MULTI_LAYER 成员**。NEXTN 是 CLI 保留别名（`spec_registry.py` 的模块级常量 `_RESERVED_ALIASES = frozenset({"NEXTN"})`），Multi-Layer 是 EAGLE 上的一个 `--enable-multi-layer-eagle` 开关。

### 2.2 谓词接口（duck-typing 契约）

枚举成员与插件（`CustomSpecAlgo`）共享同一套 `is_*()` / `supports_*()` 谓词，调用方**不用 isinstance** 就能统一分发。这也是"新增能力先加谓词、再改调用方"的唯一正确姿势。关键谓词（全部定义在 `spec_info.py:SpeculativeAlgorithm`）：

| 谓词 | 含义 | 谁为 True |
|---|---|---|
| `is_eagle()` | EAGLE 家族 | EAGLE, EAGLE3, **FROZEN_KV_MTP**（有 FIXME） |
| `is_eagle3()` | EAGLE3 | EAGLE3 |
| `is_frozen_kv_mtp()` | 冻结 KV MTP | FROZEN_KV_MTP |
| `is_dflash()` | DFlash 本体 | DFLASH |
| `is_dspark()` | DSpark 本体 | DSPARK |
| `is_dflash_family()` | 块级草稿家族 | DFLASH, DSPARK |
| `is_standalone()` | 独立草稿模型 | STANDALONE |
| `is_ngram()` | n-gram 匹配 | NGRAM |
| `need_topk()` | 需要 topk 树 | eagle 或 standalone |
| `has_draft_kv()` | draft 阶段写 KV 链 | **除 NGRAM 外都 True** |
| `carries_draft_hidden_states()` | PD 分离时传 draft hidden | **仅 `is_eagle()`** |
| `supports_target_verify_for_draft()` | draft worker 侧也要按 verify 宽度建图 | 见实现 |
| `supports_mixed_chunk()` | 允许 `--enable-mixed-chunk` | EAGLE, EAGLE3, DFLASH, DSPARK |
| `supports_grammar_overlap()` | grammar 下**仍可 overlap** | `is_eagle() or is_standalone() or is_dflash_family()` ⇒ 实际上**除 NGRAM 外全 True** |
| `supports_ragged_verify()` | verify 宽度可逐请求变长 | **仅 DSPARK** |

`has_draft_kv()` 影响 KV 预算（NGRAM 的树只活在 verify mask 里，不需要 per-topk 分页取整）；`carries_draft_hidden_states()` 影响 PD 分离时的传输载荷；`supports_mixed_chunk()` / `supports_grammar_overlap()` 分别是参数钩子关 mixed chunk（§5.3）与调度器是否降级 overlap（§19.2）的唯一判据。

同一个枚举上还挂着几个**非谓词**的策略方法，扩展算法时同样要实现：

| 方法 | 作用 |
|---|---|
| `handle_server_args()` | 派发到各 `_handle_*` 参数钩子（§5） |
| `resolve_max_speculative_num_draft_tokens()` | 给出 KV 预算用的 `num_draft_tokens` 上界（自适应时取 `max(candidate_steps)+1`，非自适应就是 `num_draft_tokens`） |
| `get_num_tokens_per_req_for_target_verify()` | target verify 相位的每请求 token 宽度（DSPARK 的 draft worker 侧返回 `num_draft_tokens - 1`）。旧名 `get_num_tokens_per_bs_for_target_verify()` 仍在，但已挂 `DeprecationWarning`，只是薄转发 |
| `create_future_map()` / `build_disagg_draft_input()` | overlap future map、PD 分离草稿输入 |

⚠️ 有两个常被误当成"枚举方法"的东西其实是 `spec_info.py` 的**模块级函数**：`spec_info.py:spec_scale_global_num_tokens()`（按 `SpecInput.num_tokens_per_req` / `num_tokens_for_logprob_per_req` 把 DP 的 per-rank 请求数放缩成 token 数——它**取代了已删除的 `EagleVerifyInput.get_spec_adjust_token_coefficient`**）与 `spec_info.py:create_dummy_verify_input()`（CUDA graph 捕获用的占位 verify 输入）。

### 2.3 worker 分发（`spec_info.py:SpeculativeAlgorithm.create_worker()`）

```
DFLASH          → DFlashWorkerV2              (dflash_worker_v2.py)
DSPARK          → DSparkWorkerV2              (dspark_components/dspark_worker_v2.py)
FROZEN_KV_MTP   → FrozenKVMTPWorkerV2         (frozen_kv_mtp_worker_v2.py)
EAGLE/EAGLE3 且 --enable-multi-layer-eagle
                → MultiLayerEagleWorkerV2     (multi_layer_eagle_worker_v2.py)
EAGLE/EAGLE3    → EAGLEWorkerV2               (eagle_worker_v2.py)   ← EAGLE 系基类
STANDALONE      → StandaloneWorkerV2          (standalone_worker_v2.py, 继承 EAGLEWorkerV2)
NGRAM           → NGRAMWorker                 (ngram_worker.py)
```

`StandaloneWorkerV2` 继承 `EAGLEWorkerV2`，`FrozenKVMTPWorkerV2` 也继承它但**不调用其 `__init__`**（避免建 draft KV 池）——可见 EAGLE 是这一支的实现基石。但注意**不是所有 worker 都在这条继承链上**：`MultiLayerEagleWorkerV2`、`DFlashWorkerV2`、`DSparkWorkerV2`、`NGRAMWorker` 都直接继承 `BaseSpecWorker`。

### 2.4 插件机制（`spec_registry.py`）

第三方可用 `@SpeculativeAlgorithm.register("MY_SPEC", ...)`（`spec_info.py:SpeculativeAlgorithm.register()`）注册自定义算法，工厂返回 worker 类。`spec_registry.py:_assert_custom_spec_algo_conforms()` 会在注册时校验插件实现了全部 `is_*` / `supports_*` 方法，否则 fail-fast（其 docstring 记载历史上 `is_some` 与 `is_frozen_kv_mtp` 都曾静默缺失）。`supports_overlap=False` 已弃用：V1 路径删除后，这类算法在 V2 schema 上以 overlap-disabled（同步）方式运行。

---

## 3. 快速上手

所有姿势都在标准 `python -m sglang.launch_server` 上加 `--speculative-*` 参数即可，overlap 调度默认开启。

### 3.1 EAGLE / EAGLE3（外挂 draft 头）

```bash
# EAGLE3 + Llama3-8B，树形草稿（topk=8）
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <eagle3-head-ckpt> \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 8 \
    --speculative-num-draft-tokens 8
```

### 3.2 MTP / NEXTN（draft 头内建于 target 权重）

```bash
# DeepSeek MTP：NEXTN 自动解析为 EAGLE，draft path 自动设为 model_path，无需单独下载
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
# 甚至可全部省略 num-steps/topk/num-draft-tokens，让 _auto_choose_speculative_params 推导
```

### 3.3 NGRAM（无模型，prompt-lookup）

```bash
# 适合大量重复/长上下文回填场景（代码补全、RAG）
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-algorithm NGRAM \
    --speculative-num-draft-tokens 12
    # topk/num-steps 由 ngram 参数派生；可选 --speculative-ngram-external-corpus-path 预热
```

### 3.4 STANDALONE（独立小模型）

```bash
# 经典 Leviathan 式：draft 是一个独立的小模型（须与 target 同 vocab）
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

### 3.5 DFLASH / DSPARK（块级草稿）

```bash
# DFLASH：一次前向提出一整块草稿；block_size 就是 num_draft_tokens 的别名
python -m sglang.launch_server \
    --model-path <target> \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path <dflash-draft-ckpt> \
    --speculative-dflash-block-size 16
```

```bash
# DSPARK：块级草稿 + 置信度头 + ragged verify（逐请求不同 verify 长度）
python -m sglang.launch_server \
    --model-path <target> \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path <dspark-draft-ckpt> \
    --speculative-dspark-block-size 7            # gamma；verify 窗口 = gamma+1
# 想真正省算力，还要开 ragged verify 并喂一张离线 profile 的 SPS 表：
#   export SGLANG_RAGGED_VERIFY_MODE=compact
#   --speculative-dspark-sps-table-path <sps.json>
#   （可选）--speculative-dspark-confidence-sts-path <sts.json>
# 注意：draft 权重若已打包在 target checkpoint 里，可省略 --speculative-draft-model-path。
```

### 3.6 自适应（动态 num_steps）

```bash
python -m sglang.launch_server --model-path ... \
    --speculative-algorithm EAGLE --speculative-eagle-topk 1 \
    --speculative-adaptive --speculative-adaptive-config <config.json>
```

---

## 4. CLI 参数全表

参数字段全部声明在 `server_args.py:ServerArgs` 的 `NS("spec")` 命名空间段落里（`speculative_*` / `decoupled_spec_*` / `spec_trace_dir`）；钩子由参数解析流水线 [`arg_groups/pipeline.py:run_resolution_pipeline()`](../../../python/sglang/srt/arg_groups/pipeline.py) 调用 [`arg_groups/speculative_hook.py:handle_speculative_decoding()`](../../../python/sglang/srt/arg_groups/speculative_hook.py) 触发。⚠️ 老版本文档里"`server_args.py` 上的 `handle_speculative_decoding` 方法"已不存在——它早已搬到 `arg_groups/`。

### 4.1 通用参数

| Flag | 字段 | 默认 | 含义 |
|---|---|---|---|
| `--speculative-algorithm` | `speculative_algorithm` | `None` | EAGLE / EAGLE3 / NEXTN / STANDALONE / NGRAM / DFLASH / **DSPARK** / FROZEN_KV_MTP / 插件名。⚠️ help 文本只列到 DSPARK，**没写 FROZEN_KV_MTP**——它主要靠 NEXTN 别名自动解析进去（§5.1），显式写 `FROZEN_KV_MTP` 也能用 |
| `--speculative-draft-model-path`（别名 `--speculative-draft-model`） | `speculative_draft_model_path` | `None` | draft / EAGLE 头 checkpoint；MTP、以及"draft 权重已打包进 target checkpoint"的 DSPARK 会自动设为 `model_path` |
| `--speculative-draft-model-revision` | `speculative_draft_model_revision` | `None`→`"main"` | 钩子里补 `"main"` |
| `--speculative-draft-load-format` | `speculative_draft_load_format` | `None` | draft 权重加载格式 |
| `--speculative-draft-model-quantization` | `speculative_draft_model_quantization` | `None` | draft 量化 |
| `--speculative-draft-kv-cache-dtype` | `speculative_draft_kv_cache_dtype` | `None`→随 `--kv-cache-dtype` | **新增**。只作用于 draft KV 池。draft 池按"每个 target token 一个 slot"分配（draft/target 共享 slot 索引空间），所以即使 draft 很小也可能与 target 池同量级（5 层 DFLASH draft，bf16 约 10240 B/token）。设 `fp8_e4m3` 可腰斩 draft 池；省下的是**空闲显存**，要再抬 `--mem-fraction-static` 才能兑换成 KV 容量 |
| `--speculative-num-steps` | `speculative_num_steps` | `None` | draft 循环步数（草稿深度） |
| `--speculative-eagle-topk` | `speculative_eagle_topk` | `None` | 每步分支数；`1`=chain，`>1`=tree |
| `--speculative-num-draft-tokens` | `speculative_num_draft_tokens` | `None` | 每请求每轮送 verify 的 token 数（树节点数） |
| `--speculative-token-map` | `speculative_token_map` | `None` | EAGLE3 FR-Spec 的热词子表映射 |
| `--speculative-attention-mode` | `speculative_attention_mode` | `"prefill"` | `prefill`/`decode`——决定 draft-extend 后端取哪个 |
| `--speculative-draft-attention-backend` | `speculative_draft_attention_backend` | `None` | 覆写 draft 后端 |
| `--speculative-dsa-topk-backend` | `speculative_dsa_topk_backend` | `"sgl-kernel"` | **新增**。draft worker 的 DSA indexer top-k 后端，可选 `sgl-kernel` / `torch` / `flashinfer`；`torch` 目前要求 `SGLANG_DSA_FUSE_TOPK=false` |
| `--speculative-draft-window-size` | `speculative_draft_window_size` | `None` | draft 侧滑窗。**仅** Llama EAGLE-3（`LlamaForCausalLMEagle3`）与 DFLASH 真正生效，其他 EAGLE-3 drafter（如 MLA 系）静默忽略。⚠️ 旧名 `--speculative-dflash-draft-window-size` 已弃用，仅作为 `dest=speculative_draft_window_size` 的兼容别名保留 |
| `--speculative-moe-runner-backend` | `speculative_moe_runner_backend` | `None`→随 `moe_runner_backend` | draft MoE runner |
| `--speculative-moe-a2a-backend` | `speculative_moe_a2a_backend` | `None` | draft MoE all2all。⚠️ DSPARK + dp attention 下**被忽略**，且若显式设成与 target 不同会直接 raise |
| `--speculative-skip-dp-mlp-sync` | `speculative_skip_dp_mlp_sync` | `False` | 跳过 draft DP MLP 同步（校验要求算法是 EAGLE 系） |
| `--spec-trace-dir` | `spec_trace_dir` | `None` | **新增**。分离式（decoupled）投机的 trace 落盘目录（§18） |

### 4.2 接受阈值 / 采样

| Flag | 字段 | 默认 | 含义 |
|---|---|---|---|
| `--speculative-use-rejection-sampling` | `speculative_use_rejection_sampling` | `False` | 拒绝采样 vs 贪心树验证 |
| `--speculative-accept-threshold-single` | `speculative_accept_threshold_single` | `1.0` | 单 token 接受概率阈值（放宽接受，牺牲严格等价换吞吐） |
| `--speculative-accept-threshold-acc` | `speculative_accept_threshold_acc` | `1.0` | 累积接受阈值 |

> `1.0` = 严格无损（输出分布 = target 单步）。调低这两个阈值可提高接受率/吞吐，但输出不再与 target 严格等价。拒绝采样路径要求两个阈值都为 `1.0`（它自带无损保证），且要求 `topk==1`、非确定性推理、非 multi-layer。

### 4.3 算法专属

| Flag | 字段 | 默认 | 归属 |
|---|---|---|---|
| `--enable-multi-layer-eagle` | `enable_multi_layer_eagle` | `False` | EAGLE 多层变体 |
| `--speculative-adaptive` | `speculative_adaptive` | `False` | 自适应步数（仅 EAGLE/EAGLE3 + topk==1） |
| `--speculative-adaptive-config` | `speculative_adaptive_config` | `None` | 自适应配置 JSON |
| `--speculative-dflash-block-size` | `speculative_dflash_block_size` | `None` | DFLASH 块大小（= `num_draft_tokens` 别名） |
| `--speculative-dspark-block-size` | `speculative_dspark_block_size` | `None` | DSPARK 的草稿块长 γ（gamma）。verify 窗口 = γ+1，因此它等价于设 `num_draft_tokens = γ+1`；不给则先从 draft checkpoint 的 `block_size` 推，再退到 `DEFAULT_DSPARK_GAMMA = 7` |
| `--speculative-dspark-sps-table-path` | `speculative_dspark_sps_table_path` | `None` | DSPARK 的离线 SPS（steps-per-second）代价表 JSON，由 `sglang.benchmark.dspark_sps_profiler` 产出，喂给 ragged-verify 预算调度器。不给 = 平坦常数表 ⇒ 预算退化成"全验"，本身不带来收益 |
| `--speculative-dspark-confidence-sts-path` | `speculative_dspark_confidence_sts_path` | `None` | DSPARK 的 per-position STS（sequential temperature scaling）标定 JSON，由 `sglang.benchmark.dspark_sts_fit` 拟合，用来校准置信度头给出的存活概率。不给 = 恒等（不校准）；**有无都不影响无损性** |
| `--speculative-dspark-align-verify-tokens-to-graph-tier` | `speculative_dspark_align_verify_tokens_to_graph_tier` | `False` | DSPARK compact ragged-verify 专用。把 per-request verify 长度补到"forward 反正要 pad 到的那个 CUDA graph 档位"，把 graph 档位取整 + DP 跨 rank 取 max 这两份本来白付的 padding 变成真实验证量。只在 `SGLANG_RAGGED_VERIFY_MODE=compact` 下生效，否则打 warning 并空转 |
| `--speculative-ngram-*` | 见下 | 见下 | NGRAM 匹配参数 |

NGRAM 参数（`server_args.py:ServerArgs` 的 "Speculative decoding (ngram)" 段）：`ngram_min_bfs_breadth`=1、`ngram_max_bfs_breadth`=10、`ngram_match_type`="BFS"、`ngram_max_trie_depth`=18、`ngram_capacity`=10M、`ngram_external_corpus_path`=None、`ngram_external_sam_budget`=0、`ngram_external_corpus_max_tokens`=10M。

分离式（decoupled）参数（同文件 `NS("disagg")` 段）：`--decoupled-spec-role {null,verifier,drafter}`、`--decoupled-spec-bind-endpoint`、`--decoupled-spec-connect-endpoints`、`--decoupled-spec-rank`，再加 `NS("spec")` 里的 `--spec-trace-dir`。

DSPARK 还有一批**环境变量**开关（`environ.py`，`SGLANG_DSPARK_*` 约 20 个）用于调试与性能实验：`SGLANG_DSPARK_FAST_KERNEL`（默认 True）、`SGLANG_DSPARK_FOLDED_SAMPLING` / `SGLANG_DSPARK_FOLDED_PROPOSAL`、`SGLANG_DSPARK_EMBED_IN_GRAPH`、`SGLANG_DSPARK_ENABLE_MULTI_STREAM`、`SGLANG_DSPARK_DEBUG_DUMP`、`SGLANG_DSPARK_STS_COLLECT_PATH`、`SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH` 等。ragged verify 的总闸是 `SGLANG_RAGGED_VERIFY_MODE`（默认 `"static"`，见 §14.6）。

> `max_speculative_num_draft_tokens`：这是 KV 预算的口径上界。它不是单点函数，而是一条链——`spec_info.py:SpeculativeAlgorithm.resolve_max_speculative_num_draft_tokens()` 给出算法侧的上界（自适应时取 `max(candidate_steps)+1`，普通情况就是 `num_draft_tokens`），再由 `arg_groups/overrides.py` 与 `runtime_context.py` 里的同名访问器暴露给调用方。§20 的所有预算公式都按这个上界预留，**不是按当前生效的 `num_draft_tokens`**。

---

## 5. 参数钩子

`arg_groups/speculative_hook.py` 是"填默认 + 校验 + 派生"的中枢。

### 5.1 别名解析（`speculative_hook.py:_resolve_speculative_algorithm_alias()`）

```
NEXTN / EAGLE ──┬── draft 架构是 Gemma4Assistant* / Gemma4UnifiedAssistant* ──→ FROZEN_KV_MTP
                └── 否则 ──────────────────────────────────────────────────→ EAGLE
EAGLE3 + Gemma4 draft ──→ raise ValueError（不支持）
```

**这是 NEXTN 塌缩到 EAGLE 的唯一入口**。枚举无 NEXTN 成员，它只在 `spec_registry.py` 的模块级常量 `_RESERVED_ALIASES` 里被保留（防插件占用）。

### 5.2 统一入口（`speculative_hook.py:handle_speculative_decoding()`）

顺序做这几件事：

1. 有 draft path 而没 revision → 补 `speculative_draft_model_revision="main"`；
2. 跑 `overrides.py:_speculative_moe_runner_default`，让 `speculative_moe_runner_backend` 默认随 `moe_runner_backend`；
3. 算法名大写化；
4. 若检测到已移除的 `SGLANG_ENABLE_SPEC_V2` 环境变量则打 warning（这里刻意用裸 `os.getenv`，因为 `environ.py` 里的描述符已删）；
5. 调 `_resolve_speculative_algorithm_alias()` 解析别名（§5.1）；
6. 校验 `--speculative-draft-window-size`：必须为正；算法不在 `("EAGLE3", "DFLASH")` 时打"无效果"warning——**注意这里的判据是解析后的算法名，所以 Llama EAGLE-3 之外的 EAGLE3 drafter 也不会被 warning 拦住，是静默忽略**；
7. 插件算法（`CustomSpecAlgo`）若带 `validate_server_args` 则回调；
8. `--speculative-skip-dp-mlp-sync` 断言算法必须是 `EAGLE`；
9. 若开了 `--speculative-adaptive`：先 `_maybe_disable_adaptive()`（不支持就静默降级成静态参数，见 §17），仍然开着才 `_init_adaptive_speculative_params()`；
10. 最后 `algo.handle_server_args()` → 派发到各 `_handle_*`。

### 5.3 各算法钩子

所有钩子都通过 `arg_groups` 的 `resolving_view()` / `resolved_view()` 读参数、用 `declare_resolution()` 写参数（**不直接改 `server_args` 字段**），这样解析来源可追溯。

| 钩子 | 关键动作 |
|---|---|
| `speculative_hook.py:_handle_dflash()` | 只许 CUDA / NPU 设备；禁 dp_attention；要求 `pp_size==1`；必须给 draft path；强制 `num_steps=1` / `topk=1`（不是 1 就 warning + 覆写）；`dflash_block_size ↔ num_draft_tokens` 互为别名，两个都给且不等则 raise；都没给则从 draft checkpoint 的 `block_size` 推，推不出来退到 16；`draft_window_size` 必须 ≥ `num_draft_tokens`；调 `_resolve_dflash_draft_attention_backend()` 定 draft 后端；`max_running=48`。⚠️ **它不关 mixed_chunk**——`supports_mixed_chunk()` 对 DFLASH 返回 True，旧文档"禁 mixed_chunk"是错的 |
| `speculative_hook.py:_handle_dspark()` | 见 §14.7（门禁最多的一个钩子） |
| `speculative_hook.py:_handle_frozen_kv_mtp()` | `max_running=48`；**关 mixed_chunk**（它确实不支持） |
| `speculative_hook.py:_handle_eagle_family()` | STANDALONE + dp_attention → raise；`max_running=48`；`_disable_overlap_schedule_for_cpu()`；`disable_overlap_schedule` 时打"走同步 spec v2"warning；`enable_mixed_chunk and not algo.supports_mixed_chunk()` → 关 mixed_chunk（**谓词驱动**，不再硬编码算法名）；DeepSeek V3/V3.2/V4、GLM4-MoE、Bailing、MistralLarge3、Pixtral、HYV3 等 MTP 架构默认 `draft_path=model_path`；参数全空则 `_auto_choose_speculative_params()`；`trtllm_mha` 要求 `topk==1`；拒绝采样门禁（见下）；`topk==1` 时把 `num_draft_tokens` 校正成 `num_steps+1`；`topk>1 且 page_size>1` 时只许 `_PAGE_TREE_SPEC_BACKENDS = ("flashinfer", "fa3", "triton")` |
| `speculative_hook.py:_handle_ngram()` | 设备限制是 **`cuda` 或 `cpu`**（旧文档写"仅 CUDA"是错的）；CPU 上调 `_disable_overlap_schedule_for_cpu()` 关 overlap；`max_running=48`；无条件关 mixed_chunk；`eagle_topk = ngram_max_bfs_breadth`；`num_draft_tokens` 默认 12；`num_steps = num_draft_tokens // eagle_topk`；外部语料三参数互相校验（`external_sam_budget ∈ (0, num_draft_tokens-1]`）；`topk>1 且 page_size>1` 只许 `flashinfer`；禁 dp_attention |

`_handle_eagle_family()` 里的拒绝采样门禁（`--speculative-use-rejection-sampling`）共 5 条：算法必须是 `EAGLE`/`EAGLE3`、`topk==1`、两个 accept threshold 都必须是 `1.0`（拒绝采样自带无损保证，会忽略阈值）、不能同时开 `--enable-deterministic-inference`（内核从全局 RNG 抽 coin，非 batch-invariant）、multi-layer EAGLE 下仍要求 `topk==1`。

### 5.4 参数自动推导（`speculative_hook.py:_auto_choose_speculative_params()`）

返回 `(num_steps, eagle_topk, num_draft_tokens)`：

| 模型架构 | 参数 |
|---|---|
| STANDALONE（任意） | `(3, 1, 4)` |
| Llama | `(5, 4, 8)` |
| DeepSeek/GLM/Bailing/GptOss/MiMoV2/… | `(3, 1, 4)` |
| Grok | `(5, 4, 8)` |
| 默认 | `(3, 1, 4)` |

自适应初始化（`speculative_hook.py:_init_adaptive_speculative_params()`）：`eagle_topk=1`；`num_steps` 未给则取 `candidate_steps` 的中位数；**给了但不在 `candidate_steps` 里则 raise**；最后 `num_draft_tokens = num_steps + 1`。

### 5.5 全局交叉校验（散落在 `arg_groups/*_hook.py`）

| 校验 | 位置 | 内容 |
|---|---|---|
| `pp_size > 1` 与 spec 不兼容 | `validation_hook.py:check_server_args()` | `pp_size>1` 时断言 `disable_overlap_schedule and speculative_algorithm is None` |
| DWDP 与 spec 不兼容 | `parallel_hook.py:handle_dwdp()` | 断言 `speculative_algorithm is None` |
| LoRA × spec 白名单 | `lora_hook.py:check_lora_speculative_compatibility()` | 白名单已**大幅放宽**：NGRAM 直接放行，其余只允许 `("EAGLE", "EAGLE3", "DFLASH", "DSPARK")`；FROZEN_KV_MTP 会给出"NEXTN+Gemma4 draft 被自动提升成 FROZEN_KV_MTP"的解释性报错。另外 4 条附加禁令：DSPARK 且 `SGLANG_RAGGED_VERIFY_MODE != static`（变长 verify 破坏 LoRA 的等宽 segment 布局）、`--speculative-adaptive`、`experimental_sgl_trtllm` MoE runner、`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` |
| trtllm_mha 要求 `topk==1` | `speculative_hook.py:_handle_eagle_family()` | 同 §5.3 |
| MiMoV2 自动开 multi-layer EAGLE | `arg_groups/model_overrides/mimo_v2.py:_mimo_v2_overrides()` | `speculative_algorithm == "EAGLE"` 时 `overrides["enable_multi_layer_eagle"] = True` |
| Step3p5 / Step3p7 自动开 multi-layer EAGLE | `arg_groups/overrides.py:_step3p_overrides()` | 同上 |
| MoE token 预算按 verify 宽度放大 | `arg_groups/overrides.py:cutedsl_moe_max_num_tokens()` | spec 下 `num_tokens_per_req = speculative_num_draft_tokens or 1`，否则 1 |

⚠️ 两处旧文档遗留的说法要作废：

- **"`dcp_size>1` 与 spec 不兼容"——现在没有这条门禁**。`parallel_hook.py:handle_dcp_validation()` 里不含 spec 判据，DCP × spec 不再被参数层拦下（只有 PD 分离侧 `pd_disaggregation_hook.py` 仍硬拒 spec + DCP 组合）。
- **"spec 下 `num_tokens_per_bs = num_draft_tokens`"**：`num_tokens_per_bs` 这个名字已不存在。等价概念现在是 `spec_utils.py:resolve_num_tokens_per_req(phase=...)`（运行期，见 §10）与 `overrides.py:cutedsl_moe_max_num_tokens()` 里的 `num_tokens_per_req`（参数期）。

---

## 6. 统一三阶段主循环

所有 EAGLE 家族 worker 的 [`eagle_worker_v2.py:EAGLEWorkerV2.forward_batch_generation()`](../../../python/sglang/srt/speculative/eagle_worker_v2.py) 按 batch 的 forward_mode 分两条路：

### 6.1 Extend / Prefill 路（新请求首次进来）

```
1. target prefill：target_worker.forward_batch_generation(batch)
   capture_hidden_mode = FULL（STANDALONE 用 NULL，它不看 target hidden）
2. batch_output.new_seq_lens = batch.seq_lens   # 约定：seq_lens 是本轮 token 前的长度
3. on_publish(new_seq_lens)                      # overlap 屏障：在 target-end 处发布
4. draft prefill：draft_worker._draft_extend_for_prefill(...)
   用 target 最后一层 hidden 生成第一个 EagleDraftInput（下一轮 draft 的种子）
```

### 6.2 Decode 路（已在解码的请求）——核心三阶段

```
                    ┌─────────── forward_batch_generation (decode) ────────────┐
                    │                                                            │
 batch.spec_info    │  ①activate_step_by_batch(bs)   （自适应时切换 num_steps）    │
 (上轮 EagleDraftInput)│  ②draft(batch) ───────────────→ EagleVerifyInput           │
                    │       draft_worker.draft，多步草稿 + 建树                     │
                    │  ③verify(batch) ──────────────→ batch_output(new_seq_lens)  │
                    │       target 一次 forward 验证 + eagle_sample 接受            │
                    │  ④on_publish(new_seq_lens)     （overlap 屏障：verify-end）  │
                    │  ⑤_draft_extend_for_decode ───→ next_draft_input             │
                    │       用接受 token 的 hidden 生成下一轮 EagleDraftInput        │
                    └────────────────────────────────────────────────────────────┘
```

三阶段职责：

| 阶段 | 方法 | 输入 | 输出 | 做什么 |
|---|---|---|---|---|
| **draft** | `eagle_worker_v2.py:EagleDraftWorker.draft()` | `EagleDraftInput`（topk_p/index + hidden） | `EagleVerifyInput` | 多步草稿前向 + `build_tree_kernel_efficient` 建树，产出 draft_token/mask/positions/retrieve_* |
| **verify** | `eagle_worker_v2.py:EAGLEWorkerV2.verify()` → [`eagle_worker_common.py:run_eagle_verify()`](../../../python/sglang/srt/speculative/eagle_worker_common.py) | `EagleVerifyInput` | `batch_output` | target 一次 forward 验证全部 draft token，`eagle_sample` 接受，抽 bonus token |
| **draft_extend** | `eagle_worker_v2.py:EagleDraftWorker._draft_extend_for_decode()` | 接受 token + hidden | `EagleDraftInput` | 把接受的 token 重新 embed 过 draft，产出下一轮 draft 的 hidden 与 topk 提议 |

⚠️ **`EAGLEWorkerV2.verify()` 现在只是一层薄转发**：真正的 verify 主体（plan stream、grammar mask、采样、接受记账）全在 `eagle_worker_common.py:run_eagle_verify()` 里，由两个开关参数 `metadata_ready_pre_pad` / `finalize_tree_path` 区分单层 EAGLE 与 Multi-Layer EAGLE 的历史差异。同理，`prepare_for_draft()` / `prepare_for_draft_extend()` / `build_eagle_verify_input()` / `duplicate_prefix_tail_to_draft_branches()` 也都在 `eagle_worker_common.py`，不再在 `base_spec_worker.py` 或 `eagle_worker_v2.py` 里。

**为什么需要 draft_extend？** draft 阶段要读"上一个已确认 token 的 hidden state"作为起点。verify 后新确认了若干 token，必须再跑一次 draft 模型的 extend forward，把这些新 token 的 hidden 算出来，才能启动下一轮 draft。它是把 verify 的结果"喂回" draft 的桥梁。

### 6.3 双 CUDA Stream 与 overlap

verify 内部（`eagle_worker_common.py:run_eagle_verify()`）用 `plan_stream_ctx` 把注意力 plan / 元数据准备放到**独立 plan stream**，与主 compute stream 并行，再用 `current_stream().wait_stream(plan_stream)` 汇合。由于 `custom_mask` / `positions` 依赖 draft 输出，plan stream 先用了"旧值"，汇合后再调 `attn_backend.update_verify_buffers_to_fill_after_draft(...)` 刷成真值（NPU + mrope 还要额外重算 `compute_spec_mrope_positions`）。

`on_publish` 回调是 overlap 调度的关键屏障：在 verify 结束（而非 draft_extend 结束）处发布 `new_seq_lens`，把"读共享 buffer 的最后一段"与调度器的 WAR（write-after-read）屏障对齐。draft_extend 是最后读共享 draft buffer 的阶段，由它记录 **`model_runner.shared_read_done_event`**（⚠️ 旧名 `war_fastpath_read_done_event` 已删除），worker 侧通过 `base_spec_worker.py:BaseSpecWorker.last_shared_read_runner` 暴露"哪个 runner 持有这个 event"。

### 6.4 skip-last-forward 优化

draft 循环（`eagle_worker_v2.py:EagleDraftWorker.draft_forward()`）只跑 `num_steps − 1` 次 draft 前向，而非 `num_steps` 次：

```python
for i in range(self.speculative_num_steps):
    input_ids, hidden_states, scores, tree_info = select_top_k_tokens(...)
    score_list.append(...); token_list.append(...); parents_list.append(...)
    if i == self.speculative_num_steps - 1:
        break   # "1 个 token 来自 draft prefill/extend，(num_steps-1) 个来自这里"
    ...draft_runner.forward(forward_batch)...       # 每步用 attn_backends[i]
```

第 1 个候选来自上一轮 draft_extend 的输出（已 embed 好的 hidden），循环里只需再产 `num_steps−1` 步。

### 6.5 `num_steps == 0` 旁路：草稿关掉但管线不拆

自适应投机（§17）在大 batch 下会把 `speculative_num_steps` 压到 **0**——意思是"这一步别造草稿了，直接当普通 decode 跑"。但如果真的绕开整条 spec 管线，就要维护第二套 forward / CUDA graph / 记账路径。SGLang 的做法是**造一个 1 节点的假草稿树**，让它照旧穿过 TARGET_VERIFY 图：

| 环节 | 符号 | 行为 |
|---|---|---|
| 造假树 | `eagle_worker_v2.py:EAGLEWorkerV2._build_trivial_verify_input()` | 造一棵只有根节点（= 上一轮的 bonus token）的 `EagleVerifyInput`。接受内核必然接受根节点、并从 target logits 采出 1 个新 bonus token——**功能上等价于一次普通 decode**，但复用的是已经按 `draft_token_num=1` 捕获好的 TARGET_VERIFY 图 |
| 跳过 draft | `forward_batch_generation()` 里 `if self.speculative_num_steps == 0:` | 直接用假树，不进 `draft()` |
| 是否还跑 draft_extend | 环境变量 `SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND`（默认 `False`） | **默认仍跑 draft_extend**——目的是"保温"：让 draft KV 链继续跟上 target，等 batch 缩小、`num_steps` 又被抬回来时能立刻接着草稿，不用重新回填 |
| 打开开关后 | `eagle_worker_v2.py:EAGLEWorkerV2._stub_skipped_draft_extend()` | 省掉 draft_extend 的那次前向，改为塞一个占位的 `EagleDraftInput`。省时间，代价是 draft KV 变冷，`num_steps` 恢复后头几步接受率会掉 |


---

## 7. 核心数据结构

三个 `SpecInput` 子类在三阶段间流转（`eagle_info.py`，仍是 `@dataclass`，属历史遗留，新代码用 `msgspec.Struct`）：

### 7.1 EagleDraftInput（draft 的输入/状态，`eagle_info.py:EagleDraftInput`）

跨轮携带的 draft 状态。关键字段：

| 字段 | 形状 | 含义 |
|---|---|---|
| `topk_p` / `topk_index` | `(b, topk)` | 上一步的 topk 概率与 token id（下一步草稿的起点） |
| `hidden_states` | `(b, hidden)` | 每请求一个 hidden，`draft` forward 消费；STANDALONE 为 None（不看 target hidden） |
| `draft_probs` | `(b, vocab)` | 仅拒绝采样：单步 draft 提议分布 q |
| `bonus_tokens` | `(b,)` | 每请求的 "+1" bonus token，draft_extend 后拷进来作下轮树根 |
| `future_indices` | — | 仅 V2 overlap：用 req_pool_indices 作 buffer slot key |

`eagle_info.py:EagleDraftInput.create_idle_input()` 造零工作量的 idle 桩；`filter_batch()` / `merge_batch()` 管批次切分/合并，`merge_batch` 用 `len(topk_index)==0` 识别 idle 桩。

### 7.2 EagleVerifyInput（draft 产出、verify 消费，`eagle_info.py:EagleVerifyInput`）

承载整棵草稿树的验证元数据：

| 字段 | 含义 |
|---|---|
| `draft_token` | 展平的树节点 token（含 bonus 树根） |
| `custom_mask` | 树形注意力掩码（哪个节点能看到哪个祖先） |
| `positions` | 各节点的 position id |
| `retrieve_index` | 各请求的树节点在展平数组中的索引 |
| `retrieve_next_token` / `retrieve_next_sibling` | 树拓扑（first-child / next-sibling 链式编码） |
| `spec_steps` / `topk` / `draft_token_num` | 树形状元数据 |
| `draft_probs` | 仅拒绝采样：堆叠的各步 q，形状 `(bs, num_steps, vocab)` |
| `num_tokens_per_req` | 本轮每请求的统一 token 宽度，`-1` 时在 `__post_init__` 里自动填成 `draft_token_num`；`num_tokens_for_logprob_per_req` 同样在那里填 |

- `eagle_info.py:EagleVerifyInput.max_tree_depth`（property）= `spec_steps + 1`，界定 accept_index 行宽；
- `eagle_info.py:EagleVerifyInput.tree_topk`（property）= `topk`，`-1` 表示不规则树（NGRAM）；
- ⚠️ **`get_spec_adjust_token_coefficient` 已删除**。DP 场景放缩 global token 数现在由模块级函数 `spec_info.py:spec_scale_global_num_tokens()` 完成，它直接读上表最后一行的 `num_tokens_per_req` / `num_tokens_for_logprob_per_req`。

### 7.3 EagleDraftExtendInput（verify 产出、draft_extend 消费，`eagle_info.py:EagleDraftExtendInput`）

| 字段 | 含义 |
|---|---|
| `hidden_states` | target hidden（供 draft-extend 前向）；STANDALONE 为 None |
| `num_correct_drafts` / `num_accept_tokens` | 每请求接受计数，二者关系见下 |
| `input_ids` | 各请求接受的 token（展平） |
| `seq_lens` / `req_pool_indices` / `positions` | draft-extend 前向的批次切片 |
| `bonus_tokens` | `(bs,)`，extend 后读出填入下轮 `EagleDraftInput.bonus_tokens` |

`EagleDraftExtendInput` 的字段注释是接受长度语义的权威出处：

> **`num_accept_tokens = num_correct_drafts + 1`**（`+1` 是 bonus token）。

### 7.4 SpecInput 基类与相位守卫（`spec_info.py:SpecInput` / `spec_info.py:SpecInputType`）

```python
class SpecInputType(IntEnum):
    EAGLE_DRAFT / EAGLE_DRAFT_EXTEND / EAGLE_VERIFY
    FROZEN_KV_MTP_DRAFT / FROZEN_KV_MTP_VERIFY
    DFLASH_DRAFT / DFLASH_VERIFY      # DSPARK 复用 DFLASH 这两个相位
    NGRAM_VERIFY
```

`spec_info.py:SpecInput.is_draft_input()` / `is_verify_input()` 让注意力后端与 ForwardBatch padding 逻辑按**相位**分发，不必硬编码具体算法类。相应 forward_mode 定义在 `forward_batch_info.py:ForwardMode`：`TARGET_VERIFY`、`DRAFT_EXTEND_V2`。

`SpecInput` 基类上还有三组**类级默认字段**（必须是类级默认而非 `__init__` 赋值：`@dataclass` 子类把它们声明成 field，`__post_init__ → super().__init__()` 在字段赋值之后才跑，init 期赋默认会把传进来的值冲掉）：

| 字段 | 用途 |
|---|---|
| `ragged_verify_layout` | ragged verify 的 per-request verify 长度布局（`ragged_verify.py:RaggedVerifyLayout`）。只有 `supports_ragged_verify()` 为真的算法（目前只有 DSPARK）会逐步覆写它，见 §14.6 |
| `num_tokens_per_req` / `num_tokens_for_logprob_per_req` | 本轮统一的每请求 token 宽度，兼作 DP attention 的 `global_num_tokens` 乘子（ragged forward 在那里填 1）；`-1` = 本流程未设置 |
| `dsa_topk_indices` / `future_dsa_topk_indices_available` / `dsa_seed_topk_capture` | DSA MTP IndexShare 的 seed 中继，让 scheduler / relay / attention 能在任意 `SpecInput` 上统一读取（只有 EAGLE 系输入会覆写） |

---

## 8. EAGLE/EAGLE3 详解

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）的核心：draft "模型" 是一个**轻量单层头**，它以 target 模型**最后一层的 hidden state** + 已确认 token 的 embedding 作为输入，预测下一个 token。因为复用了 target 的深层语义表征，接受率远高于独立小模型。

- **EAGLE-1/EAGLE**：draft 头读 target 最后一层 hidden。
- **EAGLE-3**：draft 头读 target **多层 aux hidden** 的拼接（[`eagle_utils.py:get_draft_input_from_target_hidden_dim()`](../../../python/sglang/srt/speculative/eagle_utils.py)），并可用 hot-token 缩表（`--speculative-token-map`，FR-Spec）把 draft 的 lm_head 词表裁到高频子集，进一步提速。
- **MTP/NEXTN**：draft 头是 target 权重内部切出来的 MTP 层（`num_nextn_predict_layers`），无需单独 ckpt。运行时仍走 EAGLE worker。

### 8.1 draft_worker 结构（`eagle_worker_v2.py:EagleDraftWorker`）

⚠️ 类头是 `class EagleDraftWorker(EagleDraftWorkerBase)`——基类在
[`base_spec_worker.py:EagleDraftWorkerBase`](../../../python/sglang/srt/speculative/base_spec_worker.py)，
topk==1 链式快路的 buffer 预分配方法也已经搬到基类上了。

| 方法（符号） | 所在文件 | 职责 |
|---|---|---|
| `EagleDraftWorker.__init__` | `eagle_worker_v2.py` | 存 topk / num_steps；拒绝采样时断言 `topk==1`（拒绝采样是 chain-only） |
| `EagleDraftWorkerBase._rebuild_topk1_chain_buffers` | `base_spec_worker.py` | 预分配 topk==1 链式快路的 `_topk1_parents_prealloc` / `_topk1_score_indices_prealloc`，消除逐步分配 |
| `EagleDraftWorker.init_lm_head` | `eagle_worker_v2.py` | EAGLE3 共享 target embedding；非 EAGLE3 把 lm_head 裁到 hot-token 词表 |
| `EagleDraftWorker.init_attention_backend` | `eagle_worker_v2.py` | 建 `draft_attn_backend`（多步 decode）与 `draft_extend_attn_backend` |
| `EagleDraftWorker._capture_cuda_graphs` | `eagle_worker_v2.py` | `num_steps>1` 才捕获 draft-decode 图；draft-extend 图总是捕获 |
| `EagleDraftWorker.draft` | `eagle_worker_v2.py` | 主入口：`prepare_for_draft` 分配 KV 位置 → `draft_forward` → `build_eagle_verify_input` 建树 → 返回 `EagleVerifyInput` |
| `EagleDraftWorker.draft_forward` | `eagle_worker_v2.py` | 多步草稿循环（见 §6.4） |
| `EagleDraftWorker.draft_extend` / `_draft_extend_for_prefill` / `_draft_extend_for_decode` | `eagle_worker_v2.py` | 两条 draft_extend 路（prefill 首次进来 vs verify 之后回灌） |

注意 `draft()` 的**头尾两段已经抽到公共模块** [`eagle_worker_common.py`](../../../python/sglang/srt/speculative/eagle_worker_common.py)：
`prepare_for_draft()`（KV 位置分配 + 造 ForwardBatch）与 `build_eagle_verify_input()`（建树 + 组装
`EagleVerifyInput`），这样 Multi-Layer EAGLE 等变体能直接复用。

### 8.2 chain vs tree（由 topk 决定）

**topk==1（链式）**：每步只取 1 个 token，拼成一条长度 `num_steps+1` 的链（含 bonus 树根），
走 `_rebuild_topk1_chain_buffers` 预分配的常量 parent/score-index，验证用链式采样。

**topk>1（树形）**：每步分支 topk 路，用**累积乘积**打分挑最优分支：

```python
# spec_utils.py:_select_top_k_tokens_later()
expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)  # (b, topk, topk)
topk_cs_p, topk_cs_index = fast_topk(expand_scores.flatten(1), topk, -1)  # 保留最优 topk 条
```

即每条已有分支再展开 topk 个孩子，`topk×topk` 个候选里按路径累积概率取前 topk 条延续
（入口是 [`spec_utils.py:select_top_k_tokens()`](../../../python/sglang/srt/speculative/spec_utils.py)，
第 0 步单独处理、后续步走 `_select_top_k_tokens_later()`）。最后
`eagle_utils.py:organize_draft_results()` 对 `score_list` 做 `topk(num_draft_tokens−1)` + 排序 + gather，
得到最终树节点。

### 8.3 topk==1 快路：绕开 `select_top_k_tokens`

`draft_forward` 里有一条专门给链式草稿开的**零拷贝快路**，条件（全部满足才启用）：

```python
# eagle_worker_v2.py:EagleDraftWorker.draft_forward()
topk1_chain_fits = self.topk == 1 and topk_index.shape[0] <= self._topk1_parents_prealloc.shape[0]
if topk1_chain_fits and _is_cuda and self.hot_token_id is None
        and not get_spec().speculative_use_rejection_sampling:
    draft_tokens_topk1 = torch.empty((bs, num_steps), ...)   # 一次性开好整条链
```

| 条件 | 为什么 |
|---|---|
| `topk == 1` | 快路只处理链式拓扑 |
| bs 不超过预分配 buffer 的行数 | 超了就退回逐步 `token_list` + 一次 `cat` |
| CUDA 设备 | kernel 只有 CUDA 实现 |
| `hot_token_id is None` | 需要 hot-token 映射时 `topk_index` 还要再过一次查表，不能直写 |
| 未开拒绝采样 | 拒绝采样要逐步留下 `draft_probs` |

启用后的行为差异：

- 循环里 `input_ids = topk_index.flatten()`，**完全不调 `select_top_k_tokens`**，也不再往
  `score_list` / `token_list` / `parents_list` 里 append；
- 每步采样换成 [`kernels/ops/speculative/topk1.py:draft_topk1_postprocess()`](../../../python/sglang/kernels/ops/speculative/topk1.py)，
  它把 argmax 结果**直接写进 `draft_tokens_topk1` 的第 `i+1` 列**，并顺手把
  `forward_batch.positions` 自增——argmax + 写列 + 推进 position 融合成一个 kernel；
- 收尾直接返回预分配的 `parent_list` / `top_scores_index` 和 `draft_tokens_topk1`，
  **不调 `organize_draft_results`**。

三条返回路径的层级关系（从快到慢）：直写快路 → `topk1_chain_fits` 但不满足其余条件（`torch.cat(token_list)`）
→ 通用树路（`organize_draft_results`）。

### 8.4 建树（`eagle_utils.py:build_tree_kernel_efficient()`）

```python
draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()
```

**bonus token 被 prepend 成树根**——上一轮 verify 产出的 bonus token 是本轮树的根节点。随后调 sgl_kernel 的 `sgl_build_tree_kernel_efficient`（NPU 有变体），产出：

- `tree_mask`：树形注意力掩码，模式有 `FULL_MASK` / `QLEN_ONLY` / `QLEN_ONLY_BITPACKING`（`eagle_utils.py:TreeMaskMode`，默认由 `eagle_utils.py:default_tree_mask_mode()` 给）；
- `positions`：各节点 position；
- `retrieve_index` / `retrieve_next_token` / `retrieve_next_sibling`：以 first-child + next-sibling 编码树拓扑，供接受时遍历。

有个避免 D2H 同步的技巧，在
[`eagle_worker_common.py:build_eagle_verify_input()`](../../../python/sglang/srt/speculative/eagle_worker_common.py) 里：
`build_tree_kernel_efficient` 只用 `seq_lens_sum` 去 **size 那块没有预分配的 FULL_MASK tree mask**，
所以给大了无害、给 0 也无害。于是当 `batch.seq_lens_sum is None` 时分两种情况兜：

| 情形 | 传给 kernel 的 `seq_lens_sum` |
|---|---|
| 后端已预分配 mask buffer，或 mask mode 是 `QLEN_ONLY`（尺寸只跟 bs 有关） | `0`——直接跳过 `.sum().item()` 这次 D2H |
| 其余（要真按总长开 FULL_MASK） | `bs * attn_backend.max_context_len`（上界，超配安全） |

同一函数还会优先**直写 target 注意力后端自带的 `verify_mask.buffer`**（`verify_mask.fits(bs)` 为真时），
eager 模式下 bs 超过捕获上限才退回临时分配。

### 8.5 draft KV 位置分配（`eagle_worker_common.py:prepare_for_draft()`）

- `page_size == 1` 或 `topk == 1`：连续布局，
  [`kernels/ops/speculative/cache_locs.py:assign_draft_cache_locs_contiguous()`](../../../python/sglang/kernels/ops/speculative/cache_locs.py)
  （CPU 走 `assign_draft_cache_locs_contiguous_cpu`），`out_cache_loc` 长度 `bs * topk * num_steps`。
- `page_size > 1` 且 `topk > 1`：每分支页对齐，各分支起点
  `prefix_base + t * num_new_pages * page_size + last_page`。因树分支不能共享半页，需
  `eagle_worker_common.py:duplicate_prefix_tail_to_draft_branches()` 把 prefix 尾页 KV 复制进每个分支的
  首页空洞，保证整页读一致。

收尾时该函数还会把本次 draft-decode 的动态宽度写到 draft input 上：
`draft_input.num_tokens_per_req = topk`（连同 `num_tokens_for_logprob_per_req`），
CUDA graph runner 用它和捕获宽度 `captured_req_width` 比对决定能否复用图（见 §10）。

---

## 9. Verify 与接受逻辑

⚠️ **verify 的函数体已经搬走了**：`eagle_worker_v2.py:EAGLEWorkerV2.verify()` 现在只是一层
**薄转发**，真正的实现在
[`eagle_worker_common.py:run_eagle_verify()`](../../../python/sglang/srt/speculative/eagle_worker_common.py)。
单层 EAGLE 传 `metadata_ready_pre_pad=False, finalize_tree_path=True`，Multi-Layer EAGLE
（见 §16）传相反的一对——这两个开关就是两个 worker 曾经"逐字保留"的差异，被收敛成参数了。

`run_eagle_verify()` 流程：

```
① record_stream_for_v2_verify（流记账）
② plan_stream 里 eagle_prepare_for_verify（注意力 plan）→ wait_stream 汇合
③ update_verify_buffers_to_fill_after_draft（用真值刷新 mask/position，因为 plan 时 draft 还没出结果）
④ 有 grammar 时先 GrammarTree.from_device(...)  ← 必须早于 target verify 发射（见 §19.2）
⑤ target_worker.forward_batch_generation(is_verify=True)  ← 一次验证前向
⑥ 有 grammar 时 build_grammar_vocab_mask(tree=..., barrier=grammar_barrier)
⑦ eagle_sample → (predict, accept_lens, accept_index)
⑧ new_seq_lens = batch.seq_lens + accept_lens
⑨ clear_unaccepted_c128_draft_states（若 KV cache 实现提供该钩子）
⑩ commit_mamba_states_after_verify（Mamba/GDN 混合模型）
⑪ fill_bonus_tokens_func（抽 bonus token）
⑫ return_logprob 时 compute_spec_logprobs
⑬ finalize_tree_path 且 topk>1 时 _finalize_accept_tree_path（把接受路径压到每请求块的前部）
⑭ 打包成 GenerationBatchResult（含 next_draft_input=EagleDraftInput(bonus_tokens=...)）
```

> `_finalize_accept_tree_path()` / `_compact_accept_to_front()` 现也在 `eagle_worker_common.py`。
> topk==1 时它们是恒等变换（接受路径天然就是前缀链），所以被条件跳过。
> 返回的 `GenerationBatchResult` 里 `extra_keep_alive_refs=[verify_forward_batch]`——verify 期的
> GPU 张量必须活过紧随其后的 `prepare_for_draft_extend` 对 `batch.input_ids` 的 rebind。

### 9.1 接受分派（`eagle_utils.py:eagle_sample()`）

`eagle_sample` 先施加 penalty / logit_bias / grammar mask，再按采样模式分派：

| 模式 | 触发条件 | kernel | 说明 |
|---|---|---|---|
| **贪心树验证** | `sampling_info.is_all_greedy`，**或**设备是 CPU / NPU / HIP / XPU | `eagle_utils.py:verify_tree_greedy_func()`（转 sgl_kernel） | `argmax` 出 target 预测，沿树遍历匹配最长接受路径 |
| **树形采样** | 采样 + 未开拒绝采样 | `tree_speculative_sampling_target_only`（sgl_kernel） | 温度/top-k/top-p renorm 后按树做投机采样 |
| **链式拒绝采样** | 采样 + `speculative_use_rejection_sampling` | [`kernels/ops/speculative/reject_sampling.py:chain_speculative_sampling_triton()`](../../../python/sglang/kernels/ops/speculative/reject_sampling.py) | 经典拒绝采样，见 §9.3 |

⚠️ 老文档把贪心门禁写成"`is_all_greedy` 或 NPU/HIP"——实际的**非 CUDA 兜底集合更宽**，
CPU 与 XPU 也一律走贪心树验证（这两个平台没有树形采样 kernel）。

温度/top-k/top-p 处理：对 target logits 除以温度做 softmax，再按需 `top_k_renorm_prob` /
`top_p_renorm_prob`。阈值 `threshold_single` / `threshold_acc` 从 `get_spec()` 参数包取
（`speculative_accept_threshold_single` / `_acc`），`deterministic=True`。

返回 `(predict, num_correct_drafts + 1, accept_index)`（`eagle_utils.py:eagle_sample()` 末尾）——
再次体现 `+1` bonus。

### 9.2 bonus token 抽取（`kernels/ops/speculative/eagle.py:fill_bonus_tokens()`）

```python
bonus_token_idx = accept_stride * pid + accept_len - 1
```

bonus token 是每请求最后一个接受槽的 token。Python 侧包装是同文件的 `fill_bonus_tokens_func()`。
注意 stride 用 `accept_index.shape[1]`（= `spec_steps+1`）而非 `num_draft_tokens`，否则 topk>1 树会错位——
这条注释就写在 `eagle_worker_common.py:run_eagle_verify()` 的调用点上。

### 9.3 拒绝采样内核（`kernels/ops/speculative/reject_sampling.py`，chain-only）

`speculative_sampling_classic_kernel`（`@triton.jit`）实现经典的无损拒绝采样：

**验证循环**逐个 draft token 做接受测试：

```
p = target_prob[draft_token]     # target 对该 draft token 的概率
q = draft_prob[draft_token]      # draft 对该 token 的概率
coin ~ Uniform(0,1)
if coin * q < p:   接受，num_accept += 1，前进
else:              拒绝，停止
```

kernel 往 `AcceptTokenNum` 里写的是 `num_accept`（**不含 bonus**），`+1` 是在
`eagle_sample()` 返回时补的。

**最终采样**（`Final Sampling` 段，两趟扫词表：先求归一化常数，再走 CDF）：

| 情形 | 采样分布 |
|---|---|
| 中途被拒（`all_drafts_accepted == 0`） | 归一化残差 `(p − q)+`，其中 `q` 取**被拒那一行**的 draft 分布 |
| 全部接受 | 纯 target `p`（此时 `cur_prob_row == num_steps`，越过了 `DraftProbs` 的行数，kernel 注释显式说明这一支不会解引用 `q` 指针） |

两个数值兜底：`q` 为 NaN（退化的 draft 行）按 0 处理，残差退回纯 `p`；`norm_sum == 0`
（拒绝时 `p == q` 处处相等，数值上近乎不可能）时 `final_token` 落到 `VOCAB_SIZE - 1`。

`chain_speculative_sampling_triton()` 是 Python 包装，`grid=(batch_size,)`，`BLOCK_V=4096`；
签名里 `retrive_next_token` / `retrive_next_sibling` / `deterministic` 在链式下**未使用**。

> **无损保证的直觉**：接受概率 `min(1, p/q)` + 残差重采样，是标准的 speculative sampling（Leviathan 2023 / Chen 2023）证明输出严格服从 target 分布。贪心路径则直接比对 argmax，天然无损。

### 9.4 接受长度语义

| 量 | 定义 | 含 bonus? |
|---|---|---|
| `num_correct_drafts` | 被验证通过的 draft 数 | 否 |
| `num_accept_tokens` | `num_correct_drafts + 1` | 是（含 bonus） |

这一对贯穿始终，也是 §22 两个指标差异的根源。

---

## 10. CUDA Graph

EAGLE 家族有**两个独立的 draft CUDA Graph runner**，因为 draft 的两个相位张量形状/forward_mode 根本不同，无法共用一张图。

> 📌 **术语更新**：这两个 runner 上原先叫 `num_tokens_per_bs` 的字段已经改名为
> **`captured_req_width`**（"捕获时每请求的 token 宽度"），并且不再各自硬编码，而是统一由
> [`spec_utils.py:resolve_num_tokens_per_req(phase=...)`](../../../python/sglang/srt/speculative/spec_utils.py)
> 单点推导，`phase` 取 `"draft_decode"` / `"draft_extend"` / `"target_verify"`。
> 为什么要单点：自适应投机会给每个候选 step 配置单独捕获一批图，宽度必须跟随该配置的
> **override 后**的叶子值，而不是启动时的原始值——所以宽度只能从参数包 `get_spec()` 里取。
> 与之配对的是 `SpecInput.num_tokens_per_req`（每次前向的**动态**宽度）：`can_run_graph()`
> 就是拿 `spec_info.num_tokens_per_req != self.captured_req_width` 来否掉不匹配的批次。

### 10.1 draft-decode runner（`eagle_draft_cuda_graph_runner.py:EAGLEDraftCudaGraphRunner`）

- 类头 `EAGLEDraftCudaGraphRunner(DecodeCudaGraphRunner)`，`capture_forward_mode = ForwardMode.DECODE`；
- `captured_req_width = resolve_num_tokens_per_req(phase="draft_decode")`，即 **topk**——每棵树的分支数；
- `max_num_token = max_bs * captured_req_width`；`out_cache_loc` 按 `num_steps` 倍放大以容纳每步 KV；
- `run_once()` 调 `draft_forward`，即**整个 `num_steps−1` 子前向循环被捕进一张图**；
- `execute()` 把 batch pad 到最近的捕获桶，做分组 `foreach` 拷贝。

### 10.2 draft-extend runner（`eagle_draft_extend_cuda_graph_runner.py:EAGLEDraftExtendCudaGraphRunner`）

- 类头 `EAGLEDraftExtendCudaGraphRunner(DecodeCudaGraphRunner)`，
  `forward_mode = ForwardMode.DRAFT_EXTEND_V2`——独立 forward mode；
- `captured_req_width = resolve_num_tokens_per_req(phase="draft_extend")`，即
  `speculative_num_draft_tokens`——**整棵树宽**，而不是 `num_steps+1`，否则 topk>1 的 draft-extend 会溢出 buffer；
- `run_once()` 调 `model.forward` 一次 + topk/argmax，是单次 extend 前向而非循环；
- `execute()` 收尾发布 **`shared_read_done_event`**（`self.model_runner.shared_read_done_event = read_done`）——
  draft-extend 是 EAGLE 家族**最后读共享 draft buffer 的相位**，在此触发 WAR 屏障，"最后一次写赢得信箱"。
  ⚠️ 旧名 `war_fastpath_read_done_event` 已删除；哪个 runner 负责发这个事件，由
  `base_spec_worker.py:BaseSpecWorker.last_shared_read_runner` 属性声明。

### 10.3 对比

| | draft-decode runner | draft-extend runner |
|---|---|---|
| forward_mode | DECODE | DRAFT_EXTEND_V2 |
| `captured_req_width` | `topk`（`phase="draft_decode"`） | `num_draft_tokens`（`phase="draft_extend"`，全树宽） |
| 捕获内容 | `num_steps−1` 步子前向循环 | 单次 extend 前向 + topk |
| 捕获条件 | 仅 `num_steps>1` | 总是 |
| 特殊职责 | — | 发 `shared_read_done_event`（WAR 屏障） |

---

## 11. STANDALONE

**独立小模型草稿**（经典 Leviathan / Chen 式）：draft 是一个完整的、独立的小模型，有自己的权重、KV、embedding、lm_head，仅以 token id 为条件，**不看 target hidden state**。

**核心文件**：[`standalone_worker_v2.py`](../../../python/sglang/srt/speculative/standalone_worker_v2.py)

- `StandaloneDraftWorker(EagleDraftWorker)`：加载**自己**的 draft 权重（`__init__` 里嵌套一个 `TpModelWorker(..., is_draft_worker=True)`）。
- `StandaloneDraftWorker.init_lm_head` 是**空 `pass`**——这是与 EAGLE 的定义性区别：EAGLE 把 target 的 embedding/lm_head 别名进 draft；STANDALONE 完全自给自足。
- `StandaloneWorkerV2(EAGLEWorkerV2)`：复用 EAGLE 的 draft/verify 骨架（topk 树、`eagle_sample`、`run_eagle_verify`），但 draft 是真正独立的模型，`capture_hidden_mode=NULL`，`SpeculativeAlgorithm.carries_draft_hidden_states()` 对它返回 `False`。
- `StandaloneWorkerV2._validate_vocab_compatibility`：draft 与 target 必须**同 vocab_size 且同 token→id 映射**（两边 `get_vocab()` 相等），否则 raise。要求同构词表。
- `context_length` 在 `StandaloneWorkerV2.__init__` 里强制对齐 target；`adaptive_controller = None`（尚未接自适应）。

**与 EAGLE 的对比**：EAGLE 的 draft 是条件于 target 最后 hidden 的轻量头；STANDALONE 跑一个独立小模型，接受率通常更低，但无需训练 EAGLE 头、任意现成小模型即可用。

---

## 12. NGRAM

**无模型的 prompt-lookup / 后缀自动机草稿**：草稿 token 来自对已见文本的字符串匹配，而非任何前向。适合大量重复、长上下文回填的场景（代码补全、RAG、少量改写）。

**核心文件**：[`ngram_worker.py`](../../../python/sglang/srt/speculative/ngram_worker.py)、`ngram_info.py`、`cpp_ngram/`、`external_corpus_manager.py`

- `NGRAMWorker(BaseSpecWorker)`：`draft_worker` property **返回 `None`**（"NGRAM 无 draft 模型"）。
- `NGRAMWorker.__init__` 建一个 `NgramCorpus`（`min/max_bfs_breadth`、`match_type`、`capacity`、`max_trie_depth`），可选预热外部语料；`_init_preallocated_tensors` 预开候选/mask buffer。
- `NGRAMWorker._prepare_draft_tokens`：`ngram_corpus.batch_get(...)` → `(req_drafts, mask)`，从历史 n-gram 后缀里查出候选。
- `NGRAMWorker._prepare_for_speculative_decoding`：仅 decode；`reconstruct_indices_from_tree_mask`（sgl_kernel）建树 mask（`USE_FULL_MASK=True`），设 `TARGET_VERIFY`，打包 `NgramVerifyInput`（树链关系由模块级 `_derive_tree_links()` 推出）。
- `NGRAMWorker.forward_batch_generation`：target verify → `eagle_sample`（复用 EAGLE 接受逻辑）→ `move_accept_tokens_to_target_kvcache` → `_update_ngram_corpus`（把新生成 token 回灌语料）→ 清理离场请求匹配态。

**数据结构 `ngram_info.py:NgramVerifyInput(SpecInput)`**：类型 `NGRAM_VERIFY`；`max_tree_depth` property 返回 `draft_token_num`（无深度上限——按节点预算的不规则树）；`tree_topk` property 返回 `-1`（不规则树）。

**后缀自动机 `cpp_ngram/ngram_corpus.py:NgramCorpus`**：包装 C++ 类（`jit_kernel.ngram_corpus`），按请求维护匹配态（`_req_id_to_state_id`）。Python 侧方法是
`batch_put()` / `batch_get()` / `erase_match_state()` / `reset()` / `leaf_paths_from_mask()` 等；
⚠️ **`match_stateful` 不是 Python 方法**，它是底层 C++ 对象的方法，由 `batch_get()` 内部
`self._obj.match_stateful(state_ids, batch_tokens, total_lens)` 调用（同理 `erase_match_state()`
内部调 `self._obj.erase_states()`）。

**外部语料（`cpp_ngram/external_corpus.py`）**：模块级函数 `iter_external_corpus_chunks()` 读 JSONL、分词、切块（`DEFAULT_CHUNK_SIZE=4096`），文档间插 `SEPARATOR_TOKEN=-(2**31)`。**用途**：预热 n-gram trie，让请求刚进来（还没积累重复上下文）时也能命中领域文本。`external_corpus_manager.py:ExternalCorpusManager` 是独立的 scheduler manager，后台线程异步加载，event loop 轮询 `check_pending_load()`。

> NGRAM 是唯一 `SpeculativeAlgorithm.has_draft_kv()==False` 的算法：它的树只活在 verify mask 里，不写 draft KV 链，故 KV 预算无需 per-topk 分页取整。

---

## 13. DFLASH

**块级 linear 草稿，KV 从 target hidden 投影而来**。DFlash 的 draft 模型 KV **不是**自回归跑出来的，而是把 target 的 hidden state 经投影**合成**进 draft KV；draft 用 `<|MASK|>` token 填充一个定长 block，一次性 draft 出整块，再线性（非树）验证。

**核心文件**：[`dflash_worker_v2.py`](../../../python/sglang/srt/speculative/dflash_worker_v2.py)、`dflash_info.py`、`dflash_info_v2.py`、`dflash_utils.py`、`dflash_disaggregation.py`

- `dflash_worker_v2.py:DFlashWorkerV2(BaseSpecWorker)`：建独立 draft `TpModelWorker`，有自己的 KV 池与 attn 后端。同文件的 `_DflashDraftSampler` 是 capture-safe 的贪心 argmax（over target lm_head），折进 draft cuda graph（仅 tp=1）。
- `dflash_info.py:DFlashVerifyInput(SpecInput)`：类型 `DFLASH_VERIFY`，**永远 `topk=1`、线性（非树）验证**，`generate_attn_arg_prefill()` 用标准因果 mask（无树 mask，验证是一条直线）。
- `dflash_info_v2.py:DFlashDraftInputV2(SpecInput)`：类型 `DFLASH_DRAFT`，携带 `bonus_tokens`/`new_seq_lens`/`hidden_states` + `future_indices`。`prepare_for_decode()` 超额预留 KV（`reserved_len = max(cur, committed + 2*block_size)`）。**DSPARK 直接复用这个类**（见 §14）。
- `dflash_utils.py:DFlashDraftConfig`：`num_hidden_layers`、`num_target_layers`、`block_size`、`target_layer_ids`、`mask_token`。`DEFAULT_DFLASH_MASK_TOKEN="<|MASK|>"`。

**三相位 `dflash_worker_v2.py:DFlashWorkerV2.forward_batch_generation()`**：

```
EXTEND：       target prefill（FULL hidden）→ _append_target_hidden_to_draft_kv_by_loc
               把 prompt hidden 投影/物化进 DRAFT KV
DECODE 相1(draft)： 用 mask_token 填 block_ids，position 0 = bonus token，跑 draft forward
DECODE 相2(verify)：TARGET_VERIFY + 标准因果 mask，算 accept_len/bonus
               greedy: matches = candidates[:,1:]==target_predict[:,:-1]; correct = cumprod.sum
               （见 dflash_utils.py 的贪心接受辅助函数）；非贪心用 tree_speculative_sampling_target_only
DECODE 相3(物化)：  再把已确认 verify hidden 投影进 draft KV（project_target_hidden +
               kv_proj_only/apply_k_norm/apply_k_rope + set_kv_buffer_prefix_valid）
```

**DFLASH 独特性**：定长 `block_size` 的 **linear** 草稿（无树，topk=1）；draft KV 由 target hidden **投影合成**而非 token 自回归；`<|MASK|>` 填充块位置。支持 compact draft cache（滑窗 `--speculative-draft-window-size`；旧名 `--speculative-dflash-draft-window-size` 已弃用，仅作别名保留）。

⚠️ **更正**：旧文档说 `dflash_utils.py:validate_dflash_request()` "拒绝 logprob / hidden_states / grammar 请求"。现在这个函数只剩**一条**规则——`enable_overlap and req.return_hidden_states` 才返回拒绝理由，其余一律 `None`。logprob 与 grammar **不再**被它拒（grammar 反而是 DFLASH 支持 overlap 的场景，见 §19.2）。

---

## 14. DSPARK

**DSPARK 是 DFLASH 的"带置信度 + 带预算调度"进化版**。它继承 DFLASH 的两个基本盘——
**块级（block）草稿**、**线性（非树）验证**——但多做三件事：

1. **草稿块自带 confidence**：draft 模型上挂一个 `confidence_head`，一次前向就给出块内**每个位置**
   的"能活到这里"的存活概率（survival）；
2. **按 confidence 给每个请求裁不同的验证长度**：置信度高的请求多验几个 token，低的少验，
   总量受一个 **verify token 预算**约束——这就是 §14.6 的 **ragged verify**；
3. **预算靠离线/在线成本模型定**：SPS 代价表（每步耗时 ~ (请求数, token 预算)）+ STS 温度校准表
   + 在线块接受率上界估计，三者一起把"多验一个 token 值不值"量化。

术语先钉住（下文反复用）：

| 记号 | 含义 |
|---|---|
| **γ**（gamma） | 草稿块长度，即一次 draft 出多少个 token。CLI 是 `--speculative-dspark-block-size` |
| **verify window** | `γ + 1`，即 `speculative_num_draft_tokens`——块前面还要拼一个 anchor（上轮的 bonus token） |
| **confidence / survival** | `confidence_head` 输出的 `[bs, γ]` 存活概率 |
| **verify_lens** | 每请求本步实际验证的 token 数（ragged verify 的核心张量），`≤ γ+1` |
| **budget** | 全 batch 可以"多验"的 token 总量，由 SPS 代价表反解 |

DSPARK **复用了 DFLASH 的数据结构与相位枚举**，这一点非常关键：

- draft/verify 期的 batch state 就是 `dflash_info_v2.py:DFlashDraftInputV2`——`_forward_decode()`
  开头会 `isinstance` 硬校验，不是这个类型直接 raise；
- `SpecInputType` 用的也是 `DFLASH_DRAFT` / `DFLASH_VERIFY` 两个相位，**没有单独的 DSPARK 相位**；
- 谓词上 `SpeculativeAlgorithm.is_dflash_family()` 同时对 DFLASH 与 DSPARK 为真，
  而 `is_dspark()` / `supports_ragged_verify()` 只对 DSPARK 为真（`supports_ragged_verify()`
  目前是 **DSPARK 独占**）。

### 14.1 `dspark_components/` 文件地图

DSPARK 是唯一一个**独占一个子目录**的算法家族（`python/sglang/srt/speculative/dspark_components/`，
12 个文件）。按职责分四组：

| 组 | 文件 | 关键符号 | 职责 |
|---|---|---|---|
| **顶层编排** | [`dspark_worker_v2.py`](../../../python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py) | `DSparkWorkerV2(BaseSpecWorker)`、`_forward_prefill()`、`_forward_decode()` | 三相位主循环、组件装配 |
| | `dspark_config.py` | `DEFAULT_DSPARK_GAMMA = 7`、`SUPPORTED_DSPARK_MARKOV_HEAD_TYPES = ("vanilla","gated","rnn")`、`DSV4_DRAFT_ATTENTION_BACKEND = "dsv4"`、`DSparkDraftConfig` / `DSparkRuntimeConfig`（均为 `msgspec.Struct, frozen=True`）、`parse_dspark_draft_config()`、`read_draft_checkpoint_config()`、`checkpoint_bundles_dspark_draft()`、`dspark_gamma_from_num_draft_tokens()`、`draft_is_deepseek_v4()` | 从 draft ckpt 的 hf_config 里读 γ、markov 头类型等 |
| **草稿** | `dspark_draft.py` | `DraftBlockProposer`（`propose()` / `_run_forward()` / `run_idle_participation()`）、`DraftProposal` / `DraftBlockResult` / `DraftForwardResult`、`sample_draft_block()`、`make_next_draft_input()`、`resolve_greedy_mask()`、`select_draft_hidden_without_anchor()` | 一次前向出整块草稿 |
| | `dspark_draft_sampler.py` | `DsparkDraftSampler`、`maybe_build_draft_sampler()`、`greedy_step_sampler()`、`_resolve_folded_sampling()` | 把采样折进 draft cuda graph（capture-safe） |
| | `dspark_kv_inject.py` | `TargetHiddenKvInjector`（`inject_target_hidden()` / `inject_ragged()` / `_inject_mla()`） | 把 target hidden 投影写进 draft KV（同 DFLASH 思路） |
| **计划与验证** | `dspark_planner.py` | `DSparkVerifyPlanner`、`HostConfidenceBudgetPlanner`、`VerifyWindow`、`DSparkScheduleConfig`、`alloc_verify_window()`、`compute_confidence()`、`build_markov_embed_stack()`、`idle_ragged_layout()` / `uniform_ragged_layout()`、`graph_tier_fill_budget()`、`ragged_layout_exceeds_captured_grid()` | 算 confidence → 定预算 → 排 layout |
| | `dspark_verify.py` | `TargetVerifyExecutor`（`run_compact()` / `run_non_compact()` / `run_ragged()` / `accept_and_finalize()` / `commit_hidden()`）、`DsparkVerifyEpilogue`、`accept_draft_tokens()`、`verify_logits_adjustments_are_noop()` | target 验证前向 + 接受 |
| **成本模型与观测** | `dspark_sps.py` | `SpsCostTable` / `SpsAdditiveCostTable`、`profile_sps_table()`、`load_sps_table_from_path()`、`build_uninitialized_sps_table()`、`is_uninitialized_sps_table()` | 每步耗时代价表 |
| | `dspark_sts.py` | `DSparkStsCalibration`、`StsDataRecorder`、`load_sts_calibration_from_path()` | confidence 的**逐位置温度校准** |
| | `dspark_block_accept_estimator.py` | `BlockAcceptEstimateRecorder`、`gather_chunked_token_logprobs()` | 在线估计"块接受长度的上界" |
| | `dspark_observability.py` | `DsparkInfoDumper`、`DsparkStepObservers`、`ConfidenceMetricsProbe`、`PerPositionConfidenceMetrics`、`InfoSegment` / `InfoComponent`、`DecodeStepRecord` | 逐步 trace dump + confidence 质量指标 |

> 这批新代码几乎全部用 `msgspec.Struct` 而不是 `@dataclass`——正好是仓库
> `.claude/rules/no-dataclasses.md` 要求的写法，可以当范例读。

### 14.2 `DSparkWorkerV2` 骨架

```
DSparkWorkerV2(BaseSpecWorker)          ← 直接继承基类，不继承 EAGLEWorkerV2
├── self.target_worker                  ← target TpModelWorker
├── self.draft_worker / model_runner     ← 独立 draft 模型（DFlash 系 draft 结构）
├── self._proposer      : DraftBlockProposer     （dspark_draft.py）
├── self._verify_planner: DSparkVerifyPlanner    （dspark_planner.py）
├── self._verify_executor: TargetVerifyExecutor  （dspark_verify.py）
├── self._kv_injector   : TargetHiddenKvInjector （dspark_kv_inject.py）
├── self._observers     : DsparkStepObservers    （dspark_observability.py）
└── self._tp_sync       : SpecTpSync             （spec_tp_sync.py）
```

几个需要留意的成员：

| 符号 | 说明 |
|---|---|
| `DSparkWorkerV2.carries_confidence`（property） | 转发 planner 的同名属性——draft ckpt 里有没有 `confidence_head`。**没有 head 就只能跑 static 模式** |
| `DSparkWorkerV2.spec_v2_attn_backends`（property） | 交给基类做 attn 后端的统一生命周期管理 |
| `DSparkWorkerV2.__getattr__` | 兜底转发到 draft worker，省掉一堆样板代理 |
| `DSparkWorkerV2._draft_context()` | 上下文管理器：draft 前向期间切到 draft 的 forward context |
| `DSparkWorkerV2.set_dspark_forced_budget_frac()` | 调试/压测用，强制固定预算比例 |
| `DSparkWorkerV2.note_request_finished()` | 请求离场时通知块接受率估计器（区分 `natural_stop`） |
| `DSparkWorkerV2.dump_info_records()` / `clear_info_records()` | 把 `DsparkInfoDumper` 攒的 trace 交给上层导出 |

### 14.3 Prefill 路（`DSparkWorkerV2._forward_prefill()`）

和 DFLASH 同构：**target 先 prefill，然后把 prompt 的 hidden 投影进 draft KV**。

```
1. target_worker.forward_batch_generation(batch, capture_hidden_mode=FULL)
2. _tp_sync.sync(SpecTpSyncSite.DSPARK_TARGET, next_token_ids)   ← TP 间对齐首 token
3. on_publish(new_seq_lens)                                       ← 交给重叠调度发布
4. 三个硬校验：hidden_states / (extend_lens, prefix_lens) / out_cache_loc 不能是 None
5. compute_position(...) 算出每 token 在 draft 侧的 position
6. _kv_injector.inject_target_hidden(target_hidden, cache_loc, positions, ...)
7. logits_output.hidden_states = None      ← 避免重叠调度把大 hidden 拷回 CPU
8. next_draft_input = make_next_draft_input(bonus_tokens=next_token_ids, ...)
```

两个容易忽略的细节：

- **注入必须发生在 prefill 返回之前**。代码注释写得很直白："the scheduler may update radix
  afterward, invalidating `out_cache_loc`"——调度器随后会更新 radix cache，`out_cache_loc`
  就失效了，所以不能延后。
- **unified_kv 下要多传 `state_slot` / `final_pos`**。`is_unified_kv_triton()` 为真时，注入是按
  `(draft req slot, position)` 写进一个 SWA 环形 buffer 的；prefill 的老 token 会和新 token
  抢同一个环槽位，所以要把"每 token 属于哪个请求槽 + 该请求的最终 position"一起传进去，
  让注入器**只保留最后一个 SWA 窗口**。

DP-attention 开启时的 idle 步也必须陪跑一次 target 前向（否则集合通信对不齐），
见 `_forward_prefill()` 开头的 `if batch.forward_mode.is_idle()` 分支。

### 14.4 Decode 路（`DSparkWorkerV2._forward_decode()`）——完整 12 步

这是整个 DSPARK 最核心的函数，按顺序拆解：

```
 ① 状态校验：batch.spec_info 必须是 DFlashDraftInputV2（None 则造 idle input）
 ② idle 分支：DP-attention 下 MoE draft 要 run_idle_participation，
    verify 侧也要用 _idle_verify_ragged_layout(batch) 陪跑一次
 ③ alloc_verify_window(...)  → VerifyWindow：为 γ+1 个验证槽分配 KV 位置
 ④ 【DRAFT 段】_proposer.propose(...) → DraftProposal
       .draft_block_ids / .draft_block.draft_tokens / .draft_hidden
       .confidence（草稿前向顺手算出来的话）/ .confidence_tap / .folded
 ⑤ confidence 为 None 时补算：_verify_planner.compute_confidence_tensor(...)
 ⑥ verify_token_budget = _verify_planner.resolve_verify_token_budget(...)
 ⑦ layout = _verify_planner.schedule_layout(...)   → RaggedVerifyLayout | None
       run_compact = _verify_planner.should_run_compact(layout=layout)
 ⑧ verify_ids_2d = cat([anchor(draft_block_ids[:, :1]), draft_tokens], dim=1)
    有 grammar 时 GrammarTree.from_linear_chain(verify_ids_2d)  ← 必须早于 verify 发射
 ⑨ 【TARGET_VERIFY 段】run_compact ? _verify_executor.run_compact(...)
                                   : _verify_executor.run_non_compact(...)
    有 grammar 时 build_grammar_vocab_mask(...).apply(next_token_logits)
 ⑩ accept = _verify_executor.accept_and_finalize(folded_accept=..., ...)
       → correct_len / bonus / commit_lens / cap_trim_lens / out_tokens / new_seq_lens
 ⑪ on_publish(new_seq_lens, confidence=...)；_commit_target_mamba_states_after_verify(...)
    非 folded_commit 时 _verify_executor.commit_hidden(...)  ← 把已确认 hidden 投影回 draft KV
 ⑫ _observers.observe_verify_step(...)；打包 GenerationBatchResult
       accept_lens=commit_lens, block_accept_lens=commit_lens+cap_trim_lens,
       cap_lens=layout.verify_lens, next_draft_input=make_next_draft_input(...)
```

**"folded" 是什么**：DSPARK 允许把采样与接受**折进 cuda graph**，省掉 graph 外的 kernel launch
与同步。是否可折由 `fold_eligible` 一串条件决定：

| 条件 | 原因 |
|---|---|
| `_verify_executor.verify_epilogue is not None` | 得有 `DsparkVerifyEpilogue` 这个图内尾巴 |
| `proposal.folded` | 草稿侧也得是折叠采样出来的 |
| `sampling_info is None or sampling_info.is_all_greedy` | 图内接受是贪心实现（`accept_greedy_triton`），采样批必须走 eager |
| `verify_logits_adjustments_are_noop(sampling_info)` | penalty / logit_bias 之类一旦非空，图内 logits 就不对了 |
| `self._simulate_acc_len <= 0` | 模拟接受长度的调试模式与折叠互斥 |
| `not batch.has_grammar` | **grammar mask 落不进图内 buffer**——有语法约束就强制 eager |

最终 `folded_accept = fold_eligible and run_compact and can_run_cuda_graph`——三者都成立才真折。

**三条 verify 执行路径**：

| 路径 | 触发 | 特点 |
|---|---|---|
| `run_non_compact()` | `layout is None`（static 模式）或 planner 判定不值得压 | 每请求都验满 γ+1，标准因果 mask |
| `run_compact()` | `should_run_compact(layout)` 为真 | 按 `layout.verify_lens` 把各请求**变长**地压进一段连续 token，验完再 scatter 回 `(bs*chain_len)` 行 |
| `run_ragged()`（`TargetVerifyExecutor._run_ragged()`） | `run_compact()` 内部使用 | 真正拼 ragged 几何（`qo_indptr` / `cu_seqlens`）的那层 |

> `run_compact()` 会把行 scatter 回 `(bs * chain_len)` 布局，**正是为了让 grammar mask 在两条
> verify 路径上都能和 logits 对齐**——这条注释就写在 `_forward_decode()` 的 grammar 分支上。

**接受逻辑**（`dspark_verify.py:accept_draft_tokens()`）按采样情况三分：

| 情况 | 走哪条 |
|---|---|
| 全贪心（`sampling_info is None or is_all_greedy`） | `AcceptGreedy.execute(...)` |
| 全采样（`not is_any_greedy`） | 先 `SoftmaxTemp` 出 `draft_probs`，再 `AcceptSampling.execute(...)` |
| 混合（部分请求贪心、部分采样） | **两条都跑一遍**，然后 `SelectMixedAccept.execute(greedy_mask=...)` 按请求逐行选 |

返回三元组 `(correct_len, bonus, cap_trim_lens)`。`cap_trim_lens` 是 ragged verify 特有的量：
**因为验证长度被预算裁短，本来可能被接受却没验到的 token 数**——它不算进 `accept_lens`
（那是真正提交的长度），但会加进 `block_accept_lens` 用于评估"如果验满能接受多少"。

### 14.5 三张"表"：confidence / SPS / STS

DSPARK 的调度决策靠三个东西支撑，都在 `DSparkVerifyPlanner.__init__` 里装配。

**(1) confidence head —— 在线信号**

`dspark_planner.py:compute_confidence()`：

```
confidence_raw = confidence_head(draft_hidden, markov_embed_stack)
confidence     = confidence_head.apply_sts(confidence_raw)     # 逐位置温度校准
```

`confidence_head.with_markov` 为真时，还要先用
`build_markov_embed_stack(anchor_tokens, draft_tokens, markov_head, gamma)` 把
"anchor + 草稿 token"的 embedding 堆成一叠喂进去——即置信度不只看 hidden，也看具体草稿了哪些 token。
markov 头类型由 ckpt 决定，合法值见 `dspark_config.py:SUPPORTED_DSPARK_MARKOV_HEAD_TYPES`
（`"vanilla"` / `"gated"` / `"rnn"`）。

> ⚠️ **没有 confidence head 就只能 static**。`DSparkVerifyPlanner.__init__` 里，
> 只要 ragged 模式不是 STATIC 而 `confidence_head is None`，就直接 raise，并且错误信息明确
> 指向"ckpt 不完整（没有 `enable_confidence_head` + 训练好的 `confidence_head` 权重）"。

**(2) SPS 代价表 —— 离线 profile 的"每步耗时"**

`dspark_sps.py` 里两种表：

| 类 | 形状 | 查询 |
|---|---|---|
| `SpsCostTable` | 一维：`batch_tokens → 每步耗时` | `lookup(batch_tokens)`，表间线性插值（`_interp_clamped`） |
| `SpsAdditiveCostTable` | 二维可加：`(num_reqs, budget) → 耗时` | `step_time(num_reqs=..., budget=...)` |

来源三选一：`--speculative-dspark-sps-table-path`（`load_sps_table_from_path()`）、
在线 profile（`profile_sps_table()`，`SGLANG_DSPARK_ENABLE_SPS_RECORD=1` 时录）、
或 `build_uninitialized_sps_table()` 给一张**平表**兜底。

> ⚠️ 平表意味着"多验一个 token 不要钱"，于是预算退化成**verify-all（验满）**、调度收益为零。
> planner 在 tp_rank 0 上会打这条 warning：`"DSpark SPS table is uninitialized (flat): the
> verify budget degenerates to verify-all (zero scheduling gain)."` 判定函数是
> `is_uninitialized_sps_table()`，它也是 `_is_verify_all` 这个内部标志的来源之一。

**(3) STS 校准 —— confidence 的逐位置温度**

`dspark_sts.py:DSparkStsCalibration` 存一组 `temperatures`（长度必须等于 γ），
`load_sts_calibration_from_path()` 加载，写到 `confidence_head.sts_temperatures` 上。
两条自洽检查在 planner 构造期就做：

| 检查 | 触发条件 | 行为 |
|---|---|---|
| 温度长度 ≠ γ | `sts_temperatures.numel() != self.gamma` | raise，提示"refit the table for gamma=N" |
| 采集态与非恒等温度并存 | `SGLANG_DSPARK_STS_COLLECT_PATH` 非空且温度不全为 1.0 | raise——采集校准数据必须用**未校准**的原始 logits |
| 给了路径但 ckpt 没 confidence head | `sts_path and _confidence_head is None` | 只 warning 并忽略 |

采集侧是 `dspark_sts.py:StsDataRecorder`（`record()` / `flush()`，`flush_every` 控制落盘频率）。

**(4) 在线块接受率上界 —— `BlockAcceptEstimateRecorder`**

`dspark_block_accept_estimator.py` 想回答的问题是：**"如果验满，块接受长度最多能到多少？"**
它用 target 的 chunked token logprob（`gather_chunked_token_logprobs()`）反推一个区间
`[lo, hi]`，喂进内部的 `_OnlineCeiling`（滑动窗 `window_steps`，按 `log_interval` 打日志），
输出 `_CeilingSnapshot`。对外接口：

- `observe_verify_step(...)`：每个 verify 步喂一次；
- `note_request_finished(rid=..., natural_stop=...)`：请求离场时结算（EOS 与被截断要区别对待）；
- `online_estimate()` / `estimate_log_suffix()`：取当前估计，后者被
  `DSparkWorkerV2.block_accept_estimate_log_suffix` 拼进日志；
- 离线落盘由 `SGLANG_DSPARK_BLOCK_ACCEPT_ESTIMATE_PATH` 指定，在线打印间隔由
  `SGLANG_DSPARK_BLOCK_ACCEPT_ONLINE_INTERVAL` 控制。

### 14.6 Ragged verify：三种模式

**动机**：块级投机里每个请求都验满 γ+1 个 token 是浪费——有些请求的草稿明显要在第 2 个 token
就崩，验到第 8 个纯属烧算力。ragged verify 就是**让同一 batch 里不同请求验不同长度**。

模式开关是**环境变量**而非 CLI flag：
`SGLANG_RAGGED_VERIFY_MODE`，默认 `"static"`，读取入口
[`ragged_verify.py:read_ragged_verify_mode()`](../../../python/sglang/srt/speculative/ragged_verify.py)
（非法值直接 raise 并列出合法集合）。

| 模式（`RaggedVerifyMode`） | 值 | 含义 |
|---|---|---|
| `STATIC` | `"static"`（默认） | **关闭**：每请求都验满 γ+1，`layout is None`，走 `run_non_compact()` |
| `CAP_ACCEPT` | `"cap-accept"` | 按 confidence 给每请求算一个**上限 cap**，但张量仍按 `[bs, cap]` 的稠密形状走；被裁掉的部分计入 `cap_trim_lens` |
| `COMPACT` | `"compact"` | **真压紧**：把各请求变长的验证段拼成一条连续 token 流，`total_verify_tokens` 显著小于 `bs*(γ+1)` |

⚠️ 早期文档只写了 STATIC / COMPACT 两种，**漏了 `CAP_ACCEPT`**。便捷判定函数
`ragged_verify_compact_enabled()` 只在 COMPACT 时为真。

**核心数据结构 `ragged_verify.py:RaggedVerifyLayout`**（`msgspec.Struct, frozen=True`）：

| 字段 | 作用 |
|---|---|
| `verify_lens` | `[bs]`，每请求验证长度——**整个机制的输出** |
| `graph_num_tokens` | 向上取整到 cuda graph 捕获档位后的 token 数（真正 launch 的宽度） |
| `extend_start_loc` / `qo_indptr_device` | ragged 几何的前缀和索引 |
| `verify_lens_cpu` / `total_verify_tokens` | host 侧镜像与真实 token 总数 |
| `qo_indptr_host` / `kv_indptr_host` / `kv_lens_host` / `max_q_len` / `max_kv_len` | 给需要 host 元数据的注意力后端 |
| `cap` | 每行上限（CAP_ACCEPT 的稠密变体用）；`None` 表示全覆盖变体 |

**graph 档位对齐**是这里的关键工程约束：cuda graph 只捕获了若干离散 token 数，所以真实
`total_verify_tokens` 必须**向上取整**到某个档位。

- `ragged_verify.py:round_up_grid(total, grid)`：`bisect_left` 找档位；`total` 超过最大档位时
  **直接 raise**，并在错误信息里说明"调用方必须在选档之前就拒掉这个 batch"；
- `dspark_planner.py:ragged_layout_exceeds_captured_grid()` / `ragged_capture_num_tokens()` /
  `ragged_capture_max_slots()` 就是那个"提前拒"的判断；
- 取整多出来的空位不浪费：`dspark_planner.py:graph_tier_fill_budget()` 把它**回填成额外预算**
  （反正这些 token 已经付过钱了）。这正是 `--speculative-dspark-align-verify-tokens-to-graph-tier`
  的用途，而它**只在 COMPACT 模式生效**（见 §14.7）。

**其余辅助**：

| 符号 | 作用 |
|---|---|
| `resolve_ragged_verify_layout(forward_batch)` | 从 `forward_batch.spec_info.ragged_verify_layout` 取 layout；容忍 runner 那些不带 spec_info 的临时 replay 视图 |
| `build_capture_verify_lens(num_tokens=, num_slots=, num_draft_tokens=)` | 捕获期造一份"尽量均分"的 verify_lens（`base+1` 若干行 + `base` 若干行） |
| `RaggedTargetVerifyGeometry` + `build_ragged_target_verify_geometry()` | 由 `seq_lens + verify_lens` 拼出 `cache_seqlens_int32` / `cu_seqlens_q` / `cu_seqlens_k` |
| `compute_target_verify_graph_key()` | 给 target verify 选图的 key：`layout is None` 时是 `(bs, num_draft*bs)`，否则是 `(graph_num_tokens, graph_num_tokens)`，并断言不超过满块 |
| `VerifyExtendLengths` + `compute_uniform_extend_lengths()` / `compute_ragged_extend_lengths()` | 等长 vs 变长两种 extend 长度算法，产出同一个 struct 供后端消费 |
| `idle_ragged_layout()` / `uniform_ragged_layout()`（在 `dspark_planner.py`） | idle 步陪跑用的空 layout / 等长 layout |

**预算 → verify_lens 的求解**在 `DSparkVerifyPlanner._schedule_verify_lens()`：
`ScheduleVerifyLensTopk.execute(confidence=..., budget=..., cfg=...)` 出 `verify_lens`，
随后 `_tp_sync.sync(SpecTpSyncSite.DSPARK_PLAN, verify_lens)` **在 TP 组间对齐**（各 rank 必须
排出完全一样的 layout，否则集合通信形状不一致）。开了 invariant check 还会断言
`(verify_lens - min_verify_len).sum() <= budget`。

DP-attention 下还有一层：`_dp_tier_gather_enabled` 为真时，档位要在 **DP 组之间**再 all-gather
一次（`dp_global_verify_tier_num_tokens()`），条件相当苛刻——`attn_tp_size == 1`、
`attn_cp_size == 1`、`require_mlp_tp_gather()`、未禁重叠调度、非 PD 分离、`pp_size == 1`、
未设 `SGLANG_SCHEDULER_SKIP_ALL_GATHER`。

### 14.7 `_handle_dspark()` 门禁全表

参数钩子在
[`arg_groups/speculative_hook.py:_handle_dspark()`](../../../python/sglang/srt/arg_groups/speculative_hook.py)。
**硬拒（raise）**：

| 条件 | 报错要点 |
|---|---|
| `device` 不以 `cuda` / `npu` 开头 | "only supports CUDA or NPU device" |
| DP-attention（且 `dp_size > 1`）而未开 `--enable-dp-lm-head` | "requires --enable-dp-lm-head" |
| DP-attention + 非 NPU + `moe_a2a_backend not in ("none","megamoe")` | 只支持内建 TP MoE 或 megamoe |
| DP-attention + 非 NPU + `moe_a2a_backend != "none"` 但 ragged 模式不是 STATIC | "requires SGLANG_RAGGED_VERIFY_MODE=static" |
| DP-attention + `attn_cp_size > 1` | 不支持 context parallel |
| DP-attention + 非 NPU + `speculative_moe_a2a_backend` 与 target 的 `moe_a2a_backend` 不一致 | "DSpark ignores --speculative-moe-a2a-backend"，若显式给了必须和 target 一致 |
| `pp_size != 1` | 只支持 `pp_size == 1` |
| `speculative_draft_model_path is None` **且** target ckpt 未内置 dspark draft | "requires setting --speculative-draft-model-path" |
| `--speculative-dspark-block-size <= 0` | 必须为正 |
| `speculative_num_draft_tokens != γ + 1`（两者都显式给了） | 必须严格等于 γ+1 |
| 最终 `speculative_num_draft_tokens is None` | 无法解析 γ，提示去设 `--speculative-dspark-block-size` |
| `speculative_num_draft_tokens < 2` | γ+1 至少是 2 |

> `dp_size == 1` 配 dp_attention 在 DSV4 CP 下是个退化开关，钩子注释显式说明这种情况
> **跳过所有 DP-only 检查**。

**自动推导（`declare_resolution`）**：

| 参数 | 解析结果 |
|---|---|
| `speculative_draft_model_path` | target ckpt 内置 dspark draft 时（`checkpoint_bundles_dspark_draft()`）默认成 `--model-path`，同时把 `speculative_draft_model_revision` 对齐 `--revision` |
| `speculative_num_steps` | 强制 **1**；显式给了非 1 会 warning 后覆盖 |
| `speculative_eagle_topk` | 强制 **1**；同上 |
| `speculative_num_draft_tokens` | `γ + 1` |
| `max_running_requests` | 未显式设置时定为 **48**，并打 warning 说明可用 `--max-running-requests` 覆盖 |

**γ 的解析顺序**（先到先得）：

1. `--speculative-dspark-block-size` 显式给了 → 用它；
2. 否则读 draft ckpt 配置 `read_draft_checkpoint_config()` → `DSparkDraftConfig.resolve_gamma()`；
   读失败只 warning，不致命；
3. 还没有、且 `speculative_num_draft_tokens` 也没给 → `DEFAULT_DSPARK_GAMMA = 7`，打 warning。

**两条"不报错但没用"的 warning**（很容易踩）：

| flag | 何时是 no-op |
|---|---|
| `--speculative-dspark-align-verify-tokens-to-graph-tier` | ragged 模式**不是** COMPACT 时 |
| `--speculative-dspark-sps-table-path` | ragged 模式**是** STATIC 时（预算调度整个关着） |

四个 DSPARK CLI 参数的完整语义见 §4.3；`SGLANG_DSPARK_*` 环境变量（约 20 个）也在那一节。

### 14.8 可观测性

`dspark_observability.py` 是一套独立的 trace 体系，和全局 Prometheus 指标（§22）并行：

| 符号 | 作用 |
|---|---|
| `DsparkStepObservers` | 观测门面：`begin_step()` / `segment(InfoSegment.X)` / `observe_verify_step(...)` / `note_idle_decode_step()` |
| `InfoSegment` | 分段计时的段名（`DRAFT` / `TARGET_VERIFY` / ...），`_forward_decode()` 用 `with self._observers.segment(...)` 圈出来 |
| `InfoComponent` + `resolve_enabled_components()` | 按 `SGLANG_DSPARK_DEBUG_DUMP` 选要 dump 哪些组件（`ALL_COMPONENTS_TOKEN = "all"`） |
| `DecodeStepRecord` / `ReqDetail` / `DecodeStepObservation` | 逐步、逐请求的记录 schema |
| `DsparkInfoDumper` | 攒记录并导出；上限 `INFO_DUMP_MAX_RECORDS = 200_000`、单步 CPU 预算 `INFO_DUMP_MAX_STEP_CPU_SECONDS = 1.0`（超了自我限流） |
| `PerPositionConfidenceMetrics` / `ConfidenceMetricsProbe` | 评估 confidence 本身好不好：逐位置直方图 + `_auroc_from_hist()` 算 AUROC，`format_table()` 出可读表 |
| `_report_sps_prediction()` | 按 `SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL` 定期对比"SPS 表预测耗时 vs 实测" |

### 14.9 DSPARK vs DFLASH 一张表

| 维度 | DFLASH | DSPARK |
|---|---|---|
| 草稿粒度 | 定长 block（`block_size`） | 定长 block（γ），**但验证长度可变** |
| 验证拓扑 | 线性（topk=1，无树） | 线性（topk=1，无树） |
| draft KV 来源 | target hidden 投影合成 | 同（`TargetHiddenKvInjector`） |
| 是否有 confidence | 无 | **有**（`confidence_head`，逐位置 survival） |
| 验证长度 | 固定验满 | **STATIC 验满 / CAP_ACCEPT 限上限 / COMPACT 真压紧** |
| 预算模型 | 无 | **SPS 代价表 + STS 校准 + 在线接受率上界** |
| worker | `DFlashWorkerV2(BaseSpecWorker)` | `DSparkWorkerV2(BaseSpecWorker)` |
| batch state 类 | `DFlashDraftInputV2` | **同一个类**（复用） |
| `SpecInputType` 相位 | `DFLASH_DRAFT` / `DFLASH_VERIFY` | **同两个相位**（复用） |
| `num_steps` / `topk` | 1 / 1 | 强制 1 / 1 |
| `supports_ragged_verify()` | False | **True**（独占） |
| 代码位置 | `speculative/dflash_*.py` | `speculative/dspark_components/`（12 文件） |
| 额外指标 | — | `block_accept_lens`、`cap_lens`（见 §22） |

---

## 15. FROZEN_KV_MTP

**读 target KV 只读、不建 draft KV 的 MTP**。普通 MTP/EAGLE 的 draft 维护自己的 draft KV 并每步推进 position；FROZEN_KV_MTP 完全不分配 draft KV，**只读 target 的 KV**，把 rope position **冻结**在最后一个 target 槽。专为 Gemma4 assistant draft 设计（NEXTN/EAGLE + Gemma4 draft 自动落此）。

**核心文件**：[`frozen_kv_mtp_worker_v2.py`](../../../python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py)、`frozen_kv_mtp_info.py`、`frozen_kv_mtp_utils.py`、`frozen_kv_mtp_cuda_graph_runner.py`

- `frozen_kv_mtp_info.py:FrozenKVMTPContext`（frozen dataclass）：绑定 target `token_to_kv_pool` + `physical_layer_ids` 映射，取值走 `get_physical_layer_id()`。
- `frozen_kv_mtp_info.py:FrozenKVMTPDraftInput(EagleDraftInput)` / `FrozenKVMTPVerifyInput(EagleVerifyInput)`：**故意继承 EAGLE 输入**以复用 EAGLE 契约。
- `frozen_kv_mtp_utils.py:frozen_kv_target_view()` / `target_kv_pool_view()`：上下文管理器，draft 前向期间把 `draft_attn_backend.token_to_kv_pool` 换成 target 池。
- `frozen_kv_mtp_utils.py:set_frozen_kv_positions()`：**"rope phase = 最后写入的 target 槽，不随 draft 步推进"** → `positions = clamp(seq_lens-1, min=0)`。**这就是 KV "冻结" 的含义**：draft 从不写新 KV、从不推进 position，只在固定 position 重读 target 已写的 KV。同文件还有 `expand_for_topk_draft()` / `position_for_batch()` / `select_last_extend_hidden()`。
- `frozen_kv_mtp_worker_v2.py:FrozenKVMTPDraftWorker(EagleDraftWorkerBase, TpModelWorker)`：**不拥有 KV 池**；`__init__` 里绑 target embed/head。`draft_extend()` 是**空 `pass`**——draft-extend 纯粹选"最后接受 token 的 hidden 作下轮种子，不跑前向"（`_build_seed_draft_input()`，两条 `_draft_extend_for_prefill()` / `_draft_extend_for_decode()` 都直接调它）。
- `frozen_kv_mtp_worker_v2.py:FrozenKVMTPWorkerV2(EAGLEWorkerV2)`：**不调用 `EAGLEWorkerV2.__init__`**（避免建 draft KV 池）；断言不支持自适应。
- `frozen_kv_mtp_cuda_graph_runner.py:FrozenKVMTPCudaGraphRunner`：捕获时把 `token_to_kv_pool` 换成 frozen target 池。

**与 MTP/EAGLE 对比**：普通 MTP draft 建自己的 KV、逐步推进 position；FROZEN_KV_MTP 零 draft KV、只读 target KV、position 冻结、draft-extend 无前向（选种子即可）。注意 `SpeculativeAlgorithm.is_eagle()` 对 FROZEN_KV_MTP 仍返回 True（`spec_info.py` 的谓词实现里留了 FIXME）。

---

## 16. Multi-Layer EAGLE

标准 EAGLE 用**一个** draft 模型递归跑 `num_steps` 步；Multi-Layer EAGLE 用 **N 个物理独立的 draft 层**（`draft_runner_list[step]`，每步一个 ModelRunner）。**chain-only**（无树分支）。用于 Step3p5 / MiMoV2 等多 MTP 层架构（模型侧的 `enable_multi_layer_eagle = True` 由 override 注册表打开，见 §5.5）。

**核心文件**：[`multi_layer_eagle_worker_v2.py`](../../../python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py)、`multi_layer_eagle_draft_extend_cuda_graph_runner.py`、`python/sglang/kernels/ops/speculative/multi_layer_eagle.py`

- `MultiLayerEagleDraftWorker(EagleDraftWorkerBase)`：`draft_runner_list` 每步一个独立 ModelRunner（取用走 `mtp_model_runner(step)`）；`__init__` 断言 `num_draft_tokens == num_steps + 1`（强制 chain-only）。
- `MultiLayerEagleDraftWorker.draft_forward()`：`i=0` 调一次 `select_top_k_tokens`，之后逐步读各层 `tree_info`；整理由 `_draft_forward_organize()` 做。
- `_draft_extend_for_prefill()` / `_draft_extend_for_decode()`：依次跑每个 `draft_runner_list[step]`，用
  [`kernels/ops/speculative/multi_layer_eagle.py:rotate_input_ids()`](../../../python/sglang/kernels/ops/speculative/multi_layer_eagle.py)
  在步间把输入窗口左移一位、末位写入新草稿 token（定宽窗口滑动，免重分配）。
  ⚠️ 老名字 `rotate_input_ids_triton` 已不存在；Python 包装叫 `rotate_input_ids()`，
  内核是同文件的 `rotate_input_ids_kernel`，CPU 变体在 `sgl_kernel/speculative.py:rotate_input_ids_cpu()`。
- 这个 worker 还额外维护一套 **boundary KV 修补**逻辑（`_init_boundary_kv_fix_state()` /
  `_compute_boundary_kv_locs_positions()` / `_seed_boundary_kv_stash()` /
  `_fill_boundary_kv_front_and_update_stash()`）——多层之间层边界的 KV 需要显式接缝。
- ⚠️ **`MultiLayerEagleWorkerV2` 直接继承 `BaseSpecWorker`**，不是继承 `EAGLEWorkerV2`。
  它的 `verify()` 与单层共用 `eagle_worker_common.py:run_eagle_verify()`，只是传
  `metadata_ready_pre_pad=True, finalize_tree_path=False`（省去 topk>1 树压缩，因为它本来就无树）。

---

## 17. 自适应投机

动态调 `num_steps`：观测每 batch-size 桶的实际接受长度（EMA），draft 一直被接受就多跑几步、早早被拒就少跑，避免浪费。仅 **EAGLE/EAGLE3 + topk==1**——不支持的原因由
[`adaptive_spec_params.py:adaptive_unsupported_reason()`](../../../python/sglang/srt/speculative/adaptive_spec_params.py)
逐条给出（禁 DP-attn / multi-layer / TBO / pdmux 等）；LoRA 侧也直接禁自适应（见 §5.5）。

**核心文件**：`adaptive_runtime_state.py`、`adaptive_spec_params.py`

- `adaptive_runtime_state.py:SpecRuntimeState`：**原子切换单元**——打包 `num_steps`/`num_draft_tokens` + draft/verify/extend attn 后端 + 各自 cuda-graph runner。换这个 struct 就同步换整套 draft 配置。worker 侧要实现的契约写成了 `AdaptiveSpecWorker` Protocol（`build_adaptive_runtime_state()` + `apply_runtime_state()`）。
- `adaptive_runtime_state.py:AdaptiveController`：`init_states()` 为每个候选步数预建一个 `SpecRuntimeState`（`register()` 登记）；`on_verify_complete()` 喂观测接受长度 EMA 并切换；`_activate()` 调 `worker.apply_runtime_state()`；`activate_step_by_batch()` 按 batch size 选档。
- `adaptive_spec_params.py:AdaptiveStepSlot`：per-BS EMA。`_recompute_params()` 里的公式是
  `target_steps = clamp(round(ema_accept_len)+1, min, max)`，带 up/down 迟滞（hysteresis）与天花板；
  EMA 平滑防抖，`update_interval`（默认 5 批）才重算，`warmup_batches`（默认 10）预热。
  外层门面是 `AdaptiveSpeculativeParams`（`get_steps_for_batch()` / `cuda_graph_bs_for_step()` /
  `_pad_to_cuda_graph_bs()`）。

接入点：`base_spec_worker.py:BaseSpecWorker.on_verify_complete_cpu()` 在 verify 完成、accept 计数到 CPU 后被调，喂控制器（不强制 worker 热路径同步）；`BaseSpecWorker.activate_step_by_batch()`（EAGLE 侧覆写在 `eagle_worker_v2.py:EAGLEWorkerV2.activate_step_by_batch()`）每轮 draft 前按 batch size 切最优步数。候选步数集合的合法性在参数解析期就校验——`speculative_num_steps` 不在候选集合里会直接 raise（见 §5.4）。

---

## 18. 分离式投机

**Decoupled Spec IO**（实验特性）：把 drafter 与 verifier 拆成**两个独立引擎**，经 ZMQ IPC mesh 通信。verifier 是已确认 token 的真理源，发控制消息（sync/commit/close）；drafter 流式回传 tail token。

**核心文件**：[`decoupled_spec_io.py`](../../../python/sglang/srt/speculative/decoupled_spec_io.py)（消息 schema）。

**配置**（注意命名空间不同，⚠️ 旧文档写"配置在 `server_args.py:1601-1629`"，已改为按 flag 名索引）：

| flag | 命名空间 | 含义 |
|---|---|---|
| `--decoupled-spec-role {null,verifier,drafter}` | `NS("disagg")` | `null` 关闭；`verifier` 跑 target/verify 半边；`drafter` 跑 draft 半边 |
| `--decoupled-spec-bind-endpoint` | `NS("disagg")` | 本引擎监听端点 |
| `--decoupled-spec-connect-endpoints` | `NS("disagg")` | 对端端点列表（mesh） |
| `--decoupled-spec-rank` | `NS("disagg")` | 本引擎在自己 role 空间内的 rank |
| `--spec-trace-dir` | `NS("spec")` | 分离式投机的 trace 落盘目录 |

`server_args.py` 的校验：`decoupled_spec_role != "null"` 时，bind-endpoint / connect-endpoints / rank **三者缺一即 raise**。

消息 schema（全部在 `decoupled_spec_io.py`，按符号索引）：

| 消息（符号） | 方向 | 含义 |
|---|---|---|
| `DraftReqKey`（frozen） | — | `(src_verifier_rank, request_id)`，跨 verifier 消歧；配套 `build_draft_scheduler_rid()` / `parse_draft_scheduler_rid()` 做 rid 双向转换 |
| `DraftSync` | verifier→drafter | 从 verifier 前缀开/重开一个 drafter 请求；`draft_key()` 取键 |
| `VerifyCommit` | verifier→drafter | 提交一段连续输出（drafter 须对齐/截断/重 prefill）；`validate_committed_tokens()` 自检 |
| `DraftClose` | verifier→drafter | 关闭请求 |
| `DraftControlBatch` | verifier→drafter | 控制消息封套（sync / commit / close 打包发） |
| `DraftTailStreamOutput` / `DraftTailStreamOutputBatch` | drafter→verifier | 一个（批）流式 tail token + 基准前缀长度 |
| `VerifierCommitSegment` | drafter 侧状态 | 把连续 `VerifyCommit` 拼成一段（`append_message()` / `end_committed_len()` / `extract_prefix()`） |
| `DraftControlInbox` | drafter 侧状态 | TokenSync 线程用 `add_control_batch_locked()` / `add_verify_commit_locked()` / `add_close_key_locked()` 填，drafter 调度每步用 `extract_ready_controls_locked()` 取（返回 `ReadyDraftControls`） |
| `DraftMeshMessage` | 传输封套 | `from_control_batch()` / `from_tail_stream_output_batch()` 两个工厂 |
| `DraftMeshMessageType` | 枚举 | 封套类型标签 |

`DecoupledSpecIpcConfig` = `(bind_endpoint, connect_endpoints, rank)`。仍在演进（TODO 提及 "phase 5.c" 降级为自回归），有专门测试 `test/registered/unit/spec/test_decoupled_spec_io.py`。

## 19. 调度器集成

投机解码不改 `Scheduler` 的事件循环骨架，而是在若干"钩子点"把 target worker **替换/包装**成 draft worker，并在批结果里多记账几个字段。

### 19.1 初始化链（[`scheduler.py`](../../../python/sglang/srt/managers/scheduler.py)）

| 步骤 | 位置（符号） | 动作 |
|---|---|---|
| 解析算法 | `Scheduler.__init__` | `self.spec_algorithm = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)`；`is_none()` 时全程走普通路径 |
| 建 draft worker | `Scheduler.maybe_init_draft_worker()` | `is_none()` → `draft_worker=None`；否则 `DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)` 再实例化 |
| 分配内存池 | `Scheduler.init_memory_pools()` | 先 `init_target_memory_pool()`，再把 target 的 `req_to_token_pool` / `token_to_kv_pool_allocator` 传给 `draft_worker.alloc_memory_pool(...)`（EAGLE 系会额外建 draft KV 池；NGRAM/FROZEN_KV_MTP 不建），最后 `draft_worker.init_hicache_draft_plan()`（见 §20.5） |
| 建后端/图 | `Scheduler.init_all_attention_backends()` / `Scheduler.init_all_cuda_graphs()` | target 与 draft 各调一次；draft 侧自己建 draft-decode 与 draft-extend 两套 |
| **偷梁换柱** | `Scheduler.init_model_worker()` | `self.model_worker = self.tp_worker if self.spec_algorithm.is_none() else self.draft_worker`。draft worker **内部持有** target runner，对外表现为 `model_worker`，所以事件循环调 `model_worker.forward_batch_generation` 时其实进的是投机三阶段 |
| FutureMap | `Scheduler.init_overlap()` | `self.spec_algorithm.create_future_map(...)`：overlap 的负数占位符解析在 spec 下要适配草稿 token 布局 |

### 19.2 两条事件循环共用同一个 worker + grammar barrier

`event_loop_normal`（non-overlap）与 `event_loop_overlap`（overlap）都调 `model_worker.forward_batch_generation`，进入 `BaseSpecWorker.forward_batch_generation`。**没有 spec 专属 event loop**——这是 V2 相对 V1 的关键简化。

⚠️ **重要更正**：旧文档说"spec + grammar + decode + 有在途结果 ⇒ overlap 一律强制降级"。现在这条**只对不支持 grammar overlap 的算法成立**，支持的算法改走 **grammar barrier**，不再降级。

问题的本质：语法约束（JSON / regex / EBNF）要给 draft token 算 vocab bitmask，而 bitmask 依赖**上一批已确认 token** 推进后的 FSM 状态。overlap 模式下上一批结果还躺在 `result_queue` 里没处理，FSM 就还没推进。两种解法：

| 解法 | 适用算法 | 代价 |
|---|---|---|
| **同步降级**（老办法，保留） | `supports_grammar_overlap()` 为 False 的算法。当前**只有 NGRAM**——它的草稿来自 host 侧语料查表，本来就没有 GPU draft 相位可以藏 CPU 工作，源码注释写的是 "stays synchronous by design" | 本批退回同步执行，丢掉一拍 overlap |
| **grammar barrier**（新办法） | `supports_grammar_overlap()` 为 True，即 `is_eagle() or is_standalone() or is_dflash_family()`。注意 `is_eagle()` **含 FROZEN_KV_MTP**，Multi-Layer EAGLE 的枚举也是 EAGLE/EAGLE3 ⇒ 除 NGRAM 外全走这条 | 无（CPU 侧推进与 target verify forward 重叠） |

判定谓词落在 `ScheduleBatch` 上：

```python
# schedule_batch.py:ScheduleBatch.grammar_needs_sync()
return self.has_grammar and not self.spec_algorithm.supports_grammar_overlap()
```

- **降级判定**：`scheduler.py:Scheduler.is_disable_overlap_for_batch()` 里的 `need_grammar_sync` = `spec 非 none` **and** `batch.grammar_needs_sync()` **and** `forward_mode.is_decode()` **and** `len(self.result_queue) > 0`。注意中间那项已经是 `grammar_needs_sync()`——支持 overlap 的算法直接短路掉。源码注释明确写了这是"host-draft 算法的永久路径，不是待迁移的临时逻辑"。
- **barrier 注入**：`Scheduler.event_loop_overlap()` 里，`supports_grammar_overlap()` 为 True 时把 `fwd_kwargs["grammar_barrier"] = self._advance_pending_grammar` 一起传进 `forward_batch_generation`。
- **barrier 本体**：`Scheduler._advance_pending_grammar()` 遍历 `result_queue`，对每个未处理结果调 `batch_result_processor.py:BatchResultProcessor.advance_grammar_fsm()`。它**幂等**（per-req memoization，队列空或无 grammar 时是 no-op）。
- **调用时机**：worker 在 verify 内部、**`generate_token_bitmask()` 之前**调 `barrier()`，于是 CPU 侧 FSM 推进与 GPU 侧 target verify forward 重叠。参数名 `grammar_barrier` 一路透传：`EAGLEWorkerV2.forward_batch_generation()` → `EAGLEWorkerV2.verify()` → `eagle_worker_common.py:run_eagle_verify(..., barrier=grammar_barrier)`；DFLASH/DSPARK/Multi-Layer/FROZEN_KV_MTP 各自的 `verify()` 同签名。

**GrammarTree**：bitmask 要按草稿的**树/链拓扑**逐节点算，`spec_utils.py:GrammarTree` 就是这层抽象，两个工厂：

| 工厂 | 用处 |
|---|---|
| `GrammarTree.from_device(next_token, next_sibling, verify_ids_2d)` | 树形（EAGLE topk>1）。**启动异步 D2H 拷贝**，所以必须在 target verify forward **发射之前**构造，才能把拷贝藏在 forward 后面 |
| `GrammarTree.from_linear_chain(verify_ids_2d)` | 链形（DFLASH / DSPARK / topk==1）。内部合成 `next_token` / `next_sibling` 再转调 `from_device()` |

调用点：`eagle_worker_common.py:run_eagle_verify()` 用 `from_device(...)`；`dflash_worker_v2.py` 与 `dspark_components/dspark_worker_v2.py` 用 `from_linear_chain(...)`（都带 `if batch.has_grammar else None`）。掩码生成本体是 `spec_utils.py:generate_token_bitmask()`。

- 结果回填：批结果里 `next_draft_input`（下一轮 draft 的 `EagleDraftInput`）被挂回 `batch.spec_info`，形成"本轮 verify 产出下一轮 draft 输入"的流水。

### 19.3 记账：bonus token 的加减法

[`batch_result_processor.py:BatchResultProcessor._resolve_spec_v2_tokens()`](../../../python/sglang/srt/managers/scheduler_components/batch_result_processor.py) 是 accept 记账的权威点：

```python
result.num_correct_drafts             = sum(accept_lens) - len(batch.reqs)  # 全批「纯 draft」命中数（去 bonus）
result.num_correct_drafts_per_req_cpu = [x - 1 for x in accept_lens]        # 每请求「纯 draft」命中数
```

- `accept_lens[i]` = 第 i 个请求本轮接受的 token 数（**含 bonus**，恒 ≥1）。
- 减 1 去掉 bonus → 得"draft 真正命中几个"。全批求和即 `num_correct_drafts`。
- per-req 侧再由 `BatchResultProcessor` 累到 `req.spec_num_correct_drafts` 并喂 `req.update_spec_correct_drafts_histogram()`。
- 这两个量正是 §22 里 `accept_rate`（用 `num_correct_drafts`，去 bonus）与 `accept_length`（用 `accept_lens`，含 bonus）的来源，务必区分。

## 20. 内存与 KV 预算

投机解码在 decode 阶段一步要写多个 token（draft chain / verify block），因此每步的 KV 预留远大于普通 decode 的 1。

⚠️ **文件已搬家**：核心公式**不再**在 `mem_cache/common.py`，现在集中在
[`mem_cache/allocation_sizing.py`](../../../python/sglang/srt/mem_cache/allocation_sizing.py)。
该文件的所有函数都是**无参**的——它们直接读 `runtime_context` 的 bag（`get_spec()` / `get_schedule()` / `get_parallel()`），因为自适应投机会在 publish 之后改 `num_steps` 与草稿 token 上界，而这些函数每个 decode 批都要重算。

### 20.1 页大小口径 `get_alloc_page_size()`

```python
return get_schedule().page_size * get_parallel().attn_dcp_size
```

注意乘了 `attn_dcp_size`（DCP 分支的对齐口径）；跳过 DCP 的平台分配器页更小，所以这是**上界**。下面所有公式里的 `page_size` 都指这个值，不是裸 `--page-size`。

### 20.2 每步分配长度 `get_alloc_len_per_decode()`

```
无 spec（speculative_algorithm is None）：      return 1
page_size==1 或 topk==1 或 not has_draft_kv()： max(steps*topk, num_draft_tokens)
page_size>1 且 topk>1（树）：                    max(ceil((page-1+steps+page-1)/page) * page * topk, num_draft_tokens)
```

| 分支 | 场景 | 直觉 |
|---|---|---|
| `max(steps*topk, tokens)` | 链式，或 page=1，或 NGRAM（`has_draft_kv()` 为 False） | draft chain 与 verify block 共用这段预留；取二者最大 |
| 页对齐树公式 | spec v2 树（page>1 且 topk>1） | 每条 topk 分支各占一份**页对齐**足迹（分支各自复制），尾页最坏浪费 `page-1`；PR #26972 修的就是这个"带洞"树足迹超出默认 headroom |

三个入参都有兜底：`spec_steps = speculative_num_steps or 1`、`spec_topk = speculative_eagle_topk or 1`、`spec_tokens = max_speculative_num_draft_tokens()`。

### 20.3 双缓冲 `2×`：`get_alloc_reserve_per_decode()`

```python
return 2 * get_alloc_len_per_decode()
```

`2×` 是 **overlap 模式**下吸收 `kv_committed_len` 滞后一拍（N-1 记账）的双缓冲——源码注释指向 `eagle_utils.py:eagle_prepare_for_decode()`。overlap 下当前批发射时上一批的 committed 长度还没落地，若只留 1× 会在边界撞车。

配套的整页化工具是同文件的 `page_aligned_decode_alloc_lens(reqs, *, reserve, page_size)`：把每请求的下一次分配长度按
`max(kv_allocated_len, ceil((kv_committed_len + reserve)/page)*page)` 取整，保证"分配的 == 记录的"（page>1 时不对齐的尾巴会漏）。

### 20.4 row headroom `get_req_to_token_extra_context_len()`

`req_to_token` 每行在模型上下文长度之外的余量：

```python
extra = 4 + (max_speculative_num_draft_tokens() or 0)          # 基础余量
if get_spec().speculative_algorithm is not None and page_size > 1:
    extra = max(extra, get_alloc_reserve_per_decode() + page_size - 1)
```

⚠️ **两处更正**（旧文档写错了）：

1. 第二行是 `max(extra, reserve + page_size - 1)`，**不是** `max(extra, reserve)`。多出的 `page_size - 1` 是因为 `kv_allocated_len` 已经页对齐（见 `eagle_utils.py:eagle_prepare_for_decode()`），逼近上下文上限时对齐后的预留最坏会**超出 `page_size - 1`**；没有这份余量，行写入会静悄悄落到**相邻行**里去。
2. 门禁条件只有 **`spec 非 None` + `page_size > 1`** 两项，**没有 `topk > 1`**。链式（topk==1）在 page>1 下同样吃这条 headroom。

### 20.5 HiCache draft plan：draft KV 池怎么被 HiCache 接管

开了层级缓存（或 decode 侧 `host_pool` 回退备份）之后，draft 侧的 KV 池也要挂到 HiCache 上。这套决策落在
[`base_spec_worker.py`](../../../python/sglang/srt/speculative/base_spec_worker.py) 的三个符号 + 一个入口：

| 符号 | 作用 |
|---|---|
| `HiCacheDraftMode` | 枚举三值：`NONE` / `PACKED` / `SIDECAR` |
| `HiCacheDraftPlan`（frozen） | `(mode, device_pools)`——模式 + 参与的 device 侧 KV 池元组 |
| `_can_pack_hicache_mtp(spec_algorithm, draft_runners)` | 判定能否 **packed**：①"NEXTN 型 MTP"= `is_eagle() and not is_eagle3()` 且**所有** draft runner 的 `model_config.num_nextn_predict_layers` 非空；② DSPARK 且 draft 架构是 `DeepseekV4ForCausalLMDSpark`。二者取或 |
| `BaseSpecWorker._build_hicache_draft_plan()` | 真正的决策函数 |
| `BaseSpecWorker.init_hicache_draft_plan()` | 唯一写入点，由 `Scheduler.init_memory_pools()` 在 `alloc_memory_pool()` 之后调用（见 §19.1） |

`_build_hicache_draft_plan()` 的判定顺序：

| 顺序 | 条件 | 结果 |
|---|---|---|
| 1 | `enable_hierarchical_cache` 为假 **且** `disaggregation_decode_retraction_backup != "host_pool"` | `HiCacheDraftPlan()`（即 `NONE`） |
| 2 | 没有 draft model runner | `NONE` |
| 3 | draft 架构含 `InklingForConditionalGenerationMTP` | **raise `NotImplementedError`**（HiCache 尚不支持 Inkling MTP 草稿状态） |
| 4 | `_can_pack_hicache_mtp(...)` 为真 | `PACKED`，且把 `target_model_runner.mtp_draft_device_pools` 设为**全部** draft 池 |
| 5 | 其余 | `SIDECAR`，只登记 **第一个** draft 池（`draft_pools[:1]`，保留 multi-layer EAGLE 的旧行为） |

注意函数开头无条件先把 `target_model_runner.mtp_draft_device_pools = ()` 清空，只有 `PACKED` 分支才会填回去——所以这个字段能直接当"是否 packed"的判据。

### 20.6 与 §6.4 skip-last-forward 的关系

draft 多步循环里最后一步不 forward（`eagle_worker_v2.py:EagleDraftWorker.draft_forward()`：草稿 prefill 已给出 1 个 token，这里再补 `steps-1` 个），所以实际写入 KV 的草稿 token 数 = `steps`，与上面的预留口径一致。树形下每分支还要复制前缀尾部（`eagle_worker_common.py:duplicate_prefix_tail_to_draft_branches()`），故按 `topk` 倍预留。

> ⚠️ 旧文档把这一节写成"与 §8 skip-last-forward 的关系"，且把 `duplicate_prefix_tail_to_draft_branches` 记在 `base_spec_worker.py` 下——skip-last-forward 在 §6.4，该函数已搬到 `eagle_worker_common.py`。

## 21. 注意力后端支持矩阵

投机解码的 draft 阶段有两个专属后端（**draft-decode** 多步链/树；**draft-extend** 草稿 prefill），由
[`draft_utils.py:DraftBackendFactory`](../../../python/sglang/srt/speculative/draft_utils.py) 按 target 后端映射创建。是否支持 `topk>1`（树）是关键分水岭。

### 21.1 后端 topk 能力

| 后端 | topk==1（链） | topk>1（树，page>1） | 说明 |
|---|---|---|---|
| **flashinfer** | ✅ | ✅ | 在 `_PAGE_TREE_SPEC_BACKENDS` |
| **fa3**（FlashAttention 3） | ✅ | ✅ | 在 `_PAGE_TREE_SPEC_BACKENDS` |
| **triton** | ✅ | ✅ | 在 `_PAGE_TREE_SPEC_BACKENDS` |
| **trtllm_mha** | ✅ | ❌ | `speculative_hook.py:_handle_eagle_family()` 里显式 raise：`topk > 1` 不支持 |
| **flashmla / trtllm_mla / cutedsl_mla** | ✅ | ❌ | MLA 族无法表达 per-branch 树；draft-decode 由 `DraftBackendFactory._create_trtllm_mla_decode_backend(backend="trtllm-gen")` 回退 |
| **aiter / ascend / dsa / dsv4 / fa4 / intel_amx / tokenspeed_mla** | ✅ | ⚠️ | 仅链式安全；树需落在 `_PAGE_TREE_SPEC_BACKENDS` |

**核心约束**（`speculative_hook.py:_handle_eagle_family()` 末尾）：

```python
_PAGE_TREE_SPEC_BACKENDS = ("flashinfer", "fa3", "triton")
# topk>1 且 page_size>1 且 attention_backend 不在上面三个里 ⇒ raise ValueError
```

原因（源码注释）：`topk>1 + page>1` 需要两遍级联 draft-decode（共享前缀 pass + per-branch 展开 pass，带前缀尾复制），只有这三个后端实现了 per-branch 树；`flashmla` / `trtllm_mla` / `cutlass_mla` 无法表达，直接拒绝。要用其他后端跑树，只能 `page_size==1`。

DFLASH 另有一条独立的 draft 后端解析：`speculative_hook.py:_resolve_dflash_draft_attention_backend()`。白名单是 `("flashinfer","fa3","fa4","triton","trtllm_mha","ascend")`，兜底 backend 在 ROCm 上是 `triton`、CUDA 上是 `flashinfer`；`trtllm_mha` 只在"draft 全部层都是 sliding attention 或 draft 显式 `is_causal`"时才留用，否则打 warning 回退。

### 21.2 draft 双后端的创建

| 方法（符号） | 作用 | steps≤1 时 |
|---|---|---|
| `DraftBackendFactory.create_decode_backend()` | draft 多步 decode 后端 | 返回 `None`（`speculative_num_steps <= 1` 时既没有多步，也包含 steps==0 的 nospec 情形） |
| `DraftBackendFactory.create_draft_extend_backend()` | 草稿 prefill（extend）后端 | 始终创建 |

后端选择优先级在 `DraftBackendFactory._create_backend()` 里：`self.draft_attn_backend`（即 `--speculative-draft-attention-backend`，显式）> `attention_backends()` 拆出的 `decode/prefill_attention_backend` > 基础 `attention_backend` 兜底。命中后统一给 backend 及其 `attn_backends` 子后端盖上 `prefill_attention_backend_str` / `decode_attention_backend_str` 戳；draft-decode 还会过一层 `attn_backend_wrapper_for_draft_decode()`。

`create_decode_backend()` 返回的是**每步一个后端的容器**（不是单个 `AttentionBackend`），所以 `attn_backend_wrapper_for_draft_extend` 给不了它 conv sidecar——`_assert_draft_needs_no_conv_sidecar()` 就是守这个前提的。

### 21.3 cache-loc 布局随 topk 分叉

`eagle_worker_common.py:prepare_for_draft()` 里判据是 **`if page_size == 1 or topk == 1:`** ——走**连续** cache-loc（`kernels/ops/speculative/cache_locs.py:assign_draft_cache_locs_contiguous()`）；否则每分支独立 paged loc（树需要），并配 `duplicate_prefix_tail_to_draft_branches()`。这决定了 draft KV 的写入地址形态。

> ⚠️ 旧文档把这段写成"`base_spec_worker.prepare_for_draft`（:176/:199）"与"`triton_ops/cache_locs.py:144`"——函数已搬到 `eagle_worker_common.py`，`triton_ops/` 目录整体搬到了 `python/sglang/kernels/ops/speculative/`。（此处保留旧写法仅为方便对照旧版文档。）

## 22. 指标与可观测性

投机解码有**两个**核心指标，只差一个 bonus token，但含义完全不同，是最易混淆的地方。权威公式在
[`scheduler_components/metrics_reporter.py`](../../../python/sglang/srt/managers/scheduler_components/metrics_reporter.py)。

### 22.1 两个核心指标的定义

| 指标 | 符号 | 公式（`MetricsReporter.report_decode_stats()`） | 含义 | 下界 |
|---|---|---|---|---|
| **accept length** | τ | `spec_num_accept_tokens / spec_num_forward_ct` | 每次 target forward **平均产出多少 token（含 bonus）** | ≥1 |
| **accept rate** | α | `num_correct_drafts / total_draft_tokens` | draft **提出的 token 中被接受的比例（不含 bonus）** | ≥0 |

其中（同一函数内）：
```
num_correct_drafts = spec_num_accept_tokens − spec_num_forward_ct     # 去掉每轮 1 个 bonus
draft_per_round    = num_draft_tokens − 1  （若为 0 则退用 num_steps）  # 每轮提出多少草稿
total_draft_tokens = spec_num_forward_ct × draft_per_round
```

### 22.2 累加口径 `MetricsReporter.update_spec_metrics()`

签名已经扩到 4 个参数（后两个默认 0，只有 DSPARK 会填）：

```python
def update_spec_metrics(self, bs, num_correct_drafts,
                        num_block_accept_tokens=0, num_cap_tokens=0):
    self.spec_num_accept_tokens       += num_correct_drafts + bs   # 每请求补回 1 个 bonus ⇒ 含 bonus 的接受量
    self.spec_num_forward_ct          += bs                        # forward 次数（按请求计）
    self.spec_num_block_accept_tokens += num_block_accept_tokens   # 新增
    self.spec_num_cap_tokens          += num_cap_tokens            # 新增
    self.num_generated_tokens         += num_correct_drafts        # bonus 在别处单独计
```

- `num_correct_drafts + bs`：每个请求补 1 个 bonus，得到"含 bonus 接受总量"→ 分子给 τ。
- 关系式：**`τ = α × draft_per_round + 1`**（每轮期望接受 = 提出数×接受率 + 1 个 bonus）。这就是为什么 τ 恒 ≥1 而 α 可以为 0。

### 22.3 新增的两个 DSPARK/ragged 指标

⚠️ 旧文档只写了 τ 和 α 两个量，现在**多了两个**（都在同一个 `report_decode_stats()` 里算）：

| 指标 | 公式 | 语义 | 生效条件 |
|---|---|---|---|
| **cap length** | `spec_num_cap_tokens / spec_num_forward_ct` | 每 verify 步平均被 **cap 截掉** 多少草稿 token（ragged verify 的窗口裁剪量） | `spec_num_forward_ct > 0`；日志里 `> 0` 才打印 |
| **block accept length** | `spec_num_block_accept_tokens / spec_num_forward_ct` | 每 verify 步**未被 cap 截断时**的"整块"接受长度 = 已接受 + 被 cap 裁掉的草稿 | 额外要求 `read_ragged_verify_mode() is RaggedVerifyMode.CAP_ACCEPT`，**否则恒为 0**（见 §14.6） |

数据源在 DSPARK 侧：`dspark_components/dspark_worker_v2.py` 填 `GenerationBatchResult` 的两个字段——
`block_accept_lens = accept.commit_lens + accept.cap_trim_lens`、`cap_lens = ...`（idle 批填空张量）。
两个字段定义在 `managers/utils.py:GenerationBatchResult`，且都走 `_async_d2h()` 异步搬到 host。
`batch_result_processor.py:BatchResultProcessor._resolve_spec_v2_tokens()` 里 `.tolist()` 后求和成
`result.num_block_accept_tokens` / `result.num_cap_tokens`，再喂 `update_spec_metrics()`。

`sglang:spec_block_accept_length` 的 Gauge 文档串把口径写得很直白："Mean uncapped full-block accept length per verify step (accept + cap-trimmed drafts; **exact only in DSpark cap-accept mode**)"。

### 22.4 Prometheus 导出

| Gauge | 定义处（符号） | 来源 |
|---|---|---|
| `sglang:spec_accept_length` | `metrics_collector.py:SchedulerMetricsCollector.__init__` | `stats.spec_accept_length` |
| `sglang:spec_accept_rate` | 同上 | `stats.spec_accept_rate` |
| `sglang:spec_cap_length` | 同上 | `stats.spec_cap_length` |
| `sglang:spec_block_accept_length` | 同上 | `stats.spec_block_accept_length` |
| `sglang:spec_num_steps` | 同上 | `stats.spec_num_steps`（**当前生效**的 `speculative_num_steps`，自适应下会变） |
| `sglang:spec_num_draft_tokens` | 同上 | `stats.spec_num_draft_tokens` |

字段都声明在 `metrics_collector.py:SchedulerStats`，统一由 `SchedulerMetricsCollector.log_stats()` 逐个 `_log_gauge()`。

日志行（`report_decode_stats()`）：`accept len: X.XX, accept rate: Y.YY, `，后面按需追加 `cap len: ...` / `block accept len: ...`；DSPARK 还会再拼 `draft_worker.block_accept_estimate_log_suffix()`（见 §14.8）。每次上报后
`spec_num_accept_tokens = spec_num_forward_ct = 0`、`spec_num_block_accept_tokens = spec_num_cap_tokens = 0` 滑动清零，故都是**区间**均值（累计量另存在 `spec_total_num_accept_tokens` / `spec_total_num_forward_ct`）。

### 22.5 per-request 累计与直方图

`batch_result_processor.py:BatchResultProcessor._resolve_spec_v2_tokens()` 里逐请求累：

| 请求字段（`schedule_batch.py:Req`） | 累加方式 |
|---|---|
| `req.spec_num_correct_drafts` | `+= num_correct_drafts_per_req_cpu[i]`，并 `req.update_spec_correct_drafts_histogram(...)` |
| `req.spec_num_block_accept_tokens` | `+= block_accept_lens[i]`（`block_accept_lens is not None` 时） |
| `req.spec_num_cap_tokens` | `+= cap_lens[i]`，并 `req.update_spec_cap_lens_histogram(cap_lens[i])` |

`Req.update_spec_cap_lens_histogram()` 是"按需扩容的桶数组"：`spec_cap_lens_histogram` 长度不够就补 0 补到 `cap_len`，再 `[cap_len] += 1`。

一路回传到用户：`output_streamer.py` 收集 `spec_num_cap_tokens` / `spec_cap_lens_histogram` 列表 →
`io_struct.py` 的 IPC 字段 → `detokenizer_manager.py` 转发 → `tokenizer_manager.py` 填进 `meta_info`（`spec_accept_rate` / `spec_accept_length` / `spec_block_accept_length` / `spec_cap_lens_histogram`）→ `entrypoints/openai/protocol.py` 的响应字段。多 tokenizer 场景由 `multi_tokenizer_mixin.py:_extract_field_by_index()` 按 index 切片。

## 23. 命名规范

投机解码代码有一套强制命名约定（见 `.claude/skills/speculative-naming/SKILL.md`）。**改任何 `python/sglang/srt/speculative/` 下的标识符、相关注意力后端、调度器累加器、IPC 字段、指标、CLI flag 前必须遵守**，否则 review 不过。

| # | 规则 | 反例 | 正例 |
|---|---|---|---|
| 1 | **动词原形，去 `-ed`** | `num_accepted_tokens` | `num_accept_tokens` |
| 2 | **那个 "+1" token 叫 `bonus_token`** | `verified_id` | `bonus_token` / `bonus_tokens` |
| 3 | **`accept_*` 含 bonus；`correct_*` 不含** | 用名词区分 | 用动词区分：`num_accept_tokens`（含）/ `num_correct_drafts`（不含） |
| 4 | **`num_` 计数 / `_ct` 计数器 / `_rate` 比率 / 无前缀=内容数组** | `num_X_ct`、`num_accept_rate`（混用） | `num_correct_drafts`、`spec_verify_ct`、`accept_rate`、`accept_tokens` |
| 5 | **spec 作用域内去掉冗余 `_token_id`** | `accepted_token_ids`、`curr_token_id` | `accept_tokens`、`current_token` |
| 6 | **非标量张量用复数，标量用单数** | — | `bonus_tokens: Tensor[bs]`（复）/ kernel 内 `bonus_token = tl.load(...)`（单） |

**关键例外**（Rule 3 的豁免）：`accept_rate`（α，**不含** bonus，= `correct_drafts/proposed`）与 `accept_length`（τ，**含** bonus，= `completion_tokens/verify_ct`）沿用论文/外部字段惯例，其语义由论文定义而非 Rule 3。但**内部计数器**仍守严格语义：`num_correct_drafts`（无 bonus）、`num_accept_tokens`（含 bonus）。

**保持原样**（不在本规范作用域）：PyTorch 生态名（`seq_lens`/`cu_seqlens_q`）、多模态词表名（`image_token_id`/`eos_token_id`）、请求级状态（`req.input_ids`/`req.output_ids`/`next_token_ids`）、冻结的 C++ kwargs（`accept_token_num`）、非 token 的 ID（`req_id`/`layer_id`）。

## 24. 方案对比总表

### 24.1 六大算法家族横向对比

| 维度 | EAGLE/EAGLE3 | STANDALONE | NGRAM | DFLASH | **DSPARK** | FROZEN_KV_MTP |
|---|---|---|---|---|---|---|
| **draft 机制** | target 权重内 MTP 层 / 独立小头，复用 target hidden | 独立完整草稿模型 | 无模型，n-gram 表检索历史 | 从 target hidden 投影出 draft KV | DFLASH 结构 + **confidence 头 + SPS/STS 代价表**动态定长度 | MTP 层但**不建 draft KV** |
| **是否复用 target hidden** | ✅（核心） | ❌（独立前向） | ❌ | ✅ | ✅（`TargetHiddenKvInjector` 注入） | ✅ |
| **draft KV** | ✅ 建 | ✅ 建 | ❌ 无（`has_draft_kv()=False`） | 投影得到 | 投影得到（复用 `DFlashDraftInputV2`） | ❌ 冻结/不建 |
| **is_eagle()** | ✅ | ❌ | ❌ | ❌ | ❌（属 `is_dflash_family()`） | ✅（含 FIXME） |
| **树（topk>1）** | ✅ | ✅ | ✅（BFS breadth） | ❌ 仅链 | ❌ 仅链（hook 强制 `topk=1`） | 链/树看 topk |
| **典型 (steps,topk,tokens)** | Llama (5,4,8)；DeepSeek/GLM (3,1,4) | (3,1,4) | 由 max_bfs_breadth 推 | (1,1,·) | hook 钉死 `(1, 1, γ+1)`，γ 默认 7 | 依模型 |
| **verify 长度** | 固定 `num_draft_tokens` | 固定 | 固定 | 固定 | **每请求可变**（ragged verify，见 §14.6） | 固定 |
| **draft 模型路径** | 自动=model_path（MTP）或独立 | 必填独立路径 | 无 | =model_path | =model_path（或 target ckpt 自带 DSPARK 草稿） | =model_path |
| **grammar overlap** | ✅ | ✅ | ❌（**唯一**走同步降级的算法） | ✅ | ✅ | ✅（`is_eagle()` 含它） |
| **适用场景** | 通用首选，接受率最高 | 有现成小模型 | 代码/重复文本，零训练 | 省 draft KV 显存 | 想让草稿长度**随置信度自适应**、榨干每步 verify 预算 | Gemma4 等特定架构 |

> ⚠️ 旧文档标题写"五大算法家族"且没有 DSPARK 列，已补齐为六列。DSPARK 与 DFLASH 的逐项差异见 §14.9。

### 24.2 拓扑与接受方式

| 维度 | 链式 topk==1 | 树形 topk>1 |
|---|---|---|
| 候选数/步 | 1 条路径 | topk 条分支 |
| tree kernel | 共用 `eagle_utils.py:build_tree_kernel_efficient()` | 同左 |
| 贪心接受 | `verify_tree_greedy_func` | `verify_tree_greedy_func`（树掩码） |
| 采样接受 | `chain_speculative_sampling_triton`（需 `--speculative-use-rejection-sampling`） | `tree_speculative_sampling_target_only` |
| topk==1 专属快路 | `kernels/ops/speculative/topk1.py:draft_topk1_postprocess()` 绕开 `select_top_k_tokens`（见 §8.3） | 无 |
| cache-loc | 连续（`assign_draft_cache_locs_contiguous()`） | per-branch paged + 前缀尾复制 |
| 后端限制 | 全后端 | 仅 flashinfer/fa3/triton（page>1 时） |
| KV 预留 | `max(steps*topk, tokens)` | 页对齐树公式 × topk |
| 走链的算法 | DFLASH / DSPARK / Multi-Layer EAGLE 恒链；EAGLE 系 topk==1 时 | 仅 EAGLE 系 / STANDALONE / NGRAM |

### 24.3 accept_rate vs accept_length（最易错点，重申）

| | accept_rate (α) | accept_length (τ) |
|---|---|---|
| 含 bonus | ❌ 否 | ✅ 是 |
| 公式 | `num_correct_drafts / total_draft_tokens` | `num_accept_tokens / forward_ct` |
| 下界 | 0 | 1 |
| 换算 | — | `τ = α × draft_per_round + 1` |
| 直觉 | draft 命中比例 | 一次 forward 净产出 token |

另有两个新指标（`cap length` / `block accept length`）只在 ragged verify 下有意义，见 §22.3。

## 25. 文件索引

核心目录 `python/sglang/srt/speculative/`：**顶层 35 个 `.py`**，另有 `dspark_components/` 子包 **12 个**、`cpp_ngram/` 子包 2 个。
⚠️ 旧文档写"30 个文件"且未列 `dspark_components/`，已更新。

### 25.1 框架与分发

| 文件 | 作用 |
|---|---|
| `spec_info.py` | `SpeculativeAlgorithm` 枚举与全部 `is_*()`/`supports_*()` 谓词、`SpecInput`/`SpecInputType`、`create_worker()` 分发、模块级 `spec_scale_global_num_tokens()` / `create_dummy_verify_input()` |
| `spec_registry.py` | 插件注册（`CustomSpecAlgo` / `register_algorithm`），duck-type 校验 |
| `base_spec_worker.py` | `EagleDraftWorkerBase` / `BaseSpecWorker` 抽象基类、adaptive 钩子、`HiCacheDraftMode` / `HiCacheDraftPlan` / `_can_pack_hicache_mtp()`、`last_shared_read_runner` |
| `spec_utils.py` | `_select_top_k_tokens_later()` 树打分、`resolve_num_tokens_per_req(phase=...)` 唯一派生点、`GrammarTree` / `generate_token_bitmask()`、通用工具 |
| `draft_utils.py` | `DraftBackendFactory`：按 target 后端建 draft-decode/extend 后端 |
| `draft_worker_common.py` | **（新增）** 跨算法的 draft worker 装配件：`DraftWorkerBundle`、`build_draft_tp_worker()`、`make_draft_input_v2()` / `make_draft_block_spec_info()`、`build_block_pos_offsets()` |
| `ragged_verify.py` | **（新增）** ragged verify 全套：`RaggedVerifyMode`（STATIC/CAP_ACCEPT/COMPACT）、`RaggedVerifyLayout`、`RaggedTargetVerifyGeometry`、`VerifyExtendLengths`、`round_up_grid()`、`compute_target_verify_graph_key()` |
| `spec_tp_sync.py` | **（新增）** TP 同步点枚举与解析：`SpecTpSyncSite`、`SpecTpSync`、`parse_spec_tp_sync()` |
| `decoupled_spec_io.py` | Decoupled Spec IO 的 ZMQ 消息 schema（见 §18） |

### 25.2 EAGLE / EAGLE3

| 文件 | 作用 |
|---|---|
| `eagle_worker_v2.py` | `EagleDraftWorker(EagleDraftWorkerBase)` + `EAGLEWorkerV2(BaseSpecWorker)`：`forward_batch_generation` / `draft_forward` / `verify` 三阶段与 topk==1 快路 |
| `eagle_worker_common.py` | **（新增，必看）** 多算法共享的实现体：`run_eagle_verify()`（verify 主体已从 worker 搬到这里）、`prepare_for_draft()` / `prepare_for_draft_extend()`、`build_eagle_verify_input()`、`duplicate_prefix_tail_to_draft_branches()`、`_finalize_accept_tree_path()` / `_compact_accept_to_front()` |
| `eagle_info.py` | `EagleDraftInput` / `EagleVerifyInput` / `EagleDraftExtendInput`；`num_accept = correct + 1` 的权威注释在此 |
| `eagle_utils.py` | `build_tree_kernel_efficient()`（bonus 作树根）、`eagle_sample()`、`eagle_prepare_for_decode()`、`TreeMaskMode` |
| `eagle_draft_cuda_graph_runner.py` | draft-decode CUDA graph runner（`captured_req_width = resolve_num_tokens_per_req(phase="draft_decode")`） |
| `eagle_draft_extend_cuda_graph_runner.py` | draft-extend CUDA graph runner（`forward_mode = DRAFT_EXTEND_V2`，`shared_read_done_event` 归属点） |
| `eagle_disaggregation.py` | PD 分离下的 EAGLE 草稿输入构建 |

### 25.3 其他算法家族

| 文件 | 算法 |
|---|---|
| `standalone_worker_v2.py` | STANDALONE（独立草稿模型，继承 EAGLE 两个类） |
| `ngram_worker.py` / `ngram_info.py` / `external_corpus_manager.py` / `cpp_ngram/`（`ngram_corpus.py`、`external_corpus.py`） | NGRAM（无模型 n-gram 检索）+ 外部语料 |
| `dflash_worker_v2.py` / `dflash_info.py` / `dflash_info_v2.py` / `dflash_utils.py` / `dflash_disaggregation.py` | DFLASH（target hidden 投影 KV） |
| `dspark_components/`（12 个文件）+ `dspark_disaggregation.py` | **DSPARK**：`dspark_worker_v2.py`、`dspark_config.py`、`dspark_draft.py`、`dspark_draft_sampler.py`、`dspark_kv_inject.py`、`dspark_planner.py`、`dspark_verify.py`、`dspark_sps.py`、`dspark_sts.py`、`dspark_block_accept_estimator.py`、`dspark_observability.py`、`__init__.py`。逐文件职责见 §14.1 |
| `frozen_kv_mtp_worker_v2.py` / `frozen_kv_mtp_info.py` / `frozen_kv_mtp_utils.py` / `frozen_kv_mtp_cuda_graph_runner.py` | FROZEN_KV_MTP（冻结 KV 的 MTP） |
| `multi_layer_eagle_worker_v2.py` / `multi_layer_eagle_utils.py` / `multi_layer_eagle_draft_extend_cuda_graph_runner.py` | Multi-Layer EAGLE（N 个物理独立 draft 层，chain-only） |

### 25.4 自适应

| 文件 | 作用 |
|---|---|
| `adaptive_runtime_state.py` | `SpecRuntimeState`（原子切换单元）、`AdaptiveSpecWorker` Protocol、`AdaptiveController` |
| `adaptive_spec_params.py` | `AdaptiveStepSlot` per-BS EMA、`AdaptiveSpeculativeParams`、`adaptive_unsupported_reason()` |

### 25.5 Triton / AOT 内核（**已搬出 `speculative/`**）

⚠️ 旧文档把 `reject_sampling.py` 列在 `speculative/` 下，并称内核目录为 `speculative/triton_ops/`。**两者都已过时**：内核统一搬到
`python/sglang/kernels/ops/speculative/`。

| 文件（`python/sglang/kernels/ops/speculative/`） | 作用 |
|---|---|
| `reject_sampling.py` | `speculative_sampling_classic_kernel`（`coin*q<p`）、`chain_speculative_sampling_triton` |
| `eagle.py` | `fill_bonus_tokens()` / `fill_bonus_tokens_func()` 等 EAGLE 后处理 |
| `spec_tree.py` | 建树相关内核 |
| `cache_locs.py` | `assign_draft_cache_locs_contiguous()`、`generate_draft_decode_kv_indices()`、`rebuild_compact_draft_req_to_token()` 等 |
| `topk1.py` | `draft_topk1_postprocess()`：topk==1 链的快路后处理（见 §8.3） |
| `multi_layer_eagle.py` | `rotate_input_ids()`（内核 `rotate_input_ids_kernel`；CPU 变体在 `kernels/aot/python/sgl_kernel/speculative.py:rotate_input_ids_cpu()`） |
| `ragged_verify_kernels.py` | ragged verify 的布局/裁剪内核 |
| `dflash.py` / `fused_kv_materialize.py` | DFLASH/DSPARK 的 KV 投影与物化 |
| `gather_spec_extras.py` / `ngram_embedding.py` / `ngram_corpus.py` | 采样附加量收集、NGRAM 侧内核 |

### 25.6 外围集成（非 `speculative/` 目录）

| 文件 | 作用 |
|---|---|
| `arg_groups/speculative_hook.py` | 别名解析、`handle_speculative_decoding()` 统一入口、各算法 `_handle_*()`、默认推导、后端约束校验 |
| `arg_groups/pipeline.py` | `run_resolution_pipeline()`：驱动上面那个 hook 的解析流水线 |
| `managers/scheduler.py` | `maybe_init_draft_worker()`、`init_model_worker()` 的 worker 替换、`is_disable_overlap_for_batch()` 降级判定、`_advance_pending_grammar()` grammar barrier |
| `managers/schedule_batch.py` | `ScheduleBatch.grammar_needs_sync()`、`Req` 上的 spec 累计字段与直方图 |
| `managers/utils.py` | `GenerationBatchResult` 的 spec 字段（`block_accept_lens` / `cap_lens` / `extra_keep_alive_refs`） |
| `managers/scheduler_components/batch_result_processor.py` | `BatchResultProcessor._resolve_spec_v2_tokens()` accept 记账、`advance_grammar_fsm()` |
| `managers/scheduler_components/metrics_reporter.py` | `MetricsReporter.update_spec_metrics()`、`report_decode_stats()` 里的 accept_length/rate/cap/block 计算 |
| `observability/metrics_collector.py` | Prometheus gauge `sglang:spec_accept_length` / `spec_accept_rate` / `spec_cap_length` / `spec_block_accept_length` / `spec_num_steps` / `spec_num_draft_tokens` |
| `mem_cache/allocation_sizing.py` | **（原 `mem_cache/common.py`）** `get_alloc_page_size()` / `get_alloc_len_per_decode()` / `get_alloc_reserve_per_decode()` / `page_aligned_decode_alloc_lens()` / `get_req_to_token_extra_context_len()` |
| `layers/attention/*` | 各注意力后端的 draft 支持（`_PAGE_TREE_SPEC_BACKENDS` 三家） |

---

## 结语

SGLang 投机解码的设计精髓可归纳为三点：

1. **一套框架，多种算法**：V2 统一 `draft → verify → draft_extend` 三阶段，算法只需产出/消费统一的 `SpecInput`，通过 `create_worker()` 分发；插件经 `register_algorithm` 接入，无需改核心。共享实现体进一步下沉到 `eagle_worker_common.py` / `draft_worker_common.py`，DSPARK 就是"复用 DFLASH 结构 + 换调度策略"的产物。
2. **拓扑由 topk 决定**：`topk==1` 链、`topk>1` 树，共用 tree kernel；后端能力（尤其 page>1 的树）是主要约束边界。DSPARK 在链的基础上再让**每请求的 verify 长度可变**（ragged verify）。
3. **记账严谨**：bonus token 让 `accept_length ≥ 1`，与 `accept_rate` 差一个 bonus；命名规范（`accept_*` 含 bonus / `correct_*` 不含）把这个语义固化进标识符，避免全链路记账错位。

理解这三点，再配合 §25 的文件索引，即可快速定位任意投机解码相关代码并安全扩展。
