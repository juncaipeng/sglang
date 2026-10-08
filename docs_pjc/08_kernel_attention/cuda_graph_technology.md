# SGLang CUDA Graph 技术全解（由浅到深）

> 代码基线：本仓库 `main` 分支快照（commit `33ed29a0ee`）。所有 `文件:行号` 引用均以该快照为准。
>
> 历史提示：老版本的 `python/sglang/srt/model_executor/cuda_graph_runner.py` 与其中的
> `CudaGraphRunner` 巨类**已经不存在**了。现在的实现被拆成
> `model_executor/runner/`（阶段语义层）+ `model_executor/runner_backend/`（捕获机制层）两层。
> 如果你在别处看到 `CudaGraphRunner`、"`DecodeInputBuffers` 是唯一 buffer 体系"、
> 或者只有 `--disable-cuda-graph` / `--enable-piecewise-cuda-graph` 这几个开关，那些说法都过时了。

## 这篇文档怎么读

文档按"洋葱"组织，一层比一层深，每层都能独立读完并拿到可用结论：

| 层 | 章节 | 读完你能做什么 | 适合谁 |
|---|---|---|---|
| L0 认知层 | 第 1–2 章 | 知道 CUDA Graph 解决什么问题、为什么 decode 特别需要它 | 所有人 |
| L1 地图层 | 第 3–5 章 | 看懂目录结构、两层架构、四种 backend、图的"身份"是什么 | 想定位代码的人 |
| L2 机制层 | 第 6–10 章 | 讲清 capture/replay 每一步、buffer 三层体系、attention 契约、显存池 | 要改 CUDA Graph 的人 |
| L3 深水层 | 第 11–14 章 | 读懂 piecewise 的 FX 切图、breakable 的分段捕获、图数量爆炸来源、卫星 runner | 要加新特性的人 |
| L4 运维层 | 第 15–19 章 | 配全部参数、看懂自动禁用规则、查显存、排障 | 部署/调优/排障的人 |

只有 5 分钟：读第 1.4 节"一句话总结"、第 3.2 节架构图、第 17 章参数表。

---

## 目录

1. [为什么需要 CUDA Graph](#1-为什么需要-cuda-graph)
2. [CUDA Graph 基础：三个必须理解的约束](#2-cuda-graph-基础三个必须理解的约束)
3. [SGLang CUDA Graph 全景架构](#3-sglang-cuda-graph-全景架构)
4. [四种 Capture Backend](#4-四种-capture-backend)
5. [ShapeKey：一张图的身份证](#5-shapekey一张图的身份证)
6. [Capture 全流程逐步拆解](#6-capture-全流程逐步拆解)
7. [Replay 全流程逐步拆解](#7-replay-全流程逐步拆解)
8. [Buffer 三层体系](#8-buffer-三层体系)
9. [Attention Backend 的三方法契约](#9-attention-backend-的三方法契约)
10. [显存与图内存池管理](#10-显存与图内存池管理)
11. [深水区一：Piecewise CUDA Graph](#11-深水区一piecewise-cuda-graph)
12. [深水区二：Breakable CUDA Graph](#12-深水区二breakable-cuda-graph)
13. [深水区三：图数量是怎么乘起来的](#13-深水区三图数量是怎么乘起来的)
14. [深水区四：卫星 Runner](#14-深水区四卫星-runner)
15. [非 CUDA 设备：CPU / NPU / XPU](#15-非-cuda-设备cpu--npu--xpu)
16. [启动顺序：一切是在什么时候发生的](#16-启动顺序一切是在什么时候发生的)
17. [配置面板：参数、别名、环境变量](#17-配置面板参数别名环境变量)
18. [自动禁用规则全表](#18-自动禁用规则全表)
19. [可观测性与排障](#19-可观测性与排障)
20. [附录](#20-附录)

---

## 1. 为什么需要 CUDA Graph

### 1.1 问题的根源：decode 阶段是"launch bound"的

一次 decode step 里，每个请求只产生 1 个 token。以 70B 模型 80 层为例，一层里
attention + MLP + norm + allreduce 加起来轻轻松松几十个 kernel，一次 forward 就是
**几千次 kernel launch**。而每次 launch 的 CPU 侧开销大约在 5–10 微秒量级。

于是出现这样的局面：

```
GPU:  [k1][k2]   [k3][k4]     [k5]        ← GPU 干活很快，但一直在等
CPU:  launch k1 → launch k2 → launch k3 …  ← CPU 成了瓶颈
      └────────── CPU 提交速度决定了整体速度 ──────────┘
```

小 batch decode 时，单个 kernel 的实际计算可能只要 2–3 微秒，比 launch 开销还短。
这就是典型的 **CPU launch bound**：GPU 利用率低不是因为算力不够，而是因为"派活"太慢。

### 1.2 CUDA Graph 的解法

CUDA Graph 把"一整串 GPU 操作 + 它们之间的依赖关系"预先录制成一个图对象，之后
**一次 launch 就能把整张图交给 GPU 执行**：

```
录制一次（capture）：      CPU 走完整个 forward，但 kernel 不真执行，只记录
重复执行 N 次（replay）：  CPU 只发 1 个 cudaGraphLaunch，GPU 按图跑完全部 kernel
```

收益：
- **CPU 开销从 O(kernel 数) 降到 O(1)**，小 batch 下端到端延迟常见降低 20%–50%；
- **kernel 间隙消失**，GPU 上 kernel 首尾相接；
- **调度器可以更激进**：CPU 空出来的时间可以用来做下一批的调度（overlap scheduler）。

### 1.3 代价

天下没有免费的午餐，CUDA Graph 的三个代价贯穿本文档后面所有章节：

| 代价 | 表现 | 本文档在哪里讲 |
|---|---|---|
| **形状必须固定** | 需要为每个 batch size 单独录一张图 → 图数量爆炸 | 第 5、13 章 |
| **地址必须固定** | 输入输出必须放在"图常驻"的静态 buffer 里 | 第 8 章 |
| **显存额外开销** | 每张图的中间激活都要长期占住 → 几 GB 级别 | 第 10 章 |
| **启动变慢** | 录几十上百张图要几十秒到几分钟 | 第 6.5、16 章 |
| **CPU 逻辑被冻结** | 图里录不到 host 代码，形状相关的分支会失效 | 第 9、14.3 章 |

### 1.4 一句话总结

> SGLang 用 CUDA Graph 把 decode（以及可选的 prefill）的 kernel launch 开销一次性预付掉；
> 代价是要为**每一种"形状 + 变体"组合**预先录一张图、把所有输入钉在固定地址的静态 buffer 上、
> 并接受数 GB 的额外显存和几十秒的启动开销。SGLang 的整套设计（分桶、backend 分层、
> buffer registry、attention 三方法契约、自动禁用规则）都是在**管理这三个代价**。

---

## 2. CUDA Graph 基础：三个必须理解的约束

这一章不涉及 SGLang 代码，只讲 CUDA Graph 本身的三条硬规则。理解了它们，后面
SGLang 那些看起来很绕的设计就都变得顺理成章。

### 2.1 约束一：录制的是"地址"，不是"值"

capture 时记录下来的是每个 kernel 的**参数指针**。replay 时 GPU 会读取**同一批地址**
里当时的内容。

```python
# 原生 PyTorch 的最小例子
static_input = torch.zeros(8, 4096, device="cuda")   # 地址固定
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)               # 只录制，不真算

# 之后每次要用：
static_input.copy_(real_input)   # ← 必须"写进原地址"，不能重新赋值
g.replay()
result = static_output.clone()   # ← 输出也在固定地址，下次 replay 会被覆盖
```

推论（SGLang 里到处体现）：
- `static_input = real_input` 这种**重新绑定**是无效的，必须 `copy_()`；
- 输出 buffer 会被下一次 replay 覆盖，所以要么立刻消费，要么 clone；
- 任何在 capture 期间临时分配的中间张量，其地址必须在 replay 时依然有效
  → 这就是"图内存池"存在的原因（第 10 章）。

### 2.2 约束二：形状必须完全一致

图里所有 kernel 的 grid/block 配置、张量 shape 都被写死。batch size 8 的图不能跑 batch 5。

两种应对方式，SGLang 都用：
1. **分桶 + padding**：只录若干个 batch size（如 1,2,4,8,…,256），真实 batch 向上取到
   最近的桶，多出来的槽位填 dummy 数据。见 `BaseCudaGraphRunner._pad_to_bucket`
   （`model_executor/runner/base_cuda_graph_runner.py`）。
2. **只对形状无关的片段建图**（piecewise / breakable），把形状敏感的算子留在图外用
   eager 执行。见第 11、12 章。

### 2.3 约束三：host 代码在 replay 时不会重新执行

这是最容易踩的坑，也是本文档反复强调的失效模式。

capture 只把 **device 侧操作**记进图。所有 Python/C++ host 代码（if 判断、`.item()`、
`.cpu()`、`len()`、动态 `torch.empty()`、attention 的 plan/schedule 计算）在 capture 那一刻
执行过一次，之后 replay **不会再执行**，其结果被永久冻结在图里。

```python
# 危险代码：capture 时 seq_len 是 128，replay 时永远按 128 算
if seq_len > 100:        # host 分支 → 被冻结
    use_kernel_a()
else:
    use_kernel_b()

n = seq_lens.max().item()          # host 同步 → 被冻结成 capture 时的值
buf = torch.empty(n, device="cuda")  # 动态形状分配 → 地址+大小都被冻结
```

**最危险的情况是它不报错**：图能录成功、能 replay、输出也是合法数值，只是数值/性能悄悄
错了。SGLang 里有一个教科书级例子——metadata glue graph 对 DFlash 系投机解码的影响：
capture 成功、输出正确，只有 accept length 悄悄塌掉（`runner/metadata_glue_graph.py:24-31`）。

因此 SGLang 把 attention 的元数据准备**强制拆成图内 / 图外两个方法**，见第 9 章。

### 2.4 三条约束 → SGLang 的三个机制

| 约束 | SGLang 的机制 | 章节 |
|---|---|---|
| 地址固定 | `CudaGraphBufferRegistry` + `GraphSlot` + `PaddingPolicy` | 第 8 章 |
| 形状固定 | 分桶 + `ShapeKey` + padding 语义（`seq_len_fill_value`） | 第 5、7 章 |
| host 冻结 | attention 三方法契约（`out_graph` / `in_graph`） | 第 9 章 |

---

## 3. SGLang CUDA Graph 全景架构

### 3.1 目录地图

先建立"哪个文件管什么"的直觉。以下都在 `python/sglang/srt/` 下：

```
model_executor/
├── runner/                       ← 【阶段语义层】"我在跑 decode 还是 prefill"
│   ├── base_runner.py                (699)  所有 runner 的公共父类
│   ├── base_cuda_graph_runner.py     (160)  CUDA Graph runner 的公共脚手架
│   ├── decode_cuda_graph_runner.py  (1597)  decode / target-verify
│   ├── prefill_cuda_graph_runner.py (1921)  prefill / extend
│   ├── eager_runner.py               (464)  不建图的对照路径（也持有 buffer！）
│   ├── metadata_glue_graph.py        (106)  只给 attention 元数据建的小图
│   ├── shape_key.py                   (41)  ShapeKey 定义
│   └── flashinfer_autotune.py        (417)
│
├── runner_backend/               ← 【捕获机制层】"我用什么方式录图"
│   ├── base_cuda_graph_backend.py     (84)  纯 ABC，6 个抽象方法
│   ├── full_cuda_graph_backend.py           整图捕获
│   ├── breakable_cuda_graph_backend.py(261)  可打断的分段捕获（BCG）
│   ├── tc_piecewise_cuda_graph_backend.py    torch.compile 分片捕获
│   ├── cuda_graph_dedup_mixin.py             可执行体去重
│   └── utils.py                              backend 选择与降级
│
├── runner_utils/                 ← runner 侧工具
│   ├── buffers.py                    (485)  DecodeInputBuffers / PrefillInputBuffers
│   ├── pool.py                       (231)  进程级共享图内存池 + capture stream
│   ├── capture_mode.py                      capture 上下文标记
│   ├── deepep_adapter.py                    DeepEP 在图内的适配
│   ├── shared_read_event.py
│   └── __init__.py                          两条 capture 失败提示文案（:14-21, :23-28）
│
├── runner_backend_utils/         ← backend 侧工具
│   ├── tc_piecewise_cuda_graph/{__init__,context_manager}.py
│   └── breakable_cuda_graph/{__init__,breakable_cuda_graph,context,cuda_utils}.py
│
├── cuda_graph_config.py              (283)  统一配置对象（backend 选择、bs 列表）
├── cuda_graph_buffer_registry.py    (1026)  【核心】GraphSlot / PaddingPolicy / 填充
├── input_buffers.py                   (99)  跨 runner 的 buffer 共享/合并
├── graph_memory_usage.py              (67)  显存统计上报
├── graph_shared_output.py             (70)  跨 runner 共享输出 buffer
├── cpu_graph_runner.py                      CPU 侧对应物
└── model_runner_components/
    └── cuda_graph_setup.py           (584)  【入口】启动时的编排顺序

compilation/                      ← torch.compile / piecewise 的底座
├── backend.py                             FX split_module 切图
├── compilation_config.py                  SPLIT_OPS 注册表（:7-16）
├── cuda_piecewise_backend.py         (230) 每片的 capture/replay
├── npu_piecewise_backend.py          (110)
├── xpu_piecewise_backend.py          (113)
├── compile.py / compile_phase.py / compiler_interface.py / pass_manager.py
└── torch_compile_decoration.py            旧的 decode-full + --enable-torch-compile 路径

speculative/                      ← 投机解码的专用 runner
├── eagle_draft_cuda_graph_runner.py
├── eagle_draft_extend_cuda_graph_runner.py
├── multi_layer_eagle_draft_extend_cuda_graph_runner.py
└── frozen_kv_mtp_cuda_graph_runner.py

multimodal/                       ← ViT 的专用 runner
├── vit_cuda_graph_runner.py
├── internvl_vit_cuda_graph_runner.py
└── kimi_k3_vit_cuda_graph_runner.py

hardware_backend/npu/graph_runner/*      NPU（ACL Graph）
hardware_backend/xpu/graph_runner/*      XPU

arg_groups/
├── cuda_graph_hook.py                (588) 【重要】所有自动禁用规则
├── memory_hook.py / attention_hook.py / platform_hook.py
```

> 提醒：`model_executor/breakable_cuda_graph/` 目录下只剩过期的 `__pycache__`，
> 真正的实现在 `runner_backend_utils/breakable_cuda_graph/`，不要引用前者。

### 3.2 两层架构：runner × backend

整个设计的核心是一个**正交分解**：

```
                       ┌─────────────────────────────────────┐
                       │           ModelRunner               │
                       │  model_runner_components/           │
                       │      cuda_graph_setup.py            │  ← 编排：谁先建、谁后建
                       └──────────────┬──────────────────────┘
                                      │ 构造
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────────┐    ┌──────────────────┐
│ DecodeCudaGraph  │      │ PrefillCudaGraph     │    │ EagerRunner      │
│ Runner           │      │ Runner               │    │ （不建图）        │
│                  │      │                      │    │ 但也持有 buffer   │
│ decode /         │      │ prefill / extend     │    └──────────────────┘
│ target_verify    │      │                      │
└────────┬─────────┘      └──────────┬───────────┘
         │  「阶段语义层 runner/ 」：决定 bs 列表、ForwardMode、
         │   ShapeKey 怎么算、buffer 怎么填、padding 语义是什么
         │
         ▼  持有一个 backend（组合，不是继承）
┌───────────────────────────────────────────────────────────────────┐
│  BaseCudaGraphBackend  (纯 ABC，无状态无默认实现)                    │
│  capture_session / capture_one / can_run / replay_session /        │
│  replay / cleanup                                                  │
├─────────────┬─────────────────┬──────────────────┬────────────────┤
│  FULL       │  BREAKABLE      │  TC_PIECEWISE    │  DISABLED      │
│  整图       │  分段+eager 桥   │  torch.compile   │  不建图         │
│             │  （BCG）         │  切片             │                │
└─────────────┴─────────────────┴──────────────────┴────────────────┘
             「捕获机制层 runner_backend/ 」：只管"怎么把这段 forward 录下来"
```

分层的关键约定（`runner_backend/base_cuda_graph_backend.py` 的注释原话）：

> "The outer capture loop is runner-specific; it lives on the runner, not here."

也就是说：
- **runner 知道"要录哪些形状"**（bs 列表、variant、forward mode、padding 语义）；
- **backend 只知道"给我一个 `ShapeKey` 和一个 `forward_fn`，我把它录下来"**。

这样新增一个阶段（比如 draft-extend）不用碰 backend，新增一种捕获机制（比如 BCG）
也不用碰 decode/prefill 的语义代码。

### 3.3 两个 ABC 的完整契约

**runner 侧**（`runner/base_cuda_graph_runner.py`）：

```python
class BaseCudaGraphRunner(BaseRunner):
    buffers: ForwardInputBuffers          # 静态输入 buffer
    backend: BaseCudaGraphBackend         # 组合持有捕获后端

    @staticmethod
    def _pad_to_bucket(raw_size: int, buckets: Sequence[int]) -> int:
        assert raw_size <= buckets[-1], ...
        index = bisect.bisect_left(buckets, raw_size)
        return buckets[index]            # 向上取到最近的桶

    @abstractmethod
    def capture_prepare(self, size, *args, **kwargs): ...   # 单形状捕获前的准备
    @abstractmethod
    def capture(self) -> None: ...                          # 外层循环（runner 自己写）
    @abstractmethod
    def capture_one_shape(self, size, *args, **kwargs): ...  # 录一个形状
```

同文件还有两个公共工具：

```python
@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze: gc.freeze()        # 把已有对象移出 GC 扫描集
    try: yield
    finally:
        if should_freeze: gc.unfreeze(); gc.collect()
```
`freeze_gc` 的作用：capture 期间 Python GC 如果触发，可能在图捕获中间释放/搬动对象，
既拖慢捕获也有风险。默认冻结 GC，`--enable-cudagraph-gc` 可以关掉这个冻结。

```python
def get_batch_sizes_to_capture(model_runner, captured_req_width: int = 1):
    capture_bs = list(get_exec().graph.cuda_graph_config.decode.bs)
    num_max_requests = model_runner.req_to_token_pool.size
    mul_base = get_cuda_graph_batch_size_alignment()
    alignment_width = captured_req_width
    if get_exec().overlap.enable_two_batch_overlap:
        alignment_width = 1                       # TBO 下不按 req_width 对齐
    num_max_requests = get_cuda_graph_max_batch_size(num_max_requests)
    if max(capture_bs) > num_max_requests:
        capture_bs += [num_max_requests]          # 保证顶桶能覆盖上限
    capture_bs = [bs for bs in capture_bs if bs * alignment_width % mul_base == 0]
    capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]
    capture_bs = list(sorted(set(capture_bs)))
    compile_bs = ([bs for bs in capture_bs if bs <= get_exec().graph.torch_compile_max_bs]
                  if get_flags().capture.enable_torch_compile else [])
    return capture_bs, compile_bs
```
读懂这段就明白：**你在 `--cuda-graph-bs` 里写的值不一定全部生效**——会被
`req_to_token_pool.size`、对齐倍数、最大 bs 三重过滤。

**backend 侧**（`runner_backend/base_cuda_graph_backend.py`，全文只有 84 行，
文件头明确写着 "Pure ABC: no state, no defaults"）：

| 方法 | 作用 |
|---|---|
| `capture_session(stream)` | 进入捕获模式的上下文管理器（切 stream、设标记、借内存池） |
| `capture_one(shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None)` | 录一个形状 |
| `can_run(forward_batch, shape_key)` | 这个 batch 能不能用已录的图跑 |
| `replay_session()` | 进入 replay 模式的上下文管理器 |
| `replay(shape_key, static_forward_batch, **kwargs)` | 执行一次 replay |
| `cleanup()` | 释放图与池 |

注意 `can_run` 的存在：**判断"能否用图"的责任在 backend**，因为不同 backend 的
可用条件不同（FULL 要求形状精确命中，piecewise 只要求分片形状可编译）。

---

## 4. 四种 Capture Backend

### 4.1 一张表看懂差别

| | **FULL** | **BREAKABLE (BCG)** | **TC_PIECEWISE** | **DISABLED** |
|---|---|---|---|---|
| 实现文件 | `full_cuda_graph_backend.py` | `breakable_cuda_graph_backend.py` | `tc_piecewise_cuda_graph_backend.py` | — |
| 是否用 torch.compile | 否 | **否** | **是**（Dynamo+Inductor） | — |
| 图的粒度 | 整个 forward 一张图 | 若干连续"段" | 每个非 split-op 片段一张图 | — |
| 图外 eager 部分 | 无 | 被 `@eager_on_graph` 标记的算子 | 所有 `SPLIT_OPS` | 全部 |
| 主要用途 | decode（默认） | prefill（CUDA 上默认） | prefill（非 CUDA 默认）/ 灵活性优先 | 调试、不支持的配置 |
| 启动开销 | 中 | 中 | **高**（要编译） | 无 |
| 形状灵活性 | 最差（必须命中桶） | 中 | **最好**（图外算子可变形） | 最好 |
| 兼容性风险 | 需要全链路 graph-safe | 中 | Dynamo graph break / 编译失败 | 无 |

### 4.2 默认值与降级规则

- **decode 默认 `FULL`**：`cuda_graph_config.py:147-149`。
- **prefill 默认由 `default_prefill_backend()` 决定**（`cuda_graph_config.py:112-121`）：
  ```python
  BREAKABLE if is_cuda() else TC_PIECEWISE
  ```
  即 NVIDIA/HIP CUDA 上走 BCG，其他平台走 torch.compile piecewise。
- **decode 不支持 `tc_piecewise`**：选了会 warn-once 然后**回落到 `full`**
  （`runner_backend/utils.py:94-101`）。
- **XPU 的 decode 必须是 `full`**，否则直接 `ValueError` 硬失败（`runner_backend/utils.py:79-81`）。

### 4.3 三个 backend 的共同点：capture 前必须 warmup 两次

三个 backend 都在正式 capture 之前跑**两次** eager warmup：

| backend | 位置 |
|---|---|
| FULL | `full_cuda_graph_backend.py:105-140` |
| BREAKABLE | `breakable_cuda_graph_backend.py:117-122` |
| TC_PIECEWISE | `tc_piecewise_cuda_graph_backend.py:240-243` |

为什么是两次而不是一次？第一次 warmup 触发所有惰性初始化（cuBLAS/cuDNN handle、
autotune、JIT 编译、workspace 分配），第二次确认这些副作用已经稳定、不会在 capture
过程中再触发新的分配。如果只 warmup 一次，capture 时可能仍在做首次分配，导致图里
录进本该在图外的分配行为。

另一个共同细节：**capture 循环按 batch size 从大到小执行**（`reversed(list)`）。
原因是显存池的复用——先录最大形状把内存池撑到最终大小，后面的小形状都能在已有池里
复用，从而避免"池反复扩容 + 碎片化"。

---

## 5. ShapeKey：一张图的身份证

### 5.1 定义

`runner/shape_key.py`（41 行）定义了整套体系的通用键：

```python
ShapeKey(size, stream_idx, variant_label, dsa_variant)
```

| 字段 | 含义 | 什么时候不是默认值 |
|---|---|---|
| `size` | padding 后的 batch size（或 token 数） | 总是 |
| `stream_idx` | pdmux 多流场景下的流编号 | 开 pdmux |
| `variant_label` | 语义变体标签（LoRA、ragged verify tier 等） | 见第 13 章 |
| `dsa_variant` | DSA 的 dense / sparse 双图选择 | DSA 模型 + HIP |

**图数量 = 这四个维度的笛卡尔积中被实际用到的组合数**。这就是第 13 章"图数量爆炸"的
数学来源。

### 5.2 为什么要有 ShapeKey 这个抽象

在重构之前，各处用的是裸 `bs` 做 key，于是每加一个"变体"维度就要改一遍所有字典。
统一成 `ShapeKey` 后：
- `backend.capture_one(shape_key, ...)` / `backend.replay(shape_key, ...)` 签名不用变；
- 新增变体维度只需要改 runner 侧的"怎么算出 ShapeKey"，backend 完全无感；
- 去重（第 6.6 节）也可以在 ShapeKey 粒度上做。

---

## 6. Capture 全流程逐步拆解

### 6.1 顶层时序

```
① 一次性初始化
   attention_backend.init_cuda_graph_state(max_bs, max_num_tokens)
   ↓  分配所有"图常驻"的 attention 元数据 buffer
   ↓  ★ 只有 decode runner 调这个（decode_cuda_graph_runner.py:373, :562）
   ↓    prefill runner 从不调，直接复用 decode 分配好的 buffer

② with freeze_gc(...), backend.capture_session(stream):
   for size in reversed(capture_bs):          # 大 → 小
     for variant in variants(size):           # LoRA / dsa / tier …
       ③ capture_prepare(size, variant)
          ├─ 构造 dummy ForwardBatch（引用静态 buffer）
          └─ attention_backend.init_forward_metadata_out_graph(fb, in_capture=True)
             ★ 这一步在【图外】执行，允许 .item() / host 计算 / 动态分配
       ④ 两次 eager warmup
       ⑤ backend.capture_one(shape_key, forward_fn=...)
          └─ 【图内】录制：
               attention_backend.init_forward_metadata_in_graph(fb)   ← 纯 device 操作
               model.forward(...)
       ⑥ 记录 output buffer 引用到 graph_outputs[shape_key]
```

### 6.2 步骤 ①：`init_cuda_graph_state` 只由 decode runner 调用

这是一个容易误解的点。attention backend 的图常驻 buffer（例如 FlashAttention 的
`page_table`、TRTLLM 的 workspace、FlashInfer 的 plan 输出）**全部由 decode runner
在自己的构造过程中一次性分配**，尺寸按 `max_bs × max_num_tokens` 开满：

- `decode_cuda_graph_runner.py:373`（常规路径）
- `decode_cuda_graph_runner.py:562`（另一分支）

prefill runner 完全不调用 `init_cuda_graph_state`，它直接使用同一个 attention backend
实例里已经分配好的 buffer。

**实践含义**：
- 如果关掉 decode CUDA Graph 但开着 prefill CUDA Graph，需要确认这些 buffer 仍被分配
  （这是历史上出过 bug 的组合，见第 19 章）；
- attention buffer 的显存开销只算一次，不是 prefill + decode 相加。

### 6.3 步骤 ③：dummy batch 的构造与"padding 语义"

capture 时没有真实请求，需要造一个假的 `ForwardBatch`。关键是**假数据的值必须让
attention kernel 走和真实情况相同的代码路径**，同时不能越界读写。

核心 hook 是 `get_cuda_graph_seq_len_fill_value()`：padding 出来的那些"不存在的请求"，
它们的 `seq_len` 应该填什么？

| 值 | 谁这么做 | 位置 | 为什么 |
|---|---|---|---|
| `1` | 绝大多数 backend | ABC 默认 | 长度 0 会让某些 kernel 除零或走退化分支，填 1 最安全 |
| `0` | Ascend NPU | `ascend_backend.py:794-795` | NPU kernel 对 0 长度有专门处理 |
| `num_draft_tokens` | AITER | `aiter_backend.py:2042-2043` | 投机 verify 的 kernel 假设每 req 至少这么多 token |
| 子 backend 必须一致 | TBO | `tbo_backend.py:167-173` 有 assert | 两批共用一张图，语义必须统一 |

**由此产生一个必须知道的副作用**：`seq_lens_sum` 会被 padding 撑大。

```
seq_lens_sum(图内看到的) = seq_lens_sum(真实) + (padded_bs - raw_bs) * fill_value
```

见 `decode_cuda_graph_runner.py:182-186`。任何依赖 `seq_lens_sum` 做容量判断或
归一化的逻辑，在 CUDA Graph 路径下都必须意识到这个偏差。

### 6.4 步骤 ⑤：`capture_one` 里发生什么（以 FULL 为例）

```
capture_one(shape_key, forward_fn):
    ├─ 切到共享 capture stream（runner_utils/pool.py）
    ├─ 借用进程级共享 graph memory pool
    ├─ graph = torch.cuda.CUDAGraph()
    ├─ with torch.cuda.graph(graph, pool=shared_pool, stream=capture_stream):
    │      forward_fn()          # ← 内部会调 init_forward_metadata_in_graph + model.forward
    ├─ self.graphs[shape_key] = graph
    └─ self.output_buffers[shape_key] = out    # 输出张量地址，replay 后从这里读
```

`capture_inputs` / `post_warmup_hook` 两个可选参数分别用于：把额外的静态输入登记进来，
以及在 warmup 之后、capture 之前插入一段清理/重置逻辑（比如把 warmup 污染的累加器归零）。

### 6.5 capture 有多贵

启动阶段的 capture 是同步阻塞的，主要成本：

| 成本项 | 量级 |
|---|---|
| 每张图的捕获 | 数十毫秒（两次 warmup + 一次 capture） |
| 图数量 | 常规 20–60 张；DSA 双图 +52 张；LoRA/tier 变体可再翻倍 |
| torch.compile（piecewise） | 每个不同的分片 shape 需要一次 Inductor 编译，可达分钟级 |
| 总计 | 常见 20 秒–2 分钟；极端配置（piecewise + 多变体）可到 5 分钟以上 |

减少启动时间的手段见第 17.4 节（缩 bs 列表、`--cuda-graph-max-bs`、关变体）。

### 6.6 图可执行体去重（dedup）

`runner_backend/cuda_graph_dedup_mixin.py` 利用 CUDA 的 `cudaGraphExecUpdate`：如果两张
图的**拓扑结构完全相同、只有参数不同**，那么可以让它们共用同一个 `cudaGraphExec`
可执行体，只在 replay 前 update 参数。

这对"变体维度多但结构相同"的场景收益巨大——例如 LoRA 的不同 adapter、ragged verify
的不同 tier，往往结构一致。省下的是**可执行体的显存**，不是图节点的显存。

注意：`cudaGraphExecUpdate` 有严格前置条件（节点数、依赖关系、kernel 函数指针都要一致），
不满足时会失败并回落到"各自持有独立可执行体"。

---

## 7. Replay 全流程逐步拆解

### 7.1 顶层时序

```
① can_run(forward_batch, shape_key)?           ← 不行就走 eager runner
② raw_bs = 真实 batch size
   padded_bs = _pad_to_bucket(raw_bs, capture_bs)
   shape_key = ShapeKey(padded_bs, stream_idx, variant_label, dsa_variant)
③ registry.fill_from(forward_batch)            ← 把真实数据 copy_ 进静态 buffer，
                                                  并按 PaddingPolicy 处理 padding 区
④ static_fb = 用静态 buffer 拼出的 ForwardBatch 视图
⑤ attention_backend.init_forward_metadata_out_graph(static_fb)
   ★ 【图外】执行，可以做 host 计算 / plan / .item()
⑥ with backend.replay_session():
       graph.replay()                          ← 一次 launch
⑦ out = output_buffers[shape_key][:raw_bs]     ← 只取前 raw_bs 行，丢掉 padding
```

### 7.2 关键点：真实工作分成"图外元数据"和"图内计算"

replay 路径上 CPU 仍要做三件事，它们决定了 CUDA Graph 的收益上限：
1. `fill_from`：若干次 `copy_`（device-to-device 小拷贝，成本低但不为零）；
2. `init_forward_metadata_out_graph`：attention 的 host 侧 plan/schedule；
3. 一次 `cudaGraphLaunch`。

**如果第 2 步很重（比如 FlashInfer 的 plan 在 CPU 上做大量工作），CUDA Graph 的收益会被
它吃掉**。这就是"metadata glue graph"（第 14.3 节）想解决的问题。

### 7.3 padding 槽位的正确性保证

padding 出来的槽位会被真的算一遍（图形状固定，算不了少的），必须保证：

| 要求 | 怎么做到 |
|---|---|
| 不越界读 KV | padding req 的 `req_pool_idx` 指向合法（通常是 0 号）槽位 |
| 不污染真实 KV | padding req 的 `out_cache_loc` 指向专门的 dummy slot |
| kernel 不崩 | `seq_len` 填 `get_cuda_graph_seq_len_fill_value()`（见 6.3） |
| 输出被丢弃 | 只取 `[:raw_bs]` |

第二条最重要：**如果 padding 请求的写地址落在真实请求的 KV 上，就是静默的数值污染**。

### 7.4 什么时候 replay 会被拒绝（走 eager）

`can_run` 返回 False 的典型原因：
- `raw_bs > max(capture_bs)`（超过顶桶，`_pad_to_bucket` 里的 assert 就是防这个）；
- 当前 `ForwardMode` 不在该 runner 捕获的模式里；
- 需要的 variant（LoRA adapter、tier、dsa variant）没有对应的图；
- 该 backend 被自动禁用规则关掉了（第 18 章）；
- prefill 侧 token 数超过捕获上限。

此时执行落到 `runner/eager_runner.py`。**EagerRunner 不只是"兜底"，它还在启动时先于图
runner 创建，并且它注册的 buffer 会成为共享池里的"标准件"**（见第 8.1、16 章）。

---

## 8. Buffer 三层体系

这是全篇最容易搞混的地方。**SGLang 现在有三层 buffer 机制同时在跑**，各管一件事。

### 8.1 第一层：`input_buffers.py` — 跨 runner 的 buffer 合并

`model_executor/input_buffers.py`（99 行）提供 `share_input_buffer()`：

规则是**"先到先得的规范化"**：以 `(name, numel, dtype, device)` 四元组为键，
第一个注册这个键的 runner 拿到的张量成为**canonical（规范）张量**，之后所有请求同样
四元组的 runner 都拿到同一个张量。

好处：decode runner、prefill runner、eager runner、各种 spec runner 各自都需要
`positions`、`seq_lens`、`out_cache_loc` 这类 buffer，如果各建一份就是数倍浪费；合并后
只有一份物理存储。

两个必须知道的细节：
- **NPU 上这个合并被禁用**，原因是精度问题（合并后某些 NPU kernel 行为不一致）；
- 因为"先到先得"，**创建顺序会影响谁是 canonical**。这正是 `cuda_graph_setup.py` 要把
  `EagerRunner` 放在图 runner 之前创建的原因——让 eager 路径的 buffer 成为标准件。

### 8.2 第二层：`runner_utils/buffers.py` — 物理存储的持有者

`runner_utils/buffers.py`（485 行）里的 `DecodeInputBuffers` / `PrefillInputBuffers`
**仍然存在、仍然持有真正的张量存储**。

这是老文档最容易过时的地方：它们不再是"填充逻辑的实现者"，而退化成
**"一批具名张量的容器"**。填充逻辑搬到了第三层。

```
DecodeInputBuffers
├── input_ids        : Tensor[max_bs]
├── positions        : Tensor[max_bs]
├── seq_lens         : Tensor[max_bs]
├── out_cache_loc    : Tensor[max_bs]
├── req_pool_indices : Tensor[max_bs]
└── …（mrope、spec、encoder、pdmux 等按需扩展）
```

### 8.3 第三层：`CudaGraphBufferRegistry` — 填充语义的实现者

`model_executor/cuda_graph_buffer_registry.py`（1026 行）是现在的**核心**。三个概念：

**`GraphSlot`**：描述一个图常驻槽位——名字、形状、dtype、以及最关键的
**padding 策略**和**取值来源**。

**`PaddingPolicy`** 枚举，决定 `[raw_bs : padded_bs)` 这段"多余区域"怎么处理：

| 策略 | 行为 | 典型用途 |
|---|---|---|
| `KEEP_PAD` | 保留上一次 replay 留下的内容，不动 | 内容无所谓、省一次 kernel |
| `FILL_SENTINEL` | 填一个哨兵值（如 `seq_len_fill_value`） | `seq_lens`：必须是合法值 |
| `ZERO` | 清零 | 累加器、mask |
| `FOREACH_COPY` | 用 `torch._foreach_copy_` 批量拷贝多个 buffer | 一次 launch 搞定 N 个小拷贝 |
| `FILL_ONCE` | 只在初始化时填一次，之后不再动 | 常量表、固定索引 |

`FOREACH_COPY` 是性能关键：replay 前要拷十几个小张量，逐个 `copy_` 就是十几次 launch，
恰好又把省下来的 launch 开销赔回去；`_foreach_copy_` 把它们合成一次。

**`FillContext`**：一次 `fill_from` 调用的上下文，携带 `raw_bs`、`padded_bs`、
`forward_batch`、变体信息，供各 slot 的填充函数使用。

**三个 builder**（决定每种 runner 有哪些 slot、各用什么策略）：
- `build_decode_registry`（`:510`）
- `build_prefill_registry`（`:800`）
- `build_eager_registry`（`:981`）

### 8.4 三层是怎么连起来的：`bind=` 收养机制

registry 并不自己 `torch.zeros()` 分配存储，而是通过 `bind=` 参数**收养**第二层
（`DecodeInputBuffers`）已经分配好的张量。

```
input_buffers.share_input_buffer()   →  决定"哪份物理存储是规范的"
            ↓
DecodeInputBuffers                   →  持有物理存储
            ↓  registry.add_slot(..., bind=buffers.seq_lens)
CudaGraphBufferRegistry              →  在同一块存储上定义 padding/填充语义
```

为什么要收养而不是重新分配？因为 **`data_ptr` 必须稳定**。图捕获时记下的是这块存储的
地址，registry 如果自己再分配一份，capture 和 replay 就会读写不同的地址。收养保证了
"一份存储、多个视角"。

### 8.5 三层的分工总结

| 层 | 文件 | 回答的问题 |
|---|---|---|
| 1 | `input_buffers.py` | 这份 buffer 该由谁持有、能不能和别人共用？ |
| 2 | `runner_utils/buffers.py` | 物理存储在哪里？ |
| 3 | `cuda_graph_buffer_registry.py` | replay 前怎么往里填数据、padding 区怎么处理？ |

改 buffer 相关代码时的判断法：
- 想加一个新的图输入 → 在第 2 层加字段，在第 3 层的对应 builder 里加 slot；
- 觉得显存浪费 → 看第 1 层能不能共享；
- 数值不对 / padding 污染 → 查第 3 层的 `PaddingPolicy` 选得对不对。

---

## 9. Attention Backend 的三方法契约

### 9.1 旧接口已经删除

`layers/attention/base_attn_backend.py:53-56` 明确写着：
`init_forward_metadata_capture_cuda_graph` 和 `init_forward_metadata_replay_cuda_graph`
**"fully deprecated and removed from the ABC"**。

如果你在别的资料里看到这两个方法名，那是重构前的接口。任何新写的 attention backend
实现它们都不会被调用。

### 9.2 新的三方法

| 方法 | 位置 | 执行位置 | 允许做什么 |
|---|---|---|---|
| `init_forward_metadata(fb)` | `:88-94` | 非图路径（eager） | 任意 |
| `init_forward_metadata_out_graph(fb, in_capture=False)` | `:96-116` | **图外** | host 计算、`.item()`、`.cpu()`、动态分配、plan |
| `init_forward_metadata_in_graph(fb)` | `:118-130` | **图内**（被录进图） | 只能纯 device 操作 |

`init_forward_metadata_in_graph` 的禁忌（文档字符串里明写）：
- 不能 `.item()` / `.cpu()` / `.tolist()`（host 同步 → 值被冻结）；
- 不能用动态 shape 的 `torch.empty()`（地址+大小被冻结）；
- 不能有依赖运行时值的 Python 分支。

### 9.3 完整调用时序（把 9.2 放进第 6、7 章的流程里）

```
启动一次：
    init_cuda_graph_state(max_bs, max_num_tokens)      ← 只有 decode runner 调

每个待捕获形状：
    init_forward_metadata_out_graph(fb, in_capture=True)   ← 图外，in_capture=True
    ┌─ 进入 capture ────────────────────────────────┐
    │   init_forward_metadata_in_graph(fb)          │  ← 被录进图
    │   model.forward(...)                          │
    └───────────────────────────────────────────────┘

每次 replay：
    registry.fill_from(forward_batch)
    init_forward_metadata_out_graph(fb_view)            ← 图外，in_capture=False
    graph.replay()                                       ← in_graph 的内容自动被重放
```

注意 `in_capture` 这个参数的意义：同一个 `out_graph` 方法在捕获期和运行期都会被调用，
`in_capture=True` 让 backend 知道"现在是在准备捕获，输入是 dummy 数据，不要做真实性校验，
但要把所有需要图常驻的地址固定下来"。

### 9.4 `page_size > 1` 没有统一约定

这是一个**实现分歧点**，改 attention backend 时必读：

| backend | 做法 |
|---|---|
| FlashAttention / TRTLLM-MLA | buffer 按 page 粒度收窄，并保留常驻的 `strided_indices` |
| Triton / FlashInfer | 仍按 token 粒度；FlashInfer 甚至给 `begin_forward` 传字面量 `page_size=1` |
| AITER | 把 page 数再乘回 `page_size`，因为它的写入是 token 粒度的（`aiter_backend.py:1543+` 有很长的 TODO 说明） |

结论：**不要假设"page_size 语义在各 backend 一致"**。加新 backend 时，page 相关的 buffer
尺寸要自己想清楚，不能照抄别家。

### 9.5 不支持 CUDA Graph 的 backend 是怎么被拦住的

理想做法是 ABC 抛 `NotImplementedError`，但那是最后一道网（真到 capture 才炸，用户体验差）。
实际的第一道闸门在 `arg_groups/attention_hook.py:57-96`：**按 backend 名字硬编码匹配**，
命中就直接把 CUDA Graph 关掉。

当前被硬编码禁用的：`torch_native`、`flex_attention`。

这是一个公认的脆弱点：新增一个不支持图的 backend，如果忘了在这里加名字，用户会在启动
capture 阶段撞到一个不那么友好的错误。改 attention backend 时请检查这个列表。

---

## 10. 显存与图内存池管理

### 10.1 图显存花在哪里

一张 CUDA Graph 的显存占用 = **图节点元数据（很小）+ 图内所有中间激活（大头）**。
中间激活必须长期驻留，因为图里记的是它们的地址。

```
一张 bs=256 的 decode 图：
  ├─ 静态输入 buffer                   几 MB（且跨图共享）
  ├─ 图内中间激活（每层的 qkv/attn_out/mlp 中间量）  ← 主要开销
  └─ 图结构本身                        KB 级
```

多张图之间的中间激活**可以复用同一块池**（因为它们不会同时执行），所以总开销
远小于"每张图独占"的朴素估算。这就是内存池的价值。

### 10.2 进程级共享池与共享 capture stream

`runner_utils/pool.py`（231 行）提供进程级单例：

| 函数 | 位置 | 作用 |
|---|---|---|
| `get_or_create_global_graph_memory_pool()` | `:78-84` | 全进程唯一的图内存池 |
| `get_or_create_global_graph_capture_stream()` | `:87-93` | 全进程唯一的 capture stream |
| `graph_pool_user_scope` / `graph_pool_replay_scope` / `graph_pool_capture_scope` | — | 标记当前处于哪个阶段 |
| `borrow_graph_pool` | `:180-230` | 临时借出池，用完归还 |

**为什么可以共享？** 因为 prefill runner 和 decode runner **永不并发 replay**——
一个 scheduler step 里只会跑其中一个。所以它们可以共用同一块中间激活区，
总显存是 `max(prefill, decode)` 而**不是 `prefill + decode`**。

这条性质很重要：常见的误判是"我又开了 prefill CUDA Graph，显存要翻倍"，实际上只增加
`max()` 的那部分差额。

`borrow_graph_pool` 用于那些"需要在图内存池里分配一些东西，但自己不是图 runner"的场景
（例如某些 backend 的 workspace），借用保证分配落在同一池里，不产生新的碎片。

### 10.3 显存开销的量级与调节

| 配置 | 图显存典型量级 |
|---|---|
| decode full，bs 列表 20 个桶，7B 模型 | 0.5–1.5 GB |
| 同上 + prefill breakable | +0.3–1 GB（`max()` 语义，不是翻倍） |
| DSA 双图（+52 张图） | 显著增加，同时捕获时间约 2 倍 |
| 大 bs 顶桶（如 512） | 顶桶单张图就可能几百 MB |

降低图显存的手段（按性价比排序）：
1. `--cuda-graph-max-bs` 压低顶桶——顶桶往往是最贵的一张；
2. 显式给 `--cuda-graph-bs` 一个更稀疏的列表（比如只留 1,2,4,8,16,32,64,128）；
3. 关掉不需要的变体（LoRA、DSA 双图、pdmux）；
4. 把 prefill 换成 `disabled`（prefill 的 launch 收益本来就小于 decode）；
5. 最后才考虑 `--disable-cuda-graph`。

### 10.4 与 `--mem-fraction-static` 的交互

图显存**不在** `mem_fraction_static` 预算内，它是在 KV cache 分配之后额外占用的。
所以典型的 OOM 是"KV cache 分完了，capture 时才 OOM"。

`cuda_graph_setup.py` 的编排顺序（第 16 章）就是为此设计的：MoE/DeepGEMM 的显存预算
在 capture 之前先确定，capture 在 KV pool 之后进行，这样 capture OOM 时的报错更明确。

capture 阶段 OOM 时会打印 `runner_utils/__init__.py` 里定义的提示文案：
- `CUDA_GRAPH_CAPTURE_FAILED_MSG`（`:14-21`）
- `PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG`（`:23-28`）

它们会建议你调小 `--mem-fraction-static` 或缩减 bs 列表。

---

## 11. 深水区一：Piecewise CUDA Graph

### 11.1 动机

FULL 模式要求整个 forward 都是 graph-safe，但现实中总有几个算子不行：
- 形状依赖运行时值的 attention（变长 prefill）；
- 需要 host 同步的通信（某些 allreduce 实现）；
- 依赖动态分配的稀疏算子（DSA indexer、MoE dispatch）。

Piecewise 的思路：**把这些算子设为"切点"，切点之间的连续计算各自建图，切点本身用
eager 执行**。

```
原始 forward:  [linear norm...] [attention] [linear moe...] [allreduce] [linear...]
                     ↓              ↓             ↓              ↓          ↓
piecewise:        图1           eager        图2            eager       图3
```

这样形状变化只影响 eager 部分，图部分保持固定形状，兼顾了灵活性和 launch 收益。

### 11.2 切图是怎么做的

实现在 `compilation/backend.py:225-268`，走 **Dynamo → FX graph → `split_module`** 路线：

核心是给每个 FX node 打一个"子图编号"，规则是：**遇到 split op 时，编号在它前后各自 +1**。

```
node 序列:   a  b  c  [SPLIT]  d  e  [SPLIT]  f
子图编号:    0  0  0     1      2  2     3     4
                        ↑                ↑
                     split op 独占一个子图
```

于是自然形成 `[计算][split op][计算][split op][计算]` 的交替结构。
然后在 `:442-446`：**编号属于 split-op 的那些子图被排除在编译和捕获之外**，直接 eager 跑。

这个"前后都 +1"的技巧很关键——它保证 split op 一定单独成片，不会和相邻计算混在一张图里。

### 11.3 `SPLIT_OPS` 注册表：到底哪些算子是切点

注册表定义在 `compilation/compilation_config.py:7-16`。注册点（即"谁把自己声明为切点"）
散布在各个算子实现里：

| 注册位置 | 算子 |
|---|---|
| `layers/radix_attention.py:427` / `:475` / `:517` | 三种 attention 入口 |
| `distributed/parallel_state.py:169-171` | allreduce |
| `layers/radix_linear_attention.py:212` | 线性注意力（Mamba/GDN 系） |
| DSA indexer / DSA prefill | 稀疏注意力的索引与 prefill |
| MLA forward | MLA 的融合入口 |
| transformers MoE | HF transformers 风格的 MoE |
| DeepEP / Mooncake MoE op | **条件注册**（只在启用对应后端时才是切点） |

"条件注册"意味着 **`SPLIT_OPS` 的内容依赖运行时配置**——同一个模型在开/关 DeepEP 时
切出来的片数不同，图数量和编译时间也不同。排查"为什么启动时间差这么多"时要想到这点。

### 11.4 陷阱一：`compile_sizes` 是死代码

`compilation/cuda_piecewise_backend.py:84`：

```python
self.compile_sizes = set([])      # ← 硬编码空集合
```

这一行的后果是：**"为每个具体 shape 单独做一次 Inductor 编译"这条路径永远不会走到**，
尽管文件里的 docstring 仍在描述它。

实际行为：每个分片只有**一次通用（dynamic shape）编译**，所有 shape 共用。

实践含义：
- 不要指望 piecewise 会给热门 batch size 生成特化 kernel；
- 如果你在做性能分析、发现 piecewise 的 kernel 不如 FULL 快，这就是一个原因；
- 想改这块之前，先确认这行硬编码是有意为之还是待清理的遗留。

### 11.5 陷阱二："warmup"这个词被用于两件完全不同的事

读 piecewise 代码时最容易被绕晕的一点：

| 名字 | 是什么 | 在哪 |
|---|---|---|
| **外层 warmup** | `enable_torch_compile_warmup()` 阶段：把所有 shape 都**编译**一遍，但**一张图都不录** | 编译阶段总控 |
| **内层 warmup** | `entry.num_finished_warmup < 1` 计数器：每个 shape 在录图前先**执行一次丢弃** | `cuda_piecewise_backend.py` |

两者目的不同：外层是为了"把编译成本前置到启动阶段，避免线上首次请求卡住"；
内层是为了"让惰性初始化在图外完成"（和第 4.3 节讲的两次 warmup 同理）。

看日志或读代码时如果不区分这两个 warmup，会得出"warmup 逻辑重复了"的错误结论。

### 11.6 piecewise 的三个平台实现

| 文件 | 平台 |
|---|---|
| `compilation/cuda_piecewise_backend.py` (230) | CUDA / HIP |
| `compilation/npu_piecewise_backend.py` (110) | Ascend NPU（ACL Graph） |
| `compilation/xpu_piecewise_backend.py` (113) | Intel XPU |

三者结构相同（管理 `shape → 图` 的字典、warmup 计数、replay 分发），差别在于底层
graph API 与可用的编译器。

### 11.7 相关但不同的东西：`--enable-torch-compile`

`compilation/torch_compile_decoration.py` 是**旧的**、独立于 piecewise 的路径：
用 `torch.compile` 装饰整个 decode forward，然后照常用 FULL 模式整图捕获。

区别：

| | `--enable-torch-compile`（旧） | piecewise |
|---|---|---|
| 目标 | 让 Inductor 融合 kernel，减少 kernel 数量 | 让形状敏感算子留在图外 |
| 图结构 | 仍然是**一整张**图 | 多张小图 |
| 与 `torch_compile_max_bs` | 受它限制（见 `get_batch_sizes_to_capture`） | 无关 |

两者概念上正交，但实际上很少同时开——`get_batch_sizes_to_capture` 里的 `compile_bs`
就是给旧路径算的。

---

## 12. 深水区二：Breakable CUDA Graph（BCG）

### 12.1 一句话定位

> BCG 想拿到 piecewise 的"图外 eager 算子"能力，但**不付 torch.compile 的代价**
> （不编译、不 Dynamo、不 graph break 风险、启动快得多）。

它是 CUDA 平台上 **prefill 的默认后端**（`cuda_graph_config.py:112-121`）。

### 12.2 机制：`@eager_on_graph` 装饰器

实现在 `runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:216-269`：

```python
@eager_on_graph(enable, capture_stub=None)
def some_op(...):
    ...
```

被装饰的函数在 capture 期间的行为：

```
① 结束当前正在录的 CUDAGraph 段            （段 N 封口）
② 插入 barrier（保证前面的 device 操作完成）
③ 真实执行一次这个 eager 函数              ← 目的是把输出张量的地址固定下来
④ 注册一个 replay_fn：
      "每次 replay 到这里时，重新 eager 执行一次，
        再把新结果 copy_ 进那块固定的桥接 buffer"
⑤ 开始录新的 CUDAGraph 段                  （段 N+1）
```

于是一次 forward 变成：`[段1 图] → eager → [段2 图] → eager → [段3 图]`，
replay 时按同样顺序依次 `段1.replay()` → `replay_fn()` → `段2.replay()` → …

`capture_stub` 参数用于那些"eager 执行本身在 capture 阶段不能跑"的算子——
提供一个只在 capture 期替身执行的桩函数。

### 12.3 为什么内存池不会被提前释放：`use_count` 钉住

这是 BCG 最精巧也最容易改坏的地方。

多个段共用同一个 CUDA mempool。每个段（`CUDAGraph` 对象）的析构函数都会对这个 mempool
调 `releasePool`。PyTorch 的 mempool 是引用计数的：**只要还有一个段活着，`use_count > 0`，
池就不会被真正释放**。

BCG 依赖这一点来保证 `weak_ref_tensor` 视图（段之间传递中间张量用的弱引用视图）
在整个 forward 期间地址有效。

推论：**任何"提前析构某一段"的改动都会让后续段读到已释放显存**。改 BCG 的段管理逻辑时，
必须保证所有段的生命周期一致（同生同死）。

### 12.4 已知不兼容

`breakable_cuda_graph_backend.py:88-90`：**BCG 拒绝与 memory saver 同时使用**。
memory saver 会在运行期把权重换出/换入，与 BCG 依赖的"池地址长期稳定"直接冲突。

### 12.5 BCG vs TC_PIECEWISE 怎么选

| 判断依据 | 选谁 |
|---|---|
| CUDA 平台、想要短启动时间 | **BREAKABLE**（默认） |
| 非 CUDA 平台 | **TC_PIECEWISE**（BCG 的段管理依赖 CUDA API） |
| 需要 Inductor 的算子融合收益 | TC_PIECEWISE |
| 开了 memory saver | 不能用 BCG |
| 模型里有 Dynamo 处理不了的 Python 逻辑 | BREAKABLE（不走 Dynamo，天然免疫） |
| 想要最少的图外算子（最大 launch 收益） | BREAKABLE（切点由装饰器精确控制，通常比 `SPLIT_OPS` 少） |

---

## 13. 深水区三：图数量是怎么乘起来的

理解这一章，你就能预测"我这个配置会录多少张图、启动要多久、吃多少显存"。

### 13.1 基础维度：batch size 桶

由 `get_batch_sizes_to_capture` 决定（第 3.3 节）。典型 20–30 个桶。

### 13.2 乘法因子一：ragged verify token 分层（tier）

投机解码的 verify 阶段，每个请求的 token 数不固定（不同请求 accept 的草稿数不同）。
为了不为每种组合都建图，SGLang 把 token 数分成若干**tier**，每个 tier 一个
`variant_label` → 一张图。

图数 = `len(capture_bs) × len(tiers)`。

### 13.3 乘法因子二：LoRA 变体

不同 LoRA adapter 组合（尤其是 rank 不同）会导致不同的计算图，通过 `variant_label` 区分。
配合第 6.6 节的图去重，结构相同的 adapter 可以共享可执行体，但**图节点仍然各录一份**。

### 13.4 乘法因子三：DSA 双图（dense / sparse）

DSA（DeepSeek Sparse Attention，DSV4 系）在 HIP 平台上要录两套图
（`decode_cuda_graph_runner.py:260-289`）：

- **dense 图**：当 batch 内最大 `kv_len <= index_topk` 时，稀疏索引没意义，走 dense；
- **sparse 图**：否则走稀疏路径。

host 侧的分发在 `decode_cuda_graph_runner.py:580-598` 的 `_resolve_dsa_variant`：
按 batch 的最大 `kv_len` 与 `index_topk` 比较来选。

代价实测：**约 +52 张图，capture 时间约 2 倍**。这是"HIP 上开 DSA 后启动明显变慢"的
主要原因。

### 13.5 乘法因子四：pdmux 多流

pdmux（prefill/decode 在同一 GPU 上用不同 stream 复用）需要为每个 stream 单独录图，
对应 `ShapeKey.stream_idx`。图数按 stream 数线性增长。

### 13.6 乘法因子五：TBO（two batch overlap）

开 `enable_two_batch_overlap` 时，`get_batch_sizes_to_capture` 会把
`alignment_width` 强制设为 1（第 3.3 节代码），即**不按 `captured_req_width` 对齐**。
这会改变桶列表的构成，间接影响图数。

### 13.7 估算公式

```
图数 ≈ len(capture_bs)
        × len(tiers)              # ragged verify，无投机则为 1
        × len(lora_variants)      # 无 LoRA 则为 1
        × len(dsa_variants)       # 非 DSA/非 HIP 则为 1，否则 2
        × num_streams             # 非 pdmux 则为 1
      + prefill 侧图数
      + 卫星 runner 图数（draft / draft_extend / ViT / metadata glue）
```

### 13.8 target verify 不是独立 runner（重要）

一个常见误解是"投机解码的 verify 阶段有自己的 CUDA Graph runner"。
**没有。** target verify 用的是**同一个 `DecodeCudaGraphRunner`**，只是构造参数不同：

```
captured_req_width  = decode_num_tokens_per_req(...)      # 每 req 不止 1 个 token
capture_forward_mode = ForwardMode.TARGET_VERIFY
```

所以 verify 的图和 decode 的图共享同一套 buffer registry、同一个 backend、同一个内存池。
这也解释了为什么 `get_batch_sizes_to_capture` 需要 `captured_req_width` 参数并用它做对齐。

---

## 14. 深水区四：卫星 Runner

除了 decode/prefill 两个主 runner，还有一批"卫星" runner 各自管一小块。

### 14.1 投机解码 runner

| 文件 | 负责 |
|---|---|
| `speculative/eagle_draft_cuda_graph_runner.py` | EAGLE draft 模型的 decode 步 |
| `speculative/eagle_draft_extend_cuda_graph_runner.py` | draft 模型的 extend（喂入已接受 token） |
| `speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py` | 多层 MTP 的 draft extend |
| `speculative/frozen_kv_mtp_cuda_graph_runner.py` | frozen-KV MTP 变体 |

它们的图显存分别记在 `graph_memory_usage` 的 `draft_prefill` / `draft_decode` /
`draft_extend` 三个键下（第 19.1 节）。

### 14.2 ViT runner

| 文件 | 模型 |
|---|---|
| `multimodal/vit_cuda_graph_runner.py` | 通用 ViT |
| `multimodal/internvl_vit_cuda_graph_runner.py` | InternVL |
| `multimodal/kimi_k3_vit_cuda_graph_runner.py` | Kimi K3 |

**和主 runner 最大的区别：ViT runner 不分桶、不预捕获**，而是用
**按 shape 懒加载的缓存**——第一次遇到某个图像分辨率对应的 shape 时才录图，之后命中缓存。

原因很直白：图像分辨率的取值空间比 batch size 大得多且不可枚举，预先分桶捕获不现实。
代价是**第一次遇到新分辨率的请求会有一次 capture 抖动**（延迟尖刺）。

### 14.3 Metadata Glue Graph（必读的失效模式）

`runner/metadata_glue_graph.py`（106 行）。

**动机**：第 7.2 节提到，replay 路径上的 `init_forward_metadata_out_graph` 有时很重
（大量小 kernel + host 逻辑）。既然它也是一串 device 操作，能不能**单独给它建一张小图**，
把这部分的 launch 开销也省掉？这就是 metadata glue graph。

**致命前提**（`metadata_glue_graph.py:24-31` 的注释）：
capture 只录 device 操作，所以**元数据里那些由 host 计算出来的 plan 输入会被冻结在
capture 时刻的值**。

**为什么特别危险**（注释原话的意思）：对 DFlash 系投机解码，
> "capture succeeds, outputs stay correct, only accept length collapses."

即：图录得成功、不报错、输出的 token 也合法，**只有接受长度悄悄塌到很低**，
表现为"投机解码开了但一点都不快"。没有任何异常日志。

因此 DFlash 系必须被**硬排除**在 metadata glue graph 之外。

**这是全文档最值得记住的一条经验**：CUDA Graph 的 host 冻结问题最坏的形式不是崩溃，
而是"功能正常、指标退化"。所以给任何新东西加图之前，先问：
**"这段代码里有没有 host 计算出来的、会随 batch 变化的值？"**

### 14.4 `graph_shared_output.py`

`model_executor/graph_shared_output.py`（70 行）：让多个 runner 共享同一块输出 buffer。
和 `input_buffers.py` 是对称设计（一个管输入共享，一个管输出共享），
在 `cuda_graph_setup.py` 的编排里是**第一步**创建的（第 16 章）。

---

## 15. 非 CUDA 设备：CPU / NPU / XPU

### 15.1 CPU

`model_executor/cpu_graph_runner.py`。CPU 上没有 kernel launch 开销问题，
这个 runner 的价值在于**接口对齐**（让上层代码不用区分设备）以及少量的图级优化。

### 15.2 NPU（Ascend）

`hardware_backend/npu/graph_runner/*` + `compilation/npu_piecewise_backend.py`（110 行）。
底层是 ACL Graph 而不是 CUDA Graph，但上层契约一致。

NPU 的两个特殊点（都在前面出现过，这里汇总）：
- **`get_cuda_graph_seq_len_fill_value()` 返回 0**（`ascend_backend.py:794-795`），
  不是通用的 1；
- **`input_buffers.share_input_buffer()` 的合并被禁用**，原因是精度问题。

### 15.3 XPU（Intel）

`hardware_backend/xpu/graph_runner/*` + `compilation/xpu_piecewise_backend.py`（113 行）。

**硬约束**：XPU 的 decode backend **必须是 `full`**，选其他值会直接
`raise ValueError`（`runner_backend/utils.py:79-81`）——不是 warn 降级，是硬失败。

### 15.4 三个平台的默认 prefill backend

回顾 `default_prefill_backend()`（`cuda_graph_config.py:112-121`）：

| 平台 | prefill 默认 | decode 默认 |
|---|---|---|
| CUDA (NVIDIA) | `breakable` | `full` |
| HIP (AMD) | `breakable` | `full` |
| NPU | `tc_piecewise` | `full` |
| XPU | `tc_piecewise` | `full`（强制） |
| CPU | — | — |

---

## 16. 启动顺序：一切是在什么时候发生的

`model_executor/model_runner_components/cuda_graph_setup.py:160-281` 是整套体系的编排入口。
**顺序本身就是设计**，每一步的位置都有理由。

```
① GraphSharedOutput 初始化
      理由：输出 buffer 要先存在，后面所有 runner 才能引用它

② 创建 EagerRunner
      理由（关键）：必须在 attention backend 之后、在图 runner 之前。
      因为 input_buffers 是"先到先得"，让 eager 路径的 buffer 成为 canonical，
      保证 eager 与图路径读写同一块存储。

③ MoE / DeepGEMM 显存预算确定
      理由：这些模块会预分配大块 workspace，必须在 capture 之前定下来，
            否则 capture 时的可用显存不确定。

④ Prefill capture
      理由：先 prefill 后 decode。注意 prefill runner 不调 init_cuda_graph_state，
            它依赖 decode runner 分配的 attention buffer —— 所以这两步的
            实际依赖关系比顺序看起来复杂，attention buffer 的分配发生在
            decode runner 的构造过程中。

⑤ Decode capture

⑥ 注册 forward hooks
      理由（关键）：必须在 capture 之后。否则 hook 里的算子会被录进图，
            造成"每次 replay 都执行一遍 hook 的 device 操作"这种诡异行为。

⑦ Symmetric memory 预分配

⑧ Canary 初始化
```

一个容易忽略的事实：**capture 不是由 `cuda_graph_setup.py` 直接触发的，而是在各个
runner 的构造函数里触发**。`cuda_graph_setup.py` 只负责"按什么顺序构造谁"。
所以看调用栈时，capture 会出现在 `DecodeCudaGraphRunner.__init__` 里面。

### 16.1 从这个顺序能推出的排障线索

| 现象 | 可能与顺序相关的原因 |
|---|---|
| eager 和图路径结果不一致 | ② 的 canonical buffer 归属出了问题 |
| replay 时有莫名的额外 kernel | ⑥ 的 hook 被录进了图（顺序被打乱） |
| capture 阶段 OOM 但显存看起来够 | ③ 的预算还没确定就 capture 了 |
| prefill 图能录但读到脏元数据 | attention buffer 分配时机（④ 的注释） |

---

## 17. 配置面板：参数、别名、环境变量

### 17.1 主开关

| 参数 | 说明 |
|---|---|
| `--disable-cuda-graph` | 总开关：全部关闭（decode + prefill） |
| `--disable-cuda-graph-padding` | 关闭 padding：只有精确命中桶的 batch 才用图，其余走 eager |
| `--enable-cudagraph-gc` | 不在 capture 期间冻结 GC（默认冻结，见 `freeze_gc`） |

### 17.2 backend 选择与形状控制

统一配置对象在 `model_executor/cuda_graph_config.py`（283 行），它把 decode 和 prefill
两侧的配置组织成 `cuda_graph_config.decode.*` / `cuda_graph_config.prefill.*`：

| 配置项 | 取值 | 默认 |
|---|---|---|
| decode backend | `full` / `breakable` / `tc_piecewise`(→降级为 full) / `disabled` | `full`（`:147-149`） |
| prefill backend | 同上 | `default_prefill_backend()`（`:112-121`） |
| `decode.bs` | batch size 桶列表 | 内置列表，可用 `--cuda-graph-bs` 覆盖 |
| `--cuda-graph-max-bs` | 顶桶上限 | 按 `req_to_token_pool.size` 推导 |
| `--torch-compile-max-bs` | 旧 torch.compile 路径的 bs 上限 | 见 `get_batch_sizes_to_capture` |

**桶列表的实际生效值受三重过滤**（第 3.3 节代码）：
1. `bs * alignment_width % mul_base == 0`（对齐倍数）；
2. `bs <= num_max_requests`（不超过 req pool 容量）；
3. 去重排序，并在必要时补上 `num_max_requests` 作为顶桶。

所以 `--cuda-graph-bs 1,3,5,7` 这种写法很可能大部分被过滤掉。启动日志里会打印实际
使用的桶列表，以那个为准。

### 17.3 废弃/改名的参数

| 旧写法 | 现在 |
|---|---|
| `--enable-piecewise-cuda-graph` | 改为设置 prefill backend = `tc_piecewise` |
| `--disable-piecewise-cuda-graph` | 改为设置 prefill backend = `full` 或 `disabled` |
| `--enforce-piecewise-cuda-graph` | 被"**locked**"机制取代（第 18.4 节）：**显式设置了 backend 就等于 lock**，所有自动禁用规则都会跳过 |

老脚本里如果还带着这些参数，行为可能与预期不同，建议改用新的 backend 选择方式。

### 17.4 调优决策树

```
启动太慢？
├─ 缩短 bs 列表（--cuda-graph-bs）
├─ 压低 --cuda-graph-max-bs
├─ prefill backend 改 breakable（如果之前是 tc_piecewise，省掉编译）
└─ HIP + DSA：接受 2x，或关掉 DSA 双图

显存不够（capture 阶段 OOM）？
├─ 压低 --cuda-graph-max-bs        ← 性价比最高
├─ 稀疏化 bs 列表
├─ prefill backend 改 disabled
├─ 降 --mem-fraction-static
└─ 最后才 --disable-cuda-graph

decode 延迟没改善？
├─ 确认真的命中了图（看是否走 eager，第 7.4 节的拒绝条件）
├─ 确认 batch size 落在桶里（--disable-cuda-graph-padding 会大幅降低命中率）
└─ 看 init_forward_metadata_out_graph 是不是成了新瓶颈（第 7.2 节）
```

---

## 18. 自动禁用规则全表

`arg_groups/cuda_graph_hook.py`（588 行）负责"根据其他配置自动关掉某些 CUDA Graph 能力"。
这是**线上"我明明开了图为什么没生效"问题的第一查处**。

### 18.1 规则数量总览

| 目标 | 规则数 | 位置 | 是否打日志 |
|---|---|---|---|
| prefill `tc_piecewise` | **20 条** | `:176-247` | **不打日志（静默）** |
| prefill `breakable` | **6 条** | `:272-308` | 打 warning |
| prefill `full` | **0 条** | `:329`（`rules = []`） | — |

三个数字本身就是信息量：
- `tc_piecewise` 规则最多且**静默**——这是排障陷阱，配置被悄悄关掉，日志里毫无痕迹；
- `full` 一条规则都没有——因为 full 不依赖 torch.compile / 段管理，兼容面最广；
- 相比 `tc_piecewise`，`breakable` 的兼容问题少得多（6 vs 20），这也是它成为 CUDA 默认的原因。

### 18.2 典型的禁用触发项

规则覆盖的配置维度大致分几类（具体条目见源码，会随版本变化）：

| 类别 | 例子 |
|---|---|
| 模型结构 | SWA、线性注意力/SSM、稀疏注意力、特定 MoE 实现 |
| 并行/通信 | 某些 EP/DP 组合、特定 allreduce 实现、pdmux |
| 投机解码 | 特定 spec 算法与 piecewise 不兼容 |
| 量化 | 某些量化 kernel 不能被 Inductor 处理 |
| 显存特性 | memory saver（BCG 硬拒，`breakable_cuda_graph_backend.py:88-90`） |
| attention backend | `torch_native`、`flex_attention`（在 `attention_hook.py:57-96` 硬编码禁用） |

### 18.3 排查方法

因为 `tc_piecewise` 的禁用是静默的，判断"到底有没有生效"最可靠的办法是：
1. 看启动日志里有没有 capture 相关的进度输出；
2. 查 `/get_internal_state` 里 `memory_usage.graph` 的 `prefill` 键有没有值（第 19.1 节）；
3. 直接把 backend 显式设成 `tc_piecewise`（触发 lock，见下节），如果它本来会被禁用，
   现在就会保留配置（可能因此暴露出真正的不兼容错误）。

### 18.4 `locked` 机制：显式设置 = 我知道我在做什么

**每一条自动禁用规则在检查前，都会先看这个配置键是否 `locked`。如果 `locked`，规则整条跳过。**

而 `locked` 的来源就是"**用户在命令行里显式设置了这个值**"。

这是老参数 `--enforce-piecewise-cuda-graph` 的泛化版本：以前只能对 piecewise 说
"我不管兼容性检查，强制开"，现在**任何显式设置的配置键都自动获得这个语义**。

含义与风险：
- **好处**：想做实验、想绕过保守的兼容判断时，显式写出来就行；
- **风险**：显式设置意味着你放弃了自动保护网，撞到真正的不兼容时会是崩溃或
  静默错误，而不是优雅降级；
- **提醒**：写自动化部署脚本时不要"为了明确而把所有值都显式写一遍"——那会把
  所有安全网都关掉。

---

## 19. 可观测性与排障

### 19.1 图显存怎么查

`model_executor/graph_memory_usage.py`（67 行）负责统计。
**注意：它从不打印聚合表格**，只通过三个渠道暴露：

| 渠道 | 位置 |
|---|---|
| HTTP | `GET /get_internal_state` → `memory_usage.graph` |
| Prometheus | gauge `sglang:graph_memory_usage_gb` |
| 内部负载查询 | load-inquirer 的 `graph_gb` 字段 |

`memory_usage.graph` 的键固定为六个：

```python
("prefill", "decode", "target_verify", "draft_prefill", "draft_decode", "draft_extend")
```

这六个键正好对应第 13.8、14.1 节讲的 runner 划分。看到 `target_verify` 有值就说明
投机 verify 的图录上了；看到 `prefill` 是 0 或缺失，就要怀疑第 18 章的静默禁用。

### 19.2 capture 失败的提示文案

`runner_utils/__init__.py` 里定义了两条提示：
- `CUDA_GRAPH_CAPTURE_FAILED_MSG`（`:14-21`）
- `PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG`（`:23-28`）

它们会在 capture 抛异常时打印，内容主要是引导你调 `--mem-fraction-static`
和缩减 bs 列表。看到这两条时，先按第 17.4 节的显存分支处理。

### 19.3 常见问题速查

**Q1：启动时卡在 "Capturing batches..." 很久**
正常现象。图数量看第 13.7 节的估算公式。想加速见第 17.4 节。
如果是 `tc_piecewise`，大头是 Inductor 编译而不是 capture 本身。

**Q2：capture 阶段 OOM**
图显存不在 `mem_fraction_static` 预算内（第 10.4 节）。优先压 `--cuda-graph-max-bs`。

**Q3：开了图但延迟没改善**
按顺序排查：
1. 是不是根本没命中图？看第 7.4 节的六个拒绝条件；
2. 是不是被第 18 章的规则静默禁用了？查 `memory_usage.graph`；
3. 是不是 `--disable-cuda-graph-padding` 导致命中率极低？
4. 是不是 `init_forward_metadata_out_graph` 成了新瓶颈（第 7.2 节）？用 profiler 看
   replay 前那段 CPU 时间。

**Q4：图路径和 eager 路径结果不一致**
高度怀疑两处：
1. `PaddingPolicy` 选错，padding 区污染了真实数据（第 8.3、7.3 节）；
2. `input_buffers` 的 canonical 归属错了，两条路径读写不同存储（第 8.1、16 章 ②）。

**Q5：投机解码开了但 accept length 很低、没有报错**
这是第 14.3 节讲的 metadata glue graph 失效模式的典型症状。检查你的 spec 算法是否
在 glue graph 的排除名单里。这类问题**永远不会报错**，只能靠指标发现。

**Q6：换了 attention backend 后 capture 就崩**
检查三件事：
1. 有没有实现新的三方法（`out_graph` / `in_graph`），而不是旧的
   `*_capture_cuda_graph` / `*_replay_cuda_graph`（第 9.1 节）；
2. `init_forward_metadata_in_graph` 里有没有 `.item()` / 动态 `torch.empty()`（第 9.2 节）；
3. `page_size > 1` 时 buffer 尺寸算法是否符合你这个 backend 的写入粒度（第 9.4 节）。

**Q7：`--cuda-graph-bs` 写的值没生效**
三重过滤（第 17.2 节）。以启动日志打印的实际桶列表为准。

**Q8：BCG 报错说和 memory saver 冲突**
`breakable_cuda_graph_backend.py:88-90` 的硬拒。把 prefill backend 换成 `full`
或 `tc_piecewise`，或者不用 memory saver。

**Q9：ViT 请求偶发延迟尖刺**
第 14.2 节：ViT runner 是懒加载捕获，第一次遇到新的图像 shape 会现场 capture。
不是 bug。

**Q10：改了 forward hook 之后 replay 行为诡异**
第 16 章第 ⑥ 步：hook 必须在 capture 之后注册。如果你的代码提前注册了，
hook 里的 device 操作会被录进图。

---

## 20. 附录

### 20.1 关键文件速查表

| 想改什么 | 去哪个文件 |
|---|---|
| 加一个新的捕获阶段（新 runner） | `runner/` 下新建，参考 `decode_cuda_graph_runner.py` |
| 加一种新的捕获机制 | `runner_backend/` 下实现 `BaseCudaGraphBackend` 六个方法 |
| 加一个新的图输入 | `runner_utils/buffers.py` 加字段 + `cuda_graph_buffer_registry.py` 的 builder 加 slot |
| 改 padding 语义 | `cuda_graph_buffer_registry.py` 的 `PaddingPolicy` |
| 改 bs 桶策略 | `runner/base_cuda_graph_runner.py: get_batch_sizes_to_capture` |
| 改 backend 默认/降级 | `cuda_graph_config.py` + `runner_backend/utils.py` |
| 加/改自动禁用规则 | `arg_groups/cuda_graph_hook.py` |
| 改 piecewise 切点 | `compilation/compilation_config.py: SPLIT_OPS` + 各算子的注册处 |
| 改 BCG 段管理 | `runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py` |
| 改启动编排顺序 | `model_runner_components/cuda_graph_setup.py` |
| 加显存统计维度 | `graph_memory_usage.py` |

### 20.2 术语对照

| 术语 | 含义 |
|---|---|
| capture | 录制图。CPU 走一遍 forward，kernel 不真执行 |
| replay | 重放图。一次 launch 执行全部已录 kernel |
| bucket / 桶 | 预先捕获的某个 batch size |
| padding | 把真实 batch 补到桶大小，多出的槽位算但丢弃结果 |
| ShapeKey | `(size, stream_idx, variant_label, dsa_variant)`，图的唯一标识 |
| variant | 同一 size 下的语义变体（LoRA / tier / dsa） |
| BCG | Breakable CUDA Graph，分段捕获 + eager 桥接 |
| piecewise | 用 torch.compile 按 `SPLIT_OPS` 切片后逐片建图 |
| split op | piecewise 的切点算子，本身不建图 |
| glue graph | 只为 attention 元数据准备建的小图 |
| graph-resident | "图常驻"：地址在整个进程生命周期内不变的 buffer |
| canonical buffer | `input_buffers` 共享池里被复用的那一份物理存储 |
| locked | 用户显式设置了某配置键 → 跳过所有自动禁用规则 |
| tier | ragged verify 里的 token 数分层 |

### 20.3 三条"不要这样做"

1. **不要在 `init_forward_metadata_in_graph` 里做 host 计算。**
   包括 `.item()`、`.cpu()`、`.tolist()`、`len(tensor)` 之后用于分支、动态 `torch.empty()`。
   它不会报错，只会静默错。

2. **不要重新绑定静态 buffer。**
   `self.buffers.seq_lens = new_tensor` 是错的，必须 `copy_()`。
   图记的是地址，重新绑定后图仍读旧地址。

3. **不要假设"新增一个变体维度只是多几张图"。**
   它是乘法（第 13.7 节）。加一个二值变体就是图数翻倍、capture 时间翻倍、
   显存池峰值可能上升。加之前先算一下。

### 20.4 与本目录其他文档的关系

| 文档 | 关系 |
|---|---|
| `spec_decoding_cache_centralized_vs_pd.md` | 投机解码的 KV/slot 管理；本文第 13.8、14.1 节的图侧视角与其互补 |
| `pd_decode_radix_cache_hicache_scheme.md` | PD 分离下 D 侧的 cache 体系 |
| `decode_instance_prefill_capability.md` | PD 分离下 D 侧为何不跑 prefill forward |

### 20.5 本次更新相对旧版本的主要修正

| 旧版本的说法 | 现在的事实 |
|---|---|
| 核心类是 `model_executor/cuda_graph_runner.py: CudaGraphRunner` | 该文件已删除；拆成 `runner/` + `runner_backend/` 两层 |
| buffer 体系就是 `DecodeInputBuffers` | 三层体系，填充语义在 `CudaGraphBufferRegistry` |
| 只有 full 和 piecewise 两种模式 | 四种 backend，含 `BREAKABLE`（CUDA 上 prefill 默认） |
| attention 实现 `*_capture_cuda_graph` / `*_replay_cuda_graph` | 这两个方法已从 ABC 删除，改为三方法契约 |
| 用 `--enable/disable/enforce-piecewise-cuda-graph` | 改为 backend 选择 + `locked` 机制 |
| 只讲了 piecewise 的自动禁用条件 | 三个 backend 各有规则集（20 / 6 / 0 条），且 tc_piecewise 是静默的 |
| 未涉及 | 图去重、ragged verify tier、DSA 双图、metadata glue graph、ViT 懒捕获、共享内存池的 `max()` 语义 |
