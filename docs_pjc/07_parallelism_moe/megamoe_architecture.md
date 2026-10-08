# MegaMoE 方案系统梳理（DeepGEMM 融合式 EP-MoE 大内核）

> 本文系统梳理 SGLang 中的 **MegaMoE**（`mega_moe` / `--moe-a2a-backend megamoe`）方案：它是什么、
> 解决什么问题、SGLang 侧如何接入、DeepGEMM 侧的融合大内核如何工作、性能与约束在哪里。
> 面向没有 MegaMoE 背景的读者，由浅入深；关键结论都给出 `文件:行号` 索引，便于跳转核对。
>
> 关联文档：`deepep_dispatch_combine_architecture.md`（DeepEP all-to-all，MegaMoE 的对照物）、
> `deepseek_v4_model_architecture.md`（DSV4 MoE 层本体）。

---

## 0. 一句话概览

**MegaMoE 把「EP 专家并行的 all-to-all dispatch + 分组 GEMM L1(gate/up) + SwiGLU 激活 + 分组 GEMM L2(down) + all-to-all combine」这一整条 MoE 前向链路，融合进了一个 SM100(Blackwell B200) 上一次启动、持续驻留（persistent）的 CUDA 大内核里。**

对比 DeepEP：DeepEP 是"通信库 + 若干独立 GEMM kernel"，每一步都要单独 launch、通信与计算靠 Host 编排来重叠；MegaMoE 是"通信+计算合体的一个 megakernel"，用 **warp 专业化 + 环形缓冲区（ring buffer）** 在同一个驻留 grid 内让 all-to-all 与 GEMM 天然重叠，彻底消除多次 kernel launch 的开销和 Host 往返。

```
        DeepEP（多 kernel + Host 编排）              MegaMoE（单 megakernel）
   ┌──────────┐  ┌──────────┐  ┌──────────┐      ┌───────────────────────────────┐
   │ dispatch │→ │ GEMM L1  │→ │ swiglu   │      │  一次 launch，grid 常驻：        │
   │  kernel  │  │  kernel  │  │  kernel  │      │  DISPATCH warp ─┐               │
   └──────────┘  └──────────┘  └──────────┘      │  TMA-load  warp ├→ ring buffer  │
   ┌──────────┐  ┌──────────┐                    │  MMA       warp ┤   ↓           │
   │ GEMM L2  │→ │ combine  │  ← 每步都 launch    │  EPILOGUE  warp ┘  L1→swiglu→L2 │
   │  kernel  │  │  kernel  │    Host 同步/重叠   │            └→ combine 直写远端    │
   └──────────┘  └──────────┘                    └───────────────────────────────┘
```

---

## 1. 背景：EP-MoE 的痛点，MegaMoE 想省什么

大规模 MoE（如 DeepSeek-V3/V4，256~384 专家、top-6~8）在专家并行（EP）下，每个 token 要被路由到分布在不同 GPU 上的专家。一次 MoE 层的标准流程：

| 阶段 | 做什么 | DeepEP 的实现 | 开销来源 |
|------|--------|--------------|---------|
| dispatch | 按 topk 把 token all-to-all 发到专家所在 rank | `buffer.dispatch`（NVL+RDMA） | 独立 kernel + Host 编排 |
| GEMM L1 | 每个专家对收到的 token 做 gate/up 投影 | 分组 GEMM（`m_indices`/`masked_m`） | 独立 kernel launch |
| 激活 | SwiGLU：`silu(gate)*up` | 独立 activation kernel | 独立 kernel launch |
| GEMM L2 | down 投影 | 分组 GEMM | 独立 kernel launch |
| combine | 反向 all-to-all + topk 加权求和 | `buffer.combine` | 独立 kernel + Host 编排 |

痛点有三：
1. **launch 开销**：每层 5 次以上 kernel launch，decode 小 batch 下 launch 开销占比高。
2. **通信/计算重叠难**：dispatch 的 all-to-all 和 GEMM 分属不同 kernel，重叠要靠 Host 侧多流编排、CUDA graph 捕获，脆弱且有气泡。
3. **中间张量落显存**：L1 输出→激活输入→L2 输入要在 HBM 里来回搬运。

**MegaMoE 的答案**：把这五步塞进一个 megakernel。通信用 warp 专业化的 DISPATCH warp 直接在 kernel 内经对称内存（symmetric memory）NVLink 拉取远端 token；GEMM/激活用其它 warp，两者通过片上 ring buffer 生产-消费，通信与计算在 SM 层面天然流水重叠；L1 输出经 SwiGLU 后**原地**变成 L2 输入（同一张量，宽度砍半），不落 HBM。

---

## 2. 全景架构：SGLang 编排层 vs DeepGEMM 计算层

MegaMoE 横跨两层代码，职责边界清晰：

```
┌─────────────────────────── SGLang（编排/接入层）─────────────────────────────┐
│  server_args.py           --moe-a2a-backend megamoe  → 强制 EP==TP           │
│  arg_groups/overrides.py  SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE 自动改后端         │
│  layers/moe/utils.py      MoeA2ABackend.MEGAMOE / is_megamoe()               │
│  models/deepseek_v2.py    DeepseekV2MoE.forward 首先判 should_use_mega_moe    │
│  layers/moe/mega_moe.py   ★核心接入★ should_use / forward / weight-build      │
│  layers/quantization/fp8.py  权重加载后钩子 → build_mega_moe_experts_weights   │
│  jit_kernel/dsv4/mega_moe_pre_dispatch  ★自带★ bf16→fp8 量化+打包进 symm buf   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ deep_gemm.get_symm_buffer_for_mega_moe(...)
                               │ deep_gemm.fp8_fp4_mega_moe(y, l1_w, l2_w, buf)
                               ▼
┌─────────────────────────── DeepGEMM（计算/通信层，vendored 2.6.1）──────────────┐
│  deep_gemm/mega/__init__.py       SymmBuffer / transform_weights_for_mega_moe  │
│  csrc/apis/mega.hpp               host 入口、buffer 尺寸、slice 成 12 个视图      │
│  impls/sm100_fp8_fp4_mega_moe.cuh ★megakernel★ warp 专业化 dispatch+GEMM+combine │
│  impls/sm100_bf16_mega_moe.cuh    BF16 版（无 scale factor）                    │
│  scheduler/mega_moe.cuh           跨 SM 的 L1/L2 任务动态领取调度               │
│  layout/mega_moe.cuh              symm buffer 内存布局 / TokenSrcMetadata       │
│  comm/barrier.cuh + layout/sym_buffer.cuh  NVLink barrier + 对称堆指针翻译       │
│  csrc/jit_kernels/heuristics/mega_moe.hpp  block_m/tile/stage 启发式            │
└───────────────────────────────────────────────────────────────────────────────┘
```

**一个重要的版本事实（务必记住）**：仓库 vendored 的 DeepGEMM 是 **2.6.1**，它提供 BF16 与 FP8×FP4 两条融合大内核，且 **dispatch 是在大内核内部完成的**。这个 vendored 副本里 **没有** `deep_gemm.mega_moe_pre_dispatch` 函数，也 **没有** `DG_USE_FP4_ACTS`/`DG_USE_MXF4_KIND` 环境变量（`grep` 无匹配）。SGLang 的 FP8 路径靠**自带的** jit kernel `sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh` 做前置量化打包；FP4-activation 路径（`SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS`）才依赖一个**比 vendored 更新**、暴露 `deep_gemm.mega_moe_pre_dispatch` 与 `DG_USE_*` 的 DeepGEMM 构建。文档下文凡涉 FP4 路径，均指这个更新版。

---

## 3. 如何启用：三条路径 + EP==TP 约束

### 3.1 显式开关 `--moe-a2a-backend megamoe`

- `server_args.py:277` 把 `"megamoe"` 加入 `MOE_A2A_BACKEND_CHOICES`；`:1928-1942` 定义 `moe_a2a_backend` 字段（`Literal` 含 `"megamoe"`，`:1937`）。
- `server_args.py:5862-5868` 后处理钩子：当 `a2a_backend == "megamoe"` 时，
  - 若 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY` 未显式设置，**强制置 True**；
  - 打日志说明 **EP size 被调整为等于 TP size**。

### 3.2 环境变量隐式开关（推荐的一键开）

`SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1`（`environ.py:1034`，默认 False）。在 `arg_groups/overrides.py:2087-2092` 的 `_a2a_backend_overrides` 里：若该 env 为真且当前后端不是 megamoe，则**自动改** `moe_a2a_backend = "megamoe"`。

### 3.3 EP == TP 的来历（关键约束）

`overrides.py:2072-2074` 的 `_A2A_EP_SPANNING_BACKENDS` frozenset 含 `"megamoe"`；`_a2a_ep_size`（`:2098-2102`）对任何 spanning 后端设 `ep_size = tp_size`。这就是 server_args 日志里 "expert parallel size is adjusted to be the same as the tensor parallel size" 的实现。**MegaMoE 下 EP 组 = TP 组**，因为对称内存 all-to-all 建立在同一批 NVLink 互连的 rank 上。

### 3.4 环境变量清单

| 环境变量 | 位置 | 默认 | 作用 |
|---------|------|------|------|
| `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` | `environ.py:1034` | False | 主开关，自动切 `--moe-a2a-backend megamoe`。 |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` | `environ.py:1035` | 1024 | 每 rank token 上限：既用于 symm buffer 定容，也是准入闸门（超限回退非 mega 路径）。 |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS` | `environ.py:1041` | False | 激活按 E2M1(FP4) 打包（而非 FP8 E4M3），symm buffer 占用减半；同时 `os.environ.setdefault("DG_USE_FP4_ACTS","1")` 透传给（更新版）DeepGEMM。 |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND` | `environ.py:1046` | False | L1+L2 主循环从 `mxf8f6f4`(K=32) 切到 `mxf4`(K=64 dense)；需与 FP4_ACTS 同开，否则 DeepGEMM host 侧断言。 |
| `SGLANG_OPT_FIX_MEGA_MOE_MEMORY` | `environ.py:1047` | False | 省显存的权重构建变体（w13 buffer 在 deep-ep 与 mega 两路共享）；backend==megamoe 时自动置 True。 |

---

## 4. SGLang 侧调用链（一次 MoE 前向）

### 4.1 入口：`DeepseekV2MoE.forward`（优先短路 mega 路径）

`models/deepseek_v2.py:863-921`：

```python
def forward(self, hidden_states, forward_batch, ...):
    from sglang.srt.layers.moe.mega_moe import forward_mega_moe, should_use_mega_moe  # :872
    if should_use_mega_moe(self, hidden_states):                                       # :874
        return forward_mega_moe(self, hidden_states, forward_batch,
                                input_ids_global=input_ids_global)                     # :875-880
    if not self._enable_a2a_moe:      # 普通 TP / 双流 / forward_normal
        ...
    else:                             # forward_deepep（DeepEP 路径）
        ...
```

即 mega 路径**排在所有其它路径最前**，命中就直接返回，绕开 TP 与 DeepEP。

### 4.2 准入判定：`should_use_mega_moe`（`mega_moe.py:97-111`）

```
False，除非：get_moe_a2a_backend().is_megamoe()               # :98
       且  experts._mega_moe_weights_built == True             # :100（权重已构建）
capture 模式（CUDA graph 捕获）→ 无条件 True                   # :102-103
否则：max_tokens_per_rank <= 上限 env 才 True                  # :105-111
      max_tokens_per_rank = max(get_dp_global_num_tokens())    # DP 各 rank 全局 token 数
```

capture 模式无条件 True 是因为 CUDA graph 要用固定容量捕获，不能被运行时 token 数否决。

### 4.3 主流程：`forward_mega_moe`（`mega_moe.py:114-148`）

```
shared_output = moe._forward_shared_experts(hidden_states)   # 共享专家（可选另开 alt_stream 重叠 = SBO）
y = _run_mega_routed(...)                                    # 路由专家 = megakernel
y.add_(shared_output)                                        # 合并
```

**SBO（Shared-Body Overlap）**：当 `num_fused_shared_experts==0` 且 capture 模式时（`:122-127`），共享专家放到 `moe.alt_stream` 与路由专家的 mega 流并行，用 `wait_stream` 收尾同步。若共享专家已融进路由专家（`num_fused_shared_experts>0`），则无需另开流。

### 4.4 核心：`_run_mega_routed`（`mega_moe.py:151-267`）

```
1) gate → router_logits                                        # :165
2) moe.topk(...)  → topk_ids, topk_weights                     # :167（is_hash 时传 input_ids_global）
3) num_tokens <= NUM_MAX_TOKENS_PER_RANK 断言                   # :193-198（超限直接报错，提示调 env 或缩 bs）
4) buf = _get_mega_moe_symm_buffer(ep_group, ...)              # :200 对称内存 buffer（带缓存）
5) pre-dispatch：把 bf16 激活量化+打包进 buf.x/x_sf/topk_*      # :216-243
      FP4:  deep_gemm.mega_moe_pre_dispatch(...)                # :221（更新版 DeepGEMM）
      FP8:  sglang.jit_kernel.dsv4.mega_moe_pre_dispatch(...)   # :234（SGLang 自带 jit kernel）
6) deep_gemm.fp8_fp4_mega_moe(y, mega_l1_weights,              # :253-262 ★megakernel★
                              mega_l2_weights, buf,
                              recipe=(1,1,32), activation="swiglu",
                              activation_clamp=swiglu_limit, fast_math=True)
7) 若未把 routed_scaling 融进 topk → y.mul_(routed_scaling)     # :265-266
```

**注意 pre_dispatch 的语义**：它不是做 all-to-all，而是把**本 rank 的**激活做 per-group(32) FP8-UE8M0 量化、pad 到 `padded_max`、把 topk 行拷进 symm buffer 的**本 rank 输入区**（见第 6 节 jit kernel）。真正的跨 rank all-to-all 发生在 megakernel 内部的 DISPATCH warp。所以 "pre_dispatch" = "为 in-kernel dispatch 准备好本地输入区"。

---

## 5. 权重构建生命周期

### 5.1 触发点：FP8 权重加载后钩子

`layers/quantization/fp8.py:1555-1561`，在 `process_weights_after_loading` 里：

```python
if get_moe_a2a_backend().is_megamoe():
    from sglang.srt.layers.moe.mega_moe import build_mega_moe_experts_weights
    build_mega_moe_experts_weights(layer)
    return    # 提前返回，跳过普通 deep-gemm 的 transform_sf_into_required_layout
```

`_mega_moe_weights_built` 标志仅在 `mega_moe.py` 内读写：`:100`（准入读）、`:307`（幂等早返）、`:364`（构建完置 True）。任何跳过该钩子的 experts 层会静默走非 mega 路径。

### 5.2 `build_mega_moe_experts_weights`（`mega_moe.py:301-364`）两分支

MegaMoE 要求 L1(gate/up) 权重按 DeepGEMM 约定的**交错布局**排布，scale factor 要做 **UTCCP 4×32 转置**：

| 分支 | 条件 | 做法 | 产物 |
|------|------|------|------|
| 省显存 | `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=True`（`:337-357`） | 手工交错 L1 gate/up（`_interleave_mega_moe_gate_up:270`）+ UTCCP 转置 scale（`_transpose_mega_moe_sf_for_utccp:290`），把 `w13_weight.data`/scale **原地重绑**，让 deep-ep 路径与 mega 路径**共享同一 weight buffer**；L2 权重不动直接共享。 | `mega_l1_weights=(w13.data, w13_sf_utccp)`；`mega_l2_weights=(w2.data, w2_sf_utccp)` |
| 默认 | 否则（`:358-362`） | 直接调 DeepGEMM `transform_weights_for_mega_moe((w13,w13_sf),(w2,w2_sf))` | `l1_pair`/`l2_pair` |

省显存分支的动机：不开它时 mega 权重是 deep-ep 权重之外**另一份**拷贝（显存翻倍）；开它则两路共享一份，代价是 deep-ep 路径要用 swizzle-aware 激活 kernel 消费非转置的交错 scale（见 `moe_runner/deep_gemm.py:147-150,234,942-948,1046-1048`）。因为 megamoe 通常与 deep-ep 权重布局并存，`backend==megamoe` 时该开关默认打开。

### 5.3 交错布局 `_interleave_mega_moe_gate_up`（`mega_moe.py:270-278`）

L1 权重按 8 个一组交错成 `[gate0..7, up0..7, gate8..15, up8..15, ...]`，匹配 DeepGEMM megakernel 里 L1 的 gate/up 读取顺序（SwiGLU 需要 gate 和 up 相邻以便融合）。

---

## 6. 对称内存与 pre-dispatch（SGLang jit kernel）

### 6.1 SymmBuffer：跨 GPU 的对称堆

`deep_gemm/mega/__init__.py`：`SymmBuffer` 通过 `torch.distributed._symmetric_memory` 分配（`group.size()>1` 时），做 `symm_mem.rendezvous`，使**每个 rank 在同一虚拟偏移上都能寻址到彼此的 buffer**。单 rank（EP=1）退化为普通 torch 分配。

buffer 一整块按 `csrc/apis/mega.hpp` 切成 **12 个命名视图**：
`x, x_sf, topk_idx, topk_weights`（本 rank 输入区）、
`shared_l1/l2_acts(_sf)`（共享专家中间态）、
`l1/l2_acts(_sf)`（路由专家的 **ring buffer** 中间态）。

`_get_mega_moe_symm_buffer`（`mega_moe.py:61-94`）按 `(id(group), max_tokens, num_experts, topk, hidden, inter)` 缓存，避免每次前向重建。

### 6.2 对称堆指针翻译 `SymBuffer::map`（`layout/sym_buffer.cuh`）

```
SymBuffer<kNumRanks> { int64_t base; int64_t offsets[72]; uint32_t rank_idx; }
offsets[i] = c[i] - base
map(ptr, dst_rank) = offsets[dst_rank] + ptr    // 本地指针 → 远端 rank 同名 buffer 指针
```

megakernel 就是靠 `map` 把"写到本地 buffer"变成"经 NVLink 写到远端 rank buffer"，实现 in-kernel all-to-all。`kNumMaxRanks=72`（NVL72 规模）。

### 6.3 SGLang 自带 pre-dispatch jit kernel

`jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh`（FP8 路径专用，我已通读）：

- **量化路径**：一个 CTA 处理一个有效 token。加载 bf16 激活（每线程 16B = 8 个 bf16），组内（`kGroupSize∈{32,64,128}`，此处 32）做 absmax → UE8M0 指数 → 量化成 FP8 E4M3 写 `buf.x`；每组一个线程写 UE8M0 scale 字节到 `buf.x_sf`（行主序 int32 打包，`byte = token*num_groups+group`）。同时把该 token 的 topk 行拷进 `buf.topk_idx/weights`（`:99-104`）。
- **pad 路径**：`[num_tokens, padded_max)` 的尾部 block 把 topk 填 `(-1, 0)`（`:105-116`），保证 megakernel 读到定长输入区。
- 兼容性细节：`buf.x` dtype 兼容 int8 或 float8_e4m3fn（`:158-163`）；`buf.x_sf` 形状 `(P, G/4)` 连续 int32（`:164-171`）；`buf.topk_idx` 是 int64（`:172-175`）。

**这解释了 vendored DeepGEMM 为何不需要 `mega_moe_pre_dispatch`**：FP8 输入区由 SGLang 自己的 kernel 填。

---

## 7. DeepGEMM 融合大内核内部（核心）

`impls/sm100_fp8_fp4_mega_moe.cuh`（1460 行）。kernel `sm100_fp8_fp4_mega_moe_impl`（`:56`），`__launch_bounds__(kNumThreads, 1)`，仅在 `__CUDA_ARCH__ >= 1000`（SM100）编译。参数含 SymBuffer + **18 个 `__grid_constant__` TMA descriptor**（L1/L2 acts、acts_sf、weights、weights_sf、output 及各自 shared 变体）。

### 7.1 warp 专业化（单 grid 内分工）

| warp 角色 | 索引 | 职责 |
|-----------|------|------|
| DISPATCH | `< kNumDispatchWarps` | 统计每专家 token 数；把 src token-topk 索引写到远端 rank（`sym_buffer.map`）；`nvlink_barrier`；用 TMA 从远端 rank 拉取量化 token+SF 进本地 L1 ring buffer（轮询选 rank）；写 `TokenSrcMetadata` 供 combine；清理 workspace。 |
| TMA-load acts | `== kNumDispatchWarps` | 从 ring buffer 加载激活 + SFA。 |
| TMA-load weights | `+1` | 加载权重 + SFB。 |
| MMA-issue | `+2`（仅 leader CTA） | UTCCP scale 拷贝 + `SM100_MMA_MXF8F6F4_2x1SM_SS`（2-CTA、swap-A/B、K-major）。 |
| scheduler mainloop | `+3`（仅 leader CTA） | 跨 SM 动态领取 L1/L2 任务。 |
| EPILOGUE | `>= kNumDispatchWarps + kNumMMANonEpilogueWarps` | L1/L2 后处理 + combine。 |

数据类型（`:144-146`）：激活 `float_e4m3_t`(FP8)、权重 `float_e2m1_unpacksmem_t`(FP4)、共享权重 FP8。UMMA：`UMMA_M=256`, `UMMA_N=BLOCK_M`, `UMMA_BLOCK_K=128`, `UMMA_K=32`。

### 7.2 阶段流水（一个 token 的一生）

```
DISPATCH warp：远端 rank 的 buf.x/x_sf ──(NVLink,TMA)──▶ 本地 L1 ring buffer
                                                          │
TMA-load + MMA warp：L1 分组 GEMM（gate/up）              │
                                                          ▼
L1 EPILOGUE（:987+）：SwiGLU  silu(gate)*up*topk_weight，
                     activation_clamp，amax 归约，
                     cast FP8 E4M3，UE8M0 scale → l2_sf_buffer，
                     TMA-store 到 L2 acts buffer
     ★关键：L1 输出经 SwiGLU 后 N 宽度砍半(128→64)，与 L2 输入是同一张量，原地衔接不落 HBM
                                                          │
TMA-load + MMA warp：L2 分组 GEMM（down）                 │
                                                          ▼
L2 EPILOGUE（:1192+）：BF16 输出经 sym_buffer.map 直写【远端】combine buffer
     ★这次远端写就是 combine 的 scatter，无需单独 combine kernel
                                                          │
COMBINE（:1323+）：每 warp 从 combine_token_buffer 归约本 token 的
                  top-k(+shared) 贡献，cast BF16，TMA-store 到输出 y
```

ring buffer 用 L1/L2 的 full/empty 计数做生产-消费同步——**这正是通信与计算重叠的机制**：DISPATCH warp 在填后面的 ring slot 时，MMA warp 已在算前面 slot，SM 内流水起来。

### 7.3 NVLink barrier（`comm/barrier.cuh`）

- `grid_sync<kNumSMs, idx>`（`:21`）：协作式全 grid 同步，SM0 加 `kFinishSumTag-(kNumSMs-1)`，其余加 1，`kFinishSumTag=0x80000000`。
- `nvlink_barrier<kNumRanks, kNumSMs>`（`:46`）：grid-sync 前奏 → **仅 SM0** 用 `ptx::red_add_rel_sys(sym_buffer.map(signal, tid), ...)` 带 phase/sign 翻转向远端 rank 发信号 → 等到达 → grid-sync 收尾。**基于对称内存 NVLink，不是 NVSHMEM**。超时 `kNumTimeoutCycles = 60s @ 2GHz`。

### 7.4 调度器（`scheduler/mega_moe.cuh`，420 行）

- `MegaMoEScheduler`（`:152`）：2 个调度阶段，`ClusterTransactionBarrier`，每 lane 缓存 `stored_num_tokens_per_expert`（lane 缓存专家 `i*32+lane`）。
- `fetch_expert_recv_count()`（`:249`）：自旋等到高 32 位 == `kNumSMs*kNumRanks`（所有 SM×rank 都上报完 dispatch）。
- `get_next_task()`（`:316`）：用原子计数器动态领取 L1/L2 任务（`get_l1/l2_task_count_ptr`），L2 等它依赖的 L1 完成。
- `mainloop(num_tokens)`（`:383`）：shared-L1（与 dispatch 无关，先跑）→ `fetch_expert_recv_count` → 路由任务 → shared-L2 → sentinel。
- `get_num_l1_warmup_waves`（`:17`）：给足 L1 预热波次，避免 L1→L2 交错调度死锁。

### 7.5 布局与 block_m 启发式

- `layout/mega_moe.cuh`：`kCandidateBlockM = {8,16,32,64,96,128,192}`，`kLCMCandidateBlockM=384`（token 对齐粒度）。`TokenSrcMetadata{rank_idx, token_idx, topk_idx}`（`:40`）供 combine 写回定位。`Workspace`（`:46`）字节布局在 `:129`（grid-sync 计数、NVLink barrier 计数、signals、L1/L2/shared 任务计数）。`MegaMoEBuffer`（`:331`）含 combine_token_buffer（rank 维 = `num_topk + (shared?1:0)`）。
- `heuristics/mega_moe.hpp`：`get_block_config_for_mega_moe`（`:76`）以 `num_expected_tokens_per_expert = num_tokens*num_ranks*num_topk/num_experts` 为键选 block_m：

| 每专家期望 token 数 | block_m | store_block_m | block_k | epilogue warpgroup |
|---|---|---|---|---|
| ≤8.5 | 16 | 8 | 256 | 2 |
| ≤16.5 | 32 | 16 | 128 | 2 |
| ≤32.5 | 64 | 32 | 128 | 1 |
| ≤64.5 | 96 | 16 | 128 | 2 |
| ≤96.5 | 128 | 32 | 128 | 2 |
| 更大 | 192 | 32 | 128 | 2 |

`num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage`，断言 ≥2（`:115`）。`num_bytes_per_pull = hidden*elem_bytes` 反复减半直到 ≤4096（`kPullThreshold`，`:183`）。

---

## 8. 三条数值路径对比：BF16 / FP8×FP4 / FP4-acts+MXF4

| 路径 | 激活 dtype | 权重 dtype | 主循环 kind | 触发条件 | 备注 |
|------|-----------|-----------|------------|---------|------|
| BF16 | bf16 | bf16 | `bf16xbf16` | `mma_type='bf16xbf16'` | `sm100_bf16_mega_moe.cuh`，10 个 TMA desc（无 scale），L1 epilogue 不做 amax/FP8-cast，L2 中间态 BF16，`UMMA_BLOCK_K=64`。 |
| FP8×FP4（默认） | FP8 E4M3 | FP4 E2M1 | `mxf8f6f4`(K=32) | `mma_type='fp8xfp4'`，SGLang jit pre_dispatch 量化 FP8 | vendored 2.6.1 主路径；权重 FP4，激活 FP8。 |
| FP4-acts + MXF4 | FP4 E2M1 | FP4 E2M1 | `mxf4`(K=64 dense) | `USE_FP4_ACTS=1`(+`USE_MXF4_KIND=1`)，走 `deep_gemm.mega_moe_pre_dispatch` | 需更新版 DeepGEMM；symm buffer 占用减半；两 env 必须同开否则 host 断言。 |

`recipe=(1,1,32)` 表示 scale 粒度：per-token × per-1 × group-32。`activation_clamp=swiglu_limit` 对应 DSV4 的 SwiGLU clamp。

---

## 9. 性能分析

### 9.1 MegaMoE 省了什么

| 维度 | DeepEP | MegaMoE | 收益 |
|------|--------|---------|------|
| kernel launch 次数/层 | 5+（dispatch/L1/act/L2/combine） | **1** | 消除 launch 开销，decode 小 batch 尤其显著 |
| 通信/计算重叠 | Host 多流编排 + CUDA graph | 同 grid 内 warp 专业化 + ring buffer | 天然流水，无 Host 往返，无编排气泡 |
| L1→L2 中间张量 | 落 HBM 再读回 | **原地**（同张量宽度砍半） | 省一次 HBM 往返带宽 |
| combine | 独立 kernel + 加权求和 | L2 epilogue 直写远端 + warp 归约 | 无单独 combine launch |
| 数值精度 | FP8/BF16 | 权重 FP4 + 激活 FP8/FP4 | 更省显存/带宽，配 Blackwell FP4 tensor core |

### 9.2 适用与限制

- **仅 SM100（Blackwell B200）**：kernel `__CUDA_ARCH__ >= 1000`，`arch_major==10` 才 dispatch。Hopper 及以下不可用。
- **准入受 token 上限约束**：`num_tokens > NUM_MAX_TOKENS_PER_RANK` 直接断言失败（`mega_moe.py:193-198`），要么调大 env（吃更多对称内存），要么缩 `cuda_graph_max_bs`/`chunked_prefill_size`。适合 decode + 中等 prefill；超长 prefill 需评估。
- **EP == TP**：对称内存 all-to-all 局限于同一 NVLink 域的 rank。
- **主要用户**：DeepSeek-V2/V4 系（`mega_moe.py` docstring 明言 "shared by Deepseek V2/V4"）。测试 `tests/test_mega_moe.py` 用 DSV3 形状：hidden=7168, inter=3072, experts=384, topk=6, shared=1, EP8。

### 9.3 与 SBO / 共享专家的协作

`num_fused_shared_experts>0` 时共享专家融进路由专家一次算完；否则 capture 模式下共享专家在 `alt_stream` 与 mega 流并行（SBO，`forward_mega_moe:122-136`），进一步压掉共享专家的串行开销。

---

## 10. 约束与陷阱清单

1. **版本陷阱（最重要）**：vendored DeepGEMM 2.6.1 无 `mega_moe_pre_dispatch`/`DG_USE_FP4_ACTS`；FP8 路径靠 SGLang 自带 jit kernel，FP4 路径需更新版 DeepGEMM。混用会报缺函数/缺 env。
2. **权重必须经 fp8.py 钩子构建**：`_mega_moe_weights_built` 靠 `getattr(...,False)` 防御式读取（`mega_moe.py:100,307`，违反仓库 `no-getattr-defensive` 规则），跳过钩子的 experts 会**静默回退**非 mega 路径，难排查。
3. **EP==TP 自动改写**：显式设 `--ep-size` 与 megamoe 冲突时以 EP==TP 为准，注意日志。
4. **`FIX_MEGA_MOE_MEMORY` 双刃**：省显存但要求 deep-ep 路径配套 swizzle-aware 激活（`moe_runner/deep_gemm.py` 多处断言 `USE_JIT_EP_ACTIVATION`/`SWIGLU_CLAMP_FUSION`，且与 `SGLANG_MASKED_GEMM_FAST_ACT` 不兼容）。
5. **`USE_MXF4_KIND` 依赖 `USE_FP4_ACTS`**：单开 MXF4 会被 DeepGEMM host 断言拒绝。
6. **token 上限 env 与对称内存正相关**：调大 `NUM_MAX_TOKENS_PER_RANK` 会线性放大 symm buffer 显存占用。
7. **capture 模式无条件命中**：CUDA graph 捕获时 `should_use_mega_moe` 恒 True，容量必须按上限预留。

---

## 11. 关键文件与行号索引

### SGLang 编排层
| 文件 | 关键位置 | 内容 |
|------|---------|------|
| `layers/moe/mega_moe.py` | :97-111 / :114-148 / :151-267 / :301-364 | ★核心★ 准入 / 主流程 / megakernel 调用 / 权重构建 |
| `models/deepseek_v2.py` | :863-921 | MoE forward 优先短路 mega |
| `layers/moe/utils.py` | :28,:38,:74,:316 | `MoeA2ABackend.MEGAMOE` / `is_megamoe` / `get_moe_a2a_backend` |
| `server_args.py` | :277,:1928-1942,:5862-5868 | 后端选项 / EP==TP + FIX_MEMORY 自动置 |
| `arg_groups/overrides.py` | :2072-2074,:2087-2092,:2098-2102 | spanning 后端 / env 自动改后端 / ep_size=tp_size |
| `environ.py` | :1034-1047 | 5 个 MEGA_MOE env |
| `layers/quantization/fp8.py` | :1555-1561 | 权重加载后钩子 → build_mega_moe_experts_weights |
| `layers/moe/moe_runner/deep_gemm.py` | :147-150,:234,:942-948,:1046-1048 | FIX_MEMORY 共享布局的 deep-ep 侧配套 |
| `jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh` | 全文 | ★SGLang 自带★ FP8 量化+pad 打包 |

### DeepGEMM 计算层（vendored 2.6.1）
| 文件 | 关键位置 | 内容 |
|------|---------|------|
| `deep_gemm/mega/__init__.py` | SymmBuffer / transform_weights_for_mega_moe / fp8_fp4_mega_moe / bf16_mega_moe | Python API |
| `csrc/apis/mega.hpp` | :37,:157,:288 | host 入口 / buffer 尺寸 / 12 视图 slice |
| `impls/sm100_fp8_fp4_mega_moe.cuh` | :56,:987,:1192,:1323 | ★megakernel★ / L1 epilogue / L2 epilogue / combine |
| `impls/sm100_bf16_mega_moe.cuh` | 全文 | BF16 版（10 TMA desc，无 SF） |
| `comm/barrier.cuh` | :21,:46 | grid_sync / nvlink_barrier |
| `layout/sym_buffer.cuh` | 全文 | SymBuffer::map 对称堆指针翻译 |
| `scheduler/mega_moe.cuh` | :17,:152,:249,:316,:383 | 跨 SM 动态任务领取调度 |
| `layout/mega_moe.cuh` | :40,:46,:129,:331 | TokenSrcMetadata / Workspace / MegaMoEBuffer |
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | :76,:115,:183 | block_m / stage / pull 启发式 |
| `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` | :131 | host launch（18 TMA desc） |
| `tests/test_mega_moe.py` | 全文 | DSV3 形状参考数值 + 性能基准 |
