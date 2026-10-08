# DeepEP 与 MegaMoE：数据类型与量化全链路梳理

> 本文从**数据类型（dtype）与量化**这一横切面，系统梳理 SGLang 两条 EP-MoE 通信/计算方案——
> **DeepEP**（通信库 + 独立 GEMM）与 **MegaMoE**（DeepGEMM 融合大内核）——中每一步张量的精度、
> 量化粒度、scale 布局，以及"哪一步做加权求和"这一最易错的语义。
>
> 与另两篇架构文档互补：
> - `deepep_dispatch_combine_architecture.md`：DeepEP 的**通信机制/布局**（normal vs low-latency）。
> - `megamoe_architecture.md`：MegaMoE 的**融合内核机制**（warp 专业化 / ring buffer）。
>   ⚠️ 该文相对当前 `mega_moe.py` 已部分**陈旧**（行号漂移、缺 `nvfp4xnvfp4` 路径、缺 SM90 支持、
>   FIX_MEMORY 双分支已被合并），本文第 D 节给出更正。
>
> 三个引导问题（本文按此展开）：
> 1. DeepEP 里 dispatch / combine 的 hidden_states 必须是 bf16 吗？（→ 第 A 节）
> 2. MegaMoE 的数据类型与量化通常怎么组合？（→ 第 B 节）
> 3. MegaMoE 内核内部，每个阶段处理什么 dtype？（→ 第 C 节）

---

## A. DeepEP：dispatch 可量化，combine 恒 bf16（非对称）

### A0. 一句话结论

**dispatch 默认量化到 FP8**（激活压成 FP8 减少 all-to-all 通信量），**combine 恒为 bf16 透传**
（k 个专家输出的加权求和需要精度，不量化）。所以"dispatch/combine 都必须 bf16"是**错的**——
dispatch 通常不是 bf16，combine 才是。

### A1. dispatch 支持哪些 dtype

`DispatcherOutputDtype` 枚举（`layers/moe/utils.py:279-296`）：

| 枚举值 | 含义 | 内核标志 |
|---|---|---|
| `BF16` | dispatch 原样 bf16 | 全 False |
| `FP8` | dispatch FP8 E4M3 + 块 scale（**默认**） | `use_fp8=True` |
| `INT8` | dispatch int8 | `use_fp8=True`（复用打包）|
| `NVFP4` | dispatch NVFP4 | `use_nvfp4=True` |
| `MXFP4` | dispatch MXFP4 | `use_mxfp4=True` |
| `MXFP8` | dispatch fp8_e4m3 + e8m0 块 scale | `use_mxfp8=True` |

标志映射表在 `deepep.py:441-481`（`config_map`），`_DeepEPDispatcherImplNormal.__init__` 把
`config["use_fp8"]` 存到 `self.use_fp8`（`deepep.py:488`）。

### A2. dispatch dtype 的选择优先级

`get_deepep_output_dtype`（`utils.py:326-390`）的解析顺序：

```
1. --deepep-dispatcher-output-dtype 显式指定           → 用它
2. (已弃用) SGLANG_DEEPEP_BF16_DISPATCH=1              → BF16
3. quant_config 含 input_global_scale                 → NVFP4
4. 量化配置推断
5. flashinfer_cutedsl / cutlass 后端                   → BF16（它内部自己量化）
6. NPU 默认                                            → BF16
7. 兜底默认                                            → FP8   ← DeepSeek-V3/V4 走这里
```

即绝大多数模型（DeepSeek 系）**dispatch 走 FP8**，只有少数后端/硬件回落 bf16。

### A3. dispatch 的实际量化实现

`_DeepEPDispatcherImplNormal.dispatch_a`（`deepep.py:547-564`）：

```python
if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8:   # :554
    hidden_states = sglang_per_token_group_quant_fp8(hidden_states, 128, ...)  # :556
    # → 返回 (fp8_tensor, scale)，per-token-group-128 动态量化
```

- **量化粒度**：per-token × group-128（每 128 个 hidden 维一组一个 scale）。
- **NVFP4/MXFP8** 则在 `_dispatch_core`（`deepep.py:807+`）里把 `use_nvfp4`/`use_mxfp8` 透传给
  `buffer.low_latency_dispatch`（`deepep.py:824-837`），由 DeepEP 内核负责打包。

### A4. combine：为什么恒 bf16

`_combine_core`（`deepep.py:709-720`）：

```python
combined_x, _, event = buffer.combine(x, ...)   # :712  —— 无任何量化标志
```

low-latency 侧 `buffer.low_latency_combine`（`deepep.py:939`）同样**不带量化 dtype 参数**。原因：
combine 要对一个 token 的 k 个专家输出做**加权求和**（reduction），低精度累加会显著掉精度，
所以专家输出以 bf16 回传、以 bf16/fp32 归约。

### A5. DeepEP dtype 对照表

| 阶段 | dtype | 量化粒度 | 加权求和在哪 |
|---|---|---|---|
| dispatch 输入激活 | bf16 → **FP8**（默认）/NVFP4/MXFP8/bf16 | per-token-group-128（FP8） | — |
| 专家 GEMM 输出 | bf16 | — | — |
| combine 传输 | **bf16**（恒定） | — | NORMAL：SGLang permute kernel；LL：`low_latency_combine` 内核内 |

---

## B. MegaMoE：数据类型 × 量化的组合

### B0. 一句话结论

MegaMoE 是 **W4A8/W4A4 的融合大内核**：权重恒为 **FP4**（少数为 FP8 共享权重），激活按路径分
**FP8 / MXFP4 / NVFP4**；输出恒 **bf16**。默认路径是 `fp8xfp4`（**W4A8**：权重 FP4、激活 FP8）。

### B1. 三种 mma_type + SM90/SM100 双档

`_mega_moe_mma_type(experts)`（`mega_moe.py:48-51`）按 experts 上的标志三选一：

```python
if experts._mega_moe_nvfp4:                       return "nvfp4xnvfp4"   # W4A4 NVFP4
if get_exec().moe.enable_w4a4_mxfp4_megamoe:      return "mxf4xmxf4"     # W4A4 MXFP4
return "fp8xfp4"                                                          # W4A8（默认）
```

硬件档位由 `is_mega_moe_experts_ready`（`mega_moe.py:145-151`）区分——**当前代码不再是 SM100 独占**：

- **SM90（Hopper）**：`is_sm90_fp8_mega_moe_available(experts)` 为真时走 `run_sm90_mega_routed`
  （`mega_moe.py:305-316`，见 `mega_moe_sm90.py`）。
- **SM100（Blackwell，10.x）**：`_device_sm // 10 == 10` 才启用主 `fp8_fp4_mega_moe` 内核。

### B2. 权重量化：block UE8M0 group-32 + 交错 + UTCCP 转置

`build_mega_moe_experts_weights`（`mega_moe.py:434-492`）一次性把 FP8 blockwise 权重变换为
megakernel 所需布局：

1. **scale 重排**：`transform_sf_into_required_layout(w_sf_fp32, mn=..., k=..., recipe=(1,32), disable_ue8m0_cast=False)`
   （`:454-469`）——权重 scale 按 **group-32、UE8M0** 布局，并置 `format_ue8m0=True`（`:486-487`）。
2. **L1 gate/up 交错**：`_interleave_mega_moe_l1_weights`（`:412-420`），gran = **16**（mxf4）或 **8**（其余），
   匹配 DeepGEMM L1 读取 gate/up 的顺序（SwiGLU 要 gate/up 相邻）。
3. **UTCCP 转置 scale**：`_transpose_mega_moe_sf_for_utccp`（`:423-431`），把 scale 做 4×32 转置，
   适配 tensor-core 的 UTCCP scale 拷贝路径。
4. **产物**：`mega_l1_weights=(w13交错, w13_sf_utccp)`、`mega_l2_weights=(w2, w2_sf_utccp)`（`:489-490`）。
   注意 w13/w2 的 `.data` 被**原地重绑**（`:483-485`），deep-ep 路径与 mega 路径**共享同一 weight buffer**
   （现在是唯一分支，不再有旧 doc 里的 FIX_MEMORY 双分支）。

### B3. 激活量化：pre-dispatch 三条打包路径

`run_mega_routed_experts`（`mega_moe.py:257-392`）按 mma_type 分派激活量化（"pre-dispatch" =
把本 rank 激活量化并打包进对称内存输入区，真正 all-to-all 在内核里）：

| mma_type | 打包函数 | group_size | recipe | 额外传参 |
|---|---|---|---|---|
| `fp8xfp4`（默认） | **SGLang 自带** `mega_moe_pre_dispatch(quant_group_size=32)`（`:359-368`）| 32 | `(1,1,32)` | — |
| `mxf4xmxf4` | `deep_gemm.mega_moe_pre_dispatch(group_size=32, mma_type)`（`:346-357`）| 32 | `(1,1,32)` | — |
| `nvfp4xnvfp4` | `deep_gemm.mega_moe_pre_dispatch(group_size=16, mma_type, buf_x_scales=buf.x_scales)`（`:322-334`）| 16 | `(1,1,16)` | `use_x_scales=True`、`l1_alphas`、`l2_alphas`、`l2_act_scales`（`:335-341`）|

NVFP4 的特殊性：per-token 外层 scale 走 `buf.x_scales`，内核在 L1/L2 epilogue 里把它与
每专家 alpha（`experts.mega_l1_alphas`/`mega_l2_alphas`）折叠。

### B4. recipe 语义与 shape 约束

- **recipe `(1, 1, 32)`** = per-token × per-1 × group-32 的 scale 粒度（nvfp4 是 group-16）。
- **shape 约束** `check_mega_moe_shapes`（`mega_moe.py:91-102`）：`align = 16 * scale_group`
  （nvfp4 scale_group=16 → align=256；其余 32 → align=512）；`hidden_size` 与 `moe_intermediate_size`
  必须是 `align` 的整数倍，否则报错让改后端。这是 DeepGEMM 每 token 一行 scale 需 16B TMA 对齐所致。
- **activation_clamp** = `moe_runner_config.swiglu_limit`，对应 DSV4 的 SwiGLU clamp。
- **routed_scaling_factor**：若未在 topk 融合（`should_fuse_routed_scaling_factor_in_topk`），
  内核后 `y.mul_(routed_scaling_factor)`（`:390-391`）。

### B5. MegaMoE 组合对照表

| 路径 | 激活 dtype | 权重 dtype | 激活 scale | 权重 scale | 主循环 | 触发 |
|---|---|---|---|---|---|---|
| `fp8xfp4`（默认，W4A8）| FP8 E4M3 | FP4 E2M1 | UE8M0 group-32 | UE8M0 group-32 | `mxf8f6f4` (K=32) | 默认 |
| `mxf4xmxf4`（W4A4）| MXFP4 | FP4 E2M1 | group-32 | UE8M0 group-32 | `mxf4` (K=64 dense) | `enable_w4a4_mxfp4_megamoe` |
| `nvfp4xnvfp4`（W4A4）| NVFP4 | FP4 E2M1 | FP8 E4M3 group-16 + fp32 alpha | group-16 | nvfp4 | `experts._mega_moe_nvfp4` |
| bf16（SM90/回退）| bf16 | bf16 | — | — | `bf16xbf16` | 无 SF 内核 |

**输出恒 bf16**（`y = torch.empty(..., dtype=torch.bfloat16)`，`mega_moe.py:372-376`）。

---

## C. MegaMoE 内核内部：逐阶段 dtype（默认 `fp8xfp4` 路径）

内核源码 `DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`。

### C0. 三个关键 dtype 事实（先记结论）

1. **累加器全程 FP32**：所有 MMA 结果在 TMEM 里以 FP32 累加，只有跨阶段落 buffer 时才降精度。
2. **L1→L2 原地重量化**：L1 输出经 SwiGLU 后**当场 cast 回 FP8 E4M3 + 新 UE8M0 scale**，宽度砍半
   （BLOCK_N → BLOCK_N/2），直接当 L2 的激活输入，不落 HBM。
3. **topk 加权在 L1 epilogue 做**（不是 combine 时做）——这与 DeepEP 相反。因此 MegaMoE 的
   combine 只是**纯 FP32 求和**，而非 DeepEP 那样的加权归约。

### C1. warp 角色的 dtype（源码 `:143-146`）

```cpp
using a_dtype_t       = cutlass::float_e4m3_t;                       // 激活 FP8
using b_dtype_t       = cutlass::detail::float_e2m1_unpacksmem_t;    // 权重 FP4
using shared_b_dtype_t = cutlass::float_e4m3_t;                     // 共享专家权重 FP8
// UMMA: UMMA_M=256, UMMA_BLOCK_K=128, UMMA_K=32   (:150-158)
```

### C2. 七阶段流水的输入/输出 dtype

| # | 阶段 | 输入 dtype | 输出 dtype | 源码 |
|---|---|---|---|---|
| 1 | pre-dispatch（本 rank 打包）| bf16 激活 | FP8 E4M3 + UE8M0 SF 进 `buf.x/x_sf` | SGLang jit / DeepGEMM |
| 2 | DISPATCH warp（NVLink all-to-all）| FP8+SF（远端）| FP8+SF（本地 L1 ring）| kernel |
| 3 | L1 GEMM（gate/up）| FP8 激活 × FP4 权重 | **FP32**（TMEM 累加）| `:150-189` |
| 4 | **L1 epilogue** | FP32 | **FP8 E4M3 + UE8M0 SF**（含 SwiGLU × topk_weight）| `:987-1129` |
| 5 | L2 GEMM（down）| FP8 激活 × FP4 权重 | **FP32**（TMEM 累加）| — |
| 6 | **L2 epilogue** | FP32 | **bf16**，经 `sym_buffer.map` 直写**远端** combine buffer | `:1205-1301` |
| 7 | COMBINE warp | bf16（本 token 的 topk+shared 贡献）| **bf16**（FP32 累加后 cast）→ TMA 存 y | `:1404-1430` |

关键行：
- **SwiGLU + topk 加权**（`:1071`）：`activation_values = fmul2(fmul2(gate, up), weights)`——
  `silu(gate)*up` 在 FP32 里算，**当场乘上 topk 权重**。
- **L1→L2 重量化**（`:1100-1129`）：cast 成 `__nv_fp8x4_e4m3`，SF 以 UE8M0 写 `l2_sf_buffer`。
- **L2 远端直写**（`:1205-1301`）：`math::cast_into_bf16_and_pack` → 经对称堆指针写到目标 rank 的
  combine buffer，**这一步就是 combine 的 scatter**，没有独立 combine kernel。
- **COMBINE**（`:1404-1430`）：从 `combine_token_buffer` 读 bf16，用 `ptx::accumulate` 在 FP32
  寄存器累加（纯和，权重已在阶段 4 乘过），cast 回 bf16，TMA 存到输出 `y`。

### C3. 其它路径的 dtype 差异

- **bf16 路径**（`sm100_bf16_mega_moe.cuh`）：无 scale factor（10 个 TMA desc vs FP8 路径 18 个），
  L1 epilogue 不做 amax/FP8-cast，L2 中间态直接 bf16，`UMMA_BLOCK_K=64`。
- **mxf4 路径**：主循环 `mxf4` K=64 dense；激活 MXFP4，L1→L2 中间态按 MXFP4 打包（比 FP8 再省一半）。
- **nvfp4 路径**：group-16 + FP8 E4M3 scale + per-expert fp32 alpha，alpha 在 L1/L2 epilogue 折叠
  （`use_x_scales=True`）。

---

## D. 与现有文档的关系 & `megamoe_architecture.md` 陈旧点更正

本文是 dtype/量化的**横切补充**，机制细节仍看两篇架构文档。但 `megamoe_architecture.md` 相对
当前 `mega_moe.py`（493 行）已陈旧，使用时请以下列更正为准：

| 陈旧点（旧 doc）| 当前代码（`mega_moe.py`）|
|---|---|
| 仅 SM100 独占 | **SM90 + SM100 双档**（`:145-151`，SM90 走 `run_sm90_mega_routed`）|
| 只列 BF16 / FP8×FP4 / FP4-acts+MXF4 三路 | 实为 **`fp8xfp4` / `mxf4xmxf4` / `nvfp4xnvfp4` + bf16**（`:48-51`）|
| FIX_MEMORY 省显存/默认 **两分支**权重构建 | 已合并为**单分支**：恒 `transform_sf_into_required_layout` + 手工交错 + UTCCP 转置（`:434-492`）|
| vendored DeepGEMM 2.6.1 无 `mega_moe_pre_dispatch` | 当前 mxf4/nvfp4 路径**直接调** `deep_gemm.mega_moe_pre_dispatch`（`:322,:346`）；仅 fp8 走 SGLang 自带 jit |
| 行号 `:97-111 / :114-148 / :151-267 / :301-364` | 现为 `should_use_mega_moe:154` / `forward_mega_moe:171` / `_run_mega_routed:208` / `run_mega_routed_experts:257` / `build_...:434` |

---

## E. 行号速查

### DeepEP（`layers/moe/token_dispatcher/deepep.py` + `utils.py`）
| 主题 | 位置 |
|---|---|
| `DispatcherOutputDtype` 枚举 | utils.py:279-296 |
| dtype 选择优先级 `get_deepep_output_dtype` | utils.py:326-390 |
| 标志映射 `config_map` / `self.use_fp8` | deepep.py:441-481 / :488 |
| dispatch FP8 量化 | deepep.py:554-556 |
| LL dispatch（透传 use_fp8/nvfp4/mxfp8）| deepep.py:807-837 |
| combine（无量化标志，bf16）| deepep.py:709-712 |
| LL combine | deepep.py:939 |

### MegaMoE（`layers/moe/mega_moe.py`）
| 主题 | 位置 |
|---|---|
| `_mega_moe_mma_type` 三选一 | :48-51 |
| SM90/SM100 就绪判定 | :145-151 |
| `should_use_mega_moe` | :154-168 |
| `forward_mega_moe`（+SBO）| :171-205 |
| `_run_mega_routed`（gate/topk）| :208-254 |
| `run_mega_routed_experts`（三路 pre_dispatch + megakernel）| :257-392 |
| shape 约束 `check_mega_moe_shapes` | :91-102 |
| 权重构建 `build_mega_moe_experts_weights` | :434-492 |
| 交错 / UTCCP 转置 | :395-431 |

### DeepGEMM 内核（`impls/sm100_fp8_fp4_mega_moe.cuh`）
| 主题 | 位置 |
|---|---|
| 数据类型定义 | :143-146 |
| UMMA 参数 | :150-158 |
| L1 输出 FP8 / L2 输出 bf16 | :179-189 |
| L1 epilogue（SwiGLU × topk_weight）| :987-1129（乘权重 :1071）|
| L2 epilogue（bf16 直写远端）| :1205-1301 |
| COMBINE（FP32 纯和 → bf16）| :1404-1430 |
