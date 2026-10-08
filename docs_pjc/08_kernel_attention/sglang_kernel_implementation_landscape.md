# SGLang Kernel 实现方式全景梳理

> 代码基线：`7b6b6ee25eed5e82a1270748eaa67d077cb23025`
>
> 本文回答一个容易被路径和术语混淆的问题：**SGLang 中一个高性能 kernel 到底可以怎样实现、怎样编译、怎样被 Python 调用，以及多种实现如何共存和回退？**
>
> 适合读者：需要新增 kernel、迁移旧 kernel、定位某个算子实际运行后端，或分析 CUDA / ROCm / NPU / CPU 差异的开发者。
>
> 重要路径说明：当前基线已经完成 kernel namespace 重组。旧的 `python/sglang/jit_kernel/` 已移除，公共入口和 JIT 基础设施分别位于 `python/sglang/kernels/ops/` 与 `python/sglang/kernels/jit/`；AOT 工程位于 `python/sglang/kernels/aot/`，不是仓库顶层的 `sgl-kernel/`。

---

## 目录

1. [先给结论：Kernel 体系不是一条技术路线](#1-先给结论kernel-体系不是一条技术路线)
2. [统一的公共入口和分发层](#2-统一的公共入口和分发层)
3. [几种维度必须分开](#3-几种维度必须分开)
4. [方式一：PyTorch 原生实现](#4-方式一pytorch-原生实现)
5. [方式二：PyTorch custom op 封装层](#5-方式二pytorch-custom-op-封装层)
6. [方式三：Triton kernel](#6-方式三triton-kernel)
7. [方式四：SGLang TVM-FFI JIT CUDA/HIP kernel](#7-方式四sglang-tvm-ffi-jit-cudahip-kernel)
8. [方式五：AOT `sgl-kernel`](#8-方式五aot-sgl-kernel)
9. [方式六：CuTe DSL](#9-方式六cute-dsl)
10. [方式七：TileLang](#10-方式七tilelang)
11. [方式八：外部 kernel 库封装](#11-方式八外部-kernel-库封装)
12. [方式九：平台专用 native / vendor kernel](#12-方式九平台专用-native--vendor-kernel)
13. [编译、加载和缓存时机对比](#13-编译加载和缓存时机对比)
14. [一个 kernel 从源码到请求执行的完整链路](#14-一个-kernel-从源码到请求执行的完整链路)
15. [如何选择实现方式](#15-如何选择实现方式)
16. [新增 kernel 的推荐流程](#16-新增-kernel-的推荐流程)
17. [测试、调试与性能验证](#17-测试调试与性能验证)
18. [常见误区](#18-常见误区)
19. [关键文件速查](#19-关键文件速查)

---

## 1. 先给结论：Kernel 体系不是一条技术路线

SGLang 的 kernel 可以来自以下几类来源：

| 类别 | 典型实现 | 主要编译时机 | 典型优点 | 典型代价 |
|---|---|---|---|---|
| PyTorch 原生 | `torch` 算子组合 | PyTorch 运行时 | 正确性强、跨平台、易调试 | 融合和访存控制有限 |
| PyTorch custom op | `torch.library` / `torch.ops` | 注册即时；底层实现按后端决定 | 可作为 `torch.compile` 的不透明边界 | 需要维护 schema、fake 实现和变异语义 |
| Triton | `@triton.jit` | 通常首次遇到配置时 JIT | Python 表达力强，适合快速迭代 | 编译缓存、shape 变体和低层控制需管理 |
| SGLang TVM-FFI JIT | `.cuh` + C++ wrapper + `load_jit` | 首次调用时编译 | CUDA/HIP 模板控制强，按需特化 | 首调编译延迟，需要 nvcc/hipcc |
| AOT `sgl-kernel` | C++/CUDA、CUTLASS、部分 vendored 源 | 构建 wheel 时 | 首次请求无 JIT，适合稳定热点 | 编译矩阵大，wheel 构建和架构覆盖复杂 |
| CuTe DSL | `cutlass.cute` / FlashInfer CuTe DSL | DSL JIT 或外部包构建时 | 可表达 TMA、MMA、TMEM、cluster 等新硬件特性 | 依赖 CUDA、CUTLASS 和特定架构，调试门槛高 |
| TileLang | `@tilelang.jit` | 运行时 JIT | 以 tile 为中心表达复杂 kernel | 版本、后端和 shape 约束敏感 |
| 外部 kernel 库 | FlashInfer、FlashAttention、DeepGEMM、AITER 等 | 预编译、外部 JIT 或运行时加载 | 复用成熟高性能实现 | 版本、ABI、架构和输入约束由外部库决定 |
| 平台专用实现 | `torch_npu`、HIP、CPU AMX 等 | 平台包加载或 AOT 构建 | 覆盖非 CUDA 平台和专用指令 | 平台分支多，维护和一致性成本高 |

这些类别不是互斥的。一个对外暴露的算子通常同时拥有：

```text
forward_native       正确性参考实现
forward_torch_compile torch.compile(forward_native)，默认免费继承
forward_triton       Triton 实现
forward_jit          SGLang TVM-FFI JIT 实现
forward_aot          AOT sgl-kernel 实现
forward_flashinfer   FlashInfer 封装
forward_deepgemm     DeepGEMM 封装
forward_cute_dsl     CuTe DSL 实现
forward_flydsl       FlyDSL 实现
forward_kda          Kernel Design Agent 生成的 kernel
forward_aiter        AITER 实现
forward_torch_npu    torch_npu 实现
```

因此，“这个算子是不是 JIT kernel”这个问题通常不够精确。应该继续问：

1. JIT 的是 Triton、TVM-FFI、CuTe DSL，还是外部库？
2. kernel 是由谁编译和缓存的？
3. Python 调用的是普通函数、`torch.ops`，还是 `BaseFusedOp.forward()`？
4. 该路径是默认路径、显式指定路径，还是仅作为 fallback / inventory 存在？

---

## 2. 统一的公共入口和分发层

### 2.1 公共 import namespace

当前 runtime 代码应从 `sglang.kernels.ops.*` 导入 callable kernel，例如：

```python
from sglang.kernels.ops.layernorm import rmsnorm
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.kvcache import reshape_and_cache_flash
```

公共目录、算子分组和旧路径迁移说明见 [python/sglang/kernels/README.md](../../../python/sglang/kernels/README.md)。

`python/sglang/kernels/ops/` 是调用入口，不等于具体实现目录。具体实现可以位于：

- `python/sglang/kernels/jit/`：SGLang 自有 TVM-FFI JIT 基础设施；
- `python/sglang/kernels/aot/`：AOT `sgl-kernel` 工程和 Python wrapper；
- `python/sglang/kernels/ops/<group>/`：Triton、CuTe DSL、TileLang 和轻量 wrapper；
- `python/sglang/kernels/kda_kernels/`：`KDA` backend（Kernel Design Agent 生成）的实现家，含 JIT CUDA 源（`csrc/gemm`、`csrc/diffusion`）、Triton kernel 以及 SM120/SM121 FP8/NVFP4 GEMM 等变体；
- `python/sglang/kernels/cake_kernels/`：对 FlashInfer 分发的 Cake 生成 kernel 的惰性适配层（导入不会立刻拉起 FlashInfer/CUDA）；
- `python/sglang/srt/layers/`：模型层或 attention backend 中的专用实现；
- 外部 Python/C++ wheel：FlashInfer、DeepGEMM、AITER、FlashAttention 等。

此外 `python/sglang/kernels/kernel_api_logging.py` 提供 `debug_kernel_api` 装饰器（被 `BaseFusedOp.forward` 使用），用于可选的 kernel 调用 API 日志；`python/sglang/kernels/spec.py` 是不依赖 torch 的元数据层（`KernelBackend`、`DeviceType`、`PlatformInfo`、`CapabilityRequirement`、`FormatSignature`、`KernelSpec`）。

### 2.2 `KernelRegistry`：登记元数据，不触发实现加载

[registry.py](../../../python/sglang/kernels/registry.py) 中的 `KernelRegistry`（`registry.py:17`）以 `"<group>.<name>"` 为 operator id，登记一个 `KernelSpec`（定义在 [spec.py](../../../python/sglang/kernels/spec.py)，`spec.py:205`）：

- `op`：算子 id，`"<group>.<name>"`；
- `backend`：`KernelBackend`，表示实现来源；
- `target`：`module:attribute` 形式的延迟加载路径；
- `capabilities`：`CapabilityRequirement` 集合；
- `format_signature`：format signature；
- `description`。

`KernelBackend` 枚举（`spec.py:29-52`）当前取值为：`TORCH`、`TORCH_COMPILE`、`TRITON`、`JIT`、`AOT`、`CUTE_DSL`、`FLYDSL`、`KDA`、`FLASHINFER`、`DEEPGEMM`、`AITER`、`TORCH_NPU`。另有独立的 `DeviceType` 枚举（CUDA/HIP/NPU/CPU）用于描述平台维度——backend 与 device 是两条正交的维度。

模块级函数 `register_kernel(spec)`（`registry.py:76`）把 `KernelSpec` 登记进全局 registry；`BaseFusedOp` 实例则通过 `register_fused_op()`（在 [fused_op.py](../../../python/sglang/kernels/fused_op.py)，`fused_op.py:665`）把自身所有 backend 的元数据批量登记。

登记本身不导入具体后端，也不触发 JIT 编译。这个设计让程序可以先建立“有哪些实现”的清单，再在真正调用时加载实现。

### 2.3 `select_kernel()` / `get_kernel()`：显式解析 callable

[selector.py](../../../python/sglang/kernels/selector.py) 提供两层 API：

```text
select_kernel(op, backend) -> KernelSpec
get_kernel(op, backend)    -> 已 resolve 并缓存的 callable
```

当前 registry selector 的职责是**确定性解析**，不是一个隐式性能排行榜：

- 只有一个注册 backend 时直接解析；
- 多个 backend 时按当前平台过滤；
- 仍有多个可用 backend 时要求调用方显式传 `backend=`；
- `get_kernel()` 对最终 callable 做缓存，避免重复 import。

这与 `BaseFusedOp` 的优先级分发不同。前者是 registry 级的显式/确定性解析，后者是一个逻辑算子实例内部的运行时 dispatch。

### 2.4 `BaseFusedOp`：一个逻辑算子，多种实现

[fused_op.py](../../../python/sglang/kernels/fused_op.py) 中的 `BaseFusedOp` 是统一的多 backend contract。每个具体算子：

1. 必须提供 `forward_native`；
2. 可以覆盖若干 `forward_<backend>`；
3. 可以提供平台方法，如 `forward_cuda`、`forward_hip`、`forward_npu`、`forward_cpu`；
4. 通过 `capabilities` 声明硬件能力；
5. 通过 `priority` 规定优化 backend 的优先顺序；
6. 让所有实现适配到同一个 Python signature。

默认 dispatch 顺序为：

```text
显式 backend
    ↓
全局强制 backend（SGLANG_FORCE_FUSED_OP_BACKEND）
    ↓
out-of-tree platform override
    ↓
按 priority 选择满足 capability 的优化 backend
    ↓
平台专用 forward
    ↓
forward_native
```

关键实现位置是 [fused_op.py:334-662](../../../python/sglang/kernels/fused_op.py#L334)（`BaseFusedOp` 类主体，全文件 686 行）。导入和实例化不应触发具体 backend 的导入或编译；backend 通常在第一次真正进入相应方法时加载。

### 2.5 `torch.compile` 的边界

`BaseFusedOp` 在外层模型进入 `torch.compile` 时，默认切换到 compile-safe 的 `forward_native`。原因不是 native 一定更快，而是避免 Dynamo 追踪以下不可追踪行为：

- 文件系统路径解析；
- C++/CUDA JIT 编译；
- 动态加载 `.so`；
- 外部库内部的 shape tuning 或文件 I/O。

少数算子会重写 `_torch_compile_forward()`（如 TopK 和 Fused MoE 保留 bs>1 行为），保留适合编译场景的优化路径。这个协议位于 [fused_op.py:581-622](../../../python/sglang/kernels/fused_op.py#L581)（`_torch_compile_forward` / `enter_torch_compile` / `leave_torch_compile`，均幂等）。

---

## 3. 几种维度必须分开

### 3.1 编程模型

这是“kernel 用什么语言表达”：

- PyTorch operator graph；
- Triton；
- CUDA/HIP C++；
- CuTe DSL；
- TileLang；
- 外部库 API。

### 3.2 编译形态

这是“什么时候把实现变成机器代码”：

- PyTorch 自带预编译 operator；
- AOT：构建 wheel 时完成；
- runtime JIT：第一次调用或第一次遇到某配置时完成；
- 外部库预编译 + 外部库 JIT 混合。

### 3.3 Python 集成边界

这是“Python 如何调用它”：

- 直接调用 `torch` 函数组合；
- 调用 `torch.ops.namespace.op`；
- 调用 `tvm_ffi.Module` 导出的函数；
- 调用外部 Python 函数；
- 通过 `BaseFusedOp.forward()` 进行统一 dispatch。

### 3.4 平台与后端

CUDA、HIP、NPU、CPU、XPU、MUSA 是 device/platform 维度，不应直接等同于 kernel backend。比如：

- CUDA 可以运行 AOT、JIT、Triton、CuTe DSL、FlashInfer；
- HIP 可以运行 HIP AOT、Triton HIP、AITER 和部分 FlashAttention fallback；
- `AOT` 是实现来源，CUDA/HIP 是它的 capability；
- `forward_cuda` 是平台路径，不等于 `KernelBackend.CUDA`。

`BaseFusedOp` 明确把 backend provenance 和 platform/device 拆成两个维度，见 [fused_op.py:1-64](../../../python/sglang/kernels/fused_op.py#L1)（模块 docstring）。

---

## 4. 方式一：PyTorch 原生实现

### 4.1 是什么

PyTorch 原生实现不是“没有 kernel”，而是把计算交给 PyTorch 已有 operator，例如矩阵乘、逐元素运算、softmax、reshape、index、scatter 等。PyTorch 可能进一步调用 cuBLAS、CUDA kernel、oneDNN 或其他平台实现。

在 SGLang 的统一 contract 中，它通常表现为：

```python
def forward_native(self, x, weight):
    return torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * x * weight
```

`forward_native` 是其他手写 kernel 的 correctness reference，也是所有不满足硬件、dtype 或 shape 条件时的最终 fallback。

### 4.2 优点

- 代码最短，便于验证数学语义；
- 自动覆盖多个 PyTorch 支持的平台；
- 易于被 `torch.compile` 捕获；
- 适合作为新算子的第一版和单元测试参考；
- 可以覆盖优化 kernel 没有实现的边界 shape。

### 4.3 限制

- 多个小 operator 可能产生额外 kernel launch；
- 难以精确控制线程布局、共享内存、寄存器和访存向量化；
- paged KV、稀疏索引、融合量化等场景通常无法仅靠 operator 组合达到目标性能；
- native 结果正确不代表和手写 kernel 的累加顺序、精度或 inplace 语义完全相同。

### 4.4 在 SGLang 中的角色

`BaseFusedOp.forward_native` 是强制存在的参考路径（`@abstractmethod`），[fused_op.py:418-420](../../../python/sglang/kernels/fused_op.py#L418)。开发新 kernel 时推荐先固定：

1. 输入 shape、dtype、device 和 stride contract；
2. native 参考结果；
3. 随机和边界测试；
4. 再增加 Triton/JIT/AOT 等实现。

---

## 5. 方式二：PyTorch custom op 封装层

### 5.1 custom op 不是一种 device kernel

`torch.library` custom op 主要解决的是**框架集成问题**，而不是规定底层 kernel 必须用哪种语言。底层可以是：

- C++/CUDA AOT；
- TVM-FFI JIT；
- Triton 函数；
- FlashInfer 外部函数；
- NPU `.so` 中的 operator；
- 甚至一个 native Python 实现。

因此应把它看成“PyTorch graph 和实际实现之间的 ABI/语义边界”。

### 5.2 SGLang 的注册入口

[custom_op.py](../../../python/sglang/srt/utils/custom_op.py) 提供：

```python
@register_custom_op(
    mutates_args=["x"],
    out_shape=0,
)
def op(x, y):
    ...
```

以及：

```python
register_custom_op_from_extern(
    external_fn,
    out_shape="hidden_states",
    out_dtype=torch.bfloat16,
)
```

底层通过 [common.py:3008-3090](../../../python/sglang/srt/utils/common.py#L3008) 的 `direct_register_custom_op()` 建立 `torch.library.Library("sglang", "FRAGMENT")`（`common.py:3005`）、schema 和 fake implementation。

### 5.3 它解决的四个问题

#### 1. 变异语义

`mutates_args` 说明哪些输入被 inplace 修改，避免 PyTorch 对算子别名和 autograd/graph 语义做错误假设。

#### 2. fake implementation

`torch.compile` 在 tracing 阶段通常不应真正运行 CUDA kernel。fake implementation 只根据输入的 shape、dtype、device 推断输出元信息。

#### 3. opaque boundary

动态 JIT、外部库加载和文件 I/O 放在 custom op 内部后，Dynamo 看到的是一个不透明节点，不会进入底层编译流程。

#### 4. 外部函数适配

`register_custom_op_from_extern()` 专门用于 FlashInfer 等外部函数。它可以处理输出 shape、输出 dtype，以及不希望进入 custom-op schema 的动态计算参数。

### 5.4 适用场景

- JIT/AOT kernel 需要参与 `torch.compile` 模型；
- 外部库函数内部会动态编译或加载模块；
- 输出 shape 可由某个输入推断；
- 需要明确 inplace/mutation 语义；
- 需要把一组不稳定的底层实现统一成稳定的 Python 调用接口。

### 5.5 需要注意

custom op 注册成功不代表底层实现适用于所有输入。仍需要在底层或 wrapper 层检查：

- device；
- dtype；
- contiguous/stride；
- shape 范围；
- 是否支持 CUDA Graph；
- 是否支持不同的 batch size 和 sequence length。

---

## 6. 方式三：Triton kernel

### 6.1 基本模型

Triton 使用 Python 描述以 tile 为中心的 GPU 程序，编译器负责生成 GPU 代码。典型形式是：

```python
@triton.jit
def kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x, mask=mask)
```

SGLang 中的 Triton 实现分布在 `python/sglang/kernels/ops/`、`python/sglang/srt/layers/` 和平台目录中。代表入口包括：

- [triton_backend.py](../../../python/sglang/srt/layers/attention/triton_backend.py)：paged attention、extend/prefill 等 attention 路径；
- [fused_moe_triton/](../../../python/sglang/srt/layers/moe/fused_moe_triton/)：MoE dispatch、量化和 fused MoE；
- [ops/attention/](../../../python/sglang/kernels/ops/attention/)：统一 namespace 下的 Triton attention/DSA 等实现；
- [xpu/kernels/fla/chunk_fwd.py](../../../python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_fwd.py)：XPU 上的 Triton-compatible FLA kernel。

### 6.2 编译与缓存

`@triton.jit` 函数通常在第一次调用某个 device、dtype、shape/meta-parameter 组合时触发编译，后续从 Triton cache 复用。实际缓存粒度和产物位置由 Triton 版本、driver 和运行环境共同决定。

需要区分两件事：

- AOT `sgl-kernel` 的 CMake 可能把 Triton/相关源码作为构建依赖或源文件参与编译；
- Python 中的 `@triton.jit` 是 Triton runtime JIT。

前者不等于后者。

### 6.3 优点

- Python 开发体验好，便于快速验证 tile、mask 和 layout；
- 适合 elementwise、reduction、量化、索引、部分 attention 和 MoE；
- 可用 `tl.constexpr` 对 block size、head dim、dtype 等做编译期特化；
- 比直接维护完整 CUDA C++ wrapper 的样板代码少。

### 6.4 限制

- 对极细粒度硬件指令、复杂异步流水线和新一代 Tensor Core 特性控制有限；
- 多个 `constexpr` 参数会造成编译配置数量快速增长；
- 需要自行处理 autotune、warmup、cache 和首次调用延迟；
- CUDA、HIP、XPU 的支持能力和编译器行为不完全相同；
- 某些模型的动态 shape、稀疏布局或复杂跨 block 协作不适合 Triton。

### 6.5 在统一 dispatch 中的接入

如果同一个逻辑算子同时拥有 native 和 Triton 实现，通常在 `BaseFusedOp` 子类中提供：

```python
def forward_native(...): ...
def forward_triton(...): ...
```

并在 `capabilities` 中声明 device capability。对 shape 有额外限制时，重写 `backend_eligible()`，让不支持的输入自动落到下一个 backend，而不是在 kernel 内部才失败。

### 6.6 何时优先选 Triton

优先考虑 Triton 的场景：

- 算子逻辑仍在快速变化；
- 主要是规则 tile、逐元素、归约或中等复杂度融合；
- 需要 CUDA 和部分其他 GPU 平台共享较多代码；
- 性能目标明显高于 native，但还没有必要维护完整 C++ 基础设施；
- 需要快速做 shape 扫描和原型验证。

---

## 7. 方式四：SGLang TVM-FFI JIT CUDA/HIP kernel

### 7.1 当前目录和调用链

当前基础设施位于：

- [jit/](../../../python/sglang/kernels/jit/)：构建、缓存、架构和依赖工具；
- [ops/](../../../python/sglang/kernels/ops/)：具体公共算子 wrapper；
- `jit/csrc/`：CUDA/HIP 源码；
- `jit/include/`：`sgl_kernel` 复用头文件。

典型调用链为：

```text
sglang.kernels.ops.<group>.<op>
    ↓
_jit_<op>_module(specialization_args)
    ↓
load_jit(...)
    ↓
BuildSpec + source/include/flags 解析
    ↓
内容寻址 cache 查询
    ↓
命中：tvm_ffi.load_module(.so)
未命中：生成 Ninja → nvcc/hipcc → 原子发布 → load_module
    ↓
TVM-FFI 导出函数
```

核心入口见 [loader.py:48-190](../../../python/sglang/kernels/jit/utils/compile/loader.py#L48)（`load_jit()`，全文件 231 行）。

### 7.2 `load_jit()` 做什么

`load_jit()` 接收：

- 模块特化参数 `*args`；
- C++/CUDA 源文件；
- 导出 wrapper；
- host/device 编译 flags；
- include path 和注册依赖；
- `header_only` 模式。

随后它会：

1. 对 HIP 过滤不适用的 CUDA fast-math flag；
2. 合并默认 include 和外部依赖 include；
3. 构造 `BuildSpec`；
4. 生成 Ninja build file；
5. 计算包含源码、include 闭包、wrapper、flags、target 在内的 build key；
6. 先查 cache；
7. 未命中时取得进程间 build lock；
8. 在随机 staging 目录编译；
9. 先加载验证 `.so`，再通过原子 rename 发布 cache leaf。

这套流程还处理 Tensor Parallel 多进程同时启动时的重复编译问题：通过 `fcntl.flock(LOCK_EX)` 进程间 build lock 串行化编译、staging 目录编译后原子 rename 发布，见 [loader.py:147-225](../../../python/sglang/kernels/jit/utils/compile/loader.py#L147)。

### 7.3 两种导出方式

#### `header_only=True`

多数轻量 kernel 使用纯 `.cuh` 模板，Python 传入 `cuda_wrappers=[(export_name, kernel_name)]`，框架生成一个 wrapper 源文件导出模板实例。

```text
Python 选择 hidden_size/dtype/布尔特化
    ↓
模板名拼接
    ↓
生成 TVM_FFI_DLL_EXPORT_TYPED_FUNC
    ↓
编译为模块
```

#### `header_only=False`

需要有状态对象或 C++ 侧自主注册时使用完整 `.cpp` 源。C++ 侧完成 `register_once()`，Python 通过 TVM-FFI object 协议构造对象。该模式的代表是 n-gram corpus 一类有状态组件。

### 7.4 C++ 复用头文件库

`jit/include/sgl_kernel/` 的价值不只是减少代码量，还统一了不同 kernel 的约束和行为。常见设施包括：

| 设施 | 作用 |
|---|---|
| `utils.cuh` / `utils.h` | 平台宏、错误检查、launch、runtime helper |
| `tensor.h` | `TensorMatcher`，检查 shape/dtype/device/stride |
| `type.cuh` | dtype trait、packed type、转换和数学函数 |
| `vec.cuh` | 对齐向量化 load/store |
| `tile.cuh` | thread/warp/CTA 协作式访存 |
| `warp.cuh` / `cta.cuh` | warp 和 block 归约 |
| `atomic.cuh` | CUDA/HIP 原子操作抽象 |
| `runtime.cuh` | SM、occupancy、动态 shared memory 查询 |
| `ffi.h` | C++ 侧分配和包装 TVM-FFI Tensor |

### 7.5 架构特化

JIT wrapper 会根据当前 GPU capability 注入 target flags，并将架构编码到 build key。C++ 侧可以使用架构宏选择不同实现，例如：

- Hopper 及以上启用 PDL；
- Blackwell 使用更宽的向量化访存；
- ROCm 使用 HIP API 和平台特有 FP8 表示；
- 某些 kernel 通过 `override_jit_cuda_arch()` 强制构建 `sm_100a` 等变体。

### 7.6 适用场景

优先选择 SGLang JIT CUDA/HIP 的场景：

- 需要 C++/CUDA 模板级控制；
- 特化参数多，AOT 全量编译组合过大；
- 算子尚在演进，暂时不想进入 AOT wheel；
- 需要复用 `TensorMatcher`、`LaunchKernel`、dtype trait 等基础设施；
- 首次调用延迟可以通过预热或共享 cache 消化。

### 7.7 主要代价

- 生产机器需要匹配的 CUDA toolkit 或 HIP 编译环境；
- 首次调用可能触发显著编译延迟；
- 多进程虽然有 build lock，但不同机器仍各自需要 cache；
- JIT 源码、依赖闭包和 flags 变化会导致 cache miss；
- 外层 `torch.compile` 必须通过 custom op 或 compile mode 隔离 JIT 逻辑。

---

## 8. 方式五：AOT `sgl-kernel`

### 8.1 当前工程位置

当前 AOT 工程位于 [python/sglang/kernels/aot/](../../../python/sglang/kernels/aot/)，主要包括：

- `CMakeLists.txt`：CUDA/C++ 编译组织；
- `pyproject.toml`：scikit-build-core wheel 定义；
- `python/sgl_kernel/`：Python loader 和 wrapper；
- `csrc/`：CUDA/C++ 实现；
- `csrc/cpu/`：CPU 变体。

旧资料中出现的仓库顶层 `sgl-kernel/` 是历史布局或其他 checkout 的路径，阅读当前分支应以 `python/sglang/kernels/aot/` 为准。

### 8.2 AOT 链路

```text
AOT C++/CUDA source
    ↓  构建 wheel 时
scikit-build-core
    ↓
CMake + nvcc / host compiler
    ↓
按架构生成 shared library
    ↓
wheel 内的 sgl_kernel/sm90、sm100 等目录
    ↓  运行时
sgl_kernel.load_utils 依据 compute capability 选择 .so
    ↓
Python wrapper / torch.ops.sgl_kernel.<op>
```

CMake 的 CUDA 源码枚举从 `set(SOURCES ...)`（约 `aot/CMakeLists.txt:256`）开始，整文件 553 行还包含 ROCm/MUSA 和双架构（sm90/sm100 等）构建处理，见 [aot/CMakeLists.txt](../../../python/sglang/kernels/aot/CMakeLists.txt)。运行时架构 loader 位于 [aot/python/sgl_kernel/__init__.py](../../../python/sglang/kernels/aot/python/sgl_kernel/__init__.py)（`_load_architecture_specific_ops()`）。

### 8.3 AOT 适合什么

- 线上热点路径已经稳定；
- 首请求延迟敏感；
- 目标 shape/dtype/架构组合可枚举；
- kernel 需要复杂 C++/CUDA 代码或外部头库；
- 希望生产环境不依赖 nvcc/hipcc；
- 希望把 ABI、wheel 和版本作为发布物管理。

### 8.4 AOT 的现实成本

AOT 不是“编译一次就结束”，而是把成本前移到发布流程：

- 不同 SM 需要不同 gencode；
- CUDA 版本、CUTLASS、FlashInfer、Torch ABI 需要配套；
- 编译矩阵会扩大构建时间和产物体积；
- 新 shape 或新 GPU 可能需要新增构建变体；
- 需要处理 wheel 中 `.so` 的加载和 fallback。

### 8.5 AOT 与 JIT 的边界

| 问题 | AOT | TVM-FFI JIT |
|---|---|---|
| 代码何时编译 | wheel 构建期 | 首次实际调用 |
| 首请求 | 无本地编译等待 | 可能等待 nvcc/hipcc |
| 部署依赖 | 主要依赖运行时动态库 | 还依赖编译工具链 |
| 特化范围 | 预先决定 | 调用到再决定 |
| 版本发布 | wheel/ABI/架构可审计 | cache 和源码/flags 组合影响 |
| 适合 | 稳定生产热点 | 实验、快速迭代、组合爆炸 |

### 8.6 AOT wrapper 的 PyTorch 边界

AOT wrapper 通常最终调用：

```python
torch.ops.sgl_kernel.some_op.default(...)
```

这让 AOT C++ 实现能进入 PyTorch custom op 体系，同时由 Python 层负责参数整理、能力检查和 fallback。典型 wrapper 可参考：

- [aot/python/sgl_kernel/attention.py](../../../python/sglang/kernels/aot/python/sgl_kernel/attention.py)；
- [aot/python/sgl_kernel/gemm.py](../../../python/sglang/kernels/aot/python/sgl_kernel/gemm.py)；
- [aot/python/sgl_kernel/flash_attn.py](../../../python/sglang/kernels/aot/python/sgl_kernel/flash_attn.py)。

---

## 9. 方式六：CuTe DSL

### 9.1 是什么

CuTe DSL 是 CUTLASS 提供的 Python DSL，目标是在 Python 表达力和 CUDA 硬件控制之间取得平衡。它可以表达：

- tensor layout；
- TMA copy；
- Tensor Core MMA；
- warp specialization；
- thread block cluster；
- TMEM 和特定架构指令。

它不是普通 PyTorch op，也不等于 Triton。通常需要 NVIDIA CUDA、CUTLASS 和特定 SM 架构。

### 9.2 SGLang 中的两类来源

#### SGLang / 项目侧 CuTe DSL

例如 [cutedsl_bf16_gemm.py](../../../python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py)，使用 `cutlass.cute` 实现面向 SM100/Blackwell 的 BF16 GEMM。代码会根据 tile、MMA、TMA 和 cluster 组织方式做编译期特化。

#### FlashInfer 提供的 CuTe DSL

例如 [cutedsl_mla_backend.py](../../../python/sglang/srt/layers/attention/cutedsl_mla_backend.py) 和 [cutedsl_gdn_mtp_ring.py](../../../python/sglang/kernels/ops/attention/cutedsl_gdn_mtp_ring.py)，SGLang 负责 backend 选择、输入布局适配和 fallback，底层实现来自 FlashInfer 或其 vendored DSL 代码。

### 9.3 适用场景

- Blackwell/Hopper 等新架构上的 Tensor Core 热点；
- 需要 TMA、TMEM、cluster 或 warp specialization；
- 单纯 Triton 难以表达或性能不足的 GEMM/attention；
- 目标 shape、layout 和硬件范围相对明确。

### 9.4 fallback 是设计的一部分

CuTe DSL 通常不是所有平台和所有 dtype 的唯一实现。例如：

```text
BF16 + SM100 + 合法 layout
    → CuTe DSL fast path
其他 dtype / 架构 / state 类型
    → Triton 或 native fallback
```

[fused_op.py](../../../python/sglang/kernels/fused_op.py) 的 capability 以及具体 wrapper 的 shape gate 要共同保证“不满足条件时选择可用路径”，不能只依赖 DSL 内部报错。

### 9.5 代价

- 对 CUDA、CUTLASS、Python DSL 版本要求敏感；
- 编译错误和 layout 错误的诊断成本高；
- 不能把“能生成代码”直接等同于“覆盖所有 shape”；
- 在 ROCm、NPU、CPU 上通常需要完全不同的实现或 fallback。

---

## 10. 方式七：TileLang

### 10.1 是什么

TileLang 是面向 tile/block 组织的 kernel DSL。SGLang 当前有实际使用，代表文件是 [tilelang_kernel.py](../../../python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py)。旧的 NSA 路径 [srt/layers/attention/nsa/tilelang_kernel.py](../../../python/sglang/srt/layers/attention/nsa/tilelang_kernel.py) 主要作为兼容转发入口。

典型代码使用 `@tilelang.jit` 生成 kernel，并可通过 pass config、block 配置和后端分支调整实现。

### 10.2 代表场景

DSA/稀疏 attention 中，TileLang 路径可能针对：

- activation quantization；
- FP8 index；
- sparse forward；
- paged MQA logits。

具体实现还会根据 NVIDIA 与 HIP 采用不同 block、线程数、inner iteration，以及 HIP 的 partial/combine 两阶段路径，见 [tilelang_kernel.py:1317-1425](../../../python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py#L1317)。

### 10.3 TileLang 与 Triton 的区别

| 维度 | Triton | TileLang |
|---|---|---|
| 表达重点 | program instance、指针和 tile | 更显式的 tile/block 计算组织 |
| 常见用途 | elementwise、reduction、attention、MoE | 结构化稀疏、复杂 tile pipeline |
| 调优方式 | `constexpr`、autotune、heuristics | tile layout、pass config、后端参数 |
| 运行时 | Triton compiler/cache | TileLang compiler/cache |
| 兼容性 | 依赖 Triton backend | 依赖 TileLang 版本和 target |

二者都可能是 runtime JIT，也都可能被外层 custom op 隔离。不能因为“都用 Python 写”就认为生成路径和缓存机制相同。

### 10.4 适用边界

TileLang 适合已有成熟 tile 算法、需要显式 block 结构且目标 backend 明确的场景。对于普通逐元素或简单归约，Triton 通常更容易维护；对于极致架构特化，CuTe DSL 或手写 CUDA 可能更合适。

---

## 11. 方式八：外部 kernel 库封装

外部库封装的本质是：SGLang 不重新实现底层 kernel，而是负责**选择、参数适配、能力门禁、生命周期和 fallback**。

### 11.1 FlashInfer

代表入口：

- [flashinfer_backend.py](../../../python/sglang/srt/layers/attention/flashinfer_backend.py)；
- [flash_mla_sm120.py](../../../python/sglang/kernels/ops/attention/flash_mla_sm120.py)；
- [cutedsl_mla_backend.py](../../../python/sglang/srt/layers/attention/cutedsl_mla_backend.py)。

典型链路：

```text
SGLang attention backend
    ↓
整理 paged KV / ragged / MLA metadata
    ↓
lazy import FlashInfer callable
    ↓
FlashInfer 内部选择预编译 kernel、JIT kernel 或 CuTe DSL
    ↓
必要时回退 Triton/native
```

FlashInfer 可能同时提供 attention、MLA、MoE、量化、通信和 GDN 相关 kernel。SGLang 的职责不是复刻实现，而是确保：

- page size 和 layout 转换正确；
- dtype 和设备满足外部库约束；
- `torch.compile` 不追踪外部动态加载；
- 不支持的线程数、shape 或架构有明确 fallback。

### 11.2 FlashAttention

代表入口：[flash_attention_v3.py](../../../python/sglang/kernels/ops/attention/flash_attention_v3.py)。当前可能存在多条路径：

```text
支持 FA3 的架构
    → AOT/in-tree FA3
不支持 FA3 或 ROCm/旧架构
    → 外部 flash_attn FA2 或其他 fallback
```

调用方还需要结合 [flashattention_backend.py](../../../python/sglang/srt/layers/attention/flashattention_backend.py) 的 metadata 生命周期，区分 decode、extend、draft extend、CUDA Graph 支持范围。

### 11.3 DeepGEMM

DeepGEMM 主要是外部库 wrapper，不是 SGLang TVM-FFI JIT。代表文件：

- [moe_runner/deep_gemm.py](../../../python/sglang/srt/layers/moe/moe_runner/deep_gemm.py)；
- [deep_gemm_wrapper/configurer.py](../../../python/sglang/srt/layers/deep_gemm_wrapper/configurer.py)；
- [deep_gemm_wrapper/entrypoint.py](../../../python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py)；
- [deep_gemm_wrapper/compile_utils.py](../../../python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py)。

SGLang 负责：

- 检查 DeepGEMM 是否安装；
- 根据 CUDA/MUSA 和 SM 架构启停；
- 根据 shape、dtype、MoE layout 选择接口；
- 管理 warmup、预编译、cache 和编译锁；
- 在不满足条件时回退其他 MoE runner。

`SGLANG_ENABLE_JIT_DEEPGEMM` 等开关只影响 DeepGEMM 的启用和编译策略，不应与 SGLang 自有 `load_jit()` 混淆。

### 11.4 AITER

AITER 是 ROCm 生态中的外部/平台专用高性能实现来源，在统一 `BaseFusedOp` 中以 `forward_aiter` 表示。它往往和 HIP 平台能力、特定 gfx 架构以及输入 layout 强绑定。

### 11.5 外部库封装的共同原则

1. lazy import，避免导入主包就要求所有可选库存在；
2. capability gate，明确 device、架构、dtype 和 shape；
3. custom op boundary，阻断 Dynamo 对动态编译和文件 I/O 的追踪；
4. reference parity，和 `forward_native` 或已知参考实现做数值校验；
5. fallback，外部库不满足条件时回退，而不是让调用方散落平台判断。

---

## 12. 方式九：平台专用 native / vendor kernel

### 12.1 ROCm / HIP

ROCm 路径可能来自四个层次：

1. HIP AOT/C++ 扩展；
2. Triton HIP；
3. AITER；
4. 外部 FlashAttention/FlashInfer fallback。

JIT 基础设施会把 CUDA、HIP、MUSA target 分别纳入 cache key；HIP 路径还要处理：

- `cuda*` 到 `hip*` 的 API 映射；
- shuffle、atomic 和 FP8 类型差异；
- 不适用的 CUDA fast-math flags；
- gfx942/gfx950 等架构宏。

不要把“CUDA 源码可以被 hipify”当成“CUDA 和 HIP 语义完全一致”。FP8 表示、动态 shared memory、编译器行为和 warp primitive 都需要单独验证。

### 12.2 NPU

NPU 既有 `torch_npu` native 路径，也有 standalone `.so` custom op。代表 loader 是 [extra_ops_loader.py](../../../python/sglang/srt/hardware_backend/npu/extra_ops_loader.py)：

```text
OpLibSpec
    ↓
TorchOpLoader.initialize()
    ↓
torch.ops.load_library(standalone.so)
    ↓
验证 required ops 是否注册
    ↓
NPU backend 调用 torch.ops.namespace.op
```

attention fallback 见 [ascend_torch_native_backend.py](../../../python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py)，它使用 PyTorch/NPU operator 组合完成 query/key/value、mask、GQA、softmax 等逻辑。

### 12.3 CPU

CPU 路径不是单纯的 native PyTorch。当前 AOT CPU 工程位于 [aot/csrc/cpu/](../../../python/sglang/kernels/aot/csrc/cpu/)，按 x86_64、aarch64、ppc64 等架构筛选源文件，并针对 x86 可能启用：

- AMX；
- AVX512 BF16；
- AVX512 VNNI；
- INT8；
- CPU attention / GEMM 等专用实现。

平台 dispatch 通过 `forward_cpu` 进入 AMX-capable 路径，否则通常保留 native fallback。

### 12.4 XPU / MUSA

XPU 目前可见 Triton-compatible FLA/Gated Delta Rule 路径。MUSA 则在 JIT cache 和部分外部库/平台分支中有专门处理。

平台实现的基本规则是：

- 平台方法只描述 device 选择；
- backend 方法描述实现来源；
- 不支持隐式跨平台 fallback；
- fallback 必须在 capability 或 wrapper 层明确表达。

例如 HIP 可以按既有约定回退到 CUDA 代码路径，但 MUSA 不会自动继承 `forward_cuda`，需要显式 `forward_musa` 选择是否复用。

---

## 13. 编译、加载和缓存时机对比

### 13.1 统一时间线

```text
安装/构建阶段
├─ AOT C++/CUDA/CPU extension 编译
├─ 外部 wheel 安装或下载 cubin
└─ 主包打包 JIT 源码和 Python wrapper

进程启动阶段
├─ 导入 sglang.kernels 元数据
├─ 建立 registry
└─ 不应触发所有 backend 编译

第一次实际调用
├─ native：直接调用 PyTorch
├─ AOT：lazy load 已存在的 .so
├─ Triton：遇到配置时可能 runtime JIT
├─ TVM-FFI：load_jit 查 cache，miss 时 nvcc/hipcc
├─ CuTe DSL：按 DSL/外部库机制 JIT 或加载
├─ DeepGEMM：按外部库机制 warmup/JIT/cubin
└─ NPU：load_library 后调用已注册 op

后续调用
└─ 进入已缓存的 dispatch callable 和底层 kernel cache
```

### 13.2 对首请求延迟的影响

| 路径 | 首次请求可能发生什么 | 生产建议 |
|---|---|---|
| native | 通常无额外编译 | 作为可靠 fallback |
| custom op | 注册通常很轻；底层按实现决定 | 预先完成 schema/fake 注册 |
| Triton | 可能编译新 meta 配置 | 代表 shape warmup |
| TVM-FFI JIT | 可能运行 Ninja + nvcc/hipcc | 共享 JIT cache 或显式预热 |
| AOT | 加载 `.so` | 发布前覆盖目标架构 |
| CuTe DSL | 可能 DSL 编译或外部库调优 | 预热目标 layout/shape |
| TileLang | 可能生成并编译新 kernel | 限制配置组合并预热 |
| DeepGEMM | 可能 warmup / NVRTC / 外部 JIT | 使用 compile tool 预编译 |
| NPU `.so` | 动态库加载 | 启动时验证 required ops |

### 13.3 为什么要区分“进程内缓存”和“磁盘缓存”

- `BaseFusedOp` / `get_kernel()` 缓存的是 Python callable 和 dispatch 结果；
- `load_jit()` cache 缓存的是按源码、flags、依赖和 target 计算出的 `.so`；
- Triton、TileLang、DeepGEMM、FlashInfer 还有各自的编译缓存；
- AOT wheel 不依赖这些 JIT cache 才能获得基础执行能力。

清理某一个 cache 不一定会影响其他 cache，也不一定能解决 ABI 或架构错误。

---

## 14. 一个 kernel 从源码到请求执行的完整链路

以下以一个同时存在 native、Triton、JIT、AOT 的逻辑算子抽象说明：

```text
模型层 / attention backend / scheduler helper
    ↓
sglang.kernels.ops.<group>.op(...)
    ↓
模块级 BaseFusedOp 实例
    ↓
BaseFusedOp.forward(...)
    ├─ 显式 backend？直接调用对应 forward_<backend>
    ├─ 强制 backend？尝试全局指定路径
    ├─ capability + priority？选择优化 backend
    ├─ 平台 forward？CUDA/HIP/NPU/CPU 等
    └─ forward_native？最终参考实现
    ↓
具体实现 wrapper
    ├─ native：torch operator 组合
    ├─ triton：调用 @triton.jit kernel
    ├─ jit：_jit_module → load_jit → TVM-FFI function
    ├─ aot：torch.ops.sgl_kernel.<op>
    ├─ flashinfer：外部 callable + custom op boundary
    ├─ deepgemm：外部 GEMM API + shape/config gate
    └─ cute_dsl/tilelang：DSL runtime callable
    ↓
GPU/NPU/CPU kernel
    ↓
输出张量
```

这个链路里有三种“选择”：

1. **公共入口选择**：调用哪个 operator id；
2. **backend 选择**：native、Triton、JIT、AOT 或外部库；
3. **kernel variant 选择**：在某个 backend 内按 head dim、dtype、SM、page size、batch size 选择具体模板实例。

三者不能混为一个 if/else。尤其是 `BaseFusedOp` 负责逻辑算子级选择，Triton/CUDA/DeepGEMM 内部还可能继续做 variant dispatch。

---

## 15. 如何选择实现方式

### 15.1 决策表

| 需求 | 首选 | 备选 |
|---|---|---|
| 先实现正确性和跨平台 | PyTorch native | `torch.compile` native |
| 简单融合、快速实验 | Triton | TVM-FFI JIT |
| 复杂 CUDA 模板和低层控制 | TVM-FFI JIT | AOT CUDA |
| 生产首调不能编译 | AOT `sgl-kernel` | 预热后的外部库 |
| Blackwell TMA/TMEM/cluster | CuTe DSL | 手写 CUDA/CUTLASS |
| 结构化 tile/sparse 算法 | TileLang | Triton / CuTe DSL |
| 成熟 attention/MoE/GEMM | FlashInfer/DeepGEMM/FlashAttention | AOT 自研实现 |
| ROCm 专用性能路径 | AITER/HIP/Triton HIP | native fallback |
| NPU 专用 op | torch_npu / standalone `.so` | NPU native fallback |
| CPU 专用指令优化 | CPU AOT AMX/AVX | native PyTorch |

### 15.2 选择时必须回答的六个问题

1. **首调用延迟能否接受？** 能接受才考虑 runtime JIT。
2. **目标 shape 是否稳定且可枚举？** 稳定才适合 AOT。
3. **需要控制到什么层次？** tile 级、warp 级、指令级对应 Triton、CuTe/TileLang、手写 CUDA 的不同成本。
4. **是否要支持 `torch.compile`？** 需要就准备 custom op、fake impl 或 compile-safe path。
5. **平台范围是什么？** CUDA-only、CUDA+HIP、NPU、CPU 的实现策略不同。
6. **是否已有外部成熟实现？** 如果有，优先 wrapper + capability gate，而不是重复造轮子。

### 15.3 推荐的渐进路线

对一个新算子，通常采用：

```text
native reference
    ↓
Triton 或外部库原型
    ↓
统一 BaseFusedOp / custom op 接口
    ↓
TVM-FFI JIT 或 CuTe/TileLang 特化
    ↓
热点和组合稳定后迁移 AOT
```

这不是强制流程。若算子本身是生产必需的架构专用热点，且已有成熟 CUDA 实现，可以直接进入 AOT，但仍应保留 native 参考和针对边界 shape 的 fallback。

---

## 16. 新增 kernel 的推荐流程

### 16.1 先确定逻辑 operator contract

先定义：

- operator id：`<group>.<name>`；
- 输入和输出 signature；
- dtype、device、stride 和 contiguous 要求；
- inplace 还是 out-of-place；
- 支持的 shape 和边界；
- `torch.compile` 需要的 fake output；
- 哪些 backend 是默认候选，哪些只是显式实验路径。

公共入口应放在 `python/sglang/kernels/ops/<group>/`，不要继续扩展已经移除的旧 `python/sglang/jit_kernel/` namespace。

### 16.2 先写 native reference

native 实现应作为：

- 数值正确性 ground truth；
- 不支持架构/shape 的 fallback；
- `torch.compile` 默认安全路径；
- 单测中用于 parity 的参考。

### 16.3 选择一个具体实现后端

- Triton：新增 `forward_triton` 或独立 Triton callable；
- TVM-FFI JIT：在 `kernels/jit` 基础设施上添加 `csrc` 和 `load_jit` wrapper；
- AOT：在 `kernels/aot/csrc` 和对应 Python wrapper 中接入；
- CuTe DSL：声明架构和 layout capability，并实现 fallback；
- TileLang：明确 target、pass config、shape 和后端分支；
- 外部库：使用 lazy import、custom op 和 capability gate；
- 平台专用：实现 `forward_<platform>` 或 `forward_<backend>`，不要把两者含义混写。

### 16.4 处理 compile 和加载边界

如果底层有动态编译、文件 I/O 或动态 import：

1. 用 `register_custom_op()` 或 `register_custom_op_from_extern()` 包起来；
2. 提供 fake implementation 或 output shape；
3. 确认 `mutates_args`；
4. 确认 `BaseFusedOp.enter_torch_compile()` 时不会进入不可追踪路径。

### 16.5 接入 registry 和 capability

使用 `register_fused_op()` 让实例的 backend 元数据进入 registry。`capabilities` 应描述真实硬件能力，例如 CUDA、HIP、SM90、SM100、dtype 或其他必要条件；shape-dependent gate 放到 `backend_eligible()`。

### 16.6 决定是否 AOT

满足以下条件再进入 AOT：

- 经过真实 workload benchmark；
- shape 组合已经收敛；
- 首调延迟确实是线上问题；
- 依赖和架构矩阵有维护能力；
- AOT 产物比 JIT cache 更适合发布和审计。

---

## 17. 测试、调试与性能验证

### 17.1 正确性测试

至少覆盖：

- native vs 新 backend；
- 多 dtype；
- 最小、典型、最大 shape；
- 非 contiguous 输入（如果 contract 允许）；
- page size、head dim、batch size 等真实变体；
- CUDA Graph 或 `torch.compile` 模式（如果声明支持）；
- ROCm/NPU/CPU 的平台 fallback（如果声明支持）。

统一 kernel 测试位于 `test/registered/kernels/`，测试和 benchmark 组织说明见 [python/sglang/kernels/README.md](../../../python/sglang/kernels/README.md)。

### 17.2 首次调用和 warm cache 测试

性能测试要分别记录：

1. cold process + cold compiler cache；
2. cold process + warm disk cache；
3. warm process + warm kernel cache；
4. 是否包含 kernel launch 开销；
5. 是否在 CUDA Graph 内执行。

不能只报告 warm kernel 时间，然后把首请求 JIT 延迟忽略掉。

### 17.3 backend 实际运行确认

不要只看“注册了 AOT/Triton/JIT”。可以使用：

- `BaseFusedOp` 的 trace；
- 显式 `backend=` 调用；
- `SGLANG_FORCE_FUSED_OP_BACKEND` 做全局二分；
- kernel API logging；
- profiler/Nsight/torch profiler；
- 编译 cache 的 module 和 target 信息。

注册表是 inventory，实际运行路径要结合 dispatch、capability 和 trace 判断。

### 17.4 常见性能误判

- 把 JIT 编译时间算进某个 backend 的 steady-state kernel 时间；
- native fallback 实际运行，却误以为跑的是 Triton/AOT；
- benchmark 输入没有覆盖真实 page layout；
- 只测单请求，不测 batch/seq length 变体；
- 忽略外部库的 warmup 和 autotune；
- 只测 kernel duration，不测 Python dispatch、metadata 构造和同步。

---

## 18. 常见误区

### 18.1 “JIT kernel 就是 `jit_kernel` 目录”

当前基线中旧 `python/sglang/jit_kernel/` 包已在 RFC #29630 收尾（commit `99f636a86f`）中退役：所有代码已迁走，作为 import 入口的 Python 包已不存在（磁盘上可能残留一个空的 `jit_kernel/triton/` 目录，但无任何 `.py`）。JIT 基础设施在 `python/sglang/kernels/jit/`，具体公共算子在 `python/sglang/kernels/ops/`。

### 18.2 “Triton、TVM-FFI、DeepGEMM 都是同一个 JIT”

它们的编译器、cache、target、依赖和错误边界不同：

- Triton 使用 Triton compiler/cache；
- SGLang JIT 使用 `load_jit()`、Ninja 和 TVM-FFI；
- DeepGEMM 使用外部库自己的 JIT/cubin/NVRTC 流程；
- CuTe DSL 和 TileLang 也有各自机制。

### 18.3 “AOT 目录就是顶层 `sgl-kernel`”

当前分支的 AOT 工程位于 `python/sglang/kernels/aot/`。引用路径时应以当前 checkout 为准。

### 18.4 “注册了 backend 就会自动优先使用”

registry registration 只登记元数据。`BaseFusedOp` 还要看：

- 方法是否真的覆盖；
- `capabilities` 是否声明；
- 当前平台是否满足；
- `priority` 顺序；
- `backend_eligible()` 的 per-call shape gate。

### 18.5 “CUDA/HIP 是 KernelBackend”

CUDA/HIP 是 platform/device；AOT、JIT、Triton、AITER 是 implementation provenance。统一 contract 已将两者分开。

### 18.6 “有 fallback 就不需要测试 unsupported shape”

fallback 本身需要测试。尤其要确认：

- fallback 是否真的触发；
- 输出 dtype 和 device 是否一致；
- fallback 是否改变 inplace 语义；
- 外层 `torch.compile` 是否还能追踪；
- 性能下降是否在可接受范围。

### 18.7 “可以在 kernel 内部随便 import 和编译”

动态 import、路径解析和编译应处在明确的 lazy boundary 中，并通过 custom op 或 compile mode 防止 Dynamo 追踪。公共 wrapper 要保持 signature 稳定，底层编译细节不能泄漏到模型调用方。

---

## 19. 关键文件速查

| 主题 | 文件 |
|---|---|
| 统一 namespace 和迁移说明 | [python/sglang/kernels/README.md](../../../python/sglang/kernels/README.md) |
| registry 元数据 | [python/sglang/kernels/registry.py](../../../python/sglang/kernels/registry.py) |
| callable selector | [python/sglang/kernels/selector.py](../../../python/sglang/kernels/selector.py) |
| 多 backend / 多平台 contract | [python/sglang/kernels/fused_op.py](../../../python/sglang/kernels/fused_op.py) |
| custom op 注册和外部函数封装 | [python/sglang/srt/utils/custom_op.py](../../../python/sglang/srt/utils/custom_op.py) |
| TVM-FFI JIT loader | [python/sglang/kernels/jit/utils/compile/loader.py](../../../python/sglang/kernels/jit/utils/compile/loader.py) |
| JIT cache / build key | [python/sglang/kernels/jit/utils/compile/cache.py](../../../python/sglang/kernels/jit/utils/compile/cache.py) |
| Triton attention backend | [python/sglang/srt/layers/attention/triton_backend.py](../../../python/sglang/srt/layers/attention/triton_backend.py) |
| Triton MoE | [python/sglang/srt/layers/moe/fused_moe_triton/](../../../python/sglang/srt/layers/moe/fused_moe_triton/) |
| AOT CMake | [python/sglang/kernels/aot/CMakeLists.txt](../../../python/sglang/kernels/aot/CMakeLists.txt) |
| AOT Python loader | [python/sglang/kernels/aot/python/sgl_kernel/](../../../python/sglang/kernels/aot/python/sgl_kernel/) |
| CuTe DSL GEMM | [python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py](../../../python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py) |
| CuTe DSL attention | [python/sglang/srt/layers/attention/cutedsl_mla_backend.py](../../../python/sglang/srt/layers/attention/cutedsl_mla_backend.py) |
| TileLang DSA | [python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py](../../../python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py) |
| DeepGEMM wrapper | [python/sglang/srt/layers/moe/moe_runner/deep_gemm.py](../../../python/sglang/srt/layers/moe/moe_runner/deep_gemm.py) |
| DeepGEMM configuration/cache | [python/sglang/srt/layers/deep_gemm_wrapper/](../../../python/sglang/srt/layers/deep_gemm_wrapper/) |
| FlashInfer attention | [python/sglang/srt/layers/attention/flashinfer_backend.py](../../../python/sglang/srt/layers/attention/flashinfer_backend.py) |
| FlashAttention dispatch | [python/sglang/kernels/ops/attention/flash_attention_v3.py](../../../python/sglang/kernels/ops/attention/flash_attention_v3.py) |
| NPU standalone op loader | [python/sglang/srt/hardware_backend/npu/extra_ops_loader.py](../../../python/sglang/srt/hardware_backend/npu/extra_ops_loader.py) |
| NPU native attention | [python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py](../../../python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py) |
| CPU AOT CMake | [python/sglang/kernels/aot/csrc/cpu/CMakeLists.txt](../../../python/sglang/kernels/aot/csrc/cpu/CMakeLists.txt) |

---

## 总结

SGLang 的 kernel 体系可以概括为四层：

```text
第一层：PyTorch native reference
    保证正确性、compile-safe 和最终 fallback

第二层：统一算子 contract
    BaseFusedOp + KernelRegistry + selector
    把不同来源的实现放到同一逻辑 operator 下

第三层：多种 kernel 实现方式
    Triton / TVM-FFI JIT / AOT / CuTe DSL / TileLang
    以及 FlashInfer / DeepGEMM / FlashAttention / AITER

第四层：平台和发布系统
    CUDA / HIP / NPU / CPU / XPU / MUSA
    加上 wheel、JIT cache、外部库 cache 和 runtime dispatch
```

工程上最重要的不是为每个算子选择“最底层”的实现，而是先建立稳定的 operator contract，再让不同实现按 capability、shape、平台和编译时机加入。这样可以同时获得 native 的可靠性、JIT 的迭代速度、AOT 的生产稳定性，以及外部库和新硬件 DSL 的性能收益。
