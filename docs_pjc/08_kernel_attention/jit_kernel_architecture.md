# SGLang JIT Kernel 方案系统梳理

> 路径：`python/sglang/jit_kernel/`
> 目标读者：需要在 SGLang 中编写、维护或理解轻量级 CUDA/HIP 算子的工程师。
> 本文由浅入深，覆盖设计动机、整体架构、核心编译管线、Python 包装范式、C++ 复用库、设备端工具、架构自适应、依赖注入、实战示例与 CI 集成。

---

## 目录

1. [总览：JIT Kernel 是什么、为什么需要它](#1-总览)
2. [JIT vs AOT：两条算子落地路线对比](#2-jit-vs-aot)
3. [整体架构与数据流](#3-整体架构与数据流)
4. [核心编译管线 `load_jit`](#4-核心编译管线-load_jit)
5. [Python 包装层范式](#5-python-包装层范式)
6. [C++ 复用库（include/sgl_kernel）](#6-c-复用库)
7. [`TensorMatcher`：声明式张量校验](#7-tensormatcher)
8. [`LaunchKernel` / PDL / Cluster 启动器](#8-launchkernel--pdl--cluster)
9. [dtype 系统与设备端数学库](#9-dtype-系统与设备端数学库)
10. [架构检测与 HIP/ROCm 回退](#10-架构检测与-hiprocm-回退)
11. [依赖注册（flashinfer / cutlass）](#11-依赖注册)
12. [目录结构全景](#12-目录结构全景)
13. [实战示例：由简到繁](#13-实战示例)
14. [测试、基准与 CI 集成](#14-测试基准与-ci-集成)
15. [附录：常见问题与排错](#15-附录)

---

## 1. 总览

`jit_kernel` 是 SGLang 内置的**运行时（Just-In-Time）CUDA/HIP 算子编译框架**。它允许开发者用极少的样板代码，把一个 `.cuh` 模板核函数“即写即用”地暴露成 Python 可调用函数，编译发生在**首次调用时**而非安装时。

它解决的核心痛点：

- **编译期组合爆炸**：算子常常对 `hidden_size`、`dtype`、`head_dim` 等做模板特化。AOT 方式要预先实例化所有组合，二进制体积和编译时间都不可控。JIT 只在真正用到某个组合时才编译该特化。
- **迭代速度**：改一个 `.cuh` 不需要重新 `pip install`，下次调用自动重编。
- **架构自适应**：根据当前 GPU 的 compute capability 注入正确的编译宏（如 PDL、Blackwell 256-bit 向量化），无需为每种架构维护分支二进制。

底层编译引擎是 **Apache TVM FFI**（`tvm_ffi.cpp.load_inline` / `load`），张量在 C++ 侧表现为 `tvm::ffi::TensorView`（零拷贝视图）/ `tvm::ffi::Tensor`（持有所有权）。

一句话定位：**`jit_kernel` = “一个薄 Python 包装 + 一套 C++ 复用头文件库 + TVM-FFI 运行时编译”，让 CUDA 算子的开发体验接近写普通 Python 函数。**

---

## 2. JIT vs AOT

SGLang 同时维护两条算子落地路线，二者互补：

| 维度 | JIT (`python/sglang/jit_kernel`) | AOT (`sgl-kernel`) |
|---|---|---|
| 编译时机 | 首次调用时运行时编译 | `pip install` 安装时预编译 |
| 模板组合 | 按需特化，调用到才编译 | 需预先实例化全部组合 |
| 启动延迟 | 首次调用有编译开销（之后缓存） | 零编译延迟 |
| 二进制体积 | 不占安装包体积 | 计入 wheel |
| 迭代速度 | 改 `.cuh` 即生效，无需重装 | 需重新构建 wheel |
| 适用场景 | 高维模板特化、快速实验、架构相关优化 | 稳定、热点、组合有限的核函数 |
| FFI 机制 | TVM FFI（`load_inline`/`load`） | PyTorch custom op / pybind |
| 张量类型 | `tvm::ffi::TensorView` | `at::Tensor` |

> 选择经验：算子参数空间大（多 `hidden_size` × 多 `dtype`）、或处于快速演进期，优先 JIT；算子已稳定、组合收敛、且对首调延迟敏感，考虑迁移到 AOT。

---

## 3. 整体架构与数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Python 调用方                                  │
│   rmsnorm(input, weight, out)  /  add_constant(src, c)  ...            │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │ @debug_kernel_api / @register_custom_op
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Python 包装层（norm.py / add_constant.py ...）        │
│   @cache_once _jit_xxx_module(特化参数) ──► make_cpp_args(...) ──┐      │
│   module.kernel_name(tensors...)                                 │      │
└──────────────────────────────────────────────────────────────────┼─────┘
                                                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       load_jit()  (utils.py)                           │
│   • 拼装 module_name = "sgl_kernel_jit_" + "_".join(args)              │
│   • header_only=True : 生成 #include + TVM_FFI_DLL_EXPORT_TYPED_FUNC   │
│   • header_only=False: 直接编译 .cpp，C++ 侧 register_once 导出         │
│   • 注入 arch flags (-DSGL_CUDA_ARCH=...) / ROCm flags                 │
│   • 注入 extra_dependencies 的 include 路径 (flashinfer/cutlass)       │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │ _jit_compile_context() 设置 TVM_FFI_CUDA_ARCH_LIST
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                tvm_ffi.cpp.load_inline / load                          │
│   nvcc / hipcc 编译 .cuh（含 include/sgl_kernel/*.cuh 复用库）          │
│   产出 .so，加载为 tvm_ffi.Module                                       │
└───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       C++/CUDA Kernel                                   │
│   host::LaunchKernel + TensorMatcher + device:: 工具                    │
└──────────────────────────────────────────────────────────────────────┘
```

关键设计：Python 侧只负责**特化参数选择 + 缓存 + 校验日志**，所有真正的张量校验、launch 配置、向量化、归约都下沉到 C++ 复用库，保证多个算子共享同一套经过验证的基础设施。

---

## 4. 核心编译管线 `load_jit`

`load_jit`（`utils.py`）是整个框架的心脏。签名：

```python
def load_jit(
    *args: str,                                # 模块唯一标识符（不同特化必须不同）
    cpp_files: List[str] | None = None,        # .cpp 源（相对 csrc/）
    cuda_files: List[str] | None = None,       # .cuh/.cu 源（相对 csrc/）
    cpp_wrappers: List[Tuple[str, str]] | None = None,   # (导出名, C++类/函数名)
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags / extra_cuda_cflags / extra_ldflags / extra_include_paths,
    extra_dependencies: List[str] | None = None,         # "flashinfer" / "cutlass"
    build_directory: str | None = None,
    header_only: bool = True,
) -> Module
```

### 4.1 模块命名与缓存键

```python
module_name = "sgl_kernel_jit_" + "_".join(str(arg) for arg in args)
```

`*args` 是模块的**唯一标识**，必须能区分所有特化。典型做法是把 `make_cpp_args(...)` 的结果整个展开传入——这样 `hidden_size=512, dtype=fp16` 与 `hidden_size=1024, dtype=bf16` 自然落到不同的 `.so`。TVM-FFI 内部按 `module_name` 做磁盘缓存，已编译过的特化直接复用 `.so`。

### 4.2 两种导出模式

#### `header_only=True`（绝大多数算子）

C++ 侧写的是**纯头文件模板**（`.cuh`），不含任何导出宏。`load_jit` 自动生成内联源：

```cpp
// cuda_sources 自动生成：
#include "/abs/path/csrc/elementwise/rmsnorm.cuh"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm, (RMSNormHalfKernel<512, true, fp16_t>::run));
```

- `cuda_files` → `#include "<绝对路径>"`
- `cuda_wrappers=[(export_name, kernel_name)]` → `_make_wrapper` 生成 `TVM_FFI_DLL_EXPORT_TYPED_FUNC(export_name, (kernel_name));`
- 调用侧：`module.<export_name>(args...)`

优势：C++ 代码完全是可被 `.clangd` 索引的模板，无导出污染；模板参数直接拼进 `kernel_name` 字符串。

#### `header_only=False`（如 `ngram_corpus`）

直接编译 `.cpp`，由 **C++ 侧自己注册导出**（`module.register_once()`），Python 侧用 `@tvm_ffi.register_object("sgl.XXX")` 绑定一个 `tvm_ffi.Object` 子类，通过 `__ffi_init__` 构造。适用于需要**持有状态的对象**（如语料库、缓存表），而非无状态函数。

```python
# ngram_corpus.py 模式
module = load_jit("ngram_corpus", cpp_files=[...5 个 .cpp...], header_only=False)
module.register_once()

@tvm_ffi.register_object("sgl.NgramCorpus")
class NgramCorpusFFI(tvm_ffi.Object):
    def __init__(self, ...):
        self.__init_handle_by_constructor__(self.__ffi_init__, ...)
```

| 对比 | header_only=True | header_only=False |
|---|---|---|
| C++ 形态 | 纯头模板 `.cuh` | 完整 `.cpp` |
| 导出方式 | Python 侧 wrapper 宏 | C++ 侧 `register_once` |
| 暴露内容 | 无状态函数 | 有状态对象 + 方法 |
| 典型用例 | rmsnorm/add_constant/绝大多数 | ngram_corpus 等 |

### 4.3 编译上下文与 flag 注入

`_jit_compile_context()` 是一个上下文管理器，在编译期临时设置 `TVM_FFI_CUDA_ARCH_LIST` 为当前 GPU 的 `target_name`（如 `9.0`），结束后恢复。ROCm 路径下直接 `yield`（架构通过 `TVM_FFI_ROCM_ARCH_LIST` 另行处理）。

编译 flag 由 `_get_default_target_flags()` 提供基线：

- **CUDA**：`-DSGL_CUDA_ARCH=<major*100+minor*10>`、`-std=c++20`、`-O3`、`--expt-relaxed-constexpr`
- **ROCm**：`-DUSE_ROCM`、`-std=c++20`、`-O3`，并根据 `gcnArchName` 选择 FP8 类型（`gfx942` → `-DHIP_FP8_TYPE_FNUZ=1`，否则 `-DHIP_FP8_TYPE_E4M3=1`）

默认还叠加 `DEFAULT_CFLAGS=["-std=c++20","-O3"]` 与 `DEFAULT_INCLUDE=[KERNEL_PATH/"include"]`，确保所有算子都能 `#include <sgl_kernel/...>`。

---

## 5. Python 包装层范式

几乎每个算子的 Python 文件都遵循同一套“三件套”范式。以最小例子 `add_constant.py` 为骨架：

```python
@cache_once
def _jit_add_constant_module(constant: int) -> Module:
    args = make_cpp_args(constant)               # ① 特化参数 → C++ 模板实参
    return load_jit(
        "add_constant", *args,                   # ② 编译 + 缓存
        cuda_files=["add_constant.cuh"],
        cuda_wrappers=[("add_constant", f"add_constant<{args}>")],
    )

@debug_kernel_api                                # ③ 可选：调试日志包装
def add_constant(src: torch.Tensor, constant: int) -> torch.Tensor:
    dst = torch.empty_like(src)
    _jit_add_constant_module(constant).add_constant(dst, src)
    return dst
```

### 5.1 `cache_once`：torch.compile 安全的缓存

```python
def cache_once(fn):
    result_map = {}
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]
    return wrapper
```

为什么不用 `functools.lru_cache`？因为 `lru_cache` 与 `torch.compile` 的图捕获不兼容。`cache_once` 是一个手写的“按参数永久缓存”实现，确保**同一特化只编译一次**，且能安全地出现在被 compile 的代码路径中。

### 5.2 `make_cpp_args` / `CPPArgList`：Python 值 → C++ 模板实参

```python
CPP_DTYPE_MAP = {
    torch.float:        "fp32_t",
    torch.float16:      "fp16_t",
    torch.bfloat16:     "bf16_t",
    torch.float8_e4m3fn:"fp8_e4m3_t",
    torch.int8:  "int8_t", torch.int32: "int32_t", torch.int64: "int64_t",
}
```

`make_cpp_args(*args)` 把 Python 值逐个转成 C++ 字面量：`bool → true/false`，`int/str/float → str()`，`torch.dtype → CPP_DTYPE_MAP[...]`。返回的 `CPPArgList` 是 `list[str]` 子类，其 `__str__` 用 `", ".join(...)`，因此可以直接插进模板字符串：`f"RMSNormKernel<{args}>"` 会展开成 `RMSNormKernel<512, true, fp16_t>`。

### 5.3 `@debug_kernel_api`：零成本可观测性

来自 `python/sglang/kernel_api_logging.py`。由环境变量 `SGLANG_KERNEL_API_LOGLEVEL` 门控：

- `0`（默认）：装饰器是**纯 no-op**，无任何运行时开销
- `>0`：记录函数名、参数、张量 shape/dtype/device
- `>=10`：额外把输入/输出张量 dump 到 `SGLANG_KERNEL_API_DUMP_DIR`

它在 **torch.compile 编译期与 CUDA Graph 捕获期自动跳过**，避免污染图。配套还有 `debug_torch_op`、`wrap_method_with_debug_kernel_once`。这是 `debug-cuda-crash` skill 的基础。

### 5.4 `@register_custom_op`：注册为 PyTorch 自定义算子

对需要被 `torch.compile` 追踪、或要参与 CUDA Graph 的算子（如 `hadamard.py`、`nvfp4.py`），用 `register_custom_op(op_name=, mutates_args=, fake_impl=)` 把 JIT 函数注册成正式的 PyTorch custom op，并提供 `fake_impl`（meta 实现）让 dynamo 能推断输出 shape。

---

## 6. C++ 复用库

`include/sgl_kernel/` 是一套可被所有算子 `#include` 的头文件库，分为 **host 侧**（启动、校验、内存分配）与 **device 侧**（向量化、归约、数学、原子）两类。它是 JIT 框架“样板代码极少”的根本原因——校验、launch、向量化都已封装。

### 6.1 `utils.cuh` / `utils.h`：地基

`utils.cuh`（几乎所有 kernel 都包含）提供：

| 设施 | 说明 |
|---|---|
| 类型别名 | `fp32_t/fp16_t/bf16_t/fp8_e4m3_t/int8_t/...`（CUDA 与 ROCm 各一套定义） |
| `SGL_DEVICE` | `__device__ __forceinline__` 等修饰宏 |
| `kWarpThreads` | `= 32`（warp 线程数） |
| `kMaxVecBytes` | 向量化上限：默认 16B，Blackwell 上 32B（256-bit load） |
| 架构宏 | `SGL_ARCH_HOPPER_OR_GREATER`、`SGL_ARCH_BLACKWELL_OR_GREATER`（由 `-DSGL_CUDA_ARCH` 派生） |
| PDL 助手 | `PDLWaitPrimary` / `PDLTriggerSecondary`（包装 griddepcontrol） |
| `load_as<T>` / `store_as<T>` | 按目标类型重解释读写 |
| `device::pointer::offset` | 指针偏移工具 |
| `host::RuntimeDeviceCheck` | 运行时设备能力检查 |
| `host::LaunchKernel` | RAII 核函数启动器（见 §8） |

`utils.h`（host-only）提供错误处理与小工具：

- `DebugInfo`（基于 `std::source_location` 自动捕获文件/行号）
- `PanicError`（带 `.root_cause()`）、`panic(...)`
- `RuntimeCheck` / `Panic`（利用 CTAD 推导）
- `host::pointer::offset`、`div_ceil`、`dtype_bytes`、`irange`
- `stdr` / `stdv`（`std::ranges` / `std::views` 别名）

### 6.2 `tensor.h`：`TensorMatcher`（详见 §7）

声明式张量形状 / dtype / device 校验。

### 6.3 `type.cuh`：dtype trait 系统

核心是宏 `SGL_REGISTER_DTYPE_TRAIT`，为每个 dtype 注册：

- `self_t`：自身类型
- `packed_t`：向量化打包类型（如 `half2`、`bf16x2`）
- `from(...)`：构造 / 转换
- 一元 / 二元数学函数

```cpp
// fp32 拥有完整数学函数：
abs, sqrt, rsqrt, exp, sin, cos, max, min ...
```

配套：

- `packed_t<T>`：取打包类型
- `device::cast<To, From>(x)`：类型安全转换
- `kFP8E4M3Max`：FP8 E4M3 上界（`448`，FNUZ 变体为 `224`）

### 6.4 内存工具：`vec.cuh` / `tile.cuh`

**`AlignedVector<T, N>`**（`vec.cuh`）——向量化访存：

```cpp
template <typename T, int N>
struct AlignedVector {
    static_assert(is_power_of_2(N));
    static_assert(sizeof(T) * N <= kMaxVecBytes);  // 16B / Blackwell 32B
    // .load(ptr) / .store(ptr) / .fill(v) / operator[] / .data()
    // 内部用 sized_int<T> 作为对齐存储
};
```

128-bit（4×fp32 / 8×fp16）对齐 load/store；在 Blackwell 上 `kMaxVecBytes=32` 允许 256-bit 访存，进一步提升带宽利用率。

**`tile::Memory<T>`**（`tile.cuh`）——协作式分块访问，三种工厂对应不同协作粒度：

| 工厂 | 协作范围 | tsize |
|---|---|---|
| `thread()` | 单线程 | 1 |
| `warp(warp_threads=32)` | warp 内 | 32 |
| `cta(blockDim.x)` | 整个 block | blockDim.x |

`.load(ptr, offset)` 按 `tid + offset * tsize` 索引，`.store(...)`、`.in_bound(...)` 配合边界检查。它把“线程 / 线程束 / 线程块协作读写一段内存”这一常见模式抽象掉。

### 6.5 归约与原子：`warp.cuh` / `cta.cuh` / `atomic.cuh`

| 头文件 | 提供 | CUDA 实现 | ROCm 实现 |
|---|---|---|---|
| `warp.cuh` | `device::warp::reduce_sum/reduce_max` | `__shfl_xor_sync` | `__shfl_xor` |
| `cta.cuh` | `device::cta::reduce_max`（两级：warp 内 + 跨 warp） | smem[0] 存结果，**末尾无多余 sync** | 同左 |
| `atomic.cuh` | `device::atomic::max`（float） | `atomicMax/Min` 整型重解释 | CAS 循环 |

`cta::reduce_max` 的“无 trailing sync”是刻意优化：调用方若已知后续没有对 smem 的竞争，可省一次 `__syncthreads()`。

### 6.6 `runtime.cuh`：设备运行时查询

`host::runtime::` 命名空间：`get_blocks_per_sm`、`get_sm_count`、`get_cc_major`、`get_runtime_version`、`get_available_dynamic_smem_per_block`。HIP 下通过宏别名回退到对应 hip API。这些用于在 launch 前做 occupancy 计算（如 rmsnorm 用它决定 grid 大小）。

### 6.7 `ffi.h`：在 C++ 侧分配张量

当 kernel 需要分配中间 / 输出张量时：

- `host::ffi::empty(shape, dtype, device)`：通过 TVM-FFI 环境分配器（`FromEnvAlloc`）分配
- `host::ffi::empty_like(view)`
- `host::ffi::from_blob(ptr, shape, ...)`：用自定义 deleter 包装已有指针（malloc 出 shape/stride 上下文）
- `from_blob_like`

### 6.8 `impl/norm.cuh`：算子级共享实现

把 norm 类算子的公共逻辑抽到 `impl/` 子目录。例如：

- `host::norm::is_config_supported<T, kDim>`：fp16/bf16；`kDim≤256` 须 ∈{64,128,256}，否则须 `%256==0 && ≤8192`
- `should_use_cta`（`kDim>256`）、`get_cta_threads`（`(kDim/256)*32`）
- `device::norm::apply_norm_warp` / `apply_norm_cta`
- `StorageType<T, kDim>`：cta 路径用 `AlignedVector<packed, 4>`（16B），warp 路径用 `kDim/(2*32)`
- `kSmemBufferSize = 33`（含 bank-conflict padding）

---

## 7. TensorMatcher

`tensor.h` 提供的 `TensorMatcher` 是一套**链式、声明式**的张量校验 DSL，取代了散落在每个 kernel 入口的手写 `TORCH_CHECK`。它的目标：用一行表达 “这些张量必须是某形状、某 dtype、在某设备上，且互相一致”。

### 7.1 基本用法

```cpp
TensorMatcher({N})                 // 期望形状 [N]
    .with_dtype<int32_t>()         // dtype 必须是 int32
    .with_device<kDLCUDA>(device_) // 设备必须是 CUDA，且记录到 device_
    .verify(dst)                   // 校验 dst
    .verify(src);                  // 链式校验 src（与 dst 形状/dtype/device 一致）
```

### 7.2 符号绑定（Symbolic*）

校验的精髓在三个“符号”类型，支持**先绑定后比对**：

| 符号 | 作用 |
|---|---|
| `SymbolicSize` | 形状维度。第一次遇到具体值时绑定，之后所有出现必须相等 |
| `SymbolicDType` | dtype。同上 |
| `SymbolicDevice` | device。同上 |

- 形状里写 `-1` 表示**任意尺寸**（不约束、不绑定）。
- 同一个符号变量在多个张量间复用，即可表达“这几个张量的某维度必须相同”而无需知道具体值。
- `.unwrap()` 取出已绑定的具体值（如拿到运行时的 `N` 用于算 grid）。

### 7.3 stride 规则

- 调用 `.with_strides({...})` 显式指定期望 stride。
- **省略 strides** ⇒ 要求张量 `is_contiguous()`。
- **size 为 1 的维度跳过 stride 检查**（PyTorch 对 size-1 维的 stride 不保证有意义）。

### 7.4 实现要点

`TensorMatcher` 是 **move-only 的临时对象**，链式调用返回右值引用，鼓励一次性用完即丢的写法。校验失败时通过 `utils.h` 的 `Panic`/`RuntimeCheck` 抛出带 `source_location` 的 `PanicError`，错误信息能定位到具体 kernel 文件行号。

---

## 8. LaunchKernel / PDL / Cluster

`host::LaunchKernel`（`utils.cuh`）是 RAII 风格的核函数启动器，封装了 stream 解析、launch 配置与自动错误检查。

### 8.1 基本形态

```cpp
LaunchKernel(grid, block, device)(kernel, args...);
```

- **构造**时从 `DLDevice` 解析 stream：优先 `TVMFFIEnvGetStream`（拿到 SGLang 当前的 stream），或接受裸 `cudaStream_t`。这保证 JIT kernel 跑在与框架一致的 stream 上，能正确参与 CUDA Graph 捕获与 overlap 调度。
- **调用** `operator()(kernel, args...)`：底层用 `cudaLaunchKernelEx`（CUDA）/ `hipLaunchKernelGGL`（ROCm）。
- **析构 / 调用后自动 error-check**：省去手写 `cudaGetLastError`。

### 8.2 链式开关

```cpp
LaunchKernel(grid, block, device)
    .enable_pdl(kUsePDL)       // 开启 Programmatic Dependent Launch
    .enable_cluster(...)       // 开启 thread block cluster
    (kernel, params);
```

### 8.3 PDL（Programmatic Dependent Launch）

PDL 是 Hopper（sm_90+）引入的特性，允许**后继 kernel 在前驱 kernel 仍在收尾时提前启动**，隐藏 launch 延迟、提升背靠背小 kernel 的吞吐。

- 设备端配合 `PDLWaitPrimary`（`griddepcontrol.wait`，等待前驱写入的数据就绪）与 `PDLTriggerSecondary`（`griddepcontrol.launch_dependents`，放行后继）。
- 是否启用由 `is_arch_support_pdl()`（`major >= 9`）在 Python 侧决定，并作为模板布尔参数 `kUsePDL` 传入 kernel；ROCm 上恒为 `false`。
- 典型用法：rmsnorm 等 elementwise kernel 把 `kUsePDL` 作为模板参数，`enable_pdl(kUsePDL)` 条件开启。

### 8.4 `__grid_constant__` 参数结构

复杂 kernel（如 `RMSNormParams`）把所有标量/指针打包进一个 `struct`，并以 `__grid_constant__ const` 方式传入。好处：参数走常量内存、减少寄存器压力、launch 开销稳定。

---

## 9. dtype 系统与设备端数学库

设备端数学统一走 `dtype_trait<T>` 抽象，避免在 kernel 里直接写 `__hmul`/`__float2half` 这类类型相关的内建函数。

### 9.1 分层结构

```
type.cuh          ── dtype_trait / packed_t / cast / kFP8E4M3Max
   │
math.cuh          ── device::math::{abs,sqrt,rsqrt,exp,sin,cos,max,min,...}
   │                  （泛型包装，转发到 dtype_trait 的对应实现）
   │                  常量：log2e / loge2 / FP8_E4M3_MAX
warp.cuh / cta.cuh ── 归约
atomic.cuh         ── 原子
```

### 9.2 packed 类型与向量化计算

`packed_t<T>` 让 kernel 可以一次处理 2 个元素（`half2`/`bf16x2`），配合 `AlignedVector` 的 128/256-bit load，达成“向量化读 + 向量化算”。`device::cast<To, From>` 处理混合精度（如 fp8 ↔ fp16 ↔ fp32）的安全转换。

### 9.3 FP8 上界常量

`kFP8E4M3Max`：

| 平台 | 值 | 说明 |
|---|---|---|
| CUDA / 标准 E4M3 | 448 | OCP FP8 E4M3 |
| ROCm gfx942 FNUZ | 224 | FNUZ 变体（无 inf，指数偏置不同） |

量化类 kernel（fp8_quantize、per_token_group_quant_8bit 等）用它做 clamp 上界，这是 CUDA/ROCm 数值一致性的关键点。

---

## 10. 架构检测与 HIP/ROCm 回退

### 10.1 CUDA 架构探测

```python
@dataclass
class ArchInfo:
    major: int; minor: int; suffix: str
    @property
    def target_name(self): return f"{major}.{minor}{suffix}"   # 如 "9.0", "10.0a"
    @property
    def jit_flag(self):    return f"-DSGL_CUDA_ARCH={major*100 + minor*10}"  # sm_90 -> 900
```

- `_init_jit_cuda_arch_once()`：用 `torch.cuda.get_device_capability()` 探测，失败则置 `(0,0)`（用到时触发编译错误，强制暴露问题）。
- `get_jit_cuda_arch()`：返回当前 `ArchInfo`。
- `-DSGL_CUDA_ARCH` 注入后，C++ 侧据此派生 `SGL_ARCH_HOPPER_OR_GREATER`（≥900）、`SGL_ARCH_BLACKWELL_OR_GREATER`（≥1000）等编译期分支。

### 10.2 架构覆盖：`override_jit_cuda_arch`

某些 kernel 需要特定架构变体（如 nvfp4 要 sm_100a 的 `a` suffix 才能用 tcgen05 指令）：

```python
with override_jit_cuda_arch(major, minor, suffix="a"):
    module = _jit_nvfp4_module(...)   # 临时把目标设为 10.0a
```

`is_arch_support_pdl()` = `major >= 9`（ROCm 恒 False），决定 PDL 是否可用。

### 10.3 HIP/ROCm 回退策略

JIT 框架在三个层次保证 ROCm 可用：

| 层次 | 机制 |
|---|---|
| 编译 flag | `_get_default_target_flags()` 返回 `-DUSE_ROCM` + FP8 类型探测（gfx942→FNUZ） |
| C++ 类型/API | `utils.cuh` 用宏别名把 `cuda*`→`hip*`，`__shfl_xor_sync`→`__shfl_xor` 等 |
| 整核回退 | CUDA 专有原语无法移植时，**用 Triton 重写**（如 `triton/hash_topk.py`），Python 侧 `is_hip_runtime()` 分流 |

#### Triton 回退示例（hash_topk）

`csrc/deepseek_v4/hash_topk.cuh` 用了 CUDA-only 原语，ROCm 上转而调 `triton/hash_topk.py` 的 `hash_topk_triton`——同一 Python 接口，按 `is_hip_runtime()` 选择后端。这种“CUDA 走 JIT C++，ROCm 走 Triton”的双后端模式在 dsv4 模块里多处出现。

### 10.4 CuTe DSL 路线

部分前沿 kernel（`cutedsl_gdn.py`、`cutedsl_kda.py`、`diffusion/cutedsl/`）用 **CuTe DSL**（`cutlass.cute`，`@cute.kernel`）编写，这是 CUTLASS 的 Python DSL，介于手写 CUDA 与纯 Triton 之间，适合需要精细控制 tile/copy/mma 但又想保持 Python 表达力的场景（如 GDN/KDA 线性注意力解码核）。

---

## 11. 依赖注册

JIT 算子有时需要外部头文件库（flashinfer、CUTLASS）。框架用**注册表 + 装饰器**模式管理 include 路径：

```python
_REGISTERED_DEPENDENCIES: Dict[str, Callable[[], List[str]]] = {}

@register_dependency("flashinfer")
def get_flashinfer_include_paths() -> List[str]:
    # 定位 flashinfer 包，返回 include/csrc/cutlass/spdlog 等头路径
    ...

@register_dependency("cutlass")
def get_cutlass_include_paths() -> List[str]:
    # 优先从 flashinfer/data/cutlass，回退 deep_gemm/include
    ...
```

使用时只需在 `load_jit(..., extra_dependencies=["cutlass"])` 声明，`load_jit` 内部解析并追加 `extra_include_paths`。未注册的依赖名会直接报错，避免拼写错误静默失败。

| 依赖 | 解析来源 | 用途 |
|---|---|---|
| `flashinfer` | flashinfer 包的 `data/{include,csrc,cutlass,spdlog}` | 注意力/norm 等参考实现与头 |
| `cutlass` | flashinfer 的 cutlass 或 `deep_gemm/include` | nvfp4/gemm 等需要 CUTLASS 模板 |

---

## 12. 目录结构全景

```
python/sglang/jit_kernel/
├── __init__.py                 # 空
├── __main__.py                 # 生成 .clangd（IDE 索引支持）
├── utils.py                    # ★ 核心：load_jit / cache_once / make_cpp_args / arch / 依赖
├── add_constant.py             # 最小教学示例
├── norm.py, rmsnorm_hf.py      # RMSNorm / QKNorm / fused_add_rmsnorm
├── activation.py, rope.py, fused_qknorm_rope.py, clamp_position.py
├── hadamard.py                 # register_custom_op + fast-hadamard-transform 依赖
├── nvfp4.py, mxfp8.py, fp8_quantize.py, per_tensor_quant_fp8.py,
│   per_token_group_quant_8bit.py, mla_kv_pack_quantize_fp8.py   # 量化族
├── awq_dequantize.py, awq_marlin_repack.py, gptq_marlin[_repack].py,
│   moe_wna16_marlin.py         # 权重量化/repack
├── moe_align.py, moe_fused_gate.py, moe_lora_align.py, grouped_topk.py
├── all_reduce.py               # 分布式
├── concat_mla.py, set_mla_kv_buffer.py, kvcache.py, hicache.py,
│   fused_store_index_cache.py, triton_store_cache.py, fixup_zero_kv.py,
│   resolve_future_token_ids.py, fused_metadata_copy.py   # KV cache / 调度辅助
├── hisparse.py                 # 稀疏 KV（MLA/DSv4 layout）
├── flash_attention.py / _v3.py / _v4.py    # 版本分发
├── cutedsl_gdn.py, cutedsl_kda.py          # CuTe DSL 线性注意力
├── ngram_corpus.py, ngram_embedding.py     # header_only=False 示例
├── timestep_embedding.py                   # diffusion
├── benchmark/                  # bench_*.py + diffusion/ + utils.py
├── csrc/                       # C++/CUDA 源
│   ├── elementwise/  (rmsnorm.cuh, fused_add_rmsnorm.cuh, qknorm*.cuh, ...)
│   ├── attention/  deepseek_v4/  dsa/  gemm/  moe/  lora/
│   ├── distributed/  diffusion/  ngram_corpus/  fast-hadamard-transform/
├── include/sgl_kernel/         # ★ 复用头库
│   ├── utils.cuh, utils.h, tensor.h, type.cuh, vec.cuh, tile.cuh
│   ├── math.cuh, warp.cuh, cta.cuh, atomic.cuh, runtime.cuh, ffi.h
│   ├── source_location.h, scalar_type.hpp
│   ├── impl/   (norm.cuh, ...)
│   ├── deepseek_v4/  distributed/
├── dsv4/                       # DeepSeek-V4 专用：attn/compress/elementwise/
│   │                           #   gemm/hisparse/moe/topk + make_name 前缀
├── diffusion/                  # group_norm_silu.py, qknorm_rope.py, cutedsl/, triton/
├── triton/                     # CUDA 不可移植时的 Triton 回退
│   ├── hash_topk.py, gdn_fused_proj.py
└── tests/                      # test_*.py + deepseek_v4/ + diffusion/
```

### `__main__.py` 与 IDE 支持

`python -m sglang.jit_kernel` 会调用 `generate_clangd()` 生成 `.clangd` 配置：注入 `-xcuda --cuda-gpu-arch=sm_XX -Wall -Wextra` + target flags + 各 `-isystem` include 路径（剥离 `--expt-relaxed-constexpr` 这类 clangd 不认的 flag），让编辑器能正确索引 `.cuh`。支持 `--dep`、`--cuda-target`、`--overwrite` 参数。

---

## 13. 实战示例

按复杂度递进，展示框架如何支撑从“几行”到“多变体调度”的算子。

### 13.1 入门：`add_constant`（无状态、单模板参数）

```python
# Python
@cache_once
def _jit_add_constant_module(constant: int) -> Module:
    args = make_cpp_args(constant)
    return load_jit("add_constant", *args,
                    cuda_files=["add_constant.cuh"],
                    cuda_wrappers=[("add_constant", f"add_constant<{args}>")])

@debug_kernel_api
def add_constant(src, constant):
    dst = torch.empty_like(src)
    _jit_add_constant_module(constant).add_constant(dst, src)
    return dst
```

```cpp
// add_constant.cuh
template <int32_t kConstant>
__global__ void add_constant_kernel(int32_t* dst, const int32_t* src, int n) { ... }

template <int32_t kConstant>
void add_constant(TensorView dst, TensorView src) {
    auto m = TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(device_);
    m.verify(dst).verify(src);
    LaunchKernel(grid, kBlockSize, device)(add_constant_kernel<kConstant>, ...);
}
```

把 `kConstant` 作为模板参数：常量被编译进 kernel，可被编译器常量折叠。

### 13.2 进阶：`rmsnorm`（多变体 dispatch + PDL + occupancy）

`norm.py` 根据 `hidden_size` 在 Python 侧选择 kernel 类，再拼进模板名：

```python
def _rmsnorm_kernel_class(hidden_size: int) -> str:
    if hidden_size in {64, 128, 256}:        return "RMSNormWarpKernel"
    if hidden_size == 512:                   return "RMSNormHalfKernel"
    if hidden_size >= 2048 and hidden_size % 512 == 0:
                                             return "RMSNormHalfKernel"
    return "RMSNormKernel"

@cache_once
def _jit_rmsnorm_module(hidden_size, dtype):
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    kernel_class = f"{_rmsnorm_kernel_class(hidden_size)}<{args}>"
    return load_jit("rmsnorm", *args, cuda_files=["elementwise/rmsnorm.cuh"],
                    cuda_wrappers=[("rmsnorm", f"{kernel_class}::run")])
```

| hidden_size | kernel 类 | 策略 |
|---|---|---|
| ∈ {64,128,256} | `RMSNormWarpKernel` | 单 warp 处理一行 |
| 512 | `RMSNormHalfKernel` | half 向量化 |
| ≥2048 且 %512==0 | `RMSNormHalfKernel` | block 协作 + 向量化 |
| 其他（如 2304） | `RMSNormKernel` | 通用 block 路径 |

C++ 侧 `::run` 静态方法完成 `TensorMatcher` 校验 + 基于 occupancy 计算 `num_blocks` + `LaunchKernel(...).enable_pdl(kUsePDL)(kernel, params)`。Blackwell 上 `rmsnorm_cta_wide`（32B 访存）替代 `rmsnorm_cta_double`（16B×2），体现架构自适应。

### 13.3 复杂依赖：`nvfp4`（CUTLASS + sm_100a + custom op）

```python
def _nvfp4_arch_env():
    # 要求 CC >= 10，使用 sm_100a 变体
    with override_jit_cuda_arch(major, minor, suffix="a"):
        ...

@register_custom_op(op_name=..., mutates_args=...)   # 注册为 PyTorch op
@debug_kernel_api
def nvfp4_xxx(...): ...
```

要点：`extra_dependencies=["cutlass"]` 引入 CUTLASS 头；`_nvfp4_cuda_flags()` 加 CUTLASS 编译宏；`prewarm_nvfp4_jit_modules`（`@torch.compiler.disable`）在服务启动时预热编译，避免首请求卡顿。

### 13.4 有状态对象：`ngram_corpus`（header_only=False）

```python
module = load_jit("ngram_corpus", cpp_files=[...5 个 .cpp...], header_only=False)
module.register_once()

@tvm_ffi.register_object("sgl.NgramCorpus")
class NgramCorpusFFI(tvm_ffi.Object):
    def __init__(self, ...):
        self.__init_handle_by_constructor__(self.__ffi_init__, ...)
```

与无状态函数不同，这里 C++ 侧导出的是一个**可持有数据的对象**，Python 通过 FFI object 协议构造并调用其方法。

### 13.5 双后端分发：`flash_attention` / `hisparse` / `dsv4`

- `flash_attention.py`：按 `ver=3/4` 分发到 `_v3.py`/`_v4.py`，对应 `flash_attn_with_kvcache` / `flash_attn_varlen_func`。
- `hisparse.py`：`@functools.cache` 缓存模块（参数含 `item_size_bytes/block_size/num_top_k/hot_buffer_size/is_mla/is_dsv4_layout`），MLA 与 DSv4 layout 各有 swap-in wrapper。
- `dsv4/`：`make_name()` 统一加前缀；hash_topk 在 ROCm 回退到 Triton；MoE 系列 kernel 用 `extra_cuda_cflags=["-use_fast_math"]`。

---

## 14. 测试、基准与 CI 集成

### 14.1 测试文件范式

每个 `tests/test_*.py` 都：

1. 顶部声明 CI 注册（AST 解析，运行时 no-op）：
   ```python
   register_cuda_ci(est_time=45, suite="base-b-kernel-unit-1-gpu-large")
   register_cuda_ci(est_time=240, suite="nightly-kernel-1-gpu", nightly=True)
   ```
2. 用 `pytest.mark.parametrize` 做参数矩阵；用 `get_ci_test_range(full, ci)` 在 CI 下裁剪规模（完整范围由 `SGLANG_JIT_KERNEL_RUN_FULL_TESTS=true` 开启）。
3. 与参考实现（通常是 flashinfer）交叉验证：`torch.testing.assert_close(out_sglang, out_ref, atol=1e-2, rtol=1e-2)`。
4. **必须有 `if __name__ == "__main__": sys.exit(pytest.main([__file__, "-v"]))`**——否则 CI runner 的 `python3 file.py` 会静默退出、假绿。

### 14.2 CI 注册与调度（`test/ci/ci_register.py`）

```python
def register_cuda_ci(est_time, suite=None, nightly=False, disabled=None,
                     *, stage=None, runner_config=None): ...   # 运行时 no-op
```

- 这些调用**不在运行时执行**，而是被 `RegistryVisitor`（`ast.NodeVisitor`）静态解析，提取 `(backend, est_time, suite, nightly, ...)`。
- `effective_suite`：若给了 `(stage, runner_config)` 则拼成 `{stage}-test-{runner_config}`，否则用 `suite`。新式 `(stage, runner_config)` 与旧式 `suite` 二选一，混用会报错。
- `collect_tests`：强制校验“有启用的注册 ⇒ 必须有 `__main__` 入口”，否则抛错（防假绿）。
- `auto_partition`：用 **LPT（最长处理时间优先）贪心**，按 `est_time` 把测试文件均衡切分到 `size` 个分片，支持 `live_est` 实测时间覆盖——这是 CI 多机并行的负载均衡核心。

| 后端 | 注册函数 |
|---|---|
| CUDA | `register_cuda_ci` |
| CPU | `register_cpu_ci` |
| AMD | `register_amd_ci` |
| NPU | `register_npu_ci` |
| XPU | `register_xpu_ci` |

### 14.3 基准（`benchmark/utils.py`）

- `DEFAULT_DTYPE = bfloat16`；`get_benchmark_range` 给出扫描区间。
- `run_benchmark`：用 `do_bench_cudagraph`（CUDA Graph 包裹，测纯 kernel 时间）。
- `run_benchmark_no_cudagraph`：用 `do_bench`（含 launch 开销）。
- 返回值单位为微秒（µs）。

---

## 15. 附录

### 15.1 添加一个新 JIT kernel 的标准流程

1. 在 `csrc/<category>/your_kernel.cuh` 写模板 kernel（host 入口用 `TensorMatcher` 校验 + `LaunchKernel` 启动）。
2. 在 `your_kernel.py` 写 `@cache_once _jit_xxx_module(...)`（`make_cpp_args` + `load_jit` + `cuda_wrappers`）和 `@debug_kernel_api` 公开函数。
3. 在 `tests/test_your_kernel.py` 加 `register_cuda_ci(...)` + 参数化测试 + flashinfer 交叉校验 + `__main__` 入口。
4. 需要 IDE 索引时 `python -m sglang.jit_kernel` 重生成 `.clangd`。
5. 可选：`benchmark/bench_your_kernel.py` 加性能基准。

> 该流程已沉淀为 `add-jit-kernel` skill，可直接调用获得分步指导。

### 15.2 常见排错

| 现象 | 可能原因 | 排查方向 |
|---|---|---|
| 首次调用卡顿 | JIT 编译开销 | 正常；可用 `prewarm_*` 预热 |
| `Dependency X is not registered` | `extra_dependencies` 写了未注册名 | 检查 `@register_dependency` |
| `unsupported hidden_size` | dispatch 不支持该尺寸 | 见 `_is_supported_rmsnorm_hidden_size` |
| CI 假绿（测试没跑） | 缺 `__main__` 入口 | `collect_tests` 会拦截；补 `sys.exit(pytest.main(...))` |
| ROCm 编译失败 | 用了 CUDA-only 原语 | 看是否需 Triton 回退（参考 hash_topk） |
| dtype 不支持 | 不在 `CPP_DTYPE_MAP` | 扩展映射或在 kernel 内 `device::cast` |

### 15.3 关键文件速查

| 文件 | 职责 |
|---|---|
| `utils.py` | `load_jit` / `cache_once` / `make_cpp_args` / arch / 依赖注册 |
| `kernel_api_logging.py` | `@debug_kernel_api` 可观测性 |
| `__main__.py` | `.clangd` 生成 |
| `include/sgl_kernel/utils.cuh` | 类型 / 宏 / PDL / `LaunchKernel` |
| `include/sgl_kernel/tensor.h` | `TensorMatcher` 校验 |
| `include/sgl_kernel/type.cuh` | dtype trait |
| `include/sgl_kernel/vec.cuh` `tile.cuh` | 向量化 / 协作访存 |
| `test/ci/ci_register.py` | CI 注册解析 / `auto_partition` |

### 15.4 设计哲学小结

- **薄 Python，厚 C++ 库**：Python 只做参数选择/缓存/校验日志，重逻辑下沉到可复用、可索引的 `.cuh`。
- **特化即缓存键**：`make_cpp_args` 的输出既是模板实参又是模块名后缀，天然区分特化。
- **校验声明化**：`TensorMatcher` 把分散的 check 收敛成一行链式表达。
- **架构自适应优先于多二进制**：用编译宏（`-DSGL_CUDA_ARCH`）+ `override_jit_cuda_arch` 在单套源码内分流 Hopper/Blackwell/ROCm。
- **不可移植则换后端**：CUDA-only 原语在 ROCm 上用 Triton 重写，保持 Python 接口不变。
- **可观测零成本**：`@debug_kernel_api` 默认 no-op，需要时一个环境变量打开全链路日志/dump。




