# SGLang 编译安装流程系统梳理

> 本文系统梳理 SGLang 从源码到可运行的完整编译/安装链路。核心结论先行：
> **SGLang 不是一个单体包，而是「三包 + 三种编译形态」的分层体系**，理解这一点是看懂整个 build 流程的钥匙。

---

## 0. 一句话全景

```
                        SGLang 可运行环境
                                │
   ┌────────────────────────────┼────────────────────────────┐
   │                            │                             │
① sglang (主包)          ② sglang-kernel (重型)        ③ 运行时 JIT
 python/pyproject.toml    sgl-kernel/pyproject.toml     python/sglang/jit_kernel
   │                            │                             │
 setuptools +              scikit-build-core +           tvm-ffi 按需
 setuptools-rust           CMake (AOT 编译 .cu)          编译 .cu
   │                            │                             │
 只编译一个极小的            产出双架构 .so                 首次调用时 ninja
 Rust _core (gRPC)          (sm90 / sm100)                编译，内容寻址缓存
 其余全是纯 Python           作为 wheel 依赖安装            ~/.cache/tvm-ffi
```

三者的编译时机、构建后端、产物完全不同，下面逐层展开。

---

## 1. 三种编译产物对照表（先看全局，再看细节）

| 维度 | ① 主包 `sglang` | ② `sglang-kernel` | ③ JIT kernel |
|------|----------------|-------------------|--------------|
| 路径 | `python/` | `sgl-kernel/` | `python/sglang/jit_kernel/` |
| 构建后端 | `setuptools.build_meta` | `scikit_build_core.build` | tvm-ffi（`apache-tvm-ffi`） |
| 编译什么 | 仅 Rust `_core`（gRPC） | 全量 CUDA/C++ 重型算子 | 轻量/实验性 `.cu` |
| 编译时机 | `pip install` 时 | 打 wheel 时（离线） | **运行时首次调用** |
| CUDA 参与 | 否（纯 Rust+Py） | 是（nvcc 全量 gencode） | 是（单 arch nvcc） |
| 产物 | `_core.*.so` + 纯 py | `sm90/`+`sm100/` 双 `.so` | 缓存目录里的 `.so` |
| 通常获取方式 | `pip install -e python` | **预编译 wheel 直接装** | 无需预装，跑时生成 |
| 版本 | 动态（setuptools-scm） | 硬编码 `0.4.4` | 跟随主包 |

关键认知：**日常 `pip install -e python` 几乎不编译 CUDA**——CUDA 重活由 ② 的预编译 wheel（`sglang-kernel==0.4.4`，见 [python/pyproject.toml:69](../../../python/pyproject.toml#L69)）承担，或由 ③ 在运行时 JIT。这也解释了上一个问题里"cu13/cu12"其实是在换 ② 的预编译轮，而非本地重编。

---

## 2. 主包 `sglang` 的安装流程（`pip install -e python`）

### 2.1 构建配置总览

[python/pyproject.toml](../../../python/pyproject.toml) 关键段：

```toml
[build-system]                                          # 行 1-3
requires = ["setuptools>=61.0", "setuptools-rust>=1.10",
            "setuptools-scm>=8.0", "wheel"]
build-backend = "setuptools.build_meta"

[[tool.setuptools-rust.ext-modules]]                    # 行 219-222
target = "sglang.srt.grpc._core"                        # 唯一被编译的原生扩展
path   = "../rust/sglang-grpc/Cargo.toml"
binding = "PyO3"

[tool.setuptools_scm]                                   # 行 212-217
root = ".."                                             # pyproject 在 python/，仓库根在上一级
version_file = "sglang/_version.py"
git_describe_command = ["python3", "python/tools/get_version_tag.py"]
fallback_version = "0.0.0.dev0"
```

### 2.2 安装时发生了什么

```
pip install -e python
   │
   ├─ 读 [build-system].requires → 装 setuptools/-rust/-scm/wheel（隔离构建环境）
   │
   ├─ setuptools-scm 算版本：调 python/tools/get_version_tag.py
   │     ├─ git describe --tags --exact-match --match 'v*'  (精确 tag)
   │     └─ 否则取 PEP440 排序最高的 tag（parse_version_tuple 修正 rc 排序 bug）
   │     └─ 都失败 → fallback 0.0.0.dev0；结果写入 sglang/_version.py
   │
   ├─ setuptools-rust 编 Rust 扩展 sglang.srt.grpc._core
   │     └─ crate = rust/sglang-grpc（crate-type=cdylib，PyO3 0.23）
   │     └─ build.rs 用 tonic_build 编 proto/sglang/runtime/v1/sglang.proto
   │     └─ 需要 Rust ≥1.85（edition 2024）
   │
   ├─ 解析 [project].dependencies（torch/transformers/flashinfer/... 及 sglang-kernel）
   │     └─ 这些多是预编译 wheel，不在本地编译
   │
   └─ editable 安装：把 python/sglang 挂进 site-packages（.pth 指向源码树）
```

要点：
- **主包安装唯一的"编译"就是那个 Rust `_core`**，耗时秒级，产出 `_core.<abi>.so`。它是 gRPC 前端的进程内原生实现。
- [python/tools/get_version_tag.py](../../../python/tools/get_version_tag.py) 的 `parse_version_tuple`（行 26）自己实现 PEP440 排序，绕开 `git describe` 把 `rc0` 排到正式版之上的老 bug。
- `package-data`（[python/pyproject.toml:183-188](../../../python/pyproject.toml#L183)）显式打包 `srt/**/*` 和 `jit_kernel/**/*`——**JIT 源码是随包分发、运行时才编译**（见 §4）。

---

## 3. `sglang-kernel` 的 AOT 编译流程（最重的一环）

这是唯一会大规模调用 nvcc 的部分，位于 [sgl-kernel/](../../../sgl-kernel/)，构建后端是 **scikit-build-core → CMake**。

### 3.1 构建后端与开发者入口

[sgl-kernel/pyproject.toml](../../../sgl-kernel/pyproject.toml)：

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "torch>=2.8.0", "wheel"]  # 行 2-6
build-backend = "scikit_build_core.build"                        # 行 7
[project] version = "0.4.4"                                       # 行 11（硬编码，非 scm）
[tool.scikit-build]
cmake.build-type = "Release"          # 行 37
wheel.py-api = "cp310"                # 行 40，限定 ABI = cp310 stable ABI，一轮多版本通用
wheel.packages = ["python/sgl_kernel"]# 行 42
```

开发者入口 [sgl-kernel/Makefile](../../../sgl-kernel/Makefile)：

| 目标 | 命令 | 用途 |
|------|------|------|
| `make install` (行 42) | `pip install -e . --no-build-isolation` | 本地开发 editable |
| `make build` (行 45) | `uv build --wheel -Cbuild-dir=build .` + `pip install dist/*whl` | 打正式 wheel |
| `make ln` (行 39) | `cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=YES` | 生成 compile_commands.json |
| `make submodule` (行 36) | `git submodule update --init --recursive` | 拉子模块 |
| `make rebuild` (行 57) | clean + submodule + build | 全量重编 |

并行度旋钮（Makefile 行 8-11）：`MAX_JOBS` / `CMAKE_BUILD_PARALLEL_LEVEL`（外层 job 数）与 CMake 变量 `SGL_KERNEL_COMPILE_THREADS`（单个 nvcc 的 `--threads`，控制多 arch PTXAS 并行）分层控制，OOM 时可调小。

### 3.2 CMake 内部编译编排

[sgl-kernel/CMakeLists.txt](../../../sgl-kernel/CMakeLists.txt)（497 行）核心动作：

```
project(sgl-kernel LANGUAGES CXX CUDA)              # 行 2
clear_cuda_arches(CMAKE_CUDA_FLAGS)                 # 行 46：抹掉 torch 塞进来的默认 -gencode
                                                    #        改由本工程完全掌控 arch
FetchContent 拉第三方（按 commit + SHA256 锁定）：
   cutlass / fmt / triton v3.6.0 / flashinfer /
   sgl-project/sgl-attn(flash-attn) / FlashMLA      # 行 50-87 + cmake/flashmla.cmake
                                                    # GITHUB_ARTIFACTORY 可换内网镜像（行 17）
组装 nvcc flags SGL_KERNEL_CUDA_FLAGS               # 行 121-157
   基线 -gencode arch=compute_90,code=sm_90         # 行 127
   + 各种 ENABLE_* 宏（BF16/FP8/NVFP4/FA3...）
同一份 ${SOURCES} 编译两次（双架构产物）：
   ├─ common_ops_sm90_build  (-use_fast_math)  → 装到 sgl_kernel/sm90   # 行 324-334,384
   └─ common_ops_sm100_build (precise math)    → 装到 sgl_kernel/sm100  # 行 338-347,385
可选目标：flash_ops(FA3) / spatial_ops(green ctx) / flashmla_ops
```

**"双架构 + 运行时选择"是 sgl-kernel 最独特的设计**：wheel 里同时带 `sm90/`（Hopper 专用 fast-math）和 `sm100/`（精确数学，兼容更广）两套 `.so`，安装时不挑，**跑起来才按 GPU 算力选**（见 §3.4）。

### 3.3 CUDA / SM 架构选择矩阵

架构完全由 `-gencode` 显式控制（**不用** `CUDA_ARCHITECTURES` 属性），依据检测到的 `CUDA_VERSION` + 处理器 + 特性开关，逻辑在 [CMakeLists.txt](../../../sgl-kernel/CMakeLists.txt)：

| 条件 | 追加的 arch flag | 行号 |
|------|-----------------|------|
| 恒定基线 | `compute_90,sm_90` | 127 |
| `ENABLE_BELOW_SM90`（x86 默认 ON / aarch64 OFF） | `sm_80`、`sm_89`（aarch64 加 `sm_87`） | 99-106, 194-205 |
| `CUDA≥12.8` 或 `SM100A` | `sm_100a`、`sm_120a` | 207-211 |
| `CUDA≥13.0` | `sm_103a` + `--compress-mode=size`（aarch64 加 `sm_110a`/`sm_121a`） | 213-223 |
| `CUDA 12.8–13.0` 且 aarch64 | `sm_101a` | 225-229 |
| `CUDA≥12.4` 且 `FA3` | `sm_90a` | 233-237 |
| `CUDA≥12.8` 或 `FP4` | `-DENABLE_NVFP4=1` | 239-243 |

FlashMLA 的 arch 另在 [cmake/flashmla.cmake](../../../sgl-kernel/cmake/flashmla.cmake)：`sm_90a`(CUDA>12.4) / `sm_100a`(CUDA>12.8) / `sm_103a`(CUDA≥13.0，含源码 patch)。

### 3.4 wheel 的 CUDA 打标与运行时分发

- **打标**：[sgl-kernel/rename_wheels.sh](../../../sgl-kernel/rename_wheels.sh) 在构建后把 `linux_x86_64` 改成 `manylinux2014_*`，并按检测到的 `/usr/local/cuda-*` 给 METADATA 版本追加 `+cu124/128/129/130` 后缀（行 9-33）。这就是为什么装 cu12.9 要手动指定 `sglang_kernel-0.4.4+cu129-...whl`。
- **运行时选架构**：[python/sgl_kernel/load_utils.py](../../../sgl-kernel/python/sgl_kernel/load_utils.py) `_load_architecture_specific_ops()`（行 48）
  - `_get_compute_capability()`（行 15）= `major*10+minor`；
  - `==90` → 加载 `sm90/common_ops.*`（行 60-62），否则 → `sm100/`（行 63-68，含无 GPU 兜底）；
  - 用 `importlib.util.spec_from_file_location` 从具体路径动态加载（行 89-98）。
  - `__init__.py:19` 调用它；另有 `_preload_cuda_library()` 预加载 `libcudart.so.{13,12}`。

### 3.5 CI / 容器化打轮

- [sgl-kernel/build.sh](../../../sgl-kernel/build.sh)：`build.sh <PY> <CUDA> [ARCH]`，在 `pytorch/manylinux2_28-builder` 容器里 `uv build --wheel` + `rename_wheels.sh`，挂载宿主 ccache 加速；ARM 用保守并行度。
- [sgl-kernel/Dockerfile](../../../sgl-kernel/Dockerfile)：三段 `deps`（装 CMake 3.31.1 / ccache / 对应 CUDA 的 torch）→ `build` → `artifact`（导出 `dist/*.whl`）。
- CPU 变体：[sgl-kernel/csrc/cpu/CMakeLists.txt](../../../sgl-kernel/csrc/cpu/CMakeLists.txt) 独立 C++17 构建，glob 全部 `*.cpp` 并按 `x86_64|aarch64|ppc64` 过滤 arch 目录。

---

## 4. 运行时 JIT kernel 编译（第三种编译形态）

轻量/实验性算子既不打进 sgl-kernel wheel，也不预编译，而是**首次调用时用 tvm-ffi + ninja 现场编译**，源码随主包分发。

引擎：[python/sglang/jit_kernel/utils.py](../../../python/sglang/jit_kernel/utils.py)

```
load_jit(name, cuda_files=[...], cuda_wrappers=[...])          # 行 206，核心入口
   │
   ├─ 源文件相对 KERNEL_PATH/csrc 解析                          # 行 264-265
   ├─ _local_jit_source_hash：顺 #include 图算内容哈希          # 行 74
   ├─ 内容寻址缓存目录 $TVM_FFI_CACHE_DIR（默认 ~/.cache/tvm-ffi）# 行 280-283
   │     └─ 命中 .so 直接复用，跳过 ninja                       # 行 284-296
   ├─ arch：get_jit_cuda_arch() 从 torch.cuda.get_device_capability
   │        设 TVM_FFI_CUDA_ARCH_LIST + -DSGL_CUDA_ARCH=        # 行 344,412,359
   ├─ 默认 flags：-std=c++20 -O3 --expt-relaxed-constexpr       # 行 392-397
   │        （另有 ROCm/HIP 行 378、MUSA 行 163 分支）
   └─ 依赖头注册 register_dependency：flashinfer/mathdx/cutlass  # 行 436-522
```

- 每个算子是薄封装，如 [jit_kernel/add_constant.py](../../../python/sglang/jit_kernel/add_constant.py) 用 `@cache_once` 保证进程内只编一次；仓库里约 40 个这类模块（rope、gptq_marlin、nvfp4、fused_qknorm_rope、flash_attention_v4…）。
- `依赖 apache-tvm-ffi==0.1.11`（[python/pyproject.toml:21](../../../python/pyproject.toml#L21)）。
- **代价**：首次调用有编译延迟；因此生产镜像常预热 JIT cache（如 flashinfer-jit-cache，见 §6）。

> 添加新 JIT/AOT kernel 前请读对应 skill：`add-jit-kernel`、`add-sgl-kernel`。

---

## 5. Rust 组件全景

| 组件 | 路径 | 构建后端 | 产物 / 何时编译 |
|------|------|----------|----------------|
| gRPC `_core` | [rust/sglang-grpc](../../../rust/sglang-grpc/Cargo.toml) | setuptools-rust（随主包） | `sglang.srt.grpc._core` cdylib，**`pip install` 时自动编** |
| model gateway（路由器） | [sgl-model-gateway](../../../sgl-model-gateway/Cargo.toml) | cargo / maturin | 二进制 `sgl-model-gateway` + py 包 `sglang-router` |
| 实验 router | [experimental/sgl-router](../../../experimental/sgl-router/Cargo.toml) | 纯 `cargo build` | 二进制，**无 pyproject/maturin** |

- gateway 的 pip 包 `sglang-router` 在 [sgl-model-gateway/bindings/python](../../../sgl-model-gateway/bindings/python/)，build-backend = **maturin**，模块 `sglang_router.sglang_router_rs`；开发用 `maturin develop`，发布用 `maturin build --release --features vendored-openssl`（gateway Makefile 行 98/102）。
- gateway 二进制本身：`cargo build --release`（[sgl-model-gateway/Makefile](../../../sgl-model-gateway/Makefile) 行 34），release profile `opt-level=z`+`lto=fat` 压体积。
- `rust/sglang-grpc/build.rs` 用 tonic_build 编 `proto/sglang/runtime/v1/sglang.proto`（server-only）。

---

## 6. Docker 端到端编排（把三包拼起来）

[docker/Dockerfile](../../../docker/Dockerfile) 是官方"从零到可运行"的权威流程，用 BuildKit 多阶段并行：

```
base (nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04, 默认 13.0.1)
  │  装系统依赖 + Python3.12 + RDMA/IB/MPI/gRPC/protobuf 等
  │
  ├─ torch_deps ──────────┬─ deepep_builder   (需 torch，编 DeepEP wheel，按 CUDA 选 arch)
  │  ① 装 sgl-kernel 预编译轮 │
  │    (按 CUDA 选 +cu129/cu130，行 195-215) └─ flashinfer_cache (可选预热 flashinfer-jit-cache)
  │  ② pip install .[BUILD_TYPE]（含主包依赖+主包 editable 的依赖解析）
  │    CUDA12 分支：卸掉所有 *-cu13 包→强装 cu129 torch/deep-gemm（行 243-249）
  │
  ├─ devtools_builder (独立，下 clang/cmake/just 等工具)
  ├─ gateway_builder  (独立，maturin 编 sglang-router + cargo 编 gateway 二进制)
  │
  ▼
framework (汇总各 builder 产物)
  │  装 Mooncake(按 CUDA major 选 cuda13 变体)/MSCCL++/nixl-cu12|13/cuda-python
  │
  ▼
framework_final
  └─ pip install --no-deps -e "python[BUILD_TYPE]"   ← 主包 editable 安装（行 668）
     + kernels lock/download（下 sgl-flash-attn3 cubins，失败退化到运行时 JIT）
  ▼
runtime (瘦身生产镜像，devel base 保留 nvcc 供 JIT)
```

关键分支点（也是上一问 cu12.9 的来源）：
- **sgl-kernel 选轮**（行 195-215）：`case $CUDA_VERSION` → 12.8/12.9 用 GitHub release 的 `+cu129` 轮，13.0 用 PyPI 的 cu13 轮，均 `--no-deps`。
- **CUDA12 清洗**（行 243-249）：`.[all]` 装完后 `awk '/-cu13/'` 卸掉被 `[cu13]` extra 拖进来的 nvidia 库，再 `--force-reinstall` cu129 的 torch/torchvision/torchaudio + cu129 sgl-deep-gemm。
- **per-CUDA-major 收尾**（行 595-601）：nixl-cu12/cu13、`cuda-python==12.9`/`13.2.0` 分别安装。

---

## 7. 开发者常用命令速查

| 场景 | 命令 |
|------|------|
| 装主包（editable，含全部 extras） | `pip install -e "python[all]"` |
| 只装主包运行时 | `pip install -e python` |
| 本地重编 sgl-kernel（editable） | `cd sgl-kernel && make install` |
| 打 sgl-kernel wheel | `cd sgl-kernel && make build`（可加 `MAX_JOBS=8`） |
| 限制 nvcc 并行防 OOM | `make build CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"` |
| 换内网 GitHub 镜像 | `make build GITHUB_ARTIFACTORY=<mirror>` |
| 编 router（py 包，开发） | `cd sgl-model-gateway/bindings/python && maturin develop` |
| 编 gateway 二进制 | `cd sgl-model-gateway && make build` |
| 生成 clangd 编译库（IDE） | `cd sgl-kernel && make ln` |
| 换 CUDA 12.9 装（非 docker） | 见 `docs/pjc_1/`（上一问）或 Dockerfile 行 195-249 recipe |

---

## 8. 常见坑

1. **"为什么装成 cu13"**：主包 pyproject 把 `flashinfer[cu13]`/`cutlass-dsl[cu13]`/`cuda-python>=13.0`/`torch==2.11.0`（PyPI 默认 cu13）钉死；切 cu12 需换 sgl-kernel 的 `+cu129` 轮 + 清 `*-cu13` + 强装 cu129 torch。
2. **首次推理卡顿**：JIT kernel（§4）/ DeepGEMM / FlashInfer 在运行时编译，需本地 `nvcc`（`CUDA_HOME` 指向匹配的 toolkit），生产建议预热 cache。
3. **sgl-kernel 版本硬编码**：`0.4.4` 写死在 pyproject/version.py，升级用 `make update <ver>` 一次改全（Makefile 行 70-90）。
4. **架构不匹配**：wheel 双架构 `sm90/sm100` 靠运行时算力选；若 GPU 算力既非 90 也无对应 gencode，会落到 `sm100` 精确数学分支。
5. **Rust 版本**：主包的 `_core` 需 Rust ≥1.85（edition 2024），CI 用 rustup 装。

---

## 9. 文件行号索引

| 主题 | 文件:行 |
|------|---------|
| 主包 build-system / rust ext / scm | [python/pyproject.toml:1-3,219-222,212-217](../../../python/pyproject.toml#L1) |
| 版本解析 | [python/tools/get_version_tag.py:26,128](../../../python/tools/get_version_tag.py#L26) |
| sgl-kernel build-backend | [sgl-kernel/pyproject.toml:7,40,42](../../../sgl-kernel/pyproject.toml#L7) |
| sgl-kernel Makefile 入口 | [sgl-kernel/Makefile:42,45](../../../sgl-kernel/Makefile#L42) |
| CMake 双架构 / arch 矩阵 | [sgl-kernel/CMakeLists.txt:46,127,324-347](../../../sgl-kernel/CMakeLists.txt#L127) |
| FlashMLA arch | [sgl-kernel/cmake/flashmla.cmake:32-96](../../../sgl-kernel/cmake/flashmla.cmake#L32) |
| wheel 打标 | [sgl-kernel/rename_wheels.sh:9-33](../../../sgl-kernel/rename_wheels.sh#L9) |
| 运行时选架构 | [sgl-kernel/python/sgl_kernel/load_utils.py:15,48,60-68](../../../sgl-kernel/python/sgl_kernel/load_utils.py#L48) |
| JIT 引擎 | [python/sglang/jit_kernel/utils.py:206,74,280-296](../../../python/sglang/jit_kernel/utils.py#L206) |
| gRPC crate | [rust/sglang-grpc/Cargo.toml:8-16](../../../rust/sglang-grpc/Cargo.toml#L8) |
| router py 包 | [sgl-model-gateway/bindings/python/pyproject.toml](../../../sgl-model-gateway/bindings/python/pyproject.toml) |
| Docker 编排 | [docker/Dockerfile:176,195-249,436,668](../../../docker/Dockerfile#L195) |

---

## 附录 A：`python/pyproject.toml` 逐段详解

主包 [python/pyproject.toml](../../../python/pyproject.toml) 是整个安装流程的"总装配单"。它同时扮演四个角色：**构建定义**（怎么编）、**依赖清单**（装什么）、**打包规则**（打进哪些文件）、**入口注册**（暴露哪些命令）。下面按功能分组解释各段落。

### A.1 段落地图（全文只有这些块）

| 段落 | 行 | 作用 |
|------|----|------|
| `[build-system]` | 1-3 | 声明用什么构建后端和构建期依赖 |
| `[project]` | 5-15 | 包名、动态版本、Python 下限、license、分类器 |
| `[project].dependencies` | 18-89 | **运行时依赖清单**（核心） |
| `[[tool.uv.index]]` | 91-94 | uv 的默认包索引（PyPI） |
| `[project.optional-dependencies]` | 96-169 | 可选依赖分组（extras） |
| `[tool.uv.extra-build-dependencies]` | 171-173 | uv 给 st-attn/vsa 补构建期依赖 |
| `[project.urls]` | 175-177 | 主页/issue 链接（元数据） |
| `[project.scripts]` | 179-181 | 控制台命令入口 |
| `[tool.setuptools.package-data]` | 183-188 | 非 .py 文件的打包规则 |
| `[tool.setuptools.packages.find]` | 190-199 | 包发现的排除项 |
| `[tool.wheel]` | 201-210 | wheel 打包排除项 |
| `[tool.setuptools_scm]` | 212-217 | 动态版本（见正文 §2.2 / 附录无重复） |
| `[[tool.setuptools-rust.ext-modules]]` | 219-222 | Rust `_core` 扩展（见正文 §2） |
| `[tool.kernels.dependencies]` | 224-225 | kernels 社区 cubin 依赖锁定 |

### A.2 构建与元数据段

```toml
[build-system]                                   # 行 1-3
requires = ["setuptools>=61.0", "setuptools-rust>=1.10",
            "setuptools-scm>=8.0", "wheel"]
build-backend = "setuptools.build_meta"
```
- **构建期**（非运行期）安装这四个包到隔离环境：`setuptools`（后端）、`setuptools-rust`（编 Rust `_core`）、`setuptools-scm`（算版本）、`wheel`（产轮）。
- `build-backend` 指定 PEP 517 后端为 setuptools。

```toml
[project]                                        # 行 5-15
name = "sglang"
dynamic = ["version"]     # 版本不写死，交给 setuptools-scm（见正文 §2.2）
requires-python = ">=3.10"
```
- `dynamic = ["version"]` 是把版本"外包"给 scm 的关键声明。
- `requires-python = ">=3.10"` 是安装期硬门槛。

<!-- APPEND-A-1 -->

### A.3 运行时依赖清单（行 18-89）按功能分组

这是全文件最长的一段。60+ 个依赖按用途归类如下（注释了带 CUDA 版本约束的项，正是上一问 cu13/cu12 的根因）：

| 功能域 | 代表依赖 | 说明 / CUDA 相关 |
|--------|---------|-----------------|
| **深度学习框架** | `torch==2.11.0`、`torchvision`、`torchaudio==2.11.0`、`torchao==0.17.0` | torch 2.11 的 PyPI 默认轮是 **cu13**；换 cu12 需从 `download.pytorch.org/whl/cu129` 强装 |
| **CUDA kernel 栈** | `sglang-kernel==0.4.4`、`sgl-deep-gemm==0.1.4`、`flash-attn-4==4.0.0b15`、`quack-kernels`、`tilelang==0.1.11`、`tokenspeed_mla==0.1.7` | 重型算子预编译轮（见正文 §3） |
| **CUDA 运行时/绑定** | `cuda-python>=13.0`、`nvidia-mathdx==25.6.0`、`apache-tvm-ffi==0.1.11` | `cuda-python>=13.0` 直接钉 13；`apache-tvm-ffi` 是 JIT 引擎（正文 §4） |
| **FlashInfer** | `flashinfer_python[cu13]==0.6.12`、`flashinfer_cubin==0.6.12` | `[cu13]` extra 拖入一堆 `nvidia-*-cu13` 库 |
| **CUTLASS DSL** | `nvidia-cutlass-dsl[cu13]==4.5.2` | 同上，`[cu13]` extra 是 cu 版本来源 |
| **HTTP 服务/异步** | `fastapi`、`uvicorn`、`uvloop`、`aiohttp`、`python-multipart`、`watchfiles`、`pyzmq>=25.1.2` | ZMQ 是进程间通信主干 |
| **序列化/数据结构** | `msgspec`、`orjson`、`pydantic`、`pybase64`、`zstandard`、`numpy`、`scipy` | `msgspec` 是本项目规定的数据容器（见 rule `no-dataclasses`） |
| **约束解码/语法** | `xgrammar==0.2.1`、`outlines==0.1.11`、`llguidance>=0.7.11,<0.8.0`、`interegular` | 结构化输出后端 |
| **分词/词表** | `transformers==5.12.1`、`tiktoken`、`sentencepiece`、`blobfile==3.0.0`、`gguf`、`datasets` | transformers 强钉 5.12.1 |
| **多模态** | `pillow`、`mistral_common>=1.11.5`、`soundfile==0.13.1`、`timm==1.0.16`、`torchcodec==0.11.1`、`av`、`decord2` | `av`/`decord2`/`torchcodec` 带平台条件标记（aarch64 差异） |
| **模型来源/客户端** | `modelscope`、`openai==2.6.1`、`openai-harmony==0.0.4`、`anthropic>=0.20.0` | |
| **可观测/运维** | `prometheus-client>=0.20.0`、`nvidia-ml-py`、`py-spy`、`psutil`、`setproctitle` | |
| **其它基础** | `einops`、`ninja`、`packaging`、`tqdm`、`requests`、`build`、`distro`、`easydict`、`kernels>=0.14.1,<0.15`、`smg-grpc-servicer` | `ninja` 供 JIT 编译；`kernels` 管 cubin 下载 |

依赖注释里明确写了三处约束（**改动前必读**）：
- `flashinfer_python[cu13]==0.6.12`（[行 35](../../../python/pyproject.toml#L35)）注释："keep it aligned with jit-cache version in Dockerfile"——必须和 Dockerfile 的 flashinfer-jit-cache 版本对齐。
- `easydict`（[行 30](../../../python/pyproject.toml#L30)）：被 trust_remote_code 模型（如 DeepSeek-OCR）依赖。
- `torchcodec==0.11.1`（[行 80](../../../python/pyproject.toml#L80)）：与 torch 2.11.x 绑定，0.10 ABI 不兼容；Linux ARM 上不可用。

> 硬约定：注释"keep dependency lists sorted alphabetically by package name"（[行 17](../../../python/pyproject.toml#L17)）——**依赖必须按包名字母序**，加依赖不能随手追加到末尾。

### A.4 可选依赖分组 extras（行 96-169）

`pip install "sglang[<extra>]"` 按需拉额外依赖。分组含义：

| extra | 行 | 内容 / 用途 |
|-------|----|-----------|
| `checkpoint-engine` | 97 | `checkpoint-engine`，权重热更新 |
| `runai` | 98 | runai model streamer（S3/GCS/Azure 流式加载权重） |
| `diffusion` | 99-119 | 扩散模型全家桶：diffusers、opencv、imageio、st_attn、vsa 等（含平台标记） |
| `ray` | 121-123 | `ray[default]`，分布式编排 |
| `tracing` | 125-130 | OpenTelemetry 全套，分布式追踪 |
| `http2` | 132-134 | `granian`，HTTP/2 服务器 |
| `fastokens` | 136-138 | `fastokens`，快速分词 |
| `test` | 140-161 | 测试栈：pytest、lm-eval、pandas、peft、sentence_transformers 等；**内嵌 `sglang[fastokens]`** |
| `dev` | 163 | = `sglang[test]`（别名） |
| `all` | 165-169 | = diffusion + http2 + tracing 的合集（**注意不含 test/ray**） |

- 组合可嵌套：`test` 里写了 `sglang[fastokens]`，`dev` 写了 `sglang[test]`，`all` 聚合三个功能 extra。
- **`all` 不等于"全部"**：它只含 diffusion/http2/tracing，不含 test/ray/runai/checkpoint-engine。Docker 用 `BUILD_TYPE=all`（正文 §6）即取此组。

### A.5 打包与入口段

```toml
[project.scripts]                    # 行 179-181
sglang = "sglang.cli.main:main"              # → 装出 `sglang` 命令
killall_sglang = "sglang.cli.killall:main"   # → 装出 `killall_sglang` 命令
```
安装后这两个可执行入口进 `$PATH`，对应 `python/sglang/cli/` 下的函数。

```toml
[tool.setuptools.package-data]       # 行 183-188
"sglang" = ["srt/**/*", "jit_kernel/**/*", "multimodal_gen/apps/realtime_webui/**/*"]
```
- **决定哪些非 `.py` 文件被打进包**。`jit_kernel/**/*` 把 `.cu/.cuh/.h` 源码随包分发——这正是运行时 JIT（正文 §4）能编译的前提。

```toml
[tool.setuptools.packages.find] / [tool.wheel]   # 行 190-210
exclude = ["assets*","benchmark*","docs*","dist*","playground*","scripts*","tests*"]
```
- 包发现和 wheel 打包时**排除**这些目录，避免把测试/文档/基准塞进发行包。

### A.6 uv 与 kernels 相关段

```toml
[[tool.uv.index]]                    # 行 91-94  用 uv 时的默认索引 = PyPI
[tool.uv.extra-build-dependencies]   # 行 171-173 给 st-attn/vsa 补 setuptools+torch 构建期依赖
[tool.kernels.dependencies]          # 行 224-225 锁定 kernels-community/sgl-flash-attn3 = 1
```
- `[tool.uv.*]` 只在用 `uv` 安装时生效（对 pip 无影响）。
- `[tool.kernels.dependencies]` 供 `kernels lock` / `kernels download`（Docker 行 669-694）拉取预编译 cubin；下不到时退化为运行时 JIT。

### A.7 一句话总结

`python/pyproject.toml` 用**一个文件**同时定义了：编译方式（setuptools + rust + scm）、装什么（60+ 运行时依赖 + 10 个 extras）、打进哪些文件（srt/jit_kernel 源码进包、tests/docs 排除）、暴露什么命令（sglang / killall_sglang）。它是主包一切安装行为的单一真相源，也是 cu13/cu12、依赖版本冲突等问题的第一排查点。

---

## 附录 B：`sgl-kernel/pyproject.toml` 逐段详解

与主包不同，[sgl-kernel/pyproject.toml](../../../sgl-kernel/pyproject.toml) 非常短（43 行）——因为**真正的编译逻辑全在 CMakeLists.txt（正文 §3）**，这个文件只负责"告诉打包器：用 CMake 后端、编哪个包、打成什么 ABI"。

### B.1 逐段解释

```toml
[build-system]                          # 行 1-7
requires = ["scikit-build-core>=0.10", "torch>=2.8.0", "wheel"]
build-backend = "scikit_build_core.build"
```
- **后端是 scikit-build-core**（不是主包的 setuptools）——它是"pip ↔ CMake"的桥：`pip install` 时自动跑 CMake 配置/编译/安装，再把产物收进 wheel。
- 构建期就需要 `torch>=2.8.0`：因为 CMakeLists 要读 torch 的头文件、库路径和 C++ ABI 标志（`_GLIBCXX_USE_CXX11_ABI`，正文 §3.2）。

```toml
[project]                               # 行 9-24
name = "sglang-kernel"
version = "0.4.4"                       # 硬编码，非动态
requires-python = ">=3.10"
classifiers = [..., "Environment :: GPU :: NVIDIA CUDA"]
dependencies = []                       # 关键：无运行时依赖
```
- **`version = "0.4.4"` 写死**（对比主包的 setuptools-scm 动态版本）。升级靠 `make update <ver>` 或 `scripts/release/bump_kernel_version.py` 批量改所有变体文件。
- **`dependencies = []` 是有意为之**：sgl-kernel 不声明对 torch 的运行时依赖，避免 pip 在装它时误拉一个 CUDA 不匹配的 torch。torch 由主包统一管理（正文 §6 Docker 用 `--no-deps` 装 sgl-kernel 正是这个原因）。

```toml
[tool.scikit-build]                     # 行 36-42
cmake.build-type = "Release"
minimum-version = "build-system.requires"
wheel.py-api = "cp310"                  # 稳定 ABI (abi3)
wheel.license-files = []
wheel.packages = ["python/sgl_kernel"]
```
- `cmake.build-type = "Release"`：CMake 用 Release 优化编译。
- **`wheel.py-api = "cp310"` 是最关键的一行**：产出 **abi3 稳定 ABI** 轮——一个 `cp310` 轮能在 3.10/3.11/3.12/… 所有更高 Python 上通用，不必为每个小版本各编一次。这配合 CMakeLists 的双架构 `sm90/sm100`，让一个 wheel 覆盖尽可能多的环境。
- `wheel.packages = ["python/sgl_kernel"]`：Python 侧包目录，CMake 编出的 `.so` 会被装进这里的 `sm90/` `sm100/` 子目录。
- 没有 `cmake.source-dir` → CMake 默认用仓库根的 `CMakeLists.txt`（GPU 全量算子）。

<!-- APPEND-B-1 -->

### B.2 四个平台变体（同目录下的兄弟文件）

sgl-kernel 为不同后端硬件准备了 **4 个 pyproject 变体**，构建时**把对应变体覆盖成 `pyproject.toml` 再编**（如 `cp pyproject_cpu.toml pyproject.toml`）：

| 文件 | 目标平台 | 构建后端 | torch | 包名 | 编译入口 |
|------|---------|---------|-------|------|---------|
| `pyproject.toml` | NVIDIA CUDA | scikit-build-core | `>=2.8.0` | `sglang-kernel` | 根 `CMakeLists.txt` |
| [`pyproject_cpu.toml`](../../../sgl-kernel/pyproject_cpu.toml) | CPU（x86/ARM） | scikit-build-core | **`==2.12.0`** | **`sglang-kernel-cpu`** | `csrc/cpu`（`cmake.source-dir`） |
| [`pyproject_rocm.toml`](../../../sgl-kernel/pyproject_rocm.toml) | AMD ROCm | **setuptools** | `>=2.8.0` | `sglang-kernel` | `setup_rocm.py` |
| [`pyproject_musa.toml`](../../../sgl-kernel/pyproject_musa.toml) | 摩尔线程 MUSA | **setuptools** | 不限 + `torchada>=0.1.68` | `sglang-kernel` | `setup_musa.py` |

关键差异解读：
- **后端分两派**：CUDA/CPU 用 scikit-build-core（CMake 驱动）；ROCm/MUSA 用 setuptools（各自 `setup_rocm.py` / `setup_musa.py` 驱动，因为它们的编译流程更定制化）。
- **CPU 变体最特殊**：包名改成 `sglang-kernel-cpu`（独立 PyPI 包）、torch 精确钉 `2.12.0`（比 CUDA 的 `>=2.8.0` 新）、且 `cmake.source-dir = "csrc/cpu"` 指向专门的 CPU CMakeLists（正文 §3.5），也没有 abi3 cp310 标志。
- **切换机制**：见 [docker/xeon.Dockerfile:41](../../../docker/xeon.Dockerfile#L41)（CPU）、[docker/rocm.Dockerfile:297](../../../docker/rocm.Dockerfile#L297)（ROCm）、[release-whl-kernel.yml:455](../../../.github/workflows/release-whl-kernel.yml#L455)（MUSA），都是先 `cp/mv` 变体覆盖 `pyproject.toml`。
- **版本同步**：[scripts/release/bump_kernel_version.py:23-25](../../../scripts/release/bump_kernel_version.py#L23) 会把新版本号同时写进全部 4 个变体，防止漂移（`make update` 只覆盖前 3 个，MUSA 由 release 脚本补）。

### B.3 主包 vs sgl-kernel 的 pyproject 对比

| 维度 | 主包 `python/pyproject.toml` | `sgl-kernel/pyproject.toml` |
|------|------------------------------|------------------------------|
| 后端 | setuptools + setuptools-rust | scikit-build-core（CMake） |
| 版本 | 动态（setuptools-scm，git tag） | 硬编码 `0.4.4` |
| 运行时依赖 | 60+ 个 | **空**（`dependencies = []`） |
| 编译产物 | 极小 Rust `_core` | 全量 CUDA `.so`（双架构） |
| ABI | 跟随 CPython | abi3 `cp310`（跨小版本通用） |
| 文件长度 | 226 行 | 43 行 |
| 平台变体 | 无（靠依赖 marker 区分） | 4 个独立变体文件 |

### B.4 一句话总结

`sgl-kernel/pyproject.toml` 只是 CMake 编译的"薄封装壳"：声明用 scikit-build-core 桥接 CMake、产 abi3 通用轮、不带运行时依赖（torch 交给主包）。它本身不含任何编译细节——那些全在 `CMakeLists.txt`（正文 §3）。真正让它"能跑在不同硬件上"的，是同目录的 4 个平台变体 + 构建时覆盖 `pyproject.toml` 的切换机制。
