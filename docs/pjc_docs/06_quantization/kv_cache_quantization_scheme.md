# SGLang KV Cache 量化方案系统梳理

> 面向 `--kv-cache-dtype` 全体量化路径：FP8（e4m3/e5m2）、MXFP8、FP4（nvfp4 / fp4_mx_block16）。
> 聚焦 `python/sglang/srt/mem_cache/memory_pool.py` 与 `layers/quantization/` 两套代码。
>
> 权重/激活量化以及整体框架见总纲 [`sglang_quantization_overview.md`](sglang_quantization_overview.md)（本文对应其「方向⑥」）。
>
> 校准时间：2026-09，基线 commit `33ed29a0ee`。所有行号均为该基线实测值。

---

## 0. 一句话总览

KV Cache 量化和权重量化**完全是两套独立机制**。权重量化在加载期做一次；KV 量化发生在每一步 forward 的写入/读取路径上，必须和 memory pool + attention backend 深度耦合。

它沿 **三条正交轴** 展开：

| 轴 | 取值 | 决定什么 | 代码入口 |
|----|------|----------|----------|
| **存储 dtype** | bf16 / fp16 / fp8_e4m3 / fp8_e5m2 / mxfp8 / nvfp4 / fp4_mx_block16 | KV 在显存里怎么摆 | `configure_kv_cache_dtype`（`mem_cache/kv_cache_dtype.py:22`） |
| **量化方法（method）** | `unquantized` / `nvfp4` / `fp4_mx_block16` | 是否需要打包 + per-block scale + workspace | `KVCacheQuantMethodBase`（`fp4_kv_cache_quant_method.py:110`） |
| **注意力访问方式（access）** | `plain` / `dequant_workspace` / `native_fp4` | 每个 backend 每个阶段怎么读 | `KV_CACHE_ATTENTION_ACCESS_REGISTRY`（`:777`） |

**核心分野：有三条实现路径，不是两条。**

| 路径 | 池类 | 特征 |
|------|------|------|
| **A. FP8（e4m3/e5m2）** | 复用 `MHATokenToKVPool`（`memory_pool.py:1808`） | `quant_method` 仍是 `unquantized`；`store_dtype=uint8`；**per-tensor 标量 scale**（`k_scale`/`v_scale` 从 ckpt 加载）；写时 `div_(scale)`、读时把 scale 交给 attention backend descale |
| **B. MXFP8** | **独立池** `MHATokenToKVPoolMXFP8`（`:3260`） | `store_dtype = torch.float8_e4m3fn`（`:3302`，不是 uint8！）+ **每 32 元素一个 UE8M0 block scale buffer**（`torch.float8_e8m0fnu`，`:3342-3353`） |
| **C. FP4（nvfp4 / fp4_mx_block16）** | MHA: `MHATokenToKVPoolFP4`（`:3036`，**死代码**）；MLA: `MLATokenToKVPoolFP4`（`:4256`） | 真正的策略模式：`quant_method` 是 `NVFP4KVCacheMethod` / `FP4MXBlock16KVCacheMethod`；KV 打包成半字节 + per-block scale buffer，可能还要 dequant workspace |

```
--kv-cache-dtype
      │
      ├── auto / bf16 / fp16 ────────────► 原生 dtype，store_dtype==dtype，无量化
      │
      ├── fp8_e4m3 / fp8_e5m2 ───────────► MHATokenToKVPool，store_dtype=uint8
      │                                     method=unquantized
      │                                     per-tensor k_scale/v_scale（标量）
      │                                     写: div_(scale) 读: backend descale
      │
      ├── mxfp8 ─────────────────────────► MHATokenToKVPoolMXFP8（独立池）
      │                                     store_dtype=float8_e4m3fn
      │                                     + per-32 UE8M0 block scale buffer
      │
      └── nvfp4 / fp4_mx_block16 ────────► method=NVFP4 / FP4MXBlock16（策略对象）
                                            打包 fp4 + per-block scale buffer
                                            (+ 可选 dequant workspace)
```

---

## 1. 第一层：`--kv-cache-dtype` 如何解析成 torch.dtype

参数定义 `server_args.py:604-615`，choices = `auto / fp8_e5m2 / fp8_e4m3 / mxfp8 / bf16 / bfloat16 / nvfp4 / fp4_mx_block16 / fp4_e2m1(已废弃)`，默认 `auto`。

解析在 `mem_cache/kv_cache_dtype.py:22 configure_kv_cache_dtype`：

| 入参值 | 得到的 kv_cache_dtype | 备注 |
|--------|----------------------|------|
| `auto` | 若 ckpt 的 `quant_config.kv_cache_quant_algo == "FP8"` → `fp8_e4m3`（HIP 用原生 `fp8_dtype`）；否则 = 模型 dtype | 自动跟随 ckpt 标定 |
| `fp8_e5m2` | `torch.float8_e5m2`（HIP → 原生 `fp8_dtype`） | |
| `fp8_e4m3` | `torch.float8_e4m3fn`（HIP → 原生 `fp8_dtype`） | |
| `mxfp8` | `torch.float8_e4m3fn` | 仅 FA4，SM100+ |
| `bf16/bfloat16` | `torch.bfloat16` | |
| `nvfp4` / `fp4_mx_block16` | `torch.float4_e2m1fn_x2` | 需 PyTorch 2.8.0+ / CUDA 12.8+ |
| `fp4_e2m1` | **抛错**（`:62-64`），已弃用 → 改用 `fp4_mx_block16` | |

返回 `(resolved_kv_cache_dtype: Optional[str], kv_cache_dtype: torch.dtype)`。第一个是给 auto-FP8 场景回填的字符串标签。

**特例（DFLASH）**：`kv_cache_dtype.py:83-92` —— 投机 draft worker 用 `fa4` draft attention 时，fa4 要求 `K.dtype == Q.dtype`，读不了 target 的 fp8 KV，于是把 draft 的 KV dtype 强制打回 `model_dtype`（入参 `is_dflash` 在 `:28`）。

---

## 2. `store_dtype` 技巧（内存池最底层）

`memory_pool.py:1698-1702`：

```python
if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e4m3fnuz):
    # index_put 对 float8 未实现，所以按 uint8 存
    self.store_dtype = torch.uint8
else:
    self.store_dtype = dtype
```

**为什么**：PyTorch 的 `Tensor.index_put`（散点写入 KV slot 必用）对 `float8_e5m2` 没有实现，直接用会报错。解决办法是**按 uint8 存储、bit 不变**，读的时候再 `.view(self.dtype)` 变回 fp8。这就是贯穿 MHA/MLA 池的两条线：

- 写：`cache_k = cache_k.view(self.store_dtype)`（`memory_pool.py:2380` `set_kv_buffer` 内）
- 读：`self.k_buffer[...].view(self.dtype)`（`:2325` `_get_key_buffer`）

`dtype`（逻辑/计算 dtype，如 fp8_e4m3）与 `store_dtype`（物理存储 dtype，uint8）分离是理解一切量化池的地基。FP4 池也复用这一套：`store_dtype = uint8`，逻辑 `dtype = float4_e2m1fn_x2`。

> **例外**：MXFP8 池**不用** uint8，它把 `store_dtype` 直接设成 `torch.float8_e4m3fn`（`:3302`）。见 §4。

---

## 3. FP8 方案（per-tensor 标量 scale）

### 3.1 scale 从哪来 —— `BaseKVCacheMethod` 及其 4 个子类

`layers/quantization/kv_cache.py:18`。这是挂在**注意力层**上的 quant method（**不是** KV 池的 method），职责：给 `RadixAttention` 层加两个可加载参数 `k_scale` / `v_scale`。

- `create_weights`（`:32`）：初始化 `layer.k_scale = layer.v_scale = -1.0`（无效哨兵值），标 `_skip_weight_check`。若 ckpt 里有 scale，加载时覆盖。
- `process_weights_after_loading`（`:51`）三分支：
  1. `k_scale>0 且 v_scale>0` → 分别用 ckpt 里的 k/v scale；
  2. 两者都 `<0`（没标定）→ 默认 `1.0`；
  3. 只有单个 `kv_scale` → 复制给 k 和 v。
  - HIP（fnuz 格式）额外 `*2`（`:57`、`:72`；e4m3fnuz 与 e4m3fn 指数偏置差一位）。
- 最终写回 `layer.k_scale_float` / `layer.v_scale_float`（Python float，供 backend 用）。
- **强约束**：只支持 **per-tensor（标量）** scale，非标量直接报错（`:76`）。

**4 个具体子类**（按 checkpoint 家族分，各家族的 `get_quant_method` 命中 `RadixAttention` 时返回对应实例）：

| 类 | 位置 | 家族 |
|----|------|------|
| `Fp8KVCacheMethod` | `fp8.py:2760` | 原生 fp8 / mxfp8 |
| `ModelOptFp8KVCacheMethod` | `modelopt_quant.py:675` | ModelOpt |
| `QuarkKVCacheMethod` | `quark/quark.py:1071` | Quark（AMD） |
| `CompressedTensorsKVCacheMethod` | `compressed_tensors.py:1164` | compressed-tensors |

挂载点是 `layers/radix_attention.py:148`，紧接着 `:149-150` 立刻调 `create_weights(self)`。`RadixAttention.__init__` 默认 `k_scale = v_scale = None`；只有 `quant_config.get_quant_method()` 返回了 method（即 ckpt 声明了 FP8 KV）才会被填充。

### 3.2 旁路：从独立 JSON 文件读 KV scale

除了 checkpoint 内嵌，还有一条老路径：

```
--quantization-param-path            server_args.py:579
   → load_kv_cache_scales()          model_loader/load_model_utils.py:112
       （只在 kv_cache_dtype == "fp8_e4m3" 时生效，:116）
   → kv_cache_scales_loader()        model_loader/weight_utils.py:1840
```

这条路径读一个外部 JSON（通常由 AMD quantizer 产出），把 per-layer 的 k/v scale 塞进模型。**它和 `BaseKVCacheMethod` 是互补的两个来源**，不要混淆。

### 3.3 写入路径 —— `MHATokenToKVPool.set_kv_buffer`

`memory_pool.py:2380`。FP8 走的是"非量化"分支（因为 `is_quantized_kv_cache` 仅对 FP4 method 为真）：

```python
if cache_k.dtype != self.dtype:        # 计算出的 KV 是 bf16，需要转 fp8
    if k_scale is not None:
        cache_k.div_(k_scale)          # 先按 per-tensor scale 缩放
    if v_scale is not None:
        cache_v.div_(v_scale)
    cache_k = cache_k.to(self.dtype)   # 转成 fp8
if self.store_dtype != self.dtype:
    cache_k = cache_k.view(self.store_dtype)  # fp8 bit → uint8 存储
```

`is_quantized_kv_cache`（`:1981`）= `not isinstance(quant_method, UnquantizedKVCacheMethod)`。**FP8 的 quant_method 仍是 `UnquantizedKVCacheMethod`**，所以 FP8 走的就是上面这段"普通"写入，量化只是一次 `div_ + to(fp8) + view(uint8)`。

### 3.4 读取路径 —— backend 拿 scale 做 descale

`get_key_buffer`（`:2341`，内部 `_get_key_buffer` `:2325`；value 侧 `_get_value_buffer` `:2349`）返回 `k_buffer.view(self.dtype)`（fp8 视图），真正的反量化在 attention backend。例如 FlashAttention（`flashattention_backend.py:1228-1231`）：

```python
if layer.k_scale is not None:
    fa_k_descale = layer.k_scale.expand(descale_shape)
    fa_v_descale = layer.v_scale.expand(descale_shape)
```

把 per-tensor scale 展开成 backend 需要的 descale 张量，传进 FA3 kernel 内部融合反量化。**KV 池只负责存 fp8 bit，scale 的语义解释完全交给 backend。** MXFP8 在同一文件有独立分支（`:1317`、`:1844`）。

---

## 4. MXFP8 方案（独立池 + UE8M0 block scale）

这是最容易被误解的一条路径。**MXFP8 不是"FP8 加个 per-tensor scale"**，它有自己的池类和自己的 scale buffer。

`MHATokenToKVPoolMXFP8`（`memory_pool.py:3260`），继承 `MHATokenToKVPool`：

| 项 | 值 | 位置 |
|----|----|------|
| block 大小 | `MXFP8_SCALE_BLOCK_SIZE = 32` | `:3267` |
| 存储 dtype | `torch.float8_e4m3fn`（**不是 uint8**） | `:3302` |
| k/v buffer 形状 | `(m, n, k)` / `(m, n, v)`，dtype = `float8_e4m3fn` | `:3303-3310` |
| scale buffer dtype | `torch.float8_e8m0fnu`（**UE8M0**，即纯 8bit 指数） | `:3344`、`:3350` |
| scale 维度 | `k_sf_dim = k // 32`、`v_sf_dim = v // 32` | `:3318-3319` |
| 交错标记 | `self.mxfp8_sf_interleaved = (self.page_size == 128)` | `:3320` |

构造期的三道硬校验（`:3281-3293`）：head_dim 必须被 32 整除（k 和 v 各查一次），且 PyTorch 必须有 `float8_e8m0fnu`。另有 `:3296-3300` 拒绝 `SGLANG_USE_HND_KVCACHE`——buffer 是 NHD 布局，继承来的 HND move_kv_cache 分支会搬错字节。

### 4.1 scale 写入与交错布局

`_write_scales`（`memory_pool.py:3450`）—— **这个方法属于 MXFP8 池，不属于 MLA**。

`page_size == 128` 时 scale buffer 采用交错（interleaved）布局，以匹配 kernel 期望的 scale 读取顺序（`:3477-3488`）：

```python
chunk = self.page_size // self.MXFP8_SCALE_BLOCK_SIZE      # 128 // 32 = 4
idx = (off % self.MXFP8_SCALE_BLOCK_SIZE) * chunk + (off // self.MXFP8_SCALE_BLOCK_SIZE)
... .view(torch.float8_e8m0fnu)
```

即把「连续 token 的同一 block」重排成「同一 block 的连续 token」。非 128 页大小走直写。

池构造点在 `kv_cache_configurator.py:1645` / `:1749` / `:1800`（`mha_pool_class` 参数路径）。

---

## 5. FP4 方案（策略模式，真正的量化）

FP4 是独立于 FP8/MXFP8 的一套"三 player"架构，注释在 `fp4_kv_cache_quant_method.py:14-34`：

```
quant_method (纯计算)  ►  Pool (buffer + 批量 dequant 编排)  ►  Backend (视图适配)
```

### 5.1 抽象基类 `KVCacheQuantMethodBase`（`:110`）

关键契约方法：

| 方法 | 位置 | 作用 |
|------|------|------|
| `create_buffers(size, head_num, head_dim, layer_num, device)` | `:225` | 分配 k/v_buffer + k/v_scale_buffer (+ 可选 dq workspace)，返回 dict |
| `quantize_and_store(...)` | `:241` | 把 bf16 的 cache_k/v 量化打包写进 buffer 的 loc 位置 |
| `dequantize_prev_kv(...)` | `:256` | 反量化已存 FP4（前缀 token）→ 供 attention |
| `dequantize_kv_tensor(...)` | `:270` | 反量化单个打包 FP4 张量（plain 读用） |
| `compute_cell_size(...)` | `:283` | per-token 字节数（容量估算） |
| `load_scales_from_model(model_runner)` | `:288` | 从模型各层收集全局 scale |
| `attention_accesses()` | `:121` | 查 access registry，返回本 method 的访问规则 |
| `SCALE_BLOCK_SIZE` | `:118` | 类属性，基类默认 1 |

还有 `needs_dequant_workspace()` / `dequant_workspace_dtype()` / `kv_storage_dtype()` / `needs_global_scale()` 等能力声明方法。

`UnquantizedKVCacheMethod`（`:293`，`SCALE_BLOCK_SIZE = 1`）是 no-op 实现，FP8 和 bf16 都用它。

### 5.2 NVFP4（两级 scale）—— `NVFP4KVCacheMethod`（`:333`）

- **两级缩放**：per-layer **全局 FP32 scale**（`k_scales_gpu[layer]`）+ per-block **FP8 E4M3 scale**。`SCALE_BLOCK_SIZE = 16`（`:340`）。
- **平台**：仅 SM100 / SM120（Blackwell）。
- **全局 scale 加载** `load_scales_from_model`（`:357`）：遍历模型各注意力层的 `layer.k_scale`/`v_scale` 填进 `k_scales_gpu[global_layer_id]`。**SM100 特有修正**（`:414-421`）：ckpt 存的是 `amax/(6*448)`，而 TRT-LLM XQA kernel 期望 `amax/448`，因此 `k_scale *= E2M1_MAX(=6.0)` 补齐；SM120 kernel 路径不同，无需此修正。
- **buffer 形状**（`:433`）：`k/v_buffer = (m, n, k//2)` uint8（半字节打包）；`k/v_scale_buffer = (m, n, k//16)` uint8（`:452`、`:458`）；dq workspace `(m, n, k)` FP8。
- **量化** `quantize_and_store`（`:484`）：调 `NVFP4KVQuantizeUtil.quantize`（`kvfp4_tensor.py:152`）产出打包 fp4 + block scale，全部 `.view(uint8)` 存。
- **反量化** `dequantize_prev_kv`（`:515`）：FP4 →（用两级 scale）→ 最终 `.to(torch.float8_e4m3fn)`，供 FlashInfer prefill 的 FP8 workspace。

### 5.3 fp4_mx_block16（单级 scale）—— `FP4MXBlock16KVCacheMethod`（`:559`）

- **单级缩放**：每 16 个 FP4 值一个 scale（`SCALE_BLOCK_SIZE = 16`，`:567`）。
- **注意命名**：这**不是**标准 MXFP4（标准 MXFP4 block=32），所以不叫 mxfp4。`resolve_kv_cache_quant` 里 `mxfp4` 会被明确拒绝（`:819-821`），保留给未来真正的 block-32 MXFP4。
- **scale buffer 形状**（`:594`、`:602`）：`(m, (head_num*head_dim)//16)`，把 head 维展平。
- **反量化** `dequantize_kv_tensor`（`:653`）：直接 → bf16，无需 fp8 中转。因此它走 **PLAIN** 访问（见 §5.4），triton / torch_native / flex_attention / trtllm_mha 都能直接读。
- 底层数值 kernel：`FP4MXBlock16KVQuantizeUtil`（`kvfp4_tensor.py:58`）。

### 5.4 注意力访问 Registry（最精巧的部分）

`KV_CACHE_ATTENTION_ACCESS_REGISTRY`（`:777`）声明「每个 method × 每个 phase × 哪些 backend → 用哪种 access」。三种 access（`:53`）：

| access kind | 含义 |
|-------------|------|
| `PLAIN` | KV 已是 backend 期望的 dtype/layout（或反量化到 bf16 后直接读） |
| `DEQUANT_WORKSPACE` | 存量化态，读前反量化+解包到临时 workspace |
| `NATIVE_FP4` | backend 直接吃打包 FP4 + scale |

registry 现状：

| method | prefill | decode |
|--------|---------|--------|
| `unquantized` | PLAIN（任意 backend） | PLAIN（任意 backend） |
| `nvfp4` | **DEQUANT_WORKSPACE**（flashinfer，→FP8 E4M3 workspace） | **NATIVE_FP4**（trtllm_mha，直接吃 packed FP4） |
| `fp4_mx_block16` | PLAIN（triton/torch_native/flex_attention/trtllm_mha/fa4，→bf16） | PLAIN（同上去掉 fa4，→bf16） |

**为什么要 registry**（注释 `:20-33`）：`torch.float4_e2m1fn_x2` 只描述"打包 FP4 存储"，既不区分 nvfp4 还是 fp4_mx_block16，也不说 scale 怎么解释；而且同一 recipe 的 prefill / decode 可能用不同视图（NVFP4 prefill 走 FlashInfer dequant workspace、decode 走 TRTLLM native）。把这些组合塞进各 backend 的 dtype 判断会难以维护，故抽成声明式表：backend 用 `(phase, backend_name, tags)` 解析一条规则，命中就按声明访问，否则 fail-fast。

Pool 侧据此自动决定是否建 dq workspace：`_create_quantized_buffers`（`memory_pool.py:2003`）建，`_check_quantized_buffer_access_requirements`（`:2027`）校验 method 声明的 `dequant_workspace_dtype()` 与实际建出来的 `dq_k/v_buffer` 是否一致。

### 5.5 FP4 写入/读取编排（Pool 层）

| 动作 | 调用链 |
|------|--------|
| 写 | `set_kv_buffer`（`:2380`）检测 `is_quantized_kv_cache`（`:1981`）→ `_set_quantized_kv_buffer`（`:2520`）→ `_quantized_scales`（`:2510`，NVFP4 从 `k_scales_gpu` 取全局 scale）→ `quant_method.quantize_and_store` |
| 读（plain 反量化） | `_get_key_buffer`（`:2325`）若 `needs_plain_kv_dequant_read()` → `quant_method.dequantize_kv_tensor` |
| 读（workspace 路径） | `get_flashinfer_dequant_workspace_kv_buffer`（`:2579`）—— FlashInfer prefill 吃 FP8，池负责把打包 FP4 反量化进 dq workspace 再给出 FP8 视图 |
| 读原始打包字节 | `get_raw_kv_buffer`（`:2553`）—— 给 NATIVE_FP4 backend 直接拿 packed 数据 |
| 取 workspace 句柄 | `get_dequant_workspace`（`:2572`） |
| slot 搬移（PD/DCP） | `_slot_move_pointer_buffers`（`:2061`）—— FP4 数据和 scale 分开存，slot 移动要同时更新两者 |

method 的选择与建池在 `kv_cache_configurator.py`：`_build_fp4_quant_method`（`:282`）→ `resolve_kv_cache_quant(kv_cache_dtype_str)`（`:285`）→ `get_kv_cache_quant_method(name)`（`fp4_kv_cache_quant_method.py:830`）→ `load_scales_from_model`；随后作为 `quant_method=` 传进池构造（`:1242` / `:1770`）。

---

## 6. MLA 的 FP4 变体（**不走 registry**）

DeepSeek 类 MLA 有自己的 FP4 池 `MLATokenToKVPoolFP4`（`memory_pool.py:4256`，继承 `MLATokenToKVPool` `:3968`）。它和 MHA 侧的最大区别是：

> ⚠️ **MLA FP4 池把 `FP4MXBlock16KVQuantizeUtil` 内联硬编码**（`:4305`、`:4329`、`:4368`），构造时**没有 `quant_method` 参数**，也**不查 access registry**。所谓"MLA 与 MHA 共用同一套 `KVCacheQuantMethodBase`"的说法是错的。

buffer 布局（`_create_buffers` `:4257`）—— MLA 只有一份融合的 KV（latent），所以只有一个 buffer：

| buffer | 形状 | dtype |
|--------|------|-------|
| `kv_buffer[layer]` | `(m, 1, kv_cache_dim // 2)` | `uint8`（半字节打包） |
| `kv_scale_buffer[layer]` | `(m, kv_cache_dim // 16)` | `uint8` |

其中 `m = size + page_size`（slot 0 是 padding 槽，用于写 padded token 的 dummy 输出），`n = 1`（MLA 只有一个"head"），`k = kv_cache_dim`，`scale_block_size = 16`。

关键方法：

| 方法 | 位置 | 说明 |
|------|------|------|
| `get_key_buffer` | `:4294` | `.view(uint8)` → `FP4MXBlock16KVQuantizeUtil.batched_dequantize`（`:4308`）→ 计算 dtype |
| `set_kv_buffer` | `:4315` | `batched_quantize`（`:4332`）→ `view(uint8)` 存。**没有 `div_(k_scale)`** —— MLA FP4 只有 block scale，没有 per-tensor 全局 scale |
| `set_mla_kv_buffer` | `:4346` | MLA 专用写入（nope / rope 两段分别 `batched_quantize`，`:4372`、`:4375`），scale 用专用 Triton kernel `set_mla_kv_scale_buffer_triton` 写（`:4388`，import 于 `:71`） |

池构造点：`kv_cache_configurator.py:1596`。

---

## 7. 容量与显存估算

- **非量化**：由默认 pool configurator 按 `head_num * head_dim * num_layers * 2 * dtype.itemsize` 算。
- **FP4**：`compute_cell_size`（nvfp4 `:536` / fp4_mx `:682`）= 打包数据 `head_num*(head_dim//2)*layers*2` + block scale `head_num*(head_dim//16)*layers*2`（`:543` / `:687`）+ dq workspace（**跨层共享，不乘 layer 数**）。
- **MXFP8**：数据 `head_num*head_dim*layers*2*1B` + scale `head_num*(head_dim//32)*layers*2*1B`。
- `get_kv_size_bytes`（`memory_pool.py:2247`）分别累加 k/v_buffer、scale_buffer、dq_buffer，用于启动日志与 mem_usage 统计。

---

## 8. 平台 / 后端约束速查

**重要：所有 KV 量化的平台/后端校验已从 `server_args.py` 搬到 `arg_groups/kv_cache_hook.py`。** 旧文档里指向 `server_args.py:5905` / `:5924` / `:7118` 的行号已全部失效。

| 约束 | 现在的位置 |
|------|-----------|
| `mxfp8` 兼容性（SM100+ / FA4） | `arg_groups/kv_cache_hook.py:24-33 handle_mxfp8_kv_cache_compatibility` |
| KV4（nvfp4 / fp4_mx_block16）兼容性 | `:36-55 handle_kv4_compatibility` |
| **各 attention backend 的 KV4 白名单矩阵** | `:56-109`（要不要开 KV4，先查这张表） |
| FP4 与 `--prefill-only-disable-kv-cache` 互斥 | `:415-426`（FP4 用独立分配路径；MXFP8 因有独立 scale buffer 也被拒） |
| FP4 KV cache 不支持 `dcp_kv_mask` | `memory_pool.py` FP4 写入分支 |
| 量化 KV 不支持 post-capture（CUDA-VMM）backing | `memory_pool.py` `is_quantized_kv_cache` 分支 |
| MXFP8 拒绝 `SGLANG_USE_HND_KVCACHE` | `memory_pool.py:3296-3300` |
| PD KV 传输假定 NHD slot-row（HND 不支持） | `memory_pool.py` slot 搬移分支 |
| DFLASH fa4 draft 强制打回 model_dtype | `kv_cache_dtype.py:83-92` |
| fp8 fnuz（HIP）scale ×2 | `kv_cache.py:57`、`:72` |
| KV scale 必须 per-tensor（标量） | `kv_cache.py:76` |

---

## 9. 全景对比表

| 维度 | bf16(auto) | fp8_e4m3 / e5m2 | mxfp8 | nvfp4 | fp4_mx_block16 |
|------|-----------|-----------------|-------|-------|----------------|
| 池类 | `MHATokenToKVPool` | 同左 | **`MHATokenToKVPoolMXFP8`** | `MLATokenToKVPoolFP4`（MHA 版是死代码） | 同左 |
| store_dtype | 原生 | uint8 | **`float8_e4m3fn`** | uint8 (packed) | uint8 (packed) |
| 逻辑 dtype | bf16 | fp8 | fp8_e4m3 | float4_e2m1fn_x2 | float4_e2m1fn_x2 |
| quant_method | unquantized | unquantized | unquantized | **NVFP4** | **FP4MXBlock16**（MLA 侧内联硬编码） |
| scale 粒度 | 无 | per-tensor 标量 | **per-32 block，UE8M0** | 全局 FP32 + block16(FP8 E4M3) | block16 单级 |
| scale 来源 | — | ckpt（`BaseKVCacheMethod`）或 `--quantization-param-path` | 写入时现算 | 层 k/v_scale → `k_scales_gpu` | 无全局，仅 block |
| 每 token 字节 | 2×H×D×2 | 1×H×D×2 | 1×H×D×2 + H×(D/32)×2 | 0.5×H×D×2 + scale | 0.5×H×D×2 + scale |
| 反量化位置 | 无 | attention backend | attention backend（FA4 独立分支） | prefill: FlashInfer workspace；decode: TRTLLM native | 读时 → bf16 (plain) |
| 需 dq workspace | 否 | 否 | 否 | **是**（prefill） | 否 |
| 典型后端 | 任意 | FA3 等 | **仅 FA4** | flashinfer + trtllm_mha | triton / torch_native / trtllm_mha |
| 平台 | 通用 | CUDA 11.8+ / HIP | SM100+ | SM100 / SM120 | CUDA12.8+ / PT2.8+ |

---

## 10. 关键文件索引

| 文件 | 职责 |
|------|------|
| `mem_cache/kv_cache_dtype.py:22` | `--kv-cache-dtype` 字符串 → torch.dtype |
| `mem_cache/memory_pool.py:1698` | `store_dtype` uint8 技巧（FP8/FP4 通用地基） |
| `mem_cache/memory_pool.py:1808 MHATokenToKVPool` | 主 KV 池，含 FP8/FP4 写读编排 |
| `mem_cache/memory_pool.py:3036 MHATokenToKVPoolFP4` | MHA FP4 池（**死代码**，见 §11） |
| `mem_cache/memory_pool.py:3260 MHATokenToKVPoolMXFP8` | MXFP8 独立池 + UE8M0 scale + `_write_scales` `:3450` |
| `mem_cache/memory_pool.py:3968 MLATokenToKVPool` / `:4256 MLATokenToKVPoolFP4` | MLA 池及其 FP4 变体 |
| `layers/quantization/kv_cache.py:18` | `BaseKVCacheMethod`：FP8 层 scale 加载（挂 attention 层），4 个子类见 §3.1 |
| `layers/quantization/fp4_kv_cache_quant_method.py` | FP4 策略基类 `:110` + NVFP4 `:333` / FP4MX `:559` + access registry `:777` + `resolve_kv_cache_quant` `:800` |
| `layers/quantization/kvfp4_tensor.py` | FP4 数值 kernel：`FP4MXBlock16KVQuantizeUtil` `:58`、`NVFP4KVQuantizeUtil` `:152` |
| `mem_cache/kv_cache_configurator.py:282` | `_build_fp4_quant_method`：选 method；建池在 `:1242` / `:1596` / `:1770` |
| `layers/attention/flashattention_backend.py:1228` | 读侧：把 k_scale/v_scale 展开成 descale 传 kernel（MXFP8 分支 `:1317`、`:1844`） |
| `arg_groups/kv_cache_hook.py` | **全部平台/后端校验**（含 KV4 白名单矩阵 `:56-109`） |
| `server_args.py:604-615` / `:579` | `--kv-cache-dtype` choices / `--quantization-param-path` |
| `model_loader/load_model_utils.py:112` | `load_kv_cache_scales`（外部 JSON scale，仅 fp8_e4m3） |

---

## 11. 已知粗糙边缘

1. **`MHATokenToKVPoolFP4`（`memory_pool.py:3036`）是死代码**：它的唯一工厂 `_build_mha_fp4_kv_pool`（`kv_cache_configurator.py:1776`）没有任何调用者。当前生产的 FP4 KV 只有 MLA 一条路。
2. **MLA FP4 绕过了 quant_method 抽象**：`FP4MXBlock16KVQuantizeUtil` 内联硬编码（见 §6），意味着 MLA 无法通过 registry 切换到 NVFP4，也不参与 access registry 的 fail-fast 校验。
3. **`fp4_mx_block16` 名字容易误读**：block 是 16 而不是标准 MXFP4 的 32。`mxfp4` 这个 dtype 名被显式保留并拒绝（`fp4_kv_cache_quant_method.py:819`），留给将来真正的 block-32 实现。
4. **MXFP8 的 `store_dtype` 破例**：全局是"float8 一律按 uint8 存"，MXFP8 偏偏直接用 `float8_e4m3fn`（`:3302`）。读代码时若按 uint8 的假设推 `.view()` 会推错。
5. **交错 scale 布局只在 `page_size == 128` 生效**（`:3320`），其他页大小走直写——两条路的 scale 内存布局不同，改动 kernel 时两条都要覆盖。
6. **KV scale 有两个独立来源**（checkpoint 内嵌 vs `--quantization-param-path` JSON），且后者只在 `fp8_e4m3` 下生效（`load_model_utils.py:116`），同时配置时行为不直观。
7. **NVFP4 的 SM100 scale 修正是个隐式约定**（`:414-421` 的 `*= E2M1_MAX`）：ckpt 语义（`amax/(6*448)`）和 kernel 期望（`amax/448`）不一致，靠这一行桥接。换 kernel 时极易出错。

