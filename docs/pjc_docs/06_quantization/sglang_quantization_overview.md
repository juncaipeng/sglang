# SGLang 量化方案系统梳理（总纲）

> 覆盖 `python/sglang/srt/layers/quantization/` 全体（~110 文件）+ 模型接入 + 内核分派。
> 本文是**总纲**，按「七大方向」拆解。方向⑥（KV Cache 量化）已有独立深入文档
> [`kv_cache_quantization_scheme.md`](kv_cache_quantization_scheme.md)，本文只做衔接。
>
> 校准时间：2026-09，基线 commit `33ed29a0ee`。所有行号均为该基线的实测值。

---

## 0. 一句话总览

SGLang 的量化不是「一个算法」，而是**一套可插拔框架 + 十几种 checkpoint 格式 + 十几种数值配方 + 二十来种计算内核**的组合矩阵。理解它必须先把「谁被量化、用什么格式、按什么内核算」三件事拆开。

**七大方向地图：**

| 方向 | 关注点 | 代码主场 | 本文章节 |
|------|--------|----------|----------|
| ① 框架骨架 | `--quantization` 如何变成「每层一个 method」 | `__init__.py` / `base_config.py` / `base_scheme.py` / `model_config.py` | §2 |
| ② Checkpoint 格式家族 | ModelOpt / compressed-tensors / AWQ / GPTQ / Quark … 各自怎么被识别加载 | 各 `*Config` 的 `from_config` / `override_quantization_method` | §3 |
| ③ 数值格式与算法配方 | FP8 / INT8 / W4A16 / NVFP4 / MXFP4 各自的 scale 粒度与语义 | `fp8.py` / `w8a8_*.py` / `awq/` / `gptq/` / `modelopt_quant.py` | §4 |
| ④ 计算后端 / kernel 分派 | Marlin / CUTLASS / DeepGEMM / FlashInfer / Triton 谁算这一层 | `fp8_utils.py` / `fp4_utils.py` / `marlin_utils*.py` | §5 |
| ⑤ MoE 量化专线 | 为什么专家权重要单独一套（per-expert scale + grouped GEMM） | `FusedMoEMethodBase` 子类 / `moe/moe_runner/` | §6 |
| ⑥ KV Cache 量化 | KV 存 fp8/mxfp8/fp4，见独立文档 | `mem_cache/` + `kv_cache.py` + `fp4_kv_cache_quant_method.py` | §7 → 独立文档 |
| ⑦ 平台特化 + 特殊路径 | NPU/ROCm/CPU-AMX/MPS 的覆盖；expert_pack / 在线 requant | `__init__.py` 平台叠加 / `expert_pack.py` | §8 §9 |

**核心分野（先记住）：**

```
被量化的对象   ┌── 权重 W（线性层 / MoE 专家 / embedding / LM head）──► 方向 ②③④⑤（本文主体）
              ├── 激活 A（运行时动态 / 静态标定）──────────────────► 与 W 配对，记作 WxAy
              └── KV Cache ──────────────────────────────────────► 方向 ⑥（独立文档）
```

---
## 1. 三个正交概念（读代码前必须分清）

### 1.1 量化对象 × 记法 WxAy

SGLang 沿用 `W<权重位宽>A<激活位宽>` 记法：

| 记法 | 含义 | 典型 config |
|------|------|-------------|
| **W8A8** | 权重 8bit + 激活 8bit | `w8a8_int8` / `w8a8_fp8` / `fp8`(static) |
| **W8A16** | 权重 8bit + 激活 bf16（weight-only） | compressed-tensors W8A16Fp8 |
| **W4A16** | 权重 4bit + 激活 bf16（weight-only） | `awq` / `gptq` / `moe_wna16` / `W4A16_NVFP4` |
| **W4A4** | 权重 4bit + 激活 4bit | NVFP4 / MXFP4 / MXINT4 / NPU `mxfp4` |
| **W4A8** | 权重 4bit + 激活 8bit | `w4afp8` / quark w4a8 / modelslim / `mxfp_w4a8` |

**weight-only（A16）** 只压权重、激活保持 bf16，靠反量化权重后普通 GEMM，收益是显存/带宽；**W8A8/W4A4** 连激活也量化，用低精度 Tensor Core，收益是算力。这是选型第一分水岭。

### 1.2 在线 vs 离线 vs 再量化（三态，不是两态）

- **离线**（checkpoint-serialized）：权重已在磁盘上是量化态（`is_checkpoint_fp8_serialized=True` / `is_checkpoint_nvfp4_serialized`），加载即用。绝大多数 ModelOpt/AWQ/GPTQ/compressed-tensors 走这条。
- **在线**：磁盘是 bf16/fp16，加载时**由 SGLang 现场量化**（`Fp8Config` 未 serialized 分支、`nvfp4_online`、`quark_int4fp8_moe`、NPU 的 `mxfp4`/`mxfp_w4a8`）。收益是无需专门 ckpt，代价是加载慢、激活 scale 只能 dynamic。
- **再量化（requantization）**：磁盘已是某量化格式 A，运行时**转成另一格式 B**。这是较新加入的第三态，由 `REQUANTIZATION_METHODS = ["quark_mxfp4"]`（`configs/model_config.py:261`）白名单驱动，典型用法是 NVFP4 ckpt 在 AMD 上转成 quark MXFP4。另一条同类路径是 gfx942 上的 **MXFP8 → block-FP8** 转换（见 §4.1）。

### 1.3 scale 粒度（决定精度/性能/内核）

| 粒度 | 一个 scale 覆盖 | 谁用 |
|------|----------------|------|
| per-tensor | 整个权重张量一个标量 | fp8 static、modelopt_fp8 |
| per-channel（per-column） | 每个输出通道一个 | w8a8_int8 权重、w8a8_fp8 |
| per-token | 每个 token（激活）一个 | w8a8 动态激活、nvfp4_online 激活 |
| block（block_n×block_k，常见 128×128） | 一个二维块一个 | block-fp8（DeepSeek）、block-int8 |
| group（group_size，常见 128/64/32） | 沿 K 维每 group 一个 | AWQ/GPTQ（W4A16）、w4afp8(128) |
| micro-scaling（MX，1×32） | 每 32 元素一个 E8M0/UE8M0 | MXFP8 / MXFP4 / MXINT4 |
| 两级（per-tensor FP32 + block-16 FP8） | 全局标量 × 每 16 元素 | NVFP4 |

粒度越细精度越好但 scale 存储/读取开销越大，且**内核必须支持对应粒度**——这就是为什么 block-fp8 只能走 DeepGEMM/CUTLASS block 内核，而 per-tensor fp8 能走 Marlin。

---
## 2. 方向①：框架骨架 —— 从 `--quantization` 到「每层一个 method」

这是理解一切的地基。全流程 = **注册表查类 → checkpoint 探测定名 → 仲裁 → 造 config → 每层 attach method（可能再挂 scheme）**。

### 2.1 四级抽象（比旧文档多一层 Scheme）

```
QuantizationConfig（一次/模型，描述"这个 ckpt 用什么量化"）  base_config.py:140
   │  get_quant_method(layer, prefix) —— 按层类型分派        :249
   ▼
QuantizeMethodBase（挂在 layer.quant_method）                :24
   ├── LinearMethodBase           普通线性层                 :50
   ├── FusedMoEMethodBase         MoE 专家（多 create_moe_runner）:90
   └── BaseKVCacheMethod / KVCacheQuantMethodBase（方向⑥）
   ▼（可选，多格式家族共用一个 Method 时）
BaseLinearScheme / BaseMoEScheme（挂在 layer.scheme）        base_scheme.py:11 / :56
```

**为什么有第四层 Scheme**：compressed-tensors 一个 config 要覆盖十几种 scheme，如果每种都写一个 `LinearMethod` 会爆炸。于是拆成「一个瘦 Method（负责与 layer 交互）+ 一个 Scheme（负责真正的 create_weights/apply_weights）」。这套抽象最初是从 compressed-tensors 提取的（commit `eba6af385d`，PR #17503），现已被 **AWQ / GPTQ / Quark / ModelSlim** 全部复用：

| 家族 | Scheme 基类 | 文件 |
|------|-------------|------|
| compressed-tensors | `CompressedTensorsLinearScheme` :19 / `CompressedTensorsMoEScheme` :65 | `compressed_tensors/schemes/compressed_tensors_scheme.py` |
| AWQ | `AWQLinearSchemeBase` :17 / `AWQMoESchemeBase` :33 | `awq/schemes/awq_scheme.py` |
| GPTQ | `GPTQLinearSchemeBase` :17 / `GPTQMoESchemeBase` :33 | `gptq/schemes/gptq_scheme.py` |
| Quark | `QuarkLinearScheme` :17 / `QuarkMoEScheme` :64 | `quark/schemes/quark_scheme.py` |
| ModelSlim | `ModelSlimLinearScheme` :15 / `ModelSlimMoEScheme` :54 | `modelslim/schemes/modelslim_scheme.py` |

`BaseLinearScheme` 唯一的类属性是 `requires_weight_loader_v2`（`base_scheme.py:21`）：某些 scheme 的参数只实现 v2 加载 API（`load_{column,row,merged_column,qkv}_weight`），置 True 后 `LinearBase` 会改走 `weight_loader_v2`（`linear.py:390-391`、`:1500-1501`），而不需要为此换掉整个 LinearMethod 类。

> ⚠️ **陷阱**：`layer.scheme` 必须是**普通对象、不能是 `nn.Module`**（`linear.py:159-165` 有专门注释）。否则 `nn.Module.__setattr__` 会把它塞进 `_modules`，而类默认值 `scheme = None`（`linear.py:166`）会把读取遮蔽掉。

`FusedMoEMethodBase` 除 `create_weights` 外还有三个关键钩子：`create_moe_runner`（:104）、`get_triton_quant_info`（:117，LoRA MoE runner 复用同一套 flag/scale）、`get_moe_quant_info`（:130，按 runner backend 分派，目前只实现 triton）。

### 2.2 注册表（`layers/quantization/__init__.py`）

`BASE_QUANTIZATION_METHODS`（`:79-108`）是**名字→config 类**的字典，29 条：

| 名字 | Config 类 | 说明 |
|------|-----------|------|
| `fp8` / `mxfp8` | `Fp8Config` | 同一类，`mxfp8` 走 block=[1,32] 分支 |
| `blockwise_int8` | `BlockInt8Config` | block-int8 |
| `modelopt` / `_fp8` / `_fp4` / `_mixed` | `ModelOptFp8/Fp4/MixedPrecisionConfig` | `modelopt` 默认落 FP8 类，靠探测细分 |
| `nvfp4_online` | `NvFp4OnlineConfig` | 在线 NVFP4（仅 MoE） |
| `w8a8_int8` / `w8a8_fp8` | `W8A8Int8Config` / `W8A8Fp8Config` | W8A8 |
| `awq` / `awq_marlin` | `AWQConfig` / `AWQMarlinConfig` | W4A16 非对称 |
| `gptq` / `gptq_marlin` | `GPTQConfig` / `GPTQMarlinConfig` | W4A16 可 act-order |
| `moe_wna16` | `MoeWNA16Config` | AWQ/GPTQ 的 MoE 统一封装 |
| `compressed-tensors` | `CompressedTensorsConfig` | llm-compressor 多 scheme 总入口 |
| `w4afp8` | `W4AFp8Config` | W4A8-fp8 |
| `petit_nvfp4` | `PetitNvFp4Config` | petit NVFP4（ROCm） |
| `quark` / `quark_mxfp4` / `quark_int4fp8_moe` | `QuarkConfig` / `QuarkInt4Fp8Config` | AMD Quark |
| `auto-round` / `auto-round-int8` | `AutoRoundConfig` / `W8A8Int8Config` | Intel auto-round |
| `modelslim` | `ModelSlimConfig` | Ascend ModelSlim |
| `humming` | `HummingConfig` | humming |
| `mxfp_w4a8` | `Mxfp4W4A8Config` | Ascend W4A8（MXFP4 权重 + MXFP8 激活） |
| `bitsandbytes` | `BitsAndBytesConfig` | bnb NF4/INT8 |
| `gguf` | `GGUFConfig` | llama.cpp GGUF |

**平台叠加（四组，比旧文档多 XPU）**：

| 平台 | 动作 | 行号 |
|------|------|------|
| CPU / CUDA / gfx95 | 追加 `mxfp4 → Mxfp4Config` | `:111-116` |
| NPU | 覆盖 `gptq → GPTQAscendConfig`、`mxfp4 → Mxfp4W4A4Config` | `:119-128` |
| XPU | 覆盖 `gptq → GPTQXPUConfig`、`awq → AWQXPUConfig` | `:131-137` |
| MPS | 追加 `mlx_q4` / `mlx_q8 → MlxQuantizationConfig` | `:140-146` |
| CPU-AMX | `get_quantization_config` 里改查 `CPU_QUANTIZATION_METHODS` 白名单（7 条，`awq→AWQCPUConfig`、`gptq→CPUGPTQConfig`） | `:149-157`、`:170-177` |

查类入口 `get_quantization_config(name)`（`:162-186`）：先校验名字在册 → CPU-AMX 改查白名单 → out-of-tree 平台给 `current_platform.get_quantization_config` 一次机会（`:179-184`）→ 返回。

> 两个易踩的名字冲突：
> 1. NPU 上 `Mxfp4W4A4Config.get_name()` 返回 `"mxfp4"`，**遮蔽**了 CUDA 侧的 OCP-MoE `Mxfp4Config`。
> 2. CPU-AMX 的白名单替换发生在**任何 `override_quantization_method` 之前**，所以 CPU 上看不到 Marlin 自动升级。

---
### 2.3 checkpoint 格式自动探测

用户常只写 `--quantization modelopt` 甚至不写，靠 ckpt 里的标记自动定名。探测有**两套并行实现**（历史遗留，尚未合并）：

**A. `ModelConfig._parse_quant_hf_config`（`configs/model_config.py:1243-1338`）** —— 四级回退：

| 顺序 | 源 | 行号 |
|------|-----|------|
| 1 | `hf_config.quantization_config`（经 `_quant_config_to_dict` :63 归一化） | `:1244-1246` |
| 1b | 若无 `quant_method` 或 `quant_method=="modelopt"`，回灌 `_parse_modelopt_quant_config` 结果 | `:1249-1257` |
| 2 | `hf_config.compression_config`（compressed-tensors 旧键） | `:1259-1261` |
| 3 | Hub 上独立的 `hf_quant_config.json`（ModelScope 走 `HubApi`，否则 `HfApi`；在线态先 `file_exists` 探针 + `retry(max_retry=2)`） | `:1262-1330` |
| 4 | 本地 `<model_path>/hf_quant_config.json` | `:1331-1337` |

ModelSlim **不在**这个函数里，而是独立的 `_find_quant_modelslim_config`（`:1340-1352`）读 `quant_model_description.json`，并手动注入 `quant_cfg["quant_method"] = "modelslim"`（`:1350`，因为该描述文件本身没有这个字段），draft 模型直接返回 None（`:1341-1342`）。

**B. `resolve_checkpoint_quant_spec`（`model_loader/checkpoint_quantization.py:80-100`）** —— 较新的纯数据抽取，查找顺序 `_select_hf_quant_metadata`（`:61-77`）：`quantization_config` → **`text_config.quantization_config`** → `compression_config`，返回深拷贝装进 `CheckpointQuantSpec`（`:25-36`），这样调用方可以往里塞运行期字段而不污染 HF config。
> ⚠️ 它认 `text_config.quantization_config`（多模态模型的文本子配置），而 A 路**不认**。这是两套实现的实质差异。

**ModelOpt 专用映射** `_parse_modelopt_quant_config`（`model_config.py:1354-1401`）按 `quant_algo`：

- `MIXED_PRECISION` → 扫 `quantized_layers`，含 NVFP4/W4A16_NVFP4 则 `modelopt_mixed`（`:1368`），否则 `w4afp8`（`:1369`）
- 含 `FP4`/`NVFP4` → `modelopt_fp4`（`:1370-1371`）
- `== "FP8"` → `modelopt_fp8`（`:1372-1373`）
- `== "MXFP8"` → `mxfp8`，并**合成额外字段**：`activation_scheme="dynamic"`、`weight_block_size=[1, group_size]`（默认 32）、`scale_fmt="ue8m0"`（`:1374-1399`）

权威映射表在 `layers/modelopt_utils.py:23-30`，`canonicalize_modelopt_quant_algo`（`:33-45`）做**精确 allowlist 查表**（不做子串匹配，否则 `MXFP8` 会被误判成 `FP8`）：

```
FP8 → modelopt_fp8      MXFP8 → mxfp8
FP4 / NVFP4 / NVFP4_AWQ / W4A16_NVFP4 → modelopt_fp4
```

唯一调用方：`base_config.py:217`，在 `_modelopt_override_quantization_method`（`:196-228`）内，且只在 `user_quant == "modelopt"` 时（`:216`）；`nvfp4_online` 和 `REQUANTIZATION_METHODS` 直接早退返回 None（`:209-210`）。

### 2.4 总仲裁 `_verify_quantization`（`model_config.py:1504-1667`）

这是「CLI 说的」和「ckpt 说的」打架的裁判，是排障第一现场：

```
1. 收集 quant_cfg = modelslim_config or hf_config          :1563-1566（ModelSlim 优先）
2. quant_method = quant_cfg["quant_method"] or self.quantization   :1579-1581
3. preserve_online_draft_quantization：draft 模型 + CLI=nvfp4_online
   + ckpt 是 modelopt_fp4/mixed → 保护 CLI 选择           :1587-1591
4. if self.quantization not in REQUANTIZATION_METHODS:     :1595  ← 再量化请求绝不被回改
      for _, method in QUANTIZATION_METHODS.items():        :1598-1606
          override = method.override_quantization_method(quant_cfg, self.quantization)
          第一个真值即 break，覆盖 self.quantization
5. 调和：
   - self.quantization is None → 采纳 ckpt                 :1609-1610
   - 不一致 → 查 compatible_quantization_methods           :1613-1617
       ├ 兼容 → info 日志，保留 CLI                        :1618-1624
       ├ elif is_draft_model → info 日志，改用 ckpt         :1625-1633
       ├ elif in REQUANTIZATION_METHODS → info_once，保留 CLI :1634-1637
       └ else → raise ValueError                           :1638-1644
6. 收尾校验：未知方法 raise :1647 / ROCm 白名单 raise :1652
   / 不在 optimized 列表则 warning :1657（SM100 的 mxfp4/mxfp8 豁免）
```

`compatible_quantization_methods`（`:1546-1557`）全表：

```
modelopt_fp8    : [modelopt]
modelopt_fp4    : [modelopt, fp8]        ← 纯 FP8 ckpt 也保留 fp4，让合格 MoE 专家在线再量化
modelopt_mixed  : [modelopt]
nvfp4_online    : [fp8, modelopt_fp8]
petit_nvfp4     : [modelopt]
w8a8_int8       : [compressed-tensors, compressed_tensors]
w8a8_fp8        : [compressed-tensors, compressed_tensors]
auto-round-int8 : [compressed-tensors, compressed_tensors]
```

> ⚠️ 第 4 步**遍历整个注册表 dict**，所以 `BASE_QUANTIZATION_METHODS` 的**插入顺序会影响结果**。默认基类返回 None（`base_config.py:187-194`），实际覆盖它的只有 ModelOpt 三兄弟、`AWQMarlinConfig`（`awq/awq.py:333`）、`GPTQMarlinConfig`（`gptq/gptq.py:414`）、`ModelOptMixedPrecisionConfig`（`:808`）、`HummingConfig`（`humming.py:503`）。W8A8/block-int8 不覆盖，靠上面的兼容表匹配 compressed-tensors ckpt。

同文件还有几个量化相关字段值得记住：`is_fp4_experts`（`:402-431`，读 `routed_experts_quant_method`，DSV4 走 `SGLANG_DSV4_FP4_EXPERTS`/`_DEQUANT` + 自动探测）、`nvfp4_moe_meta`（`:444-464`，`MIXED_PRECISION` + `moe_quant_algo==NVFP4` 时记录 group_size/exclude）、`get_quantization_config_log_str`（`:1403`，排障用的一行摘要）。`_validate_quantize_and_serve_config`（`:1472-1501`）末尾**无条件 raise NotImplementedError**，即 `--quantize-and-serve` 目前是硬关闭的。

---
### 2.5 造 config 实例

唯一的 `from_config` 调用集中地：`model_loader/weight_utils.py:get_quant_config`（`:263-423`）。它的分支比想象的多：

| 分支 | 行号 | 行为 |
|------|------|------|
| 取类 | `:269` | `quant_cls = get_quantization_config(model_config.quantization)` |
| gguf 快捷 | `:273` | `quant_cls.from_config({})` |
| **ckpt 内联 spec** | `:275-302` | 走 `resolve_checkpoint_quant_spec`，注入 `packed_modules_mapping`(:293)、`hf_config`(:294)，若 CLI 在 `REQUANTIZATION_METHODS` 再注入 `requantization_method`(:297-298)。`modelopt_mixed` 缺 `quantized_layers`/KV 元数据时**故意穿透**到文件路径(:282-292) |
| bitsandbytes | `:310` | `from_config({"adapter_name_or_path": ""})` |
| 无 config 文件名 | `:336-365` | `mxfp8` → `Fp8Config(use_mxfp8=True, is_checkpoint_fp8_serialized=False)`；`quark_mxfp4` 直读 `hf_quant_config.json` 并用 `producer.name` 拼 `quant_method`(:341-360) |
| 文件路径 | `:367-386` | glob `get_config_filenames()`；0 个时 `modelopt_fp4` 退到 `for_online_weight_quantization`(:373-378)，否则 raise；>1 个 raise |
| modelopt producer | `:399-420` | `quant_algo is None` → raise（Eagle3 例外返回 None :403-410）；`FP8`/`FP4` 强制改类后 `from_config` |
| 兜底 | `:421-423` | `_resolve_explicit_draft_quant_config(model_config, quant_cls.from_config(config))` |

`_resolve_explicit_draft_quant_config`（`:241-259`）专治投机解码：draft 模型显式指定量化、且 ckpt 是 serialized NVFP4 但 `mtp.layers.0.mlp.experts` 被 exclude 时，换成 `ModelOptFp4Config.for_online_weight_quantization`。

**loader 侧后处理** `loader.py:_get_quantization_config`（`:166-267`，被 17 处 loader 调用）：

1. 取 `model_class.packed_modules_mapping` / `remap_prefix`（`:171-173`，**原地改，无拷贝**）
2. quark 注入 `gate_up_proj` / `fused_qkv_a_proj_with_mqa`（`:175-181`）；NPU 注入 `visual`/`vision_model`/`model` 的嵌套映射（`:183-204`）
3. 调 `get_quant_config`；返回 None 直接放行（Llama-4-Maverick-Eagle3 变通，`:210-212`）
4. **`Fp8Config` 字段注入**（`:215-217`）：`is_fp4_experts`、`dequant_fp4_to_fp8`（来自 `SGLANG_DSV4_FP4_DEQUANT`）
5. **构造 `HybridFp8NvFp4Config`**（`:218-239`）：`nvfp4_moe_meta` 非空时，先造 `ModelOptFp4Config`（exclude 追加 `model.decoder.*` / `stages.*`，让 MTP/NextN/DSpark draft 专家留在源 MXFP4，`:228-230`），再包成 `HybridFp8NvFp4Config(fp8_config=..., nvfp4_config=...)`
6. 算力校验（`:242-254`）与 act dtype 校验（`:255-261`），不过 raise
7. `apply_weight_name_mapper`（`:262-265`），把 HF 命名的 exclude 模式映射到 sglang 结构

### 2.6 挂载点（五处，全仓库仅此五处）

`get_quant_method` 在 `base_config.py:249-261` 声明为抽象方法。**层级调用点只有 5 个**：

| 层类型 | 位置 | 无量化时回退 |
|--------|------|--------------|
| 线性层 | `layers/linear.py:192`（`LinearBase.__init__`） | `UnquantizedLinearMethod()` :187-190 |
| Embedding / LM head | `layers/vocab_parallel_embedding.py:300` | `UnquantizedEmbeddingMethod()` :301-302 |
| 注意力（KV cache） | `layers/radix_attention.py:148` | 保持 `None`（:145） |
| MoE（KTransformers EP） | `layers/moe/fused_moe_triton/layer.py:447` | `UnquantizedFusedMoEMethod(use_triton_kernels)` :448-449 |
| MoE（常规） | `layers/moe/fused_moe_triton/layer.py:453` | `UnquantizedFusedMoEMethod(三参)` :454-459 |

几个细节：

- **线性层**紧接着会 `wrap_method_with_debug_kernel_once(self.quant_method, "apply", ...)`（`linear.py:194-199`）。
- **Embedding** 多一道检查：若 `type(self) is VocabParallelEmbedding` 且 method 类没实现 `embedding`（`method_has_implemented_embedding`，`base_config.py:283`），直接 `NotImplementedError`（`:307-315`）。这就是为什么 NVFP4 要专门写 `ModelOptNvFp4EmbeddingMethod`（`modelopt_quant.py:688`）。
- **RadixAttention** 拿到 method 后**立刻调 `create_weights(self)`**（`radix_attention.py:149-150`）——这是 `k_scale`/`v_scale` 参数被注册的地方。
- **MoE 的两个分支**不是「GPU vs 非 GPU」，而是**是否存在 KTransformers EP 配置**（`kt_config = create_kt_config_from_server_args(...)`，`:442`）。有则把正常 method 包成 `KTEPWrapperMethod(gpu_method, kt_config)`（`:450`），`gpu_method` 意为「负责 GPU 常驻专家的那个 method」，其余由 KT 卸载。`FusedMoE.__init__` 还接受显式 `quant_method` 形参（`:329`，`:440` 赋值）覆盖 config 选择。

一个把四类层放在一处的范本：`compressed_tensors/compressed_tensors.py:177-244` —— `LinearBase`(:184) / `ParallelLMHead`(:195) / `RadixAttention`(:205，KV scheme 不支持时**降级返回 None + warning**，不阻塞启动) / `FusedMoE`(:222，MXFP4 预检在 :228-234 **完全绕过 scheme 抽象**直接返回 `Mxfp4MoEMethod`)。

### 2.7 全流程时序图

```
启动  --quantization=modelopt（或不填）
 ▼
ModelConfig._verify_quantization                  configs/model_config.py:1504
 │  ① _parse_quant_hf_config 四级回退读 ckpt 标记        :1243
 │  ② _parse_modelopt_quant_config 按 quant_algo 定名     :1354
 │  ③ 遍历注册表 override_quantization_method 仲裁        :1598
 │  ④ compatible 表 / draft / requant 三种调和            :1613
 │  → self.quantization = "modelopt_fp4"
 ▼
get_quantization_config("modelopt_fp4")           __init__.py:162 → ModelOptFp4Config 类
 ▼
weight_utils.get_quant_config                     weight_utils.py:263
 └─ ModelOptFp4Config.from_config(ckpt_dict)      modelopt_quant.py:1505 → config 实例
 ▼
loader._get_quantization_config 后处理            loader.py:166
 └─ 注入 packed_modules_mapping / is_fp4_experts / 可能包成 HybridFp8NvFp4Config
 ▼
逐层构造模型（5 个挂载点）
 ├─ LinearBase.__init__      → get_quant_method → ModelOptFp4LinearMethod（可能再挂 scheme）
 ├─ VocabParallelEmbedding   → ModelOptNvFp4EmbeddingMethod
 ├─ RadixAttention           → ModelOptFp8KVCacheMethod（立刻 create_weights）
 └─ FusedMoE.__init__        → ModelOptNvFp4FusedMoEMethod
 ▼
权重加载
 ├─ method.create_weights(layer, ...)             建量化参数（打包权重 + scale buffer）
 └─ method.process_weights_after_loading(layer)   转置/重打包/塌缩 per-expert scale/格式转换
 ▼
前向
 ├─ Linear: method.apply(layer, x[, bias]) → 内核分派（方向④）
 └─ MoE:    create_moe_runner + apply(dispatch_output) → *MoeQuantInfo → MoeRunner（方向⑤）
```

---
## 3. 方向②：Checkpoint 格式家族

「格式家族」= 谁生产的 ckpt、怎么被识别、映射到哪个 config。同一数值格式（如 FP8）可由多家生产，SGLang 用不同 config 类对接各家的字段布局与命名习惯。

| 家族 | 生产者 | ckpt 标记 | 对接 config | 支持格式 | 文件 |
|------|--------|-----------|-------------|----------|------|
| **ModelOpt** | NVIDIA TensorRT-ModelOpt | `quant_algo`: FP8/NVFP4/NVFP4_AWQ/MXFP8/W4A16_NVFP4/MIXED_PRECISION | `ModelOptFp8/Fp4/MixedPrecisionConfig` | FP8 / NVFP4 / MXFP8 / 逐层混精 | `modelopt_quant.py` |
| **compressed-tensors** | llm-compressor（neuralmagic/vLLM 生态） | `quantization_config.format` + 每 scheme 配置 | `CompressedTensorsConfig` → 14 个 scheme | W8A8-int8/fp8、W8A16-fp8、WNA16、NVFP4、MXINT4、W4A8 | `compressed_tensors/` |
| **AWQ** | AutoAWQ | `quant_method:awq`, `zero_point` | `AWQConfig` / `AWQMarlinConfig` / CPU / XPU | W4A16 非对称（**仅 4bit**） | `awq/awq.py` |
| **GPTQ** | AutoGPTQ / GPTQModel | `quant_method:gptq`, `desc_act`, `sym` | `GPTQConfig` / `GPTQMarlinConfig` / Ascend / CPU / XPU | W2/3/4/8 A16（可 act-order） | `gptq/gptq.py` |
| **Quark** | AMD Quark | `quant_method:quark` | `QuarkConfig` / `QuarkInt4Fp8Config` | FP8 W8A8、MXFP4 W4A4、W4A8 | `quark/` |
| **ModelSlim** | 华为 msit/ModelSlim | `quant_model_description.json` | `ModelSlimConfig` | INT4/INT8、MXFP4/MXFP8、W4A8（Ascend） | `modelslim/` |
| **bitsandbytes** | HF bnb | `quant_method:bitsandbytes` | `BitsAndBytesConfig` | NF4 / FP4 / INT8（可双量化） | `bitsandbytes.py` |
| **GGUF** | llama.cpp | `.gguf` 文件 | `GGUFConfig` | GGML K-quants 2~8bit | `gguf.py` |
| **auto-round** | Intel auto-round | `quant_method:auto-round` | `AutoRoundConfig`（内部转 GPTQ/AWQ/INT8） | W2/3/4/8 A16 / INT8 | `auto_round.py` |
| **petit** | petit | NVFP4 变体 | `PetitNvFp4Config` | NVFP4（gfx90a/gfx942） | `petit.py` |
| **humming** | humming | 声称 `mxfp4` ckpt | `HummingConfig` | 逐层混合（W4AFp8 + stacked block-FP8） | `humming.py` |
| **MLX** | Apple mlx_lm | `mlx_q4/q8` | `MlxQuantizationConfig` | 4/8bit（仅标记，实际由 mlx 量化） | `mlx.py` |

### 3.1 ModelOpt（NVIDIA，`modelopt_quant.py`，3037 行）

共同基类 `ModelOptQuantConfig`（`:284`）管三件事：`exclude_modules`/`kv_cache_quant_algo`/`use_per_token_activation`（`:293-295`），以及共享分派器 `_get_quant_method(..., Linear=, Moe=)`（`:297`：Linear/ParallelLMHead :309 / RadixAttention→`ModelOptFp8KVCacheMethod` :315 / FusedMoE :317）。`is_layer_excluded`（`:349`）支持精确名、glob、逐段匹配、剥 `language_model.` 前缀、以及**融合层模式**（`{q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj}`，`:371`）。

四个子类：

- **`ModelOptFp8Config`**（`:398`）：per-tensor FP8。`from_config`（`:437`）同时兼容扁平 `config.json:quantization_config` 与旧式 `hf_quant_config.json`。方法：`ModelOptFp8LinearMethod`（`:503`，含 SM120 cuBLAS/GEMV 快路，可用 `SGLANG_DISABLE_SM120_FP8_GEMV` 关掉）、`ModelOptFp8MoEMethod`（`:1044`，`apply` :1293 有 TRT-LLM per-tensor FP8 的 BYPASSED-routing 快路）、`ModelOptFp8KVCacheMethod`（`:675`）。
- **`ModelOptFp4Config`**（`:1392`）：NVFP4。新增 `use_per_token_activation`（`:1414`）、`is_awq`（`:1415`）、`is_w4a16`（`:1425`）、`for_online_weight_quantization`（`:1452`）。方法：`ModelOptFp4LinearMethod`（`:1677`，NVFP4_AWQ 的 per-input-channel 预 scale 在 `:1761-1763`）、weight-only `ModelOptNvFp4A16LinearMethod`（`:2093`）、`ModelOptNvFp4EmbeddingMethod`（`:688`）、MoE `ModelOptNvFp4FusedMoEMethod`（`:2244`）。
- **`ModelOptMixedPrecisionConfig`**（`:784`）：**逐层混精**。内部持有 5 个子 config（`:801-805`）：`fp8_config` / `fp8_pb_wo_config`(block 128×128) / `mxfp8_config`(block 1×32) / `nvfp4_config` / `nvfp4a16_config`。`_resolve_quant_algo`（`:935`）按 prefix 逐层解析，**融合/打包 shard 内算法不一致会 raise**（`:954`）。`get_quant_method`（`:985`）路由表：

  | 层 | FP8 | FP8_PB_WO | MXFP8 | NVFP4 | W4A16_NVFP4 |
  |----|-----|-----------|-------|-------|-------------|
  | Linear | `ModelOptFp8LinearMethod` :1003 | `Fp8LinearMethod` :1005 | `Fp8LinearMethod(mxfp8_config)` :1007 | `ModelOptFp4LinearMethod` :1009 | `ModelOptNvFp4A16LinearMethod` :1011 |
  | Embedding | — | — | — | `ModelOptNvFp4EmbeddingMethod` :1022 | — |
  | FusedMoE | `ModelOptFp8MoEMethod` :1032 | — | `Fp8MoEMethod` :1034 | `ModelOptNvFp4FusedMoEMethod` :1036 | 同 :1038 |

  > Embedding 分支必须排在 LMHead 之后（`:1014` 有注释），因为 `ParallelLMHead` 是 `VocabParallelEmbedding` 的子类。

- **`HybridFp8NvFp4Config(Fp8Config)`**（`:1638`，注意继承 `Fp8Config` 而非 ModelOpt 基类）：服务 `nvidia/DeepSeek-V4-Pro-NVFP4` 这类「`quant_method=fp8` + `moe_quant_algo=NVFP4`」的 ckpt。线性层走 FP8、MoE 走 NVFP4（`:1661`）；若 `is_fp4_experts` 则把 MTP MoE 包成 `Mxfp4FlashinferTrtllmMoEMethod(Fp8MoEMethod(self))`（`:1669`）。**唯一实例化点是 `loader.py:237`**。

探测：三个 ModelOpt 子类的 `override_quantization_method`（`:420`/`:1443`/`:808`）都委托共享的 `_modelopt_override_quantization_method`（`base_config.py:197`）。

### 3.2 compressed-tensors（`compressed_tensors/`）

`CompressedTensorsConfig`（`compressed_tensors.py:112`）是**多 scheme 总入口**。

**线性 scheme 选择** `_get_scheme_from_parts`（`:681`），优先级：

```
1. :693  WNA16 group/channel → CompressedTensorsWNA16          :698
         （须 pack_quantized 且 bits 在 WNA16_SUPPORTED_BITS，否则 ImportError :706）
2. :710  仅当 is_activation_quantization_format(quant_format)：
   2a :711 NVFP4 W4A4  → CompressedTensorsW4A4Fp4              :716
   2b :722 FP8 W8A8    → CompressedTensorsW8A8Fp8              :730
                         算力不够则降级 CompressedTensorsW8A16Fp8 :739
   2c :745 FP8 W8A16   → CompressedTensorsW8A16Fp8             :747
   2d :752 INT8 static W8A8        → (NPU)CompressedTensorsW8A8Int8 :754/:760
   2e :766 INT8 dynamic-token W8A8 → (NPU)CompressedTensorsW8A8Int8 :768/:774
3. :780  NotImplementedError
```

**MoE scheme 选择** `get_moe_scheme`（`:782`，先断言三个 projection 的量化配置一致 `:801-814`，被 ignore 的层返回 None `:816`）：

```
1. :822 WNA16（非 NPU）
   ├ :824 MXINT4-A16 且 runner=flashinfer_trtllm → CompressedTensorsMxInt4MoE :831
   ├ :832 ROCm                                  → CompressedTensorsWNA16TritonMoE :834
   ├ :851 runner=triton 或 SM100 auto 且 triton 支持 → 同上 :860（显式要 triton 但不支持则 raise :846）
   └ :863 兜底 Marlin                            → CompressedTensorsWNA16MoE :864
   NPU 分支 :866 dynamic-token W4 且无 input_quant → NPUCompressedTensorsW4A16Int4DynamicMoE :871
2. :872 NVFP4 W4A4 → CompressedTensorsW4A4Nvfp4MoE  :874
3. :875 FP8 W8A8   → CompressedTensorsW8A8Fp8MoE    :877
4. :878 INT8 dynamic-token → 仅 NPU :881，否则 NotImplementedError :883
5. :886 wint4afp8  → NPU :890 / CompressedTensorsW4AFP8MoE :892
6. :893 W4A8 dynamic-token → 仅 NPU :896，否则 NotImplementedError :898
7. :901 RuntimeError
```

判定谓词集中在 `:423-658`（`_is_dynamic_token_w4a8` :423、`_is_wint4afp8` :445、`_is_static_tensor_w8a8` :472、`_is_fp8_w8a8` :507、`_is_fp4a4_nvfp4` :569、`_is_wNa16_group_channel` :598、`_is_mxint4a16` :623 …）。scheme 文件共 14 个，在 `compressed_tensors/schemes/`。

### 3.3 AWQ vs GPTQ（两大 W4A16 权重-only）

两者都是 weight-only + group-wise scale，差别在元数据与重打包：

| 维度 | AWQ | GPTQ |
|------|-----|------|
| 位宽 | **仅 4bit**（`awq.py:84` 直接 raise） | 2/3/4/8bit（`gptq.py:106`） |
| 对称性 | **恒非对称**，带 `zero_point` | 可对称（`sym`），对称时无 zero-point |
| 激活重排 | 无 | **可选 act-order**（`desc_act`，存 `g_idx` 列置换） |
| 逐模块覆盖 | 无 | `dynamic` 正则表（`gptq.py:93`） |
| Marlin 自动升级 | `override_quantization_method` `awq.py:333`；显式写 `--quantization awq` 则保留 plain AWQ 并打日志（`:347`） | `gptq.py:414`，多一道 `check_marlin_format` 守卫（`:44`）——**已是 Marlin 格式的 ckpt 不再二次覆盖** |
| 平台变体 | `AWQCPUConfig` :180（AMX）、`AWQXPUConfig` :217（**FusedMoE 直接 raise** :233） | `GPTQAscendConfig` :194、`CPUGPTQConfig` :231、`GPTQXPUConfig` :265（**FusedMoE raise** :283） |
| MoE method | `AWQMoEMethod` :470（Marlin fused） | `GPTQMoEMethod` :529 / `GPTQMarlinMoEMethod` :621 |

**`moe_wna16`**（`moe_wna16.py:68`）：AWQ/GPTQ 本身难以把专家表达成融合 experts，`MoeWNA16Config` 是**统一 MoE weight-only 封装**，按 `linear_quant_method`（`:87`）分派到底层 GPTQMarlin/GPTQ/AWQ 线性方法（`:211/:213/:217`），`use_marlin`（`:89`）决定 Marlin 还是 Triton WNA16 grouped-GEMM。

### 3.4 其余家族一句话

- **Quark**（AMD ROCm，`quark/quark.py:289`）：FP8 W8A8 + MXFP4 W4A4（dense+MoE）+ W4A8；`online_scheme="quark_mxfp4"`（`:309`）走**在线再量化**；`QuarkFusedMoEMethod` :1018、`QuarkKVCacheMethod` :1071。
- **ModelSlim**（Ascend，`modelslim/modelslim.py:90`）：逐层读 `quant_description`，覆盖 W8A8-int8/W4A4-int4/W4A8-int8/MXFP8/MXFP4/W4A8-MXFP4，无需 CLI flag。
- **bitsandbytes**（`:44`）：4bit(nf4/fp4)/8bit，支持双量化（`:57`），MoE 见 `BitsAndBytesMoEMethod` :422。
- **GGUF**（`:85`）：GGML K-quants；HIP 上会 warning（`:90`）；Ascend 有 `GGUFMoEAscendMethod` :805。
- **auto-round**（Intel，`:39`）：不自己算，重新分派到 GPTQ/AWQ(+Marlin)/INT8 三条既有路径（`:52`）；CPU-AMX 上只允许 4bit（`:42-43`）。
- **petit**（`:36`）：ROCm gfx90a/gfx942 的 NVFP4，**仅 dense**（`PetitNvFp4LinearMethod` :137），无 MoE。
- **humming**（`:447`）：其实是**逐层混合**容器，内部两套 checkpoint schema（`_W4AFp8CheckpointWeightSchema` :226 / `_StackedBlockFp8CheckpointWeightSchema` :310）+ 逐层 `HummingLayerQuantizationConfig`（:691）。
- **MLX**（`:39`）：纯标记 config，所有 method 都 raise（`:47-52`），真正量化由 `mlx_lm` 完成。

---
## 4. 方向③：数值格式与算法配方

同一「格式家族」下真正决定精度/性能的是**数值配方**：位宽 + scale 粒度 + 在线/离线 + 激活是否量化。

### 4.1 FP8 全家（`fp8.py` 2766 行 + `fp8_utils.py`）

`Fp8Config`（`fp8.py:234`）是最重要的一类，一套 config 覆盖多种配方，靠这些字段区分（`:237-289`）：

| 字段 | 取值 | 决定 |
|------|------|------|
| `is_checkpoint_fp8_serialized` | True/False | 离线 fp8 ckpt vs 在线量化 bf16 |
| `activation_scheme` | `static` / `dynamic` | 激活 scale 来自 ckpt 标定 vs 运行时算 |
| `weight_block_size` | None / `[block_n,block_k]` | per-tensor vs block-wise |
| `use_mxfp8` | True | 强制 block=`[1,32]`（`:284-288`），`get_name()`→`mxfp8`（`:292`） |
| `ignored_layers` | list | 可用 `SGLANG_FP8_IGNORED_LAYERS` 追加（`:260`） |
| `is_fp4_experts` | True | DSV4 的 MXFP4 打包专家标记（`:249-251`） |
| `dequant_fp4_to_fp8` | True | DSV4 把 FP4 专家反量化回 FP8 |

`get_min_capability()`（`:298`）分五档：NPU→0、MUSA→31、MXFP8+gfx95→95、MXFP8+需 block 转换→94、其余 mxfp8→100 / fp8→80。

**三种 FP8 配方：**

1. **per-tensor / per-token FP8**：最通用，能走 Marlin。
2. **block-fp8**（DeepSeek 风格，128×128）：要求 serialized + dynamic 激活（`:272`/`:280`）。走 DeepGEMM / CUTLASS / FlashInfer block 内核。
3. **MXFP8**（block=`[1,32]`，micro-scaling，scale 为 UE8M0）：走独立的 `dispatch_w8a8_mxfp8_linear`（`fp8_utils.py:640`）。

`Fp8LinearMethod.apply`（`fp8.py:966`）分派顺序：

```
:972  use_marlin        → torch.ops.sglang.apply_fp8_marlin_linear
:983  use_mxfp8         → 按 mxfp8_dense_backend 选 scale 布局：
                          flashinfer_cutlass/cutedsl → weight_scale_inv_swizzled  :987
                          flashinfer_trtllm          → weight_scale_inv_shuffled  :989
                          deep_gemm                  → weight_scale_inv_deepgemm  :991
                          其他                       → 原始 weight_scale_inv      :994
:1013 block_quant       → CPU AMX fp8_scaled_mm_cpu :1015 / w8a8_block_fp8_linear :1026,:1035
:1044 输入是 tuple      → apply_fp8_linear(pre_quant_output_dtype=...)
                          （融合 RMSNorm+FP8 量化的产物 (qx, scale[, orig_dtype])）
:1062 兜底              → apply_fp8_linear
```

`use_marlin` 由 `SGLANG_FORCE_FP8_MARLIN`（`environ.py:918`）或 `can_auto_enable_marlin_fp8()` 决定（`fp8.py:469-470`）。后者在 **`fp8_utils.py:2069`**（不在 `marlin_utils_fp8.py`！），条件是 `80 <= sm < 89`——即 Ampere/Ada 这类没有原生 fp8 Tensor Core 的卡，把 fp8 权重当 weight-only 走 Marlin。

**MXFP8 → block-FP8 在线转换（容易被忽略的一条重要路径）**

`_mxfp8_to_block_fp8_required`（`fp8.py:126`）= `mxfp8_block_convert_required() or SGLANG_FORCE_MXFP8_BLOCK_CONVERT`。源判定在 `utils/common.py:4349`，**仅在 AMD gfx942（CDNA3 / MI300，且非 gfx95）为真**：该芯片没有硬件 MX-scaled matmul（Triton 的 `tl.dot_scaled` 无法 lower，gfx950 的 `mfma_scale` 指令也不存在），所以加载时把 MXFP8 ckpt（e4m3fn + 1×32 UE8M0）**反量化再重量化成 DeepSeek-V3 式 block-FP8 `[128,128]`**（e4m3fn + fp32），从而能跑原生 block-FP8 的 aiter/triton 内核。`fp8.py:124-125` 记录了实测：MiniMax-M3 上吞吐 +20%、GSM8K 精度基本持平（0.9719 vs 0.9689）。

实现在 `mxfp8_block_convert.py`：`_ue8m0_to_fp32` :22 → `dequant_mxfp8_2d_to_bf16` :27 → `bf16_to_block_fp8_128` :40（amax/448.0）→ `convert_mxfp8_weight_to_block_fp8` :63。消费方：线性层 `fp8.py:477`（用于 `:666`）、MoE `fp8.py:1092` → `_convert_mxfp8_moe_to_block_fp8` :1762。gfx95 保留原生 MX 路径。

### 4.2 INT8 系

| Config | 位置 | scale 粒度 | 备注 |
|--------|------|-----------|------|
| `W8A8Int8Config` | `w8a8_int8.py:65` | 权重 per-channel 静态对称 + 激活 per-token 动态对称（`:68-69`） | ckpt 非 serialized 时在线量化；`W8A8Int8MoEMethod` :238；CPU 走 `fused_experts_cpu` |
| `BlockInt8Config` | `blockwise_int8.py:39` | 二维 block（`weight_block_size`），激活仅 dynamic（`:68-71`） | **必须离线**（`is_checkpoint_int8_serialized` 必需，`:60`）；`BlockInt8MoEMethod` :244 |

`W8A8Fp8Config`（`w8a8_fp8.py:39`）是 FP8 版 W8A8：权重 per-channel 静态（无 CUTLASS 时降级 per-tensor，`:54-55`）+ 激活 per-token 动态，未 serialized 时加载即量化（`:53`），`W8A8FP8MoEMethod` :197，要求算力 89+。

`w8a8_int8`/`w8a8_fp8`/`auto-round-int8` 三者都在 `compatible_quantization_methods` 里声明与 compressed-tensors ckpt 兼容。

### 4.3 W4A16（weight-only 4bit）

AWQ / GPTQ / moe_wna16，group-wise scale，反量化权重后普通 GEMM，主要靠 **Marlin** 内核（§5）。ModelOpt 侧还有 `W4A16_NVFP4`（`ModelOptNvFp4A16LinearMethod`，`modelopt_quant.py:2093`）——NVFP4 权重 + bf16 激活。

### 4.4 FP4 全家（4bit 浮点）

| 格式 | scale 结构 | 平台 | 对接 |
|------|-----------|------|------|
| **NVFP4** | 两级：per-tensor 全局 FP32 + block-16 FP8 E4M3 | SM100/SM120 | `ModelOptFp4Config` / `nvfp4_online` / compressed-tensors NVFP4 / petit |
| **NVFP4_AWQ** | NVFP4 + per-input-channel 预 scale（`modelopt_quant.py:1761`） | SM100 | `ModelOptFp4Config(is_awq=True)` |
| **MXFP4** | 单级：block-32 E8M0（OCP 标准 micro-scaling） | SM100 / CDNA | `Mxfp4Config` / humming / quark_mxfp4 |
| **MXINT4 W4A4** | block INT4 | flashinfer_trtllm | compressed-tensors `CompressedTensorsMxInt4MoE` |
| **NPU MXFP4 W4A4** | 单级 MXFP4（`float4_e2m1fn_x2`），在线 | Ascend | `Mxfp4W4A4Config`（注册名就叫 `mxfp4`） |
| **NPU MXFP4 W4A8** | MXFP4 权重 + MXFP8 激活，均在线 | Ascend | `Mxfp4W4A8Config`（注册名 `mxfp_w4a8`） |

⚠️ **命名陷阱**：KV cache 侧的 `fp4_mx_block16` block=16，**不是**标准 MXFP4（block=32），二者勿混。权重侧 `mxfp4` 指标准 OCP MXFP4 block=32。

`Mxfp4Config`（`mxfp4.py:302`）**只做 MoE**：`Mxfp4MoEMethod` :383（离线）/ `Mxfp4DynamicQuantMoEMethod` :1921（在线），线性层在 HIP 上直接返回 unquantized（`:367`）。

`NvFp4OnlineConfig`（`nvfp4_online.py:32`）也是**只做 MoE**：加载时把 BF16/FP16/FP8 专家权重现场转 NVFP4（`:37-39`），group_size 16、激活 per-token FP32 scale，需要 SM100（`:107`）。

### 4.5 全景对比表（权重/激活侧，KV 见方向⑥）

| 配方 | W/A | scale 粒度 | 在线 | 典型内核 | 平台 | config |
|------|-----|-----------|------|----------|------|--------|
| per-tensor FP8 | W8A8 | per-tensor | 可 | apply_fp8_linear / Marlin | 通用 | `fp8`, `modelopt_fp8` |
| block-FP8 | W8A8 | 128×128 | 否 | DeepGEMM / CUTLASS / FlashInfer | Hopper+ | `fp8`(block) |
| MXFP8 | W8A8 | 1×32 UE8M0 | 可 | flashinfer cutedsl/cutlass/trtllm / deepgemm / gfx95 dot_scaled | SM100+ / gfx95 | `mxfp8` |
| MXFP8→blockFP8 | W8A8 | 加载时转 128×128 | 转换 | aiter/triton block-fp8 | **gfx942 专属** | `mxfp8` |
| INT8 W8A8 | W8A8 | per-ch + per-token | 是 | CUTLASS int8_mm | 通用 | `w8a8_int8` |
| block-INT8 | W8A8 | block | 否 | `int8_utils.py` | 通用 | `blockwise_int8` |
| AWQ | W4A16 | group（非对称+zp） | 否 | Marlin / awq | 通用 | `awq(_marlin)` |
| GPTQ | W2/3/4/8 A16 | group（可 act-order） | 否 | Marlin / gptq | 通用 | `gptq(_marlin)` |
| NVFP4 | W4A4 | 全局FP32 + block16 FP8 | 可 | FlashInfer cudnn/cutedsl/cutlass/trtllm / Marlin | SM100/120（Marlin 兜底 SM80+） | `modelopt_fp4`, `nvfp4_online` |
| W4A16_NVFP4 | W4A16 | 同上 | 否 | cutedsl | SM100 | `modelopt_fp4` / mixed |
| MXFP4 | W4A4 | block32 E8M0 | 可 | FlashInfer cutlass/trtllm / Marlin / humming | SM100/CDNA | `mxfp4`, `humming`, `quark_mxfp4` |
| W4A8-fp8 | W4A8 | group W(128) + fp8 A | 否 | 专用 interleave（TRT-LLM 风格） | Hopper 90+ | `w4afp8`, quark |
| 逐层混精 | 混 | 逐层不同 | 混 | 逐层选 | SM100 | `modelopt_mixed` |

---
## 5. 方向④：计算后端 / kernel 分派

"用哪个数值格式"和"用哪个 kernel 算"是**两件独立的事**。同一个 block-FP8 权重，可以由 DeepGEMM、CUTLASS、FlashInfer、Triton、AITER 中任意一个来算。这一层就是决定"谁来算"。

### 5.1 三个互相独立的 GEMM 后端枚举

这是本方向最容易踩坑的地方：**SGLang 有三个各自独立解析、互不相干的 dense GEMM 后端枚举**，只是名字长得像。

| 枚举 | 定义位置 | 成员 | CLI 参数 | 初始化 |
|------|---------|------|---------|--------|
| `Fp8GemmRunnerBackend` | `fp8_utils.py:308` | 9 个（见下） | `--fp8-gemm-backend`（`server_args.py:1729-1738`） | `initialize_fp8_gemm_config()` `:799` |
| `Mxfp8DenseGemmBackend` | `fp8_utils.py:349` | 6 个 `:353-358` | 无独立 CLI，由 `resolve_mxfp8_dense_gemm_backend()` `:574` 推 | 同上 |
| `Fp4GemmRunnerBackend` | `fp4_utils.py:93` | 6 个 `:96-101` | `--fp4-gemm-backend`（`server_args.py:1739-1748`） | `initialize_fp4_gemm_config()` `:145` |

`Fp8GemmRunnerBackend` 的 9 个成员（`fp8_utils.py:311-319`）：

```
AUTO, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, FLASHINFER_CUTEDSL,
FLASHINFER_DEEPGEMM, CUTLASS, DEEP_GEMM, TRITON, AITER
```

每个成员都有配套的 `is_xxx()` 谓词（`:321-346`），全局单例存在 `FP8_GEMM_RUNNER_BACKEND`（`:382`），读取入口 `get_fp8_gemm_runner_backend()`（`:812`）。

`Fp4GemmRunnerBackend`：`AUTO / FLASHINFER_CUDNN / FLASHINFER_CUTEDSL / FLASHINFER_CUTLASS / FLASHINFER_TRTLLM / MARLIN`，读取入口 `get_fp4_gemm_runner_backend()`（`fp4_utils.py:161`）。

### 5.2 block-FP8 的分派链（最完整的例子）

```
Fp8LinearMethod.apply (fp8.py:966)
  ├─ use_marlin           → apply_fp8_marlin_linear      (marlin_utils_fp8.py:59)
  ├─ mxfp8 权重           → dispatch_w8a8_mxfp8_linear   (fp8_utils.py:640)
  ├─ block 量化           → dispatch_w8a8_block_fp8_linear (fp8_utils.py:556)
  ├─ per-token-per-channel→ apply_fp8_ptpc_linear        (fp8_utils.py:2078)
  └─ 其余 per-tensor      → apply_fp8_linear             (fp8_utils.py:1801)
```

`dispatch_w8a8_block_fp8_linear` 内部再分两步：

1. **显式后端**：`_dispatch_explicit_backend()` `:716` —— 用户指定了 `--fp8-gemm-backend` 就直接查表。
2. **AUTO**：`_dispatch_auto_backend()` `:778` —— 按优先级挑：

| 序 | 条件 | 选中 | 代码 |
|---|------|------|------|
| 1 | DeepGEMM 可用 | `DEEP_GEMM` | `:787` |
| 2 | Blackwell (SM100+) | FlashInfer | `:789` |
| 3 | SM120 | `CUTLASS` | `:791` |
| 4 | ROCm 且 `_use_aiter` | `AITER` | `:793` |
| 5 | 兜底 | `TRITON` | `:796` |

各后端的实际入口函数：

| 后端 | 入口 |
|------|------|
| FlashInfer（通用） | `flashinfer_gemm_w8a8_block_fp8_linear_with_fallback` `:820` |
| FlashInfer-DeepGEMM | `flashinfer_deepgemm_...` `:895` |
| CUTLASS | `cutlass_...` `:969` |
| DeepGEMM | `deepgemm_...` `:1026` |
| AITER | `aiter_w8a8_block_fp8_linear` `:1167` |
| Triton | `triton_w8a8_block_fp8_linear` `:1246` |
| MXFP8 blockscaled | `flashinfer_mxfp8_blockscaled_linear` `:1301` |
| 最终兜底 | `_apply_fallback_scaled_mm` `:1753` |

FlashInfer 内部还有一层 groupwise 后端选择：`_get_flashinfer_groupwise_backend()` `:447`。MXFP8 走 `_deepgemm_w8a8_mxfp8_linear_with_fallback` `:660`。

### 5.3 Marlin：低比特权重的通用兜底

Marlin 是一套"把任意低比特权重重排成 GPU 友好布局，再用统一 kernel 算"的方案，SGLang 把它当作**跨格式的兜底路径**——AWQ / GPTQ / FP8 / NVFP4 / MXFP4 都能落到 Marlin。

| 文件 | 覆盖格式 | 关键函数 |
|------|---------|---------|
| `marlin_utils.py` | INT4/INT8（AWQ/GPTQ） | `apply_gptq_marlin_linear` `:488`、`apply_awq_marlin_linear` `:559`、permute/repack 辅助 `:284-421`、`MarlinConfig` `:626`、`MarlinLinearMethod` `:731` |
| `marlin_utils_fp8.py` | FP8 | `apply_fp8_marlin_linear` `:59`、`prepare_fp8_layer_for_marlin` `:103`、`prepare_moe_fp8_layer_for_marlin` `:197` |
| `marlin_utils_fp4.py` | NVFP4 / MXFP4 | `apply_fp4_marlin_linear` `:78`、`prepare_nvfp4_layer_for_marlin` `:139`、`prepare_moe_mxfp4_layer_for_marlin` `:299`、`prepare_moe_nvfp4_layer_for_marlin` `:440` |

进入 Marlin 的两种方式：

- **主动 override**：`AWQMarlinConfig.override_quantization_method`（`awq/awq.py:333`）、`GPTQMarlinConfig.override_quantization_method`（`gptq/gptq.py:414`）在 checkpoint 满足 `check_marlin_format`（`gptq.py:44`）时把 `quantization` 改写成 `*_marlin`。
- **FP8 自动兜底**：`Fp8LinearMethod` 在 `:469-470` 判 `use_marlin`，依据 `can_auto_enable_marlin_fp8()`（注意：这个函数在 **`fp8_utils.py:2069`**，不在 `marlin_utils_fp8.py`），或环境变量 `SGLANG_FORCE_FP8_MARLIN`（`environ.py:918`）强制。

### 5.4 其余数值工具库

| 文件 | 内容 |
|------|------|
| `int8_utils.py` | `apply_w8a8_block_int8_linear` `:11`、`input_to_int8` `:34`、`block_dequant` `:47` |
| `fp4_utils.py` | FP4 后端枚举与初始化（见 5.1） |
| `mxfp8_block_convert.py` | MXFP8→blockFP8 加载期转换：`_ue8m0_to_fp32` `:22`、`dequant_mxfp8_2d_to_bf16` `:27`、`bf16_to_block_fp8_128` `:40`、`convert_mxfp8_weight_to_block_fp8` `:63` |
| `humming_utils.py:15-124` | humming（CDNA）低比特工具 |
| `petit_utils.py:36-78` | petit NVFP4 内核封装 |
| `rocm_mxfp4_utils.py` | 14 行的 AITER MXFP4 薄封装 |

### 5.5 相关环境变量与调优开关

| 变量 | 位置 | 作用 |
|------|------|------|
| `SGLANG_FORCE_FP8_MARLIN` | `environ.py:918` | 强制 FP8 走 Marlin |
| `SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE` | `environ.py:933` | 开启 FP8 GEMM 配置自动调优 |
| `SGLANG_OPT_MOE_QUANT_ONCE` | `environ.py:1115` | MoE 激活只量化一次，复用给多个 GEMM |
| `USE_TRITON_W8A8_FP8_KERNEL` | `environ.py:1150` | 强走 Triton W8A8 FP8（读取点 `fp8_utils.py:251`）。**缺 `SGLANG_` 前缀**，违反仓库 env-var 规范 |

ROCm 上的 CK 路径：`SGLANG_FORCE_CK_W8A8` 现在**只剩注释**（`fp8_utils.py:103`），真实开关是函数 `set_force_ck_w8a8()`（`:107`）。AITER 总闸是 `_use_aiter`（`fp8_utils.py:67-68`）。

---

## 6. 方向⑤：MoE 量化专线

MoE 的量化路径**和 dense linear 完全分开**，是整个量化体系里第二套并行的分派机制。原因是 MoE 的 GEMM 是 grouped/batched 的，权重多一个 expert 维度，scale 布局、kernel 签名都不一样。

### 6.1 两阶段契约

`FusedMoEMethodBase`（`base_config.py:90`）要求量化方法实现两件事：

```
第一阶段（建图期，一次）：create_moe_runner(...)   base_config.py:104
    → 挑一个 MoeRunnerBackend，构造 MoeRunner 存到 layer 上

第二阶段（每次 forward）：apply(...)               base_config.py:110
    → 打包一个 *MoeQuantInfo（携带权重/scale/flag）
    → 交给 MoeRunner.run() 执行
```

另外两个可选钩子：`get_triton_quant_info()` `:117`、`get_moe_quant_info()` `:130`。

### 6.2 MoeRunnerBackend：18 个成员

定义在 `layers/moe/utils.py:170`（成员列在 `:172-189`），配套谓词类 `_MoeRunnerBackendPredicates` `:104`，注册表 `RegisteredMoeRunnerBackend` `:193`，类型别名 `MoeRunnerBackendLike` `:199`。

解析入口：

| 函数 | 位置 | 说明 |
|------|------|------|
| `resolve_moe_runner_backend` | `utils.py:216` | 从 server args + 量化方法推导 |
| `initialize_moe_config` | `utils.py:389` | 进程级初始化 |
| `get_moe_runner_backend` | `utils.py:440` | 运行期读取 |
| `get_speculative_moe_runner_backend` | `utils.py:447` | 投机解码的 draft MoE 单独一套（CLI: `server_args.py:2213`） |

CLI 参数 `--moe-runner-backend`（`server_args.py:2415-2423`）。当值为 `AUTO` 时，**由各量化方法自己决定**：

| 量化方法 | 解析点 |
|---------|--------|
| FP8 | `fp8.py:2365-2396`（`create_moe_runner`）；DeepGEMM 判定 `is_deepgemm_moe_runner_backend_enabled` `:1112-1134` |
| 未量化 | `unquant.py:594` / `:845` |
| ModelOpt NVFP4 | `modelopt_quant.py:2254` / `:2833` |
| MXFP4 | `mxfp4.py:1386` / `:2020` |
| humming | `humming.py:1236` |

参数侧还有一层约束：`arg_groups/overrides.py:1461 _moe_runner_backend_quant_constraints`（另见 `:1524`、`:1625`、`:773`、`:859`）、`arg_groups/moe_hook.py:194-208`。FP8 dense backend 为 auto 时自动升级到 `flashinfer_trtllm` 的逻辑在 `arg_groups/overrides.py:846-850`。

### 6.3 MoeRunner 的调度

`MoeRunner`（`moe_runner/runner.py:53`）构造时：

- hpc_ops 可用性门禁 `:70-77`
- deepep_v2 门禁 `:79-86`
- 后端→RunnerCore 分派 `:90-150`（**未覆盖的后端在 `:150` 抛 `NotImplementedError`**）
- 融合算子查表 `FusedOpPool.get_fused_func()` `:158`，CI 可用 `SGLANG_CI_DISABLE_MOE_FUSED_FUNC` 关掉（`:171-178`，注意这里是裸 `os.environ` 读取，没走 `environ.py`）
- 执行入口 `run()` `:180`

各 RunnerCore：

| Core | 位置 |
|------|------|
| `TritonRunnerCore` | `moe_runner/triton.py:77` |
| `TritonKernelsRunnerCore` | `moe_runner/triton_kernels.py:81` |
| `DeepGemmRunnerCore` | `moe_runner/deep_gemm.py:267` |
| `AiterRunnerCore` | `moe_runner/aiter.py:222` |
| `AscendRunnerCore` | `moe_runner/ascend.py:84` |
| `HummingRunnerCore` | `moe_runner/humming.py:145` |

### 6.4 15 个 `*MoeQuantInfo`

每个后端定义自己的 QuantInfo 结构体，基类 `MoeQuantInfo`（`moe_runner/base.py:89`）：

| 类 | 位置 |
|----|------|
| `TritonMoeQuantInfo` | `triton.py:54` |
| （triton_kernels） | `triton_kernels.py:64` |
| `DeepGemmMoeQuantInfo` | `deep_gemm.py:245` |
| （aiter） | `aiter.py:52` |
| （ascend） | `ascend.py:218` |
| （marlin） | `marlin.py:86` |
| （humming） | `humming.py:98` |
| （hpc_ops） | `hpc_ops.py:66` |
| （flashinfer cutlass） | `flashinfer_cutlass.py:43`、`:65` |
| （flashinfer cutedsl） | `flashinfer_cutedsl.py:386` |
| （flashinfer trtllm） | `flashinfer_trtllm.py:679`、`:962`、`:1242` |

以 `TritonMoeQuantInfo`（`triton.py:55-74`）为例，字段就是"kernel 需要知道的全部量化信息"：

```
w13_weight, w2_weight, b13, b2,
use_mxfp8, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, use_int4_w4a16,
per_channel_quant,
w13_scale, w2_scale, w13_zp, w2_zp, a13_scale, a2_scale,
block_shape, fuse_swiglu_interleaved
```

这些 flag 最终在 `kernels/ops/moe/fused_moe_triton_kernels.py:782 invoke_fused_moe_kernel` 里被翻译成 kernel 的编译期常量（分支见 `:833-836`、`:865`、`:884`、`:887`）。

> 注：所有 `*MoeQuantInfo` 目前仍是 `@dataclass`。仓库规范要求新容器用 `msgspec.Struct`，这批属于历史遗留（grandfathered）。

### 6.5 per-expert scale 折叠（一个容易被忽略的关键步骤）

fused MoE 的 w13（gate+up 合并）来自 checkpoint 里**两份独立的 scale**，但 kernel 只接受一份。`Fp8MoEMethod.process_weights_after_loading`（`fp8.py:2052`）的做法是：

```python
max_w13_scales = layer.w13_weight_scale.max(dim=1).values   # fp8.py:2151
# 然后把两个 shard 按新 scale 重量化   fp8.py:2148-2169
```

即"取两者最大值作为统一 scale，再把权重逐 shard 重新量化对齐"。这是精度和 kernel 简洁性的折中，**每个支持 fused MoE 的量化方法都要自己做一遍**。

### 6.6 MXFP4 MoE 的专用文件

| 文件 | 内容 |
|------|------|
| `mxfp4_flashinfer_cutlass_moe.py:32` | FlashInfer CUTLASS 路径 |
| `mxfp4_flashinfer_trtllm_moe.py:51` | TRT-LLM 路径（含 `maybe_fuse_routed_scale_and_shared_add` `:382`） |
| `mxfp4_marlin_moe.py:20` / `:45` | Marlin 兜底 |
| `mxfp4_humming_moe.py:18` | humming（CDNA） |

---

## 7. 方向⑥：KV Cache 量化（第三套并行体系）

KV Cache 量化**不复用**前面任何一套机制，它是完全独立的第三套体系。原因很直观：权重量化发生在加载期、一次性；KV 量化发生在每一步 forward 的写入/读取路径上，必须和 memory pool、attention backend 深度耦合。

### 7.1 两条并行的实现路线

| 路线 | 触发方式 | 机制 |
|------|---------|------|
| **A. dtype 路线（主流）** | `--kv-cache-dtype`（`server_args.py:604-615`） | `configure_kv_cache_dtype()`（`mem_cache/kv_cache_dtype.py:22`）设定 pool 的 `store_dtype`，pool 内部自己做量化/反量化 |
| **B. checkpoint scale 路线** | checkpoint 里带 KV scale | `QuantizationConfig.get_quant_method()` 对 `RadixAttention` 返回一个 `BaseKVCacheMethod`（`kv_cache.py:18`） |

路线 B 的 4 个具体实现（这是既有 KV 文档漏掉的）：

| 类 | 位置 |
|----|------|
| `Fp8KVCacheMethod` | `fp8.py:2760` |
| `ModelOptFp8KVCacheMethod` | `modelopt_quant.py:675` |
| `QuarkKVCacheMethod` | `quark/quark.py:1071` |
| `CompressedTensorsKVCacheMethod` | `compressed_tensors.py:1164` |

`BaseKVCacheMethod` 只做三件事：`create_weights` `:32` 建 `k_scale`/`v_scale` 参数、`process_weights_after_loading` `:51` 校验并落值、以及在 `:76` **硬拒非 per-tensor 的 KV scale**。

还有一条老的旁路：`--quantization-param-path`（`server_args.py:579`）→ `load_kv_cache_scales`（`load_model_utils.py:112`，**只在 `fp8_e4m3` 下生效** `:116`）→ `kv_cache_scales_loader`（`weight_utils.py:1840`），从独立 JSON 文件读 KV scale。

### 7.2 `store_dtype` 技巧

`memory_pool.py:1698-1702`：pool 对外声称的 `dtype` 和实际显存里的 `store_dtype` 可以不同。例如 KV4 时逻辑 dtype 是 bf16，`store_dtype` 是 uint8（两个 4bit 打包）。attention backend 读的时候按 `store_dtype` 取 buffer，再按需要 dequant。

### 7.3 FP4 KV 的策略模式

`fp4_kv_cache_quant_method.py` 是唯一一处把 KV 量化做成**可插拔策略**的地方：

| 组件 | 位置 |
|------|------|
| `KVCacheQuantMethodBase` | `:110` |
| `UnquantizedKVCacheMethod` | `:293` |
| `NVFP4KVCacheMethod` | `:333` |
| `FP4MXBlock16KVCacheMethod` | `:559` |
| 注册表 | `:777` |
| `resolve_kv_cache_quant` | `:800` |

底层量化工具在 `kvfp4_tensor.py`：`FP4MXBlock16KVQuantizeUtil` `:58`、`NVFP4KVQuantizeUtil` `:152`。

⚠️ **MLA 并不走这套注册表**。`MLATokenToKVPoolFP4`（`memory_pool.py:4256`）把 `FP4MXBlock16KVQuantizeUtil` **内联硬编码**（`:4304`、`:4328`、`:4367`），构造时没有 `quant_method` 参数。而 MHA 侧的 `MHATokenToKVPoolFP4`（`:3036`）目前**没有任何调用点**（工厂 `_build_mha_fp4_kv_pool` 在 `kv_cache_configurator.py:1776-1794`，同样无调用者）——是死代码。

### 7.4 平台校验已搬家

所有 KV 量化的平台/后端兼容性校验**已从 `server_args.py` 移到 `arg_groups/kv_cache_hook.py`**：

| 检查 | 位置 |
|------|------|
| MXFP8 KV 兼容性 | `handle_mxfp8_kv_cache_compatibility` `:24-33` |
| KV4 兼容性 | `handle_kv4_compatibility` `:36-55` |
| 各 attention backend 的 KV4 白名单矩阵 | `:56-109` |
| FP4 相关互斥 | `:415-426` |

> 详细的 pool 布局、attention backend 侧 descale 逻辑、以及 MXFP8 KV 的 UE8M0 block scale buffer，见 [kv_cache_quantization_scheme.md](kv_cache_quantization_scheme.md)。

---

## 8. 方向⑦：平台特化速查

同一个 `--quantization` 值在不同平台上可能落到**完全不同的 config 类**。这是通过 `__init__.py` 里的平台 overlay 实现的（见 §2.2）。

| 平台 | overlay 位置 | 替换/新增 |
|------|-------------|----------|
| CPU | `__init__.py:149-157` `CPU_QUANTIZATION_METHODS` | `awq→AWQCPUConfig`、`gptq→CPUGPTQConfig`，另有 AMX 白名单 |
| CUDA / gfx95 | `:111-116` | 新增 `mxfp4→Mxfp4Config` |
| NPU (Ascend) | `:119-128` | `gptq→GPTQAscendConfig`、`mxfp4→Mxfp4W4A4Config` |
| XPU (Intel) | `:131-137` | `gptq→GPTQXPUConfig`、`awq→AWQXPUConfig` |
| MPS (Apple) | `:140-146` | 新增 `mlx_q4/mlx_q8→MlxQuantizationConfig` |

平台专属的重点差异：

- **ROCm gfx942（MI300/CDNA3）**：没有硬件 MX-scaled matmul，所以 MXFP8 权重在**加载时**被转成 block-FP8（`_mxfp8_to_block_fp8_required` `fp8.py:126`，转换实现 `mxfp8_block_convert.py`，平台判定 `utils/common.py:4349 mxfp8_block_convert_required()` 仅 gfx942 返回 True）。MoE 侧对应 `_convert_mxfp8_moe_to_block_fp8` `fp8.py:1762`。
- **ROCm gfx95**：有 `dot_scaled`，MXFP4 原生支持，走 AITER / humming。
- **NPU**：`Mxfp4W4A4Config`（`get_name()` 返回 `"mxfp4"`）、`GGUFMoEAscendMethod`（`gguf.py:805`）、`AscendRunnerCore`。
- **Intel CPU AMX**：白名单式替换，**发生在任何 `override_quantization_method` 之前**。
- **XPU**：`AWQXPUConfig`（`awq.py:217`）和 `GPTQXPUConfig`（`gptq.py:265`）都**硬拒 MoE**（分别在 `:233` / `:283` raise）。

---

## 9. 特殊路径

### 9.1 `expert_pack`：伪装成量化方法的"加载格式"

`ExpertPackConfig`（`expert_pack.py:47`）继承 `GGUFConfig`，`get_name()` 返回 `"expert_pack"`（`:59`），`is_fp4_experts = True`（`:50`）。但它**不在 `BASE_QUANTIZATION_METHODS` 注册表里**——因为它本质是一种 **load format**，不是量化方法：

| 侧面 | 位置 |
|------|------|
| load format 枚举 | `configs/load_config.py:26 EXPERT_PACK` |
| CLI | `server_args.py:106-111`、`:454` |
| 参数钩子 | `arg_groups/expert_pack_hook.py` |
| loader | `model_loader/expert_pack_loader.py:260` |

它做的事是**把 MoE 专家权重流式化（out-of-core）**：`ExpertPackMoEMethod.create_weights`（`:88`）**不注册任何专家参数**，只留一个 `expert_pack_marker` buffer（`:127-131`）；拒绝 EP / TP>1（`:103`）；forward 时（`apply` `:255`）从 store 里 `acquire`（`:275`、`:292`）当前需要的专家，用 `mxfp4_matvec_dual` `:299` / `_clamped_swiglu` `:310` / `mxfp4_matvec` `:311` 算完再 `store.mark_used` `:325`。Kimi 有专用分支 `_apply_kimi` `:174`。

模型代码通过 `quant_config.get_name() == "expert_pack"` 判定并走特殊分支。

### 9.2 再量化（requantization）

`REQUANTIZATION_METHODS = ["quark_mxfp4"]`（`model_config.py:261`）—— 目前**只有一个成员**。含义是：checkpoint 里存的是格式 A，运行期要转成格式 B。这在 `_verify_quantization` `:1595` 有专门的守卫，防止被普通的"格式不匹配"逻辑误判为错误。

支撑再量化的两个小工具：

| 文件 | 内容 | 使用者 |
|------|------|--------|
| `dequantization.py` | `copy_missing_attrs` `:39`、`dequantize_fp8` `:49`、`dequantize_nvfp4` `:72` | quark mxfp4 schemes、`multimodal_gen/.../comfy_nvfp4.py:28` |
| `online_quantization.py` | `CopyNumelCounter(TorchDispatchMode)` `:6` —— 统计 `aten.copy_` 元素数，用来**验证在线再量化是否覆盖了全部权重** | 仅 quark mxfp4 schemes |

`CopyNumelCounter` 这个设计值得注意：它用 `TorchDispatchMode` 拦截所有 `copy_`，如果搬运的元素数不等于预期，说明有权重漏转——是一种运行期自检。

### 9.3 投机解码的独立量化配置

draft 模型可以和 target 用不同的量化：`_resolve_explicit_draft_quant_config`（`weight_utils.py:241-259`）。

---

## 10. 关键文件索引

### 框架骨架

| 文件 | 作用 |
|------|------|
| `layers/quantization/__init__.py` | 注册表 + `get_quantization_config()` `:162` |
| `layers/quantization/base_config.py` | 四级抽象的前两级 |
| `layers/quantization/base_scheme.py` | 第四级 `BaseLinearScheme` `:11` / `BaseMoEScheme` `:56` |
| `configs/model_config.py` | checkpoint 探测 `_parse_quant_hf_config` `:1243`、总仲裁 `_verify_quantization` `:1504` |
| `layers/modelopt_utils.py` | ModelOpt 算法名→方法名映射 |
| `model_loader/weight_utils.py` | `get_quant_config` `:263` |
| `model_loader/checkpoint_quantization.py` | `CheckpointQuantSpec` `:25`、`resolve_checkpoint_quant_spec` `:80` |
| `model_loader/loader.py` | `_get_quantization_config` `:166-267`（17 处调用） |

### 挂载点（只有 5 个）

| 位置 | 层类型 |
|------|--------|
| `layers/linear.py:192` | 所有 linear |
| `layers/vocab_parallel_embedding.py:300` | embedding / lm_head |
| `layers/radix_attention.py:148` | attention（KV scale），紧接 `create_weights` `:149-150` |
| `layers/moe/fused_moe_triton/layer.py:447` | MoE（KT 分支） |
| `layers/moe/fused_moe_triton/layer.py:453` | MoE（普通分支） |

### 数值配方实现

`fp8.py`(2766行) / `modelopt_quant.py`(3037行) / `compressed_tensors.py`(1323行) / `awq/` / `gptq/` / `quark/` / `modelslim/` / `mxfp4.py` / `nvfp4_online.py` / `w8a8_int8.py` / `w8a8_fp8.py` / `blockwise_int8.py` / `w4afp8.py` / `humming.py` / `petit.py` / `bitsandbytes.py` / `gguf.py` / `auto_round.py` / `mlx_quant.py` / `expert_pack.py`

### kernel 工具

`fp8_utils.py` / `int8_utils.py` / `fp4_utils.py` / `marlin_utils{,_fp8,_fp4}.py` / `mxfp8_block_convert.py` / `humming_utils.py` / `petit_utils.py` / `rocm_mxfp4_utils.py`

### MoE 专线

`layers/moe/utils.py`（后端枚举/解析）/ `layers/moe/moe_runner/`（runner + 6 个 core + 15 个 QuantInfo）/ `layers/moe/fused_moe_triton/layer.py`（挂载）

### KV Cache

`mem_cache/kv_cache_dtype.py` / `layers/quantization/kv_cache.py` / `layers/quantization/fp4_kv_cache_quant_method.py` / `layers/quantization/kvfp4_tensor.py` / `mem_cache/memory_pool.py` / `arg_groups/kv_cache_hook.py`

---

## 11. 生产配方速查

| 目标 | 推荐组合 | 说明 |
|------|---------|------|
| H100/H800 通用降本 | ModelOpt FP8 或 block-FP8 checkpoint + `--fp8-gemm-backend auto`（落 DeepGEMM） | 最成熟，精度损失小 |
| Blackwell 极致吞吐 | NVFP4（`modelopt_fp4`）+ FlashInfer trtllm/cutedsl | 需 SM100 |
| 只有 A100/A800 | AWQ 或 GPTQ（W4A16）+ Marlin | 无 FP8 硬件支持 |
| 大 MoE 显存受限 | MXFP4 专家 + `--moe-runner-backend` 按平台选 | OSS GPT / DeepSeek 系常用 |
| MI300 (gfx942) | MXFP8 checkpoint（自动转 block-FP8）或直接 block-FP8 + AITER | MXFP8 原生不支持 |
| 长上下文吃 KV | `--kv-cache-dtype fp8_e4m3`，激进可上 KV4 | 先查 `kv_cache_hook.py` 白名单 |
| 单机跑超大 MoE | `--load-format expert_pack` | 不支持 EP / TP>1 |
| Ascend NPU | `modelslim` 或 NPU 版 `gptq`/`mxfp4` | 走 `AscendRunnerCore` |

**排查顺序建议**：启动日志里先看 `get_quantization_config_log_str()`（`model_config.py:1403`）打出的最终仲裁结果，确认落到了哪个 config；再看 MoE backend 和 GEMM backend 的解析结果。

---

## 12. 已知粗糙边缘

读代码时会撞上这些，提前知道能省时间：

1. **`MoeRunnerBackend.INTEL_XPU` 是死路**：`utils.py:232-233` 的 `is_intel_xpu()` 判断在一个 `raise` 之后，永不可达；`MoeRunner` 也没有对应分支，最终在 `runner.py:150` 抛 `NotImplementedError`。
2. **`Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL` CLI 可传但会崩**：`_dispatch_explicit_backend`（`fp8_utils.py:716`）没有它的分支，落到 `:775` 抛 `ValueError`。
3. **`can_auto_enable_marlin_fp8` 位置反直觉**：在 `fp8_utils.py:2069`，不在 `marlin_utils_fp8.py`。
4. **同一个环境变量两种读法**：`get_bool_env_var("SGLANG_FORCE_FP8_MARLIN")`（`fp8.py:469`）vs `envs.SGLANG_FORCE_FP8_MARLIN.get()`（`modelopt_quant.py:525`）；`SGLANG_CI_DISABLE_MOE_FUSED_FUNC` 更是裸 `os.environ`（`runner.py:171`）。
5. **`USE_TRITON_W8A8_FP8_KERNEL` 缺 `SGLANG_` 前缀**，违反 env-var-conventions。
6. **`SGLANG_FORCE_CK_W8A8` 只剩注释**（`fp8_utils.py:103`），实际入口是 `set_force_ck_w8a8()` `:107`。
7. **三个 GEMM 后端枚举各自独立解析**，互不校验一致性（见 §5.1）。
8. **`_verify_quantization` 里 `cfg_list` 长度 >1 的分支不可达**（`model_config.py:1572-1575`）。
9. **`MHATokenToKVPoolFP4` 及其工厂是死代码**（见 §7.3）。
10. **`_validate_quantize_and_serve_config`（`model_config.py:1472-1501`）永远 raise**（`:1496` `NotImplementedError`）——quantize-and-serve 还没实装。
11. **`Mxfp4W4A4Config.get_name()` 返回 `"mxfp4"`**，在 NPU 上遮蔽了 `Mxfp4Config`，同名不同类。
12. **CPU-AMX 白名单替换发生在所有 `override_quantization_method` 之前**，会静默改变用户显式指定的方法。
13. **所有 `*MoeQuantInfo` 仍是 `@dataclass`**（仓库规范要求 `msgspec.Struct`，这批是历史遗留）。
14. **`layer.scheme` 不能是 `nn.Module`**：会被 PyTorch 当子模块递归，导致权重加载错乱（见 §2.1）。
15. **`compatible_quantization_methods` 依赖 dict 迭代顺序**（`model_config.py:1546-1557`），改动顺序会改变仲裁结果。

---

## 附：如何快速定位一个量化路径

给定一个 checkpoint，想知道"它到底走了哪条路"，按这个顺序查：

```
1. checkpoint 里有什么？
   → config.json 的 quantization_config / compression_config
   → 或 hf_quant_config.json
   查：model_config.py:_parse_quant_hf_config (:1243)

2. 最终选了哪个方法名？
   → 启动日志 get_quantization_config_log_str() (model_config.py:1403)
   → 仲裁逻辑 _verify_quantization (:1504)

3. 方法名对应哪个 config 类？
   → __init__.py:BASE_QUANTIZATION_METHODS (:79) + 平台 overlay

4. config 实例怎么造的？
   → weight_utils.py:get_quant_config (:263)
   → loader.py:_get_quantization_config (:166)（字段注入、能力校验）

5. 每层挂了什么 method？
   → config.get_quant_method(layer, prefix)
   → 5 个挂载点之一（见 §10）

6. linear 走哪个 kernel？
   → XxxLinearMethod.apply → dispatch_xxx → 具体后端
   → 例：fp8.py:966 → fp8_utils.py:556 → :778 (auto)

7. MoE 走哪个 backend？
   → create_moe_runner 里的 AUTO 解析（见 §6.2 表）
   → MoeRunner.__init__ :90-150 挑 RunnerCore
   → apply 打包 *MoeQuantInfo → run()

8. KV 走哪条？
   → --kv-cache-dtype（dtype 路线）
   → 或 checkpoint KV scale（BaseKVCacheMethod 路线）
   → 平台校验 arg_groups/kv_cache_hook.py
```






