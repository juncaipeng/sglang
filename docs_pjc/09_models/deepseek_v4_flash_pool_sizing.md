# DeepSeek-V4-Flash 显存池空间构成与计算

> 模型：`/work/models/DeepSeek-V4-Flash-FP8/`
> 核心代码：`python/sglang/srt/model_executor/pool_configurator.py`（`DSV4PoolConfigurator`）、`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`、`python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`

---

## 1. 一句话结论

DSV4-Flash 的显存池分两类：

- **按 token 容量线性缩放的 KV / state 池**（7 项系数，见第 3 节），主体是 **c4 栈**（约占 66%），其中 `c4_kv_pool` 单项即最大（~40%）。
- **请求作用域的固定池**：`c128 compress_state`（第 4 节），per-token 系数为 0，单独按并发数结算；离线默认在 512 并发下约 **5 GB**，开 online compress 后降到 **~60 MB**。

---

## 2. 模型层结构（DSV4-Flash-FP8）

每层的 `compress_ratio ∈ {0, 4, 128}`（HF config `compress_ratios` 逐层配置）：

| ratio | 含义 | 层数 | 说明 |
|---|---|---|---|
| 4 (ca4) | CSA + C4Indexer 稀疏（top-512/1024） | 21 | 建 c4_kv / c4_indexer_kv / c4 state / indexer state |
| 128 (ca128) | HCA 读全块 | 20 | 建 c128_kv / c128 state |
| 0 | 纯 SWA 全精度近窗 | 2 | 仅 SWA |
| **合计** | | **43** | 全部层都进 swa_kv_pool |

关键常量（`kv_bytes = qk_nope 448 + 2·qk_rope 64 + scale/pad 8 = 584 B/token`，`head_dim = 448 + 64 = 512`）。

---

## 3. 七类池的 per-token 计算公式与占比

per-token 字节系数模型见 `pool_configurator.py:650-659`（`_get_bytes_per_full_token`），7 个加项对应 7 类池空间。按编号 ①→⑦ 排序：

| # | 物理池 | 公式项 | 代入 | 系数 (B/token/层) | 覆盖层 | 层数 | 计算 | B/token | 占比 |
|---|---|---|---|---|---|---|---|---|---|
| ① | swa_kv_pool | `swa_ratio·kv_bytes` | 0.1×584 | 58.4 | 全部层 | 43 | 58.4×43 | 2511 | 32.6% |
| ② | c4_kv_pool | `c4_frac·kv_bytes` | 0.25×584 | 146.0 | ca4 | 21 | 146×21 | **3066** | **39.8%** |
| ③ | c128_kv_pool | `(1/128)·kv_bytes` | 584/128 | 4.56 | ca128 | 20 | 4.56×20 | 91 | 1.2% |
| ④ | c4_indexer_kv_pool | `(1/4)·indexer_bytes` | 0.25×132 | 33.0 | ca4 | 21 | 33×21 | 693 | 9.0% |
| ⑤ | c4 compress_state | `swa_ratio·c4_state_ratio·c4_state_bytes` | 0.1×0.0625×8192 | 51.2 | ca4 | 21 | 51.2×21 | 1075 | 14.0% |
| ⑥ | c128 compress_state | `c128_state_ratio(=0)·…` | — | 0（请求级另算） | ca128 | 20 | 见第 4 节 | — | — |
| ⑦ | indexer compress_state | `swa_ratio·c4_state_ratio·c4_indexer_state_bytes` | 0.1×0.0625×2048 | 12.8 | ca4 | 21 | 12.8×21 | 269 | 3.5% |
| — | **合计/token** | | | | | | | **~7705 B** | 100% |

### 3.1 观察

- **c4 栈（②⑤④⑦）合计 5103 B/token ≈ 66%**，是显存主体。`c4_kv_pool` 单项即最大池（39.8%），超过 SWA。
- `c128_kv_pool` 靠 1/128 压缩几乎免费（1.2%）。
- ⑥ `c128 compress_state` 的 `c128_state_ratio = 0`（`pool_configurator.py:646`）——不随 token 容量缩放，是请求级固定开销，单独结算。

### 3.2 占比降序（便于看主次）

```
c4_kv       ██████████████████████ 39.8%  (3066 B)
swa_kv      ██████████████████     32.6%  (2511 B)
c4_state    ████████               14.0%  (1075 B)
c4_indexer  █████                   9.0%  ( 693 B)
indexer_st  ██                      3.5%  ( 269 B)
c128_kv     █                       1.2%  (  91 B)
```

---

## 4. ⑥ c128 compress_state：请求级固定开销

### 4.1 公式

`pool_configurator.py:687-699`（`_get_c128_state_fixed_bytes`）：

```
bytes = state_rows × last_dim × dtype_size × num_layers_ca128
```

- 离线：`state_rows = num_req_slots × ring_size(128) + 129`，ceil 到 128 倍；`last_dim = 2·head_dim = 1024`
- 在线：`state_rows = (num_req_slots + ring_size(1) + 1) × (1+mtp)`；`last_dim = 3·head_dim = 1536`
- `dtype = fp32(4B)`（`SGLANG_DSV4_COMPRESS_STATE_DTYPE`，默认 float32）
- `num_layers_ca128 = 20`

ring_size 见 `deepseek_v4_compress_state.py:34-48`：c128 离线=128、投机=256、online=1。

### 4.2 @ 512 并发（num_req_slots ≈ 513）

| 模式 | state_rows | last_dim | 每层 | 总占用 |
|---|---|---|---|---|
| **离线（默认，ring 128）** | 513×128 ≈ 65792 | 1024 | ~257 MB | **≈ 5.0 GB** |
| 离线+投机（ring 256） | 513×256 | 1024 | ~514 MB | ≈ 10.0 GB |
| **在线（ring 1）** | 515 | 1536 | ~3 MB | **≈ 60 MB** |

### 4.3 online compress 省显存原理

- 开关：环境变量 `SGLANG_OPT_USE_ONLINE_COMPRESS=1`（`environ.py:917`，默认关）；仅作用于 c128 层；HIP 强制失效；默认与投机/MTP 不兼容。
- 离线：per index 存 128 槽 raw token 环（`(kv,score)`，2·head_dim），攒满窗口批量压缩。
- 在线：per index 只留一份 running reduction 状态 `(max,sum,kv)`（3·head_dim），forward 就地更新，环塌缩成 1。
- 节省倍率 = (1/128 行) × (3/2 行宽) = **3/256 ≈ 1/85**（代码注释 `pool_configurator.py:633-635`）。
- 代价：行宽 +50%/槽、不走 fused 批量压缩路径、HIP 不支持、默认关投机。

---

## 5. 文件行号索引

| 内容 | 位置 |
|---|---|
| per-token 7 项系数模型 | `pool_configurator.py:620-660`（选中段 650-659） |
| c128 state 固定字节公式 | `pool_configurator.py:679-699` |
| c128 state 最终结算 | `pool_configurator.py:738-742` |
| c128_state_ratio = 0 | `pool_configurator.py:646` |
| ring_size 定义 | `deepseek_v4_compress_state.py:34-48` |
| KVAndScore 布局 | `deepseek_v4_compress_state.py:22-81` |
| online 常量 ONLINE_C128 | `deepseek_v4_memory_pool.py:31` |
| online 开关环境变量 | `environ.py:917-919` |
| head_dim (448/64) | `configs/deepseek_v4.py:79-80` |

> 相关文档：`docs/pjc_1/deepseek_v4_cache_management.md`（六子池总览）、`docs/pjc_1/dsv4_pd_disaggregation_request_lifecycle.md`（PD 传输）。
