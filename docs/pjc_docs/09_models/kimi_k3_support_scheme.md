# SGLang 支持 Kimi-K3 模型的方案系统梳理

> 撰写基准：`/work/repos/sglang` `main` 分支（本文写作时）。
> 关键事实：**K3 的模型本体代码不在 `main`**，而在公开的 `kimi-k3` 分支
> （`https://github.com/sgl-project/sglang/tree/kimi-k3`）。`main` 上只有接入骨架 +
> cookbook。本文基于两类可在 `main` 上验证的一手材料还原方案：
> ① `main` 上的 K3 接入点与共享基础设施（含文件行号）；
> ② `main` 上的近亲模型 `KimiLinear`（`models/kimi_linear.py` + `configs/kimi_linear.py`），
> 它用生产代码完整演示了 K3 的同一套「混合 KDA + MLA + MoE」骨架。
> cookbook 中未被 `main` 代码佐证的部分（例如具体部署配方数字），本文会显式标注为「来自 cookbook」。

---

## 0. 一页速览

| 维度 | Kimi-K3 |
|---|---|
| 定位 | Moonshot AI 首个万亿级开源模型，混合 MoE **视觉语言模型（VLM）** |
| 规模 | **2.8 万亿参数**；896 路由专家 + 1 共享专家，每 token 激活 **16 个专家** |
| 骨干层 | **93 层 = 69 层 KDA（线性注意力）+ 24 层 MLA**（外加 Attention Residual、Stable LatentMoE） |
| 注意力 | **Kimi Delta Attention（KDA，门控 delta-rule 线性注意力/类 SSM）** 与 **MLA**（DeepSeek 式多头潜在注意力）交替 |
| 权重量化 | **MXFP4**（Blackwell 走 FlashInfer MXFP4/trtllm-gen SiTU，其它平台走 Marlin W4A16，短上下文批量走 MegaMoE） |
| 上下文 | 1M token 窗口 + 前缀缓存 |
| 视觉 | MoonViT 式编码器，14×14 patch，**仅支持图像输入**（拒绝视频/音频） |
| 推理模式 | **恒开思考**，`reasoning_effort` = low/high/max（默认 max） |

**一句话方案**：K3 = 「**混合注意力运行时（KDA + MLA）** + **双显存池（recurrent state 池 + 分页 MLA KV 池）** + **MXFP4/MegaMoE 专家系统** + **多轴并行（TP/EP/DP/DCP/PP）** + **DSPARK 投机** + **MoonViT VLM** + **PD/EPD 分离**」。以上每个支柱在 `main` 上都已有基础设施，K3 只是把它们组合并放大。

---

## 1. `main` 分支上的 K3 现状：只有骨架

在 `main` 上 grep `Kimi-K3 / kimi_k3 / KimiK3`，命中文件仅以下几类（**均非模型本体**）：

| 文件 | 作用 | 类别 |
|---|---|---|
| `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` | 部署 cookbook（架构描述 + 各硬件配方 + 高级特性） | 文档 |
| `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` | cookbook 的部署命令生成配置 | 文档 |
| `docs_new/src/snippets/_kimi_k3_mamba_ratio_calculator.jsx` | `--mamba-full-memory-ratio` 在线计算器 | 文档 |
| `python/sglang/srt/entrypoints/http_server.py:2058-2067` | **启动 warmup 钩子**：识别 `KimiK3ForConditionalGeneration` / `model_type=="kimi_k3"`，用 448×448 图像预热其原生 32×32 patch 网格 | 接入骨架 |
| `test/registered/unit/entrypoints/test_server_warmup.py:30` | 上述钩子的单测 | 接入骨架 |
| `.github/workflows/release-docker-amd-rocm720-nightly.yml` | AMD ROCm 7.2 K3 nightly 镜像 | CI |

**没有** `models/kimi_k3.py`、**没有** `configs/kimi_k3.py`、**没有** `KimiK3ForConditionalGeneration` 的架构注册（`model_config.py` 里只有 `KimiK25/KimiVL/KimiLinear`）。

> 结论：要读 K3 的 `forward`，必须切到 `kimi-k3` 分支。但要理解「SGLang 用什么机制支撑 K3」，读 `main` 就够了——因为这些机制都是通用基础设施，且被 `KimiLinear` 生产化验证过。

---

## 2. 近亲参照系：`KimiLinear`（同一套骨架的生产实现）

`main` 上的 `KimiLinearForCausalLM`（`models/kimi_linear.py`）与 `KimiLinearConfig`
（`configs/kimi_linear.py`）用完整生产代码演示了 K3 的核心骨架——**逐层在 KDA 与 MLA 之间二选一**：

```
KimiDecoderLayer.__init__  (kimi_linear.py:428)
   ├── if config.is_kda_layer(layer_idx):        # configs/kimi_linear.py:139
   │       self.self_attn = KimiDeltaAttention(...)   # 线性注意力  (kimi_linear.py:182)
   └── else:
           self.self_attn = KimiMLAAttention(...)      # = DeepseekV2AttentionMLA (kimi_linear.py:48/452)
```

关键设计点（`configs/kimi_linear.py`）：

- `linear_attn_config["kda_layers"]` / `["full_attn_layers"]`：**逐层布局表**，决定哪些层是 KDA、哪些是 MLA（K3 = 69 KDA + 24 MLA）。
- `is_kda_layer(i)`（:139）、`linear_layer_ids`（:146）、`full_attention_layer_ids`（:150）：层类型查询。
- `is_mla`（:114）/`is_moe`（:125）/`is_linear_attn`（:129）：能力位。
- `mamba2_cache_params`（:154）→ `KimiLinearCacheParams` / `KimiLinearStateShape`：**KDA recurrent state 的形状**由 `attn_tp_size × num_heads × head_dim × short_conv_kernel_size` 决定——这正是「state 池按最坏情况预留、决定并发上限」的根源。
- `KimiMLAAttention = DeepseekV2AttentionMLA`（直接复用 DeepSeek 的 MLA 实现，:48）。

> K3 相对 KimiLinear 的增量：① 视觉塔（MoonViT）+ VLM 输入路径；② 规模放大到 2.8T/896 专家；③ MXFP4 权重；④ 恒开思考 + K3 专属 parser。**注意力/显存/MoE/并行的底座是同一套。**

复用矩阵（K3 从既有组件继承什么）：

| 能力 | 继承自 | `main` 上位置 |
|---|---|---|
| KDA 线性注意力层 | KimiLinear | `models/kimi_linear.py:182 KimiDeltaAttention` |
| MLA 注意力层 | DeepSeek-V2/V3 | `models/deepseek_v2.py:DeepseekV2AttentionMLA` |
| 逐层混合调度 | KimiLinear | `configs/kimi_linear.py:139 is_kda_layer` |
| KDA state 池形状 | KimiLinear | `configs/mamba_utils.py KimiLinearStateShape` |
| MoE / MXFP4 / MegaMoE | DeepSeek-V2/V4 | `layers/moe/*`、`layers/quantization/modelopt_quant.py`、`layers/moe/mega_moe.py` |
| DCP（解耦 MLA KV） | DeepSeek 系 | `--dcp-size`、`layers/cp/*`（见记忆库 CP/DCP 笔记） |
| DSPARK 投机 | DeepSeek-V4 | `arg_groups/speculative_hook.py`、`layers/attention/cutedsl_mla_backend.py` |
| VLM warmup 钩子 | 新增（K3 专属） | `entrypoints/http_server.py:2058` |

---

## 3. 混合注意力运行时：KDA + MLA

### 3.1 为什么是混合

- **MLA 层**：全局注意力，精确、支持前缀缓存与超长上下文；代价是 KV 随序列线性增长（分页 KV 池）。
- **KDA 层**：门控 delta-rule 线性注意力，**状态是固定大小的 recurrent state（类 SSM）**，与序列长度无关，对 1M 长上下文的显存与吞吐友好；代价是表达力弱于全注意力。
- 交替排布（69:24）在长上下文吞吐与精度间取平衡——这与 KimiLinear 的思路一致，K3 把比例与规模放大。

### 3.2 KDA 后端与内核分派（`layers/attention/linear/kda_backend.py`）

KDA 后端基于 `MambaAttnBackendBase`（线性注意力/SSM 家族的公共基类，`hybrid_linear_attn_backend.py`）。核心是 `KDAKernelDispatcher`（:35）**按 forward 模式选内核**：

| 模式 | 语义 | 可选内核 |
|---|---|---|
| prefill / extend | chunk 扫描（`causal_conv1d_fn` + chunk KDA） | Triton / FlashKDA / CuTe DSL |
| decode | 单步 recurrent 更新（`causal_conv1d_update`） | Triton / CuTe DSL / **FlashInfer recurrent_kda（SM100）** |
| target_verify（投机） | 验证草稿 token（chain + tree） | FlashInfer 自验 / 其余走 **Triton 融合 KDA verify**（tree 参考实现） |

平台适配：`causal_conv1d_update` 在 NPU/CPU 上有各自覆盖（:22-29）；CuTe DSL / FlashInfer 内核强制要求 CUDA（:48/:58）。

> cookbook 佐证（§2 attention backend）：Blackwell 上三个注意力旋钮（prefill/decode/verify）**联合解析**为一套（`trtllm_mla` 全线，DCP 下 decode/verify 交给 `cutedsl_mla`）；H100/H200 decode 钉 `flashmla`。

---

## 4. 双显存池：recurrent state 池 + 分页 MLA KV 池

这是 K3 服务方案里**最独特、最需要调参**的一环。因为骨干里有两种截然不同的注意力，显存被切成两块：

```
                 静态显存 (mem_fraction_static)
        ┌────────────────────────┴─────────────────────────┐
        │                                                    │
  KDA state 池 (最坏情况预留)                        分页 MLA KV 池
  - 每请求固定 S 个 state slot                       - 随 token 增长，按页分配
  - 决定「并发上限 / 可入场请求数」                  - 决定 max_total_num_tokens
  - 不被 DP/EP/DCP 分片！                            - DCP 可跨 rank 去重分片
        └──────────────── 由 --mamba-full-memory-ratio 划分 ────────────────┘
```

### 4.1 一根旋钮 `--mamba-full-memory-ratio`

`server_args.py:2458` 定义。它是「KDA state 池 : MLA KV 池」的比例。cookbook 给出平衡值公式（§Mamba ratio calculator）：

```
ratio = (S + D) * state_bytes / (L * (mla_kv_bytes / DCP + draft_kv_bytes))
```

- `S`：每请求 KDA state slot 数，随 radix cache 策略变化——`extra_buffer=5`、`extra_buffer_lazy=4`、`no_buffer=3`、关 radix=1。
- `D`：投机验证中间态，DSPARK 下 = 块大小+1（默认 7 → 8），关投机=0。
- `state_bytes`：单 slot 字节数，由 K3 固定几何 + attention-TP 宽度 + SSM dtype 决定。
- `mla_kv_bytes`：单 token MLA 潜在 KV 字节（受 KV dtype 影响），DCP 按 rank 数分片。
- `L`：平均请求总长（输入+输出）。

### 4.2 相关 flag（均在 `server_args.py`）

| flag | 位置 | 作用 |
|---|---|---|
| `--mamba-ssm-dtype` | :2432 | KDA state dtype；`bfloat16` 约减半 state 字节（换并发）；投机下 bf16 会使 KDA verify 从融合内核回退到 Triton（:5906/:5938） |
| `--mamba-full-memory-ratio` | :2458 | 两池划分比 |
| `--mamba-radix-cache-strategy` | :2466 | `extra_buffer` / `extra_buffer_lazy`（每请求 4 slot 而非 5，:5464）/ `no_buffer` |
| `--kv-cache-dtype fp8_e4m3` | — | MLA KV 每 token 字节减半（换上下文长度）；PD 下两侧须一致 |

capacity 心智模型（cookbook §3.6）：**KDA state 池是并发天花板**——DP/EP/DCP 都不分片它，只有 attention-TP 宽度、SSM dtype、cache 策略能改每 GPU 账单。MLA KV 反而便宜（fp8 缩、DCP 去重）。

### 4.3 HiCache 分层

cookbook §3.3：K3 的混合 HiCache 把**分页 MLA KV 和 KDA/mamba state 一起**分层到 L1(GPU)/L2(host)/L3(Mooncake)。DCP 配方下 host 层尚未完全 DCP 感知（L3 恒可用；L1+L2 在投机开启时需退回纯 TP）。

---

## 5. MoE 专家系统：MXFP4 权重 × 三条内核路径

K3 权重以 **MXFP4** 发布，896 路由专家 + 1 共享，top-16。运行时按平台选 MoE 内核：

| 平台/场景 | MoE runner | 说明 |
|---|---|---|
| Blackwell（默认，装了 SiTU cubin pool） | **FlashInfer MXFP4**（W4A8，trtllm-gen SiTU 预编译 kernel） | cookbook §2；cubin pool 需装 `SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL` |
| Blackwell（无 cubin pool）/ H100 / H200 | **Marlin（W4A16）** | 兜底 |
| 短上下文批量吞吐 | **MegaMoE**（`--moe-a2a-backend megamoe --moe-runner-backend deep_gemm`） | 见记忆库 MegaMoE 笔记：SM100 融合式 EP-MoE 大内核，强制 EP==TP |

EP all-to-all 后端：`--moe-a2a-backend {deepep, megamoe}`（见记忆库 DeepEP 笔记）。大规模预设默认 MegaMoE on deep_gemm（§3.6）。

`main` 上支撑点：`layers/moe/mega_moe.py`、`layers/quantization/modelopt_quant.py`（MXFP4/W4A16）、DeepEP token dispatcher（`token_dispatcher/deepep.py`）。

---

## 6. 多轴并行

| 轴 | flag | 对 K3 的作用 |
|---|---|---|
| TP | `--tp-size` | 基本切分；attention-TP 宽度直接影响 KDA state 每 GPU 字节 |
| EP | `--ep-size` | MoE 专家并行；大规模走 megamoe（EP==TP） |
| DP-attention | `--enable-dp-attention --dp-size` | 数据并行注意力，配合 `--enable-dp-lm-head` |
| **DCP** | `--dcp-size` | **唯一能分片 TP-复制的 MLA KV 的轴**（decode context parallel）；Low-Latency 不用，Balanced/High-Throughput 用。DCP 下不能开 `--enable-symm-mem`（decode-graph 正确性） |
| PP | `--pp-size` | 长上下文 prefill 专用（B200 `--pp-size 8 --tp-size 1`）：P2P 重叠下一 microbatch，每 stage 独占整层（干净切 KV 与 state）。**PP>1 时禁 overlap 调度、禁 DSPARK** |

关键约束（cookbook §3.6 / §2）：
- KDA state 池不被任何并行轴分片，只随 attention-TP 变。
- DCP 去重 MLA KV：并发上限 +72%，ITL ~1.8×，适合 ctx≥16K。
- 不要在 DCP 上叠 EP：a2a buffer 会吃掉 DCP 省出的 KV。

---

## 7. 投机解码：DSPARK / DFLASH

- **DSPARK**（`--speculative-algorithm DSPARK`）：K3 的主投机方案，每步提 **7 个草稿 token**（cookbook §1/§2）。可叠在任意 `pp_size == 1` 配方上。DSPARK 在 KDA verify 上占「块大小+1=8」个中间 state（计入 `--mamba-full-memory-ratio` 的 `D`）。未设 `--max-running-requests` 时投机下重置为 48。
- **DFLASH**：cookbook 称暂无公开 draft checkpoint。
- **ReplaySSM**（`--enable-linear-replayssm-spec`）：把 verify 中间态折叠成 per-slot ring，令上式 `D` 归零。

`main` 支撑点：`arg_groups/speculative_hook.py`（DSPARK 参数注入）、`layers/attention/cutedsl_mla_backend.py` / `deepseek_v4_backend.py`（DSPARK 分支）、`managers/overlap_utils.py`、`scheduler_components/metrics_reporter.py`（投机指标）。整体投机框架见记忆库「投机解码」笔记（统一 draft→verify→draft_extend 三阶段，V2 worker）。KDA 侧的投机验证内核见 §3.2 的 `verify_kernel`。

---

## 8. VLM 视觉路径（MoonViT）

- **视觉塔**：MoonViT 式编码器，原生 **14×14 spatial patch**；448×448 图 → 32×32 patch 网格无 padding（这正是 `main` warmup 钩子选 448×448 的原因，`http_server.py:2043-2067`）。
- **仅图像**：开源 K3 serving contract 目前只支持图像输入，拒绝视频/音频（cookbook §3.5）。
- **MM encoder DP 内建**：K3 把整图跨 TP rank 分片，故 `--mm-enable-dp-encoder` **不要设**。
- **特征传输**：单节点 `--mm-feature-transport cuda_ipc`（跳过 CPU 往返，占 `SGLANG_MM_FEATURE_CACHE_MB` HBM）；多节点用 `cpu`。
- **ViT BCG**（`SGLANG_VIT_ENABLE_CUDA_GRAPH`）：默认关，仅 ViT-only/EPD 编码器且图形状重复时开。
- **EPD**：`kimi-k3` 分支支持——`--encoder-only` 视觉角色 + `--language-only` prefill 角色 + 常规 decode 角色。

---

## 9. PD / EPD 分离

因为 K3 是混合模型，PD 传输**同时搬运分页 MLA KV 和 KDA recurrent state**（cookbook §3.4）：

- **传输后端**：配方发 **NiXL**（RDMA），Playground 可选 Mooncake。
- **端口**：prefill 30000、decode 30100；`--prefill <host>:30000 8998` 里的 `8998` 必须等于 `--disaggregation-bootstrap-port`，否则只有 decode worker 注册。
- **Decode state 池**：chunk cache，每请求 1 slot，`--mamba-radix-cache-strategy` 失效；`--disaggregation-decode-extra-slots` 要显式钉住（不钉时 <32 请求默认 2×batch，>32 请求归 0）。
- **Router**：`sglang_router.launch_router --pd-disaggregation --prefill ... 8998 --decode ...`。

`main` 支撑点：`disaggregation/*`（common/conn、prefill、decode、encode_server）、`disaggregation/utils.py`。混合模型 KV+state 双通道传输的机制模式，与记忆库「DSV4 PD 分离」笔记中「六子池→两通道」思路同源（KV 主通道 + state 状态通道）。

---

## 10. 请求语义层：parser 与启动

- **恒开思考**：K3 总是先思考；`kimi_k3` reasoning parser 把 thinking 放进 `message.reasoning_content`，答案放 `message.content`。`reasoning_effort` = low/high/max（默认 max）。
- **工具调用**：`--tool-call-parser kimi_k3`，结构化 `message.tool_calls`；因为是思考模型，后续轮 `reasoning_content` 与 `content` 都可能有内容。
- **推荐采样**（模型固定，仅信息性）：`temperature=1.0, top_p=0.95, presence_penalty=0, frequency_penalty=0`。
- **启动 warmup**（`main` 唯一的 K3 运行时代码）：`_get_vlm_warmup_image_base64`（`http_server.py:2048`）识别 K3 后用 448×448 图预热原生 vision patch 网格，避免首个真实图像请求承担一次性视觉初始化开销。

> 注意：`kimi_k3` 的 reasoning/tool parser **实现体在 `kimi-k3` 分支**；`main` 上只有 cookbook 引用。

---

## 11. 部署方案矩阵（来自 cookbook，需实测复核）

> cookbook 明确标注：每个 cell 目前均为「Final Verification In Progress」，数字仅作起点。

**PD Mode**：`Unified`（合跑）/ `Prefill` / `Decode`。**Strategy**：Low-Latency（纯 TP，无 DCP，聊天）/ Balanced（保精度默认，TP16/DCP16 on B200/GB200）/ High-Throughput（大规模）/ Long-Context（B200 专属 TP8/PP2）。

| 平台 | 拓扑 | 要点 |
|---|---|---|
| B300 1×8 | TP8(+DCP8) | 精度优先默认 |
| GB300 2×4 | TP8/DCP8 | MNNVL + cuMem 自动探测 |
| B200 2×8 | TP16(+DCP16)；Long-Context TP8/PP2 | Long-Context 关 DSPARK |
| GB200 4×4 | TP16/DCP16 | MNNVL 自动 |
| H200 2×8 | TP16/EP16 + symm-mem | 需导出跨节点 NIC |
| H100 4×8 | TP32/EP32，Marlin + FlashMLA | SM90a 构建，headroom 最小(80GB) |
| MI350X/355X 1×8 | TP8 ROCm/AITER | AITER A8W4 FlyDSL MoE，Triton attention，支持 DSPARK |

**大规模两预设**（Blackwell，N=8k GPU）：
- **Peak Throughput**（`dp=k`，attention-TP 8）：state 8 路分片，KDA all-reduce 留在 8-GPU 节点内；`fp8_e4m3` KV 是硬需求。
- **Peak Capacity（+DCP8）**：去重 MLA KV，并发上限 +72%，ITL ~1.8×，适合 ctx≥16K。

典型大规模命令（32 GPU，来自 cookbook §3.6）：
```bash
SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=20480 \
sglang serve --trust-remote-code --model-path moonshotai/Kimi-K3 \
  --tp-size 32 --ep-size 32 --enable-dp-attention --dp-size 4 --enable-dp-lm-head \
  --moe-a2a-backend megamoe --moe-runner-backend deep_gemm \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy --mem-fraction-static 0.92 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
```

---

## 12. `main` 上 K3 相关文件索引

| 文件 / 位置 | 内容 |
|---|---|
| `entrypoints/http_server.py:2043-2074` | K3 VLM 启动 warmup（448×448 图，唯一运行时代码） |
| `test/registered/unit/entrypoints/test_server_warmup.py:30-33` | warmup 钩子单测 |
| `docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx` | 部署 cookbook（架构 + 配方 + 高级特性） |
| `docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx` | 命令生成配置（含「93 层 = 69 KDA + 24 MLA」注释） |
| `docs_new/src/snippets/_kimi_k3_mamba_ratio_calculator.jsx` | mamba-full-memory-ratio 计算器 |
| `models/kimi_linear.py` | **近亲参照**：混合 KDA+MLA+MoE 骨架的生产实现 |
| `configs/kimi_linear.py` | 逐层混合布局 / state 形状配置的参照 |
| `layers/attention/linear/kda_backend.py` | KDA 线性注意力后端与内核分派 |
| `layers/attention/hybrid_linear_attn_backend.py` | 线性注意力/SSM 家族公共基类 |
| `server_args.py:2432-2466` | `--mamba-ssm-dtype` / `--mamba-full-memory-ratio` / `--mamba-radix-cache-strategy` |
| `layers/moe/mega_moe.py` | MegaMoE（短上下文批量吞吐） |
| `arg_groups/speculative_hook.py` | DSPARK 参数注入 |

---

## 13. 快速上手清单（读代码的顺序）

1. 先读本文 §2 + `configs/kimi_linear.py` `is_kda_layer` / `mamba2_cache_params` → 理解「逐层混合 + state 形状」。
2. 读 `models/kimi_linear.py:428 KimiDecoderLayer` → 看 KDA/MLA 二选一如何组装。
3. 读 `layers/attention/linear/kda_backend.py:35 KDAKernelDispatcher` → 理解 prefill/decode/verify 三模式内核。
4. 读 `server_args.py:2458` 附近 + cookbook §Mamba calculator → 掌握双池划分。
5. 需要看 K3 本体（VLM 塔、2.8T MoE、MXFP4 加载、K3 parser）时，切到 `kimi-k3` 分支的 `models/kimi_k3.py` / `configs/kimi_k3.py` / parser 实现。

---

## 附：待确认 / 风险点

- 本文对 K3 **本体** `forward` 的描述来自 cookbook 二手材料，未经 `main` 代码验证；`kimi-k3` 分支 merge 进 `main` 后需按实际代码复核 §3–§10 的行号与细节。
- cookbook 所有部署数字均为「验证中」，投产前必须在自有 workload 上实测吞吐与精度。
- `KimiK3ForConditionalGeneration` 架构目前未在 `main` 的 `model_config.py` 注册，加载 K3 需 `kimi-k3` 分支代码 + `--trust-remote-code`。

