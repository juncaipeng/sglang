# DeepSeek V4.1 mHC（多流超连接）机制详解

> mHC 代码为 V4 / V4.1 共用（`models/deepseek_v4.py` + `kernels/ops/layernorm/mhc.py`）。
> **本文的具体配置数值取自实际部署的 `DeepSeek-V4-Flash-FP8`**
> （[/work/models/DeepSeek-V4-Flash-FP8/config.json](/work/models/DeepSeek-V4-Flash-FP8/config.json)，`model_type=deepseek_v4`）。
> 记号约定：`T`=token 数，`D`=hidden_size=**4096**，`M=hc_mult=4`（残差流条数），`hc_dim=M·D=16384`。
>
> ⚠️ 注意：代码里若干 mHC 融合快路径的守卫写死 `D==5120` / `hc_dim==20480`
> （如 TileLang `mhc_post_split_h` [deepseek_v4.py:2953](../../../python/sglang/srt/models/deepseek_v4.py#L2953)、
> fused-scale 写回 [:3248](../../../python/sglang/srt/models/deepseek_v4.py#L3248) 的 `x_flat.shape[1]==20480`），
> 那是**另一个 D=5120 的 V4.1 变体**（带视觉/Engram/ratio 1/2）。
> **V4-Flash 的 D=4096 不满足这些 5120 守卫，因此走通用融合路径**（FlashInfer `mhc_pre_big_fuse` / DeepGEMM tf32 / torch 参考等对 D 通用的实现，见 §8）。

---

## 1. 背景与动机：从单残差流到多流

传统 Transformer 每个子层只有一条残差流（residual stream）：

```
x = x + sublayer(x)
```

mHC（multi-stream Hyper-Connections，多流超连接）把残差扩展成 **M 条并行残差流**。核心思想：

1. **多条残差流**：模型内部同时维护 `M=4` 条残差流，张量形状 `[T, M, D]`。
2. **进子层前 mix**：用可学习权重把 4 条流"混合"成 1 条 `[T, D]`，送进 attention / MoE。
3. **出子层后 combine**：子层输出用一个**双随机(doubly-stochastic)矩阵** 把结果"散射"回 4 条流。

**为什么要双随机矩阵？** 双随机矩阵每行每列和都≈1，等价于对 4 条流做一次"质量守恒"的重排——信息可以在流之间自由重组，但整体幅度既不爆炸也不衰减。这是让极深网络稳定训练/推理的关键。

---

## 2. 关键配置

来源：实际模型 [config.json](/work/models/DeepSeek-V4-Flash-FP8/config.json)（`model_type=deepseek_v4`）；
默认值定义见 [configs/deepseek_v4.py:134](../../../python/sglang/srt/configs/deepseek_v4.py#L134)。

| 参数 | V4-Flash 实际值 | 来源 | 含义 |
|---|---|---|---|
| `hidden_size` (D) | **4096** | config.json | 隐藏维；`hc_dim = M·D = 16384` |
| `hc_mult` (M) | 4 | config.json | 残差流条数 |
| `hc_sinkhorn_iters` | 20 | config.json | Sinkhorn 双随机归一化迭代次数 |
| `hc_eps` | 1e-6 | config.json | mHC 内部数值稳定项 |
| `rms_norm_eps` | 1e-6 | config.json | mix 前 RMS 归一化 eps |
| `num_hidden_layers` | 43 | config.json | Decoder 层数（每层 2 个 mHC 子层边界） |
| `hc_pre_from_prev_sublayer` | **False** | 未在 config.json 设置 → 用 dataclass 默认（[deepseek_v4.py:135](../../../python/sglang/srt/configs/deepseek_v4.py#L135)） | 是否让本层从上一子层延迟接收 mHC 状态 |
| `q_head_norm` | **True** | 同上默认（[:136](../../../python/sglang/srt/configs/deepseek_v4.py#L136)） | 注意力侧开关，与 mHC 无直接关系 |

内部常量 `_MHC_POST_MULT_VALUE = 2.0`（post 门的放大系数）。

> **与 D=5120 的 V4.1 变体对比**：那个变体 `model_type=deepseek_v41`，config 覆盖
> `hc_pre_from_prev_sublayer=True`、`q_head_norm=False`（[deepseek_v41.py:47](../../../python/sglang/srt/configs/deepseek_v41.py#L47)）。
> 本文的 V4-Flash 取 dataclass 默认，故 **`hc_pre_from_prev_sublayer=False`**——即
> 采用每个子层独立产生并消费 mHC 状态的默认接线（§7 相应说明）。

---

## 3. 残差流的数据布局与生命周期

残差流张量形状 **`[T, M, D] = [T, 4, 4096]`**（入口 `hc_expand` 把 `[T,D]` 复制成 `[T, M·D]=[T,16384]`）。

```
                     ┌──────────────── 每个 decoder layer ────────────────┐
[T,D] ──hc_expand──▶ [T,4,D] ─▶ [attn 子层] ─▶ [ffn/MoE 子层] ─▶ [T,4,D] ─▶ … ─▶ hc_contract ─▶ [T,D] ─▶ head
                                pre→Attn→post   pre→MoE→post
```

- **入口 `hc_expand`**（[mhc.py:1774](../../../python/sglang/kernels/ops/layernorm/mhc.py#L1774)）：`x.repeat(1, M)`，把 `[T,4096]` 复制成 `[T,16384]`，即 4 条流初始完全相同。
- **出口 `hc_contract`**（[mhc.py:1778](../../../python/sglang/kernels/ops/layernorm/mhc.py#L1778)）：`x.unflatten(-1,(M,-1)).mean(-2)`，把 4 条流平均塌缩回 `[T,4096]`。
- **最终 head**（`hc_head_torch`，[deepseek_v4.py:581](../../../python/sglang/srt/models/deepseek_v4.py#L581)）：末层做一次带可学习权重的加权求和。

> 注：[deepseek_v4.py:2973](../../../python/sglang/srt/models/deepseek_v4.py#L2973) 有 `residual.shape==(T,4,5120)` 的断言，
> 但它被包在 `x.shape[1]==5120` 的融合守卫里，**只对 D=5120 变体生效**；V4-Flash（D=4096）走通用路径，不触发该断言。

**每个 decoder layer 含两个子层**（attention 与 MoE/FFN），每个子层各有独立的一套 mHC pre/post 权重。

### 3.1 各算子 I/O shape 与 dtype 速查（V4-Flash：D=4096, M=4, M·D=16384；T=token 数）

> **dtype 总原则**：残差流与子层输入/输出这些“数据张量”走 **bf16**（模型激活精度）；mHC 的 6 个权重参数（fn/base/scale）是 **fp32**；`hc_pre`/`hc_post`/`hc_head` 的**内部数值计算全程 fp32**（RMSNorm 平方和归约、24 logits 投影、sigmoid 门控、Sinkhorn 20 迭代），门控中间量 `pre/post/comb` 保持 **fp32**，只在写回残差流/子层输入时 `.to(bf16)`。这与 §2 “fp32 数值稳定”一致——mHC 是数值敏感算子，故所有归约在 fp32。

| 算子 | 代码 | 输入 (shape, dtype) | 输出 (shape, dtype) | 内部精度 |
|---|---|---|---|---|
| `hc_expand` | [mhc.py:1774](../../../python/sglang/kernels/ops/layernorm/mhc.py#L1774) | `x [T,D] bf16` | `[T,M·D]=[T,16384] bf16`（4 流相同） | — (纯 repeat) |
| `hc_contract` | [mhc.py:1778](../../../python/sglang/kernels/ops/layernorm/mhc.py#L1778) | `x [T,M·D] bf16` | `[T,D] bf16`（4 流均值） | mean 在原 dtype |
| `hc_pre` | [mhc.py:1782](../../../python/sglang/kernels/ops/layernorm/mhc.py#L1782) | `residual [T,M,D] bf16`；参数 `fn (24,16384)`、`scale (3,)`、`base (24,)` **fp32** | `y [T,D] bf16`（子层输入）；`post [T,M,1] fp32`；`comb [T,M,M]=[T,4,4] fp32`；`norm_fused bool` | **fp32**（`residual.float()`→RMSNorm→投影→sigmoid→Sinkhorn；`y` 末尾 `.to(bf16)`） |
| `hc_post` | [mhc.py:1823](../../../python/sglang/kernels/ops/layernorm/mhc.py#L1823) | `x [T,D] bf16`（子层输出）；`residual [T,M,D] bf16`；`post [T,M,1] fp32`；`comb [T,M,M] fp32` | `out [T,M,D] bf16`（新残差流，`type_as(x)`） | fp32 门控 × bf16 数据，结果 `.to(bf16)` |
| `hc_head` | [deepseek_v4.py:581](../../../python/sglang/srt/models/deepseek_v4.py#L581) | `x [T,M,D] bf16`；参数 `fn (M,16384)`、`scale (M,)`、`base (1,)` **fp32** | `[T,D] bf16`（末层加权塌缩） | **fp32** |

> - `post` 的实际存储形状是 `[T,M,1]`（`_mhc_pre_torch` 里 `post.unsqueeze(-1)`，[mhc.py:1820](../../../python/sglang/kernels/ops/layernorm/mhc.py#L1820)），广播乘 `x [T,1,D]`。
> - `residual` / `y` / `x` / `out` 的 bf16 是继承模型激活 dtype；若模型以别的激活 dtype 运行，这些张量随之变化，但门控与归约恒 fp32。
> - 融合后端（FlashInfer/TileLang/DeepGEMM tf32 等，§8）I/O 契约相同，只是把上表多步在片上用 fp32 累加融合成一个 kernel，对外仍是 bf16 进 / bf16 残差出、fp32 门控暂存。

---

## 4. 权重参数（每个子层一套）

`make_hc_mixing_params`（[deepseek_v4.py:553](../../../python/sglang/srt/models/deepseek_v4.py#L553)）为每个子层生成 6 个 **fp32** 参数：

```python
mix_hc = (2 + hc_mult) * hc_mult   # = (2+4)*4 = 24
hc_dim = hc_mult * hidden_size     # = 4*4096 = 16384
```

| 参数 | 形状 | dtype | 作用 |
|---|---|---|---|
| `hc_*_fn` | `(24, 16384)` | fp32 | 从展平残差投影出 24 个混合 logits 的矩阵 |
| `hc_*_base` | `(24,)` | fp32 | bias |
| `hc_*_scale` | `(3,)` | fp32 | 分别缩放 pre / post / comb 三组 logits |

`*` 分别为 `attn` 与 `ffn`，即 attention 子层和 FFN 子层各一套。

**24 个 logits 的切分**：`pre(4) + post(4) + comb(16=4×4)`。
这正是 `mix_hc = (2+M)·M` 的来源：2 个长度 M 的向量（pre、post）+ 1 个 M×M 矩阵（comb）。

末层另有 head 参数 `make_hc_head_params`（[deepseek_v4.py:570](../../../python/sglang/srt/models/deepseek_v4.py#L570)）：`fn (M, M·D)=(4,16384)`、`scale (M,)`、`base (1,)`。

---

## 5. hc_pre：把 4 条流混合成 1 条

参考实现：`_mhc_pre_torch`（[mhc.py:1782](../../../python/sglang/kernels/ops/layernorm/mhc.py#L1782)）+ `_hc_split_sinkhorn_torch`（[mhc.py:195](../../../python/sglang/kernels/ops/layernorm/mhc.py#L195)）。派发入口 `hc_pre`（[deepseek_v4.py:2761](../../../python/sglang/srt/models/deepseek_v4.py#L2761)）。

`hc_pre` 的职责不止是把 4 条流压成 1 条。一次调用会同时完成三件事：

1. 用 `pre` 把当前 4 条残差流汇聚为本子层的单流输入 `y`；
2. 生成并暂存 `post`，控制本子层输出稍后向 4 条流各注入多少；
3. 生成并暂存 `comb`，控制旧的 4 条残差流在 `hc_post` 时如何重排。

因此可以把它理解成“**为当前子层开一个 mHC 边界**”：`pre` 当场消费，`post/comb` 要等当前 attention 或 FFN 完成后，由与它配对的 `hc_post` 消费。

### 5.1 输入输出与一次调用的生命周期

对第 `l` 个子层，记进入边界的残差流为 `R_l`：

| 名称 | shape（M=4） | dtype | 何时使用 |
|---|---|---|---|
| `R_l` / `residual` | `[T,4,D]` | bf16 | 同时用于生成门控、汇聚 `y_l`，并原样保留到 `hc_post` |
| `pre_l` | `[T,4]` | fp32 | 在当前 `hc_pre` 内立即汇聚 4 条流 |
| `y_l` | `[T,D]` | bf16 | 送入本子层；是否已做后续 RMSNorm 由 `norm_fused` 表示 |
| `post_l` | 模型层接口为 `[T,4]`，底层参考实现为 `[T,4,1]` | fp32 | 暂存到本子层结束，控制子层输出注入 4 条新流 |
| `comb_l` | `[T,4,4]` | fp32 | 暂存到本子层结束，控制旧流到新流的重排 |
| `norm_fused` | 标量 `bool` | — | 通知调用方是否还需显式执行 input/post-attention RMSNorm |

完整生命周期如下：

```text
R_l [T,4,D]
  │
  ├─ hc_pre(R_l) ──▶ y_l [T,D] ──▶ RMSNorm（若未融合）──▶ Sublayer ──▶ z_l [T,D]
  │                    │
  │                    └─ pre_l 已在此处消费
  │
  └─ 原样保留 R_l，同时暂存 post_l / comb_l ───────────────────────────────┐
                                                                            ▼
                                          hc_post(z_l, R_l, post_l, comb_l)
                                                                            │
                                                                            ▼
                                                                  R_{l+1} [T,4,D]
```

attention 与 FFN 各有一套独立的 `hc_*_fn/base/scale`，所以 `attn_post/attn_comb` 只能关闭 attention 边界，`ffn_post/ffn_comb` 只能关闭 FFN 边界；不能跨子层混用。

### 5.2 第 1 步：整体 RMS 统计 + 24 维投影

先把 4 条流拼接，对每个 token 的全部 `M·D=16384` 个元素共同计算一个 RMS 缩放因子，再投影出 24 个 logits：

```python
x_flat = residual.view(T, M * D).float()              # [T,16384]
rsqrt = torch.rsqrt(
    x_flat.square().mean(-1, keepdim=True) + rms_eps
)                                                     # [T,1]
mixes = F.linear(x_flat, fn) * rsqrt                  # [T,24]
```

这里等价于先做无可学习 weight 的整体 RMS 归一化，再线性投影：

```text
F.linear(x_flat, fn) * rsqrt == F.linear(x_flat * rsqrt, fn)
```

需要区分两种“归一化”：

- 这里的 `rsqrt` 只用于**生成 mHC 门控**，统计范围是拼接后的 `[4D]`，没有可学习 weight；
- 子层前的 `input_layernorm` / `post_attention_layernorm` 作用于已经汇聚好的 `y [T,D]`，有可学习 weight。

两者目的和作用维度不同，前者不能替代后者。

### 5.3 第 2 步：切分出 pre / post / comb

24 个 logits 按固定顺序切成 `4 + 4 + 16`：

```python
pre_raw = mixes[:, :4]
post_raw = mixes[:, 4:8]
comb_raw = mixes[:, 8:24].view(T, 4, 4)

pre = sigmoid(pre_raw * scale[0] + base[:4]) + eps          # [T,4]
post = 2 * sigmoid(post_raw * scale[1] + base[4:8])         # [T,4]
comb_logits = comb_raw * scale[2] + base[8:24].view(4, 4)   # [T,4,4]
```

三组量的数值约束不同：

| 量 | 数值范围/约束 | 语义 |
|---|---|---|
| `pre` | 每项在 `(eps, 1+eps)`；**不要求 4 项和为 1** | 当前子层从每条旧流读取多少 |
| `post` | 每项在 `(0,2)`；**不要求 4 项和为 1** | 当前子层输出向每条新流写入多少 |
| `comb` | 非负，Sinkhorn 后每行、每列和近似 1 | 旧的 4 条流如何重排为新的 4 条流 |

尤其要注意：`pre` 是独立 sigmoid 门，不是 softmax 概率。`Σ_k pre[k]` 可以大于或小于 1，所以 `y` 是门控加权和，不是凸组合。`post` 乘 2 则允许单条流以接近 2 倍的系数接收子层输出。

### 5.4 第 3 步：把 comb 归一化为近似双随机矩阵

```python
comb = comb_logits.softmax(-1) + eps                   # 行 softmax
comb = comb / (comb.sum(-2, keepdim=True) + eps)       # 第一次列归一
for _ in range(sinkhorn_iters - 1):                    # 默认再迭代 19 轮
    comb = comb / (comb.sum(-1, keepdim=True) + eps)   # 行归一
    comb = comb / (comb.sum(-2, keepdim=True) + eps)   # 列归一
```

下标约定为 `comb[t,i,j]`：`j` 是旧流编号，`i` 是新流编号。于是 `hc_post` 中第 `i` 条新流从第 `j` 条旧流读取 `comb[t,i,j]` 的权重。

默认 `sinkhorn_iters=20` 时，初始“行 softmax + 列归一”之后，再执行 19 轮“行归一 + 列归一”。由于每次除法都带 `eps`，结果是**近似**双随机，而不是数学上严格等于 1。它约束的是旧残差分支的流间传递；子层输出的 `post * z_l` 是额外注入项，因此完整 `R_l → R_{l+1}` 更新本身并不守恒。

### 5.5 第 4 步：用 pre 把 4 条流汇聚成子层输入

`hc_combine`（[mhc.py:2553](../../../python/sglang/kernels/ops/layernorm/mhc.py#L2553)）执行：

```python
y[t, :] = Σ_k pre[t, k] * residual[t, k, :]        # [T,D]
```

参考实现会先用 fp32 做乘加，再把 `y` 转回激活 dtype（V4-Flash 通常为 bf16）。这一步只消费 `pre`；`post` 和 `comb` 不参与 `y` 的计算。

举一个只说明语义的例子：若某个 token 的 `pre=[0.8,0.1,0.6,0.2]`，则

```text
y = 0.8·R_0 + 0.1·R_1 + 0.6·R_2 + 0.2·R_3
```

权重和为 `1.7` 是合法的，因为 `pre` 的目标是门控读取，不是构造概率分布。

### 5.6 第 5 步：处理子层前 RMSNorm

模型层派发接口返回 `(y, post, comb, norm_fused)`。调用方按以下规则处理 `y`：

```python
y, post, comb, norm_fused = self.hc_pre(..., norm=layernorm)
if not norm_fused:
    y = layernorm(y)
z = sublayer(y)
```

- TileLang/XPU 等能够融合输出 RMSNorm 的路径，会直接返回归一化后的 `y`，并令 `norm_fused=True`；
- FlashInfer、HIP、NPU、DeepGEMM+torch 等路径返回未做该 RMSNorm 的 `y`，令 `norm_fused=False`，由调用方补做；
- 0-token 空 batch 返回正确 shape 的空张量和 `norm_fused=False`，避免 DP overlap 的空 ubatch 破坏控制流。

因此不同后端只改变 kernel 边界，不改变模型语义：attention 总会接收 input RMSNorm 后的单流输入，FFN/MoE 总会接收 post-attention RMSNorm 后的单流输入。

### 5.7 两层 hc_pre API 的返回顺序不要混淆

代码中有两个同名入口：

| 层次 | 入口 | 输入布局 | 返回顺序 |
|---|---|---|---|
| Decoder 模型层 | [deepseek_v4.py:2761](../../../python/sglang/srt/models/deepseek_v4.py#L2761) | `[T,M,D]` | `(y, post, comb, norm_fused)` |
| 公共 kernel/通信接口 | [mhc.py:1894](../../../python/sglang/kernels/ops/layernorm/mhc.py#L1894) | 展平的 `[T,M·D]` | `(y, h_res, h_post, norm_fused)`，其中 `h_res=comb.reshape(T,M²)` |

公共接口为通信层保存展平的 scratch 张量，因此把 `comb` 放在 `post` 前面；Decoder 内部接口则使用更直观的 `post, comb` 顺序。阅读调用链时应先确认当前使用的是哪一层接口。

---

## 6. hc_post：把子层输出散射回 4 条流

参考实现：`_mhc_post_torch`（[mhc.py:1823](../../../python/sglang/kernels/ops/layernorm/mhc.py#L1823)）/ `hc_post_torch_impl`（[deepseek_v4.py:2997](../../../python/sglang/srt/models/deepseek_v4.py#L2997)）。派发入口 `hc_post`（[deepseek_v4.py:2921](../../../python/sglang/srt/models/deepseek_v4.py#L2921)）。

`hc_post` 消费的是同一次 `hc_pre` 产生的 `post`、`comb`，并把单流子层输出重新写回多流残差状态。它不是把 `x` 复制 4 份，而是同时执行“子层输出注入”和“旧残差流重排”：

```python
out[t, i, :] = post[t, i] * x[t, :]                     # 子层输出按 post 门注入第 i 条流
             + Σ_j comb[t, i, j] * residual[t, j, :]    # 双随机矩阵重排旧的 4 条流
```

**张量形状**：`x=[T,D]`（子层输出），`residual=[T,4,D]`（旧残差流），`post=[T,4]`，`comb=[T,4,4]` → `out=[T,4,D]`（新残差流）。

将公式拆成两个部分：

```text
injected[t,i,:] = post[t,i] · x[t,:]                  # [T,4,D]
transported[t,i,:] = Σ_j comb[t,i,j] · residual[t,j,:] # [T,4,D]
out = injected + transported
```

- `injected` 把 attention 或 FFN 的结果写入每一条新残差流；4 条流使用不同的 `post[t,i]`，因此不会简单复制同一个结果。
- `transported` 只在 4 条流的索引维上做矩阵混合，不改变 hidden 维 `D`。
- `comb` 的双随机约束只作用于 `transported` 这条旧残差分支。因为还有 `injected`，所以不能把整个 `out` 理解成严格的守恒更新。

这里的 `residual` 必须是进入当前子层时保存的旧状态，而不是子层输出 `x` 或已经更新过的多流状态。也正因为如此，`hc_pre` 返回的 `post/comb` 和 `hc_pre` 的输入 residual 必须一起保存到对应的 `hc_post`。

普通 Torch 参考路径在 fp32 门控系数下完成乘加，最后通过 `.type_as(x)` 把结果转换为子层输出的 dtype。空 token batch 则直接返回 `[0,M,D]` 的空张量，不进入后端 kernel。

### 6.1 attention 与 FFN 的两次配对

一个 decoder layer 内有两组独立的边界：

```text
# attention 边界
R_attn ──hc_pre(attn 参数)──▶ y_attn ──Attention──▶ x_attn
   │                                             │
   └────────保存 R_attn、attn_post、attn_comb────┘
                         │
                         ▼
              hc_post(x_attn, R_attn, attn_post, attn_comb)
                         │
                         ▼
                       R_ffn

# FFN/MoE 边界
R_ffn ──hc_pre(ffn 参数)──▶ y_ffn ──MoE/FFN──▶ x_ffn
   │                                         │
   └──────保存 R_ffn、ffn_post、ffn_comb──────┘
                         │
                         ▼
              hc_post(x_ffn, R_ffn, ffn_post, ffn_comb)
                         │
                         ▼
                       R_next
```

对应关系不能交叉：

- `attn_post/attn_comb` 由 attention 前的 `R_attn` 生成，只能配合 attention 输出使用；
- `ffn_post/ffn_comb` 由 attention `hc_post` 后的 `R_ffn` 生成，只能配合 FFN/MoE 输出使用；
- `hc_post` 的 `residual`、`post`、`comb` 必须来自同一个 mHC 边界，否则公式虽然 shape 可能仍然匹配，但语义已经错位。

---

## 7. 层间接线与融合

decoder layer 的算子编排（[deepseek_v4.py:3928-4068](../../../python/sglang/srt/models/deepseek_v4.py#L3928)）：

| 算子 | 动作 |
|---|---|
| `op_mhc_prepare_attn` | attention 侧 `hc_pre` + input_layernorm，暂存 `attn_residual/attn_post/attn_comb` |
| `op_mhc_post_attn_pre_mlp` | attention 的 `hc_post`（完成 attention 边界）→ FFN 侧 `hc_pre` + post_attention_layernorm |
| `op_mhc_postprocess` | FFN 的 `hc_post`（完成 FFN 边界），输出下一层输入 |

默认配置 `hc_pre_from_prev_sublayer=False` 时，一个 layer 的完整顺序是：

```text
输入 R0 [T,M,D]
  │
  ├─ hc_pre(attn) + input_layernorm
  ├─ Attention
  ├─ hc_post(attn)                         # 得到 R1 [T,M,D]
  ├─ hc_pre(ffn) + post_attention_layernorm
  ├─ MoE/FFN
  └─ hc_post(ffn)                          # 得到下一层输入 R2 [T,M,D]
```

代码中的 `op_mhc_prepare_attn`、`op_mhc_post_attn_pre_mlp` 和 `op_mhc_postprocess` 正是把这三段拆开，供 operations engine 在 TBO 等场景调度。

### 7.1 同一 layer 内的 post+pre kernel 融合

当 `hc_pre_from_prev_sublayer=False` 且 `use_fused_mhc_post_pre=True` 时，SGLang 可以把一个边界上的：

```text
上一子层 hc_post
    → 当前子层 hc_pre
    → 当前子层前的 RMSNorm
```

合并为 `apply_mhc_post_pre_boundary`（[deepseek_v4.py:3032](../../../python/sglang/srt/models/deepseek_v4.py#L3032)、[deepseek_v4.py:4002](../../../python/sglang/srt/models/deepseek_v4.py#L4002)）。典型地，attention 完成后，它同时完成：

```text
attention hc_post
    → FFN hc_pre
    → post_attention_layernorm
```

融合只改变 kernel 边界，不改变数学顺序。若融合 kernel 返回 `None`，调用方会回退到显式的 `hc_post` → `hc_pre` → LayerNorm；空 DP ubatch 就可能走这条回退路径。

### 7.2 `hc_pre_from_prev_sublayer=True` 是另一种接线方式

这个配置不能简单等同于上面的 kernel 融合。它表示当前子层的 `hc_pre` 使用前一个子层边界留下的状态，模型 forward 会跨 layer 传递：

```text
prev_residual
prev_post
prev_comb
```

在下一层开始时，代码先用这些状态关闭上一层最后一个子层的 mHC 边界，再生成当前 attention 边界的输入和门控。概念顺序为：

```text
上一层 FFN 输出 x_prev [T,D]
  + 上一层保存的 prev_residual / prev_post / prev_comb
  └─ hc_post ──▶ 当前层 attention 的残差流
                    └─ hc_pre(attn) + input_layernorm
```

由于这种方案把边界状态延迟到下一层消费，最后一层还需要特殊处理：模型在 `hc_head` 前先用最后保留的 `last_pre` 做一次 `hc_combine`，再执行最终 RMSNorm；它不走默认配置下的可学习 `hc_head` 加权收缩。相关收尾逻辑见 [deepseek_v4.py:4814](../../../python/sglang/srt/models/deepseek_v4.py#L4814)。

> **本文的 V4-Flash（`hc_pre_from_prev_sublayer=False`）**使用默认的“每个子层独立产生并消费 mHC 状态”的语义。
> 如果运行时启用了 `use_fused_mhc_post_pre`，只会把相邻边界的计算 kernel 合并，不会改变
> `hc_pre → 子层 → hc_post` 的逻辑结果。

---

## 8. 具体实现：多后端派发

`hc_pre`/`hc_post` 按 **平台 + 环境变量 + token 数** 派发到不同实现，数值保持一致：

| 后端 | 入口 | 说明 |
|---|---|---|
| NPU | `npu_hc_pre` / `torch.ops.custom.npu_hc_post` | A5(arch35) 需要批量维 |
| XPU / HIP(aiter) | `mhc_pre` / `mhc_post` | 融合 rmsnorm+gemm+sinkhorn+combine |
| FlashInfer | `mhc_pre_big_fuse`（[deepseek_v4.py:316](../../../python/sglang/srt/models/deepseek_v4.py#L316)） | 支持 split-K，`SGLANG_OPT_USE_FLASHINFER_MHC` |
| TileLang | `mhc.mhc_pre` / `mhc_post` | `SGLANG_OPT_USE_TILELANG_MHC_*` |
| TileLang(优化) | `mhc_post_split_h`（[deepseek_v4.py:2948](../../../python/sglang/srt/models/deepseek_v4.py#L2948)） | Blackwell/sm90 + `M==4` + **`D==5120`** + 小 batch → **V4-Flash(D=4096) 不满足，不走此路径** |
| DeepGEMM tf32 | `tf32_hc_prenorm_gemm` | 大 M（prefill）用它，小 M（decode）走 torch，按 token 数分派 |
| Triton | `_hc_split_sinkhorn_triton`（[mhc.py:315](../../../python/sglang/kernels/ops/layernorm/mhc.py#L315)） | gfx1250 上 TileLang CK 寻址编不了时的替代 |
| torch 参考 | `hc_split_sinkhorn` + `hc_combine` | `@compile_in_capture_mode`(torch.compile) 加速 |

> **V4-Flash（D=4096）的实际路径**：写死 `D==5120` / `hc_dim==20480` 的快路径（TileLang `mhc_post_split_h`、
> fused-scale 写回）**不会命中**；V4-Flash 走 FlashInfer `mhc_pre_big_fuse`（对 D 通用，`SGLANG_OPT_USE_FLASHINFER_MHC`）
> / DeepGEMM tf32（大 M）/ TileLang 通用 `mhc_pre`/`mhc_post` / torch 参考（小 M）等实现。数值仍由 split-K + sinkhorn 融合保证批不变性。

**实现要点**：

- **批不变性(batch-invariant)**：split-K 归约与 sinkhorn 融合，保证跨 batch 数值一致（服务于 KL 一致性测试）。
- **对称内存**：`y` 分配在 NCCL symmetric memory pool（`use_symmetric_memory`），使下游 all-reduce 走低延迟路径。
- **prewarm**：加载时 `prewarm_mhc_pre` / `mhc_post`（[deepseek_v4.py:5314](../../../python/sglang/srt/models/deepseek_v4.py#L5314)）预编译 kernel，避免首个 forward 卡顿。

---

## 9. 与并行通信的耦合

残差流是 `[T, M, D]`，比普通残差多一个 `M` 维，TP/DP 通信需专门处理。

- `communicator_mhc.py` 的 `MHCState` / `MHCLayerCommunicator`（[communicator_mhc.py:66](../../../python/sglang/srt/layers/communicator_mhc.py#L66)）把 mHC 的 pre/post 插进 reduce-scatter、all-gather、DP gather/scatter 流程。
- scratch 张量 `h_res` / `h_post` 随 scatter 一起按 attn_tp 切分（[communicator_mhc.py:180](../../../python/sglang/srt/layers/communicator_mhc.py#L180)）。
- 末层用 `hc_contract` 塌缩成 `[T,D]` 再 all-gather（[communicator_mhc.py:320](../../../python/sglang/srt/layers/communicator_mhc.py#L320)）。
- 不支持 `MOE_FULL` scatter 模式（要求 `moe_dp_size == attention_context_parallel_size`，[communicator_mhc.py:450](../../../python/sglang/srt/layers/communicator_mhc.py#L450)）。

---

## 10. 与通用 HyperConnection 的区别

仓库另有一个**通用** HyperConnection 库 [layers/hyperconnection.py](../../../python/sglang/srt/layers/hyperconnection.py)：

- `GatedResidual`：低秩(low-rank) mix + gated 注入、mean 混合，**无 Sinkhorn**。
- 由 GLM5-next / Qwen4-exp / Hunyuan-v4 等模型使用。

**DeepSeek V4.1 不走这条通用路径**，而是用 `deepseek_v4.py` + `kernels/ops/layernorm/mhc.py` 里 **kernel 化 + Sinkhorn 双随机** 的专用 mHC 实现。两者是不同代码。

---

## 附：关键公式速查

设第 `l` 个子层，输入残差流 `R ∈ [T,4,D]`：

```
# pre（混合）
mixes = RMSNorm(flatten(R)) @ fnᵀ                 # [T,24]
pre   = sigmoid(mixes[:,0:4]·s₀ + b₀) + ε         # [T,4]
post  = 2·sigmoid(mixes[:,4:8]·s₁ + b₁)           # [T,4]
comb  = Sinkhorn₂₀(softmax(mixes[:,8:24]·s₂ + b₂))# [T,4,4] 双随机
y     = Σ_k pre[:,k]·R[:,k,:]                      # [T,D] 送入子层

# 子层
x = Sublayer(LayerNorm(y))                         # [T,D]

# post（散射）
R'[:,i,:] = post[:,i]·x + Σ_j comb[:,i,j]·R[:,j,:] # [T,4,D] 新残差流
```
