# Mix 实例模拟 P 实例：ITPS 差距与逼近 P 性能的调优

> 场景：部署多个 **mix 实例**（既能 prefill 也能 decode），底层通过 cache 池化复用 KV。
> router 把请求的 `max_new_tokens` 设为 1，先调度给 mix 0 → mix 0 跑完 prefill + 出 1 个 token
> 并把 KV 写入共享池 → router 再把请求调度给 mix 1 读池续跑 decode。
> 这样 **mix 0 专职 prefill、mix 1 专职 decode**，用 mix 拼凑出 PD 分离的效果。
>
> 本文回答三件事：
> 1. mix 0 模拟 P 的**理论最高 ITPS** 会不会低于真正的 P 实例？为什么？
> 2. 有哪些加速技术是**只有真 P 实例**能用的？
> 3. mix 0 通过设置哪些**环境变量 / 启动参数**可以尽量逼近 P 的性能？

---

## 一、先给结论

要区分两个"理论最高"：

| 口径 | 真 P 实例 | mix 模拟 P | 差距来源 |
|---|---|---|---|
| **纯硬件算力 roofline**（FLOPs 上限） | 相同 | 相同 | 无——同一张卡的峰值算力不因角色而变 |
| **给定配置下的可达峰值 ITPS** | 高 | **明显更低** | 配置层（显存/graph）+ 数据通路层（KV 写出不重叠） |

也就是说：**算不过物理定律的部分两者一样，但 mix 拿不到那个上限。** 峰值 ITPS 并非只由 FLOPs 决定：

```
可达 ITPS ≈ (峰值FLOPs × MFU) × (prefill计算占GPU时间的比例)
                          ↑                       ↑
              取决于单次forward的token数         取决于有多少GPU时间被
              (batch越大越接近roofline)          KV写出/混合调度/采样"偷走"
```

mix 在**这两个乘子上都吃亏**，所以可达峰值更低。下面拆开讲。

---

## 二、为什么 mix 的可达 ITPS 更低

### 2.1 显存预算 → 压不满 GEMM 的 M 维（拉低 MFU）

prefill 是 compute-bound 的大 GEMM，其效率随 batch 内 token 总数（GEMM 的 M 维）单调上升，越大越接近 roofline。谁能把更多 token 塞进一次 forward，谁的 MFU 就更高。

| 显存维度 | 真 P（`--disaggregation-mode prefill`） | mix 实例 |
|---|---|---|
| decode running batch 预留 | 无需预留 | 仍按混合模式保留 `new_token_ratio` 安全水位 |
| decode CUDA graph 显存池 | 完全不捕获，省下 | 默认仍捕获 decode graph，占显存 |
| `chunked_prefill_size` | 可拉满（KV 池几乎全给 prefill） | 受限，要给 decode 留空间 |
| 单次 forward 的 token 数 | 大 → MFU 高 | 小 → MFU 低 |

mix 即使把 `max_new_tokens=1`，调度器仍运行在**混合模式**，无法把全部 HBM 转成 prefill token budget，于是每一步 forward 的 MFU 天然低一档。

### 2.2 最关键差距：KV 写出无法与计算重叠

这是 mix 模拟 P 与真 P 之间**最本质**的区别，也是 env var 最难弥补的一环。

**真 PD 的 P 实例（逐层流式传输）：**
```
Layer 0 compute ──► Layer 1 compute ──► Layer 2 compute ──► ...
       └─ KV₀ RDMA发出 ─┐    └─ KV₁ RDMA发出 ─┐   └─ KV₂ ...
                        (与下一层计算重叠, 走NIC/copy-engine, 不吃SM)
          KV 传输在时间线上几乎"零暴露"
```
SGLang 的 disaggregation 就是按 layer/chunk 粒度 `send_kv_chunk`，GPU→GPU RDMA（mooncake / nixl）单边写，**旁路 host、旁路计算流**。

**mix + 通用 cache 池：**
```
整个 forward (全部层跑完) ─────────────────► 然后才写池
                                            └─ GPU→CPU 拷贝 (HiCache L2 host)
                                               或 L3 落盘, 串行, 吃 copy-engine/PCIe
             每个请求多出一段"串行写出"暴露在关键路径上
```
即便池支持 host 内存零拷贝布局，写出**时机在 forward 结束后**，无法与 prefill 计算重叠，直接压缩了公式里的第二个乘子（prefill 计算占 GPU 时间的比例）。

### 2.3 输出 / 调度通路开销

mix 生成那 1 个 token 仍要走完整的 **sampling → detokenize → 输出流** + router 回程调度；混合模式 event loop 每步还要做 prefill/decode 分支判断、DP sync 等。真 P 侧只需"发首 token + KV 句柄"，流程更精简且天然流水线化。

---

## 三、只有真 P 实例能用的加速技术

| 技术 | 说明 | mix 为何用不了 / 打折 |
|---|---|---|
| **逐层 KV 流式传输** | layer i 算完立即发，与 layer i+1 计算重叠 | 通用池按请求整体落池，粒度和时机对不上 |
| **GPU→GPU RDMA 零拷贝** | 跳过 host staging，不吃 SM/PCIe | 池化通常经 host（HiCache L2）或存储（L3） |
| **prefill 独立并行度** | P 可用与 D 不同的 TP/EP/DP（compute-bound 偏大 TP） | mix 集群同构，无法为 prefill 单独调并行 |
| **零 decode graph 预算** | 不捕获 decode CUDA graph，显存全给 prefill | 混合模式仍需 decode graph（只能调小，难彻底关） |
| **激进 chunked prefill** | 无 decode SLO（TPOT/ITL）约束，chunk 拉满冲 MFU | mix 要保护本机 decode 延迟，chunk 受限 |
| **prefill 专用 attention backend** | `--prefill-attention-backend` 独立选型，不迁就 decode | mix 的后端要同时兼顾 decode 路径 |
| **关闭投机 / MTP** | decode 才用 spec decoding，P 侧无草稿层开销 | mix 若开 spec，草稿层结构常驻 |

核心一句：**"逐层 RDMA 重叠写出" + "不为 decode 预留显存/graph" 这两项是 `--disaggregation-mode prefill` 才解锁的，通用 cache 池 + 混合模式拿不到。**

---

## 四、mix 逼近 P 性能的调优清单

思路对应第二节的两个乘子：**(A) 抬高 MFU**（把更多 token 塞进 prefill batch）+ **(B) 减少非 prefill 计算偷走的 GPU 时间**（压缩 KV 写出/调度/graph 开销）。

### 4.1 启动参数（主力，SGLang 主要靠 CLI 而非裸 env）

> 参数定义见 `python/sglang/srt/server_args.py`，括号内为行号。

| 参数 | 建议值 | 作用 | 对应乘子 |
|---|---|---|---|
| `--chunked-prefill-size`（:797） | 调大（如 16384/-1） | 单次 forward 塞更多 prefill token，冲 GEMM roofline | A ↑MFU |
| `--max-prefill-tokens`（:807） | 调大 | 放开单批 prefill token 上限，别成为瓶颈 | A ↑MFU |
| `--cuda-graph-max-bs-decode`（:1847） | 调**小**（如 1~4） | 缩小 decode CUDA graph，省出显存给 prefill batch | A ↑MFU |
| `--cuda-graph-max-bs-prefill`（:1852） | 视需要 | prefill 极少用 graph，一般无需大 | — |
| `--mem-fraction-static`（:770） | 调大（如 0.92+） | 更多 HBM 给 KV/激活，扩大 prefill token budget | A ↑MFU |
| `--max-running-requests`（:775） | 调**小** | mix0 只做 prefill+1token，压低 decode 并发占用 | A ↑MFU |
| `--schedule-policy fcfs`（:824） | `fcfs` | 纯 prefill 无需 LPM 的 cache-aware 复杂度，减调度开销 | B ↓开销 |
| `--schedule-conservativeness`（:882） | 调**低**（<1.0） | mix0 无 decode SLO 要保护，可激进组更大 prefill 批 | A ↑MFU |
| `--prefill-attention-backend`（:1685） | 选 prefill 最优后端 | 用 varlen/ragged prefill kernel，不迁就 decode | A ↑MFU |
| `--disable-radix-cache`（:928） | 视情况 | 若不靠前缀复用，关掉省树维护开销（但通常想留复用收益） | B ↓开销 |

### 4.2 与 cache 池 / KV 写出相关（减小 2.2 的串行代价）

| 参数 | 建议值 | 作用 |
|---|---|---|
| `--hicache-io-backend kernel`（:2648，默认已是 `kernel`） | `kernel` | 融合 CUDA gather/scatter，写出比 `direct` 快 ~3×，尽量与计算重叠 |
| `--hicache-mem-layout page_first`（:2656） | `page_first` / `page_first_direct` | 页主序布局，支持 L3 zero-copy；`page_first_direct` 配 `direct` 后端整块连续拷 |
| `--hicache-write-policy`（:2640） | 权衡见下 | `write_through`=立即写池（mix1 尽快可读，但占带宽）；`write_back`=惰性写（省带宽但可用性延迟，不利于即时 handoff） |

写策略的 tradeoff：mix1 必须能**尽快**从池读到 KV 才能续跑，所以倾向 `write_through` / `write_through_selective`（可用性优先）；`write_back` 惰性回写更省带宽但会拖慢 handoff。若池是纯共享 L1（同机/RDMA 直挂），可绕过 host tier，接近真 PD。

### 4.3 相关环境变量（辅助，`python/sglang/srt/environ.py`）

| 环境变量 | 说明 |
|---|---|
| `SGLANG_PREFETCH_BLOCK_SIZE_MB`（:269，默认 16） | mix1 侧从池 prefetch KV 的块粒度；调大可提升读回带宽利用 |
| `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP`（:441，默认 False） | 保持默认 False，允许连续 prefill 重叠 |
| `SGLANG_HICACHE_*`（:522~555） | 存储后端（file/mooncake/hf3fs/nixl）细节配置；若用 mooncake/nixl 走 RDMA 可显著降低写出暴露 |

> 说明：SGLang 绝大多数性能开关是**启动参数**而非裸 env var；真正影响 ITPS 的主力在 4.1，env var 多为存储后端/传输细节的旁路调节。

### 4.4 推荐起步配置（mix0 专职 prefill）

```bash
python -m sglang.launch_server \
    --model-path <model> --tp <N> \
    --chunked-prefill-size 16384 \
    --mem-fraction-static 0.92 \
    --cuda-graph-max-bs-decode 2 \
    --max-running-requests 8 \
    --schedule-policy fcfs \
    --schedule-conservativeness 0.8 \
    --prefill-attention-backend <fa3/flashinfer/...> \
    --enable-hierarchical-cache \
    --hicache-io-backend kernel \
    --hicache-mem-layout page_first \
    --hicache-write-policy write_through
```
（具体数值需按模型/显存实测，逐项 A/B。）

---

## 五、调优也无法弥补的部分（诚实边界）

即便把上面全调到位，仍有两块**结构性缺口**只有 `--disaggregation-mode prefill` 才能真正闭合：

1. **KV 写出无法逐层重叠**——通用池在 forward 结束后整体落池，是串行暴露；真 P 是 layer-wise RDMA 与计算重叠。这是最大的、env var 补不回来的差距。
2. **无法彻底摘掉 decode 预算**——mix 仍在混合模式，`new_token_ratio`、decode graph、decode 后端结构无法完全清零，prefill token budget 到不了真 P 的上限。

所以：**如果目标真是 P 级性能，正解是直接用 `--disaggregation-mode prefill` 起真 P 实例**（配 mooncake/nixl 做 GPU→GPU RDMA 传输），而不是用 mix 拼凑。mix 模拟只能把差距**收窄**，收窄的上限受限于"混合模式 + 通用池写出不重叠"这两堵墙。

---

## 六、一句话工程结论

- mix 模拟 P 的可达 ITPS **会低于真 P**，根因是**显存预算压低 MFU** + **KV 写出不能与计算重叠**，而非算力本质差异。
- 只有真 P 能用：**逐层 KV 传输、GPU→GPU RDMA 零拷贝、prefill 独立并行度、零 decode graph 预算、激进 chunked prefill、prefill 专用 backend、免投机开销**。
- 逼近手段：抬高 prefill batch（`chunked-prefill-size` / `mem-fraction-static` ↑、`cuda-graph-max-bs-decode` / `max-running-requests` ↓）+ 降调度开销（`fcfs`、低 `schedule-conservativeness`）+ 优化写出通路（`kernel` io backend、`page_first`、RDMA 存储后端）。
- 但"写出重叠"和"清零 decode 预算"两堵墙只有真 PD 能推倒——**要 P 性能，就起真 P**。

