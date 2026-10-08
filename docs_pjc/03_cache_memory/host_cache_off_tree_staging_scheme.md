# UnifiedRadixCache：Host Cache「去树化 + 请求级 staging」方案分析

> 目标问题：`write_through` 策略下 host cache（L2）也挂到 radix 树上。当 host 池
> 空间有限时，**写出通路 L1→L2→L3** 与 **读入通路 L3→L2→L1** 在同一个 host 池 /
> 同一把 host LRU 上互相驱逐，导致有效数据在被消费前就被踢掉，命中率下降。
>
> 提议：host cache 不挂树；L3→L2 读入后把 L2 slot 记到请求上，直到该请求
> L2→L1 加载完成才回收；L1→L2→L3 写出完成后回收 L2。让 L2 退化为**纯 staging
> 缓冲**，缓存职责全部下沉到 L3。
>
> 本文：现状机制 → 竞争根因 → 方案精确化（状态机/加锁/回收时机）→ 收益 →
> 风险与存在问题 → 改动点清单 → 更轻量替代方案对比 → 结论建议。

---

## 0. TL;DR（先给结论）

| 维度 | 结论 |
|---|---|
| 能否消除竞争 | **能**。请求级所有权 + 确定性回收，彻底去掉 host LRU 上「写出 vs 读入」互踢 |
| 最大代价 | **丢掉整个 L2 缓存层**：device 淘汰的前缀不再能只从 host 命中，每次复用要么走 L3（慢），要么重算 |
| 隐藏正确性坑 | 「写出后立即回收 L2」**必须挂 L3 落盘 ack**，不能挂 D→H ack，否则 L3 还没读完就回收 = 数据损坏 |
| 二次代价 | 跨请求 L2 前缀共享消失 → 高前缀重叠负载下 host 反而更吃紧、L3 读更多 |
| 结构影响 | tombstone / `backuped` / host LRU / `host_lock_ref` 锚点体系可整片删除，树只描述 device 驻留 |
| 更优解？ | 多数场景下 **host 池分区（写出预留 / 读入预留 双水位）** 能用 1/10 的改动量消除同一竞争，且保留 L2 缓存收益。建议先做分区，去树化作为 L3 足够强时的终极简化 |

---

## 1. 现状：Host Cache 是怎么挂在树上的

### 1.1 数据结构

每个 `UnifiedTreeNode` 的每个组件（FULL / SWA / Mamba）都带一份 device + host 双槽：

```
component_data[ct]:
  value          # device KV indices（None ⇒ 已从显存淘汰）
  host_value     # host KV indices（None ⇒ host 上没有）
  lock_ref       # device 路径锁
  host_lock_ref  # host 路径锁（与 device 锁完全平行的一套）
```

关键谓词（`unified_tree_core.py`）：

- `backuped`（:163）= `FULL.host_value is not None`：这段在 host 上有副本。
- **tombstone / 死节点**（:958）：`child.evicted and not child.backuped` ⇒ device 没了、
  host 也没备份 ⇒ 这段 KV 彻底消失，`match_prefix` 必须**在此截断**。
- 反过来 `evicted and backuped` = **host-only 节点**：显存没了但 host 有，能 `load_back` 回来，
  所以 `match_prefix` 继续往下走 —— 这就是 **L2 作为缓存层的全部价值**。

### 1.2 三条通路

| 通路 | 方向 | 入口 | host 池动作 |
|---|---|---|---|
| write-through 备份 | L1→L2 | `_execute_and_commit_kv_backup`(unified_radix_cache.py:1190) | `_execute_kv_backup` 先 `mem_pool_host.available_size()`，不够 `evict_host(needed)`（:1256-1260） |
| storage 落盘 | L2→L3 | `_finish_write_through_ack`(:1316)→`write_backup_storage`(:1546) | 备份 ack 回来后逐 fragment 推 L3 |
| storage 预取 | L3→L2 | `prefetch_from_storage`(:1602) | 各组件 `prepare_prefetch` 分配 host buffer；`prefetch_rate_limited()` 限流 |
| 加载回显存 | L2→L1 | `load_back`(:1341) | `inc_host_lock_ref` 钉住 host 源，`cache_controller.load` 异步 H→D，ack 前锁不解 |

### 1.3 host 池淘汰

`evict_host`（:1173）→ `tree_core.drive_host_eviction` → 走 **host LRU**
（`get_lru_no_host_lock`, tree_core:425，跳过 `host_lock_ref > 0` 的节点）→ 直接丢。
device 侧淘汰有「降级」（`_demote`, tree_core:1826，backuped 叶变 host-only），
host 侧淘汰没有下一级（除非 L3），**淘汰即丢**（:1178-1180）。

---

## 2. 竞争根因：一个池、一把 LRU、两个方向

```
        L3 (storage)
         ▲            │ prefetch (L3→L2)
  L2→L3  │            ▼
      ┌─────────────────────────────┐
      │   HOST POOL (mem_pool_host)  │  ← 单一池 + 单一 host LRU
      └─────────────────────────────┘
  L1→L2  ▲            │ load_back (L2→L1)
  backup │            ▼
        L1 (device KV)
```

竞争发生在 host 池空间不足时，**两个方向都要 `evict_host` 抢同一把 LRU**：

1. **backup（写出）踢掉刚 prefetch 进来、还没 load 的槽**
   `_execute_kv_backup`(:1256) 发现 host 不够 → `evict_host` → LRU 选中一个
   `host_lock_ref == 0` 的 host-only 节点。如果某个预取回来的段还没被 `load_back`
   钉住（预取 ack 与调度到该请求之间存在时间窗），它就可能被当成冷数据踢掉 →
   请求真正跑到时 L2 miss → 退回 L3 甚至重算。

2. **prefetch（读入）踢掉刚 backup 出去、还没落 L3 的槽**
   对称地，预取分配 host buffer 时也会驱逐 host-only 节点，可能踢掉一个
   write-through 刚写进 host、`write_backup_storage` 还没把它推到 L3 的段。

3. **抖动放大**：host 越小、并发越高，两个方向的 `evict_host` 越频繁地互相
   作废对方的成果，形成 **驱逐抖动（thrash）**，L2 命中率坍塌，且白白浪费了
   D→H / storage-IO 带宽。

> 本质：L2 同时承担了「共享缓存」和「双向 staging 缓冲」两种角色，而这两种角色
> 对内存的**生命周期诉求相反**（缓存想留久、staging 想快进快出），却共用一套 LRU。

---

## 3. 提议方案的精确化

### 3.1 一句话

> **把 L2 从「挂树的共享缓存层」降级为「请求私有的一次性 staging 缓冲」，
> 缓存/持久化职责整体上移到 L3。**

- host node **不再挂到 radix 树**：不再有 `host_value` 挂在树节点、不再有
  `backuped` / tombstone / host LRU / host 锚点锁。
- L2 slot 由**请求所有**（记在 `req` 上），生命周期与该请求的一次 H↔D 传输绑定。

### 3.2 两条通路的新生命周期（状态机）

**读入 L3→L2→L1（预取 + 加载）**

```
req 入队
  │  prefetch: 从请求私有额度分配 L2 slot（分配失败 ⇒ 跳过预取，退化重算/直读L3）
  ▼
L2 slot ∈ req.owned_host_slots     ← 记到请求上，不进树、不进 LRU
  │  调度到该请求
  ▼
load_back: L2→L1 异步 H→D
  │  等 load ack（loading_check）        ← 关键：回收点在这里，不是「推理结束」
  ▼
回收 L2 slot（还给请求私有额度 / host 池）
  │
  ▼
KV 已在 L1，插入 device 树，正常推理
```

> 注意「直到请求 L2→L1 完成才回收」应精确为 **直到 load-back 的 H→D ack 回来**
> （`loading_check`, :2479），而不是整条请求推理结束。后者会把 L2 slot 占满
> 整个 decode 过程，host 池瞬间被并发请求打爆。

**写出 L1→L2→L3（备份 + 落盘）**

```
节点产生（请求结束 / 淘汰触发）
  │  backup: L1→L2 异步 D→H，分配 L2 slot（请求或节点私有）
  ▼
等 D→H ack（writing_check）
  │
  ▼
write_backup_storage: L2→L3 异步落盘
  │  等 L3 落盘 ack                       ← 关键：回收点必须在这里
  ▼
回收 L2 slot
```

> 「写出完成就回收 L2」的**完成必须指 L3 落盘 ack**。如果只等 D→H ack 就回收，
> storage 线程还在从这块 L2 读数据往 L3 写，回收后 slot 被复用 → **读到别人的
> KV，静默数据损坏**。

### 3.3 加锁与回收，落到现有原语

| 现状原语 | 去树化后的替代 |
|---|---|
| `inc_host_lock_ref`/`dec_host_lock_ref`（锚点/在途 pin） | 退化为「请求持有 slot」——slot 在 `req.owned_host_slots` 里就是被 pin，回收即释放。可整片删掉 host_lock_ref 计数 |
| host LRU（`get_lru_no_host_lock`） | 删除。host 池变成纯 free-list 分配器 + 每请求配额 |
| `ongoing_prefetch` / `ongoing_load_back` / `ongoing_write_through` | 保留，但语义从「树节点在途」变为「请求 slot 在途」，回收点绑定 ack |
| `_finish_write_through_ack` 里「发布 backuped 标记 + demote 成 tombstone」 | 删除发布/tombstone，改为「L3 ack 后回收 slot」 |

---

## 4. 收益

1. **彻底消除 §2 的双向竞争**：slot 请求私有 + 确定性回收，host 池不再有跨方向
   LRU 互踢，抖动消失，D→H / storage 带宽不再被白白浪费。
2. **host 内存可精确 sizing**：容量 = `并发度 × 单请求 staging 峰值`，而不是模糊的
   「想缓存多少」。超配额直接跳过预取/备份，退化明确、无雪崩。
3. **大幅简化树与内存管理**：可删除 tombstone、`backuped`、host LRU、`host_lock_ref`
   锚点体系、host-only demote 路径。`match_prefix` 只需处理 device 驻留 + 死节点截断，
   逻辑显著变短（当前这套是 unified_tree_core.py 里最绕的部分）。
4. **回收路径可推理**：每个 slot 有唯一 owner 和唯一回收点，泄漏更容易定位。

---

## 5. 风险与存在问题（重点章节）

### 5.1 【最根本】丢掉整个 L2 缓存层 —— 这是范式改变，不是优化

现状 L2 是**真缓存**：device 淘汰的前缀 `_demote` 成 host-only 后仍挂树，
下一个请求 `match_prefix` 能命中它、`load_back` 拉回，**不碰 L3**。
去树化后：

- device 淘汰的段若已落 L3 ⇒ 复用要**走 L3 一次完整 IO**（比 host→device 慢 1~2 个量级）；
- 若还没落 L3 ⇒ **直接丢**，只能重算 prefill。

> 换言之：**你把 L2 命中全部转成了 L3 命中或重算**。这在「L2 命中本来就被抖动
> 打没了」的场景是净赚；但在「L2 本来有实打实命中」的场景是**命中率进一步下降**，
> 与初衷相反。是否可行**强依赖 L3 的延迟/带宽**能不能顶上 L2 让出的那部分命中。

**验收前必须量化**：现状 L2 命中贡献了多少（`load_back` 命中 token 数 vs 总命中），
以及 L3 读延迟能否吸收。没有这两个数字，去树化是拍脑袋。

### 5.2 【最易错的正确性坑】写出回收时机必须挂 L3 ack

见 §3.2。`_finish_write_through_ack`(:1316) 当前顺序是「发布标记→解锁→推 L3」，
推 L3 是**异步**的。若照提议「写出完成回收 L2」，务必：

```
D→H ack → 触发 L2→L3 → L3 落盘 ack → 才回收 L2 slot
```

漏掉最后一个 ack 就是 use-after-free 语义的静默数据损坏。这条要写进设计不变量并加断言。

### 5.3 跨请求 L2 前缀共享消失 → 高重叠负载下 host 反而更紧

现状两个共享前缀的请求共用同一份 `host_value`。请求私有后，**每个请求各 staging 一份**：

- 相同前缀的 N 个并发请求 → host 上 N 份重复 → host 压力 ×N；
- 且每份都要各自向 L3 读一次（L2 无 dedup）→ L3 读放大。

这恰好打在 **前缀缓存最该发光的高重叠负载**（多轮对话、共享 system prompt）上。
缓解只能靠 L3 端 dedup，但 L3 读放大依旧。

### 5.4 device 淘汰语义退化：demote → delete，并反向耦合 L3

现状 device 淘汰 backuped 叶走 `_demote`（tree_core:1826）保留 host 副本、零重新备份。
去树化后没有 host-only 节点 ⇒ **淘汰 = 删除**。为不丢数据，必须保证淘汰前该段
**已落 L3**，否则要么阻塞淘汰等 L3、要么接受丢失重算。这把「显存淘汰」和「L3 落盘
进度」耦合了起来，反而引入新的等待点（原本 write-through demote 是解耦的）。

### 5.5 host 池 sizing / OOM / 长前缀请求

请求私有额度必须能容纳**单请求最长 staging**。一个拉了超长前缀（几万 token）的请求
会长时间占用大量 L2；并发一多，host 池被少数长请求占满 ⇒ 预取/备份大面积跳过。
需要 **per-request 配额 + 全局水位 + 长度上限**的准入控制，比现在的
`prefetch_rate_limited()`（单一全局限流）复杂。

### 5.6 多组件 / sidecar 的私有化成本

unified 有 FULL / SWA / Mamba 多组件，各有独立 `prepare_prefetch` /
`prepare_load_back` / `finalize_load_back` / host buffer，还有 sidecar 池
（`_build_sidecar_transfers`, :1478，靠 `indices_from_pool` 借索引）。请求私有 staging
要为**每个组件 + 每个 sidecar** 都建立 per-request slot 记账与回收。机械但量大，
且 SWA 的 tombstone 语义（本就复杂）要一起改。

### 5.7 跨 rank 一致性不能丢

现状预取用 `all_reduce(MAX)`（`can_terminate_prefetch`, :1712）保证各 rank 取到相同
token 数、树不分叉。私有化后 slot 虽私有，但**取到多少 token、是否成功**仍必须跨
rank 归约一致，否则各 rank 插入 device 树的长度不同 → 共享 KV 槽号错位。这条约束
不能因为「slot 私有」就省掉。

### 5.8 与 write policy 语义的重叠

`write_through_threshold`（:476，write_through=1 / 否则 2）表达「命中几次才值得备份到
host」——这是**把 host 当缓存**才有的调参。L2 变纯 staging 后该阈值失去意义，
write-back 淘汰懒备份路径（`evict_device_leaf` 的 write-back 分支, tree_core:1815）也
需要重新定义。等于 write policy 这套配置要重新梳理，别留下语义悬空的旋钮。

---

## 6. 改动点清单（文件 × 函数级）

| 文件 | 位置 | 改动 |
|---|---|---|
| `unified_radix_cache.py` | `prefetch_from_storage`:1602 | host buffer 从「树锚点分配」改为「请求私有额度分配」；`ongoing_prefetch` 绑定 `req` |
| | `check_prefetch_progress`:1764 | 命中段不再 `insert_host` 挂树，改为记入 `req.owned_host_slots`；三段 host 索引处理（重复/新增/超额）保留但目标变私有池 |
| | `load_back`:1341 / `_load_back_transfers`:1394 | 源从「树 host_value」改为「req 私有 slot」；`inc/dec_host_lock_ref` 换成「持有/回收 slot」 |
| | `loading_check`:2479 | ack 回来后**回收 L2 slot**（新增回收动作） |
| | `_execute_and_commit_kv_backup`:1190 / `_execute_kv_backup`:1242 | 目标 host slot 改私有；删掉 `evict_host` 互踢路径 |
| | `_finish_write_through_ack`:1316 | 删除 backuped 发布 + tombstone；改为「L3 落盘 ack 后回收 slot」 |
| | `write_backup_storage`:1546 | 回调里挂 slot 回收 |
| | `evict_host`:1173 | 删除（host LRU 淘汰不再需要）或降级为纯配额回收 |
| | `init_hicache`:410 | `write_through_threshold` / `is_write_back` 语义重定义或废弃 |
| `unified_tree_core.py` | `match_prefix`:826 / 死节点判定:958 | 去掉 backuped 续走，device 淘汰即截断 |
| | `evict_device_leaf`:1787 / `_demote`:1826 | demote 路径删除，改 delete（且需 L3-已落盘 前置） |
| | host LRU（:406/:425）、`inc/dec_host_lock_ref`（:790/:807）、`backuped`（:163）、`commit_backup`/`commit_load_back`/`finish_write_through` | 大面积删除或改语义 |
| `components/*.py`（full/swa/mamba） | `prepare_prefetch` / `prepare_load_back` / `finalize_load_back` | host buffer 记账改请求私有；SWA tombstone 语义联动 |
| `managers/cache_controller.py` | write / load / prefetch 回调 | slot 回收钩子；per-request 配额准入 |
| `req`（schedule_batch.py） | 新增字段 | `owned_host_slots`（按组件）+ 回收状态 |

> 规模判断：这是**跨 tree_core / radix_cache / components / cache_controller / Req**
> 的结构性改动，不是局部补丁。删除量可能比新增量还大（tombstone/host-LRU 整片删）。

---

## 7. 更轻量的替代方案（强烈建议先评估）

竞争的**直接根因**是「一个 host 池、一把 LRU、两个方向」。不必去树化也能解：

### 方案 A：Host 池分区 / 双水位（改动最小，保留 L2 缓存）

把 host 池按用途切两块预留（或设两条水位线）：

```
┌──────── HOST POOL ────────┐
│  写出预留(backup+待落L3)   │  ← backup 只能在这块里 evict
│───────────────────────────│
│  读入预留(prefetch+待load) │  ← prefetch 只能在这块里 evict
└───────────────────────────┘
      共享缓存区（可被两边回收，但 min-residency 保护未消费段）
```

- backup 与 prefetch **不再互相驱逐对方在途的槽**，抖动即消失。
- 保留 tombstone / L2 命中 / 跨请求共享，**不丢缓存收益**。
- 改动量：主要在 `evict_host` / `mem_pool_host` 的分区计数 + 两条准入水位，
  以及给「已到达未消费」段一个最短驻留保护（防被 LRU 提前踢）。

### 方案 B：对称限流 + 未消费段保护

现有只有 `prefetch_rate_limited()`（读入侧限流）。补一个**备份侧限流**（host 水位高
时缓发 backup），并给「prefetch 已 ack 但未 load」「backup 已 ack 但未落 L3」的段
在 LRU 里加最短驻留 / 最高优先级，杜绝「消费前被踢」。改动更小。

### 对比

| 方案 | 消除竞争 | 保留 L2 缓存 | 改动规模 | 主要风险 |
|---|---|---|---|---|
| 去树化 + 请求私有 staging（提议） | ✅ 彻底 | ❌ 全丢，压到 L3 | 大（结构性） | L3 顶不上则命中率反降；写出回收 ack 坑；高重叠 host 放大 |
| A. Host 池分区 / 双水位 | ✅ | ✅ | 小 | 分区比例需按负载调；极端偏斜时一块闲一块紧 |
| B. 对称限流 + 未消费保护 | 🟡 大幅缓解 | ✅ | 最小 | 限流阈值需调；重压下仍可能偶发 |

---

## 8. 结论与建议

1. **提议方案能消除竞争，但代价是范式改变**：L2 从缓存层降为纯 staging，缓存职责
   全压到 L3。**只有当 L3 延迟/带宽足以吸收原 L2 命中**时才净收益，否则命中率会
   朝相反方向走。**上线前必须先量化「现状 L2 命中贡献」和「L3 读延迟」两个数**。

2. **若确实要做去树化**，把两条不变量钉死：
   - 写出回收 L2 **挂 L3 落盘 ack**（§5.2），加断言；
   - device 淘汰改 delete 前**保证已落 L3**（§5.4）。
   并配套 per-request 配额准入（§5.5）与保留跨 rank 归约（§5.7）。

3. **工程上优先试方案 A（host 池分区）**：它用远小的改动直接命中根因、且不牺牲
   L2 缓存收益。把去树化留作「L3 已足够强、且 A 仍不够」时的终极简化选项。

4. 无论走哪条路，先补 **可观测性**：`load_back` 命中 token、`evict_host` 触发次数与
   来源（backup vs prefetch）、prefetch-ack-未消费即被驱逐计数。没有这些指标，
   「命中率下降」既说不清根因、也无法验收修复效果。

---

## 附：关键代码索引

- 现状竞争点：`_execute_kv_backup` host 不足 → `evict_host`（unified_radix_cache.py:1256-1260）；
  `prefetch_from_storage` 分配 host buffer（:1602）。
- host 淘汰：`evict_host`（:1173）→ `drive_host_eviction` → host LRU（tree_core:406/425）。
- tombstone / 死节点截断：`match_prefix`（tree_core:826，死节点判定 :958）。
- demote（保留 L2 缓存的关键）：`evict_device_leaf`/`_demote`（tree_core:1787/1826）。
- 加载回收锁：`load_back`（:1341）、`loading_check`（:2479）、`writing_check`（:2398）。
- 写出收尾：`_finish_write_through_ack`（:1316）、`write_backup_storage`（:1546）。
- 写策略：`init_hicache`（:410），`write_through_threshold`（:476）。
- 跨 rank 一致：`can_terminate_prefetch` 的 `all_reduce(MAX)`（:1712）。
