# UnifiedRadixCache 统一前缀缓存：架构与核心思想

> 代码版本：`main`（2026-08，含 TreeCore/Controller 分层重构后）。
>
> 本文梳理 SGLang 的 **UnifiedRadixCache**——用"**一棵基数树 + 可插拔组件 + 机制/控制器分层**"把
> Full-Attention / SWA / Mamba(SSM) 三种 KV 缓存形态统一到同一棵树里，并原生承载 HiCache
> （GPU ↔ Host ↔ Storage 三级存储）的前缀缓存框架。
>
> 关联文档：
> - 旧的分裂式实现与 HiCache 使用姿势见 [hicache_usage_and_design.md](hicache_usage_and_design.md)；
> - SWA 单独一套树的历史实现见 [swa_radix_cache_architecture.md](swa_radix_cache_architecture.md)（双池/双锁/tombstone 语义的参考）；
> - DSV4 多层压缩池通过本框架的 **sidecar pool** 接入，见 [deepseek_v4_cache_management.md](../09_models/deepseek_v4_cache_management.md)；
> - 调度侧调用时机见 [overlap_schedule_architecture.md](../02_scheduling/overlap_schedule_architecture.md) 与 [centralized_scheduling_loop_architecture.md](../02_scheduling/centralized_scheduling_loop_architecture.md)。

---

## 0. 阅读地图（由浅入深）

| 层次 | 章节 | 你将理解 |
|---|---|---|
| 全貌 | §1 一张图看全貌 | 五层分层架构图 |
| 思想 | §2 核心思想六条 | 这套设计到底在解决什么、用什么代价换什么 |
| 动机 | §3 为什么要"统一" | 旧的四套并列基数树的痛点 |
| **骨架** | **§4 机制/控制器分层** | **TreeCore vs Cache、NodeId 边界、CacheAction 延迟动作、drain 契约、后端可插拔** |
| 结构 | §5 数据结构 | `UnifiedTreeNode` / `ComponentData` / `UnifiedLRUList` 内存布局 |
| 结构 | §6 三种组件与钩子全景 | Full / SWA / Mamba 的 value 语义、锁策略、淘汰策略 + `TreeComponent` 钩子表 |
| 概念 | §7 三个心智模型 | Tombstone（墓碑）、Leaf-set vs LRU、Cascade（级联淘汰）+ 优先级 |
| 流程 | §8 读路径 match_prefix | 多组件联合判定 + device/host 双锚点 |
| 流程 | §9 写路径 insert | 可恢复状态机 WALK → COMMIT → TAIL |
| 流程 | §10 锁与 size 记账 | path-lock / window-lock / single-node-lock ↔ evictable/protected |
| 流程 | §11 淘汰全链路 | 控制器驱动循环 + 树侧 cascade + 墓碑链清理 |
| 进阶 | §12 HiCache 三级存储 | 四个传输相位、异步流/线程、TP/PP 一致性、sidecar pool |
| 进阶 | §13 请求生命周期 | match → alloc → forward → cache_(un)finished_req 时序 |
| 集成 | §14 装配与注册 | registry 选择链 → 组件 → create_tree_core → init_hicache |
| 收口 | §15 不变量与 sanity_check | 五大类不变量 |
| 收口 | §16 关键约束与易错点 | 含两个当前代码里发现的问题 |

核心源码索引：

| 模块 | 路径 | 行数 |
|---|---|---|
| 控制器（主类） | [unified_radix_cache.py](../../../python/sglang/srt/mem_cache/unified_radix_cache.py) | 2064 |
| 机制（树本体） | [unified_cache/unified_tree_core.py](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py) | 2160 |
| 边界契约（ABC） | [unified_cache/unified_tree_core_interface.py](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core_interface.py) | 485 |
| 延迟动作定义 | [unified_cache/cache_action.py](../../../python/sglang/srt/mem_cache/unified_cache/cache_action.py) | 103 |
| 组件类型枚举 | [unified_cache/component_type.py](../../../python/sglang/srt/mem_cache/unified_cache/component_type.py) | 29 |
| TreeCore 后端注册表 | [unified_cache/tree_core_registry.py](../../../python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py) | 73 |
| 组件 ABC + 公共类型 | [unified_cache/components/tree_component.py](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py) | 516 |
| Full 组件 | [unified_cache/components/full_component.py](../../../python/sglang/srt/mem_cache/unified_cache/components/full_component.py) | 382 |
| SWA 组件 | [unified_cache/components/swa_component.py](../../../python/sglang/srt/mem_cache/unified_cache/components/swa_component.py) | 1013 |
| Mamba 组件 | [unified_cache/components/mamba_component.py](../../../python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py) | 854 |
| 作者自述设计文档 | [unified_cache/components/README.md](../../../python/sglang/srt/mem_cache/unified_cache/components/README.md) | ~360 |
| 参数/结果结构 | [base_prefix_cache.py](../../../python/sglang/srt/mem_cache/base_prefix_cache.py) | — |
| 初始化参数 | [cache_init_params.py](../../../python/sglang/srt/mem_cache/cache_init_params.py) | — |
| HiCache 池装配 | [hybrid_cache/hybrid_pool_assembler.py](../../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py) | — |
| HiCache 异步控制器 | [managers/cache_controller.py](../../../python/sglang/srt/managers/cache_controller.py) | — |
| 选择链 | [registry.py](../../../python/sglang/srt/mem_cache/registry.py) | — |
| 流式会话 | [session/streaming_session.py](../../../python/sglang/srt/session/streaming_session.py) | — |

---

## 1. 一张图看全貌

### 1.1 五层分层架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ① 调用方：Scheduler / SchedulePolicy / PrefillAdder                       │
│    match_prefix() · insert() · evict() · cache_unfinished_req()          │
│    cache_finished_req() · inc/dec_lock_ref() · check_hicache_events()    │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ BasePrefixCache 接口
                                 │ （入参/出参都是 msgspec 结构：MatchPrefixParams
                                 │   / InsertParams / EvictParams / MatchResult ...）
┌────────────────────────────────▼─────────────────────────────────────────┐
│ ② UnifiedRadixCache ——「控制器 / 执行者」  unified_radix_cache.py         │
│                                                                          │
│   · 唯一持有内存池句柄：req_to_token_pool、token_to_kv_pool_allocator      │
│   · 执行 TreeCore 交回的 CacheAction / ComponentAction                    │
│   · 排空 device_frees / host_frees → allocator.free()                     │
│   · HiCache 编排：cache_controller、write-through/back、prefetch、storage  │
│   · 分布式一致性：TP/PP/CP 的 all_reduce(MIN)、_pp_sync、work_list 背压     │
│   · StreamingSession 短路、metrics collector、sidecar pool 注册            │
└────────────────────────────────┬─────────────────────────────────────────┘
        ▲                        │  ★ 跨界只传 NodeId(int)，不传节点对象
        │ CacheAction /          │
        │ ComponentAction        ▼
        │ device_frees /   ┌──────────────────────────────────────────────┐
        │ host_frees       │ ③ UnifiedTreeCoreInterface（ABC 边界契约）     │
        └──────────────────┤    可被 Rust / C++ 实现满足，无需继承 Python   │
                           └──────────────────┬───────────────────────────┘
┌─────────────────────────────────────────────▼────────────────────────────┐
│ ④ UnifiedTreeCore ——「机制 / 决策者」 unified_cache/unified_tree_core.py  │
│                                                                          │
│   · 唯一持有树结构：root_node / children / _node_arena（NodeId → node）    │
│   · 每节点 component_data[FULL|SWA|MAMBA]（value / lock_ref / host_value）│
│   · 两套淘汰追踪：evictable_device|host_leaves（Full）+ lru_lists（aux）   │
│   · match / insert / evict 的算法本体、size 记账、sanity_check             │
│   · ★ 绝不触碰任何内存池 —— 只「决定」，把动作交回控制器执行                │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ 每个节点上串行调用「全部组件」的同名钩子
┌────────────────────────────────▼─────────────────────────────────────────┐
│ ⑤ TreeComponent（ABC）  unified_cache/components/                         │
│    ┌──────────────┐   ┌──────────────┐   ┌────────────────┐              │
│    │FullComponent │   │ SWAComponent │   │ MambaComponent │              │
│    │ 骨干·全路径  │   │ 窗口·路径数据│   │ 叶·单点状态    │              │
│    └──────┬───────┘   └──────┬───────┘   └───────┬────────┘              │
└───────────┼──────────────────┼───────────────────┼──────────────────────-┘
            ▼                  ▼                   ▼
   token_to_kv_pool_    SWATokenToKVPool     HybridReqToTokenPool
   allocator            Allocator            (mamba state slots)
            └──────────────────┴───────────────────┘
                               ▼
        host pool（HostPoolGroup） + storage backend（L3）
        由 hybrid_cache/hybrid_pool_assembler.py 在 init_hicache 时装配
```

### 1.2 一棵树上的三份数据（最关键的一张图）

统一的本质：**树结构只有一份，每个节点上并排挂三份组件数据**。

```
                        root  (三组件 lock_ref 都 = 1，永不淘汰)
                          │
      key = 页对齐的 token id 序列（RadixKey），一条边 = 一段 token
                          │
                   ┌──────┴───────┐
                node A          node B
                  │
          ┌───────┴────────┐
        node C           node D
          │
        node E (leaf)

每个 UnifiedTreeNode 内部：
┌───────────────────────────────────────────────────────────────┐
│ key: RadixKey        parent / children: dict                  │
│ last_access_time: float64（逻辑时钟）   hit_count   priority    │
│ lru_prev[2N] / lru_next[2N]  ← device 槽 ct，host 槽 ct+N      │
│                                                               │
│ component_data[0] = FULL  : value=全量 KV 下标  lock_ref  host_value │
│ component_data[1] = SWA   : value=滑窗 KV 下标  lock_ref  host_value │
│ component_data[2] = MAMBA : value=SSM state 槽  lock_ref  host_value │
└───────────────────────────────────────────────────────────────┘
       value = None 且节点还在树上  ⇒  该组件的「墓碑」(tombstone)
```

- **Full 是基座组件**（`BASE_COMPONENT_TYPE = ComponentType.FULL`，
  [component_type.py:29](../../../python/sglang/srt/mem_cache/unified_cache/component_type.py#L29)）：
  节点的 `backuped` / `evicted` 只看 Full；Full 没了这个节点就没有存在意义。
- **SWA / Mamba 是辅助（auxiliary）组件**：可以在 Full 还活着的时候单独变成墓碑。
- 纯全注意力模型 = 只装 Full 一个组件，其它组件的循环体自动为空 ⇒ **零特判、零额外开销**。

### 1.3 一次调用的动作流（控制器 ↔ 机制的往返）

```
Scheduler                UnifiedRadixCache                UnifiedTreeCore
   │                            │                                │
   │ insert(params) ───────────►│                                │
   │                            │ begin_insert(params) ─────────►│  WALK 一步
   │                            │                                │  产生 actions
   │                            │◄── InsertStepResult(actions) ───┤  （遇到不可延迟
   │                            │                                │    动作则挂起）
   │                            │ 执行 actions：                  │
   │                            │  FreeDeviceKV → allocator.free  │
   │                            │  BackupKV     → D2H 传输         │
   │                            │  ComponentAction → 组件自处理    │
   │                            │ resume_insert() ──────────────►│  继续 WALK/COMMIT/TAIL
   │                            │◄── InsertStepResult(result) ────┤  完成
   │◄──── InsertResult ─────────┤                                │
   │                            │ finally: end_insert() 排空残留   │
```

一句话：**树决定做什么，Cache 负责做**（"tree decides, cache executes"）。

---

## 2. 核心思想（六条）

| # | 核心思想 | 具体落地 | 换来了什么 |
|---|---|---|---|
| 1 | **一棵树，多组件** | 树结构/遍历/分裂/LRU 只写一遍；模型形态差异全部收进 `TreeComponent` 插件 | 消灭 4 套并列基数树（Radix / SWA / Mamba / Hi\*）与其组合爆炸（HiMambaRadixCache 这类"交叉产物"不再需要） |
| 2 | **机制与控制器分离**（tree decides, cache executes） | `UnifiedTreeCore` 只管结构与决策，**不持有任何内存池句柄**；所有释放/拷贝由 `UnifiedRadixCache` 执行 | 树逻辑变成可单测的纯数据结构；换语言实现（Rust TreeCore）只需满足 `UnifiedTreeCoreInterface` |
| 3 | **NodeId 是唯一跨界句柄** | 接口上一切节点都是 `NodeId = int`；`_node_arena: dict[NodeId, node]` 做解析；`_match_post_processor` 出口统一 `._replace(...=node.id)` | 边界零对象泄漏 ⇒ 外部代码不可能绕过接口直接改树；也让非 Python 实现可行 |
| 4 | **延迟动作（deferred action）模型** | 树侧不做副作用，只吐 `CacheAction` / `ComponentAction` 列表 + `device_frees` / `host_frees` 字典；可延迟的动作攒批，不可延迟的在 barrier 处立即执行 | 副作用点收敛到少数几处，异常路径可 fail-stop 而不留"半提交"状态；批量 free 减少 allocator 调用 |
| 5 | **逐组件资源隔离 + 级联一致性** | 每组件各自的 `lock_ref`、`evictable/protected_size`、淘汰驱动器；跨组件用 `eviction_priority` 级联（内部节点 Full 2 > SWA 1 > Mamba 0） | 各组件按自己的物理池水位独立淘汰；同时保证"路径数据"不会被拆断（SWA 需要沿路连续覆盖） |
| 6 | **HiCache 是第四个维度，不是第四棵树** | 同一批节点同时挂 device 值和 `host_value`，同时挂 device LRU 和 host LRU（指针槽错开 `ct` / `ct+N`） | 三级存储不再靠继承（`HiRadixCache`）实现，`enable_hicache` 只是同一棵树上的一组开关 |

补充两条"代价"（设计权衡，不是缺点）：

- **调用序列变复杂**：`insert` 从一次函数调用变成"泵状态机"的循环（§9），
  调用方必须保证 `end_insert()` 在 `finally` 里跑（[unified_radix_cache.py:412-414](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L412)）。
- **返回值必须被排空**：`BaseEvictionResult.__del__` 里有断言 tripwire（§4.4），
  漏了排空就会在 GC 时炸——这是刻意用运行时断言换取"不会静默泄漏 KV"。

---

## 3. 为什么要"统一"

### 3.1 旧世界：四套并列的基数树

| 实现 | 适用场景 | 文件 |
|---|---|---|
| `RadixCache` | 标准全注意力 | `radix_cache.py` |
| `SWARadixCache` | 滑动窗口注意力（Gemma、Llama4、GptOss、DSV4…） | `swa_radix_cache.py` |
| `MambaRadixCache` | Mamba / SSM 混合模型 | `mamba_radix_cache.py` |
| `HiRadixCache` / `HiMambaRadixCache` | 上述任一 + GPU↔Host↔Storage 三级 | `hiradix_cache.py` |

痛点很直接：

1. **组合爆炸**：形态数 × HiCache 与否 × 未来新形态 ⇒ 类数量乘法增长（`HiMambaRadixCache` 就是"交叉产物"）。
2. **重复代码**：radix 树的 match / split / insert / LRU / sanity 在四处各写一遍，修 bug 要修四次。
3. **难以叠加**：一个模型同时是 SWA + Mamba 混合（如 Mamba-SWA 混合架构）时，旧方案没有自然的表达方式。
4. **HiCache 靠继承注入**，Host/Storage 逻辑与树逻辑纠缠在同一个类里。

### 3.2 新世界：一棵树 + 可插拔组件

```
旧：   RadixCache      SWARadixCache     MambaRadixCache     HiRadixCache
        （各自一棵树、各自一套 match/insert/evict/LRU/sanity）

新：            ┌───────────  UnifiedTreeCore（一棵树，一套算法）  ──────────┐
                │  component_data[FULL] │ component_data[SWA] │ ...[MAMBA]  │
                └──────────┬────────────┴──────────┬──────────┴─────┬───────┘
                     FullComponent           SWAComponent     MambaComponent
                （装哪几个由 registry 按模型配置决定；HiCache 只是同一棵树的开关）
```

装配决策在 [registry.py:163-193](../../../python/sglang/srt/mem_cache/registry.py#L163) `_create_unified_radix_cache`：
`tree_components` 起始为 `[FULL]`，按 `is_hybrid_swa` 追加 `SWA`、按 mamba 配置追加 `MAMBA`，
写入 `params.tree_components` 后构造 `UnifiedRadixCache`，需要 HiCache 时再调 `init_hicache`。

启用开关：`SGLANG_ENABLE_UNIFIED_RADIX_TREE`（或 `use_mlx()`），见
[registry.py:104](../../../python/sglang/srt/mem_cache/registry.py#L104)。

---

## 4. 机制 / 控制器分层（本框架的骨架）

这是整套设计里最容易被忽略、但决定了代码形态的一层。

### 4.1 职责划分

| 关注点 | UnifiedTreeCore（机制） | UnifiedRadixCache（控制器） |
|---|---|---|
| 树结构（root/children/arena） | ✅ 唯一拥有 | ❌ 只能通过 NodeId 引用 |
| 节点上的 value / lock_ref / size 记账 | ✅ | ❌ |
| LRU / 叶集合 / 淘汰游标 | ✅ | ❌ |
| `req_to_token_pool`、`token_to_kv_pool_allocator` | ❌ 完全看不见 | ✅ 唯一拥有 |
| KV 释放（`free` / `free_segment`） | ❌ 只返回"该释放哪些下标" | ✅ 实际调用 allocator |
| D→H / H→D / H→Storage 传输 | ❌ 只生成 spec | ✅ 调 `cache_controller` |
| 分布式 all_reduce / PP sync | ❌ | ✅ |
| metrics / StreamingSession / sidecar | ❌ | ✅ |
| `sanity_check` 算法本体 | ✅ | 转发（[:1977](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1977)） |

两个类的方法表几乎一一对应：控制器上的 `match_prefix` / `insert` / `evict` / `inc_lock_ref` /
`evictable_size` 大多是"**转发到 tree_core + 执行返回的动作**"。例如
[unified_radix_cache.py:381-395](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L381)：

```python
def match_prefix(self, params):
    result = self.session.try_match_prefix(params)      # 流式会话短路
    if result is not None: return result
    if self.disable: return self.tree_core.empty_match_result
    result = self.tree_core.match_prefix(params)        # ① 树决定
    self._apply_cache_actions(result.cache_actions)     # ② Cache 执行
    for component in self._components_tuple:            # ③ 组件后处理（可能触发淘汰）
        result = component.finalize_match_result_in_cache(params, result)
    assert not result.cache_actions                     # 后处理不许再吐动作
    return result
```

### 4.2 NodeId：唯一跨界句柄

[unified_tree_core_interface.py](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core_interface.py)
头注释写得很明白：接口的存在目的是让"**另一种实现（例如 Rust TreeCore）不必继承 Python 树**就能满足契约"。
为此接口上一切节点都退化为 `NodeId = int`：

- 解析：`node_by_id(node_id)` 走 `_node_arena`（[unified_tree_core.py:407](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L407)），
  节点在 `_register_node` / `_unregister_node`（[:437](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L437)）时进出 arena。
- 出口收敛：`_match_post_processor` 最后一句
  [:674-679](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L674)
  统一把三个锚点换成 `.id`，注释 `# Expose only NodeIds outside TreeCore.`
- 控制器侧只有 `resolve_node_handle`（[:2052](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2052)）
  这种兼容旧调用方的窄口子。

### 4.3 延迟动作：CacheAction / ComponentAction

[cache_action.py](../../../python/sglang/srt/mem_cache/unified_cache/cache_action.py)（全文 103 行）
定义了树侧能"下单"的全部副作用，都是 `frozen=True` 的 `msgspec.Struct`。分两族：

**A. Cache-owned（控制器自己执行）** —— `CacheAction = ReplaceWriteThroughOnNodeSplit | FreeDeviceKV | BackupKV`

| 动作 | 语义 | 控制器落点 |
|---|---|---|
| `FreeDeviceKV(indices)` | 释放不再被引用的 device KV（SWA-aware 的联合释放） | `token_to_kv_pool_allocator.free_segment(indices, start_pos=0)`（[:783-801](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L783)） |
| `BackupKV(node_ids)` | 按序 D→H 备份，遇首个失败即停；write-through 的 id 是 **root-first 的父链**（父必在子之前），write-back 则只带一个淘汰牺牲者 | `_execute_and_commit_kv_backup`（[:829](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L829)） |
| `ReplaceWriteThroughOnNodeSplit(ack_id, old, new, new_child)` | 节点分裂后，把 in-flight write-through 记账里的旧节点换成 `new + new_child` | `_replace_pending_write_through_node`（[:884](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L884)） |

**B. Component-routed（按 `component_type` 派发给组件）** —— 基类 `ComponentAction`

| 动作 | 归属组件 | 语义 |
|---|---|---|
| `FreeComponentDeviceSlot` / `FreeComponentHostSlot` | 任意 | 只释放该组件的 device / host 槽 |
| `MambaEvictExcessPathStates(tail_node_id)` | MAMBA | 从 tail 的 root 路径上做 per-path state 数量上限淘汰；在 insert 的 **commit barrier** 执行 |
| `RebuildFullToSWAMapping(full_indices, swa_indices)` | SWA | 重建 allocator 的 full→swa 映射（load back 之后） |
| `RecoverSWAWithLockedFull(node_id, kept_full, incoming_full)` | SWA | SWA 墓碑复活但 full 被锁：保留已锁 full、把它重映射到新 full 的 SWA 翻译，只释放新来的 full |
| `SWARebuild(node_id, source_value)` | SWA | 用源 full value 翻译出该节点的 SWA value 并写回 |

派发一律走 `apply_component_action`（[tree_component.py:512](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py#L512)）：
`self.components[action.component_type].apply_component_action(action)`。

**执行纪律**（[:772-781](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L772)）：
`_apply_cache_actions` 先把列表 **reverse**，然后逐个 `pop()`——保证"已消费的动作不可能被重复执行"，
即使中途抛异常，剩余动作也还在列表里等下一次排空。

### 4.4 排空契约：`BaseEvictionResult.__del__` tripwire

淘汰类方法不直接释放内存，而是把"该释放什么"装进结果结构返回。为了防止调用方忘记排空，
[unified_tree_core_interface.py:22-42](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core_interface.py#L22)
在基类上放了一个析构断言：

```python
class BaseEvictionResult(msgspec.Struct):
    def __del__(self) -> None:
        # Drop tripwire: every returned value must be drained before disposal.
        assert not self.device_frees and not self.host_frees, \
            "BaseEvictionResult dropped with undrained values"
```

派生类：`EvictDeviceNextNodeResult` / `EvictDeviceLeafResult` / `DemoteResult` /
`DropSubtreeNoHostResult` / `DriveHostEvictionResult` / `DecSwaLockOnlyResult`。
控制器侧对应的排空器是 `_free_values`（[:442](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L442)），
它用 `try / finally` 保证 device 与 host 两路都会被排空；
`_drain_device_frees` / `_drain_host_frees`（[:803](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L803)）
是"逐 component pop 出来再释放"，pop 即消费。

### 4.5 TreeCore 后端可插拔

[tree_core_registry.py](../../../python/sglang/srt/mem_cache/unified_cache/tree_core_registry.py)：

```python
register_tree_core_backend("python", _python_tree_core_factory)   # 内置唯一实现
create_tree_core(name=envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get(), params, components)
```

- 默认 `"python"`；未注册的名字会抛出带"已注册列表"的 ValueError，并提示
  "External backends must call register_tree_core_backend(...) at import time."
- 工厂签名 `(CacheInitParams, dict[ComponentType, TreeComponent]) -> UnifiedTreeCoreInterface`，
  **返回接口类型而非具体类**——这就是 §4.2 里 NodeId 边界存在的意义所在。
- 另有一条独立的实验路径 `SGLANG_EXPERIMENTAL_CPP_RADIX_TREE`（属于老 `RadixCache`，与本框架无关，勿混淆）。

---

## 5. 数据结构

### 5.1 `UnifiedTreeNode`（[unified_tree_core.py:98](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L98)）

| 字段 | 说明 |
|---|---|
| `children: dict` | **刻意用普通 dict 而非 defaultdict**：读到缺失 key 必须抛错，绝不能悄悄造出一个未登记进 arena 的节点（[:102-104](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L102)） |
| `parent` / `key: RadixKey` | 树骨架；`key` 是页对齐的一段 token id |
| `component_data: list[ComponentData]` | **定长 `_NUM_COMPONENT_TYPES`**，按 `ComponentType` 整数枚举直接下标寻址（不是 dict） |
| `component_types: tuple` | 本次运行实际启用的组件（用于遍历） |
| `last_access_time` / `creation_time` | `float64` **逻辑时钟**（`get_and_increase_time_counter()`，[tree_component.py:96](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py#L96)），非墙钟 |
| `hit_count` | write-through 触发计数（§12.3） |
| `priority` | 优先级调度用；root 为 `-sys.maxsize` |
| `lru_prev` / `lru_next` | 长度 `_NUM_COMPONENT_TYPES * 2` 的定长数组（§5.3 指针槽） |
| `id` | 类计数器分配，就是对外的 `NodeId` |
| `write_through_pending_id` | 该节点正参与哪个 in-flight write-through 批次 |
| `hash_value: list[str]` | L3 storage 的分页 hash 链 |

两个只看 **Full** 的派生属性：

| 属性 | 定义 | 含义 |
|---|---|---|
| `backuped`（[:131](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L131)） | `component_data[FULL].host_value is not None` | Full KV 已备份到 host |
| `evicted`（[:136](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L136)） | 非 root 且 `component_data[FULL].value is None` | Full KV 不在 device（device 墓碑） |

> `evicted and not backuped` = **死节点**。match 遍历遇到它立即中断
> （[:594-595](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L594)）。

### 5.2 `ComponentData`（[tree_component.py:46](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py#L46)）

```python
@dataclasses.dataclass
class ComponentData:
    value: Optional[torch.Tensor] = None       # device 侧下标/槽（None = 墓碑）
    lock_ref: int = 0                          # device 引用计数
    metadata: dict[str, Any] = {}              # 组件私有（如 SWA 的 component_uuid）
    host_value: Optional[torch.Tensor] = None  # host 侧页（HiCache）
    host_lock_ref: int = 0                     # host 引用计数
```

一个节点上 device / host 各有独立的 value + lock_ref ⇒ **同一节点可以"device 已淘汰、host 还在"**，
这就是 HiCache 能作为"第四维度"叠加在同一棵树上的结构基础。

（注：`ComponentData` 仍是 `@dataclass`，属于仓库规则 `no-dataclasses` 的历史遗留豁免。）

### 5.3 `UnifiedLRUList` 与"指针槽分离"（[unified_tree_core.py:158](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L158)）

侵入式双向链表：指针不放在链表里，而是放在**节点自己的定长数组**上，槽位号：

```
_pt = component_type + (_NUM_COMPONENT_TYPES if use_host_ptr else 0)

lru_prev / lru_next 数组布局（N = _NUM_COMPONENT_TYPES = 3）：
 slot: [ 0    1    2  ][  3      4      5   ]
        FULL SWA MAMBA  FULL'  SWA'  MAMBA'
        └── device LRU ┘└──── host LRU ─────┘
```

于是同一个节点可以**同时**挂在「SWA 的 device LRU」和「Mamba 的 host LRU」上而指针互不干扰。
`head` / `tail` 是哨兵节点，`cache: dict[int, node]` 做 O(1) 成员判定。

**跳锁遍历**是淘汰的关键原语：`get_prev_no_lock`（[:244](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L244)）、
`get_prev_leaf_no_lock`（[:256](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L256)）、
`get_prev_no_host_lock`（[:268](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L268)）
从尾（最旧）往头走时自动跳过 `lock_ref > 0` 的节点，保证淘汰只命中真正可淘汰的对象。
MRU 提升有三种粒度：`reset_node_mru`、`reset_node_and_parents_mru`（整条 root 路径）、
`reset_node_and_window_ancestors_mru`（只到滑窗边界，SWA 专用，[:223](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L223)）。

---

## 6. 三种组件与 `TreeComponent` 钩子全景

### 6.1 三种组件对照

| 维度 | FullComponent | SWAComponent | MambaComponent |
|---|---|---|---|
| value 语义 | 全量 KV token 下标（**基座**，节点存在的理由） | 滑窗内的 KV 下标（**路径数据**：需要沿 root 路径连续覆盖） | SSM recurrent state 槽（**单点数据**：只在匹配边界有意义） |
| 挂哪种池 | `token_to_kv_pool_allocator` | `SWATokenToKVPoolAllocator`（构造时 assert） | `HybridReqToTokenPool`（构造时 assert；除 extra buffer 外要求 `page_size == 1`） |
| 淘汰追踪 | `evictable_device_leaves` / `evictable_host_leaves` **叶集合** + `last_access_time` 小顶堆 | 该组件的 `UnifiedLRUList` | 该组件的 `UnifiedLRUList` |
| 锁范围 | **path lock**：节点 → root 全路径 | **window lock**：从节点往上累加 value 长度直到 `sliding_window_size`，返回 `component_uuid` 作为边界标记 | **single-node lock**：只锁该节点 |
| 内部节点淘汰优先级 | 2（最后被淘汰） | 1 | 0（最先） |
| 叶节点淘汰优先级 | 0 | 0 | 0（叶上全部相等 ⇒ 任一被淘汰即级联全部，因为节点要整删） |
| 墓碑 | 只有 device 墓碑（`value=None` 但 `host_value` 还在） | 可在 Full 存活时单独墓碑化 | 同 SWA |

**为什么内部节点上 SWA(1) > Mamba(0)**（[tree_component.py:283-297](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py#L283) 的注释直接给了例子）：
`A→B→C→D→E`，C 与 E 都有 mamba state，窗口覆盖 `C→E`。若 C 的 mamba 被淘汰，
**C 的 SWA 必须留着**，否则 E 不再可达——SWA 是路径数据，Mamba 只在边界点有意义。

**SWA 的 window lock 细节**（[swa_component.py:472](../../../python/sglang/srt/mem_cache/unified_cache/components/swa_component.py#L472)）：
从节点往 root 走，累加各节点 SWA value 长度直到 ≥ `sliding_window_size` 就停；
沿途跳过墓碑节点（`cd.value is None` 没有可保护的 chunk）；
在 `metadata["uuid"]`（host 时 `"host_uuid"`）写入 uuid 作为"锁到哪儿"的边界标记，
后续 `dec_swa_lock_only` 依此提前释放越窗部分的 SWA 锁。

### 6.2 `TreeComponent` 钩子全景（[tree_component.py](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py)）

树侧在算法的每个"决策点"回调**全部组件**的同名钩子；组件把要做的副作用 append 进
`cache_actions` 列表，由控制器执行。按阶段分组：

| 阶段 | 钩子 | 何时调用 |
|---|---|---|
| **match** | `create_match_validator(match_device_only=)` | 生成"这个节点我认不认"的谓词；**所有组件都认才算命中**（`_all_valid`） |
| | `finalize_match_result_in_tree_core` | 树内后处理（改 `MatchResult` 字段，仍持有节点对象） |
| | `finalize_match_result_in_cache` | 控制器侧后处理（可能触发淘汰/分配，因此放在树外） |
| **insert** | `update_component_on_insert_overlap` | 走到一个已存在节点、且未被淘汰：组件"认领"重叠的 KV 槽，返回 `consumed_from` |
| | `recover_after_unevict` | 该节点刚被 Full 复活，aux 组件从同一段 value 重建自己的数据 |
| | `commit_insert_component_data` | COMMIT 阶段把物理数据挂到目标节点（Mamba 在此挂 state） |
| **split** | `redistribute_on_node_split` | 节点分裂时把 value / lock_ref / host_value 在 new_node 与 new_child 之间重分配 |
| **evict** | `eviction_priority(is_leaf)` | 级联优先级（见上表） |
| | `evict_device_start` / `evict_device_next_node` / `evict_device_end` | 组件自己的淘汰驱动器（游标/堆的建立、推进、收尾） |
| | `evict_component` | 真正释放该组件在该节点上的数据（`EvictLayer.DEVICE / HOST / ALL`） |
| | `drive_host_eviction` | host 层淘汰驱动 |
| **lock** | `acquire_component_lock` / `release_component_lock` | 三种锁范围的差异全在这两个钩子里 |
| **caching** | `prepare_for_caching_req` / `cleanup_after_caching_req` | `cache_(un)finished_req` 的前后置（如 Mamba 的 slot 处理） |
| | `free_out_of_window_slots` / `free_host_values` | 越窗释放 / host 释放 |
| **HiCache** | `prepare_load_back` / `finalize_load_back` / `prepare_prefetch` | H→D、Storage→H 的组件侧准备 |
| | `build_hicache_transfers` / `commit_hicache_transfer` | 按 `CacheTransferPhase`（`BACKUP_HOST` / `LOAD_BACK` / `BACKUP_STORAGE` / `PREFETCH`）生成与提交传输 |
| **通用** | `refresh_lru(phase, node, root)` | `WALKDOWN` / `MATCH_END` / `INSERT_END` 三个时机的 LRU 提升 |
| | `apply_component_action(action)` | 接收派发给自己的 `ComponentAction` |

> 组件把 `self.tree_core` 反向持有（在 `UnifiedTreeCore.__init__`
> [:351-352](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L351) 回填），
> 因此钩子里可以读树的 LRU / 叶集 / size 记账。

---

## 7. 三个心智模型

### 7.1 Tombstone（墓碑）：结构留下，数据消失

```
正常节点          device 墓碑             死节点（match 到此中断）
value=Tensor      value=None            value=None
host_value=None   host_value=Tensor     host_value=None
                  （backuped=True）      （evicted & not backuped）
```

墓碑的意义：**节点是"前缀路径"的载体**。数据可以走，但只要结构留着，
① 后续 insert 走到这里可以"原地复活"（不用重建路径）；
② HiCache 可以从 host / storage 把数据 load 回来；
③ 兄弟分支的前缀共享关系不被破坏。

代价是必须维护"**叶不能是墓碑**"这条不变量——否则会积累一串只有结构没有数据的死叶。
由 `_iteratively_delete_tombstone_leaf`（[:1348](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1348)）
在删除节点后向上迭代清理。

### 7.2 Leaf-set vs LRU：两套淘汰追踪结构并存

| 结构 | 服务组件 | 类型 | 淘汰粒度 |
|---|---|---|---|
| `evictable_device_leaves` / `evictable_host_leaves`（[:388-389](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L388)） | **Full** | `set[UnifiedTreeNode]` | 只淘汰**叶节点**，用 `last_access_time` 建小顶堆（[full_component.py:133-137](../../../python/sglang/srt/mem_cache/unified_cache/components/full_component.py#L133)） |
| `lru_lists[ct]` / `host_lru_lists[ct]`（[:384-393](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L384)） | **SWA / Mamba** | `UnifiedLRUList` 双向链表 | 内部节点可**就地墓碑化**、叶节点整删 |

为什么不统一成一种？因为语义不同：

- **Full 是基座**：淘汰内部节点等于切断其整棵子树的前缀路径，所以只能从叶往上剥。
  维护一个"当前哪些节点是可淘汰叶"的集合 + 每次按时间取最旧，比维护有序链表更贴合。
- **aux 组件可以墓碑化内部节点**（Full 还在，路径不断），所以用 LRU 链表按最近使用顺序走即可。

`last_access_time` 用 float64 逻辑时钟而非墙钟的原因：match 成功后从命中节点往上，
每步 `cur_time -= 0.00001`（[:639-643](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L639)），
使**父节点时间严格小于子节点**——保证 Full 叶淘汰时不会出现"父亲比子孙先被当作牺牲品"的倒挂。

### 7.3 Cascade（级联淘汰）+ 优先级

当某组件淘汰了一个节点上的数据，**同节点上优先级 ≤ 触发者的所有组件一并被淘汰**
（`_cascade_evict`，[:1253](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1253)）：

```
is_leaf = node ∈ evictable_{device,host}_leaves     （按 target 层判定）
trigger_priority = trigger.eviction_priority(is_leaf)

for comp in components:
    if comp.eviction_priority(is_leaf) <= trigger_priority and comp is not trigger:
        # ★ 特例：若 comp 的「真实内部优先级」≥ trigger 的内部优先级，
        #    说明它只是因为「叶塌陷把优先级压平成 0」才进到这个循环里，
        #    此时它上面的锁是合法 pin ⇒ continue 跳过；
        #    否则（真的是更低层级）带锁就是逻辑错误 ⇒ assert lock_ref == 0
        evict + 从对应 LRU 摘除
```

**一个微妙时序**：触发者是 Full 且 target 是 DEVICE 时，`Full.value = None` 被**推迟到级联结束之后**
（[:1304-1308](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1304)）——
因为 SWA 的 `free_swa` 需要读 Full.value 才能做 full→swa 翻译。注释原文：
`Now that all components (including SWA which depends on Full.value) have been freed, we can safely tombstone Full.value.`

---

## 8. 读路径：`match_prefix`

入口在控制器 [:381](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L381)（先经 StreamingSession 短路），
算法本体在 `tree_core.match_prefix`（[:509](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L509)）
= `_match_prefix_helper`（[:536](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L536)）
+ `_match_post_processor`（[:623](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L623)）。

### 8.1 难点一：一个节点要让"所有组件都认账"

```python
validators = tuple(comp.create_match_validator(...) for comp in self.components)
def _all_valid(validators, node):
    return all([v(node) for v in validators])
```

Full 说"我在 device 上"不够——如果这个节点的 SWA 已被墓碑化且滑窗需要它，SWA 的 validator 会否决。
**只有全部组件通过，这个节点才算合法命中点。** 纯全注意力时只有一个 validator ⇒ 退化为普通 radix match。

### 8.2 难点二：device / host 双锚点

HiCache 模式下 host-backed 的节点也能"匹配"（因为可以 load back 回来），
但**调度器拿到的 prefix indices 必须是 device 上真实存在的下标**。所以 helper 同时追踪两个锚点：

| 锚点 | 判定用的 validator | 用途 |
|---|---|---|
| `best_match_node` | `create_match_validator()`（device **或** host 都算） | 后续 `prefetch_from_storage` 的起点、`last_host_node` |
| `best_match_device_node` + `best_match_device_value_len` | `create_match_validator(match_device_only=True)` | 调度器的 `device_indices`（`torch.cat(value[:len])`）、加锁对象 |

非 HiCache 模式下 `separate_device_match = False`，两个锚点合一（只建一套 device-only validator），
省掉一半 validator 调用。

### 8.3 遍历与中断（[:590-621](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L590)）

```
while len(key) > 0 and child_key in node.children:
    child = node.children[child_key]
    if child.evicted and not child.backuped:   # ★ 死节点 → 立即 break
        break
    prefix_len = child.key.match(key, page_size)
    if prefix_len < len(child.key):            # 部分匹配 → 分裂后 break
        node, action = self._split_node(...)   # 分裂可能产生 CacheAction
        ...
        break
    value.append(child.component_data[FULL].value)   # 未淘汰才 append
    node = child;  key = key[prefix_len:]
```

注意 `_split_node` 会返回一个可能非空的 `action`（如
`ReplaceWriteThroughOnNodeSplit`），它被一路带到 `MatchResult.cache_actions`，
由控制器在 finalizer 之前执行（因为 finalizer 可能触发淘汰或抛异常）。

### 8.4 后处理（`_match_post_processor`）

1. **aux LRU 提升**：对每个非 Full 组件调 `refresh_lru(MATCH_END, ...)`（Full 用 `last_access_time` 不用 LRU）。
2. **递减时间戳**：从命中节点向上写 `last_access_time`，每步 `-= 0.00001`（§7.2）。
3. **确定 `last_host_node`**：HiCache 时 = `best_match_node`（"所有组件在 device+host 上都达成共识的节点"），
   否则 = `best_match_device_node`。
4. **拼 device_indices**：`torch.cat(value[:best_match_device_value_len])`。
5. 逐组件 `finalize_match_result_in_tree_core`。
6. **出口换 NodeId**：`result._replace(last_device_node=....id, last_host_node=....id, best_match_node=....id, cache_actions=[action] if action else [])`。

---

## 9. 写路径：可恢复 insert 状态机

### 9.1 为什么 insert 必须变成状态机

树侧不能执行副作用，但 insert 途中产生的动作**并非都能延后**：

- 可延迟（fire-and-forget）：`FreeDeviceKV`、`ReplaceWriteThroughOnNodeSplit`
  （`_is_deferrable_action`，[:803-806](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L803)）——攒批到下一个 barrier 一起执行。
- 不可延迟：`BackupKV`（D→H 备份，后续步骤依赖它已完成）、各类 `ComponentAction`
  ——必须**立刻**由控制器执行，然后树才能继续走。

于是 insert 被切成"**可挂起的多步**"：树走到产生不可延迟动作的地方就把动作交出去并挂起，
控制器执行完再 `resume_insert()` 继续。

### 9.2 三相位状态机

```
        begin_insert(params)
              │  page_align、bigram view(EAGLE)、touch root、写 priority
              │  key 为空 ⇒ 直接返回 InsertResult(prefix_len=0)
              ▼
        _InsertWalkState(phase=WALK, node=root, key, value, params, priority)
              │
   ┌──────────▼───────────────────────────────────────────────┐
   │ WALK：每次处理「一个已存在节点」                            │
   │   · child_key 不在 children  ⇒ 转 COMMIT                  │
   │   · 部分匹配 ⇒ _split_node（可能吐 action）                │
   │   · node.evicted ⇒ _unevict_node_on_insert 复活 Full      │
   │       + 逐 aux 组件 recover_after_unevict                 │
   │   · 否则 ⇒ 逐组件 update_component_on_insert_overlap       │
   │       取 min(consumed_from)，重复段吐 FreeDeviceKV          │
   │   · hit_count 到阈值 ⇒ 吐 BackupKV（write-through）        │
   └──────────┬───────────────────────────────────────────────┘
              ▼（key 走完 / 无子节点）
   ┌──────────────────────────────────────────────────────────┐
   │ COMMIT：建尾部新叶（_add_new_node），然后逐组件            │
   │         commit_insert_component_data（Mamba 在此挂 state）│
   │  「所有钩子先跑完，再执行它们吐出的动作；动作失败即 fail-stop │
   │    ⇒ 永远观察不到半提交状态」                              │
   └──────────┬───────────────────────────────────────────────┘
              ▼
   ┌──────────────────────────────────────────────────────────┐
   │ TAIL：aux 组件 refresh_lru(INSERT_END)；新叶若到阈值再吐一次 │
   │       BackupKV；返回 InsertStepResult(actions, result)     │
   └──────────────────────────────────────────────────────────┘
```

`_advance_insert`（[:779](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L779)）是泵体：

```python
while True:
    flushed_len = len(state.pending_actions)
    ... 执行当前相位一步 ...
    new_actions = state.pending_actions[flushed_len:]
    if new_actions and not all(map(self._is_deferrable_action, new_actions)):
        flushed, state.pending_actions = state.pending_actions, []
        return InsertStepResult(actions=flushed)      # 挂起
```

### 9.3 单飞（single-flight）与异常安全

- `begin_insert` 首行 `assert self._ongoing_insert_walk_state is None, "concurrent insert walks"`；
  控制器侧还有一道 `assert not self.tree_core.has_ongoing_insert(), "re-entrant insert"`
  （[:401](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L401)）——**不碰 in-flight 状态就先失败**。
- 控制器的循环包在 `try / finally` 里，`finally` 中
  `self._apply_cache_actions(self.tree_core.end_insert())`
  ——即使中途抛异常，也保证残留的 free 动作会到达 allocator（不泄漏 KV）。
- `end_insert()` 是幂等的：清空 `_ongoing_insert_walk_state` 并返回残留动作。

### 9.4 节点分裂 `_split_node`（[:914](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L914)）

```
   parent ──► node(key="ABCDE")        分裂点 prefix_len=3
               ↓
   parent ──► new_node(key="ABC") ──► new_child(key="DE")
```

分裂要做的三件事：① 逐组件 `redistribute_on_node_split` 把 value / lock_ref / host_value 切开；
② 维护 LRU 与叶集合（原节点若在某个 LRU 里，位置要转移）；
③ 若原节点正参与 in-flight write-through，吐出 `ReplaceWriteThroughOnNodeSplit`
让控制器把记账里的 `old_node_id` 换成 `new_node_id + new_child_node_id`。

---

## 10. 锁与 size 记账

### 10.1 组件分发式加锁（[:445-507](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L445)）

```python
def inc_lock_ref(self, node_id):
    node = self.node_by_id(node_id)
    result = IncLockRefResult()
    for component in self.components:
        result = component.acquire_component_lock(node=node, result=result)
    self._update_evictable_leaf_sets(node)      # 加/解锁都要重算叶资格
    return result
```

四个入口对称：`inc_lock_ref` / `dec_lock_ref` / `inc_host_lock_ref` / `dec_host_lock_ref`
（host 版就是多传 `lock_host=True`）。锁范围差异**全部封装在组件里**，树侧看不到"path/window/single"的区别。

`dec_lock_ref(..., skip_swa=True)` 用于"先只放 Full/Mamba 锁，SWA 锁另行处理"的场景。

### 10.2 三种锁范围

```
Full = path lock                SWA = window lock              Mamba = single-node
   root                            root                            root
    │ ↑lock                         │                               │
    A ↑                             A                               A
    │ ↑                             │                               │
    B ↑                             B  ← uuid 边界（累加到           B
    │ ↑                            ↑│    sliding_window_size 停）    │
    C(命中)                        ↑C(命中)                        C(命中) ↑lock
                                                                    只锁 C
```

### 10.3 `dec_swa_lock_only`：提前释放越窗的 SWA 锁

decode 推进后，早期的 SWA chunk 已经滑出窗口，可以先放锁让它被淘汰，
而 Full 锁还得留着（前缀还要复用）。
[:468-488](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L468)：

1. `swa_component.release_window_lock(node, swa_uuid_for_lock, ...)` 按 uuid 边界回退；
2. **顺带**释放同节点上"严格更低优先级"的锁（即 Mamba）——因为它们的存活依赖 SWA 覆盖。

### 10.4 size 记账：`evictable` ↔ `protected`

每组件两个全局量（`component_evictable_size_` / `component_protected_size_`，
[:381-382](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L381)）：

| 量 | 含义 | 变化时机 |
|---|---|---|
| `component_evictable_size_[ct]` | 该组件在树上、**未被锁**的数据量 | 加锁时 `-=`，解锁时 `+=`，淘汰时 `-=`，insert 新数据时 `+=` |
| `component_protected_size_[ct]` | 该组件在树上、**被锁**的数据量 | 与上互补 |

对外暴露的口径（控制器 [:1918-1943](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1918) 全是转发）：

| API | 返回 |
|---|---|
| `evictable_size()` / `protected_size()` | Full 的分量（基座口径） |
| `full_/swa_/mamba_evictable_size()`、`..._protected_size()` | 各组件分量 |
| `component_evictable_size(ct)` | 通用取法 |
| `total_size()` | `tuple`（[:2059](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L2059)） |

> 与旧 `SWARadixCache` 的差异：那里 `evictable_size()` 直接抛 `NotImplementedError` 强制用分量版；
> 统一树里 `evictable_size()` 有明确语义（= Full 分量），因为 Full 一定存在。

---

## 11. 淘汰全链路（evict）

淘汰是"机制/控制器分离"体现得最彻底的一条链路：**树决定淘汰谁，缓存执行释放**。
TreeCore 从头到尾不碰任何内存池。

### 11.1 一次 evict 的全景

```
Scheduler / allocator 发现显存不足
        │  EvictParams(num_tokens, swa_num_tokens, mamba_num)
        ▼
UnifiedRadixCache.evict()                         unified_radix_cache.py:416
        │  tracker = {FULL:0, SWA:0, MAMBA:0}      ← 各组件"已释放多少"累计器
        │  request_by_type = {FULL:num_tokens, SWA:swa_num_tokens, MAMBA:mamba_num}
        ▼
_evict_components(request_by_type, tracker)                             :497
        │
        │  for ct in tree_components:                   ← FULL → SWA → MAMBA
        │      if tracker[ct] >= request_cnt: continue  ← 已被捎带释放够了，跳过
        │      tree_core.evict_device_start(ct, request_cnt)   ← 建游标 / 堆
        │      while (node_id := next_node(ct)) is not None:
        │            evict_device_leaf(node_id)               ← 淘汰这一个叶
        │      tree_core.evict_device_end(ct)                 ← finally，必清游标
        ▼
write_back 策略下补一次 writing_check(write_back=True)                    :433
        ▼
EvictResult(num_tokens_evicted, swa_num_tokens_evicted, mamba_num_evicted)
```

四个关键设计点：

| 设计 | 说明 |
|---|---|
| **tracker 全组件共享** | Full 淘汰一个叶时会级联释放该节点上的 SWA/Mamba，这些字节记进同一个 tracker。轮到 SWA 时若 `tracker[SWA]` 已达标 ⇒ `continue` 跳过，不必再走一遍 LRU。这是统一树相对"四棵独立树"最直接的收益 |
| **start / next / end 三段式** | 游标（Full 是最小堆，SWA/Mamba 是 LRU 指针）建于 `evict_device_start`，`evict_device_end` 放在 `finally`，异常也不留悬空游标。组件基类用 `is_evict_device_ongoing` 断言防重入（[tree_component.py:310-337](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py#L310)） |
| **每步返回值必须排空** | `_evict_device_next_node` / `_evict_device_leaf` / `_demote` / `_drop_subtree_no_host`（[:463](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L463) / [:472](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L472) / [:482](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L482) / [:488](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L488)）是同一个模板：调 TreeCore → `_free_values(device_frees, host_frees)` → `_accumulate_tracker(tracker, result.tracker)`。漏排空会在 `__del__` 里炸（§4.4） |
| **回传 delta 而非总量** | `evict_device_next_node`（[:1035](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1035)）把 `tracker` 复制成 `updated_tracker` 给组件，返回时只回传差值，避免控制器重复累加 |

### 11.2 单个叶节点的淘汰：三条分支

`tree_core.evict_device_leaf(node_id, is_write_back)`
（[:1056-1083](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1056)）
先断言 `_is_device_leaf(node)`，然后按"有没有 host 备份 × 写策略"分三路：

```
                    node.backuped ?
                   ┌──── 是 ───────────────────────────┐
                   │                                   ▼
                   │                        _demote(node)              :1225
                   │                        显存释放、节点保留 → 墓碑（host-only）
                   ▼ 否
            is_write_back ?
        ┌──── 否（write_through）────┐   ┌──── 是（write_back）────┐
        ▼                                ▼
_delete_unbacked_device_leaf  :1153   result.backup_kv = BackupKV(...)  :1067
整节点删除（显存 + host 全释放）        直接 return，把动作交回控制器！
```

第三条分支是**延迟动作模型的典型用例**：write-back 下节点还没备份到 host，
不能直接扔，必须先做一次 D→H 拷贝；但 TreeCore 不能碰内存池、不能发 DMA，
于是把 `BackupKV` 装进返回值交给控制器
（[unified_radix_cache.py:512-534](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L512)）：

```python
backup_kv = self._evict_device_leaf(node_id, tracker)
if backup_kv is not None:
    written = self._execute_and_commit_kv_backup(backup_kv, write_back=True)  # D→H
    if written > 0:
        self.writing_check(write_back=True)   # 等 ack，确认 host 副本就绪
        self._demote(node_id, tracker)        # 才允许降级
    elif self._drop_subtree_no_host(node_id, tracker):
        logger.warning("write_back: KV subtree dropped without backup ...")
    else:
        logger.warning("... stays device-resident until host space frees")
```

三种结局：

| 结局 | 触发条件 | 后果 |
|---|---|---|
| 备份成功 → demote | host 池有空间，`written > 0` | 正常降级为墓碑，显存释放 |
| 备份失败 → 丢子树 | host 池满，且子树内无任何锁 | `drop_subtree_no_host`（[:1085](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1085)）把该叶及其**全部 host-only 后代**一起删掉，保证淘汰有进展 |
| 备份失败 → 放弃 | host 池满，且子树里有节点被锁 | 打 warning，该节点**继续占显存**，等 host 空间释放后下一轮再试 |

`drop_subtree_no_host` 的安全性靠三条断言撑着
（[:1092-1112](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1092)）：
调用前必须是 D-leaf；`not node.backuped and write_through_pending_id is None`
（失败的备份从未发起 DMA，没有在途读取）；每个后代必须 `evicted and backuped`
（host-only）——一个还在显存上的后代会与"本节点是 D-leaf"矛盾。

### 11.3 D-leaf / H-leaf 的判定：只看 Full

"叶资格"只由**基组件 Full** 说话，辅助组件不参与
（[:1411](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1411) /
[:1429](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1429)）：

| 判定 | 条件 |
|---|---|
| `_is_device_leaf`（D-leaf） | 非 root、`not node.evicted`（Full 显存值在）、**所有组件** `lock_ref == 0`、**没有任何子节点的 Full 还在显存上** |
| `_is_host_leaf`（H-leaf） | 非 root、`node.evicted`、`node.backuped`、所有组件 `host_lock_ref == 0`、**完全没有子节点** |

两者的"叶"含义不同：D-leaf **允许有子节点**（只要子节点 Full 已不在显存上），
H-leaf 要求 `len(children) == 0`。原因是显存层与 host 层是两棵**边界不同的重叠子树**——
显存子树是 host 子树的前缀部分，所以显存的"叶"在 host 视角看常常是内部节点。

### 11.4 级联淘汰与墓碑链清理

一个叶被淘汰后引发两级连锁：

**第一级：节点内组件级联** `_cascade_evict`（[:1253](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1253)，§7.3 已详述）——
淘汰某组件时，同节点上所有优先级 ≤ 它的组件一并释放。

**第二级：向上的墓碑链清理** `_iteratively_delete_tombstone_leaf`
（[:1348-1409](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1348)）：

```
cur = deleted_node.parent
while cur != root and len(cur.children) == 0:
    if 任一组件 lock_ref>0 or host_lock_ref>0:  break     ← 被锁，停
    if Full 显存值在:      更新叶集合; break              ← 它就是新的 D-leaf
    # Full 显存已无 —— 顺手清掉孤儿辅助组件的显存数据
    for comp: if 有显存数据: evict(EvictLayer.DEVICE)
    if Full host 值在:     更新叶集合; break              ← 它是新的 H-leaf
    # Full 两层都没了 —— 释放剩余 host 数据、删节点、继续往上
    for comp: if 有 host 数据: evict(EvictLayer.HOST)
    remove_leaf_from_parent(cur); cur = cur.parent
```

它维护了本方案最重要的一条不变量：**不允许存在"两层都没数据"的悬空节点**。
这种节点会让 `match_prefix` 在它那里断掉（`child.evicted and not child.backuped ⇒ break`，
§8.3），却仍白占 arena 里的 NodeId，是纯垃圾。

同时它顺手处理"孤儿辅助组件"：内部节点的 Full 被淘汰后，其上残留的 SWA/Mamba
已无意义（match 要求所有 validator 同时通过，Full 缺失必然失败），直接清掉。

### 11.5 统一的释放原语

所有释放路径（cascade / demote / host evict / 墓碑链 / drop subtree）最终都收敛到
`_evict_component_and_detach_lru`
（[:1318-1346](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1318)），它只做三件事：

1. 调 `comp.evict_component(node, target, device_frees, host_frees)` —— **组件把索引塞进 frees 字典，绝不直接还给分配器**；
2. 按 `target` 把释放字节数记进 `tracker`；
3. 从对应 LRU 链表（device 用 `lru_lists[ct]`、host 用 `host_lru_lists[ct]`）摘掉该节点。

第 3 步最容易漏：LRU 是侵入式链表（指针存在节点自己身上，§5.3），
若释放了数据却没摘链，节点会以"有数据"的身份继续被 LRU 游标选中，
`sanity_check` 的 `_check_lru_linked_list`（[:1980](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1980)）会抓到这个不一致。

### 11.6 host 层淘汰：由 host 池反压触发

host 层淘汰**不走 `evict()`**，而是 host 池分配不到 slot 时回调进来：

```
HostPoolGroup 分配失败
        ▼
cache.evict_host(...)                        unified_radix_cache.py:819
        ▼
tree_core.drive_host_eviction(ct, num_tokens)   unified_tree_core.py:1171
        ▼
component.drive_host_eviction(...)     ← 组件走自己的 host LRU 链
        ▼
_evict_host_leaf(node)                          :1186
        │  断言 _is_host_leaf
        │  for comp: evict(EvictLayer.ALL)      ← 一次性清全部组件，不做优先级级联
        │  evictable_host_leaves.discard(node)
        │  _remove_leaf_from_parent(node)       ← host 层淘汰 = 真删节点
        └─ _iteratively_delete_tombstone_leaf(...)
```

与 device 层的关键差别：**host 层淘汰一定是删节点**——H-leaf 既无子节点也无显存数据，
没有"降级"这一档可走；并且用 `EvictLayer.ALL` 一把清空，无需优先级级联，
因为整个节点都要消失（这正是 `eviction_priority` 在叶节点上"全部为 0"的原因，§7.3）。

### 11.7 复杂度

| 操作 | 复杂度 | 说明 |
|---|---|---|
| Full 淘汰一个叶 | `O(log E + C)` | 最小堆弹出 `O(log E)`（E = 可淘汰 D-leaf 数）+ 级联遍历 C 个组件 |
| SWA / Mamba 淘汰一个叶 | `O(C)` | LRU 尾指针 `O(1)` + 级联 |
| 墓碑链清理 | 摊还 `O(1)` | 每个节点一生只被删一次，代价摊到所有插入上 |
| 一次 `evict()` | `O(E·log E + L)` | E = 实际淘汰节点数，L = 墓碑链总长 |

对比旧世界：四套树各有一套淘汰实现，混合模型下要串行跑两遍（Full 树 + SWA 树），
且两棵树节点边界不一致、无法互相捎带释放。统一后 tracker 共享带来的"捎带"效应
在混合模型下能省掉相当一部分冗余遍历（§11.1 第一行 `continue`）。

---

## 12. HiCache 三级存储

统一树的第四个维度：同一棵树、同一批节点，同时描述 **L1 显存 / L2 主机内存 / L3 持久化存储**
三份副本的存在状态。**没有第二棵 host 树**——这是与旧 `HiRadixCache`（在 `RadixCache` 之上
再叠一层）最本质的差别。

### 12.1 三级数据流

```
              ┌───────────────────── L1: GPU KV Pool ──────────────────────┐
              │  ComponentData.value  （节点的显存副本）                    │
              └───────┬────────────────────────────────────▲───────────────┘
      BACKUP_HOST     │                                    │   LOAD_BACK
        (D→H)         │  write_stream (CUDA stream)        │  load_stream
                      ▼                                    │
              ┌───────────────── L2: Host Pool (pinned CPU) ───────────────┐
              │  ComponentData.host_value  （节点的 host 副本）             │
              │  HostPoolGroup 统一管理；满了反压 → evict_host（§11.6）      │
              └───────┬────────────────────────────────────▲───────────────┘
     BACKUP_STORAGE   │                                    │   PREFETCH
      (H→Storage)     │  backup_thread (daemon)            │  prefetch_thread
                      ▼                                    │
              ┌────────────── L3: Storage Backend ─────────────────────────┐
              │  file / mooncake / 3FS / hf3fs ...  （跨请求、跨进程持久化） │
              └────────────────────────────────────────────────────────────┘
```

四个方向恰好对应 `CacheTransferPhase` 的四个枚举值
（[tree_component.py:81-86](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py#L81)）：

| 相位 | 方向 | 触发者 | 异步载体 |
|---|---|---|---|
| `BACKUP_HOST` | D→H | write-through（`hit_count` 达阈）/ write-back（淘汰时，§11.2） | `write_stream` CUDA 流 |
| `LOAD_BACK` | H→D | `init_load_back`（命中 host 前缀，请求要用） | `load_stream` CUDA 流 |
| `BACKUP_STORAGE` | H→L3 | `write_backup_storage`（host 副本落盘） | `backup_thread` 守护线程 |
| `PREFETCH` | L3→H | `prefetch_from_storage`（新请求命中远端前缀） | `prefetch_thread` + `prefetch_io_aux_thread` |

**L1↔L2 用 CUDA 流，L2↔L3 用 Python 线程**——这个划分不是随意的：D↔H 是 DMA，
GPU 侧排队即可，用 stream + event 最省 CPU；H↔Storage 是阻塞式 IO（文件/网络），
必须真的开线程才不卡住 scheduler 主循环。
（[cache_controller.py:298-299](../../../python/sglang/srt/managers/cache_controller.py#L298) /
[:368-383](../../../python/sglang/srt/managers/cache_controller.py#L368) /
[:1052](../../../python/sglang/srt/managers/cache_controller.py#L1052)）

### 12.2 `init_hicache`：装配与参数

[unified_radix_cache.py:296-372](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L296) 干六件事：

1. **layout 修正（必须在建池之前）**：`hicache_io_backend == "direct"` 且 `mem_layout == "page_first"`
   ⇒ 通过 `server_args.override("hicache.mem_layout_force", ...)` 强改成 `page_first_direct`，
   并打 warning。原因见 `hicache_usage_and_design.md` §5：`page_first` 里固定 layer 后
   页内 token 是 strided 的，只能靠 GPU gather kernel 搬；`page_first_direct` 把 `page_size`
   维挪到 `layer` 之后，(page, layer) 固定后页内 token 物理连续，才能整块 memcpy。
2. 解析一次 storage 配置（`parse_storage_backend_extra_config`），与 assembler、树共享；
3. 调 `attach_hybrid_pool_to_unified_cache(...)` 真正建 host 池 / controller / sidecar；
4. `cache_controller is not None` ⇒ `tree_core.set_hicache_enabled()`，并把
   `has_swa_host_pool` 打到树上（决定 SWA 是否参与 host 层逻辑）；
5. 设策略常量；
6. storage 存在时 `_apply_storage_runtime_config(...)`。

关键常量：

| 字段 | 取值 | 含义 |
|---|---|---|
| `write_through_threshold` | `1 if write_policy == "write_through" else 2` | 节点 `hit_count` 达到该值才触发 D→H 备份。write_through 是"第一次插入就备"，其它策略要"被复用过一次"才备 |
| `is_write_back` | `write_policy == "write_back"` | 决定 `evict_device_leaf` 走哪条分支（§11.2） |
| `load_back_threshold` | `10` | 少于 10 token 的 host 命中不值得搬回显存 |
| `prefetch_threshold` | 默认 `256` | 少于该长度不发起 L3 预取 |
| `prefetch_timeout_base` / `_per_ki_token` | `1.0` / `0.25` 秒 | 预取超时 = base + 页数 × per_page；`_apply_storage_runtime_config` 里换算 `per_page = page_size/1024 × per_ki_token` |

### 12.3 write-through 的三段式：发起 → 追踪 → ack

D→H 不是同步完成的，所以中间必须有一个"在途"登记簿 `ongoing_write_through`：

```
_execute_and_commit_kv_backup(BackupKV)              :829
  for node_id in action.node_ids:                    ← 自上而下，遇失败即停
      if tree_core.is_backuped(node_id): continue    ← 链式动作重叠，跳过已备份
      device_value, comp_xfers = tree_core.build_backup_spec(node_id)   ← 树给规格
      host_indices = _execute_kv_backup(...)         ← host 不够先 evict_host
      if host_indices is None: return 0              ← 失败，交回 §11.2 的三种结局
      tree_core.commit_backup(node_id, host_indices, comp_xfers)  ← 记 host_value
      lock_params = inc_lock_ref(node_id) if not write_back else None  ← 钉住显存源
      _track_write_through_node(node_id, lock_params)              :874
```

`_track_write_through_node` 做两件事：`tree_core.mark_write_through_pending(node_id)`
（树上打在途标记）+ 在 `ongoing_write_through[node_id]` 记
`_OngoingWriteThrough(lock_node_id, lock_params, publish_node_ids)`。

**为什么要 `publish_node_ids` 这个列表？** 因为 D→H 在途期间，节点可能被 insert 分裂！
这时 TreeCore 会发一个 `ReplaceWriteThroughOnNodeSplit` 动作，控制器
`_replace_pending_write_through_node`（[:884](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L884)）
把登记簿里的一个 node_id 换成分裂后的两个。ack 到达时
`_finish_write_through_ack`（[:910](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L910)）
按更新后的列表逐个 publish，并且**每个碎片都要单独落 L3**——注释说得很直白：
"after a split, lock_node only holds the suffix; the prefix fragment must be persisted as well."

这是延迟动作模型（§4.3）除淘汰之外的第二个核心用例：**在途 IO 与树结构变更的解耦**。

### 12.4 `load_back`：H→D 的加锁顺序

[:923-958](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L923) 的顺序是有讲究的：

```python
host_anchor_params = self.inc_host_lock_ref(node_id).to_dec_params()  # ① 先钉 host 源
result = self.inc_lock_ref(node_id)                                   # ② 再钉 device 路径
ancestor_lock_params = result.to_dec_params()
preps = {c.component_type: c.prepare_load_back(node_id, req=req) for c in components}  # ③
success = False
try:
    success = self._load_back_transfers(...)
finally:
    for comp in components:
        comp.finalize_load_back(req, preps[comp.component_type], success)   # ④ 失败回收
```

- ① 必须先加 **host 锁**，否则构建 transfer 的过程中 host 层淘汰可能把源数据抽走；
- ② 注释写明 "Lock the path before building transfers (the aux build can evict)"——
  构建辅助组件的 transfer 本身会触发显存分配，进而触发淘汰，不先钉住路径就可能自己把自己淘汰掉；
- ③④ 组件级预分配 + `finally` 回收：例如 Mamba 需要先抢一个 device mamba slot
  （`PrepareLoadBackResult.allocated_mamba_slot`），失败路径必须还回去。

调度器侧入口是 `init_load_back`（[:1799](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1799)），
它只在"Full 显存已被淘汰 **或** 三种 host 命中长度之一 > 0"时才尝试搬回，
成功后用 `collect_full_device_indices(best, last_device)` 取回**新增**的那段 device 索引。

### 12.5 ack 轮询与跨 rank 一致性

每个 scheduler step 调一次 `check_hicache_events`
（[:1841](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1841)）：

```
_drain_async_work()          ← 先回收上一轮 PP-sync 的 isend handle（背压）
writing_check()              ← D→H 完成检查
loading_check()              ← H→D 完成检查
drain_storage_control_queues()  ← L3 控制队列（若开 storage）
log_storage_metrics()
```

`writing_check` / `loading_check`（[:1724](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1724) /
[:1763](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1763)）最关键的一句注释：

> Every rank must enter the all_reduce below; `ongoing_write_through` can diverge across ranks
> (e.g. a backup returning 0 on a subset).

即：**各 rank 的在途集合可能不同**（某个 rank 的 host 池刚好满、备份返回 0），
但树结构必须在所有 rank 上保持一致。做法是：

```python
finish_count = 0
if self.pp_rank == 0:                      # 只有 PP 首级数完成数
    for ack in cc.ack_write_queue:
        if not ack.finish_event.query(): break   # 队列有序，遇未完成即停
        finish_count += 1
_all_reduce(finish_count_tensor, ReduceOp.MIN)   # 取全局最小
while finish_count > 0: ...                      # 只处理所有 rank 都完成的前缀
```

三个要点：

| 要点 | 原因 |
|---|---|
| `ReduceOp.MIN` | 只推进"所有 rank 都已完成"的那段前缀，保证树在各 rank 上同步演进 |
| 队列有序 + 遇未完成即 `break` | ack 队列按发起顺序排列，前面没完成后面即使完成也不能提前处理 |
| 仅 `pp_rank == 0` 计数 | PP 各级共享同一棵树的逻辑视图，由首级做权威判定，其余级贡献 0 再取 MIN——等价于"跟随首级" |

`write_back=True` 的分支不同：它是**阻塞式**的（`ack.finish_event.synchronize()` 死等），
因为淘汰路径上的调用者正等着这块显存，不能推迟到下一个 step。

跨 rank 的通信组由 `_all_reduce_attn_groups`（[:207](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L207)）
和 `_pp_sync`（[:250](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L250)，标签
`P2PTag.HIRADIX_PP_SYNC`）承担，涉及 `tp_cache_group` / `attn_cp_cache_group` /
`attn_tp_cache_group` / `pp_cache_group` 四个组。

### 12.6 sidecar pool：给"额外池"留的扩展位

有些模型的 KV 不止一个池（典型是 DSV4 的六子池：swa / c4 / c128 / c4_indexer / 两类压缩态）。
这些池不适合抽象成"组件"（它们没有独立的 match 语义、生命周期完全跟随 Full），
于是走 `register_sidecar_pool(SidecarPoolSpec)`（[:374](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L374)）
注册，在每个传输相位由 `_build_sidecar_transfers`
（[:1028](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1028)）自动生成对应的
`PoolTransfer` 挂到主 KV 传输后面。

一句话区分：**组件 = 有独立 match/lock/evict 语义的一份数据；sidecar = 跟着 Full 一起搬的附属数据。**

---

## 13. 请求生命周期

把前面所有部件串成一条时间线。以一个普通生成请求为例：

```
① 调度：命中前缀
   scheduler → cache.match_prefix(MatchPrefixParams(key))          :381
             → req.prefix_indices / req.last_node / req.cache_protected_len
             → cache.inc_lock_ref(last_node)          ← 钉住命中路径，防被淘汰
   （HiCache）若 Full 显存已淘汰或有 host 命中 → init_load_back()   :1799

② 分配：为新 token 申请显存
   allocator.alloc_*  → 不够 → cache.evict(EvictParams(...))       :416

③ forward：模型写 KV 到 out_cache_loc

④ 落树：
   未完成（chunked prefill / decode 中途）→ cache_unfinished_req()  :664
   已完成（EOS / 达长度上限）              → cache_finished_req()   :581
```

### 13.1 `cache_unfinished_req`：插入 + 重新 match + 换锁

[:664-768](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L664) 的骨架：

```python
token_ids = req.get_fill_ids()
insert_params = InsertParams(prev_prefix_len=req.cache_protected_len, chunked=chunked, ...)

# ① 组件各自准备数据，并各自给出"有效缓存长度"的意见，取最小
effective_cache_len = len(token_ids)
for comp in components:
    cl = comp.prepare_for_caching_req(req=req, insert_params=..., is_finished=False)
    if cl is not None: effective_cache_len = min(effective_cache_len, cl)

if effective_cache_len <= 0:            # 全都不该缓存 → 只回填 prefix_indices 就返回
    ...; return

# ② 页对齐后插入
radix_key = RadixKey(token_ids[:eff], req.extra_key, is_bigram=is_eagle).page_aligned(page_size)
values = kv_indices[:len(radix_key)].to(torch.int64, copy=True)
result = self.insert(insert_params)                    # ← §9 的状态机

# ③ 立刻重新 match：拿到"插入之后"的规范路径
match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
req_to_token_pool.write(...)                           # 回填去重后的槽号

# ④ 换锁：先放旧的，再锁新的（顺序不可反）
self.dec_lock_ref(req.last_node, DecLockRefParams(swa_uuid_for_lock=...))
lock_result = self.inc_lock_ref(new_last_node)

# ⑤ 更新请求状态
req.prefix_indices / req.cache_protected_len / req.last_node / req.swa_uuid_for_lock
```

三个容易被忽略的点：

| 点 | 解释 |
|---|---|
| **插入后必须重新 match** | insert 可能把重复的 KV 槽合并掉（`update_component_on_insert_overlap` 让组件声明"我接管了这段槽"），重新 match 才能拿到**去重后的规范索引**，然后回填 `req_to_token`。不重新 match 会导致同一份 KV 有两个槽号，释放时双重释放 |
| **`effective_cache_len` 是组件投票的最小值** | SWA 靠 `swa_evicted_seqlen` 表达"这段已滑出窗口"，Mamba 靠 `mamba_last_track_seqlen` 表达"我只跟踪到这里"。返回 `None` 表示"没意见" |
| **页对齐 + 尾部单独释放** | `radix_key.page_aligned(page_size)` 只把整页放进树，不足一页的尾巴不入树。`cache_finished_req` 里这个尾巴与"截断产生的尾巴"合并成 `segments` 一次性 `free_segments`，注释说明是为了"a shared boundary page is emitted once"（共享边界页只发一次） |

### 13.2 `cache_finished_req`：只插入不重 match

[:581-662](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L581) 与上面的差别：

| 差异 | `cache_unfinished_req` | `cache_finished_req` |
|---|---|---|
| 插入后重新 match | **要**（请求还要继续跑，必须拿到规范索引） | **不要**（请求已结束，没人再用 `prefix_indices`） |
| 锁 | 换锁（dec 旧 + inc 新） | 只 `dec_lock_ref`，不再加锁 |
| `is_insert=False` 分支 | 无 | 有——直接把 `kv_indices[cache_protected_len:]` 还给分配器，完全不入树 |
| `kv_len_to_handle` | 隐含 `len(get_fill_ids())` | 显式参数（投机解码下已提交长度 ≠ 序列长度） |
| 截断处理 | 直接切 | 延迟到最后与非对齐尾巴一起 `free_segments` |

`disable` 短路分支（`:587`）也值得注意：即使前缀缓存关掉了，`cleanup_after_caching_req`
**仍然要调**——因为 Mamba 之类组件持有的是"请求级"资源（ping-pong buffer、mamba slot），
与树是否启用无关。

### 13.3 组件的两个请求级钩子

| 钩子 | 时机 | 典型实现 |
|---|---|---|
| `prepare_for_caching_req` | insert 之前 | Full：no-op 返回 `None`；SWA：`is_finished` 时设 `insert_params.swa_evicted_seqlen`；Mamba：准备 `mamba_value`（finished 从 ping-pong buffer 取、unfinished 从 req fork），返回 `mamba_last_track_seqlen` |
| `cleanup_after_caching_req` | 所有路径的最后（包括 early return 与 disable） | 释放组件的请求级资源。`insert_result is None` ⇒ "本次没有真正插入"，组件据此决定是否要把资源直接还掉 |

**这一对钩子的严格对称性是本方案里最容易踩坑的地方**：`cache_unfinished_req` 有三处
return（disable / `effective_cache_len <= 0` / 正常尾部），三处都调了
`cleanup_after_caching_req`，只是 `insert_result` / `insert_params` 传值不同。
新增 early return 时漏掉这一步就是资源泄漏。

---

## 14. 装配与注册：一个 UnifiedRadixCache 是怎么长出来的

### 14.1 选择链：`default_radix_cache_factory`

[registry.py:79-160](../../../python/sglang/srt/mem_cache/registry.py#L79) 是一条自上而下的短路链：

```
① disable_radix_cache + chunked_prefill ⇒ ChunkCache 家族（无树）
      ├─ 非 hybrid_swa            → ChunkCache
      ├─ full_tokens_per_layer==0 → PureSWAChunkCache
      └─ 其它                     → SWAChunkCache
② SGLANG_EXPERIMENTAL_CPP_RADIX_TREE ⇒ RadixCacheCpp（独立 C++ 路径，与统一树无关）
③ SGLANG_ENABLE_UNIFIED_RADIX_TREE 或 use_mlx() ⇒ ★UnifiedRadixCache（显式开关）
④ is_hybrid_swa
      ├─ full_tokens_per_layer==0 → PureSWARadixCache
      └─ 其它                     → ★UnifiedRadixCache      ← SWA 模型的默认实现
⑤ is_hybrid_ssm                   → ★UnifiedRadixCache      ← Mamba 模型的默认实现
⑥ enable_hierarchical_cache
      ├─ hybrid_ssm / hybrid_swa / dsa → ★UnifiedRadixCache  ← 注释："launch HiCache
      │                                   via UnifiedRadixCache by default"
      └─ 其它                          → HiRadixCache
⑦ enable_lmcache → LMCRadixCache ／ enable_flexkv → flexkv 工厂
⑧ 兜底 → RadixCache（纯 full-attention、无 HiCache）
```

值得强调：**统一树已经是混合模型（SWA / Mamba）与混合模型 HiCache 的默认实现**（第 ④⑤⑥ 条），
`SGLANG_ENABLE_UNIFIED_RADIX_TREE` 只是"连纯 full-attention 模型也用统一树"的额外开关。
旧的 `SWARadixCache` / `MambaRadixCache` 类还在文件里，但 registry 已不再引用。

### 14.2 组件选择：`_create_unified_radix_cache`

[registry.py:163-193](../../../python/sglang/srt/mem_cache/registry.py#L163)，只有 30 行，逻辑极简：

```python
tree_components = [ComponentType.FULL]              # Full 永远在
if ctx.is_hybrid_swa:   tree_components.append(ComponentType.SWA)
if ctx.is_hybrid_ssm:   tree_components.append(ComponentType.MAMBA)
params.tree_components = tuple(tree_components)

if use_mlx() and ctx.is_hybrid_ssm:                 # MLX 后端换掉 Mamba 组件实现
    params.component_registry_override = {ComponentType.MAMBA: MlxAuxiliaryStateComponent}

cache = UnifiedRadixCache(params)
if ctx.enable_hierarchical_cache:
    cache.init_hicache(server_args, params)
    ctx.tp_worker.register_hicache_layer_transfer_counter(
        cache.cache_controller.layer_done_counter)
return cache
```

**"模型能力 → 组件列表"的映射就这三行**。这正是"消除组合爆炸"的落地形态：
`{Full}` / `{Full,SWA}` / `{Full,Mamba}` / `{Full,SWA,Mamba}` 四种组合走的是同一条代码路径，
只是列表长度不同；再叠 HiCache 也只是多调一次 `init_hicache`，不产生新类。

`component_registry_override` 是硬件后端的扩展点（MLX 用自己的 `MlxAuxiliaryStateComponent`
替换默认 `MambaComponent`），配合 §4.5 的 TreeCore 后端注册表，构成两个正交的可插拔维度。

### 14.3 `UnifiedRadixCache.__init__` 的两阶段构造

[unified_radix_cache.py:121-205](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L121)，顺序不可调换：

```
① 池与开关：req_to_token_pool / token_to_kv_pool_allocator / disable / metrics
② 建组件：components = {ct: registry[ct](self, params) for ct in tree_components}
   ← 组件此刻只拿到 cache 引用，tree_core 还是 None
③ 抽取初始化期常量（遵循 general-code-style 的 "extract init-static values"）：
   is_swa_enabled / is_mamba_enabled / enable_mamba_extra_buffer / _sliding_window_size
④ 建 TreeCore：create_tree_core(name=env 后端名, params, components=self.components)
   ← 把已建好的组件交给树，树在自己 __init__ 里也会回填 component.tree_core
⑤ 回填 component.tree_core = self.tree_core        ← 与 ④ 内部重复但幂等
⑥ StreamingSession(inner=self)                    ← 常开，非流式请求零开销
⑦ 分布式组：tp / attn_cp / attn_tp / pp cache group + work_list
⑧ HiCache 字段设默认值（cache_controller=None 等），等 init_hicache 覆盖
⑨ self.reset()                                     ← 真正建 root_node / arena / LRU
```

两阶段的必然性：**组件与 TreeCore 互相持有引用**，只能先建组件（组件构造时只需 cache），
再建 TreeCore（需要组件字典），最后回填。第 ⑧ 步"先给 HiCache 字段赋 `None`/默认值"
是项目 `no-getattr-defensive` 规则的直接体现——字段永远存在，下游用 `is not None` 判断，
不用 `hasattr`。

### 14.4 HiCache 装配：strategy 表驱动

`init_hicache` 调 `attach_hybrid_pool_to_unified_cache`
（[hybrid_pool_assembler.py:1344](../../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L1344)）：

```python
kvcache = params.token_to_kv_pool_allocator.get_kvcache()
components = set(cache.components.keys())
strategy = _select_strategy(kvcache, components)      # 按 (池类型, 组件集合) 匹配
result = strategy.build(cache=..., kvcache=..., params=..., server_args=..., ...)
_apply_stack_result(cache, kvcache, params, result)
```

`_STRATEGIES`（[:1287](../../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L1287)）
是一个**有序列表**，first-match-wins：

| 顺序 | Strategy | 匹配场景 |
|---|---|---|
| 1 | `_DeepSeekV4Strategy` | DSV4 六子池（多压缩层 + sidecar） |
| 2 | `_MambaStrategy` | `{Full, Mamba}` |
| 3 | `_SwaStrategy` | `{Full, SWA}` |
| 4 | `_MambaSwaStrategy` | `{Full, SWA, Mamba}` |
| 5 | `_DsaStrategy` | DSA（DeepSeek V3.2 / GLM-5.1） |
| 6 | `_MiniMaxSparseStrategy` | MiniMax 稀疏 |
| 7 | `_PlainKvStrategy` | 兜底：普通 MHA/MLA `{Full}` |

匹配失败会抛 `AssertionError`，把 kvcache 类型和组件集合都打出来。
外部 fork 可以用 `register_stack_strategy(strategy)`（[:1298](../../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L1298)）
**前插**自己的策略——这是第三个可插拔维度（前两个是 TreeCore 后端和组件注册表）。

`_apply_stack_result`（[:1314](../../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L1314)）
把 strategy 的产出装回 cache：`host_pool_group`、`cache_controller`、
各组件的 host 池（`_COMPONENT_HOST_ATTR` 映射表同时写 cache 属性和组件属性）、
sidecar 注册、以及 `kvcache.register_layer_transfer_counter(layer_done_counter)`
（让 attention 后端能知道"第几层的 KV 已搬完"，实现逐层流水加载）。

---

## 15. 不变量与 `sanity_check`

`sanity_check` 是理解这套设计最快的入口——**它把所有隐式约定写成了可执行的断言**。
[unified_tree_core.py:1747-1969](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1747)
分五个 PART，收集错误到列表里最后一次性抛（而不是第一个断言就炸），
失败时先 `logger.error` 再 `pretty_print()` 打整棵树。
控制器侧只是转发（[unified_radix_cache.py:1977](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1977)），
把两个在途登记簿 `ongoing_write_through` / `ongoing_load_back` 传进去。

### PART 1：树结构

| 不变量 | 检查 |
|---|---|
| root 永远有 Full 显存值 | `root.component_data[FULL].value is not None` |
| root 永远被锁 | `root...lock_ref > 0`（`reset` 里把每个组件的 `lock_ref` 都初始化为 1） |
| root 无父 | `root.parent is None` |
| 父子双向一致 | `child.parent is node`，且非 root 节点 `key is not None` |

### PART 2：节点状态机与叶资格

这是全部五个 PART 里信息量最大的一段：

| 不变量 | 含义 |
|---|---|
| **辅助组件依赖 Full** | `aux.value is not None ⇒ Full.value is not None`；host 侧同理。Full 是树的骨干，SWA/Mamba 只能寄生在有 Full 的节点上 |
| **不允许死节点** | `not full_dev and not full_hst ⇒ 报错 "node dead"`。这条正是 §11.4 墓碑链清理在守护的目标 |
| **前缀单调性（device）** | 子节点在显存上 ⇒ 父节点也必须在显存上。淘汰只能自下而上（D-leaf 优先），不允许"中间被掏空" |
| **前缀单调性（host）** | 子节点已备份 ⇒ 父节点也已备份。**但 write_back 模式豁免**（`and not self.is_write_back`）——write-back 是淘汰时才备份，天然会出现"子备份了父没备"的中间态 |
| **锁层级** | `Full.lock_ref >= aux.lock_ref`（对每个辅助组件）。因为 Full 是路径锁、辅助是窗口/单点锁，后者是前者的子集 |
| **锁与数据一致** | `cd.value is None and cd.lock_ref > 0 ⇒ 报错`。墓碑不能被锁——锁的是数据，数据没了锁必须已经放掉 |
| 计数器非负 | `lock_ref >= 0`、`host_lock_ref >= 0` |

同时一趟遍历重算 `expected_dev_leaves` / `expected_hst_leaves` 供 PART 3 对账。

### PART 3：追踪结构对账

| 检查 | 说明 |
|---|---|
| 叶集合精确相等 | `evictable_device_leaves == expected_dev_leaves`（并打出 extra/missing 各前 5 个 id） |
| **D-leaf ∩ H-leaf = ∅** | 一个节点不可能同时是"显存叶"和"host 叶"——H-leaf 要求 `node.evicted`，D-leaf 要求 `not node.evicted`，互斥 |
| 无陈旧节点 | 叶集合里的节点必须都能从 root 遍历到（`- all_node_set` 为空） |
| **Full 的 LRU 必须为空** | `len(lru_lists[FULL].cache) == 0`——Full 用叶集合 + 最小堆，不用 LRU（§7.2 的双机制在这里被硬约束住） |
| 辅助组件 device LRU 精确对账 | `{有 value 的节点} == set(lru.cache.keys())` |
| 辅助组件 host LRU 精确对账 | `{value is None and host_value is not None 的节点} == set(host_lru.cache.keys())`——即"仅 host 态"的节点集合 |
| 同一节点不得同时在两条 LRU | `lru_ids & host_lru_ids == ∅` |
| 链表自身完整性 | `_check_lru_linked_list`（[:1980](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1980)）遍历 prev/next 指针，验证长度、首尾、双向一致——这是侵入式链表（§5.3）唯一的兜底 |

### PART 4：size 记账

对每个组件，遍历全树重算 `evictable` / `protected`（按 `lock_ref > 0` 分桶，`len(cd.value)` 计数），
与 `component_evictable_size_[ct]` / `component_protected_size_[ct]` 精确比对。
这条把 §10.4 里散落在几十处的 `+=` / `-=` 全部兜住了——**任何一处漏改都会在这里暴露**。

### PART 5：在途操作

`ongoing_write_through` / `ongoing_load_back` 里的每个 node_id 必须：
① 能在 `_node_arena` 里查到且仍在树上；② `Full.lock_ref > 0`。

第 ② 条是在途 IO 的安全底线：D→H 的源、H→D 的目标路径，在 DMA 完成之前必须被钉住，
否则淘汰会把正在被 DMA 读写的显存槽还给分配器。

> `StreamingSession` 打开时 `sanity_check` 会被跳过——流式会话期间树上存在临时的、
> 不满足上述不变量的中间态节点。

---

## 16. 关键约束与易错点

这一节是"上手前必须知道的坑"清单。前半是硬约束（写错就崩，或者更糟——静默出错），
后半是两个当前代码里读出来的问题。

### 16.1 环境变量三兄弟

| 变量 | 默认 | 定义处 | 作用 | 易错点 |
|---|---|---|---|---|
| `SGLANG_ENABLE_UNIFIED_RADIX_TREE` | `False` | [environ.py:849](../../../python/sglang/srt/environ.py#L849) | 把统一树**扩展到普通 full-attention 模型** | ⚠️ 它**不是**统一树的总开关。混合 SWA / 混合 SSM / 混合 HiCache 模型走 registry 链的 ④⑤⑥ 步，**默认就已经是统一树**（§14.1）。看到它是 `False` 就断言"没用统一树"是最常见的误判 |
| `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND` | `"python"` | [environ.py:851](../../../python/sglang/srt/environ.py#L851) | 选 TreeCore 后端实现（§4.5） | 名字必须已 `register_tree_core_backend` 注册，否则 `create_tree_core` 直接抛错。这是为将来 Rust TreeCore 预留的入口 |
| `SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS` | `False` | [environ.py:1061](../../../python/sglang/srt/environ.py#L1061) | 缓存请求时顺手释放已滑出窗口的 SWA 槽位 | 只有 SWA 组件真正实现（[swa_component.py:635](../../../python/sglang/srt/mem_cache/unified_cache/components/swa_component.py#L635)），基类是 no-op。调用点在 `prepare_for_caching_req` 之后、`insert` 之前（[unified_radix_cache.py:698](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L698)），传的边界是 `effective_cache_len - 1` 而非 `effective_cache_len` |

另有一个**易混**的历史变量：`SGLANG_EXPERIMENTAL_CPP_RADIX_TREE` 走老的 `RadixCacheCpp`，
与统一树无关，而且在 registry 链里**排在统一树之前**（§14.1 第 ③ 步）——两个都开时统一树不生效。

### 16.2 页对齐：三处必须一致

页对齐不是 helper 的内部细节，而是贯穿三处的契约：

```python
radix_key = RadixKey(...).page_aligned(self.page_size)   # ① 截 key，长度成为 page_size 整数倍
values = kv_indices[:page_aligned_len]                   # ② value 按对齐后长度截
assert req.cache_protected_len <= len(new_indices) + self.page_size - 1   # ③ 断言留一页余量
```

对应 [:718](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L718)、
[:720](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L720)、
[:732](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L732)。

三个易错点：

1. **尾部零头不进树**。`effective_cache_len` 算完后被 `page_aligned` 再砍一次；砍掉的尾部 token
   的 KV 槽**不归树管**，由调用方释放（§13.1）。忘了这一步就是槽位泄漏。
2. **`page_size` 有 setter**（[:2012](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L2012)），
   转发到 `tree_core.page_size`，唯一目的是让 `StreamingSession` 临时改写。别处不要动。
3. **eagle bigram**。`is_bigram=self.tree_core.is_eagle` 改变 key 构造语义（相邻 token 成对），
   投机场景下 key 长度与 token 数不是 1:1，页对齐算术必须在 bigram 之后做。

### 16.3 组件之间的隐式依赖：Full 是主，SWA / Mamba 是从

统一树把三种数据放进同一个节点，但它们**不对等**。四条不能违反的从属关系：

| 约束 | 违反后果 | 出处 |
|---|---|---|
| 辅助组件有 device 值 ⇒ Full 必须有 device 值 | SWA 的释放路径要读 `Full.value` 做 full→swa 翻译，读到 `None` 直接崩 | `sanity_check` PART 2（§15） |
| `Full.lock_ref >= aux.lock_ref` | 锁层级倒挂 ⇒ 辅助数据被钉住，但承载它的 Full 路径可被淘汰 | 同上 |
| Full 置墓碑必须**延迟**到级联淘汰之后 | 提前置 `None` ⇒ 同一轮里 SWA 拿不到翻译表 | §11.4 |
| 淘汰优先级：叶节点全 0；内部节点 Full(2) > SWA(1) > Mamba(0) | 写反 ⇒ 级联时先扔掉还要用的 path data | §7.3 |

第 4 条的理由值得重复（代码注释里写得很清楚）：**内部节点上的 SWA 是"路径数据"**
（沿路径都可能被读），**而 Mamba 只在匹配边界处才有意义**，所以内部节点上 Mamba 最先扔。
但在叶节点上三者优先级被**压平为 0**——叶节点淘汰是整节点走，不需要区分先后。

这个"压平"引出级联淘汰里唯一的特例（§7.3）：某组件的真实内部优先级 ≥ 触发者时，
它出现在循环里纯粹是因为叶优先级压平，此时它身上的锁是**合法的 pin**，`continue` 跳过；
否则 `assert lock_ref == 0`。

### 16.4 逻辑时钟必须是 `float64`

`get_and_increase_time_counter()`（[tree_component.py:96](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py#L96)）
返回单调递增的 `float`。match 完成后向上回溯给祖先打时间戳时，每上一层做 `cur_time -= 0.00001`,
保证**父节点时间戳严格小于子节点**。

效果是：LRU/最小堆先淘汰时间戳小的，而路径上父节点时间戳更小、但父节点有子节点所以进不了
叶集合——于是净效果是**同一条路径永远从叶往根逐层淘汰**，不会出现"根被淘汰而子孙还在"。

换成 `int` 计数器做不到（没有 1e-5 这样的"插空"空间），会**静默**破坏路径淘汰顺序。
这是一条没有断言保护的隐式约束。

### 16.5 指针槽分离：数组长度必须是 `_NUM_COMPONENT_TYPES * 2`

`UnifiedLRUList` 是侵入式链表，prev/next 指针直接长在节点上。同一节点可能同时挂在
"SWA-device LRU"和"SWA-host LRU"两条链上，所以槽号是：

```python
_pt = component_type + (_NUM_COMPONENT_TYPES if use_host_ptr else 0)
```

`lru_prev` / `lru_next` 数组长度必须是 `_NUM_COMPONENT_TYPES * 2`（§5.3）。
**新增组件类型时两处都要跟着长**，只改一处会越界或串链。兜底只有 `sanity_check` PART 3 的
"不得同时在 device 与 host 两条 LRU 上"和 `_check_lru_linked_list`。

### 16.6 排空契约（drain contract）

`BaseEvictionResult.__del__` 里埋了一条 tripwire：

```python
assert not self.device_frees and not self.host_frees, \
    "BaseEvictionResult dropped with undrained values"
```

含义：TreeCore 每一步淘汰返回的待释放索引，**控制器必须取走并真正还给内存池**。
漏掉一次 = 显存永久泄漏，而在旧世界里这种泄漏是完全静默的（只表现为"跑久了 OOM"）。

配套纪律是 `_apply_cache_actions` 先 reverse 再逐个 `pop()`（§4.4）——用光的列表不可能被二次应用。

新增一条淘汰路径的 checklist：

```
① 返回值排空（device_frees / host_frees 都要）
② tracker 累加（否则驱动循环不知道已经腾出多少，会多驱逐）
③ LRU 摘链
   → 三件事由 _evict_component_and_detach_lru 一起做（§11.5），优先复用它而不是手写
```

### 16.7 对称性与可重入断言

| 断言 / 契约 | 位置 | 说明 |
|---|---|---|
| `assert self._ongoing_insert_walk_state is None, "concurrent insert walks"` | [unified_tree_core.py:734](../../../python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L734) | insert 是 single-flight。可恢复状态机（§9）只有一份状态，第二个并发 insert 会冲掉第一个的进度 |
| `is_evict_device_ongoing` 必须 start/end 配对 | [tree_component.py:313-337](../../../python/sglang/srt/mem_cache/unified_cache/components/tree_component.py#L313) | `evict_device_start` 断言当前**不在**淘汰中；`evict_device_next_node` / `_end` 断言**在**淘汰中。控制器侧靠 `try/finally` 保证 `evict_device_end` 一定被调（§11.1） |
| `prepare_for_caching_req` / `cleanup_after_caching_req` 必须成对 | §13.3 | 注意 `effective_cache_len <= 0` 的提前返回分支里**也调了** cleanup（[:706-709](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L706)）。新增提前返回时忘补 cleanup 是最易漏的一处 |
| 在途 IO 节点必须 `Full.lock_ref > 0` | `sanity_check` PART 5 | DMA 期间源/目标路径必须钉住，否则淘汰会把正在 DMA 的显存槽还给分配器 |
| 所有 rank 必须进同一次 all-reduce | [:1742-1761](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1742) | `ongoing_write_through` 会跨 rank 分歧（某些 rank 的 backup 返回 0），所以 `writing_check` 里 **不能**用它做 early-return——注释原话："Every rank must enter the all_reduce below"。这是 HiCache 最典型的挂死原因 |

### 16.8 当前代码里的两个问题

**问题 1：`component.tree_core` 被赋值两次。**

```python
# unified_tree_core.py:351-352（TreeCore.__init__ 内）
for component in components.values():
    component.tree_core = self

# unified_radix_cache.py:170-171（create_tree_core 返回之后）
for component in self.components.values():
    component.tree_core = self.tree_core
```

两处写同一个对象，**幂等、无功能影响**，但它是两阶段构造（§14.3）留下的痕迹：
组件先由 `UnifiedRadixCache` 构造（那时还没有 TreeCore），再交给 TreeCore 回填。
外层这一轮是多余的。

清理时要注意：如果将来出现"不经由 `create_tree_core` 注入组件"的 TreeCore 后端
（Rust 实现完全可能这样），外层那一轮反而成了唯一保证——所以不能无脑删，
要么删外层并把"注入 tree_core"写进 `UnifiedTreeCoreInterface` 的契约，要么删内层。

**问题 2：6 处 `getattr` 防御式访问违反仓库规则。**

[`.claude/rules/no-getattr-defensive.md`](../../../.claude/rules/no-getattr-defensive.md) 明确禁止
用 `getattr(obj, "field", default)` 做防御式访问（会把"字段被改名"这类真 bug 吞成默认值）。
控制器里有 6 处：

| 行 | 代码 | 实际情况 |
|---|---|---|
| [:607](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L607) / [:685](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L685) | `priority=getattr(req, "priority", 0) or 0` | `Req` 恒有 `priority` |
| [:654](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L654) / [:744](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L744) | `getattr(req, "swa_uuid_for_lock", None)` | `Req` 恒有该字段（`schedule_batch.py` 里初始化为 `None`） |
| [:655](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L655) | `skip_swa=getattr(req, "swa_prefix_lock_released", False)` | 同上，恒有 |
| [:1231-1232](../../../python/sglang/srt/mem_cache/unified_radix_cache.py#L1231) | `getattr(operation, "pool_transfers", None)` 等 | prefetch operation 的字段 |

前 5 处按规则应直接写 `req.priority` / `req.swa_uuid_for_lock` / `req.swa_prefix_lock_released`。
第 6 处涉及 operation 的多态子类，改之前需确认所有子类都声明了该字段（否则应按规则第 2 条
"总是赋值为 `None` 再做 `None` 检查"处理）。

> 另有一处**已修好、不要再当问题写**的历史记录：
> `attach_hybrid_pool_to_unified_cache` 早期 `except Exception` 只 log 不 re-raise，
> 装配失败被吞掉，错误延迟到 `registry.py` 里表现为
> `AttributeError: 'NoneType' object has no attribute 'layer_done_counter'`（现场极难定位）。
> 当前代码（[hybrid_pool_assembler.py:1372-1374](../../../python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py#L1372)）
> 已经 `logger.exception(...); raise`，问题不存在了。

### 16.9 一页速查

```
读代码顺序（从易到难）：
  sanity_check（§15）     ← 隐式约定的可执行版本，最快建立全局观
  → match_prefix（§8）    ← 最短的一条完整链路
  → evict（§11）          ← 最能看出"统一"到底赚了什么
  → insert 状态机（§9）   ← 最绕，放最后

改代码前自问五问：
  1. 我碰的是机制还是策略？   机制 → TreeCore；策略 → Component
  2. 新增了返回待释放索引的路径？ → 排空契约（§16.6）
  3. 动了节点状态？               → 跑 sanity_check（§15）
  4. 加了组件类型？               → LRU 指针槽数组要 ×2（§16.5）
  5. 在 caching_req 里加了提前返回？ → 补 cleanup（§16.7）

改错的典型症状 → 定位表：
  跑久了 OOM，无报错        → 排空契约漏了（§16.6）
  HiCache 场景多卡挂死      → 某 rank 没进 all-reduce（§16.7 末行）
  命中率异常低              → 墓碑判定 / match 边界（§7.1、§8）
  full→swa 翻译时 None 崩   → Full 墓碑提前了（§16.3 第 3 条）
  LRU 链表断言失败          → 指针槽数组长度（§16.5）
```

