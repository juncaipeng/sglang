# SGLang HiCache 分层 KV Cache —— 使用方法与实现方案

> 关联代码:
> - `python/sglang/srt/mem_cache/hiradix_cache.py`(HiRadixTree 主体)
> - `python/sglang/srt/managers/cache_controller.py`(HiCacheController 调度器)
> - `python/sglang/srt/mem_cache/memory_pool_host.py`(L2 Host 内存池)
> - `python/sglang/srt/mem_cache/hicache_storage.py`(L3 存储抽象接口)
> - `python/sglang/srt/mem_cache/storage/`(各 L3 后端实现)
>
> 关联文档:`docs/advanced_features/hicache_design.md`、`mooncake_standalone_storage_integration.md`

## 目录

1. [HiCache 是什么、解决什么问题](#1-hicache-是什么解决什么问题)
2. [三级缓存总体架构](#2-三级缓存总体架构)
3. [HiRadixTree:元数据组织](#3-hiradixtree元数据组织)
4. [核心数据通路:Local Match / Prefetch / Write-back](#4-核心数据通路local-match--prefetch--write-back)
5. [L1↔L2 数据搬运:IO Backend 与 Mem Layout](#5-l1l2-数据搬运io-backend-与-mem-layout)
6. [L3 存储层:HiCacheStorage 抽象与后端矩阵](#6-l3-存储层hicachestorage-抽象与后端矩阵)
7. [Key 的生成:位置感知前缀哈希](#7-key-的生成位置感知前缀哈希)
8. [完整配置参数参考](#8-完整配置参数参考)
9. [参数归一化与冲突解决](#9-参数归一化与冲突解决)
10. [启动用法与部署示例](#10-启动用法与部署示例)
11. [运行时 attach/detach 存储后端](#11-运行时-attachdetach-存储后端)
12. [与 PD 分离的集成](#12-与-pd-分离的集成)
13. [性能与工程权衡](#13-性能与工程权衡)

---

## 1. HiCache 是什么、解决什么问题

LLM 推理的 prefill 阶段需要把输入序列转换为 KV cache,开销大。当多个请求共享相同前缀(system prompt、多轮对话、多 QA)时,前缀的 KV cache 是完全相同的,可以缓存复用,避免重复计算。

SGLang 原有的 **RadixAttention** 已经用 GPU 空闲显存缓存前缀 KV。**HiCache** 把这一思想扩展到 **Host 内存** 与 **分布式存储**,借鉴 CPU 三级缓存设计:

| 层级 | 介质 | 作用域 | 容量 | 延迟 |
|------|------|--------|------|------|
| **L1** | GPU 显存 | 单实例私有 | 最小 | 最低(直接算) |
| **L2** | Host 内存(CPU) | 单实例私有 | 中等(默认 2× GPU) | 中(需 H2D 拷贝) |
| **L3** | 分布式存储 | **全集群共享** | 最大 | 高(网络/磁盘) |

核心收益:在相同显存预算下大幅扩充 KV cache 容量,提升缓存命中率;L3 还能让 **整个集群的 SGLang 实例共享前缀 KV**。

---

## 2. 三级缓存总体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Scheduler 进程 (per TP rank)                     │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      HiRadixCache (HiRadixTree)                │   │
│  │  · 维护前缀树元数据:每个 TreeNode 记录 KV 落在 L1/L2/L3        │   │
│  │  · match_prefix() / prefetch_from_storage() / write_backup()   │   │
│  └───────────────┬──────────────────────────┬───────────────────┘   │
│                  │                          │                         │
│        ┌─────────▼──────────┐    ┌──────────▼──────────────────┐     │
│        │  HiCacheController  │    │   L2 Host 内存池            │     │
│        │  · write_stream     │◄──►│  MHATokenToKVPoolHost /     │     │
│        │  · load_stream      │    │  MLATokenToKVPoolHost       │     │
│        │  · prefetch_thread  │    │  (pinned host KV buffer)    │     │
│        │  · backup_thread    │    └─────────────────────────────┘     │
│        └──────┬──────────┬───┘                                        │
│   L1↔L2 DMA  │          │ L2↔L3 (zero-copy / batch)                   │
│  ┌───────────▼──┐   ┌───▼──────────────────────────────────────┐     │
│  │ L1 GPU KV池  │   │      L3 StorageBackend (可选)             │     │
│  │ (device pool)│   │  file/mooncake/hf3fs/nixl/eic/simm/aibrix │     │
│  └──────────────┘   └────────────────────────┬─────────────────┘     │
└──────────────────────────────────────────────┼───────────────────────┘
                                                │ (网络/RDMA/磁盘)
                                  ┌─────────────▼──────────────┐
                                  │   分布式存储集群(全集群共享) │
                                  └────────────────────────────┘
```

关键组件职责:

| 组件 | 文件 | 职责 |
|------|------|------|
| `HiRadixCache` | `hiradix_cache.py` | 前缀树元数据组织,触发 prefetch/write-back,跨 rank 同步 |
| `HiCacheController` | `cache_controller.py` | 管理 L1↔L2 的 CUDA stream 异步搬运、L2↔L3 的 prefetch/backup 后台线程 |
| `MHA/MLATokenToKVPoolHost` | `memory_pool_host.py` | L2 host 内存池,管理 pinned KV buffer 与多种 layout |
| `HiCacheStorage` 子类 | `storage/*` | L3 后端,统一 get/set/exists 接口 |
| 选择入口 | `mem_cache/registry.py` | `enable_hierarchical_cache=True` 时构造 `HiRadixCache` |

构建入口在 [registry.py:117-127](../../../python/sglang/srt/mem_cache/registry.py#L117-L127):`enable_hierarchical_cache` 为真时,普通模型构造 `HiRadixCache`,hybrid SSM 模型构造 `HiMambaRadixCache`,并向 `tp_worker` 注册 layer transfer counter(供 overlap 用)。

---

## 3. HiRadixTree:元数据组织

HiRadixTree 在 RadixAttention 的前缀树基础上扩展:**每个 `TreeNode` 对应一段连续 token 的 KV cache,并记录这段 KV 当前落在哪些层级**。

`TreeNode` 关键字段(见 `radix_cache.py` / `hiradix_cache.py`):

| 字段 | 含义 |
|------|------|
| `value` | L1(GPU)上的 token slot 索引;`None` 表示已从 GPU 驱逐 |
| `host_value` | L2(Host)上的 host 索引;非空表示已 backup 到 host |
| `hash_value` | 该 node 每个 page 的 L3 哈希 key 列表(位置感知前缀哈希) |
| `hit_count` | 命中次数,用于 `write_through_selective` 阈值判定 |
| `lock_ref` / `host_ref_counter` | L1 / L2 的引用计数,防止正在使用的节点被驱逐 |
| `backuped` | 是否已 backup 到 host(`host_value` 非空) |
| `evicted` | 是否已从 GPU 驱逐(`value` 为 None) |

**分层元数据策略**:
- 对 **L1/L2**,HiRadixTree 维护精确元数据(具体地址/索引)。
- 对 **L3**,为降低开销 **不持久保存也不持续同步元数据**;访问 L3 时 **实时查询后端**(`batch_exists`)获知某 page 是否存在。

一个 node 在生命周期中会在层级间迁移,典型状态:
- `value!=None, host_value=None`:仅在 GPU(刚 prefill,未 backup)
- `value!=None, host_value!=None`:GPU + Host 都有(write-through 后)
- `value=None, host_value!=None`(`evicted=True`):仅在 Host(已从 GPU 驱逐,可 load_back)
- L3 是否存在:不在树里记录,查询时实时探测

---

## 4. 核心数据通路:Local Match / Prefetch / Write-back

HiCache 工作流由三个关键操作构成。下面结合 Scheduler 的调用时序说明。

### 4.1 Local Match(本地匹配)

请求到来,先在本地 HiRadixTree 匹配 L1+L2 前缀。入口 [hiradix_cache.py:1613 `match_prefix()`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1613):

```python
value, last_node = self._match_prefix_helper(self.root_node, key)  # 遍历前缀树
...
host_hit_length = 0
last_host_node = last_node
while last_node.evicted:                 # 沿父链统计落在 L2 的命中长度
    host_hit_length += len(last_node.host_value)
    last_node = last_node.parent
while not last_host_node.backuped:        # 找到最后一个已 backup 的节点
    last_host_node = last_host_node.parent
return MatchResult(device_indices=value, last_device_node=last_node,
                   last_host_node=last_host_node, host_hit_length=host_hit_length)
```

返回结果区分:`device_indices`(L1 命中,可直接用)、`host_hit_length`(L2 命中,需 load_back 到 GPU)。**纯遍历,无数据拷贝,极快**。

### 4.2 Prefetch from L3(从 L3 预取到 L2)

本地未命中的部分尝试从 L3 预取。Scheduler 在请求入队时触发 [scheduler.py:2381 `_prefetch_kvcache()`](../../../python/sglang/srt/managers/scheduler.py#L2381) → [hiradix_cache.py:1646 `prefetch_from_storage()`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1646)。

触发条件(三者全满足才发起):
```python
if (not self.enable_storage
    or prefetch_length < self.prefetch_threshold      # 默认 256 token
    or self.cache_controller.prefetch_rate_limited()): # 占用超 80% 容量则限流
    return
```

预取流程:
1. 在 L2 host pool `alloc(prefetch_length)`,不足则 `evict_host()` 腾退后重试;
2. 调用 `cache_controller.prefetch()` 投递到 `prefetch_queue`;
3. **后台 `prefetch_thread_func`**([cache_controller.py:1047](../../../python/sglang/srt/managers/cache_controller.py#L1047)):
   - `_storage_hit_query()` 逐 batch 调用 `storage_backend.batch_exists()` 探测 L3 命中页数;
   - `all_reduce(MIN)` 让所有 rank 对命中长度达成一致;
   - 命中不足 threshold → 放入 `prefetch_revoke_queue` 撤销;否则投递到 `prefetch_buffer`;
   - **`prefetch_io_aux_func`** 实际从 L3 读数据填入 host buffer。

**预取终止策略**(`hicache_storage_prefetch_policy`),见 [can_terminate_prefetch():1489](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1489):

| 策略 | 语义 | 适用 |
|------|------|------|
| `best_effort` | 同步点可立即终止,不阻塞 | 极致低延迟 |
| `wait_complete` | 必须取完全部命中页 | 追求高命中率 |
| `timeout`(默认) | 取完 **或** 线性超时则终止 | 延迟与命中率平衡,**生产推荐** |

`timeout` 的超时为线性函数([_prefetch_timeout_check_linear_func:1483](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1483)):
```
timeout = min(max, base + per_ki_token * num_tokens / 1024)
# 默认 base=2.0s, per_ki_token=0.1s, max=30.0s
```

Scheduler 在请求出 waiting queue 前调用 [check_prefetch_progress():1541](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1541) 收尾:终止 prefetch、`all_reduce(MIN)` 取得各 rank 一致的完成长度、把取到的 host page 用 `_insert_helper_host()` 插入 HiRadixTree。

### 4.3 Load Back(L2 → L1)与 Write-back(L1 → L2 → L3)

**Load Back**:prefill 前把 L2 命中的部分搬回 GPU。在 `schedule_policy.py` 的 `add_one_req` 中,若 `host_hit_length>0` 调用 [init_load_back():1367](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1367) → [load_back():1294](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1294)。小于 `load_back_threshold`(10 token)或超出显存配额则跳过("全部加载或不加载")。

**Write-back**:由写策略 `hicache_write_policy` 控制,核心在 [_inc_hit_count():970](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L970) 与 `evict()`:

| 写策略 | `write_through_threshold` | 行为 |
|--------|---------------------------|------|
| `write_through`(默认) | 1 | node 创建后立即 backup 到 host(命中 ≥1 即写) |
| `write_through_selective` | 2 | 命中 ≥2 次才 backup,只备份热数据,降低 I/O |
| `write_back` | — | `_inc_hit_count` 直接返回;仅在 **驱逐时** 才写回([evict():1115](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1115)) |

写回链路:`write_backup()`(L1→L2 DMA)→ `writing_check()` 确认 DMA 完成并发 store(CPU) 事件 → 若启用 L3,`write_backup_storage()` 投递 backup 到 L3。**跨实例共享**:写到 L3 时只传 L3 中尚不存在的页。

```
新请求 ──► Local Match (L1+L2)
            │命中               │未命中部分
            ▼                   ▼
        直接复用            Prefetch from L3 ──► 填入 L2 host buffer
            │                   │
            └────────┬──────────┘
                     ▼
            Load Back (L2→L1, init_load_back)
                     ▼
                Prefill 计算  ──► 新生成 KV
                     ▼
        Write-back: L1 ──(write_policy)──► L2 ──► L3 (仅缺失页)
```

---

## 5. L1↔L2 数据搬运:IO Backend 与 Mem Layout

L1↔L2 的搬运是性能关键。由两个正交维度决定:**IO Backend**(怎么搬)与 **Mem Layout**(host 怎么排布)。

### 5.1 IO Backend(`--hicache-io-backend`)

| 取值 | 实现 | 说明 |
|------|------|------|
| `kernel`(默认) | GPU-assisted I/O kernel | 自定义融合 CUDA gather/scatter kernel,比 `cudaMemcpyAsync` 最高快 ~3×;直接按 token index 读写 device 池,故对 device 布局有隐含假设;仅与 `page_first_direct` 不兼容(会被强制切 `direct`) ；支持layer_first和page_first常见layout |
| `direct` | 普通 per-page direct I/O | host 侧按 `(page, layer)` 做整块连续拷贝;配 `page_first_direct` layout |
| `kernel_ascend` | Ascend NPU kernel | 昇腾 NPU 变体 |

搬运方向控制见 [cache_controller.py:760 `move_indices()`](../../../python/sglang/srt/managers/cache_controller.py#L760):`kernel` 把 host 索引搬到 GPU 上做 gather;`direct` 在 host 侧排序/索引。

> **关于 `kernel` 与 FA3 decode backend**:`kernel` backend **不是** layout 无关的纯 memcpy,而是自定义融合 CUDA gather/scatter kernel([`transfer_kv_per_layer`](../../../sgl-kernel/python/sgl_kernel/kvcacheio.py#L13)),直接按 token index + `item_size` 步长寻址 **device 侧 KV 池的原始 buffer**,因此对 device 池的物理排布有隐含假设。历史上(PR #19669 之前)这套假设与 FA3 decode 的分页布局对不上(尤其小 `page_size`),`server_args` 会强制回退:显式要 FA3 decode ⇒ 降级 `hicache_io_backend=direct`;未显式 ⇒ 改用 flashinfer/triton decode 后端。该不兼容已由 **PR #21631(重构 HiCache write-back kernel,新增 staged write-back + relayout)** 修复并**移除全部强制切换逻辑**。当前 `main` 上 `kernel + FA3 decode` 完全兼容,单测 [test_server_args.py:947 `test_hicache_kernel_keeps_implicit_fa3_decode_backend`](../../../test/registered/unit/server_args/test_server_args.py#L947) 锁定 kernel/page_first/decode 后端均**原样保留**。

### 5.2 Mem Layout(`--hicache-mem-layout`)

GPU 计算天然按层(`layer_first`),但 L3 传输希望按页连续(`page_first`)以支持 zero-copy。**注意默认值是 `page_first`(不是 `layer_first`)**([server_args.py:2230](../../../python/sglang/srt/server_args.py#L2230)),配默认 io backend `kernel`。

每种 layout 的 **host 池 buffer shape**(记 `size = page_num × page_size`)直接定义在 `init_kv_buffer` 里:

**MHA**([mha.py:130](../../../python/sglang/srt/mem_cache/pool_host/mha.py#L130),首维 `2` = K/V):

| layout | shape | zero-copy | 页内同层 token 连续 | 备注 |
|--------|-------|-----------|--------------------|------|
| `layer_first` | `(2, layer_num, size, head_num, head_dim)` | 否 | — | 层主序,与 GPU kernel 对齐;L2→GPU 需逐 token-per-layer |
| `page_first`(**默认**) | `(2, size, layer_num, head_num, head_dim)` | 是 | 否(跨 layer strided) | 同页所有 KV 连续,可作单对象传给 L3;配 `kernel`(默认 io backend,即默认组合) |
| `page_first_direct` | `(2, page_num, layer_num, page_size, head_num, head_dim)` | 是 | **是** | `page_size` 挪到 layer 之后;配 `direct`,zero-copy 性能等同 page_first |
| `page_head` | `(2, page_num, head_num, page_size, layer_num, head_dim)` | 是 | 是(但按 head 分片) | MHA 异构 TP 复用(配合 `tp_lcm_size`) |

如果只使用L2 Cache、不使用L3 Cache，推荐设置layout为`--hicache-mem-layout layer_first --hicache-io-backend kernel`，可以避免layout转换的耗时。

**MLA**([mla.py:119](../../../python/sglang/srt/mem_cache/pool_host/mla.py#L119),`head_num=1`,`kv_cache_dim = kv_lora_rank + qk_rope_head_dim`):

| layout | shape |
|--------|-------|
| `layer_first` | `(layer_num, size, 1, kv_cache_dim)` |
| `page_first` | `(size, layer_num, 1, kv_cache_dim)` |
| `page_first_direct` | `(page_num, layer_num, page_size, 1, kv_cache_dim)` |
| `page_first_kv_split` | K/V 分离两块 `(page_num, layer_num, page_size, 1, {kv_lora_rank / qk_rope_head_dim})`,Ascend 专用 |

#### 为什么需要 `page_first_direct`

`page_first` 与 `page_first_direct` **都是页主序、都能 L3 zero-copy**(一页所有 KV 物理连续)。唯一区别是 `page_size` 维的位置:

```
page_first:         [ 2, size,               layer, head, head_dim ]
                          └── token 在外，layer 在内 ──┘
page_first_direct:  [ 2, page_num, layer,  page_size,  head, head_dim ]
                                          └─ token 落到 layer 之后 ─┘
```

对**固定的一层** `layer_id`,取该页 `page_size` 个 token 时:
- `page_first`:这些 token 以 `layer_num × head_num × head_dim` 为步长 **strided**,不连续 ⇒ 只能靠 GPU gather kernel 搬运(即 `kernel` backend)。
- `page_first_direct`:`(page, layer)` 固定后 `page_size` 个 token **物理连续** ⇒ `direct` backend 可直接整块连续拷贝([mha.py:294 `transfer_kv_per_layer_direct_pf_lf`](../../../python/sglang/srt/mem_cache/pool_host/mha.py#L294)),无需 gather。

因此 **`page_first_direct` 存在的意义 = 保留 page_first 的 zero-copy 收益,同时改成 `direct` backend 能吃的连续排布**。两者 zero-copy 到 L3 的能力完全一样,只是 L1↔L2 搬运换了实现路径,性能等价。

启动时 layout↔io_backend 互斥由 [server_args.py:6284 `_resolve_layout_io_compatibility`](../../../python/sglang/srt/server_args.py#L6284) 强制对齐:

```python
# page_first_direct + kernel  → 强制切 direct
# page_first      + direct    → 强制切 page_first_direct
```

```
                    ┌─────────────── 想要 L3 zero-copy（页连续）───────────────┐
                    │                                                          │
              page_first                                            page_first_direct
                    │                                                          │
          配 kernel IO backend                                    配 direct IO backend
       (GPU gather kernel 处理 strided                        (host 侧按 (page,layer) 做
        per-layer 访问，最高 ~3×)                              整块 contiguous 拷贝，无需 gather)
                    │                                                          │
       kernel + page_first_direct 组合                          page_first + direct 组合
       被强制切 direct                                          被强制切 page_first_direct
```

### 5.3 关键优化

- **Compute-Transfer Overlap**:prefill 时边算 layer N 边加载 layer N+1 的 KV,通过 `LayerDoneCounter` / `LayerLoadingEvent`([cache_controller.py:70-135](../../../python/sglang/srt/managers/cache_controller.py#L70-L135))隐藏传输延迟。
- **MLA 写回优化**:MLA 模型各 rank 持有相同的完整 KV,因此 **仅 rank 0 写 L3**(`backup_skip = is_mla_model and tp_rank != 0`,[cache_controller.py:465](../../../python/sglang/srt/managers/cache_controller.py#L465)),避免冗余存储。
- **L2 池大小**:`HostKVCache.__init__`([pool_host/base.py:83-106](../../../python/sglang/srt/mem_cache/pool_host/base.py#L83-L106))。`hicache_size>0` 时按 GB 计算,否则按 `device_pool.size * hicache_ratio`;且断言 host 池必须 **大于** device 池;预留 10GB 给 OS。

---

## 6. L3 存储层:HiCacheStorage 抽象与后端矩阵

### 6.1 抽象接口 `HiCacheStorage(ABC)`

定义于 [hicache_storage.py:141](../../../python/sglang/srt/mem_cache/hicache_storage.py#L141)。**5 个抽象方法**(每个后端必须实现):

```python
@abstractmethod
def get(self, key, target_location=None, target_sizes=None) -> torch.Tensor | None: ...
@abstractmethod
def batch_get(self, keys, target_locations=None, target_sizes=None) -> List[...] | int: ...   # TODO: deprecate
@abstractmethod
def set(self, key, value=None, target_location=None, target_sizes=None) -> bool: ...
@abstractmethod
def batch_set(self, keys, values=None, target_locations=None, target_sizes=None) -> bool: ...  # TODO: deprecate
@abstractmethod
def exists(self, key) -> bool: ...
```

**运行时实际走的高性能路径**(非抽象,后端可覆写):

| 方法 | 用途 |
|------|------|
| `batch_exists(keys, extra_info)` → int | 返回从头连续存在的 key 数(最长前缀语义) |
| `batch_get_v1(keys, host_indices, extra_info)` → List[bool] | **zero-copy 读主路径**(hf3fs/mooncake/eic/nixl/simm) |
| `batch_set_v1(keys, host_indices, extra_info)` → List[bool] | **zero-copy 写主路径** |
| `batch_exists_v2/get_v2/set_v2(transfers, ...)` | 混合多池(KV+Mamba/SWA/Indexer)模型路径 |
| `register_mem_pool_host(mem_pool_host)` | 注册 host KV 池(zero-copy 缓冲注册) |

`HiCacheController.attach_storage_backend` 在 [cache_controller.py:495-503](../../../python/sglang/srt/managers/cache_controller.py#L495-L503) 据后端类型选择 v1 zero-copy 还是 generic 路径:
```python
if self.storage_backend_type in ["hf3fs", "mooncake", "eic", "nixl", "simm"] or \
   (dynamic 且 interface_v1=1):
    self.page_get_func = self._page_get_zero_copy   # batch_get_v1
    self.page_set_func = self._page_set_zero_copy   # batch_set_v1
```

### 6.2 后端工厂

`StorageBackendFactory`([storage/backend_factory.py](../../../python/sglang/srt/mem_cache/storage/backend_factory.py))用 **惰性加载** 注册表:只在选中某后端时才 import 对应模块(避免引入 mooncake/nixl/eic 等可选重依赖)。`create_backend()` 据 `--hicache-storage-backend` 分派,并对每个后端做差异化构造(如 hf3fs 用 `from_env_config(bytes_per_page, dtype, ...)`)。

### 6.3 后端能力矩阵

| 后端 (`--hicache-storage-backend`) | 类 | 底层技术 | zero-copy | 关键配置 |
|---|---|---|---|---|
| **file** | `HiCacheFile` | 本地文件系统(每页一个 `.bin`) | 否 | `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR`(默认 `/tmp/hicache`) |
| **mooncake** | `MooncakeStore` | Mooncake 分布式存储(RDMA/TCP,可选 SSD offload) | 是 | `master_server_address`/`protocol`/`device_name`/`global_segment_size`,env `SGLANG_HICACHE_MOONCAKE_CONFIG_PATH` |
| **hf3fs** | `HiCacheHF3FS` | DeepSeek 3FS 分布式文件系统(USRBIO) | 是(page_first) | `SGLANG_HICACHE_HF3FS_CONFIG_PATH`:`file_path_prefix`/`numjobs`/`entries`/`metadata_server_url`(MLA 需全局 metadata server) |
| **nixl** | `HiCacheNixl` | NVIDIA NIXL(3FS/POSIX/GDS/S3 插件) | 是(page_first) | `extra_config` → `NixlBackendConfig`;env `SGLANG_HICACHE_NIXL_BACKEND_PLUGIN` 等 |
| **eic** | `EICStorage` | EIC 远程 KV 存储(RDMA/GPUDirect,NIC 亲和) | 是(page_first) | YAML `REMOTE_EIC_YAML`(默认 `/sgl-workspace/config/remote-eic.yaml`) |
| **simm** | `HiCacheSiMM` | SiMM 共享内存/RDMA KV 存储(NUMA 感知) | 是 | `SGLANG_HICACHE_SIMM_CONFIG_PATH`:`manager_address`(必填) |
| **aibrix** | `AibrixKVCacheStorage` | AIBrix KVCache(块式 offloading) | 否(块拷贝) | `KVCacheConfig`;**MLA 抛 NotImplementedError** |
| **dynamic** | 自定义 | 用户自定义 | 取决于 | `extra_config` 指定 `backend_name`/`module_path`/`class_name`/`interface_v1` |

> **重要**:`LMCache` **不是** L3 存储后端,不在工厂注册表中。它是 `RadixCache` 的子类 `LMCRadixCache`,通过 **独立的 `--enable-lmcache`** 选择([registry.py:131](../../../python/sglang/srt/mem_cache/registry.py#L131)),在前缀树层面集成 LMCache,是 HiCache 的 **替代方案** 而非其 L3 后端。

### 6.4 MHA vs MLA 的 page 对象数

- **MHA**:每 rank 持 `1/tp_size` 的 KV;每页存 **2 个对象**(`_k`/`_v`),故 `batch_exists` 计数要 ÷2。
- **MLA**:各 rank 持完整相同 KV;每页 **1 个对象**(`_k`),且只 rank 0 写。

---

## 7. Key 的生成:位置感知前缀哈希

L3 的 key 是 **SHA-256 十六进制串**,由 [utils.py:106 `get_hash_str()`](../../../python/sglang/srt/mem_cache/utils.py#L106) 生成,**按页、前缀链式** 计算:

```python
def get_hash_str(token_ids, prior_hash=None):
    hasher = hashlib.sha256()
    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))   # 链入父页哈希
    for t in token_ids:
        hasher.update(t.to_bytes(4, "little", signed=False))
    return hasher.hexdigest()
```

特点:
- 某页的 key **依赖整个前缀**(链入 `prior_hash`),天然实现 **最长前缀匹配** 语义。
- `compute_node_hash_values()`([utils.py:126](../../../python/sglang/srt/mem_cache/utils.py#L126))为一个 node 的每个 page 生成哈希列表。
- 各后端在 base hash 上再加 **命名空间后缀** 区分 model/tp_rank/layout/K-V 分量(如 mooncake MHA 加 `_{rank}_k`/`_{rank}_v`)。
- **异构 TP 复用**:`tp_lcm_size` 设为所有参与共享集群 TP 的最小公倍数,使 key 跨不同 TP 部署可复用。

---

## 8. 完整配置参数参考

字段定义见 [server_args.py:2195-2260](../../../python/sglang/srt/server_args.py#L2195-L2260)(`A[...]` 注解式声明,argparse 由注解自动生成,无独立手写 add_argument 段)。

| CLI 参数 | 字段 | 类型 | 默认 | 可选值 | 说明 |
|---|---|---|---|---|---|
| `--enable-hierarchical-cache` | `enable_hierarchical_cache` | bool | `False` | — | **开启 HiCache 的总开关** |
| `--hicache-ratio` | `hicache_ratio` | float | `2.0` | >1 | L2 host 池 / L1 device 池 大小比 |
| `--hicache-size` | `hicache_size` | int | `0` | GB | L2 池绝对大小(GB,每 rank);非 0 时覆盖 ratio |
| `--hicache-write-policy` | `hicache_write_policy` | str | `write_through` | `write_back`/`write_through`/`write_through_selective` | L1→L2/L3 写策略 |
| `--hicache-io-backend` | `hicache_io_backend` | str | `kernel` | `direct`/`kernel`/`kernel_ascend` | CPU↔GPU 搬运后端 |
| `--hicache-mem-layout` | `hicache_mem_layout` | str | `page_first` | `layer_first`/`page_first`/`page_first_direct`/`page_first_kv_split`/`page_head` | L2 host 池排布 |
| `--hicache-storage-backend` | `hicache_storage_backend` | str? | `None` | `file`/`mooncake`/`hf3fs`/`nixl`/`aibrix`/`dynamic`/`eic`/`simm` | L3 后端(不设则无 L3) |
| `--hicache-storage-prefetch-policy` | `hicache_storage_prefetch_policy` | str | `timeout` | `best_effort`/`wait_complete`/`timeout` | L3 预取终止策略 |
| `--hicache-storage-backend-extra-config` | `hicache_storage_backend_extra_config` | str? | `None` | JSON 串 或 `@文件` | 后端额外配置(JSON/YAML/TOML) |
| `--page-size` | `page_size` | int? | `None`→`1`(MUSA `64`) | — | 每页 token 数,L3 存取粒度 |
| `--enable-lmcache` | `enable_lmcache` | bool | `False` | — | 用 LMCache 作为替代方案(非 L3 后端) |

### `--hicache-storage-backend-extra-config` 可识别键

解析见 [hiradix_cache.py:681 `_parse_storage_backend_extra_config`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L681)。可传 JSON 串或 `@path`(`.json`/`.toml`/`.yaml`):

| 键 | 默认 | 含义 |
|----|------|------|
| `prefetch_threshold` | `256` | 触发预取的最小命中 token 数 |
| `prefetch_timeout_base` | `2.0` | timeout 策略基础超时(秒) |
| `prefetch_timeout_per_ki_token` | `0.1` | 每 1024 token 增量超时(秒) |
| `prefetch_timeout_max` | `30.0` | 超时上限(秒) |
| `hicache_storage_pass_prefix_keys` | `False` | 是否向后端透传前缀 key |
| `tp_lcm_size` | — | 异构 TP 复用的 TP 最小公倍数 |
| `backend_name`/`module_path`/`class_name`/`interface_v1` | — | `dynamic` 后端专用 |

---

## 9. 参数归一化与冲突解决

`ServerArgs.__post_init__`([server_args.py:2883](../../../python/sglang/srt/server_args.py#L2883))调用 [`_handle_hicache()`](../../../python/sglang/srt/server_args.py#L6264)(调用点 [server_args.py:3010](../../../python/sglang/srt/server_args.py#L3010),定义 [:6264](../../../python/sglang/srt/server_args.py#L6264))在解析后自动修正冲突(仅当 `enable_hierarchical_cache` 或 `disaggregation_decode_enable_offload_kvcache` 为真时生效):

| 冲突 | 自动修正 |
|------|----------|
| `page_first_direct` + `kernel` | io → `direct`([_resolve_layout_io_compatibility:6284](../../../python/sglang/srt/server_args.py#L6284)) |
| `page_first` + `direct` | layout → `page_first_direct` |
| `mooncake` + `layer_first` | layout → `page_first_direct`(direct) 或 `page_first`(kernel)([_resolve_storage_layout_compatibility:6303](../../../python/sglang/srt/server_args.py#L6303)) |

> **已废弃的历史规则**:早期版本(PR #19669 前)还有一条「`kernel` + 有效 FA3 decode backend ⇒ io → `direct` / 换 decode 后端」。该规则已随 **PR #21631** 修复 kernel 对 FA3 的兼容性后**整体删除**;当前 `main` 上 `kernel + FA3 decode` 不再触发任何自动修正(见 5.1 节)。

其他校验:
- `enable_hierarchical_cache` 与 `disable_radix_cache` **互斥**。
- `disaggregation_decode_enable_offload_kvcache` 要求 **必须设置** `hicache_storage_backend`。
- LMCache:`page_size % block_size != 0` 时禁用 `enable_lmcache`。

---

## 10. 启动用法与部署示例

### 10.1 最小可用(仅 L1+L2,无 L3)

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --tp 2 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-io-backend kernel \
  --hicache-write-policy write_through
```

### 10.2 启用 L3(HF3FS,DeepSeek-R1)

```bash
python3 -m sglang.launch_server \
  --model-path /xxx/DeepSeek-R1/ \
  --tp 8 \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --enable-metrics --enable-cache-report \
  --enable-hierarchical-cache \
  --hicache-ratio 2 --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend hf3fs \
  --hicache-storage-prefetch-policy wait_complete
```

### 10.3 启用 L3(Mooncake,Qwen3-235B)

```bash
export MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE=816043786240
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="$DEVICE_LIST"
export MOONCAKE_MASTER=127.0.0.1:50051

python3 -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --tp 8 --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout
```

### 10.4 自定义动态后端

```bash
python3 -m sglang.launch_server \
  --model-path your-model \
  --enable-hierarchical-cache \
  --hicache-storage-backend dynamic \
  --hicache-storage-backend-extra-config \
    '{"backend_name":"my_backend","module_path":"my.module","class_name":"MyHiCache","interface_v1":1}'
```

自定义后端只需实现 `get`/`set`/`exists`(及可选的 v1 zero-copy 方法),并在工厂注册或用 `dynamic` 动态加载。

---

## 11. 运行时 attach/detach 存储后端

无需重启即可在运行中挂载/卸载 L3 后端,通过 HTTP Admin API。**严格要求服务空闲**(无 running、无 waiting 请求),否则返回 HTTP 400 且不改状态。详见 `docs/advanced_features/hicache_storage_runtime_attach_detach.md`。

控制路径:`HTTP Server` → `TokenizerManager`(FanOut 到所有 DP scheduler)→ `Scheduler`(`is_fully_idle()` 检查)→ `HiRadixCache.attach_storage_backend()` → `HiCacheController.attach_storage_backend()`(经工厂创建后端、起 prefetch/backup 线程)。

```bash
# 查询当前后端
curl -s http://127.0.0.1:30000/hicache/storage-backend

# 挂载 mooncake
curl -s -X PUT http://127.0.0.1:30000/hicache/storage-backend \
  -H 'Content-Type: application/json' \
  -d '{"hicache_storage_backend":"mooncake",
       "hicache_storage_backend_extra_config_json":"{\"master_server_address\":\"127.0.0.1:50051\",\"protocol\":\"tcp\"}",
       "hicache_storage_prefetch_policy":"timeout"}'

# 卸载(仅停用,不删远端数据)
curl -s -X DELETE http://127.0.0.1:30000/hicache/storage-backend
```

DP 语义:请求 fan-out 到所有 DP rank,`success` 仅当 **全部** rank 成功;当前无跨 rank 自动回滚,建议各 rank 配置一致,失败后先 detach 再修正重试。

---

## 12. 与 PD 分离的集成

HiCache 可与 PD(Prefill-Decode)分离同时使用,两种典型配置:

1. **Prefill-only HiCache**:仅 Prefill 节点开 HiCache,实现 Prefill 实例间前缀共享(适合 system prompt 场景)。
2. **Full HiCache + 异步 offload**:Prefill 开 HiCache,Decode 开 `--disaggregation-decode-enable-offload-kvcache`,让 Prefill 复用 Decode 产出的 KV(适合多轮对话)。

Decode 侧异步 offload 由 [`DecodeKVCacheOffloadManager`](../../../python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py) 管理,它独立构造一个 host 池并复用 `HiCacheController` 做 GPU→Host→Storage 的三级 offload。**注意**:开启该选项 **必须** 同时设置 `--hicache-storage-backend`。

```bash
# Prefill 节点
python3 -m sglang.launch_server --model-path /xxx/DeepSeek-R1/ --tp 8 \
  --page-size 64 --enable-hierarchical-cache --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct --hicache-io-backend direct \
  --hicache-storage-backend hf3fs --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0

# Decode 节点(异步 offload)
python3 -m sglang.launch_server --model-path /xxx/DeepSeek-R1/ --tp 8 \
  --page-size 64 --hicache-ratio 2 \
  --hicache-mem-layout page_first_direct --hicache-io-backend direct \
  --hicache-storage-backend hf3fs --hicache-storage-prefetch-policy wait_complete \
  --disaggregation-decode-enable-offload-kvcache \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0
```

---

## 13. 性能与工程权衡

| 维度 | 权衡 |
|------|------|
| **`hicache_ratio`/`hicache_size`** | 越大命中率越高,但收益非线性;热 token 缓存后继续增大收益递减;受 host 内存限制(预留 10GB) |
| **`page_size`** | 大页降低元数据与 I/O 开销,但部分匹配时命中率下降。长公共前缀场景用大页;前缀多样用小页 |
| **写策略** | `write_through` 缓存效果最强但 I/O 最高;`selective` 只备份热数据;`write_back` I/O 最低,适合存储容量受限 |
| **预取策略** | `best_effort` 延迟最低;`wait_complete` 命中率最高;`timeout` 折中,生产推荐 |
| **layout/io** | `page_first*` + zero-copy 显著降 L2↔L3 拷贝;`kernel` io 比 `direct` 快,但与 `page_first_direct` 不兼容(会被强制切 `direct`)。**注:`kernel` 与 FA3 decode 已兼容**(PR #21631 后) |
| **MLA 优化** | 仅 rank 0 写 L3,避免 TP 间冗余 |
| **多 rank 同步** | prefetch 命中长度、完成长度都用 `all_reduce(MIN)` 保证各 rank 一致,避免状态分叉 |

### 调优建议速查

- **延迟敏感**:`--hicache-storage-prefetch-policy best_effort`
- **命中率优先**:`wait_complete` + 较大 `hicache_size`
- **生产均衡**:`timeout`(调 `prefetch_timeout_base`/`per_ki_token`)
- **FA3 模型**:`kernel`(默认)已兼容 FA3 decode,无需特意切换;若走 `direct` 路径则用 `--hicache-io-backend direct --hicache-mem-layout page_first_direct`(系统会把 `page_first`+`direct` 自动归一化为 `page_first_direct`)
- **OOM**:降低 `--mem-fraction-static`,或减小 `--hicache-size`

---

## 附:关键代码索引

| 功能 | 位置 |
|------|------|
| HiRadixCache 构造 | [hiradix_cache.py:75](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L75) |
| match_prefix | [hiradix_cache.py:1613](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1613) |
| prefetch_from_storage | [hiradix_cache.py:1646](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1646) |
| check_prefetch_progress | [hiradix_cache.py:1541](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1541) |
| write_backup / _inc_hit_count | [hiradix_cache.py:833](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L833) / [970](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L970) |
| load_back / init_load_back | [hiradix_cache.py:1294](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1294) / [1367](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1367) |
| evict(含 write_back) | [hiradix_cache.py:1115](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1115) |
| HiCacheController.write/load | [cache_controller.py:673](../../../python/sglang/srt/managers/cache_controller.py#L673) / [743](../../../python/sglang/srt/managers/cache_controller.py#L743) |
| attach_storage_backend | [cache_controller.py:431](../../../python/sglang/srt/managers/cache_controller.py#L431) |
| prefetch_thread_func | [cache_controller.py:1047](../../../python/sglang/srt/managers/cache_controller.py#L1047) |
| HostKVCache 大小计算 | [pool_host/base.py:83](../../../python/sglang/srt/mem_cache/pool_host/base.py#L83) |
| HiCacheStorage 抽象 | [hicache_storage.py:141](../../../python/sglang/srt/mem_cache/hicache_storage.py#L141) |
| StorageBackendFactory | [storage/backend_factory.py:16](../../../python/sglang/srt/mem_cache/storage/backend_factory.py#L16) |
| get_hash_str | [utils.py:106](../../../python/sglang/srt/mem_cache/utils.py#L106) |
| 参数归一化 _handle_hicache | [server_args.py:6264](../../../python/sglang/srt/server_args.py#L6264) |
