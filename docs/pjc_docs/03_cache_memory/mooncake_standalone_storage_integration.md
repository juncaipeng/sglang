# SGLang 接入 Mooncake Store —— Standalone Storage(Dummy Client)模式方案

> 适用版本:`mooncake >= 0.3.8.post1`
> 关联代码:`python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`
> 关联文档:`mooncake_integration_architecture.md`(Mooncake 总体集成)

## 目录

1. [两种接入模式的本质区别](#1-两种接入模式的本质区别)
2. [组件拓扑与启动顺序](#2-组件拓扑与启动顺序)
3. [代码调用链路](#3-代码调用链路)
4. [为什么 Standalone 必须用 MooncakeHostTensorAllocator](#4-为什么-standalone-必须用-mooncakehosttensorallocator核心原理)
5. [数据读写路径(zero-copy)](#5-数据读写路径zero-copy)
6. [必须满足的约束](#6-必须满足的约束)
7. [性能与工程权衡](#7-性能与工程权衡)
8. [配置参考](#8-配置参考)

---

## 1. 两种接入模式的本质区别

Mooncake Store 接入 SGLang 有两种部署形态,核心差异在于 **谁持有 Transfer Engine / RDMA / 内存段**:

| 维度 | Self-hosted 模式(默认) | **Standalone Storage 模式(Dummy Client)** |
|---|---|---|
| `standalone_storage` | `False` | **`True`** |
| SGLang 进程角色 | 完整 Mooncake 节点 | **Dummy Client(瘦客户端)** |
| Transfer Engine / RDMA | 跑在 SGLang 进程内 | **跑在独立的 `mooncake_client` 进程(Real Client)** |
| 内存段 (global segment) | SGLang 进程贡献并管理 | **由 Real Client 进程持有** |
| 初始化调用 | `store.setup(...)` | **`store.setup_dummy(...)`** |
| 连接参数 | `master_server_address` | **`client_server_address`(默认 `:50052`)** |
| Host KV buffer 分配器 | 任意(默认 torch pinned) | **强制 `MooncakeHostTensorAllocator`** |
| 进程崩溃后缓存 | 随进程丢失 | **Real Client 存活则缓存保留** |

> 关键定位:Standalone 模式把"重资产"(RDMA 连接、内存池、与 master 的通信)从 SGLang 推理进程剥离到一个本地常驻的 `mooncake_client` 进程,SGLang 仅通过本地 RPC/IPC 委托 I/O。**解耦带来稳定性,且缓存可跨 SGLang 重启存活。**

---

## 2. 组件拓扑与启动顺序

```
┌──────────────────────────────────────────────────────────────┐
│                        单机节点                                 │
│                                                                │
│  ┌────────────────────────┐      RPC/IPC (50052)              │
│  │   SGLang Server         │◄──────────────────────┐          │
│  │   (Dummy Client)        │                        │          │
│  │                         │   ┌────────────────────▼───────┐  │
│  │  HiRadixCache           │   │  mooncake_client           │  │
│  │   └ MHATokenToKVPoolHost│   │  (Real Client / Store Svc) │  │
│  │       allocator=        │   │                            │  │
│  │       Mooncake分配器 ────┼──►│  · Transfer Engine (RDMA)  │  │
│  │       (共享内存KV buffer)│零拷贝│  · global_segment 内存池   │  │
│  │                         │DMA │  · 与 Master 通信           │  │
│  │  MooncakeStore          │   └──────────┬─────────────────┘  │
│  │   store.setup_dummy()   │              │ RDMA               │
│  └─────────────────────────┘              │                    │
└────────────────────────────────────────────┼────────────────────┘
                                             │
                      ┌──────────────────────▼──────────┐
                      │  mooncake_master (:50051)        │
                      │  + metadata service (可内嵌)      │
                      │  集群拓扑/对象元数据/逐出管理       │
                      └──────────────────────────────────┘
                                  ▲
                                  │ (分布式部署时其他节点的 Real Client 也接入)
```

启动顺序(单机),三个进程必须按序拉起:

```bash
# 1. Master(可内嵌 metadata service)
mooncake_master --eviction_high_watermark_ratio=0.95

# 2. Real Client(持有内存段 + RDMA),监听 50052 供 Dummy Client 连接
mooncake_client --global_segment_size=4GB

# 3. SGLang(Dummy Client)
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --hicache-mem-layout page_first \
    --model-path [model_path] \
    --hicache-storage-backend-extra-config \
      '{"standalone_storage": true, "client_server_address": "127.0.0.1:50052"}'
```

---

## 3. 代码调用链路

整个链路串联三个文件:`hiradix_cache.py` → `cache_controller.py` → `mooncake_store.py`。

```
ServerArgs(--hicache-storage-backend mooncake)
   │
   ▼
HiRadixCache.__init__                         [hiradix_cache.py:80-100]
   │  allocator_type = "mooncake"  ★关键传递★
   ▼
MHATokenToKVPoolHost(allocator_type="mooncake")
   │
   ▼
get_allocator_from_storage("mooncake")        [memory_pool_host.py:158]
   └─► MooncakeHostTensorAllocator()          [mooncake_store.py:32]
         内部用 mooncake.store.MooncakeHostMemAllocator
         → init_kv_buffer() 用它分配 host KV buffer(共享内存)
   │
   ▼
HiCacheController.attach_storage_backend      [cache_controller.py:455]
   │  _generate_storage_config() → HiCacheStorageConfig
   ▼
StorageBackendFactory.create_backend("mooncake", cfg, mem_pool_host)
   └─► MooncakeStore(storage_config, mem_pool_host)   [backend_factory.py:165]
   │
   ▼
MooncakeStore.__init__                        [mooncake_store.py:297]
   if config.standalone_storage:              [mooncake_store.py:347]
       assert isinstance(mem_pool.allocator, MooncakeHostTensorAllocator)  ★校验★
       store.setup_dummy(                      [mooncake_store.py:354]
           mem_pool.size * mem_pool.size_per_token,   # 整个 host KV 池字节数
           DEFAULT_LOCAL_BUFFER_SIZE,                 # 16MB,零拷贝其实不用
           config.client_server_address)              # 连本地 Real Client
   │
   ▼
register_mem_pool_host(mem_pool_host)         [mooncake_store.py:551]
   └─► store.register_buffer(kv_buffer)       把 host KV buffer 注册给 Mooncake
```

配置加载优先级 `_load_config()` [mooncake_store.py:262]:

```
extra_config (--hicache-storage-backend-extra-config)   ← 最高
   └─ 含 master_server_address 或 client_server_address
SGLANG_HICACHE_MOONCAKE_CONFIG_PATH (JSON 文件)
环境变量 (MOONCAKE_STANDALONE_STORAGE=1, MOONCAKE_CLIENT=...)  ← 兜底
```

---

## 4. 为什么 Standalone 必须用 MooncakeHostTensorAllocator(核心原理)

这是 standalone 模式最关键的工程点,`mooncake_store.py:347-358` 强制校验:

```python
if self.config.standalone_storage:
    if not isinstance(mem_pool.allocator, MooncakeHostTensorAllocator):
        raise RuntimeError("... requires MooncakeHostTensorAllocator ...")
```

**原因:零拷贝跨进程 DMA 的内存可见性**

- **Self-hosted 模式**:Transfer Engine 跟 KV buffer 在同一进程,默认 torch pinned memory 就能被本进程的 RDMA 直接注册访问。
- **Standalone 模式**:真正执行 RDMA 读写的是**另一个进程**(Real Client)。如果 host KV buffer 用普通 `torch.empty(pin_memory=True)` 分配,它是 SGLang 进程私有的虚拟地址,Real Client **无法 DMA 进去**。

因此 `MooncakeHostTensorAllocator.allocate()` `mooncake_store.py:40-64` 改用 `MooncakeHostMemAllocator` 从一段**跨进程可共享/可被 Real Client 注册**的内存里分配,再用 `torch.frombuffer` 包装成 tensor:

```python
ptr_int = self.allocator.alloc(size)        # Mooncake 管理的共享内存
c_array = (ctypes.c_byte * size).from_address(ptr_int)
tensor = torch.frombuffer(c_array, dtype=torch.uint8, count=size).view(dtype).view(dims)
```

这样 KV buffer 的物理页对 Real Client 可见,`store.batch_get_into / batch_put_from` 才能让 Real Client 的 Transfer Engine 直接对这段内存做零拷贝 DMA。

---

## 5. 数据读写路径(zero-copy)

Standalone 与 self-hosted 在 **I/O 路径上完全一致**(都走 zero-copy),区别只在初始化。`cache_controller.py:523-531` 把 mooncake 归入 zero-copy 路径:

```python
self.page_get_func = self._page_get_zero_copy
self.page_set_func = self._page_set_zero_copy
```

写入(backup,L2→L3),`batch_set_v1` `mooncake_store.py:815`:

```
host_indices ──► _batch_preprocess()
                   └─ MHA: 每 page 拆 K/V 两个 key + 取 buffer 指针/长度
                   └─ MLA: 每 page 一个 key(仅 K)
                 ──► _batch_exist() 去重(已存在的跳过)
                 ──► store.batch_put_from(keys, ptrs, sizes)  ← Real Client 零拷贝写
```

读取(prefetch,L3→L2),`batch_get_v1` `mooncake_store.py:790`:

```
keys ──► _batch_preprocess() 拿到 host buffer 指针
     ──► store.batch_get_into(keys, ptrs, sizes)  ← Real Client 把数据 DMA 进 host buffer
     ──► _batch_postprocess() 按 K/V 配对判断每 page 是否完整命中
```

Key 命名规则(决定 MHA/MLA/TP 切分):

| 场景 | Key 形式 | key_multiplier |
|---|---|---|
| MHA | `{hash}_{tp_rank}_k` / `{hash}_{tp_rank}_v` | 2 |
| MLA | `{hash}_{pp_rank}_k`(只存一份) | 1 |
| MHA + split heads | 每个 target rank 各一对 `_k`/`_v` | `2 * split_factor` |
| PP > 1 | suffix 追加 `_{pp_rank}` | — |

> MLA 下 `backup_skip` 让非 0 rank 不重复备份(`cache_controller.py:489`),进一步省存储与带宽。

---

## 6. 必须满足的约束

| 约束 | 位置 | 说明 |
|---|---|---|
| Host 内存布局 | `mooncake_store.py:553` | 仅支持 `page_first` / `page_first_direct` / `page_head` / `page_first_kv_split`,**不支持 `layer_first`**。CLI 需 `--hicache-mem-layout page_first` |
| 分配器 | `mooncake_store.py:348` | 必须 `MooncakeHostTensorAllocator`,要求 `mooncake >= 0.3.8.post1` |
| Real Client 先行 | — | `mooncake_client` 必须先于 SGLang 启动并监听 `client_server_address` |
| 必填配置 | `mooncake_store.py:199-205` | extra_config 里 `master_server_address` 或 `client_server_address` 至少有一个 |

---

## 7. 性能与工程权衡

**Standalone 模式的收益:**

- **进程隔离稳定性**:RDMA 连接/内存注册等易出错的重逻辑搬到独立进程,SGLang 推理进程更轻、更稳;Real Client 崩溃不必然拖垮推理,反之亦然。
- **缓存持久化**:SGLang 重启(改配置、OOM 重拉)时,KV cache 留在存活的 Real Client 内存里,**重启后命中率不归零** —— 这是相比 self-hosted 最实际的运维价值。
- **零拷贝不打折**:得益于共享内存分配器,跨进程仍是 RDMA 零拷贝,**没有引入额外的内存拷贝**。

**代价 / 注意点:**

- **多一跳 RPC/IPC 控制面开销**:数据面是零拷贝 DMA(无拷贝),但 `setup_dummy` / 元数据交互走本地 RPC,控制路径多一跳(数据量小,通常可忽略)。
- **部署复杂度**:多一个常驻进程 + 启动顺序依赖(Master → Real Client → SGLang)。
- **TP 场景**:每个 TP rank 各自起 Dummy Client 连本地 Real Client;`global_segment_size` 由 Real Client 侧统一规划,而非 self-hosted 下每 rank 贡献 `1/tp` 的逻辑。

---

## 8. 配置参考

三种等价配置方式(优先级见 §3)。

**方式一:extra-config(推荐)**

```bash
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --hicache-mem-layout page_first \
    --model-path [model_path] \
    --hicache-storage-backend-extra-config \
      '{"standalone_storage": true, "client_server_address": "127.0.0.1:50052"}'
```

**方式二:JSON 配置文件**

```bash
export SGLANG_HICACHE_MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
echo '{
    "standalone_storage": true,
    "client_server_address": "127.0.0.1:50052"
}' > ${SGLANG_HICACHE_MOONCAKE_CONFIG_PATH}

python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --hicache-mem-layout page_first \
    --model-path [model_path]
```

**方式三:环境变量**

```bash
MOONCAKE_STANDALONE_STORAGE=1 \
MOONCAKE_CLIENT="127.0.0.1:50052" \
python -m sglang.launch_server \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --hicache-mem-layout page_first \
    --model-path [model_path]
```

**关键参数对照(`MooncakeStoreConfig`,`mooncake_store.py:83`):**

| 字段 | 环境变量 | 默认 | standalone 下含义 |
|---|---|---|---|
| `standalone_storage` | `MOONCAKE_STANDALONE_STORAGE` | `False` | 置 `True` 启用 Dummy Client |
| `client_server_address` | `MOONCAKE_CLIENT` | `None` | 本地 Real Client 的 RPC 地址(默认 `:50052`) |
| `master_server_address` | `MOONCAKE_MASTER` | `None` | self-hosted 用;standalone 可不填 |
| `global_segment_size` | `MOONCAKE_GLOBAL_SEGMENT_SIZE` | `4gb` | standalone 下由 Real Client 持有,SGLang 侧不再贡献 |
| `enable_ssd_offload` | `MOONCAKE_ENABLE_SSD_OFFLOAD` | `False` | DRAM 溢出到 SSD,扩展 L3 容量 |

