# SGLang Router 使用指南与配置参数详解

## 概述

SGLang Router 是一个高性能的请求分发网关，负责将推理请求路由到后端 worker 实例。支持两种运行模式：

- **Regular 模式**：标准负载均衡，请求分发到一组 worker
- **PD 分离模式**（Prefill-Decode Disaggregation）：将 prefill 和 decode 阶段分配到不同 GPU 实例

Router 有两个实现版本：
- **Rust Router**（生产环境）：高性能，支持全部策略和完整功能
- **MiniLB**（Python，仅调试）：仅支持 random 策略的轻量级实现

## 启动方式

```bash
# 独立启动 router
python -m sglang_router.launch_router [OPTIONS]

# 通过 launch_server 集成启动（内置 router + workers）
python -m sglang_router.launch_server [OPTIONS]
```

## 使用示例

### Regular 模式

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy round_robin \
  --port 30000
```

### PD 分离模式

```bash
# 基本 PD 模式
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://prefill1:8000 9000 \
  --prefill http://prefill2:8000 \
  --decode http://decode1:8001 \
  --decode http://decode2:8001 \
  --policy cache_aware

# PD 模式 + prefill/decode 独立策略
python -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://prefill1:8000 9000 \
  --decode http://decode1:8001 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two

# 使用 MiniLB 调试
python -m sglang_router.launch_router --pd-disaggregation --mini-lb \
  --prefill http://prefill1:8000 9000 \
  --decode http://decode1:8001
```

---

## 配置参数详解

### 1. 基础服务配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--host` | str | `0.0.0.0` | 绑定地址，支持 IPv4/IPv6（如 `::` 监听所有 IPv6） |
| `--port` | int | `30000` | 监听端口 |
| `--worker-urls` | str[] | `[]` | Worker URL 列表（Regular 模式使用） |
| `--max-payload-size` | int | `536870912` (512MB) | 请求体最大字节数 |
| `--request-timeout-secs` | int | `1800` (30分钟) | 请求超时时间 |
| `--shutdown-grace-period-secs` | int | `180` (3分钟) | 关闭时等待在途请求完成的时间 |
| `--cors-allowed-origins` | str[] | `[]` | CORS 允许的来源列表 |

---

### 2. 路由策略配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--policy` | str | `cache_aware` | 全局负载均衡策略 |
| `--prefill-policy` | str | None | Prefill 节点专用策略（PD 模式，未设置则用 `--policy`） |
| `--decode-policy` | str | None | Decode 节点专用策略（PD 模式，未设置则用 `--policy`） |

可选策略值：`random`, `round_robin`, `cache_aware`, `power_of_two`, `bucket`, `manual`, `consistent_hashing`, `prefix_hash`

#### 策略详细说明

**random** — 随机选择
- 从健康 worker 中均匀随机选取
- 无状态，零协调开销

**round_robin** — 轮询
- 原子计数器递增，按健康 worker 数取模
- 严格均匀分发，不考虑请求大小差异

**cache_aware** — 缓存感知（默认策略）
- 维护每个 worker 的近似 Radix Tree，按请求文本前缀匹配路由
- 匹配率 > `cache_threshold` 时路由到最高匹配 worker（KV cache 热度最高）
- 匹配率低时路由到 tree 最小的 worker（缓存空间最多）
- 检测到负载不平衡时自动降级为 shortest queue 策略
- 需要 `request_text` 输入

**power_of_two** — 二选一最低负载
- 随机选 2 个 worker，选负载更低的
- 优先使用外部 LoadMonitor 上报的 token 级负载
- 若某 worker 缺少 token 数据则两者都 fallback 到 request count
- 以极低协调成本实现接近最优的负载均衡

**bucket** — 桶划分
- 按请求文本长度（char count）分桶，不同长度范围分配到不同 worker
- 周期性动态调整桶边界以适应流量分布变化
- 不平衡时 fallback 到最低负载 worker
- 需要 `request_text` 输入

**manual** — 手动强粘性
- 基于 `X-SMG-Routing-Key` HTTP header 做强绑定
- 新增 worker 不重分配已有 session
- 每个 routing key 保存最多 2 个候选 worker，支持快速 failover
- 空闲超时自动清理
- 无 routing key 时随机分配（不绑定）

#### Manual 策略详解

**核心机制：**

```
Client 请求
  │
  ├─ 有 X-SMG-Routing-Key: "user-123"
  │    └─ DashMap 查找 "user-123"
  │         ├─ 命中且 worker 健康 (OccupiedHit) → 路由到同一 worker
  │         ├─ 命中但 worker 不健康 (OccupiedMiss) → 按 assignment_mode 选新 worker，追加到候选列表
  │         └─ 未命中 (Vacant) → 按 assignment_mode 选 worker，写入映射表
  │
  └─ 无 routing key (NoRoutingId) → 随机选 worker（不写入映射表）
```

**数据结构：**
- 维护一张 `DashMap<routing_key, Node>`，每个 Node 保存最多 2 个候选 worker URL
- 候选列表采用 FIFO 淘汰：超过 2 个时删除最旧的
- 2 个候选设计支持主 worker 故障时快速 failover 到备用 worker，无需重新走 assignment_mode

**新 routing key 的 worker 分配模式（`assignment_mode`）：**

| 模式 | 说明 | 适用场景 |
|---|---|---|
| `random`（默认） | 随机选健康 worker | 流量均匀，无特殊倾向 |
| `min_load` | 选当前 in-flight 请求数最少的 worker | 请求处理时间差异大时更均匀 |
| `min_group` | 选已绑定 routing key 数量最少的 worker | 希望每个 worker 服务的租户数均衡 |

**TTL 自动淘汰：**
- 后台周期任务每 `eviction_interval_secs`（默认 60s）扫描一次映射表
- 超过 `max_idle_secs`（默认 4 小时）未访问的 routing key 被自动删除
- 日志示例：`ManualPolicy TTL eviction: evicted 5 entries, remaining 120 (max_idle: 14400s)`

**与 consistent_hashing 的对比：**

| 特性 | manual | consistent_hashing |
|---|---|---|
| worker 增加时 session 迁移 | **零迁移**，现有 key 不变 | ~1/N 的 key 迁移 |
| worker 故障时迁移 | 仅故障 key 迁移到备用 | hash ring 自动重路由 |
| 实现复杂度 | DashMap O(1) | hash ring O(log n) |
| 适用场景 | 强 session 亲和（有状态服务） | 缓存亲和（KV cache 复用） |

**启动配置：**

```bash
# 基本用法（默认 random 分配，4小时超时）
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8000 \
  --policy manual \
  --port 30000

# 自定义参数
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8000 http://worker3:8000 \
  --policy manual \
  --assignment-mode min_group \    # 新 key 分配到负载均衡的 worker
  --eviction-interval-secs 120 \   # 每 2 分钟扫描一次
  --max-idle-secs 3600 \           # 1 小时无活动则清理
  --port 30000
```

**客户端使用方式：**

```python
import openai

client = openai.OpenAI(base_url="http://router:30000/v1", api_key="token")

# 有 routing key：请求始终路由到同一 worker
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={"X-SMG-Routing-Key": "session-user-123"},
)

# 无 routing key：随机路由（无状态请求）
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
)
```

```bash
# curl 示例
curl http://router:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-SMG-Routing-Key: session-user-123" \
  -d '{"model": "...", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Prometheus 监控指标：**

| 指标 | 说明 |
|---|---|
| `manual_policy_branch{branch="occupied_hit"}` | routing key 命中率（黄金指标，越高表示 stickiness 越好） |
| `manual_policy_branch{branch="occupied_miss"}` | 候选 worker 不健康，触发重新分配 |
| `manual_policy_branch{branch="vacant"}` | 新 routing key 首次写入 |
| `manual_policy_branch{branch="no_routing_id"}` | 无 routing key，随机路由 |
| `manual_policy_cache_entries` | 当前映射表条目数（监控内存增长） |

**使用场景：**
- 有状态多轮对话，context 缓存在特定 worker 的 GPU 显存或内存中
- 需要严格 session 亲和且不能接受 worker 扩容时 session 迁移的场景
- 多租户服务中每个租户（routing key = tenant_id）固定路由到专属 worker

**注意事项：**
- mapping table 常驻内存，高并发场景下应关注 `manual_policy_cache_entries` 防止无界增长
- 适当设置 `--max-idle-secs`，避免大量历史 session 占用内存
- 若所有候选 worker 均不健康，会重新按 `assignment_mode` 分配，此时亲和性暂时中断

**consistent_hashing** — 一致性哈希
- 优先级：`X-SMG-Target-Worker` > `X-SMG-Routing-Key` > 隐式 key (Auth/Cookie) > random
- O(log n) 查找，worker 故障时仅 ~1/N 的 key 迁移
- 支持 failover 后自动回归

**prefix_hash** — 前缀哈希
- 取请求前 N 个 token 做 hash 路由到 hash ring 上对应 worker
- 目标 worker 过载时顺时针走 ring 找下一个
- 比 cache_aware 轻量（O(log n) vs O(prefix_len)），内存占用更低
- 需要请求中提供 tokenized input_ids

#### 策略专属参数

| 参数 | 适用策略 | 类型 | 默认值 | 说明 |
|---|---|---|---|---|
| `--cache-threshold` | cache_aware | float | `0.3` | 前缀匹配率阈值（0.0-1.0），低于此值路由到缓存空间最大的 worker |
| `--balance-abs-threshold` | cache_aware, bucket | int | `64` | 负载差绝对阈值，`(max - min) > abs` 时触发均衡 |
| `--balance-rel-threshold` | cache_aware, bucket | float | `1.5` | 负载差相对阈值，`max > min * rel` 时触发均衡 |
| `--eviction-interval-secs` | cache_aware, manual | int | `60` | 缓存/session 淘汰扫描间隔（秒） |
| `--max-tree-size` | cache_aware | int | `67108864` (2^26) | Radix Tree 最大节点数 |
| `--bucket-adjust-interval-secs` | bucket | int | `5` | 桶边界调整周期（秒） |
| `--max-idle-secs` | manual | int | `14400` (4小时) | session 空闲超时时间（秒） |
| `--assignment-mode` | manual | str | `random` | 新 routing key 分配模式：`random`/`min_load`/`min_group` |

---

### 3. PD 分离模式配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--pd-disaggregation` | flag | False | 启用 PD 分离模式 |
| `--prefill` | str (可重复) | `[]` | Prefill 实例 URL 和可选 bootstrap port。格式：`--prefill URL [PORT]` |
| `--decode` | str (可重复) | `[]` | Decode 实例 URL。格式：`--decode URL` |
| `--mini-lb` | flag | False | 使用 Python MiniLB（仅调试） |
| `--test-external-dp-routing` | flag | False | (MiniLB) 测试 DP routing 正确性 |
| `--worker-startup-timeout-secs` | int | `1800` (30分钟) | Worker 启动注册超时 |
| `--worker-startup-check-interval` | int | `30` | Worker 启动检查间隔（秒） |
| `--dp-aware` | flag | False | 启用数据并行感知调度 |
| `--enable-igw` | flag | False | 启用 IGW（推理网关）多模型路由模式 |

PD 模式下 Router 注入请求的字段：

| 注入字段 | 位置 | 说明 |
|---|---|---|
| `bootstrap_host` | JSON body | Prefill 实例 hostname |
| `bootstrap_port` | JSON body | Prefill bootstrap 通道端口 |
| `bootstrap_room` | JSON body | 随机生成的房间 ID，PD 实例配对凭证 |

---

### 4. HTTP 连接池配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--pool-idle-timeout-secs` | int | `50` | 连接池空闲连接超时（秒） |
| `--connect-timeout-secs` | int | `10` | 新建连接超时（秒） |
| `--pool-max-idle-per-host` | int | `500` | 每个 host 最大空闲连接数 |
| `--tcp-keepalive-secs` | int | `30` | TCP keepalive 空闲时间（秒） |

---

### 5. 限流配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--max-concurrent-requests` | int | `-1` | 最大并发请求数，`-1` 表示不限制 |
| `--queue-size` | int | `100` | 并发满时排队队列大小，`0` 表示无队列直接返回 429 |
| `--queue-timeout-secs` | int | `60` | 队列中等待的最长时间（秒） |
| `--rate-limit-tokens-per-second` | int | None | Token bucket 补充速率（tokens/s），默认等于 `max_concurrent_requests` |

---

### 6. 重试配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--retry-max-retries` | int | `5` | 最大重试次数 |
| `--retry-initial-backoff-ms` | int | `50` | 首次重试前等待（毫秒） |
| `--retry-max-backoff-ms` | int | `30000` | 最大退避时间（毫秒） |
| `--retry-backoff-multiplier` | float | `1.5` | 指数退避乘数 |
| `--retry-jitter-factor` | float | `0.2` | 退避抖动因子（0.0-1.0），公式：`D' = D * (1 + U[-j, +j])` |
| `--disable-retries` | flag | False | 禁用重试（等价于 max_retries=1） |

---

### 7. 熔断器配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--cb-failure-threshold` | int | `10` | 触发熔断的失败次数 |
| `--cb-success-threshold` | int | `3` | 半开状态恢复所需连续成功次数 |
| `--cb-timeout-duration-secs` | int | `60` | 熔断后重试间隔（秒） |
| `--cb-window-duration-secs` | int | `120` | 失败计数滑动窗口（秒） |
| `--disable-circuit-breaker` | flag | False | 禁用熔断器 |

---

### 8. 健康检查配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--health-failure-threshold` | int | `3` | 连续失败多少次标记为 unhealthy |
| `--health-success-threshold` | int | `2` | 连续成功多少次标记为 healthy |
| `--health-check-timeout-secs` | int | `5` | 健康检查请求超时（秒） |
| `--health-check-interval-secs` | int | `60` | 健康检查间隔（秒） |
| `--health-check-endpoint` | str | `/health` | 健康检查路径 |
| `--disable-health-check` | flag | False | 禁用健康检查 |

---

### 9. Prometheus 监控配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--prometheus-port` | int | `29000` | Prometheus metrics 暴露端口 |
| `--prometheus-host` | str | `0.0.0.0` | Prometheus 监听地址 |
| `--prometheus-duration-buckets` | float[] | None | 自定义 duration histogram 桶 |

---

### 10. 日志配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--log-dir` | str | None | 日志文件目录，不设置则输出到控制台 |
| `--log-level` | str | `info` | 日志级别：`debug`/`info`/`warn`/`error` |
| `--json-log` | flag | False | 启用结构化 JSON 日志 |

---

### 11. 分布式追踪（OpenTelemetry）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--enable-trace` | flag | False | 启用 OpenTelemetry 追踪 |
| `--otlp-traces-endpoint` | str | `localhost:4317` | OTLP collector 地址 |

---

### 12. Kubernetes 服务发现

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--service-discovery` | flag | False | 启用 K8s 服务发现 |
| `--selector` | str[] | `{}` | Pod 标签选择器（格式：`key=value`） |
| `--service-discovery-port` | int | `80` | 发现的 Pod 使用的端口 |
| `--service-discovery-namespace` | str | None | K8s namespace，不设置则监听所有 |
| `--prefill-selector` | str[] | `{}` | Prefill Pod 标签选择器（PD 模式） |
| `--decode-selector` | str[] | `{}` | Decode Pod 标签选择器（PD 模式） |
| `--bootstrap-port-annotation` | str | `sglang.ai/bootstrap-port` | K8s annotation key 获取 bootstrap port |

---

### 13. TLS/mTLS 安全配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--client-cert-path` | str | None | 客户端证书路径（mTLS 连接 worker） |
| `--client-key-path` | str | None | 客户端私钥路径 |
| `--ca-cert-paths` | str[] | `[]` | CA 证书路径（验证 worker TLS 证书） |
| `--tls-cert-path` | str | None | 服务端 TLS 证书（PEM） |
| `--tls-key-path` | str | None | 服务端 TLS 私钥（PEM） |

---

### 14. 认证与控制面

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--api-key` | str | None | 与 worker 通信的 API key |
| `--control-plane-api-keys` | str[] | `[]` | 控制面 API key（格式：`id:name:role:key`，role 为 admin/user） |
| `--control-plane-audit-enabled` | flag | False | 启用控制面审计日志 |
| `--jwt-issuer` | str | None | OIDC issuer URL |
| `--jwt-audience` | str | None | JWT audience claim |
| `--jwt-jwks-uri` | str | None | JWKS URI（不设置则从 issuer 自动发现） |
| `--jwt-role-claim` | str | `roles` | JWT 中角色字段名 |
| `--jwt-role-mapping` | str[] | `[]` | IDP 角色映射（格式：`idp_role=gateway_role`） |

---

### 15. Tokenizer 配置

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--model-path` | str | None | 模型路径（HuggingFace ID 或本地路径），用于加载 tokenizer |
| `--tokenizer-path` | str | None | 显式 tokenizer 路径（覆盖 model_path） |
| `--chat-template` | str | None | Chat template 文件路径 |
| `--tokenizer-cache-enable-l0` | flag | False | 启用 L0 缓存（完整字符串精确匹配） |
| `--tokenizer-cache-l0-max-entries` | int | `10000` | L0 缓存最大条目数 |
| `--tokenizer-cache-enable-l1` | flag | False | 启用 L1 缓存（前缀匹配） |
| `--tokenizer-cache-l1-max-memory` | int | `52428800` (50MB) | L1 缓存最大内存 |

---

### 16. 请求解析器

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--reasoning-parser` | str | None | 推理模型解析器（如 `deepseek-r1`, `qwen3`） |
| `--tool-call-parser` | str | None | 工具调用解析器（如 `json`, `qwen`） |
| `--mcp-config-path` | str | None | MCP 服务器配置文件路径 |

---

### 17. 后端与存储

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--backend` | str | `sglang` | 后端运行时：`sglang`/`openai` |
| `--history-backend` | str | `memory` | 历史存储：`memory`/`none`/`oracle`/`postgres`/`redis` |
| `--enable-wasm` | flag | False | 启用 WebAssembly 支持 |

---

### 18. 数据库配置（按 history-backend 选择）

#### Oracle

| 参数 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| `--oracle-wallet-path` | `ATP_WALLET_PATH` | None | Oracle ATP wallet 目录 |
| `--oracle-tns-alias` | `ATP_TNS_ALIAS` | None | TNS 别名 |
| `--oracle-connect-descriptor` | `ATP_DSN` | None | 连接描述符/DSN |
| `--oracle-username` | `ATP_USER` | None | 用户名 |
| `--oracle-password` | `ATP_PASSWORD` | None | 密码 |
| `--oracle-pool-min` | `ATP_POOL_MIN` | `1` | 连接池最小连接数 |
| `--oracle-pool-max` | `ATP_POOL_MAX` | `16` | 连接池最大连接数 |
| `--oracle-pool-timeout-secs` | `ATP_POOL_TIMEOUT_SECS` | `30` | 连接池超时（秒） |

#### PostgreSQL

| 参数 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| `--postgres-db-url` | `POSTGRES_DB_URL` | None | 连接 URL |
| `--postgres-pool-max` | `POSTGRES_POOL_MAX` | `16` | 最大连接数 |

#### Redis

| 参数 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| `--redis-url` | `REDIS_URL` | None | Redis 连接 URL |
| `--redis-pool-max` | `REDIS_POOL_MAX` | `16` | 最大连接数 |
| `--redis-retention-days` | `REDIS_RETENTION_DAYS` | `30` | 数据保留天数（`-1` 永久） |

---

### 19. 请求 ID 与追踪

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--request-id-headers` | str[] | None | 自定义请求 ID header 列表（如 `x-request-id x-trace-id`） |

---

## Router 请求转发的 HTTP Headers

Router 在转发请求到 worker 时会透传以下 headers：

| Header | 说明 |
|---|---|
| `authorization` | 认证信息 |
| `x-request-id` | 请求 ID |
| `x-correlation-id` | 关联 ID |
| `traceparent` | W3C 分布式追踪 |
| `tracestate` | 追踪状态 |
| `x-smg-routing-key` | 路由 key（用于 manual/consistent_hashing 策略） |
| `x-request-id-*` | 所有以此前缀开头的 header |

## 路由控制 Headers（客户端可用）

| Header | 适用策略 | 说明 |
|---|---|---|
| `X-SMG-Routing-Key` | manual, consistent_hashing | 路由 key，实现 session 亲和 |
| `X-SMG-Target-Worker` | consistent_hashing | 直接指定 worker index（0-based） |

---

## 架构图

```
                         ┌─────────────────┐
                         │   Client / LB   │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │   SGLang Router  │
                         │  (Rust/Python)   │
                         │                  │
                         │  ┌────────────┐  │
                         │  │   Policy   │  │
                         │  │  Registry  │  │
                         │  └────────────┘  │
                         └──┬───────────┬───┘
                            │           │
              ┌─────────────▼──┐    ┌───▼─────────────┐
              │  Prefill Pool   │    │   Decode Pool    │
              │  (prefill_policy)│    │  (decode_policy) │
              ├────────────────┤    ├──────────────────┤
              │ Worker P1       │    │ Worker D1        │
              │ Worker P2       │    │ Worker D2        │
              │ ...             │    │ ...              │
              └────────────────┘    └──────────────────┘
```

PD 模式请求流：
```
Client → Router → inject bootstrap fields
                → POST 同一请求到 Prefill + Decode
                → Prefill 完成 KV 计算 → 通过 bootstrap_room 传输 KV cache
                → Decode 收到 KV → 执行 decode → 返回结果给 Router → Client
```
