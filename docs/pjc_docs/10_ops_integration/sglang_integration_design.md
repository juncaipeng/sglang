# RL × SGLang 服务创建与交互设计

> 状态：Active
> 更新时间：2026-08-05
> Scope：本文描述当 `inference_backend.type = sglang` 时，PaddleRL 如何创建 SGLang 推理服务、如何与其交互（拉起 → 就绪 → 加权重 → 推理 → 换权重 → 关闭）的完整链路与方案。聚焦 SGLang 特有细节；通用后端抽象见 [inference_backend_design.md](./inference_backend_design.md)，通用调度路由见 [inference_scheduling_design.md](./inference_scheduling_design.md)。

---

## 1. 概述

PaddleRL 是**控制面**，SGLang 是被它拉起并托管的**独立 HTTP 服务子进程**。RL 通过一套「四层 Manager/Agent + 独立 Router」把每个 SGLang server 的整个生命周期管起来：

- **不是** 在 RL 进程内 `import sglang` 提供推理，而是每组 GPU 拉一个独立的 `python -m sglang.launch_server` 子进程，监听 `0.0.0.0:<port>` 的 HTTP 服务。
- Ray actor 只负责**编排、健康、权重注入**，**不代理生成流量**——生成请求由 rollout worker 经 router 直接打到 SGLang server。
- 权重从 trainer 经 **IPC / RDMA / Disk** 三种协议注入；PD 分离由 **SGLang 原生 router + Mooncake transfer engine** 撮合 prefill↔decode；MoE 训推路由一致性由 **R3 p2pstore** 回放保证。

---

## 2. 分层架构

```
RLRayTrainer (rl_ray_trainers/*.py)                  ← step 循环 / 权重同步节拍
        │
RolloutManager (controller/rollout/rollout_manager.py)   ← 编排：backend + router + reward
        ├── inference_backend_manager  ─┐
        └── inference_router_manager    │
                                        ▼
  ┌──────────────────────────┐   ┌───────────────────────────┐
  │ SGLangManager            │   │ SGLangRouterManager        │  (与通用 router 二选一)
  │ (控制面, 普通类)          │   │ / InferenceRouterManager   │
  └───────────┬──────────────┘   └────────────┬──────────────┘
              │ ray.remote 扇出                │ ray.remote
              ▼                                ▼
  ┌──────────────────────────┐   ┌───────────────────────────┐
  │ SGLangAgent (@ray.remote) │  │ SGLangRouterAgent          │
  │  一个 actor = 一个 server  │  │  拉起 sglang_router 进程    │
  └───────────┬──────────────┘   └───────────────────────────┘
              │ 持有
              ▼
  ┌──────────────────────────┐
  │ SGLangBackend (普通类)    │  ← 端口分配 + subprocess.Popen
  └───────────┬──────────────┘
              ▼
     bash start_sglang.sh → eval python -m sglang.launch_server (HTTP :port)
```

| 层级 | 组件 | 文件 | 职责 |
|------|------|------|------|
| 编排 | `RolloutManager` | `controller/rollout/rollout_manager.py` | 拉起并编排 backend + router + reward，驱动权重生命周期 |
| 管理 | `SGLangManager` | `controller/rollout/inference/backends/sglang/sglang_manager.py` | SGLang 实例拓扑扇出、权重 fan-out、健康、弹性、关闭 |
| 执行 | `SGLangAgent` | `.../backends/sglang/sglang_agent.py` | `@ray.remote`，一个 actor 管一个 server；权重注入、健康、控制面 HTTP |
| 进程 | `SGLangBackend` | `workers/inference/sglang/sglang_backend.py` | 端口分配 + `subprocess.Popen` 拉起脚本 |
| 脚本 | `start_sglang.sh` / `stop_sglang.sh` | `workers/inference/sglang/` | 实际 `launch_server` 命令 / 按端口 `kill -9` |
| 路由 | `SGLangRouterManager` / `SGLangRouterAgent` | `.../inference/router/sglang/` | 拉起 SGLang 原生 router 进程，注册 worker |

---

## 3. 选型与配置工厂

后端与 router 均在 `RolloutManager` 构造期按配置分派，二者相互独立：

- **后端选型**：`_create_inference_backend_manager`（`rollout_manager.py:161`）按 `inference_backend.type` 分派，`"sglang"` → `SGLangManager`。
- **Router 选型**：`_create_inference_router_manager`（`rollout_manager.py:181`）按 `is_sglang_router_enabled`（即 `sglang.router.enabled == true`）分派：
  - `true` → `SGLangRouterManager`（SGLang 原生 router）
  - `false` → `InferenceRouterManager`（通用 Go infer-router）

关键配置（`core/config/args.py`）：

| 配置项 | 位置 | 说明 |
|--------|------|------|
| `SGLangBackendConfig` | `args.py:1566` | `engine` / `engine_prefill` / `engine_decode` / `router` 四组子配置 |
| `engine_prefill` / `engine_decode` | 同上 | PD 分离下 prefill / decode 各自的 `extra_args` |
| `router.enabled` | 同上 | 是否启用 SGLang 原生 router |
| `InferenceRouterPolicyArguments` | `args.py:1698` | `prefill_policy` / `decode_policy` |
| `trans_protocol` | `args.py:906` | 权重传输协议（见 §8） |
| `DisaggregationWorkerArguments` | `args.py:2492` | `pd_disaggregation` 下 prefill / decode 的 GPU 拓扑 |
| `rollout_server_mode` | recipe yaml | `"disaggregation"` 时启用 PD 分离 |

recipe 示例见 `recipe/fully_async/deepseek_v4-32nodes-gbzz2-swe_cc-fully_async-256k-pd.yaml`：`engine_prefill` / `engine_decode` 各带大量 `extra_args`（:243-327），`router.enabled: true`（:332），`rollout_server_mode: "disaggregation"`（:373），`pd_disaggregation` prefill 32 GPU + decode 32 GPU（:374-380）。

---

## 4. launch 并发流程

`RolloutManager.launch`（`rollout_manager.py:312`）**并发**拉起 backend 与 router，而非串行：

1. 通过 `asyncio.create_task` 同时启动 backend、router 两条链（:341-344）。
2. 以 `asyncio.wait(..., return_when=FIRST_EXCEPTION)`（:359）等待，任一失败立即暴露。

`SGLangManager.launch`（`sglang_manager.py:424`）内部顺序：

1. **Mooncake 准备**：PD 分离时准备 transfer engine 环境。
2. **`_prepare_env`**：写 `common_env.sh` / `inference_backend_env.sh`（由 `start_sglang.sh:115-120` source）。
3. **`_create_agents`**：按拓扑创建 `SGLangAgent`；PD 分离走 disaggregation 分支，按 `server_mode` 用 `_spawn_agents_on_nodes(node_ips, wtype, node_offset)` 分别铺 prefill / decode。
4. **扇出 `agent.start.remote()`**：每个 agent 并行拉起自己的 server。

`_spawn_agent`（`sglang_manager.py:357`）用 `NodeAffinitySchedulingStrategy(soft=False)` **硬绑节点**，保证 GPU 拓扑与 IB / NVLink 亲和不被 Ray 打散。

---

## 5. 单实例进程拉起

`SGLangAgent.start`（`sglang_agent.py:405`）调 `SGLangBackend.start`（`sglang_backend.py:209`）：

1. **端口分配** `set_ports`（`sglang_backend.py:115`）：API 区与 nccl 区用 240 的 headroom 隔开，避免多 TP / 多实例撞端口；PD 分离时 prefill 额外分配 `bootstrap_port`（:199-203）。
2. **组装启动命令**（`sglang_backend.py:263-276`）：拼 `start_sglang.sh` 的定位参数（venv / port / gpu / tp / model / log / root + `EXTRA_ARGS`）。
3. **PD flags 注入**（`sglang_backend.py:255-259`）：`--disaggregation-mode {worker_type}`、`--disaggregation-transfer-backend mooncake`；prefill 追加 `--disaggregation-bootstrap-port`。
4. **拉起子进程**：`subprocess.Popen(preexec_fn=os.setsid)`（`sglang_backend.py:283`）独立进程组，便于按组回收。

`start_sglang.sh` 实际执行（:170-177）：

```bash
eval python -m sglang.launch_server \
    --model-path "$MODEL_PATH" --tp "$TP_SIZE" \
    --host 0.0.0.0 --port "$PORT" --enable-metrics "$EXTRA_ARGS" \
    >> "${IB_LOG_DIR}/${GPU_ID}-stdout.log" 2>> "${IB_LOG_DIR}/${GPU_ID}-stderr.log"
```

- `PYTHONPATH` 指向 vendored 的 `third_party/sglang/python`（:111），用仓内 SGLang 而非 pip 版本。
- `SGLANG_HEALTH_CHECK_TIMEOUT=300`（:144）：RL 场景模型初始化远超默认 20s，避免误判启动超时。
- PD 分离超时 env（:150-158，通过检测 `EXTRA_ARGS` 中的 `--disaggregation-mode` 触发）：`SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` / `WAITING_TIMEOUT` / `MC_TRANSFER_TIMEOUT` 等一并放大到 800s。

**就绪探测**：`SGLangAgent.check_instance_ready`（`sglang_agent.py:3139`）轮询健康端点，`max_retry=720`，覆盖大模型冷启动。

---

## 6. PD 分离三处协同

PD（prefill / decode disaggregation）分离在三个层面协同：

1. **拓扑分配**（`utils/core/node_selector.py`）：`get_inference_nodes_by_type`（:282）让 prefill 占前 N 个节点、decode 紧随其后；`has_shared_pd_node`（:325）判断是否 prefill / decode 共节点。
2. **进程 flags**（`sglang_backend.py:255-259`）：为每个 worker 注入 `--disaggregation-mode prefill|decode` + `--disaggregation-transfer-backend mooncake`；prefill 带 `--disaggregation-bootstrap-port`。
3. **KV cache 传输**：Mooncake transfer engine 撮合 prefill→decode 的 KV cache，SGLang 原生 router 依据 worker `role` + `bootstrap_port` 做 prefill↔decode 配对。

`SGLangAgent.get_instance`（`sglang_agent.py:705-711`）在 PD 模式下上报 `role` 与 `bootstrap_port`，供 router 注册配对使用。disaggregated 模式下权重更新会跳过 `release/resume_memory_occupation`（`sglang_agent.py:1059-1065`），因为显存由 transfer engine 侧独立管理。

---

## 7. 两类 Router 与请求路径

PaddleRL 支持两套 router，由 `sglang.router.enabled` 二选一：

| 维度 | SGLang 原生 router | 通用 Go infer-router |
|------|-------------------|---------------------|
| 开关 | `sglang.router.enabled = true` | 默认（`false`） |
| 进程 | `python -m sglang_router.launch_router` | Go 二进制 |
| API 前缀 | `/v1`（OpenAI 兼容） | `/api/v2` |
| Step 控制 | `supports_step_control = False`（`sglang_router_manager.py:51`） | 支持（`--splitwise`） |
| PD 分离 | 原生支持 prefill↔decode 配对 | 需 splitwise 模式 |

**Router 注册与刷新**：

- `RolloutManager.notify_ib_instances`（`rollout_manager.py:879`）：`get_instances` → `router.notify_instance_address`。
- `SGLangRouterManager.notify_instance_address`（`sglang_router_manager.py:239`）→ `start_inference`（:249）→ `push_instances`。
- `SGLangRouterAgent.push_instances`（`sglang_router_agent.py:833`）：**全量替换**——先 `DELETE` 所有 worker，再逐个 `POST /workers`；`_register_worker`（:572）在 prefill 时带 `bootstrap_port`（:590-591）。

**请求路径**（`rollout_worker.py` + `utils/rollout/chat_completion.py`）：

1. `get_rollout_url_pool`（`rollout_worker.py:371`）取 router / server URL 池。
2. api_prefix 选择（`rollout_worker.py:400`）：`"/v1" if is_sglang_router_enabled else "/api/v2"`。
3. `build_chat_completion_request_spec`（`chat_completion.py:184`）组装请求：
   - **session 亲和**：`X-SMG-Routing-Key` header = session_id（:298），配合 router `session_aware` 策略把同一 session 打到同一 worker（recipe `max_session_load: 48`）。
   - **PD 强制**：`request_id` = trajectory id（:314-315, :332-333），保证 prefill / decode 两段用同一 id 配对。
   - url = `{remote_url}/chat/completions`（:337）。

> 注意：生成流量**不经过** Ray actor，rollout worker 直连 router / server，Ray 侧只做编排与权重注入。

---

## 8. 权重生命周期与 TransProtocol

权重从 trainer 注入 SGLang server 有三种协议（`utils/train/comm_utils.py:496-504` 的 `TransProtocol`）：

| 值 | 协议 | 说明 |
|----|------|------|
| 0 | `IPC_ASYNC` | 同机 CUDA IPC，异步 |
| 1 | `RDMA_FULL` | 跨机 RDMA 全量 |
| 2 | `RDMA_FULL_MEMORY` | RDMA 全量（常驻内存） |
| 3 | `DISK` | 落盘 → server 从磁盘加载 |

**trainer 侧发起**：`ppo_trainer.py:2194` `sync_weights_for_rollout` → `checkpoint_transfer.send_weights_async(full_itr, step_id)`（:2308）。

**Manager fan-out**：`SGLangManager.load_weight`（:625）/ `async_load_weight`（:687）/ `clear_weight`（:806）把权重扇出到各 `SGLangAgent`。

**Agent HTTP 时序**（`sglang_agent.py:775` `load_weight` / :876 `async_load_weight`，docstring :779-784）：

```
pause_generation → resume_memory_occupation → update_weights_from_tensor/disk
                 → release / continue_generation → health
```

- disaggregated 模式跳过 `release/resume_memory_occupation`（:1059-1065）。
- `clear_weight`（`sglang_agent.py:2847`）释放注入的权重。

**fully_async 换权重节拍**（`rl_ray_trainers/fully_async_rl_ray_trainer.py:337-350`）：`asyncio.gather(reshard, abort_rollout(blocking=False))` → `create_task(async_load_weight(step+1))`——先抢占 reshard + abort 在飞请求，再异步换权重，最大化重叠。

---

## 9. 单 SGLang server 视角的信号时序与 curl 示例

本节从**单个 SGLang server** 的视角，按时间线列出 RL 侧发来的每一个交互/探查信号、触发它的代码位置、**超时上限**，以及可直接在命令行复现的 `curl`。

> **关于「多久返回」**：表中时间是**代码写死的超时上限（最长等待）**，不是实测延迟。健康/控制类通常亚秒级返回；`update_weights_*` 的真实耗时取决于模型大小与协议，是换权重最慢的一环，超时只是保护上限。真实分布见埋点 metric `paddlerl_sglang_http_duration_seconds`（按 `api` 打 tag，`sglang_agent.py:2960`）与 `paddlerl_sglang_load_weight_duration_seconds`（:855）。
>
> 下方 `curl` 用占位符 `$HOST`（server 所在 IP）、`$PORT`（`inference_port`）、`$ROUTER`（router `host:port`）、`$MODEL`（模型名）。可先 `export HOST=... PORT=... ROUTER=... MODEL=...` 再粘贴执行。

### 9.1 阶段一 · 创建服务：先探查 ready，不发业务信号

起进程后 **不发任何 HTTP 业务信号**，直接进入就绪轮询：`SGLangAgent.start`（`sglang_agent.py:405`）→ `check_instance_ready`（:3139）。

| 信号 | 触发点 | 方向 | 超时/预算 |
|------|--------|------|-----------|
| `GET /health`（启动轮询） | `_check_single_port_with_retry`（:3122，`retries=720, interval=5s` :266-267） | RL 探查 | 单次探针 **5s**；最多 720×5s ≈ **1 小时**总预算，健康即返回 |

```bash
# 启动就绪探查（RL 会重试至多 720 次，每次间隔 5s）
curl -sS -m 5 "http://$HOST:$PORT/health"

curl -v --connect-timeout 5 http://10.99.90.138:30000/health
```

### 9.2 阶段二 · 首次 ready 后：先校验+起监控，再灌权重/注册，最后才推理

ready 后 `start()` 内立刻发（:481-505），随后由 trainer 编排注册与首个推理请求。**第一个业务信号是权重校验，不是推理请求。**

| 顺序 | 信号 | 触发点 | 方向 | 超时上限 |
|------|------|--------|------|----------|
| 1 | `POST /weights_checker {"action":"checksum"}` | `weights_checker`（:3214） | RL→SGLang | **1200s** |
| 2 | 后台 `GET /health`（常驻） | `_health_check_task`（`base_agent_mixin.py:1655`） | RL 探查 | 每 **10s** 一次；连续 60 次失败（≈10min）才判死 |
| 3 | `POST /workers`（router 注册本 server 为 worker） | `_register_worker`（`sglang_router_agent.py:572`） | RL→Router | **60s** |
| 4 | `POST /chat/completions`（推理，rollout worker 直连，不经 Ray） | `build_chat_completion_request_spec`（`chat_completion.py:184`） | Worker→Router | **36000s**（`chat_completion.py:401`） |

```bash
# 1) 权重校验（首次 ready 后 start() 内立即发）
curl -sS -m 1200 -X POST "http://$HOST:$PORT/weights_checker" \
  -H 'Content-Type: application/json' \
  -d '{"action":"checksum"}'

# 2) 后台健康探针（RL 常驻，每 10s 一次）
curl -sS -m 5 "http://$HOST:$PORT/health"

# 3) 向 router 注册本 server 为 worker（mixed 模式）
curl -sS -m 60 -X POST "http://$ROUTER/workers" \
  -H 'Content-Type: application/json' \
  -d "{\"url\":\"http://$HOST:$PORT\",\"worker_type\":\"regular\"}"
#   PD 分离下 prefill worker 需带 bootstrap_port：
#   -d "{\"url\":\"http://$HOST:$PORT\",\"worker_type\":\"prefill\",\"bootstrap_port\":8998}"
#   decode worker：worker_type=decode（不带 bootstrap_port）

# 4) 推理请求（经 router；X-Request-Id=轨迹id，X-SMG-Routing-Key=session 亲和）
curl -sS -m 600 -X POST "http://$ROUTER/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "X-Request-Id: traj-0001" \
  -H "X-SMG-Routing-Key: session-abc" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":32,\"stream\":false}"
```

```
curl -v -m 1200 \
  -X POST "http://$HOST:$PORT/weights_checker" \
  -H "Content-Type: application/json" \
  -d '{"action":"checksum"}'

curl -v -m 5 \
    "http://$HOST:$PORT/health"


```

<!-- SEC9-MARKER -->

### 9.3 阶段三 · 换权重：一串有序信号 + 健康探查

每个 server 的换权重时序（DISK 协议 `_load_weight_for_instance_disk`（:1521）；IPC 协议 `_load_weight_for_instance_bucket_ipc`（:1586）；docstring :779-784）。**PD 分离下跳过 release/resume_memory**（保 Mooncake RDMA 上下文，:1061）。

| 顺序 | 信号 | 触发点 | 超时上限 | PD 分离 |
|------|------|--------|----------|---------|
| 1 | `POST /pause_generation {"mode":"abort"}` | `_pause_generation`（:2952） | **60s** | 执行 |
| 2 | 首步 `POST /release_memory_occupation {"tags":["kv_cache"]}`；后续步 `POST /resume_memory_occupation {"tags":["weights"]}` | `_release_memory`（:3027）/`_resume_memory`（:3056） | **120s** | **跳过** |
| 3 | 权重注入：DISK→`POST /update_weights_from_disk`；IPC/CT→`POST /update_weights_from_tensor`（分桶多次） | `_update_weights_from_disk`（:3081）/`_update_weights_from_tensor`（:2791） | **600s** each | 执行 |
| 4 | `POST /resume_memory_occupation {"tags":["kv_cache"]}` | `_resume_memory`（:3056） | **120s** | **跳过** |
| 5 | `POST /continue_generation` | `_continue_generation`（:2977） | **60s** | 执行 |
| 6 | `GET /health`（轮询，`retries=120, interval=1s`） | `health_with_retry`（:3162，调用点 :1568/:1826） | 最多 **120s**，健康即返回 | 执行 |
| 7 | `POST /weights_checker {"action":"checksum"}` | `weights_checker`（:3214，调用点 :866） | **1200s** | 执行 |

```bash
# 1) 暂停生成并清空请求队列
curl -sS -m 60 -X POST "http://$HOST:$PORT/pause_generation" \
  -H 'Content-Type: application/json' -d '{"mode":"abort"}'

# 2) 换权重前腾显存（首步释放 kv_cache；后续步先恢复 weights）。PD 分离下跳过此步。
curl -sS -m 120 -X POST "http://$HOST:$PORT/release_memory_occupation" \
  -H 'Content-Type: application/json' -d '{"tags":["kv_cache"]}'
#   后续步（clear_weight 已释放权重后）改为：
#   curl -sS -m 120 -X POST "http://$HOST:$PORT/resume_memory_occupation" -d '{"tags":["weights"]}'

# 3a) DISK 协议：从 checkpoint 目录加载新权重
curl -sS -m 600 -X POST "http://$HOST:$PORT/update_weights_from_disk" \
  -H 'Content-Type: application/json' \
  -d '{"model_path":"/path/to/checkpoint","load_format":"safetensors","flush_cache":true}'
# 3b) IPC/CT 协议：发送序列化的 CUDA IPC tensor（serialized_named_tensors 由 RL 侧生成，
#     此处仅示意 body 结构，实际内容为二进制序列化字符串，不适合手工构造）
curl -sS -m 600 -X POST "http://$HOST:$PORT/update_weights_from_tensor" \
  -H 'Content-Type: application/json' \
  -d '{"serialized_named_tensors":"<serialized>","load_format":null,"flush_cache":true,"weight_version":"42"}'

# 4) 权重就位后恢复 kv_cache（PD 分离下跳过）
curl -sS -m 120 -X POST "http://$HOST:$PORT/resume_memory_occupation" \
  -H 'Content-Type: application/json' -d '{"tags":["kv_cache"]}'

# 5) 恢复对外服务
curl -sS -m 60 -X POST "http://$HOST:$PORT/continue_generation" \
  -H 'Content-Type: application/json' -d '{}'

# 6) 换权重后健康探查（RL 最多重试 120 次，每次间隔 1s）
curl -sS -m 5 "http://$HOST:$PORT/health"

# 7) 权重校验
curl -sS -m 1200 -X POST "http://$HOST:$PORT/weights_checker" \
  -H 'Content-Type: application/json' -d '{"action":"checksum"}'
```

**fully_async 抢占**：换权重前 trainer 额外向所有 server 广播 abort（`fully_async_rl_ray_trainer.py:337`）：

| 信号 | 触发点 | 超时 |
|------|--------|------|
| `POST /abort_request {"abort_all":true}` | `SGLangManager.abort_all_requests`（`sglang_manager.py:886`） | **30s** |

```bash
# 抢占：掐断该 server 上所有在飞请求
curl -sS -m 30 -X POST "http://$HOST:$PORT/abort_request" \
  -H 'Content-Type: application/json' -d '{"abort_all":true}'
```

> **`/v1/steps/pause`、`/v1/steps/continue`（60s）只对通用 infer-router 生效**；SGLang 原生 router `supports_step_control=False`，对它**跳过**（`rollout_manager.py:608`），抢占完全靠上面的 `abort_request` + `push_instances` 全量替换。

### 9.4 换权重后的 router 全量刷新

换权重完成后走 `push_instances`（`sglang_router_agent.py:834`）**全量替换**：先列现有 worker，逐个删，再逐个重新注册。

| 信号 | 触发点 | 超时 |
|------|--------|------|
| `GET /workers`（列现有） | `_list_worker_ids`（调用点 :859） | **60s** |
| `DELETE /workers/{id}`（逐个删） | `_unregister_worker`（:663） | **60s** each |
| `POST /workers`（逐个注册） | `_register_worker`（:572） | **60s** each |

```bash
# 列出 router 上现有 worker
curl -sS -m 60 "http://$ROUTER/workers"

# 注销某个 worker（worker_id 取自上一步返回）
curl -sS -m 60 -X DELETE "http://$ROUTER/workers/<worker_id>"

# 重新注册（同 §9.2 第 3 步）
curl -sS -m 60 -X POST "http://$ROUTER/workers" \
  -H 'Content-Type: application/json' \
  -d "{\"url\":\"http://$HOST:$PORT\",\"worker_type\":\"regular\"}"
```

### 9.5 阶段四 · 清权重与关闭

| 信号 | 触发点 | 超时 | 说明 |
|------|--------|------|------|
| `POST /pause_generation {"mode":"abort"}` | `_clear_weight_for_instance`（:2925） | **60s** | clear_weight 第一步 |
| `POST /release_memory_occupation {"tags":["kv_cache","weights"]}` | 同上 | **120s** | 释放全部显存；PD 分离跳过 |
| 关闭 | `stop_sglang.sh` | — | `lsof -ti tcp:$PORT` + `kill -9` **按端口硬杀**，无 HTTP 优雅退出 |

```bash
# clear_weight：暂停 + 释放 kv_cache 与 weights
curl -sS -m 60  -X POST "http://$HOST:$PORT/pause_generation" -d '{"mode":"abort"}'
curl -sS -m 120 -X POST "http://$HOST:$PORT/release_memory_occupation" \
  -H 'Content-Type: application/json' -d '{"tags":["kv_cache","weights"]}'

# 关闭（RL 实际做法：按端口硬杀，无优雅退出信号）
kill -9 $(lsof -ti tcp:$PORT)
```

---

## 10. R3 MoE 路由回放（训练侧，非 HTTP 路由）

R3（Routing Replay）解决 MoE 模型**训推路由一致性**问题：rollout（SGLang 推理）时记录每个 token 的专家选择，训练时回放同一路由，避免训推 expert 分配漂移导致的 KL 偏差。

- **进程侧开关**：`SGLangAgent._handle_r3_config`（`sglang_agent.py:3175`）向 server 注入 `--enable-r3-p2pstore`，路由 tensor 经 p2pstore（rdma / gpfs）跨进程共享。
- **训练侧回放**：`utils/train/routing_manager.py`（`RoutingTensorManager`）+ `utils/train/routing_patch.py` 对 topk 选路打 patch：
  - `_process_r3_topk_indices`（`routing_patch.py:31`）：从缓存取回放 indices，用 `paddle.where(mask, cached, base)` 覆盖当前 topk。
  - `r3_patched_routing` / `paddlefleet_topk_noaux_tc_r3_patch` / `paddlefleet_call_topk_method_r3_patch`：覆盖不同 topk_method 变体。
  - 诊断：`cal_routing_overlap_ratio` 打印整体 / response-only / prompt-only 重合率，定位 PD 分离下 response 路由回放错位。

> 澄清：`routing_manager.py` / `routing_patch.py` 是 **trainer 侧 MoE 专家路由回放缓存**，与 §7 的 HTTP 请求路由是两回事，勿混淆。

---

## 11. 健康、弹性与关闭

- **健康探测**：`check_instance_ready`（`sglang_agent.py:3139`，`max_retry=720`）轮询就绪；控制面 HTTP 健康端点贯穿权重时序末端。
- **弹性接线**：`RolloutManager` 在 `:88-91` 通过 `set_inference_router_manager` 把 router 注入 elastic 流程；实例增减经 `notify_ib_instances` → `push_instances` 全量刷新到 router。
- **Step 控制门控**：`_notify_routers_pause` / `_notify_routers_continue`（`rollout_manager.py:604` / `:643`）用 `getattr(router, "supports_step_control", True)` 门控——SGLang 原生 router 为 `False`，故对它 **跳过** pause/continue，抢占完全依赖「backend abort + push_instances 全量替换」。
- **关闭**：`SGLangBackend.stop`（`sglang_backend.py:302`）调 `stop_sglang.sh`，脚本用 `lsof -ti tcp:<port>` + `kill -9` **按端口硬杀**（:22-37），无 SIGTERM 优雅退出。

---

## 12. 安全与工程注意点

| 项 | 说明 | 风险 |
|----|------|------|
| 无鉴权 | SGLang server 绑 `0.0.0.0`，控制面端点（`/update_weights_from_disk`、`/release_memory_occupation`）**无鉴权** | 同网段可越权注入权重 / 释放显存；需靠网络隔离兜底 |
| 硬杀关闭 | `stop_sglang.sh` 用 `kill -9` 按端口终止 | 无优雅退出，极端情况下可能残留临时文件 / 显存 |
| 冗余 flag | `--tp` 与 `--tp-size` 并存 | 冗余但不冲突，宜后续收敛 |
| 错误信息不符 | `inference_router_agent.py:905` 错误信息与实际分支不符 | 仅文案问题，不影响功能 |

---

## 13. 端到端时序小结

```
config(type=sglang) → RolloutManager 构造 backend+router 工厂
     → launch (并发): SGLangManager.launch          RouterManager.launch
          → _prepare_env → _create_agents              → 拉起 sglang_router
          → agent.start.remote() 扇出
               → SGLangBackend.start → start_sglang.sh → python -m sglang.launch_server (:port)
     → check_instance_ready (max_retry=720)
     → notify_ib_instances → push_instances (router 全量注册 worker)
     → [首次] load_weight: pause → memory → update_weights → continue → health
     → rollout worker 直连 router /chat/completions (session 亲和 + PD request_id 配对)
     → [每 step] fully_async: reshard + abort_rollout → async_load_weight(step+1)
     → 训练侧 R3 回放路由 (routing_patch)
     → 结束: SGLangBackend.stop → stop_sglang.sh (kill -9 by port)
```

---

## 14. 参考文件索引

| 关注点 | 文件 |
|--------|------|
| 编排 | `controller/rollout/rollout_manager.py` |
| SGLang 管理 | `controller/rollout/inference/backends/sglang/sglang_manager.py` |
| SGLang Agent | `controller/rollout/inference/backends/sglang/sglang_agent.py` |
| 进程后端 | `workers/inference/sglang/sglang_backend.py` |
| 启动 / 停止脚本 | `workers/inference/sglang/start_sglang.sh` / `stop_sglang.sh` |
| SGLang Router | `controller/rollout/inference/router/sglang/sglang_router_{manager,agent}.py` |
| 请求组装 | `utils/rollout/chat_completion.py` / `controller/rollout/rollout_worker.py` |
| 权重传输 | `workers/model/trainer/ppo_trainer.py` / `utils/train/comm_utils.py` |
| PD 拓扑 | `utils/core/node_selector.py` |
| R3 路由回放 | `utils/train/routing_manager.py` / `utils/train/routing_patch.py` |
| 配置 | `core/config/args.py` |
| PD recipe 示例 | `recipe/fully_async/deepseek_v4-32nodes-gbzz2-swe_cc-fully_async-256k-pd.yaml` |
