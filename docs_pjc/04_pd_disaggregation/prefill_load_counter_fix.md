# 修复：PD Streaming 模式下 Prefill Worker load_counter 减 1 时机过晚

## 背景与问题

在 PD 分离 streaming 模式下，prefill worker 的 `WorkerLoadGuard` 与 decode worker 的 guard 被打包进同一个 `AttachedBody`，导致 prefill worker 的 `load_counter` 要等到**整个 decode SSE 流结束**（可能持续数分钟）才减 1，而实际上 prefill 在 KV transfer 完成后就已经空闲了。

**影响：**
- prefill worker 在整个 decode 阶段 `load_counter` 虚高
- `cache_aware` 均衡检查（`max - min > 32 AND max > 1.1 * min`）可能误判为不均衡，将后续请求路由到其他 prefill worker，破坏 prefix cache 亲和性
- 高并发下 prefix cache 命中率下降

**根因**（`pd_router.rs:1022-1025`）：

```rust
// prefill 早已完成，但 guard 在这里才创建，且生命周期绑定到 decode body
let guards = vec![
    WorkerLoadGuard::new(prefill, headers.as_ref()),  // 错误：prefill 已完成
    WorkerLoadGuard::new(decode, headers.as_ref()),
];
AttachedBody::wrap_response(response, guards)
```

---

## 修改方案

**将 prefill guard 的作用域限定在 `process_prefill_response` 调用的前后，KV transfer 完成即释放；只有 decode guard 进入 `AttachedBody`。**

不需要引入新类型或新方法，纯粹是作用域调整。

### 改动 1：`execute_dual_dispatch_internal` 中调整 guard 创建位置

**文件：** `sgl-model-gateway/src/routers/http/pd_router.rs`

```rust
// 改前（lines 580-583）：
let _prefill_guard =
    (!context.is_stream).then(|| WorkerLoadGuard::new(prefill.clone(), headers));
let _decode_guard =
    (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone(), headers));
```

```rust
// 改后：
// decode guard 仅在非 streaming 时创建（函数返回时 drop，即读完 decode body 后）
let _decode_guard =
    (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone(), headers));
// prefill guard 不在这里创建，改为在 process_prefill_response 调用处用显式 block 包裹
```

然后在 lines 669-691（`process_prefill_response` 调用处）用显式 block 包裹：

```rust
// lines 669-691 改造：prefill guard 作用域精确覆盖 KV transfer 阶段
let prefill_body = {
    let _prefill_guard = WorkerLoadGuard::new(prefill.clone(), headers);
    if context.return_logprob {
        match self.process_prefill_response(...).await { ... }
    } else {
        match self.process_prefill_response(...).await { ... }
    }
    // _prefill_guard 在此 drop —— streaming 和非 streaming 均在此处减 1
};
// create_streaming_response 在 prefill guard 已 drop 后才调用
```

### 改动 2：`create_streaming_response` 中移除 prefill guard

**文件：** `sgl-model-gateway/src/routers/http/pd_router.rs` lines 1022-1025

```rust
// 改前：
let guards = vec![
    WorkerLoadGuard::new(prefill, headers.as_ref()),
    WorkerLoadGuard::new(decode, headers.as_ref()),
];
```

```rust
// 改后：仅 decode guard 绑定到 body 生命周期
let guards = vec![
    WorkerLoadGuard::new(decode, headers.as_ref()),
];
```

同时更新函数签名，移除不再使用的 `prefill` 参数：

```rust
// 改前（line 925-934）：
fn create_streaming_response(
    ...
    prefill: Arc<dyn Worker>,
    decode: Arc<dyn Worker>,
) -> Response

// 改后：
fn create_streaming_response(
    ...
    decode: Arc<dyn Worker>,
) -> Response
```

以及调用处（line 708-716）同步移除 `prefill` 参数。

---

## 涉及文件

- `sgl-model-gateway/src/routers/http/pd_router.rs`
  - lines 580-583：移除 prefill guard 的 `(!context.is_stream).then(...)` 分支
  - lines 669-691：用显式 block 包裹 `process_prefill_response`，内部创建 prefill guard
  - lines 708-716：`create_streaming_response` 调用处移除 `prefill` 参数
  - lines 925-934：`create_streaming_response` 签名移除 `prefill` 参数
  - lines 1022-1025：`guards` vec 中移除 prefill guard

---

## 修改后的负载语义

| 阶段 | prefill load_counter | decode load_counter |
|---|---|---|
| 请求到达，开始 KV transfer | +1 | 0（streaming）/ +1（非 streaming） |
| `process_prefill_response` 返回（KV transfer 完成） | **-1（归零）** | 维持 |
| decode 生成 token 期间 | 0 | +1 |
| decode 流结束 / 客户端断开 | 0 | -1 |

---

## 验证

1. 编译：`cargo build -p sgl-model-gateway`
2. 运行现有测试，确认无回归
3. 手动验证：启动 PD 服务，发送 streaming 请求，观察 prefill worker 的 `running_requests` 指标在 KV transfer 完成后迅速归零，而非等待 decode 结束
4. 指标查看：`curl prefill:port/metrics | grep running_requests`
