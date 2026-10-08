# SGLang 启动参数配置方法详解

> 目标：把 SGLang 目前这套"启动参数 → 解析 → 发布 → 运行时读取"的配置机制讲清楚。
> 面向读者：第一次接触 `arg_groups/` 这套代码，想搞懂"我加一个启动参数要怎么加、它是怎么生效的、运行时代码怎么读到它"的同学。
>
> 涉及核心目录：`python/sglang/srt/arg_groups/`、`python/sglang/srt/server_args.py`、`python/sglang/srt/runtime_context.py`

---

## 0. 一句话总览

SGLang 的启动参数经历四个阶段：

```
命令行/字典  ──①声明&解析CLI──▶  ServerArgs(原始输入)  ──②resolve_once解析管线──▶  声明栈(decisions)
                                                                                       │
                                                    ③publish 投影                       ▼
运行时业务代码  ◀──④get_exec()/get_schedule()...读取──  namespace 配置袋(config bags)
```

四个关键设计原则（贯穿全文，先记住）：

1. **`ServerArgs` 只存"用户输入的原始值"，永远不被解析逻辑改写。**
2. **解析（resolution）不写字段，只"声明"决策**，决策写进一个叫"声明栈"（stash）的地方。
3. **业务代码不读 `ServerArgs` 字段做决策**，而是读 `publish` 后投影出来的"配置袋"（config bags）。
4. **字段声明按命名空间分散在 `arg_groups/fields/` 下**，一个类 = 一个命名空间；`ServerArgs` 是把这些类组装起来的。

---

## 1. 参数是怎么"声明"的（字段定义）

### 1.1 `A[T, ...]` 注解写法

不再是手写 `parser.add_argument(...)`，而是用带元数据的类型注解声明字段。核心工具在 [arg_utils.py](../../../python/sglang/srt/arg_groups/arg_utils.py)：

```python
from sglang.srt.arg_groups.arg_utils import A, Arg

class Model(msgspec.Struct):
    _NS_PATH = "model"                       # 本类所属命名空间

    # 最简形式：裸字符串就是 help 文本
    host: A[str, "The host of the HTTP server."] = "127.0.0.1"
    port: A[int, "The port of the HTTP server."] = 30000
    trust_remote_code: A[bool, "Whether to allow custom models."] = False

    # 需要更多元数据时用 Arg(...)
    model_path: A[str, Arg(help="Path to model weights.", aliases=["--model"])]
    load_format: A[str, Arg(help="...", choices=LOAD_FORMAT_CHOICES)] = "auto"
```

- `A` 就是 `typing.Annotated` 的别名。
- 裸字符串 `A[T, "help"]` 等价于 `A[T, Arg(help="help")]`。
- CLI flag 自动从字段名推导：`model_path` → `--model-path`（`_field_to_cli_name`）。

### 1.2 `Arg` 支持的元数据（[arg_utils.py:62](../../../python/sglang/srt/arg_groups/arg_utils.py#L62)）

| 字段 | 作用 |
|------|------|
| `help` | 帮助文本 |
| `choices` | 取值枚举 |
| `aliases` | 额外的 CLI 别名，如 `["--model"]` |
| `cli_name` | 自定义主 flag 名（默认由字段名推导） |
| `type_parser` | 自定义 `type=` 解析函数（如 `json_list_type`） |
| `nargs` / `const` / `action` / `action_kwargs` | 透传给 argparse |
| `required` | 是否必填 |
| `no_cli` | **True 时不注册 CLI**，只能 Python 侧注入 |
| `resolvable` | **True 时允许解析管线改写它的"决策值"**（白名单机制，见 §3） |
| `fallback` | "没人说话时"的兜底值——读取链的最底层（见 §4.3） |

### 1.3 `Derived`：不是输入、而是"推导出来"的字段

有些值不是用户能输入的，而是配置的纯函数。用 `Derived` 声明（[arg_utils.py:95](../../../python/sglang/srt/arg_groups/arg_utils.py#L95)）：

```python
class Model(msgspec.Struct):
    is_startup_weight_load_overlap = Derived(
        fn="sglang.srt.arg_groups.model_override_base.startup_weight_load_overlap_of",
        doc="Whether weight loading overlaps startup.",
    )
```

- `Derived` 字段**不是 dataclass 字段**，不进入 `ServerArgs` 记录，不跨进程序列化。
- `fn` 是一个点分路径，在 `publish` 时惰性调用一次，把结果作为普通配置袋叶子存下来。

---

## 2. 命名空间：字段按域分文件

### 2.1 一个类 = 一个命名空间

字段声明分散在 [arg_groups/fields/](../../../python/sglang/srt/arg_groups/fields/) 下，每个文件一个/多个 `msgspec.Struct`，每个类带一个 `_NS_PATH`：

| 文件 | 类 | 命名空间 |
|------|----|---------|
| `model.py` | `Model` | `model` |
| `parallel.py` | `Parallel` | `parallel` |
| `memory.py` | `Memory` | `memory` |
| `schedule.py` | `Schedule` | `schedule` |
| `serving.py` | `Serving` | `serving` |
| `spec.py` | `Spec` | `spec`（推测解码） |
| `disagg.py` | `Disagg` | `disagg`（PD 分离） |
| `lora.py` | `Lora` | `lora` |
| `mm.py` | `Mm` | 多模态 |
| `device.py` | `Device` | `device` |
| `observability.py` | `Observability` | 可观测性 |
| `exec_.py` | `ExecKernel/ExecMoe/ExecGraph/...` | `exec.*`（执行相关子命名空间） |

**"字段声明在哪个文件里，它就属于哪个命名空间"**——文件本身就是命名空间标记，不需要每个字段额外打标。（见 [fields/__init__.py](../../../python/sglang/srt/arg_groups/fields/__init__.py) 的 `namespace_of`）

### 2.2 `ServerArgs` 是"组装"出来的，不是继承

关键在 [server_args.py:563](../../../python/sglang/srt/server_args.py#L563)：

```python
_INPUT_NAMESPACES = [
    Model, ExecDeterministic, ExecDllm, ExecOffload, ExecOverlap,
    Lora, Disagg, Spec, ExecMoe, ExecComm, ExecGraph, ExecMamba,
    ExecKernel, Observability, Device, Parallel, Memory, Schedule,
    ExecFeatures, Mm, Serving,
]

_annotations, _defaults, _namespaces = collect_input_fields(_INPUT_NAMESPACES)
ServerArgs = msgspec.defstruct("ServerArgs", [...], dict=True)
```

要点：

- **只有"用户输入"的命名空间被组装进 `ServerArgs`**；`Derived` 字段声明在旁边但不会被收集。这就是为什么这里是一次 `collect_input_fields` 调用，而不是基类列表——"记录里放什么"是一个可读的调用，而不是散落在继承链里的隐性约定。
- **字段顺序由 `field_order.py` 里的 `POSITIONAL_FIELD_ORDER` 固定**，不是按命名空间排。因为 dataclass 的字段顺序 = 位置构造参数签名，随意重排会悄悄改变 `ServerArgs(model_path, tokenizer_path)` 第二个位置参数指向的字段。新字段只能追加到末尾。
- `ServerArgs` 是 `msgspec.Struct` 且 `dict=True`——`dict=True` 让它能额外携带"非配置"的簿记信息（输入快照、声明栈、解析标志、memo 缓存槽）。

> 📌 项目规则：**新数据容器用 `msgspec.Struct`，不要用 `@dataclass`**（见 `.claude/rules/no-dataclasses.md`）。

---

## 3. CLI 注册与解析入口

### 3.1 自动注册 CLI（[arg_utils.py:361](../../../python/sglang/srt/arg_groups/arg_utils.py#L361)）

`add_cli_args_from_dataclass(parser, ServerArgs)` 遍历所有带 `Arg` 注解的字段，自动 `parser.add_argument`：

- `Optional[X]` → 自动脱壳
- `Literal[...]` → 自动变成 `choices`
- `List[X]` → `nargs="+"`
- `bool` → `action="store_true"`
- 其余标量 → 按类型推导 `type=`

### 3.2 少数需要手写的特例（[server_args.py:392](../../../python/sglang/srt/server_args.py#L392) `add_cli_args`）

只有三类必须手写 `add_argument`：

1. **废弃 flag**：用 `argparse_actions.py` 里的 `Deprecated*Action` 重定向到新字段。
2. **动态 choices**：运行时才能算出来的枚举，如 `--reasoning-parser`、`--tool-call-parser`（来自插件注册表）、`--sampling-backend`（受环境变量影响）。
3. **`--config` 元参数**：本身不是字段，用来从 YAML 读取配置再合并。

### 3.3 从命令行到 `ServerArgs`（[server_args.py:738](../../../python/sglang/srt/server_args.py#L738) `prepare_server_args`）

```
argv
 └─ ServerArgs.add_cli_args(parser)          # 注册所有 flag
 └─ 若含 --config：ConfigArgumentMerger 合并 YAML
 └─ parser.parse_args(argv)                  # 得到 Namespace
 └─ ServerArgs.from_cli_args(raw_args)       # Namespace → ServerArgs(原始输入)
 └─ server_args._launch_command = " ".join(argv)   # 记录原始命令(非字段)
```

此时得到的 `ServerArgs` **只是"用户输入的原始值"，还没解析**。

---

## 4. 解析管线（resolution）——核心中的核心

### 4.1 入口：`resolve_once`（[server_args.py:298](../../../python/sglang/srt/server_args.py#L298)）

```python
def resolve_once(self):
    if self._resolution_finished:  # 已解析过就跳过
        return
    if self._resolution_failed:    # 上次失败过则报错(不能二次解析)
        raise RuntimeError(...)
    self._input_frozen = True      # 解析期间"封印"记录：任何字段写入都会报错
    try:
        run_resolution_pipeline(self)
    except BaseException:
        self._resolution_failed = True
        raise
    finally:
        self._input_frozen = False
    self._resolution_finished = True
```

重要语义：

- **幂等保护**：解析器不是幂等的（例如 DP attention 每跑一次就把 `chunked_prefill_size` 减半），所以只能跑一次。
- **封印（`_input_frozen`）**：解析期间对字段的任何赋值都会在 `__setattr__`（[server_args.py:495](../../../python/sglang/srt/server_args.py#L495)）里抛异常。这从机制上强制"解析不写字段"。
- **谁来调**：每个"要发布配置的进程入口"都会调 `resolve_once`。子进程通过 pickle 收到已解析的记录（连声明栈一起），无需再解析。

### 4.2 管线本体：一个有序派发器（[pipeline.py:24](../../../python/sglang/srt/arg_groups/pipeline.py#L24) `run_resolution_pipeline`）

第一步先记录"原始输入快照"和保留 launcher 阶段的声明：

```python
server_args._raw_input = {f.name: getattr(server_args, f.name) for f in record_fields(...)}
server_args._resolved_overrides = list(getattr(server_args, "_resolved_overrides", ()))
```

然后按**依赖域顺序**调用一连串 handler（每步都通过 `run_hook(...)` 调用）：

```
内部/bootstrap → API/网络/协议校验 → dummy-model 短路
→ 模型源/路径解析 → 硬件/平台 → 模型专属调整 → 并行(TP/DP/CP/PP)
→ kernel/attention backend → CUDA graph → 内存/缓存 → MoE
→ 推测解码 → 加载格式 → 环境变量 → 各类校验
→ server_args._resolution_finished = True
```

派发器风格的几条纪律（写在 pipeline.py 顶部注释里）：

1. 本函数保持"有序派发器"——每步是一个具名调用，`import`/条件/改写/抛错都放进 handler 内部，不写在这里。
2. **dummy-model 边界尽量靠前**：`if cfg.model_path.lower() in ["none","dummy"]: return`，只有模型无关的 bootstrap 和网络/协议校验能在它之前跑。
3. handler 按**依赖域**排序，不按历史插入顺序。
4. 用通用的 handler 名字隐藏窄集成细节（派发器只说"处理哪个阶段"，不暴露某厂商/某 hook 的实现细节）。
5. 每个 handler 契约单一：它期望什么状态、能改什么、是否只做校验。
6. **每步都走 `run_hook(handle_x, server_args)`**，而不是直接 `handle_x(server_args)`——见 §4.4。

### 4.3 解析器怎么"声明决策"而不写字段

三种声明方式，最终都进同一个"声明栈"（`server_args._resolved_overrides`）：

**(a) 模型专属常量覆盖** —— [overrides.py](../../../python/sglang/srt/arg_groups/overrides.py) 里的 `MODEL_OVERRIDES`，键是 `hf_config.architectures[0]`，值是 `{field: value}`。

**(b) 模型专属派生逻辑** —— `@register_model_override(arch)` 装饰的函数 `fn(server_args, hf_config) -> dict`，返回声明字典。传入的 `server_args` 是只读的。（各模型在 [arg_groups/model_overrides/](../../../python/sglang/srt/arg_groups/model_overrides/) 下，如 `deepseek_v2.py`、`qwen3_moe.py`）

**(c) 后处理 pass** —— `run_post_process_pass(server_args, fn)`（[overrides.py:102](../../../python/sglang/srt/arg_groups/overrides.py#L102)）：在"解析视图"上求值，把返回的 dict 声明进栈。

底层统一入口是 `declare_resolution(server_args, source, **fields)`（[overrides.py:135](../../../python/sglang/srt/arg_groups/overrides.py#L135)）：

```python
stash = server_args._resolved_overrides
stash.append((source, dict(fields)))   # 追加 (来源, 决策字典)，后写者优先
```

- 非字段名会被拒绝（避免声明一个没人读的属性）。
- **拒绝在"已发布"的配置上声明**——发布后再声明是静默 no-op，必须走 `get_context().override(...)`（见 §5）。

**读取链的层次**（`resolution_result`，[overrides.py:238](../../../python/sglang/srt/arg_groups/overrides.py#L238)）：

```
① 声明栈里最后一条对该字段的决策（reversed 遍历，后写者优先）
② 否则：用户原始输入 _raw_input[field]
③ 否则：字段声明的 fallback（with_fallback）
```

`fallback` 是"没人说话时该字段的意思"，处在读取链最底层——用户输入的值、pass 决策的值都在它之上。

### 4.4 可覆盖 hook 机制（[resolution_hooks.py](../../../python/sglang/srt/arg_groups/resolution_hooks.py)）

为什么每步都走 `run_hook`？因为 `run_resolution_pipeline` 把步骤名硬编码、没有 `self` 派发，下游包（out-of-tree 插件）没法通过继承来改某一步的行为。`run_hook` 提供了这个能力：

```python
def run_hook(builtin, server_args):
    step = builtin
    for fn in _HOOKS.get(builtin.__name__, ()):   # 按注册顺序层层包裹
        step = _bind(fn, step)
    step(server_args)
```

- `@register_resolution_hook(name)` 注册一个覆盖函数 `fn(server_args, previous)`，`previous` 就是被它包裹的原步骤（类似 `super()`，但用显式参数表达）。
- **只允许白名单里的名字**（`_OVERRIDABLE_HOOKS`，一大串 `handle_*`/`validate_*`）。名字不在白名单就在 import 时大声报错，而不是在三个模块外的调用点静默失败。
- 覆盖只改**"某步跑什么"，从不改"何时跑"**——调用点在 pipeline.py 里位置不变。

---

## 5. 发布（publish）与运行时读取

解析完的 `ServerArgs` 依然只带"原始输入 + 声明栈"。真正让业务代码读到"生效值"的是 `publish`。

### 5.1 `publish` 投影出配置袋（[runtime_context.py:1731](../../../python/sglang/srt/runtime_context.py#L1731)）

- 每个"发布型进程入口"（scheduler、tokenizer、detokenizer、DP controller、encoder worker、benchmark 等）在进程启动时调 `publish(server_args, role=...)`。
- `publish` 把"声明栈叠加在原始字段之上"投影成一组 **namespace 配置袋（config bags）**；`Derived` 字段在此时计算并存为普通叶子。
- 构造函数不发布——`ModelRunner`/`TokenizerManager`/`MMEncoder` 只调 `assert_published`，若入口忘了发布就大声报错。
- 重复发布是 **last-publish-wins**（如 launcher 进程里 tokenizer 的发布、单测里顺序重建 engine）。

### 5.2 运行时怎么读（这是最常用的部分）

| 场景 | 正确读法 |
|------|---------|
| 读某命名空间的生效配置 | `get_exec().moe.moe_a2a_backend`、`get_schedule().max_running_requests`、`get_model().model_path` |
| 发布后改配置（唯一入口） | `get_context().override(source, **fields)`——只改配置袋叶子，**不回写** `ServerArgs` |
| 按字段名读叶子（name-driven 代码） | `get_context().config_leaf(name)` |
| 发布后控制面变更（权重更新、解析器识别等） | `TokenizerManager.record_config_updates(source, **fields)`（对 `override` 的具名封装） |
| 读"用户输入的原始命令/记录"（调试/provenance） | `get_server_args()`——**业务代码不要读它的字段做决策** |

**为什么业务代码不能读 `get_server_args()` 的字段？** 因为记录里存的是"用户输入的原始值"，不是"解析后决定的值"。读它会拿到用户输入（可能是 `None`/`auto`），而不是解析结果。运行时有一个"read ratchet"把这类字段读取数钉在零。

配置袋叶子是**真实实例属性**（dynamo-traceable），可以安全地在 `torch.compile` 追踪的代码里读。

### 5.3 `RuntimeContext` 分层（[runtime_context.py:1128](../../../python/sglang/srt/runtime_context.py#L1128)）

| 层 | 访问器 | 内容 |
|----|--------|------|
| 原始配置种子 | `get_server_args()` | 已发布的 `ServerArgs`，仅供调试/dump/provenance |
| 解析后配置 | `get_exec()/get_memory()/get_schedule()/get_model()/get_spec()/get_serving()/get_observability()/get_disagg()/get_lora()/get_mm()/get_device()` | 命名空间配置袋，**唯一真相源** |
| 运行时 flags | `get_flags()` | 非配置纯函数的状态（cuda-graph 生命周期、MoE 活跃后端、DP flags） |
| 资源 | `get_resources()/get_stream()/get_buffer()` | graph pool、EPLB、EP dispatcher、side stream、workspace buffer |
| per-forward | `get_forward()` | forward 作用域 flags（contextvar 支撑，`scoped(**kw)` 退出恢复） |
| 并行 | `get_parallel()` | rank/group 是活拓扑（read-through property），size 等是配置袋叶子 |

---

## 6. 跨进程与"每个 runner 私有值"的处理

- **`ServerArgs` 通过 pickle 跨进程**（[server_args.py:529](../../../python/sglang/srt/server_args.py#L529) `__reduce__`）：不仅带字段，还带 `dict=True` 命名空间里的输入快照、声明栈、解析标志。这样子进程能"发布父进程决定的东西"而无需重新解析。
- **配置袋不跨进程**：子进程从收到的对象重新投影自己的配置袋，所以父进程侧的 `override` 会丢失。
- **每个 runner 私有的值不放配置袋**：如 draft worker 的 `context_length`、`attention_backend`，通过**构造函数参数**（`ModelRunner(draft_attention_backend=...)`）或 **runner 属性**（`model_runner.kv_cache_dtype_str`）传递，而不是塞进共享配置对象。放进配置袋会让第二个 runner 继承第一个的答案（这正是历史上的 bug）。

---

## 7. 我要加/改一个启动参数，该怎么做？（实操清单）

### 加一个普通参数

1. 找到对应命名空间文件（如内存相关 → [fields/memory.py](../../../python/sglang/srt/arg_groups/fields/memory.py)），在合适的注释区块加字段：
   ```python
   my_new_flag: A[int, "What this flag does."] = 0
   ```
2. **不需要**手动写 `add_argument`——`add_cli_args_from_dataclass` 自动注册 `--my-new-flag`。
3. 运行时读取用 `get_<namespace>().my_new_flag`，**不要**读 `get_server_args().my_new_flag`。

### 需要根据模型/硬件调整默认值

- 不要在字段默认值里做——那是"用户输入的默认"。
- 在解析管线里加一个 handler，或用 `MODEL_OVERRIDES` / `@register_model_override`，通过 `declare_resolution` 声明决策。
- 若值是"配置的纯函数"（不依赖机器状态），用 `Derived` 字段声明。

### 需要发布后动态改（权重更新等）

- 用 `get_context().override(source, **fields)` 或 `TokenizerManager.record_config_updates(...)`。

### 环境变量相关

- 走 [environ.py](../../../python/sglang/srt/environ.py)，遵守 `env-var-conventions` skill 的约定。

---

## 8. 关键文件速查表

| 文件 | 职责 |
|------|------|
| [arg_groups/arg_utils.py](../../../python/sglang/srt/arg_groups/arg_utils.py) | `A`/`Arg`/`Derived`/`NS` 定义；`add_cli_args_from_dataclass` 自动注册 CLI；`resolution_result` 用到的 `with_fallback` |
| [arg_groups/fields/](../../../python/sglang/srt/arg_groups/fields/) | 各命名空间的字段声明（一类一命名空间） |
| [arg_groups/fields/__init__.py](../../../python/sglang/srt/arg_groups/fields/__init__.py) | `collect_input_fields` 组装字段 |
| [arg_groups/field_order.py](../../../python/sglang/srt/arg_groups/field_order.py) | `POSITIONAL_FIELD_ORDER` 固定字段位置顺序 |
| [server_args.py](../../../python/sglang/srt/server_args.py) | `ServerArgs` 组装、`resolve_once`、`add_cli_args`、`prepare_server_args`、`PortArgs` |
| [arg_groups/pipeline.py](../../../python/sglang/srt/arg_groups/pipeline.py) | `run_resolution_pipeline` 有序派发器 |
| [arg_groups/resolution_hooks.py](../../../python/sglang/srt/arg_groups/resolution_hooks.py) | `run_hook`/`register_resolution_hook` 可覆盖机制 + 白名单 |
| [arg_groups/overrides.py](../../../python/sglang/srt/arg_groups/overrides.py) | `declare_resolution`/`resolution_result`/`run_post_process_pass`/模型覆盖注册表 |
| [arg_groups/*_hook.py](../../../python/sglang/srt/arg_groups/) | 各域的 handler（parallel/memory/moe/speculative/cuda_graph/...） |
| [arg_groups/model_overrides/](../../../python/sglang/srt/arg_groups/model_overrides/) | 各模型的专属覆盖声明 |
| [runtime_context.py](../../../python/sglang/srt/runtime_context.py) | `publish`/`get_context`/`get_<ns>()` 配置袋 + 运行时分层 |

---

## 9. 心智模型总结（记住这三张图）

**数据流：**
```
用户输入 ──parse──▶ ServerArgs(原始, 只读)
                      │  resolve_once
                      ▼
                   声明栈 stash  ← handler/模型覆盖/后处理 都往这里"声明"，从不写字段
                      │  publish 投影(叠加声明+计算 Derived)
                      ▼
                   配置袋 bags   ← 运行时唯一真相源
                      │  发布后要改？只能 get_context().override()
                      ▼
              get_exec()/get_schedule()/...  ← 业务代码从这里读
```

**三条铁律：**
1. `ServerArgs` = 用户输入，只读，永不被解析改写。
2. 解析只"声明"决策进声明栈，不写字段。
3. 业务代码读配置袋（`get_<ns>()`），不读 `get_server_args()` 字段。
