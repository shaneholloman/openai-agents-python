---
search:
  exclude: true
---
# 智能体运行

你可以通过 [`Runner`][agents.run.Runner] 类运行智能体。你有 3 种选择：

1. [`Runner.run()`][agents.run.Runner.run]：异步运行并返回 [`RunResult`][agents.result.RunResult]。
2. [`Runner.run_sync()`][agents.run.Runner.run_sync]：同步方法，其内部只是运行 `.run()`。
3. [`Runner.run_streamed()`][agents.run.Runner.run_streamed]：异步运行并返回 [`RunResultStreaming`][agents.result.RunResultStreaming]。它以流式传输模式调用 LLM，并在收到事件时将其流式传输给你。

```python
from agents import Agent, Runner

async def main():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant")

    result = await Runner.run(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)
    # Code within the code,
    # Functions calling themselves,
    # Infinite loop's dance
```

请在[结果指南](results.md)中了解更多信息。

## Runner 生命周期与配置

### 智能体循环

使用 `Runner` 中的运行方法时，你需要传入一个起始智能体和输入。输入可以是：

-   字符串（视为用户消息），
-   OpenAI Responses API 格式的输入项列表，或
-   恢复中断的运行时使用的 [`RunState`][agents.run_state.RunState]。

随后，运行器会执行一个循环：

1. 我们使用当前输入为当前智能体调用 LLM。
2. LLM 生成输出。
    1. 如果 LLM 返回 `final_output`，循环结束并返回结果。
    2. 如果 LLM 执行任务转移，我们会更新当前智能体和输入，然后重新运行循环。
    3. 如果 LLM 生成工具调用，我们会运行这些工具调用、追加结果，然后重新运行循环。
3. 如果超过传入的 `max_turns`，我们会引发 [`MaxTurnsExceeded`][agents.exceptions.MaxTurnsExceeded] 异常。传入 `max_turns=None` 可禁用此轮次限制。

!!! note

    判断 LLM 输出是否被视为“最终输出”的规则是：它生成了所需类型的文本输出，并且没有工具调用。

### 流式传输

流式传输允许你在 LLM 运行时额外接收流式传输事件。流式传输结束后，[`RunResultStreaming`][agents.result.RunResultStreaming] 将包含此次运行的完整信息，包括生成的所有新输出。你可以调用 `.stream_events()` 获取流式传输事件。请在[流式传输指南](streaming.md)中了解更多信息。

#### Responses WebSocket 传输（可选辅助工具）

如果启用 OpenAI Responses websocket 传输，你仍然可以继续使用常规的 `Runner` API。建议使用 websocket 会话辅助工具来复用连接，但这不是必需的。

这是基于 websocket 传输的 Responses API，而不是 [Realtime API](realtime/guide.md)。

有关传输选择规则以及具体模型对象或自定义提供商的注意事项，请参阅[模型](models/index.md#responses-websocket-transport)。

##### 模式 1：不使用会话辅助工具（可行）

如果你只需要 websocket 传输，而不需要 SDK 为你管理共享提供商或会话，请使用此模式。

```python
import asyncio

from agents import Agent, Runner, set_default_openai_responses_transport


async def main():
    set_default_openai_responses_transport("websocket")

    agent = Agent(name="Assistant", instructions="Be concise.")
    result = Runner.run_streamed(agent, "Summarize recursion in one sentence.")

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            continue
        print(event.type)


asyncio.run(main())
```

此模式适用于单次运行。如果反复调用 `Runner.run()` / `Runner.run_streamed()`，除非手动复用同一个 `RunConfig` / 提供商实例，否则每次运行都可能重新连接。

##### 模式 2：使用 `responses_websocket_session()`（建议用于多轮复用）

如果希望在多次运行之间共享支持 websocket 的提供商和 `RunConfig`（包括继承同一 `run_config` 的嵌套智能体工具调用），请使用 [`responses_websocket_session()`][agents.responses_websocket_session]。

```python
import asyncio

from agents import Agent, responses_websocket_session


async def main():
    agent = Agent(name="Assistant", instructions="Be concise.")

    async with responses_websocket_session(
        responses_websocket_options={"ping_interval": 20.0, "ping_timeout": 60.0},
    ) as ws:
        first = ws.run_streamed(agent, "Say hello in one short sentence.")
        async for _event in first.stream_events():
            pass

        second = ws.run_streamed(
            agent,
            "Now say goodbye.",
            previous_response_id=first.last_response_id,
        )
        async for _event in second.stream_events():
            pass


asyncio.run(main())
```

请在退出上下文之前完成流式传输结果的消费。如果 websocket 请求仍在进行时退出上下文，可能会强制关闭共享连接。

服务会在每个 websocket 连接上一次处理一个响应，并将每个连接的时长限制为 60 分钟。该辅助工具会复用连接，但不会解除这些限制。重新连接后，`store=False` 和 ZDR 流程无法恢复未缓存的 `previous_response_id`；请使用完整输入上下文启动一条新链，或根据本地管理的会话状态重建该链。有关完整的恢复行为，请参阅 [Responses WebSocket 传输说明](models/index.md#responses-websocket-transport)。

如果长时间推理轮次触发 websocket keepalive 超时，请增大 `ping_timeout`，或设置 `ping_timeout=None` 以禁用心跳超时。对于可靠性比 websocket 延迟更重要的运行，请使用 HTTP/SSE 传输。

### 运行配置

`run_config` 参数可用于配置智能体运行的一些全局设置：

#### 常用运行配置目录

使用 `RunConfig` 可覆盖单次运行的行为，而无需更改每个智能体的定义。

##### 模型、提供商和会话默认值

-   [`model`][agents.run.RunConfig.model]：允许设置要使用的全局 LLM 模型，而不考虑每个 Agent 所设置的 `model`。
-   [`model_provider`][agents.run.RunConfig.model_provider]：用于按名称查找模型的模型提供商，默认为 OpenAI。
-   [`model_settings`][agents.run.RunConfig.model_settings]：覆盖智能体特定的设置。例如，你可以设置全局 `temperature` 或 `top_p`。
-   [`session_settings`][agents.run.RunConfig.session_settings]：在运行期间检索历史记录时，覆盖会话级默认值（例如 `SessionSettings(limit=...)`）。
-   [`session_input_callback`][agents.run.RunConfig.session_input_callback]：使用 Sessions 时，自定义每轮开始前将新用户输入与会话历史记录合并的方式。该回调可以是同步或异步的。

##### 安全防护措施、任务转移和模型输入塑形

-   [`input_guardrails`][agents.run.RunConfig.input_guardrails]、[`output_guardrails`][agents.run.RunConfig.output_guardrails]：要在所有运行中包含的输入或输出安全防护措施列表。
-   [`handoff_input_filter`][agents.run.RunConfig.handoff_input_filter]：如果任务转移尚未设置输入过滤器，则应用于所有任务转移的全局输入过滤器。输入过滤器允许你编辑发送给新智能体的输入。有关更多详情，请参阅 [`Handoff.input_filter`][agents.handoffs.Handoff.input_filter] 的文档。
-   [`nest_handoff_history`][agents.run.RunConfig.nest_handoff_history]：一项可选启用的测试版功能，在调用下一个智能体之前，将可总结的历史记录压缩为有序的助手摘要片段，同时在原始位置无损保留消息项。在我们完善嵌套任务转移期间，此功能默认禁用；设置为 `True` 可启用，保留为 `False` 则会直接传递原始记录。当 SDK 默认的嵌套历史记录已包含某条消息时，Sessions、`RunState` 和 `RunResult.to_input_list()` 会避免再次追加完全相同的消息实例，同时仍会保留彼此独立但内容相同的消息。如果你未传入 `RunConfig`，所有 [Runner 方法][agents.run.Runner]都会自动创建一个，因此快速入门和代码示例会保持默认关闭状态，而任何显式的 [`Handoff.input_filter`][agents.handoffs.Handoff.input_filter] 回调仍会覆盖此设置。各项任务转移可通过 [`Handoff.nest_handoff_history`][agents.handoffs.Handoff.nest_handoff_history] 覆盖此设置。
-   [`handoff_history_mapper`][agents.run.RunConfig.handoff_history_mapper]：一个可选的可调用对象，在你选择启用 `nest_handoff_history` 时接收规范化记录（历史记录 + 任务转移项）。它必须返回要转发给下一个智能体的准确输入项列表，从而替换内置的有序摘要片段，而无需编写完整的任务转移过滤器。
-   [`call_model_input_filter`][agents.run.RunConfig.call_model_input_filter]：用于在调用模型前立即编辑已完全准备好的模型输入（instructions 和输入项）的钩子，例如裁剪历史记录或注入系统提示词。
-   [`reasoning_item_id_policy`][agents.run.RunConfig.reasoning_item_id_policy]：控制运行器将先前输出转换为下一轮模型输入时，是保留还是省略推理项 ID。

##### 追踪与可观测性

-   [`tracing_disabled`][agents.run.RunConfig.tracing_disabled]：允许为整个运行禁用[追踪](tracing.md)。
-   [`tracing`][agents.run.RunConfig.tracing]：传入 [`TracingConfig`][agents.tracing.TracingConfig]，以覆盖追踪导出设置，例如每次运行使用的追踪 API 密钥。
-   [`trace_include_sensitive_data`][agents.run.RunConfig.trace_include_sensitive_data]：配置追踪是否包含潜在的敏感数据，例如 LLM 和工具调用的输入/输出。
-   [`workflow_name`][agents.run.RunConfig.workflow_name]、[`trace_id`][agents.run.RunConfig.trace_id]、[`group_id`][agents.run.RunConfig.group_id]：设置此次运行的追踪工作流名称、追踪 ID 和追踪组 ID。我们建议至少设置 `workflow_name`。组 ID 是一个可选字段，可用于关联多次运行的追踪。
-   [`trace_metadata`][agents.run.RunConfig.trace_metadata]：要包含在所有追踪中的元数据。

##### 工具执行、审批与工具错误行为

-   [`tool_execution`][agents.run.RunConfig.tool_execution]：配置本地工具调用在 SDK 端的执行行为，例如限制同时运行的工具调用数量。
-   [`tool_not_found_behavior`][agents.run.RunConfig.tool_not_found_behavior]：配置运行器如何处理模型发出的、无法解析的工具调用。默认行为是引发 `ModelBehaviorError`；也可以选择改为返回模型可见的错误输出。
-   [`tool_name_collision_policy`][agents.run.RunConfig.tool_name_collision_policy]：配置运行器如何处理发生冲突的无命名空间工具调用名称和任务转移名称。默认值 `"warn"` 会记录一条可操作的警告，并且只公开当前分派的胜出项；`"error"` 会在调用模型之前引发 `UserError`。对具有命名空间和延迟加载工具的严格验证保持不变。
-   [`tool_error_formatter`][agents.run.RunConfig.tool_error_formatter]：自定义模型可见的工具错误消息，例如审批被拒和选择启用的工具未找到输出。

嵌套任务转移是一项可选启用的测试版功能。传入 `RunConfig(nest_handoff_history=True)` 可启用有序记录压缩，或设置 `handoff(..., nest_handoff_history=True)` 为特定任务转移启用此功能。内置映射器会将生成的助手摘要片段放置在无损消息项周围，而不是将整个记录合并为一条消息。如果希望保留原始记录（默认行为），请不要设置此标志，或提供按需准确转发对话的 `handoff_input_filter`（或 `handoff_history_mapper`）。如果希望更改生成的摘要片段所使用的包装文本，而不编写自定义映射器，请调用 [`set_conversation_history_wrappers`][agents.handoffs.set_conversation_history_wrappers]（并调用 [`reset_conversation_history_wrappers`][agents.handoffs.reset_conversation_history_wrappers] 恢复默认值）。

#### 运行配置详情

##### `tool_execution`

如果希望配置本地工具调用在 SDK 端的行为，例如限制一次运行中本地工具调用的并发数量，请使用 `tool_execution`。

```python
from agents import Agent, RunConfig, Runner, ToolExecutionConfig

agent = Agent(name="Assistant", tools=[...])

result = await Runner.run(
    agent,
    "Run the required tool calls.",
    run_config=RunConfig(
        tool_execution=ToolExecutionConfig(
            max_function_tool_concurrency=2,
            pre_approval_tool_input_guardrails=True,
        ),
    ),
)
```

`max_function_tool_concurrency=None` 会保留默认行为：当模型在一轮中发出多个工具调用时，SDK 会启动所有已发出的本地工具调用。将其设置为整数值，可限制同时运行的本地工具调用数量。

这与提供商端的 [`ModelSettings.parallel_tool_calls`][agents.model_settings.ModelSettings.parallel_tool_calls] 不同。`parallel_tool_calls` 控制是否允许模型在单个响应中发出多个工具调用。`tool_execution.max_function_tool_concurrency` 控制模型发出本地工具调用后，SDK 如何执行这些调用。

`pre_approval_tool_input_guardrails=False` 会保留默认审批流程：如果工具调用需要审批，运行会先暂停，并且工具输入安全防护措施仅在审批后、紧接执行前运行。如果希望工具调用输入安全防护措施在发出待审批中断前运行，请将其设置为 `True`。通过此审批前检查的调用仍会在审批后再次运行相同的输入安全防护措施，因此执行前会重新验证时效性检查。

##### `tool_not_found_behavior`

默认情况下，如果模型发出的工具调用与当前智能体可用的任何工具调用都不匹配，运行器会引发 `ModelBehaviorError`。

如果希望运行仍可恢复，请设置 `tool_not_found_behavior="return_error_to_model"`。在该模式下，SDK 会为无法解析的工具调用追加一个 `function_call_output`，然后再次运行模型，以便模型选择可用工具，或在不使用该工具的情况下作答。

```python
from agents import Agent, RunConfig, Runner

agent = Agent(name="Assistant", tools=[...])

result = await Runner.run(
    agent,
    "Handle this request with the available tools.",
    run_config=RunConfig(tool_not_found_behavior="return_error_to_model"),
)
```

此选项目前仅适用于无法解析的工具调用。其他无效工具有效负载仍沿用现有的错误处理行为。

##### `tool_error_formatter`

使用 `tool_error_formatter` 可自定义 SDK 创建模型可见的工具错误输出时返回给模型的消息。

格式化器接收 [`ToolErrorFormatterArgs`][agents.run_config.ToolErrorFormatterArgs]，其中包含：

-   `kind`：错误目录，例如 `"approval_rejected"` 或 `"tool_not_found"`。
-   `tool_type`：工具运行时（`"function"`、`"computer"`、`"shell"`、`"apply_patch"` 或 `"custom"`）。
-   `tool_name`：工具名称。
-   `call_id`：工具调用 ID。
-   `default_message`：SDK 默认的模型可见消息。
-   `run_context`：当前运行上下文包装器。

返回字符串可替换该消息；返回 `None` 则使用 SDK 默认值。

```python
from agents import Agent, RunConfig, Runner, ToolErrorFormatterArgs


def format_rejection(args: ToolErrorFormatterArgs[None]) -> str | None:
    if args.kind == "approval_rejected":
        return (
            f"Tool call '{args.tool_name}' was rejected by a human reviewer. "
            "Ask for confirmation or propose a safer alternative."
        )
    if args.kind == "tool_not_found":
        return f"Tool '{args.tool_name}' is not available. Choose one of the listed tools."
    return None


agent = Agent(name="Assistant")
result = Runner.run_sync(
    agent,
    "Please delete the production database.",
    run_config=RunConfig(tool_error_formatter=format_rejection),
)
```

##### `reasoning_item_id_policy`

`reasoning_item_id_policy` 控制运行器向后传递历史记录时，如何将推理项转换为下一轮模型输入（例如使用 `RunResult.to_input_list()` 或由会话支持的运行时）。

-   `None` 或 `"preserve"`（默认）：保留推理项 ID。
-   `"omit"`：从生成的下一轮输入中移除推理项 ID。

`"omit"` 主要用作一种可选启用的缓解措施，用于处理一类 Responses API 400 错误：发送的推理项包含 `id`，但缺少所需的后续项（例如 `Item 'rs_...' of type 'reasoning' was provided without its required following item.`）。

这种情况可能发生在多轮智能体运行中：SDK 根据先前输出构建后续输入（包括会话持久化、服务端管理的对话增量、流式传输/非流式传输的后续轮次以及恢复路径），并保留了推理项 ID，但提供商要求该 ID 必须与其对应的后续项保持配对。

设置 `reasoning_item_id_policy="omit"` 会保留推理内容，但移除推理项的 `id`，从而避免 SDK 生成的后续输入触发该 API 不变量。

作用范围说明：

-   这只会更改 SDK 构建后续输入时生成或转发的推理项。
-   它不会重写用户提供的初始输入项。
-   应用此策略后，`call_model_input_filter` 仍可有意重新引入推理 ID。

## 状态与对话管理

### 记忆策略选择

将状态带入下一轮通常有四种方式：

| 策略 | 状态存储位置 | 最适用场景 | 下一轮传入内容 |
| --- | --- | --- | --- |
| `result.to_input_list()` | 应用内存 | 小型聊天循环、完全手动控制、任何提供商 | `result.to_input_list()` 返回的列表加上下一条用户消息 |
| `session` | 你的存储加上 SDK | 持久化聊天状态、可恢复运行、自定义存储 | 同一个 `session` 实例，或指向同一存储的另一个实例 |
| `conversation_id` | OpenAI Conversations API | 希望在多个工作进程或服务之间共享的具名服务端对话 | 同一个 `conversation_id`，并且只传入新的用户轮次 |
| `previous_response_id` | OpenAI Responses API | 无需创建对话资源的轻量级服务端管理续接 | `result.last_response_id`，并且只传入新的用户轮次 |

`result.to_input_list()` 和 `session` 由客户端管理。`conversation_id` 和 `previous_response_id` 由 OpenAI 管理，并且仅在使用 OpenAI Responses API 时适用。在大多数应用中，每个对话应选择一种持久化策略。除非你有意协调这两个层级，否则混合使用客户端管理的历史记录与 OpenAI 管理的状态可能导致上下文重复。

!!! note

    会话持久化不能在同一次运行中与服务端管理的对话设置
    （`conversation_id`、`previous_response_id` 或 `auto_previous_response_id`）
    结合使用。每次调用请选择一种方式。

### 对话/聊天线程

调用任何运行方法都可能导致一个或多个智能体运行（因而产生一次或多次 LLM 调用），但这在聊天对话中只代表一个逻辑轮次。例如：

1. 用户轮次：用户输入文本
2. 运行器运行：第一个智能体调用 LLM、运行工具、将任务转移给第二个智能体；第二个智能体运行更多工具，然后生成输出。

智能体运行结束时，你可以选择向用户显示哪些内容。例如，可以向用户显示智能体生成的每个新项目，也可以只显示最终输出。无论采用哪种方式，用户之后都可能提出后续问题，此时你可以再次调用运行方法。

#### 手动对话管理

你可以使用 [`RunResultBase.to_input_list()`][agents.result.RunResultBase.to_input_list] 方法手动管理对话历史记录，以获取下一轮的输入：

```python
from agents import Agent, Runner, trace

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    thread_id = "thread_123"  # Example thread ID
    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result = await Runner.run(agent, "What city is the Golden Gate Bridge in?")
        print(result.final_output)
        # San Francisco

        # Second turn
        new_input = result.to_input_list() + [{"role": "user", "content": "What state is it in?"}]
        result = await Runner.run(agent, new_input)
        print(result.final_output)
        # California
```

#### 使用会话自动管理对话

若要采用更简单的方式，可以使用 [Sessions](sessions/index.md) 自动处理对话历史记录，而无需手动调用 `.to_input_list()`：

```python
from agents import Agent, Runner, SQLiteSession, trace

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    # Create session instance
    session = SQLiteSession("conversation_123")

    thread_id = "thread_123"  # Example thread ID
    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result = await Runner.run(agent, "What city is the Golden Gate Bridge in?", session=session)
        print(result.final_output)
        # San Francisco

        # Second turn - agent automatically remembers previous context
        result = await Runner.run(agent, "What state is it in?", session=session)
        print(result.final_output)
        # California
```

Sessions 会自动：

-   在每次运行前检索对话历史记录
-   在每次运行后存储新消息
-   为不同的会话 ID 维护独立的对话

有关更多详情，请参阅 [Sessions 文档](sessions/index.md)。


#### 服务端管理的对话

你也可以让 OpenAI 对话状态功能在服务端管理对话状态，而不是使用 `to_input_list()` 或 `Sessions` 在本地处理。这让你无需手动重新发送所有历史消息，即可保留对话历史记录。使用以下任一服务端管理方式时，每次请求只需传入新轮次的输入并复用保存的 ID。有关更多详情，请参阅 [OpenAI 对话状态指南](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses)。

OpenAI 提供两种跨轮次追踪状态的方式：

##### 1. 使用 `conversation_id`

首先使用 OpenAI Conversations API 创建对话，然后在后续每次调用中复用其 ID：

```python
from agents import Agent, Runner
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    # Create a server-managed conversation
    conversation = await client.conversations.create()
    conv_id = conversation.id

    while True:
        user_input = input("You: ")
        result = await Runner.run(agent, user_input, conversation_id=conv_id)
        print(f"Assistant: {result.final_output}")
```

##### 2. 使用 `previous_response_id`

另一种方式是**响应链式衔接**，其中每一轮都会显式链接到上一轮的响应 ID。

```python
from agents import Agent, Runner

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    previous_response_id = None

    while True:
        user_input = input("You: ")

        # Setting auto_previous_response_id=True enables response chaining automatically
        # for the first turn, even when there's no actual previous response ID yet.
        result = await Runner.run(
            agent,
            user_input,
            previous_response_id=previous_response_id,
            auto_previous_response_id=True,
        )
        previous_response_id = result.last_response_id
        print(f"Assistant: {result.final_output}")
```

如果运行因等待审批而暂停，并且你从 [`RunState`][agents.run_state.RunState] 恢复运行，SDK 会保留已保存的 `conversation_id` / `previous_response_id` / `auto_previous_response_id` 设置，使恢复后的轮次继续使用同一个服务端管理的对话。

`conversation_id` 和 `previous_response_id` 互斥。如果需要一个可跨系统共享的具名对话资源，请使用 `conversation_id`。如果希望使用最轻量的 Responses API 基本组件从一个轮次续接到下一轮，请使用 `previous_response_id`。

!!! note

    SDK 会自动以退避方式重试 `conversation_locked` 错误。在服务端管理的
    对话运行中，它会在重试前回退内部对话追踪器输入，以便
    清晰地重新发送同一批已准备好的项目。

    在基于本地会话的运行中（它不能与 `conversation_id`、
    `previous_response_id` 或 `auto_previous_response_id` 结合使用），SDK 还会尽最大努力
    回滚最近持久化的输入项，以减少重试后产生重复的历史记录条目。

    即使未配置 `ModelSettings.retry`，也会执行此兼容性重试。有关
    更广泛的可选模型请求重试行为，请参阅[由 Runner 管理的重试](models/index.md#runner-managed-retries)。

## 钩子与自定义

### 模型调用输入过滤器

使用 `call_model_input_filter` 可在模型调用前编辑模型输入。该钩子接收当前智能体、上下文以及合并后的输入项（包括会话历史记录，如有），并返回新的 `ModelInputData`。

返回值必须是 [`ModelInputData`][agents.run.ModelInputData] 对象。其 `input` 字段为必填项，并且必须是输入项列表。返回任何其他结构都会引发 `UserError`。

```python
from agents import Agent, Runner, RunConfig
from agents.run import CallModelData, ModelInputData

def drop_old_messages(data: CallModelData[None]) -> ModelInputData:
    # Keep only the last 5 items and preserve existing instructions.
    trimmed = data.model_data.input[-5:]
    return ModelInputData(input=trimmed, instructions=data.model_data.instructions)

agent = Agent(name="Assistant", instructions="Answer concisely.")
result = Runner.run_sync(
    agent,
    "Explain quines",
    run_config=RunConfig(call_model_input_filter=drop_old_messages),
)
```

运行器会将已准备好的输入列表副本传给钩子，因此你可以对其进行裁剪、替换或重新排序，而不会就地修改调用方的原始列表。

如果使用会话，`call_model_input_filter` 会在会话历史记录加载并与当前轮次合并后运行。如果希望自定义前面的合并步骤本身，请使用 [`session_input_callback`][agents.run.RunConfig.session_input_callback]。

如果通过 `conversation_id`、`previous_response_id` 或 `auto_previous_response_id` 使用 OpenAI 服务端管理的对话状态，该钩子会针对下一次 Responses API 调用已准备好的有效负载运行。该有效负载可能已经只表示新轮次的增量，而不是对先前完整历史记录的重放。只有你返回的项目会被标记为已发送，以用于该服务端管理的续接。

通过 `run_config` 为每次运行设置该钩子，可用于遮盖敏感数据、裁剪过长的历史记录或注入额外的系统指导信息。

## 错误与恢复

### 错误处理器

所有 `Runner` 入口点都接受 `error_handlers`，它是一个以错误类型为键的字典。支持的键包括 `"max_turns"`、`"model_refusal"` 和 `"invalid_final_output"`。如果希望返回受控的最终输出，而不是以对应错误结束运行，请使用这些键。

```python
from agents import (
    Agent,
    RunErrorHandlerInput,
    RunErrorHandlerResult,
    Runner,
)

agent = Agent(name="Assistant", instructions="Be concise.")


def on_max_turns(_data: RunErrorHandlerInput[None]) -> RunErrorHandlerResult:
    return RunErrorHandlerResult(
        final_output="I couldn't finish within the turn limit. Please narrow the request.",
        include_in_history=False,
    )


result = Runner.run_sync(
    agent,
    "Analyze this long transcript",
    max_turns=3,
    error_handlers={"max_turns": on_max_turns},
)
print(result.final_output)
```

当模型消息无法通过智能体结构化 `output_type` 的验证，或模型没有返回结构化最终消息时，请使用 `"invalid_final_output"`。处理器可以返回应用特定的回退值，SDK 会使用相同的 `output_type` 对其进行验证。它不会重试模型调用，也不会重放任何工具副作用。返回 `None` 表示拒绝恢复。如果没有回退值，非空验证失败仍会引发 `ModelBehaviorError`，而空结构化响应会保留现有的下一轮行为。

```python
from pydantic import BaseModel

from agents import Agent, ModelBehaviorError, RunErrorHandlerInput, Runner


class Recipe(BaseModel):
    ingredients: list[str]
    recovered_from_invalid_output: bool = False


def on_invalid_final_output(data: RunErrorHandlerInput[None]) -> Recipe:
    assert isinstance(data.error, ModelBehaviorError)
    return Recipe(ingredients=[], recovered_from_invalid_output=True)


agent = Agent(
    name="Recipe assistant",
    instructions="Return a structured recipe.",
    output_type=Recipe,
)

result = Runner.run_sync(
    agent,
    "Plan tonight's dinner.",
    error_handlers={"invalid_final_output": on_invalid_final_output},
)
print(result.final_output)
```

`RunErrorHandlerResult.include_in_history` 默认为 `True`。对于最大轮次处理器，这会将合成的回退输出追加到对话历史记录中，并将其持久化到已配置的会话。如果希望将回退值返回给调用方，但不将其添加到结果历史记录或会话存储中，请设置 `include_in_history=False`。

如果模型拒绝响应时应生成应用特定的回退值，而不是以 `ModelRefusalError` 结束运行，请使用 `"model_refusal"`。

```python
from pydantic import BaseModel

from agents import Agent, ModelRefusalError, RunErrorHandlerInput, Runner


class Recipe(BaseModel):
    ingredients: list[str]
    refusal_reason: str | None = None


def on_model_refusal(data: RunErrorHandlerInput[None]) -> Recipe:
    assert isinstance(data.error, ModelRefusalError)
    return Recipe(ingredients=[], refusal_reason=data.error.refusal)


agent = Agent(
    name="Recipe assistant",
    instructions="Return a structured recipe.",
    output_type=Recipe,
)

result = Runner.run_sync(
    agent,
    "Make me something unsafe.",
    error_handlers={"model_refusal": on_model_refusal},
)
print(result.final_output)
```

## 持久执行集成与人在回路

有关工具审批的暂停/恢复模式，请先参阅专门的[人在回路指南](human_in_the_loop.md)。以下集成适用于运行可能经历长时间等待、重试或进程重启的持久编排。

### Dapr

你可以使用 Agents SDK 的 [Dapr](https://dapr.io) Diagrid 集成，运行持久、长时间运行的智能体。这些智能体支持人在回路，并能自动从故障中恢复。Dapr 是一个厂商中立的 [CNCF](https://cncf.io) 工作流编排器。可从[此处](https://docs.diagrid.io/getting-started/quickstarts/ai-agents/?agentframework=openai)开始使用 Dapr 和 OpenAI 智能体。

### Temporal

你可以使用 Agents SDK 的 [Temporal](https://temporal.io/) 集成运行持久、长时间运行的工作流，包括人在回路任务。你可以在[此视频](https://www.youtube.com/watch?v=fFBZqzT4DD8)中查看 Temporal 与 Agents SDK 协同完成长时间运行任务的演示，并在[此处查看文档](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents)。

### Restate

你可以使用 Agents SDK 的 [Restate](https://restate.dev/) 集成实现轻量级持久智能体，包括人工审批、任务转移和会话管理。该集成依赖 Restate 的单二进制运行时，并支持将智能体作为进程/容器或无服务函数运行。有关更多详情，请阅读[概述](https://www.restate.dev/blog/durable-orchestration-for-ai-agents-with-restate-and-openai-sdk)或查看[文档](https://docs.restate.dev/ai)。

### DBOS

你可以使用 Agents SDK 的 [DBOS](https://dbos.dev/) 集成运行可靠的智能体，使其在故障和重启后仍能保留进度。它支持长时间运行的智能体、人在回路工作流和任务转移，同时支持同步和异步方法。该集成只需要一个 SQLite 或 Postgres 数据库。有关更多详情，请查看集成[仓库](https://github.com/dbos-inc/dbos-openai-agents)和[文档](https://docs.dbos.dev/integrations/openai-agents)。

## 异常

SDK 会在特定情况下引发异常。完整列表位于 [`agents.exceptions`][]。概述如下：

-   [`AgentsException`][agents.exceptions.AgentsException]：这是 SDK 内引发的所有异常的基类。它是一个通用类型，所有其他特定异常均派生自此类。
-   [`MaxTurnsExceeded`][agents.exceptions.MaxTurnsExceeded]：当智能体运行超过传给 `Runner.run`、`Runner.run_sync` 或 `Runner.run_streamed` 方法的 `max_turns` 限制时，会引发此异常。它表示智能体无法在指定的交互轮次数内完成任务。设置 `max_turns=None` 可禁用此限制。
-   [`ModelBehaviorError`][agents.exceptions.ModelBehaviorError]：当底层模型（LLM）生成意外或无效输出时，会发生此异常。这可能包括：
    -   格式错误的 JSON：模型为工具调用或直接输出提供了格式错误的 JSON 结构，尤其是在定义了特定 `output_type` 时。
    -   意外的工具相关故障：模型未能以预期方式使用工具
-   [`ToolTimeoutError`][agents.exceptions.ToolTimeoutError]：当工具调用超过其配置的超时时间，并且该工具使用 `timeout_behavior="raise_exception"` 时，会引发此异常。
-   [`UserError`][agents.exceptions.UserError]：当你（使用 SDK 编写代码的人）在使用 SDK 时出错，会引发此异常。这通常是由代码实现不正确、配置无效或误用 SDK API 导致的。
-   [`InputGuardrailTripwireTriggered`][agents.exceptions.InputGuardrailTripwireTriggered]、[`OutputGuardrailTripwireTriggered`][agents.exceptions.OutputGuardrailTripwireTriggered]：分别在满足输入安全防护措施或输出安全防护措施的条件时引发这些异常。输入安全防护措施会在处理前检查传入消息，而输出安全防护措施会在交付前检查智能体的最终响应。