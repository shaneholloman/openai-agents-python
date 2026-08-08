---
search:
  exclude: true
---
# 结果

调用 `Runner.run` 方法时，你会收到以下两种结果类型之一：

-   从 `Runner.run(...)` 或 `Runner.run_sync(...)` 获得的 [`RunResult`][agents.result.RunResult]
-   从 `Runner.run_streamed(...)` 获得的 [`RunResultStreaming`][agents.result.RunResultStreaming]

两者都继承自 [`RunResultBase`][agents.result.RunResultBase]，后者公开了共享的结果接口，例如 `final_output`、`new_items`、`last_agent`、`raw_responses` 和 `to_state()`。

`RunResultStreaming` 增加了流式传输专用控制功能，例如 [`stream_events()`][agents.result.RunResultStreaming.stream_events]、[`current_agent`][agents.result.RunResultStreaming.current_agent]、[`is_complete`][agents.result.RunResultStreaming.is_complete] 和 [`cancel(...)`][agents.result.RunResultStreaming.cancel]。

## 合适结果接口的选择

大多数应用只需要少量结果属性或辅助方法：

| 需求 | 使用 |
| --- | --- |
| 向用户显示最终答案 | `final_output` |
| 包含完整本地对话记录、可直接用于重放的下一轮输入列表 | `to_input_list()` |
| 包含智能体、工具、任务转移和审批元数据的丰富运行条目 | `new_items` |
| 通常应处理下一轮用户交互的智能体 | `last_agent` |
| 使用 `previous_response_id` 进行 OpenAI Responses API 链式调用 | `last_response_id` |
| 待处理的审批和可恢复快照 | `interruptions` 和 `to_state()` |
| 当前嵌套 `Agent.as_tool()` 调用的元数据 | `agent_tool_invocation` |
| 原始模型调用或安全防护措施诊断信息 | `raw_responses` 和安全防护措施结果数组 |

## 最终输出

[`final_output`][agents.result.RunResultBase.final_output] 属性包含最后一个运行的智能体所生成的最终输出。其类型可能是：

-   如果最后一个智能体未定义 `output_type`，则为 `str`
-   如果最后一个智能体定义了输出类型，则为 `last_agent.output_type` 类型的对象
-   如果运行在生成最终输出前停止，则为 `None`，例如运行因审批中断而暂停

!!! note

    `final_output` 的类型为 `Any`。任务转移可能会改变最终完成运行的智能体，因此 SDK 无法静态确定所有可能的输出类型。

在流式传输模式下，`final_output` 会保持为 `None`，直到流处理完毕。有关逐事件处理流程，请参阅[流式传输](streaming.md)。

## 输入、下一轮历史记录与新条目

这些接口分别用于回答不同的问题：

| 属性或辅助方法 | 包含的内容 | 最适合的用途 |
| --- | --- | --- |
| [`input`][agents.result.RunResultBase.input] | 此运行片段的基础输入。如果任务转移输入过滤器重写了历史记录，此属性会反映运行继续执行时所使用的过滤后输入。 | 审计此运行实际使用的输入 |
| [`to_input_list()`][agents.result.RunResultBase.to_input_list] | 运行的输入条目视图。默认的 `mode="preserve_all"` 会保留来自 `new_items` 的转换后历史记录，但不会再次追加已经移入 SDK 默认嵌套任务转移历史记录中的同一个会话条目实例；当任务转移过滤重写模型历史记录时，`mode="normalized"` 优先使用规范的续接输入。 | 手动聊天循环、由客户端管理的对话状态以及纯条目历史记录检查 |
| [`new_items`][agents.result.RunResultBase.new_items] | 包含智能体、工具、任务转移和审批元数据的丰富 [`RunItem`][agents.items.RunItem] 封装对象。 | 日志、UI、审计和调试 |
| [`raw_responses`][agents.result.RunResultBase.raw_responses] | 运行中每次模型调用产生的原始 [`ModelResponse`][agents.items.ModelResponse] 对象。 | 提供商级诊断或原始响应检查 |

在实践中：

-   当你需要运行的纯输入条目视图时，使用 `to_input_list()`。
-   在任务转移过滤或嵌套任务转移历史记录重写后，当你需要用于下一次 `Runner.run(..., input=...)` 调用的规范本地输入时，使用 `to_input_list(mode="normalized")`。
-   当你希望 SDK 为你加载和保存历史记录时，使用 [`session=...`](sessions/index.md)。
-   如果你正在通过 `conversation_id` 或 `previous_response_id` 使用由 OpenAI 服务器管理的状态，通常只需传入新的用户输入并复用存储的 ID，而无需重新发送 `to_input_list()`。
-   当你需要用于日志、UI 或审计的完整转换后历史记录时，使用默认的 `to_input_list()` 模式或 `new_items`。

当 SDK 默认的嵌套任务转移历史记录逐字保留消息条目时，会话、`RunState` 和 `to_input_list()` 会追踪实际归属的条目实例，而不是按内容去重。分别出现的相同消息仍会保持独立；系统只会避免再次追加已归属的条目实例。

与 JavaScript SDK 不同，Python 不会公开单独的 `output` 属性来仅包含运行期间新生成的模型格式条目。需要 SDK 元数据时，请使用 `new_items`；需要原始模型载荷时，请检查 `raw_responses`。

将计算机工具条目作为对话输入重新提交时，会使用原始 Responses 载荷格式。预览模型的 `computer_call` 条目会保留单个 `action`，而 `gpt-5.5` 计算机调用可以保留批量的 `actions[]`。[`to_input_list()`][agents.result.RunResultBase.to_input_list] 和 [`RunState`][agents.run_state.RunState] 会保留模型生成的格式，因此，无论是预览版还是正式发布版的计算机工具调用，手动将这些条目重新提交为对话输入、执行暂停/恢复流程以及使用已存储的对话记录都可以继续正常工作。本地执行结果仍会在 `new_items` 中显示为 `computer_call_output` 条目。

### 新条目

[`new_items`][agents.result.RunResultBase.new_items] 提供运行期间所发生事件的最丰富视图。常见条目类型包括：

-   用于助手消息的 [`MessageOutputItem`][agents.items.MessageOutputItem]
-   用于推理条目的 [`ReasoningItem`][agents.items.ReasoningItem]
-   用于 Responses 工具搜索请求和已加载工具搜索结果的 [`ToolSearchCallItem`][agents.items.ToolSearchCallItem] 和 [`ToolSearchOutputItem`][agents.items.ToolSearchOutputItem]
-   用于工具调用及其结果的 [`ToolCallItem`][agents.items.ToolCallItem] 和 [`ToolCallOutputItem`][agents.items.ToolCallOutputItem]
-   用于因等待审批而暂停的工具调用的 [`ToolApprovalItem`][agents.items.ToolApprovalItem]
-   用于托管 MCP 审批和工具目录的 [`MCPApprovalRequestItem`][agents.items.MCPApprovalRequestItem]、[`MCPApprovalResponseItem`][agents.items.MCPApprovalResponseItem] 和 [`MCPListToolsItem`][agents.items.MCPListToolsItem]
-   用于任务转移请求和已完成转移的 [`HandoffCallItem`][agents.items.HandoffCallItem] 和 [`HandoffOutputItem`][agents.items.HandoffOutputItem]

当你需要智能体关联信息、工具输出、任务转移边界或审批边界时，应选择 `new_items`，而不是 `to_input_list()`。

使用托管工具搜索时，检查 `ToolSearchCallItem.raw_item` 可查看模型发出的搜索请求，检查 `ToolSearchOutputItem.raw_item` 可查看该轮加载了哪些命名空间、函数或托管 MCP 服务器。

使用程序化工具调用时，生成的 `program` 是 `ToolCallItem`，归属于该程序的普通子工具调用也是 `ToolCallItem` 条目，而对应的 `program_output` 是 `ToolCallOutputItem`。归属于程序的托管 MCP `mcp_approval_request` 和 `mcp_list_tools` 条目属于例外：它们会成为 `MCPApprovalRequestItem` 和 `MCPListToolsItem` 条目。

原始条目可以是带类型的 Responses 对象或映射。特别是，归属于程序的 shell 和补丁应用调用会使用映射。请使用可安全处理映射的检查模式：

```python
from collections.abc import Mapping


def raw_field(item, name):
    raw_item = item.raw_item
    if isinstance(raw_item, Mapping):
        return raw_item.get(name)
    return getattr(raw_item, name, None)


raw_type = raw_field(item, "type")
caller = raw_field(item, "caller")
caller_id = (
    caller.get("caller_id")
    if isinstance(caller, Mapping)
    else getattr(caller, "caller_id", None)
)
```

对于归属于程序的子调用，`caller` 的 `type` 字段为 `program`，而 `caller_id` 用于标识父程序调用。

## 对话的继续或恢复

### 下一轮智能体

[`last_agent`][agents.result.RunResultBase.last_agent] 包含最后一个运行的智能体。在任务转移后，它通常是下一轮用户交互中最适合复用的智能体。

在流式传输模式下，[`RunResultStreaming.current_agent`][agents.result.RunResultStreaming.current_agent] 会随着运行推进而更新，因此你可以在流结束前观察任务转移。

### 中断与运行状态

如果工具需要审批，待处理的审批会公开在 [`RunResult.interruptions`][agents.result.RunResult.interruptions] 或 [`RunResultStreaming.interruptions`][agents.result.RunResultStreaming.interruptions] 中。其中可能包括由直接调用的工具、任务转移后调用的工具或嵌套 [`Agent.as_tool()`][agents.agent.Agent.as_tool] 运行触发的审批。

调用 [`to_state()`][agents.result.RunResult.to_state] 以捕获可恢复的 [`RunState`][agents.run_state.RunState]，批准或拒绝待处理条目，然后使用 `Runner.run(...)` 或 `Runner.run_streamed(...)` 恢复运行。

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="Use tools when needed.")
result = await Runner.run(agent, "Delete temp files that are no longer needed.")

if result.interruptions:
    state = result.to_state()
    for interruption in result.interruptions:
        state.approve(interruption)
    result = await Runner.run(agent, state)
```

对于流式传输运行，请先完成对 [`stream_events()`][agents.result.RunResultStreaming.stream_events] 的消费，然后检查 `result.interruptions` 并从 `result.to_state()` 恢复。有关完整审批流程，请参阅[人工介入](human_in_the_loop.md)。

### 服务器管理的续接

[`last_response_id`][agents.result.RunResultBase.last_response_id] 是此次运行中最新的模型响应 ID。若要继续 OpenAI Responses API 链，请在下一轮将其作为 `previous_response_id` 传回。

如果你已经使用 `to_input_list()`、`session` 或 `conversation_id` 继续对话，通常不需要 `last_response_id`。如果需要多步骤运行中的每个模型响应，请改为检查 `raw_responses`。

## 智能体工具元数据

当结果来自嵌套的 [`Agent.as_tool()`][agents.agent.Agent.as_tool] 运行时，[`agent_tool_invocation`][agents.result.RunResultBase.agent_tool_invocation] 会公开有关外层 `Agent.as_tool()` 调用的不可变元数据：

-   `tool_name`
-   `tool_call_id`
-   `tool_arguments`

对于普通的顶层运行，`agent_tool_invocation` 为 `None`。

这在 `custom_output_extractor` 内尤其有用，因为在对嵌套结果进行后处理时，你可能需要外层 `Agent.as_tool()` 调用的工具名称、调用 ID 或原始参数。有关相关的 `Agent.as_tool()` 模式，请参阅[工具](tools.md)。

如果还需要该嵌套运行解析后的结构化输入，请读取 `context_wrapper.tool_input`。这是 [`RunState`][agents.run_state.RunState] 用于通用序列化嵌套工具输入的字段，而 `agent_tool_invocation` 会直接在结果上公开当前嵌套调用的元数据。

## 流式传输生命周期与诊断

[`RunResultStreaming`][agents.result.RunResultStreaming] 继承上述相同的结果接口，同时增加了流式传输专用控制功能：

-   使用 [`stream_events()`][agents.result.RunResultStreaming.stream_events] 消费语义流事件
-   使用 [`current_agent`][agents.result.RunResultStreaming.current_agent] 追踪运行期间的活动智能体
-   使用 [`is_complete`][agents.result.RunResultStreaming.is_complete] 查看流式传输运行是否已完全结束
-   使用 [`cancel(...)`][agents.result.RunResultStreaming.cancel] 立即停止运行或在当前轮结束后停止运行

持续消费 `stream_events()`，直到异步迭代器结束。该迭代器结束前，流式传输运行不算完成；在最后一个可见 token 到达后，`final_output`、`interruptions`、`raw_responses` 等汇总属性以及会话持久化副作用可能仍在处理。

如果调用 `cancel()`，请继续消费 `stream_events()`，以便正确完成取消和清理。

Python 不会公开单独的流式 `completed` Promise 或 `error` 属性。终止运行的流式传输故障会由 `stream_events()` 抛出，而 `is_complete` 会反映运行是否已到达终止状态。

### 原始响应

[`raw_responses`][agents.result.RunResultBase.raw_responses] 包含运行期间收集的原始模型响应。多步骤运行可能产生多个响应，例如在任务转移或重复的模型/工具/模型循环中。

[`last_response_id`][agents.result.RunResultBase.last_response_id] 只是 `raw_responses` 中最后一个条目的 ID。

### 安全防护措施结果

智能体级安全防护措施通过 [`input_guardrail_results`][agents.result.RunResultBase.input_guardrail_results] 和 [`output_guardrail_results`][agents.result.RunResultBase.output_guardrail_results] 公开。

工具安全防护措施则通过 [`tool_input_guardrail_results`][agents.result.RunResultBase.tool_input_guardrail_results] 和 [`tool_output_guardrail_results`][agents.result.RunResultBase.tool_output_guardrail_results] 单独公开。

这些数组会在整个运行期间持续累积，因此适合用于记录决策、存储额外的安全防护措施元数据，或调试运行被阻止的原因。

### 上下文与用量

[`context_wrapper`][agents.result.RunResultBase.context_wrapper] 会公开你的应用上下文，以及由 SDK 管理的运行时元数据，例如审批、用量和嵌套的 `tool_input`。

用量记录在 `context_wrapper.usage` 中。对于流式传输运行，用量总计可能要等到流的最终数据块处理完毕后才会更新。有关完整的封装结构和持久化注意事项，请参阅[上下文管理](context.md)。