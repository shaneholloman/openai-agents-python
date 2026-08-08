---
search:
  exclude: true
---
# 人工介入

使用人工介入（HITL）流程暂停智能体执行，直到有人批准或拒绝敏感工具调用。工具会声明其何时需要审批，运行结果会以中断项的形式显示待处理的审批，而 `RunState` 可让你序列化已暂停的运行，并在作出决定后恢复运行。

该审批机制适用于整个运行，并不限于当前的顶层智能体。无论工具属于当前智能体、通过任务转移到达的智能体，还是嵌套的 [`Agent.as_tool()`][agents.agent.Agent.as_tool] 执行，都采用相同的模式。在嵌套的 `Agent.as_tool()` 情况下，中断仍会显示在外层运行中，因此你需要在外层 `RunState` 上批准或拒绝它，然后恢复原始顶层运行。

使用 `Agent.as_tool()` 时，审批可能发生在两个不同层级：智能体工具本身可以通过 `Agent.as_tool(..., needs_approval=...)` 要求审批，而嵌套智能体中的工具可以在嵌套运行开始后提出各自的审批请求。二者都通过相同的外层运行中断流程处理。

本页重点介绍通过 `interruptions` 进行的人工审批流程。如果你的应用可以通过代码作出决定，某些工具类型也支持程序化审批回调，使运行无需暂停即可继续。

## 需要审批的工具标记

将 `needs_approval` 设置为 `True`，可始终要求审批；也可以提供一个异步函数，按每次调用作出决定。该可调用对象会接收运行上下文、已解析的工具参数和工具调用 ID。

当 SDK 无法安全检查参数时，可调用的审批规则会采取默认拒绝策略。如果参数是格式错误的 JSON、是有效 JSON 但并非对象（例如 `null` 或列表），或者包含 `NaN`、`Infinity` 或 `-Infinity` 等非标准常量，则不会调用该可调用对象，并且该调用需要人工审批。Runner 和 Realtime 工具调用的行为相同。

```python
from agents import Agent
from agents.decorators import tool


@tool(needs_approval=True)
async def cancel_order(order_id: int) -> str:
    return f"Cancelled order {order_id}"


async def requires_review(_ctx, params, _call_id) -> bool:
    return "refund" in params.get("subject", "").lower()


@tool(needs_approval=requires_review)
async def send_email(subject: str, body: str) -> str:
    return f"Sent '{subject}'"


agent = Agent(
    name="Support agent",
    instructions="Handle tickets and ask for approval when needed.",
    tools=[cancel_order, send_email],
)
```

`needs_approval` 可用于 [`function_tool`][agents.tool.function_tool]、[`Agent.as_tool`][agents.agent.Agent.as_tool]、[`ShellTool`][agents.tool.ShellTool] 和 [`ApplyPatchTool`][agents.tool.ApplyPatchTool]。本地 MCP服务器也通过 [`MCPServerStdio`][agents.mcp.server.MCPServerStdio]、[`MCPServerSse`][agents.mcp.server.MCPServerSse] 和 [`MCPServerStreamableHttp`][agents.mcp.server.MCPServerStreamableHttp] 上的 `require_approval` 支持审批。托管式 MCP服务器通过 [`HostedMCPTool`][agents.tool.HostedMCPTool] 支持审批，该工具使用 `tool_config={"require_approval": "always"}` 和可选的 `on_approval_request` 回调。如果你希望自动批准或自动拒绝，而不触发中断，Shell 和 apply_patch 工具可接受 `on_approval` 回调。

## 审批流程的工作原理

1. 当模型发出工具调用时，运行器会评估其审批规则（`needs_approval`、`require_approval` 或托管式 MCP 的对应规则）。
2. 如果该工具调用的审批决定已存储在 [`RunContextWrapper`][agents.run_context.RunContextWrapper] 中，运行器将直接继续执行，不再提示。每次调用的审批仅适用于特定调用 ID；传入 `always_approve=True` 或 `always_reject=True`，可在本次运行剩余期间，为以后对该工具的调用保留相同决定。
3. 如果审批规则要求审批，但尚未存储该工具调用的决定，执行会暂停，并且 `RunResult.interruptions`（或 `RunResultStreaming.interruptions`）会包含 [`ToolApprovalItem`][agents.items.ToolApprovalItem] 条目，其中具有 `agent.name`、`tool_name` 和 `arguments` 等详细信息。这包括任务转移后或嵌套 `Agent.as_tool()` 执行中提出的审批请求。
4. 使用 `result.to_state()` 将结果转换为 `RunState`，调用 `state.approve(...)` 或 `state.reject(...)`，然后使用 `Runner.run(agent, state)` 或 `Runner.run_streamed(agent, state)` 恢复运行，其中 `agent` 是该运行的原始顶层智能体。
5. 恢复后的运行会从暂停处继续；如果需要新的审批，则会再次进入此流程。

使用 `always_approve=True` 或 `always_reject=True` 创建的持久决定会存储在运行状态中，因此之后恢复同一已暂停的运行时，它们可以在 `state.to_string()` / `RunState.from_string(...)` 和 `state.to_json()` / `RunState.from_json(...)` 过程中继续保留。

你不必在同一次处理中解决所有待审批项。`interruptions` 可以同时包含常规函数工具、托管式 MCP 审批和嵌套的 `Agent.as_tool()` 审批。如果你仅批准或拒绝部分条目后重新运行，已解决的调用可以继续执行，而未解决的调用会保留在 `interruptions` 中，并再次暂停运行。

## 自定义拒绝消息

默认情况下，被拒绝的工具调用会将 SDK 的标准拒绝文本返回到运行中。你可以在两个层级自定义该消息：

-   整个运行的回退设置：设置 [`RunConfig.tool_error_formatter`][agents.run.RunConfig.tool_error_formatter]，以控制整个运行中审批被拒绝时模型可见的默认消息。
-   单次调用覆盖：如果希望某个特定的被拒绝工具调用显示不同的消息，请将 `rejection_message=...` 传给 `state.reject(...)`。

如果二者都已提供，则单次调用的 `rejection_message` 优先于整个运行的格式化程序。

```python
from agents import RunConfig, ToolErrorFormatterArgs


def format_rejection(args: ToolErrorFormatterArgs[None]) -> str | None:
    if args.kind != "approval_rejected":
        return None
    return "Publish action was canceled because approval was rejected."


run_config = RunConfig(tool_error_formatter=format_rejection)

# Later, while resolving a specific interruption:
state.reject(
    interruption,
    rejection_message="Publish action was canceled because the reviewer denied approval.",
)
```

有关同时展示这两个层级的完整代码示例，请参阅 [`examples/agent_patterns/human_in_the_loop_custom_rejection.py`](https://github.com/openai/openai-agents-python/tree/main/examples/agent_patterns/human_in_the_loop_custom_rejection.py)。

## 自动审批决定

手动处理 `interruptions` 是最通用的模式，但并非唯一模式：

-   本地 [`ShellTool`][agents.tool.ShellTool] 和 [`ApplyPatchTool`][agents.tool.ApplyPatchTool] 可以使用 `on_approval`，立即在代码中批准或拒绝。
-   [`HostedMCPTool`][agents.tool.HostedMCPTool] 可以结合使用 `tool_config={"require_approval": "always"}` 和 `on_approval_request`，作出同类程序化决定。
-   普通 [`function_tool`][agents.tool.function_tool] 工具和 [`Agent.as_tool()`][agents.agent.Agent.as_tool] 使用本页介绍的手动中断流程。

当这些回调返回决定时，运行会继续，而无需暂停以等待人工响应。对于 Realtime 和语音会话 API，请参阅 [Realtime 指南](realtime/guide.md)中的审批流程。

## 流式传输与会话

相同的中断流程也适用于流式运行。流式运行暂停后，继续使用 [`RunResultStreaming.stream_events()`][agents.result.RunResultStreaming.stream_events]，直到迭代器结束；然后检查 [`RunResultStreaming.interruptions`][agents.result.RunResultStreaming.interruptions]、处理中断项，并使用 [`Runner.run_streamed(...)`][agents.run.Runner.run_streamed] 恢复运行，以使恢复后的输出继续进行流式传输。有关该模式的流式版本，请参阅[流式传输](streaming.md)。

如果你还在使用会话，从 `RunState` 恢复时，请继续传入同一个会话实例，或传入另一个为相同会话 ID 和后端存储配置的会话对象。恢复后的轮次随后会追加到同一份已存储的对话历史中。有关会话生命周期的详细信息，请参阅[会话](sessions/index.md)。

## 示例：暂停、批准与恢复

下面的代码片段与 JavaScript HITL 指南中的流程一致：当工具需要审批时暂停，将状态持久化到磁盘，重新加载状态，并在收集到决定后恢复运行。

```python
import asyncio
import json
from pathlib import Path

from agents import Agent, Runner, RunState
from agents.decorators import tool


async def needs_oakland_approval(_ctx, params, _call_id) -> bool:
    return "Oakland" in params.get("city", "")


@tool(needs_approval=needs_oakland_approval)
async def get_temperature(city: str) -> str:
    return f"The temperature in {city} is 20° Celsius"


agent = Agent(
    name="Weather assistant",
    instructions="Answer weather questions with the provided tools.",
    tools=[get_temperature],
)

STATE_PATH = Path(".cache/hitl_state.json")


def prompt_approval(tool_name: str, arguments: str | None) -> bool:
    answer = input(f"Approve {tool_name} with {arguments}? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


async def main() -> None:
    result = await Runner.run(agent, "What is the temperature in Oakland?")

    while result.interruptions:
        # Persist the paused state.
        state = result.to_state()
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(state.to_string())

        # Load the state later (could be a different process).
        stored = json.loads(STATE_PATH.read_text())
        state = await RunState.from_json(agent, stored)

        for interruption in result.interruptions:
            approved = await asyncio.get_running_loop().run_in_executor(
                None, prompt_approval, interruption.name or "unknown_tool", interruption.arguments
            )
            if approved:
                state.approve(interruption, always_approve=False)
            else:
                state.reject(interruption)

        result = await Runner.run(agent, state)

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

在此示例中，`prompt_approval` 是同步函数，因为它使用 `input()`，并通过 `run_in_executor(...)` 执行。如果你的审批来源本身已经是异步的（例如 HTTP 请求或异步数据库查询），则可以使用 `async def` 函数，并直接对其执行 `await`。

要在可能因审批而暂停的运行中使用流式传输，请调用 `Runner.run_streamed`，持续使用 `result.stream_events()` 直至完成，然后执行上文所示的相同 `result.to_state()` 和恢复步骤。

## 仓库模式与代码示例

- **流式审批**：`examples/agent_patterns/human_in_the_loop_stream.py` 展示如何完整消费 `stream_events()`，然后批准待处理的工具调用，再使用 `Runner.run_streamed(agent, state)` 恢复运行。
- **自定义拒绝文本**：`examples/agent_patterns/human_in_the_loop_custom_rejection.py` 展示审批被拒绝时，如何将运行级 `tool_error_formatter` 与单次调用的 `rejection_message` 覆盖结合使用。
- **智能体作为工具的审批**：当委派给智能体的任务需要审核时，`Agent.as_tool(..., needs_approval=...)` 会应用相同的中断流程。嵌套中断仍会显示在外层运行中，因此应恢复原始顶层智能体，而非嵌套智能体。
- **本地 Shell 和 apply_patch 工具**：`ShellTool` 和 `ApplyPatchTool` 也支持 `needs_approval`。使用 `state.approve(interruption, always_approve=True)` 或 `state.reject(..., always_reject=True)`，可在本次运行剩余期间缓存决定，供以后对该工具的调用使用。对于自动决定，请提供 `on_approval`（参阅 `examples/tools/shell.py`）；对于手动决定，请处理中断项（参阅 `examples/tools/shell_human_in_the_loop.py`）。托管式 Shell 环境不支持 `needs_approval` 或 `on_approval`；请参阅[工具指南](tools.md)。
- **本地 MCP服务器**：使用 `MCPServerStdio` / `MCPServerSse` / `MCPServerStreamableHttp` 上的 `require_approval`，对 MCP 工具调用设置审批门控（参阅 `examples/mcp/get_all_mcp_tools_example/main.py` 和 `examples/mcp/tool_filter_example/main.py`）。
- **托管式 MCP服务器**：在 `HostedMCPTool` 上设置 `tool_config={"require_approval": "always"}` 以强制执行 HITL，也可以选择提供 `on_approval_request` 以自动批准或拒绝（参阅 `examples/hosted_mcp/human_in_the_loop.py` 和 `examples/hosted_mcp/on_approval.py`）。对于可信服务器，请使用 `"never"`（`examples/hosted_mcp/simple.py`）。
- **会话与记忆**：将会话传给 `Runner.run`，使审批和对话历史能够跨多个轮次保留。SQLite 和 OpenAI Conversations 会话变体位于 `examples/memory/memory_session_hitl_example.py` 和 `examples/memory/openai_session_hitl_example.py` 中。
- **实时智能体**：实时演示提供了 WebSocket 消息，可通过 `RealtimeSession` 上的 `approve_tool_call` / `reject_tool_call` 批准或拒绝工具调用（有关服务器端处理程序，请参阅 `examples/realtime/app/server.py`；有关 API 接口，请参阅 [Realtime 指南](realtime/guide.md#tool-approvals)）。

## 长时间运行的审批

`RunState` 采用持久化设计。使用 `state.to_json()` 或 `state.to_string()` 将待处理工作存储在数据库或队列中，之后再使用 `RunState.from_json(...)` 或 `RunState.from_string(...)` 重新创建它。

可用的序列化选项：

-   `context_serializer`：自定义非映射上下文对象的序列化方式。
-   `context_deserializer`：使用 `RunState.from_json(...)` 或 `RunState.from_string(...)` 加载状态时，重新构建非映射上下文对象。
- `strict_context=True`：除非上下文本身已是映射或你提供了 `context_serializer`，否则序列化失败；除非上下文本身已是映射或你提供了 `context_deserializer`，否则反序列化失败。
- `context_override`：加载状态时替换已序列化的上下文。如果你不想恢复原始上下文对象，此选项会很有用，但它不会从已序列化的载荷中移除该上下文。
- `include_tracing_api_key=True`：当恢复的工作需要继续使用相同凭据导出追踪数据时，在已序列化的追踪载荷中包含追踪 API 密钥。

已序列化的运行状态包括你的应用上下文，以及由 SDK 管理的运行时元数据，例如审批、用量、已序列化的 `tool_input`、嵌套的智能体作为工具的恢复信息、追踪元数据和服务器管理的对话设置。如果你计划存储或传输已序列化的状态，请将 `RunContextWrapper.context` 视为持久化数据，并避免在其中放置机密信息，除非你确实希望这些机密随状态一起传输。

## 待处理任务的版本控制

如果审批可能搁置一段时间，请将智能体定义或 SDK 的版本标记与已序列化状态一同存储。随后，你可以将反序列化路由到匹配的代码路径，以避免模型、提示词或工具定义发生变化时出现不兼容问题。