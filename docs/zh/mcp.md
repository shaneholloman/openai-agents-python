---
search:
  exclude: true
---
# Model context protocol (MCP)

[Model context protocol](https://modelcontextprotocol.io/introduction)（MCP）对应用如何向语言模型公开工具和上下文进行了标准化。官方文档中的定义如下：

> MCP是一种开放协议，对应用如何向LLMs提供上下文进行了标准化。可以将MCP视为AI
> 应用的 USB-C 端口。正如 USB-C 提供了一种将设备连接到各种外围设备和配件的标准化方式，MCP
> 也提供了一种将 AI 模型连接到不同数据源和工具的标准化方式。

Agents Python SDK支持多种MCP传输方式。这样，你可以复用现有的MCP服务，也可以构建自己的服务，向智能体公开由文件系统、HTTP 或连接器支持的工具。

!!! warning "连接前信任MCP服务"

    MCP工具可以公开模型上下文中的数据，并使用你提供的凭据执行操作。请仅连接到你信任的服务，使用最小权限凭据，将访问令牌放在授权字段或标头中而非 URL 中，并要求对敏感操作进行审批。请参阅[OpenAI MCP安全指南](https://developers.openai.com/api/docs/guides/tools-connectors-mcp#risks-and-safety)。

## MCP集成方案选择

在将MCP服务接入智能体之前，需要确定工具调用应在何处执行，以及你可以访问哪些传输方式。下表汇总了 Python SDK支持的选项。

| 你的需求                                                                        | 推荐选项                                    |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| 让OpenAI的 Responses API代表模型调用可公开访问的MCP服务| 通过[`HostedMCPTool`][agents.tool.HostedMCPTool]使用**托管式MCP服务工具** |
| 连接到你在本地或远程运行的 Streamable HTTP 服务                  | 通过[`MCPServerStreamableHttp`][agents.mcp.server.MCPServerStreamableHttp]使用**Streamable HTTP MCP服务** |
| 与实现了基于 Server-Sent Events 的 HTTP 的服务通信                          | 通过[`MCPServerSse`][agents.mcp.server.MCPServerSse]使用**基于 SSE 的 HTTP MCP服务** |
| 启动本地进程并通过 stdin/stdout 通信                             | 通过[`MCPServerStdio`][agents.mcp.server.MCPServerStdio]使用**stdio MCP服务** |

以下各节将介绍每种选项、配置方式，以及何时应优先选择某种传输方式。

## 智能体级MCP配置

除了选择传输方式之外，还可以通过设置 `Agent.mcp_config` 来调整MCP工具的准备方式。

```python
from agents import Agent

agent = Agent(
    name="Assistant",
    mcp_servers=[server],
    mcp_config={
        # Try to convert MCP tool schemas to strict JSON schema.
        "convert_schemas_to_strict": True,
        # If None, MCP tool failures are raised as exceptions instead of
        # returning model-visible error text.
        "failure_error_function": None,
        # Prefix local MCP tool names with their server name.
        "include_server_in_tool_names": True,
    },
)
```

注意：

- `convert_schemas_to_strict` 会尽力执行转换。如果某个架构无法转换，则使用原始架构。
- `failure_error_function` 控制如何向模型呈现MCP工具调用失败。
- 未设置 `failure_error_function` 时，SDK使用默认的工具错误格式化程序。
- 服务级 `failure_error_function` 会覆盖该服务的 `Agent.mcp_config["failure_error_function"]`。
- `include_server_in_tool_names` 需要显式启用。启用后，每个本地MCP工具都会以带有确定性服务前缀的名称公开给模型，这有助于避免多个MCP服务发布同名工具时发生冲突。生成的名称符合 ASCII 安全要求，不超过工具调用名称的长度限制，并且不会与同一智能体上现有的本地工具调用及已启用的任务转移名称冲突。SDK仍会在原始服务上调用原始MCP工具名称。

## 各传输方式的通用模式

选择传输方式后，大多数集成还需要做出相同的后续决策：

- 如何仅公开工具的一个子集（[工具筛选](#tool-filtering)）。
- 服务是否还提供可复用的提示词（[提示词](#prompts)）。
- 是否应缓存 `list_tools()`（[缓存](#caching)）。
- MCP活动如何显示在追踪记录中（[追踪](#tracing)）。

对于本地MCP服务（`MCPServerStdio`、`MCPServerSse`、`MCPServerStreamableHttp`），审批策略和每次调用的 `_meta` 负载也是通用概念。Streamable HTTP 一节展示了最完整的代码示例，同样的模式也适用于其他本地传输方式。

## 1. 托管式MCP服务工具

托管工具会将整个工具往返流程转移到OpenAI的基础设施中。你的代码无需列出并调用工具，[`HostedMCPTool`][agents.tool.HostedMCPTool] 会将服务标签（以及可选的连接器元数据）转发给 Responses API。模型会列出远程服务的工具并调用它们，无需再回调你的 Python 进程。托管工具目前可与支持 Responses API托管式MCP集成的OpenAI模型配合使用。

### 基础托管式MCP工具

将 [`HostedMCPTool`][agents.tool.HostedMCPTool] 添加到智能体的 `tools` 列表中，即可创建托管工具。`tool_config`
字典与发送给 REST API的 JSON 相对应：

```python
import asyncio

from agents import Agent, HostedMCPTool, Runner

async def main() -> None:
    agent = Agent(
        name="Assistant",
        instructions="Use the DeepWiki hosted MCP server to inspect openai/openai-agents-python.",
        tools=[
            HostedMCPTool(
                tool_config={
                    "type": "mcp",
                    "server_label": "deepwiki",
                    "server_url": "https://mcp.deepwiki.com/mcp",
                    "require_approval": "never",
                }
            )
        ],
    )

    result = await Runner.run(
        agent,
        "Which language is the repository openai/openai-agents-python written in?",
    )
    print(result.final_output)

asyncio.run(main())
```

托管服务会自动公开其工具；无需将其添加到 `mcp_servers`。

如果希望托管工具搜索延迟加载托管式MCP服务，请设置 `tool_config["defer_loading"] = True`，并将 [`ToolSearchTool`][agents.tool.ToolSearchTool] 添加到智能体。此功能仅受OpenAI Responses 模型支持。有关完整的工具搜索设置和限制，请参阅[工具](tools.md#hosted-tool-search)。

### 托管式MCP结果的流式传输

托管工具支持与工具调用完全相同的流式传输结果方式。使用 `Runner.run_streamed`
可以在模型仍在工作时使用增量MCP输出：

```python
result = Runner.run_streamed(agent, "Summarise this repository's top languages")
async for event in result.stream_events():
    if event.type == "run_item_stream_event":
        print(f"Received: {event.item}")
print(result.final_output)
```

### 可选审批流程

如果服务可以执行敏感操作，可以要求在每次执行工具前进行人工或程序化审批。在 `tool_config` 中配置 `require_approval`，其值可以是单一策略（`"always"`、`"never"`），也可以是将工具名称映射到策略的字典。若要在 Python 中做出决定，请提供 `on_approval_request` 回调。

```python
from agents import MCPToolApprovalFunctionResult, MCPToolApprovalRequest

SAFE_TOOLS = {"read_wiki_structure", "read_wiki_contents", "ask_question"}

def approve_tool(request: MCPToolApprovalRequest) -> MCPToolApprovalFunctionResult:
    if request.data.name in SAFE_TOOLS:
        return {"approve": True}
    return {"approve": False, "reason": "Escalate to a human reviewer"}

agent = Agent(
    name="Assistant",
    tools=[
        HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": "always",
            },
            on_approval_request=approve_tool,
        )
    ],
)
```

该回调可以是同步或异步的，并会在模型需要审批数据以继续运行时调用。

### 由连接器支持的托管服务

托管式MCP也支持OpenAI连接器。无需指定 `server_url`，只需提供 `connector_id` 和访问令牌。Responses API负责处理身份验证，托管服务则公开连接器的工具。

```python
import os

HostedMCPTool(
    tool_config={
        "type": "mcp",
        "server_label": "google_calendar",
        "connector_id": "connector_googlecalendar",
        "authorization": os.environ["GOOGLE_CALENDAR_AUTHORIZATION"],
        "require_approval": "never",
    }
)
```

完整可运行的托管工具示例（包括流式传输、审批和连接器）位于 [`examples/hosted_mcp`](https://github.com/openai/openai-agents-python/tree/main/examples/hosted_mcp)。

## 2. Streamable HTTP MCP服务

如果希望自行管理网络连接，请使用 [`MCPServerStreamableHttp`][agents.mcp.server.MCPServerStreamableHttp]。当你需要控制传输方式，或希望在自己的基础设施中运行服务并保持低延迟时，Streamable HTTP 服务是理想选择。

```python
import asyncio
import os

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings

async def main() -> None:
    token = os.environ["MCP_SERVER_TOKEN"]
    async with MCPServerStreamableHttp(
        name="Streamable HTTP Python Server",
        params={
            "url": "http://localhost:8000/mcp",
            "headers": {"Authorization": f"Bearer {token}"},
            "timeout": 10,
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    ) as server:
        agent = Agent(
            name="Assistant",
            instructions="Use the MCP tools to answer the questions.",
            mcp_servers=[server],
            model_settings=ModelSettings(tool_choice="required"),
        )

        result = await Runner.run(agent, "Add 7 and 22.")
        print(result.final_output)

asyncio.run(main())
```

构造函数还接受以下选项：

- `client_session_timeout_seconds` 控制MCP ClientSession 的读取超时。可由 `datetime.timedelta` 表示且不小于一微秒的有限正数会设置有限超时；`None` 和 `0` 会禁用超时。构造服务时，其他值将被拒绝。
- `use_structured_content` 控制是否优先使用 `tool_result.structured_content`，而不是文本输出。
- `max_retry_attempts` 和 `retry_backoff_seconds_base` 为 `list_tools()` 和 `call_tool()` 添加自动重试。
- `tool_filter` 允许你仅公开工具的一个子集（请参阅[工具筛选](#tool-filtering)）。
- `require_approval` 为本地MCP工具启用人工介入审批策略。
- `failure_error_function` 自定义模型可见的MCP工具失败消息；将其设为 `None` 则会改为抛出错误。
- `tool_meta_resolver` 在调用 `call_tool()` 前注入每次调用的MCP `_meta` 负载。

### 本地MCP服务的审批策略

`MCPServerStdio`、`MCPServerSse` 和 `MCPServerStreamableHttp` 均接受 `require_approval`。

支持的形式：

- 对所有工具使用 `"always"` 或 `"never"`。
- `True` / `False`（分别等同于始终审批/从不审批）。
- 按工具配置的映射，例如 `{"delete_file": "always", "read_file": "never"}`。
- 分组对象：`{"always": {"tool_names": [...]}, "never": {"tool_names": [...]}}`。

```python
async with MCPServerStreamableHttp(
    name="Filesystem MCP",
    params={"url": "http://localhost:8000/mcp"},
    require_approval={"always": {"tool_names": ["delete_file"]}},
) as server:
    ...
```

有关完整的暂停/恢复流程，请参阅[人工介入](human_in_the_loop.md)和 `examples/mcp/get_all_mcp_tools_example/main.py`。

### 使用 `tool_meta_resolver` 配置每次调用的元数据

当MCP服务期望在 `_meta` 中接收请求元数据（例如租户 ID 或追踪上下文）时，请使用 `tool_meta_resolver`。以下示例假设你将 `dict` 作为 `context` 传递给 `Runner.run(...)`。

```python
from agents.mcp import MCPServerStreamableHttp, MCPToolMetaContext


def resolve_meta(context: MCPToolMetaContext) -> dict[str, str] | None:
    run_context_data = context.run_context.context or {}
    tenant_id = run_context_data.get("tenant_id")
    if tenant_id is None:
        return None
    return {"tenant_id": str(tenant_id), "source": "agents-sdk"}


server = MCPServerStreamableHttp(
    name="Metadata-aware MCP",
    params={"url": "http://localhost:8000/mcp"},
    tool_meta_resolver=resolve_meta,
)
```

如果运行上下文是 Pydantic 模型、数据类或自定义类，请改用属性访问来读取租户 ID。

### MCP工具输出：文本和图像

当MCP工具返回图像内容时，SDK会自动将其映射为图像工具输出条目。混合的文本/图像响应会作为输出项列表转发，因此智能体使用MCP图像结果的方式，与使用常规工具调用所产生的图像输出相同。

## 3. 基于 SSE 的 HTTP MCP服务

!!! warning

    MCP项目已弃用 Server-Sent Events 传输。对于新集成，请优先使用 Streamable HTTP 或 stdio，仅为旧版服务保留 SSE。

如果MCP服务实现了基于 SSE 的 HTTP 传输，请实例化 [`MCPServerSse`][agents.mcp.server.MCPServerSse]。除传输方式外，其 API 与 Streamable HTTP 服务相同。

```python

from agents import Agent, Runner
from agents.model_settings import ModelSettings
from agents.mcp import MCPServerSse

workspace_id = "demo-workspace"

async with MCPServerSse(
    name="SSE Python Server",
    params={
        "url": "http://localhost:8000/sse",
        "headers": {"X-Workspace": workspace_id},
    },
    cache_tools_list=True,
) as server:
    agent = Agent(
        name="Assistant",
        mcp_servers=[server],
        model_settings=ModelSettings(tool_choice="required"),
    )
    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)
```

## 4. stdio MCP服务

对于以本地子进程方式运行的MCP服务，请使用 [`MCPServerStdio`][agents.mcp.server.MCPServerStdio]。SDK会生成进程、保持管道打开，并在上下文管理器退出时自动将其关闭。此选项适用于快速概念验证，或服务仅公开命令行入口点的情况。

```python
from pathlib import Path
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

current_dir = Path(__file__).parent
samples_dir = current_dir / "sample_files"

async with MCPServerStdio(
    name="Filesystem Server via npx",
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(samples_dir)],
    },
) as server:
    agent = Agent(
        name="Assistant",
        instructions="Use the files in the sample directory to answer questions.",
        mcp_servers=[server],
    )
    result = await Runner.run(agent, "List the files available to you.")
    print(result.final_output)
```

## 5. MCP服务管理器

如果有多个MCP服务，请使用 `MCPServerManager` 预先连接它们，并将已连接的服务子集公开给智能体。有关构造函数选项和重新连接行为，请参阅 [MCPServerManager API参考](ref/mcp/manager.md)。

```python
from agents import Agent, Runner
from agents.mcp import MCPServerManager, MCPServerStreamableHttp

servers = [
    MCPServerStreamableHttp(name="calendar", params={"url": "http://localhost:8000/mcp"}),
    MCPServerStreamableHttp(name="docs", params={"url": "http://localhost:8001/mcp"}),
]

async with MCPServerManager(servers) as manager:
    agent = Agent(
        name="Assistant",
        instructions="Use MCP tools when they help.",
        mcp_servers=manager.active_servers,
    )
    result = await Runner.run(agent, "Which MCP tools are available?")
    print(result.final_output)
```

关键行为：

- 当 `drop_failed_servers=True`（默认值）时，`active_servers` 仅包含成功连接的服务。
- 连接失败会记录在 `failed_servers` 和 `errors` 中。
- 设置 `strict=True` 可在首次连接失败时抛出异常。
- 调用 `reconnect(failed_only=True)` 可重试失败的服务，调用 `reconnect(failed_only=False)` 则会重启所有服务。
- 设置 `connect_timeout_seconds`、`cleanup_timeout_seconds` 和 `connect_in_parallel` 可调整生命周期行为。生命周期超时接受有限正秒数，也可以设为 `None` 以禁用超时，并且会在构造和赋值时进行验证；不接受零，因为零会创建立即到期的截止时间。

## 通用服务能力

以下各节适用于各种MCP服务传输方式（具体 API 范围取决于服务类）。

## 工具筛选

每个MCP服务都支持工具筛选器，因此你可以只公开智能体所需的函数。筛选可以在构造时进行，也可以在每次运行时动态进行。

### 静态工具筛选

使用 [`create_static_tool_filter`][agents.mcp.create_static_tool_filter] 配置简单的允许/阻止列表：

```python
from pathlib import Path

from agents.mcp import MCPServerStdio, create_static_tool_filter

samples_dir = Path("/path/to/files")

filesystem_server = MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(samples_dir)],
    },
    tool_filter=create_static_tool_filter(allowed_tool_names=["read_file", "write_file"]),
)
```

当同时提供 `allowed_tool_names` 和 `blocked_tool_names` 时，SDK会先应用允许列表，然后从剩余集合中移除所有被阻止的工具。

### 动态工具筛选

如需更复杂的逻辑，请传入一个接收 [`ToolFilterContext`][agents.mcp.ToolFilterContext] 的可调用对象。该可调用对象可以是同步或异步的，并在应公开该工具时返回 `True`。

```python
from pathlib import Path

from agents.mcp import MCPServerStdio, ToolFilterContext

samples_dir = Path("/path/to/files")

async def context_aware_filter(context: ToolFilterContext, tool) -> bool:
    if context.agent.name == "Code Reviewer" and tool.name.startswith("danger_"):
        return False
    return True

async with MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(samples_dir)],
    },
    tool_filter=context_aware_filter,
) as server:
    ...
```

筛选器上下文会公开活动的 `run_context`、请求工具的 `agent` 和 `server_name`。

## 提示词

MCP服务还可以提供动态生成智能体指令的提示词。支持提示词的服务会公开两种
方法：

- `list_prompts()` 枚举可用的提示词模板。
- `get_prompt(name, arguments)` 获取具体的提示词，可选择提供参数。

```python
from agents import Agent

prompt_result = await server.get_prompt(
    "generate_code_review_instructions",
    {"focus": "security vulnerabilities", "language": "python"},
)
instructions = prompt_result.messages[0].content.text

agent = Agent(
    name="Code Reviewer",
    instructions=instructions,
    mcp_servers=[server],
)
```

## 分页

内置的本地MCP服务类在列出工具和提示词时会自动跟随 `nextCursor`。`list_tools()` 会在应用筛选器或填充缓存前返回完整的工具列表，而 `list_prompts()` 会返回一个合并结果，其中 `nextCursor=None`。如果后续页面失败或服务重复返回某个游标，该操作会抛出错误，而不会公开或缓存部分结果。

资源仍会明确分页。将 `list_resources()` 或 `list_resource_templates()` 返回的 `nextCursor` 作为 `cursor` 参数传回，即可获取下一页。

## 缓存

每次智能体运行都会在每个MCP服务上调用 `list_tools()`。远程服务可能会产生明显的延迟，因此所有MCP服务类都公开了 `cache_tools_list` 选项。只有在确信工具定义不会频繁更改时，才应将其设为 `True`。若之后需要强制获取最新列表，请在服务实例上调用 `invalidate_tools_cache()`。

## 追踪

[追踪](./tracing.md)会自动捕获MCP活动，包括：

1. 为列出工具而对MCP服务进行的调用。
2. 工具调用中与MCP相关的信息。

![MCP追踪截图](../assets/images/mcp-tracing.jpg)

## 延伸阅读

- [Model Context Protocol](https://modelcontextprotocol.io/) – 规范和设计指南。
- [examples/mcp](https://github.com/openai/openai-agents-python/tree/main/examples/mcp) – 可运行的 stdio、SSE 和 Streamable HTTP 示例。
- [examples/hosted_mcp](https://github.com/openai/openai-agents-python/tree/main/examples/hosted_mcp) – 完整的托管式MCP演示，包括审批和连接器。