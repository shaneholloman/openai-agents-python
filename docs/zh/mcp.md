---
search:
  exclude: true
---
# Model context protocol (MCP)

[Model context protocol](https://modelcontextprotocol.io/introduction)（MCP）对应用程序向语言模型提供工具和上下文的方式进行了标准化。官方文档中的说明如下：

> MCP是一种开放协议，用于标准化应用程序向LLM提供上下文的方式。可以将MCP视为AI
> 应用程序的USB-C端口。正如USB-C提供了一种将设备连接到各种外围设备和配件的标准化方式，MCP
> 也提供了一种将AI模型连接到不同数据源和工具的标准化方式。

Agents Python SDK支持多种MCP传输方式。这样，你就可以复用现有MCP服务，也可以构建自己的MCP服务，从而向智能体提供由文件系统、HTTP或连接器支持的工具。

!!! warning "连接MCP服务前的信任要求"

    MCP工具可以公开模型上下文中的数据，并使用你提供的凭据执行操作。请仅连接你信任的服务、使用最小权限凭据、将访问令牌放在授权字段或请求头中而非URL中，并要求对敏感操作进行审批。请参阅[OpenAI MCP安全指南](https://developers.openai.com/api/docs/guides/tools-connectors-mcp#risks-and-safety)。

## MCP集成方案选择

在将MCP服务接入智能体之前，应先确定工具调用的执行位置，以及可访问的传输方式。下表总结了Python SDK支持的选项。

| 需求                                                                        | 推荐选项                                    |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| 让OpenAI的Responses API代表模型调用可公开访问的MCP服务| 通过[`HostedMCPTool`][agents.tool.HostedMCPTool]使用**托管式MCP服务工具** |
| 连接到你在本地或远程运行的Streamable HTTP服务                  | 通过[`MCPServerStreamableHttp`][agents.mcp.server.MCPServerStreamableHttp]使用**Streamable HTTP MCP服务** |
| 与实现了采用服务端发送事件的HTTP协议的服务通信                          | 通过[`MCPServerSse`][agents.mcp.server.MCPServerSse]使用**采用SSE的HTTP MCP服务** |
| 启动本地进程并通过stdin/stdout通信                             | 通过[`MCPServerStdio`][agents.mcp.server.MCPServerStdio]使用**stdio MCP服务** |

以下各节将逐一介绍每个选项、配置方式，以及何时应优先选择某种传输方式。

## 智能体级MCP配置

除了选择传输方式外，你还可以通过设置`Agent.mcp_config`来调整MCP工具的准备方式。

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

注意事项：

- `convert_schemas_to_strict`采用尽力而为的方式。如果某个模式无法转换，则使用原始模式。
- `failure_error_function`控制如何将MCP工具调用失败呈现给模型。
- 未设置`failure_error_function`时，SDK会使用默认的工具错误格式化程序。
- 服务级`failure_error_function`会覆盖该服务的`Agent.mcp_config["failure_error_function"]`。
- `include_server_in_tool_names`需要选择启用。启用后，每个本地MCP工具都会以带有确定性服务前缀的名称提供给模型，有助于避免多个MCP服务发布同名工具时发生冲突。生成的名称符合ASCII安全要求，不会超过工具调用名称的长度限制，并会避开同一智能体中已有的本地工具调用名称和已启用的任务转移名称。SDK仍会在原始服务上调用原始MCP工具名称。

## 各传输方式的通用模式

选择传输方式后，大多数集成都需要做出相同的后续决策：

- 如何仅公开工具的子集（[工具筛选](#tool-filtering)）。
- 服务是否还提供可复用的提示词（[提示词](#prompts)）。
- 是否应缓存`list_tools()`（[缓存](#caching)）。
- MCP活动如何显示在追踪记录中（[追踪](#tracing)）。

对于本地MCP服务（`MCPServerStdio`、`MCPServerSse`、`MCPServerStreamableHttp`），审批策略和每次调用的`_meta`载荷也是通用概念。Streamable HTTP一节提供了最完整的示例，相同模式也适用于其他本地传输方式。

## 1. 托管式MCP服务工具

托管工具会将整个工具往返流程交由OpenAI的基础设施处理。你的代码无需列出和调用工具，[`HostedMCPTool`][agents.tool.HostedMCPTool]会将服务标签（以及可选的连接器元数据）转发给Responses API。模型会列出远程服务的工具并调用它们，无需额外回调你的Python进程。托管工具目前适用于支持Responses API托管式MCP集成的OpenAI模型。

### 基础托管式MCP工具

将[`HostedMCPTool`][agents.tool.HostedMCPTool]添加到智能体的`tools`列表中，即可创建托管工具。`tool_config`
字典对应于你会发送给REST API的JSON：

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

托管式服务会自动公开其工具；无需将其添加到`mcp_servers`。

如果希望托管工具搜索延迟加载托管式MCP服务，请设置`tool_config["defer_loading"] = True`，并将[`ToolSearchTool`][agents.tool.ToolSearchTool]添加到智能体。此功能仅支持OpenAI Responses模型。有关完整的工具搜索设置和限制，请参阅[工具](tools.md#hosted-tool-search)。

### 托管式MCP结果的流式传输

托管工具对流式结果的支持方式与工具调用完全相同。使用`Runner.run_streamed`可以在模型仍在工作时
接收增量MCP输出：

```python
result = Runner.run_streamed(agent, "Summarise this repository's top languages")
async for event in result.stream_events():
    if event.type == "run_item_stream_event":
        print(f"Received: {event.item}")
print(result.final_output)
```

### 可选审批流程

如果服务可以执行敏感操作，你可以要求在每次执行工具前进行人工或程序化审批。在`tool_config`中配置`require_approval`，其值可以是单一策略（`"always"`、`"never"`），也可以是将工具名称映射到策略的字典。如需在Python中做出决定，请提供`on_approval_request`回调。

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

该回调可以是同步或异步的，并且每当模型需要审批信息才能继续运行时都会被调用。

### 连接器支持的托管式服务

托管式MCP还支持OpenAI连接器。你无需指定`server_url`，只需提供`connector_id`和访问令牌。Responses API会处理身份验证，托管式服务则会公开连接器的工具。

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

完整可运行的托管工具示例（包括流式传输、审批和连接器）位于[`examples/hosted_mcp`](https://github.com/openai/openai-agents-python/tree/main/examples/hosted_mcp)。

## 2. Streamable HTTP MCP服务

如果希望自行管理网络连接，请使用[`MCPServerStreamableHttp`][agents.mcp.server.MCPServerStreamableHttp]。当你需要控制传输方式，或希望在自己的基础设施中运行服务并保持较低延迟时，Streamable HTTP服务是理想选择。

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

- `client_session_timeout_seconds`控制HTTP读取超时。
- `use_structured_content`控制是否优先使用`tool_result.structured_content`而非文本输出。
- `max_retry_attempts`和`retry_backoff_seconds_base`为`list_tools()`和`call_tool()`添加自动重试。
- `tool_filter`用于仅公开工具的子集（请参阅[工具筛选](#tool-filtering)）。
- `require_approval`为本地MCP工具启用人工介入审批策略。
- `failure_error_function`用于自定义模型可见的MCP工具失败消息；将其设置为`None`则改为抛出错误。
- `tool_meta_resolver`会在`call_tool()`之前注入每次调用的MCP `_meta`载荷。

### 本地MCP服务的审批策略

`MCPServerStdio`、`MCPServerSse`和`MCPServerStreamableHttp`都接受`require_approval`。

支持的形式：

- 对所有工具使用`"always"`或`"never"`。
- `True` / `False`（分别等同于always/never）。
- 按工具配置的映射，例如`{"delete_file": "always", "read_file": "never"}`。
- 分组对象：`{"always": {"tool_names": [...]}, "never": {"tool_names": [...]}}`。

```python
async with MCPServerStreamableHttp(
    name="Filesystem MCP",
    params={"url": "http://localhost:8000/mcp"},
    require_approval={"always": {"tool_names": ["delete_file"]}},
) as server:
    ...
```

有关完整的暂停/恢复流程，请参阅[人工介入](human_in_the_loop.md)和`examples/mcp/get_all_mcp_tools_example/main.py`。

### 使用`tool_meta_resolver`的每次调用元数据

当MCP服务要求在`_meta`中包含请求元数据（例如租户ID或追踪上下文）时，请使用`tool_meta_resolver`。以下示例假设你将`dict`作为`context`传递给`Runner.run(...)`。

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

如果运行上下文是Pydantic模型、数据类或自定义类，请改为通过属性访问读取租户ID。

### MCP工具输出：文本与图像

当MCP工具返回图像内容时，SDK会自动将其映射为图像工具输出条目。混合文本/图像响应会作为输出项列表转发，因此智能体可以像使用普通工具调用产生的图像输出一样使用MCP图像结果。

## 3. 采用SSE的HTTP MCP服务

!!! warning

    MCP项目已弃用服务端发送事件传输方式。新集成应优先使用Streamable HTTP或stdio，仅为旧版服务保留SSE。

如果MCP服务实现了采用SSE的HTTP传输方式，请实例化[`MCPServerSse`][agents.mcp.server.MCPServerSse]。除传输方式外，其API与Streamable HTTP服务完全相同。

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

对于作为本地子进程运行的MCP服务，请使用[`MCPServerStdio`][agents.mcp.server.MCPServerStdio]。SDK会启动进程、保持管道打开，并在退出上下文管理器时自动关闭它们。此选项适用于快速构建概念验证，或服务仅提供命令行入口点的情况。

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

当你有多个MCP服务时，可使用`MCPServerManager`预先连接它们，并向智能体公开已连接的服务子集。有关构造函数选项和重新连接行为，请参阅[MCPServerManager API参考](ref/mcp/manager.md)。

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

- 当`drop_failed_servers=True`（默认值）时，`active_servers`仅包含成功连接的服务。
- 失败信息会记录在`failed_servers`和`errors`中。
- 设置`strict=True`可在第一次连接失败时抛出异常。
- 调用`reconnect(failed_only=True)`可重试连接失败的服务，调用`reconnect(failed_only=False)`则会重启所有服务。
- 使用`connect_timeout_seconds`、`cleanup_timeout_seconds`和`connect_in_parallel`调整生命周期行为。

## 常见服务能力

以下各节适用于各种MCP服务传输方式（确切的API接口取决于服务类）。

## 工具筛选

每个MCP服务都支持工具筛选，因此你可以仅公开智能体所需的函数。筛选既可以在构造时执行，也可以在每次运行时动态执行。

### 静态工具筛选

使用[`create_static_tool_filter`][agents.mcp.create_static_tool_filter]配置简单的允许/阻止列表：

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

当同时提供`allowed_tool_names`和`blocked_tool_names`时，SDK会先应用允许列表，然后从剩余集合中移除所有被阻止的工具。

### 动态工具筛选

对于更复杂的逻辑，请传入一个接收[`ToolFilterContext`][agents.mcp.ToolFilterContext]的可调用对象。该可调用对象可以是同步或异步的，并在应公开工具时返回`True`。

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

筛选上下文会提供当前`run_context`、请求工具的`agent`以及`server_name`。

## 提示词

MCP服务还可以提供用于动态生成智能体指令的提示词。支持提示词的服务会公开两个
方法：

- `list_prompts()`列出可用的提示词模板。
- `get_prompt(name, arguments)`获取具体提示词，并可选择附带参数。

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

## 缓存

每次智能体运行都会在每个MCP服务上调用`list_tools()`。远程服务可能会引入明显延迟，因此所有MCP服务类都提供`cache_tools_list`选项。仅当你确信工具定义不会频繁变化时，才将其设置为`True`。如需之后强制获取最新列表，请在服务实例上调用`invalidate_tools_cache()`。

## 追踪

[追踪](./tracing.md)会自动捕获MCP活动，包括：

1. 为列出工具而对MCP服务发起的调用。
2. 工具调用中与MCP相关的信息。

![MCP追踪截图](../assets/images/mcp-tracing.jpg)

## 延伸阅读

- [Model Context Protocol](https://modelcontextprotocol.io/) – 规范与设计指南。
- [examples/mcp](https://github.com/openai/openai-agents-python/tree/main/examples/mcp) – 可运行的stdio、SSE和Streamable HTTP代码示例。
- [examples/hosted_mcp](https://github.com/openai/openai-agents-python/tree/main/examples/hosted_mcp) – 包含审批和连接器的完整托管式MCP演示。