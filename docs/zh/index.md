---
search:
  exclude: true
---
# OpenAI Agents SDK

[OpenAI Agents SDK](https://github.com/openai/openai-agents-python)让你能够通过一个轻量、易用且仅包含少量抽象概念的软件包构建智能体式 AI 应用。它是我们此前智能体实验项目[Swarm](https://github.com/openai/swarm/tree/main)面向生产环境的升级版本。Agents SDK仅包含一组非常精简的基本组件：

-   **智能体**，即配备指令和工具的LLM
-   **Agents as tools / 任务转移**，允许智能体将特定任务委派给其他智能体
-   **安全防护措施**，用于验证智能体的输入和输出

这些基本组件与 Python 结合后，足以表达工具与智能体之间的复杂关系，让你无需经历陡峭的学习曲线即可构建实际应用。此外，SDK 还内置了**追踪**功能，可用于可视化和调试智能体流程、对其进行评估，甚至针对你的应用微调模型。

## 使用Agents SDK的理由

SDK 遵循两项核心设计原则：

1. 提供足够丰富、值得使用的功能，同时保持基本组件精简，以便快速上手。
2. 开箱即用，同时允许你精确自定义具体行为。

SDK 的主要功能包括：

-   **智能体**：使用指令、工具、安全防护措施和任务转移构建智能体，并通过内置循环持续运行，直至任务完成。
-   **沙箱智能体**：在真实的隔离工作区中运行专业智能体，支持由清单定义的文件、沙箱客户端选择，以及可恢复的沙箱会话。
-   **实时智能体**：使用`gpt-realtime-2.1`构建强大的语音智能体，支持自动中断检测、上下文管理、安全防护措施等功能。
-   **语音智能体**：构建结合语音转文本、智能体工作流和文本转语音的语音管线。
-   **Python 优先**：使用内置语言功能编排和串联智能体，无需学习新的抽象概念。
-   **Agents as tools / 任务转移**：一种强大的机制，用于在多个智能体之间协调和委派工作。
-   **安全防护措施**：在智能体执行的同时并行运行输入验证和安全检查，并在检查未通过时快速失败。
-   **工具调用**：将任意 Python 函数转换为工具，并自动生成模式，同时使用 Pydantic 进行验证。
-   **MCP服务工具调用**：内置MCP服务工具集成，使用方式与工具调用相同。
-   **会话**：一种持久化记忆层，用于在智能体循环中维护工作上下文。
-   **人在回路**：内置在多次智能体运行中引入人工参与的机制。
-   **追踪**：内置追踪功能，用于可视化、调试和监控工作流，并支持OpenAI的一整套评估、微调和蒸馏工具。

## Agents SDK与Responses API的选择

对于OpenAI模型，SDK 默认使用 Responses API，但在模型调用之外增加了更高级别的运行时。

以下情况可直接使用 Responses API：

-   你希望自行控制循环、工具分派和状态处理
-   你的工作流生命周期较短，主要目标是返回模型响应

以下情况可使用Agents SDK：

-   你希望由运行时管理轮次、工具执行、安全防护措施、任务转移或会话
-   你的智能体需要生成产物，或通过多个协调步骤执行操作
-   你需要通过[沙箱智能体](sandbox_agents.md)使用真实工作区或可恢复执行

你不必在整个应用中只选择其中一种。许多应用使用 SDK 处理受管理的工作流，并针对较底层的路径直接调用 Responses API。

## 安装

```bash
pip install openai-agents
```

## Hello world示例

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
```

(_运行此示例时，请确保已设置`OPENAI_API_KEY`环境变量_)

```bash
export OPENAI_API_KEY=sk-...
```

## 入门指南

-   通过[快速入门](quickstart.md)构建你的第一个文本智能体。
-   然后在[运行智能体](running_agents.md#choose-a-memory-strategy)中确定如何跨轮次保留状态。
-   如果任务依赖真实文件、代码仓库或每个智能体独立的工作区状态，请阅读[沙箱智能体快速入门](sandbox_agents.md)。
-   如果你正在任务转移与管理器式编排之间进行选择，请阅读[智能体编排](multi_agent.md)。

## 路径选择

当你知道要完成什么工作，但不确定哪个页面提供相关说明时，可使用下表。

| 目标 | 入门页面 |
| --- | --- |
| 构建第一个文本智能体并查看一次完整运行 | [快速入门](quickstart.md) |
| 添加工具调用、托管工具或Agents as tools | [工具](tools.md) |
| 在真实的隔离工作区中运行编码、审查或文档智能体 | [沙箱智能体快速入门](sandbox_agents.md)和[沙箱客户端](sandbox/clients.md) |
| 在任务转移与管理器式编排之间进行选择 | [智能体编排](multi_agent.md) |
| 跨轮次保留记忆 | [运行智能体](running_agents.md#choose-a-memory-strategy)和[会话](sessions/index.md) |
| 使用OpenAI模型、WebSocket 传输或非OpenAI提供商 | [模型](models/index.md) |
| 查看输出、运行项、中断和恢复状态 | [结果](results.md) |
| 使用`gpt-realtime-2.1`构建低延迟语音智能体 | [实时智能体快速入门](realtime/quickstart.md)和[实时传输](realtime/transport.md) |
| 构建语音转文本 / 智能体 / 文本转语音管线 | [语音管线快速入门](voice/quickstart.md) |