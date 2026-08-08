---
search:
  exclude: true
---
# 用量

Agents SDK会自动追踪每次运行的 token 用量。你可以从运行上下文中访问这些信息，用于监控成本、强制执行限制或记录分析数据。

## 追踪内容

- **requests**：发出的 LLM API 调用次数
- **input_tokens**：发送的输入 token 总数
- **output_tokens**：接收的输出 token 总数
- **total_tokens**：输入 + 输出
- **request_usage_entries**：每个请求的用量明细列表
- **details**：
  - `input_tokens_details.cached_tokens`
  - `output_tokens_details.reasoning_tokens`

## 运行中的用量访问

在`Runner.run(...)`完成后，通过`result.context_wrapper.usage`访问用量。

```python
result = await Runner.run(agent, "What's the weather in Tokyo?")
usage = result.context_wrapper.usage

print("Requests:", usage.requests)
print("Input tokens:", usage.input_tokens)
print("Output tokens:", usage.output_tokens)
print("Total tokens:", usage.total_tokens)
```

用量会汇总运行期间的所有模型调用，包括产生工具调用或任务转移的模型调用。

### 第三方适配器的用量启用

不同第三方适配器和提供商后端的用量报告方式各不相同。如果你通过第三方适配器访问模型，并且需要准确的`result.context_wrapper.usage`值：

- 使用`AnyLLMModel`时，如果上游提供商返回用量数据，系统会自动传递该数据。从 Chat Completions后端流式传输响应时，可能需要设置`ModelSettings(include_usage=True)`，才能发出用量数据块。
- 使用`LitellmModel`时，某些提供商后端默认不报告用量，因此通常需要`ModelSettings(include_usage=True)`。

请查看模型指南中[第三方适配器](models/index.md#third-party-adapters)一节的适配器特定说明，并在你计划部署的确切提供商后端上验证用量报告。

## 逐请求用量追踪

SDK 会自动在`request_usage_entries`中追踪每个 API 请求的用量，这有助于详细计算成本和监控上下文窗口占用情况。

```python
result = await Runner.run(agent, "What's the weather in Tokyo?")

for i, request in enumerate(result.context_wrapper.usage.request_usage_entries):
    print(f"Request {i + 1}: {request.input_tokens} in, {request.output_tokens} out")
```

## 会话中的用量访问

使用`Session`（例如`SQLiteSession`）时，每次调用`Runner.run(...)`都会返回该次特定运行的用量。会话会保留对话历史记录以提供上下文，但每次运行的用量彼此独立。

```python
session = SQLiteSession("my_conversation")

first = await Runner.run(agent, "Hi!", session=session)
print(first.context_wrapper.usage.total_tokens)  # Usage for first run

second = await Runner.run(agent, "Can you elaborate?", session=session)
print(second.context_wrapper.usage.total_tokens)  # Usage for second run
```

请注意，虽然会话会在不同运行之间保留对话上下文，但每次调用`Runner.run()`返回的用量指标仅代表该次执行。在会话中，之前的消息可能会作为输入重新送入每次运行，这会影响后续轮次的输入 token 数量。

## 钩子中的用量信息

如果你使用`RunHooks`，传递给每个钩子的`context`对象都包含`usage`。这使你可以在生命周期的关键时刻记录用量。

```python
class MyHooks(RunHooks):
    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        u = context.usage
        print(f"{agent.name} → {u.requests} requests, {u.total_tokens} total tokens")
```

## API 参考

有关详细的 API 文档，请参阅：

-   [`Usage`][agents.usage.Usage] - 用量追踪数据结构
-   [`RequestUsage`][agents.usage.RequestUsage] - 每个请求的用量详情
-   [`RunContextWrapper`][agents.run.RunContextWrapper] - 从运行上下文中访问用量
-   [`RunHooks`][agents.run.RunHooks] - 接入用量追踪生命周期