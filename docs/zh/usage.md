---
search:
  exclude: true
---
# 用量

Agents SDK会自动追踪每次运行的令牌用量。你可以从运行上下文中访问这些数据，并用其监控成本、执行限额或记录分析数据。

## 追踪内容

- **requests**：LLM API调用次数
- **input_tokens**：发送的输入令牌总数
- **output_tokens**：接收的输出令牌总数
- **total_tokens**：输入 + 输出
- **request_usage_entries**：每个请求的用量明细列表
- **details**：
  - `input_tokens_details.cached_tokens`
  - `input_tokens_details.cache_write_tokens`
  - `output_tokens_details.reasoning_tokens`

## 运行用量的访问

在 `Runner.run(...)` 执行后，通过 `result.context_wrapper.usage` 访问用量。

```python
result = await Runner.run(agent, "What's the weather in Tokyo?")
usage = result.context_wrapper.usage

print("Requests:", usage.requests)
print("Input tokens:", usage.input_tokens)
print("Output tokens:", usage.output_tokens)
print("Total tokens:", usage.total_tokens)
```

用量会汇总运行期间的所有模型调用，包括生成工具调用或任务转移的模型调用。

当 [`OpenAIResponsesCompactionSession`][agents.memory.openai_responses_compaction_session.OpenAIResponsesCompactionSession] 在运行结束前自动压缩历史记录时，该 `responses.compact` 请求报告的用量也会添加到同一次运行的总量中。在运行之外手动调用 `run_compaction()` 时，不存在相应的运行上下文，因此不会更新此前运行返回的用量对象。请参阅 [OpenAI响应压缩会话](sessions/index.md#openai-responses-compaction-sessions)。

### 第三方适配器的用量启用

不同第三方适配器和提供商后端的用量报告方式各不相同。如果你通过第三方适配器访问模型，并且需要准确的 `result.context_wrapper.usage` 值：

- 使用 `AnyLLMModel` 时，如果上游提供商返回用量数据，系统会自动传递这些数据。从Chat Completions后端流式传输响应时，可能需要设置 `ModelSettings(include_usage=True)` 才能发送用量数据块。
- 使用 `LitellmModel` 时，某些提供商后端默认不报告用量，因此通常需要设置 `ModelSettings(include_usage=True)`。

请查看模型指南中[第三方适配器](models/index.md#third-party-adapters)一节的适配器专属说明，并在计划部署的具体提供商后端上验证用量报告。

## 按请求的用量追踪

SDK会在 `request_usage_entries` 中自动追踪每个 API 请求的用量，这有助于进行详细的成本计算和监控上下文窗口消耗。

```python
result = await Runner.run(agent, "What's the weather in Tokyo?")

for i, request in enumerate(result.context_wrapper.usage.request_usage_entries):
    print(f"Request {i + 1}: {request.input_tokens} in, {request.output_tokens} out")
```

## 提供商用量载荷的保留

Agents SDK会将提供商用量标准化为 [`Usage`][agents.usage.Usage] 字段，从而在不同模型提供商之间提供一致的用量总计。当应用必须保留提供商特有的用量字段，或区分被省略的字段与提供商报告的零值时，请将 [`ModelSettings.preserve_raw_usage`][agents.model_settings.ModelSettings.preserve_raw_usage] 设置为 `True`：

```python
from agents import Agent, ModelSettings, Runner

agent = Agent(
    name="Assistant",
    model_settings=ModelSettings(preserve_raw_usage=True),
)
result = await Runner.run(agent, "What's the weather in Tokyo?")

for response in result.raw_responses:
    print(response.raw_usage)
```

Agents SDK会将每个 [`ModelResponse.raw_usage`][agents.items.ModelResponse.raw_usage] 值存储为该次模型调用中提供商载荷的独立 JSON 兼容快照。Agents SDK不会在整个运行期间汇总 `raw_usage`。当禁用保留、提供商未返回用量载荷，或上游适配器已丢弃原始字段存在性信息时，该值仍为 `None`。

`preserve_raw_usage` 仅保留到达模型适配器的用量载荷；此设置不会向提供商请求用量数据。当流式Chat Completions提供商要求显式请求用量时，还需设置 `ModelSettings(include_usage=True)`。

`LitellmModel` 目前不会在流式或非流式运行中填充 `ModelResponse.raw_usage`，因此 `preserve_raw_usage=True` 对该适配器无效。使用 `LitellmModel` 时，请继续使用标准化的 [`Usage`][agents.usage.Usage] 字段；如果需要保留提供商特有的字段存在性信息，请选择支持保留原始用量的适配器。

## 会话中的用量访问

使用 `Session`（例如 `SQLiteSession`）时，每次调用 `Runner.run(...)` 都会返回该次特定运行的用量。会话会保留对话历史记录以提供上下文，但每次运行的用量相互独立。

```python
session = SQLiteSession("my_conversation")

first = await Runner.run(agent, "Hi!", session=session)
print(first.context_wrapper.usage.total_tokens)  # Usage for first run

second = await Runner.run(agent, "Can you elaborate?", session=session)
print(second.context_wrapper.usage.total_tokens)  # Usage for second run
```

请注意，尽管会话会在多次运行之间保留对话上下文，但每次调用 `Runner.run()` 返回的用量指标仅代表该次执行。在会话中，之前的消息可能会作为输入重新提供给每次运行，这会影响后续轮次的输入令牌数量。

## 钩子中的用量使用

如果你使用 `RunHooks`，传递给每个钩子的 `context` 对象都包含 `usage`。借助此对象，你可以在生命周期的关键时刻记录用量。

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
-   [`RunContextWrapper`][agents.run.RunContextWrapper] - 从运行上下文访问用量
-   [`RunHooks`][agents.run.RunHooks] - 接入用量追踪生命周期