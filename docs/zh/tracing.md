---
search:
  exclude: true
---
# 追踪

Agents SDK 内置了追踪功能，可收集智能体运行期间的完整事件记录：LLM 生成、工具调用、任务转移、安全防护措施，甚至包括发生的自定义事件。借助[追踪仪表板](https://platform.openai.com/traces)，你可以在开发和生产环境中调试、可视化和监控工作流。

!!!note

    追踪默认启用。你可以通过以下三种常用方式将其禁用：

    1. 设置环境变量 `OPENAI_AGENTS_DISABLE_TRACING=1`，全局禁用追踪
    2. 在代码中使用 [`set_tracing_disabled(True)`][agents.set_tracing_disabled]，全局禁用追踪
    3. 将 [`agents.run.RunConfig.tracing_disabled`][] 设置为 `True`，为单次运行禁用追踪

***对于根据零数据保留（ZDR）政策使用OpenAI API 的组织，追踪功能不可用。***

## 追踪和跨度 {#traces-and-spans}

-   **追踪**表示一次“工作流”的端到端操作。它们由跨度组成。追踪具有以下属性：
    -   `workflow_name`：逻辑工作流或应用的名称。例如“代码生成”或“客户服务”。
    -   `trace_id`：追踪的唯一 ID。如果未传入，则会自动生成。格式必须为 `trace_<32_alphanumeric>`。
    -   `group_id`：可选的组 ID，用于关联同一对话中的多个追踪。例如，你可以使用聊天会话 ID。
    -   `disabled`：如果为 True，则不会记录该追踪。
    -   `metadata`：追踪的可选元数据。
-   **跨度**表示具有开始和结束时间的操作。跨度包含：
    -   `started_at` 和 `ended_at` 时间戳。
    -   `trace_id`，表示它们所属的追踪
    -   `parent_id`，指向此跨度的父跨度（如果有）
    -   `span_data`，即有关跨度的信息。例如，`AgentSpanData` 包含有关智能体的信息，`GenerationSpanData` 包含有关 LLM 生成的信息，依此类推。

## 默认追踪 {#default-tracing}

默认情况下，SDK 会追踪以下内容：

-   整个 `Runner.{run, run_sync, run_streamed}()` 都封装在 `trace()` 中。
-   每次运行器调用都封装在 `task_span()` 中。
-   每个模型轮次都封装在 `turn_span()` 中。
-   智能体每次运行时，都会封装在 `agent_span()` 中
-   LLM 生成封装在 `generation_span()` 中
-   每次函数工具调用都封装在 `function_span()` 中
-   安全防护措施封装在 `guardrail_span()` 中
-   任务转移封装在 `handoff_span()` 中
-   音频输入（语音转文本）封装在 `transcription_span()` 中
-   音频输出（文本转语音）封装在 `speech_span()` 中
-   SDK 可能会将相关的音频跨度置于 `speech_group_span()` 下

默认情况下，追踪名称是字面字符串 `Agent workflow`。如果使用 `trace`，你可以设置此名称；也可以通过 [`RunConfig`][agents.run.RunConfig] 配置名称和其他属性。

如果希望层次结构更紧凑，可以为某次运行禁用自动创建的任务跨度和轮次跨度。智能体、生成、函数、安全防护措施、任务转移和自定义跨度仍会被记录。

```python
from agents import RunConfig, Runner

result = await Runner.run(
    agent,
    "Hello",
    run_config=RunConfig(tracing={"include_task_and_turn_spans": False}),
)
```

此外，你还可以设置[自定义追踪处理器](#custom-tracing-processors)，将追踪发送到其他目标位置（作为替代目标或辅助目标）。

## 长时运行工作进程和即时导出 {#long-running-workers-and-immediate-exports}

默认的 [`BatchTraceProcessor`][agents.tracing.processors.BatchTraceProcessor] 每隔几秒在后台导出追踪；如果内存队列达到大小阈值，则会更早导出；进程退出时还会执行最终刷新。对于 Celery、RQ、Dramatiq 或 FastAPI 后台任务等长时运行的工作进程，这意味着通常无需任何额外代码即可自动导出追踪，但它们不一定会在每个作业完成后立即显示在追踪仪表板中。

如果需要确保在一个工作单元结束时立即完成传送，请在退出追踪上下文后调用 [`flush_traces()`][agents.tracing.flush_traces]。

```python
from agents import Runner, flush_traces, trace


@celery_app.task
def run_agent_task(prompt: str):
    try:
        with trace("celery_task"):
            result = Runner.run_sync(agent, prompt)
        return result.final_output
    finally:
        flush_traces()
```

```python
from fastapi import BackgroundTasks, FastAPI
from agents import Runner, flush_traces, trace

app = FastAPI()


def process_in_background(prompt: str) -> None:
    try:
        with trace("background_job"):
            Runner.run_sync(agent, prompt)
    finally:
        flush_traces()


@app.post("/run")
async def run(prompt: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_in_background, prompt)
    return {"status": "queued"}
```

[`flush_traces()`][agents.tracing.flush_traces] 会阻塞，直到当前缓冲的追踪和跨度全部导出，因此请在 `trace()` 关闭后调用它，以免刷新尚未构建完成的追踪。如果可以接受默认的导出延迟，则可以跳过此调用。

## 更高层级的追踪 {#higher-level-traces}

有时，你可能希望多次调用 `run()` 时都归入同一个追踪。为此，可以将整个代码封装在 `trace()` 中。

```python
from agents import Agent, Runner, trace

async def main():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")

    with trace("Joke workflow"): # (1)!
        first_result = await Runner.run(agent, "Tell me a joke")
        second_result = await Runner.run(agent, f"Rate this joke: {first_result.final_output}")
        print(f"Joke: {first_result.final_output}")
        print(f"Rating: {second_result.final_output}")
```

1. 由于两次 `Runner.run` 调用都封装在 `with trace()` 中，因此两次运行会成为同一个整体追踪的一部分，而不是各自创建单独的追踪。

## 追踪的创建 {#creating-traces}

你可以使用 [`trace()`][agents.tracing.trace] 函数创建追踪。追踪需要启动和结束。你可以通过以下两种方式完成：

1. **推荐**：将追踪用作上下文管理器，即 `with trace(...) as my_trace`。这会在正确的时间自动启动和结束追踪。
2. 你也可以手动调用 [`trace.start()`][agents.tracing.Trace.start] 和 [`trace.finish()`][agents.tracing.Trace.finish]。

当前追踪通过 Python 的 [`contextvar`](https://docs.python.org/3/library/contextvars.html) 进行跟踪。这意味着它可自动支持并发。如果手动启动和结束追踪，请将 `mark_as_current` 传给 `start()`，并将 `reset_current` 传给 `finish()`，以更新当前追踪。

## 跨度的创建 {#creating-spans}

你可以使用各种 [`*_span()`][agents.tracing.create] 方法创建跨度。通常无需手动创建跨度。你可以使用 [`custom_span()`][agents.tracing.custom_span] 函数跟踪自定义跨度信息。

跨度会自动成为当前追踪的一部分，并嵌套在距离最近的当前跨度下；当前跨度通过 Python 的 [`contextvar`](https://docs.python.org/3/library/contextvars.html) 进行跟踪。

## 敏感数据 {#sensitive-data}

某些跨度可能会捕获潜在的敏感数据。

`generation_span()` 会存储 LLM 生成的输入和输出，而 `function_span()` 会存储函数调用的输入和输出。这些内容可能包含敏感数据，因此你可以通过 [`RunConfig.trace_include_sensitive_data`][agents.run.RunConfig.trace_include_sensitive_data] 禁止捕获这些数据。

同样，默认情况下，音频跨度会包含输入和输出音频的 Base64 编码 PCM 数据。你可以通过配置 [`VoicePipelineConfig.trace_include_sensitive_audio_data`][agents.voice.pipeline_config.VoicePipelineConfig.trace_include_sensitive_audio_data] 禁止捕获这些音频数据。

默认情况下，`trace_include_sensitive_data` 为 `True`。你可以在运行应用之前，将 `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA` 环境变量导出为 `true/1` 或 `false/0`，从而无需编写代码即可设置默认值。

## 自定义追踪处理器 {#custom-tracing-processors}

追踪功能的高层架构如下：

-   初始化时，我们会创建一个全局 [`TraceProvider`][agents.tracing.provider.TraceProvider]，用于创建追踪。
-   我们为 `TraceProvider` 配置一个 [`BatchTraceProcessor`][agents.tracing.processors.BatchTraceProcessor]，它会将追踪和跨度分批发送到 [`BackendSpanExporter`][agents.tracing.processors.BackendSpanExporter]，后者会将这些跨度和追踪分批导出到OpenAI后端。

如需自定义此默认设置，将追踪发送到其他或额外的后端，或者修改导出器行为，你有以下两个选项：

1. [`add_trace_processor()`][agents.tracing.add_trace_processor] 允许你添加一个**额外的**追踪处理器，在追踪和跨度准备就绪时接收它们。这样，除了将追踪发送到OpenAI后端外，你还可以自行处理它们。
2. [`set_trace_processors()`][agents.tracing.set_trace_processors] 允许你使用自己的追踪处理器**替换**默认处理器。这意味着，除非你包含一个可将追踪发送到OpenAI后端的 `TracingProcessor`，否则追踪不会发送到该后端。


## 非OpenAI模型的追踪 {#tracing-with-non-openai-models}

使用非OpenAI模型时，你可以向追踪导出器提供 OpenAI API 密钥，从而无需禁用追踪，即可在OpenAI追踪仪表板中使用免费追踪功能。有关适配器的选择和设置注意事项，请参阅模型指南中的[第三方适配器](models/index.md#third-party-adapters)部分。

```python
import os
from agents import set_tracing_export_api_key, Agent
from agents.extensions.models.any_llm_model import AnyLLMModel

tracing_api_key = os.environ["OPENAI_API_KEY"]
set_tracing_export_api_key(tracing_api_key)

model = AnyLLMModel(
    model="your-provider/your-model-name",
    api_key="your-api-key",
)

agent = Agent(
    name="Assistant",
    model=model,
)
```

如果只需为单次运行使用不同的追踪密钥，请通过 `RunConfig` 传入该密钥，而不要更改全局导出器。

```python
from agents import Runner, RunConfig

await Runner.run(
    agent,
    input="Hello",
    run_config=RunConfig(tracing={"api_key": "sk-tracing-123"}),
)
```

## 补充说明 {#additional-notes}
- 可在OpenAI追踪仪表板中查看免费的追踪记录。


## 生态系统集成 {#ecosystem-integrations}

以下社区和供应商集成支持 OpenAI Agents SDK 的追踪 API 接口。

### 外部追踪处理器列表 {#external-tracing-processors-list}

-   [Weights & Biases](https://weave-docs.wandb.ai/guides/integrations/openai_agents)
-   [Arize-Phoenix](https://docs.arize.com/phoenix/tracing/integrations-tracing/openai-agents-sdk)
-   [Future AGI](https://docs.futureagi.com/docs/tracing/auto/openai_agents/)
-   [MLflow（自托管/OSS）](https://mlflow.org/docs/latest/tracing/integrations/openai-agent)
-   [MLflow（Databricks 托管）](https://docs.databricks.com/aws/en/mlflow/mlflow-tracing#-automatic-tracing)
-   [Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk)
-   [Pydantic Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents)
-   [AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk)
-   [Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration)
-   [Respan](https://respan.ai/docs/integrations/tracing/openai-agents-sdk)
-   [LangSmith](https://docs.smith.langchain.com/observability/how_to_guides/trace_with_openai_agents_sdk)
-   [Maxim AI](https://www.getmaxim.ai/docs/observe/integrations/openai-agents-sdk)
-   [Comet Opik](https://www.comet.com/docs/opik/tracing/integrations/openai_agents)
-   [Langfuse](https://langfuse.com/docs/integrations/openaiagentssdk/openai-agents)
-   [Langtrace](https://docs.langtrace.ai/supported-integrations/llm-frameworks/openai-agents-sdk)
-   [Okahu-Monocle](https://github.com/monocle2ai/monocle)
-   [Galileo](https://v2docs.galileo.ai/integrations/openai-agent-integration#openai-agent-integration)
-   [Portkey AI](https://portkey.ai/docs/integrations/agents/openai-agents)
-   [LangDB AI](https://docs.langdb.ai/getting-started/working-with-agent-frameworks/working-with-openai-agents-sdk)
-   [Agenta](https://docs.agenta.ai/observability/integrations/openai-agents)
-   [PostHog](https://posthog.com/docs/llm-analytics/installation/openai-agents)
-   [Traccia](https://traccia.ai/docs/integrations/openai-agents)
-   [PromptLayer](https://docs.promptlayer.com/features/integrations#openai-agents-sdk)
-   [HoneyHive](https://docs.honeyhive.ai/v2/integrations/openai-agents)
-   [Asqav](https://www.asqav.com/docs/integrations#openai-agents)
-   [Datadog](https://docs.datadoghq.com/llm_observability/instrumentation/auto_instrumentation/?tab=python#openai-agents)
-   [Latitude](https://docs.latitude.so/telemetry/frameworks/openai-agents)
-   [DProvenanceKit](https://dprovenance.dev/openai-agents/)
-   [Tuning Engines](https://github.com/cerebrixos-org/tuning-engines-cli/tree/main/packages/tuning-agents#openai-agents-sdk)