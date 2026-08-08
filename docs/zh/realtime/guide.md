---
search:
  exclude: true
---
# 实时智能体指南

本指南介绍 OpenAI Agents SDK 的实时层如何映射到 OpenAI Realtime API，以及 Python SDK 在此基础上增加的额外行为。

!!! note "从这里开始"

    如果你希望使用默认的 Python 路径，请先阅读[快速入门](quickstart.md)。如果你正在决定应用应使用服务端 WebSocket 还是 SIP，请阅读[实时传输](transport.md)。浏览器 WebRTC 传输不属于 Python SDK。

## 概述

实时智能体会与 Realtime API 保持长期连接，以便模型增量处理文本和音频、以流式方式输出音频、调用工具并处理中断，而无需每轮都重新发起请求。

主要 SDK 组件包括：

-   **RealtimeAgent**：一名实时专用智能体的指令、工具、输出安全防护措施和任务转移
-   **RealtimeRunner**：将起始智能体连接到实时传输的会话工厂
-   **RealtimeSession**：发送输入、接收事件、追踪历史记录并执行工具的实时会话
-   **RealtimeModel**：传输抽象。默认实现是 OpenAI 的服务端 WebSocket。

## 会话生命周期

典型的实时会话如下：

1. 创建一个或多个 `RealtimeAgent`。
2. 使用起始智能体创建 `RealtimeRunner`。
3. 调用 `await runner.run()` 获取 `RealtimeSession`。
4. 使用 `async with session:` 或 `await session.enter()` 进入会话。
5. 使用 `send_message()` 或 `send_audio()` 发送用户输入。
6. 迭代处理会话事件，直到对话结束。

与纯文本运行不同，`runner.run()` 不会立即生成最终结果。它会返回一个实时会话对象，使本地历史记录、后台工具执行、安全防护措施状态和当前智能体配置与传输层保持同步。

默认情况下，`RealtimeRunner` 使用 `OpenAIRealtimeWebSocketModel`，因此默认 Python 路径是连接到 Realtime API 的服务端 WebSocket。如果传入不同的 `RealtimeModel`，仍可使用相同的会话生命周期和智能体功能，但连接机制可以有所不同。

## 智能体与会话配置

`RealtimeAgent` 的范围有意设计得比常规 `Agent` 类型更窄：

-   模型选择在会话级别配置，而不是按智能体配置。
-   不支持 Structured outputs。
-   可以配置语音，但在会话已经生成语音后无法更改。
-   指令、函数工具、任务转移、钩子和输出安全防护措施仍然可用。

`RealtimeSessionModelSettings` 同时支持较新的嵌套 `audio` 配置和旧版扁平别名。新代码应优先使用嵌套形式，并为新的实时智能体从 `gpt-realtime-2.1` 开始：

```python
runner = RealtimeRunner(
    starting_agent=agent,
    config={
        "model_settings": {
            "model_name": "gpt-realtime-2.1",
            "audio": {
                "input": {
                    "format": "pcm16",
                    "transcription": {"model": "gpt-4o-mini-transcribe"},
                    "turn_detection": {"type": "semantic_vad", "interrupt_response": True},
                },
                "output": {"format": "pcm16", "voice": "ash"},
            },
            "tool_choice": "auto",
        }
    },
)
```

常用的会话级设置包括：

-   `audio.input.format`、`audio.output.format`
-   `audio.input.transcription`
-   `audio.input.noise_reduction`
-   `audio.input.turn_detection`
-   `audio.output.voice`、`audio.output.speed`
-   `output_modalities`
-   `tool_choice`
-   `prompt`
-   `tracing`

`RealtimeRunner(config=...)` 上常用的运行级设置包括：

-   `async_tool_calls`
-   `output_guardrails`
-   `guardrails_settings.debounce_text_length`
-   `tool_error_formatter`
-   `tracing_disabled`

有关完整的类型化接口，请参阅 [`RealtimeRunConfig`][agents.realtime.config.RealtimeRunConfig] 和 [`RealtimeSessionModelSettings`][agents.realtime.config.RealtimeSessionModelSettings]。

## 输入与输出

### 文本与结构化用户消息

使用 [`session.send_message()`][agents.realtime.session.RealtimeSession.send_message] 发送纯文本或结构化实时消息。

```python
from agents.realtime import RealtimeUserInputMessage

await session.send_message("Summarize what we discussed so far.")

message: RealtimeUserInputMessage = {
    "type": "message",
    "role": "user",
    "content": [
        {"type": "input_text", "text": "Describe this image."},
        {"type": "input_image", "image_url": image_data_url, "detail": "high"},
    ],
}
await session.send_message(message)
```

结构化消息是在实时对话中加入图像输入的主要方式。[`examples/realtime/app/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/app/server.py) 中的 Web 演示代码示例会以这种方式转发 `input_image` 消息。

### 音频输入

使用 [`session.send_audio()`][agents.realtime.session.RealtimeSession.send_audio] 流式传输原始音频字节：

```python
await session.send_audio(audio_bytes)
```

如果禁用了服务端轮次检测，你需要负责标记轮次边界。高层便捷方法如下：

```python
await session.send_audio(audio_bytes, commit=True)
```

如果需要更底层的控制，也可以通过底层模型传输直接发送 Realtime API 客户端事件，例如 `input_audio_buffer.commit`。

### 手动响应控制

`session.send_message()` 会通过高层路径发送用户输入，并为你启动响应。在某些配置中，原始音频缓冲**不会**自动执行相同操作。

在 Realtime API 层面，手动轮次控制意味着发送一个 `session.update` 事件，将 `turn_detection` 设置为 `null`，然后自行发送 `input_audio_buffer.commit` 和 `response.create`。

如果你正在手动管理轮次，可以通过模型传输发送原始客户端事件：

```python
from agents.realtime.model_inputs import RealtimeModelSendRawMessage

await session.model.send_event(
    RealtimeModelSendRawMessage(
        message={
            "type": "response.create",
        }
    )
)
```

此模式适用于以下情况：

-   禁用了 `turn_detection`，且你希望自行决定模型何时响应
-   希望在触发响应之前检查或管控用户输入
-   需要为带外响应使用自定义提示词

[`examples/realtime/twilio_sip/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio_sip/server.py) 中的 SIP 代码示例使用原始 `response.create` 强制生成开场问候。

## 事件、历史记录与中断

`RealtimeSession` 会发出更高层的 SDK 事件，同时仍会在需要时转发原始模型事件。

重要的会话事件包括：

-   `audio`、`audio_end`、`audio_interrupted`
-   `agent_start`、`agent_end`
-   `tool_start`、`tool_end`、`tool_approval_required`
-   `handoff`
-   `history_added`、`history_updated`
-   `guardrail_tripped`
-   `input_audio_timeout_triggered`
-   `error`
-   `raw_model_event`

对 UI 状态最有用的事件通常是 `history_added` 和 `history_updated`。它们以 `RealtimeItem` 对象的形式公开会话的本地历史记录，包括用户消息、助手消息和工具调用。

### 用量统计

当已完成的模型响应包含用量信息时，SDK 的 OpenAI `RealtimeModel` 传输会在 `raw_model_event` 内发出 [`RealtimeModelUsageEvent`][agents.realtime.model_events.RealtimeModelUsageEvent]。其 `usage` 字段包含该响应的 token 数量，而 `input_tokens_details` 和 `output_tokens_details` 则提供可选的模态细分。

会话还会将每个响应的用量添加到共享的 [`RunContextWrapper.usage`][agents.run_context.RunContextWrapper.usage] 中。若要查看实时会话的累计用量，可在后续的高层事件（例如 `agent_end`）中从 `event.info.context.usage` 读取。

```python
from agents.realtime import RealtimeModelUsageEvent

async for event in session:
    if event.type == "raw_model_event" and isinstance(
        event.data, RealtimeModelUsageEvent
    ):
        response_usage = event.data.usage
        print("Response tokens:", response_usage.total_tokens)
        print("Input modalities:", event.data.input_tokens_details)
        print("Output modalities:", event.data.output_tokens_details)
    elif event.type == "agent_end":
        session_usage = event.info.context.usage
        print("Session tokens:", session_usage.total_tokens)
```

只有当模型提供商在已完成的响应中包含用量信息时，才会报告用量。累计值涵盖该 `RealtimeSession` 收到的响应，并非跨会话总量。

### 中断与播放追踪

当用户打断助手时，会话会发出 `audio_interrupted` 并更新历史记录，使服务端对话与用户实际听到的内容保持一致。

对于低延迟本地播放，默认播放追踪器通常已足够。对于远程或延迟播放场景，尤其是电话场景，请使用 [`RealtimePlaybackTracker`][agents.realtime.model.RealtimePlaybackTracker]，这样被中断的响应会在实际播放位置处截断，而不是假定所有已生成的音频都已播放给用户。

[`examples/realtime/twilio/twilio_handler.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio/twilio_handler.py) 中的 Twilio 代码示例展示了此模式。

## 工具、审批、任务转移与安全防护措施

### 函数工具

实时智能体支持在实时对话期间使用函数工具：

```python
from agents.decorators import tool


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"The weather in {city} is sunny, 72F."


agent = RealtimeAgent(
    name="Assistant",
    instructions="You can answer weather questions.",
    tools=[get_weather],
)
```

### 工具审批

函数工具可以要求在执行前获得人工审批。出现这种情况时，会话会发出 `tool_approval_required`，并暂停工具运行，直到你调用 `approve_tool_call()` 或 `reject_tool_call()`。

如果工具还配置了输入安全防护措施，这些安全防护措施会在获得审批后、执行前立即运行。若要在发出审批事件前运行它们，请使用 `RealtimeRunner(..., config={"tool_execution": {"pre_approval_tool_input_guardrails": True}})` 创建运行器。通过此次审批前检查的调用，在获得审批后、执行前仍会再次接受检查。

```python
async for event in session:
    if event.type == "tool_approval_required":
        await session.approve_tool_call(event.call_id)
```

有关具体的服务端审批循环，请参阅 [`examples/realtime/app/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/app/server.py)。人工介入文档中的[人工介入](../human_in_the_loop.md)也会引用此流程。

### 任务转移

实时任务转移允许一个智能体将实时对话移交给另一个专用智能体：

```python
from agents.realtime import RealtimeAgent, realtime_handoff

billing_agent = RealtimeAgent(
    name="Billing Support",
    instructions="You specialize in billing issues.",
)

main_agent = RealtimeAgent(
    name="Customer Service",
    instructions="Triage the request and hand off when needed.",
    handoffs=[
        realtime_handoff(
            billing_agent,
            tool_description_override="Transfer to billing support",
        )
    ],
)
```

直接用作任务转移的 `RealtimeAgent` 对象会被自动封装，而 `realtime_handoff(...)` 可用于自定义名称、描述、验证、回调和可用性。实时任务转移**不**支持常规任务转移的 `input_filter`。

### 安全防护措施

实时智能体支持对智能体响应实施输出安全防护措施，以及对函数工具调用实施输入安全防护措施。输出安全防护措施检查采用防抖机制：每次检查都会基于累积的输出文本和音频转录增量运行，而不是针对每个局部增量运行，并且会发出 `guardrail_tripped`，而不是抛出异常。

```python
from agents.guardrail import GuardrailFunctionOutput, OutputGuardrail


def sensitive_data_check(context, agent, output):
    return GuardrailFunctionOutput(
        tripwire_triggered="password" in output,
        output_info=None,
    )


agent = RealtimeAgent(
    name="Assistant",
    instructions="...",
    output_guardrails=[OutputGuardrail(guardrail_function=sensitive_data_check)],
)
```

当实时输出安全防护措施因音频转录而触发时，会话会中断当前响应、强制执行 `response.cancel`、发出 `guardrail_tripped`，并发送一条后续用户消息，其中指明被触发的安全防护措施，以便模型生成替代响应。音频播放器仍应监听 `audio_interrupted` 并立即停止本地播放，因为触发机制启动时可能已有部分音频进入缓冲区。使用内置的 OpenAI Realtime 传输时，如果安全防护措施检查在其所检查的响应结束后才完成，会话只会中断该响应的缓冲播放，而不会取消之后启动的任何响应。对于纯文本输出，会话会改为发送一个响应范围内的 `response.cancel`；由于没有音频播放需要停止，因此不会发出 `audio_interrupted`。使用内置 OpenAI Realtime 模型时，纯文本路径也会发出相同的 `guardrail_tripped` 事件和后续用户消息。

自定义 `RealtimeModel` 传输必须遵循 `RealtimeModelSendInterrupt.response_id` 和 `playback_only`，以提供相同的、限定来源范围的音频中断行为。它们还必须重写 `RealtimeModel.send_event_if()`，以支持纯文本输出路径的恢复消息。实现必须在传输的实际事件提交边界重新检查所提供的条件，或者将条件检查与事件提交串行化。默认实现会安全地跳过恢复消息，因为如果它只检查一次条件，随后再单独发送事件，那么在条件检查与事件提交之间可能会启动另一个响应；响应取消和 `guardrail_tripped` 事件仍会发生。

## SIP 与电话

Python SDK 通过 [`OpenAIRealtimeSIPModel`][agents.realtime.openai_realtime.OpenAIRealtimeSIPModel] 提供原生 SIP 挂接流程。

当呼叫通过 Realtime Calls API 到达，并且你希望将智能体会话挂接到生成的 `call_id` 时，请使用此流程：

```python
from agents.realtime import RealtimeRunner
from agents.realtime.openai_realtime import OpenAIRealtimeSIPModel

runner = RealtimeRunner(starting_agent=agent, model=OpenAIRealtimeSIPModel())

async with await runner.run(
    model_config={
        "call_id": call_id_from_webhook,
    }
) as session:
    async for event in session:
        ...
```

如果需要先接听呼叫，并希望接听请求载荷与从智能体派生的会话配置一致，请使用 `OpenAIRealtimeSIPModel.build_initial_session_payload(...)`。完整流程可参阅 [`examples/realtime/twilio_sip/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio_sip/server.py)。

## 底层访问与自定义端点

可以通过 `session.model` 访问底层传输对象。

以下情况可使用此对象：

-   通过 `session.model.add_listener(...)` 添加自定义监听器
-   发送原始客户端事件，例如 `response.create` 或 `session.update`
-   通过 `model_config` 自定义 `url`、`headers` 或 `api_key` 的处理方式
-   使用 `call_id` 挂接到现有实时呼叫

`RealtimeModelConfig` 支持：

-   `api_key`
-   `url`
-   `headers`
-   `initial_model_settings`
-   `playback_tracker`
-   `call_id`

此代码仓库提供的 `call_id` 代码示例使用 SIP。更广泛的 Realtime API 还会在某些服务端控制流程中使用 `call_id`，但这里并未将其作为 Python 代码示例提供。

连接 Azure OpenAI时，请传入正式发布版 Realtime 端点 URL 和显式请求头。例如：

```python
session = await runner.run(
    model_config={
        "url": "wss://<your-resource>.openai.azure.com/openai/v1/realtime?model=<deployment-name>",
        "headers": {"api-key": "<your-azure-api-key>"},
    }
)
```

对于基于 token 的身份验证，请在 `headers` 中使用 Bearer token：

```python
session = await runner.run(
    model_config={
        "url": "wss://<your-resource>.openai.azure.com/openai/v1/realtime?model=<deployment-name>",
        "headers": {"authorization": f"Bearer {token}"},
    }
)
```

如果传入 `headers`，SDK 不会自动添加 `Authorization`。实时智能体应避免使用旧版 beta 路径（`/openai/realtime?api-version=...`）。

## 延伸阅读

-   [实时传输](transport.md)
-   [快速入门](quickstart.md)
-   [OpenAI Realtime对话](https://developers.openai.com/api/docs/guides/realtime-conversations/)
-   [OpenAI Realtime服务端控制](https://developers.openai.com/api/docs/guides/realtime-server-controls/)
-   [`examples/realtime`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime)