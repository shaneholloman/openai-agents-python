---
search:
  exclude: true
---
# 实时智能体指南

本指南说明OpenAI Agents SDK的实时层如何映射到OpenAI Realtime API，以及Python SDK在此基础上增加了哪些额外行为。

!!! note "从这里开始"

    如果你想使用默认的Python路径，请先阅读[快速入门](quickstart.md)。如果你正在确定应用应使用服务端WebSocket还是SIP，请阅读[实时传输](transport.md)。浏览器WebRTC传输不属于Python SDK的一部分。

## 概述

实时智能体会与Realtime API保持长连接，使模型能够增量处理文本和音频、流式传输音频输出、调用工具，并处理打断，而无需在每一轮都重新发起请求。

主要SDK组件包括：

-   **RealtimeAgent**: 单个实时专用智能体的指令、工具、输出安全防护措施和任务转移
-   **RealtimeRunner**: 将起始智能体连接到实时传输层的会话工厂
-   **RealtimeSession**: 用于发送输入、接收事件、跟踪历史记录和执行工具的实时会话
-   **RealtimeModel**: 传输抽象。默认实现是OpenAI的服务端WebSocket实现。

## 会话生命周期

典型的实时会话如下：

1. 创建一个或多个`RealtimeAgent`。
2. 使用起始智能体创建`RealtimeRunner`。
3. 调用`await runner.run()`以获取`RealtimeSession`。
4. 使用`async with session:`或`await session.enter()`进入会话。
5. 使用`send_message()`或`send_audio()`发送用户输入。
6. 迭代会话事件，直到对话结束。

与纯文本运行不同，`runner.run()`不会立即生成最终结果。它会返回一个实时会话对象，使本地历史记录、后台工具执行、安全防护措施状态和当前智能体配置与传输层保持同步。

默认情况下，`RealtimeRunner`使用`OpenAIRealtimeWebSocketModel`，因此默认Python路径是通过服务端WebSocket连接到Realtime API。如果传入其他`RealtimeModel`，相同的会话生命周期和智能体功能仍然适用，但连接机制可以不同。

## 智能体与会话配置

与常规`Agent`类型相比，`RealtimeAgent`的范围有意设计得更窄：

-   模型选择在会话级别配置，而不是按智能体配置。
-   不支持structured outputs。
-   可以配置语音，但会话生成语音音频后便无法更改。
-   指令、工具调用、任务转移、钩子和输出安全防护措施仍然可用。

`RealtimeSessionModelSettings`既支持较新的嵌套`audio`配置，也支持较旧的扁平别名。新代码应优先使用嵌套结构，并使用`gpt-realtime-2.1`开始构建新的实时智能体：

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

常用的会话级别设置包括：

-   `audio.input.format`, `audio.output.format`
-   `audio.input.transcription`
-   `audio.input.noise_reduction`
-   `audio.input.turn_detection`
-   `audio.output.voice`, `audio.output.speed`
-   `output_modalities`
-   `tool_choice`
-   `prompt`
-   `tracing`

`RealtimeRunner(config=...)`中常用的运行级别设置包括：

-   `async_tool_calls`
-   `output_guardrails`
-   `guardrails_settings.debounce_text_length`
-   `tool_error_formatter`
-   `tracing_disabled`

如需了解完整的类型化接口，请参阅[`RealtimeRunConfig`][agents.realtime.config.RealtimeRunConfig]和[`RealtimeSessionModelSettings`][agents.realtime.config.RealtimeSessionModelSettings]。

## 输入与输出

### 文本与结构化用户消息

使用[`session.send_message()`][agents.realtime.session.RealtimeSession.send_message]发送纯文本或结构化实时消息。

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

结构化消息是在实时对话中加入图像输入的主要方式。[`examples/realtime/app/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/app/server.py)中的Web演示代码通过这种方式转发`input_image`消息。

### 音频输入

使用[`session.send_audio()`][agents.realtime.session.RealtimeSession.send_audio]流式传输原始音频字节：

```python
await session.send_audio(audio_bytes)
```

如果禁用了服务端轮次检测，则需要自行标记轮次边界。高层便捷方法如下：

```python
await session.send_audio(audio_bytes, commit=True)
```

如果需要更底层的控制，也可以通过底层模型传输层发送原始客户端事件，例如`input_audio_buffer.commit`。

### 手动响应控制

`session.send_message()`通过高层路径发送用户输入，并自动启动响应。原始音频缓冲在所有配置中**并不**都会自动执行相同操作。

在Realtime API层面，手动控制轮次意味着通过原始`session.update`清除`turn_detection`，然后自行发送`input_audio_buffer.commit`和`response.create`。

如果你正在手动管理轮次，可以通过模型传输层发送原始客户端事件：

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

-   已禁用`turn_detection`，并且你希望自行决定模型何时响应
-   希望在触发响应前检查或控制用户输入
-   需要为带外响应使用自定义提示词

[`examples/realtime/twilio_sip/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio_sip/server.py)中的SIP代码示例使用原始`response.create`强制生成开场问候语。

## 事件、历史记录与打断

`RealtimeSession`会发出更高层的SDK事件，同时仍会转发原始模型事件，以便在需要时使用。

重要的会话事件包括：

-   `audio`, `audio_end`, `audio_interrupted`
-   `agent_start`, `agent_end`
-   `tool_start`, `tool_end`, `tool_approval_required`
-   `handoff`
-   `history_added`, `history_updated`
-   `guardrail_tripped`
-   `input_audio_timeout_triggered`
-   `error`
-   `raw_model_event`

对UI状态最有用的事件通常是`history_added`和`history_updated`。它们以`RealtimeItem`对象的形式公开会话的本地历史记录，其中包括用户消息、助手消息和工具调用。

### 用量统计

当已完成的模型响应包含用量信息时，OpenAI实时模型会在`raw_model_event`中发出[`RealtimeModelUsageEvent`][agents.realtime.model_events.RealtimeModelUsageEvent]。其`usage`字段包含该响应的令牌计数，而`input_tokens_details`和`output_tokens_details`提供可选的模态细分信息。

会话还会将每个响应的用量添加到共享的[`RunContextWrapper.usage`][agents.run_context.RunContextWrapper.usage]中。可以从后续高层事件（例如`agent_end`）的`event.info.context.usage`中读取该值，以检查实时会话的累计用量。

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

只有当模型提供商在已完成的响应中包含用量信息时，才会报告用量。累计值涵盖该`RealtimeSession`收到的响应，并不是跨会话的总量。

### 打断与播放跟踪

当用户打断助手时，会话会发出`audio_interrupted`并更新历史记录，使服务端对话与用户实际听到的内容保持一致。

对于低延迟本地播放，默认播放跟踪器通常已经足够。在远程或延迟播放场景中，尤其是电话场景，应使用[`RealtimePlaybackTracker`][agents.realtime.model.RealtimePlaybackTracker]，使打断时的截断操作基于实际播放进度，而不是假定所有已生成音频均已播放给用户。

[`examples/realtime/twilio/twilio_handler.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio/twilio_handler.py)中的Twilio代码示例展示了此模式。

## 工具、审批、任务转移与安全防护措施

### 工具调用

实时智能体支持在实时对话期间使用工具调用：

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

工具调用可以要求在执行前进行人工审批。发生这种情况时，会话会发出`tool_approval_required`，并暂停工具执行，直到你调用`approve_tool_call()`或`reject_tool_call()`。

如果工具还具有输入安全防护措施，这些安全防护措施会在审批后、执行前立即运行。若要在发出审批事件前运行它们，请使用`RealtimeRunner(..., config={"tool_execution": {"pre_approval_tool_input_guardrails": True}})`创建运行器。通过此审批前检查的调用，在审批后、执行前仍会再次接受检查。

```python
async for event in session:
    if event.type == "tool_approval_required":
        await session.approve_tool_call(event.call_id)
```

有关具体的服务端审批循环，请参阅[`examples/realtime/app/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/app/server.py)。人工介入文档中的[人工介入](../human_in_the_loop.md)也会引用此流程。

### 任务转移

实时任务转移允许一个智能体将实时对话转交给另一个专用智能体：

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

直接使用的`RealtimeAgent`任务转移会被自动包装，而`realtime_handoff(...)`允许自定义名称、描述、验证、回调和可用性。实时任务转移**不**支持常规任务转移的`input_filter`。

### 安全防护措施

实时智能体支持针对智能体响应的输出安全防护措施，以及针对工具调用的输入安全防护措施。输出安全防护措施会在经过防抖处理的输出文本和音频转录增量累积内容上运行，而不是在每个部分增量上运行；触发时会发出`guardrail_tripped`，而不是抛出异常。

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

当实时输出安全防护措施因音频转录而触发时，会话会打断当前响应，强制发出`response.cancel`，发出`guardrail_tripped`，并发送一条注明已触发安全防护措施的后续用户消息，以便模型生成替代响应。音频播放器仍应监听`audio_interrupted`并立即停止本地播放，因为触发条件生效时，部分音频可能已经进入缓冲区。使用内置OpenAI实时传输实现时，如果安全防护措施在其源响应结束后才完成，会话只会打断该响应的缓冲播放，而不会取消较新的响应。对于纯文本输出，会话会改为发送仅针对该响应的`response.cancel`；由于没有需要停止的音频播放，因此不会发出`audio_interrupted`。使用内置OpenAI实时模型时，纯文本路径也会发出相同的`guardrail_tripped`事件和后续用户消息。

自定义`RealtimeModel`传输实现必须遵循`RealtimeModelSendInterrupt.response_id`和`playback_only`，以提供相同的、限定于源响应的音频打断行为。它们还必须重写`RealtimeModel.send_event_if()`，以支持纯文本恢复消息。实现必须在传输层的实际事件提交边界重新检查给定条件，或对该条件的检查进行串行化。默认实现会安全地跳过恢复消息，因为如果在等待`send_event()`前检查条件，较新的响应可能会在消息提交前启动；响应取消和`guardrail_tripped`事件仍会发生。

## SIP与电话

Python SDK通过[`OpenAIRealtimeSIPModel`][agents.realtime.openai_realtime.OpenAIRealtimeSIPModel]提供一流的SIP附加流程。

当呼叫通过Realtime Calls API到达，并且你希望将智能体会话附加到生成的`call_id`时，请使用该流程：

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

如果需要先接受呼叫，并希望接受载荷与根据智能体生成的会话配置保持一致，请使用`OpenAIRealtimeSIPModel.build_initial_session_payload(...)`。完整流程见[`examples/realtime/twilio_sip/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio_sip/server.py)。

## 底层访问与自定义端点

可以通过`session.model`访问底层传输对象。

以下情况可使用此对象：

-   通过`session.model.add_listener(...)`添加自定义监听器
-   发送原始客户端事件，例如`response.create`或`session.update`
-   通过`model_config`自定义`url`、`headers`或`api_key`处理
-   使用`call_id`附加到现有实时呼叫

`RealtimeModelConfig`支持：

-   `api_key`
-   `url`
-   `headers`
-   `initial_model_settings`
-   `playback_tracker`
-   `call_id`

本仓库随附的`call_id`代码示例使用SIP。更广泛的Realtime API也会将`call_id`用于某些服务端控制流程，但此处并未将这些流程作为Python代码示例提供。

连接Azure OpenAI时，请传入正式版Realtime端点URL和显式标头。例如：

```python
session = await runner.run(
    model_config={
        "url": "wss://<your-resource>.openai.azure.com/openai/v1/realtime?model=<deployment-name>",
        "headers": {"api-key": "<your-azure-api-key>"},
    }
)
```

对于基于令牌的身份验证，请在`headers`中使用Bearer令牌：

```python
session = await runner.run(
    model_config={
        "url": "wss://<your-resource>.openai.azure.com/openai/v1/realtime?model=<deployment-name>",
        "headers": {"authorization": f"Bearer {token}"},
    }
)
```

如果传入`headers`，SDK不会自动添加`Authorization`。使用实时智能体时，请避免使用旧版Beta路径（`/openai/realtime?api-version=...`）。

## 延伸阅读

-   [实时传输](transport.md)
-   [快速入门](quickstart.md)
-   [OpenAI实时对话](https://developers.openai.com/api/docs/guides/realtime-conversations/)
-   [OpenAI实时服务端控制](https://developers.openai.com/api/docs/guides/realtime-server-controls/)
-   [`examples/realtime`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime)