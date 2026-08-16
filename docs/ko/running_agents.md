---
search:
  exclude: true
---
# 에이전트 실행

[`Runner`][agents.run.Runner] 클래스를 통해 에이전트를 실행할 수 있습니다. 다음 3가지 옵션이 있습니다.

1. [`Runner.run()`][agents.run.Runner.run]: 비동기로 실행되며 [`RunResult`][agents.result.RunResult]를 반환합니다.
2. [`Runner.run_sync()`][agents.run.Runner.run_sync]: 동기 메서드이며 내부적으로 `.run()`을 실행합니다.
3. [`Runner.run_streamed()`][agents.run.Runner.run_streamed]: 비동기로 실행되며 [`RunResultStreaming`][agents.result.RunResultStreaming]을 반환합니다. LLM을 스트리밍 모드로 호출하고 이벤트가 수신되는 즉시 스트리밍합니다.

```python
from agents import Agent, Runner

async def main():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant")

    result = await Runner.run(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)
    # Code within the code,
    # Functions calling themselves,
    # Infinite loop's dance
```

자세한 내용은 [결과 가이드](results.md)를 참조하세요.

## Runner 수명 주기 및 구성

### 에이전트 루프

위의 세 가지 `Runner` 메서드 중 하나를 호출할 때 시작 에이전트와 입력을 전달합니다. 입력은 다음 중 하나일 수 있습니다.

-   문자열(사용자 메시지로 처리)
-   OpenAI Responses API 형식의 입력 항목 목록
-   일시 중지된 실행 또는 `cancel(mode="after_turn")`로 중단된 실행을 재개할 때 사용하는 [`RunState`][agents.run_state.RunState]. 이 상태에는 [다음 재개 모델 호출을 위해 준비된 입력](results.md#add-input-before-resuming)도 포함할 수 있습니다.

그런 다음 Runner는 다음 루프를 실행합니다.

1. 현재 입력과 함께 현재 에이전트에 대해 LLM을 호출합니다.
2. LLM이 출력을 생성합니다.
    1. Runner가 LLM의 출력을 최종 출력으로 분류하면 루프가 종료되고 결과를 반환합니다.
    2. LLM이 핸드오프를 요청하면 현재 에이전트와 입력을 업데이트하고 루프를 다시 실행합니다.
    3. LLM이 도구 호출을 생성하면 해당 도구 호출을 실행하고 결과를 추가한 후 루프를 다시 실행합니다.
3. 전달된 `max_turns`을 초과하면 [`MaxTurnsExceeded`][agents.exceptions.MaxTurnsExceeded] 예외가 발생합니다. 이 턴 제한을 비활성화하려면 `max_turns=None`을 전달하세요.

!!! note

    LLM 출력이 "최종 출력"으로 간주되는 기준은 원하는 유형의 텍스트 출력을 생성하며 도구 호출이 없는 경우입니다.

### 스트리밍

스트리밍을 사용하면 LLM이 실행되는 동안 스트리밍 이벤트도 수신할 수 있습니다. 스트림이 완료되면 [`RunResultStreaming`][agents.result.RunResultStreaming]에 생성된 모든 새 출력을 포함한 전체 실행 정보가 포함됩니다. 스트리밍 이벤트에는 `.stream_events()`을 호출할 수 있습니다. 자세한 내용은 [스트리밍 가이드](streaming.md)를 참조하세요.

#### Responses WebSocket 전송(선택적 헬퍼)

OpenAI Responses WebSocket 전송을 활성화해도 일반 `Runner` API를 계속 사용할 수 있습니다. 연결 재사용을 위해 WebSocket 세션 헬퍼를 사용하는 것이 권장되지만 필수는 아닙니다.

이는 WebSocket 전송을 통한 Responses API이며 [Realtime API](realtime/guide.md)가 아닙니다.

전송 선택 규칙과 구체적인 모델 객체 또는 사용자 지정 프로바이더 관련 주의 사항은 [모델](models/index.md#responses-websocket-transport)을 참조하세요.

##### 패턴 1: 세션 헬퍼 미사용

WebSocket 전송만 사용하고 SDK가 공유 프로바이더나 세션을 관리할 필요가 없을 때 사용합니다.

```python
import asyncio

from agents import Agent, Runner, set_default_openai_responses_transport


async def main():
    set_default_openai_responses_transport("websocket")

    agent = Agent(name="Assistant", instructions="Be concise.")
    result = Runner.run_streamed(agent, "Summarize recursion in one sentence.")

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            continue
        print(event.type)


asyncio.run(main())
```

이 패턴은 단일 실행에 적합합니다. `Runner.run()` / `Runner.run_streamed()`을 반복해서 호출하면 동일한 `RunConfig` / 프로바이더 인스턴스를 직접 재사용하지 않는 한 실행할 때마다 다시 연결될 수 있습니다.

##### 패턴 2: `responses_websocket_session()` 사용(다중 턴 재사용에 권장)

여러 실행에 걸쳐 WebSocket을 지원하는 공유 프로바이더와 `RunConfig`을 사용하려면 [`responses_websocket_session()`][agents.responses_websocket_session]을 사용하세요. 동일한 `run_config`을 상속하는 중첩된 에이전트-도구 호출도 포함됩니다.

```python
import asyncio

from agents import Agent, responses_websocket_session


async def main():
    agent = Agent(name="Assistant", instructions="Be concise.")

    async with responses_websocket_session(
        responses_websocket_options={"ping_interval": 20.0, "ping_timeout": 60.0},
    ) as ws:
        first = ws.run_streamed(agent, "Say hello in one short sentence.")
        async for _event in first.stream_events():
            pass

        second = ws.run_streamed(
            agent,
            "Now say goodbye.",
            previous_response_id=first.last_response_id,
        )
        async for _event in second.stream_events():
            pass


asyncio.run(main())
```

컨텍스트가 종료되기 전에 스트리밍된 결과를 모두 소비하세요. WebSocket 요청이 아직 진행 중인 상태에서 컨텍스트를 종료하면 공유 연결이 강제로 닫힐 수 있습니다.

서비스는 각 WebSocket 연결에서 한 번에 하나의 응답을 처리하며 연결 시간을 60분으로 제한합니다. 헬퍼는 연결을 재사용하지만 이러한 제약을 제거하지는 않습니다. 재연결 후 `store=False` 및 ZDR 흐름에서는 캐시되지 않은 `previous_response_id`을 복구할 수 없습니다. 전체 입력 컨텍스트로 새 체인을 시작하거나 로컬에서 관리하는 세션 상태를 바탕으로 다시 구성하세요. 전체 복구 동작은 [Responses WebSocket 전송 참고 사항](models/index.md#responses-websocket-transport)을 참조하세요.

긴 추론 턴에서 WebSocket 연결 유지 시간 초과가 발생하면 `ping_timeout`을 늘리거나 `ping_timeout=None`으로 설정하여 하트비트 시간 초과를 비활성화하세요. WebSocket 지연 시간보다 안정성이 더 중요한 실행에는 HTTP/SSE 전송을 사용하세요.

### 실행 구성

`run_config` 매개변수를 사용하면 에이전트 실행의 일부 전역 설정을 구성할 수 있습니다.

#### 일반적인 실행 구성 카테고리

각 에이전트 정의를 변경하지 않고 단일 실행의 동작을 재정의하려면 `RunConfig`을 사용하세요.

##### 모델, 프로바이더 및 세션 기본값

-   [`model`][agents.run.RunConfig.model]: 각 에이전트가 보유한 `model`과 관계없이 사용할 전역 LLM 모델을 설정할 수 있습니다.
-   [`model_provider`][agents.run.RunConfig.model_provider]: 모델 이름을 조회하기 위한 모델 프로바이더이며 기본값은 OpenAI입니다.
-   [`model_settings`][agents.run.RunConfig.model_settings]: 에이전트별 설정을 재정의합니다. 예를 들어 전역 `temperature` 또는 `top_p`을 설정할 수 있습니다.
-   [`session_settings`][agents.run.RunConfig.session_settings]: 실행 중 기록을 가져올 때 세션 수준 기본값(예: `SessionSettings(limit=...)`)을 재정의합니다.
-   [`session_input_callback`][agents.run.RunConfig.session_input_callback]: Sessions를 사용할 때 각 `Runner` 실행 전에 새 사용자 입력을 세션 기록과 병합하는 방식을 사용자 지정합니다. 콜백은 동기 또는 비동기일 수 있습니다.

##### 가드레일, 핸드오프 및 모델 입력 구성

-   [`input_guardrails`][agents.run.RunConfig.input_guardrails], [`output_guardrails`][agents.run.RunConfig.output_guardrails]: 모든 실행에 포함할 입력 또는 출력 가드레일 목록입니다.
-   [`handoff_input_filter`][agents.run.RunConfig.handoff_input_filter]: 핸드오프에 자체 입력 필터가 아직 없는 경우 모든 핸드오프에 적용할 전역 입력 필터입니다. 입력 필터를 사용하면 새 에이전트로 전송되는 입력을 편집할 수 있습니다. 자세한 내용은 [`Handoff.input_filter`][agents.handoffs.Handoff.input_filter] 문서를 참조하세요.
-   [`nest_handoff_history`][agents.run.RunConfig.nest_handoff_history]: 다음 에이전트를 호출하기 전에 손실 없이 보존되는 메시지 항목을 원래 위치에 유지하면서 요약 가능한 기록을 순서가 지정된 어시스턴트 요약 세그먼트로 압축하는 옵트인 베타 기능입니다. 중첩된 핸드오프를 안정화하는 동안에는 기본적으로 비활성화되어 있습니다. 활성화하려면 `True`으로 설정하고, raw 트랜스크립트를 그대로 전달하려면 `False`으로 두세요. Sessions, `RunState`, `RunResult.to_input_list()`은 SDK 기본 중첩 기록에 이미 포함된 동일한 메시지 항목을 두 번 추가하지 않으면서도 서로 별개인 동일 메시지는 보존합니다. [Runner 메서드][agents.run.Runner]는 명시적으로 전달하지 않으면 모두 자동으로 `RunConfig`을 생성하므로 빠른 시작과 코드 예제에서는 기본값이 비활성화된 상태로 유지되며, 명시적인 [`Handoff.input_filter`][agents.handoffs.Handoff.input_filter] 콜백은 계속 이 설정을 재정의합니다. 개별 핸드오프는 [`Handoff.nest_handoff_history`][agents.handoffs.Handoff.nest_handoff_history]를 통해 이 설정을 재정의할 수 있습니다.
-   [`handoff_history_mapper`][agents.run.RunConfig.handoff_history_mapper]: `nest_handoff_history`을 옵트인할 때마다 정규화된 트랜스크립트(기록 + 핸드오프 항목)를 받는 선택적 호출 가능 객체입니다. 전체 핸드오프 필터를 작성하지 않고도 기본 제공 순서형 요약 세그먼트를 대체하며, 다음 에이전트로 전달할 정확한 입력 항목 목록을 반환해야 합니다.
-   [`call_model_input_filter`][agents.run.RunConfig.call_model_input_filter]: 모델 호출 직전에 완전히 준비된 모델 입력(instructions 및 입력 항목)을 편집하는 훅입니다. 예를 들어 기록을 줄이거나 시스템 프롬프트를 삽입할 수 있습니다.
-   [`reasoning_item_id_policy`][agents.run.RunConfig.reasoning_item_id_policy]: Runner가 이전 출력을 다음 턴의 모델 입력으로 변환할 때 추론 항목 ID를 보존할지 생략할지 제어합니다.

##### 트레이싱 및 관측 가능성

-   [`tracing_disabled`][agents.run.RunConfig.tracing_disabled]: 전체 실행에서 [트레이싱](tracing.md)을 비활성화할 수 있습니다.
-   [`tracing`][agents.run.RunConfig.tracing]: 실행별 트레이싱 API 키와 같은 트레이스 내보내기 설정을 재정의하려면 [`TracingConfig`][agents.tracing.TracingConfig]을 전달합니다.
-   [`trace_include_sensitive_data`][agents.run.RunConfig.trace_include_sensitive_data]: 트레이스에 LLM 및 도구 호출 입력/출력과 같이 잠재적으로 민감한 데이터를 포함할지 구성합니다.
-   [`workflow_name`][agents.run.RunConfig.workflow_name], [`trace_id`][agents.run.RunConfig.trace_id], [`group_id`][agents.run.RunConfig.group_id]: 실행의 트레이싱 워크플로 이름, 트레이스 ID 및 트레이스 그룹 ID를 설정합니다. 최소한 `workflow_name`은 설정하는 것이 좋습니다. 그룹 ID는 여러 실행의 트레이스를 연결할 수 있는 선택적 필드입니다.
-   [`trace_metadata`][agents.run.RunConfig.trace_metadata]: 모든 트레이스에 포함할 메타데이터입니다.

##### 도구 실행, 승인 및 도구 오류 동작

-   [`tool_execution`][agents.run.RunConfig.tool_execution]: 한 번에 실행되는 로컬 함수 도구 호출 수 제한 등 로컬 도구 호출의 SDK 측 실행 동작을 구성합니다.
-   [`tool_not_found_behavior`][agents.run.RunConfig.tool_not_found_behavior]: 모델이 생성한 함수 도구 호출의 도구 이름이 현재 에이전트에서 사용할 수 있는 함수 도구와 일치하지 않을 때 Runner가 처리하는 방식을 구성합니다. 기본적으로 `ModelBehaviorError`이 발생합니다. 대신 모델에 표시되는 오류 출력을 반환하려면 옵트인하세요.
-   [`tool_name_collision_policy`][agents.run.RunConfig.tool_name_collision_policy]: 네임스페이스가 없는 함수 도구 이름과 핸드오프 이름이 충돌할 때 Runner가 처리하는 방식을 구성합니다. 기본값인 `"warn"`은 조치 가능한 경고를 기록하고 현재 디스패치 대상으로 선택된 항목만 노출합니다. `"error"`은 모델 호출 전에 `UserError`을 발생시킵니다. 네임스페이스가 지정되었거나 지연 로딩되는 도구에 대한 엄격한 검증은 변경되지 않습니다.
-   [`tool_error_formatter`][agents.run.RunConfig.tool_error_formatter]: 승인 거부 및 옵트인된 도구 미발견 출력처럼 모델에 표시되는 도구 오류 메시지를 사용자 지정합니다.

중첩된 핸드오프는 옵트인 베타 기능으로 제공됩니다. `RunConfig(nest_handoff_history=True)`을 전달하여 순서가 지정된 트랜스크립트 압축을 활성화하거나, 특정 핸드오프에서 사용하려면 `handoff(..., nest_handoff_history=True)`을 설정하세요. 기본 제공 매퍼는 전체 트랜스크립트를 하나의 메시지로 축소하는 대신 손실 없이 보존되는 메시지 항목 주위에 생성된 어시스턴트 요약 세그먼트를 배치합니다. raw 트랜스크립트를 유지하려면(기본값) 플래그를 설정하지 않거나 대화를 필요한 방식 그대로 전달하는 `handoff_input_filter`(또는 `handoff_history_mapper`)을 제공하세요. 사용자 지정 매퍼를 작성하지 않고 생성된 요약 세그먼트에 사용되는 래퍼 텍스트를 변경하려면 [`set_conversation_history_wrappers`][agents.handoffs.set_conversation_history_wrappers]을 호출하세요. 기본값을 복원하려면 [`reset_conversation_history_wrappers`][agents.handoffs.reset_conversation_history_wrappers]을 호출합니다.

#### 실행 구성 세부 정보

##### `tool_execution`

실행에서 로컬 함수 도구의 동시 실행 수를 제한하는 등 로컬 함수 도구의 SDK 측 동작을 구성하려면 `tool_execution`을 사용하세요.

```python
from agents import Agent, RunConfig, Runner, ToolExecutionConfig

agent = Agent(name="Assistant", tools=[...])

result = await Runner.run(
    agent,
    "Run the required tool calls.",
    run_config=RunConfig(
        tool_execution=ToolExecutionConfig(
            max_function_tool_concurrency=2,
            pre_approval_tool_input_guardrails=True,
        ),
    ),
)
```

`max_function_tool_concurrency=None`은 기본 동작을 유지합니다. 모델이 한 턴에서 여러 함수 도구 호출을 생성하면 SDK는 생성된 모든 로컬 함수 도구 호출을 시작합니다. 동시에 실행되는 로컬 함수 도구 호출 수를 제한하려면 정수 값을 설정하세요.

이는 프로바이더 측 [`ModelSettings.parallel_tool_calls`][agents.model_settings.ModelSettings.parallel_tool_calls]과 별개입니다. `parallel_tool_calls`은 모델이 단일 응답에서 여러 도구 호출을 생성할 수 있는지 제어합니다. `tool_execution.max_function_tool_concurrency`은 모델이 로컬 함수 도구 호출을 생성한 후 SDK가 이를 실행하는 방식을 제어합니다.

`pre_approval_tool_input_guardrails=False`은 기본 승인 흐름을 유지합니다. 함수 도구에 승인이 필요하면 먼저 실행이 일시 중지되고, 도구 입력 가드레일은 승인 후 실행 직전에만 실행됩니다. 대기 중인 승인 인터럽션(중단 처리)이 발생하기 전에 함수 도구 입력 가드레일을 실행하려면 `True`으로 설정하세요. 이 사전 승인 검사를 통과한 호출에서도 승인 후 동일한 입력 가드레일이 다시 실행되므로 시간에 민감한 검사는 실행 전에 다시 검증됩니다.

##### `tool_not_found_behavior`

기본적으로 모델이 현재 에이전트에서 사용할 수 있는 함수 도구와 일치하지 않는 함수 도구 호출을 생성하면 Runner는 `ModelBehaviorError`을 발생시킵니다.

실행을 복구 가능한 상태로 유지하려면 `tool_not_found_behavior="return_error_to_model"`을 설정하세요. 이 모드에서 SDK는 확인되지 않은 도구 호출에 `function_call_output`을 추가하고 모델을 다시 실행하므로, 모델이 사용 가능한 도구를 선택하거나 해당 도구를 사용하지 않고 응답할 수 있습니다.

```python
from agents import Agent, RunConfig, Runner

agent = Agent(name="Assistant", tools=[...])

result = await Runner.run(
    agent,
    "Handle this request with the available tools.",
    run_config=RunConfig(tool_not_found_behavior="return_error_to_model"),
)
```

현재 이 옵션은 도구 이름 조회에 실패한 함수 도구 호출에만 적용됩니다. 그 외 잘못된 도구 페이로드에는 기존 오류 동작이 계속 적용됩니다.

##### `tool_error_formatter`

SDK가 모델에 표시되는 도구 오류 출력을 생성할 때 모델에 반환되는 메시지를 사용자 지정하려면 `tool_error_formatter`을 사용하세요.

포매터는 다음 항목이 포함된 [`ToolErrorFormatterArgs`][agents.run_config.ToolErrorFormatterArgs]를 받습니다.

-   `kind`: `"approval_rejected"` 또는 `"tool_not_found"`과 같은 오류 카테고리
-   `tool_type`: 도구 런타임(`"function"`, `"computer"`, `"shell"`, `"apply_patch"` 또는 `"custom"`)
-   `tool_name`: 도구 이름
-   `call_id`: 도구 호출 ID
-   `default_message`: SDK의 기본 모델 표시 메시지
-   `run_context`: 활성 실행 컨텍스트 래퍼

메시지를 대체할 문자열을 반환하거나 SDK 기본값을 사용하려면 `None`을 반환하세요.

```python
from agents import Agent, RunConfig, Runner, ToolErrorFormatterArgs


def format_rejection(args: ToolErrorFormatterArgs[None]) -> str | None:
    if args.kind == "approval_rejected":
        return (
            f"Tool call '{args.tool_name}' was rejected by a human reviewer. "
            "Ask for confirmation or propose a safer alternative."
        )
    if args.kind == "tool_not_found":
        return f"Tool '{args.tool_name}' is not available. Choose one of the listed tools."
    return None


agent = Agent(name="Assistant")
result = Runner.run_sync(
    agent,
    "Please delete the production database.",
    run_config=RunConfig(tool_error_formatter=format_rejection),
)
```

##### `reasoning_item_id_policy`

`reasoning_item_id_policy`은 Runner가 기록을 다음 턴으로 전달할 때(예: `RunResult.to_input_list()` 또는 세션 기반 실행을 사용할 때) 추론 항목을 다음 턴의 모델 입력으로 변환하는 방식을 제어합니다.

-   `None` 또는 `"preserve"`(기본값): 추론 항목 ID를 유지합니다.
-   `"omit"`: 생성된 다음 턴 입력에서 추론 항목 ID를 제거합니다.

추론 항목이 `id`과 함께 전송되지만 필수 후속 항목(예: `Item 'rs_...' of type 'reasoning' was provided without its required following item.`)이 없는 경우에 발생하는 Responses API 400 오류 유형을 옵트인 방식으로 완화하려면 주로 `"omit"`을 사용하세요.

SDK가 이전 출력에서 후속 입력을 구성하는 다중 턴 에이전트 실행에서 이러한 상황이 발생할 수 있습니다. 여기에는 세션 지속성, 서버 관리 대화 델타, 스트리밍/비스트리밍 후속 턴 및 재개 경로가 포함됩니다. 추론 항목 ID는 보존되지만 프로바이더가 해당 ID와 그에 대응하는 후속 항목을 함께 유지하도록 요구하는 경우입니다.

`reasoning_item_id_policy="omit"`을 설정하면 추론 내용은 유지하지만 추론 항목의 `id`을 제거하므로 SDK가 생성한 후속 입력이 해당 API 불변 조건을 위반하지 않습니다.

적용 범위 참고 사항:

-   SDK가 후속 입력을 구성할 때 생성하거나 전달하는 추론 항목만 변경합니다.
-   사용자가 제공한 초기 입력 항목은 다시 작성하지 않습니다.
-   `call_model_input_filter`은 이 정책이 적용된 후에도 의도적으로 추론 ID를 다시 추가할 수 있습니다.

## 상태 및 대화 관리

### 메모리 전략 선택

다음 턴으로 상태를 전달하는 일반적인 방법은 네 가지입니다.

| 전략 | 상태가 저장되는 위치 | 적합한 용도 | 다음 턴에 전달하는 항목 |
| --- | --- | --- | --- |
| `result.to_input_list()` | 애플리케이션 메모리 | 소규모 채팅 루프, 완전한 수동 제어, 모든 프로바이더 | `result.to_input_list()`의 목록과 다음 사용자 메시지 |
| `session` | 자체 스토리지 및 SDK | 지속형 채팅 상태, 재개 가능한 실행, 사용자 지정 저장소 | 동일한 `session` 인스턴스 또는 동일한 저장소를 가리키는 다른 인스턴스 |
| `conversation_id` | OpenAI Conversations API | 여러 워커 또는 서비스에서 공유할 명명된 서버 측 대화 | 동일한 `conversation_id`과 새 사용자 턴만 전달 |
| `previous_response_id` | OpenAI Responses API | 대화 리소스를 생성하지 않는 경량 서버 관리형 연속 처리 | `result.last_response_id`과 새 사용자 턴만 전달 |

`result.to_input_list()`과 `session`은 클라이언트에서 관리됩니다. `conversation_id`과 `previous_response_id`은 OpenAI에서 관리되며 OpenAI Responses API를 사용할 때만 적용됩니다. 대부분의 애플리케이션에서는 대화마다 하나의 지속성 전략을 선택하세요. 클라이언트 관리 기록과 OpenAI 관리 상태를 함께 사용하면 두 계층을 의도적으로 조정하지 않는 한 컨텍스트가 중복될 수 있습니다.

!!! note

    세션 지속성은 서버 관리 대화 설정
    (`conversation_id`, `previous_response_id` 또는 `auto_previous_response_id`)과 동일한
    실행에서 함께 사용할 수 없습니다. 호출마다 하나의 방식을 선택하세요.

### 대화/채팅 스레드

실행 메서드 중 하나를 호출하면 하나 이상의 에이전트가 실행될 수 있고, 그에 따라 하나 이상의 LLM 호출이 발생할 수 있지만, 채팅 대화에서는 논리적으로 하나의 턴을 나타냅니다. 예를 들면 다음과 같습니다.

1. 사용자 턴: 사용자가 텍스트를 입력합니다.
2. Runner 실행: 첫 번째 에이전트가 LLM을 호출하고 도구를 실행한 뒤 두 번째 에이전트로 핸드오프합니다. 두 번째 에이전트는 추가 도구를 실행한 다음 출력을 생성합니다.

에이전트 실행이 끝나면 사용자에게 표시할 내용을 선택할 수 있습니다. 예를 들어 에이전트가 생성한 모든 새 항목을 사용자에게 표시하거나 최종 출력만 표시할 수 있습니다. 어느 쪽이든 사용자가 후속 질문을 할 수 있으며, 이 경우 실행 메서드를 다시 호출할 수 있습니다.

#### 수동 대화 관리

[`RunResultBase.to_input_list()`][agents.result.RunResultBase.to_input_list] 메서드를 사용하여 다음 턴의 입력을 가져오고 대화 기록을 수동으로 관리할 수 있습니다.

```python
from agents import Agent, Runner, trace

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    thread_id = "thread_123"  # Example thread ID
    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result = await Runner.run(agent, "What city is the Golden Gate Bridge in?")
        print(result.final_output)
        # San Francisco

        # Second turn
        new_input = result.to_input_list() + [{"role": "user", "content": "What state is it in?"}]
        result = await Runner.run(agent, new_input)
        print(result.final_output)
        # California
```

#### 세션을 사용한 자동 대화 관리

더 간단한 방법으로, `.to_input_list()`을 수동으로 호출하지 않고 [Sessions](sessions/index.md)를 사용하여 대화 기록을 자동으로 처리할 수 있습니다.

```python
from agents import Agent, Runner, SQLiteSession, trace

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    # Create session instance
    session = SQLiteSession("conversation_123")

    thread_id = "thread_123"  # Example thread ID
    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result = await Runner.run(agent, "What city is the Golden Gate Bridge in?", session=session)
        print(result.final_output)
        # San Francisco

        # Second turn - agent automatically remembers previous context
        result = await Runner.run(agent, "What state is it in?", session=session)
        print(result.final_output)
        # California
```

Sessions는 자동으로 다음 작업을 수행합니다.

-   각 실행 전에 대화 기록 검색
-   각 실행 후 새 메시지 저장
-   서로 다른 세션 ID에 대해 별도 대화 유지

자세한 내용은 [Sessions 문서](sessions/index.md)를 참조하세요.


#### 서버 관리 대화

`to_input_list()` 또는 `Sessions`을 사용하여 로컬에서 처리하는 대신 OpenAI 대화 상태 기능이 서버 측에서 대화 상태를 관리하도록 할 수도 있습니다. 이를 통해 과거의 모든 메시지를 매번 수동으로 다시 전송하지 않고도 대화 기록을 보존할 수 있습니다. 아래의 서버 관리 방식 중 하나를 사용할 때는 각 요청에 새 턴의 입력만 전달하고 저장된 ID를 재사용하세요. 자세한 내용은 [OpenAI 대화 상태 가이드](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses)를 참조하세요.

OpenAI는 턴 사이의 상태를 추적하는 두 가지 방법을 제공합니다.

##### 1. `conversation_id` 사용

먼저 OpenAI Conversations API를 사용하여 대화를 생성한 다음 이후의 모든 호출에서 해당 ID를 재사용합니다.

```python
from agents import Agent, Runner
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    # Create a server-managed conversation
    conversation = await client.conversations.create()
    conv_id = conversation.id

    while True:
        user_input = input("You: ")
        result = await Runner.run(agent, user_input, conversation_id=conv_id)
        print(f"Assistant: {result.final_output}")
```

##### 2. `previous_response_id` 사용

또 다른 옵션은 각 턴을 이전 턴의 응답 ID에 명시적으로 연결하는 **응답 체이닝**입니다.

```python
from agents import Agent, Runner

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    previous_response_id = None

    while True:
        user_input = input("You: ")

        # Setting auto_previous_response_id=True enables response chaining automatically
        # for the first turn, even when there's no actual previous response ID yet.
        result = await Runner.run(
            agent,
            user_input,
            previous_response_id=previous_response_id,
            auto_previous_response_id=True,
        )
        previous_response_id = result.last_response_id
        print(f"Assistant: {result.final_output}")
```

실행이 승인을 위해 일시 중지되고 [`RunState`][agents.run_state.RunState]에서 재개하는 경우 SDK는 저장된 `conversation_id` / `previous_response_id` / `auto_previous_response_id` 설정을 유지하므로 재개된 턴이 동일한 서버 관리 대화에서 계속됩니다.

`conversation_id`과 `previous_response_id`은 상호 배타적입니다. 여러 시스템에서 공유할 수 있는 명명된 대화 리소스가 필요하면 `conversation_id`을 사용하세요. 한 턴에서 다음 턴으로 이어지는 가장 가벼운 Responses API 연속 처리 기본 구성 요소가 필요하면 `previous_response_id`을 사용하세요.

!!! note

    SDK는 `conversation_locked` 오류를 백오프와 함께 자동으로 재시도합니다. 서버 관리
    대화 실행에서는 재시도 전에 내부 대화 추적기 입력을 되돌려 동일하게 준비된
    항목을 다시 올바르게 전송할 수 있도록 합니다.

    로컬 세션 기반 실행(`conversation_id`, `previous_response_id` 또는
    `auto_previous_response_id`과 함께 사용할 수 없음)에서도 SDK는 최근에 지속 저장된
    입력 항목을 최선의 방식으로 롤백하여 재시도 후 기록 항목의 중복을 줄입니다.

    이 호환성 재시도는 `ModelSettings.retry`을 구성하지 않아도 수행됩니다. 모델 요청에
    대해 더 광범위한 옵트인 재시도 동작을 사용하려면 [Runner 관리 재시도](models/index.md#runner-managed-retries)를 참조하세요.

## 훅 및 사용자 지정

### 모델 호출 입력 필터

모델 호출 직전에 모델 입력을 편집하려면 `call_model_input_filter`을 사용하세요. 이 훅은 현재 에이전트, 컨텍스트 및 결합된 입력 항목(세션 기록이 있는 경우 이를 포함)을 받고 새로운 `ModelInputData`을 반환합니다.

반환 값은 [`ModelInputData`][agents.run.ModelInputData] 객체여야 합니다. 해당 객체의 `input` 필드는 필수이며 입력 항목 목록이어야 합니다. 다른 형태를 반환하면 `UserError`이 발생합니다.

```python
from agents import Agent, Runner, RunConfig
from agents.run import CallModelData, ModelInputData

def drop_old_messages(data: CallModelData[None]) -> ModelInputData:
    # Keep only the last 5 items and preserve existing instructions.
    trimmed = data.model_data.input[-5:]
    return ModelInputData(input=trimmed, instructions=data.model_data.instructions)

agent = Agent(name="Assistant", instructions="Answer concisely.")
result = Runner.run_sync(
    agent,
    "Explain quines",
    run_config=RunConfig(call_model_input_filter=drop_old_messages),
)
```

Runner는 준비된 입력 목록의 복사본을 훅에 전달하므로 호출자의 원래 목록을 인플레이스 방식으로 변경하지 않고도 목록을 줄이거나 대체하거나 순서를 변경할 수 있습니다.

세션을 사용하는 경우 `call_model_input_filter`은 세션 기록이 이미 로드되어 현재 턴과 병합된 후에 실행됩니다. 앞선 병합 단계 자체를 사용자 지정하려면 [`session_input_callback`][agents.run.RunConfig.session_input_callback]을 사용하세요.

`conversation_id`, `previous_response_id` 또는 `auto_previous_response_id`을 사용하여 OpenAI 서버 관리 대화 상태를 사용하는 경우 훅은 다음 Responses API 호출을 위해 준비된 페이로드에서 실행됩니다. 해당 페이로드는 이전 기록의 전체 재전송이 아니라 새 턴의 델타만 이미 나타낼 수 있습니다. 반환한 항목만 해당 서버 관리 연속 처리에 전송된 것으로 표시됩니다.

민감한 데이터를 수정하거나, 긴 기록을 줄이거나, 추가 시스템 지침을 삽입하려면 `run_config`을 통해 실행별로 훅을 설정하세요.

## 오류 및 복구

### 오류 처리기

모든 `Runner` 진입점은 오류 종류를 키로 사용하는 dict인 `error_handlers`을 받습니다. 지원되는 키는 `"max_turns"`, `"model_refusal"`, `"invalid_final_output"`입니다. 실행을 해당 오류로 종료하는 대신 제어된 최종 출력을 반환하려면 이를 사용하세요.

```python
from agents import (
    Agent,
    RunErrorHandlerInput,
    RunErrorHandlerResult,
    Runner,
)

agent = Agent(name="Assistant", instructions="Be concise.")


def on_max_turns(_data: RunErrorHandlerInput[None]) -> RunErrorHandlerResult:
    return RunErrorHandlerResult(
        final_output="I couldn't finish within the turn limit. Please narrow the request.",
        include_in_history=False,
    )


result = Runner.run_sync(
    agent,
    "Analyze this long transcript",
    max_turns=3,
    error_handlers={"max_turns": on_max_turns},
)
print(result.final_output)
```

모델 메시지가 에이전트의 structured `output_type`에 대해 검증되지 않거나 모델이 structured 최종 메시지를 반환하지 않는 경우 `"invalid_final_output"`을 사용하세요. 처리기는 애플리케이션별 대체 값을 반환할 수 있으며, SDK는 동일한 `output_type`에 대해 이를 검증합니다. 모델 호출을 재시도하거나 도구의 부작용을 다시 실행하지는 않습니다. `None`을 반환하면 복구를 거부합니다. 대체 값이 없으면 비어 있지 않은 검증 실패에서 계속 `ModelBehaviorError`이 발생하며, 비어 있는 structured 응답은 기존의 다음 턴 동작을 유지합니다.

```python
from pydantic import BaseModel

from agents import Agent, ModelBehaviorError, RunErrorHandlerInput, Runner


class Recipe(BaseModel):
    ingredients: list[str]
    recovered_from_invalid_output: bool = False


def on_invalid_final_output(data: RunErrorHandlerInput[None]) -> Recipe:
    assert isinstance(data.error, ModelBehaviorError)
    return Recipe(ingredients=[], recovered_from_invalid_output=True)


agent = Agent(
    name="Recipe assistant",
    instructions="Return a structured recipe.",
    output_type=Recipe,
)

result = Runner.run_sync(
    agent,
    "Plan tonight's dinner.",
    error_handlers={"invalid_final_output": on_invalid_final_output},
)
print(result.final_output)
```

`RunErrorHandlerResult.include_in_history`의 기본값은 `True`입니다. 최대 턴 처리기에서는 합성된 대체 출력을 대화 기록에 추가하고 구성된 세션에 지속 저장합니다. 대체 출력을 결과 기록이나 세션 저장소에 추가하지 않고 호출자에게 반환하려면 `include_in_history=False`으로 설정하세요.

모델의 거부가 `ModelRefusalError`로 실행을 종료하는 대신 애플리케이션별 대체 출력을 생성하도록 하려면 `"model_refusal"`을 사용하세요.

```python
from pydantic import BaseModel

from agents import Agent, ModelRefusalError, RunErrorHandlerInput, Runner


class Recipe(BaseModel):
    ingredients: list[str]
    refusal_reason: str | None = None


def on_model_refusal(data: RunErrorHandlerInput[None]) -> Recipe:
    assert isinstance(data.error, ModelRefusalError)
    return Recipe(ingredients=[], refusal_reason=data.error.refusal)


agent = Agent(
    name="Recipe assistant",
    instructions="Return a structured recipe.",
    output_type=Recipe,
)

result = Runner.run_sync(
    agent,
    "Make me something unsafe.",
    error_handlers={"model_refusal": on_model_refusal},
)
print(result.final_output)
```

## 내구성 있는 실행 통합 및 휴먼인더루프 (HITL)

도구 승인 일시 중지/재개 패턴은 전용 [휴먼인더루프 가이드](human_in_the_loop.md)에서 시작하세요. 아래 통합은 실행이 긴 대기, 재시도 또는 프로세스 재시작에 걸쳐 지속될 수 있는 내구성 있는 오케스트레이션을 위한 것입니다.

### Dapr

Agents SDK의 [Dapr](https://dapr.io) Diagrid 통합을 사용하면 장애에서 자동으로 복구되고 휴먼인더루프 워크플로를 지원하는 내구성 있는 장기 실행 에이전트를 실행할 수 있습니다. Dapr는 특정 공급업체에 종속되지 않는 [CNCF](https://cncf.io) 워크플로 오케스트레이터입니다. Dapr와 OpenAI 에이전트는 [여기](https://docs.diagrid.io/getting-started/quickstarts/ai-agents/?agentframework=openai)에서 시작할 수 있습니다.

### Temporal

Agents SDK의 [Temporal](https://temporal.io/) 통합을 사용하면 휴먼인더루프 작업을 포함한 내구성 있는 장기 실행 워크플로를 실행할 수 있습니다. Temporal과 Agents SDK가 함께 장기 실행 작업을 완료하는 데모는 [이 동영상](https://www.youtube.com/watch?v=fFBZqzT4DD8)에서 확인할 수 있으며, [문서는 여기](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents)에서 확인할 수 있습니다. 

### Restate

Agents SDK의 [Restate](https://restate.dev/) 통합을 사용하면 사람의 승인, 핸드오프 및 세션 관리를 포함하는 경량의 내구성 있는 에이전트를 실행할 수 있습니다. 이 통합에는 Restate의 단일 바이너리 런타임이 종속성으로 필요하며, 에이전트를 프로세스/컨테이너 또는 서버리스 함수로 실행할 수 있습니다. 자세한 내용은 [개요](https://www.restate.dev/blog/durable-orchestration-for-ai-agents-with-restate-and-openai-sdk)를 읽거나 [문서](https://docs.restate.dev/ai)를 참조하세요.

### DBOS

Agents SDK의 [DBOS](https://dbos.dev/) 통합을 사용하면 장애와 재시작이 발생해도 진행 상태를 보존하는 안정적인 에이전트를 실행할 수 있습니다. 장기 실행 에이전트, 휴먼인더루프 워크플로 및 핸드오프를 지원합니다. 동기 및 비동기 메서드를 모두 지원합니다. 이 통합에는 SQLite 또는 Postgres 데이터베이스만 필요합니다. 자세한 내용은 통합 [저장소](https://github.com/dbos-inc/dbos-openai-agents)와 [문서](https://docs.dbos.dev/integrations/openai-agents)를 참조하세요.

## 예외

SDK는 특정한 경우 예외를 발생시킵니다. 전체 목록은 [`agents.exceptions`][]에서 확인할 수 있습니다. 개요는 다음과 같습니다.

-   [`AgentsException`][agents.exceptions.AgentsException]: SDK가 발생시키는 모든 예외의 기본 클래스입니다. 다른 모든 특정 예외가 파생되는 일반 유형입니다.
-   [`MaxTurnsExceeded`][agents.exceptions.MaxTurnsExceeded]: 에이전트 실행이 `Runner.run`, `Runner.run_sync` 또는 `Runner.run_streamed` 메서드에 전달된 `max_turns` 제한을 초과하면 이 예외가 발생합니다. 이는 지정된 에이전트 루프 턴(LLM 호출) 횟수 내에 에이전트가 작업을 완료하지 못했음을 나타냅니다. 제한을 비활성화하려면 `max_turns=None`으로 설정하세요.
-   [`ModelTimeoutError`][agents.exceptions.ModelTimeoutError]: 모델 호출 시도가 [`ModelSettings.timeout`][agents.model_settings.ModelSettings.timeout]을 초과하면 이 예외가 발생합니다. 적용 범위와 재시도 동작은 [모델 호출 시간 초과](models/index.md#model-call-timeouts)를 참조하세요.
-   [`ModelBehaviorError`][agents.exceptions.ModelBehaviorError]: 기반 모델(LLM)이 예상치 못하거나 잘못된 출력을 생성할 때 이 예외가 발생합니다. 다음 경우가 포함될 수 있습니다.
    -   잘못된 형식의 JSON: 모델이 도구 호출 또는 직접 출력에서 잘못된 JSON 구조를 제공하는 경우이며, 특히 특정 `output_type`이 정의되어 있을 때 발생합니다.
    -   예상치 못한 도구 관련 실패: 모델이 예상된 방식으로 도구를 사용하지 못하는 경우
-   [`ToolTimeoutError`][agents.exceptions.ToolTimeoutError]: 함수 도구 호출이 구성된 시간 제한을 초과하고 도구에서 `timeout_behavior="raise_exception"`을 사용하는 경우 이 예외가 발생합니다.
-   [`UserError`][agents.exceptions.UserError]: SDK를 사용해 코드를 작성하는 사람이 SDK 사용 중 오류를 범하면 이 예외가 발생합니다. 일반적으로 잘못된 코드 구현, 유효하지 않은 구성 또는 SDK API의 오용으로 인해 발생합니다.
-   [`InputGuardrailTripwireTriggered`][agents.exceptions.InputGuardrailTripwireTriggered], [`OutputGuardrailTripwireTriggered`][agents.exceptions.OutputGuardrailTripwireTriggered]: 입력 가드레일의 조건이 충족되면 `InputGuardrailTripwireTriggered`이 발생하고, 출력 가드레일의 조건이 충족되면 `OutputGuardrailTripwireTriggered`이 발생합니다. 입력 가드레일은 처리 전에 수신 메시지를 확인하고, 출력 가드레일은 전달 전에 에이전트의 최종 응답을 확인합니다.