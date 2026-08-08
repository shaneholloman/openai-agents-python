---
search:
  exclude: true
---
# 릴리스 프로세스/변경 로그

이 프로젝트는 `0.Y.Z` 형식을 사용하는, 약간 수정된 유의적 버전 관리를 따릅니다. 맨 앞의 `0`은 SDK가 여전히 빠르게 발전하고 있음을 나타냅니다. 각 구성 요소는 다음과 같이 증가시킵니다.

## 마이너(`Y`) 버전

베타로 표시되지 않은 공개 인터페이스에 **호환성을 깨는 변경 사항**이 발생하면 마이너 버전 `Y`을 증가시킵니다. 예를 들어 `0.0.x`에서 `0.1.x`로 변경될 때 호환성을 깨는 변경 사항이 포함될 수 있습니다.

호환성을 깨는 변경 사항을 원하지 않는다면 프로젝트에서 `0.0.x` 버전으로 고정하는 것이 좋습니다.

## 패치(`Z`) 버전

호환성을 깨지 않는 다음 변경 사항에는 `Z`을 증가시킵니다.

- 버그 수정
- 새로운 기능
- 비공개 인터페이스 변경
- 베타 기능 업데이트

## 호환성 변경 로그

### 0.19.0

이 마이너 릴리스에는 호환성을 깨는 변경 사항이 **없습니다**. 마이너 버전 증가는 OpenAI Responses의 중요한 새 기능 영역인 프로그래매틱 도구 호출을 반영합니다.

주요 내용:

- 지원되는 OpenAI Responses 모델이 프로그래매틱 도구 호출을 사용할 수 있는 도구를 조정하는 JavaScript를 생성할 수 있게 해 주는 [`ProgrammaticToolCallingTool`][agents.tool.ProgrammaticToolCallingTool]이 추가되었습니다. 도구별 `allowed_callers`, `FunctionTool` 인스턴스의 structured outputs, Runner 스트리밍, 가드레일, 승인, 세션 및 `RunState`와의 통합을 지원합니다. 설정 및 제약 조건은 [프로그래매틱 도구 호출](tools.md#programmatic-tool-calling)을 참조하세요.
- 공개 `agents.decorators` 모듈과 기존 `@function_tool` 데코레이터의 짧은 별칭인 `@tool`가 기존 가드레일 데코레이터와 함께 추가되었습니다. 이제 `FunctionTool` 인스턴스는 비동기 호출 가능 객체도 지원합니다.
- 이제 SDK 구성은 에이전트, 실행, 모델, 세션, 샌드박스 및 음성 파이프라인 전반에서 타입이 지정된 설정 객체나 딕셔너리를 일관되게 허용하며, 알 수 없는 설정을 검증합니다.
- 유용한 디버깅 컨텍스트를 유지하면서 가공되지 않은 민감한 페이로드가 노출되지 않도록 모델, 도구, MCP, Realtime, 세션, 샌드박스 및 트레이싱 전반의 오류 및 진단 로깅이 강화되었습니다.
- AnyLLM, LiteLLM 및 Chat Completions 호환성이 개선되었고, 모델 재시도 간에 세션 기록이 유지되며, 응답이 시작되기 전에 발생하는 WebSocket 과부하에 대한 제공업체 재시도 지침이 추가되었습니다. 따라서 명시적으로 활성화한 Runner 재시도 정책은 허용되는 경우 실패한 시도를 다시 실행할 수 있습니다.
- `VercelCloudBucketMountStrategy`을 통해 [Vercel 샌드박스 생성 시에만 구성할 수 있는 S3 마운트](sandbox/clients.md#mounts-and-remote-storage)가 추가되었습니다. 마운트가 적용된 세션에서는 버킷 콘텐츠가 워크스페이스 영속화 대상에서 제외되며, 동적 마운트 변경이나 세션 재개는 의도적으로 지원되지 않습니다.

### 0.18.0

이 마이너 릴리스에는 호환성을 깨는 변경 사항이 **없습니다**. 마이너 버전 증가는 실시간 에이전트의 기본 모델 업데이트만 반영합니다.

주요 내용:

- 이제 실시간 에이전트는 `gpt-realtime-2.1`을 기본 모델로 사용하므로, 새 Realtime 설정에서는 추가 구성 없이 최신 권장 모델을 사용합니다.

### 0.17.0

이 버전에서 샌드박스의 로컬 소스 구체화는 소스 경로가 `Manifest.extra_path_grants`에 포함되지 않는 한 `LocalFile.src`와 `LocalDir.src`을 구체화 `base_dir` 내부에 유지합니다. `base_dir`은 매니페스트가 적용될 때 SDK 프로세스의 현재 작업 디렉터리입니다. 상대 로컬 소스는 해당 디렉터리를 기준으로 해석되며, 절대 경로 로컬 소스는 이미 그 내부에 있거나 명시적인 허용 범위 아래에 있어야 합니다. 이 변경으로 로컬 아티팩트 경계 문제가 해결되지만, 신뢰할 수 있는 호스트 파일이나 디렉터리를 해당 기본 디렉터리 외부에서 샌드박스 워크스페이스로 의도적으로 복사하는 애플리케이션에는 영향을 줄 수 있습니다.

마이그레이션하려면 `SandboxPathGrant`를 사용하여 매니페스트 수준에서 신뢰할 수 있는 호스트 루트를 허용하세요. 샌드박스에서 해당 파일을 읽기만 하면 되는 경우 읽기 전용으로 설정하는 것이 좋습니다.

```python
from pathlib import Path

from agents.sandbox import Manifest, SandboxPathGrant
from agents.sandbox.entries import Dir, LocalDir

# This is an absolute host path outside the SDK process base_dir.
TRUSTED_DOCS_ROOT = Path("/opt/my-app/docs")

manifest = Manifest(
    extra_path_grants=(
        # This host root is outside the SDK process base_dir, so the manifest must grant it.
        SandboxPathGrant(path=str(TRUSTED_DOCS_ROOT), read_only=True),
    ),
    entries={
        # No grant is needed for local sources that stay under the SDK process base_dir.
        "fixtures": LocalDir(src=Path("fixtures"), description="Local test fixtures."),
        # This entry reads from the granted host root and copies it into the sandbox workspace.
        "docs": LocalDir(src=TRUSTED_DOCS_ROOT, description="Trusted local documents."),
        # Dir creates a sandbox workspace directory; it does not read from the host filesystem.
        "output": Dir(description="Generated artifacts."),
    },
)
```

`extra_path_grants`를 신뢰할 수 있는 애플리케이션 구성으로 취급하세요. 애플리케이션에서 해당 호스트 경로를 이미 승인하지 않았다면 모델 출력이나 신뢰할 수 없는 다른 매니페스트 입력으로 허용 범위를 채우지 마세요.

### 0.16.0

이 버전에서 SDK 기본 모델은 이제 `gpt-4.1` 대신 `gpt-5.4-mini`입니다. 이는 모델을 명시적으로 설정하지 않은 에이전트와 실행에 영향을 줍니다. 새 기본값이 GPT-5 모델이므로, 명시하지 않은 기본 모델 설정에는 이제 `reasoning.effort="none"` 및 `verbosity="low"` 같은 GPT-5 기본값이 포함됩니다.

이전 기본 모델 동작을 유지해야 한다면 에이전트 또는 실행 구성에서 모델을 명시적으로 설정하거나 `OPENAI_DEFAULT_MODEL` 환경 변수를 설정하세요.

```python
agent = Agent(name="Assistant", model="gpt-4.1")
```

주요 내용:

- 이제 `Runner.run`, `Runner.run_sync` 및 `Runner.run_streamed`에서 `max_turns=None`을 사용하여 턴 제한을 비활성화할 수 있습니다.
- 이제 로컬, Docker 및 제공업체 기반 샌드박스 구현 전반에서 샌드박스 워크스페이스를 채울 때 절대 심볼릭 링크 대상을 포함하여 아카이브 루트 외부를 가리키는 심볼릭 링크가 있는 tar 아카이브를 거부합니다.

### 0.15.0

이 버전에서는 모델의 거부 응답을 빈 텍스트 출력으로 처리하거나, structured outputs의 경우 실행 루프가 `MaxTurnsExceeded`까지 재시도하도록 하는 대신 이제 `ModelRefusalError`로 명시적으로 노출합니다.

이는 이전에 거부만 포함된 모델 응답이 `final_output == ""`로 완료될 것으로 예상했던 코드에 영향을 줍니다. 예외를 발생시키지 않고 거부를 처리하려면 `model_refusal` 실행 오류 핸들러를 제공하세요.

```python
result = Runner.run_sync(
    agent,
    input,
    error_handlers={"model_refusal": lambda data: data.error.refusal},
)
```

structured outputs 에이전트의 경우 핸들러는 에이전트의 출력 스키마와 일치하는 값을 반환할 수 있으며, SDK는 다른 실행 오류 핸들러의 최종 출력과 동일하게 이를 검증합니다.

### 0.14.0

이 마이너 릴리스에는 호환성을 깨는 변경 사항이 **없지만**, 샌드박스 에이전트라는 중요한 새 베타 기능 영역과 로컬, 컨테이너화 및 호스팅 환경 전반에서 이를 사용하는 데 필요한 런타임, 백엔드 및 문서 지원이 추가되었습니다.

주요 내용:

- `SandboxAgent`, `Manifest` 및 `SandboxRunConfig`을 중심으로 하는 새로운 베타 샌드박스 런타임 인터페이스가 추가되어 에이전트가 파일, 디렉터리, Git 저장소, 마운트, 스냅샷 및 재개 기능을 갖춘 영속적이고 격리된 워크스페이스 내에서 작업할 수 있습니다.
- `UnixLocalSandboxClient` 및 `DockerSandboxClient`을 통해 로컬 및 컨테이너화된 개발을 위한 샌드박스 실행 백엔드가 추가되었으며, Python 패키지의 선택적 의존성 extras를 통해 Blaxel, Cloudflare, Daytona, E2B, Modal, Runloop 및 Vercel용 호스팅 제공업체 통합도 추가되었습니다.
- 이후 실행에서 이전 실행의 교훈을 재사용할 수 있도록 샌드박스 메모리 지원이 추가되었으며, 점진적 공개, 멀티턴 그룹화, 구성 가능한 격리 경계 및 S3 기반 워크플로를 포함한 영속 메모리 코드 예제가 제공됩니다.
- 로컬 및 합성 워크스페이스 항목, S3/R2/GCS/Azure Blob Storage/S3 Files용 원격 스토리지 마운트, 이식 가능한 스냅샷, `RunState`, `SandboxSessionState` 또는 저장된 스냅샷을 통한 재개 흐름을 포함하는 확장된 워크스페이스 및 재개 모델이 추가되었습니다.
- `examples/sandbox/` 아래에 기술을 활용한 코딩 작업, 핸드오프, 메모리, 제공업체별 설정과 코드 검토, 데이터룸 QA 및 웹사이트 복제 같은 엔드투엔드 워크플로를 다루는 다양한 샌드박스 코드 예제와 튜토리얼이 추가되었습니다.
- 샌드박스를 인식하는 세션 준비, 기능 바인딩, 상태 직렬화, 통합 트레이싱, 프롬프트 캐시 키 기본값 및 더 안전한 민감한 MCP 출력 마스킹 기능으로 핵심 런타임 및 트레이싱 스택이 확장되었습니다.

### 0.13.0

이 마이너 릴리스에는 호환성을 깨는 변경 사항이 **없지만**, 주목할 만한 Realtime 기본값 업데이트와 새로운 MCP 기능 및 런타임 안정성 수정이 포함되었습니다.

주요 내용:

- 기본 WebSocket Realtime 모델은 이제 `gpt-realtime-1.5`이므로, 새 Realtime 에이전트 설정에서는 추가 구성 없이 최신 모델을 사용합니다.
- 이제 `MCPServer`은 `list_resources()`, `list_resource_templates()` 및 `read_resource()`을 노출하고, `MCPServerStreamableHttp`는 `session_id`을 노출합니다. 따라서 MCP Streamable HTTP 전송을 사용하는 세션을 재연결이나 상태 비저장 워커 간에 재개할 수 있습니다.
- 이제 Chat Completions 통합에서 `should_replay_reasoning_content`을 통해 기존 추론 콘텐츠를 다시 전송하도록 선택할 수 있어 LiteLLM/DeepSeek 같은 어댑터의 제공업체별 추론 및 도구 호출 연속성이 개선됩니다.
- `SQLAlchemySession`의 동시 최초 쓰기, 추론 제거 후 연결 대상이 없는 어시스턴트 메시지 ID가 포함된 압축 요청, MCP/추론 항목을 남기는 `remove_all_tools()`, `FunctionTool` 인스턴스용 배치 실행기의 경합 상태를 포함한 여러 런타임 및 세션 경계 사례가 수정되었습니다.

### 0.12.0

이 마이너 릴리스에는 호환성을 깨는 변경 사항이 **없습니다**. 주요 기능 추가 사항은 [릴리스 노트](https://github.com/openai/openai-agents-python/releases/tag/v0.12.0)를 확인하세요.

### 0.11.0

이 마이너 릴리스에는 호환성을 깨는 변경 사항이 **없습니다**. 주요 기능 추가 사항은 [릴리스 노트](https://github.com/openai/openai-agents-python/releases/tag/v0.11.0)를 확인하세요.

### 0.10.0

이 마이너 릴리스에는 호환성을 깨는 변경 사항이 **없지만**, OpenAI Responses 사용자를 위한 중요한 새 기능 영역인 Responses API의 WebSocket 전송 지원이 포함되었습니다.

주요 내용:

- OpenAI Responses 모델을 위한 WebSocket 전송 지원이 추가되었습니다. 명시적으로 활성화해야 하며 HTTP는 계속 기본 전송 방식입니다.
- 여러 턴에 걸친 실행에서 공유 WebSocket 지원 제공업체와 `RunConfig`을 재사용하기 위한 `responses_websocket_session()` 헬퍼/`ResponsesWebSocketSession`이 추가되었습니다.
- 스트리밍, 도구, 승인 및 후속 턴을 다루는 새로운 WebSocket 스트리밍 코드 예제(`examples/basic/stream_ws.py`)가 추가되었습니다.

### 0.9.0

이 버전에서는 해당 메이저 버전이 3개월 전에 지원 종료(EOL)에 도달했으므로 Python 3.9를 더 이상 지원하지 않습니다. 더 최신 런타임 버전으로 업그레이드하세요.

또한 `Agent#as_tool()` 메서드에서 반환되는 값의 타입 힌트가 `Tool`에서 `FunctionTool`로 좁혀졌습니다. 일반적으로 이 변경으로 호환성 문제가 발생하지는 않지만, 코드가 더 넓은 유니온 타입에 의존한다면 일부 조정이 필요할 수 있습니다.

### 0.8.0

이 버전에서는 두 가지 런타임 동작 변경으로 인해 마이그레이션 작업이 필요할 수 있습니다.

- **동기식** Python 호출 가능 객체를 래핑하는 `FunctionTool` 인스턴스는 이제 이벤트 루프 스레드에서 실행되는 대신 `asyncio.to_thread(...)`을 통해 워커 스레드에서 실행됩니다. 도구 로직이 스레드 로컬 상태나 특정 스레드에 종속된 리소스에 의존한다면 비동기 도구 구현으로 마이그레이션하거나 도구 코드에서 스레드 종속성을 명시적으로 지정하세요.
- 이제 로컬 MCP 도구 실패 처리를 구성할 수 있으며, 기본 동작은 전체 실행을 실패시키는 대신 모델에 표시되는 오류 출력을 반환할 수 있습니다. 즉시 실패 동작에 의존하는 경우 `mcp_config={"failure_error_function": None}`을 설정하세요. 서버 수준의 `failure_error_function` 값은 에이전트 수준 설정을 재정의하므로 명시적인 핸들러가 있는 각 로컬 MCP 서버에서 `failure_error_function=None`을 설정하세요.

### 0.7.0

이 버전에는 기존 애플리케이션에 영향을 줄 수 있는 몇 가지 동작 변경 사항이 있습니다.

- 이제 중첩된 핸드오프 기록은 **명시적으로 활성화**해야 하며 기본적으로 비활성화되어 있습니다. v0.6.x의 기본 중첩 동작에 의존했다면 `RunConfig(nest_handoff_history=True)`을 명시적으로 설정하세요.
- `gpt-5.1`/`gpt-5.2`의 기본 `reasoning.effort`이 SDK 기본값으로 구성되던 이전 기본값 `"low"`에서 `"none"`로 변경되었습니다. 프롬프트나 품질/비용 프로필이 `"low"`에 의존했다면 `model_settings`에서 이를 명시적으로 설정하세요.

### 0.6.0

이 버전에서는 사용자와 어시스턴트 턴을 별도의 메시지로 전달하는 대신 기본 핸드오프 기록을 단일 어시스턴트 메시지로 패키징하여 이후 에이전트에 간결하고 예측 가능한 요약을 제공합니다
- 이제 기존의 단일 메시지 핸드오프 트랜스크립트는 기본적으로 `<CONVERSATION HISTORY>` 블록 앞에 정확한 리터럴 텍스트 `For context, here is the conversation so far between the user and the previous agent:`로 시작하므로 이후 에이전트에 명확히 표시된 요약이 제공됩니다

### 0.5.0

이 버전은 눈에 띄는 호환성 변경 사항을 도입하지 않지만, 내부적으로 새로운 기능과 몇 가지 중요한 업데이트가 포함되었습니다.

- [SIP 프로토콜 연결](https://platform.openai.com/docs/guides/realtime-sip)을 처리하기 위한 지원이 `RealtimeRunner`에 추가되었습니다.
- Python 3.14 호환성을 위해 `Runner#run_sync`의 내부 로직이 크게 개정되었습니다

### 0.4.0

이 버전에서는 [openai](https://pypi.org/project/openai/) 패키지 v1.x 버전을 더 이상 지원하지 않습니다. 이 SDK와 함께 openai v2.x를 사용하세요.

### 0.3.0

이 버전에서는 Realtime API 지원이 gpt-realtime 모델 및 해당 API 인터페이스(GA 버전)로 마이그레이션됩니다.

### 0.2.0

이 버전에서는 이전에 `Agent`을 인수로 받던 일부 위치에서 이제 `AgentBase`을 인수로 받습니다. 예를 들어 MCP 서버의 `list_tools()` 메서드 시그니처에 적용됩니다. 이는 타입만 변경된 것이며, 계속 `Agent` 객체를 받게 됩니다. 업데이트하려면 `Agent`을 `AgentBase`로 교체하여 타입 오류를 수정하면 됩니다.

### 0.1.0

이 버전에서 [`MCPServer.list_tools()`][agents.mcp.server.MCPServer]에는 `run_context` 및 `agent`이라는 두 개의 새 매개변수가 있습니다. `MCPServer`의 하위 클래스에서 재정의된 모든 `MCPServer.list_tools()` 메서드에 이 매개변수를 추가해야 합니다.