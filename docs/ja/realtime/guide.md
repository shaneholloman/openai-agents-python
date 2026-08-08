---
search:
  exclude: true
---
# リアルタイムエージェントガイド

このガイドでは、OpenAI Agents SDK のリアルタイムレイヤーと OpenAI Realtime API の対応関係、および Python SDK が追加する動作について説明します。

!!! note "最初にお読みください"

    デフォルトの Python の手順を使用する場合は、最初に[クイックスタート](quickstart.md)をお読みください。アプリでサーバー側 WebSocket と SIP のどちらを使用するか検討している場合は、[リアルタイムトランスポート](transport.md)をお読みください。ブラウザーの WebRTC トランスポートは Python SDK に含まれません。

## 概要

リアルタイムエージェントは Realtime API への長時間接続を維持するため、モデルはターンごとに新しいリクエストを開始し直すことなく、テキストと音声を段階的に処理し、音声出力をストリーミングし、ツールを呼び出し、中断を処理できます。

SDK の主なコンポーネントは次のとおりです。

-   **RealtimeAgent**: 1 つのリアルタイムスペシャリストに対する指示、ツール、出力ガードレール、ハンドオフ
-   **RealtimeRunner**: 開始エージェントをリアルタイムトランスポートに接続するセッションファクトリー
-   **RealtimeSession**: 入力の送信、イベントの受信、履歴の追跡、ツールの実行を行うライブセッション
-   **RealtimeModel**: トランスポートの抽象化。デフォルトは OpenAI のサーバー側 WebSocket 実装です。

## セッションのライフサイクル

一般的なリアルタイムセッションの流れは次のとおりです。

1. 1 つ以上の `RealtimeAgent` を作成します。
2. 開始エージェントを指定して `RealtimeRunner` を作成します。
3. `RealtimeSession` を取得するために `await runner.run()` を呼び出します。
4. `async with session:` または `await session.enter()` を使用してセッションに入ります。
5. `send_message()` または `send_audio()` を使用してユーザー入力を送信します。
6. 会話が終了するまでセッションイベントを反復処理します。

テキストのみの実行とは異なり、`runner.run()` は最終的な実行結果をすぐには生成しません。代わりに、ローカル履歴、バックグラウンドでのツール実行、ガードレールの状態、アクティブなエージェント設定をトランスポートレイヤーと同期し続けるライブセッションオブジェクトを返します。

デフォルトでは、`RealtimeRunner` は `OpenAIRealtimeWebSocketModel` を使用するため、デフォルトの Python の手順では Realtime API へのサーバー側 WebSocket 接続が使用されます。別の `RealtimeModel` を渡した場合も、接続方法は変更できますが、同じセッションライフサイクルとエージェント機能が適用されます。

## エージェントとセッションの設定

`RealtimeAgent` は、通常の `Agent` 型よりも意図的に対象範囲が限定されています。

-   モデルの選択は、エージェント単位ではなくセッションレベルで設定します。
-   structured outputs はサポートされていません。
-   音声は設定できますが、セッションが音声を一度生成した後は変更できません。
-   指示、関数ツール、ハンドオフ、フック、出力ガードレールは引き続きすべて機能します。

`RealtimeSessionModelSettings` は、新しいネスト形式の `audio` 設定と、従来のフラットなエイリアスの両方をサポートします。新しいコードではネスト形式を推奨します。また、新しいリアルタイムエージェントでは `gpt-realtime-2.1` から始めてください。

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

便利なセッションレベルの設定には次のものがあります。

-   `audio.input.format`、`audio.output.format`
-   `audio.input.transcription`
-   `audio.input.noise_reduction`
-   `audio.input.turn_detection`
-   `audio.output.voice`、`audio.output.speed`
-   `output_modalities`
-   `tool_choice`
-   `prompt`
-   `tracing`

`RealtimeRunner(config=...)` で利用できる便利な実行レベルの設定には次のものがあります。

-   `async_tool_calls`
-   `output_guardrails`
-   `guardrails_settings.debounce_text_length`
-   `tool_error_formatter`
-   `tracing_disabled`

型付けされた API 全体については、[`RealtimeRunConfig`][agents.realtime.config.RealtimeRunConfig]および [`RealtimeSessionModelSettings`][agents.realtime.config.RealtimeSessionModelSettings]を参照してください。

## 入出力

### テキストと構造化ユーザーメッセージ

プレーンテキストまたは構造化されたリアルタイムメッセージには、[`session.send_message()`][agents.realtime.session.RealtimeSession.send_message]を使用します。

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

構造化メッセージは、リアルタイムの会話に画像入力を含めるための主な方法です。[`examples/realtime/app/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/app/server.py)の Web デモのコード例では、この方法で `input_image` メッセージを転送します。

### 音声入力

raw 音声バイトをストリーミングするには、[`session.send_audio()`][agents.realtime.session.RealtimeSession.send_audio]を使用します。

```python
await session.send_audio(audio_bytes)
```

サーバー側のターン検出が無効な場合は、ターンの境界を自分で指定する必要があります。高レベルの便利な方法は次のとおりです。

```python
await session.send_audio(audio_bytes, commit=True)
```

より低レベルの制御が必要な場合は、基盤となるモデルトランスポートを通じて、`input_audio_buffer.commit` などの Realtime API クライアントイベントを直接送信することもできます。

### 手動レスポンス制御

`session.send_message()` は、高レベルの経路を使用してユーザー入力を送信し、レスポンスを自動的に開始します。一部の設定では、raw 音声のバッファリングだけでは同じ動作が**自動的には**行われません。

Realtime API レベルでターンを手動制御するには、`turn_detection` を `null` に設定する `session.update` イベントを送信し、その後に `input_audio_buffer.commit` と `response.create` を自分で送信します。

ターンを手動で管理する場合は、モデルトランスポートを通じて raw クライアントイベントを送信できます。

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

このパターンは、次の場合に役立ちます。

-   `turn_detection` が無効で、モデルが応答するタイミングを決定したい場合
-   レスポンスをトリガーする前に、ユーザー入力を検査または制御したい場合
-   帯域外レスポンスにカスタムプロンプトが必要な場合

[`examples/realtime/twilio_sip/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio_sip/server.py)の SIP のコード例では、最初の挨拶を強制するために raw `response.create` を使用しています。

## イベント、履歴、中断

`RealtimeSession` は高レベルの SDK イベントを生成しつつ、必要に応じて raw モデルイベントも転送します。

特に重要なセッションイベントには次のものがあります。

-   `audio`、`audio_end`、`audio_interrupted`
-   `agent_start`、`agent_end`
-   `tool_start`、`tool_end`、`tool_approval_required`
-   `handoff`
-   `history_added`、`history_updated`
-   `guardrail_tripped`
-   `input_audio_timeout_triggered`
-   `error`
-   `raw_model_event`

UI の状態管理に最も役立つイベントは、通常 `history_added` と `history_updated` です。これらは、ユーザーメッセージ、アシスタントメッセージ、ツール呼び出しを含むセッションのローカル履歴を、`RealtimeItem` オブジェクトとして公開します。

### 使用量の集計

完了したモデルレスポンスに使用量が含まれている場合、SDK の OpenAI `RealtimeModel` トランスポートは、`raw_model_event` 内で [`RealtimeModelUsageEvent`][agents.realtime.model_events.RealtimeModelUsageEvent]を生成します。その `usage` フィールドには、そのレスポンスのトークン数が含まれ、`input_tokens_details` と `output_tokens_details` には任意のモダリティ別内訳が含まれます。

また、セッションは各レスポンスの使用量を共有の [`RunContextWrapper.usage`][agents.run_context.RunContextWrapper.usage]に追加します。ライブセッションの累積使用量を確認するには、`agent_end` など、その後に発生する高レベルイベントの `event.info.context.usage` から読み取ります。

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

使用量は、モデルプロバイダーが完了したレスポンスに使用量を含めた場合にのみ報告されます。累積値は、その `RealtimeSession` が受信したレスポンスを対象とし、複数のセッションをまたぐ合計ではありません。

### 中断と再生位置の追跡

ユーザーがアシスタントを中断すると、セッションは `audio_interrupted` を生成し、ユーザーが実際に聞いた内容とサーバー側の会話が一致するように履歴を更新します。

低遅延のローカル再生では、多くの場合、デフォルトの再生トラッカーで十分です。リモート再生や遅延再生のシナリオ、特に電話通信では、生成済みの音声がすべて再生されたと仮定するのではなく、実際の再生位置で中断されたレスポンスを切り詰めるために、[`RealtimePlaybackTracker`][agents.realtime.model.RealtimePlaybackTracker]を使用します。

[`examples/realtime/twilio/twilio_handler.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio/twilio_handler.py)の Twilio のコード例で、このパターンを確認できます。

## ツール、承認、ハンドオフ、ガードレール

### 関数ツール

リアルタイムエージェントは、ライブ会話中の関数ツールをサポートします。

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

### ツールの承認

関数ツールでは、実行前に人間による承認を必須にできます。その場合、セッションは `tool_approval_required` を生成し、`approve_tool_call()` または `reject_tool_call()` を呼び出すまでツールの実行を一時停止します。

ツールに入力ガードレールも設定されている場合、それらのガードレールは承認後、実行直前に動作します。承認イベントが生成される前に実行するには、`RealtimeRunner(..., config={"tool_execution": {"pre_approval_tool_input_guardrails": True}})` を指定してランナーを作成します。この承認前チェックに合格した呼び出しも、承認後かつ実行前に再度チェックされます。

```python
async for event in session:
    if event.type == "tool_approval_required":
        await session.approve_tool_call(event.call_id)
```

具体的なサーバー側の承認ループについては、[`examples/realtime/app/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/app/server.py)を参照してください。Human-in-the-loop のドキュメントでも、[Human-in-the-loop](../human_in_the_loop.md)でこのフローを参照しています。

### ハンドオフ

リアルタイムハンドオフでは、あるエージェントから別のスペシャリストへライブ会話を引き継ぐことができます。

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

ハンドオフとして直接使用される `RealtimeAgent` オブジェクトは自動的にラップされます。また、`realtime_handoff(...)` を使用すると、名前、説明、検証、コールバック、利用可否をカスタマイズできます。リアルタイムハンドオフは、通常のハンドオフの `input_filter` をサポートして**いません**。

### ガードレール

リアルタイムエージェントは、エージェントのレスポンスに対する出力ガードレールと、関数ツール呼び出しに対する入力ガードレールをサポートします。出力ガードレールのチェックにはデバウンスが適用されます。各チェックは部分的な差分ごとではなく、蓄積された出力テキストと音声文字起こしの差分に対して実行され、例外を送出する代わりに `guardrail_tripped` を生成します。

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

リアルタイム出力ガードレールが音声文字起こしに対して作動すると、セッションはアクティブなレスポンスを中断し、`response.cancel` を強制し、`guardrail_tripped` を生成して、作動したガードレールの名前を含むフォローアップのユーザーメッセージを送信します。これにより、モデルは代替レスポンスを生成できます。トリップワイヤーが作動した時点ですでに一部の音声がバッファリングされている可能性があるため、音声プレーヤーは引き続き `audio_interrupted` を監視し、ローカル再生を直ちに停止する必要があります。組み込みの OpenAI Realtime トランスポートでは、チェック対象のレスポンスが終了した後にガードレールチェックが完了した場合、セッションはそのレスポンスのバッファリング済み再生のみを中断し、後から開始されたレスポンスはキャンセルしません。テキストのみの出力では、代わりにレスポンス単位の `response.cancel` が送信されます。停止すべき音声再生がないため、`audio_interrupted` は生成されません。組み込みの OpenAI Realtime モデルを使用している場合、テキストのみの経路でも、同じ `guardrail_tripped` イベントとフォローアップのユーザーメッセージが生成されます。

カスタム `RealtimeModel` トランスポートで同じ発生元レスポンス単位の音声中断動作を実現するには、`RealtimeModelSendInterrupt.response_id` と `playback_only` に従う必要があります。また、テキストのみの出力経路で復旧メッセージをサポートするには、`RealtimeModel.send_event_if()` をオーバーライドする必要があります。実装では、トランスポートが実際にイベントをコミットする境界で指定された条件を再確認するか、条件チェックとイベントのコミットをまとめて直列化する必要があります。デフォルト実装は、復旧メッセージを安全にスキップします。条件を一度チェックしてからイベントを別途送信すると、条件チェックとイベントのコミットの間に別のレスポンスが開始される可能性があるためです。ただし、レスポンスのキャンセルと `guardrail_tripped` イベントは引き続き発生します。

## SIP と電話通信

Python SDK には、[`OpenAIRealtimeSIPModel`][agents.realtime.openai_realtime.OpenAIRealtimeSIPModel]を通じた正式サポートの SIP アタッチフローが含まれています。

Realtime Calls API を通じて着信があり、その結果生成された `call_id` にエージェントセッションをアタッチする場合に使用します。

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

先に通話を受け付け、受付時のペイロードをエージェントから導出されたセッション設定と一致させる必要がある場合は、`OpenAIRealtimeSIPModel.build_initial_session_payload(...)` を使用します。完全なフローは [`examples/realtime/twilio_sip/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio_sip/server.py)で確認できます。

## 低レベルアクセスとカスタムエンドポイント

`session.model` を通じて、基盤となるトランスポートオブジェクトにアクセスできます。

これは、次のものが必要な場合に使用します。

-   `session.model.add_listener(...)` を使用したカスタムリスナー
-   `response.create` や `session.update` などの raw クライアントイベント
-   `model_config` を通じたカスタムの `url`、`headers`、`api_key` の処理
-   既存のリアルタイム通話への `call_id` のアタッチ

`RealtimeModelConfig` は次のものをサポートします。

-   `api_key`
-   `url`
-   `headers`
-   `initial_model_settings`
-   `playback_tracker`
-   `call_id`

このリポジトリに同梱されている `call_id` のコード例は SIP です。より広範な Realtime API でも一部のサーバー側制御フローに `call_id` が使用されますが、ここでは Python のコード例としてパッケージ化されていません。

Azure OpenAI に接続する場合は、GA の Realtime エンドポイント URL と明示的なヘッダーを渡します。次に例を示します。

```python
session = await runner.run(
    model_config={
        "url": "wss://<your-resource>.openai.azure.com/openai/v1/realtime?model=<deployment-name>",
        "headers": {"api-key": "<your-azure-api-key>"},
    }
)
```

トークンベースの認証では、`headers` に Bearer トークンを指定します。

```python
session = await runner.run(
    model_config={
        "url": "wss://<your-resource>.openai.azure.com/openai/v1/realtime?model=<deployment-name>",
        "headers": {"authorization": f"Bearer {token}"},
    }
)
```

`headers` を渡した場合、SDK は `Authorization` を自動的には追加しません。リアルタイムエージェントでは、従来のベータ版のパス（`/openai/realtime?api-version=...`）を使用しないでください。

## 関連資料

-   [リアルタイムトランスポート](transport.md)
-   [クイックスタート](quickstart.md)
-   [OpenAI Realtime の会話](https://developers.openai.com/api/docs/guides/realtime-conversations/)
-   [OpenAI Realtime のサーバー側制御](https://developers.openai.com/api/docs/guides/realtime-server-controls/)
-   [`examples/realtime`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime)