---
search:
  exclude: true
---
# リアルタイムエージェントガイド

このガイドでは、OpenAI Agents SDK のリアルタイムレイヤーが OpenAI Realtime API にどのように対応しているか、また Python SDK がその上にどのような追加動作を提供するかを説明します。

!!! note "まずはこちら"

    デフォルトの Python の利用方法を確認する場合は、まず [クイックスタート](quickstart.md)をお読みください。アプリでサーバー側の WebSocket と SIP のどちらを使用すべきか検討している場合は、[リアルタイムトランスポート](transport.md)をお読みください。ブラウザーの WebRTC トランスポートは Python SDK には含まれていません。

## 概要

リアルタイムエージェントは Realtime API への長時間接続を維持します。これにより、モデルはテキストと音声を逐次処理し、音声出力をストリーミングし、ツールを呼び出し、ターンごとに新しいリクエストを開始し直すことなく中断を処理できます。

主な SDK コンポーネントは次のとおりです。

-   **RealtimeAgent**: 1 つのリアルタイム専門エージェントに対する指示、ツール、出力ガードレール、ハンドオフ
-   **RealtimeRunner**: 開始エージェントをリアルタイムトランスポートに接続するセッションファクトリー
-   **RealtimeSession**: 入力の送信、イベントの受信、履歴の追跡、ツールの実行を行うライブセッション
-   **RealtimeModel**: トランスポートの抽象化。デフォルトは OpenAI のサーバー側 WebSocket 実装です。

## セッションのライフサイクル

一般的なリアルタイムセッションは次のようになります。

1. 1 つ以上の `RealtimeAgent` を作成します。
2. 開始エージェントを指定して `RealtimeRunner` を作成します。
3. `await runner.run()` を呼び出して `RealtimeSession` を取得します。
4. `async with session:` または `await session.enter()` を使用してセッションに入ります。
5. `send_message()` または `send_audio()` を使用してユーザー入力を送信します。
6. 会話が終了するまでセッションイベントを反復処理します。

テキストのみの実行とは異なり、`runner.run()` は最終実行結果をすぐには生成しません。代わりに、ローカル履歴、バックグラウンドでのツール実行、ガードレールの状態、アクティブなエージェント設定をトランスポートレイヤーと同期し続けるライブセッションオブジェクトを返します。

デフォルトでは、`RealtimeRunner` は `OpenAIRealtimeWebSocketModel` を使用するため、デフォルトの Python の利用方法では Realtime API へのサーバー側 WebSocket 接続が使用されます。別の `RealtimeModel` を渡した場合でも、接続の仕組みを変更しつつ、同じセッションライフサイクルとエージェント機能を利用できます。

## エージェントとセッションの設定

`RealtimeAgent` は通常の `Agent` 型よりも意図的に機能範囲が限定されています。

-   モデルの選択はエージェント単位ではなく、セッションレベルで設定します。
-   Structured outputs はサポートされていません。
-   音声は設定できますが、セッションが音声を生成した後は変更できません。
-   指示、関数ツール、ハンドオフ、フック、出力ガードレールはすべて引き続き利用できます。

`RealtimeSessionModelSettings` は、新しいネスト形式の `audio` 設定と従来のフラットなエイリアスの両方をサポートします。新しいコードではネスト形式を使用し、新しいリアルタイムエージェントには `gpt-realtime-2.1` を使用することを推奨します。

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

便利なセッションレベルの設定には、次のものがあります。

-   `audio.input.format`, `audio.output.format`
-   `audio.input.transcription`
-   `audio.input.noise_reduction`
-   `audio.input.turn_detection`
-   `audio.output.voice`, `audio.output.speed`
-   `output_modalities`
-   `tool_choice`
-   `prompt`
-   `tracing`

`RealtimeRunner(config=...)` で使用できる便利な実行レベルの設定には、次のものがあります。

-   `async_tool_calls`
-   `output_guardrails`
-   `guardrails_settings.debounce_text_length`
-   `tool_error_formatter`
-   `tracing_disabled`

型付けされた設定項目の全体については、[`RealtimeRunConfig`][agents.realtime.config.RealtimeRunConfig] および [`RealtimeSessionModelSettings`][agents.realtime.config.RealtimeSessionModelSettings] を参照してください。

## 入出力

### テキストと構造化ユーザーメッセージ

プレーンテキストまたは構造化されたリアルタイムメッセージには、[`session.send_message()`][agents.realtime.session.RealtimeSession.send_message] を使用します。

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

構造化メッセージは、リアルタイム会話に画像入力を含めるための主な方法です。[`examples/realtime/app/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/app/server.py) の Web デモ例では、この方法で `input_image` メッセージを転送します。

### 音声入力

生の音声バイトをストリーミングするには、[`session.send_audio()`][agents.realtime.session.RealtimeSession.send_audio] を使用します。

```python
await session.send_audio(audio_bytes)
```

サーバー側のターン検出を無効にしている場合は、ターンの境界を自身で指定する必要があります。高レベルの便利な方法は次のとおりです。

```python
await session.send_audio(audio_bytes, commit=True)
```

より低レベルの制御が必要な場合は、基盤となるモデルトランスポートを介して `input_audio_buffer.commit` などの生のクライアントイベントを送信することもできます。

### 手動レスポンス制御

`session.send_message()` は、高レベルの経路を使用してユーザー入力を送信し、レスポンスを開始します。生の音声バッファリングでは、すべての設定で同じ処理が **自動的に行われるわけではありません**。

Realtime API レベルでターンを手動制御するには、生の `session.update` で `turn_detection` をクリアし、その後に `input_audio_buffer.commit` と `response.create` を自身で送信します。

ターンを手動で管理する場合は、モデルトランスポートを介して生のクライアントイベントを送信できます。

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

-   `turn_detection` が無効で、モデルが応答するタイミングを自身で決定したい場合
-   レスポンスを開始する前にユーザー入力を検査または制限したい場合
-   会話外のレスポンスにカスタムプロンプトが必要な場合

[`examples/realtime/twilio_sip/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio_sip/server.py) の SIP の例では、生の `response.create` を使用して最初の挨拶を強制的に生成します。

## イベント、履歴、中断

`RealtimeSession` は、必要に応じて生のモデルイベントも転送しながら、より高レベルな SDK イベントを発行します。

特に重要なセッションイベントには、次のものがあります。

-   `audio`, `audio_end`, `audio_interrupted`
-   `agent_start`, `agent_end`
-   `tool_start`, `tool_end`, `tool_approval_required`
-   `handoff`
-   `history_added`, `history_updated`
-   `guardrail_tripped`
-   `input_audio_timeout_triggered`
-   `error`
-   `raw_model_event`

UI の状態に最も有用なイベントは、通常 `history_added` と `history_updated` です。これらは、ユーザーメッセージ、アシスタントメッセージ、ツール呼び出しを含むセッションのローカル履歴を `RealtimeItem` オブジェクトとして公開します。

### 使用量の集計

完了したモデルレスポンスに使用量が含まれている場合、OpenAI のリアルタイムモデルは `raw_model_event` 内で [`RealtimeModelUsageEvent`][agents.realtime.model_events.RealtimeModelUsageEvent] を発行します。その `usage` フィールドには当該レスポンスのトークン数が含まれ、`input_tokens_details` と `output_tokens_details` ではモダリティ別の内訳がオプションで提供されます。

また、セッションは各レスポンスの使用量を共有の [`RunContextWrapper.usage`][agents.run_context.RunContextWrapper.usage] に加算します。ライブセッションの累積使用量を確認するには、`agent_end` など、後続の高レベルイベントにある `event.info.context.usage` から読み取ります。

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

使用量は、モデルプロバイダーが完了したレスポンスに使用量を含めている場合にのみ報告されます。累積値の対象は、その `RealtimeSession` が受信したレスポンスです。複数のセッションをまたぐ合計値ではありません。

### 中断と再生追跡

ユーザーがアシスタントを中断すると、セッションは `audio_interrupted` を発行し、サーバー側の会話がユーザーに実際に聞こえた内容と一致するように履歴を更新します。

低遅延のローカル再生では、通常、デフォルトの再生トラッカーで十分です。リモート再生や遅延再生、特にテレフォニーのシナリオでは、生成されたすべての音声がすでに再生されたと仮定するのではなく、実際の再生進捗に基づいて中断時の切り詰めを行うために、[`RealtimePlaybackTracker`][agents.realtime.model.RealtimePlaybackTracker] を使用します。

[`examples/realtime/twilio/twilio_handler.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio/twilio_handler.py) の Twilio の例で、このパターンを確認できます。

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

関数ツールでは、実行前に人間の承認を必須にできます。その場合、セッションは `tool_approval_required` を発行し、`approve_tool_call()` または `reject_tool_call()` が呼び出されるまでツール実行を一時停止します。

ツールに入力ガードレールも設定されている場合、それらのガードレールは承認後、実行直前に実行されます。承認イベントが発行される前に実行するには、`RealtimeRunner(..., config={"tool_execution": {"pre_approval_tool_input_guardrails": True}})` を使用してランナーを作成します。この承認前チェックに合格した呼び出しは、承認後、実行前に再度チェックされます。

```python
async for event in session:
    if event.type == "tool_approval_required":
        await session.approve_tool_call(event.call_id)
```

具体的なサーバー側の承認ループについては、[`examples/realtime/app/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/app/server.py) を参照してください。ヒューマンインザループのドキュメントにある[ヒューマンインザループ](../human_in_the_loop.md)でも、このフローを参照しています。

### ハンドオフ

リアルタイムハンドオフを使用すると、あるエージェントから別の専門エージェントへライブ会話を転送できます。

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

単体の `RealtimeAgent` を指定したハンドオフは自動的にラップされます。また、`realtime_handoff(...)` を使用すると、名前、説明、検証、コールバック、利用可否をカスタマイズできます。リアルタイムハンドオフは、通常のハンドオフの `input_filter` を **サポートしていません**。

### ガードレール

リアルタイムエージェントは、エージェントのレスポンスに対する出力ガードレールと、関数ツール呼び出しに対する入力ガードレールをサポートします。出力ガードレールは、部分的な差分ごとではなく、出力テキストと音声文字起こしの差分をデバウンスして蓄積した単位で実行され、例外を発生させる代わりに `guardrail_tripped` を発行します。

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

音声文字起こしに対してリアルタイム出力ガードレールが作動すると、セッションはアクティブなレスポンスを中断し、`response.cancel` を強制実行して `guardrail_tripped` を発行します。さらに、作動したガードレールの名前を含む後続のユーザーメッセージを送信し、モデルが代替レスポンスを生成できるようにします。トリップワイヤーが作動した時点で一部の音声がすでにバッファリングされている可能性があるため、音声プレイヤーでは引き続き `audio_interrupted` を監視し、ローカル再生を直ちに停止する必要があります。組み込みの OpenAI Realtime トランスポートでは、ガードレールの処理が元のレスポンスの終了後に完了した場合、そのレスポンスのバッファリング済み再生のみを中断し、それより新しいレスポンスはキャンセルしません。テキストのみの出力では、セッションは代わりにレスポンス単位の `response.cancel` を送信します。停止すべき音声再生がないため、`audio_interrupted` は発行しません。組み込みの OpenAI Realtime モデルを使用している場合、テキストのみの経路でも同じ `guardrail_tripped` イベントと後続のユーザーメッセージが発行されます。

カスタム `RealtimeModel` トランスポートは、同じように元のレスポンス単位で音声を中断できるよう、`RealtimeModelSendInterrupt.response_id` と `playback_only` の指定に従う必要があります。また、テキストのみの復旧メッセージをサポートするには、`RealtimeModel.send_event_if()` もオーバーライドする必要があります。実装では、トランスポートが実際にイベントをコミットする境界で、指定された条件を再確認するか、その条件の処理を直列化する必要があります。デフォルト実装は復旧メッセージを安全にスキップします。これは、`send_event()` を待機する前に条件を確認すると、メッセージがコミットされる前に新しいレスポンスが開始される可能性があるためです。レスポンスのキャンセルと `guardrail_tripped` イベントは引き続き発生します。

## SIP とテレフォニー

Python SDK には、[`OpenAIRealtimeSIPModel`][agents.realtime.openai_realtime.OpenAIRealtimeSIPModel] を介したファーストクラスの SIP アタッチフローが含まれています。

Realtime Calls API を介して通話を受信し、生成された `call_id` にエージェントセッションをアタッチする場合に使用します。

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

最初に通話を受け入れる必要があり、受け入れ時のペイロードをエージェントから派生したセッション設定と一致させたい場合は、`OpenAIRealtimeSIPModel.build_initial_session_payload(...)` を使用します。完全なフローは [`examples/realtime/twilio_sip/server.py`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime/twilio_sip/server.py) に示されています。

## 低レベルアクセスとカスタムエンドポイント

基盤となるトランスポートオブジェクトには、`session.model` を介してアクセスできます。

次の処理が必要な場合に使用します。

-   `session.model.add_listener(...)` を介したカスタムリスナー
-   `response.create` や `session.update` などの生のクライアントイベント
-   `model_config` を介したカスタムの `url`、`headers`、`api_key` の処理
-   既存のリアルタイム通話への `call_id` のアタッチ

`RealtimeModelConfig` は次の項目をサポートします。

-   `api_key`
-   `url`
-   `headers`
-   `initial_model_settings`
-   `playback_tracker`
-   `call_id`

このリポジトリに含まれる `call_id` の例は SIP です。より広範な Realtime API でも、一部のサーバー側制御フローで `call_id` が使用されますが、ここでは Python の例として提供されていません。

Azure OpenAI に接続する場合は、GA 版の Realtime エンドポイント URL と明示的なヘッダーを渡します。例：

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

`headers` を渡した場合、SDK は `Authorization` を自動的に追加しません。リアルタイムエージェントでは、従来のベータ版パス（`/openai/realtime?api-version=...`）を使用しないでください。

## 関連情報

-   [リアルタイムトランスポート](transport.md)
-   [クイックスタート](quickstart.md)
-   [OpenAI Realtime の会話](https://developers.openai.com/api/docs/guides/realtime-conversations/)
-   [OpenAI Realtime のサーバー側制御](https://developers.openai.com/api/docs/guides/realtime-server-controls/)
-   [`examples/realtime`](https://github.com/openai/openai-agents-python/tree/main/examples/realtime)