---
search:
  exclude: true
---
# エージェントの実行

[`Runner`][agents.run.Runner] クラスを介してエージェントを実行できます。次の 3 つの方法があります。

1. [`Runner.run()`][agents.run.Runner.run]：非同期で実行し、[`RunResult`][agents.result.RunResult] を返します。
2. [`Runner.run_sync()`][agents.run.Runner.run_sync]：同期メソッドであり、内部では `.run()` を実行します。
3. [`Runner.run_streamed()`][agents.run.Runner.run_streamed]：非同期で実行し、[`RunResultStreaming`][agents.result.RunResultStreaming] を返します。LLM をストリーミングモードで呼び出し、イベントを受信すると順次ストリーミングします。

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

詳しくは、[実行結果ガイド](results.md)をご覧ください。

## Runner のライフサイクルと設定

### エージェントループ

`Runner` の run メソッドを使用する際は、開始エージェントと入力を渡します。入力には次のものを指定できます。

-   文字列（ユーザーメッセージとして扱われます）
-   OpenAI Responses API 形式の入力項目のリスト
-   中断された実行を再開する場合は [`RunState`][agents.run_state.RunState]

その後、Runner は次のループを実行します。

1. 現在のエージェントについて、現在の入力で LLM を呼び出します。
2. LLM が出力を生成します。
    1. LLM が `final_output` を返した場合、ループを終了して実行結果を返します。
    2. LLM がハンドオフを行った場合、現在のエージェントと入力を更新し、ループを再実行します。
    3. LLM がツール呼び出しを生成した場合、それらのツール呼び出しを実行して実行結果を追加し、ループを再実行します。
3. 渡された `max_turns` を超えた場合、[`MaxTurnsExceeded`][agents.exceptions.MaxTurnsExceeded] 例外を発生させます。このターン制限を無効にするには、`max_turns=None` を渡します。

!!! note

    LLM の出力が「最終出力」と見なされる条件は、目的の型のテキスト出力が生成され、ツール呼び出しが存在しないことです。

### ストリーミング

ストリーミングを使用すると、LLM の実行中にストリーミングイベントも受信できます。ストリームが完了すると、[`RunResultStreaming`][agents.result.RunResultStreaming] には、生成されたすべての新しい出力を含む、実行に関する完全な情報が格納されます。ストリーミングイベントには `.stream_events()` を呼び出せます。詳しくは、[ストリーミングガイド](streaming.md)をご覧ください。

#### Responses WebSocket トランスポート（オプションのヘルパー）

OpenAI Responses WebSocket トランスポートを有効にしても、通常の `Runner` API を引き続き使用できます。接続を再利用するには WebSocket セッションヘルパーの使用を推奨しますが、必須ではありません。

これは WebSocket トランスポート経由の Responses API であり、[Realtime API](realtime/guide.md)ではありません。

トランスポートの選択ルール、および具象モデルオブジェクトやカスタムプロバイダーに関する注意事項については、[モデル](models/index.md#responses-websocket-transport)をご覧ください。

##### パターン 1：セッションヘルパーなし（利用可能）

WebSocket トランスポートのみが必要で、共有プロバイダーやセッションを SDK に管理させる必要がない場合に使用します。

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

このパターンは単一の実行に適しています。`Runner.run()` / `Runner.run_streamed()` を繰り返し呼び出す場合、同じ `RunConfig` / プロバイダーインスタンスを手動で再利用しない限り、実行ごとに再接続される可能性があります。

##### パターン 2：`responses_websocket_session()` の使用（複数ターンでの再利用に推奨）

複数の実行にわたって WebSocket 対応の共有プロバイダーと `RunConfig` を使用する場合は、[`responses_websocket_session()`][agents.responses_websocket_session] を使用します。同じ `run_config` を継承する、エージェントをツールとして使用するネストされた呼び出しも対象です。

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

コンテキストを終了する前に、ストリーミングされた実行結果の消費を完了してください。WebSocket リクエストの処理中にコンテキストを終了すると、共有接続が強制的に閉じられる可能性があります。

サービスは各 WebSocket 接続で一度に 1 つのレスポンスを処理し、接続時間を 60 分に制限します。ヘルパーは接続を再利用しますが、これらの制約を取り除くものではありません。再接続後、`store=False` および ZDR フローでは、キャッシュされていない `previous_response_id` を復元できません。完全な入力コンテキストで新しいチェーンを開始するか、ローカルで管理しているセッション状態から再構築してください。完全な復旧動作については、[Responses WebSocket トランスポートに関する注意事項](models/index.md#responses-websocket-transport)をご覧ください。

長時間の推論ターンで WebSocket の keepalive タイムアウトが発生する場合は、`ping_timeout` を増やすか、`ping_timeout=None` を設定してハートビートタイムアウトを無効にしてください。WebSocket のレイテンシーより信頼性を重視する実行には、HTTP/SSE トランスポートを使用してください。

### 実行設定

`run_config` パラメーターを使用すると、エージェント実行に関するいくつかのグローバル設定を構成できます。

#### 一般的な実行設定のカテゴリー

各エージェントの定義を変更せずに、単一の実行に対する動作を上書きするには、`RunConfig` を使用します。

##### モデル、プロバイダー、セッションのデフォルト

-   [`model`][agents.run.RunConfig.model]：各 Agent に設定されている `model` に関係なく、使用するグローバルな LLM モデルを設定できます。
-   [`model_provider`][agents.run.RunConfig.model_provider]：モデル名を検索するためのモデルプロバイダーです。デフォルトは OpenAI です。
-   [`model_settings`][agents.run.RunConfig.model_settings]：エージェント固有の設定を上書きします。たとえば、グローバルな `temperature` や `top_p` を設定できます。
-   [`session_settings`][agents.run.RunConfig.session_settings]：実行中に履歴を取得する際のセッションレベルのデフォルト（例：`SessionSettings(limit=...)`）を上書きします。
-   [`session_input_callback`][agents.run.RunConfig.session_input_callback]：Sessions を使用する際に、各ターンの前に新しいユーザー入力をセッション履歴へマージする方法をカスタマイズします。コールバックは同期または非同期にできます。

##### ガードレール、ハンドオフ、モデル入力の整形

-   [`input_guardrails`][agents.run.RunConfig.input_guardrails]、[`output_guardrails`][agents.run.RunConfig.output_guardrails]：すべての実行に含める入力または出力ガードレールのリストです。
-   [`handoff_input_filter`][agents.run.RunConfig.handoff_input_filter]：ハンドオフに独自のフィルターが設定されていない場合に、すべてのハンドオフへ適用するグローバル入力フィルターです。入力フィルターを使用すると、新しいエージェントへ送信される入力を編集できます。詳しくは、[`Handoff.input_filter`][agents.handoffs.Handoff.input_filter] のドキュメントをご覧ください。
-   [`nest_handoff_history`][agents.run.RunConfig.nest_handoff_history]：次のエージェントを呼び出す前に、元の位置にあるメッセージ項目を欠損なく保持しながら、要約可能な履歴を順序付きの assistant 要約セグメントへ圧縮するオプトインのベータ機能です。ネストされたハンドオフの安定化を進めているため、デフォルトでは無効です。有効にするには `True` を設定し、raw のトランスクリプトをそのまま渡すには `False` のままにします。Sessions、`RunState`、`RunResult.to_input_list()` は、SDK のデフォルトで生成されたネスト済み履歴に同一のメッセージ出現箇所がすでに含まれている場合、そのメッセージを重複して追加しません。一方、内容が同一でも別々のメッセージは保持します。[Runner のすべてのメソッド][agents.run.Runner]は、指定されていない場合に `RunConfig` を自動作成するため、クイックスタートやコード例ではデフォルトで無効のままです。また、明示的な [`Handoff.input_filter`][agents.handoffs.Handoff.input_filter] コールバックによる上書きも引き続き有効です。個々のハンドオフでは、[`Handoff.nest_handoff_history`][agents.handoffs.Handoff.nest_handoff_history] を介してこの設定を上書きできます。
-   [`handoff_history_mapper`][agents.run.RunConfig.handoff_history_mapper]：`nest_handoff_history` を有効にした際に、正規化されたトランスクリプト（履歴とハンドオフ項目）を受け取るオプションの callable です。完全なハンドオフフィルターを記述せずに、組み込みの順序付き要約セグメントを置き換え、次のエージェントへ転送する入力項目の正確なリストを返す必要があります。
-   [`call_model_input_filter`][agents.run.RunConfig.call_model_input_filter]：モデル呼び出しの直前に、完全に準備されたモデル入力（instructions と入力項目）を編集するためのフックです。たとえば、履歴の切り詰めやシステムプロンプトの注入に使用できます。
-   [`reasoning_item_id_policy`][agents.run.RunConfig.reasoning_item_id_policy]：Runner が以前の出力を次のターンのモデル入力へ変換する際に、推論項目の ID を保持するか省略するかを制御します。

##### トレーシングとオブザーバビリティ

-   [`tracing_disabled`][agents.run.RunConfig.tracing_disabled]：実行全体の[トレーシング](tracing.md)を無効にできます。
-   [`tracing`][agents.run.RunConfig.tracing]：実行ごとのトレーシング API キーなど、トレースのエクスポート設定を上書きするには、[`TracingConfig`][agents.tracing.TracingConfig] を渡します。
-   [`trace_include_sensitive_data`][agents.run.RunConfig.trace_include_sensitive_data]：LLM やツール呼び出しの入力／出力など、機密情報である可能性のあるデータをトレースに含めるかどうかを設定します。
-   [`workflow_name`][agents.run.RunConfig.workflow_name]、[`trace_id`][agents.run.RunConfig.trace_id]、[`group_id`][agents.run.RunConfig.group_id]：実行のトレーシングワークフロー名、トレース ID、トレースグループ ID を設定します。少なくとも `workflow_name` を設定することを推奨します。グループ ID は、複数の実行にまたがるトレースを関連付けるためのオプションフィールドです。
-   [`trace_metadata`][agents.run.RunConfig.trace_metadata]：すべてのトレースに含めるメタデータです。

##### ツール実行、承認、ツールエラーの動作

-   [`tool_execution`][agents.run.RunConfig.tool_execution]：同時に実行する関数ツールの数を制限するなど、ローカルツール呼び出しに対する SDK 側の実行動作を設定します。
-   [`tool_not_found_behavior`][agents.run.RunConfig.tool_not_found_behavior]：モデルが生成した未解決の関数ツール呼び出しを Runner が処理する方法を設定します。デフォルトでは `ModelBehaviorError` が発生します。代わりに、モデルから確認できるエラー出力を返すようオプトインできます。
-   [`tool_name_collision_policy`][agents.run.RunConfig.tool_name_collision_policy]：名前空間のない関数ツール名とハンドオフ名が衝突した場合に、Runner が処理する方法を設定します。デフォルトの `"warn"` は、対応方法を示す警告をログに記録し、現在ディスパッチ対象となっているものだけを公開します。`"error"` は、モデルが呼び出される前に `UserError` を発生させます。名前空間付きツールと遅延読み込みツールに対する厳密な検証は変更されません。
-   [`tool_error_formatter`][agents.run.RunConfig.tool_error_formatter]：承認の拒否や、オプトインしたツール未検出時の出力など、モデルから確認できるツールエラーメッセージをカスタマイズします。

ネストされたハンドオフは、オプトインのベータ機能として利用できます。`RunConfig(nest_handoff_history=True)` を渡すか、`handoff(..., nest_handoff_history=True)` を設定すると、特定のハンドオフについて順序付きのトランスクリプト圧縮を有効にできます。組み込みのマッパーは、トランスクリプト全体を 1 つのメッセージへ圧縮するのではなく、欠損のないメッセージ項目を囲むように、生成された assistant 要約セグメントを配置します。raw のトランスクリプトを保持する場合（デフォルト）は、フラグを設定しないか、必要な形式で会話を転送する `handoff_input_filter`（または `handoff_history_mapper`）を指定します。カスタムマッパーを記述せずに、生成された要約セグメントで使用されるラッパーテキストを変更するには、[`set_conversation_history_wrappers`][agents.handoffs.set_conversation_history_wrappers] を呼び出します。デフォルトへ戻すには [`reset_conversation_history_wrappers`][agents.handoffs.reset_conversation_history_wrappers] を呼び出します。

#### 実行設定の詳細

##### `tool_execution`

実行中のローカル関数ツールの並行処理数を制限するなど、ローカル関数ツールに対する SDK 側の動作を設定する場合は、`tool_execution` を使用します。

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

`max_function_tool_concurrency=None` はデフォルトの動作を維持します。モデルが 1 ターンで複数の関数ツール呼び出しを生成すると、SDK は生成されたすべてのローカル関数ツール呼び出しを開始します。同時に実行するローカル関数ツールの数を制限するには、整数値を設定します。

これは、プロバイダー側の [`ModelSettings.parallel_tool_calls`][agents.model_settings.ModelSettings.parallel_tool_calls] とは別の設定です。`parallel_tool_calls` は、モデルが 1 つのレスポンスで複数のツール呼び出しを生成できるかどうかを制御します。`tool_execution.max_function_tool_concurrency` は、モデルが生成した後に、SDK がローカル関数ツール呼び出しを実行する方法を制御します。

`pre_approval_tool_input_guardrails=False` は、デフォルトの承認フローを維持します。関数ツールに承認が必要な場合、まず実行が一時停止し、ツール入力ガードレールは承認後の実行直前にのみ実行されます。保留中の承認による中断が生成される前に、関数ツールの入力ガードレールを実行する場合は `True` を設定します。この承認前チェックを通過した呼び出しでも、承認後に同じ入力ガードレールが再度実行されるため、時間依存のチェックは実行前に再検証されます。

##### `tool_not_found_behavior`

デフォルトでは、モデルが生成した関数ツール呼び出しが、現在のエージェントで利用可能な関数ツールのいずれとも一致しない場合、Runner は `ModelBehaviorError` を発生させます。

実行を復旧可能な状態に保つには、`tool_not_found_behavior="return_error_to_model"` を設定します。このモードでは、SDK は未解決のツール呼び出しに対する `function_call_output` を追加し、モデルを再実行します。これにより、モデルは利用可能なツールを選択するか、そのツールを使用せずに回答できます。

```python
from agents import Agent, RunConfig, Runner

agent = Agent(name="Assistant", tools=[...])

result = await Runner.run(
    agent,
    "Handle this request with the available tools.",
    run_config=RunConfig(tool_not_found_behavior="return_error_to_model"),
)
```

現在、このオプションは未解決の関数ツール呼び出しにのみ適用されます。その他の無効なツールペイロードでは、既存のエラー動作が引き続き使用されます。

##### `tool_error_formatter`

SDK がモデルから確認できるツールエラー出力を作成する際に、モデルへ返されるメッセージをカスタマイズするには、`tool_error_formatter` を使用します。

フォーマッターは、次の情報を含む [`ToolErrorFormatterArgs`][agents.run_config.ToolErrorFormatterArgs] を受け取ります。

-   `kind`：`"approval_rejected"` や `"tool_not_found"` などのエラーカテゴリー。
-   `tool_type`：ツールランタイム（`"function"`、`"computer"`、`"shell"`、`"apply_patch"`、または `"custom"`）。
-   `tool_name`：ツール名。
-   `call_id`：ツール呼び出し ID。
-   `default_message`：モデルから確認できる SDK のデフォルトメッセージ。
-   `run_context`：アクティブな実行コンテキストラッパー。

メッセージを置き換えるには文字列を返し、SDK のデフォルトを使用するには `None` を返します。

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

`reasoning_item_id_policy` は、Runner が履歴を次へ引き継ぐ際に、推論項目を次のターンのモデル入力へ変換する方法を制御します。たとえば、`RunResult.to_input_list()` を使用する場合や、セッションを利用した実行が対象です。

-   `None` または `"preserve"`（デフォルト）：推論項目の ID を保持します。
-   `"omit"`：生成される次のターンの入力から、推論項目の ID を削除します。

`"omit"` は主に、推論項目が `id` 付きで送信されたものの、後続に必要な項目がない場合に発生する Responses API の 400 エラーへのオプトインの緩和策として使用します。たとえば、`Item 'rs_...' of type 'reasoning' was provided without its required following item.` というエラーです。

これは、SDK が以前の出力から後続の入力を構築する複数ターンのエージェント実行で発生する可能性があります。対象には、セッションの永続化、サーバー管理の会話差分、ストリーミング／非ストリーミングの後続ターン、再開パスが含まれます。推論項目の ID が保持されていても、プロバイダーがその ID と対応する後続項目との組み合わせを維持するよう要求する場合に発生します。

`reasoning_item_id_policy="omit"` を設定すると、推論内容は保持されますが、推論項目の `id` が削除されます。これにより、SDK が生成する後続入力でその API 不変条件に抵触することを回避できます。

適用範囲に関する注意事項：

-   これは、SDK が後続入力を構築する際に生成または転送する推論項目のみを変更します。
-   ユーザーが指定した初期入力項目は書き換えません。
-   このポリシーの適用後でも、`call_model_input_filter` によって意図的に推論 ID を再導入できます。

## 状態と会話の管理

### メモリー戦略の選択

状態を次のターンへ引き継ぐ一般的な方法は 4 つあります。

| 戦略 | 状態の保存場所 | 適した用途 | 次のターンで渡すもの |
| --- | --- | --- | --- |
| `result.to_input_list()` | アプリのメモリー | 小規模なチャットループ、完全な手動制御、任意のプロバイダー | `result.to_input_list()` のリストと次のユーザーメッセージ |
| `session` | ストレージと SDK | 永続的なチャット状態、再開可能な実行、カスタムストア | 同じ `session` インスタンス、または同じストアを参照する別のインスタンス |
| `conversation_id` | OpenAI Conversations API | ワーカーやサービス間で共有する、名前付きのサーバー側会話 | 同じ `conversation_id` と新しいユーザーターンのみ |
| `previous_response_id` | OpenAI Responses API | 会話リソースを作成せずに使用する、軽量なサーバー管理の継続 | `result.last_response_id` と新しいユーザーターンのみ |

`result.to_input_list()` と `session` はクライアント管理です。`conversation_id` と `previous_response_id` は OpenAI 管理であり、OpenAI Responses API を使用している場合にのみ適用されます。ほとんどのアプリケーションでは、会話ごとに 1 つの永続化戦略を選択してください。クライアント管理の履歴と OpenAI 管理の状態を混在させると、両方のレイヤーを意図的に調整している場合を除き、コンテキストが重複する可能性があります。

!!! note

    セッションの永続化は、サーバー管理の会話設定
    （`conversation_id`、`previous_response_id`、または `auto_previous_response_id`）と
    同じ実行内で併用できません。呼び出しごとに 1 つの方法を選択してください。

### 会話／チャットスレッド

いずれかの run メソッドを呼び出すと、1 つ以上のエージェントが実行される場合があり、その結果として 1 回以上の LLM 呼び出しが発生する可能性があります。ただし、チャット会話においては論理的に 1 つのターンを表します。たとえば、次のようになります。

1. ユーザーターン：ユーザーがテキストを入力します
2. Runner の実行：最初のエージェントが LLM を呼び出し、ツールを実行して 2 番目のエージェントへハンドオフします。2 番目のエージェントがさらにツールを実行し、出力を生成します。

エージェントの実行終了時に、ユーザーへ表示する内容を選択できます。たとえば、エージェントが生成した新しい項目をすべて表示することも、最終出力だけを表示することもできます。いずれの場合も、ユーザーが追加の質問をする可能性があり、その際は run メソッドを再度呼び出せます。

#### 手動による会話管理

[`RunResultBase.to_input_list()`][agents.result.RunResultBase.to_input_list] メソッドを使用して次のターンの入力を取得し、会話履歴を手動で管理できます。

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

#### Sessions による自動会話管理

より簡単な方法として、`.to_input_list()` を手動で呼び出すことなく、[Sessions](sessions/index.md) を使用して会話履歴を自動的に処理できます。

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

Sessions は、次の処理を自動的に行います。

-   各実行前に会話履歴を取得します
-   各実行後に新しいメッセージを保存します
-   セッション ID ごとに個別の会話を維持します

詳しくは、[Sessions のドキュメント](sessions/index.md)をご覧ください。


#### サーバー管理の会話

`to_input_list()` や `Sessions` を使用してローカルで処理する代わりに、OpenAI の会話状態機能にサーバー側の会話状態を管理させることもできます。これにより、過去のすべてのメッセージを手動で再送信することなく、会話履歴を保持できます。以下のどちらのサーバー管理方式でも、各リクエストでは新しいターンの入力のみを渡し、保存した ID を再利用します。詳しくは、[OpenAI の会話状態ガイド](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses)をご覧ください。

OpenAI は、ターン間で状態を追跡するための方法を 2 つ提供しています。

##### 1. `conversation_id` の使用

まず OpenAI Conversations API を使用して会話を作成し、その後の各呼び出しでその ID を再利用します。

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

##### 2. `previous_response_id` の使用

もう 1 つの方法は **レスポンスチェーン** です。各ターンを前のターンのレスポンス ID へ明示的に関連付けます。

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

実行が承認待ちで一時停止し、[`RunState`][agents.run_state.RunState] から再開する場合、SDK は保存された `conversation_id` / `previous_response_id` / `auto_previous_response_id` の設定を維持するため、再開したターンは同じサーバー管理の会話内で継続されます。

`conversation_id` と `previous_response_id` は相互排他的です。システム間で共有できる名前付きの会話リソースが必要な場合は、`conversation_id` を使用します。ターン間で最も軽量な Responses API の継続用基本コンポーネントが必要な場合は、`previous_response_id` を使用します。

!!! note

    SDK は `conversation_locked` エラーをバックオフ付きで自動的に再試行します。サーバー管理の
    会話を使用する実行では、再試行前に内部の会話トラッカー入力を巻き戻し、
    準備済みの同じ項目を問題なく再送信できるようにします。

    ローカルのセッションベースの実行（`conversation_id`、
    `previous_response_id`、または `auto_previous_response_id` とは併用不可）では、SDK は
    再試行後の履歴項目の重複を減らすため、直近に永続化された入力項目の
    ベストエフォートなロールバックも行います。

    この互換性のための再試行は、`ModelSettings.retry` を設定していない場合でも行われます。モデルリクエストに対する
    より広範なオプトインの再試行動作については、[Runner が管理する再試行](models/index.md#runner-managed-retries)をご覧ください。

## フックとカスタマイズ

### モデル呼び出し入力フィルター

モデル呼び出しの直前にモデル入力を編集するには、`call_model_input_filter` を使用します。このフックは、現在のエージェント、コンテキスト、および結合済みの入力項目（存在する場合はセッション履歴を含む）を受け取り、新しい `ModelInputData` を返します。

戻り値は [`ModelInputData`][agents.run.ModelInputData] オブジェクトである必要があります。その `input` フィールドは必須であり、入力項目のリストでなければなりません。それ以外の形式を返すと `UserError` が発生します。

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

Runner は準備済み入力リストのコピーをフックへ渡すため、呼び出し元の元のリストをその場で変更せずに、切り詰め、置き換え、並べ替えを行えます。

セッションを使用している場合、`call_model_input_filter` はセッション履歴が読み込まれ、現在のターンとマージされた後に実行されます。それより前のマージ処理自体をカスタマイズする場合は、[`session_input_callback`][agents.run.RunConfig.session_input_callback] を使用します。

`conversation_id`、`previous_response_id`、または `auto_previous_response_id` を使用して OpenAI のサーバー管理の会話状態を利用している場合、フックは次の Responses API 呼び出し用に準備されたペイロードに対して実行されます。そのペイロードは、以前の履歴全体の再送ではなく、新しいターンの差分のみをすでに表している場合があります。返した項目だけが、そのサーバー管理の継続処理で送信済みとして記録されます。

機密データの秘匿化、長い履歴の切り詰め、追加のシステムガイダンスの注入を行うには、`run_config` を介して実行ごとにフックを設定します。

## エラーと復旧

### エラーハンドラー

すべての `Runner` エントリーポイントは、エラー種別をキーとする dict である `error_handlers` を受け入れます。サポートされるキーは `"max_turns"`、`"model_refusal"`、`"invalid_final_output"` です。対応するエラーで実行を終了する代わりに、制御された最終出力を返す場合に使用します。

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

モデルのメッセージがエージェントの構造化された `output_type` に対する検証を通過しない場合、またはモデルが構造化された最終メッセージを返さない場合は、`"invalid_final_output"` を使用します。ハンドラーはアプリケーション固有のフォールバックを返すことができ、SDK は同じ `output_type` に対してそれを検証します。モデル呼び出しの再試行や、ツールの副作用の再実行は行いません。`None` を返すと復旧を辞退します。フォールバックがない場合、空でない出力の検証失敗では引き続き `ModelBehaviorError` が発生し、空の構造化レスポンスでは既存の次ターンの動作が維持されます。

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

`RunErrorHandlerResult.include_in_history` のデフォルトは `True` です。最大ターン数のハンドラーでは、生成されたフォールバック出力を会話履歴へ追加し、設定済みのセッションに永続化します。実行結果の履歴やセッションストレージへ追加せず、呼び出し元へフォールバックを返す場合は、`include_in_history=False` を設定します。

モデルの拒否によって `ModelRefusalError` で実行を終了する代わりに、アプリケーション固有のフォールバックを生成する場合は、`"model_refusal"` を使用します。

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

## 永続実行の統合とヒューマンインザループ

ツールの承認に関する一時停止／再開パターンについては、専用の[ヒューマンインザループガイド](human_in_the_loop.md)から始めてください。以下の統合は、実行が長い待機、再試行、プロセスの再起動にまたがる可能性がある場合の永続的なオーケストレーションを目的としています。

### Dapr

Agents SDK の [Dapr](https://dapr.io) Diagrid 統合を使用すると、ヒューマンインザループをサポートし、障害から自動的に復旧する、永続的で長時間実行されるエージェントを実行できます。Dapr はベンダー中立の [CNCF](https://cncf.io) ワークフローオーケストレーターです。Dapr と OpenAI エージェントの利用は、[こちら](https://docs.diagrid.io/getting-started/quickstarts/ai-agents/?agentframework=openai)から開始できます。

### Temporal

Agents SDK の [Temporal](https://temporal.io/) 統合を使用すると、ヒューマンインザループのタスクを含む、永続的で長時間実行されるワークフローを実行できます。長時間実行タスクを完了するために Temporal と Agents SDK が連携して動作するデモは、[こちらの動画](https://www.youtube.com/watch?v=fFBZqzT4DD8)で確認できます。また、[ドキュメントはこちら](https://github.com/temporalio/sdk-python/tree/main/temporalio/contrib/openai_agents)です。 

### Restate

Agents SDK の [Restate](https://restate.dev/) 統合を使用すると、人による承認、ハンドオフ、セッション管理を含む、軽量で永続的なエージェントを実行できます。この統合は Restate の単一バイナリランタイムを依存関係として必要とし、エージェントをプロセス／コンテナまたはサーバーレス関数として実行できます。詳しくは、[概要](https://www.restate.dev/blog/durable-orchestration-for-ai-agents-with-restate-and-openai-sdk)または[ドキュメント](https://docs.restate.dev/ai)をご覧ください。

### DBOS

Agents SDK の [DBOS](https://dbos.dev/) 統合を使用すると、障害や再起動が発生しても進行状況を保持する、信頼性の高いエージェントを実行できます。長時間実行されるエージェント、ヒューマンインザループのワークフロー、ハンドオフをサポートしています。同期メソッドと非同期メソッドの両方をサポートします。この統合に必要なのは SQLite または Postgres データベースだけです。詳しくは、統合の[リポジトリ](https://github.com/dbos-inc/dbos-openai-agents)と[ドキュメント](https://docs.dbos.dev/integrations/openai-agents)をご覧ください。

## 例外

SDK は特定の場合に例外を発生させます。完全な一覧は [`agents.exceptions`][] にあります。概要は次のとおりです。

-   [`AgentsException`][agents.exceptions.AgentsException]：SDK 内で発生するすべての例外の基底クラスです。その他すべての特定の例外は、この汎用型から派生します。
-   [`MaxTurnsExceeded`][agents.exceptions.MaxTurnsExceeded]：エージェントの実行が `Runner.run`、`Runner.run_sync`、または `Runner.run_streamed` メソッドへ渡された `max_turns` 制限を超えた場合に発生します。指定された対話ターン数以内に、エージェントがタスクを完了できなかったことを示します。制限を無効にするには `max_turns=None` を設定します。
-   [`ModelBehaviorError`][agents.exceptions.ModelBehaviorError]：基盤となるモデル（LLM）が予期しない、または無効な出力を生成した場合に発生します。次のようなケースが含まれます。
    -   不正な形式の JSON：モデルがツール呼び出しまたは直接出力で不正な形式の JSON 構造を生成した場合。特に、特定の `output_type` が定義されている場合が該当します。
    -   予期しないツール関連の失敗：モデルが想定された方法でツールを使用できなかった場合
-   [`ToolTimeoutError`][agents.exceptions.ToolTimeoutError]：関数ツール呼び出しが設定済みのタイムアウトを超え、そのツールで `timeout_behavior="raise_exception"` が使用されている場合に発生します。
-   [`UserError`][agents.exceptions.UserError]：SDK を使用するコードの作成者が、SDK の使用時に誤りを犯した場合に発生します。通常は、不適切なコード実装、無効な設定、SDK API の誤用が原因です。
-   [`InputGuardrailTripwireTriggered`][agents.exceptions.InputGuardrailTripwireTriggered]、[`OutputGuardrailTripwireTriggered`][agents.exceptions.OutputGuardrailTripwireTriggered]：それぞれ、入力ガードレールまたは出力ガードレールの条件が満たされた場合に発生します。入力ガードレールは処理前に受信メッセージをチェックし、出力ガードレールは配信前にエージェントの最終レスポンスをチェックします。