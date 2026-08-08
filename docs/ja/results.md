---
search:
  exclude: true
---
# 実行結果

`Runner.run` メソッドを呼び出すと、次の 2 種類の実行結果のいずれかを受け取ります。

-   `Runner.run(...)` または `Runner.run_sync(...)` から返される [`RunResult`][agents.result.RunResult]
-   `Runner.run_streamed(...)` から返される [`RunResultStreaming`][agents.result.RunResultStreaming]

どちらも [`RunResultBase`][agents.result.RunResultBase] を継承しており、`final_output`、`new_items`、`last_agent`、`raw_responses`、`to_state()` などの共通の実行結果インターフェースを公開します。

`RunResultStreaming` には、[`stream_events()`][agents.result.RunResultStreaming.stream_events]、[`current_agent`][agents.result.RunResultStreaming.current_agent]、[`is_complete`][agents.result.RunResultStreaming.is_complete]、[`cancel(...)`][agents.result.RunResultStreaming.cancel] など、ストリーミング固有の制御が追加されています。

## 適切な実行結果インターフェースの選択

ほとんどのアプリケーションで必要となる実行結果のプロパティやヘルパーは、ごくわずかです。

| 必要なもの | 使用するもの |
| --- | --- |
| ユーザーに表示する最終回答 | `final_output` |
| ローカルの完全なトランスクリプトを含む、再実行可能な次ターンの入力リスト | `to_input_list()` |
| エージェント、ツール、ハンドオフ、承認のメタデータを含む詳細な実行項目 | `new_items` |
| 通常、次のユーザーターンを処理するエージェント | `last_agent` |
| `previous_response_id` を使用した OpenAI Responses API の連鎖 | `last_response_id` |
| 保留中の承認と再開可能なスナップショット | `interruptions` と `to_state()` |
| 現在のネストされた `Agent.as_tool()` 呼び出しに関するメタデータ | `agent_tool_invocation` |
| raw のモデル呼び出しまたはガードレールの診断情報 | `raw_responses` とガードレールの実行結果配列 |

## 最終出力

[`final_output`][agents.result.RunResultBase.final_output] プロパティには、最後に実行されたエージェントの最終出力が格納されます。次のいずれかになります。

-   最後のエージェントに `output_type` が定義されていない場合は `str`
-   最後のエージェントに出力型が定義されている場合は、`last_agent.output_type` 型のオブジェクト
-   承認による中断で一時停止した場合など、最終出力が生成される前に実行が停止した場合は `None`

!!! note

    `final_output` の型は `Any` です。ハンドオフによって実行を完了するエージェントが変わる可能性があるため、SDK は可能性のある出力型の完全な集合を静的に把握できません。

ストリーミングモードでは、ストリームの処理が完了するまで `final_output` は `None` のままです。イベントごとのフローについては、[ストリーミング](streaming.md)を参照してください。

## 入力、次ターンの履歴、新しい項目

これらのインターフェースは、それぞれ異なる目的に対応します。

| プロパティまたはヘルパー | 内容 | 最適な用途 |
| --- | --- | --- |
| [`input`][agents.result.RunResultBase.input] | この実行セグメントの基礎入力です。ハンドオフ入力フィルターによって履歴が書き換えられた場合は、実行の続行に使用されたフィルター済み入力が反映されます。 | この実行で実際に使用された入力の監査 |
| [`to_input_list()`][agents.result.RunResultBase.to_input_list] | 実行を入力項目として表現したものです。デフォルトの `mode="preserve_all"` では、`new_items` から変換された履歴が保持されます。ただし、SDK のデフォルトのネストされたハンドオフ履歴へすでに移動されたセッション項目と完全に同一の出現箇所が、再度追加されることはありません。ハンドオフのフィルタリングによってモデル履歴が書き換えられる場合、`mode="normalized"` は正規の継続入力を優先します。 | 手動のチャットループ、クライアント管理の会話状態、プレーン項目の履歴確認 |
| [`new_items`][agents.result.RunResultBase.new_items] | エージェント、ツール、ハンドオフ、承認のメタデータを含む詳細な [`RunItem`][agents.items.RunItem] ラッパーです。 | ログ、UI、監査、デバッグ |
| [`raw_responses`][agents.result.RunResultBase.raw_responses] | 実行内の各モデル呼び出しから得られた raw の [`ModelResponse`][agents.items.ModelResponse] オブジェクトです。 | プロバイダーレベルの診断または raw レスポンスの確認 |

実際には、次のように使い分けます。

-   実行をプレーンな入力項目として確認する場合は、`to_input_list()` を使用します。
-   ハンドオフのフィルタリングまたはネストされたハンドオフ履歴の書き換え後に、次の `Runner.run(..., input=...)` 呼び出しで使用する正規のローカル入力が必要な場合は、`to_input_list(mode="normalized")` を使用します。
-   SDK に履歴の読み込みと保存を任せる場合は、[`session=...`](sessions/index.md) を使用します。
-   `conversation_id` または `previous_response_id` を使用して OpenAI のサーバー管理状態を利用している場合は、通常、`to_input_list()` を再送信するのではなく、新しいユーザー入力のみを渡して保存済みの ID を再利用します。
-   ログ、UI、監査のために変換済みの完全な履歴が必要な場合は、デフォルトの `to_input_list()` モードまたは `new_items` を使用します。

SDK のデフォルトのネストされたハンドオフ履歴でメッセージ項目がそのまま保持される場合、Sessions、`RunState`、`to_input_list()` は、内容で重複排除するのではなく、所有対象となる個々の出現箇所を正確に追跡します。同じメッセージが別々に出現した場合、それぞれが別のものとして保持されます。すでに所有されている出現箇所だけが、再度追加されないようになります。

JavaScript SDK とは異なり、Python には、実行中に新しく生成されたモデル形式の項目だけを含む独立した `output` プロパティはありません。SDK のメタデータが必要な場合は `new_items` を使用し、raw のモデルペイロードが必要な場合は `raw_responses` を確認してください。

コンピューターツールの項目を会話入力として再送信する場合は、raw の Responses ペイロード形式が使用されます。プレビューモデルの `computer_call` 項目では単一の `action` が保持されますが、`gpt-5.5` のコンピューター呼び出しでは、バッチ化された `actions[]` を保持できます。[`to_input_list()`][agents.result.RunResultBase.to_input_list] と [`RunState`][agents.run_state.RunState] はモデルが生成した形式をそのまま保持するため、それらの項目を会話入力として手動で再送信する場合、一時停止と再開のフロー、および保存されたトランスクリプトは、プレビュー版と GA 版の両方のコンピューターツール呼び出しで引き続き動作します。ローカルの実行結果は、引き続き `new_items` 内の `computer_call_output` 項目として表示されます。

### 新しい項目

[`new_items`][agents.result.RunResultBase.new_items] では、実行中に発生した内容を最も詳細に確認できます。一般的な項目の型は次のとおりです。

-   アシスタントメッセージを表す [`MessageOutputItem`][agents.items.MessageOutputItem]
-   推論項目を表す [`ReasoningItem`][agents.items.ReasoningItem]
-   Responses のツール検索リクエストと読み込まれたツール検索結果を表す [`ToolSearchCallItem`][agents.items.ToolSearchCallItem] と [`ToolSearchOutputItem`][agents.items.ToolSearchOutputItem]
-   ツール呼び出しとその実行結果を表す [`ToolCallItem`][agents.items.ToolCallItem] と [`ToolCallOutputItem`][agents.items.ToolCallOutputItem]
-   承認待ちで一時停止したツール呼び出しを表す [`ToolApprovalItem`][agents.items.ToolApprovalItem]
-   ホスト型 MCP の承認とツールカタログを表す [`MCPApprovalRequestItem`][agents.items.MCPApprovalRequestItem]、[`MCPApprovalResponseItem`][agents.items.MCPApprovalResponseItem]、[`MCPListToolsItem`][agents.items.MCPListToolsItem]
-   ハンドオフリクエストと完了した転送を表す [`HandoffCallItem`][agents.items.HandoffCallItem] と [`HandoffOutputItem`][agents.items.HandoffOutputItem]

エージェントとの関連付け、ツールの出力、ハンドオフの境界、承認の境界が必要な場合は、`to_input_list()` ではなく `new_items` を選択してください。

ホスト型ツール検索を使用する場合は、モデルが発行した検索リクエストを確認するには `ToolSearchCallItem.raw_item` を、どの名前空間、関数、ホスト型 MCP サーバーがそのターン用に読み込まれたかを確認するには `ToolSearchOutputItem.raw_item` を参照してください。

プログラムによるツール呼び出しでは、生成された `program` は `ToolCallItem` となり、そのプログラムが所有する通常の子ツール呼び出しも `ToolCallItem` エントリとなり、対応する `program_output` は `ToolCallOutputItem` となります。プログラムが所有するホスト型 MCP の `mcp_approval_request` 項目と `mcp_list_tools` 項目は例外で、それぞれ `MCPApprovalRequestItem` エントリと `MCPListToolsItem` エントリになります。

raw 項目は、型付きの Responses オブジェクトまたはマッピングの場合があります。特に、プログラムが所有するシェル呼び出しとパッチ適用呼び出しではマッピングが使用されます。マッピングでも安全に確認できるパターンを使用してください。

```python
from collections.abc import Mapping


def raw_field(item, name):
    raw_item = item.raw_item
    if isinstance(raw_item, Mapping):
        return raw_item.get(name)
    return getattr(raw_item, name, None)


raw_type = raw_field(item, "type")
caller = raw_field(item, "caller")
caller_id = (
    caller.get("caller_id")
    if isinstance(caller, Mapping)
    else getattr(caller, "caller_id", None)
)
```

プログラムが所有する子呼び出しでは、`caller` の `type` フィールドは `program` となり、`caller_id` は親プログラム呼び出しを識別します。

## 会話の続行または再開

### 次ターンのエージェント

[`last_agent`][agents.result.RunResultBase.last_agent] には、最後に実行されたエージェントが格納されます。多くの場合、ハンドオフ後の次のユーザーターンで再利用するエージェントとして最適です。

ストリーミングモードでは、実行の進行に応じて [`RunResultStreaming.current_agent`][agents.result.RunResultStreaming.current_agent] が更新されるため、ストリームが完了する前にハンドオフを確認できます。

### 中断と実行状態

ツールで承認が必要な場合、保留中の承認は [`RunResult.interruptions`][agents.result.RunResult.interruptions] または [`RunResultStreaming.interruptions`][agents.result.RunResultStreaming.interruptions] で公開されます。これには、直接使用されたツール、ハンドオフ後に到達したツール、ネストされた [`Agent.as_tool()`][agents.agent.Agent.as_tool] の実行によって発生した承認が含まれる場合があります。

再開可能な [`RunState`][agents.run_state.RunState] を取得するには [`to_state()`][agents.result.RunResult.to_state] を呼び出し、保留中の項目を承認または拒否してから、`Runner.run(...)` または `Runner.run_streamed(...)` で再開します。

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="Use tools when needed.")
result = await Runner.run(agent, "Delete temp files that are no longer needed.")

if result.interruptions:
    state = result.to_state()
    for interruption in result.interruptions:
        state.approve(interruption)
    result = await Runner.run(agent, state)
```

ストリーミング実行では、まず [`stream_events()`][agents.result.RunResultStreaming.stream_events] の消費を完了してから `result.interruptions` を確認し、`result.to_state()` から再開します。承認フローの全体については、[Human-in-the-loop](human_in_the_loop.md)を参照してください。

### サーバー管理による継続

[`last_response_id`][agents.result.RunResultBase.last_response_id] は、実行から得られた最新のモデルレスポンス ID です。OpenAI Responses API の連鎖を継続する場合は、次のターンで `previous_response_id` として渡します。

すでに `to_input_list()`、`session`、`conversation_id` を使用して会話を継続している場合は、通常 `last_response_id` は必要ありません。複数ステップの実行に含まれるすべてのモデルレスポンスが必要な場合は、代わりに `raw_responses` を確認してください。

## エージェントをツールとして使用する際のメタデータ

ネストされた [`Agent.as_tool()`][agents.agent.Agent.as_tool] の実行から実行結果が返された場合、[`agent_tool_invocation`][agents.result.RunResultBase.agent_tool_invocation] は、それを囲む `Agent.as_tool()` 呼び出しに関する次の不変メタデータを公開します。

-   `tool_name`
-   `tool_call_id`
-   `tool_arguments`

通常のトップレベル実行では、`agent_tool_invocation` は `None` です。

これは特に `custom_output_extractor` 内で便利です。ネストされた実行結果を後処理する際に、それを囲む `Agent.as_tool()` 呼び出しのツール名、呼び出し ID、raw 引数が必要になる場合があります。関連する `Agent.as_tool()` のパターンについては、[ツール](tools.md)を参照してください。

そのネストされた実行で解析済みの構造化入力も必要な場合は、`context_wrapper.tool_input` を参照してください。これは、[`RunState`][agents.run_state.RunState] がネストされたツール入力用に汎用的にシリアル化するフィールドです。一方、`agent_tool_invocation` は、現在のネストされた呼び出しのメタデータを実行結果上で直接公開します。

## ストリーミングのライフサイクルと診断

[`RunResultStreaming`][agents.result.RunResultStreaming] は上記と同じ実行結果インターフェースを継承しますが、次のストリーミング固有の制御が追加されています。

-   セマンティックなストリームイベントを消費するための [`stream_events()`][agents.result.RunResultStreaming.stream_events]
-   実行中のアクティブなエージェントを追跡するための [`current_agent`][agents.result.RunResultStreaming.current_agent]
-   ストリーミング実行が完全に終了したかどうかを確認するための [`is_complete`][agents.result.RunResultStreaming.is_complete]
-   実行を直ちに、または現在のターンの後に停止するための [`cancel(...)`][agents.result.RunResultStreaming.cancel]

非同期イテレーターが完了するまで `stream_events()` を消費し続けてください。このイテレーターが終了するまで、ストリーミング実行は完了していません。最後に表示されるトークンが到着した後も、`final_output`、`interruptions`、`raw_responses` などの概要プロパティや、セッション永続化の副作用が確定処理中の場合があります。

`cancel()` を呼び出した場合は、キャンセルとクリーンアップが正しく完了するように、`stream_events()` を引き続き消費してください。

Python には、ストリーミング用の独立した `completed` Promise や `error` プロパティはありません。実行を終了させるストリーミングエラーは `stream_events()` によって送出され、`is_complete` は実行が終端状態に達したかどうかを示します。

### raw レスポンス

[`raw_responses`][agents.result.RunResultBase.raw_responses] には、実行中に収集された raw のモデルレスポンスが格納されます。複数ステップの実行では、ハンドオフや、モデル、ツール、モデルというサイクルの繰り返しなどによって、複数のレスポンスが生成される場合があります。

[`last_response_id`][agents.result.RunResultBase.last_response_id] は、`raw_responses` の最後のエントリの ID にすぎません。

### ガードレールの実行結果

エージェントレベルのガードレールは、[`input_guardrail_results`][agents.result.RunResultBase.input_guardrail_results] と [`output_guardrail_results`][agents.result.RunResultBase.output_guardrail_results] として公開されます。

ツールのガードレールは、[`tool_input_guardrail_results`][agents.result.RunResultBase.tool_input_guardrail_results] と [`tool_output_guardrail_results`][agents.result.RunResultBase.tool_output_guardrail_results] として個別に公開されます。

これらの配列は実行全体を通じて蓄積されるため、判断のログ記録、追加のガードレールメタデータの保存、実行がブロックされた理由のデバッグに役立ちます。

### コンテキストと使用量

[`context_wrapper`][agents.result.RunResultBase.context_wrapper] は、承認、使用量、ネストされた `tool_input` など、SDK が管理するランタイムメタデータとともに、アプリのコンテキストを公開します。

使用量は `context_wrapper.usage` で追跡されます。ストリーミング実行では、ストリームの最後のチャンクが処理されるまで、使用量の合計値の反映が遅れる場合があります。ラッパーの完全な形式と永続化に関する注意事項については、[コンテキスト管理](context.md)を参照してください。