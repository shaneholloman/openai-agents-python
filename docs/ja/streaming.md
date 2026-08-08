---
search:
  exclude: true
---
# ストリーミング

ストリーミングを使用すると、進行中のエージェント実行の更新を購読できます。これは、エンドユーザーに進捗状況の更新や部分的なレスポンスを表示する場合に便利です。

ストリーミングするには、[`Runner.run_streamed()`][agents.run.Runner.run_streamed] を呼び出します。これにより、[`RunResultStreaming`][agents.result.RunResultStreaming] が得られます。`result.stream_events()` を呼び出すと、以下で説明する [`StreamEvent`][agents.stream_events.StreamEvent] オブジェクトの非同期ストリームが得られます。

非同期イテレーターが終了するまで、`result.stream_events()` を消費し続けてください。ストリーミング実行は、イテレーターが終了するまで完了しません。また、セッションの永続化、承認情報の記録、履歴の圧縮などの後処理は、最後に表示されるトークンが到着した後に完了する場合があります。ループを抜けると、`result.is_complete` は最終的な実行状態を反映します。

## Raw レスポンスイベント

[`RawResponsesStreamEvent`][agents.stream_events.RawResponsesStreamEvent] オブジェクトは、LLM から直接渡される raw イベントをラップします。各オブジェクトの `data` フィールドには、`response.created` や `response.output_text.delta` などの型を持つ OpenAI Responses API イベントが格納されます。これらのイベントは、レスポンスメッセージを生成され次第ユーザーにストリーミングする場合に便利です。

コンピュータツールの raw イベントでは、保存済みの結果と同様に、プレビュー版と GA 版の区別が維持されます。プレビューのフローでは、1 つの `action` を持つ `computer_call` アイテムをストリーミングします。一方、`gpt-5.5` では、バッチ化された `actions[]` を持つ `computer_call` アイテムをストリーミングできます。上位レベルの [`RunItemStreamEvent`][agents.stream_events.RunItemStreamEvent] インターフェースでは、このためにコンピュータ専用の特別なイベント名は追加されません。どちらの形式も引き続き `tool_called` として公開され、スクリーンショットの結果は `computer_call_output` アイテムをラップする `tool_output` として返されます。

たとえば、次の例では LLM が生成したテキストをトークン単位で出力します。

```python
import asyncio
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner

async def main():
    agent = Agent(
        name="Joker",
        instructions="You are a helpful assistant.",
    )

    result = Runner.run_streamed(agent, input="Please tell me 5 jokes.")
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
```

## ストリーミングと承認

ストリーミングは、ツールの承認のために一時停止する実行にも対応しています。ツールに承認が必要な場合、`result.stream_events()` が終了し、保留中の承認が [`RunResultStreaming.interruptions`][agents.result.RunResultStreaming.interruptions] に公開されます。`result.to_state()` を使用して実行結果を [`RunState`][agents.run_state.RunState] に変換し、中断を承認または拒否してから、`Runner.run_streamed(...)` で再開します。

```python
result = Runner.run_streamed(agent, "Delete temporary files if they are no longer needed.")
async for _event in result.stream_events():
    pass

if result.interruptions:
    state = result.to_state()
    for interruption in result.interruptions:
        state.approve(interruption)
    result = Runner.run_streamed(agent, state)
    async for _event in result.stream_events():
        pass
```

一時停止と再開の完全な手順については、[人間参加型のガイド](human_in_the_loop.md)を参照してください。

## 現在のターン完了後のストリーミング停止

ストリーミング実行を途中で停止する必要がある場合は、[`result.cancel()`][agents.result.RunResultStreaming.cancel] を呼び出します。デフォルトでは、実行はすぐに停止します。現在のターンを正常に完了させてから停止するには、代わりに `result.cancel(mode="after_turn")` を呼び出します。

ストリーミング実行は、`result.stream_events()` が終了するまで完了しません。最後に表示されるトークンの後も、SDK ではセッションアイテムの永続化、承認状態の確定、履歴の圧縮が続いている可能性があります。

[`result.to_input_list(mode="normalized")`][agents.result.RunResultBase.to_input_list] から手動で続行している場合に、`cancel(mode="after_turn")` がツールのターン後に停止したときは、新しいユーザーターンをすぐに追加するのではなく、その正規化済み入力を指定して `result.last_agent` を再実行し、未完了の既存ユーザーターンを続行してください。
- ストリーミング実行がツールの承認のために停止した場合、それを新しいターンとして扱わないでください。ストリームを最後まで消費し、`result.interruptions` を確認して、`result.to_state()` から再開してください。
- [`RunConfig.session_input_callback`][agents.run.RunConfig.session_input_callback] を使用すると、取得したセッション履歴と新しいユーザー入力を、次回のモデル呼び出し前にどのように統合するかをカスタマイズできます。そこで新しいターンのアイテムを書き換えると、そのターンでは書き換え後のバージョンが永続化されます。

## 実行アイテムイベントとエージェントイベント

[`RunItemStreamEvent`][agents.stream_events.RunItemStreamEvent] は、より上位レベルのイベントです。アイテムの生成が完全に完了したときに通知されます。これにより、各トークン単位ではなく、「メッセージが生成された」「ツールが実行された」などの単位で進捗状況の更新を送信できます。同様に、[`AgentUpdatedStreamEvent`][agents.stream_events.AgentUpdatedStreamEvent] は、現在のエージェントが変更されたとき（たとえば、ハンドオフの結果として）に更新を提供します。

### 実行アイテムイベント名

`RunItemStreamEvent.name` は、固定のセマンティックイベント名を使用します。

- `message_output_created`
- `handoff_requested`
- `handoff_occured`
- `tool_called`
- `tool_search_called`
- `tool_search_output_created`
- `tool_output`
- `reasoning_item_created`
- `mcp_approval_requested`
- `mcp_approval_response`
- `mcp_list_tools`

`handoff_occured` は、後方互換性のために意図的にスペルが誤っています。

ハンドオフ呼び出しは `handoff_requested` としてのみ発行され、`tool_called` として重複して発行されることはありません。同じターン内の通常の関数ツール呼び出しでは、引き続き `tool_called` が発行されます。

ホスト型ツール検索を使用する場合、モデルがツール検索リクエストを発行すると `tool_search_called` が発行され、Responses API が読み込まれたサブセットを返すと `tool_search_output_created` が発行されます。

プログラムによるツール呼び出しでは、生成された `program` と、通常のプログラム所有の子ツール呼び出しに対して `tool_called` が発行されます。子ツールの出力と、生成された `program` に対応する `program_output` に対しては、`tool_output` が発行されます。プログラム所有のホスト型 MCP の `mcp_approval_request` アイテムと `mcp_list_tools` アイテムは例外です。それぞれ、[`MCPApprovalRequestItem`][agents.items.MCPApprovalRequestItem] をラップする `mcp_approval_requested` と、[`MCPListToolsItem`][agents.items.MCPListToolsItem] をラップする `mcp_list_tools` として発行されます。残りのアイテムを区別するには、raw アイテムの `type` を確認してください。プログラム所有の子呼び出しには `caller` も含まれ、その型は `program` で、呼び出し元 ID によって親プログラムが識別されます。

たとえば、次の例では raw イベントを無視し、更新をユーザーにストリーミングします。

```python
import asyncio
import random
from agents import Agent, ItemHelpers, Runner
from agents.decorators import tool

@tool
def how_many_jokes() -> int:
    return random.randint(1, 10)


async def main():
    agent = Agent(
        name="Joker",
        instructions="First call the `how_many_jokes` tool, then tell that many jokes.",
        tools=[how_many_jokes],
    )

    result = Runner.run_streamed(
        agent,
        input="Hello",
    )
    print("=== Run starting ===")

    async for event in result.stream_events():
        # We'll ignore the raw responses event deltas
        if event.type == "raw_response_event":
            continue
        # When the agent updates, print that
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        # When items are generated, print them
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
            else:
                pass  # Ignore other event types

    print("=== Run complete ===")


if __name__ == "__main__":
    asyncio.run(main())
```