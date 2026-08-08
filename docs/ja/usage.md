---
search:
  exclude: true
---
# 使用量

Agents SDK は、実行ごとのトークン使用量を自動的に追跡します。実行コンテキストから使用量にアクセスし、コストの監視、上限の適用、分析データの記録に利用できます。

## 追跡対象

- **requests**: LLM API の呼び出し回数
- **input_tokens**: 送信された入力トークンの合計
- **output_tokens**: 受信した出力トークンの合計
- **total_tokens**: 入力 + 出力
- **request_usage_entries**: リクエストごとの使用量内訳のリスト
- **details**:
  - `input_tokens_details.cached_tokens`
  - `output_tokens_details.reasoning_tokens`

## 実行からの使用量へのアクセス

`Runner.run(...)` の実行後、`result.context_wrapper.usage` から使用量にアクセスできます。

```python
result = await Runner.run(agent, "What's the weather in Tokyo?")
usage = result.context_wrapper.usage

print("Requests:", usage.requests)
print("Input tokens:", usage.input_tokens)
print("Output tokens:", usage.output_tokens)
print("Total tokens:", usage.total_tokens)
```

使用量は、ツール呼び出しやハンドオフを生成するモデル呼び出しを含め、実行中のすべてのモデル呼び出しを通じて集計されます。

### サードパーティーアダプターでの使用量の有効化

使用量レポートは、サードパーティーアダプターやプロバイダーのバックエンドによって異なります。サードパーティーアダプターを介してモデルにアクセスし、正確な `result.context_wrapper.usage` 値が必要な場合は、以下を確認してください。

- `AnyLLMModel` では、上流のプロバイダーが使用量を返すと、その情報が自動的に伝播されます。Chat Completions バックエンドからのレスポンスをストリーミングする場合、使用量チャンクを出力するために `ModelSettings(include_usage=True)` が必要になることがあります。
- `LitellmModel` では、一部のプロバイダーのバックエンドがデフォルトで使用量を報告しないため、多くの場合 `ModelSettings(include_usage=True)` が必要です。

モデルガイドの[サードパーティーアダプター](models/index.md#third-party-adapters)セクションにあるアダプター固有の注記を確認し、デプロイ予定のプロバイダーのバックエンドで使用量レポートを検証してください。

## リクエストごとの使用量追跡

SDK は、各 API リクエストの使用量を `request_usage_entries` で自動的に追跡します。これは、詳細なコスト計算やコンテキストウィンドウの消費量の監視に役立ちます。

```python
result = await Runner.run(agent, "What's the weather in Tokyo?")

for i, request in enumerate(result.context_wrapper.usage.request_usage_entries):
    print(f"Request {i + 1}: {request.input_tokens} in, {request.output_tokens} out")
```

## セッションでの使用量へのアクセス

`Session`（例: `SQLiteSession`）を使用する場合、`Runner.run(...)` を呼び出すたびに、その特定の実行の使用量が返されます。セッションはコンテキストとして会話履歴を保持しますが、各実行の使用量は独立しています。

```python
session = SQLiteSession("my_conversation")

first = await Runner.run(agent, "Hi!", session=session)
print(first.context_wrapper.usage.total_tokens)  # Usage for first run

second = await Runner.run(agent, "Can you elaborate?", session=session)
print(second.context_wrapper.usage.total_tokens)  # Usage for second run
```

セッションは実行間で会話コンテキストを保持しますが、各 `Runner.run()` 呼び出しによって返される使用量メトリクスは、その実行のみを表します。セッションでは、以前のメッセージが各実行への入力として再度渡される場合があり、それによって後続ターンの入力トークン数が増加します。

## フックでの使用量の利用

`RunHooks` を使用している場合、各フックに渡される `context` オブジェクトには `usage` が含まれます。これにより、ライフサイクルの主要な時点で使用量をログに記録できます。

```python
class MyHooks(RunHooks):
    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        u = context.usage
        print(f"{agent.name} → {u.requests} requests, {u.total_tokens} total tokens")
```

## API リファレンス

詳細な API ドキュメントについては、以下を参照してください。

-   [`Usage`][agents.usage.Usage] - 使用量追跡のデータ構造
-   [`RequestUsage`][agents.usage.RequestUsage] - リクエストごとの使用量の詳細
-   [`RunContextWrapper`][agents.run.RunContextWrapper] - 実行コンテキストから使用量にアクセス
-   [`RunHooks`][agents.run.RunHooks] - 使用量追跡のライフサイクルへのフック