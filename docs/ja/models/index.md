---
search:
  exclude: true
---
# モデル

Agents SDKには、すぐに使用できるOpenAIモデルのサポートが、次の 2 種類用意されています。

-   **推奨**: 新しい [Responses API](https://platform.openai.com/docs/api-reference/responses) を使用してOpenAI APIを呼び出す [`OpenAIResponsesModel`][agents.models.openai_responses.OpenAIResponsesModel]。
-   [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) を使用してOpenAI APIを呼び出す [`OpenAIChatCompletionsModel`][agents.models.openai_chatcompletions.OpenAIChatCompletionsModel]。

## モデル設定の選択

設定に適した最もシンプルな方法から始めてください。

| 目的 | 推奨される方法 | 詳細 |
| --- | --- | --- |
| OpenAIモデルのみを使用する | デフォルトのOpenAIプロバイダーと Responses モデルパスを使用する | [OpenAIモデル](#openai-models) |
| WebSocket トランスポート経由でOpenAI Responses APIを使用する | Responses モデルパスを維持し、WebSocket トランスポートを有効にする | [Responses WebSocket トランスポート](#responses-websocket-transport) |
| OpenAIがホストするサブエージェントを使用する | 実験的なホステッドマルチエージェントモデルを使用する | [ホステッドマルチエージェント](#hosted-multi-agent-experimental) |
| OpenAI以外のプロバイダーを 1 つ使用する | 組み込みのプロバイダー統合ポイントから始める | [OpenAI以外のモデル](#non-openai-models) |
| エージェント間でモデルやプロバイダーを組み合わせる | 実行ごと、またはエージェントごとにプロバイダーを選択し、機能の違いを確認する | [1 つのワークフローでのモデルの組み合わせ](#mixing-models-in-one-workflow)および[プロバイダーをまたいだモデルの組み合わせ](#mixing-models-across-providers) |
| OpenAI Responses の高度なリクエスト設定を調整する | OpenAI Responses パスで `ModelSettings` を使用する | [OpenAI Responses の高度な設定](#advanced-openai-responses-settings) |
| OpenAI以外のプロバイダーまたは複数プロバイダーのルーティングにサードパーティアダプターを使用する | サポートされているベータ版アダプターを比較し、リリース予定のプロバイダーパスを検証する | [サードパーティアダプター](#third-party-adapters) |

## OpenAIモデル

OpenAIのみを使用するほとんどのアプリでは、デフォルトのOpenAIプロバイダーで文字列のモデル名を使用し、Responses モデルパスを維持する方法を推奨します。

`Agent` の初期化時にモデルを指定しない場合は、デフォルトモデルが使用されます。現在のデフォルトは、低レイテンシーのエージェントワークフロー向けに `reasoning.effort="none"` と `verbosity="low"` を指定した [`gpt-5.4-mini`](https://developers.openai.com/api/docs/models/gpt-5.4-mini) です。アクセス権がある場合は、明示的な `model_settings` を維持しつつ、品質を高めるためにエージェントを `gpt-5.6-sol` に設定することを推奨します。

`gpt-5.6-sol` などの別のモデルへ切り替える場合、エージェントを設定する方法は 2 つあります。

### デフォルトモデル

まず、カスタムモデルが設定されていないすべてのエージェントで特定のモデルを一貫して使用するには、エージェントを実行する前に `OPENAI_DEFAULT_MODEL` 環境変数を設定します。

```bash
export OPENAI_DEFAULT_MODEL=gpt-5.6-sol
python3 my_awesome_agent.py
```

次に、`RunConfig` を使用して実行のデフォルトモデルを設定できます。エージェントにモデルを設定しない場合、この実行のモデルが使用されます。

```python
from agents import Agent, RunConfig, Runner

agent = Agent(
    name="Assistant",
    instructions="You're a helpful agent.",
)

result = await Runner.run(
    agent,
    "Hello",
    run_config=RunConfig(model="gpt-5.6-sol"),
)
```

#### GPT-5 モデル

この方法で `gpt-5.6-sol` などの GPT-5 モデルを使用すると、SDK はデフォルトの `ModelSettings` を適用します。これは、ほとんどのユースケースで最適に機能する設定です。デフォルトモデルの推論労力を調整するには、独自の `ModelSettings` を渡します。

```python
from openai.types.shared import Reasoning
from agents import Agent, ModelSettings

my_agent = Agent(
    name="My Agent",
    instructions="You're a helpful agent.",
    # If OPENAI_DEFAULT_MODEL=gpt-5.6-sol is set, passing only model_settings works.
    # It's also fine to pass a GPT-5 model name explicitly:
    model="gpt-5.6-sol",
    model_settings=ModelSettings(reasoning=Reasoning(effort="high"), verbosity="low")
)
```

レイテンシーを低くするには、GPT-5 モデルで `reasoning.effort="none"` を使用することを推奨します。

GPT-5.6 は、既存の `reasoning` 設定を通じて、推論モード、会話ターン間で引き継がれる推論コンテキスト、および `"max"` の労力レベルもサポートします。これらの制御は Responses API パスで利用できます。

```python
from openai.types.shared import Reasoning
from agents import Agent, ModelSettings

agent = Agent(
    name="Deep research agent",
    model="gpt-5.6-sol",
    model_settings=ModelSettings(
        reasoning=Reasoning(
            mode="pro",
            effort="max",
            context="all_turns",
        ),
    ),
)
```

`reasoning.mode` と `reasoning.context` は、Responses 専用の設定です。Chat Completions では `reasoning.effort` のみが使用され、サポートされる労力レベルはモデルと API サーフェスによって異なります。GPT-5.6 の `"max"` 労力には Responses APIを使用してください。Chat Completions アダプターは、警告を表示してモードとコンテキストを無視します。その警告をエラーに変えるには、OpenAIプロバイダーで `strict_feature_validation=True` を設定してください。

`context="all_turns"` を使用する場合は、`previous_response_id`、サーバー側の Responses API 会話、または前回の推論項目を次のリクエストに含めることで会話を維持してください。ステートレスな `store=False` 呼び出しでは、レスポンス内の `reasoning.encrypted_content` をリクエストし、それらの推論項目を次のリクエストの入力に含めてください。

#### ComputerTool のモデル選択

エージェントに [`ComputerTool`][agents.tool.ComputerTool] が含まれている場合、実際の Responses リクエストで有効なモデルによって、SDK が送信するコンピューターツールのペイロードが決まります。明示的な `gpt-5.5` リクエストでは、GA の組み込み `computer` ツールが使用されます。一方、明示的な `computer-use-preview` リクエストでは、従来の `computer_use_preview` ペイロードが維持されます。

主な例外は、プロンプトによって管理される呼び出しです。プロンプトテンプレートでモデルを指定し、SDK がリクエストから `model` を省略する場合、プロンプトに固定されているモデルを推測しないように、SDK はプレビュー互換のコンピューターペイロードをデフォルトで使用します。このフローで GA パスを維持するには、リクエストで `model="gpt-5.5"` を明示するか、`ModelSettings(tool_choice="computer")` または `ModelSettings(tool_choice="computer_use")` を使用して GA セレクターを強制してください。

[`ComputerTool`][agents.tool.ComputerTool] が登録されている場合、`tool_choice="computer"`、`"computer_use"`、および `"computer_use_preview"` は、有効なリクエストモデルと一致する組み込みセレクターに正規化されます。`ComputerTool` が登録されていない場合、これらの文字列は引き続き通常の関数名として動作します。

プレビュー互換のリクエストでは、`environment` とディスプレイ寸法を事前にシリアライズする必要があります。そのため、[`ComputerProvider`][agents.tool.ComputerProvider] ファクトリーを使用するプロンプト管理フローでは、具体的な `Computer` または `AsyncComputer` インスタンスを渡すか、リクエストを送信する前に GA セレクターを強制する必要があります。移行の詳細については、[ツール](../tools.md#computertool-and-the-responses-computer-tool)を参照してください。

#### GPT-5 以外のモデル

カスタム `model_settings` を指定せずに GPT-5 以外のモデル名を渡すと、SDK はどのモデルとも互換性のある汎用の `ModelSettings` に戻します。

### Responses 専用のツール機能

次のツール機能は、OpenAI Responses モデルでのみサポートされています。

-   [`ToolSearchTool`][agents.tool.ToolSearchTool]
-   [`tool_namespace()`][agents.tool.tool_namespace]
-   `@function_tool(defer_loading=True)` および、遅延読み込みに対応するその他の Responses ツールサーフェス
-   [`ProgrammaticToolCallingTool`][agents.tool.ProgrammaticToolCallingTool]、`allowed_callers`、および `tool_choice="programmatic_tool_calling"`

これらの機能は、Chat Completions モデルおよび Responses 以外のバックエンドでは拒否されます。遅延読み込みツールを使用する場合は、エージェントに `ToolSearchTool()` を追加し、ネームスペース名のみ、または遅延読み込み専用の関数名を強制するのではなく、`auto` または `required` のツール選択を通じてモデルにツールを読み込ませてください。設定の詳細と現在の制約については、[ホステッドツール検索](../tools.md#hosted-tool-search)および[プログラマティックツール呼び出し](../tools.md#programmatic-tool-calling)を参照してください。

### Responses WebSocket トランスポート

デフォルトでは、OpenAI Responses APIリクエストは HTTP トランスポートを使用します。OpenAI Responses プロバイダーパスを使用する場合、WebSocket トランスポートをオプトインで有効にできます。

#### 基本設定

```python
from agents import set_default_openai_responses_transport

set_default_openai_responses_transport("websocket")
```

これは、デフォルトのOpenAIプロバイダーがモデル名を解決した結果として得られるOpenAI Responses モデルに影響します。これには、`"gpt-5.6-sol"` などの文字列モデル名も含まれます。

トランスポートの選択は、SDK がモデル名をモデルインスタンスへ解決するときに行われます。具体的な [`Model`][agents.models.interface.Model] オブジェクトを渡す場合、そのトランスポートはすでに固定されています。[`OpenAIResponsesWSModel`][agents.models.openai_responses.OpenAIResponsesWSModel] は WebSocket、[`OpenAIResponsesModel`][agents.models.openai_responses.OpenAIResponsesModel] は HTTP を使用し、[`OpenAIChatCompletionsModel`][agents.models.openai_chatcompletions.OpenAIChatCompletionsModel] は Chat Completions のままです。`RunConfig(model_provider=...)` を渡す場合、グローバルデフォルトではなく、そのプロバイダーがトランスポートの選択を制御します。

#### プロバイダー単位または実行単位の設定

プロバイダーごと、または実行ごとに WebSocket トランスポートを設定することもできます。

```python
from agents import Agent, OpenAIProvider, RunConfig, Runner

provider = OpenAIProvider(
    use_responses_websocket=True,
    # Optional; if omitted, OPENAI_WEBSOCKET_BASE_URL is used when set.
    websocket_base_url="wss://your-proxy.example/v1",
    # Optional low-level websocket keepalive settings.
    responses_websocket_options={"ping_interval": 20.0, "ping_timeout": 60.0},
)

agent = Agent(name="Assistant")
result = await Runner.run(
    agent,
    "Hello",
    run_config=RunConfig(model_provider=provider),
)
```

SDK のOpenAI統合を経由するプロバイダーは、オプションのエージェント登録設定も受け入れます。これは、OpenAI設定でハーネス ID などのプロバイダーレベルの登録メタデータが必要となる場合の高度なオプションです。

```python
from agents import (
    Agent,
    OpenAIAgentRegistrationConfig,
    OpenAIProvider,
    RunConfig,
    Runner,
)

provider = OpenAIProvider(
    use_responses_websocket=True,
    agent_registration=OpenAIAgentRegistrationConfig(harness_id="your-harness-id"),
)

agent = Agent(name="Assistant")
result = await Runner.run(
    agent,
    "Hello",
    run_config=RunConfig(model_provider=provider),
)
```

#### `MultiProvider` による高度なルーティング

プレフィックスに基づくモデルルーティングが必要な場合、たとえば 1 回の実行で `openai/...` と `any-llm/...` のモデル名を組み合わせる場合は、[`MultiProvider`][agents.MultiProvider] を使用し、そこで `openai_use_responses_websocket=True` を設定してください。

`MultiProvider` は、過去から引き継がれた次の 2 つのデフォルト動作を維持します。

-   `openai/...` はOpenAIプロバイダーのエイリアスとして扱われるため、`openai/gpt-4.1` はモデル `gpt-4.1` としてルーティングされます。
-   不明なプレフィックスは、そのまま渡されるのではなく `UserError` を発生させます。

OpenAIプロバイダーを、リテラルなネームスペース付きモデル ID を要求するOpenAI互換エンドポイントへ接続する場合は、パススルー動作を明示的にオプトインしてください。WebSocket を有効にした設定では、`MultiProvider` でも `openai_use_responses_websocket=True` を維持してください。

```python
from agents import Agent, MultiProvider, RunConfig, Runner

provider = MultiProvider(
    openai_base_url="https://openrouter.ai/api/v1",
    openai_api_key="...",
    openai_use_responses_websocket=True,
    openai_prefix_mode="model_id",
    unknown_prefix_mode="model_id",
)

agent = Agent(
    name="Assistant",
    instructions="Be concise.",
    model="openai/gpt-4.1",
)

result = await Runner.run(
    agent,
    "Hello",
    run_config=RunConfig(model_provider=provider),
)
```

バックエンドがリテラルな `openai/...` 文字列を要求する場合は、`openai_prefix_mode="model_id"` を使用してください。バックエンドが `openrouter/openai/gpt-4.1-mini` などの他のネームスペース付きモデル ID を要求する場合は、`unknown_prefix_mode="model_id"` を使用してください。これらのオプションは、WebSocket トランスポート外の `MultiProvider` でも機能します。この例で WebSocket を有効なままにしているのは、このセクションで説明しているトランスポート設定の一部であるためです。同じオプションは [`responses_websocket_session()`][agents.responses_websocket_session] でも利用できます。

`MultiProvider` を通じてルーティングしながら、同じプロバイダーレベルの登録メタデータが必要な場合は、`openai_agent_registration=OpenAIAgentRegistrationConfig(...)` を渡してください。これは基盤となるOpenAIプロバイダーへ転送されます。

カスタムのOpenAI互換エンドポイントまたはプロキシを使用する場合、WebSocket トランスポートには互換性のある WebSocket `/responses` エンドポイントも必要です。このような設定では、`websocket_base_url` を明示的に設定する必要がある場合があります。

#### 注意事項

-   これは WebSocket トランスポート経由の Responses APIであり、[Realtime API](../realtime/guide.md) ではありません。Chat Completions には適用されません。OpenAI以外のプロバイダーには、そのプロバイダーが Responses WebSocket の `/responses` エンドポイントをサポートしている場合にのみ適用されます。
-   環境にまだ存在しない場合は、`websockets` パッケージをインストールしてください。
-   WebSocket トランスポートを有効にした後、[`Runner.run_streamed()`][agents.run.Runner.run_streamed] を直接使用できます。複数ターンにわたり同じ WebSocket 接続を再利用するワークフローでは、ネストされた Agents-as-tools 呼び出しも含め、[`responses_websocket_session()`][agents.responses_websocket_session] ヘルパーを推奨します。[エージェントの実行](../running_agents.md)ガイドおよび [`examples/basic/stream_ws.py`](https://github.com/openai/openai-agents-python/tree/main/examples/basic/stream_ws.py) を参照してください。
-   長時間の推論ターンやレイテンシーが急増するネットワークでは、`responses_websocket_options` を使用して WebSocket のキープアライブ動作をカスタマイズしてください。遅延した pong フレームを許容するには `ping_timeout` を増やします。ping を有効にしたままハートビートのタイムアウトを無効にするには、`ping_timeout=None` を設定します。WebSocket のレイテンシーより信頼性を重視する場合は、HTTP/SSE トランスポートを使用してください。
-   SDK はデフォルトで、受信メッセージサイズの上限を無効にします（`max_size=None`）。プロキシの背後にある長時間稼働するエージェントプロセスや、メモリーが制限されたコンテナでは、メッセージごとのメモリー使用量を制限するために `responses_websocket_options={"max_size": 8 * 1024 * 1024}` を設定してください。
-   [Responses API WebSocket サービス](https://developers.openai.com/api/docs/guides/websocket-mode)は、各接続で一度に 1 つのレスポンスを処理し、各接続を 60 分に制限します。その上限を超えたら新しい接続を開いてください。並列実行が必要な場合は、複数の接続を使用してください。
-   サービスは、接続ローカルのメモリーに最新のレスポンスのみを保持します。失敗した `4xx` または `5xx` のターンでは、`previous_response_id` が参照するレスポンスがそのメモリーから削除されます。再接続後も、保存済みのレスポンスが利用可能であれば処理を継続できますが、`store=False` と ZDR フローには永続化されたフォールバックがありません。`previous_response_id=None` で新しいチェーンを開始して入力コンテキスト全体を送信するか、ローカルで管理するセッション状態からそのコンテキストを再構築してください。

### ホステッドマルチエージェント（実験的）

OpenAI Responses APIのホステッドマルチエージェントベータでは、GPT-5.6 のルートモデルが、サーバーでホストされるサブエージェントを作成して連携させることができます。Agents SDKは通常の `Runner` を引き続き使用できます。ホステッドオーケストレーションはサービス上に留まり、開発者が定義した関数ツールはアプリケーション内で実行されます。

この統合は実験的であり、ローカル関数の出力を `response.inject` によりアクティブなホステッドエージェントへ返せるように、Responses WebSocket トランスポートを使用します。`client.beta.responses.connect` を公開する、バージョン 2.45.0 以降の `openai[realtime]` ビルドが必要です。インターフェースとベータ項目のスキーマは、一般提供までに変更される可能性があります。

#### モデルの設定

実験的モジュールからモデルをインポートし、SDK の `Agent` に割り当てます。

```python
from agents import Agent
from agents.extensions.experimental.hosted_multi_agent import OpenAIHostedMultiAgentModel

agent = Agent(
    name="Research coordinator",
    instructions="Delegate independent research tasks, then synthesize the findings.",
    model=OpenAIHostedMultiAgentModel(model="gpt-5.6-sol", config={"max_concurrent_subagents": 3}),
)
```

`OpenAIHostedMultiAgentModel` を構築すると `multi_agent.enabled` が有効になり、`OpenAI-Beta: responses_multi_agent=v1` WebSocket ヘッダーが送信されます。`openai_client` を指定しない場合、モデルはデフォルトのOpenAIクライアントを使用します。`max_concurrent_subagents` を省略すると、サービスのデフォルトが使用されます。

#### ローカル関数ツール

すべてのホステッドエージェントは、リクエストに設定されたモデルとツールを共有します。Responses APIは、どのホステッドエージェントが関数を呼び出すかを決定します。通常の SDK Runner は関数をローカルで実行し、同じ呼び出し ID を持つ `function_call_output` をアクティブな WebSocket レスポンスへ挿入します。これにより、サービスは元のホステッド呼び出し元を再開できます。関数の実行には、Runner の通常のガードレール、フック、および失敗時の変換が引き続き適用されます。SDK のツール承認による中断はサポートされていません。`needs_approval` 設定が `False` ではない関数ツールは、リクエストの送信前に拒否されます。

呼び出し元を考慮したログ記録や認可がツールに必要な場合は、`get_hosted_agent_metadata()` を使用してください。

```python
from typing import Any

from agents.decorators import tool
from agents.extensions.experimental.hosted_multi_agent import get_hosted_agent_metadata
from agents.tool_context import ToolContext

@tool
def lookup_document(ctx: ToolContext[Any], section: str) -> str:
    metadata = get_hosted_agent_metadata(ctx)
    caller = metadata.agent_name if metadata else "unknown"
    print(f"tool caller: {caller}; call ID: {ctx.tool_call_id}")
    return f"Contents for {section}"
```

ホステッドエージェント名は観測用のメタデータであり、ローカルルーティングの仕組みではありません。SDK が提供する呼び出し ID を使用して出力をルーティングしてください。副作用のあるツールでは、その呼び出し ID を冪等性キーとして使用し、必要な認可をツール実行前または実行中にアプリケーションコードで適用してください。このモデルでは `needs_approval` を使用しないでください。ツールの引数と出力は Responses APIの境界を越えます。

#### 出力とストリーミングの動作

フェーズが `final_answer` で、`/root` に帰属するメッセージのみが、通常の最終メッセージになります。実験的アダプターは、サブエージェントのメッセージとホステッドオーケストレーションの記録を高レベルの `RunResult` から除外します。SDK がそれらの記録をローカル関数として実行することはありません。

raw ストリーミングでは、ホステッド出力項目や `response.inject.created` の確認応答を含む、ベータ版 Responses イベントが引き続き公開されます。アダプターは、関数呼び出しの準備が整うと、アクティブな 1 つのプロバイダーレスポンスを SDK から見える論理的なモデルターンに分割し、Runner が出力を生成した後に同じプロバイダーレスポンスを再開します。raw のホステッド項目または `ToolContext` とともに `get_hosted_agent_metadata()` を使用すると、項目またはツール呼び出しが帰属するホステッドエージェントを識別できます。

#### SDK オーケストレーションとの関係

ホステッドマルチエージェントは、SDK のハンドオフおよび Agents-as-tools とは別のものです。

-   ホステッドマルチエージェントは、OpenAIサービス上にサブエージェントを作成します。アプリケーションがそれらのサブエージェントを作成またはスケジュールすることはありません。
-   SDK のハンドオフは、アクティブなローカル SDK の `Agent` を変更します。すべてのホステッドエージェントが同じハンドオフツールを受け取り、所有権の競合が生じるため、この実験的モデルを使用している場合は拒否されます。
-   Agents-as-tools は引き続き利用できますが、使用するとクライアント側とサーバー側のオーケストレーションがネストされます。追加のレイテンシー、コスト、およびツールの公開範囲を慎重に評価してください。

#### 現在の制限事項

実験的モデルは、`reasoning.summary`、`max_tool_calls`、および呼び出し元が指定する `multi_agent` または `betas` のオーバーライドを拒否します。Responses の `/compact` エンドポイントはベータ版ではサポートされていません。ただし、サービスが各ホステッドエージェントのコンテキストを個別に自動圧縮するため、明示的な `context_management.compact_threshold` は使用できます。

1 つの `OpenAIHostedMultiAgentModel` インスタンスが所有できるアクティブなホステッドレスポンスは、一度に最大 1 つです。ローカル関数の出力を待機中に実行が放棄された場合は、`await model.close()` を呼び出して WebSocket を解放してください。進行中のホステッドレスポンスを別のプロセスまたはイベントループで復元することは、現在サポートされていません。

基盤となる Responses APIベータの動作については、[OpenAIマルチエージェントガイド](https://developers.openai.com/api/docs/guides/tools-multi-agent)を参照してください。非ストリーミングおよびストリーミングでの SDK の使用方法については、[`examples/agent_patterns/hosted_multi_agent_beta.py`](https://github.com/openai/openai-agents-python/tree/main/examples/agent_patterns/hosted_multi_agent_beta.py) を参照してください。

## OpenAI以外のモデル

OpenAI以外のプロバイダーが必要な場合は、SDK の組み込みプロバイダー統合ポイントから始めてください。多くの設定では、サードパーティアダプターを追加しなくてもこれで十分です。各パターンのコード例は、[examples/model_providers](https://github.com/openai/openai-agents-python/tree/main/examples/model_providers/) にあります。

### OpenAI以外のプロバイダーの統合方法

| 方法 | 使用する状況 | 適用範囲 |
| --- | --- | --- |
| [`set_default_openai_client`][agents.set_default_openai_client] | 1 つのOpenAI互換エンドポイントを、ほとんどまたはすべてのエージェントのデフォルトにする場合 | グローバルデフォルト |
| [`ModelProvider`][agents.models.interface.ModelProvider] | 1 つのカスタムプロバイダーを単一の実行に適用する場合 | 実行単位 |
| [`Agent.model`][agents.agent.Agent.model] | エージェントごとに異なるプロバイダーまたは具体的なモデルオブジェクトが必要な場合 | エージェント単位 |
| サードパーティアダプター | 組み込みの方法では提供されないプロバイダー対応やルーティングが必要な場合 | [サードパーティアダプター](#third-party-adapters)を参照 |

次の組み込み方法で、他の LLM プロバイダーを統合できます。

1. [`set_default_openai_client`][agents.set_default_openai_client] は、`AsyncOpenAI` のインスタンスを LLM クライアントとしてグローバルに使用する場合に便利です。これは、LLM プロバイダーにOpenAI互換 API エンドポイントがあり、`base_url` と `api_key` を設定できる場合に使用します。設定可能なコード例については、[examples/model_providers/custom_example_global.py](https://github.com/openai/openai-agents-python/tree/main/examples/model_providers/custom_example_global.py) を参照してください。
2. [`ModelProvider`][agents.models.interface.ModelProvider] は `Runner.run` レベルで使用します。これにより、「この実行内のすべてのエージェントでカスタムモデルプロバイダーを使用する」と指定できます。設定可能なコード例については、[examples/model_providers/custom_example_provider.py](https://github.com/openai/openai-agents-python/tree/main/examples/model_providers/custom_example_provider.py) を参照してください。
3. [`Agent.model`][agents.agent.Agent.model] を使用すると、特定の Agent インスタンスにモデルを指定できます。これにより、エージェントごとに異なるプロバイダーを組み合わせられます。設定可能なコード例については、[examples/model_providers/custom_example_agent.py](https://github.com/openai/openai-agents-python/tree/main/examples/model_providers/custom_example_agent.py) を参照してください。

`platform.openai.com` の API キーがない場合は、`set_tracing_disabled()` を使用してトレーシングを無効にするか、[別のトレーシングプロセッサー](../tracing.md)を設定することを推奨します。

``` python
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled

set_tracing_disabled(disabled=True)

client = AsyncOpenAI(api_key="Api_Key", base_url="Base URL of Provider")
model = OpenAIChatCompletionsModel(model="Model_Name", openai_client=client)

agent= Agent(name="Helping Agent", instructions="You are a Helping Agent", model=model)
```

!!! note

    これらのコード例では、多くの LLM プロバイダーがまだ Responses APIをサポートしていないため、Chat Completions APIおよびモデルを使用しています。LLM プロバイダーが Responses をサポートしている場合は、Responses の使用を推奨します。

## 1 つのワークフローでのモデルの組み合わせ

単一のワークフロー内で、エージェントごとに異なるモデルを使用したい場合があります。たとえば、トリアージには小さく高速なモデルを使用し、複雑なタスクには大きく高性能なモデルを使用できます。[`Agent`][agents.Agent] を設定するときは、次のいずれかの方法で特定のモデルを選択できます。

1. モデル名を渡します。
2. 任意のモデル名と、その名前を Model インスタンスにマッピングできる [`ModelProvider`][agents.models.interface.ModelProvider] を渡します。
3. [`Model`][agents.models.interface.Model] の実装を直接指定します。

!!! note

    SDK は [`OpenAIResponsesModel`][agents.models.openai_responses.OpenAIResponsesModel] と [`OpenAIChatCompletionsModel`][agents.models.openai_chatcompletions.OpenAIChatCompletionsModel] の両方の形式をサポートしていますが、両者ではサポートする機能とツールの組み合わせが異なるため、ワークフローごとに単一のモデル形式を使用することを推奨します。ワークフローでモデル形式を組み合わせる必要がある場合は、使用するすべての機能が両方で利用できることを確認してください。

```python
import asyncio

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
    model="gpt-5-mini", # (1)!
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model=OpenAIChatCompletionsModel( # (2)!
        model="gpt-5-nano",
        openai_client=AsyncOpenAI()
    ),
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
    model="gpt-5.6-sol",
)

async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

1.  OpenAIモデルの名前を直接設定します。
2.  [`Model`][agents.models.interface.Model] の実装を指定します。

エージェントが使用するモデルをさらに設定する場合は、temperature などのオプションのモデル設定パラメーターを提供する [`ModelSettings`][agents.model_settings.ModelSettings] を渡せます。

```python
from agents import Agent, ModelSettings

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=0.1),
)
```

## OpenAI Responses の高度な設定

OpenAI Responses パスを使用しており、より細かな制御が必要な場合は、`ModelSettings` から始めてください。

### 一般的な高度な `ModelSettings` オプション

OpenAI Responses APIを使用している場合、いくつかのリクエストフィールドには対応する `ModelSettings` フィールドがすでに用意されているため、それらに `extra_args` を使用する必要はありません。

- `parallel_tool_calls`: 同じターンで複数のツール呼び出しを許可または禁止します。
- `truncation`: コンテキストが上限を超える場合に失敗する代わりに、Responses APIが最も古い会話項目を削除できるよう、`"auto"` を設定します。
- `store`: 生成されたレスポンスを後で取得できるよう、サーバー側に保存するかどうかを制御します。これは、レスポンス ID に依存する後続ワークフローや、`store=False` の場合にローカル入力へフォールバックする必要があるセッション圧縮フローに関係します。
- `context_management`: `compact_threshold` による Responses の圧縮など、サーバー側のコンテキスト処理を設定します。
- `prompt_cache_retention`: 以前のモデルファミリー向けの延長保持を設定します。たとえば、
  `"24h"` を使用します。
- `prompt_cache_options`: 暗黙的または明示的なプロンプトキャッシュを選択し、GPT-5.6 では `"30m"` のキャッシュ TTL を設定します。
- `response_include`: `web_search_call.action.sources`、`file_search_call.results`、`reasoning.encrypted_content` など、より詳細なレスポンスペイロードをリクエストします。
- `top_logprobs`: 出力テキストについて上位トークンの logprobs をリクエストします。SDK は `message.output_text.logprobs` も自動的に追加します。
- `retry`: モデル呼び出しについて Runner が管理する再試行設定をオプトインで有効にします。[Runner が管理する再試行](#runner-managed-retries)を参照してください。

```python
from agents import Agent, ModelSettings

research_agent = Agent(
    name="Research agent",
    model="gpt-5.6-sol",
    model_settings=ModelSettings(
        parallel_tool_calls=False,
        truncation="auto",
        store=True,
        context_management=[{"type": "compaction", "compact_threshold": 200000}],
        prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        response_include=["web_search_call.action.sources"],
        top_logprobs=5,
    ),
)
```

明示的なプロンプトキャッシュでは、再利用可能なプレフィックスの末尾となるコンテンツ部分にブレークポイントを追加します。同じ `ModelSettings.prompt_cache_options` フィールドが Responses と Chat Completions のリクエストにそのまま渡され、Chat Completions コンバーターはテキスト、画像、音声、およびファイルのコンテンツ部分にあるブレークポイントを維持します。

```python
from agents import Runner

result = await Runner.run(
    research_agent,
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Reusable background material...",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                },
                {
                    "type": "input_text",
                    "text": "Analyze the latest question.",
                },
            ],
        }
    ],
)
```

従来の保持制御を使用する以前のモデルファミリーでは、`prompt_cache_retention` を引き続き利用できます。
直接指定する `ModelSettings` フィールドと、`extra_args` 内の同じキーを組み合わせないでください。

`store=False` を設定すると、Responses APIはそのレスポンスを後でサーバー側から取得できる状態で保持しません。これは、ステートレスまたはゼロデータ保持形式のフローに便利ですが、通常ならレスポンス ID を再利用する機能が、代わりにローカルで管理される状態に依存する必要があることも意味します。たとえば、[`OpenAIResponsesCompactionSession`][agents.memory.openai_responses_compaction_session.OpenAIResponsesCompactionSession] は、最後のレスポンスが保存されていない場合、デフォルトの `"auto"` 圧縮パスを入力ベースの圧縮に切り替えます。[セッションガイド](../sessions/index.md#openai-responses-compaction-sessions)を参照してください。

サーバー側の圧縮は、[`OpenAIResponsesCompactionSession`][agents.memory.openai_responses_compaction_session.OpenAIResponsesCompactionSession] とは異なります。`context_management=[{"type": "compaction", "compact_threshold": ...}]` は各 Responses APIリクエストとともに送信され、レンダリングされたコンテキストがしきい値を超えると、API はレスポンスの一部として圧縮項目を生成できます。`OpenAIResponsesCompactionSession` はターン間でスタンドアロンの `responses.compact` エンドポイントを呼び出し、ローカルのセッション履歴を書き換えます。

### `extra_args` の受け渡し

SDK がまだトップレベルで直接公開していない、プロバイダー固有または新しいリクエストフィールドが必要な場合は、`extra_args` を使用してください。

OpenAIモデルを使用する場合、`extra_args` は Responses APIと Chat Completions APIの両方にオプションのパラメーターを渡せます。たとえば、`user` や `service_tier` です。サポートされているモデルで [Fast モード](https://developers.openai.com/api/docs/guides/fast-mode)を使用するには、`extra_args={"service_tier": "fast"}` を設定してください。`"priority"` も同等です。同じリクエストフィールドを、直接指定する `ModelSettings` フィールドでも設定しないでください。

```python
from agents import Agent, ModelSettings

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
    model="gpt-4.1",
    model_settings=ModelSettings(
        temperature=0.1,
        extra_args={"service_tier": "flex", "user": "user_12345"},
    ),
)
```

## Runner が管理する再試行

再試行はランタイム専用であり、オプトイン方式です。`ModelSettings(retry=...)` を設定し、再試行ポリシーが再試行を選択しない限り、SDK は一般的なモデルリクエストを再試行しません。

Responses WebSocket トランスポートでは、`retry_policies.provider_suggested()` はレスポンス前の過負荷フレームと、コードのない `server_error` フレームを再試行の提案として認識します。これだけでは再試行は有効になりません。引き続き `ModelRetrySettings` が必要であり、通常のリプレイ安全性チェックも適用されます。レスポンスイベントが 1 つでも到着済みの場合、SDK はリクエストをリプレイしません。

```python
from agents import Agent, ModelRetrySettings, ModelSettings, retry_policies

agent = Agent(
    name="Assistant",
    model="gpt-5.6-sol",
    model_settings=ModelSettings(
        retry=ModelRetrySettings(
            max_retries=4,
            backoff={
                "initial_delay": 0.5,
                "max_delay": 5.0,
                "multiplier": 2.0,
                "jitter": True,
            },
            policy=retry_policies.any(
                retry_policies.provider_suggested(),
                retry_policies.retry_after(),
                retry_policies.network_error(),
                retry_policies.http_status([408, 409, 429, 500, 502, 503, 504]),
            ),
        )
    ),
)
```

`ModelRetrySettings` には、次の 3 つのフィールドがあります。

<div class="field-table" markdown="1">

| フィールド | 型 | 注記 |
| --- | --- | --- |
| `max_retries` | `int | None` | 最初のリクエスト後に許可される再試行回数です。 |
| `backoff` | `ModelRetryBackoffSettings | dict | None` | ポリシーが明示的な遅延を返さずに再試行する場合のデフォルト遅延戦略です。`backoff.max_delay` は、この計算されたバックオフ遅延のみを制限します。ポリシーから返された明示的な遅延や retry-after ヒントは制限しません。 |
| `policy` | `RetryPolicy | None` | 再試行するかどうかを決定するコールバックです。このフィールドはランタイム専用であり、シリアライズされません。 |

</div>

再試行ポリシーは、次の情報を持つ [`RetryPolicyContext`][agents.retry.RetryPolicyContext] を受け取ります。

- `attempt` と `max_retries`。これにより、試行回数を考慮した判断ができます。
- `stream`。これにより、ストリーミング動作と非ストリーミング動作を分岐できます。
- raw の検査に使用する `error`。
- `status_code`、`retry_after`、`error_code`、`is_network_error`、`is_timeout`、`is_abort` などの `normalized` 情報。
- 基盤となるモデルアダプターが再試行の指針を提供できる場合の `provider_advice`。

ポリシーは、次のいずれかを返せます。

- 単純な再試行判断を示す `True` / `False`。
- 遅延をオーバーライドするか、診断理由を付加する場合の [`RetryDecision`][agents.retry.RetryDecision]。

SDK は、`retry_policies` で既製のヘルパーをエクスポートします。

| ヘルパー | 動作 |
| --- | --- |
| `retry_policies.never()` | 常にオプトアウトします。 |
| `retry_policies.provider_suggested()` | 利用可能な場合、プロバイダーの再試行アドバイスに従います。 |
| `retry_policies.network_error()` | 一時的なトランスポート障害およびタイムアウト障害に一致します。 |
| `retry_policies.http_status([...])` | 選択された HTTP ステータスコードに一致します。 |
| `retry_policies.retry_after()` | retry-after ヒントが利用可能な場合にのみ、その遅延を使用して再試行します。このヘルパーは retry-after 値を明示的なポリシー遅延として扱うため、`backoff.max_delay` はその値を制限しません。 |
| `retry_policies.any(...)` | ネストされたポリシーのいずれかがオプトインした場合に再試行します。 |
| `retry_policies.all(...)` | ネストされたすべてのポリシーがオプトインした場合にのみ再試行します。 |

ポリシーを組み合わせる場合、`provider_suggested()` は最も安全な最初の構成要素です。プロバイダーがそれらを区別できる場合に、プロバイダーによる拒否とリプレイ安全性の承認を維持するためです。

##### 安全性の境界

一部の障害は自動的に再試行されません。

- 中止エラー。
- プロバイダーのアドバイスでリプレイが安全でないと示されたリクエスト。
- 出力がすでに開始され、リプレイが安全でなくなるストリーミング実行。

`previous_response_id` または `conversation_id` を使用するステートフルな後続リクエストも、より慎重に扱われます。これらのリクエストでは、`network_error()` や `http_status([500])` など、プロバイダー由来ではない述語だけでは不十分です。再試行ポリシーには、通常は `retry_policies.provider_suggested()` を通じて、プロバイダーからリプレイが安全であるという承認を含める必要があります。

##### Runner とエージェントのマージ動作

`retry` は、Runner レベルとエージェントレベルの `ModelSettings` の間でディープマージされます。

- エージェントは `retry.max_retries` のみをオーバーライドし、Runner の `policy` を引き継げます。
- エージェントは `retry.backoff` の一部のみをオーバーライドし、Runner の他のバックオフフィールドを維持できます。
- `policy` はランタイム専用であるため、シリアライズされた `ModelSettings` は `max_retries` と `backoff` を維持しますが、コールバック自体は省略します。

さらに詳しいコード例については、[`examples/basic/retry.py`](https://github.com/openai/openai-agents-python/tree/main/examples/basic/retry.py) および[アダプターを利用した再試行のコード例](https://github.com/openai/openai-agents-python/tree/main/examples/basic/retry_litellm.py)を参照してください。

## OpenAI以外のプロバイダーのトラブルシューティング

### トレーシングクライアントエラー 401

トレーシングに関連するエラーが発生する場合、トレースがOpenAIサーバーへアップロードされる一方で、OpenAI API キーがないことが原因です。これを解決する方法は 3 つあります。

1. トレーシングを完全に無効にします: [`set_tracing_disabled(True)`][agents.set_tracing_disabled]。
2. トレーシング用のOpenAIキーを設定します: [`set_tracing_export_api_key(...)`][agents.set_tracing_export_api_key]。この API キーはトレースのアップロードにのみ使用され、[platform.openai.com](https://platform.openai.com/) で発行されたものである必要があります。
3. OpenAI以外のトレースプロセッサーを使用します。[トレーシングのドキュメント](../tracing.md#custom-tracing-processors)を参照してください。

### Responses APIのサポート

SDK はデフォルトで Responses APIを使用しますが、他の多くの LLM プロバイダーはまだこれをサポートしていません。その結果、404 や同様の問題が発生する場合があります。解決方法は 2 つあります。

1. [`set_default_openai_api("chat_completions")`][agents.set_default_openai_api] を呼び出します。これは、環境変数を通じて `OPENAI_API_KEY` と `OPENAI_BASE_URL` を設定している場合に機能します。
2. [`OpenAIChatCompletionsModel`][agents.models.openai_chatcompletions.OpenAIChatCompletionsModel] を使用します。コード例は[こちら](https://github.com/openai/openai-agents-python/tree/main/examples/model_providers/)にあります。

### Chat Completions の互換性オプション

Chat Completions を通じてルーティングする場合、SDK は Chat Completions では送信できない Responses 専用フィールドを警告なしに削除することで互換性を維持します。たとえば、`previous_response_id`、`conversation_id`、Responses APIの `prompt` フィールド、またはテキストのみではないツール出力などです。開発中にこれらの不一致を即座に失敗させる場合は、OpenAIプロバイダーで厳密な機能検証を有効にしてください。

```python
from agents import Agent, OpenAIProvider, RunConfig, Runner

provider = OpenAIProvider(
    use_responses=False,
    strict_feature_validation=True,
)

agent = Agent(name="Assistant")
result = await Runner.run(
    agent,
    "Hello",
    run_config=RunConfig(model_provider=provider),
)
```

[`MultiProvider`][agents.MultiProvider] を使用する場合は、代わりに `openai_strict_feature_validation=True` を渡してください。

一部のOpenAI互換 Chat Completions プロバイダーは、SDK による増分処理には信頼性が不十分なチャンクで、ツール呼び出しの差分をストリーミングします。その場合は、ストリーミングされるツール呼び出しのバッファリングを有効にし、プロバイダーのストリームが終了した後でのみ SDK がツール呼び出しを生成するようにしてください。

```python
from agents import OpenAIProvider

provider = OpenAIProvider(
    use_responses=False,
    buffer_streamed_tool_calls=True,
)
```

[`MultiProvider`][agents.MultiProvider] では、`openai_buffer_streamed_tool_calls=True` を使用してください。

### structured outputs のサポート

一部のモデルプロバイダーは、[structured outputs](https://platform.openai.com/docs/guides/structured-outputs)をサポートしていません。その結果、次のようなエラーが発生する場合があります。

```

BadRequestError: Error code: 400 - {'error': {'message': "'response_format.type' : value is not one of the allowed values ['text','json_object']", 'type': 'invalid_request_error'}}

```

これは一部のモデルプロバイダーの制約です。JSON 出力はサポートしていますが、出力に使用する `json_schema` を指定できません。現在この問題の修正に取り組んでいますが、JSON スキーマ出力をサポートするプロバイダーを利用することを推奨します。そうでない場合、不正な形式の JSON によりアプリケーションが頻繁に動作しなくなるためです。

## プロバイダーをまたいだモデルの組み合わせ

モデルプロバイダー間の機能差を認識しておく必要があります。そうしないと、エラーが発生する可能性があります。たとえば、OpenAIは structured outputs、マルチモーダル入力、ホステッドファイル検索、および Web 検索をサポートしていますが、他の多くのプロバイダーはこれらの機能をサポートしていません。次の制限に注意してください。

-   未対応の `tools` を、それを理解しないプロバイダーへ送信しないでください
-   テキスト専用モデルを呼び出す前に、マルチモーダル入力を除外してください
-   構造化 JSON 出力をサポートしないプロバイダーは、無効な JSON を生成する場合があることに注意してください。

## サードパーティアダプター

SDK の組み込みプロバイダー統合ポイントでは不十分な場合にのみ、サードパーティアダプターを使用してください。この SDK でOpenAIモデルのみを使用する場合は、Any-LLM や LiteLLM ではなく、組み込みの [`OpenAIResponsesModel`][agents.models.openai_responses.OpenAIResponsesModel] パスを使用してください。サードパーティアダプターは、OpenAIモデルとOpenAI以外のプロバイダーを組み合わせる必要がある場合や、アダプターでのみ提供されるプロバイダー対応またはルーティングが必要な場合に使用します。アダプターは SDK と上流のモデルプロバイダーの間に互換性レイヤーを追加するため、機能のサポートとリクエストのセマンティクスはプロバイダーによって異なる場合があります。SDK には現在、ベストエフォートのベータ版アダプター統合として Any-LLM と LiteLLM が含まれています。

### Any-LLM

Any-LLM のサポートは、Any-LLM が管理するプロバイダー対応またはルーティングが必要な場合に向けて、ベストエフォートのベータ版として含まれています。

上流のプロバイダーパスに応じて、Any-LLM は Responses API、Chat Completions 互換 API、またはプロバイダー固有の互換性レイヤーを使用する場合があります。

Any-LLM が必要な場合は、`openai-agents[any-llm]` をインストールし、[`examples/model_providers/any_llm_auto.py`](https://github.com/openai/openai-agents-python/tree/main/examples/model_providers/any_llm_auto.py) または [`examples/model_providers/any_llm_provider.py`](https://github.com/openai/openai-agents-python/tree/main/examples/model_providers/any_llm_provider.py) から始めてください。[`MultiProvider`][agents.MultiProvider] で `any-llm/...` モデル名を使用するか、`AnyLLMModel` を直接インスタンス化するか、実行スコープで `AnyLLMProvider` を使用できます。モデルサーフェスを明示的に固定する必要がある場合は、`AnyLLMModel` の構築時に `api="responses"` または `api="chat_completions"` を渡してください。

Any-LLM は引き続きサードパーティアダプターレイヤーであるため、プロバイダーの依存関係と機能上の不足は SDK ではなく、上流の Any-LLM によって定義されます。上流プロバイダーが使用量メトリクスを返す場合、それらは自動的に伝播されます。ただし、ストリーミングされる Chat Completions バックエンドでは、使用量チャンクを生成する前に `ModelSettings(include_usage=True)` が必要な場合があります。structured outputs、ツール呼び出し、使用量レポート、または Responses 固有の動作に依存する場合は、デプロイ予定の正確なプロバイダーバックエンドを検証してください。

### LiteLLM

LiteLLM のサポートは、LiteLLM 固有のプロバイダー対応またはルーティングが必要な場合に向けて、ベストエフォートのベータ版として含まれています。

LiteLLM が必要な場合は、`openai-agents[litellm]` をインストールし、[`examples/model_providers/litellm_auto.py`](https://github.com/openai/openai-agents-python/tree/main/examples/model_providers/litellm_auto.py) または [`examples/model_providers/litellm_provider.py`](https://github.com/openai/openai-agents-python/tree/main/examples/model_providers/litellm_provider.py) から始めてください。`litellm/...` モデル名を使用するか、[`LitellmModel`][agents.extensions.models.litellm_model.LitellmModel] を直接インスタンス化できます。

LiteLLM アダプターを通じてアクセスする一部のプロバイダーは、デフォルトでは SDK の使用量メトリクスを設定しません。使用量レポートが必要な場合は `ModelSettings(include_usage=True)` を渡し、structured outputs、ツール呼び出し、使用量レポート、またはアダプター固有のルーティング動作に依存する場合は、デプロイ予定の正確なプロバイダーバックエンドを検証してください。

LiteLLM がレスポンスオブジェクトについて Pydantic シリアライザーの警告を出力する場合、LiteLLM アダプターをインポートする前に、SDK の互換性パッチをオプトインで有効にできます。

```bash
export OPENAI_AGENTS_ENABLE_LITELLM_SERIALIZER_PATCH=true
```

このパッチはデフォルトでは無効で、`1` または `true` の値に対してのみ有効になります。非公開の LiteLLM ロギングヘルパーをラップすることで、LiteLLM のレスポンスシリアライズに関する特定の種類の警告を抑制します。そのため、一般的なシリアライズ設定ではなく、対象を限定した回避策として扱ってください。非公開の LiteLLM APIに依存しているため、LiteLLM をアップグレードするときは再度検証し、上流で警告が発生しなくなったら環境変数を削除してください。