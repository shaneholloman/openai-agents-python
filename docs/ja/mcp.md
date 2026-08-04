---
search:
  exclude: true
---
# Model context protocol (MCP)

[Model context protocol](https://modelcontextprotocol.io/introduction)（MCP）は、アプリケーションがツールとコンテキストを言語モデルに公開する方法を標準化します。公式ドキュメントからの引用です。

> MCP は、アプリケーションが LLM にコンテキストを提供する方法を標準化するオープンプロトコルです。MCP は、AI
> アプリケーション向けの USB-C ポートのようなものと考えてください。USB-C がデバイスをさまざまな周辺機器やアクセサリーに接続するための標準化された方法を提供するのと同様に、MCP
> は AI モデルをさまざまなデータソースやツールに接続するための標準化された方法を提供します。

Agents Python SDK は複数の MCP トランスポートに対応しています。これにより、既存の MCP サーバーを再利用したり、独自のサーバーを構築して、ファイルシステム、HTTP、またはコネクターを基盤とするツールをエージェントに公開したりできます。

!!! warning "接続前の MCP サーバーの信頼性確認"

    MCP ツールは、モデルコンテキストのデータを公開し、提供された認証情報を使用してアクションを実行できます。信頼できるサーバーにのみ接続し、最小権限の認証情報を使用し、アクセストークンは URL ではなく認証フィールドまたはヘッダーに保持し、機密性の高い操作には承認を必須としてください。[OpenAI の MCP セキュリティガイダンス](https://developers.openai.com/api/docs/guides/tools-connectors-mcp#risks-and-safety)を参照してください。

## MCP 統合の選択

MCP サーバーをエージェントに接続する前に、ツール呼び出しをどこで実行するか、またどのトランスポートにアクセスできるかを決定してください。以下の表は、Python SDK がサポートする選択肢をまとめたものです。

| 必要なこと                                                                           | 推奨オプション                                          |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| OpenAI の Responses API がモデルに代わって、公開アクセス可能な MCP サーバーを呼び出す| [`HostedMCPTool`][agents.tool.HostedMCPTool] を使用する **ホスト型 MCP サーバーツール** |
| ローカルまたはリモートで実行する Streamable HTTP サーバーに接続する                  | [`MCPServerStreamableHttp`][agents.mcp.server.MCPServerStreamableHttp] を使用する **Streamable HTTP MCP サーバー** |
| Server-Sent Events 対応 HTTP を実装するサーバーと通信する                            | [`MCPServerSse`][agents.mcp.server.MCPServerSse] を使用する **SSE 対応 HTTP MCP サーバー** |
| ローカルプロセスを起動し、stdin/stdout 経由で通信する                                | [`MCPServerStdio`][agents.mcp.server.MCPServerStdio] を使用する **stdio MCP サーバー** |

以下のセクションでは、各オプション、その設定方法、および各トランスポートを選ぶべき状況について説明します。

## エージェントレベルの MCP 設定

トランスポートの選択に加えて、`Agent.mcp_config` を設定することで、MCP ツールの準備方法を調整できます。

```python
from agents import Agent

agent = Agent(
    name="Assistant",
    mcp_servers=[server],
    mcp_config={
        # Try to convert MCP tool schemas to strict JSON schema.
        "convert_schemas_to_strict": True,
        # If None, MCP tool failures are raised as exceptions instead of
        # returning model-visible error text.
        "failure_error_function": None,
        # Prefix local MCP tool names with their server name.
        "include_server_in_tool_names": True,
    },
)
```

注意事項：

- `convert_schemas_to_strict` はベストエフォート方式です。スキーマを変換できない場合は、元のスキーマが使用されます。
- `failure_error_function` は、MCP ツール呼び出しの失敗をモデルにどのように提示するかを制御します。
- `failure_error_function` が未設定の場合、SDK はデフォルトのツールエラーフォーマッターを使用します。
- サーバーレベルの `failure_error_function` は、そのサーバーに対する `Agent.mcp_config["failure_error_function"]` を上書きします。
- `include_server_in_tool_names` はオプトインです。有効にすると、各ローカル MCP ツールは、決定論的なサーバー接頭辞付きの名前でモデルに公開されます。これにより、複数の MCP サーバーが同名のツールを公開する場合の名前の衝突を回避できます。生成される名前は ASCII セーフで、関数ツール名の長さ制限内に収まり、同じエージェント上にある既存のローカル関数ツール名や有効なハンドオフ名との衝突も回避します。SDK は引き続き、元のサーバー上で元の MCP ツール名を使用して呼び出します。

## トランスポート間で共通するパターン

トランスポートを選択した後、ほとんどの統合では、次の事項も決定する必要があります。

- ツールの一部のみを公開する方法（[ツールのフィルタリング](#tool-filtering)）。
- サーバーが再利用可能なプロンプトも提供するかどうか（[プロンプト](#prompts)）。
- `list_tools()` をキャッシュするかどうか（[キャッシュ](#caching)）。
- MCP のアクティビティをトレースにどのように表示するか（[トレーシング](#tracing)）。

ローカル MCP サーバー（`MCPServerStdio`、`MCPServerSse`、`MCPServerStreamableHttp`）では、承認ポリシーと呼び出しごとの `_meta` ペイロードも共通の概念です。Streamable HTTP のセクションに最も完全なコード例を示していますが、同じパターンを他のローカルトランスポートにも適用できます。

## 1. ホスト型 MCP サーバーツール

ホスト型ツールでは、ツール呼び出しの往復処理全体が OpenAI のインフラストラクチャ内で実行されます。コード側でツールを一覧取得して呼び出す代わりに、[`HostedMCPTool`][agents.tool.HostedMCPTool] がサーバーラベル（および任意のコネクターメタデータ）を Responses API に転送します。モデルは、Python プロセスへの追加のコールバックなしで、リモートサーバーのツールを一覧取得して呼び出します。現在、ホスト型ツールは、Responses API のホスト型 MCP 統合をサポートする OpenAI モデルで動作します。

### 基本的なホスト型 MCP ツール

エージェントの `tools` リストに [`HostedMCPTool`][agents.tool.HostedMCPTool] を追加して、ホスト型ツールを作成します。`tool_config`
辞書は、REST API に送信する JSON と同じ構造です。

```python
import asyncio

from agents import Agent, HostedMCPTool, Runner

async def main() -> None:
    agent = Agent(
        name="Assistant",
        instructions="Use the DeepWiki hosted MCP server to inspect openai/openai-agents-python.",
        tools=[
            HostedMCPTool(
                tool_config={
                    "type": "mcp",
                    "server_label": "deepwiki",
                    "server_url": "https://mcp.deepwiki.com/mcp",
                    "require_approval": "never",
                }
            )
        ],
    )

    result = await Runner.run(
        agent,
        "Which language is the repository openai/openai-agents-python written in?",
    )
    print(result.final_output)

asyncio.run(main())
```

ホスト型サーバーはツールを自動的に公開するため、`mcp_servers` に追加する必要はありません。

ホスト型ツール検索でホスト型 MCP サーバーを遅延読み込みする場合は、`tool_config["defer_loading"] = True` を設定し、[`ToolSearchTool`][agents.tool.ToolSearchTool] をエージェントに追加します。これは OpenAI Responses モデルでのみサポートされます。ツール検索の完全な設定と制約については、[ツール](tools.md#hosted-tool-search)を参照してください。

### ホスト型 MCP 実行結果のストリーミング

ホスト型ツールは、関数ツールとまったく同じ方法で実行結果のストリーミングをサポートします。モデルがまだ処理中でも、`Runner.run_streamed` を使用して
増分 MCP 出力を受け取れます。

```python
result = Runner.run_streamed(agent, "Summarise this repository's top languages")
async for event in result.stream_events():
    if event.type == "run_item_stream_event":
        print(f"Received: {event.item}")
print(result.final_output)
```

### 任意の承認フロー

サーバーが機密性の高い操作を実行できる場合、ツールを実行するたびに、人またはプログラムによる承認を必須にできます。`tool_config` の `require_approval` に、単一のポリシー（`"always"`、`"never"`）またはツール名をポリシーに対応付ける辞書を設定します。Python 内で判断するには、`on_approval_request` コールバックを指定します。

```python
from agents import MCPToolApprovalFunctionResult, MCPToolApprovalRequest

SAFE_TOOLS = {"read_wiki_structure", "read_wiki_contents", "ask_question"}

def approve_tool(request: MCPToolApprovalRequest) -> MCPToolApprovalFunctionResult:
    if request.data.name in SAFE_TOOLS:
        return {"approve": True}
    return {"approve": False, "reason": "Escalate to a human reviewer"}

agent = Agent(
    name="Assistant",
    tools=[
        HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": "always",
            },
            on_approval_request=approve_tool,
        )
    ],
)
```

コールバックは同期または非同期にでき、モデルが実行を継続するために承認データを必要とするたびに呼び出されます。

### コネクターを基盤とするホスト型サーバー

ホスト型 MCP は OpenAI コネクターもサポートします。`server_url` を指定する代わりに、`connector_id` とアクセストークンを指定します。Responses API が認証を処理し、ホスト型サーバーがコネクターのツールを公開します。

```python
import os

HostedMCPTool(
    tool_config={
        "type": "mcp",
        "server_label": "google_calendar",
        "connector_id": "connector_googlecalendar",
        "authorization": os.environ["GOOGLE_CALENDAR_AUTHORIZATION"],
        "require_approval": "never",
    }
)
```

ストリーミング、承認、コネクターを含む、完全に動作するホスト型ツールのサンプルは、[`examples/hosted_mcp`](https://github.com/openai/openai-agents-python/tree/main/examples/hosted_mcp) にあります。

## 2. Streamable HTTP MCP サーバー

ネットワーク接続を自分で管理する場合は、[`MCPServerStreamableHttp`][agents.mcp.server.MCPServerStreamableHttp] を使用します。Streamable HTTP サーバーは、トランスポートを制御する場合や、低レイテンシーを維持しながら独自のインフラストラクチャ内でサーバーを実行する場合に適しています。

```python
import asyncio
import os

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings

async def main() -> None:
    token = os.environ["MCP_SERVER_TOKEN"]
    async with MCPServerStreamableHttp(
        name="Streamable HTTP Python Server",
        params={
            "url": "http://localhost:8000/mcp",
            "headers": {"Authorization": f"Bearer {token}"},
            "timeout": 10,
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    ) as server:
        agent = Agent(
            name="Assistant",
            instructions="Use the MCP tools to answer the questions.",
            mcp_servers=[server],
            model_settings=ModelSettings(tool_choice="required"),
        )

        result = await Runner.run(agent, "Add 7 and 22.")
        print(result.final_output)

asyncio.run(main())
```

コンストラクターでは、追加のオプションを指定できます。

- `client_session_timeout_seconds` は、MCP ClientSession の読み取りタイムアウトを制御します。`datetime.timedelta` で表現可能かつ 1 マイクロ秒以上の正の有限値を指定すると有限のタイムアウトが設定され、`None` と `0` を指定すると無効になります。それ以外の値は、サーバーの構築時に拒否されます。
- `use_structured_content` は、テキスト出力より `tool_result.structured_content` を優先するかどうかを切り替えます。
- `max_retry_attempts` と `retry_backoff_seconds_base` は、`list_tools()` と `call_tool()` に自動再試行を追加します。
- `tool_filter` を使用すると、ツールの一部のみを公開できます（[ツールのフィルタリング](#tool-filtering)を参照）。
- `require_approval` は、ローカル MCP ツールで人間参加型の承認ポリシーを有効にします。
- `failure_error_function` は、モデルに表示される MCP ツール失敗メッセージをカスタマイズします。代わりにエラーを送出するには、`None` に設定します。
- `tool_meta_resolver` は、`call_tool()` の前に、呼び出しごとの MCP `_meta` ペイロードを挿入します。

### ローカル MCP サーバーの承認ポリシー

`MCPServerStdio`、`MCPServerSse`、`MCPServerStreamableHttp` は、いずれも `require_approval` を受け付けます。

サポートされる形式：

- すべてのツールに対する `"always"` または `"never"`。
- `True` / `False`（常に承認する／承認しないのと同等）。
- ツールごとのマップ。例：`{"delete_file": "always", "read_file": "never"}`。
- グループ化されたオブジェクト：`{"always": {"tool_names": [...]}, "never": {"tool_names": [...]}}`。

```python
async with MCPServerStreamableHttp(
    name="Filesystem MCP",
    params={"url": "http://localhost:8000/mcp"},
    require_approval={"always": {"tool_names": ["delete_file"]}},
) as server:
    ...
```

一時停止／再開を含む完全なフローについては、[人間参加型](human_in_the_loop.md)および `examples/mcp/get_all_mcp_tools_example/main.py` を参照してください。

### `tool_meta_resolver` による呼び出しごとのメタデータ

MCP サーバーが `_meta` 内にリクエストメタデータ（テナント ID やトレースコンテキストなど）を必要とする場合は、`tool_meta_resolver` を使用します。以下のコード例では、`Runner.run(...)` に `context` として `dict` を渡すことを前提としています。

```python
from agents.mcp import MCPServerStreamableHttp, MCPToolMetaContext


def resolve_meta(context: MCPToolMetaContext) -> dict[str, str] | None:
    run_context_data = context.run_context.context or {}
    tenant_id = run_context_data.get("tenant_id")
    if tenant_id is None:
        return None
    return {"tenant_id": str(tenant_id), "source": "agents-sdk"}


server = MCPServerStreamableHttp(
    name="Metadata-aware MCP",
    params={"url": "http://localhost:8000/mcp"},
    tool_meta_resolver=resolve_meta,
)
```

実行コンテキストが Pydantic モデル、dataclass、またはカスタムクラスの場合は、属性アクセスを使用してテナント ID を読み取ります。

### MCP ツールの出力：テキストと画像

MCP ツールが画像コンテンツを返すと、SDK はそれを画像ツールの出力エントリーに自動的にマッピングします。テキストと画像が混在するレスポンスは出力項目のリストとして転送されるため、エージェントは通常の関数ツールからの画像出力と同じ方法で、MCP の画像実行結果を利用できます。

## 3. SSE 対応 HTTP MCP サーバー

!!! warning

    MCP プロジェクトでは、Server-Sent Events トランスポートは非推奨になっています。新しい統合には Streamable HTTP または stdio を使用し、SSE はレガシーサーバーにのみ使用してください。

MCP サーバーが SSE 対応 HTTP トランスポートを実装している場合は、[`MCPServerSse`][agents.mcp.server.MCPServerSse] をインスタンス化します。トランスポートを除き、API は Streamable HTTP サーバーと同一です。

```python

from agents import Agent, Runner
from agents.model_settings import ModelSettings
from agents.mcp import MCPServerSse

workspace_id = "demo-workspace"

async with MCPServerSse(
    name="SSE Python Server",
    params={
        "url": "http://localhost:8000/sse",
        "headers": {"X-Workspace": workspace_id},
    },
    cache_tools_list=True,
) as server:
    agent = Agent(
        name="Assistant",
        mcp_servers=[server],
        model_settings=ModelSettings(tool_choice="required"),
    )
    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)
```

## 4. stdio MCP サーバー

ローカルのサブプロセスとして実行される MCP サーバーには、[`MCPServerStdio`][agents.mcp.server.MCPServerStdio] を使用します。SDK はプロセスを起動し、パイプを開いた状態に保ち、コンテキストマネージャーの終了時に自動的に閉じます。このオプションは、簡単な概念実証を行う場合や、サーバーがコマンドラインのエントリーポイントのみを公開している場合に役立ちます。

```python
from pathlib import Path
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

current_dir = Path(__file__).parent
samples_dir = current_dir / "sample_files"

async with MCPServerStdio(
    name="Filesystem Server via npx",
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(samples_dir)],
    },
) as server:
    agent = Agent(
        name="Assistant",
        instructions="Use the files in the sample directory to answer questions.",
        mcp_servers=[server],
    )
    result = await Runner.run(agent, "List the files available to you.")
    print(result.final_output)
```

## 5. MCP サーバーマネージャー

複数の MCP サーバーがある場合は、`MCPServerManager` を使用して事前に接続し、接続済みのサーバーのみをエージェントに公開します。コンストラクターのオプションと再接続の動作については、[MCPServerManager API リファレンス](ref/mcp/manager.md)を参照してください。

```python
from agents import Agent, Runner
from agents.mcp import MCPServerManager, MCPServerStreamableHttp

servers = [
    MCPServerStreamableHttp(name="calendar", params={"url": "http://localhost:8000/mcp"}),
    MCPServerStreamableHttp(name="docs", params={"url": "http://localhost:8001/mcp"}),
]

async with MCPServerManager(servers) as manager:
    agent = Agent(
        name="Assistant",
        instructions="Use MCP tools when they help.",
        mcp_servers=manager.active_servers,
    )
    result = await Runner.run(agent, "Which MCP tools are available?")
    print(result.final_output)
```

主な動作：

- `drop_failed_servers=True`（デフォルト）の場合、`active_servers` には正常に接続されたサーバーのみが含まれます。
- 失敗は `failed_servers` と `errors` に記録されます。
- 最初の接続失敗時に例外を送出するには、`strict=True` を設定します。
- 失敗したサーバーを再試行するには `reconnect(failed_only=True)` を、すべてのサーバーを再起動するには `reconnect(failed_only=False)` を呼び出します。
- ライフサイクルの動作を調整するには、`connect_timeout_seconds`、`cleanup_timeout_seconds`、`connect_in_parallel` を設定します。ライフサイクルのタイムアウトには、正の有限秒数、または無効化するための `None` を指定できます。これらは構築時と代入時の両方で検証されます。ゼロを指定すると即時の期限が設定されてしまうため、拒否されます。

## 共通のサーバー機能

以下のセクションは、MCP サーバーの各トランスポートに共通して適用されます（利用できる具体的な API はサーバークラスによって異なります）。

## ツールのフィルタリング

各 MCP サーバーはツールフィルターをサポートしているため、エージェントに必要な関数のみを公開できます。フィルタリングは構築時に行うことも、実行ごとに動的に行うこともできます。

### 静的なツールのフィルタリング

単純な許可／ブロックリストを設定するには、[`create_static_tool_filter`][agents.mcp.create_static_tool_filter] を使用します。

```python
from pathlib import Path

from agents.mcp import MCPServerStdio, create_static_tool_filter

samples_dir = Path("/path/to/files")

filesystem_server = MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(samples_dir)],
    },
    tool_filter=create_static_tool_filter(allowed_tool_names=["read_file", "write_file"]),
)
```

`allowed_tool_names` と `blocked_tool_names` の両方を指定した場合、SDK は最初に許可リストを適用し、その後、残りのセットからブロックされたツールを削除します。

### 動的なツールのフィルタリング

より複雑なロジックには、[`ToolFilterContext`][agents.mcp.ToolFilterContext] を受け取る呼び出し可能オブジェクトを渡します。この呼び出し可能オブジェクトは同期または非同期にでき、ツールを公開する場合に `True` を返します。

```python
from pathlib import Path

from agents.mcp import MCPServerStdio, ToolFilterContext

samples_dir = Path("/path/to/files")

async def context_aware_filter(context: ToolFilterContext, tool) -> bool:
    if context.agent.name == "Code Reviewer" and tool.name.startswith("danger_"):
        return False
    return True

async with MCPServerStdio(
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(samples_dir)],
    },
    tool_filter=context_aware_filter,
) as server:
    ...
```

フィルターコンテキストからは、アクティブな `run_context`、ツールを要求している `agent`、および `server_name` にアクセスできます。

## プロンプト

MCP サーバーは、エージェントへの指示を動的に生成するプロンプトも提供できます。プロンプトをサポートするサーバーは、次の 2 つの
メソッドを公開します。

- `list_prompts()` は、利用可能なプロンプトテンプレートを列挙します。
- `get_prompt(name, arguments)` は、必要に応じてパラメーターを指定し、具体的なプロンプトを取得します。

```python
from agents import Agent

prompt_result = await server.get_prompt(
    "generate_code_review_instructions",
    {"focus": "security vulnerabilities", "language": "python"},
)
instructions = prompt_result.messages[0].content.text

agent = Agent(
    name="Code Reviewer",
    instructions=instructions,
    mcp_servers=[server],
)
```

## ページネーション

組み込みのローカル MCP サーバークラスは、ツールとプロンプトの一覧取得時に `nextCursor` を自動的にたどります。`list_tools()` は、フィルターの適用またはキャッシュへの格納前にツールの完全なリストを返し、`list_prompts()` は `nextCursor=None` の 1 つに統合された実行結果を返します。後続ページの取得に失敗した場合、またはサーバーが同じカーソルを繰り返した場合、部分的な実行結果を公開またはキャッシュする代わりに、操作はエラーを送出します。

リソースは引き続き明示的にページ分割されます。次のページを取得するには、`list_resources()` または `list_resource_templates()` から返された `nextCursor` を `cursor` 引数として渡します。

## キャッシュ

エージェントを実行するたびに、各 MCP サーバーで `list_tools()` が呼び出されます。リモートサーバーでは無視できないレイテンシーが生じる可能性があるため、すべての MCP サーバークラスが `cache_tools_list` オプションを公開しています。ツール定義が頻繁に変更されないと確信できる場合にのみ、`True` に設定してください。後で最新のリストを強制的に取得するには、サーバーインスタンスの `invalidate_tools_cache()` を呼び出します。

## トレーシング

[トレーシング](./tracing.md)では、次の項目を含む MCP のアクティビティが自動的に記録されます。

1. ツール一覧を取得するための MCP サーバー呼び出し。
2. ツール呼び出しに関する MCP 関連情報。

![MCP トレーシングのスクリーンショット](../assets/images/mcp-tracing.jpg)

## 関連資料

- [Model Context Protocol](https://modelcontextprotocol.io/) – 仕様および設計ガイド。
- [examples/mcp](https://github.com/openai/openai-agents-python/tree/main/examples/mcp) – 実行可能な stdio、SSE、Streamable HTTP のサンプル。
- [examples/hosted_mcp](https://github.com/openai/openai-agents-python/tree/main/examples/hosted_mcp) – 承認とコネクターを含む、完全なホスト型 MCP のデモ。