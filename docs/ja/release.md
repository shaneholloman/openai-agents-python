---
search:
  exclude: true
---
# リリースプロセス／変更履歴

このプロジェクトでは、`0.Y.Z` 形式を使用する、セマンティックバージョニングを若干変更した方式に従います。先頭の `0` は、SDK がまだ急速に進化していることを示します。各構成要素は次のように更新します。

## マイナー（`Y`）バージョン

ベータと明記されていない公開インターフェースに **破壊的変更** がある場合、マイナーバージョン `Y` を増やします。たとえば、`0.0.x` から `0.1.x` への移行には、破壊的変更が含まれる可能性があります。

破壊的変更を避けたい場合は、プロジェクトで `0.0.x` バージョンに固定することを推奨します。

## パッチ（`Z`）バージョン

破壊的でない変更については、`Z` を増やします。

- バグ修正
- 新機能
- 非公開インターフェースへの変更
- ベータ機能の更新

## 破壊的変更の変更履歴

### 0.19.0

このマイナーリリースには、破壊的変更は **ありません**。マイナーバージョンの更新は、OpenAI Responses の重要な新機能領域であるプログラマティックツール呼び出しを反映したものです。

主な変更点：

- [`ProgrammaticToolCallingTool`][agents.tool.ProgrammaticToolCallingTool] を追加しました。これにより、対応する OpenAI Responses モデルは JavaScript を生成し、プログラマティックツール呼び出しの対象となるツールを連携させることができます。ツールごとの `allowed_callers`、`FunctionTool` インスタンスからの structured outputs、および Runner のストリーミング、ガードレール、承認、セッション、`RunState` との統合をサポートします。セットアップと制約については、[プログラマティックツール呼び出し](tools.md#programmatic-tool-calling)を参照してください。
- 公開 `agents.decorators` モジュールと、既存の `@function_tool` デコレーターの短いエイリアスである `@tool` を、既存のガードレールデコレーターと併せて追加しました。`FunctionTool` インスタンスは、非同期 callable オブジェクトもサポートするようになりました。
- SDK 設定では、エージェント、実行、モデル、セッション、サンドボックス、音声パイプラインの全体で、型付き設定オブジェクトまたは辞書のいずれかを一貫して受け付けるようになり、不明な設定も検証されます。
- モデル、ツール、MCP、Realtime、セッション、サンドボックス、トレーシング全体のエラーおよび診断ログを強化し、有用なデバッグコンテキストを維持しながら、raw な機密ペイロードが公開されないようにしました。
- AnyLLM、LiteLLM、Chat Completions との互換性を向上し、モデルの再試行間でセッション履歴を保持するようにしました。また、レスポンス開始前に発生する WebSocket の過負荷に関するプロバイダー再試行ガイダンスを追加し、許可されている場合には、オプトインの Runner 再試行ポリシーで失敗した試行を再実行できるようにしました。
- `VercelCloudBucketMountStrategy` を通じて、[Vercel サンドボックスの作成時にのみ設定できる S3 マウント](sandbox/clients.md#mounts-and-remote-storage)を追加しました。マウントされたセッションでは、バケットの内容がワークスペースの永続化対象から除外され、動的なマウント変更やセッションの再開は意図的にサポートされません。

### 0.18.0

このマイナーリリースには、破壊的変更は **ありません**。マイナーバージョンの更新は、Realtime エージェントのデフォルトモデル更新のみを反映したものです。

主な変更点：

- Realtime エージェントのデフォルトモデルが `gpt-realtime-2.1` になり、新しい Realtime セットアップでは追加設定なしで最新の推奨モデルが使用されるようになりました。

### 0.17.0

このバージョンでは、サンドボックスのローカルソースの実体化において、ソースパスが `Manifest.extra_path_grants` の対象でない限り、`LocalFile.src` と `LocalDir.src` が実体化の `base_dir` 内に維持されます。`base_dir` は、マニフェストが適用される時点での SDK プロセスの現在の作業ディレクトリです。相対ローカルソースはそのディレクトリを基準に解決されますが、絶対ローカルソースは、あらかじめそのディレクトリ内または明示的な許可対象内に存在する必要があります。これによりローカルアーティファクトの境界に関する問題は解消されますが、信頼できるホストのファイルやディレクトリを、そのベースディレクトリ外からサンドボックスワークスペースへ意図的にコピーするアプリケーションに影響する可能性があります。

移行するには、マニフェストレベルで `SandboxPathGrant` を使用して、信頼できるホストルートを許可してください。サンドボックスがそれらのファイルを読み取るだけでよい場合は、読み取り専用にすることを推奨します。

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

`extra_path_grants` は、信頼できるアプリケーション設定として扱ってください。アプリケーションが対象のホストパスをすでに承認している場合を除き、モデル出力やその他の信頼できないマニフェスト入力から許可設定を作成しないでください。

### 0.16.0

このバージョンでは、SDK のデフォルトモデルが `gpt-4.1` から `gpt-5.4-mini` に変更されました。これは、モデルを明示的に設定していないエージェントと実行に影響します。新しいデフォルトは GPT-5 モデルであるため、暗黙的なデフォルトモデル設定には `reasoning.effort="none"` や `verbosity="low"` などの GPT-5 のデフォルト値が含まれるようになりました。

以前のデフォルトモデルの動作を維持する必要がある場合は、エージェントまたは実行設定でモデルを明示的に設定するか、`OPENAI_DEFAULT_MODEL` 環境変数を設定してください。

```python
agent = Agent(name="Assistant", model="gpt-4.1")
```

主な変更点：

- `Runner.run`、`Runner.run_sync`、`Runner.run_streamed` で `max_turns=None` を指定し、ターン数の上限を無効にできるようになりました。
- ローカル、Docker、プロバイダーを利用する各サンドボックス実装において、サンドボックスワークスペースのハイドレーションで、絶対パスのシンボリックリンク先を含め、アーカイブルート外を指すシンボリックリンクを含む tar アーカイブが拒否されるようになりました。

### 0.15.0

このバージョンでは、モデルによる拒否が、空のテキスト出力として扱われたり、structured outputs の場合に実行ループが `MaxTurnsExceeded` まで再試行されたりするのではなく、`ModelRefusalError` として明示的に公開されるようになりました。

これは、拒否のみを含むモデルレスポンスが `final_output == ""` で完了することを想定していたコードに影響します。例外を送出せずに拒否を処理するには、`model_refusal` 実行エラーハンドラーを指定してください。

```python
result = Runner.run_sync(
    agent,
    input,
    error_handlers={"model_refusal": lambda data: data.error.refusal},
)
```

structured outputs を使用するエージェントの場合、ハンドラーはエージェントの出力スキーマに一致する値を返すことができ、SDK は他の実行エラーハンドラーの最終出力と同様にその値を検証します。

### 0.14.0

このマイナーリリースには破壊的変更は **ありません** が、主要な新しいベータ機能領域としてサンドボックスエージェントが追加され、ローカル、コンテナ化、ホスト環境で利用するために必要なランタイム、バックエンド、ドキュメントのサポートも追加されました。

主な変更点：

- `SandboxAgent`、`Manifest`、`SandboxRunConfig` を中心とする新しいベータ版サンドボックスランタイムサーフェスを追加しました。これにより、エージェントはファイル、ディレクトリ、Git リポジトリ、マウント、スナップショット、再開サポートを備えた、永続的で隔離されたワークスペース内で作業できます。
- `UnixLocalSandboxClient` と `DockerSandboxClient` により、ローカル開発およびコンテナ化された開発向けのサンドボックス実行バックエンドを追加しました。また、Python パッケージのオプション依存関係 extras を通じて、Blaxel、Cloudflare、Daytona、E2B、Modal、Runloop、Vercel のホスト型プロバイダー統合も追加しました。
- サンドボックスメモリのサポートを追加し、今後の実行で以前の実行から得た知見を再利用できるようになりました。段階的開示、複数ターンのグループ化、設定可能な隔離境界、および S3 を利用するワークフローを含む永続メモリのコード例を備えています。
- ローカルおよび synthetic ワークスペースエントリ、S3／R2／GCS／Azure Blob Storage／S3 Files 向けのリモートストレージマウント、移植可能なスナップショット、`RunState`、`SandboxSessionState`、保存済みスナップショットを使用する再開フローを含む、より包括的なワークスペースおよび再開モデルを追加しました。
- `examples/sandbox/` 配下に多数のサンドボックスコード例とチュートリアルを追加しました。スキル、ハンドオフ、メモリを使用するコーディングタスク、プロバイダー固有のセットアップ、コードレビュー、データルーム QA、Web サイトのクローン作成などのエンドツーエンドワークフローを扱っています。
- サンドボックス対応のセッション準備、機能のバインド、状態のシリアル化、統合トレーシング、プロンプトキャッシュキーのデフォルト値、機密性の高い MCP 出力をより安全に秘匿する処理により、コアランタイムとトレーシングスタックを拡張しました。

### 0.13.0

このマイナーリリースには破壊的変更は **ありません** が、注目すべき Realtime のデフォルト更新に加え、新しい MCP 機能とランタイムの安定性向上が含まれています。

主な変更点：

- デフォルトの WebSocket Realtime モデルが `gpt-realtime-1.5` になり、新しい Realtime エージェントのセットアップでは追加設定なしで新しいモデルが使用されるようになりました。
- `MCPServer` で `list_resources()`、`list_resource_templates()`、`read_resource()` が公開され、`MCPServerStreamableHttp` で `session_id` が公開されるようになりました。これにより、MCP Streamable HTTP トランスポートを使用するセッションを、再接続やステートレスワーカーをまたいで再開できます。
- Chat Completions 統合では、`should_replay_reasoning_content` を通じて既存の推論内容の再送信をオプトインできるようになり、LiteLLM／DeepSeek などのアダプターで、プロバイダー固有の推論およびツール呼び出しの継続性が向上しました。
- `SQLAlchemySession` での同時初回書き込み、推論除去後に孤立した assistant メッセージ ID を含む圧縮リクエスト、MCP／推論項目を残していた `remove_all_tools()`、`FunctionTool` インスタンスのバッチ実行機構における競合など、複数のランタイムおよびセッションのエッジケースを修正しました。

### 0.12.0

このマイナーリリースには、破壊的変更は **ありません**。主な機能追加については、[リリースノート](https://github.com/openai/openai-agents-python/releases/tag/v0.12.0)を確認してください。

### 0.11.0

このマイナーリリースには、破壊的変更は **ありません**。主な機能追加については、[リリースノート](https://github.com/openai/openai-agents-python/releases/tag/v0.11.0)を確認してください。

### 0.10.0

このマイナーリリースには破壊的変更は **ありません** が、OpenAI Responses ユーザー向けの重要な新機能領域として、Responses API の WebSocket トランスポートサポートが含まれています。

主な変更点：

- OpenAI Responses モデル向けの WebSocket トランスポートサポートを追加しました（オプトイン方式であり、HTTP が引き続きデフォルトのトランスポートです）。
- 複数ターンの実行にわたって、WebSocket 対応の共有プロバイダーと `RunConfig` を再利用するための `responses_websocket_session()` ヘルパー／`ResponsesWebSocketSession` を追加しました。
- ストリーミング、ツール、承認、後続ターンを扱う新しい WebSocket ストリーミングコード例（`examples/basic/stream_ws.py`）を追加しました。

### 0.9.0

このバージョンでは、Python 3.9 がサポート対象外になりました。このメジャーバージョンが 3 か月前に EOL に達したためです。より新しいランタイムバージョンにアップグレードしてください。

さらに、`Agent#as_tool()` メソッドから返される値の型ヒントが、`Tool` から `FunctionTool` に絞り込まれました。通常、この変更によって破壊的な問題が発生することはありませんが、コードがより広い union 型に依存している場合は、コード側で調整が必要になることがあります。

### 0.8.0

このバージョンでは、2 つのランタイム動作の変更により、移行作業が必要になる可能性があります。

- **同期** Python callable をラップする `FunctionTool` インスタンスは、イベントループスレッド上で実行されるのではなく、`asyncio.to_thread(...)` を介してワーカースレッド上で実行されるようになりました。ツールロジックがスレッドローカル状態またはスレッドアフィニティのあるリソースに依存する場合は、非同期ツール実装に移行するか、ツールコードでスレッドアフィニティを明示してください。
- ローカル MCP ツールの失敗処理が設定可能になり、デフォルトの動作では、実行全体を失敗させる代わりに、モデルから参照可能なエラー出力を返せるようになりました。即時失敗のセマンティクスに依存している場合は、`mcp_config={"failure_error_function": None}` を設定してください。サーバーレベルの `failure_error_function` 値はエージェントレベルの設定を上書きするため、明示的なハンドラーを持つ各ローカル MCP サーバーで `failure_error_function=None` を設定してください。

### 0.7.0

このバージョンでは、既存のアプリケーションに影響する可能性がある動作変更がいくつかあります。

- ネストされたハンドオフ履歴が **オプトイン** になりました（デフォルトでは無効です）。v0.6.x のデフォルトだったネスト動作に依存していた場合は、`RunConfig(nest_handoff_history=True)` を明示的に設定してください。
- `gpt-5.1`／`gpt-5.2` のデフォルトの `reasoning.effort` が、SDK のデフォルト値として設定されていた従来の `"low"` から `"none"` に変更されました。プロンプトまたは品質／コスト特性が `"low"` に依存していた場合は、`model_settings` で明示的に設定してください。

### 0.6.0

このバージョンでは、デフォルトのハンドオフ履歴は、ユーザーと assistant のターンを個別のメッセージとして渡すのではなく、単一の assistant メッセージにまとめられるようになり、後続のエージェントに簡潔で予測可能な要約を提供します
- 既存の単一メッセージ形式のハンドオフトランスクリプトでは、デフォルトで `<CONVERSATION HISTORY>` ブロックの前に、正確なリテラルテキスト `For context, here is the conversation so far between the user and the previous agent:` が置かれるようになり、後続のエージェントは明確なラベル付きの要約を受け取れます

### 0.5.0

このバージョンでは、外部から確認できる破壊的変更は導入されていませんが、新機能と内部実装に関する重要な更新がいくつか含まれています。

- `RealtimeRunner` に、[SIP プロトコル接続](https://platform.openai.com/docs/guides/realtime-sip)を処理するためのサポートを追加しました。
- Python 3.14 との互換性のため、`Runner#run_sync` の内部ロジックを大幅に改訂しました

### 0.4.0

このバージョンでは、[openai](https://pypi.org/project/openai/) パッケージの v1.x バージョンはサポート対象外になりました。この SDK では openai v2.x を使用してください。

### 0.3.0

このバージョンでは、Realtime API のサポートが gpt-realtime モデルとその API インターフェース（GA 版）に移行します。

### 0.2.0

このバージョンでは、以前 `Agent` を引数として受け取っていた箇所の一部が、代わりに `AgentBase` を引数として受け取るようになりました。たとえば、これは MCP サーバーの `list_tools()` メソッドシグネチャに適用されます。これは純粋に型付け上の変更であり、引き続き `Agent` オブジェクトを受け取ります。更新するには、`Agent` を `AgentBase` に置き換えて型エラーを修正してください。

### 0.1.0

このバージョンでは、[`MCPServer.list_tools()`][agents.mcp.server.MCPServer] に `run_context` と `agent` という 2 つの新しいパラメーターが追加されました。`MCPServer` のサブクラスでオーバーライドされているすべての `MCPServer.list_tools()` メソッドに、これらのパラメーターを追加する必要があります。