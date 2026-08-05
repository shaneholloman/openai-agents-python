---
search:
  exclude: true
---
# セッション

Agents SDK には、複数回のエージェント実行にわたって会話履歴を自動的に維持する組み込みのセッションメモリが用意されているため、ターン間で `.to_input_list()` を手動で処理する必要がありません。

セッションは特定のセッションの会話履歴を保存し、明示的な手動のメモリ管理を必要とせずに、エージェントがコンテキストを維持できるようにします。これは、エージェントに以前のやり取りを記憶させたいチャットアプリケーションや、複数ターンの会話を構築する場合に特に便利です。

SDK にクライアント側のメモリを管理させたい場合は、セッションを使用します。同じ実行内でセッションを `conversation_id`、`previous_response_id`、または `auto_previous_response_id` と組み合わせることはできません。代わりに OpenAI のサーバーで管理される継続機能を使用したい場合は、セッションと重ねて使用せず、これらの仕組みのいずれかを選択してください。

## クイックスタート

```python
from agents import Agent, Runner, SQLiteSession

# Create agent
agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

# Create a session instance with a session ID
session = SQLiteSession("conversation_123")

# First turn
result = await Runner.run(
    agent,
    "What city is the Golden Gate Bridge in?",
    session=session
)
print(result.final_output)  # "San Francisco"

# Second turn - agent automatically remembers previous context
result = await Runner.run(
    agent,
    "What state is it in?",
    session=session
)
print(result.final_output)  # "California"

# Also works with synchronous runner
result = Runner.run_sync(
    agent,
    "What's the population?",
    session=session
)
print(result.final_output)  # "Approximately 39 million"
```

## 同じセッションによる中断された実行の再開

実行が承認待ちで一時停止した場合は、同じセッションインスタンス（または同じバックエンドストアを参照する別のセッションインスタンス）を使用して再開し、再開後のターンが保存済みの同じ会話履歴を引き継ぐようにしてください。

```python
result = await Runner.run(agent, "Delete temporary files that are no longer needed.", session=session)

if result.interruptions:
    state = result.to_state()
    for interruption in result.interruptions:
        state.approve(interruption)
    result = await Runner.run(agent, state, session=session)
```

## セッションの基本動作

セッションメモリが有効な場合、次のように動作します。

1. **各実行前**: ランナーはセッションの会話履歴を自動的に取得し、入力項目の先頭に追加します。
2. **各実行後**: 実行中に生成されたすべての新しい項目（ユーザー入力、アシスタントの応答、ツール呼び出しなど）がセッションに自動的に保存されます。
3. **コンテキストの保持**: 同じセッションを使用する後続の各実行には完全な会話履歴が含まれるため、エージェントはコンテキストを維持できます。

これにより、`.to_input_list()` を手動で呼び出し、実行間の会話状態を管理する必要がなくなります。

## 履歴と新しい入力のマージ制御

セッションを渡すと、通常、ランナーは次の順序でモデル入力を準備します。

1. セッション履歴（`session.get_items(...)` から取得）
2. 新しいターンの入力

モデル呼び出し前のマージ処理をカスタマイズするには、[`RunConfig.session_input_callback`][agents.run.RunConfig.session_input_callback] を使用します。コールバックは次の 2 つのリストを受け取ります。

-   `history`: 取得したセッション履歴（入力項目形式に正規化済み）
-   `new_input`: 現在のターンの新しい入力項目

モデルに送信する入力項目の最終的なリストを返してください。

コールバックは両方のリストのコピーを受け取るため、安全に変更できます。返されたリストによってそのターンのモデル入力が決まりますが、SDK が永続化するのは新しいターンに属する項目だけです。したがって、古い履歴を並べ替えたりフィルタリングしたりしても、古いセッション項目が新しい入力として再び保存されることはありません。

```python
from agents import Agent, RunConfig, Runner, SQLiteSession


def keep_recent_history(history, new_input):
    # Keep only the last 10 history items, then append the new turn.
    return history[-10:] + new_input


agent = Agent(name="Assistant")
session = SQLiteSession("conversation_123")

result = await Runner.run(
    agent,
    "Continue from the latest updates only.",
    session=session,
    run_config=RunConfig(session_input_callback=keep_recent_history),
)
```

セッションによる項目の保存方法を変更せずに、履歴の独自の枝刈り、並べ替え、または選択的な追加が必要な場合に使用します。モデル呼び出しの直前に後段の最終処理が必要な場合は、[エージェント実行ガイド](../running_agents.md)の [`call_model_input_filter`][agents.run.RunConfig.call_model_input_filter] を使用してください。

## 取得する履歴の制限

各実行前に取得する履歴の量を制御するには、[`SessionSettings`][agents.memory.SessionSettings] を使用します。

-   `SessionSettings(limit=None)`（デフォルト）: 利用可能なすべてのセッション項目を取得します
-   `SessionSettings(limit=N)`: 最新の `N` 項目のみを取得します

これは、[`RunConfig.session_settings`][agents.run.RunConfig.session_settings] を使用して実行ごとに適用できます。

```python
from agents import Agent, RunConfig, Runner, SessionSettings, SQLiteSession

agent = Agent(name="Assistant")
session = SQLiteSession("conversation_123")

result = await Runner.run(
    agent,
    "Summarize our recent discussion.",
    session=session,
    run_config=RunConfig(session_settings=SessionSettings(limit=50)),
)
```

セッション実装がデフォルトのセッション設定を公開している場合、`RunConfig.session_settings` はその実行について、`None` ではない値を上書きします。これは、セッションのデフォルト動作を変更せずに取得サイズを制限したい長い会話で便利です。

## メモリ操作

### 基本操作

セッションでは、会話履歴を管理するために複数の操作を使用できます。

```python
from agents import SQLiteSession

session = SQLiteSession("user_123", "conversations.db")

# Get all items in a session
items = await session.get_items()

# Add new items to a session
new_items = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
await session.add_items(new_items)

# Remove and return the most recent item
last_item = await session.pop_item()
print(last_item)  # {"role": "assistant", "content": "Hi there!"}

# Clear all items from a session
await session.clear_session()
```

### `pop_item` を使用した修正

`pop_item` メソッドは、会話の最後の項目を取り消したり変更したりする場合に特に便利です。

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant")
session = SQLiteSession("correction_example")

# Initial conversation
result = await Runner.run(
    agent,
    "What's 2 + 2?",
    session=session
)
print(f"Agent: {result.final_output}")

# User wants to correct their question
assistant_item = await session.pop_item()  # Remove agent's response
user_item = await session.pop_item()  # Remove user's question

# Ask a corrected question
result = await Runner.run(
    agent,
    "What's 2 + 3?",
    session=session
)
print(f"Agent: {result.final_output}")
```

## 組み込みセッション実装

SDK には、さまざまなユースケースに対応する複数のセッション実装が用意されています。

### 組み込みセッション実装の選択

以下の詳細な例を読む前に、この表を使用して開始点を選択してください。

| セッションタイプ | 最適な用途 | 備考 |
| --- | --- | --- |
| `SQLiteSession` | ローカル開発とシンプルなアプリ | 組み込みで軽量。ファイルベースまたはインメモリ |
| `AsyncSQLiteSession` | `aiosqlite` を使用する非同期 SQLite | 非同期ドライバーをサポートする拡張バックエンド |
| `RedisSession` | ワーカーやサービス間の共有メモリ | 低レイテンシーの分散デプロイに最適 |
| `SQLAlchemySession` | 既存のデータベースを使用する本番アプリ | SQLAlchemy がサポートするデータベースに対応 |
| `MongoDBSession` | MongoDB をすでに使用しているアプリ、またはマルチプロセスストレージが必要なアプリ | 非同期 pymongo。順序付け用のアトミックなシーケンスカウンター |
| `DaprSession` | Dapr サイドカーを使用するクラウドネイティブなデプロイ | 複数の状態ストアに加え、TTL と整合性の制御をサポート |
| `OpenAIConversationsSession` | OpenAI でのサーバー管理ストレージ | OpenAI Conversations API をバックエンドとする履歴 |
| `OpenAIResponsesCompactionSession` | 自動圧縮を伴う長い会話 | 別のセッションバックエンドをラップ |
| `AdvancedSQLiteSession` | SQLite に加えて分岐や分析が必要な場合 | より高度な機能セット。専用ページを参照 |
| `EncryptedSession` | 別のセッションに追加する暗号化と TTL | ラッパー。最初に基盤となるバックエンドを選択 |

一部の実装には、追加の詳細を記載した専用ページがあります。それぞれのサブセクション内にリンクがあります。

ChatKit 用の Python サーバーを実装する場合は、ChatKit のスレッドと項目を永続化するために `chatkit.store.Store` 実装を使用してください。`SQLAlchemySession` などの Agents SDK セッションは SDK 側の会話履歴を管理しますが、ChatKit のストアをそのまま置き換えるものではありません。[ChatKit データストアの実装に関する `chatkit-python` ガイド](https://github.com/openai/chatkit-python/blob/main/docs/guides/respond-to-user-message.md#implement-your-chatkit-data-store)を参照してください。

### OpenAI Conversations API セッション

`OpenAIConversationsSession` を通じて、[OpenAI の Conversations API](https://platform.openai.com/docs/api-reference/conversations)を使用します。

```python
from agents import Agent, Runner, OpenAIConversationsSession

# Create agent
agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

# Create a new conversation
session = OpenAIConversationsSession()

# Optionally resume a previous conversation by passing a conversation ID
# session = OpenAIConversationsSession(conversation_id="conv_123")

# Start conversation
result = await Runner.run(
    agent,
    "What city is the Golden Gate Bridge in?",
    session=session
)
print(result.final_output)  # "San Francisco"

# Continue the conversation
result = await Runner.run(
    agent,
    "What state is it in?",
    session=session
)
print(result.final_output)  # "California"
```

### OpenAI Responses 圧縮セッション

Responses API（`responses.compact`）を使用して保存済みの会話履歴を圧縮するには、`OpenAIResponsesCompactionSession` を使用します。これは基盤となるセッションをラップし、`should_trigger_compaction` に基づいて各ターン後に自動圧縮できます。`OpenAIConversationsSession` をラップしないでください。この 2 つの機能は異なる方法で履歴を管理します。

#### 典型的な使用方法（自動圧縮）

```python
from agents import Agent, Runner, SQLiteSession
from agents.memory import OpenAIResponsesCompactionSession

underlying = SQLiteSession("conversation_123")
session = OpenAIResponsesCompactionSession(
    session_id="conversation_123",
    underlying_session=underlying,
)

agent = Agent(name="Assistant")
result = await Runner.run(agent, "Hello", session=session)
print(result.final_output)
```

デフォルトでは、候補のしきい値に達すると、各ターン後に圧縮が実行されます。

Responses API のレスポンス ID を使用してすでにターンを連結している場合は、`compaction_mode="previous_response_id"` が最適です。`compaction_mode="input"` は、代わりに現在のセッション項目から圧縮リクエストを再構築します。これは、レスポンスチェーンを利用できない場合や、セッションの内容を信頼できる情報源にしたい場合に便利です。デフォルトの `"auto"` は、利用可能な最も安全なオプションを選択します。

エージェントが `ModelSettings(store=False)` で実行されている場合、Responses API は後から参照できるように最後のレスポンスを保持しません。このステートレスな構成では、デフォルトの `"auto"` モードは `previous_response_id` に依存せず、入力ベースの圧縮にフォールバックします。完全な例については、[`examples/memory/compaction_session_stateless_example.py`](https://github.com/openai/openai-agents-python/tree/main/examples/memory/compaction_session_stateless_example.py)を参照してください。

#### 自動圧縮によるストリーミングのブロック

圧縮はセッション履歴を消去して書き換えるため、SDK は圧縮が完了するまで実行を完了と見なしません。ストリーミングモードでは、圧縮処理が重い場合、最後の出力トークンの後も `run.stream_events()` が数秒間開いたままになることがあります。

低レイテンシーのストリーミングや素早いターン移行が必要な場合は、自動圧縮を無効にし、ターン間またはアイドル時に `run_compaction()` を自分で呼び出してください。独自の基準に基づいて、圧縮を強制するタイミングを決定できます。

```python
from agents import Agent, Runner, SQLiteSession
from agents.memory import OpenAIResponsesCompactionSession

underlying = SQLiteSession("conversation_123")
session = OpenAIResponsesCompactionSession(
    session_id="conversation_123",
    underlying_session=underlying,
    # Disable triggering the auto compaction
    should_trigger_compaction=lambda _: False,
)

agent = Agent(name="Assistant")
result = await Runner.run(agent, "Hello", session=session)

# Decide when to compact (e.g., on idle, every N turns, or size thresholds).
await session.run_compaction({"force": True})
```

### SQLite セッション

SQLite を使用する、デフォルトの軽量なセッション実装です。

```python
from agents import SQLiteSession

# In-memory database (lost when process ends)
session = SQLiteSession("user_123")

# Persistent file-based database
session = SQLiteSession("user_123", "conversations.db")

# Use the session
result = await Runner.run(
    agent,
    "Hello",
    session=session
)
```

### 非同期 SQLite セッション

`aiosqlite` をバックエンドとする SQLite 永続化が必要な場合は、`AsyncSQLiteSession` を使用します。

```bash
pip install aiosqlite
```

```python
from agents import Agent, Runner
from agents.extensions.memory import AsyncSQLiteSession

agent = Agent(name="Assistant")
session = AsyncSQLiteSession("user_123", db_path="conversations.db")
result = await Runner.run(agent, "Hello", session=session)
```

### Redis セッション

複数のワーカーやサービス間でセッションメモリを共有するには、`RedisSession` を使用します。

```bash
pip install openai-agents[redis]
```

```python
from agents import Agent, Runner
from agents.extensions.memory import RedisSession

agent = Agent(name="Assistant")
session = RedisSession.from_url(
    "user_123",
    url="redis://localhost:6379/0",
)
result = await Runner.run(agent, "Hello", session=session)
await session.close()
```

`from_url(...)` は Redis クライアントを作成して所有します。`close()` の後、セッションは終了状態となり、後続のセッション操作では `RuntimeError` が発生します。`close()` を繰り返し呼び出したり、同時に呼び出したりしても安全です。アプリケーションがすでに Redis クライアントを管理している場合は、`redis_client=...` を指定して `RedisSession(...)` を直接構築してください。その場合、`close()` は何も行わず、呼び出し元がクライアントの所有権を保持し、セッションも引き続き利用できます。

### SQLAlchemy セッション

SQLAlchemy がサポートする任意のデータベースを使用する、本番環境対応の Agents SDK セッション永続化です。

```python
from agents.extensions.memory import SQLAlchemySession

# Using database URL
session = SQLAlchemySession.from_url(
    "user_123",
    url="postgresql+asyncpg://user:pass@localhost/db",
    create_tables=True
)

# Using existing engine
from sqlalchemy.ext.asyncio import create_async_engine
engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
session = SQLAlchemySession("user_123", engine=engine, create_tables=True)
```

詳細なドキュメントについては、[SQLAlchemy セッション](sqlalchemy_session.md)を参照してください。

### Dapr セッション

すでに Dapr サイドカーを実行している場合や、エージェントコードを変更せずに異なる状態ストアのバックエンド間で移行できるセッションストレージが必要な場合は、`DaprSession` を使用します。

```bash
pip install openai-agents[dapr]
```

```python
from agents import Agent, Runner
from agents.extensions.memory import DaprSession

agent = Agent(name="Assistant")

async with DaprSession.from_address(
    "user_123",
    state_store_name="statestore",
    dapr_address="localhost:50001",
) as session:
    result = await Runner.run(agent, "Hello", session=session)
    print(result.final_output)
```

注記:

-   `from_address(...)` は Dapr クライアントを作成して所有します。アプリがすでにクライアントを管理している場合は、`dapr_client=...` を指定して `DaprSession(...)` を直接構築してください。
-   コンテキストを終了するか `close()` を呼び出すと、所有クライアントを使用するセッションは終了状態になります。後続のセッション操作では `RuntimeError` が発生しますが、`close()` を繰り返し呼び出したり、同時に呼び出したりしても安全です。注入されたクライアントを使用する場合、`close()` は何も行わず、セッションは引き続き利用できます。
-   バックエンドの状態ストアが TTL をサポートしている場合に、古いセッションデータを自動的に期限切れにするには、`ttl=...` を渡します。
-   書き込み後の読み取りについて、より強い一貫性保証が必要な場合は、`consistency=DAPR_CONSISTENCY_STRONG` を渡します。
-   Dapr Python SDK は HTTP サイドカーエンドポイントも確認します。ローカル開発では、`dapr_address` で使用する gRPC ポートに加えて、`--dapr-http-port 3500` を指定して Dapr を起動してください。
-   ローカルコンポーネントやトラブルシューティングを含む完全なセットアップ手順については、[`examples/memory/dapr_session_example.py`](https://github.com/openai/openai-agents-python/tree/main/examples/memory/dapr_session_example.py)を参照してください。


### MongoDB セッション

MongoDB をすでに使用しているアプリケーションや、水平スケーリング可能なマルチプロセスのセッションストレージが必要な場合は、`MongoDBSession` を使用します。

```bash
pip install openai-agents[mongodb]
```

```python
from agents import Agent, Runner
from agents.extensions.memory import MongoDBSession

agent = Agent(name="Assistant")

# Create from URI — owns the client and closes it when session.close() is called
session = MongoDBSession.from_uri(
    "user-123",
    uri="mongodb://localhost:27017",
    database="agents",
)
result = await Runner.run(agent, "Hello", session=session)
print(result.final_output)
await session.close()
```

注記:

-   `from_uri(...)` は `AsyncMongoClient` を作成して所有し、`session.close()` で閉じます。所有クライアントを使用するセッションは `close()` 後に終了状態となり、後続のセッション操作では `RuntimeError` が発生します。アプリケーションがすでにクライアントを管理している場合は、`client=...` を指定して `MongoDBSession(...)` を直接構築してください。その場合、`session.close()` は何も行わず、ライフサイクルとセッションの利用可否は呼び出し元が管理します。
-   ほかに変更を加えることなく、`mongodb+srv://user:password@cluster.example.mongodb.net` URI を `from_uri(...)` に渡すことで、[MongoDB Atlas](https://www.mongodb.com/products/platform)に接続できます。
-   2 つのコレクションが使用され、どちらの名前も `sessions_collection=`（デフォルトは `agent_sessions`）と `messages_collection=`（デフォルトは `agent_messages`）で設定できます。インデックスは初回使用時に自動的に作成されます。各メッセージドキュメントには単調増加する `seq` カウンターが含まれ、同時実行される書き込み元やプロセス間で順序を維持します。
-   最初の実行前に接続を確認するには、`await session.ping()` を使用します。

### 高度な SQLite セッション

会話の分岐、使用状況分析、構造化クエリに対応した拡張 SQLite セッションです。

```python
from agents.extensions.memory import AdvancedSQLiteSession

# Create with advanced features
session = AdvancedSQLiteSession(
    session_id="user_123",
    db_path="conversations.db",
    create_tables=True
)

# Automatic usage tracking
result = await Runner.run(agent, "Hello", session=session)
await session.store_run_usage(result)  # Track token usage

# Conversation branching
await session.create_branch_from_turn(2)  # Branch from turn 2
```

詳細なドキュメントについては、[高度な SQLite セッション](advanced_sqlite_session.md)を参照してください。

### 暗号化セッション

任意のセッション実装に対応する透過的な暗号化ラッパーです。

```python
from agents.extensions.memory import EncryptedSession, SQLAlchemySession

# Create underlying session
underlying_session = SQLAlchemySession.from_url(
    "user_123",
    url="sqlite+aiosqlite:///conversations.db",
    create_tables=True
)

# Wrap with encryption and TTL
session = EncryptedSession(
    session_id="user_123",
    underlying_session=underlying_session,
    encryption_key="your-secret-key",
    ttl=600  # 10 minutes
)

result = await Runner.run(agent, "Hello", session=session)
```

詳細なドキュメントについては、[暗号化セッション](encrypted_session.md)を参照してください。

### その他のセッションタイプ

ほかにもいくつかの組み込みオプションがあります。`examples/memory/` および `extensions/memory/` 配下のソースコードを参照してください。

## 運用パターン

### セッション ID の命名

会話の整理に役立つ、意味のあるセッション ID を使用してください。

-   ユーザーベース: `"user_12345"`
-   スレッドベース: `"thread_abc123"`
-   コンテキストベース: `"support_ticket_456"`

### メモリの永続化

-   一時的な会話には、インメモリ SQLite（`SQLiteSession("session_id")`）を使用します
-   永続的な会話には、ファイルベースの SQLite（`SQLiteSession("session_id", "path/to/db.sqlite")`）を使用します
-   `aiosqlite` ベースの実装が必要な場合は、非同期 SQLite（`AsyncSQLiteSession("session_id", db_path="...")`）を使用します
-   共有可能で低レイテンシーのセッションメモリには、Redis をバックエンドとするセッション（`RedisSession.from_url("session_id", url="redis://...")`）を使用します
-   SQLAlchemy がサポートする既存のデータベースを使用する本番システムには、SQLAlchemy ベースのセッション（`SQLAlchemySession("session_id", engine=engine, create_tables=True)`）を使用します
-   MongoDB をすでに使用しているアプリケーションや、水平スケーリング可能なマルチプロセスのセッションストレージが必要な場合は、MongoDB セッション（`MongoDBSession.from_uri("session_id", uri="mongodb://localhost:27017")`）を使用します
-   組み込みのテレメトリ、トレーシング、データ分離機能と 30 種類超のデータベースバックエンドのサポートが必要な本番環境のクラウドネイティブデプロイには、Dapr 状態ストアセッション（`DaprSession.from_address("session_id", state_store_name="statestore", dapr_address="localhost:50001")`）を使用します
-   OpenAI Conversations API に履歴を保存したい場合は、OpenAI がホストするストレージ（`OpenAIConversationsSession()`）を使用します
-   任意のセッションを透過的な暗号化と TTL ベースの期限切れ機能でラップするには、暗号化セッション（`EncryptedSession(session_id, underlying_session, encryption_key)`）を使用します
-   より高度なユースケースでは、ほかの本番システム（Django など）向けのカスタムセッションバックエンドの実装を検討してください

### 複数のセッション

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant")

# Different sessions maintain separate conversation histories
session_1 = SQLiteSession("user_123", "conversations.db")
session_2 = SQLiteSession("user_456", "conversations.db")

result1 = await Runner.run(
    agent,
    "Help me with my account",
    session=session_1
)
result2 = await Runner.run(
    agent,
    "What are my charges?",
    session=session_2
)
```

### セッションの共有

```python
# Different agents can share the same session
support_agent = Agent(name="Support")
billing_agent = Agent(name="Billing")
session = SQLiteSession("user_123")

# Both agents will see the same conversation history
result1 = await Runner.run(
    support_agent,
    "Help me with my account",
    session=session
)
result2 = await Runner.run(
    billing_agent,
    "What are my charges?",
    session=session
)
```

## 完全な例

セッションメモリの実際の動作を示す完全な例を次に示します。

```python
import asyncio
from agents import Agent, Runner, SQLiteSession


async def main():
    # Create an agent
    agent = Agent(
        name="Assistant",
        instructions="Reply very concisely.",
    )

    # Create a session instance that will persist across runs
    session = SQLiteSession("conversation_123", "conversation_history.db")

    print("=== Sessions Example ===")
    print("The agent will remember previous messages automatically.\n")

    # First turn
    print("First turn:")
    print("User: What city is the Golden Gate Bridge in?")
    result = await Runner.run(
        agent,
        "What city is the Golden Gate Bridge in?",
        session=session
    )
    print(f"Assistant: {result.final_output}")
    print()

    # Second turn - the agent will remember the previous conversation
    print("Second turn:")
    print("User: What state is it in?")
    result = await Runner.run(
        agent,
        "What state is it in?",
        session=session
    )
    print(f"Assistant: {result.final_output}")
    print()

    # Third turn - continuing the conversation
    print("Third turn:")
    print("User: What's the population of that state?")
    result = await Runner.run(
        agent,
        "What's the population of that state?",
        session=session
    )
    print(f"Assistant: {result.final_output}")
    print()

    print("=== Conversation Complete ===")
    print("Notice how the agent remembered the context from previous turns!")
    print("Sessions automatically handles conversation history.")


if __name__ == "__main__":
    asyncio.run(main())
```

## カスタムセッション実装

[`Session`][agents.memory.session.Session] プロトコルに従うクラスを作成することで、独自のセッションメモリを実装できます。

```python
from agents.memory.session import SessionABC
from agents.items import TResponseInputItem
from typing import List

class MyCustomSession(SessionABC):
    """Custom session implementation following the Session protocol."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        # Your initialization here

    async def get_items(self, limit: int | None = None) -> List[TResponseInputItem]:
        """Retrieve conversation history for this session."""
        # Your implementation here
        pass

    async def add_items(self, items: List[TResponseInputItem]) -> None:
        """Store new items for this session."""
        # Your implementation here
        pass

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent item from this session."""
        # Your implementation here
        pass

    async def clear_session(self) -> None:
        """Clear all items for this session."""
        # Your implementation here
        pass

# Use your custom session
agent = Agent(name="Assistant")
result = await Runner.run(
    agent,
    "Hello",
    session=MyCustomSession("my_session")
)
```

## コミュニティによるセッション実装

コミュニティによって、追加のセッション実装が開発されています。

| パッケージ | 説明 |
|---------|-------------|
| [openai-django-sessions](https://pypi.org/project/openai-django-sessions/) | Django がサポートする任意のデータベース（PostgreSQL、MySQL、SQLite など）向けの Django ORM ベースのセッション |

セッション実装を構築した場合は、ここに追加するためのドキュメント PR をぜひ送信してください。

## API リファレンス

詳細な API ドキュメントについては、以下を参照してください。

-   [`Session`][agents.memory.session.Session] - プロトコルインターフェース
-   [`OpenAIConversationsSession`][agents.memory.OpenAIConversationsSession] - OpenAI Conversations API 実装
-   [`OpenAIResponsesCompactionSession`][agents.memory.openai_responses_compaction_session.OpenAIResponsesCompactionSession] - Responses API 圧縮ラッパー
-   [`SQLiteSession`][agents.memory.sqlite_session.SQLiteSession] - 基本的な SQLite 実装
-   [`AsyncSQLiteSession`][agents.extensions.memory.async_sqlite_session.AsyncSQLiteSession] - `aiosqlite` ベースの非同期 SQLite 実装
-   [`RedisSession`][agents.extensions.memory.redis_session.RedisSession] - Redis をバックエンドとするセッション実装
-   [`SQLAlchemySession`][agents.extensions.memory.sqlalchemy_session.SQLAlchemySession] - SQLAlchemy ベースの実装
-   [`MongoDBSession`][agents.extensions.memory.mongodb_session.MongoDBSession] - MongoDB をバックエンドとするセッション実装
-   [`DaprSession`][agents.extensions.memory.dapr_session.DaprSession] - Dapr 状態ストア実装
-   [`AdvancedSQLiteSession`][agents.extensions.memory.advanced_sqlite_session.AdvancedSQLiteSession] - 分岐と分析に対応した拡張 SQLite
-   [`EncryptedSession`][agents.extensions.memory.encrypt_session.EncryptedSession] - 任意のセッションに対応する暗号化ラッパー