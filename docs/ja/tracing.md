---
search:
  exclude: true
---
# トレーシング

Agents SDK には組み込みのトレーシング機能があり、エージェントの実行中に発生する LLM 生成、ツール呼び出し、ハンドオフ、ガードレール、さらにはカスタムイベントまで、包括的なイベント記録を収集します。[トレースダッシュボード](https://platform.openai.com/traces)を使用すると、開発時および本番環境でワークフローをデバッグ、可視化、監視できます。

!!!note

    トレーシングはデフォルトで有効です。一般的な無効化方法は次の 3 つです。

    1. 環境変数 `OPENAI_AGENTS_DISABLE_TRACING=1` を設定して、トレーシングをグローバルに無効化できます
    2. [`set_tracing_disabled(True)`][agents.set_tracing_disabled] を使用して、コード内でトレーシングをグローバルに無効化できます
    3. [`agents.run.RunConfig.tracing_disabled`][] を `True` に設定して、単一の実行に対するトレーシングを無効化できます

***Zero Data Retention (ZDR) ポリシーの下で OpenAI の API を使用する組織では、トレーシングを利用できません。***

## トレースとスパン {#traces-and-spans}

-   **トレース** は、「ワークフロー」における単一のエンドツーエンド操作を表します。トレースは複数のスパンで構成されます。トレースには次のプロパティがあります。
    -   `workflow_name`: 論理的なワークフローまたはアプリの名前です。たとえば、「コード生成」や「カスタマーサービス」などです。
    -   `trace_id`: トレースの一意な ID です。指定しない場合は自動的に生成されます。形式は `trace_<32_alphanumeric>` である必要があります。
    -   `group_id`: 同じ会話に属する複数のトレースを関連付けるための、省略可能なグループ ID です。たとえば、チャットスレッド ID を使用できます。
    -   `disabled`: True の場合、トレースは記録されません。
    -   `metadata`: トレースの省略可能なメタデータです。
-   **スパン** は、開始時刻と終了時刻を持つ操作を表します。スパンには次のプロパティがあります。
    -   `started_at` および `ended_at` のタイムスタンプ。
    -   `trace_id`: 所属するトレースを表します
    -   `parent_id`: このスパンの親スパンが存在する場合、その親スパンを指します
    -   `span_data`: スパンに関する情報です。たとえば、`AgentSpanData` にはエージェントに関する情報が含まれ、`GenerationSpanData` には LLM 生成に関する情報が含まれます。

## デフォルトのトレーシング {#default-tracing}

デフォルトでは、SDK は次の項目をトレーシングします。

-   `Runner.{run, run_sync, run_streamed}()` 全体が `trace()` でラップされます。
-   各ランナー呼び出しが `task_span()` でラップされます。
-   各モデルターンが `turn_span()` でラップされます。
-   エージェントが実行されるたびに、`agent_span()` でラップされます
-   LLM 生成が `generation_span()` でラップされます
-   各関数ツール呼び出しが `function_span()` でラップされます
-   ガードレールが `guardrail_span()` でラップされます
-   ハンドオフが `handoff_span()` でラップされます
-   音声入力（音声テキスト変換）が `transcription_span()` でラップされます
-   音声出力（テキスト音声変換）が `speech_span()` でラップされます
-   SDK は、関連する音声スパンを `speech_group_span()` の配下にまとめる場合があります

デフォルトでは、トレース名はリテラル文字列 `Agent workflow` です。`trace` を使用する場合はこの名前を設定でき、[`RunConfig`][agents.run.RunConfig] を使用すれば名前やその他のプロパティを設定できます。

よりコンパクトな階層にする場合は、実行に対するタスクスパンとターンスパンの自動作成を無効にします。エージェント、生成、関数、ガードレール、ハンドオフ、およびカスタムの各スパンは引き続き記録されます。

```python
from agents import RunConfig, Runner

result = await Runner.run(
    agent,
    "Hello",
    run_config=RunConfig(tracing={"include_task_and_turn_spans": False}),
)
```

さらに、[カスタムトレースプロセッサー](#custom-tracing-processors)を設定して、別の送信先へトレースを送信できます。これは、既存の送信先の代替または追加の送信先として使用できます。

## 長時間実行ワーカーと即時エクスポート {#long-running-workers-and-immediate-exports}

デフォルトの [`BatchTraceProcessor`][agents.tracing.processors.BatchTraceProcessor] は、数秒ごと、またはインメモリキューがサイズのトリガー値に達した場合はそれより早く、バックグラウンドでトレースをエクスポートします。また、プロセスの終了時には最終フラッシュも実行します。Celery、RQ、Dramatiq、FastAPI のバックグラウンドタスクなどの長時間実行ワーカーでは、通常、追加のコードなしでトレースが自動的にエクスポートされますが、各ジョブの完了直後にはトレースダッシュボードに表示されない場合があります。

作業単位の終了時に即時配信を保証する必要がある場合は、トレースコンテキストの終了後に [`flush_traces()`][agents.tracing.flush_traces] を呼び出します。

```python
from agents import Runner, flush_traces, trace


@celery_app.task
def run_agent_task(prompt: str):
    try:
        with trace("celery_task"):
            result = Runner.run_sync(agent, prompt)
        return result.final_output
    finally:
        flush_traces()
```

```python
from fastapi import BackgroundTasks, FastAPI
from agents import Runner, flush_traces, trace

app = FastAPI()


def process_in_background(prompt: str) -> None:
    try:
        with trace("background_job"):
            Runner.run_sync(agent, prompt)
    finally:
        flush_traces()


@app.post("/run")
async def run(prompt: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_in_background, prompt)
    return {"status": "queued"}
```

[`flush_traces()`][agents.tracing.flush_traces] は、現在バッファリングされているトレースとスパンがエクスポートされるまで処理をブロックします。そのため、構築途中のトレースをフラッシュしないよう、`trace()` が閉じた後に呼び出してください。デフォルトのエクスポート遅延で問題ない場合は、この呼び出しを省略できます。

## 上位レベルのトレース {#higher-level-traces}

複数回の `run()` 呼び出しを単一のトレースに含めたい場合があります。その場合は、コード全体を `trace()` でラップします。

```python
from agents import Agent, Runner, trace

async def main():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")

    with trace("Joke workflow"): # (1)!
        first_result = await Runner.run(agent, "Tell me a joke")
        second_result = await Runner.run(agent, f"Rate this joke: {first_result.final_output}")
        print(f"Joke: {first_result.final_output}")
        print(f"Rating: {second_result.final_output}")
```

1. 2 回の `Runner.run` 呼び出しが `with trace()` でラップされているため、それぞれが個別のトレースを作成するのではなく、両方の実行が 1 つの全体的なトレースに含まれます。

## トレースの作成 {#creating-traces}

[`trace()`][agents.tracing.trace] 関数を使用してトレースを作成できます。トレースは開始および終了する必要があります。これには次の 2 つの方法があります。

1. **推奨**: トレースをコンテキストマネージャーとして、すなわち `with trace(...) as my_trace` の形式で使用します。これにより、適切なタイミングでトレースが自動的に開始および終了されます。
2. [`trace.start()`][agents.tracing.Trace.start] と [`trace.finish()`][agents.tracing.Trace.finish] を手動で呼び出すこともできます。

現在のトレースは、Python の [`contextvar`](https://docs.python.org/3/library/contextvars.html) を介して追跡されます。つまり、並行処理でも自動的に機能します。トレースを手動で開始および終了する場合、現在のトレースを更新するには、`start()` に `mark_as_current` を渡し、`finish()` に `reset_current` を渡します。

## スパンの作成 {#creating-spans}

さまざまな [`*_span()`][agents.tracing.create] メソッドを使用してスパンを作成できます。通常、スパンを手動で作成する必要はありません。カスタムスパン情報を追跡するための [`custom_span()`][agents.tracing.custom_span] 関数も利用できます。

スパンは自動的に現在のトレースの一部となり、Python の [`contextvar`](https://docs.python.org/3/library/contextvars.html) を介して追跡される、最も近い現在のスパンの配下にネストされます。

## 機密データ {#sensitive-data}

一部のスパンでは、機密性の高い可能性があるデータがキャプチャされる場合があります。

`generation_span()` には LLM 生成の入力と出力が保存され、`function_span()` には関数呼び出しの入力と出力が保存されます。これらには機密データが含まれる可能性があるため、[`RunConfig.trace_include_sensitive_data`][agents.run.RunConfig.trace_include_sensitive_data] を使用して、そのデータのキャプチャを無効化できます。

同様に、音声スパンには、デフォルトで入出力音声の Base64 エンコードされた PCM データが含まれます。[`VoicePipelineConfig.trace_include_sensitive_audio_data`][agents.voice.pipeline_config.VoicePipelineConfig.trace_include_sensitive_audio_data] を設定することで、この音声データのキャプチャを無効化できます。

デフォルトでは、`trace_include_sensitive_data` は `True` です。アプリを実行する前に、環境変数 `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA` を `true/1` または `false/0` に設定してエクスポートすることで、コードを変更せずにデフォルト値を設定できます。

## カスタムトレースプロセッサー {#custom-tracing-processors}

トレーシングの高レベルアーキテクチャは次のとおりです。

-   初期化時に、トレースの作成を担当するグローバルな [`TraceProvider`][agents.tracing.provider.TraceProvider] を作成します。
-   `TraceProvider` に [`BatchTraceProcessor`][agents.tracing.processors.BatchTraceProcessor] を設定します。これは、トレースとスパンをバッチで [`BackendSpanExporter`][agents.tracing.processors.BackendSpanExporter] に送信し、そこからスパンとトレースをバッチで OpenAI バックエンドにエクスポートします。

このデフォルト設定をカスタマイズし、代替または追加のバックエンドにトレースを送信したり、エクスポーターの動作を変更したりするには、次の 2 つの方法があります。

1. [`add_trace_processor()`][agents.tracing.add_trace_processor] を使用すると、準備が整ったトレースとスパンを受け取る **追加の** トレースプロセッサーを追加できます。これにより、トレースを OpenAI バックエンドに送信しながら、独自の処理も実行できます。
2. [`set_trace_processors()`][agents.tracing.set_trace_processors] を使用すると、デフォルトのプロセッサーを独自のトレースプロセッサーで **置き換える** ことができます。この場合、送信を行う `TracingProcessor` を含めない限り、トレースは OpenAI バックエンドに送信されません。


## OpenAI 以外のモデルによるトレーシング {#tracing-with-non-openai-models}

OpenAI 以外のモデルを使用する場合、トレーシングを無効化することなく OpenAI Traces ダッシュボードで無料のトレーシングを有効にするため、トレーシングエクスポーターに OpenAI API キーを指定できます。アダプターの選択と設定に関する注意事項については、モデルガイドの[サードパーティーアダプター](models/index.md#third-party-adapters)セクションを参照してください。

```python
import os
from agents import set_tracing_export_api_key, Agent
from agents.extensions.models.any_llm_model import AnyLLMModel

tracing_api_key = os.environ["OPENAI_API_KEY"]
set_tracing_export_api_key(tracing_api_key)

model = AnyLLMModel(
    model="your-provider/your-model-name",
    api_key="your-api-key",
)

agent = Agent(
    name="Assistant",
    model=model,
)
```

単一の実行に対してのみ別のトレーシングキーが必要な場合は、グローバルエクスポーターを変更する代わりに、`RunConfig` を介して渡します。

```python
from agents import Runner, RunConfig

await Runner.run(
    agent,
    input="Hello",
    run_config=RunConfig(tracing={"api_key": "sk-tracing-123"}),
)
```

## 補足事項 {#additional-notes}
- OpenAI Traces ダッシュボードで無料のトレースを確認できます。


## エコシステム統合 {#ecosystem-integrations}

以下のコミュニティおよびベンダー統合は、OpenAI Agents SDK のトレーシング API サーフェスをサポートしています。

### 外部トレースプロセッサーの一覧 {#external-tracing-processors-list}

-   [Weights & Biases](https://weave-docs.wandb.ai/guides/integrations/openai_agents)
-   [Arize-Phoenix](https://docs.arize.com/phoenix/tracing/integrations-tracing/openai-agents-sdk)
-   [Future AGI](https://docs.futureagi.com/docs/tracing/auto/openai_agents/)
-   [MLflow (セルフホスト型 / OSS)](https://mlflow.org/docs/latest/tracing/integrations/openai-agent)
-   [MLflow (Databricks ホスト型)](https://docs.databricks.com/aws/en/mlflow/mlflow-tracing#-automatic-tracing)
-   [Braintrust](https://braintrust.dev/docs/guides/traces/integrations#openai-agents-sdk)
-   [Pydantic Logfire](https://logfire.pydantic.dev/docs/integrations/llms/openai/#openai-agents)
-   [AgentOps](https://docs.agentops.ai/v1/integrations/agentssdk)
-   [Scorecard](https://docs.scorecard.io/docs/documentation/features/tracing#openai-agents-sdk-integration)
-   [Respan](https://respan.ai/docs/integrations/tracing/openai-agents-sdk)
-   [LangSmith](https://docs.smith.langchain.com/observability/how_to_guides/trace_with_openai_agents_sdk)
-   [Maxim AI](https://www.getmaxim.ai/docs/observe/integrations/openai-agents-sdk)
-   [Comet Opik](https://www.comet.com/docs/opik/tracing/integrations/openai_agents)
-   [Langfuse](https://langfuse.com/docs/integrations/openaiagentssdk/openai-agents)
-   [Langtrace](https://docs.langtrace.ai/supported-integrations/llm-frameworks/openai-agents-sdk)
-   [Okahu-Monocle](https://github.com/monocle2ai/monocle)
-   [Galileo](https://v2docs.galileo.ai/integrations/openai-agent-integration#openai-agent-integration)
-   [Portkey AI](https://portkey.ai/docs/integrations/agents/openai-agents)
-   [LangDB AI](https://docs.langdb.ai/getting-started/working-with-agent-frameworks/working-with-openai-agents-sdk)
-   [Agenta](https://docs.agenta.ai/observability/integrations/openai-agents)
-   [PostHog](https://posthog.com/docs/llm-analytics/installation/openai-agents)
-   [Traccia](https://traccia.ai/docs/integrations/openai-agents)
-   [PromptLayer](https://docs.promptlayer.com/features/integrations#openai-agents-sdk)
-   [HoneyHive](https://docs.honeyhive.ai/v2/integrations/openai-agents)
-   [Asqav](https://www.asqav.com/docs/integrations#openai-agents)
-   [Datadog](https://docs.datadoghq.com/llm_observability/instrumentation/auto_instrumentation/?tab=python#openai-agents)
-   [Latitude](https://docs.latitude.so/telemetry/frameworks/openai-agents)
-   [DProvenanceKit](https://dprovenance.dev/openai-agents/)
-   [Tuning Engines](https://github.com/cerebrixos-org/tuning-engines-cli/tree/main/packages/tuning-agents#openai-agents-sdk)