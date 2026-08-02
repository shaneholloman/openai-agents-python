---
search:
  exclude: true
---
# ガードレール

ガードレールを使用すると、ユーザー入力とエージェント出力のチェックおよび検証を行えます。たとえば、非常に高性能である一方、低速でコストの高いモデルを使用して顧客のリクエストに対応するエージェントがあるとします。悪意のあるユーザーに、数学の宿題を手伝うようモデルへ依頼されることは避けたいでしょう。そのため、高速で低コストのモデルを使用してガードレールを実行できます。ガードレールが悪意のある利用を検出した場合、直ちにエラーを発生させて高コストのモデルの実行を防ぎ、時間と費用を節約できます **（ブロッキングガードレールを使用する場合。並列ガードレールでは、ガードレールが完了する前に高コストのモデルがすでに実行を開始している可能性があります。詳細については、以下の「実行モード」を参照してください）** 。

ガードレールには次の 2 種類があります。

1. 入力ガードレールは、最初のユーザー入力に対して実行されます
2. 出力ガードレールは、最終的なエージェント出力に対して実行されます

## ワークフローの境界

ガードレールはエージェントとツールに設定されますが、すべてがワークフロー内の同じ時点で実行されるわけではありません。

- **入力ガードレール** は、チェーン内の最初のエージェントに対してのみ実行されます。
- **出力ガードレール** は、最終出力を生成するエージェントに対してのみ実行されます。
- **ツールガードレール** は、カスタム関数ツールが呼び出されるたびに実行されます。入力ガードレールは実行前に、出力ガードレールは実行後に実行されます。

マネージャー、ハンドオフ、または委任されたスペシャリストを含むワークフローで、各カスタム関数ツールの呼び出し前後にチェックが必要な場合は、エージェントレベルの入力／出力ガードレールだけに依存せず、ツールガードレールを使用してください。

## 入力ガードレール

入力ガードレールは、次の 3 ステップで実行されます。

1. 最初に、ガードレールはエージェントに渡されたものと同じ入力を受け取ります。
2. 次に、ガードレール関数が実行されて [`GuardrailFunctionOutput`][agents.guardrail.GuardrailFunctionOutput] を生成し、それが [`InputGuardrailResult`][agents.guardrail.InputGuardrailResult] にラップされます
3. 最後に、[`.tripwire_triggered`][agents.guardrail.GuardrailFunctionOutput.tripwire_triggered] が true かどうかを確認します。true の場合、[`InputGuardrailTripwireTriggered`][agents.exceptions.InputGuardrailTripwireTriggered] 例外が発生するため、ユーザーへ適切に応答するか、例外を処理できます。

!!! Note

    入力ガードレールはユーザー入力に対して実行することを意図しているため、エージェントのガードレールは、そのエージェントが *最初の* エージェントである場合にのみ実行されます。なぜ `guardrails` プロパティを `Runner.run` に渡すのではなく、エージェントに設定するのか疑問に思うかもしれません。これは、ガードレールが実際のエージェントに関連する傾向があるためです。エージェントごとに異なるガードレールを実行するため、コードを同じ場所にまとめると可読性が向上します。

### 実行モード

入力ガードレールは、次の 2 つの実行モードをサポートしています。

- **並列実行** （デフォルト、`run_in_parallel=True`）：ガードレールはエージェントの実行と並行して実行されます。両方が同時に開始されるため、レイテンシーを最小限に抑えられます。ただし、ガードレールが失敗した場合、キャンセルされる前にエージェントがすでにトークンを消費し、ツールを実行している可能性があります。

- **ブロッキング実行** （`run_in_parallel=False`）：ガードレールは、エージェントが開始する *前に* 実行され、完了します。ガードレールのトリップワイヤーが作動した場合、エージェントは実行されないため、トークンの消費とツールの実行を防げます。コストを最適化したい場合や、ツール呼び出しによる潜在的な副作用を避けたい場合に最適です。

## 出力ガードレール

出力ガードレールは、次の 3 ステップで実行されます。

1. 最初に、ガードレールはエージェントが生成した出力を受け取ります。
2. 次に、ガードレール関数が実行されて [`GuardrailFunctionOutput`][agents.guardrail.GuardrailFunctionOutput] を生成し、それが [`OutputGuardrailResult`][agents.guardrail.OutputGuardrailResult] にラップされます
3. 最後に、[`.tripwire_triggered`][agents.guardrail.GuardrailFunctionOutput.tripwire_triggered] が true かどうかを確認します。true の場合、[`OutputGuardrailTripwireTriggered`][agents.exceptions.OutputGuardrailTripwireTriggered] 例外が発生するため、ユーザーへ適切に応答するか、例外を処理できます。

!!! Note

    出力ガードレールは最終的なエージェント出力に対して実行することを意図しているため、エージェントのガードレールは、そのエージェントが *最後の* エージェントである場合にのみ実行されます。入力ガードレールと同様に、これはガードレールが実際のエージェントに関連する傾向があるためです。エージェントごとに異なるガードレールを実行するため、コードを同じ場所にまとめると可読性が向上します。

    出力ガードレールは常にエージェントの完了後に実行されるため、`run_in_parallel` パラメーターはサポートしていません。

## ツールガードレール

ツールガードレールは **関数ツール** をラップし、実行前後にツール呼び出しを検証またはブロックできるようにします。ツール自体に設定され、そのツールが呼び出されるたびに実行されます。

- 入力ツールガードレールはツールの実行前に実行され、呼び出しのスキップ、メッセージによる出力の置き換え、またはトリップワイヤーの作動が可能です。
- 出力ツールガードレールはツールの実行後に実行され、出力の置き換え、またはトリップワイヤーの作動が可能です。
- 関数ツールに承認が必要な場合、通常、入力ツールガードレールは承認後、実行の直前に実行されます。保留中の承認による中断が発生する前にこれらの入力チェックを実行する場合は、[`RunConfig.tool_execution`][agents.run.RunConfig.tool_execution] を [`ToolExecutionConfig(pre_approval_tool_input_guardrails=True)`][agents.run.ToolExecutionConfig] に設定してください。この承認前チェックに合格した呼び出しも、承認後、ツールの実行前に再度チェックされます。
- ツールガードレールは、[`function_tool`][agents.tool.function_tool] で作成された関数ツールにのみ適用されます。ハンドオフは通常の関数ツールパイプラインではなく SDK のハンドオフパイプラインを通じて実行されるため、ツールガードレールはハンドオフ呼び出し自体には適用されません。ホスト型ツール（`WebSearchTool`、`FileSearchTool`、`HostedMCPTool`、`CodeInterpreterTool`、`ImageGenerationTool`）と組み込み実行ツール（`ComputerTool`、`ShellTool`、`ApplyPatchTool`、`LocalShellTool`）も、このガードレールパイプラインを使用しません。また、[`Agent.as_tool()`][agents.agent.Agent.as_tool] は現在、ツールガードレールのオプションを直接公開していません。

詳細については、以下のコードスニペットを参照してください。

## トリップワイヤー

入力または出力がガードレールのチェックに失敗した場合、ガードレールはトリップワイヤーによってこれを通知できます。トリップワイヤーが作動したガードレールを検出すると、直ちに `{Input,Output}GuardrailTripwireTriggered` 例外が発生し、エージェントの実行が停止します。

例外の `guardrail_result` は、トリップワイヤーを作動させたガードレールを識別します。Runner によって発生した入力トリップワイヤーの場合、`exception.run_data.input_guardrail_results` には、実行が停止する前に完了したすべての入力ガードレールの実行結果が含まれ、トリップワイヤーを作動させた実行結果も含まれます。ストリーミング実行結果では、`stream_events()` が例外を発生させた後、同じ蓄積済みの実行結果が `input_guardrail_results` を通じて公開されます。Runner が管理する実行パスの外部で例外が発生した場合、`run_data` は `None` になることがあります。

## ガードレールの実装

入力を受け取り、[`GuardrailFunctionOutput`][agents.guardrail.GuardrailFunctionOutput] を返す関数を用意する必要があります。この例では、内部でエージェントを実行することで実装します。

```python
from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
)
from agents.decorators import input_guardrail

class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reasoning: str

guardrail_agent = Agent( # (1)!
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkOutput,
)


@input_guardrail
async def math_guardrail( # (2)!
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output, # (3)!
        tripwire_triggered=result.final_output.is_math_homework,
    )


agent = Agent(  # (4)!
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    input_guardrails=[math_guardrail],
)

async def main():
    # This should trip the guardrail
    try:
        await Runner.run(agent, "Hello, can you help me solve for x: 2x + 3 = 11?")
        print("Guardrail didn't trip - this is unexpected")

    except InputGuardrailTripwireTriggered:
        print("Math homework guardrail tripped")
```

1. このエージェントをガードレール関数内で使用します。
2. これは、エージェントの入力／コンテキストを受け取り、実行結果を返すガードレール関数です。
3. ガードレールの実行結果には追加情報を含められます。
4. これは、ワークフローを定義する実際のエージェントです。

出力ガードレールも同様です。

```python
from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
)
from agents.decorators import output_guardrail
class MessageOutput(BaseModel): # (1)!
    response: str

class MathOutput(BaseModel): # (2)!
    reasoning: str
    is_math: bool

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the output includes any math.",
    output_type=MathOutput,
)

@output_guardrail
async def math_guardrail(  # (3)!
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, output.response, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math,
    )

agent = Agent( # (4)!
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    output_guardrails=[math_guardrail],
    output_type=MessageOutput,
)

async def main():
    # This should trip the guardrail
    try:
        await Runner.run(agent, "Hello, can you help me solve for x: 2x + 3 = 11?")
        print("Guardrail didn't trip - this is unexpected")

    except OutputGuardrailTripwireTriggered:
        print("Math output guardrail tripped")
```

1. これは、実際のエージェントの出力型です。
2. これは、ガードレールの出力型です。
3. これは、エージェントの出力を受け取り、実行結果を返すガードレール関数です。
4. これは、ワークフローを定義する実際のエージェントです。

最後に、ツールガードレールの例を示します。

```python
import json
from agents import (
    Agent,
    Runner,
    ToolGuardrailFunctionOutput,
)
from agents.decorators import tool, tool_input_guardrail, tool_output_guardrail

@tool_input_guardrail
def block_secrets(data):
    args = json.loads(data.context.tool_arguments or "{}")
    if "sk-" in json.dumps(args):
        return ToolGuardrailFunctionOutput.reject_content(
            "Remove secrets before calling this tool."
        )
    return ToolGuardrailFunctionOutput.allow()


@tool_output_guardrail
def redact_output(data):
    text = str(data.output or "")
    if "sk-" in text:
        return ToolGuardrailFunctionOutput.reject_content("Output contained sensitive data.")
    return ToolGuardrailFunctionOutput.allow()


@tool(
    tool_input_guardrails=[block_secrets],
    tool_output_guardrails=[redact_output],
)
def classify_text(text: str) -> str:
    """Classify text for internal routing."""
    return f"length:{len(text)}"


agent = Agent(name="Classifier", tools=[classify_text])
result = Runner.run_sync(agent, "hello world")
print(result.final_output)
```