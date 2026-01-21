import logfire

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge, HasMatchingSpan

from partial_mcp.baseline_agent import chat, model

logfire.configure(
    send_to_logfire=False,
    console=False,
)
logfire.instrument_mcp()
logfire.instrument_pydantic_ai()


dataset = Dataset(
    cases=[
        Case(
            inputs="What is 2+2?",
            expected_output="4",
            evaluators=(
                HasMatchingSpan(
                    query={"name_contains": "add"},
                ),
            ),
        ),
        Case(
            inputs="What is 5+7?",
            expected_output="The answer is 12",
            evaluators=(
                HasMatchingSpan(
                    query={"name_contains": "add"},
                ),
            ),
        ),
    ],
    evaluators=[
        LLMJudge(
            rubric="Response provides the same answer as expected, possibly with explanation",
            include_input=True,
            include_expected_output=True,
            model=model,
        ),
    ],
)


if __name__ == "__main__":
    report = dataset.evaluate_sync(chat)

    report.print(
        include_output=True,
    )
