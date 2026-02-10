import json
from typing import Any, Dict, List

from .base import Benchmark
from .evaluate import run_evaluation


class BenchmarkBIRD(Benchmark):
    use_evidence: bool = False
    """Dump evidence directly into the `user_query` instead of the prompt
    to help the model. (debatable, consult teacher)"""


    def _load_queries(self) -> List[dict]:
        with open(self.query_file, "r") as f:
            return json.load(f)

    async def predict(self, tool_dict: Dict[str, Any]) -> Dict[str, str]:
        """Returns predictions in BIRD's expected format."""
        queries = self._load_queries()
        predictions = {}

        for item in queries:
            qid = item["question_id"]
            db_id = item["db_id"]
            question = item["question"]
            evidence = item["evidence"]

            tool = tool_dict[db_id]

            if self.use_evidence:
                question = f"question: {question}, evidence (may be empty): {evidence}"

            result = await tool.query(question, check_ambiguity=True)
            print(result)

            sql_query = "error"

            # TODO: check if this result format can be reworked
            root = result.get("params", result) if isinstance(result, dict) else {}
            details = root.get("data", {}).get("details", {})

            status = details.get("status")

            if status == "ambiguous_query":
                sql_query = "ambiguous"
            elif status == "success":
                sql_query = details.get("sql_query", "error")

            predictions[str(qid)] = sql_query

        return predictions

    def _save_predictions(self, predictions: Dict[str, str], output_path: str):
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

    async def evaluate(self, predictions: Dict[str, str]) -> dict:
        self._save_predictions(predictions, "./query_results.json")
        report = run_evaluation(predictions, self.answer_file, self.db_url)

        return report
