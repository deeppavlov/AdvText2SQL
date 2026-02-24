import json
from typing import Any, Dict, List

from .base import BenchmarkBase
from .evaluate_ambrosia import run_evaluation
from .response import ToolResponse


class BenchmarkAmbrosia(BenchmarkBase):

    def _load_queries(self) -> List[dict]:
        with open(self.query_file, "r") as f:
            return json.load(f)

    async def predict(self, tool_dict: Dict[str, Any]) -> Dict[str, str]:
        """Returns predictions in the expected format."""
        queries = self._load_queries()
        predictions = {}

        for item in queries:
            qid = item["question_id"]
            db_id = item["db_id"]
            question = item["question"]

            tool = tool_dict[db_id]

            result = await tool.query(question)
            print(result)

            result = ToolResponse.model_validate(result)

            if result.status == "ambiguous":
                sql_query = "ambiguous"
            elif result.status == "success":
                sql_query = result.query
            else:
                sql_query = "error"

            predictions[str(qid)] = sql_query

        return predictions

    def _save_predictions(self, predictions: Dict[str, str], output_path: str):
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

    async def evaluate(self, predictions: Dict[str, str]) -> dict:
        self._save_predictions(predictions, "./query_results.json")
        report = run_evaluation(predictions, self.answer_file)

        return report
