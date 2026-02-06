import json
from typing import Dict, List

from .base import Benchmark
from .evaluate import run_evaluation


class BenchmarkBIRD(Benchmark):
    def _load_queries(self) -> List[dict]:
        with open(self.query_file, "r") as f:
            return json.load(f)

    def predict(self, tool_dict: Dict[str, Any]) -> Dict[str, str]:
        """Returns predictions in BIRD's expected format."""
        queries = self._load_queries()
        predictions = {}

        for item in queries:
            qid = item["question_id"]
            db_id = item["db_id"]
            question = item["question"]

            tool = tool_dict[db_id]

            result = await tool.query(question)

            sql_query = "error"

            if "params" in result:
                data = result["params"].get("data", {})
                status = data.get("status")

                if status == "ambiguous_query":
                    sql_query = "ambiguous"
                else:
                    sql_query = "error"

            elif result.get("status") == "success":
                sql_query = result["sql_query"]

            predictions[str(qid)] = f"{sql_query}\t----- bird -----\t{db_id}"

        return predictions

    def _save_predictions(self, predictions: Dict[str, str], output_path: str):
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

    def evaluate(self, predictions: Dict[str, str]):
        self._save_predictions(predictions, "./query_results.json")
        run_evaluation(predictions)
