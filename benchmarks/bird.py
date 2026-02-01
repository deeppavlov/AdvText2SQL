import json
from typing import Dict, List

from .base import Benchmark
from .evaluation import run_evaluation


class BenchmarkBIRD(Benchmark):
    def _load_queries(self) -> List[dict]:
        with open(self.query_file, "r") as f:
            return json.load(f)

    def run(self, Text2SQL) -> Dict[str, str]:
        """Returns predictions in BIRD's expected format."""
        queries = self._load_queries()
        predictions: Dict[str, str] = {}

        for item in queries:
            qid = item["question_id"]
            db_id = item["db_id"]
            question = item["question"]

            tool = self.tool_dict[db_id]
            sql = tool.query(question)

            predictions[str(qid)] = f"{sql}\t----- bird -----\t{db_id}"

        return predictions

    def _save_predictions(self, predictions: Dict[str, str], output_path: str):
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

    # TODO: connect BIRD's evaluation.py programmatically (could even change it)
    def evaluate(self, predictions: List[str]):
        self._save_predictions(predictions, "./predict_dev.json")
        run_evaluation()
