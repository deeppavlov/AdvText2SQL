import asyncio
import json
import logging
from typing import Any, Dict, List

from .base import BenchmarkBase
from .evaluate_bird import run_evaluation
from .response import ToolResponse


logger = logging.getLogger("text2sql_tool")


class BenchmarkBIRD(BenchmarkBase):
    use_evidence: bool = False
    """Dump evidence directly into the `user_query` instead of the prompt
    to help the model. (debatable, consult teacher)"""


    def _load_queries(self) -> List[dict]:
        with open(self.query_file, "r") as f:
            return json.load(f)

    async def predict(self, tool_dict: Dict[str, Any]) -> Dict[str, str]:
        """Returns predictions in the expected format."""
        queries = self._load_queries()
        predictions = {}

        for i, item in enumerate(queries):
            qid = item["question_id"]
            db_id = item["db_id"]
            question = item["question"]
            evidence = item["evidence"]

            tool = tool_dict[db_id]

            if self.use_evidence:
                question = f"question: {question}, evidence (may be empty): {evidence}"

            logger.info(f"[{i+1}/{len(queries)}] Processing q_id={qid} db={db_id}")
            result = await tool.query(question)
            print(result)

            result = ToolResponse.model_validate(result)

            if result.status == "ambiguous":
                sql_query = "ambiguous"
            elif result.status == "success":
                sql_query = result.query
            else:
                sql_query = "error"

            logger.info(f"[{i+1}/{len(queries)}] q_id={qid} -> {result.status}")
            predictions[str(qid)] = sql_query

            # Throttle to avoid rate limiting
            await asyncio.sleep(1.0)

        return predictions

    def _save_predictions(self, predictions: Dict[str, str], output_path: str):
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

    async def evaluate(self, predictions: Dict[str, str]) -> dict:
        self._save_predictions(predictions, "./query_results.json")
        report = run_evaluation(predictions, self.answer_file, self.db_url)

        return report
