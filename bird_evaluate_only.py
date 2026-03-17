import os

from benchmarks.evaluate_bird import run_evaluation, print_evaluation_report

db_url = os.getenv("BENCHMARK_DB_URL")

def _load_queries(self) -> List[dict]:
    with open("./query_results", "r") as f:
        return json.load(f)

predictions = _load_queries()

report = run_evaluation(predictions, "./data/bird_small.json", db_url)

print_evaluation_report(report)
