import os
import asyncio


from benchmarks.bird import BenchmarkBIRD
from benchmarks.evaluate_bird import print_evaluation_report
from adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
    setup_logging,
)

setup_logging()

db_url = "localhost:5444"

_size = os.environ.get("DATASET_SIZE", "large")
_dataset = f"./data/bird_{_size}.json"

benchmark = BenchmarkBIRD(
    db_url=db_url,
    query_file=_dataset,
    answer_file=_dataset,
    use_evidence=True,
)

report = asyncio.run(benchmark.run(Text2SQLGenerator))

# Prints the report in stdout
print_evaluation_report(report)

print("Report on the tokens spent: ", benchmark.llm_client.get_usage())
