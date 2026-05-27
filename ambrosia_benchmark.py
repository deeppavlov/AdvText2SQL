import os
import asyncio
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from benchmarks.ambrosia import BenchmarkAmbrosia
from benchmarks.evaluate_ambrosia import print_evaluation_report
from adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

db_url = "localhost:5444"

_size = os.environ.get("DATASET_SIZE", "large")
_dataset = f"./data/ambrosia_{_size}.json"

benchmark = BenchmarkAmbrosia(
    db_url=db_url,
    query_file=_dataset,
    answer_file=_dataset,
)

report = asyncio.run(benchmark.run(Text2SQLGenerator))

# Prints the report in stdout
print_evaluation_report(report)

print("Report on the tokens spent: ", benchmark.llm_client.get_usage())
