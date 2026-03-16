import os
import asyncio


from benchmarks.ambrosia import BenchmarkAmbrosia
from benchmarks.evaluate_ambrosia import print_evaluation_report
from adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

db_url = "localhost:5444"

benchmark = BenchmarkAmbrosia(
    db_url=db_url,
    query_file="./data/ambrosia_small.json",
    answer_file="./data/ambrosia_small.json",
)

report = asyncio.run(benchmark.run(Text2SQLGenerator))

# Prints the report in stdout
print_evaluation_report(report)

print("Report on the tokens spent: ", benchmark.llm_client.get_usage())
