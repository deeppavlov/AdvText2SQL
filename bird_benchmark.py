import os
import asyncio


from benchmarks.bird import BenchmarkBIRD
from benchmarks.evaluate_bird import print_evaluation_report
from adv_text2sql.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

db_url = "localhost:5444"

benchmark = BenchmarkBIRD(
    db_url=db_url,
    query_file="./data/bird_small.json",
    answer_file="./data/bird_small.json",
    use_evidence=True,
)

report = asyncio.run(benchmark.run(Text2SQLGenerator))

# Prints the report in stdout
print_evaluation_report(report)

print("Report on the tokens spent: ", benchmark.llm_client.get_usage())
