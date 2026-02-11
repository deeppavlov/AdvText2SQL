import os
import asyncio


from benchmarks.bird import BenchmarkBIRD
from benchmarks.evaluate import print_evaluation_report
from partial_mcp.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

db_url = os.getenv("BENCHMARK_DB_URL")

benchmark = BenchmarkBIRD(
    db_url=db_url,
    query_file="./data/some_queries.json",
    answer_file="./data/some_queries.json",
    use_evidence=True,
)

report = asyncio.run(benchmark.run(Text2SQLGenerator))

# Prints the report in stdout
print_evaluation_report(report)

print("Report on the tokens spent: ", benchmark.llm_client.get_usage())
