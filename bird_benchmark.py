import os
from pathlib import Path

from autogen_ext.models.openai import OpenAIChatCompletionClient

from benchmarks.bird import BenchmarkBIRD
from benchmarks.evaluate import print_evaluation_report
from partial_mcp.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

# Place this into env, remove 
db_url = os.getenv("BENCHMARK_DB_URL")

benchmark = BenchmarkBIRD(
    db_url=db_url,
    query_file="./data/dev.json",
    answer_file="./data/dev_gold.sql",
)

report = benchmark.run(Text2SQLGenerator)

# Prints the report in stdout
print_evaluation_report(report)
