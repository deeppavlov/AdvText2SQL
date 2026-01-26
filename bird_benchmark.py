import os
from pathlib import Path

from autogen_ext.models.openai import OpenAIChatCompletionClient
from benchmarks.bird import BenchmarkBIRD
from partial_mcp.mcp_servers.text2sql_tool.src.text2sql_implementation import (
    Text2SQLGenerator,
)

llm_client = OpenAIChatCompletionClient(
    model=os.environ["LLM_MODEL_NAME"],
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
    temperature=0.6,
    model_info={
        "json_output": False,
        "function_calling": True,
        "vision": False,
        "family": "unknown",
        "structured_output": False,
    },
)

tools = {}
db_root = Path("./data")

for db_dir in db_root.iterdir():
    if not db_dir.is_dir():
        continue

    db_id = db_dir.name
    sqlite_path = db_dir / f"{db_id}.sqlite"

    tools[db_id] = Text2SQLGenerator(
        db_path=str(sqlite_path),
        llm_client=llm_client,
    )

benchmark = BenchmarkBIRD(
    tool_dict=tools,
    query_file="./data/dev.json",
    answer_file="./data/dev_gold.sql",
)
benchmark.build()

predictions = benchmark.run()

benchmark.evaluate(predictions)
