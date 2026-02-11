import os
import json

from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel
from typing import Any, Dict, Optional

from .token_tracking import TokenTrackingClient


class Benchmark(BaseModel):
    db_url: str
    """URL to the server with the databases. (without the "https://" part)"""
    query_file: str
    """Path to the file with user queries."""
    answer_file: str
    """Path to the file with correct answers to the user queries."""
    llm_client: Optional[Any] = None
    """OpenAI LLM client used to create the Text2SQL tools."""

    async def build(self, tool_cls: Any) -> Dict[str, Any]:
        """Create every tool instance then call every tools' build() method."""

        db_username = os.environ["DB_USER"]
        db_password = os.environ["DB_PASS"]

        with open(self.query_file, "r") as f:
            data = json.load(f)
        dbs = [item["db_id"] for item in data]
        db_ids = list(set(dbs))

        # TODO: move to a better location
        if not self.llm_client:
            self.llm_client = OpenAIChatCompletionClient(
                model=os.environ["LLM_MODEL_NAME"],
                base_url=os.environ["LLM_BASE_URL"],
                api_key=os.environ["LLM_API_KEY"],
                temperature=0,
                model_info={
                    "json_output": False,
                    "function_calling": True,
                    "vision": False,
                    "family": "unknown",
                    "structured_output": False,
                },
            )
        self.llm_client = TokenTrackingClient(self.llm_client)

        # Building every tool
        tools = {}
        for db_id in db_ids:
            db_uri = f"postgresql+psycopg://{db_username}:{db_password}@{self.db_url}/{db_id}"

            tools[db_id] = tool_cls(
                db_uri=db_uri,
                llm_client=self.llm_client,
            )

        for tool in tools.values():
            tool.build()

        # Saving database schemas for debugging 
        self.dump_db_schemas_json(tools)

        return tools

    async def predict(self, tool_dict: Dict[str, Any]) -> Dict[str, str]:
        """Predictions are returned as a dictionary with question_id's as keys
        and sql queries as values."""
        raise NotImplementedError

    async def evaluate(self, predictions: Dict[str, str]):
        raise NotImplementedError

    async def run(self, tool_cls: Any):
        """Launches everything. Takes in the tool's class."""
        return await self.evaluate(await self.predict(await self.build(tool_cls)))

    def dump_db_schemas_json(self, tool_dict: Dict[str, Any], output_path: str = "db_schemas.json"):
        schemas = {
            db_id: tool.db_schema
            for db_id, tool in tool_dict.items()
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schemas, f, ensure_ascii=False, indent=2)
