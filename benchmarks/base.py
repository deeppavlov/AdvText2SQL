from pydantic import BaseModel
from typing import Any, Dict, List


class Benchmark(BaseModel):
    db_url: str
    """URL to the server with the databases."""
    config_file: str
    """JSON file with db configs (right now, list of existing db's)."""
    query_file: str
    """Path to the file with user queries."""
    answer_file: str
    """Path to the file with correct answers to the user queries."""

    def build(self, tool_cls: Any) -> Dict[str, Any]:
        """Call the build() method for every tool instance"""

    tools = {}
    for db_subdir in db_dir.iterdir():
        if not db_subdir.is_dir():
            continue

        db_id = db_subdir.name
        sqlite_path = db_dir / db_id / f"{db_id}.sqlite"

        tools[db_id] = tool_cls(
            db_path=str(sqlite_path),
            llm_client=llm_client,
        )

        for tool in self.tool_dict.values():
            tool.build()

    def predict(self, tool_dict: Dict[str, Any]) -> List[str]:
        raise NotImplementedError

    def evaluate(self, predictions: List[str]):
        raise NotImplementedError

    def run(self, tool_cls: Any):
        """Launches everything."""
        return self.evaluate(self.predict(self.build(tool_cls)))
