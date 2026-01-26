from pydantic import BaseModel
from typing import Any, Dict, List


class Benchmark(BaseModel):
    tool_dict: Dict[str, Any]
    """A dictionary with database id's as keys and tool instances as values.

    All tool classes must have the `build()` and `query()` methods defined.
    """
    query_file: str
    """Path to the file with user queries."""
    answer_file: str
    """Path to the file with correct answers to the user queries."""

    def build(self):
        """Call the build() method for every tool instance"""
        for tool in self.tool_dict.values():
            tool.build()

    def run(self) -> List[str]:
        raise NotImplementedError

    def evaluate(self, predictions: List[str]):
        raise NotImplementedError
