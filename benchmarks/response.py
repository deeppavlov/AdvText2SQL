from pydantic import BaseModel
from typing import Optional


class ToolResponse(BaseModel):
    status: str
    """Execution status. Can be either "success", "ambiguous" or "error"."""
    query: Optional[str] = None
    """The generated query. Only used if status is "success"."""

"""
So, this dictionary fits:
{
    "status": "ambiguous",
}

Or this:
{
    "status": "error",
    "query": None
}
"""