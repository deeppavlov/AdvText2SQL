from typing import Any

from fastmcp import FastMCP

from .calculator import server as calculator
from .text2sql_tool.main import server as text2sql_tool


servers: list[FastMCP[Any]] = [calculator, text2sql_tool]
