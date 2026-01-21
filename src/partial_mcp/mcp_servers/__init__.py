from typing import Any

from fastmcp import FastMCP

from .calculator import server as calculator


servers: list[FastMCP[Any]] = [calculator]
