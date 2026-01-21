from fastmcp import FastMCP


server = FastMCP("calculator")


@server.tool()
async def add(a: int, b: int) -> int:
    """Add two integers"""
    return a + b
