import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

from .mcp_servers import servers

toolsets = [FastMCPToolset(server) for server in servers]

model = OpenAIChatModel(
    model_name=os.environ["MODEL_NAME"],
    provider=OpenAIProvider(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["BASE_URL"],
    ),
)

agent = Agent(model=model, toolsets=toolsets)


async def chat(request: str) -> str:
    result = await agent.run(request)
    return result.output
