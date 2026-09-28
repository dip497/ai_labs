"""
Lazy Tool Loading with OpenAI Tool Search
=========================================

Demonstrates loading tools on demand with OpenAI's hosted tool search.

The agent has 28 tools, but each one is sent with `defer_loading: true`: the model only
sees their names and descriptions, searches for the tools it needs, and OpenAI appends
their full schemas to the end of the context. The `tools` list never changes, so the
prompt cache keeps hitting.

Needs the Responses API and gpt-5.4 or later.
Docs: https://developers.openai.com/api/docs/guides/tools-tool-search
"""

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.calculator import CalculatorTools
from agno.tools.email import EmailTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.pubmed import PubmedTools
from agno.tools.shopify import ShopifyTools
from agno.tools.website import WebsiteTools


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. "Paris".
    """
    return f"{city}: 21C and sunny"


# ---------------------------------------------------------------------------
# Model: Agno 3.0.11 doesn't send `defer_loading`, so this subclass adds it
# ---------------------------------------------------------------------------
class ToolSearchResponses(OpenAIResponses):
    def _format_tool_params(self, messages, tools=None):
        return [
            {**tool, "defer_loading": True} if tool.get("type") == "function" else tool
            for tool in super()._format_tool_params(messages, tools)
        ]

    def _using_reasoning_model(self) -> bool:
        # Chain requests with previous_response_id, so OpenAI keeps the tool search
        # results between requests. Agno does that only for o3, o4-mini and gpt-5* ids.
        return True


# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
tools = [CalculatorTools(), HackerNewsTools(), ShopifyTools(), EmailTools(), PubmedTools(), WebsiteTools(), get_weather]

agent = Agent(
    model=ToolSearchResponses(id="gpt-6-luna"),
    tools=[{"type": "tool_search"}, *tools],
    markdown=True,
    telemetry=False,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
QUESTION = "What's the weather in Paris, and what is 1234 * 5678?"

if __name__ == "__main__":
    agent.print_response(QUESTION, stream=True)
