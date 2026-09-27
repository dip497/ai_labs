"""Lazy tool loading, end to end.

An agent with a 35-tool catalog (7 real Agno toolkits plus a few of our own) answers a
question with only `search_tools` attached; the tools it finds are loaded into the same
run. The same question then goes to an agent with every tool attached, and the tool
schemas the two sent to the model are compared. With an OpenAI model, a third agent
loads its tools through OpenAI's native tool search (`lazy.wrap(model, native=True)`).

    uv run python examples/demo.py                                        # offline scripted model
    uv run --extra openai python examples/demo.py --model openai:gpt-6-luna --padding-tokens 2000
    uv run --extra anthropic python examples/demo.py --model anthropic:claude-opus-5
"""

from __future__ import annotations

import argparse
import functools
import logging
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from any directory

from agno.agent import Agent
from agno.models.base import Model
from agno.models.message import Message
from agno.run.agent import RunOutput
from agno.tools.airflow import AirflowTools
from agno.tools.calculator import CalculatorTools
from agno.tools.email import EmailTools
from agno.tools.function import Function
from agno.tools.hackernews import HackerNewsTools
from agno.tools.pubmed import PubmedTools
from agno.tools.shopify import ShopifyTools
from agno.tools.website import WebsiteTools
from agno.utils.tokens import count_text_tokens, count_tool_tokens

from lazy_tools import LazyTool, LazyTools
from lazy_tools.scripted import ScriptedModel, calls, say, tool_results

QUESTION = "What's the weather in Paris right now, and how much is 250 USD in euros?"


def get_current_weather(city: str) -> str:
    """Get the current weather conditions for a city.

    Args:
        city: City name, e.g. "Paris".
    """
    return f"{city}: 21C and sunny"


def get_weather_forecast(city: str, days: int = 3) -> str:
    """Get a daily weather forecast for a city.

    Args:
        city: City name, e.g. "Paris".
        days: Number of days to forecast, 1 to 7.
    """
    return f"{city}: sunny for the next {days} days"


def search_flights(origin: str, destination: str, date: str) -> str:
    """Search for flights between two airports.

    Args:
        origin: IATA code of the departure airport, e.g. "JFK".
        destination: IATA code of the arrival airport, e.g. "CDG".
        date: Departure date as YYYY-MM-DD.
    """
    return f"3 flights {origin} -> {destination} on {date}, from $420"


def search_hotels(city: str, check_in: str, check_out: str, guests: int = 1) -> str:
    """Search for available hotel rooms in a city.

    Args:
        city: City name.
        check_in: Check-in date as YYYY-MM-DD.
        check_out: Check-out date as YYYY-MM-DD.
        guests: Number of guests.
    """
    return f"12 hotels in {city} for {guests} guest(s), from $180/night"


def build_catalog() -> LazyTools:
    return LazyTools(
        tools=[
            # Real Agno toolkits: each of their functions becomes a deferred tool.
            CalculatorTools(),
            ShopifyTools(),
            HackerNewsTools(),
            AirflowTools(),
            EmailTools(),
            PubmedTools(),
            WebsiteTools(),
            # Our own tools.
            get_current_weather,
            get_weather_forecast,
            search_flights,
            search_hotels,
            # Registered by import path: examples.fx is imported only if one of these is loaded.
            LazyTool(
                "examples.fx:convert_currency",
                name="convert_currency",
                description="Convert an amount of money from one currency to another at today's rate",
            ),
            LazyTool(
                "examples.fx:get_exchange_rate",
                name="get_exchange_rate",
                description="Get today's exchange rate between two currencies",
            ),
        ],
        max_results=3,
    )


# What the offline scripted model does for QUESTION: (tool, search query, arguments).
PLAN = [
    ("get_current_weather", "current weather", {"city": "Paris"}),
    ("convert_currency", "convert currency", {"amount": 250, "from_currency": "USD", "to_currency": "EUR"}),
]


def scripted_policy(messages: List[Message], tool_names: List[str]):
    """Call each planned tool as soon as it is available, searching for it first if it is not."""
    results = tool_results(messages)
    todo = [step for step in PLAN if step[0] not in results]
    if not todo:
        return say(" / ".join(results[name] for name, _, _ in PLAN))
    ready = [(name, arguments) for name, _, arguments in todo if name in tool_names]
    if ready:
        return calls(*ready)
    if "search_tools" in results:
        return say("I could not find the tools I need.")
    return calls(*[("search_tools", {"query": query}) for _, query, _ in todo])


def make_model(spec: str) -> Model:
    if spec == "offline":
        return ScriptedModel(policy=scripted_policy)
    provider, _, model_id = spec.partition(":")
    if provider == "openai":
        # The Responses API has native tool search, and gpt-6-luna calls functions over Chat
        # Completions only with reasoning_effort="none". Agno 3.0.11 doesn't count gpt-6 models
        # as reasoning models, so by default it drops their reasoning items between requests;
        # with store=False and the encrypted reasoning included, it replays them.
        return openai_model_class()(id=model_id or "gpt-6-luna", store=False, include=["reasoning.encrypted_content"])
    if provider == "anthropic":
        from agno.models.anthropic import Claude

        return Claude(id=model_id or "claude-opus-5")
    raise SystemExit(f"Unknown --model {spec!r}: use offline, openai:<model id> or anthropic:<model id>")


def padding(tokens: int) -> Optional[str]:
    """About `tokens` tokens of filler, standing in for a production agent's long system prompt."""
    if tokens <= 0:
        return None
    line = "This line pads the system prompt, as the long instructions of a production agent would; ignore it."
    return "\n".join([line] * max(1, round(tokens / count_text_tokens(line))))


@functools.lru_cache(maxsize=None)
def openai_model_class() -> type:
    from agno.models.openai import OpenAIResponses  # needs the openai extra

    class Responses(OpenAIResponses):
        def _get_metrics(self, response_usage: Any) -> Any:
            # Agno 3.0.11 records OpenAI's cache reads but not its cache writes (billed at 1.25x on GPT-5.6+).
            metrics = super()._get_metrics(response_usage)
            metrics.cache_write_tokens = getattr(response_usage.input_tokens_details, "cache_write_tokens", None) or 0
            return metrics

    return Responses


def input_cost(metrics: Any) -> float:
    """Input cost in uncached-token units at GPT-5.6+ cache rates: reads 0.1x, writes 1.25x."""
    uncached = metrics.input_tokens - metrics.cache_read_tokens - metrics.cache_write_tokens
    return uncached + 0.1 * metrics.cache_read_tokens + 1.25 * metrics.cache_write_tokens


def report(run: RunOutput, tools_for_request: Callable[[List[Message]], List[Function]], live: bool) -> Tuple[int, int]:
    """Print each model request of `run`; returns (number of requests, total tool-schema tokens).

    Each assistant message is one model request; `tools_for_request` gets the messages
    that preceded it and returns the tools the model saw on that request.
    """
    messages = run.messages or []
    requests = [i for i, m in enumerate(messages) if m.role == "assistant"]
    total = 0
    for number, index in enumerate(requests, 1):
        tools = tools_for_request(messages[:index])
        tokens = count_tool_tokens(tools)
        total += tokens
        names = ", ".join(t.name for t in tools) if len(tools) <= 6 else "the whole catalog"
        made = ", ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in messages[index].tool_calls or [])
        line = f"  request {number}: {len(tools):>2} tool{'s' if len(tools) > 1 else ' '}, ~{tokens:>5,} schema tokens [{names}]"
        metrics = messages[index].metrics
        if live and metrics is not None:
            line += f"; provider: {metrics.input_tokens:,} input tokens ({metrics.cache_read_tokens:,} cached, {metrics.cache_write_tokens:,} written)"
        print(line)
        print(f"             -> {made or 'final answer'}")
    print(f"  answer: {run.content}")
    return len(requests), total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="offline", help='"offline" (default), "openai:<model id>" or "anthropic:<model id>"')
    parser.add_argument("--question", default=QUESTION, help="needs a live --model: the offline model is scripted for the default question")
    parser.add_argument(
        "--padding-tokens",
        type=int,
        default=0,
        metavar="N",
        help="add about N tokens of filler to each agent's system prompt, standing in for a production agent's "
        "long instructions: OpenAI only caches prompts of 1,024 tokens or more",
    )
    args = parser.parse_args()
    if args.model == "offline" and args.question != QUESTION:
        parser.error("--question needs a live --model: the offline model is scripted for the default question")
    # Quiets warnings such as Agno's "tiktoken not installed" (token counts then use chars / 4); errors still show.
    logging.disable(logging.WARNING)
    live = args.model != "offline"

    def run(model: Model, tools: list) -> RunOutput:
        return Agent(model=model, tools=tools, additional_context=padding(args.padding_tokens), telemetry=False).run(args.question)

    lazy = build_catalog()
    catalog = list(lazy.entries.values())
    print(f"Catalog: {len(catalog)} deferred tools; examples.fx imported: {'examples.fx' in sys.modules}")
    print(f"Q: {args.question}\n")

    print("Lazy agent (only search_tools attached):")
    search_tools = Function.from_callable(lazy.search_tools)

    def loaded(before: List[Message]) -> List[Function]:
        return [search_tools] + [e.template for e in lazy.loaded(before)]

    runs = {"lazy": run(lazy.wrap(make_model(args.model)), [lazy])}
    lazy_requests, lazy_schemas = report(runs["lazy"], loaded, live)
    print(f"  examples.fx imported: {'examples.fx' in sys.modules}\n")

    if args.model.partition(":")[0] == "openai":
        print("Lazy agent, native OpenAI tool search (`tools` stays [search_tools]; the loaded tools go in the input):")
        runs["native"] = run(lazy.wrap(make_model(args.model), native=True), [lazy])
        report(runs["native"], loaded, live)
        print()

    all_tools = [entry.template for entry in catalog]
    print(f"Eager agent (all {len(all_tools)} tools attached):")
    runs["eager"] = run(make_model(args.model), all_tools)
    eager_requests, eager_schemas = report(runs["eager"], lambda before: all_tools, live)

    # The lazy agent's instructions (which list the loadable tool names) ride along on every request.
    instructions = count_text_tokens(lazy.instructions or "") * lazy_requests
    lazy_total = lazy_schemas + instructions
    print(
        f"\nTools in context: lazy ~{lazy_total:,} tokens over {lazy_requests} requests "
        f"(~{lazy_schemas:,} schemas + ~{instructions:,} instructions) vs eager ~{eager_schemas:,} over {eager_requests} "
        f"-> {1 - lazy_total / eager_schemas:.0%} less."
    )
    if live:
        print("Provider-reported input tokens for the whole run (cost in uncached-token units, at GPT-5.6+ cache rates):")
        for name, r in runs.items():
            m = r.metrics
            if m is not None:
                print(f"  {name:>6}: {m.input_tokens:,} ({m.cache_read_tokens:,} cached, {m.cache_write_tokens:,} written) -> cost ~{input_cost(m):,.0f}")


if __name__ == "__main__":
    main()
