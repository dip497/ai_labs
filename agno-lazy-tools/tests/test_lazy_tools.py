"""End-to-end tests: real Agno agents and run loop, driven by an offline scripted model."""

import asyncio
import copy
import sys
from types import SimpleNamespace

import pytest
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.run import RunContext
from agno.run.agent import RunCompletedEvent, ToolCallCompletedEvent
from agno.tools import tool
from agno.tools.calculator import CalculatorTools
from agno.tools.function import Function

from lazy_tools import LazyTool, LazyTools, LazyToolsModelMixin
from lazy_tools.scripted import ScriptedModel, call, say, tool_results


def get_current_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. "Paris".
    """
    return f"Sunny, 21C in {city}"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount of money from one currency to another.

    Args:
        amount: Amount to convert.
        from_currency: ISO code of the source currency, e.g. "USD".
        to_currency: ISO code of the target currency, e.g. "EUR".
    """
    return f"{amount:g} {from_currency} = {amount * 0.9:.2f} {to_currency}"


def create_support_ticket(title: str, priority: str = "normal") -> str:
    """Open a support ticket in the helpdesk."""
    return f"Ticket created: {title} ({priority})"


def current_turn(messages):
    """The messages of the current run (after the latest user message)."""
    last_user = max(i for i, m in enumerate(messages) if m.role == "user")
    return messages[last_user + 1 :]


def use_tool(tool_name, query, **arguments):
    """Policy: search for `tool_name` unless it is loaded, call it, then answer with its result."""

    def policy(messages, tool_names):
        results = tool_results(current_turn(messages))
        if tool_name in results:
            return say(f"Done: {results[tool_name]}")
        if tool_name in tool_names:
            return call(tool_name, **arguments)
        if "search_tools" in results:
            return say("Tool not found")
        return call("search_tools", query=query)

    return policy


def make_agent(lazy, policy, **kwargs):
    return Agent(model=lazy.wrap(ScriptedModel(policy=policy)), tools=[lazy], telemetry=False, **kwargs)


MODES = ["sync", "sync-stream", "async", "async-stream"]


def run(agent, message, mode="sync", **kwargs):
    """Run `agent` through one of Agno's four execution paths; returns (content, tools) of the run."""
    if mode == "sync":
        return agent.run(message, **kwargs)
    if mode == "async":
        return asyncio.run(agent.arun(message, **kwargs))
    if mode == "sync-stream":
        events = list(agent.run(message, stream=True, stream_events=True, **kwargs))
    else:

        async def consume():
            return [e async for e in agent.arun(message, stream=True, stream_events=True, **kwargs)]

        events = asyncio.run(consume())
    # When streaming, tool executions arrive as their own events rather than on the completed run.
    completed = next(e for e in events if isinstance(e, RunCompletedEvent))
    tools = [e.tool for e in events if isinstance(e, ToolCallCompletedEvent)]
    return SimpleNamespace(content=completed.content, tools=tools)


@pytest.mark.parametrize("mode", MODES)
def test_searched_tool_is_loaded_and_called_in_the_same_run(mode):
    lazy = LazyTools(tools=[get_current_weather, convert_currency, create_support_ticket])
    agent = make_agent(lazy, use_tool("get_current_weather", "weather", city="Paris"))

    result = run(agent, "What's the weather in Paris?", mode)

    assert result.content == "Done: Sunny, 21C in Paris"
    assert [t.tool_name for t in result.tools] == ["search_tools", "get_current_weather"]
    # Turn 1 sees only the meta-tool; the found tool is injected from turn 2 on; the others never are.
    assert agent.model.tool_names_per_request() == [
        ["search_tools"],
        ["get_current_weather", "search_tools"],
        ["get_current_weather", "search_tools"],
    ]


def test_injected_schema_is_the_tools_full_schema():
    lazy = LazyTools(tools=[get_current_weather, convert_currency])
    agent = make_agent(lazy, use_tool("get_current_weather", "weather", city="Paris"))

    run(agent, "What's the weather in Paris?")

    injected = next(t for t in agent.model.requests[1] if t["function"]["name"] == "get_current_weather")
    assert injected == {"type": "function", "function": Function.from_callable(get_current_weather).to_dict()}
    assert injected["function"]["parameters"]["required"] == ["city"]


def test_a_tool_that_was_not_loaded_cannot_be_called():
    def call_without_searching(messages, tool_names):
        results = tool_results(messages)
        if "get_current_weather" in results:
            return say(results["get_current_weather"])
        return call("get_current_weather", city="Paris")

    lazy = LazyTools(tools=[get_current_weather])
    agent = make_agent(lazy, call_without_searching)

    result = run(agent, "What's the weather in Paris?")

    assert result.content == "Error: The requested tool does not exist or is not available."
    assert not result.tools


def test_import_path_is_only_imported_when_the_tool_is_loaded():
    sys.modules.pop("tests.heavy_tools", None)
    report = LazyTool("tests.heavy_tools:generate_report", name="generate_report", description="Generate the quarterly sales report")
    lazy = LazyTools(tools=[get_current_weather, report])
    assert "tests.heavy_tools" not in sys.modules

    run(make_agent(lazy, use_tool("get_current_weather", "weather", city="Oslo")), "Weather in Oslo?")
    assert "tests.heavy_tools" not in sys.modules and not report.is_resolved

    result = run(make_agent(lazy, use_tool("generate_report", "sales report", quarter="2026-Q3")), "Q3 sales report?")
    assert "tests.heavy_tools" in sys.modules and report.is_resolved
    assert result.content == "Done: Report for 2026-Q3: revenue up 12%"


def test_import_path_needs_name_and_description():
    with pytest.raises(ValueError, match="needs a name and a description"):
        LazyTool("tests.heavy_tools:generate_report")


def test_concurrent_runs_sharing_one_toolkit_and_model_are_isolated():
    lazy = LazyTools(tools=[get_current_weather, convert_currency])
    policies = {
        "weather": use_tool("get_current_weather", "weather", city="Paris"),
        "currency": use_tool("convert_currency", "currency", amount=100, from_currency="USD", to_currency="EUR"),
    }
    order, seen = [], {"weather": [], "currency": []}

    def policy(messages, tool_names):
        task = next(m.content for m in messages if m.role == "user")
        order.append(task)
        seen[task].append(tool_names)
        return policies[task](messages, tool_names)

    model = lazy.wrap(ScriptedModel(policy=policy))
    weather_agent, currency_agent = (Agent(model=model, tools=[lazy], telemetry=False) for _ in range(2))

    async def both():
        return await asyncio.gather(weather_agent.arun("weather"), currency_agent.arun("currency"))

    weather, currency = asyncio.run(both())

    assert set(order[:2]) == {"weather", "currency"}  # both runs were in flight at once
    assert weather.content == "Done: Sunny, 21C in Paris"
    assert currency.content == "Done: 100 USD = 90.00 EUR"
    assert not any("convert_currency" in names for names in seen["weather"])
    assert not any("get_current_weather" in names for names in seen["currency"])


def test_loaded_tools_stay_loaded_while_they_are_in_the_history():
    lazy = LazyTools(tools=[get_current_weather, convert_currency])
    searching = True

    def policy(messages, tool_names):
        results = tool_results(current_turn(messages))
        if "get_current_weather" in results:
            return say(results["get_current_weather"])
        if "get_current_weather" in tool_names:
            city = [m for m in messages if m.role == "user"][-1].content.split()[-1].rstrip("?")
            return call("get_current_weather", city=city)
        if searching and "search_tools" not in results:
            return call("search_tools", query="weather")
        return say("NOT LOADED")

    agent = make_agent(lazy, policy, db=InMemoryDb(), add_history_to_context=True)
    assert agent.run("Weather in Paris?", session_id="s1").content == "Sunny, 21C in Paris"

    searching = False  # from here on the model never searches
    # Same session: the earlier search is in the history, so the tool is loaded from the first turn.
    assert agent.run("And in Oslo?", session_id="s1").content == "Sunny, 21C in Oslo"
    assert "get_current_weather" in agent.model.tool_names_per_request()[3]
    # A new session has no such history, so nothing is loaded.
    assert agent.run("And in Rome?", session_id="s2").content == "NOT LOADED"
    assert agent.model.tool_names_per_request()[-1] == ["search_tools"]


def test_loaded_tools_get_the_runs_agent_context_and_tool_hooks():
    hook_calls = []

    def audit(agent, function_name, function_call, arguments):
        hook_calls.append((function_name, agent is not None))
        return function_call(**arguments)

    def get_user_plan(run_context: RunContext) -> str:
        """Get the current user's subscription plan."""
        return f"plan={run_context.session_state['plan']}"

    lazy = LazyTools(tools=[get_user_plan])
    agent = make_agent(lazy, use_tool("get_user_plan", "subscription plan"), tool_hooks=[audit])

    result = run(agent, "Which plan am I on?", session_state={"plan": "pro"})

    assert result.content == "Done: plan=pro"
    assert hook_calls == [("search_tools", True), ("get_user_plan", True)]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("decision", ["confirm", "reject"])
def test_a_loaded_tool_that_needs_confirmation_pauses_and_resumes(mode, decision):
    deleted = []

    @tool(requires_confirmation=True)
    def delete_account(user_id: str) -> str:
        """Permanently delete a user account."""
        deleted.append(user_id)
        return f"Deleted {user_id}"

    lazy = LazyTools(tools=[delete_account, get_current_weather])
    agent = make_agent(lazy, use_tool("delete_account", "delete account", user_id="u1"), db=InMemoryDb())

    paused = run(agent, "Delete account u1", mode, session_id="s1")
    assert paused.is_paused and not deleted
    for requirement in paused.active_requirements:
        getattr(requirement, decision)()
    # Resuming dispatches through a table rebuilt from the agent's initial tools (no delete_account).
    resume = agent.continue_run if mode == "sync" else lambda **kw: asyncio.run(agent.acontinue_run(**kw))
    result = resume(run_response=paused, requirements=paused.requirements, session_id="s1")

    assert result.status == "COMPLETED"
    if decision == "confirm":
        assert deleted == ["u1"] and result.content == "Done: Deleted u1"
    else:
        assert deleted == [] and "Deleted" not in result.content


def test_toolkits_and_decorated_functions_can_be_deferred():
    @tool
    def shout(text: str) -> str:
        """Upper-case a piece of text."""
        return text.upper()

    lazy = LazyTools(tools=[CalculatorTools(), shout])
    assert set(lazy.entries) == set(CalculatorTools().functions) | {"shout"}

    result = run(make_agent(lazy, use_tool("multiply", "multiply", a=6, b=7)), "What is 6 x 7?")
    assert result.content == 'Done: {"operation": "multiplication", "result": 42.0}'  # Agno validates a, b as floats

    result = run(make_agent(lazy, use_tool("shout", "upper-case text", text="hi")), "Shout hi")
    assert result.content == "Done: HI"


def test_search_ranking_prefixes_select_and_limits():
    lazy = LazyTools(tools=[get_current_weather, convert_currency, create_support_ticket], max_results=2)

    def names(query):
        return [e.name for e in lazy.search(query)]

    assert names("weather") == ["get_current_weather"]
    assert names("money") == ["convert_currency"]  # description match
    assert names("tickets") == ["create_support_ticket"]  # prefix match
    assert names("helpdesk ticket currency") == ["create_support_ticket", "convert_currency"]  # best first, capped
    assert names("select:create_support_ticket, get_current_weather, nope") == ["create_support_ticket", "get_current_weather"]
    assert names("quantum physics") == []
    assert names("get the tool") == []  # stopwords only


def test_search_drops_matches_much_weaker_than_the_best():
    def get_inventory_levels(product_id: str) -> str:
        """Get current inventory levels for a product."""
        return "12 in stock"

    lazy = LazyTools(tools=[get_current_weather, get_inventory_levels])

    # "current" alone (one description word) is under half the best match's score.
    assert [e.name for e in lazy.search("current weather")] == ["get_current_weather"]
    assert [e.name for e in lazy.search("current")] == ["get_current_weather", "get_inventory_levels"]


def test_wrap_copies_the_model_and_survives_deepcopy():
    lazy = LazyTools(tools=[get_current_weather])
    original = ScriptedModel(policy=use_tool("get_current_weather", "weather", city="Paris"))

    wrapped = lazy.wrap(original)

    assert type(original) is ScriptedModel and not hasattr(original, "lazy_tools")
    assert isinstance(wrapped, ScriptedModel) and isinstance(wrapped, LazyToolsModelMixin)
    assert wrapped.lazy_tools is lazy
    clone = copy.deepcopy(wrapped)  # Agno deep-copies agents and models for request isolation
    assert type(clone) is type(wrapped) and clone.lazy_tools is lazy
