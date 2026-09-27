"""Native OpenAI tool search: Agno's real `OpenAIResponses` adapter against a local mock Responses API.

The mock follows OpenAI's client-executed tool search protocol
(https://developers.openai.com/api/docs/guides/tools-tool-search): its scripted model
asks for tools with a `tool_search_call`, and the tools listed in a `tool_search_output`
become callable. Like the real API, it rejects (HTTP 400) request bodies that don't
match the request types of the `openai` SDK, which are generated from OpenAI's API spec,
and outputs that answer no call.
"""

import json
import logging
import sys
import threading
import typing
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.openai import OpenAIResponses
from agno.tools import tool
from agno.tools.function import Function
from openai.types.responses import ResponseInputItemParam, ToolParam
from pydantic import TypeAdapter, ValidationError

from lazy_tools import LazyTools
from lazy_tools.scripted import ScriptedModel
from tests.test_lazy_tools import MODES, convert_currency, get_current_weather, run

# The spec check: OpenAI's request types, by the value of their `type` field.


def _literals(annotation: typing.Any) -> typing.List[str]:
    if typing.get_origin(annotation) is typing.Literal:
        return list(typing.get_args(annotation))
    return [value for arg in typing.get_args(annotation) for value in _literals(arg)]


def _by_type(union: typing.Any) -> typing.Dict[str, type]:
    by_type: typing.Dict[str, type] = {}
    for member in typing.get_args(union):
        for value in _literals(typing.get_type_hints(member).get("type")):
            by_type.setdefault(value, member)  # "message": the first is the easy input message
    return by_type


INPUT_ITEM_TYPES, TOOL_TYPES = _by_type(ResponseInputItemParam), _by_type(ToolParam)


@lru_cache(maxsize=None)
def _adapter(typed_dict: type) -> TypeAdapter:
    return TypeAdapter(typed_dict)


# Required by the spec, so by the SDK's types, but optional in practice: OpenAI's own tool
# search examples leave `strict` out ("if omitted, Responses attempts to use strict
# validation when the schema is compatible"), and so do Agno's function tools.
REQUIRED_ONLY_BY_THE_SPEC = {"function": {"strict": None}}


# Stock Agno 3.0.11 sends these with every toolkit function in `tools` (it strips them only
# for some OpenAI-compatible providers), so they are tolerated there; not in the loaded
# definitions this POC writes into `tool_search_output` items.
AGNO_TOOL_FIELDS = {"requires_confirmation", "external_execution", "approval_type"}


def spec_problems(value: dict, types: typing.Dict[str, type], where: str, tolerated=frozenset()) -> typing.List[str]:
    kind = value.get("type", "message")
    typed_dict = types.get(kind)
    if typed_dict is None:
        return [f"{where}: unknown type {kind!r}"]
    value = {key: field for key, field in value.items() if key not in tolerated}
    problems = [f"{where}: unknown field {key!r}" for key in value if key not in typing.get_type_hints(typed_dict)]
    try:
        _adapter(typed_dict).validate_python({**REQUIRED_ONLY_BY_THE_SPEC.get(kind, {}), **value})
    except ValidationError as error:
        problems.append(f"{where}: {error}")
    return problems


ANSWERS = {"function_call_output": "function_call", "tool_search_output": "tool_search_call"}


def request_problems(body: dict, context: typing.List[dict]) -> typing.List[str]:
    """Why the API would reject `body`, whose whole conversation (after chaining) is `context`."""
    problems = []
    for i, tool in enumerate(body.get("tools", [])):
        problems += spec_problems(tool, TOOL_TYPES, f"tools[{i}]", AGNO_TOOL_FIELDS)
    for i, item in enumerate(body["input"]):
        problems += spec_problems(item, INPUT_ITEM_TYPES, f"input[{i}]")
        for j, loaded in enumerate(item.get("tools", []) if item.get("type") == "tool_search_output" else []):
            problems += spec_problems(loaded, TOOL_TYPES, f"input[{i}].tools[{j}]")
    calls = {}
    for item in context:
        kind = item.get("type")
        if kind in ANSWERS.values():
            calls[item["call_id"]] = kind
        elif kind in ANSWERS and calls.get(item.get("call_id")) != ANSWERS[kind]:
            problems.append(f"{kind} {item.get('call_id')!r} answers no {ANSWERS[kind]}")
    return problems


# The mock model and its API.


def message(n, text):
    content = [{"type": "output_text", "text": text, "annotations": []}]
    return {"type": "message", "id": f"msg_{n}", "role": "assistant", "status": "completed", "content": content}


def function_call(n, name, arguments, namespace=None):
    item = {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "name": name, "arguments": json.dumps(arguments)}
    return {**item, "status": "completed", **({"namespace": namespace} if namespace else {})}


def tool_search_call(n, query):
    item = {"type": "tool_search_call", "id": f"tsc_{n}", "call_id": f"call_{n}", "execution": "client"}
    return {**item, "status": "completed", "arguments": {"query": query}}


def searched_tools(context):
    """Tools loaded by the `tool_search_output` items in `context`."""
    return [t["name"] for item in context if item.get("type") == "tool_search_output" for t in item["tools"]]


def search(n, tools, query):
    """A search, through the client tool search tool if the request has one (native mode), else `search_tools`."""
    if any(t["type"] == "tool_search" for t in tools):
        return tool_search_call(n, query)
    return function_call(n, "search_tools", {"query": query})


def use_tools(*plan, searching=True):
    """Policy: search for the planned tools that aren't loaded, call them, then answer with their results.

    `plan` holds (tool name, search query, arguments) steps; searches and calls go out in parallel.
    """

    def policy(tools, context, n):
        turn = context[max(i for i, item in enumerate(context) if item.get("role") == "user") + 1 :]
        called = {item["call_id"]: item["name"] for item in context if item.get("type") == "function_call"}
        results = {called[i["call_id"]]: i["output"] for i in turn if i.get("type") == "function_call_output"}
        todo = [step for step in plan if step[0] not in results]
        if not todo:
            return [message(n, "Done: " + " / ".join(results[name] for name, _, _ in plan))]
        searched, declared = searched_tools(context), [t.get("name") for t in tools]
        ready = [(name, arguments) for name, _, arguments in todo if name in searched or name in declared]
        if ready:
            # A call to a searched tool carries its namespace, as in OpenAI's docs.
            return [function_call(f"{n}_{i}", name, args, namespace=name if name in searched else None) for i, (name, args) in enumerate(ready)]
        if not searching:
            return [message(n, "NOT LOADED")]
        if any(item.get("type") in ("tool_search_call", "function_call") for item in turn):
            return [message(n, "Tool not found")]
        return [search(f"{n}_{i}", tools, query) for i, (_, query, _) in enumerate(todo)]

    return policy


class MockResponsesAPI:
    """A local Responses API with a scripted model (`policy`), recording each request body."""

    def __init__(self):
        self.policy = use_tools(("get_current_weather", "weather", {"city": "Paris"}))
        self.requests: typing.List[dict] = []
        self.conversations: typing.Dict[str, typing.List[dict]] = {}  # response id -> input and output items

    def respond(self, body):
        """(HTTP status, JSON body) for a `POST /responses` request."""
        self.requests.append(body)
        n = len(self.requests)
        previous = body.get("previous_response_id")
        context = self.conversations.get(previous, []) + body["input"]
        problems = request_problems(body, context)
        if previous and previous not in self.conversations:
            problems.append(f"unknown previous_response_id {previous!r}")
        if problems:
            return 400, {"error": {"message": "; ".join(problems), "type": "invalid_request_error", "param": None, "code": None}}
        # A policy that never sees what it waits for would loop forever: end the run instead.
        output = self.policy(body.get("tools", []), context, n) if n <= 10 else [message(n, "Too many requests")]
        self.conversations[f"resp_{n}"] = context + output
        usage = {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 15,
        }
        return 200, {
            "id": f"resp_{n}", "object": "response", "created_at": 0, "model": body["model"], "status": "completed",
            "output": output, "error": None, "incomplete_details": None, "usage": usage,
        }


def stream_events(response):
    """The server-sent events that stream `response` (abridged to what clients read)."""
    events = [{"type": "response.created", "response": {**response, "status": "in_progress", "output": []}}]
    for index, item in enumerate(response["output"]):
        started = {**item, "status": "in_progress"}
        if item["type"] == "function_call":
            started["arguments"] = ""
        elif item["type"] == "tool_search_call":
            started["arguments"] = {}  # tool search calls have no argument deltas: they come with the finished item
        elif item["type"] == "message":
            started["content"] = []
        events.append({"type": "response.output_item.added", "output_index": index, "item": started})
        if item["type"] == "function_call":
            events.append({"type": "response.function_call_arguments.delta", "item_id": item["id"], "output_index": index, "delta": item["arguments"]})
        elif item["type"] == "message":
            text = item["content"][0]["text"]
            events.append({"type": "response.output_text.delta", "item_id": item["id"], "output_index": index, "content_index": 0, "delta": text, "logprobs": []})
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    events.append({"type": "response.completed", "response": response})
    return "".join(f"event: {e['type']}\ndata: {json.dumps({**e, 'sequence_number': i})}\n\n" for i, e in enumerate(events))


@pytest.fixture
def api():
    mock = MockResponsesAPI()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            status, response = mock.respond(body)
            streaming = status == 200 and body.get("stream")
            payload = (stream_events(response) if streaming else json.dumps(response)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    mock.url = f"http://127.0.0.1:{server.server_port}/v1"
    yield mock
    server.shutdown()


def make_agent(api, lazy, model_id="gpt-6-luna", native=True, **kwargs):
    model = OpenAIResponses(id=model_id, api_key="test", base_url=api.url)
    return Agent(model=lazy.wrap(model, native=native), tools=[lazy], telemetry=False, **kwargs)


def deferred(function):
    """A tool's definition as a `tool_search_output` carries it."""
    return {"type": "function", **Function.from_callable(function).to_dict(), "defer_loading": True}


TOOL_SEARCH = {
    "type": "tool_search",
    "execution": "client",
    "description": "Find tools in the catalog and load them. Loaded tools can be called directly on your next step.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": 'Keywords for the capability you need (e.g. "weather forecast"), or "select:name1,name2" to load tools by exact name.',
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


@pytest.mark.parametrize("mode", MODES)
def test_tool_search_loads_the_tool_without_changing_tools(api, mode):
    lazy = LazyTools(tools=[get_current_weather, convert_currency])
    agent = make_agent(api, lazy)  # Agno replays the whole conversation on each request for gpt-6-luna

    result = run(agent, "What's the weather in Paris?", mode)

    assert result.content == "Done: Sunny, 21C in Paris"
    assert [t.tool_name for t in result.tools] == ["search_tools", "get_current_weather"]
    first, second, third = api.requests
    # Every request has the same `tools`: only the tool search tool.
    assert first["tools"] == second["tools"] == third["tools"] == [TOOL_SEARCH]
    # The found tool arrives in the input instead, answering the model's search.
    assert second["input"][-2:] == [
        {"type": "tool_search_call", "execution": "client", "call_id": "call_1_0", "status": "completed", "arguments": {"query": "weather"}},
        {"type": "tool_search_output", "execution": "client", "call_id": "call_1_0", "status": "completed", "tools": [deferred(get_current_weather)]},
    ]
    # So each request extends the previous one: a stable, cacheable prefix.
    assert second["input"][: len(first["input"])] == first["input"]
    assert third["input"][: len(second["input"])] == second["input"]
    # The call to the loaded tool goes back with the namespace the model gave it.
    assert third["input"][-2]["type"] == "function_call" and third["input"][-2]["namespace"] == "get_current_weather"


def test_tool_search_with_previous_response_id(api):
    lazy = LazyTools(tools=[get_current_weather, convert_currency])
    agent = make_agent(api, lazy, model_id="gpt-5.4-mini")  # Agno chains responses for gpt-5 models

    result = agent.run("What's the weather in Paris?")

    assert result.content == "Done: Sunny, 21C in Paris"
    first, second, third = api.requests
    assert [r.get("previous_response_id") for r in api.requests] == [None, "resp_1", "resp_2"]
    assert first["tools"] == second["tools"] == third["tools"] == [TOOL_SEARCH]
    # Only the new items are sent: the search's output, then the weather tool's.
    assert second["input"] == [
        {"type": "tool_search_output", "execution": "client", "call_id": "call_1_0", "status": "completed", "tools": [deferred(get_current_weather)]}
    ]
    assert [item["type"] for item in third["input"]] == ["function_call_output"]


def test_without_native_the_tools_change_on_every_load(api):
    # The default mode on the same mock: it works, but each load changes `tools`, and so the prompt prefix.
    lazy = LazyTools(tools=[get_current_weather, convert_currency])
    agent = make_agent(api, lazy, native=False)

    result = agent.run("What's the weather in Paris?")

    assert result.content == "Done: Sunny, 21C in Paris"
    assert [[t["name"] for t in r["tools"]] for r in api.requests] == [
        ["search_tools"],
        ["get_current_weather", "search_tools"],
        ["get_current_weather", "search_tools"],
    ]


def test_each_tool_is_sent_once_by_the_search_that_loaded_it(api):
    def search_twice(tools, context, n):
        searches = [item for item in context if item.get("type") == "tool_search_call"]
        if len(searches) == 0:
            return [tool_search_call(n, "weather")]
        if len(searches) == 1:
            return [tool_search_call(n, "select:get_current_weather,convert_currency")]
        return [message(n, "Loaded")]

    api.policy = search_twice
    lazy = LazyTools(tools=[get_current_weather, convert_currency])

    make_agent(api, lazy).run("Load the weather and currency tools")

    outputs = [item for item in api.requests[-1]["input"] if item.get("type") == "tool_search_output"]
    assert [[t["name"] for t in output["tools"]] for output in outputs] == [["get_current_weather"], ["convert_currency"]]


def test_tools_stay_loaded_while_the_search_is_in_the_history(api):
    lazy = LazyTools(tools=[get_current_weather, convert_currency])
    agent = make_agent(api, lazy, db=InMemoryDb(), add_history_to_context=True)
    assert agent.run("Weather in Paris?", session_id="s1").content == "Done: Sunny, 21C in Paris"

    api.policy = use_tools(("get_current_weather", "weather", {"city": "Oslo"}), searching=False)
    # Same session: the earlier search and its output are replayed, so the tool is callable at once.
    assert agent.run("And in Oslo?", session_id="s1").content == "Done: Sunny, 21C in Oslo"
    assert [item["type"] for item in api.requests[3]["input"] if "type" in item][:2] == ["tool_search_call", "tool_search_output"]
    # A new session has no such history.
    assert agent.run("And in Rome?", session_id="s2").content == "NOT LOADED"


def test_a_loaded_tool_that_needs_confirmation_pauses_and_resumes(api):
    deleted = []

    @tool(requires_confirmation=True)
    def delete_account(user_id: str) -> str:
        """Permanently delete a user account."""
        deleted.append(user_id)
        return f"Deleted {user_id}"

    api.policy = use_tools(("delete_account", "delete account", {"user_id": "u1"}))
    lazy = LazyTools(tools=[delete_account, get_current_weather])
    agent = make_agent(api, lazy, db=InMemoryDb())

    paused = agent.run("Delete account u1", session_id="s1")
    assert paused.is_paused and not deleted
    for requirement in paused.active_requirements:
        requirement.confirm()
    result = agent.continue_run(run_response=paused, requirements=paused.requirements, session_id="s1")

    assert deleted == ["u1"] and result.content == "Done: Deleted u1"
    # The definition sent to OpenAI has only OpenAI's fields, not Agno's requires_confirmation.
    [output] = [item for item in api.requests[-1]["input"] if item.get("type") == "tool_search_output"]
    assert [sorted(t) for t in output["tools"]] == [["defer_loading", "description", "name", "parameters", "type"]]


def test_native_needs_an_openai_responses_model():
    lazy = LazyTools(tools=[get_current_weather])
    with pytest.raises(ValueError, match="only implemented for OpenAIResponses"):
        lazy.wrap(ScriptedModel(), native=True)


def test_the_demos_openai_agents_run_against_the_mock(api, monkeypatch, capsys):
    """`examples/demo.py --model openai:gpt-6-luna`, the path a live run takes: lazy, native and eager agents."""
    from examples import demo

    api.policy = use_tools(*demo.PLAN)
    monkeypatch.setenv("OPENAI_BASE_URL", api.url)  # read by the OpenAI SDK, as the demo sets no base_url
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(sys, "argv", ["demo.py", "--model", "openai:gpt-6-luna", "--padding-tokens", "100"])
    try:
        demo.main()
    finally:
        logging.disable(logging.NOTSET)  # main() quiets warnings for the whole process

    assert capsys.readouterr().out.count("answer: Done: Paris: 21C and sunny / 250 USD = 231.48 EUR") == 3
    assert sum(r["tools"] == [TOOL_SEARCH] for r in api.requests) == 3  # the native agent's requests
    assert all(r["store"] is False and r["include"] == ["reasoning.encrypted_content"] for r in api.requests)
    assert "pads the system prompt" in api.requests[0]["input"][0]["content"]
