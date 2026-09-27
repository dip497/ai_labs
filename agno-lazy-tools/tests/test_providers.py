"""Agno's real OpenAI and Claude adapters against a local mock API.

This checks the wire level: the `tools` array in the HTTP request bodies the provider
SDKs actually send, run through each adapter's own tool formatting.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat

from lazy_tools import LazyTools
from tests.test_lazy_tools import convert_currency, get_current_weather


def next_step(tool_names, conversation):
    """The mock model: search, call the weather tool once it is in `tools`, then answer (a str)."""
    if "Sunny, 21C in Paris" in conversation:
        return "It is sunny in Paris."
    if conversation.count("search_tools") > 4:  # the tool never showed up: stop rather than loop forever
        return "The weather tool never loaded."
    if "get_current_weather" in tool_names:
        return "get_current_weather", {"city": "Paris"}
    return "search_tools", {"query": "weather"}


def openai_reply(body):
    step = next_step([t["function"]["name"] for t in body.get("tools", [])], json.dumps(body["messages"]))
    if isinstance(step, str):
        message, finish_reason = {"role": "assistant", "content": step}, "stop"
    else:
        name, arguments = step
        tool_call = {"id": f"call_{len(body['messages'])}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        message, finish_reason = {"role": "assistant", "content": None, "tool_calls": [tool_call]}, "tool_calls"
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 0,
        "model": body["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def anthropic_reply(body):
    step = next_step([t["name"] for t in body.get("tools", [])], json.dumps(body["messages"]))
    if isinstance(step, str):
        content, stop_reason = [{"type": "text", "text": step}], "end_turn"
    else:
        name, arguments = step
        content, stop_reason = [{"type": "tool_use", "id": f"toolu_{len(body['messages'])}", "name": name, "input": arguments}], "tool_use"
    return {
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.fixture
def mock_api():
    """A local HTTP server speaking just enough of both APIs; yields (base URL, request bodies)."""
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            reply = openai_reply(body) if self.path.endswith("/chat/completions") else anthropic_reply(body)
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", bodies
    server.shutdown()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_provider_requests_gain_the_loaded_tool_mid_run(mock_api, provider):
    url, bodies = mock_api
    if provider == "openai":
        model = OpenAIChat(id="gpt-mock", api_key="test", base_url=f"{url}/v1")
    else:
        model = Claude(id="claude-mock", api_key="test", client_params={"base_url": url})

    lazy = LazyTools(tools=[get_current_weather, convert_currency])
    agent = Agent(model=lazy.wrap(model), tools=[lazy], telemetry=False)
    result = agent.run("What's the weather in Paris?")

    def tool_names(body):
        return [t["function"]["name"] if provider == "openai" else t["name"] for t in body.get("tools", [])]

    assert result.content == "It is sunny in Paris."
    assert [tool_names(body) for body in bodies] == [
        ["search_tools"],
        ["get_current_weather", "search_tools"],
        ["get_current_weather", "search_tools"],
    ]
