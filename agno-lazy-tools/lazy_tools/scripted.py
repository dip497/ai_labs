"""An offline stand-in for an LLM, used by the tests and the demo.

It plugs into Agno's real run loop as a normal `Model`. Each turn it asks a `policy`
for the next assistant message, passing only the tool names Agno actually sent that
turn, so a scripted "model" can call a tool only if it was really made available.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional
from uuid import uuid4

from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse

Policy = Callable[[List[Message], List[str]], ModelResponse]


@dataclass
class ScriptedModel(Model):
    id: str = "scripted"
    name: str = "ScriptedModel"
    provider: str = "Offline"
    policy: Optional[Policy] = None
    # The tool schemas sent on each model request, in order.
    requests: List[List[Dict[str, Any]]] = field(default_factory=list)

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._respond(kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        await asyncio.sleep(0)  # yield to the event loop like a network call, so concurrent runs interleave
        return self._respond(kwargs)

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield self._respond(kwargs)

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ModelResponse]:
        await asyncio.sleep(0)
        yield self._respond(kwargs)

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response

    def _respond(self, kwargs: Dict[str, Any]) -> ModelResponse:
        tools = copy.deepcopy(list(kwargs.get("tools") or []))
        self.requests.append(tools)
        assert self.policy is not None, "ScriptedModel needs a policy"
        return self.policy(kwargs["messages"], [Model._tool_name(t) for t in tools])

    def tool_names_per_request(self) -> List[List[str]]:
        return [[Model._tool_name(t) for t in request] for request in self.requests]


def say(text: str) -> ModelResponse:
    """An assistant turn with text and no tool calls (ends the run)."""
    return ModelResponse(role="assistant", content=text)


def call(name: str, **arguments: Any) -> ModelResponse:
    """An assistant turn that calls one tool."""
    return calls((name, arguments))


def calls(*tool_calls: "tuple[str, Dict[str, Any]]") -> ModelResponse:
    """An assistant turn that calls several tools in parallel."""
    return ModelResponse(
        role="assistant",
        tool_calls=[
            {"id": f"call_{uuid4().hex[:8]}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for n, a in tool_calls
        ],
    )


def tool_results(messages: List[Message]) -> Dict[str, str]:
    """The latest result of each tool called so far, by tool name."""
    return {m.tool_name: str(m.content) for m in messages if m.role == "tool" and m.tool_name}
