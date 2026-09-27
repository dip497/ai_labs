"""Native tool search for OpenAI's Responses API: `lazy.wrap(OpenAIResponses(...), native=True)`.

In this mode `search_tools` reaches the model as OpenAI's client-executed `tool_search`
tool rather than as a function:

  * The model asks for tools with a `tool_search_call` output item. It is parsed into an
    ordinary `search_tools` tool call, so Agno runs `LazyTools.search_tools` as usual.
  * The result goes back as a `tool_search_output` item, in place of a
    `function_call_output`. It carries the definitions of the tools that search loaded.

So the request's `tools` never change. The loaded definitions sit in the input where the
search happened, and each request only appends to the one before, which keeps the
provider's prompt cache warm. (The default mode adds loaded tools to `tools`, which come
first in the prompt, so each load misses the cache from there on.) Dispatching loaded
tools is unchanged: the `LazyToolsModelMixin` hooks add them to the run's table.

Wire format: https://developers.openai.com/api/docs/guides/tools-tool-search. Like the
hooks in `lazy_tools/model.py`, this overrides internals of Agno's `OpenAIResponses`
(verified against agno==3.0.11): `_format_tool_params`, `_format_messages`,
`_parse_provider_response` and `_parse_provider_response_delta`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from agno.models.base import Model
from agno.models.message import Message
from agno.models.openai import OpenAIResponses
from agno.models.response import ModelResponse
from agno.utils.message import normalize_tool_messages, reformat_tool_call_ids

from lazy_tools.model import LazyToolsModelMixin
from lazy_tools.toolkit import SEARCH_TOOL_NAME, tool_call_arguments

if TYPE_CHECKING:
    from lazy_tools.toolkit import LazyTool

# The OpenAI function-tool fields an Agno Function fills; Function.to_dict() also has
# Agno-only ones, such as requires_confirmation.
_FUNCTION_TOOL_FIELDS = ("type", "name", "description", "parameters", "strict")

# call_id -> (the search's arguments, the definitions of the tools it loaded)
Searches = Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]]


class OpenAIToolSearchMixin(LazyToolsModelMixin):
    """Mix in ahead of `OpenAIResponses`: lazy tools load through OpenAI's client-executed tool search."""

    def _add_loaded_schemas(self, kwargs: Dict[str, Any]) -> None:
        pass  # Loaded tools travel in `tool_search_output` items, so `tools` stays as it was.

    def _format_tool_params(self, messages: List[Message], tools: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        formatted = super()._format_tool_params(messages, tools)
        return [_as_tool_search(tool) if _is_search_function(tool) else tool for tool in formatted]

    def _format_messages(
        self, messages: List[Message], compress_tool_results: bool = False, tools: Optional[List[Any]] = None
    ) -> List[Any]:
        items = super()._format_messages(messages, compress_tool_results, tools)
        searches, namespaces = self._tool_searches(messages)
        return [_native_item(item, searches, namespaces) for item in items]

    def _tool_searches(self, messages: List[Message]) -> Tuple[Searches, Dict[str, str]]:
        """The `search_tools` calls in `messages`, and the namespaces of the other tool calls, by call_id.

        A tool is listed only by the first search that loaded it: the model keeps a loaded
        tool for the rest of the conversation.
        """
        # super()._format_messages sends these ids; Agno re-ids foreign tool calls first.
        messages = reformat_tool_call_ids(normalize_tool_messages(messages), provider="openai_responses")
        namespaces = {
            _call_id(tool_call): tool_call["namespace"]
            for message in messages
            for tool_call in message.tool_calls or []
            if tool_call.get("namespace")
        }
        searches: Searches = {}
        loaded = set()
        for tool_call, entries in self.lazy_tools.searches(messages) if self.lazy_tools else []:
            new = [entry for entry in entries if entry.name not in loaded]
            loaded.update(entry.name for entry in new)
            searches[_call_id(tool_call)] = (tool_call_arguments(tool_call), [self._deferred_tool(e) for e in new])
        return searches, namespaces

    def _deferred_tool(self, entry: "LazyTool") -> Dict[str, Any]:
        """`entry` as a deferred OpenAI function tool, formatted the way Agno formats functions."""
        [tool] = super()._format_tool_params([], [entry.template])
        return {**{field: tool[field] for field in _FUNCTION_TOOL_FIELDS if tool.get(field) is not None}, "defer_loading": True}

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        model_response = super()._parse_provider_response(response, **kwargs)
        function_calls = iter(model_response.tool_calls or [])  # one per function_call item, in order
        tool_calls = []
        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(_with_namespace(next(function_calls), item))
            elif _is_search_call(item):
                tool_calls.append(_search_tool_call(item))
        if tool_calls:
            model_response.tool_calls = tool_calls
        return model_response

    def _parse_provider_response_delta(
        self, stream_event: Any, assistant_message: Message, tool_use: Dict[str, Any]
    ) -> Tuple[ModelResponse, Dict[str, Any]]:
        model_response, tool_use = super()._parse_provider_response_delta(stream_event, assistant_message, tool_use)
        item = getattr(stream_event, "item", None)
        if stream_event.type == "response.output_item.added" and getattr(item, "type", None) == "function_call":
            _with_namespace(tool_use, item)
        elif stream_event.type == "response.output_item.done" and _is_search_call(item):
            # The SDK has no argument-delta events for tool search calls, so read the finished item.
            tool_call = _search_tool_call(item)
            model_response.tool_calls = [tool_call]
            assistant_message.tool_calls = [*(assistant_message.tool_calls or []), tool_call]  # as Agno does for function calls
        return model_response, tool_use


def native_mixin_for(model: Model) -> type:
    """The mixin that gives `model` its provider's native tool search; raises if there is none."""
    if isinstance(model, OpenAIResponses):
        return OpenAIToolSearchMixin
    raise ValueError(
        f"Native tool search is only implemented for OpenAIResponses models, not {type(model).__name__}. "
        "Use lazy.wrap(model) without native=True."
    )


def _is_search_function(tool: Dict[str, Any]) -> bool:
    return tool.get("type") == "function" and tool.get("name") == SEARCH_TOOL_NAME


def _as_tool_search(function: Dict[str, Any]) -> Dict[str, Any]:
    """The `search_tools` function as OpenAI's client-executed tool search tool."""
    return {
        "type": "tool_search",
        "execution": "client",
        "description": function.get("description"),
        "parameters": {**(function.get("parameters") or {}), "additionalProperties": False},
    }


def _native_item(item: Any, searches: Searches, namespaces: Dict[str, str]) -> Any:
    """A formatted input item, with `search_tools` calls and results as tool search items."""
    if not isinstance(item, dict):
        return item  # e.g. a replayed reasoning item
    call_id, kind = item.get("call_id"), item.get("type")
    if call_id in searches:
        arguments, tools = searches[call_id]
        if kind == "function_call":
            return {"type": "tool_search_call", "execution": "client", "call_id": call_id, "status": "completed", "arguments": arguments}
        if kind == "function_call_output":
            return {"type": "tool_search_output", "execution": "client", "call_id": call_id, "status": "completed", "tools": tools}
    if kind == "function_call" and call_id in namespaces:
        return {**item, "namespace": namespaces[call_id]}
    return item


def _is_search_call(item: Any) -> bool:
    return getattr(item, "type", None) == "tool_search_call" and getattr(item, "execution", None) == "client"


def _search_tool_call(item: Any) -> Dict[str, Any]:
    """A client `tool_search_call` output item, as the Agno tool call that runs `search_tools`."""
    call_id = item.call_id or item.id
    arguments = item.arguments if isinstance(item.arguments, str) else json.dumps(item.arguments or {})
    return {
        # Agno re-ids tool calls whose id lacks the "fc_" prefix of function calls, which can
        # also change their call_id (reformat_tool_call_ids). This id itself is never sent.
        "id": f"fc_{item.id or call_id}",
        "call_id": call_id,
        "type": "function",
        "function": {"name": SEARCH_TOOL_NAME, "arguments": arguments},
    }


def _with_namespace(tool_call: Dict[str, Any], item: Any) -> Dict[str, Any]:
    """Keep a function call's namespace, so it can be sent back with the call."""
    namespace = getattr(item, "namespace", None)
    if namespace:
        tool_call["namespace"] = namespace
    return tool_call


def _call_id(tool_call: Dict[str, Any]) -> str:
    return tool_call.get("call_id") or tool_call.get("id")  # as Agno pairs calls with their results
