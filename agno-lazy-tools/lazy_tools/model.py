"""Model-side hooks that make lazily loaded tools callable within the same run.

Agno's `Model.response()` (and its async / streaming variants) builds two per-run
structures once, *before* its tool-call loop:

  * `_tool_dicts` - the tool schemas sent to the provider on every model turn
  * `_functions`  - the name -> Function table used to dispatch tool calls

and passes the *same* two objects into these methods on every turn:

  * `_process_model_response`, `_aprocess_model_response`,
    `process_response_stream`, `aprocess_response_stream`   -> tools=_tool_dicts
  * `get_function_calls_to_run`                              -> functions=_functions

Overriding them lets us add loaded tools to both structures in place, so each
addition lasts for the rest of the run. Resuming a paused run (human-in-the-loop)
dispatches through a table rebuilt from the agent's initial tools, via
`get_function_call_to_run_from_tool_execution`, which is hooked too.

These are Agno internals (verified against agno==3.0.11). This module is the only code
that depends on them, apart from the OpenAI adapter overrides in `lazy_tools/native.py`.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, Optional, Tuple

from agno.models.base import Model
from agno.models.response import ToolExecution
from agno.tools.function import Function, FunctionCall
from agno.utils.log import log_warning

from lazy_tools.toolkit import SEARCH_TOOL_NAME

if TYPE_CHECKING:
    from lazy_tools.toolkit import LazyTool, LazyTools


class LazyToolsModelMixin(Model):
    """Mix in ahead of an Agno model class; set `lazy_tools` to the agent's `LazyTools`.

    It subclasses `Model` (rather than `object`) so that `with_lazy_tools` can swap an
    existing model's class: CPython only allows `__class__` assignment between classes
    with the same base layout, and `Model` derives from `ABC`, not `object`.
    """

    lazy_tools: Optional["LazyTools"] = None

    def _process_model_response(self, *args: Any, **kwargs: Any) -> Any:
        self._add_loaded_schemas(kwargs)
        return super()._process_model_response(*args, **kwargs)

    async def _aprocess_model_response(self, *args: Any, **kwargs: Any) -> Any:
        self._add_loaded_schemas(kwargs)
        return await super()._aprocess_model_response(*args, **kwargs)

    def process_response_stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        self._add_loaded_schemas(kwargs)
        yield from super().process_response_stream(*args, **kwargs)

    async def aprocess_response_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        self._add_loaded_schemas(kwargs)
        async for response in super().aprocess_response_stream(*args, **kwargs):
            yield response

    def get_function_calls_to_run(self, *args: Any, **kwargs: Any) -> Any:
        self._add_loaded_functions(kwargs)
        return super().get_function_calls_to_run(*args, **kwargs)

    def get_function_call_to_run_from_tool_execution(
        self, tool_execution: ToolExecution, functions: Optional[Dict[str, Function]] = None
    ) -> FunctionCall:
        # The execution being resumed is a call the model made in the paused run, which
        # could only dispatch loaded tools - so restore that tool by name.
        entry = self.lazy_tools.entries.get(tool_execution.tool_name or "") if self.lazy_tools else None
        if entry is not None and functions is not None and entry.name not in functions:
            functions[entry.name] = _per_run_function(entry, functions)
        return super().get_function_call_to_run_from_tool_execution(tool_execution, functions)

    def _hook_args(self, kwargs: Dict[str, Any], container: str) -> Optional[Tuple["LazyTools", Any, Any]]:
        if self.lazy_tools is None:
            return None
        messages, target = kwargs.get("messages"), kwargs.get(container)
        if messages is None or target is None:
            # Agno passes these by keyword in 3.0.11; anything else means its internals moved.
            log_warning(f"lazy_tools: '{container}' not passed to the model hook; lazy tools disabled for this turn.")
            return None
        return self.lazy_tools, messages, target

    def _add_loaded_schemas(self, kwargs: Dict[str, Any]) -> None:
        """Add the schemas of loaded tools to the tool list sent to the provider."""
        args = self._hook_args(kwargs, "tools")
        if args is None:
            return
        lazy_tools, messages, tool_dicts = args
        present = {Model._tool_name(t) for t in tool_dicts}
        new = [entry.template for entry in lazy_tools.loaded(messages) if entry.name not in present]
        if new:
            tool_dicts.extend(self._format_tools(new))
            tool_dicts.sort(key=Model._tool_name)  # keep Agno's deterministic (cache-friendly) order

    def _add_loaded_functions(self, kwargs: Dict[str, Any]) -> None:
        """Add loaded tools to the table Agno dispatches tool calls through."""
        args = self._hook_args(kwargs, "functions")
        if args is None:
            return
        lazy_tools, messages, functions = args
        for entry in lazy_tools.loaded(messages):
            if entry.name not in functions:
                functions[entry.name] = _per_run_function(entry, functions)


def _per_run_function(entry: "LazyTool", functions: Dict[str, Function]) -> Function:
    """A per-run copy of `entry`'s Function, bound to the same run as `functions`.

    The run's copy of the meta-tool carries its agent, run context and tool hooks,
    exactly as Agno's parse_tools set them; the loaded tool gets the same.
    """
    function = entry.template._per_run_copy()
    meta = functions.get(SEARCH_TOOL_NAME)
    if meta is not None:
        function._agent, function._team, function._run_context = meta._agent, meta._team, meta._run_context
        function._images, function._videos = meta._images, meta._videos
        function._audios, function._files = meta._audios, meta._files
        if meta.tool_hooks is not None:
            function.tool_hooks = meta.tool_hooks
    return function


_LAZY_CLASSES: Dict[Tuple[type, type], type] = {}


def with_lazy_tools(model: Model, lazy_tools: "LazyTools", native: bool = False) -> Model:
    """A shallow copy of `model` whose class also runs the lazy-tool hooks.

    With `native=True`, the hooks load tools through the provider's own tool search
    (see `lazy_tools/native.py`); this raises for models that don't have one.
    """
    mixin: type = LazyToolsModelMixin
    if native:
        from lazy_tools.native import native_mixin_for

        mixin = native_mixin_for(model)
    cls = type(model)
    if not issubclass(cls, mixin):
        if (cls, mixin) not in _LAZY_CLASSES:
            # Same class name, so logs and provider names are unchanged.
            _LAZY_CLASSES[cls, mixin] = type(cls.__name__, (mixin, cls), {"__module__": __name__})
        cls = _LAZY_CLASSES[cls, mixin]
    wrapped = copy.copy(model)
    wrapped.__class__ = cls
    wrapped.lazy_tools = lazy_tools  # type: ignore[attr-defined]
    return wrapped
