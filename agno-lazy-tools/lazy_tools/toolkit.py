"""The `search_tools` meta-tool and the catalog of deferred tools behind it."""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from agno.tools.function import Function
from agno.tools.toolkit import Toolkit

if TYPE_CHECKING:
    from agno.models.base import Model
    from agno.models.message import Message

SEARCH_TOOL_NAME = "search_tools"
_MEMO_SIZE = 4096


@dataclass
class LazyTool:
    """A catalog entry: enough metadata to be found, plus a target resolved on first load.

    `target` is a callable, an Agno `Function` (e.g. from `@tool` or a Toolkit), or an
    import path "package.module:attribute". An import path is not imported until the
    tool is loaded, so it needs an explicit `name` and `description` - the description
    is what `search_tools` matches against.
    """

    target: Union[str, Callable[..., Any], Function]
    name: Optional[str] = None
    description: Optional[str] = None
    _template: Optional[Function] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.target, str):
            if not self.name or not self.description:
                raise ValueError(
                    f"LazyTool({self.target!r}) needs a name and a description: "
                    "import paths are only imported when the tool is loaded."
                )
            return
        template = self.template
        self.name = self.name or template.name
        self.description = self.description or template.description or ""

    @property
    def is_resolved(self) -> bool:
        """Whether the target has been imported and turned into an Agno Function."""
        return self._template is not None

    @property
    def template(self) -> Function:
        """The processed Agno Function for this tool; imports the target on first access."""
        if self._template is None:
            target = self.target
            if isinstance(target, str):
                module_name, _, attribute = target.partition(":")
                target = getattr(importlib.import_module(module_name), attribute)
            if isinstance(target, Function):
                function = target._per_run_copy()
                if self.name:
                    function.name = self.name
                function.process_entrypoint()
            else:
                function = Function.from_callable(target, name=self.name)
            self._template = function
        return self._template


ToolSource = Union[Callable[..., Any], Function, Toolkit, LazyTool]


class LazyTools(Toolkit):
    """Stands in for a large tool catalog with a single `search_tools` meta-tool.

    Tools returned by a search are added to the model's tool list on the next model
    turn of the same run, and the model calls them natively. That injection happens
    in the model, so the agent's model must be wrapped:

        lazy = LazyTools(tools=[...])
        agent = Agent(model=lazy.wrap(OpenAIChat(id="gpt-6-luna")), tools=[lazy])
    """

    def __init__(
        self,
        tools: Sequence[ToolSource],
        max_results: int = 5,
        list_tool_names: bool = True,
        name: str = "lazy_tools",
        **kwargs: Any,
    ):
        self.max_results = max_results
        self._entries: Dict[str, LazyTool] = {}
        for source in tools:
            for entry in _to_entries(source):
                assert entry.name is not None
                if entry.name in self._entries:
                    raise ValueError(f"Duplicate lazy tool name: {entry.name!r}")
                self._entries[entry.name] = entry
        self._index = {
            entry_name: (_tokens(entry_name), _tokens(entry.description or ""))
            for entry_name, entry in self._entries.items()
        }
        # Search results are memoized per query: they are recomputed from the conversation
        # on every model turn (see `loaded`), so this keeps them cheap, and stable even if
        # `search` is overridden with something non-deterministic (like embeddings).
        self._memo: Dict[str, Tuple[str, ...]] = {}

        instructions = (
            f"You can load more tools on demand: {len(self._entries)} tools are available but not loaded yet. "
            f"To use one, first call `{SEARCH_TOOL_NAME}` with keywords for the capability you need "
            f'(or "select:<tool_name>" for an exact tool). The tools it returns become callable on your next step. '
            "Search before telling the user you cannot do something."
        )
        if list_tool_names:
            instructions += "\nTools you can load: " + ", ".join(self._entries)
        super().__init__(name=name, tools=[self.search_tools], instructions=instructions, add_instructions=True, **kwargs)

    @property
    def entries(self) -> Dict[str, LazyTool]:
        return dict(self._entries)

    def search_tools(self, query: str) -> str:
        """Find tools in the catalog and load them. Loaded tools can be called directly on your next step.

        Args:
            query: Keywords for the capability you need (e.g. "weather forecast"), or "select:name1,name2" to load tools by exact name.

        Returns:
            JSON listing the tools that were loaded, with their descriptions.
        """
        matches = self._search_memoized(query)
        if not matches:
            return json.dumps({"loaded": [], "hint": "No tools matched. Try other keywords, or select:<tool_name>."})
        return json.dumps({"loaded": [{"name": e.name, "description": e.description} for e in matches]})

    def search(self, query: str) -> List[LazyTool]:
        """Catalog entries matching `query`, best first. Override to plug in BM25 or embeddings."""
        query = query.strip()
        if query.lower().startswith("select:"):
            names = [n.strip() for n in query[len("select:") :].split(",")]
            return [self._entries[n] for n in dict.fromkeys(names) if n in self._entries]

        terms = _tokens(query)
        scored: List[Tuple[float, int, str]] = []
        for position, (entry_name, (name_tokens, description_tokens)) in enumerate(self._index.items()):
            score = sum(_term_score(term, name_tokens, description_tokens) for term in terms)
            if score > 0:
                scored.append((-score, position, entry_name))
        scored.sort()
        # Drop weak matches (e.g. one generic word in a description): every loaded tool costs context.
        cutoff = -scored[0][0] / 2 if scored else 0
        return [self._entries[entry_name] for score, _, entry_name in scored[: self.max_results] if -score >= cutoff]

    def loaded(self, messages: Iterable["Message"]) -> List[LazyTool]:
        """Tools loaded by the `search_tools` calls in `messages`, in load order.

        Derived from the conversation rather than stored per run: concurrent runs that
        share this toolkit cannot see each other's tools, and tools loaded in an earlier
        run stay loaded for as long as that run is part of the agent's history.
        """
        loaded: Dict[str, LazyTool] = {}
        for message in messages:
            for tool_call in message.tool_calls or []:
                function = tool_call.get("function") or {}
                if function.get("name") != SEARCH_TOOL_NAME:
                    continue
                arguments: Any = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        continue
                query = arguments.get("query") if isinstance(arguments, dict) else None
                if isinstance(query, str):
                    for entry in self._search_memoized(query):
                        loaded.setdefault(entry.name, entry)  # type: ignore[arg-type]
        return list(loaded.values())

    def wrap(self, model: "Model") -> "Model":
        """A copy of `model` that makes tools loaded by `search_tools` callable within the same run."""
        from lazy_tools.model import with_lazy_tools

        return with_lazy_tools(model, self)

    def __deepcopy__(self, memo: Dict[int, Any]) -> "LazyTools":
        # Agno deep-copies agents (and their models) for isolation; the catalog is
        # read-only after construction, so every copy can share it.
        return self

    def _search_memoized(self, query: str) -> List[LazyTool]:
        names = self._memo.get(query)
        if names is None:
            if len(self._memo) >= _MEMO_SIZE:
                self._memo.pop(next(iter(self._memo)))  # oldest first; the default search recomputes identically
            names = self._memo[query] = tuple(e.name for e in self.search(query))  # type: ignore[misc]
        return [self._entries[n] for n in names]


def _to_entries(source: ToolSource) -> List[LazyTool]:
    if isinstance(source, LazyTool):
        return [source]
    if isinstance(source, Toolkit):
        return [LazyTool(function) for function in source.get_functions().values()]
    return [LazyTool(source)]


_STOPWORDS = frozenset(
    "a an and are as at be by can do for from get how i in is it me my of on or please the to tool tools use what with".split()
)


def _tokens(text: str) -> frozenset:
    """Lowercase words of `text`, splitting snake_case and camelCase, minus stopwords."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return frozenset(t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOPWORDS)


def _term_score(term: str, name_tokens: frozenset, description_tokens: frozenset) -> float:
    """Name matches beat description matches; a prefix match ("book" ~ "booking") counts half."""

    def prefix_match(tokens: frozenset) -> bool:
        return any(
            min(len(term), len(t)) >= 4 and (t.startswith(term) or term.startswith(t)) for t in tokens
        )

    if term in name_tokens:
        return 2.0
    if term in description_tokens:
        return 1.0
    if prefix_match(name_tokens):
        return 1.0
    if prefix_match(description_tokens):
        return 0.5
    return 0.0
