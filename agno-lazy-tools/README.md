# Lazy tool loading for Agno (POC)

**Question:** can an Agno agent load tools lazily, so a tool's schema only enters the
model's context once the agent actually needs it?

**Answer: yes.** It works on stock `agno==3.0.11` without patching Agno. The agent
starts with a single meta-tool, `search_tools`. The tools a search finds are added to the
model's tool list on the **next model turn of the same run**, and the model then calls
them natively. With a 35-tool catalog, the demo question is answered with **~71% fewer
tool tokens in context** than with every tool attached, at the cost of one extra model
request.

The catch: Agno has no public hook for changing tools mid-run, so this overrides six
internal, undocumented methods of Agno's `Model` class. That code is isolated in one file
(`lazy_tools/model.py`), and the Agno version is pinned.

## Why Agno needs help here

- Agno's dynamic-tool features (`tools=<callable factory>`, `agent.add_tool()`,
  `agent.set_tools()`) are resolved **at the start of a run**, so they only take effect on
  the *next* run.
- Inside a run, `Model.response()` (`agno/models/base.py`) builds the tool schemas and the
  dispatch table **once, before** its tool-call loop. Tools added while the loop runs are
  invisible to the model and can't be dispatched.
- Upstream attempts to add this to Agno itself weren't merged:
  [PR #7528](https://github.com/agno-agi/agno/pull/7528) (`DiscoverableTools`, patched
  `models/base.py`) was closed on 2026-09-16, and
  [PR #10574](https://github.com/agno-agi/agno/pull/10574) (lazy skill tools) was closed on
  2026-09-25. [Issue #5136](https://github.com/agno-agi/agno/issues/5136) (dynamic tools)
  is still open.

## How it works

1. **`LazyTools`** is an Agno `Toolkit` that exposes one tool, `search_tools(query)`, in
   front of a catalog of *deferred* tools: plain functions, `@tool` functions, whole
   toolkits, or `"package.module:function"` import paths that are only imported when
   loaded. Search is keyword-based (name matches beat description matches); the model can
   also ask for exact names with `select:name1,name2`.
2. **`lazy.wrap(model)`** returns a copy of your model with hooks on the per-turn methods
   that `Model.response()` calls with its tool list and dispatch table. Before every model
   call, the hooks add the schemas of loaded tools. Before tool calls are dispatched, they
   add the matching Agno `Function`s. Both are bound to the run's agent, run context and
   `tool_hooks`, as Agno would bind them. A sixth hook covers resuming a paused
   (human-in-the-loop) run.
3. **"Loaded" is not stored anywhere.** It is derived from the model's own
   `search_tools` calls in the conversation. So concurrent runs that share one `LazyTools`
   and one model can't leak tools into each other. The PR #7528 review flagged this risk
   for shared state; this design avoids it by having no state. Tools also stay loaded
   while the loading run is in the agent's history.

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.calculator import CalculatorTools

from lazy_tools import LazyTool, LazyTools

lazy = LazyTools(tools=[
    CalculatorTools(),                          # every function of a toolkit becomes deferred
    get_current_weather,                        # plain functions and @tool Functions
    LazyTool("my_pkg.crm:create_ticket",        # imported only when first loaded
             name="create_ticket", description="Open a support ticket in the CRM"),
])
agent = Agent(model=lazy.wrap(OpenAIChat(id="gpt-6-luna")), tools=[lazy])  # plus any always-on tools
```

## Run it

```bash
cd agno-lazy-tools
uv sync
uv run pytest                          # 21 tests, all offline
uv run python examples/demo.py         # offline, with a scripted model

# Against a real model (needs OPENAI_API_KEY / ANTHROPIC_API_KEY):
uv run --extra openai python examples/demo.py --model openai:gpt-6-luna
uv run --extra anthropic python examples/demo.py --model anthropic:claude-opus-5 --question "..."
```

The offline "model" (`lazy_tools/scripted.py`) runs inside Agno's real run loop. It
decides each step from the tool list Agno actually sent it, so it can only call a tool
that was really loaded.

## Results

`examples/demo.py`, offline. The catalog is 35 tools: 29 from 7 Agno toolkits (Calculator,
Shopify, HackerNews, Airflow, Email, Pubmed, Website) plus 6 of our own, 2 of them
registered by import path.

```text
Lazy agent (only search_tools attached):
  request 1:  1 tool , ~   79 schema tokens [search_tools]
             -> search_tools({"query": "current weather"}), search_tools({"query": "convert currency"})
  request 2:  4 tools, ~  234 schema tokens [search_tools, get_current_weather, get_weather_forecast, convert_currency]
             -> get_current_weather({"city": "Paris"}), convert_currency({"amount": 250, "from_currency": "USD", "to_currency": "EUR"})
  request 3:  4 tools, ~  234 schema tokens [search_tools, get_current_weather, get_weather_forecast, convert_currency]
             -> final answer
  answer: Paris: 21C and sunny / 250 USD = 231.48 EUR
  examples.fx imported: True

Eager agent (all 35 tools attached):
  request 1: 35 tools, ~2,191 schema tokens [the whole catalog]
             -> get_current_weather({"city": "Paris"}), convert_currency({"amount": 250, "from_currency": "USD", "to_currency": "EUR"})
  request 2: 35 tools, ~2,191 schema tokens [the whole catalog]
             -> final answer
  answer: Paris: 21C and sunny / 250 USD = 231.48 EUR

Tools in context: lazy ~1,258 tokens over 3 requests (~547 schemas + ~711 instructions) vs eager ~4,382 over 2 -> 71% less.
```

Token counts are estimates: Agno's counter falls back to characters / 4 without
`tiktoken`. The lazy side includes its system-prompt instructions, which by default list
every loadable tool name.

**Scaling**, extrapolated from the measured unit costs:

| Unit | Tokens |
|---|---|
| One tool schema (Agno toolkit average) | ~63 |
| `search_tools` schema | ~79 |
| Instructions | ~78, plus ~4.5 per listed tool name |

Per request, after 3 tools are loaded:

| Catalog size | Eager (all schemas) | Lazy, names listed | Lazy, `list_tool_names=False` |
|---:|---:|---:|---:|
| 35 | ~2,200 | ~500 | ~350 |
| 100 | ~6,300 | ~800 | ~350 |
| 300 | ~18,800 | ~1,700 | ~350 |

Many MCP servers ship much larger schemas than Agno's toolkits, and the savings grow
with schema size.

## What the tests cover

- **All four run paths:** `run`, `run(stream=True)`, `arun`, `arun(stream=True)`. Turn 1
  sends only `search_tools`, the found tool is sent from turn 2, and unrelated tools never
  are.
- **Real provider adapters:** Agno's `OpenAIChat` and `Claude` are run against a local mock
  API. The HTTP request bodies show `tools` growing mid-run through each adapter's own
  formatting (`tests/test_providers.py`).
- **Isolation:** two concurrent runs that share one `LazyTools` *and* one model instance
  don't see each other's tools.
- **History:** a tool loaded in one run is available in the next run of the same session
  (`add_history_to_context=True`), and not in a new session.
- **Lazy import:** an import-path tool's module is imported only once that tool is loaded.
- **Binding:** loaded tools get the run's agent, `RunContext` / `session_state`, and agent
  `tool_hooks`.
- **Human-in-the-loop:** a loaded `requires_confirmation` tool pauses, and
  `continue_run` / `acontinue_run` resumes it, both confirm and reject.
- **Misc:** calling a tool that was never loaded fails; toolkits and `@tool` functions can
  be deferred; plus search ranking, `select:`, and the result cap.

Negative control: with an unwrapped model, the same scenario never loads the tool (the
tool list stays `[search_tools]`).

## Trade-offs and limitations

- **Agno internals.** The hooks override six `Model` methods:
  `_process_model_response`, `_aprocess_model_response`, `process_response_stream`,
  `aprocess_response_stream`, `get_function_calls_to_run`, and
  `get_function_call_to_run_from_tool_execution`. They also use the `Function` private
  attributes (`_agent`, `_run_context`, ..., `_per_run_copy()`). All of it lives in
  `lazy_tools/model.py`; Agno is pinned, the tests catch breakage, and the hooks log a
  warning if the arguments they rely on move.
- **Prompt caching.** Tools come first in the prompt prefix, so each load invalidates the
  provider's prompt cache from that point on. Provider-native tool search avoids this by
  appending discovered tools at the end of the context instead of changing `tools`. Both
  OpenAI (Responses API, gpt-5.4 and later, e.g. `gpt-6-luna`) and Anthropic offer it;
  see [Alternatives](#alternatives-considered).
- **Latency.** Each search costs a model round trip. The model can run several searches in
  parallel, as the demo does.
- **Not carried over from toolkits:** toolkit `instructions` (the system prompt is already
  built), `connect()` for toolkits that need it (databases, MCP), async-only variants, and
  `strict` schemas for structured outputs.
- **Unloaded tools aren't auto-loaded.** Calling a deferred tool without searching first
  returns Agno's generic "tool does not exist" error.
- **A lazy import that fails raises during the run.**
- **Search is basic keyword matching.** Override `LazyTools.search()` to use BM25 or
  embeddings; results are memoized per query, so they stay consistent within a
  conversation.
- **Not tested:** Teams, and a live LLM. The sandbox this was built in blocks
  `api.openai.com` at its network egress proxy, so the evidence is the scripted model plus
  the real provider adapters against a mock API. Run the demo with `--model` to try a
  real one.

## Alternatives considered

| Approach | Same run? | Public Agno API only? | Native tool calls? | Notes |
|---|---|---|---|---|
| **This POC:** model hooks | yes | no | yes | Provider schema validation, per-tool hooks, confirmation and events all apply. |
| Proxy tool: `search_tools` + `call_tool(name, arguments)` | yes | yes | no | Most portable. But arguments go through an untyped dict, and Agno only sees `call_tool`, so per-tool confirmation and hooks don't apply. |
| Callable tools factory (`tools=fn`, `cache_callables=False`) reading loaded names from `session_state` | no, next run | yes | yes | Needs a second run (e.g. an automatic "continue") before the tool is usable. |
| Provider-native tool search: OpenAI Responses API (`{"type": "tool_search"}` + `defer_loading: true`, gpt-5.4 and later, e.g. `gpt-6-luna`) or Anthropic (`tool_search_tool_bm25_20251119` + `defer_loading`) | yes | n/a | yes | **No prompt-cache miss**, since discovered tools are appended rather than changing `tools`. But Agno 3.0.11 can't express it yet: both its `OpenAIResponses` and `Claude` adapters serialize function tools without `defer_loading`. OpenAI responses also carry `tool_search_call` / `tool_search_output` items that Agno would need to round-trip. Untested here. |
| Upstream change in Agno | yes | yes | yes | The proper fix, e.g. `response()` re-reading a mutable tool registry each turn. PR #7528 tried this and wasn't merged. |

## Next steps (if this goes further)

- Use provider-native tool search where the model supports it (e.g. `gpt-6-luna` via
  `OpenAIResponses`), keeping this client-side injection as the fallback for other
  providers. That needs a small adapter subclass that emits `defer_loading` for deferred
  tools, validated against the live API.
- Propose a small upstream hook in Agno (a per-turn tool provider), which would remove the
  reliance on internals.
- Lazily connect MCP / database toolkits when first loaded, and add tool unloading or a
  cap on loaded tools.
- Better search (BM25 or embeddings), and Team support.
