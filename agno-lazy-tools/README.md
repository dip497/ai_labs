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

Changing the tool list has a cost of its own: tools come first in the prompt, so each load
misses the provider's prompt cache from there on. For OpenAI's Responses API there is a
**native mode**, `lazy.wrap(model, native=True)`, that loads tools through OpenAI's own
tool search instead: the tool list never changes and every request only appends to the
previous one, so the prompt cache keeps hitting. It is tested against a mock of the API
that checks each request against the OpenAI SDK's request types, but hasn't been run
against the live API yet.

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
from agno.models.openai import OpenAIResponses
from agno.tools.calculator import CalculatorTools

from lazy_tools import LazyTool, LazyTools

lazy = LazyTools(tools=[
    CalculatorTools(),                          # every function of a toolkit becomes deferred
    get_current_weather,                        # plain functions and @tool Functions
    LazyTool("my_pkg.crm:create_ticket",        # imported only when first loaded
             name="create_ticket", description="Open a support ticket in the CRM"),
])
model = OpenAIResponses(id="gpt-6-luna", store=False, include=["reasoning.encrypted_content"])  # see "Run it"
agent = Agent(model=lazy.wrap(model), tools=[lazy])  # plus any always-on tools

# Or, to load tools through OpenAI's native tool search:
agent = Agent(model=lazy.wrap(model, native=True), tools=[lazy])
```

### Native mode: OpenAI tool search

OpenAI's Responses API has [tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)
built in (gpt-5.4 and later). Its client-executed form fits this design: the model asks
for tools, and the application runs the search and returns the tools it found.
`lazy.wrap(model, native=True)` (`lazy_tools/native.py`) maps `search_tools` onto it:

- `search_tools` is sent as a `{"type": "tool_search", "execution": "client"}` tool, with
  the same description and `query` parameter, instead of as a function.
- The model asks for tools with a `tool_search_call` item. It is parsed into an ordinary
  `search_tools` call, so Agno runs the search as usual (tool hooks, events, history).
- The result goes back as a `tool_search_output` item, instead of a
  `function_call_output`, carrying the definitions of the tools found
  (`defer_loading: true`). From then on the model calls them like any function.

The difference from the default mode is where the loaded definitions go. The default mode
adds them to `tools`, which come first in the prompt, so each load changes the prompt
prefix. In native mode `tools` never changes, and the definitions sit in the input where
the search happened. Each request only appends to the previous one, from the mock-API
test:

```text
request 1   tools: [tool_search]   input: developer, user
request 2   tools: [tool_search]   input: developer, user, tool_search_call, tool_search_output
request 3   tools: [tool_search]   input: developer, user, tool_search_call, tool_search_output,
                                          function_call, function_call_output
```

That keeps the [prompt cache](https://developers.openai.com/api/docs/guides/prompt-caching)
hitting, which OpenAI's caching guide recommends tool search for ("Discovered tools are
appended at the end of context, preserving earlier reusable content"). It saves money as
well as latency: on GPT-5.6 and later, including gpt-6-luna, cache reads cost 0.1× the
input rate and cache writes 1.25×. After a load in the default mode, the whole prompt is
new to the cache. In native mode, everything before the search should be a cache read
(the live demo reports both).

Also:

- Each tool is sent once, by the first search that found it: the model keeps a loaded
  tool for the rest of the conversation.
- OpenAI returns calls to loaded tools with a `namespace` field; it is kept and sent back
  with the call.
- It works both when Agno replays the whole conversation on each request (its default for
  gpt-6 models) and when it chains requests with `previous_response_id` (its default for
  gpt-5 models).
- In the client-executed form, the catalog stays on our side, with this POC's search,
  lazy imports and per-conversation isolation. In OpenAI's hosted form, OpenAI searches
  the request's own `tools`, so every schema would have to be sent up front.

## Run it

```bash
cd agno-lazy-tools
uv sync
uv run pytest                          # 33 tests, all offline
uv run python examples/demo.py         # offline, with a scripted model

# Against a real model (needs OPENAI_API_KEY / ANTHROPIC_API_KEY):
uv run --extra openai python examples/demo.py --model openai:gpt-6-luna --padding-tokens 2000
uv run --extra anthropic python examples/demo.py --model anthropic:claude-opus-5 --question "..."
```

The offline "model" (`lazy_tools/scripted.py`) runs inside Agno's real run loop. It
decides each step from the tool list Agno actually sent it, so it can only call a tool
that was really loaded.

With an OpenAI model, the demo also runs a native-mode agent, and prints each request's
input tokens as OpenAI reports them, with how many were cached and written to the
cache. `--padding-tokens` stands in for a production agent's long system prompt, since
OpenAI only caches prompts of 1,024 tokens or more and the demo's are shorter.

Two things about gpt-6-luna, both handled by the demo:

- It is a reasoning model, and over Chat Completions it calls functions only with
  `reasoning_effort="none"` ([model page](https://developers.openai.com/api/docs/models/gpt-6-luna)).
  So the demo uses the Responses API.
- Agno 3.0.11 counts only `o3`, `o4-mini` and `gpt-5*` ids as reasoning models. For other
  ids it drops reasoning items between requests, so the demo passes `store=False` and
  `include=["reasoning.encrypted_content"]`, which make Agno replay them. This hasn't
  been checked against the live API yet.

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

Tools in context: lazy ~1,276 tokens over 3 requests (~547 schemas + ~729 instructions) vs eager ~4,382 over 2 -> 71% less.
```

Token counts are estimates: Agno's counter falls back to characters / 4 without
`tiktoken`. The lazy side includes its system-prompt instructions, which by default list
every loadable tool name. Native mode sends the same schemas, in the input rather than in
`tools`, so its counts match the lazy agent's; what it changes is the cache hits. There
are no live numbers for that yet (see [Not tested](#trade-offs-and-limitations)).

**Scaling**, extrapolated from the measured unit costs:

| Unit | Tokens |
|---|---|
| One tool schema (Agno toolkit average) | ~63 |
| `search_tools` schema | ~79 |
| Instructions | ~84, plus ~4.5 per listed tool name |

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
- **Native mode** (`tests/test_native.py`): Agno's real `OpenAIResponses` against a local
  mock of the Responses API that speaks the tool search protocol. Like the real API, the
  mock rejects a request (HTTP 400) whose items or tools don't validate against the
  `openai` SDK's request types, which are generated from OpenAI's API spec (unknown or
  missing fields, wrong values), or whose outputs answer no call. The tests check:
  - all four run paths, streaming included;
  - `tools` identical on every request, and each request's input extending the previous
    one;
  - chaining with `previous_response_id`;
  - each tool sent once;
  - tools staying loaded through the session history;
  - a loaded tool that needs confirmation, paused and resumed, whose definition carries
    only OpenAI's fields;
  - the demo's live OpenAI path (`--model openai:gpt-6-luna`: lazy, native and eager
    agents), pointed at the mock with `OPENAI_BASE_URL`.

  Each step of `native.py` was broken in turn to check that some test fails: the
  unchanged `tools`, the tool format, the item conversion, parsing, streaming,
  namespaces, and sending each tool once.

**Negative controls:** with an unwrapped model, the same scenario never loads the tool
(the tool list stays `[search_tools]`). This is a test too, so a future Agno release that
re-reads tools per turn would show up as a failure. And the default mode on the mock
Responses API shows `tools` changing on every load, which is what native mode avoids.

## Trade-offs and limitations

- **Agno internals.** The hooks override six `Model` methods:
  `_process_model_response`, `_aprocess_model_response`, `process_response_stream`,
  `aprocess_response_stream`, `get_function_calls_to_run`, and
  `get_function_call_to_run_from_tool_execution`. They also use the `Function` private
  attributes (`_agent`, `_run_context`, ..., `_per_run_copy()`). All of it lives in
  `lazy_tools/model.py`; Agno is pinned, the tests catch breakage, and the hooks log a
  warning if the arguments they rely on move. Native mode also overrides four
  `OpenAIResponses` methods (`_format_tool_params`, `_format_messages`,
  `_parse_provider_response`, `_parse_provider_response_delta`), in `lazy_tools/native.py`.
- **Prompt caching.** In the default mode, tools come first in the prompt prefix, so each
  load invalidates the provider's prompt cache from that point on. Native mode avoids that,
  but only on OpenAI's Responses API (gpt-5.4 and later). Anthropic has tool search too,
  but Agno's Claude adapter can't send it yet; see
  [Alternatives](#alternatives-considered).
- **Native mode relies on the docs, not a live run.** The wire format comes from OpenAI's
  tool search guide and the `openai` SDK's types (2.25.0 is the first SDK that has them).
  Two things there are only documented, not tried here: that the loaded tools need not be
  in the request's `tools` ("Client-executed tool search also supports more advanced
  patterns where your application returns tools that were not present in the original
  request"), and that `strict` can be left out of function definitions, as OpenAI's
  examples do even though the spec marks it required.
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
- **Not tested:** Teams, and a live LLM. The sandbox this was built in first blocked
  `api.openai.com`, then reached it with an API key that OpenAI rejected (HTTP 401). So the
  evidence is the scripted model plus the real provider adapters against mock APIs. Run
  the demo with `--model` to try a real one.

## Alternatives considered

| Approach | Same run? | Public Agno API only? | Native tool calls? | Notes |
|---|---|---|---|---|
| **This POC:** model hooks | yes | no | yes | Provider schema validation, per-tool hooks, confirmation and events all apply. |
| **This POC, native mode:** OpenAI client-executed tool search (`native=True`) | yes | no | yes | **No prompt-cache miss**: `tools` stays fixed and loaded definitions are appended to the input. OpenAI Responses API only (gpt-5.4 and later). Mock-tested, not yet run live. |
| OpenAI hosted tool search (`{"type": "tool_search"}` + `defer_loading: true` on every tool) | yes | no | yes | OpenAI runs the search, but over the request's own `tools`: every schema is sent (and imported) up front, and deferred functions still show their names and descriptions. Namespaces help. Not implemented. |
| OpenAI `additional_tools` input item after a `search_tools` result | yes | no | yes | Also appends tools without changing `tools`; OpenAI's caching guide suggests it for "tool-loading history". The model searches through an ordinary function rather than tool search. Not implemented. |
| Anthropic tool search (`tool_search_tool_bm25_20251119` + `defer_loading`) | yes | no | yes | Agno 3.0.11's `Claude` adapter rebuilds function tools without `defer_loading`. Not implemented. |
| Proxy tool: `search_tools` + `call_tool(name, arguments)` | yes | yes | no | Most portable. But arguments go through an untyped dict, and Agno only sees `call_tool`, so per-tool confirmation and hooks don't apply. |
| Callable tools factory (`tools=fn`, `cache_callables=False`) reading loaded names from `session_state` | no, next run | yes | yes | Needs a second run (e.g. an automatic "continue") before the tool is usable. |
| Upstream change in Agno | yes | yes | yes | The proper fix, e.g. `response()` re-reading a mutable tool registry each turn. PR #7528 tried this and wasn't merged. |

## Next steps (if this goes further)

- Run the demo against the live API with a working key: check native mode's wire format
  and that its second request reads the first from the cache, then record the numbers
  here.
- Native tool search for Anthropic, which needs `defer_loading` through Agno's `Claude`
  adapter; the default mode remains the fallback for providers without it.
- Propose a small upstream hook in Agno (a per-turn tool provider), which would remove the
  reliance on internals.
- Lazily connect MCP / database toolkits when first loaded, and add tool unloading or a
  cap on loaded tools.
- Better search (BM25 or embeddings), and Team support.
