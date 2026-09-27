# Working notes: agno-lazy-tools

Context for picking this work up in a new session. Last updated 2026-09-27 (second session).
`README.md` is the write-up; this file is status, findings and next steps.

## Status

- Session 1 built the POC (branch `claude/lazy-load-tools-poc-zd6pcs`). Session 2 continued
  on `claude/continue-6nsk22` and added the **native OpenAI tool search mode**:
  `lazy.wrap(OpenAIResponses(...), native=True)`, in `lazy_tools/native.py`. No PR has been
  opened.
- `uv run pytest`: 33 tests pass, all offline. 11 of them are in `tests/test_native.py`, a
  mock Responses API that validates every request against the `openai` SDK's request
  types. `uv run python examples/demo.py` runs offline.
- **Still not done: any run against a live model.** The egress proxy now lets
  `api.openai.com` through, but OpenAI rejects the environment's `OPENAI_API_KEY`:
  `GET /v1/models` returns HTTP 401 from OpenAI itself, not the proxy. The key has to be
  replaced in the environment's settings, and a new session picks it up. Don't print or
  inspect the key, or the body of the 401 (it echoes part of the key): the harness blocks
  that as credential exposure.
- The demo's live OpenAI path now uses `OpenAIResponses` and runs three agents (lazy,
  native, eager), printing the input, cached and cache-written tokens OpenAI reports for
  each request. A test runs that path against the mock.

## Next steps (in order)

1. **Check the key works.**
   `curl -sS https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" -o /dev/null -w "%{http_code}\n"`
   should print `200`. A `401` means OpenAI rejects the key; a `403` from the proxy means
   the network policy blocks the host (`curl -sS "$HTTPS_PROXY/__agentproxy/status"`).
2. **Run the live demo.**
   `uv run --extra openai python examples/demo.py --model openai:gpt-6-luna --padding-tokens 2000`
   It makes about 8 requests; gpt-6-luna costs $0.10 per 1M input tokens ($0.01 cached,
   $0.125 cache writes) and $0.50 per 1M output. Padding is needed because OpenAI only
   caches prompts of 1,024+ tokens, and the demo's are shorter. What to look for:
   - Native agent: request 2 should report roughly request 1's input as cached. The lazy
     (default-mode) agent's request 2 should report about 0 cached, because `tools`
     changed. Put the per-request numbers in the README "Results", and fix the README's
     "not run live" statements.
   - The model should actually emit `tool_search_call`s (native) or `search_tools` calls
     (default) rather than answering without tools.
   - If a request fails with a 400, the error names the field. Suspects, most likely first:
     (a) loaded tools that are absent from `tools` (the docs say that's allowed);
     (b) function definitions without `strict` (the spec says required; the docs' examples
     omit it); (c) reasoning-item replay (`store=False` + encrypted reasoning, see below);
     (d) the replayed `tool_search_call` has no `id`.
3. Then, if it goes further: native tool search for Anthropic (Agno's `Claude` adapter
   needs `defer_loading` plus a `tool_search_tool_*` tool); OpenAI's hosted mode or
   namespaces as a comparison; an upstream Agno hook proposal. See the README's
   "Alternatives considered".

## How native mode works (details the README skips)

- `search_tools` goes out as `{"type": "tool_search", "execution": "client", "description",
  "parameters"}` (the function's own description and schema, plus
  `"additionalProperties": false`).
- A `tool_search_call` output item (`execution: "client"`) is parsed into an Agno tool call
  named `search_tools`, with id `fc_<item id>`. The `fc_` prefix stops Agno's
  `reformat_tool_call_ids` from re-iding it, which could also rewrite its `call_id`; that
  id is never sent. When streaming, the SDK has no argument-delta events for tool search
  calls, so it is read from `response.output_item.done`.
- `_format_messages` post-processes Agno's items. The `search_tools` `function_call` /
  `function_call_output` pair becomes `tool_search_call` / `tool_search_output`, found by
  `call_id` after the same id rewriting Agno applies. The output's `tools` are the loaded
  tools' definitions (OpenAI fields only, plus `defer_loading: true`). Each tool is listed
  only by the first search that loaded it, so earlier items never change, and a later
  search that only re-finds loaded tools returns an empty list.
- Function calls keep the `namespace` OpenAI returns (it equals the function name for
  top-level functions), stored on the Agno tool call and sent back.
- "Loaded" is still derived from the conversation (`LazyTools.searches()`), so dispatch,
  history and isolation work as in the default mode. `_add_loaded_schemas` is a no-op, so
  `tools` never changes.

## Findings

### Session 2: OpenAI (docs now reachable)

- **Tool search guide** (developers.openai.com/api/docs/guides/tools-tool-search; append
  `.md` for markdown):
  - Client mode: the model emits `tool_search_call {id, call_id, execution: "client",
    status, arguments}` and stops. The app answers with `tool_search_output {call_id,
    execution: "client", status: "completed", tools: [...]}`.
  - Loaded tools may be absent from the request's `tools` ("advanced injection
    patterns").
  - Loaded tools stay callable for the rest of the conversation.
  - Hosted mode needs every tool in `tools` with `defer_loading`, and for individually
    deferred functions the model still sees names and descriptions. Namespaces hide them;
    OpenAI recommends namespaces or MCP servers.
  - An `additional_tools` input item (role `developer`) adds tools at a given point in the
    input.
- **OpenAI SDK:** tool search types exist since `openai` 2.25.0 (2026-03-05), now the
  minimum in `pyproject.toml`; 3.19.2 is locked. `FunctionToolParam.strict` is
  `Required[Optional[bool]]` in the SDK (from the spec), though the docs' examples leave it
  out, as does Agno.
- **gpt-6-luna** (model page):
  - It's a reasoning model (effort `none` to `max`, default `medium`).
  - Its Responses API supports `tool_search`.
  - **Chat Completions supports function calling only with `reasoning_effort="none"`**, so
    session 1's `OpenAIChat` demo path would have failed; it now uses the Responses API.
  - 1,050,000-token context.
- **Prompt caching guide**, for GPT-5.6 and later:
  - The minimum cacheable prefix is 1,024 visible tokens. Cache reads cost 0.1× the input
    rate and cache writes 1.25×.
  - The implicit breakpoint sits at the end of the latest eligible message: a user
    message, the last tool response of a group, or the last developer message of the
    initial group.
  - Any change to `tools` invalidates the cached prefix.
  - The guide recommends tool search and `additional_tools` for loading tools.
  - Usage reports `input_tokens_details.cache_write_tokens`; Agno 3.0.11 drops it, so the
    demo's `Responses` subclass records it.
- **Agno 3.0.11 `OpenAIResponses`:**
  - `_using_reasoning_model()` is true only for ids starting `o3`, `o4-mini` or `gpt-5`.
    For those it chains with `previous_response_id` (`store=True`) and sends only new items.
  - For other ids (gpt-6-*) it replays everything and drops reasoning items, unless
    `store=False` (then it replays one reasoning item per response; add
    `include=["reasoning.encrypted_content"]`, as the demo does).
  - Replayed `function_call` items carry their `fc_` ids. With `store` on and reasoning
    dropped, the API may reject them for lacking their reasoning item. That's a known
    Responses API error, unverified here, and the reason the demo uses `store=False`.
  - It sends Agno-only fields (`requires_confirmation`, `external_execution`,
    `approval_type`) inside every toolkit function in `tools`. Its Chat Completions adapter
    strips them only for some OpenAI-compatible providers, which suggests OpenAI ignores
    them. The mock tolerates them in `tools`. Native mode's `tool_search_output`
    definitions carry OpenAI fields only.

### Session 1: Agno internals (still accurate)

- **Why Agno needs help.** In `agno/models/base.py`, `response()`, `aresponse()`,
  `response_stream()` and `aresponse_stream()` build `_tool_dicts` and `_functions` once,
  before their `while True` tool loop. `tools=<callable>`, `add_tool()` and `set_tools()`
  resolve at run start, so they only affect the next run.
- **The hook points.** The same `_tool_dicts` list and `_functions` dict are passed by
  keyword on every turn, into these methods:
  - `tools=` goes to `_process_model_response`, `_aprocess_model_response`,
    `process_response_stream` and `aprocess_response_stream`.
  - `functions=` goes to `get_function_calls_to_run` (via `_prepare_function_calls`).

  `lazy_tools/model.py` mutates both in place.
- **Resuming a paused run.** `continue_run` dispatches through
  `model.get_function_call_to_run_from_tool_execution(tool, functions)`, with a table rebuilt
  from the initial tools. That is hooked too.
- **Where "loaded" comes from.** No provider overrides the hooks. But Bedrock, Gemini and
  `OpenAIResponses` override `format_function_call_results`, so the loaded set is derived
  from the assistant `tool_calls` (normalized OpenAI-style dicts for every provider), never
  from tool-result messages.
- **The mixin must subclass `Model`.** `__class__` assignment fails with "object layout
  differs" if the mixin derives from `object`, because `Model` derives from `ABC`.
- **Negative control.** Without `lazy.wrap(model)`, the tool list stays `[search_tools]` for
  the whole run (`test_without_the_model_hooks_the_tool_list_never_changes`).
- **Anthropic native tool search:** `tool_search_tool_regex_20251119` or
  `tool_search_tool_bm25_20251119`, plus `defer_loading: true` on tools; at least one tool
  must stay non-deferred. Agno's Claude adapter (`format_tools_for_model` in
  `agno/utils/models/claude.py`) rebuilds function tools without `defer_loading`.
- **Upstream.**
  - [PR #7528](https://github.com/agno-agi/agno/pull/7528) (`DiscoverableTools`, patched
    `base.py`): closed unmerged 2026-09-16.
  - [PR #10574](https://github.com/agno-agi/agno/pull/10574) (lazy skill tools): closed
    unmerged 2026-09-25.
  - [Issue #5136](https://github.com/agno-agi/agno/issues/5136) (dynamic tools): open.

## Sandbox gotchas

- **Reachable now:** `api.openai.com` (but the key gets a 401), `developers.openai.com`,
  `openaipublic.blob.core.windows.net` (tiktoken's encodings; `tiktoken` itself isn't
  installed, so token counts are still chars / 4), PyPI and GitHub.
- **Trying the demo's live path without a key:** point it at a local mock with
  `OPENAI_BASE_URL=http://127.0.0.1:<port>/v1` and a dummy `OPENAI_API_KEY`. Agno leaves
  `base_url` unset, so the SDK reads the variable.
  `test_the_demos_openai_agents_run_against_the_mock` does this.
- **`ANTHROPIC_BASE_URL`** is set for the Claude Code harness itself. The mock tests pass
  `base_url` explicitly, so they never use it.
- **Always pass `telemetry=False`** to Agents; Agno sends telemetry by default.
- **`agent.get_last_run_output()` needs a `db`.**
- **Streaming:** with `stream=True`, `RunCompletedEvent.tools` is `None`; collect
  `ToolCallCompletedEvent`s instead (see `run()` in `tests/test_lazy_tools.py`).
- **The demo's `main()` calls `logging.disable(logging.WARNING)`** for the whole process;
  the demo test re-enables logging afterwards.
