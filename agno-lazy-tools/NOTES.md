# Working notes: agno-lazy-tools

Context for picking this work up in a new session. Last updated 2026-09-27.
`README.md` is the write-up; this file is status, findings and next steps.

## Status

- The POC is complete and pushed on branch `claude/lazy-load-tools-poc-zd6pcs`. No PR has
  been opened.
- `uv run pytest`: 22 tests pass, all offline. `uv run python examples/demo.py` runs offline.
- **Not yet done: any run against a live model.** The sandbox's egress proxy denied
  `api.openai.com` (HTTP 403 on CONNECT), so a real OpenAI SDK call with the key failed with
  `APIConnectionError -> ProxyError('403 Forbidden')` before reaching OpenAI. The evidence
  so far is the scripted model plus Agno's real `OpenAIChat` / `Claude` adapters against a
  local mock API (`tests/test_providers.py`).
- The user is changing the environment's network access to allow OpenAI, and wants live
  testing with **`gpt-6-luna`**. `OPENAI_API_KEY` is set in the environment.

## Next steps (in order)

1. **Check the API is reachable.**
   `curl -sS https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" -o /dev/null -w "%{http_code}\n"`
   should print `200`. A `403` from the proxy means it is still blocked;
   `curl -sS "$HTTPS_PROXY/__agentproxy/status"` lists recent denials.
2. **Run the live demo.**
   `uv run --extra openai python examples/demo.py --model openai:gpt-6-luna`
   It makes about 5 requests; gpt-6-luna costs $0.10 / $0.50 per 1M tokens in / out.
   Put the provider-reported input tokens (lazy vs eager) into the README "Results".
   The demo uses `OpenAIChat` (Chat Completions). If gpt-6-luna rejects that endpoint,
   switch `make_model` in `examples/demo.py` to `agno.models.openai.OpenAIResponses`.
3. **Build a native OpenAI tool-search mode** (the user's interest: no prompt-cache miss).
   See the research below. Plan:
   - Subclass `agno.models.openai.OpenAIResponses`. In `_format_tool_params`, add
     `"defer_loading": true` to deferred function tools, and add `{"type": "tool_search"}`.
   - Prefer *client-executed* search: answer the model's `tool_search_call` with
     `LazyTools.search()` results as a `tool_search_output` with the same `call_id`.
     *Hosted* search is the simpler alternative.
   - Agno must carry the `tool_search_call` / `tool_search_output` output items across
     turns (Responses output parsing lives in `agno/models/openai/responses.py`), or use
     `previous_response_id`.
   - Verify the exact wire format against the live API *before* writing mock tests: OpenAI's
     docs were blocked from the sandbox, so the details below come from secondary sources.
   - Keep the current client-side injection as the fallback for providers without native
     tool search.

## Key findings (Agno 3.0.11)

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
- **Upstream.**
  - [PR #7528](https://github.com/agno-agi/agno/pull/7528) (`DiscoverableTools`, patched
    `base.py`): closed unmerged 2026-09-16.
  - [PR #10574](https://github.com/agno-agi/agno/pull/10574) (lazy skill tools): closed
    unmerged 2026-09-25.
  - [Issue #5136](https://github.com/agno-agi/agno/issues/5136) (dynamic tools): open.

## Research: provider-native tool search

### OpenAI (Responses API only, gpt-5.4 and later)

- **Request:** `{"type": "tool_search"}` in `tools`, plus `"defer_loading": true` on
  functions, MCP servers, or functions inside a `namespace` tool. There is also an
  `additional_tools` input item.
- **Modes:** hosted (OpenAI runs the search), or client-executed (the model emits
  `tool_search_call`; the app returns `tool_search_output` with the same `call_id`).
- **Output items:** `tool_search_call` and `tool_search_output`, then ordinary
  `function_call`s.
- **Caching:** "Both hosted and client-executed tool search load tools at the end of the
  model's context window. This placement preserves the model's cache between requests."
  (Azure docs, as quoted in ellmer #1128.)
- **Agno gap:** `OpenAIResponses._format_tool_params` serializes Functions via
  `Function.to_dict()`, which drops `defer_loading`. Raw dict tools whose type isn't
  `"function"` pass through unchanged.
- **Sources:**
  - [OpenAI guide](https://developers.openai.com/api/docs/guides/tools-tool-search)
    (blocked from the sandbox)
  - [OpenHands #4083](https://github.com/OpenHands/software-agent-sdk/issues/4083)
  - [ellmer #1128](https://github.com/tidyverse/ellmer/issues/1128)
  - [pydantic-ai #4566](https://github.com/pydantic/pydantic-ai/issues/4566)

### Anthropic

- **Request:** `tool_search_tool_regex_20251119` or `tool_search_tool_bm25_20251119`, plus
  `defer_loading: true` on tools. At least one tool must stay non-deferred.
- **Agno gap:** Agno's Claude adapter (`format_tools_for_model` in
  `agno/utils/models/claude.py`) rebuilds function tools as `name` / `description` /
  `input_schema` (plus `strict`), which drops `defer_loading`.

## Sandbox gotchas

- **Blocked by the egress proxy:** `api.openai.com`, `developers.openai.com`,
  `learn.microsoft.com`, and `openaipublic.blob.core.windows.net`. The last one hosts
  tiktoken's encodings, so token counts fall back to chars / 4.
- **Reachable:** PyPI and GitHub.
- **`ANTHROPIC_BASE_URL`** is set for the Claude Code harness itself. The mock tests pass
  `base_url` explicitly, so they never use it.
- **Always pass `telemetry=False`** to Agents; Agno sends telemetry by default.
- **`agent.get_last_run_output()` needs a `db`.**
- **Streaming:** with `stream=True`, `RunCompletedEvent.tools` is `None`; collect
  `ToolCallCompletedEvent`s instead (see `run()` in `tests/test_lazy_tools.py`).
