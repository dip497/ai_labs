# Lazy tool loading with OpenAI tool search

An [Agno](https://github.com/agno-agi/agno) agent with a large tool catalog, where a
tool's full schema only enters the model's context once the model needs it. It uses
OpenAI's hosted [tool search](https://developers.openai.com/api/docs/guides/tools-tool-search):

- Every function tool is sent with `defer_loading: true`, so the model sees only its name
  and description.
- A `{"type": "tool_search"}` tool lets the model load the tools it needs. OpenAI appends
  their schemas to the end of the context, so the `tools` list never changes and the
  prompt cache keeps hitting.

Agno 3.0.11 passes the `tool_search` dict through as is, but doesn't send `defer_loading`.
[`tool_search.py`](tool_search.py) adds it in a small `OpenAIResponses` subclass. The same
subclass makes Agno chain requests with `previous_response_id`, so OpenAI keeps the
tool search results between requests. Agno does that on its own only for `o3`, `o4-mini`
and `gpt-5*` model ids.

## Run it

```bash
cd agno-lazy-tools
export OPENAI_API_KEY=sk-...
uv run python tool_search.py
```

Or let the [`tool-search` workflow](../.github/workflows/tool-search.yml) run it. It uses
the `OPENAI_API_KEY` secret of this repository's `openaienv` environment, runs the example,
then compares input tokens with an agent that sends every tool schema up front.

It needs the Responses API and gpt-5.4 or later; the example uses `gpt-6-luna`.

## Notes

- Deferred functions still show their names and descriptions. Grouping them into OpenAI
  namespaces would hide those too, until the model searches.
- An earlier, larger version of this lab is in the git history, at commit `40c25b9`. It
  had a provider-agnostic `search_tools` meta-tool that hooks Agno's run loop, and a
  client-executed tool search mode.
