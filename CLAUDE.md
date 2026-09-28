# ai_labs

Small experiments and POCs. Each lab is a self-contained uv project in its own directory.
Keep labs as small as Agno's cookbook examples: one short, runnable script per idea.

## agno-lazy-tools

Lazy tool loading for Agno agents with OpenAI's hosted tool search:
`agno-lazy-tools/tool_search.py`, explained in `agno-lazy-tools/README.md`.

- Run: `cd agno-lazy-tools && uv run python tool_search.py` (needs `OPENAI_API_KEY`).
- The session sandbox's `OPENAI_API_KEY` is rejected by OpenAI (HTTP 401). The working key
  is the `OPENAI_API_KEY` secret of the GitHub environment `openaienv`, which the
  `.github/workflows/tool-search.yml` workflow uses. It runs on pushes to
  `claude/continue-6nsk22`.
- `agno==3.0.11` is pinned, because `tool_search.py` overrides two internal methods of
  `OpenAIResponses`.
- Pass `telemetry=False` to every `Agent`.
