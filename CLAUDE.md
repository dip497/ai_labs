# ai_labs

Small experiments and POCs. Each lab is a self-contained uv project in its own directory.

## agno-lazy-tools

POC: lazy (on-demand) tool loading for Agno agents. **Read `agno-lazy-tools/NOTES.md`
first**: it has the current status, findings, and the next steps (live test with
`gpt-6-luna`, then a native OpenAI tool-search mode). `agno-lazy-tools/README.md` is the
write-up.

- Set up and test: `cd agno-lazy-tools && uv sync && uv run pytest`
- Demo: `uv run python examples/demo.py` (offline), or add
  `--model openai:gpt-6-luna` for a live run (needs `uv run --extra openai`)
- `agno==3.0.11` is pinned, because `lazy_tools/model.py` overrides Agno internals. Re-run
  the tests before bumping it.
- Pass `telemetry=False` to every `Agent`.
