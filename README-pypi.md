# Headroom (`headroom-ai`)

Compress tool outputs, logs, JSON and chat history before each LLM call, so the
model sees fewer input tokens and the same facts. Runs in-process, in Python,
with no API key and no model download. Originals are cached locally and
retrievable.

Measured: 21–57% fewer input tokens on four agent scenarios, 90%+ on repeated
JSON arrays and log lines, little on prose. Numbers and how to reproduce them:
https://docs.headroomlabs.ai/docs/benchmarks and the "Proof" section of
https://github.com/headroomlabs-ai/headroom

## Use it from a Python app that calls an LLM

```bash
pip install headroom-ai
```

```python
from headroom import compress

result = compress(messages, model="gpt-4o")  # OpenAI or Anthropic message dicts
result.messages  # same shape as the input, fewer tokens
result.tokens_before, result.tokens_after, result.tokens_saved
```

Pass `result.messages` to your existing client call instead of `messages`.
Nothing else changes. Error lines, stack traces, first and last items and
statistical outliers are kept; repeated and low-information items are dropped.

Zero-network check, no provider key needed:

```bash
HEADROOM_OFFLINE=1 python -c "from headroom import compress; log='\n'.join(f'INFO step {i} ok' for i in range(3000))+'\nFATAL disk full'; r=compress([{'role':'user','content':'What failed?'},{'role':'assistant','content':None,'tool_calls':[{'id':'c1','type':'function','function':{'name':'logs','arguments':'{}'}}]},{'role':'tool','tool_call_id':'c1','content':log}], model='gpt-4o'); print(r.tokens_before, '->', r.tokens_after, 'FATAL kept:', 'FATAL' in str(r.messages))"
```

Prints on the order of `20000 -> 95 FATAL kept: True`. `HEADROOM_OFFLINE=1`
blocks every connection Headroom would make on its own: the update check, model
and pricing-table downloads, and the anonymous usage beacon the proxy and MCP
server send by default (`HEADROOM_BEACON=off` turns off only the beacon).

FastAPI example with a `POST /chat` endpoint:
https://docs.headroomlabs.ai/docs/fastapi

## Use it around a coding agent (no code changes)

```bash
uv tool install --python 3.13 "headroom-ai[all]"   # CLI in its own env; pip install "headroom-ai[proxy]" also works
headroom wrap claude        # also: codex, grok, copilot, cursor, aider, opencode, cline, continue, goose, openhands, openclaw, vibe, omp, zcode
headroom proxy --port 8787  # or run the proxy and point any client at it
```

## Other entry points

- OpenAI, Anthropic, LiteLLM, LangChain, Agno, Strands, CrewAI, AutoGen adapters:
  https://docs.headroomlabs.ai/docs (Integrations)
- MCP server (`headroom_compress`, `headroom_retrieve`, `headroom_stats`):
  https://docs.headroomlabs.ai/docs/mcp
- TypeScript SDK (`npm install headroom-ai`, needs the proxy running):
  https://www.npmjs.com/package/headroom-ai
- Full docs index for agents: https://docs.headroomlabs.ai/llms.txt
- Source and issues: https://github.com/headroomlabs-ai/headroom

Extras: `[proxy]` adds the HTTP proxy and `headroom wrap`, `[mcp]` the MCP
server, `[ml]` the Kompress text model (torch), `[all]` everything. The base
install is enough for `compress()`. Requires Python 3.11+.

Apache 2.0. The import and CLI name is `headroom`; the PyPI distribution named
`headroom` is a different project.
