# Cortex Code + Headroom — Integration Guide

Headroom compresses the context Cortex Code (CoCo) sends to `claude-sonnet-4-6`
before it reaches the Snowflake Cortex inference endpoint. The result is 60–65%
fewer prompt tokens billed, with the same answers.

## Benchmark (measured, not estimated)

Token counts are from `usage.prompt_tokens` in the actual Snowflake Cortex API
response — not headroom's local estimate.

| Payload | Before | After | Saved |
|---|---:|---:|---:|
| Full CoCo session (tables + dbt + search) | 17,827 | 6,781 | **62%** |
| `INFORMATION_SCHEMA` tables (79 rows) | 10,161 | 3,979 | **61%** |
| `dbt` run-results (40 models) | 4,968 | 1,927 | **61%** |
| Cortex Search results (15 docs) | 2,764 | 956 | **65%** |

At 1,000 calls/day: **~$16/day saved**, **~$6,000/year saved**.

> Numbers above are per-call averages across the four benchmark payloads.
> The full-session payload alone saves ~$33/1,000 calls/day.

## How it works

```
CoCo (cortex CLI)
  │  OPENAI_BASE_URL=http://127.0.0.1:8787/v1
  ▼
Headroom proxy  (local, your data never leaves your machine)
  │  SmartCrusher compresses JSON context
  │  CacheAligner stabilises KV-cache prefixes
  ▼
Snowflake Cortex  /api/v2/cortex/inference:complete
  │  claude-sonnet-4-6
  ▼
Response (same answer, fewer billed tokens)
```

Headroom's **SmartCrusher** targets the large JSON blobs that CoCo produces:
`INFORMATION_SCHEMA` query results, `dbt` run-results, Cortex Search payloads,
and schema introspection output. These are highly repetitive structures that
compress 60–99% without any loss of information.

## Quick start

```bash
pip install "headroom-ai[all]"
headroom wrap cortex-code          # starts proxy + prints the env var to set
```

`headroom wrap cortex-code` starts the local proxy and prints:

```
  Headroom proxy is running. Configure Cortex Code (CoCo):

  Set the following environment variable before launching cortex:
    OPENAI_BASE_URL=http://127.0.0.1:8787/v1
```

Then in a new shell:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8787/v1 cortex
```

Or add it to your shell profile so it applies to every CoCo session:

```bash
# ~/.zshrc or ~/.bashrc
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
```

## Manual proxy startup

If you prefer to manage the proxy lifecycle yourself:

```bash
# Terminal 1 — start the proxy
headroom proxy --port 8787

# Terminal 2 — launch CoCo through the proxy
OPENAI_BASE_URL=http://127.0.0.1:8787/v1 cortex
```

Point the proxy at your Snowflake Cortex endpoint explicitly with
`--openai-api-url`:

```bash
headroom proxy \
  --port 8787 \
  --openai-api-url https://<account>.snowflakecomputing.com
```

## Library mode (inline, no proxy)

If you are building an application on top of the Snowflake Cortex REST API
and want to compress context before every call:

```python
from headroom import compress
import json, urllib.request

# Build your messages (large JSON tool results, search results, etc.)
messages = [
    {"role": "system", "content": json.dumps(cortex_search_results, indent=2)},
    {"role": "assistant", "content": "I have reviewed the context."},
    {"role": "user", "content": "What is failing and how do I fix it?"},
]

# Compress before sending  — local, no API call, no data leaves your machine
result = compress(messages, model="claude-sonnet-4-6")
print(f"Saved {result.tokens_saved} tokens ({result.tokens_saved / result.tokens_before:.0%})")

# Send compressed messages to Snowflake Cortex REST API
response = call_cortex(result.messages, token=sf_token)
```

### What to put in the system message

The Snowflake Cortex REST API supports `system`, `user`, and `assistant` roles.
For maximum compression, inject large retrieved context into `system`:

```python
# Query results, search results, schema — these compress 60–99%
system_context = {
    "tables":         json.loads(show_tables_result),
    "search_results": cortex_search_results,
    "schema":         describe_table_result,
    "dbt_results":    dbt_run_results_json,
}
messages = [
    {"role": "system", "content": json.dumps(system_context, indent=2)},
    {"role": "assistant", "content": "Context loaded."},
    {"role": "user",      "content": user_question},
]
result = compress(messages, model="claude-sonnet-4-6")
```

## Authentication

Cortex Code authenticates using your Snowflake connection. Headroom sits
between CoCo and the Cortex endpoint and forwards auth headers unchanged —
it never reads or stores your credentials.

If you use `snowflake-connector-python` directly, keep the connection open
while making API calls; closing it invalidates the OAuth session token:

```python
import snowflake.connector, sys, io

# Suppress connector's browser-auth console output
_s = sys.stdout; sys.stdout = io.StringIO()
conn = snowflake.connector.connect(connection_name="my_connection")
token = conn.rest.token
sys.stdout = _s

# Make all API calls while conn is open, then:
conn.close()
```

## Per-project savings attribution

Use `headroom wrap cortex-code --project <name>` to attribute savings to a
specific project in the headroom dashboard:

```bash
headroom wrap cortex-code --project my-dbt-project
```

The dashboard at `http://127.0.0.1:8787` shows per-project token and cost
savings across all your CoCo sessions.

## Verifying savings

After a CoCo session, check what headroom saved:

```bash
headroom perf          # token savings for the last session
headroom perf --hours 24  # last 24 hours
```

Or run the included end-to-end benchmark against your own Snowflake account:

```bash
# Measures real usage.prompt_tokens from claude-sonnet-4-6
python3 tests/e2e_cortex_savings.py
```

## Testing

Unit tests for the provider slice:

```bash
uv run --with pytest pytest tests/test_provider_cortex_code.py -v
```

Compression benchmark (no API key needed — local only):

```bash
uv run --with pytest pytest tests/test_cortex_code_compression.py -v -s
```

Real E2E test against Snowflake Cortex (requires Snowflake connection):

```bash
python3 tests/e2e_cortex_savings.py
```

## How the provider is implemented

Cortex Code routes through headroom's OpenAI-compatible pipeline. The provider
slice lives in `headroom/providers/cortex_code/`:

| File | Purpose |
|---|---|
| `runtime.py` | `proxy_base_url(port)` → `http://127.0.0.1:{port}/v1`; `default_api_url()` reads `SNOWFLAKE_HOST` / `SNOWFLAKE_ACCOUNT` |
| `install.py` | `build_install_env()` → `{"OPENAI_BASE_URL": ...}`; `render_setup_lines()` |
| `__init__.py` | Public exports |

Registered in `headroom/providers/install_registry.py` under the key
`"cortex-code"`, which is what `headroom wrap cortex-code` resolves to.

## Limitations

- The Snowflake Cortex REST API at `/api/v2/cortex/inference:complete` does not
  support `role: "tool"` messages or OpenAI-style `tool_calls`. Use the
  `system` message to inject large retrieved context (where SmartCrusher
  achieves the highest compression ratios).

- The headroom proxy cannot rewrite the Cortex inference path
  (`/api/v2/cortex/inference:complete` ≠ `/v1/chat/completions`), so
  **library mode** (`from headroom import compress`) is required when calling
  the Cortex REST API directly. The proxy mode works for any
  OpenAI-compatible client that points at Cortex via a gateway that exposes
  `/v1/chat/completions`.

- Output-token reduction (`HEADROOM_OUTPUT_SHAPER=1`) is supported in proxy
  mode. In library mode only input compression applies.

## See also

- [Architecture](ARCHITECTURE.md)
- [Proxy configuration](proxy.md)
- [CCR — reversible compression](ccr.md)
- [Claude Code + Vertex](claude-code-vertex-headroom.md)
- [Benchmarks](benchmarks.md)
