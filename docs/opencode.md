# OpenCode + Headroom — Integration Guide

[OpenCode](https://opencode.ai) (`sst/opencode`) is an open-source terminal
coding agent. Headroom compresses the context OpenCode sends to its model —
tool outputs, file reads, grep/glob hit-lists, test logs, and LSP diagnostics —
before it reaches the provider, so you pay for far fewer prompt tokens with the
same answers.

```bash
pip install "headroom-ai[all]"
headroom wrap opencode             # starts the proxy, edits opencode.json, launches opencode
```

## How it works

Unlike Codex or Aider, OpenCode does **not** read `OPENAI_BASE_URL` /
`OPENAI_API_BASE` for its built-in providers — it is configured entirely through
`opencode.json`. So `headroom wrap opencode` routes traffic by overriding the
`baseURL` of OpenCode's built-in `anthropic` and `openai` providers to the local
proxy:

```
OpenCode  (opencode CLI / TUI)
  │  opencode.json → provider.{anthropic,openai}.options.baseURL
  ▼
Headroom proxy  (local — your data never leaves your machine)
  │  SmartCrusher compresses JSON tool output
  │  CacheAligner stabilises KV-cache prefixes
  ▼
Anthropic / OpenAI  (your existing model + key)
  ▼
Response (same answer, fewer billed tokens)
```

Your model selection is unchanged: `anthropic/claude-sonnet-4-5`,
`openai/gpt-4o`, etc. keep working — every request is just sent through Headroom
first. The edit `headroom wrap opencode` makes is exactly:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "anthropic": { "options": { "baseURL": "http://127.0.0.1:8787/v1" } },
    "openai":    { "options": { "baseURL": "http://127.0.0.1:8787/v1" } }
  }
}
```

(`http://127.0.0.1:8787/v1` carries a `/p/<project>` prefix when you launch from
a project directory, so dashboard savings are attributed per project.)

## Benchmark (local, no API key)

Token counts are Headroom's local estimate over realistic OpenCode tool output.
Reproduce with:

```bash
uv run python tests/test_opencode_compression.py
```

| Payload | Before | After | Saved |
|---|---:|---:|---:|
| Full OpenCode session (grep + read + tests + diagnostics) | 22,890 | 18,825 | **18%** |
| `grep` results (119 hits) | 5,945 | 3,617 | **39%** |
| LSP diagnostics (89) | 8,884 | 4,819 | **46%** |

Headroom's **SmartCrusher** targets the large, repetitive JSON blobs OpenCode's
tools emit (grep/glob hit-lists, LSP diagnostics, structured tool results).
These compress losslessly on key content — the failing test and its location
survive, so the agent can still act on them.

## Quick start

```bash
pip install "headroom-ai[all]"
cd your-project
headroom wrap opencode
```

`headroom wrap opencode`:

1. Starts the local Headroom proxy (port 8787 by default).
2. Sets up the CLI context tool — injects `rtk` guidance into `AGENTS.md` so
   OpenCode prefers token-optimized shell commands.
3. Merges the Headroom provider overrides into the project-local `opencode.json`
   (your existing config and keys are preserved; the pre-wrap file is backed up
   to `opencode.json.headroom-backup`).
4. Launches `opencode`, with all model traffic routed through the proxy.

Pass arguments through to OpenCode after `--`:

```bash
headroom wrap opencode -- run "fix the failing 503 retry test"
headroom wrap opencode --port 9999          # custom proxy port
headroom wrap opencode --no-context-tool    # skip the AGENTS.md rtk setup
```

## Library mode (no wrap)

Compress messages yourself, in any OpenAI-compatible flow:

```python
from headroom import compress

result = compress(messages, model="claude-sonnet-4-5-20250929")
print(result.tokens_before, "→", result.tokens_after)
send_to_model(result.messages)
```

## Undo

`headroom wrap opencode` edits the project-local `opencode.json`. Reverse it from
the same directory:

```bash
headroom unwrap opencode
```

This restores your original `opencode.json` byte-for-byte from the pre-wrap
backup (or strips just the Headroom provider overrides if the backup is gone),
and stops the local proxy.

## Limitations

- **Config-file routing only.** OpenCode ignores base-URL environment variables,
  so routing requires the `opencode.json` edit above. `headroom install opencode`
  (env-export mode) is intentionally not provided.
- **Built-in `anthropic` + `openai` providers.** Other providers (OpenRouter,
  Google, local models, custom providers) are not redirected. Point them at the
  proxy manually by setting their `options.baseURL` if they are OpenAI- or
  Anthropic-compatible.
- **Project-scoped.** The config lives in the directory you launch from. Run
  `headroom wrap opencode` (and `headroom unwrap opencode`) from your project
  root.
