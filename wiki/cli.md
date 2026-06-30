# CLI Reference

This page is the authoritative reference for the **Python Headroom CLI** exposed by the `headroom` console script.

## Global behavior

### Entry points

- Console script: `headroom`
- Python module entrypoint: `python -m headroom.cli`

### Global options

| Option | Scope | Meaning |
|---|---|---|
| `--help`, `-?` | root, groups, commands | Show help and exit |
| `--version`, `-v` | root only | Show the Headroom version and exit |

> `-v` is a **root-level version alias**. Inside subcommands such as `headroom wrap claude -v`, `-v` keeps its subcommand meaning (`--verbose`), not version.

## Command index

| Command | Purpose | Docker-native parity |
|---|---|---|
| `headroom install ...` | Install and manage persistent deployments | **python-native; Docker-native wrapper supports `persistent-docker` lifecycle subset** |
| `headroom proxy` | Run the Headroom proxy server | **native in container** |
| `headroom learn` | Learn from past tool-call failures | **native in container** |
| `headroom perf` | Summarize recent proxy performance | **native in container** |
| `headroom evals ...` | Run memory evaluation workflows | **native in container** |
| `headroom memory ...` | Inspect and manage stored memories | **native in container** |
| `headroom mcp ...` | Install, inspect, remove, or serve MCP integration | **native in container** |
| `headroom wrap claude` | Start proxy and launch Claude Code | **host-bridged** |
| `headroom wrap copilot` | Start proxy and launch GitHub Copilot CLI | **python-native only** |
| `headroom wrap codex` | Start proxy and launch Codex CLI | **host-bridged** |
| `headroom wrap aider` | Start proxy and launch Aider | **host-bridged** |
| `headroom wrap cursor` | Start proxy and print Cursor config guidance | **host-bridged** |
| `headroom wrap openclaw` | Install and configure the OpenClaw plugin | **host-bridged** |
| `headroom unwrap openclaw` | Disable the Headroom OpenClaw plugin | **host-bridged** |

## Captured `--help` output

The sections below capture the current top-level help output from the live CLI.

### `headroom --help`

```text
Usage: headroom [OPTIONS] COMMAND [ARGS]...

  Headroom - The Context Optimization Layer for LLM Applications.

  Manage memories, run the optimization proxy, and analyze metrics.

  Examples:
      headroom proxy              Start the optimization proxy
      headroom memory list        List stored memories
      headroom memory stats       Show memory statistics

Options:
  -v, --version  Show the version and exit.
  -?, --help     Show this message and exit.

Commands:
  evals   Memory evaluation commands.
  install Install and manage persistent Headroom deployments.
  learn   Learn from past tool call failures to prevent future ones.
  mcp     MCP server for Claude Code integration.
  memory  Manage memories stored in Headroom.
  perf    Analyze proxy performance from logs.
  proxy   Start the optimization proxy server.
  unwrap  Undo durable Headroom wrapping for supported tools.
  wrap    Wrap CLI tools to run through Headroom.
```

### Top-level command help snapshots

<details>
<summary><code>headroom proxy --help</code></summary>

```text
Usage: headroom proxy [OPTIONS]

  Start the optimization proxy server.

  Examples:
      headroom proxy                    Start proxy on port 8787
      headroom proxy --port 8080        Start proxy on port 8080
      headroom proxy --no-optimize      Passthrough mode (no optimization)

  Usage with Claude Code:
      ANTHROPIC_BASE_URL=http://localhost:8787 claude

  Usage with OpenAI-compatible clients:
      OPENAI_BASE_URL=http://localhost:8787/v1 your-app
```

</details>

<details>
<summary><code>headroom learn --help</code></summary>

```text
Usage: headroom learn [OPTIONS]

  Learn from past tool call failures to prevent future ones.
```

</details>

<details>
<summary><code>headroom perf --help</code></summary>

```text
Usage: headroom perf [OPTIONS]

  Analyze proxy performance from logs.
```

</details>

<details>
<summary><code>headroom evals --help</code></summary>

```text
Usage: headroom evals [OPTIONS] COMMAND [ARGS]...

  Memory evaluation commands.

Commands:
  memory     Run LoCoMo memory evaluation benchmark.
  memory-v2  Run LoCoMo V2 evaluation with LLM-controlled memory tools.
```

</details>

<details>
<summary><code>headroom memory --help</code></summary>

```text
Usage: headroom memory [OPTIONS] COMMAND [ARGS]...

  Manage memories stored in Headroom.

Commands:
  delete  Delete one or more memories by ID.
  edit    Edit a memory's content or importance.
  export  Export all memories to JSON.
  import  Import memories from a JSON file.
  list    List stored memories with optional filters.
  prune   Prune memories matching specified criteria.
  purge   Delete ALL memories from the database.
  show    Show full details of a single memory.
  stats   Show memory store statistics.
```

</details>

<details>
<summary><code>headroom mcp --help</code></summary>

```text
Usage: headroom mcp [OPTIONS] COMMAND [ARGS]...

  MCP server for Claude Code integration.

Commands:
  install    Install Headroom MCP server into Claude Code config.
  serve      Start the MCP server (called by Claude Code).
  status     Check Headroom MCP configuration status.
  uninstall  Remove Headroom MCP server from Claude Code config.
```

</details>

<details>
<summary><code>headroom install --help</code></summary>

```text
Usage: headroom install [OPTIONS] COMMAND [ARGS]...

  Install and manage persistent Headroom deployments.

Options:
  -?, --help  Show this message and exit.

Commands:
  apply    Install a persistent Headroom deployment.
  remove   Remove a persistent deployment and undo managed config.
  restart  Restart a persistent deployment.
  start    Start a persistent deployment.
  status   Show persistent deployment status.
  stop     Stop a persistent deployment.
```

</details>

<details>
<summary><code>headroom wrap --help</code></summary>

```text
Usage: headroom wrap [OPTIONS] COMMAND [ARGS]...

  Wrap CLI tools to run through Headroom.

Commands:
  aider     Launch aider through Headroom proxy.
  claude    Launch Claude Code through Headroom proxy.
  copilot   Launch GitHub Copilot CLI through Headroom proxy.
  codex     Launch OpenAI Codex CLI through Headroom proxy.
  cursor    Start Headroom proxy for use with Cursor.
  openclaw  Install and configure Headroom OpenClaw plugin in one command.
```

</details>

<details>
<summary><code>headroom unwrap --help</code></summary>

```text
Usage: headroom unwrap [OPTIONS] COMMAND [ARGS]...

  Undo durable Headroom wrapping for supported tools.

Commands:
  openclaw  Disable the Headroom OpenClaw plugin and restore the legacy engine slot.
```

</details>

## `headroom proxy`

Start the optimization proxy server.

```bash
headroom proxy
headroom proxy --port 8787
headroom proxy --mode cache
```

| Option | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Host interface to bind |
| `--port`, `-p` | `8787` | Port to bind |
| `--mode` | runtime default | Optimization mode: `token`, `cache`, `token_mode`, `cache_mode`, `token_savings`, `cost_savings`, `token_headroom` |
| `--no-optimize` | off | Disable optimization and operate in passthrough mode |
| `--no-cache` | off | Disable semantic caching |
| `--no-rate-limit` | off | Disable rate limiting |
| `--retry-max-attempts` | runtime default `3` | Maximum upstream retry attempts |
| `--request-timeout-seconds` | runtime default `300` | Request timeout in seconds |
| `--connect-timeout-seconds` | runtime default `10` | Upstream connection timeout |
| `--anthropic-pre-upstream-concurrency` | auto `max(2, min(8, cpu_count))` | Cap simultaneous pre-upstream work on `/v1/messages` (body read, deep copy, first compression stage, memory-context lookup, upstream connect). `0` or negative disables (unbounded); any positive integer is honoured verbatim. Prevents cold-start replay storms from starving `/livez`, `/readyz`, and new Codex WS opens. |
| `--anthropic-pre-upstream-acquire-timeout-seconds` | `15.0` | Fail fast when the Anthropic pre-upstream queue is saturated. Requests that wait longer return `503` with `Retry-After` instead of parking indefinitely. |
| `--anthropic-pre-upstream-memory-context-timeout-seconds` | `2.0` | Fail-open timeout for Anthropic memory-context lookup while the request still holds a pre-upstream slot. |
| `--log-file` | unset | JSONL log output path |
| `--budget` | unset | Daily USD budget limit |
| `--no-code-aware` | off | Disable AST-aware code compression |
| `--code-aware` | off | Enable code-aware compression in the proxy (env: HEADROOM_CODE_AWARE_ENABLED) |
| `--no-read-lifecycle` | off | Disable stale/superseded read compression |
| `--no-ccr-inject-tool` | off | Disable injecting the `headroom_retrieve` tool |
| `--no-ccr-marker` | off | Disable adding retrieval markers to compressed output |
| `--no-ccr-proactive-expansion` | off | Disable proactive CCR context expansion |
| `--memory` | off | Enable persistent user memory |
| `--memory-db-path` | `""` | Override memory DB path (help text: `{cwd}/.headroom/memory.db`) |
| `--no-memory-tools` | off | Disable automatic memory tool injection |
| `--no-memory-context` | off | Disable automatic memory context injection |
| `--memory-top-k` | `10` | Number of memories to inject |
| `--learn` | off | Enable live traffic learning |
| `--no-learn` | off | Explicitly disable traffic learning |
| `--backend` | `anthropic` | Backend: `anthropic`, `bedrock`, `openrouter`, `anyllm`, or `litellm-*` |
| `--anyllm-provider` | `openai` | Provider name for `anyllm` |
| `--anthropic-api-url` | unset | Custom Anthropic passthrough API URL |
| `--openai-api-url` | unset | Custom OpenAI passthrough API URL |
| `--gemini-api-url` | unset | Custom Gemini passthrough API URL |
| `--region` | `us-west-2` | Cloud region for Bedrock / Vertex / related backends |
| `--bedrock-region` | unset | Deprecated Bedrock region override |
| `--bedrock-profile` | unset | AWS profile name for Bedrock |
| `--telemetry` | off | Opt in to anonymous usage telemetry (off by default) |
| `--no-telemetry` | off | Force anonymous usage telemetry off (already the default) |

Notes:

- `--learn` implies memory unless `--no-learn` is also set.
- Proxy startup can also read environment variables such as `HEADROOM_HOST`, `HEADROOM_PORT`, `HEADROOM_BUDGET`, `HEADROOM_MODE`, `HEADROOM_ANYLLM_PROVIDER`, `HEADROOM_ANTHROPIC_PRE_UPSTREAM_CONCURRENCY`, `HEADROOM_ANTHROPIC_PRE_UPSTREAM_ACQUIRE_TIMEOUT_SECONDS`, `HEADROOM_REQUEST_TIMEOUT`, `HEADROOM_ANTHROPIC_PRE_UPSTREAM_MEMORY_CONTEXT_TIMEOUT_SECONDS`, `ANTHROPIC_TARGET_API_URL`, `OPENAI_TARGET_API_URL`, and `GEMINI_TARGET_API_URL`. CLI flags take precedence over environment variables.
- The default Anthropic pre-upstream cap is intentionally conservative for CPU/ONNX-heavy work. Larger containers may want to raise it after checking the resolved runtime values on `/readyz` or `/debug/warmup`.

See also: [Proxy Server](proxy.md), [Configuration](configuration.md)

## `headroom learn`

Learn from past tool-call failures and produce agent guidance.

```bash
headroom learn
headroom learn --apply
headroom learn --agent codex --all
```

| Option | Default | Meaning |
|---|---|---|
| `--project` | current project resolution | Target project path |
| `--all` | off | Analyze all discovered projects |
| `--apply` | off | Write recommendations instead of dry-run output |
| `--agent` | `auto` | Agent source: `auto`, built-ins (`claude`, `codex`, `gemini`), or plugin-provided names |
| `--model` | auto-detect | LLM model used for analysis |

Notes:

- `--agent auto` scans all detected agent data sources.
- If `--project` is omitted, Headroom resolves from the current directory upward.
- External agent integrations register through the `headroom.learn_plugin` entry point.

See also: [Failure Learning](learn.md)

## `headroom perf`

Summarize recent proxy performance from the local proxy log.

```bash
headroom perf
headroom perf --hours 24
headroom perf --raw
```

| Option | Default | Meaning |
|---|---|---|
| `--hours` | `168.0` | Time window in hours |
| `--raw` | off | Print raw PERF records instead of the summarized report |

The command reads `${HEADROOM_WORKSPACE_DIR}/logs/proxy.log` (defaults
to `~/.headroom/logs/proxy.log` — see the
[Filesystem Contract](filesystem-contract.md)).

## `headroom evals`

Memory evaluation command group.

### `headroom evals memory`

Run the LoCoMo memory evaluation benchmark.

```bash
headroom evals memory -n 3
headroom evals memory --answer-model gpt-4o --llm-judge
```

| Option | Default | Meaning |
|---|---|---|
| `--n-conversations`, `-n` | all available | Number of conversations to evaluate |
| `--categories` | benchmark default | Comma-separated categories |
| `--include-adversarial` | off | Include category 5 / unanswerable questions |
| `--top-k` | `10` | Memories retrieved per question |
| `--f1-threshold` | `0.5` | Threshold for correctness |
| `--answer-model` | unset | Model for answer generation |
| `--llm-judge` | off | Use LLM-as-judge scoring |
| `--judge-provider` | `litellm` | Judge provider: `openai`, `anthropic`, `litellm`, `simple` |
| `--judge-model` | `gpt-4o` | Judge model |
| `--output`, `-o` | unset | Save JSON results to a path |
| `--no-extract` | off | Disable LLM memory extraction |
| `--extraction-model` | `gpt-4o-mini` | Memory extraction model |
| `--pass-all` | off | Require all checks to pass |
| `--parallel` | `10` | Parallel worker count |
| `--debug` | off | Enable debug output |

### `headroom evals memory-v2`

Run the V2 memory evaluation flow with LLM-controlled tools.

```bash
headroom evals memory-v2
headroom evals memory-v2 --save-model gpt-4o-mini --llm-judge
```

| Option | Default | Meaning |
|---|---|---|
| `--n-conversations`, `-n` | all available | Number of conversations to evaluate |
| `--categories` | benchmark default | Comma-separated categories |
| `--include-adversarial` | off | Include adversarial questions |
| `--f1-threshold` | `0.5` | Threshold for correctness |
| `--save-model` | `gpt-4o-mini` | Model used when persisting memories |
| `--answer-model` | `gpt-4o` | Answer model |
| `--max-results` | `10` | Maximum tool results |
| `--no-graph` | off | Disable graph usage |
| `--llm-judge` | off | Use LLM-as-judge scoring |
| `--judge-model` | `gpt-4o` | Judge model |
| `--output`, `-o` | unset | Save JSON results |
| `--parallel` | `5` | Parallel worker count |
| `--debug` | off | Enable debug output |

Hidden compatibility shims exist for older command paths:

- `headroom memory-eval`
- `headroom memory-eval-v2`

These are intentionally omitted from normal usage docs.

## `headroom memory`

Memory management command group. This group is only registered when the optional memory dependencies import successfully.

### `headroom memory list`

```bash
headroom memory list
headroom memory list --scope USER --since 7d
headroom memory list -q "budget"
```

| Option | Default | Meaning |
|---|---|---|
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--limit`, `-n` | `50` | Maximum memories to show |
| `--session`, `-s` | unset | Filter by session ID |
| `--scope` | unset | `USER`, `SESSION`, `AGENT`, or `TURN` |
| `--since` | unset | Age filter using duration syntax such as `7d`, `2w`, `1m` |
| `--search`, `-q` | unset | Content search query |

### `headroom memory show <memory_id>`

```bash
headroom memory show 1234abcd
headroom memory show 1234abcd --json
```

| Argument / option | Default | Meaning |
|---|---|---|
| `memory_id` | required | Full or partial memory ID |
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--json` | off | Emit raw JSON |

### `headroom memory stats`

```bash
headroom memory stats
```

| Option | Default | Meaning |
|---|---|---|
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |

### `headroom memory edit <memory_id>`

```bash
headroom memory edit 1234abcd --content "Updated note"
headroom memory edit 1234abcd --importance 0.9
```

| Argument / option | Default | Meaning |
|---|---|---|
| `memory_id` | required | Full or partial memory ID |
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--content`, `-c` | unset | New memory content |
| `--importance`, `-i` | unset | New importance score (`0.0` to `1.0`) |

At least one of `--content` or `--importance` is required.

### `headroom memory delete <memory_ids...>`

```bash
headroom memory delete 1234abcd 5678efgh
headroom memory delete 1234abcd --force
```

| Argument / option | Default | Meaning |
|---|---|---|
| `memory_ids...` | required | One or more memory IDs |
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--force`, `-f` | off | Skip confirmation |

### `headroom memory prune`

```bash
headroom memory prune --older-than 30d --dry-run
headroom memory prune --scope SESSION --force
```

| Option | Default | Meaning |
|---|---|---|
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--older-than` | unset | Age threshold |
| `--scope` | unset | Scope filter: `USER`, `SESSION`, `AGENT`, `TURN` |
| `--low-importance` | unset | Importance cutoff |
| `--session`, `-s` | unset | Session ID filter |
| `--dry-run` | off | Show what would be removed |
| `--force`, `-f` | off | Skip confirmation |

At least one filter is required. Filters combine with **AND** semantics.

### `headroom memory purge`

```bash
headroom memory purge --confirm
```

| Option | Default | Meaning |
|---|---|---|
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--confirm` | off | Required confirmation flag |

### `headroom memory export`

```bash
headroom memory export
headroom memory export --output export.json
```

| Option | Default | Meaning |
|---|---|---|
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--output`, `-o` | stdout | Output path |

### `headroom memory import <file>`

```bash
headroom memory import export.json
headroom memory import export.json --force
```

| Argument / option | Default | Meaning |
|---|---|---|
| `file` | required | JSON file containing exported memories |
| `--db-path` | `./.headroom/memory.db` if present, else `~/.headroom/memory.db` | Memory database path |
| `--force`, `-f` | off | Skip confirmation |

The import expects a JSON array. Malformed entries are skipped.

## `headroom mcp`

Manage the Headroom MCP server integration.

### `headroom mcp install`

```bash
headroom mcp install
headroom mcp install --proxy-url http://127.0.0.1:9000
```

| Option | Default | Meaning |
|---|---|---|
| `--proxy-url` | `http://127.0.0.1:8787` | Proxy URL written into MCP config |
| `--force` | off | Overwrite an existing Headroom MCP config |

### `headroom mcp uninstall`

```bash
headroom mcp uninstall
```

This removes the Headroom MCP server entry from the Claude configuration.

### `headroom mcp status`

```bash
headroom mcp status
```

This inspects MCP SDK availability, Claude config state, and proxy reachability.

### `headroom mcp serve`

```bash
headroom mcp serve
headroom mcp serve --proxy-url http://127.0.0.1:9000 --debug
```

| Option | Default | Meaning |
|---|---|---|
| `--proxy-url` | `http://127.0.0.1:8787` | Proxy URL (also reads `HEADROOM_PROXY_URL`) |
| `--direct` | off | Disable stdio transport wrapping |
| `--debug` | off | Enable debug logging |

`serve` is part of the public CLI, but it is usually consumed by MCP host tooling rather than by humans directly.

See also: [MCP Tools](mcp.md)

## `headroom install`

Install and manage persistent local Headroom deployments.

### `headroom install apply --help`

```text
Usage: headroom install apply [OPTIONS]

  Install a persistent Headroom deployment.

Options:
  --preset [persistent-service|persistent-task|persistent-docker]
                                  Persistent runtime preset to install.
                                  [default: persistent-service]
  --runtime [python|docker]       Runtime used to execute Headroom for
                                  service/task modes.  [default: python]
  --scope [provider|user|system]  Where to apply persistent configuration.
                                  [default: user]
  --providers [auto|all|manual]   Target selection mode for direct tool
                                  configuration.  [default: auto]
  --target [claude|copilot|codex|aider|cursor|openclaw]
                                  Tool target to configure when --providers
                                  manual is used.
  --profile TEXT                  Deployment profile name.  [default: default]
  -p, --port INTEGER              Persistent proxy port.  [default: 8787]
  --backend TEXT                  Proxy backend for the persistent runtime.
                                  [default: anthropic]
  --anyllm-provider TEXT          Provider for any-llm backends when --backend
                                  anyllm is used.
  --region TEXT                   Cloud region for Bedrock / Vertex style
                                  backends.
  --mode TEXT                     Proxy optimization mode.  [default: token]
  --memory                        Enable persistent memory in the proxy runtime.
  --telemetry                     Opt in to anonymous telemetry in the runtime
                                  (off by default).
  --no-telemetry                  Force anonymous telemetry off in the runtime
                                  (already the default).
  --image TEXT                    Docker image to use when runtime=docker or
                                  preset=persistent-docker.  [default:
                                  ghcr.io/chopratejas/headroom:latest]
  -?, --help                      Show this message and exit.
```

### `headroom install apply`

```bash
headroom install apply --preset persistent-service --providers auto
headroom install apply --preset persistent-task --providers manual --target claude --target codex
headroom install apply --preset persistent-docker --scope user
```

| Option | Default | Meaning |
|---|---|---|
| `--preset` | `persistent-service` | Lifecycle preset: `persistent-service`, `persistent-task`, or `persistent-docker` |
| `--runtime` | `python` | Runtime used for service/task installs: `python` or `docker` |
| `--scope` | `user` | Config scope: `provider`, `user`, or `system` |
| `--providers` | `auto` | Target selection mode: `auto`, `all`, or `manual` |
| `--target` | repeatable | Tool target used with `--providers manual` |
| `--profile` | `default` | Deployment profile name |
| `--port`, `-p` | `8787` | Persistent proxy port |
| `--backend` | `anthropic` | Backend for the managed runtime |
| `--anyllm-provider` | unset | Provider name used with `--backend anyllm` |
| `--region` | unset | Cloud region override |
| `--mode` | `token` | Proxy optimization mode |
| `--memory` | off | Enable persistent memory in the managed runtime |
| `--telemetry` | off | Opt in to anonymous telemetry (off by default) |
| `--no-telemetry` | off | Force anonymous telemetry off (already the default) |
| `--image` | `ghcr.io/chopratejas/headroom:latest` | Docker image for Docker-backed installs |

`apply` stores a manifest under
`${HEADROOM_WORKSPACE_DIR}/deploy/<profile>/manifest.json` (default
`~/.headroom/deploy/<profile>/manifest.json`), applies managed tool
configuration, starts the chosen runtime, and waits for `readyz`.

Docker-native host wrappers expose a narrower `headroom install` subset for `persistent-docker` only: `apply`, `status`, `start`, `stop`, `restart`, and `remove`. Those wrapper flows preserve the same port and manifest behavior, but they intentionally reject `persistent-service`, `persistent-task`, and provider mutation flags like `--scope`, `--providers`, and `--target`.

### `headroom install status`

```bash
headroom install status
headroom install status --profile default
```

Shows the stored profile, preset, runtime, supervisor kind, scope, port, runtime status, readiness, and backend from `/health`.

### `headroom install start`

```bash
headroom install start
headroom install start --profile default
```

Starts a previously installed deployment profile without reapplying mutations.

### `headroom install stop`

```bash
headroom install stop
```

Stops the managed runtime for an installed deployment profile.

### `headroom install restart`

```bash
headroom install restart
```

Stops and starts the selected deployment profile.

### `headroom install remove`

```bash
headroom install remove
```

Stops the runtime, removes installed supervisor artifacts, reverts managed configuration changes, and deletes the stored manifest.

See also: [Persistent Installs](persistent-installs.md)

## `headroom wrap`

Wrap external coding tools so their traffic flows through Headroom.

### Shared semantics

- `--port`, when available, defaults to `8787`
- `--no-proxy` skips proxy startup and assumes an existing proxy
- `--learn` enables live traffic learning
- `-v`, `--verbose` means **verbose output**
- Hidden `--prepare-only` exists for internal Docker-native bridge flows and is intentionally omitted from normal usage

### `headroom wrap claude`

```bash
headroom wrap claude
headroom wrap claude --resume <session-id>
headroom wrap claude --port 9999
```

| Option / arg | Default | Meaning |
|---|---|---|
| `--port`, `-p` | `8787` | Proxy port |
| `--no-rtk` | off | Skip `rtk` installation and hook registration |
| `--no-proxy` | off | Reuse an existing proxy |
| `--learn` | off | Enable live traffic learning |
| `--verbose`, `-v` | off | Verbose output |
| `claude_args...` | passthrough | Additional Claude Code arguments |

Requires the `claude` binary on the host.

### `headroom wrap codex`

```bash
headroom wrap codex
headroom wrap codex -- "fix the bug"
headroom wrap codex --backend anyllm --anyllm-provider groq
```

| Option / arg | Default | Meaning |
|---|---|---|
| `--port`, `-p` | `8787` | Proxy port |
| `--no-rtk` | off | Skip `rtk` installation and `AGENTS.md` injection |
| `--no-proxy` | off | Reuse an existing proxy |
| `--learn` | off | Enable live traffic learning |
| `--backend` | unset | Proxy backend override |
| `--anyllm-provider` | unset | `anyllm` provider override |
| `--region` | unset | Cloud region override |
| `--verbose`, `-v` | off | Verbose output |
| `codex_args...` | passthrough | Additional Codex CLI arguments |

Requires the `codex` binary on the host.

### `headroom wrap copilot`

```bash
headroom wrap copilot -- --model claude-sonnet-4-20250514
headroom wrap copilot --backend anyllm --anyllm-provider groq -- --model gpt-4o
```

| Option / arg | Default | Meaning |
|---|---|---|
| `--port`, `-p` | `8787` | Proxy port |
| `--no-rtk` | off | Skip `rtk` installation and GitHub Copilot instructions injection |
| `--no-proxy` | off | Reuse an existing proxy |
| `--learn` | off | Enable live traffic learning |
| `--backend` | unset | Proxy backend override |
| `--anyllm-provider` | unset | `anyllm` provider override |
| `--region` | unset | Cloud region override |
| `--provider-type` | `auto` | Force Copilot BYOK provider type (`anthropic` or `openai`) |
| `--wire-api` | unset | OpenAI wire API override for OpenAI-style backends |
| `--verbose`, `-v` | off | Verbose output |
| `copilot_args...` | passthrough | Additional Copilot CLI arguments |

Requires the `copilot` binary on the host. When a matching persistent deployment exists on the requested port, `wrap copilot` reuses or recovers it before falling back to an ephemeral proxy.

### `headroom wrap aider`

```bash
headroom wrap aider
headroom wrap aider -- --model gpt-4o
headroom wrap aider --backend litellm-vertex --region us-central1
```

| Option / arg | Default | Meaning |
|---|---|---|
| `--port`, `-p` | `8787` | Proxy port |
| `--no-rtk` | off | Skip `rtk` installation and `CONVENTIONS.md` injection |
| `--no-proxy` | off | Reuse an existing proxy |
| `--learn` | off | Enable live traffic learning |
| `--backend` | unset | Proxy backend override |
| `--anyllm-provider` | unset | `anyllm` provider override |
| `--region` | unset | Cloud region override |
| `--verbose`, `-v` | off | Verbose output |
| `aider_args...` | passthrough | Additional Aider arguments |

Requires the `aider` binary on the host.

### `headroom wrap cursor`

```bash
headroom wrap cursor
headroom wrap cursor --port 9999
headroom wrap cursor --no-rtk
```

| Option | Default | Meaning |
|---|---|---|
| `--port`, `-p` | `8787` | Proxy port |
| `--no-rtk` | off | Skip `rtk` installation and `.cursorrules` injection |
| `--no-proxy` | off | Reuse an existing proxy |
| `--learn` | off | Enable live traffic learning |
| `--verbose`, `-v` | off | Verbose output |

This command prints Cursor configuration instructions and waits while the proxy stays up. It does **not** launch Cursor directly.

### `headroom wrap openclaw`

```bash
headroom wrap openclaw
headroom wrap openclaw --plugin-path ./plugins/openclaw
```

| Option | Default | Meaning |
|---|---|---|
| `--plugin-path` | unset | Local plugin source directory |
| `--plugin-spec` | `headroom-ai/openclaw` | NPM plugin spec |
| `--skip-build` | off | Skip local `npm install` / build steps |
| `--copy` | off | Copy plugin instead of linked install |
| `--proxy-port` | `8787` | Headroom proxy port |
| `--startup-timeout-ms` | `20000` | Proxy startup timeout |
| `--gateway-provider-id` | repeatable | OpenClaw provider IDs routed through Headroom |
| `--python-path` | unset | Python launcher override |
| `--no-auto-start` | off | Disable plugin auto-start behavior |
| `--no-restart` | off | Do not restart the OpenClaw gateway |
| `--verbose`, `-v` | off | Verbose output |

Requires the `openclaw` binary on the host, and local-source mode may also require `npm`. In Docker-native mode, the installed host wrapper drives the host `openclaw` CLI while the plugin auto-starts the host `headroom` wrapper from `PATH`.

## `headroom unwrap`

Undo durable wrapping for supported tools.

### `headroom unwrap openclaw`

```bash
headroom unwrap openclaw
headroom unwrap openclaw --no-restart
```

| Option | Default | Meaning |
|---|---|---|
| `--no-restart` | off | Do not restart the OpenClaw gateway |
| `--verbose`, `-v` | off | Verbose output |

This disables the Headroom OpenClaw plugin and restores the legacy context engine slot.

## Docker-native parity matrix

This matrix compares the **Python CLI contract** to the Docker-native host wrapper added in this branch.

Legend:

- **native in container** — the command runs entirely inside the Headroom container
- **host-bridged** — Headroom runs in Docker, but the wrapped external tool still runs on the host

| Command path | Python CLI | Docker-native wrapper | Parity |
|---|---|---|---|
| `headroom proxy` | native | native in container | full |
| `headroom learn` | native | native in container | full |
| `headroom perf` | native | native in container | full |
| `headroom evals memory` | native | native in container | full |
| `headroom evals memory-v2` | native | native in container | full |
| `headroom memory ...` | native (when memory deps are available) | native in container | full |
| `headroom mcp install` | native | native in container | full |
| `headroom mcp uninstall` | native | native in container | full |
| `headroom mcp status` | native | native in container | full |
| `headroom mcp serve` | native | native in container | full |
| `headroom install apply|status|start|stop|restart|remove` | native | Docker-native wrapper for `persistent-docker`; compose remains an alternative | partial |
| `headroom wrap claude` | native | host-bridged | partial |
| `headroom wrap copilot` | native | not implemented in Docker-native wrapper | none |
| `headroom wrap codex` | native | host-bridged | partial |
| `headroom wrap aider` | native | host-bridged | partial |
| `headroom wrap cursor` | native | host-bridged | partial |
| `headroom wrap openclaw` | native | host-bridged | partial |
| `headroom unwrap openclaw` | native | host-bridged | partial |

For the Docker-native execution model itself, see [Docker-Native Install](docker-install.md). For persistent service/task/docker lifecycle management, see [Persistent Installs](persistent-installs.md).

## Hidden and compatibility-only command paths

These exist in code but are intentionally excluded from normal user docs:

- `headroom memory-eval`
- `headroom memory-eval-v2`
- hidden internal `--prepare-only` flags on `wrap` subcommands

If you are documenting operational behavior or debugging internal wrapper flows, refer to the implementation in `headroom/cli/wrap.py`.
