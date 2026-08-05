# @headroom-ai/openclaw

Context compression plugin for [OpenClaw](https://github.com/openclaw/openclaw). Compresses tool outputs, code, logs, and structured data — 70-90% token savings with zero LLM calls.

## Install

Recommended one-command setup:

```bash
headroom wrap openclaw
```

Manual install:

```bash
pip install "headroom-ai[proxy]"
openclaw plugins install --dangerously-force-unsafe-install headroom-ai/openclaw
```

This plugin can auto-start a local `headroom proxy` when needed. OpenClaw treats process-launching plugins as unsafe by default, so `--dangerously-force-unsafe-install` is required even if you plan to use a remote proxy (the capability is declared at install time).

## Local Development Install (Detection-Friendly)

If you are testing from this repo, run npm install/build from the plugin directory so local launcher detection aligns with runtime paths. These linked installs are supported:

```bash
cd plugins/openclaw
npm install
npm run build
openclaw plugins install --dangerously-force-unsafe-install --link .
openclaw plugins install --dangerously-force-unsafe-install --link dist
```

From the repo root, install the plugin directory explicitly:

```bash
openclaw plugins install --dangerously-force-unsafe-install --link ./plugins/openclaw
```

Or, from inside `dist/`:

```bash
cd plugins/openclaw/dist
openclaw plugins install --dangerously-force-unsafe-install --link .
```

Why this matters:
- The plugin checks launchers in this order: PATH -> local npm bin -> global npm -> python.
- "local npm bin" means `plugins/openclaw/node_modules/.bin/headroom` relative to the source checkout.
- Using `--link dist` (or `--link .` from `dist/`) still keeps runtime code adjacent to the checkout, and launcher detection falls back to PATH/global/python if a local npm bin is not present under the installed root.
- `plugins/openclaw` also carries a no-op hook shim so OpenClaw's hook-pack fallback treats the path as valid instead of emitting a misleading `package.json missing openclaw.hooks` warning.
- If you install from a `.tgz`, local npm bin may not exist in the installed extension and detection will fall back to PATH/global/python.

## Configure

Install automatically selects the `contextEngine` slot for `headroom` on current OpenClaw releases. If you need to switch back manually, set `plugins.slots.contextEngine` to `"legacy"` or another engine id.

```json
{
  "plugins": {
    "entries": {
      "headroom": {
        "enabled": true,
        "config": {
          "proxyUrl": "http://127.0.0.1:8787"
        }
      }
    },
    "slots": {
      "contextEngine": "headroom"
    }
  }
}
```

`proxyUrl` is optional. If omitted, the plugin auto-detects on localhost:
- `http://127.0.0.1:<proxyPort>`
- `http://localhost:<proxyPort>`

Default `proxyPort` is `8787`. Auto-start is opt-in; in production, prefer an externally
managed proxy such as systemd with `proxyUrl` set and `autoStart: false`.

### Upstream gateway routing

By default, the plugin also rewrites the built-in `openai-codex` provider base URL to a verified active Headroom proxy at runtime. That means Codex provider traffic flows through Headroom, so `/stats` can observe real upstream request and cache activity instead of only local context compression.

This does not replace Headroom's existing Codex routing rules. The proxy already decides between `api.openai.com` and `chatgpt.com/backend-api/codex/responses` based on ChatGPT auth. The plugin change only points OpenClaw's provider config at the active proxy in memory and preserves the rest of the provider config.

You can also route additional provider ids such as `anthropic`, `github-copilot`, `google`, or `openrouter` through the same proxy:

```json
{
  "plugins": {
    "entries": {
      "headroom": {
        "enabled": true,
        "config": {
          "gatewayProviderIds": ["openai-codex", "anthropic", "github-copilot", "google", "openrouter"]
        }
      }
    }
  }
}
```

When `gatewayProviderIds` is set, it becomes the exact list the plugin rewrites in memory for the current gateway process.

For convenience, the plugin also accepts family aliases:
- `codex` -> `openai-codex`
- `claude` -> `anthropic`
- `copilot` -> `github-copilot`
- `gemini` -> `google`

When OpenClaw has already resolved a provider's upstream `baseUrl`, the plugin preserves protocol-specific path segments while swapping only the origin. That keeps provider families on the right proxy route:
- Codex / ChatGPT backend: `/backend-api`
- OpenAI-compatible providers: `/v1` or `/api/v1`
- GitHub Copilot Claude-family models: `/anthropic`
- Gemini: `/v1beta`

GitHub Copilot is a special case because OpenClaw can route it through either OpenAI Responses or Anthropic Messages depending on the selected model. The plugin only rewrites Copilot when OpenClaw has already resolved the upstream `baseUrl`, so it can preserve the correct `/v1` or `/anthropic` path instead of guessing.

The routing is intentionally lightweight and reversible:
- the plugin does not persist provider `baseUrl` changes back to `openclaw.json`
- disabling the plugin, clearing `gatewayProviderIds`, or restarting without Headroom restores OpenClaw's normal provider resolution
- if you want durable provider rewrites, use `headroom wrap openclaw` instead of relying on plugin install side effects

If you need to disable that behavior:

```json
{
  "plugins": {
    "entries": {
      "headroom": {
        "enabled": true,
        "config": {
          "routeCodexViaProxy": false
        }
      }
    }
  }
}
```

### Local proxy (auto-start)

When `proxyUrl` points to localhost (or is omitted), the plugin will auto-start `headroom proxy` if no running proxy is detected. Launch order:
1. `headroom` from `PATH`
2. local npm bin (`node_modules/.bin/headroom`)
3. global npm bin
4. Python module (`python -m headroom.cli proxy ...`)

If `pythonPath` is set, it is tried first in the Python fallback step.

Docker-native Headroom installs intentionally leave `pythonPath` unset so this launcher order prefers the installed host `headroom` wrapper on `PATH`, which then runs Headroom in Docker.

### Remote proxy (connect-only)

Point `proxyUrl` to any reachable Headroom instance:

```json
{
  "config": {
    "proxyUrl": "https://headroom.example.com:8787"
  }
}
```

Remote URLs are **connect-only** — the plugin probes the URL at startup and fails fast if the proxy is not reachable. No subprocess is spawned for remote addresses.

## Manual Proxy Setup

If you prefer to manage the proxy yourself (or are running a remote instance), start it before launching OpenClaw:

Python install:

```bash
pip install "headroom-ai[proxy]"
headroom proxy --host 127.0.0.1 --port 8787
```

NPM install:

```bash
npm install -g headroom-ai
headroom proxy --host 127.0.0.1 --port 8787
```

## How It Works

Every time OpenClaw assembles context for the model, the plugin compresses tool outputs and large messages:

- **JSON arrays** (tool outputs, search results) — statistical selection keeps anomalies, errors, boundaries
- **Code** — AST-aware compression via tree-sitter
- **Logs** — pattern deduplication, keeps errors and boundaries
- **Text** — ML-based token compression

Compression is lossless via CCR (Compress-Cache-Retrieve): originals are stored and the agent gets a `headroom_retrieve` tool to access full details when needed.

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `proxyUrl` | auto-detected | Optional URL of a Headroom proxy. Configured URLs are probe-gated before provider routing. Remote URLs (`https://headroom.example.com`) are connect-only. |
| `proxyPort` | `8787` | Port used for default auto-detect and optional local auto-start when `proxyUrl` is not set. |
| `pythonPath` | auto-detected | Optional Python executable override for Python fallback launcher. |
| `autoStart` | `false` | Opt-in auto-start for a local `headroom proxy` if not already running (local URLs only; ignored for remote proxies). Keep `false` when systemd owns the proxy. |
| `startupTimeoutMs` | `20000` | Time to wait for auto-started proxy to become healthy |
| `routeCodexViaProxy` | `true` | Rewrite OpenClaw's built-in `openai-codex` provider to use the active Headroom proxy in memory so upstream Codex requests pass through Headroom. |
| `gatewayProviderIds` | `[]` | Optional explicit list of OpenClaw provider ids to route through the active Headroom proxy in memory. Friendly aliases `codex`, `claude`, `copilot`, and `gemini` are also accepted. When set, this overrides the default `openai-codex` routing list. |

## Comparison with lossless-claw

| | lossless-claw | headroom |
|---|---|---|
| Compaction method | LLM summarization (DAG) | Content-aware compression (zero LLM) |
| Cost of compaction | Tokens (LLM calls) | Zero |
| Best for | Long conversations | Tool-heavy agents with large outputs |
| Retrieval | `lcm_grep`, `lcm_expand` | `headroom_retrieve` (instant) |

## License

Apache-2.0
