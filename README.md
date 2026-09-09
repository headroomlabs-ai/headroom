# Headroom — Claude Code, local-only fork

A trimmed fork of [headroomlabs-ai/headroom](https://github.com/headroomlabs-ai/headroom)
that keeps exactly one job: sit between **Claude Code** and `api.anthropic.com`
as a local proxy and compress what the model reads (tool outputs, logs, JSON,
diffs, file contents) before it is sent. Same answers, fewer input tokens.

Everything runs on this machine. This fork removes every code path that could
send data anywhere other than Anthropic's API on your behalf:

- the anonymous usage beacon (`telemetry/session.py`) and its default-on switch
- the license usage reporter and the PyPI update check
- cloud/remote compression endpoints, Langfuse tracing, OpenAI embedders (opt-in
  upstream; see "What still exists" below)
- every non-Claude agent (`codex`, `copilot`, `cursor`, `gemini`, `grok`, …),
  every non-Anthropic backend (LiteLLM, any-llm, Bedrock, Vertex, Azure Foundry),
  the OpenAI/Gemini/batch proxy handlers, framework adapters, the TypeScript SDK,
  docs site, Docker/CI/release plumbing

What remains: `headroom proxy`, `headroom wrap claude`, `headroom wrap
vscode-claude`, `headroom unwrap claude`, `headroom init claude`, `headroom
doctor`, the MCP server (`headroom mcp serve` — `headroom_retrieve` for
reversible compression), the dashboard, memory/learn/code-graph features, and
the Rust compression core.

## Install (after cloning)

Requirements: macOS (Apple Silicon or Intel) or Linux, [`uv`](https://docs.astral.sh/uv/)
(`brew install uv`), and Claude Code on your `PATH`. Python 3.13 is fetched by
`uv` if you don't have it.

The Python package hard-requires the compiled Rust extension `headroom._core`.
Pick one of two paths.

### Option A — no Rust toolchain (recommended)

```bash
git clone <your-fork-url> headroom && cd headroom
bash scripts/install_no_rust.sh
ln -sf "$PWD/.venv/bin/headroom" ~/.local/bin/headroom   # or anywhere on PATH
headroom --version
```

The script creates `.venv`, installs the runtime dependencies plus the prebuilt
`_core` extension from the matching upstream PyPI wheel, then removes the
upstream package so **this checkout's code** is what runs (a `.pth` file points
the venv at the repo — no `PYTHONPATH` needed, which matters because Claude Code
launches the MCP server with its own environment). The only network access is
PyPI. Set `VENV_DIR=…` / `PYTHON_VERSION=…` to override defaults.

### Option B — build the extension

```bash
git clone <your-fork-url> headroom && cd headroom
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # once
uv sync --extra proxy --extra dev      # maturin builds headroom._core
ln -sf "$PWD/.venv/bin/headroom" ~/.local/bin/headroom
```

`make build-ext` rebuilds the extension after changing anything under `crates/`.

## Set up Claude Code

1. Put the local-only environment in your shell profile (`~/.zshrc`):

   ```bash
   export HEADROOM_OFFLINE=1               # no HF downloads, no optional egress
   export HEADROOM_BINARIES_OFFLINE=1      # no difft/scc fetch from GitHub
   export LITELLM_LOCAL_MODEL_COST_MAP=True
   ```

   (`.env.example` has the same list with comments.)

2. Launch Claude Code through the proxy from your project directory:

   ```bash
   headroom wrap claude                    # starts proxy, launches claude
   headroom wrap claude -- --model opus    # pass args through to claude
   headroom wrap claude --code-memory none # skip the Serena MCP install
   ```

   `wrap claude` starts the proxy on `127.0.0.1:8787`, writes
   `ANTHROPIC_BASE_URL` into `./.claude/settings.local.json` (restored when the
   session ends), registers the `headroom` MCP server in `~/.claude.json`, and by
   default also registers [Serena](https://github.com/oraios/serena) via `uvx`
   (pulls from PyPI) — skip that with `--code-memory none`.

3. Check it and watch savings:

   ```bash
   headroom doctor                         # confirms routing + proxy health
   headroom dashboard                      # live savings (proxy must be running)
   headroom unwrap claude                  # restore settings, stop the proxy
   ```

Optional: `headroom init claude` installs a `SessionStart` hook and a
persistent proxy profile so plain `claude` (without `wrap`) also routes through
Headroom; `headroom wrap vscode-claude` does the same for the VS Code extension.

`CLAUDE_CODE_USE_VERTEX` / `CLAUDE_CODE_USE_FOUNDRY` are rejected: this fork
routes only to the direct Anthropic API.

## What still exists (opt-in only, nothing on by default)

| What | When it fetches / sends | Off switch |
|---|---|---|
| Kompress / tokenizer weights from huggingface.co | first use of ML compression | `HEADROOM_OFFLINE=1` (sets `HF_HUB_OFFLINE`) |
| `difft` / `scc` binaries from GitHub releases | `headroom tools install`, or `--intercept-tool-results` | `HEADROOM_BINARIES_OFFLINE=1` |
| tiktoken BPE vocab | first token count | pre-seed `TIKTOKEN_CACHE_DIR` |
| litellm price map | on import unless local | `LITELLM_LOCAL_MODEL_COST_MAP=True` (set by default in `headroom/__init__.py`) |
| Serena via `uvx` | `wrap claude` default | `--code-memory none` |
| Anthropic OAuth usage endpoint (subscription quota tile) | proxy, with your own token | `--no-subscription-tracking` |
| Remote Kompress (`transforms/kompress_remote.py`) — **sends prompt text** | only if `HEADROOM_KOMPRESS_ENDPOINT` is set | leave unset |
| Langfuse tracing (`observability/tracing.py`) | only if `HEADROOM_LANGFUSE_ENABLED=1` + keys | leave unset |
| OpenAI embedder for memory (`memory/adapters/embedders.py`) | only if you construct `OpenAIEmbedder` | default is the local embedder |

## Tests

The suite is reduced to the fundamentals (~3,400 tests, ~1 minute, serial) and
runs with a per-test timeout and a network block (`pytest-socket`) so it can
neither hang nor reach anything outside loopback:

```bash
.venv/bin/python -m pytest
```

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Upstream:
https://github.com/headroomlabs-ai/headroom.
