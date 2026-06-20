<div align="center"><pre>
  ██╗  ██╗███████╗ █████╗ ██████╗ ██████╗  ██████╗  ██████╗ ███╗   ███╗
  ██║  ██║██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗████╗ ████║
  ███████║█████╗  ███████║██║  ██║██████╔╝██║   ██║██║   ██║██╔████╔██║
  ██╔══██║██╔══╝  ██╔══██║██║  ██║██╔══██╗██║   ██║██║   ██║██║╚██╔╝██║
  ██║  ██║███████╗██║  ██║██████╔╝██║  ██║╚██████╔╝╚██████╔╝██║ ╚═╝ ██║
  ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝
                  The context compression layer for AI agents
</pre></div>

<p align="center"><strong>60–95% fewer tokens · library · proxy · MCP · 6 algorithms · local-first · reversible</strong></p>

<p align="center">
  <a href="https://github.com/chopratejas/headroom/actions/workflows/ci.yml"><img src="https://github.com/chopratejas/headroom/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://app.codecov.io/gh/chopratejas/headroom"><img src="https://codecov.io/gh/chopratejas/headroom/graph/badge.svg" alt="codecov"></a>
  <a href="https://pypi.org/project/headroom-ai/"><img src="https://img.shields.io/pypi/v/headroom-ai.svg" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/headroom-ai"><img src="https://img.shields.io/npm/v/headroom-ai.svg" alt="npm"></a>
  <a href="https://huggingface.co/chopratejas/kompress-v2-base"><img src="https://img.shields.io/badge/model-Kompress--v2--base-yellow.svg" alt="Model: Kompress-v2-base"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://headroom-docs.vercel.app/docs"><img src="https://img.shields.io/badge/docs-online-blue.svg" alt="Docs"></a>
</p>

<p align="center">
  <a href="https://headroom-docs.vercel.app/docs">Docs</a> ·
  <a href="#get-started-60-seconds">Install</a> ·
  <a href="#proof">Proof</a> ·
  <a href="#agent-compatibility-matrix">Agents</a> ·
  <a href="https://discord.gg/yRmaUNpsPJ">Discord</a> ·
  <a href="llms.txt">llms.txt</a> ·
  <a href="ENTERPRISE.md">Enterprise</a>
</p>

<p align="center"><sub>
  <b>AI agents / LLMs:</b> read <a href="llms.txt"><code>/llms.txt</code></a> here, or fetch <a href="https://headroom-docs.vercel.app/llms.txt">the live index</a> / <a href="https://headroom-docs.vercel.app/llms-full.txt">full docs blob</a>.
</sub></p>

---
<p align="center"><a href="https://trendshift.io/repositories/20881" target="_blank"><img src="https://trendshift.io/api/badge/repositories/20881" alt="chopratejas%2Fheadroom | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a></p>

Headroom compresses everything your AI agent reads — tool outputs, logs, RAG chunks, files, and conversation history — before it reaches the LLM. Same answers, fraction of the tokens.

<p align="center">
  <img src="HeadroomDemo-Fast.gif" alt="Headroom in action" width="820">
  <br/><sub>Live: 10,144 → 1,260 tokens — same FATAL found.</sub>
</p>

## What it does

- **Library** — `compress(messages)` in Python or TypeScript, inline in any app
- **Proxy** — `headroom proxy --port 8787`, zero code changes, any language
- **Agent wrap** — `headroom wrap claude|codex|cursor|aider|copilot` in one command
- **MCP server** — `headroom_compress`, `headroom_retrieve`, `headroom_stats` for any MCP client
- **Cross-agent memory** — shared store across Claude, Codex, Gemini, auto-dedup
- **`headroom learn`** — mines failed sessions, writes corrections to `CLAUDE.md` / `AGENTS.md`
- **Output token reduction** — trims what the model *writes back* (not just what you send): drops ceremony/restated code and skips deep "thinking" on routine steps. See [Output token reduction](#output-token-reduction-cut-what-the-model-writes-back).
- **Reversible (CCR)** — originals are cached for retrieval on demand

## How it works (30 seconds)

```
 Your agent / app
   (Claude Code, Cursor, Codex, LangChain, Agno, Strands, your own code…)
        │   prompts · tool outputs · logs · RAG results · files
        ▼
    ┌────────────────────────────────────────────────────┐
    │  Headroom   (runs locally — your data stays here)  │
    │  ────────────────────────────────────────────────  │
    │  CacheAligner  →  ContentRouter  →  CCR            │
    │                    ├─ SmartCrusher   (JSON)        │
    │                    ├─ CodeCompressor (AST)         │
    │                    └─ Kompress-base  (text, HF)    │
    │                                                    │
    │  Cross-agent memory  ·  headroom learn  ·  MCP     │
    └────────────────────────────────────────────────────┘
        │   compressed prompt  +  retrieval tool
        ▼
 LLM provider  (Anthropic · OpenAI · Bedrock · …)
```

- **ContentRouter** — detects content type, selects the right compressor
- **SmartCrusher / CodeCompressor / Kompress-base** — compress JSON, AST, or prose
- **CacheAligner** — stabilizes prefixes so provider KV caches actually hit
- **CCR** — stores originals locally; LLM calls `headroom_retrieve` if it needs them

→ [Architecture](https://headroom-docs.vercel.app/docs/architecture) · [CCR reversible compression](https://headroom-docs.vercel.app/docs/ccr) · [Kompress-v2-base model card](https://huggingface.co/chopratejas/kompress-v2-base)

## Get started (60 seconds)

```bash
# 1 — Install
pip install "headroom-ai[all]"          # Python
npm install headroom-ai                 # Node / TypeScript

# 2 — Pick your mode
headroom wrap claude                    # wrap a coding agent
headroom proxy --port 8787              # drop-in proxy, zero code changes
# or: from headroom import compress      # inline library

# 3 — See the savings
headroom perf
```

Granular extras: `[proxy]`, `[mcp]`, `[ml]`, `[code]`, `[memory]`, `[relevance]`, `[image]`, `[agno]`, `[langchain]`, `[evals]`, `[pytorch-mps]` (Apple-GPU memory-embedder offload — set `HEADROOM_EMBEDDER_RUNTIME=pytorch_mps`). Requires **Python 3.10+**.

## Proof

**Savings on real agent workloads:**

| Workload                      | Before | After  | Savings |
|-------------------------------|-------:|-------:|--------:|
| Code search (100 results)     | 17,765 |  1,408 | **92%** |
| SRE incident debugging        | 65,694 |  5,118 | **92%** |
| GitHub issue triage           | 54,174 | 14,761 | **73%** |
| Codebase exploration          | 78,502 | 41,254 | **47%** |

**Accuracy preserved on standard benchmarks:**

| Benchmark  | Category | N   | Baseline | Headroom | Delta      |
|------------|----------|----:|---------:|---------:|------------|
| GSM8K      | Math     | 100 |    0.870 |    0.870 | **±0.000** |
| TruthfulQA | Factual  | 100 |    0.530 |    0.560 | **+0.030** |
| SQuAD v2   | QA       | 100 |        — |  **97%** | 19% compression |
| BFCL       | Tools    | 100 |        — |  **97%** | 32% compression |

Reproduce: `python -m headroom.evals suite --tier 1` · [Full benchmarks & methodology](https://headroom-docs.vercel.app/docs/benchmarks)

## Output token reduction (cut what the model writes back)

Everything above shrinks the prompt you **send**. But you also pay for every
token the model **writes back** — and on Opus-class models output costs 5× input.
A lot of that output is waste: "Great, let me…" preambles, re-printing code you
just showed it, and deep "thinking" on routine steps like reading a file.

Headroom can trim that too, from the proxy, without you changing any code:

- **Verbosity steering** — appends a short "be terse, don't restate context"
  note to the end of the system prompt (so your prompt cache still hits).
- **Effort routing** — when a turn is just the model resuming after a tool result
  (a file read, a passing test), it dials the model's thinking effort down. New
  questions and errors keep full effort.

Turn it on:

```bash
export HEADROOM_OUTPUT_SHAPER=1     # off by default
headroom proxy --port 8787
```

> **Already running a proxy?** These switches are read *live* on every request,
> so a proxy that `headroom wrap` **reused** (rather than started) would not see
> a value you export afterwards — its environment was snapshotted at launch.
> `headroom wrap` now hot-syncs your current settings to the running proxy via a
> loopback `POST /admin/runtime-env`, so they take effect immediately with **no
> restart** (no cold start, no dropped requests, no lost caches). Set them before
> you `wrap`. On a shared proxy these overrides are global — the last explicit
> setting wins.

**Learn the right terseness for you.** People don't *say* how terse they want
answers — they *show* it (they interrupt long replies, or move on before they
could have read them). `headroom learn --verbosity` reads your past sessions and
picks the level automatically:

```bash
headroom learn --verbosity            # preview what it found (dry run)
headroom learn --verbosity --apply    # save it; the proxy uses it from now on
```

**See how many output tokens you saved.** Output savings are *counterfactual* —
we never see what the model *would* have written — so Headroom reports an honest
**estimate with a confidence range**, never a made-up number:

```bash
headroom output-savings
# Reduction: 31.7%  (95% CI 27.7% … 35.7%)   [estimated]
```

Want a *measured* number instead of an estimate? Leave 10% of conversations
unshaped as a control group: `export HEADROOM_OUTPUT_HOLDOUT=0.1`. The dashboard
shows an **Output Tokens Saved** card next to input compression, labelled
`measured` or `estimated` with the confidence band.

→ Full write-up incl. the measurement methodology: [`docs/proposals/output-token-reduction.md`](docs/proposals/output-token-reduction.md)

<a href="https://www.star-history.com/?repos=chopratejas%2Fheadroom&type=date&legend=top-left">
 <picture>
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=chopratejas/headroom&type=date&legend=top-left" />
 </picture>
</a>

## Agent compatibility matrix

| Agent       | `headroom wrap` | Notes                            |
|-------------|:---------------:|----------------------------------|
| Claude Code | ✅              | `--memory` · `--code-graph`      |
| Codex       | ✅              | shares memory with Claude        |
| Cursor      | ✅              | prints config — paste once       |
| Aider       | ✅              | starts proxy + launches          |
| Copilot CLI | ✅              | starts proxy + launches          |
| OpenClaw    | ✅              | installs as ContextEngine plugin |

Any OpenAI-compatible client works via `headroom proxy`. MCP-native: `headroom mcp install`.

### GitHub Copilot CLI subscription mode

Headroom can route GitHub Copilot CLI subscription traffic through the local proxy:

```bash
headroom copilot-auth login
headroom wrap copilot --subscription -- --model gpt-4o
```

This lets Headroom intercept OpenAI-compatible Copilot CLI requests and apply the same proxy compression pipeline before forwarding to GitHub Copilot's hosted API. The wrapper exchanges Headroom's reusable GitHub OAuth token for Copilot's short-lived API token and prints the upstream endpoint as `COPILOT_PROVIDER_API_URL=...` during launch.

`headroom copilot-auth login` stores a Headroom-specific Copilot OAuth token.
This avoids relying on generic GitHub or Copilot CLI tokens that can read
Copilot account metadata but may still be rejected by Copilot's token-exchange
endpoint.

For GitHub Enterprise Server or custom-domain Copilot deployments, set the
deployment domain before launching:

```bash
export GITHUB_COPILOT_ENTERPRISE_DOMAIN=ghe.example.com
```

For GitHub.com Enterprise Cloud URLs such as
`github.com/enterprises/your-enterprise`, do not set an enterprise-domain
override. Headroom uses GitHub's normal token-exchange endpoint and the Copilot
API endpoint advertised for the signed-in account.

Platform support note: macOS auth reuse via Copilot CLI Keychain storage has been smoke-tested. Windows Credential Manager, Linux Secret Service / `secret-tool`, and Docker/CI token-injection paths are implemented or planned as auth-discovery paths, but still need real OS validation before they should be considered fully vetted. For Docker and CI, prefer passing an explicit `GITHUB_COPILOT_TOKEN` or `GITHUB_COPILOT_GITHUB_TOKEN` rather than relying on host keychain access.

## When to use · When to skip

**Great fit if you…**
- run AI coding agents daily and want savings without changing your code
- work across multiple agents and want shared memory
- need reversible compression — originals are retrievable via CCR within the configured TTL

**Skip it if you…**
- only use a single provider's native compaction and don't need cross-agent memory
- work in a sandboxed environment where local processes can't run

<details>
<summary><b>Integrations — drop Headroom into any stack</b></summary>

| Your setup             | Hook in with                                                     |
|------------------------|------------------------------------------------------------------|
| Any Python app         | `compress(messages, model=…)`                                    |
| Any TypeScript app     | `await compress(messages, { model })`                            |
| Anthropic / OpenAI SDK | `withHeadroom(new Anthropic())` · `withHeadroom(new OpenAI())`   |
| Vercel AI SDK          | `wrapLanguageModel({ model, middleware: headroomMiddleware() })` |
| LiteLLM                | `litellm.callbacks = [HeadroomCallback()]`                       |
| LangChain              | `HeadroomChatModel(your_llm)`                                    |
| Agno                   | `HeadroomAgnoModel(your_model)`                                  |
| Strands                | [Strands guide](https://headroom-docs.vercel.app/docs/strands)  |
| ASGI apps              | `app.add_middleware(CompressionMiddleware)`                      |
| Multi-agent            | `SharedContext().put / .get`                                     |
| MCP clients            | `headroom mcp install`                                           |

</details>

<details>
<summary><b>What's inside</b></summary>

- **SmartCrusher** — universal JSON: arrays of dicts, nested objects, mixed types.
- **CodeCompressor** — AST-aware for Python, JS, Go, Rust, Java, C++.
- **Kompress-base** — our HuggingFace model, trained on agentic traces.
- **Image compression** — 40–90% reduction via trained ML router.
- **CacheAligner** — stabilizes prefixes so Anthropic/OpenAI KV caches actually hit.
- **IntelligentContext** — score-based context fitting with learned importance.
- **CCR** — reversible compression; LLM retrieves originals on demand.
- **Cross-agent memory** — shared store, agent provenance, auto-dedup.
- **SharedContext** — compressed context passing across multi-agent workflows.
- **`headroom learn`** — plugin-based failure mining for Claude, Codex, Gemini.

</details>

<details>
<summary><b>Pipeline internals</b></summary>

Headroom exposes one stable request lifecycle across `compress()`, the SDK, and the proxy:

`Setup` → `Pre-Start` → `Post-Start` → `Input Received` → `Input Cached` → `Input Routed` → `Input Compressed` → `Input Remembered` → `Pre-Send` → `Post-Send` → `Response Received`

- **Transforms** do the work: CacheAligner, ContentRouter, SmartCrusher, CodeCompressor, Kompress-base, IntelligentContext / RollingWindow.
- **Pipeline extensions** observe or customize lifecycle stages via `on_pipeline_event(...)`.
- **Compression hooks** sit alongside the canonical lifecycle as an additional extension seam.
- **Proxy extensions** remain the server/app integration seam for ASGI middleware, routes, and startup policy.

Provider and tool-specific behavior lives under `headroom/providers/` so core orchestration stays focused on lifecycle, sequencing, and policy.

- **CLI/tool slices**: `headroom/providers/claude`, `copilot`, `codex`, `openclaw`
- **Provider runtime slices**: `headroom/providers/claude`, `gemini`, plus shared backend/runtime dispatch in `headroom/providers/registry.py`
- **Core files stay orchestration-first**: `wrap.py`, `client.py`, `cli/proxy.py`, and `proxy/server.py` delegate provider-specific env shaping, API target normalization, backend selection, and transport dispatch.

</details>

## Install

```bash
pip install "headroom-ai[all]"          # Python, everything
npm install headroom-ai                 # TypeScript / Node
docker pull ghcr.io/chopratejas/headroom:latest
```

Granular extras: `[proxy]`, `[mcp]`, `[ml]` (Kompress-base), `[code]`, `[memory]`, `[relevance]`, `[image]`, `[agno]`, `[langchain]`, `[evals]`, `[pytorch-mps]` (Apple-GPU memory-embedder offload — set `HEADROOM_EMBEDDER_RUNTIME=pytorch_mps`). Requires **Python 3.10+**.

Using `pipx`? Choose a supported interpreter explicitly:

```bash
pipx install --python python3.13 "headroom-ai[all]"
```

→ [Installation guide](https://headroom-docs.vercel.app/docs/installation) — Docker tags, persistent service, PowerShell, devcontainers.

### Updating

```bash
headroom update          # detects pip / pipx / uv tool and upgrades in place
headroom update --check  # report the latest release without upgrading
headroom update --pre    # include pre-releases
```

`headroom update` figures out how Headroom was installed (pip/venv, `pip --user`,
pipx, uv tool) and runs the matching upgrade across macOS, Linux, and Windows.
For git checkouts, editable installs, Docker images, and externally-managed
system Pythons (PEP 668) it prints the correct manual step instead of guessing.

The proxy also shows a one-line "update available" notice on startup. It checks
PyPI at most once a day, in the background, and never blocks. Opt out with
`HEADROOM_UPDATE_CHECK=off` (also skipped in `--stateless` mode and CI).

### Corporate / SSL-inspection environments

If `pip install "headroom-ai[all]"` fails with `CERTIFICATE_VERIFY_FAILED`
(`unable to get local issuer certificate`), your network uses **SSL inspection** — a MITM
proxy presenting a company-issued CA. The build backend (`maturin`) downloads `rustup` over a
connection your TLS stack doesn't trust. **Install Rust first** so the build doesn't fetch it:

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && rustup default stable
# Windows
winget install Rustlang.Rustup && rustup default stable
```

Restart your shell, then `pip install "headroom-ai[all]"`. A prebuilt wheel avoids the Rust
build entirely where available: `pip install --only-binary headroom-ai headroom-ai`.

Two runtime assets are fetched over TLS; if they are blocked, trust your corporate CA via
`REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` / `CURL_CA_BUNDLE`:

- **`cdn.pyke.io`** — the ONNX Runtime for the Rust core. Alternatively pre-provide it with
  `ORT_STRATEGY=system` and `ORT_LIB_LOCATION=/path/to/onnxruntime`.
- **`huggingface.co`** — the `kompress-base` compression model. Pre-download it and run with
  `HF_HUB_OFFLINE=1`, or set `HF_ENDPOINT` to a trusted mirror.

Running with compression disabled (pure gateway) requires neither asset.

## headroom learn

<p align="center">
  <img src="headroom_learn.gif" alt="headroom learn in action" width="720">
</p>

`headroom learn` — mines failed sessions, writes corrections to `CLAUDE.md` / `AGENTS.md` / `GEMINI.md`.

## Documentation

| Start here                                                                    | Go deeper                                                                          |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| [Quickstart](https://headroom-docs.vercel.app/docs/quickstart)                | [Architecture](https://headroom-docs.vercel.app/docs/architecture)                 |
| [Proxy](https://headroom-docs.vercel.app/docs/proxy)                          | [How compression works](https://headroom-docs.vercel.app/docs/how-compression-works) |
| [MCP tools](https://headroom-docs.vercel.app/docs/mcp)                        | [CCR — reversible compression](https://headroom-docs.vercel.app/docs/ccr)          |
| [Memory](https://headroom-docs.vercel.app/docs/memory)                        | [Cache optimization](https://headroom-docs.vercel.app/docs/cache-optimization)     |
| [Failure learning](https://headroom-docs.vercel.app/docs/failure-learning)    | [Benchmarks](https://headroom-docs.vercel.app/docs/benchmarks)                    |
| [Configuration](https://headroom-docs.vercel.app/docs/configuration)          | [Limitations](https://headroom-docs.vercel.app/docs/limitations)                  |

## Compared to

Headroom runs **locally**, covers **every** content type, works with every major framework, and is **reversible**.

|                                                                              | Scope                                          | Deploy                             | Local | Reversible |
|------------------------------------------------------------------------------|------------------------------------------------|------------------------------------|:-----:|:----------:|
| **Headroom**                                                                 | All context — tools, RAG, logs, files, history | Proxy · library · middleware · MCP | Yes   | Yes        |
| [RTK](https://github.com/rtk-ai/rtk)                                        | CLI command outputs                            | CLI wrapper                        | Yes   | No         |
| [lean-ctx](https://github.com/yvgude/lean-ctx)                               | CLI commands, MCP tools, editor rules          | CLI wrapper · MCP                  | Yes   | No         |
| [Compresr](https://compresr.ai), [Token Co.](https://thetokencompany.ai)    | Text sent to their API                         | Hosted API call                    | No    | No         |
| OpenAI Compaction                                                            | Conversation history                           | Provider-native                    | No    | No         |

> **Attribution.** Headroom ships with the excellent [RTK](https://github.com/rtk-ai/rtk) binary for shell-output rewriting — `git show --short`, scoped `ls`, summarized installers. Huge thanks to the RTK team; their tool is a first-class part of our stack, and Headroom compresses everything downstream of it. Headroom can also use [lean-ctx](https://github.com/yvgude/lean-ctx) as the selected CLI context tool; set `HEADROOM_CONTEXT_TOOL=lean-ctx` before running `headroom wrap ...`.

## Contributing

```bash
git clone https://github.com/chopratejas/headroom.git && cd headroom
uv sync --extra dev && uv run pytest
```

Devcontainers in `.devcontainer/` (default + `memory-stack` with Qdrant & Neo4j). See [CONTRIBUTING.md](CONTRIBUTING.md).

## Community

- **[Discord](https://discord.gg/yRmaUNpsPJ)** — questions, feedback, war stories.
- **[Kompress-v2-base on HuggingFace](https://huggingface.co/chopratejas/kompress-v2-base)** — the model behind our text compression.

## License

Apache 2.0 — see [LICENSE](LICENSE).
