<div align="center">

# Headroom

**Compress everything your AI agent reads. Same answers, fraction of the tokens.**

[![CI](https://github.com/chopratejas/headroom/actions/workflows/ci.yml/badge.svg)](https://github.com/chopratejas/headroom/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/chopratejas/headroom/graph/badge.svg)](https://app.codecov.io/gh/chopratejas/headroom)
[![PyPI](https://img.shields.io/pypi/v/headroom-ai.svg)](https://pypi.org/project/headroom-ai/)
[![npm](https://img.shields.io/npm/v/headroom-ai.svg)](https://www.npmjs.com/package/headroom-ai)
[![Model: Kompress-base](https://img.shields.io/badge/model-Kompress--base-yellow.svg)](https://huggingface.co/chopratejas/kompress-base)
[![Tokens saved: 60B+](https://img.shields.io/badge/tokens%20saved-60B%2B-2ea44f)](https://headroomlabs.ai/dashboard)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://headroom-docs.vercel.app/docs)

<img src="HeadroomDemo-Fast.gif" alt="Headroom in action" width="820">

</div>

---

> **orq internal deployment** — if you're on the team, see **[docs/INTERNAL.md](docs/INTERNAL.md)** for the GPU proxy setup, `aurl` shell config, MCP install, and deploy instructions.

---

Every tool call, log line, DB read, RAG chunk, and file your agent injects into a prompt is mostly boilerplate. Headroom strips the noise and keeps the signal — **losslessly, locally, and without touching accuracy.**

> **100 logs. One FATAL error buried at position 67. Both runs found it.**
> Baseline **10,144 tokens** → Headroom **1,260 tokens** — **87% fewer, identical answer.**
> `python examples/needle_in_haystack_test.py`

---

## Quick start

Works with Anthropic, OpenAI, Google, Bedrock, Vertex, Azure, OpenRouter, and 100+ models via LiteLLM.

**Wrap your coding agent — one command:**

```bash
pip install "headroom-ai[all]"

headroom wrap claude      # Claude Code
headroom wrap codex       # Codex
headroom wrap cursor      # Cursor
headroom wrap aider       # Aider
headroom wrap copilot     # GitHub Copilot CLI
```

**Drop it into your own code — Python or TypeScript:**

```python
from headroom import compress

result = compress(messages, model="claude-sonnet-4-5")
response = client.messages.create(model="claude-sonnet-4-5", messages=result.messages)
print(f"Saved {result.tokens_saved} tokens ({result.compression_ratio:.0%})")
```

```typescript
import { compress } from 'headroom-ai';
const result = await compress(messages, { model: 'gpt-4o' });
```

**Or run it as a proxy — zero code changes, any language:**

```bash
headroom proxy --port 8787
ANTHROPIC_BASE_URL=http://localhost:8787 your-app
OPENAI_BASE_URL=http://localhost:8787/v1 your-app
```

---

## Why Headroom

- **Accuracy-preserving.** GSM8K **0.870 → 0.870** (±0.000). TruthfulQA **+0.030**. SQuAD v2 and BFCL both **97%** accuracy after compression. Validated on public OSS benchmarks you can rerun yourself.
- **Runs on your machine.** No cloud API, no data egress. Compression latency is milliseconds — faster end-to-end for Sonnet / Opus / GPT-4 class models than a hosted service round-trip.
- **[Kompress-base](https://huggingface.co/chopratejas/kompress-base) on HuggingFace.** Our open-source text compressor, fine-tuned on real agentic traces — tool outputs, logs, RAG chunks, code. Install with `pip install "headroom-ai[ml]"`.
- **Cross-agent memory and learning.** Claude Code saves a fact, Codex reads it back. `headroom learn` mines failed sessions and writes corrections straight to `CLAUDE.md` / `AGENTS.md` / `GEMINI.md` — reliability compounds over time.
- **Reversible (CCR).** Compression is not deletion. The model can always call `headroom_retrieve` to pull the original bytes. Nothing is thrown away.

Bundles the [RTK](https://github.com/rtk-ai/rtk) binary for shell-output rewriting — full [attribution below](#compared-to).

---

## How it fits

```
 Your agent / app
   (Claude Code, Cursor, Codex, LangChain, Agno, Strands, your own code…)
        │   prompts · tool outputs · logs · RAG results · files
        ▼
    ┌────────────────────────────────────────────────────┐
    │  Headroom   (runs locally — your data stays here)  │
    │  ───────────────────────────────────────────────   │
    │  CacheAligner  →  ContentRouter  →  CCR             │
    │                    ├─ SmartCrusher   (JSON)         │
    │                    ├─ CodeCompressor (AST)          │
    │                    └─ Kompress-base  (text, HF)     │
    │                                                     │
    │  Cross-agent memory  ·  headroom learn  ·  MCP      │
    └────────────────────────────────────────────────────┘
        │   compressed prompt  +  retrieval tool
        ▼
 LLM provider  (Anthropic · OpenAI · Bedrock · …)
```

→ [Architecture](https://headroom-docs.vercel.app/docs/architecture) · [CCR reversible compression](https://headroom-docs.vercel.app/docs/ccr) · [Kompress-base model card](https://huggingface.co/chopratejas/kompress-base)

### Canonical pipeline lifecycle

Headroom now exposes one stable request lifecycle across `compress()`, the SDK, and the proxy:

`Setup` → `Pre-Start` → `Post-Start` → `Input Received` → `Input Cached` → `Input Routed` → `Input Compressed` → `Input Remembered` → `Pre-Send` → `Post-Send` → `Response Received`

- **Transforms** still do the work: CacheAligner, ContentRouter, SmartCrusher, CodeCompressor, Kompress-base, IntelligentContext / RollingWindow.
- **Pipeline extensions** observe or customize those lifecycle stages via `on_pipeline_event(...)`.
- **Compression hooks** still work and now sit alongside the canonical lifecycle instead of being the only extension seam.
- **Proxy extensions** remain the server/app integration seam for ASGI middleware, routes, and startup policy.

### Provider slices

Provider and tool-specific behavior is being moved behind dedicated modules under `headroom/providers/` so core orchestration stays focused on lifecycle, sequencing, and policy.

- **CLI/tool slices**: `headroom/providers/claude`, `copilot`, `codex`, `openclaw`
- **Provider runtime slices**: `headroom/providers/claude`, `gemini`, plus shared backend/runtime dispatch in `headroom/providers/registry.py`
- **Core files stay orchestration-first**: `wrap.py`, `client.py`, `cli/proxy.py`, and `proxy/server.py` now delegate provider-specific env shaping, API target normalization, backend selection, and transport dispatch instead of inlining those rules.

---

## Proof

**Savings on real agent workloads:**

| Workload                      | Before | After  | Savings |
|-------------------------------|-------:|-------:|--------:|
| Code search (100 results)     | 17,765 |  1,408 | **92%** |
| SRE incident debugging        | 65,694 |  5,118 | **92%** |
| GitHub issue triage           | 54,174 | 14,761 | **73%** |
| Codebase exploration          | 78,502 | 41,254 | **47%** |

**Accuracy preserved on standard benchmarks:**

| Benchmark  | Category | N   | Baseline | Headroom | Delta     |
|------------|----------|----:|---------:|---------:|----------:|
| GSM8K      | Math     | 100 |    0.870 |    0.870 | **±0.000**|
| TruthfulQA | Factual  | 100 |    0.530 |    0.560 | **+0.030**|
| SQuAD v2   | QA       | 100 |        — |  **97%** | 19% compression |
| BFCL       | Tools    | 100 |        — |  **97%** | 32% compression |

Reproduce:

```bash
python -m headroom.evals suite --tier 1
```

**Community, live:**

<div align="center">
  <a href="https://headroomlabs.ai/dashboard">
    <img src="headroom-savings.png" alt="60B+ tokens saved — community leaderboard" width="820">
  </a>
  <p><b><a href="https://headroomlabs.ai/dashboard">60B+ tokens saved by the community in the last 20 days — live leaderboard →</a></b></p>
</div>

→ [Full benchmarks & methodology](https://headroom-docs.vercel.app/docs/benchmarks)

---

## Built for coding agents

| Agent              | One-command wrap                   | Notes                                                            |
|--------------------|------------------------------------|------------------------------------------------------------------|
| **Claude Code**    | `headroom wrap claude`             | `--memory` for cross-agent memory, `--code-graph` for codebase intel |
| **Codex**          | `headroom wrap codex --memory`     | Shares the same memory store as Claude                           |
| **Cursor**         | `headroom wrap cursor`             | Prints Cursor config — paste once, done                          |
| **Aider**          | `headroom wrap aider`              | Starts proxy, launches Aider                                     |
| **Copilot CLI**    | `headroom wrap copilot`            | Starts proxy, launches Copilot                                   |
| **OpenClaw**       | `headroom wrap openclaw`           | Installs Headroom as ContextEngine plugin                        |

MCP-native too — `headroom mcp install` exposes `headroom_compress`, `headroom_retrieve`, and `headroom_stats` to any MCP client.

<div align="center">
  <img src="headroom_learn.gif" alt="headroom learn in action" width="720">
</div>

---

## Integrations

<details>
<summary><b>Drop Headroom into any stack</b></summary>

| Your setup              | Hook in with                                                     |
|-------------------------|------------------------------------------------------------------|
| Any Python app          | `compress(messages, model=…)`                                    |
| Any TypeScript app      | `await compress(messages, { model })`                            |
| Anthropic / OpenAI SDK  | `withHeadroom(new Anthropic())` · `withHeadroom(new OpenAI())`   |
| Vercel AI SDK           | `wrapLanguageModel({ model, middleware: headroomMiddleware() })` |
| LiteLLM                 | `litellm.callbacks = [HeadroomCallback()]`                       |
| LangChain               | `HeadroomChatModel(your_llm)`                                    |
| Agno                    | `HeadroomAgnoModel(your_model)`                                  |
| Strands                 | [Strands guide](https://headroom-docs.vercel.app/docs/strands) |
| ASGI apps               | `app.add_middleware(CompressionMiddleware)`                      |
| Multi-agent             | `SharedContext().put / .get`                                     |
| MCP clients             | `headroom mcp install`                                           |

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

---

## Install

```bash
pip install "headroom-ai[all]"          # Python, everything
npm  install headroom-ai                # TypeScript / Node
docker pull ghcr.io/chopratejas/headroom:latest
```

Granular extras: `[proxy]`, `[mcp]`, `[ml]` (Kompress-base), `[agno]`, `[langchain]`, `[evals]`. Requires **Python 3.10+**.

→ [Installation guide](https://headroom-docs.vercel.app/docs/installation) — Docker tags, persistent service, PowerShell, devcontainers.

---

## Documentation

| Start here                                                              | Go deeper                                                              |
|-------------------------------------------------------------------------|------------------------------------------------------------------------|
| [Quickstart](https://headroom-docs.vercel.app/docs/quickstart)    | [Architecture](https://headroom-docs.vercel.app/docs/architecture) |
| [Proxy](https://headroom-docs.vercel.app/docs/proxy)              | [How compression works](https://headroom-docs.vercel.app/docs/how-compression-works) |
| [MCP tools](https://headroom-docs.vercel.app/docs/mcp)            | [CCR — reversible compression](https://headroom-docs.vercel.app/docs/ccr) |
| [Memory](https://headroom-docs.vercel.app/docs/memory)            | [Cache optimization](https://headroom-docs.vercel.app/docs/cache-optimization) |
| [Failure learning](https://headroom-docs.vercel.app/docs/failure-learning) | [Benchmarks](https://headroom-docs.vercel.app/docs/benchmarks) |
| [Configuration](https://headroom-docs.vercel.app/docs/configuration) | [Limitations](https://headroom-docs.vercel.app/docs/limitations) |

---

## Compared to

Headroom runs **locally**, covers **every** content type (not just CLI or text), works with every major framework, and is **reversible**.

|                                  | Scope                                           | Deploy                              | Local | Reversible |
|----------------------------------|-------------------------------------------------|-------------------------------------|:-----:|:----------:|
| **Headroom**                     | All context — tools, RAG, logs, files, history  | Proxy · library · middleware · MCP  |  Yes  |    Yes     |
| [RTK](https://github.com/rtk-ai/rtk) | CLI command outputs                         | CLI wrapper                         |  Yes  |    No      |
| [Compresr](https://compresr.ai), [Token Co.](https://thetokencompany.ai) | Text sent to their API | Hosted API call         |  No   |    No      |
| OpenAI Compaction                | Conversation history                            | Provider-native                     |  No   |    No      |

> **Attribution.** Headroom ships with the excellent [RTK](https://github.com/rtk-ai/rtk) binary for shell-output rewriting — `git show` → `git show --short`, noisy `ls` → scoped, chatty installers → summarized. Huge thanks to the RTK team; their tool is a first-class part of our stack, and Headroom compresses everything downstream of it.

---

## Contributing

```bash
git clone https://github.com/chopratejas/headroom.git && cd headroom
pip install -e ".[dev]" && pytest
```

Devcontainers in `.devcontainer/` (default + `memory-stack` with Qdrant & Neo4j). See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Community

- **[Live leaderboard](https://headroomlabs.ai/dashboard)** — 60B+ tokens saved and counting.
- **[Discord](https://discord.gg/yRmaUNpsPJ)** — questions, feedback, war stories.
- **[Kompress-base on HuggingFace](https://huggingface.co/chopratejas/kompress-base)** — the model behind our text compression.

## License

Apache 2.0 — see [LICENSE](LICENSE).
