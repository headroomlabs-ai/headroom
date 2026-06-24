# Headroom

<div class="hero" markdown>

**The Context Optimization Layer for LLM Applications**

Compress everything your AI agent reads. Same answers, fraction of the tokens.

</div>

<div class="badges" markdown>

[![PyPI](https://img.shields.io/pypi/v/headroom-ai.svg)](https://pypi.org/project/headroom-ai/)
[![Python](https://img.shields.io/pypi/pyversions/headroom-ai.svg)](https://pypi.org/project/headroom-ai/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/chopratejas/headroom/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yRmaUNpsPJ)

</div>

<div class="stats-bar" markdown>

<div class="stat">
<span class="number">87%</span>
<span class="label">Avg Token Reduction</span>
</div>
<div class="stat">
<span class="number">100%</span>
<span class="label">Answer Accuracy</span>
</div>
<div class="stat">
<span class="number">6</span>
<span class="label">Compression Algorithms</span>
</div>
<div class="stat">
<span class="number">100+</span>
<span class="label">LLM Providers</span>
</div>

</div>

---

## What It Does

Every tool call, DB query, file read, and RAG retrieval your agent makes is 70-95% boilerplate. Headroom compresses it away before it hits the model. The LLM sees less noise, responds faster, and costs less.

```
Your Agent / App
    │
    │  tool outputs, logs, DB reads, RAG results, file reads, API responses
    ▼
 Headroom  ← proxy, Python library, or framework integration
    │
    ▼
 LLM Provider  (OpenAI, Anthropic, Google, Bedrock, 100+ via LiteLLM)
```

Headroom works as a **transparent proxy** (zero code changes), a **Python function** (`compress()`), or a **framework integration** (LangChain, Agno, Strands, LiteLLM, MCP).

---

## Quick Start

=== "Proxy (Zero Code Changes)"

    ```bash
    pip install "headroom-ai[all]"
    headroom proxy
    ```

    ```bash
    # Point any tool at the proxy
    ANTHROPIC_BASE_URL=http://localhost:8787 claude
    OPENAI_BASE_URL=http://localhost:8787/v1 your-app
    ```

    That's it. Your existing code works unchanged, with 40-90% fewer tokens.

    Want an always-on local runtime instead? See [Persistent Installs &rarr;](persistent-installs.md).

=== "Python SDK"

    ```python
    from headroom import compress

    result = compress(messages, model="claude-sonnet-4-5-20250929")
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        messages=result.messages,
    )
    print(f"Saved {result.tokens_saved} tokens ({result.compression_ratio:.0%})")
    ```

    Works with any Python LLM client. [Full SDK guide &rarr;](sdk.md)

=== "Coding Agents"

    ```bash
    headroom wrap claude       # Claude Code
    headroom wrap copilot -- --model claude-sonnet-4-20250514
    headroom wrap codex        # OpenAI Codex CLI
    headroom wrap aider        # Aider
    headroom wrap cursor       # Cursor
    headroom wrap openclaw     # OpenClaw plugin bootstrap
    ```

    Starts the proxy, points your tool at it, compresses everything automatically.

    If you prefer an always-on proxy that `wrap` can reuse or recover, see [Persistent Installs &rarr;](persistent-installs.md).

=== "TypeScript SDK"

    ```typescript
    import { compress } from 'headroom-ai';

    const result = await compress(messages, { model: 'claude-sonnet-4-5-20250929' });
    // Use result.messages with any LLM client
    console.log(`Saved ${result.tokensSaved} tokens`);
    ```

    Works with Vercel AI SDK, OpenAI Node SDK, and Anthropic TS SDK. [Full TS guide &rarr;](typescript-sdk.md)

=== "LiteLLM Callback"

    ```python
    import litellm
    from headroom.integrations.litellm_callback import HeadroomCallback

    litellm.callbacks = [HeadroomCallback()]
    # All 100+ providers now compressed automatically
    ```

---

## Framework Integrations

<div class="grid-container" markdown>

<div class="grid-item" markdown>

### LangChain

Wrap any chat model. Supports memory, retrievers, tools, streaming, async.

```python
from headroom.integrations import HeadroomChatModel

llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))
```

[LangChain Guide &rarr;](langchain.md)

</div>

<div class="grid-item" markdown>

### Agno

Full agent framework integration with observability hooks.

```python
from headroom.integrations.agno import HeadroomAgnoModel

model = HeadroomAgnoModel(Claude(id="claude-sonnet-4-20250514"))
agent = Agent(model=model)
```

[Agno Guide &rarr;](agno.md)

</div>

<div class="grid-item" markdown>

### Strands

Model wrapping + tool output hook provider for Strands Agents.

```python
from headroom.integrations.strands import HeadroomStrandsModel

model = HeadroomStrandsModel(wrapped_model=bedrock_model)
agent = Agent(model=model)
```

[Strands Guide &rarr;](strands.md)

</div>

<div class="grid-item" markdown>

### MCP Tools

Three tools for Claude Code, Cursor, or any MCP client: `headroom_compress`, `headroom_retrieve`, `headroom_stats`.

```bash
headroom mcp install && claude
```

[MCP Guide &rarr;](mcp.md)

</div>

<div class="grid-item" markdown>

### TypeScript SDK

`compress()`, Vercel AI SDK middleware, OpenAI and Anthropic client wrappers.

```bash
npm install headroom-ai
```

[TypeScript SDK Guide &rarr;](typescript-sdk.md)

</div>

<div class="grid-item" markdown>

### OpenClaw

ContextEngine plugin for OpenClaw agents. Auto-compresses context in `assemble()`.

```bash
headroom wrap openclaw
```

[OpenClaw Plugin &rarr;](https://github.com/chopratejas/headroom/tree/main/plugins/openclaw)

</div>

</div>

[All integration patterns &rarr;](integration-guide.md){ .md-button }

---

## How It Works

Headroom runs a three-stage pipeline on every request:

```mermaid
graph LR
    A[Your Prompt] --> B[CacheAligner]
    B --> C[ContentRouter]
    C --> D[IntelligentContext]
    D --> E[LLM Provider]

    C -->|JSON| F[SmartCrusher]
    C -->|Code| G[CodeCompressor]
    C -->|Text| H[Kompress]
    C -->|Logs| I[LogCompressor]

    F --> D
    G --> D
    H --> D
    I --> D
```

**Stage 1: CacheAligner** — Stabilizes message prefixes so the provider's KV cache actually hits. Claude offers a 90% read discount on cached prefixes; CacheAligner makes that work.

**Stage 2: ContentRouter** — Auto-detects content type (JSON, code, logs, search results, diffs, HTML, plain text) and routes each to the optimal compressor:

| Content Type | Compressor | How It Works |
|-------------|-----------|-------------|
| JSON arrays | **SmartCrusher** | Statistical analysis: keeps errors, anomalies, boundaries. No hardcoded rules. |
| Source code | **CodeCompressor** | AST-aware (tree-sitter). Preserves function signatures, collapses bodies. |
| Plain text | **Kompress** | ModernBERT token classification. Removes redundant tokens while preserving meaning. |
| Build/test logs | **LogCompressor** | Keeps failures, errors, warnings. Drops passing noise. |
| Search results | **SearchCompressor** | Ranks by relevance to user query, keeps top matches. |
| Git diffs | **DiffCompressor** | Preserves change hunks, drops unchanged context. |
| HTML | **HTMLExtractor** | Strips markup, extracts readable content. |

**Stage 3: IntelligentContext** — If the conversation still exceeds the model's context limit, scores each message by importance (recency, references, density) and drops the lowest-value ones.

**Nothing is lost.** Compressed content goes into the CCR store (Compress-Cache-Retrieve). The LLM gets a `headroom_retrieve` tool and can fetch full originals when it needs more detail.

[Full architecture deep dive &rarr;](ARCHITECTURE.md)

---

## Results

**100 production log entries. One critical error buried at position 67.**

| Metric | Baseline | Headroom |
|--------|----------|----------|
| Input tokens | 10,144 | 1,260 |
| Correct answers | **4/4** | **4/4** |

**87.6% fewer tokens. Same answer.** The FATAL error was automatically preserved — not by keyword matching, but by statistical analysis of field variance.

### Real Workloads

| Scenario | Before | After | Savings |
|----------|--------|-------|---------|
| Code search (100 results) | 17,765 | 1,408 | **92%** |
| SRE incident debugging | 65,694 | 5,118 | **92%** |
| Codebase exploration | 78,502 | 41,254 | **47%** |
| GitHub issue triage | 54,174 | 14,761 | **73%** |

### Accuracy Benchmarks

| Benchmark | Category | N | Accuracy | Compression |
|-----------|----------|---|----------|-------------|
| GSM8K | Math | 100 | 0.870 | 0.000 delta |
| TruthfulQA | Factual | 100 | 0.560 | +0.030 delta |
| SQuAD v2 | QA | 100 | **97%** | 19% reduction |
| BFCL | Tool/Function | 100 | **97%** | 32% reduction |
| CCR Needle | Lossless | 50 | **100%** | 77% reduction |

[Full benchmark methodology &rarr;](benchmarks.md) | [Known limitations &rarr;](LIMITATIONS.md)

---

## Key Features

<div class="grid-container" markdown>

<div class="grid-item" markdown>
### Lossless Compression (CCR)
Compresses aggressively, stores originals, gives the LLM a tool to retrieve full details. Nothing is thrown away.
[Learn more &rarr;](ccr.md)
</div>

<div class="grid-item" markdown>
### Smart Content Detection
Auto-detects JSON, code, logs, text, diffs, HTML. Routes each to the best compressor. Zero configuration needed.
[Learn more &rarr;](compression.md)
</div>

<div class="grid-item" markdown>
### Cache Optimization
Stabilizes prefixes so provider KV caches hit. Tracks frozen messages to preserve the 90% read discount.
[Learn more &rarr;](ccr.md)
</div>

<div class="grid-item" markdown>
### Image Compression
40-90% token reduction via trained ML router. Automatically selects resize/quality tradeoff per image.
[Learn more &rarr;](image-compression.md)
</div>

<div class="grid-item" markdown>
### Persistent Memory
Hierarchical memory (user/session/agent/turn) with SQLite + HNSW backends. Survives across conversations.
[Learn more &rarr;](memory.md)
</div>

<div class="grid-item" markdown>
### Failure Learning
Reads past sessions, finds failed tool calls, correlates with what succeeded, writes learnings to CLAUDE.md.
[Learn more &rarr;](learn.md)
</div>

<div class="grid-item" markdown>
### Multi-Agent Context
Compress what moves between agents. Any framework.
```python
ctx = SharedContext()
ctx.put("research", big_output)
summary = ctx.get("research")  # ~80% smaller
```
[Learn more &rarr;](shared-context.md)
</div>

<div class="grid-item" markdown>
### Metrics & Observability
Prometheus endpoint, per-request logging, cost tracking, budget limits, pipeline timing breakdowns.
[Learn more &rarr;](metrics.md)
</div>

</div>

---

## Cloud Providers

Works with any LLM provider out of the box:

```bash
headroom proxy                                          # Direct Anthropic/OpenAI
headroom proxy --backend bedrock --region us-east-1     # AWS Bedrock
headroom proxy --backend vertex_ai --region us-central1 # Google Vertex AI
headroom proxy --backend azure                          # Azure OpenAI
headroom proxy --backend openrouter                     # OpenRouter (400+ models)
```

Or via LiteLLM for 100+ providers (Together, Groq, Fireworks, Ollama, vLLM, etc.).

---

## Installation

```bash
pip install headroom-ai                # Core library (Python)
pip install "headroom-ai[all]"         # Everything (recommended)
npm install headroom-ai                # TypeScript / Node.js
pip install "headroom-ai[proxy]"       # Proxy server + MCP tools
pip install "headroom-ai[ml]"          # ML compression (Kompress, requires torch)
pip install "headroom-ai[langchain]"   # LangChain integration
pip install "headroom-ai[agno]"        # Agno integration
pip install "headroom-ai[evals]"       # Evaluation framework
```

Requires Python 3.10+.

---

## Next Steps

- **[Quickstart](quickstart.md)** — Running in 5 minutes
- **[Integration Guide](integration-guide.md)** — Every way to add Headroom to your stack
- **[Architecture](ARCHITECTURE.md)** — How the pipeline works under the hood
- **[Benchmarks](benchmarks.md)** — Accuracy and latency data
- **[Limitations](LIMITATIONS.md)** — When compression helps and when it doesn't
- **[Filesystem Contract](filesystem-contract.md)** — Canonical config/workspace env vars and paths

---

Apache 2.0 — Free for commercial use. [GitHub](https://github.com/chopratejas/headroom) | [PyPI](https://pypi.org/project/headroom-ai/) | [Discord](https://discord.gg/yRmaUNpsPJ)
