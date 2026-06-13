# Memory

**Hierarchical, temporal memory for LLM applications.** Enable your AI to remember across conversations with intelligent scoping and versioning.

## Why Memory?

LLMs have two fundamental limitations:
1. **Context windows overflow** - Too much history, need to truncate
2. **No persistence** - Every conversation starts from zero

Memory solves both: **extract key facts, persist them, inject when relevant.**

This is *temporal compression* - instead of carrying 10,000 tokens of conversation history, carry 100 tokens of extracted memories.

---

## What Makes Headroom Memory Different?

| Feature | Headroom | Letta (MemGPT) | Mem0 |
|---------|----------|----------------|------|
| **Cross-Agent Memory** | Any agent shares one DB via proxy | Per-agent only | Per-user, no cross-agent |
| **Agent Provenance** | Tracks which agent saved/updated each memory | No | No |
| **LLM-Mediated Dedup** | Piggybacks on user's own LLM for merge decisions | No | Separate LLM call ($) |
| **Transparent Proxy** | Zero code changes — just route through proxy | Requires agent framework | Requires SDK integration |
| **Hierarchical Scoping** | User → Session → Agent → Turn | Flat (per-agent) | Flat (per-user) |
| **Temporal Versioning** | Full supersession chains | No | No |
| **Zero-Latency Extraction** | Inline (Letta-style) | Inline | Separate call |
| **One-Liner Integration** | `with_memory(client)` | Requires agent setup | Requires separate client |
| **Pluggable Backends** | SQLite, HNSW, FTS5, any embedder | PostgreSQL | Qdrant/Chroma |
| **Semantic + Full-Text Search** | Both | Semantic only | Semantic only |
| **Memory Bubbling** | Auto-promote important memories | No | No |
| **Protocol-Based Architecture** | Yes (dependency injection) | No | No |

---

## Cross-Agent Memory (Proxy)

The most powerful way to use memory: **any agent that routes through the proxy shares the same memory store.** Claude saves a fact, Codex reads it back. Zero configuration needed.

```bash
# Start the proxy with memory enabled
headroom proxy --memory

# Or use wrap (auto-starts proxy)
headroom wrap claude --memory    # Claude Code with persistent memory
headroom wrap codex --memory     # Codex with the SAME memory store
headroom wrap aider --memory     # Aider shares it too
```

### How It Works

```
Claude Code                  Codex CLI                  Gemini CLI
     │                          │                          │
     └── /v1/messages ──┐      └── /v1/chat/completions ──┤     └── /generateContent ──┐
                        │                                  │                            │
                        ▼                                  ▼                            ▼
                    ┌──────────────────────────────────────────────────────────────────┐
                    │                    Headroom Proxy (--memory)                      │
                    │                                                                   │
                    │  1. Search memory DB for relevant context                        │
                    │  2. Inject memories as system context (provider-native format)   │
                    │  3. Add memory_save/search/update/delete tools                   │
                    │  4. Forward to upstream LLM                                       │
                    │  5. Handle memory tool calls in response                         │
                    │  6. Async background dedup (>92% cosine → auto-remove)          │
                    │                                                                   │
                    └──────────────────────┬───────────────────────────────────────────┘
                                           │
                                           ▼
                               .headroom/memory.db
                               (project-scoped SQLite)
```

### Project-Scoped Database

Memory is stored per-project at `{cwd}/.headroom/memory.db`. Each
project has its own memory — no cross-project contamination. Override
with `--memory-db-path` for a custom location.

> **Filesystem contract note.** Project-scoped memory paths resolve
> relative to the current working directory and **do not** obey the
> canonical `HEADROOM_WORKSPACE_DIR` env var. This preserves the
> project-memory isolation invariant. Users who want a single central
> memory store should pass `--memory-db-path` explicitly. See the
> [Filesystem Contract](filesystem-contract.md) for the rationale.

### User Identity

User ID is auto-detected from `$USER` (your OS username). Override per-request with the `x-headroom-user-id` header. All memories are scoped to the user — multiple developers on the same project have separate memory stores.

### Agent Provenance

Every memory tracks which agent created or updated it:

```json
{
  "content": "Project uses alembic for migrations",
  "metadata": {
    "source_agent": "claude",
    "source_provider": "anthropic",
    "created_via": "tool_call",
    "created_at_utc": "2026-04-10T17:30:00Z"
  }
}
```

When an agent updates a memory, the update is tracked:

```json
{
  "reason": "Updated by codex via openai: Added version info"
}
```

### Intelligent Deduplication

When the LLM calls `memory_save`, headroom:

1. **Saves immediately** (zero latency)
2. **Searches for similar existing memories** (cosine similarity)
3. **Returns an enriched hint** if duplicates found:

```json
{
  "status": "saved",
  "memory_id": "abc123",
  "note": "Similar memory exists (id: def456, 89% match, saved by codex):
           'DB migration tool is alembic'. Call memory_update('def456',
           '<merged content>') to consolidate."
}
```

The LLM then decides whether to merge — using the user's own LLM, not a separate model. No extra cost to headroom.

4. **Background auto-dedup**: If similarity >92%, the older duplicate is automatically removed (async, non-blocking).

### Supported Providers

Memory works with ALL providers routing through the proxy:

| Provider | Context Injection | Memory Tools | Format |
|----------|-------------------|--------------|--------|
| **Anthropic** (Claude) | System parameter | Anthropic tool_use | Native |
| **OpenAI** (Codex, GPT) | System message | OpenAI function calling | Native |
| **Gemini** | systemInstruction | functionDeclarations | Native |
| **Any OpenAI-compatible** | System message | Function calling | OpenAI format |

---

## Quick Start

```python
from openai import OpenAI
from headroom import with_memory

# One line - that's it
client = with_memory(OpenAI(), user_id="alice")

# Use exactly like normal
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "I prefer Python for backend work"}]
)
# Memory extracted INLINE - zero extra latency

# Later, in a new conversation...
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What language should I use?"}]
)
# → Response uses the Python preference from memory
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                      with_memory()                          │
│                                                              │
│   1. INJECT: Semantic search → prepend to user message      │
│   2. INSTRUCT: Add memory extraction instruction            │
│   3. CALL: Forward to LLM                                   │
│   4. PARSE: Extract <memory> block from response            │
│   5. STORE: Save with embeddings + vector index + FTS       │
│   6. RETURN: Clean response (without memory block)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: Memory extraction happens *inline* as part of the LLM response (Letta-style). No extra API calls, no extra latency.

---

## Hierarchical Scoping

Memories exist at different scope levels, enabling fine-grained control:

```
USER (broadest)
 └── SESSION
      └── AGENT
           └── TURN (narrowest)
```

### Scope Levels

| Scope | Persists Across | Use Case |
|-------|-----------------|----------|
| **USER** | All sessions, all time | Long-term preferences, identity |
| **SESSION** | Current session only | Current task context |
| **AGENT** | Current agent in session | Agent-specific context |
| **TURN** | Single turn only | Ephemeral working memory |

### Example: Multi-Session Memory

```python
from openai import OpenAI
from headroom import with_memory

# Session 1: Morning
client1 = with_memory(
    OpenAI(),
    user_id="bob",
    session_id="morning-session",
)
response = client1.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "I prefer Go for performance-critical code"}]
)
# Memory stored at USER level (persists across sessions)

# Session 2: Afternoon (different session, same user)
client2 = with_memory(
    OpenAI(),
    user_id="bob",  # Same user
    session_id="afternoon-session",  # Different session
)
response = client2.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What language for my new microservice?"}]
)
# → Recalls Go preference from morning session!
```

---

## Temporal Versioning (Supersession)

Memories evolve over time. When facts change, Headroom creates a **supersession chain** preserving history:

```python
from headroom.memory import HierarchicalMemory, MemoryConfig

memory = await HierarchicalMemory.create()

# Original fact
orig = await memory.add(
    content="User works at Google",
    user_id="alice",
    category=MemoryCategory.FACT,
)

# User changes jobs - supersede the old memory
new = await memory.supersede(
    old_memory_id=orig.id,
    new_content="User now works at Anthropic",
)

# Query current state (excludes superseded)
current = await memory.query(MemoryFilter(
    user_id="alice",
    include_superseded=False,  # Default
))
# → Returns only "User now works at Anthropic"

# Query full history (includes superseded)
history = await memory.query(MemoryFilter(
    user_id="alice",
    include_superseded=True,
))
# → Returns both memories with validity timestamps

# Get the chain
chain = await memory.get_history(new.id)
# → [
#     Memory(content="User works at Google", valid_until=..., is_current=False),
#     Memory(content="User now works at Anthropic", valid_until=None, is_current=True),
#   ]
```

### Why Temporal Versioning Matters

1. **Audit trail** - Know what was true at any point in time
2. **Debugging** - Understand why the LLM made certain decisions
3. **Rollback** - Restore previous state if needed
4. **Analytics** - Track how user preferences evolve

---

## Memory Categories

Memories are categorized for better organization and retrieval:

| Category | Description | Examples |
|----------|-------------|----------|
| `PREFERENCE` | Likes, dislikes, preferred approaches | "Prefers Python", "Likes dark mode" |
| `FACT` | Identity, role, constraints | "Works at fintech startup", "Senior engineer" |
| `CONTEXT` | Current goals, ongoing tasks | "Migrating to microservices", "Working on auth" |
| `ENTITY` | Information about entities | "Project Apollo uses React", "Team lead is Sarah" |
| `DECISION` | Decisions made | "Chose PostgreSQL over MySQL", "Using REST not GraphQL" |
| `INSIGHT` | Derived insights | "User tends to prefer typed languages" |

---

## Memory API

The `with_memory()` wrapper provides a `.memory` API for direct access:

```python
client = with_memory(OpenAI(), user_id="alice")

# Search memories (semantic)
results = client.memory.search("python preferences", top_k=5)
for memory in results:
    print(f"{memory.content}")

# Add manual memory
client.memory.add(
    "User is a senior engineer",
    category="fact",
    importance=0.9,
)

# Get all memories
all_memories = client.memory.get_all()

# Clear memories
client.memory.clear()

# Get stats
stats = client.memory.stats()
print(f"Total memories: {stats['total']}")
print(f"By category: {stats['categories']}")
```

---

## Advanced Usage: Direct HierarchicalMemory API

For full control, use the `HierarchicalMemory` class directly:

```python
import asyncio
from headroom.memory import (
    HierarchicalMemory,
    MemoryConfig,
    MemoryCategory,
    EmbedderBackend,
)
from headroom.memory.ports import MemoryFilter, VectorFilter

async def main():
    # Create with custom configuration
    config = MemoryConfig(
        db_path="my_memory.db",
        embedder_backend=EmbedderBackend.LOCAL,  # or OPENAI, OLLAMA
        vector_dimension=384,
        cache_max_size=2000,
    )
    memory = await HierarchicalMemory.create(config)

    # Add memory with full control
    mem = await memory.add(
        content="User prefers functional programming",
        user_id="alice",
        session_id="sess-123",
        agent_id="code-assistant",
        category=MemoryCategory.PREFERENCE,
        importance=0.9,
        entity_refs=["functional-programming", "coding-style"],
        metadata={"source": "conversation", "confidence": 0.95},
    )

    # Semantic search
    results = await memory.search(
        query="programming paradigm preferences",
        user_id="alice",
        top_k=5,
        min_similarity=0.5,
        categories=[MemoryCategory.PREFERENCE],
    )
    for r in results:
        print(f"[{r.similarity:.3f}] {r.memory.content}")

    # Full-text search
    text_results = await memory.text_search(
        query="functional",
        user_id="alice",
    )

    # Query with filters
    memories = await memory.query(MemoryFilter(
        user_id="alice",
        categories=[MemoryCategory.PREFERENCE, MemoryCategory.FACT],
        min_importance=0.7,
        limit=10,
    ))

    # Convenience methods
    await memory.remember("Likes coffee", user_id="alice", importance=0.6)
    relevant = await memory.recall("beverage preferences", user_id="alice")

asyncio.run(main())
```

---

## Configuration

### Embedder Backends

```python
from headroom.memory import MemoryConfig, EmbedderBackend

# Local embeddings (recommended - fast, free, private)
config = MemoryConfig(
    embedder_backend=EmbedderBackend.LOCAL,
    embedder_model="all-MiniLM-L6-v2",  # 384 dimensions, fast
)

# OpenAI embeddings (higher quality, costs money)
config = MemoryConfig(
    embedder_backend=EmbedderBackend.OPENAI,
    openai_api_key="sk-...",
    embedder_model="text-embedding-3-small",
)

# Ollama embeddings (local server, many models)
config = MemoryConfig(
    embedder_backend=EmbedderBackend.OLLAMA,
    ollama_base_url="http://localhost:11434",
    embedder_model="nomic-embed-text",
)
```

### Embedding Runtime / GPU Offload (Apple Silicon)

By default the proxy's memory embedder runs on the **ONNX CPU** backend. This
is fast and dependency-light, but it is CPU-only — under sustained load the
embedding step can saturate the CPU and make the proxy less responsive.

On Apple Silicon you can opt in to running the embedder on the **Apple GPU
(MPS)** instead, which offloads that work off the CPU and keeps the proxy
responsive. This is especially useful on fanless Macs (e.g. the M5 Air) that
are prone to CPU-saturation timeouts.

Enable it by installing the extra and setting the env var:

```bash
pip install 'headroom-ai[pytorch-mps]'   # also works as [pytorch_mps]
export HEADROOM_EMBEDDER_RUNTIME=pytorch_mps
```

When set, the embedder runs via the torch sentence-transformers backend on the
Apple GPU instead of the default ONNX CPU embedder. Notes:

- **Strictly opt-in.** `pytorch_mps` is the only accepted value; anything else
  (or unset) keeps the default ONNX CPU embedder. Default behavior is unchanged.
- **Auto-fallback.** It only activates when Apple MPS is actually available
  (Apple Silicon + torch). If MPS is unavailable or torch/sentence-transformers
  is not installed, it logs a warning and uses the existing default embedder
  selection path: ONNX when available, then the pre-existing local
  sentence-transformers fallback.
- **MPS serialization.** torch-MPS is not thread-safe, so the embedder
  serializes MPS encode calls internally via a single-worker executor. This is
  automatic — there is nothing to configure.

### Storage Configuration

```python
config = MemoryConfig(
    db_path="memory.db",          # SQLite database path
    vector_dimension=384,          # Must match embedder output
    hnsw_ef_construction=200,      # HNSW index quality (higher = better, slower)
    hnsw_m=16,                     # HNSW connections per node
    hnsw_ef_search=50,             # HNSW search quality
    cache_enabled=True,            # Enable LRU cache
    cache_max_size=1000,           # Max cached memories
)
```

### Wrapper Configuration

```python
client = with_memory(
    OpenAI(),
    user_id="alice",
    db_path="memory.db",
    top_k=5,                       # Memories to inject per request
    session_id="optional-session",
    agent_id="optional-agent",
    embedder_backend=EmbedderBackend.LOCAL,
)
```

---

## Architecture

### Protocol-Based Design

Headroom Memory uses **Protocol interfaces** (ports) for all components, enabling easy swapping:

```
┌─────────────────────────────────────────────────────────────┐
│                   HierarchicalMemory                        │
│                     (Orchestrator)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ MemoryStore │  │ VectorIndex │  │  TextIndex  │        │
│  │  Protocol   │  │  Protocol   │  │  Protocol   │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐        │
│  │   SQLite    │  │    HNSW     │  │    FTS5     │        │
│  │  Adapter    │  │   Adapter   │  │   Adapter   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │  Embedder   │  │ MemoryCache │                          │
│  │  Protocol   │  │  Protocol   │                          │
│  └──────┬──────┘  └──────┬──────┘                          │
│         │                │                                  │
│  ┌──────▼──────┐  ┌──────▼──────┐                          │
│  │Local/OpenAI/│  │  LRU Cache  │                          │
│  │   Ollama    │  │             │                          │
│  └─────────────┘  └─────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Protocol | Default Adapter | Purpose |
|-----------|----------|-----------------|---------|
| **MemoryStore** | `MemoryStore` | `SQLiteMemoryStore` | CRUD + filtering + supersession |
| **VectorIndex** | `VectorIndex` | `HNSWVectorIndex` | Semantic similarity search |
| **TextIndex** | `TextIndex` | `FTS5TextIndex` | Full-text keyword search |
| **Embedder** | `Embedder` | `LocalEmbedder` | Text → vector conversion |
| **Cache** | `MemoryCache` | `LRUMemoryCache` | Hot memory caching |

---

## Comparison with State of the Art

### vs Letta (MemGPT)

**Letta** pioneered inline memory extraction. Headroom builds on this with:

| Aspect | Headroom | Letta |
|--------|----------|-------|
| **Scoping** | 4-level hierarchy (user/session/agent/turn) | Flat per-agent |
| **Temporal** | Full supersession chains with history | No versioning |
| **Integration** | One-liner wrapper for any client | Requires Letta agent framework |
| **Search** | Semantic + full-text | Semantic only |
| **Storage** | SQLite + HNSW (embedded) | PostgreSQL (external) |
| **Extensibility** | Protocol-based adapters | Monolithic |

**When to use Letta**: You want a full agent framework with built-in memory.
**When to use Headroom**: You want memory as a layer on your existing stack.

### vs Mem0

**Mem0** provides a managed memory service. Headroom differs:

| Aspect | Headroom | Mem0 |
|--------|----------|------|
| **Deployment** | Embedded (no server) | Managed service or self-hosted |
| **Scoping** | 4-level hierarchy | Flat per-user |
| **Temporal** | Supersession chains | No versioning |
| **Extraction** | Inline (zero latency) | Separate API call |
| **Search** | Semantic + full-text | Semantic only |
| **Cost** | Free (local embeddings) | API costs or infra costs |
| **Privacy** | All local | Data leaves your infra |

**When to use Mem0**: You want a managed service and don't mind external dependencies.
**When to use Headroom**: You want embedded memory with no external services.

### Feature Matrix

| Feature | Headroom | Letta | Mem0 |
|---------|:--------:|:-----:|:----:|
| Cross-agent sharing (proxy) | ✅ | ❌ | ❌ |
| Agent provenance tracking | ✅ | ❌ | ❌ |
| LLM-mediated dedup (no extra cost) | ✅ | ❌ | ❌ (uses separate LLM) |
| Transparent proxy (zero code) | ✅ | ❌ | ❌ |
| Hierarchical scoping | ✅ | ❌ | ❌ |
| Temporal versioning | ✅ | ❌ | ❌ |
| Zero-latency extraction | ✅ | ✅ | ❌ |
| Full-text search | ✅ | ❌ | ❌ |
| Embedded (no server) | ✅ | ❌ | ❌ |
| One-liner integration | ✅ | ❌ | ❌ |
| Protocol-based extensibility | ✅ | ❌ | ❌ |
| Memory bubbling | ✅ | ❌ | ❌ |
| Local embeddings | ✅ | ❌ | ✅ |
| Managed service option | ❌ | ❌ | ✅ |

---

## Multi-User Isolation

Memories are isolated by `user_id`:

```python
# Alice's memories
alice_client = with_memory(OpenAI(), user_id="alice")

# Bob's memories (completely separate)
bob_client = with_memory(OpenAI(), user_id="bob")

# Bob cannot see Alice's memories, even with the same database
```

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Memory injection | <50ms | Local embeddings + HNSW search |
| Memory extraction | +50-100 tokens | Part of LLM response (inline) |
| Memory storage | <10ms | SQLite + HNSW + FTS5 indexing |
| Cache hit | <1ms | LRU cache lookup |

**Overhead**: ~100 extra output tokens per response for the `<memory>` block.

---

## Providers

Memory works with any OpenAI-compatible client:

```python
from openai import OpenAI
from headroom import with_memory

# OpenAI
client = with_memory(OpenAI(), user_id="alice")

# Azure OpenAI
client = with_memory(
    OpenAI(base_url="https://your-resource.openai.azure.com/..."),
    user_id="alice",
)

# Groq
from groq import Groq
client = with_memory(Groq(), user_id="alice")

# Any OpenAI-compatible client
client = with_memory(YourClient(), user_id="alice")
```

---

## Example: Full Conversation Flow

```python
from openai import OpenAI
from headroom import with_memory

client = with_memory(OpenAI(), user_id="developer_jane")

# Conversation 1: User shares context
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "I'm a Python developer at a fintech startup. We use PostgreSQL and FastAPI."
    }]
)
# Memories extracted:
#   - [FACT] Python developer at fintech startup
#   - [PREFERENCE] Uses PostgreSQL for databases
#   - [PREFERENCE] Uses FastAPI for web APIs

# Conversation 2 (new session): User asks question
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "What database should I use for my new project?"
    }]
)
# Response references PostgreSQL preference from memory:
# → "Given your experience with PostgreSQL at your fintech company,
#    I'd recommend sticking with it for consistency..."

# Check stored memories
print("Stored memories:")
for m in client.memory.get_all():
    print(f"  [{m.category.value}] {m.content}")
```

---

## Troubleshooting

### Memories not being extracted

1. Check if the conversation has memory-worthy content (not just greetings)
2. Verify the LLM is following the memory instruction
3. Enable logging: `import logging; logging.basicConfig(level=logging.DEBUG)`

### Memories not being retrieved

1. Verify `user_id` matches between sessions
2. Check if memories exist: `client.memory.get_all()`
3. Try a more specific search query
4. Check similarity threshold

### High latency

1. Use local embeddings: `embedder_backend=EmbedderBackend.LOCAL`
2. Reduce `top_k` for fewer memories to retrieve
3. Enable caching (enabled by default)

### Memory not persisting

1. Check `db_path` is the same across sessions
2. Ensure the database file is writable
3. Check for exceptions in logs

---

## Best Practices

1. **Use consistent `user_id`** - Same ID across sessions for continuity
2. **Use session scoping** - Set `session_id` for session-specific context
3. **Start with local embeddings** - Faster, free, good enough for most cases
4. **Monitor memory growth** - Use `client.memory.stats()` to track
5. **Use importance scores** - Higher importance = more likely to be retrieved
6. **Leverage categories** - Helps with debugging and selective retrieval
7. **Consider supersession** - Use `supersede()` when facts change, not `add()`
