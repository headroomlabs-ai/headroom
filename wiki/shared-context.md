# SharedContext — Compressed Inter-Agent Context Sharing

When agents hand off to each other, context gets replayed in full. SharedContext compresses what moves between agents using Headroom's compression pipeline.

## Quick Start

```python
from headroom import SharedContext

ctx = SharedContext()

# Agent A stores large output
ctx.put("research", big_research_output, agent="researcher")

# Agent B gets compressed version (~80% smaller)
summary = ctx.get("research")

# Agent B needs full details
full = ctx.get("research", full=True)
```

## API

### `put(key, content, *, agent=None)`

Store content under a key. Compresses automatically using Headroom's full pipeline (SmartCrusher for JSON, CodeCompressor for code, Kompress for text).

```python
entry = ctx.put("findings", big_json_output, agent="researcher")

entry.original_tokens     # 20,000
entry.compressed_tokens   # 4,000
entry.savings_percent     # 80.0
entry.transforms          # ["router:json:0.20"]
```

### `get(key, *, full=False)`

Retrieve content. Returns compressed version by default, original with `full=True`.

```python
compressed = ctx.get("findings")           # 4K tokens
original = ctx.get("findings", full=True)  # 20K tokens
missing = ctx.get("nonexistent")           # None
```

### `get_entry(key)`

Get the full `ContextEntry` with metadata.

```python
entry = ctx.get_entry("findings")
entry.key                # "findings"
entry.agent              # "researcher"
entry.original_tokens    # 20000
entry.compressed_tokens  # 4000
entry.savings_percent    # 80.0
entry.timestamp          # 1710000000.0
entry.transforms         # ["router:json:0.20"]
```

### `keys()`

List all non-expired keys.

### `stats()`

Aggregated stats across all entries.

```python
stats = ctx.stats()
stats.entries                  # 3
stats.total_original_tokens    # 60000
stats.total_compressed_tokens  # 12000
stats.total_tokens_saved       # 48000
stats.savings_percent          # 80.0
```

### `clear()`

Remove all entries.

## Configuration

```python
ctx = SharedContext(
    model="claude-sonnet-4-5-20250929",  # For token counting
    ttl=3600,                             # 1 hour (default)
    max_entries=100,                       # Evicts oldest when full
)
```

## Framework Examples

### CrewAI

```python
from headroom import SharedContext

ctx = SharedContext()

# After researcher task
ctx.put("findings", researcher_task.output.raw)

# Coder task gets compressed context
coder_context = ctx.get("findings")
```

### LangGraph

```python
from headroom import SharedContext

ctx = SharedContext()

def researcher_node(state):
    result = do_research()
    ctx.put("research", result)
    return {"research_summary": ctx.get("research")}

def coder_node(state):
    # Compressed summary in state, full details on demand
    full = ctx.get("research", full=True)
    return {"code": write_code(full)}
```

### OpenAI Agents SDK

```python
from headroom import SharedContext

ctx = SharedContext()

def compress_handoff(messages):
    for msg in messages:
        if len(msg.content) > 1000:
            ctx.put(msg.id, msg.content)
            msg.content = ctx.get(msg.id)
    return messages

handoff(agent=coder, input_filter=compress_handoff)
```

### Any Framework

SharedContext is framework-agnostic. It's just `put()` and `get()`. Use it wherever context moves between agents.

## How It Works

Under the hood, `put()` calls `headroom.compress()` (the same pipeline used by the proxy) and stores the original in memory. `get()` returns the compressed version. `get(full=True)` returns the original.

- JSON arrays → SmartCrusher (70-95% compression)
- Code → CodeCompressor (AST-aware, with `[code]` extra)
- Text → Kompress (ModernBERT, with `[ml]` extra) or passthrough
- Entries expire after TTL (default 1 hour)
- Oldest entries evicted when max_entries reached
