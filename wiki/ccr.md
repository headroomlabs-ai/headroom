# CCR: Compress-Cache-Retrieve

Headroom's CCR architecture makes compression **reversible**. When content is compressed, the original data is cached. If the LLM needs more data, it can retrieve it instantly.

## The Problem with Traditional Compression

Traditional compression is lossy — if you guess wrong about what's important, data is lost forever. This creates a difficult tradeoff:

- **Aggressive compression**: Risk losing data the LLM needs
- **Conservative compression**: Miss out on token savings

CCR eliminates this tradeoff.

## CCR-Enabled Components

| Component | What it compresses | CCR integration |
|-----------|-------------------|-----------------|
| **SmartCrusher** | JSON arrays (tool outputs) | Stores original array, marker includes hash |
| **ContentRouter** | Code, logs, search results, text | Stores original content by strategy |

## How CCR Works

```
┌─────────────────────────────────────────────────────────────────┐
│  TOOL OUTPUT (1000 items)                                        │
│  └─ SmartCrusher compresses to 20 items                         │
│  └─ Original cached with hash=abc123                            │
│  └─ Retrieval tool injected into context                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM PROCESSING                                                  │
│  Option A: LLM solves task with 20 items → Done (90% savings)   │
│  Option B: LLM calls headroom_retrieve(hash=abc123)             │
│            → Response Handler executes retrieval automatically  │
│            → LLM receives full data, responds accurately        │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: Compression Store

When SmartCrusher compresses tool output:
1. Original content is stored in an LRU cache
2. A hash key is generated for retrieval
3. A marker is added to the compressed output: `[1000 items compressed to 20. Retrieve more: hash=abc123]`

### Phase 2: Tool Injection

Headroom injects a `headroom_retrieve` tool into the LLM's available tools:

```json
{
  "name": "headroom_retrieve",
  "description": "Retrieve original uncompressed data from Headroom cache",
  "parameters": {
    "hash": "The hash key from the compression marker"
  }
}
```

### Phase 3: Response Handler

When the LLM calls `headroom_retrieve`:
1. Response Handler intercepts the tool call
2. Retrieves data from the local cache (~1ms)
3. Adds the result to the conversation
4. Continues the API call automatically

**The client never sees CCR tool calls** — they're handled transparently.

### Phase 4: Context Tracker

Across multiple turns, the Context Tracker:
1. Remembers what was compressed in earlier turns
2. Analyzes new queries for relevance to compressed content
3. Proactively expands relevant data before the LLM asks

**Example:**
```
Turn 1: User searches for files
        → Tool returns 500 files
        → SmartCrusher compresses to 15, caches original (hash=abc123)
        → LLM sees 15 files, answers question

Turn 5: User asks "What about the auth middleware?"
        → Context Tracker detects "auth" might be in abc123
        → Proactively expands compressed content
        → LLM sees full file list, finds auth_middleware.py
```

## CCR Stores Content Blocks, Not Dropped Messages

Headroom never drops whole messages from conversation history. CCR is purely about compressed **content blocks** — the newest tool outputs, tool results, and user content that the live-zone pipeline compresses. The original block is stored in the cache and is retrievable on demand:

```
┌─────────────────────────────────────────────────────────────────┐
│  LATEST TOOL RESULT (500 files, 12K tokens)                      │
│  └─ ContentRouter / SmartCrusher compresses the block           │
│  └─ Original cached with hash=def456                            │
│  └─ Marker inserted: "500 items compressed, retrieve: def456"   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM PROCESSING                                                  │
│  Option A: LLM solves task with the compressed block → Done     │
│  Option B: LLM needs the full content                           │
│            → Calls headroom_retrieve(hash=def456)               │
│            → Full original block restored                        │
└─────────────────────────────────────────────────────────────────┘
```

The older conversation turns, system prompt, and tool definitions — the provider cache hot zone — are never mutated, so prompt caching keeps working. Compression happens only on the live zone (the newest content blocks) and is fully reversible via CCR.

**TOIN integration:** When users retrieve compressed content, TOIN learns to treat those patterns as higher value next time, improving future compression decisions across all users.

## Features

| Feature | Description |
|---------|-------------|
| **Automatic Response Handling** | When LLM calls `headroom_retrieve`, the proxy handles it automatically |
| **Multi-Turn Context Tracking** | Tracks compressed content across turns, proactively expands when relevant |
| **Hash-Keyed Retrieval** | `headroom_retrieve(hash)` always returns the full original content |
| **Feedback Learning** | Learns from retrieval patterns to improve future compression |

## Configuration

```bash
# Proxy with CCR enabled (default)
headroom proxy --port 8787

# Disable CCR response handling
headroom proxy --no-ccr-responses

# Disable proactive expansion
headroom proxy --no-ccr-expansion
```

## Why This Matters

| Approach | Risk | Savings |
|----------|------|---------|
| No compression | None | 0% |
| Traditional compression | Data loss | 70-90% |
| CCR compression | None (reversible) | 70-90% |

CCR gives you the savings of aggressive compression with zero risk — the LLM can always retrieve the original data if needed.

## Demo

Run the CCR demonstration to see it in action:

```bash
python examples/ccr_demo.py
```

Output:
```
1. COMPRESSION STORE
   Original: 100 items (7,059 chars)
   Compressed: 8 items (633 chars)
   Reduction: 91.0%

3. RESPONSE HANDLER
   Detected CCR tool call: True
   Retrieved 100 items automatically

4. CONTEXT TRACKER
   Turn 5: User asks "show authentication middleware"
   Tracker found 1 relevant context
   → relevance=0.73
   Proactively expanded: 100 items
```

## Architecture

For implementation details, see [ARCHITECTURE.md](ARCHITECTURE.md#ccr-compress-cache-retrieve).
