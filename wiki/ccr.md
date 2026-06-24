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
| **IntelligentContextManager** | Messages (conversation turns) | Stores dropped messages, marker includes hash |

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
    "hash": "The hash key from the compression marker",
    "query": "Optional: search within the cached data"
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

## Message-Level CCR (IntelligentContext)

IntelligentContextManager is a **message-level compressor**. When it drops low-importance messages to fit the context budget, those messages are stored in CCR:

```
┌─────────────────────────────────────────────────────────────────┐
│  LONG CONVERSATION (100 messages, 50K tokens)                    │
│  └─ IntelligentContext scores messages by importance            │
│  └─ Drops 60 low-scoring messages                               │
│  └─ Dropped messages cached with hash=def456                    │
│  └─ Marker inserted: "60 messages dropped, retrieve: def456"    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM PROCESSING                                                  │
│  Option A: LLM solves task with remaining messages → Done       │
│  Option B: LLM needs earlier context                            │
│            → Calls headroom_retrieve(hash=def456)               │
│            → Full conversation restored                          │
└─────────────────────────────────────────────────────────────────┘
```

**The marker includes the CCR reference:**
```
[Earlier context compressed: 60 message(s) dropped by importance scoring.
Full content available via ccr_retrieve tool with reference 'def456'.]
```

**TOIN integration:** When users retrieve dropped messages, TOIN learns to score those message patterns higher next time, improving future drop decisions across all users.

## Features

| Feature | Description |
|---------|-------------|
| **Automatic Response Handling** | When LLM calls `headroom_retrieve`, the proxy handles it automatically |
| **Multi-Turn Context Tracking** | Tracks compressed content across turns, proactively expands when relevant |
| **BM25 Search** | LLM can search within compressed data: `headroom_retrieve(hash, query="errors")` |
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
