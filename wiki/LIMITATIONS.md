# Headroom Limitations & Known Behavior

Honest documentation of when Headroom helps, when it doesn't, and what to watch out for.

## When Headroom Helps (and When It Doesn't)

| Content Type | Compression | Latency Impact | Best For |
|---|---|---|---|
| **JSON: Arrays of dicts** (search results, API responses, DB rows) | 86-100% | Net latency win on Sonnet/Opus | Primary use case — always use |
| **JSON: Arrays of strings** (file paths, log lines, tags) | 60-90% | Net latency win | New — works with all string arrays |
| **JSON: Arrays of numbers** (metrics, time series) | 70-85% | Net latency win | New — includes statistical summary |
| **JSON: Mixed-type arrays** | 50-70% | Net latency win | New — groups by type, compresses each |
| **Structured logs** (as JSON) | 82-95% | Net latency win | Log entries in tool outputs |
| **Agentic conversations** (25-50 turns) | 56-81% | Break-even to net win | Multi-tool agent sessions |
| **Plain text** (documentation, articles) | 43-46% | Adds latency (cost savings only) | Cost optimization, not speed |
| **Code** | Passthrough | Minimal overhead | See [Code Compression](#code-compression) |
| **RAG document contexts** | Passthrough | Minimal overhead | Not compressed (plain text in user messages) |

See [LATENCY_BENCHMARKS.md](LATENCY_BENCHMARKS.md) for full data with per-scenario timing.

## Code Compression

Headroom includes an AST-aware CodeCompressor (tree-sitter, 8 languages) but it's gated behind safety protections that prevent it from firing in most real-world scenarios. This is intentional.

**Why code mostly passes through:**

1. **Word count gate**: Content under 50 words is silently skipped
2. **Recent code protection** (`protect_recent_code=4`): Code in the last 4 messages is never compressed. In typical tool-call patterns, the tool result is always "recent"
3. **Analysis intent protection** (`protect_analysis_context=True`): If the most recent user message contains keywords like "analyze", "review", "explain", "fix", "debug", "optimize", "error", "bug" — ALL code in the conversation is protected

**Why this is the right default**: Code is almost always fetched because the user wants to work with it. Compressing function bodies would remove exactly what they need. LLMs like Claude are excellent at navigating large code files without compression.

**Where code savings come from**: The IntelligentContextManager drops old code messages that are no longer relevant (scoring-based), which is a better strategy than stripping function bodies from active code.

**Override**: Set `protect_analysis_context=False` in `ContentRouterConfig` for aggressive code compression. Requires `headroom-ai[code]` for tree-sitter.

## JSON Compression Constraints

### What gets compressed
- Arrays of **dicts**: Full statistical analysis with adaptive K (Kneedle algorithm)
- Arrays of **strings**: Dedup + adaptive sampling + error preservation
- Arrays of **numbers**: Statistical summary + outlier/change-point preservation
- **Mixed-type** arrays: Grouped by type, each group compressed independently
- **Nested** objects: Recursed into, arrays within are compressed (up to depth 5)

### What passes through
- Arrays below 5 items (`min_items_to_analyze`)
- Content below 200 tokens (`min_tokens_to_crush`)
- Bool-only arrays (not useful to compress)
- JSON objects without array values
- Malformed JSON (silently passes through, no error)
- Non-JSON content (handled by other pipeline stages)

### Edge cases
- **NaN/Infinity** in numeric fields: Filtered out before statistics are computed
- **Nesting depth > 5**: Inner arrays not examined for compression
- **Mixed-type arrays with small groups**: Groups below `min_items_to_analyze` are kept as-is

## Adaptive K: How Item Retention Works

SmartCrusher doesn't use fixed K values. It uses information-theoretic sizing:

1. **Kneedle algorithm** on bigram coverage curves finds the point where adding more items stops providing new information
2. **SimHash** fingerprinting detects near-duplicate items
3. **zlib validation** ensures the subset captures the full set's diversity
4. The resulting K is split: 30% from array start, 15% from end, 55% for importance-scored items

**Safety guarantees (additive, never dropped):**
- Error items (containing "error", "exception", "failed", "critical", etc.) — across ALL array types
- Numeric anomalies (> 2σ from mean)
- String length anomalies (> 2σ from mean length)
- Change points (sudden shifts in running values)

These are kept even if they exceed the K budget.

## ML Text Compression (Kompress, opt-in)

- **Requires**: `headroom-ai[ml]` — downloads model weights and needs GPU/CPU RAM for inference
- **First call**: model-load latency (cached globally after)
- **Latency**: Adds overhead that doesn't break even on fast models. Use for **cost savings**, not speed
- **Thread safety**: Single global model instance with lock — sequential access under concurrency

> The earlier LLMLingua-2 integration (`headroom-ai[llmlingua]`) was retired and is no longer installable.

## Error Handling

All compressors follow the same principle: **fail gracefully, return original content unchanged**.

- Invalid JSON → passthrough (no error raised)
- AST parse failure in CodeCompressor → falls back to original
- Compression makes output larger → original returned
- Missing optional dependencies (tree-sitter, ML stack) → passthrough with warning log

Errors are logged at WARNING level and never propagated to callers.

## TOIN Cold Start

The Tool Output Intelligence Network (TOIN) learns compression patterns from usage. For new tool types:

- No learned patterns exist → falls back to statistical heuristics
- Confidence below `toin_confidence_threshold` (default 0.3) → TOIN hints ignored
- Patterns build up over time as tools are used repeatedly
- Cross-session learning requires persistence (`TelemetryConfig.storage_path`)

## CacheAligner Behavior

- Only processes **system messages** for dynamic content extraction
- Dynamic content in user/assistant/tool messages is not extracted
- May add small markers (`[Dynamic Context]` separator) that slightly increase token count
- Whitespace normalization may affect content with significant indentation (code blocks, ASCII art)

## Provider Interactions

- CacheAligner is designed to maximize Anthropic/OpenAI prefix cache hit rates
- Token counting uses model-specific tokenizers (tiktoken for OpenAI, calibrated estimation for Anthropic)
- Compression works with all providers — no provider-specific limitations
- Compressed content is valid JSON — downstream tools and parsers work unchanged

## Performance Characteristics

- **ContentRouter** accounts for 91-98% of pipeline cost — it does the actual compression work
- **CacheAligner** and **RollingWindow** are sub-millisecond
- Scaling is roughly **linear** with input size
- Full benchmark data: [LATENCY_BENCHMARKS.md](LATENCY_BENCHMARKS.md)

## Configuration Tuning

| Parameter | Default | Effect |
|---|---|---|
| `min_items_to_analyze` | 5 | Arrays below this pass through |
| `min_tokens_to_crush` | 200 | Content below this passes through |
| `max_items_after_crush` | 15 | Upper bound on retained items |
| `variance_threshold` | 2.0 | Std devs for anomaly detection (lower = more preserved) |
| `first_fraction` | 0.3 | Fraction of K allocated to array start |
| `last_fraction` | 0.15 | Fraction of K allocated to array end |
| `protect_analysis_context` | True | Protect code when user asks about it |
| `protect_recent_code` | 4 | Messages from end to protect code |
| `skip_user_messages` | True | Never compress user messages |
| `toin_confidence_threshold` | 0.3 | Minimum TOIN confidence to apply hints |
