# API Reference

## HeadroomClient

The main entry point for Headroom SDK.

```python
from headroom import HeadroomClient
from openai import OpenAI

client = HeadroomClient(
    original_client=OpenAI(),
    default_mode="optimize",
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `original_client` | `OpenAI \| Anthropic` | Required | The underlying LLM client |
| `provider` | `Provider` | Auto-detected | Token counting provider |
| `default_mode` | `str` | `"audit"` | Default mode: "audit", "optimize", "off" |
| `store_url` | `str` | `None` | Storage URL for metrics |
| `smart_crusher_config` | `SmartCrusherConfig` | Default | Compression settings |
| `cache_aligner_config` | `CacheAlignerConfig` | Default | Cache alignment settings |

### Methods

#### `chat.completions.create(**kwargs)`

Create a chat completion with optional optimization.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    headroom_mode="optimize",  # Override default mode
)
```

**Additional Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `headroom_mode` | `str` | Override mode for this request |
| `headroom_query` | `str` | Query for relevance scoring |

#### `chat.completions.simulate(**kwargs)`

Preview optimization without making an API call.

```python
plan = client.chat.completions.simulate(
    model="gpt-4o",
    messages=[...],
)

print(f"Tokens before: {plan.tokens_before}")
print(f"Tokens after: {plan.tokens_after}")
print(f"Savings: {plan.savings_percent:.1f}%")
```

**Returns:** `SimulationResult`

---

## Configuration Classes

### SmartCrusherConfig

```python
from headroom import SmartCrusherConfig

config = SmartCrusherConfig(
    min_tokens_to_crush=200,
    max_items_after_crush=50,
    keep_first=3,
    keep_last=2,
    relevance_threshold=0.3,
    anomaly_std_threshold=2.0,
    preserve_errors=True,
)
```

### CacheAlignerConfig

```python
from headroom import CacheAlignerConfig

config = CacheAlignerConfig(
    extract_dates=True,
    normalize_whitespace=True,
    stable_prefix_min_tokens=100,
)
```

### RelevanceScorerConfig

```python
from headroom import RelevanceScorerConfig

config = RelevanceScorerConfig(
    scorer_type="bm25",      # "bm25", "embedding", or "hybrid"
    embedding_model=None,    # Model name for embedding scorer
    hybrid_alpha=0.5,        # Weight for hybrid scoring
)
```

---

## Data Models

### SimulationResult

Returned by `simulate()`.

```python
@dataclass
class SimulationResult:
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float
    transforms_applied: list[str]
    waste_signals: WasteSignals
```

### RequestMetrics

Metrics for a single request.

```python
@dataclass
class RequestMetrics:
    request_id: str
    timestamp: datetime
    model: str
    tokens_input_before: int
    tokens_input_after: int
    tokens_output: int
    cost_before: float
    cost_after: float
    transforms_applied: list[str]
```

### WasteSignals

Detected waste in the request.

```python
@dataclass
class WasteSignals:
    json_bloat_tokens: int
    html_noise_tokens: int
    whitespace_tokens: int
    dynamic_date_tokens: int
    repetition_tokens: int
```

---

## Providers

### OpenAIProvider

```python
from headroom import OpenAIProvider

provider = OpenAIProvider()

# Get token counter
counter = provider.get_token_counter("gpt-4o")
tokens = counter.count_text("Hello, world!")

# Get context limit
limit = provider.get_context_limit("gpt-4o")  # 128000

# Estimate cost
cost = provider.estimate_cost(
    input_tokens=1000,
    output_tokens=500,
    model="gpt-4o",
)
```

### AnthropicProvider

```python
from headroom import AnthropicProvider
from anthropic import Anthropic

provider = AnthropicProvider(client=Anthropic())

counter = provider.get_token_counter("claude-3-5-sonnet-latest")
tokens = counter.count_messages(messages)  # Accurate count via API
```

---

## Relevance Scoring

### BM25Scorer

Fast keyword-based scoring (zero dependencies).

```python
from headroom import BM25Scorer

scorer = BM25Scorer()
scores = scorer.score_items(
    items=["item 1", "item 2", ...],
    query="search query",
)
```

### EmbeddingScorer

Semantic similarity scoring (requires `sentence-transformers`).

```python
from headroom import EmbeddingScorer, embedding_available

if embedding_available():
    scorer = EmbeddingScorer(model="all-MiniLM-L6-v2")
    scores = scorer.score_items(items, query)
```

### HybridScorer

Combines BM25 and embeddings.

```python
from headroom import HybridScorer

scorer = HybridScorer(alpha=0.5)  # 50% BM25, 50% embedding
scores = scorer.score_items(items, query)
```

### create_scorer()

Factory function to create scorers.

```python
from headroom import create_scorer

# Auto-select best available scorer
scorer = create_scorer()

# Explicitly choose type
scorer = create_scorer(scorer_type="hybrid", alpha=0.7)
```

---

## Transforms (Direct Use)

### SmartCrusher

```python
from headroom import SmartCrusher

crusher = SmartCrusher()
result = crusher.crush(
    data={"results": [...]},
    query="user query",
)
```

### CacheAligner

```python
from headroom import CacheAligner

aligner = CacheAligner()
result = aligner.align(messages)
```

> **Context management** is handled automatically inside the pipeline
> (live-zone-only compression). The position-based `RollingWindow` and
> score-based `IntelligentContextManager` / `MessageScorer` APIs have been
> removed and are no longer part of Headroom.

### TransformPipeline

```python
from headroom import TransformPipeline

pipeline = TransformPipeline([
    SmartCrusher(),
    CacheAligner(),
])

result = pipeline.transform(messages)
```

---

## Utilities

### Tokenizer

```python
from headroom import Tokenizer, count_tokens_text, count_tokens_messages

# Quick counting
tokens = count_tokens_text("Hello, world!", model="gpt-4o")

# With tokenizer instance
tokenizer = Tokenizer(model="gpt-4o")
tokens = tokenizer.count_text("Hello")
tokens = tokenizer.count_messages(messages)
```

### generate_report()

Generate HTML/Markdown reports from stored metrics.

```python
from headroom import generate_report

report = generate_report(
    store_url="sqlite:///headroom.db",
    format="html",
    period="day",
)
```

---

## TypeScript SDK

For the TypeScript SDK API reference, see [TypeScript SDK](typescript-sdk.md).

The TypeScript SDK provides `compress()`, `HeadroomClient`, and framework adapters for Vercel AI SDK, OpenAI, and Anthropic.
