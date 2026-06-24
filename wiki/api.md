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
| `rolling_window_config` | `RollingWindowConfig` | Default | Context window settings |

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

### RollingWindowConfig

```python
from headroom import RollingWindowConfig

config = RollingWindowConfig(
    max_tokens=100000,
    preserve_system=True,
    preserve_recent_turns=5,
    drop_oldest_first=True,
)
```

### IntelligentContextConfig

```python
from headroom.config import IntelligentContextConfig, ScoringWeights

weights = ScoringWeights(
    recency=0.20,
    semantic_similarity=0.20,
    toin_importance=0.25,
    error_indicator=0.15,
    forward_reference=0.15,
    token_density=0.05,
)

config = IntelligentContextConfig(
    enabled=True,
    keep_system=True,
    keep_last_turns=2,
    output_buffer_tokens=4000,
    use_importance_scoring=True,
    scoring_weights=weights,
    toin_integration=True,
    recency_decay_rate=0.1,
    compress_threshold=0.1,
)
```

### ScoringWeights

```python
from headroom.config import ScoringWeights

weights = ScoringWeights(
    recency=0.20,              # Exponential decay from end
    semantic_similarity=0.20,  # Embedding similarity to recent context
    toin_importance=0.25,      # TOIN retrieval_rate
    error_indicator=0.15,      # TOIN field_semantics error detection
    forward_reference=0.15,    # Messages referenced by later messages
    token_density=0.05,        # Unique/total token ratio
)

# Weights are auto-normalized to sum to 1.0
normalized = weights.normalized()
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

### RollingWindow

```python
from headroom import RollingWindow

window = RollingWindow(config)
result = window.apply(messages, max_tokens=100000)
```

### IntelligentContextManager

```python
from headroom.transforms import IntelligentContextManager
from headroom.config import IntelligentContextConfig
from headroom.telemetry import get_toin

# With TOIN integration for learned patterns
toin = get_toin()
config = IntelligentContextConfig(
    keep_system=True,
    keep_last_turns=2,
    use_importance_scoring=True,
)

manager = IntelligentContextManager(config, toin=toin)
result = manager.apply(messages, tokenizer, model_limit=128000)

# Access scoring details
print(result.transforms_applied)  # ["intelligent_cap:3"]
print(result.tokens_before, result.tokens_after)
```

### MessageScorer

```python
from headroom.transforms import MessageScorer, MessageScore
from headroom.config import ScoringWeights

scorer = MessageScorer(
    weights=ScoringWeights(),
    toin=None,  # Optional TOIN for learned patterns
    embedding_provider=None,  # Optional for semantic similarity
    recency_decay_rate=0.1,
)

# Score messages
scores: list[MessageScore] = scorer.score_messages(
    messages=messages,
    protected_indices={0},  # System message
    tool_unit_indices={2, 3},  # Tool call + response
)

for score in scores:
    print(f"Message {score.message_index}: {score.total_score:.2f}")
    print(f"  Recency: {score.recency_score:.2f}")
    print(f"  TOIN: {score.toin_score:.2f}")
    print(f"  Protected: {score.is_protected}")
```

### TransformPipeline

```python
from headroom import TransformPipeline

pipeline = TransformPipeline([
    SmartCrusher(),
    CacheAligner(),
    RollingWindow(),
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
