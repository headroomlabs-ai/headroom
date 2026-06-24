# 008. Capabilities

**Status:** done

## Feature Matrix

| Capability | Proxy | SDK | Wrap | MCP | ASGI | LiteLLM |
|------------|:-----:|:---:|:----:|:---:|:----:|:-------:|
| Semantic compression | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Summary compression | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Token budget management | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Learn mode | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Session tracking | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CCR feedback | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TOIN tagging | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Savings tracking | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Dashboard | ✓ | - | ✓ | - | - | - |
| Health endpoints | ✓ | - | - | - | - | - |
| Metrics (Prometheus) | ✓ | - | - | - | - | - |

---

## Compression Capabilities

### Proxy Cache

**Description:** The proxy has semantic cache support and CCR-backed retrieval, controlled by CLI configuration.

**Configuration:**
```bash
headroom proxy          # cache enabled by default
headroom proxy --no-cache
```

**Behavior:**
1. Hash input content
2. Check cache for similar entries (configurable threshold)
3. Return cached response if match found (CCR retrieval)
4. Update cache on miss
5. Store compressed content for future retrieval

**CCR Configuration:**
```python
@dataclass
class CCRConfig:
    enabled: bool = True
    storage: ContentStorage | None = None
    cache_ttl: int = 3600
    similarity_threshold: float = 0.85
    max_results: int = 10
```

---

### Summary Compression

**Description:** Compresses long context using summarization.

**Configuration:**
```bash
# Note: HEADROOM_SUMMARY_* env vars are not yet implemented.
# Summary compression is currently configured programmatically only.
```

**Behavior:**
1. Count input tokens
2. If over threshold, apply summary compression
3. Preserve key information (configurable priority)
4. Track summary statistics

**Summary Pipeline:**
```python
class SummaryCompressor:
    def __init__(
        self,
        threshold: int = 5000,
        target_ratio: float = 0.3,
        priority_preservation: bool = True,
    ) -> None:
        pass
    
    def compress(
        self,
        content: str,
        context: TransformContext,
    ) -> TransformResult:
        pass
```

---

### Token Budget Manager

**Description:** Enforces token budget limits per session via RollingWindow.

**Configuration:**
```python
@dataclass
class RollingWindowConfig:
    enabled: bool = True
    keep_system: bool = True  # Never drop system prompt
    keep_last_turns: int = 2  # Never drop last N turns
    output_buffer_tokens: int = 4000  # Reserve for output
```

**Behavior:**
1. Track token usage per session
2. Enforce budget limits via rolling window
3. Preserve system prompt and last N turns
4. Reserve output buffer tokens

---

## Intelligent Context Management

**Description:** Semantic-aware context management with TOIN integration.

```python
@dataclass
class IntelligentContextConfig:
    enabled: bool = True
    use_importance_scoring: bool = True
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    toin_integration: bool = True
    toin_confidence_threshold: float = 0.3
```

**Scoring Weights:**
```python
@dataclass
class ScoringWeights:
    recency: float = 0.20
    semantic_similarity: float = 0.20
    toin_importance: float = 0.25
    error_indicator: float = 0.15
    forward_reference: float = 0.15
    token_density: float = 0.05
```

---

## CCR Configuration

**Description:** Compress-Cache-Retrieve makes compression reversible.

```python
@dataclass
class CCRConfig:
    enabled: bool = True
    store_max_entries: int = 1000
    store_ttl_seconds: int = 300
    inject_retrieval_marker: bool = True
    feedback_enabled: bool = True
    min_items_to_cache: int = 20
    inject_tool: bool = True
    inject_system_instructions: bool = False
```

---

## SmartCrusher (Statistical JSON Compression)

**Description:** Preserves JSON schema while reducing array size via statistical analysis.

```python
@dataclass
class SmartCrusherConfig:
    enabled: bool = True
    min_items_to_analyze: int = 5
    min_tokens_to_crush: int = 200
    variance_threshold: float = 2.0
    uniqueness_threshold: float = 0.1
    similarity_threshold: float = 0.8
    max_items_after_crush: int = 15
    preserve_change_points: bool = True
    use_feedback_hints: bool = True
    toin_confidence_threshold: float = 0.3
    relevance: RelevanceScorerConfig = field(default_factory=RelevanceScorerConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    dedup_identical_items: bool = True
    first_fraction: float = 0.3
    last_fraction: float = 0.15
```

**Relevance Scoring:**
```python
@dataclass
class RelevanceScorerConfig:
    tier: Literal["bm25", "embedding", "hybrid"] = "hybrid"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    embedding_model: str = field(default_factory=lambda: ML_MODEL_DEFAULTS.sentence_transformer)
    hybrid_alpha: float = 0.5
    adaptive_alpha: bool = True
    relevance_threshold: float = 0.25
```

**Anchor Allocation:**
```python
@dataclass
class AnchorConfig:
    anchor_budget_pct: float = 0.25
    min_anchor_slots: int = 3
    max_anchor_slots: int = 12
    default_front_weight: float = 0.5
    default_back_weight: float = 0.4
    default_middle_weight: float = 0.1
    search_front_weight: float = 0.75
    logs_back_weight: float = 0.75
```

---

## Agent-Specific Capabilities

### Claude

- Session branch comparison
- Token headroom mode
- Tool use tracking
- Multi-modal support (images)

### Codex

- Rate limit handling
- Code completion optimization
- Batch request support

### Gemini

- Multi-modal inputs
- Function calling support
- Context caching API

---

## Deployment Capabilities

| Feature | Docker | Native | Embedded |
|---------|:------:|:------:|:--------:|
| Standalone proxy | ✓ | ✓ | - |
| Health endpoints | ✓ | ✓ | - |
| Prometheus metrics | ✓ | ✓ | - |
| Dashboard | ✓ | ✓ | - |
| Volume mounts | ✓ | - | - |
| Environment overrides | ✓ | ✓ | ✓ |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial capabilities document |
