"""Tests for the config module.

Tests all configuration dataclasses, enums, and utility classes:
- HeadroomMode enum
- CacheAlignerConfig
- RelevanceScorerConfig, SmartCrusherConfig
- HeadroomConfig (main config)
- Block, WasteSignals, CachePrefixMetrics
- TransformResult, RequestMetrics
"""

from dataclasses import fields
from datetime import datetime

from headroom.config import (
    Block,
    CacheAlignerConfig,
    CachePrefixMetrics,
    HeadroomConfig,
    HeadroomMode,
    RelevanceScorerConfig,
    RequestMetrics,
    SmartCrusherConfig,
    TransformResult,
    WasteSignals,
)


class TestHeadroomMode:
    """Tests for HeadroomMode enum."""

    def test_enum_values(self):
        """All expected enum values exist with correct string values."""
        assert HeadroomMode.AUDIT.value == "audit"
        assert HeadroomMode.OPTIMIZE.value == "optimize"
        assert HeadroomMode.SIMULATE.value == "simulate"

    def test_string_conversion(self):
        """HeadroomMode inherits from str for string compatibility."""
        # Enum value access works as string
        assert HeadroomMode.AUDIT.value == "audit"
        assert HeadroomMode.OPTIMIZE.value == "optimize"
        assert HeadroomMode.SIMULATE.value == "simulate"
        # Can compare directly with strings since it inherits from str
        assert HeadroomMode.AUDIT == "audit"
        assert HeadroomMode.OPTIMIZE == "optimize"
        assert HeadroomMode.SIMULATE == "simulate"
        # isinstance check confirms str inheritance
        assert isinstance(HeadroomMode.AUDIT, str)


class TestCacheAlignerConfig:
    """Tests for CacheAlignerConfig dataclass."""

    def test_default_values(self):
        """Default values are correctly set."""
        config = CacheAlignerConfig()
        assert config.enabled is False
        assert config.normalize_whitespace is True
        assert config.collapse_blank_lines is True

    def test_date_patterns_default(self):
        """Default date_patterns contains expected regex patterns."""
        config = CacheAlignerConfig()
        assert isinstance(config.date_patterns, list)
        assert len(config.date_patterns) == 4
        # Verify specific patterns exist
        assert r"Current [Dd]ate:?\s*\d{4}-\d{2}-\d{2}" in config.date_patterns
        assert r"Today is \w+,?\s+\w+ \d+" in config.date_patterns
        assert r"Today's date:?\s*\d{4}-\d{2}-\d{2}" in config.date_patterns
        assert r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}" in config.date_patterns

    def test_dynamic_tail_separator_default(self):
        """Default dynamic_tail_separator has expected value."""
        config = CacheAlignerConfig()
        assert config.dynamic_tail_separator == "\n\n---\n[Dynamic Context]\n"

    def test_date_patterns_isolation(self):
        """Each instance gets its own date_patterns list."""
        config1 = CacheAlignerConfig()
        config2 = CacheAlignerConfig()
        config1.date_patterns.append(r"custom pattern")
        assert r"custom pattern" not in config2.date_patterns


class TestRelevanceScorerConfig:
    """Tests for RelevanceScorerConfig dataclass."""

    def test_default_tier_hybrid(self):
        """Default tier is hybrid."""
        config = RelevanceScorerConfig()
        assert config.tier == "hybrid"

    def test_bm25_params(self):
        """BM25 parameters have expected defaults."""
        config = RelevanceScorerConfig()
        assert config.bm25_k1 == 1.5
        assert config.bm25_b == 0.75

    def test_embedding_params(self):
        """Embedding parameters have expected defaults."""
        config = RelevanceScorerConfig()
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.hybrid_alpha == 0.5
        assert config.adaptive_alpha is True

    def test_relevance_threshold_default(self):
        """Relevance threshold defaults to 0.25."""
        config = RelevanceScorerConfig()
        assert config.relevance_threshold == 0.25


class TestSmartCrusherConfig:
    """Tests for SmartCrusherConfig dataclass."""

    def test_default_values(self):
        """Default values are correctly set."""
        config = SmartCrusherConfig()
        assert config.min_items_to_analyze == 5
        assert config.min_tokens_to_crush == 200
        assert config.variance_threshold == 2.0
        assert config.uniqueness_threshold == 0.1
        assert config.similarity_threshold == 0.8
        assert config.max_items_after_crush == 15
        assert config.preserve_change_points is True
        assert config.factor_out_constants is False
        assert config.include_summaries is False

    def test_enabled_by_default(self):
        """SmartCrusher is enabled by default."""
        config = SmartCrusherConfig()
        assert config.enabled is True

    def test_relevance_field_default(self):
        """Relevance field defaults to RelevanceScorerConfig instance."""
        config = SmartCrusherConfig()
        assert isinstance(config.relevance, RelevanceScorerConfig)
        assert config.relevance.tier == "hybrid"

    def test_relevance_isolation(self):
        """Each instance gets its own RelevanceScorerConfig."""
        config1 = SmartCrusherConfig()
        config2 = SmartCrusherConfig()
        config1.relevance.tier = "bm25"
        assert config2.relevance.tier == "hybrid"


class TestHeadroomConfig:
    """Tests for HeadroomConfig main configuration class."""

    def test_default_values(self):
        """Default values are correctly set."""
        config = HeadroomConfig()
        assert config.store_url == "sqlite:///headroom.db"
        assert config.default_mode == HeadroomMode.AUDIT
        assert config.generate_diff_artifact is False
        # Nested configs exist
        assert isinstance(config.smart_crusher, SmartCrusherConfig)
        assert isinstance(config.cache_aligner, CacheAlignerConfig)

    def test_get_context_limit_direct_match(self):
        """get_context_limit returns limit for exact model match."""
        config = HeadroomConfig(model_context_limits={"gpt-4o": 128000, "claude-3-opus": 200000})
        assert config.get_context_limit("gpt-4o") == 128000
        assert config.get_context_limit("claude-3-opus") == 200000

    def test_get_context_limit_prefix_match(self):
        """get_context_limit returns limit for prefix match."""
        config = HeadroomConfig(model_context_limits={"gpt-4": 128000, "claude-3": 200000})
        # Prefix matches
        assert config.get_context_limit("gpt-4-turbo") == 128000
        assert config.get_context_limit("gpt-4o") == 128000
        assert config.get_context_limit("claude-3-opus") == 200000
        assert config.get_context_limit("claude-3-sonnet") == 200000

    def test_get_context_limit_not_found(self):
        """get_context_limit returns None for unknown model."""
        config = HeadroomConfig(model_context_limits={"gpt-4": 128000})
        assert config.get_context_limit("unknown-model") is None
        assert config.get_context_limit("llama-2") is None

    def test_model_context_limits_isolation(self):
        """Each instance gets its own model_context_limits dict."""
        config1 = HeadroomConfig()
        config2 = HeadroomConfig()
        config1.model_context_limits["custom-model"] = 50000
        assert "custom-model" not in config2.model_context_limits


class TestBlock:
    """Tests for Block dataclass."""

    def test_block_creation(self):
        """Block can be created with required fields."""
        block = Block(
            kind="user",
            text="Hello, world!",
            tokens_est=5,
            content_hash="abc123",
            source_index=0,
        )
        assert block.kind == "user"
        assert block.text == "Hello, world!"
        assert block.tokens_est == 5
        assert block.content_hash == "abc123"
        assert block.source_index == 0
        assert block.flags == {}

    def test_block_kinds(self):
        """Block accepts all valid kind values."""
        valid_kinds = ["system", "user", "assistant", "tool_call", "tool_result", "rag", "unknown"]
        for kind in valid_kinds:
            block = Block(
                kind=kind,
                text="test",
                tokens_est=1,
                content_hash="hash",
                source_index=0,
            )
            assert block.kind == kind

    def test_block_flags_default_factory(self):
        """Each block gets its own flags dict."""
        block1 = Block(kind="user", text="a", tokens_est=1, content_hash="h1", source_index=0)
        block2 = Block(kind="user", text="b", tokens_est=1, content_hash="h2", source_index=1)
        block1.flags["custom"] = True
        assert "custom" not in block2.flags


class TestWasteSignals:
    """Tests for WasteSignals dataclass."""

    def test_total_calculation(self):
        """total() correctly sums all waste token fields."""
        signals = WasteSignals(
            json_bloat_tokens=100,
            html_noise_tokens=50,
            base64_tokens=200,
            whitespace_tokens=25,
            dynamic_date_tokens=10,
            repetition_tokens=15,
        )
        assert signals.total() == 400

    def test_total_with_defaults(self):
        """total() returns 0 when all fields are default."""
        signals = WasteSignals()
        assert signals.total() == 0

    def test_to_dict(self):
        """to_dict() returns correct dictionary representation."""
        signals = WasteSignals(
            json_bloat_tokens=100,
            html_noise_tokens=50,
            base64_tokens=200,
            whitespace_tokens=25,
            dynamic_date_tokens=10,
            repetition_tokens=15,
        )
        expected = {
            "json_bloat": 100,
            "html_noise": 50,
            "base64": 200,
            "whitespace": 25,
            "dynamic_date": 10,
            "repetition": 15,
        }
        assert signals.to_dict() == expected

    def test_to_dict_defaults(self):
        """to_dict() returns zeroes for default values."""
        signals = WasteSignals()
        result = signals.to_dict()
        assert all(v == 0 for v in result.values())
        assert len(result) == 6


class TestCachePrefixMetrics:
    """Tests for CachePrefixMetrics dataclass."""

    def test_dataclass_fields(self):
        """CachePrefixMetrics has all expected fields."""
        field_names = {f.name for f in fields(CachePrefixMetrics)}
        expected_fields = {
            "stable_prefix_bytes",
            "stable_prefix_tokens_est",
            "stable_prefix_hash",
            "prefix_changed",
            "previous_hash",
        }
        assert field_names == expected_fields

    def test_creation(self):
        """CachePrefixMetrics can be created with required fields."""
        metrics = CachePrefixMetrics(
            stable_prefix_bytes=1024,
            stable_prefix_tokens_est=256,
            stable_prefix_hash="abc123def456",
            prefix_changed=False,
        )
        assert metrics.stable_prefix_bytes == 1024
        assert metrics.stable_prefix_tokens_est == 256
        assert metrics.stable_prefix_hash == "abc123def456"
        assert metrics.prefix_changed is False
        assert metrics.previous_hash is None

    def test_previous_hash_optional(self):
        """previous_hash defaults to None."""
        metrics = CachePrefixMetrics(
            stable_prefix_bytes=512,
            stable_prefix_tokens_est=128,
            stable_prefix_hash="hash123",
            prefix_changed=True,
            previous_hash="oldhash",
        )
        assert metrics.previous_hash == "oldhash"


class TestTransformResult:
    """Tests for TransformResult dataclass."""

    def test_dataclass_fields(self):
        """TransformResult has all expected fields."""
        field_names = {f.name for f in fields(TransformResult)}
        expected_fields = {
            "messages",
            "tokens_before",
            "tokens_after",
            "transforms_applied",
            "markers_inserted",
            "warnings",
            "diff_artifact",
            "cache_metrics",
            "timing",
            "waste_signals",
        }
        assert field_names == expected_fields

    def test_default_empty_lists(self):
        """Default factory produces empty lists for optional fields."""
        result = TransformResult(
            messages=[{"role": "user", "content": "test"}],
            tokens_before=100,
            tokens_after=80,
            transforms_applied=["CacheAligner"],
        )
        assert result.markers_inserted == []
        assert result.warnings == []
        assert result.diff_artifact is None
        assert result.cache_metrics is None

    def test_list_isolation(self):
        """Each instance gets its own lists."""
        result1 = TransformResult(
            messages=[],
            tokens_before=100,
            tokens_after=80,
            transforms_applied=["Transform1"],
        )
        result2 = TransformResult(
            messages=[],
            tokens_before=100,
            tokens_after=80,
            transforms_applied=["Transform2"],
        )
        result1.markers_inserted.append("marker")
        result1.warnings.append("warning")
        assert result2.markers_inserted == []
        assert result2.warnings == []


class TestRequestMetrics:
    """Tests for RequestMetrics dataclass."""

    def test_dataclass_fields(self):
        """RequestMetrics has all expected fields."""
        field_names = {f.name for f in fields(RequestMetrics)}
        expected_fields = {
            "request_id",
            "timestamp",
            "model",
            "stream",
            "mode",
            "tokens_input_before",
            "tokens_input_after",
            "tokens_output",
            "block_breakdown",
            "waste_signals",
            "stable_prefix_hash",
            "cache_alignment_score",
            "cached_tokens",
            # Cache optimizer metrics (provider-specific)
            "cache_optimizer_used",
            "cache_optimizer_strategy",
            "cacheable_tokens",
            "breakpoints_inserted",
            "estimated_cache_hit",
            "estimated_savings_percent",
            "semantic_cache_hit",
            # Transform details
            "transforms_applied",
            "tool_units_dropped",
            "turns_dropped",
            "messages_hash",
            "error",
        }
        assert field_names == expected_fields

    def test_default_values(self):
        """Default values are correctly set for optional fields."""
        metrics = RequestMetrics(
            request_id="test-123",
            timestamp=datetime(2025, 1, 6),
            model="gpt-4o",
            stream=False,
            mode="audit",
            tokens_input_before=1000,
            tokens_input_after=800,
        )
        assert metrics.tokens_output is None
        assert metrics.block_breakdown == {}
        assert metrics.waste_signals == {}
        assert metrics.stable_prefix_hash == ""
        assert metrics.cache_alignment_score == 0.0
        assert metrics.cached_tokens is None
        assert metrics.transforms_applied == []
        assert metrics.tool_units_dropped == 0
        assert metrics.turns_dropped == 0
        assert metrics.messages_hash == ""
        assert metrics.error is None

    def test_full_creation(self):
        """RequestMetrics can be created with all fields."""
        metrics = RequestMetrics(
            request_id="req-456",
            timestamp=datetime(2025, 1, 6, 12, 30),
            model="claude-3-opus",
            stream=True,
            mode="optimize",
            tokens_input_before=2000,
            tokens_input_after=1500,
            tokens_output=500,
            block_breakdown={"system": 200, "user": 800},
            waste_signals={"json_bloat": 100},
            stable_prefix_hash="hash123",
            cache_alignment_score=95.5,
            cached_tokens=200,
            transforms_applied=["CacheAligner", "SmartCrusher"],
            tool_units_dropped=2,
            turns_dropped=1,
            messages_hash="msghash",
            error=None,
        )
        assert metrics.request_id == "req-456"
        assert metrics.model == "claude-3-opus"
        assert metrics.stream is True
        assert metrics.tokens_output == 500
        assert metrics.cache_alignment_score == 95.5

    def test_dict_isolation(self):
        """Each instance gets its own dicts and lists."""
        metrics1 = RequestMetrics(
            request_id="1",
            timestamp=datetime.now(),
            model="m",
            stream=False,
            mode="audit",
            tokens_input_before=100,
            tokens_input_after=100,
        )
        metrics2 = RequestMetrics(
            request_id="2",
            timestamp=datetime.now(),
            model="m",
            stream=False,
            mode="audit",
            tokens_input_before=100,
            tokens_input_after=100,
        )
        metrics1.block_breakdown["system"] = 50
        metrics1.waste_signals["json_bloat"] = 25
        metrics1.transforms_applied.append("Test")
        assert metrics2.block_breakdown == {}
        assert metrics2.waste_signals == {}
        assert metrics2.transforms_applied == []
