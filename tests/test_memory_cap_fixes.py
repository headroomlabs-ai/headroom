"""Tests for memory-cap fixes.

Covers:
- RequestLogger: in-memory entry cap enforcement and body truncation
- TOIN: LRU pattern eviction at max_patterns cap
- TOIN: _all_seen_instances cap (MAX_SEEN_INSTANCES = 1_000)
"""

from __future__ import annotations

from pathlib import Path

from headroom.proxy.models import RequestLog
from headroom.proxy.request_logger import RequestLogger
from headroom.telemetry import ToolSignature
from headroom.telemetry.toin import TOINConfig, ToolIntelligenceNetwork, reset_toin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_log(
    request_id: str = "req-1",
    *,
    response_content: str | None = None,
) -> RequestLog:
    """Build a minimal RequestLog for testing."""
    return RequestLog(
        request_id=request_id,
        timestamp="2026-01-01T00:00:00Z",
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_tokens_original=100,
        input_tokens_optimized=80,
        output_tokens=20,
        tokens_saved=20,
        savings_percent=20.0,
        optimization_latency_ms=1.0,
        total_latency_ms=10.0,
        tags={},
        cache_hit=False,
        transforms_applied=[],
        request_messages=None,
        response_content=response_content,
    )


def _make_toin(max_patterns: int = 10, max_seen: int = 5) -> ToolIntelligenceNetwork:
    """Create a TOIN instance with small caps for testing."""
    reset_toin()
    cfg = TOINConfig(
        storage_path="",
        auto_save_interval=0,
        max_patterns=max_patterns,
    )
    toin = ToolIntelligenceNetwork(cfg)
    # Override the per-pattern cap via class attr (affects all instances in test)
    toin.__class__  # noqa: B018 — reference to avoid pyright unused-warning
    from headroom.telemetry.toin import ToolPattern

    ToolPattern.MAX_SEEN_INSTANCES = max_seen
    return toin


def _make_signature(seed: int) -> ToolSignature:
    """Build a unique ToolSignature for each integer seed.

    Uses distinct field names per seed so each signature has a different
    ``structure_hash`` (the hash keys on field names, not values).
    """
    # Field name varies with seed -> different structure hash per seed
    items = [{f"field_{seed}": i, f"tag_{seed}": f"v{i}"} for i in range(3)]
    return ToolSignature.from_items(items)


def _compress(toin: ToolIntelligenceNetwork, sig: ToolSignature) -> None:
    toin.record_compression(
        tool_signature=sig,
        original_count=10,
        compressed_count=5,
        original_tokens=1000,
        compressed_tokens=500,
        strategy="smart_crusher",
    )


# ---------------------------------------------------------------------------
# RequestLogger — entry cap
# ---------------------------------------------------------------------------


class TestRequestLoggerCap:
    """RequestLogger respects the configured max_entries cap."""

    def test_default_cap_is_500(self) -> None:
        logger = RequestLogger(log_file=None, log_full_messages=False)
        assert logger._max_entries == 500

    def test_custom_cap_is_respected(self) -> None:
        logger = RequestLogger(log_file=None, log_full_messages=False, max_entries=10)
        assert logger._max_entries == 10

    def test_entries_capped_fifo_eviction(self) -> None:
        logger = RequestLogger(log_file=None, log_full_messages=False, max_entries=5)
        for i in range(8):
            logger.log(_make_log(f"req-{i}"))
        # Only 5 most-recent entries remain
        assert len(logger._logs) == 5
        ids = [e.request_id for e in logger._logs]
        assert ids == ["req-3", "req-4", "req-5", "req-6", "req-7"]

    def test_entries_not_capped_when_below_limit(self) -> None:
        logger = RequestLogger(log_file=None, log_full_messages=False, max_entries=100)
        for i in range(20):
            logger.log(_make_log(f"req-{i}"))
        assert len(logger._logs) == 20


# ---------------------------------------------------------------------------
# RequestLogger — body truncation
# ---------------------------------------------------------------------------


class TestRequestLoggerBodyTruncation:
    """Bodies stored in deque are truncated to MAX_BODY_BYTES."""

    def test_short_body_not_truncated(self) -> None:
        logger = RequestLogger(log_file=None, log_full_messages=True)
        short = "hello world"
        logger.log(_make_log(response_content=short))
        assert logger._logs[-1].response_content == short

    def test_long_body_truncated_in_memory(self) -> None:
        logger = RequestLogger(log_file=None, log_full_messages=True)
        big = "x" * (RequestLogger.MAX_BODY_BYTES * 3)
        logger.log(_make_log(response_content=big))
        stored = logger._logs[-1].response_content
        assert stored is not None
        assert len(stored.encode("utf-8")) <= RequestLogger.MAX_BODY_BYTES + len(" [truncated]")
        assert stored.endswith(" [truncated]")

    def test_truncated_marker_present(self) -> None:
        logger = RequestLogger(log_file=None, log_full_messages=True)
        big = "a" * 5000
        logger.log(_make_log(response_content=big))
        assert " [truncated]" in (logger._logs[-1].response_content or "")

    def test_disk_log_retains_full_body(self, tmp_path: Path) -> None:
        """On-disk JSONL receives the full body; only deque is truncated."""
        import json

        log_file = str(tmp_path / "test.jsonl")
        logger = RequestLogger(log_file=log_file, log_full_messages=True)
        full_body = "z" * 5000
        logger.log(_make_log(response_content=full_body))

        # In-memory: truncated
        assert logger._logs[-1].response_content is not None
        assert len(logger._logs[-1].response_content) < len(full_body)

        # On-disk: full body preserved
        line = Path(log_file).read_text().strip()
        row = json.loads(line)
        assert row["response_content"] == full_body

    def test_none_response_content_not_modified(self) -> None:
        logger = RequestLogger(log_file=None, log_full_messages=True)
        logger.log(_make_log(response_content=None))
        assert logger._logs[-1].response_content is None

    def test_truncate_body_helper_exact_boundary(self) -> None:
        exact = "a" * RequestLogger.MAX_BODY_BYTES
        result = RequestLogger._truncate_body(exact, RequestLogger.MAX_BODY_BYTES)
        assert result == exact  # exactly at limit — not truncated

    def test_truncate_body_helper_one_over(self) -> None:
        over = "b" * (RequestLogger.MAX_BODY_BYTES + 1)
        result = RequestLogger._truncate_body(over, RequestLogger.MAX_BODY_BYTES)
        assert result.endswith(" [truncated]")
        assert len(result.encode("utf-8")) <= RequestLogger.MAX_BODY_BYTES + len(" [truncated]")


# ---------------------------------------------------------------------------
# TOIN — LRU pattern eviction
# ---------------------------------------------------------------------------


class TestTOINPatternEviction:
    """_patterns dict is bounded to max_patterns with LRU eviction."""

    def setup_method(self) -> None:
        reset_toin()

    def teardown_method(self) -> None:
        reset_toin()
        # Restore class-level constant (tests may lower it)
        from headroom.telemetry.toin import ToolPattern

        ToolPattern.MAX_SEEN_INSTANCES = 1_000

    def test_patterns_capped_at_max(self) -> None:
        toin = _make_toin(max_patterns=3)
        for i in range(5):
            _compress(toin, _make_signature(i))
        assert len(toin._patterns) <= 3

    def test_lru_pattern_evicted_first(self) -> None:
        """The pattern that hasn't been accessed longest is evicted."""
        toin = _make_toin(max_patterns=2)
        sig0 = _make_signature(0)
        sig1 = _make_signature(1)
        sig2 = _make_signature(2)

        _compress(toin, sig0)  # access sig0 (MRU: [0])
        _compress(toin, sig1)  # access sig1 (MRU: [0, 1])
        # Touch sig0 again so sig1 becomes LRU
        _compress(toin, sig0)  # (MRU: [1, 0])
        # Adding sig2 should evict sig1 (LRU)
        _compress(toin, sig2)

        keys = {k[2] for k in toin._patterns}
        assert sig0.structure_hash in keys, "sig0 should survive (recently accessed)"
        assert sig2.structure_hash in keys, "sig2 should survive (just added)"
        assert sig1.structure_hash not in keys, "sig1 should be evicted (LRU)"

    def test_no_eviction_when_below_cap(self) -> None:
        toin = _make_toin(max_patterns=20)
        sigs = [_make_signature(i) for i in range(10)]
        for s in sigs:
            _compress(toin, s)
        assert len(toin._patterns) == 10

    def test_get_pattern_promotes_to_mru(self) -> None:
        """Reading a pattern via get_pattern should prevent its eviction."""
        toin = _make_toin(max_patterns=2)
        sig0 = _make_signature(0)
        sig1 = _make_signature(1)
        sig2 = _make_signature(2)

        _compress(toin, sig0)
        _compress(toin, sig1)
        # Read sig0 — should promote it to MRU
        toin.get_pattern(sig0.structure_hash)
        # Now sig1 is LRU; adding sig2 should evict sig1
        _compress(toin, sig2)

        keys = {k[2] for k in toin._patterns}
        assert sig0.structure_hash in keys, "sig0 promoted by get_pattern should survive"
        assert sig1.structure_hash not in keys, "sig1 not accessed — should be evicted"


# ---------------------------------------------------------------------------
# TOIN — _all_seen_instances cap
# ---------------------------------------------------------------------------


class TestTOINInstanceCap:
    """_all_seen_instances is capped at MAX_SEEN_INSTANCES."""

    def setup_method(self) -> None:
        reset_toin()

    def teardown_method(self) -> None:
        reset_toin()
        from headroom.telemetry.toin import ToolPattern

        ToolPattern.MAX_SEEN_INSTANCES = 1_000

    def test_default_max_seen_instances_is_1000(self) -> None:
        from headroom.telemetry.toin import ToolPattern

        assert ToolPattern.MAX_SEEN_INSTANCES == 1_000

    def test_all_seen_instances_capped(self) -> None:
        """_all_seen_instances never exceeds MAX_SEEN_INSTANCES."""
        from headroom.telemetry.toin import ToolPattern

        ToolPattern.MAX_SEEN_INSTANCES = 10  # tiny cap for test speed
        toin = ToolIntelligenceNetwork(TOINConfig(storage_path="", auto_save_interval=0))
        sig = _make_signature(99)

        for i in range(20):
            # Simulate different instance IDs by monkey-patching
            toin._instance_id = f"inst_{i:04x}_deadbeef"
            _compress(toin, sig)

        key = ("unknown", "unknown", sig.structure_hash)
        pattern = toin._patterns[key]
        assert len(pattern._all_seen_instances) <= 10

    def test_user_count_still_increments_beyond_cap(self) -> None:
        """user_count keeps counting even when the instance set is full."""
        from headroom.telemetry.toin import ToolPattern

        ToolPattern.MAX_SEEN_INSTANCES = 5
        toin = ToolIntelligenceNetwork(TOINConfig(storage_path="", auto_save_interval=0))
        sig = _make_signature(100)

        seen_before = 0
        for i in range(15):
            toin._instance_id = f"user_{i:04x}_deadbeef"
            _compress(toin, sig)

        key = ("unknown", "unknown", sig.structure_hash)
        pattern = toin._patterns[key]
        # user_count should equal number of unique instance IDs we injected
        assert pattern.user_count == 15
        assert len(pattern._all_seen_instances) == 5  # capped
        del seen_before  # suppress "assigned but unused" lint
