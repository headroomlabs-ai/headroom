"""Tests for the in-memory request logger.

Covers the `log_full_messages` gate, which controls whether the
pre-compression (`request_messages`) and post-compression
(`compressed_messages`) payloads persist past the in-memory entry onto disk.
Both sides are governed by the same flag so the two sides of the compression
stay in sync - it's pointless to store one without the other.
"""

from __future__ import annotations

from headroom.proxy.models import RequestLog
from headroom.proxy.request_logger import RequestLogger


def _entry(**overrides) -> RequestLog:
    base: dict = {
        "request_id": "r1",
        "timestamp": "2026-04-24T10:00:00Z",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "input_tokens_original": 100,
        "input_tokens_optimized": 40,
        "output_tokens": 10,
        "tokens_saved": 60,
        "savings_percent": 60.0,
        "optimization_latency_ms": 1.0,
        "total_latency_ms": 20.0,
        "tags": {},
        "cache_hit": False,
        "transforms_applied": ["kompress:user:0.4"],
    }
    base.update(overrides)
    return RequestLog(**base)


def test_get_recent_strips_compressed_messages_alongside_request_and_response():
    logger = RequestLogger(log_file=None, log_full_messages=True)
    logger.log(
        _entry(
            request_messages=[{"role": "user", "content": "pre"}],
            compressed_messages=[{"role": "user", "content": "post"}],
            response_content="ok",
        )
    )

    recent = logger.get_recent(10)
    assert len(recent) == 1
    assert "request_messages" not in recent[0]
    assert "compressed_messages" not in recent[0]
    assert "response_content" not in recent[0]


def test_get_recent_with_messages_returns_compressed_messages():
    logger = RequestLogger(log_file=None, log_full_messages=True)
    logger.log(
        _entry(
            request_messages=[{"role": "user", "content": "pre"}],
            compressed_messages=[{"role": "user", "content": "post"}],
        )
    )

    recent = logger.get_recent_with_messages(10)
    assert len(recent) == 1
    assert recent[0]["request_messages"] == [{"role": "user", "content": "pre"}]
    assert recent[0]["compressed_messages"] == [{"role": "user", "content": "post"}]


def test_jsonl_file_strips_both_sides_when_log_full_messages_disabled(tmp_path):
    log_file = tmp_path / "requests.jsonl"
    logger = RequestLogger(log_file=str(log_file), log_full_messages=False)
    logger.log(
        _entry(
            request_messages=[{"role": "user", "content": "pre"}],
            compressed_messages=[{"role": "user", "content": "post"}],
            response_content="ok",
        )
    )

    import json

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert "request_messages" not in obj
    assert "compressed_messages" not in obj
    assert "response_content" not in obj


def test_get_memory_stats_accounts_for_compressed_messages():
    logger = RequestLogger(log_file=None)
    logger.log(
        _entry(
            compressed_messages=[{"role": "user", "content": "post"}],
        )
    )

    stats = logger.get_memory_stats()
    assert stats.entry_count == 1
    assert stats.size_bytes > 0


def test_get_recent_never_walks_message_payloads():
    """`/stats` calls get_recent(10_000) on the event loop; the payloads it
    drops must not be traversed first (asdict deep-copied them, ~30 s per call
    on a full deque). A leaf that refuses to be deep-copied proves the walk
    is gone."""

    class _NoCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("get_recent walked a message payload")

    logger = RequestLogger(log_file=None, log_full_messages=True)
    logger.log(
        _entry(
            request_messages=[{"role": "user", "content": _NoCopy()}],
            compressed_messages=[{"role": "user", "content": _NoCopy()}],
            tags={"agent": "codex", "meta": {"depth": 1}},
            transforms_applied=["smart_crusher"],
            savings_breakdown=[{"tokens": 60}],
        )
    )

    recent = logger.get_recent(10)
    assert recent[0]["tags"] == {"agent": "codex", "meta": {"depth": 1}}
    assert recent[0]["transforms_applied"] == ["smart_crusher"]
    # Still copies, not aliases, of the entry's own containers, all the way
    # down: mutating a nested value must not change the next /stats result.
    recent[0]["tags"]["agent"] = "x"
    recent[0]["tags"]["meta"]["depth"] = 99
    recent[0]["savings_breakdown"][0]["tokens"] = 0
    again = logger.get_recent(10)[0]
    assert again["tags"] == {"agent": "codex", "meta": {"depth": 1}}
    assert again["savings_breakdown"] == [{"tokens": 60}]


def test_get_recent_with_messages_can_skip_payloads_without_walking_them():
    """A feed poller that only reads the numbers passes include_messages=False;
    the three body fields are absent and never traversed (asdict deep-copied
    them). A leaf that refuses to be deep-copied proves the walk is gone."""

    class _NoCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("get_recent_with_messages walked a message payload")

    logger = RequestLogger(log_file=None, log_full_messages=True)
    logger.log(
        _entry(
            request_messages=[{"role": "user", "content": _NoCopy()}],
            compressed_messages=[{"role": "user", "content": _NoCopy()}],
            response_content="ok",
        )
    )

    slim = logger.get_recent_with_messages(10, include_messages=False)
    assert slim[0]["tokens_saved"] == 60
    assert slim[0]["transforms_applied"] == ["kompress:user:0.4"]
    assert not {"request_messages", "compressed_messages", "response_content"} & slim[0].keys()
