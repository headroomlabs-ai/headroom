"""Coverage for the MCP-events aggregator inside the proxy /stats summary.

``headroom mcp serve`` writes every compress / retrieve invocation to a
shared file-locked log (``_append_shared_event``). Before this fix,
``/stats`` only reported proxy-HTTP-path compressions and silently
ignored the MCP tool work — which is exactly where Strands-style
agents spend most of their compression budget when the LLM calls
``headroom_compress`` directly.

These tests pin the aggregation logic in
:func:`headroom.proxy.cost._aggregate_mcp_events`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from headroom.proxy.cost import _aggregate_mcp_events


def _mock_events(events: list[dict[str, Any]]) -> Any:
    """Return a patch target that makes _read_shared_events yield ``events``."""
    return patch(
        "headroom.ccr.mcp_server._read_shared_events",
        return_value=events,
    )


def test_aggregates_compress_and_retrieve_events() -> None:
    events = [
        {"type": "compress", "input_tokens": 1000, "output_tokens": 400},
        {"type": "compress", "input_tokens": 500, "output_tokens": 300},
        {"type": "retrieve", "hash": "abc123"},
        {"type": "retrieve", "hash": "def456"},
        {"type": "retrieve", "hash": "ghi789"},
    ]
    with _mock_events(events):
        result = _aggregate_mcp_events()
    assert result == {
        "compressions": 2,
        # (1000-400) + (500-300) = 600 + 200 = 800
        "tokens_removed": 800,
        "retrievals": 3,
    }


def test_returns_zeros_when_no_events() -> None:
    with _mock_events([]):
        assert _aggregate_mcp_events() == {
            "compressions": 0,
            "tokens_removed": 0,
            "retrievals": 0,
        }


def test_unknown_event_types_are_ignored() -> None:
    events = [
        {"type": "compress", "input_tokens": 100, "output_tokens": 40},
        {"type": "unknown_future_kind", "anything": "goes"},
        {"type": "retrieve", "hash": "abc"},
        {"missing_type": True},  # malformed; should skip
    ]
    with _mock_events(events):
        result = _aggregate_mcp_events()
    assert result == {"compressions": 1, "tokens_removed": 60, "retrievals": 1}


def test_missing_token_fields_default_to_zero_without_raising() -> None:
    events = [
        {"type": "compress"},  # no token fields at all
        {"type": "compress", "input_tokens": 200, "output_tokens": None},  # None coerced
        {"type": "compress", "input_tokens": 100, "output_tokens": 100},  # zero diff
    ]
    with _mock_events(events):
        result = _aggregate_mcp_events()
    assert result["compressions"] == 3
    # First contributes 0 (both missing); second contributes 0 (None→0,
    # and output 0 → input - output = 200... wait that's NOT zero); third 0.
    # Let me re-derive: max(0, 0 - 0) + max(0, 200 - 0) + max(0, 100 - 100)
    #                 = 0 + 200 + 0 = 200
    assert result["tokens_removed"] == 200


def test_read_failure_yields_zeros() -> None:
    """If the shared-stats reader raises, the aggregator must not crash /stats."""
    with patch(
        "headroom.ccr.mcp_server._read_shared_events",
        side_effect=OSError("disk on fire"),
    ):
        assert _aggregate_mcp_events() == {
            "compressions": 0,
            "tokens_removed": 0,
            "retrievals": 0,
        }
