"""Session RTK savings must be the delta from the proxy-startup baseline.

Regression for the scope-mixing bug: the dashboard's *session* RTK number must
be computed from token deltas since the baseline pinned at proxy startup — NOT
from RTK's lifetime average (which dilutes a 62%-this-session rate down to an
18.5% all-time number). This exercises the real ``_get_context_tool_stats()``
plumbing rather than asserting the arithmetic in the abstract.
"""

from __future__ import annotations

import headroom.proxy.helpers as helpers


def _reset(monkeypatch):
    monkeypatch.delenv(helpers._RTK_GAIN_SCOPE_ENV, raising=False)
    monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "rtk")
    helpers._context_tool_stats_cache.update(
        {"expires_at": 0.0, "has_value": False, "tool": None, "value": None}
    )
    helpers._context_tool_session_baseline.update(
        {
            "initialized": False,
            "tool": None,
            "total_commands": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "tokens_saved": 0,
            "total_time_ms": 0,
            "captured_at": 0.0,
        }
    )


def _bust_cache():
    helpers._context_tool_stats_cache.update(
        {"expires_at": 0.0, "has_value": False, "tool": None, "value": None}
    )


def test_session_savings_is_delta_not_lifetime_average(monkeypatch):
    _reset(monkeypatch)

    state: dict = {"summary": None}

    def fake_lifetime(tool):
        return helpers._context_tool_summary_payload(
            tool="rtk", installed=True, scope="global", summary=state["summary"]
        )

    monkeypatch.setattr(helpers, "_read_context_tool_lifetime_stats", fake_lifetime)

    # First poll pins the baseline to the current lifetime → session delta is 0,
    # but the lifetime number is preserved untouched.
    state["summary"] = {"total_input": 1000, "total_output": 400, "total_saved": 600}
    first = helpers._get_context_tool_stats()
    assert first is not None
    assert first["session"]["tokens_saved"] == 0
    assert first["lifetime"]["tokens_saved"] == 600

    # Lifetime advances (more RTK commands run this session); the session number
    # is the DELTA, not the 800 lifetime total.
    _bust_cache()
    state["summary"] = {"total_input": 1300, "total_output": 500, "total_saved": 800}
    second = helpers._get_context_tool_stats()
    assert second["session"]["tokens_saved"] == 200  # 800 - 600
    assert second["lifetime"]["tokens_saved"] == 800
    # Session % is derived from the delta (200 saved / 300 input delta), not the
    # lifetime-diluted average.
    assert second["session"]["savings_pct"] == round(200 / 300 * 100, 4)
