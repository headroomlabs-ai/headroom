"""Re-anchored cache breakpoints must keep the client's TTL.

``normalize_message_cache_control`` deliberately preserves an explicit
``cache_control.ttl`` so a client on Anthropic's 1h cache isn't silently
downgraded to the 5-minute default (#2375). Tool-search deferral also strips a
breakpoint and re-places it, and used to hardcode a bare ephemeral marker —
undoing that guarantee. A downgrade is invisible (the request still succeeds)
and costs a full prefix re-write on every gap past 5 minutes, so it needs a test
rather than a comment.
"""

from typing import Any

from headroom.proxy.helpers import inject_tool_search_deferral

TTL_1H = {"type": "ephemeral", "ttl": "1h"}


def _tools(marker: dict[str, Any] | None) -> list[dict[str, Any]]:
    # Needs >= _TOOL_SEARCH_MIN_TOOLS (12) to trigger, with one core tool resident
    # and the tools-array breakpoint riding on a tool that will be deferred.
    tools: list[dict[str, Any]] = [{"name": "read", "description": "core", "input_schema": {}}]
    for i in range(12):
        t: dict[str, Any] = {"name": f"rare_{i}", "description": "rare", "input_schema": {}}
        if marker is not None and i == 11:
            t["cache_control"] = marker
        tools.append(t)
    return tools


def _tool_markers(tools: Any) -> list[dict[str, Any]]:
    return [t["cache_control"] for t in tools if isinstance(t, dict) and "cache_control" in t]


def test_tool_search_deferral_keeps_ttl() -> None:
    out = inject_tool_search_deferral(_tools(TTL_1H))
    assert out is not _tools(TTL_1H), "deferral did not apply — fixture no longer triggers it"
    assert _tool_markers(out) == [TTL_1H], "tools breakpoint lost the 1h ttl"


def test_tool_search_deferral_defaults_to_5m() -> None:
    out = inject_tool_search_deferral(_tools({"type": "ephemeral"}))
    assert _tool_markers(out) == [{"type": "ephemeral"}]


def test_tool_search_deferral_no_breakpoint_adds_none() -> None:
    # Nothing was stripped, so nothing should be invented.
    assert _tool_markers(inject_tool_search_deferral(_tools(None))) == []
