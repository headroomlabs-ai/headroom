"""The forwarded body must satisfy Anthropic's cache_control TTL ordering.

Anthropic evaluates cache breakpoints in one global walk -- ``tools``, then
``system``, then ``messages`` -- and rejects the whole request when a
``ttl="1h"`` marker appears after a 5-minute one (a bare ``{"type":
"ephemeral"}`` marker *is* 5m)::

    400 messages.15.content.1.cache_control.ttl: a ttl='1h' cache_control block
    must not come after a ttl='5m' cache_control block.

Headroom rewrites markers in several independent places, section by section,
and until #2939 nothing checked the rule that spans them. The failure is a dead
turn rather than a silent cost regression, so it needs tests that pin both
repair directions and, just as importantly, pin that a legal request is passed
through by identity.
"""

from typing import Any

import pytest

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.helpers import (
    cache_control_ttl_lane,
    cache_control_ttl_lanes,
    enforce_cache_control_ttl_order,
    inject_tool_search_deferral,
)

TTL_1H: dict[str, Any] = {"type": "ephemeral", "ttl": "1h"}
BARE: dict[str, Any] = {"type": "ephemeral"}
TTL_5M: dict[str, Any] = {"type": "ephemeral", "ttl": "5m"}


def _text(text: str, marker: dict[str, Any] | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "text", "text": text}
    if marker is not None:
        block["cache_control"] = marker
    return block


def _msg(*blocks: dict[str, Any], role: str = "user") -> dict[str, Any]:
    return {"role": role, "content": list(blocks)}


def _markers(system: Any, messages: Any, tools: Any) -> list[dict[str, Any]]:
    """Every marker in Anthropic's evaluation order."""
    found: list[dict[str, Any]] = []
    for tool in tools or []:
        if isinstance(tool, dict) and isinstance(tool.get("cache_control"), dict):
            found.append(tool["cache_control"])
    for block in system or []:
        if isinstance(block, dict) and isinstance(block.get("cache_control"), dict):
            found.append(block["cache_control"])
    for msg in messages or []:
        if isinstance(msg.get("cache_control"), dict):
            found.append(msg["cache_control"])
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            if isinstance(block.get("cache_control"), dict):
                found.append(block["cache_control"])
            for sub in block.get("content") or []:
                if isinstance(sub, dict) and isinstance(sub.get("cache_control"), dict):
                    found.append(sub["cache_control"])
    return found


def _is_legal(system: Any, messages: Any, tools: Any) -> bool:
    """Reimplements the API's rule independently of the code under test."""
    seen_short = False
    for marker in _markers(system, messages, tools):
        lane = cache_control_ttl_lane(marker)
        if lane == "5m":
            seen_short = True
        elif lane == "1h" and seen_short:
            return False
    return True


# ---------------------------------------------------------------------------
# Lane classification


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ({"type": "ephemeral"}, "5m"),
        ({"type": "ephemeral", "ttl": "5m"}, "5m"),
        ({"type": "ephemeral", "ttl": "1h"}, "1h"),
        ({"type": "ephemeral", "ttl": "24h"}, "other"),
        ("not-a-dict", "other"),
    ],
)
def test_lane_classification(marker: Any, expected: str) -> None:
    # A bare marker must read as 5m, not "unknown": every ordinary Claude Code
    # request sends bare markers, and calling those unknown would either mask
    # real violations or invent imaginary ones.
    assert cache_control_ttl_lane(marker) == expected


def test_lanes_survey_covers_all_three_sections() -> None:
    lanes = cache_control_ttl_lanes(
        [_text("sys", BARE)],
        [_msg(_text("hi", TTL_1H))],
        [{"name": "read", "cache_control": {"type": "ephemeral", "ttl": "7d"}}],
    )
    assert lanes == {"5m", "1h", "other"}


# ---------------------------------------------------------------------------
# Repair 1: lane containment -- the /btw case from #2939


def test_replayed_1h_is_stripped_when_client_asked_for_5m() -> None:
    # Claude Code's `/btw` forks the conversation as a "side question", which is
    # not on its 1h allowlist: the fork's tools/system breakpoints are bare 5m
    # and it does not send the extended-cache-ttl beta header. Headroom's
    # overlay of the previous turn's forwarded bytes drags a 1h marker into
    # messages behind them, which is exactly the reported 400.
    tools = [{"name": "read", "cache_control": dict(BARE)}]
    system = [_text("sys", dict(BARE))]
    messages = [_msg(_text("old"), _text("replayed", dict(TTL_1H)))]

    system, messages, tools, stats = enforce_cache_control_ttl_order(
        system, messages, tools, client_uses_1h=False
    )

    assert stats["violation"] is True
    assert stats["demoted"] == 1
    assert stats["first_long_section"] == "messages"
    assert _markers(system, messages, tools) == [BARE, BARE, BARE], (
        "the leaked 1h ttl should be dropped, leaving the marker itself in place"
    )
    assert _is_legal(system, messages, tools)


def test_containment_keeps_non_ttl_marker_fields() -> None:
    # Claude Code also sends `scope` on its markers; only the ttl is at fault.
    scoped = {"type": "ephemeral", "ttl": "1h", "scope": "global"}
    _, messages, _, stats = enforce_cache_control_ttl_order(
        None, [_msg(_text("x", scoped))], None, client_uses_1h=False
    )
    assert stats["demoted"] == 1
    assert messages[0]["content"][0]["cache_control"] == {
        "type": "ephemeral",
        "scope": "global",
    }


def test_client_1h_is_never_stripped() -> None:
    system = [_text("sys", dict(TTL_1H))]
    messages = [_msg(_text("hi", dict(TTL_1H)))]
    out_system, out_messages, out_tools, stats = enforce_cache_control_ttl_order(
        system, messages, None, client_uses_1h=True
    )
    assert stats["violation"] is False
    assert out_system is system and out_messages is messages and out_tools is None


# ---------------------------------------------------------------------------
# Repair 2: ordering -- the #2767 case


def test_5m_in_tools_before_1h_in_messages_is_promoted() -> None:
    # A transform downgraded the tools breakpoint while the client's message
    # breakpoints are still 1h. Promoting restores what the client asked for;
    # demoting would throw away 1h caching it is already paying for.
    tools = [{"name": "read", "cache_control": dict(BARE)}]
    messages = [_msg(_text("hi", dict(TTL_1H)))]

    _, messages, tools, stats = enforce_cache_control_ttl_order(
        None, messages, tools, client_uses_1h=True
    )

    assert stats["promoted"] == 1
    assert stats["first_short_section"] == "tools"
    assert stats["first_long_section"] == "messages"
    assert tools[0]["cache_control"] == TTL_1H
    assert messages[0]["content"][0]["cache_control"] == TTL_1H
    assert _is_legal(None, messages, tools)


def test_5m_in_system_before_1h_in_messages_is_promoted() -> None:
    system = [_text("sys", dict(TTL_5M))]
    messages = [_msg(_text("hi", dict(TTL_1H)))]

    system, messages, _, stats = enforce_cache_control_ttl_order(
        system, messages, None, client_uses_1h=True
    )

    assert stats["first_short_section"] == "system"
    assert system[0]["cache_control"] == TTL_1H


def test_violation_within_messages_is_promoted() -> None:
    messages = [
        _msg(_text("a", dict(BARE))),
        _msg(_text("b"), _text("c", dict(TTL_1H))),
    ]
    _, messages, _, stats = enforce_cache_control_ttl_order(
        None, messages, None, client_uses_1h=True
    )
    assert stats["promoted"] == 1
    assert messages[0]["content"][0]["cache_control"] == TTL_1H


def test_nested_tool_result_markers_participate() -> None:
    # tool_result carries its own content list; a marker hiding in there is
    # still a breakpoint the API walks, so it must count for the ordering.
    messages = [
        _msg(
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": [_text("inner", dict(BARE))],
            }
        ),
        _msg(_text("later", dict(TTL_1H))),
    ]
    _, messages, _, stats = enforce_cache_control_ttl_order(
        None, messages, None, client_uses_1h=True
    )
    assert stats["promoted"] == 1
    assert messages[0]["content"][0]["content"][0]["cache_control"] == TTL_1H


def test_only_markers_before_the_last_1h_are_promoted() -> None:
    # A 5m marker AFTER every 1h one is legal and must be left alone -- that is
    # the ordering the API documents, not something to normalise away.
    messages = [
        _msg(_text("a", dict(BARE))),
        _msg(_text("b", dict(TTL_1H))),
        _msg(_text("c", dict(BARE))),
    ]
    _, messages, _, stats = enforce_cache_control_ttl_order(
        None, messages, None, client_uses_1h=True
    )
    assert stats["promoted"] == 1
    assert [m["content"][0]["cache_control"] for m in messages] == [TTL_1H, TTL_1H, BARE]


# ---------------------------------------------------------------------------
# Pass-through cases


@pytest.mark.parametrize(
    "markers",
    [
        pytest.param([TTL_1H, TTL_1H], id="all-1h"),
        pytest.param([BARE, BARE], id="all-5m"),
        pytest.param([TTL_1H, BARE], id="1h-then-5m"),
        pytest.param([], id="no-markers"),
    ],
)
def test_legal_requests_are_returned_by_identity(markers: list[dict[str, Any]]) -> None:
    messages = [_msg(_text(f"m{i}", dict(m))) for i, m in enumerate(markers)] or [_msg(_text("m"))]
    out_system, out_messages, out_tools, stats = enforce_cache_control_ttl_order(
        None, messages, None, client_uses_1h=True
    )
    assert stats["violation"] is False
    assert out_messages is messages, "a legal body must not be rebuilt"
    assert out_system is None and out_tools is None


def test_unknown_ttl_is_left_alone() -> None:
    # Mirrors TtlOrderingWalk::observe in headroom-core: a TTL lane we don't
    # model takes no part in the rule and is never rewritten.
    messages = [_msg(_text("a", {"type": "ephemeral", "ttl": "24h"})), _msg(_text("b", dict(BARE)))]
    _, out, _, stats = enforce_cache_control_ttl_order(None, messages, None, client_uses_1h=False)
    assert stats["violation"] is False
    assert out is messages


def test_kill_switch_disables_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CACHE_CONTROL_TTL_GUARD", "0")
    tools = [{"name": "read", "cache_control": dict(BARE)}]
    messages = [_msg(_text("hi", dict(TTL_1H)))]
    _, out_messages, out_tools, stats = enforce_cache_control_ttl_order(
        None, messages, tools, client_uses_1h=True
    )
    assert stats["violation"] is False
    assert out_messages is messages and out_tools is tools


# ---------------------------------------------------------------------------
# Tool sort must not reorder markers


def _tools(count: int, marked: dict[int, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(count):
        # Names descend so an alphabetical sort is guaranteed to reorder them.
        tool: dict[str, Any] = {"name": f"tool_{count - i:02d}", "input_schema": {}}
        if marked and i in marked:
            tool["cache_control"] = dict(marked[i])
        out.append(tool)
    return out


def test_tool_sort_is_skipped_when_a_tool_carries_a_marker() -> None:
    # A breakpoint on a tool means "cache through here"; sorting changes which
    # tools are inside that prefix, and with two TTLs it can put the 1h marker
    # behind the 5m one. The Rust proxy already refuses for the same reason.
    tools = _tools(4, {1: TTL_1H, 3: BARE})
    assert AnthropicHandlerMixin._sort_tools_deterministically(tools) is tools


def test_tool_sort_still_sorts_unmarked_tools() -> None:
    tools = _tools(4)
    ordered = AnthropicHandlerMixin._sort_tools_deterministically(tools)
    assert [t["name"] for t in ordered] == sorted(t["name"] for t in tools), (
        "clients that mark no tools must keep the deterministic ordering they rely on"
    )


def test_tool_sort_would_have_created_the_violation() -> None:
    # Pins the hazard itself: without the guard, the alphabetical sort moves the
    # 5m-marked tool ahead of the 1h-marked one, which is a 400 on its own.
    tools = _tools(4, {1: TTL_1H, 3: BARE})
    assert _is_legal(None, [], tools)
    assert not _is_legal(None, [], sorted(tools, key=AnthropicHandlerMixin._tool_sort_key))


# ---------------------------------------------------------------------------
# End-to-end regression for #2939 / #2767


def test_deferral_downgrade_then_guard_yields_a_legal_body() -> None:
    # The #2767 shape: 13 tools, a 1h marker on one deferred tool and a bare
    # marker on a LATER deferred tool. inject_tool_search_deferral keeps the
    # last marker it stripped, so the tools prefix lands at 5m while the
    # client's message breakpoints are still 1h -- a 400. The guard repairs it.
    tools: list[dict[str, Any]] = [{"name": "read", "description": "core", "input_schema": {}}]
    for i in range(12):
        tool: dict[str, Any] = {"name": f"rare_{i}", "description": "rare", "input_schema": {}}
        if i == 4:
            tool["cache_control"] = dict(TTL_1H)
        if i == 9:
            tool["cache_control"] = dict(BARE)
        tools.append(tool)
    messages = [_msg(_text("history")), _msg(_text("newest", dict(TTL_1H)))]

    deferred = inject_tool_search_deferral(tools)
    assert deferred is not tools, "fixture no longer triggers the deferral"
    assert not _is_legal(None, messages, deferred), (
        "expected the downgraded tools breakpoint to make the body illegal"
    )

    _, messages, deferred, stats = enforce_cache_control_ttl_order(
        None, messages, deferred, client_uses_1h=True
    )
    assert stats["violation"] is True
    assert _is_legal(None, messages, deferred)
