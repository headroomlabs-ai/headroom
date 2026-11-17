"""The CCR retrieval tool must not enter the tools array mid-session.

``tools`` is the head of Anthropic's cache key — the prefix is ordered
tools → system → messages — so any change to the array invalidates every
cached token behind it. Injecting ``headroom_retrieve`` at the first
compression means entering the array against a fully warm prefix.

Measured against the live API with the real definition: a warm prefix reading
4,335 tokens dropped to 0 read and 4,458 written the moment the 478-byte tool
was appended. On a customer session the same effect cost 113,888 tokens of
cache write to save 2,205 tokens of content — a payback of ~594 turns *within
one session*, which no session reaches.

The fix is timing, not mechanism: inject on the first request, while the
prefix is still cold, and the array never changes again.
"""

from __future__ import annotations

import json
import re

import pytest

from headroom.cache.compression_store import CompressionStore
from headroom.ccr.tool_injection import CCR_TOOL_NAME, CCRToolInjector, create_ccr_tool_definition
from headroom.proxy.helpers import (
    _reset_session_ccr_tracker_for_test,
    apply_session_sticky_ccr_tool,
)


@pytest.fixture(autouse=True)
def _reset_tracker():
    _reset_session_ccr_tracker_for_test()
    yield
    _reset_session_ccr_tracker_for_test()


@pytest.fixture(autouse=True)
def _default_eager(monkeypatch):
    """Pin the default explicitly so a stray env cannot mask a regression."""
    monkeypatch.delenv("HEADROOM_CCR_TOOL_INJECTION", raising=False)


CLIENT_TOOLS = [
    {
        "name": "Read",
        "description": "Read a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
    {
        "name": "Bash",
        "description": "Run a command.",
        "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
    },
]


def _apply(session_id, *, compressed, provider="anthropic", allow_eager=True):
    """``allow_eager=True`` mirrors what both handlers pass for a request that
    can actually compress (optimization on, no bypass header)."""
    return apply_session_sticky_ccr_tool(
        provider=provider,
        session_id=session_id,
        request_id="req-1",
        existing_tools=CLIENT_TOOLS,
        has_compressed_content_this_turn=compressed,
        allow_eager=allow_eager,
    )


def _names(tools):
    return [t.get("name") or t.get("function", {}).get("name") for t in tools]


# ── the regression this file exists for ───────────────────────────────


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_tools_array_is_byte_stable_across_the_first_compression(provider):
    """The turn that first compresses must not change the tools array.

    This is the whole bug. Before the fix, turn 1 returned two tools and turn 2
    returned three — and because tools sit first in the cache prefix, that one
    appended entry invalidated the entire warm prefix behind it.
    """
    session = f"sess-{provider}-stability"

    before, _ = _apply(session, compressed=False, provider=provider)
    during, _ = _apply(session, compressed=True, provider=provider)
    after, _ = _apply(session, compressed=False, provider=provider)

    assert json.dumps(before) == json.dumps(during) == json.dumps(after), (
        "tools array changed across the first compression — this invalidates "
        "the whole provider cache prefix"
    )


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_the_tool_is_present_from_the_very_first_request(provider):
    """Cold prefix is the only cheap moment to add to the tools array."""
    tools, injected = _apply(f"sess-{provider}-first", compressed=False, provider=provider)
    assert injected is True
    assert CCR_TOOL_NAME in _names(tools)


def test_the_tool_definition_is_small_enough_that_eager_is_obviously_right():
    """Bound the cold-prefix overhead that justifies eager injection."""
    size = len(json.dumps(create_ccr_tool_definition()))
    assert size < 2000, (
        f"headroom_retrieve is now {size} bytes; eager injection was justified "
        "on it being ~478 bytes (~119 tokens) of the session's first cache write"
    )


# ── the escape hatch still works ──────────────────────────────────────


def test_lazy_mode_restores_the_historical_gate(monkeypatch):
    monkeypatch.setenv("HEADROOM_CCR_TOOL_INJECTION", "lazy")
    session = "sess-lazy"

    tools, injected = _apply(session, compressed=False)
    assert injected is False
    assert CCR_TOOL_NAME not in _names(tools)

    tools, injected = _apply(session, compressed=True)
    assert injected is True
    assert CCR_TOOL_NAME in _names(tools)


def test_an_invalid_mode_raises_rather_than_silently_defaulting(monkeypatch):
    from headroom.proxy.helpers import get_ccr_tool_injection_mode

    monkeypatch.setenv("HEADROOM_CCR_TOOL_INJECTION", "sometimes")
    with pytest.raises(ValueError, match="HEADROOM_CCR_TOOL_INJECTION"):
        get_ccr_tool_injection_mode()


# ── properties the fix must not break ─────────────────────────────────


@pytest.mark.parametrize("empty", [None, []])
def test_a_request_with_no_tools_is_not_armed_eagerly(empty):
    """Eager injection must not turn a no-tools request into a tools request.

    Adding the first entry to an absent tools array lets the model emit
    tool_use blocks the client never expected (#728). Such a client is also not
    a harness: it has no tool results to compress, so there is no warm tools
    segment for eager injection to protect. It keeps the historical gate.
    """
    tools, injected = apply_session_sticky_ccr_tool(
        provider="anthropic",
        session_id="sess-no-tools",
        request_id="req-1",
        existing_tools=empty,
        has_compressed_content_this_turn=False,
        allow_eager=True,
    )
    assert injected is False
    assert tools == []


def test_a_client_provided_tool_still_wins(monkeypatch):
    """An MCP-registered headroom_retrieve must not be doubled up."""
    client_owned = [*CLIENT_TOOLS, {"name": CCR_TOOL_NAME, "description": "client's own"}]
    tools, injected = apply_session_sticky_ccr_tool(
        provider="anthropic",
        session_id="sess-client-owned",
        request_id="req-1",
        existing_tools=client_owned,
        has_compressed_content_this_turn=False,
        allow_eager=True,
    )
    assert injected is False
    assert _names(tools).count(CCR_TOOL_NAME) == 1


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize(
    "alias",
    [
        "headroom_headroom_retrieve",
        "mcp__headroom__headroom_retrieve",
        "mcp_Headroom_headroom_retrieve",
    ],
)
def test_client_retrieval_alias_never_advertises_an_unavailable_tool(provider, alias):
    """Mixed-tool turns must only offer retrieval names the client can execute."""
    retrieval = create_ccr_tool_definition(provider)
    if provider == "openai":
        retrieval["function"]["name"] = alias
        read_tool = {"type": "function", "function": {"name": "Read", "parameters": {}}}
    else:
        retrieval["name"] = alias
        read_tool = CLIENT_TOOLS[0]
    client_tools = [retrieval, read_tool]
    session_id = f"sess-client-alias-{provider}-{alias}"

    # A cold request, actual compression, and later replay all retain a usable
    # retrieval tool without introducing the unavailable bare proxy spelling.
    for compressed in (False, True, False):
        tools, injected = apply_session_sticky_ccr_tool(
            provider=provider,
            session_id=session_id,
            request_id="req-client-alias",
            existing_tools=client_tools,
            has_compressed_content_this_turn=compressed,
            allow_eager=True,
        )
        assert not injected
        assert alias in _names(tools)
        assert CCR_TOOL_NAME not in _names(tools)

    # The no-session library/Google fallback must obey the same client registry.
    injector = CCRToolInjector(provider=provider, inject_tool=True)
    tools, injected = injector.inject_tool_definition(client_tools, session_has_done_ccr=True)
    assert not injected
    assert alias in _names(tools)
    assert CCR_TOOL_NAME not in _names(tools)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("alias", ["mcp__other__headroom_retrieve", "mcp_Other_headroom_retrieve"])
def test_unrelated_mcp_server_cannot_displace_proxy_retrieval(provider, alias):
    retrieval = create_ccr_tool_definition(provider)
    if provider == "openai":
        retrieval["function"]["name"] = alias
    else:
        retrieval["name"] = alias
    client_tools = [retrieval]
    tools, _ = apply_session_sticky_ccr_tool(
        provider=provider,
        session_id=None,
        request_id="req-unrelated-mcp",
        existing_tools=client_tools,
        has_compressed_content_this_turn=True,
    )
    assert CCR_TOOL_NAME in _names(tools)
    assert alias in _names(tools)

    injector = CCRToolInjector(provider=provider)
    fallback_tools, _ = injector.inject_tool_definition(client_tools, session_has_done_ccr=True)
    assert CCR_TOOL_NAME in _names(fallback_tools)
    assert alias in _names(fallback_tools)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize(
    "alias",
    [
        "headroom_headroom_retrieve",
        "mcp__headroom__headroom_retrieve",
        "mcp_Headroom_headroom_retrieve",
    ],
)
def test_retrieval_instructions_name_an_executable_client_tool(provider, alias):
    store = CompressionStore(enable_feedback=False)
    hash_key = store.store(original='{"value":42}', compressed="compressed")
    messages = [
        {"role": "system", "content": "Use the available tools."},
        {"role": "tool", "content": f"<<ccr:{hash_key} 1_rows_offloaded>>"},
    ]
    retrieval = create_ccr_tool_definition(provider)
    if provider == "openai":
        retrieval["function"]["name"] = alias
    else:
        retrieval["name"] = alias
    injector = CCRToolInjector(
        provider=provider,
        inject_system_instructions=True,
        compression_store=store,
    )
    updated_messages, updated_tools, _ = injector.process_request(messages, [retrieval])
    instructions = updated_messages[0]["content"]
    call_names = re.findall(r"`([^`]+)\(hash=", instructions)
    assert call_names == [alias]
    assert set(call_names).issubset(_names(updated_tools))
    assert hash_key in instructions


def test_sticky_replay_is_byte_identical_to_the_eager_injection():
    """Later turns replay the golden bytes recorded by the eager injection."""
    session = "sess-golden"
    first, _ = _apply(session, compressed=False)
    later, _ = _apply(session, compressed=True)
    assert json.dumps(first) == json.dumps(later)


def test_the_sessionless_path_is_also_stable():
    """No session id means no tracker, so the flag alone drove the toggle."""
    a, _ = _apply(None, compressed=False)
    b, _ = _apply(None, compressed=True)
    assert json.dumps(a) == json.dumps(b)
    assert CCR_TOOL_NAME in _names(a)


def test_a_request_that_cannot_compress_is_not_armed():
    """`--no-optimize` or a bypass header means nothing will ever be compressed.

    Pre-arming there would leave a permanently unredeemable tool in the client's
    array. The old gate got this for free: no compression meant no injection.
    """
    tools, injected = _apply("sess-no-optimize", compressed=False, allow_eager=False)
    assert injected is False
    assert CCR_TOOL_NAME not in _names(tools)
