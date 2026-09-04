"""Issue #746: keep Claude Code's on-demand tool loading active through the proxy.

Covers the two halves of the fix:

* ``headroom wrap claude`` injects ``ENABLE_TOOL_SEARCH`` into the launched
  Claude Code environment (with correct precedence / validation), and
* the proxy detects a Claude Code request that is *not* deferring tools and
  emits a single actionable hint for users who run ``claude`` manually.
"""

from __future__ import annotations

import pytest

from headroom.cli.wrap import (
    _TOOL_SEARCH_DEFAULT,
    _TOOL_SEARCH_ENV,
    _configure_tool_search_env,
    _normalize_tool_search_mode,
)
from headroom.proxy.helpers import (
    claude_code_tool_search_active,
    claude_code_tool_search_inactive,
    format_tool_search_disabled_hint,
    reset_tool_search_hint_state,
    take_tool_search_hint_slot,
    tool_search_hint_pending,
)

# ---------------------------------------------------------------------------
# wrap: ENABLE_TOOL_SEARCH value normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", "true"),
        ("TRUE", "true"),
        (" on ", "on"),
        ("1", "1"),
        ("false", "false"),
        ("off", "off"),
        ("auto", "auto"),
        ("auto:0", "auto:0"),
        ("auto:50", "auto:50"),
        ("auto:100", "auto:100"),
    ],
)
def test_normalize_tool_search_mode_accepts_valid(value: str, expected: str) -> None:
    assert _normalize_tool_search_mode(value) == expected


@pytest.mark.parametrize("value", ["yep", "auto:", "auto:101", "auto:-1", "auto:abc", ""])
def test_normalize_tool_search_mode_rejects_invalid(value: str) -> None:
    import click

    with pytest.raises(click.ClickException):
        _normalize_tool_search_mode(value)


# ---------------------------------------------------------------------------
# wrap: ENABLE_TOOL_SEARCH injection precedence
# ---------------------------------------------------------------------------


def test_configure_injects_default_when_unset() -> None:
    env: dict[str, str] = {}
    result = _configure_tool_search_env(env, None)
    assert result == _TOOL_SEARCH_DEFAULT
    assert env[_TOOL_SEARCH_ENV] == _TOOL_SEARCH_DEFAULT


def test_configure_respects_existing_env_value() -> None:
    env = {_TOOL_SEARCH_ENV: "auto:30"}
    result = _configure_tool_search_env(env, None)
    # None signals "left the user's value untouched".
    assert result is None
    assert env[_TOOL_SEARCH_ENV] == "auto:30"


def test_configure_flag_overrides_existing_env_value() -> None:
    env = {_TOOL_SEARCH_ENV: "false"}
    result = _configure_tool_search_env(env, "auto")
    assert result == "auto"
    assert env[_TOOL_SEARCH_ENV] == "auto"


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_configure_overrides_blank_env_value(blank: str) -> None:
    # Claude Code treats an empty ENABLE_TOOL_SEARCH as unset, so a blank value
    # must be replaced with the default rather than forwarded as a no-op.
    env = {_TOOL_SEARCH_ENV: blank}
    result = _configure_tool_search_env(env, None)
    assert result == _TOOL_SEARCH_DEFAULT
    assert env[_TOOL_SEARCH_ENV] == _TOOL_SEARCH_DEFAULT


def test_configure_flag_validated() -> None:
    import click

    with pytest.raises(click.ClickException):
        _configure_tool_search_env({}, "nonsense")


# ---------------------------------------------------------------------------
# proxy: detect a Claude Code request that is not deferring tools
# ---------------------------------------------------------------------------

_TOOLS = [
    {"name": "Read", "description": "read a file", "input_schema": {"type": "object"}},
    {"name": "Bash", "description": "run a command", "input_schema": {"type": "object"}},
]


def test_inactive_true_for_eager_claude_code() -> None:
    assert claude_code_tool_search_inactive(client="claude-code", tools=_TOOLS, anthropic_beta=None)


def test_inactive_false_when_tool_search_tool_present() -> None:
    tools = [*_TOOLS, {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}]
    assert not claude_code_tool_search_inactive(
        client="claude-code", tools=tools, anthropic_beta=None
    )


def test_inactive_false_when_beta_header_present() -> None:
    assert not claude_code_tool_search_inactive(
        client="claude-code",
        tools=_TOOLS,
        anthropic_beta="context-1m-2025-08-07,advanced-tool-use-2025-11-20",
    )


def test_inactive_false_for_other_clients() -> None:
    assert not claude_code_tool_search_inactive(client="codex", tools=_TOOLS, anthropic_beta=None)
    assert not claude_code_tool_search_inactive(client=None, tools=_TOOLS, anthropic_beta=None)


def test_inactive_false_when_no_tools() -> None:
    assert not claude_code_tool_search_inactive(client="claude-code", tools=[], anthropic_beta=None)
    assert not claude_code_tool_search_inactive(
        client="claude-code", tools=None, anthropic_beta=None
    )


def test_active_requires_explicit_tool_search_protocol_marker() -> None:
    assert claude_code_tool_search_active(
        client="claude-code",
        tools=[*_TOOLS, {"type": "tool_search_tool_regex_20251119"}],
        anthropic_beta=None,
    )
    assert claude_code_tool_search_active(
        client="claude-code",
        tools=_TOOLS,
        anthropic_beta="advanced-tool-use-2025-11-20",
    )
    assert not claude_code_tool_search_active(
        client="claude-code",
        tools=[{"name": "WaitForMcpServers"}],
        anthropic_beta=None,
    )


def test_active_marker_boundary_value_zero_does_not_use_tool_name() -> None:
    tools = [{"name": "WaitForMcpServers"}]
    assert not claude_code_tool_search_active(client="claude-code", tools=tools, anthropic_beta="")
    assert claude_code_tool_search_inactive(client="claude-code", tools=tools, anthropic_beta="")


# ---------------------------------------------------------------------------
# proxy: hint content + one-time guard
# ---------------------------------------------------------------------------


def test_hint_message_is_actionable() -> None:
    msg = format_tool_search_disabled_hint(_TOOLS)
    assert "ENABLE_TOOL_SEARCH=true" in msg
    assert "746" in msg
    assert str(len(_TOOLS)) in msg


def test_hint_slot_fires_once() -> None:
    reset_tool_search_hint_state()
    try:
        assert tool_search_hint_pending() is True
        assert take_tool_search_hint_slot() is True
        # Once consumed, the cheap gate flips so the hot path stops scanning.
        assert tool_search_hint_pending() is False
        assert take_tool_search_hint_slot() is False
        assert take_tool_search_hint_slot() is False
    finally:
        reset_tool_search_hint_state()


# ---------------------------------------------------------------------------
# Server-side Tool Search injection for plain-API clients (opencode)
# ---------------------------------------------------------------------------

from headroom.proxy.helpers import (  # noqa: E402
    _TOOL_SEARCH_DEFAULT_NAME,
    _TOOL_SEARCH_DEFAULT_TYPE,
    _TOOL_SEARCH_MIN_TOOLS,
    anthropic_first_party_tool_search_supported,
    inject_tool_search_deferral,
    strip_first_party_tool_search_tools_for_third_party_upstream,
)


def _tools(n: int, *, core_first: int = 0) -> list[dict]:
    core = ["bash", "read", "write", "edit", "grep"]
    out: list[dict] = []
    for i in range(n):
        name = core[i] if i < core_first and i < len(core) else f"mcp_tool_{i}"
        out.append({"name": name, "description": f"tool {i}", "input_schema": {}})
    return out


def test_inject_defers_non_core_and_injects_search_tool() -> None:
    tools = _tools(20, core_first=3)  # bash/read/write resident, rest deferred
    out = inject_tool_search_deferral(tools)
    assert out is not tools
    # search tool injected, non-deferred, correct shape
    search = out[0]
    assert search == {"type": _TOOL_SEARCH_DEFAULT_TYPE, "name": _TOOL_SEARCH_DEFAULT_NAME}
    assert "defer_loading" not in search
    # core tools stay resident; non-core deferred
    by_name = {t.get("name"): t for t in out if "name" in t}
    assert by_name["bash"].get("defer_loading") is None
    assert by_name["mcp_tool_5"].get("defer_loading") is True
    # at least one non-deferred real tool remains (Anthropic 400s otherwise)
    assert any(not t.get("type") and not t.get("defer_loading") for t in out)


def test_noop_below_min_tools() -> None:
    tools = _tools(_TOOL_SEARCH_MIN_TOOLS - 1)
    assert inject_tool_search_deferral(tools) is tools


def test_noop_when_client_already_uses_tool_search() -> None:
    tools = _tools(20) + [{"type": "tool_search_tool_regex_20251119", "name": "x"}]
    assert inject_tool_search_deferral(tools) is tools


def test_noop_when_nothing_to_defer() -> None:
    # every tool is core -> nothing deferred -> cache prefix untouched
    core = [
        "bash",
        "read",
        "write",
        "edit",
        "multiedit",
        "glob",
        "grep",
        "task",
        "todowrite",
        "todoread",
        "webfetch",
        "skill",
    ]
    tools = [{"name": n, "input_schema": {}} for n in core]
    assert inject_tool_search_deferral(tools) is tools


def test_cache_control_moved_off_deferred_tool_to_last_resident() -> None:
    tools = _tools(20, core_first=3)
    # the client's tools cache breakpoint sits on a tool we will defer
    tools[10]["cache_control"] = {"type": "ephemeral"}
    out = inject_tool_search_deferral(tools)
    # no deferred tool may carry cache_control (Anthropic 400s)
    assert all("cache_control" not in t for t in out if t.get("defer_loading"))
    # exactly one resident real tool now carries the moved breakpoint
    resident_cc = [
        t
        for t in out
        if not t.get("type") and not t.get("defer_loading") and t.get("cache_control")
    ]
    assert len(resident_cc) == 1


def test_non_dict_and_typed_tools_stay_resident() -> None:
    tools = _tools(15, core_first=2)
    tools.append({"type": "web_search_20250305", "name": "web_search"})
    out = inject_tool_search_deferral(tools)
    typed = [t for t in out if t.get("type") == "web_search_20250305"]
    assert len(typed) == 1 and typed[0].get("defer_loading") is None


def test_third_party_upstream_strips_first_party_tool_search_from_headroom_issue_2526() -> None:
    tools = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {"name": "Bash", "description": "run a command", "input_schema": {}},
        {"type": "web_search_20250305", "name": "web_search"},
    ]
    out = strip_first_party_tool_search_tools_for_third_party_upstream(
        tools,
        "https://api.deepseek.com/anthropic",
    )
    assert out is not tools
    assert [tool.get("name") for tool in out if isinstance(tool, dict)] == ["Bash", "web_search"]
    assert all(
        not str(tool.get("type", "")).startswith("tool_search_tool_")
        for tool in out
        if isinstance(tool, dict)
    )


def test_first_party_anthropic_preserves_client_tool_search_entry() -> None:
    tools = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {"name": "Bash", "description": "run a command", "input_schema": {}},
    ]
    assert anthropic_first_party_tool_search_supported("https://api.anthropic.com")
    assert (
        strip_first_party_tool_search_tools_for_third_party_upstream(
            tools,
            "https://api.anthropic.com",
        )
        is tools
    )


@pytest.mark.parametrize(
    ("api_base_url", "expected_supported"),
    [
        ("https://api.anthropic.com", True),
        ("https://api.anthropic.com/v1", True),
        ("https://api.deepseek.com/anthropic", False),
        ("http://127.0.0.1:8787", False),
    ],
)
def test_third_party_or_first_party_matrix(api_base_url: str, expected_supported: bool) -> None:
    assert anthropic_first_party_tool_search_supported(api_base_url) is expected_supported


# ---------------------------------------------------------------------------
# PascalCase clients (Claude Code). The core-tool exemption is spelled in
# lowercase, so an exact-match comparison never fired for Claude Code: every
# tool was deferred, including Claude Code's own ``ToolSearch``.
# ---------------------------------------------------------------------------


def _claude_code_tools() -> list[dict]:
    """Claude Code's surface: PascalCase built-ins, its ToolSearch, MCP tools."""
    names = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "ToolSearch"] + [
        f"mcp__srv__t{i}" for i in range(12)
    ]
    return [{"name": n, "description": n, "input_schema": {}} for n in names]


def test_core_tools_match_case_insensitively() -> None:
    # Without a case-insensitive match, routine edit/read/run loops each pay a
    # search round-trip — the exact thing _TOOL_SEARCH_CORE_TOOLS exists to avoid.
    out = inject_tool_search_deferral(_claude_code_tools())
    by_name = {t.get("name"): t for t in out if "name" in t}
    for name in ("Bash", "Read", "Write", "Edit", "Glob", "Grep"):
        assert by_name[name].get("defer_loading") is None, name
    # MCP tools are still deferred — the token saving is preserved.
    assert by_name["mcp__srv__t0"].get("defer_loading") is True


def test_client_tool_search_tool_is_never_deferred() -> None:
    # ToolSearch is the client's own schema fetcher for tools that never appear
    # in the request body (TaskCreate, WebFetch, …). Deferring it hides the only
    # tool that can load them, so they become permanently unreachable.
    out = inject_tool_search_deferral(_claude_code_tools())
    by_name = {t.get("name"): t for t in out if "name" in t}
    assert by_name["ToolSearch"].get("defer_loading") is None


def test_resident_real_tool_survives_pascal_case_surface() -> None:
    # The injected search tool is typed and does not satisfy the invariant on its
    # own; Anthropic 400s when every real tool is deferred.
    out = inject_tool_search_deferral(_claude_code_tools())
    assert any(not t.get("type") and not t.get("defer_loading") for t in out)


def _omp_tools() -> list[dict]:
    """Oh My Pi's 12-tool surface: underscore-prefixed built-ins plus typed tools."""
    named = [
        "_hub",
        "_edit",
        "_task",
        "_todo",
        "_eval",
        "_read",
        "_bash",
        "_glob",
        "_grep",
        "_write",
    ]
    return [
        *[{"name": name, "description": name, "input_schema": {}} for name in named],
        {"type": "computer_20250124", "name": "computer"},
        {"type": "web_search_20250305", "name": "web_search"},
    ]


def test_core_tools_match_leading_underscore_namespace() -> None:
    tools = _omp_tools()
    assert len(tools) == _TOOL_SEARCH_MIN_TOOLS

    out = inject_tool_search_deferral(tools)

    by_name = {tool.get("name"): tool for tool in out if isinstance(tool, dict)}
    for name in ("_edit", "_task", "_read", "_bash", "_glob", "_grep", "_write"):
        assert by_name[name].get("defer_loading") is None, name
    for name in ("_hub", "_todo", "_eval"):
        assert by_name[name].get("defer_loading") is True, name
    for name in ("computer", "web_search"):
        assert by_name[name].get("defer_loading") is None, name


# ---------------------------------------------------------------------------
# Tool-search history repair (#2805)
#
# Anthropic validates every tool_reference in the transcript against the
# request's tools array. Claude Code replays one transcript across requests
# with DIFFERENT tools arrays (main loop vs prompt-type Stop hook evaluator),
# so the side-request 400s with "Tool reference 'X' not found in available
# tools". The repair drops blocks a request cannot support.
# ---------------------------------------------------------------------------

from headroom.proxy.helpers import (  # noqa: E402
    strip_unsupported_tool_search_blocks,
)

_SEARCH_TOOL = {"type": _TOOL_SEARCH_DEFAULT_TYPE, "name": _TOOL_SEARCH_DEFAULT_NAME}


def _poisoned_transcript() -> list[dict]:
    """A transcript as Claude Code stores it after one server-side tool search."""
    return [
        {"role": "user", "content": [{"type": "text", "text": "ask the user"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Searching for a tool."},
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_01ABC",
                    "name": _TOOL_SEARCH_DEFAULT_NAME,
                    "input": {"pattern": "question|ask"},
                },
                {
                    "type": "tool_search_tool_result",
                    "tool_use_id": "srvtoolu_01ABC",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [
                            {"type": "tool_reference", "tool_name": "AskUserQuestion"}
                        ],
                    },
                },
                {"type": "text", "text": "Found it."},
            ],
        },
    ]


def test_repair_drops_blocks_the_hook_evaluator_cannot_resolve() -> None:
    # The Stop hook evaluator replays the transcript with a small tools array
    # that has neither the search tool nor AskUserQuestion -> upstream 400.
    messages, removed = strip_unsupported_tool_search_blocks(
        _poisoned_transcript(), [{"name": "Read", "input_schema": {}}]
    )
    assert removed == 2  # server_tool_use + tool_search_tool_result
    kinds = [b["type"] for b in messages[1]["content"]]
    assert kinds == ["text", "text"]  # surrounding assistant text survives
    assert messages[0]["content"][0]["text"] == "ask the user"


def test_repair_is_noop_on_the_main_loop() -> None:
    # Same transcript, but the request carries the injected search tool AND the
    # referenced tool: nothing to repair, and the object is returned by identity
    # so the outbound prefix (and its cache) is untouched.
    transcript = _poisoned_transcript()
    messages, removed = strip_unsupported_tool_search_blocks(
        transcript,
        [_SEARCH_TOOL, {"name": "AskUserQuestion", "input_schema": {}, "defer_loading": True}],
    )
    assert removed == 0
    assert messages is transcript


def test_repair_drops_a_turn_left_with_no_blocks() -> None:
    # An assistant turn that was ONLY the search round-trip must be removed, not
    # forwarded with an empty content array (which Anthropic also rejects).
    transcript = _poisoned_transcript()
    transcript[1]["content"] = transcript[1]["content"][1:3]
    messages, removed = strip_unsupported_tool_search_blocks(transcript, [])
    assert removed == 2
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_repair_leaves_other_server_tools_alone() -> None:
    # web_search / code execution use the same block type and stay untouched.
    transcript = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_web",
                    "name": "web_search",
                    "input": {"query": "x"},
                },
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_web", "content": []},
            ],
        }
    ]
    messages, removed = strip_unsupported_tool_search_blocks(transcript, [])
    assert removed == 0
    assert messages is transcript


def test_repair_is_idempotent() -> None:
    # Deterministic: repairing an already-repaired transcript is a no-op, so a
    # session's forwarded prefix stays byte-stable turn over turn.
    once, _ = strip_unsupported_tool_search_blocks(_poisoned_transcript(), [])
    twice, removed = strip_unsupported_tool_search_blocks(once, [])
    assert removed == 0
    assert twice is once


def test_repair_strips_search_history_when_only_the_tool_is_missing() -> None:
    # References all resolve, but the request has no tool_search tool at all
    # (e.g. deferral skipped below _TOOL_SEARCH_MIN_TOOLS) -- history still
    # cannot be supported, so it goes.
    _, removed = strip_unsupported_tool_search_blocks(
        _poisoned_transcript(), [{"name": "AskUserQuestion", "input_schema": {}}]
    )
    assert removed == 2


# ---------------------------------------------------------------------------
# Regression tests for the direct-Anthropic regression reported in PR #2539
# comment #5280259642: "Tool reference 'tool_search_tool_regex' not found in
# available tools".
#
# Root cause: when a client sends ``tool_search_tool_regex`` as a *typeless*
# tool, ``inject_tool_search_deferral`` would (a) not early-exit because the
# guard only checked ``type``, and (b) defer the tool.  Anthropic then found
# the deferred copy via the server-side search and stored the tool's name in a
# ``tool_reference`` entry.  On subsequent requests where the typed injected
# search tool was present, ``strip_unsupported_tool_search_blocks`` incorrectly
# treated the injected search tool's *name* as proof the reference was
# resolvable — but the typed server tool is not a valid deferred-tool target, so
# Anthropic rejected the request with 400.
#
# The two-part fix:
#   1. ``inject_tool_search_deferral`` early-exit also fires on a name-prefix
#      match, preventing double-injection when the client carries a typeless
#      ``tool_search_tool_*`` entry.
#   2. ``strip_unsupported_tool_search_blocks`` excludes typed search tools
#      from the ``available`` set — they are the search mechanism, not targets.
# ---------------------------------------------------------------------------


def _transcript_with_search_tool_regex_reference() -> list[dict]:
    """Transcript where the search found 'tool_search_tool_regex' itself.

    This happens when inject_tool_search_deferral defers a typeless client tool
    named 'tool_search_tool_regex': Anthropic finds it and stores it as a
    tool_reference.  On subsequent requests the repair must drop the block
    rather than falsely keep it because the typed injected search-tool shares
    the same name.
    """
    return [
        {"role": "user", "content": [{"type": "text", "text": "search for a tool"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_REGEX",
                    "name": _TOOL_SEARCH_DEFAULT_NAME,
                    "input": {"pattern": "regex"},
                },
                {
                    "type": "tool_search_tool_result",
                    "tool_use_id": "srvtoolu_REGEX",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [
                            {
                                "type": "tool_reference",
                                # The search found the deferred 'tool_search_tool_regex'
                                # typeless tool — this is the broken reference.
                                "tool_name": _TOOL_SEARCH_DEFAULT_NAME,
                            }
                        ],
                    },
                },
            ],
        },
    ]


@pytest.mark.parametrize(
    "name",
    [_TOOL_SEARCH_DEFAULT_NAME, "TOOL_SEARCH_TOOL_BM25"],
)
def test_inject_deferral_exits_early_on_typeless_tool_search_name(name: str) -> None:
    # A client that sends tool_search_tool_regex without a ``type`` field should
    # be treated as already using tool search (name-prefix guard), so Headroom
    # must not inject a second search tool on top of it.
    typeless_search = {"name": name, "input_schema": {}}
    tools = _tools(20) + [typeless_search]
    result = inject_tool_search_deferral(tools)
    assert result is tools  # no injection


def test_inject_deferral_does_not_false_match_similar_typeless_tool_name() -> None:
    # Keep ordinary tools whose names merely resemble the reserved prefix on the
    # normal deferral path; the trailing underscore is part of the match.
    tools = _tools(20) + [{"name": "tool_search_toolbox", "input_schema": {}}]
    result = inject_tool_search_deferral(tools)
    assert result is not tools
    by_name = {tool.get("name"): tool for tool in result}
    assert by_name["tool_search_toolbox"]["defer_loading"] is True


def test_repair_drops_search_tool_self_reference_when_inject_ran() -> None:
    # Regression for PR #2539 comment #5280259642.
    #
    # Scenario: inject ran on a previous turn (has_search_tool=True because the
    # typed search tool is present), but the transcript's tool_reference names
    # 'tool_search_tool_regex' — the search tool itself.  The typed injected
    # search tool must NOT count as a valid reference target; the block must be
    # dropped so Anthropic never sees an unresolvable tool_reference.
    transcript = _transcript_with_search_tool_regex_reference()
    # tools array after inject: typed search tool + regular deferred tools
    tools = [
        _SEARCH_TOOL,  # typed search tool — must NOT be in 'available'
        {"name": "Bash", "input_schema": {}},
        {"name": "mcp_tool_x", "input_schema": {}, "defer_loading": True},
    ]
    messages, removed = strip_unsupported_tool_search_blocks(transcript, tools)
    assert removed == 2  # server_tool_use + tool_search_tool_result both dropped
    # The assistant turn is entirely stripped (only search blocks were present).
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_repair_noop_when_referenced_tool_is_regular_deferred_tool() -> None:
    # Baseline: when the transcript references a normal deferred tool (not the
    # search tool itself) and that tool is in the current tools array, the block
    # must be kept — no false-positive stripping from the typed-search exclusion.
    transcript = _poisoned_transcript()  # references "AskUserQuestion"
    tools = [
        _SEARCH_TOOL,
        {"name": "AskUserQuestion", "input_schema": {}, "defer_loading": True},
    ]
    messages, removed = strip_unsupported_tool_search_blocks(transcript, tools)
    assert removed == 0
    assert messages is transcript


def test_repair_drops_when_referenced_tool_absent_despite_search_tool_present() -> None:
    # The referenced tool is NOT in the current tools array even though the
    # typed search tool is present (e.g. a compact request with a different tool
    # subset).  The block must be dropped.
    transcript = _poisoned_transcript()  # references "AskUserQuestion"
    tools = [
        _SEARCH_TOOL,
        {"name": "Bash", "input_schema": {}},
        # AskUserQuestion intentionally absent
    ]
    messages, removed = strip_unsupported_tool_search_blocks(transcript, tools)
    assert removed == 2
