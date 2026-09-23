"""Who defers tool schemas, and what Headroom must do about it.

Two mechanisms exist at two layers:

* **Server-side (Messages API / Responses).** ``tool_search_tool_regex`` /
  ``tool_search_tool_bm25`` types paired with ``defer_loading: true``, or
  OpenAI's ``{"type": "tool_search"}``. This shape appears ONLY when deferral
  is actually on, so it is an unambiguous signal.
* **Client-side.** Claude Code carries a tool named ``ToolSearch``. It resolves
  tools the client keeps in a local registry (TaskCreate, WebFetch, ...) and
  real traffic shows it riding alongside a fully inline MCP catalog. Its
  presence therefore says nothing about whether MCP schemas were deferred.

The distinction is load-bearing in both directions:

* Deferring on top of a client that is already deferring suppresses the
  client's mechanism and inlines the catalog we were trying to keep out.
* Standing down on the mere NAME ``ToolSearch`` would disable Headroom exactly
  when the client is sending everything eagerly — which through Kong, Bedrock
  or any custom base URL is the normal case, because Claude Code disables its
  own tool search when ``ANTHROPIC_BASE_URL`` is non-first-party.

So: key on the server-side shape, never on the bare client-side name.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.proxy.helpers import (
    _TOOL_SEARCH_MIN_TOOLS,
    claude_code_tool_search_inactive,
    inject_tool_search_deferral,
    inject_tool_search_deferral_openai,
    request_already_defers_tools,
    resolved_core_tools,
    strip_first_party_tool_search_tools_for_third_party_upstream,
)

CLAUDE_CODE_TOOL_SEARCH: dict[str, Any] = {
    "name": "ToolSearch",
    "description": "Search for tools by name or description.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}
ANTHROPIC_SEARCH_REGEX: dict[str, Any] = {
    "type": "tool_search_tool_regex_20251119",
    "name": "tool_search_tool_regex",
}
ANTHROPIC_SEARCH_BM25: dict[str, Any] = {
    "type": "tool_search_tool_bm25_20251119",
    "name": "tool_search_tool_bm25",
}
THIRD_PARTY_URL = "https://kong.internal/anthropic"


def _mcp(n: int) -> list[dict[str, Any]]:
    return [
        {"name": f"mcp__srv{i}__do", "description": "x" * 400, "input_schema": {"type": "object"}}
        for i in range(n)
    ]


def _core() -> list[dict[str, Any]]:
    return [
        {"name": n, "description": "core", "input_schema": {"type": "object"}}
        for n in sorted(resolved_core_tools())
    ]


def _searches(tools: Any) -> list[Any]:
    return [
        t
        for t in tools
        if isinstance(t, dict) and str(t.get("type", "")).startswith("tool_search_tool_")
    ]


def _deferred(tools: Any) -> list[Any]:
    return [t for t in tools if isinstance(t, dict) and t.get("defer_loading")]


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    [
        pytest.param(ANTHROPIC_SEARCH_REGEX, id="regex-type"),
        pytest.param(ANTHROPIC_SEARCH_BM25, id="bm25-type"),
        pytest.param({"name": "tool_search_tool_regex"}, id="by-name"),
        pytest.param({"name": "_tool_search_tool_regex"}, id="namespaced-name"),
        pytest.param({"type": "tool_search_tool_regex"}, id="undated-alias"),
    ],
)
def test_server_side_shape_means_the_client_defers(marker: dict[str, Any]) -> None:
    assert request_already_defers_tools([marker, *_mcp(3)]) is True


@pytest.mark.parametrize(
    "tools",
    [
        # The correction: a bare client-side name is NOT a deferral signal.
        pytest.param([CLAUDE_CODE_TOOL_SEARCH, *_mcp(3)], id="claude-code-ToolSearch"),
        pytest.param([{"name": "toolsearch"}], id="lowercase-toolsearch"),
        pytest.param([], id="empty"),
        pytest.param(None, id="not-a-list"),
        pytest.param([{"name": "mcp__search__tools"}], id="mcp-tool-named-search"),
        pytest.param([{"name": "SearchTool"}], id="reversed-words"),
        pytest.param([{"name": "tool_search"}], id="prefix-but-incomplete"),
        pytest.param([{"type": "web_search"}], id="unrelated-server-tool"),
        pytest.param(["not-a-dict", 7], id="non-dict-entries"),
    ],
)
def test_not_a_deferral_signal(tools: Any) -> None:
    assert request_already_defers_tools(tools) is False


def test_extra_names_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Escape hatch for a harness that does signal deferral by name."""
    tools = [{"name": "MyHarnessSearch"}, *_mcp(3)]
    assert request_already_defers_tools(tools) is False

    monkeypatch.setenv("HEADROOM_CLIENT_TOOL_SEARCH_NAMES", "myharnesssearch")
    assert request_already_defers_tools(tools) is True


# --------------------------------------------------------------------------
# Anthropic injection
# --------------------------------------------------------------------------


def test_stands_down_when_the_client_already_defers() -> None:
    tools = [ANTHROPIC_SEARCH_REGEX, *_core(), *_mcp(20)]

    out = inject_tool_search_deferral(tools)

    assert out is tools, "tools prefix must be byte-identical when we stand down"
    assert _deferred(out) == []


def test_defers_when_the_client_is_eager() -> None:
    """The Kong / Bedrock / custom-base-URL case: nobody else is deferring."""
    tools = [*_core(), *_mcp(20)]

    out = inject_tool_search_deferral(tools)

    assert out is not tools
    assert len(_searches(out)) == 1
    assert len(_deferred(out)) == 20


def test_toolsearch_alongside_an_inline_catalog_still_defers() -> None:
    """The case that made me get this wrong the first time.

    ``ToolSearch`` present AND the catalog inline means the client is NOT
    deferring MCP schemas. Standing down here would leave every schema in
    context, which is the opposite of what the feature is for.
    """
    tools = [CLAUDE_CODE_TOOL_SEARCH, *_core(), *_mcp(20)]

    out = inject_tool_search_deferral(tools)

    assert len(_deferred(out)) == 20
    by_name = {t.get("name"): t for t in out if isinstance(t, dict)}
    assert by_name["ToolSearch"].get("defer_loading") is None, (
        "the client's own search tool must stay resident or its local-registry "
        "tools become permanently unreachable"
    )


def test_core_tools_are_never_deferred() -> None:
    tools = [*_core(), *_mcp(20)]

    deferred_names = {t["name"] for t in _deferred(inject_tool_search_deferral(tools))}

    assert deferred_names.isdisjoint({t["name"] for t in _core()})


def test_resident_set_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """So deferring built-ins can be measured instead of guessed."""
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH_CORE_TOOLS", "bash,read")

    assert resolved_core_tools() == frozenset({"bash", "read"})

    tools = [
        {"name": "bash", "input_schema": {}},
        {"name": "read", "input_schema": {}},
        {"name": "grep", "input_schema": {}},
        *_mcp(15),
    ]
    by_name = {t.get("name"): t for t in inject_tool_search_deferral(tools) if isinstance(t, dict)}

    assert by_name["bash"].get("defer_loading") is None
    assert by_name["grep"].get("defer_loading") is True


def test_below_minimum_tool_count_is_left_alone() -> None:
    tools = _mcp(_TOOL_SEARCH_MIN_TOOLS - 1)

    assert inject_tool_search_deferral(tools) is tools


# --------------------------------------------------------------------------
# the issue-746 hint
# --------------------------------------------------------------------------


def test_hint_fires_for_an_eager_claude_code() -> None:
    assert (
        claude_code_tool_search_inactive(
            client="claude-code", tools=[*_core(), *_mcp(20)], anthropic_beta=None
        )
        is True
    )


def test_hint_is_silent_when_the_client_defers() -> None:
    assert (
        claude_code_tool_search_inactive(
            client="claude-code", tools=[ANTHROPIC_SEARCH_REGEX, *_mcp(5)], anthropic_beta=None
        )
        is False
    )


def test_hint_is_silent_with_a_tool_search_beta_marker() -> None:
    assert (
        claude_code_tool_search_inactive(
            client="claude-code",
            tools=[*_core(), *_mcp(20)],
            anthropic_beta="advanced-tool-use-2025-11-20",
        )
        is False
    )


# --------------------------------------------------------------------------
# third-party upstreams: Kong / Bedrock / Vertex
# --------------------------------------------------------------------------


def test_stripping_the_search_tool_also_clears_defer_loading() -> None:
    """Half-stripping leaves tools that nothing can resolve.

    Bedrock needs a different beta token and InvokeModel rather than Converse;
    Vertex and gateways reject the first-party type outright. Removing the
    search tool is therefore right — but leaving ``defer_loading`` behind means
    the model has no mechanism to load those tools, and a ``tool_reference``
    naming one is a documented 400.
    """
    tools = [
        ANTHROPIC_SEARCH_REGEX,
        {"name": "Bash", "input_schema": {}},
        {"name": "mcp__srv__a", "input_schema": {}, "defer_loading": True},
        {"name": "mcp__srv__b", "input_schema": {}, "defer_loading": True},
    ]

    out = strip_first_party_tool_search_tools_for_third_party_upstream(tools, THIRD_PARTY_URL)

    assert _searches(out) == []
    assert _deferred(out) == []
    # The tools themselves survive — only the deferral machinery is removed.
    assert {t["name"] for t in out} == {"Bash", "mcp__srv__a", "mcp__srv__b"}


def test_stripping_preserves_everything_else_about_a_tool() -> None:
    tools = [
        ANTHROPIC_SEARCH_REGEX,
        {"name": "mcp__srv__a", "input_schema": {"type": "object"}, "defer_loading": True},
    ]

    out = strip_first_party_tool_search_tools_for_third_party_upstream(tools, THIRD_PARTY_URL)

    assert out[0] == {"name": "mcp__srv__a", "input_schema": {"type": "object"}}


def test_first_party_upstream_is_untouched() -> None:
    tools = [ANTHROPIC_SEARCH_REGEX, {"name": "a", "defer_loading": True}]

    out = strip_first_party_tool_search_tools_for_third_party_upstream(
        tools, "https://api.anthropic.com"
    )

    assert out is tools


def test_nothing_to_strip_returns_the_same_list() -> None:
    tools = [{"name": "Bash", "input_schema": {}}, *_mcp(3)]

    out = strip_first_party_tool_search_tools_for_third_party_upstream(tools, THIRD_PARTY_URL)

    assert out is tools


# --------------------------------------------------------------------------
# OpenAI Responses path
# --------------------------------------------------------------------------


def _fn(name: str) -> dict[str, Any]:
    return {"type": "function", "name": name, "parameters": {"type": "object", "properties": {}}}


def test_openai_stands_down_on_its_own_search_tool() -> None:
    tools = [{"type": "tool_search"}, *[_fn(f"slack_{i}") for i in range(14)]]

    assert inject_tool_search_deferral_openai(tools, "gpt-5.5") is tools


def test_openai_stands_down_on_an_anthropic_shaped_search_tool() -> None:
    """A mixed harness can carry the Messages API shape onto the OpenAI path."""
    tools = [ANTHROPIC_SEARCH_REGEX, *[_fn(f"slack_{i}") for i in range(14)]]

    assert inject_tool_search_deferral_openai(tools, "gpt-5.5") is tools


def test_openai_still_defers_for_an_eager_client() -> None:
    tools = [_fn("bash"), *[_fn(f"slack_{i}") for i in range(14)]]

    out = inject_tool_search_deferral_openai(tools, "gpt-5.5")

    assert out is not tools
    assert len(_deferred(out)) == 14


def test_openai_toolsearch_name_alone_does_not_stand_us_down() -> None:
    tools = [_fn("ToolSearch"), *[_fn(f"slack_{i}") for i in range(14)]]

    out = inject_tool_search_deferral_openai(tools, "gpt-5.5")

    by_name = {t.get("name"): t for t in out if isinstance(t, dict)}
    assert by_name["ToolSearch"].get("defer_loading") is None
    assert by_name["slack_0"].get("defer_loading") is True
