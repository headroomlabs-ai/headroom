"""Who defers tool schemas, and what Headroom must do about it.

Two mechanisms exist at two layers:

* **Server-side (Messages API / Responses).** ``tool_search_tool_regex`` /
  ``tool_search_tool_bm25`` types paired with ``defer_loading: true``, or
  OpenAI's ``{"type": "tool_search"}``. This shape appears ONLY when deferral
  is actually on, so it is an unambiguous signal. It has no single spelling:
  a survey of shipped harnesses found tool search of their own in Codex
  (``tool_search``, a type with no ``name`` at all), GitHub Copilot CLI
  (``tool_search_tool``), VS Code Copilot and Kiro (``tool_search``) and
  Codebuff (``composio_search_tools``) — so the client already deferring is
  the common case, not the exception, and ``defer_loading`` on any tool is
  the one signal that needs no agreement about names.
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

import logging
from typing import Any

import pytest

from headroom.proxy.helpers import (
    _TOOL_SEARCH_MIN_TOOLS,
    claude_code_tool_search_inactive,
    inject_tool_search_deferral,
    inject_tool_search_deferral_openai,
    iter_tool_entries,
    request_already_defers_tools,
    reset_deferred_orphan_warn_state,
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
        pytest.param([{"type": "web_search"}], id="unrelated-server-tool"),
        pytest.param(["not-a-dict", 7], id="non-dict-entries"),
    ],
)
def test_not_a_deferral_signal(tools: Any) -> None:
    assert request_already_defers_tools(tools) is False


# ---------------------------------------------------------------------------
# Deferral is not an Anthropic-only feature. A survey of shipped harnesses
# found their own tool search in Codex, GitHub Copilot CLI, VS Code Copilot,
# Kiro and Codebuff, each with a different wire spelling. Matching only
# Anthropic's versioned ``tool_search_tool_*`` prefix stood down for none of
# them, so Headroom deferred on top of a client that was already deferring --
# which suppresses the client's mechanism and inlines the very catalog we were
# keeping out.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    [
        # Codex: codex-rs/tools/src/tool_spec.rs. A union variant carrying no
        # ``name`` field at all, so a name-only scan can never see it.
        pytest.param({"type": "tool_search"}, id="codex-responses-bare-type"),
        # VS Code Copilot: ToolName.tool_search; Kiro CLI uses the same name.
        pytest.param({"name": "tool_search"}, id="vscode-copilot-kiro-name"),
        # GitHub Copilot CLI: the exact string ``tool_search_tool``, which does
        # NOT start with ``tool_search_tool_`` -- the off-by-one that let the
        # prefix test miss a shipped harness.
        pytest.param({"name": "tool_search_tool"}, id="copilot-cli-exact"),
        pytest.param({"name": "tool_search_tool_regex"}, id="copilot-cli-regex"),
        # Codebuff delegates to an external catalog under its own name.
        pytest.param({"name": "composio_search_tools"}, id="codebuff"),
    ],
)
def test_every_known_meta_tool_spelling_means_the_client_defers(marker: dict[str, Any]) -> None:
    assert request_already_defers_tools([marker, *_mcp(3)]) is True


def test_defer_loading_alone_means_the_client_defers() -> None:
    """Shape-independent: the client marking its own catalog deferred is enough.

    Worth matching on its own because it needs no agreement about what the
    meta-tool is called, so it still holds for a harness this survey missed.
    """
    tools = [{"name": "mcp__github__list_issues", "defer_loading": True}, *_mcp(3)]

    assert request_already_defers_tools(tools) is True


def test_defer_loading_false_is_not_a_signal() -> None:
    """Present-and-false is a client that considered deferral and declined."""
    assert request_already_defers_tools([{"name": "read", "defer_loading": False}]) is False


# --- namespace nesting -----------------------------------------------------


def _namespace(name: str, *tools: dict[str, Any]) -> dict[str, Any]:
    """A Responses namespace group, the shape Codex ships MCP servers in."""
    return {"type": "namespace", "name": name, "tools": list(tools)}


def test_a_meta_tool_nested_in_a_namespace_is_found() -> None:
    """The wrapper is visible to a top-level scan; its contents are not."""
    tools = [_namespace("functions", {"type": "function", "name": "tool_search_tool"})]

    assert request_already_defers_tools(tools) is True


def test_defer_loading_nested_in_a_namespace_is_found() -> None:
    tools = [_namespace("github", {"type": "function", "name": "x", "defer_loading": True})]

    assert request_already_defers_tools(tools) is True


def test_a_plain_namespace_is_not_a_deferral_signal() -> None:
    """Grouping tools is not deferring them."""
    tools = [_namespace("github", {"type": "function", "name": "list_issues"})]

    assert request_already_defers_tools(tools) is False


def test_iter_tool_entries_yields_the_wrapper_and_its_members() -> None:
    nested = {"type": "function", "name": "inner"}
    group = _namespace("ns", nested)

    names = [t.get("name") for t in iter_tool_entries([{"name": "top"}, group])]

    assert names == ["top", "ns", "inner"]


@pytest.mark.parametrize(
    "tools",
    [
        pytest.param(None, id="not-a-list"),
        pytest.param([{"type": "namespace"}], id="namespace-without-tools"),
        pytest.param([{"type": "namespace", "tools": "nope"}], id="tools-not-a-list"),
        pytest.param([{"type": "namespace", "tools": [7, "x"]}], id="non-dict-members"),
    ],
)
def test_iter_tool_entries_tolerates_malformed_input(tools: Any) -> None:
    """A malformed namespace must not raise on the request path."""
    assert all(isinstance(t, dict) for t in iter_tool_entries(tools))


def test_nesting_does_not_recurse_past_one_level() -> None:
    """One level is what the API defines; deeper input is not trusted."""
    deep = _namespace("outer", _namespace("inner", {"name": "buried"}))

    assert "buried" not in [t.get("name") for t in iter_tool_entries([deep])]


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

    assert resolved_core_tools() == frozenset({"bash", "read", "toolsearch"})

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


def test_openai_counts_namespace_members_toward_the_threshold() -> None:
    """A big catalog grouped under one namespace is still a big catalog.

    Counting only top-level entries reads a 20-tool MCP server as a single
    tool and skips the request, which is precisely the request with the most
    schema to save.
    """
    top_level = [_fn(f"jira_{i}") for i in range(4)]
    tools = [
        *top_level,
        {"type": "namespace", "name": "slack", "tools": [_fn(f"slack_{i}") for i in range(20)]},
    ]

    assert len(tools) < _TOOL_SEARCH_MIN_TOOLS  # a top-level count would bail here
    out = inject_tool_search_deferral_openai(tools, "gpt-5.5")

    # The namespace itself stays untouched; the top-level tools it made us
    # look at in the first place are what get deferred.
    assert [t["name"] for t in _deferred(out)] == [t["name"] for t in top_level]


def test_openai_leaves_namespace_groups_resident() -> None:
    """We do not guess a wire shape for deferring inside a namespace.

    Marking nested members ``defer_loading`` is unverified against the API, and
    an unknown field in the wrong place is a 400. Counting them is safe;
    rewriting them is not, so the group is forwarded untouched.
    """
    group = {"type": "namespace", "name": "slack", "tools": [_fn(f"slack_{i}") for i in range(20)]}
    out = inject_tool_search_deferral_openai([_fn("bash"), group], "gpt-5.5")

    assert group in out
    assert _deferred(out) == []


def test_openai_toolsearch_name_alone_does_not_stand_us_down() -> None:
    tools = [_fn("ToolSearch"), *[_fn(f"slack_{i}") for i in range(14)]]

    out = inject_tool_search_deferral_openai(tools, "gpt-5.5")

    by_name = {t.get("name"): t for t in out if isinstance(t, dict)}
    assert by_name["ToolSearch"].get("defer_loading") is None
    assert by_name["slack_0"].get("defer_loading") is True


# --------------------------------------------------------------------------
# the resident override must mean the same thing on both paths
# --------------------------------------------------------------------------


def test_openai_keeps_terminal_resident_by_default() -> None:
    tools = [_fn("terminal"), _fn("bash"), *[_fn(f"slack_{i}") for i in range(14)]]

    by_name = {
        t.get("name"): t
        for t in inject_tool_search_deferral_openai(tools, "gpt-5.5")
        if isinstance(t, dict)
    }

    assert by_name["terminal"].get("defer_loading") is None
    assert by_name["bash"].get("defer_loading") is None
    assert by_name["slack_0"].get("defer_loading") is True


def test_openai_override_can_drop_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit core set is authoritative — provider additions included.

    ``terminal`` used to be unioned in unconditionally, so setting the knob to a
    set without it still pinned it resident. The two provider paths then honoured
    the same variable differently, and an operator asking to defer a tool was
    silently refused.
    """
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH_CORE_TOOLS", "bash,read")
    tools = [_fn("terminal"), _fn("bash"), _fn("grep"), *[_fn(f"slack_{i}") for i in range(14)]]

    by_name = {
        t.get("name"): t
        for t in inject_tool_search_deferral_openai(tools, "gpt-5.5")
        if isinstance(t, dict)
    }

    assert by_name["bash"].get("defer_loading") is None
    assert by_name["terminal"].get("defer_loading") is True
    assert by_name["grep"].get("defer_loading") is True


def test_default_resident_set_is_unchanged_for_anthropic() -> None:
    """The OpenAI addition must not leak into the Anthropic default."""
    assert "terminal" not in resolved_core_tools()
    assert "terminal" in resolved_core_tools(frozenset({"terminal"}))


# --------------------------------------------------------------------------
# the tool-search-disabled warning must not go quiet forever
# --------------------------------------------------------------------------


def test_hint_rearms_after_the_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """One line per process is not proportionate to a permanent condition.

    The warning reports a deployment that has its client's tool deferral turned
    off — which costs tokens on every request for the life of the deployment.
    Firing once and never again means a proxy up for weeks says it at startup
    and is silent through everything after.
    """
    from headroom.proxy import helpers as H

    H.reset_tool_search_hint_state()
    try:
        clock = {"t": 1000.0}
        monkeypatch.setattr(H, "_monotonic", lambda: clock["t"])

        assert H.take_tool_search_scan_slot() is True
        assert H.take_tool_search_scan_slot() is False

        clock["t"] += H._TOOL_SEARCH_HINT_INTERVAL_S - 1
        assert H.take_tool_search_scan_slot() is False, "must not re-arm early"

        clock["t"] += 2
        assert H.take_tool_search_scan_slot() is True, "must re-arm after the interval"
    finally:
        H.reset_tool_search_hint_state()


# --------------------------------------------------------------------------
# one knob, two spellings
# --------------------------------------------------------------------------


def test_legacy_env_var_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plugin shipped HEADROOM_TOOL_SEARCH_CORE for the same idea.

    Two variables for one knob means an operator sets the one they know and the
    other path silently keeps its own list, so the two providers disagree about
    which tools are visible. Either spelling now works everywhere.
    """
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH_CORE", "bash,read")

    assert resolved_core_tools() == frozenset({"bash", "read", "toolsearch"})


def test_canonical_env_var_wins_over_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH_CORE", "bash,read")
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH_CORE_TOOLS", "grep")

    assert resolved_core_tools() == frozenset({"grep", "toolsearch"})


def test_neither_set_keeps_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_TOOL_SEARCH_CORE", raising=False)
    monkeypatch.delenv("HEADROOM_TOOL_SEARCH_CORE_TOOLS", raising=False)

    assert "bash" in resolved_core_tools()
    assert "read" in resolved_core_tools()


def test_empty_override_defers_everything_non_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit empty set is a real instruction, not an unset variable."""
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH_CORE_TOOLS", "")

    assert resolved_core_tools() == frozenset({"toolsearch"})


def test_the_override_tolerates_spaces_after_commas(monkeypatch: pytest.MonkeyPatch) -> None:
    """The natural spelling of a list must not defer the tools it names.

    The key function lowercases and strips leading underscores but not spaces,
    so ``"bash, read, terminal"`` resolved to ``{" read", " terminal", "bash"}``
    and deferred exactly the two tools the operator asked to keep resident.
    """
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH_CORE_TOOLS", "bash, read , terminal")

    assert resolved_core_tools() == frozenset({"bash", "read", "terminal", "toolsearch"})


def test_an_override_cannot_defer_the_clients_search_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferring ToolSearch hides the only thing that can load what it resolves.

    Claude Code reaches tools held in a local registry through it, and nothing
    else can. An override names which ORDINARY tools stay inline, so honouring
    one that omits this would orphan a whole category rather than defer it.
    """
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH_CORE_TOOLS", "Bash,Read")
    tools = [CLAUDE_CODE_TOOL_SEARCH, {"name": "Bash", "input_schema": {}}, *_mcp(14)]

    by_name = {t.get("name"): t for t in inject_tool_search_deferral(tools) if isinstance(t, dict)}

    assert by_name["ToolSearch"].get("defer_loading") is None
    assert by_name["mcp__srv0__do"].get("defer_loading") is True


def test_the_scan_gate_closes_even_when_the_hint_does_not_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the operator FIXES the condition, the scan must stop running.

    Stamping the window only on emission meant the gate stayed open forever
    after the fix, so every later request paid the full O(tools) scan for the
    life of the process -- worst on the large tool surfaces this targets.
    """
    from headroom.proxy import helpers as H

    H.reset_tool_search_hint_state()
    try:
        clock = {"t": 1000.0}
        monkeypatch.setattr(H, "_monotonic", lambda: clock["t"])

        assert H.take_tool_search_scan_slot() is True  # scanned; found nothing
        assert H.tool_search_hint_pending() is False  # ...and the gate closed anyway
    finally:
        H.reset_tool_search_hint_state()


# --- orphaned deferrals ----------------------------------------------------


def test_deferred_tools_with_no_search_tool_warn(caplog: pytest.LogCaptureFixture) -> None:
    """Standing down here is right but ambiguous, so it must not be silent.

    Either the harness spells its search tool in a way we do not know, or an
    intermediary stripped the search tool and left the marks behind. The first
    is benign and the second is fatal upstream, and only the operator can tell
    them apart -- so name the tools and say what to do.
    """
    reset_deferred_orphan_warn_state()
    tools = [{"name": "mcp__github__list_issues", "defer_loading": True}, *_mcp(3)]

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        assert request_already_defers_tools(tools) is True

    lines = [
        r.getMessage() for r in caplog.records if "tool_search_deferred_orphan" in r.getMessage()
    ]
    assert len(lines) == 1
    assert "mcp__github__list_issues" in lines[0]
    assert "HEADROOM_CLIENT_TOOL_SEARCH_NAMES" in lines[0]


def test_a_recognized_search_tool_is_not_an_orphan(caplog: pytest.LogCaptureFixture) -> None:
    """The normal deferring client must stay quiet."""
    reset_deferred_orphan_warn_state()
    tools = [ANTHROPIC_SEARCH_REGEX, {"name": "mcp__github__x", "defer_loading": True}]

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        assert request_already_defers_tools(tools) is True

    assert [r for r in caplog.records if "tool_search_deferred_orphan" in r.getMessage()] == []


def test_the_orphan_warning_is_throttled(caplog: pytest.LogCaptureFixture) -> None:
    """It is evaluated per request; one line per request would be a flood."""
    reset_deferred_orphan_warn_state()
    tools = [{"name": "x", "defer_loading": True}]

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        for _ in range(50):
            request_already_defers_tools(tools)

    lines = [r for r in caplog.records if "tool_search_deferred_orphan" in r.getMessage()]
    assert len(lines) == 1
