"""Server-side Tool Search deferral for OpenAI Responses (gpt-5.4+).

The OpenAI-side analogue of the Anthropic path (issue #746): mark non-core
function / MCP tools ``defer_loading: true`` and inject ``{"type": "tool_search"}``
so OpenAI keeps their heavy parameter schemas out of the model's context until
searched. Explicitly opt-in via HEADROOM_OPENAI_TOOL_SEARCH or
HEADROOM_OPENAI_TOOL_SEARCH_MODELS; older models 400 on the fields.
"""

from __future__ import annotations

import copy

import pytest

from headroom.proxy.helpers import (
    _model_supports_openai_tool_search,
    inject_tool_search_deferral_openai,
    openai_tool_search_client_supported,
)


def _fn(name: str) -> dict:
    return {"type": "function", "name": name, "parameters": {"type": "object", "properties": {}}}


_CORE = ["bash", "read", "write", "edit", "grep", "glob"]
_NONCORE = [f"slack_{i}" for i in range(10)]  # 6 core + 10 non-core = 16 tools (>= min 12)


def _tools() -> list[dict]:
    return [_fn(n) for n in _CORE + _NONCORE]


# --- model gating ------------------------------------------------------------


def test_model_gate_is_off_by_default(monkeypatch):
    monkeypatch.delenv("HEADROOM_OPENAI_TOOL_SEARCH", raising=False)
    monkeypatch.delenv("HEADROOM_OPENAI_TOOL_SEARCH_MODELS", raising=False)
    assert _model_supports_openai_tool_search("gpt-5.5") is False


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.5", "gpt-5.4-2026-02-01", "gpt-6", "gpt-6.2"])
def test_model_supported_when_enabled(monkeypatch, model):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    assert _model_supports_openai_tool_search(model) is True


@pytest.mark.parametrize(
    "model", ["gpt-4o", "gpt-4.1", "gpt-5", "gpt-5.3", "o3", "", None, "claude-opus-4-8"]
)
def test_model_unsupported_when_enabled(monkeypatch, model):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    assert _model_supports_openai_tool_search(model) is False


def test_env_override_wins_then_falls_back(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH_MODELS", r"^my-model")
    assert _model_supports_openai_tool_search("my-model-v1") is True
    assert _model_supports_openai_tool_search("gpt-5.4") is False  # override replaces the gate
    # a malformed regex must not crash or silently enable a rewrite.
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH_MODELS", "[unclosed")
    assert _model_supports_openai_tool_search("gpt-5.4") is False


# --- deferral behavior -------------------------------------------------------


@pytest.mark.parametrize(
    ("client", "supported"),
    [(None, True), ("codex", False), (" CODEX ", False), ("opencode", False), ("claude", True)],
)
def test_client_supported(client, supported):
    assert openai_tool_search_client_supported(client) is supported


def test_codex_client_does_not_inject(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    tools = _tools()

    out = inject_tool_search_deferral_openai(tools, "gpt-5.5", client="codex")

    assert out is tools
    assert all(tool.get("type") != "tool_search" for tool in out)
    assert all("defer_loading" not in tool for tool in out)


@pytest.mark.parametrize("client", [None, "claude-code"])
def test_supported_clients_still_inject(monkeypatch, client):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    tools = _tools()

    out = inject_tool_search_deferral_openai(tools, "gpt-5.5", client=client)

    assert out is not tools
    assert out[0] == {"type": "tool_search"}
    assert any(tool.get("defer_loading") is True for tool in out)


def test_defers_non_core_and_injects_search_tool(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    tools = _tools()
    out = inject_tool_search_deferral_openai(tools, "gpt-5.5")
    assert out is not tools  # new list
    assert out[0] == {"type": "tool_search"}  # search tool injected, first, once
    assert sum(1 for t in out if t.get("type") == "tool_search") == 1
    by_name = {t["name"]: t for t in out if t.get("type") == "function"}
    for c in _CORE:
        assert not by_name[c].get("defer_loading")  # core stays resident
    for n in _NONCORE:
        assert by_name[n].get("defer_loading") is True  # non-core deferred


def test_terminal_reserved_namespace_stays_resident(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    terminal = _fn("terminal")
    tools = [terminal] + [_fn(f"peer_{i}") for i in range(11)]
    snapshot = copy.deepcopy(tools)

    out = inject_tool_search_deferral_openai(tools, "gpt-5.6-terra")

    forwarded = next(t for t in out if t.get("name") == "terminal")
    assert forwarded == terminal
    assert "defer_loading" not in forwarded
    assert next(t for t in out if t.get("name") == "peer_0").get("defer_loading") is True
    assert tools == snapshot


def test_terminal_helper_remains_deferrable(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    tools = [_fn("terminal_helper")] + [_fn(f"peer_{i}") for i in range(11)]

    out = inject_tool_search_deferral_openai(tools, "gpt-5.6-terra")

    helper = next(t for t in out if t.get("name") == "terminal_helper")
    assert helper.get("defer_loading") is True


def test_prefixed_core_and_terminal_names_stay_resident(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    resident = ["_bash", "_read", "_write", "_edit", "_glob", "_grep", "_terminal"]
    noncore = ["_hub", "_todo", "_eval", "mcp__server__read", "terminal_helper"]
    tools = [_fn(name) for name in resident + noncore]

    out = inject_tool_search_deferral_openai(tools, "gpt-5.6-terra")

    by_name = {tool["name"]: tool for tool in out if tool.get("type") == "function"}
    for name in resident:
        assert by_name[name].get("defer_loading") is None, name
    for name in noncore:
        assert by_name[name].get("defer_loading") is True, name


def test_defers_mcp_server(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    tools = [_fn(n) for n in _CORE] + [{"type": "mcp", "server_label": "sentry"}]
    tools += [_fn(f"x{i}") for i in range(8)]
    out = inject_tool_search_deferral_openai(tools, "gpt-5.5")
    mcp = next(t for t in out if t.get("type") == "mcp")
    assert mcp.get("defer_loading") is True


def test_hosted_tools_stay_resident(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    tools = [_fn(n) for n in _CORE] + [{"type": "web_search"}, {"type": "code_interpreter"}]
    tools += [_fn(f"x{i}") for i in range(8)]
    out = inject_tool_search_deferral_openai(tools, "gpt-5.5")
    ws = next(t for t in out if t.get("type") == "web_search")
    ci = next(t for t in out if t.get("type") == "code_interpreter")
    assert "defer_loading" not in ws  # hosted tools can't be deferred
    assert "defer_loading" not in ci


def test_does_not_mutate_input(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    tools = _tools()
    snapshot = copy.deepcopy(tools)
    inject_tool_search_deferral_openai(tools, "gpt-5.5")
    assert tools == snapshot  # deferred tools are copies; the input is untouched


# --- no-op guards ------------------------------------------------------------


def test_noop_for_unsupported_model(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    tools = _tools()
    assert inject_tool_search_deferral_openai(tools, "gpt-4o") is tools


def test_noop_below_min_tools(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    tools = [_fn(f"x{i}") for i in range(5)]  # < 12
    assert inject_tool_search_deferral_openai(tools, "gpt-5.5") is tools


def test_noop_when_tool_search_already_present(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    tools = [{"type": "tool_search"}] + [_fn(f"x{i}") for i in range(15)]
    assert inject_tool_search_deferral_openai(tools, "gpt-5.5") is tools


def test_noop_when_nothing_deferrable(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    tools = [_fn(n) for n in _CORE * 3]  # 18 core tools, none deferrable
    assert inject_tool_search_deferral_openai(tools, "gpt-5.5") is tools


def test_noop_for_non_list():
    assert inject_tool_search_deferral_openai(None, "gpt-5.5") is None


def test_resident_names_match_case_insensitively(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    # The resident-name sets are lowercase; clients are not required to be. An
    # exact match deferred every tool for a PascalCase client, including its own
    # tool-search tool. Mirrors the Anthropic-side fix.
    tools = [_fn(n) for n in ("Bash", "Read", "Edit", "Terminal", "ToolSearch")] + [
        _fn(f"slack_{i}") for i in range(10)
    ]
    out = inject_tool_search_deferral_openai(tools, "gpt-5.5")
    by_name = {t.get("name"): t for t in out if "name" in t}
    for name in ("Bash", "Read", "Edit", "Terminal", "ToolSearch"):
        assert by_name[name].get("defer_loading") is None, name
    assert by_name["slack_0"].get("defer_loading") is True


# --- client-harness exclusion (GH #2660) -------------------------------------


def test_noop_for_a_client_that_cannot_execute_the_search_tool(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    # GH #2660 reports opencode resolving tool calls against its own registry
    # and rejecting the injected tool as unavailable, so its tools stay resident
    # and untouched.
    tools = _tools()
    snapshot = copy.deepcopy(tools)

    out = inject_tool_search_deferral_openai(tools, "gpt-5.5", client="opencode")

    assert out is tools
    assert tools == snapshot
    assert not any(t.get("type") == "tool_search" for t in out)
    assert not any(t.get("defer_loading") for t in out)


def test_supported_clients_keep_the_existing_deferral(monkeypatch):
    monkeypatch.setenv("HEADROOM_OPENAI_TOOL_SEARCH", "1")
    # The exclusion is per-client, not a global default flip: anything that can
    # search still gets the same payload it got before.
    tools = _tools()

    explicit = inject_tool_search_deferral_openai(tools, "gpt-5.5", client="claude-code")
    implicit = inject_tool_search_deferral_openai(tools, "gpt-5.5")

    assert explicit == implicit
    assert implicit[0] == {"type": "tool_search"}
    assert any(t.get("defer_loading") for t in implicit)
