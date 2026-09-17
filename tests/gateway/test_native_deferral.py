"""Tool-search deferral on the gateway contract is a turn hook's job.

OSS core does not defer tool schemas on ``POST /v1/compress``; the licensed
tool-search extension's native tier does, through the turn-hook contract. These
tests use the reference hook in ``tests/gateway/deferral_hook.py`` (the same
primitives, tags and header seam the extension uses) and protect what CORE
promises to such a hook:

* I-PLAIN    - with no hook registered, 12+ Anthropic tools pass through
               compacted, ``headers`` is ``{}`` and no deferral is named.
* I-CARRY    - a hook's deferral reaches ``body.tools``; the hook's
               ``provider_headers`` become the response's ``headers`` with the
               client's ``anthropic-beta`` tokens first and Headroom's
               appended; the ``tool_search_deferred_*`` tags name the
               ``tool_search:native_deferral:<n>tools:<n>tok`` transform and
               reach the outcome, the ledger and ``metrics.record_request``.
* I-SEAM     - only allow-listed header names are honoured; a hook that is not
               stream-safe is skipped when the gateway cannot re-drive; hooks
               run by ``priority`` so a router that changes the model runs
               before the deferral gate.
* I-GATES    - the OSS gate helpers the hook applies: fewer than 12 tools, a
               client already searching, ``HEADROOM_TOOL_SEARCH=0``,
               Bedrock/Vertex ids, OpenAI models, OpenAI-shaped tools.
* I-STABLE   - identical bytes and header across three session turns.
* I-LEGACY / I-FAILOPEN - no ``gateway`` block -> no ``headers`` key; fail-open
               answers carry ``headers: {}`` even with the hook registered.
* I-REPAIR   - tool-search history repair stays in core (in-place placeholder,
               chat-path parity).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from headroom.proxy.savings_attribution import from_tags
from tests.gateway.conftest import compress
from tests.gateway.deferral_hook import (
    BETA_TOKEN,
    BufferedDeferralHook,
    DeferralHook,
    ModelRouterHook,
    register,
)
from tests.gateway.fake_provider import anthropic_usage
from tests.gateway.samples import (
    SYSTEM_PROMPT,
    anthropic_tool_history,
    anthropic_tools,
    big_tool_history,
    canonical,
    openai_tools,
    tool_names,
)

CLAUDE = "claude-sonnet-4-5"
SEARCH_TYPE_PREFIX = "tool_search_tool_regex"
CLIENT_BETA = "context-1m-2025-08-07"
DEFERRAL_LABEL = re.compile(r"^tool_search:native_deferral:(\d+)tools:(\d+)tok$")
OLD_CORE_LABEL = "router:tool_search_deferral"
CORE_NAMES = ("Bash", "Read")


@pytest.fixture
def deferral_hook() -> DeferralHook:
    return register()


def _schema(prefix: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": f"The {prefix} query."},
        },
        "required": ["query"],
    }


def _many_anthropic_tools(n_non_core: int = 12) -> list[dict[str, Any]]:
    """Two core (resident) tools plus ``n_non_core`` deferrable MCP-style ones."""
    tools: list[dict[str, Any]] = [
        {"name": name, "description": f"{name} tool.", "input_schema": _schema(name)}
        for name in CORE_NAMES
    ]
    tools.extend(
        {
            "name": f"mcp__srv__tool_{i:02d}",
            "description": f"MCP tool number {i} with a description long enough to count.",
            "input_schema": _schema(f"tool_{i}"),
        }
        for i in range(n_non_core)
    )
    return tools


def _many_openai_tools(n: int = 14) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": f"fn_{i:02d}",
                "description": f"Function number {i}.",
                "parameters": _schema(f"fn_{i}"),
            },
        }
        for i in range(n)
    ]


def _body(
    *,
    model: str = CLAUDE,
    tools: Any = None,
    messages: list[dict[str, Any]] | None = None,
    gateway: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": model,
        "messages": messages if messages is not None else anthropic_tool_history(),
        "system": SYSTEM_PROMPT,
        "max_tokens": 64,
    }
    if tools is not None:
        out["tools"] = tools
    if gateway is not None:
        out["gateway"] = gateway
    out.update(extra)
    return out


def _search_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [t for t in tools if str(t.get("type", "")).startswith(SEARCH_TYPE_PREFIX)]


def _deferred(tools: list[dict[str, Any]]) -> list[str]:
    return [str(t["name"]) for t in tools if t.get("defer_loading")]


def _deferral_labels(data: dict[str, Any]) -> list[str]:
    return [t for t in data["transforms_applied"] if DEFERRAL_LABEL.match(t)]


def _assert_untouched(data: dict[str, Any], sent_tools: list[dict[str, Any]]) -> None:
    assert canonical(tool_names(data["body"]["tools"])) == canonical(tool_names(sent_tools))
    assert not _search_tools(data["body"]["tools"])
    assert not _deferred(data["body"]["tools"])
    assert data["headers"] == {}
    assert _deferral_labels(data) == []
    assert not any(t.startswith(OLD_CORE_LABEL) for t in data["transforms_applied"])


def _assert_deferred(data: dict[str, Any], sent_tools: list[dict[str, Any]]) -> int:
    tools = data["body"]["tools"]
    search = _search_tools(tools)
    assert len(search) == 1
    assert search[0]["name"] == "tool_search_tool_regex"
    assert not search[0].get("defer_loading")
    assert set(_deferred(tools)) == {t["name"] for t in sent_tools if t["name"] not in CORE_NAMES}
    resident = [t for t in tools if not t.get("type") and not t.get("defer_loading")]
    assert [t["name"] for t in resident] == list(CORE_NAMES)
    assert set(tool_names(tools)) == set(tool_names(sent_tools)) | {"tool_search_tool_regex"}
    labels = _deferral_labels(data)
    assert len(labels) == 1, data["transforms_applied"]
    n_tools, n_tok = (int(x) for x in DEFERRAL_LABEL.match(labels[0]).groups())
    assert n_tools == len(_deferred(tools))
    assert n_tok > 0
    return n_tok


# --------------------------------------------------------------------------- #
# I-PLAIN: OSS core alone never defers on this path                            #
# --------------------------------------------------------------------------- #


def test_no_hook_leaves_tools_untouched_and_headers_empty(headroom_client, outcome_spy) -> None:
    outcomes = outcome_spy(headroom_client)
    sent = _body(
        tools=_many_anthropic_tools(),
        gateway={"request_headers": {"anthropic-beta": CLIENT_BETA}},
    )
    resp = compress(headroom_client, sent)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    _assert_untouched(data, sent["tools"])
    # The client's beta is not echoed back either: nothing asked for a header.
    assert data["headers"] == {}
    assert "tool_search_deferred_tokens" not in outcomes[0].tags


def test_no_hook_with_client_already_searching_is_passed_through(headroom_client) -> None:
    tools = [{"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}]
    tools += _many_anthropic_tools()
    sent = _body(tools=tools, gateway={})
    data = compress(headroom_client, sent).json()
    assert canonical(data["body"]["tools"]) == canonical(sent["tools"])
    assert data["headers"] == {}


# --------------------------------------------------------------------------- #
# I-CARRY: a hook's deferral, header and savings reach the gateway             #
# --------------------------------------------------------------------------- #


def test_hook_deferral_reaches_body_headers_and_outcome(
    headroom_client, outcome_spy, deferral_hook
) -> None:
    outcomes = outcome_spy(headroom_client)
    sent = _body(
        tools=_many_anthropic_tools(),
        gateway={"request_headers": {"anthropic-beta": CLIENT_BETA}},
    )
    resp = compress(headroom_client, sent)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    n_tok = _assert_deferred(data, sent["tools"])
    assert deferral_hook.applied == 1

    # Client tokens first, then the hook's, deduplicated.
    assert data["headers"] == {"anthropic-beta": f"{CLIENT_BETA},{BETA_TOKEN}"}

    assert len(outcomes) == 1
    tags = outcomes[0].tags
    assert tags["tool_search_deferred_tools"] == 12
    assert tags["tool_search_deferred_tokens"] == n_tok
    ledger = [item for item in from_tags(tags) if item["source"] == "tool_search"]
    assert ledger and ledger[0]["tokens"] == n_tok
    assert _deferral_labels(data)[0] in outcomes[0].transforms_applied
    # Deferral does not shrink the tools array, so the runner's own fold must
    # not double-count it.
    assert "turn_hook_tools_saved_tokens" not in tags


def test_hook_header_without_client_value_is_just_the_hooks(headroom_client, deferral_hook) -> None:
    data = compress(headroom_client, _body(tools=_many_anthropic_tools(), gateway={})).json()
    assert data["headers"] == {"anthropic-beta": BETA_TOKEN}


def test_client_beta_already_carrying_token_is_not_duplicated(
    headroom_client, deferral_hook
) -> None:
    sent = _body(
        tools=_many_anthropic_tools(),
        gateway={"request_headers": {"Anthropic-Beta": f"{BETA_TOKEN},{CLIENT_BETA}"}},
    )
    data = compress(headroom_client, sent).json()
    assert data["headers"] == {"anthropic-beta": f"{BETA_TOKEN},{CLIENT_BETA}"}


def test_deferral_composes_with_compaction_and_ccr_tool(headroom_client, deferral_hook) -> None:
    """The hook runs after compaction and before CCR tool injection
    (``headroom_retrieve`` stays resident so the model can call it)."""
    sent = _body(
        tools=_many_anthropic_tools(),
        config={"mode": "ccr"},
        gateway={"can_redrive": True, "session_affinity": True},
    )
    data = compress(headroom_client, sent).json()
    tools = data["body"]["tools"]
    assert _search_tools(tools)
    retrieve = [t for t in tools if t.get("name") == "headroom_retrieve"]
    if retrieve:  # only injected when markers were inserted
        assert not retrieve[0].get("defer_loading")
        assert "ccr_tool_injected" in data["transforms_applied"]
    assert data["headers"]["anthropic-beta"] == BETA_TOKEN


# --------------------------------------------------------------------------- #
# I-SEAM: allowlist, stream-safety, priority                                   #
# --------------------------------------------------------------------------- #


def test_hook_requesting_disallowed_headers_gets_only_the_allowlisted_one(headroom_client) -> None:
    register(extra_headers={"Authorization": "Bearer hook-secret", "x-api-key": "sk-hook", "": "x"})
    sent = _body(tools=_many_anthropic_tools(), gateway={})
    data = compress(headroom_client, sent).json()
    assert data["headers"] == {"anthropic-beta": BETA_TOKEN}
    assert "hook-secret" not in canonical(data)


def test_buffered_hook_is_skipped_when_the_gateway_cannot_redrive(headroom_client) -> None:
    """A hook that is not stream-safe never runs on a turn the gateway cannot
    re-drive — which is every streamed Claude Code turn. The native tier must
    therefore be stream-safe; this pins the rule that makes that necessary."""
    hook = register(BufferedDeferralHook())
    sent = _body(tools=_many_anthropic_tools(), gateway={"can_redrive": False})
    data = compress(headroom_client, sent).json()
    _assert_untouched(data, sent["tools"])
    assert hook.request_calls == 0

    sent = _body(
        tools=_many_anthropic_tools(),
        config={"session_id": "buffered-ok"},
        gateway={"can_redrive": True, "session_affinity": True},
    )
    data = compress(headroom_client, sent).json()
    _assert_deferred(data, sent["tools"])
    assert hook.request_calls == 1


def test_router_with_lower_priority_runs_before_the_deferral_gate(headroom_client) -> None:
    """Registered AFTER the deferral hook, a priority-10 router still runs
    first, so the first-party gate sees the routed Bedrock id and declines."""
    register()
    register(ModelRouterHook("anthropic.claude-sonnet-4-5-v1:0", priority=10))
    sent = _body(tools=_many_anthropic_tools(), gateway={})
    data = compress(headroom_client, sent).json()
    _assert_untouched(data, sent["tools"])


def test_router_with_higher_priority_runs_after_and_the_deferral_stands(headroom_client) -> None:
    register()
    register(ModelRouterHook("anthropic.claude-sonnet-4-5-v1:0", priority=300))
    sent = _body(tools=_many_anthropic_tools(), gateway={})
    data = compress(headroom_client, sent).json()
    _assert_deferred(data, sent["tools"])


# --------------------------------------------------------------------------- #
# I-GATES: the OSS gate helpers, as the hook applies them                      #
# --------------------------------------------------------------------------- #


def test_fewer_than_twelve_tools_untouched(headroom_client, deferral_hook) -> None:
    sent = _body(tools=_many_anthropic_tools(9), gateway={})  # 11 in total
    data = compress(headroom_client, sent).json()
    _assert_untouched(data, sent["tools"])


def test_three_sample_tools_untouched(headroom_client, deferral_hook) -> None:
    sent = _body(tools=anthropic_tools(), gateway={})
    data = compress(headroom_client, sent).json()
    _assert_untouched(data, sent["tools"])


def test_client_already_using_tool_search_untouched(headroom_client, deferral_hook) -> None:
    tools = [{"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}]
    tools += _many_anthropic_tools()
    sent = _body(tools=tools, gateway={"request_headers": {"anthropic-beta": CLIENT_BETA}})
    data = compress(headroom_client, sent).json()
    assert canonical(data["body"]["tools"]) == canonical(sent["tools"])
    assert data["headers"] == {}
    assert _deferral_labels(data) == []


def test_env_flag_off_untouched(headroom_client, deferral_hook, monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "0")
    sent = _body(tools=_many_anthropic_tools(), gateway={})
    data = compress(headroom_client, sent).json()
    _assert_untouched(data, sent["tools"])


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "vertex_ai/claude-sonnet-4@20250514",
        "claude-sonnet-4@20250514",
    ],
)
def test_bedrock_and_vertex_model_ids_untouched(headroom_client, deferral_hook, model: str) -> None:
    sent = _body(model=model, tools=_many_anthropic_tools(), gateway={})
    data = compress(headroom_client, sent).json()
    assert data["route"]["provider"] == "anthropic"
    _assert_untouched(data, sent["tools"])


def test_litellm_anthropic_prefix_is_first_party(headroom_client, deferral_hook) -> None:
    sent = _body(model="anthropic/claude-sonnet-4-5", tools=_many_anthropic_tools(), gateway={})
    data = compress(headroom_client, sent).json()
    assert _search_tools(data["body"]["tools"])
    assert data["headers"]["anthropic-beta"] == BETA_TOKEN


def test_openai_model_untouched(headroom_client, deferral_hook) -> None:
    sent = {
        "model": "gpt-4o",
        "messages": big_tool_history(),
        "tools": _many_openai_tools(),
        "gateway": {"request_headers": {"anthropic-beta": CLIENT_BETA}},
    }
    data = compress(headroom_client, sent).json()
    _assert_untouched(data, sent["tools"])


def test_openai_shaped_tools_on_claude_model_untouched(headroom_client, deferral_hook) -> None:
    sent = _body(tools=_many_openai_tools(), messages=big_tool_history(), gateway={})
    data = compress(headroom_client, sent).json()
    assert data["route"]["provider"] == "anthropic"
    _assert_untouched(data, sent["tools"])


def test_mixed_shape_tools_untouched(headroom_client, deferral_hook) -> None:
    tools = _many_anthropic_tools() + openai_tools()[:1]
    sent = _body(tools=tools, gateway={})
    data = compress(headroom_client, sent).json()
    assert not _search_tools(data["body"]["tools"])
    assert data["headers"] == {}


# --------------------------------------------------------------------------- #
# gateway.request_headers validation                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "request_headers",
    [[], "anthropic-beta: x", 1, {"anthropic-beta": 1}, {"anthropic-beta": ["a"]}],
    ids=["list", "string", "int", "int-value", "list-value"],
)
def test_invalid_request_headers_is_400(headroom_client, request_headers: Any) -> None:
    sent = _body(tools=_many_anthropic_tools(), gateway={"request_headers": request_headers})
    resp = compress(headroom_client, sent)
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["type"] == "invalid_request"


def test_other_request_headers_are_ignored(headroom_client, deferral_hook) -> None:
    sent = _body(
        tools=_many_anthropic_tools(),
        gateway={
            "request_headers": {
                "authorization": "Bearer secret",
                "x-api-key": "sk-ant-secret",
                "anthropic-beta": None,
            }
        },
    )
    data = compress(headroom_client, sent).json()
    assert data["headers"] == {"anthropic-beta": BETA_TOKEN}
    assert "authorization" not in canonical(data["headers"])
    assert "secret" not in canonical(data)


# --------------------------------------------------------------------------- #
# I-STABLE / I-LEGACY / I-FAILOPEN                                             #
# --------------------------------------------------------------------------- #


def test_byte_stable_over_three_session_turns(headroom_client, deferral_hook) -> None:
    history = anthropic_tool_history()
    tools = _many_anthropic_tools()
    seen_tools: list[str] = []
    seen_headers: list[str] = []
    seen_labels: list[list[str]] = []
    for turn in range(3):
        messages = history + [
            m
            for i in range(turn)
            for m in (
                {"role": "assistant", "content": f"answer {i}"},
                {"role": "user", "content": f"follow-up {i}"},
            )
        ]
        sent = _body(
            tools=tools,
            messages=messages,
            config={"session_id": "gw-defer-stable"},
            gateway={
                "can_relay_response": True,
                "request_headers": {"anthropic-beta": CLIENT_BETA},
            },
        )
        resp = compress(headroom_client, sent)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        seen_tools.append(canonical(data["body"]["tools"]))
        seen_headers.append(canonical(data["headers"]))
        seen_labels.append(_deferral_labels(data))
    assert len(set(seen_tools)) == 1
    assert len(set(seen_headers)) == 1
    assert seen_labels[0] and seen_labels.count(seen_labels[0]) == 3


def test_stateless_calls_are_byte_stable(headroom_client, deferral_hook) -> None:
    sent = _body(tools=_many_anthropic_tools(), gateway={})
    first = compress(headroom_client, sent).json()
    second = compress(headroom_client, sent).json()
    assert canonical(first["body"]["tools"]) == canonical(second["body"]["tools"])
    assert first["headers"] == second["headers"]


def test_legacy_mode_untouched(headroom_client, outcome_spy, deferral_hook) -> None:
    outcomes = outcome_spy(headroom_client)
    sent = _body(tools=_many_anthropic_tools())
    resp = compress(headroom_client, sent)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "headers" not in data and "body" not in data
    assert _deferral_labels(data) == []
    assert "tool_search_deferred_tokens" not in outcomes[0].tags


def test_bypass_header_fails_open_with_empty_headers(headroom_client, deferral_hook) -> None:
    sent = _body(tools=_many_anthropic_tools(), gateway={})
    data = compress(headroom_client, sent, headers={"x-headroom-bypass": "true"}).json()
    assert data["headers"] == {}
    assert canonical(data["body"]["tools"]) == canonical(sent["tools"])


def test_timeout_fails_open_with_empty_headers(headroom_client, deferral_hook, monkeypatch) -> None:
    from unittest.mock import AsyncMock

    proxy = headroom_client.app.state.proxy
    monkeypatch.setattr(proxy, "_run_compression_in_executor", AsyncMock(side_effect=TimeoutError))
    sent = _body(tools=_many_anthropic_tools(), gateway={})
    data = compress(headroom_client, sent).json()
    assert data["compression_skipped"] is True
    assert data["headers"] == {}
    assert canonical(data["body"]["tools"]) == canonical(sent["tools"])


def test_empty_messages_fails_open_with_empty_headers(headroom_client, deferral_hook) -> None:
    sent = _body(tools=_many_anthropic_tools(), messages=[], gateway={})
    data = compress(headroom_client, sent).json()
    assert data["headers"] == {}


# --------------------------------------------------------------------------- #
# I-REPAIR: history repair stays in core                                       #
# --------------------------------------------------------------------------- #


def _history_with_search_blocks(referenced: str) -> list[dict[str, Any]]:
    return anthropic_tool_history() + [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "tool_search_tool_regex",
                    "input": {"pattern": "tool_03"},
                },
                {
                    "type": "tool_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": {
                        "tool_references": [{"type": "tool_reference", "tool_name": referenced}]
                    },
                },
                {"type": "text", "text": "Found it."},
            ],
        },
        {"role": "user", "content": "Go on."},
    ]


def test_history_repair_neutralises_unsupportable_search_blocks_in_place(headroom_client) -> None:
    """No hook, three tools, a transcript with last turn's tool-search blocks:
    core repairs them exactly as the chat path does (#3457): each unsupported
    ``server_tool_use`` / ``tool_search_tool_result`` block is replaced IN
    PLACE by the constant, inert placeholder text block, so block coordinates
    (signed thinking blocks) survive and the repaired prefix is byte-stable.
    No search block reaches the provider, the client's own text is untouched,
    and the repair is named. A client that defers on its own gets this without
    any extension."""
    from headroom.proxy.helpers import strip_unsupported_tool_search_blocks

    history = _history_with_search_blocks("mcp__srv__tool_03")
    sent = _body(tools=anthropic_tools(), messages=history, gateway={})
    data = compress(headroom_client, sent).json()
    assistant = data["body"]["messages"][-2]

    # Chat-path parity: the gateway path emits the helper's own output.
    expected, stripped = strip_unsupported_tool_search_blocks(
        _history_with_search_blocks("mcp__srv__tool_03"), anthropic_tools()
    )
    assert stripped == 2
    assert canonical(assistant) == canonical(expected[-2])

    # The invariant the provider cares about: no tool-search block survives,
    # and what replaced them is text that names nothing a validator could
    # resolve (constant placeholder, not the original pattern or reference).
    types = [b["type"] for b in assistant["content"]]
    assert types == ["text", "text", "text"]
    assert not any(t in ("server_tool_use", "tool_search_tool_result") for t in types)
    placeholders = [b["text"] for b in assistant["content"][:2]]
    assert len(set(placeholders)) == 1 and "tool search omitted" in placeholders[0]
    assert "tool_03" not in placeholders[0] and "srvtoolu" not in canonical(assistant)
    assert assistant["content"][2] == {"type": "text", "text": "Found it."}

    assert "router:tool_search_repair:2blocks" in data["transforms_applied"]
    assert data["messages"] == data["body"]["messages"]


def test_history_repair_keeps_blocks_the_deferred_tools_support(
    headroom_client, deferral_hook
) -> None:
    sent = _body(
        tools=_many_anthropic_tools(),
        messages=_history_with_search_blocks("mcp__srv__tool_03"),
        gateway={},
    )
    data = compress(headroom_client, sent).json()
    assistant = data["body"]["messages"][-2]
    assert [b["type"] for b in assistant["content"]] == [
        "server_tool_use",
        "tool_search_tool_result",
        "text",
    ]
    assert not any(t.startswith("router:tool_search_repair") for t in data["transforms_applied"])


def test_history_repair_is_anthropic_only(headroom_client) -> None:
    sent = {
        "model": "gpt-4o",
        "messages": _history_with_search_blocks("mcp__srv__tool_03"),
        "tools": openai_tools(),
        "gateway": {},
    }
    data = compress(headroom_client, sent).json()
    assert [b["type"] for b in data["body"]["messages"][-2]["content"]] == [
        "server_tool_use",
        "tool_search_tool_result",
        "text",
    ]


# --------------------------------------------------------------------------- #
# I-CARRY, both halves: the Kong-shaped run                                    #
# --------------------------------------------------------------------------- #


def test_kong_shaped_turn_credits_hook_deferral_through_the_response_half(
    headroom_client, deferral_hook, monkeypatch
) -> None:
    """Request half (session + relay owed) -> response half with Anthropic
    usage -> the deferred outcome reaches ``metrics.record_request`` with the
    hook's deferred schema tokens on ``tool_search_saved`` (what ``/stats``
    reports as ``tool_schema_tokens_saved``)."""
    proxy = headroom_client.app.state.proxy
    recorded: list[dict[str, Any]] = []
    original = proxy.metrics.record_request

    async def _spy(*args: Any, **kwargs: Any) -> Any:
        recorded.append(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(proxy.metrics, "record_request", _spy)

    sent = _body(
        tools=_many_anthropic_tools(),
        config={"session_id": "kong-defer"},
        gateway={
            "can_redrive": False,
            "can_relay_response": True,
            "session_affinity": True,
            "plugin_version": "kong-headroom/0.1.0",
            "request_headers": {"anthropic-beta": CLIENT_BETA},
        },
    )
    half1 = headroom_client.post("/v1/compress", json=sent)
    assert half1.status_code == 200, half1.text
    data = half1.json()
    assert data["obligations"] == ["relay_usage"]
    assert data["headers"]["anthropic-beta"] == f"{CLIENT_BETA},{BETA_TOKEN}"
    n_tok = _assert_deferred(data, sent["tools"])
    assert recorded == []  # outcome deferred until the response half

    half2 = headroom_client.post(
        "/v1/compress/response",
        json={
            "turn_id": data["turn_id"],
            "status": 200,
            "latency_ms": 12.5,
            "usage": anthropic_usage(1200, 40, cache_read=1000, cache_write=0),
        },
    )
    assert half2.status_code == 200, half2.text
    assert half2.json()["action"] == "done"
    assert len(recorded) == 1
    assert recorded[0]["tool_search_saved"] == n_tok
    assert recorded[0]["output_tokens"] == 40
    assert proxy.metrics.tool_search_saved_total == n_tok


# --------------------------------------------------------------------------- #
# Parity with the extension's native tier, when it is installed                #
# --------------------------------------------------------------------------- #


def test_extension_native_tier_matches_the_reference_hook(headroom_client) -> None:
    """With headroom-tool-search installed, its native tier must produce the
    same tools bytes and header as the reference hook on Claude Code's tool
    list. Skipped where the extension is not importable."""
    native = pytest.importorskip("headroom_tool_search.native")
    try:
        hook = native.NativeDeferralHook(None)
    except Exception as exc:  # pragma: no cover - extension-side construction
        pytest.skip(f"NativeDeferralHook not constructible here: {exc}")
    claude_code_tools = [
        {"name": name, "description": f"{name} tool.", "input_schema": _schema(name)}
        for name in (
            "Task", "Bash", "Glob", "Grep", "ExitPlanMode", "Read", "Edit", "Write",
            "NotebookEdit", "WebFetch", "TodoWrite", "WebSearch", "BashOutput",
            "KillShell", "AskUserQuestion", "Skill", "SlashCommand", "EnterPlanMode",
            "ToolSearch", "CronCreate", "CronDelete", "CronList",
        )
    ]  # fmt: skip
    sent = _body(
        tools=claude_code_tools, gateway={"request_headers": {"anthropic-beta": CLIENT_BETA}}
    )

    register()
    reference = compress(headroom_client, sent).json()
    from headroom.proxy.turn_hooks import clear_turn_hooks

    clear_turn_hooks()
    register(hook)
    extension = compress(headroom_client, sent).json()

    assert canonical(extension["body"]["tools"]) == canonical(reference["body"]["tools"])
    assert extension["headers"] == reference["headers"]
    assert _deferral_labels(extension) == _deferral_labels(reference)
    # Chat-path parity: the same tools bytes core's own deferral would send.
    from headroom.proxy.helpers import inject_tool_search_deferral

    expected = inject_tool_search_deferral(claude_code_tools)
    assert _deferred(extension["body"]["tools"]) == _deferred(expected)
    assert "ToolSearch" not in _deferred(extension["body"]["tools"])
    assert "Bash" not in _deferred(extension["body"]["tools"])
