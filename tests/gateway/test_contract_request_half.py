"""Request half of the gateway turn contract: ``POST /v1/compress`` (spec section 1).

Invariants protected here:

* I-LEGACY   - a body WITHOUT a top-level ``gateway`` object behaves exactly as
               today: same response keys, no hooks run, no pending turn.
* I-KEYS     - gateway mode adds ``body``/``turn_id``/``route``/``obligations``/
               ``gateway`` on EVERY answer, including fail-open ones.
* I-BODY     - ``body`` is the complete provider request: pass-through fields
               echoed untouched, ``config``/``gateway``/``token_budget`` stripped,
               ``body.messages`` is the top-level ``messages``.
* I-VALID    - malformed ``gateway`` blocks are 400 ``invalid_request``.
* I-OBLIG    - obligations follow the capability matrix: ``redrive`` only when a
               re-driving hook ran under ``can_redrive`` + ``session_affinity``;
               ``relay_usage`` iff ``can_relay_response``.
* I-NOSHRINK - a re-driving hook never shrinks when nothing can re-drive.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from headroom.proxy.turn_hooks import register_turn_hook
from tests.gateway import fold_only_hook, redrive_hook_ext
from tests.gateway.conftest import compress, registry_of
from tests.gateway.samples import (
    SYSTEM_PROMPT,
    anthropic_tool_history,
    anthropic_tools,
    big_tool_history,
    canonical,
    openai_tools,
    tool_names,
)

LEGACY_STATELESS_KEYS = {
    "messages",
    "tokens_before",
    "tokens_after",
    "tokens_saved",
    "compression_ratio",
    "transforms_applied",
    "transforms_summary",
    "ccr_hashes",
}
GATEWAY_EXTRA_KEYS = {"body", "turn_id", "route", "obligations", "gateway"}
ROUTE_KEYS = {"model", "provider", "service_tier", "reason"}
GATEWAY_ECHO_KEYS = {"can_redrive", "can_relay_response", "session_affinity"}
HEX32 = re.compile(r"^[0-9a-f]{32}$")


class _RecordingHook:
    """A hook that records whether it ran; never changes anything."""

    name = "recording"
    stream_safe = True

    def __init__(self) -> None:
        self.requests = 0

    def on_request(self, ctx: Any) -> None:
        self.requests += 1


def _body(messages: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"model": "gpt-4o", "messages": messages or big_tool_history()}
    out.update(extra)
    return out


# --------------------------------------------------------------------------- #
# I-LEGACY (these pass on the current tree and must keep passing)             #
# --------------------------------------------------------------------------- #


def test_legacy_mode_keys_unchanged_stateless(headroom_client) -> None:
    """I-LEGACY: no ``gateway`` key -> exactly today's stateless key set."""
    resp = compress(headroom_client, _body())
    assert resp.status_code == 200, resp.text
    assert set(resp.json()) == LEGACY_STATELESS_KEYS


def test_legacy_mode_keys_unchanged_session(headroom_client) -> None:
    """I-LEGACY: session mode adds only ``session``."""
    resp = compress(headroom_client, _body(config={"session_id": "legacy-s"}))
    assert resp.status_code == 200, resp.text
    assert set(resp.json()) == LEGACY_STATELESS_KEYS | {"session"}


def test_legacy_mode_ignores_pass_through_fields(headroom_client) -> None:
    """I-LEGACY: extra provider fields are neither echoed nor rejected."""
    resp = compress(
        headroom_client, _body(tools=openai_tools(), system=SYSTEM_PROMPT, temperature=0.2)
    )
    assert resp.status_code == 200, resp.text
    assert set(resp.json()) == LEGACY_STATELESS_KEYS


def test_legacy_mode_runs_no_hooks(headroom_client) -> None:
    """I-LEGACY: turn hooks are a gateway-mode feature; legacy callers never see them."""
    hook = _RecordingHook()
    register_turn_hook(hook)
    fold = fold_only_hook.register(max_chars=100)
    resp = compress(headroom_client, _body())
    assert resp.status_code == 200
    assert hook.requests == 0
    assert fold.request_calls == 0
    # And the payload is untouched by the fold hook.
    assert len(resp.json()["messages"][2]["content"]) > 100


def test_legacy_mode_registers_no_pending_turn(headroom_client) -> None:
    """I-LEGACY: nothing to wait for, nothing registered."""
    compress(headroom_client, _body(config={"session_id": "legacy-p"}))
    registry = registry_of(headroom_client)
    assert registry is None or len(registry) == 0


# --------------------------------------------------------------------------- #
# I-KEYS / I-BODY                                                              #
# --------------------------------------------------------------------------- #


def _assert_contract_shape(data: dict[str, Any], sent: dict[str, Any]) -> None:
    assert GATEWAY_EXTRA_KEYS <= set(data), sorted(data)
    assert HEX32.match(data["turn_id"]), data["turn_id"]
    assert set(data["route"]) == ROUTE_KEYS
    assert isinstance(data["obligations"], list)
    assert set(data["gateway"]) == GATEWAY_ECHO_KEYS
    body = data["body"]
    assert isinstance(body, dict)
    for forbidden in ("config", "gateway", "token_budget"):
        assert forbidden not in body, forbidden
    assert body["model"] == sent["model"]
    assert body["messages"] == data["messages"]


def test_gateway_mode_adds_contract_keys(headroom_client) -> None:
    """I-KEYS: ``gateway: {}`` is gateway mode; the legacy keys are all still there."""
    sent = _body(gateway={})
    resp = compress(headroom_client, sent)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert LEGACY_STATELESS_KEYS <= set(data)
    _assert_contract_shape(data, sent)
    assert data["obligations"] == []
    assert data["gateway"] == {
        "can_redrive": False,
        "can_relay_response": False,
        "session_affinity": True,
    }
    assert data["route"]["model"] == "gpt-4o"
    assert data["route"]["provider"] == "openai"
    assert data["route"]["service_tier"] is None
    assert data["route"]["reason"] == ""
    # Compression still happened in gateway mode.
    assert data["tokens_saved"] > 0
    assert data["body"]["messages"][2]["content"] != sent["messages"][2]["content"]


def test_gateway_mode_with_session(headroom_client) -> None:
    """I-KEYS: session mode + gateway mode -> both key sets."""
    sent = _body(config={"session_id": "gw-s1"}, gateway={"can_relay_response": True})
    resp = compress(headroom_client, sent)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    _assert_contract_shape(data, sent)
    assert data["session"]["id"] == "gw-s1"
    assert data["gateway"]["can_relay_response"] is True


def test_pass_through_fields_echoed_untouched(headroom_client) -> None:
    """I-BODY: unknown provider fields come back byte-identical inside ``body``."""
    metadata = {"user_id": "u-1", "nested": {"k": [1, 2, {"z": None}]}}
    sent = _body(
        gateway={},
        temperature=0.25,
        top_p=0.9,
        service_tier="flex",
        metadata=metadata,
        stream=False,
        response_format={"type": "json_object"},
        token_budget=50_000,
    )
    resp = compress(headroom_client, sent)
    assert resp.status_code == 200, resp.text
    body = resp.json()["body"]
    for key in ("temperature", "top_p", "service_tier", "metadata", "stream", "response_format"):
        assert canonical(body[key]) == canonical(sent[key]), key
    assert "token_budget" not in body
    assert resp.json()["route"]["service_tier"] == "flex"


def test_tools_and_system_echoed(headroom_client) -> None:
    """I-BODY: ``system`` passes through as sent; ``tools`` come back (possibly
    compacted) with every tool name preserved and in order."""
    sent = _body(gateway={}, tools=openai_tools(), system=SYSTEM_PROMPT)
    resp = compress(headroom_client, sent)
    assert resp.status_code == 200, resp.text
    body = resp.json()["body"]
    assert body["system"] == SYSTEM_PROMPT
    assert tool_names(body["tools"]) == tool_names(sent["tools"])


def test_no_tools_means_no_tools_key(headroom_client) -> None:
    """I-BODY: a request without ``tools`` never grows one (an empty ``tools``
    list is a provider 400 on Anthropic)."""
    resp = compress(headroom_client, _body(gateway={}))
    assert resp.status_code == 200
    assert "tools" not in resp.json()["body"]


def test_tool_schema_compaction_is_deterministic(headroom_client) -> None:
    """I-BODY / prefix stability: identical tools in -> identical tool bytes out,
    on every call, and the transform is named when it changed anything."""
    sent = _body(gateway={}, tools=openai_tools())
    first = compress(headroom_client, sent).json()
    second = compress(headroom_client, sent).json()
    assert canonical(first["body"]["tools"]) == canonical(second["body"]["tools"])
    changed = canonical(first["body"]["tools"]) != canonical(sent["tools"])
    assert ("tool_schema_compaction" in first["transforms_applied"]) == changed


def test_route_provider_inferred_from_model(headroom_client) -> None:
    """I-BODY: ``route.provider`` follows the tracker-provider rule (claude/anthropic
    in the model name -> anthropic, else openai) and ``route.model`` is the body model."""
    sent = {
        "model": "claude-sonnet-4-5",
        "messages": anthropic_tool_history(),
        "system": SYSTEM_PROMPT,
        "tools": anthropic_tools(),
        "max_tokens": 64,
        "gateway": {},
    }
    resp = compress(headroom_client, sent)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["route"]["provider"] == "anthropic"
    assert data["route"]["model"] == "claude-sonnet-4-5"
    assert data["body"]["max_tokens"] == 64
    assert data["body"]["system"] == SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# I-VALID                                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "gateway",
    [
        [],
        "yes",
        1,
        True,
        {"can_redrive": "true"},
        {"can_relay_response": 1},
        {"session_affinity": "no"},
        {"can_redrive": None},
    ],
    ids=[
        "list",
        "string",
        "int",
        "bool",
        "redrive-str",
        "relay-int",
        "affinity-str",
        "redrive-null",
    ],
)
def test_invalid_gateway_block_is_400(headroom_client, gateway: Any) -> None:
    """I-VALID: ``gateway`` must be an object whose flags are real booleans."""
    resp = compress(headroom_client, _body(gateway=gateway))
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["type"] == "invalid_request"


def test_unknown_gateway_keys_are_ignored(headroom_client) -> None:
    """I-VALID: forward-compatible - unknown keys inside ``gateway`` are dropped."""
    resp = compress(
        headroom_client,
        _body(gateway={"plugin_version": "kong-headroom/0.1.0", "future_flag": {"x": 1}}),
    )
    assert resp.status_code == 200, resp.text
    assert set(resp.json()["gateway"]) == GATEWAY_ECHO_KEYS


def test_legacy_validation_still_applies_in_gateway_mode(headroom_client) -> None:
    """I-VALID: the pre-existing 400s (missing model/messages) are unchanged."""
    resp = compress(
        headroom_client, {"messages": [{"role": "user", "content": "x"}], "gateway": {}}
    )
    assert resp.status_code == 400
    assert "model" in resp.json()["error"]["message"]


# --------------------------------------------------------------------------- #
# I-KEYS on fail-open answers                                                  #
# --------------------------------------------------------------------------- #


def test_bypass_header_in_gateway_mode_still_returns_contract_keys(headroom_client) -> None:
    """I-KEYS: the bypass short-circuit must still hand the gateway a ``body`` it can
    forward, a ``turn_id`` and an empty obligation list."""
    sent = _body(gateway={"can_relay_response": True}, tools=openai_tools(), temperature=0.1)
    resp = compress(headroom_client, sent, headers={"x-headroom-bypass": "true"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["messages"] == sent["messages"]
    assert data["tokens_before"] == 0 and data["tokens_saved"] == 0
    assert GATEWAY_EXTRA_KEYS <= set(data)
    assert data["obligations"] == []
    assert HEX32.match(data["turn_id"])
    body = data["body"]
    assert body["messages"] == sent["messages"]
    assert canonical(body["tools"]) == canonical(sent["tools"])
    assert body["temperature"] == 0.1
    assert "gateway" not in body and "config" not in body


def test_empty_messages_in_gateway_mode_returns_contract_keys(headroom_client) -> None:
    """I-KEYS: the empty-messages short-circuit is also a gateway answer."""
    sent = _body(messages=[], gateway={})
    sent["messages"] = []
    resp = compress(headroom_client, sent)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["messages"] == []
    assert GATEWAY_EXTRA_KEYS <= set(data)
    assert data["obligations"] == []
    assert data["body"]["messages"] == []
    assert data["route"]["model"] == "gpt-4o"


def test_compression_timeout_fails_open_with_contract_keys(headroom_client, monkeypatch) -> None:
    """I-KEYS: stateless timeout -> 200 + originals + ``compression_skipped`` AND the
    contract keys, so the gateway can forward without a special case."""
    from unittest.mock import AsyncMock

    proxy = headroom_client.app.state.proxy
    monkeypatch.setattr(proxy, "_run_compression_in_executor", AsyncMock(side_effect=TimeoutError))
    sent = _body(gateway={}, tools=openai_tools(), temperature=0.3)
    resp = compress(headroom_client, sent)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["compression_skipped"] is True
    assert data["skip_reason"] == "compression_timeout"
    assert data["messages"] == sent["messages"]
    assert GATEWAY_EXTRA_KEYS <= set(data)
    assert data["obligations"] == []
    assert data["body"]["messages"] == sent["messages"]
    assert data["body"]["temperature"] == 0.3


# --------------------------------------------------------------------------- #
# I-OBLIG / I-NOSHRINK: the capability matrix                                  #
# --------------------------------------------------------------------------- #


def _expected_obligations(
    hook: str, can_redrive: bool, can_relay: bool, affinity: bool
) -> list[str]:
    out: list[str] = []
    if hook == "redrive" and can_redrive and affinity:
        out.append("redrive")
    if can_relay:
        out.append("relay_usage")
    return out


@pytest.mark.parametrize("hook", ["none", "fold", "redrive"])
@pytest.mark.parametrize("can_redrive", [False, True])
@pytest.mark.parametrize("can_relay", [False, True])
@pytest.mark.parametrize("affinity", [False, True])
def test_obligations_matrix(
    headroom_client, hook: str, can_redrive: bool, can_relay: bool, affinity: bool
) -> None:
    """I-OBLIG + I-NOSHRINK over every capability combination and hook kind."""
    fold = redrive = None
    if hook == "fold":
        fold = fold_only_hook.register(max_chars=100)
    elif hook == "redrive":
        redrive = redrive_hook_ext.register()
    sent = _body(
        gateway={
            "can_redrive": can_redrive,
            "can_relay_response": can_relay,
            "session_affinity": affinity,
        },
        tools=openai_tools(),
    )
    resp = compress(headroom_client, sent)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["obligations"] == _expected_obligations(hook, can_redrive, can_relay, affinity)
    assert data["gateway"] == {
        "can_redrive": can_redrive,
        "can_relay_response": can_relay,
        "session_affinity": affinity,
    }
    names = tool_names(data["body"]["tools"])
    if "redrive" in data["obligations"]:
        assert redrive is not None and redrive.shrink_calls == 1
        assert "search_tools" in names
        assert not any(n.startswith("deferred_") for n in names)
    else:
        # I-NOSHRINK: tools are a subset of what came in, in order.
        assert "search_tools" not in names
        assert names == tool_names(sent["tools"])
        if redrive is not None:
            assert redrive.shrink_calls == 0
    if fold is not None:
        # A stream-safe hook runs regardless of the redrive capability.
        assert fold.request_calls == 1
        assert data["body"]["messages"][2]["content"].endswith(fold_only_hook.FOLD_MARKER)
        assert "turn_hook" in data["transforms_applied"]
    # A pending turn exists iff there is something to wait for.
    registry = registry_of(headroom_client)
    assert registry is not None
    if data["obligations"]:
        assert registry.get(data["turn_id"]) is not None
    else:
        assert registry.get(data["turn_id"]) is None


def test_redrive_hook_tools_delta_is_reported(headroom_client) -> None:
    """I-OBLIG: a hook that shrinks tools is reported as ``turn_hook:tools:<n>tok``
    (mirrors the chat path) so the savings are attributable."""
    redrive_hook_ext.register()
    sent = _body(gateway={"can_redrive": True}, tools=openai_tools())
    data = compress(headroom_client, sent).json()
    assert data["obligations"] == ["redrive"]
    tags = [t for t in data["transforms_applied"] if t.startswith("turn_hook:tools:")]
    assert len(tags) == 1, data["transforms_applied"]
    assert re.match(r"^turn_hook:tools:\d+tok$", tags[0])


def test_session_mode_replays_hook_output(headroom_client) -> None:
    """I-BODY in session mode: the hook-rewritten messages are what the gateway
    forwards, so they must be what turn 2 replays byte-for-byte."""
    fold_only_hook.register(max_chars=120)
    history = big_tool_history()
    turn1 = compress(
        headroom_client, _body(history, config={"session_id": "gw-fold"}, gateway={})
    ).json()
    folded = turn1["body"]["messages"][2]["content"]
    assert folded.endswith(fold_only_hook.FOLD_MARKER)
    turn2 = compress(
        headroom_client,
        _body(
            history + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}],
            config={"session_id": "gw-fold"},
            gateway={},
        ),
    ).json()
    assert canonical(turn2["body"]["messages"][: len(turn1["body"]["messages"])]) == canonical(
        turn1["body"]["messages"]
    )
    assert turn2["body"]["messages"][-1]["content"] == "more"


def test_turn_ids_are_unique_per_request(headroom_client) -> None:
    """I-KEYS: every request half mints a fresh id, registered or not."""
    ids = {compress(headroom_client, _body(gateway={})).json()["turn_id"] for _ in range(5)}
    assert len(ids) == 5
