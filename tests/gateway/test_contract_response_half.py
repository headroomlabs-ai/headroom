"""Response half of the gateway turn contract: ``POST /v1/compress/response`` (spec section 2).

Invariants protected here:

* R-LOOKUP  - unknown/expired turn -> 404 ``unknown_turn``; a completed turn is
              gone (second post is 404).
* R-VALID   - 400 on missing/invalid ``turn_id``, non-object ``response``,
              non-object/negative/bool usage, and a missing ``response`` on a
              turn that carries ``redrive``.
* R-DONE    - shape of the ``done`` answer; ``response`` is null unless a hook
              replaced it.
* R-USAGE   - Anthropic / OpenAI / Kong usage shapes feed the session tracker
              exactly like ``/v1/usage`` (same ``frozen_message_count``); a
              usage with no cache signal is accepted but not applied.
* R-OUTCOME - with ``relay_usage`` the RequestOutcome is deferred and recorded
              exactly once, completed with the provider's numbers; without it
              the outcome is recorded at the request half as today; registry
              expiry records the draft.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests.gateway.conftest import compress, count_loop_tasks, registry_of
from tests.gateway.samples import big_tool_history

RESPONSE_HALF = "/v1/compress/response"
DONE_KEYS = {
    "action",
    "turn_id",
    "response",
    "frozen_message_count",
    "usage_applied",
    "rounds",
    "billed_usage",
}


def _open_turn(client, session_id: str | None = "resp-s", **gateway: Any) -> dict[str, Any]:
    gateway.setdefault("can_relay_response", True)
    body: dict[str, Any] = {"model": "gpt-4o", "messages": big_tool_history(), "gateway": gateway}
    if session_id is not None:
        body["config"] = {"session_id": session_id}
    resp = compress(client, body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "relay_usage" in data["obligations"]
    return data


def _post(client, payload: dict[str, Any]):
    return client.post(RESPONSE_HALF, json=payload)


# --------------------------------------------------------------------------- #
# R-LOOKUP / R-VALID                                                           #
# --------------------------------------------------------------------------- #


def test_unknown_turn_is_404(headroom_client) -> None:
    """R-LOOKUP"""
    resp = _post(headroom_client, {"turn_id": "0" * 32, "usage": {"prompt_tokens": 1}})
    assert resp.status_code == 404, resp.text
    assert resp.json()["error"]["type"] == "unknown_turn"


def test_unknown_turn_leaves_no_footprint(headroom_client) -> None:
    """R-LOOKUP: a flood of bogus ids must not grow the registry."""
    registry = registry_of(headroom_client)
    assert registry is not None
    before = len(registry)
    for i in range(10):
        assert _post(headroom_client, {"turn_id": f"{i:032x}"}).status_code == 404
    assert len(registry) == before


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"turn_id": None},
        {"turn_id": 42},
        {"turn_id": ""},
        {"turn_id": "x" * 300},
    ],
    ids=["missing", "null", "int", "empty", "too-long"],
)
def test_invalid_turn_id_is_400(headroom_client, payload: dict[str, Any]) -> None:
    """R-VALID"""
    resp = _post(headroom_client, payload)
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["type"] == "invalid_request"


def test_invalid_json_is_400(headroom_client) -> None:
    """R-VALID"""
    resp = headroom_client.post(
        RESPONSE_HALF, content=b"nope", headers={"content-type": "application/json"}
    )
    assert resp.status_code == 400


@pytest.mark.parametrize(
    "extra",
    [
        {"response": "not-an-object"},
        {"response": [1, 2]},
        {"usage": "nope"},
        {"usage": {"prompt_tokens": -1}},
        {"usage": {"cache_read_input_tokens": True}},
        {"usage": {"prompt_tokens_details": {"cached_tokens": -5}}},
    ],
    ids=[
        "response-str",
        "response-list",
        "usage-str",
        "usage-negative",
        "usage-bool",
        "cached-negative",
    ],
)
def test_malformed_response_or_usage_is_400_and_keeps_turn(
    headroom_client, extra: dict[str, Any]
) -> None:
    """R-VALID: a rejected relay must not consume the turn."""
    turn = _open_turn(headroom_client)
    resp = _post(headroom_client, {"turn_id": turn["turn_id"], **extra})
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["type"] == "invalid_request"
    assert registry_of(headroom_client).get(turn["turn_id"]) is not None


# --------------------------------------------------------------------------- #
# R-DONE                                                                        #
# --------------------------------------------------------------------------- #


def test_done_with_null_response_and_turn_removed(headroom_client) -> None:
    """R-DONE + R-LOOKUP: no hook ran, so ``response`` is null; the turn is consumed."""
    turn = _open_turn(headroom_client)
    resp = _post(
        headroom_client,
        {
            "turn_id": turn["turn_id"],
            "status": 200,
            "latency_ms": 12.5,
            "model": "gpt-4o-2024-08-06",
            "usage": {"prompt_tokens": 100, "completion_tokens": 7},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert set(data) == DONE_KEYS
    assert data["action"] == "done"
    assert data["turn_id"] == turn["turn_id"]
    assert data["response"] is None
    assert data["rounds"] == 0
    assert data["usage_applied"] is False  # no cache signal in that usage
    assert data["billed_usage"] == {"input_tokens": 100, "output_tokens": 7}
    assert isinstance(data["frozen_message_count"], int)
    assert registry_of(headroom_client).get(turn["turn_id"]) is None
    again = _post(headroom_client, {"turn_id": turn["turn_id"]})
    assert again.status_code == 404
    assert again.json()["error"]["type"] == "unknown_turn"


def test_done_without_session_has_null_frozen_count(headroom_client) -> None:
    """R-DONE: stateless turns carry no tracker, so ``frozen_message_count`` is null
    and usage is never applied."""
    turn = _open_turn(headroom_client, session_id=None)
    resp = _post(
        headroom_client,
        {
            "turn_id": turn["turn_id"],
            "usage": {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 5000},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["action"] == "done"
    assert data["frozen_message_count"] is None
    assert data["usage_applied"] is False


def test_done_with_minimal_payload(headroom_client) -> None:
    """R-DONE: ``turn_id`` alone is a valid relay (status defaults to 200)."""
    turn = _open_turn(headroom_client)
    resp = _post(headroom_client, {"turn_id": turn["turn_id"]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["action"] == "done"
    assert resp.json()["usage_applied"] is False


def test_done_leaves_no_tasks_behind(headroom_client) -> None:
    """R-DONE: a hook-less turn never starts a suspended runner, and after ``done``
    the loop has exactly the tasks it had before."""
    baseline = count_loop_tasks(headroom_client)
    turn = _open_turn(headroom_client)
    _post(headroom_client, {"turn_id": turn["turn_id"], "usage": {"prompt_tokens": 1}})
    assert count_loop_tasks(headroom_client) <= baseline


# --------------------------------------------------------------------------- #
# R-USAGE                                                                       #
# --------------------------------------------------------------------------- #

ANTHROPIC_WARM = {
    "input_tokens": 200,
    "output_tokens": 9,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 50_000,
}
OPENAI_WARM = {
    "prompt_tokens": 50_200,
    "completion_tokens": 9,
    "prompt_tokens_details": {"cached_tokens": 50_000},
}
OPENAI_TOPLEVEL_CACHED = {"prompt_tokens": 50_200, "completion_tokens": 9, "cached_tokens": 50_000}
KONG_NESTED = {"usage": OPENAI_WARM}  # Kong ai.proxy log statistics keep it nested


def _frozen_after_legacy_usage(client, history, usage: dict[str, Any]) -> int:
    """The yardstick: what ``/v1/usage`` does with the Anthropic-shaped equivalent."""
    sid = f"legacy-{time.time_ns()}"
    assert (
        compress(
            client, {"model": "gpt-4o", "messages": history, "config": {"session_id": sid}}
        ).status_code
        == 200
    )
    resp = client.post("/v1/usage", json={"session_id": sid, "usage": usage})
    assert resp.status_code == 200, resp.text
    assert resp.json()["applied"] is True
    return resp.json()["frozen_message_count"]


@pytest.mark.parametrize(
    "usage",
    [ANTHROPIC_WARM, OPENAI_WARM, OPENAI_TOPLEVEL_CACHED, KONG_NESTED],
    ids=["anthropic", "openai", "openai-toplevel-cached", "kong-nested"],
)
def test_usage_shapes_apply_to_tracker_like_v1_usage(
    headroom_client, usage: dict[str, Any]
) -> None:
    """R-USAGE: every accepted shape lands on the tracker and advances the frozen
    count exactly as the equivalent ``/v1/usage`` relay does."""
    history = big_tool_history()
    expected = _frozen_after_legacy_usage(
        headroom_client,
        history,
        {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 50_000},
    )
    turn = _open_turn(headroom_client, session_id="usage-shape")
    resp = _post(headroom_client, {"turn_id": turn["turn_id"], "usage": usage})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["usage_applied"] is True
    assert data["frozen_message_count"] == expected
    assert data["frozen_message_count"] >= 1


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 120, "completion_tokens": 3},
        {"input_tokens": 120, "output_tokens": 3},
        {"prompt_tokens": 120, "prompt_tokens_details": {"cached_tokens": 0}},
        {"input_tokens": 120, "output_tokens": 3, "cache_read_input_tokens": 0},
    ],
    ids=["openai-bare", "anthropic-bare", "openai-cached-zero", "anthropic-read-zero-only"],
)
def test_usage_without_cache_signal_is_accepted_but_not_applied(headroom_client, usage) -> None:
    """R-USAGE: unlike ``/v1/usage`` a signal-free relay is not a 400 - the response
    half still completes the outcome - but it must not touch the tracker (applying
    it would assert a fully-cold prefix the provider never reported)."""
    turn = _open_turn(headroom_client, session_id="usage-nosignal")
    before = turn["session"]["frozen_message_count"]
    resp = _post(headroom_client, {"turn_id": turn["turn_id"], "usage": usage})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["action"] == "done"
    assert data["usage_applied"] is False
    tracker = headroom_client.app.state.proxy.session_tracker_store.peek(
        "compress\x00usage-nosignal"
    )
    assert tracker is not None
    assert tracker.get_frozen_message_count() == data["frozen_message_count"]
    assert data["frozen_message_count"] <= max(before, len(big_tool_history()))


def test_both_cache_fields_zero_is_a_confirmed_cold_turn(headroom_client) -> None:
    """R-USAGE: both fields present and 0 is a genuine fully-cold signal (the same
    rule ``/v1/usage`` applies) and IS applied."""
    turn = _open_turn(headroom_client, session_id="usage-cold")
    resp = _post(
        headroom_client,
        {
            "turn_id": turn["turn_id"],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["usage_applied"] is True


# --------------------------------------------------------------------------- #
# R-OUTCOME                                                                     #
# --------------------------------------------------------------------------- #


def test_outcome_deferred_then_recorded_once_with_provider_numbers(
    headroom_client, outcome_spy
) -> None:
    """R-OUTCOME: with ``relay_usage`` nothing is recorded at the request half; the
    response half records exactly one outcome carrying output/input/cache tokens
    and the gateway-measured latency."""
    outcomes = outcome_spy(headroom_client)
    turn = _open_turn(headroom_client, session_id="outcome-defer")
    assert outcomes == [], "outcome must be deferred while relay_usage is pending"
    resp = _post(
        headroom_client,
        {
            "turn_id": turn["turn_id"],
            "status": 200,
            "latency_ms": 250.0,
            "usage": {
                "prompt_tokens": 1234,
                "completion_tokens": 56,
                "prompt_tokens_details": {"cached_tokens": 1000},
            },
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o.provider == "compress"
    assert o.model == "gpt-4o"
    assert o.output_tokens == 56
    assert o.provider_input_tokens == 1234
    assert o.cache_read_tokens == 1000
    assert o.total_latency_ms >= 250.0
    assert o.original_tokens == turn["tokens_before"]
    assert o.optimized_tokens == turn["tokens_after"]
    assert o.tokens_saved == turn["tokens_saved"]
    assert o.status_code == 200


def test_outcome_anthropic_cache_fields(headroom_client, outcome_spy) -> None:
    """R-OUTCOME: Anthropic read + write buckets land on the outcome's cache fields."""
    outcomes = outcome_spy(headroom_client)
    turn = _open_turn(headroom_client, session_id="outcome-anthropic")
    _post(
        headroom_client,
        {
            "turn_id": turn["turn_id"],
            "usage": {
                "input_tokens": 300,
                "output_tokens": 12,
                "cache_read_input_tokens": 2000,
                "cache_creation_input_tokens": 500,
            },
        },
    )
    assert len(outcomes) == 1
    o = outcomes[0]
    assert o.output_tokens == 12
    assert o.cache_read_tokens == 2000
    assert o.cache_write_tokens == 500
    assert o.provider_input_tokens == 300


def test_outcome_recorded_immediately_without_relay(headroom_client, outcome_spy) -> None:
    """R-OUTCOME: no ``relay_usage`` -> recorded at the request half, as today."""
    outcomes = outcome_spy(headroom_client)
    body = {
        "model": "gpt-4o",
        "messages": big_tool_history(),
        "gateway": {"can_relay_response": False},
    }
    data = compress(headroom_client, body).json()
    assert data["obligations"] == []
    assert len(outcomes) == 1
    assert outcomes[0].output_tokens == 0


def test_outcome_status_from_provider(headroom_client, outcome_spy) -> None:
    """R-OUTCOME: a provider 5xx relayed through the response half is recorded with
    that status so it cannot inflate save-rate (same rule as the proxy path)."""
    outcomes = outcome_spy(headroom_client)
    turn = _open_turn(headroom_client, session_id="outcome-5xx")
    resp = _post(headroom_client, {"turn_id": turn["turn_id"], "status": 529, "latency_ms": 5})
    assert resp.status_code == 200, resp.text
    assert resp.json()["action"] == "done"
    assert len(outcomes) == 1
    assert outcomes[0].status_code == 529


def test_expiry_records_draft_and_then_404(headroom_client, outcome_spy) -> None:
    """R-OUTCOME: a turn the gateway never completes is recorded as-is (output 0)
    when the registry sweeps it, and a late relay is a 404."""
    outcomes = outcome_spy(headroom_client)
    turn = _open_turn(headroom_client, session_id="outcome-expire")
    registry = registry_of(headroom_client)
    assert registry is not None and registry.get(turn["turn_id"]) is not None
    assert outcomes == []
    # In production the sweep runs inside an HTTP handler, i.e. on the server
    # loop, where the draft can be handed to the async outcome funnel. Run it
    # there (the TestClient portal) rather than from the test thread, which has
    # no loop and would only park the draft as an orphan.
    deadline = time.time() + 10_000
    headroom_client.portal.call(lambda: registry.sweep(deadline))
    assert registry.get(turn["turn_id"]) is None
    late = _post(headroom_client, {"turn_id": turn["turn_id"], "usage": {"prompt_tokens": 1}})
    assert late.status_code == 404
    assert late.json()["error"]["type"] == "unknown_turn"
    # The record is scheduled as a task; the request above let the loop run it.
    assert len(outcomes) == 1
    assert outcomes[0].output_tokens == 0
    assert outcomes[0].original_tokens == turn["tokens_before"]


def test_ttl_env_is_honoured(make_headroom_client, monkeypatch, outcome_spy) -> None:
    """R-OUTCOME: ``HEADROOM_GATEWAY_TURN_TTL_SECONDS`` bounds how long a pending
    turn lives; a lazy sweep on the next lookup reclaims it."""
    monkeypatch.setenv("HEADROOM_GATEWAY_TURN_TTL_SECONDS", "0")
    client = make_headroom_client()
    outcomes = outcome_spy(client)
    turn = _open_turn(client, session_id="ttl-zero")
    time.sleep(0.05)
    late = _post(client, {"turn_id": turn["turn_id"], "usage": {"prompt_tokens": 1}})
    assert late.status_code == 404, late.text
    assert len(outcomes) == 1


def test_registry_stats_shape(headroom_client) -> None:
    """R-LOOKUP: ``stats()`` exposes at least the pending count for /stats and debugging."""
    registry = registry_of(headroom_client)
    assert registry is not None
    turn = _open_turn(headroom_client, session_id="stats")
    stats = registry.stats()
    assert isinstance(stats, dict)
    assert stats.get("pending") == len(registry) >= 1
    _post(headroom_client, {"turn_id": turn["turn_id"]})
    assert registry.stats().get("pending") == len(registry)
