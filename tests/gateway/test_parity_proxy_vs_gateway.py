"""Parity: the proxy path and the gateway two-half path forward the same bytes.

The same scripted 3-turn conversation (large tool result, tools, session) runs
(a) through ``/v1/chat/completions`` with the proxy's upstream bridged into a
:class:`FakeProvider`, and (b) through :class:`FakeGateway` + ``/v1/compress`` +
``/v1/compress/response`` against a second :class:`FakeProvider`. Per turn the
provider-bound ``messages`` and ``tools`` must be equal and the recorded outcome
token totals must agree within a declared tolerance.

Known / expected divergences (each named at the assertion it would break):

* FREEZE POLICY - the proxy path freezes with ``FREEZE_POLICY_CONFIRMED_CLAMP``
  (never past what the provider confirmed via ``cached_tokens``) while the
  compress path uses ``FREEZE_POLICY_REPLAYABLE`` (everything it already
  returned). Both then overlay the previously forwarded prefix, so the BYTES are
  expected to match; what differs is how much the pipeline is allowed to re-touch
  before the overlay. If bytes ever diverge, this stage is the first suspect.
* TOKENIZER SCALE - the chat path counts with ``openai_pipeline``'s tokenizer,
  the compress path with the per-model registry tokenizer. Both are tiktoken for
  gpt-4o, so ``tokens_saved`` should match closely; the tolerance below absorbs
  small estimator deltas, not structural ones.
* OUTCOME PROVIDER - the proxy path records ``provider="openai"``; the gateway
  path records ``provider="compress"``. Compared by position, not by name.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from tests.gateway.conftest import proxy_config
from tests.gateway.fake_gateway import FakeGateway
from tests.gateway.fake_provider import FakeProvider, openai_text_response, openai_usage
from tests.gateway.samples import big_tool_history, canonical, openai_tools

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import create_app  # noqa: E402

OPENAI_UPSTREAM = "https://api.openai.com/v1/chat/completions"
SAVED_TOLERANCE_ABS = 32
SAVED_TOLERANCE_REL = 0.10

# The provider's answers are identical on both paths, turn for turn. Turn 1 is
# cold; turns 2 and 3 report a warm prefix so the trackers on both sides see the
# same confirmation signal.
_SCRIPT = [
    openai_text_response("Listed.", openai_usage(900, 3, 0), response_id="t1"),
    openai_text_response("Sorted.", openai_usage(1000, 3, 896), response_id="t2"),
    openai_text_response("Top 3 are 0, 30, 60.", openai_usage(1100, 5, 1000), response_id="t3"),
]


def _client_turns() -> list[list[dict[str, Any]]]:
    h = big_tool_history()
    t1 = h
    t2 = t1 + [
        {"role": "assistant", "content": "Listed."},
        {"role": "user", "content": "Sort by score."},
    ]
    t3 = t2 + [{"role": "assistant", "content": "Sorted."}, {"role": "user", "content": "Top 3?"}]
    return [t1, t2, t3]


def _body(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {"model": "gpt-4o", "messages": messages, "tools": openai_tools(), "temperature": 0}


def _make_app_client() -> TestClient:
    # CCR off on the proxy path: the chat handler would otherwise inject the
    # ``headroom_retrieve`` tool and CCR markers, which the compress path's
    # default (marker-free) mode never does. Parity is measured with both paths
    # in marker-free mode.
    app = create_app(
        proxy_config(
            ccr_inject_tool=False,
            ccr_inject_marker=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345))


def _bridge(provider: FakeProvider):
    """respx side effect: hand the proxy's upstream request to the fake provider."""

    def _side_effect(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        reply = provider.handle("/v1/chat/completions", body, dict(request.headers))
        return httpx.Response(reply.status, content=reply.content, headers=reply.headers)

    return _side_effect


def _run_proxy_path(client: TestClient, provider: FakeProvider) -> list[dict[str, Any]]:
    provider.script(list(_SCRIPT))
    finals = []
    for messages in _client_turns():
        resp = client.post(
            "/v1/chat/completions",
            json=_body(messages),
            headers={"authorization": "Bearer sk-test", "x-headroom-session-id": "parity"},
        )
        assert resp.status_code == 200, resp.text
        finals.append(resp.json())
    assert [f["id"] for f in finals] == ["t1", "t2", "t3"]
    return [c.body for c in provider.calls]


def _run_gateway_path(client: TestClient, provider: FakeProvider) -> list[dict[str, Any]]:
    provider.script(list(_SCRIPT))
    with TestClient(provider.app) as provider_client:
        gateway = FakeGateway(client, provider_client, can_redrive=True, can_relay_response=True)
        for messages in _client_turns():
            result = gateway.turn(_body(messages), "parity-gw")
            assert result.path == "proxy", (result.path, result.fail_open_reason)
            assert result.compress_status == 200
            assert "body" in result.compress_response, "gateway contract keys missing"
            relay = result.response_half_calls[-1]
            assert relay.status == 200, relay.response
    assert [t.final_response["id"] for t in gateway.turns] == ["t1", "t2", "t3"]
    return [c.body for c in provider.calls]


@respx.mock
def test_provider_bound_bytes_and_outcomes_match(monkeypatch) -> None:
    """Per turn: provider-bound ``messages`` and ``tools`` are identical on both
    paths, and the outcomes agree on provider-reported tokens and (within
    tolerance) on ``tokens_saved``."""
    proxy_provider, gateway_provider = FakeProvider(), FakeProvider()
    respx.post(OPENAI_UPSTREAM).mock(side_effect=_bridge(proxy_provider))

    outcomes: list[Any] = []
    with _make_app_client() as client:
        proxy = client.app.state.proxy

        async def _spy(outcome: Any, *a: Any, **k: Any) -> None:
            outcomes.append(outcome)

        monkeypatch.setattr(proxy, "_record_request_outcome", _spy)

        proxy_calls = _run_proxy_path(client, proxy_provider)
        gateway_calls = _run_gateway_path(client, gateway_provider)

    assert len(proxy_calls) == 3 and len(gateway_calls) == 3

    for i, (p, g) in enumerate(zip(proxy_calls, gateway_calls)):
        # Stage: session freeze/overlay (FREEZE POLICY divergence would show here).
        assert canonical(p["messages"]) == canonical(g["messages"]), (
            f"turn {i + 1}: messages differ"
        )
        # Stage: tool schema compaction (shared cache keyed by tool bytes).
        assert canonical(p["tools"]) == canonical(g["tools"]), f"turn {i + 1}: tools differ"
        assert p["model"] == g["model"] == "gpt-4o"
        assert p["temperature"] == g["temperature"] == 0

    # Both paths compressed something, otherwise parity is vacuous.
    raw = big_tool_history()[2]["content"]
    assert proxy_calls[0]["messages"][2]["content"] != raw

    proxy_outcomes = [o for o in outcomes if o.provider == "openai"]
    gateway_outcomes = [o for o in outcomes if o.provider == "compress"]
    assert len(proxy_outcomes) == 3, [o.provider for o in outcomes]
    assert len(gateway_outcomes) == 3, [o.provider for o in outcomes]
    for i, (po, go) in enumerate(zip(proxy_outcomes, gateway_outcomes)):
        # Stage: response half outcome completion (provider numbers, exact).
        assert po.output_tokens == go.output_tokens == _SCRIPT[i]["usage"]["completion_tokens"], (
            f"turn {i + 1}"
        )
        assert (
            po.provider_input_tokens
            == go.provider_input_tokens
            == _SCRIPT[i]["usage"]["prompt_tokens"]
        ), f"turn {i + 1}"
        assert po.cache_read_tokens == go.cache_read_tokens, f"turn {i + 1}"
        # Stage: tokenizer scale (declared tolerance, see module docstring).
        tolerance = max(
            SAVED_TOLERANCE_ABS, int(SAVED_TOLERANCE_REL * max(po.tokens_saved, go.tokens_saved))
        )
        assert abs(po.tokens_saved - go.tokens_saved) <= tolerance, (
            f"turn {i + 1}: tokens_saved proxy={po.tokens_saved} gateway={go.tokens_saved}"
        )
        assert abs(po.original_tokens - go.original_tokens) <= tolerance, (
            f"turn {i + 1}: original_tokens"
        )


@respx.mock
def test_proxy_path_alone_is_byte_stable_baseline() -> None:
    """The proxy-path half of the parity fixture, checked on its own so a
    gateway-side failure cannot hide a broken baseline (passes today)."""
    provider = FakeProvider()
    respx.post(OPENAI_UPSTREAM).mock(side_effect=_bridge(provider))
    with _make_app_client() as client:
        calls = _run_proxy_path(client, provider)
    assert len(calls) == 3
    for prev, nxt in zip(calls, calls[1:]):
        assert canonical(nxt["messages"][: len(prev["messages"])]) == canonical(prev["messages"])
        assert canonical(nxt["tools"]) == canonical(prev["tools"])
