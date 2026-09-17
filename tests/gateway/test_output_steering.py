"""Output shaping (verbosity steering) on the gateway contract (request-only).

Mirrors the chat handlers: gated by the ``proxy_output_shaper`` rollout
(``HEADROOM_OUTPUT_SHAPER``), ``steering_allowed_for`` (off in
``mode="cache"``), the conversation-stable ``HEADROOM_OUTPUT_HOLDOUT`` arm and
``resolve_verbosity_level``. Invariants protected here:

* I-SHAPE    - enabled -> the steering block is appended to the Anthropic
               ``system`` / the OpenAI chat system message of the provider
               body, the (arm, stratum) label and the verbosity label are in
               ``transforms_applied``, and top-level ``messages`` is still
               ``body.messages``.
* I-GATES    - disabled (default), ``mode="cache"`` (label only, no block),
               control arm (label only, no block), legacy mode (nothing).
* I-STABLE   - the block is byte-stable across three session turns and never
               double-appended.
* I-MEASURE  - the label rides the DEFERRED outcome: the response half's
               relayed ``output_tokens`` reach the outcome funnel with the
               label, ``record_from_labels`` sees them, and with a baseline
               for the stratum ``metrics.record_request`` is fed
               ``output_tokens_saved > 0``.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.proxy.output_savings import SavingsRecorder
from headroom.proxy.output_savings_policy import parse_stratum_label
from headroom.proxy.output_verbosity_policy import STEERING_SENTINEL
from tests.gateway.conftest import compress
from tests.gateway.fake_provider import anthropic_usage, openai_usage
from tests.gateway.samples import (
    SYSTEM_PROMPT,
    anthropic_tool_history,
    anthropic_tools,
    big_tool_history,
    canonical,
    openai_tools,
)

CLAUDE = "claude-sonnet-4-5"
STRATUM_PREFIX = "output_shaper:stratum:"
CONTROL_PREFIX = "output_shaper:control:"
VERBOSITY_PREFIX = "output_shaper:verbosity:L"


def _anthropic_body(**extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": CLAUDE,
        "messages": anthropic_tool_history(),
        "system": SYSTEM_PROMPT,
        "tools": anthropic_tools(),
        "max_tokens": 64,
    }
    out.update(extra)
    return out


def _openai_body(*, with_system: bool = True, **extra: Any) -> dict[str, Any]:
    messages = big_tool_history()
    if with_system:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    out: dict[str, Any] = {"model": "gpt-4o", "messages": messages, "tools": openai_tools()}
    out.update(extra)
    return out


def _shaper_labels(data: dict[str, Any]) -> list[str]:
    return [t for t in data["transforms_applied"] if str(t).startswith("output_shaper:")]


def _stratum_label(data: dict[str, Any]) -> str:
    labels = [t for t in data["transforms_applied"] if parse_stratum_label(str(t))]
    assert len(labels) == 1, data["transforms_applied"]
    return labels[0]


def _anthropic_steering_blocks(body: dict[str, Any]) -> list[str]:
    system = body.get("system")
    if not isinstance(system, list):
        return []
    return [
        b["text"]
        for b in system
        if isinstance(b, dict) and str(b.get("text", "")).startswith(STEERING_SENTINEL)
    ]


def _openai_system_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in body["messages"] if m.get("role") in ("system", "developer")]


@pytest.fixture
def shaped_client(make_headroom_client, monkeypatch):
    """A proxy built with the shaper enabled (the rollout snapshot is taken at
    config creation, so the env must be set BEFORE the app exists)."""
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.delenv("HEADROOM_OUTPUT_HOLDOUT", raising=False)
    monkeypatch.delenv("HEADROOM_VERBOSITY_LEVEL", raising=False)
    return make_headroom_client()


# --------------------------------------------------------------------------- #
# I-SHAPE                                                                      #
# --------------------------------------------------------------------------- #


def test_anthropic_system_gets_steering_block(shaped_client) -> None:
    sent = _anthropic_body(gateway={})
    resp = compress(shaped_client, sent)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    body = data["body"]
    assert isinstance(body["system"], list)
    assert body["system"][0] == {"type": "text", "text": SYSTEM_PROMPT}
    blocks = _anthropic_steering_blocks(body)
    assert len(blocks) == 1
    label = _stratum_label(data)
    assert label.startswith(STRATUM_PREFIX)
    arm, key = parse_stratum_label(label)
    assert arm == "treatment"
    assert key.startswith("sonnet|") and key.endswith("|tools")
    assert any(t.startswith(VERBOSITY_PREFIX) for t in data["transforms_applied"])
    assert data["messages"] == body["messages"]
    # Messages themselves are not where Anthropic steering goes.
    assert canonical(body["messages"]) == canonical(data["messages"])


def test_anthropic_without_system_field_gets_one(shaped_client) -> None:
    sent = _anthropic_body(gateway={})
    del sent["system"]
    data = compress(shaped_client, sent).json()
    blocks = _anthropic_steering_blocks(data["body"])
    assert len(blocks) == 1 and len(data["body"]["system"]) == 1


def test_openai_system_message_gets_steering_block(shaped_client) -> None:
    sent = _openai_body(gateway={})
    resp = compress(shaped_client, sent)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    body = data["body"]
    system = _openai_system_messages(body)
    assert len(system) == 1
    assert system[0]["content"].startswith(SYSTEM_PROMPT)
    assert system[0]["content"].count(STEERING_SENTINEL) == 1
    assert len(body["messages"]) == len(sent["messages"])
    assert data["messages"] == body["messages"]
    label = _stratum_label(data)
    assert label.startswith(STRATUM_PREFIX)
    assert parse_stratum_label(label)[1].startswith("gpt|")
    assert any(t.startswith(VERBOSITY_PREFIX) for t in data["transforms_applied"])


def test_openai_without_system_message_gets_one_inserted(shaped_client) -> None:
    sent = _openai_body(with_system=False, gateway={})
    data = compress(shaped_client, sent).json()
    body = data["body"]
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"].startswith(STEERING_SENTINEL)
    assert len(body["messages"]) == len(sent["messages"]) + 1
    assert data["messages"] == body["messages"]


def test_shaped_system_does_not_leak_into_session_snapshot(shaped_client) -> None:
    """The steering block is appended on the provider body only: a second
    turn re-appends it once, never twice, and the pre-steering prefix the
    session snapshot holds stays byte-identical."""
    session = {"session_id": "gw-shape-snapshot"}
    sent1 = _openai_body(config=session, gateway={})
    turn1 = compress(shaped_client, sent1).json()
    sent2 = _openai_body(config=session, gateway={})
    sent2["messages"] += [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "more"},
    ]
    turn2 = compress(shaped_client, sent2).json()
    assert canonical(turn2["body"]["messages"][: len(turn1["body"]["messages"])]) == canonical(
        turn1["body"]["messages"]
    )
    assert turn2["body"]["messages"][0]["content"].count(STEERING_SENTINEL) == 1


# --------------------------------------------------------------------------- #
# I-GATES                                                                      #
# --------------------------------------------------------------------------- #


def test_disabled_by_default_leaves_body_untouched(headroom_client) -> None:
    for sent in (_anthropic_body(gateway={}), _openai_body(gateway={})):
        data = compress(headroom_client, sent).json()
        assert _shaper_labels(data) == []
        body = data["body"]
        if "system" in sent:
            assert body["system"] == SYSTEM_PROMPT
        else:
            assert body["messages"][0] == sent["messages"][0]
        assert STEERING_SENTINEL not in canonical(body)


def test_env_off_wins(make_headroom_client, monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "0")
    client = make_headroom_client()
    data = compress(client, _anthropic_body(gateway={})).json()
    assert _shaper_labels(data) == []
    assert data["body"]["system"] == SYSTEM_PROMPT


def test_cache_mode_labels_but_never_steers(make_headroom_client, monkeypatch) -> None:
    """``steering_allowed_for`` is false in ``mode="cache"``: the measurement
    label still rides the outcome but the prefix-cache key is left alone."""
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    client = make_headroom_client(mode="cache")
    for sent in (_anthropic_body(gateway={}), _openai_body(gateway={})):
        data = compress(client, sent).json()
        assert _stratum_label(data).startswith(STRATUM_PREFIX)
        assert not any(t.startswith(VERBOSITY_PREFIX) for t in data["transforms_applied"])
        assert STEERING_SENTINEL not in canonical(data["body"])


def test_control_arm_labels_but_never_steers(make_headroom_client, monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "1")
    client = make_headroom_client()
    data = compress(client, _anthropic_body(gateway={})).json()
    label = _stratum_label(data)
    assert label.startswith(CONTROL_PREFIX)
    assert data["body"]["system"] == SYSTEM_PROMPT
    assert not any(t.startswith(VERBOSITY_PREFIX) for t in data["transforms_applied"])


def test_legacy_mode_is_never_shaped(shaped_client, outcome_spy) -> None:
    outcomes = outcome_spy(shaped_client)
    sent = _openai_body()  # no gateway block
    data = compress(shaped_client, sent).json()
    assert "body" not in data
    assert _shaper_labels(data) == []
    assert data["messages"][0] == sent["messages"][0]
    assert not any(str(t).startswith("output_shaper:") for t in outcomes[0].transforms_applied)


def test_fail_open_answers_are_never_shaped(shaped_client, monkeypatch) -> None:
    from unittest.mock import AsyncMock

    proxy = shaped_client.app.state.proxy
    monkeypatch.setattr(proxy, "_run_compression_in_executor", AsyncMock(side_effect=TimeoutError))
    data = compress(shaped_client, _anthropic_body(gateway={})).json()
    assert data["compression_skipped"] is True
    assert data["body"]["system"] == SYSTEM_PROMPT
    assert _shaper_labels(data) == []


# --------------------------------------------------------------------------- #
# I-STABLE                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_steering_is_byte_stable_over_three_session_turns(shaped_client, provider: str) -> None:
    session = {"session_id": f"gw-shape-stable-{provider}"}
    prefixes: list[str] = []
    labels: list[str] = []
    prev: list[dict[str, Any]] | None = None
    for turn in range(3):
        sent = (
            _anthropic_body(config=session, gateway={})
            if provider == "anthropic"
            else _openai_body(config=session, gateway={})
        )
        sent["messages"] = sent["messages"] + [
            m
            for i in range(turn)
            for m in (
                {"role": "assistant", "content": f"answer {i}"},
                {"role": "user", "content": f"follow-up {i}"},
            )
        ]
        resp = compress(shaped_client, sent)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        body = data["body"]
        if provider == "anthropic":
            prefixes.append(canonical(body["system"]))
            assert len(_anthropic_steering_blocks(body)) == 1
        else:
            prefixes.append(canonical(body["messages"][0]))
            assert body["messages"][0]["content"].count(STEERING_SENTINEL) == 1
        labels.append(_stratum_label(data))
        if prev is not None:
            assert canonical(body["messages"][: len(prev)]) == canonical(prev)
        prev = body["messages"]
    assert len(set(prefixes)) == 1
    # The label is measurement, not prefix bytes: its turn-kind field follows
    # the tail of the conversation (tool-result continuation vs new ask), so
    # only the arm and the model/tools strata are expected to hold steady.
    parsed = [parse_stratum_label(label) for label in labels]
    assert {arm for arm, _ in parsed} == {"treatment"}
    assert len({key.split("|")[0] for _, key in parsed}) == 1
    assert {key.split("|")[-1] for _, key in parsed} == {"tools"}


# --------------------------------------------------------------------------- #
# I-MEASURE                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_label_rides_the_deferred_outcome_and_yields_output_savings(
    shaped_client, monkeypatch, tmp_path, provider: str
) -> None:
    """Kong-shaped run: request half with ``relay_usage`` owed, response half
    with the provider's usage. The recorded outcome carries the stratum label
    and the relayed output tokens; the recorder observes it; and with a
    baseline for the stratum the outcome funnel hands the savings tracker a
    positive ``output_tokens_saved``."""
    proxy = shaped_client.app.state.proxy
    recorded: list[dict[str, Any]] = []
    original_record = proxy.metrics.record_request

    async def _metrics_spy(*args: Any, **kwargs: Any) -> Any:
        recorded.append(kwargs)
        return await original_record(*args, **kwargs)

    monkeypatch.setattr(proxy.metrics, "record_request", _metrics_spy)

    outcomes: list[Any] = []
    original_outcome = proxy._record_request_outcome

    async def _outcome_spy(outcome: Any, *args: Any, **kwargs: Any) -> Any:
        outcomes.append(outcome)
        return await original_outcome(outcome, *args, **kwargs)

    monkeypatch.setattr(proxy, "_record_request_outcome", _outcome_spy)

    recorder = SavingsRecorder(tmp_path / "output_savings.json", flush_every=10_000)
    monkeypatch.setattr("headroom.proxy.output_savings._RECORDER", recorder)

    sent = (
        _anthropic_body(config={"session_id": "kong-shape-a"})
        if provider == "anthropic"
        else _openai_body(config={"session_id": "kong-shape-o"})
    )
    sent["gateway"] = {"can_relay_response": True, "plugin_version": "kong-headroom/0.1.0"}
    half1 = shaped_client.post("/v1/compress", json=sent)
    assert half1.status_code == 200, half1.text
    data = half1.json()
    assert data["obligations"] == ["relay_usage"]
    label = _stratum_label(data)
    arm, key = parse_stratum_label(label)
    assert arm == "treatment"
    assert outcomes == [] and recorded == []  # deferred until the response half

    # A synthetic-control baseline for exactly this stratum: 1000 output
    # tokens unshaped; the shaped turn below reports 100.
    for _ in range(5):
        recorder._ledger.baseline.observe(key, 1000)

    usage = (
        anthropic_usage(900, 100, cache_read=0, cache_write=800)
        if provider == "anthropic"
        else openai_usage(900, 100, cached=0)
    )
    half2 = shaped_client.post(
        "/v1/compress/response",
        json={"turn_id": data["turn_id"], "status": 200, "latency_ms": 5, "usage": usage},
    )
    assert half2.status_code == 200, half2.text
    assert half2.json()["action"] == "done"

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert label in outcome.transforms_applied
    assert outcome.output_tokens == 100
    # The recorder saw the shaped observation under the treatment arm ...
    treatment = recorder._ledger.treatment.get(key)
    assert treatment is not None and treatment.n == 1
    # ... and the savings tracker was handed the counterfactual delta.
    assert len(recorded) == 1
    assert recorded[0]["output_tokens"] == 100
    assert recorded[0]["output_tokens_saved"] == 900
