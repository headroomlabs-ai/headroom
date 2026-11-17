"""The compress-turn seam: a contract on ``/v1/compress`` that is not the
built-in gateway contract, installed exactly as a third party would.

* I-CLAIM   - ``begin`` sees every body; a claimed body gets ``prepare`` ->
              ``transform`` (executor side, once) -> ``finish`` -> ``commit``.
* I-FIELDS  - ``finish``'s fields and transform labels reach the answer and
              the outcome; a replaced message list is what the answer carries.
* I-OWN     - a turn that returns ``True`` from ``commit`` owns the outcome:
              the proxy does not record it.
* I-400     - ``CompressTurnError`` from ``begin`` is a 400 ``invalid_request``.
* I-FAILOPEN- bypass header, empty messages and a compression timeout carry
              the turn's ``fail_open_fields``.
* I-FIRST   - the first registered claimant wins; the built-in gateway
              contract still claims ``gateway`` bodies on the same app.
* I-OFF     - ``HEADROOM_GATEWAY_CONTRACT=0`` builds an app with no gateway
              contract: a ``gateway`` block is ignored, the response route is
              absent, and the seam is empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from headroom.proxy.compress_turn import (
    CompressTurnError,
    FinishedTurn,
    register_compress_turn_extension,
    registered_compress_turn_extensions,
    unregister_compress_turn_extension,
)
from tests.gateway.conftest import compress
from tests.gateway.samples import anthropic_tool_history


@dataclass
class EchoTurn:
    model_name: str = ""
    transform_calls: int = 0
    prepared: bool = False
    committed: list[Any] = field(default_factory=list)
    replace_messages: bool = False

    def fail_open_fields(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"echo": {"fail_open": True, "n": len(messages)}}

    def prepare(self, *, model_name: str, tags: dict[str, Any], config: Any) -> None:
        self.prepared = True
        self.model_name = model_name
        tags["echo_prepared"] = True

    def transform(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.transform_calls += 1
        return messages  # pure

    @property
    def folded_messages(self) -> bool:
        return False

    def count_messages(self, messages: list[dict[str, Any]], fallback: int) -> int:
        return fallback

    def finish(
        self,
        *,
        messages: list[dict[str, Any]],
        tokens_before: int,
        transforms_applied: list[str],
        mode: str | None,
        ccr_hashes: list[str],
    ) -> FinishedTurn:
        out = None
        if self.replace_messages:
            out = [{"role": "user", "content": "replaced by echo"}]
        return FinishedTurn(
            fields={"echo": {"model": self.model_name, "tokens_before": tokens_before}},
            transforms=list(transforms_applied) + ["echo:seen"],
            messages=out,
        )

    def commit(self, outcome: Any, *, session_key: str | None, session_id: str | None) -> bool:
        self.committed.append(outcome)
        return True  # we own it


@dataclass
class EchoContract:
    name: str = "echo_contract"
    turns: list[EchoTurn] = field(default_factory=list)
    replace_messages: bool = False

    def begin(self, *, proxy: Any, request: Any, body: dict[str, Any], client: str | None):
        if body.get("contract") != "echo":
            return None
        if not isinstance(body.get("contract_opts", {}), dict):
            raise CompressTurnError("contract_opts must be an object.")
        turn = EchoTurn(replace_messages=self.replace_messages)
        self.turns.append(turn)
        return turn


@pytest.fixture
def echo(headroom_client):
    proxy = headroom_client.app.state.proxy
    ext = EchoContract()
    register_compress_turn_extension(proxy, ext)
    yield ext
    unregister_compress_turn_extension(proxy, ext)


def _body(**extra: Any) -> dict[str, Any]:
    out = {"model": "claude-sonnet-4-5", "messages": anthropic_tool_history(), "max_tokens": 32}
    out.update(extra)
    return out


def test_built_in_gateway_contract_is_registered_through_the_seam(headroom_client) -> None:
    proxy = headroom_client.app.state.proxy
    names = [getattr(e, "name", None) for e in registered_compress_turn_extensions(proxy)]
    assert names == ["gateway_turn_contract"]
    assert proxy.gateway_turns is not None


def test_claimed_body_runs_every_moment_and_owns_the_outcome(
    headroom_client, outcome_spy, echo
) -> None:
    outcomes = outcome_spy(headroom_client)
    resp = compress(headroom_client, _body(contract="echo"))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["echo"]["model"] == "claude-sonnet-4-5"
    assert data["echo"]["tokens_before"] == data["tokens_before"]
    assert data["transforms_applied"][-1] == "echo:seen"
    assert len(echo.turns) == 1
    turn = echo.turns[0]
    assert turn.prepared and turn.transform_calls == 1
    # I-OWN: the turn took the outcome; the proxy recorded nothing.
    assert outcomes == []
    assert len(turn.committed) == 1
    assert "echo:seen" in turn.committed[0].transforms_applied
    assert turn.committed[0].tags.get("echo_prepared") is True


def test_claimed_session_body_transforms_once_after_replay(headroom_client, echo) -> None:
    body = _body(contract="echo", config={"session_id": "echo-session"})
    first = compress(headroom_client, body).json()
    grown = dict(body)
    grown["messages"] = list(body["messages"]) + [
        {"role": "assistant", "content": "answer 0"},
        {"role": "user", "content": "follow-up 0"},
    ]
    second = compress(headroom_client, grown).json()
    assert first["session"]["id"] == second["session"]["id"] == "echo-session"
    assert [t.transform_calls for t in echo.turns] == [1, 1]
    assert second["echo"]["model"] == "claude-sonnet-4-5"


def test_finish_can_replace_the_returned_messages(headroom_client, echo) -> None:
    echo.replace_messages = True
    data = compress(headroom_client, _body(contract="echo")).json()
    assert data["messages"] == [{"role": "user", "content": "replaced by echo"}]


def test_unclaimed_body_is_legacy(headroom_client, outcome_spy, echo) -> None:
    outcomes = outcome_spy(headroom_client)
    data = compress(headroom_client, _body()).json()
    assert "echo" not in data and "body" not in data
    assert echo.turns == []
    assert len(outcomes) == 1


def test_begin_error_is_400(headroom_client, echo) -> None:
    resp = compress(headroom_client, _body(contract="echo", contract_opts="nope"))
    assert resp.status_code == 400
    assert resp.json()["error"] == {
        "type": "invalid_request",
        "message": "contract_opts must be an object.",
    }


def test_fail_open_answers_carry_the_turns_fields(headroom_client, echo, monkeypatch) -> None:
    bypass = compress(
        headroom_client, _body(contract="echo"), headers={"x-headroom-bypass": "true"}
    ).json()
    assert bypass["echo"] == {"fail_open": True, "n": len(anthropic_tool_history())}

    empty = compress(headroom_client, _body(contract="echo", messages=[])).json()
    assert empty["echo"] == {"fail_open": True, "n": 0}

    from unittest.mock import AsyncMock

    proxy = headroom_client.app.state.proxy
    monkeypatch.setattr(proxy, "_run_compression_in_executor", AsyncMock(side_effect=TimeoutError))
    timed_out = compress(headroom_client, _body(contract="echo")).json()
    assert timed_out["compression_skipped"] is True
    assert timed_out["echo"]["fail_open"] is True


def test_first_claimant_wins_and_gateway_still_claims_its_bodies(headroom_client, echo) -> None:
    # The built-in contract was registered first (at create_app), so a body
    # carrying BOTH markers goes to it and the echo contract never sees it.
    data = compress(headroom_client, _body(contract="echo", gateway={})).json()
    assert "body" in data and "turn_id" in data
    assert "echo" not in data
    assert echo.turns == []


def test_contract_disabled_by_env(make_headroom_client, monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CONTRACT", "0")
    client = make_headroom_client()
    proxy = client.app.state.proxy
    assert registered_compress_turn_extensions(proxy) == []
    assert proxy.gateway_turns is None
    data = compress(client, _body(gateway={"can_relay_response": True})).json()
    assert "body" not in data and "turn_id" not in data and "obligations" not in data
    resp = client.post("/v1/compress/response", json={"turn_id": "x", "status": 200})
    assert resp.status_code in (404, 405)
