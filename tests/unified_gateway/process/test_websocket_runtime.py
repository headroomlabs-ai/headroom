"""Responses turn accounting across actual client and TLS provider sockets."""

import json
import threading

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from tests.unified_gateway.process.http_harness import http_process
from tests.unified_gateway.test_gateway_tls import local_pki  # noqa: F401
from tests.unified_gateway.test_ws_lifecycle import completed

CREATE = json.dumps(
    {"type": "response.create", "response": {"model": "fixture-model", "input": "hello"}}
)
ADMIN = {"authorization": "Bearer operator-secret"}


def configure(raw):
    raw["client_auth"]["principals"].append(
        {"id": "operator", "secret_ref": "env:OPERATOR_TOKEN", "scopes": ["admin"], "routes": []}
    )
    raw["limits"].update(
        shutdown_drain_seconds=0.08, shutdown_cleanup_seconds=1, websocket_idle_seconds=3
    )
    raw["routes"][0]["capabilities"]["openai-responses"]["websocket"] = {
        "features": ["text", "tools"]
    }


def socket(client):
    return connect(
        str(client.base_url).replace("http://", "ws://").rstrip("/") + "/v1/responses",
        additional_headers={"authorization": "Bearer client-secret"},
        proxy=None,
        close_timeout=1,
    )


def test_second_turn_exhausted_budget_is_not_forwarded(local_pki, tmp_path):  # noqa: F811
    frames = []
    closed = threading.Event()

    def provider(ws):
        try:
            for frame in ws:
                frames.append(frame)
                ws.send(completed())
        finally:
            closed.set()

    def policy(raw):
        configure(raw)
        raw["admission"] = {"budget_usd": "0.000030", "unknown_cost_policy": "block"}

    with http_process(
        local_pki, tmp_path, lambda _: None, configure=policy, websocket_handler=provider
    ) as (client, _, _, _):
        with socket(client) as ws:
            ws.send(CREATE)
            assert ws.recv(timeout=5) == completed()
            ws.send(CREATE)
            assert json.loads(ws.recv(timeout=5))["error"]["code"] == "gateway_admission_budget"
        assert closed.wait(3)
        probe = client.get("/__test/idle", headers=ADMIN).json()
        assert len(frames) == 1
        assert probe["ledger"]["known_micro_usd"] == 7
        assert probe["totals"]["logical_requests"] == 2
        assert probe["totals"]["attempts"] == 1
        assert probe["active"] == probe["queued"] == probe["owned"] == 0


def test_accepted_frame_disconnect_never_replays(local_pki, tmp_path):  # noqa: F811
    frames = []

    def provider(ws):
        frames.append(ws.recv())
        ws.close()

    with http_process(
        local_pki, tmp_path, lambda _: None, configure=configure, websocket_handler=provider
    ) as (client, _, _, _):
        with socket(client) as ws:
            ws.send(CREATE)
            assert json.loads(ws.recv(timeout=5))["type"] == "error"
        with socket(client):
            pass  # A new caller socket must not recreate the previous turn.
        probe = client.get("/__test/idle", headers=ADMIN).json()
        assert len(frames) == 1
        assert probe["ledger"]["unknown_charge_count"] == 1
        assert probe["ledger"]["unresolved_micro_usd"] == 30
        assert probe["totals"]["attempts"] == 1


@pytest.mark.parametrize("selector", ["principal_id", "route_id", "account_id", "grant"])
def test_revoke_closes_active_turn(local_pki, tmp_path, selector):  # noqa: F811
    accepted, closed = threading.Event(), threading.Event()
    frames = []

    def provider(ws):
        try:
            frames.append(ws.recv())
            accepted.set()
            for frame in ws:
                frames.append(frame)
        finally:
            closed.set()

    def policy(raw):
        configure(raw)
        if selector == "grant":
            spare = dict(raw["routes"][0], id="spare", public_model="spare")
            raw["routes"].append(spare)
            raw["client_auth"]["principals"][0]["routes"].append("spare")

    with http_process(
        local_pki, tmp_path, lambda _: None, configure=policy, websocket_handler=provider
    ) as (client, _, _, raw):
        with socket(client) as ws:
            ws.send(CREATE)
            assert accepted.wait(5)
            if selector == "grant":
                raw["client_auth"]["principals"][0]["routes"] = ["spare"]
                (tmp_path / "http-runtime.json").write_text(json.dumps(raw))
                response = client.post("/admin/gateway/reload", headers=ADMIN)
            else:
                value = {
                    "principal_id": raw["client_auth"]["principals"][0]["id"],
                    "route_id": raw["routes"][0]["id"],
                    "account_id": raw["credentials"][0]["id"],
                }[selector]
                response = client.post(
                    "/admin/gateway/revoke", headers=ADMIN, json={selector: value}
                )
            assert response.status_code == 200, response.text
            with pytest.raises(ConnectionClosed):
                ws.recv(timeout=3)
        assert closed.wait(3)
        probe = client.get("/__test/idle", headers=ADMIN).json()
        assert len(frames) == 1
        assert probe["ledger"]["unknown_charge_count"] == 1
        assert probe["active"] == probe["queued"] == probe["owned"] == 0


def test_next_turn_captures_new_tariff_without_migrating_session(local_pki, tmp_path):  # noqa: F811
    frames, connections = [], []

    def provider(ws):
        connections.append(ws)
        for frame in ws:
            frames.append(frame)
            ws.send(completed(f"resp_{len(frames)}"))

    with http_process(
        local_pki, tmp_path, lambda _: None, configure=configure, websocket_handler=provider
    ) as (client, _, _, raw):
        with socket(client) as ws:
            ws.send(CREATE)
            ws.recv(timeout=5)
            raw["routes"][0]["pricing"].update(revision="b", output_usd_per_million="5")
            (tmp_path / "http-runtime.json").write_text(json.dumps(raw))
            assert client.post("/admin/gateway/reload", headers=ADMIN).json()["applied"]
            ws.send(CREATE)
            ws.recv(timeout=5)
        probe = client.get("/__test/idle", headers=ADMIN).json()
        assert len(connections) == 1
        assert [op["generation"] for op in probe["operations"]] == [1, 2]
        assert [op["tariff"] for op in probe["operations"]] == ["fixture", "b"]
        assert probe["ledger"]["known_micro_usd"] == 20


@pytest.mark.parametrize("selector", ["principal_id", "route_id", "account_id"])
def test_revocation_cancels_websocket_source_acquisition(local_pki, tmp_path, selector):  # noqa: F811
    with http_process(
        local_pki, tmp_path, lambda _: None, configure=configure, acquire_barrier=True
    ) as (client, calls, _, raw):
        with socket(client) as ws:
            ws.send(CREATE)
            assert client.post("/__test/acquire-entered").json()["entered"]
            value = {
                "principal_id": raw["client_auth"]["principals"][0]["id"],
                "route_id": raw["routes"][0]["id"],
                "account_id": raw["credentials"][0]["id"],
            }[selector]
            assert (
                client.post(
                    "/admin/gateway/revoke", headers=ADMIN, json={selector: value}
                ).status_code
                == 200
            )
            probe = client.get("/__test/probe", headers=ADMIN).json()
            assert probe["owned"] == probe["active"] == 0
            client.post("/__test/release-acquire", headers=ADMIN)
        assert not calls
        assert probe["ledger"]["unknown_charge_count"] == 0


def test_affinity_does_not_migrate_when_owner_cools(local_pki, tmp_path):  # noqa: F811
    frames = []

    def provider(ws):
        for frame in ws:
            frames.append(frame)
            ws.send(completed())

    def policy(raw):
        configure(raw)
        account = raw["credentials"][0]
        account.update(owner_group="fixture", billing_group="fixture")
        other = dict(account, id="other-key")
        raw["credentials"].append(other)
        raw["routes"][0]["credentials"].append("other-key")
        raw["routes"][0]["catalog"]["entitlements"]["other-key"] = "allowed"

    with http_process(
        local_pki, tmp_path, lambda _: None, configure=policy, websocket_handler=provider
    ) as (client, _, _, raw):
        with socket(client) as ws:
            ws.send(CREATE)
            ws.recv(timeout=5)
            account, route = raw["credentials"][0]["id"], raw["routes"][0]["id"]
            assert client.post(f"/__test/cool/{account}/{route}").status_code == 200
            ws.send(CREATE)
            assert json.loads(ws.recv(timeout=5))["error"]["code"] == "credential_unavailable"
        probe = client.get("/__test/idle").json()
        assert len(frames) == 1
        assert probe["identity"] == 1
