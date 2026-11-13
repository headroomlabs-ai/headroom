"""Reload/revocation oracles through actual admitted HTTP and WS work."""

import concurrent.futures
import json
import threading

import httpx
import pytest
from websockets.exceptions import ConnectionClosed

from tests.unified_gateway.process.http_harness import http_process, json_response
from tests.unified_gateway.process.test_websocket_runtime import ADMIN, CREATE, configure, socket


def test_atomic_reload_preserves_inflight_snapshot_and_ledger(local_pki, tmp_path):  # noqa: F811
    """Catch accepted invalid reloads, reset liabilities, or lost pre-reload bindings."""
    accepted, release = threading.Event(), threading.Event()

    def handler(upstream):
        payload = json.loads(upstream.body)
        if payload.get("input") == "ambiguous":
            json_response(upstream, b'{"id":"resp_ambiguous"}')
            return
        if payload.get("input") == "a":
            accepted.set()
            assert release.wait(10)
        json_response(
            upstream,
            json.dumps(
                {
                    "id": "resp_" + payload["input"],
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                }
            ).encode(),
        )

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, raw):
        for value in ("seed", "ambiguous"):
            assert (
                client.post(
                    "/v1/responses", json={"model": "fixture-model", "input": value}
                ).status_code
                == 200
            )
        baseline = client.get("/__test/idle").json()
        assert baseline["ledger"]["known_micro_usd"] == 7
        assert baseline["ledger"]["unresolved_micro_usd"] == 30
        assert baseline["ledger"]["unknown_charge_count"] == 1
        with concurrent.futures.ThreadPoolExecutor() as pool:
            first = pool.submit(
                client.post, "/v1/responses", json={"model": "fixture-model", "input": "a"}
            )
            try:
                assert accepted.wait(5)
                before = client.get("/__test/probe").json()
                status = client.get("/admin/gateway/status", headers=ADMIN).json()
                path = tmp_path / "http-runtime.json"
                path.write_text("{")
                invalid = client.post("/admin/gateway/reload", headers=ADMIN)
                assert invalid.status_code == 409
                assert invalid.json()["applied"] is False
                unchanged = client.get("/admin/gateway/status", headers=ADMIN).json()
                assert unchanged["generation"] == status["generation"] == 1
                assert unchanged["config_digest"] == status["config_digest"]
                assert client.get("/__test/probe").json()["ledger"] == before["ledger"]
                assert not first.done()
                raw["routes"][0]["pricing"].update(revision="b", output_usd_per_million="5")
                path.write_text(json.dumps(raw))
                assert client.post("/admin/gateway/reload", headers=ADMIN).json()["applied"]
                assert (
                    client.post(
                        "/v1/responses",
                        json={
                            "model": "fixture-model",
                            "input": "b",
                            "previous_response_id": "resp_seed",
                        },
                    ).status_code
                    == 200
                )
                during = client.get("/__test/probe").json()
                assert during["ledger"]["reserved_micro_usd"] == 30
                assert during["ledger"]["known_micro_usd"] == 20
                assert during["ledger"]["unresolved_micro_usd"] == 30
                assert during["ledger"]["unknown_charge_count"] == 1
                assert not first.done()
            finally:
                release.set()
            assert first.result(timeout=5).status_code == 200
        probe = client.get("/__test/idle").json()
        assert probe["ledger"]["known_micro_usd"] == 27
        assert probe["ledger"]["unresolved_micro_usd"] == 30
        assert probe["ledger"]["unknown_charge_count"] == 1
        assert probe["ledger"]["reserved_micro_usd"] == 0
        assert [op["generation"] for op in probe["operations"]] == [1, 1, 1, 2]
        assert [op["tariff"] for op in probe["operations"]] == ["fixture"] * 3 + ["b"]
        assert len(calls) == 4
        assert json.loads(calls[-1][1])["previous_response_id"] == "resp_seed"


@pytest.mark.parametrize("phase", ["queued", "acquire"])
@pytest.mark.parametrize("selector", ["principal_id", "route_id", "account_id", "grant"])
def test_revoke_denies_waiters_and_cancels_http_ws_and_acquire(
    local_pki, tmp_path, phase, selector
):  # noqa: F811
    accepted, release = threading.Event(), threading.Event()

    def handler(upstream):
        accepted.set()
        assert release.wait(10)
        try:
            json_response(upstream, b'{"usage":{"input_tokens":3,"output_tokens":2}}')
        except OSError:
            pass

    def policy(raw):
        configure(raw)
        raw["admission"].update(max_concurrency=1, queue_limit=2, queue_timeout_seconds=4)
        raw["client_auth"]["principals"][0]["admission"] = {
            "max_concurrency": 1,
            "queue_limit": 2,
            "queue_timeout_seconds": 4,
        }
        spare = dict(raw["routes"][0], id="spare", public_model="spare")
        raw["routes"].append(spare)
        raw["client_auth"]["principals"][0]["routes"].append("spare")

    with http_process(
        local_pki, tmp_path, handler, configure=policy, acquire_barrier=phase == "acquire"
    ) as (client, calls, _, raw):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(
                client.post, "/v1/responses", json={"model": "fixture-model", "input": "a"}
            )
            try:
                if phase == "queued":
                    assert accepted.wait(5)
                else:
                    assert client.post("/__test/acquire-entered").json()["entered"]
                with socket(client) as ws:
                    ws.send(CREATE)
                    assert client.post("/__test/queue-entered").json()["queued"] == 1
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
                    try:
                        assert json.loads(ws.recv(timeout=3))["type"] == "error"
                    except ConnectionClosed:
                        pass
                with httpx.Client(base_url=client.base_url, timeout=12) as control:
                    release_response = control.post("/__test/release-acquire", headers=ADMIN)
                    assert release_response.status_code == 200, release_response.text
                    probe = control.get("/__test/idle", headers=ADMIN).json()
                assert probe["active"] == probe["queued"] == probe["owned"] == 0
                assert len(calls) == (1 if phase == "queued" else 0)
                assert probe["ledger"]["unknown_charge_count"] == (1 if phase == "queued" else 0)
            finally:
                release.set()
            try:
                assert pending.result(timeout=5).status_code != 200
            except httpx.HTTPError:
                pass
