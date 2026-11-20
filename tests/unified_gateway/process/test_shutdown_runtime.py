"""Owned child identity and bounded drain before transport cancellation."""

import concurrent.futures
import json
import subprocess
import sys
import threading
from contextlib import contextmanager

import httpx
import pytest
from websockets.exceptions import ConnectionClosed

from tests.unified_gateway.process.harness import GatewayProcess, start_gateway_process
from tests.unified_gateway.process.http_harness import http_process, json_response
from tests.unified_gateway.process.test_websocket_runtime import CREATE, configure, socket
from tests.unified_gateway.test_gateway_tls import local_pki  # noqa: F401


@contextmanager
def unrelated_listener():
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.unified_gateway.process.unrelated_listener", "0"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        port = int(process.stdout.readline())
        yield process, port
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_wrong_listener_200_cannot_satisfy_startup(tmp_path, monkeypatch):
    with unrelated_listener() as (other, port):
        monkeypatch.setattr(
            "tests.unified_gateway.process.harness._unused_loopback_port", lambda: port
        )
        gateway = None
        try:
            with pytest.raises(RuntimeError):
                gateway = start_gateway_process(tmp_path)
        finally:
            if gateway is not None:
                gateway.close()
        assert other.poll() is None
        assert httpx.get(f"http://127.0.0.1:{port}/readyz").status_code == 200


def test_only_owned_children_and_sockets_are_closed(tmp_path):
    with unrelated_listener() as (other, port):
        GatewayProcess(process=None, base_url="").close()
        GatewayProcess(process=other, base_url=f"http://127.0.0.1:{port}").close()
        assert other.poll() is None
        gateway = start_gateway_process(tmp_path)
        gateway.close()
        gateway.close()
        assert gateway.process.poll() is not None
        assert other.poll() is None
        assert httpx.get(f"http://127.0.0.1:{port}/readyz").status_code == 200
        with pytest.raises(httpx.HTTPError):
            httpx.get(gateway.base_url + "/readyz", timeout=0.5)


def test_drain_then_cancel_finalizes_http_ws_and_queue(local_pki, tmp_path):  # noqa: F811
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

    with http_process(local_pki, tmp_path, handler, configure=policy) as (client, calls, _, _):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(
                client.post, "/v1/responses", json={"model": "fixture-model", "input": "a"}
            )
            try:
                assert accepted.wait(5)
                with socket(client) as ws:
                    ws.send(CREATE)
                    assert client.post("/__test/queue-entered").json()["queued"] == 1
                    probe = client.post("/__test/shutdown").json()
                    assert probe["active"] == probe["queued"] == probe["owned"] == 0
                    assert probe["ledger"]["unknown_charge_count"] == 1
                    assert probe["ledger"]["unresolved_micro_usd"] == 30
                    assert client.get("/readyz").status_code == 503
                    try:
                        assert json.loads(ws.recv(timeout=3))["type"] == "error"
                    except ConnectionClosed:
                        pass
            finally:
                release.set()
            try:
                assert pending.result(timeout=5).status_code != 200
            except httpx.HTTPError:
                pass
        assert len(calls) == 1


def test_uvicorn_shutdown_uses_gateway_drain_before_waiting_on_requests(local_pki, tmp_path):  # noqa: F811
    accepted, release = threading.Event(), threading.Event()

    def handler(upstream):
        accepted.set()
        assert release.wait(10)

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, _, _, _):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(
                client.post, "/v1/responses", json={"model": "fixture-model", "input": "a"}
            )
            try:
                assert accepted.wait(5)
                assert client.post("/__test/stop").status_code == 200
                client.gateway_test_process.wait(timeout=2.5)
            finally:
                release.set()
            try:
                assert pending.result(timeout=5).status_code != 200
            except httpx.HTTPError:
                pass


def test_admitted_work_finishes_during_drain_without_cancellation(local_pki, tmp_path):  # noqa: F811
    accepted, release = threading.Event(), threading.Event()

    def handler(upstream):
        accepted.set()
        assert release.wait(10)
        json_response(upstream, b'{"usage":{"input_tokens":3,"output_tokens":2}}')

    def policy(raw):
        configure(raw)
        raw["limits"]["shutdown_drain_seconds"] = 2

    with http_process(local_pki, tmp_path, handler, configure=policy) as (client, _, _, _):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            request = pool.submit(
                client.post, "/v1/responses", json={"model": "fixture-model", "input": "a"}
            )
            try:
                assert accepted.wait(5)
                shutdown = pool.submit(client.post, "/__test/shutdown")
                assert client.post("/__test/shutdown-started").json()["started"]
                assert client.get("/readyz").status_code == 503
            finally:
                release.set()
            assert request.result(timeout=5).status_code == 200
            probe = shutdown.result(timeout=5).json()
        assert probe["ledger"]["known_micro_usd"] == 7
        assert probe["ledger"]["unknown_charge_count"] == 0
        assert probe["operations"][0]["terminal"] == "success"


def test_shutdown_cancels_active_websocket_and_retains_liability(local_pki, tmp_path):  # noqa: F811
    accepted, closed = threading.Event(), threading.Event()

    def provider(ws):
        try:
            ws.recv()
            accepted.set()
            for _ in ws:
                pass
        finally:
            closed.set()

    with http_process(
        local_pki, tmp_path, lambda _: None, configure=configure, websocket_handler=provider
    ) as (client, _, _, _):
        with socket(client) as ws:
            ws.send(CREATE)
            assert accepted.wait(5)
            probe = client.post("/__test/shutdown").json()
            with pytest.raises(ConnectionClosed):
                ws.recv(timeout=3)
        assert closed.wait(3)
        assert probe["active"] == probe["queued"] == probe["owned"] == 0
        assert probe["ledger"]["unknown_charge_count"] == 1
        assert probe["ledger"]["unresolved_micro_usd"] == 30
