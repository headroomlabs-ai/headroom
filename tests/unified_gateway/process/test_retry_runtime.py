"""T046–T050 real transport retries and acceptance boundaries."""

import concurrent.futures
import copy
import socket
import threading
import time

import pytest

from tests.unified_gateway.process.http_harness import (
    frame,
    http_process,
    json_response,
    stream_headers,
)
from tests.unified_gateway.process.test_streaming_runtime import CHAT_FIRST


def test_proven_connect_failure_retries_once_under_same_deadline(local_pki, tmp_path):  # noqa: F811
    def handler(upstream):
        json_response(upstream, b'{"usage":{"prompt_tokens":3,"completion_tokens":2}}')

    with http_process(local_pki, tmp_path, handler, fail_connect=True) as (client, calls, _, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
        )
        assert response.status_code == 200
        probe = client.get("/__test/idle").json()
        assert len(calls) == 1
        assert probe["totals"]["attempts"] == 2
        assert probe["totals"]["logical_requests"] == 1
        assert probe["ledger"]["known_micro_usd"] == 7
        assert probe["ledger"]["unknown_charge_count"] == 0


def test_body_sent_disconnect_is_ambiguous_and_not_retried(local_pki, tmp_path):  # noqa: F811
    accepted = threading.Event()

    def handler(upstream):
        accepted.set()
        upstream.connection.shutdown(socket.SHUT_RDWR)
        upstream.connection.close()

    with http_process(local_pki, tmp_path, handler) as (client, calls, _, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
        )
        assert accepted.wait(2)
        assert response.status_code == 502
        probe = client.get("/__test/idle").json()
        assert len(calls) == probe["totals"]["attempts"] == 1
        assert probe["ledger"]["unresolved_micro_usd"] == 30
        assert probe["ledger"]["unknown_charge_count"] == 1
        assert probe["active"] == 0


def test_output_then_quota_error_never_switches_account(local_pki, tmp_path):  # noqa: F811
    def handler(upstream):
        stream_headers(upstream)
        frame(upstream, CHAT_FIRST)
        frame(
            upstream,
            b'event: error\r\ndata: {"error":{"message":"PRIVATE PROVIDER SECRET","type":"rate_limit_error"}}\r\n\r\n',
        )

    with http_process(local_pki, tmp_path, handler) as (client, calls, output, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": [], "stream": True}
        )
        assert response.content.startswith(CHAT_FIRST)
        assert b"gateway_upstream_error" in response.content
        assert b"PRIVATE PROVIDER SECRET" not in response.content
        assert "PRIVATE PROVIDER SECRET" not in "".join(output)
        probe = client.get("/__test/idle").json()
        assert len(calls) == probe["totals"]["attempts"] == 1
        assert probe["ledger"]["unresolved_micro_usd"] == 30


def test_cooldown_uses_quota_scope_and_equivalent_billing_only(local_pki, tmp_path):  # noqa: F811
    seen = []

    def handler(upstream):
        seen.append(upstream.path)
        if len(seen) == 1:
            upstream.send_response(429)
            upstream.send_header("retry-after", "0")
            body = b'{"error":{"type":"rate_limit_error","message":"SECRET"}}'
            upstream.send_header("content-length", str(len(body)))
            upstream.send_header("content-type", "application/json")
            upstream.end_headers()
            upstream.wfile.write(body)
        else:
            json_response(upstream, b'{"content":[],"usage":{"input_tokens":1,"output_tokens":1}}')

    def configure(raw):
        credential = raw["credentials"][0]
        credential.update(owner_group="owner", billing_group="billing")
        second, wrong = copy.deepcopy(credential), copy.deepcopy(credential)
        second.update(id="second-key")
        wrong.update(id="wrong-billing", billing_group="other")
        raw["credentials"].extend([second, wrong])
        route = raw["routes"][0]
        route["credentials"] = ["internal-key", "second-key"]
        route["catalog"]["entitlements"] = dict.fromkeys(route["credentials"], "allowed")
        route["selection"] = {"quota_group": "quota-a"}
        other = copy.deepcopy(route)
        other.update(
            id="other-route",
            public_model="other-model",
            upstream_model="other-model",
            selection={"quota_group": "quota-b"},
        )
        raw["routes"].append(other)
        raw["client_auth"]["principals"][0]["routes"].append("other-route")

    with http_process(local_pki, tmp_path, handler, configure=configure, public_provider=True) as (
        client,
        calls,
        _,
        _,
    ):
        response = client.post("/v1/messages", json={"model": "fixture-model", "messages": []})
        assert response.status_code == 200
        # Cool both equivalent accounts only for quota-a. Another quota still works.
        for account in ["internal-key", "second-key"]:
            assert client.post(f"/__test/cool/{account}/quota-a").status_code == 200
        denied = client.post("/v1/messages", json={"model": "fixture-model", "messages": []})
        assert denied.status_code == 503
        other = client.post("/v1/messages", json={"model": "other-model", "messages": []})
        assert other.status_code == 200
        probe = client.get("/__test/idle").json()
        assert len(calls) == 3
        assert probe["totals"]["attempts"] == 3
        assert [o["account"] for o in probe["observations"]] == ["second-key", "internal-key"]


@pytest.mark.parametrize("retry_after,maximum", [("0", 2), ("bad", 1), ("9999", 1)])
def test_overload_storm_respects_attempt_queue_and_deadline_bounds(
    local_pki,
    tmp_path,
    retry_after,
    maximum,  # noqa: F811
):  # noqa: F811
    release = threading.Event()
    upstream_arrivals = threading.Event()
    arrival_lock = threading.Lock()
    arrivals = 0

    def handler(upstream):
        nonlocal arrivals
        with arrival_lock:
            arrivals += 1
            if arrivals == 2:
                upstream_arrivals.set()
        assert release.wait(10)
        body = b'{"error":{"type":"overloaded_error","message":"SECRET"}}'
        upstream.send_response(529)
        upstream.send_header("retry-after", retry_after)
        upstream.send_header("content-length", str(len(body)))
        upstream.send_header("content-type", "application/json")
        upstream.end_headers()
        upstream.wfile.write(body)

    def configure(raw):
        raw["admission"].update(max_concurrency=2, queue_limit=2, queue_timeout_seconds=4)
        raw["client_auth"]["principals"][0]["admission"] = {
            "max_concurrency": 2,
            "queue_limit": 2,
            "queue_timeout_seconds": 4,
        }
        raw["limits"].update(request_deadline_seconds=5, shutdown_cleanup_seconds=0.5)

    with http_process(local_pki, tmp_path, handler, public_provider=True, configure=configure) as (
        client,
        calls,
        _,
        _,
    ):
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            pending = [
                pool.submit(
                    client.post, "/v1/messages", json={"model": "fixture-model", "messages": []}
                )
                for _ in range(8)
            ]
            try:
                completed = concurrent.futures.as_completed(pending, timeout=4)
                # Four local overflows prove all eight callers reached admission
                # while two provider sockets and two queue positions remain held.
                for _ in range(4):
                    assert next(completed).result().status_code == 429
                # Admission can precede TLS arrival; wait for the actual handlers.
                assert upstream_arrivals.wait(2)
                probe = client.get("/__test/probe").json()
                assert probe["active"] == probe["queued"] == 2
                assert len(calls) == 2
            finally:
                release.set()
            responses = [task.result(timeout=6) for task in pending]
        assert time.monotonic() - started < 6
        assert all(response.status_code in {429, 502, 503} for response in responses)
        probe = client.get("/__test/idle").json()
        assert 1 <= len(calls) <= 8 * maximum
        assert probe["active"] == probe["queued"] == 0
        assert probe["ledger"]["unknown_charge_count"] == 0
        # Four overflow requests are rejected before runtime registration.
        assert len(probe["operations"]) == 4
        for operation in probe["operations"]:
            assert len(operation["attempts"]) <= maximum
            assert operation["finished"] <= operation["deadline"] + 0.5
