"""HTTP subset of T052–T054: shared ledger and every accepted attempt."""

import concurrent.futures
import copy
import threading

import pytest

from tests.unified_gateway.process.http_harness import (
    frame,
    http_process,
    json_response,
    stream_headers,
)


def test_two_routes_race_one_remaining_budget_reservation(local_pki, tmp_path):  # noqa: F811
    accepted, release = threading.Event(), threading.Event()

    def handler(upstream):
        accepted.set()
        assert release.wait(10)
        json_response(upstream, b'{"usage":{"prompt_tokens":3,"completion_tokens":2}}')

    def configure(raw):
        raw["admission"].update(budget_usd="0.000030", unknown_cost_policy="block")
        route = copy.deepcopy(raw["routes"][0])
        route.update(id="second", public_model="second-model", upstream_model="second-model")
        raw["routes"].append(route)
        other = copy.deepcopy(raw["client_auth"]["principals"][0])
        other.update(
            id="tenant-b", secret_ref="env:HEADROOM_GATEWAY_CLIENT_TOKEN_B", routes=["second"]
        )
        raw["client_auth"]["principals"].append(other)

    with http_process(local_pki, tmp_path, handler, configure=configure, reserve_barrier=2) as (
        client,
        calls,
        _,
        _,
    ):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = [
                pool.submit(
                    client.post,
                    "/v1/chat/completions",
                    json={"model": model, "messages": []},
                    headers={"authorization": f"Bearer {token}"},
                )
                for model, token in [
                    ("fixture-model", "client-secret"),
                    ("second-model", "client-b"),
                ]
            ]
            try:
                denied = next(concurrent.futures.as_completed(pending, timeout=5)).result()
                assert accepted.wait(5)
                assert denied.status_code == 403
                assert "budget" in denied.json()["error"]["code"]
                assert len(calls) == 1
                probe = client.get("/__test/probe").json()
                assert len({item["principal"] for item in probe["reserve_arrivals"]}) == 2
                assert len({item["route"] for item in probe["reserve_arrivals"]}) == 2
                assert probe["identity"] == 1
                assert probe["ledger"]["reserved_micro_usd"] == 30
            finally:
                release.set()
            assert sorted(task.result(timeout=5).status_code for task in pending) == [200, 403]
        probe = client.get("/__test/idle").json()
        assert probe["ledger"]["known_micro_usd"] == 7
        assert probe["active"] == probe["queued"] == 0


def test_retry_accounting_keeps_known_and_unknown_attempts(local_pki, tmp_path):  # noqa: F811
    bodies = iter(
        [
            b'{"error":{"type":"overloaded_error"},"usage":{"input_tokens":3,"output_tokens":2}}',
            b'{"error":{"type":"overloaded_error"},"usage":{"input_tokens":4}}',
            b'{"type":"message","usage":{"input_tokens":1,"output_tokens":1}}',
        ]
    )
    statuses = iter([529, 529, 200])

    def handler(upstream):
        body = next(bodies)
        upstream.send_response(next(statuses))
        upstream.send_header("content-type", "application/json")
        upstream.send_header("content-length", str(len(body)))
        upstream.send_header("retry-after", "0")
        upstream.end_headers()
        upstream.wfile.write(body)

    def configure(raw):
        raw["routes"][0]["retry"]["max_attempts"] = 3

    with http_process(local_pki, tmp_path, handler, public_provider=True, configure=configure) as (
        client,
        calls,
        _,
        _,
    ):
        response = client.post(
            "/v1/messages", json={"model": "fixture-model", "messages": [], "max_tokens": 10}
        )
        assert response.status_code == 200
        probe = client.get("/__test/idle").json()
        assert len(calls) == probe["totals"]["attempts"] == 3
        assert probe["totals"]["logical_requests"] == 1
        assert probe["ledger"]["known_micro_usd"] == 14
        assert probe["ledger"]["unresolved_micro_usd"] == 26
        assert probe["ledger"]["unknown_charge_count"] == 1
        attempts = probe["operations"][0]["attempts"]
        assert [attempt["exposure"] for attempt in attempts] == [
            "proven_rejected",
            "proven_rejected",
            "accepted",
        ]
        assert [attempt["cost"]["known_micro_usd"] for attempt in attempts] == [7, 4, 3]
        assert [attempt["cost"]["complete"] for attempt in attempts] == [True, False, True]
        assert probe["active"] == probe["queued"] == 0


def test_strict_budget_resource_read_does_not_reserve_generation_twice(local_pki, tmp_path):  # noqa: F811
    def handler(upstream):
        json_response(upstream, b'{"id":"resp_1","usage":{"input_tokens":3,"output_tokens":2}}')

    def configure(raw):
        raw["admission"].update(budget_usd="0.000030", unknown_cost_policy="block")

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        response = client.post("/v1/responses", json={"model": "fixture-model", "input": "hello"})
        assert response.status_code == 200
        assert client.get("/v1/responses/resp_1").status_code == 200
        probe = client.get("/__test/idle").json()
        assert probe["ledger"]["known_micro_usd"] == 7
        assert probe["ledger"]["unknown_charge_count"] == 0
        assert len(calls) == probe["totals"]["attempts"] == 2


def test_noisy_tenant_cannot_consume_reserved_other_tenant_slots(local_pki, tmp_path):  # noqa: F811
    accepted, release = threading.Event(), threading.Event()

    def handler(upstream):
        if not accepted.is_set():
            accepted.set()
            assert release.wait(10)
        json_response(upstream, b'{"usage":{"prompt_tokens":1,"completion_tokens":1}}')

    def configure(raw):
        raw["admission"]["max_concurrency"] = 2
        other = copy.deepcopy(raw["client_auth"]["principals"][0])
        other.update(
            id="other",
            secret_ref="env:HEADROOM_GATEWAY_CLIENT_TOKEN_B",
            admission={"reserved_concurrency": 1},
        )
        raw["client_auth"]["principals"].append(other)

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pending = pool.submit(
                client.post, "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
            )
            try:
                assert accepted.wait(5)
                denied = client.post(
                    "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
                )
                assert denied.status_code == 429
                other = client.post(
                    "/v1/chat/completions",
                    headers={"authorization": "Bearer client-b"},
                    json={"model": "fixture-model", "messages": []},
                )
                assert other.status_code == 200
                assert len(calls) == 2
            finally:
                release.set()
            assert pending.result(timeout=5).status_code == 200
        assert client.get("/__test/idle").json()["active"] == 0


@pytest.mark.parametrize(
    "body,status",
    [(b'{"error":{"message":"secret"}}', 400), (b"x" * 2000, 200), (b"{invalid", 200)],
)
def test_terminal_path_leak_matrix(local_pki, tmp_path, body, status):  # noqa: F811
    def handler(upstream):
        json_response(upstream, body, status)

    with http_process(
        local_pki,
        tmp_path,
        handler,
        configure=lambda raw: raw["limits"].update(max_observed_json_bytes=1024),
    ) as (client, calls, _, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
        )
        assert response.status_code == 502
        assert b"secret" not in response.content
        probe = client.get("/__test/idle").json()
        assert probe["active"] == probe["queued"] == 0
        assert probe["totals"]["attempts"] == len(calls) == 1
        assert probe["totals"]["logical_requests"] == 1
        assert probe["ledger"]["unresolved_micro_usd"] == 30


@pytest.mark.parametrize("ending", ["eof", "error", "timeout", "disconnect"])
def test_anthropic_intermediate_usage_retains_strict_budget(local_pki, tmp_path, ending):
    closed = threading.Event()

    def handler(upstream):
        stream_headers(upstream)
        frame(
            upstream,
            b'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1}}}\n\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}\n\n',
        )
        if ending == "error":
            frame(
                upstream, b'event: error\ndata: {"type":"error","error":{"message":"SECRET"}}\n\n'
            )
        elif ending in {"timeout", "disconnect"}:
            upstream.connection.settimeout(3)
            assert upstream.connection.recv(1) == b""
            closed.set()

    def configure(raw):
        raw["admission"].update(budget_usd="0.000030", unknown_cost_policy="block")
        raw["limits"].update(
            request_deadline_seconds=60,
            partial_frame_seconds=60,
            stream_content_idle_seconds=0.2 if ending == "timeout" else 60,
        )

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        if ending == "disconnect":
            with client.stream(
                "POST",
                "/v1/messages",
                json={"model": "fixture-model", "messages": [], "stream": True},
            ) as response:
                assert b"message_start" in next(response.iter_raw())
        else:
            response = client.post(
                "/v1/messages", json={"model": "fixture-model", "messages": [], "stream": True}
            )
            assert b"gateway_upstream_error" in response.content
        if ending in {"disconnect", "timeout"}:
            assert closed.wait(2)
        probe = client.get("/__test/idle").json()
        assert probe["ledger"]["known_micro_usd"] == 12
        assert probe["ledger"]["unresolved_micro_usd"] == 18
        assert probe["ledger"]["unknown_charge_count"] == 1
        assert probe["active"] == 0
        assert (
            client.post("/v1/messages", json={"model": "fixture-model", "messages": []}).status_code
            == 403
        )
        assert len(calls) == 1
