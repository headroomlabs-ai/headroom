"""Round-one refusal/error and multi-candidate accounting over real TLS sockets."""

import pytest

from tests.unified_gateway.process.http_harness import (
    frame,
    http_process,
    json_response,
    stream_headers,
)


@pytest.mark.parametrize("streamed", [False, True])
def test_responses_refusal_is_failed_and_keeps_usage(local_pki, tmp_path, streamed):
    completed = b'{"id":"resp_refused","status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"Cannot comply"}]}],"usage":{"input_tokens":3,"output_tokens":2}}'

    def handler(upstream):
        if not streamed:
            json_response(upstream, completed)
            return
        stream_headers(upstream)
        frame(
            upstream,
            b'event: response.refusal.delta\ndata: {"type":"response.refusal.delta","delta":"Cannot comply"}\n\n',
        )
        frame(
            upstream,
            b'event: response.refusal.done\ndata: {"type":"response.refusal.done","refusal":"Cannot comply"}\n\n',
        )
        frame(
            upstream,
            b'event: response.completed\ndata: {"type":"response.completed","response":'
            + completed
            + b"}\n\n",
        )

    with http_process(local_pki, tmp_path, handler) as (client, calls, _, _):
        response = client.post(
            "/v1/responses", json={"model": "fixture-model", "input": "hello", "stream": streamed}
        )
        assert response.status_code == (200 if streamed else 502)
        assert b"gateway_upstream_error" in response.content
        assert b"response.completed" not in response.content
        assert b"Cannot comply" not in response.content
        assert b"response.refusal" not in response.content
        probe = client.get("/__test/idle").json()
        assert len(calls) == 1
        assert probe["operations"][0]["terminal"] == "failed"
        assert probe["ledger"]["known_micro_usd"] == 7
        assert probe["ledger"]["unknown_charge_count"] == 0
        assert probe["active"] == probe["queued"] == 0


@pytest.mark.parametrize(
    "usage,known,unresolved,unknown",
    [(b'{"prompt_tokens":3,"completion_tokens":2}', 7, 0, 0), (b'{"prompt_tokens":3}', 3, 27, 1)],
)
def test_error_event_usage_is_accounted_before_sanitization(
    local_pki, tmp_path, usage, known, unresolved, unknown
):
    def handler(upstream):
        stream_headers(upstream)
        frame(
            upstream,
            b'event: error\ndata: {"error":{"message":"SECRET"},"usage":' + usage + b"}\n\n",
        )

    with http_process(local_pki, tmp_path, handler) as (client, calls, _, _):
        response = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": [], "stream": True}
        )
        assert b"gateway_upstream_error" in response.content
        assert b"SECRET" not in response.content
        probe = client.get("/__test/idle").json()
        assert len(calls) == 1
        assert probe["operations"][0]["terminal"] == "failed"
        assert probe["ledger"]["known_micro_usd"] == known
        assert probe["ledger"]["unresolved_micro_usd"] == unresolved
        assert probe["ledger"]["unknown_charge_count"] == unknown
        assert probe["active"] == probe["queued"] == 0


def test_missing_requested_candidate_is_not_a_success(local_pki, tmp_path):
    def handler(upstream):
        stream_headers(upstream)
        frame(
            upstream,
            b'data: {"candidates":[{"index":0,"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2}}\n\n',
        )

    def configure(raw):
        raw["routes"][0].update(
            ingress_protocols=["vertex-generate"],
            native_protocols=["vertex-generate"],
            capabilities={"vertex-generate": {"http-stream": {"features": ["text"]}}},
        )

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        path = "/v1/projects/p/locations/r/publishers/google/models/fixture-model:streamGenerateContent"
        response = client.post(
            path, json={"contents": [], "generationConfig": {"candidateCount": 2}}
        )
        assert b"gateway_upstream_error" in response.content
        probe = client.get("/__test/idle").json()
        assert len(calls) == 1
        assert probe["operations"][0]["terminal"] == "failed"
        assert probe["ledger"]["known_micro_usd"] == 7
        # The single-output synthetic bound does not qualify n-way output.
        # Unknown liability is counted, never fabricated as a monetary bound.
        assert probe["ledger"]["unresolved_micro_usd"] == 0
        assert probe["operations"][0]["attempts"][0]["cost"]["reserved_upper_micro_usd"] is None
        assert probe["ledger"]["unknown_charge_count"] == 1


@pytest.mark.parametrize("count", [0, 9, "2", True])
def test_unsupported_candidate_counts_acquire_no_identity(local_pki, tmp_path, count):
    def handler(upstream):
        pytest.fail("unsupported candidate count reached provider")

    def configure(raw):
        raw["routes"][0].update(
            ingress_protocols=["vertex-generate"],
            native_protocols=["vertex-generate"],
            capabilities={"vertex-generate": {"http-stream": {"features": ["text"]}}},
        )

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        path = "/v1/projects/p/locations/r/publishers/google/models/fixture-model:streamGenerateContent"
        response = client.post(
            path, json={"contents": [], "generationConfig": {"candidateCount": count}}
        )
        assert response.status_code == 400
        assert not calls
        assert client.get("/__test/idle").json()["identity"] == 0
