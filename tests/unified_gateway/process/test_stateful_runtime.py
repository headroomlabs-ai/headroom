"""HTTP ownership/affinity subset; WebSocket turn cases remain a later batch."""

import copy

from tests.unified_gateway.process.http_harness import http_process, json_response


def test_all_route_and_upgrade_variants_enforce_runtime(local_pki, tmp_path):
    import re

    from websockets.exceptions import ConnectionClosed, InvalidStatus
    from websockets.sync.client import connect

    from headroom.providers.codex.live import CODEX_LIVE_ROUTE_PATHS
    from headroom.providers.openai_responses import (
        OPENAI_RESPONSES_ROOT_PATHS,
        OPENAI_RESPONSES_SUBPATH_ROUTES,
        OPENAI_RESPONSES_WEBSOCKET_PATHS,
    )
    from headroom.providers.route_specs import (
        ANTHROPIC_BATCH_ROUTES,
        GEMINI_BATCH_ROUTES,
        OPENAI_BATCH_ROUTES,
        PROVIDER_HANDLER_ROUTES,
        PROVIDER_PASSTHROUGH_ROUTES,
    )
    from tests.unified_gateway.process.test_websocket_runtime import configure

    def concretize(path):
        return re.sub(r"\{[^}]+\}", "missing", path)

    with http_process(local_pki, tmp_path, lambda _: None, configure=configure) as (
        client,
        calls,
        _,
        _,
    ):
        bad = {"authorization": "Bearer invalid"}
        inventory = [
            (spec.method, concretize(spec.path))
            for spec in (*PROVIDER_HANDLER_ROUTES, *PROVIDER_PASSTHROUGH_ROUTES)
        ]
        inventory += [("POST", path) for path in OPENAI_RESPONSES_ROOT_PATHS]
        inventory += [
            (method, concretize(spec.path))
            for spec in OPENAI_RESPONSES_SUBPATH_ROUTES
            for method in spec.methods
        ]
        inventory += [("POST", "/arbitrary/unmanaged/path")]
        for method, path in inventory:
            response = client.request(method, path, headers=bad, json={"model": "fixture-model"})
            assert response.status_code == 401, (method, path, response.text)
        for spec in (*ANTHROPIC_BATCH_ROUTES, *OPENAI_BATCH_ROUTES, *GEMINI_BATCH_ROUTES):
            response = client.request(
                spec.method, concretize(spec.path), json={"model": "fixture-model"}
            )
            assert response.status_code >= 400, (spec.method, spec.path, response.text)
        for spec in OPENAI_RESPONSES_SUBPATH_ROUTES:
            for method in spec.methods:
                response = client.request(method, concretize(spec.path))
                assert response.status_code >= 400
        base = str(client.base_url).replace("http://", "ws://").rstrip("/")
        for path in (
            *OPENAI_RESPONSES_WEBSOCKET_PATHS,
            *CODEX_LIVE_ROUTE_PATHS,
            "/responses",
            "/v1/batches",
            "/arbitrary/unmanaged/path",
        ):
            try:
                with connect(
                    base + path, additional_headers=bad, proxy=None, close_timeout=1
                ) as ws:
                    ws.recv(timeout=3)
            except (InvalidStatus, ConnectionClosed):
                pass
            else:
                raise AssertionError("unauthorized upgrade was accepted")
        assert client.get("/__test/probe").json()["identity"] == 0
        assert not calls


def test_other_principal_cannot_read_continue_cancel_delete(local_pki, tmp_path):  # noqa: F811
    def handler(upstream):
        json_response(upstream, b'{"id":"resp_1","usage":{"input_tokens":3,"output_tokens":2}}')

    def configure(raw):
        other = copy.deepcopy(raw["client_auth"]["principals"][0])
        other.update(id="other", secret_ref="env:HEADROOM_GATEWAY_CLIENT_TOKEN_B")
        raw["client_auth"]["principals"].append(other)

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        assert (
            client.post(
                "/v1/responses", json={"model": "fixture-model", "input": "hello"}
            ).status_code
            == 200
        )
        headers = {"authorization": "Bearer client-b"}
        assert client.get("/v1/responses/resp_1", headers=headers).status_code == 404
        assert client.post("/v1/responses/resp_1/cancel", headers=headers).status_code == 404
        assert client.delete("/v1/responses/resp_1", headers=headers).status_code == 404
        assert (
            client.post(
                "/v1/responses",
                headers=headers,
                json={"model": "fixture-model", "previous_response_id": "resp_1"},
            ).status_code
            == 404
        )
        assert len(calls) == 1


def test_affinity_does_not_migrate_when_owner_unavailable(local_pki, tmp_path):  # noqa: F811
    def handler(upstream):
        json_response(upstream, b'{"id":"resp_1"}')

    def configure(raw):
        account = raw["credentials"][0]
        account.update(owner_group="owner", billing_group="billing")
        other = copy.deepcopy(account)
        other["id"] = "other-key"
        raw["credentials"].append(other)
        raw["routes"][0]["credentials"].append("other-key")
        raw["routes"][0]["catalog"]["entitlements"]["other-key"] = "allowed"

    with http_process(local_pki, tmp_path, handler, configure=configure) as (client, calls, _, _):
        assert (
            client.post(
                "/v1/responses", json={"model": "fixture-model", "input": "hello"}
            ).status_code
            == 200
        )
        assert client.post("/__test/cool/internal-key/internal-native").status_code == 200
        assert client.get("/v1/responses/resp_1").status_code == 503
        assert (
            client.post(
                "/v1/responses", json={"model": "fixture-model", "previous_response_id": "resp_1"}
            ).status_code
            == 503
        )
        assert len(calls) == 1
