"""Catalog decisions observed across real HTTP/TLS process boundaries."""

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from tests.unified_gateway.process.harness import _unused_loopback_port, gateway_config_digest
from tests.unified_gateway.test_gateway_tls import local_pki  # noqa: F401


@pytest.fixture
def catalog_process(local_pki, tmp_path):  # noqa: F811
    context, raw = local_pki
    state = {"metadata_calls": 0, "generation_calls": 0, "fail": False, "pause": False}
    accepted, release = threading.Event(), threading.Event()
    literal = (
        b'{ "id": "reply-a", "choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1} }'
    )

    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def respond(self, code, body):
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state["metadata_calls"] += 1
            assert self.path == "/v1/models"
            assert self.headers["authorization"] == "Bearer fixture-key"
            self.respond(
                503 if state["fail"] else 200,
                b'{"error":"private-upstream-error"}'
                if state["fail"]
                else json.dumps(
                    {
                        "data": [
                            {
                                "id": "fixture-model",
                                "capabilities": {
                                    "generate": True,
                                    "stream": True,
                                    "features": ["text"],
                                },
                            }
                        ]
                    }
                ).encode(),
            )

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            state["generation_calls"] += 1
            accepted.set()
            if state["pause"]:
                assert release.wait(15)
            self.respond(200, literal)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    raw["runtime"]["port"] = _unused_loopback_port()
    raw["transport"]["ca_bundle"] = str(tmp_path / "ca.pem")
    raw["client_auth"]["principals"].append(
        {"id": "operator", "secret_ref": "env:OPERATOR_TOKEN", "scopes": ["admin"], "routes": []}
    )
    origin = f"https://llm.internal.example:{server.server_port}"
    raw["credentials"][0]["allowed_origins"] = [origin]
    route = raw["routes"][0]
    route.update(
        upstream_origin=origin, public_model="fixture-model", upstream_model="fixture-model"
    )
    route["ingress_protocols"] = route["native_protocols"] = [
        "openai-chat",
        "openai-responses",
        "gemini-generate",
    ]
    route["capabilities"] = {
        p: {"http-json": {"features": ["text"]}} for p in route["ingress_protocols"]
    }
    route["catalog"] = {
        "source": "provider",
        "ttl_seconds": 10,
        "stale_if_error_seconds": 5,
        "entitlements": {"internal-key": "allowed"},
    }
    route["pricing"] = {
        "input_usd_per_million": "1",
        "output_usd_per_million": "2",
        "revision": "tariff-a",
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(raw))
    environment = {
        k: v
        for k, v in os.environ.items()
        if k.upper()
        in {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP"}
    }
    environment.update(
        HEADROOM_GATEWAY_CLIENT_TOKEN="client-secret",
        OPERATOR_TOKEN="admin-secret",
        INTERNAL_LLM_API_KEY="fixture-key",
        HEADROOM_REQUIRE_RUST_CORE="false",
        HEADROOM_SKIP_UPDATE_CHECK="1",
        HEADROOM_SKIP_UPSTREAM_CHECK="1",
        PYTHONIOENCODING="utf-8",
    )
    (tmp_path / "private-home").mkdir()
    environment.update(
        HOME=str(tmp_path / "private-home"), USERPROFILE=str(tmp_path / "private-home")
    )
    repository = Path(__file__).parents[3]
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.unified_gateway.process.runtime_server", str(path)],
        cwd=repository,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    output = []
    reader = threading.Thread(target=lambda: output.extend(process.stdout), daemon=True)
    reader.start()

    class LocalClient(httpx.Client):
        def send(self, request, **kwargs):
            try:
                return super().send(request, **kwargs)
            except httpx.ReadTimeout:
                pass
            raise AssertionError(
                f"Local gateway timed out: {request.method} {request.url.path}\n{''.join(output)}"
            )

    client = LocalClient(
        base_url=f"http://127.0.0.1:{raw['runtime']['port']}",
        headers={"authorization": "Bearer client-secret"},
        timeout=5,
    )
    admin = {"authorization": "Bearer admin-secret"}
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail("".join(output))
            try:
                response = client.get("/readyz", timeout=0.2)
                if response.status_code == 200:
                    readiness = response.json()
                    if (
                        process.poll() is None
                        and readiness.get("service") == "headroom"
                        and readiness.get("profile") == "gateway"
                        and readiness.get("ready") is True
                        and readiness.get("config_digest") == gateway_config_digest(raw)
                    ):
                        break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        else:
            pytest.fail("owned gateway process did not become ready")
        yield client, admin, state, raw, path, accepted, release, literal
    finally:
        release.set()
        client.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        reader.join(timeout=3)
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_denied_entitlement_hides_model_and_blocks_generation(catalog_process):
    client, admin, state, raw, path, *_ = catalog_process
    assert client.post("/admin/gateway/catalog/refresh", headers=admin).json()["refreshed"] == 1
    assert client.get("/v1/models/fixture-model").status_code == 200
    assert (
        client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
        ).status_code
        == 200
    )
    raw["routes"][0]["catalog"]["entitlements"]["internal-key"] = "denied"
    path.write_text(json.dumps(raw))
    assert client.post("/admin/gateway/reload", headers=admin).json()["applied"]
    # Populate the replacement generation too: entitlement is the only denial,
    # not an empty provider cache or a broken upstream/identity fixture.
    assert client.post("/admin/gateway/catalog/refresh", headers=admin).json()["refreshed"] == 1
    identity_before = client.get("/__test/probe").json()["identity"]
    calls_before = state["metadata_calls"], state["generation_calls"]
    assert client.get("/v1/models").json()["data"] == []
    assert client.get("/v1/models/fixture-model").status_code == 404
    assert client.get("/v1beta/models").json()["models"] == []
    assert (
        client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
        ).status_code
        == 404
    )
    assert client.get("/__test/probe").json()["identity"] == identity_before
    assert (state["metadata_calls"], state["generation_calls"]) == calls_before


def test_failed_catalog_refresh_is_explicitly_stale_then_unavailable(catalog_process):
    client, admin, state, *_ = catalog_process
    assert client.get("/v1/models").json()["data"] == []
    assert client.post("/admin/gateway/catalog/refresh", headers=admin).json()["refreshed"] == 1
    metadata = client.get("/v1/models/fixture-model").json()["headroom"]
    assert metadata["provenance"] == "provider" and metadata["state"] == "fresh"
    state["fail"] = True
    client.post("/__test/clock/1011")
    assert client.post("/admin/gateway/catalog/refresh", headers=admin).json()["failed"] == 1
    assert client.get("/v1/models/fixture-model").json()["headroom"]["state"] == "stale"
    client.post("/__test/clock/1016")
    assert client.get("/v1/models").json()["data"] == []
    assert (
        client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
        ).status_code
        == 404
    )
    assert state["metadata_calls"] == 2 and state["generation_calls"] == 0


def test_capability_rejection_has_zero_identity_or_generation_calls(catalog_process):
    client, admin, state, raw, path, *_ = catalog_process
    raw["routes"][0]["catalog"]["source"] = "configured"
    path.write_text(json.dumps(raw))
    assert client.post("/admin/gateway/reload", headers=admin).json()["applied"]
    payloads = [
        {"stream": True},
        {"tools": [{"type": "function"}]},
        {"parallel_tool_calls": True},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                    ],
                }
            ]
        },
        {"response_format": {"type": "json_object"}},
        {"input": [{"type": "reasoning", "encrypted_content": "private"}]},
        {"tools": [{"type": "web_search"}]},
        {"input": [{"type": "input_audio"}]},
    ]
    for payload in payloads:
        result = client.post(
            "/v1/chat/completions", json={"model": "fixture-model", "messages": [], **payload}
        )
        assert result.status_code == 400
        assert result.json()["error"]["code"] == "gateway_unsupported_capability"
    assert client.get("/__test/probe").json()["identity"] == 0
    assert state["metadata_calls"] == state["generation_calls"] == 0


def test_inflight_catalog_refresh_preserves_metadata_and_tariff(catalog_process):
    client, admin, state, raw, path, accepted, release, literal = catalog_process
    assert client.post("/admin/gateway/catalog/refresh", headers=admin).json()["refreshed"] == 1
    old_revision = client.get("/v1/models/fixture-model").json()["headroom"]["catalog_revision"]
    state["pause"] = True
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pending = executor.submit(
            client.post, "/v1/chat/completions", json={"model": "fixture-model", "messages": []}
        )
        assert accepted.wait(5)
        raw["routes"][0]["pricing"]["revision"] = "tariff-b"
        path.write_text(json.dumps(raw))
        assert client.post("/admin/gateway/reload", headers=admin).json()["applied"]
        assert client.post("/admin/gateway/catalog/refresh", headers=admin).json()["refreshed"] == 1
        assert (
            client.get("/v1/models/fixture-model").json()["headroom"]["tariff_revision"]
            == "tariff-b"
        )
        release.set()
        assert pending.result().content == literal
    state["pause"] = False
    assert (
        client.post("/v1/chat/completions", json={"model": "fixture-model", "messages": []}).content
        == literal
    )
    observations = client.get("/__test/probe").json()["observations"]
    assert observations[0] == {
        "generation": 1,
        "catalog_revision": old_revision,
        "tariff": "tariff-a",
    }
    assert observations[1]["generation"] == 2 and observations[1]["tariff"] == "tariff-b"
