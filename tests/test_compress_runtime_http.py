"""Actual local uvicorn/HTTP exercise of the published compress endpoint."""

import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest


def test_actual_server_compress_contract(tmp_path, monkeypatch):
    for key, path in {
        "HEADROOM_WORKSPACE_DIR": "workspace",
        "HEADROOM_CONFIG_DIR": "config",
        "HEADROOM_SAVINGS_PATH": "savings.json",
        "HEADROOM_SAVINGS_EVENTS_PATH": "events.jsonl",
        "HEADROOM_TOIN_PATH": "toin.json",
    }.items():
        monkeypatch.setenv(key, str(tmp_path / path))
    for key, value in {
        "HEADROOM_OFFLINE": "1",
        "HEADROOM_TELEMETRY": "off",
        "HEADROOM_BEACON": "off",
        "HEADROOM_CC_SWITCH_RECONCILE": "0",
        "HEADROOM_KOMPRESS_BACKGROUND_WARMUP": "0",
    }.items():
        monkeypatch.setenv(key, value)
    httpx = pytest.importorskip("httpx")
    uvicorn = pytest.importorskip("uvicorn")
    native = pytest.importorskip("headroom._core")
    from headroom.proxy.server import ProxyConfig, create_app
    from tests.test_compress_session_mode import _anthropic_turns, _count_message_markers
    from tests.test_proxy_compress_endpoint import TestCompressEndpointLossyInlineMode

    native_path = Path(native.__file__).resolve()
    expected = Path(os.environ.get("HEADROOM_REVIEW_NATIVE_PATH", str(native_path))).resolve()
    assert native_path == expected
    digest = hashlib.sha256(native_path.read_bytes()).hexdigest()
    if expected_digest := os.environ.get("HEADROOM_REVIEW_NATIVE_SHA256"):
        assert digest == expected_digest
    app = create_app(
        ProxyConfig(
            host="127.0.0.1",
            optimize=True,
            offline=True,
            memory_enabled=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            image_optimize=False,
            disable_kompress=True,
            subscription_tracking_enabled=False,
            periodic_toin_stats_enabled=False,
            periodic_malloc_trim_enabled=False,
        )
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", loop="asyncio"))
    worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    cases = []
    health = None
    worker.start()
    try:
        deadline = time.monotonic() + 20
        while not server.started and worker.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started and app.state.ready
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=60
        ) as client:
            health = client.get("/health").status_code
            assert health == 200
            history = TestCompressEndpointLossyInlineMode._read_then_edit()
            for mode in [None, "lossy_inline", "lossless"]:
                payload = {"messages": history, "model": "gpt-4"}
                if mode is not None:
                    payload["config"] = {"mode": mode}
                response = client.post("/v1/compress", json=payload)
                assert response.status_code == 200, response.text
                data = response.json()
                content = data["messages"][2]["content"]
                ok = (
                    content == history[2]["content"]
                    if mode == "lossless"
                    else content.startswith("[Read content stale: /x/a.py")
                    and "hash=" not in content
                    and not data["ccr_hashes"]
                )
                cases.append(
                    {
                        "case": mode or "default",
                        "status": response.status_code,
                        "ok": ok,
                        "hashes": data["ccr_hashes"],
                        "read_bytes_preserved": content == history[2]["content"],
                        "contains_retrieval_hash": "hash=" in content,
                        "tokens_saved": data["tokens_saved"],
                    }
                )
            for marked in (False, True):
                for turn in range(1, 4):
                    messages = _anthropic_turns(turn)
                    if not marked:
                        del messages[-1]["content"][-1]["cache_control"]
                    response = client.post(
                        "/v1/compress",
                        json={
                            "messages": messages,
                            "model": "claude-sonnet-5",
                            "config": {"session_id": f"actual-http-{marked}"},
                        },
                    )
                    assert response.status_code == 200, response.text
                    data = response.json()
                    count = _count_message_markers(data["messages"])
                    cases.append(
                        {
                            "case": f"session-marked-{marked}-turn-{turn}",
                            "status": response.status_code,
                            "markers": count,
                            "ok": count == (1 if marked else 0),
                            "tokens_saved": data["tokens_saved"],
                        }
                    )
    finally:
        server.should_exit = True
        worker.join(timeout=15)
        listener.close()
    report = {
        "native_path": str(native_path),
        "native_sha256": digest,
        "port": port,
        "health": health,
        "cases": cases,
        "shutdown_joined": not worker.is_alive(),
        "scope": "Actual uvicorn and loopback HTTP with normal endpoint/pipeline;no provider upstream,billing,MLmodel or productiondeployment claim",
    }
    Path(
        os.environ.get("REVIEW4678_RUNTIME_REPORT", str(tmp_path / "runtime-report.json"))
    ).write_text(json.dumps(report, indent=2))
    print("ACTUAL_HTTP_REPORT " + json.dumps(report))
    assert not worker.is_alive()
    assert any(x["tokens_saved"] > 0 for x in cases), "compression did not execute"
    assert all(x["ok"] for x in cases), report
