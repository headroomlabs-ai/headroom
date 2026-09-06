"""End-to-end test of workspace-root registration through the real HTTP
middleware, including isolation between concurrent sessions on one proxy."""

from __future__ import annotations

import json
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from headroom import paths
from headroom.proxy.server import ProxyConfig, create_app

_TEST_PORT = 18787


def _make_app() -> FastAPI:
    return create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
            port=_TEST_PORT,
        )
    )


def _app_with_registered_cwd_probe() -> FastAPI:
    """Test-only route reading get_registered_cwd() after the middleware runs."""
    from headroom.proxy.project_context import get_registered_cwd

    app = _make_app()

    @app.get("/__test/registered-cwd")
    def _probe() -> dict[str, str | None]:
        return {"registered_cwd": get_registered_cwd()}

    # A catch-all passthrough route registered by create_app() would
    # otherwise shadow this path -- Starlette matches routes in
    # registration order, not by specificity.
    app.router.routes.insert(0, app.router.routes.pop())
    return app


def _write_marker(pid: int, session_token: str, cwd: str) -> None:
    d = paths.proxy_clients_dir(_TEST_PORT)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pid}.json").write_text(
        json.dumps({"pid": pid, "session_token": session_token, "cwd": cwd}),
        encoding="utf-8",
    )


def test_valid_session_token_resolves_registered_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_marker(os.getpid(), "tok-real", str(project_dir))

    client = TestClient(_app_with_registered_cwd_probe())
    resp = client.get("/__test/registered-cwd", headers={"x-headroom-session-token": "tok-real"})
    assert resp.json() == {"registered_cwd": str(project_dir)}


def test_opencode_registration_marker_resolves_through_real_middleware(tmp_path, monkeypatch):
    """Closes the seam the reviewer flagged: proves there's no schema drift
    between what opencode()'s wrap command actually writes
    (wrap._register_proxy_client -- the same function claude() calls, not
    a hand-rolled test marker) and what resolve_registered_cwd() reads,
    using the lowercase header casing the OpenCode transport plugin sends
    (vs. claude's title-case) over the real HTTP middleware."""
    from headroom.cli import wrap as wrap_mod

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    project_dir = tmp_path / "opencode-project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    wrap_mod._register_proxy_client(_TEST_PORT, session_token="tok-opencode")

    client = TestClient(_app_with_registered_cwd_probe())
    resp = client.get(
        "/__test/registered-cwd", headers={"x-headroom-session-token": "tok-opencode"}
    )
    assert resp.json() == {"registered_cwd": str(project_dir)}


def test_missing_or_wrong_token_resolves_to_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_marker(os.getpid(), "tok-real", str(project_dir))

    client = TestClient(_app_with_registered_cwd_probe())
    no_token = client.get("/__test/registered-cwd")
    assert no_token.json() == {"registered_cwd": None}

    wrong_token = client.get(
        "/__test/registered-cwd", headers={"x-headroom-session-token": "tok-guessed"}
    )
    assert wrong_token.json() == {"registered_cwd": None}


def test_spoofed_cwd_header_alone_never_resolves_anything(tmp_path, monkeypatch):
    """A spoofed x-headroom-cwd with no matching registration must never
    surface as the registered root."""
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    client = TestClient(_app_with_registered_cwd_probe())
    resp = client.get("/__test/registered-cwd", headers={"x-headroom-cwd": "/etc"})
    assert resp.json() == {"registered_cwd": None}


def test_two_concurrent_sessions_on_one_shared_proxy_stay_isolated(tmp_path, monkeypatch):
    """Token A must resolve only project A's cwd, never project B's."""
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()

    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _write_marker(os.getpid(), "tok-a", str(project_a))
        _write_marker(child.pid, "tok-b", str(project_b))

        client = TestClient(_app_with_registered_cwd_probe())
        resp_a = client.get("/__test/registered-cwd", headers={"x-headroom-session-token": "tok-a"})
        resp_b = client.get("/__test/registered-cwd", headers={"x-headroom-session-token": "tok-b"})
        assert resp_a.json() == {"registered_cwd": str(project_a)}
        assert resp_b.json() == {"registered_cwd": str(project_b)}
    finally:
        child.kill()
        child.wait()
