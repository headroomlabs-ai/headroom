"""A non-Headroom service squatting the wrap port must never be reused (#3360).

Observed live: a caveman gateway listening on 127.0.0.1:8787 satisfied the bare
TCP connect in ``_check_proxy``, so ``wrap claude`` printed "Proxy already
running" and routed traffic through the foreign gateway; the persistent-manifest
path stalled ~45s in recovery then raised "not healthy"; and the dead-marker
self-heal treated the squatter as a live wrapped session, permanently pinning a
stale ANTHROPIC_BASE_URL. These tests pin the foreign-listener detection at all
three sites.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from headroom.cli import wrap as wrap_cli


@contextmanager
def _http_server(handler: type[BaseHTTPRequestHandler]) -> Iterator[int]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class _ForeignHandler(BaseHTTPRequestHandler):
    """Answers TCP and HTTP, but 404s every route (caveman-gateway shape)."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = json.dumps({"error": {"code": "cave_route_not_found"}}).encode()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


class _HeadroomHandler(BaseHTTPRequestHandler):
    """Minimal Headroom-shaped /health responder."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = json.dumps({"version": "0.0.0-test", "config": {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def _closed_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --- _foreign_listener -----------------------------------------------------


def test_foreign_listener_true_for_non_headroom_http() -> None:
    with _http_server(_ForeignHandler) as port:
        assert wrap_cli._foreign_listener(port) is True


def test_foreign_listener_false_for_headroom_health() -> None:
    with _http_server(_HeadroomHandler) as port:
        assert wrap_cli._foreign_listener(port) is False


def test_foreign_listener_false_when_nothing_listens() -> None:
    assert wrap_cli._foreign_listener(_closed_port()) is False


# --- _ensure_proxy_unlocked reuse branch -----------------------------------


def _stub_ensure_proxy_env(monkeypatch: pytest.MonkeyPatch, foreign_port: int) -> dict:
    """Route _ensure_proxy_unlocked around everything but the reuse decision."""
    calls: dict = {"started": None, "recover": 0}
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda _p: None)
    monkeypatch.setattr(wrap_cli, "_find_available_port", lambda start, **_kw: foreign_port + 1)

    def _fake_start_proxy(port: int, **_kw: object) -> object:
        calls["started"] = port
        return object()

    monkeypatch.setattr(wrap_cli, "_start_proxy", _fake_start_proxy)
    return calls


def test_reuse_branch_skips_foreign_listener_and_starts_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _http_server(_ForeignHandler) as port:
        calls = _stub_ensure_proxy_env(monkeypatch, port)
        proc, actual_port = wrap_cli._ensure_proxy_unlocked(port, False)
        assert actual_port == port + 1, "must fall through to the port search"
        assert calls["started"] == port + 1
        assert proc is not None


def test_reuse_branch_still_reuses_real_headroom_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _http_server(_HeadroomHandler) as port:
        calls = _stub_ensure_proxy_env(monkeypatch, port)
        # Health payload has no pid/version mismatch triggers → plain reuse.
        monkeypatch.setattr(wrap_cli, "_proxy_needs_version_restart", lambda _p: False)
        proc, actual_port = wrap_cli._ensure_proxy_unlocked(port, False)
        assert actual_port == port
        assert proc is None
        assert calls["started"] is None


def test_persistent_manifest_on_foreign_port_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Manifest:
        profile = "test-profile"
        health_url = "http://127.0.0.1:1/readyz"

    with _http_server(_ForeignHandler) as port:
        calls = _stub_ensure_proxy_env(monkeypatch, port)
        monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda _p: _Manifest())

        def _no_recover(_p: int) -> bool:
            calls["recover"] += 1
            return False

        monkeypatch.setattr(wrap_cli, "_recover_persistent_proxy", _no_recover)
        proc, actual_port = wrap_cli._ensure_proxy_unlocked(port, False)
        assert calls["recover"] == 0, "recovery can never rebind a squatted port"
        assert actual_port == port + 1
        assert calls["started"] == port + 1


# --- dead-marker self-heal --------------------------------------------------


def _settings(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.local.json"


def test_foreign_listener_with_dead_writer_clears_marker(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    with _http_server(_ForeignHandler) as port:
        wrap_cli._write_claude_wrap_base_url(f"http://127.0.0.1:{port}", settings_path=path)
        wrap_cli._write_wrap_marker(
            path, port=port, key="ANTHROPIC_BASE_URL", previous="http://old.proxy:9000"
        )
        # Simulate the writer having died: stamp a dead pid into the marker.
        marker_path = wrap_cli._wrap_marker_path(path)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["pid"] = 2**22 + 12345  # beyond macOS/Linux default pid ranges
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

        restored = wrap_cli._check_and_clear_dead_wrap_marker(path, key="ANTHROPIC_BASE_URL")
        assert restored == "http://old.proxy:9000"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://old.proxy:9000"


def test_foreign_listener_with_live_writer_keeps_marker(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    with _http_server(_ForeignHandler) as port:
        wrap_cli._write_claude_wrap_base_url(f"http://127.0.0.1:{port}", settings_path=path)
        # Marker written by this (live) process → conservative: never cleared.
        wrap_cli._write_wrap_marker(path, port=port, key="ANTHROPIC_BASE_URL", previous=None)

        restored = wrap_cli._check_and_clear_dead_wrap_marker(path, key="ANTHROPIC_BASE_URL")
        assert restored is None
        assert wrap_cli._wrap_marker_path(path).exists()


def test_headroom_listener_is_still_never_cleared(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    with _http_server(_HeadroomHandler) as port:
        wrap_cli._write_claude_wrap_base_url(f"http://127.0.0.1:{port}", settings_path=path)
        wrap_cli._write_wrap_marker(path, port=port, key="ANTHROPIC_BASE_URL", previous=None)

        restored = wrap_cli._check_and_clear_dead_wrap_marker(path, key="ANTHROPIC_BASE_URL")
        assert restored is None
        assert wrap_cli._wrap_marker_path(path).exists()
