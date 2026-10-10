"""`headroom proxy` refuses a busy port before printing its banner.

Without the check the banner ends "Press Ctrl+C to stop", then uvicorn's bind
error is buried among startup and shutdown log lines.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main

pytestmark = [
    pytest.mark.proxy_port_preflight,
    pytest.mark.skipif(sys.platform == "win32", reason="the check is POSIX-only"),
]


@contextmanager
def _listening(host: str = "127.0.0.1") -> Iterator[int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    sock.listen(1)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _invoke_proxy(*args: str) -> tuple[object, list[object]]:
    calls: list[object] = []
    with (
        patch("headroom.proxy.server.run_server", lambda config, **kw: calls.append(config)),
        patch("headroom.cli.port_discovery.probe_headroom_proxy", return_value=False),
    ):
        result = CliRunner().invoke(main, ["proxy", *args])
    return result, calls


@pytest.mark.parametrize("workers", ["1", "2"])
def test_busy_port_fails_before_banner_with_actionable_message(workers: str) -> None:
    with _listening() as port:
        result, calls = _invoke_proxy("--port", str(port), "--workers", workers)

    assert result.exit_code == 1, result.output
    assert f"Port {port} on 127.0.0.1 is already in use by another process" in result.output
    assert "--port N" in result.output
    assert "HEADROOM PROXY" not in result.output
    assert calls == []


def test_busy_port_held_by_headroom_says_so() -> None:
    with _listening() as port:
        with (
            patch("headroom.proxy.server.run_server", lambda config, **kw: None),
            patch("headroom.cli.port_discovery.probe_headroom_proxy", return_value=True),
        ):
            result = CliRunner().invoke(main, ["proxy", "--port", str(port)])

    assert result.exit_code == 1, result.output
    assert "already in use by a running Headroom proxy" in result.output


def test_free_port_starts_normally() -> None:
    port = _free_port()
    result, calls = _invoke_proxy("--port", str(port))

    assert result.exit_code == 0, result.output
    assert "HEADROOM PROXY" in result.output
    assert len(calls) == 1


def _ipv6_loopback_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
    except OSError:
        return False
    return True


class _HeadroomLivez(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        body = json.dumps({"service": "headroom-proxy", "alive": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.mark.skipif(not _ipv6_loopback_available(), reason="needs IPv6 loopback")
def test_ipv6_conflict_is_identified_on_the_requested_host() -> None:
    """A Headroom proxy on 127.0.0.1:P must not be reported as the owner of [::1]:P."""
    headroom_v4 = HTTPServer(("127.0.0.1", 0), _HeadroomLivez)
    port = headroom_v4.server_address[1]
    threading.Thread(target=headroom_v4.serve_forever, daemon=True).start()
    other_v6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        other_v6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        other_v6.bind(("::1", port))
        other_v6.listen(1)
        with patch("headroom.proxy.server.run_server", lambda config, **kw: None):
            v6 = CliRunner().invoke(main, ["proxy", "--host", "::1", "--port", str(port)])
    finally:
        other_v6.close()
        headroom_v4.shutdown()
        headroom_v4.server_close()

    assert v6.exit_code == 1, v6.output
    assert f"Port {port} on ::1 is already in use by another process" in v6.output
    assert "Headroom proxy" not in v6.output


def test_unprobeable_host_gets_the_generic_message() -> None:
    with _listening() as port:
        with (
            patch("headroom.proxy.server.run_server", lambda config, **kw: None),
            patch("headroom.cli.proxy._probe_host_for", return_value=None),
            patch("headroom.cli.port_discovery.probe_headroom_proxy", return_value=True),
        ):
            result = CliRunner().invoke(main, ["proxy", "--port", str(port)])

    assert result.exit_code == 1, result.output
    assert f"Port {port} on 127.0.0.1 is already in use (" in result.output
    assert "Headroom proxy" not in result.output
