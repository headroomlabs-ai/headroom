"""`headroom proxy` refuses a busy port before printing its banner.

Without the check the banner ends "Press Ctrl+C to stop", then uvicorn's bind
error is buried among startup and shutdown log lines.
"""

from __future__ import annotations

import socket
import sys
from collections.abc import Iterator
from contextlib import contextmanager
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
