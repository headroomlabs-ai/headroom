"""Tests for centralized port management utilities."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from headroom.port_utils import (
    DEFAULT_PROXY_PORT,
    _MAX_PORT_SCAN,
    allocate_ports,
    find_opencode_ports,
    format_unbindable_port_error,
    is_headroom_proxy,
)


class TestDefaultPort:
    """Verify the canonical proxy port constant."""

    def test_default_port_is_8787(self):
        assert DEFAULT_PROXY_PORT == 8787

    def test_default_port_is_positive(self):
        assert DEFAULT_PROXY_PORT > 0


class TestAllocatePorts:
    """Port allocation scans for free TCP ports."""

    def test_single_free_port(self):
        ports = allocate_ports(8787, 1)
        assert len(ports) == 1
        assert ports[0] >= 8787

    def test_multiple_free_ports(self):
        ports = allocate_ports(8787, 3)
        assert len(ports) == 3
        assert ports == sorted(ports)
        assert len(set(ports)) == 3

    def test_ports_are_distinct(self):
        ports = allocate_ports(8787, 5)
        assert len(set(ports)) == 5

    def test_skip_occupied_port(self):
        with patch("socket.socket") as mock_socket_class:
            mock_sock = mock_socket_class.return_value.__enter__.return_value
            connect_calls = []

            def side_effect(*args: object, **kwargs: object) -> None:
                connect_calls.append(args)
                if len(connect_calls) == 1:
                    pass  # success — port occupied
                else:
                    raise ConnectionRefusedError

            mock_sock.connect.side_effect = side_effect
            ports = allocate_ports(8787, 1)
            assert len(ports) == 1
            assert ports[0] == 8788

    def test_all_ports_occupied_raises(self):
        with patch("socket.socket") as mock_socket_class:
            mock_sock = mock_socket_class.return_value.__enter__.return_value
            mock_sock.connect.return_value = None  # all succeed = all occupied

            with pytest.raises(ValueError, match="could not find"):
                allocate_ports(8787, 1)

    def test_zero_count_returns_empty(self):
        ports = allocate_ports(8787, 0)
        assert ports == []

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match="count must be"):
            allocate_ports(8787, -1)

    def test_port_zero_raises(self):
        with pytest.raises(ValueError, match="port 0"):
            allocate_ports(0, 1)

    def test_timeout_treated_as_free_port(self):
        """TimeoutError means no service responding — port is free."""
        with patch("socket.socket") as mock_socket_class:
            mock_sock = mock_socket_class.return_value.__enter__.return_value
            mock_sock.connect.side_effect = TimeoutError

            ports = allocate_ports(8787, 1)
            assert len(ports) == 1
            assert ports[0] == 8787

    def test_os_error_treated_as_free_port(self):
        """OSError means port is unusable or blocked but not occupied."""
        with patch("socket.socket") as mock_socket_class:
            mock_sock = mock_socket_class.return_value.__enter__.return_value
            mock_sock.connect.side_effect = OSError

            ports = allocate_ports(8787, 1)
            assert len(ports) == 1
            assert ports[0] == 8787


class TestIsHeadroomProxy:
    """is_headroom_proxy distinguishes Headroom from other services."""

    def test_returns_false_when_nothing_listening(self):
        result = is_headroom_proxy(64123)  # unlikely to be in use
        assert result is False

    def test_returns_false_on_connection_refused(self, monkeypatch: pytest.MonkeyPatch):
        called = []

        class FakeConnection:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def request(self, method: str, url: str) -> None:
                called.append("request")
                raise ConnectionRefusedError

            def getresponse(self) -> None:
                pass

            def close(self) -> None:
                pass

        monkeypatch.setattr("http.client.HTTPConnection", FakeConnection)
        result = is_headroom_proxy(8787, timeout=0.1)
        assert result is False

    def test_returns_true_for_healthy_headroom(self):
        with patch("http.client.HTTPConnection") as mock_conn_class:
            mock_conn = mock_conn_class.return_value
            mock_resp = mock_conn.getresponse.return_value
            mock_resp.status = 200
            mock_resp.read.return_value = b"ok"

            result = is_headroom_proxy(8787, timeout=0.1)
            assert result is True

    def test_returns_false_for_non_200_healthz(self):
        with patch("http.client.HTTPConnection") as mock_conn_class:
            mock_conn = mock_conn_class.return_value
            mock_resp = mock_conn.getresponse.return_value
            mock_resp.status = 503
            mock_resp.read.return_value = b""

            result = is_headroom_proxy(8787, timeout=0.1)
            assert result is False


class TestFindOpencodePorts:
    """find_opencode_ports discovers all OpenCode-managed proxies."""

    def test_returns_empty_when_no_proxies(self):
        with patch("headroom.port_utils.is_headroom_proxy", return_value=False):
            ports = find_opencode_ports(8787, max_scan=3)
            assert ports == []

    def test_returns_running_ports(self):
        def fake_is_headroom(port: int, **kwargs: object) -> bool:
            return port in (8787, 8789)

        with patch("headroom.port_utils.is_headroom_proxy", side_effect=fake_is_headroom):
            ports = find_opencode_ports(8787, max_scan=5)
            assert ports == [8787, 8789]

    def test_respects_base_port(self):
        with patch("headroom.port_utils.is_headroom_proxy", return_value=False):
            ports = find_opencode_ports(8790, max_scan=2)
            assert ports == []


class TestUnbindablePortError:
    """format_unbindable_port_error gives actionable messages."""

    def test_includes_port_and_command(self):
        exc = OSError(98, "Address already in use")
        msg = format_unbindable_port_error(8787, exc, "claude")
        assert "8787" in msg
        assert "headroom wrap claude" in msg

    def test_fallback_for_unknown_agent(self):
        exc = OSError(98, "Address already in use")
        msg = format_unbindable_port_error(8787, exc, "unknown")
        assert "headroom proxy" in msg
        assert "8787" in msg
