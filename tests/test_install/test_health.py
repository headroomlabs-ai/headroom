from __future__ import annotations

import urllib.error
from email.message import Message
from io import BytesIO

from pytest import MonkeyPatch

from headroom.install.health import probe_json, probe_ready, probe_text


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload

    @property
    def status(self) -> int:
        return 200


def test_probe_json_returns_dict(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=2.0: _Response(b'{"ready": true}'),
    )

    assert probe_json("http://example.test") == {"ready": True}


def test_probe_json_returns_none_for_invalid_payloads(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=2.0: _Response(b"[]"))
    assert probe_json("http://example.test") is None

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=2.0: _Response(b"{"))
    assert probe_json("http://example.test") is None

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=2.0: (_ for _ in ()).throw(urllib.error.URLError("boom")),
    )
    assert probe_json("http://example.test") is None


def test_probe_ready_accepts_ready_and_healthy(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "headroom.install.health.probe_json", lambda url, timeout=2.0: {"ready": True}
    )
    assert probe_ready("http://example.test")

    monkeypatch.setattr(
        "headroom.install.health.probe_json", lambda url, timeout=2.0: {"status": "healthy"}
    )
    assert probe_ready("http://example.test")

    monkeypatch.setattr("headroom.install.health.probe_json", lambda url, timeout=2.0: None)
    assert not probe_ready("http://example.test")


def test_probe_text_returns_status_and_body(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=2.0: _Response(b"headroom_requests_total 1\n"),
    )

    assert probe_text("http://example.test/metrics") == (200, "headroom_requests_total 1\n")


def test_probe_text_keeps_http_error_body(monkeypatch: MonkeyPatch) -> None:
    error = urllib.error.HTTPError(
        "http://example.test/metrics",
        500,
        "Internal Server Error",
        hdrs=Message(),
        fp=BytesIO(b"# scrape_error RuntimeError: boom\n"),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=2.0: (_ for _ in ()).throw(error),
    )

    assert probe_text("http://example.test/metrics") == (
        500,
        "# scrape_error RuntimeError: boom\n",
    )
