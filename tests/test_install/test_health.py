from __future__ import annotations

import urllib.error

from headroom.install.health import probe_json, probe_ready


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_probe_json_returns_dict(monkeypatch) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=2.0: _Response(b'{"ready": true}'),
    )

    assert probe_json("http://example.test") == {"ready": True}


def test_probe_json_returns_none_for_invalid_payloads(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=2.0: _Response(b"[]"))
    assert probe_json("http://example.test") is None

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=2.0: _Response(b"{"))
    assert probe_json("http://example.test") is None

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout=2.0: (_ for _ in ()).throw(urllib.error.URLError("boom")),
    )
    assert probe_json("http://example.test") is None


def test_probe_ready_accepts_ready_and_healthy(monkeypatch) -> None:
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
