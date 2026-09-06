"""Regression tests for TOIN composite-key identifiers (#2928)."""

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


class _FakeTOIN:
    def export_patterns(self):
        return {
            "patterns": {
                "unknown|unknown|abcdef1234567890": {
                    "sample_size": 10,
                    "total_compressions": 8,
                    "total_retrievals": 2,
                    "confidence": 0.4,
                    "skip_compression_recommended": False,
                    "optimal_max_items": 20,
                },
                "oauth|gpt-4o|1234567890abcdef": {
                    "sample_size": 5,
                    "total_compressions": 4,
                    "total_retrievals": 1,
                    "confidence": 0.2,
                    "skip_compression_recommended": True,
                    "optimal_max_items": 12,
                },
            }
        }


def test_toin_list_uses_signature_component(monkeypatch) -> None:
    monkeypatch.setattr("headroom.proxy.server.get_toin", lambda: _FakeTOIN())
    app = create_app(ProxyConfig(optimize=False, cache_enabled=False, rate_limit_enabled=False))

    response = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 12345),
    ).get("/v1/toin/patterns")

    assert response.status_code == 200
    assert {row["hash"] for row in response.json()} == {"abcdef123456", "1234567890ab"}


def test_toin_detail_round_trips_list_identifier(monkeypatch) -> None:
    monkeypatch.setattr("headroom.proxy.server.get_toin", lambda: _FakeTOIN())
    app = create_app(ProxyConfig(optimize=False, cache_enabled=False, rate_limit_enabled=False))
    client = TestClient(
        app,
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 12345),
    )

    response = client.get("/v1/toin/pattern/abcdef123456")

    assert response.status_code == 200
    assert response.json()["compressions"] == 8
