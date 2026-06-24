from __future__ import annotations

from headroom.providers.claude import DEFAULT_API_URL, proxy_base_url


def test_claude_runtime_exposes_default_api_and_local_proxy_url() -> None:
    # Arrange / Act / Assert
    assert DEFAULT_API_URL == "https://api.anthropic.com"
    assert proxy_base_url(4321) == "http://127.0.0.1:4321"
