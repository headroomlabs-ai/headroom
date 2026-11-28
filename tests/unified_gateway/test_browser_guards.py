from __future__ import annotations

import pytest
from starlette.datastructures import Headers

from headroom.proxy.gateway.auth import validate_gateway_browser_request
from headroom.proxy.gateway.errors import GatewayAuthError


@pytest.mark.parametrize(
    "headers",
    [
        Headers({"origin": "https://evil.example", "host": "127.0.0.1:8787"}),
        Headers({"origin": "null", "host": "127.0.0.1:8787"}),
        Headers({"host": "evil.example"}),
        Headers({"host": "127.0.0.1:8787", "forwarded": "host=evil.example"}),
    ],
)
def test_hostile_browser_or_forwarded_headers_are_rejected(headers: Headers) -> None:
    with pytest.raises(GatewayAuthError):
        validate_gateway_browser_request(headers)


def test_non_browser_loopback_request_is_accepted() -> None:
    validate_gateway_browser_request(Headers({"host": "127.0.0.1:8787"}))
