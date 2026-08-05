import json

from headroom.providers.codex.images import (
    codex_image_forward_error_response,
    codex_image_url,
    normalize_codex_image_headers,
    sanitize_codex_image_response_headers,
)


def test_codex_image_url_includes_optional_query() -> None:
    assert (
        codex_image_url("generations", "client_version=0.142.0")
        == "https://chatgpt.com/backend-api/codex/images/generations?client_version=0.142.0"
    )
    assert codex_image_url("edits") == "https://chatgpt.com/backend-api/codex/images/edits"


def test_codex_image_headers_drop_proxy_only_headers_and_resolve_auth() -> None:
    headers, is_chatgpt_auth = normalize_codex_image_headers(
        {
            "Host": "localhost:8787",
            "Authorization": "Bearer token",
            "Accept-Encoding": "gzip",
            "X-Headroom-Bypass": "1",
            "ChatGPT-Account-ID": "acct",
            "Content-Type": "application/json",
        }
    )

    assert is_chatgpt_auth is True
    assert headers == {
        "Authorization": "Bearer token",
        "ChatGPT-Account-ID": "acct",
        "Content-Type": "application/json",
    }


def test_codex_image_response_headers_drop_stale_framing_case_insensitive() -> None:
    assert sanitize_codex_image_response_headers(
        {
            "Content-Encoding": "gzip",
            "Content-Length": "9999",
            "Content-Type": "application/json",
            "x-upstream": "kept",
        }
    ) == {
        "Content-Type": "application/json",
        "x-upstream": "kept",
    }


def test_codex_image_forward_error_response_shape() -> None:
    response = codex_image_forward_error_response()

    assert response.status_code == 502
    assert json.loads(response.body) == {
        "error": {
            "type": "upstream_error",
            "message": "Failed to forward Codex image request",
        }
    }
