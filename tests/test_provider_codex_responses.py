from headroom.providers.codex.responses import (
    codex_responses_http_url,
    codex_responses_subpath_url,
    codex_responses_websocket_url,
    has_chatgpt_account_header,
    normalize_codex_responses_headers,
)


def test_codex_responses_subpath_url_includes_optional_query() -> None:
    assert (
        codex_responses_subpath_url("items/resp_1", "trace=2")
        == "https://chatgpt.com/backend-api/codex/responses/items/resp_1?trace=2"
    )
    assert (
        codex_responses_subpath_url("compact")
        == "https://chatgpt.com/backend-api/codex/responses/compact"
    )


def test_codex_responses_endpoint_urls_are_provider_owned() -> None:
    assert codex_responses_http_url() == "https://chatgpt.com/backend-api/codex/responses"
    assert (
        codex_responses_http_url("stream=true")
        == "https://chatgpt.com/backend-api/codex/responses?stream=true"
    )
    assert codex_responses_websocket_url() == "wss://chatgpt.com/backend-api/codex/responses"


def test_codex_responses_headers_drop_host_and_resolve_explicit_chatgpt_auth() -> None:
    headers, is_chatgpt_auth = normalize_codex_responses_headers(
        {
            "Host": "localhost:8787",
            "authorization": "Bearer token",
            "chatgpt-account-id": "acct",
            "originator": "Codex Desktop",
        }
    )

    assert is_chatgpt_auth is True
    assert headers == {
        "authorization": "Bearer token",
        "chatgpt-account-id": "acct",
        "originator": "Codex Desktop",
    }
    assert has_chatgpt_account_header(headers) is True


def test_codex_responses_headers_return_false_for_regular_openai_auth() -> None:
    headers, is_chatgpt_auth = normalize_codex_responses_headers(
        {
            "Host": "localhost:8787",
            "authorization": "Bearer sk-proj-test",
        }
    )

    assert is_chatgpt_auth is False
    assert headers == {"authorization": "Bearer sk-proj-test"}
    assert has_chatgpt_account_header(headers) is False
