from headroom.providers.codex.endpoints import (
    chatgpt_backend_url,
    codex_backend_url,
    codex_backend_ws_url,
)


def test_chatgpt_backend_url_appends_optional_query() -> None:
    assert chatgpt_backend_url("/backend-api/me") == "https://chatgpt.com/backend-api/me"
    assert (
        chatgpt_backend_url("/backend-api/me", "client_version=1")
        == "https://chatgpt.com/backend-api/me?client_version=1"
    )


def test_codex_backend_url_normalizes_path_prefix() -> None:
    assert codex_backend_url("models") == "https://chatgpt.com/backend-api/codex/models"
    assert (
        codex_backend_url("/responses/compact", "trace=1")
        == "https://chatgpt.com/backend-api/codex/responses/compact?trace=1"
    )


def test_codex_backend_ws_url_normalizes_path_prefix() -> None:
    assert codex_backend_ws_url() == "wss://chatgpt.com/backend-api/codex"
    assert codex_backend_ws_url("responses") == "wss://chatgpt.com/backend-api/codex/responses"
    assert codex_backend_ws_url("/responses") == "wss://chatgpt.com/backend-api/codex/responses"
