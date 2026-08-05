"""ChatGPT Codex backend endpoint formulas."""

from __future__ import annotations

CHATGPT_BACKEND_API_URL = "https://chatgpt.com"
CHATGPT_BACKEND_WS_URL = "wss://chatgpt.com"
CODEX_BACKEND_PREFIX = "/backend-api/codex"


def chatgpt_backend_url(path: str, query: str = "") -> str:
    """Return a ChatGPT HTTPS backend URL for an absolute backend path."""
    url = f"{CHATGPT_BACKEND_API_URL}{path}"
    if query:
        return f"{url}?{query}"
    return url


def codex_backend_url(path: str, query: str = "") -> str:
    """Return a ChatGPT HTTPS backend URL under `/backend-api/codex`."""
    path_suffix = path if path.startswith("/") else f"/{path}"
    return chatgpt_backend_url(f"{CODEX_BACKEND_PREFIX}{path_suffix}", query)


def codex_backend_ws_url(path: str = "") -> str:
    """Return a ChatGPT WebSocket backend URL under `/backend-api/codex`."""
    path_suffix = path if not path or path.startswith("/") else f"/{path}"
    return f"{CHATGPT_BACKEND_WS_URL}{CODEX_BACKEND_PREFIX}{path_suffix}"
