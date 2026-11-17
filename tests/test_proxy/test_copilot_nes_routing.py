"""VS Code's ghost text and next edits must reach Copilot's completions host.

Copilot Chat posts ``copilot-nes-*`` models to ``{overrideProxyUrl}/chat/completions``.
``headroom wrap vscode`` points that override and the CAPI override at the same
proxy, so these requests arrive on the chat handler next to ordinary chat turns.
CAPI answers them with ``400 model_not_supported`` on both the generic and the
per-plan host, so VS Code showed no ghost text at all behind Headroom. Pinned at
the send seams: the URL that was built and the credentials sent.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom import copilot_auth  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

BUSINESS_CAPI = "https://api.business.githubcopilot.com"
DEFAULT_COMPLETIONS = "https://copilot-proxy.githubusercontent.com"
NES_MODEL = "copilot-nes-pandia-25"
MINTED = "tid_minted_for_test"


class _Send:
    """Capture what the buffered or streaming arm was about to send."""

    def __init__(self) -> None:
        self.url: str | None = None
        self.headers: dict[str, str] = {}

    async def __call__(self, method, url, headers, body, **kwargs):
        self.url = url
        self.headers = {k.lower(): v for k, v in dict(headers).items()}
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
            request=httpx.Request(method, url),
        )

    async def send_stream(self, request, **kwargs):
        """Stand in for ``http_client.send``: the streaming arm's wire, after auth."""
        self.url = str(request.url)
        self.headers = {k.lower(): v for k, v in request.headers.items()}
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            request=request,
        )


@pytest.fixture(autouse=True)
def _isolated_copilot_state(monkeypatch: pytest.MonkeyPatch):
    class _Token:
        token = MINTED

    class _Provider:
        async def get_api_token(self, integration_id: str | None = None):
            return _Token()

    monkeypatch.setattr(copilot_auth, "get_copilot_token_provider", lambda: _Provider())
    for name in (
        "GITHUB_COPILOT_PROXY_URL",
        "GITHUB_COPILOT_API_URL",
        "GITHUB_COPILOT_ENTERPRISE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    copilot_auth.reset_observed_completions_endpoint()
    yield
    copilot_auth.reset_observed_completions_endpoint()


def _post(
    model: str,
    *,
    openai_api_url: str = BUSINESS_CAPI,
    stream: bool = False,
    headers: dict[str, str] | None = None,
) -> _Send:
    send = _Send()
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        mode="token",
        openai_api_url=openai_api_url,
    )
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        proxy._retry_request = send
        proxy.http_client.send = send.send_stream
        resp = client.post(
            "/chat/completions",
            json={
                "model": model,
                "stream": stream,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "def add(a, b):\n    "}],
            },
            headers={"Authorization": "Bearer vscode-session-token", **(headers or {})},
        )
    assert resp.status_code == 200
    return send


def test_next_edit_models_go_to_the_completions_host_with_copilot_credentials() -> None:
    send = _post(NES_MODEL)

    assert send.url == f"{DEFAULT_COMPLETIONS}/chat/completions"
    assert send.headers["authorization"] == f"Bearer {MINTED}"


def test_streamed_next_edits_go_to_the_completions_host_with_copilot_credentials() -> None:
    send = _post(NES_MODEL, stream=True)

    assert send.url == f"{DEFAULT_COMPLETIONS}/chat/completions"
    assert send.headers["authorization"] == f"Bearer {MINTED}"


def test_next_edit_models_follow_the_completions_host_github_advertised() -> None:
    copilot_auth._remember_completions_endpoint(
        {"endpoints": {"proxy": "https://proxy.business.githubcopilot.com"}}
    )

    send = _post(NES_MODEL)

    assert send.url == "https://proxy.business.githubcopilot.com/chat/completions"


def test_chat_models_stay_on_the_plan_host() -> None:
    send = _post("gpt-4o-mini")

    assert send.url == f"{BUSINESS_CAPI}/chat/completions"


def test_a_ghe_tenant_keeps_its_next_edits_inside_its_deployment() -> None:
    send = _post(NES_MODEL, openai_api_url="https://copilot-api.acme.ghe.com")

    assert send.url is not None
    assert send.url.startswith("https://copilot-api.acme.ghe.com/")


def test_a_configured_enterprise_deployment_keeps_next_edits_with_a_public_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_URL", "https://acme.ghe.com")

    send = _post(NES_MODEL, openai_api_url="https://api.githubcopilot.com")

    assert send.url == "https://copilot-api.acme.ghe.com/chat/completions"


def test_a_completions_host_target_is_not_moved() -> None:
    send = _post(NES_MODEL, openai_api_url="https://proxy.enterprise.githubcopilot.com")

    assert send.url == "https://proxy.enterprise.githubcopilot.com/chat/completions"


def test_a_caller_chosen_base_url_is_left_alone() -> None:
    send = _post(
        NES_MODEL,
        headers={"x-headroom-base-url": "https://api.githubcopilot.com"},
    )

    assert send.url is not None
    assert send.url.startswith("https://api.githubcopilot.com/")
