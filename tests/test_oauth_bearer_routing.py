"""Tests for OAuth Bearer token routing and auth detection."""

import httpx
from fastapi.testclient import TestClient

from headroom.providers.registry import _is_anthropic_auth as is_anthropic_auth
from headroom.proxy.server import ProxyConfig, create_app

# ---------------------------------------------------------------------------
# Unit tests: the auth classifier the router uses (providers.registry)
# ---------------------------------------------------------------------------


class TestIsAnthropicAuth:
    def test_x_api_key(self):
        assert is_anthropic_auth({"x-api-key": "sk-ant-api03-xxx"}) is True

    def test_anthropic_version(self):
        assert is_anthropic_auth({"anthropic-version": "2023-06-01"}) is True

    def test_bearer_sk_ant_oat(self):
        assert is_anthropic_auth({"authorization": "Bearer sk-ant-oat01-xxx"}) is True

    def test_bearer_sk_ant_api(self):
        """API key prefix in Bearer header still detected as Anthropic."""
        assert is_anthropic_auth({"authorization": "Bearer sk-ant-api03-xxx"}) is True

    def test_bearer_openai(self):
        assert is_anthropic_auth({"authorization": "Bearer sk-proj-xxx"}) is False

    def test_bearer_arbitrary_uuid(self):
        """Non-Anthropic Bearer tokens don't auto-detect as Anthropic."""
        assert is_anthropic_auth({"authorization": "Bearer 1a18a113-ab50-43c8"}) is False

    def test_no_auth_headers(self):
        assert is_anthropic_auth({}) is False

    def test_anthropic_version_plus_bearer(self):
        """Claude Code sends both anthropic-version and Bearer token."""
        assert (
            is_anthropic_auth(
                {
                    "anthropic-version": "2023-06-01",
                    "authorization": "Bearer 1a18a113-ab50-43c8",
                }
            )
            is True
        )

    def test_empty_authorization(self):
        assert is_anthropic_auth({"authorization": ""}) is False

    def test_claude_code_user_agent_with_gateway_bearer(self):
        """Claude Code behind a gateway token is still routed to Anthropic."""
        assert (
            is_anthropic_auth(
                {
                    "authorization": "Bearer 1a18a113-ab50-43c8",
                    "user-agent": "claude-cli/2.1.0 (external, cli)",
                }
            )
            is True
        )


# ---------------------------------------------------------------------------
# Integration tests: /v1/models routing
# ---------------------------------------------------------------------------


class FakeAsyncClient:
    """Captures outbound requests instead of making real HTTP calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def request(self, method, url, **_kwargs):
        self.calls.append((method, url))
        return httpx.Response(200, json={"data": []})

    async def aclose(self) -> None:
        return None


class TestModelsRouting:
    def _make_client(self):
        app = create_app(ProxyConfig())
        client = TestClient(app)
        fake = FakeAsyncClient()
        client.app.state.proxy.http_client = fake
        client.app.state.proxy.ANTHROPIC_API_URL = "https://api.anthropic.test"
        client.app.state.proxy.OPENAI_API_URL = "https://api.openai.test"
        return client, fake

    def test_models_with_x_api_key_routes_anthropic(self):
        with self._make_client() as (client, fake):
            client.get("/v1/models", headers={"x-api-key": "sk-ant-api03-test"})
        assert any("anthropic.test" in url for _, url in fake.calls)

    def test_models_with_anthropic_version_routes_anthropic(self):
        with self._make_client() as (client, fake):
            client.get("/v1/models", headers={"anthropic-version": "2023-06-01"})
        assert any("anthropic.test" in url for _, url in fake.calls)

    def test_models_with_bearer_sk_ant_routes_anthropic(self):
        with self._make_client() as (client, fake):
            client.get(
                "/v1/models",
                headers={"authorization": "Bearer sk-ant-oat01-testtoken"},
            )
        assert any("anthropic.test" in url for _, url in fake.calls)

    def test_models_with_openai_bearer_routes_openai(self):
        with self._make_client() as (client, fake):
            client.get(
                "/v1/models",
                headers={"authorization": "Bearer sk-proj-testtoken"},
            )
        assert any("openai.test" in url for _, url in fake.calls)

    def _make_client(self):
        """Return (client, fake) as a context manager."""
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            app = create_app(ProxyConfig())
            with TestClient(app) as client:
                fake = FakeAsyncClient()
                client.app.state.proxy.http_client = fake
                client.app.state.proxy.ANTHROPIC_API_URL = "https://api.anthropic.test"
                client.app.state.proxy.OPENAI_API_URL = "https://api.openai.test"
                yield client, fake

        return _ctx()


# ---------------------------------------------------------------------------
# Integration tests: catch-all passthrough routing
# ---------------------------------------------------------------------------


class TestCatchAllRouting:
    def _make_client(self):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            app = create_app(ProxyConfig())
            with TestClient(app) as client:
                fake = FakeAsyncClient()
                client.app.state.proxy.http_client = fake
                client.app.state.proxy.ANTHROPIC_API_URL = "https://api.anthropic.test"
                client.app.state.proxy.OPENAI_API_URL = "https://api.openai.test"
                yield client, fake

        return _ctx()

    def test_catchall_anthropic_version_routes_anthropic(self):
        with self._make_client() as (client, fake):
            client.get(
                "/v1/some/unknown/path",
                headers={"anthropic-version": "2023-06-01"},
            )
        assert any("anthropic.test" in url for _, url in fake.calls)

    def test_catchall_bearer_sk_ant_routes_anthropic(self):
        with self._make_client() as (client, fake):
            client.get(
                "/v1/some/unknown/path",
                headers={"authorization": "Bearer sk-ant-oat01-xxx"},
            )
        assert any("anthropic.test" in url for _, url in fake.calls)

    def test_catchall_no_auth_routes_openai(self):
        with self._make_client() as (client, fake):
            client.get("/v1/some/unknown/path")
        assert any("openai.test" in url for _, url in fake.calls)


# ---------------------------------------------------------------------------
# Rate-limit key: OAuth (Bearer-only) callers get their own bucket
# ---------------------------------------------------------------------------

# Same first 16 characters, different tokens: the old ``api_key[:16]`` key
# would have put both callers in one bucket.
_OAUTH_TOKEN_A = "sk-ant-oat01-shared-prefix-AAAA"
_OAUTH_TOKEN_B = "sk-ant-oat01-shared-prefix-BBBB"
_API_KEY_A = "sk-ant-api03-shared-prefix-AAAA"
_API_KEY_B = "sk-ant-api03-shared-prefix-BBBB"


def _anthropic_upstream(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


class TestRateLimitKey:
    """Drive /v1/messages with one request per minute and watch who gets a 429.

    The caller is a direct loopback peer, which ``rate_limit_identity`` trusts,
    so the provider credential is part of the bucket.
    """

    def _client(self, peer: str = "127.0.0.1") -> TestClient:
        app = create_app(
            ProxyConfig(
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=True,
                rate_limit_requests_per_minute=1,
                cost_tracking_enabled=False,
                log_requests=False,
                ccr_inject_tool=False,
                ccr_handle_responses=False,
                ccr_context_tracking=False,
                image_optimize=False,
            )
        )
        app.state.proxy.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(_anthropic_upstream)
        )
        return TestClient(app, base_url="http://127.0.0.1", client=(peer, 50000))

    def _send(self, client: TestClient, headers: dict[str, str]) -> int:
        # anthropic-version routes even a credential-less request to /v1/messages.
        response = client.post(
            "/v1/messages",
            headers={"anthropic-version": "2023-06-01", **headers},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        return response.status_code

    def test_bearer_tokens_get_separate_buckets(self):
        """Two OAuth users on one machine do not share a limit."""
        client = self._client()
        a = {"authorization": f"Bearer {_OAUTH_TOKEN_A}"}
        b = {"authorization": f"Bearer {_OAUTH_TOKEN_B}"}
        assert self._send(client, a) == 200
        assert self._send(client, a) == 429
        assert self._send(client, b) == 200

    def test_x_api_keys_get_separate_buckets(self):
        client = self._client()
        assert self._send(client, {"x-api-key": _API_KEY_A}) == 200
        assert self._send(client, {"x-api-key": _API_KEY_A}) == 429
        assert self._send(client, {"x-api-key": _API_KEY_B}) == 200

    def test_no_credential_gets_the_peer_bucket(self):
        """A credential-less caller is limited, in a bucket apart from a keyed one."""
        client = self._client()
        assert self._send(client, {"authorization": f"Bearer {_OAUTH_TOKEN_A}"}) == 200
        assert self._send(client, {}) == 200
        assert self._send(client, {}) == 429

    def test_remote_caller_cannot_rotate_tokens_for_fresh_buckets(self):
        """Without the proxy token a remote peer is keyed by address only (VAPT 01-F3)."""
        client = self._client(peer="203.0.113.5")
        assert self._send(client, {"authorization": f"Bearer {_OAUTH_TOKEN_A}"}) == 200
        assert self._send(client, {"authorization": f"Bearer {_OAUTH_TOKEN_B}"}) == 429
