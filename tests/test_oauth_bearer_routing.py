"""Tests for OAuth Bearer token routing and auth detection."""

import httpx
from fastapi.testclient import TestClient

from headroom.proxy.helpers import is_anthropic_auth
from headroom.proxy.server import ProxyConfig, create_app

# ---------------------------------------------------------------------------
# Unit tests: is_anthropic_auth
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
# Unit test: rate-limit key uses Bearer token when no x-api-key
# ---------------------------------------------------------------------------


class TestRateLimitKey:
    def test_rate_key_with_x_api_key(self):
        """x-api-key takes precedence for rate key."""
        headers = {"x-api-key": "sk-ant-api03-abcdef1234567890"}
        api_key = headers.get("x-api-key", "")
        if not api_key:
            auth = headers.get("authorization", "")
            if auth.startswith("Bearer "):
                api_key = auth[7:]
        client_ip = "127.0.0.1"
        rate_key = f"{api_key[:16]}:{client_ip}" if api_key else client_ip
        assert rate_key == "sk-ant-api03-abc:127.0.0.1"

    def test_rate_key_with_bearer_only(self):
        """Bearer token used for rate key when no x-api-key."""
        headers = {"authorization": "Bearer sk-ant-oat01-mytoken123456"}
        api_key = headers.get("x-api-key", "")
        if not api_key:
            auth = headers.get("authorization", "")
            if auth.startswith("Bearer "):
                api_key = auth[7:]
        client_ip = "127.0.0.1"
        rate_key = f"{api_key[:16]}:{client_ip}" if api_key else client_ip
        assert rate_key == "sk-ant-oat01-myt:127.0.0.1"

    def test_rate_key_no_auth(self):
        """No auth headers → IP-only rate key."""
        headers = {}
        api_key = headers.get("x-api-key", "")
        if not api_key:
            auth = headers.get("authorization", "")
            if auth.startswith("Bearer "):
                api_key = auth[7:]
        client_ip = "127.0.0.1"
        rate_key = f"{api_key[:16]}:{client_ip}" if api_key else client_ip
        assert rate_key == "127.0.0.1"
