"""Tests for upstream API targets in the proxy startup banner.

Verifies that:
1. Default API targets appear in the banner when no overrides are set
2. Custom API targets (via ProxyConfig) appear correctly in the banner
3. The banner is suppressed when print_banner=False

Closes #583.
"""

from io import StringIO
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from headroom.providers.claude import DEFAULT_API_URL as DEFAULT_ANTHROPIC_API_URL  # noqa: E402
from headroom.providers.codex import DEFAULT_API_URL as DEFAULT_OPENAI_API_URL  # noqa: E402
from headroom.providers.gemini import DEFAULT_API_URL as DEFAULT_GEMINI_API_URL  # noqa: E402
from headroom.providers.registry import (  # noqa: E402
    DEFAULT_CLOUDCODE_API_URL,
    DEFAULT_VERTEX_API_URL,
)
from headroom.proxy.models import ProxyConfig  # noqa: E402
from headroom.proxy.server import run_server  # noqa: E402


class TestBannerUpstreamTargets:
    """Verify resolved upstream API targets are displayed in the startup banner."""

    def _capture_banner(self, config: ProxyConfig | None = None) -> str:
        """Run the server with print_banner=True, intercepting stdout and uvicorn."""
        buf = StringIO()
        config = config or ProxyConfig()
        with (
            patch("sys.stdout", buf),
            patch("headroom.proxy.server.uvicorn") as mock_uvicorn,
            patch("headroom.proxy.server.create_app"),
        ):
            mock_uvicorn.run = lambda *a, **kw: None
            run_server(config, print_banner=True)
        return buf.getvalue()

    def test_default_targets_in_banner(self):
        """Default API targets should appear when no overrides are configured."""
        output = self._capture_banner()

        assert "UPSTREAM TARGETS:" in output
        assert DEFAULT_ANTHROPIC_API_URL in output
        assert DEFAULT_OPENAI_API_URL in output
        assert DEFAULT_GEMINI_API_URL in output
        assert DEFAULT_CLOUDCODE_API_URL in output
        assert DEFAULT_VERTEX_API_URL in output

    def test_custom_anthropic_target_in_banner(self):
        """A custom Anthropic API URL should be resolved and shown in the banner."""
        config = ProxyConfig(anthropic_api_url="https://litellm.internal/v1")
        output = self._capture_banner(config)

        # resolve_api_targets strips trailing /v1
        assert "https://litellm.internal" in output
        # Other providers should keep their defaults
        assert DEFAULT_OPENAI_API_URL in output

    def test_custom_openai_target_in_banner(self):
        """A custom OpenAI API URL should be resolved and shown in the banner."""
        config = ProxyConfig(openai_api_url="http://my-vllm:4000")
        output = self._capture_banner(config)

        assert "http://my-vllm:4000" in output
        assert DEFAULT_ANTHROPIC_API_URL in output

    def test_custom_gemini_target_in_banner(self):
        """A custom Gemini API URL should be resolved and shown in the banner."""
        config = ProxyConfig(gemini_api_url="http://my-gemini:5000")
        output = self._capture_banner(config)

        assert "http://my-gemini:5000" in output

    def test_custom_cloudcode_target_in_banner(self):
        """A custom Cloud Code API URL should be resolved and shown in the banner."""
        config = ProxyConfig(cloudcode_api_url="https://custom-cloudcode.example.com")
        output = self._capture_banner(config)

        assert "https://custom-cloudcode.example.com" in output

    def test_custom_vertex_target_in_banner(self):
        """A custom Vertex AI API URL should be resolved and shown in the banner."""
        config = ProxyConfig(vertex_api_url="https://europe-west4-aiplatform.googleapis.com")
        output = self._capture_banner(config)

        assert "https://europe-west4-aiplatform.googleapis.com" in output

    def test_multiple_custom_targets_in_banner(self):
        """Multiple custom targets should all appear correctly in the banner."""
        config = ProxyConfig(
            anthropic_api_url="https://anthropic.internal",
            openai_api_url="https://openai.internal",
            gemini_api_url="https://gemini.internal",
            cloudcode_api_url="https://cloudcode.internal",
            vertex_api_url="https://vertex.internal",
        )
        output = self._capture_banner(config)

        assert "https://anthropic.internal" in output
        assert "https://openai.internal" in output
        assert "https://gemini.internal" in output
        assert "https://cloudcode.internal" in output
        assert "https://vertex.internal" in output

    def test_banner_suppressed_when_disabled(self):
        """When print_banner=False, upstream targets should NOT be printed."""
        buf = StringIO()
        with (
            patch("sys.stdout", buf),
            patch("headroom.proxy.server.uvicorn") as mock_uvicorn,
            patch("headroom.proxy.server.create_app"),
        ):
            mock_uvicorn.run = lambda *a, **kw: None
            run_server(ProxyConfig(), print_banner=False)
        output = buf.getvalue()

        assert "UPSTREAM TARGETS:" not in output

    def test_trailing_v1_stripped_in_banner(self):
        """URLs ending in /v1 should be normalized (stripped) in the banner."""
        config = ProxyConfig(openai_api_url="http://my-proxy:8000/v1")
        output = self._capture_banner(config)

        # resolve_api_targets normalizes /v1 suffix
        assert "http://my-proxy:8000" in output
        assert "http://my-proxy:8000/v1" not in output
