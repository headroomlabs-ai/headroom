"""Coverage tests for the bedrock client-hook feature.

Targets lines identified in the codecov patch report:
- headroom/cli/proxy.py: _resolve_bedrock_client_hook all branches
- headroom/extras/bedrock_refresh.py: make_client ImportError path
- headroom/providers/registry.py: bedrock_client_factory kwarg routing + TypeError re-raise
"""

from __future__ import annotations

import importlib
import logging
import sys
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# _resolve_bedrock_client_hook
# ---------------------------------------------------------------------------


def _get_resolver():
    from headroom.cli.proxy import _resolve_bedrock_client_hook

    return _resolve_bedrock_client_hook


class TestResolvBedrockClientHook:
    def test_none_returns_none(self):
        assert _get_resolver()(None) is None

    def test_empty_string_returns_none(self):
        assert _get_resolver()("") is None

    def test_whitespace_only_raises(self):
        import click

        with pytest.raises(click.ClickException, match="expected 'module:function'"):
            _get_resolver()("   ")

    def test_missing_colon_raises(self):
        import click

        with pytest.raises(click.ClickException, match="expected 'module:function'"):
            _get_resolver()("mymodule")

    def test_colon_but_no_attr_raises(self):
        import click

        with pytest.raises(click.ClickException, match="expected 'module:function'"):
            _get_resolver()("mymodule:")

    def test_colon_but_no_module_raises(self):
        import click

        with pytest.raises(click.ClickException, match="expected 'module:function'"):
            _get_resolver()(":my_func")

    def test_bad_module_import_raises(self):
        import click

        with pytest.raises(click.ClickException, match="cannot import"):
            _get_resolver()("definitely_not_a_real_module_xyz:some_func")

    def test_attr_not_found_raises(self):
        import click

        # Use 'os' which definitely exists but 'nonexistent_attr' doesn't
        with pytest.raises(click.ClickException, match="not callable"):
            _get_resolver()("os:nonexistent_attr_xyz_99999")

    def test_attr_not_callable_raises(self):
        import click

        # os.sep is a string, not callable
        with pytest.raises(click.ClickException, match="not callable"):
            _get_resolver()("os:sep")

    def test_valid_spec_returns_callable(self):
        # os.path.join is a well-known callable
        result = _get_resolver()("os.path:join")
        import os

        assert result is os.path.join

    def test_whitespace_stripped_from_spec(self):
        # Surrounding whitespace on either side of colon should be stripped
        result = _get_resolver()("  os.path : join  ")
        import os

        assert result is os.path.join


# ---------------------------------------------------------------------------
# bedrock_refresh.make_client — ImportError path
# ---------------------------------------------------------------------------


class TestBedrockRefreshMakeClient:
    def test_returns_none_when_boto3_refresh_session_missing(self):
        """make_client returns None when boto3_refresh_session is not installed."""
        # Temporarily hide boto3_refresh_session from importlib
        original = sys.modules.get("boto3_refresh_session")
        sys.modules["boto3_refresh_session"] = None  # type: ignore[assignment]
        try:
            # Re-import to pick up the module state
            import headroom.extras.bedrock_refresh as mod

            # Force re-execution of make_client in a context where the import fails
            result = mod.make_client("us-east-1")
            assert result is None
        finally:
            if original is None:
                sys.modules.pop("boto3_refresh_session", None)
            else:
                sys.modules["boto3_refresh_session"] = original

    def test_returns_none_when_import_error(self, monkeypatch):
        """make_client gracefully returns None on ImportError."""
        import headroom.extras.bedrock_refresh as mod

        def _raise_import(*_args, **_kwargs):
            raise ImportError("no module")

        monkeypatch.setattr(
            importlib,
            "import_module",
            _raise_import,
        )
        # Patch the try/except by making boto3_refresh_session unavailable
        with patch.dict(sys.modules, {"boto3_refresh_session": None}):  # type: ignore[dict-item]
            result = mod.make_client("us-east-1")
        assert result is None


# ---------------------------------------------------------------------------
# registry.create_proxy_backend — bedrock_client_factory routing + TypeError
# ---------------------------------------------------------------------------


def _make_fake_litellm_backend_cls(raise_on_init=None):
    """Return a minimal LiteLLMBackend stand-in for testing registry logic."""

    class FakeBackend:
        def __init__(self, provider, region, **kwargs):
            self.provider = provider
            self.region = region
            self.kwargs = kwargs
            if raise_on_init is not None:
                raise raise_on_init

    return FakeBackend


class TestCreateProxyBackendBedrockFactory:
    def _call(self, backend_str, factory=None, litellm_cls=None):
        from headroom.providers.registry import create_proxy_backend

        return create_proxy_backend(
            backend=backend_str,
            anyllm_provider="openai",
            bedrock_region="us-east-1",
            logger=logging.getLogger("test"),
            bedrock_client_factory=factory,
            litellm_backend_cls=litellm_cls or _make_fake_litellm_backend_cls(),
        )

    def test_bedrock_provider_receives_factory_kwarg(self):
        sentinel = object()
        instance = self._call("bedrock", factory=lambda r: sentinel)
        assert instance.kwargs.get("bedrock_client_factory") is not None

    def test_non_bedrock_provider_does_not_receive_factory_kwarg(self):
        """factory_kwarg is {} for non-bedrock providers (keeps API surface honest)."""
        instance = self._call("litellm-vertex_ai", factory=lambda r: object())
        assert "bedrock_client_factory" not in instance.kwargs

    def test_none_factory_bedrock_still_works(self):
        instance = self._call("bedrock", factory=None)
        assert instance is not None

    def test_type_error_from_factory_is_reraised(self):
        """TypeError from _build_bedrock_client must propagate, not be swallowed."""
        bad_factory_cls = _make_fake_litellm_backend_cls(raise_on_init=TypeError("bad client"))
        with pytest.raises(TypeError, match="bad client"):
            self._call("bedrock", litellm_cls=bad_factory_cls)

    def test_import_error_returns_none(self):
        """If LiteLLM is not installed, create_proxy_backend returns None gracefully."""

        class NotInstalledBackend:
            def __init__(self, *args, **kwargs):
                raise ImportError("litellm not installed")

        result = self._call("bedrock", litellm_cls=NotInstalledBackend)
        assert result is None
