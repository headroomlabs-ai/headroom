"""Tests for the MiniMax upstream guardrail in HeadroomProxy.

The guardrail prevents a misconfigured LaunchAgent from silently routing
traffic to api.anthropic.com when the plist is supposed to be a "MiniMax
profile" (env marker: MINIMAX_SESSION_TOKEN). Without it, a forgotten
ANTHROPIC_TARGET_API_URL in a MiniMax plist would burn real Claude Code
quota — which is exactly the failure mode we hit on 2026-06-24.

These tests verify the helper logic only — they don't boot the proxy.
The full server-side integration is exercised in test_proxy_handler_helpers
through the SimpleNamespace config path.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from headroom.proxy.server import HeadroomProxy


class TestIsMiniMaxUpstream:
    """URL classification — both gateway and direct API are accepted."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://agent.minimax.io/mavis/api/v1/llm",
            "https://agent.minimax.io:443/mavis/api/v1/llm/v1",
            "http://agent.minimax.io/",
            "https://minimax.io",
            "https://api.minimaxi.com/anthropic",
            "https://api.minimaxi.com:443/anthropic/v1/messages",
        ],
    )
    def test_minimax_hosts_recognised(self, url: str) -> None:
        assert HeadroomProxy._is_minimax_upstream(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            # The literal Anthropic default — this is the dangerous drift
            # we're guarding against.
            "https://api.anthropic.com",
            "https://api.anthropic.com/v1/messages",
            # OpenAI / other providers
            "https://api.openai.com/v1",
            # Localhost (test doubles, not a real upstream)
            "http://127.0.0.1:8787",
            # Malformed URL — should fall through to False, not raise.
            "not-a-url",
            "",
        ],
    )
    def test_non_minimax_hosts_rejected(self, url: str) -> None:
        assert HeadroomProxy._is_minimax_upstream(url) is False

    def test_case_insensitive_host_match(self) -> None:
        # Host normalisation happens in urlparse().hostname.lower() — verify
        # that uppercase host parts are still recognised as MiniMax.
        assert (
            HeadroomProxy._is_minimax_upstream("https://AGENT.MINIMAX.IO/foo")
            is True
        )


class TestCheckMiniMaxUpstreamGuardrail:
    """The actual guardrail method — emits ERROR when the drift is detected."""

    def _make_proxy(
        self,
        *,
        minimax_session_token: str | None = None,
        anthropic_url: str,
    ) -> SimpleNamespace:
        """Build a stub that proxies the guardrail methods to the real class.

        The guardrail calls ``self._is_minimax_upstream(...)``, so we wire
        the SimpleNamespace's ``_is_minimax_upstream`` to the real
        staticmethod via __class__ trickery — too clever. Easier: just
        give the stub a real bound method.
        """
        proxy = SimpleNamespace(
            config=SimpleNamespace(
                minimax_session_token=minimax_session_token,
            ),
        )
        # Bind the static method onto the instance so `self._is_minimax_upstream`
        # works in the production code path. Also bind the new
        # _is_minimax_upstream_for_self wrapper which the guardrail calls.
        proxy._is_minimax_upstream = HeadroomProxy._is_minimax_upstream
        proxy._is_minimax_upstream_for_self = (
            HeadroomProxy._is_minimax_upstream_for_self.__get__(proxy)
        )
        # The guardrail reads from the class-level ANTHROPIC_API_URL, so
        # we patch the class attribute (restored by the fixture below).
        HeadroomProxy.ANTHROPIC_API_URL = anthropic_url
        return proxy

    @pytest.fixture(autouse=True)
    def _restore_class_attr(self):
        """Don't leak test values into the real class attribute."""
        original = HeadroomProxy.ANTHROPIC_API_URL
        yield
        HeadroomProxy.ANTHROPIC_API_URL = original

    def test_no_minimax_marker_no_log(self, monkeypatch, caplog) -> None:
        """Regular Anthropic-mode proxy (no MiniMax markers) → silent."""
        monkeypatch.delenv("ANTHROPIC_TARGET_API_URL", raising=False)
        monkeypatch.delenv("MINIMAX_SESSION_TOKEN", raising=False)
        proxy = self._make_proxy(
            anthropic_url="https://api.anthropic.com",
        )
        with caplog.at_level(logging.INFO, logger="headroom.proxy"):
            HeadroomProxy._check_minimax_upstream_guardrail(proxy)
        # No error / no warn from the guardrail.
        assert not any(
            "minimax_guardrail" in record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        )

    def test_minimax_env_target_with_minimax_upstream_no_log(
        self, monkeypatch, caplog
    ) -> None:
        """Happy path — MiniMax ANTHROPIC_TARGET_API_URL + MiniMax upstream
        → info log only (or nothing), no warning."""
        monkeypatch.setenv(
            "ANTHROPIC_TARGET_API_URL",
            "https://agent.minimax.io/mavis/api/v1/llm",
        )
        proxy = self._make_proxy(
            anthropic_url="https://agent.minimax.io/mavis/api/v1/llm",
        )
        with caplog.at_level(logging.INFO, logger="headroom.proxy"):
            HeadroomProxy._check_minimax_upstream_guardrail(proxy)
        assert not any(
            "minimax_guardrail" in record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        )

    def test_minimax_env_target_with_anthropic_upstream_logs_error(
        self, monkeypatch, caplog
    ) -> None:
        """The dangerous drift — MiniMax ANTHROPIC_TARGET_API_URL but the
        resolved Anthropic upstream is api.anthropic.com."""
        monkeypatch.setenv(
            "ANTHROPIC_TARGET_API_URL",
            "https://agent.minimax.io/mavis/api/v1/llm",
        )
        proxy = self._make_proxy(
            anthropic_url="https://api.anthropic.com",  # wrong
        )
        with caplog.at_level(logging.ERROR, logger="headroom.proxy"):
            HeadroomProxy._check_minimax_upstream_guardrail(proxy)
        errors = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR and "minimax_guardrail" in r.message
        ]
        assert len(errors) == 1
        # The error message must surface the offending URL so the user can
        # grep for it in `~/.headroom/logs/proxy.log`.
        assert "https://api.anthropic.com" in errors[0].message
        assert "ANTHROPIC_TARGET_API_URL" in errors[0].message

    def test_session_token_marker_with_anthropic_upstream_logs_error(
        self, monkeypatch, caplog
    ) -> None:
        """MINIMAX_SESSION_TOKEN env var still works as a fallback marker
        (for headroom configurations that pass the JWT via env)."""
        monkeypatch.delenv("ANTHROPIC_TARGET_API_URL", raising=False)
        monkeypatch.setenv("MINIMAX_SESSION_TOKEN", "eyJ-from-env")
        proxy = self._make_proxy(
            anthropic_url="https://api.anthropic.com",
        )
        with caplog.at_level(logging.ERROR, logger="headroom.proxy"):
            HeadroomProxy._check_minimax_upstream_guardrail(proxy)
        errors = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR and "minimax_guardrail" in r.message
        ]
        assert len(errors) == 1
