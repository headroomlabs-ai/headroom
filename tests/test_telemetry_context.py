"""Tests for headroom.telemetry.context (install_mode + headroom_stack detection)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.telemetry.context import (
    MAX_DISTINCT_STACKS,
    detect_install_mode,
    detect_stack,
    normalize_stack,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts without our env vars set."""

    monkeypatch.delenv("HEADROOM_STACK", raising=False)
    monkeypatch.delenv("HEADROOM_AGENT_TYPE", raising=False)
    yield


class TestDetectInstallMode:
    def test_wrapped_when_agent_type_set(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_AGENT_TYPE", "claude")
        assert detect_install_mode(8787) == "wrapped"

    def test_on_demand_when_no_env_and_no_manifest(self, monkeypatch):
        monkeypatch.setattr("headroom.install.state.list_manifests", lambda: [])
        assert detect_install_mode(8787) == "on_demand"

    def test_persistent_when_manifest_matches_port(self, monkeypatch):
        manifest = SimpleNamespace(port=8787, profile="default")
        monkeypatch.setattr("headroom.install.state.list_manifests", lambda: [manifest])
        assert detect_install_mode(8787) == "persistent"

    def test_on_demand_when_manifest_port_mismatches(self, monkeypatch):
        manifest = SimpleNamespace(port=9000, profile="other")
        monkeypatch.setattr("headroom.install.state.list_manifests", lambda: [manifest])
        assert detect_install_mode(8787) == "on_demand"

    def test_wrapped_takes_precedence_over_manifest(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_AGENT_TYPE", "codex")
        manifest = SimpleNamespace(port=8787, profile="default")
        monkeypatch.setattr("headroom.install.state.list_manifests", lambda: [manifest])
        assert detect_install_mode(8787) == "wrapped"

    def test_manifest_crash_falls_back_to_on_demand(self, monkeypatch):
        def _boom():
            raise RuntimeError("disk gone")

        monkeypatch.setattr("headroom.install.state.list_manifests", _boom)
        # install_mode should not raise; graceful fallback
        assert detect_install_mode(8787) == "on_demand"


class TestDetectStack:
    def test_explicit_env_wins(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_STACK", "custom_slug")
        assert detect_stack() == "custom_slug"

    def test_explicit_env_overrides_agent_type(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_STACK", "proxy")
        monkeypatch.setenv("HEADROOM_AGENT_TYPE", "claude")
        assert detect_stack() == "proxy"

    def test_wrap_slug_from_agent_type(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_AGENT_TYPE", "claude")
        assert detect_stack() == "wrap_claude"

    def test_unknown_agent_type_rejected(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_AGENT_TYPE", "somebespoke")
        assert detect_stack() == "unknown"

    def test_default_is_proxy(self):
        assert detect_stack() == "proxy"

    def test_default_is_proxy_with_empty_stats(self):
        assert detect_stack({"requests": {"by_stack": {}}}) == "proxy"

    def test_dominant_stack_from_stats(self):
        stats = {"requests": {"by_stack": {"adapter_ts_openai": 90, "adapter_ts_anthropic": 10}}}
        assert detect_stack(stats) == "adapter_ts_openai"

    def test_mixed_when_no_dominant_stack(self):
        stats = {"requests": {"by_stack": {"adapter_ts_openai": 40, "adapter_ts_anthropic": 60}}}
        assert detect_stack(stats) == "mixed"

    def test_single_stack_is_dominant(self):
        stats = {"requests": {"by_stack": {"adapter_ts_openai": 3}}}
        assert detect_stack(stats) == "adapter_ts_openai"

    def test_env_beats_stats(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_STACK", "wrap_claude")
        stats = {"requests": {"by_stack": {"adapter_ts_openai": 100}}}
        assert detect_stack(stats) == "wrap_claude"

    def test_invalid_env_falls_through_to_proxy(self, monkeypatch):
        # Garbage env var → normalize_stack rejects → falls back to default
        monkeypatch.setenv("HEADROOM_STACK", "bad slug with spaces!")
        assert detect_stack() == "proxy"

    def test_invalid_env_allows_agent_type_fallback(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_STACK", "Has-Dashes-And-Caps")
        monkeypatch.setenv("HEADROOM_AGENT_TYPE", "claude")
        assert detect_stack() == "wrap_claude"


class TestNormalizeStack:
    def test_empty_and_none(self):
        assert normalize_stack(None) is None
        assert normalize_stack("") is None
        assert normalize_stack("   ") is None

    def test_lowercases_and_strips(self):
        assert normalize_stack("  Wrap_Claude  ") == "wrap_claude"

    def test_rejects_invalid_charset(self):
        assert normalize_stack("has-dashes") is None
        assert normalize_stack("has spaces") is None
        assert normalize_stack("has.dots") is None
        assert normalize_stack("has/slashes") is None
        assert normalize_stack("1_starts_with_digit") is None

    def test_accepts_valid_slugs(self):
        for slug in ("proxy", "wrap_claude", "adapter_ts_openai", "a", "a1_2_3"):
            assert normalize_stack(slug) == slug

    def test_rejects_over_64_chars(self):
        assert normalize_stack("a" * 64) == "a" * 64
        assert normalize_stack("a" * 65) is None


class TestRecordStackValidation:
    """PrometheusMetrics.record_stack must route through normalize_stack and
    respect the cardinality cap."""

    def _metrics(self):
        from headroom.proxy.prometheus_metrics import PrometheusMetrics

        return PrometheusMetrics()

    def test_ignores_invalid_slug(self):
        m = self._metrics()
        m.record_stack("bad slug!")
        m.record_stack("has-dashes")
        m.record_stack("")
        m.record_stack(None)
        assert dict(m.requests_by_stack) == {}

    def test_counts_valid_slug(self):
        m = self._metrics()
        m.record_stack("wrap_claude")
        m.record_stack("WRAP_CLAUDE")
        assert m.requests_by_stack["wrap_claude"] == 2

    def test_cardinality_cap_rejects_new_slugs(self):
        m = self._metrics()
        for i in range(MAX_DISTINCT_STACKS):
            m.record_stack(f"slug_{i}")
        assert len(m.requests_by_stack) == MAX_DISTINCT_STACKS
        m.record_stack("slug_overflow")
        assert "slug_overflow" not in m.requests_by_stack
        # but existing slugs still increment
        m.record_stack("slug_0")
        assert m.requests_by_stack["slug_0"] == 2
