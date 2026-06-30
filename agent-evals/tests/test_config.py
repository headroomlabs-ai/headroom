"""Tests for the configuration surface (defaults + env overrides)."""

from __future__ import annotations

import pytest

from agent_evals.config import Settings
from agent_evals.models import Provider


def test_defaults() -> None:
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.stats.k_runs == 10
    assert s.stats.margin_lossy_pp == pytest.approx(2.0)
    assert s.stats.margin_ccr_pp == pytest.approx(0.0)
    assert s.proxy.port_range_start < s.proxy.port_range_end
    assert s.proxy.headroom_cmd == ["headroom", "proxy"]


def test_env_override_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EVALS_CONCURRENCY", "8")
    monkeypatch.setenv("AGENT_EVALS_MODEL_SNAPSHOT", "gpt-5.2")
    monkeypatch.setenv("AGENT_EVALS_PROVIDER", "openai")
    s = Settings()
    assert s.concurrency == 8
    assert s.model_snapshot == "gpt-5.2"
    assert s.provider == Provider.OPENAI


def test_env_override_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EVALS_STATS__K_RUNS", "20")
    monkeypatch.setenv("AGENT_EVALS_STATS__MARGIN_LOSSY_PP", "1.5")
    monkeypatch.setenv("AGENT_EVALS_PROXY__READYZ_TIMEOUT_S", "45")
    s = Settings()
    assert s.stats.k_runs == 20
    assert s.stats.margin_lossy_pp == pytest.approx(1.5)
    assert s.proxy.readyz_timeout_s == pytest.approx(45.0)


def test_alpha_bounds_validated() -> None:
    with pytest.raises(ValueError):
        Settings(stats={"alpha": 1.5})  # type: ignore[arg-type]
