"""Tests for the complete stateless write guarantee.

Covers the process-wide stateless flag and the opt-in serving writers gated by
it: the output-savings recorder and persistent memory. (Savings tracker and
TOIN are covered in their own test modules.)
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from headroom import paths
from headroom.proxy.output_savings import SavingsRecorder
from headroom.relevance.embedding import (
    _DEFAULT_MODEL_PINNED_REVISION,
    DEFAULT_MODEL_NAME,
    _pinned_revision,
)


@pytest.fixture(autouse=True)
def _reset_stateless_globals():
    """The stateless flag and TOIN singleton are process-global — never leak."""
    yield
    paths.set_process_stateless(False)
    try:
        from headroom.telemetry.toin import reset_toin

        reset_toin()
    except Exception:
        pass


# ---- process-wide stateless flag ------------------------------------------


def test_process_stateless_flag_set_and_clear(monkeypatch):
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    paths.set_process_stateless(False)
    assert paths.process_is_stateless() is False
    paths.set_process_stateless(True)
    assert paths.process_is_stateless() is True


@pytest.mark.parametrize(
    "value,expected", [("1", True), ("true", True), ("on", True), ("off", False), ("", False)]
)
def test_process_stateless_env(monkeypatch, value, expected):
    paths.set_process_stateless(False)
    monkeypatch.setenv("HEADROOM_STATELESS", value)
    assert paths.process_is_stateless() is expected


# ---- output-savings recorder ----------------------------------------------


def test_output_savings_flush_writes_nothing_when_stateless(tmp_path, monkeypatch):
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    path = tmp_path / "output_savings.json"
    paths.set_process_stateless(True)
    rec = SavingsRecorder(path)
    rec.flush()
    assert not path.exists()


def test_output_savings_flush_persists_when_not_stateless(tmp_path, monkeypatch):
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    path = tmp_path / "output_savings.json"
    paths.set_process_stateless(False)
    rec = SavingsRecorder(path)
    rec.flush()
    assert path.exists()


# ---- persistent memory ----------------------------------------------------


def test_memory_disabled_under_stateless(tmp_path, monkeypatch):
    """A stateless proxy with --memory must not initialize memory or write a DB."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(ProxyConfig(memory_enabled=True, stateless=True))
    proxy = app.state.proxy
    assert proxy.memory_handler is None
    assert not (tmp_path / ".headroom" / "memory.db").exists()


# ---- fastembed model pinning ----------------------------------------------


def test_fastembed_default_model_is_pinned_to_sha(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    rev = _pinned_revision(DEFAULT_MODEL_NAME)
    assert rev == _DEFAULT_MODEL_PINNED_REVISION
    assert len(rev) == 40 and all(c in "0123456789abcdef" for c in rev)


def test_fastembed_custom_model_not_pinned(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    assert _pinned_revision("intfloat/e5-small-v2") is None


def test_fastembed_pin_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HEADROOM_HF_PIN", "off")
    assert _pinned_revision(DEFAULT_MODEL_NAME) is None
