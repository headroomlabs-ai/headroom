"""Tests for the upstream-error diagnostic-dump gating.

The dump can contain cleartext prompt/tool/system content, so it must be OFF by
default, never written in stateless mode, and content-redacted unless the
operator explicitly opts in to full content.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from headroom.proxy.handlers._debug_dump import _debug_dump_mode, _redact_debug_value


def _config(stateless: bool = False) -> SimpleNamespace:
    return SimpleNamespace(stateless=stateless)


def test_debug_dump_off_by_default(monkeypatch):
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    assert _debug_dump_mode(_config()) == "off"


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "redacted", "REDACTED"])
def test_debug_dump_opt_in_redacted(monkeypatch, value):
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", value)
    assert _debug_dump_mode(_config()) == "redacted"


@pytest.mark.parametrize("value", ["full", "all", "content"])
def test_debug_dump_opt_in_full(monkeypatch, value):
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", value)
    assert _debug_dump_mode(_config()) == "full"


def test_debug_dump_unknown_value_is_off(monkeypatch):
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "maybe")
    assert _debug_dump_mode(_config()) == "off"


def test_stateless_forces_dump_off_even_when_opted_in(monkeypatch):
    # Stateless mode must win over any opt-in: no filesystem writes, period.
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")
    assert _debug_dump_mode(_config(stateless=True)) == "off"


def test_redact_elides_long_strings_keeps_structure():
    payload = {
        "role": "user",
        "type": "text",
        "id": "msg_123",
        "text": "secret prompt content " * 20,  # long → redacted
        "blocks": [
            {"type": "tool_use", "name": "search", "input": "x" * 500},
            {"type": "text", "text": "short"},
        ],
    }
    out = _redact_debug_value(payload)
    # Short structural fields preserved:
    assert out["role"] == "user"
    assert out["type"] == "text"
    assert out["id"] == "msg_123"
    assert out["blocks"][0]["name"] == "search"
    assert out["blocks"][1]["text"] == "short"
    # Long content elided to a length placeholder (no original content leaks):
    assert out["text"].startswith("<redacted:") and "secret prompt" not in out["text"]
    assert out["blocks"][0]["input"].startswith("<redacted:")


def test_redact_passes_through_non_strings():
    assert _redact_debug_value(42) == 42
    assert _redact_debug_value(None) is None
    assert _redact_debug_value(True) is True


@pytest.mark.parametrize("module_name", ["anthropic", "openai"])
def test_both_handlers_gate_the_dump(module_name):
    """Regression guard: every handler that writes a debug dump must gate it on
    _debug_dump_mode (off by default). Prevents reintroducing an unguarded dump
    that writes cleartext prompts to disk."""
    import importlib

    module = importlib.import_module(f"headroom.proxy.handlers.{module_name}")
    src = inspect.getsource(module)
    if "debug_400_dir(" in src:
        assert "_debug_dump_mode(self.config)" in src, (
            f"{module_name} writes a debug dump but does not gate it on _debug_dump_mode"
        )
        assert 'if dump_mode != "off":' in src, (
            f"{module_name} debug dump is not guarded by an off-by-default check"
        )
