"""Tests for the upstream-error diagnostic-dump gating.

The dump can contain cleartext prompt/tool/system content, so it must be OFF by
default, never written in stateless mode, and content-redacted unless the
operator explicitly opts in to full content.
"""

from __future__ import annotations

import inspect
import json
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


# ---------------------------------------------------------------------------
# write_upstream_error_dump: the shared writer used by the streaming handler
# ---------------------------------------------------------------------------


@pytest.fixture
def dump_dir(tmp_path, monkeypatch):
    """Point ``paths.debug_400_dir()`` at a temp dir for the writer tests."""
    from headroom import paths

    target = tmp_path / "debug_400"
    monkeypatch.setattr(paths, "debug_400_dir", lambda: target)
    return target


def _write(**kwargs):
    from headroom.proxy.handlers._debug_dump import write_upstream_error_dump

    params = {
        "request_id": "req_1",
        "url": "https://api.anthropic.com/v1/messages",
        "status": 400,
        "provider": "anthropic",
        "model": "claude-opus-5",
        "body": {"messages": [{"role": "user", "content": "secret prompt content " * 20}]},
        "transforms": ["tool_search_history_repair"],
        "stream": True,
    }
    params.update(kwargs)
    config = params.pop("config", _config())
    return write_upstream_error_dump(config, **params)


def test_upstream_dump_off_by_default_writes_nothing(dump_dir, monkeypatch):
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    assert _write() is None
    assert not dump_dir.exists()


def test_upstream_dump_stateless_writes_nothing(dump_dir, monkeypatch):
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")
    assert _write(config=_config(stateless=True)) is None
    assert not dump_dir.exists()


def test_upstream_dump_full_records_request(dump_dir, monkeypatch):
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")
    path = _write()
    assert path is not None and path.parent == dump_dir
    payload = json.loads(path.read_text())
    assert payload["request_id"] == "req_1"
    assert payload["status"] == 400
    assert payload["provider"] == "anthropic"
    assert payload["model"] == "claude-opus-5"
    assert payload["stream"] is True
    assert payload["transforms"] == ["tool_search_history_repair"]
    assert "secret prompt content" in payload["body"]["messages"][0]["content"]


def test_upstream_dump_redacted_elides_content(dump_dir, monkeypatch):
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "1")
    path = _write()
    assert path is not None
    payload = json.loads(path.read_text())
    content = payload["body"]["messages"][0]["content"]
    assert content.startswith("<redacted:") and "secret prompt" not in content
    # Structure is still there to debug with:
    assert payload["body"]["messages"][0]["role"] == "user"


def test_upstream_dump_serializes_non_json_values(dump_dir, monkeypatch):
    # ``default=str`` must keep an exotic body from raising mid-dump.
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")
    path = _write(body={"when": object()})
    assert path is not None
    assert "object object" in json.loads(path.read_text())["body"]["when"]


def test_upstream_dump_never_raises_when_write_fails(monkeypatch):
    # A diagnostic must not turn an upstream error into a proxy error.
    from headroom import paths

    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")

    def _boom():
        raise OSError("read-only filesystem")

    monkeypatch.setattr(paths, "debug_400_dir", _boom)
    assert _write() is None
