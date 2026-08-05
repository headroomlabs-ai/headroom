"""Tests for the cc-switch reconciler.

The reconciler keeps Headroom in the request path while cc-switch overwrites
``~/.claude/settings.json`` on every provider switch. See
``headroom/proxy/cc_switch_reconciler.py``.
"""

from __future__ import annotations

import json
import os

import pytest

from headroom.proxy.cc_switch_reconciler import CCSwitchReconciler

PROXY = "http://127.0.0.1:8787"
DEFAULT = "https://api.anthropic.com"


def _make(tmp_path):
    captured: list[str] = []
    sf = tmp_path / "settings.json"
    r = CCSwitchReconciler(
        proxy_url=PROXY,
        default_upstream=DEFAULT,
        set_upstream=captured.append,
        path=sf,
    )
    return r, sf, captured


def _write(sf, obj):
    sf.write_text(json.dumps(obj))
    os.utime(sf, None)


def test_third_party_captured_and_base_url_rewritten(tmp_path):
    r, sf, captured = _make(tmp_path)
    _write(
        sf,
        {
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
                "ANTHROPIC_AUTH_TOKEN": "sk-x",
                "ANTHROPIC_MODEL": "deepseek",
            }
        },
    )
    assert r.tick() is True
    env = json.loads(sf.read_text())["env"]
    # base_url repointed to Headroom; token + model preserved verbatim.
    assert env["ANTHROPIC_BASE_URL"] == PROXY
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-x"
    assert env["ANTHROPIC_MODEL"] == "deepseek"
    # Real endpoint captured as the upstream.
    assert captured[-1] == "https://api.deepseek.com/anthropic"
    assert r.current_upstream == "https://api.deepseek.com/anthropic"


def test_no_rewrite_loop(tmp_path):
    r, sf, _ = _make(tmp_path)
    _write(sf, {"env": {"ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic"}})
    assert r.tick() is True
    # Already pointing at Headroom now -> must be a no-op (no infinite loop).
    assert r.tick() is False


def test_switching_provider_recaptures(tmp_path):
    r, sf, captured = _make(tmp_path)
    _write(
        sf,
        {
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
                "ANTHROPIC_AUTH_TOKEN": "sk-d",
            }
        },
    )
    assert r.tick() is True
    _write(
        sf,
        {
            "env": {
                "ANTHROPIC_BASE_URL": "https://api.kimi.com/anthropic",
                "ANTHROPIC_AUTH_TOKEN": "sk-k",
            }
        },
    )
    assert r.tick() is True
    assert captured[-1] == "https://api.kimi.com/anthropic"
    assert json.loads(sf.read_text())["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-k"


def test_same_float_mtime_provider_switch_recaptures(tmp_path):
    r, sf, captured = _make(tmp_path)
    base_ns = 1_700_000_000_000_000_000
    _write(sf, {"env": {"ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic"}})
    os.utime(sf, ns=(base_ns, base_ns))
    assert r.tick() is True

    _write(sf, {"env": {"ANTHROPIC_BASE_URL": "https://api.kimi.com/anthropic"}})
    os.utime(sf, ns=(base_ns + 1, base_ns + 1))
    assert sf.stat().st_mtime == float(base_ns / 1_000_000_000)
    assert r.tick() is True
    assert captured[-1] == "https://api.kimi.com/anthropic"


def test_official_left_direct_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("HEADROOM_CC_SWITCH_ROUTE_OFFICIAL", raising=False)
    r, sf, _ = _make(tmp_path)
    _write(sf, {"env": {}})
    # Empty env = "Claude Official" (OAuth). Default: leave it direct.
    assert r.tick() is False
    assert json.loads(sf.read_text())["env"] == {}


def test_official_routed_when_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv("HEADROOM_CC_SWITCH_ROUTE_OFFICIAL", "1")
    r, sf, captured = _make(tmp_path)
    _write(sf, {"env": {}})
    assert r.tick() is True
    assert json.loads(sf.read_text())["env"]["ANTHROPIC_BASE_URL"] == PROXY
    assert captured[-1] == DEFAULT


def test_missing_file_is_noop(tmp_path):
    r, _, _ = _make(tmp_path)  # path does not exist yet
    assert r.tick() is False


def test_non_string_base_url_does_not_crash(tmp_path):
    r, sf, captured = _make(tmp_path)
    # A hand-edited / malformed file with a non-string base_url must not raise
    # (would otherwise blow up on .rstrip() and spam the watcher loop).
    _write(sf, {"env": {"ANTHROPIC_BASE_URL": 1234}})
    assert r.tick() is False  # treated as empty -> left direct
    assert captured == []


def test_transient_invalid_json_retries_next_tick(tmp_path):
    r, sf, captured = _make(tmp_path)
    # Mid-write garbage: read/parse fails, mtime must NOT be consumed.
    sf.write_text("{not valid json")
    os.utime(sf, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    assert r.tick() is False
    assert r._last_mtime_ns is None  # broken state not marked processed

    # File repaired at the SAME mtime: next tick must still process it.
    sf.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic"}}))
    os.utime(sf, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    assert r.tick() is True
    assert captured[-1] == "https://api.deepseek.com/anthropic"


@pytest.mark.parametrize(
    "val,expected",
    [("1", True), ("true", True), ("on", True), ("0", False), ("", False)],
)
def test_enabled_flag(monkeypatch, val, expected):
    from headroom.proxy.cc_switch_reconciler import reconciler_enabled

    monkeypatch.setenv("HEADROOM_CC_SWITCH_RECONCILE", val)
    assert reconciler_enabled() is expected
