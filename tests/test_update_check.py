"""Tests for headroom.update_check (PyPI probe, cache, banner notice)."""

from __future__ import annotations

import json
import time

import pytest

from headroom import update_check as uc


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    """Point the workspace (cache) dir at a tmp dir and enable the check."""
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("HEADROOM_UPDATE_CHECK", "on")
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.delenv("CI", raising=False)
    # Treat tests as a non-checkout, non-docker install by default.
    monkeypatch.setattr(uc, "_is_source_checkout", lambda: False)
    monkeypatch.setattr(uc, "_in_docker", lambda: False)
    return tmp_path


# --------------------------------------------------------------------------- #
# enable gate
# --------------------------------------------------------------------------- #
def test_enabled_by_default(monkeypatch):
    assert uc.is_update_check_enabled() is True


@pytest.mark.parametrize("val", ["off", "false", "0", "no", "disabled"])
def test_disabled_by_env(monkeypatch, val):
    monkeypatch.setenv("HEADROOM_UPDATE_CHECK", val)
    assert uc.is_update_check_enabled() is False


def test_disabled_in_stateless(monkeypatch):
    monkeypatch.setenv("HEADROOM_STATELESS", "1")
    assert uc.is_update_check_enabled() is False


def test_disabled_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    assert uc.is_update_check_enabled() is False


# --------------------------------------------------------------------------- #
# _select_latest
# --------------------------------------------------------------------------- #
def test_select_latest_picks_max_stable():
    data = {"releases": {"0.25.0": [{}], "0.26.0": [{}], "0.27.0rc1": [{}]}}
    assert uc._select_latest(data, allow_pre=False) == "0.26.0"


def test_select_latest_allows_pre():
    data = {"releases": {"0.26.0": [{}], "0.27.0rc1": [{}]}}
    assert uc._select_latest(data, allow_pre=True) == "0.27.0rc1"


def test_select_latest_skips_fully_yanked():
    data = {
        "releases": {
            "0.26.0": [{"yanked": False}],
            "0.27.0": [{"yanked": True}],
        }
    }
    assert uc._select_latest(data, allow_pre=False) == "0.26.0"


def test_select_latest_falls_back_to_info_version():
    data = {"releases": {}, "info": {"version": "0.26.0"}}
    assert uc._select_latest(data, allow_pre=False) == "0.26.0"


# --------------------------------------------------------------------------- #
# fetch_latest_version
# --------------------------------------------------------------------------- #
def test_fetch_latest_version_parses(monkeypatch):
    payload = json.dumps({"releases": {"0.26.0": [{}], "0.27.0": [{}]}}).encode()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return payload

    monkeypatch.setattr(uc.urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert uc.fetch_latest_version() == "0.27.0"


def test_fetch_latest_version_network_error_returns_none(monkeypatch):
    def _boom(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(uc.urllib.request, "urlopen", _boom)
    assert uc.fetch_latest_version() is None


# --------------------------------------------------------------------------- #
# cache + should_check
# --------------------------------------------------------------------------- #
def test_cache_roundtrip():
    uc.write_cache("0.27.0", now=1000.0)
    cache = uc.read_cache()
    assert cache["latest_version"] == "0.27.0"
    assert cache["last_check"] == 1000.0


def test_should_check_true_when_no_cache():
    assert uc.should_check() is True


def test_should_check_false_when_fresh():
    now = time.time()
    uc.write_cache("0.27.0", now=now)
    assert uc.should_check(now=now + 10) is False


def test_should_check_true_when_stale():
    now = time.time()
    uc.write_cache("0.27.0", now=now)
    assert uc.should_check(now=now + uc._CHECK_TTL_SECONDS + 1) is True


# --------------------------------------------------------------------------- #
# format_update_notice
# --------------------------------------------------------------------------- #
def test_notice_when_newer(monkeypatch):
    uc.write_cache("0.27.0")
    monkeypatch.setattr(uc, "installed_version", lambda: "0.26.0")
    notice = uc.format_update_notice()
    assert notice and "0.27.0" in notice and "headroom update" in notice


def test_no_notice_when_current(monkeypatch):
    uc.write_cache("0.26.0")
    monkeypatch.setattr(uc, "installed_version", lambda: "0.26.0")
    assert uc.format_update_notice() is None


def test_no_notice_in_checkout(monkeypatch):
    uc.write_cache("0.27.0")
    monkeypatch.setattr(uc, "installed_version", lambda: "0.26.0")
    monkeypatch.setattr(uc, "_is_source_checkout", lambda: True)
    assert uc.format_update_notice() is None


def test_no_notice_when_disabled(monkeypatch):
    uc.write_cache("0.27.0")
    monkeypatch.setattr(uc, "installed_version", lambda: "0.26.0")
    monkeypatch.setenv("HEADROOM_UPDATE_CHECK", "off")
    assert uc.format_update_notice() is None


def test_no_notice_when_version_unknown(monkeypatch):
    uc.write_cache("0.27.0")
    monkeypatch.setattr(uc, "installed_version", lambda: None)
    assert uc.format_update_notice() is None


# --------------------------------------------------------------------------- #
# maybe_check_async
# --------------------------------------------------------------------------- #
def test_maybe_check_async_writes_cache(monkeypatch):
    monkeypatch.setattr(uc, "fetch_latest_version", lambda **k: "0.27.0")
    thread = uc.maybe_check_async()
    assert thread is not None
    thread.join(timeout=5)
    assert uc.read_cache()["latest_version"] == "0.27.0"


def test_maybe_check_async_skips_in_checkout(monkeypatch):
    monkeypatch.setattr(uc, "_is_source_checkout", lambda: True)
    assert uc.maybe_check_async() is None


def test_maybe_check_async_skips_when_fresh(monkeypatch):
    uc.write_cache("0.27.0")
    monkeypatch.setattr(uc, "fetch_latest_version", lambda **k: pytest.fail("should not fetch"))
    assert uc.maybe_check_async() is None
