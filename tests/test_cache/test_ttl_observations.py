"""Learned TTL files must remain a best-effort input to live requests."""

import json
import os

import pytest

from headroom.cache import ttl_observations
from headroom.cache.prefix_tracker import PrefixCacheTracker
from headroom.transforms.cold_prefix import is_cold_prefix


@pytest.fixture
def learned_file(tmp_path, monkeypatch):
    path = tmp_path / "learned.json"
    monkeypatch.setenv("HEADROOM_CACHE_TTL_LEARNED_PATH", str(path))
    monkeypatch.setattr(ttl_observations, "_learned_cache", None)
    monkeypatch.setattr(ttl_observations, "_learned_mtime", -1.0)
    return path


@pytest.mark.parametrize("table", [[300], "not a table", 300, True])
def test_non_object_table_is_ignored_even_after_caching(learned_file, table):
    learned_file.write_text(json.dumps(table))
    for _ in range(2):
        assert ttl_observations.resolve_learned_ttl("openai", "fixture") is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, -30, 0])
@pytest.mark.parametrize("nested", [False, True])
def test_invalid_model_ttl_uses_valid_provider_fallback(learned_file, value, nested):
    learned_file.write_text(
        json.dumps(
            {
                "openai/fixture": {"ttl_seconds": value} if nested else value,
                "openai": {"ttl_seconds": 900},
            }
        )
    )
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") == 900


@pytest.mark.parametrize("value", [600, 600.75])
@pytest.mark.parametrize("nested", [False, True])
def test_valid_numeric_ttls_retain_integer_conversion(learned_file, value, nested):
    learned_file.write_text(json.dumps({"openai": {"ttl_seconds": value} if nested else value}))
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") == 600


@pytest.mark.parametrize("value", [True, -30])
def test_invalid_provider_ttl_does_not_make_a_warm_prefix_cold(learned_file, value):
    learned_file.write_text(json.dumps({"openai": value}))
    tracker = PrefixCacheTracker("openai")
    tracker._idle_seconds_at_fetch = 120.0
    ttl = ttl_observations.resolve_learned_ttl("openai", "fixture")
    assert ttl is None
    assert not is_cold_prefix(tracker, ttl_seconds=ttl)


def test_corrected_file_is_reloaded_and_model_still_takes_precedence(learned_file):
    learned_file.write_text('["wrong shape"]')
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") is None
    old_mtime = learned_file.stat().st_mtime
    learned_file.write_text(json.dumps({"openai/fixture": {"ttl_seconds": 600}, "openai": 900}))
    os.utime(learned_file, (old_mtime + 2, old_mtime + 2))
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") == 600
    assert ttl_observations.resolve_learned_ttl("openai", "other") == 900
