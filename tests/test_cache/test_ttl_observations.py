"""Learned TTL files must remain a best-effort input to live requests."""

import json
import os
from types import SimpleNamespace

import pytest

from headroom.cache import ttl_observations
from headroom.cache.prefix_tracker import PrefixCacheTracker
from headroom.transforms.cold_prefix import is_cold_prefix


@pytest.mark.parametrize("layout", ["filename", "nested", "absolute"])
def test_observation_path_records_each_turn(tmp_path, monkeypatch, layout):
    monkeypatch.chdir(tmp_path)
    path = {
        "filename": "observations.jsonl",
        "nested": "nested/observations.jsonl",
        "absolute": str(tmp_path / "absolute" / "observations.jsonl"),
    }[layout]
    monkeypatch.setenv("HEADROOM_CACHE_TTL_LEARN", "1")
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    monkeypatch.setenv("HEADROOM_CACHE_TTL_OBS_PATH", path)
    for miss in (False, True):
        attribution = SimpleNamespace(reason="ttl_expiry" if miss else "hit", is_miss=miss)
        ttl_observations.record_cache_observation(
            provider="openai", model="fixture", attribution=attribution
        )

    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert [row["is_miss"] for row in rows] == [False, True]
    assert all(row["model"] == "fixture" for row in rows)


@pytest.mark.parametrize("stateless", [False, True])
def test_relative_observation_path_respects_write_gates(tmp_path, monkeypatch, stateless):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEADROOM_CACHE_TTL_OBS_PATH", "observations.jsonl")
    monkeypatch.setenv("HEADROOM_CACHE_TTL_LEARN", "1" if stateless else "0")
    monkeypatch.setenv("HEADROOM_STATELESS", "1" if stateless else "0")
    ttl_observations.record_cache_observation(
        provider="openai", model="fixture", attribution=SimpleNamespace()
    )
    assert not (tmp_path / "observations.jsonl").exists()


def test_observation_path_failure_still_does_not_raise(tmp_path, monkeypatch):
    parent = tmp_path / "not-a-directory"
    parent.write_bytes(b"keep")
    monkeypatch.setenv("HEADROOM_CACHE_TTL_OBS_PATH", str(parent / "observations.jsonl"))
    monkeypatch.setenv("HEADROOM_CACHE_TTL_LEARN", "1")
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    ttl_observations.record_cache_observation(
        provider="openai", model="fixture", attribution=SimpleNamespace()
    )
    assert parent.read_bytes() == b"keep"


@pytest.mark.windows_newline
def test_observation_writer_pins_lf(tmp_path, monkeypatch):
    import builtins

    path = tmp_path / "observations.jsonl"
    monkeypatch.setenv("HEADROOM_CACHE_TTL_OBS_PATH", str(path))
    monkeypatch.setenv("HEADROOM_CACHE_TTL_LEARN", "1")
    monkeypatch.delenv("HEADROOM_STATELESS", raising=False)
    real_open = builtins.open
    writes = []

    def recording_open(*args, **kwargs):
        if args[0] == str(path) and len(args) > 1 and args[1] == "a":
            writes.append(kwargs.get("newline"))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    ttl_observations.record_cache_observation(
        provider="openai", model="fixture", attribution=SimpleNamespace()
    )
    assert writes == ["\n"]
    assert path.read_bytes().endswith(b"\n")
    assert b"\r" not in path.read_bytes()


@pytest.fixture
def learned_file(tmp_path, monkeypatch):
    path = tmp_path / "learned.json"
    monkeypatch.setenv("HEADROOM_CACHE_TTL_LEARNED_PATH", str(path))
    monkeypatch.setattr(ttl_observations, "_learned_cache", None)
    monkeypatch.setattr(ttl_observations, "_learned_mtime", -1.0)
    monkeypatch.setattr(ttl_observations, "_learned_cache_path", None, raising=False)
    return path


@pytest.mark.parametrize(
    "second_table, expected", [({"openai": 1800}, 1800), ({}, None), ([300], None)]
)
def test_learned_cache_follows_configured_path_with_equal_mtimes(
    learned_file, monkeypatch, second_table, expected
):
    learned_file.write_text(json.dumps({"openai": 300}), encoding="utf-8")
    second = learned_file.with_name("other.json")
    second.write_text(json.dumps(second_table), encoding="utf-8")
    for path in (learned_file, second):
        os.utime(path, ns=(1_700_000_000_000_000_000,) * 2)
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") == 300

    monkeypatch.setenv("HEADROOM_CACHE_TTL_LEARNED_PATH", str(second))
    ttl = ttl_observations.resolve_learned_ttl("openai", "fixture")
    assert ttl == expected
    if expected is not None:
        tracker = PrefixCacheTracker("openai")
        tracker._idle_seconds_at_fetch = 600
        assert not is_cold_prefix(tracker, ttl_seconds=ttl)

    monkeypatch.setenv("HEADROOM_CACHE_TTL_LEARNED_PATH", str(learned_file))
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") == 300


def test_relative_learned_path_tracks_its_current_directory(learned_file, tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    for directory, ttl in ((first, 300), (second, 1800)):
        directory.mkdir()
        path = directory / "learned.json"
        path.write_text(json.dumps({"openai": ttl}), encoding="utf-8")
        os.utime(path, ns=(1_700_000_000_000_000_000,) * 2)
    monkeypatch.setenv("HEADROOM_CACHE_TTL_LEARNED_PATH", "learned.json")
    monkeypatch.chdir(first)
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") == 300
    monkeypatch.chdir(second)
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") == 1800


def test_unchanged_learned_source_is_read_only_once(learned_file, monkeypatch):
    import builtins

    learned_file.write_text(json.dumps({"openai": 900}), encoding="utf-8")
    real_open = builtins.open
    reads = []

    def recording_open(*args, **kwargs):
        if os.fspath(args[0]) == str(learned_file):
            reads.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") == 900
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") == 900
    assert len(reads) == 1


def test_missing_switched_source_stays_best_effort(learned_file, monkeypatch):
    learned_file.write_text(json.dumps({"openai": 900}), encoding="utf-8")
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") == 900
    monkeypatch.setenv(
        "HEADROOM_CACHE_TTL_LEARNED_PATH", str(learned_file.with_name("missing.json"))
    )
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") is None
    monkeypatch.setenv("HEADROOM_CACHE_TTL_LEARNED_PATH", str(learned_file))
    assert ttl_observations.resolve_learned_ttl("openai", "fixture") == 900


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
