"""Regression tests for TOIN privacy and bounded retention."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.proxy.tenant_key import _sanitize_tenant_key
from headroom.telemetry.backends.filesystem import (
    FileSystemTOINBackend,
    _volume_is_case_insensitive,
)
from headroom.telemetry.models import ToolSignature
from headroom.telemetry.toin import TOINConfig, ToolIntelligenceNetwork


def test_failed_auto_saves_respect_interval_and_retry_pending_data(tmp_path, caplog, monkeypatch):
    import headroom.telemetry.toin as toin_module

    clock = [10000.0]
    monkeypatch.setattr(toin_module, "time", SimpleNamespace(time=lambda: clock[0]))
    parent = tmp_path / "state"
    parent.write_text("blocked", encoding="utf-8", newline="\n")
    path = parent / "toin.json"
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=60))
    clock[0] += 60
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            toin.record_compression(
                tool_signature=_signature(456),
                original_count=10,
                compressed_count=5,
                original_tokens=100,
                compressed_tokens=50,
                strategy="test",
            )
    failures = [
        record for record in caplog.records if getattr(record, "event", None) == "toin_save_failed"
    ]
    assert len(failures) == 1
    assert toin._dirty is True

    parent.unlink()
    clock[0] += 60
    toin._maybe_auto_save()
    restored = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path)))
    assert restored.get_pattern(_signature(456).structure_hash).total_compressions == 4
    assert toin._dirty is False


def test_manual_save_can_retry_before_auto_save_interval(tmp_path):
    parent = tmp_path / "state"
    parent.write_text("blocked", encoding="utf-8", newline="\n")
    path = parent / "toin.json"
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=3600))
    toin.record_compression(
        tool_signature=_signature(789),
        original_count=10,
        compressed_count=5,
        original_tokens=100,
        compressed_tokens=50,
        strategy="test",
    )
    toin._last_save_time = 0
    toin._maybe_auto_save()
    assert toin._dirty is True
    parent.unlink()

    toin.save()

    assert _current_store_path(path).is_file()
    assert toin._dirty is False


def test_filesystem_save_reports_parent_directory_failure(tmp_path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8", newline="\n")
    backend = FileSystemTOINBackend(str(blocked / "toin.json"))

    with pytest.raises(OSError):
        backend.save({"patterns": {}})


def test_failed_replace_preserves_store_and_cleans_temporary_file(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    path = tmp_path / "toin.json"
    backend = FileSystemTOINBackend(str(path))
    backend.save({"patterns": {"original": {}}})
    original = path.read_bytes()

    def fail_replace(self, target):
        raise OSError("replace unavailable")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace unavailable"):
        backend.save({"patterns": {"new": {}}})

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".toin_*.tmp"))


def test_filesystem_failure_keeps_toin_dirty_for_automatic_retry(tmp_path, caplog) -> None:
    parent = tmp_path / "state"
    path = parent / "toin.json"
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=3600))
    toin.record_compression(
        tool_signature=_signature(123),
        original_count=10,
        compressed_count=5,
        original_tokens=100,
        compressed_tokens=50,
        strategy="test",
    )
    toin._last_save_time = 0
    parent.write_text("temporarily blocked", encoding="utf-8", newline="\n")

    with caplog.at_level(logging.WARNING):
        toin.save()

    assert toin._dirty is True
    assert toin._last_save_time == 0
    assert any(getattr(record, "event", None) == "toin_save_failed" for record in caplog.records)

    parent.unlink()
    toin._maybe_auto_save()

    assert toin._dirty is False
    assert toin._last_save_time > 0
    restored = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path)))
    pattern = restored.get_pattern(_signature(123).structure_hash)
    assert pattern is not None
    assert pattern.total_compressions == 1


def _signature(index: int) -> ToolSignature:
    return ToolSignature.from_items([{f"field_{index}": "x"}])


def test_free_form_queries_are_not_persisted() -> None:
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=""))

    assert toin._anonymize_query_pattern("show me the contents of /private/secret.txt") is None
    assert toin._anonymize_query_pattern("status:error AND user:john") == (
        f"{toin._hash_field_name('status')}:* {toin._hash_field_name('user')}:*"
    )

    toin.record_retrieval(
        "signature",
        retrieval_type="search",
        query="show me the contents of /private/secret.txt",
    )
    pattern = toin.get_pattern("signature")
    assert pattern is not None
    assert pattern.query_pattern_frequency == {}


def test_query_pattern_length_is_bounded() -> None:
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=""))

    assert toin._anonymize_query_pattern("field:" + "x" * 1000) == (
        f"{toin._hash_field_name('field')}:*"
    )
    assert toin._anonymize_query_pattern(" ".join(f"field{i}:x" for i in range(200))) is None


def test_pattern_table_evicts_old_low_sample_patterns() -> None:
    config = TOINConfig(storage_path="", max_patterns=2)
    toin = ToolIntelligenceNetwork(config)

    for index in range(3):
        toin.record_compression(
            tool_signature=_signature(index),
            original_count=10,
            compressed_count=5,
            original_tokens=100,
            compressed_tokens=50,
            strategy="test",
        )

    assert len(toin._patterns) == 2
    assert _signature(0).structure_hash not in {
        pattern.tool_signature_hash for pattern in toin._patterns.values()
    }


def test_legacy_raw_query_keys_are_removed_on_load(tmp_path) -> None:
    path = tmp_path / "toin.json"
    payload = {
        "version": "2.0",
        "patterns": {
            "unknown|unknown|abc": {
                "tool_signature_hash": "abc",
                "query_pattern_frequency": {
                    "raw prompt containing secret source code": 4,
                    "status:* user:*": 3,
                    "1a2b3c4d:* 5e6f7a8b:*": 2,
                },
                "common_query_patterns": [
                    "raw prompt containing secret source code",
                    "status:* user:*",
                    "1a2b3c4d:* 5e6f7a8b:*",
                ],
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path)))
    pattern = next(iter(toin._patterns.values()))

    assert pattern.query_pattern_frequency == {"1a2b3c4d:* 5e6f7a8b:*": 2}
    assert pattern.common_query_patterns == ["1a2b3c4d:* 5e6f7a8b:*"]
    # Rewritten at load, not on the next observation.
    on_disk = path.read_text(encoding="utf-8")
    assert "raw prompt containing secret source code" not in on_disk
    assert "status:* user:*" not in on_disk


def test_legacy_rewrite_failure_keeps_sanitized_import(tmp_path, monkeypatch, caplog) -> None:
    """Sanitized patterns import even when the rollback file cannot be rewritten.

    When the pre-upgrade file is read-only (or otherwise unwritable), the
    sanitized import must still land in memory and be persisted to the
    versioned sibling. A warning is surfaced so an operator can redact the
    rollback copy externally; the failure never strands the legacy history
    on the next load.
    """
    path = tmp_path / "toin.json"
    payload = {
        "version": "2.0",
        "patterns": {
            "unknown|unknown|abc": {
                "tool_signature_hash": "abc",
                "query_pattern_frequency": {
                    "raw prompt containing secret source code": 4,
                    "1a2b3c4d:* 5e6f7a8b:*": 2,
                },
                "common_query_patterns": [
                    "raw prompt containing secret source code",
                    "1a2b3c4d:* 5e6f7a8b:*",
                ],
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")

    original_save = FileSystemTOINBackend.save

    def fail_rollback_save(self, data):
        # Only the rollback file is unwritable; the v3 sibling must still save.
        if self._path == path:
            raise OSError("rollback mount is read-only")
        original_save(self, data)

    monkeypatch.setattr(FileSystemTOINBackend, "save", fail_rollback_save, raising=False)

    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=0))
    pattern = next(iter(toin._patterns.values()))
    assert pattern.query_pattern_frequency == {"1a2b3c4d:* 5e6f7a8b:*": 2}

    # Sanitized import survived; rollback file still holds the raw text.
    [warning] = [
        r for r in caplog.records if getattr(r, "event", None) == "toin_rollback_rewrite_failed"
    ]
    assert "raw prompt containing secret source code" in path.read_text(encoding="utf-8")
    # The versioned sibling was written with the sanitized data.
    assert _current_store_path(path).is_file()
    assert "raw prompt containing secret source code" not in _current_store_path(path).read_text(
        encoding="utf-8"
    )


def _record(toin: ToolIntelligenceNetwork, index: int) -> None:
    toin.record_compression(
        tool_signature=_signature(index),
        original_count=10,
        compressed_count=5,
        original_tokens=100,
        compressed_tokens=50,
        strategy="test",
    )


def _current_store_path(base: Path) -> Path:
    """Where this version writes: the version-suffixed sibling of ``base``."""
    return base.with_name(f"{base.stem}.v3{base.suffix}")


def _saved_store(path: Path, count: int) -> dict:
    """Write a store fixture at ``path`` and return its parsed content."""
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path="", auto_save_interval=0))
    for index in range(count):
        _record(toin, index)
    data = toin.export_patterns()
    path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    return data


def _skip_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == "toin_import_skipped"]


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda entry: {k: v for k, v in entry.items() if k != "tool_signature_hash"},
        lambda entry: {**entry, "field_semantics": {"field_1": "not a dict"}},
        lambda entry: {**entry, "query_pattern_frequency": "not a dict"},
        lambda entry: {**entry, "query_pattern_frequency": {"1a2b3c4d:*": "many"}},
        lambda entry: {**entry, "seen_instance_hashes": "bad"},
        lambda entry: {**entry, "commonly_retrieved_fields": "abc"},
        lambda entry: {**entry, "sample_size": "x"},
        lambda entry: "not a pattern",
    ],
    ids=[
        "missing_signature_hash",
        "bad_field_semantics",
        "bad_query_frequencies",
        "bad_query_count",
        "tracking_list_is_a_string",
        "list_field_is_a_string",
        "count_is_a_string",
        "not_a_dict",
    ],
)
def test_malformed_pattern_does_not_drop_the_rest_of_the_store(tmp_path, caplog, corrupt) -> None:
    path = tmp_path / "toin.json"
    data = _saved_store(path, 5)
    keys = list(data["patterns"])
    data["patterns"][keys[1]] = corrupt(data["patterns"][keys[1]])
    path.write_text(json.dumps(data), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="headroom.telemetry.toin"):
        toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=0))

    assert len(toin._patterns) == 4
    [warning] = _skip_warnings(caplog)
    assert warning.skipped == 1
    assert warning.imported == 4

    # The next save used to write only the patterns read before the bad one.
    _record(toin, 99)
    toin.save()
    saved = json.loads(_current_store_path(path).read_text(encoding="utf-8"))["patterns"]
    assert set(keys) - {keys[1]} <= set(saved)
    assert len(saved) == 5


def test_foreign_store_keeps_the_healthy_entries_after_a_malformed_one(tmp_path, caplog) -> None:
    path = tmp_path / "toin.json"
    data = _saved_store(path, 3)
    keys = list(data["patterns"])
    # A string tracking list used to pass from_dict and then fail on .append()
    # while tracking the foreign source, outside the per-entry guard.
    data["patterns"][keys[0]]["seen_instance_hashes"] = "bad"
    data["instance_id"] = "foreign-instance"
    path.write_text(json.dumps(data), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="headroom.telemetry.toin"):
        toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=0))

    assert len(toin._patterns) == 2
    [warning] = _skip_warnings(caplog)
    assert warning.skipped == 1
    assert all("foreign-instance" in p._seen_instance_hashes for p in toin._patterns.values())

    toin.save()
    saved = json.loads(_current_store_path(path).read_text(encoding="utf-8"))["patterns"]
    assert set(saved) == set(keys[1:])


def test_failed_merge_leaves_the_existing_pattern_unchanged(caplog) -> None:
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=""))
    for index in range(2):
        _record(toin, index)
    exported = toin.export_patterns()
    keys = list(exported["patterns"])
    # to_dict() shares the pattern's containers, so snapshot through JSON.
    before = {key: json.loads(json.dumps(p.to_dict())) for key, p in toin._patterns.items()}

    bad = dict(exported["patterns"][keys[0]])
    # Passes from_dict, then fails part-way through _merge_patterns: the counts
    # and the "field_new" key are already added when the "x" value raises.
    bad["field_retrieval_frequency"] = {"field_new": 1, "field_0": "x"}
    healthy = dict(exported["patterns"][keys[1]])
    with caplog.at_level(logging.WARNING, logger="headroom.telemetry.toin"):
        toin.import_patterns(
            {"instance_id": "foreign-instance", "patterns": {keys[0]: bad, keys[1]: healthy}}
        )

    [warning] = _skip_warnings(caplog)
    assert warning.skipped == 1
    first, second = (toin._patterns[key] for key in before)
    assert json.loads(json.dumps(first.to_dict())) == before[next(iter(before))]
    assert second.total_compressions == 2


def test_float_counts_written_by_merges_still_load(tmp_path, caplog) -> None:
    # _merge_patterns stores optimal_max_items as a weighted float.
    path = tmp_path / "toin.json"
    data = _saved_store(path, 2)
    for entry in data["patterns"].values():
        entry["optimal_max_items"] = 37.5
    path.write_text(json.dumps(data), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="headroom.telemetry.toin"):
        toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=0))

    assert len(toin._patterns) == 2
    assert _skip_warnings(caplog) == []


def test_valid_store_loads_without_a_skip_warning(tmp_path, caplog) -> None:
    path = tmp_path / "toin.json"
    _saved_store(path, 5)

    with caplog.at_level(logging.WARNING, logger="headroom.telemetry.toin"):
        toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=0))

    assert len(toin._patterns) == 5
    assert _skip_warnings(caplog) == []


def test_pre_upgrade_pool_that_collides_with_a_digested_key_is_dropped(tmp_path, caplog) -> None:
    """A v2 pool a legal id owned must not be handed to the tenant whose digest hits it.

    Before the digested form existed, ``tnt-acmecorp-80a1d5cc623b165f`` was a
    legitimate tenant id and owned that pool literally. ``acme.corp`` now
    digests to exactly that key, so importing the old pool would merge one
    tenant's learned patterns into another's.
    """
    path = tmp_path / "toin.json"
    data = _saved_store(path, 1)
    legacy_tenant = "tnt-acmecorp-80a1d5cc623b165f"
    [entry] = data["patterns"].values()
    [key] = data["patterns"]
    _, auth_mode, model_family, sig_hash = key.split("|")
    data["version"] = "2.0"
    data["patterns"] = {f"{legacy_tenant}|{auth_mode}|{model_family}|{sig_hash}": entry}
    path.write_text(json.dumps(data), encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="headroom.telemetry.toin"):
        toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=0))

    assert toin._patterns == {}
    [purge] = [
        r for r in caplog.records if getattr(r, "event", None) == "toin_tenant_key_upgrade_purge"
    ]
    assert purge.purged == 1

    # Only the purged pool is gone when the next save rewrites the file.
    toin.save()
    current = json.loads(_current_store_path(path).read_text(encoding="utf-8"))
    assert current["version"] == "3.0"
    assert current["patterns"] == {}
    # The pre-upgrade file is left as the rollback view.
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == "2.0"


def test_a_conforming_pool_survives_the_upgrade(tmp_path) -> None:
    """The purge is scoped to the digested sub-space, not to v2 stores."""
    path = tmp_path / "toin.json"
    data = _saved_store(path, 3)
    keys = list(data["patterns"])
    assert all(key.startswith("global|") for key in keys)
    data["version"] = "2.0"
    path.write_text(json.dumps(data), encoding="utf-8")

    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=0))

    assert len(toin._patterns) == 3


def test_rollback_reader_cannot_resolve_v3_tenant_keys(tmp_path) -> None:
    """A rolled-back release must not read v3 pools under its old resolver.

    ``acme.corp`` records a pattern under the derived key
    ``tnt-acmecorp-80a1d5cc623b165f``. The previous release maps a tenant that
    literally sends that key onto the same pool, so it must never open the v3
    store: v3 writes to the version-suffixed sibling and the unversioned path,
    which the old release reads, stays at the pre-upgrade layout.
    """
    base = tmp_path / "toin.json"
    derived_tenant = _sanitize_tenant_key("acme.corp")
    assert derived_tenant == "tnt-acmecorp-80a1d5cc623b165f"
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(base), auto_save_interval=0))
    toin.record_compression(
        tool_signature=_signature(0),
        original_count=10,
        compressed_count=5,
        original_tokens=100,
        compressed_tokens=50,
        strategy="test",
        tenant_key=derived_tenant,
    )
    toin.save()

    v3 = json.loads(_current_store_path(base).read_text(encoding="utf-8"))
    [stored_key] = v3["patterns"]
    assert stored_key.split("|")[0] == derived_tenant

    # What the previous release opens: the configured path, its own resolver.
    previous_release_store = FileSystemTOINBackend(str(base)).load()
    assert previous_release_store == {}


def test_a_foreign_v3_store_keeps_its_digested_pools(caplog) -> None:
    """A store already written at v3 wrote those keys deliberately."""
    digest_key = "tnt-acmecorp-80a1d5cc623b165f"
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path="", auto_save_interval=0))
    _record(toin, 0)
    data = json.loads(json.dumps(toin.export_patterns()))
    [entry] = data["patterns"].values()
    [key] = data["patterns"]
    _, auth_mode, model_family, sig_hash = key.split("|")
    data["patterns"] = {f"{digest_key}|{auth_mode}|{model_family}|{sig_hash}": entry}
    assert data["version"] == "3.0"

    with caplog.at_level(logging.INFO, logger="headroom.telemetry.toin"):
        fresh = ToolIntelligenceNetwork(TOINConfig(storage_path="", auto_save_interval=0))
        fresh.import_patterns(data)

    assert len(fresh._patterns) == 1
    assert [
        r for r in caplog.records if getattr(r, "event", None) == "toin_tenant_key_upgrade_purge"
    ] == []


def test_versioned_store_writes_a_sibling_over_the_pre_upgrade_file(tmp_path) -> None:
    """v3 lives in its own file so a rolled-back reader cannot open it."""
    base = tmp_path / "toin.json"
    pre_upgrade = {"version": "2.0", "patterns": {"a": {"value": 1}}}
    base.write_text(json.dumps(pre_upgrade), encoding="utf-8")

    backend = FileSystemTOINBackend(str(base), store_version="3.0")
    assert backend.load() == pre_upgrade
    assert backend.loaded_pre_upgrade is True

    backend.save({"version": "3.0", "patterns": {}})

    current = _current_store_path(base)
    assert json.loads(current.read_text(encoding="utf-8"))["version"] == "3.0"
    assert json.loads(base.read_text(encoding="utf-8")) == pre_upgrade

    # Once this layout's file exists it wins: the pre-upgrade file is only
    # the one-time upgrade source and the rollback view.
    reloaded = FileSystemTOINBackend(str(base), store_version="3.0")
    assert reloaded.load()["version"] == "3.0"
    assert reloaded.loaded_pre_upgrade is False


def test_configured_v3_path_stays_independent_of_legacy_store(tmp_path) -> None:
    """``storage_path='toin.v3.json'`` must not import another store's data.

    Opening ``toin.v3.json`` reads/writes it in place with no pre-upgrade
    fallback, so saving under a different configured path (which lands in
    its own ``<name>.v3.json``) cannot leak into the ``toin.v3.json`` store.
    """
    legacy = tmp_path / "toin_legacy.json"
    v3 = tmp_path / "toin.v3.json"

    saved = ToolIntelligenceNetwork(TOINConfig(storage_path=str(legacy), auto_save_interval=0))
    saved.record_compression(
        tool_signature=_signature(0),
        original_count=10,
        compressed_count=5,
        original_tokens=100,
        compressed_tokens=50,
        strategy="test",
        tenant_key="acme.corp",
    )
    saved.save()
    assert saved._patterns

    opened = ToolIntelligenceNetwork(TOINConfig(storage_path=str(v3), auto_save_interval=0))
    assert opened._backend._path == v3
    assert opened._backend._pre_upgrade_path is None
    assert opened._backend.loaded_pre_upgrade is False
    assert opened._patterns == {}
    assert not v3.exists()


def test_explicit_backend_writes_v3_sibling_keeps_legacy_rollback(tmp_path) -> None:
    """An explicit backend given ``toin.json`` writes ``toin.v3.json`` in place of legacy.

    The unversioned file stays untouched as the rollback view; a previous
    release opening the legacy path reads its own layout, not v3 data.
    """
    base = tmp_path / "toin.json"
    base.write_text(json.dumps({"version": "2.0", "patterns": {}}), encoding="utf-8")
    v3 = tmp_path / "toin.v3.json"

    backend = FileSystemTOINBackend(str(base), store_version="3.0")
    assert backend._path == v3
    assert backend._pre_upgrade_path == base

    toin = ToolIntelligenceNetwork(
        TOINConfig(storage_path="", auto_save_interval=0), backend=backend
    )
    toin.record_compression(
        tool_signature=_signature(1),
        original_count=10,
        compressed_count=5,
        original_tokens=100,
        compressed_tokens=50,
        strategy="test",
        tenant_key="acme.corp",
    )
    toin.save()

    assert v3.is_file()
    assert json.loads(base.read_text(encoding="utf-8"))["version"] == "2.0"
    # Legacy reader opens its own layout, not the v3 store.
    assert FileSystemTOINBackend(str(base)).load()["version"] == "2.0"


def test_explicit_backend_without_store_version_writes_unversioned(tmp_path) -> None:
    """An explicit backend on ``toin.v3.json`` with no store_version writes in place.

    It must not double the suffix (``toin.v3.v3.json``) nor treat ``toin.json``
    as a pre-upgrade source: the names ``toin.json`` and ``toin.v3.json`` are
    independent stores.
    """
    v3 = tmp_path / "toin.v3.json"
    backend = FileSystemTOINBackend(str(v3))
    assert backend._path == v3
    assert backend._pre_upgrade_path is None

    toin = ToolIntelligenceNetwork(
        TOINConfig(storage_path="", auto_save_interval=0), backend=backend
    )
    toin.record_compression(
        tool_signature=_signature(2),
        original_count=10,
        compressed_count=5,
        original_tokens=100,
        compressed_tokens=50,
        strategy="test",
        tenant_key="acme.corp",
    )
    toin.save()

    assert v3.is_file()
    assert not (tmp_path / "toin.v3.v3.json").exists()
    assert not (tmp_path / "toin.json").exists()


def test_configured_path_cannot_clobber_another_instance_stores(tmp_path, caplog) -> None:
    """Two proxies deriving to the same write target do not overwrite each other.

    Proxy B is configured at ``toin.v3.json`` and writes it in place; proxy A
    is configured at ``toin.json`` and derives the same file. A must refuse
    rather than replace B's history, and B's store must survive untouched.
    """
    v3 = tmp_path / "toin.v3.json"
    legacy = tmp_path / "toin.json"

    backend_b = FileSystemTOINBackend(str(v3))
    toin_b = ToolIntelligenceNetwork(
        TOINConfig(storage_path="", auto_save_interval=0), backend=backend_b
    )
    for index in range(3):
        _record(toin_b, index)
    toin_b.save()
    b_before = v3.read_text(encoding="utf-8")

    # A must not import B's versioned store, even when legacy data exists.
    legacy.write_text(json.dumps(_saved_store(legacy, 2)), encoding="utf-8", newline="\n")

    with caplog.at_level(logging.WARNING, logger="headroom.telemetry.toin"):
        toin_a = ToolIntelligenceNetwork(TOINConfig(storage_path=str(legacy), auto_save_interval=0))
    assert not toin_a._patterns  # Foreign history must never be imported.
    toin_a.save()

    assert v3.read_text(encoding="utf-8") == b_before
    [failure] = [r for r in caplog.records if getattr(r, "event", None) == "toin_save_failed"]
    assert "owned by a differently configured path" in failure.error


@pytest.mark.parametrize(
    ("configured", "expected_write", "expected_pre_upgrade"),
    [
        ("toin.json", "toin.v3.json", "toin.json"),
        ("toin.v3.json", "toin.v3.json", None),
        ("toin.v3", "toin.v3", None),
        ("toin", "toin.v3", "toin"),
    ],
)
def test_version_paths_never_double_the_version_marker(
    tmp_path, configured, expected_write, expected_pre_upgrade
) -> None:
    """Only an unversioned path derives the versioned sibling.

    An extensionless ``toin.v3`` must be read as already versioned: without
    the extension-aware check, ``path.suffix`` is ``.v3`` and the derived
    sibling becomes ``toin.v3.v3``, leaving the configured file stale.
    """
    source = tmp_path / configured
    backend = FileSystemTOINBackend(str(source), store_version="3.0")

    assert backend._path == tmp_path / expected_write
    assert backend._pre_upgrade_path == (
        tmp_path / expected_pre_upgrade if expected_pre_upgrade else None
    )


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("name", ["toin.v3", "toin.v3.json"])
def test_version_like_legacy_path_preserves_rollback(tmp_path, explicit, name) -> None:
    path = tmp_path / name
    data = _saved_store(path, 2)
    data["version"] = "2.0"
    path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    before = path.read_bytes()
    backend = FileSystemTOINBackend(str(path)) if explicit else None
    toin = ToolIntelligenceNetwork(
        TOINConfig(storage_path=str(path), auto_save_interval=0), backend=backend
    )
    assert len(toin._patterns) == 2
    toin.save()
    assert path.read_bytes() == before
    assert toin._backend.path != path
    assert (
        len(
            ToolIntelligenceNetwork(
                TOINConfig(storage_path=str(path), auto_save_interval=0),
                backend=FileSystemTOINBackend(str(path)) if explicit else None,
            )._patterns
        )
        == 2
    )


def test_owner_accepts_equivalent_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    first = ToolIntelligenceNetwork(TOINConfig(storage_path="toin.json", auto_save_interval=0))
    _record(first, 1)
    first.save()
    second = ToolIntelligenceNetwork(
        TOINConfig(storage_path=str(tmp_path / "toin.json"), auto_save_interval=0)
    )
    assert len(second._patterns) == 1
    _record(second, 2)
    second.save()
    assert len(json.loads(second._backend.path.read_text(encoding="utf-8"))["patterns"]) == 2


def test_archiving_the_rollback_copy_keeps_the_migrated_store(tmp_path) -> None:
    """The write-target decision must survive the rollback copy being archived.

    A configured name that already looks versioned (``toin.v3.json``) holding
    legacy data migrates to a ``.v3`` sibling. If the decision depended on the
    configured file still existing, archiving that rollback copy would make
    the next start open a fresh store at the configured name and orphan the
    learned history in the sibling.
    """
    path = tmp_path / "toin.v3.json"
    data = _saved_store(path, 2)
    data["version"] = "2.0"
    path.write_text(json.dumps(data), encoding="utf-8", newline="\n")

    first = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=0))
    assert len(first._patterns) == 2
    first.save()
    migrated = first._backend.path
    assert migrated != path

    # Operator archives the rollback copy.
    archived = tmp_path / "archive" / path.name
    archived.parent.mkdir()
    path.rename(archived)

    second = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=0))
    assert second._backend.path == migrated
    assert len(second._patterns) == 2
    _record(second, 99)
    second.save()
    assert len(json.loads(migrated.read_text(encoding="utf-8"))["patterns"]) == 3
    # The configured name stays the rollback view: not rewritten, not recreated.
    assert not path.exists()


def test_version_probe_respects_the_size_limit(tmp_path, monkeypatch) -> None:
    """Startup must not parse an oversized version-named store to probe it."""
    import headroom.telemetry.backends.filesystem as filesystem

    path = tmp_path / "toin.v3.json"
    path.write_text(json.dumps({"version": "2.0", "patterns": {}}), encoding="utf-8", newline="\n")
    path.write_bytes(path.read_bytes() + b" " * 10_000)
    original = path.read_bytes()

    parsed: list[str] = []
    real_load = filesystem.json.load

    def counting_load(fp, *args, **kwargs):
        parsed.append(str(getattr(fp, "name", fp)))
        return real_load(fp, *args, **kwargs)

    monkeypatch.setattr(filesystem.json, "load", counting_load)
    backend = FileSystemTOINBackend(str(path), max_load_bytes=100, store_version="3.0")

    # The oversized candidate was never parsed; writes are redirected to the
    # ownership-checked sibling and the oversized file is preserved untouched
    # as the rollback view.
    assert backend.path != path
    assert backend._pre_upgrade_path == path
    assert backend.load() == {}
    assert path.read_bytes() == original
    assert not any(name.endswith("toin.v3.json") for name in parsed)


def test_oversized_legacy_store_preserves_old_data_and_persists_new_learning(
    tmp_path,
) -> None:
    """An oversized legacy store must not block persistence of fresh records.

    The oversized file is preserved byte-for-byte, new learning is saved to
    the ownership-checked sibling, and a restart with the same configuration
    rediscovers that sibling instead of losing the records.
    """
    path = tmp_path / "toin.v3.json"
    data = _saved_store(path, 2)
    data["version"] = "2.0"
    path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    path.write_bytes(path.read_bytes() + b" " * 20_000)
    original = path.read_bytes()

    # The limit fits a fresh record store but not the padded legacy file.
    toin = ToolIntelligenceNetwork(
        TOINConfig(storage_path=str(path), max_storage_bytes=4000, auto_save_interval=0)
    )
    assert toin._patterns == {}  # oversized store not imported
    _record(toin, 50)
    toin.save()

    sibling = toin._backend.path
    assert sibling != path
    assert path.read_bytes() == original
    restarted = ToolIntelligenceNetwork(
        TOINConfig(storage_path=str(path), max_storage_bytes=4000, auto_save_interval=0)
    )
    assert restarted._backend.path == sibling
    assert _signature(50).structure_hash in {
        pattern.tool_signature_hash for pattern in restarted._patterns.values()
    }


def test_case_alias_spelling_shares_one_store(tmp_path) -> None:
    """On a case-insensitive volume, a case-alias spelling must not fork the store.

    The owner identity is case-normalized there, so the version check has to
    be as well: ``TOIN.V3.JSON`` must be recognized as already versioned and
    resolve to the same owner as ``toin.v3.json``. On a case-sensitive
    filesystem the spellings are genuinely different files and stay distinct.
    """
    lower = tmp_path / "toin.v3.json"
    lower.write_text(json.dumps({"version": "3.0", "patterns": {}}), encoding="utf-8", newline="\n")
    (tmp_path / "case_probe").write_text("x", encoding="utf-8")
    if not (tmp_path / "CASE_PROBE").exists():
        pytest.skip("case-sensitive filesystem: the alias is a different file by design")

    backend = FileSystemTOINBackend(str(tmp_path / "TOIN.V3.JSON"), store_version="3.0")

    # Recognized as already versioned: written in place, no sibling derived.
    assert backend.path == tmp_path / "TOIN.V3.JSON"
    assert backend._pre_upgrade_path is None
    twin = FileSystemTOINBackend(str(lower), store_version="3.0")
    assert backend._owner_token() == twin._owner_token()
    assert twin.load() == {"version": "3.0", "patterns": {}}


def test_mixed_case_version_marker_on_case_sensitive_volume(tmp_path) -> None:
    """A mixed-case version marker is recognized on case-sensitive volumes too.

    The lowered comparison means ``toin.V3.json`` is treated as already
    versioned everywhere; on a case-sensitive volume that spelling is its
    own file, written in place with no sibling and no pre-upgrade source.
    """
    (tmp_path / "case_probe").write_text("x", encoding="utf-8")
    if (tmp_path / "CASE_PROBE").exists():
        pytest.skip("case-insensitive volume: covered by the case-alias test")

    path = tmp_path / "toin.V3.json"
    path.write_text(json.dumps({"version": "3.0", "patterns": {}}), encoding="utf-8", newline="\n")

    backend = FileSystemTOINBackend(str(path), store_version="3.0")

    assert backend.path == path
    assert backend._pre_upgrade_path is None
    assert not (tmp_path / "toin.V3.v3.json").exists()


def test_case_sensitive_volume_keeps_distinct_owners_across_a_symlink(tmp_path, caplog) -> None:
    """Case-variant configured paths on a case-sensitive volume are distinct proxies.

    Folding the owner identity unconditionally would let a second proxy whose
    configured path differs only in case import — and then overwrite — the
    first proxy's history through a symlink between the two stores. The
    identity stays exact where the volume says the spellings differ.
    """
    (tmp_path / "case_probe").write_text("x", encoding="utf-8")
    if (tmp_path / "CASE_PROBE").exists():
        pytest.skip("case-insensitive volume: the alias spellings are one store by design")

    first = ToolIntelligenceNetwork(
        TOINConfig(storage_path=str(tmp_path / "team.json"), auto_save_interval=0)
    )
    _record(first, 0)
    first.save()
    store = first._backend.path
    assert store == tmp_path / "team.v3.json"

    link = tmp_path / "Team.v3.json"
    try:
        link.symlink_to(store)
    except OSError:
        pytest.skip("symlink creation unavailable on this platform/user")

    second = ToolIntelligenceNetwork(
        TOINConfig(storage_path=str(tmp_path / "Team.json"), auto_save_interval=0)
    )
    # The first proxy's history must not be imported as the second's own.
    assert second._patterns == {}
    before = store.read_bytes()
    _record(second, 1)
    with caplog.at_level(logging.WARNING, logger="headroom.telemetry.toin"):
        second.save()
    # Its save must not replace the link: assert at the alias, since an
    # atomic save over the alias leaves the target's bytes unchanged.
    assert link.is_symlink()
    assert store.read_bytes() == before
    assert any(getattr(record, "event", None) == "toin_save_failed" for record in caplog.records)


def test_case_probe_cannot_be_redirected_onto_a_victim_file(tmp_path) -> None:
    """The probe's write must not follow a symlink pre-planted at a predictable name.

    A local writer of the store directory used to be able to plant
    ``.headroom_case_probe_<pid>`` as a symlink and have the proxy truncate
    the target with its own permissions. The probe now creates its file
    exclusively under an unpredictable name, so a planted name is never
    written through.
    """
    victim = tmp_path / "victim.txt"
    victim.write_text("do not truncate", encoding="utf-8", newline="\n")
    planted = tmp_path / f".headroom_case_probe_{os.getpid()}"
    try:
        planted.symlink_to(victim)
    except OSError:
        pytest.skip("symlink creation unavailable on this platform/user")

    _volume_is_case_insensitive(tmp_path)

    assert victim.read_text(encoding="utf-8") == "do not truncate"


def test_case_classification_from_existing_entries_writes_nothing(tmp_path, monkeypatch) -> None:
    """An existing entry classifies the volume without any write.

    A readable-but-unwritable store directory (a macOS reader without write
    access) must still be classified, or the reader hashes a different
    owner identity than the writer stamped and rejects its own saved store.
    Writes are forbidden outright, so the test fails if classification ever
    falls through to the probe.
    """
    (tmp_path / "store.json").write_text("{}", encoding="utf-8", newline="\n")

    def refuse_write(*args, **kwargs):
        raise OSError("writes are forbidden during classification")

    monkeypatch.setattr(Path, "mkdir", refuse_write)
    monkeypatch.setattr(Path, "write_text", refuse_write)

    result = _volume_is_case_insensitive(tmp_path)

    assert result in (True, False)  # measured, not unknown
    assert list(tmp_path.glob(".headroom_case_probe*")) == []


def test_unicode_basename_does_not_spoof_case_sensitivity(tmp_path) -> None:
    """Case classification must not expand Unicode characters.

    Python's ``str.upper()`` maps ``ß`` to ``SS`` one-to-many, but volumes
    fold case one-to-one: on an insensitive NTFS volume a missing
    ``STRASSE.V3.JSON`` spelling is not evidence of case sensitivity. The
    variant comparison flips a single ASCII letter instead, so the
    classification of a directory holding only ``Straße.v3.json`` stays
    measured, and a saved store is not disowned on restart.
    """
    (tmp_path / "case_probe").write_text("x", encoding="utf-8", newline="\n")
    if not (tmp_path / "CASE_PROBE").exists():
        pytest.skip("case-sensitive volume: the spelling genuinely is a different file")
    (tmp_path / "case_probe").unlink()

    (tmp_path / "Straße.v3.json").write_text("{}", encoding="utf-8", newline="\n")

    assert _volume_is_case_insensitive(tmp_path) is True


def test_unicode_configured_path_survives_restart_on_insensitive_volume(tmp_path) -> None:
    """Save and reopen a store whose configured basename expands under upper()."""
    (tmp_path / "case_probe").write_text("x", encoding="utf-8", newline="\n")
    if not (tmp_path / "CASE_PROBE").exists():
        pytest.skip("case-sensitive volume: the spelling genuinely is a different file")
    (tmp_path / "case_probe").unlink()

    toin = ToolIntelligenceNetwork(
        TOINConfig(storage_path=str(tmp_path / "Straße.json"), auto_save_interval=0)
    )
    _record(toin, 0)
    toin.save()

    restarted = ToolIntelligenceNetwork(TOINConfig(storage_path=str(tmp_path / "Straße.json")))
    assert restarted._patterns


def test_ownerless_legacy_sibling_is_not_adopted(tmp_path) -> None:
    """A legacy store at the derived sibling belongs to another proxy.

    Reusing an existing alternate store requires positive evidence of the
    migration: the expected layout version and this configured path's owner
    stamp. An ownerless sibling is left untouched and the fresh store is
    written at the configured name.
    """
    configured = tmp_path / "toin.v3.json"
    sibling = tmp_path / "toin.v3.json.v3"
    legacy = _saved_store(sibling, 2)
    legacy["version"] = "2.0"
    sibling.write_text(json.dumps(legacy), encoding="utf-8", newline="\n")
    before = sibling.read_bytes()

    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(configured), auto_save_interval=0))
    assert toin._patterns == {}  # the legacy sibling was not imported
    _record(toin, 7)
    toin.save()

    assert toin._backend.path == configured
    assert sibling.read_bytes() == before
    saved = json.loads(configured.read_text(encoding="utf-8"))
    assert saved["version"] == "3.0"
    restarted = ToolIntelligenceNetwork(
        TOINConfig(storage_path=str(configured), auto_save_interval=0)
    )
    assert _signature(7).structure_hash in {
        pattern.tool_signature_hash for pattern in restarted._patterns.values()
    }


def test_save_replaces_a_symlink_without_orphaning_the_store(tmp_path) -> None:
    """An atomic save replaces the link, so the owner token must name the link.

    ``_owner`` used to resolve through the symlink to its target; the save
    replaced the link with a regular file stamped with the target's token and
    the next start rejected its own store.
    """
    target = tmp_path / "store.json"
    # A v3-shaped store behind the link: the version probe must see the
    # current layout, not a legacy one, or the backend migrates to a sibling
    # and never touches the link.
    target.write_text(
        json.dumps({"version": "3.0", "patterns": {}}), encoding="utf-8", newline="\n"
    )
    # A version-like configured name is written in place, so the active
    # write path is the symlink itself (an unversioned name would save to
    # the .v3 sibling and never touch the link).
    link = tmp_path / "toin.v3.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable on this platform/user")

    first = ToolIntelligenceNetwork(TOINConfig(storage_path=str(link), auto_save_interval=0))
    _record(first, 0)
    first.save()

    # The link was replaced by a regular file at the configured name.
    assert link.is_symlink() is False
    restarted = ToolIntelligenceNetwork(TOINConfig(storage_path=str(link), auto_save_interval=0))
    assert restarted._patterns  # own store accepted, not rejected as foreign


def test_filesystem_backend_writes_compact_json(tmp_path) -> None:
    path = tmp_path / "toin.json"
    FileSystemTOINBackend(str(path)).save({"patterns": {"a": {"value": 1}}})

    assert "\n" not in path.read_text(encoding="utf-8")
    assert "  " not in path.read_text(encoding="utf-8")


@pytest.mark.windows_newline
def test_filesystem_backend_writes_pin_lf(monkeypatch, tmp_path) -> None:
    import headroom.telemetry.backends.filesystem as filesystem

    calls: list[dict] = []
    real_open = open

    def recording_open(fd, *args, **kwargs):
        calls.append(kwargs)
        return real_open(fd, *args, **kwargs)

    monkeypatch.setattr(filesystem, "open", recording_open, raising=False)
    FileSystemTOINBackend(str(tmp_path / "toin.json")).save({"patterns": {}})

    # Artifact bytes alone cannot detect a missing newline pin on POSIX.
    assert [call.get("newline") for call in calls] == ["\n"]


def test_filesystem_backend_skips_oversized_store(tmp_path) -> None:
    path = tmp_path / "toin.json"
    path.write_text("x" * 100, encoding="utf-8")

    backend = FileSystemTOINBackend(str(path), max_load_bytes=10)
    assert backend.load() == {}
    assert path.exists()


def test_filesystem_backend_preserves_oversized_store_on_save(tmp_path) -> None:
    path = tmp_path / "toin.json"
    original = b'{"patterns":{"existing":{"value":1}}}'
    path.write_bytes(original)
    backend = FileSystemTOINBackend(str(path), max_load_bytes=len(original) - 1)

    assert backend.load() == {}
    with pytest.raises(ValueError, match="limit"):
        backend.save({})

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".toin_*.tmp")) == []


@pytest.mark.parametrize("existing", [False, True], ids=["new_store", "existing_store"])
def test_filesystem_backend_rejects_oversized_output(tmp_path, existing) -> None:
    path = tmp_path / "toin.json"
    original = b"{}"
    if existing:
        path.write_bytes(original)
    data = {"value": "x" * 100}
    limit = len(json.dumps(data, separators=(",", ":")).encode("utf-8")) - 1
    backend = FileSystemTOINBackend(str(path), max_load_bytes=limit)

    with pytest.raises(ValueError, match="limit"):
        backend.save(data)

    if existing:
        assert path.read_bytes() == original
    else:
        assert not path.exists()
    assert list(tmp_path.glob(".toin_*.tmp")) == []


def test_filesystem_backend_accepts_exact_byte_limit(tmp_path) -> None:
    path = tmp_path / "toin.json"
    data = {"value": "雪"}
    serialized = json.dumps(data, separators=(",", ":")).encode("utf-8")
    backend = FileSystemTOINBackend(str(path), max_load_bytes=len(serialized))

    backend.save(data)
    assert path.read_bytes() == serialized
    assert backend.load() == data
    # An existing file at the boundary can still be replaced.
    backend.save({})
    assert backend.load() == {}


def test_filesystem_backend_zero_byte_limit_rejects_output(tmp_path) -> None:
    path = tmp_path / "toin.json"
    backend = FileSystemTOINBackend(str(path), max_load_bytes=0)

    with pytest.raises(ValueError, match="limit"):
        backend.save({})

    assert not path.exists()
    assert list(tmp_path.glob(".toin_*.tmp")) == []


def test_filesystem_backend_without_byte_limit_allows_growth(tmp_path) -> None:
    path = tmp_path / "toin.json"
    backend = FileSystemTOINBackend(str(path), max_load_bytes=None)
    backend.save({"value": "x" * 100})
    data = {"value": "x" * 200}

    backend.save(data)

    assert backend.load() == data


@pytest.mark.parametrize("oversized_on_load", [True, False], ids=["load_rejected", "store_grows"])
def test_toin_size_rejection_preserves_store_and_dirty_state(
    tmp_path, caplog, oversized_on_load
) -> None:
    base = tmp_path / "toin.json"
    path = _current_store_path(base)
    _saved_store(path, 3 if oversized_on_load else 1)
    original = path.read_bytes()
    limit = len(original) - 1 if oversized_on_load else len(original) + 128
    toin = ToolIntelligenceNetwork(
        TOINConfig(storage_path=str(base), max_storage_bytes=limit, auto_save_interval=0)
    )
    assert len(toin._patterns) == (0 if oversized_on_load else 1)
    _record(toin, 99)
    last_save_time = toin._last_save_time

    with caplog.at_level(logging.WARNING, logger="headroom.telemetry.toin"):
        toin.save()

    assert path.read_bytes() == original
    assert toin._dirty
    assert toin._last_save_time == last_save_time
    assert list(tmp_path.glob(".toin_*.tmp")) == []
    [warning] = [r for r in caplog.records if getattr(r, "event", None) == "toin_save_failed"]
    assert warning.error_type == "ValueError"
