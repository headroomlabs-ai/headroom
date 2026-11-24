"""Regression tests for TOIN privacy and bounded retention."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.proxy.tenant_key import _sanitize_tenant_key
from headroom.telemetry.backends.filesystem import FileSystemTOINBackend
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
    path.write_text(json.dumps(data), encoding="utf-8")
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
