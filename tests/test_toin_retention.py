"""Regression tests for TOIN privacy and bounded retention."""

from __future__ import annotations

import json
import logging

import pytest

from headroom.telemetry.backends.filesystem import FileSystemTOINBackend
from headroom.telemetry.models import ToolSignature
from headroom.telemetry.toin import TOINConfig, ToolIntelligenceNetwork


def _signature(index: int) -> ToolSignature:
    return ToolSignature.from_items([{f"field_{index}": "x"}])


def test_free_form_queries_are_not_persisted() -> None:
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=""))

    assert toin._anonymize_query_pattern("show me the contents of /private/secret.txt") is None
    assert toin._anonymize_query_pattern("status:error AND user:john") == "status:* user:*"

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

    assert toin._anonymize_query_pattern("field:" + "x" * 1000) == "field:*"
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
                    "status:* user:*": 2,
                },
                "common_query_patterns": [
                    "raw prompt containing secret source code",
                    "status:* user:*",
                ],
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path)))
    pattern = next(iter(toin._patterns.values()))

    assert pattern.query_pattern_frequency == {"status:* user:*": 2}
    assert pattern.common_query_patterns == ["status:* user:*"]


def _record(toin: ToolIntelligenceNetwork, index: int) -> None:
    toin.record_compression(
        tool_signature=_signature(index),
        original_count=10,
        compressed_count=5,
        original_tokens=100,
        compressed_tokens=50,
        strategy="test",
    )


def _saved_store(path, count: int) -> dict:
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=str(path), auto_save_interval=0))
    for index in range(count):
        _record(toin, index)
    toin.save()
    return json.loads(path.read_text(encoding="utf-8"))


def _skip_warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == "toin_import_skipped"]


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda entry: {k: v for k, v in entry.items() if k != "tool_signature_hash"},
        lambda entry: {**entry, "field_semantics": {"field_1": "not a dict"}},
        lambda entry: {**entry, "query_pattern_frequency": "not a dict"},
        lambda entry: {**entry, "query_pattern_frequency": {"status:*": "many"}},
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
    saved = json.loads(path.read_text(encoding="utf-8"))["patterns"]
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
    saved = json.loads(path.read_text(encoding="utf-8"))["patterns"]
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


def test_filesystem_backend_writes_compact_json(tmp_path) -> None:
    path = tmp_path / "toin.json"
    FileSystemTOINBackend(str(path)).save({"patterns": {"a": {"value": 1}}})

    assert "\n" not in path.read_text(encoding="utf-8")
    assert "  " not in path.read_text(encoding="utf-8")


def test_filesystem_backend_skips_oversized_store(tmp_path) -> None:
    path = tmp_path / "toin.json"
    path.write_text("x" * 100, encoding="utf-8")

    backend = FileSystemTOINBackend(str(path), max_load_bytes=10)
    assert backend.load() == {}
    assert path.exists()
