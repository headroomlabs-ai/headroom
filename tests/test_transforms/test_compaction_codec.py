"""Inverse decoder + value-factoring for the ``csv-schema`` densified format.

These exercise the Python codec (``headroom.transforms.compaction_codec``)
against *real* Rust SmartCrusher output so the encoder and decoder stay in
lockstep:

- losslessness of densification is programmatically verifiable (round-trip);
- the null / empty-string / missing-key / literal ``\\N`` distinctions survive
  (the defect the densifier previously collapsed);
- value-factoring dictionary-encodes low-cardinality columns reversibly and
  only when it saves bytes.
"""

from __future__ import annotations

import json

from headroom.transforms.compaction_codec import (
    contains_removal_marker,
    expand_compacted,
    factor_values,
    is_compacted,
)
from headroom.transforms.smart_crusher import SmartCrusher


def _crush(records: list[dict]) -> str:
    return SmartCrusher().crush_array_json(json.dumps(records)).get("compacted") or ""


def test_round_trips_real_rust_output() -> None:
    records = [
        {"id": i, "owner": f"user{i % 5}", "score": i / 3, "active": i % 2 == 0} for i in range(40)
    ]
    comp = _crush(records)
    assert is_compacted(comp)
    assert expand_compacted(comp) == records


def test_null_empty_missing_and_literal_backslash_n_are_distinct() -> None:
    records: list[dict] = []
    for i in range(40):
        if i == 0:
            records.append({"a": i, "b": i, "c": None})  # null
        elif i == 1:
            records.append({"a": i, "b": i, "c": ""})  # empty string
        elif i == 2:
            records.append({"a": i, "b": i})  # missing key
        elif i == 3:
            records.append({"a": i, "b": i, "c": "\\N"})  # literal backslash-N
        else:
            records.append({"a": i, "b": i, "c": f"v{i}"})
    comp = _crush(records)
    decoded = expand_compacted(comp)
    assert decoded is not None
    assert "c" not in decoded[2]  # missing stays absent
    assert decoded[0]["c"] is None  # null
    assert decoded[1]["c"] == ""  # empty string
    assert decoded[3]["c"] == "\\N"  # literal preserved
    assert decoded == records


def test_strings_with_commas_quotes_and_unicode() -> None:
    records = [
        {"k": i, "v": v}
        for i, v in enumerate(
            ["a,b", 'he said "hi"', "café 日本語 🔥", "trailing space ", "", "line1"]
        )
    ]
    # pad so the array clears the densification token gate
    records += [{"k": 100 + i, "v": f"pad value {i}"} for i in range(30)]
    comp = _crush(records)
    assert expand_compacted(comp) == records


def test_nested_object_flattening_round_trips() -> None:
    records = [{"id": i, "meta": {"owner": f"u{i % 3}", "rank": i}} for i in range(30)]
    comp = _crush(records)
    decoded = expand_compacted(comp)
    assert decoded == records


def test_value_factoring_beats_plain_on_repeated_paths() -> None:
    paths = ["./a/very/long/path/one.py", "./another/long/path/two.py", "./third.py"]
    records = [
        {"path": paths[i % len(paths)], "line": 100 + i, "content": f"def f_{i}(): pass"}
        for i in range(40)
    ]
    plain = _crush(records)
    factored = factor_values(plain)
    assert "__dict:" in factored
    assert len(factored) < len(plain)  # strictly smaller
    assert expand_compacted(factored) == records  # still lossless


def test_value_factoring_is_idempotent() -> None:
    records = [{"path": f"./p{i % 3}.py", "line": i, "content": f"x{i}"} for i in range(30)]
    factored = factor_values(_crush(records))
    assert factor_values(factored) == factored


def test_value_factoring_noop_when_no_benefit() -> None:
    # All-distinct high-entropy column → no repetition to hoist.
    records = [{"id": i, "uuid": f"id-{i}-{'x' * 20}"} for i in range(30)]
    plain = _crush(records)
    assert factor_values(plain) == plain


def test_value_factoring_preserves_nulls_in_dict_column() -> None:
    records = [
        {"path": (None if i % 7 == 0 else f"./p{i % 3}.py"), "line": i, "content": f"c{i}"}
        for i in range(40)
    ]
    factored = factor_values(_crush(records))
    assert expand_compacted(factored) == records


def test_removal_marker_makes_text_unreversible() -> None:
    text = "[5]{a:int,blob:string}\n0,<<ccr:abc123,string,1.2KB>>\n"
    assert contains_removal_marker(text)
    assert expand_compacted(text) is None


def test_json_string_wrapped_block_is_recognized() -> None:
    records = [{"id": i, "tag": f"t{i % 4}"} for i in range(30)]
    bare = _crush(records)
    wrapped = json.dumps(bare)  # the form ContentRouter stores for tool results
    assert is_compacted(wrapped)
    assert expand_compacted(wrapped) == records


def test_non_compacted_text_is_ignored() -> None:
    assert not is_compacted("just some prose, not densified")
    assert expand_compacted("just some prose") is None
    assert factor_values("just some prose") == "just some prose"


def test_crlf_line_endings_round_trip() -> None:
    # A harness that round-trips the block through a Windows text-mode file (or
    # otherwise normalizes line endings) must still decode. The format is
    # LF-internal; the decoder tolerates CRLF defensively.
    records = [{"id": i, "name": f"n{i % 4}", "v": f"val {i}"} for i in range(30)]
    bare = _crush(records)
    crlf = bare.replace("\n", "\r\n")
    assert expand_compacted(crlf) == records
    assert expand_compacted(crlf) == expand_compacted(bare)


def test_value_factoring_handles_comma_and_quote_values() -> None:
    # Repeated values that need CSV-quoting must still factor losslessly.
    vals = ["a, b.py", 'has "quotes"', "c/d.py"]
    records = [{"path": vals[i % len(vals)], "line": i, "content": f"x{i}"} for i in range(30)]
    factored = factor_values(_crush(records))
    assert "__dict:" in factored
    assert expand_compacted(factored) == records
