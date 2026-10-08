"""Unit tests for headroom.transforms.recursive_json — the structural (embedded)
JSON routing step. Uses a fake dispatch so the mechanism is tested in isolation
from the real compressors."""

from __future__ import annotations

import json

from headroom.transforms.recursive_json import (
    carries_record_array,
    json_document_spans,
    route_embedded_json,
)


def _upper_dispatch(span: str) -> str | None:
    """Fake compressor: returns a shorter deterministic stand-in for any span."""
    try:
        v = json.loads(span)
    except ValueError:
        return None
    return f"<TABLE n={len(v)}>" if isinstance(v, list) else "<OBJ>"


def test_embedded_json_routed_and_surroundings_exact() -> None:
    payload = json.dumps([{"id": i, "ok": True} for i in range(6)], separators=(",", ":"))
    content = f"Fetched rows from API:\n{payload}\nDone (200 OK)."
    out = route_embedded_json(content, _upper_dispatch)
    assert out is not None
    assert out.startswith("Fetched rows from API:\n")
    assert out.endswith("\nDone (200 OK).")
    assert "<TABLE n=6>" in out


def test_ccr_marker_span_passed_through() -> None:
    # A span already carrying a CCR marker must never be re-routed (R1).
    content = 'prefix [{"a":1,"b":2},{"a":3,"b":"<<ccr:deadbeef,json,900>>"}] suffix'
    out = route_embedded_json(content, _upper_dispatch)
    assert out is None  # only span contains a marker → skipped → nothing to do


def test_no_json_is_noop() -> None:
    assert route_embedded_json("just prose, nothing structured here", _upper_dispatch) is None


def test_whole_block_json_is_callers_job() -> None:
    # A block that IS a single JSON value is skipped (routed by the caller).
    content = json.dumps([{"a": i} for i in range(5)], separators=(",", ":"))
    assert route_embedded_json(content, _upper_dispatch) is None


def test_benefit_gate_declines_when_not_smaller() -> None:
    payload = json.dumps([{"a": i} for i in range(5)], separators=(",", ":"))
    content = f"x {payload} y"
    # Dispatch that returns something LARGER → must be declined (outcome gate).
    assert route_embedded_json(content, lambda s: s + " " * 999) is None


def test_deterministic() -> None:
    payload = json.dumps([{"k": i} for i in range(8)], separators=(",", ":"))
    content = f"a {payload} b {payload} c"
    r1 = route_embedded_json(content, _upper_dispatch)
    r2 = route_embedded_json(content, _upper_dispatch)
    assert r1 == r2 and r1 is not None
    assert r1.count("<TABLE n=8>") == 2  # both embedded spans routed


def test_scalar_array_not_routed() -> None:
    # array of scalars is not a "routable" JSON shape (no dict rows)
    content = "nums: [1,2,3,4,5,6,7,8] done"
    assert route_embedded_json(content, _upper_dispatch) is None


def test_json_document_spans_finds_containers_anywhere() -> None:
    doc = json.dumps({"domains": [{"name": "a"}, {"name": "b"}]})
    arr = json.dumps([1, 2, 3])
    text = "Tool result: " + doc + " and a list " + arr + " done"
    assert [text[a:b] for a, b in json_document_spans(text)] == [doc, arr]


def test_json_document_spans_ignores_scalars_prose_and_unbalanced_json() -> None:
    assert json_document_spans('"just a quoted sentence"') == []
    assert json_document_spans("42") == []
    assert json_document_spans("prose with a [note] and {braces} but no JSON") == []
    assert json_document_spans('{"truncated": [1, 2, 3') == []
    assert json_document_spans("{{HEADROOM_TAG_0}}") == []
    assert json_document_spans("") == []


def test_json_document_spans_whole_document() -> None:
    doc = json.dumps({"a": [1, 2]})
    assert json_document_spans(doc) == [(0, len(doc))]
    assert json_document_spans("  " + doc + "\n") == [(2, 2 + len(doc))]


def test_carries_record_array_separates_the_two_shapes() -> None:
    assert carries_record_array(json.dumps({"domains": [{"name": "a"}, {"name": "b"}]}))
    assert carries_record_array(json.dumps([{"id": 1}, {"id": 2}]))
    # A lone object, or an array of scalars, has no record delimiter whose
    # deletion leaves a valid-but-shorter document.
    assert not carries_record_array(json.dumps({"file": "src/mod.py", "line": 1}))
    assert not carries_record_array(json.dumps([1, 2, 3]))
    assert not carries_record_array(json.dumps([{"id": 1}]))


# --- the span walk is linear on brace-heavy text -----------------------------
#
# ``_spans`` used to call ``_match_span`` at every opening bracket, and a ``{``
# that never closes walks to the end of the text: source code and logs full of
# unmatched braces cost one full pass per brace. ``_scan_spans`` memoizes each
# bracket's verdict from the walk that already crossed it. These tests pin that
# the result did not change and that the work did.


def _reference_spans(text: str) -> list[tuple[int, int]]:
    """The previous ``_spans``: ``_match_span`` at every opening bracket."""
    from headroom.transforms.recursive_json import _match_span

    out: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] in "[{":
            end = _match_span(text, i)
            if end is not None:
                out.append((i, end))
                i = end
                continue
        i += 1
    return out


def test_span_walk_matches_the_previous_semantics_exactly() -> None:
    import random

    from headroom.transforms.recursive_json import _scan_spans

    rng = random.Random(3673)
    alphabet = '{}[]"\\ a,:\n'
    samples = [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 120))) for _ in range(4000)
    ]
    samples += [
        'noise {"a": [1, {"b": "}"}]} tail [1,2] {oops',
        'He said "a {quote" then {"k": "v"} and "another {one}"',
        '{"esc": "a \\" } still string"} {x',
        "[{]}] {[}] {{{{",
    ]
    for text in samples:
        spans, complete = _scan_spans(text)
        assert complete, text
        assert spans == _reference_spans(text), text


def test_span_walk_is_linear_on_unmatched_braces(monkeypatch) -> None:
    import headroom.transforms.recursive_json as rj

    line = '    if (flags & MASK) { log.debug("state={}", state); retry(ctx, {timeout: 30, max\n'
    text = line * 800  # ~70 KB, about three unmatched `{` per line
    walked = 0
    real = rj._scan_from

    def counting(t, start, known):  # noqa: ANN001, ANN202
        nonlocal walked
        end, n = real(t, start, known)
        walked += n
        return end, n

    monkeypatch.setattr(rj, "_scan_from", counting)
    spans, complete = rj._scan_spans(text)
    assert complete
    # One pass, give or take the per-line restarts inside quoted strings. The
    # previous walk re-read the rest of the text for every unmatched brace:
    # hundreds of times the input size here.
    assert walked <= 2 * len(text), (walked, len(text))
    # Same answer as before on the same shape (a slice keeps the quadratic
    # reference fast enough to run).
    small = line * 40
    assert rj._scan_spans(small)[0] == _reference_spans(small)


def test_scan_stops_at_its_budget_and_says_so(monkeypatch) -> None:
    import headroom.transforms.recursive_json as rj

    monkeypatch.setattr(rj, "_SCAN_BUDGET_PER_CHAR", 0)
    monkeypatch.setattr(rj, "_SCAN_BUDGET_FLOOR", 0)
    text = 'prose {"a": 1} then [{"id": 1}, {"id": 2}]'
    spans, complete = rj.scan_json_documents(text)
    assert complete is False
    assert spans == []
