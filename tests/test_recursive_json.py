"""Embedded JSON routing: preserved surroundings and per-adopted-span recovery."""

from __future__ import annotations

import json

from headroom.transforms.recursive_json import (
    carries_record_array,
    json_document_spans,
    route_embedded_json,
)


def _minify_dispatch(span: str) -> tuple[str, bool]:
    return json.dumps(json.loads(span), separators=(",", ":")), True


def test_embedded_json_routed_and_surroundings_exact() -> None:
    rows = [{"id": i, "ok": True} for i in range(6)]
    payload = json.dumps(rows, indent=2)
    content = f"Fetched rows from API:\n{payload}\nDone (200 OK)."
    assert route_embedded_json(content, _minify_dispatch) == (
        f"Fetched rows from API:\n{json.dumps(rows, separators=(',', ':'))}\nDone (200 OK).",
        True,
    )


def test_ccr_marker_span_passed_through() -> None:
    content = 'prefix [{"a":1,"b":2},{"a":3,"b":"<<ccr:deadbeef,json,900>>"}] suffix'
    assert route_embedded_json(content, _minify_dispatch) is None


def test_no_json_is_noop() -> None:
    assert route_embedded_json("just prose, nothing structured here", _minify_dispatch) is None


def test_whole_block_json_is_callers_job() -> None:
    content = json.dumps([{"a": i} for i in range(5)], separators=(",", ":"))
    assert route_embedded_json(content, _minify_dispatch) is None


def test_benefit_gate_declines_when_not_smaller() -> None:
    payload = json.dumps([{"a": i} for i in range(5)])
    assert route_embedded_json(f"x {payload} y", lambda s: (s + " " * 999, True)) is None


def test_unmarked_loss_cannot_borrow_another_spans_marker() -> None:
    first = json.dumps([{"id": i} for i in range(10)], indent=2)
    second = json.dumps([{"value": i} for i in range(10)], indent=2)

    def dispatch(span: str) -> tuple[str, bool]:
        rows = json.loads(span)
        if "id" in rows[0]:
            return "<<ccr:deadbeef,json,100>>", True
        return json.dumps(rows[:1]), False

    for left, right in ((first, second), (second, first)):
        result = route_embedded_json(f"first: {left}\nsecond: {right}", dispatch, tok=len)
        assert result is not None
        text, reversible = result
        assert "<<ccr:deadbeef,json,100>>" in text
        assert '{"value": 0}' in text and '{"value": 9}' not in text
        assert reversible is False


def test_rejected_lossy_candidate_does_not_taint_adopted_minification() -> None:
    first = json.dumps([{"id": i} for i in range(10)], indent=2)
    second = json.dumps([{"value": i} for i in range(10)], indent=2)

    def dispatch(span: str) -> tuple[str, bool]:
        rows = json.loads(span)
        if "id" in rows[0]:
            return json.dumps(rows[:1]) + " " * len(span), False
        return _minify_dispatch(span)

    assert route_embedded_json(f"first: {first}\nsecond: {second}", dispatch, tok=len) == (
        f"first: {first}\nsecond: {json.dumps(json.loads(second), separators=(',', ':'))}",
        True,
    )


def test_scalar_array_not_routed() -> None:
    assert route_embedded_json("nums: [1,2,3,4,5,6,7,8] done", _minify_dispatch) is None


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
