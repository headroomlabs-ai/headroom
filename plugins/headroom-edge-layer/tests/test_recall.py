from __future__ import annotations

from recall import extract_referenced_items, next_step_recall


def test_extracts_file_paths_and_errors() -> None:
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "src/foo.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
            {"type": "text", "text": "Traceback: NameError in src/foo.py"}
        ]}]},
    ]
    items = extract_referenced_items(messages)
    assert "src/foo.py" in items
    assert any("Error" in item for item in items)


def test_recall_is_one_when_everything_referenced_is_present() -> None:
    # Just the path, deliberately with no other 5+-char words nearby — the
    # identifier extractor is intentionally broad (see recall.py's docstring
    # on false positives), so this isolates the one item under test.
    next_messages = [{"role": "assistant", "content": [{"type": "text", "text": "src/foo.py"}]}]
    compressed_text = "the file src/foo.py contains ..."
    score = next_step_recall(compressed_text, next_messages)
    assert score == 1.0


def test_recall_drops_when_referenced_item_is_missing() -> None:
    next_messages = [{"role": "assistant", "content": [{"type": "text", "text": "check src/foo.py and src/bar.py"}]}]
    compressed_text = "only src/foo.py survived compression"
    score = next_step_recall(compressed_text, next_messages)
    assert score is not None
    assert 0.0 < score < 1.0


def test_recall_is_none_when_nothing_to_check() -> None:
    score = next_step_recall("anything", [])
    assert score is None
