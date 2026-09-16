"""Tests for inline <<ccr:...>> marker resolution (issue #2509)."""

from __future__ import annotations

import json

import pytest

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.ccr.marker_resolution import (
    resolve_markers_in_response,
    resolve_markers_in_text,
)


@pytest.fixture(autouse=True)
def reset_store():
    reset_compression_store()
    yield
    reset_compression_store()


def _store_entry(original: str) -> str:
    store = get_compression_store()
    return store.store(
        original=original,
        compressed="[]",
        original_item_count=1,
        compressed_item_count=0,
    )


def test_resolve_markers_in_text_no_marker_is_noop():
    assert resolve_markers_in_text("plain text, no markers here") == "plain text, no markers here"


def test_resolve_markers_in_text_replaces_hit():
    hash_key = _store_entry("the original uncompressed content")
    text = f"before <<ccr:{hash_key},string,23.6KB>> after"

    resolved = resolve_markers_in_text(text)

    assert resolved == "before the original uncompressed content after"


def test_resolve_markers_in_text_replaces_multiple_hits():
    hash_a = _store_entry("AAA")
    hash_b = _store_entry("BBB")
    text = f"<<ccr:{hash_a},string,1KB>> and <<ccr:{hash_b},string,1KB>>"

    resolved = resolve_markers_in_text(text)

    assert resolved == "AAA and BBB"


def test_resolve_markers_in_text_json_array_original_content():
    store = get_compression_store()
    hash_key = store.store(
        original=json.dumps([1, 2, 3]),
        compressed="[]",
        original_item_count=3,
        compressed_item_count=0,
    )
    text = f"<<ccr:{hash_key},array,3>>"

    resolved = resolve_markers_in_text(text)

    assert json.loads(resolved) == [1, 2, 3]


def test_resolve_markers_in_text_miss_leaves_marker_with_reason():
    text = "<<ccr:deadbeefdeadbeef,string,1KB>>"

    resolved = resolve_markers_in_text(text)

    assert text in resolved
    assert "[unresolved:" in resolved


def test_resolve_markers_in_response_walks_nested_structure():
    hash_key = _store_entry("full tool output")
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": f"here it is: <<ccr:{hash_key},string,1KB>>",
                }
            }
        ],
        "unrelated": 42,
        "nested": {"list": ["a", f"<<ccr:{hash_key},string,1KB>>", "c"]},
    }

    resolved = resolve_markers_in_response(response)

    assert resolved["choices"][0]["message"]["content"] == "here it is: full tool output"
    assert resolved["nested"]["list"] == ["a", "full tool output", "c"]
    assert resolved["unrelated"] == 42
