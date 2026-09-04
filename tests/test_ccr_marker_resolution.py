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


@pytest.mark.parametrize(
    "marker_template",
    [
        "<<ccr:{hash},string,23.6KB>>",
        "[100 items compressed to 10. Retrieve more: hash={hash}]",
        "[Read content stale. Retrieve original: hash={hash}]",
    ],
)
@pytest.mark.parametrize("uppercase", [False, True])
def test_resolve_markers_in_text_replaces_all_supported_hits(marker_template: str, uppercase: bool):
    hash_key = _store_entry("the original uncompressed content")
    marker_hash = hash_key.upper() if uppercase else hash_key
    text = f"before {marker_template.format(hash=marker_hash)} after"

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


@pytest.mark.parametrize(
    "marker_template",
    [
        "<<ccr:{hash},string,1KB>>",
        "[100 items compressed to 10. Retrieve more: hash={hash}]",
        "[Read content stale. Retrieve original: hash={hash}]",
    ],
)
def test_resolve_markers_in_text_miss_leaves_all_supported_markers_with_reason(
    marker_template: str,
):
    text = marker_template.format(hash="deadbeefdeadbeef")

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
