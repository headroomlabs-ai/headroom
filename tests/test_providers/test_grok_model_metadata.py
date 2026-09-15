from __future__ import annotations

import pytest

from headroom.providers.grok.model_metadata import (
    is_xai_model_list_target,
    normalize_xai_model_metadata,
)
from headroom.providers.grok.runtime import DEFAULT_API_URL


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (DEFAULT_API_URL, True),
        (f"{DEFAULT_API_URL}/v1/", True),
        ("https://api.x.ai.evil.test", False),
        ("https://api.x.ai/v2", False),
        ("https://api.openai.com", False),
    ],
)
def test_xai_model_list_target_uses_exact_host_and_optional_v1_path(
    url: str, expected: bool
) -> None:
    assert is_xai_model_list_target(url) is expected


def test_normalize_xai_model_metadata_copies_only_positive_integer_context_length() -> None:
    payload = {
        "object": "list",
        "data": [
            {"id": "grok-4.6", "context_length": 500000, "nested": {"keep": True}},
            {"id": "zero", "context_length": 0},
            {"id": "negative", "context_length": -1},
            {"id": "bool", "context_length": True},
            {"id": "string", "context_length": "500000"},
            {"id": "snake", "context_length": 10, "context_window": 20},
            {"id": "camel", "context_length": 10, "contextWindow": 30},
            "not-an-entry",
        ],
    }

    normalized = normalize_xai_model_metadata(payload)

    assert normalized["object"] == "list"
    assert normalized["data"][0] == {
        "id": "grok-4.6",
        "context_length": 500000,
        "context_window": 500000,
        "nested": {"keep": True},
    }
    assert normalized["data"][1:] == payload["data"][1:]
    assert payload["data"][0] == {
        "id": "grok-4.6",
        "context_length": 500000,
        "nested": {"keep": True},
    }


def test_normalize_xai_model_metadata_returns_none_when_no_entry_changes() -> None:
    assert normalize_xai_model_metadata({"data": [{"id": "grok", "context_length": 0}]}) is None
    assert normalize_xai_model_metadata({"data": [{"id": "grok", "contextWindow": 10}]}) is None
    assert normalize_xai_model_metadata({"data": []}) is None
    assert normalize_xai_model_metadata({"data": {}}) is None
