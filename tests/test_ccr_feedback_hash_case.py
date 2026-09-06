"""The streaming CCR feedback recorders must look the hash up in lowercase.

The compression store keys every entry by a lowercase hash, so a model that
echoes the ``headroom_retrieve`` marker hash in uppercase would miss the store
and the retrieval would never reach the TOIN feedback loop. The recorders
extract the hash directly (they do not go through ``parse_tool_call``, which was
already fixed), so they must normalise it themselves.
"""

from __future__ import annotations

import headroom.cache.compression_store as cs
from headroom.proxy.handlers.streaming import StreamingMixin

HASH_LOWER = "abc123def456abc123def456"


class _SpyStore:
    def __init__(self) -> None:
        self.looked_up: list[str] = []

    def retrieve(self, hash_key, query=None):  # noqa: ANN001, ANN201
        self.looked_up.append(hash_key)
        return None


def test_response_feedback_lowercases_hash(monkeypatch):
    spy = _SpyStore()
    monkeypatch.setattr(cs, "get_compression_store", lambda: spy)

    handler = object.__new__(StreamingMixin)
    response = {
        "content": [
            {
                "type": "tool_use",
                "name": "headroom_retrieve",
                "input": {"hash": HASH_LOWER.upper()},
            }
        ]
    }
    handler._record_ccr_feedback_from_response(response, "anthropic", "req-1")

    assert spy.looked_up == [HASH_LOWER]


def test_openai_stream_feedback_lowercases_hash(monkeypatch):
    spy = _SpyStore()
    monkeypatch.setattr(cs, "get_compression_store", lambda: spy)

    import json

    handler = object.__new__(StreamingMixin)
    sse = (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "headroom_retrieve",
                                        "arguments": json.dumps({"hash": HASH_LOWER.upper()}),
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )
        + "\n"
    )
    handler._record_ccr_feedback_from_openai_sse(sse, "req-1")

    assert spy.looked_up == [HASH_LOWER]


def test_response_feedback_ignores_non_string_hash(monkeypatch):
    # A malformed non-string hash (e.g. {"hash": 123}) must be skipped, not
    # raise in .lower().
    spy = _SpyStore()
    monkeypatch.setattr(cs, "get_compression_store", lambda: spy)

    handler = object.__new__(StreamingMixin)
    response = {
        "content": [{"type": "tool_use", "name": "headroom_retrieve", "input": {"hash": 123}}]
    }
    handler._record_ccr_feedback_from_response(response, "anthropic", "req-1")

    assert spy.looked_up == []


def test_openai_stream_feedback_ignores_non_string_hash(monkeypatch):
    import json

    spy = _SpyStore()
    monkeypatch.setattr(cs, "get_compression_store", lambda: spy)

    handler = object.__new__(StreamingMixin)
    sse = (
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "headroom_retrieve",
                                        "arguments": json.dumps({"hash": 123}),
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )
        + "\n"
    )
    handler._record_ccr_feedback_from_openai_sse(sse, "req-1")

    assert spy.looked_up == []
