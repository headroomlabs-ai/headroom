"""headroom-37g.36: standalone unit coverage for the moved agy FR compressor.

Proves the algorithm is unit-testable directly from
``headroom.transforms.agy_fr_compressor`` without booting the FastAPI app
(no ``create_app`` / ``TestClient``) -- the altitude payoff of the pure move
out of ``GeminiHandlerMixin``.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.tokenizers import get_tokenizer
from headroom.transforms.agy_fr_compressor import (
    _FR_CCR_MARKER_PREFIX,
    compress_function_response_leaves,
)

_MODEL = "gemini-3-flash-agent"

# Large, single-line, non-repeating-line leaf: well above the marker-derived
# floor (~2x a ~20-token marker), so it is compressed.
_COMPRESSIBLE_LEAF = "search result row alpha beta gamma delta epsilon zeta eta " * 40

# Tiny leaf: below the marker-derived floor, so it must be left untouched.
_SUB_FLOOR_LEAF = "ok"


@pytest.fixture
def tok() -> Any:
    return get_tokenizer(_MODEL)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    reset_compression_store()
    s = get_compression_store()
    yield s
    reset_compression_store()


def _contents() -> list[dict]:
    return [
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "search",
                        "response": {"output": _COMPRESSIBLE_LEAF},
                    }
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "search",
                        "response": {"output": _SUB_FLOOR_LEAF},
                    }
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        # Exempt: headroom_retrieve's own output is never
                        # re-compressed (would self-defeating-loop).
                        "name": "headroom_retrieve",
                        "response": {"output": _COMPRESSIBLE_LEAF},
                    }
                }
            ],
        },
    ]


def test_compress_function_response_leaves_standalone(tok: Any, store: Any) -> None:
    contents = _contents()

    before, after, leaves = compress_function_response_leaves(contents, "ccr", tok, store)

    # Only the one compressible leaf is counted.
    assert leaves == 1
    assert before > after > 0

    compressible = contents[0]["parts"][0]["functionResponse"]["response"]["output"]
    sub_floor = contents[1]["parts"][0]["functionResponse"]["response"]["output"]
    exempt = contents[2]["parts"][0]["functionResponse"]["response"]["output"]

    assert compressible.startswith(_FR_CCR_MARKER_PREFIX)
    assert sub_floor == _SUB_FLOOR_LEAF
    assert exempt == _COMPRESSIBLE_LEAF
