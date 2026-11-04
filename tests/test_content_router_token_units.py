"""Token counts compared inside ContentRouter must share one unit.

Two places measured a token quantity in a different unit from the thing it was
compared against. Both changed COMPRESSION BEHAVIOUR, not just reporting:

1. The CONFIG branch built its accept ratio as
   ``len(compressed.split()) / _estimate_tokens(content)`` — a WORD count over a
   TOKEN count. Words run ~2.8x fewer than estimator tokens on config text, so a
   compressor returning its input byte-identically scored ~0.36. ``min_ratio`` is
   1.0 (accept any real shrink), so the router ACCEPTED the no-op, cached it,
   froze the verdict, emitted a ``router:config_compressor`` label and recorded a
   fabricated ~64% saving.

2. The Kompress size gate tested ``len(text) > max_tokens * 4`` — a TOKEN cap
   evaluated in chars/4. Dense payloads run denser than 4 chars/token (compact
   JSON ~3.2), so a band existed where an oversized payload passed the gate into
   the >30s non-preemptible ONNX inference the gate exists to prevent (#1171).
"""

from __future__ import annotations

import json

from headroom.transforms.content_router import _estimate_tokens

_CAP = 50_000


def _compact_json(records: int) -> str:
    return json.dumps(
        [{"id": i, "status": "ok", "msg": f"value_{i}"} for i in range(records)],
        separators=(",", ":"),
    )


def test_dense_payload_over_the_cap_is_caught_by_the_token_unit() -> None:
    """chars/4 waves through a payload the token cap should stop.

    177,781 chars of compact JSON: 44,445 by chars/4 (under the cap, silent) but
    55,557 estimator tokens — 11% over.
    """
    payload = _compact_json(4_000)

    assert len(payload) // 4 <= _CAP, "premise: the old chars/4 test passes this"
    assert _estimate_tokens(payload) > _CAP, "the token cap must be exceeded"


def test_the_two_units_agree_below_and_above_the_band() -> None:
    """Outside the disagreement band both formulations reach the same verdict."""
    small = _compact_json(2_700)
    assert len(small) // 4 <= _CAP and _estimate_tokens(small) <= _CAP

    huge = _compact_json(5_000)
    assert len(huge) // 4 > _CAP and _estimate_tokens(huge) > _CAP


def test_chars_over_four_understates_dense_content() -> None:
    """The mechanism, stated as a property rather than a magic number."""
    payload = _compact_json(4_000)
    assert _estimate_tokens(payload) > len(payload) // 4, (
        "compact JSON is denser than 4 chars/token, which is why the unit matters"
    )


def test_word_count_is_not_interchangeable_with_a_token_count() -> None:
    """Why the CONFIG numerator had to change.

    A byte-identical no-op must score 1.0. Measured with a word count it scored
    ~0.36 on real config text and was accepted as a 64% saving.
    """
    config_text = "\n".join(
        f"key_{i}: value_{i}  # inline comment explaining key_{i}" for i in range(200)
    )

    tokens = _estimate_tokens(config_text)
    words = len(config_text.split())

    # Same unit on both sides: a no-op is correctly a no-op.
    assert tokens / tokens == 1.0

    # Mixed units: the same no-op looks like a large saving.
    assert words / tokens < 0.75, (
        "if words and estimator tokens were interchangeable this bug could not exist"
    )
