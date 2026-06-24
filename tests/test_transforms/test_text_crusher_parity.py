"""TextCrusher parity (Phase 2, #1171): the native Rust core must keep
reproducing the recorded ``compress`` output for every fixture. Catches drift
in the Rust algorithm. Re-record intentional changes with:
    python tests/parity/record_text_crusher.py
"""

from __future__ import annotations

import glob
import json
import os

import pytest

from headroom.transforms.text_crusher import TextCrusher

_FIXTURE_GLOB = os.path.join(
    os.path.dirname(__file__), "..", "parity", "fixtures", "text_crusher", "*.json"
)
FIXTURES = sorted(glob.glob(_FIXTURE_GLOB))


def test_fixtures_exist():
    assert FIXTURES, "no text_crusher parity fixtures; run tests/parity/record_text_crusher.py"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: os.path.basename(p))
def test_text_crusher_matches_recorded(path):
    with open(path, encoding="utf-8") as fh:
        fx = json.load(fh)
    inp = fx["input"]
    r = TextCrusher().compress(inp["content"], inp["context"], inp["target_ratio"])
    exp = fx["output"]
    assert r.compressed == exp["compressed"]
    assert r.compressed_tokens == exp["compressed_tokens"]
    assert r.kept_segments == exp["kept_segments"]
    assert r.total_segments == exp["total_segments"]
    assert abs(r.compression_ratio - exp["compression_ratio"]) < 1e-9
