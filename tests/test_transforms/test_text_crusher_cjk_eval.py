"""CI regression gate for CJK (zh/ja/ko) compression answer-retention.

Deterministic: a query-relevant needle buried among distractors must survive
query-aware compression (TextCrusher) and must do at least as well as the
keep-recent / random baselines. Guards the #1171/#1504 CJK TextCrusher path.
"""

import os
import sys

import pytest

# benchmarks/ is not a package on the import path by default; add the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarks.i18n_compression_eval import retention_synthetic  # noqa: E402


@pytest.mark.parametrize("lang", ["zh", "ja", "ko"])
def test_cjk_needle_survives_compression(lang: str) -> None:
    r = retention_synthetic(lang, ratio=0.3, seed=0)
    assert r["text_crusher"], f"{lang}: query-relevant needle dropped by TextCrusher"


@pytest.mark.parametrize("lang", ["zh", "ja", "ko"])
def test_text_crusher_beats_or_ties_baselines(lang: str) -> None:
    r = retention_synthetic(lang, ratio=0.3, seed=0)
    assert r["text_crusher"] >= max(r["truncate"], r["random"])
