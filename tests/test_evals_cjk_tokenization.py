"""CJK correctness for the evals metric layer.

The shared eval metrics assumed ASCII: `tokenize` used `\\b\\w+\\b` (a space-free
CJK string collapses to ONE token, so token-F1 is all-or-nothing), and
`_estimate_tokens` used `len(text)//4` (CJK is ~1-2 tokens/char, not 0.25), so
CJK compression savings were reported ~4-6x wrong.
"""

from headroom.evals.core import CompressionEvaluator
from headroom.evals.metrics import compute_f1, tokenize


def test_tokenize_splits_cjk_into_units():
    toks = tokenize("数据库连接失败")
    assert len(toks) >= 3, f"CJK must split into multiple units, got {toks}"


def test_tokenize_ascii_unchanged():
    assert tokenize("Hello, World 42") == ["hello", "world", "42"]


def test_f1_partial_credit_on_overlapping_cjk():
    # two CJK strings that share most characters must score strictly between 0 and 1
    f1 = compute_f1("数据库连接失败", "数据库连接成功")
    assert 0.0 < f1 < 1.0, f"overlapping CJK should be partial credit, got {f1}"


def test_estimate_tokens_cjk_not_underestimated():
    # 20 CJK chars: len//4 gives 5; CJK-aware should be >= ~13
    assert CompressionEvaluator._estimate_tokens(None, "数" * 20) >= 13


def test_estimate_tokens_ascii_unchanged():
    assert CompressionEvaluator._estimate_tokens(None, "x" * 40) == 10
