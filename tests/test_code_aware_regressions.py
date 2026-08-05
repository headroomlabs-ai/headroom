"""Regression tests requested in review of PR #1668 (Go AST compression bugs
and CODE_AWARE token accounting) and PR #1670 (prefer_code_aware_for_code
default flip).

1. Go `statement_list` unwrapping — a Go function body no longer produces a
   duplicated closing brace when truncated.
2. Multi-line Go signature (`) error {`) — the brace-bearing signature line
   survives truncation instead of being silently dropped.
3. ContentRouter CODE_AWARE token accounting — compressed_tokens must come
   from the same word-count metric as original_tokens, not the compressor's
   own (differently-scaled) estimator, or real savings get misread as none.
4. prefer_code_aware_for_code defaults to True, both on the dataclass and via
   the HEADROOM_PREFER_CODE_AWARE_FOR_CODE env var.
"""

from __future__ import annotations

import pytest

from headroom.proxy.server import HeadroomProxy, ProxyConfig
from headroom.transforms.code_compressor import (
    CodeAwareCompressor,
    CodeCompressorConfig,
    _check_tree_sitter_available,
)
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)

pytestmark = pytest.mark.skipif(
    not _check_tree_sitter_available(),
    reason="tree-sitter not installed (pip install headroom-ai[code])",
)

GO_FUNC = """package main

func Compute(
\ta int,
\tb int,
) error {
\tx := a + b
\ty := x * 2
\tz := y - a
\tw := z + b
\treturn nil
}
"""


def _compress_go(**config_overrides: object):
    config = CodeCompressorConfig(
        min_tokens_for_compression=0,
        max_body_lines=2,
        semantic_analysis=False,
        **config_overrides,
    )
    compressor = CodeAwareCompressor(config)
    return compressor.compress(GO_FUNC, language="go")


def test_go_statement_list_unwrap_no_duplicate_closing_brace() -> None:
    """Go wraps a block's statements in one `statement_list` node; treating
    that as a single statement made its row range swallow the block's own
    closing brace, producing a duplicate `}` in the compressed output."""
    result = _compress_go()

    closing_braces = [line for line in result.compressed.splitlines() if line.strip() == "}"]
    assert len(closing_braces) == 1, (
        f"expected exactly one closing brace line, got {len(closing_braces)}:\n{result.compressed}"
    )


def test_go_multiline_signature_brace_preserved() -> None:
    """The multi-line signature's closing line (`) error {`) shares its row
    with the brace instead of starting one of its own. Detecting the brace
    via startswith("{") missed this and silently dropped the line; it must
    survive truncation via endswith("{")."""
    result = _compress_go()

    assert result.compressed.count(") error {") == 1, (
        f"multi-line signature line missing or duplicated:\n{result.compressed}"
    )


class _FakeCodeCompressor:
    def __init__(self, compressed: str) -> None:
        self._compressed = compressed

    def compress(self, content: str, language=None, context=""):
        class _Result:
            pass

        r = _Result()
        r.compressed = self._compressed
        # Deliberately inflated/differently-scaled "own" token estimate —
        # simulates a compressor whose internal counter isn't comparable to
        # the router's len(text.split()) word count.
        r.compressed_tokens = 10_000_000
        return r


def test_content_router_code_aware_token_accounting_matches_word_count() -> None:
    """compressed_tokens must be derived from len(result.compressed.split()),
    the same metric as original_tokens, not the compressor's own estimator.
    Using the mismatched estimator made a real compression look like "no
    savings" and forced a needless fallback to Kompress."""
    original = "word " * 200  # 200 words
    compressed_text = "word " * 50  # a real, large reduction

    router = ContentRouter(ContentRouterConfig(enable_code_aware=True))
    router._code_compressor = _FakeCodeCompressor(compressed_text)

    compressed, compressed_tokens, strategy_chain = router._apply_strategy_to_content(
        original, CompressionStrategy.CODE_AWARE, context=""
    )

    assert compressed == compressed_text
    assert compressed_tokens == len(compressed_text.split())
    assert strategy_chain == ["code_aware"], (
        f"real compression must not trigger a Kompress fallback, got {strategy_chain}"
    )


def test_prefer_code_aware_for_code_dataclass_defaults_true() -> None:
    assert ContentRouterConfig().prefer_code_aware_for_code is True


def test_prefer_code_aware_for_code_env_var_defaults_true(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_PREFER_CODE_AWARE_FOR_CODE", raising=False)

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        code_aware_enabled=False,
    )
    proxy = HeadroomProxy(config)
    router = proxy.anthropic_pipeline.transforms[-1]

    assert router.config.prefer_code_aware_for_code is True
