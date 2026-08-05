"""CJK-aware relevance-query matching in the code compressor.

The symbol-importance context boost tokenized the query with an ASCII-only
delimiter class, so a CJK query (no spaces, CJK punctuation) collapsed into one
blob and never isolated/matched an ASCII symbol name the user asked to keep.
These exercise the extracted pure helpers (no tree-sitter needed).
"""

from headroom.transforms.code_compressor import (
    _query_context_tokens,
    _symbol_in_context,
)


def test_cjk_query_isolates_wrapped_ascii_symbol():
    # full-width parens around the name must still tokenize parse_config out
    words, lowered, has_cjk = _query_context_tokens("请重点保留（parse_config）的解析配置")
    assert has_cjk
    assert "parse_config" in words
    assert _symbol_in_context("parse_config", words, lowered, has_cjk)


def test_cjk_query_matches_short_ascii_name_glued_to_cjk():
    # 'db' (len 2) glued to CJK has no delimiter to isolate it; the len>3 guard is
    # relaxed for CJK so the substring fallback still matches.
    words, lowered, has_cjk = _query_context_tokens("请保留db相关的逻辑")
    assert has_cjk
    assert _symbol_in_context("db", words, lowered, has_cjk)


def test_english_short_name_substring_still_gated():
    # ASCII query unchanged: a short name that is only a substring (not a token)
    # of an English query must NOT match (avoids spurious boosts).
    words, lowered, has_cjk = _query_context_tokens("keep the database helper")
    assert not has_cjk
    assert not _symbol_in_context("db", words, lowered, has_cjk)


def test_english_exact_token_match_unchanged():
    words, lowered, has_cjk = _query_context_tokens("keep parse_config and helper")
    assert not has_cjk
    assert _symbol_in_context("parse_config", words, lowered, has_cjk)
    assert _symbol_in_context("helper", words, lowered, has_cjk)


def test_english_long_name_substring_fallback_unchanged():
    # ASCII path, len>3 substring fallback: 'parse_config' is not a standalone
    # token but is a substring of 'parse_configs' -> must still match (unchanged).
    words, lowered, has_cjk = _query_context_tokens("parse_configs and related helpers")
    assert not has_cjk
    assert "parse_config" not in words
    assert _symbol_in_context("parse_config", words, lowered, has_cjk)


def test_empty_context_matches_nothing():
    words, lowered, has_cjk = _query_context_tokens("")
    assert words == set() and lowered == "" and has_cjk is False
    assert not _symbol_in_context("foo", words, lowered, has_cjk)
