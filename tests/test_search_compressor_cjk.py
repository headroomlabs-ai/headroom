"""CJK-aware relevance scoring in the search compressor.

The relevance scorer tokenized the query on whitespace, so a spaceless CJK query
matched content only when the WHOLE query was a literal substring of a line. CJK
char bigrams now let a longer CJK query boost lines that share a substring. The
Rust<->Python parity was also hardened (dedup like Python's set; char-length
filter instead of bytes). These exercise the Python legacy scorer that mirrors
Rust.
"""

from headroom.transforms.search_compressor import (
    SearchCompressor,
    SearchCompressorConfig,
    _cjk_bigrams,
)


def test_cjk_bigrams_from_runs():
    assert _cjk_bigrams("认证令牌") == {"认证", "证令", "令牌"}
    assert _cjk_bigrams("hello world") == set()  # ASCII -> no CJK bigrams
    assert _cjk_bigrams("a认b证") == set()  # isolated CJK chars -> no adjacent pair


def test_score_matches_cjk_query_bigrams_boost():
    compressor = SearchCompressor(SearchCompressorConfig(boost_errors=False, context_keywords=[]))
    content = "\n".join(
        [
            "src/a.py:10:认证令牌已过期需要重新登录",
            "src/b.py:2:plain ascii content here",
        ]
    )
    parsed = compressor._parse_search_results(content)
    # the whole query is NOT a substring of the content line, but its bigrams are
    compressor._score_matches(parsed, "认证令牌缓存淘汰策略")
    assert parsed["src/a.py"].matches[0].score > 0  # 认证/证令/令牌 bigrams match
    assert parsed["src/b.py"].matches[0].score == 0
