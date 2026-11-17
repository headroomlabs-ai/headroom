"""CJK-aware relevance scoring in the search compressor.

The relevance scorer tokenized the query on whitespace, so a spaceless CJK query
matched content only when the WHOLE query was a literal substring of a line. CJK
char bigrams now let a longer CJK query boost lines that share a substring.
Bigram extraction itself is unit-tested in Rust (``cjk_bigrams_from_runs``);
this pins the end-to-end ranking through ``compress()``.
"""

from headroom.transforms.search_compressor import (
    SearchCompressor,
    SearchCompressorConfig,
)


def test_cjk_query_bigrams_boost_ranking():
    # Keep only the top-scored match per file, so the output shows the ranking.
    compressor = SearchCompressor(
        SearchCompressorConfig(
            max_matches_per_file=1,
            always_keep_first=False,
            always_keep_last=False,
            boost_errors=False,
            context_keywords=[],
            enable_ccr=False,
        )
    )
    content = "\n".join(
        [
            "src/a.py:10:plain ascii content here",
            "src/a.py:20:认证令牌已过期需要重新登录",
            "src/a.py:30:more plain content",
        ]
    )
    # The whole query is NOT a substring of the CJK line, but its bigrams are.
    result = compressor.compress(content, context="认证令牌缓存淘汰策略")
    assert result.compressed.startswith("src/a.py:20:认证令牌已过期需要重新登录")
    # Without a matching query the earliest line wins the tie.
    baseline = compressor.compress(content, context="缓存淘汰策略")
    assert baseline.compressed.startswith("src/a.py:10:plain ascii content here")
