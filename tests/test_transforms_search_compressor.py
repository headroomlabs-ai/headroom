from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.transforms.search_compressor import (
    FileMatches,
    SearchCompressionResult,
    SearchCompressor,
    SearchCompressorConfig,
    SearchMatch,
)


def test_parse_score_select_and_format_search_results(monkeypatch: pytest.MonkeyPatch) -> None:
    compressor = SearchCompressor(
        SearchCompressorConfig(
            max_matches_per_file=3,
            max_total_matches=4,
            max_files=2,
            context_keywords=["auth"],
        )
    )
    content = "\n".join(
        [
            "src/auth.py:10:ERROR auth failed",
            "src/auth.py-11-warning auth retry",
            "src/auth.py:12:plain auth line",
            "src/db.py:2:warning token expired",
            "not a match",
        ]
    )
    parsed = compressor._parse_search_results(content)
    assert set(parsed) == {"src/auth.py", "src/db.py"}
    assert parsed["src/auth.py"].first == SearchMatch(
        file="src/auth.py", line_number=10, content="ERROR auth failed"
    )
    assert parsed["src/auth.py"].last.line_number == 12

    compressor._score_matches(parsed, "find auth error")
    assert parsed["src/auth.py"].matches[0].score == 1.0
    assert parsed["src/db.py"].matches[0].score > 0

    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.transforms.adaptive_sizer",
        SimpleNamespace(compute_optimal_k=lambda items, **kwargs: 4),
    )
    selected = compressor._select_matches(parsed, bias=1.2)
    assert list(selected) == ["src/auth.py", "src/db.py"]
    assert [m.line_number for m in selected["src/auth.py"].matches] == [10, 11, 12]

    formatted, summaries = compressor._format_output(
        selected,
        {
            **parsed,
            "src/db.py": FileMatches(
                file="src/db.py",
                matches=[
                    SearchMatch(file="src/db.py", line_number=2, content="warning token expired"),
                    SearchMatch(file="src/db.py", line_number=3, content="another line"),
                ],
            ),
        },
    )
    assert "src/auth.py:10:ERROR auth failed" in formatted
    assert summaries["src/db.py"] == "[... and 1 more matches in src/db.py]"


def test_search_compressor_compress_paths_and_ccr() -> None:
    """Phase 3e.2: `compress()` is now a single Rust call, so this test
    exercises end-to-end behavior instead of monkeypatching internal
    helpers (which the old orchestration relied on). The CCR plumbing
    is verified via `cache_key` presence + the marker string format
    Rust emits."""
    compressor = SearchCompressor(
        SearchCompressorConfig(enable_ccr=True, min_matches_for_ccr=2, context_keywords=["auth"])
    )
    no_match = compressor.compress("plain text only")
    assert no_match.original_match_count == 0
    assert no_match.compressed == "plain text only"

    # Build a large input so compute_optimal_k's min_k=5 floor doesn't
    # absorb everything and compression actually fires (must drop the
    # ratio below `min_compression_ratio_for_ccr=0.8`).
    lines = [f"src/auth.py:{i}:auth event {i}" for i in range(1, 51)]
    lines += [f"src/db.py:{i}:db query {i}" for i in range(1, 31)]
    content = "\n".join(lines)
    result = compressor.compress(content, context="auth", bias=0.5)  # low bias = drop more
    assert result.original_match_count == 80
    assert result.files_affected == 2
    assert result.compressed_match_count < result.original_match_count
    assert result.cache_key is not None
    assert result.compressed.endswith(f". Retrieve more: hash={result.cache_key}]")
    # Summaries appear for any file whose matches were dropped.
    assert isinstance(result.summaries, dict)
    assert len(result.summaries) >= 1


def test_search_compressor_persist_to_python_ccr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 3e.2: CCR persistence is now in `_persist_to_python_ccr`,
    which delegates to the production `CompressionStore`. Failures are
    logged (not silently swallowed) — this pins both paths."""
    compressor = SearchCompressor()

    seen: dict[str, tuple[str, str, str | None]] = {}
    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.cache.compression_store",
        SimpleNamespace(
            get_compression_store=lambda: SimpleNamespace(
                store=lambda original, compressed, original_item_count=0, explicit_hash=None: (
                    seen.setdefault("call", (original, compressed, explicit_hash)) or "stored-key"
                )
            )
        ),
    )
    compressor._persist_to_python_ccr("orig", "comp", "abc123")
    # explicit_hash carries the Rust marker key so retrieval of the
    # marker hash finds the entry (issue #816).
    assert seen["call"] == ("orig", "comp", "abc123")

    # Loud failure: the store raises, but persist swallows + logs (no
    # exception propagates to the compress callsite).
    def broken_store() -> SimpleNamespace:
        raise RuntimeError("boom")

    monkeypatch.setitem(
        __import__("sys").modules,
        "headroom.cache.compression_store",
        SimpleNamespace(get_compression_store=broken_store),
    )
    compressor._persist_to_python_ccr("orig", "comp", "abc123")  # must not raise


def test_search_compression_result_properties() -> None:
    """Result-property contract preserved across the port."""
    result = SearchCompressionResult(
        compressed="tiny",
        original="this is a much longer original string",
        original_match_count=10,
        compressed_match_count=4,
        files_affected=2,
        compression_ratio=0.3,
    )
    assert result.tokens_saved_estimate > 0
    assert result.matches_omitted == 6
