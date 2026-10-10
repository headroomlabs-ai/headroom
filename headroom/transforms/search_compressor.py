"""Rust-backed search-results compressor.

Phase 3e.2 ported the implementation to
`crates/headroom-core/src/transforms/search_compressor.rs`. This module
is now a thin shim that:

1. Keeps the public dataclass surface (`SearchMatch`, `FileMatches`,
   `SearchCompressorConfig`, `SearchCompressionResult`) so existing
   call sites (`ContentRouter._get_search_compressor`) and tests don't
   change.
2. Routes `SearchCompressor.compress()` entirely through the Rust
   implementation, picking up the parser bug fixes and the
   `signals::LineImportanceDetector` trait consumer pattern.

# Bug fixes the Rust port carries (and this shim therefore inherits)

* **Windows paths.** Pre-3e.2 `_GREP_PATTERN`/`_RG_CONTEXT_PATTERN`
  regexes treated the drive-letter colon (`C:\\Users\\…`) as the
  line-number-marker separator and silently dropped every Windows-
  formatted line from `file_matches`. The Rust parser detects the
  drive prefix and starts the line-number scan after it.
* **Filenames with `-`.** Pre-3e.2 `_RG_CONTEXT_PATTERN` excluded
  dashes from the path (`[^:-]+`), so legitimate names like
  `pre-commit-config.yaml-42-line` parsed wrong. The Rust parser
  anchors on the *line-number marker* — earliest `<sep>\\d+<sep>` in
  the line — so paths can contain dashes.
* **CCR storage failures are loud.** The previous Python class
  swallowed all exceptions from the compression store. Storage
  failures now surface to logs.

# CCR plumbing note

The Rust crate carries an internal CCR store for unit testing, but
the production CCR path remains the Python `CompressionStore`. The
shim picks up the Rust-emitted `cache_key` and writes the original
through to the Python store, so retrievability semantics match
exactly what the previous Python implementation provided.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

logger = logging.getLogger(__name__)


# ─── Public dataclasses (preserve existing import surface) ──────────────────


@dataclass
class SearchMatch:
    """A single search match."""

    file: str
    line_number: int
    content: str
    score: float = 0.0


@dataclass
class FileMatches:
    """All matches in a single file."""

    file: str
    matches: list[SearchMatch] = field(default_factory=list)

    @property
    def first(self) -> SearchMatch | None:
        return self.matches[0] if self.matches else None

    @property
    def last(self) -> SearchMatch | None:
        return self.matches[-1] if self.matches else None


@dataclass
class SearchCompressorConfig:
    """Configuration for search result compression."""

    max_matches_per_file: int = 5
    always_keep_first: bool = True
    always_keep_last: bool = True
    max_total_matches: int = 30
    max_files: int = 15
    context_keywords: list[str] = field(default_factory=list)
    boost_errors: bool = True
    enable_ccr: bool = True
    min_matches_for_ccr: int = 10
    # Group output by file (`rg --heading` style): path emitted once per
    # file, then `line:content` rows. Removes per-match path repetition.
    # Default False (classic `file:line:content`); the proxy enables it
    # in token mode.
    group_by_file: bool = False


@dataclass
class SearchCompressionResult:
    """Result of search result compression."""

    compressed: str
    original: str
    original_match_count: int
    compressed_match_count: int
    files_affected: int
    compression_ratio: float
    cache_key: str | None = None
    summaries: dict[str, str] = field(default_factory=dict)

    @property
    def tokens_saved_estimate(self) -> int:
        """Estimate tokens saved (rough: 1 token per 4 chars)."""
        chars_saved = len(self.original) - len(self.compressed)
        return max(0, chars_saved // 4)

    @property
    def matches_omitted(self) -> int:
        return self.original_match_count - self.compressed_match_count


# ─── Compressor (Rust-backed) ───────────────────────────────────────────────


class SearchCompressor:
    """Compresses grep/ripgrep search results via the Rust port.

    Drop-in replacement for the retired Python class. `compress()`
    delegates to Rust end-to-end.
    """

    def __init__(self, config: SearchCompressorConfig | None = None) -> None:
        # Hard import — no fallback. If the wheel is missing, the user
        # must build it (scripts/build_rust_extension.sh) or install a
        # prebuilt one. Failing loudly here is better than silently
        # degrading; see feedback memory `feedback_no_silent_fallbacks.md`.
        from headroom._core import (
            SearchCompressor as _RustSearchCompressor,
        )
        from headroom._core import (
            SearchCompressorConfig as _RustSearchCompressorConfig,
        )

        cfg = config or SearchCompressorConfig()
        self.config = cfg
        # `min_compression_ratio_for_ccr` was inlined as 0.8 in the
        # Python original; promoted to a config field on the Rust side
        # but defaulted to 0.8 here so the existing Python config
        # surface is unchanged.
        self._rust = _RustSearchCompressor(
            _RustSearchCompressorConfig(
                max_matches_per_file=cfg.max_matches_per_file,
                always_keep_first=cfg.always_keep_first,
                always_keep_last=cfg.always_keep_last,
                max_total_matches=cfg.max_total_matches,
                max_files=cfg.max_files,
                context_keywords=list(cfg.context_keywords),
                boost_errors=cfg.boost_errors,
                enable_ccr=cfg.enable_ccr,
                min_matches_for_ccr=cfg.min_matches_for_ccr,
                min_compression_ratio_for_ccr=0.8,
                group_by_file=getattr(cfg, "group_by_file", False),
            )
        )

    # ─── Public API ─────────────────────────────────────────────────────

    def compress(
        self,
        content: str,
        context: str = "",
        bias: float = 1.0,
    ) -> SearchCompressionResult:
        rust_result = self._rust.compress(content, context, bias)
        cache_key: str | None = rust_result.cache_key
        if cache_key is not None:
            # Mirror the original Python: persist to the production CCR
            # store. The Rust crate already wrote to its in-memory test
            # store; promote that to the long-lived Python store so the
            # marker remains retrievable beyond the request lifecycle.
            self._persist_to_python_ccr(content, rust_result.compressed, cache_key)

        summaries = dict(cast("dict[str, str]", rust_result.summaries))
        return SearchCompressionResult(
            compressed=rust_result.compressed,
            original=content,
            original_match_count=rust_result.original_match_count,
            compressed_match_count=rust_result.compressed_match_count,
            files_affected=rust_result.files_affected,
            compression_ratio=rust_result.compression_ratio,
            cache_key=cache_key,
            summaries=summaries,
        )

    # ─── Internal CCR persistence ───────────────────────────────────────

    def _persist_to_python_ccr(self, original: str, compressed: str, cache_key: str) -> None:
        """Promote the Rust-emitted cache_key into the production Python
        `CompressionStore`. Failures are surfaced via logging instead of
        being silently swallowed (see no-silent-fallbacks rule).

        Note: the Rust path computes the hash and embeds it in the
        emitted marker text — the Rust hash IS the canonical one
        (MD5(original)[:24]). The store must be keyed by that exact
        hash or the marker dangles.
        """
        try:
            from ..cache.compression_store import get_compression_store
        except ImportError as e:
            logger.warning("CCR store import failed; cache_key %s won't persist: %s", cache_key, e)
            return

        try:
            store: Any = get_compression_store()
            # The Rust-emitted marker embeds MD5(original)[:24], but
            # store() has defaulted to SHA-256(original)[:24] since
            # PR #395. Pass the marker's key explicitly so retrieving
            # the marker hash actually finds the entry (issue #816).
            store.store(original, compressed, explicit_hash=cache_key)
        except Exception as e:
            logger.warning(
                "CCR store write failed; cache_key %s remains in-marker only: %s", cache_key, e
            )


__all__ = [
    "SearchCompressor",
    "SearchCompressorConfig",
    "SearchCompressionResult",
    "SearchMatch",
    "FileMatches",
]
