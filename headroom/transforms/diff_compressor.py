"""Git diff output compressor — Rust-backed via PyO3.

The Python implementation has been retired (Stage 3b, 2026-04-25). All
diff compression now goes through `headroom._core.DiffCompressor` (built
from `crates/headroom-py`). The byte-equality of the two implementations
was verified against 27 recorded fixtures before the Python source was
removed; the Rust crate has its own test coverage in `crates/headroom-core/`.

This module retains the public surface — `DiffCompressorConfig`,
`DiffCompressionResult`, `DiffCompressor` — so existing call sites
(ContentRouter, parity recorder, integrations, downstream users) keep
working unchanged. The dataclasses are still pure-Python because they
appear in dataclass-aware code paths (`asdict()`, `__dict__`, dataclass
matching). Only the `DiffCompressor` class delegates to Rust.

The `headroom._core` extension is a hard import: there is no Python
fallback. Build it locally with `scripts/build_rust_extension.sh`
(wraps `maturin develop`) or install a prebuilt wheel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DiffCompressorConfig:
    """Configuration for diff compression."""

    max_context_lines: int = 2
    max_hunks_per_file: int = 10
    max_files: int = 20
    always_keep_additions: bool = True
    always_keep_deletions: bool = True
    enable_ccr: bool = True
    min_lines_for_ccr: int = 50


@dataclass
class DiffCompressionResult:
    """Result of diff compression."""

    compressed: str
    original_line_count: int
    compressed_line_count: int
    files_affected: int
    additions: int
    deletions: int
    hunks_kept: int
    hunks_removed: int
    cache_key: str | None = None

    @property
    def compression_ratio(self) -> float:
        if self.original_line_count == 0:
            return 1.0
        return self.compressed_line_count / self.original_line_count

    @property
    def tokens_saved_estimate(self) -> int:
        lines_saved = self.original_line_count - self.compressed_line_count
        chars_saved = lines_saved * 40
        return max(0, chars_saved // 4)


class DiffCompressor:
    """Rust-backed `DiffCompressor` (via PyO3 / `headroom._core`).

    Same `__init__` and `compress` shape as the retired Python class —
    drop-in replacement. Returns Python `DiffCompressionResult` dataclass
    instances so call sites that destructure with `asdict()` or read the
    `@property` fields work unchanged.
    """

    def __init__(self, config: DiffCompressorConfig | None = None):
        # Hard import — no fallback. If the wheel is missing, the user
        # must build it (scripts/build_rust_extension.sh) or install a
        # prebuilt one. Failing loudly here is better than silently
        # degrading; see feedback memory `feedback_no_silent_fallbacks.md`.
        from headroom._core import (
            DiffCompressor as _RustDiffCompressor,
        )
        from headroom._core import (
            DiffCompressorConfig as _RustDiffCompressorConfig,
        )

        cfg = config or DiffCompressorConfig()
        self.config = cfg
        self._rust = _RustDiffCompressor(
            _RustDiffCompressorConfig(
                max_context_lines=cfg.max_context_lines,
                max_hunks_per_file=cfg.max_hunks_per_file,
                max_files=cfg.max_files,
                always_keep_additions=cfg.always_keep_additions,
                always_keep_deletions=cfg.always_keep_deletions,
                enable_ccr=cfg.enable_ccr,
                min_lines_for_ccr=cfg.min_lines_for_ccr,
            )
        )

    def compress(self, content: str, context: str = "") -> DiffCompressionResult:
        r = self._rust.compress(content, context)
        cache_key: str | None = r.cache_key
        if cache_key is not None:
            # Mirror log_compressor.py + search_compressor.py: when the
            # Rust path emits a CCR retrieval marker, persist the
            # original payload to Python's CompressionStore so the
            # marker actually resolves on the LLM's retrieval tool
            # call. Without this, every diff CCR marker emitted in
            # production is dangling — the regression fixed in the
            # audit-cleanup PR.
            self._persist_to_python_ccr(content, r.compressed, cache_key)
        return DiffCompressionResult(
            compressed=r.compressed,
            original_line_count=r.original_line_count,
            compressed_line_count=r.compressed_line_count,
            files_affected=r.files_affected,
            additions=r.additions,
            deletions=r.deletions,
            hunks_kept=r.hunks_kept,
            hunks_removed=r.hunks_removed,
            cache_key=cache_key,
        )

    def _persist_to_python_ccr(self, original: str, compressed: str, cache_key: str) -> None:
        """Promote a Rust-emitted cache_key into the production Python
        CompressionStore. Failures are logged at warning level — a
        store hiccup must not break the response, just degrade
        retrieval. Mirrors the same helper on log_compressor.py and
        search_compressor.py."""
        try:
            from ..cache.compression_store import get_compression_store
        except ImportError as e:
            logger.warning("CCR store import failed; cache_key %s won't persist: %s", cache_key, e)
            return
        try:
            store: Any = get_compression_store()
            store.store(original, compressed)
        except Exception as e:
            logger.warning(
                "CCR store write failed; cache_key %s remains in-marker only: %s",
                cache_key,
                e,
            )

    def compress_with_stats(
        self, content: str, context: str = ""
    ) -> tuple[DiffCompressionResult, Any]:
        """Sidecar API exposing the Rust-only `DiffCompressorStats` struct
        (per-file hunk drops, context lines trimmed, file_mode normalizations,
        etc.) alongside the result. Stats is the raw PyO3 wrapper — no
        Python equivalent to mirror to. Typed as `Any` because the PyO3
        class has no Python type stub.
        """
        r, stats = self._rust.compress_with_stats(content, context)
        result = DiffCompressionResult(
            compressed=r.compressed,
            original_line_count=r.original_line_count,
            compressed_line_count=r.compressed_line_count,
            files_affected=r.files_affected,
            additions=r.additions,
            deletions=r.deletions,
            hunks_kept=r.hunks_kept,
            hunks_removed=r.hunks_removed,
            cache_key=r.cache_key,
        )
        return result, stats
