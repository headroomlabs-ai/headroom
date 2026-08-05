"""Rust-backed log/build-output compressor.

Phase 3e.5 ported the implementation to
`crates/headroom-core/src/transforms/log_compressor.rs`. This module
is now a thin shim that:

1. Keeps the public dataclass and enum surface (`LogLevel`,
   `LogFormat`, `LogLine`, `LogCompressorConfig`,
   `LogCompressionResult`) so existing call sites (`ContentRouter`,
   tests) don't change.
2. Routes `LogCompressor.compress()` entirely through the Rust
   implementation, picking up the bug fixes (chained-exception trace
   survival, conservative warning dedupe, loud CCR failures).
3. Implements legacy internal helpers (`_detect_format`, `_parse_lines`,
   `_score_line`, `_select_lines`, `_select_with_first_last`,
   `_dedupe_similar`, `_format_output`) on top of the Rust building
   blocks where a Rust delegation makes sense; otherwise keeps Python
   logic that mirrors Rust scoring.

# Bug fixes the Rust port carries (and this shim therefore inherits)

* **Stack-trace state machine.** Pre-3e.5 Python terminated on any
  blank line, dropping mid-trace lines from chained-exception traces.
  Rust dispatches per language flavor so blank lines stay inside
  Python tracebacks.
* **Conservative dedupe.** Pre-3e.5 normalised digits/paths/hex
  globally, collapsing distinct error categories that shared a
  trailing variable shape. Rust splits on the first `:`/`=` and only
  normalises the trailing region — message identifiers stay distinct.
* **Loud CCR failures.** Storage failures are logged at warning level
  instead of being silently swallowed.
* **`LogLevel.FAIL` is documented as cosmetic-equivalent to
  `LogLevel.ERROR`.** Both score 1.0 in Python and Rust.

# CCR plumbing note

Same pattern as search_compressor: Rust emits a `cache_key`, the
Python shim writes the original to the production
`CompressionStore`. The Rust crate's CCR store is in-memory and
exists only for unit testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

logger = logging.getLogger(__name__)


class LogFormat(Enum):
    """Detected log format."""

    PYTEST = "pytest"
    NPM = "npm"
    CARGO = "cargo"
    MAKE = "make"
    JEST = "jest"
    GENERIC = "generic"


class LogLevel(Enum):
    """Log level for categorization."""

    ERROR = "error"
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"
    DEBUG = "debug"
    TRACE = "trace"
    UNKNOWN = "unknown"


@dataclass(eq=False)
class LogLine:
    """A single log line with metadata."""

    line_number: int
    content: str
    level: LogLevel = LogLevel.UNKNOWN
    is_stack_trace: bool = False
    is_summary: bool = False
    score: float = 0.0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogLine):
            return NotImplemented
        return self.line_number == other.line_number

    def __hash__(self) -> int:
        return hash(self.line_number)


@dataclass
class LogCompressorConfig:
    """Configuration for log compression."""

    max_errors: int = 10
    error_context_lines: int = 3
    keep_first_error: bool = True
    keep_last_error: bool = True
    max_stack_traces: int = 3
    stack_trace_max_lines: int = 20
    max_warnings: int = 5
    dedupe_warnings: bool = True
    keep_summary_lines: bool = True
    max_total_lines: int = 100
    enable_ccr: bool = True
    min_lines_for_ccr: int = 50
    # Frame collapse: when a trace exceeds stack_trace_max_lines, keep the
    # message/chain-head lines, the first trace_head_frames frames, and up to
    # trace_app_frames app-code frames; runtime/stdlib frames collapse into a
    # `[... N frames collapsed]` marker instead of blind tail-truncation.
    collapse_runtime_frames: bool = True
    trace_head_frames: int = 3
    trace_app_frames: int = 5


@dataclass
class LogCompressionResult:
    """Result of log compression."""

    compressed: str
    original: str
    original_line_count: int
    compressed_line_count: int
    format_detected: LogFormat
    compression_ratio: float
    cache_key: str | None = None
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def tokens_saved_estimate(self) -> int:
        chars_saved = len(self.original) - len(self.compressed)
        return max(0, chars_saved // 4)

    @property
    def lines_omitted(self) -> int:
        return self.original_line_count - self.compressed_line_count


# ─── LogCompressor (Rust-backed) ────────────────────────────────────────────


def _format_from_str(name: str) -> LogFormat:
    return {
        "pytest": LogFormat.PYTEST,
        "npm": LogFormat.NPM,
        "cargo": LogFormat.CARGO,
        "make": LogFormat.MAKE,
        "jest": LogFormat.JEST,
    }.get(name, LogFormat.GENERIC)


class LogCompressor:
    """Rust-backed log compressor.

    Drop-in replacement for the retired Python class. `compress()`
    delegates to Rust end-to-end; internal helpers used by the
    existing test surface keep working but route through the same
    Rust building blocks where they exist.
    """

    def __init__(self, config: LogCompressorConfig | None = None) -> None:
        # Hard import — no fallback. If the wheel is missing, the user
        # must build it. See feedback memory `feedback_no_silent_fallbacks.md`.
        from headroom._core import (
            LogCompressor as _RustLogCompressor,
        )
        from headroom._core import (
            LogCompressorConfig as _RustLogCompressorConfig,
        )

        cfg = config or LogCompressorConfig()
        self.config = cfg
        # `min_compression_ratio_for_ccr` was inlined as 0.5 in Python;
        # the Rust port promoted it to a config field but defaults
        # match.
        self._rust = _RustLogCompressor(
            _RustLogCompressorConfig(
                max_errors=cfg.max_errors,
                error_context_lines=cfg.error_context_lines,
                keep_first_error=cfg.keep_first_error,
                keep_last_error=cfg.keep_last_error,
                max_stack_traces=cfg.max_stack_traces,
                stack_trace_max_lines=cfg.stack_trace_max_lines,
                max_warnings=cfg.max_warnings,
                dedupe_warnings=cfg.dedupe_warnings,
                keep_summary_lines=cfg.keep_summary_lines,
                max_total_lines=cfg.max_total_lines,
                enable_ccr=cfg.enable_ccr,
                min_lines_for_ccr=cfg.min_lines_for_ccr,
                min_compression_ratio_for_ccr=0.5,
                collapse_runtime_frames=cfg.collapse_runtime_frames,
                trace_head_frames=cfg.trace_head_frames,
                trace_app_frames=cfg.trace_app_frames,
            )
        )

    # ─── Public API ─────────────────────────────────────────────────────

    def compress(self, content: str, context: str = "", bias: float = 1.0) -> LogCompressionResult:
        # `context` is unused upstream and unused here (Python original
        # also didn't use it). Kept in the signature for drop-in compat.
        del context
        rust_result = self._rust.compress(content, bias)
        cache_key: str | None = rust_result.cache_key
        if cache_key is not None:
            self._persist_to_python_ccr(content, rust_result.compressed, cache_key)

        stats_dict = {k: int(v) for k, v in cast("dict[str, int]", rust_result.stats).items()}
        return LogCompressionResult(
            compressed=rust_result.compressed,
            original=content,
            original_line_count=rust_result.original_line_count,
            compressed_line_count=rust_result.compressed_line_count,
            format_detected=_format_from_str(rust_result.format_detected),
            compression_ratio=rust_result.compression_ratio,
            cache_key=cache_key,
            stats=stats_dict,
        )

    # ─── Legacy internal helpers (test surface compat) ──────────────────

    def _detect_format(self, lines: list[str]) -> LogFormat:
        """Delegate to the Rust format detector."""
        from headroom._core import detect_log_format

        return _format_from_str(detect_log_format(list(lines)))

    def _parse_lines(self, lines: list[str]) -> list[LogLine]:
        """Parse + categorize lines, mirroring Rust's classification.

        Stays Python so the legacy direct-call test surface keeps
        working without rebuilding through Rust on every test. Rust
        unit tests pin Rust's behavior; this implementation must
        mirror Rust's level/stack-trace/summary classification rules.
        """
        import re

        # Mirror of Rust's level classifier: aho-corasick with
        # word-boundary post-filter. Python's `re` is fast enough for
        # the test path; the Rust path uses aho-corasick. Both share
        # the same keyword set.
        level_patterns = [
            (
                LogLevel.ERROR,
                re.compile(r"\b(?:ERROR|error|Error|FATAL|fatal|Fatal|CRITICAL|critical)\b"),
            ),
            (LogLevel.FAIL, re.compile(r"\b(?:FAIL|FAILED|fail|failed|Fail|Failed)\b")),
            (LogLevel.WARN, re.compile(r"\b(?:WARN|WARNING|warn|warning|Warn|Warning)\b")),
            (LogLevel.INFO, re.compile(r"\b(?:INFO|info|Info)\b")),
            (LogLevel.DEBUG, re.compile(r"\b(?:DEBUG|debug|Debug)\b")),
            (LogLevel.TRACE, re.compile(r"\b(?:TRACE|trace|Trace)\b")),
        ]
        stack_trace_patterns = [
            re.compile(r"^\s*Traceback \(most recent call last\)"),
            re.compile(r'^\s*File ".+", line \d+'),
            re.compile(r"^\s*at .+\(.+:\d+:\d+\)"),
            re.compile(r"^\s+at [\w.$]+\("),
            re.compile(r"^\s*--> .+:\d+:\d+"),
            re.compile(r"^\s*\d+:\s+0x[0-9a-f]+"),
            # Rust panics (RustBacktrace flavor)
            re.compile(r"^thread '[^']*' panicked at"),
            re.compile(r"^stack backtrace:"),
            re.compile(r"^\s+\d+: \S"),
            # Go panics / goroutine dumps (GoPanic flavor)
            re.compile(r"^(?:panic|fatal error): "),
            re.compile(r"^goroutine \d+ \["),
            re.compile(r"^\t\S+\.go:\d+(?: \+0x[0-9a-f]+)?$"),
            # .NET (DotNet flavor)
            re.compile(r"^Unhandled exception\."),
            re.compile(r"^\s*at .+\) in .+:line \d+"),
            # Java chained-exception continuations
            re.compile(r"^Caused by: "),
            re.compile(r"^\s*\.\.\. \d+ more$"),
        ]
        summary_patterns = [
            re.compile(r"^={3,}"),
            re.compile(r"^-{3,}"),
            re.compile(r"^\d+ (passed|failed|skipped|error|warning)"),
            re.compile(r"^(?:Tests?|Suites?):?\s+\d+"),
            re.compile(r"^(?:TOTAL|Total|Summary)"),
            re.compile(r"^(?:Build|Compile|Test).*(?:succeeded|failed|complete)"),
        ]

        log_lines: list[LogLine] = []
        in_stack_trace = False
        stack_trace_lines = 0

        for i, line in enumerate(lines):
            log_line = LogLine(line_number=i, content=line)

            for level, pattern in level_patterns:
                if pattern.search(line):
                    log_line.level = level
                    break

            for pattern in stack_trace_patterns:
                if pattern.search(line):
                    in_stack_trace = True
                    stack_trace_lines = 0
                    break

            if in_stack_trace:
                log_line.is_stack_trace = True
                stack_trace_lines += 1
                if stack_trace_lines > self.config.stack_trace_max_lines or not line.strip():
                    in_stack_trace = False

            for pattern in summary_patterns:
                if pattern.search(line):
                    log_line.is_summary = True
                    break

            log_line.score = self._score_line(log_line)
            log_lines.append(log_line)

        return log_lines

    def _score_line(self, log_line: LogLine) -> float:
        """Per-line importance scoring."""
        level_scores = {
            LogLevel.ERROR: 1.0,
            LogLevel.FAIL: 1.0,
            LogLevel.WARN: 0.5,
            LogLevel.INFO: 0.1,
            LogLevel.DEBUG: 0.05,
            LogLevel.TRACE: 0.02,
            LogLevel.UNKNOWN: 0.1,
        }
        score = level_scores.get(log_line.level, 0.1)
        if log_line.is_stack_trace:
            score += 0.3
        if log_line.is_summary:
            score += 0.4
        return min(1.0, score)

    def _select_lines(self, log_lines: list[LogLine], bias: float = 1.0) -> list[LogLine]:
        """Select important lines using the same algorithm Rust uses."""
        from headroom.transforms.adaptive_sizer import compute_optimal_k

        all_strings = [line.content for line in log_lines]
        adaptive_max = compute_optimal_k(
            all_strings, bias=bias, min_k=10, max_k=self.config.max_total_lines
        )

        errors: list[LogLine] = []
        fails: list[LogLine] = []
        warnings: list[LogLine] = []
        stack_traces: list[list[LogLine]] = []
        summaries: list[LogLine] = []
        current_stack: list[LogLine] = []

        for log_line in log_lines:
            if log_line.level == LogLevel.ERROR:
                errors.append(log_line)
            elif log_line.level == LogLevel.FAIL:
                fails.append(log_line)
            elif log_line.level == LogLevel.WARN:
                warnings.append(log_line)
            if log_line.is_stack_trace:
                current_stack.append(log_line)
            elif current_stack:
                stack_traces.append(current_stack)
                current_stack = []
            if log_line.is_summary:
                summaries.append(log_line)
        if current_stack:
            stack_traces.append(current_stack)

        selected: list[LogLine] = []
        if errors:
            selected.extend(self._select_with_first_last(errors, self.config.max_errors))
        if fails:
            selected.extend(self._select_with_first_last(fails, self.config.max_errors))
        if warnings:
            if self.config.dedupe_warnings:
                warnings = self._dedupe_similar(warnings)
            selected.extend(warnings[: self.config.max_warnings])
        for stack in stack_traces[: self.config.max_stack_traces]:
            selected.extend(stack[: self.config.stack_trace_max_lines])
        if self.config.keep_summary_lines:
            selected.extend(summaries)

        selected = self._add_context(log_lines, selected)
        selected = sorted(set(selected), key=lambda x: x.line_number)

        if len(selected) > adaptive_max:
            selected = sorted(selected, key=lambda x: x.score, reverse=True)
            selected = selected[:adaptive_max]
            selected = sorted(selected, key=lambda x: x.line_number)

        return selected

    def _select_with_first_last(self, lines: list[LogLine], max_count: int) -> list[LogLine]:
        if len(lines) <= max_count:
            return lines

        selected: list[LogLine] = []
        if self.config.keep_first_error and lines:
            selected.append(lines[0])
        if self.config.keep_last_error and lines and lines[-1] not in selected:
            selected.append(lines[-1])

        remaining = max_count - len(selected)
        if remaining > 0:
            candidates = sorted(
                (line for line in lines if line not in selected),
                key=lambda x: x.score,
                reverse=True,
            )
            selected.extend(candidates[:remaining])

        return selected

    def _dedupe_similar(self, lines: list[LogLine]) -> list[LogLine]:
        """Conservative dedupe — preserves message prefix, only
        normalises trailing variable region (digits, hex, paths).
        Mirrors Rust `normalize_for_dedupe`."""
        import re

        seen: set[str] = set()
        deduped: list[LogLine] = []
        digit_re = re.compile(r"\d+")
        hex_re = re.compile(r"0x[0-9a-fA-F]+")
        path_re = re.compile(r"/[\w/]+/")

        for line in lines:
            content = line.content
            split_at = next((i for i, c in enumerate(content) if c in (":", "=")), len(content))
            prefix = content[:split_at]
            suffix = content[split_at:]
            suffix = digit_re.sub("N", suffix)
            suffix = hex_re.sub("ADDR", suffix)
            suffix = path_re.sub("/PATH/", suffix)
            normalized = prefix + suffix
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(line)
        return deduped

    def _add_context(self, all_lines: list[LogLine], selected: list[LogLine]) -> list[LogLine]:
        selected_indices = {line.line_number for line in selected}
        context_indices: set[int] = set()
        for idx in selected_indices:
            for i in range(max(0, idx - self.config.error_context_lines), idx):
                context_indices.add(i)
            for i in range(
                idx + 1,
                min(len(all_lines), idx + self.config.error_context_lines + 1),
            ):
                context_indices.add(i)
        for idx in context_indices:
            if idx not in selected_indices and idx < len(all_lines):
                selected.append(all_lines[idx])
        return selected

    def _format_output(
        self, selected: list[LogLine], all_lines: list[LogLine]
    ) -> tuple[str, dict[str, int]]:
        stats: dict[str, int] = {
            "errors": sum(1 for line in all_lines if line.level == LogLevel.ERROR),
            "fails": sum(1 for line in all_lines if line.level == LogLevel.FAIL),
            "warnings": sum(1 for line in all_lines if line.level == LogLevel.WARN),
            "info": sum(1 for line in all_lines if line.level == LogLevel.INFO),
            "total": len(all_lines),
            "selected": len(selected),
        }
        output_lines = [line.content for line in selected]
        omitted = len(all_lines) - len(selected)
        if omitted > 0:
            summary_parts: list[str] = []
            for label, key in (
                ("ERROR", "errors"),
                ("FAIL", "fails"),
                ("WARN", "warnings"),
                ("INFO", "info"),
            ):
                count = stats[key]
                if count > 0:
                    summary_parts.append(f"{count} {label}")
            if summary_parts:
                output_lines.append(f"[{omitted} lines omitted: {', '.join(summary_parts)}]")
        return "\n".join(output_lines), stats

    def _store_in_ccr(self, original: str, compressed: str, original_count: int) -> str | None:
        """Backwards-compat shim — the legacy callsite name. Now
        delegates to `_persist_to_python_ccr`. Returns the stored
        cache_key if persistence succeeded, else None.
        """
        # Compute the same cache key the Rust path would (MD5 of
        # original truncated to 24 hex chars).
        import hashlib

        cache_key = hashlib.md5(original.encode()).hexdigest()[:24]
        try:
            from ..cache.compression_store import get_compression_store
        except ImportError as e:
            logger.warning("CCR store import failed; cache_key %s not persisted: %s", cache_key, e)
            return None
        try:
            store: Any = get_compression_store()
            return cast(
                "str | None",
                store.store(original, compressed, original_item_count=original_count),
            )
        except Exception as e:
            logger.warning("CCR store write failed; cache_key %s not persisted: %s", cache_key, e)
            return None

    def _persist_to_python_ccr(self, original: str, compressed: str, cache_key: str) -> None:
        """Promote a Rust-emitted cache_key into the production Python
        CompressionStore. Failures are logged at warning level."""
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
                "CCR store write failed; cache_key %s remains in-marker only: %s",
                cache_key,
                e,
            )


__all__ = [
    "LogCompressor",
    "LogCompressorConfig",
    "LogCompressionResult",
    "LogFormat",
    "LogLevel",
    "LogLine",
]
