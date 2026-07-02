"""LeanContext — window-based context truncation for tool outputs.

A simple, zero-dependency alternative to RTK (Rust Token Killer).
Instead of ML-based relevance scoring, it keeps a window of lines
around "signals" (error messages, edit locations, search matches)
and drops lines far from any signal.

Benefits over RTK:
- Zero external dependencies (pure Python)
- Configurable window radius
- Transparent: you can see exactly which lines were kept/dropped
- Fast: O(n) single pass, no ML inference

Usage:
    from headroom.transforms.lean_context import LeanContext
    truncator = LeanContext(window_radius=50)
    result = truncator.truncate(tool_output_text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Patterns that indicate "this line is important — keep context around it"
SIGNAL_PATTERNS = [
    # Compiler errors
    re.compile(r"error\[E\d+\]", re.IGNORECASE),
    re.compile(r"error:", re.IGNORECASE),
    re.compile(r"^\s*-->\s+\S+:\d+:\d+", re.MULTILINE),  # rustc error location
    # Test failures
    re.compile(r"FAILED|FAIL:", re.IGNORECASE),
    re.compile(r"assert.*failed", re.IGNORECASE),
    re.compile(r"panicked at", re.IGNORECASE),
    # Tracebacks
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"^\s*File \"[^\"]+\", line \d+", re.MULTILINE),
    re.compile(r"^\s+\^+$", re.MULTILINE),  # Python error pointer
    # Build output
    re.compile(r"error:\s+Could not compile", re.IGNORECASE),
    re.compile(r"ERROR in", re.IGNORECASE),
    re.compile(r"^\s*\d+\|\s", re.MULTILINE),  # Line-numbered code in errors
    # Edit/diff signals
    re.compile(r"^[+-]{3}\s", re.MULTILINE),  # diff headers
    re.compile(r"^@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@", re.MULTILINE),  # diff hunks
    # Search results
    re.compile(r"^[^:]+:\d+:", re.MULTILINE),  # grep/ripgrep output
    # General relevance
    re.compile(r"(?i)(fix|patch|resolve|address|workaround|solution)"),
    re.compile(r"(?i)(critical|fatal|panic|crash|deadlock|timeout|OOM)"),
    # npm/yarn errors
    re.compile(r"npm ERR!", re.IGNORECASE),
    re.compile(r"yarn ERR", re.IGNORECASE),
    re.compile(r"ELIFECYCLE", re.IGNORECASE),
]


@dataclass
class TruncationResult:
    """Result of window-based truncation."""
    text: str
    original_lines: int
    kept_lines: int
    dropped_lines: int
    signal_lines: int
    window_radius: int

    @property
    def savings_pct(self) -> float:
        if self.original_lines == 0:
            return 0.0
        return (self.dropped_lines / self.original_lines) * 100


class LeanContext:
    """Window-based context truncation for tool outputs.

    Keeps lines within `window_radius` of any signal line.
    Drops lines far from all signals.
    """

    def __init__(self, window_radius: int = 50):
        self.window_radius = window_radius

    def find_signals(self, lines: list[str]) -> set[int]:
        """Find line indices that match signal patterns."""
        signals: set[int] = set()
        # Check each line and its neighbors (for multi-line patterns)
        for i, line in enumerate(lines):
            # Check single line
            for pattern in SIGNAL_PATTERNS:
                if pattern.search(line):
                    signals.add(i)
                    break
            # Check 2-line windows (e.g. rustc error location + code)
            if i + 1 < len(lines):
                window2 = lines[i] + "\n" + lines[i+1]
                for pattern in SIGNAL_PATTERNS:
                    if pattern.search(window2):
                        signals.add(i)
                        signals.add(i+1)
            # Check 3-line windows (e.g. diff hunks)
            if i + 2 < len(lines):
                window3 = "\n".join(lines[i:i+3])
                for pattern in SIGNAL_PATTERNS:
                    if pattern.search(window3):
                        signals.update(range(i, i+3))
        return signals

    def truncate(self, text: str) -> TruncationResult:
        """Truncate text, keeping only lines near signals."""
        if not text.strip():
            return TruncationResult(
                text=text, original_lines=0, kept_lines=0,
                dropped_lines=0, signal_lines=0,
                window_radius=self.window_radius,
            )

        lines = text.split("\n")
        n = len(lines)

        if n <= self.window_radius * 2:
            # Text is short enough — keep everything
            return TruncationResult(
                text=text, original_lines=n, kept_lines=n,
                dropped_lines=0, signal_lines=0,
                window_radius=self.window_radius,
            )

        signals = self.find_signals(lines)

        if not signals:
            # No signals found — keep first and last N lines (headers + tail)
            keep: set[int] = set(range(min(self.window_radius, n)))
            keep.update(range(max(0, n - self.window_radius), n))
        else:
            # Keep lines within window_radius of any signal
            keep = set()
            for signal_line in signals:
                start = max(0, signal_line - self.window_radius)
                end = min(n, signal_line + self.window_radius + 1)
                keep.update(range(start, end))

        # Always keep first 3 and last 3 lines (context boundaries)
        keep.update(range(min(3, n)))
        keep.update(range(max(0, n - 3), n))

        kept_lines_list = [lines[i] for i in sorted(keep)]
        dropped = n - len(kept_lines_list)

        # Insert truncation markers where lines were dropped
        result_lines = []
        prev_kept = -2
        for i in sorted(keep):
            if i > prev_kept + 1:
                skipped = i - prev_kept - 1
                result_lines.append(f"... [{skipped} lines dropped] ...")
            result_lines.append(lines[i])
            prev_kept = i

        return TruncationResult(
            text="\n".join(result_lines),
            original_lines=n,
            kept_lines=len(kept_lines_list),
            dropped_lines=dropped,
            signal_lines=len(signals),
            window_radius=self.window_radius,
        )


# ── Singleton for proxy use ──────────────────────────────────────────
_default_truncator: LeanContext | None = None


def get_lean_context(window_radius: int = 50) -> LeanContext:
    """Get or create the default LeanContext instance."""
    global _default_truncator
    if _default_truncator is None or _default_truncator.window_radius != window_radius:
        _default_truncator = LeanContext(window_radius=window_radius)
    return _default_truncator


def truncate_text(text: str, window_radius: int = 50) -> str:
    """Convenience: truncate text and return result string."""
    return get_lean_context(window_radius).truncate(text).text
