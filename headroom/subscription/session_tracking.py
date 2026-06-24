"""Parse Claude Code transcript JSONL files for per-window token breakdowns.

Mirrors the approach in the ClaudeCacheTTLStatusLine TypeScript reference
implementation (session-tracking.ts). Reads ~/.claude/projects/**/*.jsonl
and aggregates token usage for entries whose timestamp falls within a window.

Model weights (Sonnet-normalised, empirical estimates):
  opus:   2.0×  (higher rate-limit cost)
  sonnet: 1.0×  (baseline)
  haiku:  0.5×  (cheaper, lower rate-limit cost)

The weighted_token_equivalent lets callers detect surge pricing by comparing
it against the API-reported utilisation × window_limit.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from headroom.subscription.models import WindowTokens

logger = logging.getLogger(__name__)

# Maximum bytes to read per transcript file (10 MB cap — generous, typical files <1 MB)
_MAX_FILE_BYTES = 10 * 1024 * 1024

# Sonnet-normalised model family weights
MODEL_FAMILY_WEIGHTS: dict[str, float] = {
    "opus": 2.0,
    "sonnet": 1.0,
    "haiku": 0.5,
}
DEFAULT_MODEL_WEIGHT: float = 1.0


def _claude_config_dir() -> Path:
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    return Path(base)


def get_model_weight(model_id: str) -> float:
    """Return the Sonnet-normalised weight for a model ID.

    Matches against known family names using a word-boundary check.
    Falls back to DEFAULT_MODEL_WEIGHT for unrecognised models.
    """
    lower = model_id.lower()
    import re

    for family, weight in MODEL_FAMILY_WEIGHTS.items():
        if re.search(rf"(?<![a-z]){family}(?![a-z])", lower):
            return weight
    return DEFAULT_MODEL_WEIGHT


def find_transcript_files() -> list[Path]:
    """Return all .jsonl files under ~/.claude/projects."""
    projects = _claude_config_dir() / "projects"
    results: list[Path] = []
    _walk_jsonl(projects, results)
    return results


def _walk_jsonl(directory: Path, results: list[Path]) -> None:
    try:
        entries = list(directory.iterdir())
    except (OSError, PermissionError):
        return
    for entry in entries:
        try:
            if entry.is_dir():
                _walk_jsonl(entry, results)
            elif entry.suffix == ".jsonl":
                results.append(entry)
        except OSError:
            continue


def _read_transcript_lines(path: Path) -> list[str]:
    try:
        size = path.stat().st_size
        read_size = min(size, _MAX_FILE_BYTES)
        with path.open("rb") as fh:
            raw = fh.read(read_size)
        return [line for line in raw.decode("utf-8", errors="replace").splitlines() if line.strip()]
    except Exception:
        return []


def _add_usage_to_tokens(dest: WindowTokens, usage: dict[str, Any]) -> None:
    dest.input += int(usage.get("input_tokens") or 0)
    dest.output += int(usage.get("output_tokens") or 0)
    dest.cache_reads += int(usage.get("cache_read_input_tokens") or 0)

    cache_creation = usage.get("cache_creation") or {}
    w5m = int(cache_creation.get("ephemeral_5m_input_tokens") or 0)
    w1h = int(cache_creation.get("ephemeral_1h_input_tokens") or 0)
    total_writes = int(usage.get("cache_creation_input_tokens") or (w5m + w1h))

    dest.cache_writes_5m += w5m
    dest.cache_writes_1h += w1h
    dest.cache_writes_total += total_writes


def compute_window_tokens(start_ts: float, end_ts: float) -> WindowTokens:
    """Sum transcript token usage for entries in [start_ts, end_ts).

    Args:
        start_ts: Window start as a Unix timestamp (seconds).
        end_ts: Window end as a Unix timestamp (seconds).

    Returns:
        :class:`WindowTokens` with aggregate + per-model breakdown and
        ``weighted_token_equivalent`` (Sonnet-normalised).
    """
    totals = WindowTokens()
    by_model: dict[str, WindowTokens] = {}
    unattributed = WindowTokens()

    for path in find_transcript_files():
        for line in _read_transcript_lines(path):
            try:
                entry: dict[str, Any] = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue

            ts_str = entry.get("timestamp")
            if not ts_str:
                continue
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                ts = dt.timestamp()
            except (ValueError, TypeError):
                continue

            if ts < start_ts or ts >= end_ts:
                continue

            msg = entry.get("message") or {}
            usage = msg.get("usage")
            if not usage:
                continue

            _add_usage_to_tokens(totals, usage)

            model_id: str | None = msg.get("model")
            if model_id:
                if model_id not in by_model:
                    by_model[model_id] = WindowTokens()
                _add_usage_to_tokens(by_model[model_id], usage)
            else:
                _add_usage_to_tokens(unattributed, usage)

    # Compute Sonnet-normalised weighted equivalent
    model_weights: dict[str, float] = {}
    weighted = 0.0

    for model_id, model_tokens in by_model.items():
        w = get_model_weight(model_id)
        model_weights[model_id] = w
        weighted += _total_token_count(model_tokens) * w

    weighted += _total_token_count(unattributed) * DEFAULT_MODEL_WEIGHT

    totals.weighted_token_equivalent = weighted
    totals.by_model = {mid: _window_tokens_to_dict(mt) for mid, mt in by_model.items()}

    return totals


def _total_token_count(t: WindowTokens) -> int:
    return t.input + t.output + t.cache_reads + t.cache_writes_total


def _window_tokens_to_dict(t: WindowTokens) -> dict[str, int]:
    return {
        "input": t.input,
        "output": t.output,
        "cache_reads": t.cache_reads,
        "cache_writes_5m": t.cache_writes_5m,
        "cache_writes_1h": t.cache_writes_1h,
        "cache_writes_total": t.cache_writes_total,
    }
