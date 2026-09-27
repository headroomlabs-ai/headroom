"""Edge 2 — rerank shell/test logs and web fetch/search results.

Only targets outputs over `rerank_trigger_tokens` (2,000 by default) — the
engine already skips web tools entirely, so this edge is the only compressor
those results ever see; for shell/test logs it's scoring what Headroom's
own log compressor left in the live zone.

Always keeps error/failure/exception/traceback lines (with context), the
first and last N lines, and heading-shaped lines, verbatim and in original
order. Everything else is scored against the intent query and kept until a
token budget is met; the rest is elided behind a retrievable marker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .cache_safety import (
    SavingsLedgerEntry,
    already_compressed,
    cap_input,
    estimate_tokens,
    format_elided_marker,
    store_original,
)
from .config import EdgeLayerConfig
from .scoring import Scorer

_ERROR_RE = re.compile(
    r"\b(error|exception|failure|failed|traceback|panic|fatal)\b", re.IGNORECASE
)
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6}\s|[A-Z][A-Za-z0-9 _/.-]{0,60}:\s*$)")


@dataclass
class _Chunk:
    start: int
    end: int  # exclusive
    lines: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _protected_line_indices(lines: list[str], *, config: EdgeLayerConfig) -> set[int]:
    protected: set[int] = set()
    for i in range(min(config.rerank_keep_head_lines, len(lines))):
        protected.add(i)
    for i in range(max(0, len(lines) - config.rerank_keep_tail_lines), len(lines)):
        protected.add(i)
    for i, line in enumerate(lines):
        if _ERROR_RE.search(line) or _HEADING_RE.match(line):
            lo = max(0, i - config.rerank_context_lines)
            hi = min(len(lines), i + config.rerank_context_lines + 1)
            protected.update(range(lo, hi))
    return protected


def _build_chunks(lines: list[str], *, protected: set[int], max_lines: int) -> list[_Chunk]:
    """Blank-line-bounded chunks, capped at `max_lines`, never splitting a
    protected line away from the chunk boundary logic (protected status is
    computed per-chunk afterward, by any-overlap, not per-line here).
    """
    chunks: list[_Chunk] = []
    start = 0
    current: list[str] = []
    for i, line in enumerate(lines):
        current.append(line)
        is_last = i == len(lines) - 1
        blank_boundary = not is_last and line.strip() == "" and lines[i + 1].strip() != ""
        length_boundary = len(current) >= max_lines
        if is_last or blank_boundary or length_boundary:
            chunks.append(_Chunk(start=start, end=i + 1, lines=current))
            current = []
            start = i + 1
    return chunks


def compress_log_or_web(
    text: str,
    *,
    tool_name: str,
    tool_call_id: str | None,
    intent_query: str,
    config: EdgeLayerConfig,
    store: Any,
    scorer: Scorer,
    request_id: str,
) -> tuple[str, SavingsLedgerEntry | None]:
    if already_compressed(text):
        return text, None
    capped = cap_input(text, max_chars=config.max_input_chars_per_edge)
    if capped is None:
        return text, None

    total_tokens = estimate_tokens(text)
    if total_tokens < config.rerank_trigger_tokens:
        return text, None

    lines = text.split("\n")
    protected_indices = _protected_line_indices(lines, config=config)
    chunks = _build_chunks(lines, protected=protected_indices, max_lines=config.rerank_chunk_max_lines)

    selected = [
        chunk for chunk in chunks if any(i in protected_indices for i in range(chunk.start, chunk.end))
    ]
    selected_set = set(id(c) for c in selected)
    running_tokens = sum(estimate_tokens(c.text) for c in selected)

    budget = max(config.rerank_min_tokens_kept, int(config.rerank_target_ratio * total_tokens))
    scorable = [c for c in chunks if id(c) not in selected_set]
    if scorable and running_tokens < budget:
        scores = scorer.score(intent_query, [c.text for c in scorable])
        ranked = sorted(zip(scorable, scores), key=lambda pair: pair[1], reverse=True)
        for chunk, _score in ranked:
            if running_tokens >= budget:
                break
            selected_set.add(id(chunk))
            running_tokens += estimate_tokens(chunk.text)

    if len(selected_set) == len(chunks):
        return text, None  # nothing was actually elided

    output_parts: list[str] = []
    elided_run: list[_Chunk] = []

    def _flush_elided_run() -> None:
        if not elided_run:
            return
        elided_text = "\n".join(chunk.text for chunk in elided_run)
        elided_line_count = sum(len(chunk.lines) for chunk in elided_run)
        hash_key = store_original(
            store,
            elided_text,
            elided_text,
            config_version=config.config_version,
            salt=f"rerank:{tool_name}:{elided_run[0].start}",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            query_context=intent_query,
        )
        output_parts.append(
            format_elided_marker(count=elided_line_count, unit="lines", kept=0, hash_key=hash_key)
        )
        elided_run.clear()

    for chunk in chunks:
        if id(chunk) in selected_set:
            _flush_elided_run()
            output_parts.append(chunk.text)
        else:
            elided_run.append(chunk)
    _flush_elided_run()

    compressed = "\n".join(output_parts)
    entry = SavingsLedgerEntry(
        edge="rerank",
        bucket="log" if "test" in tool_name.lower() or "bash" in tool_name.lower() else "web",
        tool_name=tool_name,
        tokens_before=total_tokens,
        tokens_after=estimate_tokens(compressed),
        hash_key=None,
        request_id=request_id,
    )
    return compressed, entry
