"""Tabular-text compressor: bridges CSV/TSV/markdown tables to SmartCrusher.

Raw tabular *text* (CSV/TSV files, markdown tables, fixed-width tables) has no
native compressor — it would otherwise fall through to plain-text Kompress,
ignoring its row/column structure. This module parses tabular text into a JSON
array of records and routes it through the existing, battle-tested
`SmartCrusher`, which already does lossless ``csv-schema`` compaction first and
lossy row-drop with reversible ``<<ccr:HASH>>`` markers as a fallback.

No new compression algorithm and no new CCR plumbing live here — only the
text→records bridge.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass

from .content_detector import ContentType, detect_content_type

# Mirrors content_detector's separator-cell pattern (e.g. ``| --- | :--: |``).
_MD_SEP_CELL = re.compile(r"^:?-{2,}:?$")


# ─── Public dataclasses (mirror SearchCompressor / LogCompressor surface) ────


@dataclass
class TabularCompressorConfig:
    """Configuration for tabular-text compression."""

    # Pass-through to SmartCrusher's lossless renderer.
    compaction_format: str = "csv-schema"
    # Only keep SmartCrusher's output if it is strictly smaller than the
    # original tabular text (already-compact CSV may not benefit losslessly).
    min_savings_chars: int = 1


@dataclass
class TabularCompressionResult:
    """Result of tabular-text compression."""

    compressed: str
    original: str
    was_modified: bool
    fmt: str  # "csv" | "markdown" | "fixed_width"
    rows: int
    columns: int
    strategy: str = "tabular"

    @property
    def compression_ratio(self) -> float:
        if not self.original:
            return 0.0
        return len(self.compressed) / len(self.original)


# ─── Parsers (text → headers + rows) ─────────────────────────────────────────


def parse_csv(content: str, delimiter: str = ",") -> tuple[list[str], list[list[str]]]:
    """Parse delimited text via the stdlib csv reader."""
    reader = csv.reader(io.StringIO(content), delimiter=delimiter)
    parsed = [row for row in reader if any(cell.strip() for cell in row)]
    if not parsed:
        return [], []
    headers = [h.strip() for h in parsed[0]]
    return headers, parsed[1:]


def parse_markdown_table(content: str) -> tuple[list[str], list[list[str]]]:
    """Parse a markdown table, dropping the ``|---|`` separator row."""

    def split_row(row: str) -> list[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    def is_separator(row: str) -> bool:
        cells = [c for c in split_row(row) if c]
        return len(cells) >= 2 and all(_MD_SEP_CELL.match(c) for c in cells)

    lines = [ln for ln in content.split("\n") if ln.strip() and "|" in ln]
    if len(lines) < 2:
        return [], []
    headers = split_row(lines[0])
    rows = [split_row(ln) for ln in lines[1:] if not is_separator(ln)]
    return headers, rows


def parse_fixed_width(content: str) -> tuple[list[str], list[list[str]]]:
    """Parse whitespace-aligned columns (best-effort, ≥ 2 spaces as a gap)."""
    lines = [ln for ln in content.split("\n") if ln.strip()]
    if len(lines) < 2:
        return [], []
    splitter = re.compile(r"\s{2,}")
    headers = splitter.split(lines[0].strip())
    rows = [splitter.split(ln.strip()) for ln in lines[1:]]
    return headers, rows


def to_records(headers: list[str], rows: list[list[str]]) -> list[dict[str, str]]:
    """Zip headers with each row into dicts, padding/truncating to width."""
    if not headers:
        return []
    width = len(headers)
    records: list[dict[str, str]] = []
    for row in rows:
        padded = (row + [""] * width)[:width]
        records.append({headers[i]: padded[i] for i in range(width)})
    return records


def parse_tabular(
    content: str,
) -> tuple[list[str], list[list[str]], str] | None:
    """Detect the tabular format and parse to (headers, rows, fmt).

    Returns ``None`` if the content is not tabular.
    """
    detection = detect_content_type(content)
    if detection.content_type is not ContentType.TABULAR:
        return None

    fmt = detection.metadata.get("format", "csv")
    if fmt == "markdown":
        headers, rows = parse_markdown_table(content)
    elif fmt == "fixed_width":
        headers, rows = parse_fixed_width(content)
    else:
        delimiter = detection.metadata.get("delimiter", ",")
        headers, rows = parse_csv(content, delimiter)

    if not headers or not rows:
        return None
    # Ragged tables (rows whose cell count differs from the header count)
    # can't be zipped into records without shifting values under the wrong
    # column — a compressed table must never state facts the original
    # didn't (#1652). Treat them as non-tabular and pass through verbatim.
    width = len(headers)
    if any(len(row) != width for row in rows):
        return None
    return headers, rows, fmt


# ─── Compressor (text → records → SmartCrusher) ──────────────────────────────


class TabularCompressor:
    """Compresses tabular text by bridging it through SmartCrusher.

    Public surface mirrors the other content-type compressors so the router
    and tests treat it uniformly.
    """

    def __init__(self, config: TabularCompressorConfig | None = None) -> None:
        self.config = config or TabularCompressorConfig()

    def compress(
        self,
        content: str,
        context: str = "",
        bias: float = 1.0,
    ) -> TabularCompressionResult:
        parsed = parse_tabular(content)
        if parsed is None:
            return TabularCompressionResult(
                compressed=content,
                original=content,
                was_modified=False,
                fmt="unknown",
                rows=0,
                columns=0,
            )

        headers, rows, fmt = parsed
        records = to_records(headers, rows)
        json_str = json.dumps(records, ensure_ascii=False)

        # Lazy import keeps the Rust dependency off the import path until a
        # tabular payload actually arrives.
        from .smart_crusher import SmartCrusher

        crusher = SmartCrusher(
            with_compaction=True,
            compaction_format=self.config.compaction_format,
        )
        result = crusher.crush(json_str, context, bias)

        # SmartCrusher compressed the JSON form; compare its output against the
        # original *tabular text*. Already-compact CSV may not beat its own
        # source, so only adopt the result when it genuinely saves bytes.
        savings = len(content) - len(result.compressed)
        if not result.was_modified or savings < self.config.min_savings_chars:
            return TabularCompressionResult(
                compressed=content,
                original=content,
                was_modified=False,
                fmt=fmt,
                rows=len(rows),
                columns=len(headers),
            )

        return TabularCompressionResult(
            compressed=result.compressed,
            original=content,
            was_modified=True,
            fmt=fmt,
            rows=len(rows),
            columns=len(headers),
            strategy=result.strategy or "tabular",
        )


__all__ = [
    "TabularCompressor",
    "TabularCompressorConfig",
    "TabularCompressionResult",
    "parse_csv",
    "parse_markdown_table",
    "parse_fixed_width",
    "parse_tabular",
    "to_records",
]
