"""Tabular-text compressor: bridges CSV/TSV/markdown tables to SmartCrusher.

Raw tabular *text* (CSV/TSV files, markdown tables, fixed-width tables) has no
native compressor — it would otherwise fall through to plain-text Kompress,
ignoring its row/column structure. This module parses tabular text into a JSON
array of records and routes it through the existing, battle-tested
`SmartCrusher`, which already does lossless ``csv-schema`` compaction first and
lossy row-drop with reversible ``<<ccr:HASH>>`` markers as a fallback.

A document may hold several tables with prose between them. Each table
carries its own header row, so they are parsed and rendered separately and
the text around them is carried through — folding them together filed the
later header rows as data and dropped the prose entirely.

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

# A parsed table: its header row, then its data rows.
Table = tuple[list[str], list[list[str]]]


# ─── Public dataclasses (mirror SearchCompressor / LogCompressor surface) ────


@dataclass
class TabularCompressorConfig:
    """Configuration for tabular-text compression."""

    # Pass-through to SmartCrusher's lossless renderer.
    compaction_format: str = "csv-schema"
    # Only keep SmartCrusher's output if it is strictly smaller than the
    # original tabular text (already-compact CSV may not benefit losslessly).
    min_savings_chars: int = 1


@dataclass(frozen=True)
class Segment:
    """One piece of a parsed document: a table, or the text around it.

    Exactly one of ``table`` / ``text`` is set.
    """

    table: Table | None = None
    text: str | None = None


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
    # Number of distinct tables found. > 1 means each kept its own schema.
    tables: int = 1

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


def split_csv_blocks(content: str, delimiter: str = ",") -> list[list[list[str]]]:
    """Split delimited text into blocks at blank-line boundaries.

    Goes through the csv reader rather than splitting raw lines so a blank
    line inside a quoted field stays part of its cell.
    """
    blocks: list[list[list[str]]] = []
    current: list[list[str]] = []
    for row in csv.reader(io.StringIO(content), delimiter=delimiter):
        if any(cell.strip() for cell in row):
            current.append(row)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def parse_csv_tables(content: str, delimiter: str = ",") -> list[Table] | None:
    """Parse delimited text into one table per blank-line-separated block.

    Returns ``None`` if any block is unusable — a header with no rows under
    it, or ragged rows that can't be zipped under the headers without
    shifting values into the wrong column (#1652) — so the caller passes the
    text through verbatim rather than emitting something the original never
    said.

    Tables run together with no blank line between them are indistinguishable
    from one table that happens to contain a row of column names, so they
    stay a single table.
    """
    tables: list[Table] = []
    for block in split_csv_blocks(content, delimiter):
        if len(block) < 2:
            return None
        headers = [h.strip() for h in block[0]]
        rows = block[1:]
        width = len(headers)
        if not width or any(len(row) != width for row in rows):
            return None
        tables.append((headers, rows))
    return tables or None


def parse_tabular_segments(
    content: str,
    fmt: str,
    delimiter: str = ",",
) -> list[Segment] | None:
    """Ordered segments for a tabular document, whatever its format.

    Markdown can carry prose between its tables; the delimited and
    fixed-width formats carry tables only.
    """
    if fmt == "markdown":
        return parse_markdown_segments(content) or None
    if fmt == "fixed_width":
        headers, rows = parse_fixed_width(content)
        width = len(headers)
        if not headers or not rows or any(len(row) != width for row in rows):
            return None
        return [Segment(table=(headers, rows))]
    tables = parse_csv_tables(content, delimiter)
    if tables is None:
        return None
    return [Segment(table=table) for table in tables]


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


def split_markdown_segments(content: str) -> list[tuple[str, str]]:
    """Split markdown into ordered ``("table" | "text", block)`` segments.

    ``parse_markdown_table`` keeps every line holding a pipe and discards
    the rest, so a document with several tables collapsed into one — the
    later header rows landing as data — and the headings between them
    vanished. Splitting on pipe/non-pipe runs first keeps each table whole
    and keeps the prose around them.
    """
    segments: list[tuple[str, str]] = []
    buf: list[str] = []
    kind: str | None = None

    for line in content.split("\n"):
        line_kind = "table" if "|" in line and line.strip() else "text"
        if kind is not None and line_kind != kind:
            segments.append((kind, "\n".join(buf)))
            buf = []
        kind = line_kind
        buf.append(line)
    if kind is not None and buf:
        segments.append((kind, "\n".join(buf)))

    return [(k, block) for k, block in segments if block.strip()]


def parse_markdown_segments(content: str) -> list[Segment]:
    """Parse markdown into ordered text blocks and well-formed tables.

    A pipe run that does not parse into a rectangular table — a lone pipe
    inside a sentence, or ragged rows that would shift values under the
    wrong column (#1652) — stays a text block rather than being guessed at.
    """
    parsed: list[Segment] = []
    for kind, block in split_markdown_segments(content):
        if kind == "table":
            headers, rows = parse_markdown_table(block)
            width = len(headers)
            if headers and rows and all(len(row) == width for row in rows):
                parsed.append(Segment(table=(headers, rows)))
                continue
        parsed.append(Segment(text=block))
    return parsed


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


def segments_to_document(segments: list[Segment]) -> dict[str, object]:
    """Build an ordered JSON document from parsed markdown segments.

    Each table becomes a record array keyed by its first column, so
    SmartCrusher's walker renders it as its own CSV+schema block. Text
    blocks keep their place under ``_text{n}`` keys so nothing between
    the tables is lost. Insertion order is the document's order.
    """
    document: dict[str, object] = {}
    text_n = 0
    for segment in segments:
        if segment.table is None:
            text_n += 1
            document[f"_text{text_n}"] = segment.text
            continue
        headers, rows = segment.table
        key = headers[0].strip() or "table"
        if key in document:
            n = 2
            while f"{key}_{n}" in document:
                n += 1
            key = f"{key}_{n}"
        document[key] = to_records(headers, rows)
    return document


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
        detection = detect_content_type(content)
        if detection.content_type is not ContentType.TABULAR:
            return self._passthrough(content, "unknown", 0, 0, 0)

        fmt = detection.metadata.get("format", "csv")
        delimiter = detection.metadata.get("delimiter", ",")
        segments = parse_tabular_segments(content, fmt, delimiter)
        if segments is None:
            return self._passthrough(content, "unknown", 0, 0, 0)

        tables = [seg.table for seg in segments if seg.table is not None]
        has_text = any(seg.table is None for seg in segments)
        if not tables:
            return self._passthrough(content, "unknown", 0, 0, 0)

        # More than one table, or prose to preserve, needs the document
        # walker: one schema per table instead of one schema for all.
        if len(tables) > 1 or has_text:
            return self._compress_document(content, segments, fmt, tables)

        headers, rows = tables[0]

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
            return self._passthrough(content, fmt, len(rows), len(headers), 1)

        return TabularCompressionResult(
            compressed=result.compressed,
            original=content,
            was_modified=True,
            fmt=fmt,
            rows=len(rows),
            columns=len(headers),
            strategy=result.strategy or "tabular",
        )

    def _compress_document(
        self,
        content: str,
        segments: list[Segment],
        fmt: str,
        tables: list[Table],
    ) -> TabularCompressionResult:
        """Render a multi-table / prose-bearing document losslessly."""
        total_rows = sum(len(rows) for _, rows in tables)
        widest = max(len(headers) for headers, _ in tables)

        from .smart_crusher import SmartCrusher

        crusher = SmartCrusher(
            with_compaction=True,
            compaction_format=self.config.compaction_format,
        )
        document = segments_to_document(segments)
        compressed = crusher.compact_document_json(json.dumps(document, ensure_ascii=False))

        savings = len(content) - len(compressed)
        if savings < self.config.min_savings_chars:
            return self._passthrough(content, fmt, total_rows, widest, len(tables))

        return TabularCompressionResult(
            compressed=compressed,
            original=content,
            was_modified=True,
            fmt=fmt,
            rows=total_rows,
            columns=widest,
            tables=len(tables),
        )

    @staticmethod
    def _passthrough(
        content: str,
        fmt: str,
        rows: int,
        columns: int,
        tables: int,
    ) -> TabularCompressionResult:
        return TabularCompressionResult(
            compressed=content,
            original=content,
            was_modified=False,
            fmt=fmt,
            rows=rows,
            columns=columns,
            tables=tables,
        )


__all__ = [
    "Segment",
    "TabularCompressor",
    "TabularCompressorConfig",
    "TabularCompressionResult",
    "parse_csv",
    "parse_markdown_table",
    "parse_fixed_width",
    "parse_csv_tables",
    "parse_markdown_segments",
    "parse_tabular",
    "parse_tabular_segments",
    "segments_to_document",
    "split_csv_blocks",
    "split_markdown_segments",
    "to_records",
]
