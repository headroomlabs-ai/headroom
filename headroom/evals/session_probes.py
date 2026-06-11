"""Deterministic retention probes over recorded compression events.

Offline scoring of what compression removed from real proxied sessions — no
LLM, no API key. For each event recorded by
``headroom.proxy.probe_recorder``, probe targets are extracted from the
ORIGINAL tool-result content and each is classified against the compressed
messages as:

- ``retained``    — appears verbatim in the compressed content, or survives in
                    punctuation-normalized form (compressors legitimately
                    reshape JSON into tables/KV; for numerics the key and the
                    value must both survive the format change)
- ``recoverable`` — absent, but the compressed content carries a CCR
                    retrieval marker, so the agent can fetch the original back
- ``lost``        — absent with no retrieval path

Dimensions follow production session-replay findings: exact numerics are the
leakiest under compression, artifact trails (paths, hashes, URLs) the weakest,
and error evidence the most critical to keep.

Known limitation: marker recoverability is event-scoped, not block-scoped — a
retrieval marker anywhere in the compressed messages marks every missing
target as recoverable, which can overcount when the marker belongs to a
different block than the loss. The metric is comparative (across ratios,
transforms, and versions), not absolute.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from headroom.learn.scanner import is_error_content

DIMENSIONS = ("numerics", "artifacts", "errors")

# Mirrors the marker shapes matched by
# headroom.transforms.compression_units._CCR_MARKER_RE (kept local so the
# evals layer does not depend on a private transforms symbol).
_CCR_MARKER_RE = re.compile(r"Retrieve more: hash=|Retrieve original: hash=|<<ccr:[^>]+>>")

# A number with its immediate key context ("retry_limit: 3", "port=8787",
# JSON's '"latency_ms": 12'). Bare numbers are skipped: without context they
# are unverifiable noise.
_NUMERIC_RE = re.compile(r"[A-Za-z_][\w.-]{0,24}\"?[ =:]{1,3}\d+(?:\.\d+)?")
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")
_PATH_RE = re.compile(r"(?:~/|\.{1,2}/|/)?(?:[\w.-]+/){2,}[\w.@-]+")
# Requires at least one a-f so bare decimal runs (timestamps, row counts) are
# not mistaken for content hashes.
_HEX_RE = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

_MIN_TARGET_LEN = 4
_ERROR_LINE_PREFIX_LEN = 160
# Final bucket catches inflation events (tokens_after > tokens_before), which
# the recorder captures because compression changed the token count.
_RATIO_BUCKETS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01), (1.01, float("inf")))

# Collapse punctuation that format conversions (JSON -> table/KV/CSV) rewrite,
# keeping path/url/hash-significant characters.
_NORMALIZE_RE = re.compile(r"[^\w./-]+")
_NUMERIC_SPLIT_RE = re.compile(r"(.+?)[\"' =:]+(\d+(?:\.\d+)?)$")


@dataclass
class DimensionTally:
    """Counts for one probe dimension."""

    total: int = 0
    retained: int = 0
    recoverable: int = 0

    @property
    def lost(self) -> int:
        return self.total - self.retained - self.recoverable

    def add(self, other: DimensionTally) -> None:
        self.total += other.total
        self.retained += other.retained
        self.recoverable += other.recoverable

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "retained": self.retained,
            "recoverable": self.recoverable,
            "lost": self.lost,
        }


@dataclass
class EventProbeResult:
    """Probe outcome for a single recorded compression event."""

    request_id: str
    ratio: float
    transforms: list[str]
    dims: dict[str, DimensionTally]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "ratio": round(self.ratio, 4),
            "transforms": self.transforms,
            "dimensions": {name: tally.to_dict() for name, tally in self.dims.items()},
        }


@dataclass
class ProbeReport:
    """Aggregate probe outcomes across all recorded events."""

    events: list[EventProbeResult] = field(default_factory=list)
    skipped_lines: int = 0

    def aggregate(self) -> dict[str, DimensionTally]:
        totals = {name: DimensionTally() for name in DIMENSIONS}
        for event in self.events:
            for name, tally in event.dims.items():
                totals[name].add(tally)
        return totals

    def by_ratio_bucket(self) -> dict[str, dict[str, DimensionTally]]:
        buckets: dict[str, dict[str, DimensionTally]] = {}
        for low, high in _RATIO_BUCKETS:
            label = (
                "1.00+ (inflated)" if high == float("inf") else f"{low:.2f}-{min(high, 1.0):.2f}"
            )
            buckets[label] = {name: DimensionTally() for name in DIMENSIONS}
            for event in self.events:
                if low <= event.ratio < high:
                    for name, tally in event.dims.items():
                        buckets[label][name].add(tally)
        return buckets

    def by_transform(self) -> dict[str, dict[str, DimensionTally]]:
        transforms: dict[str, dict[str, DimensionTally]] = {}
        for event in self.events:
            for transform in set(event.transforms):
                per_dim = transforms.setdefault(
                    transform, {name: DimensionTally() for name in DIMENSIONS}
                )
                for name, tally in event.dims.items():
                    per_dim[name].add(tally)
        return transforms

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "skipped_lines": self.skipped_lines,
            "aggregate": {name: tally.to_dict() for name, tally in self.aggregate().items()},
            "by_ratio_bucket": {
                label: {name: tally.to_dict() for name, tally in dims.items()}
                for label, dims in self.by_ratio_bucket().items()
            },
            "by_transform": {
                transform: {name: tally.to_dict() for name, tally in dims.items()}
                for transform, dims in self.by_transform().items()
            },
        }


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _tool_texts(messages: Iterable[Any] | None) -> list[str]:
    """Extract tool-result text from OpenAI (role=tool) and Anthropic blocks."""

    out: list[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if msg.get("role") == "tool":
            out.append(_to_text(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    out.append(_to_text(block.get("content")))
    return [text for text in out if text]


def _all_text(node: Any) -> Iterator[str]:
    """Yield every string leaf in a message structure (survival haystack)."""

    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _all_text(value)
    elif isinstance(node, list):
        for item in node:
            yield from _all_text(item)


def extract_probe_targets(text: str) -> dict[str, set[str]]:
    """Extract probe targets per dimension from original tool-result text."""

    targets: dict[str, set[str]] = {name: set() for name in DIMENSIONS}
    targets["numerics"].update(
        match for match in _NUMERIC_RE.findall(text) if len(match) >= _MIN_TARGET_LEN
    )
    for pattern in (_URL_RE, _PATH_RE, _HEX_RE, _UUID_RE):
        targets["artifacts"].update(
            match for match in pattern.findall(text) if len(match) >= _MIN_TARGET_LEN
        )
    for line in text.splitlines():
        stripped = line.strip()
        if len(stripped) >= _MIN_TARGET_LEN and is_error_content(stripped):
            targets["errors"].add(stripped[:_ERROR_LINE_PREFIX_LEN])
    return targets


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub(" ", text).strip()


def _target_survives(dimension: str, value: str, haystack: str, normalized_haystack: str) -> bool:
    if value in haystack:
        return True
    normalized_value = _normalize(value)
    if normalized_value and normalized_value in normalized_haystack:
        return True
    if dimension == "errors":
        # Format conversions drop JSON key prefixes ('"msg": "Error..."'
        # becomes a bare CSV/KV cell); the error substance is what matters.
        _, _, remainder = normalized_value.partition(" ")
        if len(remainder) >= _MIN_TARGET_LEN and remainder in normalized_haystack:
            return True
    if dimension == "numerics":
        # Format conversions (JSON -> table) separate key from value; count the
        # probe as retained only when both still appear.
        match = _NUMERIC_SPLIT_RE.match(value)
        if match:
            key, number = match.groups()
            normalized_key = _normalize(key)
            if (
                normalized_key
                and normalized_key in normalized_haystack
                and re.search(rf"\b{re.escape(number)}\b", normalized_haystack)
            ):
                return True
    return False


def probe_event(record: dict[str, Any]) -> EventProbeResult | None:
    """Score one recorded compression event; None if it cannot be scored."""

    tokens_before = record.get("tokens_before")
    tokens_after = record.get("tokens_after")
    if not isinstance(tokens_before, (int, float)) or not isinstance(tokens_after, (int, float)):
        return None
    if tokens_before <= 0:
        return None

    original_text = "\n".join(_tool_texts(record.get("original_messages")))
    compressed_text = "\n".join(_all_text(record.get("compressed_messages")))
    normalized_compressed = _normalize(compressed_text)
    has_marker = bool(_CCR_MARKER_RE.search(compressed_text))

    dims: dict[str, DimensionTally] = {}
    for name, values in extract_probe_targets(original_text).items():
        tally = DimensionTally(total=len(values))
        for value in values:
            if _target_survives(name, value, compressed_text, normalized_compressed):
                tally.retained += 1
            elif has_marker:
                tally.recoverable += 1
        dims[name] = tally

    transforms = [str(item) for item in record.get("transforms_applied") or []]
    return EventProbeResult(
        request_id=str(record.get("request_id", "")),
        ratio=float(tokens_after) / float(tokens_before),
        transforms=transforms,
        dims=dims,
    )


def run_probes(recordings_dir: Path) -> ProbeReport:
    """Probe every event in every ``*.jsonl`` recording under a directory."""

    report = ProbeReport()
    for path in sorted(recordings_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    report.skipped_lines += 1
                    continue
                result = probe_event(record) if isinstance(record, dict) else None
                if result is None:
                    report.skipped_lines += 1
                    continue
                report.events.append(result)
    return report


def _format_tally(tally: DimensionTally) -> str:
    if tally.total == 0:
        return "n/a (0 targets)"
    retained_pct = 100.0 * tally.retained / tally.total
    recoverable_pct = 100.0 * tally.recoverable / tally.total
    lost_pct = 100.0 * tally.lost / tally.total
    return (
        f"{retained_pct:5.1f}% retained, {recoverable_pct:5.1f}% recoverable, "
        f"{lost_pct:5.1f}% lost ({tally.total} targets)"
    )


def render_report(report: ProbeReport) -> str:
    """Render a human-readable retention report."""

    lines = [
        f"Probed {len(report.events)} compression events"
        + (f" ({report.skipped_lines} lines skipped)" if report.skipped_lines else ""),
        "",
        "Aggregate retention:",
    ]
    for name, tally in report.aggregate().items():
        lines.append(f"  {name:<10} {_format_tally(tally)}")

    lines += ["", "By compression ratio (tokens_after / tokens_before):"]
    for label, dims in report.by_ratio_bucket().items():
        if all(tally.total == 0 for tally in dims.values()):
            continue
        lines.append(f"  ratio {label}:")
        for name, tally in dims.items():
            lines.append(f"    {name:<10} {_format_tally(tally)}")

    by_transform = report.by_transform()
    if by_transform:
        lines += ["", "By transform:"]
        for transform in sorted(by_transform):
            lines.append(f"  {transform}:")
            for name, tally in by_transform[transform].items():
                lines.append(f"    {name:<10} {_format_tally(tally)}")

    return "\n".join(lines)
