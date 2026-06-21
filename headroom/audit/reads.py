"""Read-opportunity audit over local Claude Code transcripts.

Measures, from REAL session data, the addressable bytes for each Read
compression mechanism — so defaults are set from traffic, not theory.
Run it on a deployment's transcripts before tuning anything:

    headroom audit-reads
    headroom audit-reads --path /path/to/projects --format json

Read-only: streams ``<path>/**/*.jsonl`` (Claude Code session transcripts)
and never modifies anything.

What it sizes, per mechanism:

- **identical repeat** — a later Read byte-identical to an earlier Read of
  the same file. (A dedup mechanism for this was prototyped and removed:
  it measured 0.1% of Read bytes on real traffic. If this number is
  material on YOUR traffic, the implementation lives in git history —
  see the feat/compression-extraction branch.)
- **subset containment** — a later partial Read contained in an earlier
  full Read of the same file.
- **write-readback** — a Read whose content echoes a prior Write input.
- **stale** — Reads of files later edited (read_lifecycle's stale class;
  mostly freeze-blocked in production, unlockable at cache-death).
- **line-number scaffolding** — `cat -n` prefix bytes inside Read output.
- **context residency** — how many assistant turns each Read stays in
  context (the multiplier on its prefix-cache read cost; the case for
  compress-before-cache-entry).
- **cache-death windows** — inter-message gaps exceeding the provider
  cache TTL (free recompression moments).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

MIN_SIZE = 512  # matches ReadLifecycleConfig.min_size_bytes
_LINE_NUM_RE = re.compile(r"^\s*\d+\t", re.M)

_LOCK_GENERATED = (
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "cargo.lock",
    "go.sum",
    "poetry.lock",
    "uv.lock",
    "gemfile.lock",
    "composer.lock",
)
_SOURCE_EXT = (
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".rs",
    ".go",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".rb",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".zsh",
)
_DATA_EXT = (".json", ".jsonl", ".csv", ".yaml", ".yml", ".toml", ".xml")
_DOC_EXT = (".md", ".rst", ".txt")

_MUTATING_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")


@dataclass
class ReadAuditReport:
    """Aggregated audit results. All byte figures are UTF-8 bytes of
    tool_result content; tokens ≈ bytes / 4."""

    sessions: int = 0
    files_skipped: int = 0
    tool_bytes: dict[str, int] = field(default_factory=dict)
    read_calls: int = 0
    read_bytes: int = 0
    read_calls_small: int = 0
    dedup_identical_calls: int = 0
    dedup_identical_bytes: int = 0
    subset_calls: int = 0
    subset_bytes: int = 0
    write_readback_calls: int = 0
    write_readback_bytes: int = 0
    stale_calls: int = 0
    stale_bytes: int = 0
    linenum_overhead_bytes: int = 0
    class_bytes: dict[str, int] = field(default_factory=dict)
    residency_median: int = 0
    residency_p90: int = 0
    residency_mean: float = 0.0
    gaps_over_5m: int = 0
    gaps_over_1h: int = 0
    sessions_with_gap: int = 0
    reads_per_file_max_median: int = 0
    reads_per_file_max: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _classify_path(p: str) -> str:
    low = p.lower()
    name = low.rsplit("/", 1)[-1]
    if name in _LOCK_GENERATED or "/node_modules/" in low or "/dist/" in low or ".min." in name:
        return "lock/generated/vendored"
    if name.endswith(_SOURCE_EXT):
        return "source code"
    if name.endswith(_DOC_EXT):
        return "docs/text"
    if name.endswith(_DATA_EXT):
        return "data/config"
    return "other"


def _block_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _parse_ts(line: dict) -> float | None:
    ts = line.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class _Agg:
    def __init__(self) -> None:
        self.report = ReadAuditReport()
        self.tool_bytes: dict[str, int] = defaultdict(int)
        self.class_bytes: dict[str, int] = defaultdict(int)
        self.residency: list[int] = []
        self.reads_per_file_max: list[int] = []


def _audit_session(path: Path, agg: _Agg) -> None:
    r = agg.report
    tool_meta: dict[str, tuple[str, dict]] = {}
    file_reads: dict[str, list[tuple[str, str]]] = defaultdict(list)
    file_writes: dict[str, list[str]] = defaultdict(list)
    read_events: list[tuple[str, int, int, bool]] = []  # (file, size, at, deduped)
    edit_files_at: list[tuple[int, str]] = []
    assistant_idx = 0
    prev_ts: float | None = None
    had_gap = False

    with path.open(errors="replace") as f:
        for raw in f:
            try:
                line = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg = line.get("message") or {}
            role = msg.get("role")
            content = msg.get("content")

            ts = _parse_ts(line)
            if ts is not None and prev_ts is not None:
                gap = ts - prev_ts
                if gap > 3600:
                    r.gaps_over_1h += 1
                    r.gaps_over_5m += 1
                    had_gap = True
                elif gap > 300:
                    r.gaps_over_5m += 1
                    had_gap = True
            if ts is not None:
                prev_ts = ts

            if role == "assistant" and isinstance(content, list):
                assistant_idx += 1
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        name = b.get("name", "")
                        inp = b.get("input") or {}
                        tool_meta[b.get("id", "")] = (name, inp)
                        fp = inp.get("file_path") or inp.get("path") or ""
                        if name in _MUTATING_TOOLS and fp:
                            edit_files_at.append((assistant_idx, fp))
                            if name == "Write":
                                file_writes[fp].append(str(inp.get("content", "")))

            if role == "user" and isinstance(content, list):
                for b in content:
                    if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                        continue
                    tid = b.get("tool_use_id", "")
                    name, inp = tool_meta.get(tid, ("", {}))
                    text = _block_text(b.get("content"))
                    size = len(text.encode("utf-8", errors="replace"))
                    agg.tool_bytes[name or "unknown"] += size
                    if name != "Read":
                        continue

                    r.read_calls += 1
                    r.read_bytes += size
                    fp = inp.get("file_path") or inp.get("path") or ""
                    is_partial = inp.get("offset") is not None or inp.get("limit") is not None
                    if size < MIN_SIZE:
                        r.read_calls_small += 1
                    agg.class_bytes[_classify_path(fp)] += size
                    r.linenum_overhead_bytes += sum(
                        len(m.group(0)) for m in _LINE_NUM_RE.finditer(text)
                    )

                    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
                    deduped = False
                    if size >= MIN_SIZE and fp:
                        prior = file_reads[fp]
                        if any(ph == h for ph, _ in prior):
                            r.dedup_identical_bytes += size
                            r.dedup_identical_calls += 1
                            deduped = True
                        elif (
                            is_partial
                            and text
                            and any(text in pc for _, pc in prior if len(pc) > len(text))
                        ):
                            r.subset_bytes += size
                            r.subset_calls += 1
                        elif any(
                            text.strip() and w.strip() and text.strip() in w
                            for w in file_writes.get(fp, [])
                        ):
                            r.write_readback_bytes += size
                            r.write_readback_calls += 1
                    if fp:
                        file_reads[fp].append((h, text))
                    read_events.append((fp, size, assistant_idx, deduped))

    for fp, size, at, deduped in read_events:
        if size >= MIN_SIZE and fp and not deduped:
            if any(idx > at and ef == fp for idx, ef in edit_files_at):
                r.stale_bytes += size
                r.stale_calls += 1
        agg.residency.append(max(0, assistant_idx - at))

    per_file: dict[str, int] = defaultdict(int)
    for fp, _, _, _ in read_events:
        if fp:
            per_file[fp] += 1
    if per_file:
        agg.reads_per_file_max.append(max(per_file.values()))
    if had_gap:
        r.sessions_with_gap += 1
    r.sessions += 1


def audit_reads(root: Path) -> ReadAuditReport:
    """Audit all ``*.jsonl`` transcripts under ``root``."""
    agg = _Agg()
    for p in sorted(root.glob("**/*.jsonl")):
        try:
            _audit_session(p, agg)
        except OSError:
            agg.report.files_skipped += 1

    r = agg.report
    r.tool_bytes = dict(sorted(agg.tool_bytes.items(), key=lambda kv: -kv[1]))
    r.class_bytes = dict(sorted(agg.class_bytes.items(), key=lambda kv: -kv[1]))
    if agg.residency:
        rt = sorted(agg.residency)
        r.residency_median = rt[len(rt) // 2]
        r.residency_p90 = rt[int(len(rt) * 0.9)]
        r.residency_mean = sum(rt) / len(rt)
    if agg.reads_per_file_max:
        m = sorted(agg.reads_per_file_max)
        r.reads_per_file_max_median = m[len(m) // 2]
        r.reads_per_file_max = m[-1]
    return r


def _fmt(b: int) -> str:
    if b > 1_000_000:
        return f"{b / 1_000_000:.1f}MB (~{b // 4000}K tok)"
    return f"{b / 1000:.0f}KB (~{b // 4000}K tok)"


def render_text(r: ReadAuditReport) -> str:
    """Render the report as the human-readable summary."""
    total_tool = sum(r.tool_bytes.values()) or 1
    rb = r.read_bytes or 1
    out: list[str] = []
    out.append(f"sessions analyzed: {r.sessions}")
    if r.files_skipped:
        out.append(f"files skipped (unreadable): {r.files_skipped}")
    out.append("\n── tool_result bytes by tool ──")
    for name, b in list(r.tool_bytes.items())[:10]:
        out.append(f"  {name or '?':<24} {_fmt(b):<28} {100 * b / total_tool:.1f}%")
    out.append("\n── Read opportunity sizing (share of Read bytes) ──")
    out.append(f"  Read calls: {r.read_calls}  ({r.read_calls_small} below {MIN_SIZE}B floor)")
    out.append(
        f"  Read bytes: {_fmt(r.read_bytes)}  = {100 * r.read_bytes / total_tool:.1f}% of all tool bytes"
    )
    rows = [
        ("identical repeat", r.dedup_identical_calls, r.dedup_identical_bytes),
        ("subset containment", r.subset_calls, r.subset_bytes),
        ("write-readback", r.write_readback_calls, r.write_readback_bytes),
        ("stale (edit after read)", r.stale_calls, r.stale_bytes),
    ]
    for label, calls, b in rows:
        out.append(f"  {label:<32} {calls:>5} calls  {_fmt(b):<28} {100 * b / rb:.1f}%")
    out.append(
        f"  {'line-number scaffolding':<32} {'':>11}  {_fmt(r.linenum_overhead_bytes):<28} "
        f"{100 * r.linenum_overhead_bytes / rb:.1f}%"
    )
    out.append("\n── Read bytes by file class ──")
    for cls, b in r.class_bytes.items():
        out.append(f"  {cls:<24} {_fmt(b):<28} {100 * b / rb:.1f}%")
    out.append("\n── context residency (assistant turns after each Read) ──")
    out.append(f"  median {r.residency_median}, p90 {r.residency_p90}, mean {r.residency_mean:.0f}")
    out.append("\n── cache-death windows ──")
    out.append(
        f"  gaps >5min: {r.gaps_over_5m} ({r.gaps_over_1h} of them >1h); "
        f"sessions with ≥1 gap: {r.sessions_with_gap}/{r.sessions}"
    )
    out.append(
        f"  max reads of one file per session: median {r.reads_per_file_max_median}, "
        f"max {r.reads_per_file_max}"
    )
    return "\n".join(out)
