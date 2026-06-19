"""Codex transcript audit — read-pattern analysis for shell-based clients.

Codex has no structured Read tool: it reads files through shell commands
(``cat``, ``sed -n 'a,bp'``, ``head``/``tail``, ``nl``) — frequently
wrapped by rtk (``rtk read <file>``, ``rtk proxy <cmd>``). This module
classifies ``exec_command`` calls in Codex session transcripts
(``~/.codex/sessions/**/*.jsonl``) and measures the read pattern so the
read-maturation mechanism can be sized for Codex workloads.

Findings on the development corpus (2026-06-10, 144 sessions, 50MB of
tool output): reads are 51.9% of output bytes, 66% of reads are partial
slices, 55% of reads target an already-read path, and hot files are read
hundreds of times per corpus — the "slice grinder" profile. 78% of read
outputs clear the 2KB maturation floor.
"""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Programs whose output is file content. "read" is rtk's read command.
_READ_PROGS = frozenset({"cat", "sed", "head", "tail", "nl", "bat", "more", "read"})
_SEARCH_PROGS = frozenset({"rg", "grep", "ugrep", "ag", "fd", "find"})
_BUILD_PROGS = frozenset({"python", "python3", "pytest", "cargo", "npm", "make", "uv", "ruff"})
_RANGE_RE = re.compile(r"^\d+([,:-]\d+)?p?$")

MATURE_FLOOR = 2048  # ReadMaturationConfig.min_size_bytes


@dataclass
class CodexAuditReport:
    """Aggregated Codex read-pattern results."""

    sessions: int = 0
    exec_calls: int = 0
    calls_by_category: dict[str, int] = field(default_factory=dict)
    bytes_by_category: dict[str, int] = field(default_factory=dict)
    total_output_bytes: int = 0
    read_calls: int = 0
    read_bytes: int = 0
    reads_partial: int = 0
    rereads_same_path: int = 0
    distinct_files_read: int = 0
    reads_over_floor: int = 0
    read_size_p50: int = 0
    read_size_p90: int = 0
    top_reread_files: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def strip_wrappers(cmd: str) -> str:
    """Peel rtk wrappers: ``rtk <cmd>`` and ``rtk proxy <cmd>``."""
    c = cmd.strip()
    while True:
        if c.startswith("rtk "):
            c = c[4:].strip()
            continue
        if c.startswith("proxy "):
            c = c[6:].strip()
            continue
        return c


def classify_command(cmd: str, workdir: str = "") -> tuple[str, str | None, bool]:
    """Classify a shell command: (category, file_path|None, is_partial).

    Categories: read, search, git, edit, build/test, compound, other.
    For reads, the path is resolved against ``workdir`` when relative.
    """
    c = strip_wrappers(cmd)
    if "apply_patch" in c:
        return "edit", None, False
    try:
        toks = shlex.split(c)
    except ValueError:
        toks = c.split()
    if not toks:
        return "other", None, False
    prog = toks[0].rsplit("/", 1)[-1]

    if prog in _READ_PROGS:
        candidates = [
            t
            for t in toks[1:]
            if not t.startswith("-") and ("/" in t or "." in t.rsplit("/", 1)[-1])
        ]
        # Range tokens like 1,200p (sed) are not paths.
        candidates = [t for t in candidates if not _RANGE_RE.match(t.strip("'\""))]
        fpath = candidates[0] if candidates else None
        if fpath and workdir and not fpath.startswith("/"):
            fpath = f"{workdir.rstrip('/')}/{fpath}"
        partial = (
            prog in ("sed", "head", "tail")
            or any(_RANGE_RE.match(t.strip("'\"")) for t in toks[1:])
            or "--lines" in c
        )
        return "read", fpath, partial
    if prog in _SEARCH_PROGS:
        return "search", None, False
    if prog == "git":
        return "git", None, False
    if prog in _BUILD_PROGS:
        return "build/test", None, False
    if "&&" in cmd or "|" in cmd:
        for part in re.split(r"&&|\|", cmd):
            cat, fpath, partial = classify_command(part, workdir)
            if cat == "read":
                return cat, fpath, partial
        return "compound", None, False
    return "other", None, False


def _output_text(payload: dict) -> str:
    out = payload.get("output", "")
    if isinstance(out, dict):
        out = out.get("output", "") or str(out)
    return str(out)


def audit_codex(root: Path) -> CodexAuditReport:
    """Audit all Codex ``*.jsonl`` transcripts under ``root``."""
    r = CodexAuditReport()
    calls: Counter[str] = Counter()
    cat_bytes: Counter[str] = Counter()
    read_sizes: list[int] = []
    per_file_reads: Counter[str] = Counter()

    for path in sorted(root.glob("**/*.jsonl")):
        pending: dict[str, str] = {}
        seen_paths: set[str] = set()
        saw_lines = False
        try:
            with path.open(errors="replace") as f:
                for raw in f:
                    try:
                        line = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    saw_lines = True
                    pl = line.get("payload") or {}
                    t = pl.get("type")
                    if t == "function_call" and pl.get("name") == "exec_command":
                        try:
                            args = json.loads(pl.get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        cat, fpath, partial = classify_command(
                            args.get("cmd", ""), args.get("workdir", "")
                        )
                        r.exec_calls += 1
                        calls[cat] += 1
                        pending[pl.get("call_id", "")] = cat
                        if cat == "read":
                            r.read_calls += 1
                            r.reads_partial += partial
                            if fpath:
                                if fpath in seen_paths:
                                    r.rereads_same_path += 1
                                seen_paths.add(fpath)
                                per_file_reads[fpath] += 1
                    elif t == "function_call_output":
                        size = len(_output_text(pl).encode("utf-8", errors="replace"))
                        r.total_output_bytes += size
                        cat = pending.get(pl.get("call_id", ""), "untracked")
                        cat_bytes[cat] += size
                        if cat == "read":
                            r.read_bytes += size
                            read_sizes.append(size)
                            if size >= MATURE_FLOOR:
                                r.reads_over_floor += 1
        except OSError:
            continue
        if saw_lines:
            r.sessions += 1

    r.calls_by_category = dict(calls.most_common())
    r.bytes_by_category = dict(cat_bytes.most_common())
    r.distinct_files_read = len(per_file_reads)
    if read_sizes:
        rs = sorted(read_sizes)
        r.read_size_p50 = rs[len(rs) // 2]
        r.read_size_p90 = rs[int(len(rs) * 0.9)]
    r.top_reread_files = [
        (f.rsplit("/", 1)[-1], n) for f, n in per_file_reads.most_common(5) if n > 1
    ]
    return r


def render_codex_text(r: CodexAuditReport) -> str:
    """Human-readable Codex audit summary."""
    total = r.total_output_bytes or 1
    out: list[str] = []
    out.append("── codex read-pattern audit ──")
    out.append(f"  sessions: {r.sessions}, exec_command calls: {r.exec_calls}")
    out.append(f"  output bytes by category ({total / 1e6:.1f}MB total):")
    for cat, b in r.bytes_by_category.items():
        out.append(f"    {cat:<12} {b / 1e6:>6.2f}MB  {100 * b / total:.1f}%")
    rc = r.read_calls or 1
    out.append(
        f"  reads: {r.read_calls} ({100 * r.reads_partial / rc:.0f}% partial slices); "
        f"re-reads of same path: {r.rereads_same_path} ({100 * r.rereads_same_path / rc:.0f}%)"
    )
    out.append(
        f"  distinct files read: {r.distinct_files_read}; read size p50={r.read_size_p50}B "
        f"p90={r.read_size_p90}B; ≥{MATURE_FLOOR}B: {r.reads_over_floor} "
        f"({100 * r.reads_over_floor / rc:.0f}%)"
    )
    if r.top_reread_files:
        out.append(f"  most re-read files: {r.top_reread_files}")
    return "\n".join(out)
