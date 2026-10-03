"""ast-grep interceptor: replace verbose Read outputs with function-level outlines.

Matches Claude Code's `Read` tool (and equivalent) when the file is code and
the output is large enough to benefit. Invokes ast-grep to locate top-level
function and class definitions and emits a compact outline: each signature
followed by an elided body marker. Falls back to the original text if
ast-grep isn't available, the extension isn't supported, or there are fewer
than three definitions to outline.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from headroom import binaries
from headroom._subprocess import run
from headroom.proxy import runtime_env
from headroom.proxy.project_context import get_registered_cwd

from . import base

logger = logging.getLogger(__name__)


# Latency floor: below this size, the subprocess cost of running ast-grep
# isn't worth the tiny win. It is NOT a semantic threshold — the framework
# rejects any rewrite that doesn't actually shrink tokens, so we don't need
# a "big enough to matter" check here, only a "big enough to justify the
# fork()" check. Read live (not as a module constant) so a hot-reload or a
# reused proxy re-synced by ``headroom wrap`` takes effect without a restart.
def _min_chars_to_rewrite() -> int:
    try:
        return int(runtime_env.getenv("HEADROOM_INTERCEPT_READ_MIN_CHARS", "500"))
    except (TypeError, ValueError):
        return 500


# Tool_input keys that indicate the model targeted a specific line range;
# outlining would frustrate that intent and likely cause a re-read.
# Provenance of the keys we recognize:
#   offset / limit        — Claude Code's Read tool (pagination by line).
#   line_range            — Cursor / VS Code Copilot read_file with explicit range.
#   start_line / end_line — Aider, Continue, some MCP filesystem servers.
#   ranges                — OpenAI Codex file tools (list of [start,end] pairs).
_RANGE_KEYS = ("offset", "limit", "line_range", "start_line", "end_line", "ranges")

# ast-grep --lang is passed these values; only extensions with a stable
# grammar are included.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
}

# Top-level declaration patterns per language. We emit the signature line
# of whatever ast-grep matches here, so any pattern that anchors on a
# declaration's starting line works.
_PATTERNS: dict[str, list[str]] = {
    "python": ["def $NAME", "class $NAME", "async def $NAME"],
    "typescript": ["function $NAME", "class $NAME"],
    "tsx": ["function $NAME", "class $NAME"],
    "javascript": ["function $NAME", "class $NAME"],
    "jsx": ["function $NAME", "class $NAME"],
    "go": ["func $NAME"],
    "rust": ["fn $NAME", "struct $NAME", "enum $NAME"],
    "java": ["class $NAME", "interface $NAME"],
    "ruby": ["def $NAME", "class $NAME"],
    "c": ["$RET $NAME($$$ARGS) { $$$BODY }"],
    "cpp": ["$RET $NAME($$$ARGS) { $$$BODY }"],
}

OUTLINE_MARKER = "    # ... (body elided by Headroom; Read a specific line range to see it)\n"

# Per-client banner signatures -- add an entry only once a client's exact
# banner text is confirmed, never a generic keyword match.
_TRUNCATION_SIGNATURES: tuple[re.Pattern[str], ...] = (
    # Claude Code: "[Truncated: PARTIAL view -- <path>: showing lines A-B of
    # T total (...). Call Read with offset=N to see more.]"
    re.compile(
        r"\[\s*truncated\s*:\s*partial\s+view\b"
        r"[^\[\]]*?"
        r"showing\s+lines?\s+(?P<start_line>\d+)\s*[-–]\s*(?P<end_line>\d+)"
        r"\s+of\s+(?P<total_lines>\d+)\s+total"
        r"[^\[\]]*\]",
        re.IGNORECASE,
    ),
)


def _is_plausible_truncation_range(
    start_line: int, end_line: int, total_lines: int, source_line_count: int
) -> bool:
    # end_line == total_lines means the whole file was shown, not truncated.
    if start_line < 1 or end_line < start_line or end_line >= total_lines:
        return False
    return end_line <= source_line_count


def _detect_truncation(source: str) -> tuple[int, int] | None:
    """Return (end_line, total_lines) if `source` carries a recognized,
    internally-consistent upstream truncation banner, else None."""
    source_line_count = len(source.splitlines())
    for pattern in _TRUNCATION_SIGNATURES:
        for m in pattern.finditer(source):
            start_line = int(m.group("start_line"))
            end_line = int(m.group("end_line"))
            total_lines = int(m.group("total_lines"))
            if _is_plausible_truncation_range(start_line, end_line, total_lines, source_line_count):
                return end_line, total_lines
    return None


class ReadVerificationResult(Enum):
    """Client-independent fallback for `_detect_truncation`'s banner regex:
    compares tool_output against the real file on disk instead of parsing
    client-specific prose. Only used when the banner regex finds nothing.

    Always UNKNOWN on Windows (no O_NOFOLLOW/dir_fd) -- never falls back to
    a less-safe read."""

    COMPLETE = "complete"
    TRUNCATED = "truncated"
    # Unresolvable path, unreadable file, or mismatched content — never guess.
    UNKNOWN = "unknown"


def _verify_truncation_on_disk_enabled() -> bool:
    # Live read (not a module constant), matching _min_chars_to_rewrite()'s
    # hot-reload behavior.
    return runtime_env.getenv("HEADROOM_VERIFY_TRUNCATION_ON_DISK", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _max_disk_verify_bytes() -> int:
    # Live read, same hot-reload pattern as _min_chars_to_rewrite(). 5 MB
    # comfortably covers real source files while bounding worst-case read
    # time/memory for the disk-verify fallback.
    try:
        return int(runtime_env.getenv("HEADROOM_VERIFY_TRUNCATION_MAX_BYTES", "5000000"))
    except (TypeError, ValueError):
        return 5_000_000


def _dir_fd_walk_supported() -> bool:
    # hasattr, not bare os.O_NOFOLLOW/os.O_DIRECTORY refs -- missing on
    # Windows, would AttributeError at import time otherwise.
    return (
        hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY") and os.open in os.supports_dir_fd
    )


def _on_event_loop_thread() -> bool:
    """True only on a thread with a running asyncio loop (the request
    coroutine) -- false on a plain ThreadPoolExecutor worker. Lets the
    blocking read refuse itself if ever reached directly from the loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _open_regular_file_under_root(
    file_path: str, resolved_root: Path, max_bytes: int
) -> str | None:
    """Race-safe read of `file_path` confined under `resolved_root`.

    Walks from an fd opened on `resolved_root` using dir_fd-relative,
    O_NOFOLLOW opens at every hop -- no full path is ever resolved then
    reopened, so there's no TOCTOU window for a symlink swap to exploit.
    Refuses if called on the event-loop thread or if this platform lacks
    dir_fd support (notably Windows) -- no less-safe fallback either way.
    """
    if _on_event_loop_thread() or not _dir_fd_walk_supported():
        return None

    # Pure string math -- no I/O, can't itself be raced.
    candidate = (
        file_path if os.path.isabs(file_path) else os.path.join(str(resolved_root), file_path)
    )
    rel = os.path.relpath(os.path.normpath(candidate), str(resolved_root))
    if rel == os.curdir:
        return None  # the root itself, not a file within it
    if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return None  # escapes resolved_root
    segments = [s for s in rel.split(os.sep) if s]
    if not segments or any(s in (os.curdir, os.pardir) for s in segments):
        return None

    try:
        # resolved_root already went through resolve(strict=True) in the
        # caller, so it's symlink-free at that instant -- O_NOFOLLOW here
        # closes the residual window where the anchor itself gets swapped
        # for a symlink between that resolve() and this open().
        current_fd = os.open(str(resolved_root), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError:
        return None

    file_fd = -1
    try:
        for segment in segments[:-1]:
            try:
                next_fd = os.open(
                    segment, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current_fd
                )
            except OSError:
                return None
            os.close(current_fd)
            current_fd = next_fd

        final = segments[-1]
        try:
            # O_NONBLOCK: a FIFO with no writer returns immediately instead
            # of blocking -- no effect once fstat confirms a regular file.
            file_fd = os.open(
                final,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
        except OSError:
            return None

        try:
            st = os.fstat(file_fd)
            if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
                return None
            with os.fdopen(file_fd, "r", encoding="utf-8") as f:
                file_fd = -1  # ownership transferred to the file object
                return f.read()
        except (OSError, UnicodeDecodeError):
            return None
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if current_fd >= 0:
            os.close(current_fd)


def _verify_read_against_disk(
    file_path: str | None,
    received_content: str,
    registered_cwd: str | None,
) -> tuple[ReadVerificationResult, tuple[int, int] | None]:
    """Compare `received_content` against the real file at `file_path`.

    `registered_cwd` is `project_context.get_registered_cwd()`, not a raw
    header -- non-None only when a session token matched a workspace root
    `wrap` itself registered (see workspace_registry.py). Its presence is
    the trust signal; there's no separate boolean to check.

    TRUNCATED requires an exact-prefix match with strictly more on disk —
    a weaker match means the file diverged since the client read it, not
    a provable truncation, so it's UNKNOWN. Returns `(visible_lines,
    total_lines)` alongside TRUNCATED so the header can cite real numbers
    without a second, potentially racy, read.
    """
    if not file_path:
        return ReadVerificationResult.UNKNOWN, None
    if not registered_cwd or not os.path.isabs(registered_cwd):
        return ReadVerificationResult.UNKNOWN, None
    try:
        resolved_root = Path(registered_cwd).resolve(strict=True)
    except OSError:
        return ReadVerificationResult.UNKNOWN, None
    if not resolved_root.is_dir():
        return ReadVerificationResult.UNKNOWN, None
    disk_content = _open_regular_file_under_root(file_path, resolved_root, _max_disk_verify_bytes())
    if disk_content is None:
        return ReadVerificationResult.UNKNOWN, None
    if disk_content == received_content:
        return ReadVerificationResult.COMPLETE, None
    if disk_content.startswith(received_content) and len(disk_content) > len(received_content):
        visible_lines = len(received_content.splitlines())
        total_lines = len(disk_content.splitlines())
        return ReadVerificationResult.TRUNCATED, (visible_lines, total_lines)
    return ReadVerificationResult.UNKNOWN, None


class AstGrepReadOutline:
    """Interceptor that outlines verbose code-file Read outputs."""

    name = "ast-grep"

    def matches(
        self,
        tool_name: str | None,
        tool_input: dict[str, Any],
        tool_output: str,
    ) -> bool:
        if tool_name not in ("Read", "read_file", "view", "cat"):
            return False
        if len(tool_output) < _min_chars_to_rewrite():
            return False
        # Respect explicit line ranges — the model wants those specific lines.
        if any(k in tool_input for k in _RANGE_KEYS):
            return False
        return _detect_lang_from_input(tool_input) is not None

    def transform(
        self,
        tool_name: str | None,
        tool_input: dict[str, Any],
        tool_output: str,
    ) -> str | None:
        lang = _detect_lang_from_input(tool_input)
        if not lang:
            return None
        try:
            exe = binaries.resolve("ast-grep")
        except (binaries.BinaryError, KeyError, OSError) as e:
            # Covers PlatformNotSupported, OfflineError, BinaryFetchError,
            # Sha256Mismatch, unknown-tool KeyError, and FS permission errors.
            # Any of these means the interceptor simply passes through.
            logger.debug("ast-grep unavailable: %s", e)
            return None

        matches = _run_ast_grep(exe, lang, tool_output)
        if not matches:
            return None

        # Banner (cheap, no I/O) wins; disk verification is the opt-in fallback.
        truncation = _detect_truncation(tool_output)
        if truncation is None and _verify_truncation_on_disk_enabled():
            verdict, disk_truncation = _verify_read_against_disk(
                _path_from_input(tool_input),
                tool_output,
                get_registered_cwd(),
            )
            if verdict is ReadVerificationResult.TRUNCATED:
                truncation = disk_truncation
        outline = _build_outline(matches, tool_output, truncation)
        return outline if outline else None

    def progressive_disclosure_key(
        self,
        tool_name: str | None,
        tool_input: dict[str, Any],
    ) -> str | None:
        """Key by file_path so a second Read of the same file passes through."""
        path = _path_from_input(tool_input)
        if path is None:
            # matches() returned True but no recognized path key — the tool
            # may use an unknown key (e.g. some MCP servers use `file`).
            # Without a key, progressive disclosure can't protect against
            # re-outlining; log once for observability.
            logger.debug(
                "ast-grep: no path key in tool_input (keys=%s); progressive disclosure disabled",
                sorted(tool_input.keys()),
            )
        return path


def _detect_lang_from_input(tool_input: dict[str, Any]) -> str | None:
    path = _path_from_input(tool_input)
    if not path:
        return None
    ext = Path(path).suffix.lower()
    return _EXT_TO_LANG.get(ext)


def _path_from_input(tool_input: dict[str, Any]) -> str | None:
    for key in ("file_path", "path", "filePath", "filename"):
        v = tool_input.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _run_ast_grep(
    exe: Path | str,
    lang: str,
    source: str,
) -> list[dict[str, Any]]:
    """Run ast-grep against `source` and return the JSON match records.

    Writes `source` to a tempfile because ast-grep's CLI operates on files.
    """
    all_matches: list[dict[str, Any]] = []
    patterns = _PATTERNS.get(lang, [])
    if not patterns:
        return []

    # Use the canonical extension so ast-grep can pick the right grammar.
    # Write into a private mode-0700 temp dir — /tmp is shared on multi-tenant
    # systems and tool_output is untrusted content.
    ext = next((e for e, L in _EXT_TO_LANG.items() if L == lang), ".txt")
    tmp_dir = Path(tempfile.mkdtemp(prefix="headroom-sg-"))
    try:
        os.chmod(tmp_dir, 0o700)
    except OSError as e:
        # On Windows / restricted FS chmod has no effect, but silently
        # swallowing means a shared-tmp system may leave untrusted content
        # world-readable without any indication. Log so the miss is visible.
        logger.debug("chmod 0700 failed for %s: %s (hardening skipped)", tmp_dir, e)
    tmp_path = tmp_dir / f"src{ext}"
    tmp_path.write_text(source, encoding="utf-8")

    try:
        for pattern in patterns:
            try:
                completed = run(
                    [
                        str(exe),
                        "run",
                        "--pattern",
                        pattern,
                        "--lang",
                        lang,
                        "--json=stream",
                        str(tmp_path),
                    ],
                    capture_output=True,
                    text=True,
                    # ast-grep emits UTF-8 (it matches arbitrary source text);
                    # without this, Windows decodes with cp1252 and raises.
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.debug("ast-grep timed out or failed: %s", e)
                continue
            # rc=0: matches. rc=1: no matches (expected). rc>=2: real error
            # (bad syntax, grammar missing, corrupt binary) — log it so
            # users can diagnose.
            if completed.returncode == 1:
                continue
            if completed.returncode >= 2:
                logger.debug(
                    "ast-grep error (rc=%d, lang=%s, pattern=%r): %s",
                    completed.returncode,
                    lang,
                    pattern,
                    (completed.stderr or "")[:200],
                )
                continue
            lines = [ln.strip() for ln in completed.stdout.splitlines() if ln.strip()]
            parse_failures = 0
            for line in lines:
                try:
                    all_matches.append(json.loads(line))
                except json.JSONDecodeError:
                    parse_failures += 1
            if lines and parse_failures == len(lines):
                logger.warning(
                    "ast-grep produced output but every line failed to parse as JSON "
                    "(rc=0, lang=%s, pattern=%r) — likely version mismatch or corrupt binary",
                    lang,
                    pattern,
                )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return all_matches


def _build_outline(
    matches: list[dict[str, Any]],
    source: str,
    truncation: tuple[int, int] | None = None,
) -> str | None:
    """Build a compact outline from ast-grep matches.

    Emits each definition's signature line + docstring (if next line is a
    string literal) + an elision marker. Matches are sorted by byte offset
    so the outline tracks the original file order.

    `truncation`, if given, is (end_line, total_lines) from an upstream
    truncation banner already present in `source` (e.g. a client's own Read
    token-cap notice). When set, the header states that the input was a
    partial view instead of implying `source` is the whole file.
    """
    lines = source.splitlines(keepends=True)
    outline_chunks: list[str] = []
    seen_starts: set[int] = set()

    matches.sort(key=lambda m: m.get("range", {}).get("byteOffset", {}).get("start", 0))
    for m in matches:
        start = m.get("range", {}).get("start", {})
        line_idx = start.get("line")
        if not isinstance(line_idx, int) or line_idx in seen_starts:
            continue
        seen_starts.add(line_idx)
        if line_idx >= len(lines):
            continue
        signature_line = lines[line_idx].rstrip("\n")
        outline_chunks.append(signature_line + "\n")
        # Best-effort: if the next non-blank line is a docstring, keep it.
        next_idx = line_idx + 1
        while next_idx < len(lines) and not lines[next_idx].strip():
            next_idx += 1
        if next_idx < len(lines):
            nl = lines[next_idx].lstrip()
            if nl.startswith(('"""', "'''", "/**", "//", "#")):
                outline_chunks.append(lines[next_idx])
        outline_chunks.append(OUTLINE_MARKER)

    if not outline_chunks:
        return None

    if truncation:
        end_line, total_lines = truncation
        header = (
            "[headroom: outlined by ast-grep — "
            f"{len(seen_starts)} definition(s) in the visible portion; "
            f"input was truncated upstream (showing through line {end_line} of {total_lines} total). "
            "Bodies elided. Re-read remaining lines to see more.]\n"
        )
    else:
        header = (
            "[headroom: outlined by ast-grep — "
            f"{len(seen_starts)} definition(s); "
            "bodies elided. Re-read the file with a line range to see a specific body.]\n"
        )
    return header + "".join(outline_chunks)


base.register(AstGrepReadOutline())
