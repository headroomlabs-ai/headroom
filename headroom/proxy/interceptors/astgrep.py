"""ast-grep interceptor: replace verbose Read outputs with function-level outlines.

Matches Claude Code's `Read` tool (and equivalent) when the file is code and
the output is large enough to benefit. Invokes ast-grep to locate top-level
function and class definitions and emits a compact outline: each signature
followed by an elided body marker. Falls back to the original text if
ast-grep isn't available, the extension isn't supported, or there are fewer
than three definitions to outline.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from headroom import binaries
from headroom.proxy import runtime_env

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

        outline = _build_outline(matches, tool_output)
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
                completed = subprocess.run(
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


def _build_outline(matches: list[dict[str, Any]], source: str) -> str | None:
    """Build a compact outline from ast-grep matches.

    Emits each definition's signature line + docstring (if next line is a
    string literal) + an elision marker. Matches are sorted by byte offset
    so the outline tracks the original file order.
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
    header = (
        "[headroom: outlined by ast-grep — "
        f"{len(seen_starts)} definition(s); "
        "bodies elided. Re-read the file with a line range to see a specific body.]\n"
    )
    return header + "".join(outline_chunks)


base.register(AstGrepReadOutline())
