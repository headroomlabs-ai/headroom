"""AST-walking contract tests locking memory-system invariants.

These tests introspect the four handler modules' source ASTs to assert
two structural properties that, if regressed, would silently re-
introduce the bug classes this PR fixed:

(a) **No raw memory-gate conjunction**: every memory injection block
    must be gated by ``MemoryDecision``, not by an inline
    ``if self.memory_handler and memory_user_id`` conjunction. The
    raw conjunction is what allowed sites 1/2/3 to silently ignore
    ``x-headroom-bypass: true``.

(b) **No memory writes to system/instructions**: memory injection
    must target user-message-tail / body["input"] / messages, never
    ``body["instructions"]`` or system content. Pre-PR-this the WS
    handler was the lone outlier writing to instructions; the AST
    check ensures it doesn't sneak back.

The checks are static — no handler is invoked. They run in
milliseconds and catch future regressions at PR-review time.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

HANDLER_FILES = [
    Path("headroom/proxy/handlers/anthropic.py"),
    Path("headroom/proxy/handlers/openai.py"),
    Path("headroom/proxy/handlers/gemini.py"),
    Path("headroom/proxy/handlers/batch.py"),
]


# ── Invariant A — no raw memory-gate conjunction ──────────────────────


def _file_contains_raw_memory_gate(file_path: Path) -> list[tuple[int, str]]:
    """Find ``if (self.)memory_handler and memory_user_id`` raw
    conjunctions in the file. Returns list of (line, snippet).

    The acceptable replacements are ``if memory_decision.inject:`` or
    ``if (responses|ws)_memory_decision.inject:``.

    AST-walking the conditional itself is complex (BoolOp + Attribute),
    so we use a regex to scan source lines. False positives are caught
    by the test author at write time — this is a one-line invariant.
    """
    text = file_path.read_text(encoding="utf-8")
    # Match "if self.memory_handler and memory_user_id" but NOT inside
    # the helper that defines the decision (which references both names
    # for documentation purposes) — we only care about handler logic.
    pattern = re.compile(r"^\s*if\s+self\.memory_handler\s+and\s+memory_user_id\b")
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        if pattern.match(line):
            hits.append((i, line.rstrip()))
    return hits


def test_no_raw_memory_handler_gate_in_handlers() -> None:
    """Pre-PR-this, sites 1/2/3 used ``if self.memory_handler and
    memory_user_id:`` as the memory-injection gate — silently
    ignoring bypass. After PR-this, every site routes through
    ``MemoryDecision.decide(...)`` and gates on
    ``memory_decision.inject``. This test ensures the raw conjunction
    cannot return without explicit review."""
    offenders = []
    for f in HANDLER_FILES:
        offenders.extend([(f, ln, src) for (ln, src) in _file_contains_raw_memory_gate(f)])
    if offenders:
        formatted = "\n".join(f"  {f.name}:{ln}  {src!r}" for f, ln, src in offenders)
        pytest.fail(
            f"{len(offenders)} handler site(s) use the pre-PR raw memory "
            "gate `if self.memory_handler and memory_user_id`:\n"
            f"{formatted}\n\n"
            "Replace with `MemoryDecision.decide(...)` + "
            "`if memory_decision.inject:`. See PR for the canonical pattern."
        )


# ── Invariant B — memory never writes to system/instructions ─────────


_FORBIDDEN_SYSTEM_WRITES = (
    # Direct mutation of cache-hot-zone system fields by the memory
    # path. The patterns below are exact assignment forms — they match
    # the pre-PR-this WS bug at openai.py:3517.
    re.compile(r'ws_response_body\["instructions"\]\s*='),
    re.compile(r'response_body\["instructions"\]\s*='),
    re.compile(r'body\["instructions"\]\s*='),
    re.compile(r'body\["system"\]\s*='),
)


def _line_is_memory_related(line: str) -> bool:
    """Heuristic: a line that contains ``memory_context`` or
    ``memory_inject`` is in the memory-injection code path."""
    return "memory_context" in line or "memory_inject" in line


def _find_system_writes_in_memory_context(file_path: Path) -> list[tuple[int, str]]:
    """Find lines that both:
    - Look like a write to body["instructions"] / body["system"]
    - Live within ~10 lines of a ``memory_context`` reference

    This is a windowed-context check — we don't want false positives
    from unrelated instructions-writes (e.g. tool-result handling).
    """
    text = file_path.read_text(encoding="utf-8").splitlines()
    memory_line_indices = [i for i, line in enumerate(text) if _line_is_memory_related(line)]

    hits = []
    for i, line in enumerate(text):
        if not any(p.search(line) for p in _FORBIDDEN_SYSTEM_WRITES):
            continue
        # Within 10 lines of any memory-related line?
        if any(abs(i - mi) <= 10 for mi in memory_line_indices):
            hits.append((i + 1, line.strip()))
    return hits


def test_memory_never_writes_to_system_or_instructions() -> None:
    """Pre-PR-this, the WS handler wrote memory context to
    ``ws_response_body["instructions"]`` — the system / cache-hot-zone
    field. That mutated the prefix cache bytes on every turn. All
    other sites route to user-message tail / body["input"]. This
    test asserts memory_context-related code paths never write to a
    forbidden system-field assignment."""
    offenders = []
    for f in HANDLER_FILES:
        offenders.extend([(f, ln, src) for (ln, src) in _find_system_writes_in_memory_context(f)])
    if offenders:
        formatted = "\n".join(f"  {f.name}:{ln}  {src!r}" for f, ln, src in offenders)
        pytest.fail(
            f"{len(offenders)} suspected memory→system write(s):\n"
            f"{formatted}\n\n"
            "Memory must append to user-message tail (e.g. body['input'] "
            "for Responses, optimized_messages for chat). Never write to "
            "body['instructions'] or body['system'] — they are the cache "
            "hot zone (invariant I2)."
        )


# ── Invariant C — every memory-search call passes a MemoryQuery ──────


def _find_search_and_format_context_calls_without_query(
    file_path: Path,
) -> list[tuple[int, str]]:
    """Find ``search_and_format_context(...)`` invocations that DON'T
    pass a ``query=`` kwarg.

    Pre-PR-this no site passed a query — they all relied on the
    handler's internal ``_extract_user_query(messages)`` with its
    500-char truncation. The new contract: every handler builds a
    full-fidelity ``MemoryQuery`` and passes it explicitly.
    """
    text = file_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(file_path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "search_and_format_context":
            continue
        kwarg_keys = {kw.arg for kw in node.keywords if kw.arg is not None}
        if "query" not in kwarg_keys:
            line_no = node.lineno
            snippet = text.splitlines()[line_no - 1].strip()
            hits.append((line_no, snippet))
    return hits


def test_every_search_and_format_context_call_passes_query_kwarg() -> None:
    """Every handler that searches memory MUST pass a ``query=`` kwarg
    (a :class:`MemoryQuery` instance), not rely on the handler's
    internal ``_extract_user_query`` (which used to truncate to 500
    chars). Locks the full-fidelity-query contract."""
    offenders = []
    for f in HANDLER_FILES:
        offenders.extend(
            [(f, ln, src) for (ln, src) in _find_search_and_format_context_calls_without_query(f)]
        )
    if offenders:
        formatted = "\n".join(f"  {f.name}:{ln}  {src!r}" for f, ln, src in offenders)
        pytest.fail(
            f"{len(offenders)} search_and_format_context call(s) miss `query=`:\n"
            f"{formatted}\n\n"
            "Pass `query=MemoryQuery.from_messages(...)` — the multi-source, "
            "untruncated query value type. See headroom/proxy/memory_query.py."
        )
