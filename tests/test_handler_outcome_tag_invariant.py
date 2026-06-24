"""Contract test: every handler that emits a RequestOutcome must thread tags.

Prior to PR #480, 12 RequestOutcome construction sites across four
handler files emitted outcomes without passing ``tags=`` — so any
request hitting those handlers reached the dashboard / RequestLog feed
with an empty tag dict, invisible to per-harness / per-tag filtering.
Affected paths included:

* The `from_response_cache=True` early-return paths in
  ``handle_anthropic_messages`` and ``handle_openai_chat`` (so Claude
  Code's + Codex's cache-hit turns were dashboard-blind)
* The Codex WS per-turn outcome in ``handle_openai_responses_ws``
* All four Anthropic batch handlers, all four Google batch handlers,
  the OpenAI batch handler, and the OpenAI passthrough handler

This test introspects the four handler modules' ASTs and asserts that
every ``RequestOutcome(...)`` keyword-call inside any handler-shaped
method passes a ``tags=`` kwarg. The check is static; no handler is
invoked. ``from_stream`` classmethod construction is allowed (it
takes ``tags`` as a required kwarg) and the test verifies that too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

HANDLER_FILES = [
    Path("headroom/proxy/handlers/anthropic.py"),
    Path("headroom/proxy/handlers/openai.py"),
    Path("headroom/proxy/handlers/gemini.py"),
    Path("headroom/proxy/handlers/batch.py"),
]


def _collect_outcome_call_sites() -> list[tuple[Path, str, int, set[str]]]:
    """Walk each handler module's AST; for every
    ``RequestOutcome(...)`` or ``RequestOutcome.from_stream(...)`` call
    inside any ``async def handle_*`` or ``async def _*_passthrough``
    method, record (file, method_name, lineno, kwarg_keys).
    """
    sites: list[tuple[Path, str, int, set[str]]] = []
    for file_path in HANDLER_FILES:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
        for module_node in ast.walk(tree):
            if not isinstance(module_node, ast.ClassDef):
                continue
            for class_node in module_node.body:
                if not isinstance(class_node, ast.AsyncFunctionDef):
                    continue
                method_name = class_node.name
                # Only audit methods that look like request entry points
                # or batch-passthrough helpers. ``_record_request_outcome``
                # itself is a helper, not a handler — skip it.
                if not (method_name.startswith("handle_") or method_name.endswith("_passthrough")):
                    continue
                for sub_node in ast.walk(class_node):
                    if not isinstance(sub_node, ast.Call):
                        continue
                    # Match RequestOutcome(...) or RequestOutcome.from_stream(...)
                    is_outcome = False
                    if isinstance(sub_node.func, ast.Name) and sub_node.func.id == "RequestOutcome":
                        is_outcome = True
                    elif (
                        isinstance(sub_node.func, ast.Attribute)
                        and isinstance(sub_node.func.value, ast.Name)
                        and sub_node.func.value.id == "RequestOutcome"
                        and sub_node.func.attr == "from_stream"
                    ):
                        is_outcome = True
                    if not is_outcome:
                        continue
                    kwarg_keys = {kw.arg for kw in sub_node.keywords if kw.arg is not None}
                    sites.append(
                        (file_path, method_name, sub_node.lineno, kwarg_keys),
                    )
    return sites


def test_outcome_call_sites_pass_tags_kwarg() -> None:
    """Every RequestOutcome construction inside a handler method MUST
    pass ``tags=``. Otherwise the dashboard's tag-based slicing is
    silently bypassed for that traffic path.

    If this test fails on a new handler you just wrote, add
    ``tags = self._extract_tags(headers)`` near the top of your handler
    and thread ``tags=tags`` into the RequestOutcome construction.
    """
    sites = _collect_outcome_call_sites()
    assert sites, "AST walk found zero RequestOutcome sites — handler files moved?"
    missing = [(f, m, ln) for f, m, ln, kws in sites if "tags" not in kws]
    if missing:
        formatted = "\n".join(f"  {f.name}:{ln}  {m}" for f, m, ln in missing)
        pytest.fail(
            f"{len(missing)} RequestOutcome sites miss `tags=`:\n{formatted}\n\n"
            "Each handler MUST extract tags from headers and thread them "
            "into the outcome construction. See PR #480 for the pattern."
        )


def test_outcome_call_sites_pass_client_kwarg() -> None:
    """Sibling invariant: every RequestOutcome from a handler also
    threads ``client=``. We have this everywhere today; this test
    locks it so future handlers can't regress."""
    sites = _collect_outcome_call_sites()
    assert sites
    missing = [(f, m, ln) for f, m, ln, kws in sites if "client" not in kws]
    if missing:
        formatted = "\n".join(f"  {f.name}:{ln}  {m}" for f, m, ln in missing)
        pytest.fail(
            f"{len(missing)} RequestOutcome sites miss `client=`:\n{formatted}\n\n"
            "Each handler MUST classify the harness via "
            "`client = classify_client(headers)` and thread `client=client` "
            "into the outcome construction. See PR #473 for the pattern."
        )


# ── Invariant: image-compression must route through ImageCompressionDecision ──


import re  # noqa: E402  -- only used by the image-decision invariant below


def test_no_raw_image_optimize_gate_in_handlers() -> None:
    """Locks the post-this-PR contract: image compression must be
    gated by :class:`ImageCompressionDecision`, not by an inline
    ``if self.config.image_optimize and messages and not _bypass:``
    conjunction. Pre-PR-this both sites used the raw conjunction;
    consolidating into a value type means a future site (e.g., new
    provider handler) can't drift on bypass-respect or skip-reason
    observability.

    Allowed forms after this PR:
    * ``if _image_decision.should_compress``
    * ``if _image_decision.should_compress and ...``
    """
    pattern = re.compile(r"^\s*if\s*\(?\s*self\.config\.image_optimize\s+and\s+messages\b")
    offenders: list[tuple[str, int, str]] = []
    for f in HANDLER_FILES:
        text = f.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.match(line):
                offenders.append((f.name, i, line.rstrip()))
    if offenders:
        formatted = "\n".join(f"  {f}:{ln}  {src!r}" for f, ln, src in offenders)
        pytest.fail(
            f"{len(offenders)} handler site(s) use the pre-PR raw image "
            "gate `if self.config.image_optimize and messages [and ...]`:\n"
            f"{formatted}\n\n"
            "Replace with `ImageCompressionDecision.decide(...)` + "
            "`if _image_decision.should_compress:`. See "
            "headroom/proxy/image_compression_decision.py for the pattern."
        )
