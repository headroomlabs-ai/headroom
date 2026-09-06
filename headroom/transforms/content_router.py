"""Content router for intelligent compression strategy selection.

This module provides the ContentRouter, which analyzes content and routes it
to the optimal compressor. It handles mixed content by splitting, routing
each section to the appropriate compressor, and reassembling.

Supported Compressors:
- CodeAwareCompressor: Source code (AST-preserving)
- SmartCrusher: JSON arrays
- SearchCompressor: grep/ripgrep results
- LogCompressor: Build/test output
- KompressCompressor: Plain text (ML-based)
- Kompress: Plain text (ML-based, requires [ml] extra)

Routing Strategy:
1. Use source hint if available (highest confidence)
2. Check for mixed content (split and route sections)
3. Detect content type (JSON, code, search, logs, text)
4. Route to appropriate compressor
5. Reassemble and return with routing metadata

Usage:
    >>> from headroom.transforms import ContentRouter
    >>> router = ContentRouter()
    >>> result = router.compress(content)  # Auto-routes to best compressor
    >>> print(result.strategy_used)
    >>> print(result.routing_log)

Pipeline Usage:
    >>> pipeline = TransformPipeline([
    ...     ContentRouter(),   # Handles all content types
    ... ])
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from ..config import (
    DEFAULT_BYTE_EXACT_EXCLUDE_TOOLS,
    DEFAULT_EXCLUDE_TOOLS,
    DEFAULT_VERBATIM_EXCLUDE_TOOLS,
    ReadLifecycleConfig,
    RelevanceScorerConfig,
    TransformResult,
    is_tool_excluded,
    unwrap_tool_call_name,
)
from ..parser import CCR_RETRIEVAL_MARKER_RE
from ..tokenizer import Tokenizer
from ..tokenizers.base import count_content_blocks
from ..tokenizers.estimator import EstimatingTokenCounter
from . import mixed_content as _mixed_content
from .base import Transform
from .compression_policy import cache_write_multiplier_for_ttl
from .compressor_registry import (
    CompressInput,
    CompressorDescriptor,
    CompressorRegistry,
    CompressOutput,
)
from .content_detector import (
    ContentType,
    DetectionResult,
    _try_detect_log,
    _try_detect_search,
    _try_detect_structured_config,
)
from .content_detector import detect_content_type as _regex_detect_content_type
from .error_detection import content_has_strong_error_indicators
from .lossless_provider import (
    get_lossless_generation,
    get_lossless_provider,
    get_lossless_verifier,
)
from .mixed_content import ContentSection, mixed_content_indicators
from .relevance_split import build_relevance_query, plan_relevance_split

logger = logging.getLogger(__name__)

_extract_json_block = _mixed_content._extract_json_block
is_mixed_content = _mixed_content.is_mixed_content
split_into_sections = _mixed_content.split_into_sections


_detect_backend_warned = False
_detect_panic_warned = False
_detect_native_unhealthy = False  # circuit breaker: native detect hung once (#575)


# Shared calibrated fallback estimator (tiktoken cl100k_base ~90% accuracy,
# content-type aware incl. JSON). Kept as one module-level instance so the
# size heuristic lives in a single reusable place, not a hardcoded constant.
_TOKEN_ESTIMATOR = EstimatingTokenCounter()


# `kind` labels supplied by a third-party lossless provider flow into
# `transforms_applied`, `transforms_summary` and the per-strategy metric/timing
# dicts (which become Prometheus label values). An unbounded caller-controlled
# string there is a label-cardinality explosion, so only a short, lowercase,
# identifier-shaped label is accepted; anything else is replaced with
# `_PROVIDER_KIND_FALLBACK` rather than raising (a bad label must not fail a
# request, and the fold itself is still valid).
_PROVIDER_KIND_RE = re.compile(r"^[a-z0-9_]{1,32}$")
_PROVIDER_KIND_FALLBACK = "provider"


# Every marker shape that means "this text is already compressed and the
# real bytes live in the CCR store". The bracket forms come from
# SmartCrusher's row-drop summary and read_lifecycle/read_maturation; the
# `<<ccr:` form is emitted by the Rust opaque-blob and row-drop paths
# (`<<ccr:HASH,KIND,SIZE>>`, `<<ccr:HASH N_rows_offloaded>>`, `<<ccr:HASH>>`).
_ALREADY_COMPRESSED_MARKERS = (
    "Retrieve more: hash=",
    "Retrieve original: hash=",
    "<<ccr:",
)


def _is_already_compressed(text: str) -> bool:
    """True if ``text`` still carries a CCR retrieval marker.

    Re-compressing such a block is never right. Beyond the prefix-cache
    churn, the second pass treats the *compressed* text as source: a marker
    that lands in a cell wide enough to be re-offloaded gets hashed and
    stashed as the new entry's "original", so ``headroom_retrieve`` returns a
    placeholder and the inner marker's hash — the only handle on the real
    bytes — disappears from anywhere the model can see (#2694).

    ``<<ccr:`` was missing from this check, which is how opaque-blob markers
    (the base64/binary form) leaked back into the compressor.
    """
    return any(marker in text for marker in _ALREADY_COMPRESSED_MARKERS)


def _estimate_tokens(text: str) -> int:
    """Size-proportional token estimate for section ratio decisions.

    Whitespace splitting undercounts pathologically on compact
    machine-generated JSON (no spaces means the whole payload is ~1
    "word"), which made section compression ratios compute as ~1.0 and
    the min_ratio acceptance gate silently reject real compression.
    Delegates to the shared EstimatingTokenCounter — JSON/code aware and
    calibrated against tiktoken — so the estimate tracks real tokens across
    formats instead of a fixed chars/N constant.
    """
    return max(1, _TOKEN_ESTIMATOR.count_text(text))


def _compression_deadline_seconds() -> float:
    try:
        return max(
            0.0,
            float(os.environ.get("HEADROOM_COMPRESSION_DEADLINE_MS", "20000")) / 1000.0,
        )
    except ValueError:
        return 20.0


def _router_debug_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


# ── Built-in compressor inventory (registry metadata only) ────────────────────
# Declarative capability metadata for each built-in compressor. The registry is
# an *inventory*: built-ins are still constructed and dispatched by the router's
# existing if/elif in `_apply_strategy_to_content` — these descriptors change no
# routing. They give the (opt-in) `headroom.compressor` registry a name-
# addressable view of what ships in-tree, alongside any third-party compressors
# discovered from the entry-point group. The content_types / lossless /
# cost_tier / recoverable fields are declarative (they describe the built-in's
# typical behavior in the default CCR configuration) and are not read on the
# request hot path today.
_BUILTIN_COMPRESSOR_DESCRIPTORS: tuple[CompressorDescriptor, ...] = (
    CompressorDescriptor(
        name="smart_crusher",
        content_types=["application/json"],
        lossless=False,
        cost_tier="fast",
        recoverable=True,
    ),
    CompressorDescriptor(
        name="kompress",
        content_types=["text/plain"],
        lossless=False,
        cost_tier="ml",
        recoverable=True,
    ),
    CompressorDescriptor(
        name="code_aware",
        content_types=["text/x-code"],
        lossless=False,
        cost_tier="fast",
        recoverable=True,
    ),
    CompressorDescriptor(
        name="search",
        content_types=["text/x-search-results"],
        lossless=True,
        cost_tier="fast",
        recoverable=True,
    ),
    CompressorDescriptor(
        name="log",
        content_types=["text/x-log"],
        lossless=False,
        cost_tier="fast",
        recoverable=True,
    ),
    CompressorDescriptor(
        name="tabular",
        content_types=["text/csv"],
        lossless=False,
        cost_tier="fast",
        recoverable=True,
    ),
    CompressorDescriptor(
        name="config",
        content_types=["text/x-config"],
        lossless=False,
        cost_tier="fast",
        recoverable=False,
    ),
    CompressorDescriptor(
        name="html",
        content_types=["text/html"],
        lossless=False,
        cost_tier="fast",
        recoverable=False,
    ),
    CompressorDescriptor(
        name="image",
        content_types=["image/*"],
        lossless=False,
        cost_tier="ml",
        recoverable=False,
    ),
)


# ── Built-in Compressor adapters (registry delegation, additive) ──────────────
# Each built-in registry entry delegates ``compress`` to the SAME underlying
# built-in method the content router invokes in ``_apply_strategy_to_content`` —
# reached through the router's own ``_get_*`` getter so config flows through
# identically. The adapters are ADDITIVE: the router still dispatches built-ins
# via its existing if/elif and never routes a request through the registry, so
# they change no routing. Each invoker takes the owning router plus the pure-data
# :class:`CompressInput` and returns the compressed string, or ``None`` when the
# built-in is unavailable / not applicable to this str input (→ passthrough).
_BuiltinInvoke = Callable[["ContentRouter", CompressInput], "str | None"]


def _adapter_bias(inp: CompressInput) -> float:
    """Compression bias for a built-in call — the router's dispatch default (1.0).

    Callers may override via ``budget['bias']`` (the router passes ``bias`` on the
    request path); anything non-numeric falls back to the 1.0 default.
    """
    try:
        return float(inp.budget.get("bias", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _invoke_smart_crusher(router: ContentRouter, inp: CompressInput) -> str | None:
    crusher = router._get_smart_crusher()
    if crusher is None:
        return None
    # ``_get_*`` getters are typed ``Any``; pin the result to the contract type.
    compressed: str = crusher.crush(
        inp.content, query=inp.query, bias=_adapter_bias(inp)
    ).compressed
    return compressed


def _invoke_code_aware(router: ContentRouter, inp: CompressInput) -> str | None:
    compressor = router._get_code_compressor()
    if compressor is None:
        return None
    language = inp.config.get("language")
    result = compressor.compress(inp.content, language=language, context=inp.query)
    compressed: str = result.compressed
    return compressed


def _invoke_search(router: ContentRouter, inp: CompressInput) -> str | None:
    compressor = router._get_search_compressor()
    if compressor is None:
        return None
    result = compressor.compress(inp.content, context=inp.query, bias=_adapter_bias(inp))
    compressed: str = result.compressed
    return compressed


def _invoke_log(router: ContentRouter, inp: CompressInput) -> str | None:
    compressor = router._get_log_compressor()
    if compressor is None:
        return None
    compressed: str = compressor.compress(inp.content, bias=_adapter_bias(inp)).compressed
    return compressed


def _invoke_tabular(router: ContentRouter, inp: CompressInput) -> str | None:
    compressor = router._get_tabular_compressor()
    if compressor is None:
        return None
    result = compressor.compress(inp.content, context=inp.query, bias=_adapter_bias(inp))
    compressed: str = result.compressed
    return compressed


def _invoke_config(router: ContentRouter, inp: CompressInput) -> str | None:
    compressor = router._get_config_compressor()
    if compressor is None:
        return None
    result = compressor.compress(inp.content, context=inp.query, bias=_adapter_bias(inp))
    compressed: str = result.compressed
    return compressed


def _invoke_html(router: ContentRouter, inp: CompressInput) -> str | None:
    extractor = router._get_html_extractor()
    if extractor is None:
        return None
    # ``.extracted`` may be None/empty when nothing extracts; the caller maps
    # that to passthrough, matching the router's HTML branch.
    extracted: str | None = extractor.extract(inp.content).extracted
    return extracted


def _invoke_kompress(router: ContentRouter, inp: CompressInput) -> str | None:
    # The router dispatches KOMPRESS through ``_try_ml_compressor`` (size gate,
    # tag protection, background load, marker policy), so the adapter delegates
    # to the SAME method to stay byte-identical to the router's kompress path.
    # ``question`` (QA-aware compression) rides the pure-data contract via
    # ``config['question']`` — the router sets it in ``_registry_compress`` — and
    # is forwarded here so the compressed CONTENT matches the router's direct
    # ``_try_ml_compressor(content, context, question)`` call. A missing/None
    # ``question`` forwards None, exactly the no-question path. (Previously this
    # hardcoded ``None``, silently dropping the QA-aware ``question`` — that bug is
    # fixed here so the flip preserves content.)
    question = inp.config.get("question")
    compressed, _tokens = router._try_ml_compressor(inp.content, inp.query, question)
    return compressed


def _invoke_image(router: ContentRouter, inp: CompressInput) -> str | None:
    # The image built-in (``ImageCompressor``) compresses image blocks inside
    # message dicts via ``ImageCompressor.compress(messages)`` /
    # ``optimize_images_in_messages`` — it is NOT dispatched through
    # ``_apply_strategy_to_content`` and never operates on str content. The
    # str-based CompressInput/CompressOutput contract has no faithful image
    # delegation (str content is never image data), so this is a documented
    # passthrough rather than a fabricated compression.
    return None


def _invoke_passthrough(router: ContentRouter, inp: CompressInput) -> str | None:
    # Defensive default for a descriptor without a registered invoker.
    return None


#: Built-in descriptor name → the invoker that runs it via the router's getter.
_BUILTIN_COMPRESSOR_INVOKERS: dict[str, _BuiltinInvoke] = {
    "smart_crusher": _invoke_smart_crusher,
    "kompress": _invoke_kompress,
    "code_aware": _invoke_code_aware,
    "search": _invoke_search,
    "log": _invoke_log,
    "tabular": _invoke_tabular,
    "config": _invoke_config,
    "html": _invoke_html,
    "image": _invoke_image,
}


class _BuiltinCompressorEntry:
    """Registry adapter running a built-in via the router's existing dispatch path.

    ``compress`` delegates to the SAME underlying built-in method the content
    router invokes in ``_apply_strategy_to_content`` (obtained through the
    router's ``_get_*`` getter so config flows through), then maps the built-in's
    native result onto the pure-data :class:`CompressOutput` contract.

    ADDITIVE by construction: the router still dispatches built-ins through its
    own if/elif and never routes a request through the registry, so these entries
    change no routing. ``_resolve_active_external_compressors`` filters them out
    of the opt-in external-dispatch path *by type*, so the class name is load-
    bearing. Constructed lazily/cheaply — it stores only the descriptor, the
    owning router, and the invoke callable; no built-in is instantiated until
    ``compress`` runs.

    ``recoverable`` is always ``{}``: the built-ins embed CCR retrieval markers
    in the compressed content and mirror ``hash -> original`` into the CCR store
    as a side effect of their own ``compress`` call (which this adapter invokes),
    rather than returning a recovery map on their result object.
    """

    def __init__(
        self,
        descriptor: CompressorDescriptor,
        router: ContentRouter | None = None,
        invoke: _BuiltinInvoke | None = None,
    ) -> None:
        self._descriptor = descriptor
        self._router = router
        self._invoke = (
            invoke
            if invoke is not None
            else _BUILTIN_COMPRESSOR_INVOKERS.get(descriptor.name, _invoke_passthrough)
        )

    @property
    def descriptor(self) -> CompressorDescriptor:
        return self._descriptor

    def compress(self, inp: CompressInput) -> CompressOutput:
        tokens_before = _estimate_tokens(inp.content)
        raw: str | None = None
        if self._router is not None:
            raw = self._invoke(self._router, inp)
        # A ``None`` from the invoker means the built-in did not compress — it
        # was unavailable, had no bound router, or was not applicable to this str
        # input (e.g. HTML extraction found nothing). Report that with
        # ``compressed=False`` and pass the ORIGINAL content through unchanged
        # (never blank out or expand a block) so a caller can run its own
        # fallback exactly as the historical direct call did on a ``None``
        # result. A non-``None`` result is a real compression → ``compressed``
        # stays True and byte-identical to before.
        did_compress = raw is not None
        content = raw if raw is not None else inp.content
        return CompressOutput(
            content=content,
            tokens_before=tokens_before,
            tokens_after=_estimate_tokens(content),
            lossless=self._descriptor.lossless,
            markers=[],
            recoverable={},
            warnings=[],
            compressed=did_compress,
        )


def _build_compressor_registry(router: ContentRouter | None = None) -> CompressorRegistry:
    """Build the router's compressor registry: built-in adapters + discovery.

    Registers a delegating adapter for each built-in (bound to ``router`` so
    ``compress`` runs the built-in through the router's own getter), then runs
    opt-in discovery of ``headroom.compressor`` entry points. Discovery never
    invokes ``compress`` and is fail-open (a broken third-party package is logged
    and skipped). Building the registry has no side effects: adapters instantiate
    nothing until ``compress`` is called, and the router still dispatches built-
    ins via its own if/elif, so constructing this registry cannot change request
    handling. When ``router`` is ``None`` the adapters have nothing to delegate to
    and ``compress`` is an inert passthrough.
    """
    registry = CompressorRegistry()
    for descriptor in _BUILTIN_COMPRESSOR_DESCRIPTORS:
        registry.register(_BuiltinCompressorEntry(descriptor, router))
    # External compressors register under distinct names; a name collision with
    # a built-in is skipped fail-open (replace=False) so a third-party package
    # can never shadow a built-in's inventory entry.
    registry.discover()
    return registry


# Canonical map from the router's internal :class:`ContentType` to the MIME
# string an external ``headroom.compressor`` declares in
# ``CompressorDescriptor.content_types``. Mirrors the MIME strings used by the
# built-in descriptors above (so an external JSON compressor declares the same
# ``application/json`` a built-in would), with ``text/x-diff`` added for
# ``GIT_DIFF`` (no built-in descriptor covers diffs). This is the ONLY bridge
# between the enum the router routes on and the pure-string content type the
# registry contract carries; it is read solely by the opt-in external-dispatch
# branch and never on the default request path.
_CONTENT_TYPE_TO_MIME: dict[ContentType, str] = {
    ContentType.JSON_ARRAY: "application/json",
    ContentType.SOURCE_CODE: "text/x-code",
    ContentType.SEARCH_RESULTS: "text/x-search-results",
    ContentType.BUILD_OUTPUT: "text/x-log",
    ContentType.GIT_DIFF: "text/x-diff",
    ContentType.HTML: "text/html",
    ContentType.TABULAR: "text/csv",
    ContentType.STRUCTURED_CONFIG: "text/x-config",
    ContentType.PLAIN_TEXT: "text/plain",
}


def _external_compressor_matches(descriptor: CompressorDescriptor, content_mime: str) -> bool:
    """True if ``descriptor`` declares support for ``content_mime``.

    Accepts an exact MIME match, a full wildcard (``"*"`` or ``"*/*"``), or a
    type wildcard (``"text/*"`` matches ``"text/plain"``). Anything else is a
    non-match, so a selected external compressor only ever sees content it
    explicitly declared it can handle.
    """
    declared = descriptor.content_types or []
    if content_mime in declared:
        return True
    top = content_mime.split("/", 1)[0]
    type_wildcard = f"{top}/*"
    return any(d in ("*", "*/*") or d == type_wildcard for d in declared)


def _tool_call_args_text(raw: Any) -> str:
    """Compact, query-usable text from a tool call's args.

    Anthropic passes ``input`` as a dict ({"command": "grep …"}); OpenAI passes
    ``arguments`` as a JSON string. Either way we want the scalar values (the
    grep pattern, the read path) as a short query fragment. Capped so a giant
    arg blob can't dominate the relevance query.
    """
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, dict):
        text = " ".join(str(v) for v in raw.values() if isinstance(v, str | int | float | bool))
    else:
        return ""
    return " ".join(text.split())[:300]


def read_protection_enabled() -> bool:
    """True when HEADROOM_PROTECT_READS opts into byte-exact file-read protection.

    Shared by every request path (chat/Anthropic ``ContentRouter.apply`` and the
    OpenAI Responses units path) so the flag means the same thing everywhere.
    """
    return os.environ.get("HEADROOM_PROTECT_READS", "0").strip().lower() not in (
        "0",
        "",
        "false",
        "no",
    )


def _tool_call_command_text(raw: Any) -> str:
    """Extract the raw shell command from a tool call's args, if present.

    Anthropic ``input`` is a dict ({"command": "grep …"}); OpenAI ``arguments``
    is a JSON string; Codex's shell uses a ``command`` list. Returns "" when
    there is no command field (non-shell tools).
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return ""
    if not isinstance(raw, dict):
        return ""
    cmd = raw.get("command", raw.get("cmd", ""))
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    return cmd if isinstance(cmd, str) else ""


def _fenced_shell_command(content: Any) -> str:
    """Extract the shell command from a TEXT-BASED agent's fenced code block.

    Text-based harnesses (mini-swe-agent backticks, Codex, Cursor, and any
    non-native-tool OpenAI agent) put the command in a ```mswea_bash_command /
    ```bash fenced block inside the assistant's *string* content — there is no
    ``tool_use``/``tool_calls`` block. Returns the first fenced block's body, or
    "" when there is none. Shape-agnostic input to read-detection so cat/sed
    reads are protected on any model, not just those emitting tool-call blocks.
    """
    if not isinstance(content, str) or "```" not in content:
        return ""
    m = re.search(r"```(?:[\w.-]+)?[ \t]*\n(.*?)```", content, re.S)
    return m.group(1).strip() if m else ""


_READ_VERBS = ("cat", "head", "tail", "nl", "bat", "less", "more")

# Machine-generated dependency lockfiles detect as PLAIN_TEXT (so the content-based
# read gate would protect them), but they are regenerated by a tool and never patched
# byte-for-byte — the biggest, most-repetitive read in a session. Match by NAME so a
# read of one is never added to the protected set and stays compressible.
_LOCKFILE_RE = re.compile(
    r"(^|[\s/])("
    r"bun\.lock|bun\.lockb|package-lock\.json|npm-shrinkwrap\.json|yarn\.lock|"
    r"pnpm-lock\.yaml|uv\.lock|poetry\.lock|Pipfile\.lock|requirements\.txt\.lock|"
    r"Cargo\.lock|go\.sum|Gemfile\.lock|composer\.lock|flake\.lock|Package\.resolved|"
    r"gradle\.lockfile|packages\.lock\.json"
    r")(\s|$)",
    re.IGNORECASE,
)


def _strip_cd_prefix(command: str) -> str:
    """Peel leading ``cd <dir> &&|; `` chains from a shell command.

    Agent harnesses (mini-swe-agent, Codex, Cursor, …) prefix nearly every command
    with ``cd <repo> && `` (or ``cd <repo>; ``) to run in the checkout. Command-
    classification helpers must strip this first, or the parsed program is ``cd``
    instead of the real tool (``grep``/``cat``/…) — which silently disables read-
    protection and the lossless search-fold (observed: 100% of ``cd … && rg``
    output went uncompacted). Provider/harness-agnostic: operates on the plain
    shell string that every client ultimately produces. Only ``&&`` and ``;``
    connectors are peeled (a mis-parse is harmless — the caller falls through to
    the normal path, guarded by downstream reversibility checks).
    """
    if not command or not isinstance(command, str):
        return ""
    c = command.strip()
    while True:
        m = re.match(r"^cd\s+[^&;|]+(?:&&|;)\s*(.*)$", c, re.S)
        if not m:
            break
        c = m.group(1).strip()
    return c


def _is_read_command(command: str) -> bool:
    """True when a shell command's output is essentially raw FILE CONTENT the agent
    will read/edit from — ``cat``/``head``/``tail``/``nl``/``less``/``more`` of a file,
    or ``sed -n`` range-printing.

    Such reads must NOT be lossy-compressed: the agent needs the exact bytes to produce
    a precise patch. Lossy-compressing them was observed (SWE-bench, mini-swe-agent) to
    cause the agent to RE-READ the same file (cat -> cat -A -> cat -n) to recover exact
    detail — turn inflation — and, when recovery failed, resolve loss. Search/list/test
    output (grep/rg/ls/find/pytest) is derived and stays compressible.

    This identifies that a command is a file READ. Whether the read is actually PROTECTED
    is finalized downstream by CONTENT type (see ``_read_output_should_be_protected``):
    reads are protected by default, and only released to compression when the output is a
    confidently non-code DATA type. The one command-level carve-out is lockfiles: they
    detect as PLAIN_TEXT (so the content gate would protect them) yet are regenerated
    artifacts, never byte-patched — so a lockfile read returns False here and stays
    compressible.

    Excludes writes: a redirect (``>``/``>>``), ``tee``, or heredoc (``<<``) means the
    command WRITES a file (e.g. ``cat > f <<EOF``), and a bare ``sed`` (without ``-n``)
    is a stream edit — neither is a read.
    """
    if not command or not isinstance(command, str):
        return False
    # strip leading `cd <dir> && ` chains (agents prefix reads with a cd)
    c = _strip_cd_prefix(command)
    # a write / append / tee / heredoc anywhere => not a pure read
    if re.search(r"(^|\s)(>>?|tee\b|<<)", c):
        return False
    # Parse the real program with the SAME structural parser the search-fold uses
    # (_bash_program peels sudo/env/timeout/rtk wrappers + env assignments), so
    # `sudo cat f`, `timeout 30 cat f`, `rtk cat f` are recognized as reads, not
    # silently dropped by a first-token match.
    prog, rest = _bash_program(c)
    if not prog:
        return False
    if prog in {"sh", "bash", "zsh", "dash"} and rest:
        # `bash -lc "cat …"` (Codex): the real command is the -c argument.
        for j, tok in enumerate(rest):
            if tok in {"-c", "-lc", "-lic", "-ic"} and j + 1 < len(rest):
                return _is_read_command(" ".join(rest[j + 1 :]).strip("'\""))
        return False
    is_read = prog in _READ_VERBS or (
        # `sed -n '1,20p' file` prints a range (read); bare `sed` is a stream editor.
        prog == "sed" and bool(re.search(r"(^|\s)-n(\s|$)", c))
    )
    if not is_read:
        return False
    # Lockfiles are tool-regenerated, not byte-patched — never protect (keep compressible).
    return not _LOCKFILE_RE.search(c)


# Shell wrappers that prefix the real program — peeled to find it. Shell
# grammar, not tunable policy: rtk (the user's token proxy), sudo/env/timeout/…
_SHELL_WRAPPERS = frozenset(
    {
        "rtk",
        "sudo",
        "env",
        "time",
        "nice",
        "ionice",
        "nohup",
        "stdbuf",
        "command",
        "timeout",
        "xargs",
    }
)


def _bash_program(command: str) -> tuple[str, list[str]]:
    """Return ``(program_basename_lower, trailing_tokens)`` for a shell command.

    Peels leading wrappers (``rtk grep`` -> ``grep``, ``timeout 30 rg`` -> ``rg``)
    and env assignments (``FOO=1 grep`` -> ``grep``). Empty program when it can't
    be determined. Whitespace-split is deliberately simple — the reversibility
    guard downstream makes a parse miss harmless.
    """
    toks = command.strip().split()
    i = 0
    while i < len(toks):
        tok = toks[i]
        if "=" in tok and not tok.startswith("-"):  # VAR=val env assignment
            i += 1
            continue
        base = tok.rsplit("/", 1)[-1].lower()  # /usr/bin/grep -> grep
        if base in _SHELL_WRAPPERS:
            i += 1
            # Skip this wrapper's own option/numeric args (timeout 30, nice -n 5).
            while i < len(toks) and (
                toks[i].startswith("-") or toks[i].replace(".", "", 1).isdigit()
            ):
                i += 1
            continue
        return base, toks[i + 1 :]
    return "", []


def _bash_command_is_search(command: str, search_commands: frozenset[str]) -> bool:
    """True when ``command`` is a read-only search whose output folds byte-
    losslessly (grep/rg/git grep/…). Peels wrappers and recurses into ``sh -c``.
    """
    # Peel `cd <dir> && ` chains first — harnesses prefix every command with a
    # cd, so without this the parsed program is `cd` and the fold never fires.
    command = _strip_cd_prefix(command)
    prog, rest = _bash_program(command)
    if not prog:
        return False
    if prog in {"sh", "bash", "zsh", "dash"} and rest:
        # `bash -lc "grep …"` (Codex): the real command is the -c argument.
        for j, tok in enumerate(rest):
            if tok in {"-c", "-lc", "-lic", "-ic"} and j + 1 < len(rest):
                inner = " ".join(rest[j + 1 :]).strip("'\"")
                return _bash_command_is_search(inner, search_commands)
        return False
    if prog == "git" and rest and rest[0].lower() == "grep":
        return True
    return prog in search_commands


def _log_router_debug(event: str, **payload: Any) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    payload = {"event": event, **payload}
    logger.debug("event=%s %s", event, _router_debug_dumps(payload))


def _json_shape(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except Exception as exc:
        return {"is_json": False, "error": type(exc).__name__}
    if isinstance(parsed, dict):
        return {
            "is_json": True,
            "kind": "object",
            "keys": list(parsed.keys()),
            "length": len(parsed),
        }
    if isinstance(parsed, list):
        return {"is_json": True, "kind": "array", "length": len(parsed)}
    return {"is_json": True, "kind": type(parsed).__name__}


def _content_is_valid_json(content: str) -> bool:
    """Return True iff ``content`` parses as valid JSON.

    Used by the SMART_CRUSHER → Log fallback guard (#1306): the native
    magika detector tags content by shape, so truncated/mid-stream JSON
    tool outputs are misclassified as ``json_array``. SmartCrusher can't
    parse them and returns no savings; without this guard the LogCompressor
    would then collapse the broken JSON to a single CCR-retrieval marker
    (99.9% data loss). Valid JSON arrays still reach the Log fallback —
    LogCompressor is a no-op on them, so the guard is safe for the
    intended "repetitive JSONL" case.
    """
    try:
        json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return False
    except RecursionError:
        # Deeply nested JSON (e.g. ``[[[[...]]]]`` with 10k+ levels) can
        # exceed Python's recursion limit inside ``json.loads``.  Treat
        # it as invalid so the Log fallback is skipped — the content
        # passes through verbatim instead of being collapsed.
        logger.warning(
            "json.loads hit recursion limit on deeply nested JSON "
            "(%d chars); treating as invalid for Log-fallback guard",
            len(content),
        )
        return False
    return True


def _mixed_indicators(content: str) -> dict[str, bool]:
    return mixed_content_indicators(content)


def _section_debug(section: ContentSection, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "content_type": section.content_type.value,
        "language": getattr(section, "language", None),
        "start_line": getattr(section, "start_line", None),
        "end_line": getattr(section, "end_line", None),
        "is_code_fence": getattr(section, "is_code_fence", False),
        "chars": len(section.content),
        "bytes": len(section.content.encode("utf-8", errors="replace")),
        "tokens_estimate": _estimate_tokens(section.content),
        "json_shape": _json_shape(section.content),
        "content": section.content,
    }


def _resolve_detect_backend() -> str:
    """Pick the content-detection backend: ``"rust"`` or ``"python"``."""
    backend = os.environ.get("HEADROOM_DETECT_BACKEND", "").strip().lower()
    if backend in ("python", "rust"):
        return backend
    return "python" if sys.platform == "win32" else "rust"


_DETECT_TIMEOUT_ENV = "HEADROOM_DETECT_TIMEOUT_SECS"
_DEFAULT_DETECT_TIMEOUT_SECS = 5.0


def _detect_timeout_secs() -> float:
    """Watchdog budget (seconds) for one native detect call.

    Override with ``HEADROOM_DETECT_TIMEOUT_SECS``; blank, non-numeric, or
    non-positive values fall back to the default.
    """
    raw = os.environ.get(_DETECT_TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_DETECT_TIMEOUT_SECS
    try:
        secs = float(raw)
    except ValueError:
        return _DEFAULT_DETECT_TIMEOUT_SECS
    return secs if secs > 0 else _DEFAULT_DETECT_TIMEOUT_SECS


def _rust_detect_watchdogged(rust_detect: Any, content: str, timeout: float) -> Any:
    """Run the native detector under a watchdog thread, bounding the caller's wait.

    On Windows the first native ``detect_content_type`` can park forever in an
    ort/``Once`` init (``WaitOnAddress``) at 0% CPU, and a wedged native call
    cannot be cancelled from Python (#575). The native call releases the GIL
    while parked, so a watchdog thread runs it and the caller waits at most
    ``timeout`` seconds before raising ``TimeoutError`` — letting
    ``_detect_content`` degrade to the pure-Python detector instead of
    deadlocking (and, in the proxy, instead of permanently consuming a
    compression-executor worker — see #575's executor-saturation report).

    # ponytail: can't kill a GIL-released native call; the watchdog frees the
    # caller and the stuck daemon thread is left to die with the process. The
    # upgrade path is the Rust-side fix that makes first-call init non-blocking.
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["result"] = rust_detect(content)
        except BaseException as exc:  # noqa: BLE001 — relayed to the caller's degrade path
            box["error"] = exc

    worker = threading.Thread(target=_run, name="headroom-detect-watchdog", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"native detect_content_type exceeded {timeout:.1f}s watchdog")
    if "error" in box:
        raise box["error"]
    return box["result"]


# Coding agents commonly wrap each tool result in an envelope such as
# ``<returncode>0</returncode>\n<output>...</output>`` (or <stdout>/<stderr>/
# <tool_result>). Those wrapper tags make the native detector read the whole
# payload as markup (HTML/XML) even though the inner content is source code, a
# grep result, or a log. That misroutes to the HTML article-extractor, which
# blanks or corrupts code (dropping identifiers and route converters). Detect on
# the inner payload so the real content type wins; compression still runs on the
# original content.
_DETECTION_ENVELOPE_RE = re.compile(
    r"\A\s*(?:<returncode>\s*-?\d+\s*</returncode>\s*)?"
    r"<(?P<tag>output|stdout|stderr|tool_result|result)>\n?"
    r"(?P<body>.*?)"
    r"\n?</(?P=tag)>\s*\Z",
    re.DOTALL,
)


def _strip_detection_envelope(content: str) -> str:
    """Return the inner payload of a tool-output envelope, for detection only.

    Only strips when the ENTIRE string is a single wrapper envelope, so content
    that merely mentions these tags is left untouched. Never returns an empty
    probe (falls back to the original when the body is blank).
    """
    if "<" not in content:
        return content
    match = _DETECTION_ENVELOPE_RE.match(content)
    if match:
        body = match.group("body")
        if body.strip():
            return body
    return content


def _detect_content(content: str) -> DetectionResult:
    """Detect content type via the native chain, with a safe Windows default.

    Stage-3d (PR5) wired this through `headroom._core.detect_content_type`,
    which runs the magika→unidiff→PlainText chain. On Windows, native Magika
    initialization can leave an ONNX Runtime thread alive after timeout, so the
    default backend there is the pure-Python regex detector.

    Set `HEADROOM_DETECT_BACKEND=rust` or `python` to force a backend.

    The Rust binding returns the legacy `DetectionResult` shape with
    `confidence=1.0` and an empty metadata dict. Existing callers
    only consumed `.content_type` from it; the strategy mapping in
    `_strategy_from_detection` keys off that field alone.
    """
    global _detect_backend_warned, _detect_panic_warned, _detect_native_unhealthy

    # Detect on the unwrapped payload so a tool-output envelope's tags don't get
    # the whole result misclassified as HTML/XML (#route-converter corruption).
    content = _strip_detection_envelope(content)

    backend = _resolve_detect_backend()
    if backend == "python":
        if not _detect_backend_warned:
            _detect_backend_warned = True
            logger.warning(
                "Content detection using pure-Python backend "
                "(native Magika/ONNX detector is unsafe by default on Windows; "
                "override with HEADROOM_DETECT_BACKEND=rust)."
            )
        return _regex_detect_content_type(content)

    if _detect_native_unhealthy:
        # Circuit breaker (#575): the native detector hung once under the
        # watchdog; every later call would wait the full budget and strand
        # another stuck daemon thread, so route straight to pure-Python.
        return _regex_detect_content_type(content)

    # fastembed enables ort's API-24 feature. Entering the native initializer
    # with an older pip ONNX Runtime does not raise: ort recursively re-enters
    # its OnceLock error path and parks forever (#2960). Preflight before the
    # extension call so supported Python 3.10 installs degrade immediately.
    from headroom._ort import rust_ort_runtime_compatible

    if not rust_ort_runtime_compatible():
        _detect_native_unhealthy = True
        logger.warning(
            "Native content detection requires ONNX Runtime 1.24+; "
            "using pure-Python detection for this process."
        )
        return _regex_detect_content_type(content)

    from headroom._core import detect_content_type as _rust_detect

    try:
        # Native detector state can become wedged after an earlier successful
        # call (for example when another test or component initializes ORT).
        # A one-time "verified" fast path therefore turns a later native stall
        # into an unbounded process hang. Keep every call bounded; on timeout
        # the process-wide circuit breaker below makes subsequent calls use the
        # pure-Python detector without spawning more watchdog threads.
        rust_result = _rust_detect_watchdogged(_rust_detect, content, _detect_timeout_secs())
        # Rust's `content_type` is the lowercase string tag (e.g.
        # "json_array"); translate to the Python `ContentType` enum so
        # downstream mapping keys match.
        content_type = ContentType(rust_result.content_type)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException as exc:  # noqa: BLE001
        # A native Rust panic surfaces as pyo3_runtime.PanicException, which
        # derives from BaseException — so ``except Exception`` would miss it and
        # the panic would propagate out as an HTTP 500. Any detector failure
        # (panic, or an unrecognized content-type tag) degrades to the
        # pure-Python detector instead of aborting the request. See #1123.
        # Guard: don't swallow cancellation/control-flow BaseExceptions such
        # as asyncio.CancelledError — keep them propagating.
        if isinstance(exc, asyncio.CancelledError):
            raise
        if isinstance(exc, TimeoutError):
            # Watchdog tripped: the native detector hung (#575). Disable it
            # process-wide so later calls don't each wait the full budget and
            # strand another daemon thread in the wedged native call.
            _detect_native_unhealthy = True
            logger.warning(
                "Native content detector hung (%s); disabling it for this process "
                "and using pure-Python detection.",
                exc,
            )
        elif not _detect_panic_warned:
            _detect_panic_warned = True
            logger.warning(
                "Native content detector failed (%s); falling back to pure-Python detection.",
                type(exc).__name__,
            )
        return _regex_detect_content_type(content)

    # HTML misroute guard (native/magika path): dense punctuation in grep
    # output and build logs (file paths, </>, brackets) can read as markup, so
    # the native detector tags real search results / logs as HTML. Routing those
    # to the HTML article-extractor is lossy — it strips code and identifiers.
    # When the structural log/search detectors positively claim the payload,
    # trust them over the HTML verdict: tracebacks/build output win as LOG
    # (checked first), path:line grep output routes to SEARCH.
    if content_type is ContentType.HTML:
        override = _try_detect_log(content) or _try_detect_search(content)
        if override is not None:
            return override

    # Config misroute guard (native/magika path): magika classifies YAML/TOML/
    # INI as SourceCode, which routes to the (default-disabled) code path and
    # degrades to prose compression. When the structural config detector
    # positively claims the payload, trust it over the SourceCode verdict.
    if content_type is ContentType.SOURCE_CODE:
        config_override = _try_detect_structured_config(content)
        if config_override is not None and config_override.confidence >= 0.7:
            return config_override

    if content_type is ContentType.PLAIN_TEXT:
        regex_result = _regex_detect_content_type(content)
        if regex_result.content_type is not ContentType.PLAIN_TEXT:
            return regex_result
    return DetectionResult(
        content_type=content_type,
        confidence=rust_result.confidence,
        metadata={},
    )


# Content types safe to compress even when read from a file: confidently non-code,
# machine-derived DATA the agent never byte-patches. Everything else (SOURCE_CODE AND
# the PLAIN_TEXT fallback) is protected — critically, code in a language the detector
# does not recognize falls through to PLAIN_TEXT, so protecting PLAIN_TEXT keeps those
# reads safe. The code detector only knows ~6 languages, so we do NOT rely on positively
# identifying code; we release only positively-identified data.
_RELEASABLE_READ_TYPES = frozenset(
    {
        ContentType.JSON_ARRAY,
        ContentType.SEARCH_RESULTS,
        ContentType.BUILD_OUTPUT,  # compiler/test/lint logs
        ContentType.GIT_DIFF,
        ContentType.HTML,
        ContentType.TABULAR,  # CSV/TSV, tables
    }
)


def _read_output_should_be_protected(text: Any) -> bool:
    """Finalize read-protection by CONTENT — protect by default, release only DATA.

    ``_is_read_command`` says "this came from a cat/sed/head file read (and isn't a
    lockfile)". Protection exists so the agent keeps EXACT BYTES of code it will patch.
    Because the code detector recognizes only a handful of languages, we do NOT gate on
    "is this SOURCE_CODE" (that would leave Ruby/C/SQL/… code — seen as PLAIN_TEXT —
    unprotected and lossy-compressed). Instead we PROTECT unless the content is a
    confidently non-code data type (JSON object/array, CSV/tabular, build/test log, git
    diff, HTML, search output), which are never byte-patched and route to a compressor.
    JSON objects are now recognized by the content detector's real parse, so no
    separate object carve-out is needed here.
    """
    if not isinstance(text, str) or not text:
        return False
    try:
        return _detect_content(text).content_type not in _RELEASABLE_READ_TYPES
    except Exception:
        # Detection failure → protect (preserve the byte-exact default).
        return True


def _create_content_signature(
    content_type: str,
    content: str,
    language: str | None = None,
) -> Any:
    """Create a ToolSignature for non-JSON content types.

    This allows TOIN to track compression patterns for code, search results,
    logs, and text - not just JSON arrays.

    Args:
        content_type: The type of content (e.g., "code_aware", "search", "log", "text").
        content: The content being compressed (for structural hints).
        language: Optional language hint for code.

    Returns:
        A ToolSignature for TOIN tracking.
    """
    try:
        from ..telemetry.models import ToolSignature

        # Create a deterministic structure hash based on content type
        # This groups similar content types together for pattern learning
        if language:
            hash_input = f"content:{content_type}:{language}"
        else:
            hash_input = f"content:{content_type}"

        # Add a structural hint from the content (first 100 chars, hashed)
        # This helps differentiate tool outputs of the same type
        content_sample = content[:100] if content else ""
        structure_hint = hashlib.sha256(content_sample.encode()).hexdigest()[:8]
        hash_input = f"{hash_input}:{structure_hint}"

        # Keep SHA256: structure_hash feeds into TOIN which persists to disk.
        # Changing hash function would invalidate all learned patterns.
        structure_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:24]

        return ToolSignature(
            structure_hash=structure_hash,
            field_count=0,  # Not applicable for non-JSON
            has_nested_objects=False,
            has_arrays=False,
            max_depth=0,
        )
    except ImportError:
        return None


# #856 P3b: Anthropic prompt-cache entries live in a 5-minute TTL tier (the
# basis for the 1.25x write multiplier). As a session goes idle the cached
# suffix approaches lapse, so P_alive — the probability the cache survives to
# the next turn — decays toward 0. When P_alive hits 0 the net-cost penalty
# term vanishes and a deep edit near lapse is free to make (the suffix is
# about to be rebuilt cold anyway). This is the cache TTL, NOT the
# session-tracker cleanup TTL (``PrefixFreezeConfig.session_ttl_seconds``).
_NET_COST_CACHE_TTL_SECONDS = 300.0


def _net_cost_cache_ttl_seconds() -> float:
    """Provider cache TTL (seconds) used to decay P_alive from idle time.

    Defaults to Anthropic's 5-minute tier; overridable via
    ``HEADROOM_NET_COST_CACHE_TTL_SECONDS`` for other providers/tiers. A
    malformed or non-positive value falls back to the default with a warning
    rather than producing a divide-by-zero or negative TTL (same posture as
    the other ``HEADROOM_NET_COST_*`` env guards).
    """
    raw = os.environ.get("HEADROOM_NET_COST_CACHE_TTL_SECONDS", "")
    if not raw:
        return _NET_COST_CACHE_TTL_SECONDS
    try:
        ttl = float(raw)
    except ValueError:
        logger.warning(
            "HEADROOM_NET_COST_CACHE_TTL_SECONDS malformed; using default %s",
            _NET_COST_CACHE_TTL_SECONDS,
        )
        return _NET_COST_CACHE_TTL_SECONDS
    if not math.isfinite(ttl) or ttl <= 0.0:
        logger.warning(
            "HEADROOM_NET_COST_CACHE_TTL_SECONDS invalid; using default %s",
            _NET_COST_CACHE_TTL_SECONDS,
        )
        return _NET_COST_CACHE_TTL_SECONDS
    return ttl


def _gain_bucket(gain: float) -> str:
    """Quantize a net-cost gain into a coarse magnitude band for markers.

    The net-cost gate emits a ``netcost:skip:<band>`` transform marker. Using
    the raw rounded gain would make every distinct value a unique marker and
    blow up the cardinality of any ``transforms_applied`` aggregation. Bands
    keep the signal (rough magnitude + sign) while bounding cardinality to a
    handful of values. The exact gain is still logged at INFO for debugging.
    """
    if not math.isfinite(gain):
        return "nan"
    mag = abs(gain)
    if mag < 100:
        band = "lt100"
    elif mag < 1000:
        band = "lt1k"
    elif mag < 10000:
        band = "lt10k"
    else:
        band = "gte10k"
    if gain == 0:
        return "0"
    return ("neg_" if gain < 0 else "") + band


def _netcost_message_tokens(message: dict[str, Any], tokenizer: Tokenizer) -> int:
    """Token count of a message for net-cost suffix (S) estimation.

    String content is counted directly. Block-list content is delegated to the
    canonical block counter, which knows how to price non-text blocks.

    This function used to walk the list itself and fall back to
    ``str(block)`` for anything that was not ``text`` or ``tool_result``, on the
    stated assumption that such blocks "rarely dominate a suffix". An ``image``
    block is the exception that breaks it: ``str()`` embeds the whole base64
    payload, so one screenshot counted ~100,000 tokens instead of ~1,600
    (57x-146x over, growing with image size).

    That mattered because S is the cache-bust cost — the tokens re-written if
    message *j* is mutated — so an image inflated S for **every message before
    it**, and the break-even gate then refused to compress any of them.
    ``BaseTokenizer._count_content_parts`` already solves this (see its "1MB
    image = ~330K fake tokens without this" guard); this walk simply predated
    it. Delegating also means new block types are priced in one place.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return tokenizer.count_text(content)
    if not isinstance(content, list):
        return tokenizer.count_text(str(content))
    return count_content_blocks(content, tokenizer.count_text)


class CompressionCache:
    """Two-tier compression cache with TTL.  Thread-safe.

    Tier 1 (skip set): content hashes that won't compress — instant skip,
    near-zero memory (just ints in a set).

    Tier 2 (result cache): compressed results for content that DID compress —
    reuse the compressed text on subsequent requests.

    Entries expire after TTL (default 30min). No max-entries cap — TTL is the
    natural bound. Memory grows proportional to compressible content × TTL,
    which is bounded by session duration.

    Uses in-process dict for ultra-fast lookups (~100ns). Could be backed
    by memcached/Redis for multi-process deployments.

    Thread safety: a ``threading.Lock`` guards all read-modify-write
    operations.  The ``apply()`` path runs compression inside a
    ``ThreadPoolExecutor``; without the lock concurrent cache misses for
    the same content would produce duplicate compression work (correct but
    wasteful) and metrics counters would drift.
    """

    def __init__(self, ttl_seconds: int = 1800):
        import threading

        # Tier 2: compressed results {hash: (text, ratio, strategy, timestamp)}
        self._results: dict[int, tuple[str, float, str, float]] = {}
        # Tier 1: hashes of content that won't compress {hash: timestamp}
        self._skip: dict[int, float] = {}
        # Callbacks invoked (outside the lock) whenever ``clear()`` runs, so
        # sibling state keyed by the same content hashes (e.g. the router's
        # frozen-verdict store) can be reset in lock-step with the cache.
        self._on_clear: list[Any] = []
        self._ttl_seconds = ttl_seconds
        # Metrics
        self._hits = 0
        self._misses = 0
        self._skip_hits = 0
        self._evictions = 0
        self._total_lookup_ns = 0
        self._lookup_count = 0
        self._lock = threading.Lock()

    def get(self, key: int) -> tuple[str, float, str] | None:
        """Get cached compression result.  Thread-safe.

        Returns (compressed_text, ratio, strategy) or None if not found/expired.
        Use is_skipped() first to check if content is known non-compressible.
        """
        t0 = time.perf_counter_ns()
        with self._lock:
            entry = self._results.get(key)
            if entry is not None:
                compressed, ratio, strategy, created_at = entry
                if (time.monotonic() - created_at) < self._ttl_seconds:
                    self._hits += 1
                    self._total_lookup_ns += time.perf_counter_ns() - t0
                    self._lookup_count += 1
                    return (compressed, ratio, strategy)
                else:
                    del self._results[key]
                    self._evictions += 1
            self._misses += 1
            self._total_lookup_ns += time.perf_counter_ns() - t0
            self._lookup_count += 1
            return None

    def is_skipped(self, key: int) -> bool:
        """Check if content is known non-compressible (Tier 1).  Thread-safe."""
        with self._lock:
            ts = self._skip.get(key)
            if ts is not None:
                if (time.monotonic() - ts) < self._ttl_seconds:
                    self._skip_hits += 1
                    return True
                else:
                    del self._skip[key]
                    self._evictions += 1
            return False

    def put(self, key: int, compressed: str, ratio: float, strategy: str) -> None:
        """Store a compressed result (Tier 2).  Thread-safe."""
        with self._lock:
            self._results[key] = (compressed, ratio, strategy, time.monotonic())

    def mark_skip(self, key: int) -> None:
        """Mark content as non-compressible (Tier 1).  Thread-safe."""
        with self._lock:
            self._skip[key] = time.monotonic()

    def move_to_skip(self, key: int) -> None:
        """Move a result to skip set (threshold tightened, no longer qualifies).
        Thread-safe."""
        with self._lock:
            self._results.pop(key, None)
            self._skip[key] = time.monotonic()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._results)

    @property
    def skip_size(self) -> int:
        with self._lock:
            return len(self._skip)

    @property
    def stats(self) -> dict[str, int | float]:
        with self._lock:
            avg_ns = self._total_lookup_ns / self._lookup_count if self._lookup_count else 0
            return {
                "cache_hits": self._hits,
                "cache_skip_hits": self._skip_hits,
                "cache_misses": self._misses,
                "cache_evictions": self._evictions,
                "cache_size": len(self._results),
                "cache_skip_size": len(self._skip),
                "cache_avg_lookup_ns": avg_ns,
            }

    def register_on_clear(self, callback: Any) -> None:
        """Register a callback fired (outside the lock) on every ``clear()``.

        Lets state keyed by the same content hashes — e.g. the router's
        ``_frozen_verdicts`` store — be reset together with the cache so it
        cannot outlive the results/skip tiers it shadows.
        """
        self._on_clear.append(callback)

    def clear(self) -> None:
        """Clear all entries (e.g., on session end).  Thread-safe."""
        with self._lock:
            self._results.clear()
            self._skip.clear()
        # Fire callbacks outside the lock to avoid cross-lock ordering issues.
        for callback in self._on_clear:
            callback()


class CompressionStrategy(Enum):
    """Available compression strategies."""

    CODE_AWARE = "code_aware"
    SMART_CRUSHER = "smart_crusher"
    SEARCH = "search"
    LOG = "log"
    KOMPRESS = "kompress"
    TEXT = "text"
    DIFF = "diff"
    HTML = "html"
    TABULAR = "tabular"
    CONFIG = "config"
    MIXED = "mixed"
    PASSTHROUGH = "passthrough"


@dataclass
class RoutingDecision:
    """Record of a single routing decision."""

    content_type: ContentType
    strategy: CompressionStrategy
    original_tokens: int
    compressed_tokens: int
    confidence: float = 1.0
    section_index: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens


@dataclass
class RouterCompressionResult:
    """Result from ContentRouter with routing metadata.

    Attributes:
        compressed: The compressed content.
        original: Original content before compression.
        strategy_used: Primary strategy used for compression.
        routing_log: List of routing decisions made.
        sections_processed: Number of content sections processed.
        strategy_chain: Every strategy attempted in order. For a direct
            hit it's a single entry; for the SMART_CRUSHER → KOMPRESS →
            LOG fallback chain it's three. Lets log readers see *how*
            we got to the final compressor without parsing the
            decision_reason string.
        cache_hit: True when this result was reused from a cache
            instead of a fresh compression run. compress() itself
            never sets this (only apply() has the router-internal
            two-tier cache); it is set by callers that cache unit
            results — e.g. the OpenAI Responses handler marks reused
            units via ``replace(router_result, cache_hit=True)``.
    """

    compressed: str
    original: str
    strategy_used: CompressionStrategy
    routing_log: list[RoutingDecision] = field(default_factory=list)
    sections_processed: int = 1
    strategy_chain: list[str] = field(default_factory=list)
    cache_hit: bool = False

    @property
    def total_original_tokens(self) -> int:
        """Total tokens before compression."""
        return sum(r.original_tokens for r in self.routing_log)

    @property
    def total_compressed_tokens(self) -> int:
        """Total tokens after compression."""
        return sum(r.compressed_tokens for r in self.routing_log)

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio."""
        if self.total_original_tokens == 0:
            return 1.0
        return self.total_compressed_tokens / self.total_original_tokens

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved."""
        return max(0, self.total_original_tokens - self.total_compressed_tokens)

    @property
    def savings_percentage(self) -> float:
        """Percentage of tokens saved."""
        if self.total_original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.total_original_tokens) * 100

    def summary(self) -> str:
        """Human-readable routing summary."""
        if self.strategy_used == CompressionStrategy.MIXED:
            strategies = {r.strategy.value for r in self.routing_log}
            return (
                f"Mixed content: {self.sections_processed} sections, "
                f"routed to {strategies}. "
                f"{self.total_original_tokens:,}→{self.total_compressed_tokens:,} tokens "
                f"({self.savings_percentage:.0f}% saved)"
            )
        else:
            return (
                f"Pure {self.strategy_used.value}: "
                f"{self.total_original_tokens:,}→{self.total_compressed_tokens:,} tokens "
                f"({self.savings_percentage:.0f}% saved)"
            )


@dataclass
class ContentRouterConfig:
    """Configuration for intelligent content routing.

    Attributes:
        enable_code_aware: Enable AST-based code compression.
        enable_smart_crusher: Enable JSON array compression.
        enable_search_compressor: Enable search result compression.
        enable_log_compressor: Enable build/test log compression.
        enable_tabular_compressor: Enable CSV/TSV/markdown-table compression.
        enable_config_compressor: Enable YAML/TOML/INI config compression.
        enable_image_optimizer: Enable image token optimization.
        prefer_code_aware_for_code: Use CodeAware over Kompress for code.
        min_section_tokens: Minimum tokens for a section to compress.
        fallback_strategy: Strategy when no compressor matches.
        skip_user_messages: Never compress user messages (they're the subject).
        skip_recent_messages: Don't compress last N messages (likely the subject).
        protect_analysis_context: Detect "analyze/review" intent, skip compression.
    """

    # Enable/disable specific compressors
    enable_code_aware: bool = False  # Disabled: use code graph MCP tools instead
    enable_kompress: bool = True  # Kompress: ModernBERT token compressor
    enable_smart_crusher: bool = True
    enable_search_compressor: bool = True
    enable_log_compressor: bool = True
    enable_tabular_compressor: bool = True  # CSV/TSV/markdown tables via SmartCrusher
    enable_config_compressor: bool = True  # YAML/TOML/INI structural compression
    enable_html_extractor: bool = True  # HTML content extraction
    enable_image_optimizer: bool = True  # Image token optimization

    # Routing preferences
    prefer_code_aware_for_code: bool = (
        True  # Route code to CodeAware over Kompress for higher, syntax-safe compression
    )
    # Route ALL compressible content to Kompress, skipping per-type selection.
    # Tool exclusion (Read/Glob/...) and reversibility gates still apply.
    force_kompress_all: bool = False

    # Opt-in selection of EXTERNAL (non-built-in) `headroom.compressor` names to
    # route real traffic through. `None`/empty (the default) means the external-
    # dispatch branch in `_apply_strategy_to_content` is inert and the request
    # path is byte-identical to today. Built-in names are NOT put here — they are
    # selected via the `enable_*` flags above (see the proxy's
    # `_apply_compressor_selection`). `"*"` activates every discovered external
    # compressor. The router resolves these names against `compressor_registry`
    # and, when a block's detected content type matches an active external
    # compressor's declared `content_types`, runs it via the registry contract
    # instead of the built-in if/elif — fail-open back to the built-in path.
    active_external_compressors: list[str] | None = None

    # No-CCR lossless mode. When True the router compresses LOG/SEARCH/DIFF
    # content with format-native lossless compaction (headroom.transforms.
    # lossless_compaction) instead of the lossy Rust drop path, and never
    # emits a `<<ccr:…>>` / `Retrieve …` retrieval marker. SmartCrusher is
    # additionally forced marker-free via smart_crusher_lossless_only.
    lossless: bool = False
    # Cross-turn (whole-conversation) verbatim de-dup. Replaces a contiguous span
    # in a later tool output that already appeared verbatim in an earlier tool
    # output with an in-context pointer. Prefix-monotonic (cache-safe) and
    # information-preserving (the original stays in context). Env: HEADROOM_DEDUPE=1.
    # Runs in both modes: lossless references verbatim/folded content; CCR mode
    # references the earlier block's kompressed-but-CCR-recoverable form
    # (deterministic content-hash → stable → still cache-safe, no added loss).
    # Per request the fold is skipped when the caller reports the serving path
    # cannot resolve the in-context `[↑NL same as msg M]` pointer
    # (`apply(cross_turn_dedup_recoverable=False)`, e.g. OpenAI chat-completions
    # streaming, where no CCR retrieval tool can be injected and clients never
    # show the model numbered messages).
    enable_cross_turn_dedup: bool = False
    # Lossless-then-lossy. In lossy mode (not `lossless`), after a byte/data
    # lossless fold (search/log/text) run the aggressive lossy compressor
    # (Kompress) on the FOLDED remainder and keep it iff it removes a further
    # meaningful chunk — recovering the semantic word-drop that plain lossless
    # leaves on the table while never doing worse than the fold. DIFF folds are
    # never lossy-chained (Kompressing hunks breaks `git apply`). No-op in
    # lossless-only mode. Env: HEADROOM_LOSSLESS_THEN_LOSSY=1.
    lossless_then_lossy: bool = False
    min_section_tokens: int = 20  # Min tokens to compress a section

    # Fallback: Kompress handles unknown/mixed content instead of passing through
    fallback_strategy: CompressionStrategy = CompressionStrategy.KOMPRESS

    # Protection: Don't compress content that's likely the subject of analysis
    skip_user_messages: bool = True  # User messages contain what they want analyzed
    protect_recent_code: int = 4  # Don't compress CODE in last N messages (0 = disabled)
    protect_analysis_context: bool = True  # Detect "analyze/review" intent, protect code

    # Protection: failed tool calls / error outputs stay verbatim (issue #847).
    # The model needs exact tracebacks and error text to recover; compressing
    # them measurably hurts agent recovery. Outputs above the size cap still
    # compress — LogCompressor preserves error lines in big logs, so the two
    # features stay complementary.
    protect_error_outputs: bool = True
    error_protection_max_chars: int = 8000  # ~2K tokens; larger errors compress

    # Cache safety: assistant text-block compression.
    # Default OFF. Assistant content is echoed back by the client in
    # subsequent turns and becomes part of the upstream provider's
    # prefix cache (Anthropic cache_control, DeepSeek/OpenAI auto).
    # Compressing it changes the bytes that must match for a cache
    # hit on the next turn. The hash-keyed result cache makes the
    # compressed output deterministic *within* a process, but cache
    # eviction or proxy restart can re-compress with a different
    # output for stochastic compressors — and that miss costs the
    # whole prefix discount. Enable only for deployments routed to
    # backends that don't honor cache_control AND whose compressors
    # are byte-deterministic.
    compress_assistant_text_blocks: bool = False

    # Minimum content length (in chars) at which a text or tool_result
    # block is considered for compression. Below this, the overhead of
    # routing/detecting/caching exceeds any savings, so the block is
    # passed through verbatim.
    min_chars_for_block_compression: int = 500

    # Adaptive Read protection: fraction of total messages to protect from
    # compression.  At 10 msgs, protects ~5 Reads.  At 100 msgs, protects ~10.
    # Old Reads beyond this window become compressible even though they are
    # in DEFAULT_EXCLUDE_TOOLS.  0.0 = always exclude all (old behavior).
    protect_recent_reads_fraction: float = (
        0.0  # 0.0 = protect ALL excluded-tool outputs (safest for coding agents)
    )

    # Acceptance threshold. The gate accepts a compression when
    # compression_ratio < min_ratio (ratio = compressed/original). Default 1.0 at
    # every pressure = accept ANY real shrink (ratio < 1.0): any token saved is
    # worth taking. The prefix-cache-bust cost this once guarded against (a small
    # win can cost more than it saves once the invalidated suffix is re-written)
    # is instead handled precisely by the opt-in net-cost policy
    # (HEADROOM_NET_COST_POLICY=1); tool-output accuracy by the reversibility
    # gate — both independent of this floor. Lower these (e.g. 0.85/0.65) to
    # restore a savings floor that only accepts wins big enough to justify the
    # cache bust as context fills.
    min_ratio_relaxed: float = 1.0  # accept any shrink (no savings floor)
    min_ratio_aggressive: float = 1.0  # same under pressure; net-cost is the guard

    # CCR (Compress-Cache-Retrieve) settings for SmartCrusher
    ccr_enabled: bool = True  # Enable CCR marker injection for reversible compression
    ccr_inject_marker: bool = True  # Add retrieval markers to compressed content
    smart_crusher_max_items_after_crush: int | None = None
    smart_crusher_with_compaction: bool = True
    # Strict lossless-only mode for SmartCrusher. None → leave the
    # crusher config's own value untouched; True/False force it. Wired
    # from the proxy's `HEADROOM_LOSSLESS_ONLY` env var so a real session
    # can run marker-free without constructing the crusher by hand.
    smart_crusher_lossless_only: bool | None = None

    # Prompt-conditioned relevance split for the KEEP/DROP tail. When enabled,
    # LOG/SEARCH output is segmented into records, each scored against the
    # request's information need (user prompt + triggering tool-call args) via
    # `relevance` below; high-relevance records are kept verbatim and the
    # low-relevance tail is Kompressed. Works in both modes: in lossless mode
    # the tail is marker-free; in CCR mode it carries a retrieval marker (via
    # ccr_inject_marker) so dropped detail stays retrievable. On by default; the
    # embedding model is pre-warmed in the background (BM25 scores until it's
    # cached) so no request ever blocks on the download.
    relevance_split: bool = True
    relevance: RelevanceScorerConfig = field(default_factory=RelevanceScorerConfig)
    # Optional latency guard: skip the split when an output segments into more
    # than this many records, capping embedding work on the request thread.
    # 0 = no cap (default): every record is scored regardless of size. Set a
    # positive value to bound per-request embedding cost on very large outputs.
    relevance_max_records: int = 0
    # Adaptive KEEP/DROP cut: when True (default), the threshold is the natural
    # relevant/irrelevant break in each output's score distribution (Otsu),
    # floored by relevance.relevance_threshold — it moves with the content
    # instead of a fixed constant. False uses the fixed threshold exactly.
    relevance_adaptive_threshold: bool = True

    # Tag protection: preserve custom/workflow XML tags from text compression.
    # When False (default), entire <custom-tag>content</custom-tag> blocks are
    # protected verbatim.  When True, only the tag markers are protected and
    # the content between them can be compressed.
    compress_tagged_content: bool = False

    # Tools to exclude from compression (output passed through unmodified)
    # Set to None to use DEFAULT_EXCLUDE_TOOLS, or provide custom set.
    # NOTE: headroom_retrieve is excluded unconditionally regardless of this
    # setting, even if this is explicitly set to an empty set -- recompressing
    # its output would write a new <<ccr:hash>> marker the agent can never
    # redeem (see the ccr_retrieve_tool_ids guards in apply()).
    exclude_tools: set[str] | None = None

    # Excluded tools are protected only from *lossy* compression. Their output
    # is still given information-preserving compaction by detected shape (grep
    # -> ripgrep --heading fold; logs -> ANSI strip + run-collapse; JSON ->
    # whitespace-minify, data-lossless), in every path — see
    # ``_lossless_compact_excluded``. Always recoverable, so no config gate.

    # Shell tool names (case-insensitive). Their output is non-excluded/lossy,
    # BUT a read-only *search* run through them (grep/rg/git grep) yields byte-
    # losslessly foldable output — folded instead of lossy-compressed. See
    # ``_bash_search_fold``. Config so new harness tool names / search programs
    # can be added without code changes.
    bash_tool_names: frozenset[str] = frozenset({"bash", "shell", "local_shell"})
    bash_search_commands: frozenset[str] = frozenset(
        {"grep", "egrep", "fgrep", "rg", "ripgrep", "ag", "ack"}
    )

    # Read lifecycle management (stale/superseded detection)
    read_lifecycle: ReadLifecycleConfig = field(default_factory=ReadLifecycleConfig)

    # Per-tool compression profiles (tool_name → CompressionProfile)
    # Set to None to use DEFAULT_TOOL_PROFILES from config
    tool_profiles: dict[str, Any] | None = None

    # SmartCrusher configuration override. None → transforms-level
    # SmartCrusherConfig() defaults. Lets deployments tune the lossless
    # dispatch threshold and compaction heuristics without constructing
    # the crusher themselves.
    smart_crusher: Any | None = None

    # Structural compressor configuration overrides. None preserves each
    # compressor's dataclass defaults. The proxy wires environment-backed
    # overrides into these objects, while ccr_inject_marker/search grouping are
    # still enforced by ContentRouter so global safety flags win consistently.
    search_compressor: Any | None = None
    log_compressor: Any | None = None
    diff_compressor: Any | None = None
    text_crusher: Any | None = None

    # Group search-compressor output by file (`rg --heading` style).
    # Default False; the proxy enables it in token mode.
    search_group_by_file: bool = False


class ContentRouter(Transform):
    """Intelligent router that selects optimal compression strategy.

    ContentRouter is the recommended entry point for Headroom's compression.
    It analyzes content and routes it to the most appropriate compressor,
    handling mixed content by splitting and reassembling.

    Key Features:
    - Automatic content type detection
    - Source hint support for high-confidence routing
    - Mixed content handling (split → route → reassemble)
    - Graceful fallback when compressors unavailable
    - Rich routing metadata for debugging

    Example:
        >>> router = ContentRouter()
        >>>
        >>> # Automatically uses CodeAwareCompressor
        >>> result = router.compress(python_code)
        >>> print(result.strategy_used)  # CompressionStrategy.CODE_AWARE
        >>>
        >>> # Automatically uses SmartCrusher
        >>> result = router.compress(json_array)
        >>> print(result.strategy_used)  # CompressionStrategy.SMART_CRUSHER
        >>>
        >>> # Splits and routes each section
        >>> result = router.compress(readme_with_code)
        >>> print(result.strategy_used)  # CompressionStrategy.MIXED

    Pipeline Integration:
        >>> pipeline = TransformPipeline([
        ...     ContentRouter(),   # Handles ALL content types
        ... ])
    """

    name: str = "content_router"

    # Lossy summarizers that emit a CCR retrieve marker only when they store the
    # original — a marker-less result from one of these is unrecoverable. Tool
    # ground truth (role="tool") must not be replaced by such a result (#1307).
    LOSSY_UNMARKED_STRATEGIES = frozenset(
        {
            CompressionStrategy.KOMPRESS,
            CompressionStrategy.TEXT,
            CompressionStrategy.CODE_AWARE,
        }
    )

    # Lossless-then-lossy gate: the lossy pass replaces the byte-exact fold only
    # if it saves at least this fraction MORE tokens than the fold already did
    # (default 0.05 => Kompress must cut >= 5% beyond the fold). Below that the
    # marginal lossy win isn't worth the accuracy cost when a lossless fold is
    # already in hand, so the pure fold is kept. Overridable at runtime via env
    # HEADROOM_LOSSY_MIN_EXTRA_SAVINGS (read in __init__) so the gate can be tuned
    # per deployment without a code edit + overlay rebuild. Higher = stricter
    # (fewer lossy chains, safer); 0 = keep the lossy pass on any improvement.
    _DEFAULT_LOSSY_MIN_EXTRA_SAVINGS = 0.05

    # Entry cap for the `_lossless_first` memo. Sized for "the blocks of a
    # handful of in-flight requests", not for cross-request reuse — on overflow
    # the whole dict is dropped rather than evicted entry-by-entry, which keeps
    # the memo O(1) and unlockable at the cost of an occasional cold start.
    _LOSSLESS_FIRST_MEMO_MAX = 256

    def __init__(
        self,
        config: ContentRouterConfig | None = None,
        observer: Any = None,
    ):
        """Initialize content router.

        Args:
            config: Router configuration. Uses defaults if None.
            observer: Optional `CompressionObserver` (see
                `headroom.transforms.observability`) called once per
                routing decision after `compress()` finishes. The
                proxy's `PrometheusMetrics` is the production
                implementation — it increments per-strategy counters
                so silent regressions become visible. `None` disables
                observation; pick one explicitly per the no-fallback
                rule in the audit doc.
        """
        self.config = config or ContentRouterConfig()
        # No-CCR lossless mode is self-consistent regardless of how the config
        # was built: force marker-free output and marker-free SmartCrusher so
        # the invariant (no `<<ccr:…>>` / `Retrieve …`) holds even when a caller
        # constructs ContentRouterConfig(lossless=True) directly.
        if self.config.lossless:
            self.config.ccr_inject_marker = False
            self.config.smart_crusher_lossless_only = True
        self._observer = observer

        # Name-addressable compressor inventory: built-in metadata + opt-in
        # discovery of `headroom.compressor` entry points. Inventory only —
        # built-ins are still constructed and dispatched by the if/elif below,
        # so this changes no routing. Exposed for selection/routing wiring in a
        # follow-up. Failure to build it must never break the router, so it is
        # fail-open to an empty registry.
        try:
            self.compressor_registry: CompressorRegistry = _build_compressor_registry(self)
        except Exception as exc:  # noqa: BLE001 - inventory is non-critical
            logger.debug("compressor registry unavailable: %s", exc)
            self.compressor_registry = CompressorRegistry()

        # Resolve the opt-in EXTERNAL compressor selection ONCE — the registry
        # and `config.active_external_compressors` are both fixed after
        # construction. Empty unless the operator selected a non-built-in
        # compressor (via `--compressor`), so the external-dispatch branch in
        # `_apply_strategy_to_content` is a single cheap guard and the default
        # request path stays byte-identical. Built-in registry entries are
        # filtered out here so they are only ever dispatched by the if/elif.
        self._active_external_compressors: list[Any] = self._resolve_active_external_compressors()

        # Lazy-loaded compressors
        self._code_compressor: Any = None
        self._smart_crusher: Any = None
        self._search_compressor: Any = None
        self._log_compressor: Any = None
        self._diff_compressor: Any = None
        self._html_extractor: Any = None
        self._tabular_compressor: Any = None
        self._config_compressor: Any = None
        self._kompress: Any = None
        # Stage B relevance split (lazy; None until first use, sentinel-checked
        # via _relevance_scorer_tried so a failed load isn't retried per call).
        self._relevance_scorer: Any = None
        self._relevance_scorer_tried: bool = False
        self._relevance_prewarm_started: bool = False
        # STAGE 0 memo: `(hash(content), len(content), strategy) -> (best, label)`.
        # `_lossless_first` is a pure function of its arguments but is called
        # twice for the same block (`_has_lossless_fold` probes it, then STAGE 0
        # recomputes it), which also means a registered lossless provider gets
        # invoked twice. Bounded by `_LOSSLESS_FIRST_MEMO_MAX` with a wholesale
        # clear (no LRU bookkeeping — this is a within-request hit, not a
        # long-lived cache; `_cache` is the cross-request one).
        self._lossless_first_memo: dict[
            tuple[int, int, CompressionStrategy, int], tuple[str, str | None]
        ] = {}
        # Companion memo for the third-party half of STAGE 0, keyed by content
        # (plus the provider generation) but NOT by strategy — the provider
        # contract takes no strategy, so the two probes above, which pass
        # different strategies for the same block, share one provider
        # invocation. See `_lossless_provider_result`.
        self._lossless_provider_memo: dict[tuple[int, int, int], tuple[str, str] | None] = {}

        # tool_call_id → compact args text, populated by _build_tool_name_map.
        self._tool_call_args: dict[str, str] = {}
        # tool_call_id → raw shell command (bash-search fold), same population.
        self._tool_call_commands: dict[str, str] = {}

        # Phase 0 (#1171): cap the input size handed to kompress (ModernBERT
        # ONNX). Its inference scales O(tokens) and runs synchronously on the
        # request thread under the 30s compression budget; above this ceiling we
        # route to the fast LogCompressor instead so the request path stays
        # bounded. ~4 chars/token is a cheap proxy (no tokenizer needed; counts
        # dense JSON/code correctly, unlike word count). 0 disables the gate.
        try:
            self._kompress_max_tokens: int = int(
                os.environ.get("HEADROOM_KOMPRESS_MAX_TOKENS", "50000")
            )
        except ValueError:
            self._kompress_max_tokens = 50000
        self._kompress_gate_fires: int = 0
        # Phase 2 (#1171): when enabled, the size-gate routes oversized text to
        # the fast extractive TextCrusher (real prose savings) instead of the
        # LogCompressor (~0 savings on prose). Opt-in, default off.
        self._text_crusher_enabled: bool = os.environ.get(
            "HEADROOM_TEXT_CRUSHER", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        self._text_crusher: Any = None
        # Cross-turn dedup: config field OR env HEADROOM_DEDUPE (robust to how the
        # config was built). Runs in BOTH modes — the call site in ``apply()`` has
        # no lossless guard, and ``_cross_turn_dedup_messages`` documents working
        # against lossless folds and CCR-recoverable forms alike. (This comment
        # previously claimed "lossless mode only", which reads as "inert in your
        # config" to anyone auditing why dedup never fired.)
        self._cross_turn_dedup_enabled: bool = (
            self.config.enable_cross_turn_dedup
            or os.environ.get("HEADROOM_DEDUPE", "").strip().lower() in ("1", "true", "yes", "on")
        )
        # EXPERIMENT (HEADROOM_EXPERIMENTAL_READ_KEEP_RATIO): file reads are
        # protected verbatim by default so the agent keeps exact bytes to patch.
        # This probe instead LIGHTLY lossy-compresses a protected read with
        # Kompress at the given keep ratio (e.g. 0.9 = keep ~90%), trading a small
        # resolve risk for savings on the biggest untouched bucket (code reads).
        # 0/unset = OFF (verbatim, today's behavior). Resolve-risk probe only.
        try:
            self._exp_read_keep_ratio: float = float(
                os.environ.get("HEADROOM_EXPERIMENTAL_READ_KEEP_RATIO", "") or 0
            )
        except ValueError:
            self._exp_read_keep_ratio = 0.0
        # Lossless-then-lossy. Config field OR env HEADROOM_LOSSLESS_THEN_LOSSY.
        # Only takes effect in lossy mode (STAGE 0 guards on `not config.lossless`).
        self._lossless_then_lossy: bool = self.config.lossless_then_lossy or os.environ.get(
            "HEADROOM_LOSSLESS_THEN_LOSSY", ""
        ).strip().lower() in ("1", "true", "yes", "on")
        # Lossless-then-lossy gate: keep the lossy chain only if it saves at least
        # this fraction MORE than the fold. Env override
        # (HEADROOM_LOSSY_MIN_EXTRA_SAVINGS) falls back to the class default; a
        # malformed value falls back rather than crashing.
        try:
            self._lossy_min_extra_savings: float = float(
                os.environ.get("HEADROOM_LOSSY_MIN_EXTRA_SAVINGS")
                or self._DEFAULT_LOSSY_MIN_EXTRA_SAVINGS
            )
        except (TypeError, ValueError):
            self._lossy_min_extra_savings = self._DEFAULT_LOSSY_MIN_EXTRA_SAVINGS

        # TOIN integration for cross-strategy learning
        self._toin: Any = None

        # F2.2: per-request CompressionPolicy, set from
        # ``kwargs["compression_policy"]`` at the start of ``apply()``
        # and read by ``_record_to_toin`` to gate TOIN writes when
        # ``policy.toin_read_only`` is true (Subscription mode).
        # Defaults to ``None`` so direct ``compress()`` callers (e.g.
        # tests, hand-written pipelines that don't go through the
        # proxy) keep pre-F2.2 behaviour: TOIN writes are not gated.
        # Same pattern the existing ``_runtime_target_ratio`` /
        # ``_runtime_kompress_model`` fields below use.
        self._runtime_compression_policy: Any = None

        self._cache = CompressionCache()

        # Cache-churn fix (HEADROOM_FREEZE_BLOCK_DECISION, default off):
        # freeze each block's compress-vs-passthrough verdict on first
        # sighting so it stops depending on the per-turn ``min_ratio`` drift.
        # Keyed by the same ``content_key`` as ``_cache``; value True =
        # "compress", False = "skip". Only populated when the flag is on.
        #
        # Bounding: this store shadows ``_cache`` entries, so left unbounded it
        # would leak verdicts whose cache entries have long since expired. It is
        # (a) cleared in lock-step with the cache via ``register_on_clear`` and
        # (b) capped at ``_frozen_verdicts_max`` with simple insertion-order
        # (FIFO) eviction, mirroring the cache's bounded posture.
        #
        # Locking: ``CompressionCache`` guards every mutation with a lock and
        # the parallel compression pass writes verdicts from worker threads, so
        # we match that posture with a dedicated lock rather than relying on
        # GIL atomicity (which would not protect the read-then-evict sequence).
        self._frozen_verdicts: dict[int, bool] = {}
        # The store is process-wide (one router per pipeline, shared by every
        # session), so the cap must scale with the number of CONCURRENT
        # sessions, not one user's workload: at org scale (many users behind
        # one sidecar) 4096 churns in minutes and FIFO eviction lets tightened
        # thresholds flip a still-cached block's verdict — a prefix bust.
        # Read at construction so tests and multi-tenant deployments can size
        # it via HEADROOM_FROZEN_VERDICTS_MAX without a module reload.
        try:
            self._frozen_verdicts_max = max(
                256, int(os.environ.get("HEADROOM_FROZEN_VERDICTS_MAX", "4096"))
            )
        except ValueError:
            self._frozen_verdicts_max = 4096
        self._frozen_lock = threading.Lock()
        # Reset verdicts whenever the shadowed cache is cleared.
        self._cache.register_on_clear(self._clear_frozen_verdicts)
        # Instrumentation for the freeze (HEADROOM_FREEZE_BLOCK_DECISION): a
        # "pin" is a turn where the frozen compress verdict overrode a tightened
        # per-turn ``min_ratio`` that would otherwise downgrade the block to
        # skip — i.e. revert it to original text and bust the provider prefix
        # cache. Counting pins isolates the freeze's attributable payoff.
        self._freeze_pin_hits = 0
        self._freeze_pin_chars = 0

    def _record_freeze_pin(self, content: str, cached_ratio: float) -> None:
        """Count one freeze divergence (thread-safe) and log it.

        Fires only when the frozen "compress" verdict overrode a tightened
        per-turn ``min_ratio`` that would have downgraded the block to skip (a
        provider cache bust). ``preserved`` is a char-based proxy for the
        compression saving kept alive by not reverting to the original.
        """
        preserved = max(0, int(len(content) * (1.0 - cached_ratio)))
        with self._frozen_lock:
            self._freeze_pin_hits += 1
            self._freeze_pin_chars += preserved
            hits = self._freeze_pin_hits
        logger.info(
            f"FREEZE-PIN: pins={hits} cached_ratio={cached_ratio:.3f} "
            f"preserved_chars~={preserved} (cache bust avoided)"
        )

    def _record_frozen_verdict(self, content_key: int, verdict: bool) -> None:
        """Record a frozen verdict (thread-safe, bounded FIFO eviction).

        Caps the store at ``_frozen_verdicts_max`` entries, evicting the
        oldest insertion when full, so it cannot grow without bound across a
        long-lived process. Mirrors ``CompressionCache``'s locked mutations.
        """
        with self._frozen_lock:
            if (
                content_key not in self._frozen_verdicts
                and len(self._frozen_verdicts) >= self._frozen_verdicts_max
            ):
                # Evict oldest insertion (dicts preserve insertion order).
                oldest = next(iter(self._frozen_verdicts))
                del self._frozen_verdicts[oldest]
            self._frozen_verdicts[content_key] = verdict

    def _get_frozen_verdict(self, content_key: int) -> bool | None:
        """Read a frozen verdict (thread-safe). Returns None if absent."""
        with self._frozen_lock:
            return self._frozen_verdicts.get(content_key)

    def _frozen_verdict_recoverable(self, strategy: object, compressed: str | None) -> bool:
        """Whether a "compress" verdict is safe to freeze under the #1307 rule.

        A lossy-unmarked strategy that emitted no CCR retrieval marker is
        unrecoverable, so pinning it across turns would keep serving a
        fabricated summary the agent can't restore. Refuse to freeze those;
        recoverable (marked or non-lossy) compressions may be pinned.

        ``strategy`` is a ``CompressionStrategy`` on the fresh-compress path but
        its ``.value`` string on the cache-hit path, so compare by value.
        """
        value = getattr(strategy, "value", strategy)
        lossy_values = {getattr(s, "value", s) for s in self.LOSSY_UNMARKED_STRATEGIES}
        if value in lossy_values and not CCR_RETRIEVAL_MARKER_RE.search(compressed or ""):
            return False
        return True

    def _clear_frozen_verdicts(self) -> None:
        """Drop all frozen verdicts (thread-safe). Fired on cache clear."""
        with self._frozen_lock:
            self._frozen_verdicts.clear()

    def _freeze_block_decision_enabled(self) -> bool:
        """Whether the per-block verdict freeze is active (default off).

        Reads ``HEADROOM_FREEZE_BLOCK_DECISION`` each call so it can be
        toggled per-process without restart in tests. Default off → the
        verdict store is never touched and behaviour is byte-identical.
        """
        return os.environ.get("HEADROOM_FREEZE_BLOCK_DECISION", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _kompress_model_ready(self) -> bool:
        """Whether the ML compressor is ready (or deliberately disabled).

        Caveat (1): a block that fails to compress *only* because ModernBERT
        is still lazy-loading must NOT have its "skip" verdict frozen — it has
        to be re-evaluated once the model is ready. When kompress is disabled
        or unavailable there is no deferred model, so "skip" is a real verdict
        and may be frozen.
        """
        if not self.config.enable_kompress:
            return True
        if getattr(self, "_runtime_kompress_model", None) == "disabled":
            return True
        try:
            compressor = self._get_kompress()
        except Exception:
            return True
        if compressor is None:
            # No ML compressor available at all — not a deferred load.
            return True
        try:
            return bool(compressor.is_ready())
        except Exception:
            return True

    def _record_to_toin(
        self,
        strategy: CompressionStrategy,
        content: str,
        compressed: str,
        original_tokens: int,
        compressed_tokens: int,
        language: str | None = None,
        context: str = "",
    ) -> None:
        """Record compression to TOIN for cross-user learning.

        This allows TOIN to track compression patterns for ALL content types,
        not just JSON arrays. When the LLM retrieves original content via CCR,
        TOIN learns which compressions users need to expand.

        Args:
            strategy: The compression strategy used.
            content: Original content (for signature generation).
            compressed: Compressed content.
            original_tokens: Token count before compression.
            compressed_tokens: Token count after compression.
            language: Optional language hint for code.
            context: Query context for pattern learning.
        """
        # Skip SmartCrusher - it handles its own TOIN recording
        if strategy == CompressionStrategy.SMART_CRUSHER:
            return

        # Skip if no actual compression happened
        if original_tokens <= compressed_tokens:
            return

        # F2.2 gate: when the active CompressionPolicy says
        # ``toin_read_only=True`` (Subscription auth mode), don't
        # mutate the TOIN learning pool from this request. Direct
        # ``compress()`` callers don't go through ``apply()`` and
        # have ``self._runtime_compression_policy is None`` — those
        # keep their pre-F2.2 write-enabled behaviour.
        policy = self._runtime_compression_policy
        if policy is not None and policy.toin_read_only:
            logger.debug(
                "ContentRouter: skipping TOIN record_compression for %s "
                "— policy.toin_read_only=True (auth_mode resolved as "
                "Subscription, F2.2 gate)",
                strategy.value,
            )
            return

        try:
            # Lazy load TOIN
            if self._toin is None:
                from ..telemetry.toin import get_toin

                self._toin = get_toin()

            # Create a content-type signature
            signature = _create_content_signature(
                content_type=strategy.value,
                content=content,
                language=language,
            )

            if signature is None:
                return

            # Record the compression
            self._toin.record_compression(
                tool_signature=signature,
                original_count=1,  # Single content block
                compressed_count=1,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                strategy=strategy.value,
                query_context=context if context else None,
            )

            logger.debug(
                "TOIN: Recorded %s compression: %d → %d tokens",
                strategy.value,
                original_tokens,
                compressed_tokens,
            )

        except Exception as e:
            # TOIN recording should never break compression
            logger.debug("TOIN recording failed (non-fatal): %s", e)

    def _timed_compress(
        self,
        content: str,
        context: str,
        bias: float,
        precomputed_detection: DetectionResult | None = None,
    ) -> tuple[RouterCompressionResult, float]:
        """Compress with wall-clock timing.  Used by parallel executor."""
        t0 = time.perf_counter()
        result = self.compress(
            content, context=context, bias=bias, precomputed_detection=precomputed_detection
        )
        return result, (time.perf_counter() - t0) * 1000

    def compress(
        self,
        content: str,
        context: str = "",
        question: str | None = None,
        bias: float = 1.0,
        precomputed_detection: DetectionResult | None = None,
    ) -> RouterCompressionResult:
        """Compress content using optimal strategy based on content detection.

        Args:
            content: Content to compress.
            context: Optional context for relevance-aware compression.
            question: Optional question for QA-aware compression. When provided,
                tokens relevant to answering this question are preserved.
            bias: Compression bias multiplier (>1 = keep more, <1 = keep fewer).
            precomputed_detection: A ``_detect_content(content)`` result the
                caller already computed for the same content. When supplied, it
                is reused instead of re-running the native detection chain — the
                router's hottest per-message cost. ``apply`` computes detection
                once per message (for its code-protection checks) and passes it
                here so a cache-miss message is not detected twice.

        Returns:
            RouterCompressionResult with compressed content and routing metadata.
        """
        context = context or ""
        debug_enabled = logger.isEnabledFor(logging.DEBUG)
        request_debug = (
            {
                "chars": len(content),
                "bytes": len(content.encode("utf-8", errors="replace")),
                "tokens_estimate": _estimate_tokens(content),
                "json_shape": _json_shape(content),
                "mixed_indicators": _mixed_indicators(content),
                "context_chars": len(context),
                "question": question,
                "bias": bias,
                "content": content,
                "context": context,
            }
            if debug_enabled
            else {}
        )
        if not content or not content.strip():
            if debug_enabled:
                _log_router_debug(
                    "content_router_input",
                    **request_debug,
                    selected_strategy=CompressionStrategy.PASSTHROUGH.value,
                    selection_reason="empty_or_whitespace",
                )
            result = RouterCompressionResult(
                compressed=content,
                original=content,
                strategy_used=CompressionStrategy.PASSTHROUGH,
                routing_log=[],
            )
        else:
            # Determine strategy from content analysis. When runtime settings
            # force Kompress, skip the full router detection path so large
            # proxy payloads do not pay for an unused strategy decision.
            force_kompress = bool(getattr(self, "_runtime_force_kompress", False))
            if force_kompress:
                mixed = False
                detection = DetectionResult(ContentType.PLAIN_TEXT, 1.0, {})
                strategy = CompressionStrategy.KOMPRESS
            else:
                mixed = is_mixed_content(content)
                detection = (
                    precomputed_detection
                    if precomputed_detection is not None
                    else _detect_content(content)
                )
                strategy = self._determine_strategy(content, mixed=mixed, detection=detection)
            if debug_enabled:
                _log_router_debug(
                    "content_router_input",
                    **request_debug,
                    detected_content_type=detection.content_type.value,
                    detection_confidence=detection.confidence,
                    selected_strategy=strategy.value,
                    selection_reason=(
                        "runtime_force_kompress"
                        if force_kompress
                        else "mixed_content"
                        if mixed
                        else "content_detection"
                    ),
                )

            if strategy == CompressionStrategy.MIXED:
                result = self._compress_mixed(content, context, question, bias=bias)
            else:
                result = self._compress_pure(content, strategy, context, question, bias=bias)

        # Empty-output guard: compression must NEVER blank out non-empty input.
        # An empty user-message content makes Anthropic reject the whole request
        # with 400 ("messages.N: user messages must have non-empty content").
        # If any transform yields empty/whitespace from non-empty input, fall
        # back to the original content (passthrough) instead of emitting empty.
        if (
            content
            and content.strip()
            and (result.compressed is None or not str(result.compressed).strip())
        ):
            logger.warning(
                "content_router: compression produced EMPTY output from non-empty "
                "input (%d chars, strategy=%s); falling back to original to avoid 400.",
                len(content),
                getattr(result.strategy_used, "value", result.strategy_used),
            )
            result.compressed = content

        # One observer call per routing decision; the observer is the
        # forcing function for catching strategy-level regressions.
        # Empty routing_log (passthrough fast path) → no calls.
        self._observe(result)
        if debug_enabled:
            _log_router_debug(
                "content_router_output",
                selected_strategy=result.strategy_used.value,
                sections_processed=result.sections_processed,
                total_original_tokens=result.total_original_tokens,
                total_compressed_tokens=result.total_compressed_tokens,
                tokens_saved=result.tokens_saved,
                savings_percentage=result.savings_percentage,
                compression_ratio=result.compression_ratio,
                routing_log=[
                    {
                        "content_type": decision.content_type.value,
                        "strategy": decision.strategy.value,
                        "original_tokens": decision.original_tokens,
                        "compressed_tokens": decision.compressed_tokens,
                        "confidence": decision.confidence,
                        "section_index": decision.section_index,
                        "compression_ratio": decision.compression_ratio,
                    }
                    for decision in result.routing_log
                ],
                original=result.original,
                compressed=result.compressed,
            )
        return result

    def _observe(self, result: RouterCompressionResult) -> None:
        """Forward each `RoutingDecision` in `result.routing_log` to the
        configured `CompressionObserver`. No-op when no observer is set.

        Observers MUST NOT raise per the protocol contract; if one does
        anyway, swallow at debug level. Compression already succeeded;
        a buggy observer must not turn a 200 into a 500.
        """
        if self._observer is None:
            return
        for d in result.routing_log:
            try:
                self._observer.record_compression(
                    strategy=d.strategy.value,
                    original_tokens=d.original_tokens,
                    compressed_tokens=d.compressed_tokens,
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("CompressionObserver raised (non-fatal): %s", e)

    def _observe_kompress_size_gate(self, outcome: str) -> None:
        """Forward one kompress size-gate decision to the observer.

        ``outcome`` is "exceeded" when an eligible block trips the size
        ceiling and is routed off ML, "within" when it passes the gate.
        Goes through the same observer hook as ``_observe`` so the metric
        reaches the PrometheusMetrics singleton without ``content_router``
        importing ``headroom.proxy`` (no cycle). Defensive: a missing
        method or a buggy observer must not break the compression.
        """
        if self._observer is None:
            return
        try:
            self._observer.record_kompress_size_gate(outcome)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("CompressionObserver raised (non-fatal): %s", e)

    def _determine_strategy(
        self,
        content: str,
        mixed: bool | None = None,
        detection: DetectionResult | None = None,
    ) -> CompressionStrategy:
        """Determine the compression strategy from content analysis.

        Args:
            content: Content to analyze.
            mixed: Precomputed ``is_mixed_content(content)`` when the caller
                already has it. ``compress`` runs it on this exact content one
                line before calling here; recomputed only when ``None``.
            detection: Precomputed ``_detect_content(content)`` when the caller
                already has it. The native Rust/Magika pass is the router's
                hottest per-message cost — reuse it instead of a second
                identical detection; recomputed only when ``None``.

        Returns:
            Selected compression strategy.
        """
        # Reuse the caller's analysis when supplied — ``compress`` already ran
        # both on this exact content, and re-running is_mixed_content/
        # _detect_content on identical bytes is the router's hottest wasted cost.
        if mixed is None:
            mixed = is_mixed_content(content)
        if detection is None:
            detection = _detect_content(content)

        # 1. Check for mixed content
        if mixed:
            # 2. Verify with the native detector: ``is_mixed_content`` uses
            # cheap regex heuristics that produce false positives on source
            # code.  Python files with dict/list literals (``{``, ``[`` at
            # line start) trigger ``has_json_blocks``, and docstrings/comments
            # trigger ``has_prose`` — so a pure Python blob is misclassified
            # as MIXED, routed through ``_compress_mixed`` which splits it
            # into sections and dispatches each to KOMPRESS (no-op on code),
            # wasting latency on splitting without any compression.
            # When the native magika detector confidently says SOURCE_CODE,
            # trust it over the regex heuristics.
            if detection.content_type == ContentType.SOURCE_CODE and detection.confidence >= 0.8:
                return self._strategy_from_detection(detection)
            return CompressionStrategy.MIXED

        # 2. Not mixed — map the detected type straight to a strategy.
        return self._strategy_from_detection(detection)

    def _strategy_from_detection(self, detection: Any) -> CompressionStrategy:
        """Get strategy from content detection result.

        Args:
            detection: Result from detect_content_type.

        Returns:
            Selected strategy.
        """
        mapping = {
            ContentType.SOURCE_CODE: CompressionStrategy.CODE_AWARE,
            ContentType.JSON_ARRAY: CompressionStrategy.SMART_CRUSHER,
            ContentType.SEARCH_RESULTS: CompressionStrategy.SEARCH,
            ContentType.BUILD_OUTPUT: CompressionStrategy.LOG,
            ContentType.GIT_DIFF: CompressionStrategy.DIFF,
            ContentType.HTML: CompressionStrategy.HTML,
            ContentType.TABULAR: CompressionStrategy.TABULAR,
            ContentType.STRUCTURED_CONFIG: CompressionStrategy.CONFIG,
            ContentType.PLAIN_TEXT: CompressionStrategy.TEXT,
        }

        strategy = mapping.get(detection.content_type, self.config.fallback_strategy)

        # Override: prefer CodeAware for code if configured
        if (
            strategy == CompressionStrategy.CODE_AWARE
            and not self.config.prefer_code_aware_for_code
        ):
            # When CodeAware is not preferred, the intent is "let code pass
            # through unmangled" (per the config comment).  Previously this
            # fell back to KOMPRESS, which is an ML compressor that can
            # destroy code semantics — on a 45K-token Python blob it
            # compresses to 912 tokens with 11% fact recall, making the
            # code useless for an agent.  The MIXED path accidentally
            # protected code by splitting it into small sections that
            # KOMPRESS passes through, but that was a side-effect, not
            # intent.  Now that the MIXED false-positive on source code is
            # fixed (``_determine_strategy`` trusts the native detector),
            # code reaches this path directly — so use PASSTHROUGH to
            # honour the "unmangled" intent explicitly.
            strategy = CompressionStrategy.PASSTHROUGH

        return strategy

    def _compress_mixed(
        self,
        content: str,
        context: str,
        question: str | None = None,
        bias: float = 1.0,
    ) -> RouterCompressionResult:
        """Compress mixed content by splitting and routing sections.

        Args:
            content: Mixed content to compress.
            context: User context for relevance.
            question: Optional question for QA-aware compression.
            bias: Compression bias multiplier.

        Returns:
            RouterCompressionResult with reassembled content.
        """
        from .tag_protector import protect_tags, restore_tags

        # Protect custom-tag blocks BEFORE splitting into sections. Section
        # boundaries (code fences, blank lines) split a
        # ``<system-reminder>...</system-reminder>`` pair across sections, so
        # the per-section tag protection inside ``_try_ml_compressor`` never
        # sees a matched pair (an unmatched tag protects nothing) and
        # instruction blocks — Claude Code ships CLAUDE.md inside
        # <system-reminder> — leak into lossy ML compression and arrive
        # word-dropped. Protecting here keeps the whole block as one
        # placeholder that spans sections intact.
        cleaned, protected = protect_tags(
            content,
            compress_tagged_content=self.config.compress_tagged_content,
        )
        sections_source = cleaned if protected else content

        # Placeholder lines must each be their own section (see the
        # placeholder passthrough below): a placeholder sharing a section
        # with prose would drag that prose into verbatim passthrough.
        sections = split_into_sections(
            sections_source,
            isolate=tuple(placeholder for placeholder, _ in protected),
        )
        if logger.isEnabledFor(logging.DEBUG):
            _log_router_debug(
                "content_router_mixed_sections",
                section_count=len(sections),
                sections=[_section_debug(section, idx) for idx, section in enumerate(sections)],
                content=content,
            )

        if not sections:
            return RouterCompressionResult(
                compressed=content,
                original=content,
                strategy_used=CompressionStrategy.PASSTHROUGH,
            )

        # Placeholders must survive byte-exact: ``restore_tags`` DISCARDS a
        # protected block whose placeholder was stripped or rewritten
        # (Hotfix-A9), so a compressor eating a placeholder would silently
        # drop the whole tag block — worse than the mangling this fixes.
        # Any section carrying a placeholder is passed through verbatim
        # instead of ever entering a compressor.
        placeholders = [placeholder for placeholder, _ in protected]

        compressed_sections: list[str] = []
        routing_log: list[RoutingDecision] = []

        for i, section in enumerate(sections):
            if placeholders and any(ph in section.content for ph in placeholders):
                section_tokens = _estimate_tokens(section.content)
                compressed_sections.append(section.content)
                routing_log.append(
                    RoutingDecision(
                        content_type=section.content_type,
                        strategy=CompressionStrategy.PASSTHROUGH,
                        original_tokens=section_tokens,
                        compressed_tokens=section_tokens,
                        section_index=i,
                    )
                )
                continue

            # Get strategy for this section
            strategy = self._strategy_from_detection_type(section.content_type)

            # Compress section
            original_tokens = _estimate_tokens(section.content)
            compressed_content, compressed_tokens, _section_chain = self._apply_strategy_to_content(
                section.content,
                strategy,
                context,
                section.language,
                question,
                bias=bias,
            )

            # Preserve code fence markers
            if section.is_code_fence and section.language:
                compressed_content = f"```{section.language}\n{compressed_content}\n```"

            # A JSON_ARRAY section whose compressed form is a bare JSON
            # *string* (SmartCrusher's lossless CSV+schema render replaces
            # the whole array with one string value) must be spliced back
            # as the raw text it encodes. Left as the JSON literal, the
            # section lands mid-prose as one quote-wrapped line with `\n`
            # as two-character escapes — the classic "compression garbled
            # the output" report. Valid inside a JSON document; unreadable
            # inside mixed text.
            if section.content_type is ContentType.JSON_ARRAY and compressed_content.startswith(
                '"'
            ):
                try:
                    _unwrapped = json.loads(compressed_content)
                except (TypeError, ValueError):
                    _unwrapped = None
                if isinstance(_unwrapped, str):
                    compressed_content = _unwrapped

            compressed_sections.append(compressed_content)
            routing_log.append(
                RoutingDecision(
                    content_type=section.content_type,
                    strategy=strategy,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    section_index=i,
                )
            )

        compressed = "\n\n".join(compressed_sections)
        if protected:
            compressed = restore_tags(compressed, protected)

        return RouterCompressionResult(
            compressed=compressed,
            original=content,
            strategy_used=CompressionStrategy.MIXED,
            routing_log=routing_log,
            sections_processed=len(sections),
        )

    def _compress_pure(
        self,
        content: str,
        strategy: CompressionStrategy,
        context: str,
        question: str | None = None,
        bias: float = 1.0,
    ) -> RouterCompressionResult:
        """Compress pure (non-mixed) content.

        Args:
            content: Content to compress.
            strategy: Selected strategy.
            context: User context.
            question: Optional question for QA-aware compression.
            bias: Compression bias multiplier.

        Returns:
            RouterCompressionResult.
        """
        original_tokens = _estimate_tokens(content)

        compressed, compressed_tokens, strategy_chain = self._apply_strategy_to_content(
            content, strategy, context, question=question, bias=bias
        )

        return RouterCompressionResult(
            compressed=compressed,
            original=content,
            strategy_used=strategy,
            strategy_chain=strategy_chain,
            routing_log=[
                RoutingDecision(
                    content_type=self._content_type_from_strategy(strategy),
                    strategy=strategy,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                )
            ],
        )

    def _lossless_first(
        self, content: str, strategy: CompressionStrategy
    ) -> tuple[str, str | None]:
        """Byte/data-lossless first pass (intended design: always runs, pre-lossy).

        Maps the (content-detected) strategy to its format-native lossless fold —
        SEARCH -> ripgrep --heading form, LOG -> run-collapse + ANSI strip, DIFF
        -> drop ``index`` bookkeeping — and gives every other content type a
        trivial blank-run collapse. ``compact_lossless`` is self-verifying (exact
        inverse or unchanged) and returns the input when it cannot safely shrink,
        so this never loses information and is a strict no-op when nothing folds.

        Returns ``(folded, "lossless_<kind>")`` when a real byte shrink happened,
        else ``(content, None)``.
        """
        from headroom.transforms.lossless_compaction import compact_lossless

        # Memo: `_has_lossless_fold` probes this method purely to decide whether a
        # small block is worth admitting and throws the result away, and the real
        # STAGE 0 pass then recomputes it. That doubles the built-in folds *and*
        # the third-party provider call for every block that takes both paths.
        # The method is a pure function of (content, strategy), so cache it.
        # Resolved with `getattr` so the method still works on a bare instance
        # (`object.__new__(ContentRouter)`, used by unit tests that exercise the
        # folds without paying for a full router init).
        memo = getattr(self, "_lossless_first_memo", None)
        if memo is None:
            memo = self._lossless_first_memo = {}
        # The provider generation is part of the key: this memo caches a value
        # that DEPENDS on the registered provider, so a provider registered (or
        # cleared) after a block was first folded must not be served the old
        # answer forever.
        memo_key = (hash(content), len(content), strategy, get_lossless_generation())
        memoized = memo.get(memo_key)
        if memoized is not None:
            return memoized

        # Apply losslessness to the OUTPUT structure, not to the classification:
        # try the fold implied by the detected strategy first, then the others.
        # Each compact_lossless call is self-verifying (exact inverse or returns
        # the input unchanged), so attempting a fold on non-matching content is a
        # safe no-op — this recovers folds on content the detector misroutes
        # (e.g. `grep -n` of .py files classified as SOURCE_CODE still gets the
        # search fold). Keep the single fold that shrinks the most.
        primary = {
            CompressionStrategy.SEARCH: "search",
            CompressionStrategy.LOG: "log",
            CompressionStrategy.DIFF: "diff",
            CompressionStrategy.CONFIG: "config",
        }.get(strategy)
        order = ([primary] if primary else []) + [
            k for k in ("search", "paths", "log", "diff", "text") if k != primary
        ]
        # The "diff" fold (diff_strip_index) is the one compact_lossless kind that
        # is purely subtractive with NO exact-inverse check: it removes any line
        # shaped like `index <hex>..<hex>`. On non-diff content that happens to
        # contain such a line, that line is silently and unrecoverably dropped —
        # breaking the lossless contract this method's docstring promises, and
        # unmarked in CCR mode. Only fold diffs as diffs.
        if (
            "diff" in order
            and strategy is not CompressionStrategy.DIFF
            and not self._looks_like_diff(content)
        ):
            order = [k for k in order if k != "diff"]
        best, best_label = content, None
        for kind in order:
            try:
                cand = compact_lossless(content, kind)
            except Exception:
                continue
            if len(cand) < len(best):
                best, best_label = cand, f"lossless_{kind}"

        # A registered lossless provider (headroom.transforms.lossless_provider)
        # competes on the GENERAL path too, not only on excluded-tool output via
        # `_lossless_compact_excluded`. Without this an external fold only ever
        # saw Read/Grep/Glob-style results, so it was inert for gateway traffic
        # (`/v1/compress`), where tool names are the caller's own. Same contract
        # (information-preserving + deterministic) and we keep whichever output
        # is smaller, so a provider can only improve on the built-in folds.
        #
        # Diffs are off limits for a provider, for the same reason the built-in
        # "diff" fold is restricted above: diff folding is subtractive with no
        # inverse check, and a Kompressed/reflowed hunk breaks `git apply`. The
        # built-in path can at least reason about its own fold; a third-party one
        # we cannot, so we simply never offer it diff content.
        if strategy is not CompressionStrategy.DIFF and not self._looks_like_diff(content):
            best, best_label = self._apply_lossless_provider(content, best, best_label)

        # Plain dict operations only: this is a pure-function cache, so a racing
        # double-compute under the parallel compression pool (see
        # HEADROOM_COMPRESS_WORKERS) costs one redundant fold and nothing else.
        # A lock here would sit on the hot path of every block for no
        # correctness gain. Bounded by a wholesale clear rather than LRU
        # bookkeeping — the memo exists to collapse a within-request double
        # call, not to be a cross-request cache (`self._cache` is that).
        if len(memo) >= self._LOSSLESS_FIRST_MEMO_MAX:
            memo.clear()
        memo[memo_key] = (best, best_label)
        return best, best_label

    def _apply_lossless_provider(
        self, content: str, best: str, best_label: str | None
    ) -> tuple[str, str | None]:
        """Let a registered lossless provider beat ``best``; never raise.

        The provider only wins on a strictly smaller, non-blank candidate — and
        in lossless-only mode only if an optionally registered verifier confirms
        the fold is genuinely reversible.
        """
        supplied = self._lossless_provider_result(content)
        if supplied is None:
            return best, best_label
        cand, kind = supplied

        if len(cand) >= len(best):
            return best, best_label

        # Lossless-only mode has no downstream recovery: STAGE 0's output IS the
        # answer and there is no CCR marker to retrieve the original from. The
        # built-in folds are self-verifying; a provider is trusted unless the
        # operator registered a verifier, in which case it must pass.
        if self.config.lossless:
            verifier = get_lossless_verifier()
            if verifier is not None:
                try:
                    ok = verifier(content, cand)
                except Exception:  # noqa: BLE001 - a raising verifier means "unverified"
                    logger.debug("lossless verifier raised; rejecting candidate", exc_info=True)
                    return best, best_label
                if not ok:
                    logger.debug("lossless verifier rejected the provider candidate")
                    return best, best_label

        return cand, f"lossless_{kind}"

    def _lossless_provider_result(self, content: str) -> tuple[str, str] | None:
        """Validated + memoized ``(candidate, safe_kind)`` from the provider.

        Everything a third-party provider hands back is treated as hostile: a
        malformed shape, an empty result, a metric-unsafe ``kind`` label or an
        outright exception all degrade to ``None`` ("provider ignored"). This
        runs on the request path and its caller (``TransformPipeline.apply``)
        re-raises, so nothing here may escape.

        Memoized per ``(hash(content), len(content))`` — the provider contract is
        ``content -> result`` with no other input, so its answer cannot depend on
        the routing strategy. That matters because one block reaches
        ``_lossless_first`` twice under *different* strategies (the
        ``_has_lossless_fold`` admission probe passes PASSTHROUGH, STAGE 0 passes
        the detected strategy), which would otherwise invoke third-party code
        twice per block. Same no-lock rationale as the ``_lossless_first`` memo.
        """
        provider = get_lossless_provider()
        if provider is None:
            return None

        memo = getattr(self, "_lossless_provider_memo", None)
        if memo is None:
            memo = self._lossless_provider_memo = {}
        memo_key = (hash(content), len(content), get_lossless_generation())
        if memo_key in memo:
            return memo[memo_key]

        result: tuple[str, str] | None = None
        try:
            supplied = provider(content)
            if supplied is None:
                result = None
            # Validate defensively *inside* the try: unpacking a 3-tuple / a bare
            # string / a non-sequence raises, and that exception used to escape
            # all the way out of the request.
            elif not isinstance(supplied, tuple | list) or len(supplied) != 2:
                logger.debug("lossless provider returned a malformed result; ignoring")
            elif not isinstance(supplied[0], str) or not isinstance(supplied[1], str):
                logger.debug("lossless provider returned non-str members; ignoring")
            # An empty / whitespace-only "fold" wins every length comparison and
            # would silently delete the block's content. That is not lossless.
            elif not supplied[0].strip():
                logger.debug("lossless provider returned an empty result; ignoring")
            else:
                raw_kind = supplied[1]
                # fullmatch, not match: Python's `$` also matches just BEFORE a
                # trailing newline, so `re.match` would let "log\n" through and
                # put a newline into a metric label.
                kind = (
                    raw_kind if _PROVIDER_KIND_RE.fullmatch(raw_kind) else _PROVIDER_KIND_FALLBACK
                )
                result = (supplied[0], kind)
        except Exception:  # noqa: BLE001 - a broken provider must not break routing
            logger.debug("lossless provider failed in _lossless_first", exc_info=True)
            result = None

        if len(memo) >= self._LOSSLESS_FIRST_MEMO_MAX:
            memo.clear()
        memo[memo_key] = result
        return result

    @staticmethod
    def _looks_like_diff(content: str) -> bool:
        """Cheap structural sniff for unified/git-diff content.

        Used to keep the lossy-after-fold pass (Kompress) OFF diff content —
        Kompressing hunks corrupts ``git apply``. This is defense-in-depth beyond the
        DIFF-strategy and ``lossless_diff``-label checks: a diff can be folded
        best under a non-diff label (e.g. blank-line collapse → ``lossless_text``)
        or mis-detected, and must still never reach the lossy stage. It is also
        what keeps a third-party lossless provider away from diffs entirely.
        """
        return (
            "diff --git " in content
            or "\n@@ " in content
            or content.startswith("@@ ")
            or content.startswith("--- ")
        )

    def _has_lossless_fold(self, content: str) -> bool:
        """True if a byte/data-lossless fold shrinks ``content`` (any format).

        Lets small blocks bypass the lossy ``min_chars`` floor: a lossless fold
        is byte-exact and cheap (stdlib regex), so there is no size threshold
        below which it should be skipped. The floor exists only to keep the
        expensive lossy compressors off marginal blocks — it must not gate the
        free, recoverable fold.
        """
        if not isinstance(content, str):
            return False
        return self._lossless_first(content, CompressionStrategy.PASSTHROUGH)[1] is not None

    # ── External compressor dispatch (opt-in; fail-open) ──────────────────────

    def _resolve_active_external_compressors(self) -> list[Any]:
        """Resolve the opt-in external compressor selection against the registry.

        Returns the active EXTERNAL compressor objects (built-in inventory
        entries filtered out — they own the if/elif dispatch, never the
        registry). Empty when nothing external is selected or resolution fails,
        so the caller's external-dispatch branch is inert by default.
        """
        selection = self.config.active_external_compressors
        if not selection:
            return []
        try:
            active = self.compressor_registry.active(set(selection))
        except Exception as exc:  # noqa: BLE001 - selection is non-critical
            logger.debug("external compressor resolution failed: %s", exc)
            return []
        return [c for c in active if not isinstance(c, _BuiltinCompressorEntry)]

    def _try_external_compressor(
        self,
        content: str,
        strategy: CompressionStrategy,
        context: str,
        question: str | None,
    ) -> tuple[str, int, list[str]] | None:
        """Route a block through a *selected* external compressor, or ``None``.

        Opt-in and fail-open. Returns ``None`` — leaving the built-in if/elif
        dispatch to run UNCHANGED — whenever:

          * no external compressor was selected (the default: a single cheap
            guard, so the request path is byte-identical to today);
          * none of the active external compressors declares this block's
            detected content type;
          * the chosen compressor raises, returns malformed/empty output, or
            would expand the content.

        On success it returns the router's normal ``(content, tokens, chain)``
        shape: the external output, tokens counted with the router's OWN
        estimator (never the compressor's self-reported count), and an
        ``["external:<name>"]`` chain. Any ``recoverable`` (hash -> original)
        map is persisted to the CCR store exactly like SmartCrusher's mirror,
        so ``/v1/retrieve/{hash}`` resolves.

        Reached only in lossy/CCR mode: ``_apply_strategy_to_content`` returns
        earlier in lossless-only mode (STAGE 0), so an external compressor can
        never inject unrecoverable loss into a lossless-only session.
        """
        active = self._active_external_compressors
        if not active:
            return None
        content_mime = _CONTENT_TYPE_TO_MIME.get(self._content_type_from_strategy(strategy))
        if content_mime is None:
            return None
        for compressor in active:
            try:
                descriptor = compressor.descriptor
            except Exception as exc:  # noqa: BLE001 - a broken external is isolated
                logger.debug("external compressor descriptor unavailable: %s", exc)
                continue
            if not _external_compressor_matches(descriptor, content_mime):
                continue
            result = self._run_external_compressor(
                compressor, descriptor.name, content, content_mime, context, question
            )
            if result is not None:
                return result
        return None

    def _run_external_compressor(
        self,
        compressor: Any,
        name: str,
        content: str,
        content_mime: str,
        context: str,
        question: str | None,
    ) -> tuple[str, int, list[str]] | None:
        """Invoke one external compressor via the contract; fail open to ``None``."""
        inp = CompressInput(
            content=content,
            content_type=content_mime,
            query=question or context or "",
            config={},
            budget={},
        )
        try:
            out = compressor.compress(inp)
        except Exception as exc:  # noqa: BLE001 - fail open to the built-in path
            logger.warning(
                "external compressor %r raised (%s); falling back to built-in", name, exc
            )
            return None
        if not isinstance(out, CompressOutput) or not isinstance(out.content, str):
            logger.warning(
                "external compressor %r returned malformed output (%s); falling back",
                name,
                type(out).__name__,
            )
            return None
        compressed = out.content
        # Never blank out a non-empty block (an empty user/tool block makes
        # providers reject the request); fall back so the built-in path runs.
        if content.strip() and not compressed.strip():
            logger.warning(
                "external compressor %r produced empty output; falling back to built-in", name
            )
            return None
        # Never let an external compressor expand a block; fall back so the
        # built-in path (or passthrough) can do better.
        if len(compressed) > len(content):
            logger.debug(
                "external compressor %r expanded content (%d -> %d chars); falling back",
                name,
                len(content),
                len(compressed),
            )
            return None
        # Count with the router's OWN estimator, not the compressor's self-report.
        compressed_tokens = _estimate_tokens(compressed)
        # Persist the hash -> original recovery map the SAME way SmartCrusher
        # mirrors its markers, so a later /v1/retrieve resolves each hash.
        self._persist_external_recoverable(out.recoverable, name, context)
        if out.warnings:
            logger.debug(
                "external compressor %r warnings: %s", name, "; ".join(map(str, out.warnings))
            )
        if out.markers and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "external compressor %r markers: %s", name, "; ".join(map(str, out.markers))
            )
        return compressed, compressed_tokens, [f"external:{name}"]

    def _persist_external_recoverable(
        self, recoverable: dict[str, str], name: str, context: str
    ) -> None:
        """Mirror an external compressor's hash -> original map into the CCR store.

        Mirrors SmartCrusher's ``_mirror_single_hash_to_python_store``: each
        entry is stored under its own hash via ``explicit_hash`` so a
        ``/v1/retrieve/{hash}`` lookup returns the original. Best-effort — a
        store failure or a non-hex hash is logged and never breaks the request
        (the compressed block is still returned; only that entry is unretrievable).
        """
        if not recoverable:
            return
        try:
            from ..cache.compression_store import get_compression_store

            store = get_compression_store()
        except Exception as exc:  # noqa: BLE001 - CCR store optional/stripped builds
            logger.debug("external compressor %r: CCR store unavailable (%s)", name, exc)
            return
        strategy_label = f"external:{name}"
        for ccr_hash, original in recoverable.items():
            if not isinstance(ccr_hash, str) or not isinstance(original, str):
                logger.debug("external compressor %r: skipping non-str recoverable entry", name)
                continue
            try:
                store.store(
                    original=original,
                    # The compressed payload isn't meaningfully addressable per
                    # hash here; use a placeholder marker (as SmartCrusher does
                    # — /v1/retrieve returns original_content, not compressed).
                    compressed=f"<<external:{name}:{ccr_hash}>>",
                    query_context=context or None,
                    compression_strategy=strategy_label,
                    explicit_hash=ccr_hash,
                )
            except ValueError:
                # explicit_hash must be hex; a malformed hash means this entry
                # won't be retrievable, but the request must not break.
                logger.warning(
                    "external compressor %r: recoverable hash %r is not hex; not stored",
                    name,
                    ccr_hash,
                )
            except Exception as exc:  # noqa: BLE001 - defensive; never break the request
                logger.debug("external compressor %r: store.store raised (%s)", name, exc)

    def _registry_compress(
        self,
        name: str,
        strategy: CompressionStrategy,
        content: str,
        context: str,
        bias: float,
        config: dict[str, Any] | None = None,
        question: str | None = None,
    ) -> CompressOutput | None:
        """Compress ``content`` with a built-in via the registry, full output.

        Resolves the built-in named ``name`` from :attr:`compressor_registry` and
        runs it over the pure-data :class:`CompressInput` contract, returning the
        adapter's :class:`CompressOutput` — including its ``compressed`` flag,
        which reports whether the built-in actually compressed (``True``) or
        passed the content through unchanged (``False``, e.g. the built-in
        returned ``None`` / HTML extraction found nothing). Returns ``None`` only
        when the built-in is not registered (defensive; the inventory is always
        registered by ``_build_compressor_registry``).

        This is the registry-resolved equivalent of the router's historical
        ``self._get_<name>().compress(...)`` dispatch: the built-in adapter
        delegates to the SAME ``_get_*`` getter and method with the SAME
        arguments (``context`` as the query, ``bias`` via the budget, and any
        per-strategy ``config`` such as ``language`` for code_aware), so on a
        real compression the returned content is byte-identical to the direct
        call. Callers read ``.compressed`` to reproduce the historical
        ``compressed is None`` fallback/passthrough branches exactly.

        ``question`` (QA-aware compression) is carried on the pure-data contract
        via ``CompressInput.config['question']`` — a free ``dict`` — so the
        contract shape is unchanged. The ``kompress`` adapter reads it back with
        ``inp.config.get('question')`` and forwards it into
        ``_try_ml_compressor(content, context, question)``, matching the router's
        historical direct call. When ``None`` it is not injected, leaving other
        built-ins' config untouched.
        """
        merged_config = dict(config or {})
        if question is not None:
            merged_config["question"] = question
        entry = self.compressor_registry.get(name)
        if entry is None:
            return None
        return entry.compress(
            CompressInput(
                content=content,
                content_type=_CONTENT_TYPE_TO_MIME.get(
                    self._content_type_from_strategy(strategy), "text/plain"
                ),
                query=context,
                config=merged_config,
                budget={"bias": bias},
            )
        )

    def _registry_compress_content(
        self,
        name: str,
        strategy: CompressionStrategy,
        content: str,
        context: str,
        bias: float,
    ) -> str:
        """Compress ``content`` with a built-in via the compressor registry.

        Thin wrapper over :meth:`_registry_compress` returning just the
        compressed string. This is the registry-resolved equivalent of the
        router's historical ``self._get_<name>().compress(...)`` dispatch: the
        built-in adapter delegates to the SAME ``_get_*`` getter and method with
        the SAME arguments (``context`` as the query, ``bias`` via the budget), so
        the returned content is byte-identical to the direct call.

        Callers keep their own ``if self.config.enable_<x>:`` gate and ``_get_*``
        availability guard (which preserves the built-in-unavailable → passthrough
        behavior the adapter's None→content collapse would otherwise hide) and
        recompute the token count with the branch's own metric, so the branch's
        return shape is unchanged. When the built-in is not registered
        (defensive), falls back to the unchanged content.
        """
        output = self._registry_compress(name, strategy, content, context, bias)
        if output is None:
            # Built-in inventory is always registered by _build_compressor_registry;
            # defensive only — fall back to the unchanged content.
            return content
        return output.content

    def _apply_strategy_to_content(
        self,
        content: str,
        strategy: CompressionStrategy,
        context: str,
        language: str | None = None,
        question: str | None = None,
        bias: float = 1.0,
        _allow_embedded: bool = True,
    ) -> tuple[str, int, list[str]]:
        """Apply a compression strategy to content.

        Args:
            content: Content to compress.
            strategy: Strategy to use.
            context: User context.
            language: Language hint for code.
            question: Optional question for QA-aware compression.
            bias: Compression bias multiplier (>1 = keep more, <1 = keep fewer).

        Returns:
            Tuple of (compressed_content, compressed_token_count,
            strategy_chain). The chain lists every strategy attempted
            in order — first the requested one, then any fallbacks.
            Single-entry chain means a direct hit; multi-entry means
            the fallback chain fired (e.g. ``[smart_crusher, kompress,
            log]``). Log readers use this to see *how* we got to the
            final compressor without parsing decision_reason strings.
        """
        # ── STRUCTURAL (embedded) JSON routing ───────────────────────────────
        # Before anything else: if this block is not a single JSON value but
        # CONTAINS balanced JSON span(s), route each span through this very
        # dispatch and splice the result back (surrounding bytes kept exact).
        # This is how nested/embedded JSON reaches the JSON compressors at all —
        # today's linear splitter never sees it. Each span goes through the
        # UNCHANGED path, so SmartCrusher/CodeCompressor register their
        # `<<ccr:…>>` markers exactly as for a whole-block JSON (CCR is hash-
        # keyed → location-agnostic). `_allow_embedded=False` on the recursive
        # call is a one-shot re-entrancy guard (NOT a depth cap). Deterministic +
        # benefit-gated (no size/min thresholds) → prefix-cache- and CCR-store-
        # stable, and a strict no-op when the block has no embedded JSON.
        if _allow_embedded:
            from headroom.transforms.recursive_json import route_embedded_json

            def _dispatch_span(span: str) -> str | None:
                strat = self._strategy_from_detection_type(_detect_content(span).content_type)
                text, _t, _c = self._apply_strategy_to_content(
                    span,
                    strat,
                    context,
                    question=question,
                    bias=bias,
                    _allow_embedded=False,
                )
                return text if text != span else None

            routed = route_embedded_json(content, _dispatch_span, tok=_estimate_tokens)
            if routed is not None:
                return routed, _estimate_tokens(routed), ["embedded_json"]

        # Track original tokens for TOIN recording
        original_tokens = _estimate_tokens(content)
        compressed: str | None = None
        compressed_tokens: int | None = None
        requested_strategy = strategy
        actual_strategy = strategy
        compressor_name = strategy.value
        decision_reason = "strategy_not_enabled_or_unavailable"
        strategy_chain: list[str] = [strategy.value]
        error: str | None = None

        # ── STAGE 0: LOSSLESS-FIRST (unconditional floor) ────────────────────
        # A byte/data-lossless fold has ZERO accuracy cost, so it ALWAYS runs
        # first, in every mode — it banks a guaranteed, fully-recoverable win up
        # front (search --heading, log run-collapse, diff index-strip; blank-run
        # collapse otherwise). Detection is content-based (strategy is assigned by
        # content_detector on the OUTPUT), so `cd DIR && rg …`, pipes and unknown
        # tools route here by structure, not by command. `_lossless_first` is
        # self-verifying (exact inverse or unchanged) → never loses information,
        # and is a strict no-op returning (content, None) when nothing folds.
        _ll_content, _ll_label = self._lossless_first(content, strategy)

        # ── LOSSLESS-ONLY mode: stop at the byte-exact fold ──────────────────
        # HEADROOM_LOSSLESS=1 is an explicit no-unrecoverable-loss contract (the
        # constructor forces markers off + SmartCrusher lossless-only). So we
        # NEVER layer a lossy drop on top here — the fold IS the answer. When it
        # folds, return it; otherwise leave the block verbatim (passthrough),
        # never a marker-free lossy drop that could not be recovered.
        if self.config.lossless:
            if _ll_label is not None:
                return _ll_content, _estimate_tokens(_ll_content), [_ll_label]
            return content, original_tokens, [CompressionStrategy.PASSTHROUGH.value]

        # ── LOSSY / CCR mode: layer relevance-split + lossy ON TOP of the fold ─
        # The operator has opted into lossy compression, so we reclaim more than
        # the fold's byte-exact floor. This is independent of the CCR-marker
        # sub-setting: markers-on makes any drop recoverable; the no-CCR-lossy
        # mode drops it unmarked by design. Either way STAGE 0 already banked the
        # lossless win, so nothing below can do worse than the fold.
        #
        # Stage B/C — prompt-conditioned relevance split for LOG/SEARCH: keep the
        # high-relevance records byte-verbatim (lossless-folded) and send only the
        # low-value tail to the lossy compressor (Kompress; CCR-marked and thus
        # recoverable when markers are on). It self-gates on beating the whole-
        # block fold, so when it fires it is strictly smaller than the STAGE 0
        # floor; otherwise it returns None and we keep the fold below. DIFF is
        # excluded — Kompressing hunks breaks `git apply`.
        if self.config.relevance_split and strategy in (
            CompressionStrategy.LOG,
            CompressionStrategy.SEARCH,
        ):
            kind = "log" if strategy is CompressionStrategy.LOG else "search"
            split = self._relevance_split_compress(content, kind, context)
            if split is not None:
                return split, _estimate_tokens(split), [kind, "relevance_split"]

        # No relevance split adopted → return the STAGE 0 lossless fold as the
        # floor. Lossless-then-lossy: before returning, run the aggressive lossy
        # compressor on the byte-folded remainder and keep it IFF it removes a
        # further meaningful chunk (Kompress must save >= _lossy_min_extra_savings
        # beyond the fold). Keeps the fold AND reclaims the semantic word-drop
        # tail, never doing worse than the fold. DIFF folds are returned verbatim
        # — Kompressing hunks corrupts `git apply`.
        if _ll_label is not None:
            _lossy_after_fold = (
                self._lossless_then_lossy
                and strategy != CompressionStrategy.DIFF
                and _ll_label != "lossless_diff"
                and not self._looks_like_diff(content)
            )
            if _lossy_after_fold:
                _fold_tokens = _estimate_tokens(_ll_content)
                try:
                    _komp, _komp_tokens = self._try_ml_compressor(_ll_content, context, question)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("lossy-after-fold failed: %s", exc)
                    _komp, _komp_tokens = None, None
                if (
                    _komp is not None
                    and _komp_tokens is not None
                    and _komp_tokens <= _fold_tokens * (1 - self._lossy_min_extra_savings)
                    and len(_komp) < len(_ll_content)
                ):
                    return (
                        _komp,
                        _komp_tokens,
                        [_ll_label, CompressionStrategy.KOMPRESS.value],
                    )
            return _ll_content, _estimate_tokens(_ll_content), [_ll_label]

        # CCR/lossy mode, nothing foldable (code/json/text/mixed) and no relevance
        # split → fall through to the lossy compressors below (kompress /
        # smart_crusher / code), which attach CCR retrieval markers when enabled.

        # ── External compressor dispatch (opt-in) ────────────────────────────
        # Immediately before the built-in if/elif, give a *selected* external
        # `headroom.compressor` first crack at this block IFF its declared
        # content_types match the block's detected content type. This runs only
        # when the operator selected a non-built-in compressor, so with no such
        # selection it is a single cheap guard and everything below is
        # byte-identical to today. Fully fail-open: a non-match, an error,
        # malformed/empty output, or an expansion all return None and fall
        # through to the EXISTING built-in dispatch UNCHANGED.
        external = self._try_external_compressor(content, strategy, context, question)
        if external is not None:
            return external

        try:
            if strategy == CompressionStrategy.CODE_AWARE:
                if self.config.enable_code_aware:
                    compressor = self._get_code_compressor()
                    if compressor:
                        compressor_name = type(compressor).__name__
                        # Registry-resolved dispatch: the built-in "code_aware"
                        # adapter delegates to this same getter+method with the
                        # language passed through ``config``, so on a real
                        # compression the content is byte-identical to the
                        # historical direct call. If the adapter did NOT compress
                        # (``compressed=False``), leave the local ``compressed``
                        # None so the EXISTING Kompress fallback below runs
                        # exactly as today.
                        output = self._registry_compress(
                            "code_aware",
                            strategy,
                            content,
                            context,
                            bias,
                            config={"language": language},
                        )
                        if output is not None and output.compressed:
                            compressed = output.content
                            compressed_tokens = len(output.content.split())
                            decision_reason = "code_aware"
                if compressed is None:
                    # Fallback to Kompress
                    compressed, compressed_tokens = self._try_ml_compressor(
                        content, context, question
                    )
                    strategy = CompressionStrategy.KOMPRESS  # Update for TOIN
                    actual_strategy = strategy
                    compressor_name = "KompressCompressor"
                    decision_reason = "code_aware_unavailable_fallback_kompress"
                    strategy_chain.append(CompressionStrategy.KOMPRESS.value)
                elif (
                    self._lossless_then_lossy
                    and compressed_tokens is not None
                    and compressed_tokens >= original_tokens
                ):
                    # #3 — lossless-then-lossy: code-aware produced NO net shrink
                    # and the lossless fold found nothing either, so this code
                    # block would otherwise pass through uncompressed. Give the
                    # lossy ML compressor (Kompress) a shot so lossy runs even when
                    # lossless has no savings. Reads are protected upstream, so
                    # only NON-read code reaches here. Keep Kompress ONLY if it
                    # actually shrinks (never inflate).
                    _k, _kt = self._try_ml_compressor(content, context, question)
                    if (
                        _k is not None
                        and _kt is not None
                        and _kt < original_tokens
                        and len(_k) < len(content)
                    ):
                        compressed, compressed_tokens = _k, _kt
                        strategy = CompressionStrategy.KOMPRESS
                        actual_strategy = strategy
                        compressor_name = "KompressCompressor"
                        decision_reason = "code_aware_no_shrink_fallback_kompress"
                        strategy_chain.append(CompressionStrategy.KOMPRESS.value)

            elif strategy == CompressionStrategy.SMART_CRUSHER:
                # SmartCrusher handles its own TOIN recording
                if self.config.enable_smart_crusher:
                    crusher = self._get_smart_crusher()
                    if crusher:
                        compressor_name = type(crusher).__name__
                        # Registry-resolved dispatch: the built-in "smart_crusher"
                        # adapter delegates to this same getter + ``.crush(...)``
                        # with the SAME query (``context``) and bias, so on a real
                        # compression the content is byte-identical to the historical
                        # direct call and the token metric (``_estimate_tokens`` of
                        # the crush output) is unchanged. SmartCrusher's ``.crush``
                        # always returns a string (``compressed=True``), so — inside
                        # this ``if crusher`` guard where the adapter's own getter
                        # returns the same cached crusher — ``compressed`` is always
                        # set exactly as the direct call did. The ``output`` /
                        # ``output.compressed`` guard mirrors the CODE_AWARE/HTML
                        # flip and only leaves ``compressed`` None in the defensive
                        # not-registered case (never reachable for a built-in).
                        output = self._registry_compress(
                            "smart_crusher", strategy, content, context, bias
                        )
                        if output is not None and output.compressed:
                            compressed = output.content
                            compressed_tokens = _estimate_tokens(output.content)
                            decision_reason = "smart_crusher"
                        # Fallback to Kompress (and possibly Log) is
                        # handled by the unified post-strategy block below
                        # — no inline fallback here to avoid duplicate
                        # Kompress invocations.

            elif strategy == CompressionStrategy.SEARCH:
                if self.config.enable_search_compressor:
                    compressor = self._get_search_compressor()
                    if compressor:
                        compressor_name = type(compressor).__name__
                        # Registry-resolved dispatch: the built-in "search" adapter
                        # delegates to this same getter+method, so the content is
                        # byte-identical to the historical direct call.
                        compressed = self._registry_compress_content(
                            "search", strategy, content, context, bias
                        )
                        compressed_tokens = _estimate_tokens(compressed)
                        decision_reason = "search_compressor"

            elif strategy == CompressionStrategy.LOG:
                if self.config.enable_log_compressor:
                    compressor = self._get_log_compressor()
                    if compressor:
                        compressor_name = type(compressor).__name__
                        # Registry-resolved dispatch: the built-in "log" adapter
                        # delegates to this same getter+method, so the content is
                        # byte-identical to the historical direct call.
                        compressed = self._registry_compress_content(
                            "log", strategy, content, context, bias
                        )
                        # Use the same word-count metric the rest of the
                        # router uses; `compressed_line_count` is in
                        # lines, not tokens — recording it here made
                        # ratios meaningless against `original_tokens`.
                        compressed_tokens = _estimate_tokens(compressed)
                        decision_reason = "log_compressor"

            elif strategy == CompressionStrategy.TABULAR:
                if self.config.enable_tabular_compressor:
                    compressor = self._get_tabular_compressor()
                    if compressor:
                        compressor_name = type(compressor).__name__
                        # Registry-resolved dispatch: the built-in "tabular" adapter
                        # delegates to this same getter+method, so the content is
                        # byte-identical to the historical direct call.
                        compressed = self._registry_compress_content(
                            "tabular", strategy, content, context, bias
                        )
                        compressed_tokens = _estimate_tokens(compressed)
                        decision_reason = "tabular_compressor"

            elif strategy == CompressionStrategy.CONFIG:
                if self.config.enable_config_compressor:
                    compressor = self._get_config_compressor()
                    if compressor:
                        compressor_name = type(compressor).__name__
                        # Registry-resolved dispatch: the built-in "config" adapter
                        # delegates to this same getter+method, so the content is
                        # byte-identical to the historical direct call. Keep the
                        # Measured with _estimate_tokens, matching the
                        # denominator (`original_tokens`, set from
                        # _estimate_tokens(content)) and every sibling branch. It
                        # used to be len(compressed.split()) — a WORD count in the
                        # numerator of a token ratio. Words run ~2.8x fewer than
                        # estimator tokens on config text, so a compressor that
                        # returned its input byte-identically reported a ratio of
                        # ~0.36 and, because min_ratio is 1.0, the router ACCEPTED
                        # the no-op: cached it, froze the verdict, emitted a
                        # router:config_compressor label and wrote a fabricated
                        # ~64% saving to TOIN. Measured on mkdocs.yml.
                        compressed = self._registry_compress_content(
                            "config", strategy, content, context, bias
                        )
                        compressed_tokens = _estimate_tokens(compressed)
                        decision_reason = "config_compressor"

            elif strategy == CompressionStrategy.DIFF:
                compressor = self._get_diff_compressor()
                if compressor:
                    compressor_name = type(compressor).__name__
                    result = compressor.compress(content, context=context)
                    compressed, compressed_tokens = (
                        result.compressed,
                        _estimate_tokens(result.compressed),
                    )
                    decision_reason = "diff_compressor"

            elif strategy == CompressionStrategy.HTML:
                if self.config.enable_html_extractor:
                    extractor = self._get_html_extractor()
                    if extractor:
                        compressor_name = type(extractor).__name__
                        # Registry-resolved dispatch: the built-in "html" adapter
                        # delegates to this same getter + extract(). It reports
                        # ``compressed=False`` (and returns the original content)
                        # when nothing extracts, so we collapse that to
                        # ``compressed = None`` and the branch falls through to
                        # the bottom passthrough exactly as the historical
                        # ``result.extracted is None`` path (chain
                        # ``[html, passthrough]``). A real extraction is
                        # byte-identical to the historical ``result.extracted``.
                        output = self._registry_compress("html", strategy, content, context, bias)
                        compressed = (
                            output.content if output is not None and output.compressed else None
                        )
                        # Estimate tokens from extracted text (simple word count)
                        compressed_tokens = _estimate_tokens(compressed) if compressed else 0
                        decision_reason = "html_extractor"

            elif strategy == CompressionStrategy.KOMPRESS:
                # Registry-resolved dispatch: the built-in "kompress" adapter
                # delegates to the SAME ``_try_ml_compressor(content, context,
                # question)`` the router historically called here — with
                # ``question`` forwarded via the CompressInput config — so the
                # compressed CONTENT is byte-identical to the direct call.
                # ``_try_ml_compressor`` always returns a str (passthrough on a
                # no-op / unavailable model), so the adapter always reports
                # ``compressed=True``; ``output`` is ``None`` only in the
                # defensive not-registered case, which falls through to the
                # bottom passthrough exactly as before. The token count is now
                # ``_estimate_tokens(output.content)`` — the router's calibrated
                # estimate — replacing the Kompress model's own tuple
                # ``compressed_tokens``. This is the ONE approved,
                # non-byte-identical change (a reported metric only; see the
                # decision-impact note: no keep/drop, fallback, or lossless-
                # then-lossy gate reads the KOMPRESS/TEXT ``compressed_tokens``).
                output = self._registry_compress(
                    "kompress", strategy, content, context, bias, question=question
                )
                if output is not None:
                    compressed = output.content
                    compressed_tokens = _estimate_tokens(output.content)
                compressor_name = "KompressCompressor"
                decision_reason = "kompress"

            elif strategy == CompressionStrategy.TEXT:
                # Prefer Kompress ML compressor for text; passes through unchanged
                # if Kompress is not available. Registry-resolved dispatch via the
                # SAME built-in "kompress" adapter (TEXT and KOMPRESS share the ML
                # compressor) with ``question`` forwarded via config, so the
                # compressed CONTENT is byte-identical to the historical direct
                # ``_try_ml_compressor(content, context, question)`` call. The
                # token count is now ``_estimate_tokens(output.content)`` (the same
                # approved metric change as the KOMPRESS branch above).
                output = self._registry_compress(
                    "kompress", strategy, content, context, bias, question=question
                )
                if output is not None:
                    compressed = output.content
                    compressed_tokens = _estimate_tokens(output.content)
                compressor_name = "KompressCompressor"
                decision_reason = "text_uses_kompress"

            elif strategy == CompressionStrategy.PASSTHROUGH:
                compressed = content
                compressed_tokens = original_tokens
                compressor_name = "Passthrough"
                decision_reason = "explicit_passthrough"

        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            decision_reason = "compression_exception"
            logger.warning("Compression with %s failed: %s", strategy.value, e)

        # If compression succeeded, record to TOIN
        if compressed is not None and compressed_tokens is not None:
            fallback_eligible_strategy = strategy in {
                CompressionStrategy.SMART_CRUSHER,
                CompressionStrategy.CODE_AWARE,
                CompressionStrategy.TABULAR,
                CompressionStrategy.CONFIG,
            }
            fallback_no_savings = compressed == content or compressed_tokens >= original_tokens
            if fallback_eligible_strategy and fallback_no_savings:
                # Skip if Kompress was already tried by an inline fallback
                # (e.g. CODE_AWARE's code-compressor-unavailable path at
                # line 1249).  Prevents a duplicate strategy_chain entry
                # and a wasted second _try_ml_compressor call.
                already_tried_kompress = CompressionStrategy.KOMPRESS.value in strategy_chain
                if not already_tried_kompress:
                    strategy_chain.append(CompressionStrategy.KOMPRESS.value)
                    fallback_compressed, fallback_tokens = self._try_ml_compressor(
                        content, context, question
                    )
                else:
                    fallback_compressed = compressed
                    fallback_tokens = compressed_tokens
                if fallback_tokens < compressed_tokens:
                    compressed = fallback_compressed
                    compressed_tokens = fallback_tokens
                    actual_strategy = CompressionStrategy.KOMPRESS
                    compressor_name = "KompressCompressor"
                    decision_reason = f"{decision_reason}_fallback_kompress_after_no_savings"
                else:
                    # Last-ditch: line-structured compressors (the proxy's
                    # own log dumps land here — repetitive JSONL that
                    # Kompress can't shrink but the log compressor can).
                    # Only attempted when the strategy was SMART_CRUSHER so
                    # we don't reroute genuine code/diff content.
                    #
                    # JSON-validity guard (#1306): the native magika detector
                    # classifies content by shape, not parseability, so a
                    # truncated/mid-stream JSON tool output is tagged
                    # ``json_array`` and routed to SMART_CRUSHER. SmartCrusher
                    # returns it unchanged (it can't parse the broken JSON),
                    # Kompress passes it through, and then the LogCompressor
                    # treats the whole thing as a multi-thousand-line "log"
                    # and collapses it to a single CCR-retrieval marker —
                    # 99.9% data loss when CCR retrieval isn't configured.
                    # Skip the Log fallback when the content isn't actually
                    # valid JSON; the passthrough at the bottom of this
                    # function preserves it verbatim. Valid JSON arrays still
                    # get the Log fallback (LogCompressor is a no-op on them).
                    if (
                        strategy == CompressionStrategy.SMART_CRUSHER
                        and self.config.enable_log_compressor
                        and _content_is_valid_json(content)
                    ):
                        log_compressor = self._get_log_compressor()
                        if log_compressor is not None:
                            strategy_chain.append(CompressionStrategy.LOG.value)
                            try:
                                log_result = log_compressor.compress(content, bias=bias)
                            except Exception as exc:  # noqa: BLE001
                                logger.debug("Log fallback failed for SMART_CRUSHER: %s", exc)
                            else:
                                log_compressed_tokens = _estimate_tokens(log_result.compressed)
                                if log_compressed_tokens < compressed_tokens:
                                    compressed = log_result.compressed
                                    compressed_tokens = log_compressed_tokens
                                    actual_strategy = CompressionStrategy.LOG
                                    compressor_name = type(log_compressor).__name__
                                    decision_reason = (
                                        f"{decision_reason}_fallback_log_after_no_savings"
                                    )

            # ── lossless_then_lossy (general): LAYER lossy on top of a
            #    conservative strategy result ──────────────────────────────
            # SEARCH/LOG/HTML compressors are structural and often bank only a
            # trickle (e.g. search keeps every keyword-matching line, so a grep
            # dump into a large data/config file barely shrinks). The zero-
            # savings fallback above never fires in that case (there WAS a tiny
            # win), so lossy never runs. When the operator opted into
            # lossless_then_lossy, run Kompress over whatever the strategy
            # produced and KEEP it only if it removes a further meaningful chunk
            # (>= _lossy_min_extra_savings beyond the strategy result) and is
            # actually shorter — never inflating, never doing worse than the
            # strategy output. DIFF is excluded (Kompress corrupts ``git
            # apply``); TEXT/KOMPRESS already ran Kompress; CODE_AWARE has its
            # own inline no-shrink fallback; SMART_CRUSHER/TABULAR use the
            # zero-savings fallback above.
            if (
                self._lossless_then_lossy
                and compressed is not None
                and compressed_tokens is not None
                and strategy
                in {
                    CompressionStrategy.SEARCH,
                    CompressionStrategy.LOG,
                    CompressionStrategy.HTML,
                }
                and not self._looks_like_diff(content)
            ):
                try:
                    _layer_k, _layer_kt = self._try_ml_compressor(compressed, context, question)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("lossless_then_lossy layer failed: %s", exc)
                    _layer_k, _layer_kt = None, None
                if (
                    _layer_k is not None
                    and _layer_kt is not None
                    and _layer_kt <= compressed_tokens * (1 - self._lossy_min_extra_savings)
                    and len(_layer_k) < len(compressed)
                ):
                    compressed, compressed_tokens = _layer_k, _layer_kt
                    actual_strategy = CompressionStrategy.KOMPRESS
                    compressor_name = "KompressCompressor"
                    strategy_chain.append(CompressionStrategy.KOMPRESS.value)
                    decision_reason = f"{decision_reason}_lossless_then_lossy_layer"

            # Re-narrow for mypy: all reassignments above produce str, but
            # mypy 1.14.x widens after nested try/except/else reassignments.
            assert compressed is not None
            if logger.isEnabledFor(logging.DEBUG):
                _log_router_debug(
                    "content_router_strategy_result",
                    requested_strategy=requested_strategy.value,
                    actual_strategy=actual_strategy.value,
                    strategy_chain=strategy_chain,
                    compressor=compressor_name,
                    reason=decision_reason,
                    language=language,
                    question=question,
                    bias=bias,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    tokens_saved=max(0, original_tokens - compressed_tokens),
                    compression_ratio=compressed_tokens / original_tokens
                    if original_tokens
                    else 1.0,
                    json_shape=_json_shape(content),
                    input=content,
                    output=compressed,
                    error=error,
                )
            self._record_to_toin(
                strategy=strategy,
                content=content,
                compressed=compressed,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                language=language,
                context=context,
            )
            return compressed, compressed_tokens, strategy_chain

        # Fallback: return unchanged
        strategy_chain.append(CompressionStrategy.PASSTHROUGH.value)
        if logger.isEnabledFor(logging.DEBUG):
            _log_router_debug(
                "content_router_strategy_result",
                requested_strategy=requested_strategy.value,
                actual_strategy=CompressionStrategy.PASSTHROUGH.value,
                strategy_chain=strategy_chain,
                compressor=None,
                reason=decision_reason,
                language=language,
                question=question,
                bias=bias,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                tokens_saved=0,
                compression_ratio=1.0,
                json_shape=_json_shape(content),
                input=content,
                output=content,
                error=error,
            )
        return content, original_tokens, strategy_chain

    def _try_ml_compressor(
        self,
        content: str,
        context: str,
        question: str | None = None,
        target_ratio: float | None = None,
    ) -> tuple[str, int]:
        """ML-based compression using Kompress.

        Kompress (ModernBERT, trained on 330K structured tool outputs)
        auto-downloads from HuggingFace on first use. No heuristic fallback.

        Custom/workflow XML tags (<system-reminder>, <tool_call>, <thinking>)
        are protected before compression and restored after.  Standard HTML
        tags are left alone (HTMLExtractor handles those separately).

        Args:
            content: Content to compress.
            context: User context.
            question: Optional question for QA-aware compression.

        Returns:
            Tuple of (compressed, token_count).
        """
        from .tag_protector import protect_tags, restore_tags

        # Protect custom tags before any ML compression
        cleaned, protected = protect_tags(
            content,
            compress_tagged_content=self.config.compress_tagged_content,
        )

        # If the entire content is custom tags with nothing to compress
        if protected and not cleaned.strip():
            return content, _estimate_tokens(content)

        # Use the cleaned (tag-free) text for compression
        text_to_compress = cleaned if protected else content
        compressed: str | None = None
        compressed_tokens: int | None = None

        # Phase 0 (#1171): size gate. This is the single ML boundary, so gating
        # here covers EVERY kompress entry point -- TEXT, KOMPRESS-direct,
        # CODE_AWARE->KOMPRESS, and the strategy-fallback path all route through
        # _try_ml_compressor. Kompress ONNX inference is O(tokens) and runs
        # synchronously on the request thread; on a large/cold context it
        # exceeds the 30s budget and leaks a non-preemptible worker (#1171).
        # Above the ceiling, route to the fast LogCompressor (or pass through)
        # rather than ModernBERT, keeping the request path bounded.
        # Compared with _estimate_tokens, not len()/4. The cap is expressed in
        # TOKENS, and chars/4 under-counts dense payloads — compact JSON runs
        # ~3.2 chars/token — so a band existed where an oversized payload passed
        # the gate. Measured: 177,781 chars of compact JSON is 44,445 by chars/4
        # (under the 50,000 cap, gate silent) but 55,557 estimator tokens, 11%
        # over. That is exactly the >30s non-preemptible ONNX inference this gate
        # exists to prevent (#1171).
        if (
            self._kompress_max_tokens > 0
            and _estimate_tokens(text_to_compress) > self._kompress_max_tokens
        ):
            self._kompress_gate_fires += 1
            self._observe_kompress_size_gate("exceeded")
            logger.info(
                "kompress size-gate fired: ~%d tok (>%d) routed off ML (fire #%d)",
                len(text_to_compress) // 4,
                self._kompress_max_tokens,
                self._kompress_gate_fires,
            )
            out = text_to_compress
            crusher = self._get_text_crusher()
            if crusher is not None:
                try:
                    out = crusher.compress(text_to_compress, context=context or "").compressed
                except Exception as e:
                    logger.warning(
                        "Kompress size-gate -> TextCrusher failed (%s); passing through", e
                    )
                    out = text_to_compress
            elif self.config.enable_log_compressor:
                lc = self._get_log_compressor()
                if lc:
                    try:
                        out = lc.compress(text_to_compress).compressed
                    except Exception as e:
                        logger.warning(
                            "Kompress size-gate -> LogCompressor failed (%s); passing through", e
                        )
                        out = text_to_compress
            if protected:
                out = restore_tags(out, protected)
            return out, _estimate_tokens(out)

        # Reached only when the gate is enabled and this eligible block is
        # under the ceiling — the counterpart "within" outcome to the
        # "exceeded" branch above. Lets /metrics prove the gate's hit rate.
        if self._kompress_max_tokens > 0:
            self._observe_kompress_size_gate("within")

        # Primary: Kompress. On a cold cache the model is fetched once in the
        # background (ensure_background_load) instead of blocking this request
        # thread on a 274MB download that races the compression timeout and
        # fails open. Until it is cached, route around the deep path.
        # skip_kompress (cold-start fast pass) takes the identical fallback.
        if self.config.enable_kompress and not getattr(self, "_runtime_skip_kompress", False):
            compressor = self._get_kompress()
            if compressor:
                if not compressor.is_ready():
                    compressor.ensure_background_load()
                    # Surface: warn once per ContentRouter instance so operators
                    # know compression is degraded — model not cached, or
                    # HuggingFace unreachable (corporate firewall, SSL, etc.).
                    if not getattr(self, "_kompress_warned", False):
                        logger.warning(
                            "Kompress model not ready; requests will not be "
                            "compressed. Check HuggingFace connectivity or "
                            "pre-download: headroom-ai[ml] + first-run warmup."
                        )
                        self._kompress_warned = True
                else:
                    try:
                        compress_kwargs: dict[str, Any] = {
                            "context": context,
                            "question": question,
                            "target_ratio": (
                                target_ratio
                                if target_ratio is not None
                                else getattr(self, "_runtime_target_ratio", None)
                            ),
                            "allow_download": False,
                        }
                        # When custom tags are protected, ``text_to_compress`` is
                        # the placeholdered intermediate ({{HEADROOM_TAG_N}}). Pass
                        # the pre-protection ``content`` as ``ccr_original`` so CCR
                        # stores the real text, not the placeholder — otherwise a
                        # later full retrieval returns {{HEADROOM_TAG_N}} and the
                        # protected block is lost from the retrieval path. Only set
                        # it when tags were protected so callers/compressors that
                        # don't accept the kwarg are unaffected on the common path.
                        if protected:
                            compress_kwargs["ccr_original"] = content
                        result = compressor.compress(text_to_compress, **compress_kwargs)
                        compressed = result.compressed
                        compressed_tokens = result.compressed_tokens
                    except Exception as e:
                        logger.warning("Kompress failed: %s", e)

        if compressed is None:
            return content, _estimate_tokens(content)

        # Restore protected tag blocks into the compressed text
        if protected:
            compressed = restore_tags(compressed, protected)
            compressed_tokens = _estimate_tokens(compressed)

        return compressed, compressed_tokens or _estimate_tokens(compressed)

    def _experimental_compress_read(self, content: Any, context: str = "") -> str | None:
        """EXPERIMENT (HEADROOM_EXPERIMENTAL_READ_KEEP_RATIO): lightly Kompress a
        protected file read instead of passing it verbatim.

        Reads are protected to keep the exact bytes the agent patches from, so
        this is OFF by default and a resolve-risk probe: at keep ratio 0.9 the
        model still sees ~90% of the (importance-ranked) tokens. Returns the
        compressed text only when it actually shrank and is non-empty; otherwise
        None, so the caller falls back to verbatim protection. Never raises.
        """
        ratio = getattr(self, "_exp_read_keep_ratio", 0.0)
        if not ratio or not isinstance(content, str) or len(content) < 200:
            return None
        try:
            out, _ = self._try_ml_compressor(content, context or "", target_ratio=ratio)
        except Exception as exc:  # noqa: BLE001
            logger.debug("experimental read-kompress failed: %s", exc)
            return None
        return out if (out and len(out) < len(content)) else None

    def _strategy_from_detection_type(self, content_type: ContentType) -> CompressionStrategy:
        """Get strategy from ContentType enum."""
        mapping = {
            ContentType.SOURCE_CODE: CompressionStrategy.CODE_AWARE,
            ContentType.JSON_ARRAY: CompressionStrategy.SMART_CRUSHER,
            ContentType.SEARCH_RESULTS: CompressionStrategy.SEARCH,
            ContentType.BUILD_OUTPUT: CompressionStrategy.LOG,
            ContentType.GIT_DIFF: CompressionStrategy.DIFF,
            ContentType.HTML: CompressionStrategy.HTML,
            ContentType.TABULAR: CompressionStrategy.TABULAR,
            ContentType.STRUCTURED_CONFIG: CompressionStrategy.CONFIG,
            ContentType.PLAIN_TEXT: CompressionStrategy.TEXT,
        }
        return mapping.get(content_type, self.config.fallback_strategy)

    def _content_type_from_strategy(self, strategy: CompressionStrategy) -> ContentType:
        """Get ContentType from strategy."""
        mapping = {
            CompressionStrategy.CODE_AWARE: ContentType.SOURCE_CODE,
            CompressionStrategy.SMART_CRUSHER: ContentType.JSON_ARRAY,
            CompressionStrategy.SEARCH: ContentType.SEARCH_RESULTS,
            CompressionStrategy.LOG: ContentType.BUILD_OUTPUT,
            CompressionStrategy.DIFF: ContentType.GIT_DIFF,
            CompressionStrategy.HTML: ContentType.HTML,
            CompressionStrategy.TABULAR: ContentType.TABULAR,
            CompressionStrategy.CONFIG: ContentType.STRUCTURED_CONFIG,
            CompressionStrategy.TEXT: ContentType.PLAIN_TEXT,
            CompressionStrategy.KOMPRESS: ContentType.PLAIN_TEXT,
            CompressionStrategy.PASSTHROUGH: ContentType.PLAIN_TEXT,
        }
        return mapping.get(strategy, ContentType.PLAIN_TEXT)

    # Lazy compressor getters

    def _get_code_compressor(self) -> Any:
        """Get CodeAwareCompressor (lazy load)."""
        if self._code_compressor is None:
            try:
                from .code_compressor import (
                    CodeAwareCompressor,
                    CodeCompressorConfig,
                    _check_tree_sitter_available,
                )

                if _check_tree_sitter_available():
                    self._code_compressor = CodeAwareCompressor(
                        CodeCompressorConfig(
                            enable_ccr=self.config.ccr_inject_marker,
                        )
                    )
                else:
                    logger.debug("tree-sitter not available")
            except ImportError:
                logger.debug("CodeAwareCompressor not available")
        return self._code_compressor

    def _get_smart_crusher(self) -> Any:
        """Get SmartCrusher (lazy load) with CCR config."""
        if self._smart_crusher is None:
            try:
                from ..config import CCRConfig
                from .smart_crusher import SmartCrusher, SmartCrusherConfig

                # Pass CCR config for marker injection
                ccr_config = CCRConfig(
                    enabled=self.config.ccr_enabled,
                    inject_retrieval_marker=self.config.ccr_inject_marker,
                )
                # Full config override (smart_crusher) wins as the base;
                # the per-field knobs from savings profiles still apply on top.
                crusher_config = self.config.smart_crusher or SmartCrusherConfig()
                if self.config.smart_crusher_max_items_after_crush is not None:
                    crusher_config.max_items_after_crush = (
                        self.config.smart_crusher_max_items_after_crush
                    )
                if self.config.smart_crusher_lossless_only is not None:
                    crusher_config.lossless_only = self.config.smart_crusher_lossless_only
                self._smart_crusher = SmartCrusher(
                    config=crusher_config,
                    ccr_config=ccr_config,
                    with_compaction=self.config.smart_crusher_with_compaction,
                )
            except ImportError:
                logger.debug("SmartCrusher not available")
        return self._smart_crusher

    def _get_search_compressor(self) -> Any:
        """Get SearchCompressor (lazy load)."""
        if self._search_compressor is None:
            try:
                from .search_compressor import SearchCompressor, SearchCompressorConfig

                cfg = self.config.search_compressor or SearchCompressorConfig()
                cfg = replace(
                    cfg,
                    group_by_file=self.config.search_group_by_file,
                    enable_ccr=self.config.ccr_inject_marker,
                )
                self._search_compressor = SearchCompressor(cfg)
            except ImportError:
                logger.debug("SearchCompressor not available")
        return self._search_compressor

    def _get_log_compressor(self) -> Any:
        """Get LogCompressor (lazy load)."""
        if self._log_compressor is None:
            try:
                from .log_compressor import LogCompressor, LogCompressorConfig

                cfg = self.config.log_compressor or LogCompressorConfig()
                cfg = replace(cfg, enable_ccr=self.config.ccr_inject_marker)
                self._log_compressor = LogCompressor(cfg)
            except ImportError:
                logger.debug("LogCompressor not available")
        return self._log_compressor

    def _get_relevance_scorer(self) -> Any:
        """Get the relevance scorer for the split (lazy, cached, non-blocking).

        Tier comes from ``config.relevance``. For ``bm25`` this is instant. For
        ``hybrid``/``embedding`` the scorer serves **BM25 immediately** and the
        embedding model is warmed in a background thread; once it's cached the
        scorer is swapped in (GIL-atomic ref write), so a request never blocks
        on the ~30MB download. Returns None (cached) on failure. Never raises.
        """
        if self._relevance_scorer is not None or self._relevance_scorer_tried:
            return self._relevance_scorer
        self._relevance_scorer_tried = True
        tier = (self.config.relevance.tier or "hybrid").lower()
        try:
            from ..relevance import BM25Scorer

            if tier == "bm25":
                self._relevance_scorer = BM25Scorer()
            else:
                # Serve BM25 now; swap to the embedding-backed scorer once warm.
                self._relevance_scorer = BM25Scorer()
                self._start_relevance_prewarm(tier)
        except Exception as exc:  # noqa: BLE001
            logger.debug("relevance scorer unavailable: %s", exc)
            self._relevance_scorer = None
        return self._relevance_scorer

    def _start_relevance_prewarm(self, tier: str) -> None:
        """Warm the embedding model off the request thread, then swap it in.

        Idempotent. On failure (fastembed missing, download error) the router
        just stays on the BM25 scorer set by ``_get_relevance_scorer``.
        """
        if getattr(self, "_relevance_prewarm_started", False):
            return
        self._relevance_prewarm_started = True

        def _warm() -> None:
            try:
                from ..relevance import create_scorer

                scorer = create_scorer(tier)
                # Force the model download+load and a first embed here, in the
                # background — so the first real request finds it warm.
                scorer.score_batch(["warmup"], "warmup")
                self._relevance_scorer = scorer  # GIL-atomic ref swap
            except Exception as exc:  # noqa: BLE001
                logger.debug("relevance model prewarm failed; staying on BM25: %s", exc)

        threading.Thread(target=_warm, name="relevance-prewarm", daemon=True).start()

    def _relevance_split_compress(self, content: str, kind: str, query: str) -> str | None:
        """Prompt-conditioned KEEP/DROP split for the compression tail.

        Keeps high-relevance records byte-verbatim (lossless-compacted) and
        Kompresses the low-relevance tail (identifiers pinned by Kompress
        MUST_KEEP). Mode-agnostic: the tail's marker behavior is decided by
        ``_try_ml_compressor`` — marker-free in lossless mode, retrieval-marker
        in CCR mode. Returns the spliced output, or None to fall back to the
        normal path when the scorer is unavailable, the query is empty, nothing
        is dropped, or the split doesn't beat plain compaction. Never raises.

        Embedding cost is bounded two ways: the model is pre-warmed off the
        request thread (BM25 until it's ready, see _get_relevance_scorer) and
        outputs segmenting into more than ``relevance_max_records`` records skip
        the split entirely.
        """
        scorer = self._get_relevance_scorer()
        if scorer is None or not query.strip():
            return None
        from .lossless_compaction import compact_lossless

        try:
            runs = plan_relevance_split(
                content,
                query,
                scorer,
                threshold=self.config.relevance.relevance_threshold,
                adaptive=self.config.relevance_adaptive_threshold,
                max_records=self.config.relevance_max_records,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("relevance split failed (%s); falling back", exc)
            return None

        # No low-relevance tail → plain compaction is already optimal here.
        if not any(not keep for keep, _ in runs):
            return None

        out_parts: list[str] = []
        for keep, text in runs:
            if keep:
                out_parts.append(compact_lossless(text, kind))
                continue
            try:
                compressed, _ = self._try_ml_compressor(text, query)
            except Exception as exc:  # noqa: BLE001
                logger.debug("kompress tail failed (%s); keeping verbatim", exc)
                compressed = compact_lossless(text, kind)
            out_parts.append(compressed)

        result = "".join(out_parts)
        # Adopt only when it beats plain whole-block lossless compaction.
        baseline = compact_lossless(content, kind)
        return result if len(result) < len(baseline) else None

    def _get_text_crusher(self) -> Any:
        """Get TextCrusher (Phase 2, lazy load). Returns None when disabled, or
        when the native ``headroom._core`` extension is not built (mirrors the
        ImportError handling of the other ``_get_*`` compressor getters)."""
        if not getattr(self, "_text_crusher_enabled", False):
            return None
        if self._text_crusher is None:
            try:
                from .text_crusher import TextCrusher, TextCrusherConfig

                cfg = self.config.text_crusher or TextCrusherConfig()
                self._text_crusher = TextCrusher(cfg)
            except ImportError:
                logger.debug("TextCrusher (headroom._core) unavailable; disabling gate route")
                self._text_crusher_enabled = False
        return self._text_crusher

    def _get_tabular_compressor(self) -> Any:
        """Get TabularCompressor (lazy load)."""
        if self._tabular_compressor is None:
            try:
                from .tabular_ingest import TabularCompressor

                self._tabular_compressor = TabularCompressor()
            except ImportError:  # pragma: no cover - defensive; tabular_ingest is pure stdlib
                logger.debug("TabularCompressor not available")
        return self._tabular_compressor

    def _get_config_compressor(self) -> Any:
        """Get ConfigCompressor (lazy load)."""
        if self._config_compressor is None:
            try:
                from .config_compressor import ConfigCompressor, ConfigCompressorConfig

                self._config_compressor = ConfigCompressor(
                    ConfigCompressorConfig(enable_ccr=self.config.ccr_inject_marker)
                )
            except ImportError:  # pragma: no cover - defensive; module is pure stdlib
                logger.debug("ConfigCompressor not available")
        return self._config_compressor

    def _get_diff_compressor(self) -> Any:
        """Get DiffCompressor (lazy load). Rust-only — Python implementation
        retired in Stage 3b. The wheel (`headroom._core`) is a hard import.
        """
        if self._diff_compressor is None:
            from .diff_compressor import DiffCompressor, DiffCompressorConfig

            cfg = self.config.diff_compressor or DiffCompressorConfig()
            cfg = replace(cfg, enable_ccr=self.config.ccr_inject_marker)
            self._diff_compressor = DiffCompressor(cfg)
        return self._diff_compressor

    def _get_html_extractor(self) -> Any:
        """Get HTMLExtractor (lazy load)."""
        if self._html_extractor is None:
            try:
                from .html_extractor import HTMLExtractor

                self._html_extractor = HTMLExtractor()
            except ImportError:
                logger.debug("HTMLExtractor not available (install trafilatura)")
        return self._html_extractor

    @staticmethod
    def _prefetch_kompress_artifacts_async(kompress_config: Any) -> bool:
        """Start a background download of the Kompress model files, if needed.

        Files only — see ``prefetch_kompress_artifacts`` for why startup must not
        build the model. Returns ``True`` when a prefetch is running.
        """
        try:
            from .kompress_compressor import HF_MODEL_ID, ensure_background_prefetch

            model_id = getattr(kompress_config, "model_id", None) or HF_MODEL_ID
            return ensure_background_prefetch(str(model_id))
        except Exception as e:  # pragma: no cover - defensive; never break startup
            logger.debug("Kompress artifact prefetch skipped: %s", e)
            return False

    def eager_load_compressors(self) -> dict[str, str]:
        """Pre-load compressors at startup to avoid first-request latency.

        Call this during proxy startup to load models and parsers
        before any requests arrive. Eliminates cold-start latency spikes.

        Returns:
            Dict of component name -> status string for logging.
        """
        status: dict[str, str] = {}

        # 1. ML text compressor: Kompress.
        #
        # Native model initialization stays out of the blocking startup/lifespan
        # path. The existing lazy request path loads Kompress on first use. This is
        # load-bearing, NOT laziness: on RHEL/CentOS 7-family hosts entering cached
        # Kompress native init before the port binds segfaults in libarrow/jemalloc
        # with no Python traceback (#1908, fixed by #2001) — a crash no try/except
        # can catch. Do not call `preload()` here.
        #
        # What we CAN do at startup is prefetch the model FILES. Downloading is
        # pure huggingface_hub HTTP — no ONNX session, no transformers import, so it
        # never touches the native path that #1908 crashes on. That removes the real
        # cold-start cost: previously the ~4-minute download began on the FIRST
        # REQUEST, and every request in that window went silently uncompressed
        # behind a single "model not ready" warning.
        if self.config.enable_kompress:
            compressor = self._get_kompress()
            if compressor:
                if not hasattr(compressor, "preload"):
                    status["kompress"] = "enabled"
                    status["kompress_backend"] = "unknown"
                else:
                    status["kompress"] = "deferred"
                    if self._prefetch_kompress_artifacts_async(getattr(compressor, "config", None)):
                        status["kompress_artifacts"] = "prefetching"
                    logger.info("Kompress model preload deferred until first request")
            else:
                status["kompress"] = "unavailable"

        # 2. Magika content detector (avoids 100-200ms on first content detection)
        try:
            from ..compression.detector import _get_magika, _magika_available

            if _magika_available():
                _get_magika()  # Initializes the singleton
                logger.info("Magika content detector pre-loaded at startup")
                status["magika"] = "enabled"
            else:
                status["magika"] = "not installed"
        except Exception as e:
            logger.debug("Magika pre-load skipped: %s", e)
            status["magika"] = "skipped"

        # Surface which onnxruntime dylib the Rust detection chain will load.
        # On Windows `headroom._ort` pins ORT_DYLIB_PATH at import time; an
        # unset value there means the bare DLL search applies, which lands on
        # the Windows ML System32 build known to deadlock ort session init
        # (Win11 24H2+, see headroom/_ort.py).
        if sys.platform.startswith("win"):
            ort_dylib = os.environ.get("ORT_DYLIB_PATH")
            if ort_dylib:
                logger.info("ORT dylib for Rust detection: %s", ort_dylib)
                status["ort_dylib"] = ort_dylib
            else:
                logger.warning(
                    "ORT_DYLIB_PATH is unset: Rust ML detection will use the system "
                    "DLL search, which deadlocks against the Windows ML System32 "
                    "onnxruntime.dll on Windows 11 24H2+. Install the `onnxruntime` "
                    "package or set ORT_DYLIB_PATH."
                )
                status["ort_dylib"] = "unset"

        # 3. CodeAware compressor + common tree-sitter parsers
        if self.config.enable_code_aware:
            code_compressor = self._get_code_compressor()
            if code_compressor:
                status["code_aware"] = "enabled"
                # Pre-load tree-sitter parsers for common languages
                # Each parser is ~50ms to load; doing it here avoids 500ms+ on first code hit
                try:
                    from .code_compressor import _check_tree_sitter_available, _get_parser

                    if _check_tree_sitter_available():
                        common_languages = [
                            "python",
                            "javascript",
                            "typescript",
                            "go",
                            "rust",
                            "java",
                            "c",
                            "cpp",
                        ]
                        loaded = []
                        for lang in common_languages:
                            try:
                                _get_parser(lang)
                                loaded.append(lang)
                            except (ValueError, ImportError):
                                pass  # Language not available, skip
                        if loaded:
                            logger.info("Tree-sitter parsers pre-loaded: %s", ", ".join(loaded))
                            status["tree_sitter"] = f"loaded ({len(loaded)} languages)"
                except Exception as e:
                    logger.debug("Tree-sitter pre-load skipped: %s", e)
                    status["tree_sitter"] = "skipped"
            else:
                status["code_aware"] = "not installed"

        # 4. SmartCrusher (lightweight init)
        smart_crusher = self._get_smart_crusher()
        if smart_crusher:
            status["smart_crusher"] = "ready"

        # 5. HTML extractor.
        #
        # By far the most expensive lazy import in the transform tree: MEASURED
        # 978ms for trafilatura -> htmldate -> dateparser and its timezone
        # tables, against 1-20ms for every other compressor module. It fires
        # from _get_html_extractor() on the first request carrying an HTML-ish
        # block or mixed-content section, so a real user pays the full second
        # mid-request. That is the single largest first-request stall in the
        # pipeline, which is why it is worth a line here.
        try:
            if self._get_html_extractor() is not None:
                status["html_extractor"] = "ready"
            else:
                status["html_extractor"] = "not installed"
        except Exception as e:
            logger.debug("HTML extractor pre-load skipped: %s", e)
            status["html_extractor"] = "skipped"

        # 6. TOIN singleton. Constructing it reads the learned-pattern file off
        # disk (MEASURED ~150ms at 5MB, and it grows with use). SmartCrusher
        # above does NOT pull it in, despite what a previous comment here
        # claimed — the first request did.
        try:
            from ..telemetry.toin import get_toin

            get_toin()
            status["toin"] = "ready"
        except Exception as e:
            logger.debug("TOIN pre-load skipped: %s", e)
            status["toin"] = "skipped"

        return status

    def _get_kompress(self) -> Any:
        """Get KompressCompressor (lazy load). Downloads from HuggingFace on first use.

        Respects runtime kompress_model kwarg:
        - None: use default (chopratejas/kompress-v2-base) — cached on self
        - "disabled": return None (skip ML compression entirely)
        - any model ID string: create compressor with that model
          (model weights are cached at module level in kompress_compressor.py,
          so repeated calls with the same model_id are cheap)
        """
        model_id = getattr(self, "_runtime_kompress_model", None)

        # Explicitly disabled — no ML compression
        if model_id == "disabled":
            return None

        # Remote Kompress (HEADROOM_KOMPRESS_ENDPOINT): offload inference to a
        # hosted /compress endpoint so a sandboxed proxy needs no local ML deps.
        # Intercepts BOTH default and custom-model paths (the endpoint's deployed
        # model is authoritative) and bypasses is_kompress_available() — there is
        # nothing to load locally. The CCR store stays proxy-local.
        remote = self._get_remote_kompress()
        if remote is not None:
            return remote

        # Custom model — don't touch self._kompress (that's the default cache)
        if model_id:
            try:
                from .kompress_compressor import (
                    KompressCompressor,
                    KompressConfig,
                    is_kompress_available,
                )

                if is_kompress_available():
                    return KompressCompressor(
                        config=KompressConfig(
                            model_id=model_id, enable_ccr=self.config.ccr_inject_marker
                        )
                    )
            except ImportError:
                pass
            return None

        # Default path — exactly as before, cached on self
        if self._kompress is None:
            try:
                from .kompress_compressor import (
                    KompressCompressor,
                    KompressConfig,
                    is_kompress_available,
                )

                if is_kompress_available():
                    # Honor the router's marker policy. In no-CCR / lossless mode
                    # (ccr_inject_marker=False) Kompress still compresses (lossy),
                    # but must NOT append a `Retrieve more: hash=` marker or write
                    # to the CCR store — otherwise the no-MCP guarantee breaks.
                    # Matches how search/log/diff/code receive enable_ccr.
                    self._kompress = KompressCompressor(
                        config=KompressConfig(enable_ccr=self.config.ccr_inject_marker)
                    )
            except ImportError:
                logger.debug("Kompress dependencies not available")
        return self._kompress

    def _get_remote_kompress(self) -> Any:
        """Return a cached RemoteKompressCompressor when HEADROOM_KOMPRESS_ENDPOINT
        is set, else None.

        The endpoint runs the model, so this needs no local ML deps and no
        is_kompress_available() gate. Cached per ContentRouter instance so the
        httpx connection pool is reused across requests.
        """
        endpoint = os.environ.get("HEADROOM_KOMPRESS_ENDPOINT", "").strip()
        if not endpoint:
            return None
        if getattr(self, "_kompress_remote", None) is None:
            from .kompress_compressor import KompressConfig
            from .kompress_remote import (
                DEFAULT_ENDPOINT_PATH,
                RemoteKompressCompressor,
                parse_endpoint_headers,
            )

            # Defaults reproduce the previous behaviour exactly, so existing
            # (Modal) deployments are unaffected: os.environ.get with a default
            # distinguishes "unset" (use /compress) from an explicit empty value
            # (the operator's endpoint is already a complete URL).
            self._kompress_remote = RemoteKompressCompressor(
                endpoint=endpoint,
                token=os.environ.get("HEADROOM_KOMPRESS_ENDPOINT_TOKEN") or None,
                config=KompressConfig(enable_ccr=self.config.ccr_inject_marker),
                path=os.environ.get("HEADROOM_KOMPRESS_ENDPOINT_PATH", DEFAULT_ENDPOINT_PATH),
                headers=parse_endpoint_headers(
                    os.environ.get("HEADROOM_KOMPRESS_ENDPOINT_HEADERS")
                ),
            )
            logger.info("Kompress: using remote endpoint %s", self._kompress_remote.url)
        return self._kompress_remote

    def _get_image_optimizer(self) -> Any:
        """Create an ImageCompressor for one optimization pass.

        The ImageCompressor handles image token compression using:
        - Trained MiniLM classifier from HuggingFace (chopratejas/technique-router)
        - SigLIP for image analysis
        - Provider-specific compression (OpenAI detail, Anthropic/Google resize)
        """
        try:
            from ..image import ImageCompressor

            return ImageCompressor()
        except ImportError:
            logger.debug("ImageCompressor not available")
            return None

    def optimize_images_in_messages(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        provider: str = "openai",
        user_query: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Optimize images in messages.

        This is a convenience method for image optimization that can be called
        directly or as part of the transform pipeline.

        Uses ImageCompressor with trained MiniLM router from HuggingFace
        (chopratejas/technique-router) + SigLIP for image analysis.

        Args:
            messages: Messages potentially containing images.
            tokenizer: Tokenizer for token counting (unused, kept for API compat).
            provider: LLM provider (openai, anthropic, google).
            user_query: User query for task intent detection (unused, auto-extracted).

        Returns:
            Tuple of (optimized_messages, metrics).
        """
        if not self.config.enable_image_optimizer:
            return messages, {"images_optimized": 0, "tokens_saved": 0}

        compressor = self._get_image_optimizer()
        if compressor is None:
            return messages, {"images_optimized": 0, "tokens_saved": 0}

        try:
            # Check if there are images to compress
            if not compressor.has_images(messages):
                return messages, {"images_optimized": 0, "tokens_saved": 0}

            # Compress images (query is auto-extracted from messages)
            optimized = compressor.compress(messages, provider=provider)

            # Get metrics from last compression
            result = compressor.last_result
            if result:
                metrics = {
                    "images_optimized": result.compressed_tokens < result.original_tokens,
                    "tokens_before": result.original_tokens,
                    "tokens_after": result.compressed_tokens,
                    "tokens_saved": result.original_tokens - result.compressed_tokens,
                    "technique": result.technique.value,
                    "confidence": result.confidence,
                }
            else:
                metrics = {"images_optimized": 0, "tokens_saved": 0}

            return optimized, metrics
        finally:
            if hasattr(compressor, "close"):
                compressor.close()

    # Transform interface

    def _build_tool_name_map(self, messages: list[dict[str, Any]]) -> dict[str, str]:
        """Build mapping from tool_call_id to tool_name.

        Scans assistant messages to find tool calls and extract their names.
        Supports both OpenAI and Anthropic message formats. Also populates
        ``self._tool_call_args`` (id → compact args text) in the same scan, so
        the relevance split can score a tool output against the *precise* ask
        that triggered it (grep pattern, read path, …), not just the user
        prompt. Read-only after build → safe to read from the parallel
        compression pass.
        """
        mapping: dict[str, str] = {}
        args_map: dict[str, str] = {}
        commands_map: dict[str, str] = {}

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            # OpenAI format: tool_calls array. Coalesce None -> [] : OpenAI/LiteLLM
            # assistant messages carry an explicit ``tool_calls: null`` (and
            # ``function_call: null``) when there are no calls, so ``.get(k, [])``
            # returns None (not []) and iterating it crashes _build_tool_name_map ->
            # apply() -> compression silently falls through to passthrough on every
            # OpenAI turn. This is the generic OpenAI-shape fix.
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    if name:
                        # Hermes deferred tools arrive wrapped as `tool_call`
                        # with the real name inside the arguments payload.
                        name = unwrap_tool_call_name(name, fn.get("arguments"))
                    if tc_id and name:
                        mapping[tc_id] = name
                        args = _tool_call_args_text(fn.get("arguments"))
                        if args:
                            args_map[tc_id] = args
                        command = _tool_call_command_text(fn.get("arguments"))
                        if command:
                            commands_map[tc_id] = command

            # Anthropic format: content blocks with type=tool_use
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tc_id = block.get("id", "")
                        name = block.get("name", "")
                        if name:
                            # Hermes deferred tools arrive wrapped as `tool_call`
                            # with the real name inside the input payload.
                            name = unwrap_tool_call_name(name, block.get("input"))
                        if tc_id and name:
                            mapping[tc_id] = name
                            args = _tool_call_args_text(block.get("input"))
                            if args:
                                args_map[tc_id] = args
                            command = _tool_call_command_text(block.get("input"))
                            if command:
                                commands_map[tc_id] = command

        self._tool_call_args = args_map
        self._tool_call_commands = commands_map
        return mapping

    def _net_cost_allows(
        self,
        *,
        slot_idx: int,
        original_tokens: int,
        compressed_tokens: int,
        suffix_tokens: list[int],
        route_counts: dict[str, int],
        transforms_applied: list[str],
        batch_state: dict[str, int | None] | None = None,
        p_alive_override: float | None = None,
        write_multiplier: float | None = None,
    ) -> bool:
        """Break-even gate for one candidate mutation (#856 P2, flag-gated).

        Consumes ``CompressionPolicy.net_mutation_gain`` with the issue's v1
        estimators: ΔT is the candidate's exact token saving (the compressed
        form is already computed when this runs), S is the token total after
        the slot, and R / P_alive are env-tunable constants
        (``HEADROOM_NET_COST_EXPECTED_READS``, default 10;
        ``HEADROOM_NET_COST_P_ALIVE``, default 1.0 — the conservative
        full-penalty assumption). Every decision is logged with its inputs
        and counted in ``route_counts`` so the flag can be validated from
        telemetry before any default-on.

        #856 P3a (batch deep edits): a mutation at depth K busts the
        provider's cached suffix after K, so every *later* candidate at a
        deeper slot rides that same invalidation for free — mutating it adds
        no incremental cache-bust cost. ``batch_state["floor"]`` tracks the
        shallowest slot already admitted as a net-positive mutation. When the
        current candidate sits strictly deeper than that floor, S is charged
        as 0 (rather than the full invalidated suffix), so the break-even
        formula admits it on the write/read economics alone. Charging S=0 via
        the same ``net_mutation_gain`` (instead of blanket-admitting on
        ``delta_t > 0``) keeps the decision conservative: it never admits a
        mutation the real economics would reject. The floor is only set/lowered
        by full-S admits, so a slot only ever rides free behind a genuinely
        mutated shallower slot. Each batch admission emits the
        ``router:netcost_batch_admit`` marker and the ``netcost_batch_admitted``
        counter for telemetry.

        #856 P3b (idle-timer compaction): ``p_alive_override``, when supplied
        by the caller, replaces the static ``HEADROOM_NET_COST_P_ALIVE``
        constant. It is derived in ``apply`` from how long the session has
        been idle relative to the provider cache TTL
        (``max(0, 1 − idle_s / ttl)``). As the cached suffix nears lapse
        P_alive → 0, the ``P_alive·(w−r)·(S+ΔT)`` penalty vanishes, and edits
        that would lose to a warm suffix become free — the suffix is about to
        be rebuilt cold regardless. ``None`` preserves the P2 env-constant
        behaviour. An admit made under a decayed (``< 1.0``) idle P_alive emits
        the ``router:netcost_idle_compaction`` marker and the
        ``netcost_idle_admitted`` counter.
        """
        delta_t = max(0, original_tokens - compressed_tokens)
        # Batch reclaim: if a shallower slot was already admitted, its
        # cache-bust already invalidated everything after it, including this
        # slot — so charge S=0 here. Otherwise S is the full suffix after the
        # candidate (P2 v1 estimator).
        floor = batch_state.get("floor") if batch_state is not None else None
        batch_reclaim = floor is not None and slot_idx > floor
        suffix = 0 if batch_reclaim else suffix_tokens[slot_idx + 1]
        policy = self._runtime_compression_policy
        if policy is None:
            from .compression_policy import policy_default_payg

            policy = policy_default_payg()
        # Malformed env values fall back to defaults with a warning rather
        # than crashing the request path (same posture as the #851 breaker
        # env guard).
        # ``float()`` parses "nan"/"inf" without raising, so a non-finite
        # check is needed in addition to the ValueError guard — otherwise a
        # malformed-but-parseable value would be logged verbatim (misleading
        # telemetry) even though ``net_mutation_gain`` clamps it internally.
        reads, p_alive = 10.0, 1.0
        try:
            _reads = float(os.environ.get("HEADROOM_NET_COST_EXPECTED_READS", "") or 10.0)
            if not math.isfinite(_reads):
                raise ValueError("non-finite")
            reads = _reads
        except ValueError:
            logger.warning("HEADROOM_NET_COST_EXPECTED_READS malformed; using 10")
        # #856 P3b: an idle-derived override takes precedence over the static
        # env constant. ``net_mutation_gain`` clamps p_alive to [0, 1]
        # internally, but clamp here too so the value logged/branched on below
        # matches what the formula uses.
        idle_derived = p_alive_override is not None
        if p_alive_override is not None:
            p_alive = min(max(p_alive_override, 0.0), 1.0)
        else:
            try:
                _p_alive = float(os.environ.get("HEADROOM_NET_COST_P_ALIVE", "") or 1.0)
                if not math.isfinite(_p_alive):
                    raise ValueError("non-finite")
                p_alive = _p_alive
            except ValueError:
                logger.warning("HEADROOM_NET_COST_P_ALIVE malformed; using 1.0")
        gain = float(
            policy.net_mutation_gain(
                delta_t,
                suffix,
                reads,
                p_alive,
                write_multiplier=write_multiplier,
            )
        )
        allowed = gain > 0.0
        logger.info(
            "NetCostPolicy slot=%d delta_t=%d suffix=%d reads=%.1f p_alive=%.2f "
            "idle_derived=%s gain=%.0f batch_reclaim=%s -> %s",
            slot_idx,
            delta_t,
            suffix,
            reads,
            p_alive,
            idle_derived,
            gain,
            batch_reclaim,
            "mutate" if allowed else "skip",
        )
        if allowed:
            route_counts.setdefault("netcost_allowed", 0)
            route_counts["netcost_allowed"] += 1
            if idle_derived and p_alive < 1.0:
                # Admitted under an idle-decayed P_alive: the cached suffix is
                # near TTL lapse, so its invalidation penalty is discounted.
                # Independent of batch reclaim — both markers may apply.
                route_counts.setdefault("netcost_idle_admitted", 0)
                route_counts["netcost_idle_admitted"] += 1
                transforms_applied.append("router:netcost_idle_compaction")
            if batch_reclaim:
                # Rode a shallower edit's cache-bust for free — telemetry only;
                # the floor is unchanged (this slot is deeper than the floor).
                route_counts.setdefault("netcost_batch_admitted", 0)
                route_counts["netcost_batch_admitted"] += 1
                transforms_applied.append("router:netcost_batch_admit")
            elif batch_state is not None:
                # First/shallower full-S admit — open (or lower) the batch
                # floor so deeper candidates can reclaim against it.
                current = batch_state.get("floor")
                batch_state["floor"] = slot_idx if current is None else min(current, slot_idx)
        else:
            route_counts.setdefault("netcost_skipped", 0)
            route_counts["netcost_skipped"] += 1
            # Bucket the gain into a coarse magnitude band rather than emitting
            # the raw value: a distinct numeric gain per skip would explode the
            # cardinality of any ``transforms_applied`` aggregation. The exact
            # value is still in the INFO log above for debugging.
            transforms_applied.append(f"netcost:skip:{_gain_bucket(gain)}")
        return allowed

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """Apply intelligent routing to messages.

        Args:
            messages: Messages to transform.
            tokenizer: Tokenizer for counting.
            **kwargs: Additional arguments (context).

        Returns:
            TransformResult with routed and compressed messages.
        """
        # Pre-process: Read lifecycle management (stale/superseded detection)
        if self.config.read_lifecycle.enabled:
            from .read_lifecycle import ReadLifecycleManager

            # is None (not truthiness) so falsy test doubles are honored;
            # guarded import keeps read_lifecycle running in stripped builds.
            injected_store = kwargs.get("compression_store")
            if injected_store is None:
                try:
                    from ..cache.compression_store import get_compression_store

                    injected_store = get_compression_store()
                except ImportError:
                    pass

            lifecycle_mgr = ReadLifecycleManager(
                self.config.read_lifecycle,
                compression_store=injected_store,
            )
            lifecycle_result = lifecycle_mgr.apply(
                messages,
                frozen_message_count=kwargs.get("frozen_message_count", 0),
            )
            messages = lifecycle_result.messages
            # lifecycle transforms tracked separately, merged at the end
            lifecycle_transforms = lifecycle_result.transforms_applied
            lifecycle_ccr_hashes = lifecycle_result.ccr_hashes
        else:
            lifecycle_transforms = []
            lifecycle_ccr_hashes = []

        # Runtime overrides from CompressConfig (via kwargs from compress())
        # These override self.config defaults for this call only.
        skip_user = (
            kwargs.get("compress_user_messages") is not True and self.config.skip_user_messages
        )
        skip_system = kwargs.get("compress_system_messages") is not True
        protect_recent = kwargs.get("protect_recent", self.config.protect_recent_code)
        protect_analysis = kwargs.get(
            "protect_analysis_context", self.config.protect_analysis_context
        )
        min_tokens = kwargs.get("min_tokens_to_compress", 50)
        # Cache-safety knobs for content-block (Anthropic-format) handling:
        compress_assistant_text_blocks = kwargs.get(
            "compress_assistant_text_blocks",
            self.config.compress_assistant_text_blocks,
        )
        min_chars_for_block_compression = kwargs.get(
            "min_chars_for_block_compression",
            self.config.min_chars_for_block_compression,
        )
        # Store runtime options on self for access by _route_and_compress_block
        self._runtime_target_ratio: float | None = kwargs.get("target_ratio")
        self._runtime_force_kompress: bool = bool(
            kwargs.get("force_kompress", self.config.force_kompress_all)
        )
        # skip_kompress: run everything EXCEPT the Kompress ML stage this
        # call. Used by the cold-start fast pass so the request-path pass
        # stays sub-second; units routed to Kompress take the same fallback
        # they take when the model isn't ready. Wins over force_kompress.
        self._runtime_skip_kompress: bool = bool(kwargs.get("skip_kompress", False))
        self._runtime_kompress_model: str | None = kwargs.get("kompress_model")
        # F2.2: capture the per-request CompressionPolicy so
        # ``_record_to_toin`` can gate TOIN writes on
        # ``policy.toin_read_only``. ``None`` when the caller didn't
        # pass a policy — ``_record_to_toin`` treats that as "no gate"
        # to preserve pre-F2.2 behaviour for non-proxy callers.
        self._runtime_compression_policy = kwargs.get("compression_policy")
        # Cross-turn dedup recoverability gate. The fold rewrites a repeated
        # span to a bare in-context pointer (``[↑NL same as msg M]``) that names
        # Headroom's internal message index. That reference is only resolvable
        # where the model can locate the original: on the OpenAI
        # chat-completions streaming path (e.g. ``wrap copilot``) no CCR
        # retrieval tool can be injected (the path cannot intercept tool calls)
        # and the client never shows the model numbered messages, so the
        # pointer reads as deleted content and the model retry-loops on the
        # "missing" output. Same recoverability posture as the lossy
        # ``lossy_unrecoverable_skipped`` guard: when the caller reports the
        # path cannot resolve in-context pointers, skip the fold and keep the
        # bytes verbatim. Default True: every path that does not opt out keeps
        # today's behavior.
        dedup_pointers_recoverable = bool(kwargs.get("cross_turn_dedup_recoverable", True))

        tokens_before = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
        context = kwargs.get("context", "")
        hook_biases: dict[int, float] = kwargs.get("biases") or {}

        # Build tool name map for exclusion checking
        tool_name_map = self._build_tool_name_map(messages)

        # Compute excluded tool IDs based on config
        exclude_tools = (
            self.config.exclude_tools
            if self.config.exclude_tools is not None
            else DEFAULT_EXCLUDE_TOOLS
        )
        excluded_tool_ids = {
            tool_id
            for tool_id, name in tool_name_map.items()
            if is_tool_excluded(name, exclude_tools)
        }

        # CCR-retrieve tool IDs, precomputed once (mirrors excluded_tool_ids
        # above, rather than calling is_tool_excluded() per message/block). A
        # headroom_retrieve result IS already-retrieved, original CCR content --
        # recompressing it writes a new <<ccr:hash>> marker the agent can never
        # redeem (unresolvable retrieval loop). is_tool_excluded() (not a bare
        # comparison) because MCP-served tools appear here under their qualified
        # name, e.g. mcp__headroom__headroom_retrieve. Consulted unconditionally,
        # ahead of the age-decay/read-protection-window logic below: a decayed
        # CCR marker is exactly as unredeemable as a fresh one, so this must
        # never fall through to compression the way excluded_tool_ids does.
        # Known, accepted tradeoff: is_tool_excluded()'s alias matching strips
        # ANY mcp__<server>__ prefix before comparing, so a third-party server
        # exposing a tool literally named headroom_retrieve would also match
        # here. Narrowing this to headroom's own server specifically would need
        # a bespoke check inconsistent with how every other excluded-tool entry
        # is matched; given the name is this specific, the collision risk is
        # accepted rather than special-cased.
        ccr_retrieve_tool_ids = {
            tool_id
            for tool_id, name in tool_name_map.items()
            if is_tool_excluded(name, ("headroom_retrieve",))
        }

        # Read protection (HEADROOM_PROTECT_READS=1): for bash-family agents the
        # exclude-by-tool-NAME set above never catches file reads (they are `bash`
        # tool calls whose COMMAND is a cat/sed/head/...). Mark those tool_use_ids so
        # their output is never LOSSY-compressed (the agent needs exact bytes to edit;
        # lossy reads caused re-reads/turn-inflation + resolve loss on SWE-bench).
        # Type-specific by design: grep/test/ls output stays compressible, so the
        # cache-mode delta still compresses whenever the newest turn is NOT a read.
        self._protect_read_tool_ids = set()
        if read_protection_enabled():
            # Use _tool_call_commands (the parsed shell command), NOT
            # _tool_call_args (a compact free-text blob that, for OpenAI-style
            # JSON-string args, is the raw ``{"command": ...}`` JSON — on which
            # _is_read_command always returns False, silently disabling read
            # protection for OpenAI-native harnesses). _tool_call_commands is
            # extracted via _tool_call_command_text, correct for both wire shapes.
            self._protect_read_tool_ids = {
                tid
                for tid in tool_name_map
                if _is_read_command(self._tool_call_commands.get(tid, ""))
            }

        # Read protection — TEXT-BASED shape (shape-agnostic twin of the above).
        # Text-based agents (GPT-5.4/Codex/Cursor backticks) have no tool_use
        # blocks: the command is in the PRECEDING assistant message's fenced
        # block and the observation is a plain user string with no id to match.
        # Detect the producing command by walking back to that assistant turn and
        # mark the observation's message index so it is passed verbatim — so
        # cat/sed/head code reads are protected on ANY model/harness, not just
        # those that emit tool-call/tool_result blocks.
        self._protect_read_msg_indices: set[int] = set()
        if read_protection_enabled():
            for _idx, _m in enumerate(messages):
                if _m.get("role") != "user":
                    continue
                _cmd = ""
                for _j in range(_idx - 1, -1, -1):
                    _rj = messages[_j].get("role")
                    if _rj == "assistant":
                        _cmd = _fenced_shell_command(messages[_j].get("content"))
                        break
                    if _rj == "user":
                        break
                if _cmd and _is_read_command(_cmd):
                    self._protect_read_msg_indices.add(_idx)

        # --- Adaptive parameters based on context pressure ---
        num_messages = len(messages)
        model_limit = kwargs.get("model_limit", 0)

        # Adaptive Read protection: protect a fraction of recent messages
        if self.config.protect_recent_reads_fraction > 0:
            # Scale: at 10 msgs protect 5, at 50 msgs protect 25, at 200 msgs protect 100
            # But cap at a reasonable floor so very short convos still protect everything
            read_protection_window = max(
                4,  # always protect at least last 4 messages
                int(num_messages * self.config.protect_recent_reads_fraction),
            )
        else:
            read_protection_window = num_messages  # 0.0 = protect all (old behavior)
        runtime_read_protection_window = kwargs.get("read_protection_window")
        if (
            runtime_read_protection_window is not None
            and self.config.protect_recent_reads_fraction > 0
        ):
            # A profile-derived window may only narrow protection when the
            # deployment hasn't explicitly opted into "protect everything"
            # (protect_recent_reads_fraction == 0.0, set by --protect-tool-results).
            # See #1374's documented contract: protected tool output must never
            # lossy-compress "regardless of conversation depth" -- a per-request
            # savings-profile kwarg must not silently weaken that.
            read_protection_window = max(0, int(runtime_read_protection_window))

        # Adaptive compression ratio: scale with context pressure
        if model_limit > 0:
            context_pressure = min(1.0, tokens_before / model_limit)
        else:
            context_pressure = 0.5  # default: moderate

        # Linear interpolation between relaxed and aggressive thresholds
        # pressure 0.0 → relaxed, pressure 1.0 → aggressive
        min_ratio = (
            self.config.min_ratio_relaxed
            + (self.config.min_ratio_aggressive - self.config.min_ratio_relaxed) * context_pressure
        )
        # Clamp to [aggressive, relaxed] range
        min_ratio = max(
            self.config.min_ratio_aggressive,
            min(self.config.min_ratio_relaxed, min_ratio),
        )

        # Cache-churn fix: when frozen-decision mode is on, accept/skip
        # decisions use the SAME live ``min_ratio`` gate as flag-off, but once a
        # block is accepted-and-compressed its "compress" verdict is pinned
        # (accept_threshold=1.0) on every later turn. So when rising pressure
        # tightens ``min_ratio`` below the block's ratio — the turn the legacy
        # path would ``move_to_skip`` and bust the provider prefix cache — the
        # pin keeps it compressed instead. This is a pure pin: it preserves a
        # decision the live gate already made; it never loosens the first-sight
        # gate (which would be a silent ratio bet).
        #
        # (Earlier this used a FIXED ``frozen_threshold`` == the ``min_ratio``
        # aggressive floor (0.65), which made the pin dead code — a block only
        # froze when ``cached_ratio < 0.65 <= min_ratio``, one the legacy gate
        # accepts anyway, so the pin never changed an outcome and the flag only
        # suppressed borderline compression. Verified inert by A/B 2026-06-21.)
        #
        # Default off ⇒ the pin branch is never taken and the legacy
        # ``min_ratio`` path is byte-for-byte untouched.
        freeze_decision = self._freeze_block_decision_enabled()
        model_ready = self._kompress_model_ready() if freeze_decision else True

        if context_pressure > 0.3:
            logger.debug(
                "content_router adaptive: pressure=%.2f, min_ratio=%.2f, "
                "read_protect_window=%d/%d msgs",
                context_pressure,
                min_ratio,
                read_protection_window,
                num_messages,
            )

        transformed_messages: list[dict[str, Any]] = []
        transforms_applied: list[str] = []
        warnings: list[str] = []
        compressor_timing: dict[str, float] = {}  # strategy → cumulative ms

        # Routing reason counters for summary logging
        route_counts: dict[str, int] = {
            "excluded_tool": 0,
            "ccr_retrieve": 0,
            "user_msg": 0,
            "small": 0,
            "recent_code": 0,
            "analysis_ctx": 0,
            "ratio_too_high": 0,
            "non_string": 0,
            "content_blocks": 0,
        }
        compressed_details: list[str] = []  # e.g. ["code_aware:0.72", "kompress:0.65"]

        # Check for analysis intent in the most recent user message
        analysis_intent = False
        if self.config.protect_analysis_context:
            analysis_intent = self._detect_analysis_intent(messages)

        frozen_message_count = kwargs.get("frozen_message_count", 0)

        # ------------------------------------------------------------------
        # Two-pass parallel compression.
        #
        # Pass 1 (sequential): categorise every message — frozen, protected,
        #   cached, small, etc. are resolved immediately.  Cache-miss messages
        #   that need full compression are collected into *pending_tasks*.
        #
        # Pass 2 (parallel): all cache-miss compressions run concurrently in
        #   a thread pool.  Each self.compress() call is independent.
        #
        # Pass 3 (sequential): results are stitched back into message order,
        #   caches updated, and counters incremented.
        # ------------------------------------------------------------------

        # Pre-allocate result slots — None means "pending compression".
        result_slots: list[dict[str, Any] | None] = [None] * num_messages

        # #856 P2 (flag-gated, default off): net-cost mutation gate. Suffix
        # token sums are precomputed once (reverse cumulative) so each
        # candidate's S lookup is O(1). v1 estimator per the issue: S is the
        # token total of every message after the candidate.
        netcost_enabled = os.environ.get("HEADROOM_NET_COST_POLICY") == "1"
        netcost_suffix_tokens: list[int] = []
        # #856 P3a: shared batch-reclaim state for this request. ``floor`` is
        # the shallowest slot admitted as a net-positive mutation; once set,
        # deeper candidates charge S=0 (their cache-bust is already paid).
        netcost_batch_state: dict[str, int | None] = {"floor": None}
        # #856 P3b (idle-timer compaction): if the caller supplies how long the
        # session has been idle, decay P_alive from it once per request and
        # pass it to the gate. Absent/malformed → None → the gate keeps the P2
        # env-constant behaviour. Derived once here (not per slot) — idle is a
        # per-request property, like frozen_message_count.
        netcost_p_alive_override: float | None = None
        netcost_write_multiplier: float | None = None
        if netcost_enabled:
            # Prefer the authoritative per-request prompt-cache TTL when the
            # caller has one; retain the env setting for other providers and
            # legacy callers.
            request_ttl = kwargs.get("cache_ttl_seconds")
            if request_ttl is None:
                netcost_ttl = _net_cost_cache_ttl_seconds()
            else:
                try:
                    netcost_ttl = float(request_ttl)
                except (TypeError, ValueError):
                    netcost_ttl = _net_cost_cache_ttl_seconds()
                if not math.isfinite(netcost_ttl) or netcost_ttl <= 0.0:
                    netcost_ttl = _net_cost_cache_ttl_seconds()
            netcost_write_multiplier = cache_write_multiplier_for_ttl(netcost_ttl)
            netcost_suffix_tokens = [0] * (num_messages + 1)
            for j in range(num_messages - 1, -1, -1):
                netcost_suffix_tokens[j] = netcost_suffix_tokens[j + 1] + _netcost_message_tokens(
                    messages[j], tokenizer
                )
            idle_seconds = kwargs.get("idle_seconds")
            if idle_seconds is not None:
                try:
                    idle_f = float(idle_seconds)
                except (TypeError, ValueError):
                    idle_f = None
                if idle_f is not None and math.isfinite(idle_f) and idle_f >= 0.0:
                    netcost_p_alive_override = max(0.0, 1.0 - idle_f / netcost_ttl)

        # Tasks: list of (slot_index, content, context, bias, content_key,
        # enforce_reversibility, precomputed_detection). The detection is the one
        # already computed in the routing loop below, threaded through so a
        # cache-miss message is not re-detected inside compress().
        _PendingTask = tuple[int, str, str, float, int, bool, "DetectionResult | None"]
        pending_tasks: list[_PendingTask] = []

        # #856 P2b (flag-gated, default off): net-cost frozen-floor unlock.
        # Without the flag, every message in the provider's prefix cache
        # (index < frozen_message_count) is unconditionally skipped — mutating
        # one trades a 90% read discount for a 25% write penalty (Anthropic).
        # That binary floor leaves money on the table: a 50K-token stale tool
        # dump with only a 10K cached suffix after it pays for itself many
        # times over. With HEADROOM_NET_COST_POLICY=1 a *string-content*
        # frozen message instead falls through to the normal candidate
        # pipeline, where the P2 break-even gate (_net_cost_allows) decides
        # per candidate: its S is the full invalidated suffix after the slot,
        # so the deep edit proceeds only when ΔT·(w+r(R-1)) still beats the
        # cache-bust penalty. Block-list and non-string frozen content stay
        # frozen — the gate is wired into the string and parallel-merge paths
        # only, and the per-block cache_control contract in
        # _process_content_blocks is not net-cost aware, so opening them here
        # would mutate cached blocks ungated.
        frozen_unlock_slots: set[int] = set()
        for i, message in enumerate(messages):
            if i < frozen_message_count:
                if netcost_enabled and isinstance(message.get("content", ""), str):
                    # Defer to the break-even gate below instead of skipping.
                    frozen_unlock_slots.add(i)
                    route_counts.setdefault("netcost_frozen_considered", 0)
                    route_counts["netcost_frozen_considered"] += 1
                else:
                    # Frozen — byte-identical to preserve the prefix cache.
                    result_slots[i] = message
                    continue

            role = message.get("role", "")
            content = message.get("content", "")
            bias = 1.0  # Default bias, may be overridden for tool messages

            messages_from_end = num_messages - i

            # Handle list content (Anthropic format with content blocks)
            if isinstance(content, list):
                transformed_message = self._process_content_blocks(
                    message,
                    content,
                    context,
                    transforms_applied,
                    excluded_tool_ids,
                    ccr_retrieve_tool_ids,
                    tool_name_map=tool_name_map,
                    route_counts=route_counts,
                    compressed_details=compressed_details,
                    min_ratio=min_ratio,
                    read_protection_window=read_protection_window,
                    messages_from_end=messages_from_end,
                    compressor_timing=compressor_timing,
                    min_chars=min_chars_for_block_compression,
                    skip_user=skip_user,
                    skip_system=skip_system,
                    compress_assistant_text_blocks=compress_assistant_text_blocks,
                )
                result_slots[i] = transformed_message
                route_counts["content_blocks"] += 1
                continue

            # Skip non-string content (other types)
            if not isinstance(content, str):
                result_slots[i] = message
                route_counts["non_string"] += 1
                continue

            # A headroom_retrieve result IS already-retrieved, original CCR content --
            # recompressing it writes a new <<ccr:hash>> marker the agent can never
            # redeem (unresolvable retrieval loop). Covers role:"tool" (id-keyed via
            # tool_call_id -> ccr_retrieve_tool_ids, precomputed above) and legacy
            # role:"function" (that shape carries no call id -- the tool name is on
            # the message itself via "name", per OpenAI's pre-parallel-tool-calls API).
            tool_call_id = message.get("tool_call_id", "") if role in ("tool", "function") else ""
            if role in ("tool", "function") and (
                tool_call_id in ccr_retrieve_tool_ids
                or (
                    role == "function"
                    and is_tool_excluded(message.get("name", ""), ("headroom_retrieve",))
                )
            ):
                result_slots[i] = message
                transforms_applied.append("router:excluded:ccr_retrieve")
                route_counts["ccr_retrieve"] += 1
                continue

            # Skip OpenAI-style tool messages for excluded tools
            # BUT: allow compression of old excluded-tool outputs beyond the
            # adaptive protection window (age-based decay).
            if role == "tool":
                if tool_call_id in excluded_tool_ids:
                    tool_name = tool_name_map.get(tool_call_id, "")
                    if tool_name and is_tool_excluded(tool_name, DEFAULT_VERBATIM_EXCLUDE_TOOLS):
                        result_slots[i] = message
                        transforms_applied.append("router:excluded:tool")
                        route_counts["excluded_tool"] += 1
                        continue
                    if messages_from_end <= read_protection_window:
                        # Protected from lossy compression — but grep/log/json
                        # output can still be losslessly compacted, UNLESS this is
                        # a file read: every fold rewrites the bytes the model
                        # copies into `Edit(old_string=…)`, and a missed edit costs
                        # a whole turn (see DEFAULT_BYTE_EXACT_EXCLUDE_TOOLS).
                        compacted = (
                            None
                            if is_tool_excluded(tool_name, DEFAULT_BYTE_EXACT_EXCLUDE_TOOLS)
                            else self._lossless_compact_excluded(content)
                        )
                        if compacted is not None:
                            folded, kind = compacted
                            result_slots[i] = {**message, "content": folded}
                            transforms_applied.append(f"router:excluded:lossless_{kind}")
                            route_counts["excluded_tool_lossless"] = (
                                route_counts.get("excluded_tool_lossless", 0) + 1
                            )
                            continue
                        # Recent — protect as before
                        result_slots[i] = message
                        transforms_applied.append("router:excluded:tool")
                        route_counts["excluded_tool"] += 1
                        continue
                    # Old excluded-tool output — fall through to compression
                    # (the LLM is unlikely to need exact content from this far back,
                    # and CCR provides retrieval if it does)
                # Look up tool-specific compression bias for OpenAI tool messages
                tool_name = tool_name_map.get(tool_call_id, "")
                bias = self._get_tool_bias(tool_name) if tool_name else 1.0

                # Bash-search lossless pre-empt: a read-only search (grep/rg/git
                # grep) run via a shell tool yields byte-losslessly foldable
                # output. Fold it instead of the lossy strategy path.
                bash_folded = self._bash_search_fold(tool_name, tool_call_id, content)
                if bash_folded is not None:
                    result_slots[i] = {**message, "content": bash_folded}
                    transforms_applied.append("router:bash:lossless_search")
                    route_counts["bash_lossless_search"] = (
                        route_counts.get("bash_lossless_search", 0) + 1
                    )
                    continue

            # Read protection (ROLE / SHAPE-AGNOSTIC). An observation produced by
            # a file read command (cat/sed/head/…) is passed VERBATIM so the agent
            # keeps exact bytes to patch — regardless of how THIS harness labels
            # it. The SAME operation surfaces under different roles/shapes across
            # harnesses (Anthropic tool_use, OpenAI/Kimi `role:tool`, text-harness
            # `role:user` string), so we key off the OUTCOME — "a read command
            # produced code output" — not the role. Link via the observation's
            # tool_call_id/tool_use_id (tool-based) OR the preceding fenced
            # command's message index (text-based); a given harness populates
            # exactly one, so ORing them is shape-agnostic and collision-free.
            # (Anthropic tool_result BLOCKS carry list content and are protected
            # in the block path; this covers STRING-content observations.)
            if role in ("user", "tool", "function"):
                _tcid = message.get("tool_call_id") or message.get("tool_use_id") or ""
                _is_read_obs = _tcid in getattr(self, "_protect_read_tool_ids", ()) or i in getattr(
                    self, "_protect_read_msg_indices", ()
                )
                if _is_read_obs and _read_output_should_be_protected(content):
                    _exp = self._experimental_compress_read(content, context)
                    if _exp is not None:
                        result_slots[i] = {**message, "content": _exp}
                        transforms_applied.append("router:read_kompress_exp")
                        route_counts["read_kompress_exp"] = (
                            route_counts.get("read_kompress_exp", 0) + 1
                        )
                        continue
                    result_slots[i] = message
                    transforms_applied.append("router:read_protected")
                    route_counts.setdefault("read_protected", 0)
                    route_counts["read_protected"] += 1
                    continue

            # Protection 1: Never compress user messages (unless overridden)
            if skip_user and role == "user":
                result_slots[i] = message
                transforms_applied.append("router:protected:user_message")
                route_counts["user_msg"] += 1
                continue

            # Protection 1b: Never compress system/developer messages unless
            # explicitly opted in. These are cache-hot instruction bytes.
            if skip_system and role in {"system", "developer"}:
                result_slots[i] = message
                transforms_applied.append(f"router:protected:{role}_message")
                route_counts.setdefault("system_msg", 0)
                route_counts["system_msg"] += 1
                continue

            if not content or tokenizer.count_text(content) < min_tokens:
                # Skip small content
                result_slots[i] = message
                route_counts["small"] += 1
                continue

            # Protection: failed tool calls / error outputs stay verbatim
            # (issue #847). The model needs exact tracebacks to recover.
            # Strong (>=2 distinct indicators) match only — a single
            # keyword false-positives on benign outputs that mention
            # errors. Above the size cap, fall through — LogCompressor
            # preserves error lines in big logs.
            if (
                self.config.protect_error_outputs
                and role == "tool"
                and len(content) <= self.config.error_protection_max_chars
                and content_has_strong_error_indicators(content)
            ):
                result_slots[i] = message
                transforms_applied.append("router:protected:error_output")
                route_counts.setdefault("error_protected", 0)
                route_counts["error_protected"] += 1
                continue

            # Detect content type for protection decisions. Even when the
            # runtime strategy is forced to Kompress, keep code-protection
            # checks but use the lightweight regex detector instead of the
            # full router chain.
            force_kompress = bool(getattr(self, "_runtime_force_kompress", False))
            detection = (
                _regex_detect_content_type(content) if force_kompress else _detect_content(content)
            )
            is_code = detection.content_type == ContentType.SOURCE_CODE

            # Protection 2: Don't compress recent CODE
            messages_from_end = num_messages - i
            if protect_recent > 0 and messages_from_end <= protect_recent and is_code:
                result_slots[i] = message
                transforms_applied.append("router:protected:recent_code")
                route_counts["recent_code"] += 1
                continue

            # Protection 3: Don't compress CODE when analysis intent detected
            if protect_analysis and analysis_intent and is_code:
                result_slots[i] = message
                transforms_applied.append("router:protected:analysis_context")
                route_counts["analysis_ctx"] += 1
                continue

            # Compression pinning: if this message was already compressed
            # (contains a CCR retrieval marker), skip recompression.
            # Recompressing would change byte content and break provider
            # prefix caching with no meaningful further reduction.
            if _is_already_compressed(content):
                result_slots[i] = message
                route_counts.setdefault("already_compressed", 0)
                route_counts["already_compressed"] += 1
                continue

            # Route and compress based on content detection
            # Merge tool-specific bias with hook-provided bias (multiplicative)
            msg_bias = bias if role == "tool" else 1.0
            if i in hook_biases:
                msg_bias *= hook_biases[i]

            # Two-tier compression cache.
            # Tier 1 (skip): known won't-compress → instant skip.
            # Tier 2 (result): known compresses → reuse compressed text.
            # Key on the runtime target_ratio too: the same content compressed at
            # a different ratio is a different result, so it must not alias.
            content_key = hash((content, getattr(self, "_runtime_target_ratio", None)))
            # Tool ground truth is gated against lossy-unrecoverable results below
            # (#1307). Partition its cache namespace so a gated tool entry is never
            # served from — or poisons — an ungated entry for byte-identical content.
            enforce_reversibility = role == "tool"
            if enforce_reversibility:
                content_key = hash((content_key, True))

            # Tier 1: skip set — instant rejection
            if self._cache.is_skipped(content_key):
                result_slots[i] = message
                route_counts["ratio_too_high"] += 1
                route_counts.setdefault("cache_hit", 0)
                route_counts["cache_hit"] += 1
                continue

            # Tier 2: result cache — reuse compressed output
            cached = self._cache.get(content_key)
            if cached is not None:
                cached_compressed, cached_ratio, cached_strategy = cached
                # Cache-churn fix: if a verdict was frozen for this block on a
                # prior turn, REUSE it and bypass the per-turn ``min_ratio``
                # re-check + ``move_to_skip`` downgrade. The frozen verdict is
                # only ever "compress" here (a "skip" verdict never warms the
                # result cache), so this just pins the accept decision against
                # ratio drift.
                frozen_compress = freeze_decision and self._get_frozen_verdict(content_key) is True
                if frozen_compress:
                    # Pin: a frozen "compress" verdict always re-accepts,
                    # overriding the per-turn min_ratio re-check below.
                    accept_threshold = 1.0
                else:
                    # Pre-freeze / unfrozen: identical to the legacy gate. The
                    # freeze only PINS past accepts; it never loosens the
                    # first-sighting decision (that would be a silent ratio bet).
                    accept_threshold = min_ratio
                # Re-check ratio against the active threshold (shifts with
                # context pressure unless pinned by a frozen verdict).
                if cached_ratio < accept_threshold:
                    if netcost_enabled and not self._net_cost_allows(
                        slot_idx=i,
                        original_tokens=tokenizer.count_text(content),
                        compressed_tokens=tokenizer.count_text(cached_compressed),
                        suffix_tokens=netcost_suffix_tokens,
                        route_counts=route_counts,
                        transforms_applied=transforms_applied,
                        batch_state=netcost_batch_state,
                        p_alive_override=netcost_p_alive_override,
                        write_multiplier=netcost_write_multiplier,
                    ):
                        # Net-cost gate: mutation would cost more in cache
                        # invalidation than it saves — leave untouched.
                        result_slots[i] = message
                    else:
                        result_slots[i] = {**message, "content": cached_compressed}
                        transforms_applied.append(f"router:{cached_strategy}:{cached_ratio:.2f}")
                        compressed_details.append(f"{cached_strategy}:{cached_ratio:.2f}")
                        # Freeze the "compress" verdict so future turns skip the
                        # min_ratio re-check above and never downgrade it.
                        if freeze_decision:
                            self._record_frozen_verdict(content_key, True)
                        # Pin attribution: count a bust avoided only when the
                        # block actually stayed compressed (past the net-cost
                        # gate) AND the frozen verdict overrode a tightened
                        # min_ratio that flag-off would have moved to skip.
                        if frozen_compress and cached_ratio >= min_ratio:
                            self._record_freeze_pin(content, cached_ratio)
                        if i in frozen_unlock_slots:
                            transforms_applied.append("router:netcost_frozen_unlock")
                            route_counts.setdefault("netcost_frozen_unlocked", 0)
                            route_counts["netcost_frozen_unlocked"] += 1
                else:
                    # Threshold tightened — no longer qualifies. Move to skip.
                    # (Unreachable when the verdict is frozen-compress: the
                    # accept_threshold is pinned to 1.0 above.)
                    self._cache.move_to_skip(content_key)
                    result_slots[i] = message
                    route_counts["ratio_too_high"] += 1
                route_counts.setdefault("cache_hit", 0)
                route_counts["cache_hit"] += 1
                continue

            # Cache miss — defer to parallel compression pass. Carry the
            # detection already computed above so compress() need not re-run the
            # native detection chain on this same content. Only the full
            # detection is reused; under force_kompress the loop used the
            # lightweight regex detector and compress() skips detection entirely.
            route_counts.setdefault("cache_miss", 0)
            route_counts["cache_miss"] += 1
            pending_tasks.append(
                (
                    i,
                    content,
                    context,
                    msg_bias,
                    content_key,
                    enforce_reversibility,
                    None if force_kompress else detection,
                )
            )

        # --- Pass 2: Parallel compression of all cache-miss messages ---
        if pending_tasks:
            max_workers = min(
                len(pending_tasks), int(os.environ.get("HEADROOM_COMPRESS_WORKERS", "4"))
            )
            t_parallel_start = time.perf_counter()

            if max_workers <= 1 or len(pending_tasks) == 1:
                # Single task or parallelism disabled — compress inline
                task_results = []
                for _, task_content, task_ctx, task_bias, _, _, task_detection in pending_tasks:
                    t0 = time.perf_counter()
                    deadline_s = _compression_deadline_seconds() if len(pending_tasks) == 1 else 0.0
                    if deadline_s:
                        box: dict[str, Any] = {}

                        def _run(
                            _box: dict[str, Any] = box,
                            _content: str = task_content,
                            _context: str = task_ctx,
                            _bias: float = task_bias,
                            _detection: DetectionResult | None = task_detection,
                        ) -> None:
                            try:
                                _box["result"] = self.compress(
                                    _content,
                                    context=_context,
                                    bias=_bias,
                                    precomputed_detection=_detection,
                                )
                            except BaseException as exc:  # noqa: BLE001
                                _box["error"] = exc

                        # ponytail: daemon watchdog cannot stop native GIL holds; native layer owns that fix.
                        worker = threading.Thread(
                            target=_run, name="headroom-single-compress-watchdog", daemon=True
                        )
                        worker.start()
                        worker.join(deadline_s)
                        if worker.is_alive():
                            logger.warning(
                                "ContentRouter single-cache-miss compression exceeded %.1fs; "
                                "failing open via PASSTHROUGH",
                                deadline_s,
                            )
                            r = RouterCompressionResult(
                                compressed=task_content,
                                original=task_content,
                                strategy_used=CompressionStrategy.PASSTHROUGH,
                            )
                        elif "error" in box:
                            raise box["error"]
                        else:
                            r = box["result"]
                    else:
                        r = self.compress(
                            task_content,
                            context=task_ctx,
                            bias=task_bias,
                            precomputed_detection=task_detection,
                        )
                    compress_ms = (time.perf_counter() - t0) * 1000
                    task_results.append((r, compress_ms))
            else:
                # Parallel compression via thread pool
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for _, task_content, task_ctx, task_bias, _, _, task_detection in pending_tasks:
                        futures.append(
                            executor.submit(
                                self._timed_compress,
                                task_content,
                                task_ctx,
                                task_bias,
                                task_detection,
                            )
                        )
                    task_results = [f.result() for f in futures]

            parallel_ms = (time.perf_counter() - t_parallel_start) * 1000
            compressor_timing["parallel_compress_total"] = parallel_ms

            # --- Pass 3: Merge results back (sequential, updates caches) ---
            for (slot_idx, task_content, _, _, content_key, enforce_rev, _), (
                result,
                compress_ms,
            ) in zip(pending_tasks, task_results):
                message = messages[slot_idx]
                strategy_key = f"compressor:{result.strategy_used.value}"
                compressor_timing[strategy_key] = (
                    compressor_timing.get(strategy_key, 0.0) + compress_ms
                )

                # Lossless folds (search/log/diff via compact_lossless) shrink by
                # collapsing repeated path prefixes, but the gate's default ratio
                # is word count — which barely moves (a heading line can push it
                # >1.0), discarding a free, recoverable win. Measure lossless
                # results by REAL TOKEN count (what actually costs money/context),
                # not words and not bytes: accept iff tokens genuinely drop. The
                # excluded/bash paths already bypass this gate; this fixes the
                # main strategy dispatch.
                is_lossless = any(
                    s.startswith("lossless_")
                    for s in (getattr(result, "strategy_chain", None) or [])
                )
                if is_lossless and getattr(result, "original", None):
                    orig_tok = tokenizer.count_text(result.original)
                    accept_ratio = (
                        tokenizer.count_text(result.compressed) / orig_tok if orig_tok else 1.0
                    )
                else:
                    accept_ratio = result.compression_ratio
                if accept_ratio < min_ratio:
                    # tool ground truth must stay reversible — a lossy summarizer
                    # (kompress/text/code) that emitted no CCR retrieve marker is
                    # unrecoverable, so the agent would act on a fabricated summary
                    # (#1307). Keep the original verbatim instead.
                    if (
                        enforce_rev
                        and self.config.ccr_inject_marker
                        and result.strategy_used in self.LOSSY_UNMARKED_STRATEGIES
                        and not CCR_RETRIEVAL_MARKER_RE.search(result.compressed)
                    ):
                        self._cache.mark_skip(content_key)
                        result_slots[slot_idx] = message
                        route_counts["lossy_unrecoverable_skipped"] = (
                            route_counts.get("lossy_unrecoverable_skipped", 0) + 1
                        )
                        continue
                    # Compressed — store in result cache. The cache is still
                    # warmed when the net-cost gate blocks the slot: the
                    # gate's verdict is contextual (suffix size), the
                    # compression result is not.
                    self._cache.put(
                        content_key,
                        result.compressed,
                        accept_ratio,
                        result.strategy_used.value,
                    )
                    # Freeze "compress" so the cache-hit path above never
                    # re-applies a tighter min_ratio to this block.
                    if freeze_decision:
                        self._record_frozen_verdict(content_key, True)
                    if netcost_enabled and not self._net_cost_allows(
                        slot_idx=slot_idx,
                        original_tokens=tokenizer.count_text(task_content),
                        compressed_tokens=tokenizer.count_text(result.compressed),
                        suffix_tokens=netcost_suffix_tokens,
                        route_counts=route_counts,
                        transforms_applied=transforms_applied,
                        batch_state=netcost_batch_state,
                        p_alive_override=netcost_p_alive_override,
                        write_multiplier=netcost_write_multiplier,
                    ):
                        result_slots[slot_idx] = message
                        continue
                    result_slots[slot_idx] = {**message, "content": result.compressed}
                    transforms_applied.append(
                        f"router:{result.strategy_used.value}:{accept_ratio:.2f}"
                    )
                    compressed_details.append(f"{result.strategy_used.value}:{accept_ratio:.2f}")
                    if slot_idx in frozen_unlock_slots:
                        transforms_applied.append("router:netcost_frozen_unlock")
                        route_counts.setdefault("netcost_frozen_unlocked", 0)
                        route_counts["netcost_frozen_unlocked"] += 1
                else:
                    # Didn't compress — add to skip set
                    self._cache.mark_skip(content_key)
                    result_slots[slot_idx] = message
                    route_counts["ratio_too_high"] += 1
                    # Caveat (1): only freeze a "skip" verdict when the ML model
                    # is actually ready. A passthrough caused purely by a still-
                    # loading ModernBERT must stay re-evaluable on later turns,
                    # so we do NOT freeze it here (the byte-result skip cache
                    # entry can still expire/refresh per the existing TTL).
                    if freeze_decision and model_ready:
                        self._record_frozen_verdict(content_key, False)

        # Build final message list from slots
        transformed_messages = [m for m in result_slots if m is not None]

        # Cross-turn (whole-conversation) verbatim de-dup, over the FINAL block
        # forms, so it works in both modes: in lossless mode it references
        # verbatim/byte-folded content; in CCR mode it references the earlier
        # block's kompressed-but-CCR-recoverable form (deterministic — the CCR
        # hash is content-derived — so per-block forms are stable and the rewrite
        # stays prefix-monotonic → no prompt-cache bust). It never adds loss: the
        # later duplicate would carry the same (recoverable) form anyway; dedup
        # just points to the earlier copy instead of repeating it. Frozen +
        # cache_control blocks are reference targets only (never rewritten).
        if self._cross_turn_dedup_enabled and dedup_pointers_recoverable:
            transformed_messages = self._cross_turn_dedup_messages(
                transformed_messages, frozen_message_count, transforms_applied, route_counts
            )

        tokens_after = sum(
            tokenizer.count_text(str(m.get("content", ""))) for m in transformed_messages
        )

        # Log routing summary
        parts = []
        if compressed_details:
            parts.append(f"{len(compressed_details)} compressed ({', '.join(compressed_details)})")
        if route_counts["excluded_tool"]:
            parts.append(f"{route_counts['excluded_tool']} excluded (Read/Glob)")
        if route_counts["user_msg"]:
            parts.append(f"{route_counts['user_msg']} skipped (user)")
        if route_counts["small"]:
            # Report the thresholds actually in force, not a literal. This line
            # used to read "skipped (<50 words)" unconditionally: wrong number
            # (the message gate is `min_tokens`, which profiles set anywhere from
            # 10 to 250), wrong unit (tokens and characters, never words), and it
            # merged two different gates under one label. Operators read it as
            # evidence of a mis-set threshold and tuned the wrong knob.
            parts.append(
                f"{route_counts['small']} skipped "
                f"(<{min_tokens} tok msg / <{min_chars_for_block_compression} chars block)"
            )
        if route_counts["recent_code"]:
            parts.append(f"{route_counts['recent_code']} protected (recent code)")
        if route_counts["analysis_ctx"]:
            parts.append(f"{route_counts['analysis_ctx']} protected (analysis ctx)")
        if route_counts.get("already_compressed"):
            parts.append(f"{route_counts['already_compressed']} pinned (already compressed)")
        if route_counts.get("error_protected"):
            parts.append(f"{route_counts['error_protected']} protected (error output)")
        if route_counts["ratio_too_high"]:
            parts.append(f"{route_counts['ratio_too_high']} unchanged (ratio>={min_ratio:.2f})")
        if route_counts["content_blocks"]:
            parts.append(f"{route_counts['content_blocks']} content-block msgs")
        if route_counts["non_string"]:
            parts.append(f"{route_counts['non_string']} non-string")
        if route_counts.get("cache_hit"):
            parts.append(f"{route_counts['cache_hit']} cache hits")
        if route_counts.get("cache_miss"):
            parts.append(f"{route_counts['cache_miss']} cache misses")
        if route_counts.get("netcost_batch_admitted"):
            parts.append(f"{route_counts['netcost_batch_admitted']} netcost batch-admitted")
        if route_counts.get("netcost_idle_admitted"):
            parts.append(f"{route_counts['netcost_idle_admitted']} netcost idle-admitted")
        cs = self._cache.stats
        if cs["cache_size"] > 0 or cs["cache_skip_size"] > 0:
            parts.append(
                f"cache[{cs['cache_size']} results, {cs['cache_skip_size']} skips, "
                f"{cs['cache_avg_lookup_ns']:.0f}ns avg]"
            )
        if parts:
            logger.info(
                "content_router: %d msgs — %s",
                num_messages,
                ", ".join(parts),
            )

        # Per-request routing visibility (grep `[router] route_counts`): how many
        # messages/blocks hit each route this request — skip reasons (small,
        # user_msg, non_string, recent_code, analysis_ctx, content_blocks,
        # excluded_tool, read_protected, error_protected, already_compressed, …)
        # plus successful compressions. Makes "what is Headroom missing?" answerable
        # per provider shape (e.g. OpenAI plain-string user obs vs Anthropic
        # tool_result blocks) directly from a run's logs. INFO so it's on by default.
        _nonzero = {k: v for k, v in route_counts.items() if v}
        logger.info(
            "[router] route_counts=%s compressed=%d frozen=%d msgs=%d",
            _nonzero,
            len(compressed_details),
            frozen_message_count,
            num_messages,
        )

        # Forward route_counts to the observer so `/stats` can surface a
        # session-level protection breakdown (issue #454). The observer
        # may not implement this method on older versions; ignore
        # AttributeError so a non-conforming observer doesn't poison
        # routing.
        if self._observer is not None and route_counts:
            try:
                self._observer.record_router_route_counts(route_counts)
            except AttributeError:
                pass
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Router observer raised (non-fatal): %s", e)

        all_transforms = lifecycle_transforms + transforms_applied
        return TransformResult(
            messages=transformed_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=all_transforms if all_transforms else ["router:noop"],
            markers_inserted=lifecycle_ccr_hashes,
            warnings=warnings,
            timing=compressor_timing,
        )

    def _lossless_compact_excluded(self, content: Any) -> tuple[str, str] | None:
        """Information-preserving compaction for a protected (excluded) tool output.

        Excluded tools are kept out of *lossy* compression for accuracy. This
        applies only reversible/data-preserving transforms, dispatched by shape:

        * SEARCH (grep ``path:line:content``) -> ripgrep --heading fold.
          Byte-recoverable (``search_unheading`` reproduces the original). Gated
          on the dedicated ``_try_detect_search`` — the general classifier calls
          grep-over-code SOURCE_CODE and would wrongly reject it.
        * LOG (build/test/app logs) -> ANSI strip + run-collapse. Recoverable
          modulo non-semantic ANSI color (``expand_runs`` restores the lines).
        * JSON -> whitespace-minify. **Data-lossless** (``json.loads`` equals the
          original object) — same information, fewer tokens. NOT byte-exact.

        "Information-preserving" is the guarantee, and it is weaker than it looks
        from in here: recoverability is OURS, but the string the model copies into
        the next ``Edit(old_string=…)`` is the one we SHOWED it. All three folds
        show something the file does not contain — minified JSON, ``... (repeated
        N times)``, a hoisted path heading — so all three break a read-then-edit.
        Callers must therefore NOT route a file read here: every call site gates
        on ``DEFAULT_BYTE_EXACT_EXCLUDE_TOOLS`` (and the stricter
        ``DEFAULT_VERBATIM_EXCLUDE_TOOLS``, which covers Copilot's ``view``) first.
        This function takes only ``content`` and cannot make that call itself —
        which is also why the pluggable provider below, being content-only, is
        never handed a read.

        Returns ``(compacted, kind)`` when a recognized shape actually shrinks,
        else ``None``. Source code and glob path-lists match nothing -> verbatim.
        Never raises.
        """
        if not isinstance(content, str):
            return None
        provider = get_lossless_provider()
        if provider is not None:
            try:
                # A registered provider is authoritative for excluded-tool
                # compaction; fall back to the built-in folds only if it raises.
                return provider(content)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "lossless provider failed; using built-in compaction",
                    exc_info=True,
                )
        if len(content) < 200:
            return None
        try:
            from .lossless_compaction import compact_lossless

            det = _try_detect_search(content)
            if det is not None and det.content_type is ContentType.SEARCH_RESULTS:
                out = compact_lossless(content, "search")
                return (out, "search") if len(out) < len(content) else None
            if _try_detect_log(content) is not None:
                out = compact_lossless(content, "log")
                return (out, "log") if len(out) < len(content) else None
            minified = self._minify_json_data_lossless(content)
            return (minified, "json") if minified is not None else None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _minify_json_data_lossless(content: str) -> str | None:
        """Whitespace-minify a complete JSON value: data-preserving, not byte-exact.

        The ``json.loads`` parse is the data-equality guarantee (identical
        object). Returns the minified form only when the content is a JSON
        object/array and the result is smaller; ``None`` otherwise (source code,
        partial/non-JSON).
        """
        stripped = content.strip()
        if not stripped or stripped[0] not in "{[":
            return None
        obj = json.loads(stripped)
        minified = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        return minified if len(minified) < len(content) else None

    def _bash_search_fold(self, tool_name: str, tool_id: str, content: Any) -> str | None:
        """Byte-lossless fold for a read-only search run through a shell tool.

        ``bash`` is not excluded, so its output normally takes the lossy strategy
        path. But when the command is a read-only search (grep/rg/git grep/…),
        its output is byte-losslessly foldable — so fold it (the same guarantee
        excluded Grep gets) instead of lossy-compressing. The command whitelist
        is only a *gate to attempt*: ``compact_lossless`` verifies reversibility
        and returns the input unchanged when it can't safely shrink, so a mis-
        gated command (``grep -l`` path-lists, ``grep -c`` counts) simply falls
        through to the normal path with no accuracy risk.

        Returns the folded text (smaller, recoverable) or ``None`` to fall through.
        """
        if not isinstance(content, str) or len(content) < 200:
            return None
        if tool_name.lower() not in self.config.bash_tool_names:
            return None
        command = self._tool_call_commands.get(tool_id, "")
        if not command or not _bash_command_is_search(command, self.config.bash_search_commands):
            return None
        try:
            from .lossless_compaction import compact_lossless

            folded = compact_lossless(content, "search")
        except Exception:  # noqa: BLE001
            return None
        return folded if len(folded) < len(content) else None

    def _get_tool_bias(self, tool_name: str) -> float:
        """Look up compression bias for a tool name.

        Checks user-configured profiles first, then DEFAULT_TOOL_PROFILES.
        Returns 1.0 (moderate) if no profile is configured.
        """
        from ..config import DEFAULT_TOOL_PROFILES

        # Check user-configured profiles
        if self.config.tool_profiles:
            profile = self.config.tool_profiles.get(tool_name)
            if profile:
                return float(profile.bias)

        # Check default profiles
        profile = DEFAULT_TOOL_PROFILES.get(tool_name)
        if profile:
            return profile.bias

        return 1.0  # Default: moderate

    def _cross_turn_dedup_messages(
        self,
        messages: list[dict[str, Any]],
        frozen_message_count: int,
        transforms_applied: list[str],
        route_counts: dict[str, int] | None,
    ) -> list[dict[str, Any]]:
        """Whole-conversation verbatim de-dup pass (cache-safe, information-lossless).

        Runs AFTER per-block compression, over the final message forms: a span in
        a later tool output that appeared verbatim in an earlier tool output is
        replaced by an in-context pointer to the original. Frozen-prefix and
        cache_control blocks are reference targets only (never rewritten), so no
        cached bytes change. Because per-block compression here is a pure function
        of content (excluded_tool_ids is empty for bash agents, so there is no
        position-dependent gate), the rewrite is prefix-monotonic → the upstream
        prompt-cache prefix stays byte-stable across turns. Never raises.
        """
        try:
            from headroom.transforms.cross_turn_dedup import DedupBlock, dedup_blocks

            locs: list[tuple[int, int | None, int | None]] = []
            dblocks: list[DedupBlock] = []
            tool_name_map = self._build_tool_name_map(messages)
            verbatim_tool_ids = {
                tool_id
                for tool_id, name in tool_name_map.items()
                if is_tool_excluded(name, DEFAULT_VERBATIM_EXCLUDE_TOOLS)
            }

            def _is_user_read_observation(idx: int) -> bool:
                # A file read can land in a plain ``role:user`` STRING (text
                # harnesses: the assistant emits a fenced ``cat/sed/head …`` and the
                # output comes back as the next user turn). Fold those too, but ONLY
                # when the preceding assistant turn actually issued a read command —
                # never ordinary user prose. Same OUTCOME the router uses to protect
                # reads, re-derived here on dedup's own array so indices stay
                # self-consistent (no coupling to pre-scan positional indices).
                for j in range(idx - 1, -1, -1):
                    rj = messages[j].get("role")
                    if rj == "assistant":
                        cmd = _fenced_shell_command(messages[j].get("content"))
                        return bool(cmd and _is_read_command(cmd))
                    if rj == "user":
                        return False
                return False

            for i, msg in enumerate(messages):
                content = msg.get("content")
                frozen = i < frozen_message_count
                if isinstance(content, list):
                    for bidx, block in enumerate(content):
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            continue
                        tc = block.get("content")
                        protected = (
                            frozen
                            or ("cache_control" in block)
                            or block.get("tool_use_id") in verbatim_tool_ids
                        )
                        if isinstance(tc, str) and tc:
                            locs.append((i, bidx, None))
                            dblocks.append(DedupBlock(text=tc, turn=i, protected=protected))
                        elif isinstance(tc, list):
                            # Anthropic tool_result carries LIST content (a `text`
                            # sub-block holds the bash/read output). The string-only
                            # path above skipped these entirely, so reads never
                            # deduped on Anthropic models. Dedup the single text
                            # sub-block (the common form); leave multi-text/mixed
                            # blocks verbatim to stay trivially lossless.
                            text_subs = [
                                (si, sub)
                                for si, sub in enumerate(tc)
                                if isinstance(sub, dict)
                                and sub.get("type") == "text"
                                and isinstance(sub.get("text"), str)
                                and sub.get("text")
                            ]
                            if len(text_subs) == 1:
                                si, sub = text_subs[0]
                                locs.append((i, bidx, si))
                                dblocks.append(
                                    DedupBlock(text=sub["text"], turn=i, protected=protected)
                                )
                elif isinstance(content, str) and content:
                    # Tool output as a STRING under any harness label: OpenAI/Kimi
                    # ``role:tool``, legacy ``role:function``, or a text-harness
                    # ``role:user`` read output. Key off the OUTCOME, not the role —
                    # ``role:user`` is gated to genuine reads so prose never folds.
                    role = msg.get("role")
                    if role in ("tool", "function") or (
                        role == "user" and _is_user_read_observation(i)
                    ):
                        protected = (
                            frozen
                            or ("cache_control" in msg)
                            or msg.get("tool_call_id") in verbatim_tool_ids
                        )
                        locs.append((i, None, None))
                        dblocks.append(DedupBlock(text=content, turn=i, protected=protected))

            if len(dblocks) < 2:
                return messages
            deduped, stats = dedup_blocks(dblocks)
            if not stats.get("spans_folded"):
                return messages

            new_messages = list(messages)
            touched: dict[int, dict[str, Any]] = {}
            for (mi, blk_idx, sub_idx), od, nd in zip(locs, dblocks, deduped):
                if od.protected or nd.text == od.text:
                    continue
                if mi not in touched:
                    src = new_messages[mi]
                    copy = dict(src)
                    if isinstance(src.get("content"), list):
                        copy["content"] = [
                            dict(b) if isinstance(b, dict) else b for b in src["content"]
                        ]
                    touched[mi] = copy
                    new_messages[mi] = copy
                m = touched[mi]
                if blk_idx is None:
                    m["content"] = nd.text
                elif sub_idx is None:
                    m["content"][blk_idx]["content"] = nd.text
                else:
                    # List-content tool_result: deep-copy the block + its sub-list
                    # before mutating so the input message is never touched.
                    blk = dict(m["content"][blk_idx])
                    sub_list = [
                        dict(s) if isinstance(s, dict) else s for s in blk.get("content", [])
                    ]
                    sub_list[sub_idx] = dict(sub_list[sub_idx])
                    sub_list[sub_idx]["text"] = nd.text
                    blk["content"] = sub_list
                    m["content"][blk_idx] = blk

            if route_counts is not None:
                route_counts["cross_turn_dedup"] = (
                    route_counts.get("cross_turn_dedup", 0) + stats["spans_folded"]
                )
            transforms_applied.append(f"router:cross_turn_dedup:{stats['spans_folded']}")
            return new_messages
        except Exception:  # never break the proxy
            return messages

    def _process_content_blocks(
        self,
        message: dict[str, Any],
        content_blocks: list[Any],
        context: str,
        transforms_applied: list[str],
        excluded_tool_ids: set[str],
        ccr_retrieve_tool_ids: set[str],
        tool_name_map: dict[str, str] | None = None,
        route_counts: dict[str, int] | None = None,
        compressed_details: list[str] | None = None,
        min_ratio: float = 0.85,
        read_protection_window: int = 8,
        messages_from_end: int = 0,
        compressor_timing: dict[str, float] | None = None,
        min_chars: int = 500,
        skip_user: bool = True,
        skip_system: bool = True,
        compress_assistant_text_blocks: bool = False,
    ) -> dict[str, Any]:
        """Process content blocks (Anthropic format) for compression.

        Cache-safety contract:
          1. Any block carrying `cache_control` is the client's explicit
             cache breakpoint. Modifying any byte of such a block changes
             the cache key the upstream provider matches against, turning
             a 90% read discount into a 25% write penalty (Anthropic).
             We never modify cache_control'd blocks, regardless of role
             or block type.
          2. Assistant text blocks are echoed back by the client in
             subsequent turns and become part of the upstream provider's
             auto-prefix cache (DeepSeek, OpenAI). Default-skip; opt in
             via `compress_assistant_text_blocks` when the deployment
             knows the backend doesn't honor cache_control AND
             compression is byte-deterministic.
          3. User and system blocks carry the prompt the model is acting
             on; compressing them silently mutates the request. Always
             skipped per `skip_user` / `skip_system`.
          4. Tool / function blocks are tool outputs — semantically safe
             to compress (the model references them once, then moves on).

        Args:
            message: The original message.
            content_blocks: List of content blocks.
            context: Context for compression.
            transforms_applied: List to append transform names to.
            excluded_tool_ids: Tool IDs to skip compression for.
            ccr_retrieve_tool_ids: Tool IDs whose output is a headroom_retrieve result --
                always passed through verbatim (see module-level comment at the
                precompute site for why this can never fall through to compression).
            tool_name_map: Mapping from tool_call_id to tool_name for profile lookup.
            route_counts: Optional routing reason counters to update.
            compressed_details: Optional list to append compression details to.
            min_ratio: Adaptive compression ratio threshold.
            read_protection_window: Messages from end within which excluded tools are protected.
            messages_from_end: How far this message is from the end of the conversation.
            min_chars: Minimum block content length (chars) to consider for compression.
            skip_user: If True, never compress text blocks in user-role messages.
            skip_system: If True, never compress text blocks in system-role messages.
            compress_assistant_text_blocks: If True, allow compressing text blocks in
                assistant-role messages. Default False (cache-safe).

        Returns:
            Transformed message with compressed content blocks.
        """
        new_blocks = []
        any_compressed = False
        role = message.get("role", "")

        # Role-based gate for `text` blocks. Tool/function roles are tool
        # outputs and compress freely; assistant defaults to skip (cache
        # safety) with explicit opt-in; unknown roles default to skip.
        if role == "user":
            protect_text_blocks = skip_user
        elif role in {"system", "developer"}:
            protect_text_blocks = skip_system
        elif role == "assistant":
            protect_text_blocks = not compress_assistant_text_blocks
        elif role in ("tool", "function"):
            protect_text_blocks = False
        else:
            protect_text_blocks = True

        for block in content_blocks:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue

            # Defense in depth: cache_control marker is the client's
            # cache breakpoint. Frozen-message-count is a coarse
            # message-level approximation; this is the per-block
            # guarantee that we never bust an explicit cache key.
            if "cache_control" in block:
                new_blocks.append(block)
                if route_counts is not None:
                    route_counts.setdefault("cache_control_protected", 0)
                    route_counts["cache_control_protected"] += 1
                continue

            block_type = block.get("type")

            # Handle tool_result blocks
            if block_type == "tool_result":
                # Check if tool is excluded from compression
                tool_use_id = block.get("tool_use_id", "")
                # Flatten OpenAI-style list-form content up front (see fix-7 note below)
                # so both the read-protection content check and the compressor see the
                # same text.
                _tr_content = block.get("content", "")
                _tr_list_form_early = (
                    isinstance(_tr_content, list)
                    and bool(_tr_content)
                    and all(isinstance(b, dict) and b.get("type") == "text" for b in _tr_content)
                )
                _tr_text = (
                    "".join(b.get("text", "") for b in _tr_content)
                    if _tr_list_form_early
                    else _tr_content
                )
                # Read protection (HEADROOM_PROTECT_READS): never LOSSY-compress a file
                # read (cat/sed/head/...) whose content is SOURCE CODE — pass it verbatim
                # so the agent keeps exact bytes to edit from. A read of DATA (json/csv/
                # log/lockfile/text) is not byte-patched, so it falls through to its
                # content-specific compressor. Cross-turn dedup still runs later, so
                # re-reads of the same file are losslessly de-duplicated either way.
                if tool_use_id in getattr(
                    self, "_protect_read_tool_ids", ()
                ) and _read_output_should_be_protected(_tr_text):
                    _exp = self._experimental_compress_read(_tr_text, context or "")
                    if _exp is not None:
                        new_blocks.append({**block, "content": _exp})
                        any_compressed = True
                        if route_counts is not None:
                            route_counts["read_kompress_exp"] = (
                                route_counts.get("read_kompress_exp", 0) + 1
                            )
                        continue
                    new_blocks.append(block)
                    if route_counts is not None:
                        route_counts.setdefault("read_protected", 0)
                        route_counts["read_protected"] += 1
                    continue
                # Mirrors the OpenAI-shape guard above (issue #1077): a headroom_retrieve
                # result IS already-retrieved, original CCR content and must never be
                # recompressed. Precomputed set (mirrors excluded_tool_ids immediately
                # below) rather than a per-iteration is_tool_excluded() call.
                if tool_use_id in ccr_retrieve_tool_ids:
                    new_blocks.append(block)
                    transforms_applied.append("router:excluded:ccr_retrieve")
                    if route_counts is not None:
                        route_counts["ccr_retrieve"] = route_counts.get("ccr_retrieve", 0) + 1
                    continue
                if tool_use_id in excluded_tool_ids:
                    tool_name = tool_name_map.get(tool_use_id, "") if tool_name_map else ""
                    if tool_name and is_tool_excluded(tool_name, DEFAULT_VERBATIM_EXCLUDE_TOOLS):
                        new_blocks.append(block)
                        transforms_applied.append("router:excluded:tool")
                        if route_counts is not None:
                            route_counts["excluded_tool"] += 1
                        continue
                    if messages_from_end <= read_protection_window:
                        # Protected from lossy compression — but grep/log/json
                        # output can still be losslessly compacted, UNLESS this is
                        # a file read (see DEFAULT_BYTE_EXACT_EXCLUDE_TOOLS). This
                        # is the shape Claude Code's own `Read` arrives in, so it
                        # has to carry the same guard as the OpenAI branch above.
                        compacted = (
                            None
                            if is_tool_excluded(tool_name, DEFAULT_BYTE_EXACT_EXCLUDE_TOOLS)
                            else self._lossless_compact_excluded(block.get("content"))
                        )
                        if compacted is not None:
                            folded, kind = compacted
                            new_blocks.append({**block, "content": folded})
                            transforms_applied.append(f"router:excluded:lossless_{kind}")
                            if route_counts is not None:
                                route_counts["excluded_tool_lossless"] = (
                                    route_counts.get("excluded_tool_lossless", 0) + 1
                                )
                            any_compressed = True
                            continue
                        # Recent — protect as before
                        new_blocks.append(block)
                        transforms_applied.append("router:excluded:tool")
                        if route_counts is not None:
                            route_counts["excluded_tool"] += 1
                        continue
                    # Old excluded-tool output — fall through to compression

                # Look up tool-specific compression bias
                tool_name = (tool_name_map or {}).get(tool_use_id, "")
                bias = self._get_tool_bias(tool_name) if tool_name else 1.0

                # Enrich the relevance query with the triggering tool call's
                # args (grep pattern, read path, …) — the sharpest, per-output
                # signal. Gated so default behavior is byte-identical.
                block_context = context
                if self.config.relevance_split and tool_use_id:
                    call_args = self._tool_call_args.get(tool_use_id, "")
                    if call_args:
                        block_context = build_relevance_query(context, tool_name, call_args)

                tool_content = block.get("content", "")

                # fix-7: OpenAI-style clients (litellm) send tool_result `content`
                # as a LIST of text blocks ([{"type":"text","text": ...}]), not a
                # bare string. Every check/compressor below is `isinstance(str)`-
                # gated, so list-form tool outputs were skipped entirely (bucketed
                # "small" -> 0% compression even on 10k-char reads). Flatten the
                # text blocks to a string for the checks/compressors, and re-wrap
                # the result in the SAME container on write-back so the on-wire
                # shape is unchanged. Mixed / non-text content (e.g. images) does
                # NOT flatten (tool_text stays the list) -> str checks fail ->
                # block passes through unchanged, exactly as before.
                _tr_list_form = (
                    isinstance(tool_content, list)
                    and bool(tool_content)
                    and all(isinstance(b, dict) and b.get("type") == "text" for b in tool_content)
                )
                tool_text = (
                    "".join(b.get("text", "") for b in tool_content)
                    if _tr_list_form
                    else tool_content
                )

                # Bash-search lossless pre-empt (twin of the string-form path):
                # fold read-only search output (grep/rg/git grep) byte-losslessly
                # instead of taking the lossy strategy path.
                bash_folded = self._bash_search_fold(tool_name, tool_use_id, tool_text)
                if bash_folded is not None:
                    new_blocks.append(
                        {
                            **block,
                            "content": (
                                [{"type": "text", "text": bash_folded}]
                                if _tr_list_form
                                else bash_folded
                            ),
                        }
                    )
                    transforms_applied.append("router:bash:lossless_search")
                    if route_counts is not None:
                        route_counts["bash_lossless_search"] = (
                            route_counts.get("bash_lossless_search", 0) + 1
                        )
                    any_compressed = True
                    continue

                # Protection: failed tool calls / error outputs stay verbatim
                # (issue #847). `is_error` is Anthropic's explicit failure
                # flag and suffices alone; the indicator scan catches error
                # text without the flag but requires >=2 distinct keywords
                # so benign outputs mentioning errors don't skip compression.
                # Above the size cap, fall through — LogCompressor preserves
                # error lines in big logs.
                if (
                    self.config.protect_error_outputs
                    and isinstance(tool_text, str)
                    and len(tool_text) <= self.config.error_protection_max_chars
                    and (
                        block.get("is_error") is True
                        or content_has_strong_error_indicators(tool_text)
                    )
                ):
                    new_blocks.append(block)
                    transforms_applied.append("router:protected:error_output")
                    if route_counts is not None:
                        route_counts.setdefault("error_protected", 0)
                        route_counts["error_protected"] += 1
                    continue

                # Only process string content. Blocks below the lossy min_chars
                # floor still pass when a byte-lossless fold shrinks them — the
                # floor guards the lossy path only; lossless has no size floor.
                if isinstance(tool_text, str) and (
                    len(tool_text) > min_chars or self._has_lossless_fold(tool_text)
                ):
                    # Compression pinning: skip already-compressed content
                    if _is_already_compressed(tool_text):
                        new_blocks.append(block)
                        if route_counts is not None:
                            route_counts.setdefault("already_compressed", 0)
                            route_counts["already_compressed"] += 1
                        continue

                    # Two-tier compression cache → shared helper
                    compressed_content, was_compressed = self._compress_block_content(
                        content=tool_text,
                        content_key=hash((tool_text, getattr(self, "_runtime_target_ratio", None))),
                        context=block_context,
                        bias=bias,
                        min_ratio=min_ratio,
                        compressor_timing=compressor_timing,
                        transforms_applied=transforms_applied,
                        route_counts=route_counts,
                        compressed_details=compressed_details,
                        strategy_label="tool_result",
                        details_prefix="tool",
                        enforce_reversibility=True,
                    )
                    if compressed_content is not None:
                        new_blocks.append(
                            {
                                **block,
                                "content": (
                                    [{"type": "text", "text": compressed_content}]
                                    if _tr_list_form
                                    else compressed_content
                                ),
                            }
                        )
                        any_compressed = True
                    else:
                        new_blocks.append(block)
                    continue
                else:
                    if route_counts is not None:
                        route_counts["small"] += 1

            # Handle text blocks — compress for non-Anthropic clients (e.g.
            # OpenAI/DeepSeek via Cline) whose SDK normalizes content to
            # block-list form. Roles are gated above (user/system always
            # skipped; assistant default-skipped, opt-in via
            # `compress_assistant_text_blocks`).
            elif block_type == "text" and not protect_text_blocks:
                # Same CCR-retrieve exemption as the tool_result branch above, for the
                # top-level-text-block wire shape: a role:"tool"/"function" harness that
                # normalizes content to a block list without a tool_result wrapper (see
                # test_tool_role_text_blocks_compressed_by_default for why this shape is
                # real). role:"tool" resolves via the message's own tool_call_id/
                # tool_use_id through ccr_retrieve_tool_ids; legacy role:"function" has no
                # call id in that shape, so the tool name is read off the message directly.
                if role in ("tool", "function"):
                    _msg_tool_id = message.get("tool_call_id") or message.get("tool_use_id") or ""
                    if _msg_tool_id in ccr_retrieve_tool_ids or (
                        role == "function"
                        and is_tool_excluded(message.get("name", ""), ("headroom_retrieve",))
                    ):
                        new_blocks.append(block)
                        transforms_applied.append("router:excluded:ccr_retrieve")
                        if route_counts is not None:
                            route_counts["ccr_retrieve"] = route_counts.get("ccr_retrieve", 0) + 1
                        continue
                text_content = block.get("text", "")
                if isinstance(text_content, str) and (
                    len(text_content) > min_chars or self._has_lossless_fold(text_content)
                ):
                    # Pinning: skip already-compressed content
                    if _is_already_compressed(text_content):
                        new_blocks.append(block)
                        if route_counts is not None:
                            route_counts.setdefault("already_compressed", 0)
                            route_counts["already_compressed"] += 1
                        continue

                    # Two-tier compression cache → shared helper
                    compressed_content, _was_compressed = self._compress_block_content(
                        content=text_content,
                        content_key=hash(
                            (text_content, getattr(self, "_runtime_target_ratio", None))
                        ),
                        context=context,
                        bias=1.0,
                        min_ratio=min_ratio,
                        compressor_timing=compressor_timing,
                        transforms_applied=transforms_applied,
                        route_counts=route_counts,
                        compressed_details=compressed_details,
                        strategy_label="text_block",
                        details_prefix="text",
                    )
                    if compressed_content is not None:
                        new_blocks.append({**block, "text": compressed_content})
                        any_compressed = True
                    else:
                        new_blocks.append(block)
                    continue
                else:
                    if route_counts is not None:
                        route_counts["small"] += 1

            # Keep block unchanged
            new_blocks.append(block)

        if any_compressed:
            # Preserve any non-standard message-level fields (e.g., reasoning_content)
            new_msg = {**message, "content": new_blocks}
            return new_msg
        return message

    def _compress_block_content(
        self,
        content: str,
        content_key: int,
        context: str,
        bias: float,
        min_ratio: float,
        compressor_timing: dict[str, float] | None,
        transforms_applied: list[str],
        route_counts: dict[str, int] | None,
        compressed_details: list[str] | None,
        strategy_label: str,
        details_prefix: str,
        enforce_reversibility: bool = False,
    ) -> tuple[str | None, bool]:
        """Apply two-tier cache lookup + compression to a single content string.

        Encapsulates the shared cache→compress→store logic used by both
        ``tool_result`` and ``text`` block paths in ``_process_content_blocks``.
        Previously this logic was duplicated ~60 lines per path; centralising
        it ensures both paths stay in sync (cache expiry, pinning, ratio gating).

        Args:
            content: The string content to compress.
            content_key: Pre-computed ``hash(content)`` for cache lookups.
            context: User/query context for relevance-aware compression.
            bias: Compression bias multiplier (tool-specific or 1.0).
            min_ratio: Adaptive minimum compression ratio threshold.
            compressor_timing: Optional dict to accumulate per-strategy timing.
            transforms_applied: List mutated in-place with transform labels.
            route_counts: Optional dict mutated in-place with route counters.
            compressed_details: Optional list mutated with compression details.
            strategy_label: Transform label prefix (e.g. ``"tool_result"``).
            details_prefix: Compressed-details prefix (e.g. ``"tool"``).

        Returns:
            Tuple of ``(compressed_content_or_None, was_compressed)``.
            When ``compressed_content`` is ``None`` the caller should keep
            the original block unchanged. When ``was_compressed`` is
            ``True`` the caller should update the block with the returned
            content and set ``any_compressed``.
        """
        # Cache-churn fix (HEADROOM_FREEZE_BLOCK_DECISION, default off):
        # mirror the plain-STRING path's per-block verdict freeze here, because
        # for Claude Code / Anthropic traffic the prefix is dominated by
        # ``tool_result`` content-blocks that flow through this helper. This is a
        # pure pin: accept/skip uses the SAME live ``min_ratio`` gate as
        # flag-off, and an accepted "compress" verdict is recorded in
        # ``_frozen_verdicts`` and pinned on later turns so rising context
        # pressure can no longer downgrade it (which would churn the prefix
        # cache). The freeze never loosens the first-sighting decision. See the
        # STRING-path note above for why the old fixed aggressive threshold
        # (0.65) made the pin dead code. Default off ⇒ legacy ``min_ratio``
        # behaviour, byte-identical.
        freeze_decision = self._freeze_block_decision_enabled()
        model_ready = self._kompress_model_ready() if freeze_decision else True

        # In lossless-only mode a "skip" means no byte-lossless fold exists for
        # this block (e.g. source code) — it is left verbatim, which is NOT a
        # rejected compression. Bucket it honestly so it doesn't masquerade as
        # ratio_too_high (which properly means "a lossy attempt didn't shrink
        # enough"). In CCR mode the ratio_too_high meaning is unchanged.
        _noop_bucket = "lossless_noop" if self.config.lossless else "ratio_too_high"
        # Tier 1: skip set — instant rejection
        if self._cache.is_skipped(content_key):
            if route_counts is not None:
                route_counts[_noop_bucket] = route_counts.get(_noop_bucket, 0) + 1
                route_counts["cache_hit"] = route_counts.get("cache_hit", 0) + 1
            return None, False

        # Tier 2: result cache — reuse compressed output
        cached = self._cache.get(content_key)
        if cached is not None:
            cached_compressed, cached_ratio, cached_strategy = cached
            if route_counts is not None:
                route_counts["cache_hit"] = route_counts.get("cache_hit", 0) + 1
            # If a "compress" verdict was frozen for this block, REUSE it and
            # bypass the per-turn ``min_ratio`` re-check + ``move_to_skip``
            # downgrade (the frozen verdict here is only ever "compress": a
            # "skip" verdict never warms the result cache).
            if freeze_decision and self._get_frozen_verdict(content_key) is True:
                accept_threshold = 1.0  # already-decided: always accept
                if cached_ratio >= min_ratio:
                    # Counterfactual: legacy min_ratio re-check would downgrade
                    # this block (move_to_skip → original restored → prefix
                    # cache bust). The frozen verdict prevented it. The block
                    # path has no net-cost veto, so an accepted frozen block
                    # always stays compressed — this pin count is exact.
                    self._record_freeze_pin(content, cached_ratio)
            else:
                # Pre-freeze / unfrozen: same live gate as flag-off (pure pin).
                accept_threshold = min_ratio
            if cached_ratio < accept_threshold:
                transforms_applied.append(f"router:{strategy_label}:{cached_strategy}")
                if compressed_details is not None:
                    compressed_details.append(
                        f"{details_prefix}:{cached_strategy}:{cached_ratio:.2f}"
                    )
                # Freeze the "compress" verdict so future turns skip the
                # min_ratio re-check above and never downgrade it.
                if freeze_decision and self._frozen_verdict_recoverable(
                    cached_strategy, cached_compressed
                ):
                    self._record_frozen_verdict(content_key, True)
                return cached_compressed, True
            # Threshold tightened — move result to skip set.
            # (Unreachable when the verdict is frozen-compress: the
            # accept_threshold is pinned to 1.0 above.)
            self._cache.move_to_skip(content_key)
            if route_counts is not None:
                route_counts["ratio_too_high"] = route_counts.get("ratio_too_high", 0) + 1
            return None, False

        # Cache miss — run full compression
        if route_counts is not None:
            route_counts["cache_miss"] = route_counts.get("cache_miss", 0) + 1
        t0 = time.perf_counter()
        result = self.compress(content, context=context, bias=bias)
        compress_ms = (time.perf_counter() - t0) * 1000
        if compressor_timing is not None:
            key = f"compressor:{result.strategy_used.value}"
            compressor_timing[key] = compressor_timing.get(key, 0.0) + compress_ms
        # Lossless-anchored acceptance (byte-measured): a byte/data-lossless fold
        # (search --heading, log run-collapse) has ZERO accuracy cost, so it must
        # never be rejected by the WORD-ratio gate below — heading/indent folds
        # cut tokens while word count stays flat or even rises. Accept on a real
        # BYTE reduction (there is no tokenizer in scope here; byte length is a
        # faithful token proxy for these folds) and store a byte-based ratio so
        # the Tier-2 result cache reuses it on later turns.
        #
        # Two shapes take this path:
        #   • Pure fold  (chain == [lossless_*])           — byte-exact, always safe.
        #   • Fold+lossy (chain == [lossless_*, kompress]) — accepted on bytes
        #     ONLY in no-CCR mode (config.ccr_inject_marker=False), where unmarked
        #     lossy is the deliberate output. The byte-exact fold is the floor, so
        #     the block is guaranteed to shrink and never falls below the fold.
        # A pure fold bypasses the lossy-unmarked reversibility guard (it is
        # recoverable); the fold+lossy tail case only reaches here when markers are
        # off, where that guard is a no-op anyway.
        _chain = getattr(result, "strategy_chain", None) or []
        _starts_lossless = bool(_chain) and _chain[0].startswith("lossless_")
        _is_pure_lossless = _starts_lossless and all(s.startswith("lossless_") for s in _chain)
        _byte_accept = _starts_lossless and (_is_pure_lossless or not self.config.ccr_inject_marker)
        if _byte_accept and len(result.compressed) < len(content):
            _ll_ratio = len(result.compressed) / max(1, len(content))
            _ll_label = _chain[0] if _is_pure_lossless else "+".join(_chain)
            self._cache.put(content_key, result.compressed, _ll_ratio, _ll_label)
            transforms_applied.append(f"router:{strategy_label}:{_ll_label}")
            if compressed_details is not None:
                compressed_details.append(f"{details_prefix}:{_ll_label}:{_ll_ratio:.2f}")
            if route_counts is not None:
                _bucket = "lossless_accept" if _is_pure_lossless else "lossless_then_lossy_accept"
                route_counts[_bucket] = route_counts.get(_bucket, 0) + 1
            if freeze_decision and self._frozen_verdict_recoverable(_ll_label, result.compressed):
                self._record_frozen_verdict(content_key, True)
            return result.compressed, True
        if result.compression_ratio < min_ratio:
            # Tool ground truth must stay reversible: a lossy summarizer
            # (kompress/text/code) that emitted no CCR retrieve marker is
            # unrecoverable, so the agent would act on a fabricated summary
            # (#1307). The string/`role=="tool"` path guards this; mirror it
            # here for tool_result blocks (never cached, so the Tier-2 path
            # above can't serve a poisoned entry).
            #
            # EXCEPTION: no-CCR mode (config.ccr_inject_marker=False). Here the
            # operator has *deliberately* disabled retrieval markers — recovery
            # is not expected, so unmarked lossy output is the intended result,
            # not a bug to skip. This drops the marker-token overhead AND the
            # forgone compressions the guard would otherwise skip. Only applies
            # when markers are off; with markers on the guard is unchanged.
            if (
                enforce_reversibility
                and self.config.ccr_inject_marker
                and result.strategy_used in self.LOSSY_UNMARKED_STRATEGIES
                and not CCR_RETRIEVAL_MARKER_RE.search(result.compressed)
            ):
                self._cache.mark_skip(content_key)
                if route_counts is not None:
                    route_counts["lossy_unrecoverable_skipped"] = (
                        route_counts.get("lossy_unrecoverable_skipped", 0) + 1
                    )
                return None, False
            # Compressed — store in result cache
            self._cache.put(
                content_key,
                result.compressed,
                result.compression_ratio,
                result.strategy_used.value,
            )
            # Freeze "compress" so the cache-hit path above never re-applies a
            # tighter min_ratio to this block.
            if freeze_decision and self._frozen_verdict_recoverable(
                result.strategy_used, result.compressed
            ):
                self._record_frozen_verdict(content_key, True)
            transforms_applied.append(f"router:{strategy_label}:{result.strategy_used.value}")
            if compressed_details is not None:
                compressed_details.append(
                    f"{details_prefix}:{result.strategy_used.value}:{result.compression_ratio:.2f}"
                )
            return result.compressed, True
        # Didn't compress enough — add to skip set. In lossless-only mode this is
        # a "no fold available" passthrough (code/text left verbatim), not a
        # rejected lossy compression, so bucket it as lossless_noop.
        self._cache.mark_skip(content_key)
        if route_counts is not None:
            route_counts[_noop_bucket] = route_counts.get(_noop_bucket, 0) + 1
        # Caveat (1): only freeze a "skip" verdict when the ML model is actually
        # ready. A passthrough caused purely by a still-loading ModernBERT must
        # stay re-evaluable on later turns, so we do NOT freeze it here.
        if freeze_decision and model_ready:
            self._record_frozen_verdict(content_key, False)
        return None, False

    def _detect_analysis_intent(self, messages: list[dict[str, Any]]) -> bool:
        """Detect if user wants to analyze/review code.

        Looks at the most recent user message for analysis keywords.

        Args:
            messages: Conversation messages.

        Returns:
            True if analysis intent detected.
        """
        # Analysis keywords that suggest user wants full code details
        analysis_keywords = {
            "analyze",
            "analyse",
            "review",
            "audit",
            "inspect",
            "security",
            "vulnerability",
            "bug",
            "issue",
            "problem",
            "explain",
            "understand",
            "how does",
            "what does",
            "debug",
            "fix",
            "error",
            "wrong",
            "broken",
            "refactor",
            "improve",
            "optimize",
            "clean up",
        }

        # Find most recent user message
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    content_lower = content.lower()
                    for keyword in analysis_keywords:
                        if keyword in content_lower:
                            return True
                break

        return False

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Check if routing should be applied.

        Always returns True - the router handles all content types.
        """
        return True


def route_and_compress(
    content: str,
    context: str = "",
) -> str:
    """Convenience function for one-off routing and compression.

    Args:
        content: Content to compress.
        context: Optional context for relevance-aware compression.

    Returns:
        Compressed content.

    Example:
        >>> compressed = route_and_compress(mixed_content)
    """
    router = ContentRouter()
    result = router.compress(content, context=context)
    return result.compressed
