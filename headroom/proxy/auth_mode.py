"""Auth-mode classifier — Phase F PR-F1 (Python port).

Direct port of ``crates/headroom-core/src/auth_mode.rs``. The two
implementations MUST agree on the classification of every header set
the parity tests cover (``tests/test_auth_mode.py``).

See the Rust module for the full WHY of three modes (PAYG / OAuth /
Subscription) and the per-mode compression policy implications. This
port is the live classifier on the Python proxy paths until Phase H
deletes the Python proxy entirely.

The classifier is **pure** (no I/O, no logging of header values), runs
well under 10us per call, and NEVER raises on malformed headers —
non-UTF-8 / unparseable values fall through to the safe default
:data:`AuthMode.PAYG` after a ``logger.warning`` so operators can
spot bad clients without taking the proxy down.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


class AuthMode(str, enum.Enum):
    """Three auth-mode classes Headroom routes compression policy through.

    Subclasses :class:`str` so the enum members serialize transparently
    into structured logs / metric labels / TOIN aggregation keys. The
    string form matches the Rust ``AuthMode::as_str()`` output exactly.
    """

    #: Pay-as-you-go API key. Aggressive live-zone compression OK.
    PAYG = "payg"

    #: OAuth bearer / Bedrock IAM / Vertex ADC. Passthrough-prefer:
    #: no auto-cache_control, no auto-prompt_cache_key, no lossy
    #: compressors. Lossless-only path.
    OAUTH = "oauth"

    #: Subscription-bound CLI / IDE. Stealth: same as OAuth + preserve
    #: ``accept-encoding``, never strip; never inject ``X-Headroom-*``;
    #: never mutate ``User-Agent``.
    SUBSCRIPTION = "subscription"


# User-Agent prefixes that identify a UX-bound CLI / IDE.
#
# Lives at module scope (not inside :func:`classify_auth_mode`) so:
# 1. A future PR can swap this for a configurable list (Phase F
#    follow-up) without touching the function body.
# 2. Adding a new client = one-line edit here, no logic change.
#
# Match is ``str.__contains__`` against a lowercased copy of the UA —
# so the prefix can appear anywhere in the value.
SUBSCRIPTION_UA_PREFIXES: tuple[str, ...] = (
    "claude-cli/",
    "claude-code/",
    "codex-cli/",
    "cursor/",
    "grok/",
    "claude-vscode/",
    "github-copilot/",
    "anthropic-cli/",
    "antigravity/",
)


def _header_get(headers: Mapping[str, Any] | Any, name: str) -> str:
    """Read a single header, case-insensitively, returning ``""`` on miss.

    Accepts either a plain ``Mapping[str, str]`` (test fixtures) or a
    Starlette/FastAPI ``Headers`` object (production). Handles bytes
    values defensively — non-UTF-8 returns ``""`` after a warning,
    matching the Rust path's behaviour.
    """
    # Starlette `Headers` is case-insensitive natively; plain dicts
    # are not. Try a direct lookup first (covers Starlette + the
    # tests), then fall through to a manual case-insensitive scan.
    value: Any = None
    try:
        # Starlette's Headers, plain dict
        value = headers.get(name)
        if value is None:
            # Some test fixtures pass a normal dict with mixed case.
            for k, v in headers.items():  # type: ignore[union-attr]
                if isinstance(k, str) and k.lower() == name:
                    value = v
                    break
    except AttributeError:
        return ""

    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "auth_mode_classify_unparseable_%s",
                name.replace("-", "_"),
                extra={"event": f"auth_mode_classify_unparseable_{name.replace('-', '_')}"},
            )
            return ""
    return str(value)


def classify_auth_mode(headers: Mapping[str, Any] | Any) -> AuthMode:
    """Classify the auth mode of an inbound request from its headers.

    Decision order (most-specific signal wins):

    1. **Subscription UA prefix** → :data:`AuthMode.SUBSCRIPTION`.
       The CLI's own auth-mode wins over the bearer token shape it
       happens to be carrying — a Claude Code session uses a
       ``sk-ant-oat-*`` token but is a subscription client, not OAuth.
    2. **``Authorization: Bearer sk-ant-oat-*``** → :data:`AuthMode.OAUTH`
       (Claude Pro / Max OAuth). Checked before the broader ``sk-``
       PAYG rule because ``sk-ant-oat-`` shares the ``sk-`` prefix.
    3. **``Authorization: Bearer sk-ant-api*`` or ``Bearer sk-*``** →
       :data:`AuthMode.PAYG` (Anthropic / OpenAI API key).
    4. **``Authorization: Bearer <jwt>``** (3 dot-separated segments)
       → :data:`AuthMode.OAUTH` (Codex / Cursor / Copilot OAuth).
    5. **``Authorization`` present but not ``Bearer ...``** →
       :data:`AuthMode.OAUTH` (AWS SigV4 ``AWS4-HMAC-SHA256 ...`` →
       Bedrock; any other non-Bearer scheme is presumed
       passthrough-prefer too).
    6. **``x-api-key`` present** → :data:`AuthMode.PAYG` (Anthropic
       API key style).
    7. **``x-goog-api-key`` present** → :data:`AuthMode.PAYG` (Gemini
       key).
    8. **Default** → :data:`AuthMode.PAYG` (safest default; aggressive
       compression on a misclassified request just costs us a re-run,
       not a revoked subscription).

    Performance: one ``str.lower`` allocation for the UA copy. All
    other matches are zero-allocation ``startswith`` / ``in`` /
    ``str.split('.')``. Target: <10us per call.
    """
    # ── User-Agent ────────────────────────────────────────────────
    ua_lower = _header_get(headers, "user-agent").lower()
    for prefix in SUBSCRIPTION_UA_PREFIXES:
        if prefix in ua_lower:
            return AuthMode.SUBSCRIPTION

    # ── Authorization header ──────────────────────────────────────
    auth = _header_get(headers, "authorization")

    if auth.startswith("Bearer "):
        token = auth[len("Bearer ") :]
        # Order matters: `sk-ant-oat-*` shares a prefix with
        # `sk-ant-api*` only at `sk-ant-`, so check OAuth first.
        if token.startswith("sk-ant-oat-"):
            return AuthMode.OAUTH
        if token.startswith("sk-ant-api") or token.startswith("sk-"):
            return AuthMode.PAYG
        # JWT: classic three-segment `header.payload.signature`.
        # We don't validate the JWT — just count dot-separated
        # segments. Catches Codex / Cursor / Copilot OAuth.
        if len(token.split(".")) >= 3:
            return AuthMode.OAUTH
        # Unknown bearer shape — fall through.
    elif auth:
        # Authorization is present but NOT `Bearer ...` — most
        # commonly AWS SigV4 (`AWS4-HMAC-SHA256 ...`) on a Bedrock
        # request, or a `Basic ...` from a custom proxy chain. We
        # treat all such non-Bearer schemes as passthrough-prefer.
        return AuthMode.OAUTH

    # ── Vendor-specific API-key headers ───────────────────────────
    if _header_get(headers, "x-api-key"):
        return AuthMode.PAYG
    if _header_get(headers, "x-goog-api-key"):
        return AuthMode.PAYG

    # ── Default ───────────────────────────────────────────────────
    return AuthMode.PAYG


# Client (harness) identification — maps a User-Agent substring to a
# short normalized client name. The dashboard / `headroom perf` use
# this to slice traffic by harness ("aider is 30% of cache writes",
# "codex p99 latency vs claude-code", etc).
#
# Adding a new client: one line. Same surface as
# :data:`SUBSCRIPTION_UA_PREFIXES`. The classifier is intentionally
# substring-based (not regex) so a tuple of literals stays the
# extension point.
CLIENT_UA_MAP: tuple[tuple[str, str], ...] = (
    # Anthropic ecosystem
    ("claude-code/", "claude-code"),
    ("claude-cli/", "claude-code"),
    ("claude-vscode/", "claude-vscode"),
    ("anthropic-cli/", "anthropic-cli"),
    # OpenAI ecosystem
    ("codex-cli/", "codex"),
    # Editors / IDEs
    ("cursor/", "cursor"),
    ("grok/", "grok_build"),
    ("zed/", "zed"),
    # Other AI coding harnesses
    ("aider/", "aider"),
    ("droid/", "droid"),
    ("opencode/", "opencode"),
    ("github-copilot/", "copilot"),
    # Google's experimental harness
    ("antigravity/", "antigravity"),
    # AWS Strands Agents SDK. The default openai-python User-Agent
    # is `OpenAI/Python <ver>` which does not embed any Strands
    # signal, so production callers should set `X-Client: strands`
    # explicitly (see `headroom/integrations/strands/README.md`).
    # The UA prefix below covers any Strands runtime that injects
    # its own segment ahead of the openai-python UA.
    ("strands-agents/", "strands"),
)


def classify_client(headers: Mapping[str, Any] | Any, *, default: str | None = None) -> str | None:
    """Identify the client harness (Codex / Claude Code / aider / etc).

    Decision order:

    1. **``X-Client`` header** (explicit override) — clients that
       know they're talking to Headroom can self-identify with a
       short name. Trimmed, lowercased. Wins over UA matching.
    2. **User-Agent substring match** against :data:`CLIENT_UA_MAP`
       — covers the unmodified-client case. Substring, not prefix,
       because some clients prepend a corporate-wrapper UA before
       their own.
    3. **None** when neither produces a hit. ``None`` is the loud
       "unknown harness" signal; downstream consumers can group
       these as "unidentified" rather than silently bucketing them
       into a default.

    Returns ``str | None`` rather than a string default so future
    code can distinguish "no client identified" from "client is the
    empty string". The :class:`RequestOutcome` field has the same
    type for the same reason.
    """
    # 1. Explicit override
    explicit = _header_get(headers, "x-client").strip().lower()
    if explicit:
        return explicit
    # 2. User-Agent substring match
    ua_lower = _header_get(headers, "user-agent").lower()
    if not ua_lower:
        return None
    for needle, name in CLIENT_UA_MAP:
        if needle in ua_lower:
            return name
    return default


# OpenAI's Responses API endpoint. In practice this is Codex's endpoint, but a
# proxy can't assume every caller here is Codex — hence
# :func:`should_stamp_codex_client` only stamps callers that don't already
# classify.
CODEX_RESPONSES_PATH = "/v1/responses"


def should_stamp_codex_client(path: str, headers: Mapping[str, Any] | Any) -> bool:
    """Whether to stamp ``X-Client: codex`` on a request to the proxy.

    Stamping ``X-Client: codex`` on the Responses endpoint makes the backend
    take the codex fail-open branch on a compression timeout — Codex treats the
    proxy's 413/1009 refusal as a hard connection failure. This is needed
    because Codex Desktop's User-Agent (``Codex Desktop/...``) isn't in
    :data:`CLIENT_UA_MAP` and would otherwise be refused.

    Returns ``True`` only for an unidentified caller (no ``X-Client`` and no
    recognized User-Agent) on the Responses endpoint. A caller that already
    classifies is left untouched.
    """
    if path != CODEX_RESPONSES_PATH and not path.startswith(CODEX_RESPONSES_PATH + "/"):
        return False
    return classify_client(headers) is None


__all__ = [
    "AuthMode",
    "CLIENT_UA_MAP",
    "CODEX_RESPONSES_PATH",
    "SUBSCRIPTION_UA_PREFIXES",
    "classify_auth_mode",
    "classify_client",
    "should_stamp_codex_client",
]
