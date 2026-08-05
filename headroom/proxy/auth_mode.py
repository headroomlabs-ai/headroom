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

import logging
from collections.abc import Mapping
from typing import Any

from headroom.proxy.auth_policy import (
    CLIENT_UA_MAP,
    CODEX_RESPONSES_PATH,
    SUBSCRIPTION_UA_PREFIXES,
    AuthMode,
    AuthSignals,
    classify_auth_signals,
    classify_client_signals,
    should_stamp_codex_client_signals,
)

logger = logging.getLogger(__name__)


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


def _auth_signals(headers: Mapping[str, Any] | Any) -> AuthSignals:
    """Adapt a header mapping into pure auth policy inputs."""
    return AuthSignals(
        user_agent=_header_get(headers, "user-agent"),
        authorization=_header_get(headers, "authorization"),
        x_api_key=_header_get(headers, "x-api-key"),
        x_goog_api_key=_header_get(headers, "x-goog-api-key"),
        x_client=_header_get(headers, "x-client"),
    )


def classify_auth_mode(headers: Mapping[str, Any] | Any) -> AuthMode:
    """Classify the auth mode of an inbound request from its headers.

    Decision order (most-specific signal wins):

    1. **Subscription UA prefix** → :data:`AuthMode.SUBSCRIPTION`.
       The CLI's own auth-mode wins over the bearer token shape it
       happens to be carrying — a Claude Code session uses a
       ``sk-ant-oat*`` token but is a subscription client, not OAuth.
    2. **``Authorization: Bearer sk-ant-oat*``** → :data:`AuthMode.OAUTH`
       (Claude Pro / Max OAuth). Checked before the broader ``sk-``
       PAYG rule because ``sk-ant-oat`` shares the ``sk-`` prefix.
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
    return classify_auth_signals(_auth_signals(headers))


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
    return classify_client_signals(_auth_signals(headers), default=default)


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
    return should_stamp_codex_client_signals(path, _auth_signals(headers))


__all__ = [
    "AuthMode",
    "CLIENT_UA_MAP",
    "CODEX_RESPONSES_PATH",
    "SUBSCRIPTION_UA_PREFIXES",
    "classify_auth_mode",
    "classify_client",
    "should_stamp_codex_client",
]
