"""Pure key policy for proxy semantic response cache."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
from collections.abc import Mapping
from typing import Any

from headroom.proxy.internal_header_policy import INTERNAL_HEADER_PREFIX, is_credential_header

logger = logging.getLogger(__name__)

# Per-process key for the cache partition HMAC. The semantic response cache is
# an in-memory, per-process structure, so the partition only has to be stable
# for this process's lifetime; a fresh random key means a partition id can
# never be computed (or brute-forced from a guessed credential) outside it.
_PARTITION_KEY = secrets.token_bytes(32)

# A header carries caller credentials or selects the billed account when the
# shared credential-header rule matches it (``authorization``,
# ``proxy-authorization``, ``cookie``, ``x-api-key``, ...) or its name ends in
# one of these tokens (``x-goog-api-key``, ``chatgpt-account-id``,
# ``openai-organization``, ``openai-project``, ...). Every such header the
# handlers forward upstream must be in the partition. Matching by shape rather
# than by a fixed provider list keeps new providers partitioned by default.
_CREDENTIAL_HEADER_RE = re.compile(
    r"(^|[-_])(api[-_]?key|key|token|secret|account|account[-_]id|organization|project)$"
)
# Per-request nonces that match the credential shape but identify nothing: a
# fresh value per call would make every request a unique partition.
_NON_CREDENTIAL_HEADERS = frozenset({"idempotency-key", "x-idempotency-key"})
ANONYMOUS_PARTITION = "anon"


def _is_credential_header(name: str) -> bool:
    # ``x-headroom-*`` headers (including the proxy's own token, which the
    # security gate has already removed) never reach the upstream, so they
    # never identify the upstream account.
    lowered = name.lower()
    if lowered.startswith(INTERNAL_HEADER_PREFIX) or lowered in _NON_CREDENTIAL_HEADERS:
        return False
    return is_credential_header(lowered) or bool(_CREDENTIAL_HEADER_RE.search(lowered))


def compute_cache_partition(
    headers: Mapping[str, str] | Any,
    *,
    principal: str | None = None,
) -> str:
    """Return the response-cache partition for one caller.

    Two requests may share a cached response only when they present the same
    provider credentials / account selectors and the same authenticated
    principal. The partition is an HMAC under a per-process random key, never a
    plain hash, so it cannot be used to confirm a guessed credential.

    Requests with no caller credential and no principal (the proxy supplies the
    operator's key) share the ``anon`` partition: they are billed to, and
    answered by, the same account.
    """
    material: list[tuple[str, str]] = []
    items: list[tuple[Any, Any]] = list(headers.items()) if hasattr(headers, "items") else []
    for name, value in items:
        if value and _is_credential_header(str(name)):
            material.append(("h:" + str(name).lower(), str(value).strip()))
    if principal:
        material.append(("principal", principal))
    if not material:
        return ANONYMOUS_PARTITION
    material.sort()
    digest = hmac.new(
        _PARTITION_KEY, json.dumps(material, separators=(",", ":")).encode(), hashlib.sha256
    ).hexdigest()
    return "p_" + digest[:32]


def compute_request_cache_partition(request: Any) -> str | None:
    """Partition for a live proxy request: credentials + authenticated principal.

    Returns ``None`` when an installed identity resolver raises or returns no
    principal. The caller must then bypass the response cache (no lookup, no
    store): falling back to the credential-only partition would let tenants
    sharing one operator key read each other's cached responses.
    """
    from headroom.proxy.identity import resolve_authenticated_principal

    try:
        principal = resolve_authenticated_principal(request)
    except Exception:
        logger.warning(
            "Identity resolver could not establish a principal; "
            "bypassing the response cache for this request",
            exc_info=True,
        )
        return None
    headers = getattr(request, "headers", None) or {}
    return compute_cache_partition(headers, principal=principal)


# Fields whose value is opaque, model-visible payload rather than request
# structure: tool call arguments, JSON Schema bodies, and their literal values.
# ``cache_control`` inside them is application data, not a transport directive,
# so it must stay in the key — two requests that differ there can generate
# different completions (see #4272).
_LITERAL_PAYLOAD_FIELDS = frozenset(
    {
        "arguments",  # OpenAI/Anthropic tool call arguments (and its JSON string)
        "const",
        "default",
        "enum",
        "example",
        "examples",
        "input",  # Anthropic tool_use input
        "input_schema",  # Anthropic tool definition
        "parameters",  # OpenAI tool definition
        "properties",  # JSON Schema property names and their schemas
        "schema",  # OpenAI response_format json_schema
        "text",  # content-block text, when a client sends it as structured data
    }
)


def strip_cache_control(obj: Any) -> Any:
    """Drop transport prompt-cache annotations, preserving literal payloads.

    ``cache_control`` is overloaded: on a content block, a ``system`` entry or a
    tool definition it is a caching directive the provider never feeds to the
    model, so it must not fragment the key; inside tool arguments or a JSON
    Schema it is ordinary application data whose value can change the
    completion, so stripping it would alias two different requests onto one
    cached response (see #4272). Recursion therefore stops at the payload
    fields listed in ``_LITERAL_PAYLOAD_FIELDS`` and continues everywhere else.
    """
    if isinstance(obj, dict):
        return {
            k: v if k in _LITERAL_PAYLOAD_FIELDS else strip_cache_control(v)
            for k, v in obj.items()
            if k != "cache_control"
        }
    if isinstance(obj, list):
        return [strip_cache_control(item) for item in obj]
    return obj


def compute_semantic_cache_key(
    messages: list[dict],
    model: str,
    *,
    partition: str,
    **key_fields: Any,
) -> str:
    """Compute a stable cache key from request content and shaping fields.

    ``cache_control`` is stripped from ``messages`` as well as the shaping
    fields: at request-structure positions it is a prompt-caching directive for
    the upstream provider that never changes the generated completion, so a
    moved breakpoint must not fragment the key. Messages are the primary key
    component and, on the Anthropic path, the most common place a client (e.g.
    Claude Code) moves a breakpoint between turns, so leaving them un-stripped
    defeated the strip for the field that matters most.

    Opaque payload keeps its ``cache_control``-named fields: inside tool inputs,
    tool schemas, and ``const``/``default``/``enum`` values that name is
    application data the model sees, so requests differing there must not share
    a key (see :func:`strip_cache_control`).

    ``partition`` (see :func:`compute_cache_partition`) is mandatory: a shared
    proxy must never answer one caller from another caller's cached response.
    """
    if not partition:
        raise ValueError("semantic cache key requires a caller partition")
    normalized = json.dumps(
        {
            "partition": partition,
            "model": model,
            "messages": strip_cache_control(messages),
            **{k: strip_cache_control(v) for k, v in key_fields.items()},
        },
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]
