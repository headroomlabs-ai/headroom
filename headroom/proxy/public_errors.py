"""Fixed-vocabulary client-facing errors for transport and internal failures.

Why this exists
---------------
When an upstream connection fails, the exception text produced by ``httpx``,
``ssl`` or the OS names the resolved upstream host, its IP address, the
library that failed and sometimes the local interface. Returning that text to
the caller (``"message": str(e)``) turns every transport failure into an
information-disclosure finding: a client that can steer the upstream (for
example through ``x-headroom-base-url``) learns internal hostnames and
addresses one error at a time.

The rule enforced here is simple and applies to every response path:

* **Transport-layer and internal exception text never reaches a client.** It is
  logged server-side with the request id, and the client receives one of the
  fixed codes below plus that request id so the two can be correlated.
* **Only a provider's own HTTP error response is forwarded.** A model
  provider's ``{"error": {"message": ...}}`` (authentication failed, model not
  found, rate limited) is addressed to the caller. It is recognised by
  exception type (:func:`is_provider_response_error`), never by guessing from
  text; every other exception (``RuntimeError``, ``ValueError``, parser,
  database or library errors) fails closed to :data:`INTERNAL_ERROR`.

The module is pure policy: no I/O, no logging, no FastAPI types. Handlers and
backends build their wire shape (OpenAI ``{"error": {...}}`` or Anthropic
``{"type": "error", "error": {...}}``) with the helpers at the bottom so the
vocabulary cannot drift between transports.
"""

from __future__ import annotations

import asyncio
import ssl
from typing import Any, Final

try:  # httpx is a proxy-extra dependency; the SDK imports this module without it.
    import httpx
except ImportError:  # pragma: no cover - exercised only in SDK-only installs
    httpx = None  # type: ignore[assignment]

# --- The vocabulary --------------------------------------------------------

UPSTREAM_UNREACHABLE: Final = "upstream_unreachable"
UPSTREAM_TIMEOUT: Final = "upstream_timeout"
UPSTREAM_TLS_ERROR: Final = "upstream_tls_error"
UPSTREAM_PROTOCOL_ERROR: Final = "upstream_protocol_error"
INTERNAL_ERROR: Final = "internal_error"

PUBLIC_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        UPSTREAM_UNREACHABLE,
        UPSTREAM_TIMEOUT,
        UPSTREAM_TLS_ERROR,
        UPSTREAM_PROTOCOL_ERROR,
        INTERNAL_ERROR,
    }
)

# One sentence per code. These are the only strings a client sees for a
# transport or internal failure; they carry no host, address, path or
# library detail by construction.
_MESSAGES: Final[dict[str, str]] = {
    UPSTREAM_UNREACHABLE: "Failed to connect to upstream API.",
    UPSTREAM_TIMEOUT: "Upstream API did not respond in time.",
    UPSTREAM_TLS_ERROR: "Could not establish a trusted TLS connection to the upstream API.",
    UPSTREAM_PROTOCOL_ERROR: ("Upstream API closed the connection or sent a malformed response."),
    INTERNAL_ERROR: "The proxy could not complete the request.",
}

# Anthropic's error envelope uses ``error.type``; keep the values it already
# emits today so SDKs keep classifying them the same way.
_ANTHROPIC_TYPES: Final[dict[str, str]] = {
    UPSTREAM_UNREACHABLE: "connection_error",
    UPSTREAM_TIMEOUT: "connection_error",
    UPSTREAM_TLS_ERROR: "connection_error",
    UPSTREAM_PROTOCOL_ERROR: "api_error",
    INTERNAL_ERROR: "api_error",
}


# --- Classification --------------------------------------------------------


def _iter_causes(exc: BaseException) -> list[BaseException]:
    """The exception and its ``__cause__``/``__context__`` chain, outermost first."""
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen and len(seen) < 16:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _is_tls_failure(exc: BaseException) -> bool:
    for item in _iter_causes(exc):
        if isinstance(item, ssl.SSLError):
            return True
        text = str(item).upper()
        if "CERTIFICATE_VERIFY_FAILED" in text or "SSL:" in text or "TLS" in text.split():
            return True
    return False


def classify_exception(exc: BaseException) -> str | None:
    """Return the public code for a transport/runtime failure, or ``None``.

    ``None`` means "this is not a transport-layer failure". Client-facing code
    uses :func:`classify_or_internal` so that anything unrecognised becomes
    :data:`INTERNAL_ERROR`: it was never meant for a client.
    """
    if _is_tls_failure(exc):
        return UPSTREAM_TLS_ERROR

    if httpx is not None:
        if isinstance(exc, (httpx.ConnectTimeout, httpx.PoolTimeout)):
            return UPSTREAM_UNREACHABLE
        if isinstance(exc, httpx.TimeoutException):
            return UPSTREAM_TIMEOUT
        if isinstance(exc, httpx.ConnectError):
            return UPSTREAM_UNREACHABLE
        if isinstance(exc, httpx.ProtocolError):
            return UPSTREAM_PROTOCOL_ERROR
        if isinstance(exc, httpx.DecodingError):
            return UPSTREAM_PROTOCOL_ERROR
        if isinstance(exc, httpx.NetworkError):
            # ReadError / WriteError / CloseError: the connection existed and
            # then broke mid-exchange.
            return UPSTREAM_PROTOCOL_ERROR
        if isinstance(exc, httpx.TransportError):
            return UPSTREAM_UNREACHABLE

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return UPSTREAM_TIMEOUT
    if isinstance(exc, ConnectionError):
        # ConnectionResetError / BrokenPipeError / ConnectionRefusedError.
        return UPSTREAM_UNREACHABLE
    if isinstance(exc, OSError):
        # Anything else from the socket layer (EHOSTUNREACH, ENETDOWN, ...).
        return UPSTREAM_UNREACHABLE
    return None


def classify_or_internal(exc: BaseException) -> str:
    """Classify, treating anything unrecognised as :data:`INTERNAL_ERROR`."""
    return classify_exception(exc) or INTERNAL_ERROR


# --- Wire shapes -------------------------------------------------------------


def public_message(code: str, *, request_id: str | None = None, hint: str | None = None) -> str:
    """The client-facing sentence for ``code``.

    ``hint`` is an *already public* operator-facing string (for example the
    corporate TLS-inspection explanation from ``tls_diagnostics``) and is
    appended verbatim; callers must not pass exception text through it.
    """
    if code not in PUBLIC_ERROR_CODES:
        raise ValueError(f"not a public error code: {code!r}")
    text = _MESSAGES[code]
    if hint:
        text = f"{text} {hint.strip()}"
    if request_id:
        text = f"{text} (request_id={request_id})"
    return text


def openai_error_body(
    code: str,
    *,
    request_id: str | None = None,
    error_type: str | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    """``{"error": {...}}`` in the OpenAI shape.

    ``error_type`` defaults to the value each call site emitted before this
    module existed (``connection_error`` for connection failures, otherwise
    ``api_error``) so clients that switch on it keep working; ``code`` always
    carries the fixed vocabulary.
    """
    error: dict[str, Any] = {
        "message": public_message(code, request_id=request_id, hint=hint),
        "type": error_type or _ANTHROPIC_TYPES[code],
        "code": code,
    }
    if request_id:
        error["request_id"] = request_id
    return {"error": error}


def anthropic_error_body(
    code: str,
    *,
    request_id: str | None = None,
    error_type: str | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    """``{"type": "error", "error": {...}}`` in the Anthropic shape."""
    error: dict[str, Any] = {
        "type": error_type or _ANTHROPIC_TYPES[code],
        "message": public_message(code, request_id=request_id, hint=hint),
        "code": code,
    }
    body: dict[str, Any] = {"type": "error", "error": error}
    if request_id:
        body["request_id"] = request_id
    return body


def is_provider_response_error(exc: BaseException) -> bool:
    """True when ``exc`` carries a model provider's HTTP error response.

    LiteLLM's provider errors (AuthenticationError, NotFoundError,
    RateLimitError, ...) subclass ``openai.APIStatusError``. Imported lazily:
    neither SDK is needed to import this module.
    """
    try:
        import openai

        if isinstance(exc, openai.APIStatusError):
            return True
    except ImportError:  # pragma: no cover - openai is a proxy-extra dependency
        pass
    try:
        import anthropic

        if isinstance(exc, anthropic.APIStatusError):
            return True
    except ImportError:  # pragma: no cover - optional dependency
        pass
    return False


def client_message(exc: BaseException, provider_text: str) -> str:
    """The message a client may see for ``exc``.

    ``provider_text`` is returned only when ``exc`` is a provider's HTTP error
    response (authentication, model not found, rate limited): that text is
    addressed to the caller. Transport failures map to their fixed code and
    anything else to :data:`INTERNAL_ERROR`; the caller logs the real text.
    """
    if is_provider_response_error(exc):
        return provider_text
    return public_message(classify_or_internal(exc))


def tool_result_error(exc: BaseException) -> dict[str, Any]:
    """``{"status": "error", ...}`` for a JSON tool result returned to a model.

    The model only ever sees a fixed code: a storage, network or internal
    failure names no path, host or library. The caller logs the real text.
    """
    code = classify_or_internal(exc)
    return {"status": "error", "error": code, "message": public_message(code)}
