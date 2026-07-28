"""Actionable diagnostics for Vertex upstream failures.

Vertex rejects requests for a handful of boring, recurring reasons -- expired
ADC, an un-enabled partner model, a model that simply is not served in the
requested location -- and its own error text says nothing about how to fix any
of them. Users hit these while onboarding, see a bare 404 through whatever SDK
they are using, and have no way to tell a proxy bug from a project-config gap.

So annotate the failure at the proxy, where we know the location, publisher and
model that produced it, and surface the hint three ways: a WARNING in the proxy
log, an ``x-headroom-hint`` response header, and (because most SDKs only ever
show the message string) an appended note on ``error.message`` itself.

Only already-failing responses are touched; success bodies are never modified.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from headroom.providers.registry import BackendUnavailableError

logger = logging.getLogger(__name__)

HINT_HEADER = "x-headroom-hint"

# Statuses worth explaining. Everything else is either fine or genuinely
# unexpected, and inventing a hint for it would just be misleading.
_EXPLAINABLE = frozenset({401, 403, 404, 429})

_ADC_REFRESH = (
    "Credentials are missing or expired (Vertex access tokens last ~1h). "
    "Re-auth with `gcloud auth application-default login`, and note that a bare "
    "`gcloud auth print-access-token` user token is NOT accepted -- use "
    "`gcloud auth application-default print-access-token`."
)

_ENABLE_API = (
    "Confirm `aiplatform.googleapis.com` is enabled on the project and that the "
    "caller holds roles/aiplatform.user."
)

_ENABLE_PARTNER = (
    "Partner models (Anthropic Claude, Llama, Mistral) need a one-time per-project "
    "enable in Model Garden before they will serve -- an otherwise healthy project "
    "returns 404/403 until someone clicks through."
)

_LOCATION_GEMINI = (
    "Gemini 3.x has no US regional endpoint: the evergreen `-latest` aliases are "
    "global-only, and e.g. gemini-3.5-flash serves from `global`, europe-west2 and "
    "asia-northeast1. Retry against `global` before assuming the proxy is at fault."
)

_LOCATION_CLAUDE = (
    "Claude 4.6 and older serve from `us-east5`/europe-west1/asia-southeast1 or "
    "`global`; Claude 4.7+ dropped named regions and is reachable only via the "
    "`us`/`eu` multi-region or `global` endpoints."
)

_QUOTA = (
    "Quota exhausted for this model in this location. Back off and retry, request a "
    "quota increase, or send the request to another supported region."
)


_MISSING_VERTEX_SDK = (
    "`--backend vertex` routes Anthropic Messages traffic through LiteLLM's vertex_ai "
    'provider, which needs the Vertex SDK: `pip install "headroom-ai[proxy,vertex]"` '
    '(or `pip install "google-cloud-aiplatform>=1.38"`). '
    "Note the native Vertex routes (/v1/projects/.../publishers/anthropic/models/...:rawPredict) "
    "are a straight passthrough and do NOT need `--backend vertex` -- drop the flag if you are "
    "calling Vertex paths directly."
)

_MISSING_ADC = (
    "No usable Google credentials were found. Run `gcloud auth application-default login`, or "
    "point GOOGLE_APPLICATION_CREDENTIALS at a service-account key."
)


def vertex_sdk_available() -> bool:
    """Whether the LiteLLM vertex_ai provider's `vertexai` import will succeed."""
    return importlib.util.find_spec("vertexai") is not None


def ensure_vertex_sdk_available() -> None:
    """Refuse to start a Vertex LiteLLM backend that cannot possibly serve.

    Without the SDK the backend still constructs cleanly and then fails on
    *every* request with an opaque provider string. Failing here instead turns a
    per-request mystery into one startup error at the moment of misconfiguration.
    """
    if vertex_sdk_available():
        return
    raise BackendUnavailableError(
        f"Vertex backend selected but the Vertex SDK is missing. {_MISSING_VERTEX_SDK}"
    )


def _is_anthropic(publisher: str) -> bool:
    return publisher == "anthropic"


def backend_error_hint(message: str) -> str | None:
    """Explain a backend-initialization failure, or None if it is not a known one.

    These surface as opaque 500s carrying a raw LiteLLM string; the user has no
    way to know the fix is a missing optional dependency or absent ADC.
    """
    lowered = message.lower()
    if "vertexai" in lowered and ("no module named" in lowered or "import failed" in lowered):
        return _MISSING_VERTEX_SDK
    if "google-cloud-aiplatform" in lowered:
        return _MISSING_VERTEX_SDK
    if "default credentials" in lowered or "could not automatically determine" in lowered:
        return _MISSING_ADC
    return None


def annotate_backend_error_body(
    body: Any,
    status_code: int,
    *,
    logger: logging.Logger | None = None,
    request_id: str = "",
) -> Any:
    """Append a setup hint to a failing backend response body, if we recognize it.

    Backends report setup failures as an ordinary non-2xx *response* rather than
    an exception, so the raw provider string reaches the user untouched. Find
    the message wherever the provider put it and attach the fix.
    """
    if status_code < 400 or not isinstance(body, dict):
        return body

    error = body.get("error")
    holder = error if isinstance(error, dict) else body
    message = holder.get("message")
    if not isinstance(message, str):
        return body

    hint = backend_error_hint(message)
    if not hint or "[headroom] hint:" in message:
        return body

    if logger is not None:
        prefix = f"[{request_id}] " if request_id else ""
        logger.error("%sbackend setup hint: %s", prefix, hint)
    holder["message"] = f"{message}\n[headroom] hint: {hint}"
    return body


def vertex_error_hint(
    status_code: int,
    *,
    location: str = "",
    publisher: str = "",
    model: str = "",
) -> str | None:
    """Return an actionable hint for a Vertex failure, or None if we have none.

    Pure and side-effect free so it can be unit tested without a live upstream.
    """
    if status_code not in _EXPLAINABLE:
        return None

    where = f"{publisher or 'unknown'}/{model or 'unknown'} @ {location or 'unknown'}"
    partner = _is_anthropic(publisher)

    if status_code == 401:
        return f"{where}: {_ADC_REFRESH}"

    if status_code == 429:
        return f"{where}: {_QUOTA}"

    # 403 and 404 are the same user-facing problem wearing two hats: the project
    # cannot serve this model here. Which remedy applies depends on publisher.
    parts = [_ENABLE_API]
    if partner:
        parts.append(_ENABLE_PARTNER)
        parts.append(_LOCATION_CLAUDE)
    else:
        parts.append(_LOCATION_GEMINI)
    return f"{where}: " + " ".join(parts)


def annotate_vertex_error(
    response: Any,
    *,
    location: str = "",
    publisher: str = "",
    model: str = "",
) -> Any:
    """Attach a Headroom hint to a failing Vertex response, in place.

    Returns the same response object so callers can `return annotate(...)`.
    Any problem while annotating is swallowed: a diagnostic must never be able
    to turn a clean upstream error into a proxy 500.
    """
    status = getattr(response, "status_code", None)
    if not isinstance(status, int):
        return response

    hint = vertex_error_hint(status, location=location, publisher=publisher, model=model)
    if hint is None:
        return response

    # Idempotent: nested route helpers can legitimately annotate the same
    # response twice, and a doubled hint reads like a bug.
    try:
        if HINT_HEADER in response.headers:
            return response
    except Exception:  # pragma: no cover - exotic Response implementations
        pass

    logger.warning("vertex upstream %s -- %s", status, hint)

    try:
        response.headers[HINT_HEADER] = hint
    except Exception:  # pragma: no cover - exotic Response implementations
        pass

    # Streaming responses have no materialized body to rewrite; the header and
    # the log line are the whole story for those.
    body = getattr(response, "body", None)
    if not isinstance(body, (bytes, bytearray)):
        return response

    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return response

    if not isinstance(payload, dict):
        return response

    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        # Most SDKs only ever show `error.message`, so the hint has to live
        # there to be seen at all. Keep it clearly attributed to Headroom.
        error["message"] = f"{error['message']}\n[headroom] hint: {hint}"
    else:
        payload["headroom_hint"] = hint

    try:
        new_body = json.dumps(payload).encode("utf-8")
        response.body = new_body
        response.headers["content-length"] = str(len(new_body))
    except Exception:  # pragma: no cover - defensive
        return response

    return response


def with_vertex_diagnostics(
    handler: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Decorate a Vertex route so its failures carry an actionable hint.

    Reads `location`/`publisher`/`model` from the path params FastAPI already
    injects, so a route opts in with one line and no other change.
    """

    @functools.wraps(handler)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return annotate_vertex_error(
            await handler(*args, **kwargs),
            location=kwargs.get("location", ""),
            publisher=kwargs.get("publisher", ""),
            model=kwargs.get("model", ""),
        )

    return wrapper
