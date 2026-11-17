"""Actionable diagnostics for Vertex upstream failures.

Vertex rejects requests for a handful of boring, recurring reasons -- expired
ADC, an un-enabled partner model, a model that simply is not served in the
requested location -- and its own error text says nothing about how to fix any
of them. Users hit these while onboarding, see a bare 404 through whatever SDK
they are using, and have no way to tell a proxy bug from a project-config gap.

So annotate the failure at the proxy and surface a hint three ways: a WARNING in
the proxy log, an ``x-headroom-hint`` response header, and (because most SDKs
only ever show the message string) an appended note on ``error.message``.

Every hint is one of the fixed constants below: request data and exception text
select a hint but are never interpolated into it (public_errors contract).
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
from headroom.proxy.public_errors import client_message

from .runtime import is_vertex_anthropic_publisher

logger = logging.getLogger(__name__)

HINT_HEADER = "x-headroom-hint"

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
    if not vertex_sdk_available():
        raise BackendUnavailableError(
            f"Vertex backend selected but the Vertex SDK is missing. {_MISSING_VERTEX_SDK}"
        )


def backend_error_hint(message: str) -> str | None:
    """Fixed setup hint for a raw LiteLLM Vertex error, or None if unrecognized."""
    lowered = message.lower()
    if "google-cloud-aiplatform" in lowered or (
        "vertexai" in lowered and ("no module named" in lowered or "import failed" in lowered)
    ):
        return _MISSING_VERTEX_SDK
    if "default credentials" in lowered or "could not automatically determine" in lowered:
        return _MISSING_ADC
    return None


def public_backend_error_message(exc: BaseException, raw_text: str) -> str:
    """``client_message`` plus a fixed setup hint selected by ``raw_text``.

    ``raw_text`` itself never reaches the client; the caller logs it.
    """
    message = client_message(exc, raw_text)
    hint = backend_error_hint(raw_text)
    # A chained Headroom upstream may already have added one.
    if not hint or "[headroom] hint:" in message:
        return message
    return f"{message}\n[headroom] hint: {hint}"


def vertex_error_hint(status_code: int, *, publisher: str = "") -> str | None:
    """Fixed hint for a failing Vertex status, or None if we have none."""
    if status_code == 401:
        return _ADC_REFRESH
    if status_code == 429:
        return _QUOTA
    if status_code not in (403, 404):
        return None
    # 403 and 404 are the same problem for the user: the project cannot serve
    # this model here. Which remedy applies depends on the publisher.
    if is_vertex_anthropic_publisher(publisher):
        return f"{_ENABLE_API} {_ENABLE_PARTNER} {_LOCATION_CLAUDE}"
    if publisher != "google":
        return f"{_ENABLE_API} {_ENABLE_PARTNER}"
    return f"{_ENABLE_API} {_LOCATION_GEMINI}"


def annotate_vertex_error(
    response: Any,
    *,
    location: str = "",
    publisher: str = "",
    model: str = "",
) -> Any:
    """Attach a Headroom hint to a failing Vertex response, in place.

    Returns the same response. Any problem while annotating is swallowed: a
    diagnostic must never turn a clean upstream error into a proxy 500.
    """
    status = getattr(response, "status_code", None)
    hint = vertex_error_hint(status, publisher=publisher) if isinstance(status, int) else None
    if hint is None:
        return response
    try:
        # Idempotent: nested route helpers can annotate the same response twice.
        if HINT_HEADER in response.headers:
            return response
        # %r: path params are client-supplied; keep CR/LF from forging log lines.
        logger.warning(
            "vertex upstream %s for %r/%r @ %r -- %s", status, publisher, model, location, hint
        )
        response.headers[HINT_HEADER] = hint

        # Streaming responses have no materialized body; header + log suffice.
        body = getattr(response, "body", None)
        if not isinstance(body, (bytes, bytearray)):
            return response
        payload = json.loads(body)
        if not isinstance(payload, dict):
            return response
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            # Most SDKs only show `error.message`, so the hint must live there.
            error["message"] = f"{error['message']}\n[headroom] hint: {hint}"
        else:
            payload["headroom_hint"] = hint
        response.body = json.dumps(payload).encode("utf-8")
        response.headers["content-length"] = str(len(response.body))
    except Exception:  # noqa: BLE001 - see docstring
        pass
    return response


def with_vertex_diagnostics(
    handler: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Decorate a Vertex route so its failures carry an actionable hint.

    Reads `location`/`publisher`/`model` from the path params FastAPI injects.
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
