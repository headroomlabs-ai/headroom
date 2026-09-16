"""Pure Vertex provider routing formulas."""

from __future__ import annotations

import re
from dataclasses import dataclass

from headroom.providers.registry import DEFAULT_VERTEX_API_URL

# The public (multi-region) Vertex endpoint, used for ``global`` and for any
# location that is not a well-formed region.
_VERTEX_GLOBAL_API_URL = "https://aiplatform.googleapis.com"

# A GCP region label: lowercase alphanumeric groups joined by single hyphens
# (e.g. ``us-central1``, ``europe-west4``, ``asia-northeast1``). Anchored and
# deliberately strict — no dots, colons, slashes, ``#``, ``@``, uppercase, or
# empty groups — so a user-controlled ``location`` can never carry a host,
# port, path, or URL-fragment delimiter into the interpolated hostname.
_VERTEX_REGION_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

VERTEX_GOOGLE_PUBLISHER = "google"
VERTEX_ANTHROPIC_PUBLISHER = "anthropic"
VERTEX_GOOGLE_PROVIDER_NAME = "vertex:google"
VERTEX_ANTHROPIC_PROVIDER_NAME = "vertex:anthropic"


@dataclass(frozen=True, slots=True)
class VertexPublisherAction:
    """A Vertex publisher action exposed by route registration."""

    name: str
    force_stream: bool = False


VERTEX_GENERATE_CONTENT = VertexPublisherAction("generateContent")
VERTEX_STREAM_GENERATE_CONTENT = VertexPublisherAction("streamGenerateContent")
VERTEX_COUNT_TOKENS = VertexPublisherAction("countTokens")
VERTEX_RAW_PREDICT = VertexPublisherAction("rawPredict")
VERTEX_STREAM_RAW_PREDICT = VertexPublisherAction("streamRawPredict", force_stream=True)


def is_vertex_google_publisher(publisher: str) -> bool:
    """Return whether a Vertex publisher should use Gemini-style handlers."""
    return publisher == VERTEX_GOOGLE_PUBLISHER


def is_vertex_anthropic_publisher(publisher: str) -> bool:
    """Return whether a Vertex publisher should use Anthropic-style handlers."""
    return publisher == VERTEX_ANTHROPIC_PUBLISHER


def vertex_publisher_provider_name(publisher: str) -> str:
    """Return the provider label used for Vertex publisher passthrough telemetry."""
    return f"vertex:{publisher}"


def vertex_anthropic_target(base_url: str, *, versionless_route: bool = False) -> str:
    """Return the Anthropic-on-Vertex upstream target for a route shape."""
    if versionless_route:
        return base_url.rstrip("/") + "/v1"
    return base_url


def vertex_target_for_location(configured_target: str, location: str) -> str:
    """Return the Vertex upstream target for a request location.

    ``location`` is a user-controlled URL path segment that is interpolated into
    the upstream hostname, so it must be validated against the GCP region shape
    before use. Without that check a value such as ``169.254.169.254#`` (decoded
    from a percent-encoded ``%23`` in the path) produces
    ``https://169.254.169.254#-aiplatform.googleapis.com``, which an HTTP client
    parses as host ``169.254.169.254`` with the remainder treated as a URL
    fragment — a server-side request forgery to the cloud metadata endpoint
    (CWE-918). Any ``location`` that is not a well-formed region (including port,
    path, or fragment-delimiter payloads) falls back to the default public
    endpoint, which can never resolve to an attacker-chosen host.

    An explicitly configured gateway target still wins outright; region
    derivation only applies when running against the default Vertex endpoint.
    """
    if configured_target and configured_target != DEFAULT_VERTEX_API_URL:
        return configured_target
    if not location or location == "global" or not _VERTEX_REGION_RE.match(location):
        return _VERTEX_GLOBAL_API_URL
    return f"https://{location}-aiplatform.googleapis.com"
