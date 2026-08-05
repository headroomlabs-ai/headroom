"""Pure Vertex provider routing formulas."""

from __future__ import annotations

from dataclasses import dataclass

from headroom.providers.registry import DEFAULT_VERTEX_API_URL

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
    """Return the Vertex upstream target for a request location."""
    if configured_target and configured_target != DEFAULT_VERTEX_API_URL:
        return configured_target
    if not location or location == "global":
        return "https://aiplatform.googleapis.com"
    return f"https://{location}-aiplatform.googleapis.com"
