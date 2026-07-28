"""Google Vertex provider helpers."""

from .diagnostics import (
    HINT_HEADER,
    annotate_backend_error_body,
    annotate_vertex_error,
    backend_error_hint,
    vertex_error_hint,
    with_vertex_diagnostics,
)
from .runtime import (
    VERTEX_ANTHROPIC_PROVIDER_NAME,
    VERTEX_COUNT_TOKENS,
    VERTEX_GENERATE_CONTENT,
    VERTEX_GOOGLE_PROVIDER_NAME,
    VERTEX_RAW_PREDICT,
    VERTEX_STREAM_GENERATE_CONTENT,
    VERTEX_STREAM_RAW_PREDICT,
    VertexPublisherAction,
    is_vertex_anthropic_publisher,
    is_vertex_google_publisher,
    vertex_anthropic_target,
    vertex_publisher_provider_name,
    vertex_target_for_location,
)

__all__ = [
    "HINT_HEADER",
    "VERTEX_ANTHROPIC_PROVIDER_NAME",
    "VERTEX_COUNT_TOKENS",
    "VERTEX_GENERATE_CONTENT",
    "VERTEX_GOOGLE_PROVIDER_NAME",
    "VERTEX_RAW_PREDICT",
    "VERTEX_STREAM_GENERATE_CONTENT",
    "VERTEX_STREAM_RAW_PREDICT",
    "VertexPublisherAction",
    "annotate_backend_error_body",
    "annotate_vertex_error",
    "backend_error_hint",
    "is_vertex_anthropic_publisher",
    "is_vertex_google_publisher",
    "vertex_anthropic_target",
    "vertex_error_hint",
    "vertex_publisher_provider_name",
    "vertex_target_for_location",
    "with_vertex_diagnostics",
]
