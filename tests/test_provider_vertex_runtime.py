from __future__ import annotations

from headroom.providers.registry import DEFAULT_VERTEX_API_URL
from headroom.providers.vertex import (
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


def test_vertex_publisher_classification_is_explicit() -> None:
    assert is_vertex_google_publisher("google") is True
    assert is_vertex_google_publisher("anthropic") is False
    assert is_vertex_anthropic_publisher("anthropic") is True
    assert is_vertex_anthropic_publisher("google") is False


def test_vertex_provider_names_are_provider_owned() -> None:
    assert VERTEX_GOOGLE_PROVIDER_NAME == "vertex:google"
    assert VERTEX_ANTHROPIC_PROVIDER_NAME == "vertex:anthropic"
    assert vertex_publisher_provider_name("mistral") == "vertex:mistral"


def test_vertex_publisher_actions_are_named_values() -> None:
    assert VERTEX_GENERATE_CONTENT == VertexPublisherAction("generateContent")
    assert VERTEX_STREAM_GENERATE_CONTENT == VertexPublisherAction("streamGenerateContent")
    assert VERTEX_COUNT_TOKENS == VertexPublisherAction("countTokens")
    assert VERTEX_RAW_PREDICT == VertexPublisherAction("rawPredict")
    assert VERTEX_STREAM_RAW_PREDICT == VertexPublisherAction(
        "streamRawPredict",
        force_stream=True,
    )


def test_vertex_anthropic_target_adds_v1_only_for_versionless_routes() -> None:
    assert vertex_anthropic_target("https://europe-west1-aiplatform.googleapis.com") == (
        "https://europe-west1-aiplatform.googleapis.com"
    )
    assert (
        vertex_anthropic_target(
            "https://europe-west1-aiplatform.googleapis.com/",
            versionless_route=True,
        )
        == "https://europe-west1-aiplatform.googleapis.com/v1"
    )


def test_vertex_target_for_location_derives_regional_hosts_from_default_target() -> None:
    assert vertex_target_for_location(DEFAULT_VERTEX_API_URL, "europe-west1") == (
        "https://europe-west1-aiplatform.googleapis.com"
    )
    assert vertex_target_for_location(DEFAULT_VERTEX_API_URL, "global") == (
        "https://aiplatform.googleapis.com"
    )
    assert vertex_target_for_location(DEFAULT_VERTEX_API_URL, "") == (
        "https://aiplatform.googleapis.com"
    )


def test_vertex_target_for_location_honors_explicit_gateway() -> None:
    assert vertex_target_for_location("https://vertex-gateway.internal", "europe-west1") == (
        "https://vertex-gateway.internal"
    )
