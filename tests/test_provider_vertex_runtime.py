from __future__ import annotations

from urllib.parse import urlsplit

import pytest

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


_VERTEX_PUBLIC_ENDPOINT = "https://aiplatform.googleapis.com"


@pytest.mark.parametrize(
    "region",
    ["us-central1", "europe-west4", "asia-northeast1", "me-central1", "us-east5"],
)
def test_vertex_target_for_location_accepts_real_regions(region: str) -> None:
    assert vertex_target_for_location(DEFAULT_VERTEX_API_URL, region) == (
        f"https://{region}-aiplatform.googleapis.com"
    )


@pytest.mark.parametrize(
    "malicious",
    [
        "169.254.169.254#",  # fragment delimiter -> cloud metadata IP (the reported PoC)
        "127.0.0.1:44919#",  # host:port + fragment
        "169.254.169.254/latest/meta-data/iam#",  # path injection
        "169.254.169.254:80",  # port injection
        "evil.example",  # dotted host
        "foo@evil.example",  # userinfo delimiter
        "us_central1",  # underscore (not a region)
        "US-CENTRAL1",  # uppercase
        "-leading",  # leading hyphen
        "trailing-",  # trailing hyphen
        "a--b",  # empty hyphen group
    ],
)
def test_vertex_target_for_location_rejects_ssrf_payloads(malicious: str) -> None:
    """A non-region ``location`` must never carry an attacker-chosen host into
    the interpolated Vertex hostname (CWE-918). It falls back to the public
    endpoint, and the parsed host is always the legitimate Vertex host — never
    a metadata IP, loopback, or injected authority.
    """
    target = vertex_target_for_location(DEFAULT_VERTEX_API_URL, malicious)
    assert target == _VERTEX_PUBLIC_ENDPOINT
    parsed = urlsplit(target)
    assert parsed.hostname == "aiplatform.googleapis.com"
    assert parsed.port is None


def test_vertex_target_for_location_ssrf_fallback_only_on_default_target() -> None:
    """A validated non-region value still cannot override an explicitly
    configured gateway (that path returns the operator's target verbatim and
    is not user-derived)."""
    assert vertex_target_for_location("https://gw.internal", "169.254.169.254#") == (
        "https://gw.internal"
    )
