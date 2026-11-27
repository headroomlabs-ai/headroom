from __future__ import annotations

import pytest

from headroom.providers.cloudcode import (
    ANTIGRAVITY_DAILY_API_URL,
    cloudcode_passthrough_target,
    is_antigravity_user_agent,
    normalize_cloudcode_passthrough_path,
)


def test_normalize_cloudcode_passthrough_path_accepts_internal_routes() -> None:
    assert normalize_cloudcode_passthrough_path("v1internal:loadCodeAssist") == (
        "/v1internal:loadCodeAssist"
    )
    assert normalize_cloudcode_passthrough_path("/v1internal:loadCodeAssist") == (
        "/v1internal:loadCodeAssist"
    )


def test_normalize_cloudcode_passthrough_path_strips_version_prefix() -> None:
    assert normalize_cloudcode_passthrough_path("/v1/v1internal:fetchAvailableModels") == (
        "/v1internal:fetchAvailableModels"
    )


def test_normalize_cloudcode_passthrough_path_ignores_unrelated_paths() -> None:
    assert normalize_cloudcode_passthrough_path("/unrelated/v1internal:loadCodeAssist") is None
    assert normalize_cloudcode_passthrough_path("/v1beta/models") is None


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        ("Antigravity/1.2.3 Darwin/arm64", True),
        ("antigravity/1.0.0", True),
        ("google-api-nodejs-client/9.15.1", False),
        ("notantigravity/1.0", False),
        ("", False),
        (None, False),
    ],
)
def test_is_antigravity_user_agent(user_agent: str | None, expected: bool) -> None:
    assert is_antigravity_user_agent(user_agent) is expected


def test_cloudcode_passthrough_target_routes_antigravity_to_daily_host() -> None:
    assert ANTIGRAVITY_DAILY_API_URL == "https://daily-cloudcode-pa.sandbox.googleapis.com"
    assert cloudcode_passthrough_target("Antigravity/1.2.3", "https://cloudcode.example.test") == (
        ANTIGRAVITY_DAILY_API_URL
    )


def test_cloudcode_passthrough_target_keeps_default_for_other_clients() -> None:
    default = "https://cloudcode.example.test"
    assert cloudcode_passthrough_target("google-api-nodejs-client/9.15.1", default) == default
    assert cloudcode_passthrough_target(None, default) == default
