from __future__ import annotations

from headroom.providers.cloudcode import normalize_cloudcode_passthrough_path


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
