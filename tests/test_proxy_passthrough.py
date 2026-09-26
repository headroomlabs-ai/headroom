from headroom.proxy.passthrough import custom_base_passthrough_telemetry, is_opencode_zen_base


def test_custom_base_passthrough_telemetry_recognizes_opencode_zen_chat() -> None:
    assert custom_base_passthrough_telemetry(
        "POST",
        "/zen/v1/chat/completions",
        "https://opencode.ai/",
    ) == ("chat/completions", "zen")
    assert custom_base_passthrough_telemetry(
        "POST",
        "zen/v1/chat/completions",
        "https://www.opencode.ai",
    ) == ("chat/completions", "zen")


def test_custom_base_passthrough_telemetry_ignores_non_matching_traffic() -> None:
    assert custom_base_passthrough_telemetry(
        "GET",
        "/zen/v1/chat/completions",
        "https://opencode.ai/",
    ) == ("", "")
    assert custom_base_passthrough_telemetry(
        "POST",
        "/v1/chat/completions",
        "https://opencode.ai/",
    ) == ("", "")
    assert custom_base_passthrough_telemetry(
        "POST",
        "/zen/v1/chat/completions",
        "https://custom.example/",
    ) == ("", "")
    assert custom_base_passthrough_telemetry(
        "POST",
        "/zen/v1/chat/completions",
        "://bad-url",
    ) == ("", "")


def test_is_opencode_zen_base_recognizes_zen_origins() -> None:
    assert is_opencode_zen_base("https://opencode.ai")
    assert is_opencode_zen_base("https://www.opencode.ai")
    assert is_opencode_zen_base("https://opencode.ai/zen")


def test_is_opencode_zen_base_rejects_other_or_missing_bases() -> None:
    assert not is_opencode_zen_base(None)
    assert not is_opencode_zen_base("")
    assert not is_opencode_zen_base("https://custom.example")
    assert not is_opencode_zen_base("https://opencode.ai.evil.example")
    assert not is_opencode_zen_base("://bad-url")


def test_custom_base_passthrough_telemetry_recognizes_xai_chat() -> None:
    # Grok Build routes plain OpenAI chat completions through custom-base
    # routing with base_url https://api.x.ai; attribute the provider as xai
    # so savings rollups separate grok traffic from OpenAI's.
    assert custom_base_passthrough_telemetry(
        "POST",
        "/v1/chat/completions",
        "https://api.x.ai",
    ) == ("chat/completions", "xai")
    # Other xai paths and methods stay unlabelled.
    assert custom_base_passthrough_telemetry(
        "GET",
        "/v1/chat/completions",
        "https://api.x.ai",
    ) == ("", "")
    assert custom_base_passthrough_telemetry(
        "POST",
        "/v1/models",
        "https://api.x.ai",
    ) == ("", "")
