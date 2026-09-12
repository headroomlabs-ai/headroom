from headroom.proxy.passthrough import custom_base_passthrough_telemetry


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


def test_custom_base_passthrough_telemetry_recognizes_factory_openai_chat() -> None:
    # Host-agnostic: any Factory gateway (public / enterprise / EU) tags factory.
    assert custom_base_passthrough_telemetry(
        "POST",
        "/api/llm/o/v1/chat/completions",
        "https://api.factory.ai",
    ) == ("chat/completions", "factory")
    assert custom_base_passthrough_telemetry(
        "POST",
        "api/llm/o/v1/chat/completions",
        "https://factory.example-enterprise.internal",
    ) == ("chat/completions", "factory")


def test_custom_base_passthrough_telemetry_ignores_factory_non_post() -> None:
    assert custom_base_passthrough_telemetry(
        "GET",
        "/api/llm/o/v1/chat/completions",
        "https://api.factory.ai",
    ) == ("", "")


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
