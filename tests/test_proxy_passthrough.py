from headroom.proxy.passthrough import (
    custom_base_passthrough_telemetry,
    provider_label_from_model,
)


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


def test_provider_label_from_model_classifies_deepseek() -> None:
    assert provider_label_from_model("deepseek-chat") == "deepseek"
    assert provider_label_from_model("deepseek-reasoner") == "deepseek"
    assert provider_label_from_model("DeepSeek-V4-Flash") == "deepseek"
    # Vendor-prefixed routing ids resolve to the underlying model.
    assert provider_label_from_model("deepseek/deepseek-chat") == "deepseek"
    assert provider_label_from_model("github/deepseek-v4") == "deepseek"


def test_provider_label_from_model_leaves_openai_models_unlabeled() -> None:
    for model in ("gpt-5.4-mini", "openai/gpt-4o", "o3", "zen", "", None):
        assert provider_label_from_model(model) == ""
