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


def test_custom_base_passthrough_telemetry_names_known_chat_hosts() -> None:
    # Exact hosts only, fixed labels: the taxonomy, not the request, decides.
    assert custom_base_passthrough_telemetry(
        "POST",
        "/v1/chat/completions",
        "https://api.z.ai/api/coding/paas/v4",
    ) == ("chat/completions", "zai")
    assert custom_base_passthrough_telemetry(
        "POST",
        "v1/chat/completions",
        "https://api.meta.ai/v1",
    ) == ("chat/completions", "meta")
    assert custom_base_passthrough_telemetry(
        "POST",
        "/v1/chat/completions",
        "https://api.openai.com/v1",
    ) == ("chat/completions", "openai")
    # Host parsing is case-insensitive; the label stays fixed.
    assert custom_base_passthrough_telemetry(
        "POST",
        "/v1/chat/completions",
        "https://API.Z.AI/api/coding/paas/v4",
    ) == ("chat/completions", "zai")


def test_custom_base_passthrough_telemetry_keeps_everything_else_unnamed() -> None:
    # Lookalike hosts never match: an exact-host set cannot be talked into
    # naming an attacker-controlled upstream.
    assert custom_base_passthrough_telemetry(
        "POST",
        "/v1/chat/completions",
        "https://api.z.ai.evil.test/v1",
    ) == ("", "")
    # Unknown hosts, non-chat paths, non-POST methods, and bad URLs all stay
    # unnamed so nothing request-controlled becomes a telemetry label.
    assert custom_base_passthrough_telemetry(
        "POST",
        "/v1/chat/completions",
        "https://llm.example.internal/v1",
    ) == ("", "")
    assert custom_base_passthrough_telemetry(
        "POST",
        "/v1/embeddings",
        "https://api.z.ai/v1",
    ) == ("", "")
    assert custom_base_passthrough_telemetry(
        "GET",
        "/v1/chat/completions",
        "https://api.z.ai/v1",
    ) == ("", "")
    assert custom_base_passthrough_telemetry(
        "POST",
        "/v1/chat/completions",
        "://bad-url",
    ) == ("", "")


def test_custom_base_passthrough_telemetry_requires_a_chat_path_segment() -> None:
    # A bare suffix match would also name ``notchat/completions``.
    for path in ("/v1/notchat/completions", "/v1/xchat/completions", "notchat/completions"):
        assert custom_base_passthrough_telemetry("POST", path, "https://api.z.ai/v1") == ("", "")
    # The exact segment, bare or nested, still names the host.
    for path in ("chat/completions", "/chat/completions", "/api/paas/v4/chat/completions"):
        assert custom_base_passthrough_telemetry("POST", path, "https://api.z.ai/v1") == (
            "chat/completions",
            "zai",
        )
