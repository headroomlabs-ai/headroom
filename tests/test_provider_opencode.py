from __future__ import annotations

from headroom.providers.opencode import (
    MANAGED_PROVIDERS,
    apply_provider_overrides,
    build_provider_overrides,
    config_has_headroom_overrides,
    is_headroom_base_url,
    proxy_base_url,
    render_provider_config,
    render_setup_lines,
    strip_provider_overrides,
)


def test_opencode_proxy_base_url_is_openai_compatible() -> None:
    assert proxy_base_url(8787) == "http://127.0.0.1:8787/v1"


def test_opencode_proxy_base_url_uses_given_port() -> None:
    assert proxy_base_url(9999) == "http://127.0.0.1:9999/v1"


def test_opencode_build_provider_overrides_sets_both_providers() -> None:
    overrides = build_provider_overrides(8787)
    assert set(overrides) == set(MANAGED_PROVIDERS) == {"anthropic", "openai"}
    for name in MANAGED_PROVIDERS:
        assert overrides[name]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"


def test_opencode_build_provider_overrides_applies_project_prefix() -> None:
    overrides = build_provider_overrides(9999, project="myrepo")
    assert overrides["openai"]["options"]["baseURL"] == "http://127.0.0.1:9999/p/myrepo/v1"
    assert overrides["anthropic"]["options"]["baseURL"] == "http://127.0.0.1:9999/p/myrepo/v1"


def test_opencode_build_provider_overrides_ignores_blank_project() -> None:
    overrides = build_provider_overrides(9999, project="   ")
    assert overrides["openai"]["options"]["baseURL"] == "http://127.0.0.1:9999/v1"


def test_opencode_apply_overrides_on_empty_config() -> None:
    result = apply_provider_overrides({}, 8787)
    assert result == {
        "provider": {
            "anthropic": {"options": {"baseURL": "http://127.0.0.1:8787/v1"}},
            "openai": {"options": {"baseURL": "http://127.0.0.1:8787/v1"}},
        }
    }


def test_opencode_apply_overrides_does_not_mutate_input() -> None:
    source: dict = {"model": "anthropic/claude-sonnet-4-5"}
    apply_provider_overrides(source, 8787)
    assert source == {"model": "anthropic/claude-sonnet-4-5"}


def test_opencode_apply_overrides_preserves_user_provider_settings() -> None:
    source = {
        "model": "openai/gpt-4o",
        "provider": {
            "openai": {
                "options": {"apiKey": "{env:OPENAI_API_KEY}"},
                "models": {"gpt-4o": {"name": "GPT-4o"}},
            },
        },
    }
    result = apply_provider_overrides(source, 8787)
    openai = result["provider"]["openai"]
    # baseURL is injected, but the user's apiKey + models survive.
    assert openai["options"]["baseURL"] == "http://127.0.0.1:8787/v1"
    assert openai["options"]["apiKey"] == "{env:OPENAI_API_KEY}"
    assert openai["models"] == {"gpt-4o": {"name": "GPT-4o"}}
    # Untouched top-level keys are kept.
    assert result["model"] == "openai/gpt-4o"


def test_opencode_apply_overrides_is_idempotent_across_port_change() -> None:
    once = apply_provider_overrides({}, 8787)
    twice = apply_provider_overrides(once, 9999)
    # Only the new port's baseURL remains — no stale 8787 entry.
    assert twice["provider"]["openai"]["options"]["baseURL"] == "http://127.0.0.1:9999/v1"
    assert twice["provider"]["anthropic"]["options"]["baseURL"] == "http://127.0.0.1:9999/v1"
    # And re-applying did not nest or duplicate provider entries.
    assert set(twice["provider"]) == {"anthropic", "openai"}


def test_opencode_strip_overrides_round_trips_to_empty() -> None:
    wrapped = apply_provider_overrides({}, 8787)
    assert strip_provider_overrides(wrapped) == {}


def test_opencode_strip_overrides_preserves_user_baseurl() -> None:
    source = {
        "provider": {
            "openai": {"options": {"baseURL": "https://my-gateway.example.com/v1"}},
        },
    }
    # A non-Headroom baseURL must never be stripped.
    assert strip_provider_overrides(source) == source


def test_opencode_strip_overrides_keeps_sibling_provider_keys() -> None:
    source = {
        "provider": {
            "openai": {
                "options": {"baseURL": "http://127.0.0.1:8787/v1", "apiKey": "{env:KEY}"},
                "models": {"gpt-4o": {}},
            },
        },
    }
    result = strip_provider_overrides(source)
    assert result == {
        "provider": {
            "openai": {"options": {"apiKey": "{env:KEY}"}, "models": {"gpt-4o": {}}},
        },
    }


def test_opencode_strip_overrides_handles_project_prefixed_url() -> None:
    wrapped = apply_provider_overrides({}, 8787, project="myrepo")
    assert strip_provider_overrides(wrapped) == {}


def test_opencode_config_has_headroom_overrides_detects_wrapped() -> None:
    assert config_has_headroom_overrides(apply_provider_overrides({}, 8787)) is True
    assert config_has_headroom_overrides({}) is False
    assert (
        config_has_headroom_overrides(
            {"provider": {"openai": {"options": {"baseURL": "https://api.openai.com/v1"}}}}
        )
        is False
    )


def test_opencode_is_headroom_base_url() -> None:
    assert is_headroom_base_url("http://127.0.0.1:8787/v1") is True
    assert is_headroom_base_url("http://127.0.0.1:9999/p/my-repo/v1") is True
    assert is_headroom_base_url("https://api.openai.com/v1") is False
    assert is_headroom_base_url("http://127.0.0.1:8787") is False
    assert is_headroom_base_url(None) is False


def test_opencode_render_provider_config_is_valid_json() -> None:
    import json

    config = json.loads(render_provider_config(8787, project="demo"))
    assert config["$schema"] == "https://opencode.ai/config.json"
    assert config["provider"]["openai"]["options"]["baseURL"] == "http://127.0.0.1:8787/p/demo/v1"


def test_opencode_render_setup_lines_contains_proxy_url() -> None:
    joined = "\n".join(render_setup_lines(8787))
    assert "http://127.0.0.1:8787/v1" in joined
    assert "opencode.json" in joined
    assert "OpenCode" in joined


def test_opencode_render_setup_lines_project_attribution() -> None:
    joined = "\n".join(render_setup_lines(8787, project="my-project"))
    assert "my-project" in joined
    plain = "\n".join(render_setup_lines(8787))
    assert "attributed" not in plain
