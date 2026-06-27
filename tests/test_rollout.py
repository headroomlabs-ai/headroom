from __future__ import annotations

from headroom.rollout import ReleaseChannel, Rollout, feature_enabled


def test_unknown_channel_fails_closed_to_stable() -> None:
    rollout = Rollout.from_env({"HEADROOM_RELEASE_CHANNEL": "surprise"})

    assert rollout.channel is ReleaseChannel.STABLE


def test_stable_channel_blocks_canary_feature_even_when_requested() -> None:
    env = {
        "HEADROOM_RELEASE_CHANNEL": "stable",
        "HEADROOM_FEATURES": "tool-result-interceptors",
    }

    assert feature_enabled("tool_result_interceptors", environ=env) is False
    assert feature_enabled("tool_result_interceptors", explicit=True, environ=env) is False


def test_canary_channel_allows_explicit_canary_feature() -> None:
    env = {"HEADROOM_RELEASE_CHANNEL": "canary"}

    assert feature_enabled("tool_result_interceptors", explicit=True, environ=env) is True


def test_legacy_env_still_obeys_channel_gate() -> None:
    stable_env = {
        "HEADROOM_RELEASE_CHANNEL": "stable",
        "HEADROOM_INTERCEPT_ENABLED": "1",
    }
    canary_env = {
        "HEADROOM_RELEASE_CHANNEL": "canary",
        "HEADROOM_INTERCEPT_ENABLED": "1",
    }

    assert feature_enabled("tool_result_interceptors", environ=stable_env) is False
    assert feature_enabled("tool_result_interceptors", environ=canary_env) is True


def test_disable_list_wins_over_legacy_env_and_explicit_config() -> None:
    env = {
        "HEADROOM_RELEASE_CHANNEL": "canary",
        "HEADROOM_INTERCEPT_ENABLED": "1",
        "HEADROOM_DISABLE_FEATURES": "tool_result_interceptors",
    }

    assert feature_enabled("tool_result_interceptors", explicit=True, environ=env) is False


def test_unsafe_override_is_break_glass_for_lower_channels() -> None:
    env = {
        "HEADROOM_RELEASE_CHANNEL": "stable",
        "HEADROOM_FEATURES": "tool_result_interceptors",
        "HEADROOM_UNSAFE_ALLOW_UNSTABLE_FEATURES": "1",
    }

    assert feature_enabled("tool_result_interceptors", environ=env) is True
