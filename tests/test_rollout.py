from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.config import HeadroomConfig
from headroom.rollout import (
    FEATURES,
    FeatureDecisionReason,
    FeatureSpec,
    RolloutChannel,
    RolloutConfigurationError,
    RolloutSnapshot,
    current_rollout,
    feature_enabled,
    registry_digest,
    resolve_rollout,
)
from headroom.transforms.pipeline import TransformPipeline


def test_default_stable_resolution_is_versioned_and_eligible() -> None:
    snapshot = resolve_rollout({})

    assert snapshot.channel is RolloutChannel.STABLE
    assert snapshot.schema_version == 1
    assert snapshot.policy_version == "1"
    assert snapshot.qualification_eligible is True


@pytest.mark.parametrize("channel", ["beta", "canary", "dev"])
def test_valid_rollout_channels(channel: str) -> None:
    assert resolve_rollout({"HEADROOM_ROLLOUT_CHANNEL": channel}).channel.value == channel


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("prod", RolloutChannel.STABLE),
        ("production", RolloutChannel.STABLE),
        ("preview", RolloutChannel.BETA),
        ("nightly", RolloutChannel.CANARY),
        ("development", RolloutChannel.DEV),
    ],
)
def test_channel_aliases(alias: str, expected: RolloutChannel) -> None:
    assert RolloutChannel.parse(alias) is expected


def test_strict_channel_configuration_rejects_unknown_input() -> None:
    with pytest.raises(RolloutConfigurationError, match="unknown rollout channel"):
        resolve_rollout({"HEADROOM_ROLLOUT_CHANNEL": "stabel"}, strict=True)


def test_unknown_channel_fails_closed_with_diagnostic(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        snapshot = resolve_rollout({"HEADROOM_ROLLOUT_CHANNEL": "stabel"})

    assert snapshot.channel is RolloutChannel.STABLE
    assert "unknown rollout channel 'stabel'; falling back to 'stable'" in caplog.text


def test_unknown_requested_and_disabled_features_fail_closed_and_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        snapshot = resolve_rollout(
            {
                "HEADROOM_FEATURES": "typo_requested",
                "HEADROOM_DISABLE_FEATURES": "typo_disabled",
            }
        )

    assert snapshot.config.requested == frozenset()
    assert snapshot.config.disabled == frozenset()
    assert "typo_requested" in caplog.text
    assert "typo_disabled" in caplog.text


def test_strict_configuration_rejects_unknown_input() -> None:
    with pytest.raises(RolloutConfigurationError, match="unknown rollout feature"):
        resolve_rollout({"HEADROOM_FEATURES": "typo"}, strict=True)


def test_stable_blocks_explicit_canary_feature() -> None:
    snapshot = resolve_rollout(
        {
            "HEADROOM_ROLLOUT_CHANNEL": "stable",
            "HEADROOM_FEATURES": "tool-result-interceptors",
        }
    )
    decision = snapshot.decision("tool_result_interceptors")

    assert decision.enabled is False
    assert decision.reason is FeatureDecisionReason.BLOCKED_BY_CHANNEL


def test_canary_allows_explicit_request() -> None:
    snapshot = resolve_rollout(
        {
            "HEADROOM_ROLLOUT_CHANNEL": "canary",
            "HEADROOM_FEATURES": "tool_result_interceptors",
        }
    )

    assert snapshot.decision("tool_result_interceptors").reason is FeatureDecisionReason.EXPLICIT


def test_non_default_feature_remains_off_when_not_requested() -> None:
    decision = resolve_rollout({"HEADROOM_ROLLOUT_CHANNEL": "dev"}).decision(
        "tool_result_interceptors"
    )
    assert decision.enabled is False
    assert decision.reason is FeatureDecisionReason.NOT_REQUESTED


def test_legacy_alias_obeys_channel_and_has_distinct_reason() -> None:
    stable = resolve_rollout(
        {"HEADROOM_ROLLOUT_CHANNEL": "stable", "HEADROOM_INTERCEPT_ENABLED": "1"}
    )
    canary = resolve_rollout(
        {"HEADROOM_ROLLOUT_CHANNEL": "canary", "HEADROOM_INTERCEPT_ENABLED": "1"}
    )

    assert (
        stable.decision("tool_result_interceptors").reason
        is FeatureDecisionReason.BLOCKED_BY_CHANNEL
    )
    assert canary.decision("tool_result_interceptors").reason is FeatureDecisionReason.LEGACY_ALIAS


@pytest.mark.parametrize("request_source", ["HEADROOM_FEATURES", "HEADROOM_INTERCEPT_ENABLED"])
def test_disable_beats_explicit_and_legacy_request(request_source: str) -> None:
    snapshot = resolve_rollout(
        {
            "HEADROOM_ROLLOUT_CHANNEL": "canary",
            request_source: "tool_result_interceptors"
            if request_source.endswith("FEATURES")
            else "1",
            "HEADROOM_DISABLE_FEATURES": "tool_result_interceptors",
        }
    )
    decision = snapshot.decision("tool_result_interceptors")

    assert decision.enabled is False
    assert decision.reason is FeatureDecisionReason.DISABLED


def test_unsafe_override_crosses_channel_and_poisons_qualification() -> None:
    snapshot = resolve_rollout(
        {
            "HEADROOM_ROLLOUT_CHANNEL": "stable",
            "HEADROOM_FEATURES": "tool_result_interceptors",
            "HEADROOM_UNSAFE_ALLOW_UNSTABLE_FEATURES": "1",
        }
    )
    payload = snapshot.to_dict()

    assert (
        snapshot.decision("tool_result_interceptors").reason
        is FeatureDecisionReason.UNSAFE_OVERRIDE
    )
    assert payload["qualification_eligible"] is False
    assert payload["qualification_ineligible_reason"] == "unsafe_rollout_override_active"


def test_disable_still_beats_unsafe_override() -> None:
    snapshot = resolve_rollout(
        {
            "HEADROOM_FEATURES": "tool_result_interceptors",
            "HEADROOM_DISABLE_FEATURES": "tool_result_interceptors",
            "HEADROOM_UNSAFE_ALLOW_UNSTABLE_FEATURES": "1",
        }
    )
    assert snapshot.decision("tool_result_interceptors").reason is FeatureDecisionReason.DISABLED


def test_live_legacy_reresolution_preserves_channel_and_named_kill_switch() -> None:
    eligible = resolve_rollout({"HEADROOM_ROLLOUT_CHANNEL": "beta"})
    enabled = eligible.with_legacy_env({"HEADROOM_OUTPUT_SHAPER": "1"})
    disabled_again = enabled.with_legacy_env({"HEADROOM_OUTPUT_SHAPER": "0"})
    killed = resolve_rollout(
        {
            "HEADROOM_ROLLOUT_CHANNEL": "beta",
            "HEADROOM_DISABLE_FEATURES": "proxy_output_shaper",
        }
    ).with_legacy_env({"HEADROOM_OUTPUT_SHAPER": "1"})
    # ``proxy_output_shaper`` is STABLE and default-on, so it can no longer
    # show a legacy alias being refused by the channel gate. ``read_maturation``
    # is still BETA, so it carries that half of the assertion.
    blocked = resolve_rollout({}).with_legacy_env({"HEADROOM_READ_MATURATION": "1"})

    assert enabled.decision("proxy_output_shaper").reason is FeatureDecisionReason.LEGACY_ALIAS
    assert disabled_again.decision("proxy_output_shaper").reason is FeatureDecisionReason.DISABLED
    assert killed.decision("proxy_output_shaper").reason is FeatureDecisionReason.DISABLED
    assert blocked.decision("read_maturation").reason is FeatureDecisionReason.BLOCKED_BY_CHANNEL
    assert enabled.snapshot_digest != eligible.snapshot_digest


def test_empty_programmatic_feature_names_are_ignored() -> None:
    snapshot = resolve_rollout({}, requested=["", "  "], disabled=[""])

    assert snapshot.config.requested == frozenset()
    assert snapshot.config.disabled == frozenset()


def test_multi_worker_config_round_trip_preserves_typed_rollout(monkeypatch) -> None:
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import (
        _MULTI_WORKER_CONFIG_ENV,
        _proxy_config_from_env,
        _proxy_config_payload,
    )

    rollout = resolve_rollout(
        {
            "HEADROOM_ROLLOUT_CHANNEL": "beta",
            "HEADROOM_OUTPUT_SHAPER": "1",
            "HEADROOM_DISABLE_FEATURES": "read_maturation",
        }
    )
    original = ProxyConfig(
        rollout=rollout,
        worker_processes=2,
        profile_seeded_env_keys=frozenset({"HEADROOM_MODE", "HEADROOM_LOSSLESS"}),
    )
    monkeypatch.setenv(_MULTI_WORKER_CONFIG_ENV, json.dumps(_proxy_config_payload(original)))

    restored = _proxy_config_from_env()

    assert restored.rollout is not None
    assert restored.rollout.to_internal_dict() == rollout.to_internal_dict()
    assert restored.rollout.is_enabled("proxy_output_shaper") is True
    assert restored.rollout.is_enabled("read_maturation") is False
    assert restored.worker_processes == 2
    assert restored.profile_seeded_env_keys == frozenset({"HEADROOM_MODE", "HEADROOM_LOSSLESS"})


def test_documented_proxy_json_without_internal_snapshot_is_preserved(monkeypatch) -> None:
    from headroom.proxy.server import _MULTI_WORKER_CONFIG_ENV, _proxy_config_from_env

    monkeypatch.setenv(
        _MULTI_WORKER_CONFIG_ENV,
        json.dumps(
            {
                "port": 39099,
                "rate_limit_enabled": False,
                "proxy_token": "required-token",
                "offline": True,
            }
        ),
    )

    restored = _proxy_config_from_env()

    assert restored.port == 39099
    assert restored.rate_limit_enabled is False
    assert restored.proxy_token == "required-token"
    assert restored.offline is True
    assert restored.rollout is not None


def test_internal_proxy_json_still_rejects_tampered_rollout_snapshot(monkeypatch) -> None:
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import (
        _MULTI_WORKER_CONFIG_ENV,
        _proxy_config_from_env,
        _proxy_config_payload,
    )

    payload = _proxy_config_payload(ProxyConfig(port=39099))
    payload["_rollout_snapshot"]["snapshot_digest"] = "sha256:tampered"  # type: ignore[index]
    monkeypatch.setenv(_MULTI_WORKER_CONFIG_ENV, json.dumps(payload))
    monkeypatch.setenv("HEADROOM_PORT", "39100")

    restored = _proxy_config_from_env()

    assert restored.port == 39100
    assert restored.rollout is not None


@pytest.mark.parametrize("raw_config", ["null", "[]", '"not-an-object"'])
def test_non_object_proxy_json_falls_back_without_crashing(monkeypatch, raw_config: str) -> None:
    from headroom.proxy.server import _MULTI_WORKER_CONFIG_ENV, _proxy_config_from_env

    monkeypatch.setenv(_MULTI_WORKER_CONFIG_ENV, raw_config)
    monkeypatch.setenv("HEADROOM_PORT", "39100")

    restored = _proxy_config_from_env()

    assert restored.port == 39100
    assert restored.rollout is not None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema_version": 999}, "schema version"),
        ({"policy_version": "999"}, "policy version"),
        ({"unsafe_allow_unstable": "yes"}, "unsafe override"),
        ({"explicit_requested": "proxy_output_shaper"}, "explicit_requested"),
        ({"registry_digest": "sha256:tampered"}, "registry digest"),
        ({"snapshot_digest": "sha256:tampered"}, "snapshot digest"),
    ],
)
def test_worker_rollout_handoff_rejects_invalid_or_tampered_state(
    mutation: dict[str, object], message: str
) -> None:
    payload = resolve_rollout({}).to_internal_dict()
    payload.update(mutation)

    with pytest.raises(RolloutConfigurationError, match=message):
        RolloutSnapshot.from_internal_dict(payload)


def test_worker_rollout_handoff_rejects_non_object_state() -> None:
    with pytest.raises(RolloutConfigurationError, match="invalid rollout worker snapshot"):
        RolloutSnapshot.from_internal_dict([])  # type: ignore[arg-type]


def test_snapshot_query_and_compatibility_helpers() -> None:
    snapshot = current_rollout(
        {
            "HEADROOM_ROLLOUT_CHANNEL": "canary",
            "HEADROOM_FEATURES": "tool_result_interceptors",
            "HEADROOM_DISABLE_FEATURES": "read_maturation",
        }
    )

    assert snapshot.is_available("tool-result-interceptors") is True
    assert snapshot.enabled == frozenset({"tool_result_interceptors"})
    assert snapshot.disabled == frozenset({"read_maturation"})
    assert feature_enabled(
        "tool_result_interceptors",
        explicit=True,
        environ={"HEADROOM_ROLLOUT_CHANNEL": "canary"},
    )
    assert not feature_enabled("tool_result_interceptors", environ={})
    with pytest.raises(KeyError, match="missing"):
        snapshot.decision("missing")


def test_registry_and_snapshot_digests_are_deterministic_and_policy_sensitive() -> None:
    first = resolve_rollout({"HEADROOM_ROLLOUT_CHANNEL": "canary"})
    second = resolve_rollout({"HEADROOM_ROLLOUT_CHANNEL": "canary"})
    equivalent = dict(reversed(list(FEATURES.items())))
    changed = dict(FEATURES)
    changed["tool_result_interceptors"] = FeatureSpec(
        "tool_result_interceptors", RolloutChannel.BETA
    )

    assert first.registry_digest == second.registry_digest == registry_digest(equivalent)
    assert first.snapshot_digest == second.snapshot_digest
    assert registry_digest(changed) != first.registry_digest
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(
        second.to_dict(), sort_keys=True
    )


def test_pipeline_uses_config_snapshot_after_environment_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HEADROOM_ROLLOUT_CHANNEL", raising=False)
    monkeypatch.delenv("HEADROOM_FEATURES", raising=False)
    config = HeadroomConfig()
    original_digest = config.rollout.snapshot_digest if config.rollout else None
    monkeypatch.setenv("HEADROOM_ROLLOUT_CHANNEL", "canary")
    monkeypatch.setenv("HEADROOM_FEATURES", "tool_result_interceptors")

    pipeline = TransformPipeline(config)

    assert config.rollout is not None
    assert config.rollout.snapshot_digest == original_digest
    assert all(
        type(transform).__name__ != "ToolResultInterceptorTransform"
        for transform in pipeline.transforms
    )


def test_cli_json_status_and_strict_error() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "rollout",
            "status",
            "--channel",
            "canary",
            "--features",
            "tool_result_interceptors",
            "--json",
        ],
    )
    invalid = runner.invoke(main, ["rollout", "status", "--features", "typo", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["channel"] == "canary"
    assert payload["features"][2]["name"] == "tool_result_interceptors"
    assert invalid.exit_code != 0
    assert "unknown rollout feature" in invalid.output


def test_cli_human_status_exercises_disable_and_unsafe_options() -> None:
    result = CliRunner().invoke(
        main,
        [
            "rollout",
            "status",
            "--channel",
            "stable",
            "--features",
            "tool_result_interceptors",
            "--disable-features",
            "read_maturation",
            "--unsafe-allow-unstable-features",
        ],
    )

    assert result.exit_code == 0
    assert "Rollout channel: stable" in result.output
    assert "Qualification eligible: false" in result.output
    assert "tool_result_interceptors: enabled=true decision=unsafe_override" in result.output
    assert "read_maturation: enabled=false decision=disabled" in result.output


@pytest.mark.parametrize(
    ("option", "message", "required_channel"),
    [
        ("--read-maturation", "--read-maturation is not available", "beta"),
        (
            "--intercept-tool-results",
            "--intercept-tool-results is not available",
            "canary",
        ),
    ],
)
def test_proxy_cli_fails_loudly_when_explicit_feature_is_channel_blocked(
    option: str, message: str, required_channel: str
) -> None:
    result = CliRunner().invoke(
        main,
        ["proxy", option],
        env={"HEADROOM_ROLLOUT_CHANNEL": "stable"},
    )

    assert result.exit_code == 1
    assert message in result.output
    assert f"HEADROOM_ROLLOUT_CHANNEL={required_channel}" in result.output


def test_shared_python_rust_policy_vectors() -> None:
    vectors = json.loads(
        (Path(__file__).parent / "fixtures" / "rollout_policy_vectors.json").read_text()
    )
    for vector in vectors:
        env = {"HEADROOM_ROLLOUT_CHANNEL": vector["channel"]}
        if vector["requested"]:
            env["HEADROOM_FEATURES"] = "tool_result_interceptors"
        if vector["disabled"]:
            env["HEADROOM_DISABLE_FEATURES"] = "tool_result_interceptors"
        if vector["unsafe"]:
            env["HEADROOM_UNSAFE_ALLOW_UNSTABLE_FEATURES"] = "1"
        decision = resolve_rollout(env).decision("tool_result_interceptors")
        assert decision.enabled is vector["enabled"]
        assert decision.reason.value == vector["decision"]
