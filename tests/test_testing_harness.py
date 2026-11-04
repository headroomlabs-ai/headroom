from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from headroom.config import HeadroomConfig
from headroom.proxy.models import ProxyConfig
from headroom.testing import (
    AgentEvalsPricing,
    ArmName,
    Configurator,
    GuaranteeResult,
    Headroom,
    ProviderTarget,
    ScenarioOrchestrator,
    ScenarioTask,
)

AGENT_EVALS_RUN_MANIFEST_FIELDS = {
    "experiment_id",
    "created_at",
    "headroom_git_sha",
    "agent_evals_git_sha",
    "model_snapshot",
    "provider",
    "auth_mode",
    "benchmark",
    "benchmark_ref",
    "harness",
    "harness_version",
    "docker_digests",
    "arms",
    "k_runs",
    "temperature",
    "seeds",
    "alpha",
    "margins",
    "pricing",
}


def _configure_bedrock_apple(c: Configurator) -> None:
    c.kompress_enabled = False
    c.mode = "cache"
    c.default_mode = "optimize"


def test_contract_covers_current_headroom_and_proxy_config_fields() -> None:
    scenario = Headroom.scenario("contract").build()

    assert scenario.contract.field_names("headroom") == set(HeadroomConfig.__dataclass_fields__)
    assert scenario.contract.field_names("proxy") == set(ProxyConfig.__dataclass_fields__)


def test_fluent_builder_configures_real_proxy_and_sdk_configs() -> None:
    scenario = (
        Headroom.WithBedrock(region="us-east-1", profile="bench")
        .named("bedrock-apple")
        .OnAppleSilicon()
        .configure(_configure_bedrock_apple)
        .configure_proxy(savings_profile="coding", min_tokens_to_crush=10)
        .build()
    )

    assert scenario.provider is ProviderTarget.BEDROCK
    assert scenario.proxy_config.backend == "bedrock"
    assert scenario.proxy_config.bedrock_region == "us-east-1"
    assert scenario.proxy_config.bedrock_profile == "bench"
    assert scenario.proxy_config.disable_kompress is True
    assert scenario.proxy_config.mode == "cache"
    assert scenario.headroom_config.default_mode.value == "optimize"

    command = scenario.proxy_command(port=18800)
    assert command[:4] == ("headroom", "proxy", "--port", "18800")
    assert "--backend" in command
    assert "bedrock" in command
    assert "--disable-kompress" in command
    assert "--savings-profile" not in command

    env = scenario.env()
    assert env["HEADROOM_BACKEND"] == "bedrock"
    assert env["HEADROOM_DISABLE_KOMPRESS"] == "1"
    assert env["HEADROOM_BEDROCK_REGION"] == "us-east-1"
    assert env["AWS_PROFILE"] == "bench"


def test_unknown_config_field_fails_fast() -> None:
    with pytest.raises(AttributeError, match="unknown Headroom harness config field"):
        Headroom.scenario().configure(not_a_real_knob=True)


def test_bench_manifest_fragment_matches_headroom_bench_arm_shape() -> None:
    scenario = (
        Headroom.with_openai()
        .configure(mode="cache", kompress_enabled=False)
        .configure_proxy(savings_profile="coding")
        .build()
    )

    fragment = scenario.bench_manifest_fragment(provider="openai").to_dict()

    assert fragment["harness"] == "headroom.testing"
    assert [arm["name"] for arm in fragment["arms"]] == [
        ArmName.A0_DIRECT.value,
        ArmName.A1_PASSTHROUGH.value,
        ArmName.B_HEADROOM.value,
    ]
    assert fragment["arms"][0]["proxy_mode"] is None
    assert fragment["arms"][1]["proxy_mode"] == "off"
    assert fragment["arms"][2]["proxy_mode"] == "cache"
    assert fragment["arms"][2]["proxy_flags"] == ["--disable-kompress"]
    assert fragment["env"]["HEADROOM_MODE"] == "cache"
    assert fragment["env"]["HEADROOM_SAVINGS_PROFILE"] == "coding"
    assert fragment["deployment_plan"]["config_env_var"] == "HEADROOM_PROXY_CONFIG_JSON"
    assert fragment["contract_audit"]["passed"] is True
    json.dumps(fragment)


def test_agent_evals_manifest_matches_run_manifest_contract() -> None:
    now = datetime(2026, 6, 15, 9, 30, tzinfo=timezone.utc)
    scenario = (
        Headroom.with_openai()
        .named("openai-cache")
        .WithCompression(mode="cache", kompress=False)
        .Build()
    )

    manifest = scenario.agent_evals_manifest(
        benchmark="mini_swebench",
        benchmark_ref="mini@abc123",
        provider="openai",
        now=now,
        model_snapshot="openai/gpt-4o",
        headroom_repo_path="/nonexistent-headroom",
        agent_evals_repo_path="/nonexistent-agent-evals",
        k_runs=3,
        pricing=AgentEvalsPricing(input_usd_per_1m=2.5, output_usd_per_1m=10.0),
    )
    payload = manifest.to_dict()

    assert set(payload) == AGENT_EVALS_RUN_MANIFEST_FIELDS
    assert payload["experiment_id"] == "mini_swebench-openai-cache-20260615T093000Z"
    assert payload["created_at"] == "2026-06-15T09:30:00+00:00"
    assert payload["headroom_git_sha"] == "unknown"
    assert payload["agent_evals_git_sha"] == "unknown"
    assert payload["provider"] == "openai"
    assert payload["benchmark"] == "mini_swebench"
    assert payload["benchmark_ref"] == "mini@abc123"
    assert payload["harness"] == "headroom.testing"
    assert payload["model_snapshot"] == "openai/gpt-4o"
    assert payload["seeds"] == [0, 1, 2]
    assert payload["margins"] == {"ccr": 0.0, "lossy": 2.0}
    assert payload["pricing"] == {"input_usd_per_1m": 2.5, "output_usd_per_1m": 10.0}
    assert [arm["name"] for arm in payload["arms"]] == [
        "a0_direct",
        "a1_passthrough",
        "b_headroom",
    ]
    assert payload["arms"][2]["proxy_mode"] == "cache"
    assert payload["arms"][2]["proxy_flags"] == ["--disable-kompress"]
    json.dumps(payload)


def test_sdk_simulation_runs_without_provider_api_keys() -> None:
    scenario = Headroom.with_openai().configure(default_mode="optimize").build()
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Summarize this small payload."},
    ]

    result = scenario.simulate(messages, model="gpt-4o")

    assert result.tokens_before >= result.tokens_after
    assert result.tokens_saved >= 0
    assert result.messages


def test_deployment_plan_carries_full_proxy_config_payload_through_env() -> None:
    scenario = (
        Headroom.WithBedrock(region="us-east-2", profile="bench")
        .Configure(mode="cache", kompress_enabled=False)
        .ConfigureProxy(memory_enabled=True, memory_top_k=3, offline=True)
        .Build()
    )

    plan = scenario.deployment_plan(port=18888)
    payload_from_env = json.loads(plan.env["HEADROOM_PROXY_CONFIG_JSON"])

    assert plan.command[:4] == ("headroom", "proxy", "--port", "18888")
    assert payload_from_env == plan.config_payload
    assert plan.env["HEADROOM_SKIP_UPSTREAM_CHECK"] == "1"
    assert plan.config_payload["backend"] == "bedrock"
    assert plan.config_payload["memory_enabled"] is True
    assert plan.config_payload["memory_top_k"] == 3
    assert plan.config_payload["offline"] is True
    assert set(plan.config_payload) == set(ProxyConfig.__dataclass_fields__)


def test_contract_audit_reports_full_payload_coverage_and_proxy_only_notes() -> None:
    scenario = (
        Headroom.WithBedrock(region="us-east-1")
        .WithReadMaturation(enabled=True, quiesce_turns=2)
        .Build()
    )

    audit = scenario.audit_contract()

    assert audit.passed is True
    assert audit.missing_headroom_payload_fields == ()
    assert audit.missing_proxy_payload_fields == ()
    assert audit.extra_headroom_payload_fields == ()
    assert audit.extra_proxy_payload_fields == ()
    assert audit.headroom_fields_total == len(HeadroomConfig.__dataclass_fields__)
    assert audit.proxy_fields_total == len(ProxyConfig.__dataclass_fields__)
    assert "read_maturation is currently a proxy-only surface" in audit.notes
    json.dumps(audit.to_dict())


def test_write_manifest_fragment_outputs_json_file(tmp_path: Path) -> None:
    path = tmp_path / "headroom-manifest-fragment.json"
    scenario = (
        Headroom.WithOpenAI(api_url="https://openai.internal")
        .WithCompression(mode="cache", kompress=False)
        .Build()
    )

    written = scenario.write_manifest_fragment(path, provider="openai")
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert written == path
    assert payload["harness"] == "headroom.testing"
    assert payload["provider"] == "openai"
    assert (
        payload["deployment_plan"]["config_payload"]["openai_api_url"] == "https://openai.internal"
    )
    assert payload["contract_audit"]["passed"] is True


def test_feature_facets_configure_authoritative_scenario_surfaces() -> None:
    scenario = (
        Headroom.WithAnthropic(api_url="https://anthropic.internal")
        .named("enterprise-feature-matrix")
        .WithCompression(
            mode="cache",
            kompress=False,
            lossless=True,
            compressors=["smart_crusher", "log", "diff"],
            min_tokens=25,
            max_items=9,
            savings_profile="coding",
        )
        .WithCCR(
            enabled=True,
            inject_tool=False,
            inject_marker=True,
            handle_responses=True,
            proactive_expansion=False,
            max_retrieval_rounds=1,
        )
        .WithCache(enabled=True, semantic=True, ttl_seconds=120, max_entries=33)
        .WithPrefixFreeze(enabled=False, session_ttl_seconds=42)
        .WithReadMaturation(enabled=True, quiesce_turns=2, max_hold_turns=8, min_size_bytes=512)
        .WithMemory(
            enabled=True,
            backend="local",
            mode="tool",
            top_k=4,
            min_similarity=0.5,
            inject_tools=False,
            inject_context=False,
            storage_mode="project",
        )
        .Build()
    )

    assert scenario.proxy_config.anthropic_api_url == "https://anthropic.internal"
    assert scenario.proxy_config.disable_kompress is True
    assert scenario.proxy_config.lossless is True
    assert scenario.proxy_config.compressors == {"smart_crusher", "log", "diff"}
    assert scenario.proxy_config.min_tokens_to_crush == 25
    assert scenario.proxy_config.max_items_after_crush == 9
    assert scenario.headroom_config.smart_crusher.lossless_only is True
    assert scenario.headroom_config.smart_crusher.min_tokens_to_crush == 25
    assert scenario.headroom_config.smart_crusher.max_items_after_crush == 9
    assert scenario.proxy_config.ccr_inject_tool is False
    assert scenario.proxy_config.ccr_inject_marker is True
    assert scenario.proxy_config.ccr_proactive_expansion is False
    assert scenario.proxy_config.ccr_max_retrieval_rounds == 1
    assert scenario.headroom_config.ccr.enabled is True
    assert scenario.headroom_config.ccr.inject_tool is False
    assert scenario.headroom_config.ccr.inject_retrieval_marker is True
    assert scenario.proxy_config.cache_ttl_seconds == 120
    assert scenario.proxy_config.cache_max_entries == 33
    assert scenario.headroom_config.cache_optimizer.enable_semantic_cache is True
    assert scenario.proxy_config.prefix_freeze_enabled is False
    assert scenario.headroom_config.prefix_freeze.enabled is False
    assert scenario.proxy_config.read_maturation is True
    assert scenario.metadata["read_maturation"]["quiesce_turns"] == 2
    assert scenario.proxy_config.memory_enabled is True
    assert scenario.proxy_config.memory_mode == "tool"
    assert scenario.proxy_config.memory_top_k == 4
    assert scenario.proxy_config.memory_min_similarity == 0.5
    assert scenario.proxy_config.memory_inject_tools is False
    assert scenario.proxy_config.memory_inject_context is False

    payload = scenario.deployment_plan(port=18889).config_payload
    assert payload["compressors"] == ["diff", "log", "smart_crusher"]
    assert payload["read_maturation"] is True
    assert payload["memory_mode"] == "tool"


@pytest.mark.parametrize(
    ("builder", "expected_provider", "expected_backend"),
    [
        (
            lambda: Headroom.WithAnthropic(api_url="https://anthropic.internal"),
            "anthropic",
            "anthropic",
        ),
        (lambda: Headroom.WithOpenAI(api_url="https://openai.internal"), "openai", "anthropic"),
        (lambda: Headroom.WithGemini(api_url="https://gemini.internal"), "gemini", "anthropic"),
        (
            lambda: Headroom.WithCloudCode(api_url="https://cloudcode.internal"),
            "cloudcode",
            "anthropic",
        ),
        (
            lambda: Headroom.WithVertex(api_url="https://vertex.internal"),
            "vertex",
            "litellm-vertex",
        ),
        (lambda: Headroom.WithBedrock(region="us-east-1"), "bedrock", "bedrock"),
        (lambda: Headroom.WithAnyLLM(provider="mistral"), "anyllm", "anyllm"),
        (lambda: Headroom.WithLiteLLM(provider="openrouter"), "litellm", "litellm-openrouter"),
    ],
)
def test_provider_builders_cover_current_proxy_targets(
    builder: object,
    expected_provider: str,
    expected_backend: str,
) -> None:
    scenario = builder().WithCompression(mode="cache").Build()  # type: ignore[operator]
    plan = scenario.deployment_plan(port=18901)

    assert scenario.provider.value == expected_provider
    assert plan.config_payload["backend"] == expected_backend
    assert plan.command[:4] == ("headroom", "proxy", "--port", "18901")
    assert json.loads(plan.env["HEADROOM_PROXY_CONFIG_JSON"]) == plan.config_payload


def test_scenario_orchestrator_runs_multiple_scenarios_and_reports_guarantees() -> None:
    passthrough = (
        Headroom.with_openai()
        .named("passthrough")
        .Configure(optimize=False, default_mode="audit")
        .Build()
    )
    optimized = (
        Headroom.with_openai()
        .named("optimized")
        .Configure(mode="cache", default_mode="optimize")
        .Build()
    )
    task = ScenarioTask(
        task_id="tiny-chat",
        messages=[
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Summarize this small payload."},
        ],
        model="gpt-4o",
    )

    report = ScenarioOrchestrator([passthrough, optimized]).run([task])

    assert report.passed is True
    assert len(report.cases) == 2
    assert report.total_tokens_before >= report.total_tokens_after
    payload = report.to_dict()
    assert payload["total_cases"] == 2
    assert payload["cases"][0]["guarantees"]


def test_orchestrator_surfaces_custom_guarantee_failures() -> None:
    scenario = Headroom.with_openai().named("guarded").Build()
    task = ScenarioTask(
        task_id="expected-failure",
        messages=[{"role": "user", "content": "hello"}],
    )

    def always_fail(*_args: object) -> GuaranteeResult:
        return GuaranteeResult(name="always_fail", passed=False, detail="demonstration failure")

    report = ScenarioOrchestrator([scenario], guarantees=[always_fail]).run([task])

    assert report.passed is False
    assert report.cases[0].passed is False
    assert report.to_dict()["cases"][0]["guarantees"] == [
        {"name": "always_fail", "passed": False, "detail": "demonstration failure"}
    ]


def test_headroom_suite_orchestrates_matrix_and_assigns_deployment_ports(tmp_path: Path) -> None:
    suite = (
        Headroom.Suite("phase-1-matrix")
        .Add(Headroom.WithOpenAI().named("openai-cache").WithCompression(mode="cache"))
        .Add(
            Headroom.WithBedrock(region="us-east-1")
            .named("bedrock-token")
            .WithCompression(mode="token")
        )
    )
    task = ScenarioTask(
        task_id="suite-smoke",
        messages=[{"role": "user", "content": "hello"}],
    )

    report = suite.Orchestrate([task])
    plans = suite.DeploymentPlans(port_start=19000)
    bundle = suite.ManifestBundle(provider="openai", port_start=19000).to_dict()
    path = suite.WriteManifestBundle(tmp_path / "suite.json", provider="openai", port_start=19000)
    agent_paths = suite.WriteAgentEvalsManifests(
        tmp_path / "agent-evals",
        benchmark="mini_swebench",
        benchmark_ref="mini@abc123",
        provider="openai",
        now=datetime(2026, 6, 15, 9, 30, tzinfo=timezone.utc),
    )
    written = json.loads(path.read_text(encoding="utf-8"))

    assert report.passed is True
    assert len(report.cases) == 2
    assert set(plans) == {"openai-cache", "bedrock-token"}
    assert plans["openai-cache"].command[:4] == ("headroom", "proxy", "--port", "19000")
    assert plans["bedrock-token"].command[:4] == ("headroom", "proxy", "--port", "19001")
    assert bundle["name"] == "phase-1-matrix"
    assert [scenario["suite_port"] for scenario in bundle["scenarios"]] == [19000, 19001]
    assert written == bundle
    assert {path.name for path in agent_paths} == {
        "openai-cache.agent-evals.json",
        "bedrock-token.agent-evals.json",
    }
    first_agent_payload = json.loads(agent_paths[0].read_text(encoding="utf-8"))
    assert set(first_agent_payload) == AGENT_EVALS_RUN_MANIFEST_FIELDS
    assert first_agent_payload["provider"] == "openai"


def test_headroom_suite_rejects_duplicate_scenario_names() -> None:
    suite = Headroom.Suite("duplicates").Add(Headroom.WithOpenAI().named("same"))

    with pytest.raises(ValueError, match="duplicate scenario name"):
        suite.Add(Headroom.WithBedrock(region="us-east-1").named("same"))


def test_empty_suite_fails_loudly() -> None:
    with pytest.raises(ValueError, match="requires at least one scenario"):
        Headroom.Suite("empty").DeploymentPlans()
