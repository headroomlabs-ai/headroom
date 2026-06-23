from __future__ import annotations

import json
from importlib import import_module
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from headroom.agent_savings import (
    AGENT_90_PROFILE,
    apply_agent_savings_env_defaults,
    apply_agent_savings_profile,
    get_agent_savings_profile,
    proxy_pipeline_kwargs,
    with_target_savings,
)
from headroom.cli import wrap as wrap_module
from headroom.cli.main import main
from headroom.compress import CompressConfig, compress
from headroom.proxy.models import ProxyConfig
from headroom.transforms.compression_units import (
    CompressionUnit,
    compress_unit_with_router,
)
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
)

compress_module = import_module("headroom.compress")


def test_agent_90_profile_sets_accuracy_preserving_compress_config() -> None:
    cfg = CompressConfig()

    apply_agent_savings_profile(cfg, AGENT_90_PROFILE)

    assert cfg.compress_user_messages is True
    assert cfg.compress_system_messages is True
    assert cfg.protect_recent == 2
    assert cfg.protect_analysis_context is True
    assert cfg.target_ratio == 0.10
    assert cfg.min_tokens_to_compress == 120


def test_agent_90_profile_exports_cross_agent_proxy_env() -> None:
    profile = get_agent_savings_profile(AGENT_90_PROFILE)

    env = profile.proxy_env()

    assert env["HEADROOM_MODE"] == "token"
    assert env["HEADROOM_SAVINGS_PROFILE"] == "agent-90"
    assert env["HEADROOM_SAVINGS_TARGET"] == "0.90"
    assert env["HEADROOM_TARGET_RATIO"] == "0.10"
    assert env["HEADROOM_COMPRESS_USER_MESSAGES"] == "1"
    assert env["HEADROOM_COMPRESS_SYSTEM_MESSAGES"] == "1"
    assert env["HEADROOM_MAX_ITEMS"] == "8"
    assert env["HEADROOM_SMART_CRUSHER_COMPACTION"] == "0"
    assert env["HEADROOM_FORCE_KOMPRESS"] == "1"
    assert env["HEADROOM_ACCURACY_GUARD"] == "strict"


def test_agent_savings_env_defaults_preserve_user_overrides() -> None:
    env = {
        "HEADROOM_TARGET_RATIO": "0.25",
        "HEADROOM_MAX_ITEMS": "12",
    }

    apply_agent_savings_env_defaults(env, AGENT_90_PROFILE)

    assert env["HEADROOM_SAVINGS_PROFILE"] == "agent-90"
    assert env["HEADROOM_TARGET_RATIO"] == "0.25"
    assert env["HEADROOM_MAX_ITEMS"] == "12"
    assert env["HEADROOM_SMART_CRUSHER_COMPACTION"] == "0"


def test_unknown_agent_savings_profile_lists_valid_profiles() -> None:
    with pytest.raises(ValueError, match="agent-90"):
        get_agent_savings_profile("missing")


def test_with_target_savings_recomputes_target_ratio() -> None:
    profile = with_target_savings(get_agent_savings_profile("balanced"), 0.85)

    assert profile.target_savings == 0.85
    assert profile.target_ratio == 0.15


def test_agent_savings_cli_renders_shell_exports() -> None:
    result = CliRunner().invoke(main, ["agent-savings", "--profile", "agent-90"])

    assert result.exit_code == 0
    assert 'export HEADROOM_SAVINGS_PROFILE="agent-90"' in result.output
    assert 'export HEADROOM_SAVINGS_TARGET="0.90"' in result.output
    assert 'export HEADROOM_ACCURACY_GUARD="strict"' in result.output


def test_agent_savings_cli_renders_json() -> None:
    result = CliRunner().invoke(
        main,
        ["agent-savings", "--profile", "agent-90", "--format", "json"],
    )

    assert result.exit_code == 0
    assert '"HEADROOM_TARGET_RATIO": "0.10"' in result.output


def test_compress_applies_agent_savings_profile_to_pipeline(monkeypatch) -> None:
    captured: dict[str, object] = {}
    messages = [{"role": "user", "content": "x" * 500}]

    class Pipeline:
        def apply(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                messages=messages,
                tokens_before=1000,
                tokens_after=100,
                transforms_applied=["test"],
            )

    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: Pipeline())

    config = CompressConfig()
    apply_agent_savings_profile(config, AGENT_90_PROFILE)

    result = compress(messages, config=config)

    assert result.compression_ratio == 0.9
    assert captured["compress_user_messages"] is True
    assert captured["compress_system_messages"] is True
    assert captured["protect_recent"] == 2
    assert captured["protect_analysis_context"] is True
    assert captured["target_ratio"] == 0.10
    assert captured["min_tokens_to_compress"] == 120


def test_compress_savings_profile_does_not_mutate_supplied_config(monkeypatch) -> None:
    captured: dict[str, object] = {}
    messages = [{"role": "user", "content": "x" * 500}]
    config = CompressConfig(
        compress_user_messages=False,
        compress_system_messages=False,
        protect_recent=9,
        protect_analysis_context=False,
        target_ratio=None,
        min_tokens_to_compress=999,
    )

    class Pipeline:
        def apply(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                messages=messages,
                tokens_before=1000,
                tokens_after=100,
                transforms_applied=["test"],
            )

    monkeypatch.setattr(compress_module, "_get_pipeline", lambda: Pipeline())

    compress(messages, config=config, savings_profile=AGENT_90_PROFILE)

    assert captured["target_ratio"] == 0.10
    assert captured["min_tokens_to_compress"] == 120
    assert config.compress_user_messages is False
    assert config.compress_system_messages is False
    assert config.protect_recent == 9
    assert config.protect_analysis_context is False
    assert config.target_ratio is None
    assert config.min_tokens_to_compress == 999


def test_wrap_agent_savings_profile_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_SAVINGS_PROFILE", raising=False)

    assert wrap_module._wrap_agent_savings_profile("codex") is None

    monkeypatch.setenv("HEADROOM_SAVINGS_PROFILE", AGENT_90_PROFILE)

    assert wrap_module._wrap_agent_savings_profile("codex") == AGENT_90_PROFILE


def test_agent_savings_config_mismatches_requires_explicit_profile(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_SAVINGS_PROFILE", raising=False)

    assert wrap_module._agent_savings_config_mismatches({}, "claude") == []


def test_start_proxy_does_not_inject_agent_savings_by_default(monkeypatch, tmp_path) -> None:
    captured_env: dict[str, str] = {}

    class Proc:
        returncode = None

        def poll(self) -> None:
            return None

    def popen(cmd, **kwargs):  # noqa: ANN001
        captured_env.update(kwargs["env"])
        return Proc()

    monkeypatch.delenv("HEADROOM_SAVINGS_PROFILE", raising=False)
    monkeypatch.setattr(wrap_module.subprocess, "Popen", popen)
    monkeypatch.setattr(wrap_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(wrap_module, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_module, "_get_log_path", lambda: tmp_path / "proxy.log")

    wrap_module._start_proxy(8787, agent_type="codex")

    assert "HEADROOM_SAVINGS_PROFILE" not in captured_env
    assert "HEADROOM_TARGET_RATIO" not in captured_env


def test_start_proxy_injects_explicit_agent_savings_profile(monkeypatch, tmp_path) -> None:
    captured_env: dict[str, str] = {}

    class Proc:
        returncode = None

        def poll(self) -> None:
            return None

    def popen(cmd, **kwargs):  # noqa: ANN001
        captured_env.update(kwargs["env"])
        return Proc()

    monkeypatch.setenv("HEADROOM_SAVINGS_PROFILE", AGENT_90_PROFILE)
    monkeypatch.setattr(wrap_module.subprocess, "Popen", popen)
    monkeypatch.setattr(wrap_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(wrap_module, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_module, "_get_log_path", lambda: tmp_path / "proxy.log")

    wrap_module._start_proxy(8787, agent_type="codex")

    assert captured_env["HEADROOM_SAVINGS_PROFILE"] == AGENT_90_PROFILE
    assert captured_env["HEADROOM_TARGET_RATIO"] == "0.10"


def test_agent_savings_config_mismatches_returns_specific_labels(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_SAVINGS_PROFILE", AGENT_90_PROFILE)
    profile = get_agent_savings_profile(AGENT_90_PROFILE)
    running_config = {
        "savings_profile": profile.name,
        "target_ratio": 0.20,
        "compress_user_messages": profile.compress_user_messages,
        "compress_system_messages": profile.compress_system_messages,
        "protect_recent": profile.protect_recent,
        "protect_analysis_context": profile.protect_analysis_context,
        "min_tokens_to_crush": profile.min_tokens_to_compress,
        "max_items_after_crush": profile.max_items_after_crush,
        "smart_crusher_with_compaction": profile.smart_crusher_with_compaction,
        "accuracy_guard": profile.accuracy_guard,
    }

    assert wrap_module._agent_savings_config_mismatches(running_config, "codex") == ["target-ratio"]


def test_agent_savings_config_mismatches_ignores_non_target_agents() -> None:
    assert wrap_module._agent_savings_config_mismatches({}, "openhands") == []


def test_agent_savings_config_mismatches_accepts_matching_runtime_config(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_SAVINGS_PROFILE", AGENT_90_PROFILE)
    profile = get_agent_savings_profile(AGENT_90_PROFILE)
    running_config = {
        "savings_profile": profile.name,
        "target_ratio": "0.10",
        "compress_user_messages": True,
        "compress_system_messages": True,
        "protect_recent": "2",
        "protect_analysis_context": True,
        "min_tokens_to_crush": "120",
        "max_items_after_crush": "8",
        "smart_crusher_with_compaction": False,
        "accuracy_guard": "strict",
    }

    assert wrap_module._agent_savings_config_mismatches(running_config, "cursor") == []


def test_agent_savings_config_mismatches_reports_unparseable_values(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_SAVINGS_PROFILE", AGENT_90_PROFILE)
    running_config = {
        "savings_profile": None,
        "target_ratio": "not-a-float",
        "compress_user_messages": None,
        "compress_system_messages": None,
        "protect_recent": "not-an-int",
        "protect_analysis_context": None,
        "min_tokens_to_crush": object(),
        "max_items_after_crush": object(),
        "smart_crusher_with_compaction": None,
        "accuracy_guard": None,
    }

    assert wrap_module._agent_savings_config_mismatches(running_config, "claude") == [
        "savings-profile",
        "target-ratio",
        "compress-user-messages",
        "compress-system-messages",
        "protect-recent",
        "protect-analysis-context",
        "min-tokens",
        "max-items",
        "smart-crusher-compaction",
        "accuracy-guard",
    ]


def test_agent_90_profile_applies_to_proxy_config_runtime_kwargs() -> None:
    config = ProxyConfig(savings_profile="agent-90")

    kwargs = proxy_pipeline_kwargs(config)

    assert kwargs["compress_user_messages"] is True
    assert kwargs["compress_system_messages"] is True
    assert kwargs["protect_recent"] == 2
    assert kwargs["protect_analysis_context"] is True
    assert kwargs["target_ratio"] == 0.10
    assert kwargs["min_tokens_to_compress"] == 120
    assert kwargs["max_items_after_crush"] == 8
    assert kwargs["smart_crusher_with_compaction"] is False
    assert kwargs["force_kompress"] is True
    assert kwargs["read_protection_window"] == 2


def test_proxy_explicit_config_overrides_agent_90_profile() -> None:
    config = ProxyConfig(
        savings_profile="agent-90",
        target_ratio=0.25,
        protect_recent=5,
        min_tokens_to_crush=300,
    )

    kwargs = proxy_pipeline_kwargs(config)

    assert kwargs["target_ratio"] == 0.25
    assert kwargs["protect_recent"] == 5
    assert kwargs["min_tokens_to_compress"] == 300


def test_agent_90_router_uses_ccr_sampling_not_lossless_table() -> None:
    router = ContentRouter(
        ContentRouterConfig(
            smart_crusher_max_items_after_crush=8,
            smart_crusher_with_compaction=False,
        )
    )

    crusher = router._get_smart_crusher()

    assert crusher is not None
    assert crusher.config.max_items_after_crush == 8
    assert crusher._with_compaction is False


def test_agent_90_router_json_tool_output_reaches_target_with_needle() -> None:
    needle = "CRITICAL_NEEDLE_42"
    rows = [
        {
            "id": i,
            "status": "ok",
            "message": "normal repeated telemetry payload",
            "value": i % 7,
        }
        for i in range(1000)
    ]
    rows.append(
        {
            "id": 99999,
            "status": "error",
            "message": f"{needle} root cause disk full",
            "value": 999.99,
        }
    )
    router = ContentRouter(
        ContentRouterConfig(
            smart_crusher_max_items_after_crush=8,
            smart_crusher_with_compaction=False,
        )
    )

    result = router.compress(json.dumps(rows), question=f"Find {needle}")
    before = len(result.original.split())
    after = len(result.compressed.split())

    assert 1 - after / before >= 0.90
    assert needle in result.compressed
    assert "<<ccr:" in result.compressed


def test_proxy_cli_reads_agent_90_profile_env() -> None:
    captured_config: dict[str, ProxyConfig] = {}

    def mock_run_server(config: ProxyConfig, **kwargs: object) -> None:
        captured_config["config"] = config

    runner = CliRunner()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("headroom.proxy.server.run_server", mock_run_server)
        result = runner.invoke(
            main,
            ["proxy"],
            env={"HEADROOM_SAVINGS_PROFILE": "agent-90"},
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    config = captured_config["config"]
    assert config.savings_profile == "agent-90"
    assert proxy_pipeline_kwargs(config)["target_ratio"] == 0.10


def test_unit_router_receives_agent_target_ratio() -> None:
    seen: dict[str, object] = {}

    class Tokenizer:
        def count_text(self, text: str) -> int:
            return len(text.split())

    class Router:
        _runtime_target_ratio = None

        def compress(self, text: str, **kwargs: object) -> RouterCompressionResult:
            seen["target_ratio"] = self._runtime_target_ratio
            return RouterCompressionResult(
                compressed="short text",
                original=text,
                strategy_used=CompressionStrategy.KOMPRESS,
                strategy_chain=["kompress"],
            )

    unit = CompressionUnit(
        text=("long text " * 40) + "\nRetrieve more: hash=abc123\n",
        provider="openai",
        endpoint="responses",
        role="assistant",
        item_type="message",
        cache_zone="live",
        mutable=True,
        min_bytes=1,
        metadata={"compress_assistant": "true"},
    )

    result = compress_unit_with_router(
        unit,
        router=Router(),
        tokenizer=Tokenizer(),
        target_ratio=0.10,
    )

    assert result.modified is True
    assert seen["target_ratio"] == 0.10


def test_agent_savings_check_perf_and_accuracy_report_passes(
    monkeypatch,
    tmp_path,
) -> None:
    from headroom.perf import analyzer

    monkeypatch.setattr(analyzer, "parse_log_files", lambda last_n_hours: object())
    monkeypatch.setattr(
        analyzer,
        "build_perf_summary",
        lambda report: {"savings_pct": 92.0},
    )
    report = tmp_path / "eval.json"
    report.write_text(json.dumps({"totals": {"accuracy_rate": 1.0}}))

    result = CliRunner().invoke(
        main,
        [
            "agent-savings",
            "--profile",
            "agent-90",
            "--check-perf",
            "--accuracy-report",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "92.0% savings meets 90.0%" in result.output
    assert "100.0% accuracy meets 90.0%" in result.output


def test_agent_savings_accuracy_report_below_threshold_fails(
    monkeypatch,
    tmp_path,
) -> None:
    from headroom.perf import analyzer

    monkeypatch.setattr(analyzer, "parse_log_files", lambda last_n_hours: object())
    monkeypatch.setattr(
        analyzer,
        "build_perf_summary",
        lambda report: {"savings_pct": 92.0},
    )
    report = tmp_path / "eval.json"
    report.write_text(json.dumps({"totals": {"accuracy_rate": 0.89}}))

    result = CliRunner().invoke(
        main,
        [
            "agent-savings",
            "--profile",
            "agent-90",
            "--check-perf",
            "--accuracy-report",
            str(report),
        ],
    )

    assert result.exit_code != 0
    assert "89.0% accuracy below 90.0%" in result.output


def test_agent_savings_requires_each_agent_to_meet_target(monkeypatch) -> None:
    from headroom.perf import analyzer
    from headroom.perf.analyzer import PerfRecord, PerfReport

    report = PerfReport(
        perf_records=[
            PerfRecord(
                timestamp="2026-06-10 10:00:00,000",
                request_id="claude-1",
                model="claude-sonnet",
                client="claude",
                tokens_before=1000,
                tokens_after=80,
                tokens_saved=920,
            ),
            PerfRecord(
                timestamp="2026-06-10 10:01:00,000",
                request_id="codex-1",
                model="gpt-5",
                client="codex",
                tokens_before=1000,
                tokens_after=90,
                tokens_saved=910,
            ),
            PerfRecord(
                timestamp="2026-06-10 10:02:00,000",
                request_id="cursor-1",
                model="gpt-5",
                client="cursor",
                tokens_before=1000,
                tokens_after=70,
                tokens_saved=930,
            ),
        ]
    )
    monkeypatch.setattr(analyzer, "parse_log_files", lambda last_n_hours: report)

    result = CliRunner().invoke(
        main,
        [
            "agent-savings",
            "--check-perf",
            "--require-agents",
            "claude,codex,cursor",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "claude: 92.0% savings meets 90.0%" in result.output
    assert "codex: 91.0% savings meets 90.0%" in result.output
    assert "cursor: 93.0% savings meets 90.0%" in result.output


def test_agent_savings_required_agent_missing_fails(monkeypatch) -> None:
    from headroom.perf import analyzer
    from headroom.perf.analyzer import PerfRecord, PerfReport

    report = PerfReport(
        perf_records=[
            PerfRecord(
                timestamp="2026-06-10 10:00:00,000",
                request_id="claude-1",
                model="claude-sonnet",
                client="claude",
                tokens_before=1000,
                tokens_after=80,
                tokens_saved=920,
            ),
            PerfRecord(
                timestamp="2026-06-10 10:01:00,000",
                request_id="codex-1",
                model="gpt-5",
                client="codex",
                tokens_before=1000,
                tokens_after=90,
                tokens_saved=910,
            ),
        ]
    )
    monkeypatch.setattr(analyzer, "parse_log_files", lambda last_n_hours: report)

    result = CliRunner().invoke(
        main,
        [
            "agent-savings",
            "--check-perf",
            "--require-agents",
            "claude,codex,cursor",
        ],
    )

    assert result.exit_code != 0
    assert "missing required agent traffic: cursor" in result.output


def test_agent_savings_writes_three_agent_smoke_fixture(tmp_path) -> None:
    workspace = tmp_path / "workspace"

    result = CliRunner().invoke(
        main,
        ["agent-savings", "--write-smoke-fixture", str(workspace)],
    )

    assert result.exit_code == 0, result.output
    assert (workspace / "logs" / "proxy.log").exists()
    eval_report = workspace / "agent-90-eval.json"
    assert eval_report.exists()
    assert "--require-agents claude,codex,cursor" in result.output


def test_agent_savings_smoke_fixture_passes_real_gate(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    runner = CliRunner()

    write_result = runner.invoke(
        main,
        ["agent-savings", "--write-smoke-fixture", str(workspace)],
    )
    assert write_result.exit_code == 0, write_result.output

    gate_result = runner.invoke(
        main,
        [
            "agent-savings",
            "--check-perf",
            "--hours",
            "0",
            "--require-agents",
            "claude,codex,cursor",
            "--accuracy-report",
            str(workspace / "agent-90-eval.json"),
        ],
        env={"HEADROOM_WORKSPACE_DIR": str(workspace)},
    )

    assert gate_result.exit_code == 0, gate_result.output
    assert "claude: 92.0% savings meets 90.0%" in gate_result.output
    assert "codex: 91.0% savings meets 90.0%" in gate_result.output
    assert "cursor: 93.0% savings meets 90.0%" in gate_result.output
    assert "100.0% accuracy meets 90.0%" in gate_result.output
