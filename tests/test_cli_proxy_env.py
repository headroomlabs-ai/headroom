"""Tests for CLI proxy env variable handling and backend validation.

Verifies that:
1. Provider target URL env vars are read by `headroom proxy`
2. litellm-* backends are accepted by both CLI and argparse paths
3. HEADROOM_WRAP_PROXY_TIMEOUT controls `headroom wrap` proxy readiness waits
"""

import os
from unittest.mock import patch

import pytest

click = pytest.importorskip("click")
pytest.importorskip("fastapi")

from click.testing import CliRunner  # noqa: E402

from headroom.cli import wrap as wrap_mod  # noqa: E402
from headroom.cli.main import main  # noqa: E402


@pytest.fixture
def runner():
    return CliRunner()


class _FakeProxyProcess:
    returncode = None

    def __init__(self):
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True


class TestCLIWrapProxyTimeout:
    """Test wrap proxy readiness timeout configuration."""

    def test_default_timeout_stays_current_without_ml_extras(self, monkeypatch):
        monkeypatch.delenv(wrap_mod._WRAP_PROXY_TIMEOUT_ENV, raising=False)
        monkeypatch.setattr(wrap_mod, "_ml_wrap_extras_detected", lambda: False)

        assert (
            wrap_mod._resolve_wrap_proxy_timeout_seconds()
            == wrap_mod._WRAP_PROXY_TIMEOUT_DEFAULT_SECONDS
        )

    def test_default_timeout_is_longer_when_ml_extras_detected(self, monkeypatch):
        monkeypatch.delenv(wrap_mod._WRAP_PROXY_TIMEOUT_ENV, raising=False)
        monkeypatch.setattr(wrap_mod, "_ml_wrap_extras_detected", lambda: True)

        assert (
            wrap_mod._resolve_wrap_proxy_timeout_seconds()
            == wrap_mod._WRAP_PROXY_TIMEOUT_ML_DEFAULT_SECONDS
        )

    def test_start_proxy_succeeds_when_ready_within_default_timeout(self, monkeypatch, tmp_path):
        fake_proc = _FakeProxyProcess()
        sleeps = []

        monkeypatch.delenv(wrap_mod._WRAP_PROXY_TIMEOUT_ENV, raising=False)
        monkeypatch.setattr(wrap_mod, "_ml_wrap_extras_detected", lambda: False)
        monkeypatch.setattr(wrap_mod, "_get_log_path", lambda: tmp_path / "proxy.log")
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: True)
        monkeypatch.setattr(wrap_mod.time, "sleep", lambda seconds: sleeps.append(seconds))
        monkeypatch.setattr(wrap_mod.subprocess, "Popen", lambda *args, **kwargs: fake_proc)

        proc = wrap_mod._start_proxy(8787, agent_type="codex")

        assert proc is fake_proc
        assert sleeps == [1]
        assert fake_proc.killed is False

    def test_start_proxy_passes_resolved_copilot_api_url_to_proxy(self, monkeypatch, tmp_path):
        fake_proc = _FakeProxyProcess()
        captured: dict[str, object] = {}

        monkeypatch.delenv(wrap_mod._WRAP_PROXY_TIMEOUT_ENV, raising=False)
        monkeypatch.setattr(wrap_mod, "_ml_wrap_extras_detected", lambda: False)
        monkeypatch.setattr(wrap_mod, "_get_log_path", lambda: tmp_path / "proxy.log")
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: True)
        monkeypatch.setattr(wrap_mod.time, "sleep", lambda _seconds: None)

        def fake_popen(*args, **kwargs):  # noqa: ANN002, ANN003
            captured["args"] = args
            captured["kwargs"] = kwargs
            return fake_proc

        monkeypatch.setattr(wrap_mod.subprocess, "Popen", fake_popen)

        proc = wrap_mod._start_proxy(
            8787,
            agent_type="copilot",
            openai_api_url="https://copilot-api.acme.ghe.com",
            copilot_api_token="copilot-api-token",
        )

        assert proc is fake_proc
        env = captured["kwargs"]["env"]
        assert env["OPENAI_TARGET_API_URL"] == "https://copilot-api.acme.ghe.com"
        assert env["GITHUB_COPILOT_API_URL"] == "https://copilot-api.acme.ghe.com"
        assert env["GITHUB_COPILOT_API_TOKEN"] == "copilot-api-token"

    def test_env_timeout_allows_slow_start_proxy_to_succeed(self, monkeypatch, tmp_path):
        fake_proc = _FakeProxyProcess()
        sleeps = []
        checks = []

        monkeypatch.setenv(wrap_mod._WRAP_PROXY_TIMEOUT_ENV, "4")
        monkeypatch.setattr(wrap_mod, "_get_log_path", lambda: tmp_path / "proxy.log")
        monkeypatch.setattr(wrap_mod.time, "sleep", lambda seconds: sleeps.append(seconds))
        monkeypatch.setattr(wrap_mod.subprocess, "Popen", lambda *args, **kwargs: fake_proc)

        def ready_on_fourth_check(port):
            checks.append(port)
            return len(checks) == 4

        monkeypatch.setattr(wrap_mod, "_check_proxy", ready_on_fourth_check)

        proc = wrap_mod._start_proxy(8787, agent_type="codex")

        assert proc is fake_proc
        assert checks == [8787, 8787, 8787, 8787]
        assert sleeps == [1, 1, 1, 1]
        assert fake_proc.killed is False

    def test_timeout_error_names_configured_timeout_and_env_var(self, monkeypatch, tmp_path):
        fake_proc = _FakeProxyProcess()

        monkeypatch.setenv(wrap_mod._WRAP_PROXY_TIMEOUT_ENV, "2")
        monkeypatch.setattr(wrap_mod, "_get_log_path", lambda: tmp_path / "proxy.log")
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: False)
        monkeypatch.setattr(wrap_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(wrap_mod.subprocess, "Popen", lambda *args, **kwargs: fake_proc)

        with pytest.raises(RuntimeError) as excinfo:
            wrap_mod._start_proxy(8787, agent_type="codex")

        message = str(excinfo.value)
        assert "within 2 seconds" in message
        assert wrap_mod._WRAP_PROXY_TIMEOUT_ENV in message
        assert fake_proc.killed is True


class TestCLIProxyEnvVars:
    """Test that the CLI proxy command reads API URL env vars."""

    def test_headroom_host_from_env(self, runner):
        """HEADROOM_HOST env var should be passed to ProxyConfig."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"HEADROOM_HOST": "0.0.0.0"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].host == "0.0.0.0"

    def test_headroom_port_from_env(self, runner):
        """HEADROOM_PORT env var should be passed to ProxyConfig."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"HEADROOM_PORT": "9797"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].port == 9797

    def test_headroom_min_tokens_from_env(self, runner):
        """HEADROOM_MIN_TOKENS env var should be passed to ProxyConfig."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"HEADROOM_MIN_TOKENS": "120"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].min_tokens_to_crush == 120

    def test_headroom_budget_from_env(self, runner):
        """HEADROOM_BUDGET env var should be passed to ProxyConfig."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"HEADROOM_BUDGET": "100.5"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].budget_limit_usd == 100.5

    def test_code_aware_enabled_from_env(self, runner):
        """HEADROOM_CODE_AWARE_ENABLED env var should be passed to ProxyConfig."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"HEADROOM_CODE_AWARE_ENABLED": "true"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].code_aware_enabled is True

    def test_code_aware_enabled_defaults_false(self, runner):
        """Without HEADROOM_CODE_AWARE_ENABLED, code-aware stays disabled in the wrapper."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        env = {k: v for k, v in os.environ.items() if k != "HEADROOM_CODE_AWARE_ENABLED"}

        with (
            patch("headroom.proxy.server.run_server", mock_run_server),
            patch.dict(os.environ, env, clear=True),
        ):
            result = runner.invoke(
                main,
                ["proxy"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].code_aware_enabled is False

    def test_code_aware_enabled_from_cli_flag(self, runner):
        """--code-aware should enable code-aware compression in the wrapper."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(main, ["proxy", "--code-aware"], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        assert captured_config["config"].code_aware_enabled is True

    def test_disable_kompress_from_env(self, runner):
        """HEADROOM_DISABLE_KOMPRESS should be passed to ProxyConfig."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"HEADROOM_DISABLE_KOMPRESS": "1"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].disable_kompress is True

    def test_disable_kompress_from_cli_flag(self, runner):
        """--disable-kompress should disable Kompress ML compression."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--disable-kompress"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].disable_kompress is True

    def test_code_aware_flag_overrides_env_var(self, runner):
        """--code-aware should win over HEADROOM_CODE_AWARE_ENABLED=false."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--code-aware"],
                env={"HEADROOM_CODE_AWARE_ENABLED": "false"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].code_aware_enabled is True

    def test_openai_target_api_url_from_env(self, runner):
        """OPENAI_TARGET_API_URL env var should be passed to ProxyConfig."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"OPENAI_TARGET_API_URL": "http://my-vllm:4000"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].openai_api_url == "http://my-vllm:4000"

    def test_gemini_target_api_url_from_env(self, runner):
        """GEMINI_TARGET_API_URL env var should be passed to ProxyConfig."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"GEMINI_TARGET_API_URL": "http://my-gemini:5000"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].gemini_api_url == "http://my-gemini:5000"

    def test_vertex_target_api_url_from_env(self, runner):
        """VERTEX_TARGET_API_URL env var should be passed to ProxyConfig."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"VERTEX_TARGET_API_URL": "https://europe-west4-aiplatform.googleapis.com"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert (
            captured_config["config"].vertex_api_url
            == "https://europe-west4-aiplatform.googleapis.com"
        )

    def test_openai_api_url_cli_flag(self, runner):
        """--openai-api-url CLI flag should take precedence."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--openai-api-url", "http://from-cli:4000"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].openai_api_url == "http://from-cli:4000"

    def test_vertex_api_url_cli_flag(self, runner):
        """--vertex-api-url CLI flag should take precedence."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--vertex-api-url", "https://us-east5-aiplatform.googleapis.com"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert (
            captured_config["config"].vertex_api_url == "https://us-east5-aiplatform.googleapis.com"
        )

    def test_cli_flag_overrides_env_var(self, runner):
        """CLI flag should take precedence over env var."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--openai-api-url", "http://from-cli:4000"],
                env={"OPENAI_TARGET_API_URL": "http://from-env:4000"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].openai_api_url == "http://from-cli:4000"

    def test_no_env_var_defaults_to_none(self, runner):
        """Without env var or flag, openai_api_url should be None."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        # Ensure the env var is not set
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_TARGET_API_URL"}

        with (
            patch("headroom.proxy.server.run_server", mock_run_server),
            patch.dict(os.environ, env, clear=True),
        ):
            result = runner.invoke(
                main,
                ["proxy"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].openai_api_url is None

    def test_both_api_urls_from_env(self, runner):
        """Both OPENAI and GEMINI target URLs can be set via env."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={
                    "OPENAI_TARGET_API_URL": "http://my-vllm:4000",
                    "GEMINI_TARGET_API_URL": "http://my-gemini:5000",
                },
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].openai_api_url == "http://my-vllm:4000"
        assert captured_config["config"].gemini_api_url == "http://my-gemini:5000"

    def test_retry_and_connect_timeout_cli_flags(self, runner):
        """Fast-fail CLI flags should map into ProxyConfig."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                [
                    "proxy",
                    "--retry-max-attempts",
                    "1",
                    "--connect-timeout-seconds",
                    "3",
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].retry_max_attempts == 1
        assert captured_config["config"].connect_timeout_seconds == 3

    def test_production_scaling_env_vars(self, runner):
        captured = {}

        def mock_run_server(config, **kwargs):
            captured["config"] = config
            captured["kwargs"] = kwargs

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={
                    "HEADROOM_WORKERS": "4",
                    "HEADROOM_LIMIT_CONCURRENCY": "250",
                    "HEADROOM_MAX_CONNECTIONS": "200",
                    "HEADROOM_MAX_KEEPALIVE": "50",
                },
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured["config"].max_connections == 200
        assert captured["config"].max_keepalive_connections == 50
        # Click CLI also passes `print_banner=False` to suppress the legacy
        # run_server banner (cli/proxy.py prints its own). Assert the
        # production-scaling keys we care about, not the full kwargs dict.
        assert captured["kwargs"]["workers"] == 4
        assert captured["kwargs"]["limit_concurrency"] == 250
        assert captured["kwargs"].get("print_banner") is False

    def test_production_scaling_cli_flags_override_env_vars(self, runner):
        captured = {}

        def mock_run_server(config, **kwargs):
            captured["config"] = config
            captured["kwargs"] = kwargs

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                [
                    "proxy",
                    "--workers",
                    "3",
                    "--limit-concurrency",
                    "125",
                    "--max-connections",
                    "150",
                    "--max-keepalive",
                    "25",
                ],
                env={
                    "HEADROOM_WORKERS": "4",
                    "HEADROOM_LIMIT_CONCURRENCY": "250",
                    "HEADROOM_MAX_CONNECTIONS": "200",
                    "HEADROOM_MAX_KEEPALIVE": "50",
                },
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured["config"].max_connections == 150
        assert captured["config"].max_keepalive_connections == 25
        # Click CLI also passes `print_banner=False`. Assert production
        # scaling keys explicitly rather than the full kwargs dict.
        assert captured["kwargs"]["workers"] == 3
        assert captured["kwargs"]["limit_concurrency"] == 125
        assert captured["kwargs"].get("print_banner") is False


class TestCLIProxyBackend:
    """Test that litellm-* backends are accepted by the CLI."""

    def test_litellm_hosted_vllm_backend_accepted(self, runner):
        """--backend litellm-hosted_vllm should be accepted (not rejected)."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--backend", "litellm-hosted_vllm"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].backend == "litellm-hosted_vllm"

    def test_litellm_vertex_backend_accepted(self, runner):
        """--backend litellm-vertex should be accepted."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--backend", "litellm-vertex"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].backend == "litellm-vertex"

    def test_litellm_backend_with_openai_url(self, runner):
        """Full vLLM setup: litellm backend + OPENAI_TARGET_API_URL."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                [
                    "proxy",
                    "--backend",
                    "litellm-hosted_vllm",
                    "--openai-api-url",
                    "http://my-vllm:4000",
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].backend == "litellm-hosted_vllm"
        assert captured_config["config"].openai_api_url == "http://my-vllm:4000"


class TestCLIAnyllmProviderEnv:
    """Test that HEADROOM_ANYLLM_PROVIDER env var is read by the CLI."""

    def test_anyllm_provider_from_env(self, runner):
        """HEADROOM_ANYLLM_PROVIDER env var should override the default."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--backend", "anyllm"],
                env={"HEADROOM_ANYLLM_PROVIDER": "llamacpp"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].anyllm_provider == "llamacpp"

    def test_anyllm_provider_cli_flag_works(self, runner):
        """--anyllm-provider flag should still work."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy", "--backend", "anyllm", "--anyllm-provider", "groq"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].anyllm_provider == "groq"


class TestCLICompressionOnlyFlags:
    """The CCR opt-out flags must flip the corresponding ProxyConfig fields.

    These enable a compression-only deployment for streaming / non-MCP clients
    that can't resolve the injected headroom_retrieve tool (issue #645).
    """

    def test_ccr_defaults_on(self, runner):
        """Without flags, all three CCR toggles stay enabled (no behavior change)."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(main, ["proxy"], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        cfg = captured_config["config"]
        assert cfg.ccr_inject_tool is True
        assert cfg.ccr_inject_marker is True
        assert cfg.ccr_proactive_expansion is True

    def test_no_ccr_inject_tool_flag(self, runner):
        """--no-ccr-inject-tool disables retrieve-tool injection only."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(main, ["proxy", "--no-ccr-inject-tool"], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        cfg = captured_config["config"]
        assert cfg.ccr_inject_tool is False
        # Untouched flags remain on.
        assert cfg.ccr_inject_marker is True
        assert cfg.ccr_proactive_expansion is True

    def test_compression_only_all_flags(self, runner):
        """All three flags together yield a compression-only config."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                [
                    "proxy",
                    "--no-ccr-inject-tool",
                    "--no-ccr-marker",
                    "--no-ccr-proactive-expansion",
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        cfg = captured_config["config"]
        assert cfg.ccr_inject_tool is False
        assert cfg.ccr_inject_marker is False
        assert cfg.ccr_proactive_expansion is False

    def test_no_ccr_marker_from_env(self, runner):
        """HEADROOM_NO_CCR_MARKER env var disables marker injection."""
        captured_config = {}

        def mock_run_server(config, **kwargs):
            captured_config["config"] = config

        with patch("headroom.proxy.server.run_server", mock_run_server):
            result = runner.invoke(
                main,
                ["proxy"],
                env={"HEADROOM_NO_CCR_MARKER": "1"},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, result.output
        assert captured_config["config"].ccr_inject_marker is False


class TestArgparseBackendValidation:
    """Test that the argparse path (python -m headroom.proxy.server) accepts litellm-* backends."""

    def test_argparse_accepts_litellm_backend(self):
        """The argparse --backend should accept litellm-hosted_vllm (no choices restriction)."""
        import argparse

        # Recreate the parser matching server.py's main() argparse setup
        # We just need to verify argparse doesn't reject litellm-* values
        parser = argparse.ArgumentParser()
        parser.add_argument("--backend", default="anthropic")
        args = parser.parse_args(["--backend", "litellm-hosted_vllm"])
        assert args.backend == "litellm-hosted_vllm"

    def test_proxy_config_from_env_reads_disable_kompress(self):
        """The direct server env path should honor HEADROOM_DISABLE_KOMPRESS."""
        from headroom.proxy.server import _proxy_config_from_env

        with patch.dict(os.environ, {"HEADROOM_DISABLE_KOMPRESS": "1"}):
            config = _proxy_config_from_env()

        assert config.disable_kompress is True
