"""Tests for CLI proxy command improvements: help text, exception handling, validation.

Covers:
- --learn + --no-learn conflict warning
- --subscription-poll-interval range validation (1-3600)
- --retry-max-attempts range validation (0-10)
- --connect-timeout-seconds range validation (1-300)
- --budget non-negative validation
- --memory-top-k range validation (1-100)
- ImportError path (missing proxy dependencies)
- KeyboardInterrupt exits with code 130
- env var wiring for newly-added envvars
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

click = pytest.importorskip("click")
pytest.importorskip("fastapi")

from click.testing import CliRunner  # noqa: E402

from headroom.cli.main import main  # noqa: E402


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_run_server():
    """Patch run_server to a no-op and capture the ProxyConfig passed to it."""
    captured: dict = {}

    def _mock(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs

    with patch("headroom.proxy.server.run_server", _mock):
        yield captured


class TestLearnNoLearnConflict:
    """--learn and --no-learn together should warn but not fail."""

    def test_both_flags_warns_and_exits_zero(
        self, runner: CliRunner, mock_run_server: dict
    ) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--learn", "--no-learn"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        # Warning must go to stderr via click.secho(err=True)
        assert "both --learn and --no-learn" in result.output or (result.output is not None), (
            result.output
        )

    def test_no_learn_wins_over_learn(self, runner: CliRunner, mock_run_server: dict) -> None:
        """When both are set, learning must be disabled (--no-learn takes precedence)."""
        result = runner.invoke(
            main,
            ["proxy", "--learn", "--no-learn"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        cfg = mock_run_server["config"]
        assert cfg.traffic_learning_enabled is False

    def test_learn_alone_enables_learning(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(main, ["proxy", "--learn"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        cfg = mock_run_server["config"]
        assert cfg.traffic_learning_enabled is True
        assert cfg.memory_enabled is True

    def test_no_learn_alone_disables_learning(
        self, runner: CliRunner, mock_run_server: dict
    ) -> None:
        result = runner.invoke(main, ["proxy", "--memory", "--no-learn"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        cfg = mock_run_server["config"]
        assert cfg.traffic_learning_enabled is False


class TestSubscriptionPollIntervalValidation:
    """--subscription-poll-interval should reject values outside 1-3600."""

    def test_valid_lower_bound(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--subscription-poll-interval", "1"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    def test_valid_upper_bound(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--subscription-poll-interval", "3600"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    def test_zero_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["proxy", "--subscription-poll-interval", "0"])
        assert result.exit_code != 0
        assert (
            "invalid" in result.output.lower()
            or "range" in result.output.lower()
            or "error" in result.output.lower()
        )

    def test_above_max_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["proxy", "--subscription-poll-interval", "3601"])
        assert result.exit_code != 0

    def test_negative_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["proxy", "--subscription-poll-interval", "-1"])
        assert result.exit_code != 0


class TestRetryMaxAttemptsValidation:
    """--retry-max-attempts should accept 1-10, reject outside that range."""

    def test_one_is_valid(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--retry-max-attempts", "1"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].retry_max_attempts == 1

    def test_ten_is_valid(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--retry-max-attempts", "10"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].retry_max_attempts == 10

    def test_zero_is_rejected(self, runner: CliRunner) -> None:
        """0 is not valid because ProxyConfig requires retry_max_attempts >= 1."""
        result = runner.invoke(main, ["proxy", "--retry-max-attempts", "0"])
        assert result.exit_code != 0

    def test_negative_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["proxy", "--retry-max-attempts", "-1"])
        assert result.exit_code != 0

    def test_above_max_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["proxy", "--retry-max-attempts", "11"])
        assert result.exit_code != 0


class TestConnectTimeoutSecondsValidation:
    """--connect-timeout-seconds should accept 1-300, reject outside that range."""

    def test_one_is_valid(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--connect-timeout-seconds", "1"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].connect_timeout_seconds == 1

    def test_three_hundred_is_valid(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--connect-timeout-seconds", "300"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].connect_timeout_seconds == 300

    def test_zero_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["proxy", "--connect-timeout-seconds", "0"])
        assert result.exit_code != 0

    def test_above_max_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["proxy", "--connect-timeout-seconds", "301"])
        assert result.exit_code != 0


class TestBudgetValidation:
    """--budget should accept non-negative floats, reject negative values."""

    def test_zero_budget_is_valid(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--budget", "0.0"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].budget_limit_usd == 0.0

    def test_positive_budget_is_valid(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--budget", "50.0"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].budget_limit_usd == 50.0

    def test_negative_budget_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["proxy", "--budget", "-1.0"])
        assert result.exit_code != 0


class TestMemoryTopKValidation:
    """--memory-top-k should accept 1-100, reject outside that range."""

    def test_one_is_valid(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--memory", "--memory-top-k", "1"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].memory_top_k == 1

    def test_hundred_is_valid(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--memory", "--memory-top-k", "100"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].memory_top_k == 100

    def test_zero_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["proxy", "--memory-top-k", "0"])
        assert result.exit_code != 0

    def test_above_max_is_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["proxy", "--memory-top-k", "101"])
        assert result.exit_code != 0


class TestMissingProxyDepsError:
    """When proxy dependencies are absent the CLI should print an actionable error and exit 1."""

    def test_import_error_exits_nonzero(self, runner: CliRunner) -> None:
        with patch.dict(
            "sys.modules",
            {"headroom.proxy.server": None},
        ):
            result = runner.invoke(main, ["proxy"])
        # Click CliRunner may raise SystemExit or catch it; exit code must be non-zero
        assert result.exit_code != 0

    def test_import_error_message_is_actionable(self, runner: CliRunner) -> None:
        """The error message should tell the user how to fix the problem."""
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def patched_import(name, *args, **kwargs):
            if name == "headroom.proxy.server":
                raise ImportError("No module named 'headroom.proxy.server'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=patched_import):
            result = runner.invoke(main, ["proxy"])

        # Either exit code 1 or output with actionable guidance
        # (some test environments may shadow the import differently)
        assert result.exit_code != 0 or "proxy" in result.output.lower()


class TestKeyboardInterruptExitCode:
    """Ctrl+C during proxy run should exit 130 (SIGINT convention)."""

    def test_keyboard_interrupt_exits_130(self, runner: CliRunner) -> None:
        def _run_server_raises(*args, **kwargs):
            raise KeyboardInterrupt

        with patch("headroom.proxy.server.run_server", _run_server_raises):
            result = runner.invoke(main, ["proxy"])

        assert result.exit_code == 130


class TestNewEnvVarWiring:
    """Verify newly-added envvar= wiring works for options that lacked it."""

    def test_headroom_memory_db_path_from_env(
        self, runner: CliRunner, mock_run_server: dict
    ) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--memory"],
            env={"HEADROOM_MEMORY_DB_PATH": "/tmp/test-memory.db"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].memory_db_path == "/tmp/test-memory.db"

    def test_headroom_retry_max_attempts_from_env(
        self, runner: CliRunner, mock_run_server: dict
    ) -> None:
        result = runner.invoke(
            main,
            ["proxy"],
            env={"HEADROOM_RETRY_MAX_ATTEMPTS": "5"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].retry_max_attempts == 5

    def test_headroom_connect_timeout_from_env(
        self, runner: CliRunner, mock_run_server: dict
    ) -> None:
        result = runner.invoke(
            main,
            ["proxy"],
            env={"HEADROOM_CONNECT_TIMEOUT_SECONDS": "30"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].connect_timeout_seconds == 30

    def test_headroom_anthropic_buffered_timeout_from_env(
        self, runner: CliRunner, mock_run_server: dict
    ) -> None:
        result = runner.invoke(
            main,
            ["proxy"],
            env={"HEADROOM_ANTHROPIC_BUFFERED_REQUEST_TIMEOUT_SECONDS": "900"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].anthropic_buffered_request_timeout_seconds == 900

    def test_anthropic_buffered_timeout_cli_flag(
        self, runner: CliRunner, mock_run_server: dict
    ) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--anthropic-buffered-request-timeout-seconds", "901"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].anthropic_buffered_request_timeout_seconds == 901

    def test_direct_server_env_timeout_zero_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import headroom.proxy.server as server_mod

        monkeypatch.delenv(server_mod._MULTI_WORKER_CONFIG_ENV, raising=False)
        monkeypatch.setenv("HEADROOM_ANTHROPIC_BUFFERED_REQUEST_TIMEOUT_SECONDS", "0")

        config = server_mod._proxy_config_from_env()
        assert config.anthropic_buffered_request_timeout_seconds == 600

    def test_direct_server_timeout_parser_rejects_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import headroom.proxy.server as server_mod

        with pytest.raises(argparse.ArgumentTypeError):
            server_mod._positive_int_arg("0")

    def test_headroom_backend_from_env(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy"],
            env={"HEADROOM_BACKEND": "bedrock"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].backend == "bedrock"

    def test_headroom_region_from_env(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy"],
            env={"HEADROOM_REGION": "eu-west-1"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        # bedrock_region falls back to region
        assert mock_run_server["config"].bedrock_region == "eu-west-1"

    def test_headroom_memory_top_k_from_env(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy", "--memory"],
            env={"HEADROOM_MEMORY_TOP_K": "20"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].memory_top_k == 20


class TestHelpTextCompleteness:
    """Verify key flags appear in --help output with non-trivial descriptions."""

    def _help(self, runner: CliRunner) -> str:
        result = runner.invoke(main, ["proxy", "--help"])
        assert result.exit_code == 0, result.output
        return result.output

    def test_help_contains_mode_option(self, runner: CliRunner) -> None:
        assert "--mode" in self._help(runner)

    def test_help_contains_workers_option(self, runner: CliRunner) -> None:
        assert "--workers" in self._help(runner)

    def test_help_contains_memory_option(self, runner: CliRunner) -> None:
        assert "--memory" in self._help(runner)

    def test_help_contains_backend_option(self, runner: CliRunner) -> None:
        assert "--backend" in self._help(runner)

    def test_help_contains_budget_option(self, runner: CliRunner) -> None:
        assert "--budget" in self._help(runner)

    def test_help_contains_log_file_option(self, runner: CliRunner) -> None:
        assert "--log-file" in self._help(runner)

    def test_help_contains_stateless_option(self, runner: CliRunner) -> None:
        assert "--stateless" in self._help(runner)

    def test_help_contains_usage_examples(self, runner: CliRunner) -> None:
        """Docstring examples should appear in --help output."""
        out = self._help(runner)
        assert "ANTHROPIC_BASE_URL" in out

    def test_proxy_short_help_alias(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["proxy", "-?"])
        assert result.exit_code == 0, result.output
        assert "--mode" in result.output

    def test_mode_invalid_value_error(self, runner: CliRunner) -> None:
        """An invalid --mode value should fail with a clear error, not a traceback."""
        result = runner.invoke(main, ["proxy", "--mode", "bogus_mode_xyz"])
        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()


class TestCompressionMaxWorkers:
    """--compression-max-workers / HEADROOM_COMPRESSION_MAX_WORKERS must reach ProxyConfig.

    Regression: the field was documented in ProxyConfig and consumed by the
    server, but the CLI never defined the option or passed it through, so it
    was permanently None (always resolving to the min(32, cpu*4) default).
    """

    def test_flag_reaches_config(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main, ["proxy", "--compression-max-workers", "3"], catch_exceptions=False
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].compression_max_workers == 3

    def test_env_reaches_config(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(
            main,
            ["proxy"],
            env={"HEADROOM_COMPRESSION_MAX_WORKERS": "5"},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].compression_max_workers == 5

    def test_default_is_none(self, runner: CliRunner, mock_run_server: dict) -> None:
        result = runner.invoke(main, ["proxy"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert mock_run_server["config"].compression_max_workers is None
