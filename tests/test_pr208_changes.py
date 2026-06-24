"""Tests for changes introduced in PR #208 (fix/npm-version-allow-same-version).

Covers:
- jitter_delay_ms helper function (exponential backoff with jitter)
- _headroom_log_dir lazy resolution via paths module
- asyncio.timeout compatibility shim in scripts/repro_codex_replay.py
- --allow-same-version flag presence in release workflow
- SIGKILL fallback in cli/wrap.py for Windows compatibility
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# jitter_delay_ms tests
# ---------------------------------------------------------------------------


class TestJitterDelayMs:
    """Tests for headroom.proxy.helpers.jitter_delay_ms."""

    def test_attempt_zero_returns_within_base_range(self) -> None:
        from headroom.proxy.helpers import jitter_delay_ms

        # At attempt=0: capped = min(250 * 2^0, 5000) = 250
        # Result = 250 * (0.5 + random()) where random in [0, 1)
        # So range is [125, 375)
        for _ in range(100):
            val = jitter_delay_ms(base_ms=250, max_ms=5000, attempt=0)
            assert 125.0 <= val < 375.0, f"attempt=0 yielded {val}, expected [125, 375)"

    def test_exponential_growth_with_attempt(self) -> None:
        from headroom.proxy.helpers import jitter_delay_ms

        # Collect median-ish values across many samples to verify growth
        samples_a1 = [jitter_delay_ms(250, 5000, 1) for _ in range(200)]
        samples_a3 = [jitter_delay_ms(250, 5000, 3) for _ in range(200)]

        avg_a1 = sum(samples_a1) / len(samples_a1)
        avg_a3 = sum(samples_a3) / len(samples_a3)

        # attempt=1: capped = min(250*2, 5000) = 500, mean jitter = 1.0, mean = 500
        # attempt=3: capped = min(250*8, 5000) = 2000, mean jitter = 1.0, mean = 2000
        # So avg_a3 should be ~4x avg_a1
        assert avg_a3 > avg_a1 * 2.5, (
            f"Expected exponential growth: avg_a3={avg_a3:.1f} should be "
            f"much larger than avg_a1={avg_a1:.1f}"
        )

    def test_caps_at_max_ms(self) -> None:
        from headroom.proxy.helpers import jitter_delay_ms

        # At attempt=20: capped = min(250 * 2^20, 5000) = 5000
        # Result = 5000 * (0.5 + random()) => [2500, 7500)
        for _ in range(50):
            val = jitter_delay_ms(base_ms=250, max_ms=5000, attempt=20)
            assert 2500.0 <= val < 7500.0, f"attempt=20 yielded {val}, expected [2500, 7500)"

    def test_never_negative(self) -> None:
        from headroom.proxy.helpers import jitter_delay_ms

        for attempt in range(10):
            val = jitter_delay_ms(base_ms=100, max_ms=1000, attempt=attempt)
            assert val > 0, f"jitter_delay_ms returned non-positive: {val}"

    def test_jitter_produces_variance(self) -> None:
        """Multiple calls with the same parameters should produce different results."""
        from headroom.proxy.helpers import jitter_delay_ms

        values = {jitter_delay_ms(250, 5000, 2) for _ in range(20)}
        # With randomness, we should get many distinct values
        assert len(values) > 10, f"Expected variance, got only {len(values)} distinct values"


# ---------------------------------------------------------------------------
# _headroom_log_dir lazy resolution tests
# ---------------------------------------------------------------------------


class TestHeadroomLogDir:
    """Tests for _headroom_log_dir using headroom.paths.log_dir."""

    def test_log_dir_respects_workspace_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from headroom.proxy.helpers import _headroom_log_dir

        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
        result = _headroom_log_dir()
        # log_dir should be under the workspace dir
        assert str(tmp_path) in str(result)

    def test_log_dir_returns_path_object(self) -> None:
        from headroom.proxy.helpers import _headroom_log_dir

        result = _headroom_log_dir()
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# asyncio.timeout shim tests
# ---------------------------------------------------------------------------


class TestAsyncioTimeoutShim:
    """Tests for the asyncio.timeout compatibility shim in repro_codex_replay.py."""

    def _get_shim(self):
        """Import the shim from the script."""
        scripts_dir = ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import repro_codex_replay

        return repro_codex_replay._asyncio_timeout

    @pytest.mark.asyncio
    async def test_shim_does_not_raise_when_block_completes_in_time(self) -> None:
        timeout_ctx = self._get_shim()
        # Should complete without raising
        async with timeout_ctx(5.0):
            await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_shim_raises_timeout_error_when_deadline_exceeded(self) -> None:
        timeout_ctx = self._get_shim()
        with pytest.raises(asyncio.TimeoutError):
            async with timeout_ctx(0.05):
                await asyncio.sleep(5.0)

    @pytest.mark.asyncio
    async def test_shim_with_none_delay_does_not_timeout(self) -> None:
        timeout_ctx = self._get_shim()
        # None means no timeout
        async with timeout_ctx(None):
            await asyncio.sleep(0.01)

    def test_shim_uses_stdlib_on_python_311_plus(self) -> None:
        """On Python 3.11+, the shim should reference asyncio.timeout directly."""
        if sys.version_info >= (3, 11):
            timeout_ctx = self._get_shim()
            assert timeout_ctx is asyncio.timeout


# ---------------------------------------------------------------------------
# Release workflow --allow-same-version tests
# ---------------------------------------------------------------------------


class TestReleaseWorkflowAllowSameVersion:
    """Validate that --allow-same-version is present on all npm version calls."""

    def test_all_npm_version_calls_have_allow_same_version(self) -> None:
        workflow_path = ROOT / ".github" / "workflows" / "release.yml"
        content = workflow_path.read_text(encoding="utf-8")

        # Find all lines with `npm version`
        npm_version_lines = [
            line.strip()
            for line in content.splitlines()
            if "npm version" in line and "npm_version" not in line.split("npm version")[0].rstrip()
        ]

        # Filter to actual npm version command invocations (not comments or env refs)
        command_lines = [
            line for line in npm_version_lines if not line.startswith("#") and "${{" in line
        ]

        assert len(command_lines) > 0, "Expected at least one npm version command in release.yml"

        for line in command_lines:
            assert "--allow-same-version" in line, (
                f"npm version call missing --allow-same-version flag:\n  {line}\n"
                "This flag prevents failures when re-running releases with the same version."
            )

    def test_all_npm_version_calls_have_no_git_tag_version(self) -> None:
        """npm version in CI should not create git tags (handled by the release job)."""
        workflow_path = ROOT / ".github" / "workflows" / "release.yml"
        content = workflow_path.read_text(encoding="utf-8")

        npm_version_lines = [
            line.strip()
            for line in content.splitlines()
            if "npm version" in line and "${{" in line and not line.startswith("#")
        ]

        for line in npm_version_lines:
            assert "--no-git-tag-version" in line, (
                f"npm version call missing --no-git-tag-version flag:\n  {line}"
            )


# ---------------------------------------------------------------------------
# SIGKILL fallback (Windows compatibility) tests
# ---------------------------------------------------------------------------


class TestKillSignalFallback:
    """Tests for the SIGKILL -> SIGTERM fallback in wrap.py."""

    def test_sigkill_available_on_unix_platforms(self) -> None:
        """On Unix, signal.SIGKILL should exist and be used."""
        if sys.platform == "win32":
            pytest.skip("SIGKILL not available on Windows")
        kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        assert kill_signal == signal.SIGKILL

    def test_fallback_to_sigterm_when_sigkill_missing(self) -> None:
        """When SIGKILL is not available (Windows), getattr falls back to SIGTERM."""
        # Simulate the pattern used in wrap.py
        # On Windows, signal.SIGKILL doesn't exist
        import types

        fake_signal = types.SimpleNamespace(SIGTERM=15)
        kill_signal = getattr(fake_signal, "SIGKILL", fake_signal.SIGTERM)
        assert kill_signal == 15

    def test_actual_platform_fallback_pattern(self) -> None:
        """The actual getattr pattern in wrap.py works on this platform."""
        _kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        # On any platform, we should get a valid signal number
        assert isinstance(_kill_signal, int | signal.Signals)


# ---------------------------------------------------------------------------
# LatencyHistogram (from repro script) tests
# ---------------------------------------------------------------------------


class TestLatencyHistogram:
    """Tests for the LatencyHistogram dataclass used in the repro harness."""

    def _get_histogram_class(self):
        scripts_dir = ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import repro_codex_replay

        return repro_codex_replay.LatencyHistogram

    def test_empty_histogram_returns_zeros(self) -> None:
        LatencyHistogram = self._get_histogram_class()
        h = LatencyHistogram()
        summary = h.as_summary()
        assert summary["count"] == 0
        assert summary["p50"] == 0.0
        assert summary["p99"] == 0.0

    def test_single_sample_is_all_percentiles(self) -> None:
        LatencyHistogram = self._get_histogram_class()
        h = LatencyHistogram()
        h.record(42.0)
        summary = h.as_summary()
        assert summary["count"] == 1
        assert summary["p50"] == 42.0
        assert summary["p99"] == 42.0
        assert summary["max"] == 42.0

    def test_percentile_ordering(self) -> None:
        LatencyHistogram = self._get_histogram_class()
        h = LatencyHistogram()
        for v in [1.0, 2.0, 3.0, 50.0, 100.0, 200.0, 500.0, 900.0, 950.0, 999.0]:
            h.record(v)
        summary = h.as_summary()
        assert summary["p50"] <= summary["p95"] <= summary["p99"] <= summary["max"]
        assert summary["count"] == 10

    def test_percentile_boundary_zero(self) -> None:
        LatencyHistogram = self._get_histogram_class()
        h = LatencyHistogram()
        for v in [10.0, 20.0, 30.0]:
            h.record(v)
        assert h.percentile(0) == 10.0

    def test_percentile_boundary_hundred(self) -> None:
        LatencyHistogram = self._get_histogram_class()
        h = LatencyHistogram()
        for v in [10.0, 20.0, 30.0]:
            h.record(v)
        assert h.percentile(100) == 30.0


# ---------------------------------------------------------------------------
# is_anthropic_auth tests
# ---------------------------------------------------------------------------


class TestIsAnthropicAuth:
    """Tests for headroom.proxy.helpers.is_anthropic_auth."""

    def test_detects_x_api_key(self) -> None:
        from headroom.proxy.helpers import is_anthropic_auth

        assert is_anthropic_auth({"x-api-key": "sk-ant-abc123"}) is True

    def test_detects_anthropic_version_header(self) -> None:
        from headroom.proxy.helpers import is_anthropic_auth

        assert is_anthropic_auth({"anthropic-version": "2023-06-01"}) is True

    def test_detects_bearer_sk_ant_prefix(self) -> None:
        from headroom.proxy.helpers import is_anthropic_auth

        assert is_anthropic_auth({"authorization": "Bearer sk-ant-abc123"}) is True

    def test_rejects_openai_bearer_token(self) -> None:
        from headroom.proxy.helpers import is_anthropic_auth

        assert is_anthropic_auth({"authorization": "Bearer sk-openai-xyz"}) is False

    def test_rejects_empty_headers(self) -> None:
        from headroom.proxy.helpers import is_anthropic_auth

        assert is_anthropic_auth({}) is False

    def test_rejects_non_anthropic_auth(self) -> None:
        from headroom.proxy.helpers import is_anthropic_auth

        assert is_anthropic_auth({"authorization": "Bearer some-token"}) is False


# ---------------------------------------------------------------------------
# _setup_file_logging tests
# ---------------------------------------------------------------------------


class TestSetupFileLogging:
    """Tests for _setup_file_logging using the new _headroom_log_dir path."""

    def test_setup_file_logging_creates_log_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from headroom.proxy.helpers import _setup_file_logging

        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
        # Clear any cached handlers to allow fresh registration
        import logging
        from logging.handlers import RotatingFileHandler

        headroom_logger = logging.getLogger("headroom")
        headroom_logger.handlers = [
            h for h in headroom_logger.handlers if not isinstance(h, RotatingFileHandler)
        ]

        _setup_file_logging()

        # Verify a RotatingFileHandler was added
        has_rotating = any(isinstance(h, RotatingFileHandler) for h in headroom_logger.handlers)
        assert has_rotating, "Expected a RotatingFileHandler to be registered"

    def test_setup_file_logging_handles_oserror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_setup_file_logging should not raise on OSError."""
        from headroom.proxy.helpers import _setup_file_logging

        # Monkey-patch _headroom_log_dir to return a path that will cause OSError
        def _bad_log_dir():
            return Path("/nonexistent/deeply/nested/path/that/cannot/exist/___test___")

        import headroom.proxy.helpers as helpers_mod

        monkeypatch.setattr(helpers_mod, "_headroom_log_dir", _bad_log_dir)
        # Should not raise
        _setup_file_logging()

    def test_importing_server_does_not_install_file_handler(self, tmp_path: Path) -> None:
        """Importing headroom.proxy.server must NOT attach a RotatingFileHandler
        to the user's live proxy.log. The handler is installed by create_app()
        instead, so test runs and library imports do not pollute logs.
        """
        import os
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import logging
            from logging.handlers import RotatingFileHandler
            import headroom.proxy.server  # noqa: F401
            hr = logging.getLogger("headroom")
            installed = any(isinstance(h, RotatingFileHandler) for h in hr.handlers)
            print("INSTALLED" if installed else "CLEAN")
            """
        ).strip()
        env = {**os.environ, "HEADROOM_WORKSPACE_DIR": str(tmp_path)}
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert result.stdout.strip() == "CLEAN", (
            f"Importing headroom.proxy.server attached a file handler: {result.stdout!r}"
        )


# ---------------------------------------------------------------------------
# Repro script URL helpers and stats tests
# ---------------------------------------------------------------------------


class TestReproScriptHelpers:
    """Tests for helper functions in scripts/repro_codex_replay.py."""

    def _import_repro(self):
        scripts_dir = ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import repro_codex_replay

        return repro_codex_replay

    def test_http_to_ws_url_http(self) -> None:
        mod = self._import_repro()
        result = mod._http_to_ws_url("http://127.0.0.1:8787", "/v1/responses")
        assert result == "ws://127.0.0.1:8787/v1/responses"

    def test_http_to_ws_url_https(self) -> None:
        mod = self._import_repro()
        result = mod._http_to_ws_url("https://example.com:443", "/v1/responses")
        assert result == "wss://example.com:443/v1/responses"

    def test_http_to_ws_url_normalizes_path(self) -> None:
        mod = self._import_repro()
        result = mod._http_to_ws_url("http://localhost:9000", "v1/responses")
        assert result == "ws://localhost:9000/v1/responses"

    def test_http_to_ws_url_empty_path(self) -> None:
        mod = self._import_repro()
        result = mod._http_to_ws_url("http://localhost:9000", "")
        assert result == "ws://localhost:9000"

    def test_classify_exit_proxy_unreachable(self) -> None:
        mod = self._import_repro()
        result = {"reason": "proxy_unreachable"}
        assert mod._classify_exit(result) == mod.EXIT_PROXY_UNREACHABLE

    def test_classify_exit_warmup_failed(self) -> None:
        mod = self._import_repro()
        result = {"warmup": {"skipped": False, "success": False}, "ok": False}
        assert mod._classify_exit(result) == mod.EXIT_WARMUP_FAILED

    def test_classify_exit_livez_threshold(self) -> None:
        mod = self._import_repro()
        result = {
            "warmup": {"skipped": True},
            "livez": {"threshold_ok": False},
            "ok": False,
        }
        assert mod._classify_exit(result) == mod.EXIT_LIVEZ_THRESHOLD

    def test_classify_exit_ok(self) -> None:
        mod = self._import_repro()
        result = {
            "warmup": {"skipped": True},
            "livez": {"threshold_ok": True},
            "ok": True,
        }
        assert mod._classify_exit(result) == mod.EXIT_OK

    def test_classify_exit_crash(self) -> None:
        mod = self._import_repro()
        result = {
            "warmup": {"skipped": True},
            "livez": {"threshold_ok": True},
            "ok": False,
        }
        assert mod._classify_exit(result) == mod.EXIT_CRASH

    def test_format_summary_proxy_unreachable(self) -> None:
        mod = self._import_repro()
        result = {
            "reason": "proxy_unreachable",
            "url": "http://127.0.0.1:8787",
            "detail": "ConnectionRefusedError: ...",
        }
        output = mod.format_summary(result)
        assert "unreachable" in output.lower()
        assert "127.0.0.1:8787" in output

    def test_format_summary_full_result(self) -> None:
        mod = self._import_repro()
        result = {
            "ok": True,
            "warmup": {"skipped": False, "success": True, "elapsed_ms": 50.0, "note": "ok"},
            "storm": {
                "ws_clients": 8,
                "anthropic_clients": 4,
                "requested_duration_s": 30,
                "actual_duration_s": 30.5,
            },
            "livez": {
                "count": 100,
                "p50": 5.0,
                "p95": 10.0,
                "p99": 15.0,
                "max": 20.0,
                "threshold_ms": 500,
                "threshold_ok": True,
            },
            "codex_ws": {"opened": 8, "response_completed": 4, "errors": {}},
            "anthropic_http": {
                "attempted": 4,
                "ok_2xx": 4,
                "non_2xx": 0,
                "timed_out": 0,
                "errors": 0,
                "avg_first_byte_ms": 25.0,
            },
        }
        output = mod.format_summary(result)
        assert "OK" in output
        assert "ws_clients=8" in output

    def test_format_summary_warmup_skipped(self) -> None:
        mod = self._import_repro()
        result = {
            "ok": True,
            "warmup": {"skipped": True},
            "storm": {
                "ws_clients": 2,
                "anthropic_clients": 1,
                "requested_duration_s": 5,
                "actual_duration_s": 5.1,
            },
            "livez": {
                "count": 20,
                "p50": 2.0,
                "p95": 5.0,
                "p99": 8.0,
                "max": 10.0,
                "threshold_ms": 500,
                "threshold_ok": True,
            },
            "codex_ws": {"opened": 2, "response_completed": 0, "errors": {}},
            "anthropic_http": {
                "attempted": 1,
                "ok_2xx": 1,
                "non_2xx": 0,
                "timed_out": 0,
                "errors": 0,
                "avg_first_byte_ms": 10.0,
            },
        }
        output = mod.format_summary(result)
        assert "skipped" in output.lower()

    def test_build_parser_defaults(self) -> None:
        mod = self._import_repro()
        parser = mod.build_parser()
        args = parser.parse_args([])
        assert args.url == "http://127.0.0.1:8787"
        assert args.ws_clients == 8
        assert args.anthropic_clients == 4
        assert args.duration == 30.0
        assert args.livez_threshold_ms == 500.0
        assert args.no_warmup is False
        assert args.json is False

    def test_build_parser_custom_args(self) -> None:
        mod = self._import_repro()
        parser = mod.build_parser()
        args = parser.parse_args(
            [
                "--url",
                "http://localhost:9999",
                "--ws-clients",
                "2",
                "--anthropic-clients",
                "1",
                "--duration",
                "10",
                "--no-warmup",
                "--json",
            ]
        )
        assert args.url == "http://localhost:9999"
        assert args.ws_clients == 2
        assert args.anthropic_clients == 1
        assert args.duration == 10.0
        assert args.no_warmup is True
        assert args.json is True


# ---------------------------------------------------------------------------
# Repro script stats dataclass tests
# ---------------------------------------------------------------------------


class TestReproScriptStats:
    """Tests for stat tracking dataclasses in the repro harness."""

    def _import_repro(self):
        scripts_dir = ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import repro_codex_replay

        return repro_codex_replay

    def test_codex_ws_stats_record_error(self) -> None:
        mod = self._import_repro()
        stats = mod.CodexWsStats()
        stats.record_error("connect:OSError")
        stats.record_error("connect:OSError")
        stats.record_error("ws:InvalidStatus")
        assert stats.errors == {"connect:OSError": 2, "ws:InvalidStatus": 1}

    def test_anthropic_http_stats_avg_first_byte(self) -> None:
        mod = self._import_repro()
        stats = mod.AnthropicHttpStats()
        assert stats.avg_first_byte_ms == 0.0
        stats.first_byte_latency_ms = [10.0, 20.0, 30.0]
        assert stats.avg_first_byte_ms == 20.0

    def test_anthropic_http_stats_initial_state(self) -> None:
        mod = self._import_repro()
        stats = mod.AnthropicHttpStats()
        assert stats.attempted == 0
        assert stats.ok_2xx == 0
        assert stats.non_2xx == 0
        assert stats.timed_out == 0
        assert stats.errors == 0


# ---------------------------------------------------------------------------
# wrap.py _get_log_path using paths module
# ---------------------------------------------------------------------------


class TestWrapGetLogPath:
    """Tests for _get_log_path in wrap.py using headroom.paths."""

    def test_get_log_path_returns_proxy_log(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from headroom.cli.wrap import _get_log_path

        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
        result = _get_log_path()
        assert result.name == "proxy.log"
        assert str(tmp_path) in str(result)

    def test_get_log_path_creates_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from headroom.cli.wrap import _get_log_path

        log_subdir = tmp_path / "custom_logs"
        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(log_subdir))
        result = _get_log_path()
        assert result.parent.exists()
