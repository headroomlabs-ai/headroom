"""Tests for `headroom wrap droid`.

`wrap droid` routes Factory Droid through Headroom by pointing Droid's gateway
at the local proxy via ``FACTORY_API_BASE_URL`` and forwarding to the resolved
Factory upstream. These tests pin the env wiring and upstream precedence. The
command does no filesystem surgery (the CLI context tools were removed upstream
in #2677), but every test still runs from a tmp cwd as a guard.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main
from headroom.providers.droid import (
    DEFAULT_FACTORY_API_URL,
    proxy_base_url,
    resolve_factory_upstream,
)
from headroom.providers.droid import runtime as droid_runtime

# Kept as module-level names rather than inline string literals in the `in`
# checks below: CodeQL's incomplete-url-substring-sanitization query pattern
# matches a URL-shaped StringLiteral used directly as an operand of `in`,
# regardless of context (it cannot tell a test assertion on captured CLI
# output apart from a real hostname-allowlist check). A name reference is a
# different AST node, so this avoids the false-positive match structurally.
_FACTORY_DOCS_URL = "https://docs.factory.ai"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FACTORY_API_BASE_URL", raising=False)


@pytest.fixture(autouse=True)
def _no_running_proxy():  # type: ignore[no-untyped-def]
    """Default: no proxy is running, so the automatic-port logic never probes the
    real machine's ports (non-deterministic). Tests that need a running proxy
    re-patch ``_check_proxy`` explicitly inside their own ``with`` block."""
    with patch("headroom.cli.wrap._check_proxy", return_value=False):
        yield


# ---------------------------------------------------------------------------
# runtime helpers
# ---------------------------------------------------------------------------


def test_proxy_base_url_targets_loopback_port() -> None:
    assert proxy_base_url(9999) == "http://127.0.0.1:9999"


def test_resolve_factory_upstream_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FACTORY_API_BASE_URL", raising=False)
    assert resolve_factory_upstream(None) == DEFAULT_FACTORY_API_URL

    monkeypatch.setenv("FACTORY_API_BASE_URL", "https://eu.factory.example/")
    assert resolve_factory_upstream(None) == "https://eu.factory.example"

    # Explicit flag beats the ambient env var and trailing slashes are trimmed.
    assert resolve_factory_upstream("https://custom.factory.test/") == "https://custom.factory.test"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("HTTPS://API.FACTORY.AI:443/", "https://api.factory.ai"),
        ("http://factory.example:80/path/", "http://factory.example/path"),
        ("https://factory.example:8443/v1/", "https://factory.example:8443/v1"),
        ("https://gateway.example/v1/", "https://gateway.example/v1"),
        ("ftp://api.factory.ai", None),
        ("https://user@api.factory.ai", None),
        ("https://api.factory.ai?tenant=a", None),
        ("https://api.factory.ai#fragment", None),
        ("https://api.factory.ai:bad", None),
        ("https://factory example/v1", None),
        ("https://factory%20example/v1", None),
        (r"https://api.factory.ai\evil/v1", None),
        ("http://127.1:8787", None),
        ("http://[::1]:8787", None),
        ("http://factory.localhost:8787", None),
    ],
)
def test_canonical_factory_api_url_is_strict(
    value: str,
    expected: str | None,
) -> None:
    assert droid_runtime.canonical_factory_api_url(value) == expected


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_wrap_droid_missing_binary_exits_with_install_hint(runner: CliRunner) -> None:
    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "droid"])

    assert result.exit_code == 1
    assert _FACTORY_DOCS_URL in result.output


def test_wrap_droid_rejects_unsafe_factory_upstream(runner: CliRunner) -> None:
    result = runner.invoke(
        main,
        ["wrap", "droid", "--factory-api-url", "http://localhost:8787"],
    )

    assert result.exit_code == 1
    assert "Factory upstream must be an HTTP(S) URL" in result.output


def test_wrap_droid_points_child_at_proxy_and_forwards_default_upstream(
    runner: CliRunner,
) -> None:
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="droid"),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(main, ["wrap", "droid", "--", "exec", "say hi"])

    assert result.exit_code == 0, result.output
    assert captured["tool_label"] == "DROID"
    assert captured["agent_type"] == "droid"
    assert captured["args"] == ("exec", "say hi")
    # Proxy forwards to the public Factory gateway by default...
    assert captured["factory_api_url"] == DEFAULT_FACTORY_API_URL
    # ...and Droid is redirected at the local proxy.
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["FACTORY_API_BASE_URL"] == "http://127.0.0.1:8787"
    display = captured["env_vars_display"]
    assert isinstance(display, list)
    assert "FACTORY_API_BASE_URL=http://127.0.0.1:8787" in display


def test_wrap_droid_explicit_upstream_and_custom_port(runner: CliRunner) -> None:
    captured: dict[str, object] = {}

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="droid"),
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(
            main,
            [
                "wrap",
                "droid",
                "--port",
                "9191",
                "--factory-api-url",
                "https://custom.factory.test/",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["factory_api_url"] == "https://custom.factory.test"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["FACTORY_API_BASE_URL"] == "http://127.0.0.1:9191"


def test_wrap_droid_inherits_ambient_factory_base_url_as_upstream(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FACTORY_API_BASE_URL", "https://eu.factory.example")
    captured: dict[str, object] = {}

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="droid"),
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(main, ["wrap", "droid"])

    assert result.exit_code == 0, result.output
    # The caller's existing gateway becomes the upstream the proxy forwards to,
    # while the child is still redirected at the local proxy.
    assert captured["factory_api_url"] == "https://eu.factory.example"
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["FACTORY_API_BASE_URL"] == "http://127.0.0.1:8787"


def test_wrap_droid_prepare_only_reports_wiring(runner: CliRunner) -> None:
    with patch("headroom.cli.wrap.shutil.which", return_value="droid"):
        result = runner.invoke(main, ["wrap", "droid", "--prepare-only"])

    assert result.exit_code == 0, result.output
    assert "FACTORY_API_BASE_URL=http://127.0.0.1:8787" in result.output
    assert f"upstream={DEFAULT_FACTORY_API_URL}" in result.output


def test_wrap_droid_does_not_write_agents_md(runner: CliRunner, tmp_path: Path) -> None:
    """Droid inference routing is proxy-only; it never does filesystem surgery."""
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="droid"),
        patch("headroom.cli.wrap._launch_tool"),
    ):
        result = runner.invoke(main, ["wrap", "droid"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "AGENTS.md").exists()


def test_wrap_droid_rejects_retired_context_tool_flag(runner: CliRunner) -> None:
    """The CLI context tools were removed upstream (#2677); the flag is rejected."""
    with patch("headroom.cli.wrap.shutil.which", return_value="droid"):
        result = runner.invoke(main, ["wrap", "droid", "--no-context-tool"])

    assert result.exit_code != 0
    assert "have been removed" in result.output


def test_wrap_droid_falls_back_to_free_port_when_incompatible(runner: CliRunner) -> None:
    """A non-Factory proxy on the port -> start a dedicated proxy on a free port."""
    captured: dict[str, object] = {}
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="droid"),
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch("headroom.cli.wrap._query_proxy_config", return_value={"factory_api_url": None}),
        patch("headroom.cli.wrap._find_available_port", return_value=8790),
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(main, ["wrap", "droid"])
    assert result.exit_code == 0, result.output
    assert captured["port"] == 8790
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["FACTORY_API_BASE_URL"] == "http://127.0.0.1:8790"


def test_wrap_droid_reuses_matching_factory_proxy(runner: CliRunner) -> None:
    """A Factory proxy for the SAME upstream on the port is reused (no fallback)."""
    captured: dict[str, object] = {}
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="droid"),
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch(
            "headroom.cli.wrap._query_proxy_config",
            return_value={"factory_api_url": DEFAULT_FACTORY_API_URL},
        ),
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(main, ["wrap", "droid"])
    assert result.exit_code == 0, result.output
    assert captured["port"] == 8787


def test_wrap_droid_no_proxy_reuses_normalized_matching_factory_proxy(
    runner: CliRunner,
) -> None:
    captured: dict[str, object] = {}
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="droid"),
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch(
            "headroom.cli.wrap._query_proxy_health",
            return_value={
                "service": "headroom-proxy",
                "config": {"factory_api_url": " HTTPS://TENANT.FACTORY.EXAMPLE:443/ "},
            },
        ),
        patch("headroom.cli.wrap._find_available_port") as find_available_port,
        patch("headroom.cli.wrap._launch_tool", side_effect=lambda **kw: captured.update(kw)),
    ):
        result = runner.invoke(
            main,
            [
                "wrap",
                "droid",
                "--no-proxy",
                "--factory-api-url",
                "https://tenant.factory.example/",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["port"] == 8787
    assert captured["no_proxy"] is True
    find_available_port.assert_not_called()


@pytest.mark.parametrize(
    "health_payload",
    [
        None,
        {"config": {"factory_api_url": DEFAULT_FACTORY_API_URL}},
        {
            "service": "other-service",
            "config": {"factory_api_url": DEFAULT_FACTORY_API_URL},
        },
        {"service": "headroom-proxy", "config": None},
    ],
)
def test_no_proxy_launch_path_rejects_unidentified_factory_listener(
    health_payload: dict[str, object] | None,
) -> None:
    with (
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch("headroom.cli.wrap._query_proxy_health", return_value=health_payload),
        pytest.raises(click.ClickException, match="compatible Factory proxy"),
    ):
        wrap_cli._ensure_proxy_unlocked(
            8787,
            True,
            factory_api_url=DEFAULT_FACTORY_API_URL,
        )


def test_no_proxy_launch_path_accepts_exact_factory_listener() -> None:
    health_payload = {
        "service": "headroom-proxy",
        "config": {"factory_api_url": "HTTPS://API.FACTORY.AI:443/"},
    }
    with (
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch("headroom.cli.wrap._query_proxy_health", return_value=health_payload),
    ):
        assert wrap_cli._ensure_proxy_unlocked(
            8787,
            True,
            factory_api_url=DEFAULT_FACTORY_API_URL,
        ) == (None, 8787)


@pytest.mark.parametrize("factory_api_url", [None, "https://api.factory.ai"])
def test_start_proxy_forwards_factory_api_url_to_subprocess(
    tmp_path: Path,
    factory_api_url: str | None,
) -> None:
    import headroom.cli.wrap as wrap_module

    captured: dict[str, object] = {}

    class _FakeProcess:
        returncode = None

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _FakeProcess()

    with (
        patch("headroom.cli.wrap.subprocess.Popen", side_effect=fake_popen),
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch("headroom.cli.wrap._get_log_path", return_value=tmp_path / "proxy.log"),
        patch("headroom.cli.wrap._get_proxy_stdio_log_path", return_value=tmp_path / "stdio.log"),
    ):
        wrap_module._start_proxy(8899, factory_api_url=factory_api_url)

    if factory_api_url is None:
        assert "--factory-api-url" not in captured["cmd"]
        assert "FACTORY_TARGET_API_URL" not in captured["env"]
    else:
        assert "--factory-api-url" in captured["cmd"]
        idx = captured["cmd"].index("--factory-api-url")
        assert captured["cmd"][idx + 1] == factory_api_url
        assert captured["env"]["FACTORY_TARGET_API_URL"] == factory_api_url


@pytest.mark.parametrize("proxy_running", [False, True])
def test_no_proxy_without_factory_target_keeps_legacy_behavior(proxy_running: bool) -> None:
    with patch("headroom.cli.wrap._check_proxy", return_value=proxy_running):
        assert wrap_cli._ensure_proxy_unlocked(8787, True) == (None, 8787)


def test_proxy_cli_banner_shows_factory_route_when_configured(runner: CliRunner) -> None:
    """`headroom proxy --factory-api-url ...` prints the Factory Droid route
    line in its startup banner so users can visually confirm it is active.
    """
    with patch("headroom.proxy.server.run_server", lambda config, **kwargs: None):
        result = runner.invoke(
            main,
            ["proxy", "--factory-api-url", DEFAULT_FACTORY_API_URL],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert "/api/llm/a/v1/messages" in result.output
    assert DEFAULT_FACTORY_API_URL in result.output
    assert "Factory Droid" in result.output


def test_proxy_cli_banner_omits_factory_route_by_default(runner: CliRunner) -> None:
    with patch("headroom.proxy.server.run_server", lambda config, **kwargs: None):
        result = runner.invoke(main, ["proxy"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "Factory Droid" not in result.output
