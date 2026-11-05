"""Bind policy: a non-loopback bind with no inbound token is refused, not warned about.

Before this change the proxy logged ``proxy_open_bind`` and started anyway,
so any launcher that set ``--host 0.0.0.0`` / ``HEADROOM_HOST=0.0.0.0`` without
``HEADROOM_PROXY_TOKEN`` served the ``/v1/*`` relay to every peer on the
network. The policy is evaluated in :mod:`headroom.proxy.bind_policy` and
enforced at three layers -- ``create_app``, ``run_server`` and the CLI -- each
of which is covered here, plus the explicit operator acknowledgement
(``HEADROOM_ALLOW_UNAUTHENTICATED_BIND=1``) that keeps the documented
"container binds 0.0.0.0, runtime publishes on 127.0.0.1" shape working.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from click.testing import CliRunner  # noqa: E402

from headroom.cli.main import main  # noqa: E402
from headroom.exceptions import ConfigurationError  # noqa: E402
from headroom.proxy.bind_policy import (  # noqa: E402
    OPEN_BIND_ACK_ENV,
    PROXY_TOKEN_ENV,
    OpenBindRefused,
    enforce_bind_policy,
    evaluate_bind_policy,
)
from headroom.proxy.models import ProxyConfig  # noqa: E402
from headroom.proxy.server import create_app, run_server  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_bind_env(monkeypatch: pytest.MonkeyPatch):
    """The policy reads the process env; never let the developer's shell leak in."""
    monkeypatch.delenv(PROXY_TOKEN_ENV, raising=False)
    monkeypatch.delenv(OPEN_BIND_ACK_ENV, raising=False)


# ───────────────────────────── evaluate ────────────────────────────────────


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "myhost.internal", "[::]"])
def test_non_loopback_bind_without_token_is_refused(host: str) -> None:
    decision = evaluate_bind_policy(host, None, environ={})
    assert decision.open_bind is True
    assert decision.refused is True
    # The message must name both remedies so the operator can act on it.
    assert PROXY_TOKEN_ENV in decision.message()
    assert OPEN_BIND_ACK_ENV in decision.message()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.5.5.5", "::ffff:127.0.0.1"])
def test_loopback_bind_without_token_is_fine(host: str) -> None:
    decision = evaluate_bind_policy(host, None, environ={})
    assert decision.loopback is True
    assert decision.open_bind is False
    assert decision.refused is False


@pytest.mark.parametrize("host", [None, "", "   "])
def test_unset_host_means_uvicorn_default_loopback(host: str | None) -> None:
    """An absent host is uvicorn's 127.0.0.1, never 'unknown therefore public'."""
    decision = evaluate_bind_policy(host, None, environ={})
    assert decision.host == "127.0.0.1"
    assert decision.refused is False


def test_token_makes_any_bind_acceptable() -> None:
    assert evaluate_bind_policy("0.0.0.0", "s3cr3t", environ={}).refused is False
    # Same rule the security gate uses: the env var is the fallback source.
    assert (
        evaluate_bind_policy("0.0.0.0", None, environ={PROXY_TOKEN_ENV: "s3cr3t"}).refused is False
    )
    # Whitespace-only is not a token.
    assert evaluate_bind_policy("0.0.0.0", "   ", environ={}).refused is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_acknowledgement_env_allows_the_open_bind(value: str) -> None:
    decision = evaluate_bind_policy("0.0.0.0", None, environ={OPEN_BIND_ACK_ENV: value})
    assert decision.open_bind is True
    assert decision.acknowledged is True
    assert decision.refused is False


@pytest.mark.parametrize("value", ["0", "false", "no", "", "maybe"])
def test_non_truthy_acknowledgement_still_refuses(value: str) -> None:
    assert evaluate_bind_policy("0.0.0.0", None, environ={OPEN_BIND_ACK_ENV: value}).refused


# ───────────────────────────── enforce ─────────────────────────────────────


def test_enforce_raises_a_configuration_error_subclass() -> None:
    with pytest.raises(OpenBindRefused) as exc_info:
        enforce_bind_policy("0.0.0.0", None, environ={})
    assert isinstance(exc_info.value, ConfigurationError)
    assert exc_info.value.decision.host == "0.0.0.0"


def test_enforce_keeps_the_open_bind_warning_when_acknowledged() -> None:
    """Log-based monitoring keyed on ``proxy_open_bind`` must keep working."""
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    log = logging.getLogger("headroom.proxy.bind_policy")
    log.addHandler(handler)
    previous = log.level
    log.setLevel(logging.DEBUG)
    try:
        decision = enforce_bind_policy("0.0.0.0", None, environ={OPEN_BIND_ACK_ENV: "1"})
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)
    assert decision.acknowledged is True
    messages = [r.getMessage() for r in records]
    assert any("event=proxy_open_bind " in m for m in messages), messages
    assert not any("proxy_open_bind_refused" in m for m in messages)


# ───────────────────────────── create_app ──────────────────────────────────


def _config(**overrides) -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        **overrides,
    )


def test_create_app_refuses_open_bind() -> None:
    with pytest.raises(OpenBindRefused):
        create_app(_config(host="0.0.0.0", proxy_token=None))


def test_create_app_accepts_open_bind_with_token() -> None:
    assert create_app(_config(host="0.0.0.0", proxy_token="s3cr3t")) is not None


def test_create_app_accepts_acknowledged_open_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPEN_BIND_ACK_ENV, "1")
    assert create_app(_config(host="0.0.0.0", proxy_token=None)) is not None


def test_create_app_default_loopback_bind_needs_nothing() -> None:
    assert create_app(_config()) is not None


# ───────────────────────────── run_server ──────────────────────────────────


def test_run_server_exits_before_uvicorn_on_open_bind(capsys: pytest.CaptureFixture[str]) -> None:
    """One clean exit code, not N workers dying in create_app."""
    called = []
    with (
        patch("headroom.proxy.server.uvicorn.run", lambda *a, **k: called.append(1)),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_server(_config(host="0.0.0.0", proxy_token=None), print_banner=False)
    assert exc_info.value.code == 2
    assert called == []
    err = capsys.readouterr().err
    assert "Refusing to bind '0.0.0.0'" in err
    assert PROXY_TOKEN_ENV in err and OPEN_BIND_ACK_ENV in err


def test_run_server_proceeds_with_token() -> None:
    called = []
    with patch("headroom.proxy.server.uvicorn.run", lambda *a, **k: called.append(k)):
        run_server(_config(host="0.0.0.0", proxy_token="s3cr3t"), print_banner=False)
    assert len(called) == 1
    assert called[0]["host"] == "0.0.0.0"


# ───────────────────────────── CLI ─────────────────────────────────────────


def test_cli_refuses_open_bind_with_actionable_error() -> None:
    runner = CliRunner()
    called = []
    with patch("headroom.proxy.server.run_server", lambda config, **kw: called.append(config)):
        result = runner.invoke(main, ["proxy", "--host", "0.0.0.0"])
    assert result.exit_code != 0, result.output
    assert called == []
    assert "Refusing to bind '0.0.0.0'" in result.output
    assert PROXY_TOKEN_ENV in result.output
    assert OPEN_BIND_ACK_ENV in result.output
    # A refusal is an error message, never a traceback.
    assert "Traceback" not in result.output


def test_cli_env_host_is_refused_too() -> None:
    """``HEADROOM_HOST=0.0.0.0`` in a systemd unit is the common real-world shape."""
    runner = CliRunner()
    with patch("headroom.proxy.server.run_server", lambda config, **kw: None):
        result = runner.invoke(main, ["proxy"], env={"HEADROOM_HOST": "0.0.0.0"})
    assert result.exit_code != 0, result.output
    assert "Refusing to bind '0.0.0.0'" in result.output


def test_cli_acknowledged_open_bind_starts_and_says_so() -> None:
    runner = CliRunner()
    captured = {}
    with patch(
        "headroom.proxy.server.run_server",
        lambda config, **kw: captured.setdefault("config", config),
    ):
        result = runner.invoke(
            main,
            ["proxy", "--host", "0.0.0.0"],
            env={OPEN_BIND_ACK_ENV: "1"},
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert captured["config"].host == "0.0.0.0"
    assert "UNAUTHENTICATED" in result.output
    assert OPEN_BIND_ACK_ENV in result.output


def test_cli_offline_banner_still_flags_acknowledged_open_bind() -> None:
    """Offline mode must not hide that /v1/* is unauthenticated."""
    runner = CliRunner()
    with patch("headroom.proxy.server.run_server", lambda config, **kw: None):
        result = runner.invoke(
            main,
            ["proxy", "--host", "0.0.0.0"],
            env={OPEN_BIND_ACK_ENV: "1", "HEADROOM_OFFLINE": "1"},
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert "OFFLINE" in result.output
    assert "UNAUTHENTICATED" in result.output


def test_cli_open_bind_with_token_starts() -> None:
    runner = CliRunner()
    captured = {}
    with patch(
        "headroom.proxy.server.run_server",
        lambda config, **kw: captured.setdefault("config", config),
    ):
        result = runner.invoke(
            main,
            ["proxy", "--host", "0.0.0.0"],
            env={PROXY_TOKEN_ENV: "s3cr3t"},
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert "inbound token REQUIRED" in result.output
