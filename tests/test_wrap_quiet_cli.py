"""`headroom wrap` reduce-at-source: SAFE quiet-CLI env defaults for the launched
agent. They fill in only when the user hasn't set the value, augment (never
clobber) PYTEST_ADDOPTS, and are fully opt-out via HEADROOM_WRAP_QUIET."""

from __future__ import annotations

from headroom.cli.wrap import _configure_quiet_cli_env, _quiet_cli_enabled


def test_defaults_injected_into_empty_env(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_WRAP_QUIET", raising=False)
    env: dict[str, str] = {}
    written = _configure_quiet_cli_env(env)
    assert env["GIT_PAGER"] == "cat"
    assert env["PIP_QUIET"] == "1"
    assert env["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert env["npm_config_fund"] == "false"
    assert env["npm_config_audit"] == "false"
    assert env["npm_config_progress"] == "false"
    assert env["PYTEST_ADDOPTS"] == "-q"
    assert "GIT_PAGER" in written and "PYTEST_ADDOPTS" in written


def test_telemetry_and_nag_suppressors_injected(monkeypatch) -> None:
    # Telemetry pings, version nags, and first-run/logo banners are zero-signal
    # tokens; the SAFE reduce-at-source set suppresses them across common tools.
    monkeypatch.delenv("HEADROOM_WRAP_QUIET", raising=False)
    env: dict[str, str] = {}
    written = _configure_quiet_cli_env(env)
    expected = {
        "npm_config_update_notifier": "false",
        "DO_NOT_TRACK": "1",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "DOTNET_NOLOGO": "1",
        "NEXT_TELEMETRY_DISABLED": "1",
        "GATSBY_TELEMETRY_DISABLED": "1",
        "ASTRO_TELEMETRY_DISABLED": "1",
        "NG_CLI_ANALYTICS": "false",
        "HOMEBREW_NO_ENV_HINTS": "1",
    }
    for name, value in expected.items():
        assert env[name] == value
        assert name in written


def test_user_value_always_wins(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_WRAP_QUIET", raising=False)
    env = {"GIT_PAGER": "less -R", "PIP_QUIET": "0"}
    written = _configure_quiet_cli_env(env)
    assert env["GIT_PAGER"] == "less -R"  # untouched
    assert env["PIP_QUIET"] == "0"  # untouched
    assert "GIT_PAGER" not in written and "PIP_QUIET" not in written
    # ...but absent ones are still filled.
    assert env["npm_config_fund"] == "false"


def test_pytest_addopts_augmented_not_clobbered(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_WRAP_QUIET", raising=False)
    env = {"PYTEST_ADDOPTS": "-p no:cacheprovider"}
    _configure_quiet_cli_env(env)
    assert env["PYTEST_ADDOPTS"] == "-p no:cacheprovider -q"  # preserved + augmented
    # already-quiet stays as-is (no duplicate -q)
    env2 = {"PYTEST_ADDOPTS": "-q --tb=short"}
    written = _configure_quiet_cli_env(env2)
    assert env2["PYTEST_ADDOPTS"] == "-q --tb=short"
    assert "PYTEST_ADDOPTS" not in written


def test_opt_out_disables_injection(monkeypatch) -> None:
    for off in ("0", "false", "no", "OFF"):
        monkeypatch.setenv("HEADROOM_WRAP_QUIET", off)
        assert _quiet_cli_enabled() is False
        env: dict[str, str] = {}
        assert _configure_quiet_cli_env(env) == []
        assert env == {}  # nothing injected


def test_enabled_by_default_and_on_truthy(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_WRAP_QUIET", raising=False)
    assert _quiet_cli_enabled() is True
    monkeypatch.setenv("HEADROOM_WRAP_QUIET", "1")
    assert _quiet_cli_enabled() is True
