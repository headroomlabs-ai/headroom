from __future__ import annotations

import pytest
from click.testing import CliRunner

from headroom.providers.copilot.wrap import (
    COPILOT_BYOK_ENV_VARS,
    COPILOT_NATIVE_API_URL_ENV,
    build_launch_env,
    build_native_launch_env,
    native_api_url_supported,
)


def test_native_env_redirects_api_and_clears_all_byok_state() -> None:
    seeded = dict.fromkeys(COPILOT_BYOK_ENV_VARS, "stale")
    seeded["UNRELATED"] = "preserved"
    env, _ = build_native_launch_env(port=8890, environ=seeded, project="repo name")

    assert env[COPILOT_NATIVE_API_URL_ENV] == "http://127.0.0.1:8890/p/repo%20name"
    assert env["UNRELATED"] == "preserved"
    assert not any(variable in env for variable in COPILOT_BYOK_ENV_VARS)


def test_byok_builder_remains_disjoint_from_native_mode() -> None:
    env, _ = build_launch_env(
        port=8787,
        provider_type="openai",
        wire_api="responses",
        environ={},
    )
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "responses"
    assert COPILOT_NATIVE_API_URL_ENV not in env


def test_native_support_probe_distinguishes_unknown_and_unsupported(tmp_path) -> None:
    local = tmp_path / "local"
    assert native_api_url_supported(environ={"LOCALAPPDATA": str(local)}) is None

    bundle = local / "copilot" / "pkg" / "platform" / "1.0" / "app.js"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("no override here", encoding="utf-8")
    assert native_api_url_supported(environ={"LOCALAPPDATA": str(local)}) is False

    bundle.write_text("process.env.COPILOT_API_URL", encoding="utf-8")
    assert native_api_url_supported(environ={"LOCALAPPDATA": str(local)}) is True


def test_native_support_probe_skips_unreadable_bundle(monkeypatch, tmp_path) -> None:
    local = tmp_path / "local"
    bundle = local / "copilot" / "pkg" / "platform" / "1.0" / "app.js"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("process.env.COPILOT_API_URL", encoding="utf-8")

    def _unreadable(*_args, **_kwargs):
        raise OSError("synthetic unreadable bundle")

    monkeypatch.setattr("builtins.open", _unreadable)
    assert native_api_url_supported(environ={"LOCALAPPDATA": str(local)}) is False


def _invoke_native(
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str] | None = None,
    *,
    support: bool | None = True,
):
    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    captured: dict[str, object] = {}

    class Resolution:
        token = "copilot-token"
        api_url = "https://api.business.githubcopilot.com"
        refresh_oauth_token = "refresh-token"
        api_token_expires_at = 123.0

    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: False)
    monkeypatch.setattr(wrap_mod, "_require_copilot_subscription_resolution", lambda: Resolution())
    monkeypatch.setattr(wrap_mod, "_native_api_url_supported", lambda **_kwargs: support)
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(
        main,
        ["wrap", "copilot", "--native", "--port", "8890", *(extra or [])],
    )
    return result, captured


def test_implicit_oauth_uses_native_routing_without_flag(monkeypatch) -> None:
    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    captured: dict[str, object] = {}
    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: False)
    monkeypatch.setattr(wrap_mod, "has_oauth_auth", lambda: True)
    monkeypatch.setattr(wrap_mod, "resolve_client_bearer_token", lambda: "oauth-token")
    monkeypatch.setattr(
        wrap_mod, "resolve_copilot_api_url", lambda _token: "https://api.githubcopilot.com"
    )
    monkeypatch.setattr(wrap_mod, "_native_api_url_supported", lambda **_kwargs: True)
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **kwargs: captured.update(kwargs))

    result = CliRunner().invoke(
        main,
        ["wrap", "copilot", "--port", "8890", "--", "--model", "claude-sonnet-5"],
    )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert COPILOT_NATIVE_API_URL_ENV in env
    assert not any(variable in env for variable in COPILOT_BYOK_ENV_VARS)


def test_native_cli_routes_both_protocols_to_tenant_host(monkeypatch) -> None:
    result, captured = _invoke_native(monkeypatch)
    assert result.exit_code == 0, result.output
    assert captured["openai_api_url"] == "https://api.business.githubcopilot.com"
    assert captured["anthropic_api_url"] == "https://api.business.githubcopilot.com"
    env = captured["env"]
    assert isinstance(env, dict)
    assert COPILOT_NATIVE_API_URL_ENV in env
    assert not any(variable in env for variable in COPILOT_BYOK_ENV_VARS)


@pytest.mark.parametrize("extra", [["--wire-api", "responses"], ["--provider-type", "anthropic"]])
def test_native_cli_rejects_byok_only_options(monkeypatch, extra) -> None:
    result, captured = _invoke_native(monkeypatch, extra)
    assert result.exit_code != 0
    assert not captured


def test_native_cli_refuses_known_unsupported_bundle(monkeypatch) -> None:
    result, captured = _invoke_native(monkeypatch, support=False)
    assert result.exit_code != 0
    assert "COPILOT_API_URL" in result.output
    assert not captured


def test_native_cli_reports_unknown_support_in_verbose_mode(monkeypatch) -> None:
    result, captured = _invoke_native(monkeypatch, ["--verbose"], support=None)
    assert result.exit_code == 0, result.output
    assert "could not verify" in result.output
    assert captured
