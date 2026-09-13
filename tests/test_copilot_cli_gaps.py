"""Copilot CLI native-lane hardening.

Each test names the failure it guards against. The Copilot CLI resolves its API
host as ``settings.copilotUrl || COPILOT_API_URL || token.endpoints.api``, honours
``HTTP_PROXY``/``HTTPS_PROXY`` for every request (including the loopback hop to
Headroom), and since 1.0.8x ships as a native binary inside a per-platform npm
package with ``app.js`` beside it. A wrapper that ignores any of these prints a
successful launch while traffic never reaches the proxy, or fails with a
connection error the user cannot attribute.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from headroom import copilot_auth
from headroom.providers.copilot import wrap as copilot_wrap
from headroom.providers.copilot.install import build_install_env, install_uses_native_lane
from headroom.providers.copilot.wrap import (
    COPILOT_NATIVE_API_URL_ENV,
    LOOPBACK_NO_PROXY_HOSTS,
    build_launch_env,
    build_native_launch_env,
    check_copilot_url_setting,
    copilot_home,
    ensure_loopback_no_proxy,
    native_api_url_supported,
    read_copilot_url_setting,
    read_copilot_url_settings,
)

PROXY_URL = "http://127.0.0.1:8787/p/repo"


@pytest.fixture(autouse=True)
def _clean_copilot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "GITHUB_COPILOT_API_URL",
        "GITHUB_COPILOT_ENTERPRISE_URL",
        "GITHUB_COPILOT_ENTERPRISE_DOMAIN",
        copilot_auth.USE_ADVERTISED_HOST_ENV,
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


def _write(path: Path, payload: object) -> None:
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Gap 4: the loopback hop must be exempt from a corporate proxy
# --------------------------------------------------------------------------- #
def test_loopback_no_proxy_is_added_and_idempotent() -> None:
    env: dict[str, str] = {}
    ensure_loopback_no_proxy(env)
    first = env["NO_PROXY"]
    assert set(first.split(",")) == set(LOOPBACK_NO_PROXY_HOSTS)

    ensure_loopback_no_proxy(env)
    assert env["NO_PROXY"] == first, "second call must not duplicate entries"
    assert "no_proxy" not in env, "lowercase variant is only touched when it already exists"


def test_loopback_no_proxy_preserves_existing_entries_and_syncs_lowercase() -> None:
    env = {"NO_PROXY": "corp.internal, .example.com", "no_proxy": "corp.internal"}
    ensure_loopback_no_proxy(env)
    for variable in ("NO_PROXY", "no_proxy"):
        entries = env[variable].split(",")
        assert entries[0] == "corp.internal"
        assert all(host in entries for host in LOOPBACK_NO_PROXY_HOSTS)


def test_both_cli_lanes_exempt_loopback_from_proxies() -> None:
    native_env, _ = build_native_launch_env(
        port=8787, environ={"HTTPS_PROXY": "http://proxy.corp:3128"}
    )
    byok_env, _ = build_launch_env(
        port=8787,
        provider_type="openai",
        wire_api=None,
        environ={"HTTPS_PROXY": "http://proxy.corp:3128"},
    )
    for env in (native_env, byok_env):
        assert "127.0.0.1" in env["NO_PROXY"].split(",")
        assert env["HTTPS_PROXY"] == "http://proxy.corp:3128", "the corporate proxy stays"


def test_subscription_lane_exempts_loopback_from_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """The --subscription branch builds its env inline, outside the shared builders."""
    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    class Resolution:
        token = "gho-subscription"
        api_url = copilot_auth.DEFAULT_API_URL
        refresh_oauth_token = None
        api_token_expires_at = None
        advertised_api_url = None

    captured: dict[str, object] = {}
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp:3128")
    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: False)
    monkeypatch.setattr(wrap_mod, "_require_copilot_subscription_resolution", lambda: Resolution())
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **kwargs: captured.update(kwargs))

    result = CliRunner().invoke(
        main, ["wrap", "copilot", "--subscription", "--", "--model", "gpt-5.5"]
    )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert "127.0.0.1" in env["NO_PROXY"].split(",")
    assert env["HTTPS_PROXY"] == "http://proxy.corp:3128"


# --------------------------------------------------------------------------- #
# Gap 1: the support probe must find the bundle of the binary being launched
# --------------------------------------------------------------------------- #
def _npm_layout(root: Path, *, mentions_env: bool) -> tuple[Path, Path]:
    """Model an ``npm install -g @github/copilot`` tree.

    ``bin/copilot`` is a symlink to the shim package's ``npm-loader.js``; the
    platform package next to it carries the native binary and ``app.js``.
    """
    scope = root / "lib" / "node_modules" / "@github"
    shim = scope / "copilot"
    platform = scope / "copilot-darwin-arm64"
    shim.mkdir(parents=True)
    platform.mkdir(parents=True)
    loader = shim / "npm-loader.js"
    loader.write_text("spawn platform package", encoding="utf-8")
    (platform / "copilot").write_bytes(b"\xcf\xfa\xed\xfe native binary")
    (platform / "app.js").write_text(
        "process.env.COPILOT_API_URL" if mentions_env else "no override here",
        encoding="utf-8",
    )
    bin_dir = root / "bin"
    bin_dir.mkdir()
    shim_link = bin_dir / "copilot"
    os.symlink(loader, shim_link)
    return shim_link, platform / "copilot"


def _no_legacy_roots(tmp_path: Path) -> dict[str, str]:
    """An environment in which the legacy ``pkg`` directories do not exist."""
    return {"LOCALAPPDATA": str(tmp_path / "nowhere"), "HOME": str(tmp_path / "nowhere")}


def test_probe_follows_the_npm_shim_to_the_platform_bundle(tmp_path: Path) -> None:
    shim_link, _native = _npm_layout(tmp_path, mentions_env=True)
    # The old probe answered None here: no `pkg` directory anywhere.
    assert native_api_url_supported(environ=_no_legacy_roots(tmp_path), copilot_bin=str(shim_link))


def test_probe_reads_the_bundle_beside_a_native_binary(tmp_path: Path) -> None:
    _shim, native = _npm_layout(tmp_path, mentions_env=True)
    assert native_api_url_supported(environ=_no_legacy_roots(tmp_path), copilot_bin=str(native))


def test_probe_handles_a_windows_npm_shim_script(tmp_path: Path) -> None:
    """``copilot.cmd`` is a script in the npm prefix; the packages hang off it."""
    prefix = tmp_path / "npm"
    platform = prefix / "node_modules" / "@github" / "copilot-win32-x64"
    platform.mkdir(parents=True)
    (platform / "app.js").write_text("process.env.COPILOT_API_URL", encoding="utf-8")
    shim = prefix / "copilot.cmd"
    shim.write_text("@node npm-loader.js %*", encoding="utf-8")
    assert native_api_url_supported(environ=_no_legacy_roots(tmp_path), copilot_bin=str(shim))


def test_probe_reports_unsupported_when_the_launched_bundle_lacks_the_hook(
    tmp_path: Path,
) -> None:
    shim_link, _native = _npm_layout(tmp_path, mentions_env=False)
    assert (
        native_api_url_supported(environ=_no_legacy_roots(tmp_path), copilot_bin=str(shim_link))
        is False
    )


def test_probe_stays_unknown_without_any_bundle(tmp_path: Path) -> None:
    env = _no_legacy_roots(tmp_path)
    assert native_api_url_supported(environ=env, copilot_bin=str(tmp_path / "missing")) is None
    assert native_api_url_supported(environ=env, copilot_bin=None) is None


# --------------------------------------------------------------------------- #
# Gap 2: a `copilotUrl` setting outranks COPILOT_API_URL
# --------------------------------------------------------------------------- #
def test_copilot_home_honours_override_then_default(tmp_path: Path) -> None:
    assert copilot_home({"COPILOT_HOME": str(tmp_path / "cfg")}) == tmp_path / "cfg"
    assert copilot_home({"HOME": str(tmp_path)}) == tmp_path / ".copilot"


def test_read_copilot_url_settings_reports_every_pin(tmp_path: Path) -> None:
    env = {"COPILOT_HOME": str(tmp_path)}
    assert read_copilot_url_settings(env) == []
    assert read_copilot_url_setting(env) is None

    _write(tmp_path / "config.json", {"copilotUrl": "https://legacy.example"})
    assert read_copilot_url_settings(env) == [(tmp_path / "config.json", "https://legacy.example")]

    _write(tmp_path / "settings.json", {"copilotUrl": "https://current.example/"})
    assert read_copilot_url_settings(env) == [
        (tmp_path / "settings.json", "https://current.example/"),
        (tmp_path / "config.json", "https://legacy.example"),
    ]
    assert read_copilot_url_setting(env) == (tmp_path / "settings.json", "https://current.example/")


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[]",
        json.dumps({"copilotUrl": "  "}),
        json.dumps({"copilotUrl": 123}),
        json.dumps({"copilotUrl": None}),
        json.dumps({"copilotUrl": ["http://127.0.0.1:1"]}),
        json.dumps({"copilotUrl": True}),
    ],
)
def test_read_copilot_url_settings_ignores_unusable_files(tmp_path: Path, payload: str) -> None:
    _write(tmp_path / "settings.json", payload)
    assert read_copilot_url_settings({"COPILOT_HOME": str(tmp_path)}) == []


def test_check_refuses_a_pin_on_another_origin(tmp_path: Path) -> None:
    _write(tmp_path / "settings.json", {"copilotUrl": "https://api.githubcopilot.com"})
    with pytest.raises(click.ClickException) as excinfo:
        check_copilot_url_setting(PROXY_URL, environ={"COPILOT_HOME": str(tmp_path)})
    assert "copilotUrl" in excinfo.value.message
    # The remedy names the bare origin, not this launch's project-prefixed URL,
    # so following it does not pin every future session to one project.
    assert "'http://127.0.0.1:8787'" in excinfo.value.message
    assert PROXY_URL not in excinfo.value.message


def test_check_refuses_when_only_the_legacy_file_pins_another_origin(tmp_path: Path) -> None:
    """The CLI still honours a legacy value it has not migrated; so must the check."""
    _write(tmp_path / "settings.json", {"copilotUrl": PROXY_URL})
    _write(tmp_path / "config.json", {"copilotUrl": "http://127.0.0.1:9999"})
    with pytest.raises(click.ClickException) as excinfo:
        check_copilot_url_setting(PROXY_URL, environ={"COPILOT_HOME": str(tmp_path)})
    assert "config.json" in excinfo.value.message


def test_check_refuses_the_same_host_on_a_different_port(tmp_path: Path) -> None:
    """Another Headroom instance is still not this one."""
    _write(tmp_path / "settings.json", {"copilotUrl": "http://127.0.0.1:9999/p/repo"})
    with pytest.raises(click.ClickException):
        check_copilot_url_setting(PROXY_URL, environ={"COPILOT_HOME": str(tmp_path)})


def test_check_accepts_a_pin_that_names_this_proxy_exactly(tmp_path: Path) -> None:
    _write(tmp_path / "settings.json", {"copilotUrl": "HTTP://127.0.0.1:8787/p/repo/"})
    check_copilot_url_setting(PROXY_URL, environ={"COPILOT_HOME": str(tmp_path)})


def test_check_accepts_the_same_origin_without_the_project_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A durable install wrote the bare proxy URL: traffic still flows through Headroom."""
    _write(tmp_path / "settings.json", {"copilotUrl": "http://127.0.0.1:8787"})
    check_copilot_url_setting(PROXY_URL, environ={"COPILOT_HOME": str(tmp_path)})
    out = capsys.readouterr().out
    assert out.startswith("  Note:")
    assert "copilotUrl='http://127.0.0.1:8787'" in out


def test_check_is_silent_without_any_pin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    check_copilot_url_setting(PROXY_URL, environ={"COPILOT_HOME": str(tmp_path)})
    assert capsys.readouterr().out == ""


def _native_resolution(advertised: str | None = None):
    class Resolution:
        token = "copilot-token"
        api_url = copilot_auth.DEFAULT_API_URL
        refresh_oauth_token = "refresh-token"
        api_token_expires_at = 123.0
        advertised_api_url = advertised

    return Resolution()


def test_native_wrap_fails_closed_on_a_foreign_copilot_url_pin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The user-visible contract: refuse rather than launch a bypassed session."""
    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    _write(tmp_path / "settings.json", {"copilotUrl": "https://api.githubcopilot.com"})
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path))

    captured: dict[str, object] = {}
    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: False)
    monkeypatch.setattr(
        wrap_mod, "_require_copilot_subscription_resolution", lambda: _native_resolution()
    )
    monkeypatch.setattr(wrap_mod, "_native_api_url_supported", lambda **_kwargs: True)
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **kwargs: captured.update(kwargs))

    result = CliRunner().invoke(main, ["wrap", "copilot", "--native", "--port", "8890"])

    assert result.exit_code != 0
    assert "copilotUrl" in result.output
    assert not captured, "the CLI must not be launched"


def test_native_wrap_passes_copilot_bin_to_the_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    seen: dict[str, object] = {}

    def probe(**kwargs):  # noqa: ANN003
        seen.update(kwargs)
        return True

    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/opt/copilot/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: False)
    monkeypatch.setattr(
        wrap_mod, "_require_copilot_subscription_resolution", lambda: _native_resolution()
    )
    monkeypatch.setattr(wrap_mod, "_native_api_url_supported", probe)
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **kwargs: None)

    result = CliRunner().invoke(main, ["wrap", "copilot", "--native", "--port", "8890"])

    assert result.exit_code == 0, result.output
    assert seen["copilot_bin"] == "/opt/copilot/bin/copilot"


# --------------------------------------------------------------------------- #
# Gap 3: persistent installs default to the native lane
# --------------------------------------------------------------------------- #
def test_install_lane_selection() -> None:
    assert install_uses_native_lane("anthropic", {}) is True
    assert install_uses_native_lane("", {}) is True
    assert install_uses_native_lane("anyllm", {}) is False
    assert install_uses_native_lane("litellm-vertex", {}) is False
    assert install_uses_native_lane("anthropic", {"COPILOT_PROVIDER_API_KEY": "sk-x"}) is False
    assert install_uses_native_lane(None, {}) is True  # planner may pass no backend at all


def test_native_install_env_is_only_the_api_url_hook() -> None:
    env = build_install_env(port=8891, backend="anthropic", environ={})
    assert env == {COPILOT_NATIVE_API_URL_ENV: "http://127.0.0.1:8891"}
    assert "NO_PROXY" not in env, "an install must not overwrite the user's NO_PROXY"


# --------------------------------------------------------------------------- #
# Gap 5: advertised per-plan and data-residency hosts
# --------------------------------------------------------------------------- #
def test_ghe_data_residency_host_is_never_folded_into_the_public_host() -> None:
    advertised = "https://copilot-api.acme.ghe.com"
    assert (
        copilot_auth._subscription_api_url_from_user_info_payload(
            {"endpoints": {"api": advertised}}
        )
        == advertised
    )


def test_ghe_host_from_a_token_exchange_is_honoured() -> None:
    resolved = copilot_auth._api_url_from_exchange_payload(
        {"endpoints": {"api": "https://copilot-api.acme.ghe.com"}}, oauth_token="gho-oauth"
    )
    assert resolved == "https://copilot-api.acme.ghe.com"


@pytest.mark.parametrize(
    "advertised",
    ["https://api.business.githubcopilot.com", "https://api.enterprise.githubcopilot.com"],
)
def test_segmented_plan_hosts_normalize_by_default_and_honour_the_opt_in(
    monkeypatch: pytest.MonkeyPatch, advertised: str
) -> None:
    payload = {"endpoints": {"api": advertised}}
    assert (
        copilot_auth._subscription_api_url_from_user_info_payload(payload)
        == copilot_auth.DEFAULT_API_URL
    )
    assert copilot_auth.advertised_host_is_normalized(advertised) is True

    monkeypatch.setenv(copilot_auth.USE_ADVERTISED_HOST_ENV, "1")
    assert copilot_auth._subscription_api_url_from_user_info_payload(payload) == advertised
    assert copilot_auth.advertised_host_is_normalized(advertised) is False


def test_explicit_pin_still_beats_the_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(copilot_auth.USE_ADVERTISED_HOST_ENV, "1")
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://egress.corp.example")
    assert (
        copilot_auth._subscription_api_url_from_user_info_payload(
            {"endpoints": {"api": "https://api.business.githubcopilot.com"}}
        )
        == "https://egress.corp.example"
    )


def test_individual_host_stays_normalized_even_with_the_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#610: the individual segmented host did not serve newer models."""
    monkeypatch.setenv(copilot_auth.USE_ADVERTISED_HOST_ENV, "true")
    assert (
        copilot_auth._subscription_api_url_from_user_info_payload(
            {"endpoints": {"api": "https://api.individual.githubcopilot.com"}}
        )
        == copilot_auth.DEFAULT_API_URL
    )


def test_resolution_records_the_advertised_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_COPILOT_API_TOKEN", raising=False)
    monkeypatch.delenv("COPILOT_PROVIDER_BEARER_TOKEN", raising=False)
    monkeypatch.setattr(
        copilot_auth,
        "iter_oauth_token_candidates",
        lambda: [
            copilot_auth.CopilotTokenCandidate(
                token="gho-oauth",
                source="headroom-copilot-auth:/tmp/copilot_auth.json",
                confidence="copilot-oauth",
            )
        ],
    )
    monkeypatch.setattr(
        copilot_auth.CopilotTokenProvider,
        "_exchange_token_sync",
        staticmethod(
            lambda _headers: {
                "token": "copilot-api",
                "expires_at": int(time.time()) + 3600,
                "endpoints": {"api": "https://api.business.githubcopilot.com"},
            }
        ),
    )

    resolution = copilot_auth.resolve_subscription_bearer_token_details()

    assert resolution is not None
    assert resolution.api_url == copilot_auth.DEFAULT_API_URL
    assert resolution.advertised_api_url == "https://api.business.githubcopilot.com"


def test_wrap_prints_the_advertised_host_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: False)
    monkeypatch.setattr(
        wrap_mod,
        "_require_copilot_subscription_resolution",
        lambda: _native_resolution("https://api.business.githubcopilot.com"),
    )
    monkeypatch.setattr(wrap_mod, "_native_api_url_supported", lambda **_kwargs: True)
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **kwargs: None)

    result = CliRunner().invoke(main, ["wrap", "copilot", "--native", "--port", "8890"])

    assert result.exit_code == 0, result.output
    assert "api.business.githubcopilot.com" in result.output
    assert copilot_auth.USE_ADVERTISED_HOST_ENV in result.output


def test_wrap_stays_quiet_when_the_advertised_host_is_the_one_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: False)
    monkeypatch.setattr(
        wrap_mod,
        "_require_copilot_subscription_resolution",
        lambda: _native_resolution(copilot_auth.DEFAULT_API_URL),
    )
    monkeypatch.setattr(wrap_mod, "_native_api_url_supported", lambda **_kwargs: True)
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **kwargs: None)

    result = CliRunner().invoke(main, ["wrap", "copilot", "--native", "--port", "8890"])

    assert result.exit_code == 0, result.output
    assert copilot_auth.USE_ADVERTISED_HOST_ENV not in result.output


# --------------------------------------------------------------------------- #
# Review follow-ups
# --------------------------------------------------------------------------- #
def test_loopback_no_proxy_seeds_uppercase_from_a_lowercase_only_value() -> None:
    """undici reads ``no_proxy`` first, reqwest and Go read ``NO_PROXY`` first;
    a fresh uppercase value missing the corporate entries would split them."""
    env = {"no_proxy": "corp.internal,.example.com"}
    ensure_loopback_no_proxy(env)
    assert env["NO_PROXY"] == env["no_proxy"]
    assert env["NO_PROXY"].split(",")[:2] == ["corp.internal", ".example.com"]
    assert "127.0.0.1" in env["NO_PROXY"].split(",")


def test_loopback_no_proxy_unions_disagreeing_spellings() -> None:
    env = {"NO_PROXY": "a.internal", "no_proxy": "b.internal"}
    ensure_loopback_no_proxy(env)
    assert env["NO_PROXY"] == env["no_proxy"]
    assert env["NO_PROXY"].split(",")[:2] == ["a.internal", "b.internal"]


def test_copilot_home_follows_node_homedir_on_windows() -> None:
    """Node ignores HOME on Windows; so must the lookup, or a Git-for-Windows
    shell with HOME set would look in the wrong place and fail open."""
    env = {"HOME": "/home/from-msys", "USERPROFILE": "C:\\Users\\dev"}
    assert copilot_home(env, windows=True) == Path("C:\\Users\\dev") / ".copilot"
    assert copilot_home(env, windows=False) == Path("/home/from-msys") / ".copilot"


def _legacy_pkg_root(home: Path, *, mentions_env: bool) -> None:
    pkg = home / ".local" / "share" / "copilot" / "pkg" / "1.0.60"
    pkg.mkdir(parents=True)
    (pkg / "app.js").write_text(
        "process.env.COPILOT_API_URL" if mentions_env else "nothing", encoding="utf-8"
    )


def test_stale_installer_bundle_cannot_vouch_for_a_launched_build(tmp_path: Path) -> None:
    """The bundle beside the binary about to run says no; an old installer copy
    under the legacy root says yes. The launched build decides."""
    home = tmp_path / "home"
    _legacy_pkg_root(home, mentions_env=True)
    shim_link, _native = _npm_layout(tmp_path, mentions_env=False)
    env = {"HOME": str(home), "LOCALAPPDATA": str(tmp_path / "nowhere")}
    assert native_api_url_supported(environ=env, copilot_bin=str(shim_link)) is False


def test_legacy_root_is_read_from_the_given_environment(tmp_path: Path) -> None:
    """The probe must not reach for the developer's real home directory."""
    home = tmp_path / "home"
    _legacy_pkg_root(home, mentions_env=True)
    env = {"HOME": str(home), "LOCALAPPDATA": str(tmp_path / "nowhere")}
    assert native_api_url_supported(environ=env, copilot_bin=None) is True
    assert native_api_url_supported(environ=_no_legacy_roots(tmp_path), copilot_bin=None) is None


def test_bundle_scan_finds_a_mention_that_straddles_a_chunk_boundary(tmp_path: Path) -> None:
    bundle = tmp_path / "app.js"
    filler = "x" * ((1 << 20) - 5)  # the needle starts 5 bytes before the first chunk ends
    bundle.write_text(filler + COPILOT_NATIVE_API_URL_ENV + ";", encoding="utf-8")
    assert copilot_wrap._bundle_mentions_native_api_url(str(bundle)) is True


def test_byok_lane_drops_a_durable_installs_native_hook() -> None:
    env, _ = build_launch_env(
        port=8787,
        provider_type="openai",
        wire_api=None,
        environ={COPILOT_NATIVE_API_URL_ENV: "http://127.0.0.1:9999"},
    )
    assert COPILOT_NATIVE_API_URL_ENV not in env


def test_subscription_lane_drops_a_durable_installs_native_hook_and_prints_the_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    captured: dict[str, object] = {}
    monkeypatch.setenv(COPILOT_NATIVE_API_URL_ENV, "http://127.0.0.1:9999")
    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: False)
    monkeypatch.setattr(
        wrap_mod,
        "_require_copilot_subscription_resolution",
        lambda: _native_resolution("https://api.enterprise.githubcopilot.com"),
    )
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **kwargs: captured.update(kwargs))

    result = CliRunner().invoke(
        main, ["wrap", "copilot", "--subscription", "--", "--model", "gpt-5.5"]
    )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert COPILOT_NATIVE_API_URL_ENV not in env
    assert "api.enterprise.githubcopilot.com" in result.output
