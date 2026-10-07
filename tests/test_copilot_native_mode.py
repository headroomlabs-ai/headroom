"""`wrap copilot --native`: route Copilot's own API through Headroom.

Why this mode exists
--------------------
BYOK is the CLI's provider override, and it *replaces* Copilot's model routing:
it accepts exactly one model, so the `/models` picker collapses to that model and
the MAIN agent cannot be switched mid-session. Native mode redirects the CLI's
own GitHub-authenticated API surface instead, leaving Copilot's routing intact —
the CLI still fetches `GET /models`, so the picker stays populated.

The tests below pin the two things that make the mode work and the one thing that
makes it safe: the BYOK path must be completely unaffected.
"""

from __future__ import annotations

import pytest

from headroom.providers.copilot.wrap import (
    COPILOT_BYOK_ENV_VARS,
    COPILOT_NATIVE_API_URL_ENV,
    build_launch_env,
    build_native_launch_env,
    native_api_url_supported,
)


def test_native_env_redirects_the_cli_api_surface() -> None:
    env, display = build_native_launch_env(port=8890, environ={}, project=None)
    assert env[COPILOT_NATIVE_API_URL_ENV] == "http://127.0.0.1:8890"
    assert any(COPILOT_NATIVE_API_URL_ENV in line for line in display)


def test_native_env_clears_every_byok_variable() -> None:
    """Any leftover BYOK var keeps the CLI single-model — the exact bug being fixed.

    That failure would be silent: the session would look fine and simply behave
    like BYOK, so it is asserted over the whole documented variable set rather
    than the few that happen to be set today.
    """
    seeded = dict.fromkeys(COPILOT_BYOK_ENV_VARS, "leftover")
    env, _ = build_native_launch_env(port=8890, environ=seeded, project=None)
    still_set = [v for v in COPILOT_BYOK_ENV_VARS if v in env]
    assert not still_set, f"BYOK vars survived into native mode: {still_set}"


def test_native_env_keeps_the_project_prefix() -> None:
    """Per-project savings ride the base URL: the CLI cannot send custom headers."""
    env, _ = build_native_launch_env(port=8890, environ={}, project="myproj")
    assert env[COPILOT_NATIVE_API_URL_ENV] == "http://127.0.0.1:8890/p/myproj"


def test_native_env_url_encodes_the_project_name() -> None:
    """A project directory name is not necessarily a valid URL path segment."""
    env, _ = build_native_launch_env(port=8890, environ={}, project="repo name")
    assert env[COPILOT_NATIVE_API_URL_ENV] == "http://127.0.0.1:8890/p/repo%20name"


def test_native_env_preserves_unrelated_environment() -> None:
    env, _ = build_native_launch_env(
        port=8890, environ={"PATH": "/usr/bin", "MY_VAR": "keep"}, project=None
    )
    assert env["PATH"] == "/usr/bin"
    assert env["MY_VAR"] == "keep"


def test_native_support_probe_is_tri_state(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown must be distinguishable from unsupported.

    If the variable ever stops working the failure is silent — the CLI talks
    straight to GitHub and Headroom simply sees nothing — so "no bundle found"
    (proceed with a note) and "bundle found, variable absent" (refuse) have to be
    different answers.
    """
    empty = {"LOCALAPPDATA": str(tmp_path / "none")}
    # `home` is passed explicitly so a dev box with the CLI at the XDG location
    # cannot make this machine-dependent.
    assert native_api_url_supported(environ=empty, home=str(tmp_path / "none")) is None

    pkg = tmp_path / "copilot" / "pkg" / "win32-x64" / "1.0.0"
    pkg.mkdir(parents=True)
    (pkg / "app.js").write_text("var x = 1;  // no override support\n", encoding="utf-8")
    assert (
        native_api_url_supported(
            environ={"LOCALAPPDATA": str(tmp_path)}, home=str(tmp_path / "none")
        )
        is False
    )

    (pkg / "app.js").write_text(
        "function qc(){return process.env.COPILOT_API_URL?1:2}\n", encoding="utf-8"
    )
    assert (
        native_api_url_supported(
            environ={"LOCALAPPDATA": str(tmp_path)}, home=str(tmp_path / "none")
        )
        is True
    )


def test_native_support_probe_is_unknown_when_a_bundle_cannot_be_fully_read(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle that opens but fails mid-read must not be reported as "unsupported".

    Only a bundle whose content was actually inspected end to end may answer
    False. One that opens and then fails to read (deleted or locked
    concurrently, a transient I/O error) was never actually inspected, so the
    honest answer is "unknown" -- reporting "unsupported" here would refuse a
    native launch that could in fact work.
    """
    pkg = tmp_path / "copilot" / "pkg" / "win32-x64" / "1.0.0"
    pkg.mkdir(parents=True)
    bundle = pkg / "app.js"
    bundle.write_text("var x = 1;\n", encoding="utf-8")

    real_open = open

    class _FlakyFile:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self, *_args, **_kwargs):
            raise OSError("synthetic mid-read failure")

    def _fake_open(path, *args, **kwargs):
        if str(path) == str(bundle):
            return _FlakyFile()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert (
        native_api_url_supported(
            environ={"LOCALAPPDATA": str(tmp_path)}, home=str(tmp_path / "none")
        )
        is None
    )


def test_native_support_probe_never_lets_an_older_bundle_vote_for_an_unreadable_newest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Newest bundle unreadable + an older one that still has the variable = unknown.

    Falling back to the older bundle is exactly the "old bundle vouches for a new
    one that may have dropped support" case the newest-only rule exists to stop.
    """
    root = tmp_path / "copilot" / "pkg" / "win32-x64"
    old = root / "1.0.70"
    new = root / "1.0.77"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "app.js").write_text(f"process.env.{COPILOT_NATIVE_API_URL_ENV};\n", encoding="utf-8")
    newest = new / "app.js"
    newest.write_text("var x = 1;\n", encoding="utf-8")

    real_open = open

    def _fake_open(path, *args, **kwargs):
        if str(path) == str(newest):
            raise OSError("locked by an updater")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _fake_open)
    assert (
        native_api_url_supported(
            environ={"LOCALAPPDATA": str(tmp_path)}, home=str(tmp_path / "none")
        )
        is None
    )


# ---------------------------------------------------------------------------
# The BYOK path must be untouched — it works today and is still the only option
# for genuine third-party keys.
# ---------------------------------------------------------------------------


def test_byok_env_builder_is_unchanged_by_native_mode() -> None:
    env, display = build_launch_env(
        port=8787, provider_type="openai", wire_api="responses", environ={}, project=None
    )
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "responses"
    # And it must NOT start setting the native override.
    assert COPILOT_NATIVE_API_URL_ENV not in env
    assert any("COPILOT_PROVIDER_BASE_URL" in line for line in display)


def test_byok_and_native_are_mutually_exclusive_shapes() -> None:
    """The two modes must never both be configured on one launch."""
    byok, _ = build_launch_env(
        port=8787, provider_type="openai", wire_api=None, environ={}, project=None
    )
    native, _ = build_native_launch_env(port=8787, environ={}, project=None)
    assert "COPILOT_PROVIDER_BASE_URL" in byok
    assert COPILOT_NATIVE_API_URL_ENV not in byok
    assert COPILOT_NATIVE_API_URL_ENV in native
    assert "COPILOT_PROVIDER_BASE_URL" not in native


# ---------------------------------------------------------------------------
# CLI wiring: flag validation and launch plumbing
# ---------------------------------------------------------------------------


def _invoke_copilot(monkeypatch: pytest.MonkeyPatch, args: list[str]):
    """Run `wrap copilot` with the launch intercepted, returning what it would do."""
    import shutil

    from click.testing import CliRunner

    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    captured: dict[str, object] = {}

    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: False)

    class _Res:
        token = "gho_test"
        api_url = "https://api.githubcopilot.com"
        refresh_oauth_token = None
        api_token_expires_at = None

    monkeypatch.setattr(wrap_mod, "_require_copilot_subscription_resolution", lambda: _Res())
    monkeypatch.setattr(wrap_mod, "resolve_client_bearer_token", lambda: "gho_test")
    monkeypatch.setattr(wrap_mod, "_native_api_url_supported", lambda **_k: True)
    monkeypatch.setattr(wrap_mod, "_live_copilot_model_ids", lambda *_a, **_k: [])
    monkeypatch.setattr(
        wrap_mod, "_resolve_copilot_wire_api_for_model", lambda *_a, **_k: "responses"
    )

    def _fake_launch(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(wrap_mod, "_launch_tool", _fake_launch)
    result = CliRunner().invoke(main, ["wrap", "copilot", *args])
    return result, captured


def _invoke_native(
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str] | None = None,
    *,
    support: bool | None = True,
):
    """Like `_invoke_copilot`, but resolves a tenant-pinned Business host.

    Business / data-residency accounts advertise their own API host through the
    token exchange (#610) rather than the generic ``api.githubcopilot.com``, and
    both the OpenAI and Anthropic upstreams must follow it in --native mode.
    """
    import shutil

    from click.testing import CliRunner

    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    captured: dict[str, object] = {}

    class _Res:
        token = "copilot-token"
        api_url = "https://api.business.githubcopilot.com"
        refresh_oauth_token = "refresh-token"
        api_token_expires_at = 123.0

    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: False)
    monkeypatch.setattr(wrap_mod, "_require_copilot_subscription_resolution", lambda: _Res())
    monkeypatch.setattr(wrap_mod, "_native_api_url_supported", lambda **_k: support)
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **kwargs: captured.update(kwargs))
    result = CliRunner().invoke(
        main, ["wrap", "copilot", "--native", "--port", "8890", *(extra or [])]
    )
    return result, captured


def test_native_flag_sets_the_override_and_no_byok_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    result, captured = _invoke_copilot(monkeypatch, ["--native", "--port", "8890"])
    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[COPILOT_NATIVE_API_URL_ENV].startswith("http://127.0.0.1:8890")
    assert "COPILOT_PROVIDER_BASE_URL" not in env


def test_native_points_the_anthropic_upstream_at_copilot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native CLI sends Claude traffic on /v1/messages.

    Without this the proxy forwards it to api.anthropic.com and the session dies
    with 401 — reproduced live before the fix.
    """
    _result, captured = _invoke_copilot(monkeypatch, ["--native", "--port", "8890"])
    assert captured["anthropic_api_url"] == "https://api.githubcopilot.com"
    assert captured["openai_api_url"] == "https://api.githubcopilot.com"


def test_native_cli_routes_both_protocols_to_tenant_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Business/data-residency accounts pin their own host; both wires must follow it.

    A hardcoded ``api.githubcopilot.com`` assumption would silently route a
    tenant's traffic to the wrong host — the same upstream-mismatch failure mode
    ``test_native_points_the_anthropic_upstream_at_copilot`` pins for the generic
    host.
    """
    result, captured = _invoke_native(monkeypatch)
    assert result.exit_code == 0, result.output
    assert captured["openai_api_url"] == "https://api.business.githubcopilot.com"
    assert captured["anthropic_api_url"] == "https://api.business.githubcopilot.com"
    env = captured["env"]
    assert isinstance(env, dict)
    assert COPILOT_NATIVE_API_URL_ENV in env
    assert not any(variable in env for variable in COPILOT_BYOK_ENV_VARS)


def test_byok_launch_leaves_the_anthropic_upstream_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard: BYOK must not start rewriting the Anthropic upstream."""
    _result, captured = _invoke_copilot(
        monkeypatch, ["--subscription", "--port", "8890", "--", "--model", "gpt-5.4"]
    )
    assert captured["anthropic_api_url"] is None
    env = captured["env"]
    assert isinstance(env, dict)
    assert "COPILOT_PROVIDER_BASE_URL" in env
    assert COPILOT_NATIVE_API_URL_ENV not in env


def test_native_implies_subscription_so_no_model_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--model` is mandatory under BYOK and optional in native mode.

    The "requires a model" note is gated on `not subscription`, so this really
    pins `native => subscription`; it fails if that coupling is removed. Also
    asserts the mode actually engaged, which the original did not.
    """
    result, captured = _invoke_copilot(monkeypatch, ["--native", "--port", "8890"])
    assert result.exit_code == 0, result.output
    assert "requires a model" not in result.output
    assert COPILOT_NATIVE_API_URL_ENV in captured["env"]


def test_implicit_oauth_uses_native_routing_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """GitHub-OAuth users with no provider key get native routing without --native (#1910).

    The old implicit-OAuth lane still configured Copilot as a one-model BYOK
    client. Native aliases and runtime `/model` switches were then forwarded
    literally and rejected by GitHub. Explicit `--subscription` keeps its
    existing fixed-wire behavior; only the *implicit* GitHub-OAuth lane is
    upgraded to native routing automatically.
    """
    from click.testing import CliRunner

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


def test_implicit_native_rejects_no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The --no-proxy guard must fire for implicit native routing too, not just --native.

    Checking only the explicit flag let an implicitly-native session (OAuth, no
    provider key, no --subscription) through with --no-proxy: the already-running
    proxy can't be repointed at the Copilot host, so the upstream overrides this
    mode needs are silently discarded and every request 401s against the wrong
    vendor -- the exact failure `--native --no-proxy` is refused for explicitly.
    """
    from click.testing import CliRunner

    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: False)
    monkeypatch.setattr(wrap_mod, "has_oauth_auth", lambda: True)
    monkeypatch.setattr(wrap_mod, "resolve_client_bearer_token", lambda: "oauth-token")
    monkeypatch.setattr(
        wrap_mod, "resolve_copilot_api_url", lambda _token: "https://api.githubcopilot.com"
    )
    monkeypatch.setattr(wrap_mod, "_native_api_url_supported", lambda **_kwargs: True)
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **_k: None)

    result = CliRunner().invoke(
        main,
        ["wrap", "copilot", "--no-proxy", "--port", "8890", "--", "--model", "claude-sonnet-5"],
    )

    assert result.exit_code != 0
    assert "--no-proxy" in result.output


def test_native_rejects_wire_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """--wire-api pins one wire for the session, which native mode exists to avoid."""
    result, _captured = _invoke_copilot(
        monkeypatch, ["--native", "--wire-api", "responses", "--port", "8890"]
    )
    assert result.exit_code != 0
    assert "--wire-api" in result.output


def test_native_rejects_anthropic_provider_type(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _captured = _invoke_copilot(
        monkeypatch, ["--native", "--provider-type", "anthropic", "--port", "8890"]
    )
    assert result.exit_code != 0
    assert "provider-type anthropic" in result.output


def test_native_warns_when_cli_lacks_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warn loudly, but do not refuse.

    If a future CLI drops ``COPILOT_API_URL`` the failure is invisible — the CLI
    talks straight to GitHub and Headroom never sees the traffic — so silence is
    not acceptable. But the probe reads an undocumented needle out of a shipped
    bundle and *has* produced false negatives (a needle split across a read
    boundary), and refusing to launch a CLI that actually works is the worse
    outcome. So: warn, name how to confirm, and proceed.
    """
    import shutil

    from click.testing import CliRunner

    from headroom.cli import wrap as wrap_mod
    from headroom.cli.main import main

    class _Res:
        token = "gho_test"
        api_url = "https://api.githubcopilot.com"
        refresh_oauth_token = None
        api_token_expires_at = None

    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/copilot")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: False)
    monkeypatch.setattr(wrap_mod, "_require_copilot_subscription_resolution", lambda: _Res())
    monkeypatch.setattr(wrap_mod, "_launch_tool", lambda **_k: None)
    # The decisive stub: report the installed CLI as NOT supporting the override.
    monkeypatch.setattr(wrap_mod, "_native_api_url_supported", lambda **_k: False)

    result = CliRunner().invoke(main, ["wrap", "copilot", "--native", "--port", "8890"])
    assert result.exit_code == 0, result.output
    assert "COPILOT_API_URL" in result.output
    assert "silently get no compression" in result.output
    assert "without --native" in result.output


def test_native_cli_reports_unknown_support_in_verbose_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The "could not verify" note only surfaces with --verbose, and never blocks launch."""
    result, captured = _invoke_native(monkeypatch, ["--verbose"], support=None)
    assert result.exit_code == 0, result.output
    assert "could not verify" in result.output
    assert captured


def test_native_skips_the_model_list_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native mode has a real picker, so the instruction-file workaround is not needed.

    Spies on the injector rather than checking for the file: the earlier version
    stubbed the model list to `[]`, and the injector early-returns on empty, so the
    file was absent on the BYOK path too and the test could not fail.
    """
    from headroom.cli import wrap as wrap_mod

    monkeypatch.setenv("HEADROOM_MODEL_CATALOG", "true")
    calls: list[tuple] = []
    monkeypatch.setattr(
        wrap_mod,
        "_inject_copilot_models_instructions",
        lambda *a, **k: calls.append(a) or True,
    )
    monkeypatch.setattr(wrap_mod, "_live_copilot_model_ids", lambda *_a, **_k: ["gpt-5.4"])

    _r, _c = _invoke_copilot(monkeypatch, ["--native", "--port", "8890"])
    assert calls == [], "native mode should not write the model list"

    _r2, _c2 = _invoke_copilot(
        monkeypatch, ["--subscription", "--port", "8890", "--", "--model", "gpt-5.4"]
    )
    assert len(calls) == 1, "BYOK mode must still write it (proves the spy works)"


# ---------------------------------------------------------------------------
# Cross-contamination: a native proxy must never be reused by `wrap claude`
#
# These follow the shared-proxy ownership contract in headroom/cli/wrap.py: an
# idle mismatched proxy is restarted on the same port, an attached one is left
# alone and the session gets a dedicated proxy elsewhere.
# ---------------------------------------------------------------------------


@pytest.fixture
def _hermetic_routing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own upstream overrides would change the requested routing."""
    for name in (
        "ANTHROPIC_TARGET_API_URL",
        "OPENAI_TARGET_API_URL",
        "GEMINI_TARGET_API_URL",
        "CLOUDCODE_TARGET_API_URL",
        "VERTEX_TARGET_API_URL",
        "AUGMENT_TARGET_API_URL",
        "FACTORY_TARGET_API_URL",
        "HEADROOM_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)


def _drive_ensure_proxy(monkeypatch: pytest.MonkeyPatch, running_config: dict, **kwargs):
    """Drive the real `_ensure_proxy` against a running proxy, recording its actions.

    Exercises the end-to-end decision rather than a helper predicate: an earlier
    version of this guard tested a predicate that was correct while the product
    ignored it on every path, so the test was green and the bug was live.
    """
    from headroom.cli import wrap as wrap_mod

    calls: list[tuple] = []
    health = {
        "version": wrap_mod._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {
            "pid": "4242",
            "memory": False,
            "learn": False,
            "code_graph": False,
            **running_config,
        },
    }
    monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: len(calls) == 0)
    monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(wrap_mod, "_port_bind_error", lambda port: None)
    # Idle unless a test says otherwise: a real wrap session on the developer's
    # machine must not flip these into the "attached" branch.
    monkeypatch.setattr(wrap_mod, "_live_proxy_clients", lambda *a, **k: [])
    monkeypatch.setattr(
        wrap_mod, "_kill_proxy_by_pid", lambda pid, port: calls.append(("kill", pid, port)) or True
    )
    monkeypatch.setattr(wrap_mod, "_start_proxy", lambda *a, **k: calls.append(("start", k)))
    _proc, port = wrap_mod._ensure_proxy(8787, False, **kwargs)
    return port, calls


@pytest.mark.usefixtures("_hermetic_routing_env")
def test_native_proxy_is_not_shared_with_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    """The credential-substitution hazard: `wrap claude` must not inherit it.

    `--native` points BOTH upstreams at the Copilot host. Sharing that proxy would
    send Claude Code's /v1/messages to api.githubcopilot.com — and since `sk-ant-*`
    bearers are not forwardable, `apply_copilot_api_auth` strips the user's
    Anthropic credential and substitutes the Copilot token. Silently, because the
    proxy reports healthy.
    """
    port, calls = _drive_ensure_proxy(
        monkeypatch,
        {
            "anthropic_api_url": "https://api.githubcopilot.com",
            "openai_api_url": "https://api.githubcopilot.com",
        },
        # `wrap claude` wants the DEFAULT upstreams — the case a
        # "only compare when the caller pinned one" check never noticed.
    )
    assert any(c[0] == "kill" for c in calls), "the Copilot-pinned proxy was reused"
    started = [c for c in calls if c[0] == "start"]
    assert started, "no replacement proxy was started"
    assert started[0][1].get("anthropic_api_url") is None
    assert started[0][1].get("openai_api_url") is None
    assert port == 8787, "an idle proxy is restarted on the same port, not abandoned"


@pytest.mark.usefixtures("_hermetic_routing_env")
def test_plain_proxy_is_still_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard against over-correcting into restarting on every launch."""
    port, calls = _drive_ensure_proxy(
        monkeypatch, {"anthropic_api_url": None, "openai_api_url": None}
    )
    assert not calls, f"a matching proxy was needlessly restarted: {calls}"
    assert port == 8787


@pytest.mark.usefixtures("_hermetic_routing_env")
def test_matching_native_proxy_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two `--native` sessions on one port share the proxy rather than fighting."""
    copilot = "https://api.githubcopilot.com"
    port, calls = _drive_ensure_proxy(
        monkeypatch,
        {"anthropic_api_url": copilot, "openai_api_url": copilot},
        anthropic_api_url=copilot,
        openai_api_url=copilot,
    )
    assert not calls, f"a matching native proxy was needlessly restarted: {calls}"
    assert port == 8787


@pytest.mark.usefixtures("_hermetic_routing_env")
def test_pinned_proxy_is_not_shared_even_when_another_wrapper_is_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An upstream conflict must not fall into the "reuse as-is" branch.

    That branch exists so a *missing feature* does not disrupt other sessions'
    in-flight requests -- correct for a feature, wrong for an upstream: sharing a
    Copilot-pinned proxy sends this session's traffic to another vendor with the
    credential substituted. Under the ownership contract the attached proxy is
    left alone and this session gets a dedicated proxy on another port.
    """
    from headroom.cli import wrap as wrap_mod

    calls: list[str] = []
    health = {
        "version": wrap_mod._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {
            "pid": "4242",
            "memory": False,
            "learn": False,
            "code_graph": False,
            "anthropic_api_url": "https://api.githubcopilot.com",
            "openai_api_url": "https://api.githubcopilot.com",
        },
    }
    monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(wrap_mod, "_port_bind_error", lambda port: None)
    monkeypatch.setattr(wrap_mod, "_live_proxy_clients", lambda port, exclude_self: ["copilot"])
    monkeypatch.setattr(wrap_mod, "_find_available_port", lambda start: 8788)
    monkeypatch.setattr(
        wrap_mod, "_kill_proxy_by_pid", lambda pid, port: calls.append("kill") or True
    )
    monkeypatch.setattr(wrap_mod, "_start_proxy", lambda *a, **k: calls.append("start"))

    _proc, port = wrap_mod._ensure_proxy(8787, False, agent_type="claude")

    assert port != 8787, "shared a proxy pinned to another vendor's upstream"
    assert "kill" not in calls, "disrupted the attached session instead of isolating"
