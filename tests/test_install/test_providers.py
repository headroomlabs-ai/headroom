from __future__ import annotations

import json
import os
from pathlib import Path

import click
import pytest

from headroom.install.models import DeploymentManifest, ManagedMutation
from headroom.install.providers import _apply_windows_env_scope, _remove_windows_env_scope
from headroom.providers.claude.install import apply_provider_scope as apply_claude_provider_scope
from headroom.providers.claude.install import build_install_env as build_claude_install_env
from headroom.providers.claude.install import revert_provider_scope as revert_claude_provider_scope
from headroom.providers.codex.install import apply_provider_scope as apply_codex_provider_scope
from headroom.providers.codex.install import build_install_env as build_codex_install_env
from headroom.providers.codex.install import revert_provider_scope as revert_codex_provider_scope
from headroom.providers.copilot.install import build_install_env as build_copilot_install_env


def _manifest(tmp_path: Path) -> DeploymentManifest:
    return DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="provider",
        provider_mode="manual",
        targets=["claude", "codex"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        memory_db_path=str(tmp_path / "memory.db"),
        tool_envs={
            "claude": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"},
            "codex": {"OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"},
        },
    )


def test_apply_and_revert_claude_provider_scope(monkeypatch, tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"env": {"ANTHROPIC_API_KEY": "keep", "ANTHROPIC_BASE_URL": "https://old"}})
    )
    monkeypatch.setattr(
        "headroom.providers.claude.install.claude_settings_path", lambda: settings_path
    )
    manifest = _manifest(tmp_path)

    mutation = apply_claude_provider_scope(manifest)
    payload = json.loads(settings_path.read_text())
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert payload["env"]["ANTHROPIC_API_KEY"] == "keep"

    assert mutation is not None
    revert_claude_provider_scope(mutation, manifest)
    reverted = json.loads(settings_path.read_text())
    assert reverted["env"]["ANTHROPIC_BASE_URL"] == "https://old"
    assert reverted["env"]["ANTHROPIC_API_KEY"] == "keep"


def test_apply_and_revert_codex_provider_scope(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-4o"\n')
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)

    mutation = apply_codex_provider_scope(manifest)
    content = config_path.read_text()
    assert 'model_provider = "headroom"' in content
    assert 'base_url = "http://127.0.0.1:8787/v1"' in content
    assert 'env_key = "OPENAI_API_KEY"' not in content
    assert "requires_openai_auth" not in content

    assert mutation is not None
    revert_codex_provider_scope(mutation, manifest)
    reverted = config_path.read_text()
    assert 'model_provider = "headroom"' not in reverted
    assert reverted.strip() == 'model = "gpt-4o"'


def test_apply_codex_provider_scope_emits_flag_for_chatgpt_auth(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    (tmp_path / "auth.json").write_text('{"auth_mode": "chatgpt"}')
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)

    apply_codex_provider_scope(manifest)

    assert "requires_openai_auth = true" in config_path.read_text()


def test_codex_build_install_env_returns_proxy_base_url() -> None:
    env = build_codex_install_env(port=5566, backend="ignored")

    assert env == {"OPENAI_BASE_URL": "http://127.0.0.1:5566/v1"}


def test_apply_codex_provider_scope_skips_non_provider_scope(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)
    manifest.scope = "user"

    mutation = apply_codex_provider_scope(manifest)

    assert mutation is None
    assert not config_path.exists()


def test_apply_codex_provider_scope_replaces_existing_managed_block(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'model = "gpt-4o"\n\n'
        "# --- Headroom persistent provider ---\n"
        'model_provider = "headroom"\n\n'
        "[model_providers.headroom]\n"
        'name = "Headroom persistent proxy"\n'
        'base_url = "http://127.0.0.1:1111/v1"\n'
        "requires_openai_auth = true\n"
        "supports_websockets = true\n"
        "# --- end Headroom persistent provider ---\n"
    )
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)
    manifest.port = 9999

    apply_codex_provider_scope(manifest)

    content = config_path.read_text()
    assert content.count("# --- Headroom persistent provider ---") == 1
    assert 'base_url = "http://127.0.0.1:9999/v1"' in content
    assert 'base_url = "http://127.0.0.1:1111/v1"' not in content
    # Bug 3 (#406): the replacement block must NOT carry requires_openai_auth.
    assert "requires_openai_auth" not in content


def test_apply_codex_provider_scope_creates_new_config_when_missing(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "nested" / "config.toml"
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)

    mutation = apply_codex_provider_scope(manifest)

    assert mutation is not None
    assert 'base_url = "http://127.0.0.1:8787/v1"' in config_path.read_text()


def test_revert_codex_provider_scope_ignores_missing_path_and_file(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    revert_codex_provider_scope(
        ManagedMutation(target="codex", kind="toml-block"),
        manifest,
    )
    revert_codex_provider_scope(
        ManagedMutation(
            target="codex",
            kind="toml-block",
            path=str(tmp_path / "missing.toml"),
        ),
        manifest,
    )


def test_revert_codex_provider_scope_ignores_files_without_managed_block(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-4o"\n')
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)
    mutation = ManagedMutation(target="codex", kind="toml-block", path=str(config_path))

    revert_codex_provider_scope(mutation, manifest)

    assert config_path.read_text() == 'model = "gpt-4o"\n'


def test_apply_openclaw_provider_scope_uses_manifest_port(monkeypatch, tmp_path: Path) -> None:
    recorded: list[list[str]] = []
    monkeypatch.setattr("headroom.providers.openclaw.install.shutil_which", lambda name: "openclaw")
    monkeypatch.setattr(
        "headroom.providers.openclaw.install.resolve_headroom_command",
        lambda: ["headroom"],
    )
    monkeypatch.setattr(
        "headroom.providers.openclaw.install._invoke_openclaw",
        lambda command: recorded.append(command),
    )
    monkeypatch.setattr(
        "headroom.providers.openclaw.install.openclaw_config_path",
        lambda: tmp_path / "openclaw.json",
    )
    manifest = _manifest(tmp_path)
    manifest.port = 9999

    from headroom.providers.openclaw.install import (
        apply_provider_scope as apply_openclaw_provider_scope,
    )

    apply_openclaw_provider_scope(manifest)

    assert recorded == [["headroom", "wrap", "openclaw", "--no-auto-start", "--proxy-port", "9999"]]


def test_openclaw_apply_provider_scope_requires_installed_binary(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("headroom.providers.openclaw.install.shutil_which", lambda name: None)

    with pytest.raises(click.ClickException, match="openclaw not found"):
        from headroom.providers.openclaw.install import (
            apply_provider_scope as apply_openclaw_provider_scope,
        )

        apply_openclaw_provider_scope(_manifest(tmp_path))


def test_openclaw_helper_wrappers_delegate_to_stdlib(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    recorded: list[tuple[list[str], bool]] = []

    def fake_run(command: list[str], check: bool) -> None:
        recorded.append((command, check))

    monkeypatch.setattr("subprocess.run", fake_run)

    from headroom.providers.openclaw.install import _invoke_openclaw, shutil_which

    assert shutil_which("openclaw") == "/fake/openclaw"
    _invoke_openclaw(["headroom", "wrap", "openclaw"])

    assert recorded == [(["headroom", "wrap", "openclaw"], True)]


def test_openclaw_revert_provider_scope_skips_without_binary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("headroom.providers.openclaw.install.shutil_which", lambda name: None)
    called = False

    def fail_if_called(command: list[str]) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("headroom.providers.openclaw.install._invoke_openclaw", fail_if_called)

    from headroom.providers.openclaw.install import (
        revert_provider_scope as revert_openclaw_provider_scope,
    )

    revert_openclaw_provider_scope(
        ManagedMutation(target="openclaw", kind="openclaw-wrap", path=str(tmp_path / "cfg.json")),
        _manifest(tmp_path),
    )

    assert called is False


def test_openclaw_revert_provider_scope_invokes_unwrap(monkeypatch, tmp_path: Path) -> None:
    recorded: list[list[str]] = []
    monkeypatch.setattr("headroom.providers.openclaw.install.shutil_which", lambda name: "openclaw")
    monkeypatch.setattr(
        "headroom.providers.openclaw.install.resolve_headroom_command",
        lambda: ["headroom"],
    )
    monkeypatch.setattr(
        "headroom.providers.openclaw.install._invoke_openclaw",
        lambda command: recorded.append(command),
    )

    from headroom.providers.openclaw.install import (
        revert_provider_scope as revert_openclaw_provider_scope,
    )

    revert_openclaw_provider_scope(
        ManagedMutation(target="openclaw", kind="openclaw-wrap", path=str(tmp_path / "cfg.json")),
        _manifest(tmp_path),
    )

    assert recorded == [["headroom", "unwrap", "openclaw"]]


def test_windows_env_scope_restores_previous_values(monkeypatch, tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.scope = "user"
    manifest.targets = ["claude"]
    manifest.base_env = {"HEADROOM_PORT": "8787"}
    manifest.tool_envs = {"claude": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}}

    calls: list[list[str]] = []
    previous_values = {
        "HEADROOM_PORT": "7777",
        "ANTHROPIC_BASE_URL": "https://old",
    }

    class Result:
        def __init__(self, stdout: str = "") -> None:
            self.stdout = stdout

    def fake_run(command: list[str], **kwargs):
        calls.append(command)
        script = command[-1]
        if "GetEnvironmentVariable" in script:
            name = script.split("GetEnvironmentVariable('", 1)[1].split("'", 1)[0]
            value = previous_values.get(name, "__HEADROOM_UNSET__")
            return Result(stdout=value)
        return Result()

    monkeypatch.setattr("headroom.install.providers.subprocess.run", fake_run)

    mutations = _apply_windows_env_scope(manifest)
    _remove_windows_env_scope(mutations)

    previous_by_name = {mutation.data["name"]: mutation.data["previous"] for mutation in mutations}
    assert previous_by_name["HEADROOM_PORT"] == "7777"
    assert previous_by_name["ANTHROPIC_BASE_URL"] == "https://old"
    assert any(
        "[Environment]::SetEnvironmentVariable('HEADROOM_PORT','7777','User')" in command[-1]
        for command in calls
    )
    assert any(
        "[Environment]::SetEnvironmentVariable('ANTHROPIC_BASE_URL','https://old','User')"
        in command[-1]
        for command in calls
    )


def test_remove_windows_env_scope_requires_name_and_scope() -> None:
    try:
        _remove_windows_env_scope([ManagedMutation(target="env", kind="windows-env", data={})])
    except ValueError as exc:
        assert "variable name" in str(exc)
    else:
        raise AssertionError("expected missing variable name to raise")

    try:
        _remove_windows_env_scope(
            [ManagedMutation(target="env", kind="windows-env", data={"name": "X", "scope": 1})]
        )
    except ValueError as exc:
        assert "valid scope" in str(exc)
    else:
        raise AssertionError("expected invalid scope to raise")


def test_apply_mutations_runs_openclaw_for_user_scope(monkeypatch, tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.scope = "user"
    manifest.targets = ["openclaw"]
    manifest.base_env = {"HEADROOM_PORT": "8787"}
    manifest.tool_envs = {}

    if os.name == "nt":
        monkeypatch.setattr(
            "headroom.install.providers._apply_windows_env_scope", lambda deployment: []
        )
    else:
        monkeypatch.setattr(
            "headroom.install.providers._apply_unix_env_scope", lambda deployment: []
        )
    monkeypatch.setattr(
        "headroom.install.providers.apply_provider_scope_mutations",
        lambda deployment: [ManagedMutation(target="openclaw", kind="openclaw-wrap")],
    )

    from headroom.install.providers import apply_mutations

    mutations = apply_mutations(manifest)

    assert [mutation.kind for mutation in mutations] == ["openclaw-wrap"]


def test_claude_build_install_env_returns_proxy_base_url() -> None:
    # Arrange / Act
    env = build_claude_install_env(port=5566, backend="ignored")

    # Assert
    assert env == {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:5566",
        "ENABLE_TOOL_SEARCH": "true",
    }


def test_copilot_build_install_env_uses_provider_type_specific_proxy_urls() -> None:
    anthropic_env = build_copilot_install_env(port=8787, backend="anthropic")
    openai_env = build_copilot_install_env(port=8787, backend="anyllm")

    assert anthropic_env == {
        "COPILOT_PROVIDER_TYPE": "anthropic",
        "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:8787",
    }
    assert openai_env == {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:8787/v1",
        "COPILOT_PROVIDER_WIRE_API": "completions",
    }


def test_apply_claude_provider_scope_skips_non_provider_scope(monkeypatch, tmp_path: Path) -> None:
    # Arrange
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(
        "headroom.providers.claude.install.claude_settings_path", lambda: settings_path
    )
    manifest = _manifest(tmp_path)
    manifest.scope = "user"

    # Act
    mutation = apply_claude_provider_scope(manifest)

    # Assert
    assert mutation is None
    assert not settings_path.exists()


def test_revert_claude_provider_scope_removes_new_values_from_non_mapping_env(
    monkeypatch, tmp_path: Path
) -> None:
    # Arrange
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"env": ["not-a-map"]}))
    monkeypatch.setattr(
        "headroom.providers.claude.install.claude_settings_path", lambda: settings_path
    )
    manifest = _manifest(tmp_path)

    # Act
    mutation = apply_claude_provider_scope(manifest)
    apply_payload = json.loads(settings_path.read_text())
    revert_claude_provider_scope(mutation, manifest)
    reverted_payload = json.loads(settings_path.read_text())

    # Assert
    assert mutation is not None
    assert mutation.data["previous"] == {"ANTHROPIC_BASE_URL": None}
    assert apply_payload["env"] == {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}
    assert reverted_payload["env"] == {}


def test_apply_claude_provider_scope_creates_settings_when_missing(
    monkeypatch, tmp_path: Path
) -> None:
    # Arrange
    settings_path = tmp_path / "nested" / "settings.json"
    monkeypatch.setattr(
        "headroom.providers.claude.install.claude_settings_path", lambda: settings_path
    )
    manifest = _manifest(tmp_path)

    # Act
    mutation = apply_claude_provider_scope(manifest)

    # Assert
    assert mutation is not None
    assert json.loads(settings_path.read_text()) == {
        "env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787"}
    }


def test_revert_claude_provider_scope_ignores_missing_mutation_path(tmp_path: Path) -> None:
    # Arrange
    manifest = _manifest(tmp_path)
    mutation = ManagedMutation(target="claude", kind="json-env", data={"previous": {}})

    # Act / Assert
    revert_claude_provider_scope(mutation, manifest)


def test_revert_claude_provider_scope_ignores_missing_settings_file(tmp_path: Path) -> None:
    # Arrange
    manifest = _manifest(tmp_path)
    mutation = ManagedMutation(
        target="claude",
        kind="json-env",
        path=str(tmp_path / "missing-settings.json"),
        data={"previous": {}},
    )

    # Act / Assert
    revert_claude_provider_scope(mutation, manifest)


# ---------------------------------------------------------------------------
# Bug 3 regression tests (#406): requires_openai_auth and openai_base_url
# must never appear in the headroom provider block.
# ---------------------------------------------------------------------------


def test_headroom_provider_block_never_sets_requires_openai_auth(
    monkeypatch, tmp_path: Path
) -> None:
    """apply_provider_scope must NOT emit requires_openai_auth in the headroom block.

    Bug 3 (#406): requires_openai_auth = true on a custom [model_providers.headroom]
    block forces codex to demand OpenAI OAuth login for headroom-routed traffic.
    Headroom is a local proxy — it must not require OpenAI auth.
    """
    for port in (8787, 9999):
        config_path = tmp_path / f"config_{port}.toml"
        monkeypatch.setattr(
            "headroom.providers.codex.install.codex_config_path",
            lambda _p=config_path: _p,
        )
        manifest = _manifest(tmp_path)
        manifest.port = port

        apply_codex_provider_scope(manifest)

        content = config_path.read_text()
        # The rendered TOML for the headroom provider block must not contain this
        # field.  If it does, codex will prompt for OpenAI OAuth on every startup.
        assert "requires_openai_auth" not in content, (
            f"requires_openai_auth must be absent from the headroom provider block "
            f"(port={port}); got:\n{content}"
        )
        # Sanity: the block itself is present and points at the right port.
        assert f'base_url = "http://127.0.0.1:{port}/v1"' in content
        assert "[model_providers.headroom]" in content


def test_inject_codex_provider_config_writes_openai_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_inject_codex_provider_config MUST write a top-level openai_base_url key.

    Bug 3 (#406) has two halves:
    1. Strip ``requires_openai_auth`` from the headroom provider block (done in 3ca48d3).
    2. Inject ``openai_base_url`` at the top level so subscription (ChatGPT plan)
       users are also routed through headroom.

    Without the top-level ``openai_base_url`` override, Codex subscription mode
    uses the built-in ``openai`` provider with ``chatgpt.com/backend-api/codex``
    as the base URL, bypassing the proxy entirely.  This key is the only way to
    intercept subscription traffic.
    """
    from headroom.cli import wrap as wrap_mod

    home = tmp_path
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    wrap_mod._inject_codex_provider_config(8787)

    config_file = home / ".codex" / "config.toml"
    assert config_file.exists(), "inject must create ~/.codex/config.toml"
    content = config_file.read_text()

    # The top-level openai_base_url key must be present at exactly this value.
    # Any port drift (e.g. 9999) will break subscription routing and must cause
    # this test to fail.
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in content, (
        "openai_base_url must appear at the top level of config.toml after injection "
        "so that Codex subscription (ChatGPT plan) users are routed through headroom; "
        f"got:\n{content}"
    )
    # Sanity: the provider block is actually there.
    assert "[model_providers.headroom]" in content
    assert 'base_url = "http://127.0.0.1:8787/v1"' in content
    # requires_openai_auth must also be absent (bug 3 regression guard).
    assert "requires_openai_auth" not in content, (
        f"requires_openai_auth must be absent from the injected provider block; got:\n{content}"
    )
    # openai_base_url must be in the top-level block, NOT inside [model_providers.*]
    lines = content.splitlines()
    in_provider_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_provider_section = stripped.startswith("[model_providers")
        if in_provider_section and stripped.startswith("openai_base_url"):
            raise AssertionError(
                f"openai_base_url must be top-level, not inside a [model_providers.*] section; "
                f"found it inside a provider section. Config:\n{content}"
            )


def test_unwrap_removes_top_level_openai_base_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After unwrap, the top-level openai_base_url key must be gone.

    Verifies that ``_restore_codex_provider_config`` (unwrap) removes the
    ``openai_base_url`` key so orphaned entries don't accumulate between
    wrap/unwrap cycles.
    """
    from headroom.cli import wrap as wrap_mod

    home = tmp_path
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    wrap_mod._inject_codex_provider_config(8787)
    config_file = home / ".codex" / "config.toml"
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in config_file.read_text()

    wrap_mod._restore_codex_provider_config()

    # After unwrap the file should be gone (no prior content) or cleaned.
    if config_file.exists():
        content = config_file.read_text()
        assert "openai_base_url" not in content, (
            f"openai_base_url must not remain in config.toml after unwrap; got:\n{content}"
        )
    # Also verify via _strip_codex_headroom_blocks directly — the orphan-cleanup
    # path is exercised when there is no backup file (crash-recovery path).
    orphan_content = (
        'model = "gpt-4o"\n'
        'openai_base_url = "http://127.0.0.1:8787/v1"\n'
        'model_provider = "headroom"\n'
    )
    stripped = wrap_mod._strip_codex_headroom_blocks(orphan_content)
    assert "openai_base_url" not in stripped, (
        f"_strip_codex_headroom_blocks must remove orphaned openai_base_url lines; got:\n{stripped}"
    )
    assert 'model = "gpt-4o"' in stripped


# ---------------------------------------------------------------------------
# Bug 3 (#406): openai_base_url must appear in persistent-install and init paths
# ---------------------------------------------------------------------------


def test_apply_provider_scope_writes_openai_base_url(monkeypatch, tmp_path: Path) -> None:
    """apply_provider_scope must write openai_base_url at the top level (not inside
    a [model_providers.*] block) so subscription (ChatGPT plan) users are routed
    through headroom regardless of which entry point they used."""
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-4o"\n')
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)
    manifest.port = 8787

    apply_codex_provider_scope(manifest)

    content = config_path.read_text()
    # Must be present as a top-level key.
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in content, (
        f"openai_base_url missing from persistent-install config:\n{content}"
    )
    # Must NOT be inside any [model_providers.*] block — the key must appear
    # before the first [section] header that follows it.
    lines = content.splitlines()
    in_provider_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_provider_section = True
        if in_provider_section and stripped.startswith("openai_base_url"):
            raise AssertionError(f"openai_base_url appeared inside a section block:\n{content}")
    # Bug 3 regression: requires_openai_auth must NOT appear.
    assert "requires_openai_auth" not in content, (
        f"requires_openai_auth must not be present in config:\n{content}"
    )


def test_persistent_install_strip_removes_openai_base_url(monkeypatch, tmp_path: Path) -> None:
    """revert_provider_scope must remove openai_base_url during uninstall, and
    must also clean up orphaned openai_base_url lines left by a crashed install."""
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-4o"\n')
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)
    manifest = _manifest(tmp_path)
    manifest.port = 8787

    mutation = apply_codex_provider_scope(manifest)
    assert mutation is not None
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in config_path.read_text()

    revert_codex_provider_scope(mutation, manifest)

    reverted = config_path.read_text()
    assert "openai_base_url" not in reverted, (
        f"openai_base_url must be removed after revert:\n{reverted}"
    )
    assert "requires_openai_auth" not in reverted
    # User's original content must survive.
    assert 'model = "gpt-4o"' in reverted

    # Also verify orphan-cleanup path: revert on a config where openai_base_url
    # was left outside the marker block (crash-recovery scenario).
    config_path.write_text(
        'model = "gpt-4o"\n'
        'openai_base_url = "http://127.0.0.1:8787/v1"\n'
        'model_provider = "headroom"\n'
    )
    revert_codex_provider_scope(
        ManagedMutation(target="codex", kind="toml-block", path=str(config_path)),
        manifest,
    )
    orphan_reverted = config_path.read_text()
    assert "openai_base_url" not in orphan_reverted, (
        f"orphaned openai_base_url must be removed by revert:\n{orphan_reverted}"
    )
    assert 'model = "gpt-4o"' in orphan_reverted
