"""Tests for `headroom wrap omp` / `headroom unwrap omp`.

Covers the omp runtime override contract (fresh-create vs merge-preserving
injection, pristine backups, re-injection idempotency, restore statuses) and
the CLI wiring that drives it. Every test isolates omp's agent directory via
``PI_CODING_AGENT_DIR`` and runs from a tmp cwd so the real ``~/.omp`` is
never touched.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.providers.omp import (
    MANAGED_MARKER,
    backup_path,
    build_launch_env,
    inject_models_override,
    models_yml_path,
    restore_models_override,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def omp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate omp's agent dir under tmp_path and run from a tmp cwd.

    Returns the ``models.yml`` path the runtime resolves to.
    """
    agent_dir = tmp_path / "omp-agent"
    agent_dir.mkdir()
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.chdir(tmp_path)
    return agent_dir / "models.yml"


# ---------------------------------------------------------------------------
# runtime: path resolution
# ---------------------------------------------------------------------------


def test_models_yml_path_honors_pi_coding_agent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "relocated-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(custom))
    monkeypatch.chdir(tmp_path)
    assert models_yml_path() == custom / "models.yml"


# ---------------------------------------------------------------------------
# runtime: injection
# ---------------------------------------------------------------------------


def test_inject_fresh_create_writes_managed_marker_and_no_backup(omp_home: Path) -> None:
    models_file, base_url = inject_models_override(8787, "proj")

    assert models_file == omp_home
    assert base_url == "http://127.0.0.1:8787/p/proj"

    text = models_file.read_text(encoding="utf-8")
    assert MANAGED_MARKER in text
    assert yaml.safe_load(text)["providers"]["anthropic"]["baseUrl"] == base_url

    # Nothing pre-existed, so there is nothing to snapshot.
    assert not backup_path(models_file).exists()


def test_inject_over_existing_backs_up_pristine_and_merges(omp_home: Path) -> None:
    original = (
        "providers:\n"
        "  anthropic:\n"
        "    apiKey: sk-user-secret\n"
        "  openai:\n"
        "    baseUrl: https://api.openai.com/v1\n"
        "models:\n"
        "  - id: my-custom-model\n"
    )
    omp_home.write_bytes(original.encode("utf-8"))

    _, base_url = inject_models_override(8787, "proj")

    # Pre-wrap file is snapshotted byte-for-byte.
    assert backup_path(omp_home).read_bytes() == original.encode("utf-8")

    merged = yaml.safe_load(omp_home.read_text(encoding="utf-8"))
    assert merged["providers"]["anthropic"]["baseUrl"] == base_url
    # Only anthropic.baseUrl is set; every other user key survives the merge.
    assert merged["providers"]["anthropic"]["apiKey"] == "sk-user-secret"
    assert merged["providers"]["openai"]["baseUrl"] == "https://api.openai.com/v1"
    assert merged["models"] == [{"id": "my-custom-model"}]
    assert MANAGED_MARKER in omp_home.read_text(encoding="utf-8")


def test_reinject_new_port_regenerates_from_pristine_backup(omp_home: Path) -> None:
    original = "providers:\n  anthropic:\n    apiKey: sk-user-secret\n"
    omp_home.write_bytes(original.encode("utf-8"))
    backup = backup_path(omp_home)

    inject_models_override(8787, "proj")
    assert backup.read_bytes() == original.encode("utf-8")

    _, base_url_9999 = inject_models_override(9999, "proj")

    # Re-injection never clobbers the pristine pre-wrap backup.
    assert backup.read_bytes() == original.encode("utf-8")

    merged = yaml.safe_load(omp_home.read_text(encoding="utf-8"))
    assert base_url_9999 == "http://127.0.0.1:9999/p/proj"
    assert merged["providers"]["anthropic"]["baseUrl"] == base_url_9999
    # Regenerated from the backup, so user creds still survive the new port.
    assert merged["providers"]["anthropic"]["apiKey"] == "sk-user-secret"


# ---------------------------------------------------------------------------
# runtime: restore
# ---------------------------------------------------------------------------


def test_restore_restores_pristine_and_removes_backup(omp_home: Path) -> None:
    original = "providers:\n  anthropic:\n    apiKey: sk-user-secret\n"
    omp_home.write_bytes(original.encode("utf-8"))
    inject_models_override(8787, "proj")

    assert restore_models_override() == "restored"
    assert omp_home.read_bytes() == original.encode("utf-8")
    assert not backup_path(omp_home).exists()


def test_restore_removes_wrap_created_file(omp_home: Path) -> None:
    inject_models_override(8787, "proj")  # fresh create → no backup
    assert omp_home.exists()

    assert restore_models_override() == "removed"
    assert not omp_home.exists()
    assert not backup_path(omp_home).exists()


def test_restore_noop_when_nothing_managed(omp_home: Path) -> None:
    assert restore_models_override() == "noop"


def test_restore_leaves_unmanaged_file_untouched(omp_home: Path) -> None:
    user_content = "providers:\n  anthropic:\n    apiKey: sk-user-secret\n"
    omp_home.write_bytes(user_content.encode("utf-8"))

    assert restore_models_override() == "noop"
    # A models.yml the wrap does not manage is never modified or deleted.
    assert omp_home.read_bytes() == user_content.encode("utf-8")
    assert not backup_path(omp_home).exists()


# ---------------------------------------------------------------------------
# runtime: launch env
# ---------------------------------------------------------------------------


def test_build_launch_env_passes_env_through_and_emits_display(omp_home: Path) -> None:
    source = {"PATH": "/usr/bin", "ANTHROPIC_BASE_URL": "https://api.anthropic.com"}
    env, display = build_launch_env(8787, source, project="proj")

    # The redirect lives in models.yml, so env is a verbatim copy — notably
    # ANTHROPIC_BASE_URL is NOT rewritten to the proxy.
    assert env == source
    assert env is not source  # a copy, so caller's environ can't be mutated
    assert display == ["models.yml: providers.anthropic.baseUrl=http://127.0.0.1:8787/p/proj"]


# ---------------------------------------------------------------------------
# CLI: wrap omp
# ---------------------------------------------------------------------------


def test_wrap_omp_missing_binary_exits_with_install_hint(runner: CliRunner, omp_home: Path) -> None:
    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "omp"])

    assert result.exit_code == 1
    assert "npm install -g @oh-my-pi/pi-coding-agent" in result.output
    # Fail fast before mutating omp's config.
    assert not omp_home.exists()


def test_wrap_omp_happy_path_injects_before_launch(runner: CliRunner, omp_home: Path) -> None:
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)
        # Prove models.yml is on disk BEFORE omp is launched.
        captured["models_text_at_launch"] = (
            omp_home.read_text(encoding="utf-8") if omp_home.exists() else None
        )

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="omp"),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(main, ["wrap", "omp", "--", "-p", "fix the bug"])

    assert result.exit_code == 0, result.output
    assert captured["tool_label"] == "OMP"
    assert captured["agent_type"] == "omp"
    assert captured["args"] == ("-p", "fix the bug")

    text_at_launch = captured["models_text_at_launch"]
    assert isinstance(text_at_launch, str)
    assert MANAGED_MARKER in text_at_launch
    base_url = yaml.safe_load(text_at_launch)["providers"]["anthropic"]["baseUrl"]
    assert base_url.startswith("http://127.0.0.1:8787/p/")

    display = captured["env_vars_display"]
    assert isinstance(display, list)
    assert f"models.yml: providers.anthropic.baseUrl={base_url}" in display


def test_wrap_omp_does_not_write_agents_md(
    runner: CliRunner, omp_home: Path, tmp_path: Path
) -> None:
    """`wrap omp` redirects via models.yml only; it never authors AGENTS.md."""
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="omp"),
        patch("headroom.cli.wrap._launch_tool"),
    ):
        result = runner.invoke(main, ["wrap", "omp"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "AGENTS.md").exists()


# ---------------------------------------------------------------------------
# CLI: unwrap omp
# ---------------------------------------------------------------------------


def test_unwrap_omp_restores_pristine_and_stops_proxy(runner: CliRunner, omp_home: Path) -> None:
    original = "providers:\n  anthropic:\n    apiKey: sk-user-secret\n"
    omp_home.write_bytes(original.encode("utf-8"))
    inject_models_override(8787, "proj")

    stopped: list[int] = []
    with patch(
        "headroom.cli.wrap._stop_local_proxy_for_unwrap",
        side_effect=lambda port: stopped.append(port) or "not_running",
    ):
        result = runner.invoke(main, ["unwrap", "omp"])

    assert result.exit_code == 0, result.output
    assert "Restored pre-wrap models.yml" in result.output
    assert omp_home.read_bytes() == original.encode("utf-8")
    assert not backup_path(omp_home).exists()

    # A real restore (not a noop) attempts to stop the proxy on the given port.
    assert stopped == [8787]


def test_unwrap_omp_removes_wrap_created_file(runner: CliRunner, omp_home: Path) -> None:
    inject_models_override(8787, "proj")  # fresh create → no backup

    stopped: list[int] = []
    with patch(
        "headroom.cli.wrap._stop_local_proxy_for_unwrap",
        side_effect=lambda port: stopped.append(port) or "not_running",
    ):
        result = runner.invoke(main, ["unwrap", "omp", "--port", "9191"])

    assert result.exit_code == 0, result.output
    assert "Removed wrap-created models.yml" in result.output
    assert not omp_home.exists()
    assert stopped == [9191]


def test_unwrap_omp_noop_leaves_unmanaged_and_skips_proxy_stop(
    runner: CliRunner, omp_home: Path
) -> None:
    user_content = "providers:\n  anthropic:\n    apiKey: sk-user-secret\n"
    omp_home.write_bytes(user_content.encode("utf-8"))

    with patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as stop_proxy:
        result = runner.invoke(main, ["unwrap", "omp"])

    assert result.exit_code == 0, result.output
    assert "nothing to restore" in result.output
    # Unmanaged file is left exactly as the user had it.
    assert omp_home.read_bytes() == user_content.encode("utf-8")
    # noop status → the proxy is left running (the `status != "noop"` guard).
    stop_proxy.assert_not_called()
