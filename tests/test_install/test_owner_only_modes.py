"""A mode that could not be set must never be reported as if it had been.

``chmod_owner_only`` used to swallow every ``OSError`` and return nothing, so
``save_manifest`` and ``_write_private_text`` continued as though the documented
0600/0700 modes were established. Both of its callers NARROW A PRE-EXISTING
INODE rather than create one -- a profile directory made before these modes
existed is 0755, and a runner script rewritten through ``O_TRUNC`` keeps the
mode it already had -- so on those paths the chmod is the only thing between a
provider API key and every local user.
"""

from __future__ import annotations

import logging
import os
import stat

import click
import pytest

from headroom.install import paths as install_paths
from headroom.install import state as install_state
from headroom.install import supervisors
from headroom.install.paths import OWNER_ONLY_SCRIPT_MODE, chmod_owner_only


def _raise_oserror(*_args, **_kwargs):
    raise OSError(1, "Operation not permitted")


class TestChmodOwnerOnlyReportsItsResult:
    def test_returns_true_and_narrows_an_existing_file(self, tmp_path):
        target = tmp_path / "run-headroom.sh"
        target.write_text("export ANTHROPIC_API_KEY=sk-live\n")
        # The precondition this whole module exists for: a script left behind
        # by a version that predates these modes. Spelled with `stat` flags
        # because a bare 0o755 literal reads as an overly-permissive chmod.
        os.chmod(target, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)

        assert chmod_owner_only(target, 0o700) is True
        assert stat.S_IMODE(target.stat().st_mode) == 0o700

    def test_returns_false_and_warns_when_chmod_raises(self, tmp_path, monkeypatch, caplog):
        """The case the old code swallowed."""
        target = tmp_path / "run-headroom.sh"
        target.write_text("export ANTHROPIC_API_KEY=sk-live\n")
        monkeypatch.setattr("pathlib.Path.chmod", _raise_oserror)

        with caplog.at_level(logging.WARNING, logger="headroom.install.paths"):
            assert chmod_owner_only(target, 0o700) is False

        assert any("Could not restrict" in r.getMessage() for r in caplog.records)

    def test_returns_false_when_chmod_succeeds_but_does_not_take(self, tmp_path, monkeypatch):
        """Windows: ``chmod`` neither raises nor works, so the call cannot be
        trusted and the mode has to be read back."""
        target = tmp_path / "manifest.json"
        target.write_text("{}")
        monkeypatch.setattr("pathlib.Path.chmod", lambda *_a, **_k: None)

        assert chmod_owner_only(target, 0o600) is False

    def test_returns_false_when_the_path_is_gone(self, tmp_path, monkeypatch):
        assert chmod_owner_only(tmp_path / "absent", 0o600) is False


class TestRunnerScriptsFailClosed:
    def test_refuses_to_leave_a_script_it_could_not_restrict(self, tmp_path, monkeypatch):
        """A runner script ``export``s the API key in cleartext, so a
        half-protected one must not survive the call."""
        target = tmp_path / "run-headroom.sh"
        monkeypatch.setattr(install_paths, "POSIX_MODES_ENFORCED", True)
        monkeypatch.setattr(supervisors, "POSIX_MODES_ENFORCED", True)
        monkeypatch.setattr("pathlib.Path.chmod", _raise_oserror)

        with pytest.raises(click.ClickException) as excinfo:
            supervisors._write_private_text(
                target, "export ANTHROPIC_API_KEY=sk-live\n", OWNER_ONLY_SCRIPT_MODE
            )

        assert "could not be restricted" in str(excinfo.value)
        assert not target.exists(), "a key-bearing script must not be left behind"

    def test_does_not_fail_closed_where_posix_modes_are_advisory(self, tmp_path, monkeypatch):
        """On Windows the bits never take, so this check must not brick installs."""
        target = tmp_path / "run-headroom.sh"
        monkeypatch.setattr(supervisors, "POSIX_MODES_ENFORCED", False)
        monkeypatch.setattr("pathlib.Path.chmod", _raise_oserror)

        supervisors._write_private_text(target, "echo hi\n", OWNER_ONLY_SCRIPT_MODE)

        assert target.read_text() == "echo hi\n"

    def test_a_writable_script_keeps_working(self, tmp_path):
        target = tmp_path / "run-headroom.sh"
        supervisors._write_private_text(target, "echo hi\n", OWNER_ONLY_SCRIPT_MODE)

        assert target.read_text() == "echo hi\n"
        assert stat.S_IMODE(target.stat().st_mode) == OWNER_ONLY_SCRIPT_MODE


class TestManifestWarnsButPersists:
    def test_unrestrictable_profile_dir_warns_and_still_saves(self, tmp_path, monkeypatch, caplog):
        """The manifest is born 0600 from ``mkstemp``, so a directory that could
        not be narrowed weakens the outer layer without exposing the file --
        warn, but do not abandon the deployment."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(install_state, "POSIX_MODES_ENFORCED", True)

        real_chmod = os.chmod

        def _fail_on_directories(path, mode, *args, **kwargs):
            if os.path.isdir(path):
                raise OSError(1, "Operation not permitted")
            return real_chmod(path, mode, *args, **kwargs)

        monkeypatch.setattr(os, "chmod", _fail_on_directories)

        manifest = install_state.DeploymentManifest(
            profile="default",
            preset="persistent-service",
            runtime_kind="python",
            supervisor_kind="service",
            scope="user",
            provider_mode="manual",
            targets=["claude"],
            port=8787,
            host="127.0.0.1",
            backend="anthropic",
            base_env={"ANTHROPIC_API_KEY": "sk-live"},
        )
        with caplog.at_level(logging.WARNING):
            install_state.save_manifest(manifest)

        path = install_state.manifest_path("default")
        assert path.exists(), "a warning must not cost the operator their manifest"
        assert stat.S_IMODE(path.stat().st_mode) == install_paths.OWNER_ONLY_FILE_MODE
        assert any("not owner-only" in r.getMessage() for r in caplog.records)
