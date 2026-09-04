"""Prove settings.json > env > default precedence in a genuinely separate process.

A mocked ``restart_current_deployment`` proves dispatch only -- it can't
prove settings actually take effect the way a real restarted proxy would
pick them up. This spawns a real subprocess that imports
``headroom.settings_store`` cold and reports what it observes, closing the
gap a Codex red-team pass flagged in the original (mock-only) test plan.
"""

import json
import os
import subprocess
import sys

import pytest

from headroom import paths, settings_store


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the workspace dir (settings.json) at an isolated tmp dir."""
    monkeypatch.setenv(paths.HEADROOM_WORKSPACE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(paths.HEADROOM_SETTINGS_PATH_ENV, raising=False)
    return tmp_path


def test_file_beats_env_beats_default_in_subprocess(workspace, monkeypatch):
    for field in settings_store.SETTINGS:
        monkeypatch.delenv(field.env, raising=False)
    # rpm/target_ratio are normal knobs; port is manifest_managed (env wins there).
    settings_store.save({"target_ratio": 0.3, "rpm": 20, "port": 9898})

    script = (
        "import os, json\n"
        "from headroom import settings_store\n"
        "settings_store.apply_to_environ(settings_store.load())\n"
        "print(json.dumps({'rpm': os.environ.get('HEADROOM_RPM'), "
        "'target_ratio': os.environ.get('HEADROOM_TARGET_RATIO'), "
        "'port': os.environ.get('HEADROOM_PORT')}))\n"
    )
    env = dict(os.environ)
    env["HEADROOM_WORKSPACE_DIR"] = str(workspace)
    env["HEADROOM_RPM"] = "999"  # explicit export: the file's 20 now wins (settings-first)
    env["HEADROOM_PORT"] = "7777"  # manifest_managed: the export must keep precedence
    env.pop(
        "HEADROOM_TARGET_RATIO", None
    )  # not exported: file's 0.3 must win over the code default

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["rpm"] == "20"  # settings.json overrides the shell export
    assert out["target_ratio"] == "0.3"  # file over code default
    assert out["port"] == "7777"  # manifest_managed knob: the export still wins
