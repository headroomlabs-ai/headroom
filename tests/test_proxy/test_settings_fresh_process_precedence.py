"""Prove env > file > default precedence in a genuinely separate process.

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


def test_env_beats_file_beats_default_in_subprocess(workspace, monkeypatch):
    for field in settings_store.SETTINGS:
        monkeypatch.delenv(field.env, raising=False)
    settings_store.save({"target_ratio": 0.3, "rpm": 20})

    script = (
        "import os, json\n"
        "from headroom import settings_store\n"
        "settings_store.apply_to_environ(settings_store.load())\n"
        "print(json.dumps({'rpm': os.environ.get('HEADROOM_RPM'), "
        "'target_ratio': os.environ.get('HEADROOM_TARGET_RATIO')}))\n"
    )
    env = dict(os.environ)
    env["HEADROOM_WORKSPACE_DIR"] = str(workspace)
    env["HEADROOM_RPM"] = "999"  # explicit export: must win over the file's 20
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
    assert out["rpm"] == "999"
    assert out["target_ratio"] == "0.3"


@pytest.mark.parametrize("exported_mode", [None, "token", "cache"])
@pytest.mark.parametrize("apply_count", [1, 2])
def test_saved_settings_stay_editable_after_startup(
    workspace, monkeypatch, exported_mode, apply_count
):
    for field in settings_store.SETTINGS:
        monkeypatch.delenv(field.env, raising=False)
    settings_store.save(
        {
            "mode": "token",
            "target_ratio": 0.3,
            "disable_kompress": True,
            "code_aware_enabled": False,
        }
    )
    env = dict(os.environ)
    if exported_mode is not None:
        env["HEADROOM_MODE"] = exported_mode
    script = (
        "import json\n"
        "from headroom import settings_store\n"
        f"for _ in range({apply_count}):\n"
        "    settings_store.apply_to_environ(settings_store.load())\n"
        "print(json.dumps({f['key']: f for f in settings_store.to_schema()['fields']}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=30
    )
    assert result.returncode == 0, result.stderr
    fields = json.loads(result.stdout)
    assert fields["mode"]["value"] == (exported_mode or "token")
    # A real export must stay locked, even if its value equals settings.json.
    assert fields["mode"]["env_override"] is (exported_mode is not None)
    for key in ("target_ratio", "disable_kompress", "code_aware_enabled"):
        assert fields[key]["env_override"] is False


@pytest.mark.parametrize("change", ["save", "export"])
def test_seed_provenance_tracks_the_running_value(workspace, monkeypatch, change):
    for field in settings_store.SETTINGS:
        monkeypatch.delenv(field.env, raising=False)
    settings_store.save({"mode": "token"})
    update = (
        "settings_store.save({'mode': 'cache'})"
        if change == "save"
        else "os.environ['HEADROOM_MODE'] = 'cache'"
    )
    script = (
        "import json, os\n"
        "from headroom import settings_store\n"
        "settings_store.apply_to_environ(settings_store.load())\n"
        f"{update}\n"
        "print(json.dumps(next(f for f in settings_store.to_schema()['fields'] "
        "if f['key'] == 'mode')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    field = json.loads(result.stdout)
    assert field["env_override"] is (change == "export")
    assert field["value"] == ("cache" if change == "export" else "token")


@pytest.mark.parametrize("exported_mode", [None, "token"])
def test_saved_settings_provenance_survives_worker_config(workspace, monkeypatch, exported_mode):
    for field in settings_store.SETTINGS:
        monkeypatch.delenv(field.env, raising=False)
    settings_store.save({"mode": "token"})
    env = dict(os.environ)
    if exported_mode is not None:
        env["HEADROOM_MODE"] = exported_mode
    # Build the same serialized config used by run_server before spawning workers.
    parent_script = (
        "import json\n"
        "from headroom import settings_store\n"
        "from headroom.proxy.models import ProxyConfig\n"
        "from headroom.proxy.server import _proxy_config_payload\n"
        "settings_store.apply_to_environ(settings_store.load())\n"
        "print(json.dumps(_proxy_config_payload(ProxyConfig(mode='token', worker_processes=2))))\n"
    )
    parent = subprocess.run(
        [sys.executable, "-c", parent_script], capture_output=True, text=True, env=env, timeout=30
    )
    assert parent.returncode == 0, parent.stderr
    env["HEADROOM_PROXY_CONFIG_JSON"] = parent.stdout.strip()
    env["HEADROOM_MODE"] = "token"  # inherited by a spawned worker in both cases
    worker_script = (
        "import json\n"
        "from headroom import settings_store\n"
        "from headroom.proxy.server import _proxy_config_from_env\n"
        "config = _proxy_config_from_env()\n"
        "schema = settings_store.to_schema(env_override_exclusions=config.profile_seeded_env_keys)\n"
        "print(json.dumps(next(f for f in schema['fields'] if f['key'] == 'mode')))\n"
    )
    worker = subprocess.run(
        [sys.executable, "-c", worker_script], capture_output=True, text=True, env=env, timeout=30
    )
    assert worker.returncode == 0, worker.stderr
    assert json.loads(worker.stdout)["env_override"] is (exported_mode is not None)
