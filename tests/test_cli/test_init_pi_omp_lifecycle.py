from __future__ import annotations

import inspect
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner, Result

from headroom.cli.main import main
from headroom.install.paths import manifest_path, pid_path, profile_root
from headroom.providers.omp.runtime import backup_path
from headroom.providers.pi_extension import PACKAGE_NAME, extension_config_path


@pytest.fixture
def isolated_hosts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    agent = tmp_path / "omp-agent"
    bin_dir = tmp_path / "bin"
    host_state = tmp_path / "host-state.json"
    host_log = tmp_path / "host-operations.jsonl"
    scheduler = tmp_path / "crontab"
    for path in (home, workspace, agent, bin_dir):
        path.mkdir(parents=True)

    host_state.write_text("{}\n", encoding="utf-8")
    host_log.write_text("", encoding="utf-8")
    python = bin_dir / "python3"
    python.symlink_to(Path(sys.executable))
    fake_host = bin_dir / "fake-host"
    fake_host.write_text(
        """#!/usr/bin/env python3
import json
import os
import pathlib
import sys

host = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
state_path = pathlib.Path(os.environ["HEADROOM_TEST_HOST_STATE"])
log_path = pathlib.Path(os.environ["HEADROOM_TEST_HOST_LOG"])
state = json.loads(state_path.read_text())
package = "headroom-pi"

def save(action, version=None):
    state_path.write_text(json.dumps(state, sort_keys=True) + "\\n")
    with log_path.open("a") as log:
        log.write(json.dumps({"host": host, "action": action, "version": version}) + "\\n")

if host == "pi" and args[:1] == ["install"] and len(args) == 2:
    source = args[1].removeprefix("npm:")
    name, version = source.rsplit("@", 1)
    if name != package:
        raise SystemExit(2)
    state[host] = version
    settings = pathlib.Path(os.environ["PI_CODING_AGENT_DIR"]) / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({"packages": [f"npm:{package}@{version}"]}) + "\\n")
    save("install", version)
elif host == "pi" and args == ["remove", f"npm:{package}"]:
    state.pop(host, None)
    settings = pathlib.Path(os.environ["PI_CODING_AGENT_DIR"]) / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text('{"packages": []}\\n')
    save("remove")
elif host == "omp" and args == ["plugin", "list", "--json"]:
    version = state.get(host)
    plugins = [] if version is None else [{"name": package, "version": version, "enabled": True}]
    print(json.dumps({"npm": plugins}))
elif host == "omp" and len(args) == 5 and args[:2] == ["plugin", "install"] and args[4] == "--json":
    raise SystemExit(2)
elif host == "omp" and len(args) == 4 and args[:2] == ["plugin", "install"] and args[3] == "--json":
    name, version = args[2].rsplit("@", 1)
    if name != package:
        raise SystemExit(2)
    state[host] = version
    save("install", version)
    print('{}')
elif host == "omp" and args == ["plugin", "uninstall", package, "--json"]:
    state.pop(host, None)
    save("remove")
    print('{}')
else:
    raise SystemExit(f"unsupported {host} command: {args!r}")
""",
        encoding="utf-8",
    )
    fake_host.chmod(fake_host.stat().st_mode | stat.S_IXUSR)
    for host in ("pi", "omp"):
        (bin_dir / host).symlink_to(fake_host)

    crontab = bin_dir / "crontab"
    crontab.write_text(
        """#!/usr/bin/env python3
import os
import pathlib
import sys

path = pathlib.Path(os.environ["HEADROOM_TEST_CRONTAB"])
args = sys.argv[1:]
if args == ["-l"]:
    if not path.exists():
        print("no crontab for test", file=sys.stderr)
        raise SystemExit(1)
    sys.stdout.buffer.write(path.read_bytes())
elif args == ["-"]:
    path.write_bytes(sys.stdin.buffer.read())
elif args == ["-r"]:
    if not path.exists():
        print("no crontab for test", file=sys.stderr)
        raise SystemExit(1)
    path.unlink()
else:
    raise SystemExit(f"unsupported crontab command: {args!r}")
""",
        encoding="utf-8",
    )
    crontab.chmod(crontab.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent))
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("HEADROOM_TEST_HOST_STATE", str(host_state))
    monkeypatch.setenv("HEADROOM_TEST_HOST_LOG", str(host_log))
    monkeypatch.setenv("HEADROOM_TEST_CRONTAB", str(scheduler))
    monkeypatch.setenv("HEADROOM_UPDATE_CHECK", "off")

    from headroom.install import supervisors

    callback = cast(Any, main.commands["init"].callback)
    init_globals = inspect.unwrap(callback).__globals__
    monkeypatch.setattr(init_globals["sys"], "platform", "linux")
    monkeypatch.setattr(supervisors.sys, "platform", "linux")

    def start_runtime(manifest: Any) -> None:
        path = pid_path(manifest.profile)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("424242", encoding="utf-8")

    monkeypatch.setitem(init_globals, "_start_profile_strict_locked", start_runtime)
    monkeypatch.setitem(
        init_globals,
        "stop_runtime",
        lambda manifest: pid_path(manifest.profile).unlink(missing_ok=True),
    )

    return {
        "home": home,
        "workspace": workspace,
        "agent": agent,
        "host_state": host_state,
        "host_log": host_log,
        "scheduler": scheduler,
    }


def _invoke(*args: str) -> Result:
    result = CliRunner().invoke(main, ["init", "-g", *args])
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    return result


def _operations(path: Path) -> list[dict[str, str | None]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _manifest() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(manifest_path("init-user").read_text(encoding="utf-8"))
    return payload


def _assert_wrapper_unchanged(models: Path, models_bytes: bytes, backup_bytes: bytes) -> None:
    assert models.read_bytes() == models_bytes
    assert backup_path(models).read_bytes() == backup_bytes


def test_pi_omp_durable_lifecycle_is_idempotent_and_wrapper_safe(
    monkeypatch: pytest.MonkeyPatch, isolated_hosts: dict[str, Path]
) -> None:
    callback = cast(Any, main.commands["init"].callback)
    init_globals = inspect.unwrap(callback).__globals__

    models = isolated_hosts["agent"] / "models.yml"
    models_bytes = b"providers:\n  anthropic:\n    baseUrl: http://wrapper.invalid\n"
    backup_bytes = b"providers:\n  anthropic:\n    baseUrl: https://api.anthropic.com\n"
    models.write_bytes(models_bytes)
    backup_path(models).write_bytes(backup_bytes)
    assert Path.home() == isolated_hosts["home"]
    assert isolated_hosts["workspace"] / "deploy" / "init-user" == profile_root("init-user")
    assert isolated_hosts["home"].parent / "bin" == Path(os.environ["PATH"])

    monkeypatch.setitem(init_globals, "_HEADROOM_VERSION", "0.34.0")
    _invoke("pi")
    _assert_wrapper_unchanged(models, models_bytes, backup_bytes)
    assert pid_path("init-user").is_file()
    config_034 = extension_config_path().read_bytes()
    assert json.loads(config_034) == {"baseUrl": "http://127.0.0.1:8787"}
    task_marker = b"# >>> headroom init-user >>>"
    assert json.loads(isolated_hosts["host_state"].read_text()) == {"pi": "0.34.0"}
    assert _operations(isolated_hosts["host_log"]) == [
        {"host": "pi", "action": "install", "version": "0.34.0"}
    ]
    manifest = _manifest()
    assert manifest["profile"] == "init-user"
    assert manifest["targets"] == ["pi"]
    assert manifest["supervisor_kind"] == "task"
    package = next(item for item in manifest["artifacts"] if item["kind"] == "pi-extension-package")
    assert package == {
        "kind": "pi-extension-package",
        "path": "pi",
        "metadata": {
            "package": PACKAGE_NAME,
            "version": "0.34.0",
            "owned": True,
            "source": "npm",
        },
    }
    config = next(item for item in manifest["artifacts"] if item["kind"] == "pi-extension-config")
    assert config["path"] == str(extension_config_path().absolute())
    assert config["metadata"]["previous_exists"] is False
    assert config["metadata"]["previous_base64"] == ""
    assert config["metadata"]["managed_sha256"]
    assert {item["kind"] for item in manifest["artifacts"]} >= {"script", "crontab"}

    _invoke("pi")
    _assert_wrapper_unchanged(models, models_bytes, backup_bytes)
    assert pid_path("init-user").is_file()
    artifacts = _manifest()["artifacts"]
    assert len({(item["kind"], item["path"]) for item in artifacts}) == len(artifacts)
    assert extension_config_path().read_bytes() == config_034
    assert isolated_hosts["scheduler"].read_bytes().count(task_marker) == 1
    assert len(_operations(isolated_hosts["host_log"])) == 1

    _invoke("omp")
    _assert_wrapper_unchanged(models, models_bytes, backup_bytes)
    assert pid_path("init-user").is_file()
    assert json.loads(isolated_hosts["host_state"].read_text()) == {
        "pi": "0.34.0",
        "omp": "0.34.0",
    }
    assert _manifest()["targets"] == ["omp", "pi"]
    assert extension_config_path().read_bytes() == config_034
    assert isolated_hosts["scheduler"].read_bytes().count(task_marker) == 1

    monkeypatch.setitem(init_globals, "_HEADROOM_VERSION", "0.35.0")
    _invoke("pi")
    _assert_wrapper_unchanged(models, models_bytes, backup_bytes)
    assert pid_path("init-user").is_file()
    _invoke("omp")
    _assert_wrapper_unchanged(models, models_bytes, backup_bytes)
    assert pid_path("init-user").is_file()
    assert json.loads(isolated_hosts["host_state"].read_text()) == {
        "pi": "0.35.0",
        "omp": "0.35.0",
    }

    monkeypatch.setitem(init_globals, "_HEADROOM_VERSION", "0.34.0")
    _invoke("pi")
    _assert_wrapper_unchanged(models, models_bytes, backup_bytes)
    assert pid_path("init-user").is_file()
    _invoke("omp")
    _assert_wrapper_unchanged(models, models_bytes, backup_bytes)
    assert pid_path("init-user").is_file()
    assert json.loads(isolated_hosts["host_state"].read_text()) == {
        "pi": "0.34.0",
        "omp": "0.34.0",
    }

    _invoke("remove", "pi")
    _assert_wrapper_unchanged(models, models_bytes, backup_bytes)
    assert pid_path("init-user").is_file()
    assert json.loads(isolated_hosts["host_state"].read_text()) == {"omp": "0.34.0"}
    assert _manifest()["targets"] == ["omp"]
    assert extension_config_path().read_bytes() == config_034
    assert isolated_hosts["scheduler"].read_bytes().count(task_marker) == 1

    _invoke("remove", "omp")
    _assert_wrapper_unchanged(models, models_bytes, backup_bytes)
    assert json.loads(isolated_hosts["host_state"].read_text()) == {}
    assert not extension_config_path().exists()
    assert isolated_hosts["scheduler"].read_bytes() == b""
    assert not profile_root("init-user").exists()
    assert not pid_path("init-user").exists()

    versions = [
        entry["version"]
        for entry in _operations(isolated_hosts["host_log"])
        if entry["action"] == "install"
    ]
    assert versions == ["0.34.0", "0.34.0", "0.35.0", "0.35.0", "0.34.0", "0.34.0"]


def test_preexisting_pi_package_is_never_removed(
    monkeypatch: pytest.MonkeyPatch, isolated_hosts: dict[str, Path]
) -> None:
    callback = cast(Any, main.commands["init"].callback)
    init_globals = inspect.unwrap(callback).__globals__

    settings = isolated_hosts["agent"] / "settings.json"
    settings.write_text(json.dumps({"packages": [f"npm:{PACKAGE_NAME}@0.34.0"]}) + "\n")
    isolated_hosts["host_state"].write_text('{"pi": "0.34.0"}\n', encoding="utf-8")
    monkeypatch.setitem(init_globals, "_HEADROOM_VERSION", "0.34.0")

    _invoke("pi")
    package = next(
        item for item in _manifest()["artifacts"] if item["kind"] == "pi-extension-package"
    )
    assert package["metadata"]["owned"] is False
    _invoke("remove", "pi")

    assert json.loads(isolated_hosts["host_state"].read_text()) == {"pi": "0.34.0"}
    assert _operations(isolated_hosts["host_log"]) == []
