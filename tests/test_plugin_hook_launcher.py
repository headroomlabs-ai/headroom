from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_FIXTURE = REPO_ROOT / "tests/fixtures/headroom-issue-3039.json"


def _load_reproduction_artifact() -> dict[str, object]:
    configured = os.environ.get("HEADROOM_REPRO_ARTIFACT")
    path = Path(configured) if configured else REPRO_FIXTURE
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_commands() -> list[str]:
    manifest = json.loads(
        (REPO_ROOT / "plugins/headroom-agent-hooks/hooks/hooks.json").read_text(encoding="utf-8")
    )
    return [
        entry["hooks"][0]["command"] for entries in manifest["hooks"].values() for entry in entries
    ]


LAUNCHER = REPO_ROOT / "plugins/headroom-agent-hooks/bin/headroom-hook.sh"
SHELL = shutil.which("sh") or "/bin/sh"


def _posix(path: Path) -> str:
    value = path.resolve().as_posix()
    if len(value) > 2 and value[1] == ":":
        return f"/{value[0].lower()}{value[2:]}"
    return value


def _write_recorder(path: Path, receipt: Path, source: str) -> None:
    path.write_text(
        f'#!/bin/sh\nprintf \'%s %s\\n\' "{source}" "$*" >>"$HEADROOM_RECEIPT"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run_launcher(tmp_path: Path, **extra: str) -> subprocess.CompletedProcess[str]:
    environment = {
        "HOME": _posix(tmp_path / "home"),
        "PATH": "/no-such-bin",
        "HEADROOM_RECEIPT": _posix(tmp_path / "receipt"),
        **extra,
    }
    return subprocess.run(
        [SHELL, _posix(LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def _receipt(tmp_path: Path) -> list[str]:
    receipt = tmp_path / "receipt"
    return receipt.read_text(encoding="utf-8").splitlines() if receipt.exists() else []


def test_manifest_command_recovers_reported_non_login_environment(tmp_path: Path) -> None:
    artifact = _load_reproduction_artifact()
    body = artifact["body"]
    assert isinstance(body, str)
    path_match = re.search(r"PATH=([^\s\\]+)", body)
    assert path_match is not None
    reporter_path = path_match.group(1)
    assert (
        'env -i HOME="$HOME" PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin '
        "/bin/sh -c 'headroom init hook ensure'"
    ) in body
    assert "/bin/sh: headroom: command not found" in body

    home = tmp_path / "home"
    headroom = home / ".local/bin/headroom"
    headroom.parent.mkdir(parents=True)
    receipt = tmp_path / "receipt"
    _write_recorder(headroom, receipt, "home-local")

    environment = {
        "HOME": _posix(home),
        "PATH": reporter_path,
        "HEADROOM_RECEIPT": _posix(receipt),
        "CLAUDE_PLUGIN_ROOT": _posix(REPO_ROOT / "plugins/headroom-agent-hooks"),
    }
    for command in _manifest_commands():
        result = subprocess.run(
            [SHELL, "-c", command],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    assert receipt.read_text(encoding="utf-8").splitlines() == [
        "home-local init hook ensure",
        "home-local init hook ensure",
    ]
    assert "command not found" not in result.stderr


def test_launcher_resolution_precedence(tmp_path: Path) -> None:
    home = tmp_path / "home"
    override = tmp_path / "override"
    path_dir = tmp_path / "path"
    prefix = home / ".local/bin/headroom"
    path_dir.mkdir(parents=True)
    prefix.parent.mkdir(parents=True)
    _write_recorder(override, tmp_path / "receipt", "override")
    _write_recorder(path_dir / "headroom", tmp_path / "receipt", "path")
    _write_recorder(prefix, tmp_path / "receipt", "prefix")

    result = _run_launcher(
        tmp_path,
        HOME=_posix(home),
        PATH=_posix(path_dir),
        HEADROOM_BIN=_posix(override),
    )

    assert result.returncode == 0
    assert _receipt(tmp_path) == ["override init hook ensure"]

    (tmp_path / "receipt").unlink()
    result = _run_launcher(tmp_path, HOME=_posix(home), PATH=_posix(path_dir))
    assert result.returncode == 0
    assert _receipt(tmp_path) == ["path init hook ensure"]

    (tmp_path / "receipt").unlink()
    result = _run_launcher(tmp_path, HOME=_posix(home), PATH="/no-such-bin")
    assert result.returncode == 0
    assert _receipt(tmp_path) == ["prefix init hook ensure"]


def test_launcher_resolves_standard_prefixes(tmp_path: Path) -> None:
    prefixes = [
        tmp_path / "home/.local/bin",
        tmp_path / "home/.local/share/uv/tools/headroom-ai/bin",
        tmp_path / "home/.local/pipx/venvs/headroom-ai/bin",
    ]
    for index, prefix in enumerate(prefixes):
        prefix.mkdir(parents=True)
        name = "headroom.exe" if index == 0 else "headroom"
        _write_recorder(prefix / name, tmp_path / "receipt", f"prefix-{index}")
        result = _run_launcher(tmp_path)
        assert result.returncode == 0
        assert _receipt(tmp_path) == [f"prefix-{index} init hook ensure"]
        (tmp_path / "receipt").unlink()
        (prefix / name).unlink()


def test_launcher_uses_importable_python_module(tmp_path: Path) -> None:
    interpreter = tmp_path / "python-fallback"
    interpreter.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ]; then exit 0; fi\n'
        'printf \'%s\\n\' "$*" >"$HEADROOM_RECEIPT"\n',
        encoding="utf-8",
    )
    interpreter.chmod(0o755)

    result = _run_launcher(tmp_path, HEADROOM_PYTHON=_posix(interpreter))

    assert result.returncode == 0
    assert _receipt(tmp_path) == ["-m headroom.cli init hook ensure"]


def test_launcher_missing_cli_is_nonblocking(tmp_path: Path) -> None:
    result = _run_launcher(tmp_path)

    assert result.returncode == 0
    assert _receipt(tmp_path) == []
    assert result.stderr.splitlines() == [
        "headroom: CLI not found; install with 'uv tool install headroom-ai' or set HEADROOM_BIN; compression hooks are inactive."
    ]


def test_launcher_skips_non_executable_fixed_prefix_candidate(tmp_path: Path) -> None:
    candidate = tmp_path / "home/.local/bin/headroom"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("not executable", encoding="utf-8")

    result = _run_launcher(tmp_path)

    assert result.returncode == 0
    assert _receipt(tmp_path) == []
    assert "CLI not found" in result.stderr


def test_launcher_skips_non_importable_python(tmp_path: Path) -> None:
    interpreter = tmp_path / "python-not-headroom"
    interpreter.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    interpreter.chmod(0o755)

    result = _run_launcher(tmp_path, HEADROOM_PYTHON=_posix(interpreter))

    assert result.returncode == 0
    assert _receipt(tmp_path) == []
    assert "CLI not found" in result.stderr


def test_launcher_declares_absolute_standard_prefixes() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert '"/opt/homebrew/bin"' in source
    assert '"/usr/local/bin"' in source


def test_manifest_rootless_tail_reaches_path(tmp_path: Path) -> None:
    stub = tmp_path / "path/headroom"
    receipt = tmp_path / "receipt"
    stub.parent.mkdir(parents=True)
    _write_recorder(stub, receipt, "rootless-path")

    for plugin_root in (None, ""):
        environment = {
            "PATH": _posix(stub.parent) + os.pathsep + os.environ["PATH"],
            "HEADROOM_RECEIPT": _posix(receipt),
            "CLAUDE_PLUGIN_ROOT": plugin_root,
        }
        for command in _manifest_commands():
            result = subprocess.run(
                [SHELL, "-c", command],
                cwd=REPO_ROOT,
                env={key: value for key, value in environment.items() if value is not None},
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr

    assert _receipt(tmp_path) == [
        "rootless-path init hook ensure",
        "rootless-path init hook ensure",
        "rootless-path init hook ensure",
        "rootless-path init hook ensure",
    ]


def test_manifest_nonempty_missing_or_unreadable_root_reaches_path(tmp_path: Path) -> None:
    stub = tmp_path / "path/headroom"
    receipt = tmp_path / "receipt"
    stub.parent.mkdir(parents=True)
    _write_recorder(stub, receipt, "fallback-path")

    missing_root = tmp_path / "missing-plugin"
    unreadable_root = tmp_path / "unreadable-plugin"
    unreadable_root.write_text("plugin root is not a directory", encoding="utf-8")

    for plugin_root in (missing_root, unreadable_root):
        environment = {
            "PATH": _posix(stub.parent) + ":" + os.environ["PATH"],
            "HEADROOM_RECEIPT": _posix(receipt),
            "CLAUDE_PLUGIN_ROOT": _posix(plugin_root),
        }
        for command in _manifest_commands():
            result = subprocess.run(
                [SHELL, "-c", command],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
        assert _receipt(tmp_path) == [
            "fallback-path init hook ensure",
            "fallback-path init hook ensure",
        ]
        receipt.unlink()
