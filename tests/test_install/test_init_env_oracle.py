"""Real-shell regression controls for the init E2E environment assertions."""

import json
import os
import shutil
from pathlib import Path

import pytest

from e2e._lib import CaseContext
from e2e.init.run import _verify_copilot_global
from headroom.install.providers import _env_block

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason="The Docker init E2E oracle requires a native POSIX Bash",
)

EXPECTED = {
    "COPILOT_PROVIDER_TYPE": "openai",
    "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:9005/v1",
    "COPILOT_PROVIDER_WIRE_API": "completions",
}


@pytest.fixture
def copilot_context(tmp_path: Path) -> CaseContext:
    home = tmp_path / "home with spaces"
    (home / ".copilot").mkdir(parents=True)
    config = {
        "hooks": {
            "SessionStart": [{"command": "headroom init hook ensure --profile init-user"}],
            "PreToolUse": [],
        }
    }
    (home / ".copilot" / "config.json").write_text(
        json.dumps(config), encoding="utf-8", newline="\n"
    )
    log = tmp_path / "shim-log.jsonl"
    calls = [
        ["plugin", "marketplace", "add", "/workspace"],
        ["plugin", "install", "headroom@headroom-marketplace"],
    ]
    log.write_text(
        "".join(json.dumps({"tool": "copilot", "argv": argv}) + "\n" for argv in calls),
        encoding="utf-8",
        newline="\n",
    )
    for name in [".bashrc", ".zshrc", ".profile"]:
        (home / name).write_text(_env_block(EXPECTED), encoding="utf-8", newline="\n")
    return CaseContext(
        name="seq_copilot_global",
        home=home,
        project=tmp_path,
        shim_dir=tmp_path,
        shim_log=log,
        stdout="",
        stderr="",
        exit_code=0,
    )


@pytest.mark.parametrize("style", ["producer", "double", "single"])
def test_copilot_e2e_accepts_equivalent_export_spellings(
    copilot_context: CaseContext, style: str
) -> None:
    if style != "producer":
        quote = '"' if style == "double" else "'"
        content = "\n".join(
            f"export {name}={quote}{value}{quote}" for name, value in EXPECTED.items()
        )
        for name in [".bashrc", ".zshrc", ".profile"]:
            (copilot_context.home / name).write_text(content, encoding="utf-8", newline="\n")
    _verify_copilot_global(copilot_context)


@pytest.mark.parametrize("shell_file", [".bashrc", ".zshrc", ".profile"])
@pytest.mark.parametrize("mutation", ["missing", "unexported", "wrong", "source-failure"])
def test_copilot_e2e_rejects_invalid_runtime_environment(
    copilot_context: CaseContext, monkeypatch: pytest.MonkeyPatch, shell_file: str, mutation: str
) -> None:
    # Ambient variables must never turn a missing/unexported setting into a pass.
    for name, value in EXPECTED.items():
        monkeypatch.setenv(name, value)
    startup = copilot_context.home / "ambient-startup"
    startup.write_text(_env_block(EXPECTED), encoding="utf-8", newline="\n")
    monkeypatch.setenv("BASH_ENV", str(startup))
    content = _env_block(EXPECTED)
    if mutation == "missing":
        content = "\n".join(
            line
            for line in content.splitlines()
            if not line.startswith("export COPILOT_PROVIDER_TYPE=")
        )
    elif mutation == "unexported":
        content = content.replace("export COPILOT_PROVIDER_TYPE=", "COPILOT_PROVIDER_TYPE=")
    elif mutation == "wrong":
        content = content.replace("COPILOT_PROVIDER_TYPE=openai", "COPILOT_PROVIDER_TYPE=wrong")
    else:
        content = "return 23\n" + content
    (copilot_context.home / shell_file).write_text(content, encoding="utf-8", newline="\n")
    with pytest.raises(AssertionError):
        _verify_copilot_global(copilot_context)
