"""Keep local Docker action metadata and its CI coverage valid."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_local_action_edits_trigger_code_and_workflow_checks() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    filters = yaml.safe_load(workflow["jobs"]["changes"]["steps"][1]["with"]["filters"])
    for group in ("code", "workflows"):
        assert ".github/actions/**" in filters[group], group


def test_vendored_commitlint_action_metadata() -> None:
    actions = [ROOT / ".github/actions/commitlint/action.yml"]
    for path in actions:
        action = yaml.safe_load(path.read_text())
        assert isinstance(action, dict), path
        for key in ("name", "description"):
            assert isinstance(action.get(key), str) and action[key].strip(), (path, key)
        runs = action["runs"]
        assert runs["using"] == "docker", path
        assert re.fullmatch(r"docker://[^\s@]+@sha256:[0-9a-f]{64}", runs["image"]), path
        for name, definition in action.get("inputs", {}).items():
            assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name), path
            assert isinstance(definition["description"], str), (path, name)
            if "default" in definition:
                assert isinstance(definition["default"], str), (path, name)
            if "required" in definition:
                assert isinstance(definition["required"], bool), (path, name)
        for name, definition in action.get("outputs", {}).items():
            assert isinstance(definition["description"], str), (path, name)
