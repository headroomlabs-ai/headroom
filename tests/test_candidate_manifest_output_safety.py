import os
import sys

import pytest

from scripts.candidate_manifest import main
from tests.test_candidate_manifest import _create


@pytest.mark.parametrize("target", ["artifact", "rollout", "runtime_payload"])
@pytest.mark.parametrize("alias", ["direct", "hardlink", "symlink"])
def test_create_rejects_output_aliasing_an_input(monkeypatch, tmp_path, target, alias):
    artifact, manifest, runtime_payload = _create(monkeypatch, tmp_path)
    inputs = {
        "artifact": artifact,
        "rollout": tmp_path / "rollout.json",
        "runtime_payload": runtime_payload,
    }
    protected = inputs[target]
    output = protected
    if alias == "hardlink":
        output = tmp_path / "output-alias.json"
        os.link(protected, output)
    elif alias == "symlink":
        output = tmp_path / "output-alias.json"
        try:
            output.symlink_to(protected)
        except OSError:
            pytest.skip("symlink creation unavailable")
    before = {path: path.read_bytes() for path in inputs.values()}
    argv = list(sys.argv)
    argv[argv.index("--output") + 1] = str(output)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="output"):
        main()
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("alias", ["direct", "hardlink", "symlink", "new_inside"])
def test_inventory_output_cannot_modify_its_payload(monkeypatch, tmp_path, alias):
    directory = tmp_path / "dist"
    directory.mkdir()
    payload = directory / "candidate.whl"
    payload.write_bytes(b"original package bytes")
    output = payload
    if alias == "hardlink":
        output = tmp_path / "inventory-alias.json"
        os.link(payload, output)
    elif alias == "new_inside":
        output = directory / "inventory.json"
    elif alias == "symlink":
        output = tmp_path / "inventory-alias.json"
        try:
            output.symlink_to(payload)
        except OSError:
            pytest.skip("symlink creation unavailable")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "candidate_manifest.py",
            "inventory",
            "--directory",
            str(directory),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="output"):
        main()
    assert payload.read_bytes() == b"original package bytes"
    assert not (directory / "inventory.json").exists()


def test_failed_atomic_replace_preserves_output_and_inputs(monkeypatch, tmp_path):
    artifact, manifest, runtime_payload = _create(monkeypatch, tmp_path)
    before = {path: path.read_bytes() for path in (artifact, manifest, runtime_payload)}

    def fail_replace(source, destination):
        raise OSError("injected replacement failure")

    monkeypatch.setattr("scripts.candidate_manifest.os.replace", fail_replace)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--candidate-id", "replacement-candidate"])
    with pytest.raises(OSError, match="replacement failure"):
        main()
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not list(tmp_path.glob(".candidate-output-*.tmp"))
