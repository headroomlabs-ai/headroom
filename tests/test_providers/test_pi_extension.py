import base64
import hashlib
import json
import os
import stat
from pathlib import Path
from subprocess import CompletedProcess
from typing import Literal

import click
import pytest

import headroom.providers.pi_extension as module
from headroom.install.models import ArtifactRecord
from headroom.providers.pi_extension import (
    PACKAGE_NAME,
    PackageState,
    _inspect_pi_settings,
    ensure_extension_config,
    ensure_host_package,
    extension_config_path,
    extension_release_version,
    inspect_host_package,
    remove_owned_extension_config,
    remove_owned_host_package,
)


def successful(calls: list[list[str]], command: list[str], stdout: str = ""):
    calls.append(command)
    return CompletedProcess(command, 0, stdout, "")


def failed(calls: list[list[str]], command: list[str]):
    calls.append(command)
    return CompletedProcess(command, 1, "", "native command failed")


def package_artifact(host: str, version: str, *, owned: bool) -> ArtifactRecord:
    return ArtifactRecord(
        kind="pi-extension-package",
        path=host,
        metadata={
            "package": PACKAGE_NAME,
            "version": version,
            "owned": owned,
            "source": "npm",
        },
    )


def test_release_version_rejects_development_builds() -> None:
    with pytest.raises(click.ClickException, match="released Headroom version"):
        extension_release_version("0.35.0-dev")


@pytest.mark.parametrize(
    "entry",
    [
        "npm:headroom-pi@0.34.0",
        {"source": "npm:headroom-pi@0.34.0"},
    ],
)
def test_pi_settings_support_string_and_object_entries(tmp_path, entry) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"packages": [entry]}), encoding="utf-8")

    assert _inspect_pi_settings(settings) == PackageState("0.34.0", "npm")


def test_pi_settings_respect_relocated_config_directory(tmp_path, monkeypatch) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"packages": ["npm:headroom-pi@0.34.0"]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))

    assert inspect_host_package("pi", "/bin/pi") == PackageState("0.34.0", "npm")


def test_pi_local_package_is_identified_by_manifest_name(tmp_path) -> None:
    extension = tmp_path / "extension"
    extension.mkdir()
    (extension / "package.json").write_text(
        json.dumps({"name": PACKAGE_NAME, "version": "0.34.0"}),
        encoding="utf-8",
    )
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"packages": [str(extension)]}), encoding="utf-8")

    assert _inspect_pi_settings(settings) == PackageState("0.34.0", "local")


def test_malformed_pi_settings_are_actionable(tmp_path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{", encoding="utf-8")

    with pytest.raises(click.ClickException, match="settings.json"):
        _inspect_pi_settings(settings)


def test_omp_list_uses_name_version_and_enabled(monkeypatch) -> None:
    payload = {
        "npm": [
            {"name": PACKAGE_NAME, "version": "0.34.0", "enabled": True},
            {"name": "other", "version": "9.9.9", "enabled": True},
        ],
        "marketplace": [],
    }
    monkeypatch.setattr(
        module,
        "run",
        lambda command, **kwargs: CompletedProcess(command, 0, json.dumps(payload), ""),
    )

    assert inspect_host_package("omp", "/bin/omp") == PackageState("0.34.0", "npm")


def test_disabled_omp_package_is_not_compatible(monkeypatch) -> None:
    payload = {
        "npm": [{"name": PACKAGE_NAME, "version": "0.34.0", "enabled": False}],
        "marketplace": [],
    }
    monkeypatch.setattr(
        module,
        "run",
        lambda command, **kwargs: CompletedProcess(command, 0, json.dumps(payload), ""),
    )

    assert inspect_host_package("omp", "/bin/omp") is None


def test_malformed_omp_json_is_actionable(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "run",
        lambda command, **kwargs: CompletedProcess(command, 0, "{", ""),
    )

    with pytest.raises(click.ClickException, match="plugin list"):
        inspect_host_package("omp", "/bin/omp")


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        (
            "pi",
            [
                "/bin/pi",
                "install",
                "npm:headroom-pi@0.34.0",
            ],
        ),
        (
            "omp",
            [
                "/bin/omp",
                "plugin",
                "install",
                "headroom-pi@0.34.0",
                "--json",
            ],
        ),
    ],
)
def test_missing_package_is_installed_at_exact_version(monkeypatch, host, expected) -> None:
    states = iter([None, PackageState("0.34.0", "npm")])
    monkeypatch.setattr(module, "inspect_host_package", lambda *_: next(states))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        module,
        "run",
        lambda command, **kwargs: successful(calls, command, "{}"),
    )

    artifact = ensure_host_package(host, f"/bin/{host}", "0.34.0", None)

    assert calls == [expected]
    assert artifact.metadata["owned"] is True


def test_compatible_preexisting_package_remains_user_owned(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "inspect_host_package",
        lambda *_: PackageState("0.34.0", "local"),
    )

    artifact = ensure_host_package("pi", "/bin/pi", "0.34.0", None)

    assert artifact.metadata["owned"] is False


def test_incompatible_preexisting_package_is_not_overwritten(monkeypatch) -> None:
    monkeypatch.setattr(
        module,
        "inspect_host_package",
        lambda *_: PackageState("0.33.0", "npm"),
    )

    with pytest.raises(click.ClickException, match="pre-existing"):
        ensure_host_package("omp", "/bin/omp", "0.34.0", None)


@pytest.mark.parametrize("previous", ["0.33.0", "0.35.0"])
def test_owned_package_is_moved_to_exact_requested_version(monkeypatch, previous) -> None:
    states = iter([PackageState(previous, "npm"), PackageState("0.34.0", "npm")])
    monkeypatch.setattr(module, "inspect_host_package", lambda *_: next(states))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        module,
        "run",
        lambda command, **kwargs: successful(calls, command, "{}"),
    )

    artifact = ensure_host_package(
        "omp",
        "/bin/omp",
        "0.34.0",
        package_artifact("omp", previous, owned=True),
    )

    assert calls[0][-2] == "headroom-pi@0.34.0"
    assert artifact.metadata["version"] == "0.34.0"


def test_nonzero_install_is_rolled_back_even_when_registration_succeeds(
    monkeypatch,
) -> None:
    states = iter([None, PackageState("0.34.0", "npm"), None])
    monkeypatch.setattr(module, "inspect_host_package", lambda *_: next(states))
    calls: list[list[str]] = []

    def fail_install(command, **kwargs):
        if not calls:
            return failed(calls, command)
        return successful(calls, command)

    monkeypatch.setattr(module, "run", fail_install)

    with pytest.raises(click.ClickException, match="install pi package") as exc_info:
        ensure_host_package("pi", "/bin/pi", "0.34.0", None)

    assert "Rollback also failed" not in str(exc_info.value)
    assert len(calls) == 2
    with pytest.raises(StopIteration):
        next(states)


def test_failed_new_install_is_removed(monkeypatch) -> None:
    states = iter([None, None, None])
    monkeypatch.setattr(module, "inspect_host_package", lambda *_: next(states))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        module,
        "run",
        lambda command, **kwargs: successful(calls, command, "{}"),
    )

    with pytest.raises(click.ClickException, match="verify"):
        ensure_host_package("pi", "/bin/pi", "0.34.0", None)

    assert calls == [
        [
            "/bin/pi",
            "install",
            "npm:headroom-pi@0.34.0",
        ],
        ["/bin/pi", "remove", "npm:headroom-pi"],
    ]
    with pytest.raises(StopIteration):
        next(states)


def test_nonzero_new_install_and_rollback_report_both_failures(monkeypatch) -> None:
    states = iter([None, PackageState("0.34.0", "npm"), PackageState("0.34.0", "npm")])
    monkeypatch.setattr(module, "inspect_host_package", lambda *_: next(states))
    calls: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda command, **kwargs: failed(calls, command))

    with pytest.raises(click.ClickException) as exc_info:
        ensure_host_package("pi", "/bin/pi", "0.34.0", None)

    message = str(exc_info.value)
    assert "install pi package" in message
    assert "Rollback also failed" in message
    assert "roll back pi package install" in message
    assert "verify rollback" in message
    assert len(calls) == 2
    with pytest.raises(StopIteration):
        next(states)


def test_nonzero_new_install_with_verified_nonzero_rollback_reports_original_only(
    monkeypatch,
) -> None:
    states = iter([None, PackageState("0.34.0", "npm"), None])
    monkeypatch.setattr(module, "inspect_host_package", lambda *_: next(states))
    calls: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda command, **kwargs: failed(calls, command))

    with pytest.raises(click.ClickException, match="install pi package") as exc_info:
        ensure_host_package("pi", "/bin/pi", "0.34.0", None)

    assert "Rollback also failed" not in str(exc_info.value)
    assert len(calls) == 2
    with pytest.raises(StopIteration):
        next(states)


def test_failed_owned_upgrade_restores_previous_version(monkeypatch) -> None:
    states = iter(
        [
            PackageState("0.33.0", "npm"),
            PackageState("0.35.0", "npm"),
            PackageState("0.33.0", "npm"),
        ]
    )
    monkeypatch.setattr(module, "inspect_host_package", lambda *_: next(states))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        module,
        "run",
        lambda command, **kwargs: successful(calls, command, "{}"),
    )

    with pytest.raises(click.ClickException, match="verify"):
        ensure_host_package(
            "omp",
            "/bin/omp",
            "0.34.0",
            package_artifact("omp", "0.33.0", owned=True),
        )

    assert calls == [
        [
            "/bin/omp",
            "plugin",
            "install",
            "headroom-pi@0.34.0",
            "--json",
        ],
        [
            "/bin/omp",
            "plugin",
            "install",
            "headroom-pi@0.33.0",
            "--json",
        ],
    ]


def test_nonzero_upgrade_restores_previous_version_and_reports_original_only(
    monkeypatch,
) -> None:
    states = iter(
        [
            PackageState("0.33.0", "npm"),
            PackageState("0.35.0", "npm"),
            PackageState("0.33.0", "npm"),
        ]
    )
    monkeypatch.setattr(module, "inspect_host_package", lambda *_: next(states))
    calls: list[list[str]] = []
    results = iter(
        [
            CompletedProcess([], 1, "", "upgrade failed"),
            CompletedProcess([], 1, "", "rollback failed"),
        ]
    )

    def run_next(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append(command)
        result = next(results)
        return CompletedProcess(command, result.returncode, result.stdout, result.stderr)

    monkeypatch.setattr(module, "run", run_next)

    with pytest.raises(click.ClickException, match="install omp package") as exc_info:
        ensure_host_package(
            "omp",
            "/bin/omp",
            "0.34.0",
            package_artifact("omp", "0.33.0", owned=True),
        )

    assert "Rollback also failed" not in str(exc_info.value)
    assert len(calls) == 2
    with pytest.raises(StopIteration):
        next(states)


def test_remove_preserves_user_owned_and_changed_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_owned = package_artifact("pi", "0.34.0", owned=False)
    assert remove_owned_host_package("pi", "/bin/pi", user_owned) == "preserved"

    owned = package_artifact("pi", "0.34.0", owned=True)
    monkeypatch.setattr(
        module,
        "inspect_host_package",
        lambda *_: PackageState("0.35.0", "npm"),
    )
    assert remove_owned_host_package("pi", "/bin/pi", owned) == "preserved"


def test_nonzero_remove_is_accepted_when_verification_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter([PackageState("0.34.0", "npm"), None])
    monkeypatch.setattr(module, "inspect_host_package", lambda *_: next(states))
    calls: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda command, **kwargs: failed(calls, command))

    result = remove_owned_host_package(
        "pi", "/bin/pi", package_artifact("pi", "0.34.0", owned=True)
    )

    assert result == "removed"
    assert len(calls) == 1
    with pytest.raises(StopIteration):
        next(states)


def test_nonzero_remove_reports_command_and_verification_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = PackageState("0.34.0", "npm")
    states = iter([state, state])
    monkeypatch.setattr(module, "inspect_host_package", lambda *_: next(states))
    calls: list[list[str]] = []
    monkeypatch.setattr(module, "run", lambda command, **kwargs: failed(calls, command))

    with pytest.raises(click.ClickException) as exc_info:
        remove_owned_host_package("pi", "/bin/pi", package_artifact("pi", "0.34.0", owned=True))

    message = str(exc_info.value)
    assert "remove owned pi package" in message
    assert "verify removal" in message
    with pytest.raises(StopIteration):
        next(states)


@pytest.mark.parametrize(
    ("host", "command"),
    [
        ("pi", ["/bin/pi", "remove", "npm:headroom-pi"]),
        (
            "omp",
            [
                "/bin/omp",
                "plugin",
                "uninstall",
                "headroom-pi",
                "--json",
            ],
        ),
    ],
)
def test_remove_owned_package_uses_host_native_command(
    monkeypatch: pytest.MonkeyPatch,
    host: Literal["pi", "omp"],
    command: list[str],
) -> None:
    states = iter([PackageState("0.34.0", "npm"), None])
    monkeypatch.setattr(module, "inspect_host_package", lambda *_: next(states))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        module,
        "run",
        lambda actual, **kwargs: successful(calls, actual, "{}"),
    )

    result = remove_owned_host_package(
        host,
        f"/bin/{host}",
        package_artifact(host, "0.34.0", owned=True),
    )

    assert result == "removed"
    assert calls == [command]


def test_extension_config_path_matches_runtime() -> None:
    assert extension_config_path() == Path.home() / ".headroom/integrations/pi-extension.json"


def test_config_merge_preserves_unrelated_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    path.write_text(
        '{\n  "enabled": false,\n  "minResultChars": 9000\n}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "extension_config_path", lambda: path)

    artifact = ensure_extension_config(9444, None)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {
        "enabled": False,
        "minResultChars": 9000,
        "baseUrl": "http://127.0.0.1:9444",
    }
    assert artifact.metadata["previous_exists"] is True
    assert artifact.kind == "pi-extension-config"
    assert artifact.path == str(path)


def test_config_remove_restores_exact_prior_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    before = b'{"enabled":false}\n'
    path.write_bytes(before)
    monkeypatch.setattr(module, "extension_config_path", lambda: path)

    artifact = ensure_extension_config(8787, None)

    assert artifact.metadata["previous_base64"] == base64.b64encode(before).decode()
    assert artifact.metadata["managed_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert remove_owned_extension_config(artifact) == "restored"
    assert path.read_bytes() == before


def test_config_remove_preserves_user_edits_after_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    path.write_text(
        '{"baseUrl":"http://127.0.0.1:8787","enabled":false}\n',
        encoding="utf-8",
    )
    edited = path.read_bytes()

    assert ensure_extension_config(8787, artifact) == artifact
    assert path.read_bytes() == edited
    assert remove_owned_extension_config(artifact) == "preserved"
    assert json.loads(path.read_text())["enabled"] is False


def test_new_config_is_removed_when_still_managed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)

    artifact = ensure_extension_config(8787, None)

    assert remove_owned_extension_config(artifact) == "removed"
    assert not path.exists()


def test_empty_config_retains_exact_prior_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    path.write_bytes(b"")
    monkeypatch.setattr(module, "extension_config_path", lambda: path)

    artifact = ensure_extension_config(8787, None)

    assert remove_owned_extension_config(artifact) == "restored"
    assert path.read_bytes() == b""


@pytest.mark.parametrize("content", ["{", "[]", "null"])
def test_malformed_and_non_object_config_are_not_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    path = tmp_path / "pi-extension.json"
    before = content.encode()
    path.write_bytes(before)
    monkeypatch.setattr(module, "extension_config_path", lambda: path)

    with pytest.raises(click.ClickException, match="pi-extension.json"):
        ensure_extension_config(8787, None)

    assert path.read_bytes() == before


def test_repeated_init_is_byte_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    before = path.read_bytes()

    repeated_artifact = ensure_extension_config(8787, artifact)

    assert path.read_bytes() == before
    assert repeated_artifact == artifact


def test_port_change_rewrites_only_unchanged_managed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)

    updated_artifact = ensure_extension_config(9444, artifact)

    assert json.loads(path.read_bytes())["baseUrl"] == "http://127.0.0.1:9444"
    assert updated_artifact.metadata["previous_base64"] == artifact.metadata["previous_base64"]
    assert updated_artifact.metadata["managed_sha256"] != artifact.metadata["managed_sha256"]


def test_conflicting_user_base_url_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    before = b'{"baseUrl":"http://127.0.0.1:9999","enabled":false}\n'
    path.write_bytes(before)

    with pytest.raises(click.ClickException, match="Refusing to overwrite"):
        ensure_extension_config(9444, artifact)

    assert path.read_bytes() == before


def test_config_create_preserves_file_created_during_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    user_bytes = b'{"enabled":false}\n'
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    publish = module._publish_staged

    def create_before_publish(staged: Path, destination: Path) -> bool:
        destination.write_bytes(user_bytes)
        return publish(staged, destination)

    monkeypatch.setattr(module, "_publish_staged", create_before_publish)

    with pytest.raises(click.ClickException, match="changed during configuration"):
        ensure_extension_config(8787, None)

    assert path.read_bytes() == user_bytes


def test_config_update_preserves_file_changed_during_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    user_bytes = b'{"baseUrl":"http://127.0.0.1:9999"}\n'
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    displace = module._displace_candidate

    def change_before_displace(candidate: Path, displaced: Path) -> None:
        candidate.write_bytes(user_bytes)
        displace(candidate, displaced)

    monkeypatch.setattr(module, "_displace_candidate", change_before_displace)

    with pytest.raises(click.ClickException, match="changed during configuration"):
        ensure_extension_config(9444, artifact)

    assert path.read_bytes() == user_bytes


def test_config_restore_preserves_file_changed_during_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    path.write_bytes(b'{"enabled":false}\n')
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    user_bytes = b'{"baseUrl":"http://127.0.0.1:9999"}\n'
    displace = module._displace_candidate

    def change_before_displace(candidate: Path, displaced: Path) -> None:
        candidate.write_bytes(user_bytes)
        displace(candidate, displaced)

    monkeypatch.setattr(module, "_displace_candidate", change_before_displace)

    assert remove_owned_extension_config(artifact) == "preserved"
    assert path.read_bytes() == user_bytes


def test_config_remove_preserves_file_changed_during_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    user_bytes = b'{"enabled":false}\n'
    displace = module._displace_candidate

    def change_before_displace(candidate: Path, displaced: Path) -> None:
        candidate.write_bytes(user_bytes)
        displace(candidate, displaced)

    monkeypatch.setattr(module, "_displace_candidate", change_before_displace)

    assert remove_owned_extension_config(artifact) == "preserved"
    assert path.read_bytes() == user_bytes


def test_config_remove_preserves_file_created_after_displacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    user_bytes = b'{"enabled":false}\n'
    displace = module._displace_candidate

    def create_after_displace(candidate: Path, displaced: Path) -> None:
        displace(candidate, displaced)
        candidate.write_bytes(user_bytes)

    monkeypatch.setattr(module, "_displace_candidate", create_after_displace)

    assert remove_owned_extension_config(artifact) == "removed"
    assert path.read_bytes() == user_bytes


def test_config_remove_does_not_unlink_competitor_after_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    user_bytes = b'{"enabled":false}\n'
    remove_displaced = module._remove_displaced

    def create_before_cleanup(displaced: Path) -> None:
        path.write_bytes(user_bytes)
        remove_displaced(displaced)

    monkeypatch.setattr(module, "_remove_displaced", create_before_cleanup)

    assert remove_owned_extension_config(artifact) == "removed"
    assert path.read_bytes() == user_bytes


def test_config_publish_collision_retains_named_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    before = path.read_bytes()
    user_bytes = b'{"baseUrl":"http://127.0.0.1:9999"}\n'
    publish = module._publish_staged

    def collide_publish(staged: Path, destination: Path) -> bool:
        destination.write_bytes(user_bytes)
        return publish(staged, destination)

    monkeypatch.setattr(module, "_publish_staged", collide_publish)

    with pytest.raises(click.ClickException, match=r"recovery path (.+\.recovery)") as exc_info:
        ensure_extension_config(9444, artifact)

    recovery = Path(str(exc_info.value).split("recovery path ", 1)[1].split(";", 1)[0])
    assert path.read_bytes() == user_bytes
    assert recovery.read_bytes() == before
    assert not list(tmp_path.glob("*.stage"))


def test_config_verification_error_restores_displaced_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    before = path.read_bytes()
    candidate_matches = module._candidate_matches

    def fail_verification(candidate: Path, expected: object) -> bool:
        raise OSError("injected verification failure")

    monkeypatch.setattr(module, "_candidate_matches", fail_verification)

    with pytest.raises(click.ClickException, match="injected verification failure"):
        ensure_extension_config(9444, artifact)

    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.stage"))
    assert not list(tmp_path.glob("*.recovery"))
    monkeypatch.setattr(module, "_candidate_matches", candidate_matches)


def test_config_verification_and_recovery_error_retains_named_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    before = path.read_bytes()

    def fail_verification(candidate: Path, expected: object) -> bool:
        path.write_bytes(b'{"enabled":false}\n')
        raise OSError("injected verification failure")

    monkeypatch.setattr(module, "_candidate_matches", fail_verification)

    with pytest.raises(click.ClickException, match=r"recovery path (.+\.recovery)") as exc_info:
        ensure_extension_config(9444, artifact)

    recovery = Path(str(exc_info.value).split("recovery path ", 1)[1].split(";", 1)[0])
    assert path.read_bytes() == b'{"enabled":false}\n'
    assert recovery.read_bytes() == before
    assert not list(tmp_path.glob("*.stage"))


def test_config_publish_error_restores_displaced_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    before = path.read_bytes()

    def fail_publish(staged: Path, destination: Path) -> bool:
        raise OSError("injected publish failure")

    monkeypatch.setattr(module, "_publish_staged", fail_publish)

    with pytest.raises(click.ClickException, match="injected publish failure"):
        ensure_extension_config(9444, artifact)

    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.recovery"))


def test_config_publish_and_recovery_error_retains_named_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    before = path.read_bytes()

    def fail_publish(staged: Path, destination: Path) -> bool:
        raise OSError("injected publish failure")

    def fail_recovery(source: Path, destination: Path) -> None:
        raise OSError("injected recovery failure")

    monkeypatch.setattr(module, "_publish_staged", fail_publish)
    monkeypatch.setattr(module.os, "link", fail_recovery)

    with pytest.raises(click.ClickException, match=r"recovery path (.+\.recovery)") as exc_info:
        ensure_extension_config(9444, artifact)

    recovery = Path(str(exc_info.value).split("recovery path ", 1)[1].split(";", 1)[0])
    assert recovery.read_bytes() == before
    assert not path.exists()


@pytest.mark.parametrize("operation", ["ensure", "remove"])
def test_symlink_config_is_rejected_without_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b'{"enabled":false}\n')
    path = tmp_path / "pi-extension.json"
    path.symlink_to(target)
    monkeypatch.setattr(module, "extension_config_path", lambda: path)

    with pytest.raises(click.ClickException, match="symbolic link"):
        if operation == "ensure":
            ensure_extension_config(8787, None)
        else:
            remove_owned_extension_config(
                ArtifactRecord(
                    kind="pi-extension-config",
                    path=str(path),
                    metadata={
                        "previous_exists": False,
                        "previous_base64": "",
                        "managed_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    },
                )
            )

    assert path.is_symlink()
    assert target.read_bytes() == b'{"enabled":false}\n'


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_config_update_and_restore_preserve_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    path.write_bytes(b'{"enabled":false}\n')
    path.chmod(0o640)
    monkeypatch.setattr(module, "extension_config_path", lambda: path)

    artifact = ensure_extension_config(8787, None)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640

    updated = ensure_extension_config(9444, artifact)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640

    assert remove_owned_extension_config(updated) == "restored"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_config_remove_reports_absent_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    artifact = ensure_extension_config(8787, None)
    path.unlink()

    assert remove_owned_extension_config(artifact) == "absent"


def test_config_lifecycle_has_no_omp_models_path_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pi-extension.json"
    monkeypatch.setattr(module, "extension_config_path", lambda: path)
    original_read_bytes = module.Path.read_bytes
    original_write_bytes = module.Path.write_bytes

    def guarded_read_bytes(candidate: Path) -> bytes:
        assert candidate.name != "models.yml"
        return original_read_bytes(candidate)

    def guarded_write_bytes(candidate: Path, content: bytes) -> int:
        assert candidate.name != "models.yml"
        return original_write_bytes(candidate, content)

    monkeypatch.setattr(module.Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(module.Path, "write_bytes", guarded_write_bytes)

    artifact = ensure_extension_config(8787, None)
    assert remove_owned_extension_config(artifact) == "removed"
