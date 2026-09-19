"""Ownership-safe lifecycle for the shared Pi/OMP extension package."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Literal, cast

import click

from headroom._subprocess import run
from headroom._version import normalize_release_version
from headroom.fsutil import write_text
from headroom.install.models import ArtifactRecord

PACKAGE_NAME = "headroom-pi"
HostName = Literal["pi", "omp"]


@dataclass(frozen=True)
class PackageState:
    version: str
    source: str | None


@dataclass(frozen=True)
class _ConfigCandidate:
    content: bytes
    device: int
    inode: int
    mode: int


def extension_release_version(version_label: str) -> str:
    """Return the exact extension release matching a released Headroom build."""
    version = normalize_release_version(version_label)
    if version is None:
        raise click.ClickException(
            "Durable Pi/OMP init requires a released Headroom version; "
            f"current version is {version_label!r}."
        )
    return version


def _read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Could not read valid {description} at {path}: {exc}") from exc


def extension_config_path() -> Path:
    """Return the shared config path consumed by the Pi/OMP extension."""
    return Path("~/.headroom/integrations/pi-extension.json").expanduser()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@contextmanager
def _extension_config_lock(path: Path) -> Iterator[None]:
    lock_path = path.parent / ".pi-extension.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            msvcrt_any = cast(Any, msvcrt)
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt_any.locking(lock_file.fileno(), msvcrt_any.LK_LOCK, 1)
        else:
            import fcntl

            fcntl_any = cast(Any, fcntl)
            fcntl_any.flock(lock_file.fileno(), fcntl_any.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                msvcrt_any = cast(Any, msvcrt)
                lock_file.seek(0)
                msvcrt_any.locking(lock_file.fileno(), msvcrt_any.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl_any = cast(Any, fcntl)
                fcntl_any.flock(lock_file.fileno(), fcntl_any.LOCK_UN)


def _capture_candidate(path: Path) -> _ConfigCandidate | None:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise click.ClickException(
                f"Refusing to manage Pi extension config symbolic link at {path}."
            )
        content = path.read_bytes()
        after = path.lstat()
    except FileNotFoundError:
        return None
    if (
        (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise click.ClickException(f"Pi extension config at {path} changed while being read.")
    return _ConfigCandidate(content, after.st_dev, after.st_ino, stat.S_IMODE(after.st_mode))


def _stage_bytes(path: Path, content: bytes, suffix: str, mode: int = 0o600) -> Path:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=suffix)
    os.close(fd)
    staged = Path(name)
    try:
        write_text(staged, content.decode("utf-8"))
        staged.chmod(mode)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _publish_staged(staged: Path, destination: Path) -> bool:
    try:
        os.link(staged, destination)
    except FileExistsError:
        return False
    return True


def _displace_candidate(candidate: Path, displaced: Path) -> None:
    os.replace(candidate, displaced)


def _remove_displaced(displaced: Path) -> None:
    try:
        displaced.unlink()
    except OSError as exc:
        raise click.ClickException(
            f"Could not clean up Pi extension config recovery path {displaced}: {exc}"
        ) from exc


def _recover_displaced(displaced: Path, path: Path, cause: BaseException) -> None:
    try:
        os.link(displaced, path)
    except OSError as exc:
        raise click.ClickException(
            f"Pi extension config commit failed: {cause}. The displaced config is "
            f"retained at recovery path {displaced}; recovery failed: {exc}"
        ) from cause
    _remove_displaced(displaced)


def _candidate_matches(path: Path, expected: _ConfigCandidate) -> bool:
    try:
        stat = path.lstat()
        content = path.read_bytes()
    except FileNotFoundError:
        return False
    return (
        stat.st_dev == expected.device
        and stat.st_ino == expected.inode
        and content == expected.content
    )


def _commit_config(path: Path, expected: _ConfigCandidate | None, desired: bytes | None) -> bool:
    mode = expected.mode if expected is not None else 0o600
    staged = _stage_bytes(path, desired, ".stage", mode) if desired is not None else None
    displaced: Path | None = None
    try:
        if expected is None:
            return staged is not None and _publish_staged(staged, path)

        displaced = _stage_bytes(path, b"", ".recovery", expected.mode)
        displaced.unlink()
        try:
            _displace_candidate(path, displaced)
        except FileNotFoundError:
            return False
        try:
            matches = _candidate_matches(displaced, expected)
        except OSError as exc:
            _recover_displaced(displaced, path, exc)
            raise exc
        if not matches:
            _recover_displaced(
                displaced,
                path,
                click.ClickException(f"Pi extension config at {path} changed"),
            )
            return False
        if staged is None:
            _remove_displaced(displaced)
            return True
        try:
            published = _publish_staged(staged, path)
        except OSError as exc:
            _recover_displaced(displaced, path, exc)
            raise exc
        if not published:
            raise click.ClickException(
                f"Pi extension config at {path} changed during configuration; "
                f"the competing file was preserved and prior bytes are retained "
                f"at recovery path {displaced}; manual recovery may be required."
            )
        _remove_displaced(displaced)
        return True
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def _config_payload(content: bytes, path: Path) -> dict[str, Any]:
    if not content:
        return {}
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise click.ClickException(
            f"Could not read valid pi-extension.json at {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"Pi extension config at {path} must contain an object.")
    return payload


def _config_artifact(
    path: Path,
    previous_exists: bool,
    previous: bytes,
    managed: bytes,
) -> ArtifactRecord:
    return ArtifactRecord(
        kind="pi-extension-config",
        path=str(path),
        metadata={
            "previous_exists": previous_exists,
            "previous_base64": base64.b64encode(previous).decode("ascii"),
            "managed_sha256": _sha256(managed),
        },
    )


def _owned_config_metadata(artifact: ArtifactRecord, path: Path) -> tuple[bool, bytes, str]:
    metadata = artifact.metadata
    if artifact.kind != "pi-extension-config" or Path(artifact.path) != path:
        raise click.ClickException(f"Invalid Pi extension config ownership record for {path}.")
    previous_exists = metadata.get("previous_exists")
    previous_base64 = metadata.get("previous_base64")
    managed_sha256 = metadata.get("managed_sha256")
    if (
        not isinstance(previous_exists, bool)
        or not isinstance(previous_base64, str)
        or not isinstance(managed_sha256, str)
    ):
        raise click.ClickException(f"Invalid Pi extension config ownership record for {path}.")
    try:
        previous = base64.b64decode(previous_base64, validate=True)
    except ValueError as exc:
        raise click.ClickException(
            f"Invalid Pi extension config ownership record for {path}."
        ) from exc
    return previous_exists, previous, managed_sha256


def ensure_extension_config(port: int, existing_artifact: ArtifactRecord | None) -> ArtifactRecord:
    """Set the loopback endpoint while preserving exact prior config bytes."""
    path = extension_config_path().absolute()
    requested_url = f"http://127.0.0.1:{port}"

    with _extension_config_lock(path):
        candidate = _capture_candidate(path)
        current = candidate.content if candidate is not None else b""
        if existing_artifact is None:
            previous_exists = candidate is not None
            previous = current
            payload = _config_payload(current, path)
        else:
            previous_exists, previous, managed_sha256 = _owned_config_metadata(
                existing_artifact, path
            )
            if candidate is None:
                raise click.ClickException(
                    f"Refusing to recreate changed Pi extension config at {path}."
                )
            payload = _config_payload(current, path)
            if _sha256(current) != managed_sha256:
                if payload.get("baseUrl") == requested_url:
                    return existing_artifact
                raise click.ClickException(
                    f"Refusing to overwrite changed Pi extension config at {path}; "
                    f"baseUrl does not match {requested_url}."
                )
            if payload.get("baseUrl") == requested_url:
                return existing_artifact

        payload["baseUrl"] = requested_url
        managed = (json.dumps(payload, indent=2) + "\n").encode()
        try:
            committed = _commit_config(path, candidate, managed)
        except OSError as exc:
            raise click.ClickException(
                f"Could not write Pi extension config at {path}: {exc}"
            ) from exc
        if not committed:
            raise click.ClickException(
                f"Pi extension config at {path} changed during configuration; "
                "the competing file was preserved."
            )
        return _config_artifact(path, previous_exists, previous, managed)


def remove_owned_extension_config(
    artifact: ArtifactRecord,
) -> Literal["restored", "removed", "preserved", "absent"]:
    """Undo config ownership only while the managed bytes remain unchanged."""
    path = extension_config_path().absolute()
    previous_exists, previous, managed_sha256 = _owned_config_metadata(artifact, path)
    with _extension_config_lock(path):
        candidate = _capture_candidate(path)
        if candidate is None:
            return "absent"
        if _sha256(candidate.content) != managed_sha256:
            return "preserved"
        try:
            committed = _commit_config(path, candidate, previous if previous_exists else None)
        except (OSError, UnicodeDecodeError) as exc:
            raise click.ClickException(
                f"Could not remove Pi extension config at {path}: {exc}"
            ) from exc
        if not committed:
            return "preserved"
        return "restored" if previous_exists else "removed"


def _npm_state(source: str) -> PackageState | None:
    prefix = f"npm:{PACKAGE_NAME}"
    if source == prefix:
        return PackageState("unknown", "npm")
    version_prefix = f"{prefix}@"
    if source.startswith(version_prefix):
        return PackageState(source.removeprefix(version_prefix) or "unknown", "npm")
    return None


def _local_state(source: str, settings_path: Path) -> PackageState | None:
    package_dir = Path(source).expanduser()
    if not package_dir.is_absolute():
        package_dir = settings_path.parent / package_dir
    manifest_path = package_dir / "package.json"
    if not manifest_path.is_file():
        return None
    payload = _read_json(manifest_path, "local package.json")
    if not isinstance(payload, dict) or payload.get("name") != PACKAGE_NAME:
        return None
    version = payload.get("version")
    return PackageState(version if isinstance(version, str) else "unknown", "local")


def _inspect_pi_settings(path: Path) -> PackageState | None:
    """Inspect Pi's persisted package settings without parsing human CLI output."""
    if not path.exists():
        return None
    payload = _read_json(path, "Pi settings.json")
    if not isinstance(payload, dict):
        raise click.ClickException(f"Pi settings.json at {path} must contain an object.")
    packages = payload.get("packages", [])
    if not isinstance(packages, list):
        raise click.ClickException(f"Pi settings.json at {path} has an invalid packages value.")

    for entry in packages:
        source = (
            entry
            if isinstance(entry, str)
            else entry.get("source")
            if isinstance(entry, dict)
            else None
        )
        if not isinstance(source, str):
            continue
        state = _npm_state(source)
        if state is not None:
            return state
        state = _local_state(source, path)
        if state is not None:
            return state
    return None


def _run(command: list[str], description: str) -> str:
    try:
        result = run(command, capture_output=True, text=True, check=True)
    except (CalledProcessError, OSError) as exc:
        raise click.ClickException(f"Failed to {description}: {exc}") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise click.ClickException(f"Failed to {description}{detail}")
    return cast(str, result.stdout or "")


def inspect_host_package(host: HostName, binary: str) -> PackageState | None:
    """Return the enabled package state persisted by a supported host."""
    if host == "pi":
        config_dir = Path(os.environ.get("PI_CODING_AGENT_DIR", "~/.pi/agent")).expanduser()
        return _inspect_pi_settings(config_dir / "settings.json")

    stdout = _run([binary, "plugin", "list", "--json"], "run OMP plugin list --json")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Could not parse OMP plugin list --json output: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("npm", []), list):
        raise click.ClickException("OMP plugin list --json returned an invalid payload.")
    for entry in payload.get("npm", []):
        if (
            isinstance(entry, dict)
            and entry.get("name") == PACKAGE_NAME
            and entry.get("enabled") is not False
        ):
            version = entry.get("version")
            return PackageState(version if isinstance(version, str) else "unknown", "npm")
    return None


def _install_command(host: HostName, binary: str, version: str) -> list[str]:
    spec = f"{PACKAGE_NAME}@{version}"
    if host == "pi":
        return [binary, "install", f"npm:{spec}"]
    return [binary, "plugin", "install", spec, "--json"]


def _remove_command(host: HostName, binary: str) -> list[str]:
    if host == "pi":
        return [binary, "remove", f"npm:{PACKAGE_NAME}"]
    return [binary, "plugin", "uninstall", PACKAGE_NAME, "--json"]


def _artifact(host: HostName, version: str, *, owned: bool, source: str | None) -> ArtifactRecord:
    return ArtifactRecord(
        kind="pi-extension-package",
        path=host,
        metadata={
            "package": PACKAGE_NAME,
            "version": version,
            "owned": owned,
            "source": source,
        },
    )


def _owns_state(artifact: ArtifactRecord | None, host: HostName, state: PackageState) -> bool:
    return bool(
        artifact is not None
        and artifact.kind == "pi-extension-package"
        and artifact.path == host
        and artifact.metadata.get("package") == PACKAGE_NAME
        and artifact.metadata.get("owned") is True
        and artifact.metadata.get("version") == state.version
        and artifact.metadata.get("source") == state.source
    )


def _mutation_error(command: list[str], description: str) -> click.ClickException | None:
    try:
        _run(command, description)
    except click.ClickException as exc:
        return exc
    return None


def _verified_cleanup(
    command_error: click.ClickException | None,
    verification_error: click.ClickException | None,
) -> None:
    if verification_error is None:
        return
    if command_error is not None:
        raise click.ClickException(f"{command_error} {verification_error}") from verification_error
    raise verification_error


def _rollback_new_install(host: HostName, binary: str) -> None:
    command_error = _mutation_error(
        _remove_command(host, binary), f"roll back {host} package install"
    )
    try:
        remaining = inspect_host_package(host, binary)
        verification_error = (
            click.ClickException(f"Failed to verify rollback of the {host} package install.")
            if remaining is not None
            else None
        )
    except click.ClickException as exc:
        verification_error = click.ClickException(
            f"Failed to verify rollback of the {host} package install: {exc}"
        )
    _verified_cleanup(command_error, verification_error)


def _rollback_owned_upgrade(host: HostName, binary: str, previous: PackageState) -> None:
    command_error = _mutation_error(
        _install_command(host, binary, previous.version),
        f"restore {host} package version {previous.version}",
    )
    try:
        restored = inspect_host_package(host, binary)
        verification_error = (
            click.ClickException(
                f"Failed to verify rollback to {host} package version {previous.version}."
            )
            if restored != previous
            else None
        )
    except click.ClickException as exc:
        verification_error = click.ClickException(
            f"Failed to verify rollback to {host} package version {previous.version}: {exc}"
        )
    _verified_cleanup(command_error, verification_error)


def ensure_host_package(
    host: HostName,
    binary: str,
    version: str,
    existing_artifact: ArtifactRecord | None,
) -> ArtifactRecord:
    """Ensure an exact package version without claiming user-owned installations."""
    version = extension_release_version(version)
    previous = inspect_host_package(host, binary)
    if previous is not None and previous.version == version:
        owned = _owns_state(existing_artifact, host, previous)
        return _artifact(host, version, owned=owned, source=previous.source)
    if previous is not None and not _owns_state(existing_artifact, host, previous):
        raise click.ClickException(
            f"Refusing to overwrite pre-existing {host} package version {previous.version}."
        )

    install_error = _mutation_error(
        _install_command(host, binary, version),
        f"install {host} package version {version}",
    )
    try:
        installed = inspect_host_package(host, binary)
        verification_error = (
            click.ClickException(
                f"Could not verify exact {host} package version {version} after install."
            )
            if installed != PackageState(version, "npm")
            else None
        )
    except click.ClickException as exc:
        verification_error = click.ClickException(
            f"Could not verify exact {host} package version {version} after install: {exc}"
        )
    if install_error is None:
        install_error = verification_error
    elif verification_error is not None:
        install_error = click.ClickException(f"{install_error} {verification_error}")

    if install_error is not None:
        try:
            if previous is None:
                _rollback_new_install(host, binary)
            else:
                _rollback_owned_upgrade(host, binary, previous)
        except click.ClickException as rollback_error:
            raise click.ClickException(
                f"{install_error} Rollback also failed: {rollback_error}"
            ) from rollback_error
        raise install_error

    return _artifact(host, version, owned=True, source="npm")


def remove_owned_host_package(
    host: HostName, binary: str, artifact: ArtifactRecord
) -> Literal["removed", "preserved", "absent"]:
    """Remove a package only while it still matches Headroom's ownership record."""
    if artifact.metadata.get("owned") is not True:
        return "preserved"
    current = inspect_host_package(host, binary)
    if current is None:
        return "absent"
    if not _owns_state(artifact, host, current):
        return "preserved"

    command_error = _mutation_error(_remove_command(host, binary), f"remove owned {host} package")
    try:
        remaining = inspect_host_package(host, binary)
        verification_error = (
            click.ClickException(f"Could not verify removal of the owned {host} package.")
            if remaining is not None
            else None
        )
    except click.ClickException as exc:
        verification_error = click.ClickException(
            f"Could not verify removal of the owned {host} package: {exc}"
        )
    _verified_cleanup(command_error, verification_error)
    return "removed"
