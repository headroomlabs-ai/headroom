"""Tool-target configuration for persistent deployments."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from headroom import fsutil
from headroom._subprocess import run
from headroom.providers.install_registry import (
    apply_provider_scope_mutations,
    revert_provider_scope_mutation,
)

from .models import ConfigScope, DeploymentManifest, ManagedMutation
from .paths import (
    unix_system_env_targets,
    unix_user_env_targets,
)

_ENV_MARKER_START = "# >>> headroom persistent env >>>"
_ENV_MARKER_END = "# <<< headroom persistent env <<<"
_ENV_PATTERN = re.compile(
    re.escape(_ENV_MARKER_START) + r".*?" + re.escape(_ENV_MARKER_END),
    re.DOTALL,
)


def _merge_marker_block(file_path: Path, block: str, pattern: re.Pattern[str], marker: str) -> str:
    if file_path.exists():
        existing = fsutil.read_text(file_path)
        if marker in existing:
            return pattern.sub(block, existing)
        return existing.rstrip() + "\n\n" + block + "\n"
    return block + "\n"


def _env_block(values: dict[str, str]) -> str:
    lines = [_ENV_MARKER_START]
    for name, value in values.items():
        lines.append(f'export {name}="{value}"')
    lines.append(_ENV_MARKER_END)
    return "\n".join(lines)


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _unix_scope_values(manifest: DeploymentManifest) -> dict[str, str]:
    merged = dict(manifest.base_env)
    for env_map in manifest.tool_envs.values():
        merged.update(env_map)
    return merged


def _apply_unix_env_scope(manifest: DeploymentManifest) -> list[ManagedMutation]:
    values = _unix_scope_values(manifest)
    block = _env_block(values)
    if manifest.scope == ConfigScope.USER.value:
        targets = unix_user_env_targets()
    else:
        targets = unix_system_env_targets()
    mutations: list[ManagedMutation] = []
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = _merge_marker_block(path, block, _ENV_PATTERN, _ENV_MARKER_START)
        fsutil.write_text(path, merged)
        mutations.append(ManagedMutation(target="env", kind="shell-block", path=str(path)))
    return mutations


def _remove_unix_env_scope(mutations: list[ManagedMutation]) -> None:
    for mutation in mutations:
        if mutation.kind != "shell-block" or not mutation.path:
            continue
        path = Path(mutation.path)
        if not path.exists():
            continue
        content = fsutil.read_text(path)
        if _ENV_MARKER_START not in content:
            continue
        fsutil.write_text(path, _ENV_PATTERN.sub("", content).strip() + "\n")


def _apply_windows_env_scope(manifest: DeploymentManifest) -> list[ManagedMutation]:
    scope_name = "Machine" if manifest.scope == ConfigScope.SYSTEM.value else "User"
    merged = _unix_scope_values(manifest)
    mutations: list[ManagedMutation] = []
    for name, value in merged.items():
        previous = run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"$value = [Environment]::GetEnvironmentVariable({_powershell_literal(name)},{_powershell_literal(scope_name)}); "
                "if ($null -eq $value) { '__HEADROOM_UNSET__' } else { $value }",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"[Environment]::SetEnvironmentVariable({_powershell_literal(name)},{_powershell_literal(value)},{_powershell_literal(scope_name)})",
        ]
        subprocess.run(command, check=True)
        mutations.append(
            ManagedMutation(
                target="env",
                kind="windows-env",
                data={
                    "name": name,
                    "scope": scope_name,
                    "previous": None if previous == "__HEADROOM_UNSET__" else previous,
                },
            )
        )
    return mutations


def _remove_windows_env_scope(mutations: list[ManagedMutation]) -> None:
    for mutation in mutations:
        if mutation.kind != "windows-env":
            continue
        name = mutation.data.get("name")
        if not isinstance(name, str):
            raise ValueError("Windows environment mutation is missing a variable name")
        scope_name = mutation.data.get("scope", "User")
        if not isinstance(scope_name, str):
            raise ValueError("Windows environment mutation is missing a valid scope")
        previous = mutation.data.get("previous")
        if previous is None:
            value_literal = "$null"
        else:
            value_literal = _powershell_literal(previous)
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"[Environment]::SetEnvironmentVariable({_powershell_literal(name)},{value_literal},{_powershell_literal(scope_name)})",
        ]
        subprocess.run(command, check=True)


def apply_mutations(manifest: DeploymentManifest) -> list[ManagedMutation]:
    """Apply provider/user/system configuration for a deployment."""

    mutations: list[ManagedMutation] = []
    if manifest.scope in {ConfigScope.USER.value, ConfigScope.SYSTEM.value}:
        if os.name == "nt":
            mutations.extend(_apply_windows_env_scope(manifest))
        else:
            mutations.extend(_apply_unix_env_scope(manifest))
        mutations.extend(apply_provider_scope_mutations(manifest))
        return mutations

    return [*mutations, *apply_provider_scope_mutations(manifest)]


def revert_mutations(manifest: DeploymentManifest) -> None:
    """Undo the stored mutations for a deployment."""

    if manifest.scope in {ConfigScope.USER.value, ConfigScope.SYSTEM.value}:
        shell_mutations = [m for m in manifest.mutations if m.target == "env"]
        if os.name == "nt":
            _remove_windows_env_scope(shell_mutations)
        else:
            _remove_unix_env_scope(shell_mutations)

    for mutation in manifest.mutations:
        revert_provider_scope_mutation(manifest, mutation)
