"""Models used by the install / deployment subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class InstallPreset(str, Enum):
    """User-facing persistent runtime presets."""

    PERSISTENT_SERVICE = "persistent-service"
    PERSISTENT_TASK = "persistent-task"
    PERSISTENT_DOCKER = "persistent-docker"


class RuntimeKind(str, Enum):
    """Runtime used to execute Headroom."""

    PYTHON = "python"
    DOCKER = "docker"


class SupervisorKind(str, Enum):
    """How a persistent deployment is kept alive."""

    SERVICE = "service"
    TASK = "task"
    NONE = "none"


class ProviderSelectionMode(str, Enum):
    """How tool targets are selected for configuration."""

    AUTO = "auto"
    ALL = "all"
    MANUAL = "manual"


class ConfigScope(str, Enum):
    """Where persistent configuration should be applied."""

    PROVIDER = "provider"
    USER = "user"
    SYSTEM = "system"


class ToolTarget(str, Enum):
    """Supported tool targets for persistent proxy wiring."""

    CLAUDE = "claude"
    COPILOT = "copilot"
    CODEX = "codex"
    AIDER = "aider"
    CURSOR = "cursor"
    GROK_BUILD = "grok_build"
    OPENCLAW = "openclaw"
    OPENCODE = "opencode"


def iso_utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class ManagedMutation:
    """A reversible change applied by `headroom install`."""

    target: str
    kind: str
    path: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactRecord:
    """A rendered file or platform object owned by the deployment."""

    kind: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentManifest:
    """Persisted deployment state for a named profile."""

    profile: str
    preset: str
    runtime_kind: str
    supervisor_kind: str
    scope: str
    provider_mode: str
    targets: list[str]
    port: int
    host: str
    backend: str
    anyllm_provider: str | None = None
    region: str | None = None
    proxy_mode: str = "token"
    memory_enabled: bool = False
    memory_db_path: str = ""
    telemetry_enabled: bool = True
    image: str = "ghcr.io/chopratejas/headroom:latest"
    service_name: str = "headroom"
    container_name: str = "headroom-persistent"
    health_url: str = "http://127.0.0.1:8787/readyz"
    base_env: dict[str, str] = field(default_factory=dict)
    tool_envs: dict[str, dict[str, str]] = field(default_factory=dict)
    proxy_args: list[str] = field(default_factory=list)
    mutations: list[ManagedMutation] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    created_at: str = field(default_factory=iso_utc_now)
    updated_at: str = field(default_factory=iso_utc_now)
