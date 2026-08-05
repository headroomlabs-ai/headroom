"""Runtime helpers for Oh My Pi (omp) integrations.

omp resolves its Anthropic chat endpoint from the model registry
(``providers.anthropic.baseUrl`` in ``~/.omp/agent/models.yml``), not from
``ANTHROPIC_BASE_URL`` — that env var only feeds omp's web-search helper.
Verified empirically: with ``ANTHROPIC_BASE_URL`` pointed at a local probe
server, omp's chat traffic still went to the real Anthropic endpoint; with a
``models.yml`` same-ID override, every ``/v1/messages`` request arrived at the
probe.  A same-ID override keeps omp's bundled Anthropic model catalog and
stored credentials (both keyed by provider id ``anthropic``), so only the
endpoint moves.

The wrap therefore injects a marker-fenced ``providers.anthropic.baseUrl``
override into ``models.yml``, snapshotting the pre-wrap file byte-for-byte
first — the same durable-wrap + backup + ``headroom unwrap`` contract the
Codex wrap uses for ``config.toml``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from headroom.providers.claude import proxy_base_url as claude_proxy_base_url
from headroom.proxy.project_context import with_project_prefix

MANAGED_MARKER = "# managed by `headroom wrap omp`"
_MANAGED_HEADER = (
    f"{MANAGED_MARKER} — do not hand-edit while wrapped.\n"
    "# `headroom unwrap omp` restores the pre-wrap file (or removes this one\n"
    "# if it did not exist). The original is kept at <models.yml.headroom-backup>.\n"
)
BACKUP_SUFFIX = ".headroom-backup"


def models_yml_path() -> Path:
    """Path to omp's model/provider registry file.

    ``PI_CODING_AGENT_DIR`` relocates omp's ``~/.omp/agent`` state directory
    (per omp's environment-variable reference); ``models.yml`` moves with it.
    """
    base = os.environ.get("PI_CODING_AGENT_DIR", "").strip()
    agent_dir = Path(base).expanduser() if base else Path.home() / ".omp" / "agent"
    return agent_dir / "models.yml"


def backup_path(models_file: Path) -> Path:
    """Backup location for the pre-wrap ``models.yml`` snapshot."""
    return models_file.with_name(models_file.name + BACKUP_SUFFIX)


def proxy_anthropic_base_url(port: int, project: str | None = None) -> str:
    """Proxy base URL omp's ``anthropic`` provider is pointed at.

    ``project`` (the wrap launch directory) is encoded as a ``/p/<name>``
    base-URL prefix — same as the Aider wrap — so the proxy attributes
    savings per project.  omp appends ``/v1/messages`` after any path segments
    (its docs show path-carrying Anthropic base URLs, e.g. the Cloudflare AI
    Gateway example), and the proxy strips the prefix on arrival.
    """
    return with_project_prefix(claude_proxy_base_url(port), project)


def is_managed(models_file: Path) -> bool:
    """Whether ``models_file`` is currently a wrap-managed override."""
    if not models_file.exists():
        return False
    try:
        head = models_file.read_bytes()
    except OSError:
        return False
    return MANAGED_MARKER.encode("utf-8") in head


def inject_models_override(port: int, project: str | None = None) -> tuple[Path, str]:
    """Point ``providers.anthropic.baseUrl`` at the local proxy.

    Returns ``(models_file, base_url)``.

    * First injection snapshots the user's pre-wrap file byte-for-byte to
      ``models.yml.headroom-backup`` so unwrap can restore it exactly.  A file
      that is already wrap-managed is never re-snapshotted (that would clobber
      the pristine backup — same guard as the Codex config snapshot).
    * The managed file is regenerated from the backup (or from scratch when
      the user had no ``models.yml``) on every call, so re-running with a
      different ``--port`` updates the override idempotently.
    * Any user-defined providers/models from the pre-wrap file are preserved:
      the override only deep-sets ``providers.anthropic.baseUrl``.
    """
    import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs; lint env installs no deps

    models_file = models_yml_path()
    backup = backup_path(models_file)
    base_url = proxy_anthropic_base_url(port, project)

    original_bytes: bytes | None = None
    if backup.exists():
        original_bytes = backup.read_bytes()
    elif models_file.exists():
        if is_managed(models_file):
            # Managed file without a backup: we created it from scratch —
            # regenerate from scratch rather than merging our own output.
            original_bytes = None
        else:
            # Snapshot as raw bytes so the unwrap restore is byte-for-byte
            # even for non-UTF-8 or newline-sensitive files (text-mode I/O
            # would translate newlines on Windows).
            backup.parent.mkdir(parents=True, exist_ok=True)
            original_bytes = models_file.read_bytes()
            backup.write_bytes(original_bytes)

    data: dict = {}
    if original_bytes:
        loaded = yaml.safe_load(original_bytes.decode("utf-8", errors="replace"))
        if isinstance(loaded, dict):
            data = loaded
    providers = data.setdefault("providers", {})
    if not isinstance(providers, dict):  # malformed user value: keep it in backup only
        providers = {}
        data["providers"] = providers
    anthropic = providers.setdefault("anthropic", {})
    if not isinstance(anthropic, dict):
        anthropic = {}
        providers["anthropic"] = anthropic
    anthropic["baseUrl"] = base_url

    models_file.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    models_file.write_text(_MANAGED_HEADER + rendered, encoding="utf-8", newline="\n")
    return models_file, base_url


def restore_models_override() -> str:
    """Undo :func:`inject_models_override`.

    Returns one of ``"restored"`` (backup moved back), ``"removed"``
    (wrap-created file deleted), or ``"noop"`` (nothing wrap-managed found).
    Never touches a ``models.yml`` the wrap does not manage.
    """
    models_file = models_yml_path()
    backup = backup_path(models_file)

    if backup.exists():
        backup.replace(models_file)
        return "restored"
    if is_managed(models_file):
        models_file.unlink()
        return "removed"
    return "noop"


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Build the launch environment and display lines for the omp wrap.

    The endpoint redirect itself lives in ``models.yml`` (see module
    docstring), so the environment passes through unchanged; the display
    lines surface where the override went.
    """
    env = dict(environ or os.environ)
    base_url = proxy_anthropic_base_url(port, project)
    return env, [f"models.yml: providers.anthropic.baseUrl={base_url}"]
