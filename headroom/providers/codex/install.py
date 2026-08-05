"""Codex install-time helpers."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from headroom._subprocess import run
from headroom.install.models import ConfigScope, DeploymentManifest, ManagedMutation, ToolTarget
from headroom.install.paths import codex_config_path

from .runtime import proxy_base_url
from .threads import retag_to_headroom, retag_to_native

_CODEX_MARKER_START = "# --- Headroom persistent provider ---"
_CODEX_MARKER_END = "# --- end Headroom persistent provider ---"
_CODEX_PATTERN = re.compile(
    re.escape(_CODEX_MARKER_START) + r".*?" + re.escape(_CODEX_MARKER_END),
    re.DOTALL,
)

# Orphan-key patterns: strip any top-level keys that a crashed or partial write
# may have left outside the marker block.
_ORPHAN_MODEL_PROVIDER = re.compile(r'(?m)^[ \t]*model_provider[ \t]*=[ \t]*"headroom"[ \t]*\r?\n')
_ORPHAN_OPENAI_BASE_URL = re.compile(
    r'(?m)^[ \t]*openai_base_url[ \t]*=[ \t]*"http://127\.0\.0\.1:\d+/v1"[ \t]*\r?\n'
)
_ORPHAN_HEADROOM_TABLE = re.compile(
    r"(?ms)^\[model_providers\.headroom\][^\[]*?"
    r'base_url[ \t]*=[ \t]*"http://127\.0\.0\.1:\d+/v1"[^\[]*?'
    r"(?=^\[|\Z)"
)

_TOML_TABLE_HEADER_RE = re.compile(r"^[ \t]*(?:\[\[[^\]\r\n]+\]\]|\[[^\]\r\n]+\])[ \t]*(?:#.*)?$")
_ROOT_MODEL_PROVIDER_RE = re.compile(r"^[ \t]*model_provider[ \t]*=")
_ROOT_OPENAI_BASE_URL_RE = re.compile(r"^[ \t]*openai_base_url[ \t]*=")


def _codex_credential_store(config_dir: Path) -> str | None:
    try:
        config = tomllib.loads((config_dir / "config.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    store = config.get("cli_auth_credentials_store")
    return store.lower() if isinstance(store, str) else None


def _codex_login_status(config_dir: Path) -> bool:
    env = {**os.environ, "CODEX_HOME": str(config_dir)}
    try:
        result = run(
            ["codex", "login", "status"],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
            env=env,
        )
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return False
    message = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0 and message.casefold() == "logged in using chatgpt"


def codex_uses_chatgpt_auth(auth_path: Path) -> bool:
    """Whether Codex authenticated via ChatGPT OAuth (vs an OpenAI API key).

    The account menu (profile/email/plan/usage) only renders when the active
    provider carries ``requires_openai_auth = true``, but that flag forces codex
    to demand an OpenAI OAuth login (#406) and would break API-key users.  So we
    emit it only in ChatGPT-OAuth mode, read from the sibling ``auth.json``.
    """
    try:
        raw = auth_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if _codex_credential_store(auth_path.parent) not in {"keyring", "auto"}:
            return False
        return _codex_login_status(auth_path.parent)
    except OSError:
        return False
    try:
        data = json.loads(raw)
    except ValueError:
        return False
    if not isinstance(data, dict):
        return False
    mode = data.get("auth_mode")
    if isinstance(mode, str):
        return mode.lower() == "chatgpt"
    # Older auth.json files predate `auth_mode`: infer from an OAuth account id.
    tokens = data.get("tokens")
    if isinstance(tokens, dict):
        account_id = tokens.get("account_id")
        return isinstance(account_id, str) and bool(account_id.strip())
    return False


def build_provider_section(
    *,
    port: int,
    name: str,
    marker_start: str = _CODEX_MARKER_START,
    marker_end: str = _CODEX_MARKER_END,
    include_markers: bool = True,
    requires_openai_auth: bool = False,
) -> str:
    """Build a managed Codex provider block.

    ``requires_openai_auth`` is emitted only for ChatGPT-OAuth users: the flag
    is what makes codex render the account menu, but it also forces codex to
    demand an OpenAI OAuth login (#406), which breaks API-key users.  Callers
    pass the result of :func:`codex_uses_chatgpt_auth`; it defaults to ``False``.
    """
    body = (
        "[model_providers.headroom]\n"
        f'name = "{name}"\n'
        f'base_url = "{proxy_base_url(port)}"\n'
        "supports_websockets = true\n"
    )
    if requires_openai_auth:
        body += "requires_openai_auth = true\n"
    if not include_markers:
        return body
    return f"{marker_start}\n{body}{marker_end}\n"


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Codex."""
    del backend
    return {"OPENAI_BASE_URL": proxy_base_url(port)}


def _insert_block_at_root(content: str, block: str) -> str:
    """Place a marker block carrying top-level keys above the first TOML table.

    Codex scopes bare keys under the preceding ``[table]`` header, so a
    ``model_provider`` appended after a table (e.g. ``[features]``) is silently
    ignored and routing never switches (#260). Land the block at the document
    root instead.
    """
    block = block.strip()
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if _TOML_TABLE_HEADER_RE.search(line):
            head = "\n".join(lines[:index]).rstrip()
            tail = "\n".join(lines[index:]).lstrip("\n")
            prefix = f"{head}\n\n" if head else ""
            return (f"{prefix}{block}\n\n{tail}").rstrip() + "\n"
    return (content.rstrip() + "\n\n" + block + "\n").lstrip()


def _strip_root_provider_assignments(content: str) -> str:
    """Remove root provider assignments without touching table-scoped settings."""
    lines = content.splitlines(keepends=True)
    kept: list[str] = []
    in_root = True
    for line in lines:
        if in_root and _TOML_TABLE_HEADER_RE.search(line):
            in_root = False
        if in_root and (
            _ROOT_MODEL_PROVIDER_RE.match(line) or _ROOT_OPENAI_BASE_URL_RE.match(line)
        ):
            continue
        kept.append(line)
    return "".join(kept)


def apply_provider_scope(manifest: DeploymentManifest) -> ManagedMutation | None:
    """Apply Codex provider-scope configuration when requested."""
    if manifest.scope != ConfigScope.PROVIDER.value:
        return None

    path = codex_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    section = (
        f"{_CODEX_MARKER_START}\n"
        'model_provider = "headroom"\n'
        f'openai_base_url = "{proxy_base_url(manifest.port)}"\n\n'
        + build_provider_section(
            port=manifest.port,
            name="Headroom persistent proxy",
            include_markers=False,
            requires_openai_auth=codex_uses_chatgpt_auth(path.parent / "auth.json"),
        )
        + f"{_CODEX_MARKER_END}\n"
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    # Drop our previous block and any prior top-level provider assignment so the
    # managed keys override the user's, then land them at the document root.
    existing = _CODEX_PATTERN.sub("", existing)
    existing = _strip_root_provider_assignments(existing)
    merged = _insert_block_at_root(existing, section)
    path.write_text(merged, encoding="utf-8")
    # Pull existing native threads into the headroom-provider menu so Codex's
    # history list stays whole once it routes through Headroom. Best-effort.
    retag_to_headroom(path.parent)
    return ManagedMutation(target=ToolTarget.CODEX.value, kind="toml-block", path=str(path))


def revert_provider_scope(mutation: ManagedMutation, manifest: DeploymentManifest) -> None:
    """Revert Codex provider-scope configuration."""
    del manifest
    if not mutation.path:
        return
    path = Path(mutation.path)
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    # Remove the managed marker block.
    if _CODEX_MARKER_START in content:
        content = _CODEX_PATTERN.sub("", content)
    # Strip any orphan top-level keys that a crashed or partial write may have
    # left outside the marker block (mirrors wrap.py _strip_codex_headroom_blocks).
    content = _ORPHAN_MODEL_PROVIDER.sub("", content)
    content = _ORPHAN_OPENAI_BASE_URL.sub("", content)
    content = _ORPHAN_HEADROOM_TABLE.sub("", content)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    # Hand the threads back to the native-provider menu so the full history stays
    # visible once Codex no longer routes through Headroom. Best-effort.
    retag_to_native(path.parent)
