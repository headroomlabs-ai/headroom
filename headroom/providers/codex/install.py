"""Codex install-time helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

from headroom.install.models import ConfigScope, DeploymentManifest, ManagedMutation, ToolTarget
from headroom.install.paths import codex_config_path

from .runtime import proxy_base_url

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


def codex_uses_chatgpt_auth(auth_path: Path) -> bool:
    """Whether Codex authenticated via ChatGPT OAuth (vs an OpenAI API key).

    The account menu (profile/email/plan/usage) only renders when the active
    provider carries ``requires_openai_auth = true``, but that flag forces codex
    to demand an OpenAI OAuth login (#406) and would break API-key users.  So we
    emit it only in ChatGPT-OAuth mode, read from the sibling ``auth.json``.
    """
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
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
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if _CODEX_MARKER_START in existing:
            merged = _CODEX_PATTERN.sub(section, existing)
        else:
            merged = existing.rstrip() + "\n\n" + section + "\n"
    else:
        merged = section + "\n"
    path.write_text(merged, encoding="utf-8")
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
