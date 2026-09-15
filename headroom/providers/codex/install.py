"""Codex install-time helpers."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

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
_CODEX_API_KEY_HELPER_NAME = ".headroom-codex-auth.py"
_CODEX_API_KEY_HELPER = """from pathlib import Path
import json


try:
    auth = json.loads(Path(__file__).with_name("auth.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(2)

key = auth.get("OPENAI_API_KEY") if isinstance(auth, dict) else None
if not isinstance(key, str) or not key.strip():
    raise SystemExit(2)

print(key, end="")
"""


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
        if isinstance(account_id, str) and account_id.strip():
            return True
        return _id_token_carries_chatgpt_account(tokens.get("id_token"))
    return False


def codex_uses_api_key_auth(auth_path: Path) -> bool:
    """Whether Codex has a file-backed OpenAI API key.

    A custom provider does not read Codex's ``auth.json`` when
    ``requires_openai_auth`` is false.  Detect the API-key shape separately so
    the generated provider can use Codex's command-backed bearer-token config
    without forcing API-key users into the OAuth login flow.
    """
    try:
        data = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and isinstance(data.get("OPENAI_API_KEY"), str)
        and bool(data["OPENAI_API_KEY"].strip())
    )


def build_codex_auth_config(auth_path: Path | None) -> str:
    """Build a Codex provider auth command for a file-backed API key.

    The helper is generated next to ``auth.json`` and emits only the token at
    runtime.  The token is never copied into Headroom's environment or the
    generated TOML configuration.
    """
    if auth_path is None or codex_uses_chatgpt_auth(auth_path):
        return ""
    if not codex_uses_api_key_auth(auth_path):
        return ""

    helper_path = auth_path.parent / _CODEX_API_KEY_HELPER_NAME
    if not _ensure_codex_auth_helper(helper_path):
        return ""

    return (
        "auth = { command = "
        f"{json.dumps(sys.executable)}, args = [{json.dumps(str(helper_path.resolve()))}], "
        "refresh_interval_ms = 300000 }\n"
    )


def _ensure_codex_auth_helper(helper_path: Path) -> bool:
    """Create or validate the private, Headroom-owned auth helper."""
    created = False
    try:
        # Never follow a user-created link or overwrite an unrelated file.
        if helper_path.is_symlink():
            return False
        if helper_path.exists():
            if not helper_path.is_file():
                return False
            if helper_path.read_text(encoding="utf-8") != _CODEX_API_KEY_HELPER:
                return False
            helper_path.chmod(0o600)
            return True

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(helper_path, flags, 0o600)
        created = True
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                fd = -1
                handle.write(_CODEX_API_KEY_HELPER)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
        helper_path.chmod(0o600)
        return True
    except (OSError, UnicodeError):
        if created:
            try:
                if helper_path.is_file() and not helper_path.is_symlink():
                    helper_path.unlink()
            except OSError:
                pass
        return False


def cleanup_codex_auth_helper(auth_path: Path) -> None:
    """Remove a private helper only when it still contains Headroom code."""
    helper_path = auth_path.parent / _CODEX_API_KEY_HELPER_NAME
    try:
        if (
            helper_path.is_symlink()
            or not helper_path.is_file()
            or helper_path.read_text(encoding="utf-8") != _CODEX_API_KEY_HELPER
        ):
            return
        helper_path.unlink()
    except (OSError, UnicodeError):
        return


def codex_auth_helper_is_referenced(content: str, helper_path: str) -> bool | None:
    """Whether parsed Codex TOML names this exact generated helper.

    ``None`` means the configuration is not parseable, so callers can retain
    the helper instead of risking deletion of a pre-existing user file.
    """
    try:
        document = tomllib.loads(content)
    except (tomllib.TOMLDecodeError, TypeError):
        return None

    providers = document.get("model_providers")
    headroom = providers.get("headroom") if isinstance(providers, dict) else None
    auth = headroom.get("auth") if isinstance(headroom, dict) else None
    args = auth.get("args") if isinstance(auth, dict) else None
    return isinstance(args, list) and helper_path in args


def _id_token_carries_chatgpt_account(raw: Any) -> bool:
    """Whether an ``id_token`` carries the ChatGPT account claim (#3206).

    Newer Codex releases can write an ``auth.json`` with neither ``auth_mode``
    nor a top-level ``tokens.account_id``; the account identity lives only in
    the ``id_token`` claims. Those configs then read as API-key mode, so
    ``requires_openai_auth`` is omitted, Codex attaches no Authorization
    header, and every request 401s with "Missing bearer".

    The payload is decoded, not verified. This is a local config file the user
    already owns, and the result only decides which key we write into their own
    ``config.toml`` -- nothing is authenticated or authorised on the strength
    of it. An API-key user has no ChatGPT id_token, so this cannot resurrect
    the forced-OAuth-login regression in #406.
    """
    if not isinstance(raw, str):
        return False
    parts = raw.split(".")
    if len(parts) != 3:
        return False
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
    except Exception:
        return False
    if not isinstance(claims, dict):
        return False
    auth_claim = claims.get("https://api.openai.com/auth")
    if not isinstance(auth_claim, dict):
        return False
    account_id = auth_claim.get("chatgpt_account_id")
    return isinstance(account_id, str) and bool(account_id.strip())


def build_provider_section(
    *,
    port: int,
    name: str,
    marker_start: str = _CODEX_MARKER_START,
    marker_end: str = _CODEX_MARKER_END,
    include_markers: bool = True,
    requires_openai_auth: bool = False,
    auth_path: Path | None = None,
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
        f"{build_codex_auth_config(auth_path)}"
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
            auth_path=path.parent / "auth.json",
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
    helper_path = path.parent / _CODEX_API_KEY_HELPER_NAME
    helper_was_referenced = codex_auth_helper_is_referenced(
        content, str(helper_path.resolve())
    )
    # Remove the managed marker block.
    if _CODEX_MARKER_START in content:
        content = _CODEX_PATTERN.sub("", content)
    # Strip any orphan top-level keys that a crashed or partial write may have
    # left outside the marker block (mirrors wrap.py _strip_codex_headroom_blocks).
    content = _ORPHAN_MODEL_PROVIDER.sub("", content)
    content = _ORPHAN_OPENAI_BASE_URL.sub("", content)
    content = _ORPHAN_HEADROOM_TABLE.sub("", content)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    if helper_was_referenced is True:
        cleanup_codex_auth_helper(path.parent / "auth.json")
    # Hand the threads back to the native-provider menu so the full history stays
    # visible once Codex no longer routes through Headroom. Best-effort.
    retag_to_native(path.parent)
