"""OpenCode config file helpers for wrap and persistent install."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import click

from headroom import fsutil
from headroom.install.paths import opencode_config_path

# Headroom-managed JSON marker comments for idempotent block injection.
_PROVIDER_MARKER_START = "// --- Headroom proxy provider ---"
_PROVIDER_MARKER_END = "// --- end Headroom proxy provider ---"
_MCP_MARKER_START = "// --- Headroom MCP server ---"
_MCP_MARKER_END = "// --- end Headroom MCP server ---"

# Regex to strip headroom blocks (including the marker comments).
_PROVIDER_BLOCK_RE = re.compile(
    re.escape(_PROVIDER_MARKER_START) + r".*?" + re.escape(_PROVIDER_MARKER_END),
    re.DOTALL,
)
_MCP_BLOCK_RE = re.compile(
    re.escape(_MCP_MARKER_START) + r".*?" + re.escape(_MCP_MARKER_END),
    re.DOTALL,
)
HEADROOM_OPENCODE_PLUGIN = "headroom-opencode"

# Models exposed by the injected `headroom` provider. This provider uses
# ``@ai-sdk/openai-compatible`` and the proxy's ``/v1/chat/completions`` path,
# which is routed to the configured OpenAI upstream. Do not advertise Claude
# models here: OpenCode would send them through the OpenAI endpoint and report
# an ``invalid_api_key`` error instead of reaching Anthropic. Claude models are
# available through OpenCode's native ``anthropic`` provider, whose base URL is
# also redirected to Headroom by ``build_opencode_config_content``.
#
# OpenCode only resolves ``headroom/<id>`` for ids listed in this map, so an
# empty map means every documented ``headroom/*`` model fails with "Model not
# found".
HEADROOM_OPENCODE_MODELS: dict[str, Any] = {
    "gpt-4o": {
        "name": "GPT-4o",
        "limit": {"context": 128000, "output": 16384},
    },
    "gpt-4.1": {
        "name": "GPT-4.1",
        "limit": {"context": 1048576, "output": 32768},
    },
}


# Env var holding comma-separated extra model specs (same format as the
# `--extra-model` flag) so custom models survive re-wraps.
HEADROOM_OPENCODE_EXTRA_MODELS_ENV = "HEADROOM_OPENCODE_EXTRA_MODELS"

# Defaults for extra models when the spec omits limits.
_EXTRA_MODEL_DEFAULT_CONTEXT = 200000
_EXTRA_MODEL_DEFAULT_OUTPUT = 16384


def parse_extra_model_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse an ``id[:name[:context_window]]`` extra-model spec.

    Returns a ``(model_id, entry)`` pair shaped like the values in
    ``HEADROOM_OPENCODE_MODELS``. ``name`` defaults to the id and
    ``context_window`` defaults to 200000. Raises ``ValueError`` on an
    empty id or a non-integer context window.
    """
    parts = spec.split(":", 2)
    model_id = parts[0].strip()
    if not model_id:
        raise ValueError(f"extra model spec has an empty model id: {spec!r}")
    name = parts[1].strip() if len(parts) > 1 and parts[1].strip() else model_id
    context = _EXTRA_MODEL_DEFAULT_CONTEXT
    if len(parts) > 2 and parts[2].strip():
        try:
            context = int(parts[2].strip())
        except ValueError:
            raise ValueError(
                f"extra model spec has a non-integer context window: {spec!r}"
            ) from None
        if context <= 0:
            raise ValueError(f"extra model spec has a non-positive context window: {spec!r}")
    return model_id, {
        "name": name,
        "limit": {"context": context, "output": _EXTRA_MODEL_DEFAULT_OUTPUT},
    }


def extra_models_from_env() -> dict[str, Any]:
    """Parse extra models from ``HEADROOM_OPENCODE_EXTRA_MODELS`` (comma-separated)."""
    raw = os.environ.get(HEADROOM_OPENCODE_EXTRA_MODELS_ENV, "")
    models: dict[str, Any] = {}
    for spec in raw.split(","):
        spec = spec.strip()
        if not spec:
            continue
        model_id, entry = parse_extra_model_spec(spec)
        models[model_id] = entry
    return models


def headroom_provider_entry(
    port: int, extra_models: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the `headroom` provider block pointed at the local proxy."""
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Headroom Proxy",
        "options": {"baseURL": f"http://127.0.0.1:{port}/v1"},
        "models": {**HEADROOM_OPENCODE_MODELS, **(extra_models or {})},
    }


def _opencode_home_dir() -> Path:
    """Return the OpenCode home/config directory."""
    env_path = os.environ.get("OPENCODE_HOME", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".config" / "opencode"


def opencode_config_paths() -> tuple[Path, Path]:
    """Return ``(config_file, backup_file)`` for OpenCode."""
    config_file = opencode_config_path()
    backup_file = config_file.with_name(config_file.name + ".headroom-backup")
    return config_file, backup_file


def snapshot_opencode_config_if_unwrapped(config_file: Path, backup_file: Path) -> None:
    """Snapshot ``opencode.json`` to ``backup_file`` before the first injection.

    Guarantees that ``headroom unwrap opencode`` can restore the user's
    original file byte-for-byte.
    """
    if backup_file.exists():
        return
    if not config_file.exists():
        return
    try:
        content = fsutil.read_text(config_file)
    except OSError:
        return
    if _PROVIDER_MARKER_START in content or _MCP_MARKER_START in content:
        return
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_file, backup_file)


def strip_opencode_headroom_blocks(content: str, *, remove_mcp: bool = True) -> str:
    """Remove all Headroom-managed blocks from opencode JSON text.

    Preserves user content. Returns the cleaned string.
    """
    content = _PROVIDER_BLOCK_RE.sub("", content)
    if remove_mcp:
        content = _MCP_BLOCK_RE.sub("", content)
    # Collapse multiple blank lines left behind by block removal.
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def _render_provider_block(port: int, extra_models: dict[str, Any] | None = None) -> str:
    """Render a Headroom provider block as a JSON comment-wrapped snippet."""
    provider = {"headroom": headroom_provider_entry(port, extra_models)}
    lines = [
        _PROVIDER_MARKER_START,
        f'"provider": {json.dumps(provider, indent=2)},',
        _PROVIDER_MARKER_END,
    ]
    return "\n".join(lines)


def _parse_json_loose(text: str) -> dict[str, Any]:
    """Parse JSON text, stripping line comments (// ...) when needed.

    Tries standard JSON first to avoid corrupting URLs that contain ``//``.
    Falls back to stripping ``//`` comments when standard parsing fails.
    Two-pass: (1) remove comment-only lines, (2) strip inline trailing
    comments that follow a comma.
    """
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    # Pass 1: remove lines that are ONLY a comment.
    cleaned = re.sub(r"^\s*//[^\n]*\n", "", text, flags=re.MULTILINE)
    # Pass 2: remove inline trailing comments (", // comment").
    cleaned = re.sub(r",\s*//[^\n]*", ",", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _inject_key_into_json(data: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Merge ``value`` into ``data[key]`` idempotently."""
    existing = data.get(key)
    if isinstance(existing, dict) and isinstance(value, dict):
        merged = {**existing, **value}
        data[key] = merged
    else:
        data[key] = value
    return data


def append_headroom_plugin(config: dict[str, object]) -> bool:
    """Append the optional OpenCode plugin entry if it is not already present."""
    plugin = config.get("plugin")
    if plugin is None:
        config["plugin"] = [HEADROOM_OPENCODE_PLUGIN]
        return True

    if not isinstance(plugin, list):
        return False

    for entry in plugin:
        if entry == HEADROOM_OPENCODE_PLUGIN:
            return False
        if isinstance(entry, list) and entry and entry[0] == HEADROOM_OPENCODE_PLUGIN:
            return False

    plugin.append(HEADROOM_OPENCODE_PLUGIN)
    return True


def inject_opencode_provider_config(port: int, extra_models: dict[str, Any] | None = None) -> None:
    """Inject a Headroom model provider into OpenCode's config file.

    Safe to call multiple times — the injected block is fully replaced on
    each call, so re-running with a different ``port`` updates the config.
    Before the first injection, the pre-wrap file is snapshotted to
    ``opencode.json.headroom-backup`` so ``headroom unwrap opencode``
    can restore it byte-for-byte.
    """
    config_file, backup_file = opencode_config_paths()
    config_dir = config_file.parent

    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        snapshot_opencode_config_if_unwrapped(config_file, backup_file)

        if config_file.exists():
            content = fsutil.read_text(config_file)
            data = _parse_json_loose(content)
        else:
            content = ""
            data = {}

        # Strip any prior Headroom-managed blocks before re-injecting.
        if _PROVIDER_MARKER_START in content or _MCP_MARKER_START in content:
            content = strip_opencode_headroom_blocks(content)
            data = _parse_json_loose(content)

        # Merge provider into the JSON data structure.
        provider = {"headroom": headroom_provider_entry(port, extra_models)}
        data = _inject_key_into_json(data, "provider", provider)

        # Write back as formatted JSON (opencode uses standard JSON with comments).
        output = json.dumps(data, indent=2) + "\n"
        config_file.write_text(output, encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(
            f"could not write OpenCode config at {config_file}: {exc}"
        ) from exc
