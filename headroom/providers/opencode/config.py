"""OpenCode config file helpers for wrap and persistent install."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
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
# `--extra-model` flag) so custom models survive re-wraps. Names in it cannot
# contain a comma.
HEADROOM_OPENCODE_EXTRA_MODELS_ENV = "HEADROOM_OPENCODE_EXTRA_MODELS"

# Defaults for extra models when the spec omits limits. The output default is
# also capped at the context window.
_EXTRA_MODEL_DEFAULT_CONTEXT = 200000
_EXTRA_MODEL_DEFAULT_OUTPUT = 16384


def _extra_model_limit(raw: str, label: str, spec: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"extra model spec has a non-integer {label}: {spec!r}") from None
    if value <= 0:
        raise ValueError(f"extra model spec has a non-positive {label}: {spec!r}")
    return value


def parse_extra_model_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse an ``id[=name[=context_window[=output_limit]]]`` extra-model spec.

    Fields are split on ``=`` because model ids routinely contain ``:``
    (Ollama ``qwen2.5-coder:7b``, OpenRouter ``vendor/model:free``).
    Returns a ``(model_id, entry)`` pair shaped like the values in
    ``HEADROOM_OPENCODE_MODELS``. ``name`` defaults to the id,
    ``context_window`` to 200000 and ``output_limit`` to
    ``min(16384, context_window)``. Raises ``ValueError`` on an empty id or a
    non-integer or non-positive limit.
    """
    parts = [part.strip() for part in spec.split("=", 3)]
    model_id = parts[0]
    if not model_id:
        raise ValueError(f"extra model spec has an empty model id: {spec!r}")
    name = parts[1] if len(parts) > 1 and parts[1] else model_id
    context = (
        _extra_model_limit(parts[2], "context window", spec)
        if len(parts) > 2 and parts[2]
        else _EXTRA_MODEL_DEFAULT_CONTEXT
    )
    output = (
        _extra_model_limit(parts[3], "output limit", spec)
        if len(parts) > 3 and parts[3]
        else min(_EXTRA_MODEL_DEFAULT_OUTPUT, context)
    )
    return model_id, {"name": name, "limit": {"context": context, "output": output}}


def extra_models_from_env(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Parse extra models from ``HEADROOM_OPENCODE_EXTRA_MODELS`` (comma-separated).

    Reads ``environ`` when given (a deployment manifest's env), else ``os.environ``.
    """
    env = os.environ if environ is None else environ
    raw = env.get(HEADROOM_OPENCODE_EXTRA_MODELS_ENV, "")
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
    """Return the selected config and its canonical Headroom backup path."""
    config_file = opencode_config_path()
    return config_file, config_file.with_name(config_file.name + ".headroom-backup")


def migrate_legacy_opencode_jsonc_backup(config_file: Path, backup_file: Path) -> None:
    """Migrate only legacy JSONC snapshots whose ownership is unambiguous.

    Remember ambiguity on disk before a canonical snapshot can be consumed.
    Otherwise a later install/remove cycle could claim an unrelated legacy
    backup after the canonical backup or sibling JSON config disappears.
    """
    if config_file.suffix.lower() != ".jsonc":
        return
    legacy_backup = config_file.with_suffix(".json.headroom-backup")
    migration_block = legacy_backup.with_name(legacy_backup.name + ".jsonc-migration-blocked")
    if os.path.lexists(migration_block) or not legacy_backup.exists():
        return
    if os.path.lexists(backup_file) or os.path.lexists(config_file.with_suffix(".json")):
        # Keep the legacy bytes in place for their owner or manual recovery.
        # Fail closed if this marker cannot be persisted, before any restore.
        migration_block.touch(exist_ok=True)
        return
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_backup.replace(backup_file)


def snapshot_opencode_config_if_unwrapped(config_file: Path, backup_file: Path) -> None:
    """Snapshot ``opencode.json`` to ``backup_file`` before the first injection.

    Guarantees that ``headroom unwrap opencode`` can restore the user's
    original file byte-for-byte.
    """
    migrate_legacy_opencode_jsonc_backup(config_file, backup_file)
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


def _strip_jsonc_comments(text: str) -> str:
    """Remove JSONC line and block comments while preserving quoted strings."""
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            output.append(char)
        elif text.startswith("//", index):
            newline = text.find("\n", index + 2)
            if newline < 0:
                break
            output.append("\n")
            index = newline
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                break
            output.extend("\n" for char in text[index : end + 2] if char == "\n")
            index = end + 1
        else:
            output.append(char)
        index += 1
    return "".join(output)


def _strip_jsonc_trailing_commas(text: str) -> str:
    """Remove commas before JSONC closing delimiters without touching strings."""
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            output.append(char)
        elif char == ",":
            next_index = index + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if next_index < len(text) and text[next_index] in "}]":
                index += 1
                continue
            output.append(char)
        else:
            output.append(char)
        index += 1
    return "".join(output)


def _parse_json_loose(text: str) -> dict[str, Any]:
    """Parse JSON text and JSONC comments/trailing commas.

    Standard JSON is tried first so URLs containing ``//`` remain untouched.
    """
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    cleaned = _strip_jsonc_comments(text)
    cleaned = _strip_jsonc_trailing_commas(cleaned)
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


def inject_opencode_provider_config(
    port: int, extra_models: dict[str, Any] | None = None, *, keep_user_entries: bool = False
) -> None:
    """Inject a Headroom model provider into OpenCode's config file.

    Safe to call multiple times — the injected block is replaced on each call,
    so re-running with a different ``port`` updates the config. With
    ``keep_user_entries`` the model ids and options the user added under the
    ``headroom`` provider are kept; callers pass it only when they have pointed
    the proxy at an explicit upstream.
    Before the first injection, the pre-wrap file is snapshotted to
    ``opencode.json.headroom-backup`` so ``headroom unwrap opencode``
    can restore it byte-for-byte.
    """
    config_file, backup_file = opencode_config_paths()
    config_dir = config_file.parent

    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        migrate_legacy_opencode_jsonc_backup(config_file, backup_file)
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
        entry = headroom_provider_entry(port, extra_models)
        # Keep the user's own model ids and options (an apiKey, say) under the
        # headroom provider: OpenCode only resolves `headroom/<id>` for listed
        # ids, so a third-party upstream needs them. Only when the caller named
        # that upstream, though: otherwise the proxy forwards to OpenAI, and a
        # kept third-party key would be sent there. Headroom still owns npm,
        # name and baseURL.
        providers = data.get("provider")
        existing = providers.get("headroom") if isinstance(providers, dict) else None
        if keep_user_entries and isinstance(existing, dict):
            if isinstance(existing.get("options"), dict):
                entry["options"] = {**existing["options"], **entry["options"]}
            if isinstance(existing.get("models"), dict):
                entry["models"] = {**entry["models"], **existing["models"], **(extra_models or {})}
        provider = {"headroom": entry}
        data = _inject_key_into_json(data, "provider", provider)

        # Write back as formatted JSON (opencode uses standard JSON with comments).
        output = json.dumps(data, indent=2) + "\n"
        config_file.write_text(output, encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(
            f"could not write OpenCode config at {config_file}: {exc}"
        ) from exc
