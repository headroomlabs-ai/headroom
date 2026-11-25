"""Operator-configured model settings (``HEADROOM_MODEL_LIMITS`` / ``models.json``).

The OpenAI section of that configuration holds context limits, pricing and
encodings. ``OpenAIProvider`` takes context limits and pricing from
:func:`load_openai_model_config`; the provider and the tokenizer registry both
take encodings from :func:`load_openai_encodings`, so they cannot disagree about
a model's vocabulary. Both functions read the same sources in the same order. It lives here rather than in
``headroom.providers.openai`` because ``headroom.tokenizers`` sits below the
providers and must not import them.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from headroom import paths as _paths

logger = logging.getLogger(__name__)

_ENV_VAR = "HEADROOM_MODEL_LIMITS"


def _read_env_config(value: str) -> Any:
    """Parse ``HEADROOM_MODEL_LIMITS``: a path to a JSON file, else a JSON string."""
    if os.path.isfile(value):
        return _read_json_file(value)
    return json.loads(value)


def _read_json_file(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _models_config_file() -> Path | None:
    """The ``models.json`` to read, if any.

    Prefer the canonical config-dir location, then fall back to the legacy
    workspace-root location for backward compatibility.
    """
    config_file = _paths.models_config_path()
    if config_file.exists():
        return config_file
    legacy_models = _paths.workspace_dir() / "models.json"
    if legacy_models.exists():
        return legacy_models
    return None


def load_openai_model_config() -> dict[str, Any]:
    """Load custom model configuration from environment or config file.

    Checks (in order):
    1. HEADROOM_MODEL_LIMITS environment variable (JSON string or file path)
    2. ~/.headroom/models.json config file

    Returns:
        Dict with 'context_limits', 'pricing' and 'encodings' keys. Callers
        that need encodings use :func:`load_openai_encodings` instead, which
        does not depend on the other sections being well-formed.
    """
    config: dict[str, Any] = {"context_limits": {}, "pricing": {}, "encodings": {}}

    # Check environment variable
    env_config = os.environ.get(_ENV_VAR, "")
    if env_config:
        try:
            loaded = _read_env_config(env_config)

            if not isinstance(loaded, dict):
                raise ValueError(
                    f"HEADROOM_MODEL_LIMITS must be a JSON object, got {type(loaded).__name__}"
                )

            openai_config = loaded.get("openai", loaded)
            if "context_limits" in openai_config:
                config["context_limits"].update(openai_config["context_limits"])
            if "pricing" in openai_config:
                config["pricing"].update(openai_config["pricing"])
            if "encodings" in openai_config:
                config["encodings"].update(openai_config["encodings"])

            logger.debug("Loaded custom OpenAI model config from HEADROOM_MODEL_LIMITS")
        except (ValueError, OSError) as e:
            # ValueError covers json.JSONDecodeError (a subclass) and the
            # non-object guard above, so a malformed value warns and falls back
            # to defaults instead of crashing provider init.
            logger.warning(f"Failed to load HEADROOM_MODEL_LIMITS: {e}")

    # Check config file
    config_file = _models_config_file()
    if config_file is not None:
        try:
            loaded = _read_json_file(config_file)

            if not isinstance(loaded, dict):
                raise ValueError(f"{config_file} must contain a JSON object")

            openai_config = loaded.get("openai", {})
            if "context_limits" in openai_config:
                for model, limit in openai_config["context_limits"].items():
                    if model not in config["context_limits"]:
                        config["context_limits"][model] = limit
            if "pricing" in openai_config:
                for model, pricing in openai_config["pricing"].items():
                    if model not in config["pricing"]:
                        config["pricing"][model] = pricing
            if "encodings" in openai_config:
                for model, encoding in openai_config["encodings"].items():
                    if model not in config["encodings"]:
                        config["encodings"][model] = encoding

            logger.debug(f"Loaded custom OpenAI model config from {config_file}")
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to load {config_file}: {e}")

    return config


def load_openai_encodings() -> dict[str, str]:
    """Load only the operator's model -> tiktoken encoding mapping.

    Reads the same sources in the same order as :func:`load_openai_model_config`
    (``HEADROOM_MODEL_LIMITS`` wins over ``models.json``) but looks at nothing
    except ``openai.encodings``. The tokenizer registry resolves every model's
    counter from this, so a malformed ``pricing`` or ``context_limits`` section
    must not change how Claude or Gemini are counted, and ``OpenAIProvider``
    uses it too, so the provider and the registry always agree on a model's
    encoding. Anything unusable (unreadable source, invalid JSON, non-object
    sections, non-string encodings) is skipped with a warning; this never
    raises.
    """
    encodings: dict[str, str] = {}

    env_config = os.environ.get(_ENV_VAR, "")
    if env_config:
        encodings.update(
            _encodings_from(_ENV_VAR, partial(_read_env_config, env_config), env_source=True)
        )

    try:
        config_file = _models_config_file()
    except Exception as e:
        # Path.exists() raises PermissionError on Python 3.11/3.12.
        logger.warning("Ignoring model encodings from models.json: %s", e)
        config_file = None
    if config_file is not None:
        for model, encoding in _encodings_from(
            str(config_file), partial(_read_json_file, config_file), env_source=False
        ).items():
            encodings.setdefault(model, encoding)

    return encodings


def _encodings_from(source: str, read: Callable[[], Any], *, env_source: bool) -> dict[str, str]:
    """The valid ``model -> encoding`` entries of one configuration source.

    ``env_source`` keeps the convention of :func:`load_openai_model_config`:
    ``HEADROOM_MODEL_LIMITS`` may hold the OpenAI section at the top level,
    ``models.json`` must nest it under ``"openai"``.
    """
    try:
        loaded = read()
    except Exception as e:
        # Broad on purpose: besides ValueError/OSError, deeply nested JSON
        # raises RecursionError. Every model's counter depends on this read.
        logger.warning("Ignoring model encodings from %s: %s", source, e)
        return {}
    if not isinstance(loaded, dict):
        logger.warning(
            "Ignoring model encodings from %s: expected a JSON object, got %s",
            source,
            type(loaded).__name__,
        )
        return {}

    section = loaded.get("openai", loaded if env_source else {})
    if not isinstance(section, dict):
        logger.warning(
            "Ignoring model encodings from %s: 'openai' must be an object, got %s",
            source,
            type(section).__name__,
        )
        return {}
    raw = section.get("encodings", {})
    if not isinstance(raw, dict):
        logger.warning(
            "Ignoring model encodings from %s: 'encodings' must be an object, got %s",
            source,
            type(raw).__name__,
        )
        return {}

    encodings = {model: encoding for model, encoding in raw.items() if isinstance(encoding, str)}
    skipped = sorted(model for model in raw if model not in encodings)
    if skipped:
        logger.warning(
            "Ignoring model encodings from %s for %s: the encoding must be a string",
            source,
            ", ".join(skipped),
        )
    return encodings
