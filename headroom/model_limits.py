"""Operator-configured model settings (``HEADROOM_MODEL_LIMITS`` / ``models.json``).

The OpenAI section of that configuration is read by ``OpenAIProvider`` (context
limits, pricing, encodings) and by the tokenizer registry (encodings), so both
see one parse of the same file. It lives here rather than in
``headroom.providers.openai`` because ``headroom.tokenizers`` sits below the
providers and must not import them.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from headroom import paths as _paths

logger = logging.getLogger(__name__)


def load_openai_model_config() -> dict[str, Any]:
    """Load custom model configuration from environment or config file.

    Checks (in order):
    1. HEADROOM_MODEL_LIMITS environment variable (JSON string or file path)
    2. ~/.headroom/models.json config file

    Returns:
        Dict with 'context_limits', 'pricing' and 'encodings' keys.
    """
    config: dict[str, Any] = {"context_limits": {}, "pricing": {}, "encodings": {}}

    # Check environment variable
    env_config = os.environ.get("HEADROOM_MODEL_LIMITS", "")
    if env_config:
        try:
            # Check if it's a file path
            if os.path.isfile(env_config):
                with open(env_config, encoding="utf-8") as f:
                    loaded = json.load(f)
            else:
                # Try to parse as JSON string
                loaded = json.loads(env_config)

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

    # Check config file. Prefer the canonical config-dir location, then fall
    # back to the legacy workspace-root location for backward compatibility.
    config_file = _paths.models_config_path()
    if not config_file.exists():
        legacy_models = _paths.workspace_dir() / "models.json"
        if legacy_models.exists():
            config_file = legacy_models
    if config_file.exists():
        try:
            with open(config_file, encoding="utf-8") as f:
                loaded = json.load(f)

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
