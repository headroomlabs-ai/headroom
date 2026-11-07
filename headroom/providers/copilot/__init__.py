"""Copilot-specific provider helpers."""

from .vscode import (
    configure_vscode_proxy_settings,
    remove_vscode_proxy_settings,
    unrouted_vscode_profiles,
    vscode_proxy_url,
    vscode_settings_path,
    vscode_user_dir,
)
from .wrap import (
    VSCODE_MODEL_ID_PREFIX,
    build_launch_env,
    copilot_model_from_args,
    default_wire_api_for_model,
    detect_running_proxy_backend,
    is_auto_model,
    model_configured,
    model_prefers_responses_api,
    model_requires_chat_completions,
    provider_key_source,
    query_proxy_config,
    resolve_provider_type,
    strip_auto_model_args,
    validate_configuration,
)

__all__ = [
    "VSCODE_MODEL_ID_PREFIX",
    "build_launch_env",
    "copilot_model_from_args",
    "default_wire_api_for_model",
    "detect_running_proxy_backend",
    "is_auto_model",
    "model_prefers_responses_api",
    "model_requires_chat_completions",
    "model_configured",
    "provider_key_source",
    "query_proxy_config",
    "resolve_provider_type",
    "strip_auto_model_args",
    "validate_configuration",
    "configure_vscode_proxy_settings",
    "remove_vscode_proxy_settings",
    "unrouted_vscode_profiles",
    "vscode_proxy_url",
    "vscode_settings_path",
    "vscode_user_dir",
]
