"""OpenCode-specific provider helpers."""

from .config import (
    _MCP_MARKER_END,
    _MCP_MARKER_START,
    _PROVIDER_MARKER_END,
    _PROVIDER_MARKER_START,
    inject_opencode_provider_config,
    opencode_config_paths,
    snapshot_opencode_config_if_unwrapped,
    strip_opencode_headroom_blocks,
)
from .install import apply_provider_scope, build_install_env, revert_provider_scope
from .runtime import build_launch_env, build_opencode_config_content, proxy_base_url

__all__ = [
    "_MCP_MARKER_END",
    "_MCP_MARKER_START",
    "_PROVIDER_MARKER_END",
    "_PROVIDER_MARKER_START",
    "apply_provider_scope",
    "build_install_env",
    "build_launch_env",
    "build_opencode_config_content",
    "inject_opencode_provider_config",
    "opencode_config_paths",
    "proxy_base_url",
    "revert_provider_scope",
    "snapshot_opencode_config_if_unwrapped",
    "strip_opencode_headroom_blocks",
]
