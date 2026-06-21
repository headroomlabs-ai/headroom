"""OpenCode-specific provider helpers."""

from .config import (
    _MCP_MARKER_END,
    _MCP_MARKER_START,
    _PROVIDER_MARKER_END,
    _PROVIDER_MARKER_START,
    _opencode_config_path,
    opencode_config_paths,
    snapshot_opencode_config_if_unwrapped,
    strip_opencode_headroom_blocks,
)
from .install import apply_provider_scope, build_install_env, revert_provider_scope
from .runtime import (
    build_launch_env,
    build_overlay,
    build_provider_upstream_map,
    discover_user_providers,
    has_zen_auth,
    proxy_base_url,
)

__all__ = [
    "_MCP_MARKER_END",
    "_MCP_MARKER_START",
    "_opencode_config_path",
    "_PROVIDER_MARKER_END",
    "_PROVIDER_MARKER_START",
    "apply_provider_scope",
    "build_install_env",
    "build_launch_env",
    "build_overlay",
    "build_provider_upstream_map",
    "discover_user_providers",
    "has_zen_auth",
    "opencode_config_paths",
    "proxy_base_url",
    "revert_provider_scope",
    "snapshot_opencode_config_if_unwrapped",
    "strip_opencode_headroom_blocks",
]
