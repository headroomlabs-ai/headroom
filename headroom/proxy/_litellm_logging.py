"""LiteLLM logging configuration for proxy internals."""

from __future__ import annotations

from typing import Any


def suppress_litellm_debug_output(litellm_module: Any) -> Any:
    """Disable LiteLLM's provider-list banner and verbose debug output."""
    litellm_module.suppress_debug_info = True
    litellm_module.set_verbose = False
    return litellm_module
