"""DeepSeek Harness (dsh) install-time helpers."""

from __future__ import annotations

from .runtime import proxy_base_url


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for DeepSeek Harness."""
    del backend
    return {"DEEPSEEK_BASE_URL": proxy_base_url(port)}
