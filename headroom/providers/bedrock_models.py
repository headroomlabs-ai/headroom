"""Model-name rules for Bedrock, shared by the LiteLLM backend and the proxy's
model routes. No third-party imports: proxy startup and the backend both load it."""

from __future__ import annotations

from collections.abc import Iterable


def newest_snapshot(alias: str, names: Iterable[str]) -> str | None:
    """Newest dated snapshot ("claude-haiku-4-5-20251001") among ``names`` for
    an undated model name ("claude-haiku-4-5"), or None.

    Only an eight-digit date suffix counts, so "claude-sonnet-4" never
    resolves to "claude-sonnet-4-6".
    """
    dated = [
        name
        for name in names
        if name.startswith(f"{alias}-")
        and len(name) == len(alias) + 9
        and name[len(alias) + 1 :].isdigit()
    ]
    return max(dated) if dated else None
