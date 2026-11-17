"""Session-affinity headers for hosted providers that route on them.

Some providers run many replicas, each with its own prompt cache, and pick one
per request. Without a hint, consecutive turns of one conversation can land on
different replicas and miss the cache the previous turn built. Providers that
route on a header get Headroom's session id in it, so a conversation stays on
one replica.

Only hosts listed here receive the header; every other upstream (Anthropic,
OpenAI, Bedrock, Vertex, ...) is left untouched. A header the client already
set wins.
"""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit

# Host -> header it routes on. Fireworks: docs.fireworks.ai prompt caching.
AFFINITY_HEADERS: dict[str, str] = {"api.fireworks.ai": "x-session-affinity"}


def session_affinity_headers(
    url: str, session_id: str | None, headers: Mapping[str, str]
) -> dict[str, str]:
    """Headers to add for ``url``: the provider's affinity header, or nothing."""
    name = AFFINITY_HEADERS.get(urlsplit(url).hostname or "")
    if name is None or not session_id or any(k.lower() == name for k in headers):
        return {}
    return {name: session_id}
