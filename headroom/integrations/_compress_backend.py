"""Local or Headroom Cloud compression, shared by the ASGI and LiteLLM integrations.

``CompressionMiddleware`` (asgi.py) and ``HeadroomCallback`` (litellm_callback.py)
compress the same way: in-process with ``headroom.compress()`` by default, or
through the Headroom Cloud API when an API key is set.

This module must not import starlette or litellm, so either integration can be
used without the other's dependency.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from headroom.log_safety import redact_url
from headroom.offline import guard_egress

_DEFAULT_CLOUD_URL = "https://api.headroomlabs.ai"
_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


_MEDIA_TYPE_RE = re.compile(r"[A-Za-z0-9!#$&^_.+-]{1,64}/[A-Za-z0-9!#$&^_.+-]{1,64}")


def _media_type(content_type: str) -> str:
    """Return just the ``type/subtype`` of a Content-Type header, or ``<other>``.

    Parameters and anything that is not a plain media type are dropped, so an
    error response cannot smuggle echoed content into the log through a header.
    """
    media = content_type.split(";", 1)[0].strip()
    return media.lower() if _MEDIA_TYPE_RE.fullmatch(media) else "<other>"


class CompressBackend:
    """Mixin that picks local or cloud compression and holds the cloud client.

    Subclasses call ``_init_compress_backend`` from ``__init__`` and set
    ``_log`` to their module logger, so log lines keep that logger's name.
    """

    _log: logging.Logger

    def _init_compress_backend(
        self,
        model_limit: int,
        hooks: Any,
        api_key: str | None,
        api_url: str | None,
    ) -> None:
        self._model_limit = model_limit
        self._hooks = hooks

        # Cloud mode: if api_key is set, compress via Headroom Cloud API
        # Falls back to HEADROOM_API_KEY env var
        self._api_key = api_key or os.environ.get("HEADROOM_API_KEY", "").strip() or None
        self._api_url = (
            api_url or os.environ.get("HEADROOM_API_URL", "").strip() or _DEFAULT_CLOUD_URL
        ).rstrip("/")
        self._client: Any = None  # Lazy-initialized httpx.AsyncClient

    @property
    def cloud_mode(self) -> bool:
        """Whether cloud compression is enabled."""
        return self._api_key is not None

    def _local_compress(self, messages: list[dict], model: str) -> dict[str, Any] | None:
        """Compress locally using headroom.compress()."""
        from headroom.compress import compress

        result = compress(
            messages=messages,
            model=model or _DEFAULT_MODEL,
            model_limit=self._model_limit,
            hooks=self._hooks,
        )
        return {
            "messages": result.messages,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "tokens_saved": result.tokens_saved,
            "compression_ratio": result.compression_ratio,
        }

    async def _cloud_compress(self, messages: list[dict], model: str) -> dict[str, Any] | None:
        """Compress via Headroom Cloud API (managed CCR, TOIN, analytics).

        This is the one path in these integrations that puts the caller's
        prompt content on the wire to a Headroom-operated host, so it is
        exactly what HEADROOM_OFFLINE exists to stop. "Opt-in by configuration"
        was the old reason for leaving it open, and it is not good enough: an
        operator who sets an air-gap switch is overriding earlier configuration
        on purpose. The refusal is loud rather than a silent fall-through to
        local compression, because silently compressing locally would hide the
        fact that the deployment is no longer doing what it was configured to do.
        """
        guard_egress("Headroom Cloud compression", self._api_url)
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    "httpx is required for Headroom Cloud mode: pip install httpx"
                ) from e
            self._client = httpx.AsyncClient(timeout=30.0)

        client = self._client
        assert client is not None
        resp = await client.post(
            f"{self._api_url}/v1/saas/compress",
            headers={
                "X-Headroom-Key": self._api_key,
                "Content-Type": "application/json",
            },
            content=json.dumps(
                {
                    "messages": messages,
                    "model": model or _DEFAULT_MODEL,
                    "model_limit": self._model_limit,
                }
            ),
        )

        if resp.status_code != 200:
            # The response body can echo request content: log metadata only, at
            # every level.
            self._log.warning(
                "Headroom Cloud API error: HTTP %d from %s; request sent uncompressed",
                resp.status_code,
                redact_url(self._api_url),
            )
            self._log.debug(
                "Headroom Cloud API error response: content-type=%s, %d bytes",
                _media_type(resp.headers.get("content-type", "")),
                len(resp.content),
            )
            return None

        result: dict[str, Any] = resp.json()
        return result
