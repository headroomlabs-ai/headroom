"""MiniMax handler mixin for HeadroomProxy.

Provides a thin provider-specific wrapper around the Anthropic
handler so MiniMax traffic is recorded with ``provider="minimax"``
instead of ``provider="anthropic"``. The wire format is identical to
Anthropic (MiniMax exposes an Anthropic-compatible /v1/messages API),
so we delegate the heavy lifting to the existing AnthropicHandlerMixin.

Why this exists:
- Headroom's cost tracker keys everything on ``provider``. Without
  this shim, M3/M2.7 traffic buckets under Anthropic and the per-model
  breakdown in the dashboard is wrong.
- SmartCrusher and cache alignment work the same on MiniMax wire
  format, but their billing-side accounting needs the right provider.

Differences from AnthropicProvider:
- Auth: MiniMax accepts a per-session JWT via the ``Token:`` header
  (Mavis Code managed provider). Handled by Mavis Code at the client
  side; Headroom passes the header through.
- Pricing: MiniMax has its own price table in
  ``headroom/providers/minimax.py`` (``MODEL_INPUT_COST`` /
  ``MODEL_OUTPUT_COST``).
- Default base URL: ``https://api.minimaxi.com/anthropic`` for the
  direct Anthropic-compat API, or
  ``https://agent.minimax.io/mavis/api/v1/llm`` for the Mavis Code
  gateway.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import Response, StreamingResponse

from headroom.providers.minimax import MiniMaxProvider
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

logger = logging.getLogger("headroom.proxy.minimax")


class MiniMaxHandlerMixin:
    """Mixin providing MiniMax-specific proxy handler for HeadroomProxy.

    Routes traffic to the Anthropic handler (wire-compatible) but
    overrides the provider name and cost table so M3/M2.7 traffic
    is bucketed correctly in the dashboard.
    """

    @staticmethod
    def _is_minimax_model(model: str) -> bool:
        """Return True if the given model name belongs to MiniMax.

        Accepts both bare (``MiniMax-M3``) and prefixed (``minimax/MiniMax-M3``)
        forms. Conservative: returns False for unknown models so
        Anthropic traffic is never accidentally routed here.
        """
        if not model:
            return False
        m = model.strip().lower()
        if m.startswith("minimax/"):
            return True
        return m.startswith("minimax-m") or "minimax-m" in m

    @staticmethod
    def _strip_minimax_prefix(model: str) -> str:
        """Strip the ``minimax/`` provider prefix from a model name."""
        if not model:
            return model
        if "/" in model:
            head, _, tail = model.partition("/")
            if head.lower() == "minimax":
                return tail
        return model

    async def handle_minimax_messages(
        self,
        request: "Request",
        upstream_base_url: str | None = None,
        provider_name: str = "minimax",
        model_override: str | None = None,
        force_stream: bool = False,
    ) -> "Response | StreamingResponse":
        """Handle ``POST /v1/messages`` for MiniMax traffic.

        Delegates to :meth:`AnthropicHandlerMixin.handle_anthropic_messages`
        because the wire format is identical, but:

        - Sets ``provider_name="minimax"`` so the request outcome is
          recorded with the correct provider and the dashboard
          per-model breakdown attributes savings to MiniMax, not
          Anthropic.
        - Strips the ``minimax/`` prefix from the model name before
          forwarding upstream, since the MiniMax gateway expects
          bare model names (``MiniMax-M3``, not ``minimax/MiniMax-M3``).

        Note on upstream URL resolution:
            We do NOT pass a default ``upstream_base_url`` here. The
            delegate falls back to ``self.ANTHROPIC_API_URL``, which is
            set at proxy construction time from the
            ``ANTHROPIC_TARGET_API_URL`` env var (or the operator's CLI
            flag). This keeps a single source of truth for upstream
            routing — operators pin the Mavis Code gateway URL once,
            in the LaunchAgent plist, and both Anthropic and MiniMax
            traffic flow through it. The fork-specific
            ``MINIMAX_TARGET_API_URL`` env var is intentionally NOT
            consulted here: it would override the Mavis Code gateway
            with the direct ``api.minimaxi.com/anthropic`` upstream,
            which doesn't accept ``Token: <jwt>`` headers.
        """
        # Strip the prefix from the incoming model name so the upstream
        # gateway recognises it. Use a fresh request body if needed.
        if model_override is None:
            try:
                # Read body, mutate model, replace request._receive so
                # downstream readers see the cleaned body. Anthropic
                # handler reads request.json() lazily — we patch the
                # cached body if Starlette has already buffered it.
                body_bytes = await request.body()
                parsed = json.loads(body_bytes or b"{}")
                if isinstance(parsed, dict) and "model" in parsed:
                    parsed["model"] = self._strip_minimax_prefix(parsed["model"])
                    new_body = json.dumps(parsed).encode()
                    # Starlette caches the body; replace the cached value
                    # via _body. This is the same trick the
                    # mini-headroom proxy uses for the minimax
                    # auth shim.
                    try:
                        request._body = new_body  # type: ignore[attr-defined]
                    except AttributeError:
                        # Fallback: rely on Anthropic handler parsing the
                        # body and patching model via model_override.
                        model_override = parsed["model"]
            except (json.JSONDecodeError, ValueError):
                # Malformed body — let the Anthropic handler surface the
                # error to the client.
                logger.warning("minimax: could not parse body for model strip")

        # Delegate. We pass model_override so the Anthropic handler
        # sees the cleaned model name even if the body-patch failed.
        # upstream_base_url is NOT passed: delegate falls back to
        # self.ANTHROPIC_API_URL (set from ANTHROPIC_TARGET_API_URL env var).
        return await AnthropicHandlerMixin.handle_anthropic_messages(
            self,
            request,
            provider_name=provider_name,  # "minimax" — overrides default
            model_override=model_override,
            force_stream=force_stream,
        )


__all__ = [
    "MiniMaxHandlerMixin",
    "MiniMaxProvider",
]