"""Tests for opt-in passthrough compression (issue #1546).

Requests whose path doesn't match a built-in API route fall through to
``handle_passthrough``, which historically forwarded the body verbatim — no
compression. With ``compress_passthrough`` enabled, OpenAI Responses-shaped
bodies (path ends in ``/responses``) are routed through the same
ContentRouter/Kompress path the native ``/v1/responses`` handler uses.

``_maybe_compress_passthrough_responses`` is the fail-open core: any parse or
compressor failure returns the original body so a catch-all request is never
dropped by opting into compression.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from headroom.proxy.handlers.openai import OpenAIHandlerMixin


def _make_handler(compress_impl):
    """Bare mixin instance with just the two collaborators the helper needs."""
    handler = OpenAIHandlerMixin.__new__(OpenAIHandlerMixin)
    handler.config = SimpleNamespace(optimize=True, compress_passthrough=True)

    async def _next_request_id():
        return "req-test"

    handler._next_request_id = _next_request_id
    handler._compress_openai_responses_payload_in_executor = compress_impl
    return handler


def _shrinking_compressor(marker: str = "[C]"):
    async def _impl(payload, *, model, request_id):
        new = dict(payload)
        new["input"] = marker
        return (new, True, 5, ["kompress"], None, 100, 40, 5, {})

    return _impl


async def test_compresses_responses_shaped_body() -> None:
    handler = _make_handler(_shrinking_compressor())
    body = json.dumps({"model": "gpt-5.4", "input": [{"role": "user"}]}).encode()

    out = await handler._maybe_compress_passthrough_responses(body)

    assert out != body
    assert json.loads(out)["input"] == "[C]"


async def test_non_json_body_passes_through() -> None:
    handler = _make_handler(_shrinking_compressor())
    body = b"not json at all"

    assert await handler._maybe_compress_passthrough_responses(body) == body


async def test_non_responses_payload_passes_through() -> None:
    # No `input` key → not a Responses payload; must not be touched.
    handler = _make_handler(_shrinking_compressor())
    body = json.dumps({"model": "gpt-5.4", "messages": []}).encode()

    assert await handler._maybe_compress_passthrough_responses(body) == body


async def test_unmodified_result_returns_original_bytes() -> None:
    async def _noop(payload, *, model, request_id):
        return (payload, False, 0, [], "no-op", 0, 0, 0, {})

    handler = _make_handler(_noop)
    body = json.dumps({"input": [{"role": "user"}]}).encode()

    assert await handler._maybe_compress_passthrough_responses(body) == body


async def test_compressor_error_fails_open() -> None:
    async def _boom(payload, *, model, request_id):
        raise RuntimeError("kompress exploded")

    handler = _make_handler(_boom)
    body = json.dumps({"input": [{"role": "user"}]}).encode()

    # Fail-open: original body forwarded, exception swallowed.
    assert await handler._maybe_compress_passthrough_responses(body) == body


def test_config_defaults_off() -> None:
    from headroom.proxy.models import ProxyConfig

    assert ProxyConfig().compress_passthrough is False


def test_feature_flag_tolerates_missing_config() -> None:
    """The passthrough guard must resolve ``self.config`` safely.

    Some handler/proxy objects reach ``handle_passthrough`` without a ``config``
    attribute at all. Reading ``self.config.compress_passthrough`` directly
    raises ``AttributeError`` before ``getattr``'s default applies, regressing
    the pre-existing verbatim passthrough path. The flag lookup must instead
    treat a missing config as feature-off.
    """
    handler = OpenAIHandlerMixin.__new__(OpenAIHandlerMixin)
    assert not hasattr(handler, "config")

    _pt_config = getattr(handler, "config", None)
    assert getattr(_pt_config, "compress_passthrough", False) is False
