from __future__ import annotations

import base64
import builtins
import json
from unittest.mock import patch

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.handlers.openai import OpenAIHandlerMixin, _decode_openai_bearer_payload


def _jwt(payload: object) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def encode(part: object) -> str:
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode(header)}.{encode(payload)}."


class _ImageCompressor:
    def __init__(self, compressed_message):
        self._compressed_message = compressed_message

    def compress(self, messages, provider):  # noqa: ANN001, ANN201
        assert provider == "anthropic"
        return [self._compressed_message]


class _FreshCompressor:
    instances = 0

    def __init__(self):
        type(self).instances += 1


def test_decode_openai_bearer_payload_handles_missing_and_non_mapping_payloads() -> None:
    assert _decode_openai_bearer_payload({}) is None
    assert _decode_openai_bearer_payload({"authorization": "Basic abc"}) is None
    assert (
        _decode_openai_bearer_payload({"authorization": f"Bearer {_jwt(['not', 'a', 'dict'])}"})
        is None
    )


def test_openai_handler_prefix_helpers_cover_edge_cases() -> None:
    assert OpenAIHandlerMixin._strict_previous_turn_frozen_count([], 2) == 2
    assert (
        OpenAIHandlerMixin._strict_previous_turn_frozen_count(
            [{"role": "assistant"}, {"role": "user"}],
            0,
        )
        == 1
    )
    assert (
        OpenAIHandlerMixin._strict_previous_turn_frozen_count(
            [{"role": "user"}, {"role": "assistant"}],
            0,
        )
        == 2
    )

    original = [{"role": "system", "content": "keep"}, {"role": "user", "content": "hello"}]
    restored, changed = OpenAIHandlerMixin._restore_frozen_prefix(
        original,
        [],
        frozen_message_count=1,
    )
    assert restored == [{"role": "system", "content": "keep"}]
    assert changed == 1

    restored, changed = OpenAIHandlerMixin._restore_frozen_prefix(
        original,
        [{"role": "system", "content": "changed"}, {"role": "user", "content": "hello"}],
        frozen_message_count=1,
    )
    assert restored == original
    assert changed == 1


def test_anthropic_tool_sort_and_context_append_helpers() -> None:
    tools = [
        {"type": "function", "function": {"name": "beta"}},
        {"name": "alpha"},
        {"type": "tool"},
    ]

    sorted_tools = AnthropicHandlerMixin._sort_tools_deterministically(tools)

    assert [AnthropicHandlerMixin._tool_sort_key(tool)[0] for tool in sorted_tools] == [
        "alpha",
        "beta",
        "tool",
    ]
    assert AnthropicHandlerMixin._sort_tools_deterministically(None) is None
    assert (
        AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
            [], "ctx", frozen_message_count=0
        )
        == []
    )
    assert AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        [{"role": "user", "content": "hello"}],
        "ctx",
        frozen_message_count=0,
    ) == [{"role": "user", "content": "hello\n\nctx"}]
    # PR-A2 semantics: list-content user messages get the context appended
    # to the first text block (live-zone-tail injection).
    assert AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "ctx",
        frozen_message_count=0,
    ) == [{"role": "user", "content": [{"type": "text", "text": "hello\n\nctx"}]}]


def test_anthropic_image_compression_helper_only_rewrites_latest_eligible_turn() -> None:
    image_message = {
        "role": "user",
        "content": [{"type": "image", "source": {"type": "base64", "data": "abc"}}],
    }
    compressed = {
        "role": "user",
        "content": [{"type": "image", "source": {"type": "base64", "data": "xyz"}}],
    }

    assert (
        AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
            [],
            frozen_message_count=0,
            compressor=_ImageCompressor(compressed),
        )
        == []
    )
    assert AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        [image_message],
        frozen_message_count=1,
        compressor=_ImageCompressor(compressed),
    ) == [image_message]
    assert AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        [{"role": "assistant", "content": image_message["content"]}],
        frozen_message_count=0,
        compressor=_ImageCompressor(compressed),
    ) == [{"role": "assistant", "content": image_message["content"]}]
    assert AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        [{"role": "user", "content": "no-image"}],
        frozen_message_count=0,
        compressor=_ImageCompressor(compressed),
    ) == [{"role": "user", "content": "no-image"}]
    assert AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        [image_message],
        frozen_message_count=0,
        compressor=_ImageCompressor(image_message),
    ) == [image_message]
    assert AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        [image_message],
        frozen_message_count=0,
        compressor=_ImageCompressor(compressed),
    ) == [compressed]


def test_proxy_helper_creates_fresh_image_compressors(monkeypatch) -> None:
    from headroom.proxy import helpers

    monkeypatch.setattr(helpers, "_image_compressor_available", None)
    _FreshCompressor.instances = 0

    with patch("headroom.image.ImageCompressor", _FreshCompressor):
        first = helpers._get_image_compressor()
        second = helpers._get_image_compressor()

    assert isinstance(first, _FreshCompressor)
    assert isinstance(second, _FreshCompressor)
    assert first is not second
    assert _FreshCompressor.instances == 2


def test_proxy_helper_caches_image_stack_import_failure(monkeypatch) -> None:
    from headroom.proxy import helpers

    real_import = builtins.__import__
    calls = 0

    def fake_import(name, *args, **kwargs):  # noqa: ANN001, ANN202
        nonlocal calls
        if name == "headroom.image":
            calls += 1
            raise ImportError("image extras unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(helpers, "_image_compressor_available", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert helpers._get_image_compressor() is None
    assert helpers._get_image_compressor() is None
    assert calls == 1
    assert helpers._image_compressor_available is False


def test_anthropic_cache_delta_helpers_cover_string_list_and_role_mismatch() -> None:
    previous_original = [{"role": "user", "content": "hello"}]
    previous_forwarded = [{"role": "user", "content": "HELLO"}]

    assert AnthropicHandlerMixin._extract_cache_stable_delta(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "next"}],
        previous_original,
        previous_forwarded,
    ) == (previous_forwarded, [{"role": "assistant", "content": "next"}])
    assert (
        AnthropicHandlerMixin._extract_cache_stable_delta(
            [{"role": "assistant", "content": "hello"}],
            previous_original,
            previous_forwarded,
        )
        is None
    )

    string_suffix = AnthropicHandlerMixin._extract_cache_stable_last_message_suffix(
        [{"role": "user", "content": "hello world"}],
        previous_original,
        previous_forwarded,
    )
    assert string_suffix == ([], previous_forwarded[0], [{"role": "user", "content": " world"}])

    list_suffix = AnthropicHandlerMixin._extract_cache_stable_last_message_suffix(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            }
        ],
        [{"role": "user", "content": [{"type": "text", "text": "a"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "A"}]}],
    )
    assert list_suffix == (
        [],
        {"role": "user", "content": [{"type": "text", "text": "A"}]},
        [{"role": "user", "content": [{"type": "text", "text": "b"}]}],
    )

    assert AnthropicHandlerMixin._merge_appended_message_delta(
        {"role": "user", "content": "HELLO"},
        {"role": "user", "content": " world"},
    ) == {"role": "user", "content": "HELLO world"}
    assert AnthropicHandlerMixin._merge_appended_message_delta(
        {"role": "user", "content": [{"type": "text", "text": "A"}]},
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
    ) == {"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "b"}]}
    assert (
        AnthropicHandlerMixin._merge_appended_message_delta(
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        )
        is None
    )


def test_anthropic_assistant_message_helper_requires_assistant_role() -> None:
    assert AnthropicHandlerMixin._assistant_message_from_response_json(None) is None
    assert AnthropicHandlerMixin._assistant_message_from_response_json({"role": "user"}) is None
    assert AnthropicHandlerMixin._assistant_message_from_response_json(
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
    ) == {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
