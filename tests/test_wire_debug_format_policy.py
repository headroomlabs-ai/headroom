from __future__ import annotations

from headroom.proxy.wire_debug_format_policy import (
    WIRE_DEBUG_NAME_MAX_CHARS,
    safe_wire_debug_name,
    wire_debug_preview,
)


def test_safe_wire_debug_name_replaces_path_unsafe_characters() -> None:
    assert safe_wire_debug_name("req/id:with spaces") == "req_id_with_spaces"


def test_safe_wire_debug_name_caps_length() -> None:
    assert safe_wire_debug_name("a" * 200) == "a" * WIRE_DEBUG_NAME_MAX_CHARS


def test_wire_debug_preview_compacts_json_like_values() -> None:
    preview = wire_debug_preview({"message": "hello\nworld", "count": 2})

    assert preview == '{"message":"hello\\nworld","count":2}'


def test_wire_debug_preview_decodes_bytes_and_truncates() -> None:
    assert wire_debug_preview(b"hello world", max_chars=8) == "hello w…"


def test_wire_debug_preview_returns_empty_string_for_none() -> None:
    assert wire_debug_preview(None) == ""
