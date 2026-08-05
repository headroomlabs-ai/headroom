from __future__ import annotations

from headroom.proxy.diagnostic_decode_policy import safe_decode_for_logging
from headroom.proxy.helpers import safe_decode_for_logging as helper_safe_decode_for_logging


def test_safe_decode_for_logging_decodes_utf8() -> None:
    assert safe_decode_for_logging("hello \u2603".encode()) == "hello \u2603"


def test_safe_decode_for_logging_replaces_invalid_bytes() -> None:
    assert safe_decode_for_logging(b"ok\xffdone") == "ok\ufffddone"


def test_safe_decode_for_logging_honors_max_bytes_before_decoding() -> None:
    assert safe_decode_for_logging(b"abcdef", max_bytes=3) == "abc"


def test_helpers_safe_decode_delegates_to_policy() -> None:
    assert helper_safe_decode_for_logging(b"ok\xffdone") == safe_decode_for_logging(b"ok\xffdone")
