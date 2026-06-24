"""Phase G PR-G3 (P4-45) — request-logger image base64 redaction.

The Headroom proxy logs LLM request/response payloads to a JSONL feed
when ``log_full_messages=True``. Vision-shape requests (Anthropic
``image`` blocks; OpenAI ``image_url`` data URLs) carry base64-encoded
binary payloads that can be multiple megabytes each. Phase G PR-G3
replaces those over-threshold strings with a size-only placeholder so
the JSONL and the in-memory deque stay bounded.

These tests cover the redaction primitive in isolation (no proxy
required). The Rust ``proxy_image_generation_call_log_redacted_total``
counter is exercised by the Rust integration suite; this file owns the
Python contract.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile

import pytest

from headroom.proxy.models import RequestLog
from headroom.proxy.request_logger import (
    IMAGE_BASE64_REDACT_THRESHOLD_BYTES,
    IMAGE_BASE64_REPLACEMENT_TEMPLATE,
    RequestLogger,
    redact_image_base64,
    redactions_total,
)


def _big_base64(byte_size: int) -> str:
    """Produce a deterministic base64 string of exactly ``byte_size`` chars."""
    raw = os.urandom(max(byte_size, 1))
    encoded = base64.b64encode(raw).decode("ascii")
    return encoded[:byte_size].ljust(byte_size, "A")


def _make_request_log(*, request_messages=None, response_content=None) -> RequestLog:
    return RequestLog(
        request_id="req-test",
        timestamp="2026-05-21T00:00:00Z",
        provider="anthropic",
        model="claude-3-5-sonnet-20240620",
        input_tokens_original=100,
        input_tokens_optimized=100,
        output_tokens=20,
        tokens_saved=0,
        savings_percent=0.0,
        optimization_latency_ms=0.0,
        total_latency_ms=10.0,
        tags={},
        cache_hit=False,
        transforms_applied=[],
        request_messages=request_messages,
        response_content=response_content,
    )


def test_large_base64_truncated():
    """A multi-MB base64 payload in a request message is replaced with
    a size-only placeholder. This is the load-bearing assertion from
    REALIGNMENT/09-phase-G-rtk-observability.md:155."""
    big = _big_base64(IMAGE_BASE64_REDACT_THRESHOLD_BYTES * 4)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": big,
                    },
                },
            ],
        }
    ]
    redacted = redact_image_base64(messages)
    payload = redacted[0]["content"][1]["source"]["data"]
    assert IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=len(big)) == payload
    # The non-image fields must survive verbatim — redaction must
    # never perturb the structure outside the targeted payload.
    assert redacted[0]["content"][0] == {"type": "text", "text": "describe this image"}


def test_short_base64_passes_through():
    """A short base64-looking string (e.g. a 64-byte signature, a tool
    ``id``) must NOT be redacted — the threshold gates real image
    payloads against legitimate small strings."""
    short = _big_base64(64)
    assert len(short) < IMAGE_BASE64_REDACT_THRESHOLD_BYTES
    messages = [{"role": "user", "content": [{"type": "text", "text": short}]}]
    redacted = redact_image_base64(messages)
    assert redacted[0]["content"][0]["text"] == short


def test_data_url_redacted():
    """OpenAI vision shape: ``data:image/png;base64,<payload>`` URLs
    are redacted when the payload is over the threshold."""
    payload = _big_base64(IMAGE_BASE64_REDACT_THRESHOLD_BYTES * 2)
    data_url = f"data:image/png;base64,{payload}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            ],
        }
    ]
    redacted = redact_image_base64(messages)
    final_url = redacted[0]["content"][0]["image_url"]["url"]
    assert final_url == IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=len(data_url))


def test_redact_idempotent():
    """Applying redaction twice yields the same structure — the
    placeholder is short enough to stay below the threshold so the
    second pass is a no-op. The ``data`` key is one of the
    image-bearing field names so a big string inside redacts."""
    big = _big_base64(IMAGE_BASE64_REDACT_THRESHOLD_BYTES * 3)
    once = redact_image_base64({"data": big})
    twice = redact_image_base64(once)
    assert once == twice


def test_logger_writes_redacted_payload_to_jsonl():
    """End-to-end: writing a RequestLog with a big base64 payload via
    ``RequestLogger.log`` produces a JSONL line that does NOT contain
    the raw payload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "requests.jsonl")
        logger = RequestLogger(log_file=log_path, log_full_messages=True)
        big = _big_base64(IMAGE_BASE64_REDACT_THRESHOLD_BYTES * 5)
        entry = _make_request_log(
            request_messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "data": big},
                        }
                    ],
                }
            ]
        )
        logger.log(entry)
        with open(log_path) as f:
            line = f.read().strip()
        parsed = json.loads(line)
        # The raw base64 string must NOT appear in the JSONL line.
        assert big not in line, "raw base64 leaked into JSONL"
        # The placeholder must appear.
        assert "image:base64-redacted" in line
        # The structure must remain intact.
        data_field = parsed["request_messages"][0]["content"][0]["source"]["data"]
        assert data_field == IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=len(big))


def test_response_content_bare_base64_passes_through():
    """M2 remediation: a bare base64-shaped string in
    ``response_content`` is NOT redacted. The earlier "density
    heuristic" over-fired on encrypted blobs, signed tokens,
    minified JSON, and tool outputs. The new contract: only
    redact strings inside known image-bearing JSON paths OR
    strings starting with ``data:image/``."""
    logger = RequestLogger(log_file=None, log_full_messages=True)
    big = _big_base64(IMAGE_BASE64_REDACT_THRESHOLD_BYTES * 3)
    entry = _make_request_log(response_content=big)
    logger.log(entry)
    recent = logger.get_recent_with_messages(n=1)
    # Verbatim — no redaction applied.
    assert recent[0]["response_content"] == big


def test_response_content_data_image_url_redacted():
    """When ``response_content`` does start with ``data:image/`` —
    e.g. a tool wrote an image back via a data URL — redaction
    still fires."""
    logger = RequestLogger(log_file=None, log_full_messages=True)
    payload = _big_base64(IMAGE_BASE64_REDACT_THRESHOLD_BYTES * 2)
    data_url = f"data:image/png;base64,{payload}"
    entry = _make_request_log(response_content=data_url)
    logger.log(entry)
    recent = logger.get_recent_with_messages(n=1)
    assert recent[0]["response_content"] == IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(
        n=len(data_url)
    )


def test_non_image_path_base64_passes_through():
    """M2: a big base64-shaped string at a non-image-bearing key
    (e.g. an encrypted blob under ``signature`` or a tool output
    under ``arguments``) is NOT redacted."""
    big = _big_base64(IMAGE_BASE64_REDACT_THRESHOLD_BYTES * 2)
    payload = {
        "tool_use_id": "tool_xyz",
        "signature": big,  # NOT an image-bearing key
        "arguments": big,  # NOT an image-bearing key
    }
    redacted = redact_image_base64(payload)
    assert redacted["signature"] == big
    assert redacted["arguments"] == big


def test_image_path_redacts_without_density_check():
    """M2: once inside an image-bearing JSON path (e.g.
    ``source.data``), a sufficiently-long string is redacted
    regardless of its character density. Real images may not be
    base64 only — they may be webp / avif transcoded with
    different alphabets — but we still want them redacted to
    keep logs bounded."""
    big = "x" * (IMAGE_BASE64_REDACT_THRESHOLD_BYTES * 2)  # NOT base64
    payload = {"source": {"type": "base64", "data": big}}
    redacted = redact_image_base64(payload)
    expected_bytes = len(big.encode("utf-8"))
    assert redacted["source"]["data"] == IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=expected_bytes)


def test_byte_count_label_is_utf8_bytes_not_chars():
    """M5: the ``bytes=`` label is the UTF-8 byte length of the
    redacted string, not the character count. For ASCII base64
    payloads the two coincide (so existing tests still pass), but
    a non-ASCII string under an image-bearing key reports byte
    length faithfully."""
    # 3-byte UTF-8 character ('€' = U+20AC) repeated; the
    # character count is half the byte count.
    chars = "€" * (IMAGE_BASE64_REDACT_THRESHOLD_BYTES + 1)
    payload = {"data": chars}
    redacted = redact_image_base64(payload)
    char_count = len(chars)
    byte_count = len(chars.encode("utf-8"))
    assert byte_count == 3 * char_count
    assert redacted["data"] == IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=byte_count)


def test_redactions_counter_advances():
    """``redactions_total`` increases by one per redacted payload."""
    before = redactions_total()
    big = _big_base64(IMAGE_BASE64_REDACT_THRESHOLD_BYTES * 2)
    redact_image_base64({"data": big})
    after = redactions_total()
    assert after == before + 1


def test_none_payload_safe():
    """``RequestLogger.log`` must not crash when ``request_messages``
    or ``response_content`` is None — many requests have neither."""
    logger = RequestLogger(log_file=None, log_full_messages=True)
    entry = _make_request_log(request_messages=None, response_content=None)
    logger.log(entry)  # must not raise
    assert len(logger.get_recent(n=1)) == 1


@pytest.mark.parametrize(
    "value",
    [
        12345,
        None,
        True,
        3.14,
        b"not-a-string",
    ],
)
def test_non_string_values_pass_through(value):
    """The redactor walks JSON-ish structures only; primitives that
    aren't strings must round-trip verbatim."""
    assert redact_image_base64(value) == value
