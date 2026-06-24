"""Rust binding tests for `/v1/responses` live-zone compression.

The default Python CLI runtime currently compresses Responses payloads
through CompressionUnit extraction plus ContentRouter. This module keeps
the lower-level PyO3 live-zone binding covered so Rust migration work
cannot silently break the exposed bridge.

These tests pin:

1. The binding is exposed and callable.
2. Round-trip: a body with no eligible content passes through unchanged.
3. Round-trip: a body with a compressible function-call output gets compressed.
4. Errors are non-fatal: malformed JSON / missing input array → passthrough.
5. Auth-mode parsing accepts every variant the F1 classifier produces.
"""

from __future__ import annotations

import json

import pytest


def _ensure_binding():
    """Skip if the Rust extension hasn't been built (mirrors existing pattern)."""
    try:
        from headroom._core import compress_openai_responses_live_zone

        return compress_openai_responses_live_zone
    except ImportError:
        pytest.skip("headroom._core not built — run scripts/build_rust_extension.sh")


class TestBindingExposed:
    """The pyfunction is reachable from Python."""

    def test_callable(self):
        compress = _ensure_binding()
        assert callable(compress), "compress_openai_responses_live_zone must be callable"


class TestPassthroughCases:
    """Bodies the dispatcher cannot compress should be returned byte-for-byte
    with `modified=False`. Matches the Rust proxy's `Outcome::Passthrough`
    contract."""

    def test_not_json_passthrough(self):
        compress = _ensure_binding()
        body = b"this is not JSON at all"
        out, modified, _saved, _transforms, _reason = compress(body, "payg", "gpt-4o-mini")
        assert out == body
        assert modified is False

    def test_no_input_array_passthrough(self):
        compress = _ensure_binding()
        body = json.dumps({"model": "gpt-4o-mini"}).encode()
        out, modified, _saved, _transforms, _reason = compress(body, "payg", "gpt-4o-mini")
        assert out == body
        assert modified is False

    def test_empty_input_array_passthrough(self):
        compress = _ensure_binding()
        body = json.dumps({"model": "gpt-4o-mini", "input": []}).encode()
        out, modified, _saved, _transforms, _reason = compress(body, "payg", "gpt-4o-mini")
        assert out == body
        assert modified is False

    def test_no_eligible_items_passthrough(self):
        compress = _ensure_binding()
        # Single user message under the byte threshold — no compression
        # applies, but still valid input.
        body = json.dumps(
            {
                "model": "gpt-4o-mini",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            }
        ).encode()
        out, modified, _saved, _transforms, _reason = compress(body, "payg", "gpt-4o-mini")
        assert modified is False
        # Body should be byte-equal (passthrough, not re-serialized).
        assert out == body


class TestAuthModeAccepted:
    """Every F1 AuthMode value is accepted; unrecognised falls back to
    Unknown (does not raise)."""

    @pytest.mark.parametrize(
        "auth_mode",
        ["payg", "oauth", "subscription", "unknown", "", "garbage"],
    )
    def test_accepts(self, auth_mode):
        compress = _ensure_binding()
        body = json.dumps({"model": "gpt-4o-mini", "input": []}).encode()
        # Should not raise on any string input.
        out, modified, _saved, _transforms, _reason = compress(body, auth_mode, "gpt-4o-mini")
        assert isinstance(out, bytes)
        assert modified is False


class TestModelDefault:
    """Empty `model` defaults to `headroom_core`'s `DEFAULT_MODEL`."""

    def test_empty_model_uses_default(self):
        compress = _ensure_binding()
        body = json.dumps({"input": []}).encode()
        out, modified, _saved, _transforms, _reason = compress(body, "payg", "")
        assert isinstance(out, bytes)
        assert modified is False


class TestNoExceptionsLeak:
    """The binding's contract is `never raises` (matches the Rust proxy's
    `compress_openai_responses_request` passthrough-on-error semantics).
    Pin this so future maintainers don't accidentally introduce a
    raising path."""

    def test_garbage_bytes_no_raise(self):
        compress = _ensure_binding()
        out, modified, _saved, _transforms, _reason = compress(
            b"\xff\xfe\x00\xff", "payg", "gpt-4o-mini"
        )
        assert modified is False
        assert out == b"\xff\xfe\x00\xff"

    def test_empty_body_no_raise(self):
        compress = _ensure_binding()
        out, modified, _saved, _transforms, _reason = compress(b"", "payg", "gpt-4o-mini")
        assert modified is False
        assert out == b""


class TestTelemetryFields:
    """The 4-tuple return surfaces ``tokens_saved`` (sum of
    `original_tokens − compressed_tokens` across the manifest's
    Compressed outcomes) and ``transforms_applied`` (deduplicated list
    of compressor strategy names). The Python proxy uses these to
    populate /transformations/feed and the dashboard's per-request log
    without recounting tokens. See `crates/headroom-core/src/transforms/
    live_zone.rs::CompressionManifest::tokens_saved` /
    `::transforms_applied`."""

    def test_no_change_returns_zero_savings_and_empty_transforms(self):
        compress = _ensure_binding()
        body = json.dumps({"model": "gpt-4o-mini", "input": []}).encode()
        out, modified, saved, transforms, reason = compress(body, "payg", "gpt-4o-mini")
        assert modified is False
        assert out == body
        assert saved == 0
        assert transforms == []
        assert reason == "no_eligible_items"

    def test_field_types(self):
        """Pin the wire shape so downstream callers don't break."""
        compress = _ensure_binding()
        body = json.dumps({"model": "gpt-4o-mini", "input": []}).encode()
        result = compress(body, "payg", "gpt-4o-mini")
        assert isinstance(result, tuple)
        assert len(result) == 5
        out, modified, saved, transforms, reason = result
        assert isinstance(out, bytes)
        assert isinstance(modified, bool)
        assert isinstance(saved, int)
        assert isinstance(transforms, list)
        assert all(isinstance(t, str) for t in transforms)
        assert reason is None or isinstance(reason, str)

    def test_large_local_shell_output_compresses_with_telemetry(self):
        """End-to-end check: a payload large enough to clear the
        per-item byte threshold produces ``modified=True`` plus a
        non-zero ``tokens_saved`` and a populated ``transforms``
        list. Mirrors the shape in the Rust crate's
        ``large_log_output_compressed`` test."""
        compress = _ensure_binding()
        log_body = "".join(
            f"[2024-01-01 00:00:00] INFO compile.rs:42 building module foo_{i}\n"
            for i in range(400)
        )
        assert len(log_body) > 2048
        body = json.dumps(
            {
                "model": "gpt-4o",
                "input": [
                    {
                        "type": "local_shell_call_output",
                        "call_id": "c1",
                        "output": log_body,
                    }
                ],
            }
        ).encode()
        out, modified, saved, transforms, reason = compress(body, "payg", "gpt-4o")
        assert modified is True
        assert saved > 0
        assert transforms, "expected at least one strategy in transforms"
        assert reason is None
        new_doc = json.loads(out)
        assert new_doc["input"][0]["type"] == "local_shell_call_output"
        assert len(new_doc["input"][0]["output"]) < len(log_body)
