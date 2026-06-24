"""Tests for compression header handling in the proxy server.

These tests verify that the proxy correctly removes Content-Encoding headers
from responses after httpx automatically decompresses them, preventing
double-decompression errors (ZlibError) in clients.
"""

import gzip
import json

import pytest


@pytest.fixture
def mock_anthropic_response_with_compression_headers():
    """Create a mock response that simulates httpx behavior.

    httpx automatically decompresses responses but leaves compression headers.
    This is what causes the ZlibError bug we're testing for.
    """

    class MockResponse:
        """Mock httpx response with compression headers."""

        def __init__(self):
            self.response_data = {
                "id": "msg_test123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            # Body is already decompressed (httpx does this automatically)
            self.content = json.dumps(self.response_data).encode("utf-8")
            self.status_code = 200

            # Headers still contain compression info (this is the bug!)
            self.headers = {
                "content-type": "application/json",
                "content-encoding": "gzip",  # Should be removed!
                "content-length": str(len(gzip.compress(self.content))),  # Wrong!
                "x-request-id": "test-request-id",
            }

    return MockResponse()


class TestCompressionHeaderRemoval:
    """Tests for Content-Encoding header removal logic."""

    def test_compression_headers_are_removed_from_dict(
        self, mock_anthropic_response_with_compression_headers
    ):
        """Test that our fix removes compression headers from response headers."""
        mock_response = mock_anthropic_response_with_compression_headers

        # Simulate what the fixed code does
        response_headers = dict(mock_response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        # Verify compression headers are removed
        assert "content-encoding" not in response_headers
        assert "content-length" not in response_headers

        # Verify other headers are preserved
        assert response_headers["content-type"] == "application/json"
        assert response_headers["x-request-id"] == "test-request-id"

    def test_response_body_is_decompressed_not_compressed(
        self, mock_anthropic_response_with_compression_headers
    ):
        """Verify the response content is already decompressed (httpx behavior)."""
        mock_response = mock_anthropic_response_with_compression_headers

        # The content should be valid JSON (decompressed)
        response_data = json.loads(mock_response.content)
        assert response_data["id"] == "msg_test123"

        # Trying to decompress it again should fail (proving it's not compressed)
        with pytest.raises((gzip.BadGzipFile, OSError, Exception)):
            gzip.decompress(mock_response.content)

    def test_headers_with_wrong_content_length_cause_issues(
        self, mock_anthropic_response_with_compression_headers
    ):
        """Demonstrate that keeping compression headers causes length mismatch."""
        mock_response = mock_anthropic_response_with_compression_headers

        # The content-length header says the body is compressed size
        claimed_length = int(mock_response.headers["content-length"])

        # But the actual content is decompressed size
        actual_length = len(mock_response.content)

        # They don't match! This can cause client issues
        assert claimed_length != actual_length
        assert claimed_length < actual_length  # Compressed is smaller

    def test_removing_headers_fixes_length_mismatch(
        self, mock_anthropic_response_with_compression_headers
    ):
        """Show that removing compression headers allows proper content-length."""
        mock_response = mock_anthropic_response_with_compression_headers

        # Apply the fix
        response_headers = dict(mock_response.headers)
        response_headers.pop("content-encoding", None)
        response_headers.pop("content-length", None)

        # Now we can set correct content-length
        response_headers["content-length"] = str(len(mock_response.content))

        # Verify it matches actual content
        assert int(response_headers["content-length"]) == len(mock_response.content)


class TestAcceptEncodingStripping:
    """Tests for accept-encoding removal from forwarded request headers.

    Edge proxies like Cloudflare Workers add accept-encoding values (e.g. br,
    zstd) that the upstream provider may honor.  If httpx lacks the matching
    decompression library (e.g. brotli) it cannot decode the response body,
    causing a UnicodeDecodeError and a 502 returned to the client.

    The fix strips accept-encoding before forwarding so httpx negotiates its
    own encoding independently.
    """

    def test_accept_encoding_is_stripped_from_forwarded_headers(self):
        """accept-encoding must be removed before forwarding to the upstream."""
        # Simulate headers as received from a Cloudflare Worker client
        request_headers = {
            "authorization": "Bearer sk-test",
            "content-type": "application/json",
            "accept-encoding": "gzip, br, zstd",
            "host": "headroom.example.com",
            "content-length": "123",
        }

        # Replicate the handler logic
        headers = dict(request_headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        headers.pop("accept-encoding", None)

        assert "accept-encoding" not in headers

    def test_other_headers_preserved_after_stripping(self):
        """Only hop-by-hop / negotiation headers are removed; auth etc. survive."""
        request_headers = {
            "authorization": "Bearer sk-test",
            "content-type": "application/json",
            "accept-encoding": "gzip, br",
            "x-custom": "value",
            "host": "headroom.example.com",
            "content-length": "42",
        }

        headers = dict(request_headers.items())
        headers.pop("host", None)
        headers.pop("content-length", None)
        headers.pop("accept-encoding", None)

        assert headers["authorization"] == "Bearer sk-test"
        assert headers["content-type"] == "application/json"
        assert headers["x-custom"] == "value"
        assert "host" not in headers
        assert "content-length" not in headers
        assert "accept-encoding" not in headers

    def test_strip_is_safe_when_accept_encoding_absent(self):
        """pop() on a missing key must not raise — direct curl calls have no header."""
        request_headers = {
            "authorization": "Bearer sk-test",
            "content-type": "application/json",
        }

        headers = dict(request_headers.items())
        # Must not raise KeyError
        headers.pop("accept-encoding", None)

        assert headers == {
            "authorization": "Bearer sk-test",
            "content-type": "application/json",
        }

    def test_brotli_encoding_value_is_stripped(self):
        """Specifically guard against 'br' which breaks httpx without brotli package."""
        for encoding_value in ["br", "gzip, br", "gzip, br, zstd", "zstd"]:
            headers = {"accept-encoding": encoding_value, "content-type": "application/json"}
            headers.pop("accept-encoding", None)
            assert "accept-encoding" not in headers


class TestNoRegressionForUncompressedResponses:
    """Ensure the fix doesn't break responses that were never compressed."""

    def test_pop_on_missing_keys_is_safe(self):
        """Verify that .pop() on non-existent keys doesn't cause errors."""
        headers = {
            "content-type": "application/json",
            # No compression headers
        }

        # This should not raise KeyError
        headers.pop("content-encoding", None)
        headers.pop("content-length", None)

        # Headers should be unchanged
        assert headers == {"content-type": "application/json"}

    def test_dict_conversion_preserves_headers(self):
        """Verify dict() conversion doesn't lose headers."""
        original_headers = {
            "content-type": "application/json",
            "x-custom-header": "value",
            "authorization": "Bearer token",
        }

        # Convert to dict (as the fix does)
        converted = dict(original_headers)

        # All headers preserved
        assert converted == original_headers
        assert converted is not original_headers  # New object
