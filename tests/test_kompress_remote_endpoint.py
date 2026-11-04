"""Endpoint resolution for bring-your-own Kompress deployments.

The load-bearing test here is the first one: an operator with only
``HEADROOM_KOMPRESS_ENDPOINT`` set must get the exact same request as before
these knobs existed. Everything else is additive.
"""

from __future__ import annotations

import pytest

from headroom.transforms.kompress_remote import (
    DEFAULT_ENDPOINT_PATH,
    RemoteKompressCompressor,
    parse_endpoint_headers,
)


class TestNoRegressionForExistingDeployments:
    """Modal users set one env var and must be unaffected."""

    def test_default_appends_compress(self):
        c = RemoteKompressCompressor(endpoint="https://acme--kompress.modal.run")
        assert c.url == "https://acme--kompress.modal.run/compress"

    def test_trailing_slash_does_not_double_up(self):
        c = RemoteKompressCompressor(endpoint="https://acme--kompress.modal.run/")
        assert c.url == "https://acme--kompress.modal.run/compress"

    def test_token_still_sent_as_bearer(self):
        c = RemoteKompressCompressor(endpoint="https://x.modal.run", token="secret")
        assert c._headers["authorization"] == "Bearer secret"
        assert c._headers["content-type"] == "application/json"

    def test_no_token_means_no_auth_header(self):
        """Self-hosted stacks frequently need no credential at all."""
        c = RemoteKompressCompressor(endpoint="https://ml.internal")
        assert "authorization" not in c._headers

    def test_default_path_constant_is_the_historical_value(self):
        assert DEFAULT_ENDPOINT_PATH == "/compress"


class TestSelfHostedPaths:
    """Real inference servers do not serve at /compress."""

    @pytest.mark.parametrize(
        "endpoint,path,expected",
        [
            # KServe / Seldon
            (
                "https://ml.acme.com",
                "/v1/models/kompress:predict",
                "https://ml.acme.com/v1/models/kompress:predict",
            ),
            # TorchServe
            (
                "https://torchserve.acme.com",
                "/predictions/kompress",
                "https://torchserve.acme.com/predictions/kompress",
            ),
            # SageMaker
            (
                "https://runtime.sagemaker.internal",
                "/invocations",
                "https://runtime.sagemaker.internal/invocations",
            ),
            # A leading slash is optional in the env var.
            ("https://ml.acme.com", "invocations", "https://ml.acme.com/invocations"),
            # Endpoint with its own base path, plus a suffix.
            (
                "https://gw.acme.com/kompress",
                "/compress",
                "https://gw.acme.com/kompress/compress",
            ),
        ],
    )
    def test_path_override(self, endpoint, path, expected):
        assert RemoteKompressCompressor(endpoint=endpoint, path=path).url == expected

    @pytest.mark.parametrize("empty", ["", None])
    def test_empty_path_uses_the_url_verbatim(self, empty):
        """The escape hatch: the operator supplies a complete URL.

        Without this, an endpoint that is already a full path gets /compress
        appended and 404s — and because remote Kompress fails open, that 404 is
        invisible: compression silently stops instead of erroring.
        """
        url = "https://ml.acme.com/v1/models/kompress:predict"
        assert RemoteKompressCompressor(endpoint=url, path=empty).url == url

    def test_verbatim_url_keeps_its_trailing_slash_untouched(self):
        url = "https://ml.acme.com/predict/"
        assert RemoteKompressCompressor(endpoint=url, path="").url == url


class TestCustomHeaders:
    def test_extra_headers_are_merged(self):
        c = RemoteKompressCompressor(
            endpoint="https://ml.acme.com",
            headers={"x-tenant-id": "acme", "x-env": "prod"},
        )
        assert c._headers["x-tenant-id"] == "acme"
        assert c._headers["x-env"] == "prod"
        assert c._headers["content-type"] == "application/json"

    def test_headers_can_replace_the_bearer_scheme(self):
        """A gateway wanting x-api-key should not need a new setting."""
        c = RemoteKompressCompressor(
            endpoint="https://ml.acme.com",
            token="ignored",
            headers={"authorization": "Token abc123"},
        )
        assert c._headers["authorization"] == "Token abc123"

    def test_api_key_header_without_any_token(self):
        c = RemoteKompressCompressor(endpoint="https://ml.acme.com", headers={"x-api-key": "k"})
        assert c._headers["x-api-key"] == "k"
        assert "authorization" not in c._headers


class TestHeaderParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (None, {}),
            ("", {}),
            ("   ", {}),
            ("x-api-key=abc", {"x-api-key": "abc"}),
            ("a=1,b=2", {"a": "1", "b": "2"}),
            (" a = 1 , b = 2 ", {"a": "1", "b": "2"}),
            ("malformed", {}),
            ("a=1,malformed,b=2", {"a": "1", "b": "2"}),
            ("a=", {}),
            ("=1", {}),
            # A value containing '=' (e.g. base64) must survive intact.
            ("authorization=Basic dXNlcjpwYXNz==", {"authorization": "Basic dXNlcjpwYXNz=="}),
        ],
    )
    def test_parse(self, raw, expected):
        assert parse_endpoint_headers(raw) == expected
