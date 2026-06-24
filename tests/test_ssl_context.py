"""Unit tests for headroom.proxy.ssl_context.find_ca_bundle.

Covers:
- Returns None when no env var is set
- Returns an ssl.SSLContext when SSL_CERT_FILE points to a valid PEM file
- Returns an ssl.SSLContext when REQUESTS_CA_BUNDLE points to a valid PEM file
- Replacement contexts relax OpenSSL VERIFY_X509_STRICT for custom CA bundles
- Returns an ssl.SSLContext when NODE_EXTRA_CA_CERTS points to a valid PEM file
- The NODE_EXTRA_CA_CERTS SSLContext is additive: default/system roots are preserved (#998)
- Priority order: SSL_CERT_FILE beats REQUESTS_CA_BUNDLE beats NODE_EXTRA_CA_CERTS
- Nonexistent paths are skipped (returns None if all paths are missing)
"""

from __future__ import annotations

import ssl

import pytest

from headroom.proxy import ssl_context
from headroom.proxy.ssl_context import find_ca_bundle

# Minimal self-signed CA certificate (PEM) used only to verify that
# load_verify_locations accepts the file.  Generated offline; never used
# for real TLS handshakes in these tests.
_SELF_SIGNED_CA_PEM = b"""\
-----BEGIN CERTIFICATE-----
MIIDFzCCAf+gAwIBAgIUWP49K8QzU5B68/BZSmeqPCDaBoQwDQYJKoZIhvcNAQEL
BQAwGzEZMBcGA1UEAwwQaGVhZHJvb20tdGVzdC1jYTAeFw0yNjA2MDgxNDIwMzFa
Fw0zNjA2MDUxNDIwMzFaMBsxGTAXBgNVBAMMEGhlYWRyb29tLXRlc3QtY2EwggEi
MA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCvTqYZXAhet9yw1n4cFeC8HosC
1Od/bibXyW7ko7aOuuzUT7B9l7MwDfgrE2mjHecoSe2qbknFcv6hxbYojh4J7C8r
UPgCA2QTtU3pBxQdwO156YAOmFPuBFPb19NAErOVlnHCU+NXCVSsE5y+AJjM161S
W0HnZgO8OADZHBs5jSAGDE3ymMw+8xpuvRKJnuvK0Tcu6bOqOTMbnggwmPBZBBLW
PrurPTN0vV9C2oyHA1tXgEJyYtEPoMfaqyE80GxYeUujt9EQWrLp+3k8ufB/yJ1b
DaSrH0GZYx2HUn0p1mqWzXcKZrSrL1o+38gCmCivG0movXt6z1tUly8mTGz/AgMB
AAGjUzBRMB0GA1UdDgQWBBTyJ8OWE/bpWbKM3SB52P+9DhGN/TAfBgNVHSMEGDAW
gBTyJ8OWE/bpWbKM3SB52P+9DhGN/TAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3
DQEBCwUAA4IBAQAb44h2gg9wWU5todvwSXVAlBb/WZD1l/NG2PeTsGoH7xqmfgq9
DxV6tvoIuDlu6OKz071ljSqRh0Mesh1ma1cj6snsc/jqgsakSlcOpOCsrTCvw2DB
2oTztHnO4PiZAPtuKiawhVQpJfEna9/xOkbalazecSGngtSzd/oIJEXe299hE1/1
Tfx2hBGZ0UogmREaXFi099rmaueZ0HIBn51b3kYqc7of5TI0fHwSHF4GdXXs2OZi
6EVQWhKx5nQbklTYP5/ge9olEIsMdGqJEiz7WfSC6QBBgvoYyH596GiSGRZcX67p
kF9agIt8Q8t/2kviMn2roInGTwTyPYOEQV0m
-----END CERTIFICATE-----
"""


@pytest.fixture()
def ca_pem_file(tmp_path):
    """Write the self-signed CA PEM to a temp file and return its path."""
    p = tmp_path / "ca.pem"
    p.write_bytes(_SELF_SIGNED_CA_PEM)
    return str(p)


def _clean_env(monkeypatch):
    """Remove all three CA-bundle env vars so tests start from a clean state."""
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"):
        monkeypatch.delenv(var, raising=False)


class FakeSSLContext:
    def __init__(self, verify_flags: int = 0) -> None:
        self.verify_flags = verify_flags
        self.loaded_cafile: str | None = None
        self.alpn_protocols: list[str] | None = None

    def load_verify_locations(self, *, cafile: str) -> None:
        self.loaded_cafile = cafile

    def set_alpn_protocols(self, protocols: list[str]) -> None:
        self.alpn_protocols = protocols


class TestFindCaBundleNoEnvVars:
    def test_returns_none_when_no_env_var_set(self, monkeypatch):
        _clean_env(monkeypatch)
        assert find_ca_bundle() is None


class TestFindCaBundleWithValidPem:
    def test_ssl_cert_file_returns_ssl_context(self, monkeypatch, ca_pem_file):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SSL_CERT_FILE", ca_pem_file)
        ctx = find_ca_bundle()
        assert isinstance(ctx, ssl.SSLContext)

    def test_requests_ca_bundle_returns_ssl_context(self, monkeypatch, ca_pem_file):
        _clean_env(monkeypatch)
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", ca_pem_file)
        ctx = find_ca_bundle()
        assert isinstance(ctx, ssl.SSLContext)

    def test_replacement_ca_context_relaxes_x509_strict(self, monkeypatch, ca_pem_file):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SSL_CERT_FILE", ca_pem_file)
        strict_flag = 0x20
        created_context = FakeSSLContext(verify_flags=strict_flag | 0x100)

        def fake_create_default_context(*, cafile: str | None = None):
            assert cafile == ca_pem_file
            return created_context

        monkeypatch.setattr(ssl_context.ssl, "VERIFY_X509_STRICT", strict_flag, raising=False)
        monkeypatch.setattr(ssl_context.ssl, "create_default_context", fake_create_default_context)

        ctx = find_ca_bundle()

        assert ctx is created_context
        assert created_context.verify_flags & strict_flag == 0
        assert created_context.verify_flags & 0x100
        assert created_context.alpn_protocols == ["h2", "http/1.1"]

    def test_node_extra_ca_certs_returns_ssl_context(self, monkeypatch, ca_pem_file):
        """NODE_EXTRA_CA_CERTS returns an SSLContext, not a bare path (#998)."""
        _clean_env(monkeypatch)
        monkeypatch.setenv("NODE_EXTRA_CA_CERTS", ca_pem_file)
        ctx = find_ca_bundle()
        assert isinstance(ctx, ssl.SSLContext)

    def test_node_extra_ca_certs_is_additive(self, monkeypatch, ca_pem_file):
        """The SSLContext must contain default/system roots plus the extra cert (#998)."""
        _clean_env(monkeypatch)
        monkeypatch.setenv("NODE_EXTRA_CA_CERTS", ca_pem_file)
        ctx = find_ca_bundle()
        assert isinstance(ctx, ssl.SSLContext)
        stats = ctx.cert_store_stats()
        # The default trust store has dozens of CAs; if only the test cert
        # were loaded (replacement), x509_ca would be 1.
        assert stats["x509_ca"] > 1


class TestFindCaBundlePriority:
    def test_ssl_cert_file_beats_requests_ca_bundle(self, monkeypatch, tmp_path):
        """SSL_CERT_FILE is used first even when REQUESTS_CA_BUNDLE is also set."""
        _clean_env(monkeypatch)
        pem1 = tmp_path / "first.pem"
        pem2 = tmp_path / "second.pem"
        pem1.write_bytes(_SELF_SIGNED_CA_PEM)
        pem2.write_bytes(_SELF_SIGNED_CA_PEM)
        monkeypatch.setenv("SSL_CERT_FILE", str(pem1))
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(pem2))
        created_context = FakeSSLContext()

        def fake_create_default_context(*, cafile: str | None = None):
            assert cafile == str(pem1)
            return created_context

        monkeypatch.setattr(ssl_context.ssl, "create_default_context", fake_create_default_context)

        assert find_ca_bundle() is created_context

    def test_ssl_cert_file_beats_node_extra_ca_certs(self, monkeypatch, tmp_path):
        """SSL_CERT_FILE takes precedence over NODE_EXTRA_CA_CERTS."""
        _clean_env(monkeypatch)
        pem = tmp_path / "ca.pem"
        pem.write_bytes(_SELF_SIGNED_CA_PEM)
        monkeypatch.setenv("SSL_CERT_FILE", str(pem))
        monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/nonexistent/node.pem")
        created_context = FakeSSLContext()

        def fake_create_default_context(*, cafile: str | None = None):
            assert cafile == str(pem)
            return created_context

        monkeypatch.setattr(ssl_context.ssl, "create_default_context", fake_create_default_context)

        assert find_ca_bundle() is created_context

    def test_requests_ca_bundle_beats_node_extra_ca_certs(self, monkeypatch, tmp_path):
        """REQUESTS_CA_BUNDLE is used before NODE_EXTRA_CA_CERTS."""
        _clean_env(monkeypatch)
        pem = tmp_path / "ca.pem"
        pem.write_bytes(_SELF_SIGNED_CA_PEM)
        monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/ssl.pem")
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(pem))
        monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/nonexistent/node.pem")
        created_context = FakeSSLContext()

        def fake_create_default_context(*, cafile: str | None = None):
            assert cafile == str(pem)
            return created_context

        monkeypatch.setattr(ssl_context.ssl, "create_default_context", fake_create_default_context)

        assert find_ca_bundle() is created_context


class TestFindCaBundleNonexistentPaths:
    def test_nonexistent_path_is_skipped(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/path/ca.pem")
        assert find_ca_bundle() is None

    def test_all_nonexistent_returns_none(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("SSL_CERT_FILE", "/no/such/file1.pem")
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/no/such/file2.pem")
        monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/no/such/file3.pem")
        assert find_ca_bundle() is None

    def test_first_nonexistent_falls_through_to_valid(self, monkeypatch, ca_pem_file):
        """When the first env var path is missing, the next valid one is used."""
        _clean_env(monkeypatch)
        monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/ssl.pem")
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", ca_pem_file)

        ctx = find_ca_bundle()

        assert isinstance(ctx, ssl.SSLContext)
