"""Headroom behind a corporate TLS-inspection gateway ("fake Zscaler").

A local HTTPS server presents a certificate signed by a root that looks like
Zscaler's: ``O=Zscaler Inc.``, and — like the real 2014 Zscaler root —
``basicConstraints`` present but **not critical**, which Python 3.13's
``VERIFY_X509_STRICT`` rejects. These tests pin the behavior customers see:

* certifi alone rejects the chain (the original bug);
* the default trust policy plus the corporate root (``HEADROOM_CA_BUNDLE`` or
  ``NODE_EXTRA_CA_CERTS``) accepts it, with no strict-mode opt-out;
* the failure explanation names Zscaler and the fix;
* process-wide injection respects the operator's replacement bundle;
* ``wrap``'s loopback ``NO_PROXY`` merge and the new ``doctor`` rows.

The OS-store half of the default policy cannot be exercised without admin
rights to install a root, so it is covered by the ``truststore`` context type
and by the additive-bundle path, which goes through the same verifier.
"""

from __future__ import annotations

import datetime as _dt
import http.server
import json
import re
import ssl
import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from headroom.proxy import ssl_context, tls_diagnostics
from headroom.proxy.ssl_context import (
    build_httpx_verify,
    build_urlopen_context,
    build_websocket_ssl,
    cert_store_sources,
    describe_trust_policy,
    ensure_process_trust,
)

x509 = pytest.importorskip("cryptography.x509")
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

_ENV_VARS = (
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "HEADROOM_CA_BUNDLE",
    "HEADROOM_CERT_STORE",
    "HEADROOM_TLS_STRICT",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    tls_diagnostics._probe_cache.clear()
    tls_diagnostics._logged_hosts.clear()


def _name(org: str, cn: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )


@pytest.fixture(scope="module")
def fake_zscaler(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str | int]]:
    """HTTPS server on 127.0.0.1 behind a Zscaler-style re-signing root."""
    d: Path = tmp_path_factory.mktemp("fake-zscaler")
    now = _dt.datetime.now(_dt.timezone.utc)

    root_key = ec.generate_private_key(ec.SECP256R1())
    root_name = _name("Zscaler Inc.", "Zscaler Root CA")
    root = (
        x509.CertificateBuilder()
        .subject_name(root_name)
        .issuer_name(root_name)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(days=1))
        .not_valid_after(now + _dt.timedelta(days=30))
        # Not critical, exactly like the real Zscaler root: the RFC 5280
        # strict check in Python 3.13 rejects this.
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=False)
        .sign(root_key, hashes.SHA256())
    )

    leaf_key = ec.generate_private_key(ec.SECP256R1())
    import ipaddress

    leaf = (
        x509.CertificateBuilder()
        .subject_name(_name("Zscaler Inc.", "127.0.0.1"))
        .issuer_name(root_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(days=1))
        .not_valid_after(now + _dt.timedelta(days=30))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .sign(root_key, hashes.SHA256())
    )

    root_pem = d / "zscaler-root.pem"
    root_pem.write_bytes(root.public_bytes(serialization.Encoding.PEM))
    leaf_pem = d / "leaf.pem"
    leaf_pem.write_bytes(
        leaf.public_bytes(serialization.Encoding.PEM)
        + leaf_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
            body = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    server_ctx.load_cert_chain(str(leaf_pem))
    server.socket = server_ctx.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            "url": f"https://127.0.0.1:{server.server_address[1]}/v1/models",
            "port": server.server_address[1],
            "root": str(root_pem),
        }
    finally:
        server.shutdown()
        server.server_close()


class TestTrustPolicy:
    def test_default_sources_are_system_and_bundled(self) -> None:
        assert cert_store_sources() == {"system", "bundled"}

    def test_default_verify_is_os_store_context(self) -> None:
        truststore = pytest.importorskip("truststore")
        verify = build_httpx_verify()
        assert isinstance(verify, truststore.SSLContext)
        assert verify.verify_mode == ssl.CERT_REQUIRED

    def test_bundled_restores_legacy_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_CERT_STORE", "bundled")
        truststore = pytest.importorskip("truststore")
        for ctx in (build_httpx_verify(), build_websocket_ssl()):
            assert isinstance(ctx, ssl.SSLContext)
            assert not isinstance(ctx, truststore.SSLContext)
            assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert build_urlopen_context() is None

    def test_unknown_tokens_fall_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_CERT_STORE", "keychain")
        assert cert_store_sources() == {"system", "bundled"}

    def test_replacement_bundle_disables_os_store(
        self, monkeypatch: pytest.MonkeyPatch, fake_zscaler: dict
    ) -> None:
        monkeypatch.setenv("SSL_CERT_FILE", fake_zscaler["root"])
        policy = describe_trust_policy()
        assert policy["system_store_active"] is False
        assert policy["replacement_bundle"]["env_var"] == "SSL_CERT_FILE"

    def test_websocket_and_urlopen_use_os_store(self) -> None:
        truststore = pytest.importorskip("truststore")
        assert isinstance(build_websocket_ssl(), truststore.SSLContext)
        assert isinstance(build_urlopen_context(), truststore.SSLContext)


class TestFakeZscaler:
    def test_certifi_alone_rejects_the_chain(
        self, monkeypatch: pytest.MonkeyPatch, fake_zscaler: dict
    ) -> None:
        """The original bug: no corporate root, every upstream call fails."""
        monkeypatch.setenv("HEADROOM_CERT_STORE", "bundled")
        with pytest.raises(httpx.ConnectError) as info:
            httpx.get(fake_zscaler["url"], verify=build_httpx_verify())
        assert tls_diagnostics.find_cert_verify_error(info.value) is not None

    @pytest.mark.parametrize("var", ["HEADROOM_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"])
    def test_default_policy_plus_corporate_root_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, fake_zscaler: dict, var: str
    ) -> None:
        """Additive root on top of the OS store, no HEADROOM_TLS_STRICT needed."""
        pytest.importorskip("truststore")
        monkeypatch.setenv(var, fake_zscaler["root"])
        resp = httpx.get(fake_zscaler["url"], verify=build_httpx_verify())
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_websocket_context_trusts_corporate_root(
        self, monkeypatch: pytest.MonkeyPatch, fake_zscaler: dict
    ) -> None:
        import socket

        monkeypatch.setenv("HEADROOM_CA_BUNDLE", fake_zscaler["root"])
        ctx = build_websocket_ssl()
        assert isinstance(ctx, ssl.SSLContext)
        with socket.create_connection(("127.0.0.1", fake_zscaler["port"]), timeout=5) as raw:
            with ctx.wrap_socket(raw, server_hostname="127.0.0.1") as tls:
                assert tls.version() is not None

    def test_probe_identifies_zscaler(self, fake_zscaler: dict) -> None:
        info = tls_diagnostics.probe_presented_chain(
            "127.0.0.1", fake_zscaler["port"], allow_private=True
        )
        assert info.reachable
        assert info.issuer is not None and "Zscaler Root CA" in info.issuer
        assert info.inspection_vendor == "Zscaler"

    def test_failure_message_names_vendor_and_fix(
        self, monkeypatch: pytest.MonkeyPatch, fake_zscaler: dict
    ) -> None:
        monkeypatch.setenv("HEADROOM_CERT_STORE", "bundled")
        with pytest.raises(httpx.ConnectError) as info:
            httpx.get(fake_zscaler["url"], verify=build_httpx_verify())
        message = tls_diagnostics.describe_upstream_failure(
            info.value, fake_zscaler["url"], allow_private=True
        )
        assert message is not None
        assert "Zscaler" in message
        assert "HEADROOM_CA_BUNDLE" in message
        assert "headroom doctor --network" in message

    def test_request_path_probe_skips_private_hosts(self, fake_zscaler: dict) -> None:
        """A per-request upstream must not reveal an internal service's certificate."""
        info = tls_diagnostics.probe_presented_chain("127.0.0.1", fake_zscaler["port"])
        assert not info.reachable
        assert info.issuer is None
        assert "non-public" in (info.error or "")

    def test_non_tls_errors_are_left_alone(self) -> None:
        err = httpx.ConnectError("connection refused")
        assert tls_diagnostics.describe_upstream_failure(err, "https://x.invalid") is None

    def test_doctor_endpoint_probe_reports_vendor(
        self, monkeypatch: pytest.MonkeyPatch, fake_zscaler: dict
    ) -> None:
        monkeypatch.setenv("HEADROOM_CERT_STORE", "bundled")
        failing = tls_diagnostics.probe_endpoint("fake", fake_zscaler["url"])
        assert not failing.ok
        assert failing.chain.inspection_vendor == "Zscaler"

        monkeypatch.setenv("HEADROOM_CA_BUNDLE", fake_zscaler["root"])
        monkeypatch.delenv("HEADROOM_CERT_STORE")
        passing = tls_diagnostics.probe_endpoint("fake", fake_zscaler["url"])
        assert passing.ok and passing.status == 200


class TestProcessTrust:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
        calls: list[int] = []
        truststore = pytest.importorskip("truststore")
        monkeypatch.setattr(truststore, "inject_into_ssl", lambda: calls.append(1))
        monkeypatch.setattr(ssl_context, "_process_trust_injected", False)
        yield calls

    def test_injects_once(self, _reset: list[int]) -> None:
        assert ensure_process_trust() is True
        assert ensure_process_trust() is True
        assert _reset == [1]

    def test_respects_replacement_bundle(
        self, monkeypatch: pytest.MonkeyPatch, fake_zscaler: dict, _reset: list[int]
    ) -> None:
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", fake_zscaler["root"])
        assert ensure_process_trust() is False
        assert _reset == []

    def test_respects_opt_out(self, monkeypatch: pytest.MonkeyPatch, _reset: list[int]) -> None:
        monkeypatch.setenv("HEADROOM_CERT_STORE", "bundled")
        assert ensure_process_trust() is False
        assert _reset == []


class TestLoopbackNoProxy:
    def test_gap_detected_only_with_proxy(self) -> None:
        assert tls_diagnostics.loopback_no_proxy_gap({}) is None
        assert (
            tls_diagnostics.loopback_no_proxy_gap({"HTTPS_PROXY": "http://proxy:8080"})
            == "HTTPS_PROXY"
        )
        assert (
            tls_diagnostics.loopback_no_proxy_gap(
                {"HTTPS_PROXY": "http://p:1", "no_proxy": "localhost,127.0.0.1"}
            )
            is None
        )

    def test_wrap_merges_both_spellings(self) -> None:
        from headroom.cli.wrap import _ensure_loopback_no_proxy

        env = {"HTTPS_PROXY": "http://p:1", "NO_PROXY": "corp.internal", "no_proxy": ".svc"}
        written = _ensure_loopback_no_proxy(env)
        assert set(written) == {"NO_PROXY", "no_proxy"}
        assert env["NO_PROXY"] == env["no_proxy"] == "corp.internal,.svc,127.0.0.1,localhost,::1"
        assert _ensure_loopback_no_proxy(env) == []

    def test_wrap_leaves_env_alone_without_proxy(self) -> None:
        from headroom.cli.wrap import _ensure_loopback_no_proxy

        env = {"NO_PROXY": "corp.internal"}
        assert _ensure_loopback_no_proxy(env) == []
        assert env == {"NO_PROXY": "corp.internal"}


class TestDoctorRows:
    def test_trust_policy_rows(self, monkeypatch: pytest.MonkeyPatch, fake_zscaler: dict) -> None:
        from headroom.cli.doctor import PASS, WARN, check_trust_policy

        pytest.importorskip("truststore")
        assert check_trust_policy(describe_trust_policy()).status == PASS
        monkeypatch.setenv("HEADROOM_CERT_STORE", "bundled")
        assert check_trust_policy(describe_trust_policy()).status == WARN
        monkeypatch.setenv("SSL_CERT_FILE", fake_zscaler["root"])
        row = check_trust_policy(describe_trust_policy())
        assert row.status == PASS and "SSL_CERT_FILE" in row.summary

    def test_network_rows_and_it_request(
        self, monkeypatch: pytest.MonkeyPatch, fake_zscaler: dict
    ) -> None:
        from headroom.cli.doctor import FAIL, check_network_endpoints

        monkeypatch.setenv("HEADROOM_CERT_STORE", "bundled")
        report = tls_diagnostics.probe_endpoint("api.example", fake_zscaler["url"])
        rows = check_network_endpoints([report], required={"api.example"})
        assert rows[0].status == FAIL
        assert "Zscaler Root CA" in rows[0].summary
        it_row = rows[-1]
        assert it_row.name == "it request"
        assert "Zscaler" in (it_row.hint or "")
        assert "127.0.0.1" in (it_row.hint or "")

    def test_block_page_detection(self) -> None:
        html = b"<html><title>Zscaler</title>Website blocked by policy</html>"
        assert tls_diagnostics._looks_like_block_page(403, "text/html", html)
        assert not tls_diagnostics._looks_like_block_page(
            200, "text/html", b"<html>Hugging Face privacy policy</html>"
        )
        assert not tls_diagnostics._looks_like_block_page(403, "application/json", b"{}")


class TestStreamingErrorSurface:
    """The agent's error line names the TLS cause instead of a bare 502."""

    @pytest.mark.asyncio
    async def test_cert_failure_reaches_client_as_actionable_sse_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from tests.test_h2_stream_reset_retry import _mock_proxy, _run_stream

        issued_by = tls_diagnostics.ChainInfo(
            host="api.anthropic.com",
            port=443,
            reachable=True,
            issuer="O=Zscaler Inc., CN=Zscaler Intermediate Root CA",
            inspection_vendor="Zscaler",
        )
        monkeypatch.setattr(tls_diagnostics, "probe_presented_chain", lambda *a, **k: issued_by)
        cert_err = ssl.SSLCertVerificationError(
            1, "certificate verify failed: unable to get local issuer certificate"
        )
        cert_err.verify_message = "unable to get local issuer certificate"
        connect_err = httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]")
        connect_err.__cause__ = cert_err

        proxy = _mock_proxy()
        proxy.http_client.build_request = MagicMock(return_value=MagicMock())
        proxy.http_client.send = AsyncMock(side_effect=connect_err)

        result = await _run_stream(proxy)
        body = b"".join([chunk async for chunk in result.body_iterator]).decode()

        assert result.status_code == 502
        assert "Zscaler is inspecting this connection" in body
        assert "HEADROOM_CA_BUNDLE" in body
        assert re.search(r"certificate for api\.anthropic\.com \(", body)


class TestReviewFixes:
    def test_opt_out_aliases_are_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HEADROOM_CERT_STORE", "bundle")
        assert cert_store_sources() == {"bundled"}
        monkeypatch.setenv("HEADROOM_CERT_STORE", "os")
        assert cert_store_sources() == {"system"}

    def test_system_context_is_built_once_per_config(
        self, monkeypatch: pytest.MonkeyPatch, fake_zscaler: dict
    ) -> None:
        pytest.importorskip("truststore")
        ssl_context._system_ctx_cache.clear()
        first = build_websocket_ssl()
        assert build_websocket_ssl() is first
        monkeypatch.setenv("HEADROOM_CA_BUNDLE", fake_zscaler["root"])
        assert build_websocket_ssl() is not first

    def test_partial_chain_is_enabled(self) -> None:
        pytest.importorskip("truststore")
        flag = getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN", 0)
        if not flag:
            pytest.skip("OpenSSL build without partial-chain support")
        ctx = build_httpx_verify()
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.verify_flags & flag

    def test_proxy_startup_only_injects_in_cli_owned_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        truststore = pytest.importorskip("truststore")
        calls: list[int] = []
        monkeypatch.setattr(truststore, "inject_into_ssl", lambda: calls.append(1))
        monkeypatch.setattr(ssl_context, "_process_trust_injected", False)
        monkeypatch.delenv(ssl_context.PROCESS_TRUST_ENV, raising=False)
        assert ssl_context.ensure_process_trust_if_owned() is False
        assert calls == []
        monkeypatch.setenv(ssl_context.PROCESS_TRUST_ENV, "1")
        assert ssl_context.ensure_process_trust_if_owned() is True
        assert calls == [1]

    def test_concurrent_probes_share_one_handshake(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time as _time
        from concurrent.futures import ThreadPoolExecutor

        calls: list[str] = []

        def slow_probe(host: str, port: int, **_: object) -> tls_diagnostics.ChainInfo:
            calls.append(host)
            _time.sleep(0.2)
            return tls_diagnostics.ChainInfo(host=host, port=port, reachable=True)

        monkeypatch.setattr(tls_diagnostics, "_probe_uncached", slow_probe)
        with ThreadPoolExecutor(8) as pool:
            list(
                pool.map(
                    lambda _: tls_diagnostics.probe_presented_chain("api.example.com"), range(8)
                )
            )
        assert calls == ["api.example.com"]

    def test_private_allowed_probe_is_not_served_to_request_path(self, fake_zscaler: dict) -> None:
        port = fake_zscaler["port"]
        allowed = tls_diagnostics.probe_presented_chain("127.0.0.1", port, allow_private=True)
        assert allowed.issuer
        guarded = tls_diagnostics.probe_presented_chain("127.0.0.1", port)
        assert guarded.issuer is None

    def test_probe_ignores_os_proxy_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import urllib.request

        monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": "http://os-proxy:3128"})
        assert tls_diagnostics._proxy_for("api.anthropic.com") is None
        monkeypatch.setenv("HTTPS_PROXY", "http://env-proxy:8080")
        assert tls_diagnostics._proxy_for("api.anthropic.com") == "http://env-proxy:8080"

    def test_wrap_notice_goes_to_stderr(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`wrap <tool> --prepare-only` stdout is machine-read JSON; keep it clean."""
        import click

        from headroom.cli import wrap as wrap_mod

        monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
        # Registered so monkeypatch restores them: the callback writes os.environ.
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        monkeypatch.setattr(wrap_mod, "_should_purge_context_tools", lambda ctx: False)
        with click.Context(wrap_mod.wrap) as ctx:
            ctx.invoke(wrap_mod.wrap.callback)
        out, err = capsys.readouterr()
        assert out == ""
        assert "NO_PROXY" in err
