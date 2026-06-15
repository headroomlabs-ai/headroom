"""Tests for headroom wrap agy / unwrap agy and agent-aware _inject_ssl_bypass.

TDD: written before implementation — tests should FAIL on first run.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WRAP_MODULE = "headroom.cli.wrap"


def _import_inject_ssl_bypass():
    """Import _inject_ssl_bypass fresh (avoids stale module state)."""
    import importlib

    import headroom.cli.wrap as wrap_mod
    importlib.reload(wrap_mod)
    return wrap_mod._inject_ssl_bypass  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _inject_ssl_bypass — agent-aware regression guard
# ---------------------------------------------------------------------------


class TestInjectSslBypassAgentAware:
    """Verify agent-aware behaviour without touching the old path."""

    def _get_fn(self):
        from headroom.cli.wrap import _inject_ssl_bypass
        return _inject_ssl_bypass

    # ------------------------------------------------------------------
    # agy: bypass vars MUST NOT be injected even when HEADROOM_SSL_VERIFY=false
    # ------------------------------------------------------------------

    def test_agy_does_not_set_node_tls_reject_unauthorized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "false")
        fn = self._get_fn()
        env: dict[str, str] = {}
        fn(env, agent_type="agy")
        assert "NODE_TLS_REJECT_UNAUTHORIZED" not in env

    def test_agy_does_not_set_pythonhttpsverify(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "false")
        fn = self._get_fn()
        env: dict[str, str] = {}
        fn(env, agent_type="agy")
        assert "PYTHONHTTPSVERIFY" not in env

    def test_agy_does_not_blank_ssl_cert_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "false")
        fn = self._get_fn()
        env: dict[str, str] = {"SSL_CERT_FILE": "/some/bundle.pem"}
        fn(env, agent_type="agy")
        assert env["SSL_CERT_FILE"] == "/some/bundle.pem"

    def test_agy_does_not_blank_cacert_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "false")
        fn = self._get_fn()
        env: dict[str, str] = {"CACERT_PATH": "/some/bundle.pem"}
        fn(env, agent_type="agy")
        assert env["CACERT_PATH"] == "/some/bundle.pem"

    def test_agy_does_not_blank_node_extra_ca_certs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "false")
        fn = self._get_fn()
        env: dict[str, str] = {"NODE_EXTRA_CA_CERTS": "/some/bundle.pem"}
        fn(env, agent_type="agy")
        assert env["NODE_EXTRA_CA_CERTS"] == "/some/bundle.pem"

    def test_agy_does_not_blank_curl_ca_bundle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "false")
        fn = self._get_fn()
        env: dict[str, str] = {"CURL_CA_BUNDLE": "/some/bundle.pem"}
        fn(env, agent_type="agy")
        assert env["CURL_CA_BUNDLE"] == "/some/bundle.pem"

    # ------------------------------------------------------------------
    # REGRESSION: other agent types keep byte-identical old behaviour
    # ------------------------------------------------------------------

    def test_claude_sets_node_tls_reject_unauthorized_0(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "false")
        fn = self._get_fn()
        env: dict[str, str] = {}
        fn(env, agent_type="claude")
        assert env["NODE_TLS_REJECT_UNAUTHORIZED"] == "0"

    def test_claude_sets_pythonhttpsverify_0(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "false")
        fn = self._get_fn()
        env: dict[str, str] = {}
        fn(env, agent_type="claude")
        assert env["PYTHONHTTPSVERIFY"] == "0"

    def test_claude_blanks_curl_ca_bundle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "false")
        fn = self._get_fn()
        env: dict[str, str] = {}
        fn(env, agent_type="claude")
        assert env["CURL_CA_BUNDLE"] == ""

    def test_claude_blanks_ssl_cert_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "false")
        fn = self._get_fn()
        env: dict[str, str] = {}
        fn(env, agent_type="claude")
        assert env["SSL_CERT_FILE"] == ""

    def test_default_unknown_agent_keeps_old_behaviour_when_ssl_bypass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "false")
        fn = self._get_fn()
        env: dict[str, str] = {}
        fn(env)  # no agent_type -> "unknown"
        assert env["NODE_TLS_REJECT_UNAUTHORIZED"] == "0"
        assert env["PYTHONHTTPSVERIFY"] == "0"

    def test_no_mutation_when_ssl_verify_is_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HEADROOM_SSL_VERIFY", "true")
        fn = self._get_fn()
        env: dict[str, str] = {}
        fn(env, agent_type="agy")
        assert env == {}

    def test_no_mutation_when_ssl_verify_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HEADROOM_SSL_VERIFY", raising=False)
        fn = self._get_fn()
        env: dict[str, str] = {}
        fn(env, agent_type="agy")
        assert env == {}


# ---------------------------------------------------------------------------
# headroom wrap agy — CLI integration tests
# ---------------------------------------------------------------------------


def _get_main():
    from headroom.cli.main import main
    return main


class TestWrapAgyBinaryMissing:
    """Binary-missing path must exit 1 with install hint."""

    def test_exits_1_when_agy_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("shutil.which", lambda _: None)
        runner = CliRunner()
        result = runner.invoke(_get_main(), ["wrap", "agy"])
        assert result.exit_code == 1

    def test_prints_install_hint_when_agy_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("shutil.which", lambda _: None)
        runner = CliRunner()
        result = runner.invoke(_get_main(), ["wrap", "agy"])
        assert "agy" in result.output.lower() or "install" in result.output.lower()


class TestWrapAgyRustBackendFails:
    """Rust backend must hard-fail with a clear message."""

    def _run_with_rust_backend(self, monkeypatch: pytest.MonkeyPatch, via_env: bool):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)
        if via_env:
            monkeypatch.setenv("HEADROOM_BACKEND", "rust")
        runner = CliRunner()
        args = ["wrap", "agy"] if not via_env else ["wrap", "agy"]
        if not via_env:
            args += ["--backend", "rust"]
        return runner.invoke(_get_main(), args)

    def test_rust_backend_flag_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._run_with_rust_backend(monkeypatch, via_env=False)
        assert result.exit_code == 1

    def test_rust_backend_flag_prints_clear_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._run_with_rust_backend(monkeypatch, via_env=False)
        output = result.output.lower()
        assert "rust" in output or "python" in output or "not supported" in output

    def test_rust_backend_env_exits_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._run_with_rust_backend(monkeypatch, via_env=True)
        assert result.exit_code == 1


class TestWrapAgyDisclosureBanner:
    """TLS interception disclosure banner must name the intercepted host."""

    _INTERCEPTED_HOST = "daily-cloudcode-pa.googleapis.com"

    def _invoke_agy(self, monkeypatch: pytest.MonkeyPatch, extra_args: list[str] | None = None):
        """Invoke wrap agy with servers and subprocess fully stubbed out."""
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)

        # Stub the lifecycle helper so no real servers start
        import headroom.cli.wrap as wrap_mod

        fake_servers = MagicMock()
        fake_servers.terminator.address = ("127.0.0.1", 54321)
        fake_servers.dispatch.address = ("127.0.0.1", 54322)

        def fake_start_agy_servers(ca_key, ca_cert, base_dir=None):
            return fake_servers

        monkeypatch.setattr(wrap_mod, "_start_agy_servers", fake_start_agy_servers)
        monkeypatch.setattr(wrap_mod, "_stop_agy_servers", lambda s: None)

        # Stub ensure_root_ca + build_combined_bundle
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test CA")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(key, hashes.SHA256())
        )

        monkeypatch.setattr(
            "headroom.proxy.agy_ca.ensure_root_ca",
            lambda base_dir=None: (key, cert, Path("/tmp/ca.key"), Path("/tmp/ca.crt")),
        )
        monkeypatch.setattr(
            "headroom.proxy.agy_ca.build_combined_bundle",
            lambda base_dir=None, corp_env_vars=None: Path("/tmp/bundle.pem"),
        )

        # Stub subprocess.run so agy never actually launches
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: MagicMock(returncode=0))

        runner = CliRunner()
        args = ["wrap", "agy"] + (extra_args or [])
        return runner.invoke(_get_main(), args, catch_exceptions=False)

    def test_disclosure_banner_names_intercepted_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._invoke_agy(monkeypatch)
        assert self._INTERCEPTED_HOST in result.output

    def test_disclosure_banner_mentions_no_intercept_option(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._invoke_agy(monkeypatch)
        assert "--no-intercept" in result.output

    def test_disclosure_banner_mentions_unwrap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self._invoke_agy(monkeypatch)
        assert "unwrap" in result.output.lower()


class TestWrapAgyNoIntercept:
    """--no-intercept flag must change behavior (no MITM server startup)."""

    def test_no_intercept_does_not_start_servers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)

        import headroom.cli.wrap as wrap_mod
        server_started = []

        def fake_start(ca_key, ca_cert, base_dir=None):
            server_started.append(True)
            raise AssertionError("Servers must NOT start in --no-intercept mode")

        monkeypatch.setattr(wrap_mod, "_start_agy_servers", fake_start)
        monkeypatch.setattr(wrap_mod, "_stop_agy_servers", lambda s: None)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: MagicMock(returncode=0))

        runner = CliRunner()
        runner.invoke(_get_main(), ["wrap", "agy", "--no-intercept"])
        # Must not have started servers (no AssertionError bubbled = no start call)
        assert not server_started


class TestWrapAgySignalTeardown:
    """SIGTERM during the agy run must tear the MITM servers down (and the
    pre-existing handlers must be restored afterwards)."""

    def _stub_ca(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test CA")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(key, hashes.SHA256())
        )
        monkeypatch.setattr(
            "headroom.proxy.agy_ca.ensure_root_ca",
            lambda base_dir=None: (key, cert, Path("/tmp/ca.key"), Path("/tmp/ca.crt")),
        )
        monkeypatch.setattr(
            "headroom.proxy.agy_ca.build_combined_bundle",
            lambda base_dir=None, corp_env_vars=None: Path("/tmp/bundle.pem"),
        )

    def test_sigterm_during_run_tears_down_and_restores_handlers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import signal

        import headroom.cli.wrap as wrap_mod

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)
        self._stub_ca(monkeypatch)

        fake_servers = MagicMock()
        fake_servers.terminator.address = ("127.0.0.1", 54321)
        fake_servers.dispatch.address = ("127.0.0.1", 54322)
        monkeypatch.setattr(
            wrap_mod, "_start_agy_servers", lambda ca_key, ca_cert, base_dir=None: fake_servers
        )

        stop_calls: list[object] = []
        monkeypatch.setattr(wrap_mod, "_stop_agy_servers", lambda s: stop_calls.append(s))

        captured: dict[str, object] = {}

        def fake_run(*_a, **_kw):
            # Simulate agy receiving SIGTERM mid-run: invoke the handler that
            # production installed. It must stop the servers and raise SystemExit(143).
            captured["sigterm"] = signal.getsignal(signal.SIGTERM)
            captured["sigint"] = signal.getsignal(signal.SIGINT)
            handler = captured["sigterm"]
            assert callable(handler)
            handler(signal.SIGTERM, None)  # raises SystemExit(143)
            raise AssertionError("SIGTERM handler did not raise")  # pragma: no cover

        monkeypatch.setattr("subprocess.run", fake_run)

        original_sigterm = signal.getsignal(signal.SIGTERM)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["wrap", "agy"])

        # The SIGTERM handler raised SystemExit(143) -> that is the exit code.
        assert result.exit_code == 143
        # SIGINT was delegated to agy via the ignore-child handler.
        assert captured["sigint"] is wrap_mod._ignore_child_sigint
        # The installed SIGTERM handler was a real handler (not default/ignore).
        assert captured["sigterm"] not in (signal.SIG_DFL, signal.SIG_IGN)
        # Servers were stopped (handler + finally both call _stop_agy_servers).
        assert len(stop_calls) >= 1
        # Prior SIGTERM handler restored — no leak into the host process.
        assert signal.getsignal(signal.SIGTERM) is original_sigterm


# ---------------------------------------------------------------------------
# headroom unwrap agy
# ---------------------------------------------------------------------------


class TestUnwrapAgy:
    """unwrap agy must be a safe no-op + print a status message."""

    def test_unwrap_agy_exits_0(self) -> None:
        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0

    def test_unwrap_agy_prints_status_message(self) -> None:
        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        # Should have some output acknowledging the command ran
        assert result.output.strip() != ""
