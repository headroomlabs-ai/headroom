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
        # No retrieve listener here: this test only checks the disclosure
        # banner, and a real port would trigger MCP registration against the
        # real ~/.gemini. retrieve_port=None makes agy() skip registration.
        fake_servers.retrieve_port = None

        def fake_start_agy_servers(ca_key, ca_cert, base_dir=None, *, start_retrieve=False):
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

    def test_disclosure_banner_names_every_allowlisted_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Consent surface must not understate interception: the banner must
        name EVERY host the terminator's allowlist will TLS-terminate, not
        just the primary one."""
        from headroom.proxy.agy_terminator import DEFAULT_ALLOWLIST

        result = self._invoke_agy(monkeypatch)
        for host in DEFAULT_ALLOWLIST:
            assert host in result.output, f"disclosure omits intercepted host {host}"

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

        def fake_start(ca_key, ca_cert, base_dir=None, *, start_retrieve=False):
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
        # No retrieve listener: keep this signal-teardown test focused and avoid
        # touching the real ~/.gemini via MCP registration.
        fake_servers.retrieve_port = None
        monkeypatch.setattr(
            wrap_mod,
            "_start_agy_servers",
            lambda ca_key, ca_cert, base_dir=None, *, start_retrieve=False: fake_servers,
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
    """unwrap agy reverts GEMINI.md block and MCP registration."""

    def test_unwrap_agy_exits_0(self) -> None:
        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0

    def test_unwrap_agy_prints_status_message(self) -> None:
        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        # Should have some output acknowledging the command ran
        assert result.output.strip() != ""


# ---------------------------------------------------------------------------
# T9: GEMINI.md block injection / removal
# ---------------------------------------------------------------------------


class TestGeminiMdBlock:
    """_inject_gemini_md_block and _remove_gemini_md_block preserve user content."""

    def _get_helpers(self):
        from headroom.cli.wrap import (
            _AGY_GEMINI_BLOCK_END,
            _AGY_GEMINI_BLOCK_START,
            _inject_gemini_md_block,
            _remove_gemini_md_block,
        )
        return _inject_gemini_md_block, _remove_gemini_md_block, _AGY_GEMINI_BLOCK_START, _AGY_GEMINI_BLOCK_END

    def test_inject_creates_file_when_absent(self, tmp_path: Path) -> None:
        inject, _, start, end = self._get_helpers()
        gemini_md = tmp_path / ".gemini" / "GEMINI.md"
        inject(gemini_md, "## Headroom\nContext instructions.", verbose=False)
        assert gemini_md.exists()
        text = gemini_md.read_text()
        assert start in text
        assert end in text
        assert "## Headroom" in text

    def test_inject_preserves_existing_user_content(self, tmp_path: Path) -> None:
        inject, _, start, end = self._get_helpers()
        gemini_md = tmp_path / "GEMINI.md"
        gemini_md.write_text("# User instructions\n\nSome personal notes.\n")
        inject(gemini_md, "## Headroom\nContext instructions.", verbose=False)
        text = gemini_md.read_text()
        assert "# User instructions" in text
        assert "Some personal notes." in text
        assert start in text
        assert end in text

    def test_inject_is_idempotent(self, tmp_path: Path) -> None:
        inject, _, start, end = self._get_helpers()
        gemini_md = tmp_path / "GEMINI.md"
        inject(gemini_md, "## Headroom\nContext instructions.", verbose=False)
        inject(gemini_md, "## Headroom\nContext instructions.", verbose=False)
        text = gemini_md.read_text()
        # Block should appear exactly once
        assert text.count(start) == 1
        assert text.count(end) == 1

    def test_inject_replaces_stale_block(self, tmp_path: Path) -> None:
        inject, _, start, end = self._get_helpers()
        gemini_md = tmp_path / "GEMINI.md"
        inject(gemini_md, "old content", verbose=False)
        inject(gemini_md, "new content", verbose=False)
        text = gemini_md.read_text()
        assert "new content" in text
        assert "old content" not in text
        assert text.count(start) == 1

    def test_remove_deletes_only_headroom_block(self, tmp_path: Path) -> None:
        inject, remove, start, end = self._get_helpers()
        gemini_md = tmp_path / "GEMINI.md"
        gemini_md.write_text("# User content\nKeep this.\n")
        inject(gemini_md, "## Headroom\nContext.", verbose=False)
        removed = remove(gemini_md, verbose=False)
        assert removed is True
        text = gemini_md.read_text()
        assert "# User content" in text
        assert "Keep this." in text
        assert start not in text
        assert end not in text

    def test_remove_is_idempotent(self, tmp_path: Path) -> None:
        inject, remove, start, end = self._get_helpers()
        gemini_md = tmp_path / "GEMINI.md"
        inject(gemini_md, "## Headroom\nContext.", verbose=False)
        assert remove(gemini_md, verbose=False) is True
        assert remove(gemini_md, verbose=False) is False

    def test_remove_returns_false_when_file_absent(self, tmp_path: Path) -> None:
        _, remove, _, _ = self._get_helpers()
        gemini_md = tmp_path / "GEMINI.md"
        assert remove(gemini_md, verbose=False) is False

    def test_remove_returns_false_when_no_block(self, tmp_path: Path) -> None:
        _, remove, _, _ = self._get_helpers()
        gemini_md = tmp_path / "GEMINI.md"
        gemini_md.write_text("# User content only\n")
        assert remove(gemini_md, verbose=False) is False


# ---------------------------------------------------------------------------
# T9: unwrap agy reverts GEMINI.md block (integration via CLI runner)
# ---------------------------------------------------------------------------


class TestUnwrapAgyReverts:
    """unwrap agy removes headroom block; preserves user content; is idempotent."""

    def test_unwrap_removes_gemini_md_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.cli.wrap import (
            _AGY_GEMINI_BLOCK_END,
            _AGY_GEMINI_BLOCK_START,
        )

        gemini_md = tmp_path / ".gemini" / "GEMINI.md"
        gemini_md.parent.mkdir(parents=True, exist_ok=True)
        gemini_md.write_text(
            f"# User content\n\n{_AGY_GEMINI_BLOCK_START}\n## Headroom\n"
            f"{_AGY_GEMINI_BLOCK_END}\n"
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0
        text = gemini_md.read_text()
        assert _AGY_GEMINI_BLOCK_START not in text
        assert "# User content" in text

    def test_unwrap_is_idempotent_when_already_clean(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gemini_md = tmp_path / ".gemini" / "GEMINI.md"
        gemini_md.parent.mkdir(parents=True, exist_ok=True)
        gemini_md.write_text("# User content only\n")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# T9: MCP retrieve tool wiring (N/A-v1 for per-run ephemeral port)
# ---------------------------------------------------------------------------


class TestAgyMcpRetrieveNa:
    """Verify wrap agy does NOT register a retrieve MCP entry outside the
    interactive MITM path.

    Interactive MITM now starts a per-run PLAIN-HTTP loopback retrieve listener
    and registers a per-run headroom MCP entry pointing at it (reverted on
    teardown — see TestAgyRetrieveMcpWiring).  But --no-intercept (passthrough)
    starts no servers, so it must register nothing: there is no listener to
    point a persistent entry at.
    """

    def test_agy_mcp_config_not_written_during_wrap_no_intercept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--no-intercept path: no MCP registration should happen."""
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)

        # Redirect HOME so we never touch the real ~/.gemini.
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: MagicMock(returncode=0))

        runner = CliRunner()
        runner.invoke(_get_main(), ["wrap", "agy", "--no-intercept"])

        mcp_config = tmp_path / ".gemini" / "antigravity-cli" / "mcp_config.json"
        # No per-run registration: file must not exist OR must not contain an
        # ephemeral headroom entry (port range check omitted; just assert no
        # ephemeral entry was written for "headroom").
        if mcp_config.exists():
            import json

            cfg = json.loads(mcp_config.read_text())
            assert "headroom" not in cfg.get("mcpServers", {}), (
                "wrap agy must not register an ephemeral headroom MCP entry"
            )


# ---------------------------------------------------------------------------
# T9 Fix 1: Serena MCP WIRED for agy (full MITM path, all servers stubbed)
# ---------------------------------------------------------------------------


def _stub_agy_mitm_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_uvx: bool = True,
):
    """Stub the full agy MITM run so wrap agy reaches the MCP wiring.

    Redirects HOME to tmp_path (isolating ~/.gemini and ~/.headroom ledger),
    stubs server lifecycle + CA + subprocess so nothing real launches.  When
    ``with_uvx`` is True, shutil.which("uvx") resolves so _setup_serena_mcp
    proceeds.  Pre-creates ~/.gemini/antigravity-cli so AgyRegistrar.detect()
    returns True.
    """
    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    import headroom.cli.wrap as wrap_mod

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Pre-create the agy config dir so AgyRegistrar.detect() is True.
    (tmp_path / ".gemini" / "antigravity-cli").mkdir(parents=True, exist_ok=True)

    def fake_which(name: str):
        if name == "agy":
            return "/usr/bin/agy"
        if name == "uvx" and with_uvx:
            return "/usr/bin/uvx"
        return None

    monkeypatch.setattr("shutil.which", fake_which)

    fake_servers = MagicMock()
    fake_servers.terminator.address = ("127.0.0.1", 54321)
    fake_servers.dispatch.address = ("127.0.0.1", 54322)
    # Interactive-mode retrieve listener port (a real int so the headroom MCP
    # spec gets a well-formed loopback URL). _agy_start_calls records the
    # start_retrieve flag each call so tests can assert print-mode skips it.
    fake_servers.retrieve_port = 54323
    fake_servers.retrieve = MagicMock()

    def _fake_start_agy_servers(ca_key, ca_cert, base_dir=None, *, start_retrieve=False):
        _agy_start_calls.append(start_retrieve)
        # In print mode the real server starts no retrieve listener: model that
        # so the agy() guard (servers.retrieve_port is not None) holds.
        if not start_retrieve:
            fake_servers.retrieve = None
            fake_servers.retrieve_port = None
        else:
            fake_servers.retrieve = MagicMock()
            fake_servers.retrieve_port = 54323
        return fake_servers

    _agy_start_calls: list[bool] = []
    fake_servers._agy_start_calls = _agy_start_calls
    monkeypatch.setattr(wrap_mod, "_start_agy_servers", _fake_start_agy_servers)
    monkeypatch.setattr(wrap_mod, "_stop_agy_servers", lambda s: None)
    # Default the MCP handshake smoke check to PASS so interactive registrations
    # survive; individual tests override this when they exercise the failure path.
    monkeypatch.setattr(wrap_mod, "_smoke_verify_mcp_handshake", lambda *a, **kw: True)

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
    # Force the RTK context-tool path so _setup_lean_ctx_agent (which would shell
    # out to the real lean-ctx binary) is not invoked.
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: MagicMock(returncode=0))


class TestAgySerenaWired:
    """wrap agy registers Serena via AgyRegistrar; --no-serena removes/skips it."""

    def test_wrap_agy_registers_serena(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["wrap", "agy"], catch_exceptions=False)
        assert result.exit_code == 0

        reg = AgyRegistrar(home_dir=tmp_path)
        spec = reg.get_server("serena")
        assert spec is not None, "wrap agy must register a 'serena' MCP entry"
        assert spec.command == "uvx"
        assert "ide-assistant" in spec.args

    def test_wrap_agy_no_serena_does_not_register(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--no-serena"], catch_exceptions=False
        )
        assert result.exit_code == 0

        reg = AgyRegistrar(home_dir=tmp_path)
        assert reg.get_server("serena") is None, (
            "--no-serena must not leave a Serena MCP entry"
        )

    def test_wrap_agy_no_serena_removes_prior_headroom_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--no-serena actively removes a Headroom-installed Serena entry."""
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.install import build_serena_spec
        from headroom.mcp_registry.ledger import record_install

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        # Seed a Headroom-installed Serena entry + ledger record.
        reg = AgyRegistrar(home_dir=tmp_path)
        serena_spec = build_serena_spec("ide-assistant")
        reg.register_server(serena_spec)
        record_install("agy", serena_spec)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--no-serena"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server("serena") is None


# ---------------------------------------------------------------------------
# T9 Fix 2: unwrap_agy Serena removal is ledger-gated (falsification guard)
# ---------------------------------------------------------------------------


class TestUnwrapAgySerena:
    """unwrap_agy removes only Headroom-installed Serena; preserves user entries."""

    def test_unwrap_removes_headroom_installed_serena(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.install import build_serena_spec
        from headroom.mcp_registry.ledger import record_install

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        reg = AgyRegistrar(home_dir=tmp_path)
        serena_spec = build_serena_spec("ide-assistant")
        reg.register_server(serena_spec)
        record_install("agy", serena_spec)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server("serena") is None

    def test_unwrap_preserves_user_managed_serena(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user-managed serena entry (absent from ledger) must survive unwrap."""
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        reg = AgyRegistrar(home_dir=tmp_path)
        # User-managed entry: different command, NOT recorded in ledger.
        user_spec = ServerSpec(
            name="serena",
            command="/opt/my-serena/bin/serena",
            args=("custom",),
            env={},
        )
        reg.register_server(user_spec)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0
        survived = AgyRegistrar(home_dir=tmp_path).get_server("serena")
        assert survived is not None, "user-managed serena must not be removed"
        assert survived.command == "/opt/my-serena/bin/serena"


# ---------------------------------------------------------------------------
# WU-0: agy print-mode MCP hang fix + lean-ctx context-tool wiring
# ---------------------------------------------------------------------------


class TestAgyPrintModeDetection:
    """_agy_print_mode flags single-shot non-interactive invocations."""

    def _fn(self):
        from headroom.cli.wrap import _agy_print_mode
        return _agy_print_mode

    def test_detects_print(self) -> None:
        assert self._fn()(("--print", "hello")) is True

    def test_detects_short_p(self) -> None:
        assert self._fn()(("-p", "hello")) is True

    def test_detects_prompt_alias(self) -> None:
        assert self._fn()(("--prompt", "hello")) is True

    def test_detects_print_equals_joined(self) -> None:
        # agy accepts `--print=hi` (live-verified) — must be treated as print mode,
        # else the interactive branch persists an MCP and the hang returns.
        assert self._fn()(("--print=hi",)) is True

    def test_detects_prompt_equals_joined(self) -> None:
        assert self._fn()(("--prompt=hi",)) is True

    def test_detects_short_p_equals_joined(self) -> None:
        # agy accepts `-p=hi` (live-verified).
        assert self._fn()(("-p=hi",)) is True

    def test_attached_short_p_value_is_false(self) -> None:
        # agy REJECTS `-pVALUE` (exit 2, "flags provided but not defined") — it
        # never reaches MCP init, so it must NOT be treated as print mode.
        assert self._fn()(("-pHI",)) is False

    def test_interactive_is_false(self) -> None:
        assert self._fn()(()) is False
        assert self._fn()(("--model", "x")) is False
        assert self._fn()(("--model=x",)) is False


class TestAgyPrintModeSuppressesMcp:
    """Print-mode wrap agy must activate NO MCP server (else agy hangs)."""

    def test_print_mode_does_not_register_serena(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--", "--print", "hi"], catch_exceptions=False
        )
        assert result.exit_code == 0
        reg = AgyRegistrar(home_dir=tmp_path)
        assert reg.get_server("serena") is None, (
            "print mode must not register a Serena MCP entry (it hangs agy)"
        )

    def test_print_mode_removes_prior_headroom_serena(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.install import build_serena_spec
        from headroom.mcp_registry.ledger import record_install

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        reg = AgyRegistrar(home_dir=tmp_path)
        serena_spec = build_serena_spec("ide-assistant")
        reg.register_server(serena_spec)
        record_install("agy", serena_spec)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--", "-p", "hi"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server("serena") is None, (
            "print mode must remove a Headroom-installed Serena entry"
        )

    def test_print_mode_does_not_register_lean_ctx(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--", "--print", "hi"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server("lean-ctx") is None, (
            "print mode must not register a lean-ctx MCP entry"
        )


class TestAgyLeanCtxMcpWiring:
    """Interactive lean-ctx context tool registers a correct, verified MCP entry."""

    def test_registers_correct_spec_and_keeps_on_smoke_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import headroom.cli.wrap as wrap_mod
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")
        monkeypatch.setattr(
            "headroom.lean_ctx.get_lean_ctx_path", lambda: Path("/usr/bin/lean-ctx")
        )
        # Smoke handshake passes.
        monkeypatch.setattr(
            wrap_mod, "_smoke_verify_mcp_handshake", lambda *a, **kw: True
        )

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy"], catch_exceptions=False
        )
        assert result.exit_code == 0
        spec = AgyRegistrar(home_dir=tmp_path).get_server("lean-ctx")
        assert spec is not None, "interactive lean-ctx must register an MCP entry"
        assert spec.command == "/usr/bin/lean-ctx"
        assert spec.args == ("mcp",), "must register 'lean-ctx mcp', not a bare command"
        assert "LEAN_CTX_DATA_DIR" in spec.env

    def test_removes_entry_when_smoke_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import headroom.cli.wrap as wrap_mod
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")
        monkeypatch.setattr(
            "headroom.lean_ctx.get_lean_ctx_path", lambda: Path("/usr/bin/lean-ctx")
        )
        # Smoke handshake FAILS -> entry must be removed (never persist a hanger).
        monkeypatch.setattr(
            wrap_mod, "_smoke_verify_mcp_handshake", lambda *a, **kw: False
        )

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server("lean-ctx") is None, (
            "a lean-ctx entry that fails the handshake must be removed"
        )

    def test_skips_when_lean_ctx_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")
        monkeypatch.setattr("headroom.lean_ctx.get_lean_ctx_path", lambda: None)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server("lean-ctx") is None


class TestAgyRetrieveMcpWiring:
    """Headroom retrieve MCP: interactive-only, per-run loopback, reverted.

    The retrieve listener is an ephemeral PLAIN-HTTP loopback server started in
    interactive mode only; its port is registered as the headroom MCP's
    HEADROOM_PROXY_URL, then REVERTED on teardown so no stale pointer survives.
    Print mode starts no listener and registers no entry (any MCP hangs agy).
    """

    def test_interactive_registers_then_reverts_retrieve_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Interactive: headroom entry registered with the live loopback port
        DURING the run, then reverted on teardown (no stale entry remains)."""
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        # Capture whether the headroom entry was live AT THE MOMENT agy ran
        # (i.e. while subprocess.run executes), proving it existed mid-session.
        seen: dict[str, object] = {}

        def _capture_run(cmd, *a, **kw):
            spec = AgyRegistrar(home_dir=tmp_path).get_server("headroom")
            seen["spec"] = spec
            return MagicMock(returncode=0)

        monkeypatch.setattr("subprocess.run", _capture_run)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["wrap", "agy"], catch_exceptions=False)
        assert result.exit_code == 0

        live_spec = seen["spec"]
        assert live_spec is not None, "interactive run must register a headroom retrieve entry"
        # The entry must point at the live loopback retrieve port (54323 from the
        # stub), via HEADROOM_PROXY_URL on the headroom mcp serve child.
        assert live_spec.command == "headroom"
        assert live_spec.args == ("mcp", "serve")
        assert live_spec.env.get("HEADROOM_PROXY_URL") == "http://127.0.0.1:54323"

        # After teardown the ephemeral entry MUST be gone (no dead pointer).
        assert AgyRegistrar(home_dir=tmp_path).get_server("headroom") is None, (
            "the per-run retrieve entry must be reverted on teardown"
        )

    def test_print_mode_does_not_register_retrieve_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Print mode: no retrieve listener, no headroom MCP entry (would hang agy)."""
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        # Capture mid-session too: even DURING the run no headroom entry exists.
        seen: dict[str, object] = {}

        def _capture_run(cmd, *a, **kw):
            seen["spec"] = AgyRegistrar(home_dir=tmp_path).get_server("headroom")
            return MagicMock(returncode=0)

        monkeypatch.setattr("subprocess.run", _capture_run)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--", "--print", "hi"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert seen["spec"] is None, "print mode must not register a headroom retrieve entry mid-run"
        assert AgyRegistrar(home_dir=tmp_path).get_server("headroom") is None

    def test_print_mode_does_not_start_retrieve_listener(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Print mode: _start_agy_servers is called with start_retrieve=False."""
        import headroom.cli.wrap as wrap_mod

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        captured: list[bool] = []
        real_stub = wrap_mod._start_agy_servers

        def _spy(ca_key, ca_cert, base_dir=None, *, start_retrieve=False):
            captured.append(start_retrieve)
            return real_stub(ca_key, ca_cert, base_dir, start_retrieve=start_retrieve)

        monkeypatch.setattr(wrap_mod, "_start_agy_servers", _spy)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--", "-p", "hi"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert captured == [False], "print mode must not start the retrieve listener"

    def test_interactive_starts_retrieve_listener(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Interactive: _start_agy_servers is called with start_retrieve=True."""
        import headroom.cli.wrap as wrap_mod

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        captured: list[bool] = []
        real_stub = wrap_mod._start_agy_servers

        def _spy(ca_key, ca_cert, base_dir=None, *, start_retrieve=False):
            captured.append(start_retrieve)
            return real_stub(ca_key, ca_cert, base_dir, start_retrieve=start_retrieve)

        monkeypatch.setattr(wrap_mod, "_start_agy_servers", _spy)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["wrap", "agy"], catch_exceptions=False)
        assert result.exit_code == 0
        assert captured == [True], "interactive mode must start the retrieve listener"

    def test_failed_smoke_handshake_removes_retrieve_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retrieve entry that fails the MCP handshake must not persist."""
        import headroom.cli.wrap as wrap_mod
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        # Handshake FAILS -> verify-then-remove path for the headroom entry.
        monkeypatch.setattr(wrap_mod, "_smoke_verify_mcp_handshake", lambda *a, **kw: False)

        seen: dict[str, object] = {}

        def _capture_run(cmd, *a, **kw):
            seen["spec"] = AgyRegistrar(home_dir=tmp_path).get_server("headroom")
            return MagicMock(returncode=0)

        monkeypatch.setattr("subprocess.run", _capture_run)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["wrap", "agy"], catch_exceptions=False)
        assert result.exit_code == 0
        assert seen["spec"] is None, (
            "a headroom entry that fails the handshake must be removed before agy runs"
        )
        assert AgyRegistrar(home_dir=tmp_path).get_server("headroom") is None


class TestAgyRtkGate:
    """RTK GEMINI.md block is injected only when the rtk binary is present."""

    def test_rtk_block_skipped_when_rtk_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        # Default context tool is rtk; _stub sets which() to resolve only agy/uvx,
        # so shutil.which("rtk") is None.
        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy"], catch_exceptions=False
        )
        assert result.exit_code == 0
        gemini_md = tmp_path / ".gemini" / "GEMINI.md"
        if gemini_md.exists():
            assert "rtk-instructions" not in gemini_md.read_text(), (
                "RTK block must not be injected when rtk is not installed"
            )

    def test_rtk_block_injected_when_rtk_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        def which_with_rtk(name: str):
            if name in ("agy", "rtk"):
                return f"/usr/bin/{name}"
            if name == "uvx":
                return "/usr/bin/uvx"
            return None

        monkeypatch.setattr("shutil.which", which_with_rtk)
        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy"], catch_exceptions=False
        )
        assert result.exit_code == 0
        gemini_md = tmp_path / ".gemini" / "GEMINI.md"
        assert gemini_md.exists()
        assert "rtk-instructions" in gemini_md.read_text()


class TestUnwrapAgyLeanCtx:
    """unwrap agy removes the lean-ctx context-tool MCP entry it left behind."""

    def test_unwrap_removes_lean_ctx_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.install import build_lean_ctx_spec

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        reg = AgyRegistrar(home_dir=tmp_path)
        reg.register_server(
            build_lean_ctx_spec("/usr/bin/lean-ctx", "/x/data")
        )

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server("lean-ctx") is None, (
            "unwrap agy must remove the lean-ctx MCP entry"
        )

    def test_unwrap_preserves_unrelated_user_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        reg = AgyRegistrar(home_dir=tmp_path)
        reg.register_server(
            ServerSpec(name="my-tool", command="/opt/my-tool", args=(), env={})
        )

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0
        survived = AgyRegistrar(home_dir=tmp_path).get_server("my-tool")
        assert survived is not None, "unrelated user MCP entries must survive unwrap"


class TestSmokeVerifyMcpHandshake:
    """_smoke_verify_mcp_handshake: pass on a real responder, fail on a broken one."""

    def test_returns_true_for_responding_server(self, tmp_path: Path) -> None:
        from headroom.cli.wrap import _smoke_verify_mcp_handshake

        # A tiny stdio server that echoes a JSON-RPC initialize response.
        server = tmp_path / "fake_mcp.py"
        server.write_text(
            "import sys, json\n"
            "line = sys.stdin.readline()\n"
            "req = json.loads(line)\n"
            "print(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': {}}))\n"
            "sys.stdout.flush()\n"
        )
        import sys as _sys

        ok = _smoke_verify_mcp_handshake(_sys.executable, [str(server)], {}, timeout=10.0)
        assert ok is True

    def test_returns_false_for_nonexistent_command(self) -> None:
        from headroom.cli.wrap import _smoke_verify_mcp_handshake

        assert (
            _smoke_verify_mcp_handshake("/nonexistent/mcp-bin", [], {}, timeout=5.0)
            is False
        )

    def test_returns_false_when_no_response_in_time(self, tmp_path: Path) -> None:
        from headroom.cli.wrap import _smoke_verify_mcp_handshake

        # A server that reads but never replies — must time out -> False.
        server = tmp_path / "silent_mcp.py"
        server.write_text("import sys, time\nsys.stdin.readline()\ntime.sleep(30)\n")
        import sys as _sys

        ok = _smoke_verify_mcp_handshake(_sys.executable, [str(server)], {}, timeout=2.0)
        assert ok is False


# ---------------------------------------------------------------------------
# WU-B: build_codegraph_spec shape
# ---------------------------------------------------------------------------


class TestBuildCodegraphSpec:
    """build_codegraph_spec produces a correctly-shaped ServerSpec."""

    def test_name_matches_cbm_server_name_constant(self) -> None:
        from headroom.cli.wrap import _CBM_MCP_SERVER_NAME
        from headroom.mcp_registry.install import build_codegraph_spec

        spec = build_codegraph_spec("/usr/local/bin/cbm")
        assert spec.name == _CBM_MCP_SERVER_NAME

    def test_command_is_cbm_bin(self) -> None:
        from headroom.mcp_registry.install import build_codegraph_spec

        spec = build_codegraph_spec("/usr/local/bin/cbm")
        assert spec.command == "/usr/local/bin/cbm"

    def test_no_extra_args(self) -> None:
        from headroom.mcp_registry.install import build_codegraph_spec

        spec = build_codegraph_spec("/usr/local/bin/cbm")
        assert spec.args == ()

    def test_no_env(self) -> None:
        from headroom.mcp_registry.install import build_codegraph_spec

        spec = build_codegraph_spec("/usr/local/bin/cbm")
        assert spec.env == {}

    def test_exported_from_mcp_registry(self) -> None:
        from headroom.mcp_registry import build_codegraph_spec  # noqa: F401


# ---------------------------------------------------------------------------
# WU-B: --code-graph flag on wrap agy (interactive + print mode + default off)
# ---------------------------------------------------------------------------


def _stub_agy_with_cbm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cbm_bin: str = "/usr/local/bin/cbm",
    cbm_exists: bool = True,
    smoke_passes: bool = True,
) -> None:
    """Extend _stub_agy_mitm_run with cbm binary stubs."""
    import headroom.cli.wrap as wrap_mod

    _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

    # Stub cbm binary resolution so no network download occurs.
    from pathlib import Path as _Path

    monkeypatch.setattr(
        "headroom.graph.installer.get_cbm_path",
        lambda: _Path(cbm_bin) if cbm_exists else None,
    )
    monkeypatch.setattr(
        "headroom.graph.installer.ensure_cbm",
        lambda: _Path(cbm_bin) if cbm_exists else None,
    )
    # Stub _setup_code_graph so no real indexing runs.
    monkeypatch.setattr(wrap_mod, "_setup_code_graph", lambda verbose=False: True)
    # Override smoke verify for code-graph tests.
    monkeypatch.setattr(
        wrap_mod, "_smoke_verify_mcp_handshake", lambda *a, **kw: smoke_passes
    )


class TestAgyCodeGraphFlag:
    """--code-graph flag wiring: interactive registers cbm MCP; print-mode skips it."""

    def test_code_graph_interactive_registers_cbm_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Interactive --code-graph: cbm entry registered via AgyRegistrar."""
        from headroom.cli.wrap import _CBM_MCP_SERVER_NAME
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_with_cbm(tmp_path, monkeypatch, smoke_passes=True)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--code-graph"], catch_exceptions=False
        )
        assert result.exit_code == 0
        spec = AgyRegistrar(home_dir=tmp_path).get_server(_CBM_MCP_SERVER_NAME)
        assert spec is not None, "interactive --code-graph must register the cbm MCP entry"
        assert spec.command == "/usr/local/bin/cbm"

    def test_code_graph_interactive_calls_smoke_verify(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Interactive --code-graph: _smoke_verify_mcp_handshake is called."""
        import headroom.cli.wrap as wrap_mod

        _stub_agy_with_cbm(tmp_path, monkeypatch, smoke_passes=True)
        smoke_calls: list[tuple] = []

        def _spy(*a, **kw):
            smoke_calls.append(a)
            return True

        monkeypatch.setattr(wrap_mod, "_smoke_verify_mcp_handshake", _spy)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--code-graph"], catch_exceptions=False
        )
        assert result.exit_code == 0
        # Smoke was called at least once (for cbm).
        assert len(smoke_calls) >= 1

    def test_code_graph_print_mode_skips_registration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--code-graph + print mode: cbm entry must NOT be registered."""
        from headroom.cli.wrap import _CBM_MCP_SERVER_NAME
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_with_cbm(tmp_path, monkeypatch, smoke_passes=True)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(),
            ["wrap", "agy", "--code-graph", "--", "--print", "hi"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server(_CBM_MCP_SERVER_NAME) is None, (
            "--code-graph + print mode must NOT register cbm (agy hangs with MCP in print mode)"
        )

    def test_no_code_graph_flag_does_not_register_cbm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without --code-graph, no cbm entry registered (default off)."""
        from headroom.cli.wrap import _CBM_MCP_SERVER_NAME
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_with_cbm(tmp_path, monkeypatch, smoke_passes=True)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server(_CBM_MCP_SERVER_NAME) is None, (
            "omitting --code-graph must NOT register cbm (default off)"
        )

    def test_failed_smoke_removes_cbm_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Interactive --code-graph: if smoke fails, cbm entry must be removed."""
        from headroom.cli.wrap import _CBM_MCP_SERVER_NAME
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_with_cbm(tmp_path, monkeypatch, smoke_passes=False)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--code-graph"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server(_CBM_MCP_SERVER_NAME) is None, (
            "a cbm entry that fails the handshake must be removed"
        )


# ---------------------------------------------------------------------------
# WU-B: unwrap agy removes ledger-owned cbm entry; preserves user-managed one
# ---------------------------------------------------------------------------


class TestUnwrapAgyCbm:
    """unwrap agy removes only Headroom-installed cbm; preserves user entries."""

    def test_unwrap_removes_headroom_installed_cbm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.cli.wrap import _CBM_MCP_SERVER_NAME
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.install import build_codegraph_spec
        from headroom.mcp_registry.ledger import record_install

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        reg = AgyRegistrar(home_dir=tmp_path)
        cbm_spec = build_codegraph_spec("/usr/local/bin/cbm")
        reg.register_server(cbm_spec)
        record_install("agy", cbm_spec)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server(_CBM_MCP_SERVER_NAME) is None, (
            "unwrap agy must remove a Headroom-installed cbm MCP entry"
        )

    def test_unwrap_preserves_user_managed_cbm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user-managed cbm entry (absent from ledger) must survive unwrap."""
        from headroom.cli.wrap import _CBM_MCP_SERVER_NAME
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        reg = AgyRegistrar(home_dir=tmp_path)
        # User-managed entry: different command, NOT recorded in ledger.
        user_spec = ServerSpec(
            name=_CBM_MCP_SERVER_NAME,
            command="/opt/my-cbm/bin/cbm",
            args=(),
            env={},
        )
        reg.register_server(user_spec)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0
        survived = AgyRegistrar(home_dir=tmp_path).get_server(_CBM_MCP_SERVER_NAME)
        assert survived is not None, "user-managed cbm entry must not be removed by unwrap"
        assert survived.command == "/opt/my-cbm/bin/cbm"


# ---------------------------------------------------------------------------
# headroom-30y.15: fail-open observability + session compression summary
# ---------------------------------------------------------------------------


class TestAgySessionCompressionSummary:
    """Integration: wrap agy prints a session compression summary on normal exit."""

    def test_summary_line_appears_on_normal_exit_mixed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Summary appears in combined output when mix_stderr=True (default)."""
        from unittest.mock import patch

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        _empty_stats = {
            "entry_count": 0,
            "total_original_tokens": 0,
            "total_compressed_tokens": 0,
        }

        with patch(
            "headroom.providers.agy.stats._get_compression_stats",
            return_value=_empty_stats,
        ):
            runner = CliRunner()
            result = runner.invoke(
                _get_main(), ["wrap", "agy"], catch_exceptions=False
            )

        assert result.exit_code == 0
        assert "Headroom agy session" in result.output

    def test_fail_open_handler_removed_after_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The FailOpenWarnHandler must NOT remain on the logger after agy exits."""
        import logging
        from unittest.mock import patch

        from headroom.providers.agy.stats import _GEMINI_LOGGER

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        _empty_stats = {
            "entry_count": 0,
            "total_original_tokens": 0,
            "total_compressed_tokens": 0,
        }

        logger = logging.getLogger(_GEMINI_LOGGER)
        handlers_before = list(logger.handlers)

        with patch(
            "headroom.providers.agy.stats._get_compression_stats",
            return_value=_empty_stats,
        ):
            runner = CliRunner()
            runner.invoke(_get_main(), ["wrap", "agy"], catch_exceptions=False)

        # No new handlers leaked
        assert logger.handlers == handlers_before
