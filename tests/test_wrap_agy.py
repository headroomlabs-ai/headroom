"""Tests for headroom wrap agy / unwrap agy."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from headroom.cli.wrap import _PROXY_URL_REDACTED_PLACEHOLDER, redact_proxy_url

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WRAP_MODULE = "headroom.cli.wrap"


# ---------------------------------------------------------------------------
# headroom wrap agy — CLI integration tests
# ---------------------------------------------------------------------------


def _get_main():
    from headroom.cli.main import main

    return main


@pytest.fixture(autouse=True)
def _never_start_a_real_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``_ensure_proxy`` for every test in this module.

    ``_ensure_proxy`` -> ``_start_proxy`` calls ``subprocess.Popen`` directly,
    which none of this file's per-test ``subprocess.run`` stubs touch. Left
    unstubbed, any test that drives the full ``agy``/``unwrap`` CLI spawns a
    real ``headroom.cli proxy`` subprocess that binds a real port — on a dev
    machine with a live proxy already on that port, this evicts it. No test
    in this file exercises ``_ensure_proxy`` itself (that's covered
    elsewhere), so stubbing it here is safe for all of them.
    """
    import headroom.cli.wrap as wrap_mod

    def _fake_ensure_proxy(port, no_proxy=False, **_kwargs):
        return None, port

    monkeypatch.setattr(wrap_mod, "_ensure_proxy", _fake_ensure_proxy)
    # These tests verify MCP configuration, not Serena's external indexer.
    # A fake shutil.which("uvx") must not launch the real executable on PATH.
    monkeypatch.setattr(wrap_mod, "_index_serena_project", lambda **kw: None)


class TestWrapAgyBinaryMissing:
    """Binary-missing path must exit 1 with install hint."""

    def test_exits_1_when_agy_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _: None)
        runner = CliRunner()
        result = runner.invoke(_get_main(), ["wrap", "agy"])
        assert result.exit_code == 1

    def test_prints_install_hint_when_agy_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_rust_backend_flag_prints_clear_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

        def fake_start_agy_servers(
            ca_key, ca_cert, base_dir=None, *, start_retrieve=False, project=None
        ):
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

    def test_disclosure_banner_mentions_unwrap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._invoke_agy(monkeypatch)
        assert "unwrap" in result.output.lower()


class TestWrapAgyMcpFlagParity:
    """agy exposes the same MCP opt-out surface as its sibling subcommands."""

    def test_no_mcp_is_offered_like_the_siblings(self) -> None:
        result = CliRunner().invoke(_get_main(), ["wrap", "agy", "--help"])

        assert result.exit_code == 0
        assert "--no-mcp" in result.output
        assert "--no-serena" in result.output

    def test_no_mcp_promises_the_same_thing_as_the_siblings(self) -> None:
        """Same flag, same promise — drift between siblings is the bug being fixed."""
        agy_help = " ".join(
            CliRunner().invoke(_get_main(), ["wrap", "agy", "--help"]).output.split()
        )
        assert "--no-mcp Skip headroom MCP server registration" in agy_help


class TestWrapAgyNoIntercept:
    """--no-intercept flag must change behavior (no MITM server startup)."""

    def test_no_intercept_does_not_start_servers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)

        import headroom.cli.wrap as wrap_mod

        server_started = []

        def fake_start(ca_key, ca_cert, base_dir=None, *, start_retrieve=False, project=None):
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
            lambda ca_key, ca_cert, base_dir=None, *, start_retrieve=False, project=None: (
                fake_servers
            ),
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
# T9: GEMINI.md block removal (legacy blocks from pre-2677 installs)
# ---------------------------------------------------------------------------


class TestGeminiMdBlock:
    """_remove_gemini_md_block deletes only the Headroom block.

    `wrap agy` no longer writes a GEMINI.md block (the rtk context-tool
    instructions it carried were removed upstream), but `unwrap agy` must still
    clean a block an older install left behind.
    """

    def _get_helpers(self):
        from headroom.cli.wrap import (
            _AGY_GEMINI_BLOCK_END,
            _AGY_GEMINI_BLOCK_START,
            _remove_gemini_md_block,
        )

        return (_remove_gemini_md_block, _AGY_GEMINI_BLOCK_START, _AGY_GEMINI_BLOCK_END)

    def _write_legacy(self, gemini_md: Path, user_text: str = "") -> None:
        """Write a GEMINI.md exactly as an older `wrap agy` left it."""
        _, start, end = self._get_helpers()
        block = f"{start}\n## Headroom\nContext.\n{end}\n"
        gemini_md.parent.mkdir(parents=True, exist_ok=True)
        gemini_md.write_text(f"{user_text}\n\n{block}" if user_text else block)

    def test_remove_deletes_only_headroom_block(self, tmp_path: Path) -> None:
        remove, start, end = self._get_helpers()
        gemini_md = tmp_path / "GEMINI.md"
        self._write_legacy(gemini_md, "# User content\nKeep this.")
        assert remove(gemini_md, verbose=False) is True
        text = gemini_md.read_text()
        assert "# User content" in text
        assert "Keep this." in text
        assert start not in text
        assert end not in text

    def test_remove_is_idempotent(self, tmp_path: Path) -> None:
        remove, _, _ = self._get_helpers()
        gemini_md = tmp_path / "GEMINI.md"
        self._write_legacy(gemini_md)
        assert remove(gemini_md, verbose=False) is True
        assert remove(gemini_md, verbose=False) is False

    def test_remove_returns_false_when_file_absent(self, tmp_path: Path) -> None:
        remove, _, _ = self._get_helpers()
        gemini_md = tmp_path / "GEMINI.md"
        assert remove(gemini_md, verbose=False) is False

    def test_remove_returns_false_when_no_block(self, tmp_path: Path) -> None:
        remove, _, _ = self._get_helpers()
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
            f"# User content\n\n{_AGY_GEMINI_BLOCK_START}\n## Headroom\n{_AGY_GEMINI_BLOCK_END}\n"
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

    Interactive MITM registers a persistent, ledger-recorded headroom MCP
    retrieve entry (stable spec, on-disk store resolution — see
    TestAgyRetrieveMcpWiring).  But --no-intercept (passthrough) starts no
    servers, so on a fresh config it registers nothing.
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

        # agy 1.1.x read-path (migrated from .gemini/antigravity-cli/).
        mcp_config = tmp_path / ".gemini" / "config" / "mcp_config.json"
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

    def _fake_start_agy_servers(
        ca_key, ca_cert, base_dir=None, *, start_retrieve=False, project=None
    ):
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
    # Default the stubbed agy to a known-good version so print-mode MCP wiring is
    # exercised (headroom-37g.37 gates print-mode MCP on agy >= 1.0.16). The
    # suppress/purge path for older/unknown agy is covered separately in
    # tests/test_agy_print_mode_version_gate.py.
    monkeypatch.setattr(wrap_mod, "_detect_agy_version", lambda _agy_bin: (1, 0, 16))
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
        result = runner.invoke(_get_main(), ["wrap", "agy", "--no-serena"], catch_exceptions=False)
        assert result.exit_code == 0

        reg = AgyRegistrar(home_dir=tmp_path)
        assert reg.get_server("serena") is None, "--no-serena must not leave a Serena MCP entry"

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
        result = runner.invoke(_get_main(), ["wrap", "agy", "--no-serena"], catch_exceptions=False)
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server("serena") is None


# ---------------------------------------------------------------------------
# WU1: current main retired tokensave; AGY uses Serena code memory.
# Print mode wires MCP identically to interactive once the version gate allows it.
# ---------------------------------------------------------------------------


class TestAgySerenaCodeMemory:
    """wrap agy retires stale tokensave entries and registers Serena by default."""

    def _spy_helpers(self, monkeypatch: pytest.MonkeyPatch):
        """Replace code-memory helpers with call-recording spies."""
        import headroom.cli.wrap as wrap_mod

        calls: dict[str, list] = {
            "disable_tokensave": [],
            "setup_serena": [],
            "disable_serena": [],
        }

        def _disable_tokensave(registrar, *, verbose=False):
            calls["disable_tokensave"].append(verbose)

        def _setup_serena(registrar, *, context, verbose=False, force=False):
            calls["setup_serena"].append(context)

        def _disable_serena(registrar, *, verbose=False, reason="--no-serena"):
            calls["disable_serena"].append(reason)

        monkeypatch.setattr(wrap_mod, "_disable_tokensave_mcp", _disable_tokensave)
        monkeypatch.setattr(wrap_mod, "_setup_serena_mcp", _setup_serena)
        monkeypatch.setattr(wrap_mod, "_disable_serena_mcp", _disable_serena)
        return calls

    def test_interactive_retires_tokensave_and_registers_serena(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        calls = self._spy_helpers(monkeypatch)

        result = CliRunner().invoke(_get_main(), ["wrap", "agy"], catch_exceptions=False)
        assert result.exit_code == 0
        assert calls["disable_tokensave"], "stale Headroom-installed tokensave must be retired"
        assert calls["setup_serena"] == ["ide-assistant"]
        assert not calls["disable_serena"]

    def test_interactive_no_tokensave_flag_is_compat_and_uses_serena(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        calls = self._spy_helpers(monkeypatch)

        result = CliRunner().invoke(
            _get_main(), ["wrap", "agy", "--no-tokensave"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert calls["disable_tokensave"], "--no-tokensave still cleans up stale tokensave"
        assert calls["setup_serena"] == ["ide-assistant"]

    def test_print_mode_retires_tokensave_and_registers_serena(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        calls = self._spy_helpers(monkeypatch)

        result = CliRunner().invoke(
            _get_main(), ["wrap", "agy", "--", "--print", "hi"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert calls["disable_tokensave"]
        assert calls["setup_serena"] == ["ide-assistant"]
        assert not calls["disable_serena"]


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


class TestUnwrapAgyTokensave:
    """unwrap_agy removes only Headroom-installed tokensave; preserves user entries."""

    def test_unwrap_removes_headroom_installed_tokensave(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec
        from headroom.mcp_registry.ledger import record_install

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        reg = AgyRegistrar(home_dir=tmp_path)
        spec = ServerSpec(name="tokensave", command="tokensave", args=("serve",))
        reg.register_server(spec)
        record_install("agy", spec)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0
        assert AgyRegistrar(home_dir=tmp_path).get_server("tokensave") is None

    def test_unwrap_preserves_user_managed_tokensave(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user-managed tokensave entry (absent from ledger) must survive unwrap."""
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        reg = AgyRegistrar(home_dir=tmp_path)
        user_spec = ServerSpec(
            name="tokensave",
            command="/opt/my-tokensave/bin/tokensave",
            args=("serve",),
            env={},
        )
        reg.register_server(user_spec)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0
        survived = AgyRegistrar(home_dir=tmp_path).get_server("tokensave")
        assert survived is not None, "user-managed tokensave must not be removed"
        assert survived.command == "/opt/my-tokensave/bin/tokensave"


# ---------------------------------------------------------------------------
# agy print-mode MCP hang fix
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
    """Print-mode wrap agy skips a context tool only when its binary is absent.

    (Print mode otherwise wires MCP identically to interactive — see the
    tokensave/retrieve/code-graph parity tests; agy no longer hangs on MCP.)
    """


class TestAgyRetrieveMcpWiring:
    """Headroom retrieve MCP: persistent, local-store-backed, ledger-recorded.

    The retrieve entry is a stable ``headroom mcp serve`` server (no ephemeral
    port; ``env={}`` — it resolves markers from the on-disk CCR store). Started
    in BOTH print and interactive mode, it is registered PERSISTENTLY and
    recorded in the install ledger (like Serena/CBM), NOT reverted on teardown,
    so agy can cache and expose ``headroom_retrieve`` across sessions.
    """

    def test_interactive_registers_persistent_retrieve_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Interactive: headroom entry registered DURING the run and PERSISTS
        after teardown (ledger-recorded, resolves from the on-disk store)."""
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
        # The entry invokes `headroom mcp serve`, resolved via
        # resolve_headroom_command() — either the resolved `headroom` binary or
        # `<python> -m headroom.cli` when the binary is not on PATH.  Assert
        # against the actual resolution rather than a hard-coded "headroom" so
        # the test is robust across dev (editable) and CI installs.
        from headroom.install.runtime import resolve_headroom_command

        expected = resolve_headroom_command()
        assert live_spec.command == expected[0]
        assert live_spec.args == (*expected[1:], "mcp", "serve")
        # Stable, port-independent spec: no ephemeral HEADROOM_PROXY_URL — the
        # child resolves markers from the shared on-disk CCR store.
        assert dict(live_spec.env) == {}

        # The persistent entry SURVIVES teardown (like Serena/CBM) so agy caches
        # and exposes it next session.
        assert AgyRegistrar(home_dir=tmp_path).get_server("headroom") is not None, (
            "the persistent retrieve entry must survive teardown"
        )

    def test_print_mode_registers_persistent_retrieve_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Print mode wires the retrieve MCP like interactive: a stable headroom
        entry is live mid-run and PERSISTS after teardown."""
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        # Capture mid-session: the headroom entry must exist DURING the run.
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
        live_spec = seen["spec"]
        assert live_spec is not None, "print mode must register a headroom retrieve entry mid-run"
        # Stable, port-independent spec (on-disk store resolution).
        assert dict(live_spec.env) == {}
        # Persistent: survives teardown.
        assert AgyRegistrar(home_dir=tmp_path).get_server("headroom") is not None, (
            "the persistent retrieve entry must survive teardown"
        )

    def test_print_mode_starts_retrieve_listener(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Print mode: _start_agy_servers is called with start_retrieve=True (parity)."""
        import headroom.cli.wrap as wrap_mod

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        captured: list[bool] = []
        real_stub = wrap_mod._start_agy_servers

        def _spy(ca_key, ca_cert, base_dir=None, *, start_retrieve=False, project=None):
            captured.append(start_retrieve)
            return real_stub(
                ca_key, ca_cert, base_dir, start_retrieve=start_retrieve, project=project
            )

        monkeypatch.setattr(wrap_mod, "_start_agy_servers", _spy)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--", "-p", "hi"], catch_exceptions=False
        )
        assert result.exit_code == 0
        assert captured == [True], "print mode must start the retrieve listener (parity)"

    def test_interactive_starts_retrieve_listener(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Interactive: _start_agy_servers is called with start_retrieve=True."""
        import headroom.cli.wrap as wrap_mod

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        captured: list[bool] = []
        real_stub = wrap_mod._start_agy_servers

        def _spy(ca_key, ca_cert, base_dir=None, *, start_retrieve=False, project=None):
            captured.append(start_retrieve)
            return real_stub(
                ca_key, ca_cert, base_dir, start_retrieve=start_retrieve, project=project
            )

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


class TestUnwrapAgyUserEntries:
    """unwrap agy leaves MCP entries Headroom never installed untouched."""

    def test_unwrap_preserves_unrelated_user_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        reg = AgyRegistrar(home_dir=tmp_path)
        reg.register_server(ServerSpec(name="my-tool", command="/opt/my-tool", args=(), env={}))

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

        assert _smoke_verify_mcp_handshake("/nonexistent/mcp-bin", [], {}, timeout=5.0) is False

    def test_returns_false_when_no_response_in_time(self, tmp_path: Path) -> None:
        from headroom.cli.wrap import _smoke_verify_mcp_handshake

        # A server that reads but never replies — must time out -> False.
        server = tmp_path / "silent_mcp.py"
        server.write_text("import sys, time\nsys.stdin.readline()\ntime.sleep(30)\n")
        import sys as _sys

        ok = _smoke_verify_mcp_handshake(_sys.executable, [str(server)], {}, timeout=2.0)
        assert ok is False


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
            result = runner.invoke(_get_main(), ["wrap", "agy"], catch_exceptions=False)

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


# ---------------------------------------------------------------------------
# WU-s04.3: print-mode purges stale "headroom" retrieve MCP entry
# ---------------------------------------------------------------------------


class TestPrintModePurgesStaleHeadroomEntry:
    """Print-mode wrap agy must not remove user-managed MCP entries.

    (Print mode now wires the headroom retrieve MCP like interactive; it no
    longer scrubs a stale 'headroom' entry, since MCP no longer hangs agy in
    --print mode. User-managed entries are still left untouched.)
    """

    def test_print_mode_purge_does_not_remove_user_managed_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Print mode only removes the 'headroom' retrieve entry; user entries survive."""
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)

        reg = AgyRegistrar(home_dir=tmp_path)
        user_spec = ServerSpec(name="my-tool", command="/opt/my-tool", args=(), env={})
        reg.register_server(user_spec)

        runner = CliRunner()
        result = runner.invoke(
            _get_main(), ["wrap", "agy", "--", "--print", "hi"], catch_exceptions=False
        )
        assert result.exit_code == 0
        survived = AgyRegistrar(home_dir=tmp_path).get_server("my-tool")
        assert survived is not None, "user-managed entries must not be removed by print-mode purge"

    @pytest.mark.parametrize("name", ["headroom", "codebase-memory-mcp"])
    @pytest.mark.parametrize("owned", [False, True])
    def test_unsafe_print_purge_honors_entry_ownership(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, owned: bool
    ) -> None:
        from headroom.cli.wrap import _purge_agy_mcp_entries
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec
        from headroom.mcp_registry.ledger import record_install

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
        reg = AgyRegistrar(home_dir=tmp_path)
        user_spec = ServerSpec(name=name, command="/opt/user-mcp", args=(), env={})
        reg.register_server(user_spec)
        if owned:
            record_install("agy", user_spec)
        _purge_agy_mcp_entries(reg)
        assert reg.get_server(name) == (None if owned else user_spec)

    @pytest.mark.parametrize("owned", [False, True])
    def test_unwrap_preserves_unowned_legacy_code_graph(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owned: bool
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec
        from headroom.mcp_registry.ledger import record_install

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
        reg = AgyRegistrar(home_dir=tmp_path)
        spec = ServerSpec(name="codebase-memory-mcp", command="/opt/user-mcp", args=(), env={})
        reg.register_server(spec)
        if owned:
            record_install("agy", spec)
        result = CliRunner().invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0, result.output
        assert reg.get_server(spec.name) == (None if owned else spec)

    @pytest.mark.parametrize("version", [(1, 0, 8), None])
    def test_unsafe_print_refuses_remaining_user_mcp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version
    ) -> None:
        import headroom.cli.wrap as wrap_mod
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        monkeypatch.setattr(wrap_mod, "_detect_agy_version", lambda _: version)
        reg = AgyRegistrar(home_dir=tmp_path)
        user_spec = ServerSpec(name="my-tool", command="/opt/user-mcp", args=(), env={})
        reg.register_server(user_spec)
        launches = []
        monkeypatch.setattr(
            "subprocess.run", lambda cmd, **kw: launches.append(cmd) or MagicMock(returncode=0)
        )
        result = CliRunner().invoke(_get_main(), ["wrap", "agy", "--", "--print", "hi"])
        assert result.exit_code != 0
        assert "upgrade agy" in result.output.lower()
        assert launches == []
        assert reg.get_server("my-tool") == user_spec

    @pytest.mark.parametrize("owned", [True, False])
    def test_no_mcp_removes_only_owned_persistent_retrieve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owned: bool
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec
        from headroom.mcp_registry.ledger import record_install

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        reg = AgyRegistrar(home_dir=tmp_path)
        spec = ServerSpec(name="headroom", command="/opt/retrieve", args=(), env={})
        reg.register_server(spec)
        if owned:
            record_install("agy", spec)
        result = CliRunner().invoke(_get_main(), ["wrap", "agy", "--no-mcp"])
        assert result.exit_code == 0, result.output
        assert reg.get_server("headroom") == (None if owned else spec)

    def test_unsafe_print_rejects_unreadable_mcp_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import headroom.cli.wrap as wrap_mod
        from headroom.mcp_registry.agy import AgyRegistrar

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        monkeypatch.setattr(wrap_mod, "_detect_agy_version", lambda _: None)
        path = AgyRegistrar(home_dir=tmp_path).config_dir / "mcp_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{broken user config", encoding="utf-8")
        launches = []
        monkeypatch.setattr(
            "subprocess.run", lambda cmd, **kw: launches.append(cmd) or MagicMock(returncode=0)
        )
        result = CliRunner().invoke(_get_main(), ["wrap", "agy", "--", "--print", "hi"])
        assert result.exit_code != 0
        assert launches == []
        assert path.read_text(encoding="utf-8") == "{broken user config"


# ---------------------------------------------------------------------------
# WU s04.4: graceful failure modes
# ---------------------------------------------------------------------------


class TestAgyProxyFallbackMarkers:
    def test_owned_retrieve_entry_migrates(self, tmp_path, monkeypatch):
        import headroom.cli.wrap as wrap_mod
        from headroom.mcp_registry import build_headroom_spec
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec
        from headroom.mcp_registry.ledger import headroom_installed_matching, record_install

        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
        registrar = AgyRegistrar(home_dir=tmp_path)
        old = ServerSpec(name="headroom", command="old-headroom", args=("serve",))
        registrar.register_server(old)
        record_install("agy", old)
        monkeypatch.setattr(wrap_mod, "_smoke_verify_mcp_handshake", lambda *args: True)
        monkeypatch.setattr(wrap_mod, "_prime_agy_retrieve_tool_cache", lambda *args: None)
        assert wrap_mod._setup_headroom_retrieve_mcp_agy(registrar)
        assert registrar.get_server("headroom") == build_headroom_spec()
        assert headroom_installed_matching("agy", registrar.get_server("headroom"))

    @pytest.mark.parametrize("handshake_ok", [True, False])
    def test_matching_user_retrieve_entry_is_not_claimed(self, tmp_path, monkeypatch, handshake_ok):
        import headroom.cli.wrap as wrap_mod
        from headroom.mcp_registry import build_headroom_spec
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.ledger import headroom_installed_matching

        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
        registrar = AgyRegistrar(home_dir=tmp_path)
        spec = build_headroom_spec()
        registrar.register_server(spec)
        monkeypatch.setattr(wrap_mod, "_smoke_verify_mcp_handshake", lambda *args: handshake_ok)
        monkeypatch.setattr(wrap_mod, "_prime_agy_retrieve_tool_cache", lambda *args: None)
        assert wrap_mod._setup_headroom_retrieve_mcp_agy(registrar) is handshake_ok
        assert not headroom_installed_matching("agy", registrar.get_server("headroom"))
        assert wrap_mod._remove_headroom_installed_retrieve_mcp(registrar) == "not_headroom_owned"
        assert registrar.get_server("headroom") == spec

    @pytest.mark.parametrize("name", ["headroom", "serena"])
    def test_wrap_preserves_conflicting_user_mcp_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
        registrar = AgyRegistrar(home_dir=tmp_path)
        user_spec = ServerSpec(name=name, command="/opt/user-mcp", args=("--custom",), env={})
        registrar.register_server(user_spec)
        result = CliRunner().invoke(_get_main(), ["wrap", "agy"])
        assert result.exit_code == 0, result.output
        assert registrar.get_server(name) == user_spec

    @pytest.mark.parametrize("exit_path", ["normal", "sigterm"])
    def test_marker_follows_bound_port_and_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_path: str
    ) -> None:
        import signal

        import headroom.cli.wrap as wrap_mod

        _stub_agy_mitm_run(tmp_path, monkeypatch, with_uvx=True)
        monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
        requested, actual = 59123, 59124
        monkeypatch.setattr(wrap_mod, "_ensure_proxy", lambda *a, **kw: (None, actual))
        seen = []

        def child(cmd, **kw):
            seen.append(
                (
                    wrap_mod._client_marker_path(requested).exists(),
                    wrap_mod._client_marker_path(actual).exists(),
                )
            )
            if exit_path == "sigterm":
                handler = signal.getsignal(signal.SIGTERM)
                assert callable(handler)
                handler(signal.SIGTERM, None)
            return MagicMock(returncode=0)

        monkeypatch.setattr("subprocess.run", child)
        result = CliRunner().invoke(_get_main(), ["wrap", "agy", "--port", str(requested)])
        assert result.exit_code == (143 if exit_path == "sigterm" else 0), result.output
        assert seen == [(False, True)], "only the bound proxy must see the live session marker"
        assert not wrap_mod._client_marker_path(requested).exists()
        assert not wrap_mod._client_marker_path(actual).exists()


class TestAgyGracefulFailures:
    """agy launch must fail loud and clean on every expected error path.

    WU s04.4: watchdog, preflight, port-in-use, terminal restore.
    """

    @pytest.mark.parametrize("failure", ["dispatch", "terminator", "retrieve"])
    def test_partial_startup_stops_every_initialized_server(
        self, monkeypatch: pytest.MonkeyPatch, failure: str
    ) -> None:
        # Load the retrieve subclass before replacing its dispatch base class.
        import headroom.proxy.agy_retrieve  # noqa: F401
        from headroom.cli.wrap import _start_agy_servers

        active = set()
        stopped = set()

        class Server:
            def __init__(self, name: str):
                self.name = name
                self.address = ("127.0.0.1", 54321)

            async def start(self):
                active.add(self.name)
                if self.name == failure:
                    raise RuntimeError(f"injected {failure} startup failure")

            async def stop(self):
                stopped.add(self.name)
                active.discard(self.name)

        monkeypatch.setattr(
            "headroom.proxy.agy_dispatch.AgyDispatchServer", lambda **kw: Server("dispatch")
        )
        monkeypatch.setattr(
            "headroom.proxy.agy_terminator.AgyCONNECTTerminator", lambda **kw: Server("terminator")
        )
        monkeypatch.setattr(
            "headroom.proxy.agy_retrieve.AgyRetrieveServer", lambda **kw: Server("retrieve")
        )
        with pytest.raises(RuntimeError, match=f"injected {failure} startup failure"):
            _start_agy_servers(None, None, start_retrieve=True)
        assert active == set(), "partial startup must release listener and lifespan resources"
        expected = {"dispatch"}
        if failure != "dispatch":
            expected.add("terminator")
        if failure == "retrieve":
            expected.add("retrieve")
        assert expected <= stopped

    # ------------------------------------------------------------------
    # Shared CA stubs (avoid real cert generation in every test).
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_ca(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "headroom.proxy.agy_ca.ensure_root_ca",
            lambda base_dir=None: (object(), object(), None, None),
        )
        monkeypatch.setattr(
            "headroom.proxy.agy_ca.build_combined_bundle",
            lambda base_dir=None, corp_env_vars=None: "/tmp/fake-bundle.pem",
        )

    # ------------------------------------------------------------------
    # 1. agy-not-installed: clear, actionable error — no raw traceback.
    # ------------------------------------------------------------------

    def test_agy_not_installed_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """agy binary unresolvable → click.ClickException, nonzero exit, no traceback."""
        monkeypatch.setattr("shutil.which", lambda _: None)
        runner = CliRunner()
        result = runner.invoke(_get_main(), ["wrap", "agy"])
        # Must exit non-zero.
        assert result.exit_code != 0
        output = result.output
        # Discriminating: this exact text is produced ONLY by the binary
        # preflight (ClickException). If the preflight were removed, the run
        # would fail elsewhere without this message and the test would fail —
        # so it is NOT satisfied by the wrap banner or downstream errors.
        assert "'agy' not found in PATH" in output
        assert "github.com/google/agy" in output
        # click.ClickException formats with an "Error: " prefix.
        assert "error" in output.lower()
        # Must NOT contain a raw Python traceback.
        assert "Traceback" not in output
        assert "FileNotFoundError" not in output

    # ------------------------------------------------------------------
    # 2. Watchdog: MITM thread death → abort before subprocess, clear message.
    # ------------------------------------------------------------------

    def test_agy_mitm_thread_death_aborts_launch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MITM server startup failure → subprocess NOT invoked, clear error message."""
        import headroom.cli.wrap as wrap_mod

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)
        self._patch_ca(monkeypatch)

        # Simulate _start_agy_servers failing (e.g. the daemon thread dies).
        def _fail_startup(*a, **kw):
            raise RuntimeError(
                "agy MITM server startup failed: connection refused on dispatch bind"
            )

        monkeypatch.setattr(wrap_mod, "_start_agy_servers", _fail_startup)

        subprocess_called: list[list[str]] = []

        def _capture_run(cmd, *a, **kw):
            subprocess_called.append(list(cmd))
            return MagicMock(returncode=0)

        monkeypatch.setattr("subprocess.run", _capture_run)

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["wrap", "agy"])

        # Subprocess (agy) must NOT have been invoked.
        assert subprocess_called == [], (
            "agy subprocess must NOT be launched when the MITM servers fail to start; "
            f"got calls: {subprocess_called}"
        )
        # Must exit non-zero.
        assert result.exit_code != 0
        # Must produce a clear, human-readable message — not a raw exception chain.
        output = result.output.lower()
        assert "error" in output or "failed" in output
        assert "Traceback" not in result.output

    # ------------------------------------------------------------------
    # 3. Port-in-use: OSError(EADDRINUSE) → explicit "port" mention in error.
    # ------------------------------------------------------------------

    def test_agy_port_in_use_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MITM bind failure (EADDRINUSE) → message explicitly names port-in-use problem."""
        import errno

        import headroom.cli.wrap as wrap_mod

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)
        self._patch_ca(monkeypatch)

        bind_error = OSError(errno.EADDRINUSE, "Address already in use")

        def _fail_with_bind_error(*a, **kw):
            # Simulate what _start_agy_servers raises when the async bind fails.
            raise RuntimeError(f"agy MITM server startup failed: {bind_error}") from bind_error

        monkeypatch.setattr(wrap_mod, "_start_agy_servers", _fail_with_bind_error)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: MagicMock(returncode=0))

        runner = CliRunner()
        result = runner.invoke(_get_main(), ["wrap", "agy"])

        assert result.exit_code != 0
        output = result.output.lower()
        # The error message must name the problem as a port conflict, not just
        # re-echo the raw OSError.  The word "port" must appear in isolation
        # (i.e., not just as part of "transport").
        import re

        assert re.search(r"\bport\b", output), (
            f"Expected 'port' (as a word) in output; got: {output!r}"
        )
        # And must still mention that it's in use / unavailable.
        assert "in use" in output or "unavailable" in output or "address already in use" in output
        # Must not be a raw traceback.
        assert "Traceback" not in result.output

    # ------------------------------------------------------------------
    # 4. Terminal/env restore: _stop_agy_servers called in finally on error.
    # ------------------------------------------------------------------

    def test_agy_server_stop_called_on_error_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_stop_agy_servers is called in the finally block even when startup raises."""
        import headroom.cli.wrap as wrap_mod

        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)
        self._patch_ca(monkeypatch)

        monkeypatch.setattr(
            wrap_mod,
            "_start_agy_servers",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        stop_calls: list[object] = []
        original_stop = wrap_mod._stop_agy_servers

        def _spy_stop(servers: object) -> None:
            stop_calls.append(servers)
            original_stop(servers)  # type: ignore[arg-type]

        monkeypatch.setattr(wrap_mod, "_stop_agy_servers", _spy_stop)
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: MagicMock(returncode=0))

        runner = CliRunner()
        runner.invoke(_get_main(), ["wrap", "agy"])

        # The finally block must have called _stop_agy_servers.
        assert len(stop_calls) >= 1, (
            "_stop_agy_servers must run in finally even when startup raises"
        )


# ---------------------------------------------------------------------------
# Regression: unwrap agy removes ALL Headroom-added agy config entries
# ---------------------------------------------------------------------------


class TestUnwrapAgyRemovesAllHeadroomConfig:
    """unwrap agy removes every entry Headroom wrote; user entries survive."""

    def test_unwrap_agy_removes_all_headroom_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All Headroom-added entries gone after unwrap; user entry preserved.

        Arrange a temp HOME with:
        - GEMINI.md containing a headroom-marked block (plus user content)
        - AgyRegistrar config with a ledger-recorded persistent "headroom" retrieve entry
        - AgyRegistrar config with a ledger-recorded serena entry
        - AgyRegistrar config with a user-managed "my-tool" entry (no ledger)

        Act: run `unwrap agy` via CliRunner.

        Assert:
        - GEMINI.md headroom block is removed; user content survives
        - "headroom" retrieve entry is gone
        - serena entry is gone (was ledger-recorded)
        - "my-tool" entry is preserved (never in ledger)
        - ~/.headroom/ca directory is NOT removed (shared CA is headroom state,
          not reverted by unwrap — by design)
        """
        from headroom.cli.wrap import _AGY_GEMINI_BLOCK_END, _AGY_GEMINI_BLOCK_START
        from headroom.mcp_registry.agy import AgyRegistrar
        from headroom.mcp_registry.base import ServerSpec
        from headroom.mcp_registry.install import (
            build_headroom_spec,
            build_serena_spec,
        )
        from headroom.mcp_registry.ledger import record_install

        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # --- Arrange GEMINI.md with headroom block + user content ---
        gemini_md = tmp_path / ".gemini" / "GEMINI.md"
        gemini_md.parent.mkdir(parents=True, exist_ok=True)
        gemini_md.write_text(
            f"# User Instructions\nKeep this.\n\n"
            f"{_AGY_GEMINI_BLOCK_START}\n## Headroom\nContext.\n{_AGY_GEMINI_BLOCK_END}\n"
        )

        # --- Arrange AgyRegistrar entries ---
        reg = AgyRegistrar(home_dir=tmp_path)

        # Persistent, ledger-recorded "headroom" retrieve entry (as wrap agy now
        # installs it: stable spec, env={}, recorded in the ledger). Ledger-gated
        # unwrap removes it because it is Headroom-owned. (A NON-ledgered headroom
        # entry — e.g. a `headroom mcp install` fleet entry — is left in place;
        # that path is covered by test_agy_retrieve_persistent.)
        headroom_spec = build_headroom_spec()
        reg.register_server(headroom_spec)
        record_install("agy", headroom_spec)

        # Headroom-installed serena entry (recorded in ledger).
        serena_spec = build_serena_spec("ide-assistant")
        reg.register_server(serena_spec)
        record_install("agy", serena_spec)

        # User-managed entry: NOT in ledger — must survive.
        user_spec = ServerSpec(name="my-tool", command="/opt/my-tool", args=(), env={})
        reg.register_server(user_spec)

        # Arrange a fake ~/.headroom/ca dir to prove unwrap does NOT touch it.
        ca_dir = tmp_path / ".headroom" / "ca"
        ca_dir.mkdir(parents=True, exist_ok=True)
        (ca_dir / "ca.crt").write_text("fake cert")

        # --- Act ---
        runner = CliRunner()
        result = runner.invoke(_get_main(), ["unwrap", "agy"])
        assert result.exit_code == 0, f"unwrap agy failed:\n{result.output}"

        # --- Assert: GEMINI.md ---
        gemini_text = gemini_md.read_text()
        assert _AGY_GEMINI_BLOCK_START not in gemini_text, (
            "headroom block START marker must be removed from GEMINI.md"
        )
        assert _AGY_GEMINI_BLOCK_END not in gemini_text, (
            "headroom block END marker must be removed from GEMINI.md"
        )
        assert "# User Instructions" in gemini_text, "user content must survive GEMINI.md cleanup"
        assert "Keep this." in gemini_text, "user content body must survive GEMINI.md cleanup"

        # --- Assert: AgyRegistrar entries removed ---
        reg2 = AgyRegistrar(home_dir=tmp_path)
        assert reg2.get_server("headroom") is None, (
            "the ledger-recorded persistent 'headroom' retrieve entry must be removed by unwrap"
        )
        assert reg2.get_server("serena") is None, (
            "the Headroom-installed serena MCP entry must be removed by unwrap"
        )
        # --- Assert: user-managed entry preserved ---
        survived = reg2.get_server("my-tool")
        assert survived is not None, "user-managed 'my-tool' entry must survive unwrap"
        assert survived.command == "/opt/my-tool"

        # --- Assert: CA directory intentionally NOT removed (by design) ---
        assert ca_dir.exists(), "unwrap must NOT remove ~/.headroom/ca (shared headroom CA state)"
        assert (ca_dir / "ca.crt").exists(), "CA certificate must remain intact after unwrap"


# ---------------------------------------------------------------------------
# headroom-n0i.7 — corporate proxy credentials must never leak
# ---------------------------------------------------------------------------


class TestWrapAgyCorpProxyRedaction(TestWrapAgyDisclosureBanner):
    """HTTPS_PROXY / https_proxy userinfo must never reach the launch banner
    or logs, while the host:port is still surfaced for operator visibility."""

    _CORP_PROXY_USER = "user"
    _CORP_PROXY_PASS = "s3cr3t-pw"
    _CORP_PROXY_HOSTPORT = "proxy.example:3128"
    _CORP_PROXY_USERINFO = f"{_CORP_PROXY_USER}:{_CORP_PROXY_PASS}@"
    _CORP_PROXY_URL = f"http://{_CORP_PROXY_USERINFO}{_CORP_PROXY_HOSTPORT}"

    def _invoke_with_corp_proxy(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, *, lowercase: bool
    ):
        caplog.set_level(logging.DEBUG)
        monkeypatch.delenv("HTTPS_PROXY", raising=False)
        monkeypatch.delenv("https_proxy", raising=False)
        monkeypatch.setenv("https_proxy" if lowercase else "HTTPS_PROXY", self._CORP_PROXY_URL)
        return self._invoke_agy(monkeypatch)

    def test_uppercase_https_proxy_credentials_absent_from_banner_and_logs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = self._invoke_with_corp_proxy(monkeypatch, caplog, lowercase=False)
        assert self._CORP_PROXY_USERINFO not in result.output
        assert self._CORP_PROXY_PASS not in result.output
        assert self._CORP_PROXY_USERINFO not in caplog.text
        assert self._CORP_PROXY_PASS not in caplog.text
        assert self._CORP_PROXY_HOSTPORT in result.output

    def test_lowercase_https_proxy_credentials_absent_from_banner_and_logs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        result = self._invoke_with_corp_proxy(monkeypatch, caplog, lowercase=True)
        assert self._CORP_PROXY_USERINFO not in result.output
        assert self._CORP_PROXY_PASS not in result.output
        assert self._CORP_PROXY_USERINFO not in caplog.text
        assert self._CORP_PROXY_PASS not in caplog.text
        assert self._CORP_PROXY_HOSTPORT in result.output


class TestRedactProxyUrl:
    """Parametrized table over every ``redact_proxy_url`` edge case."""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            (
                "http://user:secret@proxy.example:3128",
                "http://proxy.example:3128",
            ),
            ("http://proxy.corp", "http://proxy.corp:80"),
            ("https://proxy.corp", "https://proxy.corp:443"),
            ("http://proxy.corp:abc", _PROXY_URL_REDACTED_PLACEHOLDER),
            ("user:pass@proxy.corp:3128", _PROXY_URL_REDACTED_PLACEHOLDER),
            ("http://u:p@[::1]:8080", "http://[::1]:8080"),
            ("http://u:p@a@h:80", "http://h:80"),
            ("http://ho\x1bst:80", "http://host:80"),
        ],
    )
    def test_redact_proxy_url_table(self, url: str, expected: str) -> None:
        assert redact_proxy_url(url) == expected

    def test_schemeless_url_never_leaks_password(self) -> None:
        assert "pass" not in redact_proxy_url("user:pass@proxy.corp:3128")

    def test_control_bytes_never_reach_result(self) -> None:
        assert "\x1b" not in redact_proxy_url("http://ho\x1bst:80")
