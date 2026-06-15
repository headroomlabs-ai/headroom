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
    """Verify that wrap agy does NOT register an ephemeral per-run MCP entry.

    The agy dispatch server binds an ephemeral port (port=0) that dies when the
    session exits.  Registering it in the persistent mcp_config.json would leave
    a dead pointer for the next session.  The correct policy is N/A-v1: the
    AgyRegistrar is available for stable-proxy scenarios via 'headroom mcp
    install', but no registration occurs during a wrap-agy run.
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
    monkeypatch.setattr(
        wrap_mod, "_start_agy_servers", lambda ca_key, ca_cert, base_dir=None: fake_servers
    )
    monkeypatch.setattr(wrap_mod, "_stop_agy_servers", lambda s: None)

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
