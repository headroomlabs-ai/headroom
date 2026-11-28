"""Tests for headroom.providers.agy env builder.

TDD: written before implementation — all tests should fail on first run.
"""

from __future__ import annotations

from pathlib import Path

from headroom.providers.agy.runtime import build_agy_env


class TestBuildAgyEnv:
    """Pure-function tests for build_agy_env."""

    def test_sets_https_and_http_proxy_to_terminator(self, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle.pem"
        bundle.touch()
        env = build_agy_env(
            terminator_url="http://127.0.0.1:54321",
            bundle_path=bundle,
            base_env={},
        )
        assert env["HTTPS_PROXY"] == "http://127.0.0.1:54321"
        assert env["HTTP_PROXY"] == "http://127.0.0.1:54321"

    def test_sets_no_proxy_loopback(self, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle.pem"
        bundle.touch()
        env = build_agy_env(
            terminator_url="http://127.0.0.1:54321",
            bundle_path=bundle,
            base_env={},
        )
        assert env["NO_PROXY"] == "127.0.0.1,localhost"

    def test_preserves_inherited_no_proxy_entries(self, tmp_path: Path) -> None:
        """A corporate NO_PROXY names hosts that must bypass the proxy.

        Replacing it would tunnel them through the terminator; the loopback
        entries are prepended to what the user already had.
        """
        bundle = tmp_path / "bundle.pem"
        bundle.touch()
        env = build_agy_env(
            terminator_url="http://127.0.0.1:54321",
            bundle_path=bundle,
            base_env={"NO_PROXY": "internal.corp,10.0.0.0/8"},
        )
        assert env["NO_PROXY"] == "127.0.0.1,localhost,internal.corp,10.0.0.0/8"

    def test_session_scoped_savings_vars_are_not_inherited(self, tmp_path: Path) -> None:
        """The wrapper redirects its OWN funnel to a temp dir deleted at exit.

        The agy child (and the `headroom mcp serve` grandchild it spawns) must
        not inherit that redirection, or it writes savings into a sink that
        disappears and marks itself as an inbox emitter.
        """
        bundle = tmp_path / "bundle.pem"
        bundle.touch()
        env = build_agy_env(
            terminator_url="http://127.0.0.1:54321",
            bundle_path=bundle,
            base_env={
                "HEADROOM_AGY_INBOX_EMIT": "1",
                "HEADROOM_SAVINGS_PATH": "/tmp/gone/proxy_savings.json",
                "HEADROOM_SAVINGS_EVENTS_PATH": "/tmp/gone/savings_events.jsonl",
                "HEADROOM_OTEL_METRICS_ENABLED": "0",
                "PATH": "/usr/bin",
            },
        )
        assert "HEADROOM_AGY_INBOX_EMIT" not in env
        assert "HEADROOM_SAVINGS_PATH" not in env
        assert "HEADROOM_SAVINGS_EVENTS_PATH" not in env
        assert "HEADROOM_OTEL_METRICS_ENABLED" not in env
        assert env["PATH"] == "/usr/bin"

    def test_sets_all_three_ca_vars_to_bundle(self, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle.pem"
        bundle.touch()
        env = build_agy_env(
            terminator_url="http://127.0.0.1:54321",
            bundle_path=bundle,
            base_env={},
        )
        assert env["SSL_CERT_FILE"] == str(bundle)
        assert env["CACERT_PATH"] == str(bundle)
        assert env["NODE_EXTRA_CA_CERTS"] == str(bundle)

    def test_corp_proxy_not_leaked_into_child_and_base_env_unmutated(self, tmp_path: Path) -> None:
        """A pre-existing corporate HTTPS_PROXY must NOT leak into the child agy
        env as its proxy (the child must talk to the terminator), and build_agy_env
        must NOT mutate base_env — so the terminator, running in the PARENT process,
        still reads the original corporate os.environ["HTTPS_PROXY"] for chaining
        non-allowlisted CONNECTs."""
        bundle = tmp_path / "bundle.pem"
        bundle.touch()
        upstream = "http://corp-proxy.internal:3128"
        base_env = {"HTTPS_PROXY": upstream}
        env = build_agy_env(
            terminator_url="http://127.0.0.1:54321",
            bundle_path=bundle,
            base_env=base_env,
        )
        # Child agy talks to the terminator, NOT the corp proxy directly.
        assert env["HTTPS_PROXY"] == "http://127.0.0.1:54321"
        # Corp proxy is preserved in the caller's env (parent keeps it for the
        # terminator's blind-tunnel chaining); build_agy_env never clobbers it.
        assert base_env["HTTPS_PROXY"] == upstream
        # No dead chaining var is fabricated in the child env.
        assert "HEADROOM_UPSTREAM_HTTPS_PROXY" not in env

    def test_base_env_merged_into_result(self, tmp_path: Path) -> None:
        """Other base_env keys must be present in the returned dict."""
        bundle = tmp_path / "bundle.pem"
        bundle.touch()
        env = build_agy_env(
            terminator_url="http://127.0.0.1:54321",
            bundle_path=bundle,
            base_env={"MY_CUSTOM_KEY": "my_value"},
        )
        assert env["MY_CUSTOM_KEY"] == "my_value"

    def test_returns_new_dict_does_not_mutate_base_env(self, tmp_path: Path) -> None:
        bundle = tmp_path / "bundle.pem"
        bundle.touch()
        base = {"SOME_KEY": "val"}
        result = build_agy_env(
            terminator_url="http://127.0.0.1:54321",
            bundle_path=bundle,
            base_env=base,
        )
        assert result is not base
        assert base == {"SOME_KEY": "val"}
