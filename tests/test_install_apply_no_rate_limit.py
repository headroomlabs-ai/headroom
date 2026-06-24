"""Tests that --no-rate-limit is wired through install apply to proxy_args.

Regression for https://github.com/headroomlabs-ai/headroom/issues/1350:
agentic CLI targets (Claude Code, Codex) burst above the 60 req/min default,
causing the proxy to self-throttle instead of forwarding to Anthropic.
--no-rate-limit on `headroom proxy` existed, but `headroom install apply`
had no flag to persist it — reinstalls silently reintroduced throttling.
"""

from __future__ import annotations

import pytest

from headroom.install.planner import build_manifest


_BASE_KWARGS = dict(
    profile="default",
    preset="persistent-service",
    runtime_kind="python",
    scope="user",
    provider_mode="auto",
    targets=[],
    port=8787,
    backend="anthropic",
    anyllm_provider=None,
    region=None,
    proxy_mode="token",
    memory_enabled=False,
    telemetry_enabled=False,
    image="ghcr.io/chopratejas/headroom:latest",
)


def test_no_rate_limit_false_by_default() -> None:
    """Default manifest must NOT include --no-rate-limit in proxy_args."""
    manifest = build_manifest(**_BASE_KWARGS)
    assert "--no-rate-limit" not in manifest.proxy_args


def test_no_rate_limit_true_appends_flag() -> None:
    """no_rate_limit=True must append --no-rate-limit to proxy_args."""
    manifest = build_manifest(**_BASE_KWARGS, no_rate_limit=True)
    assert "--no-rate-limit" in manifest.proxy_args


def test_no_rate_limit_flag_position() -> None:
    """--no-rate-limit must appear after --no-telemetry in proxy_args
    so the manifest is deterministic and diffable."""
    manifest = build_manifest(**_BASE_KWARGS, no_rate_limit=True)
    args = manifest.proxy_args
    assert "--no-telemetry" in args
    assert args.index("--no-rate-limit") > args.index("--no-telemetry")


def test_no_rate_limit_idempotent_on_reinstall() -> None:
    """Rebuilding the manifest with the same flag must not duplicate it."""
    manifest1 = build_manifest(**_BASE_KWARGS, no_rate_limit=True)
    manifest2 = build_manifest(**_BASE_KWARGS, no_rate_limit=True)
    assert manifest1.proxy_args.count("--no-rate-limit") == 1
    assert manifest2.proxy_args.count("--no-rate-limit") == 1


def test_cli_flag_wired_through_install_apply() -> None:
    """headroom install apply --no-rate-limit must produce a manifest
    with --no-rate-limit in proxy_args (integration: CLI → planner)."""
    pytest.importorskip("fastapi")
    from click.testing import CliRunner
    from headroom.cli.install import install_apply

    runner = CliRunner()
    # --dry-run or equivalent: we just check the manifest is built correctly.
    # Since install_apply does real side-effects, we patch the heavy parts.
    from unittest.mock import patch, MagicMock

    built_manifests = []

    def capture_manifest(**kwargs):
        m = build_manifest(**kwargs)
        built_manifests.append(m)
        return m

    with (
        patch("headroom.cli.install.build_manifest", side_effect=capture_manifest),
        patch("headroom.cli.install.load_manifest", return_value=None),
        patch("headroom.cli.install.apply_mutations", return_value=[]),
        patch("headroom.cli.install.install_supervisor", return_value=[]),
        patch("headroom.cli.install.save_manifest"),
        patch("headroom.cli.install._start_deployment"),
        patch("headroom.cli.install.click.echo"),
    ):
        result = runner.invoke(
            install_apply,
            ["--no-rate-limit", "--scope", "user"],
            catch_exceptions=False,
        )

    assert built_manifests, "build_manifest was never called"
    assert "--no-rate-limit" in built_manifests[0].proxy_args, (
        f"--no-rate-limit not in proxy_args: {built_manifests[0].proxy_args}"
    )
