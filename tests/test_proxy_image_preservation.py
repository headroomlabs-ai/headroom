from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.proxy.server import ProxyConfig, create_app
from headroom.transforms import ContentRouter


@pytest.mark.parametrize("selection", [None, {"image", "kompress"}])
def test_image_disable_preserves_text_compression(selection):
    app = create_app(
        ProxyConfig(
            image_optimize=False,
            compressors=selection,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )
    router = next(
        t for t in app.state.proxy.anthropic_pipeline.transforms if isinstance(t, ContentRouter)
    )
    assert not router.config.enable_image_optimizer
    assert router.config.enable_kompress


@pytest.mark.parametrize("enabled", [True, False])
def test_cli_image_policy_reaches_proxy(monkeypatch, enabled):
    monkeypatch.setenv("HEADROOM_IMAGE_OPTIMIZE", "1" if enabled else "0")
    monkeypatch.setenv("HEADROOM_MALLOC_TUNING", "0")
    with patch("headroom.proxy.server.run_server") as run:
        result = CliRunner().invoke(main, ["proxy", "--no-telemetry"])
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].image_optimize is enabled
