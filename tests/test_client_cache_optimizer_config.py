"""The effective config owns SDK cache optimization and its setup provenance."""

import pytest
from openai import OpenAI

from headroom import HeadroomClient, OpenAIProvider
from headroom.cache.openai import OpenAICacheOptimizer
from headroom.config import HeadroomConfig
from headroom.pipeline import PipelineStage


class SetupRecorder:
    def __init__(self):
        self.events = []

    def on_pipeline_event(self, event):
        if event.stage == PipelineStage.SETUP:
            self.events.append(event)


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("constructor", [False, True])
def test_config_cache_enabled_takes_precedence(tmp_path, configured, constructor):
    recorder = SetupRecorder()
    url = f"sqlite:///{tmp_path}/metrics.db"
    config = HeadroomConfig(
        store_url=url,
        discover_pipeline_extensions=False,
        pipeline_extensions=[recorder],
    )
    config.cache_optimizer.enabled = configured
    with OpenAI(api_key="local-fixture") as original:
        with HeadroomClient(
            original,
            OpenAIProvider(),
            store_url=url,
            config=config,
            enable_cache_optimizer=constructor,
        ) as client:
            expected = "openai-prefix-stabilizer" if configured else None
            assert client.get_stats()["config"]["cache_optimizer"] == expected
    assert recorder.events[0].metadata["cache_optimizer_enabled"] is configured


def test_disabled_config_also_skips_explicit_optimizer(tmp_path):
    url = f"sqlite:///{tmp_path}/metrics.db"
    config = HeadroomConfig(store_url=url, discover_pipeline_extensions=False)
    config.cache_optimizer.enabled = False
    with OpenAI(api_key="local-fixture") as original:
        with HeadroomClient(
            original,
            OpenAIProvider(),
            store_url=url,
            config=config,
            cache_optimizer=OpenAICacheOptimizer(),
        ) as client:
            assert client.get_stats()["config"]["cache_optimizer"] is None
