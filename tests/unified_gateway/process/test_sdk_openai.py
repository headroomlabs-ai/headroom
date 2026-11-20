from __future__ import annotations

from openai import OpenAI


def test_openai_sdk_lists_only_principal_visible_models(gateway_process) -> None:
    client = OpenAI(
        api_key="client-secret",
        base_url=f"{gateway_process.base_url}/v1",
        max_retries=0,
    )
    models = client.models.list()
    assert [model.id for model in models.data] == [
        "REPLACE_WITH_ENABLED_ANTHROPIC_MODEL",
        "REPLACE_WITH_ENABLED_GEMINI_MODEL",
        "REPLACE_WITH_ENABLED_OPENAI_MODEL",
    ]
