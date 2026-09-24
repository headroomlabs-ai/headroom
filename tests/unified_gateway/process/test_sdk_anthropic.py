from __future__ import annotations

import anthropic
import pytest


def test_anthropic_sdk_receives_local_unknown_model_error(gateway_process) -> None:
    client = anthropic.Anthropic(
        api_key="client-secret",
        base_url=gateway_process.base_url,
        max_retries=0,
    )
    with pytest.raises(anthropic.NotFoundError):
        client.messages.create(
            model="not-granted",
            max_tokens=1,
            messages=[{"role": "user", "content": "must-not-egress"}],
        )
