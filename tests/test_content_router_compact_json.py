"""Compact machine-generated JSON must not evade compression (token-estimate bug).

Whitespace-split token counting made a compact JSON payload (no spaces —
the default output of json.dumps with separators, JSON.stringify, boto3)
count as ~1 token, so compression ratios computed as ~1.0 and the
min_ratio gate rejected SmartCrusher's real output.
"""

from __future__ import annotations

import json
import random

import pytest

from headroom.transforms.content_router import (
    ContentRouter,
    ContentRouterConfig,
    _estimate_tokens,
)


def test_estimate_tokens_monotone_on_compact_json() -> None:
    small = json.dumps([{"a": 1}] * 5, separators=(",", ":"))
    large = json.dumps([{"a": 1}] * 500, separators=(",", ":"))
    assert _estimate_tokens(large) > _estimate_tokens(small) > 1


def test_compact_json_tool_result_compresses() -> None:
    tokenizer = pytest.importorskip("headroom.tokenizers.estimator")
    from headroom.tokenizer import Tokenizer

    tok = Tokenizer(tokenizer.EstimatingTokenCounter())
    random.seed(42)
    rows = [
        {
            "serviceArn": f"arn:aws:ecs:us-east-1:123456789012:service/x/svc-{s:03d}",
            "serviceName": f"svc-{s:03d}",
            "status": "ACTIVE",
            "desiredCount": random.randint(1, 6),
            "runningCount": random.randint(0, 6),
        }
        for s in range(150)
    ]
    payload = json.dumps(rows, separators=(",", ":"))
    assert " " not in payload[:200]  # genuinely compact
    messages = [
        {"role": "user", "content": "Investigate."},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_1", "name": "list_services", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": payload}],
        },
        {"role": "user", "content": "Summarize in one sentence."},
    ]
    router = ContentRouter(ContentRouterConfig(skip_user_messages=False))
    before = tok.count_messages(messages)
    result = router.apply(
        [json.loads(json.dumps(m)) for m in messages],
        tok,
        context="Summarize",
        frozen_message_count=0,
    )
    after = tok.count_messages(result.messages)
    assert after < before * 0.9, (before, after, result.transforms_applied[:5])
