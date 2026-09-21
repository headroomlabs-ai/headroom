#!/usr/bin/env python3
"""Demo: Jev-backed semantic model routing for Headroom (#3690).

Docs: https://docs.typesafe.ai/introduction

Offline (default) — uses a mocked System One response:

    python examples/jev_model_router/demo.py

Live — requires TYPESAFE_API_KEY (never commit the key):

    export TYPESAFE_API_KEY=...
    export HEADROOM_JEV_TIER_MODELS='{"economy":"gpt-mini","standard":"gpt","frontier":"gpt-pro"}'
    python examples/jev_model_router/demo.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from headroom.proxy.jev_model_policy import JevModelPolicy, JevModelPolicyConfig

SAMPLES = [
    ("Rename the helper foo to bar in utils.py", "economy-ish"),
    ("Add a unit test for the date parser", "standard-ish"),
    (
        "Redesign our OAuth refresh flow to eliminate the race on concurrent "
        "token rotation across three services, and propose a migration plan.",
        "frontier-ish",
    ),
]


def _mock_post_factory():
    import httpx

    def post(url, **kwargs):
        state = str(kwargs.get("json", {}).get("state", "")).lower()
        if "rename" in state or "foo to bar" in state:
            tier, conf = "economy", 0.92
        elif "oauth" in state or "redesign" in state:
            tier, conf = "frontier", 0.88
        else:
            tier, conf = "standard", 0.8
        body = {
            "model": "jev-1.13.0",
            "answers": {
                "tier": {
                    "type": "choice",
                    "choice": tier,
                    "confidence": conf,
                    "probabilities": {tier: conf},
                },
                "difficulty": {
                    "type": "score",
                    "score": {"economy": 0.2, "standard": 1.0, "frontier": 2.0}[tier],
                    "confidence": 0.9,
                },
            },
        }
        return httpx.Response(
            200,
            json=body,
            request=httpx.Request("POST", url),
        )

    return post


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call api.typesafe.ai (requires TYPESAFE_API_KEY)",
    )
    args = parser.parse_args()

    tier_models = {
        "economy": "economy-model",
        "standard": "standard-model",
        "frontier": "frontier-model",
    }
    raw_tiers = os.environ.get("HEADROOM_JEV_TIER_MODELS")
    if raw_tiers:
        tier_models = json.loads(raw_tiers)

    if args.live:
        key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not key:
            print("TYPESAFE_API_KEY required for --live", file=sys.stderr)
            return 2
        policy = JevModelPolicy(
            JevModelPolicyConfig(enabled=True, api_key=key, tier_models=tier_models)
        )
        print("mode: live TypeSafe System One")
    else:
        policy = JevModelPolicy(
            JevModelPolicyConfig(enabled=True, api_key="demo", tier_models=tier_models),
            post=_mock_post_factory(),
        )
        print("mode: offline mock")

    source = "frontier-model"
    for text, hint in SAMPLES:
        decision = policy.select(model=source, user_text=text)
        print("---")
        print(f"hint={hint}")
        print(f"user={text[:80]}{'…' if len(text) > 80 else ''}")
        print(f"changed={decision.changed} routed={decision.routed_model}")
        print(f"reason={decision.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
