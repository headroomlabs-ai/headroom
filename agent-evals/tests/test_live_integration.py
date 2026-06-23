"""Opt-in LIVE tests (``-m live``): spawn a real ``headroom proxy`` and hit a real upstream.

Skipped unless the ``anthropic`` SDK is installed AND ``ANTHROPIC_API_KEY`` is set. These encode
the Phase-0 acceptance criteria that cannot be checked without real infra:

* ``test_a0_a1_transparency`` — passthrough (A1) must produce the same round-trip as direct (A0):
  the proxy hop alters nothing. Uses a constrained, deterministic echo prompt at temperature 0 so
  the assertion isolates proxy fidelity from model sampling noise.
* ``test_b_arm_captures_savings_live`` — the B arm emits ``x-headroom-*`` headers that the client
  shim captures into a SavingsStore, and ``/stats`` is readable for run-level reconciliation.

Keys are read from the environment only; nothing here writes or logs a key.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from agent_evals.arms import HeadroomArm
from agent_evals.config import Settings
from agent_evals.metrics.savings import SavingsStore, fetch_run_savings, make_response_hook
from agent_evals.models import ArmName, ArmSpec, Provider, ProxyMode

pytestmark = pytest.mark.live

_ECHO_PROMPT = [{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}]


def _require_anthropic() -> object:
    anthropic = pytest.importorskip("anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    return anthropic


def _arm(
    name: ArmName, mode: ProxyMode | None, settings: Settings, run_dir: Path, **kw: object
) -> HeadroomArm:
    spec = ArmSpec(name=name, provider=Provider.ANTHROPIC, proxy_mode=mode, label=name.value)
    return HeadroomArm(spec, settings, run_dir, **kw)  # type: ignore[arg-type]


async def test_a0_a1_transparency(tmp_path: Path) -> None:
    anthropic = _require_anthropic()
    settings = Settings()

    async with _arm(ArmName.A0_DIRECT, None, settings, tmp_path) as h0:
        c0 = anthropic.Anthropic(base_url=h0.base_url)  # type: ignore[attr-defined]
        r0 = c0.messages.create(
            model=settings.model_snapshot, max_tokens=16, temperature=0, messages=_ECHO_PROMPT
        )
    async with _arm(ArmName.A1_PASSTHROUGH, ProxyMode.OFF, settings, tmp_path) as h1:
        c1 = anthropic.Anthropic(base_url=h1.base_url)  # type: ignore[attr-defined]
        r1 = c1.messages.create(
            model=settings.model_snapshot, max_tokens=16, temperature=0, messages=_ECHO_PROMPT
        )

    text0 = "".join(b.text for b in r0.content if b.type == "text").strip()
    text1 = "".join(b.text for b in r1.content if b.type == "text").strip()
    # Passthrough must not alter the round-trip relative to talking to the provider directly.
    assert text0 == text1
    assert r0.model == r1.model


async def test_b_arm_captures_savings_live(tmp_path: Path) -> None:
    anthropic = _require_anthropic()
    settings = Settings()
    pricing = settings.pricing
    store = SavingsStore()
    task_id = "live-t1"
    hook = make_response_hook(store, lambda: task_id, pricing)

    arm = _arm(
        ArmName.B_HEADROOM,
        ProxyMode.TOKEN,
        settings,
        tmp_path,
        savings_provider=lambda tid: store.aggregate(tid, pricing),
    )
    async with arm as handle:
        client = anthropic.Anthropic(  # type: ignore[attr-defined]
            base_url=handle.base_url,
            http_client=httpx.Client(event_hooks={"response": [hook]}),
        )
        bulky = "repetitive tool output line\n" * 800
        client.messages.create(
            model=settings.model_snapshot,
            max_tokens=16,
            temperature=0,
            messages=[{"role": "user", "content": bulky + "\nReply with the word OK."}],
        )
        captured = handle.capture_savings(task_id)
        with httpx.Client() as stats_client:
            run = fetch_run_savings(handle.base_url + settings.proxy.stats_path, stats_client)

    # The proxy always emits token headers on the per-request path, so capture must succeed.
    assert captured is not None
    assert captured.tokens_before >= captured.tokens_after >= 0
    assert captured.source == "headers"
    # /stats is readable for run-level reconciliation (lifetime aggregate, not asserted tight).
    assert run.cache_read_tokens >= 0
