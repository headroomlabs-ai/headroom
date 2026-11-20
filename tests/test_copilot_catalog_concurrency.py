"""Per-request isolation and refresh behaviour on the catalog hot path.

Both properties here were broken in review and neither had coverage:

* the transport plan was stashed on the process-wide proxy and read back across
  an ``await``, so one request could be bridged using another request's model
  capabilities — reinstating the very ``reasoning_effort`` 400 the bridge exists
  to avoid, nondeterministically, under subagent fan-out;
* a single failed ``/models`` fetch overwrote a good catalog with ``{}`` and
  pinned that for the full TTL, silently reverting routing to the name heuristic.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from headroom.models.copilot_catalog import CopilotModelCatalog, parse_models_payload

FIXTURE = Path(__file__).parent / "fixtures" / "copilot_models" / "models_list.json"


@pytest.fixture(scope="module")
def cards() -> dict[str, Any]:
    return parse_models_payload(json.loads(FIXTURE.read_text(encoding="utf-8")))


def _key(catalog: CopilotModelCatalog, fp: str = "aaaa") -> tuple[str, str, str]:
    return catalog.cache_key(
        base_url="https://api.githubcopilot.com", integration_id="vscode-chat", token_fingerprint=fp
    )


# ---------------------------------------------------------------------------
# No per-request state on the shared proxy object
# ---------------------------------------------------------------------------


def test_plan_is_never_stashed_on_the_proxy_instance() -> None:
    """Structural guard against the singleton-state regression returning.

    `_plan_responses_upstream_path` must RETURN the plan. If it ever stashes it on
    `self` again, per-request state lives on a process-wide object and can leak
    between concurrent requests.
    """
    import ast
    import inspect
    import textwrap

    from headroom.proxy.handlers.openai import OpenAIHandlerMixin

    def assigns_or_reads_plan_on_self(func: Any) -> bool:
        """True if the function stores/reads a `_last_*` attribute on `self`.

        Parsed rather than grepped: the docstrings deliberately name the removed
        attribute to explain why it must not come back, so a substring check
        would match its own warning.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        return any(
            isinstance(node, ast.Attribute)
            and node.attr.startswith("_last_")
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            for node in ast.walk(tree)
        )

    assert not assigns_or_reads_plan_on_self(OpenAIHandlerMixin._plan_responses_upstream_path)
    assert not assigns_or_reads_plan_on_self(OpenAIHandlerMixin.handle_openai_responses)


def test_fallback_plan_is_consistent_with_its_path() -> None:
    """Every fallback must return a plan matching the path it routes to.

    `card=None` on a bridged route must drop `reasoning_effort` — the safe,
    pre-existing behaviour. A `None` plan reaching the bridge would forward the
    caller's effort verbatim to a model that may reject it.
    """
    from headroom.proxy.transport_planner import (
        CHAT_COMPLETIONS_PATH,
        RESPONSES_PATH,
        plan_transport,
    )

    plan = plan_transport(
        inbound_path=RESPONSES_PATH,
        card=None,
        heuristic_prefers_responses=False,
        reasoning_effort="xhigh",
    )
    assert plan.upstream_path == CHAT_COMPLETIONS_PATH
    assert plan.request_bridge == "responses->chat"
    assert "reasoning_effort" in plan.drop_fields


# ---------------------------------------------------------------------------
# Refresh failure must not destroy a good catalog
# ---------------------------------------------------------------------------


def test_transient_failure_keeps_the_previous_catalog(cards: dict[str, Any]) -> None:
    catalog = CopilotModelCatalog()
    key = _key(catalog)
    catalog.put(key, cards)
    catalog.note_fetch_failure(key)
    still = catalog.cards(key)
    assert still is not None
    assert "mai-code-1-flash-picker" in still, "a good catalog was destroyed by a failed refresh"


def test_failure_suppresses_refetch_only_briefly(cards: dict[str, Any]) -> None:
    catalog = CopilotModelCatalog()
    key = _key(catalog)
    catalog.put(key, cards)
    catalog.note_fetch_failure(key, now=1000.0)
    assert catalog.refresh_suppressed(key, now=1030.0) is True  # inside the 60s backoff
    assert catalog.refresh_suppressed(key, now=1100.0) is False  # not the full 900s TTL


def test_success_clears_a_prior_failure(cards: dict[str, Any]) -> None:
    catalog = CopilotModelCatalog()
    key = _key(catalog)
    catalog.note_fetch_failure(key, now=1000.0)
    catalog.put(key, cards, now=1001.0)
    assert catalog.refresh_suppressed(key, now=1002.0) is False


def test_cards_enforces_the_staleness_bound(cards: dict[str, Any]) -> None:
    """`cards()` is the only guarded accessor; the request path must use it."""
    catalog = CopilotModelCatalog()
    key = _key(catalog)
    catalog.put(key, cards, now=0.0)
    assert catalog.cards(key, now=100.0) is not None
    assert catalog.cards(key, now=90_000.0) is None


def test_empty_catalog_reads_as_absent(cards: dict[str, Any]) -> None:
    catalog = CopilotModelCatalog()
    key = _key(catalog)
    catalog.put(key, {})
    assert catalog.cards(key) is None


# ---------------------------------------------------------------------------
# Single-flight
# ---------------------------------------------------------------------------


def test_fetch_lock_is_per_credential_and_stable() -> None:
    catalog = CopilotModelCatalog()
    a, b = _key(catalog, "aaaa"), _key(catalog, "bbbb")
    assert catalog.fetch_lock(a) is catalog.fetch_lock(a)
    assert catalog.fetch_lock(a) is not catalog.fetch_lock(b)


@pytest.mark.asyncio
async def test_lock_serialises_a_concurrent_wave(cards: dict[str, Any]) -> None:
    """One fetch must serve a fan-out, not N parallel /models GETs."""
    catalog = CopilotModelCatalog()
    key = _key(catalog)
    fetches = 0

    async def fetch_once() -> None:
        nonlocal fetches
        async with catalog.fetch_lock(key):
            if catalog.cards(key) is not None:
                return
            fetches += 1
            await asyncio.sleep(0)
            catalog.put(key, cards)

    await asyncio.gather(*(fetch_once() for _ in range(8)))
    assert fetches == 1, f"{fetches} fetches issued for one credential"


# ---------------------------------------------------------------------------
# The cache identity names the integration the traffic actually goes out under
# ---------------------------------------------------------------------------


def test_catalog_key_follows_the_clients_integration_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two integrations on one credential must never share a catalog entry.

    ``apply_copilot_api_auth`` sends the client's own ``Copilot-Integration-Id``
    when it has one (falling back to ``GITHUB_COPILOT_INTEGRATION_ID``), and
    ``/models`` can answer differently per integration. Keying on the env
    default alone let the first integration's catalog route the other's
    requests.
    """
    from types import SimpleNamespace

    from starlette.datastructures import Headers

    from headroom.proxy.handlers.openai import OpenAIHandlerMixin

    monkeypatch.delenv("GITHUB_COPILOT_INTEGRATION_ID", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_API_TOKEN", raising=False)
    handler = SimpleNamespace(
        _resolve_openai_upstream=lambda request: "https://api.githubcopilot.com"
    )

    def key_for(**headers: str) -> tuple[str, str, str]:
        request = SimpleNamespace(
            headers=Headers(headers={"authorization": "Bearer gho_same", **headers})
        )
        return OpenAIHandlerMixin._copilot_catalog_key(handler, request)  # type: ignore[arg-type]

    vscode = key_for(**{"copilot-integration-id": "vscode-chat"})
    cli = key_for(**{"copilot-integration-id": "copilot-developer-cli"})
    assert vscode != cli, "two integrations shared one catalog entry"
    assert vscode[0] == cli[0] and vscode[2] == cli[2], "only the integration should differ"

    # Without a client header the configured default applies, exactly as auth does.
    monkeypatch.setenv("GITHUB_COPILOT_INTEGRATION_ID", "copilot-developer-cli")
    assert key_for() == cli
