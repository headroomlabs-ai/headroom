"""Catalog + transport-planner tests, driven off a real Copilot ``/models`` body.

The fixture in ``tests/fixtures/copilot_models/models_list.json`` is a verbatim
capture from ``GET https://api.githubcopilot.com/models`` (account-specific
billing text stripped). Using the real payload matters: every defect these tests
guard against was found *because* the live data does not look like the tidy
shape a hand-written fixture would have -- 17 of 40 entries publish no
``supported_endpoints``, 17 publish no ``policy``, and endpoint ordering is
inconsistent within a single vendor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.models.copilot_catalog import (
    BRIDGEABLE_ENDPOINTS,
    CopilotModelCatalog,
    ModelCard,
    parse_models_payload,
    resolve_model_id,
)
from headroom.proxy.transport_planner import (
    CHAT_COMPLETIONS_PATH,
    RESPONSES_PATH,
    plan_transport,
)

FIXTURE = Path(__file__).parent / "fixtures" / "copilot_models" / "models_list.json"


@pytest.fixture(scope="module")
def cards() -> dict[str, ModelCard]:
    return parse_models_payload(json.loads(FIXTURE.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------


def test_parses_every_live_model(cards: dict[str, ModelCard]) -> None:
    assert len(cards) == 40


def test_absent_supported_endpoints_means_no_constraint(cards: dict[str, ModelCard]) -> None:
    """The single most common shape in the live data.

    17/40 entries omit ``supported_endpoints`` entirely. Reading that as an
    empty allow-list would break the whole ``gpt-4*``/``gpt-3.5*`` family, so it
    must mean "the upstream published no constraint".
    """
    unconstrained = [c for c in cards.values() if not c.constrains_endpoints()]
    assert len(unconstrained) == 17
    gpt4o = cards["gpt-4o"]
    assert gpt4o.endpoints == ()
    assert gpt4o.constrains_endpoints() is False
    # "No constraint" must answer True for anything, not False for everything.
    assert gpt4o.supports_endpoint(CHAT_COMPLETIONS_PATH) is True
    assert gpt4o.supports_endpoint(RESPONSES_PATH) is True


def test_absent_policy_means_enabled_not_denied(cards: dict[str, ModelCard]) -> None:
    """``gpt-5.3-codex`` has no ``policy`` object but is picker-enabled.

    Requiring ``policy.state == "enabled"`` would permanently hide a flagship
    coding model. Absent policy means there is no terms gate.
    """
    codex = cards["gpt-5.3-codex"]
    assert codex.selectable is True
    assert sum(1 for c in cards.values() if c.selectable) > 15


def test_embeddings_are_not_chat_models(cards: dict[str, ModelCard]) -> None:
    assert cards["text-embedding-3-small"].is_chat_model is False
    assert cards["claude-opus-4.8"].is_chat_model is True


def test_capability_fields_are_read_from_live_data(cards: dict[str, ModelCard]) -> None:
    opus = cards["claude-opus-4.6"]
    assert opus.vendor == "Anthropic"
    assert opus.tier == "powerful"
    assert opus.display_name == "Claude Opus 4.6"
    # Bridged requests are buffered, so the non-streaming cap is load-bearing.
    assert opus.max_non_streaming_output_tokens == 16000
    assert opus.max_output_tokens == 64000


# ---------------------------------------------------------------------------
# reasoning_effort is a value set, not a flag  (verified live: see below)
# ---------------------------------------------------------------------------


def test_reasoning_effort_value_sets_differ_per_model(cards: dict[str, ModelCard]) -> None:
    assert "xhigh" not in cards["claude-opus-4.6"].reasoning_efforts
    assert "max" in cards["claude-opus-4.6"].reasoning_efforts
    assert "xhigh" in cards["gpt-5.4"].reasoning_efforts
    assert "max" not in cards["gpt-5.4"].reasoning_efforts
    assert cards["kimi-k2.7-code"].reasoning_efforts == ()


@pytest.mark.parametrize(
    ("model", "requested", "expected"),
    [
        # Each row was confirmed against the live API: the "reject" rows
        # returned 400 and the "accept" rows returned 200.
        # Equidistant between `high` and `max`; ties resolve DOWN, because
        # reasoning effort is spend and a silent upgrade is the worse surprise.
        ("claude-opus-4.6", "xhigh", "high"),  # live: xhigh -> 400
        ("claude-opus-4.6", "max", "max"),  # live: max   -> 200
        ("gpt-5.4", "max", "xhigh"),  # live: max   -> 400
        ("gpt-5.4", "xhigh", "xhigh"),  # live: xhigh -> 200
        ("gemini-3.5-flash", "minimal", "minimal"),  # live: minimal -> 200
        ("kimi-k2.7-code", "medium", None),  # live: medium -> 400
    ],
)
def test_reasoning_effort_clamps_to_nearest_supported(
    cards: dict[str, ModelCard], model: str, requested: str, expected: str | None
) -> None:
    assert cards[model].clamp_reasoning_effort(requested) == expected


# ---------------------------------------------------------------------------
# Transport planning
# ---------------------------------------------------------------------------


def test_responses_only_model_is_never_downgraded(cards: dict[str, ModelCard]) -> None:
    """The regression a name-based heuristic causes.

    ``mai-code-1-flash-picker`` is served only on ``/responses`` but does not
    match ``gpt-5*/o1*/o3*``, so the heuristic answers "prefers chat" and a
    name-driven downgrade routes it to ``/chat/completions``. Live, that is a
    400; on ``/responses`` it is a 200.
    """
    plan = plan_transport(
        inbound_path=RESPONSES_PATH,
        card=cards["mai-code-1-flash-picker"],
        heuristic_prefers_responses=False,  # what the name heuristic says
    )
    assert plan.upstream_path == RESPONSES_PATH
    assert plan.request_bridge is None


def test_endpoint_choice_ignores_published_order(cards: dict[str, ModelCard]) -> None:
    """Order is not semantic, so it must not drive selection.

    ``claude-opus-4.6`` lists ``/v1/messages`` first; ``claude-sonnet-4.6``
    lists ``/chat/completions`` first. Same vendor, opposite order. "Pick the
    first supported endpoint" would send the former to the Anthropic wire,
    which these handlers cannot build a body for.
    """
    assert cards["claude-opus-4.6"].endpoints[0] == "/v1/messages"
    assert cards["claude-sonnet-4.6"].endpoints[0] == CHAT_COMPLETIONS_PATH
    for model in ("claude-opus-4.6", "claude-sonnet-4.6"):
        plan = plan_transport(
            inbound_path=RESPONSES_PATH,
            card=cards[model],
            heuristic_prefers_responses=True,
        )
        assert plan.upstream_path == CHAT_COMPLETIONS_PATH, model
        assert plan.request_bridge == "responses->chat", model


def test_unknown_model_falls_back_to_heuristic() -> None:
    """No catalog entry => the existing name heuristic decides, unchanged."""
    for inbound, prefers, expected in (
        (RESPONSES_PATH, True, RESPONSES_PATH),
        (CHAT_COMPLETIONS_PATH, False, CHAT_COMPLETIONS_PATH),
        (RESPONSES_PATH, False, CHAT_COMPLETIONS_PATH),
    ):
        plan = plan_transport(
            inbound_path=inbound,
            card=None,
            heuristic_prefers_responses=prefers,
        )
        assert plan.upstream_path == expected
        assert plan.catalog_backed is False


def test_chat_inbound_never_attempts_the_unimplemented_bridge() -> None:
    """Scope boundary, asserted rather than assumed.

    Only the responses->chat bridge is implemented. When a chat-inbound request
    targets a responses-only model, the planner must forward on the inbound
    wire (reproducing today's behaviour and yielding a clear upstream error)
    rather than emit a plan the request path cannot execute.
    """
    plan = plan_transport(
        inbound_path=CHAT_COMPLETIONS_PATH,
        card=None,
        heuristic_prefers_responses=True,
    )
    assert plan.upstream_path == CHAT_COMPLETIONS_PATH
    assert plan.request_bridge is None
    assert plan.executable is True
    assert "not implemented" in plan.reason


def test_model_without_published_endpoints_falls_back_to_heuristic(
    cards: dict[str, ModelCard],
) -> None:
    plan = plan_transport(
        inbound_path=CHAT_COMPLETIONS_PATH,
        card=cards["gpt-4o"],
        heuristic_prefers_responses=False,
    )
    assert plan.upstream_path == CHAT_COMPLETIONS_PATH
    assert plan.request_bridge is None


def test_native_wire_is_preferred_over_bridging(cards: dict[str, ModelCard]) -> None:
    plan = plan_transport(
        inbound_path=RESPONSES_PATH,
        card=cards["gpt-5.4"],
        heuristic_prefers_responses=True,
    )
    assert plan.upstream_path == RESPONSES_PATH
    assert plan.request_bridge is None


def test_every_plan_is_executable_across_the_whole_catalog(
    cards: dict[str, ModelCard],
) -> None:
    """Invariant: the planner never emits a bridge that does not exist.

    This is the guard that keeps catalog routing from being *worse* than the
    heuristic it replaces -- emitting an unimplementable plan would turn a
    working request into a failure.
    """
    for card in cards.values():
        if not card.is_chat_model:
            continue
        for inbound in (RESPONSES_PATH, CHAT_COMPLETIONS_PATH):
            for prefers in (True, False):
                plan = plan_transport(
                    inbound_path=inbound,
                    card=card,
                    heuristic_prefers_responses=prefers,
                )
                assert plan.executable, f"{card.id} {inbound} -> {plan.request_bridge}"
                assert plan.upstream_path in BRIDGEABLE_ENDPOINTS


def test_buffered_bridge_clamps_output_to_non_streaming_cap(
    cards: dict[str, ModelCard],
) -> None:
    """Bridged requests are sent ``stream: false``; Claude caps that at 16k."""
    plan = plan_transport(
        inbound_path=RESPONSES_PATH,
        card=cards["claude-opus-5"],
        heuristic_prefers_responses=True,
        requested_max_output_tokens=64000,
        will_buffer=True,
    )
    assert plan.request_bridge == "responses->chat"
    assert plan.clamp["max_output_tokens"] == 16000


def test_reasoning_effort_is_kept_when_the_target_supports_it(
    cards: dict[str, ModelCard],
) -> None:
    """The capability regression a blanket strip causes.

    ``claude-opus-4.8`` accepts ``reasoning_effort`` (live: 200), so dropping it
    silently downgrades exactly the high-capability subagent this feature
    exists to reach.
    """
    plan = plan_transport(
        inbound_path=RESPONSES_PATH,
        card=cards["claude-opus-4.8"],
        heuristic_prefers_responses=True,
        reasoning_effort="medium",
    )
    assert "reasoning_effort" not in plan.drop_fields
    assert "reasoning_effort" not in plan.clamp  # already supported, unchanged


def test_reasoning_effort_is_dropped_only_when_unsupported(
    cards: dict[str, ModelCard],
) -> None:
    plan = plan_transport(
        inbound_path=RESPONSES_PATH,
        card=cards["kimi-k2.7-code"],
        heuristic_prefers_responses=True,
        reasoning_effort="medium",
    )
    assert "reasoning_effort" in plan.drop_fields


# ---------------------------------------------------------------------------
# Model-id resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Claude Opus 4.6", "claude-opus-4.6"),
        ("claude opus 4.6", "claude-opus-4.6"),
        ("Gemini 3.1 Pro (Preview)", "gemini-3.1-pro-preview"),
        ("GPT-5.5", "gpt-5.5"),
    ],
)
def test_display_labels_resolve_from_live_names(
    cards: dict[str, ModelCard], label: str, expected: str
) -> None:
    assert resolve_model_id(label, cards) == expected


def test_canonical_ids_are_left_alone(cards: dict[str, ModelCard]) -> None:
    assert resolve_model_id("claude-opus-4.6", cards) is None


@pytest.mark.parametrize("label", ["Gemini", "GPT", "definitely-not-a-model", ""])
def test_ambiguous_or_unknown_labels_are_not_guessed(
    cards: dict[str, ModelCard], label: str
) -> None:
    """An unresolvable name must fail loudly upstream, not be guessed."""
    assert resolve_model_id(label, cards) is None


def test_embeddings_are_excluded_from_resolution(cards: dict[str, ModelCard]) -> None:
    assert resolve_model_id("Text Embedding 3 Small", cards) is None


# ---------------------------------------------------------------------------
# Per-credential cache isolation
# ---------------------------------------------------------------------------


def test_catalog_is_keyed_per_credential(cards: dict[str, ModelCard]) -> None:
    """Two accounts on one machine must not share a catalog.

    Copilot entitlements differ per token and per integration id, so a shared
    catalog would describe traffic it does not belong to.
    """
    catalog = CopilotModelCatalog()
    key_a = catalog.cache_key(
        base_url="https://api.githubcopilot.com",
        integration_id="vscode-chat",
        token_fingerprint="aaaa",
    )
    key_b = catalog.cache_key(
        base_url="https://api.githubcopilot.com",
        integration_id="vscode-chat",
        token_fingerprint="bbbb",
    )
    assert key_a != key_b
    catalog.put(key_a, cards)
    assert (catalog.cards(key_a) or {}).get("claude-opus-4.6") is not None
    assert (catalog.cards(key_b) or {}).get("claude-opus-4.6") is None


def test_indefinitely_stale_entries_are_dropped(cards: dict[str, ModelCard]) -> None:
    """Past the staleness bound, "unknown" beats "confidently wrong"."""
    catalog = CopilotModelCatalog()
    key = catalog.cache_key(
        base_url="https://api.githubcopilot.com",
        integration_id="vscode-chat",
        token_fingerprint="aaaa",
    )
    catalog.put(key, cards, now=0.0)
    assert (catalog.cards(key, now=100.0) or {}).get("claude-opus-4.6") is not None
    assert catalog.cards(key, now=90_000.0) is None


def test_invalidate_forces_a_refetch(cards: dict[str, ModelCard]) -> None:
    catalog = CopilotModelCatalog()
    key = catalog.cache_key(
        base_url="https://api.githubcopilot.com",
        integration_id="vscode-chat",
        token_fingerprint="aaaa",
    )
    catalog.put(key, cards)
    catalog.invalidate(key)
    assert catalog.cards(key) is None


# ---------------------------------------------------------------------------
# Launcher wire-API selection
# ---------------------------------------------------------------------------


def test_launcher_wire_api_falls_back_without_credentials() -> None:
    """No token or url => keep the name heuristic; launch must not depend on it."""
    from headroom.providers.copilot.wrap import resolve_wire_api_for_model

    assert resolve_wire_api_for_model("gpt-5.4", api_url=None, token=None) == "responses"
    assert resolve_wire_api_for_model("claude-opus-4.8", api_url=None, token=None) == "completions"
    assert resolve_wire_api_for_model(None, api_url=None, token=None) == "completions"


def test_launcher_wire_api_uses_published_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """The defect this closes, at the launcher rather than the proxy.

    ``mai-code-1-flash-picker`` is /responses-only but does not match
    ``gpt-5*``, so the name heuristic pinned the session to ``completions`` and
    every turn failed with ``400 ... not accessible via the /chat/completions
    endpoint`` -- reproduced through the real Copilot CLI before this fix.
    """
    import httpx

    from headroom.providers.copilot import wrap as copilot_wrap

    monkeypatch.setenv("HEADROOM_MODEL_CATALOG", "true")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: httpx.Response(200, json=payload), raising=True
    )

    resolve = copilot_wrap.resolve_wire_api_for_model
    # /responses-only, and the name heuristic gets it wrong.
    assert copilot_wrap.default_wire_api_for_model("mai-code-1-flash-picker") == "completions"
    assert resolve(
        "mai-code-1-flash-picker", api_url="https://api.githubcopilot.com", token="t"
    ) == ("responses")
    # chat-only models must not be pushed onto /responses.
    assert resolve("claude-opus-4.8", api_url="https://api.githubcopilot.com", token="t") == (
        "completions"
    )
    assert resolve(
        "gemini-3.1-pro-preview", api_url="https://api.githubcopilot.com", token="t"
    ) == ("completions")
    # A model with no published endpoints keeps the heuristic.
    assert resolve("gpt-4o", api_url="https://api.githubcopilot.com", token="t") == "completions"


def test_live_catalog_follows_the_rollout_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in on stable, default-on from beta, and one kill switch everywhere."""
    from headroom.models.copilot_catalog import catalog_enabled

    for name in (
        "HEADROOM_MODEL_CATALOG",
        "HEADROOM_ROLLOUT_CHANNEL",
        "HEADROOM_FEATURES",
        "HEADROOM_DISABLE_FEATURES",
    ):
        monkeypatch.delenv(name, raising=False)

    # Stable (the default channel): the heuristic, unless asked for.
    assert catalog_enabled() is False
    monkeypatch.setenv("HEADROOM_MODEL_CATALOG", "true")
    assert catalog_enabled() is True
    monkeypatch.delenv("HEADROOM_MODEL_CATALOG")
    monkeypatch.setenv("HEADROOM_FEATURES", "copilot_model_catalog")
    assert catalog_enabled() is True
    monkeypatch.delenv("HEADROOM_FEATURES")

    # Beta and later: on by default...
    monkeypatch.setenv("HEADROOM_ROLLOUT_CHANNEL", "beta")
    assert catalog_enabled() is True
    # ...and the legacy variable is still the kill switch.
    monkeypatch.setenv("HEADROOM_MODEL_CATALOG", "off")
    assert catalog_enabled() is False
    monkeypatch.delenv("HEADROOM_MODEL_CATALOG")
    monkeypatch.setenv("HEADROOM_DISABLE_FEATURES", "copilot_model_catalog")
    assert catalog_enabled() is False


def test_catalog_switch_keeps_its_old_spellings(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`enable` / `disable` meant on / off before the switch became a rollout alias."""
    from headroom.models import copilot_catalog
    from headroom.models.copilot_catalog import catalog_enabled

    # The warning fires on a cache miss only; start from a clean memo.
    copilot_catalog._catalog_policy_cache.clear()

    for name in ("HEADROOM_ROLLOUT_CHANNEL", "HEADROOM_FEATURES", "HEADROOM_DISABLE_FEATURES"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("HEADROOM_MODEL_CATALOG", "enable")
    assert catalog_enabled() is True
    monkeypatch.setenv("HEADROOM_ROLLOUT_CHANNEL", "beta")
    monkeypatch.setenv("HEADROOM_MODEL_CATALOG", "disable")
    assert catalog_enabled() is False

    # An unrecognised value is ignored (channel default), and says so.
    monkeypatch.setenv("HEADROOM_MODEL_CATALOG", "maybe")
    with caplog.at_level("WARNING", logger="headroom.models.copilot_catalog"):
        assert catalog_enabled() is True
    assert "not a recognised on/off value" in caplog.text


def test_catalog_policy_is_resolved_once_per_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """It runs on every Copilot request: no per-request resolution or warning flood."""
    import headroom.rollout as rollout_mod
    from headroom.models import copilot_catalog

    copilot_catalog._catalog_policy_cache.clear()
    monkeypatch.setenv("HEADROOM_FEATURES", "copilot_model_catalog,not_a_real_feature")
    monkeypatch.delenv("HEADROOM_MODEL_CATALOG", raising=False)
    calls: list[object] = []
    real = rollout_mod.resolve_rollout

    def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(rollout_mod, "resolve_rollout", counting)
    assert all(copilot_catalog.catalog_enabled() for _ in range(50))
    assert len(calls) == 1, "the policy was re-resolved per call"

    # A changed environment is still picked up.
    monkeypatch.setenv("HEADROOM_MODEL_CATALOG", "off")
    assert copilot_catalog.catalog_enabled() is False
    assert len(calls) == 2


def test_rollout_registry_declares_the_catalog() -> None:
    from headroom.rollout import FEATURES, RolloutChannel

    spec = FEATURES["copilot_model_catalog"]
    assert spec.available_in is RolloutChannel.STABLE
    assert spec.default_enabled_in is RolloutChannel.BETA
    assert spec.legacy_env == ("HEADROOM_MODEL_CATALOG",)


def test_launcher_wire_api_survives_an_unreachable_models_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from headroom.providers.copilot import wrap as copilot_wrap

    def _boom(*a: object, **k: object) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom, raising=True)
    assert (
        copilot_wrap.resolve_wire_api_for_model(
            "mai-code-1-flash-picker", api_url="https://api.githubcopilot.com", token="t"
        )
        == "completions"
    )


# ---------------------------------------------------------------------------
# Agent-hallucinated model names (observed live in proxy.log)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        # Version separator confusion: agents write '-' where the id has '.'.
        ("claude-sonnet-4-6", "claude-sonnet-4.6"),
        ("claude-opus-4-8", "claude-opus-4.8"),
        ("gpt-5-5", "gpt-5.5"),
    ],
)
def test_version_separator_confusion_resolves(
    cards: dict[str, ModelCard], typed: str, expected: str
) -> None:
    """`claude-sonnet-4-6` reached the proxy live; the real id is `claude-sonnet-4.6`."""
    assert resolve_model_id(typed, cards) == expected


@pytest.mark.parametrize(
    "retired",
    [
        # All of these were reached for by orchestrating agents in proxy.log and
        # none exist in the account's live catalog. They must NOT be "corrected"
        # into something plausible -- a wrong model silently answering is worse
        # than a clear upstream 400.
        "claude-3.5-sonnet",
        "gemini-2.5-pro",
        "glm-5.2",
        "gpt-5.2",
        "o1-experimental",
    ],
)
def test_retired_or_foreign_models_are_not_invented(
    cards: dict[str, ModelCard], retired: str
) -> None:
    assert resolve_model_id(retired, cards) is None


def test_static_table_no_longer_lists_retired_models() -> None:
    """The hardcoded fallback set must not legitimize models GitHub retired.

    Keeping `gemini-2.5-pro` / `gpt-5.2` / `o1-experimental` in it made a
    hallucinated name look recognized, so it was confidently forwarded and 400'd.
    """
    from headroom.proxy.handlers.openai import _KNOWN_COPILOT_MODEL_IDS

    for retired in ("gemini-2.5-pro", "gpt-5.2", "o1-experimental"):
        assert retired not in _KNOWN_COPILOT_MODEL_IDS
