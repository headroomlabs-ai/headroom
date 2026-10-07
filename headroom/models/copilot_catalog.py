"""Live model catalog for GitHub Copilot's hosted API.

Why this exists
---------------
Routing a Copilot request correctly needs four facts about the target model:
which wire API serves it, whether it accepts ``reasoning_effort`` (and which
values), how much output it can produce without streaming, and whether it is a
chat model at all. Headroom used to *guess* all four from the model name --
``model_prefers_responses_api()`` is ``startswith(("gpt-5", "o1", "o3"))`` and
the known-model set was a hand-maintained frozenset.

Those guesses are measurably wrong. ``GET /models`` on the Copilot host already
publishes every one of those facts, and it is reachable with the exact headers
Headroom already builds (``Copilot-Integration-Id: vscode-chat``). Concretely,
against a live subscription:

* ``mai-code-1-flash-picker`` is served **only** on ``/responses`` yet does not
  match the ``gpt-5*`` heuristic, so name-based routing sends it to
  ``/chat/completions`` -> ``400 unsupported_api_for_model``.
* ``reasoning_effort`` is a per-model **value set**, not a flag:
  ``claude-opus-4.6`` accepts ``max`` but rejects ``xhigh``; ``gpt-5.4`` accepts
  ``xhigh`` but rejects ``max``. Both rejections are hard 400s.
* Every ``claude-*`` model caps non-streaming output at 16k against a 64k
  streaming ceiling -- which matters because bridged requests are buffered.

Design constraints learned the hard way
---------------------------------------
**Absent ``supported_endpoints`` means "no constraint", never "nothing works."**
17 of 40 live models (the whole ``gpt-4*`` / ``gpt-3.5*`` / embeddings tail) omit
the key entirely. Treating that as an empty allow-list would break models that
work today, so those fall through to the legacy heuristic.

**Absent ``policy`` means "no gate", never "denied."** 17 models have no
``policy`` object, including ``gpt-5.3-codex`` -- a flagship coding model with
``model_picker_enabled: true``. Requiring ``policy.state == "enabled"`` would
hide it permanently.

**The catalog is per-credential, not global.** ``copilot_auth`` documents that a
client's own token and Headroom's exchanged token can receive different
entitlements for the same model, and ``/models`` itself 400s for an unknown
``Copilot-Integration-Id``. A catalog fetched on one lane does not describe
traffic on another, so entries are keyed by
``(api_base_url, integration_id, token_fingerprint)`` and never shared.

Everything here fails open: any parse or transport failure degrades to "unknown
model", and callers keep their previous name-based behaviour. The catalog can
never make a request fail that would otherwise have succeeded.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Wire endpoints this proxy's OpenAI-surface handlers can actually target.
#: ``/v1/messages`` appears in Copilot's ``supported_endpoints`` for Claude
#: models and ``ws:/responses`` for several OpenAI ones, but neither is a legal
#: target from the chat/responses HTTP handlers, so they are filtered out before
#: any routing decision. Selecting an endpoint we cannot build a body for would
#: be strictly worse than the heuristic it replaces.
BRIDGEABLE_ENDPOINTS: frozenset[str] = frozenset({"/responses", "/chat/completions"})

#: Ordered ``reasoning_effort`` ladder, weakest to strongest. Used to clamp an
#: inbound value to the nearest rung the target model actually accepts, rather
#: than dropping reasoning entirely (which silently downgrades quality) or
#: forwarding it verbatim (which 400s).
REASONING_EFFORT_LADDER: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

_DEFAULT_TTL_SECONDS = 900.0
#: An entry older than this is not served even when a refresh fails. Beyond it,
#: "stale but present" is more dangerous than "unknown": GitHub does change
#: ``supported_endpoints`` between releases, and an indefinitely-stale entry
#: would pin a model to a wire that no longer serves it, with no recovery.
_MAX_STALE_SECONDS = 86_400.0
#: Absent models are remembered briefly so a genuinely unentitled model does not
#: trigger a refresh on every request in a subagent fan-out.
_NEGATIVE_TTL_SECONDS = 60.0


def catalog_enabled() -> bool:
    """Whether live catalog-driven routing is on, per the runtime rollout policy.

    Rollout-managed as ``copilot_model_catalog`` (see ``headroom/rollout.py``):
    default-on from the ``beta`` channel, opt-in on ``stable`` via
    ``HEADROOM_MODEL_CATALOG=1`` or ``HEADROOM_FEATURES=copilot_model_catalog``,
    and ``HEADROOM_MODEL_CATALOG=0`` is the kill switch on every channel.
    Resolved per call so a changed environment takes effect, and never raises:
    an unreadable policy keeps today's name heuristic.
    """
    from headroom.rollout import resolve_rollout

    try:
        return resolve_rollout().is_enabled("copilot_model_catalog")
    except Exception:  # noqa: BLE001 - routing must never fail on policy resolution
        logger.debug("rollout policy unavailable; live model catalog stays off", exc_info=True)
        return False


def catalog_ttl_seconds() -> float:
    """Refresh interval, overridable via ``HEADROOM_MODEL_CATALOG_TTL``."""
    raw = os.environ.get("HEADROOM_MODEL_CATALOG_TTL", "").strip()
    if not raw:
        return _DEFAULT_TTL_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.debug("invalid HEADROOM_MODEL_CATALOG_TTL %r; using default", raw)
        return _DEFAULT_TTL_SECONDS
    return value if value > 0 else _DEFAULT_TTL_SECONDS


@dataclass(frozen=True, slots=True)
class ModelCard:
    """One model as the upstream describes it.

    ``endpoints`` deserves care: an **empty tuple means the upstream published
    no endpoint constraint**, not that no endpoint works. Callers must treat it
    as "fall back to your own heuristic", which is why
    :meth:`constrains_endpoints` exists rather than callers testing truthiness
    and accidentally reading it as "nothing supported".
    """

    id: str
    display_name: str = ""
    vendor: str = ""
    tier: str | None = None
    kind: str = "chat"
    endpoints: tuple[str, ...] = ()
    reasoning_efforts: tuple[str, ...] = ()
    context_window: int | None = None
    max_output_tokens: int | None = None
    max_non_streaming_output_tokens: int | None = None
    selectable: bool = False
    preview: bool = False

    @property
    def is_chat_model(self) -> bool:
        """False for embeddings and other non-conversational entries.

        ``text-embedding-3-small`` and friends are returned by ``/models`` and
        must never reach chat routing or model-name resolution.
        """
        return self.kind == "chat"

    def constrains_endpoints(self) -> bool:
        """True only when the upstream actually published an endpoint list."""
        return bool(self.endpoints)

    def bridgeable_endpoints(self) -> tuple[str, ...]:
        """Published endpoints this proxy can target, in published order."""
        return tuple(e for e in self.endpoints if e in BRIDGEABLE_ENDPOINTS)

    def supports_endpoint(self, endpoint: str) -> bool:
        """True when ``endpoint`` is published, or when nothing was published."""
        if not self.constrains_endpoints():
            return True
        return endpoint in self.endpoints

    def clamp_reasoning_effort(self, effort: str | None) -> str | None:
        """Return the closest accepted effort, or ``None`` to drop the field.

        Returns ``None`` when the model publishes no ``reasoning_effort``
        support at all (send nothing), and otherwise the nearest rung on
        :data:`REASONING_EFFORT_LADDER` that the model accepts. Clamping rather
        than dropping preserves as much of the caller's intent as the target
        model can express -- a ``max`` request against a model that tops out at
        ``high`` should still reason hard, not stop reasoning.

        **Ties resolve downward.** ``xhigh`` on a model offering
        ``[low, medium, high, max]`` is equidistant from ``high`` and ``max``;
        it clamps to ``high``. Reasoning effort maps to spend and latency, so
        when the caller's intent is genuinely ambiguous the cheaper rung is the
        defensible default -- silently *upgrading* someone into the most
        expensive tier is a worse surprise than staying a notch below.
        """
        if not effort or not self.reasoning_efforts:
            return None
        if effort in self.reasoning_efforts:
            return effort
        if effort not in REASONING_EFFORT_LADDER:
            return None
        wanted = REASONING_EFFORT_LADDER.index(effort)
        ranked = sorted(
            (e for e in self.reasoning_efforts if e in REASONING_EFFORT_LADDER),
            key=REASONING_EFFORT_LADDER.index,
        )
        if not ranked:
            return None
        # Sorted ascending + strict `<` on distance makes the lower rung win a
        # tie, independent of the order the upstream happened to publish.
        best = ranked[0]
        best_distance = abs(REASONING_EFFORT_LADDER.index(best) - wanted)
        for candidate in ranked[1:]:
            distance = abs(REASONING_EFFORT_LADDER.index(candidate) - wanted)
            if distance < best_distance:
                best, best_distance = candidate, distance
        return best


def _as_str(value: Any, *, default: str = "") -> str:
    """Coerce an untrusted JSON value to a definite ``str``.

    The upstream payload is not schema-validated, so a field can be missing, or
    an unexpected type. Narrowing here keeps every ``ModelCard`` field concrete
    rather than pushing ``Any`` through the routing decisions downstream.
    """
    return value if isinstance(value, str) else default


def _as_opt_str(value: Any) -> str | None:
    """Coerce to ``str`` or ``None`` for genuinely optional fields."""
    return value if isinstance(value, str) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def parse_model_card(entry: Any) -> ModelCard | None:
    """Build a :class:`ModelCard` from one ``/models`` entry, or ``None``.

    Never raises: an entry Headroom cannot understand is skipped so one odd
    record cannot poison the whole catalog.
    """
    if not isinstance(entry, dict):
        return None
    model_id = entry.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None

    capabilities = entry.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    supports = capabilities.get("supports")
    supports = supports if isinstance(supports, dict) else {}
    limits = capabilities.get("limits")
    limits = limits if isinstance(limits, dict) else {}

    raw_endpoints = entry.get("supported_endpoints")
    endpoints = (
        tuple(e for e in raw_endpoints if isinstance(e, str))
        if isinstance(raw_endpoints, list)
        else ()
    )

    raw_efforts = supports.get("reasoning_effort")
    if isinstance(raw_efforts, list):
        efforts = tuple(e for e in raw_efforts if isinstance(e, str))
    elif raw_efforts is True:
        # Defensive: an older/leaner payload may publish a bare flag. Treat it
        # as the conventional three rungs rather than as "no support".
        efforts = ("low", "medium", "high")
    else:
        efforts = ()

    # Absent `policy` means there is no terms-acceptance gate, so the model is
    # usable. Only an explicitly non-enabled state disqualifies it.
    policy = entry.get("policy")
    policy_ok = True
    if isinstance(policy, dict):
        state = policy.get("state")
        policy_ok = state is None or state == "enabled"

    return ModelCard(
        id=model_id,
        display_name=_as_str(entry.get("name")),
        vendor=_as_str(entry.get("vendor")),
        tier=_as_opt_str(entry.get("model_picker_category")),
        kind=_as_str(capabilities.get("type"), default="chat"),
        endpoints=endpoints,
        reasoning_efforts=efforts,
        context_window=_as_int(limits.get("max_context_window_tokens")),
        max_output_tokens=_as_int(limits.get("max_output_tokens")),
        max_non_streaming_output_tokens=_as_int(limits.get("max_non_streaming_output_tokens")),
        selectable=bool(entry.get("model_picker_enabled")) and policy_ok,
        preview=bool(entry.get("preview")),
    )


def parse_models_payload(payload: Any) -> dict[str, ModelCard]:
    """Parse a ``GET /models`` body into ``{model_id: ModelCard}``.

    Accepts both the documented ``{"data": [...]}`` envelope and a bare list.
    Returns ``{}`` for anything unrecognizable -- callers treat that as "no
    catalog" and keep their heuristic.
    """
    if isinstance(payload, dict):
        entries = payload.get("data")
    else:
        entries = payload
    if not isinstance(entries, list):
        return {}
    cards: dict[str, ModelCard] = {}
    for entry in entries:
        card = parse_model_card(entry)
        if card is not None:
            cards[card.id] = card
    return cards


@dataclass
class _CatalogEntry:
    cards: dict[str, ModelCard]
    fetched_at: float

    def age(self, now: float) -> float:
        return max(0.0, now - self.fetched_at)


class CopilotModelCatalog:
    """Per-credential cache of Copilot's published model list.

    Deliberately *not* a singleton keyed by URL alone. Two sessions on one
    machine can hold different Copilot entitlements, and serving one account's
    catalog to another would produce confidently wrong routing.
    """

    def __init__(self, *, ttl_seconds: float | None = None) -> None:
        self._ttl = ttl_seconds if ttl_seconds is not None else catalog_ttl_seconds()
        self._entries: dict[tuple[str, str, str], _CatalogEntry] = {}
        self._failed_at: dict[tuple[str, str, str], float] = {}
        self._locks: dict[tuple[str, str, str], asyncio.Lock] = {}

    @staticmethod
    def cache_key(
        *, base_url: str, integration_id: str, token_fingerprint: str
    ) -> tuple[str, str, str]:
        """Identity a catalog is valid for. All three parts are load-bearing."""
        return ((base_url or "").rstrip("/"), integration_id or "", token_fingerprint or "")

    def put(
        self, key: tuple[str, str, str], cards: dict[str, ModelCard], *, now: float | None = None
    ) -> None:
        self._entries[key] = _CatalogEntry(
            cards=dict(cards), fetched_at=now if now is not None else time.time()
        )
        # A successful fetch clears any backoff from an earlier failure.
        self._failed_at.pop(key, None)

    def is_fresh(self, key: tuple[str, str, str], *, now: float | None = None) -> bool:
        entry = self._entries.get(key)
        if entry is None:
            return False
        return entry.age(now if now is not None else time.time()) < self._ttl

    def cards(
        self, key: tuple[str, str, str], *, now: float | None = None
    ) -> dict[str, ModelCard] | None:
        """All cards for ``key``, or ``None`` when there is nothing usable.

        The single accessor the request path should use, because it is where the
        staleness bound is enforced. Reading ``_entries`` directly bypassed that
        guard, so name normalization could run off an arbitrarily old catalog.

        An entry past :data:`_MAX_STALE_SECONDS` is dropped and reported as
        absent: an indefinitely stale endpoint list is worse than no information,
        since it routes confidently to a wire that may no longer serve the model,
        whereas ``None`` restores the caller's heuristic.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.age(now if now is not None else time.time()) > _MAX_STALE_SECONDS:
            self._entries.pop(key, None)
            return None
        return dict(entry.cards) if entry.cards else None

    def note_fetch_failure(self, key: tuple[str, str, str], *, now: float | None = None) -> None:
        """Record that a refresh failed, without discarding what we already had.

        Overwriting a good catalog with an empty one on a transient ``/models``
        blip would silently revert routing to the name heuristic for a whole TTL
        -- and that heuristic is wrong in the unsafe direction for
        ``/responses``-only models. Instead the previous cards stay live (up to
        the staleness bound) and refetching is suppressed only briefly, so a real
        outage does not turn into a per-request retry storm.
        """
        self._failed_at[key] = now if now is not None else time.time()

    def refresh_suppressed(self, key: tuple[str, str, str], *, now: float | None = None) -> bool:
        """True while a recent fetch failure should stop us retrying."""
        failed = self._failed_at.get(key)
        if failed is None:
            return False
        return (now if now is not None else time.time()) - failed < _NEGATIVE_TTL_SECONDS

    def fetch_lock(self, key: tuple[str, str, str]) -> asyncio.Lock:
        """Per-credential lock so one refresh serves a concurrent wave."""
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def invalidate(self, key: tuple[str, str, str]) -> None:
        """Drop an entry so the next lookup refetches.

        Called when the upstream contradicts the catalog (``unsupported_api_for_model``
        / ``model_not_supported``), which is the only reliable signal that a
        cached endpoint list went stale before its TTL expired.
        """
        self._entries.pop(key, None)
        self._failed_at.pop(key, None)


def resolve_model_id(
    raw_model: str, cards: Mapping[str, ModelCard] | Iterable[ModelCard]
) -> str | None:
    """Map a casually-typed model label to a canonical id, or ``None``.

    Clients sometimes send a display label instead of an API id -- an agent told
    to "review with Claude Opus 4.8" may pass that literal string, which the
    upstream rejects. Correction is derived from the live catalog's own ``name``
    field rather than a hand-maintained alias table, so it tracks GitHub's
    naming automatically.

    Returns ``None`` (leave the caller's value alone) when the label is unknown
    **or ambiguous**. Ambiguity is real: "GPT-4o" matches three ids in the live
    catalog. Guessing would silently route to a model the user did not ask for,
    so an unresolvable name is left to fail loudly upstream instead.
    """
    if not raw_model or not raw_model.strip():
        return None
    # Annotated explicitly: narrowing on the unparameterised `Mapping` ABC loses
    # the value type, which would leak `Any` out of every `return card.id` below.
    card_list: list[ModelCard] = list(cards.values()) if isinstance(cards, Mapping) else list(cards)
    chat_cards = [c for c in card_list if c.is_chat_model]

    for card in chat_cards:
        if card.id == raw_model:
            return None  # already canonical; nothing to change

    def _norm(value: str) -> str:
        cleaned = value.replace("(", " ").replace(")", " ").strip().lower()
        return "-".join(cleaned.split())

    target = _norm(raw_model)
    if not target:
        return None

    by_id = [c for c in chat_cards if _norm(c.id) == target]
    if len(by_id) == 1:
        return by_id[0].id
    if len(by_id) > 1:
        return None

    by_name = [c for c in chat_cards if c.display_name and _norm(c.display_name) == target]
    if len(by_name) == 1:
        return by_name[0].id

    # Version separators get confused constantly: agents write
    # ``claude-sonnet-4-6`` for ``claude-sonnet-4.6`` (observed live in
    # proxy.log). Compare with '.' and '-' collapsed to a single token so the
    # two spellings meet, but only accept a UNIQUE match -- if flattening makes
    # a name ambiguous, leave it alone rather than pick one.
    def _flatten(value: str) -> str:
        return value.replace(".", "").replace("-", "").replace("_", "")

    flat_target = _flatten(target)
    if flat_target:
        by_flat = [c for c in chat_cards if _flatten(c.id) == flat_target]
        if len(by_flat) == 1:
            return by_flat[0].id
        if not by_flat:
            by_flat_name = [
                c
                for c in chat_cards
                if c.display_name and _flatten(_norm(c.display_name)) == flat_target
            ]
            if len(by_flat_name) == 1:
                return by_flat_name[0].id
    return None


async def fetch_cards(
    http_client: Any,
    *,
    base_url: str,
    headers: Mapping[str, str],
    timeout: float = 5.0,
) -> dict[str, ModelCard]:
    """GET ``{base_url}/models`` and parse it. Returns ``{}`` on any failure.

    Deliberately total: a refused connection, a 400 from an unknown
    ``Copilot-Integration-Id``, a timeout, or an unparseable body all yield
    ``{}``, which callers read as "no catalog" and handle by keeping their
    name-based heuristic. Discovery must never be able to fail a request that
    would otherwise have succeeded.
    """
    url = f"{(base_url or '').rstrip('/')}/models"
    try:
        response = await http_client.get(url, headers=dict(headers), timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — discovery is strictly best-effort
        logger.debug("copilot model catalog fetch failed for %s: %s", url, exc)
        return {}
    if response.status_code != 200:
        logger.debug("copilot model catalog fetch returned %s for %s", response.status_code, url)
        return {}
    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("copilot model catalog body was not JSON: %s", exc)
        return {}
    cards = parse_models_payload(payload)
    if cards:
        logger.info("copilot model catalog: loaded %d models from %s", len(cards), url)
    return cards
