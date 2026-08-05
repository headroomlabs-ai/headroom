"""Session aggregation for the anonymous telemetry beacon.

One wide event per session, emitted when the session goes idle. Everything here
is off unless ``HEADROOM_TELEMETRY`` is explicitly on — see
:mod:`headroom.telemetry.beacon`.

# What counts as a session

A contiguous burst of proxy activity from one install, closed after
``IDLE_TIMEOUT_S`` of quiet. This is the web-analytics definition (GA uses a
30-minute inactivity window) and it is deliberately *not* keyed on conversation
content.

The alternative was to key sessions on
:func:`headroom.proxy.output_savings_policy.conversation_key_from_body`, which
is stable across the turns of one agent loop. That was rejected twice over:

* It is content-derived — a SHA256 over the first 512 chars of the first user
  message. Exporting it, even blinded, puts a function of the user's prompt on
  the wire. For a proxy whose entire promise is safe prompt handling, that is
  the wrong default, and defending it in a threat model costs more than the
  grouping is worth.
* ``RequestOutcome`` is built at 30 sites across six handlers, and the parsed
  body is not available at any shared chokepoint (the HTTP middleware sees
  headers only — see ``set_current_project`` at ``proxy/server.py``). Plumbing
  a conversation key to all of them is a large diff for a metric nobody has
  asked for yet.

Burst sessions serve every metric the beacon exists for: session count,
duration, turns, tokens saved per session, and retention. What they lose is
conversation *boundaries* — two concurrent conversations merge into one burst.
``turn_id`` is already on every outcome if conversation-level grouping is ever
needed; it can be added without changing this shape.

# Wire format

OTLP/HTTP JSON logs, POSTed straight to the collector. Deliberately *not* the
OTel logs SDK: that API is still ``opentelemetry.sdk._logs`` (underscore =
unstable), and the beacon should not break on an SDK minor bump. The OTLP JSON
wire format is a stable spec and plain stdlib gets us there.

The record body is an OTLP kvlist (not a JSON string) so the collector's
``keep_keys(body, [...])`` allowlist can introspect and drop unknown fields
server-side. That is the only control point for clients already in the wild.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import platform
import re
import secrets
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Where session events go. The receiver is open source: deploy/beacon/worker.js.
#
# TEMPORARY: this is a Cloudflare workers.dev address, which is a *vendor*
# hostname. OSS users pin versions and old releases live for years, so whatever
# ships in a tagged release is effectively permanent — a vendor URL here marries
# Headroom to Cloudflare and looks alarming to anyone who runs `strings` on the
# package. Move this to otlp.headroomlabs.ai (one CNAME, or a Cloudflare zone)
# before cutting a release that contains it.
DEFAULT_ENDPOINT = "https://headroom-beacon.headroom-beacon.workers.dev/v1/logs"
SCHEMA_VERSION = 1

# A gap longer than this closes the session. Agent loops are bursty; 15 minutes
# separates "reading the diff before the next prompt" from "walked away".
IDLE_TIMEOUT_S = 900.0

# How often a live session reports in. Each report is a *cumulative snapshot*
# of the same session, not a slice of it — see _Session.payload.
FLUSH_INTERVAL_S = 300.0

_POST_TIMEOUT_S = 5.0

# Reason/enum values are bounded vocabularies in code, but they reach us as
# free strings. Validate rather than trust: anything off-pattern becomes
# "other" so a future tag value cannot smuggle text onto the wire.
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

# Tags whose values are skip/bypass reasons — the "why was compression low"
# vocabulary. Values are slug-validated before they are counted.
_REASON_TAGS = ("passthrough_reason", "image_skip_reason", "memory_skip_reason")


def _pct(numerator: float, denominator: float) -> float:
    """Percentage to 2dp, or 0.0 when undefined.

    Zero denominator returns 0.0 rather than null: the raw counts ship
    alongside every ratio, so "0.0 with original=0" is unambiguous, and it
    keeps the field a plain number for every consumer.
    """
    if not denominator:
        return 0.0
    return round(numerator / denominator * 100.0, 2)


def _safe_slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if _SLUG_RE.match(text) else "other"


_model_cache: dict[str, str | None] = {}


def _public_model(model: str) -> str | None:
    """Return the model id only if it appears in a public model registry.

    Model ids are public product SKUs, not user data — withholding them costs
    real signal (Haiku and Opus are the same ``provider`` but completely
    different compression economics). The exception is custom deployments:
    ``ft:gpt-4o:acme-corp:internal-bot:abc123`` carries an org name, and a
    self-hosted model can be called anything at all.

    litellm's cost map is exactly the "is this a public SKU" oracle, and
    Headroom already consults it for pricing in ``proxy/savings_tracker.py``.
    Unknown model, or litellm absent (it is gated to Python < 3.14), means no
    model field. Under-reporting is the correct failure direction here.
    """
    if not model:
        return None
    if model in _model_cache:
        return _model_cache[model]

    resolved: str | None = None
    try:
        import litellm

        registry = litellm.model_cost
        for candidate in (model, model.rsplit("/", 1)[-1], model.rsplit(".", 1)[-1]):
            if candidate in registry:
                resolved = candidate
                break
    except Exception:
        resolved = None

    # ponytail: unbounded dict, but it is keyed by distinct model ids seen in
    # one process — a handful. Cap it if a router ever fans out over hundreds.
    _model_cache[model] = resolved
    return resolved


# --------------------------------------------------------------------------
# install identity
# --------------------------------------------------------------------------

_identity_lock = threading.Lock()
_install_id: str | None = None


def _install_id_path() -> Path:
    from headroom.paths import config_dir

    return config_dir() / "install_id"


def install_id() -> str:
    """Random per-install id. Not a machine fingerprint — delete the file to reset.

    Deliberately not derived from hostname, MAC, or any hardware property:
    anything a user cannot change reads as tracking, and a random UUID answers
    every question the beacon actually asks.
    """
    global _install_id
    with _identity_lock:
        if _install_id is not None:
            return _install_id
        path = _install_id_path()
        try:
            existing = path.read_text().strip()
            if existing:
                _install_id = existing
                return _install_id
        except OSError:
            pass
        _install_id = uuid.uuid4().hex
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(_install_id)
        except OSError:
            logger.debug("telemetry: could not persist install_id", exc_info=True)
        return _install_id


def resource_attributes(
    *, install_mode: str | None = None, stack: str | None = None
) -> dict[str, str]:
    """Set once per process; every signal inherits these for free.

    Adding a field here applies it to every signal, retroactively — this is the
    cheap lane. ``install_mode`` and ``stack`` are passed in rather than
    detected: :func:`headroom.telemetry.context.detect_install_mode` needs the
    bound port and :func:`~headroom.telemetry.context.detect_stack` needs the
    live stats dict, both of which only the proxy has.
    """
    from headroom._version import get_version

    attrs = {
        "service.name": "headroom",
        "service.version": get_version(),
        "headroom.install_id": install_id(),
        "os.type": platform.system().lower(),
        "host.arch": platform.machine().lower(),
    }
    if install_mode:
        attrs["headroom.install_mode"] = install_mode
    if stack:
        attrs["headroom.stack"] = stack
    return attrs


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


@dataclass
class _Session:
    sid: str
    started: float
    last_seen: float
    last_emit: float = 0.0
    seq: int = 0
    turns: int = 0
    original_tokens: int = 0
    attempted_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_saved: int = 0
    tool_saved_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    uncached_tokens: int = 0
    failures: int = 0
    passthrough_turns: int = 0
    response_cache_hits: int = 0
    overhead_ms: float = 0.0
    latency_ms: float = 0.0
    transforms: dict[str, int] = field(default_factory=dict)
    skips: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)
    providers: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)

    def payload(self, reason: str) -> dict[str, Any]:
        """Cumulative snapshot of this session. Content-free.

        Every counter is a running total since the session began, NOT a delta
        since the last report. That is what makes the stream dedupable: a
        session that reports 8 times produces 8 rows sharing one ``id``, and
        the one with the highest ``seq`` is the complete picture. Deduping is
        then a window function, and no summing is involved:

            SELECT * FROM events
            QUALIFY row_number() OVER (
                PARTITION BY resource['headroom.install_id'], session.id
                ORDER BY session.seq DESC
            ) = 1

        Cumulative also makes the stream loss-tolerant: a dropped report costs
        nothing, because the next one restates everything. Deltas would leave a
        permanent hole. The redundancy is ~2KB per report, which is free.

        ``seq`` increments on every emission, so this is not a pure read.
        """
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "session": {
                "id": self.sid,
                "seq": self.seq,
                "duration_s": int(self.last_seen - self.started),
                "turns": self.turns,
                # "active" means more reports are coming for this id. Anything
                # else is the last word on this session.
                "ended": reason,
                "final": reason != "active",
            },
            # original -> attempted -> saved is the whole diagnostic chain.
            # A low `saved` means two completely different things:
            #   attempted << original  -> little was eligible (frozen cache
            #                             prefix, passthrough, system prompts)
            #   saved << attempted     -> compression ran and found nothing,
            #                             which is the real quality signal
            # Shipping only `saved` cannot tell those apart.
            "tokens": {
                "original": self.original_tokens,
                "attempted": self.attempted_tokens,
                "input": self.input_tokens,
                "output": self.output_tokens,
                "saved": self.tokens_saved,
                # Tool-schema savings (deferral + turn-hook shrink) never move
                # original/optimized — outcome.py says so explicitly. Without
                # this field a tool-heavy session reports saved=0 while having
                # genuinely saved thousands, which understates the product.
                "tool_saved": self.tool_saved_tokens,
                "cache_read": self.cache_read_tokens,
                "cache_write": self.cache_write_tokens,
                "uncached": self.uncached_tokens,
            },
            # Convenience ratios for this ONE session, 0.0 when the denominator
            # is zero. Do not average these across sessions to get a fleet
            # number — that weights a 10-token session equal to a 1M-token one.
            # Fleet rates come from summing the raw counts above.
            "rates": {
                # Of everything sent, how much did we remove?
                "saved_pct": _pct(self.tokens_saved, self.original_tokens),
                # Of everything sent, how much were we even allowed to touch?
                # A low value here means frozen cache prefix / passthrough /
                # system prompts — a ceiling, not a compressor failure.
                "eligible_pct": _pct(self.attempted_tokens, self.original_tokens),
                # Of what we touched, how much did we remove? THIS is the
                # compressor quality number, and the one Kompress moves.
                "yield_pct": _pct(self.tokens_saved, self.attempted_tokens),
                # Provider prompt cache participation. Headroom freezes prefixes
                # to protect this, so it is the other side of eligible_pct.
                "cache_read_pct": _pct(self.cache_read_tokens, self.original_tokens),
                # Headroom's share of REQUEST time: sum(overhead) / sum(latency),
                # i.e. a latency-weighted average across turns.
                #
                # NOT a fraction of wall-clock, which is what this comment used to
                # claim. Both terms are sums over turns, and turns run
                # concurrently (parallel tool calls, subagents, several clients on
                # one proxy), so each sum can exceed the session's elapsed time —
                # observed at 1.84x on a 1393-turn session. Dividing one
                # over-counted sum by another still yields a meaningful per-request
                # share, but it says nothing about how much longer the session took.
                # For that, compare `overhead_ms_per_turn` against
                # `session.duration_s / turns`.
                "overhead_pct": _pct(self.overhead_ms, self.latency_ms),
            },
            "compression": {
                "transforms": dict(self.transforms),
                "overhead_ms_total": round(self.overhead_ms, 1),
                # Sum of per-request durations, NOT elapsed time: concurrent turns
                # make this exceed `session.duration_s`. Kept under the original
                # name for schema-v1 consumers; read the per-turn values below for
                # anything comparable across sessions.
                "latency_ms_total": round(self.latency_ms, 1),
                # Unambiguous under concurrency: a mean per request, independent of
                # how many were in flight. This is the pair to reason about.
                "overhead_ms_per_turn": round(self.overhead_ms / self.turns, 1)
                if self.turns
                else 0.0,
                "latency_ms_per_turn": round(self.latency_ms / self.turns, 1)
                if self.turns
                else 0.0,
                "passthrough_turns": self.passthrough_turns,
                # Served from Headroom's own response cache — the provider was
                # never called at all. 100% saving on those turns.
                "response_cache_hits": self.response_cache_hits,
            },
            # Why compression did not fire, by bounded reason slug.
            "skips": dict(self.skips),
            # Turns by origin: "proxy" for requests through the HTTP proxy,
            # "mcp" for headroom_compress tool calls. Those have different
            # shapes — an MCP call has no provider, no upstream latency, and
            # everything handed to it is eligible — so aggregate stats must be
            # able to separate them rather than silently blending the two.
            "sources": dict(self.sources),
            "providers": sorted(self.providers),
            "models": sorted(self.models),
            "failures": self.failures,
        }
        self.seq += 1
        return snapshot


class SessionAggregator:
    """Folds per-request outcomes into one event per activity burst.

    Thread-safe. One live session per process, so there is no LRU to bound and
    nothing to evict. Reaping happens on write rather than on a background
    task — an idle process has nothing to flush anyway, and ``flush_all``
    covers shutdown.
    """

    def __init__(
        self,
        emit: Callable[[dict[str, Any]], None],
        *,
        idle_s: float = IDLE_TIMEOUT_S,
        flush_s: float = FLUSH_INTERVAL_S,
    ) -> None:
        self._emit = emit
        self._idle_s = idle_s
        self._flush_s = flush_s
        self._lock = threading.Lock()
        self._current: _Session | None = None

    def record(self, outcome: Any, *, source: str = "proxy", now: float | None = None) -> None:
        """Fold one request outcome into the live session. Never raises."""
        now = time.time() if now is None else now
        pending: list[dict[str, Any]] = []
        try:
            with self._lock:
                live = self._current
                # Quiet for longer than the idle window: that session is over.
                # We only notice on the next request, so its closing report is
                # late — but every counter in it is already correct, because
                # snapshots are cumulative.
                if live is not None and now - live.last_seen >= self._idle_s:
                    pending.append(live.payload("idle"))
                    live = None
                if live is None:
                    live = _Session(
                        sid=secrets.token_hex(8),
                        started=now,
                        last_seen=now,
                        last_emit=now,
                    )
                    self._current = live
                _fold(live, outcome, now, source)
                # Heartbeat. Same session id, running totals — a later report
                # supersedes an earlier one rather than adding to it.
                if now - live.last_emit >= self._flush_s:
                    live.last_emit = now
                    pending.append(live.payload("active"))
        except Exception:  # telemetry must never break the proxy
            logger.debug("telemetry: session record failed", exc_info=True)
            return
        # Emit outside the lock: the POST must not queue every other caller.
        for snapshot in pending:
            self._safe_emit(snapshot)

    def flush_all(
        self,
        reason: str = "shutdown",
        *,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Close and report the live session.

        ``emit`` overrides the configured sink for this call only. The shutdown
        path needs that: the normal sink hands off to a daemon thread, and
        daemon threads are killed before they finish once atexit handlers are
        running, so the final report would never leave the process.
        """
        with self._lock:
            pending, self._current = self._current, None
        if pending is None:
            return
        sink = emit or self._emit
        try:
            sink(pending.payload(reason))
        except Exception:
            logger.debug("telemetry: session emit failed", exc_info=True)

    def _safe_emit(self, payload: dict[str, Any]) -> None:
        try:
            self._emit(payload)
        except Exception:
            logger.debug("telemetry: session emit failed", exc_info=True)


def _fold(sess: _Session, outcome: Any, now: float, source: str = "proxy") -> None:
    """Duck-typed against RequestOutcome so field moves don't break the beacon.

    Duck-typing is what lets the MCP path reuse this: ``_McpCompression`` sets
    only the handful of fields it actually knows, and everything else falls
    through to a neutral default instead of needing a fake RequestOutcome.
    """

    def get(name: str, default: Any = 0) -> Any:
        return getattr(outcome, name, default)

    sess.last_seen = now
    sess.turns += 1
    sess.sources[source] = sess.sources.get(source, 0) + 1
    sess.original_tokens += int(get("original_tokens") or 0)
    sess.attempted_tokens += int(get("attempted_input_tokens") or 0)
    # Billed/volume figure, so prefer the provider's own count and fall back to
    # the local one. It sits beside output/cache_read/cache_write/uncached, which
    # are all provider-reported, so making it local would put one local number in
    # a dict of provider numbers — and `tokens.input` is what a reader sums the
    # cache buckets against.
    sess.input_tokens += int(get("provider_input_tokens") or 0) or int(get("optimized_tokens") or 0)
    sess.output_tokens += int(get("output_tokens") or 0)
    sess.tokens_saved += int(get("tokens_saved") or 0)
    sess.cache_read_tokens += int(get("cache_read_tokens") or 0)
    sess.cache_write_tokens += int(get("cache_write_tokens") or 0)
    sess.uncached_tokens += int(get("uncached_input_tokens") or 0)
    sess.overhead_ms += float(get("overhead_ms", 0.0) or 0.0)
    sess.latency_ms += float(get("total_latency_ms", 0.0) or 0.0)
    if int(get("status_code", 200) or 200) >= 500:
        sess.failures += 1
    if get("from_response_cache", False):
        sess.response_cache_hits += 1

    provider = get("provider", "") or ""
    if provider:
        sess.providers.add(str(provider)[:32])

    public = _public_model(str(get("model", "") or ""))
    if public:
        sess.models.add(public)

    # Skip/bypass reasons. These are the answer to "why was compression low" —
    # a session where every turn is passthrough:bypass_header is a config
    # problem, not a compression problem, and the two are indistinguishable
    # from the token counts alone.
    tags = get("tags", None) or {}
    if isinstance(tags, dict):
        # Tool-schema savings live only in tags — see the same arithmetic in
        # emit_request_outcome, which feeds the local dashboard.
        for tag in ("tool_search_deferred_tokens", "turn_hook_tools_saved_tokens"):
            try:
                sess.tool_saved_tokens += int(tags.get(tag, 0) or 0)
            except (TypeError, ValueError):
                pass
        for tag in _REASON_TAGS:
            if tag in tags:
                key = f"{tag.removesuffix('_reason')}:{_safe_slug(tags[tag])}"
                sess.skips[key] = sess.skips.get(key, 0) + 1
        if "passthrough_reason" in tags:
            sess.passthrough_turns += 1

    for name in get("transforms_applied", ()) or ():
        # Keep only the part before the first colon. Some transforms encode
        # per-request detail in a suffix — output_shaper ships
        # "output_shaper:stratum:<model_family>|<turn_kind>|<input_bucket>"
        # (see output_savings_policy.stratum_label), which would put the model
        # tier and a request-size bucket on the wire as a side effect of a
        # label format that has nothing to do with telemetry. The beacon only
        # wants "which transforms ran, how often"; the A/B detail belongs to
        # the local recorder that owns it.
        key = str(name).split(":", 1)[0][:64]
        sess.transforms[key] = sess.transforms.get(key, 0) + 1


# --------------------------------------------------------------------------
# OTLP/HTTP JSON transport
# --------------------------------------------------------------------------


def _any_value(value: Any) -> dict[str, Any]:
    """Python -> OTLP AnyValue. bool is checked before int: bool subclasses int."""
    # OTLP has no null. Without this, None falls through to str() and ships the
    # literal text "None" as a value.
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}  # int64 is a string in OTLP JSON
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, dict):
        return {
            "kvlistValue": {
                "values": [{"key": str(k), "value": _any_value(v)} for k, v in value.items()]
            }
        }
    if isinstance(value, (list, tuple, set)):
        return {"arrayValue": {"values": [_any_value(v) for v in value]}}
    return {"stringValue": str(value)}


def build_otlp_logs(payload: dict[str, Any], resource: dict[str, str]) -> dict[str, Any]:
    """Wrap one payload in an OTLP/HTTP JSON ExportLogsServiceRequest."""
    return {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {"key": k, "value": {"stringValue": v}} for k, v in resource.items()
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "headroom.telemetry.session"},
                        "logRecords": [
                            {
                                "timeUnixNano": str(int(time.time() * 1_000_000_000)),
                                "body": _any_value(payload),
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _post_blocking(payload: dict[str, Any], timeout: float = _POST_TIMEOUT_S) -> None:
    endpoint = os.environ.get("HEADROOM_TELEMETRY_ENDPOINT", DEFAULT_ENDPOINT)
    try:
        from headroom._version import get_version

        data = json.dumps(
            build_otlp_logs(payload, resource_attributes()), separators=(",", ":")
        ).encode()
        request = urllib.request.Request(
            endpoint,
            data=data,
            method="POST",
            headers={
                "content-type": "application/json",
                # Required, not cosmetic. urllib defaults to "Python-urllib/3.x",
                # which Cloudflare blocks outright by browser signature (error
                # 1010) before the request ever reaches the Worker. Combined
                # with fire-and-forget error handling that failure mode is
                # invisible: every upload 403s and the beacon looks healthy.
                "user-agent": f"headroom-beacon/{get_version()}",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout):
            pass
    except Exception:
        logger.debug("telemetry: session POST failed", exc_info=True)


def post_session_event(payload: dict[str, Any]) -> None:
    """Ship one session event on a daemon thread. Never blocks, never raises.

    The caller is the proxy's outcome funnel, which is async — a synchronous
    urllib POST there would stall the event loop for up to ``_POST_TIMEOUT_S``.
    Sessions close at most once per ``IDLE_TIMEOUT_S``, so a thread per event
    is a handful per hour; a pool would be machinery for nothing.
    ponytail: unbounded thread spawn is safe only because the close rate is
    bounded by the idle timeout. Revisit if anything else starts emitting here.
    """
    try:
        threading.Thread(
            target=_post_blocking,
            args=(payload,),
            name="headroom-beacon",
            daemon=True,
        ).start()
    except Exception:
        logger.debug("telemetry: could not start beacon thread", exc_info=True)


_aggregator: SessionAggregator | None = None
_aggregator_lock = threading.Lock()


def get_session_aggregator() -> SessionAggregator:
    global _aggregator
    with _aggregator_lock:
        if _aggregator is None:
            _aggregator = SessionAggregator(post_session_event)
            atexit.register(_flush_at_exit)
        return _aggregator


# A shutdown POST must not hang the process. Shorter than the normal timeout:
# a user quitting their agent should not wait on our collector.
_EXIT_POST_TIMEOUT_S = 2.0


def _flush_at_exit() -> None:
    """Report the open session on graceful exit, synchronously.

    Synchronous on purpose. ``post_session_event`` normally hands off to a
    daemon thread so the request path never blocks, but by the time atexit
    handlers run the interpreter is shutting down and daemon threads are killed
    before they can finish — a thread started here simply never posts.

    This is load-bearing well beyond the ``final`` marker: sessions shorter than
    ``FLUSH_INTERVAL_S`` have not heartbeated even once, so without a working
    exit flush every short session would report nothing at all.

    Still does not cover SIGKILL or a closed laptop. Heartbeats bound the loss
    there to one flush interval.
    """
    aggregator = _aggregator
    if aggregator is None:
        return
    aggregator.flush_all(emit=lambda payload: _post_blocking(payload, timeout=_EXIT_POST_TIMEOUT_S))


@dataclass(frozen=True)
class _McpCompression:
    """Minimal outcome shim for a ``headroom_compress`` MCP tool call.

    Only the fields an MCP compression actually knows. Everything else _fold
    reads falls through to a neutral default — there is no provider, no
    upstream latency, and no cache participation, because the tool never talks
    to a model.

    ``attempted_input_tokens == original_tokens`` on purpose: the caller hands
    the tool exactly the content it wants compressed, so all of it is eligible.
    That makes ``eligible_pct`` 100% for MCP turns, which is correct rather
    than flattering — and it is why ``sources`` has to stay in the payload, so
    MCP turns can be excluded when reading the proxy's eligibility ceiling.
    """

    original_tokens: int
    attempted_input_tokens: int
    optimized_tokens: int
    tokens_saved: int
    model: str = ""


def record_mcp_compression(
    *, original_tokens: int, compressed_tokens: int, model: str | None = None
) -> None:
    """Beacon entry point for the MCP tool path.

    Separate from :func:`record_outcome` because MCP servers are separate,
    often short-lived processes — the main one plus one per subagent, per
    ``headroom.savings_ledger``. Each gets its own aggregator and reports its
    own session, which is accurate: that really is separate work. They share
    ``install_id``, so the sessions can be grouped per install downstream.

    Short-lived processes depend entirely on the atexit flush, since they may
    never live long enough to heartbeat. See :func:`_flush_at_exit`.
    """
    from headroom.telemetry.beacon import is_beacon_enabled

    if not is_beacon_enabled():
        return
    try:
        before = int(original_tokens or 0)
        after = int(compressed_tokens or 0)
    except (TypeError, ValueError):
        return
    if before <= 0:
        return
    get_session_aggregator().record(
        _McpCompression(
            original_tokens=before,
            attempted_input_tokens=before,
            optimized_tokens=after,
            tokens_saved=max(before - after, 0),
            model=str(model or ""),
        ),
        source="mcp",
    )


def record_outcome(outcome: Any) -> None:
    """Beacon entry point, called from the proxy's outcome funnel.

    Off by default and cheap when off: the enabled check short-circuits before
    the aggregator is ever constructed, so a user who never opted in pays one
    env lookup per request and allocates nothing.
    """
    from headroom.telemetry.beacon import is_beacon_enabled

    if not is_beacon_enabled():
        return
    get_session_aggregator().record(outcome)


def demo() -> None:
    """Self-check: python -m headroom.telemetry.session"""

    class FakeOutcome:
        provider = "anthropic"
        model = "claude-3-5-sonnet-20241022"
        optimized_tokens = 100
        output_tokens = 20
        tokens_saved = 50
        cache_read_tokens = 10
        overhead_ms = 1.5
        status_code = 200
        # Widened so subclasses below can override with a different arity —
        # RequestOutcome declares tuple[str, ...] too.
        transforms_applied: tuple[str, ...] = ("crush", "dedupe")

    # bool must not be encoded as int (bool subclasses int).
    assert _any_value(True) == {"boolValue": True}
    assert _any_value(1) == {"intValue": "1"}
    assert _any_value(2.5) == {"doubleValue": 2.5}
    assert _any_value({"a": [1, 2]})["kvlistValue"]["values"][0]["key"] == "a"
    assert _any_value(None) == {}, "None must not serialise as the text 'None'"

    # Ratios: the three the product is judged on, plus a zero-denominator guard.
    assert _pct(25, 100) == 25.0
    assert _pct(1, 3) == 33.33
    assert _pct(5, 0) == 0.0

    class Rich(FakeOutcome):
        original_tokens = 1000
        attempted_input_tokens = 400  # 40% eligible
        tokens_saved = 300  # 30% overall, 75% yield on eligible
        cache_read_tokens = 500  # 50% cache read
        cache_write_tokens = 100
        uncached_input_tokens = 400
        total_latency_ms = 1000.0
        overhead_ms = 50.0  # 5% overhead
        from_response_cache = True
        tags = {"tool_search_deferred_tokens": "800", "turn_hook_tools_saved_tokens": 200}

    rates_out: list[dict[str, Any]] = []
    ra = SessionAggregator(rates_out.append)
    ra.record(Rich(), now=100.0)
    ra.flush_all()
    r = rates_out[0]
    assert r["rates"]["saved_pct"] == 30.0, r["rates"]
    assert r["rates"]["eligible_pct"] == 40.0, r["rates"]
    assert r["rates"]["yield_pct"] == 75.0, r["rates"]
    assert r["rates"]["cache_read_pct"] == 50.0, r["rates"]
    assert r["rates"]["overhead_pct"] == 5.0, r["rates"]
    # Tool savings are invisible in `saved` by design; they must not be lost.
    assert r["tokens"]["tool_saved"] == 1000, r["tokens"]
    assert r["compression"]["response_cache_hits"] == 1
    assert r["tokens"]["cache_write"] == 100 and r["tokens"]["uncached"] == 400

    emitted: list[dict[str, Any]] = []
    agg = SessionAggregator(emitted.append, idle_s=10.0)

    agg.record(FakeOutcome(), now=1000.0)
    agg.record(FakeOutcome(), now=1002.0)
    assert emitted == [], "a live session must not emit"

    # Quiet past the timeout, then activity -> the old burst closes.
    agg.record(FakeOutcome(), now=1100.0)
    assert len(emitted) == 1, emitted
    event = emitted[0]
    assert event["session"]["turns"] == 2
    assert event["session"]["ended"] == "idle"
    assert event["session"]["duration_s"] == 2
    assert event["tokens"]["saved"] == 100
    assert event["tokens"]["input"] == 200
    assert event["compression"]["transforms"] == {"crush": 2, "dedupe": 2}
    assert event["providers"] == ["anthropic"]
    assert event["failures"] == 0

    # The new burst is a distinct session, not a continuation.
    agg.flush_all()
    assert len(emitted) == 2, emitted
    assert emitted[1]["session"]["turns"] == 1
    assert emitted[1]["session"]["id"] != emitted[0]["session"]["id"]
    assert emitted[1]["session"]["ended"] == "shutdown"

    # Nothing derived from prompt content or the model id reaches the wire.
    wire = json.dumps(emitted)
    assert "sonnet" not in wire and "claude" not in wire, wire

    # Model ids reach the wire only when a public registry knows them. A
    # fine-tune id carries an org name and must never survive.
    assert _public_model("ft:gpt-4o:acme-corp:internal-bot:abc123") is None
    assert _public_model("acme-internal-llama") is None
    assert _public_model("") is None
    if _public_model("gpt-4o") is None:
        print("  (litellm unavailable — model field degrades to absent)")
    else:
        assert _public_model("gpt-4o") == "gpt-4o"

    # Reason slugs are validated, not trusted.
    assert _safe_slug("bypass_header") == "bypass_header"
    assert _safe_slug("/Users/me/secret/path.py") == "other"
    assert _safe_slug(None) == "other"

    # Low compression must be explainable: a passthrough session and a
    # ran-but-found-nothing session must not look the same.
    class Bypassed(FakeOutcome):
        original_tokens = 5000
        attempted_input_tokens = 0
        tokens_saved = 0
        transforms_applied = ()
        tags = {"passthrough_reason": "bypass_header"}

    class Barren(FakeOutcome):
        original_tokens = 5000
        attempted_input_tokens = 4000
        tokens_saved = 12
        tags: dict[str, str] = {}

    diag: list[dict[str, Any]] = []
    agg_bypassed = SessionAggregator(diag.append)
    agg_bypassed.record(Bypassed(), now=5000.0)
    agg_bypassed.flush_all()
    agg_barren = SessionAggregator(diag.append)
    agg_barren.record(Barren(), now=6000.0)
    agg_barren.flush_all()

    bypassed, barren = diag[0], diag[1]
    assert bypassed["tokens"]["attempted"] == 0
    assert bypassed["skips"] == {"passthrough:bypass_header": 1}
    assert bypassed["compression"]["passthrough_turns"] == 1
    assert barren["tokens"]["attempted"] == 4000
    assert barren["skips"] == {}
    # Both saved ~nothing, but the reason is now distinguishable.
    assert bypassed["tokens"]["saved"] == 0 and barren["tokens"]["saved"] == 12

    # A live session heartbeats under ONE id with cumulative totals, so the
    # highest-seq row is the whole session and dedupe is last-write-wins.
    beats: list[dict[str, Any]] = []
    hb = SessionAggregator(beats.append, idle_s=900.0, flush_s=100.0)
    hb.record(FakeOutcome(), now=7000.0)
    hb.record(FakeOutcome(), now=7050.0)
    assert beats == [], "must not emit before the flush interval"

    hb.record(FakeOutcome(), now=7100.0)  # first heartbeat
    hb.record(FakeOutcome(), now=7150.0)
    hb.record(FakeOutcome(), now=7250.0)  # second heartbeat
    hb.flush_all()  # final

    assert len(beats) == 3, beats
    ids = {b["session"]["id"] for b in beats}
    assert len(ids) == 1, f"one session must not fragment into {len(ids)} ids"
    assert [b["session"]["seq"] for b in beats] == [0, 1, 2]
    assert [b["session"]["final"] for b in beats] == [False, False, True]
    assert [b["session"]["ended"] for b in beats] == ["active", "active", "shutdown"]

    # Cumulative, not deltas: turns and tokens only ever climb, and the last
    # row alone reconstructs the session.
    assert [b["session"]["turns"] for b in beats] == [3, 5, 5]
    assert [b["tokens"]["saved"] for b in beats] == [150, 250, 250]

    # Dedupe by (install, session id), keep max seq -> exactly one row.
    latest: dict[str, dict[str, Any]] = {}
    for b in beats:
        key = b["session"]["id"]
        if key not in latest or b["session"]["seq"] > latest[key]["session"]["seq"]:
            latest[key] = b
    assert len(latest) == 1
    only = next(iter(latest.values()))
    assert only["session"]["turns"] == 5 and only["tokens"]["saved"] == 250

    # Losing a heartbeat costs nothing — the survivor still restates everything.
    survivors = [beats[0], beats[2]]
    assert max(s["session"]["seq"] for s in survivors) == 2
    assert survivors[-1]["session"]["turns"] == 5

    # Going quiet starts a genuinely new session, not a continuation.
    hb.record(FakeOutcome(), now=90000.0)
    hb.flush_all()
    assert beats[-1]["session"]["id"] != beats[0]["session"]["id"]
    assert beats[-1]["session"]["turns"] == 1

    # A transform label carrying a stratum suffix must be reduced to its prefix
    # — otherwise output_shaper smuggles the model tier and a size bucket out.
    class Shaped(FakeOutcome):
        transforms_applied = ("output_shaper:stratum:sonnet|tool_result|8k", "crush")

    shaped: list[dict[str, Any]] = []
    agg_s = SessionAggregator(shaped.append)
    agg_s.record(Shaped(), now=1500.0)
    agg_s.flush_all()
    assert shaped[0]["compression"]["transforms"] == {"output_shaper": 1, "crush": 1}
    assert "sonnet" not in json.dumps(shaped[0]), shaped[0]

    # A 5xx counts as a failure without poisoning the token stats.
    class Failed(FakeOutcome):
        status_code = 529

    agg2 = SessionAggregator(emitted.append)
    agg2.record(Failed(), now=2000.0)
    agg2.flush_all()
    assert emitted[-1]["failures"] == 1

    # Flushing an empty aggregator is a no-op, not a null event.
    before = len(emitted)
    SessionAggregator(emitted.append).flush_all()
    assert len(emitted) == before

    # MCP turns must be distinguishable from proxy turns in the same session.
    # Blending them would corrupt eligible_pct: everything handed to the tool is
    # eligible by construction, so MCP turns always read 100% and would drag the
    # proxy's real eligibility ceiling upward.
    mixed: list[dict[str, Any]] = []
    mx = SessionAggregator(mixed.append)
    mx.record(FakeOutcome(), now=9000.0)
    mx.record(
        _McpCompression(
            original_tokens=1000,
            attempted_input_tokens=1000,
            optimized_tokens=300,
            tokens_saved=700,
        ),
        source="mcp",
        now=9001.0,
    )
    mx.flush_all()
    mixed_ev = mixed[0]
    assert mixed_ev["sources"] == {"proxy": 1, "mcp": 1}, mixed_ev["sources"]
    assert mixed_ev["session"]["turns"] == 2
    # An MCP-only shim contributes tokens but no provider and no latency.
    assert mixed_ev["tokens"]["saved"] == 700 + 50
    assert mixed_ev["providers"] == ["anthropic"]  # only the proxy turn had one

    mcp_only: list[dict[str, Any]] = []
    mo = SessionAggregator(mcp_only.append)
    mo.record(
        _McpCompression(
            original_tokens=800,
            attempted_input_tokens=800,
            optimized_tokens=200,
            tokens_saved=600,
        ),
        source="mcp",
        now=9100.0,
    )
    mo.flush_all()
    only_ev = mcp_only[0]
    assert only_ev["sources"] == {"mcp": 1}
    assert only_ev["rates"]["eligible_pct"] == 100.0
    assert only_ev["rates"]["yield_pct"] == 75.0
    assert only_ev["rates"]["saved_pct"] == 75.0
    assert only_ev["providers"] == [] and only_ev["models"] == []
    # No upstream call means no latency, and the ratio must not divide by zero.
    assert only_ev["rates"]["overhead_pct"] == 0.0

    # record_mcp_compression: a real call records, a degenerate one does not.
    # Beacon forced on for this block so the assertions cannot pass merely
    # because the ambient environment has it disabled.
    _prev_agg = _aggregator
    _prev_env = {
        k: os.environ.get(k) for k in ("HEADROOM_BEACON", "DO_NOT_TRACK", "HEADROOM_OFFLINE")
    }
    os.environ["HEADROOM_BEACON"] = "on"
    # DO_NOT_TRACK and offline mode outrank an explicit opt-in, so they have to
    # be cleared here or these assertions test the wrong thing.
    os.environ.pop("DO_NOT_TRACK", None)
    os.environ.pop("HEADROOM_OFFLINE", None)
    try:
        globals()["_aggregator"] = SessionAggregator(lambda _p: None)
        record_mcp_compression(original_tokens=0, compressed_tokens=0)
        assert globals()["_aggregator"]._current is None, "zero-token call recorded"

        record_mcp_compression(original_tokens=500, compressed_tokens=100)
        live = globals()["_aggregator"]._current
        assert live is not None, "valid MCP call did not record"
        assert live.sources == {"mcp": 1}
        assert live.tokens_saved == 400

        # A compression that grew the content must not report negative savings.
        record_mcp_compression(original_tokens=100, compressed_tokens=250)
        assert globals()["_aggregator"]._current.tokens_saved == 400

        # DO_NOT_TRACK outranks an explicit HEADROOM_BEACON=on.
        os.environ["DO_NOT_TRACK"] = "1"
        globals()["_aggregator"] = SessionAggregator(lambda _p: None)
        record_mcp_compression(original_tokens=500, compressed_tokens=100)
        assert globals()["_aggregator"]._current is None, "DO_NOT_TRACK was ignored"
        os.environ.pop("DO_NOT_TRACK", None)
    finally:
        globals()["_aggregator"] = _prev_agg
        for _k, _v in _prev_env.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v

    # flush_all must honour an emit override. The atexit path depends on this:
    # the normal sink defers to a daemon thread, and daemon threads are killed
    # before they finish once the interpreter is shutting down, so a thread
    # started there never posts. Without the override, every session shorter
    # than FLUSH_INTERVAL_S would report nothing at all.
    default_sink: list[dict[str, Any]] = []
    override_sink: list[dict[str, Any]] = []
    ex = SessionAggregator(default_sink.append)
    ex.record(FakeOutcome(), now=8000.0)
    ex.flush_all(emit=override_sink.append)
    assert override_sink and not default_sink, "flush_all ignored the emit override"
    assert override_sink[0]["session"]["final"] is True

    # An emit that blows up must not propagate to the caller.
    def boom(_payload: dict[str, Any]) -> None:
        raise RuntimeError("collector down")

    agg3 = SessionAggregator(boom, idle_s=1.0)
    agg3.record(FakeOutcome(), now=3000.0)
    agg3.record(FakeOutcome(), now=3100.0)
    agg3.flush_all()

    # A malformed outcome must not raise either.
    SessionAggregator(emitted.append).record(object(), now=4000.0)

    print("ok")


if __name__ == "__main__":
    demo()
