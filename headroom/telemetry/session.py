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

Bodies can be gzipped (``Content-Encoding: gzip``, as OTLP/HTTP specifies). The
kvlist encoding costs about 3.8x over plain JSON -- every integer becomes
``{"intValue":"1200"}`` -- and the result is repetitive enough to compress ~6x,
which is worth several times more than every available schema trim put
together. It also restores the premise the cumulative-snapshot design rests on:
``payload`` justifies re-sending everything on each heartbeat because "the
redundancy is ~2KB per report, which is free", and at schema v2 that report is
~8KB uncompressed and ~1.3KB gzipped. Compression can turn itself off -- see
``_post_blocking`` -- so an endpoint that cannot read it costs a retry, not the
data.

It is nonetheless OFF in this release: v2 and the transport are separate
mechanisms and ship in separate releases, so that a failure in the first is
never ambiguous. See ``_GZIP_DEFAULT``. Note that it saves upload bandwidth
only -- the receiver stores plain NDJSON either way, so corpus size and read
times are identical with it on or off.

# Schema versions

``schema_version: 1`` reported volume: how many tokens moved, which strategies
ran, and how much was saved. It answered "how much did we save?" and nothing
else, because every counter in it is a session-wide SUM and none of them
describes the input.

``schema_version: 2`` is strictly additive — every v1 field keeps its name, its
type and its arithmetic, and the folding for the new signals is isolated in
:func:`_fold_extra` so a bug there cannot reach one. It adds:

* ``quality`` — the regret label. ``reread_compressed_tokens`` is tokens the
  agent had to fetch again *because compression removed them*, which is the
  only thing that makes ``saved_pct`` falsifiable: without it, deleting the
  entire context scores 100%.
* ``hist`` — fixed-bucket distributions for quantities previously reported only
  as totals, since a sum cannot distinguish uniformly-mediocre from bimodal.
  Each ships its own bucket edges, so a row is readable without knowing which
  client version wrote it.
* ``shapes`` — ``(content_type -> strategy -> yield)``, the input term the
  corpus never had. ``by_strategy`` can rank compressors against each other but
  cannot say which one suits a given piece of content, which is the decision
  the router actually makes.
* ``cache`` — prompt-cache hit rate indexed by the gap since the previous turn,
  i.e. an empirical survival curve for a TTL no provider documents accurately.
* ``trajectory`` — the session's shape over time rather than its totals.
* ``clients`` / ``errors`` / ``strata`` / ``config`` — segmentation. 4xx is
  counted in ``errors`` and deliberately NOT folded into ``failures``, which
  has been 5xx-only in every row of the corpus and must stay that way.

Everything added is a counter, a ratio, or a slug from a vocabulary that is
closed in code. Two things are still refused on principle: anything derived
from prompt or file content (which is why ``turn_id`` and TOIN's
``structure_hash`` are absent — both are digests of user data), and the source
IP, which the receiver never reads. ``headroom telemetry --show`` prints the
exact payload, so none of this has to be taken on trust.
"""

from __future__ import annotations

import atexit
import gzip
import json
import logging
import os
import platform
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterable
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
SCHEMA_VERSION = 2

# A gap longer than this closes the session. Agent loops are bursty; 15 minutes
# separates "reading the diff before the next prompt" from "walked away".
IDLE_TIMEOUT_S = 900.0

# How often a live session reports in. Each report is a *cumulative snapshot*
# of the same session, not a slice of it — see _Session.payload.
FLUSH_INTERVAL_S = 300.0

_POST_TIMEOUT_S = 5.0

# Level 6 is zlib's default and the knee of the curve here: a real payload is
# ~8KB of repetitive JSON and compresses ~6x, and level 9 buys under 2% more for
# noticeably more CPU. This runs on a daemon thread, never the request path.
_GZIP_LEVEL = 6

# The only statuses that mean "I could not read this body". Narrow on purpose.
#
# 413 in particular must NOT be here. It means the body was too big, and the
# fallback answers a refusal by re-sending the SAME payload uncompressed, which
# is ~6x larger -- so a 413 would trigger a retry guaranteed to 413 again, and
# permanently disable the compression that was the only thing keeping the
# request under the cap. The rest of the 4xx range (401, 403, 404, 429) is
# equally not about encoding, and an uncompressed retry does not help there
# either.
_GZIP_REFUSED_STATUSES = frozenset((400, 415))

# Compression ships OFF and is opted into with HEADROOM_BEACON_GZIP=1.
#
# Staged deliberately. Schema v2 and the gzip transport are two independent new
# mechanisms, and shipping both in one release means any upload failure needs
# disambiguating before it can be fixed. With this default the first v2 release
# sends plain bodies through the byte-identical path v1 already uses, so a
# failure there is unambiguously the payload.
#
# It also keeps `inflate()` in the Worker dead: the receiver sniffs gzip magic
# bytes, so if nothing compresses, nothing reaches the one code path with no
# fallback. Flip this to True in a later release once v2 itself is proven.
#
# Note what this default is NOT: a rollback. A released client reads the
# operator's environment, not ours, so this cannot be changed for installs in
# the wild. The retroactive switch is server-side -- have the Worker answer a
# gzipped body with 415 and every client falls back to plain and stays there
# for the life of the process. Same reason the privacy allowlist lives there.
_GZIP_DEFAULT = False

# Set once the far end proves it cannot take a compressed body -- see
# `_post_blocking`. Process-wide rather than per-endpoint: one endpoint per
# process.
_gzip_lock = threading.Lock()
_gzip_supported = True


class _CompressionRejected(Exception):
    """The endpoint 4xx'd a gzipped body. Retry it uncompressed."""


def _gzip_enabled() -> bool:
    """Whether to compress the next upload.

    Off unless the operator opts in with HEADROOM_BEACON_GZIP=1, and off
    regardless once this process has watched the endpoint reject a compressed
    body. Both an explicit off-value and an unrecognised value mean off, so a
    typo degrades to the safe, already-proven transport rather than to gzip.

    Cannot raise. It is called from `_post_blocking` before that function's
    try block, and `_post_blocking` runs as a bare daemon-thread target and as
    an atexit handler -- both places where an escaping exception is printed to
    the user's terminal. Telemetry is never allowed to do that, so a failure
    here degrades to "do not compress" instead.
    """
    try:
        from headroom.telemetry.beacon import _OFF_VALUES, _ON_VALUES

        raw = os.environ.get("HEADROOM_BEACON_GZIP", "").lower().strip()
        if raw in _ON_VALUES:
            enabled = True
        elif raw in _OFF_VALUES:
            enabled = False
        else:
            enabled = _GZIP_DEFAULT
        if not enabled:
            return False
        with _gzip_lock:
            return _gzip_supported
    except Exception:
        logger.debug("telemetry: gzip gate failed; sending uncompressed", exc_info=True)
        return False


def _disable_gzip() -> None:
    global _gzip_supported
    with _gzip_lock:
        _gzip_supported = False


# Reason/enum values are bounded vocabularies in code, but they reach us as
# free strings. Validate rather than trust: anything off-pattern becomes
# "other" so a future tag value cannot smuggle text onto the wire.
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

# Tags whose values are skip/bypass reasons — the "why was compression low"
# vocabulary. Values are slug-validated before they are counted.
_REASON_TAGS = ("passthrough_reason", "image_skip_reason", "memory_skip_reason")

# Cardinality cap on `by_strategy`. The real vocabulary is CompressionStrategy
# plus a couple of literals — under a dozen — but `record_compression` takes a
# free string, so an extension or a future caller could invent keys per request.
# Matches the same guard on `requests_by_stack` (MAX_DISTINCT_STACKS).
MAX_STRATEGIES = 32

# ---------------------------------------------------------------- schema v2 --
#
# Everything below is additive. Not one v1 counter changes meaning, and the
# folding for all of it lives in `_fold_extra` rather than `_fold`, so a bug in
# a new signal cannot corrupt a number the corpus already depends on — the
# worst case is a missing v2 key. See the note on `_fold_extra`.

# Fixed histogram edges, shipped alongside the counts so a reader never has to
# know which client version wrote a row. `counts` is always len(edges) + 1: one
# bucket per interval plus the open-ended tail.
#
# Sums answer "how much"; only a distribution answers "how consistently". A
# session at saved_pct 12 is either uniformly mediocre or half-brilliant and
# half-useless, and those need opposite fixes.
_HIST_EDGES: dict[str, tuple[float, ...]] = {
    # Per-turn request size, pre-compression.
    "turn_tokens": (1_000, 4_000, 16_000, 64_000, 128_000, 256_000),
    # Per-turn saved/original, in percent. Only defined when original > 0.
    "saved_pct": (0.5, 5.0, 10.0, 20.0, 40.0, 60.0),
    # Per-turn compression dispatch time.
    "overhead_ms": (1.0, 5.0, 25.0, 100.0, 500.0, 2_000.0),
    # Time to first upstream byte. 0 means unmeasured, not fast, so zeros are
    # excluded rather than piled into the first bucket.
    "ttfb_ms": (200.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 15_000.0),
    # Messages in the request. Grows monotonically within an agent loop, so the
    # shape of this is the shape of the conversation.
    "msgs": (5, 20, 50, 100, 250, 500),
}

# Gap since the previous turn, in seconds, for the cache-survival curve. The
# provider's real prompt-cache TTL is an empirical question — nobody documents
# it accurately and it drifts — and (gap -> hit rate) across a fleet is the
# measurement. Edges bracket the two published Anthropic TTLs (5m, 1h).
_CACHE_GAP_EDGES: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0, 600.0, 1_800.0)

# Turn-index buckets for the trajectory curve: turn 1, turns 2-3, 4-7, ... 512+.
# Log-spaced rather than deciles because a session's turn count is not known
# until it ends, and a cumulative snapshot cannot re-bin what it already wrote.
# Ten buckets covers a 1000-turn session, which is past the observed maximum.
_TRAJECTORY_BUCKETS = 10

# Turn kinds, in priority order — the first that matches wins. A closed
# alphabet: `_turn_kind` can only ever return one of these, so no free text
# reaches the run-length string.
#
#   c  served from Headroom's own response cache (provider never called)
#   f  upstream 5xx
#   e  upstream 4xx (429 lives here, and is the interesting one)
#   p  passthrough — compression was bypassed
#   z  compression ran and removed nothing
#   s  compression ran and removed something
#   o  none of the above (nothing was eligible, no bypass reason given)
_TURN_KINDS = frozenset("cfepzso")

# Cap on the run-length trajectory string. A 1400-turn session that alternates
# kinds every turn would otherwise write ~2.8KB of runs on its own.
MAX_TRAJECTORY_RUNS = 64

# Cardinality cap for the v2 maps that take a slug from a bounded vocabulary
# but are still fed by a free string somewhere upstream. Same reasoning and the
# same number as MAX_STRATEGIES.
MAX_SHAPES = 64

# Row cap for the two shape TABLES, which is a different question: their
# vocabulary is a cross product, not a list. ContentType has 9 members and
# CompressionStrategy 12, plus the "other" each collapses to, so
# `shapes.by_content` can legitimately hold 10 x 13 = 130 distinct rows. Capped
# at MAX_SHAPES the table would drop 66 of them — and drop them
# FIRST-COME-WINS, which does not merely lose rows, it biases the table toward
# whatever the session happened to route early. That table is the one a routing
# policy would be fitted on, so a systematic bias in it is worse than its
# bytes. Sized above the cross product so the cap cannot bite in normal
# operation and remains purely a guard against a caller inventing keys.
MAX_SHAPE_ROWS = 160

# The only waste signals that reach the wire, by name. `waste_signals` is typed
# `dict[str, int]` and has more than one producer, so an allowlist here — not a
# copy of whatever the dict happens to hold — is what keeps a future key from
# shipping before anyone has looked at it.
#
# `reread_compressed` is the reason this section exists. Headroom computes it
# already (see WasteSignals in config.py): re-reads attributable to our own
# over-compression rather than to agent behaviour. That is a regret label for
# the compressor, and a compression ratio without one is unfalsifiable — you
# can drive `saved_pct` to 90 by deleting the context and nothing here would
# object.
_WASTE_KEYS = (
    "json_bloat",
    "html_noise",
    "base64",
    "whitespace",
    "dynamic_date",
    "repetition",
)

# The output_shaper A/B labels, duplicated from
# `headroom.proxy.output_savings_policy` rather than imported. The telemetry
# package deliberately does not import `headroom.proxy` (see `record_stack`),
# and two string constants are a smaller price than that dependency.
_STRATUM_TREATMENT = "output_shaper:stratum:"
_STRATUM_CONTROL = "output_shaper:control:"

# Environment variables whose values may be reported, by name. An allowlist,
# never a dump of the environment: a variable absent from this tuple cannot
# reach the wire whatever it holds, and every value that does is put through
# `_safe_slug` first, so a var that unexpectedly holds a URL or a path reports
# "other" rather than its contents.
#
# Endpoints, tokens, paths, ports and hostnames are excluded by construction —
# only settings whose vocabulary is a small closed set are listed. This answers
# the question the corpus currently cannot: whether a session with poor savings
# is a compression failure or simply a configuration.
_CONFIG_ENV_KEYS = (
    "HEADROOM_MODE",
    "HEADROOM_SAVINGS_PROFILE",
    "HEADROOM_DEPLOYMENT_PROFILE",
    "HEADROOM_ROLLOUT_CHANNEL",
    "HEADROOM_CCR_BACKEND",
    "HEADROOM_MEMORY_INJECTION_MODE",
    "HEADROOM_OUTPUT_SHAPER",
    "HEADROOM_OUTPUT_HOLDOUT",
    "HEADROOM_LOSSLESS",
    "HEADROOM_DEDUPE",
    "HEADROOM_STATELESS",
    "HEADROOM_CODE_AWARE_ENABLED",
    "HEADROOM_PROTECT_TOOL_RESULTS",
    "HEADROOM_INTERCEPT_TOOL_RESULTS",
    "HEADROOM_DISABLE_KOMPRESS",
)

# Compression events arrive on the compression executor thread, mid-request,
# before that request's outcome ever reaches `SessionAggregator.record`. They
# are staged here rather than written straight into the live session, which
# keeps three things true at once:
#
#   * The executor thread never takes the aggregator's lock, so compression
#     cannot serialise against the request path. The beacon is on by default,
#     and ContentRouter observes once per routing decision — once per content
#     section per request — so that contention would be real.
#   * A compression event cannot CREATE a session. Sessions are started only by
#     an outcome, which preserves the invariant that every emitted session has
#     turns >= 1; otherwise a request abandoned between compression and its
#     outcome (Claude Code users interrupt streaming routinely) would emit a
#     phantom all-zero row that inflates fleet session and install counts.
#   * The first turn's numbers still survive, because the outcome that follows
#     milliseconds later drains this into the session it opens.
#
# A request that dies before its outcome leaves its events staged, and they are
# attributed to the next session instead. That is a rounding error against
# inventing a session that never happened.
_staged_lock = threading.Lock()
_staged_strategies: dict[str, list[int]] = {}
# Per-request stack slugs, for `detect_stack`'s by_stack branch. Same staging
# and the same reason: the proxy sees the X-Headroom-Stack header per request,
# and this is the only place the beacon can learn it without importing proxy.
_staged_stacks: dict[str, int] = {}
# (content_type, strategy) -> [sections, tokens_in, tokens_out]. Staged for the
# same reason as `_staged_strategies`: ContentRouter observes on the compression
# executor thread, mid-request, and must not take the aggregator lock.
#
# `_staged_strategies` records what each compressor YIELDED but never what it
# was HANDED, so the corpus has no left-hand side and cannot express the one
# function worth learning: shape -> strategy -> yield. This is that term.
_staged_shapes: dict[tuple[str, str], list[int]] = {}
# tool-output shape slug -> [events, tokens_in, tokens_out]. The structural
# descriptor only — never TOIN's `structure_hash`. See `record_tool_shape`.
_staged_tool_shapes: dict[str, list[int]] = {}


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


def _client_slug(value: Any) -> str:
    """Slug a client-harness id, translating `-` to `_` first.

    `classify_client` returns hyphenated ids -- `claude-code`, `claude-vscode`,
    `anthropic-cli` -- and `_SLUG_RE` does not admit a hyphen, so slugging one
    directly reports "other" for three of the thirteen known harnesses,
    including the one that drives most of the fleet. That defeats the entire
    point of reporting the field.

    Deliberately NOT fixed by widening `_SLUG_RE`: that validator guards the
    v1 skip reasons and strategy names, where a value that fails today already
    ships as "other" and has done so in every row of the corpus. Widening it
    would quietly change what those existing fields mean. A separate function
    for the one field with a different convention is the smaller change.
    """
    text = str(value or "").strip().lower().replace("-", "_")
    return text if _SLUG_RE.match(text) else "other"


def _bucket(edges: tuple[float, ...], value: float) -> int:
    """Index of the first edge `value` falls under, else the open-ended tail."""
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def _bump(counter: dict[str, int], key: str, *, cap: int = MAX_SHAPES) -> None:
    """Increment a bounded slug->count map. Over the cap, new keys are dropped.

    Dropped rather than bucketed into "other": every caller here feeds a
    vocabulary that is closed in code, so hitting the cap means something
    upstream is generating keys per request, and a silently growing "other"
    would hide that where a flat count makes it obvious.
    """
    if key not in counter and len(counter) >= cap:
        return
    counter[key] = counter.get(key, 0) + 1


def _turn_kind(*, cached: bool, status: int, passthrough: bool, attempted: int, saved: int) -> str:
    """Classify one turn into the closed `_TURN_KINDS` alphabet.

    Structural only: every input is a status code, a token count, or a flag the
    proxy already set. Nothing here reads content, and the return value is one
    of seven literals, so the trajectory string cannot carry free text.
    """
    if cached:
        return "c"
    if status >= 500:
        return "f"
    if 400 <= status < 500:
        return "e"
    if passthrough:
        return "p"
    if attempted > 0:
        return "s" if saved > 0 else "z"
    return "o"


def _config_snapshot() -> dict[str, str]:
    """Allowlisted, slug-validated view of the operator's configuration.

    Two layers of protection, because one is not enough for anything that reads
    the environment: only the names in `_CONFIG_ENV_KEYS` are consulted at all,
    and every value that survives that is put through `_safe_slug`, so a
    variable that unexpectedly holds a path, a URL or a token reports "other".

    Unset variables are omitted rather than reported as a default, so absent
    means "operator expressed no preference" and the reader is never guessing
    which release's default applied.
    """
    out: dict[str, str] = {}
    for name in _CONFIG_ENV_KEYS:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        out[name.removeprefix("HEADROOM_").lower()] = _safe_slug(raw)
    return out


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


def read_install_id() -> str | None:
    """The install id if one already exists, WITHOUT creating one.

    Split out of `install_id` because minting an identifier is a side effect,
    and there are callers that must not have it. `headroom telemetry --show`
    is the one that matters: someone running it to find out what leaves their
    machine — very plausibly on the way to switching the beacon off — must not
    have that inspection be the thing that gives the machine its id.
    """
    global _install_id
    with _identity_lock:
        if _install_id is not None:
            return _install_id
        try:
            existing = _install_id_path().read_text().strip()
        except OSError:
            return None
        if existing:
            _install_id = existing
            return _install_id
        return None


def install_id() -> str:
    """Random per-install id. Not a machine fingerprint — delete the file to reset.

    Deliberately not derived from hostname, MAC, or any hardware property:
    anything a user cannot change reads as tracking, and a random UUID answers
    every question the beacon actually asks.

    Creates and persists one if none exists. Use :func:`read_install_id` where
    that side effect is not wanted.
    """
    global _install_id
    existing = read_install_id()
    if existing is not None:
        return existing
    with _identity_lock:
        # Re-check: another thread may have won the race while this one was
        # outside the lock in read_install_id().
        if _install_id is not None:
            return _install_id
        path = _install_id_path()
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
    *,
    install_mode: str | None = None,
    stack: str | None = None,
    create_install_id: bool = True,
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
        "os.type": platform.system().lower(),
        "host.arch": platform.machine().lower(),
    }
    # `create_install_id=False` is for callers that only want to LOOK at what
    # would be sent. Minting an id is a side effect, and an inspection command
    # must not have it — see `read_install_id`. The key is omitted rather than
    # blank so a reader can tell "no id yet" from "id withheld".
    resolved = install_id() if create_install_id else read_install_id()
    if resolved:
        attrs["headroom.install_id"] = resolved
    if install_mode:
        attrs["headroom.install_mode"] = install_mode
    # Detect when the caller did not supply one. Every caller so far supplies
    # nothing, so `headroom.stack` was absent from the entire corpus while the
    # detector sat unused — which made the fleet unsegmentable by agent, the
    # question the corpus is most often asked ("what does this look like under
    # Claude Code?").
    #
    # The env vars detect_stack checks first are only set by `headroom wrap`.
    # The common deployment points an agent at a persistent proxy through
    # ANTHROPIC_BASE_URL and sets neither, so environment-only detection would
    # answer the literal "proxy" for almost the whole fleet — a populated,
    # authoritative-looking column that cannot answer the question it exists
    # for. The slugs staged by `record_stack` are that fleet's only real
    # signal, so they are fed to detect_stack's by_stack branch.
    if stack is None:
        try:
            from headroom.telemetry.context import detect_stack

            with _staged_lock:
                by_stack = dict(_staged_stacks)
            stack = detect_stack({"requests": {"by_stack": by_stack}} if by_stack else None)
        except Exception:  # a broken detector must not silence telemetry
            logger.debug("telemetry: stack detection failed", exc_info=True)
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
    failure_statuses: dict[str, int] = field(default_factory=dict)
    passthrough_turns: int = 0
    response_cache_hits: int = 0
    overhead_ms: float = 0.0
    latency_ms: float = 0.0
    transforms: dict[str, int] = field(default_factory=dict)
    # strategy slug -> [events, tokens_in, tokens_out]. `transforms` says which
    # compressors ran; this says whether they were worth running. A list rather
    # than three parallel dicts so the three numbers cannot drift apart.
    strategies: dict[str, list[int]] = field(default_factory=dict)
    skips: dict[str, int] = field(default_factory=dict)
    sources: dict[str, int] = field(default_factory=dict)
    providers: set[str] = field(default_factory=set)
    models: set[str] = field(default_factory=set)

    # ---------------------------------------------------------- schema v2 --
    # Additive only. Every field below is a running total like the v1 counters
    # above it, so the "highest seq per (install, session) is the whole
    # session" dedupe rule is unchanged and a dropped heartbeat still costs
    # nothing. Anything whose natural form is a last-value (msgs_max) is
    # written as a maximum instead, which behaves identically under that rule.
    hists: dict[str, list[int]] = field(
        default_factory=lambda: {
            name: [0] * (len(edges) + 1) for name, edges in _HIST_EDGES.items()
        }
    )
    waste: dict[str, int] = field(default_factory=dict)
    waste_turns: int = 0
    reread_tokens: int = 0
    reread_compressed_tokens: int = 0
    shapes: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    tool_shapes: dict[str, list[int]] = field(default_factory=dict)
    cache_gap_hits: list[int] = field(default_factory=lambda: [0] * (len(_CACHE_GAP_EDGES) + 1))
    cache_gap_misses: list[int] = field(default_factory=lambda: [0] * (len(_CACHE_GAP_EDGES) + 1))
    cache_write_5m: int = 0
    cache_write_1h: int = 0
    cache_inferred_turns: int = 0
    traj_turns: list[int] = field(default_factory=lambda: [0] * _TRAJECTORY_BUCKETS)
    traj_tokens: list[int] = field(default_factory=lambda: [0] * _TRAJECTORY_BUCKETS)
    traj_saved: list[int] = field(default_factory=lambda: [0] * _TRAJECTORY_BUCKETS)
    # [kind, run_length] pairs, capped at MAX_TRAJECTORY_RUNS.
    kind_runs: list[list[Any]] = field(default_factory=list)
    kinds_truncated: bool = False
    msgs_max: int = 0
    clients: dict[str, int] = field(default_factory=dict)
    # 4xx is counted HERE and never in `failures` / `failure_statuses`. Those
    # two are 5xx-only in every row of the existing corpus, and widening them
    # would silently redefine a metric rather than add one — a session that
    # rate-limited twice would start reading as a session that failed twice.
    client_errors: int = 0
    client_error_statuses: dict[str, int] = field(default_factory=dict)
    strata_arm: dict[str, int] = field(default_factory=dict)
    strata_turn_kind: dict[str, int] = field(default_factory=dict)
    strata_input_bucket: dict[str, int] = field(default_factory=dict)
    strata_tools: dict[str, int] = field(default_factory=dict)

    def observe(self, name: str, value: float) -> None:
        """Record one sample into a named histogram."""
        counts = self.hists.get(name)
        if counts is not None:
            counts[_bucket(_HIST_EDGES[name], value)] += 1

    def append_kind(self, kind: str) -> None:
        """Extend the run-length trajectory string by one turn.

        Stops dead at the cap rather than continuing to grow the last run.
        Letting a matching kind keep incrementing looks harmless and is not:
        once a run has been dropped the remaining turns are no longer
        contiguous, so `s6` would claim six consecutive turns of `s` where the
        session actually did `s q s q s q ...`. A run length that is really a
        filtered count is worse than a short string, because nothing in the
        payload says which one you are reading.

        Stopping keeps the string a faithful PREFIX of the session — every run
        in it is true — with `kinds_truncated` marking that more happened after
        it. `trajectory.turns` is uncapped and still sums to `session.turns`,
        so the turn count is never lost.
        """
        if self.kinds_truncated:
            return
        if self.kind_runs and self.kind_runs[-1][0] == kind:
            self.kind_runs[-1][1] += 1
            return
        if len(self.kind_runs) >= MAX_TRAJECTORY_RUNS:
            self.kinds_truncated = True
            return
        self.kind_runs.append([kind, 1])

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
        permanent hole. The redundancy is ~1.3KB per report on the wire, which
        is free -- but only because the body is gzipped. Uncompressed, a
        schema-v2 report is ~8KB and re-sending it every five minutes is not
        free at all; see the module docstring. Anything that grows the payload
        further should be weighed against this line, not against the ~2KB it
        used to say.

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
                # The two above are context-compression only, because
                # `tool_saved` never lands in original/attempted. On a
                # tool-heavy fleet that understates the product several-fold —
                # observed 2.80% vs 12.82% across the first 516 sessions.
                # These two are what the dashboard headline shows
                # (`tokens.savings_percent` / `active_savings_percent` in
                # server.py): tool-schema savings added to BOTH sides, since
                # deferred schemas were attempted work that succeeded whole.
                # Kept alongside rather than folded into `saved_pct`, which
                # already means context-only in every row of the corpus.
                "all_layers_saved_pct": _pct(
                    self.tokens_saved + self.tool_saved_tokens,
                    self.original_tokens + self.tool_saved_tokens,
                ),
                "all_layers_yield_pct": _pct(
                    self.tokens_saved + self.tool_saved_tokens,
                    self.attempted_tokens + self.tool_saved_tokens,
                ),
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
                # Per-strategy effectiveness. `transforms` counts invocations,
                # which cannot tell a compressor that saved 60% from one that
                # ran constantly and saved nothing — the fleet's top transform
                # by count contributes an unknown share of `tokens.saved`.
                #
                # These do NOT sum to `tokens.saved`, and must not be presented
                # as if they do: strategies compose (the router routes, a
                # strategy runs inside it) so the same text is measured by more
                # than one, and tool-schema savings never appear here at all.
                # Read a row as "of what this strategy was handed, it removed
                # this much" — a per-strategy yield, not a share of the total.
                #
                # A LIST of uniform records, not a {strategy: {...}} object,
                # and that shape is deliberate. DuckDB infers a JSON object as
                # a STRUCT while its keys are few and consistent and as a MAP
                # once they are not, so an object keyed by strategy would
                # change COLUMN TYPE as the fleet adopts new compressors —
                # exactly the break that silently took out the `transforms`
                # report. A list of records has fixed field names, so the type
                # is the same on day one and after the 30th strategy ships, and
                # a new field inside a record is absorbed by union_by_name.
                # Sorted so a payload is byte-comparable between heartbeats.
                "by_strategy": [
                    {
                        "strategy": name,
                        "n": counts[0],
                        "tokens_in": counts[1],
                        "tokens_out": counts[2],
                    }
                    for name, counts in sorted(self.strategies.items())
                ],
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
            # The same failures split by status, because the count alone cannot
            # answer the only question worth asking about it: a 529 is the
            # provider shedding load (nothing to fix here) and a 500 is usually
            # ours. Keys are the bare status string; the set is closed and tiny
            # (500/502/503/504/529), so this needs no slug bounding.
            "failure_statuses": dict(self.failure_statuses),
        }
        # Merged rather than built inline, so the isolation the v2 fields claim
        # elsewhere is structural here too. Built inline, one bad expression in
        # any v2 section aborts the whole dict literal and the event is dropped
        # -- taking `tokens.saved` and every other v1 counter with it. The worst
        # case has to be a missing v2 key, never a lost session.
        try:
            snapshot.update(self._payload_v2())
        except Exception:
            logger.debug("telemetry: v2 payload section failed", exc_info=True)
        self.seq += 1
        return snapshot

    def _payload_v2(self) -> dict[str, Any]:
        """The schema-v2 half of the snapshot. See `payload`."""
        return {
            # Nothing below reads or rewrites a key above it. Dropping any one
            # of these from the worker's ALLOWED_KEYS reverts that signal on
            # its own, without touching the v1 payload.
            #
            # Did compression COST anything? `saved_pct` cannot answer that,
            # and on its own it is a metric you can maximise by deleting the
            # context. `reread_compressed` is the counter-pressure: tokens the
            # agent had to fetch again because we compressed them away. Read
            # the pair — a rising `saved_pct` with a rising
            # `reread_compressed` is not a win.
            #
            # `reread_compressed_tokens` is a SUBSET of `reread_tokens`, not a
            # sibling of it: the same tokens are counted in both, exactly as
            # WasteSignals.total() documents upstream. Never add them. The
            # ratio between them is the useful quantity — it splits re-reads
            # the agent would have done anyway from the ones Headroom caused.
            #
            # `waste_turns` counts turns where ANY of these fired, re-reads
            # included, so it is the denominator for this whole block rather
            # than for `waste` alone.
            "quality": {
                "reread_tokens": self.reread_tokens,
                "reread_compressed_tokens": self.reread_compressed_tokens,
                # Turns where any waste signal fired at all, so the sums below
                # have a denominator that is not "every turn".
                "waste_turns": self.waste_turns,
                # Waste DETECTED in the incoming request, by kind — the
                # headroom that existed, whether or not anything took it.
                # Sorted, like every map here, so heartbeats stay
                # byte-comparable.
                # A list of records, not an object keyed by waste kind. See
                # the note on `clients` — the key set here is data-dependent
                # too (only the kinds that fired appear), so an object would
                # infer a different column type from one corpus to the next.
                "waste": [
                    {"kind": kind, "tokens": tokens} for kind, tokens in sorted(self.waste.items())
                ],
            },
            # Distributions for the quantities the payload otherwise reports
            # only as totals. `counts` is len(edges) + 1; `edges` ships with
            # every row so a reader never has to know the client version.
            #
            # `turn_tokens` and `overhead_ms` are observed on every turn and
            # sum to `session.turns`. The other three are observed only where
            # the quantity is defined (original > 0, a measured ttfb, a known
            # message count) and sum to less.
            "hist": {
                name: {"edges": list(_HIST_EDGES[name]), "counts": list(counts)}
                for name, counts in sorted(self.hists.items())
            },
            # The left-hand side compression has never had. `by_content` is
            # the joint table of (what the router saw) x (what it chose) x
            # (what that returned) — a per-shape yield, and the table a
            # routing policy is fitted on. `by_tool` is the same question for
            # JSON tool output, keyed by structure rather than by type.
            #
            # Lists of records, not objects keyed by shape, for the reason
            # spelled out on `by_strategy`: DuckDB infers an object with few
            # consistent keys as a STRUCT and one with many as a MAP, so an
            # object here would change column type as the fleet meets new
            # content. A list of records has fixed field names forever.
            "shapes": {
                "by_content": [
                    {
                        "content": content,
                        "strategy": strategy,
                        "n": counts[0],
                        "tokens_in": counts[1],
                        "tokens_out": counts[2],
                    }
                    for (content, strategy), counts in sorted(self.shapes.items())
                ],
                "by_tool": [
                    {
                        "shape": shape,
                        "n": counts[0],
                        "tokens_in": counts[1],
                        "tokens_out": counts[2],
                    }
                    for shape, counts in sorted(self.tool_shapes.items())
                ],
                # Set when `_fit_to_wire` had to shed rows to stay under the
                # receiver's body cap. ALWAYS present, never conditional: an
                # optional key would give `shapes` two different STRUCT layouts
                # across the corpus, which is the same failure the lists above
                # exist to avoid.
                "truncated": False,
            },
            # Prompt-cache physics. `hits`/`misses` are indexed by how long the
            # turn waited since the previous one, which makes (gap -> hit rate)
            # an empirical survival curve for a TTL no provider documents
            # honestly. The first turn of a session has no previous turn and is
            # excluded from both, so hits + misses == turns - 1.
            "cache": {
                "gap_edges_s": list(_CACHE_GAP_EDGES),
                "hits": list(self.cache_gap_hits),
                "misses": list(self.cache_gap_misses),
                "write_5m": self.cache_write_5m,
                "write_1h": self.cache_write_1h,
                # Turns whose write column the provider did not report and
                # Headroom inferred (OpenAI). Without this the two are
                # indistinguishable and the estimate reads as measurement.
                "inferred_turns": self.cache_inferred_turns,
            },
            # The session's SHAPE over time rather than its totals. Buckets are
            # log-spaced by turn index (turn 1, turns 2-3, 4-7, ... 512+), so
            # the arrays describe how context grew and where compression
            # started or stopped paying — the input to predicting a session's
            # terminal size from its opening turns.
            "trajectory": {
                "buckets": _TRAJECTORY_BUCKETS,
                "turns": list(self.traj_turns),
                "tokens": list(self.traj_tokens),
                "saved": list(self.traj_saved),
                # Run-length encoded turn kinds from the closed `_TURN_KINDS`
                # alphabet, e.g. "s12p3z40s2". Structural: a kind is derived
                # from a status code and two token counts, never from content.
                "kinds": "".join(f"{kind}{run}" for kind, run in self.kind_runs),
                "kinds_truncated": self.kinds_truncated,
                "msgs_max": self.msgs_max,
            },
            # Which harness drove the session. `headroom.stack` answers this by
            # detection and, as its own note concedes, resolves to the literal
            # "proxy" for most of the fleet; this is the value the handler
            # actually identified. Turns with no identified client are absent
            # rather than bucketed, so the values sum to <= `session.turns`.
            # A LIST of records, not an object keyed by client. This is the
            # same decision `by_strategy` documents, and it is not stylistic:
            # DuckDB infers a JSON object as a STRUCT while its keys are few
            # and identifier-shaped, and as a MAP once they are not, so the
            # COLUMN TYPE would be a function of which harnesses happened to
            # appear in the data. Measured on a real corpus: two harnesses
            # infer STRUCT(claude_code, codex) and a query for
            # `clients.cursor` fails to bind; twenty-two infer
            # MAP(VARCHAR, BIGINT) and the same query succeeds. That is
            # exactly the break that silently took out the transforms report,
            # and the client vocabulary grows every time a harness is added.
            #
            # A list of records has fixed field names forever, so the type is
            # the same on day one and after the thirtieth harness ships.
            "clients": [
                {"client": client, "n": turns} for client, turns in sorted(self.clients.items())
            ],
            # 4xx, kept strictly apart from `failures`. 429 is the one that
            # matters: fleet-wide rate-limit rate per provider and hour is a
            # live view of which upstream is degraded, which is a routing
            # input and is invisible in a 5xx-only count.
            "errors": {
                "count": self.client_errors,
                "by_status": dict(sorted(self.client_error_statuses.items())),
            },
            # output_shaper A/B strata, as validated enums rather than as the
            # free-text label suffix `_fold` strips. Same instinct as that
            # strip — the FORMAT was the problem — but the arm and the request
            # features are what make the experiment readable at all.
            #
            # Model family is deliberately NOT here. It is the one component of
            # the stratum key that names something outside the request, the
            # existing `models` field already reports it wherever a public
            # registry vouches for the id, and this payload has no business
            # being a second, unvetted route for it.
            # One list rather than four objects, for the reason on `clients`:
            # each of these was an object whose keys are whichever stratum
            # values the session happened to hit, so all four had a
            # data-dependent column type. Flattened to (dimension, value,
            # count) records they are one stable column that pivots in a
            # GROUP BY, and a new stratum dimension adds rows instead of
            # changing a type.
            "strata": [
                {"dim": dim, "value": value, "n": turns}
                for dim, counter in (
                    ("arm", self.strata_arm),
                    ("turn_kind", self.strata_turn_kind),
                    ("input_bucket", self.strata_input_bucket),
                    ("tools", self.strata_tools),
                )
                for value, turns in sorted(counter.items())
            ],
            # Allowlisted configuration. Lets the corpus separate "compression
            # underperformed" from "compression was switched off", which the
            # token counts alone cannot do. Names and values are both bounded —
            # see `_config_snapshot`.
            # Records, not an object: every operator sets a different subset
            # of `_CONFIG_ENV_KEYS`, so the key set is the most data-dependent
            # of the four. Same failure, same fix.
            "config": [
                {"key": key, "value": value} for key, value in sorted(_config_snapshot().items())
            ],
        }


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

    # Captured before `last_seen` moves: the gap between this turn and the one
    # before it is what the cache-survival curve is indexed on. None on the
    # first turn of a session, which has no predecessor to measure against.
    prev_seen = sess.last_seen if sess.turns else None
    sess.last_seen = now
    sess.turns += 1
    sess.sources[source] = sess.sources.get(source, 0) + 1

    # Compression ran on the executor thread before this outcome arrived; take
    # what it staged. Done here rather than in the observer so the executor
    # thread never touches the aggregator lock — see the note on _staged_lock.
    for name, staged in _drain_staged_strategies().items():
        counts = sess.strategies.get(name)
        if counts is None:
            if len(sess.strategies) >= MAX_STRATEGIES:
                continue
            counts = [0, 0, 0]
            sess.strategies[name] = counts
        counts[0] += staged[0]
        counts[1] += staged[1]
        counts[2] += staged[2]
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
    status = int(get("status_code", 200) or 200)
    if status >= 500:
        sess.failures += 1
        # ponytail: str(status) verbatim for the 5xx range, one bucket for
        # anything outside it. Nothing here can be user data, and the range
        # check is what keeps a garbage status_code from inventing map keys.
        key = str(status) if status < 600 else "other"
        sess.failure_statuses[key] = sess.failure_statuses.get(key, 0) + 1
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

    # Every v1 counter is folded above and is not read below. Guarded for the
    # same reason `payload` guards its v2 half: unguarded, a raise here aborts
    # `_fold` before the caller's heartbeat check and costs a whole emission,
    # which is a v1 consequence for a v2 fault.
    try:
        _fold_extra(sess, get, now, prev_seen, tags, status)
    except Exception:
        logger.debug("telemetry: v2 fold failed", exc_info=True)


def _fold_extra(
    sess: _Session,
    get: Callable[..., Any],
    now: float,
    prev_seen: float | None,
    tags: Any,
    status: int,
) -> None:
    """Fold the schema-v2 signals. Additive by construction.

    Split out of `_fold` rather than appended to it, for one reason: every v1
    counter is finished before this runs and nothing here reads or writes one.
    A mistake in a new signal can therefore only cost that signal. The corpus
    has years of rows whose meaning depends on `tokens.saved` and
    `failure_statuses` staying exactly what they have always been, and a
    boundary is a better guarantee of that than care is.

    Duck-typed through the same `get` as `_fold`, so `_McpCompression` — which
    knows four fields — still folds without inventing the rest.
    """
    original = int(get("original_tokens") or 0)
    saved = int(get("tokens_saved") or 0)
    attempted = int(get("attempted_input_tokens") or 0)
    is_dict_tags = isinstance(tags, dict)
    passthrough = is_dict_tags and "passthrough_reason" in tags
    cached = bool(get("from_response_cache", False))

    # -- distributions ------------------------------------------------------
    sess.observe("turn_tokens", original)
    sess.observe("overhead_ms", float(get("overhead_ms", 0.0) or 0.0))
    if original > 0:
        # Undefined without a denominator, and piling those into the 0% bucket
        # would report a compression failure where there was no request.
        sess.observe("saved_pct", _pct(saved, original))
    ttfb = float(get("ttfb_ms", 0.0) or 0.0)
    if ttfb > 0:
        # 0 is the convention for "not measured" on this field (non-streaming
        # paths), not for "instant".
        sess.observe("ttfb_ms", ttfb)
    msgs = int(get("num_messages") or 0)
    if msgs > 0:
        sess.observe("msgs", msgs)
        if msgs > sess.msgs_max:
            sess.msgs_max = msgs

    # -- regret / waste -----------------------------------------------------
    waste = get("waste_signals", None)
    if isinstance(waste, dict):
        touched = False
        for name in (*_WASTE_KEYS, "reread", "reread_compressed"):
            try:
                amount = int(waste.get(name, 0) or 0)
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            touched = True
            if name == "reread":
                sess.reread_tokens += amount
            elif name == "reread_compressed":
                sess.reread_compressed_tokens += amount
            else:
                sess.waste[name] = sess.waste.get(name, 0) + amount
        if touched:
            sess.waste_turns += 1

    # -- cache physics ------------------------------------------------------
    if prev_seen is not None:
        index = _bucket(_CACHE_GAP_EDGES, now - prev_seen)
        if int(get("cache_read_tokens") or 0) > 0:
            sess.cache_gap_hits[index] += 1
        else:
            sess.cache_gap_misses[index] += 1
    sess.cache_write_5m += int(get("cache_write_5m_tokens") or 0)
    sess.cache_write_1h += int(get("cache_write_1h_tokens") or 0)
    if get("cache_inferred", False):
        sess.cache_inferred_turns += 1

    # -- trajectory ---------------------------------------------------------
    # `sess.turns` was incremented by _fold, so turn 1 lands in bucket 0.
    index = min(sess.turns.bit_length() - 1, _TRAJECTORY_BUCKETS - 1)
    sess.traj_turns[index] += 1
    sess.traj_tokens[index] += original
    sess.traj_saved[index] += saved
    sess.append_kind(
        _turn_kind(
            cached=cached,
            status=status,
            passthrough=passthrough,
            attempted=attempted,
            saved=saved,
        )
    )

    # -- harness and 4xx ----------------------------------------------------
    client = get("client", None)
    if client:
        _bump(sess.clients, _client_slug(client))
    if 400 <= status < 500:
        sess.client_errors += 1
        # `status` is already an int in this range, so the key is a bounded
        # numeric string and needs no slug guard.
        _bump(sess.client_error_statuses, str(status))

    # -- A/B strata ---------------------------------------------------------
    # Re-read rather than shared with the transforms loop above: that loop owns
    # a v1 field and is left exactly as it was.
    for name in get("transforms_applied", ()) or ():
        text = str(name)
        if text.startswith(_STRATUM_TREATMENT):
            arm, key = "treatment", text[len(_STRATUM_TREATMENT) :]
        elif text.startswith(_STRATUM_CONTROL):
            arm, key = "control", text[len(_STRATUM_CONTROL) :]
        else:
            continue
        _bump(sess.strata_arm, arm)
        # family|turn_kind|input_bucket|tools. Index 0 is the model family and
        # is deliberately dropped -- see the note in payload().
        parts = key.split("|")
        for position, counter in (
            (1, sess.strata_turn_kind),
            (2, sess.strata_input_bucket),
            (3, sess.strata_tools),
        ):
            if len(parts) > position:
                _bump(counter, _safe_slug(parts[position]))

    # -- shape tables -------------------------------------------------------
    # Drained here for the same reason `_staged_strategies` is drained in
    # `_fold`: a compression event must not be able to open a session.
    for shape_key, staged in _drain_staged_shapes().items():
        counts = sess.shapes.get(shape_key)
        if counts is None:
            if len(sess.shapes) >= MAX_SHAPE_ROWS:
                continue
            counts = [0, 0, 0]
            sess.shapes[shape_key] = counts
        for i in range(3):
            counts[i] += staged[i]
    for tool_key, staged in _drain_staged_tool_shapes().items():
        counts = sess.tool_shapes.get(tool_key)
        if counts is None:
            if len(sess.tool_shapes) >= MAX_SHAPE_ROWS:
                continue
            counts = [0, 0, 0]
            sess.tool_shapes[tool_key] = counts
        for i in range(3):
            counts[i] += staged[i]


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


def _send(endpoint: str, body: bytes, agent: str, timeout: float, *, compress: bool) -> None:
    """One POST. Raises `_CompressionRejected` if gzip is why it was refused."""
    headers = {
        "content-type": "application/json",
        # Required, not cosmetic. urllib defaults to "Python-urllib/3.x",
        # which Cloudflare blocks outright by browser signature (error
        # 1010) before the request ever reaches the Worker. Combined
        # with fire-and-forget error handling that failure mode is
        # invisible: every upload 403s and the beacon looks healthy.
        "user-agent": agent,
    }
    if compress:
        body = gzip.compress(body, _GZIP_LEVEL)
        # Spec-correct for OTLP/HTTP, which is what makes this work against a
        # generic collector via HEADROOM_TELEMETRY_ENDPOINT. Our own Worker
        # ignores it and sniffs the gzip magic number instead, because an
        # intermediary may have decompressed the body before it arrives.
        headers["content-encoding"] = "gzip"
    request = urllib.request.Request(endpoint, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            pass
    except urllib.error.HTTPError as err:
        try:
            err.close()
        except Exception:
            pass
        # "Could not read this body": a Worker older than the one that learned
        # to inflate (its JSON.parse throws on the gzip bytes and it answers
        # 400), or a collector that never will (415). Deliberately not the
        # whole 4xx range -- see _GZIP_REFUSED_STATUSES. 5xx is the endpoint
        # being unwell and says nothing about gzip, so it must not disable
        # compression forever either.
        if compress and err.code in _GZIP_REFUSED_STATUSES:
            raise _CompressionRejected from err
        raise


# What one POST may put on the wire.
#
# The receiver answers an oversized body with 413, and 413 is deliberately
# absent from `_GZIP_REFUSED_STATUSES` because the uncompressed retry is larger
# still -- so a 413 is total loss of the event, v1 counters included. The client
# cannot discover the far end's cap (the response arrives after the body), so it
# has to bound itself instead of hoping the two numbers match.
#
# 60KB sits under the OLDEST receiver still in service, whose cap is 64KB. That
# is the number that matters, not the 256KB the current Worker allows: an
# install upgrades on its own schedule and may be pointed at any deployment.
#
# Measured: schema v2 is ~8KB before the shape tables and crosses 64KB at ~116
# rows in each, against caps of MAX_SHAPE_ROWS. So this binds only the busiest
# sessions, and only while bodies are uncompressed -- gzip puts a saturated
# payload at ~4KB, where the budget is unreachable and nothing is ever shed.
_MAX_WIRE_BYTES = 60 * 1024


def _fit_to_wire(payload: dict[str, Any], resource: dict[str, str], *, compress: bool) -> bytes:
    """Encode `payload`, shedding shape rows until it fits `_MAX_WIRE_BYTES`.

    Budgets what ARRIVES, so the check runs against the compressed length when
    compression is on. That is what makes this self-cancelling: once gzip ships,
    a saturated payload is ~4KB and this never trims anything.

    Mutates `payload` in place when it trims. Safe because the aggregator
    builds a fresh snapshot per emit and hands it to exactly one sink.

    `shapes` is what gets shed because it is ~90% of a large body and the only
    part that is genuinely aggregate -- the same (content x strategy) cells are
    re-measured by every other session, so dropping the thinnest rows from one
    of them costs resolution, not a fact. Everything else is a session-scoped
    total that exists nowhere else. Rows are ranked by `n` so what survives is
    what carries the most evidence.

    Never raises: a failure here degrades to sending the untrimmed body, which
    is exactly what would have been sent without this function.
    """

    def encode(candidate: dict[str, Any]) -> tuple[bytes, int]:
        raw = json.dumps(build_otlp_logs(candidate, resource), separators=(",", ":")).encode()
        return raw, len(gzip.compress(raw, _GZIP_LEVEL)) if compress else len(raw)

    raw, wire = encode(payload)
    try:
        if wire <= _MAX_WIRE_BYTES:
            return raw
        shapes = payload.get("shapes")
        if not isinstance(shapes, dict):
            return raw
        by_content = sorted(shapes.get("by_content") or [], key=lambda r: -r.get("n", 0))
        by_tool = sorted(shapes.get("by_tool") or [], key=lambda r: -r.get("n", 0))
        if not (by_content or by_tool):
            return raw

        # Largest surviving prefix that fits. Binary search rather than dropping
        # a row at a time: a saturated payload is 320 rows, and each probe
        # re-encodes ~80KB.
        fitted = False
        lo, hi = 0, max(len(by_content), len(by_tool))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            shapes["by_content"] = by_content[:mid]
            shapes["by_tool"] = by_tool[:mid]
            shapes["truncated"] = True
            _, wire = encode(payload)
            if wire <= _MAX_WIRE_BYTES:
                fitted, lo = True, mid
            else:
                hi = mid - 1
        if fitted:
            # Re-sorted back into the canonical order the payload emits. The
            # ranking above is only a selection rule; letting it leak into the
            # wire format would mean the row order of `by_content` silently
            # depended on whether the body happened to need trimming.
            shapes["by_content"] = sorted(
                by_content[:lo],
                key=lambda r: (r.get("content", ""), r.get("strategy", "")),
            )
            shapes["by_tool"] = sorted(by_tool[:lo], key=lambda r: r.get("shape", ""))
            shapes["truncated"] = True
            # Re-encoded rather than reusing the probe's bytes: reordering the
            # same records cannot change the length, so this still fits.
            return encode(payload)[0]
        # Even zero shape rows did not fit. Send the stripped body anyway: the
        # v1 counters are the point, and a 413 would lose them too.
        shapes["by_content"] = []
        shapes["by_tool"] = []
        shapes["truncated"] = True
        return encode(payload)[0]
    except Exception:
        logger.debug("telemetry: wire-size trim failed", exc_info=True)
        return raw


def _post_blocking(payload: dict[str, Any], timeout: float = _POST_TIMEOUT_S) -> None:
    endpoint = os.environ.get("HEADROOM_TELEMETRY_ENDPOINT", DEFAULT_ENDPOINT)
    # Hoisted above the encode: the wire budget measures what actually leaves,
    # so it has to know whether that will be compressed or plain.
    compress = _gzip_enabled()
    try:
        from headroom._version import get_version

        resource = resource_attributes()
        body = _fit_to_wire(payload, resource, compress=compress)
        agent = f"headroom-beacon/{get_version()}"
    except Exception:
        logger.debug("telemetry: session POST failed", exc_info=True)
        return

    # Compress by default, and fall back the moment the endpoint says it cannot
    # read it. This is what makes the rollout order forgiving: shipping this
    # client against a Worker that predates gzip costs one extra round trip per
    # process and no data, instead of silently dropping every upload.
    #
    # That failure mode is not hypothetical here. Uploads are fire-and-forget
    # and the status is otherwise ignored, so "every POST is refused" and
    # "everything is fine" look identical from inside the process -- exactly
    # how the Cloudflare 1010 user-agent block went unnoticed.
    if compress:
        try:
            _send(endpoint, body, agent, timeout, compress=True)
            return
        except _CompressionRejected:
            logger.debug("telemetry: endpoint refused gzip; falling back", exc_info=True)
            _disable_gzip()
            # Re-budget before resending. `body` was measured compressed, and
            # the same bytes plain can be ~6x larger -- the one case where the
            # fallback could turn a refusal into a 413 and lose the event it
            # was added to save.
            try:
                body = _fit_to_wire(payload, resource, compress=False)
            except Exception:
                logger.debug("telemetry: re-trim after gzip refusal failed", exc_info=True)
        except Exception:
            logger.debug("telemetry: session POST failed", exc_info=True)
            return
    try:
        _send(endpoint, body, agent, timeout, compress=False)
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


def record_compression(strategy: str, original_tokens: int, compressed_tokens: int) -> None:
    """Beacon entry point for one compression event.

    Signature-compatible with
    :class:`headroom.transforms.observability.CompressionObserver`, so the
    proxy's existing observer can forward here without a second measurement
    pass — the numbers are already computed on the hot path for Prometheus
    (``PrometheusMetrics.tokens_saved_by_strategy``); they just never left the
    process.

    Same discipline as the rest of this module: off by default and cheap when
    off, never raises. This runs once per routing decision, so it must not do
    anything a request would notice.
    """
    from headroom.telemetry.beacon import is_beacon_enabled

    if not is_beacon_enabled():
        return
    # A slug, not the raw string. The real values are CompressionStrategy enum
    # tags, but the observer protocol takes a free string, and anything that is
    # not already a bounded lowercase identifier collapses to "other" rather
    # than reaching the wire.
    slug = _safe_slug(strategy)
    try:
        before = int(original_tokens or 0)
        after = int(compressed_tokens or 0)
    except (TypeError, ValueError):
        return
    if before <= 0:
        return
    # Clamped at the input: a compressor that emits more than it received is a
    # bug, and letting `out` exceed `in` would surface downstream as negative
    # savings rather than as the bug it is. Prometheus clamps the same way.
    after = min(max(after, 0), before)
    with _staged_lock:
        counts = _staged_strategies.get(slug)
        if counts is None:
            if len(_staged_strategies) >= MAX_STRATEGIES:
                return
            counts = [0, 0, 0]
            _staged_strategies[slug] = counts
        counts[0] += 1
        counts[1] += before
        counts[2] += after


def _stage(store: dict[Any, list[int]], key: Any, before: int, after: int) -> None:
    """Add one (n, tokens_in, tokens_out) observation to a staged shape table.

    Shared by the two `record_*_shape` entry points, which differ only in what
    they key on. Assumes the caller holds `_staged_lock`.
    """
    counts = store.get(key)
    if counts is None:
        if len(store) >= MAX_SHAPE_ROWS:
            return
        counts = [0, 0, 0]
        store[key] = counts
    counts[0] += 1
    counts[1] += before
    counts[2] += after


def _clean_pair(original_tokens: Any, compressed_tokens: Any) -> tuple[int, int] | None:
    """Validate one before/after token pair, or None if it is not usable.

    Clamped exactly as `record_compression` clamps: a compressor that emits
    more than it received is a bug, and letting `out` exceed `in` would surface
    downstream as negative savings rather than as the bug it is.
    """
    try:
        before = int(original_tokens or 0)
        after = int(compressed_tokens or 0)
    except (TypeError, ValueError):
        return None
    if before <= 0:
        return None
    return before, min(max(after, 0), before)


def record_content_shapes(observations: Iterable[tuple[str, str, int, int]]) -> None:
    """Beacon entry point for a batch of routing decisions from one compress().

    Batched, not one call per decision, because `_staged_lock` is also held by
    `record_compression` on the same executor thread and the note there is
    explicit that the contention is real: the router observes once per content
    section per request. Taking the lock once per compress() rather than once
    per section keeps this addition off that critical path.

    Each observation is ``(content_type, strategy, original_tokens,
    compressed_tokens)``. Both strings are enum values in code -- `ContentType`
    has nine members and `CompressionStrategy` a handful -- but both arrive as
    free strings, so both are slugged before they can reach the wire.
    """
    from headroom.telemetry.beacon import is_beacon_enabled

    if not is_beacon_enabled():
        return
    cleaned: list[tuple[tuple[str, str], int, int]] = []
    for content_type, strategy, original_tokens, compressed_tokens in observations:
        pair = _clean_pair(original_tokens, compressed_tokens)
        if pair is None:
            continue
        cleaned.append(((_safe_slug(content_type), _safe_slug(strategy)), *pair))
    if not cleaned:
        return
    with _staged_lock:
        for key, before, after in cleaned:
            _stage(_staged_shapes, key, before, after)


def record_content_shape(
    content_type: str, strategy: str, original_tokens: int, compressed_tokens: int
) -> None:
    """Beacon entry point for one routing decision, keyed by what was routed.

    `record_compression` already reports what each strategy yielded. This adds
    the term that was missing: what it was handed. Without a content axis the
    corpus can rank compressors against each other but cannot say which one to
    pick for a given input, which is the decision the router actually makes.

    Both arguments are enum values in code — `ContentType` has nine members and
    `CompressionStrategy` a handful — but both arrive here as free strings, so
    both are slugged before they can reach the wire.

    Single-observation form of :func:`record_content_shapes`, which is what the
    router actually calls. Same discipline as everything else in this module:
    off by default, cheap when off, never raises, and never touches the
    aggregator lock.
    """
    record_content_shapes(((content_type, strategy, original_tokens, compressed_tokens),))


def record_tool_shape(signature: Any, original_tokens: int, compressed_tokens: int) -> None:
    """Beacon entry point for one tool-output compression, keyed by structure.

    TOIN's docstring has always promised a network effect -- "more users ->
    more compression events -> better aggregated fields" -- and there has never
    been a transport, so every install learns alone and `import_patterns` has
    no source to import from. This is that transport, minus the part that
    cannot safely leave a machine.

    What ships is the STRUCTURAL DESCRIPTOR only: a bucketed field count, a
    nesting depth, and the boolean shape flags `ToolSignature` documents as
    "pattern indicators (without revealing actual field names)". What does not
    ship is `structure_hash`, which is a SHA256 over sorted field names and
    types -- a function of the tool's data, and the same objection that
    (rightly) kept `conversation_key_from_body` off the wire applies to it
    unchanged. Nor do `field_retrieval_frequency` or `common_query_patterns`,
    the two `ToolPattern` fields that carry real content.

    A descriptor also needs no k-anonymity machinery to be safe, because it is
    not invertible in the first place: thousands of distinct tools collapse
    into a few dozen shape classes, which is exactly the clustering the
    recommendations want anyway.

    `signature` is read through `getattr` and never required to be a real
    `ToolSignature`, so this cannot break a caller that passes something else.
    """
    from headroom.telemetry.beacon import is_beacon_enabled

    if not is_beacon_enabled():
        return
    pair = _clean_pair(original_tokens, compressed_tokens)
    if pair is None:
        return
    key = _tool_shape_key(signature)
    if key is None:
        return
    with _staged_lock:
        _stage(_staged_tool_shapes, key, *pair)


# Shape flags, in a fixed order so one structure always produces one key.
# Letters, not the field names, and distinct from each other: `status` takes
# `u` because `s` is already `score`.
_SHAPE_FLAGS = (
    ("has_arrays", "a"),
    ("has_nested_objects", "n"),
    ("has_id_like_field", "i"),
    ("has_score_like_field", "s"),
    ("has_timestamp_like_field", "t"),
    ("has_status_like_field", "u"),
    ("has_error_like_field", "e"),
    ("has_message_like_field", "m"),
)


def _tool_shape_key(signature: Any) -> str | None:
    """Structural descriptor for one tool output, as a bounded slug.

    Shaped like ``f16d3_anit``: field count rounded DOWN to a power of two,
    nesting depth, then the shape flags that are set. Rounding the field count
    is what makes the key a class rather than a fingerprint -- 17 fields and 31
    fields are the same shape for every purpose this serves.

    Returns None when the object is not a signature, rather than a key built
    from defaults, so a bad caller contributes nothing instead of contributing
    noise.
    """
    try:
        fields = int(getattr(signature, "field_count", 0) or 0)
        depth = int(getattr(signature, "max_depth", 0) or 0)
    except (TypeError, ValueError):
        return None
    if fields <= 0:
        return None
    bucket = min(1 << (fields.bit_length() - 1), 64)
    flags = "".join(
        letter for attribute, letter in _SHAPE_FLAGS if getattr(signature, attribute, False)
    )
    key = f"f{bucket}d{min(max(depth, 0), 9)}" + (f"_{flags}" if flags else "")
    # Belt and braces: the key is built from integers and literals above, so it
    # already matches, but nothing reaches the wire without passing the same
    # validator every other slug does.
    return _safe_slug(key)


def record_stack(slug: str) -> None:
    """Beacon entry point for one request's stack slug.

    The harness identity lives in the ``X-Headroom-Stack`` header, which only
    the proxy sees, and per request rather than per process. Counting slugs
    here lets :func:`resource_attributes` answer ``detect_stack``'s by_stack
    branch without the telemetry package importing ``headroom.proxy``.

    Without this the beacon can only read the two environment variables, so
    every install that points an agent at a persistent proxy — the common
    deployment for Claude Code, Cursor, Codex and the adapters — reports the
    literal ``"proxy"`` and the fleet is unsegmentable by agent.
    """
    from headroom.telemetry.beacon import is_beacon_enabled

    if not is_beacon_enabled():
        return
    # normalize_stack is the same chokepoint the proxy applies at ingress; an
    # unbounded header value must not reach the wire or grow this dict.
    from headroom.telemetry.context import normalize_stack

    clean = normalize_stack(slug)
    if not clean:
        return
    with _staged_lock:
        if clean not in _staged_stacks and len(_staged_stacks) >= MAX_STRATEGIES:
            return
        _staged_stacks[clean] = _staged_stacks.get(clean, 0) + 1


class BeaconCompressionObserver:
    """A `CompressionObserver` that forwards to the beacon and nothing else.

    The proxy's `PrometheusMetrics` is already an observer and forwards from
    there, so this is for the paths that never had one: the MCP servers, the
    bare transform pipeline, and the LangChain/Strands integrations. Those
    processes report `tokens.saved` either way, so without this they emit
    sessions with real token totals and an empty `by_strategy` — a silently
    biased subset that cannot be reconciled with the fleet totals.

    Only `record_compression` is implemented. ContentRouter's two other
    observer hooks (`record_kompress_size_gate`, `record_router_route_counts`)
    are each individually guarded at the call site, and both feed `/stats`
    rather than the beacon.
    """

    __slots__ = ()

    def record_compression(
        self, strategy: str, original_tokens: int, compressed_tokens: int
    ) -> None:
        record_compression(strategy, original_tokens, compressed_tokens)


def _drain_staged_strategies() -> dict[str, list[int]]:
    """Take everything staged since the last drain. Caller merges it."""
    with _staged_lock:
        if not _staged_strategies:
            return {}
        drained = {name: counts[:] for name, counts in _staged_strategies.items()}
        _staged_strategies.clear()
        return drained


def _drain_staged_shapes() -> dict[tuple[str, str], list[int]]:
    """Take every staged (content_type, strategy) observation."""
    with _staged_lock:
        if not _staged_shapes:
            return {}
        drained = {key: counts[:] for key, counts in _staged_shapes.items()}
        _staged_shapes.clear()
        return drained


def _drain_staged_tool_shapes() -> dict[str, list[int]]:
    """Take every staged tool-structure observation."""
    with _staged_lock:
        if not _staged_tool_shapes:
            return {}
        drained = {key: counts[:] for key, counts in _staged_tool_shapes.items()}
        _staged_tool_shapes.clear()
        return drained


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
    # ...and the all-layers rates are the ones that do count them: 1300 saved
    # of 2000 sent, 1300 of 1400 attempted. Both denominators grow too.
    assert r["rates"]["all_layers_saved_pct"] == 65.0, r["rates"]
    assert r["rates"]["all_layers_yield_pct"] == 92.86, r["rates"]
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
    assert event["failure_statuses"] == {}

    # The new burst is a distinct session, not a continuation.
    agg.flush_all()
    assert len(emitted) == 2, emitted
    assert emitted[1]["session"]["turns"] == 1

    # --- per-strategy compression -----------------------------------------
    # Compression runs on the executor thread before its request's outcome
    # arrives, so events are staged and drained by the next outcome. That is
    # what keeps the first turn's numbers while letting only an outcome open a
    # session. `_staged_*` is module state, so clear it between cases.
    _staged_strategies.clear()
    _staged_stacks.clear()

    strat: list[dict[str, Any]] = []
    sa = SessionAggregator(strat.append, idle_s=10.0)
    record_compression("smart_crusher", 1000, 400)
    record_compression("smart_crusher", 500, 300)
    record_compression("code_aware", 800, 800)
    assert sa._current is None, "a compression event must not open a session"
    sa.record(FakeOutcome(), now=2000.0)
    sa.flush_all()
    by = {row["strategy"]: row for row in strat[-1]["compression"]["by_strategy"]}
    assert by["smart_crusher"] == {
        "strategy": "smart_crusher",
        "n": 2,
        "tokens_in": 1500,
        "tokens_out": 700,
    }, by
    # A strategy that ran and saved nothing must still appear: "ran 800 tokens
    # through and removed none" is the finding, and dropping it would make
    # every strategy look effective.
    assert by["code_aware"]["tokens_in"] == by["code_aware"]["tokens_out"] == 800, by
    assert strat[-1]["session"]["turns"] == 1, "compression events are not turns"
    # A list of records, not an object keyed by strategy: the type must not
    # change as strategies are added. See the note in payload().
    assert isinstance(strat[-1]["compression"]["by_strategy"], list)
    assert [r["strategy"] for r in strat[-1]["compression"]["by_strategy"]] == sorted(
        r["strategy"] for r in strat[-1]["compression"]["by_strategy"]
    ), "sorted so heartbeats are byte-comparable"

    # Draining is exhaustive: a second session must not re-count the first
    # session's events.
    assert not _staged_strategies, "record() drains everything it staged"
    again: list[dict[str, Any]] = []
    sb = SessionAggregator(again.append, idle_s=10.0)
    sb.record(FakeOutcome(), now=3000.0)
    sb.flush_all()
    assert again[-1]["compression"]["by_strategy"] == [], again[-1]

    # An abandoned request — compression ran, the outcome never arrived — must
    # not invent a session. Before staging, this emitted a phantom turns=0 row
    # with all-zero tokens that inflated fleet session and install counts.
    ghost: list[dict[str, Any]] = []
    sc = SessionAggregator(ghost.append, idle_s=10.0)
    record_compression("smart_crusher", 900, 100)
    sc.flush_all()
    assert ghost == [], "no outcome, no session"
    _staged_strategies.clear()

    # Over the cardinality cap, extra strategies are dropped rather than
    # allowed to grow the payload without bound.
    cap: list[dict[str, Any]] = []
    cc = SessionAggregator(cap.append, idle_s=10.0)
    for i in range(MAX_STRATEGIES + 5):
        record_compression(f"s{i}", 100, 50)
    cc.record(FakeOutcome(), now=4000.0)
    cc.flush_all()
    assert len(cap[-1]["compression"]["by_strategy"]) == MAX_STRATEGIES, cap[-1]
    _staged_strategies.clear()

    # Strategy names are slugged, never passed through: the observer protocol
    # takes a free string and this is the only chokepoint before the wire.
    assert _safe_slug("smart_crusher") == "smart_crusher"
    assert _safe_slug("../../etc/passwd") == "other"

    # --- stack detection ---------------------------------------------------
    # Environment-only detection answers "proxy" for every install that points
    # an agent at a persistent proxy instead of using `headroom wrap` — i.e.
    # most of the fleet. The per-request slugs are the only real signal.
    _staged_stacks.clear()
    for _ in range(9):
        record_stack("wrap_claude")
    record_stack("wrap_cursor")
    assert resource_attributes()["headroom.stack"] == "wrap_claude", "dominant stack wins"
    _staged_stacks.clear()
    for _ in range(5):
        record_stack("wrap_claude")
    for _ in range(5):
        record_stack("wrap_cursor")
    assert resource_attributes()["headroom.stack"] == "mixed", "no dominant stack"
    _staged_stacks.clear()
    record_stack("../../etc/passwd")
    assert not _staged_stacks, "junk slugs never reach the wire"
    assert resource_attributes()["headroom.stack"] == "proxy", "falls back with no signal"

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

    class Broke(FakeOutcome):
        status_code = 500

    agg2 = SessionAggregator(emitted.append)
    agg2.record(Failed(), now=2000.0)
    agg2.record(Failed(), now=2001.0)
    agg2.record(Broke(), now=2002.0)
    agg2.flush_all()
    assert emitted[-1]["failures"] == 3
    # Provider load-shedding and our own 500s have to be separable, or the
    # count says "0.7% of turns failed" and nothing about whose fault it is.
    assert emitted[-1]["failure_statuses"] == {"529": 2, "500": 1}

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

    # --- schema v2 ---------------------------------------------------------
    # The full v2 surface is covered by tests/test_beacon_schema_v2.py. What
    # belongs here is the part that has to hold for the module on its own: the
    # additions are additive, and the two things they read that nobody had read
    # before — the environment, and the output_shaper stratum label — cannot
    # carry anything out with them.
    _staged_shapes.clear()
    _staged_tool_shapes.clear()

    class V2(FakeOutcome):
        original_tokens = 1000
        attempted_input_tokens = 400
        tokens_saved = 300
        ttfb_ms = 300.0
        num_messages = 30
        # Hyphenated, as `classify_client` really returns it.
        client = "claude-code"
        cache_write_5m_tokens = 80
        cache_write_1h_tokens = 20
        waste_signals = {"reread": 900, "reread_compressed": 400, "json_bloat": 50}
        # Carries a model family, a turn kind and a size bucket. Only the last
        # two may leave, and neither the label nor the family may.
        transforms_applied = ("output_shaper:stratum:sonnet|tool_result|l|tools", "crush")

    class V2Limited(V2):
        status_code = 429

    v2_out: list[dict[str, Any]] = []
    v2 = SessionAggregator(v2_out.append)
    v2.record(V2(), now=10_000.0)
    v2.record(V2Limited(), now=10_045.0)
    v2.record(V2(), now=10_090.0)
    v2.flush_all()
    row = v2_out[-1]

    # v1 is untouched by all of it. A 4xx in particular must not reach the
    # failure counters, which have been 5xx-only in every row of the corpus.
    assert row["failures"] == 0 and row["failure_statuses"] == {}, row
    assert row["errors"] == {"count": 1, "by_status": {"429": 1}}, row["errors"]
    assert row["compression"]["transforms"] == {"output_shaper": 3, "crush": 3}

    # The regret label: saved_pct alone cannot say whether compression cost the
    # agent anything, and this is the counter-pressure on it.
    assert row["quality"]["reread_compressed_tokens"] == 1200, row["quality"]
    assert row["quality"]["waste"] == [{"kind": "json_bloat", "tokens": 150}], (
        "waste kinds are allowlisted"
    )

    # Distributions ship their bucket edges, so a row is readable without
    # knowing which client version wrote it.
    assert row["hist"]["turn_tokens"]["edges"] == list(_HIST_EDGES["turn_tokens"])
    assert sum(row["hist"]["turn_tokens"]["counts"]) == 3
    assert len(row["hist"]["saved_pct"]["counts"]) == len(_HIST_EDGES["saved_pct"]) + 1

    # Cumulative like every v1 counter, so max(seq) still reconstructs the
    # session and a dropped heartbeat still costs nothing.
    assert sum(row["trajectory"]["turns"]) == row["session"]["turns"]
    assert row["trajectory"]["kinds"] == "s1e1s1", row["trajectory"]
    assert set(row["trajectory"]["kinds"]) - set("0123456789") <= _TURN_KINDS

    # The first turn has no predecessor, so the cache curve counts turns - 1.
    assert sum(row["cache"]["hits"]) + sum(row["cache"]["misses"]) == 2

    # Strata reach the wire as validated enums; the model family does not.
    # Hyphenated client ids must survive; _safe_slug alone would drop them.
    assert row["clients"] == [{"client": "claude_code", "n": 3}], row["clients"]
    # Lists of records, never objects keyed by a value that came from data —
    # an object's COLUMN TYPE in DuckDB is a function of which keys showed up.
    assert {"dim": "arm", "value": "treatment", "n": 3} in row["strata"], row["strata"]
    assert {"dim": "input_bucket", "value": "l", "n": 3} in row["strata"], row["strata"]
    assert isinstance(row["config"], list) and isinstance(row["quality"]["waste"], list)
    assert "sonnet" not in json.dumps(row), "the stratum label leaked a model family"

    # Structural descriptor only, never TOIN's structure_hash.
    assert _tool_shape_key(object()) is None
    assert (
        _tool_shape_key(
            type(
                "Sig",
                (),
                {"field_count": 17, "max_depth": 3, "has_arrays": True, "has_id_like_field": True},
            )()
        )
        == "f16d3_ai"
    )

    # The environment is read through an allowlist, and every value that gets
    # past it is slugged — so a variable holding a path or a URL reports
    # "other" rather than its contents.
    _cfg_prev = {k: os.environ.get(k) for k in ("HEADROOM_MODE", "HEADROOM_KOMPRESS_ENDPOINT")}
    try:
        os.environ["HEADROOM_MODE"] = "/Users/someone/private"
        os.environ["HEADROOM_KOMPRESS_ENDPOINT"] = "https://secret.internal/v1"
        snapshot = _config_snapshot()
        assert snapshot.get("mode") == "other", snapshot
        assert "kompress_endpoint" not in snapshot, snapshot
    finally:
        for _k, _v in _cfg_prev.items():
            if _v is None:
                os.environ.pop(_k, None)
            else:
                os.environ[_k] = _v

    _staged_shapes.clear()
    _staged_tool_shapes.clear()

    print("ok")


if __name__ == "__main__":
    demo()
