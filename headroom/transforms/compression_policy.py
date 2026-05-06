"""Per-auth-mode compression policy — Phase F PR-F2.1, extended in F2.2 (Python parity).

Hand-mirrored port of `headroom_core::compression_policy::CompressionPolicy`
(Rust). The Rust crate is the source of truth; this module exists so
the Python proxy's `TransformPipeline` (which still runs `CacheAligner`
and other detector-only transforms) can read the same per-mode flags
the Rust dispatcher does.

A parity test (`tests/test_compression_policy.py`) instantiates one of
each variant and asserts the field map matches what the Rust unit
tests assert. F2.2 should consider exposing the Rust struct via PyO3
to retire this hand-mirror — that's deliberately out of scope here so
F2.1/F2.2 can ship.

See `crates/headroom-core/src/compression_policy.rs` for the canonical
docstring (per-mode rationale, why-a-struct, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from headroom.proxy.auth_mode import AuthMode

# ── F2.2 per-mode default values (CONSERVATIVE pending bake telemetry) ──
# Mirrors the Rust ``pub(crate) const`` block in
# ``crates/headroom-core/src/compression_policy.rs``. Centralised here so
# a follow-up tune lands in one place per language. The Rust tests assert
# the values directly; the Python parity tests assert the *fields* match
# (not the values — that would double-pin against drift in a way that
# masks real divergence).
#
# Per the realignment build constraints (project memory
# ``feedback_realignment_build_constraints.md``): "configurable / no
# hardcoded values". The configuration *is* the per-mode default — we
# deliberately do NOT add a separate env var per field. Operators tune
# by editing these constants and shipping a new build, mirroring the
# Rust pattern exactly.

#: PAYG: aggressive — let volatile content noise up to ~128 tokens slip
#: before flagging. Higher than Subscription because PAYG users opt in
#: to aggressive compression.
_VOLATILE_TOKEN_THRESHOLD_PAYG: int = 128

#: Subscription: conservative — flag volatile content earlier (32
#: tokens) so cache prefixes stay stable.
_VOLATILE_TOKEN_THRESHOLD_SUBSCRIPTION: int = 32

#: PAYG: cap lossy compression at 45% of original tokens. Aggressive
#: but bounded — F2.1 had no cap (effectively ``1.0``), F2.2 introduces one.
_MAX_LOSSY_RATIO_PAYG: float = 0.45

#: Subscription: conservative cap at 25%. Cache stability over savings.
_MAX_LOSSY_RATIO_SUBSCRIPTION: float = 0.25


@dataclass(frozen=True, slots=True)
class CompressionPolicy:
    """Per-auth-mode policy that downstream compression stages consult.

    Five fields after F2.2:

    - ``live_zone_only`` (F2.1) — gate for the Python TransformPipeline
      to skip pre-cache-marker mutation.
    - ``cache_aligner_enabled`` (F2.1) — gate for the Python
      ``CacheAligner`` transform.
    - ``volatile_token_threshold`` (F2.2) — per-mode token threshold
      below which content is treated as cache-stable. Plumbed through
      the struct; no current detector consumes it (intentional — the
      detector refactor is a follow-up).
    - ``max_lossy_ratio`` (F2.2) — per-mode upper bound on lossy
      compression aggressiveness (fraction in ``[0.0, 1.0]``). Plumbed
      through the struct; no current compressor consumes it.
    - ``toin_read_only`` (F2.2) — TOIN learning gate. ``True`` =
      serve cached patterns but never write new observations from
      this request (Subscription).
    """

    live_zone_only: bool
    """When True, transforms MUST NOT modify bytes outside the post-
    cache-marker live zone. The Rust live-zone dispatcher is already
    live-zone-only by construction, so this flag is effectively a
    no-op on the Rust path; it exists for the Python TransformPipeline
    where transforms like CacheAligner inspect the cached prefix."""

    cache_aligner_enabled: bool
    """When False, the CacheAligner transform's `should_apply` MUST
    return False — that's the load-bearing F2.1 gate for the cache-
    instability complaints in #327 / #388."""

    volatile_token_threshold: int
    """F2.2: per-mode token-count threshold below which content is
    treated as cache-stable. Subscription is conservative
    (low → flag aggressively → keep prompts stable); PAYG aggressive
    (high → tolerate more volatile noise). Plumbed but unconsumed in
    F2.2 — the volatile detector in ``cache_aligner.py`` is shape-
    based, not token-count-based; wiring it is a follow-up."""

    max_lossy_ratio: float
    """F2.2: per-mode upper bound on lossy compression aggressiveness,
    expressed as the fraction of original tokens that may be dropped
    (``0.0`` = no lossy, ``1.0`` = unlimited). Subscription ``0.25``;
    PAYG ``0.45``. Plumbed but unconsumed in F2.2 — distinct from the
    caller-driven ``target_ratio`` kwarg in the Python ContentRouter."""

    toin_read_only: bool
    """F2.2: when True, TOIN serves cached recommendations but
    never writes new pattern observations from this request.
    Subscription True (consistency over learning); PAYG/OAuth False
    (network effect keeps growing). The gate is read by
    ``smart_crusher.py`` and ``content_router.py`` at the
    ``record_compression`` call sites."""


def policy_for_mode(mode: AuthMode) -> CompressionPolicy:
    """Resolve the F2.1+F2.2 policy for an auth mode.

    PAYG and OAuth are identical (aggressive: live-zone-not-only,
    cache-aligner on, relaxed thresholds, TOIN write-enabled).
    Subscription is the user-visible win: live-zone-only with cache
    aligner disabled, tighter thresholds, TOIN read-only.

    F2.2-followup may diverge OAuth from PAYG once telemetry is
    collected.
    """
    if mode == AuthMode.PAYG:
        return CompressionPolicy(
            live_zone_only=False,
            cache_aligner_enabled=True,
            volatile_token_threshold=_VOLATILE_TOKEN_THRESHOLD_PAYG,
            max_lossy_ratio=_MAX_LOSSY_RATIO_PAYG,
            toin_read_only=False,
        )
    if mode == AuthMode.OAUTH:
        # Identical to PAYG in F2.1/F2.2. The parity test in
        # ``tests/test_compression_policy.py`` is the canary that
        # catches a future divergence and forces a deliberate update
        # there + in the Rust crate.
        return CompressionPolicy(
            live_zone_only=False,
            cache_aligner_enabled=True,
            volatile_token_threshold=_VOLATILE_TOKEN_THRESHOLD_PAYG,
            max_lossy_ratio=_MAX_LOSSY_RATIO_PAYG,
            toin_read_only=False,
        )
    if mode == AuthMode.SUBSCRIPTION:
        return CompressionPolicy(
            live_zone_only=True,
            cache_aligner_enabled=False,
            volatile_token_threshold=_VOLATILE_TOKEN_THRESHOLD_SUBSCRIPTION,
            max_lossy_ratio=_MAX_LOSSY_RATIO_SUBSCRIPTION,
            toin_read_only=True,
        )
    raise ValueError(f"Unhandled AuthMode variant: {mode!r}")


def policy_default_payg() -> CompressionPolicy:
    """The PAYG-equivalent policy used when the
    ``HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT`` flag is disabled
    (default in F2.1 c1-c4; flipped to enabled in c5/5).

    Centralised so the proxy handlers do not duplicate the constant,
    and so a future change to PAYG semantics propagates to both the
    enforcement-on and enforcement-off paths.
    """
    return policy_for_mode(AuthMode.PAYG)


_ENFORCEMENT_ENV = "HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT"


def is_enforcement_enabled() -> bool:
    """Read the enforcement flag from the environment.

    Same env var the Rust proxy reads (``Config::auth_mode_policy_enforcement``)
    so the two paths stay in lockstep with one operator switch.
    Default (when unset): ``True`` from F2.1 c5/5 onward, matching
    the Rust default after the c5 flip.

    NOT cached — read every call so an operator can flip the env var
    in a hot-reload scenario. The cost is one ``dict.get`` per call,
    well below noise.
    """
    val = os.environ.get(_ENFORCEMENT_ENV, "enabled").strip().lower()
    # Same set of off-values the telemetry beacon honours
    # (`headroom/telemetry/beacon.py::_OFF_VALUES`) so operators don't
    # have to remember a different vocabulary per flag.
    return val not in ("disabled", "off", "false", "0", "no")


def resolve_policy(auth_mode: AuthMode | None) -> CompressionPolicy:
    """Resolve the effective ``CompressionPolicy`` for a request.

    - If the enforcement flag is off, returns PAYG-equivalent
      regardless of the classified auth mode.
    - If the enforcement flag is on and ``auth_mode`` is ``None``,
      returns PAYG-equivalent (defensive default for the unclassified
      / batch-row path).
    - Otherwise returns the per-mode policy.

    This is the single public entry point handlers should call when
    deriving the policy from a request's classification result.
    """
    if auth_mode is None or not is_enforcement_enabled():
        return policy_default_payg()
    return policy_for_mode(auth_mode)
