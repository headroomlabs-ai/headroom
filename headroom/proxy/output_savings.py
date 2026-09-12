"""Counterfactual estimation of output-token reduction.

The hard problem: output-token savings are **counterfactual**. When the shaper
makes a request terser, the model emits N output tokens — but we never observe
what it *would* have emitted unshaped. Input compression is a pure function, so
``tokens_before``/``tokens_after`` are both observable. Output is not: only one
side of the counterfactual happens per request. So a flat "we save 30%" claim
is marketing, not measurement.

This module makes the estimate honest by separating three tiers:

1. **Estimated (synthetic control).** A per-stratum baseline of unshaped output
   tokens — built by ``learn --verbosity`` from session history that predates
   the shaper — gives an expected output for each request's feature stratum.
   ``estimate = Σ (baseline_mean[stratum] − observed_output)`` over shaped
   requests, summed as **signed** deltas (never clamped per-request — clamping
   biases upward). Reported with a propagated confidence interval and always
   labelled an estimate, never "measured".

2. **Measured (A/B holdout).** When a small holdout fraction of conversations
   is left unshaped, the difference of per-stratum means between the treatment
   and control arms is an unbiased causal estimate. This is the only number we
   call "measured". Assignment is **conversation-stable** (a whole conversation
   is in one arm) for two reasons that happen to align: mixing shaped and
   unshaped turns within one conversation would (a) pollute the comparison and
   (b) bust the prefix cache by changing the system-prompt tail mid-stream.

3. **Direct waste (no counterfactual).** Echo ratio — n-gram overlap between a
   response and the context it was given — is a property of a single response,
   measurable with no counterfactual. "32% of output restated existing context"
   is an honest standalone fact and the shaper's target. See ``echo_ratio``.

Stratification uses only features observable at request time (never the output):
turn kind, input-token bucket, model family, whether tools are present.

Pure module: no I/O except explicit ``load``/``save``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .output_savings_policy import (
    assign_arm as assign_arm,
)
from .output_savings_policy import (
    conversation_key_from_body as conversation_key_from_body,
)
from .output_savings_policy import (
    conversation_key_from_responses_body as conversation_key_from_responses_body,
)
from .output_savings_policy import (
    conversation_label as conversation_label,
)
from .output_savings_policy import (
    input_bucket as input_bucket,
)
from .output_savings_policy import (
    model_family as model_family,
)
from .output_savings_policy import (
    parse_conversation_label,
    parse_stratum_label,
)
from .output_savings_policy import (
    stratum_key as stratum_key,
)
from .output_savings_policy import (
    stratum_label as stratum_label,
)

logger = logging.getLogger(__name__)

# A stratum enters the measured (A/B) estimate only once BOTH arms hold this
# many distinct conversations. Assignment is per conversation, so conversations
# -- not requests -- are the independent draws: one agent session in the holdout
# can leave 2,500 control requests in a single stratum, and every request-count
# gate we have waves that through as a well-sampled arm. On a real ledger that
# produced a -1.6% "measured" reduction whose two largest terms came from strata
# with four control requests apiece.
MEASURED_MIN_CLUSTERS = 5

# Distinct conversations tracked per arm/stratum. The count is only ever
# compared against the threshold above, so there is nothing to gain from an
# exact tally of a busy stratum -- and this keeps a flushed-every-25-requests
# ledger from growing a set per conversation forever.
_CLUSTER_CAP = 32


@dataclass
class _Accum:
    """Running count / sum / sum-of-squares for online mean & variance."""

    n: int = 0
    sum: float = 0.0
    sumsq: float = 0.0
    #: Distinct conversation ids behind ``qn``, capped at ``_CLUSTER_CAP``.
    clusters: set[str] = field(default_factory=set)
    #: The conversation-QUALIFIED subset of n / sum / sumsq: observations that
    #: arrived carrying a conversation label. Kept separate because the cluster
    #: count alone cannot vouch for the totals. An accumulator upgraded from an
    #: older ledger holds requests of unknown provenance -- possibly one
    #: conversation, possibly thousands -- and five fresh labelled
    #: conversations arriving afterwards would otherwise qualify all of that
    #: legacy traffic for the measured estimate too, which is exactly the case
    #: the cluster gate exists to exclude. Later unlabelled requests are
    #: likewise kept out of an already-qualified stratum. The full totals stay
    #: intact for the estimated / modelled tiers and historical reporting.
    qn: int = 0
    qsum: float = 0.0
    qsumsq: float = 0.0

    def add(self, x: float, cluster: str | None = None) -> None:
        self.n += 1
        self.sum += x
        self.sumsq += x * x
        if cluster is None:
            return
        self.qn += 1
        self.qsum += x
        self.qsumsq += x * x
        if len(self.clusters) < _CLUSTER_CAP:
            self.clusters.add(cluster)

    @property
    def n_clusters(self) -> int:
        """Distinct conversations observed, saturating at ``_CLUSTER_CAP``.

        0 for an accumulator written before conversations were tracked, which
        is why that data cannot clear :data:`MEASURED_MIN_CLUSTERS`: an
        unverifiable arm is treated as an unqualified one.
        """
        return len(self.clusters)

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n else 0.0

    @property
    def var(self) -> float:
        """Sample variance (unbiased). 0 when fewer than 2 observations."""
        if self.n < 2:
            return 0.0
        return max(0.0, (self.sumsq - self.sum * self.sum / self.n) / (self.n - 1))

    @property
    def qmean(self) -> float:
        """Mean over the conversation-qualified observations only."""
        return self.qsum / self.qn if self.qn else 0.0

    @property
    def qvar(self) -> float:
        """Sample variance over the conversation-qualified observations only."""
        if self.qn < 2:
            return 0.0
        return max(0.0, (self.qsumsq - self.qsum * self.qsum / self.qn) / (self.qn - 1))

    def merge(self, other: _Accum) -> None:
        """Fold another accumulator's observations into this one.

        n / sum / sumsq are additive, so merging is element-wise addition and
        is exactly equivalent to having ``add``-ed both observation streams.
        """
        self.n += other.n
        self.sum += other.sum
        self.sumsq += other.sumsq
        self.qn += other.qn
        self.qsum += other.qsum
        self.qsumsq += other.qsumsq
        self.clusters |= set(list(other.clusters)[: _CLUSTER_CAP - len(self.clusters)])

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"n": self.n, "sum": self.sum, "sumsq": self.sumsq}
        # Omitted when empty so a baseline model (which has no conversations to
        # track) serializes exactly as it did before.
        if self.clusters:
            d["clusters"] = sorted(self.clusters)
        if self.qn:
            d["qn"] = self.qn
            d["qsum"] = self.qsum
            d["qsumsq"] = self.qsumsq
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _Accum:
        a = cls()
        a.n = int(d.get("n", 0))
        a.sum = float(d.get("sum", 0.0))
        a.sumsq = float(d.get("sumsq", 0.0))
        a.clusters = {str(c) for c in (d.get("clusters") or ())}
        # Absent in a pre-upgrade ledger, which is the point: those requests
        # carry no conversation provenance and stay out of the measured arm.
        a.qn = int(d.get("qn", 0))
        a.qsum = float(d.get("qsum", 0.0))
        a.qsumsq = float(d.get("qsumsq", 0.0))
        return a


@dataclass
class BaselineModel:
    """Per-stratum baseline of unshaped output tokens (the synthetic control).

    Built offline by ``learn --verbosity`` from pre-shaper history. ``strata``
    maps a stratum key to its accumulator; ``glob`` is the all-requests
    fallback for strata never seen during training.
    """

    strata: dict[str, _Accum] = field(default_factory=dict)
    glob: _Accum = field(default_factory=_Accum)

    def observe(self, key: str, output_tokens: int) -> None:
        self.strata.setdefault(key, _Accum()).add(output_tokens)
        self.glob.add(output_tokens)

    def merge(self, other: BaselineModel) -> None:
        """Fold another baseline's observations into this one.

        Per-stratum and global accumulators are additive, so merging is
        element-wise and order-independent — the result is identical to having
        observed both corpora against a single model. Used to aggregate a
        cross-project baseline from per-project ``analyze`` results without
        re-reading transcripts.
        """
        for key, acc in other.strata.items():
            self.strata.setdefault(key, _Accum()).merge(acc)
        self.glob.merge(other.glob)

    def lookup(self, key: str) -> tuple[float, float, int]:
        """Return ``(mean, var, n)`` for *key* with hierarchical back-off.

        Falls back by trimming trailing (least-specific) stratum fields, then
        to the global mean. Back-off keeps the estimate defined for strata the
        baseline never saw, at the cost of specificity.
        """
        acc = self.strata.get(key)
        if acc is not None and acc.n > 0:
            return acc.mean, acc.var, acc.n
        parts = key.split("|")
        while len(parts) > 1:
            parts = parts[:-1]
            prefix = "|".join(parts)
            for k, a in self.strata.items():
                if k.startswith(prefix + "|") and a.n > 0:
                    return a.mean, a.var, a.n
        return self.glob.mean, self.glob.var, self.glob.n

    def to_dict(self) -> dict[str, Any]:
        return {
            "strata": {k: a.to_dict() for k, a in self.strata.items()},
            "glob": self.glob.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BaselineModel:
        m = cls()
        for k, a in (d.get("strata") or {}).items():
            m.strata[k] = _Accum.from_dict(a)
        m.glob = _Accum.from_dict(d.get("glob") or {})
        return m

    @property
    def total_samples(self) -> int:
        return self.glob.n


# Benchmark-derived reduction factors, consulted ONLY when a deployment has
# neither a holdout nor a learned baseline.
#
# THIS TABLE SHIPS EMPTY, AND THAT IS THE INTENDED OPEN-SOURCE BEHAVIOUR.
# Headroom can apply verbosity steering out of the box -- set
# ``HEADROOM_VERBOSITY_LEVEL`` and the tokens are really saved. What it cannot
# do out of the box is tell you HOW MUCH it saved without measuring your own
# traffic, because a credible factor is not a constant: it depends on the model
# family, the shape of the turn, and the exact steering text, and producing one
# means running a paired benchmark across models and paying for both arms.
#
# Two ways to populate it, in order of strength:
#
#   1. Run a holdout. ``SavingsLedger.estimate_from_holdout`` measures YOUR
#      traffic and outranks anything here -- see :meth:`best_estimate`. This is
#      the honest answer and it needs no factor table at all.
#   2. Register factors from a benchmark, via :func:`register_modelled_factors`.
#      An extension that has done the measurement can install them at startup.
#
# An empty table means :meth:`estimate_from_model` returns ``None`` for every
# level, so the dashboard shows a dash rather than a number nobody measured.
# A dash is the correct rendering of "not measured"; an invented constant is
# not, and would be the one failure mode this module exists to prevent.
MODELLED_REDUCTION: dict[int, tuple[float, float]] = {}


def register_modelled_factors(level: int, conservative: float, optimistic: float) -> None:
    """Install benchmark-derived reduction factors for one verbosity level.

    Extension seam. ``conservative`` and ``optimistic`` are fractions in
    ``(0, 1)`` -- the low and high ends of the measured reduction, where the
    low end becomes the headline so the number under-reports rather than
    flatters.

    Registering a level twice replaces it, so an extension may refresh factors
    after a re-measurement. Values outside ``(0, 1)`` are rejected: the
    estimator inverts them as ``r/(1-r)``, which is nonsense at 0 and divides
    by zero at 1.
    """
    if not 0.0 < conservative < 1.0 or not 0.0 < optimistic < 1.0:
        raise ValueError(
            f"reduction factors must lie in (0, 1); got ({conservative}, {optimistic})"
        )
    if conservative > optimistic:
        raise ValueError(f"conservative factor {conservative} exceeds optimistic {optimistic}")
    MODELLED_REDUCTION[level] = (conservative, optimistic)


@dataclass
class SavingsEstimate:
    """Result of an estimation pass."""

    tokens_saved: float
    baseline_tokens: float
    pct: float
    ci_low_pct: float
    ci_high_pct: float
    n_requests: int
    # "measured" (A/B holdout) > "estimated" (synthetic control) >
    # "modelled" (benchmark default factor -- not this deployment's traffic)
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Emitted by ``output_shaper.shape_request`` only when it actually changed the
# request, so its presence is the per-request proof that shaping happened.
_SHAPED_LABEL_PREFIX = "output_shaper:verbosity:"


@dataclass
class SavingsLedger:
    """Accumulates shaped (treatment) and unshaped (control) observations and
    produces honest reduction estimates.

    ``baseline`` is the offline synthetic control. ``treatment``/``control``
    are live per-stratum accumulators of observed output tokens, used both for
    the A/B "measured" number (when a holdout exists) and to keep the ledger
    self-describing.
    """

    baseline: BaselineModel = field(default_factory=BaselineModel)
    treatment: dict[str, _Accum] = field(default_factory=dict)
    control: dict[str, _Accum] = field(default_factory=dict)

    # ---- recording -------------------------------------------------------

    def record(
        self, arm: str, key: str, output_tokens: int, conversation: str | None = None
    ) -> None:
        target = self.treatment if arm == "treatment" else self.control
        target.setdefault(key, _Accum()).add(output_tokens, conversation)

    # ---- estimation ------------------------------------------------------

    def estimate_from_baseline(self) -> SavingsEstimate:
        """Synthetic-control estimate: treatment output vs. offline baseline.

        Aggregate signed delta ``Σ_s n_s·(μ_s − ȳ_s)`` where μ_s is the
        baseline mean and ȳ_s the observed treatment mean. Variance propagates
        both the observed-output spread and the finite-baseline-sample error:

            Var ≈ Σ_s [ n_s·σ²_y,s  +  n_s²·σ²_μ,s / m_s ]
        """
        total_saved = 0.0
        total_baseline = 0.0
        var = 0.0
        n_requests = 0
        for key, acc in self.treatment.items():
            if acc.n == 0:
                continue
            mu, mu_var, m = self.baseline.lookup(key)
            if m == 0:
                continue
            n = acc.n
            n_requests += n
            total_saved += n * (mu - acc.mean)
            total_baseline += n * mu
            var += n * acc.var
            if m > 0:
                var += (n * n) * (mu_var / m)
        return self._finalize(total_saved, total_baseline, var, n_requests, "estimated")

    def estimate_from_holdout(self) -> SavingsEstimate | None:
        """A/B measurement: per-stratum control mean minus treatment mean.

        Only strata with conversation-labelled data in BOTH arms contribute,
        and only once both arms hold :data:`MEASURED_MIN_CLUSTERS` distinct
        conversations. Returns ``None`` if no such stratum exists (no holdout
        traffic yet, or none of it spread across enough conversations).
        Weighted by treatment volume; this is the unbiased causal number.

        The number is built from the QUALIFIED subset of each arm, never the
        arm totals: an upgraded ledger's legacy requests have no conversation
        provenance, so five fresh conversations arriving afterwards must not
        drag thousands of unattributable requests into the measurement with
        them. The totals remain available to the estimated and modelled tiers.

        The cluster gate is not a sample-size nicety. Assignment is
        conversation-stable, so the requests inside one conversation are one
        draw answering one question, and the variance below (which divides by
        the REQUEST count) reads a single 2,500-request session as a precise
        measurement. Strata that thin get excluded rather than down-weighted:
        the arm they describe is one conversation's worth of work, and no
        weighting recovers a comparison that was never made.
        """
        total_saved = 0.0
        total_baseline = 0.0
        var = 0.0
        n_requests = 0
        contributing = 0
        for key, t in self.treatment.items():
            c = self.control.get(key)
            if c is None or c.qn == 0 or t.qn == 0:
                continue
            if c.n_clusters < MEASURED_MIN_CLUSTERS or t.n_clusters < MEASURED_MIN_CLUSTERS:
                continue
            contributing += 1
            # Everything below reads the qualified subset only. The clusters
            # vouch for those observations and for nothing else.
            n = t.qn
            n_requests += n
            delta = c.qmean - t.qmean  # tokens saved per request in this stratum
            total_saved += n * delta
            total_baseline += n * c.qmean
            # Var of (c.mean - t.mean) = σ²_c/n_c + σ²_t/n_t, scaled by n².
            var += (n * n) * (c.qvar / c.qn + t.qvar / t.qn)
        if contributing == 0:
            return None
        return self._finalize(total_saved, total_baseline, var, n_requests, "measured")

    @staticmethod
    def _finalize(
        total_saved: float,
        total_baseline: float,
        var: float,
        n_requests: int,
        kind: str,
    ) -> SavingsEstimate:
        pct = (total_saved / total_baseline * 100.0) if total_baseline > 0 else 0.0
        se = math.sqrt(var)
        # 95% normal-approx band on the token total, converted to percent.
        lo = total_saved - 1.96 * se
        hi = total_saved + 1.96 * se
        ci_low = (lo / total_baseline * 100.0) if total_baseline > 0 else 0.0
        ci_high = (hi / total_baseline * 100.0) if total_baseline > 0 else 0.0
        return SavingsEstimate(
            tokens_saved=total_saved,
            baseline_tokens=total_baseline,
            pct=pct,
            ci_low_pct=ci_low,
            ci_high_pct=ci_high,
            n_requests=n_requests,
            kind=kind,
        )

    def estimate_from_model(self, level: int) -> SavingsEstimate | None:
        """Weakest tier: apply a benchmark factor to observed treatment output.

        Used only when this deployment has produced no counterfactual of its
        own. Returns ``None`` for a level that was never benchmarked, so an
        unmeasured level shows nothing rather than a guess.

        The arithmetic is the part worth getting right. Observed output is
        already POST-shaping, so the saving is not ``observed x r``. If the
        unshaped response would have been ``U`` and we observed
        ``O = U(1-r)``, then ``saved = U - O = O * r/(1-r)``. At r=0.20 that is
        0.25 of observed, not 0.20 -- the naive form understates, and by more
        as r grows.
        """
        factors = MODELLED_REDUCTION.get(level)
        if factors is None:
            return None
        observed = 0.0
        n_requests = 0
        for acc in self.treatment.values():
            if acc.n == 0:
                continue
            observed += acc.n * acc.mean
            n_requests += acc.n
        if n_requests == 0 or observed <= 0:
            return None

        def saved_for(r: float) -> float:
            return observed * r / (1.0 - r)

        lo_r, hi_r = factors
        saved = saved_for(lo_r)
        baseline = observed + saved
        # The band is the spread between the two benchmarked models, NOT a
        # sampling CI -- there is no sample here. Callers must not label it
        # "95% CI"; the dashboard branches on kind for exactly this reason.
        lo_pct = lo_r * 100.0
        hi_pct = hi_r * 100.0
        return SavingsEstimate(
            tokens_saved=saved,
            baseline_tokens=baseline,
            pct=lo_pct,
            ci_low_pct=lo_pct,
            ci_high_pct=hi_pct,
            n_requests=n_requests,
            kind="modelled",
        )

    def best_estimate(self, level: int | None = None) -> SavingsEstimate:
        """Strongest available tier: measured > estimated > modelled.

        ``level`` enables the modelled fallback; without it the behaviour is
        unchanged from before, which keeps every existing caller honest.
        """
        measured = self.estimate_from_holdout()
        if measured is not None:
            return measured
        estimated = self.estimate_from_baseline()
        if estimated.n_requests > 0:
            return estimated
        if level is not None:
            modelled = self.estimate_from_model(level)
            if modelled is not None:
                return modelled
        return estimated

    # ---- persistence -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            # Marks arms accumulated under the shaped-only recording rule (see
            # ``record_from_labels``). Absent = written before that rule, so the
            # arms may hold unshaped observations.
            "shaped_only": True,
            "baseline": self.baseline.to_dict(),
            "treatment": {k: a.to_dict() for k, a in self.treatment.items()},
            "control": {k: a.to_dict() for k, a in self.control.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SavingsLedger:
        ledger = cls(baseline=BaselineModel.from_dict(d.get("baseline") or {}))
        if not d.get("shaped_only"):
            # Pre-rule arms cannot be told apart from shaped ones entry by
            # entry, and republishing them is the reported bug. Drop them and
            # re-accumulate from live traffic (hours, not weeks). The offline
            # baseline is kept: it is learned from pre-shaper history, costs a
            # `learn --verbosity` run to rebuild, and was never the poisoned part.
            if d.get("treatment") or d.get("control"):
                logger.warning(
                    "output-savings ledger predates shaped-only recording; "
                    "dropping %d treatment and %d control strata and "
                    "re-accumulating (baseline kept)",
                    len(d.get("treatment") or {}),
                    len(d.get("control") or {}),
                )
            return ledger
        for k, a in (d.get("treatment") or {}).items():
            ledger.treatment[k] = _Accum.from_dict(a)
        for k, a in (d.get("control") or {}).items():
            ledger.control[k] = _Accum.from_dict(a)
        return ledger

    def save(self, path: Any) -> None:
        from pathlib import Path

        from headroom import fsutil

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # fsutil.write_text is atomic (temp file + os.replace), so a crash
        # mid-write cannot truncate the ledger already on disk (#18).
        fsutil.write_text(p, json.dumps(self.to_dict(), separators=(",", ":")))

    @classmethod
    def load(cls, path: Any) -> SavingsLedger:
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            # Fail open (empty ledger), but surface the loss — silently
            # swallowing a corrupt file made lost history indistinguishable
            # from no history yet (#18).
            logger.warning("output-savings ledger %s unreadable, starting empty: %s", p, exc)
            return cls()


# --------------------------------------------------------------------------
# Live recording — rides the existing ``transforms_applied`` label channel so
# every response path (streaming, non-streaming, backend) feeds the ledger with
# no changes to RequestOutcome or its construction sites.
# --------------------------------------------------------------------------


class SavingsRecorder:
    """In-memory ledger with periodic flush, safe for concurrent requests.

    Loads the baseline (written by ``learn --verbosity``) from disk, accumulates
    live treatment/control observations in memory, and flushes every
    ``flush_every`` records so a busy proxy doesn't do a read-modify-write of the
    JSON file on every request.
    """

    def __init__(self, path: Any, flush_every: int = 25) -> None:
        import threading
        from pathlib import Path

        self._path = Path(path)
        self._lock = threading.Lock()
        self._ledger = SavingsLedger.load(self._path)
        self._flush_every = flush_every
        self._since_flush = 0

    def record_from_labels(self, labels: Any, output_tokens: int) -> bool:
        """Record one outcome given its transforms_applied labels. Returns True
        if a shaping label was found and recorded.

        The conversation label may sit either side of the stratum label, so the
        labels are scanned once for both before recording. A request that
        carries no conversation label (an older client, or a path that has not
        adopted it) still records its output tokens; it just does not advance
        the stratum's cluster count.
        """
        label_strings = tuple(str(label) for label in labels or ())
        arm_key: tuple[str, str] | None = None
        conversation: str | None = None
        for label in label_strings:
            text = str(label)
            if arm_key is None:
                arm_key = parse_stratum_label(text)
                if arm_key is not None:
                    continue
            if conversation is None:
                conversation = parse_conversation_label(text)
        if arm_key is None:
            return False
        arm, key = arm_key
        if arm == "treatment" and not any(
            str(label).startswith(_SHAPED_LABEL_PREFIX) for label in label_strings
        ):
            return False
        with self._lock:
            self._ledger.record(arm, key, output_tokens, conversation)
            self._since_flush += 1
            if self._since_flush >= self._flush_every:
                self._flush_locked()
        return True

    def estimate_request_savings(self, labels: Any, output_tokens: int) -> int:
        """Per-request output tokens saved, for the savings rollup.

        For a treatment request, the synthetic-control estimate
        ``baseline_mean(stratum) - output_tokens``; 0 for control, unknown
        strata, or when no shaping label is present. Read-only: unlike
        ``record_from_labels`` it does not mutate the ledger, so the two
        compose without double-counting.

        The delta is **clamped at zero here**, which is deliberate and is not
        in tension with this module's "never clamped per-request" rule. That
        rule governs the tier-1 estimate in :meth:`estimate`, which sums signed
        deltas (``total_saved += n * (mu - acc.mean)``) precisely so that
        chattier-than-baseline turns pull the headline down. This method feeds
        something else: the per-request savings rollup on ``RequestOutcome``,
        which flows to Prometheus and the savings ledger. Those are
        accumulate-only surfaces — both consumers floor it again
        (``savings_tracker`` at the ``estimate_request_savings_usd`` and
        ``record_request`` boundaries), and ``prometheus_metrics`` clamps a
        negative ``tokens_saved`` with an "artifact" log for the same reason.
        Keeping the floor here makes that boundary explicit rather than
        relying on every downstream caller to reapply it."""
        for label in labels or ():
            parsed = parse_stratum_label(str(label))
            if parsed is None:
                continue
            arm, key = parsed
            if arm != "treatment":
                return 0
            with self._lock:
                mean, _var, n = self._ledger.baseline.lookup(key)
            return max(0, int(round(mean - output_tokens))) if n > 0 else 0
        return 0

    def _reload_baseline_locked(self) -> None:
        """Adopt the on-disk baseline written by ``learn --verbosity --apply``.

        ``learn`` rewrites the baseline in place in the same file a running proxy
        holds open, while the recorder only ever appends treatment/control
        samples and never touches the baseline. Without re-reading it, two things
        break: (1) a baseline learned while the proxy is up never takes effect
        until a restart, so treatment lookups all miss (``m == 0``) and the
        output-reduction tile stays at "—"; and (2) our periodic flush would
        write our in-memory (empty) baseline straight over the one ``learn`` just
        persisted.

        Adopt the disk baseline whenever it carries samples and differs from
        ours. Comparing content (not just sample count) means a re-learn with the
        same number of samples still takes effect, and the empty-disk guard keeps
        a truncated file from wiping a baseline we already hold."""
        try:
            disk = SavingsLedger.load(self._path)
        except OSError:
            return
        if disk.baseline.total_samples == 0:
            return
        if disk.baseline.to_dict() != self._ledger.baseline.to_dict():
            self._ledger.baseline = disk.baseline

    def _flush_locked(self) -> None:
        from ..paths import process_is_stateless

        if process_is_stateless():
            # Stateless: keep the in-memory ledger but never write to disk.
            self._since_flush = 0
            return
        try:
            self._reload_baseline_locked()
            self._ledger.save(self._path)
            self._since_flush = 0
        except OSError:
            pass

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def estimate(self, level: int | None = None) -> SavingsEstimate:
        with self._lock:
            self._reload_baseline_locked()
            return self._ledger.best_estimate(level)


_RECORDER: SavingsRecorder | None = None


def get_recorder() -> SavingsRecorder:
    """Process-wide recorder singleton, rooted at the workspace dir."""
    global _RECORDER
    if _RECORDER is None:
        from ..paths import workspace_dir

        _RECORDER = SavingsRecorder(workspace_dir() / "output_savings.json")
    return _RECORDER


def echo_ratio(output_text: str, context_text: str, n: int = 8) -> float:
    """Fraction of the response's n-grams that already appear in the context.

    A measured (non-counterfactual) waste signal: high overlap means the model
    re-emitted code/text it was already shown. Token-ish word n-grams; cheap
    and language-agnostic. Returns 0.0 when the output is shorter than *n*.
    """
    out_words = output_text.split()
    if len(out_words) < n:
        return 0.0
    ctx_words = context_text.split()
    ctx_grams = {" ".join(ctx_words[i : i + n]) for i in range(max(0, len(ctx_words) - n + 1))}
    if not ctx_grams:
        return 0.0
    out_grams = [" ".join(out_words[i : i + n]) for i in range(len(out_words) - n + 1)]
    if not out_grams:
        return 0.0
    hits = sum(1 for g in out_grams if g in ctx_grams)
    return hits / len(out_grams)
