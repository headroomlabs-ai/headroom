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
    input_bucket as input_bucket,
)
from .output_savings_policy import (
    model_family as model_family,
)
from .output_savings_policy import (
    parse_stratum_label,
)
from .output_savings_policy import (
    stratum_key as stratum_key,
)
from .output_savings_policy import (
    stratum_label as stratum_label,
)


@dataclass
class _Accum:
    """Running count / sum / sum-of-squares for online mean & variance."""

    n: int = 0
    sum: float = 0.0
    sumsq: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        self.sum += x
        self.sumsq += x * x

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n else 0.0

    @property
    def var(self) -> float:
        """Sample variance (unbiased). 0 when fewer than 2 observations."""
        if self.n < 2:
            return 0.0
        return max(0.0, (self.sumsq - self.sum * self.sum / self.n) / (self.n - 1))

    def merge(self, other: _Accum) -> None:
        """Fold another accumulator's observations into this one.

        n / sum / sumsq are additive, so merging is element-wise addition and
        is exactly equivalent to having ``add``-ed both observation streams.
        """
        self.n += other.n
        self.sum += other.sum
        self.sumsq += other.sumsq

    def to_dict(self) -> dict[str, float]:
        return {"n": self.n, "sum": self.sum, "sumsq": self.sumsq}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> _Accum:
        a = cls()
        a.n = int(d.get("n", 0))
        a.sum = float(d.get("sum", 0.0))
        a.sumsq = float(d.get("sumsq", 0.0))
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


@dataclass
class SavingsEstimate:
    """Result of an estimation pass."""

    tokens_saved: float
    baseline_tokens: float
    pct: float
    ci_low_pct: float
    ci_high_pct: float
    n_requests: int
    kind: str  # "estimated" (synthetic control) or "measured" (A/B holdout)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def record(self, arm: str, key: str, output_tokens: int) -> None:
        target = self.treatment if arm == "treatment" else self.control
        target.setdefault(key, _Accum()).add(output_tokens)

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

        Only strata with data in BOTH arms contribute. Returns ``None`` if no
        such stratum exists (no holdout traffic yet). Weighted by treatment
        volume; this is the unbiased causal number.
        """
        total_saved = 0.0
        total_baseline = 0.0
        var = 0.0
        n_requests = 0
        contributing = 0
        for key, t in self.treatment.items():
            c = self.control.get(key)
            if c is None or c.n == 0 or t.n == 0:
                continue
            contributing += 1
            n = t.n
            n_requests += n
            delta = c.mean - t.mean  # tokens saved per request in this stratum
            total_saved += n * delta
            total_baseline += n * c.mean
            # Var of (c.mean - t.mean) = σ²_c/n_c + σ²_t/n_t, scaled by n².
            var += (n * n) * (c.var / c.n + t.var / t.n)
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

    def best_estimate(self) -> SavingsEstimate:
        """Prefer the measured A/B number; fall back to the baseline estimate."""
        measured = self.estimate_from_holdout()
        return measured if measured is not None else self.estimate_from_baseline()

    # ---- persistence -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "treatment": {k: a.to_dict() for k, a in self.treatment.items()},
            "control": {k: a.to_dict() for k, a in self.control.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SavingsLedger:
        ledger = cls(baseline=BaselineModel.from_dict(d.get("baseline") or {}))
        for k, a in (d.get("treatment") or {}).items():
            ledger.treatment[k] = _Accum.from_dict(a)
        for k, a in (d.get("control") or {}).items():
            ledger.control[k] = _Accum.from_dict(a)
        return ledger

    def save(self, path: Any) -> None:
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), separators=(",", ":")))

    @classmethod
    def load(cls, path: Any) -> SavingsLedger:
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except (json.JSONDecodeError, ValueError, OSError):
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
        if a shaping label was found and recorded."""
        for label in labels or ():
            parsed = parse_stratum_label(str(label))
            if parsed is None:
                continue
            arm, key = parsed
            with self._lock:
                self._ledger.record(arm, key, output_tokens)
                self._since_flush += 1
                if self._since_flush >= self._flush_every:
                    self._flush_locked()
            return True
        return False

    def estimate_request_savings(self, labels: Any, output_tokens: int) -> int:
        """Per-request output tokens saved, for the savings rollup.

        For a treatment request, the synthetic-control estimate
        ``max(0, baseline_mean(stratum) - output_tokens)``; 0 for control,
        unknown strata, or when no shaping label is present. Read-only:
        unlike ``record_from_labels`` it does not mutate the ledger, so the
        two compose without double-counting."""
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

    def estimate(self) -> SavingsEstimate:
        with self._lock:
            self._reload_baseline_locked()
            return self._ledger.best_estimate()


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
