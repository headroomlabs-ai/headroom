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

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

# Coarse input-token buckets. Coarse on purpose: too many strata make
# per-stratum baselines sparse and noisy. Boundaries in tokens.
_INPUT_BUCKETS = (2_000, 8_000, 32_000, 128_000)


def input_bucket(input_tokens: int) -> str:
    """Map an input-token count to a coarse bucket label."""
    if input_tokens < _INPUT_BUCKETS[0]:
        return "xs"
    if input_tokens < _INPUT_BUCKETS[1]:
        return "s"
    if input_tokens < _INPUT_BUCKETS[2]:
        return "m"
    if input_tokens < _INPUT_BUCKETS[3]:
        return "l"
    return "xl"


def model_family(model: str) -> str:
    """Collapse a model id to a coarse family for stratification.

    Token-spend behaviour clusters by family far more than by point release,
    so we bucket (e.g.) every ``claude-opus-*`` together.
    """
    m = model.lower()
    for fam in ("opus", "sonnet", "haiku", "fable", "mythos", "gpt", "gemini"):
        if fam in m:
            return fam
    return "other"


def stratum_key(
    *,
    turn_kind: str,
    input_tokens: int,
    model: str,
    has_tools: bool,
) -> str:
    """Build a stratum key from request features observable BEFORE the response.

    Order is most→least specific so :meth:`BaselineModel.lookup` can back off
    by trimming trailing fields.
    """
    return "|".join(
        (
            model_family(model),
            turn_kind,
            input_bucket(input_tokens),
            "tools" if has_tools else "notools",
        )
    )


def conversation_key_from_body(body: dict[str, Any]) -> str:
    """Derive a conversation-stable key for holdout assignment.

    Stable across every turn of one conversation (so the whole conversation
    lands in one arm) and cheap: a hash of the model plus the first user
    message's text. The first user turn is immutable for a conversation's
    lifetime, which is exactly the stability we need.
    """
    model = str(body.get("model", ""))
    seed = model
    for msg in body.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                seed += "\x00" + content[:512]
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        seed += "\x00" + str(block.get("text", ""))[:512]
                        break
            break
    return hashlib.sha256(seed.encode("utf-8", "ignore")).hexdigest()


def assign_arm(conversation_key: str, holdout_fraction: float) -> str:
    """Deterministically assign a conversation to ``treatment`` or ``control``.

    ``holdout_fraction`` in [0, 1] is the share routed to ``control`` (left
    unshaped for measurement). Hashing the conversation key keeps assignment
    stable across the conversation's turns and uniform across conversations.
    """
    if holdout_fraction <= 0.0:
        return "treatment"
    if holdout_fraction >= 1.0:
        return "control"
    digest = hashlib.sha256(("arm:" + conversation_key).encode()).hexdigest()
    # Map the first 8 hex digits to [0, 1).
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return "control" if frac < holdout_fraction else "treatment"


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

_STRATUM_LABEL = "output_shaper:stratum:"
_CONTROL_LABEL = "output_shaper:control:"


def stratum_label(arm: str, key: str) -> str:
    """Encode (arm, stratum) as a transforms_applied label."""
    prefix = _STRATUM_LABEL if arm == "treatment" else _CONTROL_LABEL
    return prefix + key


def parse_stratum_label(label: str) -> tuple[str, str] | None:
    """Decode a label into ``(arm, stratum)``, or None if not one of ours."""
    if label.startswith(_STRATUM_LABEL):
        return "treatment", label[len(_STRATUM_LABEL) :]
    if label.startswith(_CONTROL_LABEL):
        return "control", label[len(_CONTROL_LABEL) :]
    return None


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

    def _flush_locked(self) -> None:
        try:
            self._ledger.save(self._path)
            self._since_flush = 0
        except OSError:
            pass

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def estimate(self) -> SavingsEstimate:
        with self._lock:
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
