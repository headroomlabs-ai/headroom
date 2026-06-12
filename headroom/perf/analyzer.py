"""Analyze headroom proxy logs for performance insights.

Parses PERF log lines from ~/.headroom/logs/proxy.log* and produces
actionable reports on token savings, cache efficiency, and transform impact.

Cost accounting is **cache-aware**: saved tokens that would have been served
from the provider's prompt cache are valued at cache_read price (~10% for
Anthropic), not the full input price.  This prevents overstating dollar savings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

from headroom import paths as _paths
from headroom.pricing.litellm_pricing import resolve_litellm_model

log = logging.getLogger(__name__)

LOG_DIR = _paths.log_dir()

# Matches: 2026-03-07 13:38:31,009 - headroom.proxy - INFO - [hr_...] PERF model=... ...
_PERF_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .* \[(?P<rid>[^\]]+)\] PERF (?P<kv>.+)$"
)

# Matches: content_router: 51 msgs — ...
_ROUTER_RE = re.compile(r"content_router: (?P<msgs>\d+) msgs — (?P<detail>.+)$")

# Matches: Transform content_router: 52503 -> 26006 tokens (saved 26497)
_TRANSFORM_RE = re.compile(
    r"Transform (?P<name>\w+): (?P<before>\d+) -> (?P<after>\d+) tokens \(saved (?P<saved>\d+)\)"
)

# Matches: Pipeline complete: 52503 -> 26006 tokens (saved 26497, 50.5% reduction)
_PIPELINE_RE = re.compile(
    r"Pipeline complete: (?P<before>\d+) -> (?P<after>\d+) tokens "
    r"\(saved (?P<saved>\d+), (?P<pct>[\d.]+)% reduction\)"
)

# Matches: TOIN: 105 patterns, 3837 compressions, 0 retrievals, 0.0% retrieval rate
_TOIN_RE = re.compile(
    r"TOIN: (?P<patterns>\d+) patterns, (?P<compressions>\d+) compressions, "
    r"(?P<retrievals>\d+) retrievals, (?P<rate>[\d.]+)% retrieval rate"
)


# ---------------------------------------------------------------------------
# Cache-aware pricing via LiteLLM
# ---------------------------------------------------------------------------

# LiteLLM already knows per-token costs for 100+ models including
# cache_read and cache_creation pricing.  We call it directly instead
# of maintaining our own pricing tables.

try:
    import litellm as _litellm

    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False


def _litellm_cost(
    model: str,
    prompt_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Compute input cost via litellm.cost_per_token (cache-aware).

    Returns total input cost in USD, or None if model not found.
    """
    if not _LITELLM_AVAILABLE:
        return None
    resolved = resolve_litellm_model(model)
    try:
        input_cost, _ = _litellm.cost_per_token(
            model=resolved,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            cache_read_input_tokens=cache_read_tokens,
            cache_creation_input_tokens=cache_write_tokens,
        )
        return float(input_cost)
    except Exception:
        return None


def _get_list_price(model: str) -> float | None:
    """Get list input price per 1M tokens."""
    if not _LITELLM_AVAILABLE:
        return None
    resolved = resolve_litellm_model(model)
    info = _litellm.model_cost.get(resolved, {})
    cost_per_token = info.get("input_cost_per_token")
    return cost_per_token * 1_000_000 if cost_per_token else None


def _parse_kv(kv_str: str) -> dict[str, str]:
    """Parse key=value pairs from a PERF log line.

    The ``transforms=`` field is always last and its value may contain spaces
    (e.g. ``transforms=router:excluded:tool*32 read_lifecycle:stale*17``).
    Everything after ``transforms=`` is captured as a single value.
    """
    result: dict[str, str] = {}
    # Handle transforms= specially since its value contains spaces
    if "transforms=" in kv_str:
        before, transforms_val = kv_str.split("transforms=", 1)
        transform_parts: list[str] = []
        for part in transforms_val.split():
            if "=" in part:
                k, v = part.split("=", 1)
                result[k] = v
            else:
                transform_parts.append(part)
        result["transforms"] = " ".join(transform_parts).strip()
        kv_str = before
    for part in kv_str.split():
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


@dataclass
class PerfRecord:
    """A single parsed PERF log entry."""

    timestamp: str
    request_id: str
    model: str = ""
    num_messages: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_hit_pct: int = 0
    optimization_ms: float = 0
    transforms: list[str] = field(default_factory=list)
    client: str = ""


@dataclass
class RouterRecord:
    """A parsed content_router summary line."""

    timestamp: str
    num_messages: int = 0
    compressed: int = 0
    excluded: int = 0
    skipped: int = 0
    unchanged: int = 0
    content_blocks: int = 0


@dataclass
class TransformRecord:
    """A parsed per-transform line."""

    timestamp: str
    name: str = ""
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0


@dataclass
class ToinRecord:
    """A parsed TOIN status line."""

    timestamp: str
    patterns: int = 0
    compressions: int = 0
    retrievals: int = 0
    retrieval_rate: float = 0.0


@dataclass
class PerfReport:
    """Aggregated performance report."""

    perf_records: list[PerfRecord] = field(default_factory=list)
    router_records: list[RouterRecord] = field(default_factory=list)
    transform_records: list[TransformRecord] = field(default_factory=list)
    toin_records: list[ToinRecord] = field(default_factory=list)
    log_files_read: int = 0
    total_lines_parsed: int = 0
    # Window covered by the report. `requested_hours` is what the caller
    # asked for; `oldest_kept_ts` / `newest_kept_ts` are the actual
    # timestamps of the oldest and newest records that survived the
    # filter (may be narrower if the log doesn't go back that far).
    # All optional so existing callers keep working.
    requested_hours: float | None = None
    oldest_kept_ts: str | None = None
    newest_kept_ts: str | None = None
    records_filtered_out: int = 0


# Log timestamps are emitted by Python's `logging` formatter as
# `YYYY-MM-DD HH:MM:SS,fff`. We keep the parser permissive so the perf
# CLI never throws on a stray malformed line — unparsable records are
# just kept (better to over-report than to silently drop data).
_LOG_TS_FMT = "%Y-%m-%d %H:%M:%S,%f"


def _parse_log_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, _LOG_TS_FMT)
    except ValueError:
        return None


def parse_log_files(last_n_hours: float = 168.0) -> PerfReport:
    """Parse all proxy log files and return structured records.

    Args:
        last_n_hours: Only include records from the last N hours (default 7 days).
            Records with un-parseable timestamps are kept (fail-open) — the
            window in the report header reflects the actual timestamps that
            survived the filter, so the user can see whether the log went
            back far enough.

    Returns:
        PerfReport with all parsed records.
    """
    report = PerfReport()
    report.requested_hours = last_n_hours

    log_dir = _paths.log_dir()
    if not log_dir.exists():
        return report

    cutoff = datetime.now() - timedelta(hours=last_n_hours) if last_n_hours > 0 else None

    def _within_window(ts_str: str | None) -> bool:
        # Fail-open: records without a parseable timestamp are kept. The
        # alternative (silent drop) makes `headroom perf` lie about coverage.
        if cutoff is None:
            return True
        ts = _parse_log_ts(ts_str)
        if ts is None:
            return True
        return ts >= cutoff

    def _track_window(ts_str: str | None) -> None:
        if not ts_str:
            return
        if report.oldest_kept_ts is None or ts_str < report.oldest_kept_ts:
            report.oldest_kept_ts = ts_str
        if report.newest_kept_ts is None or ts_str > report.newest_kept_ts:
            report.newest_kept_ts = ts_str

    # Collect log files: proxy.log, proxy.log.1, proxy.log.2, ...
    log_files = sorted(log_dir.glob("proxy.log*"), key=lambda p: p.stat().st_mtime)

    for log_file in log_files:
        report.log_files_read += 1
        try:
            with open(log_file, encoding="utf-8", errors="replace") as f:
                for line in f:
                    report.total_lines_parsed += 1
                    line = line.rstrip()

                    # PERF lines (richest data)
                    m = _PERF_RE.match(line)
                    if m:
                        kv = _parse_kv(m.group("kv"))
                        transforms_str = kv.get("transforms", "none")
                        # Handle both old comma-separated and new space-separated *N format
                        if transforms_str == "none":
                            transforms: list[str] = []
                        elif "*" in transforms_str or " " in transforms_str:
                            # New format: "router:excluded:tool*32 read_lifecycle:stale*17"
                            transforms = []
                            for part in transforms_str.split():
                                if "*" in part:
                                    name, _ = part.rsplit("*", 1)
                                else:
                                    name = part
                                transforms.append(name)
                        else:
                            # Old comma-separated format
                            transforms = transforms_str.split(",")
                        ts = m.group("ts")
                        if not _within_window(ts):
                            report.records_filtered_out += 1
                            continue
                        _track_window(ts)
                        report.perf_records.append(
                            PerfRecord(
                                timestamp=ts,
                                request_id=m.group("rid"),
                                model=kv.get("model", ""),
                                num_messages=int(kv.get("msgs", 0)),
                                tokens_before=int(kv.get("tok_before", 0)),
                                tokens_after=int(kv.get("tok_after", 0)),
                                tokens_saved=int(kv.get("tok_saved", 0)),
                                cache_read=int(kv.get("cache_read", 0)),
                                cache_write=int(kv.get("cache_write", 0)),
                                cache_hit_pct=int(kv.get("cache_hit_pct", 0)),
                                optimization_ms=float(kv.get("opt_ms", 0)),
                                transforms=transforms,
                                client=kv.get("client", ""),
                            )
                        )
                        continue

                    # content_router summary lines
                    if "content_router:" in line and "msgs" in line:
                        m2 = _ROUTER_RE.search(line)
                        if m2:
                            ts = line[:23]
                            if not _within_window(ts):
                                report.records_filtered_out += 1
                                continue
                            _track_window(ts)
                            detail = m2.group("detail")
                            rec = RouterRecord(
                                timestamp=ts,
                                num_messages=int(m2.group("msgs")),
                            )
                            # Parse counts from detail string
                            for part in detail.split(","):
                                part = part.strip()
                                num_match = re.match(r"(\d+)\s+(\w+)", part)
                                if num_match:
                                    count = int(num_match.group(1))
                                    kind = num_match.group(2)
                                    if kind == "compressed":
                                        rec.compressed = count
                                    elif kind == "excluded":
                                        rec.excluded = count
                                    elif kind == "skipped":
                                        rec.skipped = count
                                    elif kind == "unchanged":
                                        rec.unchanged = count
                                    elif kind == "content" and "block" in part:
                                        rec.content_blocks = count
                            report.router_records.append(rec)
                            continue

                    # Per-transform lines
                    m3 = _TRANSFORM_RE.search(line)
                    if m3:
                        ts = line[:23]
                        if not _within_window(ts):
                            report.records_filtered_out += 1
                            continue
                        _track_window(ts)
                        report.transform_records.append(
                            TransformRecord(
                                timestamp=ts,
                                name=m3.group("name"),
                                tokens_before=int(m3.group("before")),
                                tokens_after=int(m3.group("after")),
                                tokens_saved=int(m3.group("saved")),
                            )
                        )
                        continue

                    # TOIN status lines
                    m4 = _TOIN_RE.search(line)
                    if m4:
                        ts = line[:23]
                        if not _within_window(ts):
                            report.records_filtered_out += 1
                            continue
                        _track_window(ts)
                        report.toin_records.append(
                            ToinRecord(
                                timestamp=ts,
                                patterns=int(m4.group("patterns")),
                                compressions=int(m4.group("compressions")),
                                retrievals=int(m4.group("retrievals")),
                                retrieval_rate=float(m4.group("rate")),
                            )
                        )

        except OSError:
            continue

    return report


def format_report(report: PerfReport) -> str:
    """Format a PerfReport into a human-readable string."""
    lines: list[str] = []

    if not report.perf_records and not report.router_records:
        lines.append("No performance data found in ~/.headroom/logs/")
        lines.append("")
        lines.append("Start the proxy to begin collecting data:")
        lines.append("  headroom proxy")
        return "\n".join(lines)

    # Header
    lines.append("Headroom Performance Report")
    lines.append("=" * 60)
    if report.requested_hours is not None:
        if report.oldest_kept_ts and report.newest_kept_ts:
            window_str = (
                f"Window: last {report.requested_hours:g}h "
                f"(actual data: {report.oldest_kept_ts[:19]} → "
                f"{report.newest_kept_ts[:19]})"
            )
        else:
            window_str = f"Window: last {report.requested_hours:g}h (no records found in window)"
        lines.append(window_str)
        if report.records_filtered_out > 0:
            lines.append(
                f"Records outside window:  {report.records_filtered_out:,} "
                "(filtered out — increase --hours to include them)"
            )
    lines.append("")

    records = report.perf_records

    if records:
        # Overview
        total_before = sum(r.tokens_before for r in records)
        total_after = sum(r.tokens_after for r in records)
        total_saved = sum(r.tokens_saved for r in records)
        pct = (total_saved / total_before * 100) if total_before > 0 else 0

        lines.append(f"Requests:     {len(records)}")
        lines.append(f"Tokens:       {total_before:,} -> {total_after:,} ({pct:.1f}% reduction)")
        lines.append(f"Total saved:  {total_saved:,} tokens")
        lines.append("")

        # Per-model breakdown with list prices
        by_model: dict[str, list[PerfRecord]] = {}
        for r in records:
            by_model.setdefault(r.model, []).append(r)

        lines.append("Per-Model Breakdown")
        lines.append("-" * 40)
        for model, model_recs in sorted(by_model.items()):
            m_saved = sum(r.tokens_saved for r in model_recs)
            m_before = sum(r.tokens_before for r in model_recs)
            m_pct = (m_saved / m_before * 100) if m_before > 0 else 0
            list_price = _get_list_price(model)
            price_str = f"${list_price:.2f}/MTok" if list_price else "unknown"
            est_str = (
                f"  ~${m_saved * list_price / 1_000_000:.2f} at list price" if list_price else ""
            )
            lines.append(
                f"  {model}: {len(model_recs)} reqs, "
                f"{m_saved:,} tokens saved ({m_pct:.0f}%), "
                f"list price {price_str}{est_str}"
            )
        lines.append("  * Actual bill savings depend on provider caching behavior")
        lines.append("")

        # Cache analysis
        cache_records = [r for r in records if (r.cache_read + r.cache_write) > 0]
        if cache_records:
            lines.append("Cache Performance")
            lines.append("-" * 40)
            total_cr = sum(r.cache_read for r in cache_records)
            total_cw = sum(r.cache_write for r in cache_records)
            total_cache = total_cr + total_cw
            hit_pct = (total_cr / total_cache * 100) if total_cache > 0 else 0
            lines.append(f"  Cache read:    {total_cr:,} tokens")
            lines.append(f"  Cache write:   {total_cw:,} tokens")
            lines.append(f"  Hit rate:      {hit_pct:.1f}%")

            # Identify cache instability: requests where write >> read
            unstable = [r for r in cache_records if r.cache_write > r.cache_read * 2]
            if unstable:
                lines.append(
                    f"  Unstable:      {len(unstable)}/{len(cache_records)} requests "
                    f"had cache_write > 2x cache_read"
                )

            # Show cache progression (first 5 vs last 5)
            if len(cache_records) >= 10:
                first5_cr = sum(r.cache_read for r in cache_records[:5])
                first5_cw = sum(r.cache_write for r in cache_records[:5])
                last5_cr = sum(r.cache_read for r in cache_records[-5:])
                last5_cw = sum(r.cache_write for r in cache_records[-5:])
                lines.append(f"  First 5 avg:   read={first5_cr // 5:,} write={first5_cw // 5:,}")
                lines.append(f"  Last 5 avg:    read={last5_cr // 5:,} write={last5_cw // 5:,}")
                if last5_cr > first5_cr * 2:
                    lines.append("  -> Cache stabilizing over conversation lifetime")
                elif first5_cw > first5_cr * 3:
                    lines.append(
                        "  ! Early turns have poor cache hits — "
                        "compression decisions may be flipping"
                    )
            lines.append("")

        # Optimization latency
        opt_times = [r.optimization_ms for r in records if r.optimization_ms > 0]
        if opt_times:
            avg_opt = sum(opt_times) / len(opt_times)
            max_opt = max(opt_times)
            lines.append("Optimization Overhead")
            lines.append("-" * 40)
            lines.append(f"  Average:  {avg_opt:.0f}ms")
            lines.append(f"  Max:      {max_opt:.0f}ms")
            slow = [t for t in opt_times if t > 500]
            if slow:
                lines.append(f"  >500ms:   {len(slow)} requests")
            lines.append("")

        # Conversation size distribution
        msg_counts = [r.num_messages for r in records if r.num_messages > 0]
        if msg_counts:
            lines.append("Conversation Size")
            lines.append("-" * 40)
            lines.append(f"  Min msgs:  {min(msg_counts)}")
            lines.append(f"  Max msgs:  {max(msg_counts)}")
            lines.append(f"  Avg msgs:  {sum(msg_counts) // len(msg_counts)}")
            lines.append("")

    # Transform effectiveness (from transform_records)
    if report.transform_records:
        lines.append("Transform Effectiveness")
        lines.append("-" * 40)
        by_name: dict[str, list[TransformRecord]] = {}
        for tr in report.transform_records:
            by_name.setdefault(tr.name, []).append(tr)
        for name, recs in sorted(by_name.items(), key=lambda x: -sum(r.tokens_saved for r in x[1])):
            total_s = sum(r.tokens_saved for r in recs)
            total_b = sum(r.tokens_before for r in recs)
            avg_pct = (total_s / total_b * 100) if total_b > 0 else 0
            lines.append(
                f"  {name}: {avg_pct:.1f}% avg reduction, {len(recs)} uses, {total_s:,} saved"
            )
        lines.append("")

    # Router routing breakdown
    if report.router_records:
        lines.append("Content Router Routing")
        lines.append("-" * 40)
        total_compressed = sum(r.compressed for r in report.router_records)
        total_excluded = sum(r.excluded for r in report.router_records)
        total_skipped = sum(r.skipped for r in report.router_records)
        total_unchanged = sum(r.unchanged for r in report.router_records)
        total_all = total_compressed + total_excluded + total_skipped + total_unchanged
        if total_all > 0:
            lines.append(
                f"  Compressed:  {total_compressed} ({total_compressed / total_all * 100:.0f}%)"
            )
            lines.append(
                f"  Excluded:    {total_excluded} ({total_excluded / total_all * 100:.0f}%) — Read/Glob outputs"
            )
            lines.append(
                f"  Skipped:     {total_skipped} ({total_skipped / total_all * 100:.0f}%) — <50 words"
            )
            lines.append(
                f"  Unchanged:   {total_unchanged} ({total_unchanged / total_all * 100:.0f}%) — ratio too high"
            )
        if total_excluded > total_compressed * 3:
            lines.append("  ! Excluded tools dominate — consider compressing stale Read outputs")
        lines.append("")

    # TOIN status — log-derived counters first, then live-store highlights.
    if report.toin_records:
        latest = report.toin_records[-1]
        lines.append("TOIN Learning")
        lines.append("-" * 40)
        lines.append(f"  Patterns:     {latest.patterns}")
        lines.append(f"  Compressions: {latest.compressions:,}")
        lines.append(f"  Retrievals:   {latest.retrievals} ({latest.retrieval_rate}%)")
        if latest.retrieval_rate == 0 and latest.compressions > 100:
            lines.append("  ! 0% retrieval rate — TOIN learning but never used")
        lines.append("")

    # TOIN highlights — read the live on-disk pattern store and surface
    # strategy distribution + top high-impact patterns in human-readable
    # form. Pattern keys are opaque hashes, so the actionable signal is
    # *which strategies are winning* and *how many patterns have crossed
    # the recommendation threshold*. Best-effort: if TOIN isn't installed
    # or the store is empty, we skip the section silently.
    toin_lines = _format_toin_highlights()
    if toin_lines:
        lines.extend(toin_lines)
        lines.append("")

    # Recommendations
    recommendations = _generate_recommendations(report)
    if recommendations:
        lines.append("Recommendations")
        lines.append("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

    # Footer
    lines.append(
        f"Log files: {report.log_files_read} | Lines parsed: {report.total_lines_parsed:,}"
    )
    lines.append(f"Log dir: {_paths.log_dir()}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Machine-readable views (JSON / CSV) — issue #595
# ---------------------------------------------------------------------------
#
# `parse_log_files()` already returns a fully-structured `PerfReport`; these
# helpers expose it without forcing CI pipelines, dashboards, or agent
# harnesses to scrape the colored text report. The aggregate numbers mirror
# `format_report()` exactly so a JSON consumer and a human reading the report
# never disagree.

# Column order for the per-record (`--raw`) machine output. Kept as a module
# constant so the CLI's CSV writer and any external consumer share one source
# of truth.
PERF_RECORD_FIELDS = [
    "timestamp",
    "request_id",
    "model",
    "client",
    "num_messages",
    "tokens_before",
    "tokens_after",
    "tokens_saved",
    "cache_read",
    "cache_write",
    "cache_hit_pct",
    "optimization_ms",
    "transforms",
]


def _pct(saved: int, before: int) -> float:
    """Reduction percentage, rounded to 1dp, guarding divide-by-zero."""
    return round(saved / before * 100, 1) if before > 0 else 0.0


def build_perf_summary(report: PerfReport) -> dict:
    """Aggregate a ``PerfReport`` into a JSON-serialisable summary dict.

    The shape mirrors the human-readable ``format_report`` numbers so the same
    data drives CI regression guards (``jq '.savings_pct < 70'``), dashboards,
    and end-of-session savings summaries in agent wrappers.
    """
    records = report.perf_records

    total_before = sum(r.tokens_before for r in records)
    total_after = sum(r.tokens_after for r in records)
    total_saved = sum(r.tokens_saved for r in records)

    total_cr = sum(r.cache_read for r in records)
    total_cw = sum(r.cache_write for r in records)
    total_cache = total_cr + total_cw
    cache_hit_pct = round(total_cr / total_cache * 100, 1) if total_cache > 0 else 0.0

    by_model_groups: dict[str, list[PerfRecord]] = {}
    for r in records:
        by_model_groups.setdefault(r.model, []).append(r)
    by_model = []
    for model, recs in sorted(by_model_groups.items()):
        m_before = sum(r.tokens_before for r in recs)
        m_after = sum(r.tokens_after for r in recs)
        m_saved = sum(r.tokens_saved for r in recs)
        by_model.append(
            {
                "model": model,
                "requests": len(recs),
                "tokens_before": m_before,
                "tokens_after": m_after,
                "tokens_saved": m_saved,
                "savings_pct": _pct(m_saved, m_before),
                "list_price_per_mtok": _get_list_price(model),
            }
        )

    by_transform_groups: dict[str, list[TransformRecord]] = {}
    for tr in report.transform_records:
        by_transform_groups.setdefault(tr.name, []).append(tr)
    by_transform = []
    for name, t_recs in sorted(
        by_transform_groups.items(), key=lambda kv: -sum(r.tokens_saved for r in kv[1])
    ):
        t_before = sum(r.tokens_before for r in t_recs)
        t_saved = sum(r.tokens_saved for r in t_recs)
        by_transform.append(
            {
                "transform": name,
                "uses": len(t_recs),
                "tokens_before": t_before,
                "tokens_saved": t_saved,
                "savings_pct": _pct(t_saved, t_before),
            }
        )

    return {
        "window_hours": report.requested_hours,
        "actual_window": {
            "oldest": report.oldest_kept_ts,
            "newest": report.newest_kept_ts,
        },
        "records_filtered_out": report.records_filtered_out,
        "total_requests": len(records),
        "total_tokens_before": total_before,
        "total_tokens_after": total_after,
        "tokens_saved": total_saved,
        "savings_pct": _pct(total_saved, total_before),
        "cache_read_tokens": total_cr,
        "cache_write_tokens": total_cw,
        "cache_hit_pct": cache_hit_pct,
        "by_model": by_model,
        "by_transform": by_transform,
        "log_files_read": report.log_files_read,
        "total_lines_parsed": report.total_lines_parsed,
    }


def perf_records_as_dicts(report: PerfReport) -> list[dict]:
    """Per-record view of the parsed PERF entries (for ``--raw`` machine output).

    ``transforms`` stays a list so JSON consumers keep structure; the CSV
    writer flattens it to a comma-joined string at the edge.
    """
    return [asdict(r) for r in report.perf_records]


def _format_toin_highlights() -> list[str]:
    """Render a human-readable TOIN highlights block from the live store.

    Returns an empty list when TOIN is unavailable or has no patterns.
    Pattern keys (auth_mode, model_family, structure_hash) are opaque
    hashes so we don't print them as rows — instead we group by the
    learned ``optimal_strategy`` (a human-readable string like
    ``"lossless:table(240->len=7026)"``) and surface the highest-impact
    slices via ``avg_token_reduction``.
    """
    try:
        from headroom.telemetry.toin import get_toin
    except ImportError:
        return []

    try:
        pairs = get_toin().iter_patterns()
    except Exception:  # noqa: BLE001 — perf must never fail on TOIN errors
        return []

    if not pairs:
        return []

    # Strategy distribution: how many patterns settled on each strategy.
    strategy_counts: dict[str, int] = {}
    for _key, pattern in pairs:
        strategy = pattern.optimal_strategy or "default"
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    # Top patterns by avg token reduction (the high-impact learnings).
    by_impact = sorted(
        pairs,
        key=lambda kp: kp[1].avg_token_reduction,
        reverse=True,
    )[:5]

    # How many patterns have enough samples to drive a recommendation.
    # Falls back to 0 if the threshold attr isn't reachable.
    try:
        from headroom.telemetry.toin import get_toin as _get

        threshold = _get()._config.min_samples_for_recommendation
    except Exception:  # noqa: BLE001
        threshold = 1
    qualified = sum(1 for _k, p in pairs if p.sample_size >= threshold)

    lines: list[str] = []
    lines.append("TOIN Highlights (live store)")
    lines.append("-" * 40)
    lines.append(
        f"  {qualified}/{len(pairs)} patterns have ≥{threshold} samples "
        f"(eligible for `python -m headroom.cli.toin_publish`)"
    )
    lines.append("")
    lines.append("  Strategy distribution:")
    for strategy, count in sorted(strategy_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]:
        lines.append(f"    {count:>4} pattern(s)  {strategy}")

    # Only surface patterns with non-trivial impact AND a non-default
    # strategy — single-digit-token "wins" against the default strategy
    # are noise, not insight.
    impact_rows = [
        (kp[1].avg_token_reduction, kp[1].total_compressions, kp[1].optimal_strategy or "default")
        for kp in by_impact
        if kp[1].avg_token_reduction >= 50 and (kp[1].optimal_strategy or "default") != "default"
    ]
    if impact_rows:
        lines.append("")
        lines.append("  Top patterns by avg token reduction:")
        for avg_red, n, strategy in impact_rows:
            lines.append(f"    {avg_red:>7.0f} tok avg ({n:>3} compression(s))  {strategy}")

    if qualified == 0 and len(pairs) > 0:
        lines.append("")
        lines.append(
            f"  ! No pattern has reached {threshold} samples — TOIN is still warming up. "
            "Recommendations TOML will be empty until traffic grows."
        )

    return lines


def _generate_recommendations(report: PerfReport) -> list[str]:
    """Generate actionable recommendations from the report data."""
    recs: list[str] = []

    if report.perf_records:
        cache_recs = [r for r in report.perf_records if (r.cache_read + r.cache_write) > 0]
        if cache_recs:
            total_cr = sum(r.cache_read for r in cache_recs)
            total_cw = sum(r.cache_write for r in cache_recs)
            if total_cw > total_cr * 1.5:
                recs.append(
                    "Cache prefix unstable — compression decisions may be flipping "
                    "across turns due to adaptive min_ratio threshold"
                )

            # Check early-turn instability
            if len(cache_recs) >= 5:
                first5 = cache_recs[:5]
                early_ratio = sum(r.cache_read for r in first5) / max(
                    1, sum(r.cache_write for r in first5)
                )
                if early_ratio < 0.5:
                    recs.append(
                        "First 5 turns have very low cache hit ratio — "
                        "consider pinning compression decisions for prefix stability"
                    )

        # Optimization latency
        slow = [r for r in report.perf_records if r.optimization_ms > 500]
        if len(slow) > len(report.perf_records) * 0.2:
            recs.append(
                f"{len(slow)} requests took >500ms for optimization — "
                "consider reducing transform pipeline"
            )

    if report.router_records:
        total_excluded = sum(r.excluded for r in report.router_records)
        total_compressed = sum(r.compressed for r in report.router_records)
        if total_excluded > 0 and total_compressed > 0:
            if total_excluded > total_compressed * 3:
                recs.append(
                    "Read/Glob outputs are majority of messages but excluded — "
                    "compress stale reads (>10 turns old) for significant savings"
                )

    if report.toin_records:
        latest = report.toin_records[-1]
        if latest.retrieval_rate == 0 and latest.compressions > 100:
            recs.append(
                "TOIN has 0% retrieval rate with "
                f"{latest.compressions:,} compressions — review CCR integration"
            )

    # Check cache_aligner effectiveness from transform records
    for tr in report.transform_records:
        if tr.name == "cache_aligner" and tr.tokens_saved < 10:
            recs.append(
                "cache_aligner saving <10 tokens — "
                "consider disabling (system prompt likely has no dynamic content)"
            )
            break

    return recs
