"""License validation and usage reporting for managed/enterprise deployments.

Phone-home module that validates license keys and reports aggregate usage
statistics to the Headroom cloud for billing. Designed to be non-intrusive:
the proxy works normally even if the cloud API is completely unreachable.

Privacy guarantees:
- Never sends message content, API keys, prompts, tool results, or user data
- Only sends aggregate counts: requests, tokens saved, model distribution
- All communication is over HTTPS

Usage:
    reporter = UsageReporter(license_key="hlk_...")
    license_info = await reporter.validate_license()
    await reporter.start(proxy)  # starts background loop
    ...
    await reporter.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from headroom import paths as _paths

if TYPE_CHECKING:
    from headroom.proxy.server import HeadroomProxy

logger = logging.getLogger("headroom.telemetry.reporter")

# Grace period: if the cloud API is unreachable, use cached license for up to 7 days
GRACE_PERIOD_SECONDS = 7 * 24 * 3600  # 7 days

# Default cache location (workspace bucket, respects HEADROOM_WORKSPACE_DIR).
LICENSE_CACHE_PATH = _paths.license_cache_path()


@dataclass
class LicenseInfo:
    """Cached license validation result."""

    status: str  # "active", "trial", "expired", "invalid"
    org_id: str | None = None
    org_name: str | None = None
    plan: str | None = None
    quota_tokens: int | None = None  # None = unlimited
    trial_expires_at: datetime | None = None
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON caching."""
        d = asdict(self)
        d["validated_at"] = self.validated_at.isoformat()
        if self.trial_expires_at:
            d["trial_expires_at"] = self.trial_expires_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LicenseInfo:
        """Deserialize from JSON cache."""
        validated_at = data.get("validated_at")
        if isinstance(validated_at, str):
            data["validated_at"] = datetime.fromisoformat(validated_at)
        trial_expires_at = data.get("trial_expires_at")
        if isinstance(trial_expires_at, str):
            data["trial_expires_at"] = datetime.fromisoformat(trial_expires_at)
        elif trial_expires_at is None:
            data["trial_expires_at"] = None
        return cls(
            status=data.get("status", "invalid"),
            org_id=data.get("org_id"),
            org_name=data.get("org_name"),
            plan=data.get("plan"),
            quota_tokens=data.get("quota_tokens"),
            trial_expires_at=data.get("trial_expires_at"),
            validated_at=data.get("validated_at", datetime.now(timezone.utc)),
        )


class UsageReporter:
    """Background license validator and aggregate usage reporter.

    Validates the license key on startup, then periodically sends aggregate
    usage stats to the Headroom cloud. If the cloud is unreachable, the proxy
    continues to operate using cached license info (grace period: 7 days).

    Never sends: message content, API keys, prompts, tool results, user data.
    """

    def __init__(
        self,
        license_key: str,
        cloud_url: str = "https://app.headroomlabs.ai",
        report_interval: int = 300,
        cache_path: Path | None = None,
    ):
        self._license_key = license_key
        self._cloud_url = cloud_url.rstrip("/")
        self._report_interval = report_interval
        self._cache_path = cache_path or LICENSE_CACHE_PATH
        self._license_info: LicenseInfo | None = None
        self._proxy: HeadroomProxy | None = None
        self._task: asyncio.Task[None] | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._stopped = False

        # Snapshot of proxy metrics at last report (for computing deltas)
        self._last_report_time: datetime | None = None
        self._last_tokens_saved_by_model: dict[str, int] = {}
        self._last_tokens_sent_by_model: dict[str, int] = {}
        self._last_requests_by_model: dict[str, int] = {}

    async def validate_license(self) -> LicenseInfo:
        """Validate the license key against the cloud API.

        On failure, falls back to cached license info if within grace period.
        """
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._cloud_url}/v1/license/validate",
                json={"license_key": self._license_key},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                trial_expires_at = data.get("trial_expires_at")
                if isinstance(trial_expires_at, str):
                    trial_expires_at = datetime.fromisoformat(trial_expires_at)

                self._license_info = LicenseInfo(
                    status=data.get("status", "invalid"),
                    org_id=data.get("org_id"),
                    org_name=data.get("org_name"),
                    plan=data.get("plan"),
                    quota_tokens=data.get("quota_tokens"),
                    trial_expires_at=trial_expires_at,
                    validated_at=datetime.now(timezone.utc),
                )
                self._save_cache()
                logger.info(
                    "License validated: status=%s org=%s plan=%s",
                    self._license_info.status,
                    self._license_info.org_name,
                    self._license_info.plan,
                )
                return self._license_info
            else:
                logger.warning(
                    "License validation returned status %d, using cached info",
                    resp.status_code,
                )
        except Exception:
            logger.warning(
                "Could not reach license server at %s, using cached info",
                self._cloud_url,
                exc_info=True,
            )

        # Fallback to cache
        return self._load_cache_or_default()

    async def start(self, proxy: HeadroomProxy) -> None:
        """Start the background reporting loop. Called during proxy startup."""
        self._proxy = proxy
        self._stopped = False

        # Validate license first
        await self.validate_license()

        # Take initial snapshot of metrics
        self._snapshot_metrics()

        # Start background reporting loop
        self._task = asyncio.create_task(self._report_loop())
        logger.info(
            "Usage reporter started (interval=%ds, cloud=%s)",
            self._report_interval,
            self._cloud_url,
        )

    async def stop(self) -> None:
        """Stop the background reporting loop. Called during proxy shutdown."""
        self._stopped = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("Usage reporter stopped")

    @property
    def is_active(self) -> bool:
        """Whether the license is valid and usage is within quota."""
        if self._license_info is None:
            return True  # No license info yet = allow (grace)
        return self._license_info.status in ("active", "trial")

    @property
    def should_compress(self) -> bool:
        """Whether compression should be applied. False = passthrough mode.

        Returns True (allow compression) unless the license is definitively expired
        AND outside the grace period.
        """
        if self._license_info is None:
            return True  # No license info yet = allow compression
        if self._license_info.status in ("active", "trial"):
            return True
        if self._license_info.status == "expired":
            # Check grace period
            age = (datetime.now(timezone.utc) - self._license_info.validated_at).total_seconds()
            if age < GRACE_PERIOD_SECONDS:
                return True
            return False
        # "invalid" or unknown status: still allow compression (fail open)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
                headers={"User-Agent": "headroom-proxy"},
            )
        return self._http_client

    async def _report_loop(self) -> None:
        """Background loop: report usage every N seconds."""
        while not self._stopped:
            try:
                await asyncio.sleep(self._report_interval)
                if self._stopped:
                    break
                await self._report_usage()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning("Usage report failed, will retry next interval", exc_info=True)

    async def _report_usage(self) -> None:
        """Collect aggregate stats from the proxy and send to cloud."""
        if self._proxy is None:
            return

        cost_tracker = self._proxy.cost_tracker
        if cost_tracker is None:
            return

        now = datetime.now(timezone.utc)
        period_start = self._last_report_time or now

        # Compute deltas since last report
        current_saved = dict(cost_tracker._tokens_saved_by_model)
        current_sent = dict(cost_tracker._tokens_sent_by_model)
        current_reqs = dict(cost_tracker._requests_by_model)

        delta_saved_by_model: dict[str, int] = {}
        delta_sent_by_model: dict[str, int] = {}
        delta_reqs_by_model: dict[str, int] = {}

        all_models = set(current_saved) | set(current_sent) | set(current_reqs)
        total_tokens_saved = 0
        total_tokens_before = 0
        total_tokens_after = 0
        total_requests = 0

        for model in all_models:
            saved = current_saved.get(model, 0) - self._last_tokens_saved_by_model.get(model, 0)
            sent = current_sent.get(model, 0) - self._last_tokens_sent_by_model.get(model, 0)
            reqs = current_reqs.get(model, 0) - self._last_requests_by_model.get(model, 0)
            if reqs > 0:
                delta_reqs_by_model[model] = reqs
            if saved > 0:
                delta_saved_by_model[model] = saved
            if sent > 0:
                delta_sent_by_model[model] = sent
            total_tokens_saved += max(0, saved)
            total_tokens_after += max(0, sent)
            total_tokens_before += max(0, saved) + max(0, sent)
            total_requests += max(0, reqs)

        # Skip empty reports
        if total_requests == 0:
            self._last_report_time = now
            return

        payload = {
            "license_key": self._license_key,
            "period_start": period_start.isoformat(),
            "period_end": now.isoformat(),
            "requests": total_requests,
            "tokens_before": total_tokens_before,
            "tokens_after": total_tokens_after,
            "tokens_saved": total_tokens_saved,
            "models": delta_reqs_by_model,
        }

        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._cloud_url}/v1/license/usage",
                json=payload,
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                if status == "expired" and self._license_info:
                    self._license_info.status = "expired"
                    self._save_cache()
                    logger.warning("License expired: %s", data.get("message", ""))
                elif status and self._license_info:
                    self._license_info.status = status
                logger.debug(
                    "Usage reported: %d requests, %d tokens saved",
                    total_requests,
                    total_tokens_saved,
                )
                # Only advance the baseline after a confirmed 200. Usage is a
                # delta against the last snapshot, so snapshotting on a failed
                # send (non-200 or exception) would permanently drop this
                # window's requests/tokens from billing — the next report starts
                # from the advanced baseline and never re-includes them. Leaving
                # both baselines intact makes the next report retry the full
                # period. (The empty-report early-return above only advances the
                # time, since there is nothing to lose.)
                self._snapshot_metrics()
                self._last_report_time = now
            else:
                logger.warning("Usage report returned status %d", resp.status_code)
        except Exception:
            logger.warning("Failed to send usage report", exc_info=True)

    def _snapshot_metrics(self) -> None:
        """Take a snapshot of current proxy metrics for delta computation."""
        if self._proxy is None or self._proxy.cost_tracker is None:
            return
        ct = self._proxy.cost_tracker
        self._last_tokens_saved_by_model = dict(ct._tokens_saved_by_model)
        self._last_tokens_sent_by_model = dict(ct._tokens_sent_by_model)
        self._last_requests_by_model = dict(ct._requests_by_model)
        self._last_report_time = datetime.now(timezone.utc)

    def _save_cache(self) -> None:
        """Save license info to local cache file."""
        if self._license_info is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._license_info.to_dict(), indent=2), encoding="utf-8"
            )
        except OSError:
            logger.warning("Could not save license cache to %s", self._cache_path)

    def _load_cache_or_default(self) -> LicenseInfo:
        """Load cached license info, or return a default if expired/missing."""
        try:
            if self._cache_path.exists():
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                cached = LicenseInfo.from_dict(data)
                age = (datetime.now(timezone.utc) - cached.validated_at).total_seconds()
                if age < GRACE_PERIOD_SECONDS:
                    logger.info(
                        "Using cached license (age=%.1fh, status=%s)",
                        age / 3600,
                        cached.status,
                    )
                    self._license_info = cached
                    return cached
                else:
                    logger.warning(
                        "Cached license expired (age=%.1fd), marking as expired",
                        age / 86400,
                    )
        except (OSError, json.JSONDecodeError, KeyError):
            logger.warning("Could not read license cache")

        # No valid cache — return expired but still allow proxy to work
        self._license_info = LicenseInfo(status="expired")
        return self._license_info
