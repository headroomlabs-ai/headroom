"""JSONL file storage implementation for Headroom SDK."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import RequestMetrics
from ..utils import format_timestamp, parse_timestamp
from .base import Storage


class JSONLStorage(Storage):
    """JSONL file-based metrics storage."""

    def __init__(self, file_path: str):
        """
        Initialize JSONL storage.

        Args:
            file_path: Path to JSONL file.
        """
        self.file_path = file_path
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create file and parent directories if they don't exist."""
        path = Path(self.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()

    def _metrics_to_dict(self, metrics: RequestMetrics) -> dict[str, Any]:
        """Convert RequestMetrics to serializable dict."""
        return {
            "id": metrics.request_id,
            "timestamp": format_timestamp(metrics.timestamp),
            "model": metrics.model,
            "stream": metrics.stream,
            "mode": metrics.mode,
            "tokens_input_before": metrics.tokens_input_before,
            "tokens_input_after": metrics.tokens_input_after,
            "tokens_output": metrics.tokens_output,
            "block_breakdown": metrics.block_breakdown,
            "waste_signals": metrics.waste_signals,
            "stable_prefix_hash": metrics.stable_prefix_hash,
            "cache_alignment_score": metrics.cache_alignment_score,
            "cached_tokens": metrics.cached_tokens,
            "transforms_applied": metrics.transforms_applied,
            "tool_units_dropped": metrics.tool_units_dropped,
            "turns_dropped": metrics.turns_dropped,
            "messages_hash": metrics.messages_hash,
            "error": metrics.error,
        }

    def _dict_to_metrics(self, data: dict[str, Any]) -> RequestMetrics:
        """Convert dict to RequestMetrics."""
        return RequestMetrics(
            request_id=data["id"],
            timestamp=parse_timestamp(data["timestamp"]),
            model=data["model"],
            stream=data["stream"],
            mode=data["mode"],
            tokens_input_before=data["tokens_input_before"],
            tokens_input_after=data["tokens_input_after"],
            tokens_output=data.get("tokens_output"),
            block_breakdown=data.get("block_breakdown", {}),
            waste_signals=data.get("waste_signals", {}),
            stable_prefix_hash=data.get("stable_prefix_hash", ""),
            cache_alignment_score=data.get("cache_alignment_score", 0.0),
            cached_tokens=data.get("cached_tokens"),
            transforms_applied=data.get("transforms_applied", []),
            tool_units_dropped=data.get("tool_units_dropped", 0),
            turns_dropped=data.get("turns_dropped", 0),
            messages_hash=data.get("messages_hash", ""),
            error=data.get("error"),
        )

    def save(self, metrics: RequestMetrics) -> None:
        """Save request metrics."""
        data = self._metrics_to_dict(metrics)
        with open(self.file_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def get(self, request_id: str) -> RequestMetrics | None:
        """Get metrics by request ID."""
        for metrics in self.iter_all():
            if metrics.request_id == request_id:
                return metrics
        return None

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        mode: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RequestMetrics]:
        """Query metrics with filters."""
        results: list[RequestMetrics] = []
        skipped = 0

        for metrics in self.iter_all():
            # Apply filters
            if start_time is not None and metrics.timestamp < start_time:
                continue
            if end_time is not None and metrics.timestamp > end_time:
                continue
            if model is not None and metrics.model != model:
                continue
            if mode is not None and metrics.mode != mode:
                continue

            # Handle offset
            if skipped < offset:
                skipped += 1
                continue

            results.append(metrics)

            # Handle limit
            if len(results) >= limit:
                break

        # Sort by timestamp descending
        results.sort(key=lambda m: m.timestamp, reverse=True)
        return results

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        mode: str | None = None,
    ) -> int:
        """Count metrics matching filters."""
        count = 0

        for metrics in self.iter_all():
            if start_time is not None and metrics.timestamp < start_time:
                continue
            if end_time is not None and metrics.timestamp > end_time:
                continue
            if model is not None and metrics.model != model:
                continue
            if mode is not None and metrics.mode != mode:
                continue
            count += 1

        return count

    def iter_all(self) -> Iterator[RequestMetrics]:
        """Iterate over all stored metrics."""
        if not Path(self.file_path).exists():
            return

        with open(self.file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield self._dict_to_metrics(data)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

    def get_summary_stats(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get summary statistics."""
        total_requests = 0
        total_tokens_before = 0
        total_tokens_after = 0
        total_cache_alignment = 0.0
        audit_count = 0
        optimize_count = 0

        for metrics in self.iter_all():
            if start_time is not None and metrics.timestamp < start_time:
                continue
            if end_time is not None and metrics.timestamp > end_time:
                continue

            total_requests += 1
            total_tokens_before += metrics.tokens_input_before
            total_tokens_after += metrics.tokens_input_after
            total_cache_alignment += metrics.cache_alignment_score

            if metrics.mode == "audit":
                audit_count += 1
            elif metrics.mode == "optimize":
                optimize_count += 1

        total_tokens_saved = total_tokens_before - total_tokens_after
        avg_tokens_saved = total_tokens_saved / total_requests if total_requests > 0 else 0
        avg_cache_alignment = total_cache_alignment / total_requests if total_requests > 0 else 0

        return {
            "total_requests": total_requests,
            "total_tokens_before": total_tokens_before,
            "total_tokens_after": total_tokens_after,
            "total_tokens_saved": total_tokens_saved,
            "avg_tokens_saved": avg_tokens_saved,
            "avg_cache_alignment": avg_cache_alignment,
            "audit_count": audit_count,
            "optimize_count": optimize_count,
        }

    def close(self) -> None:
        """No-op for file storage."""
        pass
