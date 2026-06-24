"""LangSmith integration for Headroom compression metrics.

This module provides HeadroomLangSmithCallbackHandler, a LangChain callback
handler that adds Headroom compression metrics to LangSmith traces.

When used with HeadroomChatModel, it automatically captures:
- Tokens before/after optimization
- Savings percentage
- Transforms applied
- Per-request compression details

Example:
    import os
    from langchain_openai import ChatOpenAI
    from headroom.integrations import (
        HeadroomChatModel,
        HeadroomLangSmithCallbackHandler,
    )

    # Enable LangSmith tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = "..."

    # Create handler
    handler = HeadroomLangSmithCallbackHandler()

    # Use with HeadroomChatModel
    llm = HeadroomChatModel(
        ChatOpenAI(model="gpt-4o"),
        callbacks=[handler],
    )

    # Traces will include headroom.* metadata
    response = llm.invoke("Hello!")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

# LangChain imports - these are optional dependencies
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore[misc,assignment]
    LLMResult = object  # type: ignore[misc,assignment]

# LangSmith imports - optional
try:
    from langsmith import Client as LangSmithClient

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    LangSmithClient = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


def _check_langchain_available() -> None:
    """Raise ImportError if LangChain is not installed."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for this integration. "
            "Install with: pip install headroom[langchain] "
            "or: pip install langchain-core"
        )


@dataclass
class PendingMetrics:
    """Metrics pending attachment to a LangSmith run."""

    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float
    transforms_applied: list[str]
    timestamp: datetime = field(default_factory=datetime.now)


class HeadroomLangSmithCallbackHandler(BaseCallbackHandler):
    """Callback handler that adds Headroom metrics to LangSmith traces.

    Integrates with LangSmith to provide visibility into context
    optimization within traces. Metrics appear as metadata with
    the `headroom.` prefix.

    Works automatically when:
    1. LANGCHAIN_TRACING_V2=true is set
    2. Used as a callback with HeadroomChatModel
    3. LangSmith API key is configured

    Example:
        from headroom.integrations import (
            HeadroomChatModel,
            HeadroomLangSmithCallbackHandler,
        )

        handler = HeadroomLangSmithCallbackHandler()
        llm = HeadroomChatModel(
            ChatOpenAI(model="gpt-4o"),
            callbacks=[handler],
        )

        response = llm.invoke("Hello!")
        # LangSmith trace now includes:
        # - headroom.tokens_before
        # - headroom.tokens_after
        # - headroom.tokens_saved
        # - headroom.savings_percent
        # - headroom.transforms_applied

    Attributes:
        langsmith_client: LangSmith client for updating runs.
        pending_metrics: Metrics waiting to be attached to runs.
    """

    def __init__(
        self,
        langsmith_client: Any = None,
        auto_update_runs: bool = True,
    ):
        """Initialize HeadroomLangSmithCallbackHandler.

        Args:
            langsmith_client: LangSmith client instance. Auto-creates
                one if not provided and LangSmith is available.
            auto_update_runs: If True, automatically updates LangSmith
                runs with Headroom metadata. Default True.
        """
        _check_langchain_available()

        self._client = langsmith_client
        self._auto_update = auto_update_runs
        self._pending_metrics: dict[str, PendingMetrics] = {}
        self._run_metrics: dict[str, dict[str, Any]] = {}

        # Initialize LangSmith client if available and not provided
        if self._client is None and LANGSMITH_AVAILABLE and auto_update_runs:
            try:
                if os.environ.get("LANGCHAIN_API_KEY"):
                    self._client = LangSmithClient()
            except Exception as e:
                logger.debug(f"Could not initialize LangSmith client: {e}")

    def set_headroom_metrics(
        self,
        run_id: str | UUID,
        tokens_before: int,
        tokens_after: int,
        transforms_applied: list[str] | None = None,
    ) -> None:
        """Set Headroom metrics for a run.

        Call this from HeadroomChatModel after optimization to attach
        metrics to the current run.

        Args:
            run_id: The LangSmith run ID.
            tokens_before: Token count before optimization.
            tokens_after: Token count after optimization.
            transforms_applied: List of transforms that were applied.
        """
        run_id_str = str(run_id)
        tokens_saved = tokens_before - tokens_after
        savings_percent = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0.0

        metrics = PendingMetrics(
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            savings_percent=savings_percent,
            transforms_applied=transforms_applied or [],
        )

        self._pending_metrics[run_id_str] = metrics

        logger.debug(
            f"Headroom metrics set for run {run_id_str}: "
            f"{tokens_before} -> {tokens_after} tokens ({savings_percent:.1f}% saved)"
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when chat model starts.

        Records the run ID for later metric attachment.
        """
        run_id_str = str(run_id)
        # Initialize empty metrics for this run
        self._run_metrics[run_id_str] = {}

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Called when LLM completes.

        Attaches pending Headroom metrics to the LangSmith run.
        """
        run_id_str = str(run_id)

        # Check for pending metrics
        if run_id_str in self._pending_metrics:
            metrics = self._pending_metrics.pop(run_id_str)
            self._attach_metrics_to_run(run_id_str, metrics)

    def _attach_metrics_to_run(self, run_id: str, metrics: PendingMetrics) -> None:
        """Attach Headroom metrics to a LangSmith run.

        Args:
            run_id: The run ID.
            metrics: Metrics to attach.
        """
        metadata = {
            "headroom.tokens_before": metrics.tokens_before,
            "headroom.tokens_after": metrics.tokens_after,
            "headroom.tokens_saved": metrics.tokens_saved,
            "headroom.savings_percent": round(metrics.savings_percent, 2),
            "headroom.transforms_applied": metrics.transforms_applied,
            "headroom.optimization_timestamp": metrics.timestamp.isoformat(),
        }

        # Store in run metrics
        self._run_metrics[run_id] = metadata

        # Update LangSmith run if client available
        if self._client and self._auto_update:
            try:
                self._client.update_run(
                    run_id=run_id,
                    extra={"metadata": metadata},
                )
                logger.debug(f"Updated LangSmith run {run_id} with Headroom metrics")
            except Exception as e:
                logger.debug(f"Could not update LangSmith run: {e}")

    def get_run_metrics(self, run_id: str | UUID) -> dict[str, Any]:
        """Get Headroom metrics for a specific run.

        Args:
            run_id: The run ID.

        Returns:
            Dictionary of headroom.* metrics for the run.
        """
        return self._run_metrics.get(str(run_id), {})

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get all recorded run metrics.

        Returns:
            Dictionary mapping run IDs to their metrics.
        """
        return self._run_metrics.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics across all runs.

        Returns:
            Summary with total runs, tokens saved, etc.
        """
        if not self._run_metrics:
            return {
                "total_runs": 0,
                "total_tokens_saved": 0,
                "average_savings_percent": 0,
            }

        total_saved = sum(m.get("headroom.tokens_saved", 0) for m in self._run_metrics.values())
        savings_percents = [
            m.get("headroom.savings_percent", 0) for m in self._run_metrics.values()
        ]

        return {
            "total_runs": len(self._run_metrics),
            "total_tokens_saved": total_saved,
            "average_savings_percent": (
                sum(savings_percents) / len(savings_percents) if savings_percents else 0
            ),
        }

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self._pending_metrics.clear()
        self._run_metrics.clear()


def is_langsmith_available() -> bool:
    """Check if LangSmith is available and configured.

    Returns:
        True if LangSmith is installed and API key is set.
    """
    return LANGSMITH_AVAILABLE and bool(os.environ.get("LANGCHAIN_API_KEY"))


def is_langsmith_tracing_enabled() -> bool:
    """Check if LangSmith tracing is enabled.

    Returns:
        True if LANGCHAIN_TRACING_V2 is set to "true".
    """
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
