"""Agno hooks for Headroom integration.

This module provides pre_hooks and post_hooks that can be used with
Agno agents to apply Headroom optimization at the agent level.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from headroom import HeadroomConfig, HeadroomMode

logger = logging.getLogger(__name__)


@dataclass
class HookMetrics:
    """Metrics collected by Headroom pre-hooks.

    Note: These metrics track request counts and timing, not token savings.
    For actual token optimization metrics, use HeadroomAgnoModel which
    wraps the model and provides detailed compression statistics.
    """

    request_id: str
    timestamp: datetime
    # These fields are kept for API compatibility but are always 0
    # Use HeadroomAgnoModel for actual token optimization
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    savings_percent: float = 0.0
    transforms_applied: list[str] = field(default_factory=list)


class HeadroomPreHook:
    """Pre-hook for Agno agents that tracks request metrics.

    This hook runs before the agent sends messages to the LLM,
    providing observability into request patterns. For actual token
    optimization, use HeadroomAgnoModel to wrap your model.

    Note: Agno pre_hooks receive the user input string, not the full
    message history, so optimization is best done at the model level
    using HeadroomAgnoModel.

    Example:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from headroom.integrations.agno import HeadroomPreHook, HeadroomAgnoModel

        # For request tracking only
        pre_hook = HeadroomPreHook()

        # For actual optimization, wrap the model
        model = HeadroomAgnoModel(OpenAIChat(id="gpt-4o"))

        agent = Agent(
            model=model,
            pre_hooks=[pre_hook],
        )

        response = agent.run("Hello!")
        print(f"Requests tracked: {len(pre_hook.metrics_history)}")
        print(f"Tokens saved: {model.total_tokens_saved}")
    """

    def __init__(
        self,
        config: HeadroomConfig | None = None,
        mode: HeadroomMode = HeadroomMode.OPTIMIZE,
        model: str = "gpt-4o",
    ) -> None:
        """Initialize HeadroomPreHook.

        Args:
            config: HeadroomConfig for optimization settings (stored for future use)
            mode: HeadroomMode (stored for future use)
            model: Default model name for token estimation (stored for future use)
        """
        self.config = config or HeadroomConfig()
        self.mode = mode
        self.model = model

        self._metrics_history: list[HookMetrics] = []
        self._total_tokens_saved: int = 0
        self._lock = threading.Lock()  # Thread safety for metrics

    @property
    def total_tokens_saved(self) -> int:
        """Total tokens saved across all calls (thread-safe)."""
        with self._lock:
            return self._total_tokens_saved

    @property
    def metrics_history(self) -> list[HookMetrics]:
        """History of optimization metrics (thread-safe copy)."""
        with self._lock:
            return self._metrics_history.copy()

    def __call__(self, run_input: Any, **kwargs: Any) -> Any:
        """Track the run input.

        This is called by Agno before the LLM processes the input.
        The hook logs the request and returns input unchanged.

        Args:
            run_input: The input from the agent
            **kwargs: Additional arguments (for forward compatibility with Agno)

        Returns:
            The unchanged run_input
        """
        request_id = str(uuid4())
        logger.debug(f"HeadroomPreHook tracking request {request_id}")

        # Record that we processed this input (timing/tracking only)
        metrics = HookMetrics(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc),
        )

        # Thread-safe metrics update
        with self._lock:
            self._metrics_history.append(metrics)

            # Keep only last 100 metrics
            if len(self._metrics_history) > 100:
                self._metrics_history = self._metrics_history[-100:]

        # Return input unchanged (use HeadroomAgnoModel for actual optimization)
        return run_input

    def get_savings_summary(self) -> dict[str, Any]:
        """Get summary of token savings (thread-safe)."""
        with self._lock:
            if not self._metrics_history:
                return {
                    "total_requests": 0,
                    "total_tokens_saved": 0,
                    "average_savings_percent": 0,
                }

            return {
                "total_requests": len(self._metrics_history),
                "total_tokens_saved": self._total_tokens_saved,
                "average_savings_percent": (
                    sum(m.savings_percent for m in self._metrics_history)
                    / len(self._metrics_history)
                    if self._metrics_history
                    else 0
                ),
            }


class HeadroomPostHook:
    """Post-hook for Agno agents that tracks optimization results.

    This hook runs after the agent generates a response,
    tracking metrics and providing observability.

    Example:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from headroom.integrations.agno import HeadroomPostHook

        post_hook = HeadroomPostHook()

        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            post_hooks=[post_hook],
        )

        response = agent.run("Hello!")
        print(f"Requests tracked: {post_hook.total_requests}")
    """

    def __init__(
        self,
        log_level: str = "INFO",
        token_alert_threshold: int | None = None,
    ) -> None:
        """Initialize HeadroomPostHook.

        Args:
            log_level: Logging level ("DEBUG", "INFO", "WARNING")
            token_alert_threshold: Alert if response exceeds this many tokens
        """
        self.log_level = log_level
        self.token_alert_threshold = token_alert_threshold

        self._requests: list[dict[str, Any]] = []
        self._alerts: list[str] = []
        self._lock = threading.Lock()  # Thread safety for requests/alerts

    @property
    def total_requests(self) -> int:
        """Total number of requests tracked."""
        with self._lock:
            return len(self._requests)

    @property
    def alerts(self) -> list[str]:
        """List of alerts triggered (thread-safe copy)."""
        with self._lock:
            return self._alerts.copy()

    def __call__(self, run_output: Any, **kwargs: Any) -> Any:
        """Track the run output.

        This is called by Agno after the LLM generates a response.

        Args:
            run_output: The output from the agent
            **kwargs: Additional arguments (for forward compatibility with Agno)

        Returns:
            The unchanged run_output
        """
        # Record request
        request_info: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc),
            "output_type": type(run_output).__name__,
        }

        # Try to extract token usage if available
        alert_to_add: str | None = None
        if hasattr(run_output, "metrics"):
            metrics = run_output.metrics
            if hasattr(metrics, "input_tokens"):
                request_info["input_tokens"] = metrics.input_tokens
            if hasattr(metrics, "output_tokens"):
                request_info["output_tokens"] = metrics.output_tokens
            if hasattr(metrics, "total_tokens"):
                request_info["total_tokens"] = metrics.total_tokens

                # Check alert threshold
                if self.token_alert_threshold and metrics.total_tokens > self.token_alert_threshold:
                    alert_to_add = (
                        f"Token alert: {metrics.total_tokens} tokens exceeds "
                        f"threshold {self.token_alert_threshold}"
                    )

        # Try to get content length
        if hasattr(run_output, "content") and run_output.content:
            request_info["content_length"] = len(run_output.content)

        # Thread-safe update of requests and alerts
        with self._lock:
            self._requests.append(request_info)

            # Keep only last 1000 requests
            if len(self._requests) > 1000:
                self._requests = self._requests[-1000:]

            if alert_to_add:
                self._alerts.append(alert_to_add)

        # Log alert outside of lock to avoid holding lock during I/O
        if alert_to_add:
            logger.warning(alert_to_add)

        if self.log_level in ("DEBUG", "INFO"):
            logger.log(
                logging.DEBUG if self.log_level == "DEBUG" else logging.INFO,
                f"Agno request completed: {request_info}",
            )

        # Return output unchanged
        return run_output

    def get_summary(self) -> dict[str, Any]:
        """Get summary of tracked requests (thread-safe)."""
        with self._lock:
            if not self._requests:
                return {
                    "total_requests": 0,
                    "total_tokens": 0,
                    "alerts": len(self._alerts),
                }

            total_tokens = sum(r.get("total_tokens", 0) for r in self._requests)

            return {
                "total_requests": len(self._requests),
                "total_tokens": total_tokens,
                "average_tokens": total_tokens / len(self._requests) if self._requests else 0,
                "alerts": len(self._alerts),
            }

    def reset(self) -> None:
        """Reset all tracked metrics (thread-safe)."""
        with self._lock:
            self._requests = []
            self._alerts = []


def create_headroom_hooks(
    config: HeadroomConfig | None = None,
    mode: HeadroomMode = HeadroomMode.OPTIMIZE,
    model: str = "gpt-4o",
    log_level: str = "INFO",
    token_alert_threshold: int | None = None,
) -> tuple[HeadroomPreHook, HeadroomPostHook]:
    """Create a pair of Headroom hooks for Agno agents.

    This is a convenience function to create both pre and post hooks
    with consistent configuration.

    Args:
        config: HeadroomConfig for optimization settings
        mode: HeadroomMode (AUDIT, OPTIMIZE, or SIMULATE)
        model: Default model name for token estimation
        log_level: Logging level for post-hook
        token_alert_threshold: Alert threshold for post-hook

    Returns:
        Tuple of (pre_hook, post_hook)

    Example:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from headroom.integrations.agno import create_headroom_hooks

        pre_hook, post_hook = create_headroom_hooks(
            token_alert_threshold=10000,
        )

        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            pre_hooks=[pre_hook],
            post_hooks=[post_hook],
        )
    """
    pre_hook = HeadroomPreHook(config=config, mode=mode, model=model)
    post_hook = HeadroomPostHook(
        log_level=log_level,
        token_alert_threshold=token_alert_threshold,
    )
    return pre_hook, post_hook
