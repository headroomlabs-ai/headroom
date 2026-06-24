"""Subscription window tracking for Anthropic Claude Code accounts and Codex rate limits."""

from headroom.subscription.base import (
    QuotaTracker,
    QuotaTrackerRegistry,
    get_quota_registry,
    reset_quota_registry,
)
from headroom.subscription.client import SubscriptionClient, read_cached_oauth_token
from headroom.subscription.codex_rate_limits import (
    CodexCreditsSnapshot,
    CodexRateLimitSnapshot,
    CodexRateLimitState,
    CodexRateLimitWindow,
    get_codex_rate_limit_state,
    parse_codex_rate_limits,
)
from headroom.subscription.copilot_quota import (
    CopilotQuotaCategory,
    CopilotQuotaSnapshot,
    CopilotQuotaState,
    discover_github_token,
    get_copilot_quota_tracker,
    parse_copilot_quota,
)
from headroom.subscription.models import (
    ExtraUsage,
    HeadroomContribution,
    RateLimitWindow,
    SubscriptionSnapshot,
    SubscriptionState,
    WindowDiscrepancy,
    WindowTokens,
)
from headroom.subscription.tracker import (
    SubscriptionTracker,
    configure_subscription_tracker,
    get_subscription_tracker,
    shutdown_subscription_tracker,
)

__all__ = [
    "CodexCreditsSnapshot",
    "CodexRateLimitSnapshot",
    "CodexRateLimitState",
    "CodexRateLimitWindow",
    "CopilotQuotaCategory",
    "CopilotQuotaSnapshot",
    "CopilotQuotaState",
    "ExtraUsage",
    "HeadroomContribution",
    "QuotaTracker",
    "QuotaTrackerRegistry",
    "RateLimitWindow",
    "SubscriptionClient",
    "SubscriptionSnapshot",
    "SubscriptionState",
    "SubscriptionTracker",
    "WindowDiscrepancy",
    "WindowTokens",
    "configure_subscription_tracker",
    "discover_github_token",
    "get_codex_rate_limit_state",
    "get_copilot_quota_tracker",
    "get_quota_registry",
    "get_subscription_tracker",
    "parse_codex_rate_limits",
    "parse_copilot_quota",
    "read_cached_oauth_token",
    "reset_quota_registry",
    "shutdown_subscription_tracker",
]
