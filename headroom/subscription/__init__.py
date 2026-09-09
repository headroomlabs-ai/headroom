"""Subscription window tracking for Anthropic Claude Code accounts."""

from headroom.subscription.base import (
    QuotaTracker,
    QuotaTrackerRegistry,
    get_quota_registry,
    reset_quota_registry,
)
from headroom.subscription.client import SubscriptionClient, read_cached_oauth_token
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
    "get_quota_registry",
    "get_subscription_tracker",
    "read_cached_oauth_token",
    "reset_quota_registry",
    "shutdown_subscription_tracker",
]
