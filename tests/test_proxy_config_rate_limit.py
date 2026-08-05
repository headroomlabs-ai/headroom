"""ProxyConfig must reject a 0 requests-per-minute limit when rate limiting is on
(it would divide by zero in the token-bucket wait computation and 500 every
request), while leaving it inert when limiting is off."""

from __future__ import annotations

import pytest

from headroom.proxy.models import ProxyConfig


def test_zero_rpm_with_limiting_enabled_is_rejected():
    with pytest.raises(ValueError, match="rate_limit_requests_per_minute must be >= 1"):
        ProxyConfig(rate_limit_enabled=True, rate_limit_requests_per_minute=0)


def test_negative_rpm_with_limiting_enabled_is_rejected():
    with pytest.raises(ValueError, match="rate_limit_requests_per_minute must be >= 1"):
        ProxyConfig(rate_limit_enabled=True, rate_limit_requests_per_minute=-5)


def test_zero_rpm_is_inert_when_limiting_disabled():
    # Limiting off -> the bucket is never consulted, so a 0 limit is harmless.
    config = ProxyConfig(rate_limit_enabled=False, rate_limit_requests_per_minute=0)
    assert config.rate_limit_requests_per_minute == 0


def test_valid_rpm_is_accepted():
    config = ProxyConfig(rate_limit_enabled=True, rate_limit_requests_per_minute=60)
    assert config.rate_limit_requests_per_minute == 60
