from __future__ import annotations

import asyncio

import pytest

from headroom.proxy.gateway.oauth import (
    DeviceAuthorizationTransaction,
    DeviceFlowCancelled,
    DeviceFlowError,
)


@pytest.mark.asyncio
async def test_device_flow_honors_pending_and_slow_down() -> None:
    replies = iter(
        [
            {"error": "authorization_pending"},
            {"error": "slow_down"},
            {"access_token": "opaque", "account_ref": "approved-account"},
        ]
    )
    delays: list[float] = []

    async def poll() -> dict[str, str]:
        return next(replies)

    async def delay(seconds: float) -> None:
        delays.append(seconds)

    transaction = DeviceAuthorizationTransaction(
        poller=poll, interval_seconds=1.0, deadline=20.0, clock=lambda: 0.0, sleep=delay
    )
    credential = await transaction.poll(asyncio.Event())
    assert credential.account_ref == "approved-account"
    assert delays == [1.0, 6.0]


@pytest.mark.asyncio
async def test_device_flow_cancellation_stops_before_polling() -> None:
    called = False

    async def poll() -> dict[str, str]:
        nonlocal called
        called = True
        return {}

    cancelled = asyncio.Event()
    cancelled.set()
    transaction = DeviceAuthorizationTransaction(
        poller=poll, interval_seconds=1.0, deadline=20.0, clock=lambda: 0.0
    )
    with pytest.raises(DeviceFlowCancelled):
        await transaction.poll(cancelled)
    assert called is False


@pytest.mark.asyncio
async def test_device_flow_denial_is_terminal() -> None:
    async def poll() -> dict[str, str]:
        return {"error": "access_denied"}

    transaction = DeviceAuthorizationTransaction(
        poller=poll, interval_seconds=1.0, deadline=20.0, clock=lambda: 0.0
    )
    with pytest.raises(DeviceFlowError, match="access_denied"):
        await transaction.poll(asyncio.Event())
