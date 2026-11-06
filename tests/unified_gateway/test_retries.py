from __future__ import annotations

import pytest

from headroom.proxy.gateway.routing import ProviderContract, RetryDecision, TransportFailure


def test_retry_decision_obeys_attempt_deadline_and_proven_acceptance_contract():
    assert hasattr(RetryDecision, "decide"), "bounded retry decision missing"
    from headroom.proxy.gateway.config import RetryConfig

    policy = RetryConfig(
        max_attempts=3,
        max_retry_after_seconds=5,
        base_backoff_seconds=0.1,
        ambiguous_commit="never",
        after_output="never",
    )
    contract = ProviderContract(
        max_attempts=3,
        retryable_failures=frozenset({"connect", "rate_limit"}),
        max_retry_after=5,
        rejected_failures=frozenset({"rate_limit"}),
    )

    def decide(kind="connect", exposure="unsent", count=1, deadline=10, retry_after=None):
        return RetryDecision.decide(
            TransportFailure(kind, retry_after),
            exposure,
            contract,
            attempt_count=count,
            deadline=deadline,
            now=0,
            policy=policy,
        )

    assert decide().allowed and decide().delay == 0.1
    assert not decide(count=3).allowed
    assert not decide(deadline=0.05).allowed
    assert not decide(exposure="acceptance_unknown").allowed
    assert not decide(exposure="output_exposed").allowed
    assert not decide(kind="read_timeout").allowed
    assert decide(kind="rate_limit", exposure="proven_rejected", retry_after=2).delay == 2
    assert not decide(kind="rate_limit", exposure="proven_rejected", retry_after=6).allowed
    assert not decide(retry_after=float("nan")).allowed
    assert not decide(retry_after="Wed, 21 Oct 2099 07:28:00 GMT").allowed


@pytest.mark.parametrize("exposure", ["ambiguous_acceptance", "first_byte", "tool_event"])
def test_exposed_or_ambiguous_operation_is_never_retried(exposure: str) -> None:
    assert (
        RetryDecision.classify(
            TransportFailure(kind="connect", retry_after=None),
            exposure,
            ProviderContract(max_attempts=2, retryable_failures=frozenset({"connect"})),
        )
        is False
    )


def test_precommit_retry_honors_contract_and_retry_after_bound() -> None:
    contract = ProviderContract(
        max_attempts=2,
        retryable_failures=frozenset({"connect", "rate_limit"}),
        max_retry_after=5.0,
    )

    assert RetryDecision.classify(
        TransportFailure(kind="connect", retry_after=None), "none", contract
    )
    assert RetryDecision.classify(
        TransportFailure(kind="rate_limit", retry_after=4.0), "none", contract
    )
    assert not RetryDecision.classify(
        TransportFailure(kind="rate_limit", retry_after=6.0), "none", contract
    )
