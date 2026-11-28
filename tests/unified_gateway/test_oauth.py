from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

from headroom.proxy.gateway.oauth import (
    BrowserAuthorizationTransaction,
    OAuthReplayRejected,
    OAuthValidationError,
)


def _transaction() -> BrowserAuthorizationTransaction:
    return BrowserAuthorizationTransaction.create(
        authorization_endpoint="https://issuer.example/authorize",
        issuer="https://issuer.example",
        client_id="headroom-test",
        redirect_uri="http://127.0.0.1:43119/callback",
        scopes=("inference",),
        allowed_account="approved-account",
        lifetime_seconds=60,
        now=100.0,
    )


def test_browser_transaction_uses_pkce_s256_and_exact_callback() -> None:
    transaction = _transaction()
    query = parse_qs(urlsplit(transaction.authorization_url).query)
    assert query["code_challenge_method"] == ["S256"]
    assert query["state"] == [transaction.state]
    assert query["redirect_uri"] == ["http://127.0.0.1:43119/callback"]
    assert "code_challenge" in query


def test_callback_state_is_consumed_once() -> None:
    transaction = _transaction()
    credential = transaction.consume(
        callback_uri=(
            "http://127.0.0.1:43119/callback"
            f"?code=code-1&state={transaction.state}&issuer=https%3A%2F%2Fissuer.example"
            "&account=approved-account"
        ),
        now=101.0,
    )
    assert credential.account_ref == "approved-account"
    assert credential.authorization_code == "code-1"
    with pytest.raises(OAuthReplayRejected):
        transaction.consume(callback_uri="http://127.0.0.1:43119/callback", now=102.0)


@pytest.mark.parametrize(
    "callback_uri",
    [
        "http://127.0.0.1.evil.test:43119/callback?code=x&state=x",
        "http://127.0.0.1:43119/other?code=x&state=x",
        "http://127.0.0.1:43119/callback?code=x&state=x&issuer=https://lookalike.example&account=approved-account",
    ],
)
def test_callback_rejects_lookalikes_and_mismatched_binding(callback_uri: str) -> None:
    with pytest.raises(OAuthValidationError):
        _transaction().consume(callback_uri=callback_uri, now=101.0)
