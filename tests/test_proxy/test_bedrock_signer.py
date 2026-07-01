"""Tests for the AWS SigV4 re-signer used by the Bedrock direct-to-AWS path.

These cover ``headroom.proxy.bedrock_signer.BedrockSigner`` in isolation — no
proxy app, no ``headroom._core``. botocore is the only dependency, and it is a
hard dep of the package.

What matters for correctness:

1. The endpoint host is the regional Bedrock runtime host.
2. The signature is computed over the *exact body bytes* we pass — change the
   body, get a different signature (this is the whole point: we re-sign after
   compression mutates the body).
3. Stale signing headers (Authorization, X-Amz-Date, the old Host) are replaced,
   not carried forward.
4. Non-auth passthrough headers (content-type, anthropic-beta, Bedrock guardrail
   headers) survive into the signed set.
5. Missing credentials raise a clear error rather than forwarding unsigned.
"""

from __future__ import annotations

import pytest

botocore = pytest.importorskip("botocore")

from botocore.credentials import Credentials  # noqa: E402

from headroom.proxy.bedrock_signer import BedrockSigner, BedrockSigningError  # noqa: E402

_FAKE_CREDS = Credentials(access_key="AKIDEXAMPLE", secret_key="secret", token="sess-token")
_REGION = "us-west-2"
_URL = (
    f"https://bedrock-runtime.{_REGION}.amazonaws.com"
    "/model/us.anthropic.claude-opus-4-8-v1%3A0/invoke"
)


def _signer(creds: Credentials | None = _FAKE_CREDS) -> BedrockSigner:
    s = BedrockSigner(region=_REGION)
    # Inject creds so we never touch the real AWS credential chain in unit tests.
    s._credentials = creds  # type: ignore[attr-defined]
    return s


def test_endpoint_base_is_regional_host():
    assert (
        BedrockSigner(region="eu-central-1").endpoint_base()
        == "https://bedrock-runtime.eu-central-1.amazonaws.com"
    )


def test_sign_sets_sigv4_authorization_and_host():
    headers = _signer().sign(
        url=_URL,
        body=b'{"messages":[],"max_tokens":8}',
        inbound_headers={"content-type": "application/json"},
    )
    auth = headers.get("Authorization", "")
    assert auth.startswith("AWS4-HMAC-SHA256 ")
    assert f"/{_REGION}/bedrock/aws4_request" in auth
    assert "X-Amz-Date" in headers
    assert "X-Amz-Security-Token" in headers  # session token threaded through
    assert headers["host"] == f"bedrock-runtime.{_REGION}.amazonaws.com"


def test_signature_covers_body_bytes():
    """Different body → different signature. This is why we sign AFTER compression."""
    sign = _signer().sign
    a = sign(url=_URL, body=b'{"a":1}', inbound_headers={"content-type": "application/json"})
    b = sign(url=_URL, body=b'{"a":2}', inbound_headers={"content-type": "application/json"})
    assert a["Authorization"] != b["Authorization"]


def test_stale_signing_headers_are_dropped_not_reused():
    """The inbound request carried a SigV4 signature over the OLD body; it must
    not leak into the re-signed set."""
    headers = _signer().sign(
        url=_URL,
        body=b'{"messages":[]}',
        inbound_headers={
            "content-type": "application/json",
            "Authorization": "AWS4-HMAC-SHA256 Credential=STALE/old/bedrock/aws4_request",
            "X-Amz-Date": "20200101T000000Z",
            "host": "bedrock-runtime.us-east-1.amazonaws.com",  # wrong region!
        },
    )
    # Re-signed, not the stale value.
    assert "STALE" not in headers["Authorization"]
    assert headers["X-Amz-Date"] != "20200101T000000Z"
    # Host points at the URL we actually forward to.
    assert headers["host"] == f"bedrock-runtime.{_REGION}.amazonaws.com"


def test_passthrough_headers_survive():
    """Guardrail + anthropic beta headers must reach Bedrock (and be signed)."""
    headers = _signer().sign(
        url=_URL,
        body=b"{}",
        inbound_headers={
            "content-type": "application/json",
            "anthropic-beta": "context-1m-2025-08-07",
            "x-amzn-bedrock-guardrailidentifier": "gr-123",
        },
    )
    assert headers["anthropic-beta"] == "context-1m-2025-08-07"
    assert headers["x-amzn-bedrock-guardrailidentifier"] == "gr-123"


def test_signed_headers_list_includes_passthrough():
    """A passthrough header that SigV4 signs must appear in SignedHeaders, or AWS
    rejects the request. Guards against signing a header set that omits it."""
    headers = _signer().sign(
        url=_URL,
        body=b"{}",
        inbound_headers={
            "content-type": "application/json",
            "anthropic-beta": "context-1m-2025-08-07",
        },
    )
    auth = headers["Authorization"]
    signed = auth.split("SignedHeaders=", 1)[1].split(",", 1)[0]
    assert "anthropic-beta" in signed
    assert "host" in signed


def test_missing_credentials_raises():
    s = BedrockSigner(region=_REGION)
    # Force the boto3 chain to yield nothing.
    import boto3

    class _NoCredSession:
        def __init__(self, *a, **k):
            pass

        def get_credentials(self):
            return None

    orig = boto3.Session
    boto3.Session = _NoCredSession  # type: ignore[assignment]
    try:
        with pytest.raises(BedrockSigningError):
            s.sign(url=_URL, body=b"{}", inbound_headers={})
    finally:
        boto3.Session = orig  # type: ignore[assignment]
