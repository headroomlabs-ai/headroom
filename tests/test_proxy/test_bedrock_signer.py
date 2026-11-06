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
6. Every field of one signature comes from a single credential generation, even
   when the underlying refreshable provider rotates between reads.
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


def test_control_endpoint_base_is_regional_host():
    assert (
        BedrockSigner(region="eu-central-1").control_endpoint_base()
        == "https://bedrock.eu-central-1.amazonaws.com"
    )


def test_sign_supports_control_plane_get():
    url = f"https://bedrock.{_REGION}.amazonaws.com/inference-profiles?maxResults=5"
    headers = _signer().sign(
        method="GET", url=url, body=b"", inbound_headers={"accept": "application/json"}
    )
    assert headers["host"] == f"bedrock.{_REGION}.amazonaws.com"
    assert f"/{_REGION}/bedrock/aws4_request" in headers["Authorization"]


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


class _RotatingCredentials:
    """A refreshable provider that rotates on every *direct* attribute read.

    ``RefreshableCredentials`` refreshes lazily when its cached credentials near
    expiry, and ``SigV4Auth`` reads ``access_key``, ``secret_key`` and ``token``
    as three separate lookups — so a refresh can land between them. This
    exaggerates that window to every read, while ``get_frozen_credentials()``
    returns one coherent generation the way botocore's does. A signer that hands
    the live provider to ``SigV4Auth`` therefore mixes generations here; one that
    snapshots first cannot.
    """

    def __init__(self) -> None:
        self.generation = 0
        self.frozen_calls = 0

    def _advance(self) -> int:
        self.generation += 1
        return self.generation

    @property
    def access_key(self) -> str:
        return f"AKIDGEN{self._advance()}"

    @property
    def secret_key(self) -> str:
        return f"secret-gen-{self._advance()}"

    @property
    def token(self) -> str:
        return f"token-gen-{self._advance()}"

    def get_frozen_credentials(self):
        from botocore.credentials import ReadOnlyCredentials

        self.frozen_calls += 1
        gen = self._advance()
        return ReadOnlyCredentials(f"AKIDGEN{gen}", f"secret-gen-{gen}", f"token-gen-{gen}")


def _signed_access_key(headers: dict[str, str]) -> str:
    """The access key in the ``Credential=`` scope of an Authorization header."""
    return headers["Authorization"].split("Credential=", 1)[1].split("/", 1)[0]


@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("POST", _URL, b'{"messages":[],"max_tokens":8}'),
        ("GET", f"https://bedrock.{_REGION}.amazonaws.com/inference-profiles", b""),
    ],
    ids=["runtime-post", "discovery-get"],
)
def test_one_credential_generation_per_signature(method: str, url: str, body: bytes):
    """A rotation landing mid-signature must not mix credential generations.

    The failure this guards against is an old ``X-Amz-Security-Token`` paired
    with a newer ``Credential`` access key and signing secret: three fields from
    three generations, which authenticate as no credential set at all and get a
    403 ``InvalidSignatureException`` that is gone by the time you look.
    """
    creds = _RotatingCredentials()
    headers = _signer(creds).sign(  # type: ignore[arg-type]
        method=method, url=url, body=body, inbound_headers={"content-type": "application/json"}
    )

    # Exactly one snapshot, and no direct reads of the rotating attributes —
    # so all three signed fields provably share generation 1.
    assert creds.frozen_calls == 1
    assert creds.generation == 1
    assert _signed_access_key(headers) == "AKIDGEN1"
    assert headers["X-Amz-Security-Token"] == "token-gen-1"


def test_next_request_observes_refreshed_credentials():
    """The provider is cached, the snapshot is not: request N+1 re-freezes.

    Snapshotting once at resolve time would pin a long-lived proxy to expired
    credentials, which is the reason the refreshable provider is kept.
    """
    creds = _RotatingCredentials()
    signer = _signer(creds)  # type: ignore[arg-type]
    inbound = {"content-type": "application/json"}

    first = signer.sign(url=_URL, body=b'{"a":1}', inbound_headers=inbound)
    second = signer.sign(url=_URL, body=b'{"a":2}', inbound_headers=inbound)

    assert creds.frozen_calls == 2
    assert _signed_access_key(first) == "AKIDGEN1"
    assert first["X-Amz-Security-Token"] == "token-gen-1"
    # Rotated forward, and still internally consistent.
    assert _signed_access_key(second) == "AKIDGEN2"
    assert second["X-Amz-Security-Token"] == "token-gen-2"


def test_non_refreshable_credentials_still_sign():
    """A static ``Credentials`` provider also implements the frozen contract."""
    headers = _signer().sign(url=_URL, body=b"{}", inbound_headers={})
    assert _signed_access_key(headers) == "AKIDEXAMPLE"
    assert headers["X-Amz-Security-Token"] == "sess-token"


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
