"""AWS SigV4 re-signing for the Bedrock ``InvokeModel`` passthrough.

Claude Code (and the AWS SDK/CLI) sign every Bedrock runtime request with
SigV4 — and the signature covers a SHA-256 hash of the request body. Headroom's
Bedrock handler rewrites that body to compress it, which invalidates the
caller's signature: forwarding it to raw AWS then fails with a 403
``InvalidSignatureException``.

This module re-signs the *post-compression* body so the bytes AWS receives are
the bytes that were signed. It is the Python analogue of the Rust proxy's
native SigV4 surface (``docs/bedrock.md``), and it lets ``headroom wrap claude``
support ``CLAUDE_CODE_USE_BEDROCK=1`` direct-to-AWS — no re-signing gateway
(LiteLLM / LocalStack) required.

The credential *provider* is resolved once, lazily, via the standard ``boto3``
credential chain (env vars, shared config/credentials profile, SSO cache, IMDS,
ECS/EKS role). The chain is the same one ``aws`` and Claude Code already use, so
if Claude Code can reach Bedrock without Headroom, the signer can too. Each
individual signing operation then takes a frozen snapshot of that provider — see
:meth:`BedrockSigner._frozen_credentials`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from botocore.credentials import Credentials, ReadOnlyCredentials

logger = logging.getLogger("headroom.proxy")

# Bedrock runtime signs under this service name.
_SERVICE = "bedrock"

# Headers that must never be copied from the inbound request into the set we
# re-sign: the host changes (we forward to the regional Bedrock endpoint), the
# old signature/date are stale, and content-length/encoding describe the
# pre-compression body. SigV4 recomputes the ones it needs.
_DROP_FOR_SIGNING = frozenset(
    {
        "host",
        "authorization",
        "x-amz-date",
        "x-amz-security-token",
        "x-amz-content-sha256",
        "content-length",
        "content-encoding",
        "accept-encoding",
        "connection",
    }
)


class BedrockSigningError(RuntimeError):
    """Raised when credentials cannot be resolved or signing fails."""


class BedrockSigner:
    """Re-signs Bedrock InvokeModel requests with SigV4 after compression.

    One instance per proxy process. The credential provider is resolved on first
    use and cached; ``botocore``'s credential objects refresh themselves for the
    refreshable sources (SSO, assume-role, IMDS), so a long-lived proxy keeps
    working across credential rotation without re-resolving.
    """

    def __init__(self, region: str, profile: str | None = None) -> None:
        self._region = region
        self._profile = profile
        self._credentials: Credentials | None = None

    @property
    def region(self) -> str:
        return self._region

    def endpoint_base(self) -> str:
        """Regional Bedrock runtime host, e.g. ``https://bedrock-runtime.us-west-2.amazonaws.com``."""
        return f"https://bedrock-runtime.{self._region}.amazonaws.com"

    def control_endpoint_base(self) -> str:
        """Regional Bedrock control-plane host used by inference-profile APIs."""
        return f"https://bedrock.{self._region}.amazonaws.com"

    def _resolve_credentials(self) -> Credentials:
        if self._credentials is not None:
            return self._credentials
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - boto3 is a hard dep
            raise BedrockSigningError(
                "boto3 is required for Bedrock SigV4 signing. "
                'Install with: pip install "headroom-ai[bedrock]"'
            ) from exc

        session = boto3.Session(profile_name=self._profile, region_name=self._region)
        creds = session.get_credentials()
        if creds is None:
            raise BedrockSigningError(
                "No AWS credentials found for Bedrock signing. Configure the "
                "default credential chain (env vars, ~/.aws/credentials, SSO, "
                "or an instance/task role)."
            )
        self._credentials = creds
        return creds

    def _frozen_credentials(self) -> ReadOnlyCredentials:
        """Snapshot the cached provider's credentials for one signing operation.

        ``SigV4Auth`` reads ``access_key``, ``secret_key`` and ``token`` as three
        separate attribute lookups. On a ``RefreshableCredentials`` provider
        (SSO, assume-role, IMDS — the long-lived cases this signer exists for)
        each of those lookups can trigger a refresh, so a rotation landing
        mid-signature yields a request mixing generations: an old
        ``X-Amz-Security-Token`` with a new ``Credential`` access key and signing
        secret. AWS rejects that with 403 ``InvalidSignatureException``, and it
        is unreproducible after the fact.

        ``get_frozen_credentials()`` takes all three under the provider's
        refresh lock and returns an immutable ``ReadOnlyCredentials``, which is
        what botocore's own ``RequestSigner.get_auth_instance`` hands to the auth
        class. Called per request, so the *next* request still observes freshly
        refreshed credentials — the caching is of the provider, not of a
        snapshot.
        """
        return self._resolve_credentials().get_frozen_credentials()

    def sign(
        self,
        *,
        url: str,
        body: bytes,
        inbound_headers: dict[str, str],
        method: str = "POST",
    ) -> dict[str, str]:
        """Return outbound headers carrying a fresh SigV4 signature for ``body``.

        Args:
            url: The absolute regional Bedrock URL the request will be sent to.
            body: The exact bytes that will be written on the wire (post
                compression). The signature hashes these bytes, so the caller
                MUST forward this same ``body`` unchanged.
            inbound_headers: The original request headers; non-hop, non-auth
                entries (notably ``content-type`` and ``anthropic-*`` /
                ``x-amzn-bedrock-*`` passthroughs) are preserved.

        Raises:
            BedrockSigningError: if credentials cannot be resolved.
        """
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        # One immutable snapshot per signing operation, so all three credential
        # fields in the signature come from the same generation even if the
        # provider refreshes concurrently (see _frozen_credentials).
        frozen = self._frozen_credentials()

        # Preserve everything the caller sent except the headers SigV4 will
        # recompute or that describe the stale (pre-compression) body. Keeping
        # content-type and any x-amzn-bedrock-* / anthropic-* headers matters:
        # Bedrock guardrail headers and the anthropic beta headers travel here.
        passthrough = {
            k: v for k, v in inbound_headers.items() if k.lower() not in _DROP_FOR_SIGNING
        }
        host = urlsplit(url).netloc
        passthrough["host"] = host

        aws_request = AWSRequest(method=method, url=url, data=body, headers=passthrough)
        SigV4Auth(frozen, _SERVICE, self._region).add_auth(aws_request)
        return dict(aws_request.headers.items())
