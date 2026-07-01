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

Credentials are resolved once, lazily, via the standard ``boto3`` credential
chain (env vars, shared config/credentials profile, SSO cache, IMDS, ECS/EKS
role). The chain is the same one ``aws`` and Claude Code already use, so if
Claude Code can reach Bedrock without Headroom, the signer can too.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from botocore.credentials import Credentials

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

    One instance per proxy process. Credentials are resolved on first use and
    cached; ``botocore``'s credential objects refresh themselves for the
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

    def sign(
        self,
        *,
        url: str,
        body: bytes,
        inbound_headers: dict[str, str],
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

        creds = self._resolve_credentials()

        # Preserve everything the caller sent except the headers SigV4 will
        # recompute or that describe the stale (pre-compression) body. Keeping
        # content-type and any x-amzn-bedrock-* / anthropic-* headers matters:
        # Bedrock guardrail headers and the anthropic beta headers travel here.
        passthrough = {
            k: v for k, v in inbound_headers.items() if k.lower() not in _DROP_FOR_SIGNING
        }
        host = urlsplit(url).netloc
        passthrough["host"] = host

        aws_request = AWSRequest(method="POST", url=url, data=body, headers=passthrough)
        SigV4Auth(creds, _SERVICE, self._region).add_auth(aws_request)
        return dict(aws_request.headers.items())
