"""Reference implementation of a ``--bedrock-client-hook`` for Headroom.

This module shows how a user can plug a custom boto3 session
(``boto3-refresh-session`` here) into Headroom so STS credentials
are refreshed transparently, without restarting the proxy every hour.

Usage::

    pip install boto3-refresh-session
    headroom proxy \\
        --backend bedrock \\
        --region ap-northeast-1 \\
        --bedrock-client-hook headroom.extras.bedrock_refresh:make_client
"""

from __future__ import annotations

import os
from typing import Any


def make_client(region: str | None) -> Any:
    """Build a ``bedrock-runtime`` boto3 client whose STS credentials
    are refreshed in-place before they expire.

    Returns ``None`` if the optional dependency is not installed —
    Headroom then falls back to the default env-based client.
    """
    try:
        from boto3_refresh_session import refreshable_session  # type: ignore[import-not-found]
    except ImportError:
        return None

    # ``refreshable_session`` wraps a boto3 Session in one whose
    # credentials are re-resolved on the fly (assume-role, profile, etc.)
    # with a configurable refresh window. This is the exact property we
    # need: STS sessions do not need to be passed back into litellm on
    # every request — boto3 itself re-uses the same session object and
    # re-signs each SigV4 request with a fresh credential.
    session = refreshable_session(
        # ``additional_not_after`` makes boto3 refresh credentials 5 min
        # before expiry. Tune to taste.
        additional_not_after=300,
    )

    return session.client(
        "bedrock-runtime",
        region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
    )
