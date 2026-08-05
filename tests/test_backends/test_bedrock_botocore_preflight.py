"""Preflight guard for Bedrock + temporary credentials without botocore (#1551).

litellm takes the botocore-backed ``_auth_with_aws_session_token`` path as
soon as ``AWS_SESSION_TOKEN`` is present. botocore ships only with the
``bedrock`` extra, so on a slim install the failure used to surface at request
time as a misleading ``authentication_error: No module named 'botocore'``.
``LiteLLMBackend`` now fails fast at startup with an actionable message.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from headroom.backends import litellm as litellm_mod
from headroom.backends.litellm import LiteLLMBackend


def test_bedrock_session_token_without_botocore_raises_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm_mod, "LITELLM_AVAILABLE", True)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "tmp-session-token")

    # Simulate botocore not installed (the slim default-image case).
    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(ImportError, match="botocore") as exc:
            LiteLLMBackend(provider="bedrock", region="us-west-2")

    # The message must point at the fix, not just name the missing module.
    assert "headroom-ai[bedrock]" in str(exc.value)


def test_bedrock_without_session_token_does_not_trip_botocore_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static-credential Bedrock users don't need botocore — guard stays quiet."""
    monkeypatch.setattr(litellm_mod, "LITELLM_AVAILABLE", True)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    # botocore missing, but no session token → guard must NOT raise. Stub the
    # downstream model-map fetch and litellm handle so construction completes.
    with patch("importlib.util.find_spec", return_value=None):
        with patch.object(litellm_mod, "_fetch_bedrock_inference_profiles", return_value={}):
            with patch.object(litellm_mod, "litellm", create=True):
                backend = LiteLLMBackend(provider="bedrock", region="us-west-2")

    assert backend.provider == "bedrock"
