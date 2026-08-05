"""Runtime helpers for Codex/OpenAI-facing integrations."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_API_URL = "https://api.openai.com"


@dataclass(frozen=True, slots=True)
class CodexRoutingDecision:
    """Resolved Codex routing headers and whether they target ChatGPT auth."""

    headers: dict[str, str]
    is_chatgpt_auth: bool


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenAI-compatible integrations."""
    return f"http://127.0.0.1:{port}/v1"


def build_launch_env(
    port: int, environ: Mapping[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for Codex through the local proxy."""
    env = dict(environ or os.environ)
    base_url = proxy_base_url(port)
    env["OPENAI_BASE_URL"] = base_url
    return env, [f"OPENAI_BASE_URL={base_url}"]


def decode_openai_bearer_payload(headers: Mapping[str, str]) -> dict[str, Any] | None:
    """Best-effort decode of an OpenAI OAuth bearer token payload.

    This intentionally does not verify the JWT signature. The decoded payload is
    only a routing hint; upstream still performs the actual authorization.
    """
    auth = headers.get("authorization") or headers.get("Authorization")
    if not auth:
        return None

    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or token.count(".") < 2:
        return None

    payload = token.split(".", 2)[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None

    return data if isinstance(data, dict) else None


def resolve_codex_routing(headers: Mapping[str, str]) -> CodexRoutingDecision:
    """Resolve ChatGPT Codex routing hints from explicit headers or OAuth JWT."""
    resolved = dict(headers)
    lower_lookup = {key.lower(): key for key in resolved}

    if "chatgpt-account-id" in lower_lookup:
        return CodexRoutingDecision(headers=resolved, is_chatgpt_auth=True)

    payload = decode_openai_bearer_payload(resolved)
    auth_claims = payload.get("https://api.openai.com/auth") if isinstance(payload, dict) else None
    account_id = auth_claims.get("chatgpt_account_id") if isinstance(auth_claims, dict) else None
    if isinstance(account_id, str) and account_id.strip():
        resolved["ChatGPT-Account-ID"] = account_id.strip()
        return CodexRoutingDecision(headers=resolved, is_chatgpt_auth=True)

    return CodexRoutingDecision(headers=resolved, is_chatgpt_auth=False)


def resolve_codex_routing_headers(headers: Mapping[str, str]) -> tuple[dict[str, str], bool]:
    """Resolve Codex routing headers as a legacy tuple."""
    decision = resolve_codex_routing(headers)
    return decision.headers, decision.is_chatgpt_auth
