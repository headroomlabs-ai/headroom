"""Pure auth-mode and client classification policy."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class AuthMode(str, enum.Enum):
    """Auth-mode classes Headroom routes compression policy through."""

    PAYG = "payg"
    OAUTH = "oauth"
    SUBSCRIPTION = "subscription"


SUBSCRIPTION_UA_PREFIXES: tuple[str, ...] = (
    "claude-cli/",
    "claude-code/",
    "codex-cli/",
    "cursor/",
    "grok/",
    "claude-vscode/",
    "github-copilot/",
    "anthropic-cli/",
    "antigravity/",
)


CLIENT_UA_MAP: tuple[tuple[str, str], ...] = (
    ("claude-code/", "claude-code"),
    ("claude-cli/", "claude-code"),
    ("claude-vscode/", "claude-vscode"),
    ("anthropic-cli/", "anthropic-cli"),
    ("codex-cli/", "codex"),
    ("cursor/", "cursor"),
    ("grok/", "grok_build"),
    ("zed/", "zed"),
    ("aider/", "aider"),
    ("droid/", "droid"),
    ("opencode/", "opencode"),
    ("github-copilot/", "copilot"),
    ("antigravity/", "antigravity"),
    ("strands-agents/", "strands"),
)


CODEX_RESPONSES_PATH = "/v1/responses"


@dataclass(frozen=True)
class AuthSignals:
    """Normalized header signals used by pure auth/client classifiers."""

    user_agent: str = ""
    authorization: str = ""
    x_api_key: str = ""
    x_goog_api_key: str = ""
    x_client: str = ""

    @property
    def user_agent_lower(self) -> str:
        return self.user_agent.lower()


def classify_auth_signals(signals: AuthSignals) -> AuthMode:
    """Classify auth mode from normalized header values."""
    ua_lower = signals.user_agent_lower
    for prefix in SUBSCRIPTION_UA_PREFIXES:
        if prefix in ua_lower:
            return AuthMode.SUBSCRIPTION

    auth = signals.authorization
    if auth.startswith("Bearer "):
        token = auth[len("Bearer ") :]
        if token.startswith("sk-ant-oat"):
            return AuthMode.OAUTH
        if token.startswith("sk-ant-api") or token.startswith("sk-"):
            return AuthMode.PAYG
        if len(token.split(".")) >= 3:
            return AuthMode.OAUTH
    elif auth:
        return AuthMode.OAUTH

    if signals.x_api_key:
        return AuthMode.PAYG
    if signals.x_goog_api_key:
        return AuthMode.PAYG
    return AuthMode.PAYG


def classify_client_signals(signals: AuthSignals, *, default: str | None = None) -> str | None:
    """Identify the client harness from normalized client signals."""
    explicit = signals.x_client.strip().lower()
    if explicit:
        return explicit
    ua_lower = signals.user_agent_lower
    if not ua_lower:
        return None
    for needle, name in CLIENT_UA_MAP:
        if needle in ua_lower:
            return name
    return default


def is_codex_responses_path(path: str) -> bool:
    """Return True for the OpenAI Responses endpoint and its subpaths."""
    return path == CODEX_RESPONSES_PATH or path.startswith(CODEX_RESPONSES_PATH + "/")


def should_stamp_codex_client_signals(path: str, signals: AuthSignals) -> bool:
    """Whether an unidentified Responses caller should be stamped as Codex."""
    return is_codex_responses_path(path) and classify_client_signals(signals) is None
