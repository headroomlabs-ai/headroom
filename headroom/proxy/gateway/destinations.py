"""Pure URL and provider audience validation, usable during offline validation."""

from __future__ import annotations

import re
from urllib.parse import SplitResult, urlsplit

SOURCE_KINDS = {
    "openai": "env",
    "anthropic": "env",
    "gemini": "env",
    "compatible": "env",
    "vertex": "gcp-adc",
    "bedrock": "aws-chain",
}
PUBLIC_HOSTS = {
    "openai": "api.openai.com",
    "anthropic": "api.anthropic.com",
    "gemini": "generativelanguage.googleapis.com",
}


def validate_path(path: str) -> None:
    if (
        not path.startswith("/")
        or path.startswith("//")
        or any(character in path for character in "%\\?#")
        or any(ord(character) <= 32 or ord(character) >= 127 for character in path)
        or any(segment in {".", ".."} for segment in path.split("/"))
    ):
        raise ValueError("ambiguous upstream path")


def https_destination(url: str) -> SplitResult:
    if not url.startswith("https://") or any(ord(c) <= 32 or ord(c) >= 127 for c in url):
        raise ValueError("upstream requires an exact HTTPS URL")
    parsed = urlsplit(url)
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or "#" in url
        or any(c in parsed.netloc for c in "%\\")
        or parsed.hostname.endswith(".")
        or not 1 <= (parsed.port or 443) <= 65535
        or parsed.port == 0
        or parsed.netloc.endswith(":")
    ):
        raise ValueError("invalid upstream authority")
    validate_path(parsed.path or "/")
    return parsed


def normalized_origin(url: str) -> str:
    parsed = https_destination(url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    return f"https://{host.lower()}:{parsed.port or 443}"


def path_within(path: str, prefix: str) -> bool:
    if prefix.endswith("/"):
        return path.startswith(prefix)
    return path == prefix or path.startswith(prefix + "/")


def validate_audience(
    provider: str, url: str, *, project: str | None = None, region: str | None = None
) -> None:
    parsed = https_destination(url)
    if provider in PUBLIC_HOSTS:
        if parsed.hostname != PUBLIC_HOSTS[provider] or (parsed.port or 443) != 443:
            raise ValueError("provider audience mismatch")
    elif provider == "bedrock":
        if (
            not region
            or parsed.hostname != f"bedrock-runtime.{region}.amazonaws.com"
            or (parsed.port or 443) != 443
        ):
            raise ValueError("AWS region audience mismatch")
    elif provider == "vertex":
        match = re.fullmatch(r"([a-z0-9-]+)-aiplatform\.googleapis\.com", parsed.hostname or "")
        if not match or not project or (parsed.port or 443) != 443:
            raise ValueError("Google project audience mismatch")
        if parsed.path and not path_within(
            parsed.path, f"/v1/projects/{project}/locations/{match[1]}/"
        ):
            raise ValueError("Google project or region audience mismatch")
    elif provider != "compatible":
        raise ValueError("unadmitted provider")
