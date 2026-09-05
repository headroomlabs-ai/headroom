"""GitHub Copilot OAuth discovery and API-token exchange helpers."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Mapping
from contextvars import ContextVar
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlencode, urlparse

from headroom import paths
from headroom._subprocess import run
from headroom.copilot_linux_secret import read_copilot_oauth_token as read_linux_secret_token
from headroom.copilot_macos_keychain import read_copilot_oauth_token as read_macos_keychain_token
from headroom.proxy import ssl_context as proxy_ssl_context

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.githubcopilot.com"
# Copilot serves *chat* from the CAPI host above and *inline completions* from a
# separate proxy host. GitHub's own client library keeps them apart:
#
#     _getCAPIUrl(t)   -> t?.endpoints.api   || "https://api.githubcopilot.com"
#     _getProxyUrl(t)  -> t?.endpoints.proxy || DEFAULT_PROXY_BASE_URL
#     DEFAULT_PROXY_BASE_URL = "https://copilot-proxy.githubusercontent.com"
#
# and builds completions as `${proxyBaseURL}/v1/engines/<engine>/completions`
# (@vscode/copilot-api 0.5.2). Sending that path to the CAPI host is the wrong
# surface, so the completions default has to be its own constant (#3076).
DEFAULT_COMPLETIONS_PROXY_URL = "https://copilot-proxy.githubusercontent.com"
DEFAULT_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
DEFAULT_USER_INFO_URL = "https://api.github.com/copilot_internal/user"
DEFAULT_GITHUB_HOST = "github.com"
COPILOT_CHAT_OAUTH_CLIENT_ID = "Iv1.b507a08c87ecfe98"
_TOKEN_EXPIRY_BUFFER_S = 60
_DEFAULT_EDITOR_VERSION = "vscode/1.107.0"
_DEFAULT_USER_AGENT = "GitHubCopilotChat/0.35.0"
_DEFAULT_EDITOR_PLUGIN_VERSION = "copilot-chat/0.35.0"
_DEFAULT_COPILOT_INTEGRATION_ID = "vscode-chat"
_DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"

_API_TOKEN_ENV_VARS = (
    "GITHUB_COPILOT_API_TOKEN",
    "COPILOT_PROVIDER_BEARER_TOKEN",
)
_REFRESH_OAUTH_TOKEN_ENV_VAR = "GITHUB_COPILOT_REFRESH_OAUTH_TOKEN"
_API_TOKEN_EXPIRES_AT_ENV_VAR = "GITHUB_COPILOT_API_TOKEN_EXPIRES_AT"
_COPILOT_OAUTH_TOKEN_ENV_VARS = (
    "GITHUB_COPILOT_GITHUB_TOKEN",
    "GITHUB_COPILOT_TOKEN",
    "COPILOT_GITHUB_TOKEN",
)
_GENERIC_GITHUB_TOKEN_ENV_VARS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
)
_OAUTH_TOKEN_KEYS = (
    "oauth_token",
    "oauthToken",
    "token",
    "access_token",
    "accessToken",
)
_EXPIRY_KEYS = ("expires_at", "expiresAt", "expiry", "expires")


def _urlopen(request: urllib_request.Request, *, timeout: float) -> Any:
    """Open a GitHub request with Headroom's configured corporate trust roots."""

    context = proxy_ssl_context.build_urlopen_context()
    if context is not None:
        return urllib_request.urlopen(request, timeout=timeout, context=context)
    return urllib_request.urlopen(request, timeout=timeout)


@dataclass(frozen=True)
class CopilotAPIToken:
    """Short-lived API token exchanged from a GitHub OAuth token."""

    token: str
    expires_at: float
    api_url: str = DEFAULT_API_URL
    refresh_in: int | None = None
    sku: str | None = None

    @property
    def is_valid(self) -> bool:
        return time.time() < (self.expires_at - _TOKEN_EXPIRY_BUFFER_S)


@dataclass(frozen=True)
class CopilotTokenCandidate:
    """A discovered reusable token plus enough metadata to reason about trust."""

    token: str
    source: str
    confidence: str
    validate_for_subscription: bool = True


@dataclass(frozen=True)
class CopilotSubscriptionTokenResolution:
    """A Copilot subscription token plus safe routing metadata."""

    token: str
    source: str
    confidence: str
    api_url: str
    token_fingerprint: str
    refresh_oauth_token: str | None = None
    api_token_expires_at: float | None = None


def token_fingerprint(token: str) -> str:
    """Return a stable non-secret fingerprint for comparing token handoffs."""

    digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).hexdigest()
    return f"sha256:{digest[:12]}"


def _github_host() -> str:
    explicit = os.environ.get("GITHUB_COPILOT_HOST", "").strip().lower()
    if explicit:
        return explicit

    enterprise_domain = _configured_enterprise_domain()
    if enterprise_domain:
        return enterprise_domain

    configured_url = os.environ.get("GITHUB_COPILOT_API_URL", "").strip()
    if configured_url:
        hostname = _configured_url_hostname(configured_url)
        if _is_public_copilot_api_host(hostname):
            return DEFAULT_GITHUB_HOST
        for prefix in ("copilot-api.", "api."):
            if hostname.startswith(prefix):
                hostname = hostname[len(prefix) :]
                break
        if hostname and hostname not in {
            DEFAULT_GITHUB_HOST,
            "api.github.com",
            "githubcopilot.com",
        }:
            return hostname

    return DEFAULT_GITHUB_HOST


def headroom_copilot_auth_path() -> Path:
    """Return the path where Headroom stores its Copilot OAuth token."""

    override = os.environ.get("HEADROOM_COPILOT_AUTH_FILE", "").strip()
    if override:
        return Path(override).expanduser()
    return paths.workspace_dir() / "copilot_auth.json"


def normalize_copilot_enterprise_url(enterprise_url: str) -> str:
    """Normalize a GitHub Enterprise URL or domain."""

    return enterprise_url.strip().replace("https://", "").replace("http://", "").rstrip("/")


def _enterprise_hostname(enterprise_url: str) -> str:
    normalized = normalize_copilot_enterprise_url(enterprise_url)
    if not normalized:
        return ""
    try:
        parsed = urlparse(f"https://{normalized}")
        _ = parsed.port
    except ValueError:
        return ""
    hostname = (parsed.hostname or "").strip().lower()
    return hostname if hostname and " " not in hostname else ""


def _configured_url_hostname(configured_url: str) -> str:
    raw = configured_url.strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        if not parsed.scheme or not parsed.netloc:
            return ""
        _ = parsed.port
    except ValueError:
        return ""
    hostname = (parsed.hostname or "").strip().lower()
    return hostname if hostname and " " not in hostname else ""


def _copilot_subdomain_enterprise_host(enterprise_url: str) -> str | None:
    """Return a host that supports api.<host> and copilot-api.<host> URLs.

    GitHub.com Enterprise Cloud URLs such as ``github.com/enterprises/acme``
    identify an account, not an API hostname.
    """

    host = _enterprise_hostname(enterprise_url)
    for prefix in ("copilot-api.", "api."):
        if host.startswith(prefix):
            host = host[len(prefix) :]
            break
    if (
        not host
        or host in {"github.com", "www.github.com", "api.github.com"}
        or _is_public_copilot_api_host(host)
    ):
        return None
    return host


def copilot_api_url_from_enterprise_url(enterprise_url: str) -> str:
    """Return a Copilot API base for GitHub Enterprise Server/custom domains."""

    host = _copilot_subdomain_enterprise_host(enterprise_url)
    if host is None:
        return DEFAULT_API_URL
    return f"https://copilot-api.{host}"


def _configured_enterprise_domain() -> str | None:
    enterprise_url = (
        os.environ.get("GITHUB_COPILOT_ENTERPRISE_URL", "").strip()
        or os.environ.get("GITHUB_COPILOT_ENTERPRISE_DOMAIN", "").strip()
    )
    if not enterprise_url:
        return None
    return _copilot_subdomain_enterprise_host(enterprise_url)


def default_oauth_domain() -> str:
    """Return the OAuth domain from GITHUB_COPILOT_ENTERPRISE_URL, or github.com."""
    domain = _configured_enterprise_domain()
    return domain if domain else DEFAULT_GITHUB_HOST


def _configured_api_url_override() -> str | None:
    api_url = os.environ.get("GITHUB_COPILOT_API_URL", "").strip()
    if api_url:
        return api_url.rstrip("/")

    enterprise_domain = _configured_enterprise_domain()
    if enterprise_domain:
        return copilot_api_url_from_enterprise_url(enterprise_domain).rstrip("/")

    return None


def _configured_api_url() -> str:
    configured = _configured_api_url_override()
    if configured:
        return configured
    return DEFAULT_API_URL


def copilot_api_url() -> str:
    """Return the configured Copilot API base URL without any network calls.

    Resolves ``GITHUB_COPILOT_API_URL``, then the configured enterprise domain,
    then ``api.githubcopilot.com``. Unlike :func:`resolve_copilot_api_url` this
    performs no token exchange, so it is safe to call while routing a request.
    """

    return _configured_api_url()


# GitHub's token exchange advertises the host that serves inline completions
# under ``endpoints.proxy``, alongside the ``endpoints.api`` chat host. It is
# recorded here when observed so completions routing uses GitHub's own answer
# instead of an assumption about which host serves that endpoint (#3076).
_observed_completions_base_url: str | None = None


def _remember_completions_endpoint(payload: Any) -> None:
    """Record the completions host advertised by a token-exchange payload."""

    global _observed_completions_base_url
    endpoints = payload.get("endpoints") if isinstance(payload, dict) else None
    proxy_url = endpoints.get("proxy") if isinstance(endpoints, dict) else None
    if isinstance(proxy_url, str) and proxy_url.strip():
        _observed_completions_base_url = proxy_url.strip().rstrip("/")


def reset_observed_completions_endpoint() -> None:
    """Forget the advertised completions host (test isolation)."""

    global _observed_completions_base_url
    _observed_completions_base_url = None


def _url_host(value: str) -> str:
    """Hostname for a URL, tolerating a scheme-less value.

    Mirrors the normalization :func:`is_copilot_api_url` performs, so a host
    configured without "https://" is not silently treated as a different host.
    """

    parsed = urlparse(value)
    netloc_or_path = parsed.netloc.lower() or parsed.path.lower()
    return (parsed.hostname or netloc_or_path.split("/", 1)[0]).lower()


def is_copilot_completions_host(url: str | None) -> bool:
    """Return True when *url* already points at a Copilot inline-completions host.

    Distinct from :func:`is_copilot_api_url`, which matches the CAPI (chat)
    surface. A CAPI host is *not* a completions host, so the two must not be
    conflated when deciding whether a completions request is already addressed
    correctly.
    """

    if not url:
        return False
    # Compare hosts, never whole strings: this is asked both about a bare base
    # URL (routing) and about a fully-built URL with the path appended (auth).
    # A string compare answers True for the first and False for the second, so
    # an operator override would route correctly and then be forwarded with no
    # credentials at all.
    host = _url_host(url)
    if not host:
        return False
    override = os.environ.get("GITHUB_COPILOT_PROXY_URL", "").strip()
    if override and host == _url_host(override):
        return True
    if host == "copilot-proxy.githubusercontent.com":
        return True
    # Per-SKU hosts GitHub hands out via `endpoints.proxy`, e.g.
    # proxy.individual.githubcopilot.com / proxy.business… / proxy.enterprise….
    return host.startswith("proxy.") and host.endswith(".githubcopilot.com")


def copilot_completions_base_url() -> str:
    """Return the base URL serving Copilot's inline-completions endpoint.

    Resolution order, most authoritative first:

    1. ``GITHUB_COPILOT_PROXY_URL`` — an explicit operator override, so a
       network that fronts Copilot behind its own gateway (or a GitHub change
       to this endpoint) is a config edit rather than a code change.
    2. ``endpoints.proxy`` from the last Copilot token exchange — GitHub
       telling us directly where completions go.
    3. ``copilot-proxy.githubusercontent.com`` — GitHub's own documented
       default for this endpoint (see ``DEFAULT_COMPLETIONS_PROXY_URL``).
    4. For an enterprise or otherwise custom Copilot deployment, that
       deployment's own host. Falling back to the public GitHub host there would
       send an enterprise tenant's keystrokes outside their deployment, which is
       worse than failing to resolve.

    Note what step 4 must *not* capture: a configured API URL that is itself a
    public Copilot host. ``headroom wrap vscode`` sets ``GITHUB_COPILOT_API_URL``
    to the resolved subscription URL (e.g. ``api.business.githubcopilot.com``),
    which is the chat surface — returning it here would put the completions path
    straight back on the host that answers it with 404. Only a host outside
    ``*.githubcopilot.com`` indicates a deployment whose traffic has to stay put.

    Never performs I/O; step 2 only reads what a previous exchange recorded.
    """

    override = os.environ.get("GITHUB_COPILOT_PROXY_URL", "").strip()
    if override:
        return override.rstrip("/")
    if _observed_completions_base_url:
        return _observed_completions_base_url
    configured = _configured_api_url_override()
    if configured and not _is_public_copilot_api_host(_url_host(configured)):
        return configured
    return DEFAULT_COMPLETIONS_PROXY_URL


def _github_oauth_domain(domain: str | None = None) -> str:
    raw = (domain or DEFAULT_GITHUB_HOST).strip()
    if not raw:
        return DEFAULT_GITHUB_HOST
    host = _enterprise_hostname(raw)
    return host or DEFAULT_GITHUB_HOST


def _github_oauth_urls(domain: str) -> dict[str, str]:
    normalized = _github_oauth_domain(domain)
    return {
        "device_code": f"https://{normalized}/login/device/code",
        "access_token": f"https://{normalized}/login/oauth/access_token",
    }


def _token_exchange_url() -> str:
    override = os.environ.get("GITHUB_COPILOT_TOKEN_EXCHANGE_URL", "").strip()
    if override:
        return override

    enterprise_domain = _configured_enterprise_domain()
    if enterprise_domain:
        return f"https://api.{enterprise_domain}/copilot_internal/v2/token"

    return DEFAULT_TOKEN_EXCHANGE_URL


def _user_info_url() -> str:
    override = os.environ.get("GITHUB_COPILOT_USER_INFO_URL", "").strip()
    if override:
        return override

    enterprise_domain = _configured_enterprise_domain()
    if enterprise_domain:
        return f"https://api.{enterprise_domain}/copilot_internal/user"

    return DEFAULT_USER_INFO_URL


def _should_exchange_oauth_token() -> bool:
    raw = os.environ.get("GITHUB_COPILOT_USE_TOKEN_EXCHANGE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_token_file_paths() -> list[Path]:
    override = os.environ.get("GITHUB_COPILOT_TOKEN_FILE", "").strip()
    if override:
        return [Path(override).expanduser()]

    paths: list[Path] = []
    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        base = Path(local_appdata) / "github-copilot"
        paths.extend([base / "apps.json", base / "hosts.json"])

    config_base = Path.home() / ".config" / "github-copilot"
    paths.extend([config_base / "apps.json", config_base / "hosts.json"])
    return paths


def _read_gh_cli_oauth_token() -> str | None:
    gh_bin = os.environ.get("GH_PATH", "").strip() or "gh"
    command = [gh_bin, "auth", "token"]
    host = _github_host()
    if host and host != DEFAULT_GITHUB_HOST:
        command.extend(["--hostname", host])

    try:
        result = run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        logger.debug("Unable to invoke GitHub CLI for Copilot auth discovery: %s", exc)
        return None

    if result.returncode != 0:
        logger.debug("GitHub CLI auth token lookup failed with exit code %s", result.returncode)
        return None

    token = result.stdout.strip()
    return token or None


def _read_macos_keychain_oauth_token() -> str | None:
    """Best-effort Copilot CLI token lookup from macOS Keychain."""

    return read_macos_keychain_token(host=_github_host())


def _read_linux_secret_oauth_token() -> str | None:
    """Best-effort Copilot CLI token lookup from Linux Secret Service."""

    return read_linux_secret_token(host=_github_host())


def _read_windows_copilot_cli_oauth_token() -> str | None:
    if os.name != "nt":
        return None

    class FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    class CREDENTIAL(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", wintypes.LPVOID),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    cred_ptr = ctypes.POINTER(CREDENTIAL)
    credentials = ctypes.POINTER(cred_ptr)()
    count = wintypes.DWORD()
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return None

    advapi32 = win_dll("Advapi32.dll")
    advapi32.CredEnumerateW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(ctypes.POINTER(cred_ptr)),
    ]
    advapi32.CredEnumerateW.restype = wintypes.BOOL
    advapi32.CredFree.argtypes = [wintypes.LPVOID]

    try:
        if not advapi32.CredEnumerateW(None, 0, ctypes.byref(count), ctypes.byref(credentials)):
            return None
    except OSError as exc:
        logger.debug("Unable to enumerate Windows credentials for Copilot auth discovery: %s", exc)
        return None

    host = _github_host().lower()
    bare_host = host.removeprefix("https://").removeprefix("http://")

    gh_prefix = f"gh:{bare_host}:"
    copilot_prefixes = [f"copilot-cli/{host}:"]
    if "://" not in host:
        copilot_prefixes.append(f"copilot-cli/https://{host}:")
        copilot_prefixes.append(f"copilot-cli/https://{host}/")

    gh_tokens: list[str] = []
    copilot_tokens: list[str] = []

    try:
        for idx in range(count.value):
            credential = credentials[idx].contents
            target = (credential.TargetName or "").strip().lower()
            if credential.CredentialBlobSize <= 0 or not credential.CredentialBlob:
                continue
            blob = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
            token = blob.decode("utf-8", errors="replace").strip()
            if not token:
                continue
            if target.startswith(gh_prefix):
                gh_tokens.append(token)
            elif any(target.startswith(p) for p in copilot_prefixes):
                copilot_tokens.append(token)
    finally:
        if credentials:
            advapi32.CredFree(credentials)

    for token in gh_tokens + copilot_tokens:
        return token

    return None


def _parse_expiry(value: Any) -> float | None:
    if value in (None, ""):
        return None

    if isinstance(value, int | float):
        number = float(value)
        if not math.isfinite(number):
            return None
        if number > 10_000_000_000:
            return number / 1000.0
        return number

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            return _parse_expiry(int(raw))
        try:
            return _parse_expiry(float(raw))
        except ValueError:
            pass
        try:
            normalized = raw.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            return None

    return None


def _entry_expired(entry: dict[str, Any]) -> bool:
    for key in _EXPIRY_KEYS:
        expiry = _parse_expiry(entry.get(key))
        if expiry is None:
            continue
        return time.time() >= (expiry - _TOKEN_EXPIRY_BUFFER_S)
    return False


def read_headroom_copilot_oauth_token() -> str | None:
    """Return Headroom's saved Copilot OAuth token, if one is available."""

    try:
        payload = json.loads(headroom_copilot_auth_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug("Unable to read Headroom Copilot auth file: %s", exc)
        return None

    if not isinstance(payload, dict) or payload.get("type") != "oauth":
        return None
    token = payload.get("refresh")
    return token.strip() if isinstance(token, str) and token.strip() else None


def save_headroom_copilot_oauth_token(
    token: str,
    *,
    domain: str = DEFAULT_GITHUB_HOST,
) -> Path:
    """Persist the Copilot OAuth token returned by GitHub device login."""

    token = token.strip()
    if not token:
        raise ValueError("Copilot OAuth token must not be empty.")

    path = headroom_copilot_auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "type": "oauth",
        "provider": "github-copilot",
        "refresh": token,
        "domain": _github_oauth_domain(domain),
        "created_at": int(time.time()),
    }
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def start_copilot_device_authorization(
    *,
    domain: str = DEFAULT_GITHUB_HOST,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Start the GitHub Copilot OAuth device-code flow."""

    urls = _github_oauth_urls(domain)
    body = urlencode({"client_id": COPILOT_CHAT_OAUTH_CLIENT_ID, "scope": "read:user"}).encode(
        "utf-8"
    )
    request = urllib_request.Request(
        urls["device_code"],
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": _DEFAULT_USER_AGENT,
        },
        method="POST",
    )
    with _urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise RuntimeError("GitHub device authorization returned an invalid response.")
    return payload


def poll_copilot_device_authorization(
    device_code: str,
    *,
    domain: str = DEFAULT_GITHUB_HOST,
    interval: int = 5,
    expires_in: int = 900,
    timeout: float = 10.0,
) -> str:
    """Poll GitHub until the device-code OAuth flow returns an access token."""

    urls = _github_oauth_urls(domain)
    deadline = time.time() + max(1, expires_in)
    poll_interval = max(1, interval)
    while time.time() < deadline:
        body = urlencode(
            {
                "client_id": COPILOT_CHAT_OAUTH_CLIENT_ID,
                "device_code": device_code,
                "grant_type": _DEVICE_CODE_GRANT_TYPE,
            }
        ).encode("utf-8")
        request = urllib_request.Request(
            urls["access_token"],
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": _DEFAULT_USER_AGENT,
            },
            method="POST",
        )
        with _urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        if not isinstance(payload, dict):
            raise RuntimeError("GitHub device authorization returned an invalid response.")

        access_token = payload.get("access_token")
        if isinstance(access_token, str) and access_token.strip():
            return access_token.strip()

        error = str(payload.get("error") or "").strip()
        if error == "authorization_pending":
            time.sleep(poll_interval)
            continue
        if error == "slow_down":
            poll_interval += 5
            time.sleep(poll_interval)
            continue
        if error == "expired_token":
            raise RuntimeError("GitHub device authorization expired.")
        if error:
            description = str(payload.get("error_description") or error).strip()
            raise RuntimeError(f"GitHub device authorization failed: {description}")

        time.sleep(poll_interval)

    raise RuntimeError("GitHub device authorization expired.")


def _extract_oauth_token(entry: dict[str, Any]) -> str | None:
    if _entry_expired(entry):
        return None

    for key in _OAUTH_TOKEN_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for value in entry.values():
        if isinstance(value, dict):
            nested = _extract_oauth_token(value)
            if nested:
                return nested

    return None


def _iter_file_entries(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                entries.append((str(key), value))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            if isinstance(value, dict):
                key = str(value.get("host") or value.get("githubHost") or idx)
                entries.append((key, value))
    return entries


def read_cached_oauth_token() -> str | None:
    """Return a GitHub OAuth token for Copilot, if one is available."""

    for candidate in iter_oauth_token_candidates():
        return candidate.token
    return None


def iter_oauth_token_candidates() -> list[CopilotTokenCandidate]:
    """Return reusable token candidates in safest-first discovery order."""

    candidates: list[CopilotTokenCandidate] = []

    headroom_copilot_token = read_headroom_copilot_oauth_token()
    if headroom_copilot_token:
        candidates.append(
            CopilotTokenCandidate(
                token=headroom_copilot_token,
                source=f"headroom-copilot-auth:{headroom_copilot_auth_path()}",
                confidence="copilot-oauth",
            )
        )

    for env_var in _COPILOT_OAUTH_TOKEN_ENV_VARS:
        token = os.environ.get(env_var, "").strip()
        if token:
            candidates.append(
                CopilotTokenCandidate(
                    token=token,
                    source=f"env:{env_var}",
                    confidence="explicit",
                )
            )

    windows_copilot_token = _read_windows_copilot_cli_oauth_token()
    if windows_copilot_token:
        candidates.append(
            CopilotTokenCandidate(
                token=windows_copilot_token,
                source="windows-credential-manager:copilot-cli",
                confidence="high",
            )
        )

    macos_copilot_token = _read_macos_keychain_oauth_token()
    if macos_copilot_token:
        candidates.append(
            CopilotTokenCandidate(
                token=macos_copilot_token,
                source="macos-keychain:copilot-cli",
                confidence="high",
            )
        )

    linux_copilot_token = _read_linux_secret_oauth_token()
    if linux_copilot_token:
        candidates.append(
            CopilotTokenCandidate(
                token=linux_copilot_token,
                source="linux-secret-service:copilot-cli",
                confidence="high",
            )
        )

    candidates.extend(_read_file_oauth_token_candidates())

    for env_var in _GENERIC_GITHUB_TOKEN_ENV_VARS:
        token = os.environ.get(env_var, "").strip()
        if token:
            candidates.append(
                CopilotTokenCandidate(
                    token=token,
                    source=f"env:{env_var}",
                    confidence="generic-github",
                )
            )

    gh_token = _read_gh_cli_oauth_token()
    if gh_token:
        candidates.append(
            CopilotTokenCandidate(
                token=gh_token,
                source="gh-cli",
                confidence="generic-github",
            )
        )

    return _dedupe_token_candidates(candidates)


def _read_file_oauth_token_candidates() -> list[CopilotTokenCandidate]:
    """Return token candidates from Copilot/GitHub credential files."""

    candidates: list[CopilotTokenCandidate] = []
    host = _github_host()
    for path in _resolve_token_file_paths():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.debug("Unable to read Copilot credentials file %s: %s", path, exc)
            continue

        for key, entry in _iter_file_entries(payload):
            if host not in key.lower():
                continue
            cached_token = _extract_oauth_token(entry)
            if cached_token:
                candidates.append(
                    CopilotTokenCandidate(
                        token=cached_token,
                        source=f"file:{path}",
                        confidence="medium",
                    )
                )

    return candidates


def _dedupe_token_candidates(
    candidates: list[CopilotTokenCandidate],
) -> list[CopilotTokenCandidate]:
    seen: set[str] = set()
    deduped: list[CopilotTokenCandidate] = []
    for candidate in candidates:
        if candidate.token in seen:
            continue
        seen.add(candidate.token)
        deduped.append(candidate)
    return deduped


def resolve_client_bearer_token() -> str | None:
    """Return a bearer token suitable for satisfying Copilot provider auth checks."""

    for env_var in _API_TOKEN_ENV_VARS:
        token = os.environ.get(env_var, "").strip()
        if token:
            return token
    return read_cached_oauth_token()


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    """Case-insensitive header lookup."""
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def resolve_copilot_integration_id(client_value: str | None = None) -> str:
    """Return the integration ID this request's credential must be bound to.

    GitHub binds a Copilot API token to the ``Copilot-Integration-Id`` it was
    minted under and verifies the pairing with an HMAC. Presenting a token
    minted for one integration alongside a header naming another fails with:

        401 unauthorized: unable to validate HMAC for the given
            Copilot-Integration-ID

    Resolution order — the client's own header wins, matching the long-standing
    contract that ``GITHUB_COPILOT_INTEGRATION_ID`` configures the DEFAULT this
    proxy sends rather than overriding a client that stated its own identity
    (pinned by ``test_apply_copilot_api_auth_preserves_existing_copilot_headers``):

    1. The client's own header — a Copilot CLI session identifies as something
       other than ``vscode-chat``, and minting under its ID keeps GitHub's usage
       attribution pointing at the surface that actually made the call.
    2. ``GITHUB_COPILOT_INTEGRATION_ID`` — the operator-configured default.
    3. The historical built-in default.
    """
    if client_value and client_value.strip():
        return client_value.strip()
    configured = os.environ.get("GITHUB_COPILOT_INTEGRATION_ID", "").strip()
    if configured:
        return configured
    return _DEFAULT_COPILOT_INTEGRATION_ID


def _copilot_chat_header_defaults(integration_id: str | None = None) -> dict[str, str]:
    return {
        "User-Agent": os.environ.get("GITHUB_COPILOT_USER_AGENT", _DEFAULT_USER_AGENT).strip()
        or _DEFAULT_USER_AGENT,
        "Editor-Version": os.environ.get(
            "GITHUB_COPILOT_EDITOR_VERSION", _DEFAULT_EDITOR_VERSION
        ).strip()
        or _DEFAULT_EDITOR_VERSION,
        "Editor-Plugin-Version": os.environ.get(
            "GITHUB_COPILOT_EDITOR_PLUGIN_VERSION",
            _DEFAULT_EDITOR_PLUGIN_VERSION,
        ).strip()
        or _DEFAULT_EDITOR_PLUGIN_VERSION,
        "Copilot-Integration-Id": integration_id or resolve_copilot_integration_id(),
    }


def _overwrite_header(headers: dict[str, str], name: str, value: str) -> None:
    """Set a header, replacing any case-variant already present.

    Writes through the EXISTING key when there is one, so a client that sent
    ``copilot-integration-id`` does not end up with a second
    ``Copilot-Integration-Id`` beside it — duplicate case-variants are what
    ``_set_header_default`` exists to avoid, and the same care applies when
    overwriting.
    """
    for key in list(headers):
        if key.lower() == name.lower():
            headers[key] = value
            return
    headers[name] = value


def _set_header_default(headers: dict[str, str], name: str, value: str) -> None:
    """Set a header default without duplicating case-insensitive equivalents."""

    name_lower = name.lower()
    if any(existing.lower() == name_lower for existing in headers):
        return
    headers[name] = value


def _copilot_token_exchange_headers(
    oauth_token: str, *, integration_id: str | None = None
) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {oauth_token}",
        **_copilot_chat_header_defaults(integration_id),
    }


def _api_url_from_payload(payload: dict[str, Any] | None) -> str | None:
    endpoints = payload.get("endpoints") if isinstance(payload, dict) else None
    api_url = endpoints.get("api") if isinstance(endpoints, dict) else None
    if isinstance(api_url, str) and api_url.strip():
        return api_url.strip().rstrip("/")
    return None


def _subscription_api_url_from_user_info_payload(payload: dict[str, Any] | None) -> str:
    configured = _configured_api_url_override()
    if configured:
        return configured

    api_url = _api_url_from_payload(payload)
    if not api_url:
        return DEFAULT_API_URL

    host = urlparse(api_url).netloc.lower()
    if host in {
        "api.githubcopilot.com",
        "api.individual.githubcopilot.com",
        "api.business.githubcopilot.com",
        "api.enterprise.githubcopilot.com",
    }:
        return DEFAULT_API_URL
    if host.endswith(".githubcopilot.com"):
        return api_url
    return DEFAULT_API_URL


def _subscription_api_url_from_user_info(oauth_token: str) -> str:
    return _subscription_api_url_from_user_info_payload(_fetch_copilot_user_info(oauth_token))


def _api_url_from_exchange_payload(payload: dict[str, Any], *, oauth_token: str) -> str:
    configured = _configured_api_url_override()
    if configured:
        return configured

    api_url = _api_url_from_payload(payload)
    if api_url:
        if is_copilot_api_url(api_url):
            return _subscription_api_url_from_user_info_payload({"endpoints": {"api": api_url}})
        logger.warning(
            "Ignoring non-Copilot API URL from token exchange payload: %s",
            api_url,
        )

    return _subscription_api_url_from_user_info(oauth_token)


def _subscription_resolution(
    *,
    token: str,
    source: str,
    confidence: str,
    api_url: str,
    refresh_oauth_token: str | None = None,
    api_token_expires_at: float | None = None,
) -> CopilotSubscriptionTokenResolution:
    return CopilotSubscriptionTokenResolution(
        token=token,
        source=source,
        confidence=confidence,
        api_url=api_url,
        token_fingerprint=token_fingerprint(token),
        refresh_oauth_token=refresh_oauth_token,
        api_token_expires_at=api_token_expires_at,
    )


def _subscription_resolution_from_token_exchange(
    candidate: CopilotTokenCandidate,
) -> CopilotSubscriptionTokenResolution | None:
    """Exchange a reusable GitHub OAuth token for a Copilot API token."""

    try:
        payload = CopilotTokenProvider._exchange_token_sync(
            _copilot_token_exchange_headers(candidate.token)
        )
    except Exception as exc:
        logger.debug(
            "Unable to exchange Copilot OAuth token from %s via %s: %s",
            candidate.source,
            _token_exchange_url(),
            exc,
        )
        return None

    token = str(payload.get("token") or "").strip()
    if not token:
        logger.debug("Copilot token exchange from %s returned no token", candidate.source)
        return None

    return _subscription_resolution(
        token=token,
        source=f"{candidate.source}:token-exchange",
        confidence="copilot-token-exchange",
        api_url=_api_url_from_exchange_payload(payload, oauth_token=candidate.token),
        refresh_oauth_token=candidate.token,
        api_token_expires_at=_parse_expiry(payload.get("expires_at")),
    )


def resolve_subscription_bearer_token_details() -> CopilotSubscriptionTokenResolution | None:
    """Return the first discovered token that GitHub accepts for subscription APIs."""

    for env_var in _API_TOKEN_ENV_VARS:
        token = os.environ.get(env_var, "").strip()
        if not token:
            continue
        payload = _fetch_copilot_user_info(token)
        if payload is not None:
            return _subscription_resolution(
                token=token,
                source=f"env:{env_var}",
                confidence="explicit-api-token",
                api_url=_subscription_api_url_from_user_info_payload(payload),
            )

    for candidate in iter_oauth_token_candidates():
        if not candidate.validate_for_subscription:
            continue
        if _is_copilot_api_token(candidate.token):
            payload = _fetch_copilot_user_info(candidate.token)
            if payload is not None:
                logger.debug(
                    "Using Copilot API subscription token from %s (%s)",
                    candidate.source,
                    candidate.confidence,
                )
                return _subscription_resolution(
                    token=candidate.token,
                    source=candidate.source,
                    confidence=candidate.confidence,
                    api_url=_subscription_api_url_from_user_info_payload(payload),
                )
            continue

        exchanged = _subscription_resolution_from_token_exchange(candidate)
        if exchanged is not None:
            logger.debug(
                "Using exchanged Copilot subscription token from %s (%s)",
                candidate.source,
                candidate.confidence,
            )
            return exchanged

    return None


def resolve_subscription_bearer_token() -> str | None:
    """Return the first discovered token that GitHub accepts for Copilot subscription APIs."""

    resolution = resolve_subscription_bearer_token_details()
    return resolution.token if resolution is not None else None


def has_oauth_auth() -> bool:
    """Return True when existing Copilot auth can be reused."""

    return resolve_client_bearer_token() is not None


def is_copilot_api_url(url: str | None) -> bool:
    """Return True when the upstream URL points at GitHub Copilot."""

    if not url:
        return False
    parsed = urlparse(url)
    host = parsed.netloc.lower() or parsed.path.lower()
    configured_host = urlparse(_configured_api_url()).netloc.lower()
    if configured_host and host == configured_host:
        return True
    hostname = (parsed.hostname or host.split("/", 1)[0]).lower()
    return _is_public_copilot_api_host(hostname) or _is_ghe_copilot_api_host(hostname)


def is_copilot_upstream_url(url: str | None) -> bool:
    """Return True for any Copilot-served upstream: chat (CAPI) or completions.

    Copilot has two surfaces on two different hosts, and code that asks "is this
    request going to Copilot?" means the union. :func:`is_copilot_api_url` alone
    answers only for chat, so the completions host looked like a stranger:
    ``apply_copilot_api_auth`` attached no credentials to it (401) and
    ``build_copilot_upstream_url`` skipped ``mark_request_routed_to_copilot``,
    which mislabels the provider in telemetry.

    Deliberately *not* folded into :func:`is_copilot_api_url`, which also gates
    validation of the ``endpoints.api`` value from a token exchange and the
    Responses-API preference check — neither of which should treat a completions
    host as a chat host (#3076).
    """

    return is_copilot_api_url(url) or is_copilot_completions_host(url)


def _is_public_copilot_api_host(host: str) -> bool:
    """Return True for GitHub-hosted Copilot API domains."""

    return host == "githubcopilot.com" or host.endswith(".githubcopilot.com")


def _is_ghe_copilot_api_host(host: str) -> bool:
    """Return True for GitHub Enterprise Copilot API hosts.

    GHE Copilot deployments use hosts like ``copilot-api.<tenant>.ghe.com``.
    Restrict this to the Copilot API subdomain so unrelated GHE hosts do not
    receive Copilot auth headers or Copilot-specific path normalization.
    """

    return host == "copilot-api.ghe.com" or (
        host.startswith("copilot-api.") and host.endswith(".ghe.com")
    )


# Per-request flag: set when a request is routed to the GitHub Copilot API so
# the single outcome funnel can label the provider "copilot" regardless of the
# wire shape (OpenAI or Anthropic) the request travelled on. A ContextVar is
# task-local, so it never bleeds across concurrent requests. A ContextVar value
# nonetheless persists until overwritten within a single execution context, so
# the outcome funnel consumes it (read-and-clear) rather than just reading it —
# otherwise a later non-Copilot outcome in the same context (e.g. successive
# messages on one WebSocket task) would be mislabeled.
_request_routed_to_copilot: ContextVar[bool] = ContextVar(
    "_request_routed_to_copilot", default=False
)


def mark_request_routed_to_copilot() -> None:
    """Flag the current request as routed to the GitHub Copilot API."""
    _request_routed_to_copilot.set(True)


def request_routed_to_copilot() -> bool:
    """Return True when the current request was routed to the Copilot API.

    Read-only; does not clear the flag. Prefer :func:`consume_request_routed_to_copilot`
    at the point the label is applied so the flag cannot leak to a later outcome.
    """
    return _request_routed_to_copilot.get()


def consume_request_routed_to_copilot() -> bool:
    """Return whether the current request was routed to the Copilot API, and
    clear the flag so a subsequent outcome emitted in the same execution context
    is not mislabeled."""
    routed = _request_routed_to_copilot.get()
    if routed:
        reset_request_routed_to_copilot()
    return routed


def reset_request_routed_to_copilot() -> None:
    """Clear the Copilot routing flag. For test isolation and any explicit
    request-boundary reset (build_copilot_upstream_url sets it as a side effect,
    so callers outside a request task should reset it to avoid leaking state)."""
    _request_routed_to_copilot.set(False)


def is_copilot_completions_path(path: str) -> bool:
    """Return True for Copilot's inline-completions ("ghost text") endpoint.

    The Copilot editor extensions send code completions to
    ``/v1/engines/<engine>/completions`` on whatever host
    ``github.copilot.advanced.debug.overrideProxyUrl`` names — so when that
    setting points at Headroom, this is the path that arrives.

    The shape identifies GitHub Copilot on its own. OpenAI's Engines API was
    removed years ago and no other provider Headroom fronts serves it, so a
    request on this path is Copilot's and can never be answered by the default
    OpenAI target (#3076).
    """

    normalized = (path if path.startswith("/") else f"/{path}").rstrip("/")
    prefix = "/v1/engines/"
    suffix = "/completions"
    if not normalized.startswith(prefix) or not normalized.endswith(suffix):
        return False
    engine = normalized[len(prefix) : -len(suffix)]
    return bool(engine) and "/" not in engine


def build_copilot_upstream_url(base_url: str, path: str) -> str:
    """Build an upstream URL, normalizing GitHub Copilot's non-/v1 path layout."""

    normalized_base = base_url.rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    if is_copilot_upstream_url(normalized_base):
        # Single routing chokepoint for every Copilot surface (OpenAI
        # chat/responses and Anthropic messages all build their upstream URL
        # here), so mark the request for provider relabeling downstream.
        mark_request_routed_to_copilot()
        # Copilot serves its OpenAI-compatible surface WITHOUT a ``/v1`` prefix
        # (``/chat/completions``, ``/responses``, ...), so strip it there. But its
        # Anthropic surface for Claude models IS ``/v1/messages`` (with the
        # ``/v1``); stripping it forwarded ``/messages`` and Copilot returned 404
        # for claude-* models (#2409). Keep ``/v1`` for the messages endpoint.
        #
        # Inline completions are the same story: the Copilot extension itself
        # builds ``/v1/engines/<engine>/completions``, so the path that reaches
        # us is already the exact path Copilot serves. Stripping ``/v1`` there
        # rewrites a Copilot-native path into one that 404s (#3076). The rule
        # this encodes: strip only for clients speaking generic-OpenAI at
        # Copilot, never for Copilot's own paths.
        keep_v1 = normalized_path.startswith("/v1/messages") or is_copilot_completions_path(
            normalized_path
        )
        if normalized_path.startswith("/v1/") and not keep_v1:
            normalized_path = normalized_path[3:]
    else:
        reset_request_routed_to_copilot()
    return f"{normalized_base}{normalized_path}"


def resolve_copilot_api_url(oauth_token: str | None = None) -> str:
    """Return the Copilot API host to route wrapped requests through.

    Resolution order:

    1. An explicit ``GITHUB_COPILOT_API_URL`` — the operator's escape hatch
       (corporate proxy, enterprise / data-residency host, tests).
    2. The generic public host ``https://api.githubcopilot.com``.

    The account-specific ``endpoints.api`` advertised by ``/copilot_internal/user``
    is intentionally NOT used to route. It returns a segmented host (e.g.
    ``api.individual.githubcopilot.com``) that does not serve newer models on the
    responses API — wrapping such a request regressed after 0.22.4 (#610) — and it
    is not the host the official Copilot client routes with (that comes from the
    token-exchange endpoint, not user info). Accounts that genuinely require a
    dedicated host set ``GITHUB_COPILOT_API_URL`` explicitly. ``oauth_token`` is
    accepted for call-site compatibility but no longer triggers a network lookup.
    """

    del oauth_token  # reserved; routing no longer depends on a user-info lookup
    return _configured_api_url()


def _fetch_copilot_user_info(token: str) -> dict[str, Any] | None:
    """Fetch Copilot account metadata for a reusable OAuth-style token."""

    token = token.strip()
    if not token:
        return None

    headers = _copilot_token_exchange_headers(token)
    request = urllib_request.Request(_user_info_url(), headers=headers, method="GET")
    try:
        with _urlopen(request, timeout=10.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("Unable to resolve Copilot API URL from user info: %s", exc)
        return None

    return payload if isinstance(payload, dict) else None


class CopilotTokenProvider:
    """Resolve and cache short-lived Copilot API tokens."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        # Keyed by integration ID: GitHub binds each token to the
        # ``Copilot-Integration-Id`` it was minted under and HMAC-verifies the
        # pairing, so a token cached for one integration is NOT reusable for
        # another. A single slot handed a vscode-chat token to a CLI session
        # and GitHub answered 401 "unable to validate HMAC for the given
        # Copilot-Integration-ID".
        self._cached_by_integration: dict[str, CopilotAPIToken] = {}

    @property
    def _cached(self) -> CopilotAPIToken | None:
        """Back-compat view of the default integration's token (tests/callers)."""
        return self._cached_by_integration.get(resolve_copilot_integration_id())

    @_cached.setter
    def _cached(self, value: CopilotAPIToken | None) -> None:
        key = resolve_copilot_integration_id()
        if value is None:
            self._cached_by_integration.pop(key, None)
        else:
            self._cached_by_integration[key] = value

    async def get_api_token(self, *, integration_id: str | None = None) -> CopilotAPIToken:
        key = resolve_copilot_integration_id(integration_id)
        explicit_api_token = os.environ.get("GITHUB_COPILOT_API_TOKEN", "").strip()
        refresh_oauth_token = os.environ.get(_REFRESH_OAUTH_TOKEN_ENV_VAR, "").strip()
        if explicit_api_token and not refresh_oauth_token:
            return CopilotAPIToken(
                token=explicit_api_token,
                expires_at=time.time() + 3600,
                api_url=_configured_api_url(),
            )

        cached = self._cached_by_integration.get(key)
        if cached is not None and cached.is_valid:
            return cached

        async with self._lock:
            cached = self._cached_by_integration.get(key)
            if cached is not None and cached.is_valid:
                return cached

            if explicit_api_token and refresh_oauth_token:
                if cached is None:
                    seeded_expires_at = _parse_expiry(os.environ.get(_API_TOKEN_EXPIRES_AT_ENV_VAR))
                    seeded = CopilotAPIToken(
                        token=explicit_api_token,
                        expires_at=seeded_expires_at if seeded_expires_at is not None else 0.0,
                        api_url=_configured_api_url(),
                    )
                    self._cached_by_integration[key] = seeded
                    if seeded.is_valid:
                        return seeded
                exchanged = await self._exchange_token(refresh_oauth_token, integration_id=key)
                self._cached_by_integration[key] = exchanged
                return exchanged

            oauth_token = read_cached_oauth_token()
            if not oauth_token:
                raise RuntimeError("No GitHub Copilot OAuth token is available.")

            if not _should_exchange_oauth_token():
                direct_token = CopilotAPIToken(
                    token=oauth_token,
                    expires_at=time.time() + 3600,
                    api_url=_configured_api_url(),
                )
                self._cached_by_integration[key] = direct_token
                return direct_token

            exchanged = await self._exchange_token(oauth_token, integration_id=key)
            self._cached_by_integration[key] = exchanged
            return exchanged

    async def _exchange_token(
        self, oauth_token: str, *, integration_id: str | None = None
    ) -> CopilotAPIToken:
        headers = _copilot_token_exchange_headers(oauth_token, integration_id=integration_id)
        payload = await asyncio.to_thread(self._exchange_token_sync, headers)
        token = str(payload.get("token") or "").strip()
        if not token:
            raise RuntimeError("Copilot token exchange returned an empty token.")

        expires_at = _parse_expiry(payload.get("expires_at")) or (time.time() + 1800)
        api_url = await asyncio.to_thread(
            _api_url_from_exchange_payload,
            payload,
            oauth_token=oauth_token,
        )
        refresh_in = payload.get("refresh_in")
        sku = payload.get("sku")
        return CopilotAPIToken(
            token=token,
            expires_at=expires_at,
            api_url=api_url,
            refresh_in=int(refresh_in) if isinstance(refresh_in, int | float) else None,
            sku=str(sku) if isinstance(sku, str) and sku.strip() else None,
        )

    @staticmethod
    def _exchange_token_sync(headers: dict[str, str]) -> dict[str, Any]:
        request = urllib_request.Request(_token_exchange_url(), headers=headers, method="GET")
        try:
            with _urlopen(request, timeout=10.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    return {}
                # Every exchange funnels through here, so this is the one place
                # that sees GitHub's advertised completions host (#3076).
                _remember_completions_endpoint(payload)
                return payload
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Copilot token exchange failed with HTTP {exc.code}: {body}"
            ) from exc


_provider: CopilotTokenProvider | None = None


def get_copilot_token_provider() -> CopilotTokenProvider:
    """Return the shared Copilot token provider."""

    global _provider
    if _provider is None:
        _provider = CopilotTokenProvider()
    return _provider


def _is_copilot_api_token(token: str) -> bool:
    """Return True when the token looks like a short-lived Copilot API token.

    Copilot API tokens currently use the "tid_" prefix.
    GitHub OAuth tokens (for example "gho_", "ghs_", "ghp_", "github_pat_")
    should be exchanged and must not be forwarded directly to the Copilot
    *subscription/user-info* APIs (see resolve_subscription_bearer_token_details()),
    which is the only caller of this helper. It intentionally stays narrow:
    broadening it here would also change subscription-resolution behavior,
    which is a separate concern from forwarding a bearer token for
    chat-completion/inference requests -- see _is_forwardable_copilot_bearer_token()
    for that case.
    """
    normalized = token.strip()
    if not normalized:
        return False

    if (
        normalized.startswith("gho_")
        or normalized.startswith("ghs_")
        or normalized.startswith("ghp_")
        or normalized.startswith("github_pat_")
    ):
        return False

    return normalized.startswith("tid_")


def _is_forwardable_copilot_bearer_token(token: str) -> bool:
    """Return True when a bearer token should be forwarded as-is for Copilot inference.

    Unlike _is_copilot_api_token() (used only for subscription/user-info
    resolution), this accepts both short-lived Copilot API tokens (`tid_`)
    AND GitHub OAuth tokens (`gho_`, `ghs_`, `ghp_`, `github_pat_`) as valid,
    forwardable Copilot bearer credentials for chat-completion/inference
    requests.

    Verified live in two independent reports that a caller-supplied `gho_`
    token is already valid and correctly entitled when sent directly to
    the real Copilot inference API, and that replacing it with Headroom's
    own independently-fetched/exchanged token is actively harmful:

    - A live Copilot CLI session's own `gho_`-prefixed token worked
      end-to-end for model "claude-sonnet-5" when forwarded unchanged, but
      got `400 model_not_supported` once Headroom swapped in a different,
      less-entitled re-exchanged token for the exact same request.
    - headroomlabs-ai/headroom#1813: OpenCode's native Copilot integration
      sends its own `gho_` token directly; replacing it changes the
      effective client/integrator lane Copilot's backend sees, breaking
      model discovery/inference parity with native (non-proxied) behavior.

    Only genuinely non-Copilot-shaped or blank/whitespace-only tokens fall
    through to replacement.
    """
    normalized = token.strip()
    if not normalized:
        return False

    return normalized.startswith(("tid_", "gho_", "ghs_", "ghp_", "github_pat_"))


def _token_kind(token: str) -> str:
    """Return a non-sensitive label for the token type, safe to log."""
    t = token.strip()
    for prefix in ("tid_", "gho_", "ghs_", "ghp_", "github_pat_"):
        if t.startswith(prefix):
            return prefix + "***"
    return "unknown" if t else "empty"


def _is_managed_copilot_seeded_bearer(token: str) -> bool:
    """Return True when the incoming bearer is the proxy's seeded wrapper token."""

    refresh_oauth_token = os.environ.get(_REFRESH_OAUTH_TOKEN_ENV_VAR, "").strip()
    explicit_api_token = os.environ.get("GITHUB_COPILOT_API_TOKEN", "").strip()
    normalized = token.strip()
    return bool(
        refresh_oauth_token
        and explicit_api_token
        and normalized
        and normalized == explicit_api_token
    )


async def apply_copilot_api_auth(headers: dict[str, str], *, url: str) -> dict[str, str]:
    """Apply Copilot auth headers for GitHub Copilot API requests."""
    resolved = dict(headers)
    # Both Copilot surfaces need credentials. Gating on the chat host alone left
    # inline completions unauthenticated: the request reached
    # copilot-proxy.githubusercontent.com with no Authorization header, and that
    # host answers 401 (#3076).
    if not is_copilot_upstream_url(url):
        return resolved

    # Read the CLIENT's integration ID before any default is applied, so the
    # credential we mint below can be bound to the surface that actually made
    # the call rather than to whatever this proxy happens to default to.
    client_integration_id = _header_value(resolved, "Copilot-Integration-Id")
    integration_id = resolve_copilot_integration_id(client_integration_id)

    for name, value in _copilot_chat_header_defaults(integration_id).items():
        _set_header_default(resolved, name, value)

    incoming_auth = next((v for k, v in resolved.items() if k.lower() == "authorization"), None)
    if incoming_auth:
        scheme, _, raw_token = incoming_auth.partition(" ")
        if (
            scheme.lower() == "bearer"
            and raw_token
            and _is_forwardable_copilot_bearer_token(raw_token)
        ):
            if _is_managed_copilot_seeded_bearer(raw_token):
                logger.info(
                    "apply_copilot_api_auth: managed seed token kind=%s, will replace",
                    _token_kind(raw_token),
                )
            else:
                logger.info(
                    "apply_copilot_api_auth: passing through client token kind=%s",
                    _token_kind(raw_token),
                )
                for key in list(resolved):
                    if key.lower() == "x-api-key":
                        resolved.pop(key)
                return resolved
        logger.info(
            "apply_copilot_api_auth: incoming token not suitable (kind=%s), will replace",
            _token_kind(raw_token) if raw_token else "none",
        )

    token = await get_copilot_token_provider().get_api_token(integration_id=integration_id)
    for key in list(resolved):
        if key.lower() in {"authorization", "x-api-key"}:
            resolved.pop(key)
    resolved["Authorization"] = f"Bearer {token.token}"
    # The credential and the integration ID must leave together. Until now the
    # ID was applied with set-default semantics BEFORE this branch was chosen,
    # so replacing the client's token left its ID in place next to OUR token —
    # a pair GitHub cannot HMAC-validate:
    #
    #   401 unauthorized: unable to validate HMAC for the given
    #       Copilot-Integration-ID
    #
    # It surfaced first on model discovery (`Failed to fetch models`), which
    # left the client falling back to its built-in model list. Overwrite here,
    # never above: the pass-through branch returns before this point and keeps
    # the client's own ID beside the client's own token, which is equally the
    # matched pair.
    _overwrite_header(resolved, "Copilot-Integration-Id", integration_id)
    return resolved
