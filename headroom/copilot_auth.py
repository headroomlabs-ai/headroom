"""GitHub Copilot OAuth discovery and API-token exchange helpers."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import logging
import os
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from headroom import paths
from headroom.copilot_linux_secret import read_copilot_oauth_token as read_linux_secret_token
from headroom.copilot_macos_keychain import read_copilot_oauth_token as read_macos_keychain_token

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.githubcopilot.com"
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


def token_fingerprint(token: str) -> str:
    """Return a stable non-secret fingerprint for comparing token handoffs."""

    digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).hexdigest()
    return f"sha256:{digest[:12]}"


def _github_host() -> str:
    return (os.environ.get("GITHUB_COPILOT_HOST") or DEFAULT_GITHUB_HOST).strip().lower()


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
    parsed = urlparse(f"https://{normalized}")
    return (parsed.hostname or normalized.split("/", 1)[0]).lower()


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
    if not host or host in {"github.com", "www.github.com", "api.github.com"}:
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


def _configured_api_url() -> str:
    api_url = os.environ.get("GITHUB_COPILOT_API_URL", "").strip()
    if api_url:
        return api_url.rstrip("/")

    enterprise_domain = _configured_enterprise_domain()
    if enterprise_domain:
        return copilot_api_url_from_enterprise_url(enterprise_domain).rstrip("/")

    return DEFAULT_API_URL


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
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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
    body = json.dumps(
        {
            "client_id": COPILOT_CHAT_OAUTH_CLIENT_ID,
            "scope": "read:user",
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib_request.Request(
        urls["device_code"],
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": _DEFAULT_USER_AGENT,
        },
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=timeout) as response:
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
        body = json.dumps(
            {
                "client_id": COPILOT_CHAT_OAUTH_CLIENT_ID,
                "device_code": device_code,
                "grant_type": _DEVICE_CODE_GRANT_TYPE,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib_request.Request(
            urls["access_token"],
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": _DEFAULT_USER_AGENT,
            },
            method="POST",
        )
        with urllib_request.urlopen(request, timeout=timeout) as response:
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


def _copilot_chat_header_defaults() -> dict[str, str]:
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
        "Copilot-Integration-Id": os.environ.get(
            "GITHUB_COPILOT_INTEGRATION_ID",
            _DEFAULT_COPILOT_INTEGRATION_ID,
        ).strip()
        or _DEFAULT_COPILOT_INTEGRATION_ID,
    }


def _set_header_default(headers: dict[str, str], name: str, value: str) -> None:
    """Set a header default without duplicating case-insensitive equivalents."""

    name_lower = name.lower()
    if any(existing.lower() == name_lower for existing in headers):
        return
    headers[name] = value


def _copilot_token_exchange_headers(oauth_token: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {oauth_token}",
        **_copilot_chat_header_defaults(),
    }


def _api_url_from_payload(payload: dict[str, Any] | None) -> str | None:
    endpoints = payload.get("endpoints") if isinstance(payload, dict) else None
    api_url = endpoints.get("api") if isinstance(endpoints, dict) else None
    if isinstance(api_url, str) and api_url.strip():
        return api_url.strip().rstrip("/")
    return None


def _subscription_api_url_from_user_info_payload(payload: dict[str, Any] | None) -> str:
    api_url = _api_url_from_payload(payload)
    if not api_url:
        return _configured_api_url()

    host = urlparse(api_url).netloc.lower()
    if host in {"api.githubcopilot.com", "api.individual.githubcopilot.com"}:
        return _configured_api_url()
    if host.endswith(".githubcopilot.com"):
        return api_url
    return _configured_api_url()


def _subscription_api_url_from_user_info(oauth_token: str) -> str:
    return _subscription_api_url_from_user_info_payload(_fetch_copilot_user_info(oauth_token))


def _api_url_from_exchange_payload(payload: dict[str, Any], *, oauth_token: str) -> str:
    configured = _configured_api_url()
    if configured != DEFAULT_API_URL:
        return configured

    api_url = _api_url_from_payload(payload)
    if api_url:
        return api_url

    return _subscription_api_url_from_user_info(oauth_token)


def _subscription_resolution(
    *,
    token: str,
    source: str,
    confidence: str,
    api_url: str,
) -> CopilotSubscriptionTokenResolution:
    return CopilotSubscriptionTokenResolution(
        token=token,
        source=source,
        confidence=confidence,
        api_url=api_url,
        token_fingerprint=token_fingerprint(token),
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


def build_copilot_upstream_url(base_url: str, path: str) -> str:
    """Build an upstream URL, normalizing GitHub Copilot's non-/v1 path layout."""

    normalized_base = base_url.rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    if is_copilot_api_url(normalized_base) and normalized_path.startswith("/v1/"):
        normalized_path = normalized_path[3:]
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
        with urllib_request.urlopen(request, timeout=10.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logger.debug("Unable to resolve Copilot API URL from user info: %s", exc)
        return None

    return payload if isinstance(payload, dict) else None


class CopilotTokenProvider:
    """Resolve and cache short-lived Copilot API tokens."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._cached: CopilotAPIToken | None = None

    async def get_api_token(self) -> CopilotAPIToken:
        explicit_api_token = os.environ.get("GITHUB_COPILOT_API_TOKEN", "").strip()
        if explicit_api_token:
            return CopilotAPIToken(
                token=explicit_api_token,
                expires_at=time.time() + 3600,
                api_url=_configured_api_url(),
            )

        cached = self._cached
        if cached is not None and cached.is_valid:
            return cached

        async with self._lock:
            cached = self._cached
            if cached is not None and cached.is_valid:
                return cached

            oauth_token = read_cached_oauth_token()
            if not oauth_token:
                raise RuntimeError("No GitHub Copilot OAuth token is available.")

            if not _should_exchange_oauth_token():
                direct_token = CopilotAPIToken(
                    token=oauth_token,
                    expires_at=time.time() + 3600,
                    api_url=_configured_api_url(),
                )
                self._cached = direct_token
                return direct_token

            exchanged = await self._exchange_token(oauth_token)
            self._cached = exchanged
            return exchanged

    async def _exchange_token(self, oauth_token: str) -> CopilotAPIToken:
        headers = _copilot_token_exchange_headers(oauth_token)
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
            with urllib_request.urlopen(request, timeout=10.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload if isinstance(payload, dict) else {}
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
    should be exchanged and must not be forwarded directly.
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


def _token_kind(token: str) -> str:
    """Return a non-sensitive label for the token type, safe to log."""
    t = token.strip()
    for prefix in ("tid_", "gho_", "ghs_", "ghp_", "github_pat_"):
        if t.startswith(prefix):
            return prefix + "***"
    return "unknown" if t else "empty"


async def apply_copilot_api_auth(headers: dict[str, str], *, url: str) -> dict[str, str]:
    """Apply Copilot auth headers for GitHub Copilot API requests."""
    resolved = dict(headers)
    if not is_copilot_api_url(url):
        return resolved

    for name, value in _copilot_chat_header_defaults().items():
        _set_header_default(resolved, name, value)

    incoming_auth = next((v for k, v in resolved.items() if k.lower() == "authorization"), None)
    if incoming_auth:
        scheme, _, raw_token = incoming_auth.partition(" ")
        if scheme.lower() == "bearer" and raw_token and _is_copilot_api_token(raw_token):
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

    token = await get_copilot_token_provider().get_api_token()
    for key in list(resolved):
        if key.lower() in {"authorization", "x-api-key"}:
            resolved.pop(key)
    resolved["Authorization"] = f"Bearer {token.token}"
    return resolved
