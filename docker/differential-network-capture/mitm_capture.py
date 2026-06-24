"""mitmproxy addon that writes sanitized HTTP exchanges as JSONL."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from mitmproxy import http

LANE = os.environ.get("CAPTURE_LANE", "unknown")
OUTPUT = Path(os.environ.get("CAPTURE_OUTPUT", f"/captures/{LANE}.jsonl"))
INCLUDE_HOSTS = {
    host.strip().lower()
    for host in os.environ.get("CAPTURE_INCLUDE_HOSTS", "api.anthropic.com").split(",")
    if host.strip()
}
BODY_BYTES = int(os.environ.get("CAPTURE_BODY_BYTES", "262144"))
SENSITIVE_HEADER_PARTS = ("authorization", "api-key", "apikey", "token", "secret", "cookie")
SENSITIVE_QUERY_PARTS = ("key", "token", "secret", "signature", "code")
_sequence = 0


def _redact_headers(headers: http.Headers) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in headers.items(multi=True):
        if any(part in key.lower() for part in SENSITIVE_HEADER_PARTS):
            result[key] = "<redacted>"
        else:
            result[key] = value
    return result


def _sanitize_url(url: str) -> str:
    parsed = urlsplit(url)
    pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if any(part in key.lower() for part in SENSITIVE_QUERY_PARTS):
            pairs.append((key, "<redacted>"))
        else:
            pairs.append((key, value))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(pairs), ""))


def _request_json(content: bytes) -> object | None:
    try:
        return json.loads(content.decode("utf-8"))
    except Exception:
        return None


def response(flow: http.HTTPFlow) -> None:
    global _sequence
    host = flow.request.pretty_host.lower()
    if INCLUDE_HOSTS and host not in INCLUDE_HOSTS:
        return

    _sequence += 1
    request_body = flow.request.raw_content or b""
    response_body = flow.response.raw_content if flow.response else b""
    record = {
        "lane": LANE,
        "sequence": _sequence,
        "timestamp": time.time(),
        "method": flow.request.method,
        "url": _sanitize_url(flow.request.pretty_url),
        "host": flow.request.pretty_host,
        "request_headers": _redact_headers(flow.request.headers),
        "request_body_size": len(request_body),
        "request_body_sha256": hashlib.sha256(request_body).hexdigest() if request_body else None,
        "request_body_b64": base64.b64encode(request_body[:BODY_BYTES]).decode("ascii"),
        "request_body_truncated": len(request_body) > BODY_BYTES,
        "request_json": _request_json(request_body),
        "response_status": flow.response.status_code if flow.response else None,
        "response_headers": _redact_headers(flow.response.headers) if flow.response else {},
        "response_body_size": len(response_body),
        "response_body_sha256": hashlib.sha256(response_body).hexdigest()
        if response_body
        else None,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        fh.write("\n")
