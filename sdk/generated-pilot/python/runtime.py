"""Handwritten JSON transport kernel, copied verbatim by sdkgen. No retries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

SCHEMAS = json.loads(Path(__file__).with_name("schemas.json").read_text(encoding="utf-8"))


class ProtocolError(ValueError):
    """The wire value does not satisfy the declared contract."""


class APIError(Exception):
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        super().__init__(f"Headroom HTTP {status}")  # Never put response payloads into logs.
        self.status = status
        self.headers = headers
        self.body = body


def validate(value: Any, schema: dict[str, Any], path: str = "$", depth: int = 0) -> None:
    if depth > 100:
        raise ProtocolError(f"{path}: maximum JSON nesting exceeded")
    if "$ref" in schema:
        validate(value, SCHEMAS[schema["$ref"].rsplit("/", 1)[1]], path, depth + 1)
        return
    if "anyOf" in schema:
        for choice in schema["anyOf"]:
            try:
                validate(value, choice, path, depth + 1)
                return
            except ProtocolError:
                pass
        raise ProtocolError(f"{path}: value is outside nullable union")
    typ = schema.get("type")
    valid = {
        "string": lambda: isinstance(value, str),
        "integer": lambda: type(value) is int,
        "number": lambda: (
            type(value) in (int, float) and (type(value) is int or math.isfinite(value))
        ),
        "boolean": lambda: type(value) is bool,
        "null": lambda: value is None,
        "array": lambda: isinstance(value, list),
        "object": lambda: isinstance(value, dict),
    }
    if typ is not None and (typ not in valid or not valid[typ]()):
        raise ProtocolError(f"{path}: expected {typ}")
    if "enum" in schema and not any(type(value) is type(x) and value == x for x in schema["enum"]):
        raise ProtocolError(f"{path}: unknown enum value")
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ProtocolError(f"{path}: object keys must be strings")
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ProtocolError(f"{path}.{key}: required field missing")
        for key, item in value.items():
            rule = props.get(key, schema.get("additionalProperties", True))
            if rule is False:
                raise ProtocolError(f"{path}.{key}: additional property forbidden")
            validate(item, rule if isinstance(rule, dict) else {}, f"{path}.{key}", depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{path}[{index}]", depth + 1)
    elif type(value) is float and not math.isfinite(value):
        raise ProtocolError(f"{path}: non-finite JSON number")
    elif not (value is None or type(value) in (str, bool, int, float)):
        raise ProtocolError(f"{path}: not a JSON value")


def path_segment(value: str) -> str:
    if not isinstance(value, str) or value in {"", ".", ".."}:
        raise ValueError("Path segment must be a nonempty string, not a dot segment")
    return quote(value, safe="")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


class Transport:
    def __init__(
        self,
        base_url: str = "http://localhost:8787",
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30,
        max_response_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        url = urlsplit(base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) URL without credentials, query, or fragment"
            )
        if any(ord(c) < 33 for c in base_url) or timeout <= 0 or max_response_bytes <= 0:
            raise ValueError("Invalid URL, timeout, or response size limit")
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        # Explicitly bypass environment proxies; avoid accidental CCR payload disclosure.
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def request(
        self, method: str, path: str, body: Any, request_model: str | None, response_model: str
    ) -> Any:
        payload = None
        if request_model is not None:
            validate(body, SCHEMAS[request_model])
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {**self.headers, "Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=payload, headers=headers, method=method)
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except HTTPError as exc:
            response = exc
        with response:
            raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise ProtocolError("Response exceeds configured byte limit")
            status = response.code
            response_headers = dict(response.headers.items())
            if not 200 <= status < 300:
                raise APIError(status, response_headers, raw)
            media = response.headers.get_content_type()
            if media != "application/json" and not media.endswith("+json"):
                raise ProtocolError("Expected a JSON Content-Type")
            try:

                def invalid_constant(_: str) -> None:
                    raise ValueError("nonstandard JSON constant")

                value = json.loads(raw.decode("utf-8"), parse_constant=invalid_constant)
            except (UnicodeError, ValueError) as exc:
                raise ProtocolError("Invalid JSON response") from exc
            validate(value, SCHEMAS[response_model])
            return value
