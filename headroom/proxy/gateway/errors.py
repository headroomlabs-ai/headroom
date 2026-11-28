"""Gateway errors that are safe to map onto public protocol responses."""

from typing import Any


class GatewayConfigurationError(ValueError):
    """Raised when a gateway configuration violates a cross-field invariant."""


class GatewayPublicError(Exception):
    """A bounded public error that never embeds secret or provider detail."""

    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class GatewayAuthError(GatewayPublicError):
    """Caller authentication or browser-boundary failure."""


class GatewayAuthorizationError(GatewayPublicError):
    """Authenticated caller lacks the requested scope, route, or capability."""


class GatewayCredentialUnavailable(GatewayPublicError):
    """A configured credential source cannot currently issue a lease."""


class GatewayEgressDenied(GatewayPublicError):
    """The final upstream destination is outside a credential's audience."""


def protocol_error_payload(protocol: str, error: GatewayPublicError) -> dict[str, Any]:
    """Render only the bounded gateway message, never a provider exception/body."""
    if protocol in {"gemini-generate", "vertex-generate"}:
        return {
            "error": {
                "code": error.status_code,
                "status": {
                    400: "INVALID_ARGUMENT",
                    401: "UNAUTHENTICATED",
                    403: "PERMISSION_DENIED",
                    404: "NOT_FOUND",
                    429: "RESOURCE_EXHAUSTED",
                    502: "UNAVAILABLE",
                    503: "UNAVAILABLE",
                }.get(error.status_code, "INTERNAL"),
                "message": error.message,
                "details": [{"reason": error.code}],
            }
        }
    result: dict[str, Any] = {
        "error": {"type": "gateway_error", "code": error.code, "message": error.message}
    }
    if protocol == "anthropic-messages":
        result["type"] = "error"
        result["error"]["type"] = (
            "api_error" if error.status_code >= 500 else "invalid_request_error"
        )
    elif protocol == "openai-responses":
        result["type"] = "error"
    return result
