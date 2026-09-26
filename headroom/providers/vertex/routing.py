"""Provider-owned Vertex publisher route dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from fastapi import Request

VertexRouteKind = Literal[
    "anthropic_messages",
    "gemini_count_tokens",
    "gemini_generate_content",
    "passthrough",
]


@dataclass(frozen=True)
class VertexRouteDecision:
    """Resolved proxy handler for a Vertex publisher endpoint."""

    kind: VertexRouteKind
    provider: str
    action: str
    force_stream: bool = False


def resolve_vertex_route(publisher: str, action: str) -> VertexRouteDecision:
    """Resolve provider-specific Vertex behavior from publisher and RPC action."""
    provider = f"vertex:{publisher}"
    routes: dict[tuple[str, str], VertexRouteDecision] = {
        ("google", "generateContent"): VertexRouteDecision(
            "gemini_generate_content",
            "vertex:google",
            action,
        ),
        ("google", "streamGenerateContent"): VertexRouteDecision(
            "gemini_generate_content",
            "vertex:google",
            action,
        ),
        ("google", "countTokens"): VertexRouteDecision(
            "gemini_count_tokens",
            "vertex:google",
            action,
        ),
        ("anthropic", "rawPredict"): VertexRouteDecision(
            "anthropic_messages",
            "vertex:anthropic",
            action,
        ),
        ("anthropic", "streamRawPredict"): VertexRouteDecision(
            "anthropic_messages",
            "vertex:anthropic",
            action,
            force_stream=True,
        ),
    }
    return routes.get((publisher, action), VertexRouteDecision("passthrough", provider, action))


async def dispatch_vertex_publisher_request(
    proxy: Any,
    request: Request,
    *,
    publisher: str,
    action: str,
    model: str,
    upstream_base_url: str,
) -> Any:
    """Dispatch a Vertex publisher request through the provider-owned route matrix."""
    route = resolve_vertex_route(publisher, action)

    if route.kind == "gemini_generate_content":
        return await proxy.handle_gemini_generate_content(
            request,
            model,
            upstream_base_url,
            route.provider,
        )
    if route.kind == "gemini_count_tokens":
        return await proxy.handle_gemini_count_tokens(
            request,
            model,
            upstream_base_url,
            route.provider,
        )
    if route.kind == "anthropic_messages":
        return await proxy.handle_anthropic_messages(
            request,
            upstream_base_url,
            route.provider,
            model,
            route.force_stream,
        )

    return await proxy.handle_passthrough(
        request,
        upstream_base_url,
        route.action,
        route.provider,
    )
