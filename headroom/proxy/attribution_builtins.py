"""Built-in attribution dimensions.

Each resolver reads a signal the proxy already understands; none of them adds a
new trust assumption:

``project``
    ``X-Headroom-Project`` (sent by ``headroom wrap``) or a ``/p/<name>``
    base-URL prefix, exactly as :mod:`headroom.proxy.project_context` reads it.
``client``
    The harness, from :func:`headroom.proxy.auth_mode.classify_client`.
``tenant``
    The tenant-id header TOIN partitions on (``HEADROOM_TENANT_KEY_HEADER``),
    only when a caller sent one.
"""

from __future__ import annotations

from headroom.proxy.attribution import AttributionRequest, _register_if_absent


def resolve_project(request: AttributionRequest) -> str | None:
    from headroom.proxy.project_policy import classify_project

    return classify_project(request.headers) or request.path_project


def resolve_client(request: AttributionRequest) -> str | None:
    from headroom.proxy.auth_mode import classify_client

    return classify_client(request.headers)


def resolve_tenant(request: AttributionRequest) -> str | None:
    import os

    from headroom.proxy.tenant_key import (
        DEFAULT_TENANT_KEY_HEADER,
        TENANT_KEY_HEADER_ENV_VAR,
        _sanitize_tenant_key,
    )

    header = os.environ.get(TENANT_KEY_HEADER_ENV_VAR, "").strip() or DEFAULT_TENANT_KEY_HEADER
    raw = request.header(header)
    if not raw:
        return None
    return _sanitize_tenant_key(raw) or None


BUILTIN_RESOLVERS = (
    ("project", resolve_project),
    ("client", resolve_client),
    ("tenant", resolve_tenant),
)


def install_builtin_dimensions() -> None:
    """Register the built-ins. Idempotent; an extension's resolver is kept."""
    for name, resolver in BUILTIN_RESOLVERS:
        _register_if_absent(name, resolver)


__all__ = [
    "BUILTIN_RESOLVERS",
    "install_builtin_dimensions",
]
