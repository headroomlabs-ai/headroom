"""Tool-result interceptors.

An interceptor rewrites a single tool_result's text before it reaches the
model. Each interceptor is self-contained: declare a `matches()` predicate
and a `transform()` function, register it in the `INTERCEPTORS` list, and
the proxy pipeline will call it automatically.

Adding a new interceptor later is one file plus one `register()` call — no
proxy or metrics changes required.
"""

# Side-effect: register the built-in interceptors.
from . import astgrep  # noqa: F401
from .base import (
    INTERCEPTORS,
    InterceptionResult,
    ToolResultInterceptor,
    ToolResultInterceptorTransform,
    TransformSpan,
    apply_to_messages,
    interceptor_failure_counts,
    register,
)

__all__ = [
    "INTERCEPTORS",
    "InterceptionResult",
    "ToolResultInterceptor",
    "ToolResultInterceptorTransform",
    "TransformSpan",
    "apply_to_messages",
    "interceptor_failure_counts",
    "register",
]
