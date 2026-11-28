"""Build-time wire contracts. This module has no imports from the proxy runtime.

The marker returns the original handler unchanged: it does not wrap, validate,
register, or alter FastAPI response_model behavior. tools.sdkgen reads its AST.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def sdk_operation(
    *,
    operation_id: str,
    response: type,
    access: str,
    request: type | None = None,
    errors: dict[int, type] | None = None,
) -> Callable[[F], F]:
    """Declare a reviewed wire contract without changing the decorated handler."""

    def mark(function: F) -> F:
        return function

    return mark


class RetrieveRequest(TypedDict):
    hash: str


class RetrieveResponse(TypedDict):
    # All keys are emitted on success. tool_name is required AND nullable.
    hash: str
    original_content: str
    original_tokens: int
    original_item_count: int
    compressed_item_count: int
    tool_name: str | None
    retrieval_count: int


class RetrievalError(TypedDict):
    detail: str
