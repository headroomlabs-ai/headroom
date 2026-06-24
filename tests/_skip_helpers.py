"""Helpers for skipping tests when external model dependencies are unavailable."""

from __future__ import annotations

from collections.abc import Iterator


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its direct cause/context chain."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def external_model_skip_reason(exc: BaseException) -> str | None:
    """Return a pytest skip reason for transient or offline model dependency errors."""
    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None

    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:  # pragma: no cover
        LocalEntryNotFoundError = None

    for candidate in _iter_exception_chain(exc):
        if httpx is not None and isinstance(candidate, httpx.ReadTimeout):
            return "Skipped due to network timeout (flaky CI)"

        if LocalEntryNotFoundError is not None and isinstance(candidate, LocalEntryNotFoundError):
            return "Skipped because required Hugging Face model files are unavailable offline"

        if isinstance(candidate, OSError):
            message = str(candidate)
            if (
                "couldn't connect to 'https://huggingface.co'" in message
                and "couldn't find them in the cached files" in message
            ):
                return "Skipped because required Hugging Face model files are unavailable offline"

    return None
