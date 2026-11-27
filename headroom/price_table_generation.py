"""The generation of LiteLLM's price table, and memoisation keyed on it.

A price refresh replaces ``litellm.model_cost`` while request threads may be
pricing against it. A lookup that read the old table and finishes after the
refresh cleared Headroom's caches would store old rates again, and they would
serve every later request. Each published table therefore gets a new
generation, and memoised results are keyed on the generation read when the
lookup started: a result computed from a replaced table lands under an old
generation, where no later lookup reads it.

Outside :mod:`headroom.pricing`, whose package import pulls in litellm (~4s):
:mod:`headroom.proxy.savings_tracker` imports this at module scope, on the
proxy's startup path.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_generation = 0


def price_table_generation() -> int:
    """The generation of the price table published last."""
    return _generation


def advance_price_table_generation() -> None:
    """Mark a newly published price table. Call after binding it, never before."""
    global _generation
    _generation += 1


class GenerationCached(Generic[P, R]):
    """``functools.lru_cache`` keyed on the price-table generation as well as the arguments."""

    def __init__(self, fn: Callable[P, R], maxsize: int) -> None:
        self._fn = fn
        self._cached = functools.lru_cache(maxsize=maxsize)(self._call_at)
        functools.update_wrapper(self, fn)

    def _call_at(self, generation: int, args: tuple[Any, ...], kwargs: tuple[Any, ...]) -> R:
        return self._fn(*args, **dict(kwargs))  # type: ignore[arg-type]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._cached(_generation, args, tuple(sorted(kwargs.items())))

    def cache_clear(self) -> None:
        self._cached.cache_clear()

    def cache_info(self) -> Any:
        return self._cached.cache_info()


def generation_cached(maxsize: int) -> Callable[[Callable[P, R]], GenerationCached[P, R]]:
    """Decorator: memoise like ``lru_cache(maxsize)``, per price-table generation."""

    def decorate(fn: Callable[P, R]) -> GenerationCached[P, R]:
        return GenerationCached(fn, maxsize)

    return decorate
