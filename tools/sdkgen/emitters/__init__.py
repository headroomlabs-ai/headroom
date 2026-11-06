"""Language-emitter registry for the deterministic SDK pilot."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

Emitter: TypeAlias = Callable[[dict[str, Any]], dict[str, bytes]]


def default_emitters() -> tuple[Emitter, ...]:
    """Return emitters in stable output order without import cycles."""
    from .dotnet import emit as emit_dotnet
    from .go import emit as emit_go
    from .python import emit as emit_python
    from .rust import emit as emit_rust
    from .typescript import emit as emit_typescript

    return (emit_python, emit_typescript, emit_go, emit_rust, emit_dotnet)
