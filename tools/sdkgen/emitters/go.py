"""Go output adapter."""

from __future__ import annotations

from typing import Any


def emit(document: dict[str, Any]) -> dict[str, bytes]:
    from ..emit import emit_go, operations

    files = emit_go(document["components"]["schemas"], operations(document))
    return {f"go/{path}": content for path, content in files.items()}
