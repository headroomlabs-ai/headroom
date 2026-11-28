"""TypeScript output adapter."""

from __future__ import annotations

from typing import Any


def emit(document: dict[str, Any]) -> dict[str, bytes]:
    from ..emit import emit_typescript, operations

    files = emit_typescript(document["components"]["schemas"], operations(document))
    return {f"typescript/{path}": content for path, content in files.items()}
