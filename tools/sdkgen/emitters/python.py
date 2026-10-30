"""Python output adapter."""

from __future__ import annotations

from typing import Any


def emit(document: dict[str, Any]) -> dict[str, bytes]:
    from ..emit import emit_python, operations

    files = emit_python(document["components"]["schemas"], operations(document))
    return {f"python/{path}": content for path, content in files.items()}
