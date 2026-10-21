"""Compression summary generator — describes what was dropped.

When code is compressed, the LLM needs to know what it's missing. Instead of
just "[5 bodies compressed]", we list the removed function/class names (from
the AST, language-agnostic): "5 bodies compressed: authenticate(), ...".

This helps the LLM decide whether to call headroom_retrieve and what to search for.

Used by CodeCompressor.
"""

from __future__ import annotations

import re


def summarize_compressed_code(
    function_bodies: list[tuple[str, str, int]],
    compressed_bodies_count: int,
) -> str:
    """Generate a summary of compressed code sections from AST data.

    Language-agnostic: works with any language tree-sitter supports because
    it reads function signatures directly from the CodeCompressor's AST output.

    Args:
        function_bodies: List of (signature, body, line) from CodeStructure.
        compressed_bodies_count: Number of bodies that were compressed.

    Returns:
        Summary string like "5 bodies compressed: authenticate(), validate_token(), ..."
        or empty string.
    """
    if not function_bodies or compressed_bodies_count == 0:
        return ""

    # Extract short names from signatures
    names = []
    for sig, _body, _line in function_bodies:
        name = _extract_name_from_signature(sig)
        if name:
            names.append(name)

    if not names:
        return f"{compressed_bodies_count} function bodies compressed"

    # Show up to 6 names
    shown = names[:6]
    result = f"{compressed_bodies_count} bodies compressed: {', '.join(shown)}"
    if len(names) > 6:
        result += f" (+{len(names) - 6} more)"
    return result


# ---- Internal helpers ----


def _extract_name_from_signature(sig: str) -> str:
    """Extract the function/method name from a signature string.

    Works for any language because it looks for common patterns:
    - Python: "def authenticate(", "async def fetch("
    - JavaScript: "function authenticate(", "async function fetch("
    - Go: "func (s *Server) HandleRequest("
    - Rust: "fn authenticate("
    - Java/C++: "public void authenticate("
    """
    # Try common function definition patterns
    match = re.search(r"(?:def|func|fn|function)\s+(?:\([^)]*\)\s*)?(\w+)", sig)
    if match:
        return match.group(1) + "()"

    # Try method patterns: "public static void methodName("
    match = re.search(r"(?:public|private|protected|static|async|export)\s+.*?(\w+)\s*\(", sig)
    if match:
        return match.group(1) + "()"

    # Try class patterns
    match = re.search(r"class\s+(\w+)", sig)
    if match:
        return match.group(1)

    # Fallback: last word before (
    match = re.search(r"(\w+)\s*\(", sig)
    if match:
        return match.group(1) + "()"

    return ""
