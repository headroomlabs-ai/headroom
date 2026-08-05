"""Code structure handler using AST parsing.

Extracts structural elements from source code:
- Import statements
- Function/method signatures
- Class definitions
- Type annotations
- Decorators

Function bodies are marked as compressible while preserving signatures.
This enables the LLM to see all available functions/methods while body
implementations are compressed.

Uses tree-sitter for parsing when available, falls back to regex patterns.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Any

from headroom.compression.handlers.base import BaseStructureHandler, HandlerResult
from headroom.compression.masks import StructureMask

logger = logging.getLogger(__name__)

# Lazy-loaded tree-sitter
_tree_sitter_available: bool | None = None
_tree_sitter_local = threading.local()


def _check_tree_sitter() -> bool:
    """Check if tree-sitter is available and can actually parse.

    Constructs a parser and runs a minimal parse so that ABI mismatches
    between ``tree_sitter`` and ``tree_sitter_language_pack`` surface here
    instead of silently falling back to the text compressor at request time.
    """
    global _tree_sitter_available
    if _tree_sitter_available is None:
        try:
            from tree_sitter import Parser
            from tree_sitter_language_pack import get_language

            parser = Parser()
            parser.language = get_language("python")
            tree = parser.parse(b"x = 1\n")
            if tree.root_node.child_count == 0:
                raise RuntimeError("tree-sitter parse returned empty tree")
            _tree_sitter_available = True
        except ImportError:
            _tree_sitter_available = False
        except Exception:
            logger.warning(
                "tree-sitter imported but failed to parse; "
                "code-aware compression disabled (ABI mismatch?)"
            )
            _tree_sitter_available = False
    return _tree_sitter_available


def _get_parser(language: str) -> Any:
    """Return a **thread-local** tree-sitter parser for ``language``.

    tree-sitter ``Parser`` objects are pyo3 ``unsendable`` — touching
    one from a thread other than its creator panics. Handlers run on
    executor pool threads in the proxy, so parsers must never be shared
    across threads. One parser per (thread, language); same fix as
    ``transforms/code_compressor.py`` (#604).
    """
    if not _check_tree_sitter():
        raise ImportError("tree-sitter-language-pack not installed")

    cache: dict[str, Any] | None = getattr(_tree_sitter_local, "parsers", None)
    if cache is None:
        cache = {}
        _tree_sitter_local.parsers = cache

    if language not in cache:
        from tree_sitter_language_pack import get_parser

        cache[language] = get_parser(language)  # type: ignore[arg-type]

    return cache[language]


# tree-sitter API compatibility. tree-sitter-language-pack switched to a
# Rust binding (>=1.0) where node accessors are METHODS (kind(),
# start_byte(), child(i)) and parse() takes str; the classic pybind API
# uses attributes (.type, .start_byte, .children) and parse(bytes).
# Without this shim the tree-sitter path raises TypeError on modern
# installs and silently falls back to regex.


def _ts_parse(parser: Any, content: str) -> Any:
    try:
        return parser.parse(content.encode("utf-8"))
    except TypeError:
        return parser.parse(content)


def _ts_root(tree: Any) -> Any:
    root = tree.root_node
    return root() if callable(root) else root


def _ts_kind(node: Any) -> str:
    kind = getattr(node, "type", None)
    if isinstance(kind, str):
        return kind
    return str(node.kind())


def _ts_start_byte(node: Any) -> int:
    start = node.start_byte
    return int(start()) if callable(start) else int(start)


def _ts_end_byte(node: Any) -> int:
    end = node.end_byte
    return int(end()) if callable(end) else int(end)


def _ts_children(node: Any) -> list[Any]:
    children = getattr(node, "children", None)
    if children is not None and not callable(children):
        return list(children)
    return [node.child(i) for i in range(node.child_count())]


@dataclass
class CodeSpan:
    """A span of code with its structural role."""

    start: int
    end: int
    role: str  # "import", "signature", "body", "decorator", etc.
    is_structural: bool


# Language-specific AST node types that are structural
_STRUCTURAL_NODE_TYPES: dict[str, set[str]] = {
    "python": {
        "import_statement",
        "import_from_statement",
        "function_definition",  # Just the signature part
        "class_definition",
        "decorated_definition",
        "type_alias_statement",
    },
    "javascript": {
        "import_statement",
        "export_statement",
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",  # Signature only
    },
    "typescript": {
        "import_statement",
        "export_statement",
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
    },
    "go": {
        "import_declaration",
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "interface_type",
    },
    "rust": {
        "use_declaration",
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
    },
    "java": {
        "import_declaration",
        "class_declaration",
        "method_declaration",
        "interface_declaration",
        "annotation",
    },
    "perl": {
        "use_statement",
        "use_version_statement",
        "subroutine_declaration_statement",
        "method_declaration_statement",
        "package_statement",
        "class_statement",
        "role_statement",
    },
}

# Regex patterns for fallback detection
_SIGNATURE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "python": [
        re.compile(r"^\s*(async\s+)?def\s+\w+\s*\([^)]*\)\s*(->\s*[^:]+)?:", re.MULTILINE),
        re.compile(r"^\s*class\s+\w+(\([^)]*\))?:", re.MULTILINE),
        re.compile(r"^\s*@\w+(\([^)]*\))?\s*$", re.MULTILINE),
    ],
    "javascript": [
        re.compile(r"^\s*(async\s+)?function\s+\w+\s*\([^)]*\)", re.MULTILINE),
        re.compile(r"^\s*class\s+\w+(\s+extends\s+\w+)?", re.MULTILINE),
        re.compile(r"^\s*(const|let|var)\s+\w+\s*=\s*(async\s+)?\([^)]*\)\s*=>", re.MULTILINE),
    ],
    "typescript": [
        re.compile(r"^\s*(async\s+)?function\s+\w+\s*(<[^>]+>)?\s*\([^)]*\)", re.MULTILINE),
        re.compile(r"^\s*class\s+\w+(<[^>]+>)?(\s+extends\s+\w+)?", re.MULTILINE),
        re.compile(r"^\s*interface\s+\w+(<[^>]+>)?", re.MULTILINE),
        re.compile(r"^\s*type\s+\w+(<[^>]+>)?\s*=", re.MULTILINE),
    ],
    "go": [
        re.compile(r"^\s*func\s+(\([^)]+\)\s+)?\w+\s*\([^)]*\)", re.MULTILINE),
        re.compile(r"^\s*type\s+\w+\s+(struct|interface)", re.MULTILINE),
    ],
    "rust": [
        re.compile(r"^\s*(pub\s+)?(async\s+)?fn\s+\w+\s*(<[^>]+>)?\s*\([^)]*\)", re.MULTILINE),
        re.compile(r"^\s*(pub\s+)?struct\s+\w+", re.MULTILINE),
        re.compile(r"^\s*(pub\s+)?enum\s+\w+", re.MULTILINE),
        re.compile(r"^\s*(pub\s+)?trait\s+\w+", re.MULTILINE),
        re.compile(r"^\s*impl(<[^>]+>)?\s+\w+", re.MULTILINE),
    ],
    "java": [
        re.compile(
            r"^\s*(public|private|protected)?\s*(static\s+)?\w+\s+\w+\s*\([^)]*\)", re.MULTILINE
        ),
        re.compile(r"^\s*(public\s+)?(class|interface|enum)\s+\w+", re.MULTILINE),
        re.compile(r"^\s*@\w+(\([^)]*\))?\s*$", re.MULTILINE),
    ],
    "perl": [
        re.compile(r"^\s*sub\s+\w+\s*(\([^)]*\))?", re.MULTILINE),
        re.compile(r"^\s*(package|class|role)\s+[\w:]+", re.MULTILINE),
    ],
}

# Body child node types for container definitions (classes, impls,
# traits). A container's span up to its body is structural (the
# signature); the body itself is NOT marked — recursion into the body
# emits signature spans for nested functions/methods, leaving their
# bodies compressible.
_CONTAINER_BODY_TYPES: frozenset[str] = frozenset(
    {
        "block",  # python class body
        "statement_block",  # js/ts
        "compound_statement",  # c/cpp
        "class_body",  # js/ts/java class body
        "interface_body",  # java/ts interface body
        "declaration_list",  # rust impl/trait body
        "enum_body",  # java enum body
    }
)

# Language-detection markers for _detect_language
_LANGUAGE_MARKERS: dict[str, list[str]] = {
    "python": ["def ", "import ", "from ", "class ", "async def"],
    "javascript": ["function ", "const ", "let ", "var ", "=>"],
    "typescript": ["interface ", "type ", ": string", ": number"],
    "go": ["func ", "package ", "import (", "type "],
    "rust": ["fn ", "let mut", "impl ", "pub fn", "use "],
    "java": ["public class", "private ", "protected ", "void "],
    "perl": ["sub ", "my $", "our $", "package ", "use strict"],
}

# Import patterns for fallback
_IMPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^\s*(import\s+\w+|from\s+\w+\s+import)", re.MULTILINE),
    "javascript": re.compile(r"^\s*(import\s+.*from|require\s*\()", re.MULTILINE),
    "typescript": re.compile(r"^\s*(import\s+.*from|require\s*\()", re.MULTILINE),
    "go": re.compile(r'^\s*import\s+(\(|")', re.MULTILINE),
    "rust": re.compile(r"^\s*use\s+\w+", re.MULTILINE),
    "java": re.compile(r"^\s*import\s+[\w.]+;", re.MULTILINE),
    "perl": re.compile(r"^\s*(use|require)\s+[\w:]+", re.MULTILINE),
}


class CodeStructureHandler(BaseStructureHandler):
    """Handler for source code.

    Preserves:
    - Import/use statements
    - Function/method signatures (not bodies)
    - Class/struct/interface definitions
    - Type declarations
    - Decorators/annotations

    Marks as compressible:
    - Function/method bodies
    - Comments (optionally preserved)
    - Whitespace

    Example:
        >>> handler = CodeStructureHandler()
        >>> code = '''
        ... def hello(name: str) -> str:
        ...     message = f"Hello, {name}!"
        ...     return message
        ... '''
        >>> result = handler.get_mask(code, language="python")
        >>> # Signature "def hello(name: str) -> str:" preserved
        >>> # Body content compressed
    """

    def __init__(
        self,
        preserve_comments: bool = False,
        use_tree_sitter: bool = True,
        default_language: str = "python",
    ):
        """Initialize the code handler.

        Args:
            preserve_comments: Whether to preserve comments as structural.
            use_tree_sitter: Whether to use tree-sitter for parsing.
                Falls back to regex if False or unavailable.
            default_language: Default language when detection fails.
        """
        super().__init__(name="code")
        self.preserve_comments = preserve_comments
        self.use_tree_sitter = use_tree_sitter
        self.default_language = default_language

    def can_handle(self, content: str) -> bool:
        """Check if content looks like source code."""
        # Quick heuristic checks
        code_indicators = [
            "def ",
            "class ",
            "function ",
            "import ",
            "const ",
            "let ",
            "var ",
            "func ",
            "fn ",
            "pub ",
            "package ",
            "struct ",
            "interface ",
        ]
        return any(indicator in content for indicator in code_indicators)

    def _extract_mask(
        self,
        content: str,
        tokens: list[str],
        language: str | None = None,
        **kwargs: Any,
    ) -> HandlerResult:
        """Extract structure mask from code.

        Args:
            content: Source code content.
            tokens: Character-level tokens.
            language: Programming language (auto-detected if None).
            **kwargs: Additional options.

        Returns:
            HandlerResult with mask marking structural elements.
        """
        # Detect language if not provided
        if language is None:
            language = self._detect_language(content)

        # Try tree-sitter first
        if self.use_tree_sitter and _check_tree_sitter():
            try:
                return self._extract_with_tree_sitter(content, tokens, language)
            except Exception as e:
                logger.debug("Tree-sitter parsing failed, using fallback: %s", e)

        # Fallback to regex
        return self._extract_with_regex(content, tokens, language)

    def _extract_with_tree_sitter(
        self,
        content: str,
        tokens: list[str],
        language: str,
    ) -> HandlerResult:
        """Extract structure using tree-sitter AST.

        Args:
            content: Source code.
            tokens: Character tokens.
            language: Language name.

        Returns:
            HandlerResult with mask.
        """
        parser = _get_parser(language)
        tree = _ts_parse(parser, content)

        # Collect structural spans
        spans: list[CodeSpan] = []

        def visit_node(node: Any, depth: int = 0) -> None:
            """Visit AST node and collect structural spans."""
            node_type = _ts_kind(node)
            structural_types = _STRUCTURAL_NODE_TYPES.get(language, set())
            children = _ts_children(node)

            # Check if this is a structural node type
            if node_type in structural_types:
                # For functions, only the signature is structural
                if "function" in node_type or "method" in node_type:
                    # Find the body node and exclude it
                    body_node = None
                    for child in children:
                        if _ts_kind(child) in ("block", "statement_block", "compound_statement"):
                            body_node = child
                            break

                    if body_node:
                        # Signature is from start to body start
                        spans.append(
                            CodeSpan(
                                start=_ts_start_byte(node),
                                end=_ts_start_byte(body_node),
                                role="signature",
                                is_structural=True,
                            )
                        )
                        # Body is compressible
                        spans.append(
                            CodeSpan(
                                start=_ts_start_byte(body_node),
                                end=_ts_end_byte(body_node),
                                role="body",
                                is_structural=False,
                            )
                        )
                    else:
                        # No body found, preserve whole thing
                        spans.append(
                            CodeSpan(
                                start=_ts_start_byte(node),
                                end=_ts_end_byte(node),
                                role=node_type,
                                is_structural=True,
                            )
                        )
                elif node_type == "decorated_definition":
                    # Wrapper around decorator(s) + definition. Emit no
                    # span: recursion marks the decorators and gives the
                    # inner function its signature/body split. A whole-
                    # node span here would preserve the function body.
                    pass
                else:
                    # Container definitions (class, impl, trait): the
                    # signature runs to the body start; the body is NOT
                    # marked, so nested function bodies stay compressible
                    # (recursion emits their signature spans). Leaf
                    # declarations (imports, type aliases, structs) have
                    # no such body child and are preserved whole.
                    body_node = None
                    for child in children:
                        if _ts_kind(child) in _CONTAINER_BODY_TYPES:
                            body_node = child
                            break

                    if body_node is not None:
                        spans.append(
                            CodeSpan(
                                start=_ts_start_byte(node),
                                end=_ts_start_byte(body_node),
                                role="signature",
                                is_structural=True,
                            )
                        )
                    else:
                        spans.append(
                            CodeSpan(
                                start=_ts_start_byte(node),
                                end=_ts_end_byte(node),
                                role=node_type,
                                is_structural=True,
                            )
                        )
            elif node_type == "decorator":
                # Decorators are structural (preserved) on their own so
                # the decorated_definition wrapper doesn't need a span.
                spans.append(
                    CodeSpan(
                        start=_ts_start_byte(node),
                        end=_ts_end_byte(node),
                        role="decorator",
                        is_structural=True,
                    )
                )
            elif node_type == "comment" and self.preserve_comments:
                spans.append(
                    CodeSpan(
                        start=_ts_start_byte(node),
                        end=_ts_end_byte(node),
                        role="comment",
                        is_structural=True,
                    )
                )

            # Recurse into children
            for child in children:
                visit_node(child, depth + 1)

        visit_node(_ts_root(tree))

        # tree-sitter spans are BYTE offsets into the UTF-8 encoding;
        # the mask is indexed by CHARACTER. Any non-ASCII character
        # (docstrings, comments, string literals) shifts every later
        # span, so convert before masking. Skipped for pure-ASCII
        # content where the offsets coincide.
        spans = self._byte_spans_to_char_spans(spans, content)

        # Build mask from spans
        mask = self._spans_to_mask(spans, len(content))

        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=0.95,
            metadata={
                "language": language,
                "parser": "tree-sitter",
                "structural_spans": len([s for s in spans if s.is_structural]),
            },
        )

    def _extract_with_regex(
        self,
        content: str,
        tokens: list[str],
        language: str,
    ) -> HandlerResult:
        """Extract structure using regex patterns (fallback).

        Args:
            content: Source code.
            tokens: Character tokens.
            language: Language name.

        Returns:
            HandlerResult with mask.
        """
        spans: list[CodeSpan] = []

        # Match imports
        import_pattern = _IMPORT_PATTERNS.get(language)
        if import_pattern:
            for match in import_pattern.finditer(content):
                # Find end of import line
                end = content.find("\n", match.end())
                if end == -1:
                    end = len(content)
                spans.append(
                    CodeSpan(
                        start=match.start(),
                        end=end,
                        role="import",
                        is_structural=True,
                    )
                )

        # Match signatures
        signature_patterns = _SIGNATURE_PATTERNS.get(language, [])
        for pattern in signature_patterns:
            for match in pattern.finditer(content):
                spans.append(
                    CodeSpan(
                        start=match.start(),
                        end=match.end(),
                        role="signature",
                        is_structural=True,
                    )
                )

        # Build mask from spans
        mask = self._spans_to_mask(spans, len(content))

        return HandlerResult(
            mask=StructureMask(tokens=tokens, mask=mask),
            handler_name=self.name,
            confidence=0.7,  # Lower confidence for regex
            metadata={
                "language": language,
                "parser": "regex",
                "structural_spans": len(spans),
            },
        )

    @staticmethod
    def _byte_spans_to_char_spans(spans: list[CodeSpan], content: str) -> list[CodeSpan]:
        """Convert byte-offset spans to character-offset spans.

        tree-sitter reports node positions as byte offsets in the UTF-8
        encoding. For pure-ASCII content byte == char and the spans are
        returned unchanged. Otherwise a byte->char table is built once
        and every span endpoint is remapped.
        """
        n_bytes = len(content.encode("utf-8"))
        if n_bytes == len(content):
            return spans

        # byte_to_char[b] = index of the character containing byte b;
        # byte_to_char[n_bytes] = len(content) so exclusive ends map.
        byte_to_char = [0] * (n_bytes + 1)
        byte_pos = 0
        for char_idx, ch in enumerate(content):
            ch_width = len(ch.encode("utf-8"))
            for b in range(byte_pos, byte_pos + ch_width):
                byte_to_char[b] = char_idx
            byte_pos += ch_width
        byte_to_char[n_bytes] = len(content)

        return [
            CodeSpan(
                start=byte_to_char[min(span.start, n_bytes)],
                end=byte_to_char[min(span.end, n_bytes)],
                role=span.role,
                is_structural=span.is_structural,
            )
            for span in spans
        ]

    def _spans_to_mask(self, spans: list[CodeSpan], length: int) -> list[bool]:
        """Convert spans to character-level mask.

        Args:
            spans: List of code spans.
            length: Total content length.

        Returns:
            Boolean mask aligned to characters.
        """
        mask = [False] * length

        for span in spans:
            if span.is_structural:
                start = min(span.start, length)
                end = min(span.end, length)
                if start < end:
                    mask[start:end] = [True] * (end - start)

        return mask

    def _detect_language(self, content: str) -> str:
        """Detect programming language from content.

        Args:
            content: Source code content.

        Returns:
            Language name (lowercase).
        """
        scores: dict[str, int] = {}
        for lang, patterns in _LANGUAGE_MARKERS.items():
            scores[lang] = sum(1 for p in patterns if p in content)

        if not scores or max(scores.values()) == 0:
            return self.default_language

        return max(scores, key=lambda k: scores[k])


def is_tree_sitter_available() -> bool:
    """Check if tree-sitter is available."""
    return _check_tree_sitter()
