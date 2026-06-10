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
from enum import Enum
from typing import Any

from headroom.compression.handlers.base import BaseStructureHandler, HandlerResult
from headroom.compression.masks import StructureMask

logger = logging.getLogger(__name__)

# Lazy-loaded tree-sitter
_tree_sitter_available: bool | None = None
# Per-thread parser cache: pyo3 marks Parser as Unsendable, so parser objects created on
# one thread panic if accessed from another. threading.local() ensures every thread
# maintains its own isolated set of parser instances.
_tree_sitter_local = threading.local()


def _check_tree_sitter() -> bool:
    """Check if tree-sitter is available."""
    global _tree_sitter_available
    if _tree_sitter_available is None:
        try:
            import tree_sitter_language_pack  # noqa: F401

            _tree_sitter_available = True
        except ImportError:
            _tree_sitter_available = False
    return _tree_sitter_available


def _get_parser(language: str) -> Any:
    """Get tree-sitter parser for language (per-thread cache)."""
    if not _check_tree_sitter():
        raise ImportError("tree-sitter-language-pack not installed")

    thread_parsers: dict[str, Any] | None = getattr(_tree_sitter_local, "parsers", None)
    if thread_parsers is None:
        thread_parsers = {}
        _tree_sitter_local.parsers = thread_parsers

    if language not in thread_parsers:
        from tree_sitter_language_pack import get_parser

        thread_parsers[language] = get_parser(language)  # type: ignore[arg-type]

    return thread_parsers[language]


class CodeLanguage(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    C = "c"
    CPP = "cpp"


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
}

# Import patterns for fallback
_IMPORT_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^\s*(import\s+\w+|from\s+\w+\s+import)", re.MULTILINE),
    "javascript": re.compile(r"^\s*(import\s+.*from|require\s*\()", re.MULTILINE),
    "typescript": re.compile(r"^\s*(import\s+.*from|require\s*\()", re.MULTILINE),
    "go": re.compile(r'^\s*import\s+(\(|")', re.MULTILINE),
    "rust": re.compile(r"^\s*use\s+\w+", re.MULTILINE),
    "java": re.compile(r"^\s*import\s+[\w.]+;", re.MULTILINE),
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
        tree = parser.parse(content.encode("utf-8"))

        # Collect structural spans
        spans: list[CodeSpan] = []

        def visit_node(node: Any, depth: int = 0) -> None:
            """Visit AST node and collect structural spans."""
            node_type = node.type
            structural_types = _STRUCTURAL_NODE_TYPES.get(language, set())

            # Check if this is a structural node type
            if node_type in structural_types:
                # For functions, only the signature is structural
                if "function" in node_type or "method" in node_type:
                    # Find the body node and exclude it
                    body_node = None
                    for child in node.children:
                        if child.type in ("block", "statement_block", "compound_statement"):
                            body_node = child
                            break

                    if body_node:
                        # Signature is from start to body start
                        spans.append(
                            CodeSpan(
                                start=node.start_byte,
                                end=body_node.start_byte,
                                role="signature",
                                is_structural=True,
                            )
                        )
                        # Body is compressible
                        spans.append(
                            CodeSpan(
                                start=body_node.start_byte,
                                end=body_node.end_byte,
                                role="body",
                                is_structural=False,
                            )
                        )
                    else:
                        # No body found, preserve whole thing
                        spans.append(
                            CodeSpan(
                                start=node.start_byte,
                                end=node.end_byte,
                                role=node_type,
                                is_structural=True,
                            )
                        )
                else:
                    # Non-function structural nodes
                    spans.append(
                        CodeSpan(
                            start=node.start_byte,
                            end=node.end_byte,
                            role=node_type,
                            is_structural=True,
                        )
                    )
            elif node_type == "comment" and self.preserve_comments:
                spans.append(
                    CodeSpan(
                        start=node.start_byte,
                        end=node.end_byte,
                        role="comment",
                        is_structural=True,
                    )
                )

            # Recurse into children
            for child in node.children:
                visit_node(child, depth + 1)

        visit_node(tree.root_node)

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
                for i in range(span.start, min(span.end, length)):
                    mask[i] = True

        return mask

    def _detect_language(self, content: str) -> str:
        """Detect programming language from content.

        Args:
            content: Source code content.

        Returns:
            Language name (lowercase).
        """
        # Check for language-specific markers
        markers = {
            "python": ["def ", "import ", "from ", "class ", "async def"],
            "javascript": ["function ", "const ", "let ", "var ", "=>"],
            "typescript": ["interface ", "type ", ": string", ": number"],
            "go": ["func ", "package ", "import (", "type "],
            "rust": ["fn ", "let mut", "impl ", "pub fn", "use "],
            "java": ["public class", "private ", "protected ", "void "],
        }

        scores: dict[str, int] = {}
        for lang, patterns in markers.items():
            scores[lang] = sum(1 for p in patterns if p in content)

        if not scores or max(scores.values()) == 0:
            return self.default_language

        return max(scores, key=lambda k: scores[k])


def is_tree_sitter_available() -> bool:
    """Check if tree-sitter is available."""
    return _check_tree_sitter()
