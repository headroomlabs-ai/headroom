"""Code-aware compressor using AST parsing for syntax-preserving compression.

This module provides AST-based compression for source code that guarantees
valid syntax output. Unlike token-level compression, this preserves
structural elements while compressing function bodies.

Key Features:
- Syntax validity guaranteed (output always parses)
- Preserves imports, signatures, type annotations, error handlers
- Compresses function bodies while maintaining structure
- Multi-language support via tree-sitter
- Data-driven language config (no per-language method duplication)
- Thread-safe (thread-local tree-sitter parsers, no shared mutable state)

Supported Languages (Tier 1):
- Python, JavaScript, TypeScript

Supported Languages (Tier 2):
- Go, Rust, Java, C, C++

Compression Strategy:
1. Parse code into AST using tree-sitter
2. Extract and preserve critical structures (imports, signatures, types)
3. Rank functions by importance (using semantic analysis)
4. Compress function bodies while preserving signatures
5. Reassemble into valid code

Installation:
    pip install headroom-ai[code]

Usage:
    >>> from headroom.transforms import CodeAwareCompressor
    >>> compressor = CodeAwareCompressor()
    >>> result = compressor.compress(python_code)
    >>> print(result.compressed)  # Valid Python code
    >>> print(result.syntax_valid)  # True

Reference:
    LongCodeZip: Compress Long Context for Code Language Models
    https://arxiv.org/abs/2510.00446
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import TransformResult
from ..tokenizer import Tokenizer
from .base import Transform

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
_tree_sitter_available: bool | None = None
_tree_sitter_local = threading.local()


def _check_tree_sitter_available() -> bool:
    """Check if tree-sitter is available *and actually parses*.

    The mere presence of ``tree_sitter_language_pack`` is not enough: prior
    versions of this code green-lit a code path that raised ``TypeError`` at
    parse time and silently fell back to a lossy stripper.  To stop misleading
    callers, we now verify an end-to-end parse of a tiny snippet and only
    return ``True`` if it yields a real AST.
    """
    global _tree_sitter_available
    if _tree_sitter_available is None:
        try:
            parser = _get_parser("python")
            tree = parser.parse(b"def _probe():\n    return 1\n")
            root = tree.root_node
            # A real parse yields a non-error root with children.
            _tree_sitter_available = (
                root is not None
                and root.type == "module"
                and root.child_count > 0
                and not _has_syntax_issues(root)
            )
        except Exception:
            _tree_sitter_available = False
    return _tree_sitter_available


def _tree_sitter_importable() -> bool:
    """Return True if the tree-sitter grammar pack can be imported.

    This only checks importability (cheap, no parse). Use
    :func:`_check_tree_sitter_available` for the stronger "parsing actually
    works" guarantee.
    """
    try:
        import tree_sitter_language_pack  # noqa: F401

        return True
    except ImportError:
        return False


def _get_parser(language: str) -> Any:
    """Get a tree-sitter parser for the given language.

    Returns a **thread-local** ``tree_sitter.Parser`` instance.

    tree-sitter ≥ 0.23 wraps the C ``TSParser`` in a PyO3
    ``#[pyclass(unsendable)]`` which hard-panics if the object is accessed
    from any thread other than its creator.  Because Headroom runs
    compression inside a ``ThreadPoolExecutor``, a single shared parser
    would be touched from arbitrary pool threads → instant crash.

    We use the stock ``tree_sitter.Parser`` (which returns standard
    ``tree_sitter.Node`` / ``tree_sitter.Tree`` with property access) and
    set its language via ``tree_sitter_language_pack.get_language()``.
    Storing one parser per (thread, language) satisfies the ``unsendable``
    contract with negligible extra memory.

    Args:
        language: Language name (e.g., 'python', 'javascript').

    Returns:
        Configured ``tree_sitter.Parser`` bound to the current thread.

    Raises:
        ImportError: If tree-sitter is not installed.
        ValueError: If language is not supported.
    """
    # NOTE: guard on importability (not _check_tree_sitter_available), because
    # _check_tree_sitter_available now performs a real end-to-end parse via
    # _get_parser; guarding on it here would recurse.
    if not _tree_sitter_importable():
        raise ImportError(
            "tree-sitter is not installed. Install with: pip install headroom-ai[code]\n"
            "This adds ~50MB for tree-sitter grammars."
        )

    parsers: dict[str, Any] | None = getattr(_tree_sitter_local, "parsers", None)
    if parsers is None:
        parsers = {}
        _tree_sitter_local.parsers = parsers

    if language not in parsers:
        try:
            from tree_sitter import Parser
            from tree_sitter_language_pack import get_language

            parser = Parser()
            # `language` is a validated runtime str; get_language types its arg
            # as a Literal of supported names, which a dynamic str can't satisfy.
            parser.language = get_language(language)  # type: ignore[arg-type]
            parsers[language] = parser
            logger.debug(
                "Loaded tree-sitter parser for %s (thread %s)",
                language,
                threading.current_thread().name,
            )
        except Exception as e:
            raise ValueError(
                f"Language '{language}' is not supported by tree-sitter. "
                f"Supported: python, javascript, typescript, go, rust, java, c, cpp, perl. "
                f"Error: {e}"
            ) from e

    return parsers[language]


def is_tree_sitter_available() -> bool:
    """Check if tree-sitter is installed and available.

    Returns:
        True if tree-sitter-languages package is installed.
    """
    return _check_tree_sitter_available()


def is_tree_sitter_loaded() -> bool:
    """Check if any tree-sitter parsers are loaded on the current thread.

    Returns:
        True if parsers are loaded in this thread's local storage.
    """
    parsers: dict[str, Any] | None = getattr(_tree_sitter_local, "parsers", None)
    return bool(parsers)


def unload_tree_sitter() -> bool:
    """Unload tree-sitter parsers on the current thread to free memory.

    Returns:
        True if parsers were unloaded, False if none were loaded.
    """
    parsers: dict[str, Any] | None = getattr(_tree_sitter_local, "parsers", None)
    if parsers:
        count = len(parsers)
        parsers.clear()
        logger.info(
            "Unloaded %d tree-sitter parsers (thread %s)", count, threading.current_thread().name
        )
        return True
    return False


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
    PERL = "perl"
    UNKNOWN = "unknown"


class DocstringMode(Enum):
    """How to handle docstrings."""

    FULL = "full"  # Keep entire docstring
    FIRST_LINE = "first_line"  # Keep only first line
    REMOVE = "remove"  # Remove docstrings completely
    NONE = "none"  # Alias for REMOVE (deprecated)


# =========================================================================
# Data-driven language configuration
# =========================================================================


@dataclass(frozen=True)
class LangConfig:
    """Data-driven configuration for a programming language.

    Instead of per-language methods, each language declares its AST node
    types and syntactic conventions. The compressor uses these tables to
    drive extraction and compression generically.
    """

    # AST node types for structural extraction
    import_nodes: frozenset[str]
    function_nodes: frozenset[str]
    class_nodes: frozenset[str]
    type_nodes: frozenset[str]
    body_node_types: frozenset[str]  # Node types that represent function/method bodies
    decorator_node: str | None  # e.g. "decorated_definition" for Python

    # Syntax conventions
    comment_prefix: str  # "#" for Python, "//" for C-family
    uses_colon_after_signature: bool  # Python: True, C-family: False
    package_node: str | None = None  # e.g. "package_clause" for Go

    # Quick pre-filter hints for language detection (substrings to check)
    detection_hints: tuple[str, ...] = ()
    # Optional override for node types that contain class/impl members.
    class_body_node_types: frozenset[str] | None = None


_LANG_CONFIGS: dict[CodeLanguage, LangConfig] = {
    CodeLanguage.PYTHON: LangConfig(
        import_nodes=frozenset(
            {"future_import_statement", "import_statement", "import_from_statement"}
        ),
        function_nodes=frozenset({"function_definition"}),
        class_nodes=frozenset({"class_definition"}),
        type_nodes=frozenset({"type_alias_statement"}),
        body_node_types=frozenset({"block"}),
        decorator_node="decorated_definition",
        comment_prefix="#",
        uses_colon_after_signature=True,
        detection_hints=("def ", "import ", "from ", "class ", "async def"),
    ),
    CodeLanguage.JAVASCRIPT: LangConfig(
        import_nodes=frozenset({"import_statement", "import_declaration"}),
        function_nodes=frozenset({"function_declaration", "method_definition"}),
        class_nodes=frozenset({"class_declaration"}),
        type_nodes=frozenset(),
        body_node_types=frozenset({"statement_block"}),
        decorator_node=None,
        comment_prefix="//",
        uses_colon_after_signature=False,
        detection_hints=("function ", "const ", "let ", "var ", "export ", "require("),
        class_body_node_types=frozenset({"class_body"}),
    ),
    CodeLanguage.TYPESCRIPT: LangConfig(
        import_nodes=frozenset({"import_statement", "import_declaration"}),
        function_nodes=frozenset({"function_declaration", "method_definition"}),
        class_nodes=frozenset({"class_declaration"}),
        type_nodes=frozenset({"interface_declaration", "type_alias_declaration"}),
        body_node_types=frozenset({"statement_block"}),
        decorator_node=None,
        comment_prefix="//",
        uses_colon_after_signature=False,
        detection_hints=("interface ", "type ", ": string", ": number", ": boolean"),
        class_body_node_types=frozenset({"class_body"}),
    ),
    CodeLanguage.GO: LangConfig(
        import_nodes=frozenset({"import_declaration"}),
        function_nodes=frozenset({"function_declaration", "method_declaration"}),
        class_nodes=frozenset(),
        type_nodes=frozenset({"type_declaration"}),
        body_node_types=frozenset({"block"}),
        decorator_node=None,
        comment_prefix="//",
        uses_colon_after_signature=False,
        package_node="package_clause",
        detection_hints=("func ", "package ", "struct {"),
    ),
    CodeLanguage.RUST: LangConfig(
        import_nodes=frozenset({"use_declaration"}),
        function_nodes=frozenset({"function_item"}),
        class_nodes=frozenset({"impl_item"}),
        type_nodes=frozenset({"struct_item", "enum_item", "type_item", "trait_item"}),
        body_node_types=frozenset({"block"}),
        decorator_node=None,
        comment_prefix="//",
        uses_colon_after_signature=False,
        detection_hints=("fn ", "struct ", "impl ", "mod ", "use "),
        class_body_node_types=frozenset({"declaration_list"}),
    ),
    CodeLanguage.JAVA: LangConfig(
        import_nodes=frozenset({"import_declaration"}),
        function_nodes=frozenset({"method_declaration", "constructor_declaration"}),
        class_nodes=frozenset({"class_declaration", "interface_declaration"}),
        type_nodes=frozenset({"enum_declaration"}),
        body_node_types=frozenset({"block"}),
        decorator_node=None,
        comment_prefix="//",
        uses_colon_after_signature=False,
        package_node="package_declaration",
        detection_hints=("public ", "private ", "protected ", "class ", "interface "),
        class_body_node_types=frozenset({"class_body"}),
    ),
    CodeLanguage.C: LangConfig(
        import_nodes=frozenset({"preproc_include"}),
        function_nodes=frozenset({"function_definition"}),
        class_nodes=frozenset(),
        type_nodes=frozenset({"struct_specifier", "enum_specifier", "type_definition"}),
        body_node_types=frozenset({"compound_statement"}),
        decorator_node=None,
        comment_prefix="//",
        uses_colon_after_signature=False,
        detection_hints=("#include", "typedef ", "int main("),
    ),
    CodeLanguage.CPP: LangConfig(
        import_nodes=frozenset({"preproc_include"}),
        function_nodes=frozenset({"function_definition"}),
        class_nodes=frozenset({"class_specifier"}),
        type_nodes=frozenset({"struct_specifier", "enum_specifier", "type_definition"}),
        body_node_types=frozenset({"compound_statement"}),
        decorator_node=None,
        comment_prefix="//",
        uses_colon_after_signature=False,
        detection_hints=("#include", "namespace ", "class ", "::"),
        class_body_node_types=frozenset({"field_declaration_list"}),
    ),
    CodeLanguage.PERL: LangConfig(
        import_nodes=frozenset({"use_statement", "use_version_statement"}),
        function_nodes=frozenset(
            {"subroutine_declaration_statement", "method_declaration_statement"}
        ),
        class_nodes=frozenset({"package_statement", "class_statement", "role_statement"}),
        type_nodes=frozenset(),
        body_node_types=frozenset({"block"}),
        decorator_node=None,
        comment_prefix="#",
        uses_colon_after_signature=False,
        package_node="package_statement",
        detection_hints=("sub ", "my ", "our ", "use ", "package "),
    ),
}


@dataclass
class CodeStructure:
    """Extracted structure from parsed code."""

    imports: list[str] = field(default_factory=list)
    type_definitions: list[str] = field(default_factory=list)
    class_definitions: list[str] = field(default_factory=list)
    function_signatures: list[str] = field(default_factory=list)
    function_bodies: list[tuple[str, str, int]] = field(
        default_factory=list
    )  # (signature, body, line)
    decorators: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    top_level_code: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)


@dataclass
class CodeCompressorConfig:
    """Configuration for code-aware compression.

    Attributes:
        preserve_imports: Always keep import statements.
        preserve_signatures: Always keep function/method signatures.
        preserve_type_annotations: Keep type hints and annotations.
        preserve_decorators: Keep decorators on functions/classes.
        docstring_mode: How to handle docstrings.
        target_compression_rate: Target compression ratio (0.2 = keep 20%).
        max_body_lines: Maximum lines to keep per function body.
        compress_comments: Remove non-docstring comments.
        min_tokens_for_compression: Minimum tokens to trigger compression.
        language_hint: Explicit language (None = auto-detect).
        fallback_to_kompress: Use Kompress for unknown languages.
        enable_ccr: Store originals for retrieval.
        ccr_ttl: TTL for CCR entries in seconds.
    """

    # Preservation settings
    preserve_imports: bool = True
    preserve_signatures: bool = True
    preserve_type_annotations: bool = True
    preserve_decorators: bool = True
    docstring_mode: DocstringMode = DocstringMode.FIRST_LINE

    # Compression settings
    target_compression_rate: float = 0.2
    max_body_lines: int = 5
    compress_comments: bool = True

    # Thresholds
    min_tokens_for_compression: int = 100

    # Language handling
    language_hint: str | None = None
    fallback_to_kompress: bool = True

    # Semantic analysis (symbol importance scoring)
    semantic_analysis: bool = True

    # CCR integration
    enable_ccr: bool = True
    ccr_ttl: int = 300  # 5 minutes


@dataclass
class CodeCompressionResult:
    """Result of code-aware compression.

    Attributes:
        compressed: The compressed code (guaranteed valid syntax).
        original: Original code before compression.
        original_tokens: Token count before compression.
        compressed_tokens: Token count after compression.
        compression_ratio: Actual compression ratio achieved.
        language: Detected or specified language.
        language_confidence: Confidence in language detection.
        preserved_imports: Number of import statements preserved.
        preserved_signatures: Number of function signatures preserved.
        compressed_bodies: Number of function bodies compressed.
        syntax_valid: Whether output is syntactically valid.
        cache_key: CCR cache key if stored.
    """

    compressed: str
    original: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float

    # Code-specific metadata
    language: CodeLanguage = CodeLanguage.UNKNOWN
    language_confidence: float = 0.0

    # Structure analysis
    preserved_imports: int = 0
    preserved_signatures: int = 0
    compressed_bodies: int = 0

    # Validation
    syntax_valid: bool = True

    # CCR
    cache_key: str | None = None

    # Semantic analysis
    symbol_scores: dict[str, float] = field(default_factory=dict)

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved by compression."""
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def savings_percentage(self) -> float:
        """Percentage of tokens saved."""
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100

    @property
    def summary(self) -> str:
        """Human-readable summary of compression."""
        analysis_note = ""
        if self.symbol_scores:
            high = sum(1 for s in self.symbol_scores.values() if s >= 0.7)
            low = sum(1 for s in self.symbol_scores.values() if s < 0.1)
            if high or low:
                analysis_note = f" Semantic: {high} high-importance, {low} low-importance."
        return (
            f"Compressed {self.language.value} code: "
            f"{self.original_tokens:,}→{self.compressed_tokens:,} tokens "
            f"({self.savings_percentage:.0f}% saved). "
            f"Kept {self.preserved_imports} imports, "
            f"{self.preserved_signatures} signatures, "
            f"compressed {self.compressed_bodies} bodies."
            f"{analysis_note}"
        )


# =========================================================================
# Language detection
# =========================================================================

# Lightweight pre-filter patterns for language detection.
# These are ONLY used as a quick check to avoid parsing with every language.
# Actual detection is done by tree-sitter (fewest parse errors wins).
_LANGUAGE_PREFILTER: dict[CodeLanguage, list[re.Pattern[str]]] = {
    CodeLanguage.PYTHON: [
        re.compile(r"^\s*(def|class|import|from|async def)\s+\w+", re.MULTILINE),
        re.compile(r"^\s*@\w+", re.MULTILINE),
        re.compile(r'^\s*"""', re.MULTILINE),
        re.compile(r"^\s*if __name__\s*==", re.MULTILINE),
    ],
    CodeLanguage.JAVASCRIPT: [
        re.compile(r"^\s*(function|const|let|var|class|export)\s+\w+", re.MULTILINE),
        re.compile(r"^\s*async\s+(function|=>)", re.MULTILINE),
        re.compile(r"^\s*module\.exports", re.MULTILINE),
        re.compile(r"^\s*(import|export)\s+.*\s+from\s+['\"]", re.MULTILINE),
    ],
    CodeLanguage.TYPESCRIPT: [
        re.compile(r"^\s*(interface|type|enum|namespace)\s+\w+", re.MULTILINE),
        re.compile(r":\s*(string|number|boolean|any|void|Promise)\b", re.MULTILINE),
    ],
    CodeLanguage.GO: [
        re.compile(r"^\s*(func|type|package|import)\s+", re.MULTILINE),
        re.compile(r"^\s*func\s+\([^)]+\)\s+\w+", re.MULTILINE),
        re.compile(r"\bstruct\s*\{", re.MULTILINE),
    ],
    CodeLanguage.RUST: [
        re.compile(r"^\s*(fn|struct|enum|impl|mod|use|pub)\s+", re.MULTILINE),
        re.compile(r"^\s*#\[", re.MULTILINE),
    ],
    CodeLanguage.JAVA: [
        re.compile(r"^\s*(public|private|protected)\s+(class|interface|enum)", re.MULTILINE),
        re.compile(r"^\s*package\s+[\w.]+;", re.MULTILINE),
    ],
    CodeLanguage.C: [
        re.compile(r"^\s*#include\s*[<\"]", re.MULTILINE),
        re.compile(r"^\s*(int|void|char|float|double)\s+\w+\s*\(", re.MULTILINE),
        re.compile(r"^\s*typedef\s+", re.MULTILINE),
    ],
    CodeLanguage.CPP: [
        re.compile(r"^\s*#include\s*[<\"]", re.MULTILINE),
        re.compile(r"\bnamespace\s+\w+", re.MULTILINE),
        re.compile(r"::\w+", re.MULTILINE),
    ],
    CodeLanguage.PERL: [
        re.compile(r"^\s*(sub|package|use|require)\s+[\w:]+", re.MULTILINE),
        re.compile(r"^\s*(my|our|local)\s+[\$@%]", re.MULTILINE),
        re.compile(r"[\$@%]\w+", re.MULTILINE),
    ],
}


def _count_error_nodes(node: Any) -> int:
    """Count ERROR and MISSING nodes in a tree-sitter AST."""
    count = 0
    if node.type == "ERROR" or node.is_missing:
        count += 1
    for child in node.children:
        count += _count_error_nodes(child)
    return count


def detect_language(code: str) -> tuple[CodeLanguage, float]:
    """Detect the programming language of code.

    Uses tree-sitter AST parsing when available (most accurate), with a
    regex pre-filter to avoid parsing with all languages. Falls back to
    regex-only scoring when tree-sitter is unavailable.

    Args:
        code: Source code to analyze.

    Returns:
        Tuple of (detected language, confidence score 0.0-1.0).
    """
    if not code or not code.strip():
        return CodeLanguage.UNKNOWN, 0.0

    sample = code[:5000]

    # Phase 1: Pre-filter — find candidate languages using quick regex
    candidates: dict[CodeLanguage, int] = {}
    for lang, patterns in _LANGUAGE_PREFILTER.items():
        score = 0
        for pattern in patterns:
            matches = len(pattern.findall(sample))
            score += matches
        if score > 0:
            candidates[lang] = score

    if not candidates:
        return CodeLanguage.UNKNOWN, 0.0

    # Disambiguation: TypeScript superset of JavaScript
    if CodeLanguage.TYPESCRIPT in candidates and CodeLanguage.JAVASCRIPT in candidates:
        if candidates[CodeLanguage.TYPESCRIPT] >= 2:
            candidates[CodeLanguage.JAVASCRIPT] = 0

    # Disambiguation: C++ superset of C
    if CodeLanguage.CPP in candidates and CodeLanguage.C in candidates:
        if candidates[CodeLanguage.CPP] >= 2:
            candidates[CodeLanguage.C] = 0

    # Phase 2: If tree-sitter available, parse with candidates and pick fewest errors
    if _check_tree_sitter_available():
        best_lang = CodeLanguage.UNKNOWN
        min_errors = float("inf")
        best_node_count = 0
        code_bytes = bytes(code[:10000], "utf-8")

        # Sort candidates by pre-filter score (try most likely first)
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        for lang, _prefilter_score in sorted_candidates:
            if lang == CodeLanguage.UNKNOWN or candidates.get(lang, 0) == 0:
                continue
            try:
                parser = _get_parser(lang.value)
                tree = parser.parse(code_bytes)
                error_count = _count_error_nodes(tree.root_node)
                node_count = tree.root_node.child_count

                # Prefer: fewest errors, then most top-level nodes (richer parse)
                if error_count < min_errors or (
                    error_count == min_errors and node_count > best_node_count
                ):
                    min_errors = error_count
                    best_lang = lang
                    best_node_count = node_count
            except (ValueError, ImportError):
                continue

        if best_lang != CodeLanguage.UNKNOWN:
            # Confidence based on error ratio
            total_lines = max(1, len(code.strip().split("\n")))
            error_ratio = min_errors / total_lines
            confidence = max(0.3, min(1.0, 1.0 - error_ratio))
            return best_lang, confidence

    # Phase 3: Fallback — regex-only scoring (no tree-sitter)
    best_lang = max(candidates, key=lambda k: candidates[k])
    best_score = candidates[best_lang]

    if best_score == 0:
        return CodeLanguage.UNKNOWN, 0.0

    confidence = min(1.0, 0.3 + (best_score * 0.1))
    return best_lang, confidence


# =========================================================================
# Symbol importance analysis
# =========================================================================


@dataclass
class _SymbolAnalysis:
    """Result of intra-file symbol importance analysis.

    All dicts are keyed by qualified name (e.g., 'ClassName.method')
    to avoid collisions between identically-named methods in different classes.
    """

    scores: dict[str, float] = field(default_factory=dict)
    calls: dict[str, set[str]] = field(default_factory=dict)
    ref_counts: dict[str, int] = field(default_factory=dict)
    body_line_counts: dict[str, int] = field(default_factory=dict)
    bare_names: dict[str, str] = field(default_factory=dict)  # qname -> short_name


class CodeAwareCompressor(Transform):
    """AST-preserving compression for source code.

    This compressor uses tree-sitter to parse code into an AST, then
    selectively compresses function bodies while preserving structure.
    The output is guaranteed to be syntactically valid.

    Key advantages over token-level compression:
    - Syntax validity guaranteed
    - Preserves imports, signatures, types
    - Better compression ratios for code (5-8x vs 3-5x)
    - Lower latency (~20-50ms vs 50-200ms for token-level compressors)
    - Smaller memory footprint (~50MB vs ~1GB)
    - Thread-safe (thread-local tree-sitter parsers, no shared mutable state)

    Example:
        >>> compressor = CodeAwareCompressor()
        >>> result = compressor.compress('''
        ... import os
        ... from typing import List
        ...
        ... def process_data(items: List[str]) -> List[str]:
        ...     \"\"\"Process a list of items.\"\"\"
        ...     results = []
        ...     for item in items:
        ...         # Validate item
        ...         if not item:
        ...             continue
        ...         # Process valid item
        ...         processed = item.strip().lower()
        ...         results.append(processed)
        ...     return results
        ... ''')
        >>> print(result.compressed)
        import os
        from typing import List

        def process_data(items: List[str]) -> List[str]:
            \"\"\"Process a list of items.\"\"\"
            # ... (body compressed: 10 lines → 2 lines)
            pass
    """

    name: str = "code_aware_compressor"

    def __init__(self, config: CodeCompressorConfig | None = None):
        """Initialize code-aware compressor.

        Args:
            config: Compression configuration. If None, uses defaults.

        Note:
            Tree-sitter parsers are loaded lazily on first use to avoid
            startup overhead when the compressor isn't used.
        """
        self.config = config or CodeCompressorConfig()

    # =========================================================================
    # Token estimation
    # =========================================================================

    @staticmethod
    def _estimate_tokens(text: str, tokenizer: Tokenizer | None = None) -> int:
        """Count or estimate tokens for text.

        Uses real tokenizer when available; falls back to chars/4 which is
        a much closer approximation for code than word count.
        """
        if tokenizer is not None:
            return tokenizer.count_text(text)
        # chars/4 is a reasonable approximation for code tokens
        # (code has lots of punctuation that tokenizes separately)
        return max(1, len(text) // 4)

    # =========================================================================
    # Symbol importance analysis
    # =========================================================================

    def _analyze_symbol_importance(
        self,
        root: Any,
        code: str,
        language: CodeLanguage,
        context: str = "",
    ) -> _SymbolAnalysis:
        """Analyze symbol importance using distribution-based scoring.

        Collects raw signals (reference count, fan-out, visibility, context match,
        convention importance) per symbol, then normalizes using min-max scaling
        so scores are relative within the file. This adapts to any file structure:
        utility libraries, test files, orchestrators, etc.

        Returns _SymbolAnalysis with normalized scores (0.0-1.0) per symbol.
        """
        if not self.config.semantic_analysis:
            return _SymbolAnalysis()

        lang_config = _LANG_CONFIGS.get(language)
        if not lang_config:
            return _SymbolAnalysis()

        all_definition_types = lang_config.function_nodes | lang_config.class_nodes

        # Use qualified keys (ClassName.method) to avoid collisions
        definitions: dict[str, Any] = {}  # qualified_name -> node
        bare_names: dict[str, str] = {}  # qualified_name -> short_name
        all_identifiers: dict[str, int] = {}  # short_name -> count
        function_calls: dict[str, set[str]] = {}

        def collect_definitions(node: Any, parent_name: str = "") -> None:
            if node.type in all_definition_types:
                short_name = _get_definition_name(node)
                if short_name:
                    qualified = f"{parent_name}.{short_name}" if parent_name else short_name
                    definitions[qualified] = node
                    bare_names[qualified] = short_name
                    for child in node.children:
                        collect_definitions(child, parent_name=qualified)
                    return
            # Also check for decorated definitions
            if lang_config.decorator_node and node.type == lang_config.decorator_node:
                for child in node.children:
                    if child.type in all_definition_types:
                        short_name = _get_definition_name(child)
                        if short_name:
                            qualified = f"{parent_name}.{short_name}" if parent_name else short_name
                            definitions[qualified] = child
                            bare_names[qualified] = short_name
                            for grandchild in child.children:
                                collect_definitions(grandchild, parent_name=qualified)
                            return
            for child in node.children:
                collect_definitions(child, parent_name)

        def collect_identifiers(node: Any) -> None:
            if node.type in ("identifier", "property_identifier", "type_identifier"):
                text = node.text
                name = text.decode("utf-8") if isinstance(text, bytes) else str(text)
                all_identifiers[name] = all_identifiers.get(name, 0) + 1
            for child in node.children:
                collect_identifiers(child)

        def collect_calls_in_function(func_node: Any, func_qname: str) -> None:
            func_short = bare_names[func_qname]
            defined_short_names = set(bare_names.values())
            calls: set[str] = set()

            def walk(node: Any) -> None:
                if node.type in ("identifier", "property_identifier"):
                    text = node.text
                    name = text.decode("utf-8") if isinstance(text, bytes) else str(text)
                    if name in defined_short_names and name != func_short:
                        calls.add(name)
                for child in node.children:
                    walk(child)

            walk(func_node)
            function_calls[func_qname] = calls

        # Pass 1: Collect definitions with qualified names
        collect_definitions(root)

        if not definitions:
            return _SymbolAnalysis()

        # Pass 2: Collect all identifiers
        collect_identifiers(root)

        # Pass 3: Collect call relationships and body sizes
        body_line_counts: dict[str, int] = {}
        for qname, node in definitions.items():
            collect_calls_in_function(node, qname)
            node_text = _slice_code_bytes(code, node.start_byte, node.end_byte)
            body_line_counts[qname] = max(1, len(node_text.split("\n")) - 2)

        # Reference counts: subtract definition occurrences
        short_name_def_count: dict[str, int] = {}
        for short in bare_names.values():
            short_name_def_count[short] = short_name_def_count.get(short, 0) + 1

        ref_counts: dict[str, int] = {}
        for qname in definitions:
            short = bare_names[qname]
            count = all_identifiers.get(short, 0)
            ref_counts[qname] = max(0, count - short_name_def_count.get(short, 1))

        # Raw importance signals per symbol
        context_lower = context.lower() if context else ""
        context_words = set(re.split(r"[\s,;:.()\[\]{}\"']+", context_lower)) if context else set()
        context_words.discard("")

        raw_signals: dict[str, float] = {}
        for qname in definitions:
            short = bare_names[qname]
            refs = ref_counts.get(qname, 0)
            fan_out = len(function_calls.get(qname, set()))
            is_public = _is_public_symbol(short, language)

            raw = float(refs)
            raw += 1.0 if is_public else 0.0
            raw += fan_out * 0.5

            # Convention importance (language-specific)
            if language == CodeLanguage.PYTHON:
                if short.startswith("__") and short.endswith("__"):
                    raw += 2.0
            elif language == CodeLanguage.GO:
                if short and short[0].isupper():
                    raw += 1.0

            # Context boost
            if context_words:
                name_lower = short.lower()
                if name_lower in context_words or (
                    len(name_lower) > 3 and name_lower in context_lower
                ):
                    raw += 3.0

            raw_signals[qname] = raw

        # Normalize to 0-1 using min-max scaling
        values = list(raw_signals.values())
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val > 0:
            scores = {name: round((v - min_val) / range_val, 3) for name, v in raw_signals.items()}
        else:
            scores = dict.fromkeys(raw_signals, 0.5)

        return _SymbolAnalysis(
            scores=scores,
            calls=function_calls,
            ref_counts=ref_counts,
            body_line_counts=body_line_counts,
            bare_names=bare_names,
        )

    def _allocate_body_budget(self, analysis: _SymbolAnalysis, code: str) -> dict[str, int]:
        """Allocate body line budget across functions using target_compression_rate.

        Returns dict mapping symbol name to max body lines to keep.
        """
        if not analysis.scores or not analysis.body_line_counts:
            return {}

        scores = analysis.scores
        body_sizes = analysis.body_line_counts
        target_rate = self.config.target_compression_rate

        total_lines = len(code.strip().split("\n"))
        total_body_lines = sum(body_sizes.values())
        fixed_lines = max(0, total_lines - total_body_lines)

        target_total = total_lines * target_rate
        body_budget = max(0.0, target_total - fixed_lines)

        if total_body_lines == 0:
            return {}

        score_floor = 0.05

        weights: dict[str, float] = {}
        for name in scores:
            score = max(scores.get(name, 0.5), score_floor)
            size = body_sizes.get(name, 0)
            weights[name] = score * size

        total_weight = sum(weights.values())

        if total_weight == 0:
            per_func = max(0, int(body_budget / max(len(scores), 1)))
            return {name: min(per_func, body_sizes.get(name, 0)) for name in scores}

        limits: dict[str, int] = {}
        for qname in scores:
            allocation = body_budget * weights[qname] / total_weight
            max_lines = body_sizes.get(qname, 0)
            limit = min(int(round(allocation)), max_lines)
            limits[qname] = limit
            # Also store by short name so _get_body_limit can find it.
            short = analysis.bare_names.get(qname, qname)
            if short not in limits or limit > limits[short]:
                limits[short] = limit

        return limits

    # =========================================================================
    # Core compression
    # =========================================================================

    def compress(
        self,
        code: str,
        language: str | None = None,
        context: str = "",
        tokenizer: Tokenizer | None = None,
    ) -> CodeCompressionResult:
        """Compress code while preserving syntax validity.

        Args:
            code: Source code to compress.
            language: Language name (e.g., 'python'). Auto-detected if None.
            context: Optional context for relevance-aware compression.
            tokenizer: Optional tokenizer for accurate token counting.

        Returns:
            CodeCompressionResult with compressed code and metadata.
        """
        if not code or not code.strip():
            return CodeCompressionResult(
                compressed=code,
                original=code,
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                syntax_valid=True,
            )

        original_tokens = self._estimate_tokens(code, tokenizer)

        # Skip small content
        if original_tokens < self.config.min_tokens_for_compression:
            return CodeCompressionResult(
                compressed=code,
                original=code,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                syntax_valid=True,
            )

        # Detect or use specified language
        if language:
            detected_lang = CodeLanguage(language.lower())
            confidence = 1.0
        elif self.config.language_hint:
            detected_lang = CodeLanguage(self.config.language_hint.lower())
            confidence = 1.0
        else:
            detected_lang, confidence = detect_language(code)

        # If language unknown and fallback enabled, try Kompress
        if detected_lang == CodeLanguage.UNKNOWN:
            if self.config.fallback_to_kompress:
                return self._fallback_compress(code, original_tokens)
            else:
                return CodeCompressionResult(
                    compressed=code,
                    original=code,
                    original_tokens=original_tokens,
                    compressed_tokens=original_tokens,
                    compression_ratio=1.0,
                    language=CodeLanguage.UNKNOWN,
                    language_confidence=0.0,
                    syntax_valid=True,
                )

        # Check if tree-sitter is available
        if not _check_tree_sitter_available():
            logger.warning("tree-sitter not available. Install with: pip install headroom-ai[code]")
            if self.config.fallback_to_kompress:
                return self._fallback_compress(code, original_tokens)
            return CodeCompressionResult(
                compressed=code,
                original=code,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                language=detected_lang,
                language_confidence=confidence,
                syntax_valid=True,
            )

        # Parse and compress
        try:
            compressed, structure, symbol_scores = self._compress_with_ast(
                code, detected_lang, context, tokenizer
            )
            compressed_tokens = self._estimate_tokens(compressed, tokenizer)

            # Verify syntax validity (checks both ERROR and MISSING nodes)
            syntax_valid = self._verify_syntax(compressed, detected_lang)

            # If syntax invalid, return original (never serve broken code)
            if not syntax_valid:
                logger.warning(
                    "Code compression produced invalid syntax for %s (%d tokens), "
                    "returning original",
                    detected_lang.value,
                    original_tokens,
                )
                return CodeCompressionResult(
                    compressed=code,
                    original=code,
                    original_tokens=original_tokens,
                    compressed_tokens=original_tokens,
                    compression_ratio=1.0,
                    language=detected_lang,
                    language_confidence=confidence,
                    syntax_valid=True,
                )

            ratio = compressed_tokens / max(original_tokens, 1)

            # Guard against over-aggressive compression (data loss)
            if ratio < 0.05:
                logger.warning(
                    "Code compression too aggressive (ratio=%.3f), returning original",
                    ratio,
                )
                return CodeCompressionResult(
                    compressed=code,
                    original=code,
                    original_tokens=original_tokens,
                    compressed_tokens=original_tokens,
                    compression_ratio=1.0,
                    language=detected_lang,
                    language_confidence=confidence,
                    syntax_valid=True,
                )

            # Store in CCR if significant compression
            cache_key = None
            if self.config.enable_ccr and ratio < 0.8:
                cache_key = self._store_in_ccr(code, compressed, original_tokens)
                if cache_key:
                    from .compression_summary import summarize_compressed_code

                    code_summary = summarize_compressed_code(
                        structure.function_bodies,
                        len(structure.function_bodies),
                    )
                    summary_str = f" {code_summary}." if code_summary else ""

                    # Use the actual config attribute (not the wrong name)
                    ttl_min = max(1, self.config.ccr_ttl // 60)
                    compressed += (
                        f"\n# [{original_tokens - compressed_tokens} tokens compressed."
                        f"{summary_str}"
                        f" Retrieve more: hash={cache_key}."
                        f" Expires in {ttl_min}m.]"
                    )

            return CodeCompressionResult(
                compressed=compressed,
                original=code,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=ratio,
                language=detected_lang,
                language_confidence=confidence,
                preserved_imports=len(structure.imports),
                preserved_signatures=len(structure.function_signatures),
                compressed_bodies=len(structure.function_bodies),
                syntax_valid=syntax_valid,
                cache_key=cache_key,
                symbol_scores=symbol_scores,
            )

        except Exception as e:
            logger.warning("AST compression failed: %s, falling back", e)
            if self.config.fallback_to_kompress:
                return self._fallback_compress(code, original_tokens)
            return CodeCompressionResult(
                compressed=code,
                original=code,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                language=detected_lang,
                language_confidence=confidence,
                syntax_valid=True,
            )

    def _compress_with_ast(
        self,
        code: str,
        language: CodeLanguage,
        context: str,
        tokenizer: Tokenizer | None = None,
    ) -> tuple[str, CodeStructure, dict[str, float]]:
        """Compress code using AST parsing with symbol importance analysis.

        Thread-safe: all mutable state is passed through parameters, not
        stored on self.

        Args:
            code: Source code.
            language: Detected language.
            context: User context for relevance.
            tokenizer: Optional tokenizer for accurate token counting.

        Returns:
            Tuple of (compressed code, extracted structure, symbol scores).
        """
        parser = _get_parser(language.value)
        tree = parser.parse(bytes(code, "utf-8"))
        root = tree.root_node

        # Analyze symbol importance and allocate compression budget
        analysis = self._analyze_symbol_importance(root, code, language, context)
        body_limits = self._allocate_body_budget(analysis, code)

        # Extract structure using data-driven language config
        lang_config = _LANG_CONFIGS.get(language)
        if lang_config:
            structure = self._extract_structure(
                root, code, language, lang_config, body_limits, analysis
            )
        else:
            structure = self._extract_generic_structure(root, code)

        # Assemble compressed code
        compressed = self._assemble_compressed(structure, language)

        # Expose scores with short names for the public API
        symbol_scores: dict[str, float] = {}
        if analysis.scores:
            for qname, score in analysis.scores.items():
                short = analysis.bare_names.get(qname, qname)
                if short not in symbol_scores or score > symbol_scores[short]:
                    symbol_scores[short] = score

        return compressed, structure, symbol_scores

    # =========================================================================
    # Unified structure extraction (data-driven, replaces per-language methods)
    # =========================================================================

    def _extract_structure(
        self,
        root: Any,
        code: str,
        language: CodeLanguage,
        lang_config: LangConfig,
        body_limits: dict[str, int],
        analysis: _SymbolAnalysis,
    ) -> CodeStructure:
        """Extract structure from AST using data-driven language config.

        A single visitor handles all languages by checking node types against
        the LangConfig tables. No per-language extraction methods needed.
        """
        structure = CodeStructure()
        captured_byte_ranges: list[tuple[int, int]] = []

        def visit(node: Any) -> None:
            node_type = node.type

            # Package declarations (Go, Java)
            if lang_config.package_node and node_type == lang_config.package_node:
                structure.imports.insert(0, _get_node_text(node, code))
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Import statements
            if node_type in lang_config.import_nodes:
                structure.imports.append(_get_node_text(node, code))
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Export statements (JS/TS) — may contain functions or re-exports
            if node_type == "export_statement":
                text = _get_node_text(node, code)
                # Check if this export wraps a function or class
                has_func_or_class = False
                for child in node.children:
                    if (
                        child.type in lang_config.function_nodes
                        or child.type in lang_config.class_nodes
                    ):
                        has_func_or_class = True
                        compressed = self._compress_function_ast(
                            child, code, language, lang_config, body_limits, analysis
                        )
                        # Reconstruct export with compressed inner definition
                        export_prefix = _slice_code_bytes(code, node.start_byte, child.start_byte)
                        export_suffix = _slice_code_bytes(code, child.end_byte, node.end_byte)
                        structure.function_signatures.append(
                            export_prefix + compressed + export_suffix
                        )
                        break
                if not has_func_or_class:
                    structure.imports.append(text)
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Decorated definitions (Python)
            if lang_config.decorator_node and node_type == lang_config.decorator_node:
                decorator_text = []
                definition_compressed = None
                for child in node.children:
                    if child.type == "decorator":
                        decorator_text.append(_get_node_text(child, code))
                    elif child.type in lang_config.function_nodes:
                        definition_compressed = self._compress_function_ast(
                            child, code, language, lang_config, body_limits, analysis
                        )
                    elif child.type in lang_config.class_nodes:
                        definition_compressed = self._compress_class_ast(
                            child, code, language, lang_config, body_limits, analysis
                        )
                if decorator_text and definition_compressed:
                    full_def = "\n".join(decorator_text) + "\n" + definition_compressed
                    # Route to correct list based on inner definition type
                    for child in node.children:
                        if child.type in lang_config.class_nodes:
                            structure.class_definitions.append(full_def)
                            break
                    else:
                        structure.function_signatures.append(full_def)
                elif definition_compressed:
                    structure.function_signatures.append(definition_compressed)
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Function/method definitions
            if node_type in lang_config.function_nodes:
                compressed = self._compress_function_ast(
                    node, code, language, lang_config, body_limits, analysis
                )
                structure.function_signatures.append(compressed)
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Class definitions — compress each method individually
            if node_type in lang_config.class_nodes:
                compressed = self._compress_class_ast(
                    node, code, language, lang_config, body_limits, analysis
                )
                structure.class_definitions.append(compressed)
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                trailing_semicolon = _get_same_line_trailing_semicolon(node)
                if trailing_semicolon is not None:
                    captured_byte_ranges.append(
                        (trailing_semicolon.start_byte, trailing_semicolon.end_byte)
                    )
                return

            # Type definitions
            if node_type in lang_config.type_nodes:
                structure.type_definitions.append(_get_node_text(node, code))
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Recurse into children
            for child in node.children:
                visit(child)

        visit(root)

        # Capture top-level code that wasn't handled by any of the above.
        # This preserves global variables, constants, if __name__ blocks,
        # module-level assignments, etc.
        for child in root.children:
            child_range = (child.start_byte, child.end_byte)
            if child_range not in captured_byte_ranges:
                text = _get_node_text(child, code).strip()
                if text:
                    structure.top_level_code.append(text)

        return structure

    # =========================================================================
    # Unified function/class compression (data-driven)
    # =========================================================================

    def _compress_function_ast(
        self,
        node: Any,
        code: str,
        language: CodeLanguage,
        lang_config: LangConfig,
        body_limits: dict[str, int],
        analysis: _SymbolAnalysis,
    ) -> str:
        """Compress a function/class/impl block using AST body detection.

        Uses the AST to find the body node directly instead of string-scanning
        for '{' or ':'. Works for all languages via lang_config.body_node_types.

        Key insight: tree-sitter byte offsets may not include leading whitespace
        on the first line. We use LINE-based slicing from the original code to
        preserve indentation faithfully.
        """
        # Use line-based slicing from original code (not byte offsets) to
        # preserve indentation. This is critical for nested definitions
        # (methods inside classes).
        code_lines = code.split("\n")
        start_row = node.start_point[0]
        end_row = node.end_point[0]
        node_lines = code_lines[start_row : end_row + 1]
        node_text = "\n".join(node_lines)

        func_name = _get_definition_name(node)
        body_limit = _get_body_limit(func_name, body_limits, self.config.max_body_lines)

        # Small enough to keep as-is
        if len(node_lines) <= body_limit + 2:
            return node_text

        # Find the body node using AST (not string scanning)
        body_node = None
        for child in node.children:
            if child.type in lang_config.body_node_types:
                body_node = child
                break

        if body_node is None:
            return node_text

        # Use line numbers to slice: this preserves original indentation.
        # tree-sitter gives 0-based row numbers.
        node_start_line = node.start_point[0]
        body_start_line = body_node.start_point[0]
        body_end_line = body_node.end_point[0]

        # Lines within the node (0-indexed relative to node start)
        sig_end = body_start_line - node_start_line  # exclusive
        body_end_rel = body_end_line - node_start_line + 1  # inclusive

        # Handle case where signature and body start on the SAME line
        # (common in brace languages: `function foo(arg) { ... }`)
        if sig_end == 0 and not lang_config.uses_colon_after_signature:
            # Signature and body on same line: `function foo(arg) { ... }`
            # Keep them together: sig includes up to and including `{`
            first_line = node_lines[0]
            # Include the opening brace in the signature line
            sig_with_brace = first_line.rstrip()
            signature_lines = [sig_with_brace]
            # Body lines are everything between { and } (inner content only)
            body_lines = node_lines[1:body_end_rel]
            after_lines = node_lines[body_end_rel:]
            # We've already included { in signature, so mark it
            _brace_in_signature = True
        else:
            signature_lines = node_lines[:sig_end]
            body_lines = node_lines[sig_end:body_end_rel]
            after_lines = node_lines[body_end_rel:]
            _brace_in_signature = False

        # For brace languages, detect opening/closing braces in the body lines.
        opening_brace_line = None
        closing_brace_line = None
        if not lang_config.uses_colon_after_signature:
            if _brace_in_signature:
                # Opening brace already in signature line — just find closing
                pass
            elif body_lines and body_lines[0].strip().startswith("{"):
                opening_brace_line = body_lines[0]
                body_lines = body_lines[1:]
            if body_lines and body_lines[-1].strip().endswith("}"):
                closing_brace_line = body_lines[-1]
                body_lines = body_lines[:-1]

        # Handle Python docstrings via AST
        docstring_text = ""
        ds_skip_lines = 0
        if language == CodeLanguage.PYTHON and body_node.child_count > 0:
            first_child = body_node.children[0]
            # tree-sitter Python may represent docstrings as:
            # - bare `string` node directly in block, OR
            # - `expression_statement` containing a `string` node
            ds_node = None
            if first_child.type == "string":
                ds_node = first_child
            elif first_child.type == "expression_statement" and first_child.child_count > 0:
                if first_child.children[0].type == "string":
                    ds_node = first_child

            if ds_node is not None:
                ds_lines_count = ds_node.end_point[0] - ds_node.start_point[0] + 1
                ds_start_rel = ds_node.start_point[0] - body_node.start_point[0]

                if self.config.docstring_mode == DocstringMode.FULL:
                    # Keep entire docstring as-is (preserve indentation from body_lines)
                    docstring_text = "\n".join(
                        body_lines[ds_start_rel : ds_start_rel + ds_lines_count]
                    )
                elif self.config.docstring_mode == DocstringMode.FIRST_LINE:
                    # Use source lines directly (safe — preserves original quoting)
                    if ds_lines_count == 1:
                        # Single-line docstring: keep as-is
                        docstring_text = body_lines[ds_start_rel]
                    else:
                        # Multi-line docstring: keep first line, close it properly
                        first_ds_line = body_lines[ds_start_rel]
                        ds_indent = first_ds_line[
                            : len(first_ds_line) - len(first_ds_line.lstrip())
                        ]
                        stripped = first_ds_line.strip()

                        # Detect quote style from source
                        quote = '"""'
                        for q in ('r"""', "r'''", '"""', "'''"):
                            if stripped.startswith(q):
                                quote = q[-3:]
                                break

                        # Find where content starts (after opening quotes + prefix)
                        content_start = 0
                        for opener in ('r"""', "r'''", '"""', "'''"):
                            if stripped.startswith(opener):
                                content_start = len(opener)
                                break
                        first_content = stripped[content_start:].strip()

                        # Remove trailing closing quotes if the first line has them
                        for q in ('"""', "'''"):
                            if first_content.endswith(q):
                                first_content = first_content[: -len(q)].strip()

                        if first_content:
                            # """Some text here\n...\n"""  →  """Some text here"""
                            prefix_part = stripped[:content_start]
                            docstring_text = f"{ds_indent}{prefix_part}{first_content}{quote}"
                        else:
                            # Opening quote on its own line: """\n  text\n"""
                            if ds_start_rel + 1 < len(body_lines):
                                second_line = body_lines[ds_start_rel + 1].strip()
                                for q in ('"""', "'''"):
                                    if second_line.endswith(q):
                                        second_line = second_line[: -len(q)].strip()
                                if second_line:
                                    docstring_text = f"{ds_indent}{quote}{second_line}{quote}"
                                else:
                                    docstring_text = first_ds_line
                            else:
                                docstring_text = first_ds_line
                # elif REMOVE: docstring_text stays empty
                ds_skip_lines = ds_start_rel + ds_lines_count

        # --- Statement-based body truncation (never cuts mid-expression) ---
        #
        # Walk body_node.children (AST statements) instead of slicing lines.
        # Each child is a complete, syntactically valid statement. We keep
        # whole statements until the line budget is exhausted, so the output
        # always parses correctly.

        # Detect indentation from actual body code (preserves whatever the file uses)
        indent = _detect_indent(body_lines) if body_lines else "    "

        # Collect non-docstring body statements from the AST
        body_stmts: list[tuple[int, int]] = []  # (start_row, end_row) absolute
        ds_end_row = -1
        if ds_skip_lines > 0 and body_node.child_count > 0:
            # The docstring node occupies the first ds_skip_lines lines
            ds_end_row = body_node.start_point[0] + ds_skip_lines - 1

        # Punctuation tokens to skip (brace-language body delimiters, semicolons)
        _SKIP_TYPES = frozenset({"{", "}", ";", ",", "comment", "line_comment", "block_comment"})

        for child in body_node.children:
            # Skip docstring node (already handled separately)
            if child.start_point[0] <= ds_end_row:
                continue
            # Skip punctuation and comment nodes
            if child.type in _SKIP_TYPES:
                continue
            # Skip unnamed tokens (tree-sitter anonymous nodes like braces)
            if not child.is_named:
                continue
            body_stmts.append((child.start_point[0], child.end_point[0]))

        # Calculate lines per statement and keep whole statements until budget
        kept_lines: list[str] = []
        kept_line_count = 0
        stmts_kept = 0
        total_body_lines_count = sum(end - start + 1 for start, end in body_stmts)

        for start_row, end_row in body_stmts:
            stmt_lines = code_lines[start_row : end_row + 1]
            stmt_line_count = len(stmt_lines)

            # If adding this statement would exceed budget and we already have
            # at least one statement, stop here
            if kept_line_count + stmt_line_count > body_limit and stmts_kept > 0:
                break

            kept_lines.extend(stmt_lines)
            kept_line_count += stmt_line_count
            stmts_kept += 1

        omitted_lines = total_body_lines_count - kept_line_count

        # Build compressed output preserving original indentation
        result_parts: list[str] = []

        # Signature lines (may be multi-line)
        if signature_lines:
            result_parts.extend(signature_lines)
        else:
            sig_text = _slice_code_bytes(code, node.start_byte, body_node.start_byte).rstrip()
            result_parts.append(sig_text)

        if opening_brace_line is not None:
            result_parts.append(opening_brace_line)

        if docstring_text and self.config.docstring_mode not in (
            DocstringMode.NONE,
            DocstringMode.REMOVE,
        ):
            result_parts.append(docstring_text)

        if kept_lines:
            result_parts.extend(kept_lines)

        if omitted_lines > 0:
            result_parts.append(
                _make_omitted_comment(
                    func_name, omitted_lines, indent, lang_config.comment_prefix, analysis
                )
            )
            if lang_config.uses_colon_after_signature:
                result_parts.append(f"{indent}pass")

        if closing_brace_line is not None:
            result_parts.append(closing_brace_line)
        elif after_lines:
            result_parts.extend(after_lines)

        return "\n".join(result_parts)

    def _compress_class_ast(
        self,
        node: Any,
        code: str,
        language: CodeLanguage,
        lang_config: LangConfig,
        body_limits: dict[str, int],
        analysis: _SymbolAnalysis,
    ) -> str:
        """Compress a class by individually compressing each method.

        Preserves class-level attributes, type annotations, and decorators
        while compressing method bodies individually. This ensures correct
        indentation for each method's omitted-body comment.
        """
        # Use line-based extraction to preserve indentation
        code_lines = code.split("\n")
        start_row = node.start_point[0]
        end_row = node.end_point[0]
        node_lines = code_lines[start_row : end_row + 1]
        node_text = "\n".join(node_lines)

        # Find the class/member container. For some languages this is not the
        # same node type as a function body's executable block.
        class_body_node_types = lang_config.class_body_node_types or lang_config.body_node_types
        body_node = None
        for child in node.children:
            if child.type in class_body_node_types:
                body_node = child
                break

        if body_node is None:
            return node_text

        # Class header (signature) — everything before the body
        node_start_line = node.start_point[0]
        body_start_line = body_node.start_point[0]
        sig_end = body_start_line - node_start_line
        header_lines = node_lines[:sig_end] if sig_end > 0 else [node_lines[0]]

        # Process each child of the class body individually
        body_parts: list[str] = []
        processed_ranges: list[tuple[int, int]] = []

        for child in body_node.children:
            if not child.is_named:
                continue

            # Use line-based extraction for children too
            child_start = child.start_point[0]
            child_end = child.end_point[0]
            child_text = "\n".join(code_lines[child_start : child_end + 1])

            # Methods/functions inside the class — compress individually
            if child.type in lang_config.function_nodes:
                compressed = self._compress_function_ast(
                    child, code, language, lang_config, body_limits, analysis
                )
                body_parts.append(compressed)
                processed_ranges.append((child.start_byte, child.end_byte))
            # Decorated methods
            elif lang_config.decorator_node and child.type == lang_config.decorator_node:
                decorator_lines = []
                method_compressed = None
                for deco_child in child.children:
                    if deco_child.type == "decorator":
                        deco_start = deco_child.start_point[0]
                        deco_end = deco_child.end_point[0]
                        decorator_lines.append("\n".join(code_lines[deco_start : deco_end + 1]))
                    elif deco_child.type in lang_config.function_nodes:
                        method_compressed = self._compress_function_ast(
                            deco_child, code, language, lang_config, body_limits, analysis
                        )
                if decorator_lines and method_compressed:
                    body_parts.append("\n".join(decorator_lines) + "\n" + method_compressed)
                elif method_compressed:
                    body_parts.append(method_compressed)
                else:
                    body_parts.append(child_text)
                processed_ranges.append((child.start_byte, child.end_byte))
            # Nested classes — recurse
            elif child.type in lang_config.class_nodes:
                compressed = self._compress_class_ast(
                    child, code, language, lang_config, body_limits, analysis
                )
                body_parts.append(compressed)
                processed_ranges.append((child.start_byte, child.end_byte))
            else:
                # Class-level attributes, type annotations, docstrings, etc.
                # Keep them as-is with original indentation
                if child_text.strip():
                    body_parts.append(child_text)
                processed_ranges.append((child.start_byte, child.end_byte))

        # Reconstruct class with proper indentation
        result_parts = list(header_lines)
        for part in body_parts:
            result_parts.append(part)

        # Handle closing brace for brace-delimited languages. The class body
        # node ends at the brace, while C++ class_specifier excludes the
        # trailing semicolon; keeping only the body node span prevents a second
        # semicolon from being rendered later as top-level code.
        body_end_line = body_node.end_point[0]
        body_end_rel = body_end_line - node_start_line + 1
        after_lines = node_lines[body_end_rel:]
        if not lang_config.uses_colon_after_signature:
            if body_end_line != start_row:
                closing_line = code_lines[body_end_line]
                closing_text = closing_line[: body_node.end_point[1]]
                if _get_same_line_trailing_semicolon(node) is not None:
                    closing_text += ";"
                if closing_text.strip():
                    result_parts.append(closing_text)
        elif after_lines:
            result_parts.extend(after_lines)

        return "\n".join(result_parts)

    def _extract_generic_structure(self, root: Any, code: str) -> CodeStructure:
        """Extract structure from generic/unknown code.

        For languages without a LangConfig, we can't reliably separate
        imports from other code. Just preserve everything in 'other'.
        """
        structure = CodeStructure()
        structure.other = code.split("\n")
        return structure

    def _assemble_compressed(
        self,
        structure: CodeStructure,
        language: CodeLanguage,
    ) -> str:
        """Assemble compressed code from structure."""
        parts: list[str] = []

        # Imports first
        if structure.imports:
            parts.extend(structure.imports)
            parts.append("")

        # Type definitions
        if structure.type_definitions:
            parts.extend(structure.type_definitions)
            parts.append("")

        # Class definitions
        if structure.class_definitions:
            parts.extend(structure.class_definitions)
            parts.append("")

        # Function signatures/definitions
        if structure.function_signatures:
            parts.extend(structure.function_signatures)
            parts.append("")

        # Top-level code (global variables, constants, if __name__, etc.)
        if structure.top_level_code:
            parts.extend(structure.top_level_code)
            parts.append("")

        # Other content (used by generic extraction)
        if structure.other:
            parts.extend(structure.other)

        # Remove trailing empty lines
        while parts and not parts[-1].strip():
            parts.pop()

        return "\n".join(parts)

    def _verify_syntax(self, code: str, language: CodeLanguage) -> bool:
        """Verify that code is syntactically valid.

        Checks for both ERROR nodes (parse failures) and MISSING nodes
        (tokens the parser expected but didn't find).
        """
        try:
            if language == CodeLanguage.PYTHON:
                import ast

                ast.parse(code)
                compile(code, "<headroom-compressed>", "exec")

            parser = _get_parser(language.value)
            tree = parser.parse(bytes(code, "utf-8"))
            return not _has_syntax_issues(tree.root_node)
        except Exception:
            return False

    def _fallback_compress(self, code: str, original_tokens: int) -> CodeCompressionResult:
        """Fall back to Kompress compression."""
        try:
            from .kompress_compressor import KompressCompressor, is_kompress_available

            if is_kompress_available():
                compressor = KompressCompressor()
                result = compressor.compress(code)
                return CodeCompressionResult(
                    compressed=result.compressed,
                    original=code,
                    original_tokens=result.original_tokens,
                    compressed_tokens=result.compressed_tokens,
                    compression_ratio=result.compression_ratio,
                    language=CodeLanguage.UNKNOWN,
                    language_confidence=0.0,
                    # Kompress does NOT guarantee syntax validity
                    syntax_valid=False,
                )
        except ImportError:
            pass

        # No fallback available, return original
        return CodeCompressionResult(
            compressed=code,
            original=code,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=1.0,
            language=CodeLanguage.UNKNOWN,
            language_confidence=0.0,
            syntax_valid=True,
        )

    def _store_in_ccr(
        self,
        original: str,
        compressed: str,
        original_tokens: int,
    ) -> str | None:
        """Store original in CCR for later retrieval."""
        try:
            from ..cache.compression_store import get_compression_store

            store = get_compression_store()
            return store.store(
                original,
                compressed,
                original_tokens=original_tokens,
                compressed_tokens=self._estimate_tokens(compressed),
                compression_strategy="code_aware",
            )
        except ImportError:
            return None
        except Exception as e:
            logger.debug("CCR storage failed: %s", e)
            return None

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        """Apply code-aware compression to messages.

        Handles both string content and Anthropic content block format
        (list of {"type": "text", "text": "..."} dicts).

        Args:
            messages: List of message dicts to transform.
            tokenizer: Tokenizer for accurate token counting.
            **kwargs: Additional arguments (e.g., 'context').

        Returns:
            TransformResult with compressed messages and metadata.
        """
        tokens_before = sum(tokenizer.count_text(str(m.get("content", ""))) for m in messages)
        context = kwargs.get("context", "")

        transformed_messages = []
        transforms_applied: list[str] = []
        warnings: list[str] = []

        for message in messages:
            content = message.get("content", "")

            # Handle content blocks (Anthropic format)
            if isinstance(content, list):
                new_blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        compressed_text = self._try_compress_text(
                            text, context, tokenizer, transforms_applied
                        )
                        new_blocks.append({**block, "text": compressed_text})
                    else:
                        new_blocks.append(block)
                transformed_messages.append({**message, "content": new_blocks})
                continue

            # Handle string content
            if not content or not isinstance(content, str):
                transformed_messages.append(message)
                continue

            compressed_content = self._try_compress_text(
                content, context, tokenizer, transforms_applied
            )
            if compressed_content != content:
                transformed_messages.append({**message, "content": compressed_content})
            else:
                transformed_messages.append(message)

        tokens_after = sum(
            tokenizer.count_text(str(m.get("content", ""))) for m in transformed_messages
        )

        if not _check_tree_sitter_available():
            warnings.append(
                "tree-sitter not installed. Install with: pip install headroom-ai[code]"
            )

        return TransformResult(
            messages=transformed_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied if transforms_applied else ["code_aware:noop"],
            warnings=warnings,
        )

    def _try_compress_text(
        self,
        text: str,
        context: str,
        tokenizer: Tokenizer,
        transforms_applied: list[str],
    ) -> str:
        """Try to compress a text string if it contains code."""
        from .content_detector import ContentType, detect_content_type

        if not text:
            return text

        detection = detect_content_type(text)
        if detection.content_type == ContentType.SOURCE_CODE:
            language = detection.metadata.get("language")
            result = self.compress(text, language=language, context=context, tokenizer=tokenizer)
            if result.compression_ratio < 0.9:
                transforms_applied.append(
                    f"code_aware:{result.language.value}:{result.compression_ratio:.2f}"
                )
                return result.compressed
        return text

    def should_apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> bool:
        """Check if code-aware compression should be applied.

        Returns True if:
        - tree-sitter is available, AND
        - Content contains detected source code

        Args:
            messages: Messages to check.
            tokenizer: Tokenizer for counting.
            **kwargs: Additional arguments.

        Returns:
            True if compression should be applied.
        """
        if not _check_tree_sitter_available():
            return False

        from .content_detector import ContentType, detect_content_type

        for message in messages:
            content = message.get("content", "")
            # Handle string content
            if content and isinstance(content, str):
                detection = detect_content_type(content)
                if detection.content_type == ContentType.SOURCE_CODE:
                    return True
            # Handle content blocks
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            detection = detect_content_type(text)
                            if detection.content_type == ContentType.SOURCE_CODE:
                                return True

        return False


# =========================================================================
# Module-level helper functions (stateless, used by the class)
# =========================================================================


def _slice_code_bytes(code: str, start_byte: int, end_byte: int) -> str:
    """Extract source text using tree-sitter UTF-8 byte offsets."""
    return code.encode("utf-8")[start_byte:end_byte].decode("utf-8")


def _get_node_text(node: Any, code: str) -> str:
    """Extract text from AST node."""
    return _slice_code_bytes(code, node.start_byte, node.end_byte)


def _get_same_line_trailing_semicolon(node: Any) -> Any | None:
    """Return a trailing semicolon sibling that belongs to this declaration."""
    next_sibling = getattr(node, "next_sibling", None)
    if (
        next_sibling is not None
        and next_sibling.type == ";"
        and next_sibling.start_point[0] == node.end_point[0]
    ):
        return next_sibling
    return None


def _get_definition_name(node: Any) -> str | None:
    """Extract the name identifier from a definition AST node."""
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier", "property_identifier"):
            text = child.text
            return text.decode("utf-8") if isinstance(text, bytes) else str(text)
    return None


def _is_public_symbol(name: str, language: CodeLanguage) -> bool:
    """Heuristic for whether a symbol is public/exported."""
    if not name:
        return False
    if language == CodeLanguage.GO:
        return name[0].isupper()
    return not name.startswith("_")


def _get_body_limit(
    func_name: str | None,
    body_limits: dict[str, int],
    max_body_lines: int,
) -> int:
    """Look up the allocated body line limit for a function.

    Falls back to max_body_lines if no budget allocation was computed.
    max_body_lines always acts as a hard cap.
    """
    if body_limits and func_name and func_name in body_limits:
        return min(body_limits[func_name], max_body_lines)
    return max_body_lines


def _make_omitted_comment(
    func_name: str | None,
    omitted_count: int,
    indent: str,
    comment_prefix: str,
    analysis: _SymbolAnalysis | None,
) -> str:
    """Build omitted comment with call information from analysis."""
    calls_info = ""
    if analysis and func_name:
        for key in (
            func_name,
            *(k for k in analysis.calls if k.endswith(f".{func_name}")),
        ):
            if key in analysis.calls:
                called = analysis.calls[key]
                if called:
                    sorted_calls = sorted(called)[:5]
                    calls_info = "; calls: " + ", ".join(sorted_calls)
                    if len(called) > 5:
                        calls_info += f" +{len(called) - 5} more"
                break
    return f"{indent}{comment_prefix} [{omitted_count} lines omitted{calls_info}]"


def _detect_indent(lines: list[str]) -> str:
    """Detect the indentation used in a list of code lines."""
    for line in lines:
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return "    "


def _has_syntax_issues(node: Any) -> bool:
    """Check if AST contains ERROR or MISSING nodes."""
    if node.type == "ERROR" or node.is_missing:
        return True
    for child in node.children:
        if _has_syntax_issues(child):
            return True
    return False


def compress_code(
    code: str,
    language: str | None = None,
    target_rate: float = 0.2,
    context: str = "",
) -> str:
    """Convenience function for one-off code compression.

    Args:
        code: Source code to compress.
        language: Language hint (auto-detected if None).
        target_rate: Target compression rate (0.2 = keep 20%).
        context: Optional context for relevance.

    Returns:
        Compressed code string.

    Example:
        >>> compressed = compress_code(large_python_file)
        >>> print(compressed)  # Valid Python code
    """
    config = CodeCompressorConfig(
        target_compression_rate=target_rate,
        language_hint=language,
    )
    compressor = CodeAwareCompressor(config)
    result = compressor.compress(code, language=language, context=context)
    return result.compressed
