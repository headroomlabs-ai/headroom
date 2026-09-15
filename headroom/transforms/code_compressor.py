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
- Bash (validated and preserved losslessly)

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
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable
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


_UNSAFE_TREE_SITTER_LANGUAGES: frozenset[str] = frozenset({"perl"})

# ---------------------------------------------------------------------------
# Per-language invalid-syntax circuit breaker.
#
# The AST pass validates its own output and discards anything that no longer
# parses ("never serve broken code"), so a misbehaving grammar costs the full
# compression latency and then throws the work away — measured in the field at
# 270 discarded TypeScript compressions in one install's retained logs, each
# preceded by ~1.2s of compressor time. When a language keeps failing its own
# validation, stop attempting it for a cooldown instead of burning the latency
# on every request. The static _UNSAFE_TREE_SITTER_LANGUAGES set above is for
# grammars unsafe to LOAD; this breaker is for grammars that load fine but
# mangle real code.
#
# SCOPE: process-global, deliberately. One pathological payload pauses that
# language for every session served by this process, which is right for the
# single-user local proxy this ships as (a failing grammar is a property of the
# process's tree-sitter build, not of the caller) and wrong for a shared or
# hosted deployment, where one tenant would pause a language for the others.
# Keying by tenant needs a caller identity plumbed through the Transform
# interface, which carries code and language only; until something needs that,
# HEADROOM_CODE_SYNTAX_BREAKER=0 is the per-deployment opt-out.
_SYNTAX_BREAKER_WINDOW = 20
_SYNTAX_BREAKER_MIN_FAILURES = 10
_SYNTAX_BREAKER_COOLDOWN_S = 1800.0
_syntax_breaker_lock = threading.Lock()
_syntax_breaker_outcomes: dict[str, deque[bool]] = {}
_syntax_breaker_open_until: dict[str, float] = {}
_syntax_breaker_trips: dict[str, int] = {}


def _now() -> float:
    """Monotonic clock behind one indirection, so tests patch this instead of
    the real ``time`` module (which anything else running concurrently reads)."""
    return time.monotonic()


def syntax_breaker_status() -> dict[str, dict[str, Any]]:
    """Per-language breaker state, surfaced on /stats.

    While the breaker is open that language compresses at ratio 1.0, so savings
    drop with nothing in the payload explaining why. Empty until something
    trips, so a healthy install carries no extra keys.
    """
    now = _now()
    with _syntax_breaker_lock:
        languages = set(_syntax_breaker_trips) | set(_syntax_breaker_open_until)
        return {
            language: {
                "open": _syntax_breaker_open_until.get(language, 0.0) > now,
                "trips": _syntax_breaker_trips.get(language, 0),
                "reopens_in_seconds": max(
                    0.0, round(_syntax_breaker_open_until.get(language, 0.0) - now, 1)
                ),
            }
            for language in sorted(languages)
        }


def _syntax_breaker_enabled() -> bool:
    return os.environ.get("HEADROOM_CODE_SYNTAX_BREAKER", "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


def _syntax_breaker_open(language: str) -> bool:
    """True while AST compression for *language* is paused after repeat failures."""
    if not _syntax_breaker_enabled():
        return False
    with _syntax_breaker_lock:
        if _syntax_breaker_open_until.get(language, 0.0) <= _now():
            return False
    logger.debug("Code compression for %s skipped: syntax breaker open", language)
    return True


def _record_syntax_outcome(language: str, valid: bool) -> None:
    """Track validation outcomes; trip the breaker on a failing window."""
    if not _syntax_breaker_enabled():
        return
    tripped_failures = 0
    with _syntax_breaker_lock:
        window = _syntax_breaker_outcomes.setdefault(language, deque(maxlen=_SYNTAX_BREAKER_WINDOW))
        window.append(valid)
        failures = sum(1 for ok in window if not ok)
        if failures >= _SYNTAX_BREAKER_MIN_FAILURES:
            _syntax_breaker_open_until[language] = _now() + _SYNTAX_BREAKER_COOLDOWN_S
            _syntax_breaker_trips[language] = _syntax_breaker_trips.get(language, 0) + 1
            # A fresh window after the cooldown must re-earn the trip.
            window.clear()
            tripped_failures = failures
    if tripped_failures:
        logger.warning(
            "Code compression for %s paused for %.0f min: %d of the last %d attempts "
            "produced invalid syntax and were discarded "
            "(HEADROOM_CODE_SYNTAX_BREAKER=0 disables this breaker)",
            language,
            _SYNTAX_BREAKER_COOLDOWN_S / 60.0,
            tripped_failures,
            _SYNTAX_BREAKER_WINDOW,
        )


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
    if language in _UNSAFE_TREE_SITTER_LANGUAGES:
        raise ValueError(f"Language '{language}' is quarantined for code-aware compression.")
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
                f"Supported: python, javascript, typescript, go, rust, java, c, cpp, csharp, php, bash. "
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
    CSHARP = "csharp"
    PHP = "php"
    BASH = "bash"
    SHELL = "bash"  # Alias: tree-sitter-language-pack exposes the Bash grammar.
    UNKNOWN = "unknown"


# Common language hints and markdown fence tags that are not the canonical
# ``CodeLanguage`` value. Mapping them here keeps ``` ```js ``` / ``` ```ts ```
# / ``` ```py ``` fenced blocks (and callers that pass an alias) on the
# code-aware path instead of raising ValueError.
_LANGUAGE_ALIASES: dict[str, CodeLanguage] = {
    "js": CodeLanguage.JAVASCRIPT,
    "jsx": CodeLanguage.JAVASCRIPT,
    "mjs": CodeLanguage.JAVASCRIPT,
    "cjs": CodeLanguage.JAVASCRIPT,
    "node": CodeLanguage.JAVASCRIPT,
    "ts": CodeLanguage.TYPESCRIPT,
    "tsx": CodeLanguage.TYPESCRIPT,
    "py": CodeLanguage.PYTHON,
    "python3": CodeLanguage.PYTHON,
    "golang": CodeLanguage.GO,
    "rs": CodeLanguage.RUST,
    "c++": CodeLanguage.CPP,
    "cxx": CodeLanguage.CPP,
    "cc": CodeLanguage.CPP,
    "hpp": CodeLanguage.CPP,
    "pl": CodeLanguage.PERL,
    "phtml": CodeLanguage.PHP,
    "php5": CodeLanguage.PHP,
    "php7": CodeLanguage.PHP,
    "php8": CodeLanguage.PHP,
    "shell": CodeLanguage.BASH,
    "sh": CodeLanguage.BASH,
    "zsh": CodeLanguage.BASH,
    "shellscript": CodeLanguage.BASH,
}


def coerce_language(value: str) -> CodeLanguage:
    """Map a language hint or markdown fence tag to a ``CodeLanguage``.

    Accepts the canonical enum values and common aliases/fence tags
    (``js``/``ts``/``py``/...). Unknown strings return ``CodeLanguage.UNKNOWN``
    instead of raising ``ValueError`` from ``CodeLanguage(value)``, so an
    unrecognized fence tag falls back to content-based detection rather than
    crashing the caller (or, inside the router, silently skipping code-aware
    compression because the ValueError is swallowed).
    """
    key = (value or "").strip().lower()
    if not key:
        return CodeLanguage.UNKNOWN
    try:
        return CodeLanguage(key)
    except ValueError:
        return _LANGUAGE_ALIASES.get(key, CodeLanguage.UNKNOWN)


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

    # Statement-retention node families
    terminal_nodes: frozenset[str] = frozenset()
    raise_nodes: frozenset[str] = frozenset()
    yield_nodes: frozenset[str] = frozenset()
    assignment_nodes: frozenset[str] = frozenset()
    call_nodes: frozenset[str] = frozenset()
    identifier_nodes: frozenset[str] = frozenset()
    call_target_nodes: frozenset[str] = frozenset()
    guard_statement_nodes: frozenset[str] = frozenset()
    else_clause_nodes: frozenset[str] = frozenset()
    member_access_nodes: frozenset[str] = frozenset()
    mutation_receiver_nodes: frozenset[str] = frozenset()
    effect_boundary_nodes: frozenset[str] = frozenset()

    # Quick pre-filter hints for language detection (substrings to check)
    detection_hints: tuple[str, ...] = ()
    # Optional override for node types that contain class/impl members.
    class_body_node_types: frozenset[str] | None = None
    # Optional container nodes (e.g. C# block-scoped namespaces) that wrap
    # type declarations in a member list. Routed through class compression so
    # their members compress and the wrapper isn't re-emitted verbatim.
    container_node_types: frozenset[str] | None = None
    # Optional opaque nodes (e.g. C# `#if`/`#endif` conditionals) preserved
    # verbatim without recursing into them. Recursing would capture their
    # descendants individually while the top-level pass re-emits the whole
    # wrapper verbatim, duplicating content. Prefer the false negative.
    opaque_node_types: frozenset[str] | None = None


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
        terminal_nodes=frozenset({"return_statement", "raise_statement", "yield_statement"}),
        raise_nodes=frozenset({"raise_statement"}),
        yield_nodes=frozenset({"yield_statement", "yield_expression"}),
        assignment_nodes=frozenset({"assignment", "augmented_assignment"}),
        call_nodes=frozenset({"call"}),
        identifier_nodes=frozenset({"identifier", "property_identifier", "type_identifier"}),
        call_target_nodes=frozenset({"identifier", "attribute", "property_identifier"}),
        guard_statement_nodes=frozenset({"if_statement"}),
        else_clause_nodes=frozenset({"else_clause"}),
        member_access_nodes=frozenset({"attribute", "subscript"}),
        mutation_receiver_nodes=frozenset({"attribute", "subscript"}),
        effect_boundary_nodes=frozenset({"raise_statement", "try_statement", "with_statement"}),
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
        terminal_nodes=frozenset({"return_statement", "throw_statement"}),
        raise_nodes=frozenset({"throw_statement"}),
        yield_nodes=frozenset(),
        assignment_nodes=frozenset(
            {"assignment_expression", "lexical_declaration", "variable_declaration"}
        ),
        call_nodes=frozenset({"call_expression"}),
        identifier_nodes=frozenset({"identifier", "property_identifier"}),
        call_target_nodes=frozenset({"identifier", "property_identifier", "member_expression"}),
        guard_statement_nodes=frozenset({"if_statement"}),
        else_clause_nodes=frozenset({"else_clause"}),
        member_access_nodes=frozenset({"member_expression", "subscript_expression"}),
        mutation_receiver_nodes=frozenset({"member_expression", "subscript_expression"}),
        effect_boundary_nodes=frozenset({"throw_statement", "try_statement"}),
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
        terminal_nodes=frozenset({"return_statement", "throw_statement"}),
        raise_nodes=frozenset({"throw_statement"}),
        yield_nodes=frozenset(),
        assignment_nodes=frozenset(
            {"assignment_expression", "lexical_declaration", "variable_declaration"}
        ),
        call_nodes=frozenset({"call_expression"}),
        identifier_nodes=frozenset({"identifier", "property_identifier", "type_identifier"}),
        call_target_nodes=frozenset({"identifier", "property_identifier", "member_expression"}),
        guard_statement_nodes=frozenset({"if_statement"}),
        else_clause_nodes=frozenset({"else_clause"}),
        member_access_nodes=frozenset({"member_expression", "subscript_expression"}),
        mutation_receiver_nodes=frozenset({"member_expression", "subscript_expression"}),
        effect_boundary_nodes=frozenset({"throw_statement", "try_statement"}),
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
        terminal_nodes=frozenset({"return_statement"}),
        raise_nodes=frozenset(),
        yield_nodes=frozenset(),
        assignment_nodes=frozenset({"short_var_declaration", "assignment_statement"}),
        call_nodes=frozenset({"call_expression"}),
        identifier_nodes=frozenset({"identifier", "field_identifier", "type_identifier"}),
        call_target_nodes=frozenset({"identifier", "selector_expression"}),
        guard_statement_nodes=frozenset({"if_statement"}),
        else_clause_nodes=frozenset({"else_clause"}),
        member_access_nodes=frozenset({"selector_expression", "index_expression"}),
        mutation_receiver_nodes=frozenset({"selector_expression", "index_expression"}),
        effect_boundary_nodes=frozenset(),
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
        terminal_nodes=frozenset({"return_expression"}),
        raise_nodes=frozenset(),
        yield_nodes=frozenset(),
        assignment_nodes=frozenset({"let_declaration", "assignment_expression"}),
        call_nodes=frozenset({"call_expression"}),
        identifier_nodes=frozenset({"identifier", "field_identifier", "type_identifier"}),
        call_target_nodes=frozenset({"identifier", "field_expression"}),
        guard_statement_nodes=frozenset({"if_expression"}),
        else_clause_nodes=frozenset(),
        member_access_nodes=frozenset({"field_expression", "index_expression"}),
        mutation_receiver_nodes=frozenset({"field_expression", "index_expression"}),
        effect_boundary_nodes=frozenset(),
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
        terminal_nodes=frozenset({"return_statement", "throw_statement"}),
        raise_nodes=frozenset({"throw_statement"}),
        yield_nodes=frozenset(),
        assignment_nodes=frozenset({"assignment_expression", "local_variable_declaration"}),
        call_nodes=frozenset({"method_invocation", "object_creation_expression"}),
        identifier_nodes=frozenset({"identifier", "type_identifier"}),
        call_target_nodes=frozenset({"identifier", "field_access"}),
        guard_statement_nodes=frozenset({"if_statement"}),
        else_clause_nodes=frozenset({"else_clause"}),
        member_access_nodes=frozenset({"field_access", "array_access"}),
        mutation_receiver_nodes=frozenset({"field_access", "array_access"}),
        effect_boundary_nodes=frozenset({"throw_statement", "try_statement"}),
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
        terminal_nodes=frozenset({"return_statement"}),
        raise_nodes=frozenset(),
        yield_nodes=frozenset(),
        assignment_nodes=frozenset({"assignment_expression", "declaration"}),
        call_nodes=frozenset({"call_expression"}),
        identifier_nodes=frozenset({"identifier", "field_identifier", "type_identifier"}),
        call_target_nodes=frozenset({"identifier", "field_expression", "pointer_expression"}),
        guard_statement_nodes=frozenset({"if_statement"}),
        else_clause_nodes=frozenset({"else_clause"}),
        member_access_nodes=frozenset(
            {"field_expression", "subscript_expression", "pointer_expression"}
        ),
        mutation_receiver_nodes=frozenset(
            {"field_expression", "subscript_expression", "pointer_expression"}
        ),
        effect_boundary_nodes=frozenset(),
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
        terminal_nodes=frozenset({"return_statement", "throw_statement"}),
        raise_nodes=frozenset({"throw_statement"}),
        yield_nodes=frozenset(),
        assignment_nodes=frozenset({"assignment_expression", "declaration"}),
        call_nodes=frozenset({"call_expression"}),
        identifier_nodes=frozenset({"identifier", "field_identifier", "type_identifier"}),
        call_target_nodes=frozenset({"identifier", "field_expression", "pointer_expression"}),
        guard_statement_nodes=frozenset({"if_statement"}),
        else_clause_nodes=frozenset({"else_clause"}),
        member_access_nodes=frozenset(
            {"field_expression", "subscript_expression", "pointer_expression"}
        ),
        mutation_receiver_nodes=frozenset(
            {"field_expression", "subscript_expression", "pointer_expression"}
        ),
        effect_boundary_nodes=frozenset({"throw_statement", "try_statement"}),
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
    CodeLanguage.CSHARP: LangConfig(
        import_nodes=frozenset({"using_directive", "file_scoped_namespace_declaration"}),
        function_nodes=frozenset(
            {
                "method_declaration",
                "constructor_declaration",
                "destructor_declaration",
                "operator_declaration",
                "local_function_statement",
            }
        ),
        class_nodes=frozenset(
            {
                "class_declaration",
                "struct_declaration",
                "record_declaration",
                "interface_declaration",
            }
        ),
        type_nodes=frozenset({"enum_declaration", "delegate_declaration"}),
        body_node_types=frozenset({"block"}),
        decorator_node=None,
        comment_prefix="//",
        uses_colon_after_signature=False,
        detection_hints=("using ", "namespace ", "public ", "private ", "void "),
        class_body_node_types=frozenset({"declaration_list"}),
        container_node_types=frozenset({"namespace_declaration"}),
        opaque_node_types=frozenset({"preproc_if"}),
    ),
    CodeLanguage.PHP: LangConfig(
        import_nodes=frozenset({"namespace_use_declaration"}),
        function_nodes=frozenset({"function_definition", "method_declaration"}),
        class_nodes=frozenset({"class_declaration", "interface_declaration", "trait_declaration"}),
        type_nodes=frozenset({"enum_declaration"}),
        body_node_types=frozenset({"compound_statement"}),
        decorator_node=None,
        comment_prefix="//",
        uses_colon_after_signature=False,
        # Statement-scoped `namespace App;` hoists to the top of the output
        # (before the use declarations); the rarer block-scoped
        # `namespace A { ... }` form takes the same path and is preserved
        # verbatim — valid output, no compression inside the block.
        package_node="namespace_definition",
        detection_hints=("<?php", "function ", "namespace ", "->", "$this"),
        class_body_node_types=frozenset({"declaration_list"}),
    ),
    CodeLanguage.BASH: LangConfig(
        # Shell control-flow nodes are opaque: their `then`/`fi`, `do`/`done`,
        # and `case` delimiters are anonymous grammar tokens.  The generic
        # extractor must not descend into them and re-emit only their commands,
        # which would silently corrupt otherwise valid shell code.
        import_nodes=frozenset(),
        function_nodes=frozenset(),
        class_nodes=frozenset(),
        type_nodes=frozenset(),
        body_node_types=frozenset(),
        decorator_node=None,
        comment_prefix="#",
        uses_colon_after_signature=False,
        detection_hints=("#!/bin/bash", "#!/usr/bin/env bash", " then", " fi", " esac"),
        opaque_node_types=frozenset(
            {
                "if_statement",
                "for_statement",
                "while_statement",
                "until_statement",
                "case_statement",
                "select_statement",
                "function_definition",
                "subshell",
                "compound_statement",
            }
        ),
    ),
}


@dataclass
class CodeStructure:
    """Extracted structure from parsed code."""

    header_code: list[str] = field(default_factory=list)
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
    CodeLanguage.CSHARP: [
        re.compile(r"^\s*using\s+[\w.]+\s*;", re.MULTILINE),
        re.compile(r"^\s*namespace\s+[\w.]+", re.MULTILINE),
        re.compile(
            r"^\s*(public|private|protected|internal|sealed|static|abstract|partial)\s+"
            r"(class|struct|record|interface|enum)\b",
            re.MULTILINE,
        ),
        re.compile(r"\bget;\s*set;", re.MULTILINE),
    ],
    CodeLanguage.PHP: [
        re.compile(r"<\?php\b"),
        re.compile(r"^\s*namespace\s+[\w\\]+\s*;", re.MULTILINE),
        re.compile(r"^\s*use\s+[\w\\]+(\s+as\s+\w+)?\s*;", re.MULTILINE),
        re.compile(
            r"^\s*(public|private|protected|static|abstract|final)?\s*function\s+\w+\s*\(",
            re.MULTILINE,
        ),
        re.compile(r"\$this->|->\w+\s*\(", re.MULTILINE),
    ],
    CodeLanguage.BASH: [
        re.compile(r"^\s*#!.*\b(?:bash|sh|zsh)\b", re.MULTILINE),
        re.compile(
            r"^\s*(?:if|then|elif|else|fi|for|while|until|do|done|case|esac|select)\b",
            re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:export|source|shopt|declare|local|alias|set|readonly|typeset)\b",
            re.MULTILINE,
        ),
        re.compile(r"\[\[.*\]\]|\$\{[^}]+\}|\$\([^)]+\)", re.MULTILINE),
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

    # Disambiguation: PHP's sigil variables ($x) overlap Perl's prefilter.
    # An explicit `<?php` open tag is unambiguous — no Perl source contains
    # it, so drop Perl from the candidates before the Perl-dominance guard
    # below returns UNKNOWN for what is actually PHP.
    if CodeLanguage.PHP in candidates and "<?php" in sample:
        candidates.pop(CodeLanguage.PERL, None)

    perl_score = candidates.get(CodeLanguage.PERL, 0)
    if perl_score > 0:
        best_non_perl = max(
            (score for lang, score in candidates.items() if lang != CodeLanguage.PERL),
            default=0,
        )
        if perl_score > best_non_perl:
            return CodeLanguage.UNKNOWN, 0.0
        candidates.pop(CodeLanguage.PERL, None)

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


class _StatementKind(Enum):
    """Lightweight statement categories for function-body selection."""

    RETURN = "return"
    RAISE = "raise"
    YIELD = "yield"
    IF = "if"
    FOR = "for"
    WHILE = "while"
    TRY = "try"
    WITH = "with"
    ASSIGN = "assign"
    CALL = "call"
    ASSERT = "assert"
    EXPR = "expr"
    UNKNOWN = "unknown"


@dataclass
class _StatementUnit:
    """A top-level function-body statement plus lightweight semantic signals."""

    idx: int
    node: Any
    text_lines: list[str]
    start_row: int
    end_row: int
    line_count: int
    kind: _StatementKind
    defined_names: set[str] = field(default_factory=set)
    referenced_names: set[str] = field(default_factory=set)
    called_names: set[str] = field(default_factory=set)
    has_return: bool = False
    has_raise: bool = False
    has_yield: bool = False
    has_side_effect: bool = False
    is_guard_clause: bool = False
    context_hit: bool = False
    importance_score: float = 0.0


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
        context_words, context_lower, context_has_cjk = _query_context_tokens(context)

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

            # Context boost: the relevance query named this symbol.
            if _symbol_in_context(short.lower(), context_words, context_lower, context_has_cjk):
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

        # Detect or use specified language. An explicit hint or fence tag may be
        # an alias (js/ts/py/...) or something we don't recognize — coerce it
        # instead of constructing CodeLanguage() directly (which raises), and
        # fall back to content detection when the hint is unknown.
        if language:
            detected_lang = coerce_language(language)
            if detected_lang == CodeLanguage.UNKNOWN:
                detected_lang, confidence = detect_language(code)
            else:
                confidence = 1.0
        elif self.config.language_hint:
            detected_lang = coerce_language(self.config.language_hint)
            if detected_lang == CodeLanguage.UNKNOWN:
                detected_lang, confidence = detect_language(code)
            else:
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
        if detected_lang == CodeLanguage.PERL:
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

        # Bash control-flow is recognized and parsed for validation, but is not
        # rewritten yet.  Its grammar uses anonymous delimiters (`then`, `fi`,
        # `do`, `done`, `esac`) that a generic AST reassembler cannot safely
        # preserve.  Returning the validated source keeps shell scripts
        # byte-for-byte intact instead of sending them to lossy Kompress.
        if detected_lang == CodeLanguage.BASH:
            # Validate only when a parser exists.  Without tree-sitter there is
            # nothing to validate against, and the source is returned unchanged
            # either way — report it as valid rather than as broken, matching
            # the tree-sitter-unavailable return below.
            syntax_valid = (
                self._verify_syntax(code, detected_lang) if _check_tree_sitter_available() else True
            )
            return CodeCompressionResult(
                compressed=code,
                original=code,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                language=detected_lang,
                language_confidence=confidence,
                syntax_valid=syntax_valid,
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

        # A language that keeps failing its own output validation gets its
        # attempts paused (see _record_syntax_outcome) — return the original
        # without paying the parse/compress latency for work that would be
        # discarded anyway.
        if _syntax_breaker_open(detected_lang.value):
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
            original_syntax_valid = True

            if not syntax_valid and detected_lang == CodeLanguage.PYTHON:
                original_syntax_valid = self._verify_syntax(code, detected_lang)
                if original_syntax_valid:
                    compressed, structure, symbol_scores = self._compress_with_ast(
                        code,
                        detected_lang,
                        context,
                        tokenizer,
                        recover_invalid_python_nodes=True,
                    )
                    compressed_tokens = self._estimate_tokens(compressed, tokenizer)
                    syntax_valid = self._verify_syntax(compressed, detected_lang)

            _record_syntax_outcome(detected_lang.value, syntax_valid)

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
        recover_invalid_python_nodes: bool = False,
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
        candidate_validator: Callable[[Any, str], str] | None = None

        if recover_invalid_python_nodes and language == CodeLanguage.PYTHON:
            code_bytes = code.encode("utf-8")

            def candidate_validator(node: Any, candidate_text: str) -> str:
                original_text = _slice_code_bytes(code, node.start_byte, node.end_byte)
                if candidate_text == original_text:
                    return candidate_text
                candidate_module = (
                    code_bytes[: node.start_byte]
                    + candidate_text.encode("utf-8")
                    + code_bytes[node.end_byte :]
                ).decode("utf-8")
                if self._verify_syntax(candidate_module, CodeLanguage.PYTHON):
                    return candidate_text
                return original_text

        # Analyze symbol importance and allocate compression budget
        analysis = self._analyze_symbol_importance(root, code, language, context)
        body_limits = self._allocate_body_budget(analysis, code)

        # Extract structure using data-driven language config
        lang_config = _LANG_CONFIGS.get(language)
        if lang_config:
            structure = self._extract_structure(
                root,
                code,
                language,
                lang_config,
                body_limits,
                analysis,
                context,
                candidate_validator=candidate_validator,
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
        context: str,
        candidate_validator: Callable[[Any, str], str] | None = None,
    ) -> CodeStructure:
        """Extract structure from AST using data-driven language config.

        A single visitor handles all languages by checking node types against
        the LangConfig tables. No per-language extraction methods needed.
        """
        structure = CodeStructure()
        captured_byte_ranges: list[tuple[int, int]] = []

        def _validated_candidate(node: Any, compressed: str) -> str:
            if candidate_validator is None:
                return compressed
            return candidate_validator(node, compressed)

        def visit(node: Any) -> None:
            node_type = node.type

            # Package declarations (Go, Java)
            if lang_config.package_node and node_type == lang_config.package_node:
                leading = _get_leading_comment_text(node, code, captured_byte_ranges)
                structure.imports.insert(0, leading + _get_node_text(node, code))
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Import statements
            if node_type in lang_config.import_nodes:
                leading = _get_leading_comment_text(node, code, captured_byte_ranges)
                structure.imports.append(leading + _get_node_text(node, code))
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Export statements (JS/TS) — may contain functions or re-exports
            if node_type == "export_statement":
                leading = _get_leading_comment_text(node, code, captured_byte_ranges)
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
                            child, code, language, lang_config, body_limits, analysis, context
                        )
                        # Reconstruct export with compressed inner definition
                        export_prefix = _slice_code_bytes(code, node.start_byte, child.start_byte)
                        export_suffix = _slice_code_bytes(code, child.end_byte, node.end_byte)
                        structure.function_signatures.append(
                            leading
                            + _validated_candidate(
                                node,
                                export_prefix + compressed + export_suffix,
                            )
                        )
                        break
                if not has_func_or_class:
                    structure.imports.append(leading + text)
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Decorated definitions (Python)
            if lang_config.decorator_node and node_type == lang_config.decorator_node:
                leading = _get_leading_comment_text(node, code, captured_byte_ranges)
                decorator_text = []
                definition_compressed = None
                for child in node.children:
                    if child.type == "decorator":
                        decorator_text.append(_get_node_text(child, code))
                    elif child.type in lang_config.function_nodes:
                        definition_compressed = self._compress_function_ast(
                            child, code, language, lang_config, body_limits, analysis, context
                        )
                    elif child.type in lang_config.class_nodes:
                        definition_compressed = self._compress_class_ast(
                            child, code, language, lang_config, body_limits, analysis, context
                        )
                if decorator_text and definition_compressed:
                    full_def = _validated_candidate(
                        node,
                        "\n".join(decorator_text) + "\n" + definition_compressed,
                    )
                    # Route to correct list based on inner definition type
                    for child in node.children:
                        if child.type in lang_config.class_nodes:
                            structure.class_definitions.append(leading + full_def)
                            break
                    else:
                        structure.function_signatures.append(leading + full_def)
                elif definition_compressed:
                    structure.function_signatures.append(
                        leading + _validated_candidate(node, definition_compressed)
                    )
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Function/method definitions
            if node_type in lang_config.function_nodes:
                leading = _get_leading_comment_text(node, code, captured_byte_ranges)
                compressed = self._compress_function_ast(
                    node, code, language, lang_config, body_limits, analysis, context
                )
                structure.function_signatures.append(
                    leading + _validated_candidate(node, compressed)
                )
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            if lang_config.container_node_types and node_type in lang_config.container_node_types:
                compressed = self._compress_class_ast(
                    node, code, language, lang_config, body_limits, analysis, context
                )
                structure.class_definitions.append(_validated_candidate(node, compressed))
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Class definitions — compress each method individually
            if node_type in lang_config.class_nodes:
                leading = _get_leading_comment_text(node, code, captured_byte_ranges)
                compressed = self._compress_class_ast(
                    node, code, language, lang_config, body_limits, analysis, context
                )
                structure.class_definitions.append(leading + _validated_candidate(node, compressed))
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                trailing_semicolon = _get_same_line_trailing_semicolon(node)
                if trailing_semicolon is not None:
                    captured_byte_ranges.append(
                        (trailing_semicolon.start_byte, trailing_semicolon.end_byte)
                    )
                return

            # Type definitions
            if node_type in lang_config.type_nodes:
                leading = _get_leading_comment_text(node, code, captured_byte_ranges)
                structure.type_definitions.append(leading + _get_node_text(node, code))
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Opaque regions (e.g. C# preprocessor conditionals) — preserved
            # verbatim, never recursed into. Recursing captures descendants
            # individually while the top-level pass re-emits the uncaptured
            # wrapper verbatim, duplicating its whole content in the output.
            # Blocks wrapping only import directives (`#if NET6_0\nusing X;`)
            # are emitted with the imports: usings must precede type
            # declarations, so appending them as trailing top-level code would
            # produce invalid output (and fall back to no compression).
            if lang_config.opaque_node_types and node_type in lang_config.opaque_node_types:
                child_types = [child.type for child in node.named_children]
                has_import = any(t in lang_config.import_nodes for t in child_types)
                has_declaration = any(
                    t in lang_config.class_nodes
                    or t in lang_config.type_nodes
                    or t in lang_config.function_nodes
                    or (lang_config.container_node_types and t in lang_config.container_node_types)
                    for t in child_types
                )
                if has_import and not has_declaration:
                    structure.imports.append(_get_node_text(node, code))
                else:
                    structure.top_level_code.append(_get_node_text(node, code))
                captured_byte_ranges.append((node.start_byte, node.end_byte))
                return

            # Recurse into children
            for child in node.children:
                visit(child)

        visit(root)

        # Capture top-level code that wasn't handled by any of the above.
        # This preserves global variables, constants, if __name__ blocks,
        # module-level assignments, etc. Uncaptured nodes that precede the
        # first captured node are file headers (license banners, C# `#region
        # License` blocks, module comments): they are emitted first, in
        # original order, rather than relocated to the end of the output —
        # e.g. tree-sitter-c-sharp rejects top-level `#region` after a type
        # declaration, which would fail validation and forfeit compression.
        first_captured = min((r[0] for r in captured_byte_ranges), default=None)
        for child in root.children:
            child_range = (child.start_byte, child.end_byte)
            if child_range not in captured_byte_ranges:
                text = _get_node_text(child, code).strip()
                if text:
                    if first_captured is not None and child.end_byte <= first_captured:
                        structure.header_code.append(text)
                    else:
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
        context: str = "",
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
        node_lines = _get_node_lines(node, code_lines)
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
            elif body_lines and body_lines[0].strip().endswith("{"):
                # Matches both a bare `{` line and a multi-line signature's
                # closing line (e.g. Go's `) error {`), where the brace
                # shares a line with the closing paren/return type rather
                # than starting one of its own.
                opening_brace_line = body_lines[0]
                body_lines = body_lines[1:]
            if body_lines and body_lines[-1].strip().endswith("}"):
                closing_brace_line = body_lines[-1]
                body_lines = body_lines[:-1]

        # Handle Python docstrings via AST
        docstring_text = ""
        ds_end_row = body_node.start_point[0] - 1
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
                ds_end_row = ds_node.end_point[0]

        # --- Statement-based body truncation (never cuts mid-expression) ---
        #
        # Walk body_node.children (AST statements) instead of slicing lines.
        # Each child is a complete, syntactically valid statement. We keep
        # whole statements until the line budget is exhausted, so the output
        # always parses correctly.

        # Detect indentation from actual body code (preserves whatever the file uses)
        indent = _detect_indent(body_lines) if body_lines else "    "

        # Collect non-docstring body statements from the AST
        statement_units = _extract_statement_units(
            body_node,
            code_lines,
            ds_end_row,
            context,
            analysis,
            func_name,
            language,
            lang_config,
        )
        total_body_lines_count = sum(unit.line_count for unit in statement_units)

        if not statement_units:
            kept_lines: list[str] = []
            kept_line_count = 0
        else:
            selected_units = _select_statement_units(statement_units, body_limit)
            kept_lines = []
            kept_line_count = 0
            for unit in selected_units:
                kept_lines.extend(unit.text_lines)
                kept_line_count += unit.line_count

        omitted_lines = max(0, total_body_lines_count - kept_line_count)

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
        context: str = "",
    ) -> str:
        """Compress a class by individually compressing each method.

        Preserves class-level attributes, type annotations, and decorators
        while compressing method bodies individually. This ensures correct
        indentation for each method's omitted-body comment.
        """
        # Use line-based extraction to preserve indentation
        code_lines = code.split("\n")
        start_row = node.start_point[0]
        node_lines = _get_node_lines(node, code_lines)
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
        if sig_end > 0:
            header_lines = node_lines[:sig_end]
            brace_line = node_lines[sig_end]
            if brace_line.strip().startswith("{"):
                header_lines = [*header_lines, brace_line]
        else:
            header_lines = [node_lines[0]]

        # Process each child of the class body individually
        body_parts: list[str] = []
        processed_ranges: list[tuple[int, int]] = []

        for child in body_node.children:
            if not child.is_named:
                continue

            # Use line-based extraction for children too
            child_start = child.start_point[0]
            child_end = child.end_point[0]
            if child.end_point[1] == 0 and child_end > child_start:
                child_end -= 1
            child_text = "\n".join(code_lines[child_start : child_end + 1])

            # Methods/functions inside the class — compress individually
            if child.type in lang_config.function_nodes:
                compressed = self._compress_function_ast(
                    child, code, language, lang_config, body_limits, analysis, context
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
                            deco_child,
                            code,
                            language,
                            lang_config,
                            body_limits,
                            analysis,
                            context,
                        )
                if decorator_lines and method_compressed:
                    body_parts.append("\n".join(decorator_lines) + "\n" + method_compressed)
                elif method_compressed:
                    body_parts.append(method_compressed)
                else:
                    body_parts.append(child_text)
                processed_ranges.append((child.start_byte, child.end_byte))
            # Nested classes / containers (e.g. nested namespaces) — recurse
            elif child.type in lang_config.class_nodes or (
                lang_config.container_node_types and child.type in lang_config.container_node_types
            ):
                compressed = self._compress_class_ast(
                    child, code, language, lang_config, body_limits, analysis, context
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

        # File header (license banners, top-of-file comments) stays on top
        if structure.header_code:
            parts.extend(structure.header_code)
            parts.append("")

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


def _iter_descendants(node: Any) -> Any:
    """Yield a node and all descendants depth-first."""
    yield node
    for child in getattr(node, "children", []):
        yield from _iter_descendants(child)


_STATEMENT_KIND_MAP: dict[str, _StatementKind] = {
    "return_statement": _StatementKind.RETURN,
    "raise_statement": _StatementKind.RAISE,
    "yield_statement": _StatementKind.YIELD,
    "yield_expression": _StatementKind.YIELD,
    "if_statement": _StatementKind.IF,
    "for_statement": _StatementKind.FOR,
    "for_in_statement": _StatementKind.FOR,
    "while_statement": _StatementKind.WHILE,
    "try_statement": _StatementKind.TRY,
    "with_statement": _StatementKind.WITH,
    "assignment": _StatementKind.ASSIGN,
    "assignment_expression": _StatementKind.ASSIGN,
    "augmented_assignment": _StatementKind.ASSIGN,
    "lexical_declaration": _StatementKind.ASSIGN,
    "variable_declaration": _StatementKind.ASSIGN,
    "expression_statement": _StatementKind.EXPR,
    "assert_statement": _StatementKind.ASSERT,
    "throw_statement": _StatementKind.RAISE,
}


def _slice_code_bytes(code: str, start_byte: int, end_byte: int) -> str:
    """Extract source text using tree-sitter UTF-8 byte offsets."""
    return code.encode("utf-8")[start_byte:end_byte].decode("utf-8")


def _get_node_text(node: Any, code: str) -> str:
    """Extract text from AST node."""
    return _slice_code_bytes(code, node.start_byte, node.end_byte)


_COMMENT_NODE_TYPES = frozenset({"comment", "line_comment", "block_comment"})


def _get_leading_comment_text(
    node: Any, code: str, captured_byte_ranges: list[tuple[int, int]]
) -> str:
    """Collect contiguous doc-comment siblings immediately preceding a node.

    Doc comments are top-level siblings of the declaration they document, not
    children of it. Left uncaptured, they fall through to the leftover
    top-level sweep and get grouped separately from the declarations they
    document instead of staying attached. Only comments with no blank line
    before the node (or the next comment) are treated as attached.
    """
    comments: list[Any] = []
    sibling = getattr(node, "prev_sibling", None)
    anchor_row = node.start_point[0]
    while (
        sibling is not None
        and sibling.type in _COMMENT_NODE_TYPES
        and (anchor_row - sibling.end_point[0] <= 1)
    ):
        comments.append(sibling)
        anchor_row = sibling.start_point[0]
        sibling = getattr(sibling, "prev_sibling", None)
    if not comments:
        return ""
    comments.reverse()
    captured_byte_ranges.extend((c.start_byte, c.end_byte) for c in comments)
    return "\n".join(_get_node_text(c, code) for c in comments) + "\n"


def _get_node_lines(node: Any, code_lines: list[str]) -> list[str]:
    """Line-based slice of a node's source, preserving original indentation.

    Line-based (not byte-offset) slicing is used deliberately so leading
    whitespace survives for indented nested definitions (e.g. methods inside
    a class). But when a node shares its first line with a preceding sibling
    (e.g. the `export` keyword in `export function foo() {`), a naive
    full-line slice pulls in that sibling's text too — and callers that
    reconstruct the wrapper (re-adding the sibling text themselves) end up
    duplicating it. Trim the sibling prefix from the first line when it's not
    pure whitespace; keep the whole line (indentation intact) otherwise.
    """
    start_row, start_col = node.start_point
    end_row = node.end_point[0]
    node_lines = list(code_lines[start_row : end_row + 1])
    if node_lines and node_lines[0][:start_col].strip():
        node_lines[0] = node_lines[0][start_col:]
    return node_lines


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


def _classify_statement_kind(node: Any, lang_config: LangConfig) -> _StatementKind:
    """Map a tree-sitter node to a lightweight statement kind."""
    node_type = getattr(node, "type", "")
    if node_type in lang_config.raise_nodes:
        return _StatementKind.RAISE
    if node_type in lang_config.yield_nodes:
        return _StatementKind.YIELD
    if node_type in lang_config.terminal_nodes:
        return _StatementKind.RETURN
    if node_type in lang_config.assignment_nodes:
        return _StatementKind.ASSIGN
    if node_type in lang_config.call_nodes:
        return _StatementKind.CALL
    return _STATEMENT_KIND_MAP.get(node_type, _StatementKind.UNKNOWN)


def _collect_statement_symbols(
    node: Any,
    lang_config: LangConfig,
) -> tuple[set[str], set[str], set[str]]:
    """Collect simple def/use/call signals for a statement node."""
    defined: set[str] = set()
    referenced: set[str] = set()
    called: set[str] = set()

    def record_text(n: Any) -> str | None:
        text = getattr(n, "text", None)
        if text is None:
            return None
        return text.decode("utf-8") if isinstance(text, bytes) else str(text)

    def add_identifiers(target: set[str], subtree: Any) -> None:
        for desc in _iter_descendants(subtree):
            if getattr(desc, "type", "") in lang_config.identifier_nodes:
                name = record_text(desc)
                if name:
                    target.add(name)

    # Definition heuristics: look at the left-hand side / declaration target only.
    assignment_nodes: list[Any] = []
    if getattr(node, "type", "") in lang_config.assignment_nodes:
        assignment_nodes.append(node)
    assignment_nodes.extend(
        child
        for child in getattr(node, "children", [])
        if getattr(child, "type", "") in lang_config.assignment_nodes
    )
    for assignment in assignment_nodes:
        assignment_children = getattr(assignment, "children", [])
        first_named = next((c for c in assignment_children if getattr(c, "is_named", False)), None)
        if first_named is not None:
            add_identifiers(defined, first_named)
        else:
            add_identifiers(defined, assignment)

    # Reference / call heuristics from descendants.
    for desc in _iter_descendants(node):
        node_type = getattr(desc, "type", "")
        if node_type in lang_config.identifier_nodes:
            name = record_text(desc)
            if name:
                referenced.add(name)

        if node_type in lang_config.call_nodes:
            for call_child in getattr(desc, "children", []):
                child_type = getattr(call_child, "type", "")
                if (
                    child_type
                    in lang_config.call_target_nodes
                    | lang_config.identifier_nodes
                    | lang_config.member_access_nodes
                ):
                    name = record_text(call_child)
                    if name:
                        called.add(name.split(".")[-1])
                    break

    referenced -= defined
    return defined, referenced, called


def _detect_guard_clause(node: Any, lang_config: LangConfig) -> bool:
    """Detect simple early-return / early-raise guard clauses from AST structure."""
    if getattr(node, "type", "") not in lang_config.guard_statement_nodes:
        return False

    named_children = [c for c in getattr(node, "children", []) if getattr(c, "is_named", False)]
    if len(named_children) < 2:
        return False

    consequence = named_children[1]
    if consequence is None:
        return False

    terminal_count = 0
    named_descendants = 0
    for desc in _iter_descendants(consequence):
        if not getattr(desc, "is_named", False):
            continue
        named_descendants += 1
        if getattr(desc, "type", "") in lang_config.terminal_nodes:
            terminal_count += 1

    has_else = any(
        getattr(child, "type", "") in lang_config.else_clause_nodes for child in named_children[2:]
    )
    return terminal_count > 0 and named_descendants <= 12 and not has_else


def _detect_side_effect(unit: _StatementUnit, lang_config: LangConfig) -> bool:
    """Detect likely side effects from AST structure rather than source-text keywords."""
    node_type = getattr(unit.node, "type", "")
    if node_type in lang_config.effect_boundary_nodes or unit.kind in {
        _StatementKind.RAISE,
        _StatementKind.TRY,
        _StatementKind.WITH,
    }:
        return True

    assignment_nodes = [
        desc
        for desc in _iter_descendants(unit.node)
        if getattr(desc, "type", "") in lang_config.assignment_nodes
    ]
    for assignment in assignment_nodes:
        lhs = getattr(assignment, "children", [])[:1]
        for child in lhs:
            if getattr(child, "type", "") in lang_config.mutation_receiver_nodes:
                return True

    if unit.kind in {_StatementKind.EXPR, _StatementKind.CALL} and unit.called_names:
        call_like = [
            desc
            for desc in _iter_descendants(unit.node)
            if getattr(desc, "type", "") in lang_config.call_nodes
        ]
        if call_like:
            return True

    return False


def _context_words(context: str) -> set[str]:
    return {w for w in re.split(r"[\s,;:.()\[\]{}\"']+", context.lower()) if len(w) > 2}


def _score_statement_unit(
    unit: _StatementUnit,
    analysis: _SymbolAnalysis,
    context_terms: set[str],
) -> float:
    """Compute a deterministic importance score for a statement."""
    score = 0.0

    if unit.has_return:
        score += 3.0
    if unit.has_raise:
        score += 2.5
    if unit.has_yield:
        score += 2.0
    if unit.is_guard_clause:
        score += 2.5
    if unit.has_side_effect:
        score += 1.8
    if unit.kind in {
        _StatementKind.IF,
        _StatementKind.FOR,
        _StatementKind.WHILE,
        _StatementKind.TRY,
    }:
        score += 0.8

    for called in unit.called_names:
        for qname, short in analysis.bare_names.items():
            if short == called:
                score += analysis.scores.get(qname, 0.0) * 1.5
                break

    if context_terms:
        symbol_terms = {
            name.lower()
            for name in (unit.defined_names | unit.referenced_names | unit.called_names)
        }
        hits = sum(1 for term in context_terms if term in symbol_terms)
        if hits:
            score += min(2.0, hits * 0.75)
            unit.context_hit = True

    score -= min(0.75, max(0, unit.line_count - 1) * 0.08)
    return round(score, 3)


def _nearest_definitions(
    selected: list[_StatementUnit], units: list[_StatementUnit]
) -> list[_StatementUnit]:
    """Add one backward definition layer for referenced local names."""
    selected_ids = {unit.idx for unit in selected}
    ordered = sorted(selected, key=lambda u: u.idx)
    for unit in list(ordered):
        needed = {name for name in unit.referenced_names if name and not name.startswith("__")}
        for name in needed:
            for candidate in reversed(units[: unit.idx]):
                if name in candidate.defined_names and candidate.idx not in selected_ids:
                    selected_ids.add(candidate.idx)
                    ordered.append(candidate)
                    break
    ordered.sort(key=lambda u: u.idx)
    return ordered


def _select_statement_units(units: list[_StatementUnit], body_limit: int) -> list[_StatementUnit]:
    """Select statements within a body budget using must-keep + scoring."""
    if not units:
        return []

    must_keep: list[_StatementUnit] = []
    must_keep.extend(unit for unit in units if unit.is_guard_clause)
    terminal_candidates = [u for u in units if u.has_return or u.has_raise or u.has_yield]
    if terminal_candidates:
        must_keep.append(terminal_candidates[-1])
    must_keep.extend(unit for unit in units if unit.context_hit)

    # Preserve likely side effects before low-value setup statements. This lets
    # calls/member mutations compete with the first statement instead of being
    # starved when a tiny budget is already consumed by first+terminal.
    side_effects = [unit for unit in units if unit.has_side_effect]
    if side_effects:
        side_effects.sort(key=lambda u: (-u.importance_score, u.idx))
        must_keep.append(side_effects[0])

    dedup: dict[int, _StatementUnit] = {u.idx: u for u in must_keep}
    selected = sorted(dedup.values(), key=lambda u: u.idx)
    selected = _nearest_definitions(selected, units)
    selected_ids = {u.idx for u in selected}
    used = sum(u.line_count for u in selected)

    candidates = [u for u in units if u.idx not in selected_ids]
    candidates.sort(key=lambda u: (-u.importance_score, u.idx))

    # Keep the first statement when it fits or when nothing stronger was selected,
    # but do not let a low-value first assignment crowd out terminal/guard/effect
    # statements under very small budgets.
    if units[0].idx not in selected_ids and (
        used + units[0].line_count <= body_limit or not selected
    ):
        selected.append(units[0])
        selected_ids.add(units[0].idx)
        used += units[0].line_count

    for unit in candidates:
        if unit.idx in selected_ids:
            continue
        if used + unit.line_count <= body_limit:
            selected.append(unit)
            selected_ids.add(unit.idx)
            used += unit.line_count

    if not selected:
        selected = [units[0]]
    elif len(selected) == 1 and selected[0].line_count > body_limit and body_limit > 0:
        selected = [selected[0]]

    # If no extra scored statements fit, still prefer a single high-value unit
    if len(selected) == 1 and body_limit > selected[0].line_count:
        for unit in candidates:
            selected.append(unit)
            break

    selected.sort(key=lambda u: u.idx)
    return selected


def _extract_statement_units(
    body_node: Any,
    code_lines: list[str],
    ds_end_row: int,
    context: str,
    analysis: _SymbolAnalysis,
    func_name: str | None,
    language: CodeLanguage,
    lang_config: LangConfig,
) -> list[_StatementUnit]:
    """Extract top-level function-body statements with lightweight semantic signals."""
    del func_name  # reserved for future use
    context_terms = _context_words(context)
    skip_types = frozenset({"{", "}", ";", ",", "comment", "line_comment", "block_comment"})
    units: list[_StatementUnit] = []

    children: list[Any] = []
    for child in getattr(body_node, "children", []):
        if getattr(child, "type", "") == "statement_list":
            children.extend(getattr(child, "children", []))
        else:
            children.append(child)

    for child in children:
        if child.start_point[0] <= ds_end_row:
            continue
        if child.type in skip_types or not getattr(child, "is_named", False):
            continue
        start_row = child.start_point[0]
        end_row = child.end_point[0]
        text_lines = code_lines[start_row : end_row + 1]
        kind = _classify_statement_kind(child, lang_config)
        defined, referenced, called = _collect_statement_symbols(child, lang_config)
        unit = _StatementUnit(
            idx=len(units),
            node=child,
            text_lines=text_lines,
            start_row=start_row,
            end_row=end_row,
            line_count=len(text_lines),
            kind=kind,
            defined_names=defined,
            referenced_names=referenced,
            called_names=called,
            has_return=kind == _StatementKind.RETURN,
            has_raise=kind == _StatementKind.RAISE,
            has_yield=kind == _StatementKind.YIELD,
        )
        unit.is_guard_clause = _detect_guard_clause(child, lang_config)
        unit.has_side_effect = _detect_side_effect(unit, lang_config)
        unit.importance_score = _score_statement_unit(unit, analysis, context_terms)
        units.append(unit)

    return units


def _get_definition_name(node: Any) -> str | None:
    """Extract the name identifier from a definition AST node."""
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier", "property_identifier"):
            text = child.text
            return text.decode("utf-8") if isinstance(text, bytes) else str(text)
    return None


# Symbol names are ASCII identifiers; CJK relevance queries have no spaces and use
# CJK/full-width punctuation, so the ASCII-only delimiter class would collapse the
# whole query into one blob and never isolate an ASCII name the user asked to keep.
_CONTEXT_DELIMS = re.compile(r"[\s,;:.()\[\]{}\"'，、；：。．！？（）【】「」『』《》〈〉·…—　]+")
_CJK_CHARS = re.compile(r"[　-鿿가-힯＀-￯]")


def _query_context_tokens(context: str) -> tuple[set[str], str, bool]:
    """Tokenize a relevance query for symbol-name matching (CJK-aware).

    Returns (word set, lowercased query, has_cjk). CJK/full-width punctuation and
    the ideographic space are delimiters so an ASCII symbol name wrapped in CJK is
    still isolated as its own token.
    """
    if not context:
        return set(), "", False
    lowered = context.lower()
    words = set(_CONTEXT_DELIMS.split(lowered))
    words.discard("")
    return words, lowered, bool(_CJK_CHARS.search(lowered))


def _symbol_in_context(name_lower: str, words: set[str], context_lower: str, has_cjk: bool) -> bool:
    """Whether the relevance query names this symbol.

    Exact token match, or a substring fallback gated by len>3 for ASCII queries
    (avoids spurious short-name matches) but relaxed for CJK queries -- a short
    ASCII name glued to CJK has no delimiter to isolate it, so exact-match can't
    fire and the guard would wrongly drop it.
    """
    if not words or not name_lower:
        return False
    if name_lower in words:
        return True
    return name_lower in context_lower and (len(name_lower) > 3 or has_cjk)


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
