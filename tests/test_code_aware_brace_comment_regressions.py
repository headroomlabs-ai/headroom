"""Regression tests for two CodeAwareCompressor AST-reassembly bugs found
while investigating a reported Go brace-duplication issue.

1. `export` keyword duplication (TS/JS): `_compress_function_ast` and
   `_compress_class_ast` use LINE-based slicing (not byte-offset) to
   preserve indentation for nested definitions. When a function/class shares
   its first line with a preceding sibling — e.g. the `export` keyword in
   `export function foo() {`, a sibling of the function inside
   `export_statement`, not part of the function node itself — a naive
   full-line slice pulled that sibling's text in too. The `export_statement`
   handler then re-prepended the same `export` text, producing
   `export export function foo() {` (invalid syntax, silently discarded by
   the `_verify_syntax` fallback).
2. Doc-comment displacement (all languages): doc comments are top-level
   siblings of the declaration they document, not children of it. Left
   unattached during AST extraction, they fell through to a "leftover
   top-level code" bucket that `_assemble_compressed` emits as one block
   after every function signature — detaching every doc comment from what it
   documents and dumping them all at the end of the file.
"""

from __future__ import annotations

import pytest

from headroom.transforms.code_compressor import (
    CodeAwareCompressor,
    CodeCompressorConfig,
    CodeLanguage,
    _check_tree_sitter_available,
)

pytestmark = pytest.mark.skipif(
    not _check_tree_sitter_available(),
    reason="tree-sitter not installed (pip install headroom-ai[code])",
)

TS_EXPORTED = """export interface User {
  id: string;
  name: string;
}

/**
 * Fetches a user by id.
 */
export function getUser(id: string): User {
  return { id, name: "test" };
}

/**
 * Greets a user by name.
 */
export function greet(user: User): string {
  return `Hello, ${user.name}!`;
}

export class UserStore {
  private users: User[] = [];

  add(user: User): void {
    this.users.push(user);
  }
}
"""

GO_DOC_COMMENTS = """package main

import "fmt"

// Add adds two integers together and returns the sum.
func Add(a, b int) int {
\treturn a + b
}

// Greet returns a friendly greeting for the given name.
func Greet(name string) string {
\treturn fmt.Sprintf("Hello, %s!", name)
}
"""


def _compress_ast(code: str, language: CodeLanguage):
    compressor = CodeAwareCompressor(CodeCompressorConfig())
    compressed, _, _ = compressor._compress_with_ast(code, language, "", None)
    return compressor, compressed


def test_ts_export_keyword_not_duplicated() -> None:
    """`export function`/`export class` must not become `export export ...`."""
    compressor, compressed = _compress_ast(TS_EXPORTED, CodeLanguage.TYPESCRIPT)

    assert "export export" not in compressed, compressed
    assert compressor._verify_syntax(compressed, CodeLanguage.TYPESCRIPT) is True


def test_ts_doc_comments_stay_attached_to_declaration() -> None:
    """A `/** ... */` doc comment must stay immediately before the function it
    documents, not get dumped in a cluster at the end of the output."""
    _, compressed = _compress_ast(TS_EXPORTED, CodeLanguage.TYPESCRIPT)

    assert "/**\n * Fetches a user by id.\n */\nexport function getUser" in compressed
    assert "/**\n * Greets a user by name.\n */\nexport function greet" in compressed


def test_go_doc_comments_stay_attached_to_function() -> None:
    """Same doc-comment-attachment bug, Go's `//` line-comment form."""
    compressor, compressed = _compress_ast(GO_DOC_COMMENTS, CodeLanguage.GO)

    assert "// Add adds two integers together and returns the sum.\nfunc Add" in compressed
    assert "// Greet returns a friendly greeting for the given name.\nfunc Greet" in compressed
    assert compressor._verify_syntax(compressed, CodeLanguage.GO) is True


def test_actual_typescript_compression() -> None:
    """Parity with the existing JS/Python/Go 'actual compression' tests —
    real TS input must actually compress, not silently no-op."""
    config = CodeCompressorConfig(min_tokens_for_compression=10, enable_ccr=False)
    compressor = CodeAwareCompressor(config)
    code = TS_EXPORTED * 3  # large enough to trigger body elision

    result = compressor.compress(code, language="typescript")

    assert result.compression_ratio < 1.0
    assert result.syntax_valid is True
    assert result.language == CodeLanguage.TYPESCRIPT
