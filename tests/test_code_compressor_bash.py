"""Regression tests for lossless Bash handling in CodeAwareCompressor."""

from headroom.transforms.code_compressor import (
    CodeAwareCompressor,
    CodeCompressorConfig,
    CodeLanguage,
    coerce_language,
    detect_language,
)

BASH_SNIPPET = """#!/usr/bin/env bash
set -euo pipefail

if [ -x "$(command -v /opt/bin/forgejo)" ]
then
\texport GITEA_WORK_DIR=/git/forgejo
\texport FORGEJO_CUSTOM=/etc/forgejo
\talias forge="sudo -Eu git GITEA_CUSTOM=${FORGEJO_CUSTOM} GITEA_WORK_DIR=${GITEA_WORK_DIR} /opt/bin/forgejo"
fi

if [[ "$( git config --global alias.pushall 2>/dev/null )" != '!git remote | xargs -L1 git push --all' ]]
then
\tgit config --global --unset-all alias.pushall
git config --global --add alias.pushall '!git remote | xargs -L1 git push --all'
fi
"""


def test_bash_aliases_are_canonicalized():
    assert coerce_language("bash") is CodeLanguage.BASH
    assert coerce_language("shell") is CodeLanguage.BASH
    assert coerce_language("sh") is CodeLanguage.BASH
    assert coerce_language("zsh") is CodeLanguage.BASH


def test_bash_is_detected_from_control_flow():
    detected, confidence = detect_language(BASH_SNIPPET.removeprefix("#!/usr/bin/env bash\n"))
    assert detected is CodeLanguage.BASH
    assert confidence > 0


def test_bash_is_detected_from_shebang():
    detected, confidence = detect_language("#!/bin/sh\necho hello\n")
    assert detected is CodeLanguage.BASH
    assert confidence > 0


def test_bash_compression_is_lossless_and_does_not_use_kompress():
    compressor = CodeAwareCompressor(
        CodeCompressorConfig(min_tokens_for_compression=1, enable_ccr=False)
    )
    result = compressor.compress(BASH_SNIPPET)

    assert result.language is CodeLanguage.BASH
    assert result.syntax_valid is True
    assert result.compressed == BASH_SNIPPET
    assert result.compressed_tokens == result.original_tokens
    assert result.compression_ratio == 1.0


def test_shell_hint_is_lossless():
    compressor = CodeAwareCompressor(
        CodeCompressorConfig(min_tokens_for_compression=1, enable_ccr=False)
    )
    result = compressor.compress(BASH_SNIPPET, language="shell")

    assert result.language is CodeLanguage.BASH
    assert result.compressed == BASH_SNIPPET
    assert result.syntax_valid is True
