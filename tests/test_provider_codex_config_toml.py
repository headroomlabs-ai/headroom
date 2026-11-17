"""Tests for parsing Codex config.toml with the syntax Codex accepts."""

from __future__ import annotations

import sys

import pytest

from headroom.providers.codex.config_toml import (
    loads_codex_config,
    root_assignment_lines,
    split_toml_lines,
)

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

# Shape from #4378: an inline table split across lines with a trailing comma.
MULTILINE_INLINE_TABLE = (
    "[mcp_servers.dingtalk]\n"
    'command = "npx"\n'
    "env = {\n"
    '  DINGTALK_Client_ID = "**********",\n'
    "  \n"
    "    }\n"
)


def test_loads_codex_config_accepts_multiline_inline_table() -> None:
    assert loads_codex_config(MULTILINE_INLINE_TABLE) == {
        "mcp_servers": {"dingtalk": {"command": "npx", "env": {"DINGTALK_Client_ID": "**********"}}}
    }


def test_loads_codex_config_raises_tomllib_error_for_invalid_toml() -> None:
    with pytest.raises(tomllib.TOMLDecodeError):
        loads_codex_config("env = {\n")


def test_root_assignment_lines_skip_keys_nested_in_multiline_inline_table() -> None:
    content = (
        'model_provider = "openai"\n'  # 0: root
        "profiles = {\n"  # 1: root
        "  work = {\n"
        '    model_provider = "company",\n'
        "  },\n"
        "}\n"
        'notes = """\n'  # 6: root
        'model_provider = "in a string"\n'
        '"""\n'
        "# comment\n"
        'openai_base_url = "https://example.test/v1"  # {\n'  # 10: root
        "\n"
        "[profiles.other]\n"
        'model_provider = "other"\n'
    )
    assert root_assignment_lines(content) == {0, 1, 6, 10}


# str.splitlines() also breaks at these. TOML only ends a line at LF or CRLF,
# and allows U+2028, U+2029 and U+0085 inside a single-line string.
SPLITLINES_ONLY_SEPARATORS = [" ", " ", "\x85", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e"]


@pytest.mark.parametrize("sep", SPLITLINES_ONLY_SEPARATORS)
def test_root_assignment_lines_keep_single_line_string_whole(sep: str) -> None:
    content = f"notes = 'hello{sep}model_provider = \"company\"{sep}world'\n"
    assert root_assignment_lines(content) == {0}


def test_split_toml_lines_breaks_only_at_lf_and_crlf() -> None:
    content = "a = 'x y'\r\nb = 1\n\x85c = 2"
    assert split_toml_lines(content, keepends=True) == ["a = 'x y'\r\n", "b = 1\n", "\x85c = 2"]
    assert split_toml_lines(content) == ["a = 'x y'", "b = 1", "\x85c = 2"]
    assert split_toml_lines("a = 1\n") == ["a = 1"]
    assert split_toml_lines("") == []
