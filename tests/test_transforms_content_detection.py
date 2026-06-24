from __future__ import annotations

from headroom.transforms.content_detector import (
    ContentType,
    _try_detect_code,
    _try_detect_diff,
    _try_detect_html,
    _try_detect_json,
    _try_detect_log,
    _try_detect_search,
    detect_content_type,
    is_json_array_of_dicts,
)
from headroom.transforms.error_detection import (
    ERROR_INDICATOR_KEYWORDS,
    ERROR_KEYWORDS,
    ERROR_PATTERN,
    IMPORTANCE_KEYWORDS,
    IMPORTANCE_PATTERN,
    PRIORITY_PATTERNS_DIFF,
    PRIORITY_PATTERNS_SEARCH,
    PRIORITY_PATTERNS_TEXT,
    SECURITY_KEYWORDS,
    SECURITY_PATTERN,
    WARNING_PATTERN,
    content_has_error_indicators,
)


def test_detect_content_type_handles_empty_and_plain_text() -> None:
    empty = detect_content_type("   ")
    assert empty.content_type is ContentType.PLAIN_TEXT
    assert empty.confidence == 0.0

    plain = detect_content_type("just a normal paragraph with no strong patterns")
    assert plain.content_type is ContentType.PLAIN_TEXT
    assert plain.confidence == 0.5


def test_json_detection_distinguishes_dict_arrays_and_other_lists() -> None:
    dict_result = _try_detect_json('[{"id": 1}, {"id": 2}]')
    assert dict_result is not None
    assert dict_result.content_type is ContentType.JSON_ARRAY
    assert dict_result.confidence == 1.0
    assert dict_result.metadata == {"item_count": 2, "is_dict_array": True}

    scalar_result = _try_detect_json("[1, 2, 3]")
    assert scalar_result is not None
    assert scalar_result.confidence == 0.8
    assert scalar_result.metadata == {"item_count": 3, "is_dict_array": False}

    empty_result = _try_detect_json("[]")
    assert empty_result is not None
    assert empty_result.metadata == {"item_count": 0, "is_dict_array": False}

    assert _try_detect_json('{"id": 1}') is None
    assert _try_detect_json("[not valid json") is None
    assert is_json_array_of_dicts('[{"id": 1}]') is True
    assert is_json_array_of_dicts('["value"]') is False


def test_diff_detection_tracks_headers_and_changes() -> None:
    diff = "\n".join(
        [
            "diff --git a/app.py b/app.py",
            "--- a/app.py",
            "@@ -1,2 +1,2 @@",
            "-old line",
            "+new line",
        ]
    )
    result = _try_detect_diff(diff)
    assert result is not None
    assert result.content_type is ContentType.GIT_DIFF
    assert result.metadata["header_matches"] == 3
    assert result.metadata["change_lines"] == 2
    assert result.confidence == 1.0
    assert detect_content_type(diff).content_type is ContentType.GIT_DIFF

    assert _try_detect_diff("+not enough by itself") is None


def test_html_detection_requires_real_structure() -> None:
    html = """
    <!DOCTYPE html>
    <html>
      <head><meta charset="utf-8"></head>
      <body><main><section><div>Hello</div><nav>Links</nav></section></main></body>
    </html>
    """
    result = _try_detect_html(html)
    assert result is not None
    assert result.content_type is ContentType.HTML
    assert result.metadata["has_doctype"] is True
    assert result.metadata["has_html_tag"] is True
    assert result.metadata["structural_tags"] >= 3
    assert detect_content_type(html).content_type is ContentType.HTML

    assert _try_detect_html("<div>only one tag</div>") is None
    assert _try_detect_html("<html><title>too sparse</title></html>") is None


def test_search_detection_uses_match_ratio() -> None:
    search_output = "\n".join(
        [
            "src/app.py:10:def main():",
            "src/app.py:20:print('hello')",
            "README.md:5:usage docs",
            "plain text footer",
        ]
    )
    result = _try_detect_search(search_output)
    assert result is not None
    assert result.content_type is ContentType.SEARCH_RESULTS
    assert result.metadata == {"matching_lines": 3, "total_lines": 4}
    assert result.confidence == 0.85
    assert detect_content_type(search_output).content_type is ContentType.SEARCH_RESULTS

    assert _try_detect_search("one:1:match\nplain\nplain\nplain") is None
    assert _try_detect_search("\n\n") is None


def test_log_detection_prefers_build_output_patterns() -> None:
    log_output = "\n".join(
        [
            "2025-01-01 Starting run",
            "ERROR failed to compile",
            "WARNING retrying build",
            "PASSED unit test",
            "Traceback (most recent call last)",
            "plain footer",
        ]
    )
    result = _try_detect_log(log_output)
    assert result is not None
    assert result.content_type is ContentType.BUILD_OUTPUT
    assert result.metadata == {"pattern_matches": 5, "error_matches": 2, "total_lines": 6}
    assert result.confidence == 0.8166666666666667
    assert detect_content_type(log_output).content_type is ContentType.BUILD_OUTPUT

    assert _try_detect_log("plain\ntext\nonly") is None
    assert _try_detect_log("\n\n") is None
    assert _try_detect_log("\n".join(["ERROR one", *["plain"] * 15])) is None


def test_code_detection_identifies_language_and_thresholds() -> None:
    python_code = "\n".join(
        [
            "import os",
            "from pathlib import Path",
            "",
            "@cached",
            "class App:",
            "    pass",
            "def main():",
            '    """Run."""',
        ]
    )
    result = _try_detect_code(python_code)
    assert result is not None
    assert result.content_type is ContentType.SOURCE_CODE
    assert result.metadata == {"language": "python", "pattern_matches": 6}
    assert result.confidence == 0.8628571428571429
    assert detect_content_type(python_code).content_type is ContentType.SOURCE_CODE

    assert _try_detect_code("function maybe() {}\nplain text") is None
    assert _try_detect_code("import os\ndef main():") is None
    assert _try_detect_code("\n\n") is None


def test_detect_content_type_respects_priority_order() -> None:
    diff_like_search = "\n".join(
        [
            "diff --git a/src/app.py b/src/app.py",
            "@@ -1,1 +1,1 @@",
            "+src/app.py:10:def still_diff_first()",
        ]
    )
    assert detect_content_type(diff_like_search).content_type is ContentType.GIT_DIFF


def test_error_detection_keywords_patterns_and_indicator_helper() -> None:
    assert {"error", "failed", "critical"} <= ERROR_KEYWORDS
    # fixed_in_3e1: ERROR_KEYWORDS canonically had timeout/abort/denied/rejected
    # but the regex omitted them; the Rust port + Python shim now align.
    assert {"timeout", "abort", "denied", "rejected"} <= ERROR_KEYWORDS
    assert {"warning", "todo", "fix"} <= IMPORTANCE_KEYWORDS
    # fixed_in_3e1: 'token' was dropped from SECURITY_KEYWORDS because it
    # false-positived on every LLM-token reference (input_tokens, etc.) in
    # an LLM-proxy product. 'auth' carries the real security signal.
    assert {"security", "password", "auth", "secret"} <= SECURITY_KEYWORDS
    assert "token" not in SECURITY_KEYWORDS
    assert ERROR_INDICATOR_KEYWORDS[0] == "error"

    assert ERROR_PATTERN.search("Fatal error occurred")
    # fixed_in_3e1: timeout now matched by ERROR_PATTERN regex.
    assert ERROR_PATTERN.search("Connection timeout occurred")
    assert WARNING_PATTERN.search("warning: be careful")
    assert IMPORTANCE_PATTERN.search("TODO fix this hack")
    # fixed_in_3e1: pre-3e1 this matched via 'token'; now matches via 'auth'.
    assert SECURITY_PATTERN.search("rotate the auth header")
    # fixed_in_3e1: lone 'token' references no longer trigger security routing.
    assert SECURITY_PATTERN.search("input_tokens=512 output_tokens=128") is None

    assert PRIORITY_PATTERNS_SEARCH[:3] == [ERROR_PATTERN, WARNING_PATTERN, IMPORTANCE_PATTERN]
    assert PRIORITY_PATTERNS_DIFF == [ERROR_PATTERN, IMPORTANCE_PATTERN, SECURITY_PATTERN]
    assert PRIORITY_PATTERNS_TEXT[0] is ERROR_PATTERN
    assert PRIORITY_PATTERNS_TEXT[1] is IMPORTANCE_PATTERN
    # The Rust-supplied markdown_prefixes order is `# `, `## `, `### `, `#### `,
    # `**`, `> ` (see KeywordRegistry::default_set). Index 2 is `# ` not `## `,
    # so anchor each assertion on the prefix it actually owns.
    assert PRIORITY_PATTERNS_TEXT[2].match("# Top-level heading")
    assert PRIORITY_PATTERNS_TEXT[3].match("## Subheading")
    assert PRIORITY_PATTERNS_TEXT[6].match("**Bold")
    assert PRIORITY_PATTERNS_TEXT[7].match("> quote")

    assert content_has_error_indicators("TRACEBACK: Fatal crash in worker") is True
    assert content_has_error_indicators("Everything completed successfully") is False
