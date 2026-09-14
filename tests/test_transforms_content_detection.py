from __future__ import annotations

import json

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
    normalize_concatenated_json,
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

    # JSON OBJECTS are recognized too (config/data files are ``{…}``, not arrays).
    object_result = _try_detect_json('{"id": 1}')
    assert object_result is not None
    assert object_result.content_type is ContentType.JSON_ARRAY
    assert object_result.metadata == {"is_dict_array": False, "is_object": True}

    assert _try_detect_json("[not valid json") is None
    assert is_json_array_of_dicts('[{"id": 1}]') is True
    assert is_json_array_of_dicts('["value"]') is False


def test_space_separated_json_objects_detected_as_array() -> None:
    # Typical web_search output: back-to-back JSON objects, no array brackets.
    content = " ".join(
        json.dumps({"title": f"Result {i}", "url": f"http://example.com/{i}"}) for i in range(3)
    )
    result = _try_detect_json(content)
    assert result is not None
    assert result.content_type is ContentType.JSON_ARRAY
    assert result.confidence == 1.0
    assert result.metadata == {"item_count": 3, "is_dict_array": True, "concatenated": True}

    # Reaches the same verdict through the top-level detector (not PLAIN_TEXT).
    assert detect_content_type(content).content_type is ContentType.JSON_ARRAY
    assert is_json_array_of_dicts(content) is True

    # Newline separation is just as common and must also be recognized.
    newline_sep = "\n".join(json.dumps({"id": i, "snippet": "x"}) for i in range(2))
    assert _try_detect_json(newline_sep).content_type is ContentType.JSON_ARRAY


def test_json_detection_is_liberal_but_bulk_gated() -> None:
    # Liberal (parse-based): a lone JSON object IS structured data worth routing —
    # config/data files are ``{...}`` (chosen over the earlier conservative stance
    # when this PR merged with the concatenated-JSON detector, #1742).
    assert _try_detect_json('{"id": 1}').content_type is ContentType.JSON_ARRAY
    # But a JSON fragment that is only a minority of the content (prose or a loose
    # scalar around it) is NOT claimed — the decoded value must be the bulk.
    assert _try_detect_json('{"id": 1} then some prose {"id": 2}') is None
    assert _try_detect_json('{"id": 1} "loose string"') is None


def test_normalize_concatenated_json_roundtrips_to_array() -> None:
    content = '{"a": 1} {"b": 2}'
    normalized = normalize_concatenated_json(content)
    assert normalized is not None
    assert json.loads(normalized) == [{"a": 1}, {"b": 2}]

    # Already-valid arrays and single objects are left for the caller as-is.
    assert normalize_concatenated_json('[{"a": 1}]') is None
    assert normalize_concatenated_json('{"a": 1}') is None


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


def test_search_detection_rejects_datetime_prefixed_user_message() -> None:
    """Regression: wrap-copilot ate one-line interactive prompts (2026-08-23).

    Copilot CLI prepends ``<current_datetime>…</current_datetime>`` to every
    interactive user turn. The ISO-8601 ``T09:57:59`` matched the grep
    ``file:line:`` pattern, so a datetime + one-line prompt classified as
    SEARCH_RESULTS (1 match / 2 lines = 50% ≥ 30%) and the SearchCompressor
    deleted the prompt line — the model received only the timestamp.
    """
    incident = (
        "<current_datetime>2026-08-23T09:57:59.792+02:00</current_datetime>\n"
        "\n"
        "Please update the PR desc and check .overlay/ for hints."
    )
    assert _try_detect_search(incident) is None
    assert detect_content_type(incident).content_type is not ContentType.SEARCH_RESULTS


def test_search_detection_requires_two_matching_lines() -> None:
    """A single coincidental ``word:digits:`` line must not classify prose."""
    assert _try_detect_search("src/foo.py:12:def foo():") is None
    assert (
        _try_detect_search(
            "Meeting at 09:30:00 tomorrow.\nBring the reports.\nDo not forget coffee."
        )
        is None
    )
    # Two genuine grep lines still classify.
    two = "src/foo.py:12:def foo():\nsrc/bar.py:34:    foo()"
    result = _try_detect_search(two)
    assert result is not None
    assert result.content_type is ContentType.SEARCH_RESULTS


def test_search_detection_rejects_tag_like_and_key_value_prefixes() -> None:
    """Markup / key=value lines are not file paths even with ``:\\d+:`` inside."""
    assert (
        _try_detect_search('<log time="10:00:00">started</log>\n<log time="10:00:01">stopped</log>')
        is None
    )
    assert _try_detect_search("timeout=30:12:retried\ntimeout=31:12:retried") is None


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


_ISSUE_3580_SRC = [
    'env = payload.get("env")',
    "env_map = dict(env) if isinstance(env, dict) else {}",
    "previous = {name: env_map.get(name) for name in values}",
    'payload["env"] = env_map',
    'path.write_text(json.dumps(payload, indent=2), encoding="utf-8")',
]
_ISSUE_3580_FILE = "./headroom/providers/claude/install.py"


def _issue_3580_build(first_sep: str, second_sep: str) -> str:
    """70-line grep-shaped payload from the issue's reproducer (14 repeats)."""
    return "\n".join(
        f"{_ISSUE_3580_FILE}{first_sep}{40 + i}{second_sep}{line}"
        for _ in range(14)
        for i, line in enumerate(_ISSUE_3580_SRC)
    )


def test_search_detection_recognizes_grep_context_lines() -> None:
    """Regression (#3580): `-`-separated grep context lines are search output.

    GNU grep -A/-B/-C (and ripgrep / `git grep -A`) emit context lines as
    ``path-NN-content``. They must route to the lossless search fold, never
    to the word-dropping Kompress prose path.
    """
    payload = _issue_3580_build("-", "-")  # real `grep -A` emission shape
    result = _try_detect_search(payload)
    assert result is not None
    assert result.content_type is ContentType.SEARCH_RESULTS
    assert result.metadata == {"matching_lines": 70, "total_lines": 70}
    assert detect_content_type(payload).content_type is ContentType.SEARCH_RESULTS


def test_search_detection_rejects_colon_dash_mix() -> None:
    """The synthetic `path:NN-content` mix is NOT a real grep emission.

    Real grep uses `-` for both separators on context lines
    (``path-NN-content``) and `:` for both on match lines; the colon-dash mix
    only occurs in the issue's reproducer script. Accepting it would newly
    claim changelog/log shorthand such as `notes.txt:3-updated`, so it stays
    PLAIN_TEXT by design (solution review for #3580).
    """
    payload = _issue_3580_build(":", "-")
    assert _try_detect_search(payload) is None
    assert detect_content_type(payload).content_type is ContentType.PLAIN_TEXT


def test_search_detection_recognizes_dashed_file_names_in_context_lines() -> None:
    """Context lines keep working when the file name itself contains dashes."""
    payload = "\n".join(f"my-file.py-{40 + i}-    value = {i}" for i in range(8))
    result = _try_detect_search(payload)
    assert result is not None
    assert result.content_type is ContentType.SEARCH_RESULTS
    assert detect_content_type(payload).content_type is ContentType.SEARCH_RESULTS


def test_search_detection_mixed_match_and_context_lines() -> None:
    """Realistic `rg -A2` block: match lines, context lines, `--` separators.

    With match lines alone the ratio is 14/56 = 0.25 < 0.3, so the payload
    misdetected as PLAIN_TEXT before the fix; counting context lines gives
    42/56 = 0.75.
    """
    block: list[str] = []
    for rep in range(14):
        block.append(f"src/app.py:{10 + rep}:def handler_{rep}():")
        block.append(f"src/app.py-{11 + rep}-    first = compute()")
        block.append(f"src/app.py-{12 + rep}-    return first")
        block.append("--")
    payload = "\n".join(block)
    result = _try_detect_search(payload)
    assert result is not None
    assert result.content_type is ContentType.SEARCH_RESULTS
    assert result.metadata == {"matching_lines": 42, "total_lines": 56}
    assert detect_content_type(payload).content_type is ContentType.SEARCH_RESULTS


def test_search_detection_context_lines_require_path_shape() -> None:
    """The `-` branch must not claim dated logs or dashed prose (#3580).

    `2026-09-14` and `version-2-release` both fit `word-digits-dash`, but
    their prefixes are not file paths (no `/` or `.`), so they stay out.
    """
    dated = "\n".join(f"2026-09-14 worker heartbeat ok cycle {i}" for i in range(8))
    assert _try_detect_search(dated) is None
    assert detect_content_type(dated).content_type is not ContentType.SEARCH_RESULTS

    prose = "\n".join(
        [
            "version-2-release notes for the quarterly update",
            "well-42-known edge cases in the migration guide",
            "follow the step-1-setup instructions carefully",
            "see the self-9-reported issues in the tracker",
        ]
    )
    assert _try_detect_search(prose) is None

    # The `<`/`>`/`=` prefix exclusions apply to the `-` branch unchanged.
    key_value = "\n".join(f"timeout={30 + i}-12-retried" for i in range(4))
    assert _try_detect_search(key_value) is None
    markup = "\n".join(f'<log-{i}-started id="{i}">' for i in range(4))
    assert _try_detect_search(markup) is None


def test_search_detection_single_context_line_does_not_classify() -> None:
    """The two-line floor applies to context lines exactly as to matches."""
    assert _try_detect_search("src/main.py-42-def process():") is None
