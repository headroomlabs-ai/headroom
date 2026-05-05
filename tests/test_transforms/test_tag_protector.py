"""Tests for tag_protector: protect custom/workflow XML tags from compression."""

from __future__ import annotations

from headroom.transforms.tag_protector import (
    KNOWN_HTML_TAGS,
    _is_html_tag,
    protect_tags,
    restore_tags,
)


class TestKnownHTMLTags:
    def test_common_tags_are_html(self):
        for tag in ["div", "span", "p", "a", "table", "tr", "td", "h1", "body", "html"]:
            assert _is_html_tag(tag), f"{tag} should be recognized as HTML"

    def test_case_insensitive(self):
        assert _is_html_tag("DIV")
        assert _is_html_tag("Span")
        assert _is_html_tag("TABLE")

    def test_custom_tags_not_html(self):
        for tag in [
            "system-reminder",
            "tool_call",
            "thinking",
            "EXTREMELY_IMPORTANT",
            "context",
            "user-prompt-submit-hook",
            "result",
            "anthrpoic_thinking",
        ]:
            assert not _is_html_tag(tag), f"{tag} should NOT be recognized as HTML"

    def test_tag_set_not_empty(self):
        assert len(KNOWN_HTML_TAGS) > 100


class TestProtectTags:
    def test_no_tags_passthrough(self):
        text = "Just plain text with no tags at all."
        cleaned, protected = protect_tags(text)
        assert cleaned == text
        assert protected == []

    def test_no_angle_brackets_fast_path(self):
        text = "No angle brackets here"
        cleaned, protected = protect_tags(text)
        assert cleaned == text
        assert protected == []

    def test_html_tags_not_protected(self):
        text = "<div>Some content</div>"
        cleaned, protected = protect_tags(text)
        assert cleaned == text
        assert protected == []

    def test_custom_tag_protected(self):
        text = "Before <system-reminder>Important rule</system-reminder> After"
        cleaned, protected = protect_tags(text)
        assert "<system-reminder>" not in cleaned
        assert "Important rule" not in cleaned
        assert "Before" in cleaned
        assert "After" in cleaned
        assert len(protected) == 1
        assert protected[0][1] == "<system-reminder>Important rule</system-reminder>"

    def test_multiple_custom_tags(self):
        text = "<thinking>Step 1</thinking> middle <context>data</context>"
        cleaned, protected = protect_tags(text)
        assert len(protected) == 2
        assert "<thinking>" not in cleaned
        assert "<context>" not in cleaned

    def test_custom_tag_with_attributes(self):
        text = '<context key="session" type="persistent">user data</context>'
        cleaned, protected = protect_tags(text)
        assert len(protected) == 1
        assert 'key="session"' in protected[0][1]

    def test_self_closing_custom_tag(self):
        text = "Text <marker/> more text"
        cleaned, protected = protect_tags(text)
        assert len(protected) == 1
        assert protected[0][1] == "<marker/>"

    def test_self_closing_html_tag_not_protected(self):
        text = "Text <br/> more <hr/> text"
        cleaned, protected = protect_tags(text)
        assert cleaned == text
        assert protected == []

    def test_mixed_html_and_custom(self):
        text = "<div>HTML content</div> <system-reminder>Rule</system-reminder> <p>More HTML</p>"
        cleaned, protected = protect_tags(text)
        assert "<div>" in cleaned
        assert "<p>" in cleaned
        assert "<system-reminder>" not in cleaned
        assert len(protected) == 1

    def test_nested_custom_tags(self):
        text = "<outer><inner>deep content</inner></outer>"
        cleaned, protected = protect_tags(text)
        # Both should be protected (inner first, then outer)
        assert "<outer>" not in cleaned
        assert "<inner>" not in cleaned
        assert len(protected) >= 1

    def test_real_workflow_tags(self):
        """Tags actually used in LLM workflows."""
        tags = [
            "<tool_call>search({query: 'test'})</tool_call>",
            "<thinking>Let me analyze this step by step</thinking>",
            "<EXTREMELY_IMPORTANT>Never skip validation</EXTREMELY_IMPORTANT>",
            "<user-prompt-submit-hook>check permissions</user-prompt-submit-hook>",
            "<system-reminder>Follow these rules exactly</system-reminder>",
            "<result>Success: 42 items processed</result>",
        ]
        for tag_text in tags:
            text = f"Before {tag_text} After"
            cleaned, protected = protect_tags(text)
            assert len(protected) == 1, f"Failed to protect: {tag_text}"
            assert protected[0][1] == tag_text

    def test_empty_string(self):
        cleaned, protected = protect_tags("")
        assert cleaned == ""
        assert protected == []


class TestProtectTagsCompressContent:
    def test_compress_tagged_content_true(self):
        text = "Before <system-reminder>Compressible content here</system-reminder> After"
        cleaned, protected = protect_tags(text, compress_tagged_content=True)
        # Tags protected, but content between them is exposed for compression
        assert "<system-reminder>" not in cleaned
        assert "</system-reminder>" not in cleaned
        assert "Compressible content here" in cleaned
        assert len(protected) == 2  # Opening tag + closing tag

    def test_compress_tagged_content_false_default(self):
        text = "Before <system-reminder>Protected content</system-reminder> After"
        cleaned, protected = protect_tags(text)
        assert "Protected content" not in cleaned
        assert len(protected) == 1  # Entire block


class TestRestoreTags:
    def test_basic_restore(self):
        original = "Before <system-reminder>Rule</system-reminder> After"
        cleaned, protected = protect_tags(original)
        restored = restore_tags(cleaned, protected)
        assert "<system-reminder>Rule</system-reminder>" in restored
        assert "Before" in restored
        assert "After" in restored

    def test_restore_empty_protected(self):
        text = "No tags here"
        assert restore_tags(text, []) == text

    def test_restore_multiple(self):
        original = "<thinking>A</thinking> gap <context>B</context>"
        cleaned, protected = protect_tags(original)
        restored = restore_tags(cleaned, protected)
        assert "<thinking>A</thinking>" in restored
        assert "<context>B</context>" in restored

    def test_lost_placeholder_discards_wrap(self):
        """Hotfix-A9: when compression strips a placeholder, the wrap
        is DISCARDED — the compressed text is returned as-is and the
        original tag bytes are NOT re-injected anywhere. The original
        "append at the trailing edge" fallback produced silently
        malformed XML (orphan opening tag with no closing tag) on
        ~350 production requests over 9 days; that bug is gone."""
        protected = [("{{HEADROOM_TAG_0}}", "<tag>data</tag>")]
        compressed = "text without placeholder"
        result = restore_tags(compressed, protected)
        # Compressed text returned unchanged; original tag NOT injected.
        assert result == compressed
        assert "<tag>" not in result
        assert "</tag>" not in result
        assert "<tag>data</tag>" not in result

    def test_lost_placeholder_idempotent_when_all_missing(self):
        """Invariant: if every placeholder is missing from compressed,
        restore_tags returns compressed byte-for-byte unchanged."""
        protected = [
            ("{{HEADROOM_TAG_0}}", "<a>1</a>"),
            ("{{HEADROOM_TAG_1}}", "<b>2</b>"),
            ("{{HEADROOM_TAG_2}}", "<c>3</c>"),
        ]
        compressed = "compressor stripped every placeholder"
        assert restore_tags(compressed, protected) == compressed

    def test_partial_loss_keeps_present_discards_lost(self):
        """Mixed case: some placeholders survive, others are lost.
        Surviving ones get substituted; lost ones are discarded with
        zero orphan-tag injection."""
        protected = [
            ("{{HEADROOM_TAG_0}}", "<a>1</a>"),
            ("{{HEADROOM_TAG_1}}", "<lost>x</lost>"),
        ]
        result = restore_tags("head {{HEADROOM_TAG_0}} tail", protected)
        assert result == "head <a>1</a> tail"
        assert "<lost" not in result
        assert "</lost>" not in result

    def test_roundtrip_preserves_content(self):
        original = (
            "Start <system-reminder>Rule 1: always validate</system-reminder> "
            "middle <tool_call>search(q='test')</tool_call> end"
        )
        cleaned, protected = protect_tags(original)
        restored = restore_tags(cleaned, protected)
        assert "<system-reminder>Rule 1: always validate</system-reminder>" in restored
        assert "<tool_call>search(q='test')</tool_call>" in restored


class TestBugFixesPhase3e4:
    """Bug fixes baked into the Phase 3e.4 Rust port. Each test pins
    behavior the Python regex implementation got wrong."""

    def test_fixed_in_3e4_duplicate_blocks_get_distinct_placeholders(self):
        """Bug #2: Python's `result.replace(orig, ph, 1)` replaces the
        FIRST textual match of `orig`, not the matched offset. Two
        identical custom-tag blocks in the same input collapsed to a
        single placeholder + a stray duplicate of the second block.
        The Rust walker emits offset-based output, so distinct blocks
        always get distinct placeholders."""
        text = (
            "<system-reminder>same</system-reminder> middle <system-reminder>same</system-reminder>"
        )
        cleaned, protected = protect_tags(text)
        assert len(protected) == 2
        placeholders = {p[0] for p in protected}
        assert len(placeholders) == 2  # two DIFFERENT placeholders
        assert "<system-reminder>" not in cleaned
        # Roundtrip is exact byte-for-byte.
        assert restore_tags(cleaned, protected) == text

    def test_fixed_in_3e4_handles_60_nested_custom_tags(self):
        """Bug #3: Python had a hard `max_iterations = 50` safety cap
        that quietly stopped protecting deeper nested input. The Rust
        walker is bounded by input length only."""
        depth = 60
        text = "<lvl>" * depth + "core" + "</lvl>" * depth
        cleaned, protected = protect_tags(text)
        # Outermost span eats everything → ONE placeholder, no leaks.
        assert "<lvl>" not in cleaned
        assert "</lvl>" not in cleaned
        assert len(protected) == 1
        assert restore_tags(cleaned, protected) == text

    def test_fixed_in_3e4_self_closing_duplicates_distinct(self):
        """Bug #4: same first-occurrence-replace bug for self-closing
        tags. Two identical `<marker/>` would collapse to one
        placeholder + a stray dup."""
        text = "<marker/> middle <marker/>"
        cleaned, protected = protect_tags(text)
        assert len(protected) == 2
        assert protected[0][0] != protected[1][0]
        assert "<marker/>" not in cleaned
        assert restore_tags(cleaned, protected) == text

    def test_fixed_in_3e4_placeholder_collision_avoided(self):
        """Bug #5: input contains a literal `{{HEADROOM_TAG_…}}`
        substring. Python silently used the same prefix and let the
        collision break restoration. Rust salts the prefix when this
        happens."""
        text = (
            "User wrote {{HEADROOM_TAG_0}} on purpose. <system-reminder>real one</system-reminder>"
        )
        cleaned, protected = protect_tags(text)
        assert len(protected) == 1
        # Placeholder picked must NOT collide with the user's literal.
        assert protected[0][0] != "{{HEADROOM_TAG_0}}"
        # Roundtrip is exact (the user's literal stays intact).
        assert restore_tags(cleaned, protected) == text
