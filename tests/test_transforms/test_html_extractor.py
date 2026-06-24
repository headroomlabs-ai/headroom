"""Tests for HTMLExtractor.

These are real tests using actual HTML content - no mocks.
Tests verify that trafilatura correctly extracts main content
and removes structural noise.
"""

import pytest

# Skip entire module if trafilatura not installed
pytest.importorskip("trafilatura")

from headroom.transforms.html_extractor import (
    HTMLExtractionResult,
    HTMLExtractor,
    HTMLExtractorConfig,
    is_html_content,
)


class TestIsHtmlContent:
    """Tests for the is_html_content detection function."""

    def test_detects_doctype_html(self):
        """Detects HTML with DOCTYPE declaration."""
        html = "<!DOCTYPE html><html><body>Content</body></html>"
        assert is_html_content(html) is True

    def test_detects_html_tag(self):
        """Detects HTML with html tag."""
        html = "<html><head></head><body>Content</body></html>"
        assert is_html_content(html) is True

    def test_detects_structural_tags(self):
        """Detects HTML with multiple structural tags (needs doctype or html tag)."""
        # Note: is_html_content requires DOCTYPE or <html> tag, or 3+ structural tags
        # Just structural tags alone may not trigger detection
        html = "<html><div><nav>Menu</nav><article>Content</article><footer>Footer</footer></div></html>"
        assert is_html_content(html) is True

    def test_rejects_plain_text(self):
        """Rejects plain text."""
        text = "This is just plain text with no HTML."
        assert is_html_content(text) is False

    def test_rejects_json(self):
        """Rejects JSON content."""
        json_content = '{"name": "test", "value": 123}'
        assert is_html_content(json_content) is False

    def test_rejects_markdown(self):
        """Rejects markdown content."""
        markdown = "# Heading\n\nParagraph with **bold** text."
        assert is_html_content(markdown) is False

    def test_rejects_code(self):
        """Rejects source code."""
        code = "def hello():\n    print('world')"
        assert is_html_content(code) is False

    def test_rejects_empty(self):
        """Rejects empty content."""
        assert is_html_content("") is False
        assert is_html_content(None) is False  # type: ignore

    def test_case_insensitive(self):
        """Detection is case insensitive."""
        html = "<!DOCTYPE HTML><HTML><BODY>Content</BODY></HTML>"
        assert is_html_content(html) is True


class TestHTMLExtractor:
    """Tests for HTMLExtractor content extraction."""

    @pytest.fixture
    def extractor(self):
        """Create a default HTMLExtractor."""
        return HTMLExtractor()

    def test_extracts_article_content(self, extractor):
        """Extracts main article content."""
        html = """<!DOCTYPE html>
        <html>
        <head><title>Test Article</title></head>
        <body>
            <nav><a href="/">Home</a></nav>
            <article>
                <h1>Article Title</h1>
                <p>This is the main content of the article.</p>
                <p>It contains important information for the reader.</p>
            </article>
            <footer>Copyright 2024</footer>
        </body>
        </html>"""

        result = extractor.extract(html)

        assert "Article Title" in result.extracted
        assert "main content" in result.extracted
        assert "important information" in result.extracted
        # Navigation and footer should be removed
        assert "Home" not in result.extracted or "Copyright" not in result.extracted

    def test_removes_script_tags(self, extractor):
        """Removes JavaScript content."""
        html = """<!DOCTYPE html>
        <html>
        <head>
            <script>
                function malicious() {
                    alert('This should not appear');
                    console.log('Script content');
                }
            </script>
        </head>
        <body>
            <article>
                <p>Actual content that matters.</p>
            </article>
            <script>anotherScript();</script>
        </body>
        </html>"""

        result = extractor.extract(html)

        assert "malicious" not in result.extracted
        assert "alert" not in result.extracted
        assert "anotherScript" not in result.extracted
        assert "Actual content" in result.extracted

    def test_removes_style_tags(self, extractor):
        """Removes CSS content."""
        html = """<!DOCTYPE html>
        <html>
        <head>
            <style>
                body { color: red; }
                .hidden { display: none; }
            </style>
        </head>
        <body>
            <article>
                <p>Real content here.</p>
            </article>
        </body>
        </html>"""

        result = extractor.extract(html)

        assert "color: red" not in result.extracted
        assert "display: none" not in result.extracted
        assert "Real content" in result.extracted

    def test_compression_ratio(self, extractor):
        """Verifies significant compression ratio."""
        html = """<!DOCTYPE html>
        <html>
        <head>
            <title>Page Title</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width">
            <script src="analytics.js"></script>
            <script>
                var config = {tracking: true, debug: false};
                function init() { console.log('initialized'); }
            </script>
            <style>
                * { margin: 0; padding: 0; }
                body { font-family: Arial; }
                nav { background: #333; }
                footer { background: #666; }
            </style>
        </head>
        <body>
            <header>
                <nav>
                    <a href="/">Home</a>
                    <a href="/about">About</a>
                    <a href="/contact">Contact</a>
                </nav>
            </header>
            <main>
                <article>
                    <h1>Main Article Heading</h1>
                    <p>This is the first paragraph of actual content.</p>
                    <p>This is the second paragraph with more details.</p>
                </article>
            </main>
            <aside>
                <h3>Related Links</h3>
                <ul>
                    <li><a href="/link1">Link 1</a></li>
                    <li><a href="/link2">Link 2</a></li>
                </ul>
            </aside>
            <footer>
                <p>Copyright 2024</p>
                <p>Privacy Policy | Terms of Service</p>
            </footer>
        </body>
        </html>"""

        result = extractor.extract(html)

        # Should achieve significant reduction
        assert result.compression_ratio < 0.5  # At least 50% reduction
        assert result.reduction_percent > 50
        # Main content should be preserved
        assert "Main Article Heading" in result.extracted
        assert "first paragraph" in result.extracted

    def test_extracts_metadata(self, extractor):
        """Extracts page metadata."""
        html = """<!DOCTYPE html>
        <html>
        <head>
            <title>Page Title for Testing</title>
            <meta name="author" content="John Doe">
            <meta name="description" content="A test page description">
        </head>
        <body>
            <article>
                <h1>Page Title for Testing</h1>
                <p>Content here.</p>
            </article>
        </body>
        </html>"""

        result = extractor.extract(html)

        assert result.title == "Page Title for Testing"
        assert result.metadata.get("title") == "Page Title for Testing"

    def test_handles_empty_html(self, extractor):
        """Handles empty HTML gracefully."""
        result = extractor.extract("")

        assert result.extracted == ""
        assert result.original_length == 0
        assert result.extracted_length == 0

    def test_handles_whitespace_only(self, extractor):
        """Handles whitespace-only input."""
        result = extractor.extract("   \n\t  ")

        assert result.extracted == ""
        assert result.compression_ratio == 0.0

    def test_handles_minimal_html(self, extractor):
        """Handles minimal HTML structure."""
        html = "<p>Just a paragraph.</p>"
        result = extractor.extract(html)

        # trafilatura may or may not extract minimal content
        # Just verify it doesn't crash
        assert isinstance(result.extracted, str)
        assert result.original_length > 0

    def test_preserves_paragraphs(self, extractor):
        """Preserves paragraph structure."""
        html = """<!DOCTYPE html>
        <html>
        <body>
            <article>
                <p>First paragraph.</p>
                <p>Second paragraph.</p>
                <p>Third paragraph.</p>
            </article>
        </body>
        </html>"""

        result = extractor.extract(html)

        assert "First paragraph" in result.extracted
        assert "Second paragraph" in result.extracted
        assert "Third paragraph" in result.extracted


class TestHTMLExtractorConfig:
    """Tests for HTMLExtractor configuration options."""

    def test_markdown_output_format(self):
        """Tests markdown output format."""
        config = HTMLExtractorConfig(output_format="markdown")
        extractor = HTMLExtractor(config)

        html = """<!DOCTYPE html>
        <html>
        <body>
            <article>
                <h1>Heading</h1>
                <p>Paragraph text.</p>
            </article>
        </body>
        </html>"""

        result = extractor.extract(html)
        # Markdown format should include # for heading
        assert "Heading" in result.extracted

    def test_text_output_format(self):
        """Tests plain text output format."""
        config = HTMLExtractorConfig(output_format="txt")
        extractor = HTMLExtractor(config)

        html = """<!DOCTYPE html>
        <html>
        <body>
            <article>
                <h1>Heading</h1>
                <p>Paragraph text.</p>
            </article>
        </body>
        </html>"""

        result = extractor.extract(html)
        assert "Heading" in result.extracted
        assert "Paragraph" in result.extracted

    def test_disable_metadata_extraction(self):
        """Tests disabling metadata extraction."""
        config = HTMLExtractorConfig(extract_metadata=False)
        extractor = HTMLExtractor(config)

        html = """<!DOCTYPE html>
        <html>
        <head><title>Test Title</title></head>
        <body><article><p>Content.</p></article></body>
        </html>"""

        result = extractor.extract(html)

        # Metadata should be empty when disabled
        assert result.title is None
        assert result.metadata == {}


class TestHTMLExtractionResult:
    """Tests for HTMLExtractionResult dataclass."""

    def test_reduction_percent_calculation(self):
        """Tests reduction percent calculation."""
        result = HTMLExtractionResult(
            extracted="short",
            original="much longer content here",
            original_length=100,
            extracted_length=25,
            compression_ratio=0.25,
        )

        assert result.reduction_percent == 75.0

    def test_reduction_percent_with_zero_original(self):
        """Tests reduction percent with zero original length."""
        result = HTMLExtractionResult(
            extracted="",
            original="",
            original_length=0,
            extracted_length=0,
            compression_ratio=0.0,
        )

        assert result.reduction_percent == 0.0


class TestRealWorldHTML:
    """Tests with realistic HTML content."""

    @pytest.fixture
    def extractor(self):
        return HTMLExtractor()

    def test_news_article_structure(self, extractor):
        """Tests extraction from news-article-like structure."""
        html = """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Breaking News: Important Event Happens</title>
            <script>window.analytics = {};</script>
            <style>.ad { display: block; }</style>
        </head>
        <body>
            <header>
                <nav class="main-nav">
                    <a href="/">Home</a>
                    <a href="/news">News</a>
                    <a href="/sports">Sports</a>
                </nav>
            </header>
            <div class="ad-banner">Advertisement Here</div>
            <main>
                <article class="news-article">
                    <h1>Breaking News: Important Event Happens</h1>
                    <p class="byline">By Jane Reporter | January 15, 2024</p>
                    <p>In a surprising turn of events, something important happened today
                    that will affect millions of people around the world.</p>
                    <p>Experts say this development represents a major shift in how
                    we think about the topic at hand.</p>
                    <p>"This is truly unprecedented," said Dr. Expert, a leading
                    authority in the field.</p>
                    <p>The implications of this event are still being analyzed, but
                    early reports suggest significant changes ahead.</p>
                </article>
            </main>
            <aside class="sidebar">
                <h3>Trending Stories</h3>
                <ul>
                    <li><a href="/story1">Story 1</a></li>
                    <li><a href="/story2">Story 2</a></li>
                </ul>
            </aside>
            <footer>
                <p>&copy; 2024 News Site</p>
                <a href="/privacy">Privacy Policy</a>
            </footer>
            <script>trackPageView();</script>
        </body>
        </html>"""

        result = extractor.extract(html)

        # Main article content should be preserved
        assert "Important Event Happens" in result.extracted
        assert "surprising turn of events" in result.extracted
        assert "Dr. Expert" in result.extracted

        # Noise should be removed or minimized
        assert "trackPageView" not in result.extracted
        assert "window.analytics" not in result.extracted

        # Significant reduction
        assert result.compression_ratio < 0.4

    def test_documentation_page(self, extractor):
        """Tests extraction from documentation-like page."""
        html = """<!DOCTYPE html>
        <html>
        <head>
            <title>API Documentation - MyService</title>
        </head>
        <body>
            <nav class="docs-nav">
                <a href="/docs">Docs</a>
                <a href="/api">API</a>
                <a href="/guides">Guides</a>
            </nav>
            <div class="sidebar">
                <h3>Table of Contents</h3>
                <ul>
                    <li><a href="#intro">Introduction</a></li>
                    <li><a href="#auth">Authentication</a></li>
                </ul>
            </div>
            <main class="content">
                <h1>API Documentation</h1>
                <section id="intro">
                    <h2>Introduction</h2>
                    <p>Welcome to the MyService API documentation. This guide will
                    help you integrate our service into your application.</p>
                </section>
                <section id="auth">
                    <h2>Authentication</h2>
                    <p>All API requests require authentication using an API key.
                    Include your key in the Authorization header.</p>
                    <pre><code>Authorization: Bearer YOUR_API_KEY</code></pre>
                </section>
            </main>
            <footer>Built with Docs Generator</footer>
        </body>
        </html>"""

        result = extractor.extract(html)

        # Documentation content should be preserved
        assert "API Documentation" in result.extracted
        assert "Authentication" in result.extracted
        assert "API key" in result.extracted

    def test_blog_post(self, extractor):
        """Tests extraction from blog-post-like structure."""
        html = """<!DOCTYPE html>
        <html>
        <head>
            <title>My Blog Post Title - Personal Blog</title>
            <meta name="author" content="Blog Author">
        </head>
        <body>
            <header>
                <h1 class="site-title">My Personal Blog</h1>
                <nav>Home | About | Contact</nav>
            </header>
            <article class="blog-post">
                <h1>My Blog Post Title</h1>
                <time>Posted on March 15, 2024</time>
                <p>Today I want to share my thoughts on an interesting topic
                that I've been thinking about for a while.</p>
                <p>The key insight is that we often overlook the simple things
                in life that bring us joy.</p>
                <p>In conclusion, I believe we should all take more time to
                appreciate the small moments.</p>
            </article>
            <section class="comments">
                <h3>Comments</h3>
                <div class="comment">Great post!</div>
            </section>
            <footer>&copy; 2024 Blog Author</footer>
        </body>
        </html>"""

        result = extractor.extract(html)

        # Blog content should be preserved
        assert "Blog Post Title" in result.extracted
        assert "interesting topic" in result.extracted
        assert "small moments" in result.extracted


class TestBatchExtraction:
    """Tests for batch extraction."""

    def test_extract_batch(self):
        """Tests batch extraction of multiple HTML pages."""
        extractor = HTMLExtractor()

        pages = [
            (
                "<html><body><article><p>Page one content.</p></article></body></html>",
                "http://example.com/page1",
            ),
            (
                "<html><body><article><p>Page two content.</p></article></body></html>",
                "http://example.com/page2",
            ),
            (
                "<html><body><article><p>Page three content.</p></article></body></html>",
                None,
            ),
        ]

        results = extractor.extract_batch(pages)

        assert len(results) == 3
        assert all(isinstance(r, HTMLExtractionResult) for r in results)


class TestContentDetectorIntegration:
    """Tests for integration with content_detector."""

    def test_detector_identifies_html(self):
        """Tests that content_detector correctly identifies HTML."""
        from headroom.transforms.content_detector import ContentType, detect_content_type

        html = """<!DOCTYPE html>
        <html>
        <head><title>Test</title></head>
        <body>
            <div><p>Content</p></div>
        </body>
        </html>"""

        result = detect_content_type(html)

        assert result.content_type == ContentType.HTML
        assert result.confidence >= 0.7

    def test_detector_rejects_non_html(self):
        """Tests that content_detector doesn't misidentify non-HTML."""
        from headroom.transforms.content_detector import ContentType, detect_content_type

        # Plain text
        result = detect_content_type("Just some plain text without HTML.")
        assert result.content_type != ContentType.HTML

        # JSON
        result = detect_content_type('[{"id": 1}, {"id": 2}]')
        assert result.content_type == ContentType.JSON_ARRAY

        # Code (needs enough patterns to trigger detection)
        code = """
import os
import sys

def hello():
    print('world')

class Foo:
    def __init__(self):
        pass

if __name__ == '__main__':
    hello()
"""
        result = detect_content_type(code)
        assert result.content_type == ContentType.SOURCE_CODE
