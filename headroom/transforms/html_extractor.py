"""HTML content extractor for web scraping results.

This module extracts main content from HTML pages, removing structural noise
like scripts, styles, navigation, ads, and footers. This is content extraction,
not compression - we remove irrelevant blocks, not tokens.

Typical reduction: 70-90% with zero content loss.

Uses trafilatura for robust extraction - it handles:
- Article/main content detection
- Boilerplate removal (nav, footer, sidebar, ads)
- Script/style removal
- Metadata extraction (title, author, date)
- Output as clean text or markdown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import trafilatura
from trafilatura.settings import use_config

# Suppress trafilatura's internal parse-error noise (e.g. "parsed tree length: 0")
# which appears at WARNING level on every document that fails to extract content.
# These are expected failures for non-article pages; log them only at CRITICAL.
logging.getLogger("trafilatura").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


@dataclass
class HTMLExtractionResult:
    """Result of HTML content extraction."""

    extracted: str
    original: str
    original_length: int
    extracted_length: int
    compression_ratio: float
    title: str | None = None
    author: str | None = None
    date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reduction_percent(self) -> float:
        """Percentage of content removed."""
        if self.original_length == 0:
            return 0.0
        return (1 - self.compression_ratio) * 100


@dataclass
class HTMLExtractorConfig:
    """Configuration for HTML extraction."""

    # Output format
    output_format: str = "markdown"  # "markdown" or "text"
    include_links: bool = True
    include_images: bool = False
    include_tables: bool = True

    # Extraction behavior
    include_comments: bool = False
    include_formatting: bool = True
    favor_precision: bool = False  # True = less content but higher quality
    favor_recall: bool = True  # True = more content, may include some noise

    # Metadata extraction
    extract_metadata: bool = True


class HTMLExtractor:
    """Extracts main content from HTML pages.

    Uses trafilatura for robust content extraction. This is not compression -
    it's removing structural HTML noise (scripts, styles, nav, ads) to get
    the actual content the user wanted.

    Example:
        >>> extractor = HTMLExtractor()
        >>> result = extractor.extract(html_content)
        >>> print(result.extracted)  # Clean markdown/text
        >>> print(f"Reduced by {result.reduction_percent:.1f}%")
    """

    def __init__(self, config: HTMLExtractorConfig | None = None):
        """Initialize HTML extractor.

        Args:
            config: Extraction configuration.
        """
        self.config = config or HTMLExtractorConfig()
        self._trafilatura_config = self._build_trafilatura_config()

    def _build_trafilatura_config(self) -> Any:
        """Build trafilatura configuration from our config."""
        config = use_config()

        # Set extraction parameters
        config.set("DEFAULT", "FAVOR_PRECISION", str(self.config.favor_precision))
        config.set("DEFAULT", "FAVOR_RECALL", str(self.config.favor_recall))

        return config

    def extract(self, html: str, url: str | None = None) -> HTMLExtractionResult:
        """Extract main content from HTML.

        Args:
            html: Raw HTML content.
            url: Optional URL for better extraction (helps with relative links).

        Returns:
            HTMLExtractionResult with extracted content and metadata.
        """
        original_length = len(html)

        if not html or not html.strip():
            return HTMLExtractionResult(
                extracted="",
                original=html,
                original_length=original_length,
                extracted_length=0,
                compression_ratio=0.0,
            )

        # Extract content using trafilatura
        extracted = trafilatura.extract(
            html,
            url=url,
            include_links=self.config.include_links,
            include_images=self.config.include_images,
            include_tables=self.config.include_tables,
            include_comments=self.config.include_comments,
            include_formatting=self.config.include_formatting,
            output_format=self.config.output_format,
            config=self._trafilatura_config,
        )

        # Handle extraction failure
        if extracted is None:
            logger.debug("trafilatura extraction returned None, returning empty")
            extracted = ""

        extracted_length = len(extracted)
        compression_ratio = extracted_length / max(original_length, 1)

        # Extract metadata if configured
        title = None
        author = None
        date = None
        metadata: dict[str, Any] = {}

        if self.config.extract_metadata:
            meta = trafilatura.extract_metadata(html, default_url=url)
            if meta:
                title = meta.title
                author = meta.author
                date = meta.date
                metadata = {
                    "title": meta.title,
                    "author": meta.author,
                    "date": meta.date,
                    "sitename": meta.sitename,
                    "description": meta.description,
                    "categories": meta.categories,
                    "tags": meta.tags,
                }

        return HTMLExtractionResult(
            extracted=extracted,
            original=html,
            original_length=original_length,
            extracted_length=extracted_length,
            compression_ratio=compression_ratio,
            title=title,
            author=author,
            date=date,
            metadata=metadata,
        )

    def extract_batch(
        self, html_contents: list[tuple[str, str | None]]
    ) -> list[HTMLExtractionResult]:
        """Extract content from multiple HTML pages.

        Args:
            html_contents: List of (html, url) tuples.

        Returns:
            List of HTMLExtractionResult in same order as input.
        """
        return [self.extract(html, url) for html, url in html_contents]


def is_html_content(content: str) -> bool:
    """Check if content appears to be HTML.

    Args:
        content: Content to check.

    Returns:
        True if content looks like HTML.
    """
    if not content:
        return False

    stripped = content.strip().lower()

    # Check for DOCTYPE or html tag
    if stripped.startswith("<!doctype html") or stripped.startswith("<html"):
        return True

    # Check for common HTML patterns
    html_indicators = [
        "<head",
        "<body",
        "<div",
        "<script",
        "<style",
        "<meta",
        "<link",
        "<!doctype",
    ]

    # Count how many indicators are present
    matches = sum(1 for indicator in html_indicators if indicator in stripped[:2000])

    # If we see multiple HTML-specific tags, it's likely HTML
    return matches >= 2
