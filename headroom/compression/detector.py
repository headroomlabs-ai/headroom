"""ML-based content type detection using Google's Magika.

Magika is a deep learning model for content type detection that:
- Runs locally (~5ms latency)
- Supports 100+ content types
- Has 99%+ accuracy on supported types
- Requires no configuration

This replaces rule-based detection with learned detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magika import Magika
    from magika.types import MagikaResult

logger = logging.getLogger(__name__)

# Lazy-loaded Magika instance (singleton)
_magika_instance: Magika | None = None


class ContentType(Enum):
    """High-level content categories for compression routing."""

    JSON = "json"
    CODE = "code"
    LOG = "log"
    DIFF = "diff"
    MARKDOWN = "markdown"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """Result of ML-based content detection."""

    content_type: ContentType
    confidence: float  # 0.0 to 1.0
    raw_label: str  # Original Magika label
    language: str | None = None  # For code: python, javascript, etc.
    metadata: dict = field(default_factory=dict)


# Map Magika labels to our content types
# This is the ONLY place where we map labels - no hardcoding elsewhere
_CODE_LABELS = frozenset(
    {
        "python",
        "javascript",
        "typescript",
        "go",
        "rust",
        "java",
        "c",
        "cpp",
        "csharp",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "scala",
        "shell",
        "bash",
        "powershell",
        "sql",
        "r",
        "perl",
        "lua",
        "haskell",
        "elixir",
        "erlang",
        "clojure",
        "ocaml",
        "fsharp",
        "dart",
        "julia",
        "zig",
        "nim",
        "crystal",
        "v",
        "solidity",
        "move",
        "cairo",
        "vyper",
    }
)

_STRUCTURED_LABELS = frozenset(
    {
        "json",
        "jsonl",
        "yaml",
        "toml",
        "xml",
        "html",
        "csv",
        "tsv",
        "ini",
        "properties",
    }
)

_LOG_LABELS = frozenset(
    {
        "log",
        "syslog",
    }
)

_MARKDOWN_LABELS = frozenset(
    {
        "markdown",
        "rst",
        "asciidoc",
        "org",
    }
)


def _get_magika() -> Magika:
    """Get or create the singleton Magika instance.

    Lazy-loads on first use to avoid import cost if not needed.
    """
    global _magika_instance
    if _magika_instance is None:
        try:
            from magika import Magika

            _magika_instance = Magika()
            logger.debug("Magika model loaded successfully")
        except ImportError as e:
            raise ImportError(
                "Magika is required for ML-based content detection. "
                "Install with: pip install magika"
            ) from e
    return _magika_instance


def _magika_available() -> bool:
    """Check if Magika is available without loading it."""
    try:
        import magika  # noqa: F401

        return True
    except ImportError:
        return False


class MagikaDetector:
    """ML-based content type detector using Google's Magika.

    This detector uses a deep learning model to identify content types
    without relying on file extensions or brittle regex patterns.

    Example:
        detector = MagikaDetector()
        result = detector.detect('def hello(): print("hi")')
        # result.content_type == ContentType.CODE
        # result.language == "python"
    """

    def __init__(self, min_confidence: float = 0.5):
        """Initialize the detector.

        Args:
            min_confidence: Minimum confidence threshold. Below this,
                returns ContentType.UNKNOWN.
        """
        self.min_confidence = min_confidence
        self._magika: Magika | None = None

    def _ensure_magika(self) -> Magika:
        """Ensure Magika is loaded."""
        if self._magika is None:
            self._magika = _get_magika()
        return self._magika

    def detect(self, content: str) -> DetectionResult:
        """Detect content type using ML.

        Args:
            content: The content to analyze.

        Returns:
            DetectionResult with type, confidence, and metadata.

        Example:
            >>> detector = MagikaDetector()
            >>> result = detector.detect('{"users": [{"id": 1}]}')
            >>> result.content_type
            ContentType.JSON
        """
        if not content or not content.strip():
            return DetectionResult(
                content_type=ContentType.UNKNOWN,
                confidence=0.0,
                raw_label="empty",
            )

        # Get Magika prediction
        magika = self._ensure_magika()
        result: MagikaResult = magika.identify_bytes(content.encode("utf-8"))

        raw_label = result.output.label
        confidence = result.score

        # Map to our content type
        content_type, language = self._map_label(raw_label)

        # Apply confidence threshold
        if confidence < self.min_confidence:
            content_type = ContentType.UNKNOWN

        return DetectionResult(
            content_type=content_type,
            confidence=confidence,
            raw_label=raw_label,
            language=language,
            metadata={
                "magika_group": result.output.group,
                "magika_mime": result.output.mime_type,
            },
        )

    def detect_batch(self, contents: list[str]) -> list[DetectionResult]:
        """Detect content types for multiple contents.

        Args:
            contents: List of content strings to analyze.

        Returns:
            List of DetectionResults in same order as input.
        """
        if not contents:
            return []

        results = []

        for content in contents:
            if not content or not content.strip():
                results.append(
                    DetectionResult(
                        content_type=ContentType.UNKNOWN,
                        confidence=0.0,
                        raw_label="empty",
                    )
                )
                continue

            magika_result = self._ensure_magika().identify_bytes(content.encode("utf-8"))
            raw_label = magika_result.output.label
            confidence = magika_result.score
            content_type, language = self._map_label(raw_label)

            if confidence < self.min_confidence:
                content_type = ContentType.UNKNOWN

            results.append(
                DetectionResult(
                    content_type=content_type,
                    confidence=confidence,
                    raw_label=raw_label,
                    language=language,
                    metadata={
                        "magika_group": magika_result.output.group,
                        "magika_mime": magika_result.output.mime_type,
                    },
                )
            )

        return results

    def _map_label(self, label: str) -> tuple[ContentType, str | None]:
        """Map Magika label to our ContentType.

        Args:
            label: Raw Magika label (e.g., "python", "json").

        Returns:
            Tuple of (ContentType, optional language).
        """
        label_lower = label.lower()

        # Check code languages
        if label_lower in _CODE_LABELS:
            return ContentType.CODE, label_lower

        # Check structured data
        if label_lower in _STRUCTURED_LABELS:
            # JSON gets its own type for specialized handling
            if label_lower in ("json", "jsonl"):
                return ContentType.JSON, None
            # Other structured data treated as JSON-like
            return ContentType.JSON, None

        # Check logs
        if label_lower in _LOG_LABELS:
            return ContentType.LOG, None

        # Check markdown/docs
        if label_lower in _MARKDOWN_LABELS:
            return ContentType.MARKDOWN, None

        # Diff format (Magika detects this with score=1.0)
        if label_lower == "diff":
            return ContentType.DIFF, None

        # Text types
        if label_lower in ("txt", "text", "ascii", "utf8", "empty"):
            return ContentType.TEXT, None

        # Default: treat as text
        return ContentType.TEXT, None

    @staticmethod
    def is_available() -> bool:
        """Check if Magika is available."""
        return _magika_available()


class FallbackDetector:
    """Simple fallback detector when Magika is not available.

    Uses basic heuristics - not as accurate but requires no dependencies.
    """

    def __init__(self, min_confidence: float = 0.5):
        """Initialize the fallback detector."""
        self.min_confidence = min_confidence

    def detect(self, content: str) -> DetectionResult:
        """Detect content type using simple heuristics.

        Args:
            content: The content to analyze.

        Returns:
            DetectionResult with type and confidence.
        """
        if not content or not content.strip():
            return DetectionResult(
                content_type=ContentType.UNKNOWN,
                confidence=0.0,
                raw_label="empty",
            )

        stripped = content.strip()

        # JSON detection (simple but effective)
        if stripped.startswith(("{", "[")):
            try:
                import json

                json.loads(stripped)
                return DetectionResult(
                    content_type=ContentType.JSON,
                    confidence=1.0,
                    raw_label="json",
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Code detection (look for common patterns)
        code_indicators = [
            "def ",
            "class ",
            "function ",
            "import ",
            "const ",
            "let ",
            "var ",
            "func ",
            "fn ",
            "pub ",
            "package ",
        ]
        if any(indicator in content for indicator in code_indicators):
            return DetectionResult(
                content_type=ContentType.CODE,
                confidence=0.7,
                raw_label="code",
            )

        # Log detection
        log_indicators = ["ERROR", "WARN", "INFO", "DEBUG", "FATAL"]
        if any(indicator in content for indicator in log_indicators):
            return DetectionResult(
                content_type=ContentType.LOG,
                confidence=0.6,
                raw_label="log",
            )

        # Default to text
        return DetectionResult(
            content_type=ContentType.TEXT,
            confidence=0.5,
            raw_label="text",
        )


def get_detector(prefer_magika: bool = True) -> MagikaDetector | FallbackDetector:
    """Get the best available detector.

    Args:
        prefer_magika: If True, use Magika if available.

    Returns:
        MagikaDetector if available and preferred, else FallbackDetector.
    """
    if prefer_magika and MagikaDetector.is_available():
        return MagikaDetector()
    return FallbackDetector()
