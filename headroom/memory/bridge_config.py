"""Configuration for the Memory Bridge.

The Memory Bridge provides bidirectional sync between markdown memory files
(used by Claude Code, ChatGPT, etc.) and Headroom's semantic memory system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from headroom import paths as _paths


class MarkdownFormat(Enum):
    """Supported markdown memory formats."""

    CLAUDE_CODE = "claude_code"  # ~/.claude/projects/<project>/memory/MEMORY.md
    CHATGPT = "chatgpt"  # Flat fact list, one per line
    GENERIC = "generic"  # Structured markdown with headers
    AUTO = "auto"  # Auto-detect from content


@dataclass
class BridgeConfig:
    """Configuration for the Memory Bridge.

    Attributes:
        md_paths: List of markdown file paths to bridge.
        md_format: Format of the markdown files (auto-detected if AUTO).
        user_id: Default user_id for imported memories.
        default_importance: Base importance for imported memories.
        heading_importance_map: Map heading depth to importance score.
        sync_state_path: Path to store sync state (hashes, timestamps).
        auto_import_on_startup: Whether to import on proxy startup.
        export_path: Where to write exported markdown.
        export_format: Format for exported markdown.
        extract_entities: Whether to extract entities during import.
        chunk_by_section: Split markdown by section headers for granular memories.
        dedup_similarity_threshold: Similarity above which a memory is a duplicate.
        source_tag: Metadata tag added to all bridged memories for tracking.
    """

    md_paths: list[Path] = field(default_factory=list)
    md_format: MarkdownFormat = MarkdownFormat.AUTO
    user_id: str = "default"
    default_importance: float = 0.6
    heading_importance_map: dict[int, float] = field(
        default_factory=lambda: {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5, 6: 0.4}
    )
    sync_state_path: Path = field(default_factory=_paths.bridge_state_path)
    auto_import_on_startup: bool = False
    export_path: Path | None = None
    export_format: MarkdownFormat = MarkdownFormat.GENERIC
    extract_entities: bool = True
    chunk_by_section: bool = True
    dedup_similarity_threshold: float = 0.92
    source_tag: str = "memory_bridge"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.default_importance <= 1.0:
            raise ValueError(f"default_importance must be 0.0-1.0, got {self.default_importance}")
        if not 0.0 <= self.dedup_similarity_threshold <= 1.0:
            raise ValueError(
                f"dedup_similarity_threshold must be 0.0-1.0, got {self.dedup_similarity_threshold}"
            )
        self.md_paths = [Path(p) if isinstance(p, str) else p for p in self.md_paths]
        if isinstance(self.sync_state_path, str):
            self.sync_state_path = Path(self.sync_state_path)
        if self.export_path and isinstance(self.export_path, str):
            self.export_path = Path(self.export_path)
