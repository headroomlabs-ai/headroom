"""Conversation scanners — backwards-compatible re-exports.

Concrete scanner implementations have moved to headroom.learn.plugins.*.
This module re-exports them so existing imports continue to work:

    from headroom.learn.scanner import ClaudeCodeScanner  # still works
    from headroom.learn.scanner import is_error_content    # still works
"""

from __future__ import annotations

# Shared helpers (moved to _shared.py)
from ._shared import classify_error, is_error_content  # noqa: F401

# ConversationScanner ABC (canonical home is base.py)
from .base import ConversationScanner  # noqa: F401

# Concrete scanners (moved to plugins/*, aliased to old names)
from .plugins.claude import ClaudeCodePlugin as ClaudeCodeScanner  # noqa: F401
from .plugins.claude import (  # noqa: F401
    _component_tokenizations,
    _decode_project_path,
    _greedy_path_decode,
)
from .plugins.codex import CodexPlugin as CodexScanner  # noqa: F401
from .plugins.gemini import GeminiPlugin as GeminiScanner  # noqa: F401
