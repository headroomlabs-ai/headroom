"""Structure handlers for different content types.

Each handler knows how to extract structural information from a specific
content type and create a StructureMask marking what should be preserved.

Handlers don't compress - they only identify structure. The actual
compression is done by Kompress on the non-structural parts.
"""

from headroom.compression.handlers.base import (
    HandlerResult,
    StructureHandler,
)
from headroom.compression.handlers.code_handler import CodeStructureHandler
from headroom.compression.handlers.json_handler import JSONStructureHandler

__all__ = [
    "StructureHandler",
    "HandlerResult",
    "JSONStructureHandler",
    "CodeStructureHandler",
]
