"""headroom-edge-layer — the Compression SaaS / Edge Plan test protocol.

A `headroom.pipeline_extension` that compresses content Headroom's engine
skips by default (file reads, grep, web pages), scored against the agent's
own latest reasoning as a relevance signal. See ``../BASELINE.md`` and the
plan doc for the full test protocol this implements, and each edge module's
docstring for what it does and does not build (outlines for large files and
tool-definition shortening are explicitly deferred, per the plan's own
gating).

Enable with ``HEADROOM_PIPELINE_EXTENSIONS=edge-layer`` — installing this
package alone does not turn it on (opt-in by design, matching
``headroom/pipeline.py``'s proxy-extension seam).
"""

from __future__ import annotations

from .config import EdgeLayerConfig
from .pipeline_extension import EdgeLayerExtension, build_extension

__all__ = ["EdgeLayerConfig", "EdgeLayerExtension", "build_extension"]
