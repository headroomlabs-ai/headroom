"""Never decode untrusted images with a Pillow that has known decoder bugs.

Pillow 13.0.0 is the only fix for ten decoder CVEs on untrusted images
(CVE-2026-112395, -112895 and others). uv.lock and the release image take 13, but the published
package metadata cannot require it: fastembed caps ``pillow<13``, so
``pip install headroom-ai[all]`` resolves 12.x, and Python 3.10 has no Pillow 13
at all. On those installs Headroom forwards images unchanged instead of opening
them. Remove this once fastembed allows Pillow 13 and the image extra requires it.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

FIXED_PILLOW_MAJOR = 13


@functools.cache
def pillow_is_safe() -> bool:
    """True when the imported Pillow has the decoder fixes; warns once if not."""
    try:
        import PIL
    except ImportError:
        return False  # Nothing can decode; the image paths already skip.
    installed = PIL.__version__
    if int(installed.split(".")[0]) >= FIXED_PILLOW_MAJOR:
        return True
    logger.warning(
        "Headroom will not decode images: Pillow %s has decoder vulnerabilities "
        "fixed only in Pillow %d, so images are forwarded unchanged and not "
        "compressed. Install pillow>=%d (Python 3.11+) to turn image compression on.",
        installed,
        FIXED_PILLOW_MAJOR,
        FIXED_PILLOW_MAJOR,
    )
    return False
