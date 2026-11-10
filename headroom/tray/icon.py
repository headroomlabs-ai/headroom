"""Tray icon: the Headroom logomark, drawn with Pillow so no binary assets ship in the wheel.

Geometry and colours are the published brand logomark (see ``docs/public/assets/headroom-logomark.svg``):
two square brackets around a compressing pair of bars, in Headroom Mint on Obsidian Space.
"""

from __future__ import annotations

from typing import Any

MINT = (0, 240, 181, 255)  # #00F0B5
OBSIDIAN = (7, 10, 15, 255)  # #070A0F
OFFLINE = (113, 113, 122, 255)
BADGE = (255, 140, 40, 255)
WHITE = (255, 255, 255, 255)

# Logomark rectangles on its native 32x32 grid: (x0, y0, x1, y1).
_MARK_RECTS = (
    (6, 7, 12, 9),  # left bracket, top arm
    (6, 7, 8, 25),  # left bracket, spine
    (6, 23, 12, 25),  # left bracket, bottom arm
    (20, 7, 26, 9),  # right bracket, top arm
    (24, 7, 26, 25),  # right bracket, spine
    (20, 23, 26, 25),  # right bracket, bottom arm
    (11, 14, 21, 16),  # upper bar
    (13, 18, 19, 20),  # lower bar
)


def render_icon(*, online: bool = True, update: bool = False, size: int = 64) -> Any:
    """Logomark on a dark rounded tile; grey when offline, orange dot when an update exists."""
    from PIL import Image, ImageDraw

    ss = 4  # supersample for smooth edges
    s = size * ss
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, s - 1, s - 1), radius=s // 5, fill=OBSIDIAN)

    # Crop the 32-unit grid to 4..28 so the mark fills the tile at 16px.
    lo, span = 4.0, 24.0
    k = s / span
    color = MINT if online else OFFLINE
    for x0, y0, x1, y1 in _MARK_RECTS:
        d.rectangle(
            ((x0 - lo) * k, (y0 - lo) * k, (x1 - lo) * k - 1, (y1 - lo) * k - 1), fill=color
        )

    if update:
        r, pad = s * 0.19, s * 0.01
        d.ellipse((s - 2 * r - pad, pad, s - pad, 2 * r + pad), fill=BADGE, outline=WHITE, width=ss)
    return img.resize((size, size), Image.Resampling.LANCZOS)
