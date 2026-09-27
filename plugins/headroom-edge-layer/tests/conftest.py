from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for path in (_ROOT / "scripts", _ROOT / "replay", _ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
