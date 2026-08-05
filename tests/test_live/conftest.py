"""Shared fixtures for live API tests.

These tests hit real provider APIs and cost (small amounts of) money.
They are skipped unless the relevant key is present, and live in their
own directory so broad suite runs can exclude them wholesale:

    python -m pytest tests/ --ignore=tests/test_live

Keys are loaded from the repo-root .env when present so the suite works
in the same environment the proxy runs in.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()
