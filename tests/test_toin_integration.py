"""Integration tests for the full TOIN feedback loop.

Tests the complete flow:
1. SmartCrusher compresses data and records compression event
2. compression_store stores with correct tool_signature_hash
3. User retrieves cached data (triggering feedback)
4. TOIN learns from retrieval event
5. Future compressions get improved recommendations
"""

import tempfile
from pathlib import Path

import pytest

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.telemetry.toin import (
    TOINConfig,
    get_toin,
    reset_toin,
)


@pytest.fixture
def fresh_toin():
    """Create a fresh TOIN instance with temporary storage."""
    reset_toin()
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = str(Path(tmpdir) / "toin.json")
        toin = get_toin(
            TOINConfig(
                storage_path=storage_path,
                auto_save_interval=0,  # No auto-persist during tests
            )
        )
        yield toin
        reset_toin()


@pytest.fixture
def fresh_store():
    """Create a fresh compression store."""
    reset_compression_store()
    store = get_compression_store(max_entries=100, default_ttl=300)
    yield store
    reset_compression_store()
